import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset_bci_iv_2a.dataset import BciIvDataset
from device import get_system_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


LOG: logging.Logger = logging.getLogger(__name__)


def configure_logging(log_level: int) -> None:
    log_format: str = "%(module)s - %(levelname)s - %(message)s"
    formatter: logging.Formatter = logging.Formatter(log_format)

    # handler: logging.Handler = logging.StreamHandler(sys.stdout)
    handler: logging.Handler = logging.FileHandler("lstm.log", mode="w")
    handler.setFormatter(formatter)
    
    LOG.setLevel(log_level)
    LOG.addHandler(handler)


class LSTM(nn.Module):
    """
    LSTM Design for classifying EEG signals.

    Following general LSTM architecture guidelines from the following literature review:
    https://iopscience.iop.org/article/10.1088/1741-2552/ab0ab5

    NOTE - the literature review seems to indicate that RNN's don't perform as well as CNN's for EEG Motor Imagery classification.
    """

    def __init__(self) -> None:
        super(LSTM, self).__init__()

        # The input to the Neural Network will be a tensor of size:
        # (window_size, num_eeg_channels)

        self.lstm = nn.LSTM(
            input_size=22,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(p=0.5)

        self.fc0 = nn.LazyLinear(out_features=1000)
        self.fc1 = nn.LazyLinear(out_features=200)
        self.fc2 = nn.LazyLinear(out_features=5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input size: (batch_size, window_size, num_eeg_channels)
        lstm_out, _ = self.lstm(input)

        # Input size: (batch_size, window_size, 64)
        x = self.relu(lstm_out.flatten(start_dim=1))
        # Output size: (batch_size, window_size * 64)

        x = self.dropout(x)

        x = self.sigmoid(self.fc0(x))
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        # Output size: (batch_size, 5)

        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=20):
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'precision': [],
        'f1': [],
        'conf_matrix': []
    }

    for epoch in tqdm(range(num_epochs)):
        # model should be in training mode
        model.train()

        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # NOTE - need to swap the dimensions of the input tensor from (num_eeg_channels, num_time_points) to (num_time_points, num_eeg_channels)
            outputs = model(inputs.permute(0, 2, 1))

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total_predictions += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct_predictions / train_total_predictions

        # Record training metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)

        LOG.debug(f'\nEpoch {epoch + 1} / {num_epochs}:')
        LOG.debug(f'Train Loss: {avg_train_loss:.3f}, Train Acc: {train_accuracy:.2f}%')

    # Testing phase - switch model to evaluation mode
    model.eval()

    test_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs.permute(0, 2, 1))

            # Loss calculation
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get the predictions
            _, predicted = torch.max(outputs, 1)

            # Collect all the predictions made 
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Average loss and accuracy:
    test_total_predictions = len(all_labels)
    test_correct_predictions = sum([1 for i in range(test_total_predictions) if all_labels[i] == all_predictions[i]])

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct_predictions / test_total_predictions

    # Addional metrics: precision, f1-score, confusion matrix
    precision = precision_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    history['test_loss'].append(avg_test_loss)
    history['test_acc'].append(test_accuracy)
    history['precision'].append(precision)
    history['f1'].append(f1)
    history['conf_matrix'].append(conf_matrix)

    # Save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'LSTM_best.pth')
    
    return history


if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    
    device = get_system_device()
    
    # Get data with k-fold splits
    LOG.debug("Loading dataset...")

    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90"
    df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90.parquet", engine="pyarrow")

    LOG.debug(f"DF Shape: {df.shape}")
    LOG.debug(df.head(3))

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_labels = train["Label"]
    train_features = train.drop(columns=["Label"])

    test_labels = test["Label"]
    test_features = test.drop(columns=["Label"])

    LOG.debug(f"Creating datasets...")

    train_dataset = BciIvDataset(train_features, train_labels)
    test_dataset = BciIvDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    LOG.debug(f"Initializing model...")

    model = LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)

    LOG.debug(f"Training...")

    # Clear GPU memory
    if "cuda" in device.type:
        torch.cuda.empty_cache()
    
    elif "xpu" in device.type:
        torch.xpu.empty_cache()

    try:
        history = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device
        )

    except Exception as e:
        LOG.error(f"Caught exception: {e}")
        exit(1)

    LOG.debug(f"Test Loss: {history['test_loss'][-1]:.3f}, Test Acc: {history['test_acc'][-1]:.2f}%")

    np.save(f"LSTM_history.npy", history)
