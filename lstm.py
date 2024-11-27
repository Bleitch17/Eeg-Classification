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
from sklearn.model_selection import KFold


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


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_f1': [],
        'val_conf_matrix': [],
        'lr': [],
    }
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    best_val_acc = 0.0

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

        # Validation phase - switch model to evaluation mode
        model.eval()

        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs.permute(0, 2, 1))

                # Loss calculation
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get the predictions
                _, predicted = torch.max(outputs, 1)

                # Collect all the predictions made 
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Average loss and accuracy:
        val_total_predictions = len(all_labels)
        val_correct_predictions = sum([1 for i in range(val_total_predictions) if all_labels[i] == all_predictions[i]])

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct_predictions / val_total_predictions

        # Addional metrics: precision, f1-score, confusion matrix
        precision = precision_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Record metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_precision'].append(precision)
        history['val_f1'].append(f1)
        history['val_conf_matrix'].append(conf_matrix)

        print(f'\nEpoch {epoch + 1} / {num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.3f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.3f}, Val Acc: {val_accuracy:.2f}%')

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, 'LSTM_best_temp.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
    
    return history, best_val_acc


if __name__ == "__main__":
    device = get_system_device()
    
    # Get data with k-fold splits
    print("Loading dataset...")

    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90"
    df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90.parquet", engine="pyarrow")

    print(f"DF Shape: {df.shape}")
    print(df.head(3))

    labels: pd.Series = df["Label"]
    features: pd.DataFrame = df.drop(columns=["Label"])

    # Gets indicies to split the data into k-folds
    # See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store the results from k-fold cross validation:
    fold_accuracies = []
    best_fold_acc = 0.0
    best_fold = 0

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f"Fold {fold_index + 1}")

        # Clear GPU memory before each fold
        if "cuda" in device.type:
            torch.cuda.empty_cache()
        
        elif "xpu" in device.type:
            torch.xpu.empty_cache()

        print(f"Creating datasets for fold {fold_index + 1}...")

        train_dataset = BciIvDataset(features.iloc[train_idx], labels.iloc[train_idx])
        val_dataset = BciIvDataset(features.iloc[val_idx], labels.iloc[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        print(f"Initializing model for fold {fold_index + 1}...")

        model = LSTM().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
        
        # See: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.1,
            min_lr=1e-6
        )

        print(f"Training fold {fold_index + 1}...")

        history, fold_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device
        )
        fold_accuracies.append(fold_val_acc)

        print(f"Fold {fold_index + 1} validation accuracy: {fold_val_acc:.2f}")

        print(f"Saving results for fold {fold_index + 1}...")
        np.save(f"LSTM_fold_{fold_index + 1}_history.npy", history)

        if fold_val_acc > best_fold_acc:
            best_fold_acc = fold_val_acc
            best_fold = fold_index + 1

            if os.path.exists("LSTM_best_temp.pth"):
                os.replace("LSTM_best_temp.pth", f"LSTM_best_fold_{best_fold}.pth")

        else:
            # Remove temporary model if its not the best
            if os.path.exists("LSTM_best_temp.pth"):
                os.remove("LSTM_best_temp.pth")

        # Clear model from GPU:
        del model

    print(f"5-Fold Cross Validation Results:")
    print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.2f}")
    print(f"S.D. of Validation Accuracies: {np.std(fold_accuracies):.2f}")
    print(f"Individual Fold Accuracies: {[f'Fold: {fold + 1}: {acc:.2f}' for fold, acc in enumerate(fold_accuracies)]}")
    print(f"Best model saved from fold {best_fold} with accuracy: {best_fold_acc:.2f}")
