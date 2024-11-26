import torch
import torch.nn as nn
import torch.optim as optim

from dataset_bci_iv_2a.dataset import BciIvDatasetFactory
from device import get_system_device
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    # TODO - fix this to optionally work with cross validation and any changes to the dataset.
    device = get_system_device()
    
    trainset, testset = BciIvDatasetFactory.create(1, 100, 90)
    batch_size: int = 16

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    print(f"Trainset size: {len(trainset)}")
    print(f"Testset size: {len(testset)}")

    classes = ('Rest', 'Left', 'Right', 'Feet', 'Tongue')

    lstm = LSTM()
    lstm.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(15):
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # NOTE - need to swap the dimensions of the input tensor from (num_eeg_channels, num_time_points) to (num_time_points, num_eeg_channels)
            outputs = lstm(inputs.permute(0, 2, 1))

            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}, accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0
    
    print("Finished training")

    # Switch lstm to evaluation mode:
    lstm.eval()

    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = lstm(inputs.permute(0, 2, 1))

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    save = input("Do you want to save the model? (y/n): ")
    if save.lower() != "y":
        print("Model not saved")
        exit()

    # Save the model for later use
    torch.save(lstm.state_dict(), "./eeg_lstm.pth")
