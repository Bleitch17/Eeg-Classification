import torch
import torch.nn as nn
import torch.optim as optim

from dataset_bci_iv_2a.dataset import BciIvDatasetFactory
from torch.utils.data import DataLoader


class Net(nn.Module):
    """
    CNN design for classifying EEG signals inspired by the following paper:
    https://www.sciencedirect.com/science/article/pii/S0030402616312980
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        
        # The input to the Neural Network will be a tensor of size: 
        # (1, num_eeg_channels, window_size)

        # Spatial Convolutional Layer: extract spatial features
        # Will produce feature maps of size: (1 x window_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(22, 1), stride=1, padding=0)

        # Temporal Convolutional Layer: extract temporal features from the spatial features
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=40, kernel_size=(1, 30), stride=1, padding=0)

        self.fc0 = nn.LazyLinear(out_features=200)

        # Fully connected layer with 100 neurons
        self.fc1 = nn.LazyLinear(out_features=100)

        # Fully connected output layer with 5 neurons (one for each class)
        self.fc2 = nn.LazyLinear(out_features=5)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input size: (batch_size, 1, 22, 20)
        c1 = self.relu(self.conv1(input))
        # Output size: (batch_size, 8, 1, 20)
        
        # Input size: (batch_size, 8, 1, 20)
        c2 = self.relu(self.conv2(c1))
        # Output size: (batch_size, 40, 1, 4)

        # Flatten the output of the second convolutional layer
        # Input size: (batch_size, 40, 1, 4)
        c2_flat = torch.flatten(c2, start_dim=1)
        # Output size: (batch_size, 160)

        x = self.dropout(c2_flat)

        x = self.sigmoid(self.fc0(x))

        # Input size: (batch_size, 160)
        f3 = self.sigmoid(self.fc1(x))
        # Output size: (batch_size, 100)

        # Input size: (batch_size, 100)
        output = self.fc2(f3)
        # Output size: (batch_size, 5)

        return output


if __name__ == "__main__":
    device = torch.device("cpu")

    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda:0")

    elif torch.xpu.is_available():
        print("XPU is available")
        device = torch.device("xpu:0")

    # TODO - would like to customize the number of EEG channels
    trainset, testset = BciIvDatasetFactory.create(1, 100, 95)
    batch_size: int = 16

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    print(f"Trainset size: {len(trainset)}")
    print(f"Testset size: {len(testset)}")

    classes = ('Rest', 'Left', 'Right', 'Feet', 'Tongue')

    net = Net()

    # Move the network to the GPU if available
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # See: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(15):
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for i, data in enumerate(trainloader, 0):
            # Move data to the GPU if available
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Need to add a channel dimension to the input tensor, since the kernels are defined
            # as 2 dimensional, even though they are vectors.
            outputs = net(inputs.unsqueeze(1))

            # Record the number of correct predictions for training accuracy calculation
            # torch.max returns the maximum values along the specified axis, 
            # and the indices of the maximum values. Only need the indices, so the second
            # return value is stored in the predicted variable
            _, predicted = torch.max(outputs, 1)

            # The total number of predictions made in this iteration is the batch size
            total += labels.size(0)

            # A label is an index in the classes list - so the number of correct predictions are
            # the number of times the predicted index matches the label
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
    
    print("Finished Training")

    # Switch the model to evaluation mode:
    net.eval()

    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs.unsqueeze(1))

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    save = input("Do you want to save the model? (y/n): ")
    if save.lower() != "y":
        print("Model not saved")
        exit()

    # Save the model for later use
    torch.save(net.state_dict(), "./eeg_net.pth")
