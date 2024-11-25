import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from dataset_bci_iv_2a.dataset import (
    BciIvDatasetFactory,
    BciIvDataset
)
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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15) -> Dict[str, List[float]]:
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    early_stopping = EarlyStopping(patience=5)
    best_val_acc = 0.0
    
    # Use tqdm for epoch-level progress
    for epoch in tqdm(range(num_epochs), desc='Training epochs', leave=True):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for batch-level progress with different formatting
        for inputs, labels in train_loader:  # Removed nested tqdm
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss/len(train_loader)
        train_accuracy = 100.*train_correct/train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100.*val_correct/val_total
        
        # Record metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Print progress clearly
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.3f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.3f}, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, f'best_model_temp.pth')
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    return history, best_val_acc

if __name__ == "__main__":
    print("Starting program...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data with k-fold splits
    print("Loading dataset...")
    features, labels, kfold = BciIvDatasetFactory.create_k_fold(1, 100, 95, k_folds=5)
    
    # K-fold cross validation
    fold_results = []
    best_fold_acc = 0.0
    best_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f'\nStarting Fold {fold+1}/5')
        try:
            # Clear GPU memory before each fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Creating datasets for fold {fold+1}...")
            train_dataset = BciIvDataset(
                features.iloc[train_idx], 
                labels.iloc[train_idx], 
                window_size=100
            )
            val_dataset = BciIvDataset(
                features.iloc[val_idx], 
                labels.iloc[val_idx], 
                window_size=100
            )
            
            trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            print(f"Initializing model for fold {fold+1}...")
            model = Net().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            
            print(f"Training fold {fold+1}...")
            history, fold_val_acc = train_model(model, trainloader, valloader, criterion, optimizer, device)
            fold_results.append(fold_val_acc)
            
            print(f"Fold {fold+1} completed with validation accuracy: {fold_val_acc:.2f}%")
            
            # Save results
            print(f"Saving results for fold {fold+1}...")
            np.save(f'fold_{fold+1}_history.npy', history)
            
            # Keep track of best performing fold
            if fold_val_acc > best_fold_acc:
                best_fold_acc = fold_val_acc
                best_fold = fold + 1
                # Rename the best model from this fold
                import os
                if os.path.exists('best_cnn_temp.pth'):
                    os.replace('best_cnn_temp.pth', f'best_cnn_fold_{best_fold}.pth')
            else:
                # Remove temporary model if it's not the best
                if os.path.exists('best_cnn_temp.pth'):
                    os.remove('best_cnn_temp.pth')
            
            # Clear model from GPU
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in fold {fold+1}: {str(e)}")
            continue
    
    if fold_results:
        print(f'\nK-fold cross validation results:')
        print(f'Average accuracy: {np.mean(fold_results):.2f}%')
        print(f'Std deviation: {np.std(fold_results):.2f}%')
        print("\nIndividual fold accuracies:")
        for i, acc in enumerate(fold_results, 1):
            print(f"Fold {i}: {acc:.2f}%")
        print(f"\nBest model saved from fold {best_fold} with accuracy: {best_fold_acc:.2f}%")
    else:
        print("No results were collected. All folds failed.")
