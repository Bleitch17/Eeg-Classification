import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List

from dataset_bci_iv_2a.dataset import BciIvDatasetFactory
from torch.utils.data import DataLoader


class LSTM(nn.Module):
    """
    LSTM design for classifying EEG signals.
    Structured to be comparable with the CNN implementation.
    """

    def __init__(self) -> None:
        super(LSTM, self).__init__()
        
        # The input to the Neural Network will be a tensor of size: 
        # (batch_size, 1, num_eeg_channels, window_size)
        
        self.lstm = nn.LSTM(
            input_size=100,  # window_size
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # Match CNN's fully connected layer structure
        self.fc0 = nn.LazyLinear(out_features=200)
        self.fc1 = nn.LazyLinear(out_features=100)
        self.fc2 = nn.LazyLinear(out_features=5)
        
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input size: (batch_size, 1, 22, 100)
        batch_size = input.size(0)
        
        # Reshape for LSTM: (batch_size, 22, 100)
        x = input.squeeze(1)
        
        # Transpose to (batch_size, 22, 100)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, 22, 128)
        
        # Use last time step output
        x = lstm_out[:, -1, :]  # (batch_size, 128)
        
        # Match CNN's fully connected structure
        x = self.dropout(x)
        x = self.sigmoid(self.fc0(x))
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15) -> Dict[str, List[float]]:
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    early_stopping = EarlyStopping(patience=7)
    best_val_acc = 0.0
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Save metrics
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_lstm_temp.pth')
        
        # Early stopping check
        early_stopping(val_loss/len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return history, best_val_acc

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
            model = LSTM().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            
            print(f"Training fold {fold+1}...")
            history, fold_val_acc = train_model(model, trainloader, valloader, criterion, optimizer, device)
            fold_results.append(fold_val_acc)
            
            # Save results with LSTM-specific names
            print(f"Saving results for fold {fold+1}...")
            np.save(f'lstm_fold_{fold+1}_history.npy', history)
            
            if fold_val_acc > best_fold_acc:
                best_fold_acc = fold_val_acc
                best_fold = fold + 1
                if os.path.exists('best_lstm_temp.pth'):
                    os.replace('best_lstm_temp.pth', f'best_lstm_fold_{best_fold}.pth')
            else:
                if os.path.exists('best_lstm_temp.pth'):
                    os.remove('best_lstm_temp.pth')
            
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