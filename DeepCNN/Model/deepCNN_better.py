import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import ast

from dataset_bci_iv_2a.dataset import (
    BciIvDataset
)
from device import get_system_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from sklearn.model_selection import KFold

DEFAULT_CONFIG = {
    'batch_size': 32,      # Larger batch = faster training while reduce generalization
    'lr': 0.0005,         # Learning rate: higher = faster learning while unstable,  increased from 0.0001 from Trial 1
    'weight_decay': 0.005, # L2 regularization: tuned higher for more regularization, reduces overfitting
    'scheduler': 'onecycle', # Learning rate scheduling strategy
    'max_epochs': 50,      # Maximum training iterations, we use 50 epochs to balance between training time and generalization
    'label_smoothing': 0.05,# Reduces overconfidence: higher for more regularization, increased from 0.01 from Trial 1
    'accumulation_steps': 1, # Simulates larger batch size with limited memory
    'num_workers': 4       # Parallel threads
}

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: Adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    
    Parameters:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (higher = more compression)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Global pooling
        self.excitation = nn.Sequential(
            # Dimensionality reduction
            nn.Linear(channels, channels // reduction),
            nn.ELU(),
            # Dimensionality restoration
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()  # Scale each channel between 0-1
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y

class ResidualBlock(nn.Module):
    """
    Residual Block with SE attention: Allows network to learn residual functions,
    making it easier to train deeper networks.
    
    Parameters:
        channels: Number of input/output channels
    Effects:
        - Larger kernel_size = larger receptive field but more parameters
        - More channels = more capacity but more memory usage
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(channels)
        self.elu = nn.ELU()
        self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += residual
        x = self.elu(x)
        return x

class Net(nn.Module):
    """
    Modern CNN architecture for EEG classification combining:
    1. Spatial attention for channel importance
    2. Residual connections for deep network training
    3. SE blocks for channel-wise feature recalibration
    
    Architecture Effects:
    - More conv filters
    - More residual blocks = deeper network, better features
    - Higher dropout = more regularization and we tuned to reduce the risk of underfitting
    
    Deep CNN design for classifying EEG signals inspired by the following paper:
    https://www.sciencedirect.com/science/article/pii/S0030402616312980
    https://arxiv.org/pdf/1611.08024
    https://arxiv.org/pdf/1512.03385
    https://arxiv.org/pdf/1709.01507
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        
        # Spatial attention: Learn importance of each EEG channel
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=1),  # Channel-wise projection
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 22, kernel_size=1),  # Project back to channel space
            nn.Softmax(dim=1)  # Normalize attention weights
        )
        
        # Feature extraction pipeline
        self.feature_extractor = nn.Sequential(
            # Initial feature extraction
            nn.Conv1d(22, 64, kernel_size=31, padding=15),  # Large kernel for temporal patterns
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.2),  # Initial regularization, imrpoved from 0.5 tp 0.3 then 0.2
            
            # Deep feature processing
            ResidualBlock(64),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Temporal downsampling
            nn.Dropout(0.2),
            
            ResidualBlock(64),
            nn.Conv1d(64, 128, kernel_size=1),  # Increase channels for more features
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)  # Final regularization
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # Input is 128 with global pooling
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5)
        )
        
        # Add weight initialization, but not significant
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply spatial attention
        attention = self.spatial_attention(x)
        x = x * attention  # Element-wise multiplication
        
        # Extract features
        x = self.feature_extractor(x)  # Shape: (batch, 128, 25)
        
        # Global pooling
        x = self.global_pool(x)  # Shape: (batch, 128, 1)
        x = x.squeeze(-1)  # Shape: (batch, 128)
        
        # Classification
        x = self.classifier(x)
        return x


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, val_acc, epoch, model):
        improved = False
        
        # We increased patience for early stopping to get better results
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            improved = True
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            improved = True
            
        if improved:
            self.counter = 0
            self.best_epoch = epoch
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Best epoch was {self.best_epoch} with loss {self.best_loss:.4f} and accuracy {self.best_val_acc:.2f}%")

#A big bunch of code here is just to help collect metrics for plotting after training
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 
        'lr': [],
        'train_precision': [], 'val_precision': [],
        'train_f1': [], 'val_f1': [],
        'train_confusion_matrix': [], 'val_confusion_matrix': []
    }
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    accumulation_steps = DEFAULT_CONFIG['accumulation_steps']
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}')):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Collect predictions
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate training loss/acc
        avg_train_loss = train_loss/len(train_loader)
        train_accuracy = 100.*train_correct/train_total
        
      
        from sklearn.metrics import precision_score, f1_score, confusion_matrix
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        train_cm = confusion_matrix(all_train_labels, all_train_preds)
        
        # Store metrics here
        history['train_precision'].append(train_precision)
        history['train_f1'].append(train_f1)
        history['train_confusion_matrix'].append(train_cm)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
            # Calculate all other metrics needed for our evaluation like f1 and precision
            from sklearn.metrics import precision_score, f1_score, confusion_matrix
            val_precision = precision_score(all_labels, all_preds, average='macro')
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_cm = confusion_matrix(all_labels, all_preds)
            
            # Store 
            history['val_precision'].append(val_precision)
            history['val_f1'].append(val_f1)
            history['val_confusion_matrix'].append(val_cm)
        
        # Calculate validation metrics
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100.*val_correct/val_total
        
        # Print metrics during training for each epoch to help with debugging and tuning (early discovery of overfitting)
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
        
        # Single early stopping check that handles both metrics
        early_stopping(avg_val_loss, val_accuracy, epoch, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model_state)
            break
        
        # Update scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # SAve metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(scheduler.get_last_lr()[0])
    
    # Kepp using best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    return history, early_stopping.best_val_acc

def plot_training_results(histories):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, result in enumerate(results):
        history = result['history']
        config = result['config']
        
        axes[0,0].plot(history['train_loss'], label=f'Config {idx+1}')
        axes[0,1].plot(history['val_loss'], label=f'Config {idx+1}')
        axes[1,0].plot(history['train_acc'], label=f'Config {idx+1}')
        axes[1,1].plot(history['val_acc'], label=f'Config {idx+1}')
    
    axes[0,0].set_title('Training Loss')
    axes[0,1].set_title('Validation Loss')
    axes[1,0].set_title('Training Accuracy')
    axes[1,1].set_title('Validation Accuracy')
    
    for ax in axes.flat:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('ELU_training_results.png')
    plt.close()

def load_and_verify_data(file_path):
    try:
        df = pd.read_parquet(file_path, engine="pyarrow")
        print("Data shape:", df.shape)
        print("\nSample of first row:")
        
        # Print first 3 columns without trying to slice the data
        for col in list(df.columns)[:3]:
            print(f"{col}: {df[col].iloc[0]}")
            
        # Verify data structure
        print("\nVerifying data structure...")
        if "Label" not in df.columns:
            raise ValueError("Dataset missing 'Label' column")
            
        # Print label distribution
        print("\nLabel distribution:")
        print(df["Label"].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Set multiprocessing start method
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    device = get_system_device()
    print("Loading dataset...")
    
    try:
        df = load_and_verify_data("dataset_bci_iv_2a/A01_100_90.parquet")
        
        # Verify data types and convert if necessary
        print("\nVerifying data types...")
        for col in df.columns:
            if col != "Label":
                # Check if the column contains string representations of lists
                if isinstance(df[col].iloc[0], str):
                    print(f"Converting column {col} from string to array...")
                    df[col] = df[col].apply(ast.literal_eval)
        
        # Split features and labels
        labels = df["Label"]
        features = df.drop(columns=["Label"])
        
        # Verify feature dimensions
        first_feature = features.iloc[0, 0]
        if isinstance(first_feature, (list, np.ndarray)):
            print(f"\nFeature dimensions: {len(features.columns)} channels x {len(first_feature)} timepoints")
        
        # Initialize k-fold
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = []
        all_fold_histories = []  # store all fold histories
        best_fold_acc = 0
        best_fold = 0
        
        # K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
            print(f'\nStarting Fold {fold+1}/{k_folds}')
            
            try:
                # Create train and validation datasets
                train_dataset = BciIvDataset(
                    features.iloc[train_idx], 
                    labels.iloc[train_idx]
                )
                val_dataset = BciIvDataset(
                    features.iloc[val_idx], 
                    labels.iloc[val_idx]
                )
                
                # Create data loaders
                trainloader = DataLoader(
                    train_dataset, 
                    batch_size=DEFAULT_CONFIG['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                valloader = DataLoader(
                    val_dataset, 
                    batch_size=DEFAULT_CONFIG['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Initialize model and training components
                model = Net().to(device)
                criterion = nn.CrossEntropyLoss(
                    weight=None,
                    label_smoothing=DEFAULT_CONFIG['label_smoothing']
                )
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=DEFAULT_CONFIG['lr'],
                    weight_decay=DEFAULT_CONFIG['weight_decay'],
                    betas=(0.9, 0.999)
                )
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=DEFAULT_CONFIG['lr'],
                    epochs=DEFAULT_CONFIG['max_epochs'],
                    steps_per_epoch=len(trainloader),
                    pct_start=0.2,
                    div_factor=10.0,
                    final_div_factor=50.0
                )
                
                # Train the model
                history, val_acc = train_model(
                    model=model,
                    train_loader=trainloader,
                    val_loader=valloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=DEFAULT_CONFIG['max_epochs']
                )
                
                # Store history for this fold
                all_fold_histories.append({
                    'fold': fold + 1,
                    'history': history,
                    'val_acc': val_acc
                })
                
                fold_results.append(val_acc)
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_fold = fold + 1
                    # Save best model WITH ALL histories
                    torch.save({
                        'fold': fold + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': DEFAULT_CONFIG,
                        'best_fold_history': history,  # History of best fold
                        'all_fold_histories': all_fold_histories,  # Histories of ALL folds
                        'class_names': ['Rest', 'Left Hand', 'Right Hand', 'Feet', 'Tongue']  # Add this
                    }, 'deepCNN_better.pth')
            
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
        
        # Hyperparameter configurations to try
        configs = [DEFAULT_CONFIG]  # Use the same config for all runs
        
        results = []
        
        for config_idx, config in enumerate(configs):
            print(f"\nTrying configuration {config_idx + 1}")
            print(config)
            
            trainloader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            model = Net().to(device)
            criterion = nn.CrossEntropyLoss(
                weight=None,
                label_smoothing=config['label_smoothing']
            )
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                betas=(0.9, 0.999)
            )
            if config['scheduler'] == 'onecycle':
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=config['lr'],
                    epochs=config['max_epochs'],
                    steps_per_epoch=len(trainloader),
                    pct_start=0.2,
                    div_factor=10.0,
                    final_div_factor=50.0
                )
            elif config['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=5,
                    T_mult=2,
                    eta_min=1e-6
                )
            
            history, val_acc = train_model(
                model=model,
                train_loader=trainloader,
                val_loader=valloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=config['max_epochs']
            )
            
            results.append({
                'config': config,
                'val_acc': val_acc,
                'history': history
            })
        
        plot_training_results(results)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
