import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset_bci_iv_2a.dataset import (
    BciIvDataset
)
from device import get_system_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, f1_score, confusion_matrix


class Net(nn.Module):
    """
    CNN design for classifying EEG signals inspired by the following paper:
    https://www.sciencedirect.com/science/article/pii/S0030402616312980
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(40)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(22, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=40, kernel_size=(1, 30), stride=1, padding=0)
        
        # Fully connected layers with increased capacity
        self.fc0 = nn.LazyLinear(out_features=400)
        self.fc1 = nn.LazyLinear(out_features=200)
        self.fc2 = nn.LazyLinear(out_features=5)
        
        # Improved regularization
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Change to ELU activation
        self.activation = nn.ELU(alpha=1.0)  # alpha=1.0 is the default value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Spatial features
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation(x)  # Using ELU
        
        # Temporal features
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)  # Using ELU
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(x)
        
        # Dense layers
        x = self.fc0(x)
        x = self.activation(x)  # Using ELU
        x = self.dropout2(x)
        
        x = self.fc1(x)
        x = self.activation(x)  # Using ELU
        
        output = self.fc2(x)
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15):
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_f1': [],
        'lr': [], 'val_confusion_matrix': []
    }
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    best_val_acc = 0.0
    
    # Initialize scaler based on device type
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
    else:
        use_amp = False
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Handle both CPU and GPU cases
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs.unsqueeze(1))
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # Calculate additional metrics
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_precision = precision_score(val_true_labels, val_predictions, average='macro')
        val_f1 = f1_score(val_true_labels, val_predictions, average='macro')
        cm = confusion_matrix(val_true_labels, val_predictions)
        
        # Record all metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)
        history['val_confusion_matrix'].append(cm)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
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
            }, f'cnn_elu_best.pth')
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
    
    return history, best_val_acc

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


if __name__ == "__main__":
    device = get_system_device()

    # Get data with k-fold splits
    print("Loading dataset...")

    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90"
    df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90.parquet", engine="pyarrow")
    
    print(df.head(3))

    labels: pd.Series = df["Label"]
    features: pd.DataFrame = df.drop(columns=["Label"])

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # K-fold cross validation
    fold_results = []
    best_fold_acc = 0.0
    best_fold = 0
    
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f'\nStarting Fold {fold+1}/5')
        try:
            # Clear GPU memory before each fold
            if "cuda" in device.type:
                torch.cuda.empty_cache()
            
            elif "xpu" in device.type:
                torch.xpu.empty_cache()

            print(f"Creating datasets for fold {fold+1}...")
            train_dataset = BciIvDataset(
                features.iloc[train_idx], 
                labels.iloc[train_idx]
            )
            val_dataset = BciIvDataset(
                features.iloc[val_idx], 
                labels.iloc[val_idx]
            )
            
            trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            print(f"Initializing model for fold {fold+1}...")
            model = Net().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=3,
                factor=0.1,
                min_lr=1e-6
            )
            
            print(f"Training fold {fold+1}...")
            history, fold_val_acc = train_model(model, trainloader, valloader, criterion, optimizer, scheduler, device)
            fold_results.append(fold_val_acc)
            
            print(f"Fold {fold+1} completed with validation accuracy: {fold_val_acc:.2f}%")
            
            # Save results
            print(f"Saving results for fold {fold+1}...")
            np.save(f'ELU_fold_{fold+1}_history.npy', history)
            
            # Keep track of best performing fold
            if fold_val_acc > best_fold_acc:
                best_fold_acc = fold_val_acc
                best_fold = fold + 1
                # Save as cnn_elu_best.pth
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_fold_acc,
                    'fold': best_fold,
                    'all_fold_histories': fold_histories,
                    'best_fold_history': history,
                    'class_names': ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'],
                    'config': {
                        'batch_size': 16,
                        'learning_rate': 0.001,
                        'weight_decay': 0.00005,
                        'dropout1': 0.2,
                        'dropout2': 0.3
                    }
                }, 'cnn_elu_best.pth')
            else:
                # Remove temporary model if it's not the best
                if os.path.exists('cnn_elu_best.pth'):
                    os.remove('cnn_elu_best.pth')
            
            # Clear model from GPU
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            fold_histories.append({
                'fold': fold + 1,
                'history': history,
                'val_acc': fold_val_acc
            })
            
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
    configs = [
        {
            'batch_size': 16,
            'lr': 0.001,
            'weight_decay': 0.00005,
            'dropout1': 0.2,
            'dropout2': 0.3
        },
        {
            'batch_size': 32,
            'lr': 0.0005,
            'weight_decay': 0.00001,
            'dropout1': 0.3,
            'dropout2': 0.4
        }
    ]
    
    results = []
    
    for config_idx, config in enumerate(configs):
        print(f"\nTrying configuration {config_idx + 1}")
        print(config)
        
        trainloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        model = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.1,
            min_lr=1e-6
        )
        
        history, val_acc = train_model(
            model, trainloader, valloader,
            criterion, optimizer, scheduler,
            device
        )
        
        results.append({
            'config': config,
            'val_acc': val_acc,
            'history': history
        })
    
    plot_training_results(results)