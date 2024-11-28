import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

def plot_training_metrics(model_path):
    """Plot comprehensive training metrics from the saved model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    base_name = os.path.splitext(model_path)[0]
    
    # Create main figure
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('ELU CNN Training Results', fontsize=16)
    
    # Plot metrics
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy', 'train_acc', 'val_acc'),
        ('Precision', 'train_precision', 'val_precision'),
        ('F1-Score', 'train_f1', 'val_f1')
    ]
    
    for idx, (title, train_key, val_key) in enumerate(metrics, 1):
        ax = plt.subplot(3, 2, idx)
        for fold_data in checkpoint['all_fold_histories']:
            history = fold_data['history']
            fold = fold_data['fold']
            ax.plot(history[train_key], label=f'Fold {fold} Train')
            ax.plot(history[val_key], '--', label=f'Fold {fold} Val')
        ax.set_title(f'{title} Over Time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)
    
    # Plot learning rate
    ax = plt.subplot(3, 2, 5)
    best_history = checkpoint['best_fold_history']
    if 'lr' in best_history:
        ax.plot(best_history['lr'])
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True)
    
    # Add configuration summary
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')
    config = checkpoint['config']
    config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
    ax.text(0.1, 0.9, f'Best Model Configuration:\n\n{config_text}', 
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{base_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    best_fold_data = next(data for data in checkpoint['all_fold_histories'] 
                         if data['fold'] == checkpoint['fold'])
    final_cm = best_fold_data['history']['val_confusion_matrix'][-1]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=checkpoint['class_names'],
                yticklabels=checkpoint['class_names'])
    plt.title(f'Confusion Matrix (Best Fold {checkpoint["fold"]})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{base_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Best Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Best F1-Score: {max(best_history['val_f1']):.2f}")
    print(f"Best Precision: {max(best_history['val_precision']):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python elu_plot.py <path_to_model.pth>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found!")
        sys.exit(1)
        
    try:
        plot_training_metrics(model_path)
        print("\nSuccessfully generated all plots!")
    except Exception as e:
        print(f"Error generating plots: {str(e)}") 