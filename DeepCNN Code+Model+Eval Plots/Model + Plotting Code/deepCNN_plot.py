import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

def plot_training_history(model_path):
    """
    Enhanced plotting function to show all metrics:
    1. Loss and Accuracy
    2. Precision and F1-Score
    3. Confusion Matrix
    4. Learning Rate Schedule
    5. Loss and Accuracy specific to the best fold
    """
    # Load the saved model and results
    checkpoint = torch.load(model_path, map_location='cpu')
    base_name = os.path.splitext(model_path)[0]
    
    # Create figure for metrics
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training/Validation Loss
    ax1 = plt.subplot(3, 2, 1)
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax1.plot(history['train_loss'], label=f'Fold {fold} Train')
        ax1.plot(history['val_loss'], '--', label=f'Fold {fold} Val')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Training/Validation Accuracy
    ax2 = plt.subplot(3, 2, 2)
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax2.plot(history['train_acc'], label=f'Fold {fold} Train')
        ax2.plot(history['val_acc'], '--', label=f'Fold {fold} Val')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Precision
    ax3 = plt.subplot(3, 2, 3)
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax3.plot(history['train_precision'], label=f'Fold {fold} Train')
        ax3.plot(history['val_precision'], '--', label=f'Fold {fold} Val')
    ax3.set_title('Precision Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # 4. F1-Score
    ax4 = plt.subplot(3, 2, 4)
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax4.plot(history['train_f1'], label=f'Fold {fold} Train')
        ax4.plot(history['val_f1'], '--', label=f'Fold {fold} Val')
    ax4.set_title('F1-Score Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1-Score')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Learning Rate Schedule
    ax5 = plt.subplot(3, 2, 5)
    best_history = checkpoint['best_fold_history']
    if 'lr' in best_history:
        ax5.plot(best_history['lr'])
        ax5.set_title('Learning Rate Schedule')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.set_yscale('log')
        ax5.grid(True)
    
    # 6. Configuration Summary
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    config = checkpoint['config']
    config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
    ax6.text(0.1, 0.9, f'Best Model Configuration:\n\n{config_text}', 
             verticalalignment='top', fontsize=10)
    
    plt.suptitle('Deep CNN Training Results', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save main metrics plot
    output_path = f'{base_name}_all_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix for best fold
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
    best_history = checkpoint['best_fold_history']
    print(f"Best F1-Score: {max(best_history['val_f1']):.2f}")
    print(f"Best Precision: {max(best_history['val_precision']):.2f}")

def plot_confusion_matrix(y_true, y_pred, class_names, base_name):
    """
    Plot confusion matrix with proper labeling
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{base_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_folds(model_path):
    """Plot all folds with train and validation metrics in separate plots"""
    checkpoint = torch.load(model_path, map_location='cpu')
    base_name = os.path.splitext(model_path)[0]
    
    # Create figure for training metrics
    fig_train = plt.figure(figsize=(20, 15))
    plt.suptitle('Training Metrics Across All Folds', fontsize=16)
    
    # Training metrics
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax1.plot(history['train_loss'], label=f'Fold {fold}')
        ax2.plot(history['train_acc'], label=f'Fold {fold}')
        ax3.plot(history['train_precision'], label=f'Fold {fold}')
        ax4.plot(history['train_f1'], label=f'Fold {fold}')
    
    ax1.set_title('Training Loss')
    ax2.set_title('Training Accuracy')
    ax3.set_title('Training Precision')
    ax4.set_title('Training F1-Score')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{base_name}_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for validation metrics
    fig_val = plt.figure(figsize=(20, 15))
    plt.suptitle('Validation Metrics Across All Folds', fontsize=16)
    
    # Validation metrics
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    for fold_data in checkpoint['all_fold_histories']:
        history = fold_data['history']
        fold = fold_data['fold']
        ax1.plot(history['val_loss'], label=f'Fold {fold}')
        ax2.plot(history['val_acc'], label=f'Fold {fold}')
        ax3.plot(history['val_precision'], label=f'Fold {fold}')
        ax4.plot(history['val_f1'], label=f'Fold {fold}')
    
    ax1.set_title('Validation Loss')
    ax2.set_title('Validation Accuracy')
    ax3.set_title('Validation Precision')
    ax4.set_title('Validation F1-Score')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{base_name}_validation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_fold(model_path):
    """Plot metrics for only the best performing fold"""
    checkpoint = torch.load(model_path, map_location='cpu')
    base_name = os.path.splitext(model_path)[0]
    best_history = checkpoint['best_fold_history']
    best_fold = checkpoint['fold']
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(f'Best Fold (Fold {best_fold}) Metrics', fontsize=16)
    
    # Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(best_history['train_loss'], label='Train')
    ax1.plot(best_history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(best_history['train_acc'], label='Train')
    ax2.plot(best_history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Precision
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(best_history['train_precision'], label='Train')
    ax3.plot(best_history['val_precision'], label='Validation')
    ax3.set_title('Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # F1-Score
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(best_history['train_f1'], label='Train')
    ax4.plot(best_history['val_f1'], label='Validation')
    ax4.set_title('F1-Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1-Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{base_name}_best_fold_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python deepCNN_plot.py <path_to_model.pth>")
            sys.exit(1)
            
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"Error: File '{model_path}' not found!")
            sys.exit(1)
            
        # Plot all metrics
        plot_all_folds(model_path)
        plot_best_fold(model_path)
        
        # Load checkpoint for summary statistics and confusion matrix
        checkpoint = torch.load(model_path, map_location='cpu')
        best_history = checkpoint['best_fold_history']
        base_name = os.path.splitext(model_path)[0]
        
        # Plot confusion matrix for best fold
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
        
        print("\nSuccessfully generated all plots!")
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
