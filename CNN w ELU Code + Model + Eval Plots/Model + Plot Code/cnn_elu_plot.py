import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_training_history(history_path):
    """Plot training metrics from .npy file"""
    # Check if file exists
    if not os.path.exists(history_path):
        print(f"Error: File '{history_path}' not found!")
        return
    
    try:
        # Load the history
        history = np.load(history_path, allow_pickle=True).item()
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ELU CNN Training Results', fontsize=16)
        
        # Plot Loss
        axes[0,0].plot(history['train_loss'], label='Train')
        axes[0,0].plot(history['val_loss'], label='Validation')
        axes[0,0].set_title('Loss Over Time')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot Accuracy
        axes[0,1].plot(history['train_acc'], label='Train')
        axes[0,1].plot(history['val_acc'], label='Validation')
        axes[0,1].set_title('Accuracy Over Time')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot F1 Score
        axes[1,0].plot(history['val_f1'], label='Validation')
        axes[1,0].set_title('F1 Score Over Time')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot Confusion Matrix
        if 'val_confusion_matrix' in history:
            final_cm = history['val_confusion_matrix'][-1]
            sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
            axes[1,1].set_title('Final Confusion Matrix')
            axes[1,1].set_ylabel('True Label')
            axes[1,1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nTraining Summary:")
        print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
        print(f"Best F1-Score: {max(history['val_f1']):.2f}")
        print(f"Final Loss: {history['val_loss'][-1]:.4f}")
    except Exception as e:
        print(f"Error loading history file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_training_history(sys.argv[1])
    else:
        print("Usage: python cnn_elu_plot.py <history_file>") 