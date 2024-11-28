import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

def plot_training_metrics(history_path):
    """Plot training metrics from .npy file"""
    try:
        # Load history using numpy for .npy files
        history = np.load(history_path, allow_pickle=True).item()
        base_name = os.path.splitext(history_path)[0]
        
        # Create figure with subplots (3x2 layout to include precision)
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('ReLU CNN Training Results', fontsize=16)
        
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
        axes[1,0].plot(history['train_f1'], label='Train')
        axes[1,0].plot(history['val_f1'], label='Validation')
        axes[1,0].set_title('F1 Score Over Time')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot Precision
        axes[1,1].plot(history['train_precision'], label='Train')
        axes[1,1].plot(history['val_precision'], label='Validation')
        axes[1,1].set_title('Precision Over Time')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Plot Confusion Matrix
        if 'val_confusion_matrix' in history:
            final_cm = history['val_confusion_matrix'][-1]
            sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', ax=axes[2,0])
            axes[2,0].set_title('Final Confusion Matrix')
            axes[2,0].set_ylabel('True Label')
            axes[2,0].set_xlabel('Predicted Label')
        
        # Remove empty subplot
        fig.delaxes(axes[2,1])
        
        plt.tight_layout()
        plt.savefig(f'{base_name}_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nTraining Summary:")
        print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
        print(f"Best F1-Score: {max(history['val_f1']):.2f}")
        print(f"Final Loss: {history['val_loss'][-1]:.4f}")
        
        # Print the history structure for debugging
        print("\nHistory keys:", history.keys())
        print("\nShape of val_acc:", np.array(history['val_acc']).shape)
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cnn_relu_plot.py <path_to_history.npy>")
        sys.exit(1)
        
    history_path = sys.argv[1]
    if not os.path.exists(history_path):
        print(f"Error: File '{history_path}' not found!")
        sys.exit(1)
        
    plot_training_metrics(history_path) 