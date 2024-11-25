import numpy as np

# Load training history
def load_training_history(fold_number):
    history = np.load(f'fold_{fold_number}_history.npy', allow_pickle=True)
    history = history.item()  # Convert from numpy object to dictionary
    
    print("Available metrics:", history.keys())
    print(f"Training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    return history

# Plot training history
def plot_fold_history(history):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


history = load_training_history(3)  # TODO: Change this to the fold number you want to plot
plot_fold_history(history)
