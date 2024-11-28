import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def plot_train_test_split_label_distribution() -> None:
    df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90.parquet", engine="pyarrow")

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Visualize the distribution of the "Label" column values in the train set
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    train_counts = train['Label'].value_counts().sort_index()
    plt.bar(train_counts.index, train_counts.values)
    plt.title('Train Set Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')

    # Visualize the distribution of the "Label" column values in the test set
    plt.subplot(1, 2, 2)
    test_counts = test['Label'].value_counts().sort_index()
    plt.bar(test_counts.index, test_counts.values)
    plt.title('Test Set Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(conf_matrix: np.ndarray) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = conf_matrix.shape[0]
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_train_loss_and_accuracy(history: dict) -> None:
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.axhline(y=history['test_loss'], color='orange', linestyle='--', label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.axhline(y=history['test_acc'], color='orange', linestyle='--', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_train_test_split_label_distribution()
    
    parser = argparse.ArgumentParser(description="View model results saved in .npy files.")
    parser.add_argument("file", type=str, help="Path to .npy file.")
    args = parser.parse_args()

    history = np.load(args.file, allow_pickle=True).item()

    print("Keys in history:")
    print(history.keys())

    # Example - show the confusion matrix of the last epoch
    # print(history['conf_matrix'][-1])
    # print(history['train_acc'])
    # print(history['train_loss'])
    print(history['precision'])
    print(history['f1'])

    # visualize_confusion_matrix(history['conf_matrix'][-1])
    plot_train_loss_and_accuracy(history)

