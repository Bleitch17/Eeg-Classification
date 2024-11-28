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


if __name__ == "__main__":
    plot_train_test_split_label_distribution()
    
    parser = argparse.ArgumentParser(description="View model results saved in .npy files.")
    parser.add_argument("file", type=str, help="Path to .npy file.")
    args = parser.parse_args()

    history = np.load(args.file, allow_pickle=True).item()

    print("Keys in history:")
    print(history.keys())

    # Example - show the confusion matrix of the last epoch
    print(history['conf_matrix'][-1])
    