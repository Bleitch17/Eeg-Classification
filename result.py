import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View model results saved in .npy files.")
    parser.add_argument("file", type=str, help="Path to .npy file.")
    args = parser.parse_args()

    history = np.load(args.file, allow_pickle=True).item()

    print("Keys in history:")
    print(history.keys())

    # Example - show the confusion matrix of the last epoch
    print(history['val_conf_matrix'][-1])