import os
import pandas as pd
import sys

from joblib import load
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from dataset_bci_iv_2a.dataset import BciIvDataset
from device import get_system_device


def create_test_parquet() -> None:
    # NOTE - best fold is fold 0
    flat_df = pd.read_parquet("../../dataset_bci_iv_2a/A01_100_90_flattened.parquet")
    print(flat_df.head(3))

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (_, test_idx) in enumerate(kfold.split(flat_df)):
        test_df = flat_df.iloc[test_idx]

    test_df.to_parquet("A01_100_90_naive_bayes.parquet", engine="pyarrow")


if __name__ == "__main__":
    if not os.path.exists("A01_100_90_naive_bayes.parquet"):
        print("==================================================================================================================")
        print("Missing required files for inference.")
        print("Please download the following files from Google Drive, and place them in the LSTM/Inference directory:")
        print("Missing: A01_100_90_naive_bayes.parquet")
        print("Google Drive Url: https://drive.google.com/drive/folders/1zoJrLGljjhZXrgfG6OspXZZ7gZ3NFLOL?usp=sharing")
        print("==================================================================================================================")

        exit()
    
    gnb = load("BAYES.joblib")
    df = pd.read_parquet("A01_100_90_naive_bayes.parquet", engine="pyarrow")

    labels = df["Label"]
    features = df.drop(columns=["Label"])

    predictions = gnb.predict(features)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    conf_matrix = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {conf_matrix}")
