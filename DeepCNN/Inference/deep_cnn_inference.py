import os
import pandas as pd
import torch
import torch.nn as nn
import sys

from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from dataset_bci_iv_2a.dataset import BciIvDataset
from DeepCNN.Model.deepCNN_better import Net
from device import get_system_device


def create_test_parquet() -> None:
    # NOTE - best fold is fold 4, which is fold index 3
    df = pd.read_parquet("../../dataset_bci_iv_2a/A01_100_90.parquet")
    print(df.head(3))

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (_, test_idx) in enumerate(kfold.split(df)):
        if fold != 3:
            continue

        test_df = df.iloc[test_idx]
        test_df.to_parquet("A01_100_90_deep_cnn.parquet", engine="pyarrow")
        break


if __name__ == "__main__":
    if not os.path.exists("A01_100_90_deep_cnn.parquet"):
        print("==================================================================================================================")
        print("Missing required files for inference.")
        print("Please download the following files from Google Drive, and place them in the DeepCNN/Inference directory:")
        print("Missing: A01_100_90_deep_cnn.parquet")
        print("Google Drive Url: https://drive.google.com/drive/folders/1zoJrLGljjhZXrgfG6OspXZZ7gZ3NFLOL?usp=sharing")
        print("==================================================================================================================")

        exit()
    
    test_df = pd.read_parquet("A01_100_90_deep_cnn.parquet", engine="pyarrow")
    test_labels = test_df["Label"]
    test_features = test_df.drop(columns=["Label"])

    test_dataset = BciIvDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("deepCNN_better.pth")['model_state_dict'])

    device = get_system_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_loss += loss.item()
    
    # Loss and accuracy
    test_total_predictions = len(all_labels)
    test_correct_predictions = sum([1 for i in range(test_total_predictions) if all_labels[i] == all_predictions[i]])
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct_predictions / test_total_predictions

    precision = precision_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Test Loss: {avg_test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {conf_matrix}")
