import os
import pandas as pd
import torch
import torch.nn as nn
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from dataset_bci_iv_2a.dataset import BciIvDataset
from device import get_system_device
from LSTM.Model.lstm import LSTM
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


if __name__ == "__main__":
    if not os.path.exists("LSTM.pth") or not os.path.exists("A01_100_90_lstm.parquet"):
        print("==================================================================================================================")
        print("Missing required files for inference.")
        print("Please download the following files from Google Drive, and place them in the LSTM/Inference directory:")
        
        if not os.path.exists("LSTM.pth"):
            print("Missing: LSTM.pth")
        
        if not os.path.exists("A01_100_90_lstm.parquet"):
            print("Missing: A01_100_90_lstm.parquet")
        
        print("Google Drive Url: https://drive.google.com/drive/folders/1zoJrLGljjhZXrgfG6OspXZZ7gZ3NFLOL?usp=sharing")
        print("==================================================================================================================")

        exit()

    model = LSTM()
    model.load_state_dict(torch.load("LSTM.pth")['model_state_dict'])

    device = get_system_device()
    model.to(device)

    print("Model loaded successfully")

    test_df = pd.read_parquet("A01_100_90_lstm.parquet", engine="pyarrow")
    test_labels = test_df["Label"]
    test_features = test_df.drop(columns=["Label"])

    test_dataset = BciIvDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # NOTE - need to switch the dimensions of the input tensor for compatibility with the LSTM:
            # LSTM expects input tensor of shape (batch_size, seq_len, input_size)
            outputs = model(inputs.permute(0, 2, 1))

            # Loss calculation
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get the predictions
            _, predicted = torch.max(outputs, 1)

            # Collect all the predictions made 
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Loss and accuracy
    test_total_predictions = len(all_labels)
    test_correct_predictions = sum([1 for i in range(test_total_predictions) if all_labels[i] == all_predictions[i]])
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct_predictions / test_total_predictions

    print(f"Test Loss: {avg_test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Additional metrics: precision, f1-score, confusion matrix
    precision = precision_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {conf_matrix}")
