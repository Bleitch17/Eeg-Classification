import os
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from typing import Callable


# Type aliases
AccuracyScore = float
ConfusionMatrix = list[list[int]]
ClassificationReport = str

# Other constants
BCI_COMP_DATASET_PATH: str = "dataset_bci_iv_2a/"

# Used as seed for test_train_split
RANDOM_STATE: int = 17 
TEST_SIZE: float = 0.2


def train_naive_bayes(X: pd.DataFrame, y: pd.DataFrame) -> tuple[list[float], list[float], list[float], ConfusionMatrix, ClassificationReport]:
    """
    Train a Naive Bayes classifier using k-fold cross validation.
    Returns:
        tuple containing:
        - list of accuracies per fold
        - list of precisions per fold
        - list of f1-scores per fold
        - final confusion matrix
        - classification report
    """
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_precisions = []
    fold_f1s = []
    
    final_confusion_matrix = None
    final_classification_report = None
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_val)
        
        # Calculate metrics for this fold
        fold_accuracies.append(accuracy_score(y_val, y_pred))
        fold_precisions.append(precision_score(y_val, y_pred, average='macro'))
        fold_f1s.append(f1_score(y_val, y_pred, average='macro'))
        
        # Store last fold's detailed results
        if fold == k_folds - 1:
            final_confusion_matrix = confusion_matrix(y_val, y_pred)
            final_classification_report = classification_report(y_val, y_pred)
    
    print(f"Fold accuracies: {[f'{acc:.2f}' for acc in fold_accuracies]}")
    print(f"Fold precisions: {[f'{prec:.2f}' for prec in fold_precisions]}")
    print(f"Fold F1-scores: {[f'{f1:.2f}' for f1 in fold_f1s]}")
    print(f"Std deviation: {np.std(fold_accuracies):.2f}")
    
    return fold_accuracies, fold_precisions, fold_f1s, final_confusion_matrix, final_classification_report


def experiment_initial(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Remove the resting state
    df = df[df["Label"] != 0]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Get all timepoints for the 5 channels
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_5_channel_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Remove the resting state
    df = df[df["Label"] != 0]

    # Get all timepoints for the 5 channels
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_left_right_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_left_right_hand_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Get all timepoints for the 5 channels
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Get all timepoints for C3 and C4 channels
    channels = ["C3", "C4"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]
    
    # Get all timepoints for C3 and C4 channels
    channels = ["C3", "C4"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_c4(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Get all timepoints for C4 channel
    selected_columns = [col for col in df.columns if col.startswith('C4_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_c3(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Get all timepoints for C3 channel
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left/right hand data (fixed criterion)
    criterion = df["Label"].map(lambda x: x == 0 or x == 1 or x == 2)  # Changed to include label 2
    df = df[criterion]

    # Get all timepoints for C3 and C4 channels
    channels = ["C3", "C4"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_all(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    df = df.copy()

    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_all_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Get all timepoints for the 5 channels
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


def experiment_resting_vs_all_single_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Get all timepoints for C3 channel
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_naive_bayes(X, y)


if __name__ == "__main__":
    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90 --flatten"
    flat_df = pd.read_parquet("../../dataset_bci_iv_2a/A01_100_90_flattened.parquet")

    experiments: list[tuple[str, Callable[[pd.DataFrame], tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]]]] = [
        ("Initial", experiment_initial),
        # ("Resting vs Left Hand", experiment_resting_vs_left_hand),
        # ("Resting vs Left Hand 2 Channel", experiment_resting_vs_left_hand_2_channel),
        # ("Resting vs All Single Channel", experiment_resting_vs_all_single_channel),
        # ("Resting vs All 5 Channel", experiment_resting_vs_all_5_channel),
        # ("Resting vs All", experiment_resting_vs_all),
    ]

    for experiment_name, experiment_function in experiments:
        print(f"\nRunning experiment: {experiment_name}")
        accuracy, conf_matrix, class_report = experiment_function(flat_df)
        print(f"Average accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)