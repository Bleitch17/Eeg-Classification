import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Callable

# Type aliases
AccuracyScore = float
ConfusionMatrix = list[list[int]]
ClassificationReport = str

def train_random_forest(X: pd.DataFrame, y: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train a Random Forest classifier using k-fold cross validation.
    """
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    # Store last fold's results for detailed reporting
    final_confusion_matrix = None
    final_classification_report = None
    
    scaler = StandardScaler()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train_scaled, y_train)
        y_pred = rfc.predict(X_val_scaled)
        
        fold_accuracies.append(accuracy_score(y_val, y_pred))
        
        # Store last fold's detailed results
        if fold == k_folds - 1:
            final_confusion_matrix = confusion_matrix(y_val, y_pred)
            final_classification_report = classification_report(y_val, y_pred)
    
    mean_accuracy = np.mean(fold_accuracies)
    print(f"Fold accuracies: {[f'{acc:.2f}' for acc in fold_accuracies]}")
    print(f"Std deviation: {np.std(fold_accuracies):.2f}")
    
    return mean_accuracy, final_confusion_matrix, final_classification_report

def experiment_initial(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df[df["Label"] != 0]
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_5_channel_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df[df["Label"] != 0]
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_left_right_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_left_right_hand_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

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
    return train_random_forest(X, y)

def experiment_resting_vs_left_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]
    
    # Get all timepoints for all channels
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

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
    return train_random_forest(X, y)

def experiment_resting_vs_all_single_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    df.loc[df["Label"] >= 1, "Label"] = 1
    
    # Get all timepoints for C3 channel
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_resting_vs_left_hand_c3(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_resting_vs_left_hand_c4(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]
    selected_columns = [col for col in df.columns if col.startswith('C4_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_resting_vs_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
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
    return train_random_forest(X, y)

def experiment_resting_vs_all(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    df.loc[df["Label"] >= 1, "Label"] = 1
    
    # Get all timepoints for all channels
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

def experiment_resting_vs_all_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    df.loc[df["Label"] >= 1, "Label"] = 1
    
    # Get all timepoints for all channels
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_random_forest(X, y)

if __name__ == "__main__":
    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90 --flatten"
    flat_df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90_flattened.parquet")

    experiments: list[tuple[str, Callable[[pd.DataFrame], tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]]]] = [
        ("Initial (All Channels)", experiment_initial),
        # ("No Rest (All Channels)", experiment_no_rest),
        # ("5 Channel", experiment_5_channel),
        # ("5 Channel No Rest", experiment_5_channel_no_rest),
        # ("Left vs Right Hand", experiment_left_right_hand),
        # # ("Left vs Right Hand 5 Channel", experiment_left_right_hand_5_channel),
        # ("Left vs Right Hand 2 Channel", experiment_left_right_hand_2_channel),
        # ("Resting vs Left Hand", experiment_resting_vs_left_hand),
        # ("Resting vs Left Hand 2 Channel", experiment_resting_vs_left_hand_2_channel),
        # ("Resting vs Left Hand C3", experiment_resting_vs_left_hand_c3),
        # ("Resting vs Left Hand C4", experiment_resting_vs_left_hand_c4),
        # ("Resting vs Left/Right Hand 2 Channel", experiment_resting_vs_left_right_hand_2_channel),
        # ("Resting vs All Single Channel", experiment_resting_vs_all_single_channel),
        # ("Resting vs All 5 Channel", experiment_resting_vs_all_5_channel),
        # ("Resting vs All", experiment_resting_vs_all)
    ]

    for experiment_name, experiment_function in experiments:
        print(f"\nRunning experiment: {experiment_name}")
        accuracy, conf_matrix, class_report = experiment_function(flat_df)
        print(f"Average accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
