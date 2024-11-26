import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Callable, Tuple, List

# Type aliases
AccuracyScore = float
ConfusionMatrix = list[list[int]]
ClassificationReport = str

def train_svm(X: pd.DataFrame, y: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train an SVM classifier using k-fold cross validation.
    Using RBF kernel which is suitable for EEG data's non-linear patterns.
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
        
        # Using RBF kernel which is good for EEG data
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_val_scaled)
        
        fold_accuracies.append(accuracy_score(y_val, y_pred))
        
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
    return train_svm(X, y)

def experiment_resting_vs_left_hand_c3(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Get all timepoints for C3 channel
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_svm(X, y)

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
    return train_svm(X, y)

def experiment_resting_vs_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    # Extract resting state and left/right hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1 or x == 2)
    df = df[criterion]

    # Get all timepoints for C3 and C4 channels
    channels = ["C3", "C4"]
    selected_columns = []
    for channel in channels:
        selected_columns.extend([col for col in df.columns if col.startswith(channel + '_t')])
    
    X = df[selected_columns]
    y = df["Label"]
    return train_svm(X, y)

def experiment_resting_vs_all(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_svm(X, y)

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
    return train_svm(X, y)

def experiment_resting_vs_all_single_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    df = df.copy()
    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Get all timepoints for C3 channel
    selected_columns = [col for col in df.columns if col.startswith('C3_t')]
    X = df[selected_columns]
    y = df["Label"]
    return train_svm(X, y)

if __name__ == "__main__":
    # NOTE - can produce by running "python dataset_bci_iv_2a/dataset.py 1 100 90 --flatten"
    flat_df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90_flattened.parquet")
    
    # Verify columns
    print("Columns in dataset:", flat_df.columns[:5], "...", flat_df.columns[-5:])
    print("Total features:", len(flat_df.columns) - 1)  # -1 for Label column
    
    experiments = [
        ("Initial", experiment_initial),
        # ("Resting vs Left Hand (C3)", experiment_resting_vs_left_hand_c3),
        # ("Resting vs Left Hand 2 Channel", experiment_resting_vs_left_hand_2_channel),
        # ("Resting vs Left/Right Hand (2 Channel)", experiment_resting_vs_left_right_hand_2_channel),
        # ("Resting vs All Single Channel", experiment_resting_vs_all_single_channel),
        # ("Resting vs All 5 Channel", experiment_resting_vs_all_5_channel),
        # ("Resting vs All", experiment_resting_vs_all),
    ]

    for experiment_name, experiment_function in experiments:
        print(f"\nRunning experiment: {experiment_name}")
        try:
            accuracy, conf_matrix, class_report = experiment_function(flat_df)
            print(f"Average accuracy: {accuracy:.2f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(class_report)
        except Exception as e:
            print(f"Error in experiment {experiment_name}: {str(e)}")