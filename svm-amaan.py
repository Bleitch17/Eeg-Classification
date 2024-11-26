import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from typing import Callable, Tuple, List
from dataset_bci_iv_2a.dataset import BciIvDatasetFactory, preprocessing_config

# Type aliases
AccuracyScore = float
ConfusionMatrix = List[List[int]]
ClassificationReport = str

# Parameters for Bagging
bagging_params = {
    "n_estimators": 20,
    "max_samples": 0.8,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

# Train SVM
def train_svm(X: pd.DataFrame, y: pd.Series) -> Tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train an SVM classifier using k-fold cross-validation.
    """
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    final_conf_matrix = None
    final_class_report = None
    scaler = StandardScaler()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train SVM
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_val_scaled)

        fold_accuracies.append(accuracy_score(y_val, y_pred))

        if fold == k_folds - 1:
            final_conf_matrix = confusion_matrix(y_val, y_pred)
            final_class_report = classification_report(y_val, y_pred)

    mean_accuracy = np.mean(fold_accuracies)
    print(f"Fold accuracies: {fold_accuracies}")
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    return mean_accuracy, final_conf_matrix, final_class_report

# Train Bagging SVM
def train_bagging_svm(X: pd.DataFrame, y: pd.Series) -> Tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train a BaggingClassifier with SVM as the base estimator.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Bagging SVM
    svm_base = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    bagging_clf = BaggingClassifier(base_estimator=svm_base, **bagging_params)
    bagging_clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = bagging_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    return accuracy, conf_matrix, class_report

def flatten_features(features: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Flatten the windowed features into a single DataFrame with consistent dimensions.
    """
    flat_features = []
    invalid_windows = 0

    for col in features.columns:
        if col == "Recording":
            continue

        # Convert each feature column's list of arrays into a DataFrame
        column_data = features[col].tolist()
        processed_data = []
        for row in column_data:
            if isinstance(row, (list, np.ndarray)) and len(row) == window_size:
                processed_data.append(row)
            else:
                # If the row doesn't match the expected window size, skip it
                invalid_windows += 1
                processed_data.append([np.nan] * window_size)  # Use NaN padding for mismatched rows

        flat_feature = pd.DataFrame(
            processed_data, columns=[f"{col}_t{i}" for i in range(window_size)]
        )
        flat_features.append(flat_feature)

    print(f"Skipped {invalid_windows} windows with invalid shapes.")
    return pd.concat(flat_features, axis=1)

# Define Experiments
def run_experiment(
    df: pd.DataFrame,
    experiment_fn: Callable[[pd.DataFrame], Tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]],
    experiment_type: str,
):
    print(f"\nRunning {experiment_type} experiment...")
    accuracy, conf_matrix, class_report = experiment_fn(df)
    print(f"Final Accuracy: {accuracy:.2f}")
    if conf_matrix:
        print("\nConfusion Matrix:")
        print(conf_matrix)
    if class_report:
        print("\nClassification Report:")
        print(class_report)

# Experiment Implementations
def experiment_resting_vs_left_hand_c3(df: pd.DataFrame, trainer=train_svm):
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]
    selected_columns = [col for col in df.columns if col.startswith("C3_t")]
    X = df[selected_columns]
    y = df["Label"]
    return trainer(X, y)

def experiment_resting_vs_all_5_channel(df: pd.DataFrame, trainer=train_svm):
    df = df.copy()
    df.loc[df["Label"] >= 1, "Label"] = 1
    channels = ["Fz", "C3", "Cz", "C4", "Pz"]
    selected_columns = [col for ch in channels for col in df.columns if col.startswith(f"{ch}_t")]
    X = df[selected_columns]
    y = df["Label"]
    return trainer(X, y)

# Main Script
if __name__ == "__main__":
    # Dataset parameters
    subject_number = 1
    window_size = 100
    window_overlap = 95

    # Load data
    features, labels, _ = BciIvDatasetFactory.create_k_fold(subject_number, window_size, window_overlap, preprocessing_config)

    # Flatten features
    flat_features = flatten_features(features, window_size)
    flat_features["Label"] = labels

    # Run SVM and Bagging SVM experiments
    experiments = [
        ("Resting vs Left Hand (C3) - SVM", lambda df: experiment_resting_vs_left_hand_c3(df, trainer=train_svm)),
        ("Resting vs Left Hand (C3) - Bagging SVM", lambda df: experiment_resting_vs_left_hand_c3(df, trainer=train_bagging_svm)),
        ("Resting vs All (5 Channels) - SVM", lambda df: experiment_resting_vs_all_5_channel(df, trainer=train_svm)),
        ("Resting vs All (5 Channels) - Bagging SVM", lambda df: experiment_resting_vs_all_5_channel(df, trainer=train_bagging_svm)),
    ]

    for experiment_name, experiment_function in experiments:
        run_experiment(flat_features, experiment_function, experiment_name)
