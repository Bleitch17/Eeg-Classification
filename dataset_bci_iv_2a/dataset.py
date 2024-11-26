import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import logging
from typing import List, Tuple
import os
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Global Constants
CSV_DELIMITER: str = ","

# Configuration for preprocessing
preprocessing_config = {
    "lowcut": 8.0,
    "highcut": 30.0,
    "fs": 250,  # Sampling frequency
    "filter_order": 4,
    "baseline_samples": 100,
    "artifact_threshold": 3.0,
    "augmentation_noise_std": 0.01
}

# Butterworth Bandpass Filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < 3 * order:
        logging.warning("Data segment too short for filtering. Returning original data.")
        return data

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        return signal.filtfilt(b, a, data)
    except ValueError as e:
        logging.warning(f"filtfilt failed: {e}. Using lfilter instead.")
        return signal.lfilter(b, a, data)

# Parallelized Filtering
def filter_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    filtered_df = df.copy()
    data_columns = df.columns.drop(["Recording", "Label"])
    lowcut, highcut, fs, order = config["lowcut"], config["highcut"], config["fs"], config["filter_order"]

    for recording in df["Recording"].unique():
        recording_indices = df["Recording"] == recording
        segment_length = sum(recording_indices)
        if segment_length < 50:
            logging.warning(f"Skipping short recording: {recording} (Length: {segment_length})")
            continue

        filtered = Parallel(n_jobs=-1)(
            delayed(butter_bandpass_filter)(
                df.loc[recording_indices, column].values,
                lowcut, highcut, fs, order
            ) for column in data_columns
        )
        for i, column in enumerate(data_columns):
            filtered_df.loc[recording_indices, column] = filtered[i]
    return filtered_df

# Advanced Preprocessing
class EnhancedPreprocessing:
    @staticmethod
    def apply_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        logging.info("Starting preprocessing...")

        # Ensure indices are unique to avoid alignment issues
        if df.index.duplicated().any():
            df = df.reset_index(drop=True)

        # Filter DataFrame
        filtered_df = filter_dataframe(df, config)
        logging.info("Bandpass filtering complete.")

        # Normalize
        features = filtered_df.drop(columns=["Recording", "Label"]).ffill().bfill()
        scaler = StandardScaler()
        normalized_df = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index  # Keep index alignment
        )
        logging.info("Normalization complete.")

        # Baseline Correction
        def apply_baseline_correction(data, recording_series):
            baseline_df = data.copy()
            for recording in recording_series.unique():
                indices = recording_series[recording_series == recording].index
                if len(indices) >= config["baseline_samples"]:
                    baseline = data.loc[indices[:config["baseline_samples"]]].mean()
                    baseline_df.loc[indices] = data.loc[indices] - baseline
            return baseline_df

        corrected_df = apply_baseline_correction(normalized_df, df["Recording"])
        logging.info("Baseline correction complete.")

        # Artifact Removal
        def remove_artifacts(data, threshold):
            z_scores = stats.zscore(data, nan_policy="omit")
            mask = np.abs(z_scores) > threshold
            return data[~mask.any(axis=1)]

        clean_df = remove_artifacts(corrected_df, config["artifact_threshold"])
        logging.info("Artifact removal complete.")

        # Align Labels and Recording columns
        aligned_df = clean_df.copy()
        aligned_df["Label"] = df.loc[clean_df.index, "Label"]
        aligned_df["Recording"] = df.loc[clean_df.index, "Recording"]

        return aligned_df

# Data Augmentation
def augment_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    augmented_df = df.copy()
    for column in augmented_df.columns:
        if column.startswith("C"):  # Assuming EEG channel columns start with 'C'
            augmented_df[column] += np.random.normal(0, config["augmentation_noise_std"], size=len(augmented_df))
    return augmented_df

# Stratified K-Fold Creation
def create_stratified_k_fold(features: pd.DataFrame, labels: pd.Series, k_folds=5):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    return skf.split(features, labels)

# Main Data Processing and K-Fold Handling
class BciIvDatasetFactory:
    @staticmethod
    def create_k_fold(subject_number: int, window_size: int, window_overlap: int, config: dict, k_folds: int = 5):
        logging.info(f"Preparing dataset for subject {subject_number}...")
        subject_str = f"0{subject_number}" if subject_number < 10 else str(subject_number)

        # Define base path
        base_path = "/home/ubuntu/Eeg-Classification/dataset_bci_iv_2a"
        eval_file = os.path.join(base_path, f"A{subject_str}E.csv")
        train_file = os.path.join(base_path, f"A{subject_str}T.csv")

        if not os.path.exists(eval_file) or not os.path.exists(train_file):
            raise FileNotFoundError(f"Required files not found: {eval_file}, {train_file}")

        eval_parser = BciIvCsvParser(eval_file)
        train_parser = BciIvCsvParser(train_file)

        eval_df = eval_parser.get_dataframe()
        train_df = train_parser.get_dataframe()

        # Combine and preprocess
        train_df["Recording"] += eval_df["Recording"].max() + 1
        combined_df = pd.concat([eval_df, train_df])
        processed_df = EnhancedPreprocessing.apply_preprocessing(combined_df, config)

        # Augment Data
        augmented_df = augment_data(processed_df, config)
        final_df = pd.concat([processed_df, augmented_df])
        logging.info("Data augmentation complete.")

        # Feature Extraction
        feature_columns = [col for col in final_df.columns if col not in ["Label", "Recording"]]
        labels = final_df["Label"]

        # Stratified K-Fold
        kfold = create_stratified_k_fold(final_df[feature_columns], labels, k_folds)
        return final_df[feature_columns], labels, kfold

class BciIvCsvParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading CSV file: {self.file_path}")
            df = pd.read_csv(self.file_path, delimiter=CSV_DELIMITER)
            logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {self.file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading CSV file {self.file_path}: {e}")
            raise
