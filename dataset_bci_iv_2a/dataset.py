import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


CSV_DELIMITER: str = ","


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < 3*order:  # Check if data is long enough
        return data  # Return original data if too short
        
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    
    try:
        # Apply the filter forwards and backwards to remove phase delay
        y = signal.filtfilt(b, a, data)
        return y
    except ValueError:
        # If filtfilt fails, try using regular filter
        y = signal.lfilter(b, a, data)
        return y


def filter_dataframe(df: pd.DataFrame, lowcut: float, highcut: float, fs: float, order: int = 4) -> pd.DataFrame:
    filtered_df = df.copy()
    data_columns = df.columns.drop(["Recording", "Label"])

    for recording in df["Recording"].unique():
        recording_indices = df["Recording"] == recording
        segment_length = sum(recording_indices)
        
        # Skip very short segments or adjust filter order
        if segment_length < 50:  # minimum length threshold
            continue
            
        for column in data_columns:
            data = df.loc[recording_indices, column].values
            
            # Adjust filter order based on segment length
            actual_order = min(order, int(segment_length/10))
            if actual_order < 1:
                actual_order = 1
                
            try:
                filtered_df.loc[recording_indices, column] = butter_bandpass_filter(
                    data, lowcut, highcut, fs, actual_order
                )
            except ValueError:
                # If filtering fails, keep original data
                print(f"Warning: Filtering failed for recording {recording}, channel {column}")
                continue

    return filtered_df


class BciIvCsvParser:
    def __init__(self, csv_file_path: str) -> None:
        # Data will be internally represented as a dictionary of lists,
        # for convenient conversion to a pandas DataFrame
        self.data: dict[str, list[float]] = {}
        self.headers: list[str] = []

        self.parse(csv_file_path)

    def parse(self, csv_file_path: str) -> None:    
        """
        Loads contents of the CSV file into the internal container.
        """

        with open(csv_file_path, "r") as csv_file:
            # The first row is expected to contain the headers:
            self.headers = csv_file.readline().strip().split(CSV_DELIMITER)
            self.data = {header: [] for header in self.headers}

            while line := csv_file.readline():
                if not line or line == "\n":
                    # End of file reached, should be no blank lines except for the last one
                    break

                data_segments: list[float] = list(map(float, line.strip().split(CSV_DELIMITER)))

                for measurement, header in zip(data_segments, self.headers):
                    self.data[header].append(measurement)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)


class BciIvDataset(Dataset):
    """
    Provides an interface for retrieving samples from the BCI Competition IV dataset,
    for use with the PyTorch package.
    """

    def __init__(self, data: pd.DataFrame, labels: pd.Series, window_size: int) -> None:
        """
        The data parameter should be of size (M, 22) where N is the number of samples, and 22 is the number of EEG channels.
        The labels parameter should be of size N.
        """
        self.data: pd.DataFrame = data
        self.labels: pd.Series = labels
        self.window_size: int = window_size

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        array = np.stack(self.data.iloc[idx].values)

        # Note - for each item, labels should return an index into a list of classes.
        # In this case, the classes (in order) are: "Rest", "Left", "Right", "Feet", and "Tongue"
        return torch.tensor(array, dtype=torch.float32), int(self.labels.iloc[idx])


def create_windowed_dictionary(df: pd.DataFrame, window_size: int, window_overlap: int) -> dict[str, list[list[float]] | list[float]]:
    if window_size < 1:
        raise ValueError(f"Window size must be at least 1, but got {window_size}")

    if window_overlap >= window_size:
        raise ValueError(f"Window overlap must be less than window size")

    if len(df) < window_size:
        return {}

    label_column: str = "Label"
    data_columns: list[str] = df.columns.drop(label_column).tolist()

    # Will have columns for each EEG channel, plus a label column
    windowed_data: dict[str, list[list[float]]] = {header: [] for header in data_columns}
    windowed_data[label_column] = []

    # Modified windowing process
    for window_base_index in range(0, len(df) - window_size + 1, window_size - window_overlap):
        window_end = window_base_index + window_size
        
        # Get window for each channel
        for data_column in data_columns:
            # Note: No need to filter here as we already did in preprocessing
            window = df[data_column].values[window_base_index:window_end]
            windowed_data[data_column].append(window)
        
        # Get most common label in this window
        window_labels = df[label_column].values[window_base_index:window_end]
        most_common_label = stats.mode(window_labels, keepdims=True)[0][0]
        windowed_data[label_column].append(most_common_label)

    return windowed_data


class EnhancedPreprocessing:
    @staticmethod
    def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # 1. Bandpass filter (8-30 Hz)
        filtered_df = filter_dataframe(filtered_df, 8.0, 30.0, 250.0)
        
        # 2. Per-channel normalization
        columns_to_drop = []
        if "Label" in filtered_df.columns:
            columns_to_drop.append("Label")
        if "Recording" in filtered_df.columns:
            columns_to_drop.append("Recording")
            
        features = filtered_df.drop(columns=columns_to_drop)
        
        # Fix deprecated fillna warning
        features = features.ffill().bfill()
        
        scaler = StandardScaler()
        normalized_df = pd.DataFrame(
            scaler.fit_transform(features), 
            columns=features.columns,
            index=features.index
        )
        
        # 3. Baseline correction with proper indexing
        def apply_baseline_correction(data, recording_series):
            baseline_df = data.copy()
            for recording in recording_series.unique():
                try:
                    # Get indices for this recording
                    recording_mask = recording_series == recording
                    recording_indices = recording_series[recording_mask].index
                    
                    if len(recording_indices) >= 100:
                        # Calculate baseline using first 100 samples
                        baseline = data.loc[recording_indices[:100]].mean()
                        # Apply correction using aligned indices
                        baseline_df.loc[recording_indices] = data.loc[recording_indices] - baseline
                except Exception as e:
                    print(f"Warning: Baseline correction failed for recording {recording}: {e}")
            return baseline_df
        
        # Pass the Recording series for proper index alignment
        corrected_df = apply_baseline_correction(normalized_df, df["Recording"])
        
        # 4. Artifact removal with proper indexing
        def remove_artifacts(data, z_threshold=3):
            clean_df = data.copy()
            for col in data.columns:
                try:
                    z_scores = stats.zscore(data[col], nan_policy='omit')
                    artifacts = abs(z_scores) > z_threshold
                    clean_df.loc[artifacts, col] = data[col].mean()
                except Exception as e:
                    print(f"Warning: Artifact removal failed for column {col}: {e}")
            return clean_df
        
        clean_df = remove_artifacts(corrected_df)
        
        # Restore Label and Recording columns with proper indexing
        clean_df["Label"] = df["Label"]
        if "Recording" in df.columns:
            clean_df["Recording"] = df["Recording"]
        
        return clean_df


class BciIvDatasetFactory:
    @staticmethod
    def create_k_fold(subject_number: int, window_size: int, window_overlap: int, k_folds: int = 5) -> Tuple[pd.DataFrame, pd.Series, KFold]:
        try:
            subject_str = f"0{subject_number}" if subject_number < 10 else str(subject_number)
            evaluation_csv_parser = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}E.csv")
            training_csv_parser = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}T.csv")
            
            evaluation_df = evaluation_csv_parser.get_dataframe()
            training_df = training_csv_parser.get_dataframe()
            
            # Print actual columns for debugging
            print("Evaluation columns:", evaluation_df.columns.tolist())
            print("Training columns:", training_df.columns.tolist())
            
            # Check for required column types instead of exact names
            required_column_types = {
                'eeg': 22,  # Need 22 EEG channels
                'special': ['Label', 'Recording']  # Need these exact columns
            }
            
            def validate_columns(df):
                columns = df.columns.tolist()
                eeg_channels = [col for col in columns if col not in ['EOGL', 'EOGM', 'EOGR', 'Label', 'Recording']]
                special_columns = ['Label', 'Recording']
                
                if len(eeg_channels) != required_column_types['eeg']:
                    raise ValueError(f"Expected 22 EEG channels, found {len(eeg_channels)}")
                if not all(col in columns for col in special_columns):
                    raise ValueError(f"Missing required columns: {[col for col in special_columns if col not in columns]}")
                
                return eeg_channels
            
            # Validate both dataframes
            eval_eeg = validate_columns(evaluation_df)
            train_eeg = validate_columns(training_df)
            
            # Ensure both dataframes have the same columns
            if set(eval_eeg) != set(train_eeg):
                raise ValueError("EEG channel names don't match between evaluation and training data")
            
            # Drop EOG channels first
            evaluation_df = evaluation_df.drop(columns=["EOGL", "EOGM", "EOGR"])
            training_df = training_df.drop(columns=["EOGL", "EOGM", "EOGR"])
            
            # Combine datasets
            recording_offset = evaluation_df["Recording"].max() + 1
            training_df["Recording"] += recording_offset
            raw_df = pd.concat([evaluation_df, training_df])
            
            # Apply preprocessing
            processed_df = EnhancedPreprocessing.apply_preprocessing(raw_df)
            
            # Create windows
            labeled_df = processed_df.drop(columns=["Recording"])
            windowed_data = create_windowed_dictionary(labeled_df, window_size, window_overlap)
            
            if not windowed_data:
                raise ValueError("No windows created - check data and window parameters")
            
            # Convert to DataFrame
            windowed_df = pd.DataFrame(windowed_data)
            
            # Split features and labels
            labels = windowed_df["Label"]
            features = windowed_df.drop(columns=["Label"])
            
            # Create K-fold splits (removed augmentation)
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            return features, labels, kfold
            
        except Exception as e:
            raise Exception(f"Error processing dataset: {str(e)}") from e