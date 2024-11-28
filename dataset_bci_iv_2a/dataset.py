import argparse
import ast
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


CSV_DELIMITER: str = ","


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < 3 * order:  # Check if data is long enough
        print(f"WARNING: Data length {len(data)} is too short for filter order {order}")
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

    def __init__(self, eeg_features: pd.DataFrame, eeg_labels: pd.Series) -> None:
        """
        The data parameter should be of size (M, 22) where N is the number of samples, and 22 is the number of EEG channels.
        The labels parameter should be of size N.
        """
        self.labels: pd.Series = eeg_labels
        self.data: pd.DataFrame = eeg_features

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Join the channels along the first axis to create a ndarray of shape (num_channels, window_size)
        array = np.stack(self.data.iloc[idx].values)

        # Note - for each item, labels should return an index into a list of classes.
        # In this case, the classes (in order) are: "Rest", "Left", "Right", "Feet", and "Tongue"
        # The tensor has shape (num_channels, window_size)
        return torch.tensor(array, dtype=torch.float32), int(self.labels.iloc[idx])


class EnhancedPreprocessing:
    @staticmethod
    def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # 1. Bandpass filter (8-30 Hz) for reference not using it
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


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Validate that the columns in a DataFrame parsed from a CSV file match the expected format.
    """
    
    # Check for required column types instead of exact names
    required_column_types = {
        'eeg': 22,  # Need 22 EEG channels
        'special': ['Label', 'Recording']  # Need these exact columns
    }

    columns = df.columns.tolist()
    eeg_channels = [col for col in columns if col not in ['EOGL', 'EOGM', 'EOGR', 'Label', 'Recording']]
    special_columns = ['Label', 'Recording']
    
    if len(eeg_channels) != required_column_types['eeg']:
        raise ValueError(f"Expected 22 EEG channels, found {len(eeg_channels)}")
    
    if not all(col in columns for col in special_columns):
        raise ValueError(f"Missing required columns: {[col for col in special_columns if col not in columns]}")
    
    return True


def validate_dataframes(evaluation_df: pd.DataFrame, training_df: pd.DataFrame) -> bool:
    """
    Validate that the two dataframes have a compatible column structure.
    Should be called before combining the training and evaluation dataframes.
    """
    
    evaluation_df_columns = evaluation_df.columns.tolist()
    training_df_columns = training_df.columns.tolist()

    evaluation_eeg_channels = [col for col in evaluation_df.columns.tolist() if col not in ['EOGL', 'EOGM', 'EOGR', 'Label', 'Recording']]
    training_eeg_channels = [col for col in training_df.columns.tolist() if col not in ['EOGL', 'EOGM', 'EOGR', 'Label', 'Recording']]

    if set(evaluation_eeg_channels) != set(training_eeg_channels):
        raise ValueError("EEG channel names don't match between evaluation and training data")
    
    return True


def create_windowed_dictionary(df: pd.DataFrame, window_size: int, window_overlap: int, flatten: bool = False) -> dict[str, list[list[float]] | list[float]]:
    """
    Given a dataframe containing consecutive EEG recording samples, creates a dictionary of windows,
    where every window is a list of consecutive samples.
    """

    if len(df) < window_size:
        # It might happen that bursts of NaN values resulted in the matlab script creating very short consecutive recording sequences. If that happens,
        # ignore them and return an empty dictionary.
        return {}

    label_column: str = "Label"
    original_eeg_columns: list[str] = [column_name for column_name in df.columns if column_name != label_column]
    data_columns: list[str] = []

    if flatten:
        for column_name in original_eeg_columns:
            for sample_index in range(window_size):
                data_columns.append(f"{column_name}_{sample_index}")
    
    else:
        data_columns = original_eeg_columns[:]

    # Will have columns for each EEG channel, plus a label column
    windowed_data: dict[str, list[list[float]] | list[float]] = {column_name: [] for column_name in data_columns}

    if flatten:
        # E.g.: if window size is 100 and flatten is true, should be 2200 columns
        assert(len(windowed_data) == 22 * window_size)

    # Regardless of flattening, will always have a label column
    windowed_data[label_column] = []

    for window_base_index in range(0, len(df) - window_size + 1, window_size - window_overlap):
        # NOTE - important to iterate over the original columns here, to get the data from the provided dataframe
        for data_column in original_eeg_columns:
            window_samples = df[data_column].values[window_base_index:window_base_index + window_size]
            
            # TODO - apply preprocessing functions, e.g.: normalization, more filters, etc. in order on the window
            window_samples = butter_bandpass_filter(window_samples, 8.0, 50.0, 250.0, order=4)
        
            if flatten:
                for sample_index in range(window_size):
                    # This should append a single point in the window to the column
                    windowed_data[f"{data_column}_{sample_index}"].append(window_samples[sample_index])

            else:
                # This should append a list of points, i.e.: the window, to the column
                windowed_data[data_column].append(window_samples)

        # NOTE - currently, this dataset has been parsed such that all samples in the same recording index will have the same label
        # Therefore, can just take the label from the first sample in the window
        windowed_data[label_column].append(df[label_column].values[window_base_index])

    return windowed_data


def create_windowed_dataframe(df: pd.DataFrame, window_size: int, window_overlap: int, flatten: bool = False) -> pd.DataFrame:
    """
    Expects a dataframe with EEG columns, a label column, and a recording column.
    Creates a new dataframe with windows of EEG data, where the number of samples in each window is determined by the window size.

    If flatten is True, each time point in a window will have a distinct data column - use this mode for scikit-learn classifiers.
    If flatten is False, each window will be a list of samples - use this mode for PyTorch models.

    Returns a new dataframe with the windowed data.

    TODO - add an argument for pre-processing functions to apply on the windows.
    """
    
    if window_size < 1:
        raise ValueError(f"Window size must be at least 1, but got {window_size}")

    if window_overlap >= window_size:
        raise ValueError(f"Window overlap must be less than window size")

    if len(df) < window_size:
        raise ValueError(f"DataFrame length {len(df)} is less than window size {window_size}")
    
    original_eeg_columns: list[str] = [column_name for column_name in df.columns if column_name not in ["Label", "Recording"]]

    # Pandas will automatically convert a dictionary of lists into a DataFrame, so create a dictionary of lists to store the windowed data
    windowed_data: dict[str, list[list[float]] | list[float]] = { }

    if flatten:
        for column_name in original_eeg_columns:
            for sample_index in range(window_size):
                windowed_data[f"{column_name}_{sample_index}"] = []

    else:
        windowed_data = {column_name: [] for column_name in original_eeg_columns}

    # Regardless of flattening, will always have a label column
    windowed_data["Label"] = []

    df_group_iterable = df.drop(columns=["Recording"]).groupby(df["Recording"].values)

    # For each recording index, create a dictionary of windowed data then append it to the larger dictionary
    for _, df in df_group_iterable:
        recording_windows = create_windowed_dictionary(df, window_size, window_overlap, flatten=flatten)
        
        for column_name, windowed_column in recording_windows.items():
            windowed_data[column_name].extend(windowed_column)

    return pd.DataFrame(windowed_data)


def data(subject_number: int, window_size: int, window_overlap: int, flatten: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reads both the train and evaluation csv files into DataFrames, concatenates them, and creates windows.

    When flatten is True, the output can be used for scikit-learn classifiers.
    When flatten is False, the output can be used with the BciIvDataset class for compatibility with PyTorch models.

    Returns a tuple of (eeg_features, labels).

    TODO - add another argument for pre-processing functions to apply on the windows.
    """

    if subject_number < 1 or subject_number > 9:
        raise ValueError("Subject number must be between 1 and 9")

    evaluation_csv_parser = BciIvCsvParser(f"dataset_bci_iv_2a/A0{subject_number}E.csv")
    training_csv_parser = BciIvCsvParser(f"dataset_bci_iv_2a/A0{subject_number}T.csv")
    
    evaluation_df = evaluation_csv_parser.get_dataframe()
    training_df = training_csv_parser.get_dataframe()
    
    # Print actual columns for debugging
    print("Evaluation columns:", evaluation_df.columns.tolist())
    print("Training columns:", training_df.columns.tolist())
    
    # DataFrame validation
    validate_columns(evaluation_df)
    validate_columns(training_df)
    
    validate_dataframes(evaluation_df, training_df)
    
    # Drop EOG channels first
    evaluation_df = evaluation_df.drop(columns=["EOGL", "EOGM", "EOGR"])
    training_df = training_df.drop(columns=["EOGL", "EOGM", "EOGR"])
    
    # Combine datasets
    recording_offset = evaluation_df["Recording"].max() + 1
    training_df["Recording"] += recording_offset
    raw_df = pd.concat([evaluation_df, training_df])
    
    print("Raw DataFrame shape:", raw_df.shape)
    exit()

    # TODO - move preprocessing to the window creation step, since want to apply preprocessing on a per-window basis
    # processed_df = EnhancedPreprocessing.apply_preprocessing(raw_df)
    
    # Arrange the dataframe into windows, so temporal information can be fed into the classifiers
    windowed_df = create_windowed_dataframe(raw_df, window_size, window_overlap, flatten=flatten)
    
    # Split features and labels
    labels = windowed_df["Label"]
    features = windowed_df.drop(columns=["Label"])

    return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG data for BCI Competition IV.")
    parser.add_argument("subject_number", type=int, choices=range(1, 10), help="Subject number (1-9)")
    parser.add_argument("window_size", type=int, help="Size of the window")
    parser.add_argument("window_overlap", type=int, help="Overlap of the window")
    parser.add_argument("--flatten", action="store_true", help="Flatten the windowed data")

    args = parser.parse_args()

    features, labels = data(args.subject_number, args.window_size, args.window_overlap, args.flatten)

    # Concatentate into dataframe (column-wise), then write to CSV file
    output_df = pd.concat([labels, features], axis=1)
    output_file_name: str = ""

    if args.flatten:
        output_file_name = f"A0{args.subject_number}_{args.window_size}_{args.window_overlap}_flattened.parquet"

    else:
        output_file_name = f"A0{args.subject_number}_{args.window_size}_{args.window_overlap}.parquet"

    output_df.to_parquet(f"dataset_bci_iv_2a/{output_file_name}", engine="pyarrow", index=False)