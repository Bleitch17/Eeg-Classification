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
    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.features = features
        self.labels = labels
        
        # Convert labels to zero-based indexing if needed
        if self.labels.min() == 1:
            self.labels = self.labels - 1
            
        # Convert features to numpy arrays when initializing
        self.feature_arrays = []
        for idx in range(len(features)):
            row = features.iloc[idx]
            # Stack the channel arrays into a single (22, window_size) array
            channel_arrays = [np.array(row[col], dtype=np.float32) for col in features.columns]
            stacked_array = np.stack(channel_arrays)
            self.feature_arrays.append(stacked_array)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get pre-converted features
        features = self.feature_arrays[idx]
        
        # Reshape to (22, window_size) if needed
        if len(features.shape) == 1:
            features = features.reshape(22, -1)
        
        # Convert to torch tensors - add channel dimension to make it (1, 22, window_size)
        features = torch.FloatTensor(features).unsqueeze(0)
        label = torch.LongTensor([self.labels.iloc[idx]])[0]
        
        return features, label


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


class OnlinePreprocessing:
    def __init__(self, baseline_recordings: pd.DataFrame):
        """Initialize with baseline recordings to compute statistics"""
        # Compute baseline statistics for each channel
        self.channel_stats = {}
        for column in baseline_recordings.columns:
            if column not in ['Label', 'Recording']:
                self.channel_stats[column] = {
                    'mean': baseline_recordings[column].mean(),
                    'std': baseline_recordings[column].std()
                }
        
    def process_window(self, window_data: np.ndarray, channel: str, fs: float = 250.0) -> np.ndarray:
        """Process a single window of data as it would be in real-time"""
        # 1. Bandpass filter (8-50 Hz)
        filtered_data = butter_bandpass_filter(
            window_data, 
            lowcut=8.0, 
            highcut=50.0, 
            fs=fs
        )
        
        # 2. Normalize using pre-computed channel-specific baseline statistics
        normalized_data = (filtered_data - self.channel_stats[channel]['mean']) / self.channel_stats[channel]['std']
        
        return normalized_data


def create_windowed_dataframe(df: pd.DataFrame, window_size: int, 
                            baseline_recordings: List[int], flatten: bool = False) -> pd.DataFrame:
    # Get baseline data
    baseline_data = df[df['Recording'].isin(baseline_recordings)]
    
    # Initialize online processor with baseline data
    processor = OnlinePreprocessing(baseline_data)
    
    windowed_data = []
    label_column = "Label"
    
    # Explicitly exclude EOG channels and non-feature columns
    excluded_patterns = ['EOGL', 'EOGM', 'EOGR', 'Label', 'Recording']
    eeg_columns = [col for col in df.columns 
                  if not any(pattern in col for pattern in excluded_patterns)]
    
    print("Selected channels:", len(eeg_columns))
    print("Example channels:", eeg_columns[:5])
    
    # Process each recording separately
    for recording_id in df['Recording'].unique():
        if recording_id in baseline_recordings:
            continue  # Skip baseline recordings
            
        recording_data = df[df['Recording'] == recording_id].copy()
        
        # Skip if recording is too short
        if len(recording_data) < window_size:
            continue
            
        # Process each window in this recording
        for start_idx in range(0, len(recording_data) - window_size + 1, window_size):
            window_data = {}
            window_slice = recording_data.iloc[start_idx:start_idx + window_size]
            
            # Process each EEG channel
            for channel in eeg_columns:
                channel_data = window_slice[channel].values
                processed_window = processor.process_window(channel_data, channel)
                
                if flatten:
                    # Add each timepoint as a separate feature
                    for t in range(window_size):
                        window_data[f"{channel}_t{t}"] = float(processed_window[t])
                else:
                    # Store as numpy array directly
                    window_data[channel] = processed_window.tolist()  # Convert to list for storage
            
            # Add label
            window_data['Label'] = int(window_slice['Label'].iloc[0])
            windowed_data.append(window_data)
    
    return pd.DataFrame(windowed_data)


def data(subject_number: int, window_size: int, flatten: bool = False):
    """Load and process data for a subject"""
    # Load evaluation and training data
    evaluation_df = pd.read_csv(f"dataset_bci_iv_2a/A0{subject_number}E.csv", delimiter=CSV_DELIMITER)
    training_df = pd.read_csv(f"dataset_bci_iv_2a/A0{subject_number}T.csv", delimiter=CSV_DELIMITER)
    
    # Adjust recording numbers to ensure they're unique
    recording_offset = evaluation_df["Recording"].max() + 1
    training_df["Recording"] += recording_offset
    raw_df = pd.concat([evaluation_df, training_df])
    
    # Use first few recordings as baseline/calibration data
    baseline_recordings = list(raw_df['Recording'].unique())[:2]
    
    # Create windows with online processing simulation
    windowed_df = create_windowed_dataframe(
        raw_df, 
        window_size=window_size,
        baseline_recordings=baseline_recordings,
        flatten=flatten
    )
    
    # Split features and labels
    labels = windowed_df['Label']
    features = windowed_df.drop(columns=['Label'])
    
    return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG data for BCI Competition IV.")
    parser.add_argument("subject_number", type=int, choices=range(1, 10), 
                       help="Subject number (1-9)")
    parser.add_argument("window_size", type=int, help="Size of the window")
    parser.add_argument("--flatten", action="store_true", help="Flatten the windowed data")
    
    args = parser.parse_args()
    
    # Removed window_overlap parameter since we're not using it
    features, labels = data(args.subject_number, args.window_size, args.flatten)

    # Concatentate into dataframe (column-wise), then write to CSV file
    output_df = pd.concat([labels, features], axis=1)
    output_file_name: str = ""

    if args.flatten:
        output_file_name = f"A0{args.subject_number}_{args.window_size}_flattened.parquet"

    else:
        output_file_name = f"A0{args.subject_number}_{args.window_size}.parquet"

    output_df.to_parquet(f"dataset_bci_iv_2a/{output_file_name}", engine="pyarrow", index=False)
