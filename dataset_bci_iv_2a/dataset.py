import numpy as np
import pandas as pd
import scipy.signal as signal
import torch

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
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    
    # Apply the filter forwards and backwards to remove phase delay
    y = signal.filtfilt(b, a, data)
    return y


def filter_dataframe(df: pd.DataFrame, lowcut: float, highcut: float, fs: float, order: int = 4) -> pd.DataFrame:
    filtered_df = df.copy()
    data_columns = df.columns.drop(["Recording", "Label"])

    for recording in df["Recording"].unique():
        recording_indices = df["Recording"] == recording
        
        for column in data_columns:
            filtered_df.loc[recording_indices, column] = butter_bandpass_filter(df.loc[recording_indices, column].values, lowcut, highcut, fs, order)

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

    def __init__(self, labeled_data: pd.DataFrame) -> None:
        """
        The data parameter should be of size (M, 22) where N is the number of samples, and 22 is the number of EEG channels.
        The labels parameter should be of size N.
        """
        self.labels: pd.Series = labeled_data["Label"]
        self.data: pd.DataFrame = labeled_data.drop(columns=["Label"])

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Join the channels along the first axis to create a ndarray of shape (num_channels, window_size)
        array = np.stack(self.data.iloc[idx].values)

        # Note - for each item, labels should return an index into a list of classes.
        # In this case, the classes (in order) are: "Rest", "Left", "Right", "Feet", and "Tongue"
        # The tensor has shape (num_channels, window_size)
        return torch.tensor(array, dtype=torch.float32), int(self.labels.iloc[idx])


def create_windowed_dictionary(df: pd.DataFrame, window_size: int, window_overlap: int) -> dict[str, list[list[float]] | list[float]]:
    if window_size < 1:
        raise ValueError(f"Window size must be at least 1, but got {window_size}")

    if window_overlap >= window_size:
        raise ValueError(f"Window overlap must be less than window size, but got window size: {window_size}, window overlap: {window_overlap}")

    if len(df) < window_size:
        # This can happen if there are a few samples that were stuck between groups of samples containing NaN values.
        # If this occurs, creating a windowed dictionary won't be possible, so just return.
        return {}

    label_column: str = "Label"
    data_columns: list[str] = df.columns.drop(label_column).tolist()

    # Will have columns for each EEG channel, plus a label column
    windowed_data: dict[str, list[list[float]]] = {header: [] for header in data_columns}
    windowed_data[label_column] = []

    for window_base_index in range(0, len(df) - window_size + 1, window_size - window_overlap):
        for data_column in data_columns:
            filtered_window = butter_bandpass_filter(df[data_column].values[window_base_index:window_base_index + window_size], 8, 50, 250)

            # windowed_data[data_column].append(df[data_column].values[window_base_index:window_base_index + window_size])
            windowed_data[data_column].append(filtered_window)

        windowed_data[label_column].append(df[label_column].values[window_base_index])

    return windowed_data


class BciIvDatasetFactory:
    @staticmethod
    def create(subject_number: int, window_size: int, window_overlap: int) -> tuple[BciIvDataset, BciIvDataset]:
        """
        Returns a tuple of two BciIvDataset objects: [training, testing].
        """
        
        if subject_number < 1 or subject_number > 9:
            raise ValueError(f"Subject number must be between 1 and 9, but got {subject_number}")
        
        if window_size < 1:
            raise ValueError(f"Window size must be at least 1, but got {window_size}")

        if window_overlap >= window_size:
            raise ValueError(f"Window overlap must be less than window size, but got window size: {window_size}, window overlap: {window_overlap}")
        
        evaluation_csv_parser: BciIvCsvParser = BciIvCsvParser(f"dataset_bci_iv_2a/A0{subject_number}E.csv")
        evaluation_df: pd.DataFrame = evaluation_csv_parser.get_dataframe()
        
        # Last row of the DataFrame should contain the largest recording index
        recording_offset: int = evaluation_df.iloc[-1]["Recording"] + 1

        # Since this is a separate CSV file, the recording index starts from 0, so need to add the largest recording index from the previous CSV file.
        training_csv_parser: BciIvCsvParser = BciIvCsvParser(f"dataset_bci_iv_2a/A0{subject_number}T.csv")
        training_df: pd.DataFrame = training_csv_parser.get_dataframe()
        training_df["Recording"] += recording_offset

        # Don't care about EOG
        raw_df: pd.DataFrame = pd.concat([evaluation_df, training_df]).drop(columns=["EOGL", "EOGM", "EOGR"])

        # scalar: StandardScaler = StandardScaler()
        # features: pd.DataFrame = raw_df.drop(columns=["Label", "Recording"])
        # normalized_df: pd.DataFrame = pd.DataFrame(scalar.fit_transform(features), columns=features.columns)
        # normalized_df["Label"] = raw_df["Label"].values
        labeled_df: pd.DataFrame = raw_df.drop(columns=["Recording"])

        # The dictionary object from which a dataframe will be created.
        # Stores windows of data per EEG column: each window is a list of floating point values.
        # Each window has an associated label.
        windowed_data: dict[str, list[list[float]] | list[float]] = {header: [] for header in labeled_df.columns}
        
        # Group the dataframe by recording index - this is important, as windows must be created from sequential samples.
        df_group_iterable = labeled_df.groupby(raw_df["Recording"].values)

        # For each recording index, create a dictionary of windowed data, then append it to the larger dictionary.
        for _, df in df_group_iterable:
            windowed_dict: dict[str, list[list[float]] | list[float]] = create_windowed_dictionary(df, window_size, window_overlap)
            
            for header in windowed_dict:
                windowed_data[header].extend(windowed_dict[header])

        # Pandas supports directly creating a DataFrame from a dictionary of lists.
        windowed_df: pd.DataFrame = pd.DataFrame(windowed_data)

        train, test = train_test_split(windowed_df, test_size=0.15, random_state=42)

        return BciIvDataset(train), BciIvDataset(test)


if __name__ == "__main__":
    # Testing the dataset creation process on one CSV file:
    subject_number: int = 2

    evaluation_csv_parser: BciIvCsvParser = BciIvCsvParser(f"A0{subject_number}E.csv")
    evaluation_df: pd.DataFrame = evaluation_csv_parser.get_dataframe()

    raw_df: pd.DataFrame = evaluation_df.drop(columns=["EOGL", "EOGM", "EOGR"])
    df_group_iterator = iter(raw_df.drop(columns=["Recording"]).groupby(raw_df["Recording"].values))

    _, df = next(df_group_iterator)
    windowed_dict: dict[str, list[list[float]] | list[float]] = create_windowed_dictionary(df, 100, 95)

    windowed_df: pd.DataFrame = pd.DataFrame(windowed_dict)

    train, _ = train_test_split(windowed_df, test_size=0.15, random_state=42)

    trainset = BciIvDataset(train)

    print(f"Trainset size: {len(trainset)}")

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    trainiter = iter(trainloader)

    # Should return tensor containing batch_size samples, and a tensor containing batch_size labels
    sample, label = next(trainiter)

    print(f"Sample shape: {sample.shape}")
    print(f"Label shape: {label.shape}")

    # The above shape can be used with CNN, but LSTM expects input in the form of (batch_size, sequence_length, num_features)
    lstm_sample = sample.permute(0, 2, 1)

    print(f"LSTM sample shape: {lstm_sample.shape}")
    print(f"Label shape: {label.shape}")
