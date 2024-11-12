import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


CSV_DELIMITER: str = ","


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
            windowed_data[data_column].append(df[data_column].values[window_base_index:window_base_index + window_size])

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

        training_csv_parser: BciIvCsvParser = BciIvCsvParser(f"dataset_bci_iv_2a/A0{subject_number}T.csv")
        training_df: pd.DataFrame = training_csv_parser.get_dataframe()
        training_df["Recording"] += recording_offset

        # Don't care about EOG
        raw_df: pd.DataFrame = pd.concat([evaluation_df, training_df]).drop(columns=["EOGL", "EOGM", "EOGR"])

        scalar: StandardScaler = StandardScaler()
        features: pd.DataFrame = raw_df.drop(columns=["Label", "Recording"])
        normalized_df: pd.DataFrame = pd.DataFrame(scalar.fit_transform(features), columns=features.columns)
        normalized_df["Label"] = raw_df["Label"].values

        windowed_data: dict[str, list[list[float]] | list[float]] = {header: [] for header in normalized_df.columns}
        df_group_iterable = normalized_df.groupby(raw_df["Recording"].values)

        for _, df in df_group_iterable:
            windowed_dict: dict[str, list[list[float]] | list[float]] = create_windowed_dictionary(df, window_size, window_overlap)
            
            for header in windowed_dict:
                windowed_data[header].extend(windowed_dict[header])

        windowed_df: pd.DataFrame = pd.DataFrame(windowed_data)
        labels: pd.Series = windowed_df["Label"]
        windowed_df: pd.DataFrame = windowed_df.drop(columns=["Label"])

        X_train, X_test, y_train, y_test = train_test_split(windowed_df, labels, test_size=0.1, random_state=42)

        return BciIvDataset(X_train, y_train, window_size), BciIvDataset(X_test, y_test, window_size)
