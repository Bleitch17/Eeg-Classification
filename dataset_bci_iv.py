import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import TextIO


CSV_DELIMITER: str = ","
EVENT_DELIMITER: str = " "

COL_EVENT_TYPES: str = "Event Types"
COL_EVENT_DURATIONS: str = "Event Durations"

COL_INDEX_EVENT_TYPES: int = -2
COL_INDEX_EVENT_DURATION: int = -1

# Name of the column in the label DataFrame
LABEL_COL_NAME: str = "Label"
ARTIFACT_COL_NAME: str = "Artifact"

# 22 EEG columns, 3 EOG columns
NUM_DATA_COLUMNS: int = 25

SAMPLE_RATE_HZ: int = 250


class BciIvCsvParserII:
    def __init__(self, csv_file_path: str) -> None:
        # Data will be internally represented as a dictionary of lists,
        # for convenient conversion to a pandas DataFrame
        self.data: dict[str, list[float]] = {}
        self.headers: list[str] = []
        
        # Maintain information about consecutive samples for each class label.
        # This will be used to create windowed data.
        self.metadata: dict[str, list[int]] = {
            "NumSamples": [],
            "ClassLabel": [],
        }

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
                
                line_segments: list[str] = line.strip().split(CSV_DELIMITER)
                event_types_string: str = line_segments[COL_INDEX_EVENT_TYPES]





class BciIvCsvParser:
    def __init__(self, csv_file_path: str) -> None:
        # Store a pointer to the CSV file
        self.csv_file: TextIO = open(csv_file_path, "r")

        # Expect first row to be the header
        self.csv_headers: list[str] = self.csv_file.readline().strip().split(CSV_DELIMITER)
        self.csv_header_index: dict[str, int] = {self.csv_headers[i]: i for i in range(len(self.csv_headers))}
        
        # Column headers for the data container - NUM_DATA_COLUMNS data columns, 1 label column and 1 artifact status column
        self.data_headers: list[str] = self.csv_headers[:NUM_DATA_COLUMNS] + [LABEL_COL_NAME, ARTIFACT_COL_NAME]

        self.data: dict[str, list[float]] = {header: [] for header in self.data_headers}

        # List with tuples of the form (class_label, number_of_samples, artifact_status). Will be used for creating windowed data.
        self.durations: list[tuple[int, int, int]] = []

    def __del__(self):
        self.csv_file.close()

    def get_class_label(self, event_type: int) -> int:
        """
        Return a class label given an event type.

        :param event_type: The event type, as described in the dataset description: https://www.bbci.de/competition/iv/desc_2a.pdf
        :return: The class label, if applicable: otherwise, -1.
        """
        match event_type:
            case 276:
                return 0
            case 769:
                return 1
            case 770:
                return 2
            case 771:
                return 3
            case 772:
                return 4
            case _:
                return -1

    def skip_rows(self, n_rows: int) -> None:
        for _ in range(n_rows):
            self.csv_file.readline()
    
    def read_rows(self, n_rows: int, artifact_status: int, label: int) -> None:
        if artifact_status not in {0, 1}:
            raise ValueError("Artifact status must be 0 or 1")
        
        if label not in {0, 1, 2, 3, 4}:
            raise ValueError("Label must be 0, 1, 2, 3, or 4")

        for _ in range(n_rows):
            line: str = self.csv_file.readline()
            
            if not line or line == "\n":
                # End of file reached, sould be no blank lines except for the last one
                return

            line_segments: list[str] = line.strip().split(CSV_DELIMITER)
            data_segments: list[float] = list(map(float, line_segments[:NUM_DATA_COLUMNS])) + [float(label), float(artifact_status)]

            # If there are NaN values in the data, set the artifact status to 1
            if np.any(np.isnan(data_segments)):
                data_segments[-1] = 1.0

            if len(data_segments) != len(self.data_headers):
                raise(ValueError(f"Expected {len(self.data_headers)} data columns, got {len(data_segments)}"))

            for measurement, header in zip(data_segments, self.data_headers):
                self.data[header].append(measurement)

    def parse(self) -> None:
        current_artifact_status: int = 0
        
        while line := self.csv_file.readline():
            if line == "\n":
                # Hit last line(s) of the file
                break

            line_segments: list[str] = line.strip().split(CSV_DELIMITER)
            event_types_string: str = line_segments[self.csv_header_index[COL_EVENT_TYPES]]
            
            if not event_types_string:
                continue
            
            event_types: list[int] = list(map(int, event_types_string.split(EVENT_DELIMITER)))
            event_durations: list[int] = list(map(int, line_segments[self.csv_header_index[COL_EVENT_DURATIONS]].split(EVENT_DELIMITER)))

            if len(event_types) != len(event_durations):
                raise ValueError("Event types and durations must have the same length")

            # checking if the current trial was a rejected one
            if 1023 in event_types:
                current_artifact_status = 1
                continue

            # Expect the relevant event type to be the last one in the list
            class_label: int = self.get_class_label(event_types[-1])
            
            # Duration represents how many more rows to read
            duration: int = event_durations[-1]

            if class_label == 0:
                # Resting state EEG data - can just read rows normally
                # Also need to capture the current row, which should not have artifacts
                data_segments: list[float] = list(map(float, line_segments[:NUM_DATA_COLUMNS])) + [float(class_label), 0.0]
                
                # If there are NaN values in the data, set the artifact status to 1
                if np.any(np.isnan(data_segments)):
                    data_segments[-1] = 1.0

                if len(data_segments) != len(self.data_headers):
                    raise(ValueError(f"Expected {len(self.data_headers)} data columns, got {len(data_segments)}"))

                for measurement, header in zip(data_segments, self.data_headers):
                    self.data[header].append(measurement)
                
                # Record number of samples for the current class label
                # Note that since durations is how many more samples to read, the number of samples here will be durations + 1 to
                # include the current row.
                # Also note that these rows are not part of the motor imagery tasks, so the artifact status is always 0.
                # However, they may still contain NaN values.
                self.durations.append((class_label, duration + 1, 0))

                # Read the remaining rows into the data container
                self.read_rows(duration, 0, class_label)

            elif class_label > 0:
                # Found a cue onset, so current row is t = 2s
                # Skip 249 rows, so next read will be t = 3s
                self.skip_rows(SAMPLE_RATE_HZ - 1)
                self.read_rows(SAMPLE_RATE_HZ * 3, current_artifact_status, class_label)

                # Record number of samples for the current class label - will always be SAMPLE_RATE_HZ * 3
                self.durations.append((class_label, SAMPLE_RATE_HZ * 3, current_artifact_status))

                # Flip the artifact status back to 0, if the current trial had an artifact
                if current_artifact_status:
                    current_artifact_status = 0
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Expects parse() to have been called first. Otherwise, returns an empty DataFrame.
        """
        
        return pd.DataFrame(self.data)

    
    def get_windowed_dataframe(self, window_size: int, window_overlap: int) -> pd.DataFrame:
        if window_size < 1:
            raise ValueError("Window size must be at least 1")

        if window_overlap < 0:
            raise ValueError("Window overlap may not be negative")

        if window_overlap >= window_size:
            raise ValueError("Window overlap must be less than window size")

        if window_size == 1:
            return pd.DataFrame(self.data)
        
        windowed_data_headers: list[str] = self.data_headers[:]
        windowed_data: dict[str, list[list[float]] | list[float]] = {header: [] for header in windowed_data_headers}

        # NOTE: the windows must capture EEG data that was recorded sequentially. There is no overlap in the data
        # recording between the resting state and the motor imagery tasks. There is also no overlap between the motor imagery
        # tasks in different trials, since the measurements were taken from the t = 3s to t = 6s periods of each trial.

        row_index: int = 0

        # Loop over each collection of consecutive samples
        for class_label, num_samples, artifact_status in self.durations:
            
            # Loop over each window in the collection of consecutive samples
            for window_base_index in range(0, num_samples - window_size + 1, window_size - window_overlap):

                # For each window, start the list with an empty list for each data header
                for header in self.data_headers[:NUM_DATA_COLUMNS]:
                    windowed_data[header].append([])

                insert_nan_artifact: bool = False

                # Loop over each sample in the window
                for header in self.data_headers[:NUM_DATA_COLUMNS]:
                    window_data: list[float] = self.data[header][row_index + window_base_index:row_index + window_base_index + window_size]
                    
                    if np.any(np.isnan(window_data)):
                        insert_nan_artifact = True

                    windowed_data[header][-1].append(window_data)

                # Each window should have a single class label and single artifact status
                windowed_data[LABEL_COL_NAME].append(class_label)
                windowed_data[ARTIFACT_COL_NAME].append(artifact_status)

                if insert_nan_artifact:
                    windowed_data[ARTIFACT_COL_NAME][-1] = 1
                    insert_nan_artifact = False

            row_index += num_samples
    
        return pd.DataFrame(windowed_data)


class BciIvDataset(Dataset):
    """
    Provides an interface for retrieving samples from the BCI Competition IV dataset,
    for use with the PyTorch package.
    """

    def __init__(self, subject_number: int, window_size: int, window_overlap: int) -> None:
        if subject_number < 1 or subject_number > 9:
            raise ValueError(f"Subject number must be between 1 and 9, but got {subject_number}")
        
        if window_size < 1:
            raise ValueError(f"Window size must be at least 1, but got {window_size}")

        if window_overlap >= window_size:
            raise ValueError(f"Window overlap must be less than window size, but got window size: {window_size}, window overlap: {window_overlap}")

        self.subject_number: int = subject_number
        self.window_size: int = window_size

        # Load both CSV files for a particular subject into dataframes, remove artifact rows, concatenate them, and create a windowed dataframe