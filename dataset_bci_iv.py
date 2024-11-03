import pandas as pd

from typing import TextIO


CSV_DELIMITER: str = ","
EVENT_DELIMITER: str = " "

COL_EVENT_TYPES: str = "Event Types"
COL_EVENT_DURATIONS: str = "Event Durations"

# Name of the column in the label DataFrame
LABEL_COL_NAME: str = "Label"

# 22 EEG columns, 3 EOG columns, and 1 artifact status column
NUM_DATA_COLUMNS: int = 26

SAMPLE_RATE_HZ: int = 250


class BciIvCsvParser:
    def __init__(self, csv_file_path: str) -> None:
        # Store a pointer to the CSV file
        self.csv_file: TextIO = open(csv_file_path, "r")

        # Expect first row to be the header
        self.csv_headers: list[str] = self.csv_file.readline().strip().split(CSV_DELIMITER)
        self.csv_header_index: dict[str, int] = {self.csv_headers[i]: i for i in range(len(self.csv_headers))}
        
        # Column headers for the data container - NUM_DATA_COLUMNS data columns, and 1 label column
        self.data_headers: list[str] = self.csv_headers[:NUM_DATA_COLUMNS] + [LABEL_COL_NAME]

        self.data: dict[str, list[float]] = {header: [] for header in self.data_headers}

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
    
    def read_rows(self, n_rows: int, label: int) -> None:
        for _ in range(n_rows):
            line: str = self.csv_file.readline()
            
            if not line or line == "\n":
                # End of file reached, sould be no blank lines except for the last one
                return

            line_segments: list[str] = line.strip().split(CSV_DELIMITER)
            data_segments: list[float] = list(map(float, line_segments[:NUM_DATA_COLUMNS])) + [float(label)]

            if len(data_segments) != len(self.data_headers):
                raise(ValueError(f"Expected {len(self.data_headers)} data columns, got {len(data_segments)}"))

            for measurement, header in zip(data_segments, self.data_headers):
                self.data[header].append(measurement)

    def get_dataframe(self) -> pd.DataFrame:
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

            # Expect the relevant event type to be the last one in the list
            class_label: int = self.get_class_label(event_types[-1])
            
            # Duration represents how many more rows to read
            duration: int = event_durations[-1]

            if class_label == 0:
                # Resting state EEG data - can just read rows normally
                # Also need to capture the current row
                data_segments: list[float] = list(map(float, line_segments[:NUM_DATA_COLUMNS])) + [float(class_label)]
                
                if len(data_segments) != len(self.data_headers):
                    raise(ValueError(f"Expected {len(self.data_headers)} data columns, got {len(data_segments)}"))

                for measurement, header in zip(data_segments, self.data_headers):
                    self.data[header].append(measurement)
                
                # Read the remaining rows into the data container
                self.read_rows(duration, class_label)

            elif class_label > 0:
                # Found a cue onset, so current row is t = 2s
                # Skip 249 rows, so next read will be t = 3s
                self.skip_rows(SAMPLE_RATE_HZ - 1)
                self.read_rows(SAMPLE_RATE_HZ * 3, class_label)

        return pd.DataFrame(self.data)
