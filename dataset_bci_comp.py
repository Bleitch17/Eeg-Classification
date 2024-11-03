import pandas as pd

from typing import TextIO


# Map the column names from the .csv files to the electrode positions in the 10-10 system:
# https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)#/media/File:EEG_10-10_system_with_additional_information.svg
COL_ELECTRODE_MAP: dict[str, str] = {
    "EEG_1": "Fz",
    "EEG_2": "FC3",
    "EEG_3": "FC1",
    "EEG_4": "FCz",
    "EEG_5": "FC2",
    "EEG_6": "FC4",
    "EEG_7": "C5",
    "EEG_8": "C3",
    "EEG_9": "C1",
    "EEG_10": "Cz",
    "EEG_11": "C2",
    "EEG_12": "C4",
    "EEG_13": "C6",
    "EEG_14": "CP3",
    "EEG_15": "CP1",
    "EEG_16": "CPz",
    "EEG_17": "CP2",
    "EEG_18": "CP4",
    "EEG_19": "P1",
    "EEG_20": "Pz",
    "EEG_21": "P2",
    "EEG_22": "POz",
    "EOG_1": "EOG_L",
    "EOG_2": "EOG_M",
    "EOG_3": "EOG_R"
}

CSV_DELIMITER: str = ","
EVENT_DELIMITER: str = " "

COL_EVENT_TYPES: str = "Event Types"
COL_EVENT_DURATIONS: str = "Event Durations"

# Name of the column in the label DataFrame
LABEL: str = "Label"

NUM_DATA_COLUMNS: int = 25

CUE_ONSET_SKIP_BOUND: int = 249
MI_CAPTURE_BOUND: int = 750


def parse_rows(csv_file: TextIO, num_iterations: int, class_label, header_index: dict[str, int]) -> tuple[list[dict[str, float]], dict[str, list[int]]]:
    """
    Reads a number of rows from the csv file, and assigns the given class label to each row.

    :param csv_file: The csv file pointer to read from.
    :param num_iterations: The number of rows to read, i.e.: the number of times read() will be called.
    :param class_label: The class label to assign to each row that is read.
    :param header_index: The column header index dictionary.

    :return: A tuple with a list of dictionaries with the EEG data, and a dictionary with the row labels.
    """

    row_data: list[dict[str, float]] = []
    row_labels: dict[str, list[int]] = { LABEL: [] }

    for _ in range(num_iterations):
        line_segments: list[str] = csv_file.readline().strip().split(CSV_DELIMITER)

        row_data.append(
            {COL_ELECTRODE_MAP[header]: float(line_segments[header_index[header]]) for header in header_index.keys()}
        )

        row_labels[LABEL].append(class_label)
        
    return row_data, row_labels


def get_class_label(event_type: int) -> int:
    """
    Get the class label for the current trial row.

    :param full_row_data: The row data.
    :param column_index: The column index dictionary.
    :return: The class label.
    """

    # Event types from dataset description:
    # https://www.bbci.de/competition/iv/desc_2a.pdf
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


def get_dataframe(csv_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data in the csv file path, and return a dataframe with 23 columns.
    The first 22 columns are the EEG data from the 22 electrodes, and the last column is the event class,
    0: rest,
    1: left hand,
    2: right hand,
    3: feet,
    4: tongue.

    :param csv_file_path: The path to the csv file. Acceptable paths are: A0[1-9][ET].csv
    :return: A DataFrame tuple (X, y) where X is the EEG data and y is the event class.
    """

    row_data: list[dict[str, float]] = []
    row_labels: dict[str, list[int]] = { LABEL: [] }

    # TODO - clean this up, if needed

    with open(csv_file_path, "r") as csv_file:
        # First row will be the column headers:
        column_headers: list[str] = csv_file.readline().strip().split(CSV_DELIMITER)
        full_header_index: dict[str, int] = {column_headers[i]: i for i in range(len(column_headers))}
        data_header_index: dict[str, int] = {header: full_header_index[header] for header in COL_ELECTRODE_MAP.keys()}

        while line := csv_file.readline():
            row_segments = line.strip().split(CSV_DELIMITER)

            event_types_string: str = row_segments[full_header_index[COL_EVENT_TYPES]]
            
            if not event_types_string:
                continue
            
            event_types: list[int] = list(map(int, event_types_string.split(EVENT_DELIMITER)))
            event_durations: list[int] = list(map(int, row_segments[full_header_index[COL_EVENT_DURATIONS]].split(EVENT_DELIMITER)))

            # Expect the relevant event type to be the last one in the list
            class_label: int = get_class_label(event_types[-1])
            duration: int = event_durations[-1]

            if class_label == 0:
                # Resting state EEG data - can just read rows normally
                # Also need to read the current row
                row_data.append(
                    {COL_ELECTRODE_MAP[column_headers[i]]: float(row_segments[i]) for i in range(len(COL_ELECTRODE_MAP.keys()))}
                )

                row_labels[LABEL].append(class_label)

                data_subset, labels_subset = parse_rows(csv_file, duration, class_label, data_header_index)

                row_data.extend(data_subset)
                row_labels[LABEL].extend(labels_subset[LABEL])

            elif class_label > 0:
                # Found a cue onset, so current row is t = 2s
                # Skip so next read will be t = 3s
                for _ in range(CUE_ONSET_SKIP_BOUND):
                    csv_file.readline()
                
                data_subset, labels_subset = parse_rows(csv_file, MI_CAPTURE_BOUND, class_label, data_header_index)

                row_data.extend(data_subset)
                row_labels[LABEL].extend(labels_subset[LABEL])

    return pd.DataFrame(row_data), pd.DataFrame(row_labels)