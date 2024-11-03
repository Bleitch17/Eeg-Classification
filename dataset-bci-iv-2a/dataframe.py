import pandas as pd


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
    "EEG_22": "POz"
}

CSV_DELIMITER: str = ","

COL_EVENT_276: str = "Event_276" # Idling EEG (eyes open)
COL_EVENT_768: str = "Event_768" # Start of a trial
COL_EVENT_769: str = "Event_769" # Cue onset left
COL_EVENT_770: str = "Event_770" # Cue onset right
COL_EVENT_771: str = "Event_771" # Cue onset foot
COL_EVENT_772: str = "Event_772" # Cue onset tongue

# Name of the column in the label DataFrame
LABEL: str = "Label"
NUM_EEG_CHANNELS: int = 22
TRIAL_SAMPLES_SKIP: int = 750
TRIAL_SAMPLES_CAPTURE: int = 750


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

    preprocessed_row_data: list[dict[str, float]] = []
    data_labels: dict[str, list[int]] = {
        LABEL: []
    }

    with open(csv_file_path, "r") as csv_file:
        # First row will be the column headers:
        column_headers: list[str] = csv_file.readline().strip().split(CSV_DELIMITER)
        column_header_index: dict[str, int] = {column_headers[i]: i for i in range(len(column_headers))}

        while line := csv_file.readline():
            full_row_data: list[float] = list(map(float, line.strip().split(CSV_DELIMITER)))

            # Should be safe to capture all the eyes open EEG rows
            if full_row_data[column_header_index[COL_EVENT_276]]:
                preprocessed_row_data.append(
                    {COL_ELECTRODE_MAP[column_headers[i]]: full_row_data[i] for i in range(NUM_EEG_CHANNELS)}
                )
                data_labels[LABEL].append(0)
            
            # Current row contains start of the trial - advance 749 samples, then start capturing the row data
            elif full_row_data[column_header_index[COL_EVENT_768]]:
                for _ in range(TRIAL_SAMPLES_SKIP - 1):
                    csv_file.readline()
                
                sample_class: int = 0
                full_row_data = list(map(float, csv_file.readline().strip().split(CSV_DELIMITER)))

                if full_row_data[column_header_index[COL_EVENT_769]]:
                    sample_class = 1
                
                elif full_row_data[column_header_index[COL_EVENT_770]]:
                    sample_class = 2
                
                elif full_row_data[column_header_index[COL_EVENT_771]]:
                    sample_class = 3

                elif full_row_data[column_header_index[COL_EVENT_772]]:
                    sample_class = 4

                else:
                    print(f"ERROR: Expected MI event, but found none. Row: {full_row_data}")
                    raise ValueError("MI event not found")

                data_labels[LABEL].append(sample_class)

                preprocessed_row_data.append(
                    {COL_ELECTRODE_MAP[column_headers[i]]: full_row_data[i] for i in range(NUM_EEG_CHANNELS)}
                )

                for _ in range(TRIAL_SAMPLES_CAPTURE - 1):
                    full_row_data = list(map(float, csv_file.readline().strip().split(CSV_DELIMITER)))

                    preprocessed_row_data.append(
                        {COL_ELECTRODE_MAP[column_headers[i]]: full_row_data[i] for i in range(NUM_EEG_CHANNELS)}
                    )

                    data_labels[LABEL].append(sample_class)
    
    return pd.DataFrame(preprocessed_row_data), pd.DataFrame(data_labels)