import pandas as pd

from dataset_bci_iv_2a.dataset import BciIvCsvParser, filter_dataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Callable


# Type aliases
AccuracyScore = float
ConfusionMatrix = list[list[int]]
ClassificationReport = str

# Other constants
BCI_COMP_DATASET_PATH: str = "dataset_bci_iv_2a/"

# Used as seed for test_train_split
RANDOM_STATE: int = 17 
TEST_SIZE: float = 0.2


def train_random_forest(X: pd.DataFrame, y: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train a Random Forest classifier on the given data and labels.

    :param X: The data to train on.
    :param y: The labels to train on.

    :return: A tuple containing the accuracy, confusion matrix and classification report.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    rfc_accuracy = accuracy_score(y_test, y_pred)
    rfc_confusion_matrix = confusion_matrix(y_test, y_pred)
    rfc_classification_report = classification_report(y_test, y_pred)
    
    return rfc_accuracy, rfc_confusion_matrix, rfc_classification_report


def experiment_5_state_mi_classification(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_random_forest(X, y)


def experiment_5_state_mi_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    # Separate data from labels
    X = df[["Fz", "C3", "Cz", "C4", "Pz"]]
    y = df["Label"]

    return train_random_forest(X, y)


def experiment_left_right_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_random_forest(X, y)


def experiment_resting_vs_left_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_random_forest(X, y)


def experiment_resting_vs_all_single_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    df = df.copy()

    # Re-label all rows with labels greater than or equal to 1 to 1
    df.loc[df["Label"] >= 1, "Label"] = 1

    # Separate data from labels
    X = df[["C3"]]
    y = df["Label"]

    return train_random_forest(X, y)


if __name__ == "__main__":
    bci_iv_parser_session_train: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01T.csv")
    bci_iv_parser_session_eval: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01E.csv")

    df_train: pd.DataFrame = bci_iv_parser_session_train.get_dataframe()
    df_eval: pd.DataFrame = bci_iv_parser_session_eval.get_dataframe()

    raw_df: pd.DataFrame = pd.concat([df_train, df_eval])

    print(f"DataFrame shape:\n{raw_df.shape}")
    print(f"First few rows:\n{raw_df.head()}")

    # Not dealing with EOG data for now
    raw_df = raw_df.drop(columns=["EOGL", "EOGM", "EOGR"])

    # Normalize all columns except the Label column
    # scaler = StandardScaler()
    # features = raw_df.drop(columns=["Label", "Recording"])
    # scaled_features = scaler.fit_transform(features)
    # scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    # scaled_df["Label"] = raw_df["Label"].values
    # scaled_df["Recording"] = raw_df["Recording"].values

    filtered_df = filter_dataframe(raw_df, 8.0, 30.0, 250.0).drop(columns=["Recording"])

    experiments: list[tuple[str, Callable[[pd.DataFrame], tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]]]] = [
        # ("5 State MI Classification", experiment_5_state_mi_classification),
        # ("5 State MI Classification, 5 Channel", experiment_5_state_mi_5_channel),
        # ("Left vs Right Hand", experiment_left_right_hand)
        # ("Resting vs Left Hand", experiment_resting_vs_left_hand)
        ("Resting vs All Single Channel", experiment_resting_vs_all_single_channel)
    ]

    for experiment_name, experiment_function in experiments:
        print(f"Running experiment: {experiment_name}")
        accuracy, _, _ = experiment_function(filtered_df)
        print(f"Accuracy: {accuracy}")
