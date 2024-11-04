import pandas as pd

from dataset_bci_iv import BciIvCsvParser
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from typing import Callable

# Type aliases
AccuracyScore = float
ConfusionMatrix = list[list[int]]
ClassificationReport = str

# Other constants
BCI_COMP_DATASET_PATH: str = "dataset-bci-iv-2a/"

# Used as seed for test_train_split
RANDOM_STATE: int = 17 
TEST_SIZE: float = 0.2


def train_naive_bayes(X: pd.DataFrame, y: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    """
    Train a Naive Bayes classifier on the given data and labels.

    :param X: The data to train on.
    :param y: The labels to train on.

    :return: A tuple containing the accuracy, confusion matrix and classification report.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    gnb_accuracy = accuracy_score(y_test, y_pred)
    gnb_confusion_matrix = confusion_matrix(y_test, y_pred)
    gnb_classification_report = classification_report(y_test, y_pred)
    
    return gnb_accuracy, gnb_confusion_matrix, gnb_classification_report


def experiment_initial(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Remove the resting state
    df = df[df["Label"] != 0]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Separate data from labels
    X = df[["Fz", "C3", "Cz", "C4", "Pz"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_5_channel_no_rest(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Remove the resting state
    df = df[df["Label"] != 0]

    # Separate data from labels
    X = df[["Fz", "C3", "Cz", "C4", "Pz"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_left_right_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_left_right_hand_5_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df[["Fz", "C3", "Cz", "C4", "Pz"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract left and right hand data
    criterion = df["Label"].map(lambda x: x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df[["C3", "C4"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df[["C3", "C4"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_c4(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df[["C4"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_hand_c3(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:
    
    # Extract resting state and left hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1)
    df = df[criterion]

    # Separate data from labels
    X = df[["C3"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


def experiment_resting_vs_left_right_hand_2_channel(df: pd.DataFrame) -> tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]:

    # Extract resting state and left/right hand data
    criterion = df["Label"].map(lambda x: x == 0 or x == 1 or x == 2)
    df = df[criterion]

    # Separate data from labels
    X = df[["C3", "C4"]]
    y = df["Label"]

    return train_naive_bayes(X, y)


if __name__ == "__main__":
    bci_iv_parser_session_train: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01T.csv")
    bci_iv_parser_session_eval: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01E.csv")    

    df_train: pd.DataFrame = bci_iv_parser_session_train.get_dataframe()
    df_eval: pd.DataFrame = bci_iv_parser_session_eval.get_dataframe()

    df: pd.DataFrame = pd.concat([df_train, df_eval])

    print(f"Dataframe shape:\n{df.shape}")
    print(f"First few rows:\n{df.head()}")
    print(f"Artifact rows:\n{df[df["Artifact"] == 1.0].shape[0]}")

    # Not dealing with EOG data for now
    df = df.drop(columns=["EOGL", "EOGM", "EOGR"])

    # Removing rejected trials
    df = df[df["Artifact"] != 1.0]
    df = df.drop(columns=["Artifact"])

    # There may still be some NaN values from the resting state recording, so remove those rows too.
    df = df.dropna()

    experiments: list[tuple[str, Callable[[pd.DataFrame], tuple[AccuracyScore, ConfusionMatrix, ClassificationReport]]]] = [
        # ("Initial", experiment_initial),
        # ("No Rest", experiment_no_rest),
        # ("5 Channel", experiment_5_channel),
        # ("5 Channel No Rest", experiment_5_channel_no_rest),
        ("Left Right Hand", experiment_left_right_hand),
        ("Left Right Hand 5 Channel", experiment_left_right_hand_5_channel),
        ("Left Right Hand 2 Channel", experiment_left_right_hand_2_channel),
        ("Resting vs Left Hand", experiment_resting_vs_left_hand),
        ("Resting vs Left Hand 2 Channel", experiment_resting_vs_left_hand_2_channel),
        ("Resting vs Left Hand C4", experiment_resting_vs_left_hand_c4),
        ("Resting vs Left Hand C3", experiment_resting_vs_left_hand_c3),
        ("Resting vs Left Right Hand 2 Channel", experiment_resting_vs_left_right_hand_2_channel)
    ]

    for experiment_name, experiment_function in experiments:
        print(f"Running experiment: {experiment_name}")
        accuracy, _, _ = experiment_function(df)
        print(f"Accuracy: {accuracy}")
