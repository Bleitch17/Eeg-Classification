import pandas as pd

from dataset_bci_iv import BciIvCsvParser
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


BCI_COMP_DATASET_PATH: str = "dataset-bci-iv-2a/"


if __name__ == "__main__":
    bci_iv_parser: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01E.csv")
    
    df: pd.DataFrame = bci_iv_parser.get_dataframe()

    # Not dealing with EOG data for now
    df = df.drop(columns=["EOGL", "EOGM", "EOGR"])

    print(df.shape)
    print(df.head())

    artifact_count = df[df["Artifact"] == 1.0].shape[0]
    print(f"Number of rows with Artifact = 1.0: {artifact_count}")

    # Removing artifacts
    df = df[df["Artifact"] != 1.0]
    df = df.drop(columns=["Artifact"])

    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
