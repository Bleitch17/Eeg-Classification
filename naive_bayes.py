import dataset_bci_comp

import pandas as pd


BCI_COMP_DATASET_PATH: str = "dataset-bci-iv-2a/"


if __name__ == "__main__":
    X, y = dataset_bci_comp.get_dataframe(f"{BCI_COMP_DATASET_PATH}A01T.csv")

    print(X.head())
    print(y.head())

    print(X.shape)

    print(X.iloc[99999])
    print(y.iloc[99999])

    # TODO - finish this