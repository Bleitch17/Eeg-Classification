import pandas as pd

from dataset_bci_iv import BciIvCsvParser


BCI_COMP_DATASET_PATH: str = "dataset-bci-iv-2a/"


if __name__ == "__main__":
    bci_iv_parser: BciIvCsvParser = BciIvCsvParser(f"{BCI_COMP_DATASET_PATH}A01E.csv")
    
    df: pd.DataFrame = bci_iv_parser.get_dataframe()

    print(df.shape)
    print(df.head())

    # TODO - finish this