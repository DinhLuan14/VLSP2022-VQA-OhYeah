# src/dataset/preprocess.py

import pandas as pd
from datasets import Dataset, Image


def df_to_dataset(df):
    """Convert dataframe to Image Dataset."""

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.remove_columns(["filename"])

    return dataset
