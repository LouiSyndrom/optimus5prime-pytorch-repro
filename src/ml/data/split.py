"""Random 80/10/10 splitting, using the same method as the CNN."""

import random

import pandas as pd


def create_splits(
    df: pd.DataFrame, *, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation and test DataFrames from cleaned data."""
    data = df.sort_values("source_row").reset_index(drop=True)
    num_seq = len(data)
    index_list = list(range(num_seq))

    random_num_gen = random.Random(seed)
    training_indexes = []
    for _ in range(int(num_seq * 0.8)):
        random_num = random_num_gen.randint(0, len(index_list) - 1)
        training_indexes.append(index_list.pop(random_num))

    random_num_gen = random.Random(seed)
    validating_indexes = []
    for _ in range(int(num_seq * 0.1)):
        random_num = random_num_gen.randint(0, len(index_list) - 1)
        validating_indexes.append(index_list.pop(random_num))

    training = data.iloc[training_indexes].reset_index(drop=True)
    validating = data.iloc[validating_indexes].reset_index(drop=True)
    testing = data.iloc[index_list].reset_index(drop=True)

    return training, validating, testing
