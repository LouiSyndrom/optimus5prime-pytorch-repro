from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.config import (
    BASE_ORDER,
    EXPECTED_ROWS,
    FRACTION_COLUMNS,
    HIGH_READ_SIZE,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)


def load_data(path):
    path = Path(path)
    columns = ["utr", "rl", "total_reads"] + FRACTION_COLUMNS
    data = pd.read_csv(path, usecols=columns)

    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(data)}")

    valid_sequences = data["utr"].str.fullmatch(f"[{BASE_ORDER}]{{{SEQUENCE_LENGTH}}}", na=False)
    if not valid_sequences.all():
        raise ValueError("The dataset contains invalid 50-nucleotide sequences")

    numeric_columns = ["rl", "total_reads"] + FRACTION_COLUMNS
    numeric_values = data[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("The dataset contains missing or non-finite numeric values")

    fractions = data[FRACTION_COLUMNS].to_numpy(dtype=np.float64)
    if (fractions < 0).any() or not np.allclose(fractions.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Auxiliary fractions must be non-negative and sum to one for KL divergence")

    data["source_row"] = np.arange(len(data))
    return data


def create_data_splits(data, split_path):
    data = data.sort_values("total_reads", ascending=False, kind="stable").reset_index(drop=True)

    middle_end = HIGH_READ_SIZE + TRAIN_SIZE + VALIDATION_SIZE
    rng = np.random.default_rng(SEED)
    middle_indexes = rng.permutation(np.arange(HIGH_READ_SIZE, middle_end))
    training_indexes = middle_indexes[:TRAIN_SIZE]
    validation_indexes = middle_indexes[TRAIN_SIZE:]
    high_read_indexes = np.arange(HIGH_READ_SIZE)
    low_read_indexes = np.arange(middle_end, len(data))

    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        training=training_indexes,
        validation=validation_indexes,
        high_read_test=high_read_indexes,
        low_read_test=low_read_indexes,
    )

    return {
        "training": data.iloc[training_indexes].reset_index(drop=True),
        "validation": data.iloc[validation_indexes].reset_index(drop=True),
        "high_read_test": data.iloc[high_read_indexes].reset_index(drop=True),
        "low_read_test": data.iloc[low_read_indexes].reset_index(drop=True),
    }


def one_hot_encode(sequences):
    encoded = np.zeros((len(sequences), 4, SEQUENCE_LENGTH), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, nucleotide in enumerate(sequence):
            encoded[row, BASE_ORDER.index(nucleotide), position] = 1.0
    return encoded


def make_baseline_data(frame):
    features = one_hot_encode(frame["utr"].tolist())
    features = features.reshape(len(frame), 4 * SEQUENCE_LENGTH)
    labels = frame["rl"].to_numpy(dtype=np.float32)
    return features, labels


def make_cnn_dataset(frame, mrl_mean, mrl_std):
    sequences = one_hot_encode(frame["utr"].tolist())
    mrls = frame["rl"].to_numpy(dtype=np.float32)
    mrls = (mrls - mrl_mean) / mrl_std
    fractions = frame[FRACTION_COLUMNS].to_numpy(dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(sequences),
        torch.from_numpy(mrls),
        torch.from_numpy(fractions),
    )
