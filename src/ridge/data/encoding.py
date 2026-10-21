"""One-Hot encoding for cleaned DNA sequences."""

from collections.abc import Sequence, Sized
from numbers import Integral

import numpy as np

_BASE_TO_CHANNEL = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(
    sequences: Sequence[str],
    *,
    sequence_length: int = 50,
) -> np.ndarray:
    """Return float32 (N, sequence_length, 4) in A/C/G/T order.

    Preserve sample order; empty input returns (0, sequence_length, 4).
    Invalid samples raise ValueError with their zero-based sample index.
    """
    if (
        not isinstance(sequence_length, Integral)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sized):
        raise ValueError("sequences must be a sized collection of strings")

    encoded = np.zeros((len(sequences), sequence_length, 4), dtype=np.float32)
    positions = np.arange(sequence_length)
    for sample_index, sequence in enumerate(sequences):
        if (
            not isinstance(sequence, str)
            or len(sequence) != sequence_length
            or not set(sequence).issubset(_BASE_TO_CHANNEL)
        ):
            raise ValueError(
                f"Sample {sample_index}: expected a string of length "
                f"{sequence_length} containing only A/C/G/T"
            )
        channels = np.fromiter(
            (_BASE_TO_CHANNEL[base] for base in sequence),
            dtype=np.intp,
            count=sequence_length,
        )
        encoded[sample_index, positions, channels] = 1.0
    return encoded
