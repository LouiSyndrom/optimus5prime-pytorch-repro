"""One-Hot encoding for cleaned DNA sequences."""

from collections.abc import Sequence

import numpy as np

_BASE_TO_CHANNEL = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(
    sequences: Sequence[str],
    *,
    sequence_length: int = 50,
) -> np.ndarray:
    """Return float32 (N, sequence_length, 4) in A/C/G/T order.
    Preserve sample order; empty input returns (0, sequence_length, 4).
    Input must contain cleaned A/C/G/T strings of sequence_length.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if isinstance(sequences, (str, bytes)):
        raise TypeError("sequences must be a collection of strings")

    encoded = np.zeros((len(sequences), sequence_length, 4), dtype=np.float32)
    positions = np.arange(sequence_length)
    
    for sample_index, sequence in enumerate(sequences):
        channels = np.fromiter(
            (_BASE_TO_CHANNEL[base] for base in sequence),
            dtype=np.intp,
        ).reshape(sequence_length)
        encoded[sample_index, positions, channels] = 1.0
    return encoded
