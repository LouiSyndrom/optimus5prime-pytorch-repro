"""Fixed coverage-based splits and manifests matched by source_row and utr."""

from dataclasses import dataclass
import json
from math import ceil
from numbers import Integral, Real
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ridge.data.io import _require_columns

_NAMES = ("train", "val", "test")
_DATA_COLUMNS = ["source_row", "utr", "rl", "total_reads"]
_SORT_ORDER = "total_reads descending, source_row ascending"


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


def _check_data(df):
    _require_columns(df, _DATA_COLUMNS)
    rows = df["source_row"]
    sequences = df["utr"].astype("string")
    if (
        not pd.api.types.is_integer_dtype(rows.dtype)
        or rows.isna().any() or rows.lt(0).any() or not rows.is_unique
        or not sequences.is_unique
        or not sequences.str.fullmatch("[ACGT]+", na=False).all()
        or sequences.str.len().nunique() > 1
    ):
        raise ValueError("Expected cleaned data with unique source_row and utr sequences")
    numbers = df[["rl", "total_reads"]]
    if (
        not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numbers.dtypes)
        or not np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan)).all()
        or df["total_reads"].lt(0).any()
    ):
        raise ValueError("Expected finite numeric labels and non-negative total_reads")


def _split_counts(mode, seed, n_total, n_test, val_fraction):
    if mode not in ("paper", "benchmark"):
        raise ValueError("mode must be 'paper' or 'benchmark'")
    if (
        any(not isinstance(n, Integral) or isinstance(n, bool) for n in (seed, n_total, n_test))
        or not 0 <= seed < 2**32
        or not 0 < n_test < n_total
    ):
        raise ValueError("Require integer 0 < n_test < n_total and seed in [0, 2**32)")
    if (
        not isinstance(val_fraction, Real) or isinstance(val_fraction, bool)
        or not 0 <= val_fraction < 1
        or (mode == "benchmark" and val_fraction == 0)
    ):
        raise ValueError("val_fraction must be in (0, 1) for benchmark, or [0, 1) for paper")
    n_val = ceil((n_total - n_test) * val_fraction) if mode == "benchmark" else 0
    n_train = n_total - n_test - n_val
    if n_train < 1:
        raise ValueError("Split would leave an empty training set")
    return {"train": int(n_train), "val": int(n_val), "test": int(n_test)}


def _check_metadata(metadata):
    required = {"schema_version", "mode", "seed", "n_total", "n_test",
                "val_fraction", "input_rows", "counts", "sort_order"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Incomplete manifest metadata")
    if metadata["schema_version"] != 1 or metadata["sort_order"] != _SORT_ORDER:
        raise ValueError("Unsupported manifest schema or sort_order")
    counts = _split_counts(*(metadata[key] for key in ("mode", "seed", "n_total", "n_test", "val_fraction")))
    if (
        metadata["counts"] != counts
        or not isinstance(metadata["input_rows"], int)
        or metadata["input_rows"] < metadata["n_total"]
    ):
        raise ValueError("Manifest metadata counts do not match split parameters")
    return counts


def create_splits(
    df: pd.DataFrame,
    *,
    mode: str = "benchmark",
    seed: int = 42,
    n_total: int = 280_000,
    n_test: int = 20_000,
    val_fraction: float = 0.1,
) -> DataSplits:
    """Select by coverage; keep the highest-coverage test set independent of seed."""
    counts = _split_counts(mode, seed, n_total, n_test, val_fraction)
    _check_data(df)
    if len(df) < n_total:
        raise ValueError(f"Need {n_total} cleaned rows, but only {len(df)} are available")
    selected = df.sort_values(["total_reads", "source_row"], ascending=[False, True]).iloc[:n_total]
    test, pool = selected.iloc[:n_test], selected.iloc[n_test:]
    if mode == "benchmark":
        train, val = train_test_split(pool, test_size=counts["val"], random_state=int(seed))
    else:
        train, val = pool, pool.iloc[:0]
    metadata = {
        "schema_version": 1, "mode": mode, "seed": int(seed),
        "n_total": int(n_total), "n_test": int(n_test), "val_fraction": float(val_fraction),
        "input_rows": len(df), "counts": counts, "sort_order": _SORT_ORDER,
    }
    return DataSplits(*(part.reset_index(drop=True).copy() for part in (train, val, test)), metadata)


def save_split_manifest(
    splits: DataSplits,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save CSV plus .meta.json; JSON also records each split's source_row order."""
    counts = _check_metadata(splits.metadata)
    parts = {name: getattr(splits, name) for name in _NAMES}
    if {name: len(part) for name, part in parts.items()} != counts:
        raise ValueError("Split sizes do not match metadata counts")
    for part in parts.values():
        _require_columns(part, _DATA_COLUMNS)
    manifest = pd.concat([
        part[_DATA_COLUMNS].assign(split=name, position=np.arange(len(part)))
        for name, part in parts.items()
    ], ignore_index=True)
    _check_data(manifest)  # Also rejects overlap between the three splits.
    metadata = dict(splits.metadata)
    metadata["source_rows"] = {name: part["source_row"].tolist() for name, part in parts.items()}
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not overwrite and (path.exists() or meta_path.exists()):
        raise FileExistsError(f"Manifest already exists: {path} or {meta_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish both files before publishing. Exclusive links protect existing files.
    with TemporaryDirectory(prefix=".ridge-manifest-", dir=path.parent) as directory:
        csv_temp, json_temp = Path(directory) / "manifest.csv", Path(directory) / "meta.json"
        manifest.to_csv(csv_temp, index=False, float_format="%.17g")
        json_temp.write_text(metadata_text, encoding="utf-8")
        if overwrite:
            os.replace(csv_temp, path)
            os.replace(json_temp, meta_path)
        else:
            os.link(csv_temp, path)
            try:
                os.link(json_temp, meta_path)
            except OSError:
                path.unlink()
                raise


def load_split_manifest(df: pd.DataFrame, path: str | Path) -> DataSplits:
    """Restore saved order without splitting again; check selected sample values.

    Match source_row, utr, rl and total_reads, plus the full input row count.
    Unselected rows and extra-column contents are not fingerprinted or compared.
    """
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    counts = _check_metadata(metadata)
    source_rows = metadata.pop("source_rows", None)
    if not isinstance(source_rows, dict) or set(source_rows) != set(_NAMES):
        raise ValueError("Missing or invalid source_rows in manifest metadata")
    _check_data(df)
    if len(df) != metadata["input_rows"]:
        raise ValueError("Input row count differs from the saved dataset")
    manifest = pd.read_csv(path, dtype={"utr": "string", "split": "string"}, float_precision="round_trip")
    _require_columns(manifest, _DATA_COLUMNS + ["split", "position"])
    _check_data(manifest)
    if not manifest["split"].isin(_NAMES).all() or not pd.api.types.is_integer_dtype(manifest["position"].dtype):
        raise ValueError("Invalid manifest split names or positions")
    if not manifest["source_row"].isin(df["source_row"]).all():
        raise ValueError("Manifest source_row is missing from the supplied data")

    indexed = df.set_index("source_row", drop=False)
    restored = {}
    for name in _NAMES:
        part = manifest.loc[manifest["split"].eq(name)].sort_values("position").reset_index(drop=True)
        if len(part) != counts[name] or part["position"].tolist() != list(range(counts[name])):
            raise ValueError(f"Invalid {name} manifest count or positions")
        if part["source_row"].tolist() != source_rows[name]:
            raise ValueError(f"Manifest CSV and metadata disagree on {name} source_row order")
        current = indexed.loc[part["source_row"].tolist()].reset_index(drop=True).copy()
        if not part[_DATA_COLUMNS].eq(current[_DATA_COLUMNS]).all().all():
            raise ValueError(f"Saved {name} sample differs in source_row, utr, rl or total_reads")
        restored[name] = current
    return DataSplits(*(restored[name] for name in _NAMES), metadata)
