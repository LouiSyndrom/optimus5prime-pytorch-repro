"""Fixed coverage-based splits and manifests matched by source_row and utr."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import train_test_split

_NAMES = ("train", "val", "test")
_DATA_COLUMNS = ["source_row", "utr", "rl", "total_reads"]
_SORT_ORDER = "total_reads descending, source_row ascending"


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


def create_splits(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    n_total: int = 280_000,
    n_test: int = 20_000,
    val_fraction: float = 0.1,
) -> DataSplits:
    """Benchmark split of cleaned data: fixed test, random train/validation."""
    if not 0 < n_test < n_total <= len(df):
        raise ValueError("Require 0 < n_test < n_total <= number of cleaned rows")
    selected = df.sort_values(["total_reads", "source_row"], ascending=[False, True]).iloc[:n_total]
    test, pool = selected.iloc[:n_test], selected.iloc[n_test:]
    train, val = train_test_split(pool, test_size=float(val_fraction), random_state=seed)
    counts = {name: len(part) for name, part in zip(_NAMES, (train, val, test))}
    metadata = {
        "schema_version": 1, "mode": "benchmark", "seed": int(seed),
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
    parts = {name: getattr(splits, name) for name in _NAMES}
    if (
        splits.metadata["mode"] != "benchmark"
        or {name: len(part) for name, part in parts.items()} != splits.metadata["counts"]
    ):
        raise ValueError("Expected benchmark splits with matching metadata counts")
    manifest = pd.concat([
        part[_DATA_COLUMNS].assign(split=name, position=range(len(part)))
        for name, part in parts.items()
    ], ignore_index=True)
    if not manifest["source_row"].is_unique or not manifest["utr"].is_unique:
        raise ValueError("Splits must not contain duplicate samples or overlap")
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

    Expect clean_data() output; match source_row, utr, rl, total_reads and row count.
    Unselected rows and extra-column contents are not fingerprinted or compared.
    """
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if (
        metadata["mode"] != "benchmark"
        or metadata["schema_version"] != 1
        or metadata["sort_order"] != _SORT_ORDER
    ):
        raise ValueError("Expected a benchmark manifest with supported schema and sort_order")
    source_rows = metadata.pop("source_rows")
    if len(df) != metadata["input_rows"]:
        raise ValueError("Input row count differs from the saved dataset")
    manifest = pd.read_csv(path, dtype={"utr": "string", "split": "string"}, float_precision="round_trip")
    if (
        not manifest["split"].isin(_NAMES).all()
        or not manifest["source_row"].is_unique
        or not manifest["utr"].is_unique
    ):
        raise ValueError("Manifest contains unknown splits or duplicate samples")

    indexed = df.set_index("source_row", drop=False, verify_integrity=True)
    restored = {}
    for name in _NAMES:
        part = manifest.loc[manifest["split"].eq(name)].sort_values("position").reset_index(drop=True)
        if (
            len(part) != metadata["counts"][name]
            or part["position"].tolist() != list(range(len(part)))
            or part["source_row"].tolist() != source_rows[name]
        ):
            raise ValueError(f"Saved {name} count or order does not match metadata")
        current = indexed.loc[part["source_row"].tolist()].reset_index(drop=True).copy()
        if not part[_DATA_COLUMNS].eq(current[_DATA_COLUMNS]).fillna(False).all().all():
            raise ValueError(f"Saved {name} sample differs in source_row, utr, rl or total_reads")
        restored[name] = current
    return DataSplits(*(restored[name] for name in _NAMES), metadata)
