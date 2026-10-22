"""CNN-compatible random 80/10/10 splits and reusable sample manifests."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import pandas as pd

_NAMES = ("train", "val", "test")
_DATA_COLUMNS = ["source_row", "utr", "rl", "total_reads"]
_SORT_ORDER = "source_row ascending"
_SPLIT_METHOD = "cnn_random_80_10_10"


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
) -> DataSplits:
    """Split all cleaned rows using the CNN's sequential random-pop algorithm.

    Restore CSV order before sampling. Train and validation each restart
    random.Random(seed); their sizes are floor(0.8*N) and floor(0.1*N).
    Test keeps the remaining indices in CSV order. Matching CNN membership
    requires the same sequence pool (cleaning must not remove/change samples).
    """
    selected = df.sort_values("source_row")
    remaining = list(range(len(selected)))
    sampled = []
    for fraction in (0.8, 0.1):
        rng = random.Random(seed)
        indices = [
            remaining.pop(rng.randint(0, len(remaining) - 1))
            for _ in range(int(len(selected) * fraction))
        ]
        sampled.append(selected.iloc[indices])
    train, val = sampled
    test = selected.iloc[remaining]
    counts = {name: len(part) for name, part in zip(_NAMES, (train, val, test))}
    metadata = {
        "schema_version": 2, "mode": "benchmark", "seed": int(seed),
        "split_method": _SPLIT_METHOD, "train_fraction": 0.8, "val_fraction": 0.1,
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
        or splits.metadata["schema_version"] != 2
        or splits.metadata.get("split_method") != _SPLIT_METHOD
        or splits.metadata["sort_order"] != _SORT_ORDER
        or {name: len(part) for name, part in parts.items()} != splits.metadata["counts"]
    ):
        raise ValueError("Expected CNN 80/10/10 splits with matching metadata counts")
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
    Extra-column contents are not fingerprinted or compared.
    """
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if (
        metadata["mode"] != "benchmark"
        or metadata["schema_version"] != 2
        or metadata.get("split_method") != _SPLIT_METHOD
        or metadata["sort_order"] != _SORT_ORDER
    ):
        raise ValueError("Expected a CNN 80/10/10 manifest (schema 2); regenerate old splits")
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
