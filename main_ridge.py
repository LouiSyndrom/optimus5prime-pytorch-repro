"""Fit Ridge at one fixed alpha and evaluate the validation and held-out test splits."""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter

DATA_PATH = Path(__file__).resolve().parent / "data/raw/GSM3130435_egfp_unmod_1.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42
FINAL_ALPHA = 1000.0


def xy(frame):
    return one_hot_encode(frame["utr"].tolist()), frame["rl"].to_numpy(dtype="float32")


def build_model(*, alpha, sequence_length=50):
    return SklearnAdapter(Ridge(alpha=alpha), sequence_length=sequence_length)


def regression_metrics(y_true, predictions):
    """Use Pearson r² as the paper metric; keep regression R² and MRL errors separate."""
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    constant = np.ptp(y_true) == 0 or np.ptp(predictions) == 0
    r = np.nan if constant else float(np.corrcoef(y_true, predictions)[0, 1])
    return {
        "pearson_r_squared": r ** 2,
        "pearson_r": r,
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "regression_r2": float(r2_score(y_true, predictions)),
        "pearson_note": "undefined: constant target or prediction" if constant else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="To search multiple alphas, edit CANDIDATES in grid_search_ridge.py and run it.",
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--alpha", type=float, default=FINAL_ALPHA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = RESULTS_DIR / f"ridge_fixed_seed{args.seed}"
        if output_dir.exists():
            output_dir = output_dir.with_name(f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")

    cleaned, _ = clean_data(load_data(args.data_path))
    if len(cleaned) < 20:
        parser.error("At least 20 cleaned rows are needed for train/validation/test metrics")
    train, val, test = create_splits(cleaned, seed=args.seed)
    print(f"Cleaned rows: {len(cleaned)}; train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    del cleaned
    X_train, y_train = xy(train)
    model = build_model(alpha=args.alpha).fit(X_train, y_train)
    del X_train
    train_mean = y_train.mean(dtype=np.float64)
    lines = [f"Parameters: alpha={args.alpha}", f"Training samples: {len(y_train):,}"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temp:
        staging = Path(temp) / "final"
        staging.mkdir()
        model_path = staging / "ridge.joblib"
        model.save(model_path)
        restored = SklearnAdapter.load(model_path)
        rows = []
        for name, part in (("validation", val), ("test", test)):
            X, labels = xy(part)
            predictions = restored.predict(X)
            if name == "validation":
                np.testing.assert_allclose(predictions, model.predict(X), equal_nan=False)
            scores = regression_metrics(labels, predictions)
            rows.append({"split": name, **scores})
            part[["source_row", "utr", "rl"]].assign(prediction=predictions.astype(float)).to_csv(
                staging / f"{name}_predictions.csv", index=False,
            )
            baseline_rmse = np.sqrt(np.mean((labels.astype(float) - train_mean) ** 2))
            lines.append(
                f"{name.title()}: n={len(labels):,}, RMSE {scores['rmse']:.6f}, "
                f"Pearson r {scores['pearson_r']:.6f}, Pearson r² {scores['pearson_r_squared']:.6f}, "
                f"training-mean baseline RMSE {baseline_rmse:.6f}"
            )
        pd.DataFrame(rows).to_csv(staging / "metrics.csv", index=False)
        summary = "\n".join(lines) + "\n"
        (staging / "summary.txt").write_text(summary, encoding="utf-8")
        staging.rename(output_dir)
    print(summary, end="")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
