"""Run a small real-data Ridge smoke test, without tuning or test-set scoring."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
from sklearn.linear_model import Ridge

from ridge.data.encoding import one_hot_encode
from ridge.data.io import clean_data, load_data
from ridge.data.split import create_splits
from ridge.models.sklearn_adapter import SklearnAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/GSM3130435_egfp_unmod_1.csv"))
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke/ridge"))
    args = parser.parse_args()

    raw = load_data(args.data_path)
    cleaned, qc = clean_data(raw)
    if not 10 <= args.n_samples <= len(cleaned):
        parser.error(f"--n-samples must be between 10 and {len(cleaned)} cleaned rows")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    sampled = cleaned.sample(n=args.n_samples, random_state=args.seed)
    train, val, test = create_splits(sampled, seed=args.seed)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([part["source_row"].to_numpy() for part in (train, val, test)])),
        np.sort(sampled["source_row"].to_numpy()),
    )

    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")
    model = SklearnAdapter(Ridge(alpha=args.alpha))
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    predictions = model.predict(X_val)
    assert np.isfinite(predictions).all()

    model_path = args.output_dir / "ridge.joblib"
    model.save(model_path)
    restored = SklearnAdapter.load(model_path)
    np.testing.assert_allclose(restored.predict(X_val), predictions, equal_nan=False)
    val[["source_row", "utr", "rl"]].assign(prediction=predictions).to_csv(
        args.output_dir / "validation_predictions.csv", index=False,
    )
    rmse = float(np.sqrt(np.mean((predictions - y_val) ** 2)))
    versions = ", ".join(f"{name}={version(name)}" for name in ("numpy", "pandas", "scikit-learn", "joblib"))
    summary = (
        "run_type: smoke\n"
        f"data_path: {args.data_path.resolve()}\n"
        f"raw_rows: {len(raw)}, cleaned_rows: {len(cleaned)}\n"
        f"cleaning_removed: {qc['removed']}\n"
        f"sampled_rows: {len(sampled)}, seed: {args.seed}, alpha: {args.alpha}\n"
        f"split_counts: train={len(train)}, val={len(val)}, test={len(test)}\n"
        f"X_train: {X_train.shape}, y_train: {y_train.shape}\n"
        f"X_val: {X_val.shape}, y_val: {y_val.shape}, predictions: {predictions.shape}\n"
        f"validation_rmse: {rmse:.6f}\n"
        "reload_predictions_match: True\n"
        "test_evaluated: False\n"
        f"python: {platform.python_version()}, {versions}\n"
    )
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
