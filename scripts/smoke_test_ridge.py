"""Run a small real-data Ridge smoke test, without tuning or test-set scoring."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/GSM3130435_egfp_unmod_1.csv"))
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke/ridge"))
    args = parser.parse_args()

    cleaned, _ = clean_data(load_data(args.data_path))
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
    labels = y_val.astype(float)
    rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    baseline_rmse = np.sqrt(np.mean((labels - y_train.mean(dtype=np.float64)) ** 2))
    constant = np.ptp(labels) == 0 or np.ptp(predictions) == 0
    r = np.nan if constant else np.corrcoef(labels, predictions)[0, 1]
    summary = (
        f"Parameters: alpha={args.alpha}\n"
        f"Training samples: {len(y_train):,}\n"
        f"Validation: n={len(labels):,}, RMSE {rmse:.6f}, Pearson r {r:.6f}, "
        f"Pearson r² {r ** 2:.6f}, training-mean baseline RMSE {baseline_rmse:.6f}\n"
        "Test: not evaluated\n"
    )
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
