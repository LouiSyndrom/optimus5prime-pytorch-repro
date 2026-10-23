"""Train Ridge, select alpha by validation RMSE, and report Pearson r²."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ridge.data.encoding import one_hot_encode
from ridge.data.io import clean_data, load_data
from ridge.data.split import create_splits
from ridge.models.sklearn_adapter import SklearnAdapter


def regression_metrics(y_true, predictions):
    """Return Pearson r/r² and RMSE; constant inputs have undefined correlation."""
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    constant = np.ptp(y_true) == 0 or np.ptp(predictions) == 0
    r = np.nan if constant else float(np.corrcoef(y_true, predictions)[0, 1])
    return {
        "pearson_r_squared": r ** 2,
        "pearson_r": r,
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def select_alpha(X_train, y_train, X_val, y_val, alphas):
    """Keep the train-only model with lowest validation RMSE; ties keep first."""
    best_model, best_rmse = None, float("inf")
    for alpha in alphas:
        model = SklearnAdapter(Ridge(alpha=alpha), sequence_length=X_train.shape[1])
        model.fit(X_train, y_train)
        rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
        if rmse < best_rmse:
            best_model, best_rmse = model, rmse
    return best_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/GSM3130435_egfp_unmod_1.csv"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ridge"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data, qc = clean_data(load_data(args.data_path))
    if len(data) < 20:
        parser.error("At least 20 cleaned rows are needed for train/validation/test metrics")
    train, val, test = create_splits(data, seed=args.seed)
    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")

    print("Training Ridge...", flush=True)
    model = select_alpha(X_train, y_train, X_val, y_val, args.alphas)
    val_predictions = model.predict(X_val)
    model_path = args.output_dir / "ridge.joblib"
    model.save(model_path)
    del X_train, X_val

    # Evaluate test only after alpha is fixed; never use it for fitting or selection.
    test_predictions = model.predict(one_hot_encode(test["utr"].tolist()))
    y_test = test["rl"].to_numpy(dtype="float32")
    metrics = pd.DataFrame([
        {"split": "validation", **regression_metrics(y_val, val_predictions)},
        {"split": "test", **regression_metrics(y_test, test_predictions)},
    ])
    test[["source_row", "utr", "rl"]].assign(prediction=test_predictions).to_csv(
        args.output_dir / "test_predictions.csv", index=False,
    )
    summary = (
        f"data: {args.data_path.resolve()}\n"
        f"rows: raw={qc['input_rows']}, cleaned={len(data)}\n"
        f"seed: {args.seed}; train={len(train)}, val={len(val)}, test={len(test)}\n"
        f"alphas: {args.alphas}; selection: lowest validation RMSE\n"
        f"best_alpha: {model.estimator.alpha}\n\n"
        + metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n"
        + f"model: {model_path.resolve()}\n"
    )
    if metrics["pearson_r_squared"].isna().any():
        summary += "Pearson NaN: constant target or predictions.\n"
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
