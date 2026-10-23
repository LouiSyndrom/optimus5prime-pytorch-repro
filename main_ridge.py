"""Train Ridge and report Pearson r² as in the paper; select alpha by validation RMSE."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ridge.data.encoding import one_hot_encode
from ridge.data.io import clean_data, load_data
from ridge.data.split import create_splits
from ridge.models.sklearn_adapter import SklearnAdapter


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


def select_alpha(X_train, y_train, X_val, y_val, alphas):
    """Return the best train-only model and validation search results; ties keep first."""
    best_model, best_rmse = None, float("inf")
    records = []
    for alpha in alphas:
        model = SklearnAdapter(Ridge(alpha=alpha), sequence_length=X_train.shape[1])
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predictions = model.predict(X_val)
        predict_seconds = perf_counter() - start
        rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))
        records.append({"alpha": alpha, "validation_rmse": rmse,
                        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        print(f"alpha={alpha:g}, validation RMSE={rmse:.6f}", flush=True)
        if rmse < best_rmse:
            best_model, best_rmse = model, rmse
    return best_model, pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/GSM3130435_egfp_unmod_1.csv"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ridge"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_start = perf_counter()

    raw = load_data(args.data_path)
    cleaned, qc = clean_data(raw)
    raw_rows = len(raw)
    del raw
    if len(cleaned) < 20:
        parser.error("At least 20 cleaned rows are needed for train/validation/test metrics")
    train, val, test = create_splits(cleaned, seed=args.seed)
    print(f"Cleaned rows: {len(cleaned)}; train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")

    start = perf_counter()
    model, search = select_alpha(X_train, y_train, X_val, y_val, args.alphas)
    search_seconds = perf_counter() - start
    search.to_csv(args.output_dir / "alpha_search.csv", index=False)
    val_predictions = model.predict(X_val)
    model.save(args.output_dir / "ridge.joblib")
    restored = SklearnAdapter.load(args.output_dir / "ridge.joblib")
    np.testing.assert_allclose(restored.predict(X_val), val_predictions, equal_nan=False)
    del X_train, X_val

    # Alpha is now fixed. Test data never enters fitting or parameter selection.
    X_test = one_hot_encode(test["utr"].tolist())
    y_test = test["rl"].to_numpy(dtype="float32")
    start = perf_counter()
    test_predictions = restored.predict(X_test)
    test_predict_seconds = perf_counter() - start
    metrics = pd.DataFrame([
        {"split": "validation", **regression_metrics(y_val, val_predictions)},
        {"split": "test", **regression_metrics(y_test, test_predictions)},
    ])
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    for name, part, predictions in [("validation", val, val_predictions), ("test", test, test_predictions)]:
        part[["source_row", "utr", "rl"]].assign(prediction=predictions).to_csv(
            args.output_dir / f"{name}_predictions.csv", index=False,
        )

    versions = ", ".join(f"{name}={version(name)}" for name in ("numpy", "pandas", "scikit-learn", "joblib"))
    summary = (
        "run_type: full_data_regression\n"
        "primary_metric: pearson_r_squared = corr(rl, prediction)^2 (paper definition)\n"
        f"data_path: {args.data_path.resolve()}\n"
        f"raw_rows: {raw_rows}, cleaned_rows: {len(cleaned)}\n"
        f"qc: {qc}\n"
        f"seed: {args.seed}, split_method: CNN random 80/10/10\n"
        f"split_counts: train={len(train)}, val={len(val)}, test={len(test)}\n"
        "features: utr, one_hot (50,4), ACGT, flatten C, float32\n"
        "target: rl, original MRL; no feature or target scaling\n"
        f"alphas: {args.alphas}, selection: lowest validation RMSE (ties keep first)\n"
        f"best_alpha: {model.estimator.alpha}, fit_scope: train only\n"
        f"model_config: {model.get_config()}\n"
        f"search_seconds: {search_seconds:.3f}, total_fit_seconds: {search['fit_seconds'].sum():.3f}\n"
        f"test_predict_seconds: {test_predict_seconds:.3f}, run_seconds: {perf_counter() - run_start:.3f}\n"
        "reload_predictions_match: True; test evaluated after alpha selection\n"
        f"python: {platform.python_version()}, {versions}\n\n"
        + metrics.to_string(index=False) + "\n"
    )
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\nPrimary metric: Pearson r² (paper definition); alpha selection: validation RMSE")
    print(f"Selected alpha: {model.estimator.alpha}\n{metrics.to_string(index=False)}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
