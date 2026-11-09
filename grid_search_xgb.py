"""Select XGBoost parameters on the project's full, fixed validation split."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from main_rf import regression_metrics
from main_xgb import DATA_PATH, SPLIT_SEED, build_model, xy
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits

# DATA_PATH and SPLIT_SEED are shared with main_xgb.py.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_SEED = 42
N_JOBS = 2
CANDIDATES = (
    {"n_estimators": 200, "max_depth": 3},
    {"n_estimators": 400, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 6},
    {"n_estimators": 400, "max_depth": 6},
)


def search_candidates(
    X_train, y_train, X_val, y_val, *, candidates, model_seed=42, n_jobs=2, output_dir,
):
    """Fit each candidate on train only. Keep the first model at the lowest RMSE."""
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required")
    best_model, best_rmse = None, float("inf")
    records = []
    path = Path(output_dir) / "validation_results.csv"
    for number, params in enumerate(candidates, start=1):
        model = build_model(**params, model_seed=model_seed, n_jobs=n_jobs)
        print(f"Fitting {number}/{len(candidates)}: {params}", flush=True)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        metrics = regression_metrics(y_val, model.predict(X_val))
        rmse = metrics["rmse"]
        records.append({
            "candidate": number, **params, "validation_rmse": rmse,
            "pearson_r": metrics["pearson_r"], "pearson_r_squared": metrics["pearson_r_squared"],
            "pearson_note": metrics["pearson_note"], "fit_seconds": fit_seconds,
        })
        results = pd.DataFrame(records)
        temporary = path.with_suffix(".tmp")
        results.to_csv(temporary, index=False)
        temporary.replace(path)
        if rmse < best_rmse:
            best_model, best_rmse = model, rmse
        print(f"Validation RMSE={rmse:.9f}, fit={fit_seconds:.2f}s", flush=True)
    return best_model, results


def main():
    cleaned, _ = clean_data(load_data(DATA_PATH))
    train, val, test = create_splits(cleaned, seed=SPLIT_SEED)
    test_rows = len(test)
    del cleaned, test
    if len(train) < 2 or len(val) < 2:
        raise ValueError("At least two training and validation rows are required")
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    del train
    output_dir = RESULTS_DIR / f"xgb_search_seed{SPLIT_SEED}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Full train={len(y_train)}, validation={len(y_val)}, held-out test={test_rows}", flush=True)

    best_model, results = search_candidates(
        X_train, y_train, X_val, y_val, candidates=CANDIDATES,
        model_seed=MODEL_SEED, n_jobs=N_JOBS, output_dir=output_dir,
    )
    best = results.loc[results["validation_rmse"].idxmin()]
    predictions = best_model.predict(X_val)
    metrics = regression_metrics(y_val, predictions)
    np.testing.assert_allclose(metrics["rmse"], best["validation_rmse"], rtol=1e-12)
    val[["source_row", "utr", "rl"]].assign(prediction=predictions.astype(float)).to_csv(
        output_dir / "validation_predictions.csv", index=False,
    )

    best_params = CANDIDATES[int(best["candidate"]) - 1]
    settings = ", ".join(f"{key}={value}" for key, value in best_params.items())
    train_mean = y_train.mean(dtype=np.float64)
    baseline_rmse = np.sqrt(np.mean((y_val.astype(float) - train_mean) ** 2))
    summary = (
        f"Parameters: {settings}\n"
        f"Training samples: {len(y_train):,}\n"
        f"Validation: n={len(y_val):,}, RMSE {metrics['rmse']:.6f}, "
        f"Pearson r {metrics['pearson_r']:.6f}, Pearson r² {metrics['pearson_r_squared']:.6f}, "
        f"training-mean baseline RMSE {baseline_rmse:.6f}\n"
        "Test: not evaluated\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Best candidate {int(best['candidate'])}, validation RMSE={metrics['rmse']:.9f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
