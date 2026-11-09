"""Select Random Forest parameters on the project's full, fixed validation split."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from main_rf import DATA_PATH as DEFAULT_DATA_PATH
from main_rf import RF_PARAMS, regression_metrics
from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter

# Edit the run configuration here.
DATA_PATH = DEFAULT_DATA_PATH
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_SEED = 42
MODEL_SEED = 42
N_JOBS = 6
BASELINE_MODEL = None  # Optional path to a previously saved RF model.
CANDIDATES = (
    {"n_estimators": 50, "max_depth": 16},
    {"n_estimators": 100, "max_depth": 16},
    {"n_estimators": 50, "max_depth": 24},
    {"n_estimators": 100, "max_depth": 24},
)


def search_candidates(
    X_train, y_train, X_val, y_val, *, candidates, model_seed=MODEL_SEED, n_jobs=N_JOBS, output_dir,
):
    """Fit each candidate on train only. Keep the first model at the lowest RMSE."""
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required")
    best_model, best_rmse = None, float("inf")
    records = []
    path = Path(output_dir) / "validation_results.csv"
    for number, params in enumerate(candidates, start=1):
        estimator = RandomForestRegressor(
            **(RF_PARAMS | params), random_state=model_seed, n_jobs=n_jobs,
        )
        model = SklearnAdapter(estimator, sequence_length=X_train.shape[1])
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
    if len(train) < 2 or len(val) < 2:
        raise ValueError("At least two training and validation rows are required")
    test_rows = len(test)
    del test, cleaned
    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")
    del train
    if BASELINE_MODEL is not None:
        baseline = SklearnAdapter.load(BASELINE_MODEL)
        previous = regression_metrics(y_val, baseline.predict(X_val))
        print(f"Previous model validation: RMSE {previous['rmse']:.6f}, "
              f"Pearson r {previous['pearson_r']:.6f}, Pearson r² {previous['pearson_r_squared']:.6f}")
        del baseline

    output_dir = RESULTS_DIR / f"rf_grid4_seed{SPLIT_SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Full train={len(y_train)}, validation={len(y_val)}, held-out test={test_rows}", flush=True)
    model, results = search_candidates(
        X_train, y_train, X_val, y_val, candidates=CANDIDATES,
        model_seed=MODEL_SEED, n_jobs=N_JOBS, output_dir=output_dir,
    )
    best = results.loc[results["validation_rmse"].idxmin()]
    best_params = CANDIDATES[int(best["candidate"]) - 1]
    predictions = model.predict(X_val)
    selected_metrics = regression_metrics(y_val, predictions)
    np.testing.assert_allclose(selected_metrics["rmse"], best["validation_rmse"], rtol=1e-12)
    model.save(output_dir / "random_forest.joblib")
    val[["source_row", "utr", "rl"]].assign(prediction=predictions).to_csv(
        output_dir / "validation_predictions.csv", index=False,
    )
    settings = ", ".join(f"{key}={value}" for key, value in best_params.items())
    train_mean = y_train.mean(dtype=np.float64)
    baseline_rmse = np.sqrt(np.mean((y_val.astype(float) - train_mean) ** 2))
    summary = (
        f"Parameters: {settings}\n"
        f"Training samples: {len(y_train):,}\n"
        f"Validation: n={len(y_val):,}, RMSE {selected_metrics['rmse']:.6f}, "
        f"Pearson r {selected_metrics['pearson_r']:.6f}, Pearson r² {selected_metrics['pearson_r_squared']:.6f}, "
        f"training-mean baseline RMSE {baseline_rmse:.6f}\n"
        "Test: not evaluated\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
