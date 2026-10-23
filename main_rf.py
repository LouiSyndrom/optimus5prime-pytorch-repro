"""Train and evaluate one Random Forest with fixed parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ridge.data.encoding import one_hot_encode
from ridge.data.io import clean_data, load_data
from ridge.data.split import create_splits
from ridge.models.sklearn_adapter import SklearnAdapter

# Edit the run configuration here.
DATA_PATH = Path(__file__).resolve().parent / "data/raw/GSM3130435_egfp_unmod_1.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results/rf_fixed"
SEED = 42
N_JOBS = 2
RF_PARAMS = {
    "n_estimators": 100, "max_depth": 24, "min_samples_leaf": 20, "max_features": 1.0,
    "max_leaf_nodes": 10_000, "bootstrap": True, "max_samples": 0.8,
    "criterion": "squared_error",
}


def regression_metrics(y_true, predictions):
    """Report Pearson r² separately from MRL error; constant correlation is undefined."""
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    constant = np.ptp(y_true) == 0 or np.ptp(predictions) == 0
    r = np.nan if constant else float(np.corrcoef(y_true, predictions)[0, 1])
    return {
        "pearson_r_squared": r ** 2,
        "pearson_r": r,
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "pearson_note": "undefined: constant target or prediction" if constant else "",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    # Load and clean the data, then use the existing train/validation/test split.
    cleaned, _ = clean_data(load_data(DATA_PATH))
    train, val, test = create_splits(cleaned, seed=SEED)
    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")

    # Fit one model on the training set with fixed parameters.
    model = SklearnAdapter(
        RandomForestRegressor(**RF_PARAMS, random_state=SEED, n_jobs=N_JOBS),
        sequence_length=X_train.shape[1],
    ).fit(X_train, y_train)
    model.save(OUTPUT_DIR / "random_forest.joblib")
    del X_train

    # Evaluate validation and test sets, then save test predictions and main metrics.
    val_predictions = model.predict(one_hot_encode(val["utr"].tolist()))
    test_predictions = model.predict(one_hot_encode(test["utr"].tolist()))
    metrics = [
        {"split": "validation", **regression_metrics(val["rl"].to_numpy(dtype="float32"), val_predictions)},
        {"split": "test", **regression_metrics(test["rl"].to_numpy(dtype="float32"), test_predictions)},
    ]
    test[["source_row", "utr", "rl"]].assign(prediction=test_predictions).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False,
    )
    model_params = {key: RF_PARAMS[key] for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_features")}
    table = pd.DataFrame(metrics).drop(columns="pearson_note")
    summary = (
        f"model_params: {model_params}\n"
        f"seed: {SEED}\n"
        f"used_rows: train={len(train)}, val={len(val)}, test={len(test)}\n\n"
        + table.to_string(index=False, float_format=lambda value: f"{value:.6f}") + "\n"
    )
    if any(row["pearson_note"] for row in metrics):
        summary += "Pearson r/r² is undefined for constant targets or predictions.\n"
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
