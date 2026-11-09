"""Train and evaluate one Random Forest with fixed parameters."""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter

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
    y_val = val["rl"].to_numpy(dtype="float32")
    y_test = test["rl"].to_numpy(dtype="float32")
    test[["source_row", "utr", "rl"]].assign(prediction=test_predictions).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False,
    )
    keys = ("n_estimators", "max_depth", "min_samples_leaf", "max_features")
    settings = ", ".join(f"{key}={RF_PARAMS[key]}" for key in keys)
    lines = [f"Parameters: {settings}", f"Training samples: {len(y_train):,}"]
    train_mean = y_train.mean(dtype=np.float64)
    for split, labels, predictions in (("Validation", y_val, val_predictions), ("Test", y_test, test_predictions)):
        metrics = regression_metrics(labels, predictions)
        baseline_rmse = np.sqrt(np.mean((labels.astype(float) - train_mean) ** 2))
        lines.append(
            f"{split}: n={len(labels):,}, RMSE {metrics['rmse']:.6f}, "
            f"Pearson r {metrics['pearson_r']:.6f}, Pearson r² {metrics['pearson_r_squared']:.6f}, "
            f"training-mean baseline RMSE {baseline_rmse:.6f}"
        )
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
