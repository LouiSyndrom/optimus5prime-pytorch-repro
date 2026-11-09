"""Fit the frozen XGBoost model and evaluate the held-out test split."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from main_rf import regression_metrics
from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter

DATA_PATH = Path(__file__).resolve().parent / "data/raw/GSM3130435_egfp_unmod_1.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEARCH_DIR = RESULTS_DIR / "xgb_search_seed42"
OUTPUT_DIR = RESULTS_DIR / "xgb_fixed_seed42"
SPLIT_SEED = 42
MODEL_SEED = 42
N_JOBS = 2
FINAL_PARAMS = {"n_estimators": 400, "max_depth": 6}
COMMON_PARAMS = {
    "objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.05,
    "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
}


def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return position-wise A/C/G/T features and MRL labels in the same row order."""
    return one_hot_encode(frame["utr"].tolist()), frame["rl"].to_numpy(dtype="float32")


def build_model(
    *, n_estimators: int, max_depth: int, model_seed: int = 42, n_jobs: int = 2,
) -> SklearnAdapter:
    """Construct a fixed-round XGBoost regressor for the shared One-Hot input."""
    from xgboost import XGBRegressor

    estimator = XGBRegressor(
        **COMMON_PARAMS,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=model_seed,
        n_jobs=n_jobs,
    )
    return SklearnAdapter(estimator, sequence_length=50)


def verify_validation(model, val):
    """Reproduce the search winner's validation rows, predictions and RMSE."""
    scores = pd.read_csv(SEARCH_DIR / "validation_results.csv")
    winner = scores.loc[scores["validation_rmse"].idxmin()]
    if any(winner[key] != value for key, value in FINAL_PARAMS.items()):
        raise ValueError("Frozen parameters differ from the validation winner")
    reference = pd.read_csv(SEARCH_DIR / "validation_predictions.csv", float_precision="round_trip")
    columns = ["source_row", "utr", "rl"]
    pd.testing.assert_frame_equal(val[columns], reference[columns], check_dtype=False)
    X_val, y_val = xy(val)
    predictions = model.predict(X_val)
    np.testing.assert_allclose(predictions, reference["prediction"], rtol=1e-7, atol=0)
    metrics = regression_metrics(y_val, predictions)
    np.testing.assert_allclose(metrics["rmse"], winner["validation_rmse"], rtol=1e-9, atol=0)
    return metrics


def main():
    cleaned, qc = clean_data(load_data(DATA_PATH))
    train, val, test = create_splits(cleaned, seed=SPLIT_SEED)
    del cleaned

    for key in ("source_row", "utr"):
        members = pd.concat([train[key], val[key], test[key]], ignore_index=True)
        if len(members) != qc["output_rows"] or not members.is_unique:
            raise ValueError(f"Expected each cleaned {key} exactly once across the splits")

    X_train, y_train = xy(train)
    model = build_model(**FINAL_PARAMS, model_seed=MODEL_SEED, n_jobs=N_JOBS)
    print(f"Fitting {FINAL_PARAMS} on {len(train):,} training rows.", flush=True)
    start = perf_counter()
    model.fit(X_train, y_train)
    print(f"Fit completed in {perf_counter() - start:.2f} s.", flush=True)
    del X_train

    output_dir = OUTPUT_DIR
    if output_dir.exists():
        output_dir = output_dir.with_name(f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temp:
        staging = Path(temp) / "final"
        staging.mkdir()
        model_path = staging / "xgboost.joblib"
        model.save(model_path)
        restored = SklearnAdapter.load(model_path)
        validation = verify_validation(restored, val)
        print(f"Validation reproduced on {len(val):,} rows. Evaluating test.", flush=True)

        X_test, y_test = xy(test)
        predictions = restored.predict(X_test)
        test_metrics = regression_metrics(y_test, predictions)
        test[["source_row", "utr", "rl"]].assign(prediction=predictions.astype(float)).to_csv(
            staging / "test_predictions.csv", index=False,
        )
        settings = ", ".join(f"{key}={value}" for key, value in FINAL_PARAMS.items())
        lines = [f"Parameters: {settings}", f"Training samples: {len(y_train):,}"]
        train_mean = y_train.mean(dtype=np.float64)
        y_val = val["rl"].to_numpy(dtype="float32")
        for split, labels, metrics in (("Validation", y_val, validation), ("Test", y_test, test_metrics)):
            baseline_rmse = np.sqrt(np.mean((labels.astype(float) - train_mean) ** 2))
            lines.append(
                f"{split}: n={len(labels):,}, RMSE {metrics['rmse']:.6f}, "
                f"Pearson r {metrics['pearson_r']:.6f}, Pearson r² {metrics['pearson_r_squared']:.6f}, "
                f"training-mean baseline RMSE {baseline_rmse:.6f}"
            )
        summary = "\n".join(lines) + "\n"
        (staging / "summary.txt").write_text(summary, encoding="utf-8")
        staging.rename(output_dir)
    print(summary, end="")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
