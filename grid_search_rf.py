"""Exhaustive RF search on the existing validation split, without reusing the test set."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit

from main_rf import regression_metrics
from ridge.data.encoding import one_hot_encode
from ridge.data.io import clean_data, load_data
from ridge.data.split import create_splits
from ridge.models.sklearn_adapter import SklearnAdapter

PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [24, None],
    "min_samples_leaf": [5, 20],
    "max_features": [0.6, 1.0],
}


def search_random_forest(
    X_train, y_train, X_val, y_val, param_grid, *, model_seed=42, n_jobs=6, verbose=2,
    checkpoint_path=None,
):
    """Search one fixed validation split; disable automatic train+validation refitting."""
    X = np.concatenate([
        X_train.reshape(len(X_train), -1, order="C"),
        X_val.reshape(len(X_val), -1, order="C"),
    ])
    y = np.concatenate([y_train, y_val])
    folds = np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_val), dtype=int)])
    estimator = RandomForestRegressor(
        criterion="squared_error", max_leaf_nodes=10_000, bootstrap=True,
        max_samples=0.8, random_state=model_seed, n_jobs=n_jobs,
    )
    scoring = "neg_root_mean_squared_error"
    if checkpoint_path is not None:
        scorer = get_scorer(scoring)
        completed = []

        def scoring(fitted, X_validation, y_validation):
            score = scorer(fitted, X_validation, y_validation)
            params = fitted.get_params()
            completed.append({**{key: params[key] for key in param_grid}, "validation_rmse": -score})
            table = pd.DataFrame(completed)
            table["rank"] = table["validation_rmse"].rank(method="min").astype(int)
            temporary = Path(checkpoint_path).with_suffix(".tmp")
            table.sort_values("rank", kind="stable").to_csv(temporary, index=False, na_rep="None")
            temporary.replace(checkpoint_path)
            print(f"Completed {len(completed)}/{len(ParameterGrid(param_grid))}: validation RMSE={-score:.6f}", flush=True)
            return score

    search = GridSearchCV(
        estimator, param_grid, scoring=scoring,
        cv=PredefinedSplit(folds), refit=False, n_jobs=1,
        error_score="raise", return_train_score=False, verbose=verbose,
    )
    return search.fit(X, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/GSM3130435_egfp_unmod_1.csv"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=6, help="Forest threads; grid candidates run sequentially")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rf_grid16_seed42"))
    parser.add_argument("--baseline-model", type=Path, help="Optional trusted prior model for validation comparison")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    raw = load_data(args.data_path)
    cleaned, _ = clean_data(raw)
    del raw
    train, val, test = create_splits(cleaned, seed=args.split_seed)
    if len(train) < 2 or len(val) < 2:
        parser.error("At least two training and validation rows are required")
    test_rows = len(test)
    del test, cleaned
    X_train = one_hot_encode(train["utr"].tolist())
    y_train = train["rl"].to_numpy(dtype="float32")
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")
    metrics = []
    if args.baseline_model is not None:
        baseline = SklearnAdapter.load(args.baseline_model)
        metrics.append({"model": "previous", "split": "validation",
                        **regression_metrics(y_val, baseline.predict(X_val))})
        del baseline

    args.output_dir.mkdir(parents=True, exist_ok=False)
    count = len(ParameterGrid(PARAM_GRID))
    print(f"Full train={len(train)}, validation={len(val)}; searching {count} combinations; test not evaluated", flush=True)
    search = search_random_forest(
        X_train, y_train, X_val, y_val, PARAM_GRID,
        model_seed=args.model_seed, n_jobs=args.n_jobs,
        checkpoint_path=args.output_dir / "cv_results.csv",
    )
    results = pd.DataFrame(search.cv_results_["params"])
    results["validation_rmse"] = -search.cv_results_["mean_test_score"]
    results["rank"] = search.cv_results_["rank_test_score"]
    results["fit_seconds"] = search.cv_results_["mean_fit_time"]
    temporary = args.output_dir / "cv_results.tmp"
    results.sort_values("rank", kind="stable").to_csv(temporary, index=False, na_rep="None")
    temporary.replace(args.output_dir / "cv_results.csv")
    print(f"Best parameters: {search.best_params_}; fitting on training data only", flush=True)
    model = SklearnAdapter(
        clone(search.estimator).set_params(**search.best_params_), sequence_length=X_train.shape[1],
    ).fit(X_train, y_train)
    predictions = model.predict(X_val)
    selected_metrics = regression_metrics(y_val, predictions)
    np.testing.assert_allclose(selected_metrics["rmse"], -search.best_score_, rtol=1e-10)
    metrics.append({"model": "grid_search", "split": "validation", **selected_metrics})
    model.save(args.output_dir / "random_forest.joblib")
    val[["source_row", "utr", "rl"]].assign(prediction=predictions).to_csv(
        args.output_dir / "validation_predictions.csv", index=False,
    )
    table = pd.DataFrame(metrics).drop(columns="pearson_note")
    summary = (
        "search: GridSearchCV, one fixed validation split (not K-fold CV)\n"
        f"data_path: {args.data_path.resolve()}\n"
        f"split_seed: {args.split_seed}, model_seed: {args.model_seed}\n"
        f"used_rows: train={len(train)}, validation={len(val)}; held_out_test={test_rows}\n"
        f"candidates: {count}; forest_threads={args.n_jobs}; concurrent_candidates=1\n"
        f"grid: {PARAM_GRID}\n"
        f"best_params: {search.best_params_}\n"
        "selection: lowest validation RMSE; refit=False; selected model fitted on train only\n"
        "test_evaluated: False; existing test set was evaluated in an earlier experiment\n\n"
        + table.to_string(index=False, float_format=lambda value: f"{value:.6f}") + "\n"
    )
    if any(row["pearson_note"] for row in metrics):
        summary += "Pearson r/r² is undefined for constant targets or predictions.\n"
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
