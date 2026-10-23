"""Check grid selection, train-only fitting, and the saved validation result."""

from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit

from ridge.models.sklearn_adapter import SklearnAdapter


def _load_program(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    return runpy.run_path(root / "grid_search_rf.py")


def test_grid_uses_existing_validation_split_and_selects_lowest_rmse(monkeypatch):
    program = _load_program(monkeypatch)
    X_train = np.eye(4, dtype=np.float32)[[0, 1] * 40][:, None, :]
    y_train = np.array([0, 2] * 40, dtype=np.float32)
    X_val = X_train[:4].copy()
    y_val = np.array([0.1, 1.9] * 2, dtype=np.float32)
    param_grid = {
        "n_estimators": [7], "max_depth": [2],
        "min_samples_leaf": [50, 1, 80], "max_features": [1.0],
    }
    fitted_inputs = []
    original_fit = RandomForestRegressor.fit

    def record_fit(estimator, X, y, *args, **kwargs):
        fitted_inputs.append((np.array(X, copy=True), np.array(y, copy=True)))
        return original_fit(estimator, X, y, *args, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", record_fit)
    search = program["search_random_forest"](
        X_train, y_train, X_val, y_val, param_grid,
        model_seed=23, n_jobs=2, verbose=0,
    )

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.cv, PredefinedSplit)
    np.testing.assert_array_equal(search.cv.test_fold, [-1] * 80 + [0] * 4)
    assert search.n_splits_ == 1
    assert search.refit is False
    assert not hasattr(search, "best_estimator_")
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.n_jobs == 1
    assert search.estimator.n_jobs == 2
    assert search.estimator.random_state == 23
    assert search.estimator.bootstrap is True
    assert search.estimator.max_samples == 0.8
    assert search.estimator.max_leaf_nodes == 10_000
    assert len(fitted_inputs) == len(ParameterGrid(param_grid))
    for fitted_X, fitted_y in fitted_inputs:
        np.testing.assert_array_equal(fitted_X, X_train.reshape(80, 4))
        np.testing.assert_array_equal(fitted_y, y_train)
    assert search.best_index_ == 1
    assert search.best_params_["min_samples_leaf"] == 1
    assert search.best_score_ == pytest.approx(-0.1, abs=1e-7)
    assert search.best_index_ == np.argmax(search.cv_results_["mean_test_score"])


def test_cli_refits_only_training_rows_and_saves_validation_artifacts(tmp_path, monkeypatch):
    program = _load_program(monkeypatch)
    namespace = program["main"].__globals__
    rng = np.random.default_rng(29)
    channels = rng.integers(0, 4, size=(80, 50))
    sequences = ["".join(np.array(list("ACGT"))[row]) for row in channels]
    data_path = tmp_path / "synthetic.csv"
    pd.DataFrame({
        "utr": sequences,
        "rl": 2 + (channels[:, 0] == 0) * 3 + rng.normal(0, 0.2, len(channels)),
        "total_reads": np.arange(len(channels)) + 100,
    }).to_csv(data_path, index=False)
    cleaned, _ = namespace["clean_data"](namespace["load_data"](data_path))
    train, val, test = namespace["create_splits"](cleaned, seed=42)
    splits = {"train": train, "val": val, "test": test}
    param_grid = {
        "n_estimators": [5], "max_depth": [2, 4],
        "min_samples_leaf": [1, 2], "max_features": [1.0],
    }
    monkeypatch.setitem(namespace, "PARAM_GRID", param_grid)
    encoded, fit_inputs, checkpoint_sizes = {}, [], []
    original_encode = namespace["one_hot_encode"]
    original_fit = RandomForestRegressor.fit

    def record_encode(values, *args, **kwargs):
        values = list(values)
        names = [name for name, frame in splits.items() if values == frame["utr"].tolist()]
        assert len(names) == 1
        name = names[0]
        assert name != "test", "Grid search must not encode the held-out test set"
        assert name not in encoded
        encoded[name] = original_encode(values, *args, **kwargs)
        return encoded[name]

    def record_fit(estimator, X, y, *args, **kwargs):
        if fit_inputs:
            # Finished candidates must survive interruption of the next fit.
            checkpoint = pd.read_csv(output_dir / "cv_results.csv")
            assert len(checkpoint) == len(fit_inputs)
            assert np.isfinite(checkpoint["validation_rmse"]).all()
            assert set(param_grid) <= set(checkpoint.columns)
            assert checkpoint.loc[checkpoint["validation_rmse"].idxmin(), "rank"] == 1
            checkpoint_sizes.append(len(checkpoint))
        fit_inputs.append((np.array(X, copy=True), np.array(y, copy=True)))
        return original_fit(estimator, X, y, *args, **kwargs)

    monkeypatch.setitem(namespace, "one_hot_encode", record_encode)
    monkeypatch.setattr(RandomForestRegressor, "fit", record_fit)
    output_dir = tmp_path / "grid_results"
    monkeypatch.setattr(sys, "argv", [
        "grid_search_rf.py", "--data-path", str(data_path),
        "--split-seed", "42", "--model-seed", "23", "--n-jobs", "1",
        "--output-dir", str(output_dir),
    ])
    program["main"]()

    assert set(encoded) == {"train", "val"}
    assert len(fit_inputs) == len(ParameterGrid(param_grid)) + 1
    assert checkpoint_sizes == list(range(1, len(ParameterGrid(param_grid)) + 1))
    expected_y = train["rl"].to_numpy(dtype="float32")
    for fitted_X, fitted_y in fit_inputs:
        np.testing.assert_array_equal(fitted_X, encoded["train"].reshape(len(train), -1))
        np.testing.assert_array_equal(fitted_y, expected_y)
    assert {path.name for path in output_dir.iterdir()} == {
        "cv_results.csv", "random_forest.joblib", "validation_predictions.csv", "summary.txt",
    }
    results = pd.read_csv(output_dir / "cv_results.csv")
    assert len(results) == len(ParameterGrid(param_grid))
    assert np.isfinite(results["validation_rmse"]).all()
    best_row = results.loc[results["validation_rmse"].idxmin()]
    assert best_row["rank"] == 1
    restored = SklearnAdapter.load(output_dir / "random_forest.joblib")
    assert restored.estimator.random_state == 23
    for key in param_grid:
        assert restored.estimator.get_params()[key] == best_row[key]
    predictions = pd.read_csv(output_dir / "validation_predictions.csv")
    np.testing.assert_array_equal(predictions["source_row"], val["source_row"])
    assert predictions["utr"].tolist() == val["utr"].tolist()
    np.testing.assert_allclose(predictions["rl"], val["rl"])
    np.testing.assert_allclose(predictions["prediction"], restored.predict(encoded["val"]))
    rmse = np.sqrt(np.mean((predictions["prediction"] - val["rl"].to_numpy(dtype="float32")) ** 2))
    assert rmse == pytest.approx(best_row["validation_rmse"])
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    assert "search: GridSearchCV" in summary
    assert f"candidates: {len(ParameterGrid(param_grid))}" in summary
    assert "best_params:" in summary
    assert "test_evaluated: False" in summary
    assert all(column in summary for column in ("pearson_r_squared", "rmse"))
    assert any("validation" in line.split() for line in summary.splitlines())
