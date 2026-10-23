"""Small synthetic checks for fixed-parameter Random Forest training."""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ridge.models.sklearn_adapter import SklearnAdapter


# Load the standalone entry point without running real-data training.
program = runpy.run_path(Path(__file__).resolve().parents[1] / "main_rf.py")
regression_metrics = program["regression_metrics"]


def _sample_data():
    rng = np.random.default_rng(17)
    X = np.eye(4, dtype=np.float32)[rng.integers(0, 4, size=(48, 4))]
    y = 2 + 3 * X[:, 0, 0] * X[:, 2, 3] - X[:, 3, 1]
    return X, y


def _estimator():
    return RandomForestRegressor(
        n_estimators=9,
        max_depth=4,
        min_samples_leaf=2,
        max_features=0.75,
        max_samples=0.8,
        random_state=17,
        n_jobs=1,
    )


def test_adapter_matches_manually_flattened_random_forest():
    X, y = _sample_data()
    model = SklearnAdapter(_estimator(), sequence_length=4).fit(X[:40], y[:40])
    reference = _estimator().fit(X[:40].reshape(40, 16, order="C"), y[:40])
    held_out = X[[47, 40, 44, 41]]

    predictions = model.predict(held_out)
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(
        predictions, reference.predict(held_out.reshape(4, 16, order="C"))
    )


def test_random_forest_save_load_preserves_predictions_and_config(tmp_path):
    X, y = _sample_data()
    model = SklearnAdapter(_estimator(), sequence_length=4).fit(X[:40], y[:40])
    path = tmp_path / "random_forest.joblib"
    model.save(path)
    restored = SklearnAdapter.load(path)

    assert isinstance(restored.estimator, RandomForestRegressor)
    assert restored.get_config() == model.get_config()
    np.testing.assert_allclose(restored.predict(X[40:]), model.predict(X[40:]))


def test_validation_labels_do_not_change_a_fixed_candidate():
    X, y = _sample_data()
    original = SklearnAdapter(_estimator(), sequence_length=4).fit(
        X[:40], y[:40], X_val=X[40:], y_val=y[40:]
    )
    changed = SklearnAdapter(_estimator(), sequence_length=4).fit(
        X[:40], y[:40], X_val=X[40:], y_val=y[40:] + 1_000_000
    )

    np.testing.assert_array_equal(original.predict(X), changed.predict(X))


def test_pearson_sign_squared_correlation_rmse_and_constant_predictions():
    metrics = regression_metrics([1, 2, 3], [2, 4, 6])
    assert metrics["pearson_r"] == pytest.approx(1)
    assert metrics["pearson_r_squared"] == pytest.approx(1)
    assert metrics["rmse"] == pytest.approx(np.sqrt(14 / 3))

    negative = regression_metrics([1, 2, 3], [3, 2, 1])
    assert negative["pearson_r"] == pytest.approx(-1)
    assert negative["pearson_r_squared"] == pytest.approx(1)

    for y_true, predictions in (([1, 2, 3], [2, 2, 2]), ([2, 2, 2], [1, 2, 3])):
        constant = regression_metrics(y_true, predictions)
        assert np.isnan(constant["pearson_r"])
        assert np.isnan(constant["pearson_r_squared"])
        assert constant["rmse"] == pytest.approx(np.sqrt(2 / 3))
        assert constant["pearson_note"]


def test_entrypoint_fits_fixed_parameters_once_and_delays_test_until_save(
    tmp_path, monkeypatch,
):
    """Exercise the direct entry point with 100 trees and the complete data splits."""
    entrypoint = runpy.run_path(Path(__file__).resolve().parents[1] / "main_rf.py")
    namespace = entrypoint["main"].__globals__
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
    train, val, test = namespace["create_splits"](cleaned, seed=23)
    splits = {"train": train, "val": val, "test": test}
    expected_params = {
        "n_estimators": 100, "max_depth": 24, "min_samples_leaf": 20,
        "max_features": 1.0, "max_leaf_nodes": 10_000, "bootstrap": True,
        "max_samples": 0.8, "criterion": "squared_error", "random_state": 23,
        "n_jobs": 1,
    }
    events, encoded, fit_inputs, saved = [], {}, [], {}
    original_encode = namespace["one_hot_encode"]
    original_fit = RandomForestRegressor.fit
    original_save = SklearnAdapter.save
    original_predict = SklearnAdapter.predict

    def record_encode(values, *args, **kwargs):
        values = list(values)
        names = [name for name, frame in splits.items() if values == frame["utr"].tolist()]
        assert len(names) == 1
        name = names[0]
        events.append(f"encode_{name}")
        encoded[name] = (list(values), original_encode(values, *args, **kwargs))
        return encoded[name][1]

    def record_fit(estimator, X, y, *args, **kwargs):
        for key, value in expected_params.items():
            assert estimator.get_params()[key] == value
        events.append("fit")
        fit_inputs.append((np.array(X, copy=True), np.array(y, copy=True)))
        model = original_fit(estimator, X, y, *args, **kwargs)
        events.append("fit_complete")
        return model

    def record_save(model, path):
        original_save(model, path)
        saved["model"] = model
        events.append("saved")

    def record_predict(model, X):
        names = [
            name for name, (_, values) in encoded.items()
            if np.shares_memory(X, values)
        ]
        assert len(names) == 1
        name = names[0]
        predictions = original_predict(model, X)
        events.append(f"predict_{name}")
        return predictions

    monkeypatch.setitem(namespace, "one_hot_encode", record_encode)
    monkeypatch.setattr(RandomForestRegressor, "fit", record_fit)
    monkeypatch.setattr(SklearnAdapter, "save", record_save)
    monkeypatch.setattr(SklearnAdapter, "predict", record_predict)
    output_dir = tmp_path / "rf_results"
    monkeypatch.setitem(namespace, "DATA_PATH", data_path)
    monkeypatch.setitem(namespace, "OUTPUT_DIR", output_dir)
    monkeypatch.setitem(namespace, "SEED", 23)
    monkeypatch.setitem(namespace, "N_JOBS", 1)
    entrypoint["main"]()

    expected_sizes = {"train": 64, "val": 8, "test": 8}
    assert set(encoded) == set(expected_sizes)
    for name, count in expected_sizes.items():
        assert len(encoded[name][0]) == count
        assert encoded[name][0] == splits[name]["utr"].tolist()
    assert len(fit_inputs) == 1
    expected_y = train.set_index("utr").loc[encoded["train"][0], "rl"].to_numpy(dtype="float32")
    for fitted_X, fitted_y in fit_inputs:
        np.testing.assert_array_equal(fitted_X, encoded["train"][1].reshape(len(expected_y), -1))
        np.testing.assert_array_equal(fitted_y, expected_y)
    assert events.index("fit_complete") < events.index("saved") < events.index("predict_val")
    assert events.index("saved") < events.index("encode_test") < events.index("predict_test")
    assert events.count("predict_val") == events.count("predict_test") == 1
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    assert "model_params:" in summary
    assert "seed: 23" in summary
    assert "used_rows:" in summary
    assert all(column in summary for column in ("split", "pearson_r_squared", "pearson_r", "rmse"))
    assert any(line.split()[0] == "validation" for line in summary.splitlines() if line.split())
    assert all(detail not in summary for detail in (
        "candidate_results:", "fit_seconds:", "model_config:", "cleaning_removed:",
        "reload_predictions_match:", "validation_metrics:", "test_metrics:",
    ))
    assert {path.name for path in output_dir.iterdir()} == {
        "random_forest.joblib", "test_predictions.csv", "summary.txt",
    }
    output_predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert output_predictions["utr"].tolist() == test["utr"].tolist()
    assert np.isfinite(output_predictions["prediction"]).all()
    np.testing.assert_array_equal(output_predictions["source_row"], test["source_row"])
    assert any(line.split()[0] == "test" for line in summary.splitlines() if line.split())

    restored = SklearnAdapter.load(output_dir / "random_forest.joblib")
    assert restored.get_config() == saved["model"].get_config()
    assert len(restored.estimator.estimators_) == 100
    np.testing.assert_allclose(
        output_predictions["prediction"], original_predict(restored, encoded["test"][1]),
    )
