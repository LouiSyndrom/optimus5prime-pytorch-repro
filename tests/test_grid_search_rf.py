"""Check the four-candidate RF search and its validation-only outputs."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter


@pytest.fixture
def search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return import_module("grid_search_rf")


def test_search_fits_train_only_and_keeps_first_tied_winner(search, tmp_path, monkeypatch):
    # These four sequences require exactly two splits to recover their labels.
    bases = np.zeros((4, 50), dtype=int)
    bases[:, :2] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    patterns = np.eye(4, dtype="float32")[bases]
    X_train = np.tile(patterns, (100, 1, 1))
    y_train = np.tile(np.array([1, 3, 7, 9], dtype="float32"), 100)
    X_val = patterns[::-1].copy()
    y_val = np.array([9.1, 6.9, 3.1, 0.9], dtype="float32")
    candidates = [{"n_estimators": 5, "max_depth": depth} for depth in (1, 2, 3)]
    fitted = []
    original_fit = RandomForestRegressor.fit

    def record_fit(estimator, X, y, **kwargs):
        np.testing.assert_array_equal(X, X_train.reshape(len(X_train), -1))
        np.testing.assert_array_equal(y, y_train)
        if fitted:
            checkpoint = pd.read_csv(tmp_path / "validation_results.csv")
            assert checkpoint["max_depth"].tolist() == [model.max_depth for model in fitted]
            assert np.isfinite(checkpoint["validation_rmse"]).all()
        fitted.append(estimator)
        return original_fit(estimator, X, y, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", record_fit)
    winner, results = search.search_candidates(
        X_train, y_train, X_val, y_val,
        candidates=candidates, model_seed=23, n_jobs=1, output_dir=tmp_path,
    )

    assert [model.max_depth for model in fitted] == [1, 2, 3]
    assert winner.estimator is fitted[1]
    errors = results["validation_rmse"].to_numpy()
    assert errors[0] > errors[1] == errors[2]
    for estimator, row in zip(fitted, results.itertuples(), strict=True):
        predictions = estimator.predict(X_val.reshape(len(X_val), -1))
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
        pearson_r = np.corrcoef(y_val, predictions)[0, 1]
        assert row.validation_rmse == pytest.approx(rmse)
        assert row.pearson_r == pytest.approx(pearson_r)
        assert row.pearson_r_squared == pytest.approx(pearson_r ** 2)
        assert row.fit_seconds >= 0


def test_main_runs_four_candidates_and_saves_winning_model(search, tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(29)
    bases = rng.choice(list("ACGT"), size=(160, 50))
    data_path = tmp_path / "synthetic.csv"
    pd.DataFrame({
        "utr": ["".join(row) for row in bases],
        "rl": 2 + 3 * (bases[:, 0] == "A") + rng.normal(0, 0.2, len(bases)),
        "total_reads": np.arange(len(bases)) + 100,
    }).to_csv(data_path, index=False)
    cleaned, _ = clean_data(load_data(data_path))
    train, val, _ = create_splits(cleaned, seed=17)
    X_val = one_hot_encode(val["utr"].tolist())
    y_val = val["rl"].to_numpy(dtype="float32")
    baseline = SklearnAdapter(RandomForestRegressor(n_estimators=3, random_state=42))
    baseline.fit(one_hot_encode(train["utr"].tolist()), train["rl"].to_numpy(dtype="float32"))
    baseline_path = tmp_path / "previous.joblib"
    baseline.save(baseline_path)
    previous_rmse = np.sqrt(np.mean((baseline.predict(X_val) - y_val) ** 2))

    encoded, fits = [], []
    original_fit = RandomForestRegressor.fit

    def record_encode(sequences):
        encoded.append(list(sequences))
        return one_hot_encode(sequences)

    def record_fit(estimator, X, y, **kwargs):
        fits.append((estimator.n_estimators, estimator.max_depth))
        assert estimator.min_samples_leaf == 20
        assert estimator.max_features == 1.0
        assert estimator.bootstrap is True
        assert estimator.max_samples == 0.8
        assert estimator.max_leaf_nodes == 10_000
        assert estimator.random_state == 23
        assert estimator.n_jobs == 1
        return original_fit(estimator, X, y, **kwargs)

    monkeypatch.setattr(search, "one_hot_encode", record_encode)
    monkeypatch.setattr(RandomForestRegressor, "fit", record_fit)
    results_dir = tmp_path / "search"
    output = results_dir / "rf_grid4_seed17"
    monkeypatch.setattr(search, "DATA_PATH", data_path)
    monkeypatch.setattr(search, "SPLIT_SEED", 17)
    monkeypatch.setattr(search, "MODEL_SEED", 23)
    monkeypatch.setattr(search, "N_JOBS", 1)
    monkeypatch.setattr(search, "BASELINE_MODEL", baseline_path)
    monkeypatch.setattr(search, "RESULTS_DIR", results_dir)
    search.main()

    assert fits == [(50, 16), (100, 16), (50, 24), (100, 24)]
    assert encoded == [train["utr"].tolist(), val["utr"].tolist()]
    assert {path.name for path in output.iterdir()} == {
        "validation_results.csv", "random_forest.joblib", "validation_predictions.csv", "summary.txt",
    }
    results = pd.read_csv(output / "validation_results.csv")
    assert results["candidate"].tolist() == [1, 2, 3, 4]
    assert list(zip(results["n_estimators"], results["max_depth"])) == fits
    best = results.loc[results["validation_rmse"].idxmin()]
    restored = SklearnAdapter.load(output / "random_forest.joblib")
    assert (restored.estimator.n_estimators, restored.estimator.max_depth) == (best.n_estimators, best.max_depth)
    predictions = pd.read_csv(output / "validation_predictions.csv")
    columns = ["source_row", "utr", "rl"]
    pd.testing.assert_frame_equal(predictions[columns], val[columns], check_dtype=False)
    np.testing.assert_allclose(predictions["prediction"], restored.predict(X_val))
    rmse = np.sqrt(np.mean((predictions["prediction"] - y_val) ** 2))
    assert rmse == pytest.approx(best.validation_rmse)
    lines = (output / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split(":")[0] for line in lines] == ["Parameters", "Training samples", "Validation", "Test"]
    assert f"n_estimators={restored.estimator.n_estimators}" in lines[0]
    assert f"max_depth={restored.estimator.max_depth}" in lines[0]
    assert lines[1] == f"Training samples: {len(train)}"
    assert lines[2].startswith(f"Validation: n={len(val)}, RMSE {rmse:.6f}, Pearson r ")
    assert lines[3] == "Test: not evaluated"
    assert f"Previous model validation: RMSE {previous_rmse:.6f}" in capsys.readouterr().out

    monkeypatch.setattr(search, "CANDIDATES", ({"n_estimators": 2, "max_depth": 3},))
    monkeypatch.setattr(search, "BASELINE_MODEL", None)
    search.main()
    assert list(results_dir.iterdir()) == [output]
    repeated = pd.read_csv(output / "validation_results.csv")
    assert repeated[["candidate", "n_estimators", "max_depth"]].values.tolist() == [[1, 2, 3]]
    assert (output / "summary.txt").read_text().splitlines()[0] == "Parameters: n_estimators=2, max_depth=3"
    new_model = SklearnAdapter.load(output / "random_forest.joblib")
    assert (new_model.estimator.n_estimators, new_model.estimator.max_depth) == (2, 3)
    new_predictions = pd.read_csv(output / "validation_predictions.csv")
    np.testing.assert_allclose(new_predictions["prediction"], new_model.predict(X_val))
