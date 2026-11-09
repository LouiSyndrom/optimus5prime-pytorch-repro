"""Check Ridge validation search, frozen training, and smoke-run summaries."""

from importlib import import_module
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from ml.data.encoding import one_hot_encode
from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter


@pytest.fixture
def root(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory))
    return directory


@pytest.fixture
def data_path(tmp_path):
    rng = np.random.default_rng(42)
    sequences = ["".join(row) for row in rng.choice(list("ACGT"), size=(50, 50))]
    path = tmp_path / "data.csv"
    pd.DataFrame({"utr": sequences, "rl": np.arange(50), "total_reads": 100}).to_csv(path, index=False)
    return path


def test_alpha_search_keeps_first_tied_winner(root, tmp_path, monkeypatch):
    search = import_module("grid_search_ridge")
    X = np.eye(4, dtype=np.float32)[[0, 1, 0, 1]][:, None, :]
    y_train = np.array([0, 2, 0, 2], dtype=np.float32)
    y_val = np.array([0.9, 1.1, 0.9, 1.1], dtype=np.float32)
    candidates = [{"alpha": alpha} for alpha in (0.01, 18.0, 18.0, 1000.0)]
    fitted = []
    original_fit = Ridge.fit

    def record_fit(estimator, features, labels, **kwargs):
        np.testing.assert_array_equal(features, X.reshape(4, 4))
        np.testing.assert_array_equal(labels, y_train)
        if fitted:
            checkpoint = pd.read_csv(tmp_path / "validation_results.csv")
            assert checkpoint["alpha"].tolist() == [ridge.alpha for ridge in fitted]
        fitted.append(estimator)
        return original_fit(estimator, features, labels, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record_fit)
    model, results = search.search_candidates(X, y_train, X, y_val, candidates=candidates, output_dir=tmp_path)
    scores = results["validation_rmse"].to_numpy()
    assert len(fitted) == len(candidates)
    assert scores[0] > scores[1] == scores[2] < scores[3]
    assert model.estimator is fitted[1]
    np.testing.assert_allclose(model.predict(X), y_val, atol=1e-6)


def test_search_exports_six_scores_and_winner_validation_rows(root, data_path, tmp_path, monkeypatch):
    search = import_module("grid_search_ridge")
    cleaned, _ = clean_data(load_data(data_path))
    train, val, _ = create_splits(cleaned, seed=42)
    X_train, X_val = (one_hot_encode(frame["utr"].tolist()) for frame in (train, val))
    fits = []
    original_fit, original_predict = SklearnAdapter.fit, SklearnAdapter.predict

    def track_fit(model, X, y, **kwargs):
        np.testing.assert_array_equal(X, X_train)
        np.testing.assert_array_equal(y, train["rl"].to_numpy(dtype="float32"))
        fits.append(model.estimator.alpha)
        return original_fit(model, X, y, **kwargs)

    def guard_predict(model, X):
        np.testing.assert_array_equal(X, X_val)
        return original_predict(model, X)

    monkeypatch.setattr(SklearnAdapter, "fit", track_fit)
    monkeypatch.setattr(SklearnAdapter, "predict", guard_predict)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(search, "DATA_PATH", data_path)
    monkeypatch.setattr(search, "RESULTS_DIR", results_dir)
    monkeypatch.chdir(tmp_path)
    search.main()

    output = results_dir / "ridge_search_seed42"
    assert fits == [0.1, 1, 10, 100, 1000, 10000]
    assert {path.name for path in output.iterdir()} == {
        "validation_results.csv", "validation_predictions.csv", "summary.txt",
    }
    scores = pd.read_csv(output / "validation_results.csv")
    assert scores["alpha"].tolist() == fits
    assert scores["candidate"].tolist() == list(range(1, 7))
    predictions = pd.read_csv(output / "validation_predictions.csv")
    columns = ["source_row", "utr", "rl"]
    pd.testing.assert_frame_equal(predictions[columns], val[columns], check_dtype=False)
    y_val = val["rl"].to_numpy(dtype="float32").astype(float)
    y_pred = predictions["prediction"].to_numpy(dtype="float32").astype(float)
    rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
    winner = scores.loc[scores["validation_rmse"].idxmin()]
    assert winner.validation_rmse == pytest.approx(rmse)
    pearson_r = np.corrcoef(y_val, y_pred)[0, 1]
    assert winner.pearson_r == pytest.approx(pearson_r)
    assert winner.pearson_r_squared == pytest.approx(pearson_r ** 2)
    lines = (output / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert float(lines[0].removeprefix("Parameters: alpha=")) == winner.alpha
    assert lines[1] == f"Training samples: {len(train)}"
    assert lines[2].startswith(f"Validation: n={len(val)}, RMSE {rmse:.6f},")
    assert lines[3] == "Test: not evaluated"

    monkeypatch.setattr(search, "CANDIDATES", [{"alpha": 3.5}])
    search.main()
    assert fits[-1] == 3.5
    assert list(results_dir.iterdir()) == [output]
    updated_scores = pd.read_csv(output / "validation_results.csv")
    assert updated_scores["alpha"].tolist() == [3.5]
    assert (output / "summary.txt").read_text().splitlines()[0] == "Parameters: alpha=3.5"
    updated = pd.read_csv(output / "validation_predictions.csv")
    rmse = np.sqrt(np.mean((updated["prediction"] - y_val) ** 2))
    assert rmse == pytest.approx(updated_scores.loc[0, "validation_rmse"], abs=1e-12)


def test_pearson_and_rmse(root):
    regression_metrics = import_module("main_ridge").regression_metrics
    metrics = regression_metrics([1, 2, 3], [2, 4, 6])
    assert metrics["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert metrics["pearson_r"] == pytest.approx(1)
    assert metrics["pearson_r_squared"] == pytest.approx(1)

    constant = regression_metrics([1, 2, 3], [2, 2, 2])
    assert np.isnan(constant["pearson_r"])
    assert np.isnan(constant["pearson_r_squared"])


@pytest.mark.parametrize("smoke,alpha", [(False, 0.5), (False, None), (True, 0.5)],
                         ids=["final-override", "final-default", "smoke"])
def test_ridge_outputs_shared_summary(root, data_path, tmp_path, monkeypatch, smoke, alpha):
    output = tmp_path / "results"
    script = root / ("scripts/smoke_test_ridge.py" if smoke else "main_ridge.py")
    options = ["--n-samples", "50"] if smoke else []
    if alpha is not None:
        options.extend(["--alpha", str(alpha)])
    expected_alpha = 1000.0 if alpha is None else alpha
    fitted_alphas = []
    original_fit = Ridge.fit

    def record_fit(estimator, X, y, **kwargs):
        fitted_alphas.append(estimator.alpha)
        return original_fit(estimator, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record_fit)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--data-path", str(data_path), "--output-dir", str(output), *options,
    ])
    runpy.run_path(str(script), run_name="__main__")

    assert fitted_alphas == [expected_alpha]
    files = {"summary.txt", "ridge.joblib", "validation_predictions.csv"}
    if not smoke:
        files.update(["metrics.csv", "test_predictions.csv"])
    assert {path.name for path in output.iterdir()} == files
    lines = (output / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert float(lines[0].removeprefix("Parameters: alpha=")) == expected_alpha
    assert lines[1] == "Training samples: 40"
    for split, line in zip(("validation", "test"), lines[2:]):
        if smoke and split == "test":
            assert line == "Test: not evaluated"
            continue
        frame = pd.read_csv(output / f"{split}_predictions.csv")
        errors = frame["prediction"].to_numpy(dtype="float32").astype(float) - frame["rl"]
        rmse = np.sqrt(np.mean(errors ** 2))
        assert line.startswith(f"{split.title()}: n=5, RMSE {rmse:.6f},")
