"""Exercise the XGBoost adapter and the fixed-validation search."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.io import clean_data, load_data
from ml.data.split import create_splits
from ml.models.sklearn_adapter import SklearnAdapter

xgboost = pytest.importorskip("xgboost")


@pytest.fixture
def programs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return import_module("main_xgb"), import_module("grid_search_xgb")


def test_adapter_matches_direct_fit_and_reloaded_model(programs, tmp_path):
    main, _ = programs
    rng = np.random.default_rng(17)
    X = np.eye(4, dtype="float32")[rng.integers(0, 4, (96, 50))]
    y = 2 + 3 * X[:, 0, 0] - X[:, 9, 1]
    params = dict(n_estimators=12, max_depth=3, model_seed=17, n_jobs=1)
    model = main.build_model(**params).fit(X[:80], y[:80])
    reference = xgboost.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", learning_rate=0.05,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_estimators=12, max_depth=3, random_state=17, n_jobs=1,
    ).fit(X[:80].reshape(80, 200, order="C"), y[:80])
    expected = reference.predict(X[80:].reshape(16, 200, order="C"))
    assert np.ptp(expected) > 0
    np.testing.assert_allclose(model.predict(X[80:]), expected)

    path = tmp_path / "xgboost.joblib"
    model.save(path)
    restored = SklearnAdapter.load(path)
    assert isinstance(restored.estimator, xgboost.XGBRegressor)
    np.testing.assert_allclose(restored.predict(X[80:]), expected)


def test_search_fits_train_only_and_keeps_first_tied_winner(programs, monkeypatch, tmp_path):
    _, search = programs
    # Features have exactly two possible rows, so depths one and two tie.
    patterns = np.eye(4, dtype="float32")[[[0] * 50, [1] * 50]]
    X_train = np.tile(patterns, (32, 1, 1))
    y_train = np.tile(np.array([2, 8], dtype="float32"), 32)
    X_val = patterns[::-1].copy()
    y_val = np.array([8.1, 1.9], dtype="float32")
    candidates = [{"n_estimators": 1, "max_depth": 1},
                  {"n_estimators": 20, "max_depth": 1},
                  {"n_estimators": 20, "max_depth": 2}]
    fits = []
    original_fit = xgboost.XGBRegressor.fit

    def record_fit(estimator, X, y, **kwargs):
        np.testing.assert_array_equal(X, X_train.reshape(64, 200))
        np.testing.assert_array_equal(y, y_train)
        assert kwargs.get("eval_set") is None
        if fits:
            checkpoint = pd.read_csv(tmp_path / "validation_results.csv")
            assert len(checkpoint) == len(fits)
        fits.append((estimator.n_estimators, estimator.max_depth))
        return original_fit(estimator, X, y, **kwargs)

    monkeypatch.setattr(xgboost.XGBRegressor, "fit", record_fit)
    winner, results = search.search_candidates(
        X_train, y_train, X_val, y_val, candidates=candidates,
        model_seed=42, n_jobs=1, output_dir=tmp_path,
    )
    assert fits == [(1, 1), (20, 1), (20, 2)]
    errors = results["validation_rmse"].to_numpy()
    assert errors[0] > errors[1] == errors[2]
    assert (winner.estimator.n_estimators, winner.estimator.max_depth) == (20, 1)
    predictions = winner.predict(X_val)
    rmse = np.sqrt(np.mean((predictions.astype(float) - y_val) ** 2))
    assert errors[1] == pytest.approx(rmse)


def test_direct_run_uses_full_existing_splits_and_exports_validation_rows(
    programs, monkeypatch, tmp_path,
):
    main, search = programs
    rng = np.random.default_rng(29)
    bases = rng.choice(list("ACGT"), size=(100, 50))
    data_path = tmp_path / "sequences.csv"
    pd.DataFrame({
        "utr": ["".join(row) for row in bases],
        "rl": 2 + 3 * (bases[:, 0] == "A") + rng.normal(0, 0.2, 100),
        "total_reads": np.arange(100),
    }).to_csv(data_path, index=False)
    cleaned, _ = clean_data(load_data(data_path))
    train, val, _ = create_splits(cleaned, seed=42)
    encoded = []
    original_encode = main.one_hot_encode

    def record_encode(sequences):
        encoded.append(list(sequences))
        return original_encode(sequences)

    monkeypatch.setattr(main, "one_hot_encode", record_encode)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(search, "DATA_PATH", data_path)
    monkeypatch.setattr(search, "RESULTS_DIR", results_dir)
    monkeypatch.chdir(tmp_path)
    search.main()

    output, = results_dir.glob("xgb_search_seed42_*")
    assert encoded == [train["utr"].tolist(), val["utr"].tolist()]
    results = pd.read_csv(output / "validation_results.csv")
    assert list(zip(results["n_estimators"], results["max_depth"])) == [
        (200, 3), (400, 3), (200, 6), (400, 6),
    ]
    predictions = pd.read_csv(output / "validation_predictions.csv")
    assert predictions["source_row"].tolist() == val["source_row"].tolist()
    assert predictions["utr"].tolist() == val["utr"].tolist()
    np.testing.assert_allclose(predictions["rl"], val["rl"])
    rmse = np.sqrt(np.mean((predictions["prediction"] - val["rl"].to_numpy("float32")) ** 2))
    assert rmse == pytest.approx(results["validation_rmse"].min())
    assert not (output / "test_predictions.csv").exists()

    original_summary = (output / "summary.txt").read_bytes()
    lines = original_summary.decode("utf-8").splitlines()
    assert [line.split(":")[0] for line in lines] == ["Parameters", "Training samples", "Validation", "Test"]
    assert f"RMSE {rmse:.6f}" in lines[2]
    assert lines[3] == "Test: not evaluated"
    search.main()
    next_output, = set(results_dir.glob("xgb_search_seed42_*")) - {output}
    assert (next_output / "summary.txt").is_file()
    assert (output / "summary.txt").read_bytes() == original_summary


@pytest.fixture
def final_run(programs, monkeypatch, tmp_path):
    main, search = programs
    rng = np.random.default_rng(51)
    bases = rng.choice(list("ACGT"), size=(160, 50))
    data_path = tmp_path / "sequences.csv"
    pd.DataFrame({
        "utr": ["".join(row) for row in bases],
        "rl": 3 + (bases[:, 0] == "G") + rng.normal(0, 0.1, len(bases)),
        "total_reads": np.ones(len(bases), dtype=int),
    }).to_csv(data_path, index=False)
    cleaned, _ = clean_data(load_data(data_path))
    train, val, test = create_splits(cleaned, seed=42)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    model, _ = search.search_candidates(
        *main.xy(train), *main.xy(val),
        candidates=[{"n_estimators": 400, "max_depth": 6}], output_dir=search_dir,
    )
    val[["source_row", "utr", "rl"]].assign(
        prediction=model.predict(main.xy(val)[0]).astype(float),
    ).to_csv(search_dir / "validation_predictions.csv", index=False)
    monkeypatch.setattr(main, "DATA_PATH", data_path)
    monkeypatch.setattr(main, "SEARCH_DIR", search_dir)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "final")
    return main, train, val, test


def test_final_model_fits_once_and_evaluates_reloaded_model(final_run, monkeypatch):
    main, train, val, test = final_run
    X_train, y_train = main.xy(train)
    X_val, _ = main.xy(val)
    X_test, _ = main.xy(test)
    fits, predicted = [], []
    original_fit, original_predict = SklearnAdapter.fit, SklearnAdapter.predict

    def track_fit(model, X, y):
        np.testing.assert_array_equal(X, X_train)
        np.testing.assert_array_equal(y, y_train)
        assert (model.estimator.n_estimators, model.estimator.max_depth) == (400, 6)
        fits.append(model)
        return original_fit(model, X, y)

    def track_predict(model, X):
        assert len(fits) == 1 and model is not fits[0]
        predicted.append(X)
        return original_predict(model, X)

    monkeypatch.setattr(SklearnAdapter, "fit", track_fit)
    monkeypatch.setattr(SklearnAdapter, "predict", track_predict)
    main.main()
    assert len(predicted) == 2
    np.testing.assert_array_equal(predicted[0], X_val)
    np.testing.assert_array_equal(predicted[1], X_test)
    output = main.OUTPUT_DIR
    assert {p.name for p in output.iterdir()} == {"xgboost.joblib", "test_predictions.csv", "summary.txt"}
    exported = pd.read_csv(output / "test_predictions.csv")
    lines = (output / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split(":")[0] for line in lines] == ["Parameters", "Training samples", "Validation", "Test"]
    assert lines[2].startswith(f"Validation: n={len(val):,}, RMSE ")
    assert lines[3].startswith(f"Test: n={len(test):,}, RMSE ")
    pd.testing.assert_frame_equal(exported[["source_row", "utr", "rl"]], test[["source_row", "utr", "rl"]],
                                  check_dtype=False)
    restored = SklearnAdapter.load(output / "xgboost.joblib")
    np.testing.assert_allclose(exported["prediction"], original_predict(restored, X_test))


@pytest.mark.parametrize("mismatch", ["prediction", "source_row", "validation_rmse"])
def test_final_run_stops_before_test_if_validation_differs(final_run, monkeypatch, mismatch):
    main, _, _, test = final_run
    filename = "validation_results.csv" if mismatch == "validation_rmse" else "validation_predictions.csv"
    reference = main.SEARCH_DIR / filename
    frame = pd.read_csv(reference)
    frame.loc[0, mismatch] += 1
    frame.to_csv(reference, index=False)

    original_xy = main.xy

    def guard_test(frame):
        if frame["source_row"].tolist() == test["source_row"].tolist():
            pytest.fail("Test encoding started before the validation match")
        return original_xy(frame)

    monkeypatch.setattr(main, "xy", guard_test)
    with pytest.raises((ValueError, AssertionError)):
        main.main()
    assert not main.OUTPUT_DIR.exists()
