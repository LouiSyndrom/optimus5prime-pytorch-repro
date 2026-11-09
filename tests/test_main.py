"""Essential tests for parameter selection and regression metrics."""

from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

# Load the standalone entry point without depending on pytest's import path.
program = runpy.run_path(Path(__file__).resolve().parents[1] / "main_ridge.py")
regression_metrics = program["regression_metrics"]
select_alpha = program["select_alpha"]


def test_alpha_search_keeps_best_train_only_model():
    X = np.eye(4, dtype=np.float32)[[0, 1, 0, 1]][:, None, :]
    y_train = np.array([0, 2, 0, 2], dtype=np.float32)
    y_val = np.array([0.9, 1.1, 0.9, 1.1], dtype=np.float32)
    model, _ = select_alpha(X, y_train, X, y_val, [0.01, 18.0, 1000.0])

    assert model.estimator.alpha == 18.0  # Best candidate is neither first nor last.
    reference = Ridge(alpha=18.0).fit(X.reshape(4, 4), y_train)
    np.testing.assert_allclose(model.estimator.coef_, reference.coef_)
    np.testing.assert_allclose(model.estimator.intercept_, reference.intercept_)
    np.testing.assert_allclose(model.predict(X), y_val, atol=1e-6)


def test_pearson_and_rmse():
    metrics = regression_metrics([1, 2, 3], [2, 4, 6])
    assert metrics["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert metrics["pearson_r"] == pytest.approx(1)
    assert metrics["pearson_r_squared"] == pytest.approx(1)

    constant = regression_metrics([1, 2, 3], [2, 2, 2])
    assert np.isnan(constant["pearson_r"])
    assert np.isnan(constant["pearson_r_squared"])


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "validation-only"])
def test_ridge_outputs_shared_summary(tmp_path, monkeypatch, smoke):
    rng = np.random.default_rng(42)
    sequences = ["".join(row) for row in rng.choice(list("ACGT"), size=(50, 50))]
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"utr": sequences, "rl": np.arange(50), "total_reads": 100}).to_csv(
        data_path, index=False,
    )
    output = tmp_path / "results"
    root = Path(__file__).resolve().parents[1]
    script = root / ("scripts/smoke_test_ridge.py" if smoke else "main_ridge.py")
    options = ["--n-samples", "50", "--alpha", "0.5"] if smoke else ["--alphas", "0.5"]
    monkeypatch.setattr(sys, "argv", [
        str(script), "--data-path", str(data_path), "--output-dir", str(output), *options,
    ])
    runpy.run_path(str(script), run_name="__main__")

    lines = (output / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert lines[:2] == ["Parameters: alpha=0.5", "Training samples: 40"]
    for split, line in zip(("validation", "test"), lines[2:]):
        if smoke and split == "test":
            assert line == "Test: not evaluated"
            continue
        frame = pd.read_csv(output / f"{split}_predictions.csv")
        errors = frame["prediction"].to_numpy(dtype="float32").astype(float) - frame["rl"]
        rmse = np.sqrt(np.mean(errors ** 2))
        assert line.startswith(f"{split.title()}: n=5, RMSE {rmse:.6f},")
