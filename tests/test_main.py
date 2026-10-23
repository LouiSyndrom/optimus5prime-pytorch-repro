"""Essential tests for parameter selection and regression metrics."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# Load the standalone entry point without depending on pytest's import path.
program = runpy.run_path(Path(__file__).resolve().parents[1] / "main.py")
regression_metrics = program["regression_metrics"]
select_alpha = program["select_alpha"]


def test_alpha_search_keeps_best_train_only_model():
    X = np.eye(4, dtype=np.float32)[[0, 1, 0, 1]][:, None, :]
    y_train = np.array([0, 2, 0, 2], dtype=np.float32)
    y_val = np.array([0.9, 1.1, 0.9, 1.1], dtype=np.float32)
    model = select_alpha(X, y_train, X, y_val, [0.01, 18.0, 1000.0])

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
