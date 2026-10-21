"""Adapt a single-target sklearn regressor to the shared One-Hot input."""

import json
import platform
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.utils.validation import check_is_fitted

_SCHEMA_VERSION = 1
_FEATURES = {
    "encoding": "one_hot",
    "base_order": "ACGT",
    "flatten_order": "C",
    "target_unit": "MRL",
}


def _validate_vector(values, n_samples, name):
    values = np.asarray(values)
    if (
        values.shape != (n_samples,)
        or values.dtype.kind not in "biuf"
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must contain finite real numbers with shape ({n_samples},)")
    return values


class SklearnAdapter:
    """Flatten sequence features without scaling or selecting hyperparameters."""

    def __init__(self, estimator, *, sequence_length: int = 50):
        if (
            not isinstance(sequence_length, Integral)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError("sequence_length must be a positive integer")
        self.estimator = estimator
        self.sequence_length = int(sequence_length)

    def _flatten(self, X, *, allow_empty=False):
        X = np.asarray(X)
        if (
            X.ndim != 3
            or X.shape[1:] != (self.sequence_length, 4)
            or (X.shape[0] == 0 and not allow_empty)
            or X.dtype.kind not in "biuf"
            or not np.isfinite(X).all()
        ):
            raise ValueError(
                f"X must be finite real numbers with shape (N, {self.sequence_length}, 4)"
                + ("" if allow_empty else " and N > 0")
            )
        # Explicit width also supports an empty validation set.
        return X.reshape(X.shape[0], 4 * self.sequence_length, order="C")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SklearnAdapter":
        """Train on X/y only; validation inputs are checked but never fitted."""
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together")
        X_flat = self._flatten(X)
        y = _validate_vector(y, X_flat.shape[0], "y")
        if X_val is not None:
            val_flat = self._flatten(X_val, allow_empty=True)
            _validate_vector(y_val, val_flat.shape[0], "y_val")
        self.estimator.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return finite predictions in the original MRL units and sample order."""
        check_is_fitted(self.estimator)
        X_flat = self._flatten(X)
        predictions = self.estimator.predict(X_flat)
        return _validate_vector(predictions, X_flat.shape[0], "predictions")

    def get_config(self) -> dict:
        """Return JSON-compatible metadata; unsupported parameter types raise."""
        def json_value(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(f"Unsupported configuration type: {type(value).__name__}")

        config = {
            "adapter_type": type(self).__name__,
            "estimator_type": type(self.estimator).__name__,
            "estimator_module": type(self.estimator).__module__,
            "estimator_params": self.estimator.get_params(deep=False),
            "sequence_length": self.sequence_length,
            **_FEATURES,
        }
        return json.loads(json.dumps(config, default=json_value, allow_nan=False))

    def save(self, path: str | Path) -> None:
        """Save a fitted estimator together with its feature conventions."""
        check_is_fitted(self.estimator)
        package = {
            "schema_version": _SCHEMA_VERSION,
            "estimator": self.estimator,
            **self.get_config(),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(package, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnAdapter":
        """Load only trusted joblib files; cross-version compatibility is not guaranteed."""
        package = joblib.load(path)
        if not isinstance(package, dict) or package.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported model package schema")
        if any(package.get(key) != value for key, value in _FEATURES.items()):
            raise ValueError("Incompatible feature conventions")
        if not {"estimator", "sequence_length", "software_versions"} <= package.keys():
            raise ValueError("Incomplete model package")

        model = cls(package["estimator"], sequence_length=package["sequence_length"])
        check_is_fitted(model.estimator)
        if any(package.get(key) != value for key, value in model.get_config().items()):
            raise ValueError("Model metadata does not match the estimator configuration")
        if getattr(model.estimator, "n_features_in_", None) != 4 * model.sequence_length:
            raise ValueError("Model feature count does not match sequence_length")
        return model
