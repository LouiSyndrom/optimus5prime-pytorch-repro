# Adapted from feature/ml-baselines by Zehao.

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import RANDOM_FOREST_PARAMS, RIDGE_PARAMS, SEED, XGBOOST_PARAMS


def build_baseline_model(model_name):
    if model_name == "ridge":
        return Ridge(**RIDGE_PARAMS)

    if model_name == "random_forest":
        return RandomForestRegressor(
            **RANDOM_FOREST_PARAMS,
            random_state=SEED,
            n_jobs=2,
        )

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **XGBOOST_PARAMS,
            random_state=SEED,
            n_jobs=2,
        )

    raise ValueError(f"Unknown baseline model: {model_name}")


def save_baseline_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline_model(path):
    return joblib.load(path)
