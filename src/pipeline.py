from pathlib import Path

import pandas as pd

from src.config import DATA_FILE, MODEL_NAMES, RESULTS_DIR
from src.data import create_data_splits, load_data
from src.evaluation import evaluate_saved_model
from src.training import train_models


def load_pipeline_data(data_path=DATA_FILE):
    data = load_data(data_path)
    return create_data_splits(data, RESULTS_DIR / "split_indexes.npz")


def train_selected_models(model_names, splits):
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
    return train_models(model_names, splits, RESULTS_DIR)


def run_inference(model_names, splits):
    all_metrics = []
    for model_name in model_names:
        metrics = evaluate_saved_model(model_name, splits, RESULTS_DIR)
        all_metrics.append(metrics)

    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comparison


def load_predictions(model_name, dataset_name):
    path = RESULTS_DIR / model_name / f"{dataset_name}_predictions.csv"
    return pd.read_csv(path)


def load_cnn_history():
    return pd.read_csv(Path(RESULTS_DIR) / "cnn/training_history.csv")
