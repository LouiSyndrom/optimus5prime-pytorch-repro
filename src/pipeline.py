import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_FILE, MODEL_NAMES, PROJECT_DIR, RESULTS_DIR
from src.data import create_data_splits, load_data


# Load the data and create shared splits
def load_pipeline_data(data_path=DATA_FILE):
    data = load_data(data_path)
    return create_data_splits(data, RESULTS_DIR / "split_indexes.npz")


# Train selected models in separate processes
def train_selected_models(model_names, splits):
    # Reload the same data inside each model process
    del splits
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
        subprocess.run(
            [sys.executable, "-m", "src.runner", "train", model_name],
            cwd=PROJECT_DIR,
            check=True,
        )


# Evaluate selected models in separate processes
def run_inference(model_names, splits):
    # Keep native ML tools out of the notebook process
    del splits
    all_metrics = []
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
        subprocess.run(
            [sys.executable, "-m", "src.runner", "evaluate", model_name],
            cwd=PROJECT_DIR,
            check=True,
        )
        all_metrics.append(pd.read_csv(RESULTS_DIR / model_name / "metrics.csv"))

    # Combine all model metrics
    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comparison


# Load saved predictions for one test set
def load_predictions(model_name, dataset_name):
    path = RESULTS_DIR / model_name / f"{dataset_name}_predictions.csv"
    return pd.read_csv(path)


# Load the saved CNN training history
def load_cnn_history():
    return pd.read_csv(Path(RESULTS_DIR) / "cnn/training_history.csv")
