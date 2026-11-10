from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.config import BATCH_SIZE, MODEL_FILES
from src.data import make_baseline_data, make_cnn_dataset
from src.models.baselines import load_baseline_model


# Calculate prediction quality values
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mse = float(np.mean((actual - predicted) ** 2))

    # Handle arrays with no variation
    if np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(actual, predicted)[0, 1])

    return {
        "pearson_r": pearson_r,
        "pearson_r_squared": pearson_r ** 2,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "regression_r2": float(r2_score(actual, predicted)),
    }


# Predict all batches with the CNN
def predict_cnn_loader(model, loader, mrl_mean, mrl_std, device):
    import torch

    model.eval()
    actual = []
    predicted = []

    # Turn off gradients during prediction
    with torch.inference_mode():
        for sequences, mrls, _ in loader:
            predictions, _ = model(sequences.to(device))
            actual.append(mrls.numpy())
            predicted.append(predictions.cpu().numpy())

    # Return values to the original MRL scale
    actual = np.concatenate(actual) * mrl_std + mrl_mean
    predicted = np.concatenate(predicted) * mrl_std + mrl_mean
    return actual, predicted


# Build a loader and run CNN prediction
def predict_cnn(model, frame, mrl_mean, mrl_std, device):
    from torch.utils.data import DataLoader

    dataset = make_cnn_dataset(frame, mrl_mean, mrl_std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return predict_cnn_loader(model, loader, mrl_mean, mrl_std, device)


# Predict values with a baseline model
def predict_baseline(model, frame):
    features, actual = make_baseline_data(frame)
    predicted = model.predict(features)
    return actual, predicted


# Evaluate one saved model on both test sets
def evaluate_saved_model(model_name, splits, results_dir, device=None):
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    model_path = model_dir / MODEL_FILES[model_name]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load the requested model
    if model_name == "cnn":
        import torch

        from src.models.cnn import load_cnn_model

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        model, checkpoint = load_cnn_model(model_path, device)
    else:
        model = load_baseline_model(model_path)

    # Evaluate high-read and low-read data
    metric_rows = []
    for split_name in ["high_read_test", "low_read_test"]:
        frame = splits[split_name]
        if model_name == "cnn":
            actual, predicted = predict_cnn(
                model,
                frame,
                checkpoint["mrl_mean"],
                checkpoint["mrl_std"],
                device,
            )
        else:
            actual, predicted = predict_baseline(model, frame)

        metrics = calculate_metrics(actual, predicted)
        metric_rows.append(
            {"model": model_name, "dataset": split_name, "n": len(actual), **metrics}
        )

        # Save each prediction with its source sequence
        predictions = frame[["source_row", "utr", "total_reads"]].copy()
        predictions["actual_mrl"] = actual
        predictions["predicted_mrl"] = predicted
        predictions.to_csv(model_dir / f"{split_name}_predictions.csv", index=False)

    # Save the model metrics
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(model_dir / "metrics.csv", index=False)
    return metrics
