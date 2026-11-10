import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AUXILIARY_LOSS_WEIGHT,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_FILES,
    SEED,
    WEIGHT_DECAY,
)
from src.data import make_baseline_data, make_cnn_dataset
from src.models.baselines import build_baseline_model, save_baseline_model


# Train and save one baseline model
def train_baseline(model_name, splits, results_dir):
    # Prepare training features and labels
    features, labels = make_baseline_data(splits["training"])
    model = build_baseline_model(model_name)
    model.fit(features, labels)

    model_path = Path(results_dir) / model_name / MODEL_FILES[model_name]
    save_baseline_model(model, model_path)
    return model


# Train the CNN for one full epoch
def run_cnn_epoch(model, loader, optimizer, device):
    import torch
    from torch import nn

    model.train()
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    total_loss = 0.0

    # Update the model with each batch
    for sequences, mrls, fractions in loader:
        sequences = sequences.to(device)
        mrls = mrls.to(device)
        fractions = fractions.to(device)

        # Calculate the main and auxiliary losses
        optimizer.zero_grad()
        predictions, fraction_scores = model(sequences)
        loss = mse_loss(predictions, mrls)
        loss += AUXILIARY_LOSS_WEIGHT * kl_loss(
            torch.log_softmax(fraction_scores, dim=1),
            fractions,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(mrls)

    return total_loss / len(loader.dataset)


# Train the CNN with validation and early stopping
def train_cnn(splits, results_dir, device=None):
    import torch
    from torch.utils.data import DataLoader

    from src.evaluation import calculate_metrics, predict_cnn_loader
    from src.models.cnn import MultiscaleCNN, save_cnn_model

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Choose the best available device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    training = splits["training"]
    validation = splits["validation"]
    mrl_mean = float(training["rl"].mean())
    mrl_std = float(training["rl"].std(ddof=0))

    # Build training and validation loaders
    training_set = make_cnn_dataset(training, mrl_mean, mrl_std)
    validation_set = make_cnn_dataset(validation, mrl_mean, mrl_std)
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE)

    # Build the model and training tools
    model = MultiscaleCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    model_dir = Path(results_dir) / "cnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_r2 = -1.0
    best_mse = float("inf")
    epochs_without_improvement = 0

    # Train until the limit or early stopping
    for epoch in range(MAX_EPOCHS):
        train_loss = run_cnn_epoch(model, training_loader, optimizer, device)
        train_actual, train_predicted = predict_cnn_loader(
            model, training_loader, mrl_mean, mrl_std, device
        )
        val_actual, val_predicted = predict_cnn_loader(
            model, validation_loader, mrl_mean, mrl_std, device
        )
        # Measure training and validation quality
        train_metrics = calculate_metrics(train_actual, train_predicted)
        val_metrics = calculate_metrics(val_actual, val_predicted)

        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "train_r": train_metrics["pearson_r"],
                "val_r": val_metrics["pearson_r"],
                "train_r2": train_metrics["pearson_r_squared"],
                "val_r2": val_metrics["pearson_r_squared"],
            }
        )
        # Save progress after every epoch
        pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

        # Keep the best validation checkpoint
        val_r2 = val_metrics["pearson_r_squared"]
        val_mse = val_metrics["mse"]
        if val_r2 > best_r2 or (val_r2 == best_r2 and val_mse < best_mse):
            best_r2 = val_r2
            best_mse = val_mse
            epochs_without_improvement = 0
            checkpoint = {
                "mrl_mean": mrl_mean,
                "mrl_std": mrl_std,
                "epoch": epoch + 1,
                "val_r2": val_r2,
                "val_mse": val_mse,
                "seed": SEED,
            }
            save_cnn_model(model, model_dir / MODEL_FILES["cnn"], checkpoint)
        else:
            epochs_without_improvement += 1

        # Lower the learning rate when progress slows
        scheduler.step(val_r2)
        # Stop when validation does not improve
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    return model, pd.DataFrame(history)


# Train each requested model
def train_models(model_names, splits, results_dir):
    for model_name in model_names:
        if model_name == "cnn":
            train_cnn(splits, results_dir)
        else:
            train_baseline(model_name, splits, results_dir)
