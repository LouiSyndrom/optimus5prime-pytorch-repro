"""Evaluate a saved Optimus 5-Prime CNN and interpret first-layer filter motifs.

Run from the project folder: python evaluate.py
Requires your updated model.py, data.py, the original CSV, and outcomes/best_model.pth.
Optional: outcomes/training_history.csv for epoch-by-epoch learning curves.
No retraining or checkpoint overwriting occurs.
"""

from pathlib import Path
import heapq
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from data import read_utr_mrl, encode_sequence, assign_data, data_to_tensors
from model import CNN


DATA_FILE = "data/GSM3130435_egfp_unmod_1.csv"
CHECKPOINT = "outcomes/best_model.pth"
HISTORY_FILE = "outcomes/training_history.csv"
BATCH_SIZE = 128
FILTER_INDEX = 0         # Zero-based: 0 means the first of 120 filters
TOP_N = 1000            # One strongest 8-nt window per input sequence
NT_ORDER = "ATGC"      # Matches one_hot_encode() in data.py
COLORS = {"A": "#2ca02c", "T": "#d62728", "G": "#ff9900", "C": "#1f77b4"}

OUTCOMES_DIR = Path(__file__).resolve().parent / "outcomes"
OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)


def predict_on_test(model, testing_tensors, device):
    """Get predictions and labels without updating the saved model."""
    xs, ys = testing_tensors
    predictions = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(xs), BATCH_SIZE):
            batch = xs[start:start + BATCH_SIZE].to(device)
            predictions.append(model(batch).squeeze(1).cpu().numpy())

    return np.concatenate(predictions), ys.numpy()


def print_metrics(actual, pred):
    mse = np.mean((pred - actual) ** 2)
    rmse = np.sqrt(mse)
    pearson = np.corrcoef(actual, pred)[0, 1]
    r_squared = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Pearson r: {pearson:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Actual: mean={actual.mean():.4f}, SD={actual.std():.4f}, range=({actual.min():.4f}, {actual.max():.4f})")
    print(f"Predicted: mean={pred.mean():.4f}, SD={pred.std():.4f}, range=({pred.min():.4f}, {pred.max():.4f})")
    return mse, pearson


def plot_predictions(actual, pred, mse, pearson):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, pred, s=5, alpha=0.15)
    minimum = min(actual.min(), pred.min())
    maximum = max(actual.max(), pred.max())
    ax.plot([minimum, maximum], [minimum, maximum], "r--", label="Perfect prediction")
    ax.set(xlabel="Experimental MRL", ylabel="Predicted MRL",
           title=f"Optimus 5-Prime CNN\nPearson r = {pearson:.3f}, MSE = {mse:.3f}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTCOMES_DIR / "mrl_predictions.png", dpi=300)

    fig, ax = plt.subplots(figsize=(9, 5))
    # Shared bin edges make both histograms directly comparable.
    bins = np.linspace(minimum, maximum, 61)
    ax.hist(actual, bins=bins, alpha=0.5, label="Experimental MRL")
    ax.hist(pred, bins=bins, alpha=0.5, label="Predicted MRL")
    ax.set(xlabel="MRL", ylabel="Number of sequences", title="Experimental vs Predicted MRL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTCOMES_DIR / "mrl_distributions.png", dpi=300)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(actual, pred - actual, s=5, alpha=0.15)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Experimental MRL", ylabel="Residual (predicted - experimental)",
           title="CNN Prediction Residuals")
    fig.tight_layout()
    fig.savefig(OUTCOMES_DIR / "mrl_residuals.png", dpi=300)



def plot_training_history(history_file=HISTORY_FILE):
    """Plot epoch-by-epoch training and validation R² and MSE from a CSV.

    Expected CSV columns: epoch, train_r2, val_r2, train_mse, val_mse.
    A saved model checkpoint alone cannot reconstruct earlier epochs.
    """
    path = Path(history_file)
    if not path.is_file():
        print(f"\nTraining history not found: {path}")
        print("Skipping learning-curve plots. Save epoch metrics from train.py "
              "to this CSV to generate them.")
        return

    required = ("epoch", "train_r2", "val_r2", "train_mse", "val_mse")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"{path} must have columns: {', '.join(required)}")
        rows = list(reader)

    if not rows:
        print(f"Training history file is empty: {path}")
        return

    epochs = np.array([float(row["epoch"]) for row in rows])
    for metric, ylabel, filename in (
        ("r2", "R² score", "model_r2.png"),
        ("mse", "Mean squared error", "model_mse.png"),
    ):
        train = np.array([float(row[f"train_{metric}"]) for row in rows])
        validation = np.array([float(row[f"val_{metric}"]) for row in rows])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train, label="Train")
        ax.plot(epochs, validation, label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"Model {metric.upper()} by Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTCOMES_DIR / filename, dpi=300)
        print(f"Saved {OUTCOMES_DIR / filename}")


def top_activating_windows(model, training_tensors, device, filter_index=0, top_n=1000):
    """Keep the globally top-scoring *one complete window per training sequence*.

    conv1 output positions 3..45 map to unpadded input starts 0..42.
    Select raw pre-ReLU convolution scores, which rank windows in the same
    order as ReLU wherever the score is positive.
    """
    xs, _ = training_tensors
    kernel = model.conv1.kernel_size[0]
    left_pad = model.padding.padding[0]
    sequence_length = xs.shape[-1]
    complete_window_count = sequence_length - kernel + 1
    heap = []  # (activation, original_sequence_index, start_position)
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(xs), BATCH_SIZE):
            batch = xs[start:start + BATCH_SIZE].to(device)
            responses = model.conv1(model.padding(batch))[:, filter_index, :]
            valid = responses[:, left_pad:left_pad + complete_window_count]
            scores, positions = valid.max(dim=1)
            scores = scores.cpu().numpy()
            positions = positions.cpu().numpy()

            for offset, (score, position) in enumerate(zip(scores, positions)):
                item = (float(score), start + offset, int(position))
                if len(heap) < top_n:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    top = sorted(heap, reverse=True)
    windows = []
    for score, index, position in top:
        one_hot_window = xs[index, :, position:position + kernel]
        nt_indices = one_hot_window.argmax(dim=0).tolist()
        sequence = "".join(NT_ORDER[n] for n in nt_indices)
        windows.append((sequence, score, index, position))
    return windows


def position_frequency_matrix(windows, kernel_size):
    """Return rows A,T,G,C and columns of nucleotide positions."""
    counts = np.zeros((len(NT_ORDER), kernel_size), dtype=np.float64)
    lookup = {n: i for i, n in enumerate(NT_ORDER)}
    for window, *_ in windows:
        for pos, nt in enumerate(window):
            counts[lookup[nt], pos] += 1
    return counts / len(windows)


def plot_sequence_logo(pfm, filter_number):
    """Frequency-based stacked-letter sequence logo, no extra packages."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    font = FontProperties(family="DejaVu Sans", weight="bold")

    for pos in range(pfm.shape[1]):
        bottom = 0.0
        for nt, height in sorted(zip(NT_ORDER, pfm[:, pos]), key=lambda x: x[1]):
            if height <= 0:
                continue
            glyph = TextPath((0, 0), nt, size=1, prop=font)
            bounds = glyph.get_extents()
            transform = (Affine2D()
                         .translate(-bounds.x0, -bounds.y0)
                         .scale(0.78 / bounds.width, height / bounds.height)
                         .translate(pos + 0.11, bottom))
            ax.add_patch(PathPatch(glyph, transform=transform + ax.transData,
                                   facecolor=COLORS[nt], edgecolor="none"))
            bottom += height

    ax.set_xlim(0, pfm.shape[1])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(pfm.shape[1]) + 0.5, np.arange(1, pfm.shape[1] + 1))
    ax.set(xlabel="Position in 8-nt window", ylabel="Nucleotide frequency",
           title=f"Filter {filter_number}: Top {TOP_N} Activating Windows (Training Data)")
    fig.tight_layout()
    fig.savefig(OUTCOMES_DIR / f"filter_{filter_number}_sequence_logo.png", dpi=300)


def plot_filter_weights(model, filter_index):
    weights = model.conv1.weight[filter_index].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    vmax = max(np.abs(weights.min()), np.abs(weights.max()))
    im = ax.imshow(weights, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(4), list(NT_ORDER))
    ax.set_xticks(range(weights.shape[1]), range(1, weights.shape[1] + 1))
    ax.set(xlabel="Position within filter", ylabel="Nucleotide",
           title=f"Convolutional Filter {filter_index + 1}")
    fig.colorbar(im, ax=ax, label="Filter weight")
    fig.tight_layout()
    fig.savefig(OUTCOMES_DIR / f"filter_{filter_index + 1}_weights.png", dpi=300)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not Path(CHECKPOINT).is_file():
        raise FileNotFoundError(f"Cannot find {CHECKPOINT} in the current folder")

    # Reproduce the same fixed-seed split used during training.
    # Keep input CSV, data ordering and assign_data() unchanged.
    seq_dict = read_utr_mrl(DATA_FILE)
    arrays = encode_sequence(seq_dict)
    training, _, testing = assign_data(arrays)
    training_tensors = data_to_tensors(training)
    testing_tensors = data_to_tensors(testing)

    model = CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    pred, actual = predict_on_test(model, testing_tensors, device)
    mse, pearson = print_metrics(actual, pred)
    plot_predictions(actual, pred, mse, pearson)
    plot_training_history()

    plot_filter_weights(model, FILTER_INDEX)
    windows = top_activating_windows(model, training_tensors, device,
                                     filter_index=FILTER_INDEX, top_n=TOP_N)
    print(f"\nTop 10 windows for filter {FILTER_INDEX + 1}:")
    for window, activation, _, position in windows[:10]:
        print(f"{window} (activation={activation:.4f}, start={position})")

    pfm = position_frequency_matrix(windows, model.conv1.kernel_size[0])
    preferred = "".join(NT_ORDER[i] for i in pfm.argmax(axis=0))
    print("Consensus of top windows:", preferred)
    print("Position frequency matrix (rows A, T, G, C):\n", np.round(pfm, 3))
    np.savetxt(OUTCOMES_DIR / f"filter_{FILTER_INDEX + 1}_pfm.csv", pfm, delimiter=",",
               header="Columns: positions 1 through 8; rows in order A,T,G,C", comments="")
    plot_sequence_logo(pfm, FILTER_INDEX + 1)

    plt.show()  # One call: all figures display together.


if __name__ == "__main__":
    main()
