from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(metrics, save_path=None):
    high_read = metrics[metrics["dataset"] == "high_read_test"]
    model_names = high_read["model"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(model_names, high_read["pearson_r_squared"])
    axes[0].set(ylabel="Pearson r squared", title="High-read prediction correlation")
    axes[1].bar(model_names, high_read["rmse"])
    axes[1].set(ylabel="RMSE", title="High-read prediction error")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200)
    return fig


def plot_actual_vs_predicted(predictions, model_name, save_path=None):
    actual = predictions["actual_mrl"].to_numpy()
    predicted = predictions["predicted_mrl"].to_numpy()
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, predicted, s=4, alpha=0.15)
    axis.plot([minimum, maximum], [minimum, maximum], "r--")
    axis.set(
        xlabel="Experimental MRL",
        ylabel="Predicted MRL",
        title=f"{model_name}: actual versus predicted MRL",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200)
    return fig


def plot_cnn_learning_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_r2"], label="Train")
    axes[0].plot(history["epoch"], history["val_r2"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Pearson r squared")
    axes[1].plot(history["epoch"], history["train_mse"], label="Train")
    axes[1].plot(history["epoch"], history["val_mse"], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="MSE")
    for axis in axes:
        axis.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200)
    return fig


def plot_read_group_comparison(metrics, save_path=None):
    table = metrics.pivot(index="model", columns="dataset", values="pearson_r_squared")
    positions = np.arange(len(table))
    width = 0.35

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        positions - width / 2,
        table["high_read_test"],
        width,
        label="High-read",
    )
    axis.bar(
        positions + width / 2,
        table["low_read_test"],
        width,
        label="Low-read",
    )
    axis.set_xticks(positions, table.index, rotation=25)
    axis.set(ylabel="Pearson r squared", title="High-read versus low-read performance")
    axis.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200)
    return fig
