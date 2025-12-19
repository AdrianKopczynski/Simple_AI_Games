"""
Author: Gabriel Francke, Adrian Kopczyński
Description:
    Plotting utilities for training curves and confusion matrices.
    Designed for saving screenshots/figures to the repository.

Usage (example):
    from src.plots import plot_training_history, plot_confusion_matrix

    plot_training_history(history, save_path="../screenshots/fashion_history.png")
    plot_confusion_matrix(y_true, y_pred, class_names, save_path="../screenshots/fashion_cm.png")
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_training_history(history, title: str = "Training History", save_path: Optional[str] = None) -> None:
    """
    Plot training and validation accuracy/loss curves from a Keras History object.

    Parameters:
        history   : tf.keras.callbacks.History
        title     : plot title
        save_path : if provided, save the figure to this path
    """
    metrics = history.history

    plt.figure(figsize=(8, 4))
    plt.plot(metrics.get("accuracy", []), label="train_accuracy")
    plt.plot(metrics.get("val_accuracy", []), label="val_accuracy")
    plt.title(f"{title} — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(metrics.get("loss", []), label="train_loss")
    plt.plot(metrics.get("val_loss", []), label="val_loss")
    plt.title(f"{title} — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        # Save loss plot with a simple naming convention
        loss_path = save_path.replace(".png", "_loss.png")
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Plot a confusion matrix using Matplotlib (no seaborn).

    Parameters:
        y_true      : true class indices
        y_pred      : predicted class indices
        class_names : list of class names in index order
        title       : title for the plot
        save_path   : if provided, save the figure to this path

    Returns:
        cm : confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Write values inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return cm
