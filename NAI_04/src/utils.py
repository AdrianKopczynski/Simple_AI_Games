"""
Utility functions for visualization and data processing.
Author: Gabriel Francke, Adrian Kopczyński
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Union
from sklearn.decomposition import PCA

def plot_pca_2d(X: np.ndarray, y: np.ndarray, title: str = "PCA 2D Visualization",
        save_path: str | None = None) -> None:
    """
    Perform PCA to reduce features to 2 dimensions and create a scatter plot.

    Parameters:
    X          : Input data to be transformed using PCA.
    y          : Class labels corresponding to samples in X.
    title      : Title of the plot (default: "PCA 2D Visualization").
    save_path  : If provided, the plot will be saved to the given path.

    Returns:
    None
    """

    # Perform PCA to reduce features to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create DataFrame with PCA components and class labels
    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "class": y
    })

    # Plot the PCA scatter plot
    plt.figure(figsize=(8, 6))

    # Plot each class with a different color
    for label in df_pca["class"].unique():
        subset = df_pca[df_pca["class"] == label]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Class {label}", alpha=0.7)

    # Add labels and title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()


def predict_single_sample_class(
        model: Any,
        sample: list | tuple,
        scaler=None
) -> Union[int, float]:
    """
    Predict the class label for a single input sample using a trained model.

    Parameters:
    model  : Trained ML model (DecisionTreeClassifier, SVC, etc.)
    sample : List or tuple of feature values representing one example.
    scaler : Optional scaler used during training (e.g., StandardScaler).
             If provided, the sample will be scaled before prediction.

    Returns:
    prediction : int
        The predicted class label.
    """

    # Convert to 2D array
    sample_arr = np.array(sample).reshape(1, -1)

    # Scale if scaler was used during training (SVM needs scaling)
    if scaler is not None:
        sample_arr = scaler.transform(sample_arr)

    # Predict class
    prediction = model.predict(sample_arr)[0]

    return prediction
