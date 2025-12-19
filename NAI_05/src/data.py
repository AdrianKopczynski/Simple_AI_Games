"""
Author: Gabriel Francke, Adrian Kopczyński

Description:
    Dataset loading and preprocessing utilities used across all notebooks.

Usage (example):
    from src.data import load_seeds_dataset

    X_train, X_test, y_train, y_test, scaler, class_names = load_seeds_dataset(
        "../data/seeds_dataset.txt"
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Helpers

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    import os
    os.makedirs(path, exist_ok=True)

# TABULAR: Seeds dataset

def load_seeds_dataframe(path: str) -> pd.DataFrame:
    """
    Load the Seeds dataset into a DataFrame.

    The dataset is whitespace-separated and has 7 features + 1 class label.
    """
    columns = [
        "area",
        "perimeter",
        "compactness",
        "kernel_length",
        "kernel_width",
        "asymmetry",
        "groove_length",
        "class",
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    return df


def load_seeds_dataset(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Load and prepare the Seeds dataset for NN training.

    Returns:
        X_train, X_test, y_train, y_test, scaler, class_names
    """
    df = load_seeds_dataframe(path)

    X = df.drop("class", axis=1).astype(np.float32).values
    y_raw = df["class"].astype(int).values  # classes: 1,2,3

    # Convert to 0..(num_classes-1) for Keras
    y = y_raw - 1
    num_classes = int(np.max(y)) + 1
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    class_names = ["Class 1", "Class 2", "Class 3"]
    return X_train, X_test, y_train, y_test, scaler, class_names


# TABULAR: Diabetes dataset

def load_diabetes_dataframe(path: str, sep: str = "\t") -> pd.DataFrame:
    """
    Load the Diabetes dataset from a local text file.

    Expected format:
        - 10 feature columns + 1 target column (last)
    """
    df = pd.read_csv(path, sep=sep, header=None)
    return df


def diabetes_to_binary_labels(y_raw: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert regression target to binary label using the median threshold.

    Returns:
        y_binary (0/1), threshold
    """
    threshold = float(np.median(y_raw))
    y_binary = (y_raw > threshold).astype(int)
    return y_binary, threshold


def load_diabetes_dataset(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str], float]:
    """
    Load and prepare the Diabetes dataset for binary classification with NN.

    Pipeline:
        - Load file
        - Split X and regression target y_raw
        - Convert y_raw -> binary labels (median split)
        - One-hot encode labels for Keras
        - Train/test split (stratified)
        - Optional scaling

    Returns:
        X_train, X_test, y_train, y_test, scaler, class_names, threshold
    """
    df = load_diabetes_dataframe(path)

    X = df.iloc[:, :-1].astype(np.float32).values
    y_raw = df.iloc[:, -1].astype(np.float32).values

    y_bin, threshold = diabetes_to_binary_labels(y_raw)
    y_onehot = tf.keras.utils.to_categorical(y_bin, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bin
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    class_names = ["Class 0", "Class 1"]
    return X_train, X_test, y_train, y_test, scaler, class_names, threshold


# IMAGE: CIFAR-10

def load_cifar10(
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load CIFAR-10 dataset (animals/objects). Returns train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    return X_train, X_test, y_train, y_test, class_names


# IMAGE: Fashion-MNIST

def load_fashion_mnist(
    normalize: bool = True,
    expand_channel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load Fashion-MNIST dataset. Returns train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    if expand_channel:
        # (N, 28, 28) -> (N, 28, 28, 1)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    return X_train, X_test, y_train, y_test, class_names
