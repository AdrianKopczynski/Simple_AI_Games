"""
Author: Gabriel Francke, Adrian Kopczyński

Description:
    Keras model factory functions used across all notebooks.
    Includes:
      - Tabular MLP (small/large) for Seeds/Diabetes
      - CNN (small/large) for CIFAR-10 and Fashion-MNIST

Usage (example):
    from src.models import build_mlp_small

    model = build_mlp_small(input_dim=7, num_classes=3)
"""

from __future__ import annotations

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models


# Helpers

def compile_classification_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Compile a Keras classification model with standard settings.

    Parameters:
        model         : tf.keras.Model - model to compile
        learning_rate : float - Adam learning rate

    Returns:
        compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# TABULAR: MLP models

def build_mlp_small(
    input_dim: int,
    num_classes: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.2
) -> tf.keras.Model:
    """
    Small MLP for tabular classification.

    Architecture:
        Dense(64) -> Dropout -> Dense(32) -> Dense(num_classes, softmax)
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="MLP_Small",
    )
    return compile_classification_model(model, learning_rate=learning_rate)


def build_mlp_large(
    input_dim: int,
    num_classes: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.3
) -> tf.keras.Model:
    """
    Larger MLP for tabular classification.

    Architecture:
        Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(32) -> Dense(num_classes, softmax)
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="MLP_Large",
    )
    return compile_classification_model(model, learning_rate=learning_rate)


# IMAGE: CNN models (generic)

def build_cnn_small(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.3
) -> tf.keras.Model:
    """
    Small CNN for image classification (Fashion-MNIST or CIFAR-10).

    Architecture:
        Conv -> Conv -> MaxPool
        Conv -> MaxPool
        Flatten -> Dense -> Dropout -> Dense(softmax)
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="CNN_Small",
    )
    return compile_classification_model(model, learning_rate=learning_rate)


def build_cnn_large(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.4
) -> tf.keras.Model:
    """
    Larger CNN for image classification (Fashion-MNIST or CIFAR-10).

    Architecture:
        Conv -> Conv -> MaxPool
        Conv -> Conv -> MaxPool
        Conv -> MaxPool
        Flatten -> Dense -> Dropout -> Dense(softmax)
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="CNN_Large",
    )
    return compile_classification_model(model, learning_rate=learning_rate)
