"""
Author: Gabriel Francke, Adrian Kopczyński

Description:
    Generic training and evaluation utilities for Keras models.
    Used by all notebooks to keep training logic consistent.

Usage (example):
    from src.train import train_model, evaluate_model

    history = train_model(model, X_train, y_train, X_val, y_val)
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import tensorflow as tf


# Training

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """
    Train a Keras classification model.

    Parameters:
        model       : compiled tf.keras.Model
            The neural network to be trained. It must be compiled before calling this function.

        X_train     : np.ndarray
            Training feature data used for learning the model weights.

        y_train     : np.ndarray
            Training labels in one-hot encoded format.

        X_val       : np.ndarray
            Validation feature data used to monitor model performance during training.

        y_val       : np.ndarray
            Validation labels in one-hot encoded format.

        epochs      : int
            Number of full passes through the training dataset.
            More epochs allow the model to learn more, but may cause overfitting.

        batch_size  : int
            Number of samples processed before updating model weights.
            Smaller values train slower but may generalize better.

        verbose     : int
            Controls training output.
            0 = no output, 1 = progress bar per epoch.

    Returns:
        history : tf.keras.callbacks.History
            Object containing loss and accuracy values for each training epoch.
    """

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    return history


# Evaluation

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate a trained model and return accuracy and predictions.

    Parameters:
        model  : trained tf.keras.Model
        X_test : test features
        y_test : test labels (one-hot)

    Returns:
        accuracy : float
        y_pred   : predicted class indices
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    return accuracy, y_pred


def get_true_labels(y_test: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded labels to class indices.

    Parameters:
        y_test : one-hot encoded labels

    Returns:
        y_true : class indices
    """
    return np.argmax(y_test, axis=1)
