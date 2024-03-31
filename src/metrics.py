"""Module metrics.py
"""

import numpy as np


def reconstruction_error(
    input_image: np.ndarray,
    reconstructed_image: np.ndarray,
    decimals: int = 3
) -> float:
    """
    Compute reconstruction error.

    Parameters:
    - input (numpy.ndarray): Original input data.
    - image (numpy.ndarray): Reconstructed image.

    Returns:
    - float: Reconstruction error.
    """
    return np.round(np.power(input_image - reconstructed_image, 2).mean(), decimals)


def accuracy(predictions: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Calculate the accuracy of the model.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels, shape (n_samples, n_classes).
    - true_labels (numpy.ndarray): True labels, shape (n_samples, n_classes).

    Returns:
    - float: Accuracy of the model.
    """
    # Count the number of correct predictions
    correct_predictions = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1)
    )

    # Calculate accuracy
    acc = correct_predictions / len(true_labels)

    return acc


def classification_error_rate(
    predictions: np.ndarray, true_labels: np.ndarray
) -> float:
    """
    Calculate the classification error rate.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels, shape (n_samples, n_classes).
    - true_labels (numpy.ndarray): True labels, shape (n_samples, n_classes).

    Returns:
    - float: Classification error rate.
    """
    # Calculate accuracy
    acc = accuracy(predictions, true_labels)

    # Calculate classification error rate
    error_rate = 1 - acc

    return error_rate
