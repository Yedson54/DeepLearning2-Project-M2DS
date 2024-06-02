"""Module functionals.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Parameters:
    - x (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Result of applying the sigmoid function to the input.
    """
    return 1 / (1 + np.exp(-x))


def softmax(hidden_activations: np.ndarray) -> np.ndarray:
    """
    Calculate softmax probabilities for the output units.

    Parameters:
    - input_data (numpy.ndarray): Input data, shape (n_samples, n_visible).

    Returns:
    - numpy.ndarray: Softmax probabilities, shape (n_samples, n_hidden).
    """
    # Compute softmax probabilities for the output layer
    exp_hidden_activations = np.exp(hidden_activations)
    softmax_probs = exp_hidden_activations / np.sum(
        exp_hidden_activations, axis=1, keepdims=True
    )
    return softmax_probs


def cross_entropy(
    batch_labels: np.ndarray,
    output_probs: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Calculate the cross entropy between the batch labels and output probabilities.

    Parameters:
    - batch_labels (numpy.ndarray): True labels for the batch, shape (batch_size, n_classes).
    - output_probs (numpy.ndarray): Predicted probabilities for the batch,
        shape (batch_size, n_classes).
    - eps (float): Small value to avoid numerical instability in logarithm calculation.
        Default is 1e-15.

    Returns:
    - float: Cross entropy value.
    """
    return -np.sum(batch_labels * np.log(output_probs + eps))
