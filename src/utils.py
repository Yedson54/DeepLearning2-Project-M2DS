"""Module utils.py
"""

import os
from typing import Dict, List, Tuple, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

DATA_FOLDER = "../data/"
ALPHA_DIGIT_PATH = os.path.join(DATA_FOLDER, "binaryalphadigs.mat")


###########################################
################ DATA: ETL ################
###########################################
def load_alphadigit(alphadigit_path):
    return scipy.io.loadmat(alphadigit_path)["dat"]


def load_mnist(
    mnist_path: str, dataset_type: Literal["train", "test", "all"] = "all"
    mnist_path: str, dataset_type: Literal["train", "test", "all"] = "all"
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Load the MNIST dataset.

    Parameters:
        mnist_path (str): Path to the MNIST dataset file.
        dataset_type (Literal['train', 'test', 'all'], optional): Type of dataset to return.
            Defaults to 'all'.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            If dataset_type is 'train' or 'test', returns a tuple of numpy arrays (X_data, y_labels).
            If dataset_type is 'all', returns a tuple of four numpy arrays (X_train, y_train, X_test, y_test).
    """
    mnist = scipy.io.loadmat(mnist_path)
    mnist_train = np.concatenate([mnist[f"train{i}"] for i in range(10)], axis=0)
    mnist_test = np.concatenate([mnist[f"test{i}"] for i in range(10)], axis=0)

    mnist_train_labels = np.concatenate(
        [np.eye(10)[[i] * mnist[f"train{i}"].shape[0]] for i in range(10)], axis=0
    )
    mnist_test_labels = np.concatenate(
        [np.eye(10)[[i] * mnist[f"test{i}"].shape[0]] for i in range(10)], axis=0
    )

    if dataset_type == "train":
        return mnist_train, mnist_train_labels
    elif dataset_type == "test":
        return mnist_test, mnist_test_labels
    elif dataset_type == "all":
        return mnist_train, mnist_train_labels, mnist_test, mnist_test_labels
    else:
        raise ValueError(
            f"Invalid dataset_type {dataset_type}. \
                         Choose from 'train', 'test', or 'all'."
        )
        raise ValueError(
            f"Invalid dataset_type {dataset_type}. \
                         Choose from 'train', 'test', or 'all'."
        )


def load_data(file_path: str, which: Literal["alphadigit", "mnist"]) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
def load_data(file_path: str, which: Literal["alphadigit", "mnist"]) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load Binary AlphaDigits data from a .mat file.

    Parameters:
    - file_path (str): Path to the .mat file containing the data.
    - which (Literal["alphadigit", "mnist"], optional): Specifies
        which data to load. The default value is "alphadigit".

    Returns:
    - data (dict): A dictionary containing the loaded data.

    Example Usage:
    ```python
    data = _load_data("data.mat", "alphadigit")
    ```
    """
    if which == "alphadigit":
        return load_alphadigit(file_path)
    if which == "mnist":
        return load_mnist(file_path)


def map_characters_to_indices(
    characters: Union[str, int, List[Union[str, int]]]
) -> List[int]:
    """
    Map alphanumeric character to its corresponding index.

    Parameters:
    - character (str, int, list of str or int): Alphanumeric character or its index.

    Returns:
    - char_index (int): Corresponding index for the character.
    """
    if isinstance(characters, list):
        return [map_characters_to_indices(char) for char in characters]
    if isinstance(characters, int) and 0 <= characters <= 35:
        return [characters]
    if (
        isinstance(characters, str)
        and characters.isdigit()
        and 0 <= int(characters) <= 9
    ):
        return [int(characters)]
    if (
        isinstance(characters, str)
        and characters.isalpha()
        and "A" <= characters.upper() <= "Z"
    ):
        return [ord(characters.upper()) - ord("A") + 10]

    raise ValueError(
        f"Invalid character input {characters}. It should be an alphanumeric"
        f"character '[0-9|A-Z]' or its index representing '[0-35]'."
    )


def read_alpha_digit(
    characters: Optional[Union[str, int, List[Union[str, int]]]] = None,
    file_path: Optional[str] = ALPHA_DIGIT_PATH,
    data: Optional[Dict[str, np.ndarray]] = None,
    use_data: bool = False,
) -> np.ndarray:
    """
    Reads binary AlphaDigits data from a .mat file or uses already loaded data.
    It extracts the data for a specified alphanumeric character or its index, and
    flattens the images into one-dimensional vectors.

    Parameters:
    - characters (Union[str, int, List[Union[str, int]]], optional): Alphanumeric character
        or its index whose data needs to be extracted. It can be a single character or
        a list of characters. Default is None.
    - file_path (str, optional): Path to the .mat file containing the data.
        Default is None.
    - data (dict, optional): Already loaded data dictionary.
        Default is None.
    - use_data (bool): Flag to indicate whether to use already loaded data.
        Default is False.

    Returns:
    - flattened_images (numpy.ndarray): Flattened images for the specified character(s).
    """
    if not use_data:
        data = load_data(file_path, which="alphadigit")

    char_indices = map_characters_to_indices(characters)

    # Select the rows corresponding to the characters indices.
    char_data: np.ndarray = data[char_indices]

    # Flatten each image into a one-dimensional vector.
    flattened_images = np.array([image.flatten() for image in char_data.flatten()])

    return flattened_images


def get_predictions_one_hot(y_pred_probas: np.ndarray) -> np.ndarray:
    """
    Convert softmax probabilities to one-hot encoded predictions.

    Parameters:
    - y_pred_probas (numpy.ndarray): Softmax probabilities, shape (n_samples, n_classes).

    Returns:
    - numpy.ndarray: One-hot encoded predictions, shape (n_samples, n_classes).
    """
    # Convert softmax probabilities to predictions
    predictions = np.argmax(y_pred_probas, axis=1)
    # Create one-hot encoding
    num_classes = y_pred_probas.shape[1]
    predictions_one_hot = np.eye(num_classes)[predictions]
    return predictions_one_hot


######################################
################ PLOT ################
######################################
def plot_characters_alphadigit(
    chars: List[Union[str, int]],
    data: np.ndarray,
    reshape=(20, 16),
    cmap="gray",
    **kwargs,
    chars: List[Union[str, int]],
    data: np.ndarray,
    reshape=(20, 16),
    cmap="gray",
    **kwargs,
) -> None:
    """_summary_

    Parameters:
        chars (List[str, int]): _description_
        data (np.ndarray): _description_
        reshape (tuple, optional): _description_. Defaults to (20, 16).
    """
    num_chars = len(chars) + (
        len(chars) == 1
    )  # HACK: add 1 if only 1 character (to ensure axis is suscriptable)
    num_images_per_char = data.shape[0] // num_chars
    _, ax = plt.subplots(1, num_chars, figsize=(num_chars * 2, 2))

    for i, char in enumerate(chars):
        # Find the index of the first image corresponding to the current char
        start_index = i * num_images_per_char
        image = data[start_index].reshape(reshape)
        ax[i].imshow(image, cmap=cmap, **kwargs)
        ax[i].imshow(image, cmap=cmap, **kwargs)
        ax[i].set_title(f"Char: {char}")
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_generated_images(
    generated_samples: np.ndarray,
    n_cols: int = 10,
    cmap: str = "gray",
    reshape_generated: Tuple[int, int] = (20, 16),
    **kwargs,
) -> None:
    """
    Plot generated images.

    Parameters:
        generated_samples (np.ndarray): Generated samples.
        n_cols (int, optional): Number of columns in the plot. Defaults to 10.
        cmap (str, optional): Color map for the plot. Defaults to "gray".
        reshape_generated (Tuple[int, int], optional): Reshape dimensions for the generated images.
            Defaults to (20, 16).
        **kwargs: Additional keyword arguments to be passed to the `imshow` function.
    """
    n_images = generated_samples.shape[0]
    n_rows = (n_images - 1) // n_cols + 1

    # Plot generated samples
    plt.figure(figsize=(n_cols * 2, n_rows * 2))

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(generated_samples[i].reshape(reshape_generated), cmap=cmap, **kwargs)
        plt.title("Generated")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
