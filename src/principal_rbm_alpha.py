"""Principal RBM alpha.
#TODO: control for verbosity (add 'verbose' arg / think about where to progression bar with `tqdm`) 
#TODO: add a representation "__repr__" to the class. Look like `RBM(n_visible, n_hidden, rng)`.
#TODO: move `sigmoid`, and function related to data into others modules (utils, load_data).
# HACK: optimize code, accelerate matrix computation with numba, parallelized when possible.
#TODO: check relevance of using the RBM's RNG for generation phase (look inside the gibbs sampling).
# If a seed has been define, the gibbs sampling step will return the same sample for each it will
# sample the same h from the binomial -> #WARNING there might be something wrong with the function
# --------------------------- Other Tags (Example usage) ---------------------.
# FIXME: Example: This function is returning incorrect results for negative input values.
# BUG: Example: Division by zero error occurs in certain cases.
# HACK: Example: This code temporarily fixes the issue, but needs a proper solution.
"""

import os
from typing import Dict, Optional, Union

import numpy as np
import scipy.io
from tqdm import tqdm

DATA_FOLDER = "../data/"
ALPHA_DIGIT_PATH = os.path.join(DATA_FOLDER, "binaryalphadigs.mat")


def _load_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load Binary AlphaDigits data from a .mat file.

    Parameters:
    - file_path (str): Path to the .mat file containing the data.

    Returns:
    - data (dict): Loaded data dictionary.
    """
    if file_path is None:
        raise ValueError("File path must be provided.")
    return scipy.io.loadmat(file_path)


def _map_character_to_index(character: Union[str, int]) -> int:
    """
    Map alphanumeric character to its corresponding index.

    Parameters:
    - character (str or int): Alphanumeric character or its index.

    Returns:
    - char_index (int): Corresponding index for the character.
    """
    if isinstance(character, int) and 0 <= character <= 35:
        return character
    elif (
        isinstance(character, str) and character.isdigit(
        ) and 0 <= int(character) <= 9
    ):
        return int(character)
    elif (
        isinstance(character, str)
        and character.isalpha()
        and "A" <= character.upper() <= "Z"
    ):
        return ord(character.upper()) - ord("A") + 10
    else:
        raise ValueError(
            "Invalid character input. It should be an alphanumeric"
            "character '[0-9|A-Z]' or its index representing '[0-35]'."
        )


def read_alpha_digit(
    character: Optional[Union[str, int]] = None,
    file_path: Optional[str] = ALPHA_DIGIT_PATH,
    data_mat: Optional[Dict[str, np.ndarray]] = None,
    use_data: bool = False,
) -> np.ndarray:
    """
    Read Binary AlphaDigits data from a .mat file or use already loaded data,
    get the index associated with the alphanumeric character, and flatten the
    images.

    Parameters:
    - file_path (str, optional): Path to the .mat file containing the data.
        Default is None.
    - data_mat (dict, optional): Already loaded data dictionary.
        Default is None.
    - use_data (bool): Flag to indicate whether to use already loaded data.
        Default is False.
    - character (str or int, optional): Alphanumeric character or its index
        whose data needs to be extracted. Default is None.

    Returns:
    - flattened_images (numpy.ndarray): Flattened images for the specified character.
    """
    if not use_data:
        data_mat = _load_data(file_path)

    char_index = _map_character_to_index(character)

    # Select the row corresponding to the character index
    char_data: np.ndarray = data_mat["dat"][char_index]

    # Flatten each image into a one-dimensional vector
    flattened_images = np.array([image.flatten() for image in char_data])

    return flattened_images


class RBM:
    """
    Initialize the Restricted Boltzmann Machine.

    Parameters:
    - n_visible (int): Number of visible units.
    - n_hidden (int): Number of hidden units.
    - random_state: Random seed for reproducibility.

    Methods:
    - _sigmoid(self, x: np.ndarray) -> np.ndarray: Sigmoid activation function.
    - _reconstruction_error(self, input_img: np.ndarray, output_img: np.ndarray) -> float: 
        Compute reconstruction error.
    - entree_sortie(self, data: np.ndarray) -> np.ndarray: 
        Compute hidden units given visible units.
    - sortie_entree(self, data_h: np.ndarray) -> np.ndarray: 
        Compute visible units given hidden units.
    - train(self, data: np.ndarray, learning_rate: float=0.1, n_epochs: int=10, 
        batch_size: int=10, print_each=10) -> "RBM": Train the RBM using Contrastive Divergence.
    - generer_image(self, n_samples: int = 1, n_gibbs_steps: int = 1) -> np.ndarray: 
        Generate samples from the RBM using Gibbs sampling.

    Fields:
    - n_visible: Number of visible units.
    - n_hidden: Number of hidden units.
    - a: Biases for visible units.
    - b: Biases for hidden units.
    - rng: Random number generator.
    - W: Weight matrix connecting visible and hidden units.
    """
    def __init__(self, n_visible: int, n_hidden: int, random_state=None) -> None:
        """
        Initialize the Restricted Boltzmann Machine.

        Parameters:
        - n_visible (int): Number of visible units.
        - n_hidden (int): Number of hidden units.
        - random_state: Random seed for reproducibility.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.a = np.zeros((1, n_visible))
        self.b = np.zeros((1, n_hidden))
        self.rng = np.random.default_rng(random_state)
        self.W = 1e-4 * self.rng.standard_normal(size=(n_visible, n_hidden))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input array.

        Returns:
        - numpy.ndarray: Result of applying the sigmoid function to the input.
        """
        return 1 / (1 + np.exp(-x))

    def _reconstruction_error(
        self, input_img: np.ndarray, output_img: np.ndarray
    ) -> float:
        """
        Compute reconstruction error.

        Parameters:
        - input (numpy.ndarray): Original input data.
        - image (numpy.ndarray): Reconstructed image.

        Returns:
        - float: Reconstruction error.
        """
        return np.round(np.power(output_img - input_img, 2).mean(), 3)

    def input_output(self, data: np.ndarray) -> np.ndarray:
        """
        Compute hidden units given visible units.

        Parameters:
        - data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
        - numpy.ndarray: Hidden unit activations, shape (n_samples, n_hidden).
        """
        return self._sigmoid(data @ self.W + self.b)

    def output_input(self, data_h: np.ndarray) -> np.ndarray:
        """
        Compute visible units given hidden units.

        Parameters:
        - data_h (numpy.ndarray): Hidden unit activations, shape (n_samples, n_hidden).

        Returns:
        - numpy.ndarray: Reconstructed visible units, shape (n_samples, n_visible).
        """
        return self._sigmoid(data_h @ self.W.T + self.a)

    def train(
        self,
        data: np.ndarray,
        learning_rate: float = 0.1,
        n_epochs: int = 10,
        batch_size: int = 10,
        print_each=10,
    ) -> "RBM":
        """
        Train the RBM using Contrastive Divergence.

        Parameters:
        - data (numpy.ndarray): Input data, shape (n_samples, n_visible).
        - learning_rate (float): Learning rate for gradient descent. Default is 0.1.
        - n_epochs (int): Number of training epochs. Default is 10.
        - batch_size (int): Size of mini-batches. Default is 10.

        Returns:
        - RBM: Trained RBM instance.
        """
        n_samples = data.shape[0]
        for epoch in range(n_epochs):
            self.rng.shuffle(data)
            for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch}"):
                batch = data[i: i + batch_size]
                pos_h_probs = self.input_output(batch)
                pos_v_probs = self.output_input(pos_h_probs)
                neg_h_probs = self.input_output(pos_v_probs)

                # Update weights and biases
                self.W += (
                    learning_rate
                    * (batch.T @ pos_h_probs - pos_v_probs.T @ neg_h_probs)
                    / batch_size
                )
                self.b += learning_rate * (
                    pos_h_probs.mean(axis=0) - neg_h_probs.mean(axis=0)
                )
                self.a += learning_rate * (
                    batch.mean(axis=0) - pos_v_probs.mean(axis=0)
                )

            if epoch % print_each == 0:
                tqdm.write(
                    f"Reconstruction error: {self._reconstruction_error(batch, pos_v_probs)}."
                )

        return self

    def generate_image(self, n_samples: int=1, n_gibbs_steps: int=1) -> np.ndarray:
        """
        Generate samples from the RBM using Gibbs sampling.

        Parameters:
        - n_samples (int): Number of samples to generate. Default is 1.
        - n_gibbs_steps (int): Number of Gibbs sampling steps. Default is 100.

        Returns:
        - numpy.ndarray: Generated samples, shape (n_samples, n_visible).
        """
        samples = np.zeros((n_samples, self.n_visible))

        # Matrix of initlization value of Gibbs samples for each sample.
        V = self.rng.binomial(
            1, self.rng.random(), size=n_samples * self.n_visible
        ).reshape((n_samples, self.n_visible))
        for i in range(n_samples):
            h_probs = self._sigmoid(V[i] @ self.W + self.b)
            for _ in range(n_gibbs_steps):
                h = self.rng.binomial(1, h_probs)
                v_probs = self._sigmoid(h @ self.W.T + self.a)
                v = self.rng.binomial(1, v_probs)
            samples[i] = v
        return samples
