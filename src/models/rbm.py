"""Principal RBM alpha.
#HACK: optimize code, accelerate matrix computation with numba, parallelized when possible.
"""

from typing import Dict, Tuple, Optional

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import functionals as F
import metrics as M


class RBM:
    """
    Initialize the Restricted Boltzmann Machine.

    Parameters:
    - n_visible (int): Number of visible units.
    - n_hidden (int): Number of hidden units.
    - random_state: Random seed for reproducibility.

    Attributes:
    - n_visible: Number of visible units.
    - n_hidden: Number of hidden units.
    - a: Biases for visible units.
    - b: Biases for hidden units.
    - rng: Random number generator.
    - W: Weight matrix connecting visible and hidden units.

    Main Methods:
    - input_output(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute hidden units given visible units.
    - output_input(self, data_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Compute visible units given hidden units.
    - train(self, input_data: Optional[np.ndarray] = None, learning_rate: float = 0.1,
        n_epochs: int = 10, batch_size: int = 10, print_each: int = None,
        plot_errors: bool = False, save_each: Optional[int] = None,
        save_path: Optional[str] = None, verbose=False) -> "RBM":
        Train the RBM using Contrastive Divergence.
    - generate_image(self, n_samples: int = 1, n_gibbs_steps: int = 1) -> np.ndarray:
        Generate samples from the RBM using Gibbs sampling.
    """

    def __init__(
        self,
        n_visible: Optional[int],
        n_hidden: int,
        training_data: Optional[np.ndarray] = None,
        random_state=None,
    ) -> None:
        """
        Initialize the Restricted Boltzmann Machine.

        Parameters:
        - n_visible (int): Number of visible units.
        - n_hidden (int): Number of hidden units.
        - training_data (numpy.ndarray): Training data, shape (n_samples, n_visible).
        - random_state: Random seed for reproducibility.
        """
        if training_data is not None:
            n_visible = training_data.shape[1]
        elif n_visible is None:
            raise ValueError("Either n_visible or training_data has to be filled")

        self.n_visible = n_visible  # p
        self.n_hidden = n_hidden  # q
        self.rng = np.random.default_rng(random_state)
        self.init_rbm()
        self.X = training_data

    def __repr__(self) -> str:
        return f"RBM(n_visible={self.n_visible}, n_hidden={self.n_hidden})"

    def init_rbm(self):
        """Randomly initialize the weights and biases of the RBM."""
        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(self.n_hidden)
        self.W = self.rng.normal(
            loc=0, scale=1e-1, size=(self.n_visible, self.n_hidden)
        )

    def update_training_data(self, input_data: np.ndarray) -> None:
        """
        Update the training data and number of visible units.

        Parameters:
        - input_data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
        - None
        """
        if len(input_data.shape) <= 1:
            input_data = input_data[np.newaxis, :]
        self.X = input_data
        self.n_visible = self.X.shape[1]

    def input_output(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hidden units given visible units.

        Parameters:
            data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
            probabilities (numpy.ndarray): Hidden unit probabilities, shape (n_samples, n_hidden).
            values (numpy.ndarray): Hidden unit activations, shape (n_samples, n_hidden).
        """
        probabilities = F.sigmoid(data @ self.W + self.b)
        values = self.rng.binomial(n=1, p=probabilities)
        return probabilities, values

    def output_input(self, data_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute visible units given hidden units.

        Parameters:
        - data_h (numpy.ndarray): Hidden unit activations, shape (n_samples, n_hidden).

        Returns:
        - probabilities (numpy.ndarray): Visible unit probabilities, shape (n_samples, n_visible).
        - values (numpy.ndarray): Visible unit activations, shape (n_samples, n_visible).
        """
        probabilities = F.sigmoid(data_h @ self.W.T + self.a)
        values = self.rng.binomial(n=1, p=probabilities)
        return probabilities, values

    def train(
        self,
        input_data: Optional[np.ndarray] = None,
        learning_rate: float = 0.1,
        n_epochs: int = 10,
        batch_size: int = 10,
        print_each: int = None,
        plot_errors: bool = False,
        save_each: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose=False,
    ) -> "RBM":
        """
        Train the RBM using Contrastive Divergence.

        Parameters:
        - input_data (numpy.ndarray): Input data, shape (n_samples, n_visible).
        - learning_rate (float): Learning rate for gradient descent.
            Default is 0.1.
        - n_epochs (int): Number of training epochs.
            Default is 10.
        - batch_size (int): Size of mini-batches.
            Default is 10.
        - print_each (int): Number of epochs between printing progress.
            Default is 10.
        - plot_errors (bool): Whether to plot the reconstruction errors during training.
            Default is False.
        - save_each (int): Number of epochs between saving the model.
            If None, the model will not be saved during training.
            Default is None.
        - save_path (str): Path to save the model.
            Required if save_each is not None.
        - verbose (bool): Whether to print additional information during training.
            Default is False.

        Returns:
        - errors (list): list of reconstruction errors (mean square error, MSE).
        """
        if input_data is not None:
            self.update_training_data(input_data)
        elif self.X is not None:
            input_data = self.X
        else:
            raise ValueError(
                "`input_data` is required in `train` if no training_data had been provided in `init`."
            )
        if print_each is None:
            print_each = (n_epochs < 10) + (n_epochs > 10) * (n_epochs // 10)

        n_samples = input_data.shape[0]
        errors = []
        for epoch in range(n_epochs):
            # Shuffle the data.
            data = self.rng.permutation(input_data, axis=0)

            quadratic_error = 0
            for i in range(0, n_samples, batch_size):
                batch = data[i : i + batch_size]

                # Gibbs sampling.
                positive_h_probs, h0 = self.input_output(batch)  # probas_h_given_v0
                positive_v_probs, v1 = self.output_input(h0)
                negative_h_probs, h1 = self.input_output(v1)  # probas_h_given_v1

                # Compute gradient Update weights and biases
                grad_a = np.sum(batch - v1, axis=0)
                grad_b = np.sum(positive_h_probs - negative_h_probs, axis=0)
                grad_W = batch.T @ positive_h_probs - v1.T @ negative_h_probs
                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b
                self.W += learning_rate * grad_W

            # Compute reconstruction error
            quadratic_error += np.sum((v1 - batch) ** 2) / (n_samples * self.n_visible)
            # quadratic_error = M.reconstruction_error(batch, positive_v_probs)
            errors.append(quadratic_error)

            if (epoch % print_each == 0 or epoch == n_epochs - 1) * verbose:
                tqdm.write(f"Epoch {epoch}. Reconstruction error: {quadratic_error: .4f}.")

            # save the model weights if wanted
            if save_each and (epoch % save_each == 0):
                if save_path is None:
                    raise ValueError("`save_path` is required when `save_each` is not None.")
                save_path_ = save_path.split('.pkl')[0]
                save_path_ = f"{save_path_}_{epoch}.pkl"
                self.save_weights(save_path_)

        if plot_errors:
            plt.figure(figsize=(7, 4))
            plt.plot(errors)
            plt.grid('on')
            plt.title('MSE during training')
            plt.show()

        return errors

    def generate_image(self, n_samples: int = 1, n_gibbs_steps: int = 1) -> np.ndarray:
        """
        Generate samples from the RBM using Gibbs sampling.

        Parameters:
        - n_samples (int): Number of samples to generate. Default is 1.
        - n_gibbs_steps (int): Number of Gibbs sampling steps. Default is 100.

        Returns:
        - numpy.ndarray: Generated samples, shape (n_samples, n_visible).
        """
        samples = np.zeros((n_samples, self.n_visible))

        # Random initlization matrix value of Gibbs samples for each sample.
        v_init_matrix = self.rng.binomial(
            1, self.rng.random(), size=n_samples * self.n_visible
        ).reshape((n_samples, self.n_visible))

        for i in range(n_samples):
            _, h = self.input_output(v_init_matrix[i])
            for _ in range(n_gibbs_steps):
                _, v = self.output_input(h)
                _, h = self.input_output(v)
            samples[i] = v

        return samples

    def save_weights(self, path: str) -> None:
        """
        Save the weights of the Restricted Boltzmann Machine (RBM) to a file.

        Parameters:
            path (str): The path to the file where the weights will be saved.

        Returns:
            None
        """
        dict_weights = {"a": self.a, "W": self.W, "b": self.b}
        if path is None:
            return dict_weights

        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, "wb") as f:
            pickle.dump(dict_weights, f)
        return dict_weights

    def load_weights(
        self, 
        path: str,
        dict_weights: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Load the weights from a file.

        Parameters:
            path (str): The path to the file containing the weights.
            dict_weights (dict, optional): A dictionary containing the weights.
                If not provided, the weights will be loaded from the file specified by `path`.

        Returns:
            None
        """
        if dict_weights is None:
            with open(path, "rb") as f:
                dict_weights = pickle.load(f)
        self.a = dict_weights["a"]
        self.b = dict_weights["b"]
        self.W = dict_weights["W"]
