"""Principal RBM alpha.
#HACK: optimize code, accelerate matrix computation with numba, parallelized when possible.
"""

import numpy as np
# from numba import jit
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

    Methods:
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

    def __repr__(self) -> str:
        return f"RBM(n_visible={self.n_visible}, n_hidden={self.n_hidden})"


    def input_output(self, data: np.ndarray) -> np.ndarray:
        """
        Compute hidden units given visible units.

        Parameters:
        - data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
        - numpy.ndarray: Hidden unit activations, shape (n_samples, n_hidden).
        """
        return F.sigmoid(data @ self.W + self.b)

    def output_input(self, data_h: np.ndarray) -> np.ndarray:
        """
        Compute visible units given hidden units.

        Parameters:
        - data_h (numpy.ndarray): Hidden unit activations, shape (n_samples, n_hidden).

        Returns:
        - numpy.ndarray: Reconstructed visible units, shape (n_samples, n_visible).
        """
        return F.sigmoid(data_h @ self.W.T + self.a)

    def train(
        self,
        input_data: np.ndarray,
        learning_rate: float = 0.1,
        n_epochs: int = 10,
        batch_size: int = 10,
        print_each: int = 10,
        verbose=False,
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
        #TODO: review TQDM position.
        """
        n_samples = input_data.shape[0]
        for epoch in range(n_epochs):
            data = self.rng.permutation(input_data, axis=0)
            for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch}"):
                batch = data[i:i+batch_size]
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

            if (epoch % print_each == 0) * verbose:
                tqdm.write(
                    f"Reconstruction error: \
                        {M.reconstruction_error(batch, pos_v_probs)}."
                )

        return self


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

        # Matrix of initlization value of Gibbs samples for each sample.
        v_init_matrix = self.rng.binomial(
            1, self.rng.random(), size=n_samples * self.n_visible
        ).reshape((n_samples, self.n_visible))

        for i in range(n_samples):
            h_probs = F.sigmoid(v_init_matrix[i] @ self.W + self.b)
            for _ in range(n_gibbs_steps):
                h = self.rng.binomial(1, h_probs)
                v_probs = F.sigmoid(h @ self.W.T + self.a)
                v = self.rng.binomial(1, v_probs)
            samples[i] = v

        return samples
