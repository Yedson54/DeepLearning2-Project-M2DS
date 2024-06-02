"""
Module: principal_dbn_alpha.py
Module providing implementation of Deep Belief Network (DBN). 
"""

from typing import List

import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

from models.rbm import RBM


class DBN:
    """
    Implementation of a Deep Belief Network (DBN).

    Attributes:
    - n_visible (int): Number of visible units.
    - hidden_layer_sizes (List[int]): List of sizes for each hidden layer.
    - rbms (List[RBM]): List of Restricted Boltzmann Machines (RBMs) forming the DBN.
    - rng (numpy.random.Generator): Random number generator for sampling.
    """

    def __init__(
        self, n_visible: int, hidden_layer_sizes: List[int], random_state=None
    ):
        """
        Initialize the Deep Belief Network.

        Parameters:
        - n_visible (int): Number of visible units.
        - hidden_layer_sizes (List[int]): List of sizes for each hidden layer.
        - random_state: Random seed for reproducibility.
        """
        self.n_visible = n_visible
        self.hidden_layer_sizes = hidden_layer_sizes
        self.rbms: List[RBM] = []
        self.rng = np.random.default_rng(random_state)

        # Initialize the first RBM
        first_rbm = RBM(
            n_visible=self.n_visible,
            n_hidden=hidden_layer_sizes[0],
            random_state=random_state,
        )
        self.rbms.append(first_rbm)

        # Initialize RBMs for subsequent hidden layers
        for i, size in enumerate(hidden_layer_sizes[1:], start=1):
            self.rbms.append(
                RBM(
                    n_visible=hidden_layer_sizes[i - 1],
                    n_hidden=size,
                    random_state=random_state,
                )
            )

    def __getitem__(self, key):
        return self.rbms[key]

    def __repr__(self):
        """
        Return a string representation of the DBN object.
        """
        rbm_reprs = [repr(rbm) for rbm in self.rbms]
        join_rbm_reprs = ",\n ".join(rbm_reprs)
        return f"DBN([\n {join_rbm_reprs}\n])"

    def train(
        self,
        data: np.ndarray,
        learning_rate: float = 0.1,
        n_epochs: int = 10,
        batch_size: int = 10,
        print_each: int = 10,
        verbose=False,
        **kwargs,
    ) -> "DBN":
        """
        Train the DBN using Greedy layer-wise procedure.

        Parameters:
        - data (numpy.ndarray): Input data, shape (n_samples, n_visible).
        - learning_rate (float): Learning rate for gradient descent. Default is 0.1.
        - n_epochs (int): Number of training epochs. Default is 10.
        - batch_size (int): Size of mini-batches. Default is 10.
        - print_each: Print reconstruction error each `print_each` epochs.
        - verbose (bool): Whether to print training progress. Default is False.
        - **kwargs: Additional arguments to be passed to RBM.train().

        Returns:
        - DBN: Trained DBN instance.
        """
        input_data = data.copy()

        errors = []
        losses = []
        for rbm in tqdm(self.rbms, desc="Training RBM layers", unit="layer"):
            errors_rbm = rbm.train(
                input_data,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                batch_size=batch_size,
                print_each=print_each,
                verbose=verbose,
                **kwargs,
            )
            errors.extend(errors_rbm)
            losses.append(errors_rbm[-1])
            # Update input data for the next RBM
            h_probs, input_data = rbm.input_output(input_data)

        return errors, losses

    def generate_image(
        self, n_samples: int = 1,
        n_gibbs_steps: int = 100
    ) -> np.ndarray:
        """
        Generate samples from the DBN using Gibbs sampling.

        Parameters:
        - n_samples (int): Number of samples to generate. Default is 1.
        - n_gibbs_steps (int): Number of Gibbs sampling steps. Default is 100.

        Returns:
        - numpy.ndarray: Generated samples, shape (n_samples, n_visible).
        """
        # samples = np.zeros((n_samples, self.n_visible))

        # Generate samples using the first RBM in the DBN
        samples = self.rbms[-1].generate_image(n_samples, n_gibbs_steps)

        for rbm in reversed(self.rbms[:-1]):
            # Sample from the conditional probability of layer l-1 given layer l: p(h_{s-1}|h_{s}).
            h_probs, samples = rbm.output_input(samples)
            
        return samples
