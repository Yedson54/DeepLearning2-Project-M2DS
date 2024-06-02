"""Module implementing a Deep Neural Network (DNN) pretrained with a 
Deep Belief Network (DBN)
"""

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import functionals as F
from models.dbn import DBN
from models.rbm import RBM
from metrics import classification_error_rate
from utils import get_predictions_one_hot


class DNN(DBN):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_sizes: List[int],
        random_state=None,
    ):
        """
        Initialize the Deep Neural Network (DNN).

        Parameters:
        - input_dim (int): Dimension of the input.
        - output_dim (int): Dimension of the output.
        - hidden_layer_sizes (List[int]): List of sizes for each hidden layer.
        - random_state: Random seed for reproducibility.
        """
        super().__init__(
            n_visible=input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
        )
        # --> self.rbms contains only the pre-trainable RBMs
        self.clf = RBM(self.rbms[-1].n_hidden, output_dim)
        # DNN = [DBN + Classifier] ~ [RBM_0,...,RBM_N, RBM_Clf]
        self.network = self.rbms + [self.clf]
        self.n_iter = 0

    def __getitem__(self, key):
        return self.network[key]

    def __len__(self):
        return len(self.network)

    def __repr__(self):
        join_repr = "\n".join([f"{'':4}{repr(rbm)}," for rbm in self.network])
        return f"DNN([\n{join_repr} <CLF>\n])"

    def pretrain(
        self,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        data: np.ndarray,
        print_each=20,
        verbose=False,
    ) -> "DNN":
        """
        Pretrain the hidden layers of the DNN using the DBN training method.

        Parameters:
        - n_epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for gradient descent.
        - batch_size (int): Size of mini-batches.
        - data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
        - DNN: Pretrained DNN instance.
        """
        # NOTE: Use the inherited `train` method to perform pre-training since `self.rbms`
        # only contains the pre-trainable RBMs.
        return self.train(
            data,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            print_each=print_each,
            verbose=verbose,
        )

    def input_output_network(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Get the outputs on each layer of the DNN and the softmax probabilities on the output layer.

        Parameters:
        - input_data (numpy.ndarray): Input data, shape (n_samples, n_visible).

        Returns:
        - List[np.ndarray]: Input data, outputs on each layer and softmax probabilities.
        """
        layer_outputs = [input_data]
        layer_probas = []
        for rbm in self.rbms:
            h_probas, h_predictions = rbm.input_output(layer_outputs[-1])
            layer_outputs.append(h_predictions)
            layer_probas.append(h_probas)

        output_logits, output_prediction = self.network[-1].input_output(layer_outputs[-1])
        layer_outputs.append(F.softmax(h_probas))

        return layer_outputs

    def update(
        self,
        dZ_lead: np.ndarray,
        dW_lead: np.ndarray,
        layer_outputs: List[np.ndarray],
        id_layer: int,
        batch_size: int,
        learning_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the weights and biases of a layer.

        Parameters:
        - dZ_lead (numpy.ndarray): Gradient with respect to the layer's output.
        - dW_lead (numpy.ndarray): Gradient with respect to the layer's weights.
        - layer_outputs (List[np.ndarray]): Outputs of each layer.
        - id_layer (int): Index of the layer.
        - batch_size (int): Size of mini-batches.
        - learning_rate (float): Learning rate.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Updated gradient with respect to the
            layer's output and weights.
        """
        # Compute gradient (layer no. `id_layer` + 1)
        dZ = (
            (dZ_lead @ dW_lead.T)
            * layer_outputs[id_layer]
            * (1 - layer_outputs[id_layer])
            / batch_size
        )
        dW = layer_outputs[id_layer - 1].T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update hidden layer weights and biases (layer no. `id_layer` + 1).
        self.network[id_layer].W -= learning_rate * dW
        self.network[id_layer].b -= learning_rate * db
        self.n_iter += 1
        return dZ, dW

    def backpropagation(
        self,
        input_data: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        learning_rate: float = 0.1,
        batch_size: int = 10,
        eps: float = 1e-15,
        # continuation=False
    ) -> "DNN":
        """
        Estimate the weights/biases of the network using backpropagation algorithm.

        Parameters:
        - input_data (numpy.ndarray): Input data, shape (n_samples, n_visible).
        - labels (numpy.ndarray): Labels for the input data, shape
            (n_samples, n_classes).
        - n_epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for gradient descent.
        - batch_size (int): Size of mini-batches.
        - eps (float): Small value to avoid numerical instability in logarithm
            calculation. Default is 1e-15.

        Returns:
        - DNN: Updated DNN instance.
        """
        n_samples = input_data.shape[0]

        for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_input = input_data[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                # Forward pass
                layer_outputs = self.input_output_network(batch_input)

                # Backward pass (update weights and biases)

                # if continuation:
                #     dZ = self.
                # Compute output (last) layer gradients (layer L).
                dZ = (
                    layer_outputs[-1] - batch_labels
                ) / batch_size  # -> (n_samples, output_dim)
                # -> (self[-2].n_hidden, self[-1].n_hidden)
                dW = layer_outputs[-2].T @ dZ
                # -> (1, self[-1].n_hidden)
                db = np.sum(dZ, axis=0, keepdims=True)
                # Update output (last) layer parameters (layer L).
                self.network[-1].W -= learning_rate * dW
                self.network[-1].b -= learning_rate * db
                # self.dZs.append(dZ)
                # self.dWs.append(dW)
                # self.dbs.append(db)
                self.n_iter += 1

                # Iterate layer in reverse order
                for id_layer in range(-2, -len(self.network)):
                    dZ, dW = self.update(
                        dZ_lead=dZ,
                        dW_lead=dW,
                        id_layer=id_layer,
                        layer_outputs=layer_outputs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                    )

            # HACK: update discrepancy / force
            self.rbms = self.network[:-1]

            # Calculate cross entropy after each epoch
            loss = F.cross_entropy(batch_labels, layer_outputs[-1], eps)
            tqdm.write(f"Epoch {epoch + 1}/{n_epochs}, Cross Entropy: {loss}")

        return self

    def test(self, test_data: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Test the performance of the trained DNN on a test dataset.

        Parameters:
        - test_data (numpy.ndarray): Test data, shape (n_samples, n_visible).
        - true_labels (numpy.ndarray): True labels for the test data,
            shape (n_samples, n_classes).

        Returns:
        - float: Classification error rate.
        """
        # Estimate labels using the trained DNN
        estimated_labels = self.input_output_network(test_data)[-1]

        # Convert softmax probabilities to one-hot encoded predictions
        estimated_labels_one_hot = get_predictions_one_hot(estimated_labels)

        # Calculate classification error rate
        error_rate = classification_error_rate(estimated_labels_one_hot, true_labels)

        return error_rate
