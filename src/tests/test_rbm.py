import os
import numpy as np
import scipy.io
import unittest
from rbm import RBM, _load_data, _map_character_to_index, read_alpha_digit

DATA_FOLDER = "../data/"
ALPHA_DIGIT_PATH = os.path.join(DATA_FOLDER, "binaryalphadigs.mat")


class TestRBM(unittest.TestCase):
    def setUp(self):
        # Load alpha_digit data for testing
        self.data = read_alpha_digit(file_path=ALPHA_DIGIT_PATH, character='A')
        self.n_samples = self.data.shape[0]
        self.n_visible = self.data.shape[1]
        self.n_hidden = 100
        self.rbm = RBM(n_visible=self.n_visible, n_hidden=self.n_hidden)

    def test__sigmoid(self):
        # Test _sigmoid method with positive and negative values
        x = np.array([1, -1, 0])
        sigmoid_x = self.rbm._sigmoid(x)
        self.assertTrue(np.allclose(sigmoid_x, [0.73105858, 0.26894142, 0.5]))

    def test__reconstruction_error(self):
        # Test _reconstruction_error method with arrays containing zeros
        input_img = np.zeros_like(self.data)
        output_img = np.zeros_like(self.data)
        error = self.rbm._reconstruction_error(input_img, output_img)
        self.assertEqual(error, 0.0)

    def test_entree_sortie(self):
        # Test entree_sortie method with a small input array
        input_data = np.ones((2, self.n_visible))
        output = self.rbm.entree_sortie(input_data)
        self.assertEqual(output.shape, (2, self.n_hidden))

    def test_sortie_entree(self):
        # Test sortie_entree method with a small input array
        input_data = np.ones((2, self.n_hidden))
        output = self.rbm.sortie_entree(input_data)
        self.assertEqual(output.shape, (2, self.n_visible))

    def test_train(self):
        # Test train method
        trained_rbm = self.rbm.train(
            self.data[:100], learning_rate=0.1, n_epochs=1, batch_size=10
        )
        self.assertTrue(hasattr(trained_rbm, 'W'))
        self.assertTrue(hasattr(trained_rbm, 'a'))
        self.assertTrue(hasattr(trained_rbm, 'b'))

    def test_generer_image(self):
        # Test generer_image method
        samples = self.rbm.generer_image(n_samples=2, n_gibbs_steps=10)
        self.assertEqual(samples.shape, (2, self.n_visible))
        self.assertTrue(np.all((samples == 0) | (samples == 1)))


if __name__ == "__main__":
    unittest.main()
