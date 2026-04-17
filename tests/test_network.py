import unittest
from random import seed

from ann.Core.network import Network


class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
        seed(42)

    def test_invalid_network_structure_raises(self) -> None:
        with self.assertRaises(ValueError):
            Network([2, 1], 0.1)

    def test_outputs_shape_matches_last_layer(self) -> None:
        network = Network([2, 3, 1], 0.5)
        result = network.outputs([0.0, 1.0])
        self.assertEqual(len(result), 1)

    def test_train_updates_weights(self) -> None:
        network = Network([2, 2, 1], 0.5)
        before = [neuron.weights[:] for neuron in network.layers[1].neurons]
        before_bias = [neuron.bias for neuron in network.layers[1].neurons]

        inputs = [[0.0, 0.0], [1.0, 1.0]]
        expected = [[0.0], [1.0]]
        network.train(inputs, expected)

        after = [neuron.weights[:] for neuron in network.layers[1].neurons]
        after_bias = [neuron.bias for neuron in network.layers[1].neurons]
        self.assertNotEqual(before, after)
        self.assertNotEqual(before_bias, after_bias)


if __name__ == "__main__":
    unittest.main()
