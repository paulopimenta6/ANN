import unittest

from ann.Core.util import (
    dot_product,
    normalize_by_feature_scaling,
    sigmoid,
    derivative_sigmoid,
    tanh_activation,
    derivative_tanh_activation,
    relu_activation,
    derivative_relu_activation,
    leaky_relu_activation,
    derivative_leaky_relu_activation,
    resolve_activation_functions,
)


class TestUtilFunctions(unittest.TestCase):
    def test_dot_product(self) -> None:
        self.assertAlmostEqual(dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_sigmoid_and_derivative(self) -> None:
        value = sigmoid(0.0)
        self.assertAlmostEqual(value, 0.5)
        self.assertAlmostEqual(derivative_sigmoid(0.0), 0.25)

    def test_sigmoid_extreme_values(self) -> None:
        high = sigmoid(1000.0)
        low = sigmoid(-1000.0)
        self.assertAlmostEqual(high, 1.0, places=12)
        self.assertAlmostEqual(low, 0.0, places=12)

    def test_tanh_and_derivative(self) -> None:
        self.assertAlmostEqual(tanh_activation(0.0), 0.0)
        self.assertAlmostEqual(derivative_tanh_activation(0.0), 1.0)

        positive = tanh_activation(2.0)
        negative = tanh_activation(-2.0)
        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)

    def test_relu_and_derivative(self) -> None:
        self.assertEqual(relu_activation(-3.0), 0.0)
        self.assertEqual(relu_activation(0.0), 0.0)
        self.assertEqual(relu_activation(2.5), 2.5)
        self.assertEqual(derivative_relu_activation(-3.0), 0.0)
        self.assertEqual(derivative_relu_activation(2.5), 1.0)

    def test_leaky_relu_and_derivative(self) -> None:
        self.assertAlmostEqual(leaky_relu_activation(-2.0), -0.02)
        self.assertEqual(leaky_relu_activation(3.0), 3.0)
        self.assertAlmostEqual(derivative_leaky_relu_activation(-2.0), 0.01)
        self.assertEqual(derivative_leaky_relu_activation(3.0), 1.0)
        self.assertAlmostEqual(leaky_relu_activation(-2.0, alpha=0.1), -0.2)
        self.assertAlmostEqual(derivative_leaky_relu_activation(-2.0, alpha=0.1), 0.1)

    def test_resolve_activation_functions(self) -> None:
        activation, derivative = resolve_activation_functions("sigmoid")
        self.assertAlmostEqual(activation(0.0), 0.5)
        self.assertAlmostEqual(derivative(0.0), 0.25)

        activation, derivative = resolve_activation_functions("leaky_relu", leaky_alpha=0.2)
        self.assertAlmostEqual(activation(-2.0), -0.4)
        self.assertAlmostEqual(derivative(-2.0), 0.2)

        with self.assertRaises(ValueError):
            resolve_activation_functions("invalida")

    def test_normalize_by_feature_scaling(self) -> None:
        dataset = [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0],
            [3.0, 30.0, 5.0],
        ]
        normalize_by_feature_scaling(dataset)

        self.assertEqual(dataset[0], [0.0, 0.0, 0.0])
        self.assertEqual(dataset[1], [0.5, 0.5, 0.0])
        self.assertEqual(dataset[2], [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
