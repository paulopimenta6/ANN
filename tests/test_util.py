import unittest

from ann.Core.util import dot_product, normalize_by_feature_scaling, sigmoid, derivative_sigmoid


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
