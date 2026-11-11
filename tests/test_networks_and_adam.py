"""Multilayer perceptron and Adam minimization (EQ-NN-MLP-001, EQ-OPT-ADAM-002).

Oracles: hand-written forward passes, closed-form derivatives of the
activations, forward-mode against reverse-mode gradients, a quadratic with a
known minimizer, and a least-squares fit whose optimum is the ordinary
least-squares solution.
"""

from __future__ import annotations

import math
import unittest

from probstat.foundations.random_streams import RandomStream
from probstat.numerics import (
    MLP,
    admath,
    gradient,
    reverse_value_and_gradient,
)
from probstat.optimization import adam_minimize


class MultilayerPerceptronTests(unittest.TestCase):
    """Architecture, initialization, and differentiability of :class:`MLP`."""

    def test_parameter_count_and_layout(self) -> None:
        """``(in + 1) * out`` per layer; a hand-built network matches."""
        network = MLP((2, 3, 1))
        self.assertEqual(network.parameter_count, 3 * 3 + 4 * 1)
        weights = (1.0, -1.0, 0.5, 0.0, 2.0, 1.0)
        biases = (0.1, -0.2, 0.3)
        out_weights = (1.0, 2.0, -1.0)
        out_bias = (0.5,)
        parameters = weights + biases + out_weights + out_bias
        x = (0.3, -0.4)
        hidden = [
            math.tanh(weights[2 * row] * x[0] + weights[2 * row + 1] * x[1] + b)
            for row, b in enumerate(biases)
        ]
        products = zip(out_weights, hidden, strict=True)
        expected = out_bias[0] + sum(w * h for w, h in products)
        (value,) = network.forward(parameters, x)
        self.assertAlmostEqual(value, expected, places=15)

    def test_activations_match_closed_forms(self) -> None:
        """Sigmoid, softplus and identity outputs equal their formulas."""
        for name, function in (
            ("sigmoid", lambda z: 1.0 / (1.0 + math.exp(-z))),
            ("softplus", lambda z: math.log1p(math.exp(z))),
            ("identity", lambda z: z),
        ):
            network = MLP((1, 1), output_activation=name)
            for z in (-30.0, -1.5, 0.0, 0.7, 25.0):
                with self.subTest(name=name, z=z):
                    (value,) = network.forward((1.0, 0.0), (z,))
                    tolerance = 1e-14 * (1 + abs(z))
                    self.assertAlmostEqual(value, function(z), delta=tolerance)

    def test_softplus_does_not_overflow(self) -> None:
        """``softplus(1000)`` is 1000, not an overflow."""
        network = MLP((1, 1), output_activation="softplus")
        (value,) = network.forward((1.0, 0.0), (1000.0,))
        self.assertEqual(value, 1000.0)

    def test_glorot_bounds_and_zero_biases(self) -> None:
        """Weights lie in ``[-r, r]``; biases are zero; seeds reproduce."""
        network = MLP((4, 8, 2))
        parameters = network.initial_parameters(11)
        self.assertEqual(parameters, network.initial_parameters(RandomStream(11)))
        first = parameters[: 4 * 8]
        radius = math.sqrt(6.0 / 12.0)
        self.assertTrue(all(abs(w) <= radius for w in first))
        self.assertEqual(parameters[32:40], (0.0,) * 8)
        scaled = network.initial_parameters(11, output_scale=0.0)
        self.assertEqual(scaled[40:56], (0.0,) * 16)

    def test_reverse_gradient_matches_forward_mode(self) -> None:
        """Reverse and forward mode agree on the gradient of a network loss."""
        network = MLP((2, 5, 5, 1), activation="softplus")
        parameters = network.initial_parameters(3)
        data = ((0.1, 0.2, 0.5), (-0.7, 0.4, -0.3), (1.2, -0.9, 0.8))

        def loss(theta: tuple[object, ...]) -> object:
            """Squared error of the network on three points.

            References:
                [EQ-NN-MLP-001].

            """
            total: object = 0.0
            for x0, x1, y in data:
                (prediction,) = network.forward(theta, (x0, x1))
                total = total + (prediction - y) ** 2
            return total

        value, reverse = reverse_value_and_gradient(loss, parameters)
        forward = gradient(loss, parameters)
        self.assertAlmostEqual(value, float(loss(parameters)), places=13)  # type: ignore[arg-type]
        for left, right in zip(reverse, forward, strict=True):
            self.assertAlmostEqual(left, right, delta=1e-12)

    def test_tanh_derivative_closed_form(self) -> None:
        """``d tanh(w x + b) / dw = x (1 - tanh^2)`` through the network."""
        network = MLP((1, 1), output_activation="tanh")
        _, grad = reverse_value_and_gradient(
            lambda theta: network.forward(theta, (0.8,))[0], (0.6, -0.1)
        )
        t = math.tanh(0.6 * 0.8 - 0.1)
        self.assertAlmostEqual(grad[0], 0.8 * (1 - t * t), places=14)
        self.assertAlmostEqual(grad[1], 1 - t * t, places=14)
        self.assertAlmostEqual(admath.tanh(0.3), math.tanh(0.3), places=15)

    def test_invalid_architectures_and_lengths(self) -> None:
        """Bad widths, activations, and vector lengths raise ``ValueError``."""
        for sizes in ((3,), (2, 0), (2, True)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                MLP(sizes)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MLP((1, 1), activation="relu")
        network = MLP((2, 1))
        with self.assertRaises(ValueError):
            network.forward((0.0,) * 2, (1.0, 2.0))
        with self.assertRaises(ValueError):
            network.forward((0.0,) * 3, (1.0,))
        with self.assertRaises(ValueError):
            network.initial_parameters(0, output_scale=-1.0)


class AdamMinimizeTests(unittest.TestCase):
    """Convergence and bookkeeping of :func:`adam_minimize`."""

    def test_quadratic_minimizer(self) -> None:
        """Deterministic quadratic: converges to its minimizer."""
        target = (1.5, -2.0, 0.25)

        def objective(
            x: tuple[float, ...], _: RandomStream
        ) -> tuple[float, list[float]]:
            """Return ``sum (x - target)^2`` and its gradient.

            References:
                [EQ-OPT-ADAM-002].

            """
            residual = [a - b for a, b in zip(x, target, strict=True)]
            return sum(r * r for r in residual), [2 * r for r in residual]

        result = adam_minimize(objective, (0.0, 0.0, 0.0), iterations=3000,
                               learning_rate=0.05)
        for value, expected in zip(result.x, target, strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-6)
        self.assertEqual(result.gradient_evaluations, 3000)
        self.assertEqual(len(result.losses), 3000)
        self.assertLess(result.smoothed_loss(10), 1e-10)

    def test_stochastic_least_squares_with_tail_averaging(self) -> None:
        """Mini-batch least squares reaches the OLS slope; averaging helps."""
        xs = [i / 50.0 for i in range(-50, 51)]
        ys = [2.0 * x + 0.5 + 0.3 * math.sin(7.0 * x) for x in xs]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        pairs = zip(xs, ys, strict=True)
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        slope = covariance / sum((x - mean_x) ** 2 for x in xs)
        intercept = mean_y - slope * mean_x

        def objective(
            theta: tuple[float, ...], stream: RandomStream
        ) -> tuple[float, list[float]]:
            """Mean squared error on a mini-batch of eight points.

            References:
                [EQ-OPT-ADAM-002].

            """
            generator = stream.generator()
            batch = [generator.randrange(n) for _ in range(8)]
            loss = 0.0
            grad = [0.0, 0.0]
            for index in batch:
                residual = theta[0] * xs[index] + theta[1] - ys[index]
                loss += residual * residual / 8
                grad[0] += 2 * residual * xs[index] / 8
                grad[1] += 2 * residual / 8
            return loss, grad

        averaged = adam_minimize(objective, (0.0, 0.0), iterations=4000,
                                 learning_rate=0.02, tail_fraction=0.5, seed=5)
        self.assertEqual(averaged.averaged_over, 2000)
        error_avg = math.hypot(averaged.x[0] - slope, averaged.x[1] - intercept)
        error_last = math.hypot(
            averaged.final_iterate[0] - slope, averaged.final_iterate[1] - intercept
        )
        self.assertLess(error_avg, 0.02)
        self.assertLessEqual(error_avg, error_last + 1e-12)
        again = adam_minimize(objective, (0.0, 0.0), iterations=4000,
                              learning_rate=0.02, tail_fraction=0.5, seed=5)
        self.assertEqual(again.x, averaged.x)

    def test_invalid_settings(self) -> None:
        """Invalid settings and non-finite gradients raise ``ValueError``."""

        def objective(
            x: tuple[float, ...], _: RandomStream
        ) -> tuple[float, list[float]]:
            """Return a constant loss and zero gradient.

            References:
                [EQ-OPT-ADAM-002].

            """
            return 0.0, [0.0] * len(x)

        for kwargs in (
            {"iterations": 0},
            {"learning_rate": 0.0},
            {"beta1": 1.0},
            {"beta2": -0.1},
            {"epsilon": math.inf},
            {"tail_fraction": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adam_minimize(objective, (1.0,), **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            adam_minimize(objective, ())
        with self.assertRaises(ValueError):
            adam_minimize(lambda x, s: (0.0, [0.0, 0.0]), (1.0,))
        with self.assertRaises(ValueError):
            adam_minimize(lambda x, s: (math.nan, [0.0]), (1.0,))


if __name__ == "__main__":
    unittest.main()
