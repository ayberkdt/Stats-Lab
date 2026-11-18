"""Tests for deep-ensemble neural-network surrogates.

The networks are tiny (one hidden layer of eight units, three members) so the
whole file runs in well under a minute of scalar Python. Accuracy thresholds
are loose and fixed seeds are used only to pin coarse properties: the
training loop calls ``tanh``/``exp``/``log1p`` from the platform libm, whose
last-bit differences across operating systems are amplified by a thousand
Adam steps, so no digits are pinned across platforms.
"""

from __future__ import annotations

import math
import random
import unittest
from functools import cache

from probstat.distributions import Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import MLP, admath, reverse_value_and_gradient
from probstat.surrogates import (
    DeepEnsemble,
    DeepEnsemblePrediction,
    Surrogate,
    fit_deep_ensemble,
    fit_deep_ensemble_to_model,
    surrogate_probabilistic_model,
)
from probstat.surrogates.ensemble import _member_loss_and_gradient, _weight_mask
from probstat.uq import QuantityOfInterest

TINY = {
    "members": 3,
    "hidden": (8,),
    "iterations": 1500,
    "learning_rate": 0.02,
    "batch_size": 32,
}


def _smooth_1d(x: float) -> float:
    """Return the 1-D smooth test function ``sin(x) + 0.3 x``."""
    return math.sin(x) + 0.3 * x


def _smooth_2d(a: float, b: float) -> float:
    """Return the 2-D smooth test function ``sin(a) cos(b)``."""
    return math.sin(a) * math.cos(b)


def _noise_sd(x: float) -> float:
    """Return the true noise standard deviation of the heteroscedastic case."""
    return 0.05 + 0.4 * x


@cache
def _ensemble_1d() -> DeepEnsemble:
    """Fit (once) the ensemble on 20 equally spaced points of ``[0, 6]``."""
    points = [[6.0 * i / 19] for i in range(20)]
    values = [_smooth_1d(p[0]) for p in points]
    return fit_deep_ensemble(points, values, seed=1, **TINY)


@cache
def _ensemble_2d() -> DeepEnsemble:
    """Fit (once) the ensemble on 60 uniform points of ``[-2, 2]^2``."""
    generator = random.Random(2)
    points = [[generator.uniform(-2, 2), generator.uniform(-2, 2)] for _ in range(60)]
    values = [_smooth_2d(*p) for p in points]
    return fit_deep_ensemble(points, values, seed=2, **TINY)


@cache
def _ensemble_noisy(heteroscedastic: bool) -> DeepEnsemble:
    """Fit (once) the ensemble on ``y = 2x + noise`` with growing noise."""
    generator = random.Random(5)
    points = [[generator.uniform(0.0, 1.0)] for _ in range(200)]
    values = [2.0 * p[0] + generator.gauss(0.0, _noise_sd(p[0])) for p in points]
    return fit_deep_ensemble(
        points, values, heteroscedastic=heteroscedastic, seed=3, **TINY
    )


class GradientTests(unittest.TestCase):
    """The closed-form backward pass equals reverse-mode differentiation."""

    def test_gradient_matches_reverse_mode_for_every_activation(self) -> None:
        """Loss and gradient agree with ``reverse_value_and_gradient``."""
        generator = random.Random(3)
        inputs = [[generator.gauss(0, 1) for _ in range(3)] for _ in range(6)]
        targets = [generator.gauss(0, 1) for _ in range(6)]
        decay = 0.01
        for activation in ("tanh", "sigmoid", "softplus", "identity"):
            for heteroscedastic in (True, False):
                network = MLP(
                    (3, 4, 5, 2 if heteroscedastic else 1), activation=activation
                )
                parameters = [
                    value + 0.1 * generator.gauss(0, 1)
                    for value in network.initial_parameters(1)
                ]
                if not heteroscedastic:
                    parameters.append(0.3)
                mask = _weight_mask(network, heteroscedastic)

                def loss(theta, network=network, het=heteroscedastic, mask=mask):  # type: ignore[no-untyped-def]
                    """Return the reference loss written with ``MLP.forward``."""
                    total = 0.0
                    for x, y in zip(inputs, targets, strict=True):
                        out = network.forward(theta[: network.parameter_count], x)
                        log_variance = out[1] if het else theta[-1]
                        variance = admath.exp(log_variance) + 1e-6
                        total = total + 0.5 * (
                            math.log(2.0 * math.pi)
                            + admath.log(variance)
                            + (y - out[0]) ** 2 / variance
                        )
                    penalty = 0.0
                    for index, is_weight in enumerate(mask):
                        if is_weight:
                            penalty = penalty + theta[index] * theta[index]
                    return total / len(inputs) + 0.5 * decay * penalty

                expected_value, expected_gradient = reverse_value_and_gradient(
                    loss, parameters
                )
                value, gradient = _member_loss_and_gradient(
                    network, heteroscedastic, decay, mask, parameters, inputs, targets
                )
                with self.subTest(activation=activation, het=heteroscedastic):
                    self.assertAlmostEqual(value, expected_value, places=12)
                    self.assertEqual(len(gradient), len(expected_gradient))
                    for a, b in zip(gradient, expected_gradient, strict=True):
                        self.assertAlmostEqual(a, b, places=12)


class FitTests(unittest.TestCase):
    """Accuracy and uncertainty behaviour on smooth and noisy functions."""

    def test_smooth_1d_function_and_epistemic_growth(self) -> None:
        """Small validation error; variance grows away from the data."""
        ensemble = _ensemble_1d()
        midpoints = [[6.0 * (i + 0.5) / 19] for i in range(19)]
        errors = [ensemble.predict(p) - _smooth_1d(p[0]) for p in midpoints]
        rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
        self.assertLess(rmse, 0.05 * ensemble.output_scale)
        self.assertLess(ensemble.training_rmse, 0.05 * ensemble.output_scale)
        inside = math.fsum(ensemble.epistemic_variance(p) for p in ensemble.points)
        inside /= len(ensemble.points)
        far = [[-3.0], [-2.0], [8.0], [9.0]]
        outside = math.fsum(ensemble.epistemic_variance(p) for p in far) / len(far)
        self.assertGreater(outside, 5.0 * inside)
        total_inside = math.fsum(
            ensemble.predict_with_variance(p)[1] for p in ensemble.points
        ) / len(ensemble.points)
        total_far = math.fsum(ensemble.predict_with_variance(p)[1] for p in far) / 4
        self.assertGreater(total_far, total_inside)

    def test_smooth_2d_function_and_variance_away_from_data(self) -> None:
        """Validation RMSE below 10% of the output scale on ``[-2, 2]^2``."""
        ensemble = _ensemble_2d()
        generator = random.Random(7)
        validation = [
            [generator.uniform(-2, 2), generator.uniform(-2, 2)] for _ in range(100)
        ]
        rmse = math.sqrt(
            math.fsum((ensemble.predict(p) - _smooth_2d(*p)) ** 2 for p in validation)
            / len(validation)
        )
        self.assertLess(rmse, 0.1 * ensemble.output_scale)
        inside = math.fsum(
            ensemble.predict_with_variance(p)[1] for p in ensemble.points
        ) / len(ensemble.points)
        ring = [[5.0 * math.cos(k), 5.0 * math.sin(k)] for k in range(8)]
        outside = math.fsum(ensemble.predict_with_variance(p)[1] for p in ring) / 8
        self.assertGreater(outside, 5.0 * inside)

    def test_heteroscedastic_noise_is_tracked(self) -> None:
        """Aleatoric standard deviation within a factor 1.5 of the truth."""
        ensemble = _ensemble_noisy(True)
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            predicted = math.sqrt(ensemble.aleatoric_variance([x]))
            with self.subTest(x=x):
                self.assertLess(predicted, 1.5 * _noise_sd(x))
                self.assertGreater(predicted, _noise_sd(x) / 1.5)
                self.assertLess(abs(ensemble.predict([x]) - 2.0 * x), 0.1)
        # The predicted noise increases across the range, as the truth does.
        self.assertGreater(
            ensemble.aleatoric_variance([0.9]), 4.0 * ensemble.aleatoric_variance([0.1])
        )

    def test_homoscedastic_noise_is_constant(self) -> None:
        """One noise level per member, near the average true noise level."""
        ensemble = _ensemble_noisy(False)
        self.assertEqual(ensemble.network.output_dimension, 1)
        low = ensemble.aleatoric_variance([0.1])
        high = ensemble.aleatoric_variance([0.9])
        self.assertAlmostEqual(low, high, delta=1e-12 * high)
        # Root-mean-square true noise on [0, 1] is about 0.27.
        self.assertGreater(math.sqrt(low), 0.15)
        self.assertLess(math.sqrt(low), 0.45)


class PredictionContractTests(unittest.TestCase):
    """Mixture algebra, determinism, and protocol conformance."""

    def test_mixture_variance_split(self) -> None:
        """Variance equals aleatoric plus epistemic and the mixture formula."""
        ensemble = _ensemble_1d()
        for x in ([0.7], [3.1], [7.5]):
            prediction = ensemble.predict_decomposed(x)
            self.assertIsInstance(prediction, DeepEnsemblePrediction)
            pairs = ensemble.member_predictions(x)
            mean = math.fsum(m for m, _ in pairs) / len(pairs)
            second = math.fsum(v + m * m for m, v in pairs) / len(pairs)
            self.assertAlmostEqual(prediction.mean, mean, places=12)
            self.assertAlmostEqual(
                prediction.variance,
                prediction.aleatoric_variance + prediction.epistemic_variance,
                delta=1e-12 * prediction.variance,
            )
            self.assertAlmostEqual(
                prediction.variance, second - mean * mean, delta=1e-9 * second
            )
            self.assertEqual(ensemble.predict(x), prediction.mean)
            self.assertEqual(
                ensemble.predict_with_variance(x),
                (prediction.mean, prediction.variance),
            )
            self.assertEqual(
                ensemble.predict_with_variance(x, include_aleatoric=False),
                (prediction.mean, prediction.epistemic_variance),
            )
            self.assertEqual(len(prediction.member_means), ensemble.size)

    def test_fixed_seed_is_deterministic(self) -> None:
        """Same seed, same members; another seed, different members."""
        points = [[0.1 * i] for i in range(12)]
        values = [math.cos(3.0 * p[0]) for p in points]
        settings = {"members": 2, "hidden": (4,), "iterations": 60}
        first = fit_deep_ensemble(points, values, seed=11, **settings)
        second = fit_deep_ensemble(points, values, seed=11, **settings)
        third = fit_deep_ensemble(points, values, seed=12, **settings)
        self.assertEqual(first, second)
        self.assertEqual(first.members, second.members)
        self.assertNotEqual(first.members, third.members)
        self.assertNotEqual(first.members[0], first.members[1])
        self.assertEqual(len(first.loss_traces), 2)
        self.assertEqual(len(first.loss_traces[0]), 60)
        self.assertTrue(all(math.isfinite(v) for v in first.member_losses))

    def test_surrogate_protocol_and_model_wrapper(self) -> None:
        """The wrapper trains on a design and plugs into the central model."""
        model = ProbabilisticModel(
            JointInput.independent(
                RandomInput("a", Uniform(-1.0, 1.0)),
                RandomInput("b", Uniform(-1.0, 1.0)),
            ),
            lambda p: {"y": p["a"] + 0.5 * p["b"] ** 2, "z": p["a"]},
            (QuantityOfInterest("y", "sum"), QuantityOfInterest("z", "first")),
        )
        before = model.evaluations
        ensemble = fit_deep_ensemble_to_model(
            model,
            "y",
            size=16,
            seed=4,
            members=2,
            hidden=(4,),
            iterations=40,
            bootstrap=True,
            heteroscedastic=False,
        )
        self.assertEqual(model.evaluations - before, 16)
        self.assertEqual(len(ensemble.points), 16)
        self.assertTrue(ensemble.bootstrap)
        self.assertIsInstance(ensemble, Surrogate)
        mean, variance = ensemble.predict_with_variance((0.2, -0.3))
        self.assertTrue(math.isfinite(mean))
        self.assertGreater(variance, 0.0)
        emulator = surrogate_probabilistic_model(model, {"y": ensemble})
        self.assertAlmostEqual(
            emulator.evaluate({"a": 0.2, "b": -0.3})["y"], mean, places=12
        )
        with self.assertRaises(KeyError):
            fit_deep_ensemble_to_model(model, "missing", size=8, seed=1)


class ValidationTests(unittest.TestCase):
    """Invalid arguments raise clear errors."""

    def test_invalid_data(self) -> None:
        """Reject too few, ragged, non-finite, or mismatched training data."""
        cases = [
            ([[0.0]], [1.0]),
            ([[0.0], [1.0, 2.0]], [1.0, 2.0]),
            ([[0.0], [math.nan]], [1.0, 2.0]),
            ([[0.0], [1.0]], [1.0, math.inf]),
            ([[0.0], [1.0]], [1.0]),
            ([[], []], [1.0, 2.0]),
            ("ab", [1.0, 2.0]),
        ]
        for points, values in cases:
            with self.subTest(points=points, values=values):
                with self.assertRaises(ValueError):
                    fit_deep_ensemble(points, values, iterations=1)

    def test_invalid_settings(self) -> None:
        """Reject bad counts, widths, activations, rates, and flags."""
        points = [[0.0], [0.5], [1.0]]
        values = [0.0, 1.0, 0.0]
        bad = [
            {"members": 0},
            {"members": True},
            {"members": 2.0},
            {"hidden": (0,)},
            {"hidden": (4, -1)},
            {"activation": "relu"},
            {"iterations": 0},
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"learning_rate": math.nan},
            {"weight_decay": -1.0},
            {"tail_fraction": 1.0},
            {"tail_fraction": -0.1},
            {"warmup_fraction": 1.0},
            {"warmup_fraction": -0.5},
            {"heteroscedastic": 1},
            {"bootstrap": "yes"},
        ]
        for settings in bad:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    fit_deep_ensemble(points, values, **{"iterations": 1, **settings})  # type: ignore[arg-type]

    def test_invalid_prediction_input_and_members(self) -> None:
        """Reject wrong-length or non-finite inputs and malformed members."""
        ensemble = fit_deep_ensemble(
            [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]],
            [0.0, 1.0, 2.0],
            members=1,
            hidden=(2,),
            iterations=2,
            seed=0,
        )
        with self.assertRaises(ValueError):
            ensemble.predict([0.0])
        with self.assertRaises(ValueError):
            ensemble.predict_with_variance([0.0, math.nan])
        with self.assertRaises(ValueError):
            DeepEnsemble(
                **{**_fields(ensemble), "members": ((0.0,),)},
            )
        with self.assertRaises(ValueError):
            DeepEnsemble(**{**_fields(ensemble), "members": ()})


def _fields(ensemble: DeepEnsemble) -> dict[str, object]:
    """Return the constructor arguments of a fitted ensemble."""
    names = (
        "network",
        "members",
        "heteroscedastic",
        "points",
        "values",
        "input_mean",
        "input_scale",
        "output_mean",
        "output_scale",
        "member_losses",
        "loss_traces",
        "training_rmse",
        "iterations",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "tail_fraction",
        "bootstrap",
        "warmup_fraction",
    )
    return {name: getattr(ensemble, name) for name in names}


if __name__ == "__main__":
    unittest.main()
