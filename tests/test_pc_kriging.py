"""Tests for universal kriging and polynomial-chaos Kriging.

References:
    [REF-SCHOBI-SUDRET-WIART-2015], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2],
    [REF-PYTHON-UNITTEST]. Equations [EQ-SUR-UK-001] to [EQ-SUR-UK-004],
    [EQ-SUR-PCK-001], and [EQ-SUR-GP-008].

"""

from __future__ import annotations

import math
import unittest
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from probstat.distributions import Normal, Uniform
from probstat.foundations.random_streams import resolve_stream
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    ishigami_model,
)
from probstat.propagation import make_design
from probstat.surrogates import (
    KERNELS,
    PC_KRIGING_VARIANTS,
    ChaosTrend,
    PCKriging,
    UniversalKriging,
    fit_gaussian_process,
    fit_pc_kriging,
    fit_universal_kriging,
    gaussian_process_log_marginal_likelihood,
    kernel_value,
    pc_kriging_from_samples,
    surrogate_probabilistic_model,
)
from probstat.surrogates.pc_kriging import _objective, _pair_data
from probstat.uq import QuantityOfInterest

# Seven points in [0, 1]^2 with y = sin(3 x1) + x2^2; Matern 5/2 correlation
# with fixed length scales (0.4, 0.6) and nugget 1e-10.
POINTS = (
    (0.1, 0.2),
    (0.9, 0.4),
    (0.5, 0.8),
    (0.3, 0.6),
    (0.7, 0.1),
    (0.2, 0.95),
    (0.85, 0.75),
)
VALUES = tuple(math.sin(3.0 * a) + b * b for a, b in POINTS)
SCALES = (0.4, 0.6)
PROBES = ((0.4, 0.4), (1.2, -0.1))

# Pinned from an independent NumPy implementation (np.linalg.inv of R and of
# the bordered matrices; scratch script, not part of the package): GLS beta,
# sigma^2 = e'R^-1 e / N, mu = f'beta + r'R^-1 e, s^2 = sigma^2 (1 - [f' r']
# [[0, F'], [F, R]]^-1 [f; r]), and LOO from B = [[sigma^2 R, F], [F', 0]]^-1.
ORDINARY: dict[str, Any] = {
    "beta": (0.9037284790958735,),
    "variance": 0.2928835590928269,
    "probes": (
        (0.9902289855663007, 0.03269931931861698),
        (0.6360244158041141, 0.2660118013447749),
    ),
    "loo_means": (
        0.8992921608283556,
        0.9491712241026521,
        1.4209802854061955,
        1.2124826774817692,
        0.5541001510225396,
        1.2245557610938873,
        0.9416986546257908,
    ),
    "loo_variances": (
        0.1807729316221638,
        0.06633424707783879,
        0.06684593570193603,
        0.05478398631028581,
        0.1264749173316276,
        0.13494405117324226,
        0.07844536322977246,
    ),
}
LINEAR: dict[str, Any] = {
    "beta": (0.3303673757072842, 0.05704524981118529, 1.0616659662253258),
    "variance": 0.17297838439682509,
    "probes": (
        (1.0364471130317554, 0.019821887620408964),
        (0.17987584254426986, 0.32941275964814587),
    ),
    "loo_means": (
        0.8123765773571598,
        0.9198535171499908,
        1.4280550814235822,
        1.1965443532167017,
        0.2730449507707772,
        1.554300663008165,
        1.0922320413581785,
    ),
    "loo_variances": (
        0.18160036333252844,
        0.04189999517437128,
        0.03961292253840265,
        0.03335107092480882,
        0.08509826676586103,
        0.12366074365567321,
        0.05359717768169615,
    ),
}


def _linear(x: tuple[float, ...]) -> tuple[float, ...]:
    """Return the linear regressors ``(1, x1, x2)``.

    References:
        [EQ-SUR-UK-001].

    """
    return (1.0, x[0], x[1])


def _branin(point: tuple[float, ...]) -> float:
    """Branin function with both inputs rescaled from ``[0, 1]``.

    References:
        Test function for [EQ-SUR-PCK-001].

    """
    x1 = 15.0 * point[0] - 5.0
    x2 = 15.0 * point[1]
    return (
        (x2 - 5.1 / (4.0 * math.pi**2) * x1 * x1 + 5.0 / math.pi * x1 - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * math.pi)) * math.cos(x1)
        + 10.0
    )


def _relative_error(
    predict: Callable[[Sequence[float]], float],
    points: Sequence[Sequence[float]],
    values: Sequence[float],
) -> float:
    """Return the validation relative mean squared error.

    References:
        [EQ-SUR-PCE-006].

    """
    xs = list(points)
    ys = list(values)
    mean = math.fsum(ys) / len(ys)
    spread = math.fsum((y - mean) ** 2 for y in ys) / len(ys)
    return (
        math.fsum((predict(x) - y) ** 2 for x, y in zip(xs, ys, strict=True))
        / len(ys)
        / spread
    )


class UniversalKrigingReferenceTests(unittest.TestCase):
    """Universal kriging against pinned NumPy values and closed forms."""

    def _check(self, model: UniversalKriging, reference: dict[str, Any]) -> None:
        """Compare a fixed-hyperparameter model with a pinned reference.

        References:
            [EQ-SUR-UK-001], [EQ-SUR-UK-003], [EQ-SUR-UK-004].

        """
        for got, want in zip(
            model.trend_coefficients,
            reference["beta"],
            strict=True,
        ):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(
            model.process_variance,
            reference["variance"],
            places=12,
        )
        for probe, (mean, variance) in zip(
            PROBES,
            reference["probes"],
            strict=True,
        ):
            got_mean, got_variance = model.predict_with_variance(probe)
            self.assertAlmostEqual(got_mean, mean, places=11)
            self.assertAlmostEqual(got_variance, variance, places=11)
        loo = model.leave_one_out()
        for got, want in zip(loo.means, reference["loo_means"], strict=True):
            self.assertAlmostEqual(got, want, places=10)
        for got, want in zip(
            loo.variances,
            reference["loo_variances"],
            strict=True,
        ):
            self.assertAlmostEqual(got, want, places=10)

    def test_ordinary_kriging_matches_reference(self) -> None:
        """The constant regressor reproduces ordinary kriging."""
        model = fit_universal_kriging(POINTS, VALUES, length_scales=SCALES)
        self._check(model, ORDINARY)
        explicit = fit_universal_kriging(
            POINTS, VALUES, regressors=[lambda x: 1.0], length_scales=SCALES
        )
        self.assertEqual(explicit.trend_coefficients, model.trend_coefficients)
        self.assertEqual(explicit.predict(PROBES[0]), model.predict(PROBES[0]))

    def test_linear_trend_matches_reference(self) -> None:
        """A linear trend reproduces the bordered-matrix formulas."""
        model = fit_universal_kriging(
            POINTS, VALUES, regressors=_linear, length_scales=SCALES
        )
        self._check(model, LINEAR)
        self.assertEqual(model.size, 3)
        self.assertAlmostEqual(
            model.trend(PROBES[0]),
            math.fsum(
                b * f for b, f in zip(LINEAR["beta"], _linear(PROBES[0]), strict=True)
            ),
            places=12,
        )

    def test_fitted_model_interpolates_the_nodes(self) -> None:
        """At the training points the mean is the data and the variance ~0."""
        model = fit_universal_kriging(POINTS, VALUES, regressors=_linear, seed=3)
        self.assertTrue(model.converged)
        self.assertEqual(len(model.length_scales), 2)
        for point, value in zip(POINTS, VALUES, strict=True):
            mean, variance = model.predict_with_variance(point)
            self.assertAlmostEqual(mean, value, delta=1e-6)
            self.assertLess(variance, 1e-8 * model.process_variance)
        self.assertGreater(model.predict_with_variance(PROBES[1])[1], 0.0)

    def test_closed_form_loo_matches_brute_force_refits(self) -> None:
        """Dubrule's LOO equals refits without each point at fixed ``ell``."""
        model = fit_universal_kriging(POINTS, VALUES, regressors=_linear, seed=3)
        loo = model.leave_one_out()
        for i in range(len(POINTS)):
            points = POINTS[:i] + POINTS[i + 1 :]
            values = VALUES[:i] + VALUES[i + 1 :]
            refit = fit_universal_kriging(
                points,
                values,
                regressors=_linear,
                length_scales=model.length_scales,
            )
            mean, variance = refit.predict_with_variance(POINTS[i])
            self.assertAlmostEqual(loo.means[i], mean, delta=1e-9)
            # The closed form keeps the full-data sigma^2.
            rescaled = variance * model.process_variance / refit.process_variance
            self.assertAlmostEqual(loo.variances[i], rescaled, delta=1e-9)
            self.assertAlmostEqual(loo.residuals[i], VALUES[i] - mean, delta=1e-9)
        center = math.fsum(VALUES) / len(VALUES)
        spread = math.fsum((v - center) ** 2 for v in VALUES) / (len(VALUES) - 1)
        expected = math.fsum(r * r for r in loo.residuals) / len(VALUES) / spread
        self.assertAlmostEqual(loo.relative_error, expected, places=14)
        self.assertEqual(model.loo_error, loo.relative_error)

    def test_exact_polynomial_trend_has_no_residual_process(self) -> None:
        """A quadratic trend reproduces a quadratic exactly and extrapolates it."""

        def quadratic(x: tuple[float, ...]) -> float:
            """Return the target polynomial.

            References:
                [EQ-SUR-UK-003].

            """
            return 1.0 + 2.0 * x[0] - 0.5 * x[1] + 0.75 * x[0] * x[1] - x[1] ** 2

        regressors = [
            lambda x: 1.0,
            lambda x: x[0],
            lambda x: x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] ** 2,
            lambda x: x[1] ** 2,
        ]
        values = [quadratic(point) for point in POINTS]
        model = fit_universal_kriging(POINTS, values, regressors=regressors, seed=1)
        self.assertTrue(model.exact_trend)
        expected = (1.0, 2.0, -0.5, 0.75, 0.0, -1.0)
        for got, want in zip(model.trend_coefficients, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-9)
        for point in ((3.0, -2.0), (-4.0, 5.0), (0.5, 0.5)):
            mean, variance = model.predict_with_variance(point)
            self.assertAlmostEqual(mean, quadratic(point), delta=1e-8)
            self.assertLess(variance, 1e-20)

    def test_objective_gradient_matches_finite_differences(self) -> None:
        """The concentrated-likelihood gradient is exact for every kernel."""
        data = _pair_data(POINTS)
        rows = tuple(_linear(point) for point in POINTS)
        for kernel in KERNELS:
            for estimate in (False, True):
                theta = [math.log(0.4), math.log(0.6)]
                if estimate:
                    theta.append(math.log(1e-3))
                _, gradient = _objective(
                    data, VALUES, rows, theta, 2, 1e-10, estimate, kernel
                )
                for j in range(len(theta)):
                    up = list(theta)
                    down = list(theta)
                    up[j] += 1e-6
                    down[j] -= 1e-6
                    numeric = (
                        _objective(data, VALUES, rows, up, 2, 1e-10, estimate, kernel)[
                            0
                        ]
                        - _objective(
                            data, VALUES, rows, down, 2, 1e-10, estimate, kernel
                        )[0]
                    ) / 2e-6
                    self.assertAlmostEqual(gradient[j], numeric, delta=1e-6)

    def test_estimated_nugget_smooths_noisy_data(self) -> None:
        """With noise the estimated nugget grows and enters the noisy variance."""
        generator = resolve_stream(7).generator()
        points = [(i / 19.0,) for i in range(20)]
        values = [1.0 + 2.0 * x[0] + 0.05 * generator.gauss(0.0, 1.0) for x in points]
        model = fit_universal_kriging(
            points,
            values,
            regressors=lambda x: (1.0, x[0]),
            estimate_nugget=True,
            seed=2,
        )
        self.assertGreater(model.nugget, 1e-6)
        latent = model.predict_with_variance((0.5,))[1]
        noisy = model.predict_with_variance((0.5,), include_noise=True)[1]
        self.assertAlmostEqual(
            noisy - latent, model.process_variance * model.nugget, places=12
        )
        self.assertAlmostEqual(model.trend_coefficients[1], 2.0, delta=0.2)

    def test_invalid_arguments_raise(self) -> None:
        """Bad shapes, kernels, regressors, and option mixes are refused."""
        with self.assertRaises(ValueError):
            fit_universal_kriging(POINTS, VALUES, kernel="cubic")
        with self.assertRaises(ValueError):
            fit_universal_kriging(POINTS, VALUES, nugget=0.0)
        with self.assertRaises(ValueError):
            fit_universal_kriging(POINTS, VALUES[:3])
        with self.assertRaises(TypeError):
            fit_universal_kriging(POINTS, VALUES, regressors=[1.0])
        with self.assertRaises(ValueError):
            fit_universal_kriging(
                POINTS[:3], VALUES[:3], regressors=_linear, length_scales=SCALES
            )
        with self.assertRaises(ValueError):
            fit_universal_kriging(
                POINTS, VALUES, length_scales=SCALES, estimate_nugget=True
            )
        with self.assertRaises(ValueError):
            fit_universal_kriging(POINTS, VALUES, length_scales=(0.4,))
        with self.assertRaises(ValueError):
            fit_universal_kriging(
                POINTS, VALUES, regressors=lambda x: (1.0, 1.0), length_scales=SCALES
            )
        model = fit_universal_kriging(POINTS, VALUES, length_scales=SCALES)
        with self.assertRaises(ValueError):
            model.predict((0.1,))


class Matern32KernelTests(unittest.TestCase):
    """The Matern 3/2 kernel joins the GP kernel set."""

    def test_kernel_value_and_set(self) -> None:
        """Matern 3/2 has the closed form ``(1 + sqrt(3) r) exp(-sqrt(3) r)``."""
        self.assertEqual(KERNELS, ("squared_exponential", "matern52", "matern32"))
        r = math.sqrt((0.3 / 0.5) ** 2 + (0.4 / 2.0) ** 2)
        expected = 2.0 * (1.0 + math.sqrt(3.0) * r) * math.exp(-math.sqrt(3.0) * r)
        got = kernel_value("matern32", (0.0, 0.0), (0.3, 0.4), (0.5, 2.0), 2.0)
        self.assertAlmostEqual(got, expected, places=15)

    def test_gp_likelihood_gradient(self) -> None:
        """The GP log marginal likelihood gradient is exact for Matern 3/2."""
        targets = [(v - 1.0) for v in VALUES]
        theta = [0.2, math.log(0.4), math.log(0.6)]
        _, gradient = gaussian_process_log_marginal_likelihood(
            POINTS, targets, theta, kernel="matern32"
        )
        for j in range(3):
            up = list(theta)
            down = list(theta)
            up[j] += 1e-6
            down[j] -= 1e-6
            numeric = (
                gaussian_process_log_marginal_likelihood(
                    POINTS, targets, up, kernel="matern32"
                )[0]
                - gaussian_process_log_marginal_likelihood(
                    POINTS, targets, down, kernel="matern32"
                )[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[j], numeric, delta=1e-5)
        fitted = fit_gaussian_process(POINTS, VALUES, kernel="matern32", seed=1)
        self.assertAlmostEqual(fitted.predict(POINTS[2]), VALUES[2], delta=1e-6)


def _branin_model() -> ProbabilisticModel:
    """Return the Branin function on independent uniforms on ``[0, 1]^2``.

    References:
        Test model for [EQ-SUR-PCK-001].

    """
    return ProbabilisticModel(
        JointInput.independent(
            RandomInput("x1", Uniform(0.0, 1.0)), RandomInput("x2", Uniform(0.0, 1.0))
        ),
        lambda p: {"y": _branin((p["x1"], p["x2"]))},
        (QuantityOfInterest("y", "Branin function"),),
    )


class PCKrigingTests(unittest.TestCase):
    """PC-Kriging against plain GP and sparse PCE at small designs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the shared surrogates once.

        References:
            [EQ-SUR-PCK-001].

        """
        warnings.simplefilter("ignore")
        cls.branin = _branin_model()
        cls.branin_fits = {
            variant: fit_pc_kriging(
                cls.branin,
                size=30,
                degrees=range(1, 16),
                basis="legendre",
                variant=variant,
                seed=1,
            )
            for variant in PC_KRIGING_VARIANTS
        }
        unit = make_design("mc", 2000, 2, seed=resolve_stream(999))
        cls.branin_x = cls.branin.inputs.sample_design(unit)
        cls.branin_y = cls.branin.evaluate_many(cls.branin_x)["y"]
        cls.ishigami = ishigami_model()
        cls.ishigami_fits = {
            variant: fit_pc_kriging(
                cls.ishigami,
                size=40,
                degrees=range(1, 11),
                q_norms=(0.5, 0.75, 1.0),
                basis="legendre",
                variant=variant,
                seed=1,
            )
            for variant in PC_KRIGING_VARIANTS
        }
        unit = make_design("mc", 1000, 3, seed=resolve_stream(999))
        cls.ishigami_x = cls.ishigami.inputs.sample_design(unit)
        cls.ishigami_y = cls.ishigami.evaluate_many(cls.ishigami_x)["y"]

    def _errors(
        self,
        fits: dict[str, PCKriging],
        xs: Sequence[Sequence[float]],
        ys: Sequence[float],
    ) -> dict[str, float]:
        """Return validation errors of PC-Kriging, sparse PCE, and plain GP.

        References:
            [EQ-SUR-PCK-001].

        """
        sequential = fits["sequential"]
        gp = fit_gaussian_process(sequential.design, sequential.responses, seed=1)
        return {
            "sequential": _relative_error(sequential.predict, xs, ys),
            "optimal": _relative_error(fits["optimal"].predict, xs, ys),
            "pce": _relative_error(sequential.expansion.predict, xs, ys),
            "gp": _relative_error(gp.predict, xs, ys),
        }

    def test_optimal_loo_never_exceeds_sequential(self) -> None:
        """The optimal variant's last candidate is the sequential model."""
        for fits in (self.branin_fits, self.ishigami_fits):
            sequential, optimal = fits["sequential"], fits["optimal"]
            self.assertLessEqual(optimal.loo_error, sequential.loo_error)
            self.assertEqual(optimal.candidate_loo_errors[-1], sequential.loo_error)
            self.assertEqual(min(optimal.candidate_loo_errors), optimal.loo_error)
            self.assertEqual(len(sequential.candidate_loo_errors), 1)
            ranked = len(sequential.trend_indices)
            self.assertEqual(len(optimal.candidate_loo_errors), ranked)
            self.assertTrue(all(a == 0 for a in sequential.trend_indices[0]))
            self.assertEqual(
                set(sequential.trend_indices), set(sequential.expansion.indices)
            )
            self.assertEqual(
                optimal.trend_indices,
                sequential.trend_indices[: len(optimal.trend_indices)],
            )

    def test_branin_beats_gp_and_sparse_pce(self) -> None:
        """On Branin with 30 runs PC-Kriging is far more accurate than both."""
        errors = self._errors(self.branin_fits, self.branin_x, self.branin_y)
        for variant in PC_KRIGING_VARIANTS:
            self.assertLess(errors[variant], 0.2 * errors["gp"])
            self.assertLess(errors[variant], 0.01 * errors["pce"])

    def test_ishigami_is_comparable_to_the_best_competitor(self) -> None:
        """On Ishigami with 40 runs PC-Kriging matches sparse PCE, beats GP."""
        errors = self._errors(self.ishigami_fits, self.ishigami_x, self.ishigami_y)
        for variant in PC_KRIGING_VARIANTS:
            self.assertLess(errors[variant], 1.1 * errors["pce"])
            self.assertLess(errors[variant], 0.6 * errors["gp"])

    def test_predictions_use_the_standardized_coordinates(self) -> None:
        """PC-Kriging predicts through the kriging model on ``xi(x)``."""
        fit = self.branin_fits["sequential"]
        point = (0.3, 0.7)
        xi = fit.standardize(point)
        self.assertAlmostEqual(xi[0], -0.4, places=14)
        self.assertAlmostEqual(xi[1], 0.4, places=14)
        self.assertEqual(fit.predict(point), fit.kriging.predict(xi))
        self.assertEqual(
            fit.predict_with_variance(point), fit.kriging.predict_with_variance(xi)
        )
        self.assertEqual(fit.length_scales, fit.kriging.length_scales)
        self.assertEqual(fit.process_variance, fit.kriging.process_variance)
        self.assertEqual(
            fit.leave_one_out().relative_error,
            fit.kriging.leave_one_out().relative_error,
        )
        trend = fit.kriging.regressor
        self.assertIsInstance(trend, ChaosTrend)
        self.assertEqual(len(trend(xi)), len(fit.trend_indices))
        for x, y in zip(fit.design[:5], fit.responses[:5], strict=True):
            self.assertAlmostEqual(fit.predict(x), y, delta=1e-6 * abs(y) + 1e-8)

    def test_samples_tuple_matches_model_path(self) -> None:
        """Passing ``(inputs, points, values)`` refits the same surrogate."""
        fit = self.branin_fits["sequential"]
        again = fit_pc_kriging(
            (self.branin.inputs, fit.design, fit.responses),
            degrees=range(1, 16),
            basis="legendre",
            seed=1,
        )
        self.assertEqual(again.trend_indices, fit.trend_indices)
        self.assertEqual(again.kriging.length_scales, fit.kriging.length_scales)
        self.assertEqual(again.predict((0.2, 0.9)), fit.predict((0.2, 0.9)))
        self.assertEqual(again.quantity, "y")

    def test_surrogate_plugs_into_the_central_model(self) -> None:
        """A PC-Kriging fit replaces the model's forward map."""
        fit = self.branin_fits["optimal"]
        replaced = surrogate_probabilistic_model(self.branin, {"y": fit})
        point = {"x1": 0.25, "x2": 0.5}
        self.assertAlmostEqual(
            replaced.evaluate(point)["y"], fit.predict((0.25, 0.5)), places=14
        )
        mean, variance = fit.predict_with_variance((0.25, 0.5))
        self.assertIsInstance(mean, float)
        self.assertGreaterEqual(variance, 0.0)

    def test_hermite_basis_on_gaussian_inputs(self) -> None:
        """The default Hermite basis works on Gaussian inputs."""
        model = ProbabilisticModel(
            JointInput.independent(
                RandomInput("a", Normal(1.0, 0.5)), RandomInput("b", Normal(0.0, 2.0))
            ),
            lambda p: {"y": p["a"] ** 2 + math.sin(p["b"])},
            (QuantityOfInterest("y", "test"),),
        )
        fit = pc_kriging_from_samples(
            model.inputs,
            *_samples(model, 25),
            degrees=range(1, 5),
            kernel="matern32",
            seed=4,
        )
        self.assertEqual(fit.basis, "hermite")
        self.assertEqual(fit.kriging.kernel, "matern32")
        self.assertLess(fit.loo_error, 0.2)

    def test_invalid_arguments_raise(self) -> None:
        """Unknown variants, missing sizes, and bad bases are refused."""
        with self.assertRaises(ValueError):
            fit_pc_kriging(self.branin, size=20, variant="best", basis="legendre")
        with self.assertRaises(ValueError):
            fit_pc_kriging(self.branin, basis="legendre")
        with self.assertRaises(ValueError):
            fit_pc_kriging(self.branin, size=2, basis="legendre")
        with self.assertRaises(ValueError):
            fit_pc_kriging(self.ishigami, size=20, basis="legendre", kernel="rbf")
        normal = ProbabilisticModel(
            JointInput.independent(RandomInput("a", Normal(0.0, 1.0))),
            lambda p: {"y": p["a"]},
            (QuantityOfInterest("y", "test"),),
        )
        with self.assertRaises(ValueError):
            fit_pc_kriging(normal, size=10, basis="legendre")


def _samples(
    model: ProbabilisticModel, size: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return an LHS design of a model and its responses.

    References:
        [EQ-SAMP-INV-001].

    """
    unit = make_design("lhs", size, model.dimension, seed=resolve_stream(11))
    points = model.inputs.sample_design(unit)
    return points, tuple(model.evaluate_many(points)["y"])


if __name__ == "__main__":
    unittest.main()
