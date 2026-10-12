"""Regression tests for GP hyperparameter regularization and the CE coverage check.

ADR-0054. The benchmark suite (ADR-0049) found that unregularized maximum
likelihood ends at near-singular GP hyperparameters on smooth limit states
(AK-MCS then stops after its initial design with ``min U`` in the hundreds)
and at a white-noise optimum on the four-branch function, and that
cross-entropy importance sampling misses three of four failure regions of the
four-branch system without any warning.
"""

from __future__ import annotations

import math
import unittest
import warnings
from collections.abc import Callable, Mapping

from probstat.distributions import Normal
from probstat.foundations.random_streams import resolve_stream
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.optimization import central_gradient
from probstat.propagation import make_design
from probstat.reliability import (
    AKMCS_GP_REGULARIZATION,
    CE_MAX_UNCOVERED_ELITE_FRACTION,
    ModelLimitState,
    ak_mcs,
    cross_entropy_importance_sampling,
)
from probstat.surrogates import (
    GaussianProcessRegularization,
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
    gaussian_process_log_parameter_bounds,
)
from probstat.uq import QuantityOfInterest

SQRT2 = math.sqrt(2.0)


def parabola(u: tuple[float, ...]) -> float:
    """Return the benchmark parabolic limit state ``3 + u1^2 / 4 - u2``."""
    return 3.0 + 0.25 * u[0] ** 2 - u[1]


def four_branch(u: tuple[float, ...]) -> float:
    """Return the four-branch series system (Echard et al. 2011, ``k = 6``)."""
    x1, x2 = u
    s = 0.1 * (x1 - x2) ** 2
    return min(
        3.0 + s - (x1 + x2) / SQRT2,
        3.0 + s + (x1 + x2) / SQRT2,
        (x1 - x2) + 6.0 / SQRT2,
        (x2 - x1) + 6.0 / SQRT2,
    )


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int = 2
) -> ModelLimitState:
    """Wrap ``G(u)`` on standard normal inputs as a ``G <= 0`` limit state."""
    names = tuple(f"u{i + 1}" for i in range(dimension))

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from named inputs."""
        return {"y": function(tuple(p[name] for name in names))}

    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )
    model = ProbabilisticModel(
        inputs, response, (QuantityOfInterest("y", "response"),)
    )
    return ModelLimitState(model, "y", 0.0, "below")


def akmcs_design(seed: int) -> list[tuple[float, ...]]:
    """Return the 12-point LHS initial design AK-MCS uses for ``seed``."""
    unit = make_design(
        "lhs", 12, 2, seed=resolve_stream(seed).child("ak-mcs", "design")
    )
    return [tuple(5.0 * (2.0 * p - 1.0) for p in point) for point in unit.points]


class GaussianProcessRegularizationTests(unittest.TestCase):
    """Bounded, floored hyperparameter estimation of GP regression."""

    def test_parabolic_fit_is_no_longer_near_singular(self) -> None:
        """Regularization removes the singular optimum and keeps a variance.

        Regression: on the AK-MCS design (seed 1) of ``3 + u1^2/4 - u2``
        unregularized maximum likelihood gives length scales ``1.5e3`` and
        ``1.6e5``, signal-to-noise ``4.7e18``, and predictive standard
        deviation exactly ``0`` off the design.
        """
        points = akmcs_design(1)
        values = [parabola(u) for u in points]
        legacy = fit_gaussian_process(points, values, seed=3)
        self.assertGreater(min(legacy.hyperparameters.length_scales), 500.0)
        self.assertEqual(legacy.predict_with_variance((0.5, 2.0))[1], 0.0)
        self.assertEqual(legacy.log_parameter_bounds, ((-25.0, 25.0),) * 3)
        settings = GaussianProcessRegularization()
        fitted = fit_gaussian_process(
            points, values, seed=3, regularization=settings
        )
        signal = math.exp(fitted.log_parameters[0])
        self.assertLessEqual(
            signal / fitted.standardized_noise_variance,
            1.0 / settings.min_relative_nugget,
        )
        for value, (lower, upper) in zip(
            fitted.log_parameters, fitted.log_parameter_bounds, strict=True
        ):
            self.assertTrue(lower <= value <= upper)
        # The trend in u2 is linear: its length scale ends on the upper bound.
        self.assertIn("length_scale[1]:upper", fitted.at_bound)
        self.assertTrue(fitted.converged)
        mean, variance = fitted.predict_with_variance((0.5, 2.0))
        sd = math.sqrt(variance)
        self.assertGreater(sd, 0.01)
        self.assertLess(abs(mean - parabola((0.5, 2.0))), 3.0 * sd)

    def test_white_noise_optimum_is_excluded(self) -> None:
        """Length scales cannot fall far below the smallest design gap.

        Regression: on the four-branch AK-MCS design of seed 5 the
        unregularized fit ends at length scales ``0.0017`` and ``0.0020``
        (smallest coordinate gap ``0.33``), a white-noise fit whose mean is
        the output mean everywhere; warm-started refits never left it.
        """
        points = akmcs_design(5)
        values = [four_branch(u) for u in points]
        stream = resolve_stream(5).child("ak-mcs", "fit", 1)
        legacy = fit_gaussian_process(points, values, seed=stream)
        self.assertLess(max(legacy.hyperparameters.length_scales), 0.01)
        settings = GaussianProcessRegularization()
        fitted = fit_gaussian_process(
            points, values, seed=stream, regularization=settings
        )
        bounds = gaussian_process_log_parameter_bounds(points, settings)
        self.assertEqual(fitted.log_parameter_bounds, bounds)
        for axis, scale in enumerate(fitted.hyperparameters.length_scales):
            column = sorted(point[axis] for point in points)
            gap = min(b - a for a, b in zip(column, column[1:], strict=False))
            self.assertGreater(scale, settings.lower_spacing_factor * gap)
            self.assertAlmostEqual(
                bounds[1 + axis][0], math.log(settings.lower_spacing_factor * gap)
            )
            self.assertGreater(scale, 0.5)
        self.assertEqual(fitted.at_bound, ())

    def test_relative_nugget_gradient_and_reported_likelihood(self) -> None:
        """The gradient with ``tau > 0`` is exact; the likelihood excludes the prior."""
        points = akmcs_design(2)
        values = [parabola(u) for u in points]
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(
            math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
        )
        targets = [(v - mean) / sd for v in values]
        theta = (0.4, math.log(2.0), math.log(3.0))

        def objective(t: tuple[float, ...]) -> float:
            """Return the log marginal likelihood with a relative nugget."""
            return gaussian_process_log_marginal_likelihood(
                points, targets, t, nugget=1e-8, relative_nugget=1e-3
            )[0]

        _, analytic = gaussian_process_log_marginal_likelihood(
            points, targets, theta, nugget=1e-8, relative_nugget=1e-3
        )
        numeric, _ = central_gradient(objective, theta)
        for a, b in zip(analytic, numeric, strict=True):
            self.assertAlmostEqual(a, b, 5)
        settings = GaussianProcessRegularization(length_scale_prior_sd=0.5)
        fitted = fit_gaussian_process(
            points, values, seed=2, regularization=settings
        )
        plain = fit_gaussian_process(
            points, values, seed=2, regularization=GaussianProcessRegularization()
        )
        # The prior pulls the near-linear direction back towards the extent.
        self.assertLess(
            fitted.hyperparameters.length_scales[1],
            plain.hyperparameters.length_scales[1],
        )
        value, _ = gaussian_process_log_marginal_likelihood(
            points,
            targets,
            fitted.log_parameters,
            nugget=fitted.nugget,
            relative_nugget=settings.min_relative_nugget,
        )
        self.assertAlmostEqual(
            fitted.log_marginal_likelihood, value - len(values) * math.log(sd), 9
        )

    def test_validation_and_default_unchanged(self) -> None:
        """Bad settings raise; the default fit is the unregularized one."""
        for bad in (
            {"lower_spacing_factor": 0.0},
            {"upper_extent_factor": -1.0},
            {"signal_variance_bounds": (2.0, 1.0)},
            {"min_relative_nugget": -1e-9},
            {"length_scale_prior_sd": 0.0},
        ):
            with self.assertRaises(ValueError):
                GaussianProcessRegularization(**bad)  # type: ignore[arg-type]
        points = akmcs_design(1)
        values = [parabola(u) for u in points]
        with self.assertRaises(TypeError):
            fit_gaussian_process(points, values, regularization="on")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gaussian_process_log_parameter_bounds(
                [(0.0, 1.0), (0.0, 2.0)], GaussianProcessRegularization()
            )
        default = fit_gaussian_process(points, values, seed=4)
        explicit = fit_gaussian_process(points, values, seed=4, regularization=None)
        self.assertEqual(default.log_parameters, explicit.log_parameters)
        self.assertEqual(default.relative_nugget, 0.0)
        self.assertIsNone(default.regularization)


class AKMCSRegularizationTests(unittest.TestCase):
    """AK-MCS with the regularized GP fit (the new default)."""

    def test_parabola_no_longer_stops_on_a_singular_surrogate(self) -> None:
        """``min U`` stays near the threshold instead of the hundreds.

        Regression (ADR-0049 open issue): unregularized, the 12-point design
        gives ``min U = 1.2e3`` and learning stops at once; the population
        signs happened to be right at ``N = 2000``, but at ``N = 50 000``
        such surrogates misclassified 1-2 points with ``min U`` 278-639.
        """
        state = make_state(parabola)
        legacy = ak_mcs(
            state, 2000, max_population=2000, target_cov=1.0, seed=1,
            regularization=None,
        )
        self.assertGreater(legacy.min_learning, 100.0)
        result = ak_mcs(
            make_state(parabola),
            2000,
            max_population=2000,
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=1,
        )
        self.assertIs(result.surrogate.regularization, AKMCS_GP_REGULARIZATION)
        self.assertGreaterEqual(result.min_learning, 2.0)
        self.assertLess(result.min_learning, 20.0)
        self.assertEqual(result.misclassified, 0)
        signal = math.exp(result.surrogate.log_parameters[0])
        self.assertLessEqual(
            signal / result.surrogate.standardized_noise_variance, 1e8
        )

    def test_four_branch_seed_five_learns_normally(self) -> None:
        """No white-noise surrogate: about 30 calls and exact signs.

        Regression: unregularized, seed 5 kept the white-noise optimum of
        its first fit, added 198 points (42 s), and misclassified 7 of 2000
        population points (``P_f = 1.0e-3`` against ``4.5e-3``).
        """
        result = ak_mcs(
            make_state(four_branch),
            2000,
            max_population=2000,
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=5,
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.evaluations, 40)
        self.assertEqual(result.misclassified, 0)
        self.assertEqual(result.probability, result.reference_probability)


class CrossEntropyCoverageTests(unittest.TestCase):
    """The CE-IS coverage check flags missed failure regions."""

    def test_four_branch_warns(self) -> None:
        """One Gaussian proposal on four failure regions is flagged.

        Regression: the run below returns ``1.39e-3`` for an exact
        ``4.46e-3`` (``z = -29``) with a healthy failure-weight ESS (126) and
        used to emit no warning.
        """
        with self.assertWarnsRegex(UserWarning, "mixture_importance_sampling"):
            result = cross_entropy_importance_sampling(
                make_state(four_branch), samples_per_level=500, seed=3
            )
        self.assertTrue(result.multiple_failure_regions_suspected)
        self.assertGreater(result.uncovered_elite_fraction, 0.3)
        self.assertFalse(result.weights_degenerate)
        self.assertLess(result.probability, 0.5 * 4.457331e-3)

    def test_unimodal_linear_case_is_quiet(self) -> None:
        """A single planar failure region gives no coverage warning."""
        state = make_state(lambda u: 3.5 - (u[0] + u[1]) / SQRT2)
        for seed in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = cross_entropy_importance_sampling(
                    state, samples_per_level=500, seed=seed
                )
            self.assertFalse(result.multiple_failure_regions_suspected)
            self.assertLessEqual(
                result.uncovered_elite_fraction, CE_MAX_UNCOVERED_ELITE_FRACTION
            )
            self.assertEqual(result.uncovered_elite_fraction, 0.0)

    def test_check_is_skipped_near_the_origin(self) -> None:
        """An adapted proposal centred near the origin covers every direction."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cross_entropy_importance_sampling(
                make_state(four_branch),
                samples_per_level=1000,
                adapt_covariance=True,
                seed=1,
            )
        self.assertTrue(math.isnan(result.uncovered_elite_fraction))
        self.assertFalse(result.multiple_failure_regions_suspected)


if __name__ == "__main__":
    unittest.main()
