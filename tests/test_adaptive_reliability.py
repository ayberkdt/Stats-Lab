"""Tests for line sampling, multiple design points, AK-MCS, and the CE guard."""

from __future__ import annotations

import functools
import math
import unittest
import warnings
from collections.abc import Callable, Mapping

from probstat.distributions import Normal
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.reliability import (
    AKMCSResult,
    ModelLimitState,
    ak_mcs,
    bivariate_normal_cdf,
    bulge_value,
    cross_entropy_importance_sampling,
    find_design_points,
    form_hlrf,
    line_sampling,
    mixture_importance_sampling,
    multi_point_form,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)
SQRT2 = math.sqrt(2.0)


def standard_inputs(dimension: int) -> JointInput:
    """Return ``dimension`` independent standard normal inputs ``u1..ud``."""
    return JointInput.independent(
        *(RandomInput(f"u{i + 1}", Normal(0.0, 1.0)) for i in range(dimension))
    )


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on standard normal inputs as a ``g <= 0`` limit state."""
    names = tuple(f"u{i + 1}" for i in range(dimension))

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from the named inputs."""
        return {"y": function(tuple(p[name] for name in names))}

    model = ProbabilisticModel(standard_inputs(dimension), response, Y)
    return ModelLimitState(model, "y", 0.0, "below")


def linear_state(beta: float = 4.0) -> ModelLimitState:
    """Return ``G = beta - (u1 + u2) / sqrt 2`` with ``P_f = Phi(-beta)``."""
    return make_state(lambda u: beta - (u[0] + u[1]) / SQRT2, 2)


def plane_state(beta: float, alpha: tuple[float, ...]) -> ModelLimitState:
    """Return the hyperplane ``G = beta - alpha . u`` for a unit ``alpha``."""
    return make_state(
        lambda u: beta - math.fsum(a * x for a, x in zip(alpha, u, strict=True)),
        len(alpha),
    )


@functools.cache
def parabolic_exact(kappa: float, beta: float = 3.0) -> float:
    """Return ``E[Phi(-(beta + kappa Z^2 / 2))]`` by Gauss-Hermite quadrature."""
    nodes, weights = gauss_hermite(60)
    return math.fsum(
        w * standard_normal_cdf(-(beta + 0.5 * kappa * z * z))
        for z, w in zip(nodes, weights, strict=True)
    )


B1, B2, A2 = 3.0, 3.2, (-0.6, 0.8)


def two_planes(u: tuple[float, ...]) -> float:
    """Return ``min(B1 - u1, B2 - A2 . u)``: two linear modes, rho = -0.6."""
    return min(B1 - u[0], B2 - (A2[0] * u[0] + A2[1] * u[1]))


def two_planes_exact() -> float:
    """Return the exact union probability of the two half-planes."""
    rho = A2[0]
    return (
        standard_normal_cdf(-B1)
        + standard_normal_cdf(-B2)
        - bivariate_normal_cdf(-B1, -B2, rho)
    )


def four_branch(u: tuple[float, ...], k: float = 6.0) -> float:
    """Return the classic four-branch series system (Echard et al. 2011)."""
    x1, x2 = u
    s = 0.1 * (x1 - x2) ** 2
    return min(
        3.0 + s - (x1 + x2) / SQRT2,
        3.0 + s + (x1 + x2) / SQRT2,
        (x1 - x2) + k / SQRT2,
        (x2 - x1) + k / SQRT2,
    )


class CrossEntropyGuardTests(unittest.TestCase):
    """The covariance floor fixes CE-IS bias and degeneracy is reported."""

    exact = standard_normal_cdf(-4.0)

    def test_unguarded_adaptation_warns(self) -> None:
        """Without the floor the proposal collapses and a warning fires."""
        with self.assertWarns(UserWarning):
            result = cross_entropy_importance_sampling(
                linear_state(),
                samples_per_level=1000,
                adapt_covariance=True,
                covariance_floor=None,
                seed=1,
            )
        self.assertTrue(result.weights_degenerate)
        self.assertLessEqual(result.min_covariance_eigenvalue, 0.5)
        # Biased low far beyond its own reported error (measured: 0.38 x exact).
        self.assertLess(result.probability, 0.6 * self.exact)

    def test_floor_removes_bias(self) -> None:
        """Guarded runs agree with Phi(-4) within three standard errors."""
        for seed in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = cross_entropy_importance_sampling(
                    linear_state(),
                    samples_per_level=1000,
                    adapt_covariance=True,
                    seed=seed,
                )
            self.assertFalse(result.weights_degenerate)
            self.assertGreaterEqual(result.min_covariance_eigenvalue, 0.55 - 1e-12)
            self.assertLess(
                abs(result.probability - self.exact), 3.0 * result.standard_error
            )
            self.assertGreater(result.failure_effective_sample_size, 100.0)

    def test_mean_shift_path_unchanged(self) -> None:
        """The floor is ignored by the mean-shift default."""
        default = cross_entropy_importance_sampling(
            linear_state(), samples_per_level=500, seed=4
        )
        unguarded = cross_entropy_importance_sampling(
            linear_state(), samples_per_level=500, covariance_floor=None, seed=4
        )
        self.assertEqual(default.probability, unguarded.probability)
        self.assertEqual(default.min_covariance_eigenvalue, 1.0)
        self.assertFalse(default.weights_degenerate)
        with self.assertRaises(ValueError):
            cross_entropy_importance_sampling(
                linear_state(), adapt_covariance=True, covariance_floor=0.5
            )


class LineSamplingTests(unittest.TestCase):
    """Line sampling exactness, accuracy, and lines without a root."""

    def test_linear_limit_state_is_exact(self) -> None:
        """A hyperplane gives Phi(-beta) on every line: zero variance."""
        beta = 3.5
        direction = (1.0 / math.sqrt(3.0),) * 3
        state = plane_state(beta, direction)
        result = line_sampling(state, 30, seed=11)
        exact = standard_normal_cdf(-beta)
        self.assertAlmostEqual(result.probability / exact, 1.0, places=8)
        self.assertLess(result.coefficient_of_variation, 1e-7)
        self.assertGreater(result.form_evaluations, 0)

    def test_parabolic_limit_state_matches_quadrature(self) -> None:
        """A curved surface agrees with the closed form within three errors."""
        kappa = 0.3
        state = make_state(lambda u: 3.0 - u[1] + 0.5 * kappa * u[0] ** 2, 2)
        result = line_sampling(state, 200, seed=5)
        exact = parabolic_exact(kappa)
        self.assertLess(abs(result.probability - exact), 3.0 * result.standard_error)
        self.assertLess(result.coefficient_of_variation, 0.05)

    def test_lines_without_root_contribute_zero_or_one(self) -> None:
        """Safe-everywhere lines give 0, failed-everywhere lines give 1."""
        safe_band = make_state(
            lambda u: 3.0 - u[1] if abs(u[0]) < 1.0 else 5.0 + u[1] ** 2, 2
        )
        result = line_sampling(safe_band, 400, direction=(0.0, 1.0), seed=2)
        exact = (2.0 * standard_normal_cdf(1.0) - 1.0) * standard_normal_cdf(-3.0)
        self.assertLess(abs(result.probability - exact), 3.0 * result.standard_error)
        zeros = [
            p
            for p, c in zip(result.line_probabilities, result.distances, strict=True)
            if c == math.inf
        ]
        self.assertTrue(zeros)
        self.assertTrue(all(p == 0.0 for p in zeros))
        failed_band = make_state(
            lambda u: -1.0 - u[1] ** 2 if u[0] > 1.5 else 3.0 - u[1], 2
        )
        result = line_sampling(failed_band, 400, direction=(0.0, 1.0), seed=3)
        tail = standard_normal_cdf(-1.5)
        exact = tail + (1.0 - tail) * standard_normal_cdf(-3.0)
        ones = [
            p
            for p, c in zip(result.line_probabilities, result.distances, strict=True)
            if c == -math.inf
        ]
        self.assertTrue(ones)
        self.assertTrue(all(p == 1.0 for p in ones))
        self.assertLess(abs(result.probability - exact), 3.0 * result.standard_error)


class MultipleDesignPointTests(unittest.TestCase):
    """Bulged searches, multi-point FORM, and mixture importance sampling."""

    def test_bulge_profile(self) -> None:
        """The bulge equals its height at the centre and vanishes at r."""
        self.assertEqual(bulge_value((1.0, 0.0), [(1.0, 0.0)], [2.0], [3.0]), 3.0)
        self.assertEqual(bulge_value((3.0, 0.0), [(1.0, 0.0)], [2.0], [3.0]), 0.0)
        self.assertAlmostEqual(
            bulge_value((2.0, 0.0), [(1.0, 0.0)], [2.0], [3.0]), 3.0 * 0.75**2
        )

    def test_two_linear_modes_match_bivariate_normal(self) -> None:
        """Both design points are found; multi-point FORM is exact."""
        state = make_state(two_planes, 2)
        result = find_design_points(state, 4)
        self.assertEqual(len(result.forms), 2)
        self.assertAlmostEqual(result.betas[0], B1, places=6)
        self.assertAlmostEqual(result.betas[1], B2, places=6)
        self.assertAlmostEqual(result.forms[1].design_point_u[0], B2 * A2[0], 5)
        self.assertEqual(result.evaluations, state.model.evaluations)
        exact = two_planes_exact()
        combined = multi_point_form(result)
        self.assertEqual(combined.method, "bivariate")
        self.assertAlmostEqual(combined.probability / exact, 1.0, places=6)
        self.assertAlmostEqual(combined.correlations[0][1], A2[0], places=6)
        self.assertLess(result.forms[0].probability, 0.7 * exact)
        self.assertLessEqual(combined.bounds.ditlevsen_lower, exact * (1 + 1e-6))
        self.assertGreaterEqual(combined.bounds.ditlevsen_upper, exact * (1 - 1e-6))

    def test_origin_start_alone_misses_the_second_mode(self) -> None:
        """A single restart from the origin is drawn back to the first point."""
        result = find_design_points(make_state(two_planes, 2), 3, starts="origin")
        self.assertEqual(len(result.forms), 1)
        self.assertIn("no start", result.message)

    def test_separation_of_variables_for_three_modes(self) -> None:
        """Genz SOV matches exact unions, including a rho = -1 pair."""
        beta, rho = 3.0, 0.5
        # alpha_i = sqrt(rho) e_0 + sqrt(1 - rho) e_i: equicorrelated modes.
        alphas = [
            (math.sqrt(rho),) + tuple(math.sqrt(1 - rho) * (i == j) for j in range(3))
            for i in range(3)
        ]
        forms = [form_hlrf(plane_state(beta, a)) for a in alphas]
        nodes, weights = gauss_hermite(80)
        safe = math.fsum(
            w
            * standard_normal_cdf((beta - math.sqrt(rho) * z) / math.sqrt(1 - rho)) ** 3
            for z, w in zip(nodes, weights, strict=True)
        )
        exact = 1.0 - safe
        combined = multi_point_form(forms, samples=4000, seed=1)
        self.assertEqual(combined.method, "genz-sov")
        self.assertAlmostEqual(combined.correlations[0][1], rho, places=6)
        self.assertLess(
            abs(combined.probability - exact),
            3.0 * combined.standard_error + 1e-9 * exact,
        )
        self.assertLess(combined.standard_error / exact, 0.04)
        tail = standard_normal_cdf(-beta)
        antipodal = [
            form_hlrf(make_state(lambda u: beta - u[0], 2)),
            form_hlrf(make_state(lambda u: beta + u[0], 2)),
            form_hlrf(make_state(lambda u: beta - u[1], 2)),
        ]
        exact = 1.0 - (1.0 - 2.0 * tail) * (1.0 - tail)
        combined = multi_point_form(antipodal, samples=2000, seed=2)
        # The rho = -1 pair folds into one two-sided interval: no sampling noise.
        self.assertLess(combined.standard_error, 1e-12)
        self.assertAlmostEqual(combined.probability / exact, 1.0, places=8)

    def test_mixture_importance_sampling_is_unbiased(self) -> None:
        """The mixture estimate covers both modes within three errors."""
        state = make_state(two_planes, 2)
        points = find_design_points(state, 2)
        exact = two_planes_exact()
        for weighting in ("equal", "form"):
            result = mixture_importance_sampling(
                state, points, 2000, weighting=weighting, seed=7
            )
            self.assertLess(
                abs(result.probability - exact), 3.0 * result.standard_error
            )
            self.assertLess(result.coefficient_of_variation, 0.08)
            self.assertEqual(result.evaluations, 2000)
        again = mixture_importance_sampling(state, points, 2000, seed=7)
        first = mixture_importance_sampling(state, points, 2000, seed=7)
        self.assertEqual(again.probability, first.probability)

    def test_invalid_inputs(self) -> None:
        """Unsafe origins and bad options raise explicit errors."""
        with self.assertRaises(ValueError):
            find_design_points(make_state(lambda u: -1.0 - u[0], 2))
        with self.assertRaises(ValueError):
            find_design_points(make_state(two_planes, 2), starts="random")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            multi_point_form([])
        with self.assertRaises(ValueError):
            mixture_importance_sampling(
                make_state(two_planes, 2),
                [form_hlrf(make_state(two_planes, 2))],
                weighting="bad",  # type: ignore[arg-type]
            )


class AKMCSTests(unittest.TestCase):
    """AK-MCS against crude Monte Carlo on the same population."""

    def test_four_branch_matches_crude_monte_carlo(self) -> None:
        """Same-population crude MC is reproduced with ~30 model calls."""
        state = make_state(four_branch, 2)
        result = ak_mcs(
            state, 5000, target_cov=1.0, compare_with_monte_carlo=True, seed=1
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.min_learning, 2.0)
        self.assertEqual(result.misclassified, 0)
        self.assertEqual(result.probability, result.reference_probability)
        self.assertGreater(result.probability, 0.0)
        self.assertLessEqual(result.evaluations, 60)
        self.assertEqual(result.evaluations, result.initial_size + result.added)
        self.assertEqual(result.reference_evaluations, 5000)
        self.assertEqual(
            state.model.evaluations, result.evaluations + result.reference_evaluations
        )

    def test_unconverged_estimate_uses_the_returned_surrogate(self) -> None:
        """After max_added, P_f equals the returned GP's population fraction."""
        from probstat.foundations.random_streams import resolve_stream

        state = make_state(four_branch, 2)
        result = ak_mcs(state, 5000, max_added=4, refit_growth=1.2, seed=3)
        self.assertFalse(result.converged)
        generator = resolve_stream(3).child("ak-mcs", "population").generator()
        population = [
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
            for _ in range(result.population)
        ]
        fraction = sum(
            1 for point in population if result.surrogate.predict(point) <= 0.0
        ) / len(population)
        self.assertAlmostEqual(result.probability, fraction, places=15)

    def test_population_posterior_matches_surrogate_on_near_singular_fit(
        self,
    ) -> None:
        """The population means agree with the returned GP on a parabola.

        Regression (benchmark suite, ADR-0049): on ``G = 3 + u1^2/4 - u2``
        the likelihood fit ends at length scales ~1e3-1e5 and a signal to
        nugget ratio near ``1/eps``. The incremental posterior re-derived the
        Cholesky factor in original units, its last pivots collapsed to the
        noise floor, and AK-MCS returned ``P_f = 0.641`` (1282 of 2000 points
        failing) although the returned surrogate itself predicts about 0.001
        (exact ``8.2e-4``).
        """
        from probstat.foundations.random_streams import resolve_stream

        state = make_state(lambda u: 3.0 + 0.25 * u[0] ** 2 - u[1], 2)
        result = ak_mcs(state, 2000, max_population=2000, target_cov=1.0, seed=2)
        generator = resolve_stream(2).child("ak-mcs", "population").generator()
        population = [
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
            for _ in range(result.population)
        ]
        fraction = sum(
            1 for point in population if result.surrogate.predict(point) <= 0.0
        ) / len(population)
        self.assertLessEqual(abs(result.probability - fraction), 1.0 / 2000)
        self.assertLess(result.probability, 0.01)

    def test_population_enlargement_and_determinism(self) -> None:
        """A c.o.v. target enlarges the population; seeds reproduce runs."""

        def run() -> AKMCSResult:
            """Run AK-MCS on a beta = 2.5 plane with a c.o.v. target."""
            return ak_mcs(
                linear_state(2.5),
                1500,
                target_cov=0.2,
                population_growth=1500,
                max_population=9000,
                seed=3,
            )

        first = run()
        second = run()
        self.assertEqual(first.probability, second.probability)
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(first.history, second.history)
        self.assertGreater(first.population, 1500)
        self.assertLessEqual(first.coefficient_of_variation, 0.2)
        exact = standard_normal_cdf(-2.5)
        self.assertLess(abs(first.probability - exact), 3.0 * first.standard_error)


if __name__ == "__main__":
    unittest.main()
