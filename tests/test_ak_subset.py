"""Tests for AK-SS (active-learning Kriging on subset-simulation levels)."""

from __future__ import annotations

import math
import random
import unittest
import warnings
from collections.abc import Callable, Mapping

from probstat.distributions import Normal
from probstat.modeling import (
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch_model,
    standard_normal_cdf,
)
from probstat.reliability import (
    AKSS_GP_REGULARIZATION,
    AKSSResult,
    GaussianProcessPopulationPosterior,
    ModelLimitState,
    ak_ss,
)
from probstat.surrogates import (
    GaussianProcessFactory,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)

# FORM design points of the four-branch function with k = 7: the two
# parabolic branches at distance 3, the two linear branches at distance 3.5.
_A = 3.0 / math.sqrt(2.0)
_B = 7.0 / (2.0 * math.sqrt(2.0))
FOUR_BRANCH_DESIGN_POINTS = ((_A, _A), (-_A, -_A), (_B, -_B), (-_B, _B))


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on standard normal inputs as a ``g <= 0`` limit state."""
    names = tuple(f"u{i + 1}" for i in range(dimension))

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from the named inputs."""
        return {"y": function(tuple(p[name] for name in names))}

    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )
    return ModelLimitState(ProbabilisticModel(inputs, response, Y), "y", 0.0, "below")


def linear_state(beta: float, dimension: int = 2) -> ModelLimitState:
    """Return ``G(u) = beta - sum u_i / sqrt(d)`` with ``P_f = Phi(-beta)``."""
    root = math.sqrt(dimension)
    return make_state(lambda u: beta - sum(u) / root, dimension)


def four_branch_state() -> ModelLimitState:
    """Return the four-branch limit state with ``k = 7``."""
    return ModelLimitState(four_branch_model(7.0), "g", 0.0, "below")


def within_error(result: AKSSResult, reference: float, factor: float) -> bool:
    """Return whether the estimate is within ``factor`` subset c.o.v. of the truth."""
    scale = factor * result.coefficient_of_variation * reference
    return abs(result.probability - reference) <= scale


class FourBranchTests(unittest.TestCase):
    """Four-branch series system, ``k = 7`` (reference ``2.2245e-3``)."""

    def test_estimate_calls_and_classification(self) -> None:
        """Two seeds: estimate within the error scale, O(50-150) calls."""
        for seed in (3, 4):
            with self.subTest(seed=seed):
                result = ak_ss(
                    four_branch_state(),
                    samples_per_level=1000,
                    check_population=10_000,
                    check_levels=True,
                    seed=seed,
                )
                self.assertTrue(result.converged, result.message)
                self.assertTrue(
                    within_error(result, FOUR_BRANCH_REFERENCE_PROBABILITY, 3.0),
                    (result.probability, result.coefficient_of_variation),
                )
                self.assertGreaterEqual(result.evaluations, 30)
                self.assertLessEqual(result.evaluations, 150)
                self.assertEqual(result.evaluations, result.initial_size + result.added)
                self.assertEqual(result.levels, 3)
                self.assertEqual(result.thresholds[-1], 0.0)
                self.assertEqual(len(result.conditional_probabilities), 3)
                self.assertEqual(result.conditional_probabilities[0], 0.1)
                product = math.prod(result.conditional_probabilities)
                self.assertAlmostEqual(
                    result.probability, product, delta=1e-12 * product
                )
                # Classification checks against the true limit state.
                self.assertEqual(result.check_evaluations, 10_000)
                self.assertLessEqual(result.check_misclassified, 3)
                self.assertLess(abs(result.check_probability - 2.2e-3), 1.5e-3)
                self.assertEqual(result.level_evaluations, result.pool_size)
                self.assertLessEqual(result.level_misclassified, 10)
                # Bounds and the per-round record.
                self.assertLessEqual(result.lower_probability, result.probability)
                self.assertGreaterEqual(result.upper_probability, result.probability)
                width = result.upper_probability - result.lower_probability
                self.assertLessEqual(width, 0.05 * result.probability + 1e-15)
                self.assertEqual(len(result.history), result.rounds)
                self.assertEqual(sum(r.added for r in result.history), result.added)
                self.assertEqual(result.history[-1].added, 0)
                self.assertEqual(result.history[0].training_size, 12)
                self.assertEqual(result.history[-1].probability, result.probability)
                self.assertIsInstance(result.surrogate, GaussianProcessRegression)
                self.assertEqual(len(result.training_points), result.evaluations)
                self.assertEqual(
                    len(result.surrogate.points),  # type: ignore[attr-defined]
                    result.evaluations,
                )
                self.assertAlmostEqual(
                    result.beta,
                    -_quantile(result.probability),
                    delta=1e-9,
                )

    def test_reproducible(self) -> None:
        """The same seed gives the same run."""
        state = linear_state(3.5)
        first = ak_ss(state, samples_per_level=500, seed=11)
        second = ak_ss(linear_state(3.5), samples_per_level=500, seed=11)
        self.assertEqual(first.probability, second.probability)
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(first.training_points, second.training_points)


def _quantile(probability: float) -> float:
    """Return ``Phi^-1(p)`` by bisection on the library CDF (test helper)."""
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if standard_normal_cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class RareEventTests(unittest.TestCase):
    """A failure probability far below the comfortable range of AK-MCS."""

    def test_linear_beta_four_and_a_half(self) -> None:
        """``P_f = Phi(-4.5) = 3.40e-6`` with tens of calls instead of ~3e6."""
        reference = standard_normal_cdf(-4.5)
        for seed in (1, 2):
            with self.subTest(seed=seed):
                result = ak_ss(linear_state(4.5), samples_per_level=1000, seed=seed)
                self.assertTrue(result.converged, result.message)
                self.assertTrue(
                    within_error(result, reference, 3.0),
                    (result.probability, result.coefficient_of_variation),
                )
                self.assertGreaterEqual(result.levels, 5)
                # Crude MC needs about 100 / P_f = 3e7 calls for a 10% c.o.v.
                self.assertLessEqual(result.evaluations, 40)
                self.assertLess(result.coefficient_of_variation, 0.5)


class BlindSpotTests(unittest.TestCase):
    """Initial designs that miss failure regions, and the remedies."""

    def test_phi_inverse_design_misses_branches(self) -> None:
        """A small Phi^-1 design converges to a wrong value; box or seeds fix it."""
        options = {
            "samples_per_level": 1000,
            "initial_size": 8,
            "check_population": 10_000,
            "seed": 5,
        }
        blind = ak_ss(four_branch_state(), design="lhs", design_bound=None, **options)
        # The surrogate is confident and wrong: it reports convergence while
        # the failure regions it never saw are predicted safe.
        self.assertTrue(blind.converged, blind.message)
        self.assertLess(blind.probability, 0.2 * FOUR_BRANCH_REFERENCE_PROBABILITY)
        self.assertGreaterEqual(blind.check_misclassified, 10)
        self.assertGreater(blind.check_probability, 1.0e-3)

        box = ak_ss(four_branch_state(), **options)
        seeded = ak_ss(
            four_branch_state(),
            design="lhs",
            design_bound=None,
            extra_points=FOUR_BRANCH_DESIGN_POINTS,
            **options,
        )
        self.assertEqual(seeded.initial_size, 12)
        for fixed in (box, seeded):
            self.assertTrue(fixed.converged, fixed.message)
            self.assertTrue(
                within_error(fixed, FOUR_BRANCH_REFERENCE_PROBABILITY, 3.0),
                (fixed.probability, fixed.coefficient_of_variation),
            )
            self.assertEqual(fixed.check_misclassified, 0)


class SurrogateFactoryTests(unittest.TestCase):
    """The refit-hook path of ADR-0107."""

    def test_gaussian_process_factory(self) -> None:
        """A GP factory gives a converged, accurate run on a small case."""
        reference = standard_normal_cdf(-4.5)
        factory = GaussianProcessFactory(regularization=AKSS_GP_REGULARIZATION, seed=1)
        result = ak_ss(
            linear_state(4.5), samples_per_level=500, surrogate=factory, seed=1
        )
        self.assertTrue(result.converged, result.message)
        self.assertTrue(within_error(result, reference, 3.0))
        self.assertIsInstance(result.surrogate, GaussianProcessRegression)
        self.assertEqual(result.refits, 1 + result.added)
        self.assertEqual(result.evaluations, result.initial_size + result.added)


class OptionTests(unittest.TestCase):
    """Budgets, kernels, and learning functions."""

    def test_budget_exhausted(self) -> None:
        """``max_added`` stops learning and the run is not converged."""
        result = ak_ss(four_branch_state(), samples_per_level=500, max_added=5, seed=1)
        self.assertFalse(result.converged)
        self.assertIn("max_added", result.message)
        self.assertEqual(result.added, 5)
        self.assertEqual(result.evaluations, result.initial_size + 5)
        nothing = ak_ss(four_branch_state(), samples_per_level=500, max_added=0, seed=1)
        self.assertFalse(nothing.converged)
        self.assertEqual(nothing.evaluations, nothing.initial_size)
        self.assertEqual(nothing.rounds, 1)

    def test_other_samplers_and_learning(self) -> None:
        """Every subset kernel and the EFF learning function run."""
        reference = standard_normal_cdf(-3.5)
        for sampler, learning in (("mma", "U"), ("cs", "U"), ("hmc", "EFF")):
            with self.subTest(sampler=sampler, learning=learning):
                result = ak_ss(
                    linear_state(3.5),
                    samples_per_level=500,
                    sampler=sampler,
                    learning=learning,
                    seed=2,
                )
                self.assertEqual(result.sampler, sampler)
                self.assertEqual(result.learning, learning)
                self.assertTrue(result.converged, result.message)
                self.assertTrue(within_error(result, reference, 3.0))

    def test_validation(self) -> None:
        """Invalid options raise."""
        state = linear_state(3.0)
        with self.assertRaises(TypeError):
            ak_ss(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ak_ss(state, surrogate=3)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ak_ss(state, check_levels=1)  # type: ignore[arg-type]
        bad: tuple[dict[str, object], ...] = (
            {"sampler": "gibbs"},
            {"samples_per_level": 5},
            {"samples_per_level": 1005},
            {"samples_per_level": 1000, "level_probability": 0.3, "sampler": "mma"},
            {"level_probability": 0.6},
            {"level_probability": 0.0},
            {"max_levels": 0},
            {"learning": "V"},
            {"learning_threshold": 0.0},
            {"bounds_tolerance": 0.0},
            {"initial_size": 1},
            {"design": "grid"},
            {"design_bound": -1.0},
            {"kernel": "linear"},
            {"max_added": -1},
            {"max_rounds": 0},
            {"refit_growth": 1.0},
            {"restarts": 0},
            {"nugget": 0.0},
            {"proposal_spread": 0.0},
            {"correlation": 1.0},
            {"initial_scale": 0.0},
            {"target_acceptance": 1.0},
            {"trajectory_time": 2.0},
            {"check_population": -1},
            {"extra_points": [(1.0, 2.0, 3.0)]},
        )
        for options in bad:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ak_ss(state, **options)  # type: ignore[arg-type]


class PopulationReuseTests(unittest.TestCase):
    """``replace_population(reuse=True)`` keeps the states of repeated points."""

    def test_reuse_matches_fresh_solve(self) -> None:
        """Kept states equal the old ones; all agree with a fresh posterior."""
        generator = random.Random(3)
        train = [(generator.uniform(-3, 3), generator.uniform(-3, 3)) for _ in range(8)]
        values = [3.0 - (u + v) / math.sqrt(2.0) for u, v in train]
        first = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(20)]
        new = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(10)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = fit_gaussian_process(
                train, values, seed=1, regularization=GaussianProcessRegularization()
            )
        posterior = GaussianProcessPopulationPosterior(gp, first)
        extra = (0.5, -0.5)
        posterior.add_training(extra, 3.0)
        means = posterior.means
        kept = dict(zip(first, posterior.means, strict=True))
        second = [*first[5:15], *new, first[5]]
        posterior.replace_population(second, reuse=True)
        self.assertIs(posterior.means, means)
        self.assertEqual(posterior.population, second)
        for point, mean in zip(second[:10], posterior.means[:10], strict=True):
            self.assertEqual(mean, kept[point])
        fresh = GaussianProcessPopulationPosterior(gp, [])
        fresh.add_training(extra, 3.0)
        fresh.add_population(second)
        for left, right in zip(posterior.means, fresh.means, strict=True):
            self.assertAlmostEqual(left, right, delta=1e-9 * max(1.0, abs(right)))
        for left, right in zip(posterior.variances, fresh.variances, strict=True):
            self.assertAlmostEqual(left, right, delta=1e-9)
        # The repeated last point got its own state (no shared list).
        posterior.add_training((1.0, 1.0), 3.0 - math.sqrt(2.0))
        self.assertEqual(len(posterior.v[0]), len(posterior.v[-1]))
        self.assertEqual(len(posterior.v[0]), 10)


if __name__ == "__main__":
    unittest.main()
