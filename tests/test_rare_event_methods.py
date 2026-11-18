"""Tests for moving particles, directional simulation, and splitting theory.

Seeded Markov-chain results are never pinned to digits: every check compares
with an exact answer at a tolerance set by the estimator's own error scale,
or checks a structural property of the run.

References:
    [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015],
    [REF-BJERAGER-1988], [REF-LAGNOUX-2006].

"""

from __future__ import annotations

import math
import statistics
import unittest
from collections.abc import Callable, Mapping

from probstat.distributions import Normal
from probstat.distributions._special import regularized_gamma_q
from probstat.modeling import (
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch_model,
)
from probstat.reliability import (
    OPTIMAL_LEVEL_EXPONENT,
    OPTIMAL_LEVEL_PROBABILITY,
    OPTIMAL_LEVELS_PER_LOG,
    ModelLimitState,
    default_radius_max,
    directional_simulation,
    moving_particles,
    moving_particles_coefficient_of_variation,
    moving_particles_interval,
    optimal_splitting,
    splitting_coefficient_of_variation,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)
PHI = Normal(0.0, 1.0)


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on standard-normal inputs as a ``g <= 0`` limit state.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    names = tuple(f"u{i + 1}" for i in range(dimension))
    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from the named inputs.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        return {"y": function(tuple(p[name] for name in names))}

    return ModelLimitState(ProbabilisticModel(inputs, response, Y), "y", 0.0, "below")


def linear_state(beta: float, dimension: int = 2) -> ModelLimitState:
    """Return ``G = beta - sum(u) / sqrt(d)`` with ``P_f = Phi(-beta)``.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    scale = 1.0 / math.sqrt(dimension)
    return make_state(lambda u: beta - scale * math.fsum(u), dimension)


def radius(u: tuple[float, ...]) -> float:
    """Return the Euclidean norm of ``u``.

    References:
        [REF-BJERAGER-1988].

    """
    return math.sqrt(math.fsum(x * x for x in u))


def chi_tail(dimension: int, r: float) -> float:
    """Return ``P(R > r)`` for ``R^2 ~ chi-square_d``.

    References:
        [EQ-REL-DIR-001].

    """
    return regularized_gamma_q(0.5 * dimension, 0.5 * r * r)


def poisson_cdf(count: int, mean: float) -> float:
    """Return ``P(X <= count)`` for ``X ~ Poisson(mean)`` by direct summation.

    References:
        [REF-GARWOOD-1936].

    """
    term = math.exp(-mean)
    total = term
    for k in range(1, count + 1):
        term *= mean / k
        total += term
    return total


class SplittingTheoryTests(unittest.TestCase):
    """Ideal splitting c.o.v., its optimum, and the comparison values."""

    def test_one_level_is_crude_monte_carlo(self) -> None:
        """``M = 1`` reproduces ``sqrt((1 - p) / (p N))``."""
        for p, n in ((1e-3, 1000), (0.3, 17), (1e-7, 10**6)):
            self.assertAlmostEqual(
                splitting_coefficient_of_variation(p, 1, n),
                math.sqrt((1.0 - p) / (p * n)),
                delta=1e-12 * math.sqrt((1.0 - p) / (p * n)),
            )

    def test_formula_matches_product_of_independent_levels(self) -> None:
        """The c.o.v. equals ``prod(1 + delta_m^2) - 1`` built level by level."""
        p, levels, n = 1e-6, 6, 6000
        pm = p ** (1.0 / levels)
        per_level = (1.0 - pm) / (pm * (n / levels))
        expected = math.sqrt((1.0 + per_level) ** levels - 1.0)
        self.assertAlmostEqual(
            splitting_coefficient_of_variation(p, levels, n), expected, places=12
        )

    def test_optimal_exponent_solves_the_stationarity_equation(self) -> None:
        """``x*`` solves ``x e^x = 2 (e^x - 1)`` and gives 0.2032 and 0.6275."""
        x = OPTIMAL_LEVEL_EXPONENT
        self.assertAlmostEqual(x * math.exp(x), 2.0 * math.expm1(x), places=12)
        self.assertAlmostEqual(OPTIMAL_LEVEL_PROBABILITY, 0.2032, places=4)
        self.assertAlmostEqual(OPTIMAL_LEVELS_PER_LOG, 0.6275, places=4)
        # Brute-force minimum of (e^x - 1) / x^2 on a fine grid.
        grid = [1.0 + k * 1e-5 for k in range(100001)]
        best = min(grid, key=lambda t: math.expm1(t) / (t * t))
        self.assertAlmostEqual(best, x, delta=2e-5)

    def test_integer_optimum_matches_brute_force(self) -> None:
        """The reported ``M`` minimizes the exact c.o.v. over every ``M <= N``."""
        for p, n in ((2.2245e-3, 10000), (1e-6, 500), (1e-9, 10**5), (0.05, 40)):
            result = optimal_splitting(p, n)
            scan_limit = min(n, 400)
            covs = [
                splitting_coefficient_of_variation(p, m, n)
                for m in range(1, scan_limit + 1)
            ]
            best = min(range(len(covs)), key=covs.__getitem__) + 1
            self.assertEqual(result.levels, best)
            self.assertAlmostEqual(
                result.coefficient_of_variation, covs[best - 1], places=12
            )
            self.assertAlmostEqual(result.level_probability, p ** (1.0 / best))

    def test_large_budget_approaches_the_continuous_optimum(self) -> None:
        """For large ``N`` the optimum is ``-0.6275 ln p`` at c.o.v. ~ asymptote."""
        p = 1e-12
        result = optimal_splitting(p, 10**9)
        self.assertAlmostEqual(result.continuous_levels, -0.6275 * math.log(p), 2)
        self.assertLessEqual(abs(result.levels - result.continuous_levels), 1.0)
        self.assertAlmostEqual(
            result.coefficient_of_variation
            / result.asymptotic_coefficient_of_variation,
            1.0,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            result.monte_carlo_coefficient_of_variation,
            math.sqrt((1.0 - p) / (p * 1e9)),
        )
        self.assertLess(
            result.coefficient_of_variation,
            result.monte_carlo_coefficient_of_variation,
        )

    def test_moving_particles_coefficient_of_variation(self) -> None:
        """``sqrt(p^(-1/N) - 1)`` tends to ``sqrt(-ln(p) / N)``."""
        p = 1e-5
        for n in (100, 10000):
            exact = moving_particles_coefficient_of_variation(p, n)
            self.assertAlmostEqual(exact, math.sqrt(p ** (-1.0 / n) - 1.0))
            self.assertAlmostEqual(
                exact / math.sqrt(-math.log(p) / n), 1.0, delta=0.06 * 100 / n
            )
        budget = 10**5
        particles = budget / (1.0 - math.log(p))
        self.assertAlmostEqual(
            optimal_splitting(p, budget).moving_particles_coefficient_of_variation,
            moving_particles_coefficient_of_variation(p, particles),
        )

    def test_invalid_arguments(self) -> None:
        """Domain errors are reported."""
        with self.assertRaises(ValueError):
            splitting_coefficient_of_variation(0.0, 2, 10)
        with self.assertRaises(ValueError):
            splitting_coefficient_of_variation(0.1, 0, 10)
        with self.assertRaises(ValueError):
            splitting_coefficient_of_variation(0.1, 5, 4)
        with self.assertRaises(ValueError):
            optimal_splitting(1.0, 10)
        with self.assertRaises(ValueError):
            optimal_splitting(0.1, 0)
        with self.assertRaises(ValueError):
            moving_particles_coefficient_of_variation(0.1, 1.0)


class PoissonIntervalTests(unittest.TestCase):
    """The exact interval of the number of moves maps to ``p`` correctly."""

    def test_endpoints_are_garwood_tail_inversions(self) -> None:
        """``P(X <= M | lambda_hi) = P(X >= M | lambda_lo) = alpha / 2``."""
        particles, alpha = 50, 0.05
        for moves in (1, 7, 120, 600):
            lower, upper = moving_particles_interval(moves, particles, 1.0 - alpha)
            lam_hi = -particles * math.log(lower)
            lam_lo = -particles * math.log(upper)
            self.assertAlmostEqual(poisson_cdf(moves, lam_hi), alpha / 2, places=7)
            self.assertAlmostEqual(
                1.0 - poisson_cdf(moves - 1, lam_lo), alpha / 2, places=7
            )
            estimate = (1.0 - 1.0 / particles) ** moves
            self.assertLess(lower, estimate)
            self.assertLess(estimate, upper)

    def test_zero_moves_has_unit_upper_limit(self) -> None:
        """With ``M = 0`` the upper limit is one and the lower one is positive."""
        lower, upper = moving_particles_interval(0, 10, 0.9)
        self.assertEqual(upper, 1.0)
        self.assertAlmostEqual(lower, math.exp(math.log(0.05) / 10))

    def test_invalid_arguments(self) -> None:
        """Domain errors are reported."""
        with self.assertRaises(ValueError):
            moving_particles_interval(-1, 10)
        with self.assertRaises(ValueError):
            moving_particles_interval(3, 1)
        with self.assertRaises(ValueError):
            moving_particles_interval(3, 10, 1.0)


class MovingParticlesTests(unittest.TestCase):
    """The last-particle algorithm on limit states with known answers."""

    def test_run_structure(self) -> None:
        """Levels decrease, particles fail, and the cost is ``N + M steps``."""
        state = linear_state(2.5)
        result = moving_particles(state, 30, steps=5, seed=11)
        self.assertTrue(result.converged)
        self.assertEqual(result.moves, len(result.levels))
        self.assertTrue(all(level > 0.0 for level in result.levels))
        self.assertTrue(
            all(a >= b for a, b in zip(result.levels, result.levels[1:], strict=False))
        )
        self.assertTrue(all(value <= 0.0 for value in result.final_values))
        self.assertEqual(len(result.final_particles), 30)
        self.assertEqual(result.evaluations, 30 + 5 * result.moves)
        self.assertAlmostEqual(result.probability, (1.0 - 1.0 / 30) ** result.moves)
        lower, upper = result.confidence_interval
        self.assertLess(lower, result.probability)
        self.assertLess(result.probability, upper)
        self.assertTrue(0.0 < result.acceptance_rate < 1.0)
        self.assertAlmostEqual(result.beta, -PHI.quantile(result.probability))
        self.assertAlmostEqual(
            result.variance,
            result.probability**2 * (result.probability ** (-1.0 / 30) - 1.0),
        )
        repeat = moving_particles(state, 30, steps=5, seed=11)
        self.assertEqual(repeat.moves, result.moves)
        self.assertEqual(repeat.final_particles, result.final_particles)

    def test_linear_limit_state_mean_of_replicates(self) -> None:
        """Replicate means match ``Phi(-3)`` within a few standard errors."""
        exact = PHI.cdf(-3.0)
        particles, replicates = 50, 8
        estimates = [
            moving_particles(linear_state(3.0), particles, steps=10, seed=k).probability
            for k in range(replicates)
        ]
        ideal = exact * moving_particles_coefficient_of_variation(exact, particles)
        spread = max(ideal, statistics.stdev(estimates)) / math.sqrt(replicates)
        self.assertLess(abs(statistics.fmean(estimates) - exact), 4.0 * spread)

    def test_poisson_law_of_moves_and_interval_coverage(self) -> None:
        """``M`` has mean ``-N ln p`` and the 95 % interval covers ``p``."""
        exact = PHI.cdf(-2.0)
        particles, replicates = 10, 120
        lam = -particles * math.log(exact)
        state = linear_state(2.0, dimension=1)
        moves: list[int] = []
        covered = 0
        for k in range(replicates):
            result = moving_particles(
                state, particles, steps=10, correlation=0.5, seed=1000 + k
            )
            moves.append(result.moves)
            lower, upper = result.confidence_interval
            covered += lower <= exact <= upper
        self.assertLess(
            abs(statistics.fmean(moves) - lam), 4.0 * math.sqrt(lam / replicates)
        )
        self.assertGreaterEqual(covered / replicates, 0.85)

    def test_four_branch_reference(self) -> None:
        """Replicates on the four-branch function reach ``2.2245e-3``."""
        state = ModelLimitState(four_branch_model(), "g", 0.0, "below")
        particles, replicates = 100, 3
        results = [
            moving_particles(state, particles, steps=15, seed=20 + k)
            for k in range(replicates)
        ]
        exact = FOUR_BRANCH_REFERENCE_PROBABILITY
        ideal = exact * moving_particles_coefficient_of_variation(exact, particles)
        mean = statistics.fmean(result.probability for result in results)
        # Allow a 1.5 inflation of the ideal spread for chain correlation.
        self.assertLess(abs(mean - exact), 4.0 * 1.5 * ideal / math.sqrt(replicates))
        self.assertTrue(all(result.converged for result in results))

    def test_max_moves_stops_unconverged(self) -> None:
        """A move limit below the needed count returns ``converged=False``."""
        result = moving_particles(linear_state(4.0), 20, steps=2, seed=3, max_moves=5)
        self.assertFalse(result.converged)
        self.assertEqual(result.moves, 5)

    def test_failing_population_needs_no_move(self) -> None:
        """An always-failing limit state gives ``p = 1`` with zero moves."""
        result = moving_particles(make_state(lambda u: -1.0, 2), 10, seed=1)
        self.assertEqual(result.moves, 0)
        self.assertEqual(result.probability, 1.0)
        self.assertEqual(result.confidence_interval[1], 1.0)
        self.assertTrue(math.isnan(result.acceptance_rate))

    def test_invalid_arguments(self) -> None:
        """Domain and type errors are reported."""
        state = linear_state(2.0)
        with self.assertRaises(TypeError):
            moving_particles(object(), 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            moving_particles(state, 1)
        with self.assertRaises(ValueError):
            moving_particles(state, 10, steps=0)
        with self.assertRaises(ValueError):
            moving_particles(state, 10, correlation=1.0)
        with self.assertRaises(ValueError):
            moving_particles(state, 10, confidence=0.0)
        with self.assertRaises(ValueError):
            moving_particles(state, 10, max_moves=0)


class DirectionalSimulationTests(unittest.TestCase):
    """Directional simulation on spheres, shells, planes, and four branches."""

    def test_sphere_is_exact_with_zero_variance(self) -> None:
        """``G = r0 - |u|`` gives ``Q(d/2, r0^2/2)`` on every ray."""
        r0, dimension = 3.2, 3
        state = make_state(lambda u: r0 - radius(u), dimension)
        result = directional_simulation(state, 10, seed=4)
        exact = chi_tail(dimension, r0)
        self.assertAlmostEqual(result.probability, exact, delta=1e-9 * exact)
        self.assertLess(result.standard_error, 1e-12)
        self.assertFalse(result.origin_fails)
        self.assertTrue(all(len(roots) == 1 for roots in result.roots))
        for roots in result.roots:
            self.assertAlmostEqual(roots[0], r0, places=8)

    def test_shell_has_two_roots_per_ray(self) -> None:
        """Failure on ``r1 <= |u| <= r2`` adds one finite segment per ray."""
        r1, r2 = 2.0, 3.0
        state = make_state(lambda u: (radius(u) - r1) * (radius(u) - r2), 2)
        result = directional_simulation(state, 6, grid=40, seed=5, antithetic=True)
        exact = chi_tail(2, r1) - chi_tail(2, r2)
        self.assertAlmostEqual(result.probability, exact, delta=1e-9)
        self.assertEqual(result.rays, 12)
        self.assertTrue(all(len(roots) == 2 for roots in result.roots))

    def test_failing_origin_counts_the_inner_segment(self) -> None:
        """Failure inside ``|u| <= r1`` integrates the chi law from zero."""
        r1 = 1.5
        state = make_state(lambda u: radius(u) - r1, 4)
        result = directional_simulation(state, 5, seed=6)
        self.assertTrue(result.origin_fails)
        self.assertAlmostEqual(result.probability, 1.0 - chi_tail(4, r1), delta=1e-9)

    def test_linear_limit_state_within_standard_errors(self) -> None:
        """The plane ``beta = 3`` agrees with ``Phi(-3)`` (plain and antithetic)."""
        exact = PHI.cdf(-3.0)
        state = linear_state(3.0)
        for antithetic in (False, True):
            result = directional_simulation(state, 300, antithetic=antithetic, seed=7)
            self.assertLess(
                abs(result.probability - exact), 4.0 * result.standard_error
            )
            lower, upper = result.confidence_interval
            self.assertLess(lower, result.probability)
            self.assertLess(result.probability, upper)

    def test_four_branch_reference(self) -> None:
        """A few hundred directions reach ``2.22e-3`` within a few SE."""
        state = ModelLimitState(four_branch_model(), "g", 0.0, "below")
        result = directional_simulation(state, 200, seed=8)
        exact = FOUR_BRANCH_REFERENCE_PROBABILITY
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)
        self.assertLess(result.coefficient_of_variation, 0.25)
        self.assertGreaterEqual(result.evaluations, 1 + 200 * result.grid)
        self.assertEqual(len(result.ray_probabilities), 200)

    def test_default_radius_leaves_negligible_mass(self) -> None:
        """For ``d = 2`` the default radius is ``sqrt(2 ln 1e15)``."""
        self.assertAlmostEqual(
            default_radius_max(2), math.sqrt(2.0 * math.log(1e15)), places=6
        )
        self.assertLess(chi_tail(10, default_radius_max(10)), 1.01e-15)

    def test_invalid_arguments(self) -> None:
        """Domain and type errors are reported."""
        state = linear_state(2.0)
        with self.assertRaises(TypeError):
            directional_simulation(object(), 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            directional_simulation(state, 1)
        with self.assertRaises(ValueError):
            directional_simulation(state, 10, grid=0)
        with self.assertRaises(ValueError):
            directional_simulation(state, 10, radius_max=-1.0)
        with self.assertRaises(ValueError):
            directional_simulation(state, 10, confidence=1.5)
        with self.assertRaises(ValueError):
            default_radius_max(0)


if __name__ == "__main__":
    unittest.main()
