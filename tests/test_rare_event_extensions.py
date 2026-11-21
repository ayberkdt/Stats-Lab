"""Tests for improved cross entropy, sequential importance sampling, and BUS."""

from __future__ import annotations

import math
import random
import statistics
import unittest
import warnings
from collections.abc import Callable, Sequence

from probstat.distributions import Normal
from probstat.modeling import (
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch,
    standard_normal_cdf,
)
from probstat.reliability import (
    BUS_MARGIN,
    ICE_FAMILIES,
    BUSResult,
    ImprovedCrossEntropyResult,
    ModelLimitState,
    SequentialImportanceResult,
    bivariate_normal_cdf,
    bus_limit_state,
    bus_subset_simulation,
    improved_cross_entropy,
    sequential_importance_sampling,
    smoothing_scale,
    subset_simulation,
)
from probstat.reliability.improved_cross_entropy import (
    _fit_mixture,
    _log_cv,
    _log_smooth_indicator,
    _weighted_moments,
)
from probstat.reliability.sequential_importance import _smoothed_chain
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)


class DirectLimitState(ModelLimitState):
    """Limit state evaluated directly in standard-normal space.

    The inputs are independent standard normals, so the isoprobabilistic map
    is the identity; :meth:`standard` skips it and still counts every call
    on the model's evaluation counter.
    """

    __slots__ = ()

    def standard(self, u: Sequence[float]) -> float:
        """Count one evaluation and return the limit-state value at ``u``."""
        self.model.counter.add(1)
        return self.margin(FUNCTIONS[self.name](u))


FUNCTIONS: dict[str, Callable[[Sequence[float]], float]] = {}


def direct(
    name: str, dimension: int, function: Callable[[Sequence[float]], float]
) -> ModelLimitState:
    """Return ``function`` as a limit state on ``dimension`` standard normals."""
    inputs = JointInput(
        tuple(RandomInput(f"u{i}", Normal(0.0, 1.0)) for i in range(dimension))
    )
    model = ProbabilisticModel(inputs, lambda p: {"y": function(tuple(p.values()))}, Y)
    FUNCTIONS[name] = function
    return DirectLimitState(model, "y", 0.0, "below", name)


def linear(beta: float, dimension: int) -> ModelLimitState:
    """Return ``g = beta - sum(u)/sqrt(d)`` with ``P_f = Phi(-beta)``."""
    return direct(
        f"linear-{beta}-{dimension}",
        dimension,
        lambda u: beta - math.fsum(u) / math.sqrt(dimension),
    )


def parallel_pair() -> ModelLimitState:
    """Return the parallel system of ``2.5 - u1`` and ``2 - 0.6 u1 - 0.8 u2``.

    Both components must fail, so ``g = max(g1, g2)`` and
    ``P_f = Phi_2(-2.5, -2; 0.6)``.
    """
    return direct(
        "parallel", 2, lambda u: max(2.5 - u[0], 2.0 - 0.6 * u[0] - 0.8 * u[1])
    )


PARALLEL_EXACT = bivariate_normal_cdf(-2.5, -2.0, 0.6)


def four_branch_state() -> ModelLimitState:
    """Return the four-branch series limit state (``k = 7``)."""
    return direct("four-branch", 2, lambda u: four_branch(u[0], u[1]))


def z_score(result: ImprovedCrossEntropyResult, exact: float) -> float:
    """Return ``(estimate - exact) / standard error``."""
    return (result.probability - exact) / result.standard_error


class SmoothingScaleTests(unittest.TestCase):
    """The one-dimensional search of the smoothing scale."""

    def test_attains_the_target_coefficient_of_variation(self) -> None:
        """The weights at the returned scale have the requested c.o.v."""
        generator = random.Random(3)
        values = [3.0 - generator.gauss(0.0, 1.0) for _ in range(500)]
        for target in (0.5, 1.5, 3.0):
            sigma, attained = smoothing_scale(values, target_cv=target)
            self.assertAlmostEqual(attained, target, places=6)
            recomputed = _log_cv([_log_smooth_indicator(g, sigma) for g in values])
            self.assertAlmostEqual(recomputed, target, places=6)

    def test_scale_decreases_below_the_previous_one(self) -> None:
        """With base weights of the previous level the new scale is smaller."""
        generator = random.Random(4)
        values = [3.0 - generator.gauss(0.0, 1.0) for _ in range(500)]
        first, _ = smoothing_scale(values, target_cv=1.0)
        base = [-_log_smooth_indicator(g, first) for g in values]
        second, attained = smoothing_scale(values, base, previous=first, target_cv=1.0)
        self.assertLess(second, first)
        self.assertAlmostEqual(attained, 1.0, places=6)

    def test_smooth_indicator_limits(self) -> None:
        """``Phi(-g/sigma)`` is 1/2 at ``sigma = inf`` and the indicator at 0."""
        self.assertAlmostEqual(_log_smooth_indicator(2.0, math.inf), math.log(0.5))
        self.assertEqual(_log_smooth_indicator(-1.0, 0.0), 0.0)
        self.assertEqual(_log_smooth_indicator(1.0, 0.0), -math.inf)
        self.assertAlmostEqual(
            _log_smooth_indicator(1.2, 0.8), math.log(standard_normal_cdf(-1.5))
        )
        # Far tail: the logarithm stays finite where erfc underflows.
        tail = _log_smooth_indicator(50.0, 1.0)
        self.assertTrue(math.isfinite(tail))
        self.assertAlmostEqual(tail / (-0.5 * 50.0**2), 1.0, delta=0.01)

    def test_boundary_cases(self) -> None:
        """Without a crossing the search returns the documented end point."""
        # Every point failed: the indicator itself has zero variation.
        sigma, attained = smoothing_scale([-1.0, -2.0, -3.0], target_cv=1.5)
        self.assertLess(attained, 1.5)
        self.assertLess(sigma, 1e-8)
        # Base weights already above the target: the previous scale is kept.
        sigma, attained = smoothing_scale(
            [1.0, 1.0, 1.0, 1.0], [0.0, -50.0, -50.0, -50.0], previous=2.0
        )
        self.assertEqual(sigma, 2.0)
        self.assertGreater(attained, 1.5)

    def test_validation(self) -> None:
        """Invalid arguments raise ``ValueError``."""
        with self.assertRaises(ValueError):
            smoothing_scale([1.0])
        with self.assertRaises(ValueError):
            smoothing_scale([1.0, math.nan])
        with self.assertRaises(ValueError):
            smoothing_scale([1.0, 2.0], [0.0])
        with self.assertRaises(ValueError):
            smoothing_scale([1.0, 2.0], [-math.inf, -math.inf])
        with self.assertRaises(ValueError):
            smoothing_scale([1.0, 2.0], target_cv=0.0)
        with self.assertRaises(ValueError):
            smoothing_scale([1.0, 2.0], previous=-1.0)


class WeightedFitTests(unittest.TestCase):
    """Weighted maximum likelihood of the proposal families."""

    def test_integer_weights_equal_repeated_points(self) -> None:
        """Weighted moments with weight 3 equal the moments of three copies."""
        generator = random.Random(5)
        points = [
            (generator.gauss(0.0, 1.0), generator.gauss(1.0, 2.0)) for _ in range(30)
        ]
        weights = [float(1 + index % 3) for index in range(30)]
        repeated = [
            point
            for point, weight in zip(points, weights, strict=True)
            for _ in range(int(weight))
        ]
        _, mean, covariance = _weighted_moments(points, weights, None)
        mu = [statistics.fmean(p[i] for p in repeated) for i in range(2)]
        for i in range(2):
            self.assertAlmostEqual(mean[i], mu[i], places=12)
            for j in range(2):
                expected = statistics.fmean(
                    (p[i] - mu[i]) * (p[j] - mu[j]) for p in repeated
                )
                self.assertAlmostEqual(covariance[i][j], expected, places=8)

    def test_covariance_floor_bounds_the_eigenvalues(self) -> None:
        """A floored covariance has no variance below the floor."""
        points = [(0.01 * index, float(index % 5)) for index in range(40)]
        _, _, covariance = _weighted_moments(points, [1.0] * 40, 0.55)
        self.assertGreaterEqual(covariance[0][0], 0.55 - 1e-9)

    def test_weighted_em_recovers_weighted_clusters(self) -> None:
        """EM finds two separated clusters and their weighted proportions."""
        generator = random.Random(6)
        points = [
            (generator.gauss(-4.0, 0.5), generator.gauss(0.0, 0.5)) for _ in range(200)
        ] + [(generator.gauss(4.0, 0.5), generator.gauss(1.0, 0.5)) for _ in range(200)]
        weights = [3.0] * 200 + [1.0] * 200
        mixture = _fit_mixture(points, weights, 2, None, random.Random(1), 100)
        order = sorted(range(2), key=lambda j: mixture.means[j][0])
        left, right = order
        self.assertAlmostEqual(mixture.weights[left], 0.75, places=6)
        self.assertAlmostEqual(mixture.means[left][0], -4.0, delta=0.15)
        self.assertAlmostEqual(mixture.means[right][0], 4.0, delta=0.15)
        self.assertAlmostEqual(mixture.means[right][1], 1.0, delta=0.15)
        self.assertAlmostEqual(mixture.covariances[left][0][0], 0.25, delta=0.08)


class ImprovedCrossEntropyTests(unittest.TestCase):
    """Improved cross-entropy importance sampling against exact values."""

    def test_linear_limit_states(self) -> None:
        """Linear states: within four standard errors of ``Phi(-beta)``."""
        for beta, dimension, seeds in (
            (3.0, 2, (1, 2, 3)),
            (4.5, 2, (1, 2, 3)),
            (3.0, 20, (1, 2)),
            (4.5, 20, (1,)),
        ):
            exact = standard_normal_cdf(-beta)
            for seed in seeds:
                with self.subTest(beta=beta, dimension=dimension, seed=seed):
                    result = improved_cross_entropy(linear(beta, dimension), seed=seed)
                    self.assertTrue(result.converged)
                    self.assertLess(abs(z_score(result, exact)), 4.0)
                    self.assertLess(result.coefficient_of_variation, 0.2)
                    self.assertEqual(
                        result.evaluations,
                        result.samples_per_level * (result.levels + 1),
                    )
                    self.assertFalse(result.weights_heavy_tailed)

    def test_reported_cv_matches_replicates(self) -> None:
        """The reported c.o.v. agrees with the scatter of 16 seeded runs.

        Measured over 20 seeds on the six ADR cases the ratio of replicate
        to mean reported c.o.v. lay between 0.73 and 1.27; the test allows
        a factor of two.
        """
        exact = standard_normal_cdf(-3.0)
        results = [
            improved_cross_entropy(linear(3.0, 2), seed=seed) for seed in range(16)
        ]
        estimates = [item.probability for item in results]
        replicate = statistics.stdev(estimates) / exact
        reported = statistics.fmean(item.coefficient_of_variation for item in results)
        self.assertLess(replicate / reported, 2.0)
        self.assertGreater(replicate / reported, 0.5)
        standard_error = statistics.stdev(estimates) / math.sqrt(len(estimates))
        self.assertLess(abs(statistics.fmean(estimates) - exact), 4 * standard_error)

    def test_parallel_system(self) -> None:
        """Parallel system of two linear states against ``Phi_2``."""
        for seed in (1, 2, 3):
            result = improved_cross_entropy(parallel_pair(), seed=seed)
            self.assertTrue(result.converged)
            self.assertLess(abs(z_score(result, PARALLEL_EXACT)), 4.0)

    def test_four_branch_needs_a_mixture(self) -> None:
        """Four-branch system: the mixture converges, one Gaussian does not."""
        exact = FOUR_BRANCH_REFERENCE_PROBABILITY
        mixture_cvs = []
        mixture_cost = []
        for seed in (1, 2, 3, 4, 5):
            result = improved_cross_entropy(
                four_branch_state(), family="mixture", components=4, seed=seed
            )
            self.assertTrue(result.converged)
            self.assertLess(abs(z_score(result, exact)), 4.0)
            self.assertGreaterEqual(len(result.mixture_weights), 3)
            self.assertAlmostEqual(math.fsum(result.mixture_weights), 1.0)
            mixture_cvs.append(result.coefficient_of_variation)
            mixture_cost.append(result.evaluations)
        self.assertLessEqual(max(mixture_cost), 5000)
        with self.assertWarns(UserWarning):
            single = improved_cross_entropy(four_branch_state(), max_levels=8, seed=1)
        self.assertFalse(single.converged)
        self.assertEqual(single.levels, 8)
        # Nine samples of the single Gaussian still give a larger c.o.v. than
        # the four samples of the mixture.
        self.assertEqual(single.evaluations, 9000)
        self.assertGreater(single.coefficient_of_variation, 1.5 * max(mixture_cvs))

    def test_unguarded_covariance_is_flagged(self) -> None:
        """Without the floor the collapsed covariance is reported and warned."""
        with self.assertWarns(UserWarning):
            result = improved_cross_entropy(
                linear(4.5, 2), covariance_floor=None, seed=1
            )
        self.assertTrue(result.weights_heavy_tailed)
        self.assertLessEqual(result.min_covariance_eigenvalue, 0.5)
        guarded = improved_cross_entropy(linear(4.5, 2), seed=1)
        self.assertGreaterEqual(guarded.min_covariance_eigenvalue, 0.55 - 1e-9)

    def test_history_and_reproducibility(self) -> None:
        """Scales decrease, the history has consistent lengths, seeds repeat."""
        first = improved_cross_entropy(linear(3.0, 2), samples_per_level=400, seed=9)
        again = improved_cross_entropy(linear(3.0, 2), samples_per_level=400, seed=9)
        self.assertEqual(first, again)
        self.assertEqual(len(first.scales), first.levels)
        self.assertEqual(
            len(first.stopping_coefficients_of_variation), first.levels + 1
        )
        self.assertEqual(len(first.level_failures), first.levels + 1)
        self.assertTrue(
            all(a > b for a, b in zip(first.scales, first.scales[1:], strict=False))
        )
        self.assertLessEqual(first.stopping_coefficients_of_variation[-1], 1.5)
        self.assertEqual(first.failures, first.level_failures[-1])
        self.assertGreater(first.failure_effective_sample_size, 0.0)
        self.assertIsInstance(first, ImprovedCrossEntropyResult)

    def test_frequent_event_needs_no_level(self) -> None:
        """A frequent event stops on the nominal sample (crude Monte Carlo)."""
        result = improved_cross_entropy(linear(0.2, 2), seed=1)
        self.assertEqual(result.levels, 0)
        self.assertEqual(result.evaluations, 1000)
        self.assertAlmostEqual(result.probability, result.failures / 1000)

    def test_validation(self) -> None:
        """Invalid options raise."""
        state = linear(3.0, 2)
        with self.assertRaises(TypeError):
            improved_cross_entropy("state")  # type: ignore[arg-type]
        for options in (
            {"samples_per_level": 5},
            {"target_cv": 0.0},
            {"family": "vmfn"},
            {"components": 2},
            {"family": "mixture", "components": 0},
            {"covariance_floor": 0.5},
            {"max_levels": 0},
            {"em_iterations": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                improved_cross_entropy(state, **options)  # type: ignore[arg-type]
        self.assertEqual(ICE_FAMILIES, ("gaussian", "mixture"))


class SequentialImportanceTests(unittest.TestCase):
    """Sequential importance sampling against exact values."""

    def test_kernel_leaves_the_smooth_target_invariant(self) -> None:
        """A long chain reproduces the mean of ``Phi((u - b)/s) phi(u)``.

        For ``g = b - u`` the target has normalizing constant
        ``P = Phi(-b / r)`` with ``r = sqrt(1 + s^2)`` and, by Stein's
        identity, mean ``phi(b / r) / (r P)``.
        """
        b, s = 1.0, 0.5
        r = math.sqrt(1.0 + s * s)
        constant = standard_normal_cdf(-b / r)
        mean = math.exp(-0.5 * (b / r) ** 2) / math.sqrt(2.0 * math.pi) / (r * constant)
        points: list[tuple[float, ...]] = []
        values: list[float] = []
        accepted = _smoothed_chain(
            lambda u: b - u[0],
            random.Random(11),
            (1.0,),
            0.0,
            s,
            40000,
            100,
            0.6,
            points,
            values,
        )
        self.assertEqual(len(points), 40000)
        self.assertGreater(accepted, 0)
        sample = [point[0] for point in points]
        # Batch means over 40 batches give the Monte Carlo error of the mean.
        batches = [
            statistics.fmean(sample[i : i + 1000]) for i in range(0, 40000, 1000)
        ]
        error = statistics.stdev(batches) / math.sqrt(len(batches))
        self.assertLess(abs(statistics.fmean(sample) - mean), 4.0 * error)
        self.assertLess(error, 0.02)

    def test_intermediate_constants_of_a_linear_state(self) -> None:
        """The product of ratios estimates ``P_L = Phi(-beta/sqrt(1 + sigma_L^2))``.

        This fixes the level-0 constant: with ``Phi(0) = 1/2`` in the first
        weight the product would be twice the exact value.
        """
        beta = 3.0
        logs = []
        for seed in range(8):
            result = sequential_importance_sampling(linear(beta, 2), seed=seed)
            exact = standard_normal_cdf(-beta / math.sqrt(1.0 + result.scales[-1] ** 2))
            logs.append(math.log(math.prod(result.ratios) / exact))
        self.assertLess(abs(statistics.fmean(logs)), 0.3)
        self.assertLess(max(abs(item) for item in logs), math.log(2.0) - 0.1)

    def test_estimates_against_exact_values(self) -> None:
        """Linear, parallel, and four-branch states within the stated scatter.

        The reported c.o.v. is a heuristic; over 20 seeds on the six ADR
        cases the replicate c.o.v. was between 0.84 and 1.50 times the mean
        reported one and ``|log(p / exact)|`` stayed below 3.4 reported
        c.o.v. The test allows four on fixed seeds.
        """
        cases: tuple[tuple[Callable[[], ModelLimitState], float, tuple[int, ...]], ...]
        cases = (
            (lambda: linear(3.0, 2), standard_normal_cdf(-3.0), (1, 2, 3)),
            (lambda: linear(4.5, 2), standard_normal_cdf(-4.5), (1, 2)),
            (lambda: linear(3.0, 20), standard_normal_cdf(-3.0), (1, 2)),
            (lambda: linear(4.5, 20), standard_normal_cdf(-4.5), (1, 2)),
            (parallel_pair, PARALLEL_EXACT, (1, 2)),
            (four_branch_state, FOUR_BRANCH_REFERENCE_PROBABILITY, (1, 2)),
        )
        for index, (factory, exact, seeds) in enumerate(cases):
            for seed in seeds:
                with self.subTest(case=index, seed=seed):
                    result = sequential_importance_sampling(factory(), seed=seed)
                    self.assertTrue(result.converged)
                    error = abs(math.log(result.probability / exact))
                    self.assertLess(error, 4.0 * result.coefficient_of_variation)
                    self.assertAlmostEqual(
                        result.probability,
                        math.prod(result.ratios) * result.correction,
                    )

    def test_replicate_mean_and_scatter(self) -> None:
        """Sixteen runs: the mean is unbiased within its replicate error."""
        exact = standard_normal_cdf(-3.0)
        results = [
            sequential_importance_sampling(linear(3.0, 2), seed=seed)
            for seed in range(16)
        ]
        estimates = [item.probability for item in results]
        scatter = statistics.stdev(estimates)
        self.assertLess(abs(statistics.fmean(estimates) - exact), 4.0 * scatter / 4.0)
        reported = statistics.fmean(item.coefficient_of_variation for item in results)
        self.assertLess(scatter / exact / reported, 2.5)
        self.assertGreater(scatter / exact / reported, 0.4)

    def test_cost_and_history(self) -> None:
        """Evaluations follow the chain layout and the history is consistent."""
        result = sequential_importance_sampling(
            linear(3.0, 2), samples_per_level=500, chain_fraction=0.1, burn_in=2, seed=3
        )
        self.assertIsInstance(result, SequentialImportanceResult)
        self.assertEqual(result.chains, 50)
        per_level = 50 * (2 + 10 - 1)
        self.assertEqual(result.evaluations, 500 + result.levels * per_level)
        self.assertEqual(len(result.scales), result.levels)
        self.assertEqual(len(result.acceptance_rates), result.levels)
        self.assertTrue(
            all(a > b for a, b in zip(result.scales, result.scales[1:], strict=False))
        )
        self.assertTrue(all(0.0 < rate < 1.0 for rate in result.acceptance_rates))
        again = sequential_importance_sampling(
            linear(3.0, 2), samples_per_level=500, chain_fraction=0.1, burn_in=2, seed=3
        )
        self.assertEqual(result, again)

    def test_frequent_event_is_crude_monte_carlo(self) -> None:
        """With no smooth level the estimate is the failure fraction."""
        result = sequential_importance_sampling(linear(0.2, 2), seed=1)
        self.assertEqual(result.levels, 0)
        self.assertEqual(result.evaluations, 1000)
        self.assertAlmostEqual(result.probability, result.failures / 1000)
        binomial = math.sqrt((1 - result.probability) / (result.probability * 1000))
        self.assertAlmostEqual(result.coefficient_of_variation, binomial, places=3)

    def test_unconverged_run_is_flagged(self) -> None:
        """Too few levels warn and set ``converged=False``."""
        with self.assertWarns(UserWarning):
            result = sequential_importance_sampling(
                linear(4.5, 2), max_levels=1, seed=1
            )
        self.assertFalse(result.converged)
        self.assertEqual(result.levels, 1)

    def test_validation(self) -> None:
        """Invalid options raise."""
        state = linear(3.0, 2)
        with self.assertRaises(TypeError):
            sequential_importance_sampling(None)  # type: ignore[arg-type]
        for options in (
            {"samples_per_level": 5},
            {"target_cv": -1.0},
            {"chain_fraction": 0.0},
            {"chain_fraction": 0.003},
            {"chain_fraction": 1.0, "burn_in": 0},
            {"burn_in": -1},
            {"initial_scale": 0.0},
            {"target_acceptance": 1.0},
            {"adaptation_chains": 0},
            {"max_levels": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sequential_importance_sampling(state, **options)  # type: ignore[arg-type]


MU0, S0, NOISE = 1.0, 2.0, 0.5
DATA = (3.1, 2.7, 3.4, 2.9, 3.3)


def normal_log_likelihood(theta: Sequence[float]) -> float:
    """Return the log-likelihood of ``DATA`` under ``N(theta, NOISE^2)``."""
    return math.fsum(
        -0.5 * ((y - theta[0]) / NOISE) ** 2
        - math.log(NOISE)
        - 0.5 * math.log(2.0 * math.pi)
        for y in DATA
    )


def conjugate_reference() -> tuple[float, float, float, float]:
    """Return posterior mean, variance, log evidence, and ``-sup log L``.

    Normal prior ``N(MU0, S0^2)`` and normal data with known noise. The
    evidence follows from ``Z = L(t) prior(t) / posterior(t)`` at any ``t``.
    """
    count = len(DATA)
    average = math.fsum(DATA) / count
    precision = 1.0 / S0**2 + count / NOISE**2
    mean = (MU0 / S0**2 + count * average / NOISE**2) / precision
    variance = 1.0 / precision
    log_prior = Normal(MU0, S0).log_pdf(mean)
    log_posterior = Normal(mean, math.sqrt(variance)).log_pdf(mean)
    log_evidence = normal_log_likelihood((mean,)) + log_prior - log_posterior
    return mean, variance, log_evidence, -normal_log_likelihood((average,))


PRIOR = JointInput((RandomInput("theta", Normal(MU0, S0)),))


class BayesianUpdatingTests(unittest.TestCase):
    """BUS on a conjugate normal-normal problem."""

    def test_limit_state_encodes_the_acceptance_event(self) -> None:
        """``g <= 0`` exactly when ``Phi(u_p) <= c L(theta)``."""
        _, _, _, log_c = conjugate_reference()
        state = bus_limit_state(normal_log_likelihood, PRIOR, log_constant=log_c)
        self.assertEqual(state.dimension, 2)
        self.assertEqual(state.quantity, BUS_MARGIN)
        generator = random.Random(2)
        for _ in range(50):
            u = (generator.gauss(1.0, 0.3), generator.gauss(-1.0, 1.5))
            theta = MU0 + S0 * u[0]
            scaled = math.exp(log_c + normal_log_likelihood((theta,)))
            self.assertLessEqual(scaled, 1.0 + 1e-12)
            accept = standard_normal_cdf(u[1]) <= scaled
            self.assertEqual(state.standard(u) <= 0.0, accept)

    def test_acceptance_probability_is_c_times_evidence(self) -> None:
        """Crude Monte Carlo on the limit state gives ``c Z``."""
        _, _, log_evidence, log_c = conjugate_reference()
        state = bus_limit_state(normal_log_likelihood, PRIOR, log_constant=log_c)
        generator = random.Random(7)
        draws = 4000
        hits = sum(
            1
            for _ in range(draws)
            if state.standard((generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)))
            <= 0.0
        )
        exact = math.exp(log_evidence + log_c)
        error = math.sqrt(exact * (1.0 - exact) / draws)
        self.assertLess(abs(hits / draws - exact), 4.0 * error)

    def test_conjugate_posterior_and_evidence(self) -> None:
        """Posterior moments and evidence against the closed form."""
        mean, variance, log_evidence, log_c = conjugate_reference()
        results = [
            bus_subset_simulation(
                normal_log_likelihood, PRIOR, log_constant=log_c, seed=seed
            )
            for seed in range(8)
        ]
        for result in results:
            self.assertIsInstance(result, BUSResult)
            self.assertTrue(result.converged)
            self.assertFalse(result.constant_violated)
            self.assertLessEqual(result.max_log_scaled_likelihood, 0.0)
            self.assertEqual(len(result.samples), 1000)
            self.assertLess(
                abs(result.log_evidence - log_evidence),
                4.0 * result.coefficient_of_variation_upper,
            )
        means = [item.mean()[0] for item in results]
        variances = [item.covariance()[0][0] for item in results]
        root = math.sqrt(len(results))
        self.assertLess(
            abs(statistics.fmean(means) - mean), 4.0 * statistics.stdev(means) / root
        )
        self.assertLess(
            abs(statistics.fmean(variances) - variance),
            4.0 * statistics.stdev(variances) / root,
        )
        self.assertLess(statistics.stdev(means), 0.5 * math.sqrt(variance))
        logs = [item.log_evidence for item in results]
        self.assertLess(
            abs(statistics.fmean(logs) - log_evidence),
            4.0 * statistics.stdev(logs) / root + 0.02,
        )

    def test_matches_library_subset_simulation(self) -> None:
        """The acceptance probability equals ``subset_simulation`` on the state."""
        _, _, _, log_c = conjugate_reference()
        for sampler in ("acs", "cs"):
            result = bus_subset_simulation(
                normal_log_likelihood,
                PRIOR,
                log_constant=log_c,
                sampler=sampler,
                final_level=False,
                seed=5,
            )
            reference = subset_simulation(
                bus_limit_state(normal_log_likelihood, PRIOR, log_constant=log_c),
                sampler=sampler,
                seed=5,
            )
            self.assertEqual(result.acceptance_probability, reference.probability)
            self.assertEqual(result.evaluations, reference.evaluations)
            self.assertEqual(result.thresholds, reference.thresholds)
            self.assertAlmostEqual(
                result.evidence, reference.probability / math.exp(log_c)
            )
            self.assertFalse(result.final_level)
            self.assertGreaterEqual(len(result.samples), 100)
            self.assertLess(len(result.samples), 1000)

    def test_draws_satisfy_the_event_and_report_likelihoods(self) -> None:
        """Returned draws map through the prior and carry their ``log L``."""
        _, _, _, log_c = conjugate_reference()
        result = bus_subset_simulation(
            normal_log_likelihood, PRIOR, log_constant=log_c, seed=2
        )
        for theta, u, value in zip(
            result.samples[:50],
            result.standard_samples[:50],
            result.log_likelihoods[:50],
            strict=True,
        ):
            self.assertAlmostEqual(theta[0], MU0 + S0 * u[0], places=9)
            self.assertAlmostEqual(value, normal_log_likelihood(theta), places=7)

    def test_too_large_constant_is_flagged(self) -> None:
        """``c`` above ``1 / sup L`` warns, is counted, and distorts the draws."""
        _, variance, _, log_c = conjugate_reference()
        with self.assertWarns(UserWarning):
            result = bus_subset_simulation(
                normal_log_likelihood, PRIOR, log_constant=log_c + 2.0, seed=1
            )
        self.assertTrue(result.constant_violated)
        self.assertGreater(result.constant_violations, 0)
        self.assertGreater(result.max_log_scaled_likelihood, 1.0)
        self.assertLessEqual(result.max_log_scaled_likelihood, 2.0)
        # The truncated likelihood is flatter, so the draws are too wide.
        self.assertGreater(result.covariance()[0][0], 1.4 * variance)

    def test_standard_normal_prior_without_joint_input(self) -> None:
        """``prior=None`` uses ``theta = u``; one observation, closed form."""
        observation, noise = 1.5, 0.8

        def log_likelihood(theta: Sequence[float]) -> float:
            """Return ``log N(observation; theta, noise^2)``."""
            return Normal(theta[0], noise).log_pdf(observation)

        log_c = -Normal(0.0, noise).log_pdf(0.0)
        variance = 1.0 / (1.0 + 1.0 / noise**2)
        mean = variance * observation / noise**2
        log_evidence = Normal(0.0, math.sqrt(1.0 + noise**2)).log_pdf(observation)
        result = bus_subset_simulation(
            log_likelihood, dimension=1, log_constant=log_c, seed=4
        )
        self.assertLess(abs(result.mean()[0] - mean), 0.15)
        self.assertLess(abs(result.covariance()[0][0] - variance), 0.12)
        self.assertLess(
            abs(result.log_evidence - log_evidence),
            4.0 * result.coefficient_of_variation_upper,
        )

    def test_zero_likelihood_and_validation(self) -> None:
        """Zero likelihood is never accepted; invalid inputs raise."""
        state = bus_limit_state(lambda theta: -math.inf, dimension=1, log_constant=0.0)
        self.assertGreater(state.standard((0.0, -5.0)), 0.0)
        bad = bus_limit_state(lambda theta: math.nan, dimension=1, log_constant=0.0)
        with self.assertRaises(ValueError):
            bad.standard((0.0, 0.0))
        with self.assertRaises(TypeError):
            bus_limit_state(3.0, dimension=1, log_constant=0.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bus_limit_state(normal_log_likelihood, "prior", log_constant=0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bus_limit_state(normal_log_likelihood, log_constant=0.0)
        with self.assertRaises(ValueError):
            bus_limit_state(normal_log_likelihood, PRIOR, dimension=3, log_constant=0.0)
        with self.assertRaises(ValueError):
            bus_limit_state(normal_log_likelihood, PRIOR, log_constant=math.inf)
        for options in (
            {"sampler": "mma"},
            {"samples_per_level": 5},
            {"level_probability": 0.7},
            {"samples_per_level": 105},
            {"correlation": 1.0},
            {"initial_scale": 0.0},
            {"target_acceptance": 0.0},
            {"max_levels": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bus_subset_simulation(
                    normal_log_likelihood,
                    PRIOR,
                    log_constant=0.0,
                    **options,  # type: ignore[arg-type]
                )
        empty = BUSResult(
            (), (), (), 0.0, -math.inf, 0.0, math.inf, math.inf, (0.0,), 1, 10,
            False, 10, False, "acs", 0.0, -1.0, 0, False,
        )  # fmt: skip
        with self.assertRaises(ValueError):
            empty.mean()
        with self.assertRaises(ValueError):
            empty.covariance()

    def test_unconverged_run_is_flagged(self) -> None:
        """A level limit below the need warns and sets ``converged=False``."""
        _, _, _, log_c = conjugate_reference()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bus_subset_simulation(
                normal_log_likelihood,
                PRIOR,
                log_constant=log_c - 12.0,
                max_levels=2,
                seed=1,
            )
        self.assertFalse(result.converged)
        self.assertTrue(any("did not reach" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
