"""Tests for nested R-hat, SVGD, slice sampling, MALA, stretch move, and PT.

Oracles: hand computation and algebraic identities (nested R-hat), gradient
ascent and Gaussian moments (SVGD), quadrature moments of a bimodal density
and conjugate Gaussian posteriors (slice samplers), numerical detailed
balance (MALA), exact affine equivariance (stretch move), and the known mode
weights and conjugate evidence (parallel tempering).
"""

from __future__ import annotations

import math
import random
import unittest
import warnings

from probstat.foundations.random_streams import RandomStream
from probstat.inference import (
    ConvergenceWarning,
    elliptical_slice_sampler,
    mala_log_acceptance_ratio,
    median_bandwidth,
    metropolis_adjusted_langevin,
    nested_rhat,
    parallel_tempering,
    random_walk_metropolis,
    slice_sampler,
    stein_variational_gradient_descent,
    stretch_move_sampler,
    svgd_direction,
)
from probstat.linalg import inverse_spd, matvec

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _within_mcse(case: unittest.TestCase, result, exact, factor: float = 4.5) -> None:
    """Assert that every posterior mean is within ``factor`` MCSE of ``exact``."""
    for summary, value in zip(result.summary(), exact, strict=True):
        case.assertAlmostEqual(summary.mean, value, delta=factor * summary.mcse_mean)


# ---------------------------------------------------------------------------
# Nested R-hat
# ---------------------------------------------------------------------------


class NestedRhatTests(unittest.TestCase):
    """The nested R-hat of Margossian et al."""

    def test_hand_computed_example(self) -> None:
        """Two superchains of two chains of two draws, computed by hand."""
        superchains = [[[0.0, 2.0], [2.0, 4.0]], [[5.0, 7.0], [9.0, 11.0]]]
        # chain means 1, 3 | 6, 10; superchain means 2, 8; B = 18.
        # B~ = 2, 8; within-chain variances all 2; W = ((2 + 2) + (8 + 2)) / 2 = 7.
        result = nested_rhat(superchains)
        self.assertAlmostEqual(result.between_superchain_variance, 18.0, places=13)
        self.assertAlmostEqual(result.within_superchain_variance, 7.0, places=13)
        self.assertAlmostEqual(result.value, math.sqrt(1.0 + 18.0 / 7.0), places=13)
        self.assertEqual(
            (result.superchains, result.chains_per_superchain, result.draws_per_chain),
            (2, 2, 2),
        )

    def test_single_chain_superchains_relate_to_classical_rhat(self) -> None:
        """With M = 1, nR-hat^2 exceeds the non-split R-hat^2 by exactly 1/N."""
        generator = random.Random(3)
        chains = [[generator.gauss(0.3 * k, 1.0) for _ in range(25)] for k in range(5)]
        result = nested_rhat([[chain] for chain in chains])
        n = 25
        means = [math.fsum(chain) / n for chain in chains]
        within = math.fsum(
            math.fsum((x - m) ** 2 for x in chain) / (n - 1)
            for chain, m in zip(chains, means, strict=True)
        ) / len(chains)
        grand = math.fsum(means) / len(means)
        between = math.fsum((m - grand) ** 2 for m in means) / (len(means) - 1)
        classical_squared = (n - 1) / n + between / within
        self.assertAlmostEqual(result.value**2 - classical_squared, 1.0 / n, places=12)

    def test_one_draw_per_chain(self) -> None:
        """With N = 1 the within term is the between-chain variance alone."""
        result = nested_rhat([[[1.0], [3.0]], [[4.0], [8.0]], [[0.0], [2.0]]])
        # chain values are the chain means; B~ = 2, 8, 2; W = 4.
        self.assertAlmostEqual(result.within_superchain_variance, 4.0, places=13)
        # superchain means 2, 6, 1 -> variance 7.
        self.assertAlmostEqual(result.between_superchain_variance, 7.0, places=13)

    def test_stationary_chains_give_the_expected_ratio(self) -> None:
        """For iid draws E[B] / E[W] = 1 / (M (N + 1))."""
        generator = random.Random(11)
        superchain_count, chain_count, draws = 16, 8, 4
        ratios_b = []
        ratios_w = []
        for _ in range(200):
            data = [
                [
                    [generator.gauss(0.0, 1.0) for _ in range(draws)]
                    for _ in range(chain_count)
                ]
                for _ in range(superchain_count)
            ]
            result = nested_rhat(data)
            ratios_b.append(result.between_superchain_variance)
            ratios_w.append(result.within_superchain_variance)
        mean_b = math.fsum(ratios_b) / 200
        mean_w = math.fsum(ratios_w) / 200
        self.assertAlmostEqual(mean_w, 1.0 + 1.0 / draws, delta=0.02)
        self.assertAlmostEqual(mean_b, 1.0 / (chain_count * draws), delta=0.003)
        self.assertAlmostEqual(
            mean_b / mean_w, 1.0 / (chain_count * (draws + 1)), delta=0.003
        )

    def test_detects_chains_that_remember_their_start(self) -> None:
        """Superchains still near distinct starts give a value far above one."""
        generator = random.Random(5)
        stuck = [
            [[start + 0.1 * generator.gauss(0, 1) for _ in range(3)] for _ in range(10)]
            for start in (-3.0, 0.0, 2.0, 5.0)
        ]
        mixed = [
            [[generator.gauss(0, 1) for _ in range(3)] for _ in range(10)]
            for _ in range(4)
        ]
        self.assertGreater(nested_rhat(stuck).value, 10.0)
        self.assertLess(nested_rhat(mixed).value, 1.2)

    def test_degenerate_and_invalid_input(self) -> None:
        """Constant draws give nan/inf; malformed input raises."""
        self.assertTrue(math.isnan(nested_rhat([[[1.0, 1.0]], [[1.0, 1.0]]]).value))
        self.assertEqual(nested_rhat([[[1.0, 1.0]], [[2.0, 2.0]]]).value, math.inf)
        with self.assertRaisesRegex(ValueError, "at least 2 superchains"):
            nested_rhat([[[1.0, 2.0]]])
        with self.assertRaisesRegex(ValueError, "same positive number of chains"):
            nested_rhat([[[1.0, 2.0]], [[1.0, 2.0], [2.0, 3.0]]])
        with self.assertRaisesRegex(ValueError, "same positive number of draws"):
            nested_rhat([[[1.0, 2.0]], [[1.0]]])
        with self.assertRaisesRegex(ValueError, "no within-superchain"):
            nested_rhat([[[1.0]], [[2.0]]])
        with self.assertRaises(ValueError):
            nested_rhat([[[1.0, math.nan]], [[1.0, 2.0]]])


# ---------------------------------------------------------------------------
# Stein variational gradient descent
# ---------------------------------------------------------------------------


def _normal_particles(count: int, dimension: int, seed: int) -> list[tuple[float, ...]]:
    """Draw seeded standard-normal starting particles."""
    generator = RandomStream(seed).generator()
    return [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    ]


class SVGDTests(unittest.TestCase):
    """Stein variational gradient descent."""

    def test_direction_matches_brute_force_formula(self) -> None:
        """phi(x_i) equals the double sum of the source, coded independently."""
        particles = _normal_particles(6, 2, 4)
        scores = [(-x[0] + 0.3, -2.0 * x[1]) for x in particles]
        h = 0.8
        got = svgd_direction(particles, scores, bandwidth=h)
        for i, x in enumerate(particles):
            for a in range(2):
                total = 0.0
                for j, xj in enumerate(particles):
                    kernel = math.exp(-(math.dist(x, xj) ** 2) / h)
                    # grad_{x_j} k(x_j, x) = (2 / h) (x - x_j) k
                    total += kernel * scores[j][a] + 2.0 / h * (x[a] - xj[a]) * kernel
                self.assertAlmostEqual(got[i][a], total / 6, places=13)

    def test_median_bandwidth(self) -> None:
        """H = med^2 / log n for three collinear points; fallbacks are 1."""
        # pairwise distances 1, 3, 4 -> median 3.
        self.assertAlmostEqual(
            median_bandwidth([(0.0,), (1.0,), (4.0,)]), 9.0 / math.log(3), places=13
        )
        self.assertEqual(median_bandwidth([(2.0, 1.0)]), 1.0)
        self.assertEqual(median_bandwidth([(2.0,), (2.0,), (2.0,)]), 1.0)

    def test_single_particle_is_gradient_ascent(self) -> None:
        """N = 1: phi is the score and the particle ends at the mode."""
        score = (0.7, -1.1)
        self.assertEqual(svgd_direction([(3.0, 4.0)], [score]), (score,))

        def log_density(x):
            """Gaussian with mode (1, -2) and unequal curvatures."""
            return -0.5 * ((x[0] - 1.0) ** 2 / 4.0 + (x[1] + 2.0) ** 2 / 0.25)

        result = stein_variational_gradient_descent(
            log_density, [(5.0, 5.0)], step_size=0.1, iterations=2000, tolerance=1e-10
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.particles[0][0], 1.0, places=7)
        self.assertAlmostEqual(result.particles[0][1], -2.0, places=7)
        self.assertEqual(result.gradient_method, "autodiff")
        # one fixed-step gradient-ascent iterate, by hand
        one = stein_variational_gradient_descent(
            log_density, [(5.0, 5.0)], step_size=0.1, iterations=1
        )
        self.assertAlmostEqual(
            one.particles[0][0], 5.0 + 0.1 * (-(5.0 - 1.0) / 4.0), places=12
        )
        self.assertAlmostEqual(
            one.particles[0][1], 5.0 + 0.1 * (-(5.0 + 2.0) / 0.25), places=12
        )

    def test_gaussian_mean_and_variance_in_one_dimension(self) -> None:
        """The particle mean is the target mean; the variance grows towards it."""
        variances = []
        for count in (5, 10, 25):
            result = stein_variational_gradient_descent(
                lambda x: -0.5 * ((x[0] - 1.5) / 2.0) ** 2,
                _normal_particles(count, 1, 1),
                step_size=1.0,
                iterations=800,
                tolerance=1e-7,
                gradient=lambda x: ((1.5 - x[0]) / 4.0,),
            )
            self.assertAlmostEqual(result.mean()[0], 1.5, delta=2e-3)
            self.assertLess(result.update_norm, 1e-3)
            variances.append(result.covariance()[0][0])
        self.assertEqual(variances, sorted(variances))
        self.assertTrue(all(v < 4.0 for v in variances))
        self.assertGreater(variances[0], 0.6 * 4.0)
        self.assertGreater(variances[-1], 0.85 * 4.0)

    def test_two_dimensional_gaussian(self) -> None:
        """Mean recovered; both variances within a factor of the target's."""
        result = stein_variational_gradient_descent(
            lambda x: -0.5 * ((x[0] - 1.0) ** 2 / 4.0 + (x[1] + 2.0) ** 2 / 0.25),
            _normal_particles(20, 2, 1),
            step_size=0.1,
            iterations=1500,
        )
        self.assertFalse(result.converged)  # tolerance 0 is never met exactly
        self.assertEqual(result.iterations, 1500)
        self.assertAlmostEqual(result.mean()[0], 1.0, delta=0.02)
        self.assertAlmostEqual(result.mean()[1], -2.0, delta=0.02)
        covariance = result.covariance()
        self.assertTrue(0.7 * 4.0 < covariance[0][0] < 4.0)
        self.assertTrue(0.7 * 0.25 < covariance[1][1] < 0.25)

    def test_variance_is_underestimated_with_few_particles_in_high_dimension(
        self,
    ) -> None:
        """Ten particles: the per-coordinate variance falls as d grows."""
        spreads = []
        for dimension in (1, 4, 8):
            result = stein_variational_gradient_descent(
                lambda x: -0.5 * math.fsum(v * v for v in x),
                _normal_particles(10, dimension, 1),
                step_size=0.3,
                iterations=600,
                gradient=lambda x: tuple(-v for v in x),
            )
            covariance = result.covariance()
            spreads.append(
                math.fsum(covariance[i][i] for i in range(dimension)) / dimension
            )
        self.assertGreater(spreads[0], 0.7)
        self.assertLess(spreads[1], 0.55)
        self.assertLess(spreads[2], spreads[1])

    def test_gradient_sources_agree_and_run_is_deterministic(self) -> None:
        """Autodiff, analytic, and finite-difference scores give the same run."""
        start = _normal_particles(6, 1, 2)
        runs = [
            stein_variational_gradient_descent(
                lambda x: -0.5 * x[0] * x[0],
                start,
                step_size=0.5,
                iterations=50,
                gradient=option,
            )
            for option in ("auto", (lambda x: (-x[0],)), "finite-difference")
        ]
        self.assertEqual(
            [r.gradient_method for r in runs],
            ["autodiff", "analytic", "finite-difference"],
        )
        for a, b in zip(runs[0].particles, runs[1].particles, strict=True):
            self.assertAlmostEqual(a[0], b[0], places=12)
        for a, b in zip(runs[0].particles, runs[2].particles, strict=True):
            self.assertAlmostEqual(a[0], b[0], places=6)
        self.assertEqual(runs[0].gradient_evaluations, 6 * 51)
        self.assertEqual(runs[0].initial, tuple(start))

    def test_validation(self) -> None:
        """Invalid arguments raise; a non-finite score is reported."""
        log_density = lambda x: -0.5 * x[0] * x[0]  # noqa: E731
        with self.assertRaises(TypeError):
            stein_variational_gradient_descent(
                1.0, [(0.0,)], step_size=1.0, iterations=1
            )  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "at least one particle"):
            stein_variational_gradient_descent(
                log_density, [], step_size=1.0, iterations=1
            )
        with self.assertRaisesRegex(ValueError, "share a positive dimension"):
            stein_variational_gradient_descent(
                log_density, [(0.0,), (1.0, 2.0)], step_size=1.0, iterations=1
            )
        with self.assertRaisesRegex(ValueError, "step_size must be positive"):
            stein_variational_gradient_descent(
                log_density, [(0.0,)], step_size=0.0, iterations=1
            )
        with self.assertRaisesRegex(ValueError, "bandwidth must be positive"):
            stein_variational_gradient_descent(
                log_density, [(0.0,)], step_size=1.0, iterations=1, bandwidth=-1.0
            )
        with self.assertRaisesRegex(ValueError, "bandwidth must be positive"):
            svgd_direction([(0.0,)], [(0.0,)], bandwidth=0.0)
        with self.assertRaisesRegex(ValueError, "shape of particles"):
            svgd_direction([(0.0,), (1.0,)], [(0.0,)])
        with self.assertRaisesRegex(ValueError, "not finite at a particle"):
            stein_variational_gradient_descent(
                lambda x: math.log(x[0]),
                [(-1.0,)],
                step_size=0.1,
                iterations=1,
                gradient=lambda x: (math.nan,),
            )


# ---------------------------------------------------------------------------
# Slice sampling
# ---------------------------------------------------------------------------


def _bimodal(x) -> float:
    """Log-density of 0.3 N(-2, 0.8^2) + 0.7 N(2, 0.8^2) up to a constant."""
    a = math.log(0.3) - 0.5 * ((x[0] + 2.0) / 0.8) ** 2
    b = math.log(0.7) - 0.5 * ((x[0] - 2.0) / 0.8) ** 2
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


def _quadrature_moments(log_density, lower: float, upper: float) -> tuple[float, float]:
    """Return mean and variance of a 1-D density by the trapezoidal rule."""
    count = 20000
    step = (upper - lower) / count
    xs = [lower + step * i for i in range(count + 1)]
    ws = [math.exp(log_density((x,))) for x in xs]
    total = math.fsum(ws)
    mean = math.fsum(w * x for w, x in zip(ws, xs, strict=True)) / total
    second = math.fsum(w * x * x for w, x in zip(ws, xs, strict=True)) / total
    return mean, second - mean * mean


class SliceSamplerTests(unittest.TestCase):
    """Coordinate-wise slice sampling with stepping out and shrinkage."""

    def test_bimodal_moments_match_quadrature(self) -> None:
        """Mean and variance of a two-mode density agree with quadrature."""
        mean, variance = _quadrature_moments(_bimodal, -10.0, 10.0)
        self.assertAlmostEqual(mean, 0.8, places=6)
        self.assertAlmostEqual(variance, 4.0, places=6)
        result = slice_sampler(
            _bimodal, [0.0], draws=6000, warmup=200, chains=2, widths=4.0, seed=5
        )
        self.assertEqual(result.warnings, ())
        _within_mcse(self, result, (mean,))
        self.assertAlmostEqual(result.covariance()[0][0], variance, delta=0.15)
        self.assertEqual(result.acceptance_rate, 1.0)
        self.assertEqual(result.algorithm, "slice sampling")
        self.assertGreater(result.evaluations, 2 * 6200 * 2)

    def test_limited_stepping_out_is_still_exact(self) -> None:
        """A step limit m with a small width keeps the Gaussian moments."""
        result = slice_sampler(
            lambda x: -0.5 * ((x[0] - 1.0) / 2.0) ** 2,
            [1.0],
            draws=6000,
            warmup=100,
            chains=2,
            widths=0.7,
            max_steps=2,
            seed=8,
            check_convergence=False,
        )
        _within_mcse(self, result, (1.0,))
        self.assertAlmostEqual(result.covariance()[0][0], 4.0, delta=0.35)

    def test_correlated_gaussian_and_bounded_support(self) -> None:
        """Two coordinates: a correlated Gaussian and an exponential on x > 0."""

        def log_density(x):
            """N(0, [[1, .8], [.8, 1]]) in (x0, x1) times Exp(1) in x2."""
            if x[2] <= 0.0:
                return -math.inf
            quadratic = (x[0] ** 2 - 1.6 * x[0] * x[1] + x[1] ** 2) / (1 - 0.64)
            return -0.5 * quadratic - x[2]

        result = slice_sampler(
            log_density,
            [0.0, 0.0, 1.0],
            draws=5000,
            warmup=200,
            chains=2,
            widths=[2.0, 2.0, 1.0],
            seed=3,
            names=("a", "b", "e"),
        )
        _within_mcse(self, result, (0.0, 0.0, 1.0))
        covariance = result.covariance()
        self.assertAlmostEqual(covariance[0][0], 1.0, delta=0.15)
        self.assertAlmostEqual(covariance[0][1], 0.8, delta=0.15)
        self.assertAlmostEqual(covariance[2][2], 1.0, delta=0.15)
        self.assertTrue(all(s[2] > 0.0 for s in result.pooled()))
        self.assertEqual(result.names, ("a", "b", "e"))

    def test_reproducible(self) -> None:
        """The same seed gives the same chain; another seed differs."""
        runs = [
            slice_sampler(_bimodal, [0.0], draws=50, seed=seed, check_convergence=False)
            for seed in (1, 1, 2)
        ]
        self.assertEqual(runs[0].chains[0].samples, runs[1].chains[0].samples)
        self.assertNotEqual(runs[0].chains[0].samples, runs[2].chains[0].samples)

    def test_validation(self) -> None:
        """Invalid widths, step limits, starts, and improper targets raise."""
        with self.assertRaisesRegex(ValueError, "widths must be positive"):
            slice_sampler(_bimodal, [0.0], draws=5, widths=0.0)
        with self.assertRaisesRegex(ValueError, "one entry per coordinate"):
            slice_sampler(_bimodal, [0.0], draws=5, widths=[1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "max_steps must be at least 1"):
            slice_sampler(_bimodal, [0.0], draws=5, max_steps=0)
        with self.assertRaisesRegex(ValueError, "finite log-density"):
            slice_sampler(lambda x: -math.inf, [0.0], draws=5)
        with self.assertRaisesRegex(ValueError, "stepping out did not end"):
            slice_sampler(lambda x: 0.0, [0.0], draws=5, seed=1)
        with self.assertRaises(TypeError):
            slice_sampler(None, [0.0], draws=5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "draws must be positive"):
            slice_sampler(_bimodal, [0.0], draws=0)


class EllipticalSliceTests(unittest.TestCase):
    """Elliptical slice sampling for Gaussian priors."""

    prior_covariance = ((2.0, 0.8), (0.8, 1.0))
    prior_mean = (1.0, -1.0)
    data = (0.5, 2.0)
    noise_variance = 0.5

    def log_likelihood(self, f) -> float:
        """Gaussian likelihood of the two observations."""
        return (
            -0.5
            * math.fsum((y - v) ** 2 for y, v in zip(self.data, f, strict=True))
            / self.noise_variance
        )

    def test_conjugate_posterior_moments(self) -> None:
        """Posterior mean and covariance match the Gaussian closed form."""
        precision = inverse_spd(self.prior_covariance)
        posterior_precision = tuple(
            tuple(
                precision[i][j] + (1.0 / self.noise_variance if i == j else 0.0)
                for j in range(2)
            )
            for i in range(2)
        )
        covariance = inverse_spd(posterior_precision)
        mean = matvec(
            covariance,
            tuple(
                a + y / self.noise_variance
                for a, y in zip(
                    matvec(precision, self.prior_mean), self.data, strict=True
                )
            ),
        )
        result = elliptical_slice_sampler(
            self.log_likelihood,
            [0.0, 0.0],
            prior_covariance=self.prior_covariance,
            prior_mean=self.prior_mean,
            draws=5000,
            warmup=200,
            chains=2,
            seed=3,
        )
        self.assertEqual(result.warnings, ())
        _within_mcse(self, result, mean)
        got = result.covariance()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(got[i][j], covariance[i][j], delta=0.03)
        self.assertEqual(result.chains[0].proposal_covariance, self.prior_covariance)
        self.assertEqual(result.algorithm, "elliptical slice sampling")

    def test_flat_likelihood_returns_the_prior(self) -> None:
        """With L = 1 every proposal is accepted and the draws follow the prior."""
        result = elliptical_slice_sampler(
            lambda f: 0.0,
            [0.0, 0.0],
            prior_covariance=self.prior_covariance,
            draws=6000,
            chains=2,
            seed=1,
        )
        self.assertEqual(result.evaluations, 2 * (6000 + 1))
        _within_mcse(self, result, (0.0, 0.0))
        got = result.covariance()
        self.assertAlmostEqual(got[0][0], 2.0, delta=0.15)
        self.assertAlmostEqual(got[0][1], 0.8, delta=0.1)
        self.assertAlmostEqual(got[1][1], 1.0, delta=0.08)

    def test_truncating_likelihood_stays_in_support(self) -> None:
        """A half-space indicator likelihood gives the half-normal mean."""
        result = elliptical_slice_sampler(
            lambda f: 0.0 if f[0] > 0.0 else -math.inf,
            [1.0],
            prior_covariance=((1.0,),),
            draws=6000,
            chains=2,
            seed=4,
        )
        self.assertTrue(all(s[0] > 0.0 for s in result.pooled()))
        _within_mcse(self, result, (math.sqrt(2.0 / math.pi),))

    def test_validation(self) -> None:
        """Bad covariance, mean, or start raise ValueError."""
        with self.assertRaisesRegex(ValueError, "positive definite"):
            elliptical_slice_sampler(
                lambda f: 0.0,
                [0.0, 0.0],
                prior_covariance=((1.0, 2.0), (2.0, 1.0)),
                draws=5,
            )
        with self.assertRaisesRegex(ValueError, "match the state dimension"):
            elliptical_slice_sampler(
                lambda f: 0.0, [0.0], prior_covariance=self.prior_covariance, draws=5
            )
        with self.assertRaisesRegex(ValueError, "prior_mean must match"):
            elliptical_slice_sampler(
                lambda f: 0.0,
                [0.0],
                prior_covariance=((1.0,),),
                prior_mean=(0.0, 1.0),
                draws=5,
            )
        with self.assertRaisesRegex(ValueError, "finite log-likelihood"):
            elliptical_slice_sampler(
                lambda f: -math.inf, [0.0], prior_covariance=((1.0,),), draws=5
            )


# ---------------------------------------------------------------------------
# MALA
# ---------------------------------------------------------------------------


class MALATests(unittest.TestCase):
    """Metropolis-adjusted Langevin algorithm."""

    def test_detailed_balance_of_the_kernel_density(self) -> None:
        """pi(x) q(y|x) alpha(x,y) = pi(y) q(x|y) alpha(y,x) on a grid of pairs."""

        def log_pi(x: float) -> float:
            """Return a non-Gaussian, asymmetric log-density."""
            return -0.25 * x**4 + 0.3 * x**3 - 0.5 * x * x

        def score(x: float) -> float:
            """Return the derivative of ``log_pi``."""
            return -(x**3) + 0.9 * x * x - x

        def log_q(to: float, start: float, eps: float) -> float:
            """Log-density of the Langevin proposal, coded independently."""
            mean = start + 0.5 * eps * eps * score(start)
            return -0.5 * ((to - mean) / eps) ** 2 - math.log(eps) - LOG_SQRT_TWO_PI

        for eps in (0.3, 0.9):
            for x in (-1.5, -0.2, 0.4, 1.3):
                for y in (-1.1, 0.1, 0.9, 2.0):
                    forward = mala_log_acceptance_ratio(
                        log_pi(x), log_pi(y), [x], [y], [score(x)], [score(y)], eps
                    )
                    backward = mala_log_acceptance_ratio(
                        log_pi(y), log_pi(x), [y], [x], [score(y)], [score(x)], eps
                    )
                    self.assertAlmostEqual(forward, -backward, places=12)
                    self.assertAlmostEqual(
                        forward,
                        log_pi(y) + log_q(x, y, eps) - log_pi(x) - log_q(y, x, eps),
                        places=12,
                    )
                    left = log_pi(x) + log_q(y, x, eps) + min(0.0, forward)
                    right = log_pi(y) + log_q(x, y, eps) + min(0.0, backward)
                    self.assertAlmostEqual(left, right, places=12)

    def test_gaussian_moments(self) -> None:
        """Mean and covariance of a correlated Gaussian within Monte Carlo error."""

        def log_density(x):
            """x0 ~ N(0, 1), x1 | x0 ~ N(0.5 x0 + 1, 0.5)."""
            return -0.5 * (x[0] ** 2 + (x[1] - 1.0 - 0.5 * x[0]) ** 2 / 0.5)

        result = metropolis_adjusted_langevin(
            log_density,
            [0.0, 1.0],
            step_size=0.7,
            draws=8000,
            warmup=500,
            chains=2,
            seed=1,
        )
        self.assertEqual(result.warnings, ())
        _within_mcse(self, result, (0.0, 1.0))
        covariance = result.covariance()
        self.assertAlmostEqual(covariance[0][0], 1.0, delta=0.08)
        self.assertAlmostEqual(covariance[0][1], 0.5, delta=0.06)
        self.assertAlmostEqual(covariance[1][1], 0.75, delta=0.06)
        self.assertTrue(0.6 < result.acceptance_rate < 0.98)
        self.assertEqual(result.chains[0].proposal_covariance[0][0], 0.7**2)
        self.assertEqual(result.algorithm, "Metropolis-adjusted Langevin")

    def test_non_gaussian_target_moments_match_quadrature(self) -> None:
        """A skewed quartic target: the corrected chain has the exact mean."""

        def log_density(x):
            """Return the density of the detailed-balance test."""
            return -0.25 * x[0] ** 4 + 0.3 * x[0] ** 3 - 0.5 * x[0] ** 2

        mean, variance = _quadrature_moments(log_density, -6.0, 6.0)
        result = metropolis_adjusted_langevin(
            log_density, [0.0], step_size=0.9, draws=8000, warmup=300, chains=2, seed=6
        )
        _within_mcse(self, result, (mean,))
        self.assertAlmostEqual(result.covariance()[0][0], variance, delta=0.05)

    def test_gradient_sources_and_reproducibility(self) -> None:
        """Analytic and autodiff gradients give the same chain for one seed."""
        kwargs = dict(step_size=0.8, draws=200, seed=4, check_convergence=False)
        auto = metropolis_adjusted_langevin(lambda x: -0.5 * x[0] ** 2, [0.5], **kwargs)
        analytic = metropolis_adjusted_langevin(
            lambda x: -0.5 * x[0] ** 2, [0.5], gradient=lambda x: (-x[0],), **kwargs
        )
        for a, b in zip(
            auto.chains[0].samples, analytic.chains[0].samples, strict=True
        ):
            self.assertAlmostEqual(a[0], b[0], places=10)
        self.assertEqual(auto.chains[0].evaluations, 201)

    def test_acceptance_falls_with_the_step(self) -> None:
        """A larger step lowers the acceptance rate."""
        rates = [
            metropolis_adjusted_langevin(
                lambda x: -0.5 * math.fsum(v * v for v in x),
                [0.0] * 5,
                step_size=step,
                draws=1500,
                seed=2,
                check_convergence=False,
            ).acceptance_rate
            for step in (0.3, 1.0, 1.8)
        ]
        self.assertGreater(rates[0], 0.95)
        self.assertGreater(rates[0], rates[1])
        self.assertGreater(rates[1], rates[2])

    def test_validation(self) -> None:
        """Invalid steps, starts, and ratio arguments raise."""
        with self.assertRaisesRegex(ValueError, "step_size must be positive"):
            metropolis_adjusted_langevin(lambda x: 0.0, [0.0], step_size=0.0, draws=5)
        with self.assertRaisesRegex(ValueError, "finite log-density"):
            metropolis_adjusted_langevin(
                lambda x: -math.inf, [0.0], step_size=0.1, draws=5
            )
        with self.assertRaises(TypeError):
            metropolis_adjusted_langevin(3, [0.0], step_size=0.1, draws=5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "same length"):
            mala_log_acceptance_ratio(0.0, 0.0, [0.0], [0.0, 1.0], [0.0], [0.0], 0.1)
        with self.assertRaisesRegex(ValueError, "step_size must be positive"):
            mala_log_acceptance_ratio(0.0, 0.0, [0.0], [0.0], [0.0], [0.0], -1.0)


# ---------------------------------------------------------------------------
# Stretch move
# ---------------------------------------------------------------------------


class StretchMoveTests(unittest.TestCase):
    """The affine-invariant ensemble sampler."""

    matrix = ((3.0, 1.0), (-2.0, 0.5))
    shift = (10.0, -4.0)

    def affine(self, x):
        """Return ``A x + b``."""
        return tuple(
            self.matrix[i][0] * x[0] + self.matrix[i][1] * x[1] + self.shift[i]
            for i in range(2)
        )

    def inverse(self, y):
        """Return ``A^-1 (y - b)``."""
        (a, b), (c, d) = self.matrix
        det = a * d - b * c
        u, v = y[0] - self.shift[0], y[1] - self.shift[1]
        return ((d * u - b * v) / det, (-c * u + a * v) / det)

    @staticmethod
    def log_density(x) -> float:
        """Return a non-Gaussian log-density (heavy tails, banana)."""
        return -math.log1p(x[0] ** 2 / 3.0) * 2.0 - 0.5 * (x[1] - 0.3 * x[0] ** 2) ** 2

    def test_trajectory_is_affine_equivariant(self) -> None:
        """Sampling pi(A^-1 (y - b)) from mapped starts gives the mapped path."""
        start = _normal_particles(8, 2, 0)
        base = stretch_move_sampler(self.log_density, start, draws=40, seed=11)
        mapped = stretch_move_sampler(
            lambda y: self.log_density(self.inverse(y)),
            [self.affine(x) for x in start],
            draws=40,
            seed=11,
        )
        worst = max(
            abs(u - v)
            for walker, other in zip(base.walkers, mapped.walkers, strict=True)
            for state, image in zip(walker, other, strict=True)
            for u, v in zip(self.affine(state), image, strict=True)
        )
        self.assertLess(worst, 1e-9)
        self.assertEqual(base.acceptance_rates, mapped.acceptance_rates)

    def test_gaussian_moments(self) -> None:
        """Mean and standard deviations of a badly scaled correlated Gaussian."""

        def log_density(y):
            """Image of the standard normal under the affine map."""
            x = self.inverse(y)
            return -0.5 * (x[0] ** 2 + x[1] ** 2)

        start = [self.affine(x) for x in _normal_particles(10, 2, 0)]
        result = stretch_move_sampler(
            log_density, start, draws=3000, warmup=300, seed=11, names=("u", "v")
        )
        # Cov = A A': variances 10 and 4.25.
        exact_sd = (math.sqrt(10.0), math.sqrt(4.25))
        mean = result.mean()
        sd = result.standard_deviation()
        for i in range(2):
            ess = result.effective_sample_size(i)
            self.assertGreater(ess, 300.0)
            self.assertAlmostEqual(
                mean[i], self.shift[i], delta=4.5 * exact_sd[i] / math.sqrt(ess)
            )
            self.assertAlmostEqual(sd[i], exact_sd[i], delta=0.08 * exact_sd[i])
        self.assertTrue(0.4 < result.acceptance_rate < 0.95)
        self.assertEqual(len(result.walkers), 10)
        self.assertEqual(len(result.walkers[0]), 3000)
        self.assertEqual(len(result.pooled()), 30000)
        self.assertEqual(result.evaluations, 10 + 10 * 3300)
        self.assertEqual(result.names, ("u", "v"))
        self.assertEqual(result.dimension, 2)
        self.assertEqual(len(result.columns(1)), 10)

    def test_one_dimension_and_odd_ensemble(self) -> None:
        """D = 1 (no Jacobian power) with five walkers samples N(2, 1)."""
        start = [(x[0] + 2.0,) for x in _normal_particles(5, 1, 3)]
        result = stretch_move_sampler(
            lambda x: -0.5 * (x[0] - 2.0) ** 2, start, draws=4000, warmup=200, seed=5
        )
        ess = result.effective_sample_size(0)
        self.assertAlmostEqual(result.mean()[0], 2.0, delta=4.5 / math.sqrt(ess))
        self.assertAlmostEqual(result.standard_deviation()[0], 1.0, delta=0.06)

    def test_reproducible(self) -> None:
        """The same seed reproduces the ensemble exactly."""
        start = _normal_particles(6, 2, 1)
        a = stretch_move_sampler(self.log_density, start, draws=20, seed=3)
        b = stretch_move_sampler(self.log_density, start, draws=20, seed=3)
        c = stretch_move_sampler(self.log_density, start, draws=20, seed=4)
        self.assertEqual(a.walkers, b.walkers)
        self.assertNotEqual(a.walkers, c.walkers)
        self.assertEqual(a.log_density[0][-1], self.log_density(a.walkers[0][-1]))

    def test_validation(self) -> None:
        """Too few or coincident walkers and a bad stretch raise."""
        with self.assertRaisesRegex(ValueError, "more walkers than dimensions"):
            stretch_move_sampler(self.log_density, [(0.0, 0.0), (1.0, 1.0)], draws=5)
        with self.assertRaisesRegex(ValueError, "at least 2 walkers"):
            stretch_move_sampler(lambda x: 0.0, [(0.0,)], draws=5)
        with self.assertRaisesRegex(ValueError, "distinct"):
            stretch_move_sampler(
                self.log_density, [(0.0, 0.0), (0.0, 0.0), (1.0, 2.0)], draws=5
            )
        with self.assertRaisesRegex(ValueError, "stretch must be greater than 1"):
            stretch_move_sampler(
                self.log_density, _normal_particles(4, 2, 1), draws=5, stretch=1.0
            )
        with self.assertRaisesRegex(ValueError, "finite log-density"):
            stretch_move_sampler(
                lambda x: -math.inf, _normal_particles(4, 2, 1), draws=5
            )
        with self.assertRaises(IndexError):
            stretch_move_sampler(
                self.log_density, _normal_particles(4, 2, 1), draws=5, seed=1
            ).columns(2)


# ---------------------------------------------------------------------------
# Parallel tempering
# ---------------------------------------------------------------------------


def _mixture_log_likelihood(x) -> float:
    """Return the normalized 0.3 N(-5, 0.5^2) + 0.7 N(5, 0.5^2) log-density."""
    a = math.log(0.3) - 0.5 * ((x[0] + 5.0) / 0.5) ** 2
    b = math.log(0.7) - 0.5 * ((x[0] - 5.0) / 0.5) ** 2
    top = max(a, b)
    return (
        top
        + math.log(math.exp(a - top) + math.exp(b - top))
        - math.log(0.5)
        - LOG_SQRT_TWO_PI
    )


def _wide_log_prior(x) -> float:
    """Return the normalized N(0, 10^2) log-density."""
    return -0.5 * (x[0] / 10.0) ** 2 - math.log(10.0) - LOG_SQRT_TWO_PI


class ParallelTemperingTests(unittest.TestCase):
    """Replica exchange with deterministic even-odd swaps."""

    betas = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    scales = (10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.6)

    @classmethod
    def setUpClass(cls) -> None:
        """Run the two-mode problem once."""
        cls.result = parallel_tempering(
            _mixture_log_likelihood,
            _wide_log_prior,
            [-5.0],
            betas=cls.betas,
            draws=20000,
            warmup=1000,
            proposal_scale=cls.scales,
            seed=2,
        )

    def test_cold_chain_recovers_the_mode_weights(self) -> None:
        """The right mode has posterior mass 0.7; plain RWM never leaves the left."""
        samples = self.result.samples
        fraction = sum(1 for s in samples if s[0] > 0.0) / len(samples)
        # The prior is symmetric, so the posterior weights are the mixture's.
        self.assertAlmostEqual(fraction, 0.7, delta=0.03)
        self.assertEqual(self.result.cold.warnings, ())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            plain = random_walk_metropolis(
                lambda x: _mixture_log_likelihood(x) + _wide_log_prior(x),
                [-5.0],
                proposal_covariance=((0.36,),),
                draws=20000,
                warmup=1000,
                seed=2,
            )
        self.assertEqual(sum(1 for s in plain.chains[0].samples if s[0] > 0.0), 0)

    def test_exchange_statistics(self) -> None:
        """Rejection rates are proper fractions and states make round trips."""
        result = self.result
        self.assertEqual(len(result.swap_rejection_rates), 6)
        self.assertTrue(all(0.0 < r < 0.8 for r in result.swap_rejection_rates))
        for accept, reject in zip(
            result.swap_acceptance_rates, result.swap_rejection_rates, strict=True
        ):
            self.assertAlmostEqual(accept + reject, 1.0, places=14)
        self.assertGreater(result.round_trips, 500)
        self.assertEqual(len(result.local_acceptance_rates), 7)
        self.assertEqual(result.betas, self.betas)
        # mean log-likelihood increases with beta (d log Z / d beta = E[l])
        means = result.mean_log_likelihoods
        self.assertEqual(list(means), sorted(means))
        self.assertGreater(result.evaluations, 7 * 21000)

    def test_stepping_stone_evidence_of_the_mixture(self) -> None:
        """Log Z matches the closed-form marginal likelihood of the mixture."""
        exact = math.log(
            math.exp(-0.5 * 25.0 / 100.25) / math.sqrt(2.0 * math.pi * 100.25)
        )
        self.assertIsNotNone(self.result.log_evidence)
        self.assertAlmostEqual(self.result.log_evidence, exact, delta=0.05)
        self.assertAlmostEqual(
            self.result.log_evidence,
            math.fsum(self.result.log_evidence_increments),
            places=12,
        )

    def test_conjugate_posterior_and_evidence(self) -> None:
        """Normal-normal model: posterior moments and log evidence are exact."""
        y, s = 1.3, 0.7

        def log_likelihood(x):
            """N(y; x, s^2), normalized."""
            return -0.5 * ((y - x[0]) / s) ** 2 - math.log(s) - LOG_SQRT_TWO_PI

        def log_prior(x):
            """N(0, 1), normalized."""
            return -0.5 * x[0] ** 2 - LOG_SQRT_TWO_PI

        result = parallel_tempering(
            log_likelihood,
            log_prior,
            [0.0],
            betas=[0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0],
            draws=20000,
            warmup=500,
            proposal_scale=1.0,
            seed=1,
        )
        variance = s * s / (1.0 + s * s)
        _within_mcse(self, result.cold, (y / (1.0 + s * s),))
        self.assertAlmostEqual(result.cold.covariance()[0][0], variance, delta=0.02)
        exact = -0.5 * y * y / (1.0 + s * s) - 0.5 * math.log(2 * math.pi * (1 + s * s))
        self.assertAlmostEqual(result.log_evidence, exact, delta=0.04)

    def test_flat_likelihood_accepts_every_swap(self) -> None:
        """With a constant likelihood the swap ratio is one."""
        result = parallel_tempering(
            lambda x: 0.0,
            lambda x: -0.5 * x[0] ** 2,
            [0.0],
            betas=[0.2, 0.6, 1.0],
            draws=300,
            seed=3,
            check_convergence=False,
        )
        self.assertEqual(result.swap_rejection_rates, (0.0, 0.0))
        self.assertIsNone(result.log_evidence)
        self.assertEqual(result.log_evidence_increments, (0.0, 0.0))

    def test_custom_kernels_reproduce_the_default(self) -> None:
        """User kernels equal to the default random walk give the same chain."""

        def make(scale):
            """Return a random-walk Metropolis kernel written by the user."""

            def kernel(log_density, state, current, generator):
                """One Metropolis step with an isotropic Gaussian proposal."""
                proposal = tuple(x + scale * generator.gauss(0.0, 1.0) for x in state)
                if math.log(1.0 - generator.random()) < log_density(proposal) - current:
                    return proposal
                return state

            return kernel

        kwargs = dict(betas=[0.0, 0.2, 1.0], draws=400, seed=9, check_convergence=False)
        default = parallel_tempering(
            _mixture_log_likelihood,
            _wide_log_prior,
            [5.0],
            proposal_scale=[5.0, 2.0, 0.5],
            **kwargs,
        )
        custom = parallel_tempering(
            _mixture_log_likelihood,
            _wide_log_prior,
            [5.0],
            kernels=[make(5.0), make(2.0), make(0.5)],
            **kwargs,
        )
        self.assertEqual(default.samples, custom.samples)
        self.assertEqual(default.evaluations, custom.evaluations)
        self.assertEqual(default.swap_rejection_rates, custom.swap_rejection_rates)
        self.assertTrue(math.isnan(custom.cold.chains[0].proposal_covariance[0][0]))
        self.assertEqual(default.cold.chains[0].proposal_covariance[0][0], 0.25)

    def test_validation(self) -> None:
        """Invalid ladders, scales, kernels, and starts raise."""
        args = (_mixture_log_likelihood, _wide_log_prior, [0.0])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            parallel_tempering(*args, betas=[1.0], draws=5)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parallel_tempering(*args, betas=[0.0, 0.5, 0.5, 1.0], draws=5)
        with self.assertRaisesRegex(ValueError, "last beta must be 1"):
            parallel_tempering(*args, betas=[0.0, 0.5], draws=5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parallel_tempering(*args, betas=[-0.1, 1.0], draws=5)
        with self.assertRaisesRegex(ValueError, "one value per temperature"):
            parallel_tempering(*args, betas=[0.0, 1.0], draws=5, proposal_scale=[1.0])
        with self.assertRaisesRegex(ValueError, "proposal_scale must be positive"):
            parallel_tempering(*args, betas=[0.0, 1.0], draws=5, proposal_scale=0.0)
        with self.assertRaisesRegex(ValueError, "one kernel per temperature"):
            parallel_tempering(*args, betas=[0.0, 1.0], draws=5, kernels=[])
        with self.assertRaises(TypeError):
            parallel_tempering(*args, betas=[0.0, 1.0], draws=5, kernels=[1, 2])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "finite log-prior"):
            parallel_tempering(
                _mixture_log_likelihood,
                lambda x: -math.inf,
                [0.0],
                betas=[0.0, 1.0],
                draws=5,
            )
        with self.assertRaisesRegex(ValueError, "one initial point per chain"):
            parallel_tempering(
                _mixture_log_likelihood,
                _wide_log_prior,
                [[0.0], [1.0], [2.0]],
                betas=[0.0, 1.0],
                draws=5,
            )


if __name__ == "__main__":
    unittest.main()
