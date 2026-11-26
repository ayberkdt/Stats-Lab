"""Tests for the S01, M01 and F01 inverse benchmark cases (roadmap 19).

Order inside each family: closed-form cases first, then independent oracles
(explicit loops, dense inverses, Simpson quadrature, closed-form spectra, the
library's Poisson maximum likelihood and linear-Gaussian posterior), then the
noisy and stochastic checks with fixed seeds and Monte Carlo bands, then the
refinement and seed semantics, then adversarial inputs. The expensive coverage
study (1000 replicates) runs only when ``PROBSTAT_SLOW=1``; a 100-replicate
variant always runs.
"""

from __future__ import annotations

import math
import os
import random
import unittest
from collections.abc import Callable

from probstat.inverse import benchmark_catalog as catalog
from probstat.inverse import spectral_diagnostics
from probstat.inverse.benchmark_catalog import (
    CATALOG_BENCHMARK_CASES,
    COVERAGE_SEEDS_FULL,
    COVERAGE_SEEDS_QUICK,
    discrepancy_benchmark,
    discrepancy_coverage_study,
    fredholm_kernel,
    functional_analysis,
    functional_benchmark,
    functional_truth_profile,
    gaussian_conditional_reference,
    poisson_em,
    poisson_image_benchmark,
    poisson_scalar_benchmark,
    poisson_scalar_reference,
    zero_count_upper_limit,
)
from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.inverse.observation_laws import PoissonLaw, maximum_likelihood_linear
from probstat.linalg import (
    adjoint_test,
    inverse_spd,
    materialize,
    singular_value_decomposition,
)


def poisson_cdf(k: int, mean: float) -> float:
    """Return ``P(Poisson(mean) <= k)`` by direct summation.

    References:
        [EQ-INV-BENCH3-002].

    """
    return math.fsum(
        math.exp(-mean + j * math.log(mean) - math.lgamma(j + 1.0))
        for j in range(k + 1)
    )


def poisson_sf(k: int, mean: float) -> float:
    """Return ``P(Poisson(mean) >= k)`` for ``k >= 1``.

    References:
        [EQ-INV-BENCH3-002].

    """
    return 1.0 - poisson_cdf(k - 1, mean)


def simpson(function: object, lower: float, upper: float, panels: int) -> float:
    """Composite Simpson rule (independent of the Gauss rule of the module).

    References:
        [EQ-INV-BENCH3-008].

    """
    call = function  # type: ignore[assignment]
    h = (upper - lower) / panels
    total = call(lower) + call(upper)  # type: ignore[operator]
    for i in range(1, panels):
        total += (4.0 if i % 2 else 2.0) * call(lower + i * h)  # type: ignore[operator]
    return total * h / 3.0


class PoissonScalarTests(unittest.TestCase):
    """S01 exact scalar case: closed-form MLE, Fisher information, zero counts."""

    def test_closed_form_mle_and_information(self) -> None:
        """``y = 7``, ``t = 2.5``: MLE 2.8 and information ``t / x = 0.892857``."""
        ref = poisson_scalar_reference(7, 2.5)
        self.assertAlmostEqual(ref.mle, 2.8, places=14)
        self.assertAlmostEqual(ref.fisher_information or 0.0, 2.5 / 2.8, places=14)
        self.assertAlmostEqual(
            ref.standard_error or 0.0, math.sqrt(2.8 / 2.5), places=14
        )
        expected = 7.0 * math.log(7.0) - 7.0 - math.lgamma(8.0)
        self.assertAlmostEqual(ref.log_likelihood, expected, places=12)
        self.assertFalse(ref.boundary)

    def test_library_poisson_mle_agrees(self) -> None:
        """The library's Fisher scoring finds ``y / t`` and the same information."""
        case = poisson_scalar_benchmark(count=7, exposure=2.5)
        fit = maximum_likelihood_linear(
            case.operator, case.data, PoissonLaw("identity"), initial=[1.0]
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.estimate[0], 2.8, places=8)
        info = PoissonLaw("identity").fisher_information((2.5 * 2.8,))[0] * 2.5**2
        self.assertAlmostEqual(case.reference.fisher_information[0][0], info, places=12)  # type: ignore[index]
        self.assertEqual(case.reference.method, "closed_form")
        self.assertAlmostEqual(
            fit.log_likelihood, case.reference.log_likelihood, places=10
        )

    def test_zero_count_boundary(self) -> None:
        """A zero count has the boundary MLE 0, no information, an exact limit."""
        ref = poisson_scalar_reference(0, 2.0)
        self.assertEqual(ref.mle, 0.0)
        self.assertTrue(ref.boundary)
        self.assertIsNone(ref.fisher_information)
        self.assertIsNone(ref.standard_error)
        limit = ref.zero_count_upper_limit
        self.assertAlmostEqual(limit or 0.0, -math.log(0.05) / 2.0, places=14)
        self.assertAlmostEqual(math.exp(-2.0 * (limit or 0.0)), 0.05, places=14)
        self.assertAlmostEqual(
            zero_count_upper_limit(2.0, 0.9), -math.log(0.1) / 2.0, 14
        )
        lower, upper = ref.garwood_interval(0.95)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, -math.log(0.025) / 2.0, places=8)
        case = poisson_scalar_benchmark(count=0, exposure=2.0)
        self.assertEqual(case.zero_count_indices, (0,))
        self.assertIn("zero_counts", case.pathologies)
        self.assertEqual(case.reference.estimate, (0.0,))
        # The library iteration approaches the boundary without leaving it.
        fit = maximum_likelihood_linear(
            case.operator, case.data, PoissonLaw("identity"), initial=[1.0]
        )
        self.assertLess(abs(fit.estimate[0]), 1e-9)

    def test_zero_intensity_always_gives_zero_count(self) -> None:
        """``x = 0`` has mean count 0 and the sampler returns 0."""
        case = poisson_scalar_benchmark(intensity=0.0, exposure=3.0, seed=5)
        self.assertEqual(case.data, (0.0,))
        self.assertEqual(case.exact_data, (0.0,))

    def test_garwood_interval_is_exact(self) -> None:
        """Garwood limits satisfy the defining Poisson tail equations."""
        ref = poisson_scalar_reference(7, 2.5)
        lower, upper = ref.garwood_interval(0.9)
        self.assertAlmostEqual(poisson_sf(7, 2.5 * lower), 0.05, places=8)
        self.assertAlmostEqual(poisson_cdf(7, 2.5 * upper), 0.05, places=8)

    def test_count_sampling_moments(self) -> None:
        """Counts over 400 seeds have mean and variance ``t x`` within 4 MCSE."""
        mu = 20.0
        counts = [
            poisson_scalar_benchmark(intensity=4.0, exposure=5.0, seed=s).data[0]
            for s in range(400)
        ]
        mean = math.fsum(counts) / len(counts)
        var = math.fsum((c - mean) ** 2 for c in counts) / (len(counts) - 1)
        self.assertLess(abs(mean - mu), 4.0 * math.sqrt(mu / 400))
        # Var(S^2) = mu (2 mu / (N - 1) + 1 / N) for a Poisson sample.
        self.assertLess(abs(var - mu), 4.0 * math.sqrt(mu * (2 * mu / 399 + 1 / 400)))

    def test_scalar_failures(self) -> None:
        """Invalid scalar inputs are rejected."""
        with self.assertRaises(ValueError):
            poisson_scalar_benchmark(intensity=-1.0)
        with self.assertRaises(ValueError):
            poisson_scalar_benchmark(intensity=math.nan)
        with self.assertRaises(ValueError):
            poisson_scalar_benchmark(exposure=0.0)
        with self.assertRaises(ValueError):
            poisson_scalar_benchmark(exposure=math.inf)
        with self.assertRaises(ValueError):
            poisson_scalar_benchmark(count=-2)
        with self.assertRaises(TypeError):
            poisson_scalar_benchmark(count=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            poisson_scalar_benchmark(intensity=True)
        with self.assertRaises(TypeError):
            poisson_scalar_reference(True, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            zero_count_upper_limit(1.0, 1.0)
        with self.assertRaises(ValueError):
            poisson_scalar_reference(3, 1.0).garwood_interval(0.0)


class PoissonImageTests(unittest.TestCase):
    """S01 small synthetic counts: pixelwise, blurred 1D and small 2D."""

    def test_pixelwise_reference_is_exact(self) -> None:
        """Without blur the reference is ``y / t``, KKT-exact, with exact Fisher."""
        case = poisson_image_benchmark(64, width=None, exposure=50.0)
        ref = case.reference
        self.assertEqual(ref.method, "closed_form")
        for k, x in zip(case.data, ref.estimate, strict=True):
            self.assertEqual(x, k / 50.0)
        self.assertLess(ref.kkt_residual, 1e-12)
        assert ref.fisher_information is not None
        for i, x in enumerate(ref.estimate):
            self.assertAlmostEqual(ref.fisher_information[i][i], 50.0 / x, places=9)
        self.assertTrue(case.same_discretization)
        # Pearson statistic of the pixelwise MLE against the truth ~ chi-square(64).
        stat = math.fsum(
            (x - tr) ** 2 / (tr / 50.0)
            for x, tr in zip(ref.estimate, case.truth, strict=True)
        )
        self.assertLess(abs(stat - 64.0), 5.0 * math.sqrt(2.0 * 64.0))

    def test_pixelwise_zero_counts_are_boundary(self) -> None:
        """At low exposure zero counts appear and map to ``x = 0`` exactly."""
        case = poisson_image_benchmark(32, width=None, exposure=1.0, background=0.2)
        self.assertGreater(len(case.zero_count_indices), 0)
        for i in case.zero_count_indices:
            self.assertEqual(case.reference.estimate[i], 0.0)
        self.assertIsNone(case.reference.fisher_information)
        self.assertIn("zero_counts", case.pathologies)
        self.assertLess(case.reference.kkt_residual, 1e-12)

    def test_em_identity_operator_is_one_step(self) -> None:
        """For ``A = I`` the ML-EM update lands on the exact MLE ``y`` at once."""
        counts = (3.0, 0.0, 7.0, 1.0)
        identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        result = poisson_em(
            identity, counts, initial=(2.0, 2.0, 2.0, 2.0), max_iterations=3
        )
        for x, k in zip(result.estimate, counts, strict=True):
            self.assertAlmostEqual(x, k, places=12)
        self.assertEqual(result.estimate[1], 0.0)
        self.assertLess(result.kkt_residual, 1e-12)

    def test_em_likelihood_is_monotone_and_positive(self) -> None:
        """Each ML-EM iterate is positive and the likelihood never decreases."""
        case = poisson_image_benchmark(12, width=0.05, exposure=30.0)
        previous = -math.inf
        for steps in range(1, 8):
            result = poisson_em(case.operator, case.data, max_iterations=steps)
            self.assertGreaterEqual(result.log_likelihood, previous - 1e-12)
            self.assertGreater(min(result.estimate), 0.0)
            previous = result.log_likelihood

    def test_scoring_and_em_agree_on_interior_mle(self) -> None:
        """Library Fisher scoring and ML-EM reach the same interior MLE."""
        scoring = poisson_image_benchmark(16, width=0.03, exposure=50.0)
        em = poisson_image_benchmark(
            16, width=0.03, exposure=50.0, reference_method="em"
        )
        self.assertEqual(scoring.data, em.data)
        self.assertTrue(scoring.reference.converged)
        self.assertTrue(em.reference.converged)
        self.assertTrue(scoring.reference.positive)
        self.assertEqual(scoring.zero_count_indices, ())
        gap = max(
            abs(a - b)
            for a, b in zip(
                scoring.reference.estimate, em.reference.estimate, strict=True
            )
        )
        self.assertLess(gap, 1e-6)
        self.assertLess(scoring.reference.kkt_residual, 1e-8)
        # The MLE maximizes the likelihood: it beats the truth.
        law = PoissonLaw("identity")
        at_truth = law.log_likelihood(
            scoring.data, scoring.operator.matvec(scoring.truth)
        )
        self.assertGreater(scoring.reference.log_likelihood, at_truth)

    def test_fisher_information_matches_explicit_loops(self) -> None:
        """``A' diag(1 / (A x)) A`` equals an explicit triple loop."""
        case = poisson_image_benchmark(10, width=0.05, exposure=40.0)
        dense = materialize(case.operator)
        rates = case.operator.matvec(case.reference.estimate)
        assert case.reference.fisher_information is not None
        for j in range(10):
            for k in range(10):
                expected = 0.0
                for i in range(10):
                    expected += dense[i][j] * dense[i][k] / rates[i]
                self.assertAlmostEqual(
                    case.reference.fisher_information[j][k], expected, places=8
                )

    def test_zero_counts_reveal_positivity_difference(self) -> None:
        """With zero counts, scoring leaves ``x >= 0``; ML-EM stays on the boundary."""
        scoring = poisson_image_benchmark(16, width=0.04, exposure=1.0)
        em = poisson_image_benchmark(
            16, width=0.04, exposure=1.0, reference_method="em"
        )
        self.assertGreater(len(scoring.zero_count_indices), 0)
        self.assertFalse(scoring.reference.positive)
        self.assertLess(scoring.reference.min_estimate, 0.0)
        self.assertTrue(em.reference.positive)
        self.assertGreaterEqual(em.reference.min_estimate, 0.0)
        self.assertLess(em.reference.kkt_residual, 1e-6)
        self.assertGreater(scoring.reference.kkt_residual, 0.1)
        self.assertIn("x >= 0", scoring.positivity)

    def test_all_zero_counts_give_boundary_reference(self) -> None:
        """No counts at all: the constrained MLE is ``x = 0`` and is reported."""
        case = poisson_image_benchmark(8, width=0.04, exposure=1e-9, background=1e-3)
        self.assertEqual(case.data, (0.0,) * 8)
        self.assertEqual(case.reference.method, "boundary_zero")
        self.assertEqual(case.reference.estimate, (0.0,) * 8)
        with self.assertRaises(ValueError):
            poisson_em(case.operator, case.data)

    def test_ridge_reference_is_reported(self) -> None:
        """The penalized reference carries its weight and its true KKT residual."""
        case = poisson_image_benchmark(
            16, width=0.04, exposure=20.0, reference_regularization=0.1
        )
        self.assertEqual(case.reference.regularization, 0.1)
        self.assertTrue(case.reference.converged)
        self.assertGreater(case.reference.kkt_residual, 1e-3)

    def test_two_dimensional_case(self) -> None:
        """The 2D operator has an exact adjoint and the reference converges."""
        case = poisson_image_benchmark(6, dimension=2, width=0.1, exposure=60.0)
        self.assertEqual(len(case.data), 36)
        rng = random.Random(3)
        x = [rng.gauss(0.0, 1.0) for _ in range(36)]
        y = [rng.gauss(0.0, 1.0) for _ in range(36)]
        self.assertLess(adjoint_test(case.operator, x, y).relative_error, 1e-12)
        self.assertTrue(case.reference.converged)
        self.assertGreater(min(case.exact_data), 0.0)
        self.assertGreater(min(case.truth), 0.0)

    def test_refinement_flag_and_discretization_error(self) -> None:
        """Fine-grid mean counts differ from ``A x_true``; refinement 1 does not."""
        crime = poisson_image_benchmark(16, width=0.04, refinement=1)
        fine = poisson_image_benchmark(16, width=0.04, refinement=4)
        self.assertTrue(crime.same_discretization)
        self.assertFalse(fine.same_discretization)
        direct = crime.operator.matvec(crime.truth)
        self.assertEqual(
            max(abs(a - b) for a, b in zip(direct, crime.exact_data, strict=True)), 0.0
        )
        mismatch = max(
            abs(a - b)
            for a, b in zip(
                fine.operator.matvec(fine.truth), fine.exact_data, strict=True
            )
        )
        self.assertGreater(mismatch, 1e-3)
        fine2 = poisson_image_benchmark(6, dimension=2, width=0.1, refinement=2)
        self.assertFalse(fine2.same_discretization)

    def test_seed_separation_and_reproducibility(self) -> None:
        """The truth does not depend on the seed; counts do and reproduce."""
        a = poisson_image_benchmark(16, seed=1)
        b = poisson_image_benchmark(16, seed=1)
        c = poisson_image_benchmark(16, seed=2)
        self.assertEqual(a.data, b.data)
        self.assertNotEqual(a.data, c.data)
        self.assertEqual(a.truth, c.truth)
        self.assertEqual(a.exact_data, c.exact_data)
        self.assertEqual(a.seeds["noise"], "S01/noise")
        total = math.fsum(a.data)
        expected = math.fsum(a.exact_data)
        self.assertLess(abs(total - expected), 5.0 * math.sqrt(expected))

    def test_image_failures(self) -> None:
        """Invalid sizes, parameters and counts are rejected."""
        with self.assertRaises(ValueError):
            poisson_image_benchmark(4)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(3, dimension=2)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, dimension=3)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, exposure=-1.0)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, background=0.0)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, width=-0.1)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, truth="noise")
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, reference_method="newton")
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, refinement=0)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, exposure=math.nan)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16, reference_regularization=-1.0)
        with self.assertRaises(ValueError):
            poisson_image_benchmark(16.0)  # type: ignore[arg-type]

    def test_em_failures(self) -> None:
        """ML-EM rejects invalid counts, starts and inconsistent sizes."""
        a = [[1.0, 0.5], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, -1.0))
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, 2.5))
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, math.nan))
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, 2.0), initial=(1.0, 0.0))
        with self.assertRaises(ValueError):
            poisson_em([[1.0, 0.0], [1.0, 0.0]], (1.0, 1.0))
        with self.assertRaises(ValueError):
            poisson_em(a, (1.0, 2.0), max_iterations=0)
        with self.assertRaises(ValueError):
            poisson_em([[1.0, -2.0], [0.0, 1.0]], (1.0, 1.0), initial=(5.0, 5.0))
        with self.assertRaises(TypeError):
            poisson_em(a, (True, 2.0))  # type: ignore[arg-type]
        # A positive count where the predicted rate is zero.
        with self.assertRaises(ValueError):
            poisson_em([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], (1.0, 2.0, 1.0))


class DiscrepancyBenchmarkTests(unittest.TestCase):
    """M01: exact marginalised conditional and the overconfidence of ignoring delta."""

    def test_scalar_parameter_closed_form(self) -> None:
        """One parameter, two correlated observations: hand-derived conditional."""
        a, b, p, m0 = 0.5, 0.2, 4.0, 0.5
        y = (1.0, 2.0)
        ref = gaussian_conditional_reference(
            [[1.0], [1.0]], y, [[a, b], [b, a]], [m0], [[p]]
        )
        precision = 1.0 / p + 2.0 / (a + b)
        mean = ((y[0] + y[1]) / (a + b) + m0 / p) / precision
        self.assertAlmostEqual(ref.mean[0], mean, places=13)
        self.assertAlmostEqual(ref.covariance[0][0], 1.0 / precision, places=13)
        s11, s12 = a + p, b + p
        det = s11 * s11 - s12 * s12
        r0, r1 = y[0] - m0, y[1] - m0
        quad = (s11 * r0 * r0 - 2.0 * s12 * r0 * r1 + s11 * r1 * r1) / det
        evidence = -0.5 * (quad + math.log(det) + 2.0 * math.log(2.0 * math.pi))
        self.assertAlmostEqual(ref.log_evidence, evidence, places=12)
        lo, hi = ref.interval(0.95)[0]
        self.assertAlmostEqual(
            hi - lo, 2.0 * 1.959963984540054 * math.sqrt(1.0 / precision), 10
        )

    def test_matches_library_linear_gaussian_posterior(self) -> None:
        """The reference equals ``linear_gaussian_posterior`` on the total ``C``."""
        case = discrepancy_benchmark(16)
        post = linear_gaussian_posterior(
            case.operator,
            case.data,
            noise=case.total_covariance,
            prior_mean=case.prior_mean,
            prior_covariance=case.prior_covariance,
        )
        for a, b in zip(post.mean, case.reference.mean, strict=True):
            self.assertAlmostEqual(a, b, places=9)
        self.assertAlmostEqual(post.log_evidence, case.reference.log_evidence, places=8)
        for j in range(2):
            column = post.covariance_action(
                (1.0 if j == 0 else 0.0, 1.0 if j == 1 else 0.0)
            )
            for i in range(2):
                self.assertAlmostEqual(
                    column[i], case.reference.covariance[i][j], places=9
                )

    def test_dense_inverse_oracle(self) -> None:
        """An explicit ``C^-1`` computation reproduces mean and covariance."""
        case = discrepancy_benchmark(12, seed=4)
        c = case.conditional_reference("full")
        n = 12
        dense_c = [
            [
                case.noise_level**2 * 0.6 ** abs(i - j)
                + case.discrepancy_covariance.covariance_operator().matvec(
                    tuple(1.0 if t == j else 0.0 for t in range(n))
                )[i]
                for j in range(n)
            ]
            for i in range(n)
        ]
        cinv = inverse_spd(tuple(tuple(r) for r in dense_c))
        h = [(1.0, x) for x in case.design_points]
        htcinv_h = [
            [
                math.fsum(
                    h[i][a] * cinv[i][k] * h[k][b] for i in range(n) for k in range(n)
                )
                for b in range(2)
            ]
            for a in range(2)
        ]
        precision = [
            [htcinv_h[a][b] + (0.01 if a == b else 0.0) for b in range(2)]
            for a in range(2)
        ]
        cov = inverse_spd(tuple(tuple(r) for r in precision))
        rhs = [
            math.fsum(
                h[i][a] * cinv[i][k] * case.data[k] for i in range(n) for k in range(n)
            )
            for a in range(2)
        ]
        mean = [math.fsum(cov[a][b] * rhs[b] for b in range(2)) for a in range(2)]
        for a in range(2):
            self.assertAlmostEqual(c.mean[a], mean[a], places=8)
            for b in range(2):
                self.assertAlmostEqual(c.covariance[a][b], cov[a][b], places=9)

    def test_ignoring_discrepancy_shrinks_intervals(self) -> None:
        """The misspecified covariance is smaller, so its interval is narrower."""
        case = discrepancy_benchmark(24)
        full = case.conditional_reference("full").standard_deviations()
        naive = case.conditional_reference("no_discrepancy").standard_deviations()
        white = case.conditional_reference("white_noise").standard_deviations()
        for f, nv, w in zip(full, naive, white, strict=True):
            self.assertGreater(f, nv)
            self.assertGreater(nv, w)

    def test_vanishing_discrepancy_recovers_noise_only_model(self) -> None:
        """For ``sigma_delta -> 0`` the full and the no-discrepancy models agree."""
        case = discrepancy_benchmark(16, discrepancy_sd=1e-6)
        full = case.conditional_reference("full")
        naive = case.conditional_reference("no_discrepancy")
        for a, b in zip(full.mean, naive.mean, strict=True):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(
            full.standard_deviations(), naive.standard_deviations(), strict=True
        ):
            self.assertAlmostEqual(a, b, places=6)

    def test_noise_generator_is_ar1(self) -> None:
        """The AR(1) recursion gives variance ``sigma^2`` and lag-1 correlation."""
        lag = 0.0
        var = 0.0
        pairs = 0
        points = 0
        for seed in range(150):
            case = discrepancy_benchmark(12, seed=seed)
            e = case.noise_truth
            lag += math.fsum(e[i] * e[i + 1] for i in range(11))
            var += math.fsum(v * v for v in e)
            pairs += 11
            points += 12
        self.assertLess(abs(var / points - 0.1**2), 0.1**2 * 0.15)
        self.assertLess(abs((lag / pairs) / (var / points) - 0.6), 0.07)

    def test_data_decomposition_and_streams(self) -> None:
        """``y = H theta + delta + e`` exactly, and seeds reproduce or differ."""
        case = discrepancy_benchmark(20, seed=9)
        again = discrepancy_benchmark(20, seed=9)
        other = discrepancy_benchmark(20, seed=10)
        self.assertEqual(case.data, again.data)
        self.assertNotEqual(case.data, other.data)
        self.assertEqual(case.truth, other.truth)
        for i in range(20):
            sim = case.truth[0] + case.truth[1] * case.design_points[i]
            self.assertAlmostEqual(
                case.exact_data[i], sim + case.discrepancy_truth[i], places=13
            )
            self.assertAlmostEqual(
                case.data[i], case.exact_data[i] + case.noise_truth[i], places=13
            )
        self.assertAlmostEqual(
            case.realized_noise_norm, math.hypot(*case.noise_truth), places=13
        )
        self.assertTrue(case.same_discretization)
        self.assertEqual(set(case.seeds), {"truth", "noise"})

    def test_matern_kernel_variant(self) -> None:
        """The Matern 5/2 kernel gives a valid instance with the same API."""
        case = discrepancy_benchmark(16, kernel="matern52")
        self.assertEqual(len(case.reference.mean), 2)
        self.assertTrue(math.isfinite(case.reference.log_evidence))

    def test_quick_coverage_over_predeclared_seeds(self) -> None:
        """100 predeclared seeds: the full model is calibrated, ignoring delta not."""
        self.assertEqual(len(COVERAGE_SEEDS_QUICK), 100)
        study = discrepancy_coverage_study(COVERAGE_SEEDS_QUICK)
        self.assertEqual(study.replicates, 100)
        for covered in study.coverage_full:
            self.assertLess(abs(covered - 0.95), 3.0 * study.mcse)
        for covered in study.coverage_no_discrepancy:
            self.assertLess(covered, 0.95 - 8.0 * study.mcse)
        for covered in study.coverage_white_noise:
            self.assertLess(covered, 0.95 - 8.0 * study.mcse)
        for full, naive in zip(
            study.mean_width_full, study.mean_width_no_discrepancy, strict=True
        ):
            self.assertGreater(full, 2.0 * naive)
        self.assertAlmostEqual(study.mcse, math.sqrt(0.95 * 0.05 / 100), places=14)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_full_coverage_study(self) -> None:
        """1000 predeclared seeds: tighter calibration band for the full model."""
        study = discrepancy_coverage_study(COVERAGE_SEEDS_FULL)
        print(
            f"M01 coverage N={study.replicates} full={study.coverage_full} "
            f"no_delta={study.coverage_no_discrepancy} "
            f"white={study.coverage_white_noise} mcse={study.mcse:.4f}"
        )
        for covered in study.coverage_full:
            self.assertLess(abs(covered - 0.95), 3.0 * study.mcse)
        for covered in study.coverage_no_discrepancy:
            self.assertLess(covered, 0.7)

    def test_discrepancy_failures(self) -> None:
        """Invalid sizes, parameters and unknown models are rejected."""
        with self.assertRaises(ValueError):
            discrepancy_benchmark(3)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(discrepancy_sd=0.0)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(discrepancy_length=-1.0)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(noise_sd=math.nan)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(noise_correlation=1.0)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(noise_correlation=-0.1)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(prior_sd=0.0)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(kernel="rbf")
        with self.assertRaises(ValueError):
            discrepancy_benchmark(theta=(1.0,))
        with self.assertRaises(ValueError):
            discrepancy_benchmark(discrepancy_nugget=-1.0)
        with self.assertRaises(TypeError):
            discrepancy_benchmark(noise_sd=True)
        with self.assertRaises(ValueError):
            discrepancy_benchmark(24, discrepancy_nugget=0.0, discrepancy_length=0.5)
        case = discrepancy_benchmark(8)
        with self.assertRaises(ValueError):
            case.conditional_reference("oracle")
        with self.assertRaises(ValueError):
            gaussian_conditional_reference(
                [[1.0]], (1.0, 2.0), [[1.0]], (0.0,), [[1.0]]
            )
        with self.assertRaises(ValueError):
            gaussian_conditional_reference([[1.0]], (1.0,), [[-1.0]], (0.0,), [[1.0]])
        with self.assertRaises(ValueError):
            gaussian_conditional_reference(
                [[1.0]], (math.nan,), [[1.0]], (0.0,), [[1.0]]
            )
        with self.assertRaises(ValueError):
            case.conditional_reference("full").interval(1.0)
        with self.assertRaises(ValueError):
            discrepancy_coverage_study(())
        with self.assertRaises(ValueError):
            discrepancy_coverage_study((1, 1))
        with self.assertRaises(TypeError):
            discrepancy_coverage_study((1.5,))  # type: ignore[arg-type]


class FunctionalBenchmarkTests(unittest.TestCase):
    """F01: Fredholm kernel, data-informed modes and averaging kernels."""

    def test_brownian_operator_closed_form(self) -> None:
        """``A_ij = int_{cell j} min(s_i, t) dt`` in closed form, to rounding."""
        n, m = 16, 12
        case = functional_benchmark(n, m, kernel="brownian", fine_data=False)
        dense = materialize(case.operator)
        for i in range(m):
            s = (i + 0.5) / m
            for j in range(n):
                a, b = j / n, (j + 1) / n
                if s >= b:
                    expected = 0.5 * (b * b - a * a)
                elif s <= a:
                    expected = s * (b - a)
                else:
                    expected = 0.5 * (s * s - a * a) + s * (b - s)
                self.assertAlmostEqual(dense[i][j], expected, places=13)
        ones = case.operator.matvec((1.0,) * n)
        for i in range(m):
            s = (i + 0.5) / m
            self.assertAlmostEqual(ones[i], s - 0.5 * s * s, places=13)

    def test_functional_row_and_truth(self) -> None:
        """``q`` is the exact window overlap, sums to one and averages constants."""
        case = functional_benchmark(32, 16, window_centre=0.4, window_halfwidth=0.1)
        lower, upper = case.window
        self.assertAlmostEqual(lower, 0.3, places=14)
        self.assertAlmostEqual(math.fsum(case.functional), 1.0, places=13)
        for j, value in enumerate(case.functional):
            overlap = max(0.0, min((j + 1) / 32, upper) - max(j / 32, lower))
            self.assertAlmostEqual(value, overlap / (upper - lower), places=14)
        profile = functional_truth_profile("smooth")
        reference = simpson(profile, lower, upper, 2000) / (upper - lower)
        self.assertAlmostEqual(case.functional_truth, reference, places=9)
        cell_average = [
            simpson(profile, j / 32, (j + 1) / 32, 200) * 32 for j in range(32)
        ]
        for a, b in zip(case.truth, cell_average, strict=True):
            self.assertAlmostEqual(a, b, places=10)

    def test_fine_data_matches_independent_simpson(self) -> None:
        """The data are the true integrals (Simpson split at the kink)."""
        for name in ("gaussian", "brownian"):
            case = functional_benchmark(16, 8, kernel=name, fine_data=True)
            kernel = fredholm_kernel(name, width=0.1)
            profile = functional_truth_profile("smooth")
            for s, b in zip(case.observation_points, case.exact_data, strict=True):

                def integrand(
                    t: float,
                    s: float = s,
                    kernel: Callable[[float, float], float] = kernel,
                    profile: Callable[[float], float] = profile,
                ) -> float:
                    return kernel(s, t) * profile(t)

                reference = simpson(integrand, 0.0, s, 4000) + simpson(
                    integrand, s, 1.0, 4000
                )
                self.assertAlmostEqual(b, reference, places=9)

    def test_inverse_crime_flag(self) -> None:
        """Fine data differ from ``A x_true``; the crime variant equals it."""
        fine = functional_benchmark(16, 12, fine_data=True)
        crime = functional_benchmark(16, 12, fine_data=False)
        self.assertFalse(fine.same_discretization)
        self.assertTrue(crime.same_discretization)
        gap = max(
            abs(a - b)
            for a, b in zip(
                fine.operator.matvec(fine.truth), fine.exact_data, strict=True
            )
        )
        self.assertGreater(gap, 1e-6)
        self.assertEqual(crime.exact_data, tuple(crime.operator.matvec(crime.truth)))

    def test_brownian_singular_values_match_continuum(self) -> None:
        """The leading singular values approach ``1 / ((k - 1/2)^2 pi^2)``."""
        case = functional_benchmark(64, 64, kernel="brownian", fine_data=False)
        assert case.singular_values is not None
        for k in range(4):
            exact = 1.0 / ((k + 0.5) ** 2 * math.pi**2)
            self.assertLess(abs(case.singular_values[k] / exact - 1.0), 0.01)

    def test_full_rank_functional_is_recovered_exactly(self) -> None:
        """Square non-singular case without noise: ``kappa = q`` and ``Q = q'x``."""
        case = functional_benchmark(32, 32, kernel="brownian", fine_data=False)
        result = functional_analysis(
            case, data=case.exact_data, truth=case.truth, rank=32
        )
        self.assertAlmostEqual(result.resolved_fraction, 1.0, places=10)
        self.assertAlmostEqual(result.estimate, case.functional_truth_coarse, places=7)
        self.assertLess(abs(result.bias_truth or 0.0), 1e-9)
        for a, b in zip(result.averaging_kernel, case.functional, strict=True):
            self.assertAlmostEqual(a, b, places=8)

    def test_averaging_kernel_against_dense_svd_oracle(self) -> None:
        """``kappa = V_k V_k' q`` from an independent dense SVD, and the weights."""
        case = functional_benchmark(24, 12, kernel="gaussian")
        result = functional_analysis(case, rank=5)
        svd = singular_value_decomposition(materialize(case.operator))
        n = 24
        q = case.functional
        kappa = [0.0] * n
        for i in range(5):
            v = [svd.v[j][i] for j in range(n)]
            c = math.fsum(q[j] * v[j] for j in range(n))
            for j in range(n):
                kappa[j] += c * v[j]
        for a, b in zip(result.averaging_kernel, kappa, strict=True):
            self.assertAlmostEqual(a, b, places=9)
        # Noise-free linearity: kappa' x equals the TSVD of noise-free data.
        truth = case.truth
        noise_free = tuple(case.operator.matvec(truth))
        clean = functional_analysis(case, rank=5, data=noise_free)
        expected = math.fsum(a * b for a, b in zip(kappa, truth, strict=True))
        self.assertAlmostEqual(clean.estimate, expected, places=9)
        for a, b in zip(svd.singular_values, result.singular_values, strict=False):
            self.assertAlmostEqual(a, b, places=10)

    def test_informed_modes_follow_spectral_diagnostics(self) -> None:
        """Data-informed modes are the leading singular values above the noise."""
        case = functional_benchmark(32, 16, kernel="gaussian", relative_noise=1e-2)
        result = functional_analysis(case)
        flags = spectral_diagnostics(case.operator, noise=case.noise).detectable(1.0)
        leading = 0
        for flag in flags:
            if not flag:
                break
            leading += 1
        self.assertEqual(result.informed_modes, leading)
        self.assertEqual(result.rank, leading)
        self.assertGreater(leading, 3)
        self.assertLess(leading, 16)
        quiet = functional_benchmark(32, 16, kernel="gaussian", relative_noise=1e-4)
        self.assertGreater(functional_analysis(quiet).informed_modes, leading)

    def test_estimator_is_linear_with_exact_standard_error(self) -> None:
        """``Q = a'y`` and its Monte Carlo spread equals ``s ||a||``."""
        case = functional_benchmark(32, 16, kernel="gaussian")
        result = functional_analysis(case)
        direct = math.fsum(
            a * b for a, b in zip(result.estimator_weights, case.data, strict=True)
        )
        self.assertAlmostEqual(result.estimate, direct, places=12)
        self.assertAlmostEqual(
            result.standard_error,
            case.noise_level * math.hypot(*result.estimator_weights),
            places=12,
        )
        # The estimate is linear in the data: a few full analyses confirm that
        # ``Q(y) = a'y`` for fresh random data (each analysis needs one SVD).
        rng = random.Random(2024)
        for _ in range(3):
            noisy = tuple(
                b + case.noise_level * rng.gauss(0.0, 1.0) for b in case.exact_data
            )
            again = functional_analysis(case, data=noisy, rank=result.rank).estimate
            linear = math.fsum(
                a * b for a, b in zip(result.estimator_weights, noisy, strict=True)
            )
            self.assertAlmostEqual(again, linear, places=12)
        # 2000 noise draws pushed through the linear weights: sd = s ||a||.
        draws = []
        for _ in range(2000):
            noisy = [
                b + case.noise_level * rng.gauss(0.0, 1.0) for b in case.exact_data
            ]
            draws.append(
                math.fsum(
                    a * b for a, b in zip(result.estimator_weights, noisy, strict=True)
                )
            )
        mean = math.fsum(draws) / len(draws)
        sd = math.sqrt(math.fsum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
        self.assertLess(
            abs(sd / result.standard_error - 1.0), 4.0 / math.sqrt(2.0 * 2000)
        )

    def test_tradeoff_is_monotone(self) -> None:
        """More modes resolve more of ``q`` and amplify more noise."""
        case = functional_benchmark(32, 16, kernel="gaussian")
        result = functional_analysis(case, truth=case.truth)
        previous_fraction = 0.0
        previous_error = 0.0
        for row in result.tradeoff:
            self.assertGreaterEqual(row.resolved_fraction, previous_fraction - 1e-12)
            self.assertGreaterEqual(row.standard_error, previous_error)
            self.assertIsNotNone(row.bias_truth)
            previous_fraction = row.resolved_fraction
            previous_error = row.standard_error
        self.assertEqual(result.tradeoff[result.rank - 1].rank, result.rank)
        self.assertLessEqual(result.resolved_fraction, 1.0 + 1e-12)

    def test_truth_is_not_an_input_of_the_estimate(self) -> None:
        """Passing the truth only adds the bias; the estimate is unchanged."""
        case = functional_benchmark(32, 16)
        plain = functional_analysis(case)
        with_truth = functional_analysis(case, truth=case.truth)
        self.assertIsNone(plain.bias_truth)
        self.assertIsNotNone(with_truth.bias_truth)
        self.assertEqual(plain.estimate, with_truth.estimate)
        self.assertEqual(plain.averaging_kernel, with_truth.averaging_kernel)

    def test_unresolved_narrow_feature_has_bias(self) -> None:
        """A narrow bump leaves part of the functional in the null space."""
        case = functional_benchmark(32, 16, kernel="gaussian", truth="bump")
        result = functional_analysis(case, truth=case.truth)
        self.assertLess(result.resolved_fraction, 0.99)
        self.assertGreater(abs(result.bias_truth or 0.0), 3.0 * result.standard_error)
        self.assertLess(abs(result.kernel_centre - 0.4), 0.15)
        self.assertIn("unresolved_functional", case.pathologies)

    def test_seed_semantics(self) -> None:
        """The truth is analytic; only the noise depends on the seed."""
        a = functional_benchmark(seed=1)
        b = functional_benchmark(seed=1)
        c = functional_benchmark(seed=2)
        self.assertEqual(a.data, b.data)
        self.assertNotEqual(a.data, c.data)
        self.assertEqual(a.truth, c.truth)
        self.assertEqual(a.exact_data, c.exact_data)
        self.assertEqual(a.seeds["noise"], "F01/noise")
        self.assertAlmostEqual(
            a.realized_noise_norm,
            math.hypot(*(y - b_ for y, b_ in zip(a.data, a.exact_data, strict=True))),
            places=13,
        )

    def test_functional_failures(self) -> None:
        """Invalid sizes, windows, names and ranks are rejected."""
        with self.assertRaises(ValueError):
            functional_benchmark(4)
        with self.assertRaises(ValueError):
            functional_benchmark(16, 1)
        with self.assertRaises(ValueError):
            functional_benchmark(kernel="sinc")
        with self.assertRaises(ValueError):
            functional_benchmark(truth="step")
        with self.assertRaises(ValueError):
            functional_benchmark(window_centre=0.95, window_halfwidth=0.1)
        with self.assertRaises(ValueError):
            functional_benchmark(window_halfwidth=0.0)
        with self.assertRaises(ValueError):
            functional_benchmark(relative_noise=0.0)
        with self.assertRaises(ValueError):
            functional_benchmark(kernel_width=-1.0)
        with self.assertRaises(ValueError):
            functional_benchmark(fine_panels=2)
        with self.assertRaises(ValueError):
            functional_benchmark(32.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            functional_benchmark(window_centre=math.nan)
        case = functional_benchmark(16, 8)
        with self.assertRaises(ValueError):
            functional_analysis(case, rank=0)
        with self.assertRaises(ValueError):
            functional_analysis(case, rank=100)
        with self.assertRaises(ValueError):
            functional_analysis(case, threshold=0.0)
        with self.assertRaises(ValueError):
            functional_analysis(case, data=(1.0, 2.0))
        with self.assertRaises(ValueError):
            functional_analysis(case, truth=(1.0,))
        with self.assertRaises(ValueError):
            functional_analysis(case, data=(math.nan,) * 8)
        hopeless = functional_benchmark(16, 8, relative_noise=1e6)
        with self.assertRaises(ValueError):
            functional_analysis(hopeless)
        with self.assertRaises(ValueError):
            fredholm_kernel("gaussian", width=0.0)
        with self.assertRaises(ValueError):
            functional_truth_profile("other")


class CatalogEntryTests(unittest.TestCase):
    """The catalog lists S01, M01 and F01 with resolvable builders."""

    def test_entries_resolve(self) -> None:
        """Every entry names an existing builder in this module."""
        self.assertEqual(
            [c.identifier for c in CATALOG_BENCHMARK_CASES], ["S01", "M01", "F01"]
        )
        for entry in CATALOG_BENCHMARK_CASES:
            name = entry.builder.rsplit(".", 1)[1]
            self.assertTrue(callable(getattr(catalog, name)))
            self.assertTrue(entry.reference)
            self.assertTrue(entry.methods)

    def test_identifiers_of_instances(self) -> None:
        """Instances carry their catalog identifiers."""
        self.assertEqual(poisson_scalar_benchmark().identifier, "S01")
        self.assertEqual(poisson_image_benchmark(8).identifier, "S01")
        self.assertEqual(discrepancy_benchmark(8).identifier, "M01")
        self.assertEqual(functional_benchmark(8, 4).identifier, "F01")


if __name__ == "__main__":
    unittest.main()
