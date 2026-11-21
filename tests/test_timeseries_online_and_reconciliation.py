"""Tests for Bayesian online change-point detection and forecast reconciliation."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.linalg import cholesky, householder_qr, matmul, matvec
from probstat.timeseries import (
    GammaPoissonPrior,
    HierarchyStructure,
    NormalInverseGammaPrior,
    bayesian_online_changepoint,
    hierarchy_summing_matrix,
    reconcile_forecasts,
    shrink_covariance,
)


def _log_sum_exp(values: list[float]) -> float:
    """Return the log of a sum of exponentials."""
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _enumerated_evidence(
    data: list[float],
    hazard: float,
    prior: NormalInverseGammaPrior | GammaPoissonPrior,
) -> float:
    """Sum the product-partition evidence over all ``2^(T-1)`` segmentations."""
    terms: list[float] = []
    for cuts in itertools.product((False, True), repeat=len(data) - 1):
        value = 0.0
        start = 0
        for index, cut in enumerate(cuts):
            value += math.log(hazard) if cut else math.log(1.0 - hazard)
            if cut:
                value += prior.log_marginal_likelihood(data[start : index + 1])
                start = index + 1
        value += prior.log_marginal_likelihood(data[start:])
        terms.append(value)
    return _log_sum_exp(terms)


class ConjugateModelTests(unittest.TestCase):
    """Closed-form evidences against numerical integration."""

    def test_poisson_evidence_matches_quadrature(self) -> None:
        """The Gamma-Poisson evidence equals the integral over the rate."""
        prior = GammaPoissonPrior(shape=2.5, rate=0.8)
        counts = [3, 0, 5, 2]

        def integrand(rate: float) -> float:
            """Return prior density times Poisson likelihood."""
            log_value = (
                prior.shape * math.log(prior.rate)
                - math.lgamma(prior.shape)
                + (prior.shape - 1.0) * math.log(rate)
                - prior.rate * rate
            )
            for count in counts:
                log_value += count * math.log(rate) - rate - math.lgamma(count + 1.0)
            return math.exp(log_value)

        steps = 20000
        width = 40.0 / steps
        total = width * sum(integrand((k + 0.5) * width) for k in range(steps))
        self.assertAlmostEqual(
            prior.log_marginal_likelihood(counts), math.log(total), places=8
        )

    def test_gaussian_evidence_matches_quadrature(self) -> None:
        """The normal-inverse-gamma evidence equals a two-dimensional integral."""
        prior = NormalInverseGammaPrior(
            mean=0.3, precision_scale=1.5, shape=2.0, scale=1.2
        )
        data = [0.9, -0.4]
        grid = 400
        total = 0.0
        # Integrate over mu and s = log(sigma^2) by the midpoint rule.
        for i in range(grid):
            s = -6.0 + 12.0 * (i + 0.5) / grid
            variance = math.exp(s)
            log_prior_variance = (
                prior.shape * math.log(prior.scale)
                - math.lgamma(prior.shape)
                - (prior.shape + 1.0) * s
                - prior.scale / variance
                + s
            )
            inner = 0.0
            for j in range(grid):
                mu = -12.0 + 24.0 * (j + 0.5) / grid
                log_value = -0.5 * math.log(
                    2.0 * math.pi * variance / prior.precision_scale
                ) - prior.precision_scale * (mu - prior.mean) ** 2 / (2.0 * variance)
                for value in data:
                    log_value += -0.5 * math.log(2.0 * math.pi * variance) - (
                        value - mu
                    ) ** 2 / (2.0 * variance)
                inner += math.exp(log_value)
            total += math.exp(log_prior_variance) * inner
        total *= (12.0 / grid) * (24.0 / grid)
        self.assertAlmostEqual(
            prior.log_marginal_likelihood(data), math.log(total), places=5
        )

    def test_prior_predictive_is_the_one_point_evidence(self) -> None:
        """The predictive of the prior equals the evidence of one observation."""
        gaussian = NormalInverseGammaPrior(1.0, 0.5, 3.0, 2.0)
        self.assertAlmostEqual(
            gaussian.log_predictive(0.7),
            gaussian.log_marginal_likelihood([0.7]),
            places=12,
        )
        poisson = GammaPoissonPrior(1.7, 2.2)
        self.assertAlmostEqual(
            poisson.log_predictive(4), poisson.log_marginal_likelihood([4]), places=12
        )
        self.assertAlmostEqual(
            sum(math.exp(poisson.log_predictive(k)) for k in range(200)), 1.0, places=12
        )
        self.assertEqual(gaussian.log_marginal_likelihood([]), 0.0)
        self.assertEqual(poisson.log_marginal_likelihood([]), 0.0)

    def test_prior_validation(self) -> None:
        """Invalid hyperparameters and counts are rejected."""
        with self.assertRaises(ValueError):
            NormalInverseGammaPrior(precision_scale=0.0)
        with self.assertRaises(ValueError):
            NormalInverseGammaPrior(scale=-1.0)
        with self.assertRaises(ValueError):
            NormalInverseGammaPrior(mean=math.nan)
        with self.assertRaises(ValueError):
            GammaPoissonPrior(shape=0.0)
        with self.assertRaises(ValueError):
            GammaPoissonPrior().log_predictive(1.5)
        with self.assertRaises(ValueError):
            GammaPoissonPrior().log_marginal_likelihood([2, -1])
        with self.assertRaises(TypeError):
            GammaPoissonPrior().log_predictive("3")  # type: ignore[arg-type]


class OnlineChangePointTests(unittest.TestCase):
    """Run-length recursion against exact oracles."""

    def setUp(self) -> None:
        """Create a short Gaussian and a short count series with a shift."""
        generator = random.Random(1)
        self.gaussian = [generator.gauss(0.0, 1.0) for _ in range(5)] + [
            generator.gauss(4.0, 1.0) for _ in range(5)
        ]
        self.counts = [3, 2, 4, 1, 12, 15, 9, 11]

    def test_evidence_equals_enumeration_gaussian(self) -> None:
        """The Gaussian evidence equals the sum over all 512 segmentations."""
        prior = NormalInverseGammaPrior(0.5, 2.0, 1.5, 0.7)
        for hazard in (0.05, 0.2, 0.7):
            result = bayesian_online_changepoint(
                self.gaussian, hazard=hazard, prior=prior
            )
            self.assertAlmostEqual(
                result.log_evidence,
                _enumerated_evidence(self.gaussian, hazard, prior),
                places=10,
            )

    def test_evidence_equals_enumeration_poisson(self) -> None:
        """The Poisson evidence equals the sum over all 128 segmentations."""
        prior = GammaPoissonPrior(2.0, 0.5)
        result = bayesian_online_changepoint(self.counts, hazard=0.3, prior=prior)
        self.assertAlmostEqual(
            result.log_evidence,
            _enumerated_evidence([float(c) for c in self.counts], 0.3, prior),
            places=10,
        )

    def test_evidence_path_matches_enumeration_of_prefixes(self) -> None:
        """Every prefix evidence equals the enumeration of that prefix."""
        prior = NormalInverseGammaPrior()
        result = bayesian_online_changepoint(self.gaussian, hazard=0.25, prior=prior)
        for length in (1, 2, 5, 8):
            self.assertAlmostEqual(
                result.log_evidence_path[length - 1],
                _enumerated_evidence(self.gaussian[:length], 0.25, prior),
                places=10,
            )

    def test_zero_hazard_is_conjugate_updating(self) -> None:
        """With ``H = 0`` the run never resets and the evidence is closed-form."""
        for data, prior in (
            (self.gaussian, NormalInverseGammaPrior(0.5, 2.0, 1.5, 0.7)),
            (self.counts, GammaPoissonPrior(2.0, 0.5)),
        ):
            result = bayesian_online_changepoint(data, hazard=0.0, prior=prior)
            self.assertAlmostEqual(
                result.log_evidence, prior.log_marginal_likelihood(data), places=10
            )
            self.assertEqual(result.map_run_length, tuple(range(1, len(data) + 1)))
            for time, row in enumerate(result.run_length_posterior):
                self.assertEqual(row[time + 1], 1.0)
                self.assertEqual(sum(row), 1.0)

    def test_unit_hazard_gives_independent_prior_predictives(self) -> None:
        """With ``H = 1`` every observation is its own segment."""
        prior = GammaPoissonPrior(2.0, 0.5)
        result = bayesian_online_changepoint(self.counts, hazard=1.0, prior=prior)
        self.assertAlmostEqual(
            result.log_evidence,
            sum(prior.log_predictive(count) for count in self.counts),
            places=10,
        )
        self.assertEqual(set(result.map_run_length), {0})

    def test_posterior_rows_are_distributions_and_reset_mass_is_the_hazard(
        self,
    ) -> None:
        """Rows sum to one and ``P(r_t = 0 | x_1:t)`` equals the hazard."""
        result = bayesian_online_changepoint(self.gaussian, hazard=0.2)
        for time, row in enumerate(result.run_length_posterior):
            self.assertEqual(len(row), time + 2)
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertAlmostEqual(row[0], 0.2, places=12)
            self.assertAlmostEqual(
                result.expected_run_length[time],
                sum(k * p for k, p in enumerate(row)),
                places=12,
            )
        self.assertAlmostEqual(result.segment_start_probability(1)[0], 0.8, places=12)
        self.assertEqual(result.predictive_evaluations, 10 * 11 // 2)
        self.assertEqual(result.run_length_probability(3, 99), 0.0)
        self.assertEqual(
            result.run_length_probability(3, 2), result.run_length_posterior[3][2]
        )

    def test_known_mean_shift_is_detected(self) -> None:
        """A shift of three standard deviations at index 150 is located."""
        generator = random.Random(2007)
        series = [generator.gauss(0.0, 1.0) for _ in range(150)] + [
            generator.gauss(3.0, 1.0) for _ in range(150)
        ]
        result = bayesian_online_changepoint(series, hazard=0.01)
        self.assertEqual(result.map_run_length[149], 150)
        self.assertLess(result.map_run_length[159], 15)
        starts = result.map_change_points()
        self.assertTrue(starts)
        self.assertTrue(all(abs(start - 150) <= 3 for start in starts))
        probability = result.segment_start_probability(delay=10)
        best = max(range(1, 290), key=probability.__getitem__)
        self.assertLessEqual(abs(best - 150), 1)
        self.assertGreater(probability[best], 0.3)
        self.assertTrue(math.isnan(probability[-1]))
        self.assertEqual(result.model, "gaussian")
        self.assertFalse(result.truncated)

    def test_poisson_rate_shift_is_detected(self) -> None:
        """A rate change from 2 to 9 at index 80 is located."""
        generator = random.Random(11)

        def poisson(rate: float) -> int:
            """Draw a Poisson count by multiplying uniforms."""
            limit, product, count = math.exp(-rate), generator.random(), 0
            while product > limit:
                product *= generator.random()
                count += 1
            return count

        series = [poisson(2.0) for _ in range(80)] + [poisson(9.0) for _ in range(80)]
        result = bayesian_online_changepoint(
            series, hazard=0.02, prior=GammaPoissonPrior(1.0, 0.25)
        )
        self.assertEqual(result.model, "poisson")
        starts = result.map_change_points()
        self.assertTrue(any(abs(start - 80) <= 3 for start in starts))
        self.assertLess(result.map_run_length[-1], 85)
        self.assertGreater(result.map_run_length[-1], 70)

    def test_truncation_is_a_lower_bound_and_exact_when_wide(self) -> None:
        """Truncated evidence never exceeds the exact one and matches for large R."""
        exact = bayesian_online_changepoint(self.gaussian, hazard=0.2)
        wide = bayesian_online_changepoint(
            self.gaussian, hazard=0.2, max_run_length=len(self.gaussian)
        )
        self.assertEqual(wide.log_evidence, exact.log_evidence)
        self.assertFalse(wide.truncated)
        self.assertEqual(wide.discarded_mass, 0.0)
        narrow = bayesian_online_changepoint(
            self.gaussian, hazard=0.2, max_run_length=3
        )
        self.assertTrue(narrow.truncated)
        self.assertGreater(narrow.discarded_mass, 0.0)
        self.assertLess(narrow.log_evidence, exact.log_evidence)
        self.assertTrue(all(len(row) <= 4 for row in narrow.run_length_posterior))
        self.assertLess(narrow.predictive_evaluations, exact.predictive_evaluations)
        for row in narrow.run_length_posterior:
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_input_validation(self) -> None:
        """Bad series, hazards, priors, and bounds raise informative errors."""
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([], hazard=0.1)
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([], hazard=0.1, prior=GammaPoissonPrior())
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.0, math.inf], hazard=0.1)
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.0], hazard=1.5)
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.0], hazard=-0.1)
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.5], hazard=0.1, prior=GammaPoissonPrior())
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.0], hazard=0.1, max_run_length=0)
        with self.assertRaises(TypeError):
            bayesian_online_changepoint([1.0], hazard=0.1, max_run_length=2.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bayesian_online_changepoint([1.0], hazard=0.1, prior="gaussian")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bayesian_online_changepoint([1.0, 2.0], hazard=0.0, max_run_length=1)
        result = bayesian_online_changepoint([1.0, 2.0], hazard=0.1)
        with self.assertRaises(ValueError):
            result.segment_start_probability(0)
        with self.assertRaises(ValueError):
            result.run_length_probability(2, 0)


def _random_spd(size: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return a seeded symmetric positive-definite matrix."""
    generator = random.Random(seed)
    factor = [[generator.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return tuple(
        tuple(
            sum(factor[i][k] * factor[j][k] for k in range(size))
            + (0.5 if i == j else 0.0)
            for j in range(size)
        )
        for i in range(size)
    )


class HierarchyTests(unittest.TestCase):
    """Summing matrices from hierarchy specifications."""

    def test_two_level_hierarchy(self) -> None:
        """A tree gives one indicator row per series."""
        structure = hierarchy_summing_matrix(
            {"total": ["A", "B"], "A": ["AA", "AB"], "B": ["BA", "BB", "BC"]}
        )
        self.assertEqual(
            structure.names, ("total", "A", "B", "AA", "AB", "BA", "BB", "BC")
        )
        self.assertEqual(structure.bottom_names, ("AA", "AB", "BA", "BB", "BC"))
        self.assertEqual(structure.matrix[0], (1.0,) * 5)
        self.assertEqual(structure.matrix[1], (1.0, 1.0, 0.0, 0.0, 0.0))
        self.assertEqual(structure.matrix[2], (0.0, 0.0, 1.0, 1.0, 1.0))
        self.assertEqual(structure.matrix[5], (0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(
            structure.aggregate([1.0, 2.0, 3.0, 4.0, 5.0]),
            (15.0, 3.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        )

    def test_grouped_structure(self) -> None:
        """Two crossed groupings share the same bottom series."""
        structure = hierarchy_summing_matrix(
            {
                "total": ["a", "b", "c", "d"],
                "north": ["a", "b"],
                "south": ["c", "d"],
                "new": ["a", "c"],
                "old": ["b", "d"],
            }
        )
        self.assertEqual(len(structure.matrix), 9)
        self.assertEqual(structure.matrix[3], (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(structure.matrix[4], (0.0, 1.0, 0.0, 1.0))

    def test_invalid_specifications(self) -> None:
        """Empty, cyclic, and double-counting specifications are rejected."""
        with self.assertRaises(ValueError):
            hierarchy_summing_matrix({})
        with self.assertRaises(ValueError):
            hierarchy_summing_matrix({"total": []})
        with self.assertRaises(ValueError):
            hierarchy_summing_matrix({"total": ["a", "a"]})
        with self.assertRaises(ValueError):
            hierarchy_summing_matrix({"a": ["b", "x"], "b": ["a"]})
        with self.assertRaises(ValueError):
            hierarchy_summing_matrix({"total": ["A", "x"], "A": ["x", "y"]})
        with self.assertRaises(TypeError):
            hierarchy_summing_matrix({"total": [1, 2]})  # type: ignore[list-item]
        structure = hierarchy_summing_matrix({"total": ["a", "b"]})
        with self.assertRaises(ValueError):
            structure.aggregate([1.0])


class ReconciliationTests(unittest.TestCase):
    """Projection properties and optimality of the reconcilers."""

    def setUp(self) -> None:
        """Create a hierarchy, a random covariance, and base forecasts."""
        self.structure = hierarchy_summing_matrix(
            {"total": ["A", "B"], "A": ["AA", "AB"], "B": ["BA", "BB", "BC"]}
        )
        self.size = len(self.structure.names)
        self.covariance = _random_spd(self.size, 3)
        generator = random.Random(4)
        self.base = [generator.gauss(10.0, 3.0) for _ in range(self.size)]
        self.options: dict[str, dict[str, object]] = {
            "ols": {},
            "bottom_up": {},
            "wls": {"variances": [self.covariance[i][i] for i in range(self.size)]},
            "mint": {"covariance": self.covariance},
        }

    def _result(self, method: str, base: list[float] | None = None):  # type: ignore[no-untyped-def]
        """Reconcile the stored (or given) base forecasts with one method."""
        return reconcile_forecasts(
            self.base if base is None else base,
            self.structure,
            method=method,
            **self.options[method],  # type: ignore[arg-type]
        )

    def test_reconciled_forecasts_are_coherent(self) -> None:
        """Aggregating the reconciled bottom level reproduces every series."""
        for method in self.options:
            result = self._result(method)
            self.assertEqual(result.reconciled, self.structure.aggregate(result.bottom))
            total, a, b, aa, ab, ba, bb, bc = result.reconciled
            self.assertAlmostEqual(total, a + b, places=10)
            self.assertAlmostEqual(a, aa + ab, places=10)
            self.assertAlmostEqual(b, ba + bb + bc, places=10)
            self.assertGreater(result.base_incoherence, 1.0)
            self.assertEqual(result.method, method)

    def test_coherent_base_forecasts_are_unchanged(self) -> None:
        """Every method returns coherent base forecasts unchanged."""
        coherent = list(self.structure.aggregate([1.0, 2.0, 3.0, 4.0, 5.0]))
        for method in self.options:
            result = self._result(method, coherent)
            for new, old in zip(result.reconciled, coherent, strict=True):
                self.assertAlmostEqual(new, old, places=10)
            self.assertLess(result.base_incoherence, 1e-10)
            self.assertLess(max(abs(v) for v in result.adjustment), 1e-10)

    def test_projection_is_idempotent_and_fixes_the_summing_matrix(self) -> None:
        """``(S G)^2 = S G`` and ``G S = I`` for every method."""
        for method in self.options:
            result = self._result(method)
            projector = result.projection()
            squared = matmul(projector, projector)
            for row, expected in zip(squared, projector, strict=True):
                for value, target in zip(row, expected, strict=True):
                    self.assertAlmostEqual(value, target, places=9)
            identity = matmul(result.mapping, self.structure.matrix)
            for i, row in enumerate(identity):
                for j, value in enumerate(row):
                    self.assertAlmostEqual(value, 1.0 if i == j else 0.0, places=9)
            again = self._result(method, list(result.reconciled))
            for new, old in zip(again.reconciled, result.reconciled, strict=True):
                self.assertAlmostEqual(new, old, places=9)

    def test_ols_equals_least_squares_by_qr(self) -> None:
        """``W = I`` reproduces the Householder least-squares fit on ``S``."""
        result = self._result("ols")
        bottom = householder_qr(self.structure.matrix).solve(tuple(self.base))
        for value, expected in zip(result.bottom, bottom, strict=True):
            self.assertAlmostEqual(value, expected, places=10)
        fitted = matvec(self.structure.matrix, bottom)
        for value, expected in zip(result.reconciled, fitted, strict=True):
            self.assertAlmostEqual(value, expected, places=10)
        residual = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.base, fitted, strict=True))
        )
        self.assertAlmostEqual(result.base_incoherence, residual, places=10)

    def test_wls_with_equal_variances_and_mint_with_diagonal_agree(self) -> None:
        """Equal variances give OLS, and a diagonal MinT covariance gives WLS."""
        ols = self._result("ols")
        equal = reconcile_forecasts(
            self.base, self.structure, method="wls", variances=[2.5] * self.size
        )
        for value, expected in zip(equal.reconciled, ols.reconciled, strict=True):
            self.assertAlmostEqual(value, expected, places=10)
        variances = [self.covariance[i][i] for i in range(self.size)]
        diagonal = [
            [variances[i] if i == j else 0.0 for j in range(self.size)]
            for i in range(self.size)
        ]
        wls = self._result("wls")
        mint = reconcile_forecasts(
            self.base, self.structure, method="mint", covariance=diagonal
        )
        for value, expected in zip(mint.reconciled, wls.reconciled, strict=True):
            self.assertAlmostEqual(value, expected, places=10)

    def test_bottom_up_uses_only_the_bottom_level(self) -> None:
        """Bottom-up returns the bottom base forecasts and their sums."""
        result = self._result("bottom_up")
        self.assertEqual(result.bottom, tuple(self.base[3:]))
        self.assertAlmostEqual(result.reconciled[0], sum(self.base[3:]), places=12)

    def test_mint_minimizes_the_trace_for_the_true_covariance(self) -> None:
        """MinT has the smallest ``tr(S G W G' S')`` among the reconcilers."""
        for seed in (3, 17, 99):
            covariance = _random_spd(self.size, seed)
            options: dict[str, dict[str, object]] = {
                "ols": {},
                "bottom_up": {},
                "wls": {"variances": [covariance[i][i] for i in range(self.size)]},
                "mint": {"covariance": covariance},
            }
            traces = {
                method: reconcile_forecasts(
                    self.base,
                    self.structure,
                    method=method,
                    **keywords,  # type: ignore[arg-type]
                ).total_error_variance(covariance)
                for method, keywords in options.items()
            }
            for method in ("ols", "bottom_up", "wls"):
                self.assertLess(traces["mint"], traces[method])
        # Random unbiased competitors G = G_mint + Z (I - S G_ols) also lose.
        mint = self._result("mint")
        ols = self._result("ols")
        generator = random.Random(8)
        complement = tuple(
            tuple((1.0 if i == j else 0.0) - value for j, value in enumerate(row))
            for i, row in enumerate(ols.projection())
        )
        best = mint.total_error_variance(self.covariance)
        for _ in range(20):
            noise = tuple(
                tuple(generator.gauss(0.0, 0.3) for _ in range(self.size))
                for _ in range(5)
            )
            shift = matmul(noise, complement)
            mapping = tuple(
                tuple(a + b for a, b in zip(row, extra, strict=True))
                for row, extra in zip(mint.mapping, shift, strict=True)
            )
            identity = matmul(mapping, self.structure.matrix)
            self.assertAlmostEqual(identity[0][0], 1.0, places=9)
            projector = matmul(self.structure.matrix, mapping)
            spread = matmul(
                matmul(projector, self.covariance),
                tuple(zip(*projector, strict=True)),
            )
            self.assertGreater(sum(spread[i][i] for i in range(self.size)), best)

    def test_simulated_squared_error_ranks_mint_first(self) -> None:
        """With known error covariance MinT has the smallest squared error."""
        factor = cholesky(self.covariance)
        generator = random.Random(2019)
        truth = self.structure.aggregate([4.0, 6.0, 3.0, 5.0, 2.0])
        replicates = 600
        errors: dict[str, list[float]] = {name: [] for name in self.options}
        errors["base"] = []
        for _ in range(replicates):
            normal = [generator.gauss(0.0, 1.0) for _ in range(self.size)]
            base = [
                truth[i] + sum(factor[i][k] * normal[k] for k in range(i + 1))
                for i in range(self.size)
            ]
            errors["base"].append(
                sum((a - b) ** 2 for a, b in zip(base, truth, strict=True))
            )
            for method in self.options:
                reconciled = self._result(method, base).reconciled
                errors[method].append(
                    sum((a - b) ** 2 for a, b in zip(reconciled, truth, strict=True))
                )
        means = {name: sum(values) / replicates for name, values in errors.items()}
        for name in ("ols", "wls", "bottom_up", "base"):
            self.assertLess(means["mint"], means[name])
        for method in self.options:
            expected = self._result(method).total_error_variance(self.covariance)
            deviation = math.sqrt(
                sum((v - means[method]) ** 2 for v in errors[method])
                / (replicates - 1)
                / replicates
            )
            self.assertLess(abs(means[method] - expected), 4.0 * deviation)

    def test_plain_matrix_input_and_validation(self) -> None:
        """A raw summing matrix works and malformed input is rejected."""
        matrix = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        result = reconcile_forecasts([10.0, 3.0, 4.0], matrix)
        # Hand calculation: subtract the component along (1, -1, -1), which
        # is (10 - 3 - 4) / 3 = 1 per coordinate.
        self.assertAlmostEqual(result.reconciled[0], 9.0, places=12)
        self.assertAlmostEqual(result.base_incoherence, math.sqrt(3.0), places=12)
        self.assertAlmostEqual(result.reconciled[1], 4.0, places=12)
        self.assertAlmostEqual(result.reconciled[2], 5.0, places=12)
        self.assertIsInstance(self.structure, HierarchyStructure)
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0], matrix)
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0, 4.0], matrix, method="top_down")
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0, 4.0], matrix, method="wls")
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0, 4.0], matrix, variances=[1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                [10.0, 3.0, 4.0], matrix, method="wls", variances=[1.0, 0.0, 1.0]
            )
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                [10.0, 3.0, 4.0], matrix, method="wls", variances=[1.0, 1.0]
            )
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0, 4.0], matrix, method="mint")
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                [10.0, 3.0, 4.0],
                matrix,
                method="mint",
                covariance=[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                [10.0, 3.0, 4.0], matrix, method="mint", covariance=[[1.0]]
            )
        with self.assertRaises(ValueError):
            reconcile_forecasts([10.0, 3.0, 4.0], [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                [10.0, 3.0, 4.0],
                [[1.0, 1.0], [1.0, 0.0], [1.0, 2.0]],
                method="bottom_up",
            )
        with self.assertRaises(ValueError):
            reconcile_forecasts([1.0], [[1.0, 1.0]])
        with self.assertRaises(ValueError):
            result.error_covariance([[1.0]])


class ShrinkageTests(unittest.TestCase):
    """Shrinkage of the base-error covariance towards its diagonal."""

    @staticmethod
    def _sample(
        size: int, count: int, rho: float, generator: random.Random
    ) -> list[list[float]]:
        """Draw correlated Gaussian residuals with AR(1)-type correlation."""
        covariance = tuple(
            tuple(rho ** abs(i - j) * (1 + i) * (1 + j) / 4.0 for j in range(size))
            for i in range(size)
        )
        factor = cholesky(covariance)
        rows = []
        for _ in range(count):
            normal = [generator.gauss(0.0, 1.0) for _ in range(size)]
            rows.append(
                [
                    sum(factor[i][k] * normal[k] for k in range(i + 1))
                    for i in range(size)
                ]
            )
        return rows

    @staticmethod
    def _correlation_loss(matrix: tuple[tuple[float, ...], ...], rho: float) -> float:
        """Return the squared error of the off-diagonal correlations."""
        size = len(matrix)
        return sum(
            (matrix[i][j] / math.sqrt(matrix[i][i] * matrix[j][j]) - rho ** abs(i - j))
            ** 2
            for i in range(size)
            for j in range(size)
            if i != j
        )

    def test_fixed_intensity_interpolates(self) -> None:
        """Intensity zero is the sample covariance and one its diagonal."""
        rows = self._sample(4, 15, 0.5, random.Random(1))
        sample = shrink_covariance(rows, intensity=0.0)
        diagonal = shrink_covariance(rows, intensity=1.0)
        half = shrink_covariance(rows, intensity=0.5)
        means = [sum(row[j] for row in rows) / 15 for j in range(4)]
        for i in range(4):
            for j in range(4):
                expected = (
                    sum((row[i] - means[i]) * (row[j] - means[j]) for row in rows) / 14
                )
                self.assertAlmostEqual(sample.covariance[i][j], expected, places=12)
                self.assertAlmostEqual(
                    diagonal.covariance[i][j], expected if i == j else 0.0, places=12
                )
                self.assertAlmostEqual(
                    half.covariance[i][j],
                    expected if i == j else 0.5 * expected,
                    places=12,
                )
        self.assertFalse(sample.estimated)
        self.assertEqual(sample.sample_size, 15)

    def test_estimated_intensity_tracks_the_optimal_fixed_intensity(self) -> None:
        """The plug-in intensity is near the risk-optimal one and reduces risk."""
        generator = random.Random(6)
        size, count, rho, replicates = 8, 12, 0.3, 150
        grid = [k / 10 for k in range(11)]
        risk = [0.0] * len(grid)
        estimated_risk = 0.0
        intensities = []
        for _ in range(replicates):
            rows = self._sample(size, count, rho, generator)
            fitted = shrink_covariance(rows)
            self.assertTrue(fitted.estimated)
            self.assertTrue(0.0 <= fitted.intensity <= 1.0)
            intensities.append(fitted.intensity)
            estimated_risk += self._correlation_loss(fitted.covariance, rho)
            for index, value in enumerate(grid):
                risk[index] += self._correlation_loss(
                    shrink_covariance(rows, intensity=value).covariance, rho
                )
        best = grid[min(range(len(grid)), key=risk.__getitem__)]
        # Derived optimum: sum Var(r) / sum E[r^2] with E[r^2] = Var + rho^2.
        variance_sum = risk[0] / replicates
        signal = sum(
            rho ** (2 * abs(i - j)) for i in range(size) for j in range(size) if i != j
        )
        derived = variance_sum / (variance_sum + signal)
        self.assertLess(abs(derived - best), 0.06)
        self.assertLess(abs(sum(intensities) / replicates - derived), 0.1)
        self.assertLess(estimated_risk, 0.4 * risk[0])
        self.assertLess(estimated_risk, 1.4 * min(risk))

    def test_intensity_is_the_schafer_strimmer_estimator(self) -> None:
        """The plug-in equals the published correlation-shrinkage intensity.

        Literal evaluation for centred residuals standardized with the
        divisor-``N`` deviation: ``sum_(i != j) v_ij / sum_(i != j) r_ij^2``
        with ``v_ij = sum_k (w_kij - w_bar_ij)^2 / (N (N - 1))`` and
        ``r_ij = w_bar_ij``. Uncentred residuals give a different value.
        """

        def literal(rows: list[list[float]], *, center: bool) -> float:
            """Evaluate the intensity from its definition."""
            count, width = len(rows), len(rows[0])
            if center:
                means = [sum(row[j] for row in rows) / count for j in range(width)]
                rows = [[row[j] - means[j] for j in range(width)] for row in rows]
            scales = [
                math.sqrt(sum(row[j] ** 2 for row in rows) / count)
                for j in range(width)
            ]
            unit = [[row[j] / scales[j] for j in range(width)] for row in rows]
            top = bottom = 0.0
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    products = [row[i] * row[j] for row in unit]
                    mean = sum(products) / count
                    top += sum((w - mean) ** 2 for w in products) / (
                        count * (count - 1)
                    )
                    bottom += mean * mean
            return min(1.0, max(0.0, top / bottom))

        rng = random.Random(2005)
        for count, width in ((12, 8), (20, 6), (200, 5), (3, 4)):
            common = [rng.gauss(0.0, 1.0) for _ in range(count)]
            rows = [
                [rng.gauss(0.3, 1.0) + 0.5 * common[k] for _ in range(width)]
                for k in range(count)
            ]
            with self.subTest(count=count, width=width):
                fitted = shrink_covariance(rows).intensity
                self.assertAlmostEqual(fitted, literal(rows, center=True), delta=1e-12)
                if count <= 20:
                    self.assertGreater(abs(fitted - literal(rows, center=False)), 1e-3)

    def test_intensity_vanishes_with_many_observations(self) -> None:
        """With a long residual history hardly any shrinkage is applied."""
        rows = self._sample(4, 2000, 0.5, random.Random(9))
        self.assertLess(shrink_covariance(rows).intensity, 0.02)

    def test_intensity_is_scale_invariant_and_makes_mint_feasible(self) -> None:
        """Rescaling a series keeps the intensity; n < p still reconciles."""
        structure = hierarchy_summing_matrix(
            {"total": ["A", "B"], "A": ["AA", "AB"], "B": ["BA", "BB", "BC"]}
        )
        rows = self._sample(8, 6, 0.4, random.Random(12))
        fitted = shrink_covariance(rows)
        scaled = shrink_covariance([[row[0] * 100.0, *row[1:]] for row in rows])
        self.assertAlmostEqual(fitted.intensity, scaled.intensity, places=10)
        self.assertGreater(fitted.intensity, 0.0)
        base = [20.0, 9.0, 12.0, 4.0, 6.0, 3.0, 5.0, 2.0]
        result = reconcile_forecasts(
            base, structure, method="mint", covariance=fitted.covariance
        )
        self.assertEqual(result.reconciled, structure.aggregate(result.bottom))
        with self.assertRaises(ValueError):
            reconcile_forecasts(
                base,
                structure,
                method="mint",
                covariance=shrink_covariance(rows, intensity=0.0).covariance,
            )

    def test_validation(self) -> None:
        """Too few rows, constant series, and bad intensities are rejected."""
        with self.assertRaises(ValueError):
            shrink_covariance([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            shrink_covariance([[1.0, 2.0], [1.0, 3.0]])
        with self.assertRaises(ValueError):
            shrink_covariance([[1.0, 2.0], [2.0, 3.0], [0.0, 1.0]], intensity=1.5)
        with self.assertRaises(ValueError):
            shrink_covariance([[1.0, 2.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
