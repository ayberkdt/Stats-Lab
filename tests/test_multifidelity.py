"""Tests for multifidelity Monte Carlo, approximate control variates, and MLBLUE.

The oracle is a family of jointly Gaussian "models" ``Y_i = mu_i + (L z)_i``
built from a known covariance ``C = L L'``, so every predicted variance is
exact and every estimator with fixed weights is exactly Gaussian. Independent
checks: an exact variance calculator for linear combinations of sample means
over index sets, a brute-force generalized least squares solve, closed forms,
and seeded replicate simulations.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import unittest
from collections.abc import Callable, Sequence

from probstat.foundations.random_streams import RandomStream
from probstat.linalg import cholesky
from probstat.monte_carlo import (
    ACVDesign,
    MFMCAllocation,
    MultifidelityEstimate,
    acv_allocation,
    acv_design,
    acv_estimate,
    compare_multifidelity_estimators,
    mfmc_admissible,
    mfmc_allocation,
    mfmc_estimate,
    mfmc_variance,
    mlblue_allocation,
    mlblue_estimate,
    mlblue_groups,
    mlblue_variance,
    pilot_statistics,
    select_mfmc_models,
)

Matrix = list[list[float]]
Term = tuple[float, int, frozenset[tuple[str, int]]]

MEANS = (1.0, -0.5, 2.0, 0.25)
SIGMAS = (1.0, 1.2, 0.8, 2.0)
CORRELATIONS = (
    (1.0, 0.95, 0.85, 0.6),
    (0.95, 1.0, 0.9, 0.7),
    (0.85, 0.9, 1.0, 0.8),
    (0.6, 0.7, 0.8, 1.0),
)
COSTS = (1.0, 0.1, 0.01, 0.001)


def covariance_of(size: int = 4) -> Matrix:
    """Return the leading ``size`` block of the reference covariance."""
    return [
        [SIGMAS[i] * SIGMAS[j] * CORRELATIONS[i][j] for j in range(size)]
        for i in range(size)
    ]


def gaussian_models(
    covariance: Sequence[Sequence[float]], means: Sequence[float]
) -> list[Callable[[tuple[float, ...]], float]]:
    """Return linear maps of shared standard normals with a known covariance."""
    lower = cholesky(tuple(tuple(row) for row in covariance))

    def make(index: int) -> Callable[[tuple[float, ...]], float]:
        """Return model ``index``."""
        row = lower[index]
        shift = means[index]

        def model(z: tuple[float, ...]) -> float:
            """Evaluate ``mu_i + (L z)_i``."""
            return shift + math.fsum(a * b for a, b in zip(row, z, strict=False))

        return model

    return [make(index) for index in range(len(means))]


def normal_inputs(size: int) -> Callable[[int, RandomStream], list[tuple[float, ...]]]:
    """Return a sampler of ``size``-dimensional standard normal inputs."""

    def sampler(count: int, stream: RandomStream) -> list[tuple[float, ...]]:
        """Draw ``count`` inputs from ``stream``."""
        generator = stream.generator()
        return [
            tuple(generator.gauss(0.0, 1.0) for _ in range(size)) for _ in range(count)
        ]

    return sampler


def pool(label: str, start: int, stop: int) -> frozenset[tuple[str, int]]:
    """Return the index set ``{(label, start), ..., (label, stop - 1)}``."""
    return frozenset((label, index) for index in range(start, stop))


def exact_variance(
    terms: Sequence[Term], covariance: Sequence[Sequence[float]]
) -> float:
    """Return the variance of ``sum_a c_a mean_{A_a}(Y_{i_a})`` exactly.

    Uses ``Cov(mean_A Y_i, mean_B Y_j) = C_ij |A & B| / (|A| |B|)`` for
    independent identically distributed inputs.
    """
    total = 0.0
    for coefficient_a, model_a, set_a in terms:
        for coefficient_b, model_b, set_b in terms:
            overlap = len(set_a & set_b)
            if overlap:
                total += (
                    coefficient_a
                    * coefficient_b
                    * covariance[model_a][model_b]
                    * overlap
                    / (len(set_a) * len(set_b))
                )
    return total


def mfmc_terms(allocation: MFMCAllocation) -> list[Term]:
    """Write an MFMC estimator as weighted sample means over index sets."""
    terms: list[Term] = [(1.0, 0, pool("x", 0, allocation.samples[0]))]
    for position in range(1, len(allocation.models)):
        a = allocation.coefficients[position]
        index = allocation.models[position]
        terms.append((a, index, pool("x", 0, allocation.samples[position])))
        terms.append((-a, index, pool("x", 0, allocation.samples[position - 1])))
    return terms


def acv_terms(design: ACVDesign, weights: Sequence[float] | None = None) -> list[Term]:
    """Write an ACV estimator as weighted sample means over index sets."""
    base = design.samples[0]
    alpha = design.weights if weights is None else weights
    terms: list[Term] = [(1.0, 0, pool("x", 0, base))]
    for index in range(1, len(design.samples)):
        count = design.samples[index]
        if count == 0:
            continue
        if design.kind == "MF":
            full = pool("x", 0, count)
        else:
            full = pool("x", 0, base) | pool(f"extra-{index}", 0, count - base)
        terms.append((alpha[index - 1], index, pool("x", 0, base)))
        terms.append((-alpha[index - 1], index, full))
    return terms


def gauss_jordan_inverse(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Invert a matrix by Gauss-Jordan elimination with partial pivoting."""
    size = len(matrix)
    work = [
        list(row) + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(work[r][column]))
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row != column and work[row][column] != 0.0:
                factor = work[row][column]
                work[row] = [
                    a - factor * b for a, b in zip(work[row], work[column], strict=True)
                ]
    return [row[size:] for row in work]


def stacked_gls(
    covariance: Sequence[Sequence[float]],
    groups: Sequence[Sequence[int]],
    samples: Sequence[int],
    observations: Sequence[float] | None = None,
) -> tuple[Matrix, list[float]]:
    """Return ``(H' S^-1 H)^-1`` and the GLS solution for stacked observations.

    Every scalar model output is one row of the design matrix ``H`` (a unit
    vector) and ``S`` is the full observation covariance, block diagonal with
    one block ``C_k`` per group sample.
    """
    size = len(covariance)
    rows: list[tuple[int, int, int]] = []
    for k, (group, count) in enumerate(zip(groups, samples, strict=True)):
        for sample in range(count):
            rows.extend((k, sample, index) for index in group)
    total = len(rows)
    big = [[0.0] * total for _ in range(total)]
    for a, (group_a, sample_a, model_a) in enumerate(rows):
        for b, (group_b, sample_b, model_b) in enumerate(rows):
            if group_a == group_b and sample_a == sample_b:
                big[a][b] = covariance[model_a][model_b]
    precision = gauss_jordan_inverse(big)
    normal = [[0.0] * size for _ in range(size)]
    right = [0.0] * size
    for a, (_, _, model_a) in enumerate(rows):
        for b, (_, _, model_b) in enumerate(rows):
            normal[model_a][model_b] += precision[a][b]
            if observations is not None:
                right[model_a] += precision[a][b] * observations[b]
    inverse = gauss_jordan_inverse(normal)
    solution = [
        math.fsum(inverse[i][j] * right[j] for j in range(size)) for i in range(size)
    ]
    return inverse, solution


def replicate(
    run: Callable[[int], MultifidelityEstimate], count: int
) -> tuple[float, float]:
    """Return the mean and unbiased variance of ``count`` seeded estimates."""
    values = [run(seed).estimate for seed in range(count)]
    return statistics.fmean(values), statistics.variance(values)


def random_covariance(generator: random.Random, size: int) -> Matrix:
    """Return a random well-conditioned covariance with correlated models."""
    shared = [generator.gauss(0.0, 1.0) for _ in range(size + 1)]
    loadings = [
        [shared[m] + 0.5 * generator.gauss(0.0, 1.0) for m in range(size + 1)]
        for _ in range(size)
    ]
    return [
        [
            math.fsum(a * b for a, b in zip(loadings[i], loadings[j], strict=True))
            for j in range(size)
        ]
        for i in range(size)
    ]


class ReplicateAssertions(unittest.TestCase):
    """Shared assertions for replicate simulations of Gaussian estimators."""

    def assert_matches_prediction(
        self, mean: float, variance: float, count: int, truth: float, predicted: float
    ) -> None:
        """Check unbiasedness (4 SE) and the variance (chi-square-type band).

        For Gaussian estimates ``(R - 1) S^2 / sigma^2`` is chi-square with
        ``R - 1`` degrees of freedom, so ``S^2 / sigma^2`` has standard
        deviation ``sqrt(2 / (R - 1))``; the band is 4.5 of those.
        """
        self.assertLess(abs(mean - truth), 4.0 * math.sqrt(predicted / count))
        band = 4.5 * math.sqrt(2.0 / (count - 1))
        self.assertLess(abs(variance / predicted - 1.0), band)


class PilotStatisticsTests(unittest.TestCase):
    """Joint pilot statistics of several models."""

    def test_matches_the_standard_library_statistics(self) -> None:
        """Means, covariance, and correlation equal ``statistics`` results."""
        models = [lambda u: math.exp(u), lambda u: 1.0 + u, lambda u: u * u]

        def sampler(count: int, stream: RandomStream) -> list[float]:
            """Draw uniform inputs."""
            generator = stream.generator()
            return [generator.random() for _ in range(count)]

        pilot = pilot_statistics(models, sampler, 200, seed=5)
        inputs = sampler(200, RandomStream(5).child("pilot"))
        outputs = [[model(u) for u in inputs] for model in models]
        for i in range(3):
            self.assertAlmostEqual(pilot.means[i], statistics.fmean(outputs[i]))
            for j in range(3):
                self.assertAlmostEqual(
                    pilot.covariance[i][j],
                    statistics.covariance(outputs[i], outputs[j]),
                    places=12,
                )
                self.assertAlmostEqual(
                    pilot.correlation[i][j],
                    statistics.correlation(outputs[i], outputs[j]),
                    places=12,
                )
        self.assertEqual(pilot.evaluations, (200, 200, 200))
        self.assertAlmostEqual(pilot.cost([1.0, 0.1, 0.01]), 222.0)
        self.assertAlmostEqual(
            pilot.standard_deviations[1], statistics.stdev(outputs[1])
        )

    def test_is_reproducible_and_separate_from_the_estimator_stream(self) -> None:
        """One seed gives one pilot; the estimator draws other inputs."""
        seen: dict[str, list[float]] = {}

        def sampler(count: int, stream: RandomStream) -> list[float]:
            """Draw normal inputs and record them by call order."""
            generator = stream.generator()
            values = [generator.gauss(0.0, 1.0) for _ in range(count)]
            seen[f"call-{len(seen)}"] = values
            return values

        models = [lambda z: z, lambda z: 0.9 * z + 0.3 * z * z]
        first = pilot_statistics(models, sampler, 30, seed=11)
        second = pilot_statistics(models, sampler, 30, seed=11)
        self.assertEqual(first, second)
        allocation = mfmc_allocation(first.covariance, [1.0, 0.01], 20.0)
        mfmc_estimate(models, sampler, allocation, seed=11)
        self.assertEqual(seen["call-0"], seen["call-1"])
        self.assertTrue(set(seen["call-0"]).isdisjoint(seen["call-2"]))

    def test_rejects_invalid_input(self) -> None:
        """Empty model lists, tiny samples, and bad outputs are refused."""
        sampler = normal_inputs(1)
        with self.assertRaisesRegex(ValueError, "at least one model"):
            pilot_statistics([], sampler, 10, seed=1)
        with self.assertRaisesRegex(ValueError, "samples"):
            pilot_statistics([lambda z: z[0]], sampler, 1, seed=1)
        with self.assertRaisesRegex(ValueError, "output of model 0"):
            pilot_statistics([lambda z: math.nan], sampler, 5, seed=1)
        with self.assertRaisesRegex(ValueError, "returned 3 inputs"):
            pilot_statistics([lambda z: 1.0], lambda n, s: [0.0] * 3, 5, seed=1)
        constant = pilot_statistics([lambda z: 1.0], sampler, 5, seed=1)
        with self.assertRaisesRegex(ValueError, "zero sample variance"):
            _ = constant.correlation


class MFMCAllocationTests(unittest.TestCase):
    """Closed-form MFMC allocation, admissibility, and model selection."""

    def test_closed_form_allocation_by_hand(self) -> None:
        """Ratios, sample sizes, and coefficients equal the stated formulas."""
        covariance = covariance_of(3)
        plan = mfmc_allocation(covariance, COSTS[:3], 100.0)
        r1 = math.sqrt(1.0 * (0.95**2 - 0.85**2) / (0.1 * (1.0 - 0.95**2)))
        r2 = math.sqrt(1.0 * 0.85**2 / (0.01 * (1.0 - 0.95**2)))
        m0 = 100.0 / (1.0 + 0.1 * r1 + 0.01 * r2)
        self.assertEqual(plan.models, (0, 1, 2))
        for got, want in zip(plan.ratios, (1.0, r1, r2), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(
            plan.continuous_samples, (m0, r1 * m0, r2 * m0), strict=True
        ):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(plan.coefficients[1], 0.95 * 1.0 / 1.2, places=12)
        self.assertAlmostEqual(plan.coefficients[2], 0.85 * 1.0 / 0.8, places=12)
        self.assertEqual(plan.samples, tuple(math.floor(m0 * r) for r in (1.0, r1, r2)))
        self.assertLessEqual(plan.cost, 100.0)
        self.assertEqual(plan.evaluations, plan.samples)
        self.assertAlmostEqual(plan.mc_variance, 1.0 / 100.0)

    def test_closed_form_variance_equals_the_general_formula(self) -> None:
        """The optimal-variance closed form agrees with the lemma formula."""
        covariance = covariance_of(4)
        plan = mfmc_allocation(covariance, COSTS, 250.0)
        general = mfmc_variance(covariance, plan.continuous_samples, models=plan.models)
        self.assertAlmostEqual(plan.continuous_variance / general, 1.0, places=12)
        self.assertGreaterEqual(plan.variance, plan.continuous_variance)
        self.assertLess(plan.variance_ratio, 0.3)
        spent = math.fsum(
            COSTS[i] * m
            for i, m in zip(plan.models, plan.continuous_samples, strict=True)
        )
        self.assertAlmostEqual(spent, 250.0, places=9)

    def test_variance_formula_equals_the_index_set_oracle(self) -> None:
        """The MFMC variance matches exact covariances of nested means."""
        covariance = covariance_of(4)
        plan = mfmc_allocation(covariance, COSTS, 40.0)
        self.assertAlmostEqual(
            plan.variance / exact_variance(mfmc_terms(plan), covariance), 1.0, places=12
        )
        arbitrary = mfmc_variance(
            covariance, [5, 9, 9, 30], coefficients=[1.0, 0.4, -0.3, 0.2]
        )
        terms: list[Term] = [(1.0, 0, pool("x", 0, 5))]
        for index, (a, low, high) in enumerate(
            [(0.4, 5, 9), (-0.3, 9, 9), (0.2, 9, 30)], start=1
        ):
            terms.append((a, index, pool("x", 0, high)))
            terms.append((-a, index, pool("x", 0, low)))
        self.assertAlmostEqual(
            arbitrary / exact_variance(terms, covariance), 1.0, places=12
        )

    def test_allocation_is_a_constrained_minimum(self) -> None:
        """Feasible perturbations of samples or coefficients do not help."""
        covariance = covariance_of(4)
        plan = mfmc_allocation(covariance, COSTS, 500.0)
        generator = random.Random(4)
        for _ in range(200):
            trial = [
                m * math.exp(0.2 * generator.gauss(0.0, 1.0))
                for m in plan.continuous_samples
            ]
            trial.sort()
            spent = math.fsum(w * m for w, m in zip(COSTS, trial, strict=True))
            trial = [m * 500.0 / spent for m in trial]
            self.assertGreaterEqual(
                mfmc_variance(covariance, trial), plan.continuous_variance * (1 - 1e-12)
            )
            coefficients = [
                a + 0.1 * generator.gauss(0.0, 1.0) for a in plan.coefficients
            ]
            self.assertGreaterEqual(
                mfmc_variance(
                    covariance, plan.continuous_samples, coefficients=coefficients
                ),
                plan.continuous_variance * (1 - 1e-12),
            )

    def test_two_model_case_tends_to_the_control_variate_limit(self) -> None:
        """``Var = s^2 (1 - rho^2 (1 - n/m)) / n -> (1 - rho^2) s^2 / n``."""
        rho, s0, s1, n = 0.9, 2.0, 0.5, 20
        covariance = [[s0 * s0, rho * s0 * s1], [rho * s0 * s1, s1 * s1]]
        for m in (20, 40, 400, 10**9):
            self.assertAlmostEqual(
                mfmc_variance(covariance, [n, m]),
                s0 * s0 * (1.0 - rho * rho * (1.0 - n / m)) / n,
                places=12,
            )
        limit = (1.0 - rho * rho) * s0 * s0 / n
        self.assertAlmostEqual(mfmc_variance(covariance, [n, 10**12]), limit, places=9)
        self.assertAlmostEqual(mfmc_variance(covariance, [n, n]), s0 * s0 / n)

    def test_admissibility_conditions(self) -> None:
        """Ordering and cost-ratio violations are detected and reported."""
        covariance = covariance_of(4)
        self.assertTrue(mfmc_admissible(covariance, COSTS))
        self.assertTrue(mfmc_admissible(covariance, COSTS, models=[0]))
        self.assertTrue(mfmc_admissible(covariance, COSTS, models=[0, 3, 1]))
        expensive = (1.0, 0.9, 0.01, 0.001)
        # (rho_0^2 - rho_1^2) / (rho_1^2 - rho_2^2) = 0.0975 / 0.18 = 0.54 < 1/0.9
        self.assertTrue(mfmc_admissible(covariance, expensive))
        flat = (1.0, 0.1, 0.09, 0.001)
        # w_1 / w_2 = 1.11 but (0.9025 - 0.7225) / (0.7225 - 0.36) = 0.497: fine;
        # w_0 / w_1 = 10 > 0.54: fine.
        self.assertTrue(mfmc_admissible(covariance, flat))
        costly_low = (1.0, 0.1, 0.01, 0.02)
        # w_2 / w_3 = 0.5 is not above (0.7225 - 0.36) / 0.36 = 1.007
        self.assertFalse(mfmc_admissible(covariance, costly_low))
        with self.assertRaisesRegex(ValueError, "not admissible"):
            mfmc_allocation(covariance, costly_low, 100.0)
        tied = [[1.0, 0.5, 0.5], [0.5, 1.0, 0.2], [0.5, 0.2, 1.0]]
        self.assertFalse(mfmc_admissible(tied, (1.0, 0.1, 0.01)))
        uncorrelated = [[1.0, 0.0], [0.0, 1.0]]
        self.assertFalse(mfmc_admissible(uncorrelated, (1.0, 0.1)))

    def test_selection_equals_brute_force_over_admissible_subsets(self) -> None:
        """The selected subset minimizes the closed-form variance."""
        generator = random.Random(8)
        dropped = 0
        for _ in range(25):
            size = generator.choice([3, 4, 5])
            covariance = random_covariance(generator, size)
            costs = [10.0 ** (-i * generator.uniform(0.2, 1.2)) for i in range(size)]
            best = select_mfmc_models(covariance, costs, 300.0)
            rho2 = [
                covariance[0][i] ** 2 / (covariance[0][0] * covariance[i][i])
                for i in range(size)
            ]
            values = {}
            for extra in range(size):
                for chosen in itertools.combinations(range(1, size), extra):
                    subset = [0, *chosen]
                    if not mfmc_admissible(covariance, costs, models=subset):
                        continue
                    order = [0, *sorted(chosen, key=lambda i: -rho2[i])]
                    squares = [1.0] + [rho2[i] for i in order[1:]] + [0.0]
                    values[tuple(order)] = (
                        covariance[0][0]
                        / 300.0
                        * math.fsum(
                            math.sqrt(costs[i] * (squares[j] - squares[j + 1]))
                            for j, i in enumerate(order)
                        )
                        ** 2
                    )
            self.assertAlmostEqual(
                best.continuous_variance / min(values.values()), 1.0, places=10
            )
            self.assertAlmostEqual(
                best.continuous_variance / values[best.models], 1.0, places=10
            )
            self.assertLessEqual(
                best.continuous_variance, best.mc_variance * (1 + 1e-12)
            )
            dropped += len(best.models) < size
        self.assertGreater(dropped, 0)

    def test_selection_falls_back_to_plain_monte_carlo(self) -> None:
        """A useless low-fidelity model is dropped; max_models is honored."""
        weak = [[1.0, 0.3], [0.3, 1.0]]
        plan = select_mfmc_models(weak, (1.0, 0.95), 50.0)
        self.assertEqual(plan.models, (0,))
        self.assertEqual(plan.samples, (50,))
        self.assertAlmostEqual(plan.variance, 1.0 / 50.0)
        self.assertEqual(plan.evaluations, (50, 0))
        limited = select_mfmc_models(covariance_of(4), COSTS, 100.0, max_models=2)
        self.assertEqual(len(limited.models), 2)
        full = select_mfmc_models(covariance_of(4), COSTS, 100.0)
        self.assertLess(full.continuous_variance, limited.continuous_variance)

    def test_rejects_invalid_input(self) -> None:
        """Malformed covariances, costs, budgets, and subsets are refused."""
        covariance = covariance_of(3)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            mfmc_allocation([[1.0, 1.0], [1.0, 1.0]], (1.0, 0.1), 10.0)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            mfmc_allocation([[1.0, 0.5], [0.4, 1.0]], (1.0, 0.1), 10.0)
        with self.assertRaisesRegex(ValueError, "one entry per model"):
            mfmc_allocation(covariance, (1.0, 0.1), 10.0)
        with self.assertRaisesRegex(ValueError, "costs must be positive"):
            mfmc_allocation(covariance, (1.0, 0.0, 0.1), 10.0)
        with self.assertRaisesRegex(ValueError, "budget must be positive"):
            mfmc_allocation(covariance, COSTS[:3], 0.0)
        with self.assertRaisesRegex(ValueError, "budget is too small"):
            mfmc_allocation(covariance, COSTS[:3], 1.0)
        with self.assertRaisesRegex(ValueError, "start with the high-fidelity"):
            mfmc_allocation(covariance, COSTS[:3], 10.0, models=[1, 2])
        with self.assertRaisesRegex(ValueError, "repeat"):
            mfmc_allocation(covariance, COSTS[:3], 10.0, models=[0, 1, 1])
        with self.assertRaisesRegex(ValueError, "outside"):
            mfmc_allocation(covariance, COSTS[:3], 10.0, models=[0, 3])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            mfmc_variance(covariance, [5, 4, 9])
        with self.assertRaisesRegex(ValueError, "one entry per selected"):
            mfmc_variance(covariance, [5, 9])
        with self.assertRaisesRegex(ValueError, "below the cost"):
            select_mfmc_models(covariance, COSTS[:3], 0.5)
        with self.assertRaisesRegex(ValueError, "max_models"):
            select_mfmc_models(covariance, COSTS[:3], 5.0, max_models=0)


class MFMCEstimatorTests(ReplicateAssertions):
    """The MFMC estimator on Gaussian models with known covariance."""

    def test_replicates_are_unbiased_with_the_predicted_variance(self) -> None:
        """400 seeded replicates match the mean and the exact variance."""
        covariance = covariance_of(4)
        models = gaussian_models(covariance, MEANS)
        plan = mfmc_allocation(covariance, COSTS, 12.0)
        sampler = normal_inputs(4)
        mean, variance = replicate(
            lambda seed: mfmc_estimate(models, sampler, plan, seed=seed), 400
        )
        self.assert_matches_prediction(mean, variance, 400, MEANS[0], plan.variance)

    def test_fixed_wrong_coefficients_stay_unbiased(self) -> None:
        """Weights from a wrong covariance keep the estimator unbiased.

        Its variance follows the general formula evaluated with the true
        covariance, not the prediction made from the wrong one.
        """
        truth = covariance_of(3)
        wrong = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        models = gaussian_models(truth, MEANS[:3])
        plan = mfmc_allocation(wrong, (1.0, 0.05, 0.001), 12.0)
        actual = mfmc_variance(
            truth, plan.samples, models=plan.models, coefficients=plan.coefficients
        )
        self.assertGreater(abs(actual / plan.variance - 1.0), 0.2)
        sampler = normal_inputs(3)
        mean, variance = replicate(
            lambda seed: mfmc_estimate(models, sampler, plan, seed=seed), 400
        )
        self.assert_matches_prediction(mean, variance, 400, MEANS[0], actual)

    def test_estimate_recomputed_from_the_documented_stream(self) -> None:
        """The estimate equals the formula applied to the ``mfmc`` child stream."""
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        sampler = normal_inputs(3)
        plan = mfmc_allocation(covariance, COSTS[:3], 15.0, models=[0, 2])
        result = mfmc_estimate(models, sampler, plan, costs=COSTS[:3], seed=3)
        inputs = sampler(plan.samples[-1], RandomStream(3).child("mfmc"))
        high = [models[0](z) for z in inputs[: plan.samples[0]]]
        low = [models[2](z) for z in inputs]
        expected = statistics.fmean(high) + plan.coefficients[1] * (
            statistics.fmean(low) - statistics.fmean(low[: plan.samples[0]])
        )
        self.assertAlmostEqual(result.estimate, expected, places=12)
        self.assertAlmostEqual(result.high_fidelity_mean, statistics.fmean(high))
        self.assertEqual(result.method, "MFMC")
        self.assertEqual(result.evaluations, (plan.samples[0], 0, plan.samples[1]))
        self.assertEqual(result.total_evaluations, sum(plan.samples))
        self.assertAlmostEqual(result.cost or 0.0, plan.cost)
        self.assertAlmostEqual(result.standard_error, math.sqrt(plan.variance))
        self.assertEqual(
            result, mfmc_estimate(models, sampler, plan, costs=COSTS[:3], seed=3)
        )
        self.assertIsNone(mfmc_estimate(models, sampler, plan, seed=3).cost)
        with self.assertRaisesRegex(ValueError, "not supplied"):
            mfmc_estimate(models[:2], sampler, plan, seed=3)


class ACVTests(ReplicateAssertions):
    """Approximate control variates: weights, variance factors, allocation."""

    def test_variance_factors_equal_the_index_set_oracle(self) -> None:
        """The ACV-MF and ACV-IS factors reproduce exact mean covariances.

        ACV-MF uses ``F_ij = (min(r_i, r_j) - 1) / min(r_i, r_j)``.
        """
        covariance = covariance_of(4)
        for kind in ("MF", "IS"):
            for samples in ((6, 40, 15, 90), (4, 9, 9, 5), (5, 0, 30, 12)):
                design = acv_design(covariance, COSTS, samples, kind=kind)
                oracle = exact_variance(acv_terms(design), covariance)
                self.assertAlmostEqual(design.variance / oracle, 1.0, places=11)
                self.assertAlmostEqual(design.mc_variance, 1.0 / samples[0])
                self.assertLess(design.variance, design.mc_variance)

    def test_weights_minimize_the_exact_variance(self) -> None:
        """Perturbing ``alpha* = -Cov(D, D)^-1 Cov(D, Q_0)`` never helps."""
        covariance = covariance_of(4)
        generator = random.Random(2)
        for kind in ("MF", "IS"):
            design = acv_design(covariance, COSTS, (6, 40, 15, 90), kind=kind)
            for _ in range(50):
                trial = [w + 0.05 * generator.gauss(0.0, 1.0) for w in design.weights]
                self.assertGreater(
                    exact_variance(acv_terms(design, trial), covariance),
                    design.variance,
                )

    def test_large_ratios_reach_the_control_variate_limit(self) -> None:
        """As ``r -> infinity`` the variance tends to ``(C_00 - c' C_LL^-1 c) / N``."""
        covariance = covariance_of(3)
        low = [row[1:] for row in covariance[1:]]
        cross = covariance[0][1:]
        inverse = gauss_jordan_inverse(low)
        explained = math.fsum(
            cross[i] * inverse[i][j] * cross[j] for i in range(2) for j in range(2)
        )
        limit = (covariance[0][0] - explained) / 10
        for kind in ("MF", "IS"):
            design = acv_design(
                covariance, COSTS[:3], (10, 10**9, 2 * 10**9), kind=kind
            )
            self.assertAlmostEqual(design.variance / limit, 1.0, places=6)
            near = acv_design(covariance, COSTS[:3], (10, 11, 12), kind=kind)
            self.assertGreater(near.variance, 0.8 * near.mc_variance)
        rho, n = 0.95, 10
        two = acv_design(covariance_of(2), COSTS[:2], (n, 10**10))
        self.assertAlmostEqual(two.variance / ((1 - rho * rho) / n), 1.0, places=6)

    def test_two_models_reduce_to_mfmc(self) -> None:
        """With one control ACV-MF and ACV-IS equal MFMC on the same samples."""
        covariance = covariance_of(2)
        for count in (13, 40, 500):
            reference = mfmc_variance(covariance, [12, count])
            for kind in ("MF", "IS"):
                design = acv_design(covariance, COSTS[:2], (12, count), kind=kind)
                self.assertAlmostEqual(design.variance / reference, 1.0, places=12)
                self.assertAlmostEqual(design.ratios[0], count / 12)
                self.assertAlmostEqual(design.variance_ratio, reference * 12, places=12)

    def test_replicates_are_unbiased_with_the_predicted_variance(self) -> None:
        """Replicate simulation confirms the ACV-MF and ACV-IS variances."""
        covariance = covariance_of(4)
        models = gaussian_models(covariance, MEANS)
        sampler = normal_inputs(4)
        for kind in ("MF", "IS"):
            design = acv_design(covariance, COSTS, (6, 40, 15, 90), kind=kind)
            mean, variance = replicate(
                lambda seed, d=design: acv_estimate(models, sampler, d, seed=seed), 400
            )
            self.assert_matches_prediction(
                mean, variance, 400, MEANS[0], design.variance
            )

    def test_estimate_accounting_and_reproducibility(self) -> None:
        """Evaluations, cost, method label, and seeding are reported."""
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        sampler = normal_inputs(3)
        design = acv_design(covariance, COSTS[:3], (5, 0, 20), kind="IS")
        result = acv_estimate(models, sampler, design, seed=9)
        self.assertEqual(result.method, "ACV-IS")
        self.assertEqual(result.evaluations, (5, 0, 20))
        self.assertAlmostEqual(result.cost or 0.0, 5.0 + 0.2)
        self.assertEqual(design.weights[0], 0.0)
        self.assertEqual(result, acv_estimate(models, sampler, design, seed=9))
        self.assertNotEqual(
            result.estimate, acv_estimate(models, sampler, design, seed=10).estimate
        )
        shared = sampler(5, RandomStream(9).child("acv-is", "shared"))
        extra = sampler(15, RandomStream(9).child("acv-is", "extra", 2))
        low = [models[2](z) for z in shared + extra]
        high = statistics.fmean(models[0](z) for z in shared)
        expected = high + design.weights[1] * (
            statistics.fmean(low[:5]) - statistics.fmean(low)
        )
        self.assertAlmostEqual(result.estimate, expected, places=12)

    def test_allocation_respects_the_budget_and_beats_monte_carlo(self) -> None:
        """The heuristic allocation is feasible and no worse than plain MC."""
        covariance = covariance_of(4)
        for kind in ("MF", "IS"):
            plan = acv_allocation(covariance, COSTS, 60.0, kind=kind)
            self.assertLessEqual(plan.design.cost, 60.0 * (1 + 1e-9))
            self.assertLess(plan.design.variance, 1.0 / 60.0)
            self.assertLess(plan.continuous_variance, plan.design.variance * 1.05)
            self.assertIsInstance(plan.converged, bool)
            self.assertEqual(plan, acv_allocation(covariance, COSTS, 60.0, kind=kind))
            spent = math.fsum(
                w * n for w, n in zip(COSTS, plan.continuous_samples, strict=True)
            )
            self.assertAlmostEqual(spent, 60.0, places=6)

    def test_two_model_allocation_recovers_the_mfmc_optimum(self) -> None:
        """One control: the numerical optimum equals the MFMC closed form."""
        covariance = covariance_of(2)
        reference = mfmc_allocation(covariance, COSTS[:2], 80.0)
        plan = acv_allocation(covariance, COSTS[:2], 80.0)
        self.assertTrue(plan.converged)
        self.assertAlmostEqual(
            plan.continuous_variance / reference.continuous_variance, 1.0, places=7
        )
        for got, want in zip(
            plan.continuous_samples, reference.continuous_samples, strict=True
        ):
            self.assertAlmostEqual(got / want, 1.0, places=3)

    def test_allocation_drops_a_useless_control(self) -> None:
        """An expensive weakly correlated model gets no samples."""
        weak = [[1.0, 0.3], [0.3, 1.0]]
        plan = acv_allocation(weak, (1.0, 0.95), 50.0)
        self.assertEqual(plan.design.samples, (50, 0))
        self.assertAlmostEqual(plan.design.variance, 1.0 / 50.0)
        self.assertIn("plain Monte Carlo", plan.message)

    def test_rejects_invalid_input(self) -> None:
        """Bad sample designs, kinds, and budgets are refused."""
        covariance = covariance_of(3)
        with self.assertRaisesRegex(ValueError, "kind must be"):
            acv_design(covariance, COSTS[:3], (5, 10, 20), kind="KL")
        with self.assertRaisesRegex(ValueError, "exceed the high-fidelity"):
            acv_design(covariance, COSTS[:3], (5, 5, 20))
        with self.assertRaisesRegex(ValueError, "one entry per model"):
            acv_design(covariance, COSTS[:3], (5, 20))
        with self.assertRaisesRegex(ValueError, r"samples\[0\]"):
            acv_design(covariance, COSTS[:3], (0, 10, 20))
        with self.assertRaisesRegex(ValueError, r"samples\[1\]"):
            acv_design(covariance, COSTS[:3], (5, 10.5, 20))  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "below the cost"):
            acv_allocation(covariance, COSTS[:3], 0.5)
        with self.assertRaisesRegex(ValueError, "starts"):
            acv_allocation(covariance, COSTS[:3], 5.0, starts=0)
        design = acv_design(covariance, COSTS[:3], (5, 10, 20))
        with self.assertRaisesRegex(ValueError, "disagree"):
            acv_estimate([lambda z: 0.0], normal_inputs(1), design, seed=1)


class MLBLUEVarianceTests(ReplicateAssertions):
    """The multilevel BLUE and its variance."""

    GROUPS = ((0, 1, 2), (1, 2), (2,), (0, 2), (1,))
    SAMPLES = (3, 4, 9, 2, 5)

    def test_variance_equals_brute_force_generalized_least_squares(self) -> None:
        """``[Psi^-1]_00`` equals the stacked GLS covariance entry."""
        covariance = covariance_of(3)
        inverse, _ = stacked_gls(covariance, self.GROUPS, self.SAMPLES)
        predicted = mlblue_variance(covariance, self.GROUPS, self.SAMPLES)
        self.assertAlmostEqual(predicted / inverse[0][0], 1.0, places=10)

    def test_estimate_equals_brute_force_generalized_least_squares(self) -> None:
        """The estimate equals GLS on the outputs of the documented streams."""
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        sampler = normal_inputs(3)
        result = mlblue_estimate(
            models,
            sampler,
            covariance,
            self.GROUPS,
            self.SAMPLES,
            costs=COSTS[:3],
            seed=6,
        )
        observations: list[float] = []
        for k, (group, count) in enumerate(zip(self.GROUPS, self.SAMPLES, strict=True)):
            inputs = sampler(count, RandomStream(6).child("mlblue-group", k))
            for z in inputs:
                observations.extend(models[index](z) for index in group)
        inverse, solution = stacked_gls(
            covariance, self.GROUPS, self.SAMPLES, observations
        )
        self.assertAlmostEqual(result.estimate, solution[0], places=10)
        self.assertAlmostEqual(result.variance / inverse[0][0], 1.0, places=10)
        self.assertEqual(result.method, "MLBLUE")
        self.assertEqual(result.evaluations, (5, 12, 18))
        self.assertEqual(result.high_fidelity_samples, 5)
        self.assertAlmostEqual(result.cost or 0.0, 5 + 1.2 + 0.18)
        again = mlblue_estimate(
            models,
            sampler,
            covariance,
            self.GROUPS,
            self.SAMPLES,
            costs=COSTS[:3],
            seed=6,
        )
        self.assertEqual(result, again)

    def test_single_group_is_plain_monte_carlo(self) -> None:
        """Only the group ``{0}``: sample mean with variance ``C_00 / n``."""
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        result = mlblue_estimate(
            models, normal_inputs(3), covariance, [(0,), (1, 2)], [7, 0], seed=2
        )
        self.assertAlmostEqual(result.variance, covariance[0][0] / 7)
        self.assertAlmostEqual(result.estimate, result.high_fidelity_mean, places=12)
        self.assertEqual(result.evaluations, (7, 0, 0))
        full = mlblue_variance(covariance, [(0, 1, 2)], [7])
        self.assertAlmostEqual(full, covariance[0][0] / 7)

    def test_mfmc_groups_reproduce_the_mfmc_variance(self) -> None:
        """Nested MFMC groups give at most the MFMC variance.

        Equality holds for two models and for a Markov (AR(1)) correlation;
        in general the BLUE on the same samples is strictly better.
        """
        two = covariance_of(2)
        self.assertAlmostEqual(
            mlblue_variance(two, [(0, 1), (1,)], [10, 30])
            / mfmc_variance(two, [10, 40]),
            1.0,
            places=12,
        )
        markov = [[0.9 ** abs(i - j) for j in range(4)] for i in range(4)]
        nested = [(0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)]
        self.assertAlmostEqual(
            mlblue_variance(markov, nested, [10, 30, 160, 300])
            / mfmc_variance(markov, [10, 40, 200, 500]),
            1.0,
            places=12,
        )
        generator = random.Random(12)
        strict = 0
        for _ in range(20):
            covariance = random_covariance(generator, 4)
            blue = mlblue_variance(covariance, nested, [10, 30, 160, 300])
            mfmc = mfmc_variance(covariance, [10, 40, 200, 500])
            self.assertLessEqual(blue, mfmc * (1 + 1e-12))
            strict += blue < mfmc * (1 - 1e-6)
        self.assertGreater(strict, 0)

    def test_blue_is_no_worse_than_acv_on_the_same_samples(self) -> None:
        """ACV estimators are linear and unbiased, so the BLUE bounds them.

        The bound is strict for this ACV-MF design and attained by ACV-IS.
        """
        covariance = covariance_of(4)
        nested = acv_design(covariance, COSTS, (6, 40, 15, 90), kind="MF")
        blue = mlblue_variance(
            covariance, [(0, 1, 2, 3), (1, 2, 3), (1, 3), (3,)], [6, 9, 25, 50]
        )
        self.assertLess(blue, nested.variance)
        independent = acv_design(covariance, COSTS, (6, 40, 15, 90), kind="IS")
        blue = mlblue_variance(
            covariance, [(0, 1, 2, 3), (1,), (2,), (3,)], [6, 34, 9, 84]
        )
        # For the ACV-IS sample structure the optimal ACV weights are the BLUE.
        self.assertAlmostEqual(blue / independent.variance, 1.0, places=10)

    def test_mlmc_groups_reproduce_the_mlmc_variance(self) -> None:
        """Telescoping groups give at most the MLMC variance ``sum V_l / n_l``.

        Equality holds when the unit telescoping weights are the optimal ones.
        """
        groups = [(0, 1), (1, 2), (2,)]
        counts = [8, 30, 200]

        def mlmc(covariance: Sequence[Sequence[float]]) -> float:
            """Return the MLMC variance with model 0 as the finest level."""
            return (
                (covariance[0][0] + covariance[1][1] - 2 * covariance[0][1]) / counts[0]
                + (covariance[1][1] + covariance[2][2] - 2 * covariance[1][2])
                / counts[1]
                + covariance[2][2] / counts[2]
            )

        generator = random.Random(21)
        for _ in range(20):
            covariance = random_covariance(generator, 3)
            self.assertLessEqual(
                mlblue_variance(covariance, groups, counts),
                mlmc(covariance) * (1 + 1e-12),
            )
        # Two levels: MLMC is the BLUE iff C_01 = C_11 (1 + n_1 / n_2).
        n1, n2 = 10, 40
        special = [[3.0, 1.25], [1.25, 1.0]]
        two_level = (3.0 + 1.0 - 2.5) / n1 + 1.0 / n2
        self.assertAlmostEqual(
            mlblue_variance(special, [(0, 1), (1,)], [n1, n2]), two_level, places=12
        )
        terms: list[Term] = [
            (1.0, 0, pool("a", 0, n1)),
            (-1.0, 1, pool("a", 0, n1)),
            (1.0, 1, pool("b", 0, n2)),
        ]
        self.assertAlmostEqual(exact_variance(terms, special), two_level, places=12)

    def test_replicates_are_unbiased_with_the_predicted_variance(self) -> None:
        """Replicate simulation confirms the MLBLUE variance.

        A wrong fixed covariance keeps the estimator unbiased.
        """
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        sampler = normal_inputs(3)
        predicted = mlblue_variance(covariance, self.GROUPS, self.SAMPLES)
        mean, variance = replicate(
            lambda seed: mlblue_estimate(
                models, sampler, covariance, self.GROUPS, self.SAMPLES, seed=seed
            ),
            400,
        )
        self.assert_matches_prediction(mean, variance, 400, MEANS[0], predicted)
        wrong = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        mean, variance = replicate(
            lambda seed: mlblue_estimate(
                models, sampler, wrong, self.GROUPS, self.SAMPLES, seed=seed
            ),
            400,
        )
        self.assertLess(abs(mean - MEANS[0]), 4.0 * math.sqrt(variance / 400))
        self.assertGreater(variance, predicted)

    def test_groups_and_validation(self) -> None:
        """Group enumeration and malformed inputs."""
        self.assertEqual(len(mlblue_groups(4)), 15)
        self.assertEqual(
            mlblue_groups(3, max_size=2),
            ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)),
        )
        covariance = covariance_of(3)
        with self.assertRaisesRegex(ValueError, "high-fidelity model 0"):
            mlblue_variance(covariance, [(0, 1), (1, 2)], [0, 5])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            mlblue_variance(covariance, [(0, 1)], [-1])
        with self.assertRaisesRegex(ValueError, "one entry per group"):
            mlblue_variance(covariance, [(0, 1)], [1, 2])
        with self.assertRaisesRegex(ValueError, "distinct"):
            mlblue_variance(covariance, [(0, 1), (1, 0)], [1, 2])
        with self.assertRaisesRegex(ValueError, "outside"):
            mlblue_variance(covariance, [(0, 3)], [1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            mlblue_variance(covariance, [()], [1])
        with self.assertRaisesRegex(ValueError, "at least one model group"):
            mlblue_variance(covariance, [], [])
        with self.assertRaisesRegex(ValueError, "models"):
            mlblue_groups(0)
        with self.assertRaisesRegex(ValueError, "at most 4095"):
            mlblue_groups(13)
        models = gaussian_models(covariance, MEANS[:3])
        with self.assertRaisesRegex(ValueError, "samples"):
            mlblue_estimate(
                models,
                normal_inputs(3),
                covariance,
                [(0,)],
                [1.5],
                seed=1,  # type: ignore[list-item]
            )
        with self.assertRaisesRegex(ValueError, "one row per model"):
            mlblue_estimate(models[:2], normal_inputs(3), covariance, [(0,)], [2])


class MLBLUEAllocationTests(unittest.TestCase):
    """Continuous relaxation, certificate, and rounding of the allocation."""

    def test_two_models_recover_the_mfmc_closed_form(self) -> None:
        """With two models the optimum is the MFMC closed-form allocation."""
        covariance = covariance_of(2)
        reference = mfmc_allocation(covariance, COSTS[:2], 80.0)
        plan = mlblue_allocation(covariance, COSTS[:2], 80.0)
        self.assertTrue(plan.converged)
        self.assertAlmostEqual(
            plan.continuous_variance / reference.continuous_variance, 1.0, places=6
        )
        by_group = dict(zip(plan.groups, plan.continuous_samples, strict=True))
        m0, m1 = reference.continuous_samples
        self.assertAlmostEqual(by_group[(0, 1)] / m0, 1.0, places=3)
        self.assertAlmostEqual(by_group[(1,)] / (m1 - m0), 1.0, places=3)
        self.assertLess(by_group[(0,)], 1e-3)

    def test_all_groups_are_never_worse_than_mfmc(self) -> None:
        """Certified optimum: lower bound <= MLBLUE <= MFMC for one budget."""
        generator = random.Random(31)
        cases = [(covariance_of(4), list(COSTS))]
        for _ in range(8):
            size = generator.choice([3, 4])
            cases.append(
                (
                    random_covariance(generator, size),
                    [10.0 ** (-i * generator.uniform(0.3, 1.2)) for i in range(size)],
                )
            )
        strict = 0
        for covariance, costs in cases:
            mfmc = select_mfmc_models(covariance, costs, 200.0)
            plan = mlblue_allocation(covariance, costs, 200.0)
            self.assertTrue(plan.converged, plan.message)
            self.assertLessEqual(plan.relative_gap, 1e-6)
            self.assertLessEqual(plan.lower_bound, plan.continuous_variance)
            self.assertLessEqual(
                plan.continuous_variance, mfmc.continuous_variance * (1 + 1e-6)
            )
            self.assertLessEqual(plan.lower_bound, mfmc.continuous_variance)
            self.assertLessEqual(
                plan.continuous_variance, plan.mc_variance * (1 + 1e-6)
            )
            strict += plan.continuous_variance < mfmc.continuous_variance * (1 - 1e-4)
            self.assertAlmostEqual(math.fsum(plan.shares), 1.0, places=12)
            self.assertAlmostEqual(
                plan.continuous_variance
                / mlblue_variance(covariance, plan.groups, plan.continuous_samples),
                1.0,
                places=9,
            )
        self.assertGreater(strict, 0)

    def test_no_feasible_allocation_beats_the_lower_bound(self) -> None:
        """Random budget-feasible allocations stay above the certified bound."""
        covariance = covariance_of(4)
        plan = mlblue_allocation(covariance, COSTS, 150.0)
        generator = random.Random(13)
        for _ in range(300):
            raw = [generator.expovariate(1.0) ** 3 for _ in plan.groups]
            total = math.fsum(raw)
            counts = [
                150.0 * value / (total * cost)
                for value, cost in zip(raw, plan.group_costs, strict=True)
            ]
            self.assertGreaterEqual(
                mlblue_variance(covariance, plan.groups, counts), plan.lower_bound
            )

    def test_rounded_allocation_is_feasible_and_consistent(self) -> None:
        """Integers respect the budget and their variance is recomputed."""
        covariance = covariance_of(4)
        for budget in (3.0, 30.0, 1000.0):
            plan = mlblue_allocation(covariance, COSTS, budget)
            self.assertLessEqual(plan.cost, budget * (1 + 1e-9))
            self.assertGreater(plan.cost, 0.97 * budget)
            self.assertGreaterEqual(
                plan.variance, plan.continuous_variance * (1 - 1e-9)
            )
            self.assertAlmostEqual(
                plan.variance / mlblue_variance(covariance, plan.groups, plan.samples),
                1.0,
                places=12,
            )
            evaluations = [0] * 4
            for group, count in zip(plan.groups, plan.samples, strict=True):
                for index in group:
                    evaluations[index] += count
            self.assertEqual(plan.evaluations, tuple(evaluations))
            self.assertAlmostEqual(plan.variance_ratio, plan.variance * budget)
        self.assertLess(plan.variance / plan.continuous_variance, 1.001)

    def test_restricted_groups_and_size_limit(self) -> None:
        """Fewer groups cannot lower the optimal variance."""
        covariance = covariance_of(4)
        full = mlblue_allocation(covariance, COSTS, 100.0)
        pairs = mlblue_allocation(covariance, COSTS, 100.0, max_group_size=2)
        nested = mlblue_allocation(
            covariance, COSTS, 100.0, groups=[(0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)]
        )
        only = mlblue_allocation(covariance, COSTS, 100.0, groups=[(0,)])
        self.assertTrue(pairs.converged and nested.converged and only.converged)
        self.assertEqual(len(pairs.groups), 10)
        self.assertLessEqual(
            full.continuous_variance, pairs.continuous_variance * (1 + 1e-6)
        )
        self.assertLessEqual(
            full.continuous_variance, nested.continuous_variance * (1 + 1e-6)
        )
        self.assertAlmostEqual(only.continuous_variance, 0.01, places=12)
        self.assertEqual(only.samples, (100,))
        mfmc = mfmc_allocation(covariance, COSTS, 100.0)
        self.assertLessEqual(
            nested.continuous_variance, mfmc.continuous_variance * (1 + 1e-6)
        )

    def test_non_convergence_is_flagged(self) -> None:
        """Without restarts and with a tiny tolerance the flag is false."""
        plan = mlblue_allocation(
            covariance_of(4), COSTS, 100.0, tolerance=1e-15, max_restarts=0
        )
        self.assertFalse(plan.converged)
        self.assertIn("gap exceeds the tolerance", plan.message)
        self.assertGreater(plan.relative_gap, 1e-15)
        self.assertLess(plan.relative_gap, 1e-3)

    def test_rejects_invalid_input(self) -> None:
        """Groups without model 0, tiny budgets, and bad options are refused."""
        covariance = covariance_of(3)
        with self.assertRaisesRegex(ValueError, "no group contains"):
            mlblue_allocation(covariance, COSTS[:3], 10.0, groups=[(1, 2)])
        with self.assertRaisesRegex(ValueError, "does not pay for one sample"):
            mlblue_allocation(covariance, COSTS[:3], 0.5)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            mlblue_allocation(covariance, COSTS[:3], 10.0, tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "max_restarts"):
            mlblue_allocation(covariance, COSTS[:3], 10.0, max_restarts=-1)


class ComparisonTests(unittest.TestCase):
    """The comparison table."""

    def test_rows_agree_with_the_individual_allocations(self) -> None:
        """Every row repeats the result of its own allocation routine."""
        covariance = covariance_of(4)
        table = compare_multifidelity_estimators(covariance, COSTS, 100.0)
        self.assertEqual(
            [row.method for row in table.rows], ["MC", "MFMC", "ACV-MF", "MLBLUE"]
        )
        plain = table.row("MC")
        self.assertAlmostEqual(plain.variance, 1.0 / 100)
        self.assertEqual(plain.evaluations, (100, 0, 0, 0))
        mfmc = select_mfmc_models(covariance, COSTS, 100.0)
        self.assertEqual(table.row("MFMC").variance, mfmc.variance)
        self.assertEqual(table.mfmc_models, mfmc.models)
        acv = acv_allocation(covariance, COSTS, 100.0)
        self.assertEqual(table.row("ACV-MF").variance, acv.design.variance)
        blue = mlblue_allocation(covariance, COSTS, 100.0)
        self.assertEqual(table.row("MLBLUE").variance, blue.variance)
        self.assertEqual(table.best.method, "MLBLUE")
        for row in table.rows:
            self.assertLessEqual(row.cost, 100.0 * (1 + 1e-9))
            self.assertAlmostEqual(row.variance_ratio, row.variance / plain.variance)
            self.assertLessEqual(
                table.row("MLBLUE").continuous_variance,
                row.continuous_variance * (1 + 1e-6),
            )
        self.assertTrue(any(0 in group for group in table.mlblue_groups))
        text = table.format()
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn("MLBLUE", text)
        with self.assertRaisesRegex(ValueError, "unknown method"):
            table.row("QMC")
        with self.assertRaisesRegex(ValueError, "below the cost"):
            compare_multifidelity_estimators(covariance, COSTS, 0.5)

    def test_pilot_to_estimate_workflow(self) -> None:
        """Pilot covariance, allocation, and estimate run end to end."""
        covariance = covariance_of(3)
        models = gaussian_models(covariance, MEANS[:3])
        sampler = normal_inputs(3)
        pilot = pilot_statistics(models, sampler, 60, seed=77)
        plan = mlblue_allocation(pilot.covariance, COSTS[:3], 50.0)
        result = mlblue_estimate(
            models,
            sampler,
            pilot.covariance,
            plan.groups,
            plan.samples,
            costs=COSTS[:3],
            seed=77,
        )
        self.assertAlmostEqual(result.cost or 0.0, plan.cost, places=9)
        self.assertLess(abs(result.estimate - MEANS[0]), 5.0 * result.standard_error)
        truth = mlblue_variance(covariance, plan.groups, plan.samples)
        self.assertLess(truth, covariance[0][0] / 50.0)


if __name__ == "__main__":
    unittest.main()
