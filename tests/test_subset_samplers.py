"""Tests for the Markov-chain kernels of subset simulation."""

from __future__ import annotations

import math
import statistics
import unittest
from collections.abc import Callable, Sequence

from probstat.distributions import Normal
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch_model,
    standard_normal_cdf,
)
from probstat.reliability import (
    AVAILABLE_METHODS,
    METHODS,
    SUBSET_SAMPLERS,
    ModelLimitState,
    SubsetSimulationResult,
    compare_reliability_methods,
    subset_simulation,
)
from probstat.reliability.simulation import _level_cov
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)


def standard_pair() -> JointInput:
    """Return two independent standard-normal inputs ``u1`` and ``u2``."""
    return JointInput(
        (RandomInput("u1", Normal(0.0, 1.0)), RandomInput("u2", Normal(0.0, 1.0)))
    )


def linear_pair(beta: float) -> ModelLimitState:
    """Return ``G = beta - (u1 + u2)/sqrt 2`` with ``P_f = Phi(-beta)``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": beta - (p["u1"] + p["u2"]) / math.sqrt(2.0)},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


def parabolic_pair(kappa: float) -> ModelLimitState:
    """Return ``G = 3 - u2 + kappa u1^2 / 2``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": 3.0 - p["u2"] + 0.5 * kappa * p["u1"] ** 2},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


def four_branch_state() -> ModelLimitState:
    """Return the four-branch series limit state (``k = 7``)."""
    return ModelLimitState(four_branch_model(), "g", 0.0, "below")


#: Reference failure probability of the four-branch limit state with
#: ``k = 7`` (2e8 crude Monte Carlo samples, standard error 3.3e-6).
FOUR_BRANCH_REFERENCE = 2.2245e-3


class StandardLinearLimitState(ModelLimitState):
    """``G = beta - sum(u_i)/sqrt(d)`` evaluated directly in standard space.

    The inputs are independent standard normals, so the isoprobabilistic map
    is the identity; overriding :meth:`standard` skips the per-coordinate
    transform (the dominant cost at ``d = 100``) while still counting every
    call on the model's evaluation counter. ``P_f = Phi(-beta)`` for any
    ``d``; ``beta`` is minus the threshold.
    """

    __slots__ = ()

    def standard(self, u: Sequence[float]) -> float:
        """Count one evaluation and return ``beta - sum(u)/sqrt(d)``."""
        self.model.counter.add(1)
        return self.margin(-math.fsum(u) / math.sqrt(len(u)))


def high_dimensional_linear(beta: float, dimension: int = 100) -> ModelLimitState:
    """Return the ``d``-dimensional linear limit state with index ``beta``."""
    inputs = JointInput(
        tuple(RandomInput(f"u{i}", Normal(0.0, 1.0)) for i in range(dimension))
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": -math.fsum(p.values()) / math.sqrt(dimension)},
        Y,
    )
    return StandardLinearLimitState(model, "y", -beta, "below")


def within_error_scale(
    result: SubsetSimulationResult, exact: float, k: float = 3.0
) -> bool:
    """Return whether ``|ln(P_hat / P)| <= k delta_upper`` for the run's own c.o.v."""
    return (
        abs(math.log(result.probability / exact))
        <= k * result.coefficient_of_variation_upper
    )


#: Results of ``subset_simulation`` recorded before the kernel refactor.
#: Probabilities, conditional probabilities, acceptance rates, and call
#: counts come from integer counts and products of ``p0``; thresholds and
#: c.o.v. values involve the Gaussian draws and are compared with a relative
#: tolerance of 1e-12 so that last-digit libm differences between platforms
#: cannot fail the test.
PINNED: tuple[
    tuple[str, Callable[[], ModelLimitState], dict[str, object], dict[str, object]], ...
] = (
    (
        "linear-3.5-seed-1",
        lambda: linear_pair(3.5),
        {"samples_per_level": 500, "seed": 1},
        {
            "probability": 0.00013400000000000003,
            "thresholds": (
                2.2932613337190704,
                1.3466852352384786,
                0.615792216023668,
                0.0,
            ),
            "conditional_probabilities": (0.1, 0.1, 0.1, 0.134),
            "level_coefficients_of_variation": (
                0.1341640786499874,
                0.22090722034374521,
                0.2638181191654584,
                0.22498236360670812,
            ),
            "coefficient_of_variation": 0.43245469581686935,
            "acceptance_rates": (0.47555555555555556, 0.4, 0.34),
            "evaluations": 1704,
            "levels": 4,
        },
    ),
    (
        "linear-3.5-seed-7-spread-0.8",
        lambda: linear_pair(3.5),
        {"samples_per_level": 500, "seed": 7, "proposal_spread": 0.8},
        {
            "probability": 0.00036600000000000006,
            "thresholds": (
                2.3312100033774774,
                1.2145409302059549,
                0.26554800542438506,
                0.0,
            ),
            "conditional_probabilities": (0.1, 0.1, 0.1, 0.366),
            "level_coefficients_of_variation": (
                0.1341640786499874,
                0.2683281572999748,
                0.3346640106136302,
                0.11134944130962919,
            ),
            "coefficient_of_variation": 0.4630320702499629,
            "acceptance_rates": (
                0.5688888888888889,
                0.4666666666666667,
                0.37777777777777777,
            ),
            "evaluations": 1728,
            "levels": 4,
        },
    ),
    (
        "parabolic-p0-0.2-seed-11",
        lambda: parabolic_pair(-0.2),
        {"samples_per_level": 400, "level_probability": 0.2, "seed": 11},
        {
            "probability": 0.0014520000000000004,
            "thresholds": (
                2.1349010547167757,
                1.2620434200052113,
                0.45953544664084983,
                0.02212604393242658,
                0.0,
            ),
            "conditional_probabilities": (0.2, 0.2, 0.2, 0.2, 0.9075),
            "level_coefficients_of_variation": (
                0.1,
                0.17410485346480148,
                0.16677080080157916,
                0.17230060940112774,
                0.029655938473810615,
            ),
            "coefficient_of_variation": 0.31415278876171454,
            "acceptance_rates": (0.559375, 0.5125, 0.39375, 0.4125),
            "evaluations": 1530,
            "levels": 5,
        },
    ),
    (
        "four-branch-seed-3",
        four_branch_state,
        {"samples_per_level": 500, "seed": 3},
        {
            "probability": 0.0026000000000000007,
            "thresholds": (1.4598383794175807, 0.5025665813290323, 0.0),
            "conditional_probabilities": (0.1, 0.1, 0.26),
            "level_coefficients_of_variation": (
                0.1341640786499874,
                0.22978250586152113,
                0.16817854699288803,
            ),
            "coefficient_of_variation": 0.31477614850658403,
            "acceptance_rates": (0.45111111111111113, 0.32),
            "evaluations": 1298,
            "levels": 3,
        },
    ),
)

EXACT_FIELDS = (
    "probability",
    "conditional_probabilities",
    "acceptance_rates",
    "evaluations",
    "levels",
)
TOLERANT_FIELDS = (
    "thresholds",
    "level_coefficients_of_variation",
    "coefficient_of_variation",
)


class DefaultSamplerIdentityTests(unittest.TestCase):
    """The default kernel reproduces the results recorded before the refactor."""

    def test_default_results_are_unchanged(self) -> None:
        """Seeded MMA runs match the pinned values field by field."""
        for name, make, options, expected in PINNED:
            with self.subTest(case=name):
                result = subset_simulation(make(), **options)  # type: ignore[arg-type]
                self.assertTrue(result.converged)
                for field in EXACT_FIELDS:
                    self.assertEqual(getattr(result, field), expected[field], field)
                for field in TOLERANT_FIELDS:
                    actual = getattr(result, field)
                    wanted = expected[field]
                    if isinstance(wanted, tuple):
                        self.assertEqual(len(actual), len(wanted), field)
                        pairs = list(zip(actual, wanted, strict=True))
                    else:
                        pairs = [(actual, wanted)]
                    for got, target in pairs:
                        self.assertTrue(
                            math.isclose(got, target, rel_tol=1e-12, abs_tol=1e-15),
                            f"{field}: {got!r} != {target!r}",
                        )


class DefaultSamplerMetadataTests(unittest.TestCase):
    """The default result records its kernel and per-level spread."""

    def test_default_sampler_fields(self) -> None:
        """``sampler`` is ``"mma"`` and each level reports the spread ``s``."""
        result = subset_simulation(linear_pair(3.5), samples_per_level=500, seed=1)
        self.assertEqual(result.sampler, "mma")
        self.assertEqual(result.level_scales, (1.0, 1.0, 1.0))
        self.assertEqual(SUBSET_SAMPLERS, ("mma", "cs", "acs", "hmc"))


class FourBranchTests(unittest.TestCase):
    """Every kernel recovers the four-branch reference within its error scale."""

    def test_four_branch_all_samplers(self) -> None:
        """``|P_hat - P_ref| <= 3 delta_upper P_ref`` for each sampler."""
        for sampler in SUBSET_SAMPLERS:
            with self.subTest(sampler=sampler):
                result = subset_simulation(
                    four_branch_state(), sampler=sampler, seed=2026
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.sampler, sampler)
                self.assertEqual(len(result.level_scales), result.levels - 1)
                self.assertLessEqual(
                    abs(result.probability - FOUR_BRANCH_REFERENCE),
                    3.0 * result.coefficient_of_variation_upper * FOUR_BRANCH_REFERENCE,
                )


class AdaptationTests(unittest.TestCase):
    """aCS and RS-HMC move their scale toward the target acceptance."""

    def test_acs_lambda_grows_from_a_small_start(self) -> None:
        """Tiny steps accept nearly always, so ``lambda`` increases."""
        result = subset_simulation(
            linear_pair(3.5), sampler="acs", initial_scale=0.05, seed=4
        )
        self.assertGreater(result.level_scales[0], 0.3)
        self.assertGreater(result.acceptance_rates[0], 0.6)
        self.assertLess(abs(result.acceptance_rates[-1] - 0.44), 0.15)

    def test_acs_lambda_shrinks_from_a_large_start(self) -> None:
        """Independent proposals rarely land in ``F_j``, so ``lambda`` decreases."""
        result = subset_simulation(
            high_dimensional_linear(3.0), sampler="acs", initial_scale=3.0, seed=8
        )
        self.assertLess(result.level_scales[0], 3.0)
        self.assertTrue(
            all(
                a > b
                for a, b in zip(
                    result.level_scales, result.level_scales[1:], strict=False
                )
            )
        )
        self.assertGreater(result.acceptance_rates[-1], result.acceptance_rates[0])
        self.assertLess(abs(result.acceptance_rates[-1] - 0.44), 0.15)

    def test_hmc_trajectory_time_shrinks_when_acceptance_is_low(self) -> None:
        """A long trajectory leaves ``F_j`` too often, so ``t_f`` decreases."""
        result = subset_simulation(
            linear_pair(3.5), sampler="hmc", trajectory_time=math.pi / 2.0, seed=5
        )
        self.assertLess(result.level_scales[-1], math.pi / 2.0)
        self.assertGreater(result.acceptance_rates[-1], 0.2)

    def test_seed_reference_scale(self) -> None:
        """``scale_reference="seeds"`` runs and stays within the error scale."""
        result = subset_simulation(
            linear_pair(3.5), sampler="acs", scale_reference="seeds", seed=6
        )
        self.assertTrue(result.converged)
        self.assertTrue(within_error_scale(result, standard_normal_cdf(-3.5)))


class ChainLengthTests(unittest.TestCase):
    """Chains of unequal length and the generalized level c.o.v."""

    def test_unequal_chains_for_adaptive_samplers(self) -> None:
        """``N p0`` need not divide ``N`` except for ``"mma"``."""
        with self.assertRaises(ValueError):
            subset_simulation(
                linear_pair(3.0), samples_per_level=1000, level_probability=0.15
            )
        for sampler in ("cs", "acs", "hmc"):
            with self.subTest(sampler=sampler):
                result = subset_simulation(
                    linear_pair(3.0),
                    samples_per_level=1000,
                    level_probability=0.15,
                    sampler=sampler,
                    seed=12,
                )
                self.assertTrue(result.converged)
                self.assertTrue(within_error_scale(result, standard_normal_cdf(-3.0)))

    def test_level_cov_with_unequal_lengths(self) -> None:
        """A hand-computed two-chain example of the pair-count weights."""
        indicators = [True, True, False, True, False]
        p = 0.6
        variance = p * (1.0 - p)
        lag1 = 2.0 * (3 / 5) * (1 / 3 - p * p) / variance
        lag2 = 2.0 * (1 / 5) * (0 / 1 - p * p) / variance
        expected = math.sqrt((1.0 - p) / (5 * p) * max(0.0, 1.0 + lag1 + lag2))
        self.assertAlmostEqual(
            _level_cov(indicators, 2, p, True, [3, 2]), expected, places=14
        )

    def test_level_cov_equal_lengths_matches_default_path(self) -> None:
        """Equal lengths give the same value with or without ``lengths``."""
        indicators = [True, False, True, True, False, False, True, True, False]
        self.assertEqual(
            _level_cov(indicators, 3, 5 / 9, True),
            _level_cov(indicators, 3, 5 / 9, True, [3, 3, 3]),
        )


class ValidationTests(unittest.TestCase):
    """Invalid kernel options raise ``ValueError``."""

    def test_invalid_options(self) -> None:
        """Unknown sampler and out-of-range kernel options are rejected."""
        state = linear_pair(3.0)
        bad: tuple[dict[str, object], ...] = (
            {"sampler": "gibbs"},
            {"sampler": "cs", "correlation": 1.0},
            {"sampler": "cs", "correlation": -0.1},
            {"sampler": "acs", "initial_scale": 0.0},
            {"sampler": "acs", "target_acceptance": 1.0},
            {"sampler": "acs", "adaptation_chains": 0},
            {"sampler": "acs", "scale_reference": "median"},
            {"sampler": "hmc", "trajectory_time": 2.0},
            {"sampler": "hmc", "trajectory_time": 0.0},
        )
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                subset_simulation(state, samples_per_level=100, seed=1, **options)  # type: ignore[arg-type]


class ComparisonTests(unittest.TestCase):
    """The adaptive kernels are opt-in rows of ``compare_reliability_methods``."""

    def test_opt_in_rows(self) -> None:
        """``"acs"`` and ``"hmc-ss"`` add rows without changing ``METHODS``."""
        self.assertEqual(METHODS, ("form", "sorm", "is", "subset", "mc"))
        self.assertIn("acs", AVAILABLE_METHODS)
        self.assertIn("hmc-ss", AVAILABLE_METHODS)
        comparison = compare_reliability_methods(
            linear_pair(3.0), methods=("acs", "hmc-ss"), subset_samples=500, seed=9
        )
        self.assertEqual(
            [row.method for row in comparison.rows], ["Subset-aCS", "Subset-HMC"]
        )
        exact = standard_normal_cdf(-3.0)
        for row in comparison.rows:
            self.assertIsNotNone(row.coefficient_of_variation)
            self.assertLess(abs(math.log(row.probability / exact)), 1.5)


class HighDimensionalLinearTests(unittest.TestCase):
    """``G = beta - sum(u_i)/sqrt(100)`` for every kernel.

    The runs are shared between the accuracy and comparison tests: two seeds
    at ``beta = 3`` and three at ``beta = 5`` (``N = 1000``, ``p0 = 0.1``).
    """

    runs: dict[tuple[float, str], list[SubsetSimulationResult]]

    @classmethod
    def setUpClass(cls) -> None:
        """Run every sampler on both reliability indices."""
        cls.runs = {}
        for beta, seeds in ((3.0, (1, 2)), (5.0, (1, 2, 3))):
            state = high_dimensional_linear(beta)
            for sampler in SUBSET_SAMPLERS:
                cls.runs[beta, sampler] = [
                    subset_simulation(state, sampler=sampler, seed=seed)
                    for seed in seeds
                ]

    def test_every_sampler_is_within_its_error_scale(self) -> None:
        """``|ln(P_hat/Phi(-beta))| <= 3 delta_upper`` for every run."""
        for (beta, sampler), results in self.runs.items():
            exact = standard_normal_cdf(-beta)
            for result in results:
                with self.subTest(beta=beta, sampler=sampler):
                    self.assertTrue(result.converged)
                    self.assertEqual(result.sampler, sampler)
                    self.assertTrue(within_error_scale(result, exact))

    def test_adaptive_kernels_have_smaller_cov_than_mma(self) -> None:
        """At ``beta = 5`` aCS and RS-HMC report a clearly smaller c.o.v.

        Over 50 seeds the mean reported c.o.v. was 0.455 (sd 0.021) for MMA,
        0.374 (sd 0.010) for aCS, and 0.397 (sd 0.011) for RS-HMC, so a margin
        of 0.03 on the mean of three seeds is several standard errors wide.
        """

        def mean_cov(sampler: str) -> float:
            """Return the mean reported c.o.v. of one sampler at ``beta = 5``."""
            return statistics.fmean(
                result.coefficient_of_variation for result in self.runs[5.0, sampler]
            )

        reference = mean_cov("mma")
        for sampler in ("acs", "hmc"):
            with self.subTest(sampler=sampler):
                self.assertLess(mean_cov(sampler), reference - 0.03)

    def test_adaptive_acceptance_stays_near_target(self) -> None:
        """The aCS deep levels stay near 0.44; MMA acceptance collapses."""
        for result in self.runs[5.0, "acs"]:
            for rate in result.acceptance_rates[1:]:
                self.assertLess(abs(rate - 0.44), 0.1)
            self.assertLess(result.level_scales[-1], 0.6)
        for result in self.runs[5.0, "hmc"]:
            for rate in result.acceptance_rates[1:]:
                self.assertGreater(rate, 0.2)
                self.assertLess(rate, 0.6)
        for result in self.runs[5.0, "mma"]:
            self.assertLess(result.acceptance_rates[-1], 0.2)


if __name__ == "__main__":
    unittest.main()
