"""Uncertainty budgets, combination rules, and Monte Carlo propagation.

Oracles: closed forms for linear models and for the law of propagation with
correlation; a hand-computed Welch-Satterthwaite example with the Student-t
quantile pinned from R 4.6.2; the exact lognormal law of a product/quotient
model; the Irwin-Hall law of a sum of rectangular inputs (the additive model
of JCGM 101:2008 Section 9.2.3); the multivariate delta method of
``probstat.multivariate``. Monte Carlo results are judged on their own
standard errors, never on pinned digits. The tests never call R.

References:
    [REF-JCGM-100-2008], [REF-JCGM-101-2008], [REF-WELCH-1947],
    [REF-SATTERTHWAITE-1946], [REF-NELSEN-2006].
    Checks [EQ-UQ-BUDGET-001], [EQ-UQ-BUDGET-002], [EQ-UQ-BUDGET-003],
    [EQ-UQ-BUDGET-004], [EQ-UQ-BUDGET-005], [EQ-UQ-BUDGET-006],
    [EQ-UQ-BUDGET-007], [EQ-UQ-BUDGET-008], [EQ-UQ-BUDGET-009],
    [EQ-UQ-BUDGET-010], [EQ-BATCH-001], [EQ-RNG-SPLITMIX-001].

"""

from __future__ import annotations

import math
import operator
import statistics
import unittest
from collections.abc import Sequence
from dataclasses import replace
from statistics import NormalDist

from probstat.distributions import Exponential, LogNormal, Normal, Triangular, Uniform
from probstat.foundations import (
    AskTellSession,
    RandomStream,
    batch_from_pointwise,
    run_procedure,
)
from probstat.multivariate import delta_method
from probstat.uq import (
    AdaptiveMonteCarloResult,
    BudgetEntry,
    CombinedUncertainty,
    QuantityKind,
    UncertaintyBudget,
    adaptive_monte_carlo_budget,
    adaptive_monte_carlo_budget_procedure,
    adaptive_stopping_rule,
    average_standard_deviation,
    budget_from_model,
    compare_with_law_of_propagation,
    monte_carlo_budget,
    numerical_tolerance,
    sample_coverage_intervals,
)
from probstat.uq.budget import (
    ADEQUACY_OTHER_INTERVAL,
    ADEQUACY_PROXY,
    ADEQUACY_STABILIZATION,
)


def irwin_hall_cdf(x: float, count: int) -> float:
    """Return the distribution function of a sum of ``count`` uniforms on [0, 1].

    The explicit alternating-sum (Irwin-Hall) formula, used as an oracle that
    shares no code with the library.

    References:
        [REF-JCGM-101-2008, Section 9.2.3 and Annex E]. Checks
        [EQ-UQ-BUDGET-007].

    """
    if x <= 0.0:
        return 0.0
    if x >= count:
        return 1.0
    total = 0.0
    for k in range(int(math.floor(x)) + 1):
        total += (-1) ** k * math.comb(count, k) * (x - k) ** count
    return total / math.factorial(count)


def sum_of_rectangles_quantile(
    probability: float, half_width: float, count: int
) -> float:
    """Return a quantile of a sum of ``count`` uniforms on ``[-a, a]`` by bisection.

    References:
        [REF-JCGM-101-2008, Section 9.2.3]. Checks [EQ-UQ-BUDGET-007].

    """
    low, high = 0.0, float(count)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if irwin_hall_cdf(mid, count) < probability:
            low = mid
        else:
            high = mid
    return 2.0 * half_width * (0.5 * (low + high) - count / 2.0)


class BudgetEntryTests(unittest.TestCase):
    """Type B conversions and entry validation.

    References:
        [REF-JCGM-100-2008, Sections 4.3.4, 4.3.7 and 4.3.9],
        [REF-JCGM-101-2008, Sections 6.4.2, 6.4.5 and 6.4.6]. Checks
        [EQ-UQ-BUDGET-003].

    """

    def test_half_width_conversions(self) -> None:
        """Rectangular, triangular, arcsine and normal divisors are exact.

        References:
            [REF-JCGM-100-2008, Section 4.3.7]. Checks [EQ-UQ-BUDGET-003].

        """
        a = 1.5
        self.assertAlmostEqual(BudgetEntry.rectangular("r", a).u, a / math.sqrt(3.0))
        self.assertAlmostEqual(BudgetEntry.triangular("t", a).u, a / math.sqrt(6.0))
        self.assertAlmostEqual(BudgetEntry.arcsine("s", a).u, a / math.sqrt(2.0))
        normal = BudgetEntry.normal("n", 3.0, 2.0)
        self.assertAlmostEqual(normal.u, 1.5)
        self.assertEqual(normal.half_width, 3.0)
        self.assertEqual(BudgetEntry.rectangular("r", a).half_width, a)

    def test_guide_example_values(self) -> None:
        """The GUM handbook example gives 0.23e-6 and the 99 % resistor 50 uOhm.

        References:
            [REF-JCGM-100-2008, Sections 4.3.4 and 4.3.7]. Checks
            [EQ-UQ-BUDGET-003].

        """
        copper = BudgetEntry.rectangular("alpha", 0.40e-6)
        self.assertAlmostEqual(copper.u, 0.23e-6, delta=0.005e-6)
        resistor = BudgetEntry.normal("R", 129e-6, 2.58)
        self.assertAlmostEqual(resistor.u, 50e-6, delta=0.5e-6)

    def test_from_distribution_matches_closed_forms(self) -> None:
        """The standard deviation of a law reproduces the bound conversions.

        References:
            [REF-JCGM-101-2008, Section 6.4]. Checks [EQ-UQ-BUDGET-003].

        """
        a = 2.0
        uniform = BudgetEntry.from_distribution("u", Uniform(-a, a))
        self.assertAlmostEqual(uniform.u, a / math.sqrt(3.0))
        triangle = BudgetEntry.from_distribution("t", Triangular(-a, 0.0, a))
        self.assertAlmostEqual(triangle.u, a / math.sqrt(6.0))
        with self.assertRaises(ValueError):
            BudgetEntry.from_distribution("c", Exponential(1.0), half_width=-1.0)
        with self.assertRaises(TypeError):
            BudgetEntry.from_distribution("x", object())

    def test_from_observations_type_a(self) -> None:
        """``s / sqrt(n)`` with ``n - 1`` degrees of freedom, by hand.

        References:
            [REF-JCGM-100-2008, Sections 4.2.2-4.2.3]. Checks
            [EQ-UQ-BUDGET-003].

        """
        readings = [10.0, 12.0, 11.0, 13.0]
        # mean 11.5, squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5, s^2 = 5/3
        entry = BudgetEntry.from_observations("v", readings)
        self.assertAlmostEqual(entry.u, math.sqrt(5.0 / 3.0 / 4.0))
        self.assertEqual(entry.dof, 3.0)
        self.assertEqual(entry.evaluation, "A")
        with self.assertRaises(ValueError):
            BudgetEntry.from_observations("v", [1.0])
        with self.assertRaises(TypeError):
            BudgetEntry.from_observations("v", "12")

    def test_validation(self) -> None:
        """Bad fields are rejected with the right exception type.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            BudgetEntry("x", -1.0)
        with self.assertRaises(ValueError):
            BudgetEntry("x", 1.0, dof=0.0)
        with self.assertRaises(ValueError):
            BudgetEntry("x", 1.0, evaluation="C")
        with self.assertRaises(ValueError):
            BudgetEntry("x", 1.0, nature="vague")
        with self.assertRaises(ValueError):
            BudgetEntry("x", 1.0, kind="drift")
        with self.assertRaises(ValueError):
            BudgetEntry("x", 1.0, half_width=-0.1)
        with self.assertRaises(ValueError):
            BudgetEntry.normal("x", 1.0, 0.0)
        with self.assertRaises(TypeError):
            BudgetEntry("", 1.0)
        with self.assertRaises(TypeError):
            BudgetEntry("x", 1.0, evaluation=1)  # type: ignore[arg-type]
        self.assertEqual(BudgetEntry("x", 1.0, c=-1.0).contribution, -1.0)


class LawOfPropagationTests(unittest.TestCase):
    """Combined uncertainty, correlation terms, and shares against closed forms.

    References:
        [REF-JCGM-100-2008, Sections 5.1.2 and 5.2.2]. Checks
        [EQ-UQ-BUDGET-001] and [EQ-UQ-BUDGET-005].

    """

    def test_linear_model_independent(self) -> None:
        """``y = 2 x1 - 3 x2 + x3`` gives ``sqrt(sum (c u)^2)``.

        References:
            [REF-JCGM-100-2008, Section 5.1.2]. Checks [EQ-UQ-BUDGET-001].

        """
        budget = UncertaintyBudget(
            (
                BudgetEntry("x1", 0.5, c=2.0),
                BudgetEntry("x2", 0.2, c=-3.0),
                BudgetEntry("x3", 1.0),
            )
        )
        expected = math.sqrt(1.0 + 0.36 + 1.0)
        self.assertAlmostEqual(budget.combined_standard_uncertainty(), expected)
        self.assertAlmostEqual(budget.law_of_propagation().value, expected)
        self.assertEqual(
            budget.law_of_propagation().kind, QuantityKind.STANDARD_UNCERTAINTY
        )

    def test_two_term_correlation_closed_form(self) -> None:
        """``u_c^2 = a^2 + b^2 + 2 a b r`` with signed ``a = c1 u1``, ``b = c2 u2``.

        References:
            [REF-JCGM-100-2008, Section 5.2.2, Equation 13]. Checks
            [EQ-UQ-BUDGET-001].

        """
        for r in (-1.0, -0.3, 0.0, 0.6, 1.0):
            budget = UncertaintyBudget(
                (BudgetEntry("a", 0.4, c=1.0), BudgetEntry("b", 0.5, c=-2.0)),
                correlation=((1.0, r), (r, 1.0)),
            )
            a, b = 0.4, -1.0
            expected = math.sqrt(max(0.0, a * a + b * b + 2.0 * a * b * r))
            self.assertAlmostEqual(budget.combined_standard_uncertainty(), expected)

    def test_perfect_cancellation(self) -> None:
        """Equal and opposite fully correlated terms cancel exactly.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-001].

        """
        budget = UncertaintyBudget(
            (BudgetEntry("a", 1.0, c=1.0), BudgetEntry("b", 1.0, c=-1.0)),
            correlation=((1.0, 1.0), (1.0, 1.0)),
        )
        self.assertAlmostEqual(budget.combined_standard_uncertainty(), 0.0, places=12)
        self.assertTrue(all(math.isnan(row.share) for row in budget.contributions()))

    def test_agrees_with_multivariate_delta_method(self) -> None:
        """The scalar budget equals ``J Sigma J'`` from ``delta_method``.

        References:
            [REF-JCGM-100-2008, Section 5.2.2], [REF-CASELLA-BERGER-2002,
            Section 5.5.4]. Checks [EQ-UQ-BUDGET-001].

        """
        coefficients = (1.5, -0.7, 2.0)
        u = (0.3, 0.8, 0.1)
        corr = ((1.0, 0.5, -0.2), (0.5, 1.0, 0.1), (-0.2, 0.1, 1.0))
        covariance = tuple(
            tuple(u[i] * u[j] * corr[i][j] for j in range(3)) for i in range(3)
        )
        result = delta_method(
            lambda x: (sum(c * v for c, v in zip(coefficients, x, strict=True)),),
            (1.0, 2.0, 3.0),
            covariance,
        )
        budget = UncertaintyBudget(
            tuple(BudgetEntry(f"x{i}", u[i], c=coefficients[i]) for i in range(3)),
            correlation=corr,
        )
        self.assertAlmostEqual(
            budget.combined_variance(), result.covariance[0][0], places=8
        )

    def test_shares_sum_to_one_and_can_be_negative(self) -> None:
        """Shares add to one; anticorrelation gives a negative share.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-005].

        """
        independent = UncertaintyBudget((BudgetEntry("a", 3.0), BudgetEntry("b", 4.0)))
        shares = [row.share for row in independent.contributions()]
        self.assertAlmostEqual(shares[0], 9.0 / 25.0)
        self.assertAlmostEqual(shares[1], 16.0 / 25.0)
        correlated = UncertaintyBudget(
            (BudgetEntry("a", 3.0), BudgetEntry("b", 1.0)),
            correlation=((1.0, -0.8), (-0.8, 1.0)),
        )
        rows = correlated.contributions()
        self.assertAlmostEqual(sum(row.share for row in rows), 1.0)
        # t_b = 1 * (3 * -0.8 + 1) = -1.4 < 0: this term reduces u_c.
        self.assertLess(rows[1].share, 0.0)
        self.assertGreater(rows[0].share, 1.0)

    def test_correlation_validation(self) -> None:
        """Malformed, indefinite, or out-of-range correlation is rejected.

        References:
            [REF-JCGM-100-2008, Section 5.2.2].

        """
        pair = (BudgetEntry("a", 1.0), BudgetEntry("b", 1.0))
        with self.assertRaises(ValueError):
            UncertaintyBudget(pair, correlation=((1.0, 0.5), (0.4, 1.0)))
        with self.assertRaises(ValueError):
            UncertaintyBudget(pair, correlation=((1.0, 1.5), (1.5, 1.0)))
        with self.assertRaises(ValueError):
            UncertaintyBudget(pair, correlation=((2.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            UncertaintyBudget(pair, correlation=((1.0,),))
        three = (*pair, BudgetEntry("c", 1.0))
        indefinite = ((1.0, 0.9, -0.9), (0.9, 1.0, 0.9), (-0.9, 0.9, 1.0))
        with self.assertRaises(ValueError):
            UncertaintyBudget(three, correlation=indefinite)
        with self.assertRaises(ValueError):
            UncertaintyBudget((), correlation=None)
        with self.assertRaises(ValueError):
            UncertaintyBudget((BudgetEntry("a", 1.0), BudgetEntry("a", 2.0)))
        with self.assertRaises(TypeError):
            UncertaintyBudget((1.0,))  # type: ignore[arg-type]
        identity = UncertaintyBudget(pair, correlation=((1.0, 0.0), (0.0, 1.0)))
        self.assertFalse(identity.has_correlation)


class WelchSatterthwaiteTests(unittest.TestCase):
    """Effective degrees of freedom and the expanded uncertainty.

    References:
        [REF-JCGM-100-2008, Sections 6.2 and G.4.1], [REF-WELCH-1947],
        [REF-SATTERTHWAITE-1946]. Checks [EQ-UQ-BUDGET-002].

    """

    def setUp(self) -> None:
        """Build the hand example: ``u = (1, 2)``, ``nu = (4, 9)``.

        References:
            [REF-JCGM-100-2008, Section G.4.1].

        """
        self.budget = UncertaintyBudget(
            (BudgetEntry("a", 1.0, dof=4.0), BudgetEntry("b", 2.0, dof=9.0))
        )

    def test_hand_computed_effective_dof(self) -> None:
        """``nu_eff = (1 + 4)^2 / (1/4 + 16/9) = 900/73``.

        References:
            [REF-JCGM-100-2008, Section G.4.1, Equation G.2b]. Checks
            [EQ-UQ-BUDGET-002].

        """
        self.assertAlmostEqual(
            self.budget.effective_degrees_of_freedom(), 900.0 / 73.0, places=12
        )

    def test_expanded_uncertainty_against_r(self) -> None:
        """``k = t_0.975(900/73)`` pinned from R and ``U = k u_c``.

        References:
            [REF-JCGM-100-2008, Section G.4.1]. Checks [EQ-UQ-BUDGET-002].

        """
        result = self.budget.expanded_uncertainty(0.95)
        # R: qt(0.975, 900/73) -> 2.17238622610457
        self.assertAlmostEqual(result.coverage_factor, 2.17238622610457, places=8)
        self.assertAlmostEqual(
            result.value, 2.17238622610457 * math.sqrt(5.0), places=7
        )
        self.assertEqual(result.warnings, ())
        low, high = result.interval(10.0)
        self.assertAlmostEqual(high - low, 2.0 * result.value)
        # R: qt(0.95, 900/73) -> 1.7783358537717
        self.assertAlmostEqual(
            self.budget.expanded_uncertainty(0.90).coverage_factor,
            1.7783358537717,
            places=8,
        )

    def test_truncated_dof(self) -> None:
        """Truncating to the next lower integer uses ``nu = 12`` (GUM G.4.1 Note 1).

        References:
            [REF-JCGM-100-2008, Section G.4.1, Note 1]. Checks
            [EQ-UQ-BUDGET-002].

        """
        truncated = self.budget.expanded_uncertainty(0.95, truncate_dof=True)
        self.assertEqual(truncated.effective_dof, 12.0)
        self.assertGreater(
            truncated.coverage_factor,
            self.budget.expanded_uncertainty().coverage_factor,
        )
        tiny = UncertaintyBudget((BudgetEntry("a", 1.0, dof=0.5),))
        with self.assertRaises(ValueError):
            tiny.expanded_uncertainty(truncate_dof=True)

    def test_infinite_dof_gives_normal_factor(self) -> None:
        """All-infinite degrees of freedom give the normal factor 1.96.

        References:
            [REF-JCGM-100-2008, Section 6.2]. Checks [EQ-UQ-BUDGET-002].

        """
        budget = UncertaintyBudget((BudgetEntry("a", 1.0), BudgetEntry("b", 2.0)))
        self.assertTrue(math.isinf(budget.effective_degrees_of_freedom()))
        result = budget.expanded_uncertainty(0.95)
        self.assertAlmostEqual(
            result.coverage_factor, NormalDist().inv_cdf(0.975), places=9
        )
        zero = UncertaintyBudget((BudgetEntry("a", 0.0, dof=3.0),))
        self.assertTrue(math.isinf(zero.effective_degrees_of_freedom()))

    def test_single_entry_keeps_its_dof(self) -> None:
        """One entry has ``nu_eff = nu`` exactly.

        References:
            [REF-JCGM-100-2008, Section G.4.1]. Checks [EQ-UQ-BUDGET-002].

        """
        budget = UncertaintyBudget((BudgetEntry("a", 0.7, c=3.0, dof=5.0),))
        self.assertAlmostEqual(budget.effective_degrees_of_freedom(), 5.0)

    def test_correlated_inputs_are_flagged(self) -> None:
        """With correlation the heuristic runs and warns; ``error`` refuses.

        References:
            [REF-JCGM-100-2008, Section G.4.1]. Checks [EQ-UQ-BUDGET-002].

        """
        budget = UncertaintyBudget(
            (BudgetEntry("a", 1.0, dof=4.0), BudgetEntry("b", 2.0, dof=9.0)),
            correlation=((1.0, 0.5), (0.5, 1.0)),
        )
        result = budget.expanded_uncertainty()
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("uncorrelated", result.warnings[0])
        u_c = budget.combined_standard_uncertainty()
        expected = u_c**4 / (1.0 / 4.0 + 16.0 / 9.0)
        self.assertAlmostEqual(result.effective_dof, expected)
        with self.assertRaises(ValueError):
            budget.expanded_uncertainty(correlated="error")
        with self.assertRaises(ValueError):
            budget.effective_degrees_of_freedom(correlated="maybe")

    def test_coverage_validation(self) -> None:
        """Coverage must be strictly inside ``(0, 1)``.

        References:
            [REF-JCGM-100-2008, Section 6.2].

        """
        for bad in (0.0, 1.0, 1.2):
            with self.assertRaises(ValueError):
                self.budget.expanded_uncertainty(bad)


class CombinationRuleTests(unittest.TestCase):
    """Root-sum-square, worst-case and mixed rules, with a decibel link budget.

    References:
        [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

    """

    def link_budget(self) -> UncertaintyBudget:
        """Return a six-term link budget in dB (terms with ``c = +1`` or ``-1``).

        References:
            [REF-JCGM-100-2008, Section 4.3.7]. Checks [EQ-UQ-BUDGET-003].

        """
        return UncertaintyBudget(
            (
                BudgetEntry.rectangular("transmit power", 0.5, kind="random"),
                BudgetEntry.rectangular("transmit antenna gain", 0.7, kind="bias"),
                BudgetEntry.rectangular("path loss", 0.2, c=-1.0, kind="bias"),
                BudgetEntry.triangular("atmospheric loss", 1.0, c=-1.0, kind="bias"),
                BudgetEntry.rectangular("pointing loss", 0.6, c=-1.0, kind="random"),
                BudgetEntry.rectangular("receive gain", 0.4, kind="random"),
            )
        )

    def test_link_budget_hand_values(self) -> None:
        """RSS, linear sum and mixed rule against explicit hand arithmetic.

        References:
            [REF-JCGM-100-2008, Section 5.1.2]. Checks [EQ-UQ-BUDGET-004].

        """
        budget = self.link_budget()
        a = [0.5, 0.7, 0.2, 1.0, 0.6, 0.4]
        u = [x / math.sqrt(3.0) for x in a]
        u[3] = 1.0 / math.sqrt(6.0)
        rss = math.sqrt(sum(v * v for v in u))
        self.assertAlmostEqual(budget.root_sum_square().value, rss)
        self.assertAlmostEqual(budget.law_of_propagation().value, rss)
        bound = budget.worst_case_sum()
        self.assertAlmostEqual(bound.value, sum(a))
        self.assertEqual(bound.kind, QuantityKind.WORST_CASE_BOUND)
        linear_standard = budget.worst_case_sum("standard_uncertainty")
        self.assertAlmostEqual(linear_standard.value, sum(u))
        self.assertEqual(linear_standard.kind, QuantityKind.STANDARD_UNCERTAINTY)
        biases, randoms = (1, 2, 3), (0, 4, 5)
        mixed = budget.mixed_rule()
        expected = sum(u[i] for i in biases) + math.sqrt(
            sum(u[i] ** 2 for i in randoms)
        )
        self.assertAlmostEqual(mixed.value, expected)
        self.assertEqual(mixed.kind, QuantityKind.STANDARD_UNCERTAINTY)
        # Ordering of the three standard-uncertainty rules.
        self.assertLessEqual(rss, mixed.value)
        self.assertLessEqual(mixed.value, linear_standard.value)

    def test_mixed_rule_at_bound_level(self) -> None:
        """Bias half-widths plus ``k`` times the random RSS, with ``k`` stated.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

        """
        budget = self.link_budget()
        result = budget.mixed_rule(level="bound", random_coverage_factor=2.0)
        random_u = math.sqrt((0.5**2 + 0.6**2 + 0.4**2) / 3.0)
        self.assertAlmostEqual(result.value, 0.7 + 0.2 + 1.0 + 2.0 * random_u)
        self.assertEqual(result.kind, QuantityKind.WORST_CASE_BOUND)
        self.assertEqual(result.coverage_factor, 2.0)
        with self.assertRaises(ValueError):
            budget.mixed_rule(level="bound")
        with self.assertRaises(ValueError):
            budget.mixed_rule(level="standard", random_coverage_factor=2.0)
        with self.assertRaises(ValueError):
            budget.mixed_rule(level="bound", random_coverage_factor=-1.0)
        with self.assertRaises(ValueError):
            budget.mixed_rule(level="other")
        no_bound = UncertaintyBudget((BudgetEntry("b", 1.0, kind="bias"),))
        with self.assertRaises(ValueError):
            no_bound.mixed_rule(level="bound", random_coverage_factor=2.0)

    def test_mixed_rule_without_biases_or_randoms(self) -> None:
        """All-random reduces to RSS; all-bias reduces to the linear sum.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

        """
        random_only = UncertaintyBudget((BudgetEntry("a", 3.0), BudgetEntry("b", 4.0)))
        self.assertAlmostEqual(random_only.mixed_rule().value, 5.0)
        bias_only = UncertaintyBudget(
            (
                BudgetEntry("a", 3.0, kind="bias"),
                BudgetEntry("b", 4.0, c=-1.0, kind="bias"),
            )
        )
        self.assertAlmostEqual(bias_only.mixed_rule().value, 7.0)

    def test_worst_case_needs_half_widths(self) -> None:
        """An entry without a half-width makes the bound undefined.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

        """
        budget = UncertaintyBudget((BudgetEntry("a", 1.0),))
        with self.assertRaises(ValueError):
            budget.worst_case_sum()
        with self.assertRaises(ValueError):
            budget.worst_case_sum("something")
        self.assertAlmostEqual(budget.worst_case_sum("standard_uncertainty").value, 1.0)

    def test_root_sum_square_ignores_correlation_and_says_so(self) -> None:
        """With correlation the RSS differs from the law of propagation and notes it.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

        """
        budget = UncertaintyBudget(
            (BudgetEntry("a", 1.0), BudgetEntry("b", 1.0)),
            correlation=((1.0, 0.5), (0.5, 1.0)),
        )
        rss = budget.root_sum_square()
        self.assertAlmostEqual(rss.value, math.sqrt(2.0))
        self.assertAlmostEqual(budget.law_of_propagation().value, math.sqrt(3.0))
        self.assertEqual(len(rss.notes), 1)
        self.assertIn("ignored", rss.notes[0])

    def test_kinds_cannot_be_mixed_by_arithmetic(self) -> None:
        """The result record defines no arithmetic or ordering.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-004].

        """
        budget = self.link_budget()
        standard = budget.root_sum_square()
        bound = budget.worst_case_sum()
        self.assertIsInstance(standard, CombinedUncertainty)
        with self.assertRaises(TypeError):
            operator.add(standard, bound)
        with self.assertRaises(TypeError):
            operator.lt(standard, bound)
        self.assertIn("worst-case bound", bound.describe())
        self.assertIn("standard uncertainty", standard.describe())


class SubsetAndRenderingTests(unittest.TestCase):
    """Subsets, grouping, and table rendering.

    References:
        [REF-JCGM-100-2008, Sections 5.2.2 and 7.2]. Checks
        [EQ-UQ-BUDGET-005].

    """

    def budget(self) -> UncertaintyBudget:
        """Return a three-entry budget with a correlation matrix.

        References:
            [REF-JCGM-100-2008, Section 5.2.2].

        """
        return UncertaintyBudget(
            (
                BudgetEntry("a", 1.0, evaluation="A", nature="aleatory", kind="random"),
                BudgetEntry("b", 2.0, evaluation="B", nature="epistemic", kind="bias"),
                BudgetEntry(
                    "c", 3.0, evaluation="B", nature="epistemic", kind="random"
                ),
            ),
            correlation=((1.0, 0.2, 0.4), (0.2, 1.0, -0.5), (0.4, -0.5, 1.0)),
        )

    def test_select_restricts_the_correlation_matrix(self) -> None:
        """The sub-budget keeps the matching sub-matrix and combines on its own.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-001].

        """
        epistemic = self.budget().select(nature="epistemic")
        self.assertEqual(epistemic.names, ("b", "c"))
        self.assertEqual(epistemic.correlation, ((1.0, -0.5), (-0.5, 1.0)))
        expected = math.sqrt(4.0 + 9.0 + 2.0 * 2.0 * 3.0 * -0.5)
        self.assertAlmostEqual(epistemic.combined_standard_uncertainty(), expected)
        self.assertEqual(self.budget().select(evaluation="A").names, ("a",))
        self.assertEqual(self.budget().select(names=["c", "a"]).names, ("a", "c"))
        with self.assertRaises(ValueError):
            self.budget().select(evaluation="A", kind="bias")
        with self.assertRaises(ValueError):
            self.budget().select(names=["zzz"])

    def test_group_by(self) -> None:
        """Groups follow first appearance and partition the entries.

        References:
            [REF-JCGM-100-2008, Section 5.2.2].

        """
        groups = self.budget().group_by("kind")
        self.assertEqual(list(groups), ["random", "bias"])
        self.assertEqual(groups["random"].names, ("a", "c"))
        self.assertEqual(sum(len(g) for g in groups.values()), 3)
        with self.assertRaises(ValueError):
            self.budget().group_by("dof")

    def test_rendering(self) -> None:
        """Rows, Markdown and text carry names, shares and the total.

        References:
            [REF-JCGM-100-2008, Clause 7]. Checks [EQ-UQ-BUDGET-005].

        """
        budget = UncertaintyBudget(
            (BudgetEntry("a", 3.0), BudgetEntry("b", 4.0, dof=9.0))
        )
        rows = budget.to_rows(3)
        self.assertEqual(rows[0][0], "name")
        self.assertEqual(rows[1][-1], "36.0%")
        self.assertEqual(rows[2][-1], "64.0%")
        self.assertEqual(rows[-1][6], "5")
        markdown = budget.to_markdown(3, coverage=0.95)
        self.assertTrue(markdown.startswith("| name |"))
        self.assertIn("nu_eff", markdown)
        self.assertEqual(len(markdown.splitlines()[0].split("|")), len(rows[0]) + 2)
        text = budget.to_text()
        self.assertIn("combined u_c", text)
        with self.assertRaises(ValueError):
            budget.to_rows(0)


class BudgetFromModelTests(unittest.TestCase):
    """Sensitivity coefficients by differentiation and the second-order check.

    References:
        [REF-JCGM-100-2008, Sections 5.1.2-5.1.4]. Checks
        [EQ-UQ-BUDGET-001] and [EQ-UQ-BUDGET-006].

    """

    def test_linear_model_exact(self) -> None:
        """A linear model recovers its coefficients and ``u_c`` in closed form.

        References:
            [REF-JCGM-100-2008, Section 5.1.2]. Checks [EQ-UQ-BUDGET-001].

        """
        result = budget_from_model(
            lambda x: 2.0 * x[0] - 3.0 * x[1] + 5.0, (1.0, 2.0), (0.5, 0.2)
        )
        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertAlmostEqual(result.gradient[0], 2.0, places=6)
        self.assertAlmostEqual(result.gradient[1], -3.0, places=6)
        self.assertAlmostEqual(
            result.budget.combined_standard_uncertainty(),
            math.sqrt(1.0 + 0.36),
            places=6,
        )
        assert result.linearization is not None
        self.assertFalse(result.linearization.poor)
        self.assertAlmostEqual(result.linearization.mean_shift, 0.0, places=5)

    def test_product_quotient_relative_uncertainties(self) -> None:
        """For ``y = x1 x2 / x3`` the relative variances add (GUM 5.1.6).

        References:
            [REF-JCGM-100-2008, Section 5.1.6]. Checks [EQ-UQ-BUDGET-001].

        """
        x = (2.0, 3.0, 4.0)
        u = (0.02, 0.03, 0.04)
        result = budget_from_model(
            lambda v: v[0] * v[1] / v[2],
            x,
            u,
            second_order=False,
            names=["a", "b", "c"],
        )
        y = 2.0 * 3.0 / 4.0
        expected = y * math.sqrt(
            sum((ui / xi) ** 2 for ui, xi in zip(u, x, strict=True))
        )
        self.assertAlmostEqual(
            result.budget.combined_standard_uncertainty(), expected, places=6
        )
        self.assertIsNone(result.linearization)
        self.assertEqual(result.budget.names, ("a", "b", "c"))

    def test_correlated_model_matches_delta_method(self) -> None:
        """A correlated nonlinear model agrees with ``J Sigma J'`` of the delta method.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. Checks [EQ-UQ-BUDGET-001].

        """

        def model(v: tuple[float, ...]) -> float:
            """Return ``x1^2 + sin(x2)``.

            References:
                [REF-JCGM-100-2008, Section 5.1.2]. No mathematical
                equation is evaluated.

            """
            return v[0] ** 2 + math.sin(v[1])

        u = (0.1, 0.2)
        corr = ((1.0, 0.7), (0.7, 1.0))
        cov = tuple(tuple(u[i] * u[j] * corr[i][j] for j in range(2)) for i in range(2))
        reference = delta_method(lambda v: (model(tuple(v)),), (1.5, 0.4), cov)
        result = budget_from_model(
            lambda v: model(tuple(v)),
            (1.5, 0.4),
            u,
            correlation=corr,
            second_order=False,
        )
        self.assertAlmostEqual(
            result.budget.combined_variance(), reference.covariance[0][0], places=6
        )

    def test_second_order_check_closed_form(self) -> None:
        """``y = x^2`` at ``x = 1``: shift ``u^2``, extra variance ``2 u^4``.

        References:
            [REF-JCGM-101-2008, Section 5.8]. Checks [EQ-UQ-BUDGET-006].

        """
        u = 0.5
        check = budget_from_model(lambda v: v[0] ** 2, (1.0,), (u,)).linearization
        assert check is not None
        self.assertAlmostEqual(check.mean_shift, u * u, places=5)
        self.assertAlmostEqual(check.second_order_variance, 2.0 * u**4, places=5)
        self.assertAlmostEqual(check.linear_variance, 4.0 * u * u, places=5)
        self.assertAlmostEqual(check.mean_ratio, u * u / (2.0 * u), places=5)
        self.assertAlmostEqual(
            check.spread_ratio,
            math.sqrt(1.0 + 2.0 * u**4 / (4.0 * u * u)) - 1.0,
            places=5,
        )
        self.assertTrue(check.poor)
        small = budget_from_model(lambda v: v[0] ** 2, (1.0,), (0.02,)).linearization
        assert small is not None
        self.assertFalse(small.poor)

    def test_degenerate_linear_variance(self) -> None:
        """Zero gradient with non-zero curvature is flagged as poor.

        References:
            [REF-JCGM-101-2008, Section 5.8]. Checks [EQ-UQ-BUDGET-006].

        """
        check = budget_from_model(lambda v: v[0] ** 2, (0.0,), (0.3,)).linearization
        assert check is not None
        self.assertTrue(check.poor)

    def test_validation(self) -> None:
        """Shape, type and range errors.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(TypeError):
            budget_from_model(3.0, (1.0,), (0.1,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            budget_from_model(lambda v: v[0], (1.0, 2.0), (0.1,))
        with self.assertRaises(ValueError):
            budget_from_model(lambda v: v[0], (1.0,), (0.1,), tolerance=0.0)
        with self.assertRaises(ValueError):
            budget_from_model(lambda v: v[0], (1.0,), (0.1,), names=["a", "b"])
        with self.assertRaises(ValueError):
            budget_from_model(lambda v: v[0], (1.0,), (0.1,), dof=[1.0, 2.0])
        with self.assertRaises(ValueError):
            budget_from_model(lambda v: math.nan, (1.0,), (0.1,))


class SampleIntervalTests(unittest.TestCase):
    """Symmetric and shortest coverage intervals of a sample (JCGM 101, 7.7.2).

    References:
        [REF-JCGM-101-2008, Sections 5.3 and 7.7.2]. Checks
        [EQ-UQ-BUDGET-007].

    """

    def test_hand_example(self) -> None:
        """Values 1..100 at ``p = 0.9``: ``q = 90``, ``r = 5``, interval [5, 95].

        References:
            [REF-JCGM-101-2008, Section 7.7.2]. Checks [EQ-UQ-BUDGET-007].

        """
        symmetric, shortest = sample_coverage_intervals(range(1, 101), 0.9)
        self.assertEqual(symmetric, (5.0, 95.0))
        self.assertEqual(shortest[1] - shortest[0], 90.0)

    def test_symmetric_law_gives_equal_intervals(self) -> None:
        """A sample at the quantiles of a symmetric law: shortest equals symmetric.

        References:
            [REF-JCGM-101-2008, Section 5.3.5]. Checks [EQ-UQ-BUDGET-007].

        """
        law = Normal(0.0, 1.0)
        m = 4000
        values = [law.quantile((i + 0.5) / m) for i in range(m)]
        symmetric, shortest = sample_coverage_intervals(values, 0.95)
        self.assertAlmostEqual(symmetric[0], -1.959964, delta=0.01)
        self.assertAlmostEqual(symmetric[1], 1.959964, delta=0.01)
        length_symmetric = symmetric[1] - symmetric[0]
        length_shortest = shortest[1] - shortest[0]
        self.assertLessEqual(length_shortest, length_symmetric + 1e-12)
        self.assertAlmostEqual(length_shortest, length_symmetric, delta=1e-3)

    def test_skewed_law_gives_strictly_shorter_interval(self) -> None:
        """Exponential(1): shortest 95 % interval starts at 0 and is shorter.

        References:
            [REF-JCGM-101-2008, Section 5.3.4]. Checks [EQ-UQ-BUDGET-007].

        """
        m = 20000
        law = Exponential(1.0)
        values = [law.quantile((i + 0.5) / m) for i in range(m)]
        symmetric, shortest = sample_coverage_intervals(values, 0.95)
        exact_symmetric = (-math.log(1.0 - 0.025), -math.log(0.025))
        exact_shortest_high = -math.log(0.05)
        self.assertAlmostEqual(symmetric[0], exact_symmetric[0], delta=1e-3)
        self.assertAlmostEqual(symmetric[1], exact_symmetric[1], delta=2e-2)
        self.assertAlmostEqual(shortest[0], 0.0, delta=1e-3)
        self.assertAlmostEqual(shortest[1], exact_shortest_high, delta=2e-2)
        self.assertLess(shortest[1] - shortest[0], 0.9 * (symmetric[1] - symmetric[0]))

    def test_validation(self) -> None:
        """Too-small samples and bad coverage are rejected.

        References:
            [REF-JCGM-101-2008, Section 7.2.2].

        """
        with self.assertRaises(ValueError):
            sample_coverage_intervals([1.0, 2.0, 3.0], 0.999)
        with self.assertRaises(ValueError):
            sample_coverage_intervals(range(100), 1.0)
        with self.assertRaises(ValueError):
            sample_coverage_intervals([1.0, math.inf, 3.0, 4.0], 0.5)


class MonteCarloTests(unittest.TestCase):
    """Monte Carlo propagation judged on its own standard errors.

    References:
        [REF-JCGM-101-2008, Sections 7.2-7.7 and 9.2]. Checks
        [EQ-UQ-BUDGET-007] and [EQ-UQ-BUDGET-008].

    """

    def test_additive_rectangular_model(self) -> None:
        """Four unit-variance rectangular inputs: the Supplement's 9.2.3 case.

        The interval ``+/- 2 sqrt(3)(2 - (3/5)^(1/4))`` of the Supplement is
        first checked against the Irwin-Hall oracle, then the Monte Carlo
        endpoints against it on their estimated standard errors.

        References:
            [REF-JCGM-101-2008, Section 9.2.3]. Checks [EQ-UQ-BUDGET-007].

        """
        a = math.sqrt(3.0)
        exact_high = sum_of_rectangles_quantile(0.975, a, 4)
        supplement = 2.0 * math.sqrt(3.0) * (2.0 - (3.0 / 5.0) ** 0.25)
        self.assertAlmostEqual(exact_high, supplement, places=9)
        result = monte_carlo_budget(
            lambda x: sum(x), [Uniform(-a, a)] * 4, size=120000, seed=20080101
        )
        self.assertAlmostEqual(
            result.estimate, 0.0, delta=5.0 * result.estimate_standard_error
        )
        self.assertAlmostEqual(
            result.standard_uncertainty,
            2.0,
            delta=5.0 * result.uncertainty_standard_error,
        )
        low_error, high_error = result.endpoint_standard_errors
        self.assertAlmostEqual(
            result.symmetric_interval[0], -exact_high, delta=5.0 * low_error
        )
        self.assertAlmostEqual(
            result.symmetric_interval[1], exact_high, delta=5.0 * high_error
        )
        self.assertAlmostEqual(result.skewness, 0.0, delta=0.05)
        # kurtosis of a sum of four rectangles: 3 - 1.2/4 = 2.7
        self.assertAlmostEqual(result.kurtosis, 2.7, delta=0.1)
        self.assertEqual(result.size, 120000)
        self.assertFalse(result.correlated)

    def test_product_quotient_against_lognormal(self) -> None:
        """``Y = X1 X2 / X3`` with lognormal inputs is exactly lognormal.

        References:
            [REF-JCGM-101-2008, Section 5.9]. Checks [EQ-UQ-BUDGET-007].

        """
        mus = (0.1, -0.2, 0.3)
        sigmas = (0.10, 0.15, 0.12)
        mu = mus[0] + mus[1] - mus[2]
        variance = sum(s * s for s in sigmas)
        sigma = math.sqrt(variance)
        exact_mean = math.exp(mu + 0.5 * variance)
        exact_sd = exact_mean * math.sqrt(math.expm1(variance))
        law = NormalDist(mu, sigma)
        result = monte_carlo_budget(
            lambda x: x[0] * x[1] / x[2],
            [LogNormal(m, s) for m, s in zip(mus, sigmas, strict=True)],
            size=100000,
            seed=7,
        )
        self.assertAlmostEqual(
            result.estimate, exact_mean, delta=5.0 * result.estimate_standard_error
        )
        self.assertAlmostEqual(
            result.standard_uncertainty,
            exact_sd,
            delta=5.0 * result.uncertainty_standard_error,
        )
        low_error, high_error = result.endpoint_standard_errors
        self.assertAlmostEqual(
            result.symmetric_interval[0],
            math.exp(law.inv_cdf(0.025)),
            delta=5.0 * low_error,
        )
        self.assertAlmostEqual(
            result.symmetric_interval[1],
            math.exp(law.inv_cdf(0.975)),
            delta=5.0 * high_error,
        )
        # Right-skewed output: the shortest interval is shifted left and shorter.
        symmetric_length = result.symmetric_interval[1] - result.symmetric_interval[0]
        shortest_length = result.shortest_interval[1] - result.shortest_interval[0]
        self.assertLess(shortest_length, symmetric_length)
        self.assertLess(result.shortest_interval[1], result.symmetric_interval[1])

    def test_symmetric_law_intervals_nearly_coincide(self) -> None:
        """A normal output: shortest and symmetric lengths agree within noise.

        References:
            [REF-JCGM-101-2008, Section 5.3.5]. Checks [EQ-UQ-BUDGET-007].

        """
        result = monte_carlo_budget(
            lambda x: x[0] + x[1],
            [Normal(1.0, 0.3), Normal(2.0, 0.4)],
            size=100000,
            seed=3,
        )
        symmetric_length = result.symmetric_interval[1] - result.symmetric_interval[0]
        shortest_length = result.shortest_interval[1] - result.shortest_interval[0]
        self.assertLessEqual(shortest_length, symmetric_length)
        self.assertAlmostEqual(shortest_length, symmetric_length, delta=0.02)
        self.assertAlmostEqual(
            result.standard_uncertainty,
            0.5,
            delta=5.0 * result.uncertainty_standard_error,
        )

    def test_gaussian_copula_correlation(self) -> None:
        """Sum and difference of correlated normals have variance ``2(1 +/- r)``.

        For normal marginals the copula correlation is the Pearson
        correlation, so ``Var(X1 + X2) = 2 (1 + r)``.

        References:
            [REF-NELSEN-2006], [REF-JCGM-101-2008, Section 6.4.8]. Checks
            [EQ-UQ-BUDGET-008].

        """
        r = 0.8
        corr = ((1.0, r), (r, 1.0))
        law = [Normal(0.0, 1.0), Normal(0.0, 1.0)]
        plus = monte_carlo_budget(
            lambda x: x[0] + x[1], law, size=20000, seed=11, correlation=corr
        )
        minus = monte_carlo_budget(
            lambda x: x[0] - x[1], law, size=20000, seed=11, correlation=corr
        )
        self.assertTrue(plus.correlated)
        self.assertAlmostEqual(
            plus.standard_uncertainty,
            math.sqrt(2.0 * (1.0 + r)),
            delta=5.0 * plus.uncertainty_standard_error,
        )
        self.assertAlmostEqual(
            minus.standard_uncertainty,
            math.sqrt(2.0 * (1.0 - r)),
            delta=5.0 * minus.uncertainty_standard_error,
        )

    def test_copula_marginals_are_preserved(self) -> None:
        """With a copula each marginal keeps its law: uniform mean and sd.

        References:
            [REF-NELSEN-2006]. Checks [EQ-UQ-BUDGET-008].

        """
        corr = ((1.0, -0.6), (-0.6, 1.0))
        first = monte_carlo_budget(
            lambda x: x[0],
            [Uniform(0.0, 1.0), Exponential(1.0)],
            size=60000,
            seed=5,
            correlation=corr,
        )
        second = monte_carlo_budget(
            lambda x: x[1],
            [Uniform(0.0, 1.0), Exponential(1.0)],
            size=60000,
            seed=5,
            correlation=corr,
        )
        self.assertAlmostEqual(
            first.estimate, 0.5, delta=5.0 * first.estimate_standard_error
        )
        self.assertAlmostEqual(
            first.standard_uncertainty,
            math.sqrt(1.0 / 12.0),
            delta=5.0 * first.uncertainty_standard_error,
        )
        self.assertAlmostEqual(
            second.estimate, 1.0, delta=5.0 * second.estimate_standard_error
        )
        self.assertAlmostEqual(
            second.standard_uncertainty,
            1.0,
            delta=5.0 * second.uncertainty_standard_error,
        )

    def test_reproducible_and_stream_accepted(self) -> None:
        """A fixed seed reproduces the result; a stream and its seed agree.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. No mathematical equation is evaluated.

        """
        law = [Normal(0.0, 1.0), Uniform(0.0, 1.0)]

        def model(x: tuple[float, ...]) -> float:
            """Return ``x1 * x2``.

            References:
                [REF-JCGM-101-2008, Section 7.4]. No mathematical equation is
                evaluated.

            """
            return x[0] * x[1]

        first = monte_carlo_budget(model, law, size=2000, seed=42)
        second = monte_carlo_budget(model, law, size=2000, seed=42)
        third = monte_carlo_budget(model, law, size=2000, seed=RandomStream(42))
        other = monte_carlo_budget(model, law, size=2000, seed=43)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertNotEqual(first.estimate, other.estimate)

    def test_validation(self) -> None:
        """Argument errors.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        law = [Normal(0.0, 1.0)]
        with self.assertRaises(TypeError):
            monte_carlo_budget(1.0, law, size=100, seed=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            monte_carlo_budget(lambda x: x[0], [object()], size=100, seed=1)  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            monte_carlo_budget(lambda x: x[0], law, size=100, seed="a")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            monte_carlo_budget(lambda x: x[0], law, size=5, seed=1)
        with self.assertRaises(ValueError):
            monte_carlo_budget(lambda x: x[0], [], size=100, seed=1)
        with self.assertRaises(ValueError):
            monte_carlo_budget(lambda x: x[0], law, size=100, seed=1, coverage=1.0)
        with self.assertRaises(ValueError):
            monte_carlo_budget(
                lambda x: x[0],
                law * 2,
                size=100,
                seed=1,
                correlation=((1.0, 2.0), (2.0, 1.0)),
            )
        with self.assertRaises(ValueError):
            monte_carlo_budget(lambda x: math.inf, law, size=100, seed=1)


class ValidationOfGumTests(unittest.TestCase):
    """The Supplement's validation of the GUM uncertainty framework.

    References:
        [REF-JCGM-101-2008, Sections 7.9.2, 8.1.3 and 8.2]. Checks
        [EQ-UQ-BUDGET-009].

    """

    def test_numerical_tolerance_examples(self) -> None:
        """The worked examples of Section 7.9.2.

        References:
            [REF-JCGM-101-2008, Section 7.9.2, Examples 1-3]. Checks
            [EQ-UQ-BUDGET-009].

        """
        self.assertAlmostEqual(numerical_tolerance(0.00035, 2), 5e-6)
        self.assertAlmostEqual(numerical_tolerance(0.00035, 1), 5e-5)
        self.assertAlmostEqual(numerical_tolerance(2.0, 1), 0.5)
        self.assertAlmostEqual(numerical_tolerance(2.0, 2), 0.05)
        # 9.96 rounds up to 10 at two digits, i.e. 10 x 10^0, not 100 x 10^-1.
        self.assertAlmostEqual(numerical_tolerance(9.96, 2), 0.5)
        with self.assertRaises(ValueError):
            numerical_tolerance(0.0, 2)
        with self.assertRaises(ValueError):
            numerical_tolerance(1.0, 0)

    def test_additive_rectangular_case_is_validated_at_one_digit(self) -> None:
        """The additive model passes at ``ndig = 1`` (``delta = 0.5``).

        References:
            [REF-JCGM-101-2008, Section 9.2.3]. Checks [EQ-UQ-BUDGET-009].

        """
        a = math.sqrt(3.0)
        result = monte_carlo_budget(
            lambda x: sum(x), [Uniform(-a, a)] * 4, size=60000, seed=99
        )
        outcome = compare_with_law_of_propagation(
            0.0, 2.0, result, significant_digits=1
        )
        self.assertEqual(outcome.delta, 0.5)
        self.assertTrue(outcome.validated)
        self.assertAlmostEqual(
            outcome.coverage_factor, NormalDist().inv_cdf(0.975), places=9
        )
        self.assertAlmostEqual(outcome.gum_interval[1], 1.959964 * 2.0, places=4)
        # Endpoint differences equal the stated formulas (19) and (20).
        self.assertAlmostEqual(
            outcome.d_low, abs(outcome.gum_interval[0] - result.symmetric_interval[0])
        )
        self.assertAlmostEqual(
            outcome.d_high, abs(outcome.gum_interval[1] - result.symmetric_interval[1])
        )

    def test_nonlinear_model_is_rejected(self) -> None:
        """``Y = exp(X)``, ``X ~ N(0, 0.5^2)``: the linearised interval fails.

        The GUM gives ``1 +/- 0.98``; the exact 95 % interval is
        ``exp(+/- 0.98) = [0.375, 2.66]``.

        References:
            [REF-JCGM-101-2008, Section 8.1]. Checks [EQ-UQ-BUDGET-009].

        """
        result = monte_carlo_budget(
            lambda x: math.exp(x[0]), [Normal(0.0, 0.5)], size=60000, seed=13
        )
        outcome = compare_with_law_of_propagation(
            1.0, 0.5, result, significant_digits=2
        )
        self.assertFalse(outcome.validated)
        self.assertGreater(outcome.d_low, outcome.delta)
        self.assertAlmostEqual(
            outcome.monte_carlo_interval[0], math.exp(-0.98), delta=0.02
        )
        shortest = compare_with_law_of_propagation(
            1.0, 0.5, result, significant_digits=2, interval="shortest"
        )
        self.assertEqual(shortest.interval, "shortest")
        self.assertFalse(shortest.validated)

    def test_decision_rule_on_constructed_result(self) -> None:
        """Both endpoint differences must be at most ``delta`` (Eqs. 19 and 20).

        References:
            [REF-JCGM-101-2008, Section 8.1.3]. Checks [EQ-UQ-BUDGET-009].

        """
        base = monte_carlo_budget(lambda x: x[0], [Normal(0.0, 2.0)], size=2000, seed=1)

        def result_with(low: float, high: float) -> object:
            """Return ``base`` with chosen symmetric-interval endpoints.

            References:
                [REF-JCGM-101-2008, Section 8.1.3]. No mathematical equation
                is evaluated.

            """
            return replace(
                base,
                symmetric_interval=(low, high),
                coverage=0.95,
            )

        k = NormalDist().inv_cdf(0.975)
        gum_edge = 2.0 * k
        inside = result_with(-gum_edge + 0.04, gum_edge - 0.04)
        outside = result_with(-gum_edge + 0.04, gum_edge - 0.06)
        self.assertTrue(
            compare_with_law_of_propagation(0.0, 2.0, inside).validated  # type: ignore[arg-type]
        )
        self.assertFalse(
            compare_with_law_of_propagation(0.0, 2.0, outside).validated  # type: ignore[arg-type]
        )

    def test_validation(self) -> None:
        """Argument errors.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        result = monte_carlo_budget(
            lambda x: x[0], [Normal(0.0, 1.0)], size=200, seed=1
        )
        with self.assertRaises(TypeError):
            compare_with_law_of_propagation(0.0, 1.0, "x")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compare_with_law_of_propagation(0.0, 0.0, result)
        with self.assertRaises(ValueError):
            compare_with_law_of_propagation(0.0, 1.0, result, coverage_factor=-1.0)
        with self.assertRaises(ValueError):
            compare_with_law_of_propagation(0.0, 1.0, result, interval="widest")


def binomial_success_threshold(trials: int, probability: float, alpha: float) -> int:
    """Return the largest ``k`` with ``P(Binomial(trials, probability) < k) <= alpha``.

    Observing at least ``k`` successes is then not evidence against a true
    success probability of ``probability`` at the level ``alpha``.

    References:
        Elementary binomial law; used with [REF-JCGM-101-2008, Section 7.9.3].
        Checks [EQ-UQ-BUDGET-010].

    """
    cumulative = 0.0
    threshold = 0
    for k in range(trials + 1):
        # cumulative is P(X < k) here
        if cumulative <= alpha:
            threshold = k
        cumulative += (
            math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
        )
    return threshold


def replay_adaptive(
    batches: list[list[float]],
    coverage: float,
    significant_digits: int,
    tolerance_divisor: float = 1.0,
    interval: str = "symmetric",
) -> tuple[int, bool, float, tuple[float, float, float, float]]:
    """Replay the clause 7.9.4 loop on recorded batch outputs, independently.

    Uses :mod:`statistics` and ``sample_coverage_intervals`` instead of the
    library's procedure; returns the number of batches used, whether it
    stabilised, the final ``delta`` and the four deviations.

    References:
        [REF-JCGM-101-2008, Section 7.9.4]. Checks [EQ-UQ-BUDGET-010].

    """
    estimates: list[float] = []
    uncertainties: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    pooled: list[float] = []
    delta = math.inf
    deviations = (math.inf,) * 4
    for h, values in enumerate(batches, start=1):
        pooled.extend(values)
        estimates.append(statistics.fmean(values))
        uncertainties.append(statistics.stdev(values))
        symmetric, shortest = sample_coverage_intervals(values, coverage)
        low, high = symmetric if interval == "symmetric" else shortest
        lows.append(low)
        highs.append(high)
        if h == 1:
            continue
        delta = numerical_tolerance(statistics.stdev(pooled), significant_digits)
        delta /= tolerance_divisor
        deviations = (
            statistics.stdev(estimates) / math.sqrt(h),
            statistics.stdev(uncertainties) / math.sqrt(h),
            statistics.stdev(lows) / math.sqrt(h),
            statistics.stdev(highs) / math.sqrt(h),
        )
        if all(2.0 * s <= delta for s in deviations):
            return h, True, delta, deviations
    return len(batches), False, delta, deviations


def additive_model_arguments() -> tuple[object, list[Uniform], float]:
    """Return the additive rectangular model, its inputs and the exact endpoint.

    References:
        [REF-JCGM-101-2008, Section 9.2.3]. Checks [EQ-UQ-BUDGET-010].

    """
    half = math.sqrt(3.0)
    return (
        lambda x: sum(x),
        [Uniform(-half, half)] * 4,
        sum_of_rectangles_quantile(0.975, half, 4),
    )


class AdaptiveStoppingRuleTests(unittest.TestCase):
    """The pure stopping rule on hand-built sequences of batch statistics.

    References:
        [REF-JCGM-101-2008, Section 7.9.4, steps g, h and k]. Checks
        [EQ-UQ-BUDGET-010].

    """

    def test_average_standard_deviation_by_hand(self) -> None:
        """``s^2 = sum (v - mean)^2 / (h (h - 1))`` on small hand cases.

        References:
            [REF-JCGM-101-2008, Section 7.9.4, step g]. Checks
            [EQ-UQ-BUDGET-010].

        """
        # Values 0 and 2: mean 1, sum of squares 2, s^2 = 2 / (2 * 1) = 1.
        self.assertAlmostEqual(average_standard_deviation((0.0, 2.0)), 1.0)
        # Values 1.0, 1.2, 1.1: sum of squares 0.02, s^2 = 0.02 / 6.
        self.assertAlmostEqual(
            average_standard_deviation((1.0, 1.2, 1.1)), math.sqrt(0.02 / 6.0)
        )
        # Equal values give zero; the sample standard deviation over sqrt(h).
        self.assertEqual(average_standard_deviation((3.0, 3.0, 3.0)), 0.0)
        data = (0.3, 0.9, 0.4, 0.8, 0.1)
        self.assertAlmostEqual(
            average_standard_deviation(data),
            statistics.stdev(data) / math.sqrt(len(data)),
        )
        with self.assertRaises(ValueError):
            average_standard_deviation((1.0,))
        with self.assertRaises(ValueError):
            average_standard_deviation((1.0, math.nan))

    def test_rule_boundary_is_inclusive(self) -> None:
        """The procedure continues only if ``2 s`` *exceeds* ``delta``.

        References:
            [REF-JCGM-101-2008, Section 7.9.4, step k]. Checks
            [EQ-UQ-BUDGET-010].

        """
        self.assertTrue(adaptive_stopping_rule((1.0,), 2.0))
        self.assertFalse(adaptive_stopping_rule((1.0,), 1.999))
        self.assertTrue(adaptive_stopping_rule((0.0, 0.0, 0.0, 0.0), 1e-12))
        # One value over the line keeps the whole computation going.
        self.assertFalse(adaptive_stopping_rule((0.1, 0.1, 0.1, 0.6), 1.0))
        self.assertTrue(adaptive_stopping_rule((0.1, 0.1, 0.1, 0.5), 1.0))

    def test_rule_on_a_hand_built_sequence_of_batch_statistics(self) -> None:
        """The first ``h`` at which all four ``2 s`` are at most ``delta``.

        Per-batch estimates 1.0, 1.2, 1.1, 1.1 give, by hand with
        ``delta = 0.1``: ``h = 2``: ``s = 0.1``, ``2 s = 0.2`` (continue);
        ``h = 3``: ``s = sqrt(0.02/6) = 0.0577``, ``2 s = 0.1155``
        (continue); ``h = 4``: ``s = sqrt(0.02/12) = 0.0408``, ``2 s = 0.0816``
        (stop) -- provided the other three statistics are quiet. A noisier
        right endpoint postpones the stop to the batch where it settles.

        References:
            [REF-JCGM-101-2008, Section 7.9.4, steps f-l]. Checks
            [EQ-UQ-BUDGET-010].

        """
        estimates = (1.0, 1.2, 1.1, 1.1)
        quiet = (5.0, 5.0, 5.0, 5.0)
        delta = 0.1

        def first_stop(columns: tuple[tuple[float, ...], ...]) -> int | None:
            """Return the first ``h >= 2`` at which the rule stops.

            References:
                [REF-JCGM-101-2008, Section 7.9.4, step k]. Checks
                [EQ-UQ-BUDGET-010].

            """
            for h in range(2, len(columns[0]) + 1):
                deviations = [average_standard_deviation(c[:h]) for c in columns]
                if adaptive_stopping_rule(deviations, delta):
                    return h
            return None

        self.assertAlmostEqual(average_standard_deviation(estimates[:2]), 0.1)
        self.assertEqual(first_stop((estimates, quiet, quiet, quiet)), 4)
        # The right endpoint 5.0, 5.2, 5.0, 5.2, 5.1, 5.1 gives by hand:
        # h = 2: 2 s = 0.2; h = 3: s = sqrt(0.02667/6) = 0.0667, 2 s = 0.133;
        # h = 4: s = sqrt(0.04/12) = 0.0577, 2 s = 0.1155 (all continue);
        # h = 5: s = sqrt(0.04/20) = 0.0447, 2 s = 0.0894 (stop).
        wandering = (5.0, 5.2, 5.0, 5.2, 5.1, 5.1)
        longer = estimates + (1.1, 1.1)
        self.assertEqual(
            first_stop((longer, quiet + (5.0, 5.0), quiet + (5.0, 5.0), wandering)),
            5,
        )
        # Never quiet enough: no stop (the caller then reports stabilized=False).
        loud = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        self.assertIsNone(first_stop((loud, loud, loud, loud)))

    def test_rule_validation(self) -> None:
        """Argument errors of the pure functions.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            adaptive_stopping_rule((), 1.0)
        with self.assertRaises(ValueError):
            adaptive_stopping_rule((0.1,), 0.0)
        with self.assertRaises(ValueError):
            adaptive_stopping_rule((-0.1,), 1.0)
        with self.assertRaises(ValueError):
            adaptive_stopping_rule((math.inf,), 1.0)


class AdaptiveMonteCarloTests(unittest.TestCase):
    """The adaptive procedure judged against closed forms and a replay.

    References:
        [REF-JCGM-101-2008, Sections 7.9.2-7.9.4 and 9.2]. Checks
        [EQ-UQ-BUDGET-010].

    """

    def test_additive_model_lands_within_its_own_tolerance(self) -> None:
        """Irwin-Hall endpoints are met to ``delta`` in a stated fraction of seeds.

        With ``ndig = 2`` and ``u(y) = 2`` the tolerance is ``delta = 0.05``.
        The clause promises no probability (7.9.3, Note 1), and stopping at
        the first of two batches can be lucky; a fraction of 0.84 of seeds
        with *both* endpoints within ``delta`` was measured over 200 seeds
        when this test was written. The assertion is therefore a one-sided
        binomial one against a deliberately lower success probability 0.70:
        the count of successes in 30 seeds must reach the threshold that a
        true rate of 0.70 would fall below with probability at most 0.001.

        References:
            [REF-JCGM-101-2008, Sections 7.9.3 and 9.2.3]. Checks
            [EQ-UQ-BUDGET-010].

        """
        function, laws, exact = additive_model_arguments()
        seeds = 30
        successes = 0
        for seed in range(seeds):
            result = adaptive_monte_carlo_budget(
                function,  # type: ignore[arg-type]
                laws,
                seed=1000 + seed,
            )
            self.assertTrue(result.stabilized)
            self.assertAlmostEqual(result.delta, 0.05, places=12)
            self.assertEqual(result.size, result.batches * result.batch_size)
            low, high = result.symmetric_interval
            if abs(low + exact) <= result.delta and abs(high - exact) <= result.delta:
                successes += 1
        self.assertGreaterEqual(
            successes, binomial_success_threshold(seeds, 0.70, 1e-3)
        )

    def test_lognormal_product_lands_within_its_own_tolerance(self) -> None:
        """The exact lognormal interval is met to ``delta`` in a stated fraction.

        ``Y = X1 X2 / X3`` with lognormal inputs is lognormal with ``mu =
        -0.4`` and ``sigma = sqrt(0.0469)``; the 95 % endpoints are
        ``exp(mu -/+ 1.96 sigma)``. A fraction 0.86 of 200 seeds was
        measured; the assertion uses the same lower-rate binomial threshold
        as the additive case.

        References:
            [REF-JCGM-101-2008, Sections 5.9 and 7.9.3]. Checks
            [EQ-UQ-BUDGET-010].

        """
        mus = (0.1, -0.2, 0.3)
        sigmas = (0.10, 0.15, 0.12)
        law = NormalDist(
            mus[0] + mus[1] - mus[2], math.sqrt(sum(s * s for s in sigmas))
        )
        exact = (math.exp(law.inv_cdf(0.025)), math.exp(law.inv_cdf(0.975)))
        seeds = 30
        successes = 0
        for seed in range(seeds):
            result = adaptive_monte_carlo_budget(
                lambda x: x[0] * x[1] / x[2],
                [LogNormal(m, s) for m, s in zip(mus, sigmas, strict=True)],
                seed=2000 + seed,
            )
            self.assertTrue(result.stabilized)
            low, high = result.symmetric_interval
            if (
                abs(low - exact[0]) <= result.delta
                and abs(high - exact[1]) <= result.delta
            ):
                successes += 1
        self.assertGreaterEqual(
            successes, binomial_success_threshold(seeds, 0.70, 1e-3)
        )

    def test_loop_matches_an_independent_replay(self) -> None:
        """Recorded batches, replayed with ``statistics``, give the same stop.

        The batches are recorded by driving the procedure through an
        ask-tell session; the replay recomputes every per-batch quantity,
        the pooled ``delta`` and the four deviations without the library's
        loop, for both coverage-interval kinds and with ``delta / 5``.

        References:
            [REF-JCGM-101-2008, Section 7.9.4, steps d-l]. Checks
            [EQ-UQ-BUDGET-010].

        """
        cases = (
            ("symmetric", 1.0, 2),
            ("shortest", 1.0, 2),
            ("symmetric", 5.0, 1),
        )
        for kind, divisor, ndig in cases:
            with self.subTest(interval=kind, divisor=divisor):
                procedure = adaptive_monte_carlo_budget_procedure(
                    [Normal(0.0, 0.5)],
                    significant_digits=ndig,
                    batch_size=2000,
                    max_batches=40,
                    seed=11,
                    interval=kind,
                    tolerance_divisor=divisor,
                )
                session = AskTellSession(procedure)
                batches: list[list[float]] = []
                while (pending := session.ask()) is not None:
                    outputs = [math.exp(point[0]) for point in pending.points]
                    batches.append(outputs)
                    session.tell(outputs)
                result = session.result
                h, stable, delta, deviations = replay_adaptive(
                    batches, 0.95, ndig, divisor, kind
                )
                self.assertEqual(result.batches, h)
                self.assertEqual(len(batches), h)
                self.assertEqual(result.stabilized, stable)
                self.assertAlmostEqual(result.delta, delta, places=12)
                for found, expected in zip(
                    (
                        result.estimate_deviation,
                        result.uncertainty_deviation,
                        result.low_deviation,
                        result.high_deviation,
                    ),
                    deviations,
                    strict=True,
                ):
                    self.assertAlmostEqual(found, expected, places=12)
                everything = [v for batch in batches for v in batch]
                self.assertAlmostEqual(result.estimate, statistics.fmean(everything))
                self.assertAlmostEqual(
                    result.standard_uncertainty, statistics.stdev(everything)
                )
                self.assertEqual(result.interval, kind)
                self.assertEqual(result.size, len(everything))

    def test_not_stabilized_when_max_batches_is_too_small(self) -> None:
        """Running out of batches is reported, never raised or hidden.

        Three significant digits need far more than two batches; the record
        says ``stabilized=False`` and its deviations are over the line.

        References:
            [REF-JCGM-101-2008, Section 7.9.3, Note 1]. Checks
            [EQ-UQ-BUDGET-010].

        """
        function, laws, _ = additive_model_arguments()
        result = adaptive_monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            significant_digits=3,
            max_batches=2,
            seed=5,
        )
        self.assertFalse(result.stabilized)
        self.assertEqual(result.batches, 2)
        self.assertEqual(result.size, 20000)
        self.assertAlmostEqual(result.delta, 0.005, places=12)
        deviations = (
            result.estimate_deviation,
            result.uncertainty_deviation,
            result.low_deviation,
            result.high_deviation,
        )
        self.assertFalse(adaptive_stopping_rule(deviations, result.delta))
        # The numbers are still those of all 20 000 trials.
        self.assertAlmostEqual(
            result.standard_uncertainty,
            2.0,
            delta=5.0 * result.uncertainty_standard_error,
        )

    def test_scheduling_independence(self) -> None:
        """Same seed, different chunking of the same batches: identical results.

        Each batch draws from its own child stream, so evaluating a request
        whole, in chunks of any size, through an ask-tell session answered
        piecewise, or through the plain function changes nothing, bit for bit.

        References:
            [REF-JCGM-101-2008, Section 7.9.4]; [EQ-BATCH-001]. Checks
            [EQ-UQ-BUDGET-010].

        """
        laws = [Uniform(-1.0, 1.0), Normal(0.0, 0.5), Exponential(2.0)]

        def model(x: Sequence[float]) -> float:
            """Return a smooth nonlinear model of three inputs.

            References:
                Test fixture; no mathematical equation of the library.

            """
            return x[0] * x[1] + math.exp(0.3 * x[2])

        settings = {
            "significant_digits": 2,
            "batch_size": 1500,
            "max_batches": 12,
            "seed": 77,
        }
        plain = adaptive_monte_carlo_budget(model, laws, **settings)  # type: ignore[arg-type]
        batch_model = batch_from_pointwise(model)

        def run(chunk: int | None) -> AdaptiveMonteCarloResult:
            """Run the procedure with batches of at most ``chunk`` points.

            References:
                [EQ-BATCH-001]. Checks [EQ-UQ-BUDGET-010].

            """
            return run_procedure(
                adaptive_monte_carlo_budget_procedure(laws, **settings),  # type: ignore[arg-type]
                batch_model,
                max_batch_size=chunk,
            )

        self.assertEqual(run(None), plain)
        for chunk in (1, 7, 500, 1499, 1500, 4000):
            with self.subTest(chunk=chunk):
                self.assertEqual(run(chunk), plain)
        session = AskTellSession(
            adaptive_monte_carlo_budget_procedure(laws, **settings)
        )  # type: ignore[arg-type]
        while (pending := session.ask()) is not None:
            half = pending.size // 2
            first = batch_model(pending.points[:half])
            second = batch_model(pending.points[half:])
            session.tell([*first, *second])
        self.assertEqual(session.result, plain)
        self.assertEqual(session.evaluations["model"], plain.size)
        self.assertEqual(session.requests, plain.batches)

    def test_batches_do_not_depend_on_how_many_follow(self) -> None:
        """The points of batch ``b`` depend on ``(seed, b)`` only.

        References:
            [EQ-RNG-SPLITMIX-001]. Checks [EQ-UQ-BUDGET-010].

        """
        laws = [Normal(0.0, 1.0), Normal(1.0, 2.0)]

        def first_requests(limit: int) -> list[tuple[tuple[float, ...], ...]]:
            """Return the points of the first two requests.

            References:
                [EQ-BATCH-001]. Checks [EQ-UQ-BUDGET-010].

            """
            session = AskTellSession(
                adaptive_monte_carlo_budget_procedure(
                    laws, batch_size=300, max_batches=limit, seed=3
                )
            )
            points = []
            for _ in range(2):
                pending = session.ask()
                assert pending is not None
                points.append(pending.points)
                session.tell([p[0] + p[1] for p in pending.points])
            return points

        self.assertEqual(first_requests(2), first_requests(9))
        # A different seed gives different points.
        other = AskTellSession(
            adaptive_monte_carlo_budget_procedure(
                laws, batch_size=300, max_batches=2, seed=4
            )
        ).ask()
        assert other is not None
        self.assertNotEqual(other.points, first_requests(2)[0])

    def test_stream_seed_and_copula(self) -> None:
        """A stream seed works, and a copula correlation reaches the batches.

        ``Y = X1 + X2`` with correlation ``r = 0.5`` of normal scores has
        standard deviation ``sqrt(2 (1 + r))`` for normal marginals.

        References:
            [REF-JCGM-101-2008, Section 6.4.8]; [REF-NELSEN-2006]. Checks
            [EQ-UQ-BUDGET-010].

        """
        result = adaptive_monte_carlo_budget(
            lambda x: x[0] + x[1],
            [Normal(0.0, 1.0), Normal(0.0, 1.0)],
            seed=RandomStream(5),
            correlation=((1.0, 0.5), (0.5, 1.0)),
            significant_digits=1,
        )
        self.assertTrue(result.correlated)
        self.assertAlmostEqual(
            result.standard_uncertainty,
            math.sqrt(3.0),
            delta=5.0 * result.uncertainty_standard_error,
        )
        self.assertEqual(
            result,
            adaptive_monte_carlo_budget(
                lambda x: x[0] + x[1],
                [Normal(0.0, 1.0), Normal(0.0, 1.0)],
                seed=RandomStream(5),
                correlation=((1.0, 0.5), (0.5, 1.0)),
                significant_digits=1,
            ),
        )

    def test_clause_batch_size_and_divisor(self) -> None:
        """``M = max(ceil(100/(1 - p)), 10^4)`` and ``delta / 5`` of Section 8.2.

        References:
            [REF-JCGM-101-2008, Section 7.9.4, step b, and Section 8.2]. Checks
            [EQ-UQ-BUDGET-010].

        """
        function, laws, _ = additive_model_arguments()
        usual = adaptive_monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            significant_digits=1,
            seed=8,
        )
        self.assertEqual(usual.batch_size, 10_000)
        self.assertAlmostEqual(usual.delta, 0.5, places=12)
        strict = adaptive_monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            significant_digits=1,
            tolerance_divisor=5.0,
            seed=8,
        )
        self.assertAlmostEqual(strict.delta, 0.1, places=12)
        self.assertEqual(strict.tolerance_divisor, 5.0)
        self.assertGreaterEqual(strict.batches, usual.batches)
        # p = 0.999: J = 100 / 0.001 = 100 000 exceeds 10^4 (7.9.4 b).
        procedure = adaptive_monte_carlo_budget_procedure(
            [Normal(0.0, 1.0)], coverage=0.999, seed=1
        )
        first = AskTellSession(procedure).ask()
        assert first is not None
        self.assertEqual(first.size, 100_000)

    def test_failures_of_the_model(self) -> None:
        """A non-finite value or a constant model is an error, not a result.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION];
            [REF-JCGM-101-2008, Section 7.9.2] (``delta`` needs ``u(y) > 0``).

        """
        with self.assertRaises(ValueError):
            adaptive_monte_carlo_budget(
                lambda x: math.nan, [Normal(0.0, 1.0)], batch_size=100, seed=1
            )
        with self.assertRaises(ValueError):
            adaptive_monte_carlo_budget(
                lambda x: 1.0, [Normal(0.0, 1.0)], batch_size=100, seed=1
            )
        with self.assertRaises(ValueError):
            run_procedure(
                adaptive_monte_carlo_budget_procedure(
                    [Normal(0.0, 1.0)], batch_size=100, seed=1
                ),
                lambda points: [0.0],
            )

    def test_validation(self) -> None:
        """Arguments are checked when the function or procedure is called.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        laws = [Normal(0.0, 1.0)]

        def procedure(**changes: object) -> object:
            """Build the procedure with one argument replaced.

            References:
                Project validation policy [REF-PROBSTAT-VALIDATION].

            """
            arguments: dict[str, object] = {"seed": 1}
            arguments.update(changes)
            return adaptive_monte_carlo_budget_procedure(laws, **arguments)  # type: ignore[arg-type]

        self.assertIsNotNone(procedure())
        for changes in (
            {"significant_digits": 0},
            {"coverage": 1.0},
            {"coverage": 0.0},
            {"batch_size": 10},
            {"batch_size": 20, "coverage": 0.99},
            {"max_batches": 1},
            {"interval": "widest"},
            {"tolerance_divisor": 0.0},
            {"label": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                procedure(**changes)
        with self.assertRaises(TypeError):
            procedure(seed="a")
        with self.assertRaises(TypeError):
            adaptive_monte_carlo_budget_procedure([object()], seed=1)  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            adaptive_monte_carlo_budget_procedure([], seed=1)
        with self.assertRaises(TypeError):
            adaptive_monte_carlo_budget("f", laws, seed=1)  # type: ignore[arg-type]


class AdaptiveValidationTests(unittest.TestCase):
    """The Section 8.2 requirement read from the stabilisation statistics.

    References:
        [REF-JCGM-101-2008, Sections 8.1.3 and 8.2]. Checks
        [EQ-UQ-BUDGET-009] and [EQ-UQ-BUDGET-010].

    """

    def test_adequacy_uses_the_stabilisation_deviations(self) -> None:
        """Adequate exactly when twice each deviation is at most ``delta / 5``.

        References:
            [REF-JCGM-101-2008, Section 8.2]. Checks [EQ-UQ-BUDGET-009].

        """
        function, laws, _ = additive_model_arguments()
        result = adaptive_monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            significant_digits=1,
            tolerance_divisor=5.0,
            seed=21,
        )
        self.assertTrue(result.stabilized)
        outcome = compare_with_law_of_propagation(
            0.0, 2.0, result, significant_digits=1
        )
        self.assertEqual(outcome.delta, 0.5)
        self.assertTrue(outcome.validated)
        self.assertTrue(outcome.monte_carlo_adequate)
        self.assertEqual(outcome.adequacy_basis, ADEQUACY_STABILIZATION)
        # delta / 5 = 0.1: a deviation of 0.05 passes, one above does not.
        quiet = replace(
            result,
            estimate_deviation=0.01,
            uncertainty_deviation=0.01,
            low_deviation=0.01,
            high_deviation=0.05,
        )
        noisy = replace(quiet, high_deviation=0.0501)
        self.assertTrue(
            compare_with_law_of_propagation(
                0.0, 2.0, quiet, significant_digits=1
            ).monte_carlo_adequate
        )
        self.assertFalse(
            compare_with_law_of_propagation(
                0.0, 2.0, noisy, significant_digits=1
            ).monte_carlo_adequate
        )
        # A run to the plain tolerance (delta, not delta / 5) is not enough.
        plain_run = replace(
            result,
            estimate_deviation=0.2,
            uncertainty_deviation=0.2,
            low_deviation=0.2,
            high_deviation=0.2,
        )
        self.assertFalse(
            compare_with_law_of_propagation(
                0.0, 2.0, plain_run, significant_digits=1
            ).monte_carlo_adequate
        )

    def test_other_interval_is_not_assessed(self) -> None:
        """An adaptive run that tracked the other interval is not called adequate.

        References:
            [REF-JCGM-101-2008, Section 8.2]. Checks [EQ-UQ-BUDGET-009].

        """
        function, laws, _ = additive_model_arguments()
        result = adaptive_monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            significant_digits=1,
            seed=22,
        )
        outcome = compare_with_law_of_propagation(
            0.0, 2.0, result, significant_digits=1, interval="shortest"
        )
        self.assertFalse(outcome.monte_carlo_adequate)
        self.assertEqual(outcome.adequacy_basis, ADEQUACY_OTHER_INTERVAL)

    def test_fixed_size_result_keeps_the_proxy(self) -> None:
        """A plain :class:`MonteCarloResult` still gets the standard-error proxy.

        References:
            [REF-JCGM-101-2008, Section 8.2]. Checks [EQ-UQ-BUDGET-009].

        """
        function, laws, _ = additive_model_arguments()
        result = monte_carlo_budget(
            function,  # type: ignore[arg-type]
            laws,
            size=20000,
            seed=3,
        )
        outcome = compare_with_law_of_propagation(
            0.0, 2.0, result, significant_digits=1
        )
        self.assertEqual(outcome.adequacy_basis, ADEQUACY_PROXY)


if __name__ == "__main__":
    unittest.main()
