"""Tests for component time models and common-cause failure models.

Oracles: closed forms; ``scipy.linalg.expm`` of the two-state generator and
``mpmath`` at 50 digits (values pasted from the verification environment);
an independent numerical integration with ``probstat.numerics``; the
NUREG/CR-5485 normalisation identity; and brute-force enumeration over all
independent CCF basic events.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.distributions import LogNormal, Weibull
from probstat.numerics import integrate
from probstat.systems import (
    AlphaFactorGroup,
    BetaFactorGroup,
    ComponentModel,
    DemandComponent,
    ExponentialComponent,
    FixedProbability,
    LifeDistributionComponent,
    MultipleGreekLetterGroup,
    PeriodicallyTestedComponent,
    RepairableComponent,
    SystemStructure,
    WeibullComponent,
    expand_common_cause,
    k_out_of_n,
    normalisation,
    order_probabilities,
    parallel,
    series,
)


class ComponentModelTests(unittest.TestCase):
    """Closed forms and accuracy of the component models."""

    def test_protocol_and_simple_models(self) -> None:
        """Every model implements the protocol; simple models are constant."""
        models = [
            FixedProbability(0.9),
            DemandComponent(0.01),
            ExponentialComponent(1e-3),
            WeibullComponent(2.0, 100.0),
            LifeDistributionComponent(LogNormal(3.0, 0.5)),
            RepairableComponent(0.01, 0.1),
            PeriodicallyTestedComponent(1e-3, 100.0),
        ]
        for model in models:
            self.assertIsInstance(model, ComponentModel)
            for t in (0.0, 3.0, 250.0):
                self.assertAlmostEqual(
                    model.reliability(t) + model.unreliability(t), 1.0, delta=1e-15
                )
                self.assertAlmostEqual(
                    model.availability(t) + model.unavailability(t), 1.0, delta=1e-15
                )
        self.assertEqual(FixedProbability(0.9).reliability(5.0), 0.9)
        self.assertEqual(DemandComponent(0.01).unavailability(5.0), 0.01)

    def test_exponential_and_weibull(self) -> None:
        """exp(-lambda t), exp(-(t/eta)^beta), and tiny unreliabilities."""
        model = ExponentialComponent(2e-3)
        self.assertAlmostEqual(model.reliability(150.0), math.exp(-0.3), delta=1e-16)
        tiny = ExponentialComponent(1e-12)
        self.assertAlmostEqual(tiny.unreliability(1.0) / 1e-12, 1.0, delta=1e-12)
        self.assertEqual(model.reliability(math.inf), 0.0)
        self.assertEqual(ExponentialComponent(0.0).reliability(math.inf), 1.0)
        weibull = WeibullComponent(1.5, 200.0)
        self.assertAlmostEqual(
            weibull.reliability(100.0), math.exp(-(0.5**1.5)), delta=1e-16
        )
        law = LifeDistributionComponent(Weibull(1.5, 200.0))
        for t in (10.0, 100.0, 500.0):
            self.assertAlmostEqual(
                law.reliability(t), weibull.reliability(t), delta=1e-14
            )
            self.assertAlmostEqual(
                law.unreliability(t), weibull.unreliability(t), delta=1e-14
            )
        self.assertEqual(law.reliability(math.inf), 0.0)

    def test_repairable_availability(self) -> None:
        """A(t) equals expm of the generator and solves the Markov ODE.

        Reference values: scipy.linalg.expm([[-0.01, 0.01], [0.1, -0.1]] t)
        entry (0, 0) at t = 0.5, 7.5, 40.
        """
        model = RepairableComponent(0.01, 0.1)
        for t, expected in (
            (0.5, 0.99513501345031674),
            (7.5, 0.94893045386044994),
            (40.0, 0.9102070309002791),
        ):
            self.assertAlmostEqual(model.availability(t), expected, delta=2e-16)
        self.assertEqual(model.availability(0.0), 1.0)
        self.assertAlmostEqual(model.availability(math.inf), 0.1 / 0.11, delta=1e-16)
        self.assertAlmostEqual(model.steady_state_availability, 0.1 / 0.11, delta=1e-16)
        h = 1e-4
        for t in (1.0, 20.0):
            derivative = (model.availability(t + h) - model.availability(t - h)) / (
                2 * h
            )
            a = model.availability(t)
            self.assertAlmostEqual(derivative, -0.01 * a + 0.1 * (1 - a), delta=1e-9)
        self.assertAlmostEqual(model.reliability(10.0), math.exp(-0.1), delta=1e-16)

    def test_periodic_test_average_unavailability(self) -> None:
        """Exact average 1 - (1 - e^-x)/x against mpmath and quadrature.

        Reference values: mpmath at 50 digits.
        """
        for x, expected in (
            (1e-6, 4.9999983333337499999e-7),
            (0.003, 0.0014985011243253373554),
            (0.3, 0.13606073560572622022),
            (0.49, 0.20944162078452258977),
            (0.5, 0.21306131942526684721),
            (2.0, 0.56766764161830634595),
        ):
            model = PeriodicallyTestedComponent(x / 10.0, 10.0)
            self.assertAlmostEqual(
                model.average_unavailability() / expected, 1.0, delta=1e-14
            )
            self.assertAlmostEqual(
                model.average_unavailability_approximation(), x / 2, delta=1e-17
            )
            self.assertGreaterEqual(
                model.average_unavailability_approximation(), expected
            )
        model = PeriodicallyTestedComponent(0.02, 30.0)
        average = (
            integrate(model.unavailability, 30.0, 60.0, rel_tol=1e-12).value / 30.0
        )
        self.assertAlmostEqual(average, model.average_unavailability(), delta=1e-12)

    def test_periodic_test_sawtooth_and_staggering(self) -> None:
        """Unavailability restarts at each test; first_test shifts the tests."""
        model = PeriodicallyTestedComponent(0.01, 100.0)
        self.assertAlmostEqual(
            model.unavailability(50.0), -math.expm1(-0.5), delta=1e-16
        )
        self.assertAlmostEqual(
            model.unavailability(150.0), -math.expm1(-0.5), delta=1e-16
        )
        self.assertEqual(model.unavailability(100.0), 0.0)
        staggered = PeriodicallyTestedComponent(0.01, 100.0, first_test=50.0)
        self.assertAlmostEqual(
            staggered.unavailability(120.0), -math.expm1(-0.7), delta=1e-16
        )
        self.assertAlmostEqual(
            staggered.unavailability(40.0), -math.expm1(-0.4), delta=1e-16
        )
        self.assertAlmostEqual(model.reliability(250.0), math.exp(-2.5), delta=1e-16)
        with self.assertRaises(ValueError):
            model.unavailability(math.inf)

    def test_validation(self) -> None:
        """Booleans, negative values and bad laws are rejected."""
        with self.assertRaises(TypeError):
            FixedProbability(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FixedProbability(1.5)
        with self.assertRaises(ValueError):
            ExponentialComponent(-1.0)
        with self.assertRaises(ValueError):
            WeibullComponent(0.0, 1.0)
        with self.assertRaises(ValueError):
            RepairableComponent(0.1, 0.0)
        with self.assertRaises(ValueError):
            PeriodicallyTestedComponent(0.1, -1.0)
        with self.assertRaises(TypeError):
            LifeDistributionComponent(object())
        with self.assertRaises(ValueError):
            ExponentialComponent(1.0).reliability(-1.0)
        with self.assertRaises(TypeError):
            ExponentialComponent(1.0).reliability(True)  # type: ignore[arg-type]


class CCFModelTests(unittest.TestCase):
    """Parametric CCF models and their exact expansion."""

    def test_normalisation_of_every_model(self) -> None:
        """sum_k C(m-1, k-1) Q_k = Q_t for random parameters and m = 2..6."""
        rng = random.Random(2)
        for m in range(2, 7):
            members = tuple(f"c{i}" for i in range(m))
            weights = [rng.random() for _ in range(m)]
            alphas = tuple(w / sum(weights) for w in weights)
            groups = [
                BetaFactorGroup(members, rng.random()),
                MultipleGreekLetterGroup(
                    members, tuple(rng.random() for _ in range(m - 1))
                ),
                AlphaFactorGroup(members, alphas, "non-staggered"),
                AlphaFactorGroup(members, alphas, "staggered"),
            ]
            for group in groups:
                self.assertAlmostEqual(normalisation(group), 1.0, delta=1e-14)
                q = order_probabilities(group, 0.013)
                total = math.fsum(
                    math.comb(m - 1, k - 1) * q[k - 1] for k in range(1, m + 1)
                )
                self.assertAlmostEqual(total, 0.013, delta=1e-16)

    def test_textbook_formulas(self) -> None:
        """Explicit three-member formulas of NUREG/CR-5485."""
        members = ("A", "B", "C")
        qt = 1e-3
        beta = order_probabilities(BetaFactorGroup(members, 0.1), qt)
        self.assertEqual(beta, ((1 - 0.1) * qt, 0.0, 0.1 * qt))
        b, g = 0.1, 0.3
        mgl = order_probabilities(MultipleGreekLetterGroup(members, (b, g)), qt)
        expected = ((1 - b) * qt, b * (1 - g) * qt / 2, b * g * qt)
        for got, want in zip(mgl, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-18)
        alphas = (0.95, 0.04, 0.01)
        at = 0.95 + 2 * 0.04 + 3 * 0.01
        alpha = order_probabilities(AlphaFactorGroup(members, alphas), qt)
        expected = (0.95 / at * qt, 2 / 2 * 0.04 / at * qt, 3 * 0.01 / at * qt)
        for got, want in zip(alpha, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-18)
        staggered = order_probabilities(
            AlphaFactorGroup(members, alphas, "staggered"), qt
        )
        expected = (0.95 * qt, 0.04 * qt / 2, 0.01 * qt)
        for got, want in zip(staggered, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-18)

    def test_staggered_alpha_equals_mgl_with_mapped_factors(self) -> None:
        """alpha_k = (prod_{i<=k} rho_i)(1 - rho_(k+1)) makes the two agree."""
        rho = (1.0, 0.2, 0.4, 0.5, 0.0)
        alphas = tuple(math.prod(rho[:k]) * (1 - rho[k]) for k in range(1, 5))
        members = ("a", "b", "c", "d")
        left = order_probabilities(AlphaFactorGroup(members, alphas, "staggered"), 0.01)
        right = order_probabilities(MultipleGreekLetterGroup(members, rho[1:4]), 0.01)
        for got, want in zip(left, right, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-17)

    def test_expansion_events_and_marginal(self) -> None:
        """Event names, counts, and the exact member marginal."""
        members = ("A", "B", "C")
        group = MultipleGreekLetterGroup(members, (0.1, 0.3))
        expansion = expand_common_cause(series(*members), [group])
        self.assertEqual(len(expansion.events), 7)
        self.assertEqual(
            expansion.component_events["A"],
            ("IND[A]", "CCF[A,B]", "CCF[A,C]", "CCF[A,B,C]"),
        )
        beta = expand_common_cause(parallel(*members), [BetaFactorGroup(members, 0.1)])
        self.assertEqual(
            sorted(beta.events), ["CCF[A,B,C]", "IND[A]", "IND[B]", "IND[C]"]
        )
        # Exact marginal of A: 1 - prod_k (1 - Q_k)^C(m-1, k-1) < Q_t.
        qt = 0.05
        q = order_probabilities(group, qt)
        expected = 1 - (1 - q[0]) * (1 - q[1]) ** 2 * (1 - q[2])
        structure = SystemStructure(
            expand_common_cause(series("A", "B", "C"), [group]).expression
        )
        only_a = SystemStructure(series(*(expansion.component_events["A"])))
        probabilities = {
            name: 1 - event.fraction * qt for name, event in expansion.events.items()
        }
        marginal = only_a.unreliability(
            {n: probabilities[n] for n in only_a.components}
        )
        self.assertAlmostEqual(marginal, expected, delta=1e-16)
        self.assertLess(marginal, qt)
        self.assertEqual(len(structure.components), 7)

    def brute_force(self, structure_expression, members, group, qt):
        """Return the exact system failure probability by event enumeration."""
        expansion = expand_common_cause(structure_expression, [group])
        events = list(expansion.events.values())
        total = 0.0
        terms = []
        for bits in itertools.product((0, 1), repeat=len(events)):
            weight = 1.0
            occurred = set()
            for event, bit in zip(events, bits, strict=True):
                probability = event.fraction * qt
                weight *= probability if bit else 1 - probability
                if bit:
                    occurred.update(event.members)
            state = {m: 0 if m in occurred else 1 for m in members}
            structure = SystemStructure(structure_expression)
            if not structure.structure_function(state):
                terms.append(weight)
        total = math.fsum(terms)
        return total

    def test_expanded_structure_matches_event_enumeration(self) -> None:
        """2-out-of-3 and 1-out-of-4 systems with each CCF model."""
        cases = [
            (
                k_out_of_n(2, "A", "B", "C"),
                ("A", "B", "C"),
                BetaFactorGroup(("A", "B", "C"), 0.08),
            ),
            (
                k_out_of_n(2, "A", "B", "C"),
                ("A", "B", "C"),
                MultipleGreekLetterGroup(("A", "B", "C"), (0.1, 0.27)),
            ),
            (
                parallel("A", "B", "C", "D"),
                ("A", "B", "C", "D"),
                AlphaFactorGroup(("A", "B", "C", "D"), (0.9, 0.06, 0.03, 0.01)),
            ),
        ]
        qt = 0.02
        for expression, members, group in cases:
            expansion = expand_common_cause(expression, [group])
            structure = SystemStructure(expansion.expression)
            q = {name: event.fraction * qt for name, event in expansion.events.items()}
            p = {name: 1 - value for name, value in q.items()}
            value = structure.unreliability(p, unreliabilities=q)
            self.assertAlmostEqual(
                value, self.brute_force(expression, members, group, qt), delta=1e-17
            )
        # Rare-event check for the 2oo3 beta case: Q ~ 3 Q_1^2 + Q_3.
        q1, q3 = 0.92 * qt, 0.08 * qt
        expansion = expand_common_cause(cases[0][0], [cases[0][2]])
        structure = SystemStructure(expansion.expression)
        q = {name: event.fraction * qt for name, event in expansion.events.items()}
        value = structure.unreliability(
            {k: 1 - v for k, v in q.items()}, unreliabilities=q
        )
        self.assertAlmostEqual(value / (3 * q1**2 + q3), 1.0, delta=0.02)

    def test_validation(self) -> None:
        """Invalid groups and expansions are rejected."""
        with self.assertRaises(ValueError):
            BetaFactorGroup(("A",), 0.1)
        with self.assertRaises(ValueError):
            BetaFactorGroup(("A", "A"), 0.1)
        with self.assertRaises(ValueError):
            MultipleGreekLetterGroup(("A", "B", "C"), (0.1,))
        with self.assertRaises(ValueError):
            AlphaFactorGroup(("A", "B"), (0.5, 0.6))
        with self.assertRaises(ValueError):
            AlphaFactorGroup(("A", "B"), (0.5, 0.5), "sometimes")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            expand_common_cause(series("A", "B"), [BetaFactorGroup(("A", "Z"), 0.1)])
        with self.assertRaises(ValueError):
            expand_common_cause(
                series("A", "B", "C"),
                [BetaFactorGroup(("A", "B"), 0.1), BetaFactorGroup(("B", "C"), 0.1)],
            )
        with self.assertRaises(TypeError):
            expand_common_cause(series("A", "B"), ["group"])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
