"""Tests for ``SystemModel`` and phased missions.

Oracles: MTTF closed forms (series ``1 / sum lambda``, parallel
``1/l1 + 1/l2 - 1/(l1 + l2)``, Weibull ``eta Gamma(1 + 1/beta)``); products
of component availabilities; exact moments versus a Monte Carlo propagation
through ``UQStudy``; and, for phased missions, brute-force enumeration over
every component's phase of failure with a direct evaluation of each phase
structure, plus closed forms for two-phase missions.
"""

from __future__ import annotations

import itertools
import math
import unittest

from probstat.distributions import Beta, Gamma
from probstat.modeling import ProbabilisticModel, RandomInput
from probstat.propagation import propagate
from probstat.systems import (
    BetaFactorGroup,
    Component,
    DemandComponent,
    ExponentialComponent,
    FixedProbability,
    Parallel,
    Phase,
    PhasedMission,
    RepairableComponent,
    Series,
    SystemModel,
    SystemStructure,
    WeibullComponent,
    bridge,
    k_out_of_n,
    parallel,
    series,
)
from probstat.workflow import UQStudy


def phi(expression, state) -> bool:
    """Evaluate a structure expression directly (oracle)."""
    if isinstance(expression, Component):
        return bool(state[expression.name])
    values = [phi(item, state) for item in expression.items]
    if isinstance(expression, Series):
        return all(values)
    if isinstance(expression, Parallel):
        return any(values)
    return sum(values) >= expression.k


class SystemModelTests(unittest.TestCase):
    """Mission reliability, availability, MTTF, and importance."""

    def test_mttf_closed_forms(self) -> None:
        """Series, parallel, and single Weibull MTTF to 1e-9 relative."""
        l1, l2, l3 = 1e-3, 2e-3, 5e-4
        series_model = SystemModel(
            series("A", "B", "C"),
            {
                "A": ExponentialComponent(l1),
                "B": ExponentialComponent(l2),
                "C": ExponentialComponent(l3),
            },
        )
        result = series_model.mttf()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.value * (l1 + l2 + l3), 1.0, delta=1e-9)
        self.assertLess(result.error_estimate, 1e-6 * result.value)
        parallel_model = SystemModel(
            parallel("A", "B"),
            {"A": ExponentialComponent(l1), "B": ExponentialComponent(l2)},
        )
        expected = 1 / l1 + 1 / l2 - 1 / (l1 + l2)
        self.assertAlmostEqual(parallel_model.mttf().value / expected, 1.0, delta=1e-9)
        weibull = SystemModel("A", {"A": WeibullComponent(2.5, 300.0)})
        self.assertAlmostEqual(
            weibull.mttf().value / (300.0 * math.gamma(1 + 1 / 2.5)), 1.0, delta=1e-9
        )
        # Two-out-of-three of identical exponentials: MTTF = 5 / (6 lambda).
        voting = SystemModel(
            k_out_of_n(2, "A", "B", "C"), dict.fromkeys("ABC", ExponentialComponent(l1))
        )
        self.assertAlmostEqual(voting.mttf().value * 6 * l1 / 5, 1.0, delta=1e-9)

    def test_infinite_mttf_is_rejected(self) -> None:
        """A component that never fails in parallel makes MTTF infinite."""
        model = SystemModel(
            parallel("A", "B"),
            {"A": FixedProbability(0.5), "B": ExponentialComponent(1e-3)},
        )
        with self.assertRaises(ValueError):
            model.mttf()

    def test_reliability_and_availability(self) -> None:
        """Mission reliability uses R_i(t); availability uses A_i(t)."""
        models = {
            "a": ExponentialComponent(1e-3),
            "b": RepairableComponent(2e-3, 0.1),
            "c": DemandComponent(0.01),
            "d": WeibullComponent(1.5, 2000.0),
            "e": RepairableComponent(1e-3, 0.05),
        }
        model = SystemModel(bridge(*"abcde"), models)
        t = 250.0
        p = {name: m.reliability(t) for name, m in models.items()}
        a = {name: m.availability(t) for name, m in models.items()}
        structure = SystemStructure(bridge(*"abcde"))
        self.assertAlmostEqual(
            model.reliability(t), structure.reliability(p), delta=1e-15
        )
        self.assertAlmostEqual(
            model.availability(t), structure.reliability(a), delta=1e-15
        )
        self.assertAlmostEqual(
            model.reliability(t) + model.unreliability(t), 1.0, delta=1e-15
        )
        # Series of repairables: A = prod A_i; tiny unavailability of a
        # triple-redundant repairable system keeps full relative accuracy.
        triple = SystemModel(
            parallel("x", "y", "z"),
            dict.fromkeys("xyz", RepairableComponent(1e-6, 1.0)),
        )
        u = RepairableComponent(1e-6, 1.0).unavailability(1e3)
        self.assertAlmostEqual(triple.unavailability(1e3) / u**3, 1.0, delta=1e-13)
        with self.assertRaises(ValueError):
            model.reliability(-1.0)
        with self.assertRaises(ValueError):
            model.importance(1.0, measure="other")  # type: ignore[arg-type]

    def test_ccf_system(self) -> None:
        """Beta-factor redundant pair: hazard and probability splits."""
        lam, beta, t = 1e-3, 0.1, 100.0
        models = {"A": ExponentialComponent(lam), "B": ExponentialComponent(lam)}
        group = [BetaFactorGroup(("A", "B"), beta)]
        model = SystemModel(parallel("A", "B"), models, ccf_groups=group)
        # Hazard split (default for mission reliability): independent rate
        # (1 - beta) lambda per unit and CCF rate beta lambda.
        q1 = -math.expm1(-(1 - beta) * lam * t)
        q2 = -math.expm1(-beta * lam * t)
        expected = q2 + (1 - q2) * q1 * q1
        self.assertAlmostEqual(model.unreliability(t), expected, delta=1e-17)
        # R(t) = 2 exp(-lambda t) - exp(-(2 - beta) lambda t), so
        # MTTF = 2 / lambda - 1 / ((2 - beta) lambda).
        self.assertAlmostEqual(
            model.reliability(t),
            2 * math.exp(-lam * t) - math.exp(-(2 - beta) * lam * t),
            delta=1e-15,
        )
        mttf = model.mttf().value
        self.assertAlmostEqual(mttf * lam / (2 - 1 / (2 - beta)), 1.0, delta=1e-9)
        independent = SystemModel(parallel("A", "B"), models).mttf().value
        self.assertLess(mttf, independent)
        # Each member's marginal is exactly Q_t under the hazard split.
        p, q = model.event_probabilities(t)
        marginal = 1 - p["IND[A]"] * p["CCF[A,B]"]
        self.assertAlmostEqual(marginal, -math.expm1(-lam * t), delta=1e-15)
        # Probability split (NUREG/CR-5485 on Q_t): Q_k = c_k Q_t.
        by_probability = SystemModel(
            parallel("A", "B"), models, ccf_groups=group, ccf_basis="probability"
        )
        qt = -math.expm1(-lam * t)
        q1, q2 = (1 - beta) * qt, beta * qt
        self.assertAlmostEqual(
            by_probability.unreliability(t), q2 + (1 - q2) * q1 * q1, delta=1e-17
        )
        # Availability uses the probability split by default.
        repairable = {n: RepairableComponent(lam, 0.1) for n in "AB"}
        available = SystemModel(parallel("A", "B"), repairable, ccf_groups=group)
        u = RepairableComponent(lam, 0.1).unavailability(t)
        u1, u2 = (1 - beta) * u, beta * u
        self.assertAlmostEqual(
            available.unavailability(t), u2 + (1 - u2) * u1 * u1, delta=1e-17
        )
        self.assertEqual(set(model.basic_events), {"IND[A]", "IND[B]", "CCF[A,B]"})
        cuts = set(model.minimal_cut_sets())
        self.assertEqual(
            cuts, {frozenset({"CCF[A,B]"}), frozenset({"IND[A]", "IND[B]"})}
        )
        measures = model.importance(t)
        self.assertGreater(
            measures["CCF[A,B]"].fussell_vesely, measures["IND[A]"].fussell_vesely
        )
        mismatched = SystemModel(
            parallel("A", "B"),
            {"A": ExponentialComponent(lam), "B": ExponentialComponent(2 * lam)},
            ccf_groups=group,
        )
        with self.assertRaises(ValueError):
            mismatched.reliability(t)
        with self.assertRaises(ValueError):
            SystemModel(
                parallel("A", "B"),
                models,
                ccf_groups=group,
                ccf_basis="x",  # type: ignore[arg-type]
            )

    def test_validation(self) -> None:
        """Missing, unknown, and invalid component models are rejected."""
        with self.assertRaises(KeyError):
            SystemModel(series("A", "B"), {"A": ExponentialComponent(1.0)})
        with self.assertRaises(KeyError):
            SystemModel(
                "A", {"A": ExponentialComponent(1.0), "Z": ExponentialComponent(1.0)}
            )
        with self.assertRaises(TypeError):
            SystemModel("A", {"A": 0.5})  # type: ignore[dict-item]


class DocumentedExampleTests(unittest.TestCase):
    """The cooling-loop example of docs/foundations/system-reliability.md."""

    def test_cooling_loop(self) -> None:
        """Reliability, MTTF (closed form), cut sets and importance."""
        structure = series(
            parallel("P1", "P2"), "V", k_out_of_n(2, "S1", "S2", "S3"), "C"
        )
        rates = {
            "P1": 1e-4,
            "P2": 1e-4,
            "V": 2e-5,
            "S1": 5e-5,
            "S2": 5e-5,
            "S3": 5e-5,
            "C": 1e-5,
        }
        components = {n: ExponentialComponent(r) for n, r in rates.items()}
        plain = SystemModel(structure, components)
        ccf = SystemModel(
            structure, components, ccf_groups=[BetaFactorGroup(("P1", "P2"), 0.05)]
        )
        t = 1000.0
        lp, ls, base = 1e-4, 5e-5, 3e-5
        reliability = (
            (2 * math.exp(-lp * t) - math.exp(-2 * lp * t))
            * (3 * math.exp(-2 * ls * t) - 2 * math.exp(-3 * ls * t))
            * math.exp(-base * t)
        )
        self.assertAlmostEqual(plain.reliability(t), reliability, delta=1e-15)
        self.assertAlmostEqual(plain.reliability(t), 0.955018267, delta=5e-10)
        self.assertAlmostEqual(ccf.reliability(t), 0.951063146, delta=5e-10)
        self.assertAlmostEqual(ccf.unreliability(t), 0.048936854, delta=5e-10)
        mttf = (
            6 / (lp + 2 * ls + base)
            - 4 / (lp + 3 * ls + base)
            - 3 / (2 * lp + 2 * ls + base)
            + 2 / (2 * lp + 3 * ls + base)
        )
        self.assertAlmostEqual(plain.mttf().value / mttf, 1.0, delta=1e-10)
        self.assertAlmostEqual(plain.mttf().value, 7973.491040, delta=5e-6)
        self.assertAlmostEqual(ccf.mttf().value, 7903.806339, delta=5e-6)
        self.assertIn(frozenset({"CCF[P1,P2]"}), ccf.minimal_cut_sets())
        self.assertEqual(len(plain.minimal_cut_sets()), 6)
        measures = ccf.importance(t)
        expected = {
            "V": (0.970, 0.393, 0.405, 20.4, 1.65),
            "C": (0.961, 0.195, 0.203, 20.4, 1.24),
            "IND[P1]": (0.087, 0.161, 0.168, 2.62, 1.19),
            "CCF[P1,P2]": (0.956, 0.097, 0.102, 20.4, 1.11),
            "S1": (0.089, 0.089, 0.095, 2.73, 1.10),
        }
        for name, values in expected.items():
            m = measures[name]
            got = (
                m.birnbaum,
                m.criticality,
                m.fussell_vesely,
                m.risk_achievement_worth,
                m.risk_reduction_worth,
            )
            for value, want in zip(got, values, strict=True):
                assert value is not None
                self.assertAlmostEqual(value / want, 1.0, delta=0.006)


class ProbabilisticModelBridgeTests(unittest.TestCase):
    """Exact system QoIs inside the central probabilistic model."""

    def setUp(self) -> None:
        """Build a bridge of fixed-probability components."""
        self.names = "abcde"
        self.system = SystemModel(
            bridge(*self.names), {n: FixedProbability(0.8) for n in self.names}
        )
        self.law = Beta(8.0, 2.0)

    def test_qoi_values_equal_direct_evaluation(self) -> None:
        """Evaluating the model equals the exact structure reliability."""
        uncertain = {n: RandomInput(f"p_{n}", self.law) for n in self.names}
        model = self.system.as_probabilistic_model(
            uncertain, [1.0], measures=("reliability", "unreliability")
        )
        self.assertIsInstance(model, ProbabilisticModel)
        point = {f"p_{n}": 0.5 + 0.1 * i for i, n in enumerate(self.names)}
        outputs = model.evaluate(point)
        structure = SystemStructure(bridge(*self.names))
        expected = structure.reliability({n: point[f"p_{n}"] for n in self.names})
        self.assertAlmostEqual(outputs["reliability@1.0"], expected, delta=1e-15)
        self.assertAlmostEqual(outputs["unreliability@1.0"], 1 - expected, delta=1e-15)

    def test_propagated_mean_matches_exact_moments(self) -> None:
        """Monte Carlo propagation agrees with the exact E[R] and Var[R]."""
        uncertain = {n: RandomInput(f"p_{n}", self.law) for n in self.names}
        model = self.system.as_probabilistic_model(uncertain, [1.0])
        study = propagate(model, 4000, design="mc", seed=7)
        output = study.outputs["reliability@1.0"]
        exact = SystemStructure(bridge(*self.names)).reliability_moments(
            dict.fromkeys(self.names, self.law.mean),
            dict.fromkeys(self.names, self.law.variance),
        )
        standard_error = exact.standard_deviation / math.sqrt(4000)
        self.assertLess(abs(output.mean - exact.mean), 5 * standard_error)
        self.assertAlmostEqual(
            output.standard_deviation / exact.standard_deviation, 1.0, delta=0.08
        )
        # The same model runs inside a UQStudy.
        report = UQStudy(model, seed=3).propagate(256)
        self.assertEqual(report.model_evaluations, 256)

    def test_shared_input_models_state_of_knowledge_correlation(self) -> None:
        """Components mapped to one input name share one value."""
        shared = RandomInput("p_shared", self.law)
        uncertain = {"a": shared, "b": shared}
        model = self.system.as_probabilistic_model(uncertain, [1.0])
        self.assertEqual(model.inputs.names, ("p_shared",))
        value = model.evaluate({"p_shared": 0.6})["reliability@1.0"]
        p = {n: 0.8 for n in self.names}
        p.update(a=0.6, b=0.6)
        self.assertAlmostEqual(
            value, SystemStructure(bridge(*self.names)).reliability(p), delta=1e-15
        )

    def test_rates_and_availability_qois(self) -> None:
        """Gamma-distributed failure rates drive time-dependent QoIs."""
        system = SystemModel(
            parallel("A", "B"),
            {"A": RepairableComponent(1e-3, 0.1), "B": ExponentialComponent(2e-3)},
        )
        model = system.as_probabilistic_model(
            {"A": RandomInput("lambda_A", Gamma(2.0, 2000.0))},
            [10.0, 100.0],
            measures=("unavailability",),
        )
        self.assertEqual(
            model.quantity_names, ("unavailability@10.0", "unavailability@100.0")
        )
        out = model.evaluate({"lambda_A": 1e-3})
        self.assertAlmostEqual(
            out["unavailability@100.0"], system.unavailability(100.0), delta=1e-16
        )

    def test_validation(self) -> None:
        """Bad times, measures, inputs and builders are rejected."""
        uncertain = {"a": RandomInput("p_a", self.law)}
        with self.assertRaises(ValueError):
            self.system.as_probabilistic_model(uncertain, [])
        with self.assertRaises(ValueError):
            self.system.as_probabilistic_model(uncertain, [1.0], measures=("mttf",))
        with self.assertRaises(KeyError):
            self.system.as_probabilistic_model(
                {"z": RandomInput("p_z", self.law)}, [1.0]
            )
        with self.assertRaises(ValueError):
            self.system.as_probabilistic_model(
                {
                    "a": RandomInput("p", self.law),
                    "b": RandomInput("p", Beta(2.0, 2.0)),
                },
                [1.0],
            )
        with self.assertRaises(KeyError):
            self.system.as_probabilistic_model(
                uncertain, [1.0], builders={"b": FixedProbability}
            )
        weird = SystemModel("A", {"A": _CustomModel()})
        with self.assertRaises(TypeError):
            weird.as_probabilistic_model({"A": RandomInput("x", self.law)}, [1.0])
        custom = weird.as_probabilistic_model(
            {"A": RandomInput("x", self.law)}, [1.0], builders={"A": FixedProbability}
        )
        self.assertAlmostEqual(
            custom.evaluate({"x": 0.25})["reliability@1.0"], 0.25, delta=1e-16
        )


class _CustomModel:
    """A user-defined component model without a default parameter."""

    def reliability(self, t: float) -> float:
        """Return a constant reliability."""
        return 0.5

    def unreliability(self, t: float) -> float:
        """Return a constant unreliability."""
        return 0.5

    def availability(self, t: float) -> float:
        """Return a constant availability."""
        return 0.5

    def unavailability(self, t: float) -> float:
        """Return a constant unavailability."""
        return 0.5


class PhasedMissionTests(unittest.TestCase):
    """Exact phased-mission reliability against enumeration."""

    def brute_force(self, mission: PhasedMission):
        """Enumerate every combination of phases of failure."""
        names = mission.components
        laws = {n: mission.phase_of_failure(n) for n in names}
        phases = mission.phases
        count = len(phases)
        success = 0.0
        cumulative = [0.0] * count
        terms: list[float] = []
        cumulative_terms: list[list[float]] = [[] for _ in range(count)]
        for outcomes in itertools.product(range(1, count + 2), repeat=len(names)):
            weight = math.prod(
                laws[n][k - 1] for n, k in zip(names, outcomes, strict=True)
            )
            if weight == 0.0:
                continue
            failure_phase = dict(zip(names, outcomes, strict=True))
            ok = True
            for j, phase in enumerate(phases, start=1):
                state = {n: int(j < failure_phase[n]) for n in names}
                if ok and not phi(phase.structure, state):
                    ok = False
                if ok:
                    cumulative_terms[j - 1].append(weight)
            if ok:
                terms.append(weight)
        success = math.fsum(terms)
        cumulative = [math.fsum(t) for t in cumulative_terms]
        return success, cumulative

    def test_against_enumeration(self) -> None:
        """Three phases, five components, shared and phase-specific use."""
        phases = [
            Phase("launch", 0.1, series("A", parallel("B", "C"))),
            Phase("cruise", 10.0, k_out_of_n(2, "B", "C", "D")),
            Phase("landing", 0.2, bridge("A", "B", "E", "C", "D")),
        ]
        components = {
            "A": [
                ExponentialComponent(0.5),
                ExponentialComponent(0.01),
                ExponentialComponent(0.4),
            ],
            "B": ExponentialComponent(0.03),
            "C": WeibullComponent(1.7, 30.0),
            "D": [
                ExponentialComponent(0.2),
                ExponentialComponent(0.05),
                ExponentialComponent(0.3),
            ],
            "E": [FixedProbability(1.0), FixedProbability(1.0), DemandComponent(0.02)],
        }
        mission = PhasedMission(phases, components)
        result = mission.evaluate()
        success, cumulative = self.brute_force(mission)
        self.assertAlmostEqual(result.reliability, success, delta=1e-15)
        self.assertAlmostEqual(result.unreliability, 1 - success, delta=1e-15)
        for got, want in zip(result.cumulative_reliability, cumulative, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-15)
        self.assertAlmostEqual(
            math.fsum(result.phase_failure_probability), 1 - success, delta=1e-15
        )
        product = math.prod(r for r in result.conditional_reliability if r is not None)
        self.assertAlmostEqual(product, success, delta=1e-15)
        self.assertAlmostEqual(mission.reliability(), success, delta=1e-15)
        for name in mission.components:
            self.assertAlmostEqual(
                math.fsum(mission.phase_of_failure(name)), 1.0, delta=1e-15
            )

    def test_two_phase_closed_forms(self) -> None:
        """Parallel-then-series and series-then-parallel missions.

        Phase 2 needing both units means both must survive to its end:
        ``R = sA1 sA2 sB1 sB2``; the reverse needs both through phase 1 and
        one through phase 2: ``R = sA1 sB1 [1 - (1 - sA2)(1 - sB2)]``. A
        naive product of separate phase reliabilities gets both wrong.
        """
        rates = {"A": (0.01, 0.02), "B": (0.03, 0.04)}
        durations = (10.0, 5.0)
        s = {
            n: tuple(math.exp(-r * d) for r, d in zip(rates[n], durations, strict=True))
            for n in rates
        }
        models = {n: [ExponentialComponent(r) for r in rates[n]] for n in rates}
        first = PhasedMission(
            [Phase("p1", 10.0, parallel("A", "B")), Phase("p2", 5.0, series("A", "B"))],
            models,
        )
        expected = s["A"][0] * s["A"][1] * s["B"][0] * s["B"][1]
        self.assertAlmostEqual(first.reliability(), expected, delta=1e-15)
        second = PhasedMission(
            [Phase("p1", 10.0, series("A", "B")), Phase("p2", 5.0, parallel("A", "B"))],
            models,
        )
        expected = s["A"][0] * s["B"][0] * (1 - (1 - s["A"][1]) * (1 - s["B"][1]))
        self.assertAlmostEqual(second.reliability(), expected, delta=1e-15)

    def test_single_model_ages_on_cumulative_time(self) -> None:
        """A constant-rate single model equals per-phase models of that rate."""
        phases = [
            Phase("p1", 3.0, parallel("A", "B")),
            Phase("p2", 4.0, series("A", "B")),
            Phase("p3", 2.0, "A"),
        ]
        single = PhasedMission(
            phases, {"A": ExponentialComponent(0.1), "B": ExponentialComponent(0.2)}
        )
        per_phase = PhasedMission(
            phases,
            {
                "A": [ExponentialComponent(0.1)] * 3,
                "B": [ExponentialComponent(0.2)] * 3,
            },
        )
        self.assertAlmostEqual(
            single.reliability(), per_phase.reliability(), delta=1e-15
        )
        # Mission reliability: A survives 9, B survives 7.
        self.assertAlmostEqual(
            single.reliability(), math.exp(-0.9) * math.exp(-1.4), delta=1e-15
        )
        aging = PhasedMission(
            phases, {"A": WeibullComponent(2.0, 10.0), "B": ExponentialComponent(0.2)}
        )
        law = aging.phase_of_failure("A")
        expected = [
            math.exp(-((0 / 10) ** 2)) - math.exp(-((3 / 10) ** 2)),
            math.exp(-((3 / 10) ** 2)) - math.exp(-((7 / 10) ** 2)),
            math.exp(-((7 / 10) ** 2)) - math.exp(-((9 / 10) ** 2)),
            math.exp(-((9 / 10) ** 2)),
        ]
        for got, want in zip(law, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-15)
        self.assertAlmostEqual(
            aging.reliability(), self.brute_force(aging)[0], delta=1e-15
        )

    def test_validation(self) -> None:
        """Invalid phases and models are rejected."""
        with self.assertRaises(ValueError):
            Phase("p", 0.0, "A")
        with self.assertRaises(ValueError):
            Phase("", 1.0, "A")
        with self.assertRaises(TypeError):
            Phase("p", 1.0, 3)  # type: ignore[arg-type]
        phase = Phase("p", 1.0, series("A", "B"))
        self.assertIsInstance(
            Phase("q", 1.0, SystemStructure("A")).structure, Component
        )
        with self.assertRaises(ValueError):
            PhasedMission([], {})
        with self.assertRaises(ValueError):
            PhasedMission(
                [phase, phase],
                {"A": ExponentialComponent(1.0), "B": ExponentialComponent(1.0)},
            )
        with self.assertRaises(KeyError):
            PhasedMission([phase], {"A": ExponentialComponent(1.0)})
        with self.assertRaises(KeyError):
            PhasedMission(
                [phase],
                {
                    "A": ExponentialComponent(1.0),
                    "B": ExponentialComponent(1.0),
                    "Z": ExponentialComponent(1.0),
                },
            )
        with self.assertRaises(ValueError):
            PhasedMission(
                [phase],
                {"A": [ExponentialComponent(1.0)] * 2, "B": ExponentialComponent(1.0)},
            )


if __name__ == "__main__":
    unittest.main()
