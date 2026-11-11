"""Tests for dynamic fault trees in ``probstat.systems``.

The exact state-space quantification is checked against closed forms for
the elementary dynamic gates (PAND, cold, warm and hot spares, shared spare
pools, SEQ, FDEP), against the BDD path for static trees, and against the
seeded event-time Monte Carlo simulation, which evaluates the gates from
failure times instead of chain states.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.systems import (
    DFTMonteCarloEstimate,
    DynamicBasicEvent,
    DynamicFaultTree,
    DynamicGate,
    and_gate,
    fdep_gate,
    monte_carlo_unreliability,
    or_gate,
    pand_gate,
    seq_gate,
    spare_gate,
    vote_gate,
)

E = DynamicBasicEvent
TIMES = (0.25, 1.0, 2.5, 6.0)


def pand_closed_form(a: float, b: float, t: float) -> float:
    """Return ``P(A fails before B, both by t)`` (EQ-SYS-DFT-004)."""
    return a / (a + b) * -math.expm1(-(a + b) * t) - math.exp(-b * t) * -math.expm1(
        -a * t
    )


def warm_reliability(a: float, b: float, alpha: float, t: float) -> float:
    """Return the warm-spare reliability with primary ``a``, spare ``b``."""
    c = a + alpha * b - b
    if c == 0.0:
        return math.exp(-a * t) * (1.0 + a * t)
    return math.exp(-a * t) + a * math.exp(-b * t) * -math.expm1(-c * t) / c


def hypoexponential_cdf(r1: float, r2: float, t: float) -> float:
    """Return the CDF of ``Exp(r1) + Exp(r2)`` (Erlang when equal)."""
    if r1 == r2:
        return 1.0 - math.exp(-r1 * t) * (1.0 + r1 * t)
    return 1.0 - (r2 * math.exp(-r1 * t) - r1 * math.exp(-r2 * t)) / (r2 - r1)


class DynamicGateClosedFormTests(unittest.TestCase):
    """The Markov path reproduces the closed forms of EQ-SYS-DFT-004."""

    def test_pand_matches_integral(self) -> None:
        """PAND(A, B) equals the order-statistics integral at every time."""
        for a, b in ((0.3, 0.7), (1.0, 1.0), (2.0, 0.1)):
            tree = DynamicFaultTree(
                [E("A", a), E("B", b)], [pand_gate("P", "A", "B")], "P"
            )
            for t, value in zip(TIMES, tree.unreliability_curve(TIMES), strict=True):
                self.assertAlmostEqual(
                    value.value, pand_closed_form(a, b, t), delta=1e-13
                )
                self.assertLess(value.error_bound, 1e-12)
            # B before A blocks the gate forever: the MTTF is infinite.
            self.assertEqual(tree.mttf(), math.inf)
            self.assertTrue(tree.state_space().absorbing_up_states)

    def test_pand_order_matters(self) -> None:
        """Swapping the inputs gives the complementary ordering."""
        a, b, t = 0.4, 0.9, 1.7
        forward = DynamicFaultTree(
            [E("A", a), E("B", b)], [pand_gate("P", "A", "B")], "P"
        ).unreliability(t)
        backward = DynamicFaultTree(
            [E("A", a), E("B", b)], [pand_gate("P", "B", "A")], "P"
        ).unreliability(t)
        both = -math.expm1(-a * t) * -math.expm1(-b * t)
        self.assertAlmostEqual(forward.value + backward.value, both, delta=1e-13)

    def test_cold_spare_is_erlang(self) -> None:
        """A cold spare pair of identical units fails as Erlang(2, lambda)."""
        lam = 0.4
        tree = DynamicFaultTree(
            [E("P", lam), E("S", lam, 0.0)], [spare_gate("G", "P", "S")], "G"
        )
        for t, value in zip(TIMES, tree.unreliability_curve(TIMES), strict=True):
            expected = 1.0 - math.exp(-lam * t) * (1.0 + lam * t)
            self.assertAlmostEqual(value.value, expected, delta=1e-13)
        self.assertAlmostEqual(tree.mttf(), 2.0 / lam, delta=1e-12)

    def test_hot_spare_is_parallel(self) -> None:
        """A hot spare pair equals active parallel redundancy."""
        lam = 0.4
        tree = DynamicFaultTree(
            [E("P", lam), E("S", lam, 1.0)], [spare_gate("G", "P", "S")], "G"
        )
        parallel = DynamicFaultTree(
            [E("P", lam), E("S", lam)], [and_gate("G", "P", "S")], "G"
        )
        for t in TIMES:
            expected = (-math.expm1(-lam * t)) ** 2
            self.assertAlmostEqual(tree.unreliability(t).value, expected, delta=1e-13)
            self.assertAlmostEqual(
                tree.unreliability(t).value,
                parallel.unreliability(t, method="bdd").value,
                delta=1e-13,
            )
        self.assertAlmostEqual(tree.mttf(), 1.5 / lam, delta=1e-12)

    def test_warm_spare_is_hypoexponential(self) -> None:
        """Warm spares with unequal rates match the closed form and its mean."""
        for a, b, alpha in ((0.4, 0.4, 0.3), (0.5, 0.2, 0.25), (0.3, 0.9, 0.6)):
            tree = DynamicFaultTree(
                [E("P", a), E("S", b, alpha)], [spare_gate("G", "P", "S")], "G"
            )
            for t in TIMES:
                reliability = tree.reliability(t)
                self.assertAlmostEqual(
                    reliability.value, warm_reliability(a, b, alpha, t), delta=1e-13
                )
                self.assertAlmostEqual(
                    tree.unreliability(t).value,
                    1.0 - warm_reliability(a, b, alpha, t),
                    delta=1e-13,
                )
            first = a + alpha * b
            mean = 1.0 / first + (a / first) / b + (alpha * b / first) / a
            self.assertAlmostEqual(tree.mttf(), mean, delta=1e-12)

    def test_two_spares_in_order(self) -> None:
        """Two cold spares give Erlang(3, lambda)."""
        lam = 0.7
        tree = DynamicFaultTree(
            [E("P", lam), E("S1", lam, 0.0), E("S2", lam, 0.0)],
            [spare_gate("G", "P", "S1", "S2")],
            "G",
        )
        for t in TIMES:
            x = lam * t
            expected = 1.0 - math.exp(-x) * (1.0 + x + x * x / 2.0)
            self.assertAlmostEqual(tree.unreliability(t).value, expected, delta=1e-13)
        self.assertAlmostEqual(tree.mttf(), 3.0 / lam, delta=1e-12)

    def test_shared_spare_pool(self) -> None:
        """Two gates sharing one spare: the second unit loss fails an OR top."""
        lam = 0.4
        for alpha in (0.0, 0.5, 1.0):
            tree = DynamicFaultTree(
                [E("P1", lam), E("P2", lam), E("S", lam, alpha)],
                [
                    spare_gate("G1", "P1", "S"),
                    spare_gate("G2", "P2", "S"),
                    or_gate("top", "G1", "G2"),
                ],
                "top",
            )
            r1, r2 = (2.0 + alpha) * lam, 2.0 * lam
            for t in TIMES:
                self.assertAlmostEqual(
                    tree.unreliability(t).value,
                    hypoexponential_cdf(r1, r2, t),
                    delta=1e-13,
                )
            self.assertAlmostEqual(tree.mttf(), 1.0 / r1 + 1.0 / r2, delta=1e-12)

    def test_seq_is_cold_standby(self) -> None:
        """SEQ(A, B) equals a cold spare of A by B (a hypoexponential law)."""
        a, b = 0.5, 0.9
        seq = DynamicFaultTree([E("A", a), E("B", b)], [seq_gate("S", "A", "B")], "S")
        cold = DynamicFaultTree(
            [E("A", a), E("B", b, 0.0)], [spare_gate("S", "A", "B")], "S"
        )
        for t in TIMES:
            expected = hypoexponential_cdf(a, b, t)
            self.assertAlmostEqual(seq.unreliability(t).value, expected, delta=1e-13)
            self.assertAlmostEqual(cold.unreliability(t).value, expected, delta=1e-13)
        self.assertAlmostEqual(seq.mttf(), 1.0 / a + 1.0 / b, delta=1e-12)

    def test_fdep_is_or_of_trigger(self) -> None:
        """FDEP(T -> A, B) under AND(A, B) equals the static T or (A and B)."""
        rates = {"T": 0.1, "A": 0.5, "B": 0.6}
        events = [E(name, rate) for name, rate in rates.items()]
        tree = DynamicFaultTree(
            events, [and_gate("top", "A", "B"), fdep_gate("F", "T", "A", "B")], "top"
        )
        static = DynamicFaultTree(
            events,
            [and_gate("ab", "A", "B"), or_gate("top", "T", "ab")],
            "top",
        )
        self.assertFalse(tree.is_static)
        self.assertTrue(static.is_static)
        for t in TIMES:
            f = {name: -math.expm1(-rate * t) for name, rate in rates.items()}
            expected = 1.0 - (1.0 - f["T"]) * (1.0 - f["A"] * f["B"])
            self.assertAlmostEqual(tree.unreliability(t).value, expected, delta=1e-13)
            self.assertAlmostEqual(
                tree.unreliability(t).value,
                static.unreliability(t, method="bdd").value,
                delta=1e-14,
            )

    def test_fdep_gate_trigger_and_simultaneous_pand(self) -> None:
        """A gate trigger fails dependents at once; ties count as in order."""
        # T = AND(X, Y) triggers A and B together, which fires PAND(A, B).
        rates = {"X": 0.8, "Y": 0.6, "A": 0.05, "B": 0.07}
        tree = DynamicFaultTree(
            [E(name, rate) for name, rate in rates.items()],
            [
                and_gate("T", "X", "Y"),
                fdep_gate("F", "T", "A", "B"),
                pand_gate("P", "A", "B"),
            ],
            "P",
        )
        estimate = monte_carlo_unreliability(tree, (2.0,), samples=6000, seed=11)[0]
        exact = tree.unreliability(2.0).value
        self.assertTrue(estimate.contains(exact), (estimate, exact))
        # Without the dependency only the order-statistics term remains.
        plain = pand_closed_form(rates["A"], rates["B"], 2.0)
        self.assertGreater(exact, plain)


class StaticAgreementTests(unittest.TestCase):
    """Static trees give the BDD result on the Markov path (to 1e-14)."""

    def test_random_static_trees_match_bdd(self) -> None:
        """Random AND/OR/vote trees with shared events agree with the BDD."""
        rng = random.Random(20261111)
        for trial in range(25):
            count = rng.randint(2, 5)
            names = [f"e{i}" for i in range(count)]
            events = [E(name, rng.uniform(0.05, 1.5)) for name in names]
            gates: list[DynamicGate] = []
            pool = list(names)
            for g in range(rng.randint(1, 4)):
                width = rng.randint(1, min(3, len(pool)))
                inputs = rng.sample(pool, width)
                kind = rng.choice(("and", "or", "vote"))
                name = f"g{g}"
                if kind == "vote":
                    gates.append(vote_gate(name, rng.randint(1, width), *inputs))
                elif kind == "and":
                    gates.append(and_gate(name, *inputs))
                else:
                    gates.append(or_gate(name, *inputs))
                pool.append(name)
            tree = DynamicFaultTree(events, gates, gates[-1].name)
            self.assertTrue(tree.is_static)
            for t in (0.1, 1.0, 4.0):
                markov = tree.unreliability(t, method="markov", tolerance=1e-15)
                bdd = tree.unreliability(t, method="bdd")
                self.assertLess(abs(markov.value - bdd.value), 1e-14, (trial, t))
                self.assertEqual(tree.unreliability(t).value, bdd.value)

    def test_static_module_inside_dynamic_tree(self) -> None:
        """AND(static module, PAND) factorises into the BDD and the closed form."""
        events = [E("a", 0.3), E("b", 0.5), E("c", 0.2), E("x", 0.6), E("y", 0.4)]
        module = [vote_gate("m", 2, "a", "b", "c")]
        tree = DynamicFaultTree(
            events,
            [*module, pand_gate("p", "x", "y"), and_gate("top", "m", "p")],
            "top",
        )
        static = DynamicFaultTree(events[:3], module, "m")
        for t in TIMES:
            expected = static.unreliability(t, method="bdd").value * pand_closed_form(
                0.6, 0.4, t
            )
            self.assertAlmostEqual(tree.unreliability(t).value, expected, delta=1e-14)

    def test_static_mttf_and_reliability(self) -> None:
        """A 2-out-of-3 failure vote has MTTF 5/(6 lambda) and R = 1 - F."""
        lam = 0.25
        tree = DynamicFaultTree(
            [E(n, lam) for n in "abc"], [vote_gate("v", 2, "a", "b", "c")], "v"
        )
        self.assertAlmostEqual(tree.mttf(), 5.0 / (6.0 * lam), delta=1e-12)
        for t in TIMES:
            r_markov = tree.reliability(t, method="markov").value
            r_bdd = tree.reliability(t, method="bdd").value
            self.assertAlmostEqual(r_markov, r_bdd, delta=1e-13)
            self.assertAlmostEqual(
                r_bdd + tree.unreliability(t).value, 1.0, delta=1e-15
            )


class MonteCarloAgreementTests(unittest.TestCase):
    """The event-time simulation brackets every exact value."""

    def assert_inside(self, tree: DynamicFaultTree, seed: int) -> None:
        """Check that each exact ``F(t)`` lies in the 99.9% interval.

        With 20000 histories the interval half-width is at most about
        0.012, so a semantic difference of that size would be detected.
        """
        times = (0.5, 1.5, 4.0)
        estimates = monte_carlo_unreliability(
            tree, times, samples=20000, seed=seed, confidence=0.999
        )
        for estimate, exact in zip(
            estimates, tree.unreliability_curve(times), strict=True
        ):
            self.assertIsInstance(estimate, DFTMonteCarloEstimate)
            self.assertTrue(estimate.contains(exact.value), (estimate, exact.value))
            self.assertLessEqual(estimate.lower, estimate.estimate)
            self.assertLessEqual(estimate.estimate, estimate.upper)

    def test_elementary_gates(self) -> None:
        """PAND, cold, warm and hot spares, SEQ and FDEP agree with simulation."""
        lam = 0.4
        trees = [
            DynamicFaultTree(
                [E("A", 0.3), E("B", 0.7)], [pand_gate("P", "A", "B")], "P"
            ),
            DynamicFaultTree(
                [E("P", lam), E("S", lam, 0.0)], [spare_gate("G", "P", "S")], "G"
            ),
            DynamicFaultTree(
                [E("P", lam), E("S", lam, 0.3)], [spare_gate("G", "P", "S")], "G"
            ),
            DynamicFaultTree(
                [E("P", lam), E("S", lam, 1.0)], [spare_gate("G", "P", "S")], "G"
            ),
            DynamicFaultTree(
                [E("A", 0.5), E("B", 0.9)], [seq_gate("S", "A", "B")], "S"
            ),
            DynamicFaultTree(
                [E("T", 0.1), E("A", 0.5), E("B", 0.6)],
                [and_gate("top", "A", "B"), fdep_gate("F", "T", "A", "B")],
                "top",
            ),
        ]
        for seed, tree in enumerate(trees, start=1):
            self.assert_inside(tree, seed)

    def test_shared_pool_with_and_top(self) -> None:
        """Shared warm spares under an AND top agree with simulation."""
        tree = DynamicFaultTree(
            [
                E("P1", 0.5),
                E("P2", 0.3),
                E("S1", 0.4, 0.2),
                E("S2", 0.6, 0.5),
            ],
            [
                spare_gate("G1", "P1", "S1", "S2"),
                spare_gate("G2", "P2", "S2", "S1"),
                and_gate("top", "G1", "G2"),
            ],
            "top",
        )
        self.assert_inside(tree, 21)

    def test_mixed_tree(self) -> None:
        """A tree with every gate kind agrees with simulation."""
        tree = DynamicFaultTree(
            [
                E("pump", 0.3),
                E("backup", 0.3, 0.1),
                E("valve", 0.2),
                E("power", 0.05),
                E("sensor1", 0.4),
                E("sensor2", 0.5),
                E("seq1", 0.6),
                E("seq2", 0.8),
            ],
            [
                spare_gate("pumps", "pump", "backup"),
                fdep_gate("blackout", "power", "valve", "sensor1"),
                pand_gate("late", "sensor1", "sensor2"),
                seq_gate("chain", "seq1", "seq2"),
                vote_gate("vote", 2, "pumps", "valve", "late", "chain"),
                or_gate("top", "vote", "power"),
            ],
            "top",
        )
        self.assert_inside(tree, 31)

    def test_seeded_reproducibility(self) -> None:
        """The same seed gives the same estimate."""
        tree = DynamicFaultTree(
            [E("A", 0.3), E("B", 0.7)], [pand_gate("P", "A", "B")], "P"
        )
        first = monte_carlo_unreliability(tree, (1.0,), samples=500, seed=5)
        second = monte_carlo_unreliability(tree, (1.0,), samples=500, seed=5)
        self.assertEqual(first, second)


class StateSpaceTests(unittest.TestCase):
    """Chain size, pruning and the state-explosion guard."""

    def test_chain_size_is_reported(self) -> None:
        """A cold spare pair has three working states and one failure state."""
        tree = DynamicFaultTree(
            [E("P", 1.0), E("S", 1.0, 0.0)], [spare_gate("G", "P", "S")], "G"
        )
        space = tree.state_space()
        self.assertEqual(space.size, 3)
        self.assertEqual(space.failure_state, "failed")
        self.assertEqual(space.transitions, 2)
        self.assertIs(tree.state_space(), space)

    def test_irrelevant_elements_are_pruned(self) -> None:
        """Events and gates that cannot reach the top event are not generated."""
        tree = DynamicFaultTree(
            [E("A", 1.0), E("B", 1.0), E("C", 1.0), E("D", 1.0)],
            [and_gate("top", "A", "B"), or_gate("unused", "C", "D")],
            "top",
        )
        self.assertEqual([event.name for event in tree.events], ["A", "B"])
        self.assertEqual(tree.state_space().size, 4)

    def test_spare_gate_outside_top_still_claims(self) -> None:
        """A gate outside the top event that shares the spare is kept."""
        lam = 0.4
        tree = DynamicFaultTree(
            [E("P1", lam), E("P2", lam), E("S", lam, 0.0)],
            [spare_gate("G1", "P1", "S"), spare_gate("G2", "P2", "S")],
            "G1",
        )
        self.assertEqual(len(tree.events), 3)
        estimate = monte_carlo_unreliability(tree, (2.0,), samples=4000, seed=3)[0]
        exact = tree.unreliability(2.0).value
        self.assertTrue(estimate.contains(exact))
        alone = 1.0 - math.exp(-lam * 2.0) * (1.0 + lam * 2.0)
        self.assertGreater(exact, alone)

    def test_state_limit(self) -> None:
        """Exceeding ``max_states`` raises a clear error."""
        events = [E(f"e{i}", 1.0) for i in range(10)]
        gates = [pand_gate("top", *(event.name for event in events))]
        tree = DynamicFaultTree(events, gates, "top", max_states=50)
        with self.assertRaisesRegex(ValueError, "max_states=50"):
            tree.state_space()
        with self.assertRaisesRegex(ValueError, "max_states"):
            tree.mttf()

    def test_top_that_cannot_fail(self) -> None:
        """A zero-rate event gives zero unreliability and infinite MTTF."""
        tree = DynamicFaultTree([E("A", 0.0)], [], "A")
        self.assertEqual(tree.unreliability(3.0, method="markov").value, 0.0)
        self.assertEqual(tree.mttf(), math.inf)


class ValidationTests(unittest.TestCase):
    """Invalid events, gates and trees raise."""

    def test_invalid_events_and_gates(self) -> None:
        """Rates, dormancy factors, kinds, arities and thresholds are checked."""
        with self.assertRaises(ValueError):
            E("A", -1.0)
        with self.assertRaises(ValueError):
            E("A", 1.0, 1.5)
        with self.assertRaises(ValueError):
            E("", 1.0)
        with self.assertRaises(ValueError):
            DynamicGate("g", "xor", ("A", "B"))
        with self.assertRaises(ValueError):
            pand_gate("p", "A")
        with self.assertRaises(ValueError):
            spare_gate("s", "A")
        with self.assertRaises(ValueError):
            vote_gate("v", 3, "A", "B")
        with self.assertRaises(ValueError):
            DynamicGate("v", "vote", ("A", "B"))
        with self.assertRaises(ValueError):
            DynamicGate("a", "and", ("A", "B"), 1)
        with self.assertRaises(ValueError):
            and_gate("a", "A", "A")
        with self.assertRaises(TypeError):
            DynamicGate("a", "and", "AB")  # type: ignore[arg-type]

    def test_invalid_trees(self) -> None:
        """Structural errors in a tree raise ``ValueError``."""
        a, b, c = E("A", 1.0), E("B", 1.0), E("C", 1.0)
        cases = [
            ([a, b], [and_gate("g", "A", "Z")], "g"),
            ([a, b], [and_gate("g", "A", "B")], "missing"),
            ([a, E("A", 2.0)], [], "A"),
            ([a, b], [and_gate("A", "B")], "A"),
            ([a, b], [and_gate("g", "A", "h"), or_gate("h", "B", "g")], "g"),
            ([a, b, c], [fdep_gate("f", "A", "B"), and_gate("g", "f", "C")], "g"),
            ([a, b], [fdep_gate("f", "A", "B")], "f"),
            ([a, b, c], [and_gate("h", "A", "B"), spare_gate("s", "h", "C")], "s"),
            ([a, b, c], [and_gate("h", "A", "B"), fdep_gate("f", "C", "h")], "h"),
            ([a, b, c], [spare_gate("s", "A", "B"), spare_gate("t", "B", "C")], "s"),
            ([a, b, c], [spare_gate("s", "A", "B"), spare_gate("t", "A", "C")], "s"),
            ([a, b, c], [and_gate("h", "A", "B"), seq_gate("q", "C", "h")], "q"),
            ([a, b, c], [seq_gate("q", "A", "B"), spare_gate("s", "C", "B")], "q"),
            ([a, b, c], [seq_gate("q", "A", "B"), seq_gate("r", "C", "B")], "q"),
        ]
        for events, gates, top in cases:
            with self.subTest(gates=gates, top=top):
                with self.assertRaises(ValueError):
                    DynamicFaultTree(events, gates, top)
        with self.assertRaises(ValueError):
            DynamicFaultTree([a], [], "A", max_states=0)
        with self.assertRaises(TypeError):
            DynamicFaultTree(["A"], [], "A")  # type: ignore[list-item]

    def test_invalid_queries(self) -> None:
        """Methods reject bad methods, times and sample sizes."""
        tree = DynamicFaultTree(
            [E("A", 0.3), E("B", 0.7)], [pand_gate("P", "A", "B")], "P"
        )
        with self.assertRaises(ValueError):
            tree.unreliability(1.0, method="bdd")
        with self.assertRaises(ValueError):
            tree.static_structure()
        with self.assertRaises(ValueError):
            tree.unreliability(1.0, method="exact")
        with self.assertRaises(ValueError):
            tree.unreliability(-1.0)
        with self.assertRaises(ValueError):
            monte_carlo_unreliability(tree, (1.0,), samples=0, seed=1)
        with self.assertRaises(ValueError):
            monte_carlo_unreliability(tree, (), samples=10, seed=1)
        with self.assertRaises(TypeError):
            monte_carlo_unreliability("tree", (1.0,), samples=10)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
