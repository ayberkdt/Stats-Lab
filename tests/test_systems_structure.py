"""Tests for structure expressions and ``SystemStructure``.

Oracles: a direct recursive evaluator of the expression (independent of the
BDD) with brute-force enumeration over all ``2^n`` component states in exact
rational arithmetic; the textbook bridge polynomial; binomial and
Poisson-binomial sums for k-out-of-n; the consecutive-2-out-of-n:F
recursion; ``risk.FaultTree`` (Shannon decomposition and MOCUS); and a
numpy Monte Carlo run in the verification environment for epistemic moments.
"""

from __future__ import annotations

import itertools
import math
import random
import time
import unittest
from fractions import Fraction

from probstat.risk import BasicEvent, FaultTree, and_gate, k_of_n_gate, or_gate
from probstat.systems import (
    Component,
    KOutOfN,
    Parallel,
    Series,
    SystemStructure,
    bridge,
    from_fault_tree,
    k_out_of_n,
    parallel,
    series,
)


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


def brute_reliability(expression, names, p) -> Fraction:
    """Return the exact reliability by enumeration of all states."""
    total = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(names)):
        state = dict(zip(names, bits, strict=True))
        if phi(expression, state):
            weight = Fraction(1)
            for name in names:
                weight *= p[name] if state[name] else 1 - p[name]
            total += weight
    return total


def brute_minimal_sets(expression, names, *, cuts: bool):
    """Return minimal cut sets (cuts=True) or path sets by enumeration."""
    found = []
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            chosen = set(subset)
            if cuts:
                state = {n: 0 if n in chosen else 1 for n in names}
                hit = not phi(expression, state)
            else:
                state = {n: 1 if n in chosen else 0 for n in names}
                hit = phi(expression, state)
            if hit and not any(f <= chosen for f in found):
                found.append(frozenset(chosen))
    return set(found)


def random_expression(rng: random.Random, names, depth: int):
    """Return a random monotone structure with repeated components."""
    if depth == 0 or rng.random() < 0.3:
        return Component(rng.choice(names))
    kind = rng.choice(("series", "parallel", "kofn"))
    items = [random_expression(rng, names, depth - 1) for _ in range(rng.randint(2, 4))]
    if kind == "series":
        return series(*items)
    if kind == "parallel":
        return parallel(*items)
    return k_out_of_n(rng.randint(1, len(items)), *items)


def bridge_polynomial(p1, p2, p3, p4, p5):
    """Return the textbook bridge reliability polynomial."""
    return (
        p1 * p4
        + p2 * p5
        + p1 * p3 * p5
        + p2 * p3 * p4
        - p1 * p2 * p3 * p4
        - p1 * p2 * p3 * p5
        - p1 * p2 * p4 * p5
        - p1 * p3 * p4 * p5
        - p2 * p3 * p4 * p5
        + 2 * p1 * p2 * p3 * p4 * p5
    )


class ExpressionTests(unittest.TestCase):
    """Builders and validation."""

    def test_builders_and_validation(self) -> None:
        """Names become components; invalid items and thresholds are rejected."""
        expression = series("A", parallel("B", Component("C")))
        self.assertIsInstance(expression.items[0], Component)
        with self.assertRaises(ValueError):
            series()
        with self.assertRaises(TypeError):
            parallel("A", 3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            k_out_of_n(3, "A", "B")
        with self.assertRaises(ValueError):
            k_out_of_n(0, "A", "B")
        with self.assertRaises(TypeError):
            k_out_of_n(True, "A", "B")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Component(" ")
        with self.assertRaises(TypeError):
            SystemStructure(42)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SystemStructure(series("A", "B"), order=["Z"])
        with self.assertRaises(ValueError):
            SystemStructure(series("A", "B"), order=["A", "A"])


class ReliabilityTests(unittest.TestCase):
    """Exact reliability against independent oracles."""

    def test_bridge_closed_form_and_sets(self) -> None:
        """Bridge reliability equals the textbook polynomial; sets are known."""
        structure = SystemStructure(bridge("1", "2", "3", "4", "5"))
        rng = random.Random(1)
        for _ in range(20):
            values = [rng.random() for _ in range(5)]
            p = dict(zip("12345", values, strict=True))
            self.assertAlmostEqual(
                structure.reliability(p), bridge_polynomial(*values), delta=1e-15
            )
            self.assertAlmostEqual(
                structure.unreliability(p), 1 - bridge_polynomial(*values), delta=1e-15
            )
        paths = {frozenset(s) for s in ("14", "25", "135", "234")}
        cuts = {frozenset(s) for s in ("12", "45", "135", "234")}
        self.assertEqual(set(structure.minimal_path_sets()), paths)
        self.assertEqual(set(structure.minimal_cut_sets()), cuts)
        self.assertEqual(structure.count_minimal_cut_sets(), 4)
        self.assertEqual(structure.count_minimal_path_sets(), 4)
        # Equal reliabilities 0.9: R = 2p^2 + 2p^3 - 5p^4 + 2p^5 = 0.97848.
        self.assertAlmostEqual(
            structure.reliability(dict.fromkeys("12345", 0.9)), 0.97848, delta=1e-15
        )

    def test_random_structures_against_enumeration(self) -> None:
        """Reliability, cut sets and path sets of random structures."""
        rng = random.Random(7)
        names = ("a", "b", "c", "d", "e", "f")
        for _ in range(40):
            expression = random_expression(rng, names, 3)
            structure = SystemStructure(expression)
            used = structure.components
            p = {n: Fraction(rng.randint(1, 99), 100) for n in used}
            pf = {n: float(v) for n, v in p.items()}
            exact = brute_reliability(expression, used, p)
            self.assertAlmostEqual(structure.reliability(pf), float(exact), delta=2e-15)
            self.assertAlmostEqual(
                structure.unreliability(pf), float(1 - exact), delta=2e-15
            )
            self.assertEqual(
                set(structure.minimal_cut_sets()),
                brute_minimal_sets(expression, used, cuts=True),
            )
            self.assertEqual(
                set(structure.minimal_path_sets()),
                brute_minimal_sets(expression, used, cuts=False),
            )
            for bits in itertools.product((0, 1), repeat=len(used)):
                state = dict(zip(used, bits, strict=True))
                self.assertEqual(
                    structure.structure_function(state), int(phi(expression, state))
                )

    def test_k_out_of_n_identical_matches_binomial(self) -> None:
        """k-out-of-n:G with identical p equals the binomial upper tail."""
        names = [f"c{i}" for i in range(10)]
        for k in range(1, 11):
            structure = SystemStructure(k_out_of_n(k, *names))
            p = 0.83
            expected = math.fsum(
                math.comb(10, j) * p**j * (1 - p) ** (10 - j) for j in range(k, 11)
            )
            self.assertAlmostEqual(
                structure.reliability(dict.fromkeys(names, p)), expected, delta=1e-14
            )

    def test_k_out_of_n_non_identical_matches_poisson_binomial(self) -> None:
        """Non-identical items: Poisson-binomial dynamic programming."""
        rng = random.Random(4)
        names = [f"c{i}" for i in range(9)]
        p = {n: rng.random() for n in names}
        for k in (1, 4, 9):
            distribution = [1.0]
            for n in names:
                nxt = [0.0] * (len(distribution) + 1)
                for count, mass in enumerate(distribution):
                    nxt[count] += mass * (1 - p[n])
                    nxt[count + 1] += mass * p[n]
                distribution = nxt
            expected = math.fsum(distribution[k:])
            structure = SystemStructure(k_out_of_n(k, *names))
            self.assertAlmostEqual(structure.reliability(p), expected, delta=1e-14)

    def test_k_out_of_n_of_substructures(self) -> None:
        """2-out-of-3 of shared sub-structures against enumeration."""
        expression = k_out_of_n(
            2, series("A", "B"), parallel("B", "C"), bridge("A", "C", "D", "E", "B")
        )
        structure = SystemStructure(expression)
        p = {n: Fraction(i + 3, 10) for i, n in enumerate("ABCDE")}
        exact = brute_reliability(expression, structure.components, p)
        self.assertAlmostEqual(
            structure.reliability({n: float(v) for n, v in p.items()}),
            float(exact),
            delta=1e-15,
        )

    def test_irrelevant_components_and_order(self) -> None:
        """Absorbed components are irrelevant; the order does not change R."""
        structure = SystemStructure(parallel("A", series("A", "B")))
        self.assertEqual(structure.irrelevant_components, ("B",))
        expression = bridge("a", "b", "c", "d", "e")
        p = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        reference = SystemStructure(expression).reliability(p)
        reordered = SystemStructure(expression, order=["e", "c", "a"])
        self.assertEqual(reordered.components[:3], ("e", "c", "a"))
        self.assertAlmostEqual(reordered.reliability(p), reference, delta=1e-15)
        self.assertEqual(reference, 0.766)

    def test_small_unreliability_accuracy(self) -> None:
        """A triple-redundant system with q = 1e-9 has Q = 1e-27 to 1e-14."""
        structure = SystemStructure(parallel("A", "B", "C"))
        q = dict.fromkeys("ABC", 1e-9)
        p = {n: 1 - v for n, v in q.items()}
        value = structure.unreliability(p, unreliabilities=q)
        self.assertAlmostEqual(value / 1e-27, 1.0, delta=1e-14)

    def test_probability_input_validation(self) -> None:
        """Unknown and missing components are rejected."""
        structure = SystemStructure(series("A", "B"))
        with self.assertRaises(KeyError):
            structure.reliability({"A": 0.5})
        with self.assertRaises(KeyError):
            structure.reliability({"A": 0.5, "B": 0.5, "Z": 0.5})
        with self.assertRaises(ValueError):
            structure.reliability(
                {"A": 0.5, "B": 0.5}, unreliabilities={"A": 0.5, "B": 0.6}
            )
        with self.assertRaises(KeyError):
            structure.structure_function({"A": 1, "B": 1, "Z": 0})


class FaultTreeTests(unittest.TestCase):
    """Comparison with the static fault tree of ``probstat.risk``."""

    def trees(self):
        """Return fault trees with repeated (shared) events."""
        e = {
            n: BasicEvent(n, q)
            for n, q in zip("ABCDEF", (0.1, 0.05, 0.2, 0.15, 0.3, 0.02), strict=True)
        }
        yield FaultTree(
            or_gate(
                "top",
                and_gate("g1", e["A"], e["B"]),
                and_gate("g2", e["A"], e["C"], or_gate("g3", e["D"], e["B"])),
                k_of_n_gate("g4", 2, e["C"], e["D"], e["E"]),
            )
        )
        yield FaultTree(
            and_gate(
                "top",
                or_gate("g1", e["A"], e["B"], e["F"]),
                or_gate("g2", e["A"], e["C"]),
                k_of_n_gate(
                    "g3", 2, e["B"], e["C"], e["D"], or_gate("g5", e["E"], e["A"])
                ),
            )
        )

    def test_unreliability_equals_top_event_probability(self) -> None:
        """System unreliability equals P(top) and cut sets equal MOCUS."""
        for tree in self.trees():
            structure = SystemStructure(from_fault_tree(tree))
            q = {name: event.probability for name, event in tree.basic_events.items()}
            p = {name: 1 - value for name, value in q.items()}
            self.assertAlmostEqual(
                structure.unreliability(p, unreliabilities=q),
                tree.top_event_probability(),
                delta=1e-15,
            )
            self.assertAlmostEqual(
                structure.unreliability(p, unreliabilities=q),
                tree.top_event_probability(method="inclusion_exclusion"),
                delta=1e-14,
            )
            self.assertEqual(
                set(structure.minimal_cut_sets()), set(tree.minimal_cut_sets)
            )

    def test_conversion_rejects_non_tree(self) -> None:
        """Only fault trees convert."""
        with self.assertRaises(TypeError):
            from_fault_tree("tree")  # type: ignore[arg-type]
        self.assertIsInstance(
            from_fault_tree(
                FaultTree(k_of_n_gate("t", 2, *(BasicEvent(n, 0.1) for n in "xyz")))
            ),
            KOutOfN,
        )


class BoundsTests(unittest.TestCase):
    """Esary-Proschan and min-path/min-cut bounds."""

    def test_bounds_bracket_and_are_tight_where_known(self) -> None:
        """Bounds bracket R; EP is exact for pure series and parallel."""
        rng = random.Random(9)
        for _ in range(30):
            structure = SystemStructure(
                random_expression(rng, ("a", "b", "c", "d", "e"), 3)
            )
            p = {n: rng.uniform(0.5, 0.99) for n in structure.components}
            bounds = structure.bounds(p)
            tol = 1e-14
            self.assertLessEqual(bounds.esary_proschan_lower, bounds.reliability + tol)
            self.assertLessEqual(bounds.reliability, bounds.esary_proschan_upper + tol)
            self.assertLessEqual(bounds.min_path_lower, bounds.reliability + tol)
            self.assertLessEqual(bounds.reliability, bounds.min_cut_upper + tol)
        p = {"A": 0.9, "B": 0.8, "C": 0.7}
        series_bounds = SystemStructure(series("A", "B", "C")).bounds(p)
        self.assertAlmostEqual(series_bounds.esary_proschan_lower, 0.504, delta=1e-15)
        parallel_bounds = SystemStructure(parallel("A", "B", "C")).bounds(p)
        self.assertAlmostEqual(parallel_bounds.esary_proschan_upper, 0.994, delta=1e-15)
        # Bridge with every p = 0.9: cuts {12}, {45} (q^2) and {135}, {234}
        # (q^3) give EP lower (1 - 0.01)^2 (1 - 0.001)^2.
        bridge_bounds = SystemStructure(bridge("1", "2", "3", "4", "5")).bounds(
            dict.fromkeys("12345", 0.9)
        )
        self.assertAlmostEqual(
            bridge_bounds.esary_proschan_lower, 0.99**2 * 0.999**2, delta=1e-15
        )
        self.assertAlmostEqual(
            bridge_bounds.esary_proschan_upper,
            1 - (1 - 0.81) ** 2 * (1 - 0.729) ** 2,
            delta=1e-15,
        )
        self.assertAlmostEqual(bridge_bounds.min_path_lower, 0.81, delta=1e-15)
        self.assertAlmostEqual(bridge_bounds.min_cut_upper, 0.99, delta=1e-15)


def beta_mixed(a: Fraction, b: Fraction, j: int, k: int) -> Fraction:
    """Return E[q^j (1 - q)^k] for q ~ Beta(a, b) exactly."""
    value = Fraction(1)
    for i in range(j):
        value *= a + i
    for i in range(k):
        value *= b + i
    for i in range(j + k):
        value /= a + b + i
    return value


class MomentTests(unittest.TestCase):
    """Exact epistemic moments with and without shared parameters."""

    def oracle(self, expression, names, laws, groups):
        """Return E[R], Var[R] with Beta laws and shared groups, exactly.

        ``laws`` maps a block label to its Beta(a, b); ``groups`` maps each
        component to its block label.
        """
        states = [
            dict(zip(names, bits, strict=True))
            for bits in itertools.product((0, 1), repeat=len(names))
        ]
        good = [s for s in states if phi(expression, s)]

        def expectation(counts_up, counts_down):
            value = Fraction(1)
            for label, (a, b) in laws.items():
                value *= beta_mixed(
                    a, b, counts_up.get(label, 0), counts_down.get(label, 0)
                )
            return value

        mean = Fraction(0)
        for s in good:
            up, down = {}, {}
            for n in names:
                target = up if s[n] else down
                target[groups[n]] = target.get(groups[n], 0) + 1
            mean += expectation(up, down)
        second = Fraction(0)
        for s in good:
            for t in good:
                up, down = {}, {}
                for n in names:
                    label = groups[n]
                    up[label] = up.get(label, 0) + s[n] + t[n]
                    down[label] = down.get(label, 0) + 2 - s[n] - t[n]
                second += expectation(up, down)
        return mean, second - mean * mean

    def test_independent_moments_match_rational_oracle(self) -> None:
        """Bridge with Beta(8, 2) reliabilities, independent components."""
        names = "12345"
        structure = SystemStructure(bridge(*names))
        a, b = Fraction(8), Fraction(2)
        mean_q = a / (a + b)
        var_q = a * b / ((a + b) ** 2 * (a + b + 1))
        result = structure.reliability_moments(
            dict.fromkeys(names, float(mean_q)), dict.fromkeys(names, float(var_q))
        )
        mean, variance = self.oracle(
            bridge(*names),
            tuple(names),
            {n: (a, b) for n in names},
            {n: n for n in names},
        )
        self.assertAlmostEqual(result.mean, float(mean), delta=1e-14)
        self.assertAlmostEqual(result.variance, float(variance), delta=1e-14)
        # numpy Monte Carlo with 4e6 draws (verification environment):
        # mean 0.911314 +- 0.000027, variance 0.0030008.
        self.assertAlmostEqual(result.mean, 0.911314, delta=4 * 0.000027)
        self.assertAlmostEqual(result.variance, 0.0030008, delta=2e-5)
        self.assertEqual(result.shared_groups, ())

    def test_shared_group_moments_match_rational_oracle(self) -> None:
        """Components 1 and 2 share one Beta(8, 2) value."""
        names = "12345"
        a, b = Fraction(8), Fraction(2)
        mean_q = float(a / (a + b))
        var_q = float(a * b / ((a + b) ** 2 * (a + b + 1)))
        structure = SystemStructure(bridge(*names))
        result = structure.reliability_moments(
            dict.fromkeys(names, mean_q),
            dict.fromkeys(names, var_q),
            shared={"1": "g", "2": "g"},
        )
        groups = {"1": "g", "2": "g", "3": "3", "4": "4", "5": "5"}
        laws = {label: (a, b) for label in set(groups.values())}
        mean, variance = self.oracle(bridge(*names), tuple(names), laws, groups)
        self.assertAlmostEqual(result.mean, float(mean), delta=1e-14)
        self.assertAlmostEqual(result.variance, float(variance), delta=1e-13)
        self.assertEqual(result.beta_assumed_groups, ("g",))
        # numpy Monte Carlo with 4e6 draws: mean 0.898318 +- 0.000037,
        # variance 0.0055785.
        self.assertAlmostEqual(result.mean, 0.898318, delta=4 * 0.000037)
        self.assertAlmostEqual(result.variance, 0.0055785, delta=3e-5)
        # Explicit raw moments of the same beta law give the same answer.
        raw = [float(beta_mixed(a, b, k, 0)) for k in range(1, 5)]
        explicit = structure.reliability_moments(
            dict.fromkeys(names, mean_q),
            dict.fromkeys(names, var_q),
            shared={"1": "g", "2": "g"},
            shared_moments={"g": raw},
        )
        self.assertAlmostEqual(explicit.mean, result.mean, delta=1e-14)
        self.assertAlmostEqual(explicit.variance, result.variance, delta=1e-13)
        self.assertEqual(explicit.beta_assumed_groups, ())

    def test_random_structures_with_shared_groups(self) -> None:
        """Random structures, groups of up to three, rational oracle."""
        rng = random.Random(13)
        names = ("a", "b", "c", "d", "e")
        for _ in range(8):
            expression = random_expression(rng, names, 3)
            structure = SystemStructure(expression)
            used = structure.components
            if len(used) < 3:
                continue
            members = rng.sample(list(used), min(3, len(used)))
            groups = {n: ("G" if n in members else n) for n in used}
            laws = {}
            for label in set(groups.values()):
                laws[label] = (Fraction(rng.randint(2, 9)), Fraction(rng.randint(1, 4)))
            means = {n: float(laws[groups[n]][0] / sum(laws[groups[n]])) for n in used}
            variances = {}
            for n in used:
                a, b = laws[groups[n]]
                variances[n] = float(a * b / ((a + b) ** 2 * (a + b + 1)))
            result = structure.reliability_moments(
                means, variances, shared={n: "G" for n in members}
            )
            mean, variance = self.oracle(expression, used, laws, groups)
            self.assertAlmostEqual(result.mean, float(mean), delta=1e-13)
            self.assertAlmostEqual(result.variance, float(variance), delta=1e-12)

    def test_state_of_knowledge_correlation_raises_mean_unreliability(self) -> None:
        """Two redundant units: E[Q] = E[q^2] shared versus E[q]^2."""
        structure = SystemStructure(parallel("A", "B"))
        m, v = 0.99, 1e-4  # reliability mean and variance; q = 1 - p
        independent = structure.reliability_moments({"A": m, "B": m}, {"A": v, "B": v})
        shared = structure.reliability_moments(
            {"A": m, "B": m}, {"A": v, "B": v}, shared={"A": "g", "B": "g"}
        )
        self.assertAlmostEqual(
            independent.unreliability_mean, (1 - m) ** 2, delta=1e-17
        )
        self.assertAlmostEqual(shared.unreliability_mean, (1 - m) ** 2 + v, delta=1e-16)
        self.assertGreater(shared.variance, independent.variance)

    def test_moment_validation(self) -> None:
        """Unequal group members, short raw moments and unknown groups fail."""
        structure = SystemStructure(parallel("A", "B"))
        with self.assertRaises(ValueError):
            structure.reliability_moments(
                {"A": 0.9, "B": 0.8},
                {"A": 0.01, "B": 0.01},
                shared={"A": "g", "B": "g"},
            )
        with self.assertRaises(ValueError):
            structure.reliability_moments(
                {"A": 0.9, "B": 0.9},
                {"A": 0.01, "B": 0.01},
                shared={"A": "g", "B": "g"},
                shared_moments={"g": [0.9, 0.82]},
            )
        with self.assertRaises(KeyError):
            structure.reliability_moments(
                {"A": 0.9, "B": 0.9},
                {"A": 0.01, "B": 0.01},
                shared={"A": "g", "B": "g"},
                shared_moments={"h": [0.9]},
            )
        with self.assertRaises(KeyError):
            structure.reliability_moments({"A": 0.9}, {"A": 0.01})


class ScalingTests(unittest.TestCase):
    """Large systems with repeated events finish quickly and exactly."""

    def test_consecutive_two_out_of_sixty_failed(self) -> None:
        """Consecutive-2-out-of-60:F equals the classical recursion.

        The system fails when two adjacent components fail; it is the series
        of the 59 parallel pairs, so every inner component is a repeated
        event. ``R(n) = p R(n-1) + q p R(n-2)`` with ``R(0) = R(1) = 1``.
        """
        start = time.perf_counter()
        n = 60
        names = [f"c{i}" for i in range(n)]
        structure = SystemStructure(
            series(*(parallel(names[i], names[i + 1]) for i in range(n - 1)))
        )
        for p in (0.9, 0.5, 0.999):
            q = 1 - p
            values = [1.0, 1.0]
            for _ in range(2, n + 1):
                values.append(p * values[-1] + q * p * values[-2])
            self.assertAlmostEqual(
                structure.reliability(dict.fromkeys(names, p)) / values[n],
                1.0,
                delta=1e-13,
            )
        self.assertEqual(structure.count_minimal_cut_sets(), n - 1)
        self.assertLess(time.perf_counter() - start, 5.0)

    def test_sixty_component_voting_and_bridge_ladder(self) -> None:
        """40-out-of-60 equals the binomial tail; 20 bridges in series."""
        start = time.perf_counter()
        names = [f"v{i}" for i in range(60)]
        voting = SystemStructure(k_out_of_n(40, *names))
        p = 0.7
        expected = math.fsum(
            math.comb(60, j) * p**j * (1 - p) ** (60 - j) for j in range(40, 61)
        )
        self.assertAlmostEqual(
            voting.reliability(dict.fromkeys(names, p)), expected, delta=1e-13
        )
        ladder = SystemStructure(
            series(*(bridge(*(f"b{k}_{i}" for i in range(5))) for k in range(20)))
        )
        self.assertEqual(len(ladder.components), 100)
        value = ladder.reliability(dict.fromkeys(ladder.components, 0.9))
        self.assertAlmostEqual(value / 0.97848**20, 1.0, delta=1e-13)
        self.assertLess(time.perf_counter() - start, 5.0)


if __name__ == "__main__":
    unittest.main()
