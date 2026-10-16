"""Tests for exact importance measures of ``probstat.systems``.

Oracles: brute-force enumeration over all component states in exact
rational arithmetic (conditional unreliabilities, critical state counts,
cut-set union probabilities), and ``risk.FaultTree.importance``, an
independent implementation (Shannon decomposition per conditioned event).
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from fractions import Fraction

from probstat.risk import BasicEvent, FaultTree, and_gate, k_of_n_gate, or_gate
from probstat.systems import (
    Component,
    Parallel,
    Series,
    SystemStructure,
    bridge,
    from_fault_tree,
    importance_measures,
    k_out_of_n,
    parallel,
    series,
    structural_importance,
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


def unreliability(expression, names, p, fixed=None) -> Fraction:
    """Return P(system failed | fixed states) by enumeration."""
    total = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(names)):
        state = dict(zip(names, bits, strict=True))
        if fixed and any(state[k] != v for k, v in fixed.items()):
            continue
        if phi(expression, state):
            continue
        weight = Fraction(1)
        for name in names:
            if fixed and name in fixed:
                continue
            weight *= p[name] if state[name] else 1 - p[name]
        total += weight
    return total


def cut_union(cuts, names, p) -> Fraction:
    """Return P(at least one cut set is entirely failed) by enumeration."""
    total = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(names)):
        state = dict(zip(names, bits, strict=True))
        if any(all(state[n] == 0 for n in cut) for cut in cuts):
            weight = Fraction(1)
            for name in names:
                weight *= p[name] if state[name] else 1 - p[name]
            total += weight
    return total


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


class ImportanceTests(unittest.TestCase):
    """Every measure against enumeration."""

    def check(self, expression, p_exact) -> None:
        """Compare every measure for one structure and probability table."""
        structure = SystemStructure(expression)
        names = structure.components
        p = {n: float(p_exact[n]) for n in names}
        measures = importance_measures(structure, p)
        q_system = unreliability(expression, names, p_exact)
        cuts = structure.minimal_cut_sets()
        for name in names:
            m = measures[name]
            q_failed = unreliability(expression, names, p_exact, {name: 0})
            q_perfect = unreliability(expression, names, p_exact, {name: 1})
            birnbaum = q_failed - q_perfect
            tol = 1e-14
            self.assertAlmostEqual(
                m.unreliability_if_failed, float(q_failed), delta=tol
            )
            self.assertAlmostEqual(
                m.unreliability_if_perfect, float(q_perfect), delta=tol
            )
            self.assertAlmostEqual(m.birnbaum, float(birnbaum), delta=tol)
            self.assertAlmostEqual(
                m.improvement_potential,
                float(birnbaum * (1 - p_exact[name])),
                delta=tol,
            )
            if q_system > 0:
                self.assertAlmostEqual(
                    m.criticality,
                    float(birnbaum * (1 - p_exact[name]) / q_system),
                    delta=1e-12,
                )
                # Criticality equals the fractional reduction (Q - Q(0_i)) / Q.
                self.assertAlmostEqual(
                    m.criticality, float((q_system - q_perfect) / q_system), delta=1e-12
                )
                containing = [c for c in cuts if name in c]
                union = cut_union(containing, names, p_exact)
                self.assertAlmostEqual(
                    m.fussell_vesely, float(union / q_system), delta=1e-12
                )
                self.assertGreaterEqual(m.fussell_vesely + 1e-12, m.criticality)
                self.assertAlmostEqual(
                    m.risk_achievement_worth, float(q_failed / q_system), delta=1e-11
                )
                if q_perfect > 0:
                    self.assertAlmostEqual(
                        m.risk_reduction_worth / float(q_system / q_perfect),
                        1.0,
                        delta=1e-12,
                    )
                else:
                    self.assertEqual(m.risk_reduction_worth, math.inf)
            # Structural importance: critical vectors of the others / 2^(n-1).
            others = [n for n in names if n != name]
            critical = 0
            for bits in itertools.product((0, 1), repeat=len(others)):
                state = dict(zip(others, bits, strict=True))
                if phi(expression, {**state, name: 1}) and not phi(
                    expression, {**state, name: 0}
                ):
                    critical += 1
            self.assertAlmostEqual(
                m.structural, critical / 2 ** len(others), delta=1e-15
            )

    def test_bridge(self) -> None:
        """Bridge network with distinct reliabilities."""
        p = {n: Fraction(v, 10) for n, v in zip("abcde", (9, 8, 7, 6, 5), strict=True)}
        self.check(bridge(*"abcde"), p)
        # Known structural importances of the bridge: 3/8 for the four
        # outer components and 1/8 for the bridging component.
        structural = structural_importance(SystemStructure(bridge(*"abcde")))
        self.assertEqual(structural["c"], 0.125)
        self.assertEqual(structural["a"], 0.375)

    def test_random_structures(self) -> None:
        """Random structures with repeated components."""
        rng = random.Random(31)
        names = ("a", "b", "c", "d", "e", "f")
        for _ in range(25):
            expression = random_expression(rng, names, 3)
            used = SystemStructure(expression).components
            p = {n: Fraction(rng.randint(1, 99), 100) for n in used}
            self.check(expression, p)

    def test_against_fault_tree_importance(self) -> None:
        """Birnbaum, FV, RAW and RRW equal risk.FaultTree.importance."""
        e = {
            n: BasicEvent(n, q)
            for n, q in zip("ABCDE", (0.1, 0.05, 0.2, 0.15, 0.3), strict=True)
        }
        tree = FaultTree(
            or_gate(
                "top",
                and_gate("g1", e["A"], e["B"]),
                and_gate("g2", e["A"], e["C"], or_gate("g3", e["D"], e["B"])),
                k_of_n_gate("g4", 2, e["C"], e["D"], e["E"]),
            )
        )
        reference = tree.importance()
        structure = SystemStructure(from_fault_tree(tree))
        q = {n: ev.probability for n, ev in tree.basic_events.items()}
        p = {n: 1 - v for n, v in q.items()}
        measures = importance_measures(structure, p, unreliabilities=q)
        for name, expected in reference.items():
            m = measures[name]
            self.assertAlmostEqual(m.birnbaum, expected.birnbaum, delta=1e-14)
            self.assertAlmostEqual(
                m.fussell_vesely, expected.fussell_vesely, delta=1e-13
            )
            self.assertAlmostEqual(
                m.risk_achievement_worth, expected.risk_achievement_worth, delta=1e-12
            )
            self.assertAlmostEqual(
                m.risk_reduction_worth, expected.risk_reduction_worth, delta=1e-12
            )

    def test_undefined_ratios(self) -> None:
        """A system that cannot fail reports undefined ratios as None."""
        structure = SystemStructure(parallel("A", "B"))
        measures = importance_measures(structure, {"A": 1.0, "B": 0.5})
        self.assertIsNone(measures["A"].criticality)
        self.assertIsNone(measures["A"].fussell_vesely)
        self.assertEqual(measures["B"].birnbaum, 0.0)
        # Q > 0 but a perfect component removes all risk: RRW is infinite.
        measures = importance_measures(SystemStructure(series("A")), {"A": 0.5})
        self.assertEqual(measures["A"].risk_reduction_worth, math.inf)


if __name__ == "__main__":
    unittest.main()
