"""Tests for BDD variable reordering (prescribed order and Rudell sifting).

Function preservation is checked on every assignment; probability
preservation is checked exactly with rational arithmetic on the transferred
diagrams; size reduction is checked on Bryant's interleaved example, whose
node counts are known in closed form.
"""

from __future__ import annotations

import itertools
import random
import unittest
from fractions import Fraction

from probstat.systems import BDD, ReorderResult, SystemStructure, parallel, series


def interleaved(n: int) -> tuple[BDD, int]:
    """Return ``x1 x2 + x3 x4 + ...`` in the bad order ``x1, x3, ..., x2, x4, ...``."""
    bad = [f"x{i}" for i in range(1, 2 * n, 2)] + [
        f"x{i}" for i in range(2, 2 * n + 1, 2)
    ]
    bdd = BDD(bad)
    f = bdd.disjunction(
        bdd.apply_and(bdd.variable(f"x{2 * i - 1}"), bdd.variable(f"x{2 * i}"))
        for i in range(1, n + 1)
    )
    return bdd, f


def exact_probability(bdd: BDD, root: int, p: dict[str, Fraction]) -> Fraction:
    """Return ``P(f = 1)`` by the Shannon recursion in exact rationals."""
    values: dict[int, Fraction] = {0: Fraction(0), 1: Fraction(1)}
    for node in reversed(bdd.nodes(root)):
        name, low, high = bdd.node(node)
        values[node] = (1 - p[name]) * values[low] + p[name] * values[high]
    return values[root]


def random_roots(rng: random.Random, names: list[str]) -> tuple[BDD, list[int]]:
    """Return a manager with three random (possibly non-monotone) functions."""
    bdd = BDD(names)
    nodes = [bdd.variable(name) for name in names]
    for _ in range(12):
        a, b = rng.sample(nodes, 2)
        op = rng.choice(("and", "or", "not"))
        if op == "and":
            nodes.append(bdd.apply_and(a, b))
        elif op == "or":
            nodes.append(bdd.apply_or(a, b))
        else:
            nodes.append(bdd.negate(a))
    return bdd, nodes[-3:]


class SiftingTests(unittest.TestCase):
    """Sifting preserves functions and probabilities and never grows."""

    def test_interleaved_example_reaches_linear_size(self) -> None:
        """Bryant's example drops from ``2^(n+1) - 2`` to ``2n`` nodes."""
        for n in range(2, 9):
            bdd, f = interleaved(n)
            result = bdd.sift(f)
            self.assertIsInstance(result, ReorderResult)
            self.assertEqual(result.nodes_before, 2 ** (n + 1) - 2)
            self.assertEqual(result.nodes_after, 2 * n)
            self.assertLessEqual(result.nodes_after, result.nodes_before)
            self.assertEqual(result.manager.node_count(result.roots[0]), 2 * n)
            self.assertGreater(result.swaps, 0)
            # Each pair x(2i-1), x(2i) ends up adjacent.
            positions = {name: k for k, name in enumerate(result.order)}
            for i in range(1, n + 1):
                gap = abs(positions[f"x{2 * i - 1}"] - positions[f"x{2 * i}"])
                self.assertEqual(gap, 1)

    def test_interleaved_function_and_probability_are_exact(self) -> None:
        """Every assignment and the rational probability are unchanged."""
        rng = random.Random(7)
        for n in (3, 5):
            bdd, f = interleaved(n)
            result = bdd.sift(f)
            g = result.roots[0]
            names = bdd.variables
            for bits in itertools.product((0, 1), repeat=len(names)):
                assignment = dict(zip(names, bits, strict=True))
                self.assertEqual(
                    bdd.evaluate(f, assignment), result.manager.evaluate(g, assignment)
                )
            p = {name: Fraction(rng.randint(1, 99), 100) for name in names}
            self.assertEqual(
                exact_probability(bdd, f, p), exact_probability(result.manager, g, p)
            )
            floats = {name: float(value) for name, value in p.items()}
            self.assertAlmostEqual(
                bdd.probability(f, floats),
                result.manager.probability(g, floats),
                delta=1e-15,
            )

    def test_random_shared_functions(self) -> None:
        """Random multi-root diagrams keep every function and never grow."""
        rng = random.Random(1993)
        names = [f"v{i}" for i in range(7)]
        for trial in range(60):
            bdd, roots = random_roots(rng, names)
            growth = rng.choice((None, 1.0, 1.2, 2.0))
            result = bdd.sift(roots, max_growth=growth, converge=rng.random() < 0.5)
            self.assertLessEqual(result.nodes_after, result.nodes_before, trial)
            self.assertEqual(result.nodes_before, bdd.shared_node_count(roots))
            self.assertEqual(
                result.nodes_after, result.manager.shared_node_count(result.roots)
            )
            self.assertEqual(sorted(result.order), sorted(names))
            for bits in itertools.product((0, 1), repeat=len(names)):
                assignment = dict(zip(names, bits, strict=True))
                for old, new in zip(roots, result.roots, strict=True):
                    self.assertEqual(
                        bdd.evaluate(old, assignment),
                        result.manager.evaluate(new, assignment),
                    )
            p = {name: Fraction(rng.randint(1, 9), 10) for name in names}
            for old, new in zip(roots, result.roots, strict=True):
                self.assertEqual(
                    exact_probability(bdd, old, p),
                    exact_probability(result.manager, new, p),
                )

    def test_source_manager_is_unchanged(self) -> None:
        """Sifting works on a copy: node ids and sizes of the source stay."""
        bdd, f = interleaved(4)
        size = bdd.size
        order = bdd.variables
        bdd.sift(f, max_growth=None, converge=True)
        self.assertEqual(bdd.size, size)
        self.assertEqual(bdd.variables, order)
        self.assertEqual(bdd.node_count(f), 30)

    def test_terminals_and_minimal_diagrams(self) -> None:
        """Constant roots transfer as constants; an optimal order is kept."""
        bdd = BDD(["a", "b"])
        result = bdd.sift([BDD.TRUE, BDD.FALSE])
        self.assertEqual(result.roots, (BDD.TRUE, BDD.FALSE))
        self.assertEqual(result.nodes_after, 0)
        f = bdd.apply_and(bdd.variable("a"), bdd.variable("b"))
        result = bdd.sift(f)
        self.assertEqual(result.nodes_after, 2)

    def test_structure_reliability_after_sifting(self) -> None:
        """A system BDD compiled in a poor order keeps its reliability."""
        pairs = [parallel(f"a{i}", f"b{i}") for i in range(5)]
        order = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
        structure = SystemStructure(series(*pairs), order=order)
        bdd, root = structure.bdd, structure.root
        result = bdd.sift(root)
        self.assertLess(result.nodes_after, result.nodes_before)
        p = {name: 0.9 - 0.01 * k for k, name in enumerate(order)}
        q = {name: 0.1 + 0.01 * k for k, name in enumerate(order)}
        self.assertAlmostEqual(
            result.manager.probability(result.roots[0], p, q),
            structure.reliability(p),
            delta=1e-15,
        )
        self.assertAlmostEqual(
            result.manager.probability(result.roots[0], p, q, value=False),
            structure.unreliability(p, unreliabilities=q),
            delta=1e-17,
        )

    def test_invalid_arguments(self) -> None:
        """Bad roots, growth factors and flags raise."""
        bdd, f = interleaved(2)
        with self.assertRaises(ValueError):
            bdd.sift([])
        with self.assertRaises(ValueError):
            bdd.sift(10_000)
        with self.assertRaises(ValueError):
            bdd.sift(f, max_growth=0.5)
        with self.assertRaises(TypeError):
            bdd.sift(f, converge=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bdd.sift(True)


class PrescribedReorderTests(unittest.TestCase):
    """``reorder`` rebuilds diagrams in a given order."""

    def test_good_and_bad_orders(self) -> None:
        """The natural order gives ``2n`` nodes and the bad one ``2^(n+1) - 2``."""
        n = 4
        bdd, f = interleaved(n)
        good = bdd.reorder(f, [f"x{i}" for i in range(1, 2 * n + 1)])
        self.assertEqual(good.nodes_before, 2 ** (n + 1) - 2)
        self.assertEqual(good.nodes_after, 2 * n)
        self.assertEqual(good.swaps, 0)
        back = good.manager.reorder(good.roots, bdd.variables)
        self.assertEqual(back.nodes_after, 2 ** (n + 1) - 2)
        names = bdd.variables
        for bits in itertools.product((0, 1), repeat=len(names)):
            assignment = dict(zip(names, bits, strict=True))
            self.assertEqual(
                bdd.evaluate(f, assignment),
                good.manager.evaluate(good.roots[0], assignment),
            )

    def test_partial_order_is_completed(self) -> None:
        """Unlisted variables follow in their current order."""
        bdd, f = interleaved(2)
        result = bdd.reorder(f, ["x2"])
        self.assertEqual(result.order, ("x2", "x1", "x3", "x4"))

    def test_invalid_orders(self) -> None:
        """Unknown or repeated variables raise."""
        bdd, f = interleaved(2)
        with self.assertRaises(ValueError):
            bdd.reorder(f, ["x9"])
        with self.assertRaises(ValueError):
            bdd.reorder(f, ["x1", "x1"])


if __name__ == "__main__":
    unittest.main()
