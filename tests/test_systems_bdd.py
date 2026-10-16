"""Tests for the reduced ordered BDD engine of ``probstat.systems``.

Every quantity is checked against an independent oracle: brute-force
enumeration of all ``2^n`` variable assignments of a formula evaluated by
plain Python, exact rational arithmetic (``fractions.Fraction``), or a closed
form.
"""

from __future__ import annotations

import itertools
import random
import unittest
from fractions import Fraction

from probstat.systems import BDD

NAMES = ("a", "b", "c", "d", "e")


def random_formula(rng: random.Random, depth: int, monotone: bool) -> tuple:
    """Return a random formula as nested tuples."""
    if depth == 0 or rng.random() < 0.25:
        return ("var", rng.choice(NAMES))
    choices = ["and", "or"] if monotone else ["and", "or", "not", "ite"]
    op = rng.choice(choices)
    if op == "not":
        return ("not", random_formula(rng, depth - 1, monotone))
    if op == "ite":
        return (
            "ite",
            random_formula(rng, depth - 1, monotone),
            random_formula(rng, depth - 1, monotone),
            random_formula(rng, depth - 1, monotone),
        )
    return (
        op,
        random_formula(rng, depth - 1, monotone),
        random_formula(rng, depth - 1, monotone),
    )


def evaluate_formula(formula: tuple, state: dict[str, int]) -> bool:
    """Evaluate a nested-tuple formula directly (the oracle)."""
    op = formula[0]
    if op == "var":
        return bool(state[formula[1]])
    if op == "not":
        return not evaluate_formula(formula[1], state)
    if op == "ite":
        return (
            evaluate_formula(formula[2], state)
            if evaluate_formula(formula[1], state)
            else evaluate_formula(formula[3], state)
        )
    left = evaluate_formula(formula[1], state)
    right = evaluate_formula(formula[2], state)
    return (left and right) if op == "and" else (left or right)


def build(bdd: BDD, formula: tuple) -> int:
    """Build a formula in a BDD."""
    op = formula[0]
    if op == "var":
        return bdd.variable(formula[1])
    if op == "not":
        return bdd.negate(build(bdd, formula[1]))
    if op == "ite":
        return bdd.ite(
            build(bdd, formula[1]), build(bdd, formula[2]), build(bdd, formula[3])
        )
    left, right = build(bdd, formula[1]), build(bdd, formula[2])
    return bdd.apply_and(left, right) if op == "and" else bdd.apply_or(left, right)


def states():
    """Yield every assignment of NAMES."""
    for bits in itertools.product((0, 1), repeat=len(NAMES)):
        yield dict(zip(NAMES, bits, strict=True))


def enumerate_probability(formula, p, value=True, fixed=None):
    """Return P(formula = value) by enumeration with exact Fractions."""
    total = Fraction(0)
    for state in states():
        if fixed and any(state[k] != v for k, v in fixed.items()):
            continue
        weight = Fraction(1)
        for name in NAMES:
            if fixed and name in fixed:
                continue
            weight *= p[name] if state[name] else 1 - p[name]
        if evaluate_formula(formula, state) == value:
            total += weight
    return total


class BooleanOperationTests(unittest.TestCase):
    """Canonicity and Boolean operations against truth tables."""

    def test_canonical_forms(self) -> None:
        """Equivalent formulas share one node id."""
        bdd = BDD(NAMES)
        x, y, z = (bdd.variable(n) for n in "abc")
        left = bdd.apply_or(bdd.apply_and(x, y), bdd.apply_and(x, z))
        right = bdd.apply_and(x, bdd.apply_or(y, z))
        self.assertEqual(left, right)
        self.assertEqual(
            bdd.negate(bdd.apply_or(x, y)),
            bdd.apply_and(bdd.negate(x), bdd.negate(y)),
        )
        self.assertEqual(bdd.apply_and(x, bdd.negate(x)), BDD.FALSE)
        self.assertEqual(bdd.apply_or(x, bdd.negate(x)), BDD.TRUE)
        self.assertEqual(bdd.conjunction([]), BDD.TRUE)
        self.assertEqual(bdd.disjunction([]), BDD.FALSE)

    def test_random_formulas_match_truth_tables(self) -> None:
        """Forty random non-monotone formulas agree on all 32 assignments."""
        rng = random.Random(11)
        for _ in range(40):
            formula = random_formula(rng, 4, monotone=False)
            bdd = BDD(NAMES)
            node = build(bdd, formula)
            for state in states():
                self.assertEqual(
                    bdd.evaluate(node, state), evaluate_formula(formula, state)
                )
            negated = bdd.negate(node)
            self.assertEqual(bdd.negate(negated), node)
            dual = bdd.dual(node)
            self.assertEqual(bdd.dual(dual), node)
            for state in states():
                flipped = {k: 1 - v for k, v in state.items()}
                self.assertEqual(
                    bdd.evaluate(negated, state), not evaluate_formula(formula, state)
                )
                self.assertEqual(
                    bdd.evaluate(dual, state), not evaluate_formula(formula, flipped)
                )
            for name in NAMES:
                for value in (False, True):
                    restricted = bdd.restrict(node, name, value)
                    self.assertNotIn(name, bdd.support(restricted))
                    for state in states():
                        fixed = dict(state, **{name: int(value)})
                        self.assertEqual(
                            bdd.evaluate(restricted, state),
                            evaluate_formula(formula, fixed),
                        )

    def test_support_nodes_and_node_access(self) -> None:
        """Support lists exactly the variables the function depends on."""
        bdd = BDD(NAMES)
        node = bdd.apply_or(
            bdd.apply_and(bdd.variable("a"), bdd.variable("c")), bdd.variable("e")
        )
        self.assertEqual(bdd.support(node), ("a", "c", "e"))
        self.assertEqual(bdd.node_count(node), 3)
        self.assertEqual(len(bdd.nodes(node)), 3)
        name, low, high = bdd.node(node)
        self.assertEqual(name, "a")
        self.assertEqual(bdd.variables, NAMES)
        self.assertEqual(bdd.level("d"), 3)
        with self.assertRaises(ValueError):
            bdd.node(BDD.TRUE)

    def test_bryant_ordering_example(self) -> None:
        """OR_i (x_i and y_i): 2n nodes interleaved, 2^(n+1) - 2 separated.

        Bryant (1986, Fig. 3) gives these sizes for the two orders.
        """
        n = 4
        xs = [f"x{i}" for i in range(n)]
        ys = [f"y{i}" for i in range(n)]
        interleaved = BDD([v for pair in zip(xs, ys, strict=True) for v in pair])
        separated = BDD(xs + ys)
        for bdd, expected in ((interleaved, 2 * n), (separated, 2 ** (n + 1) - 2)):
            node = bdd.disjunction(
                bdd.apply_and(bdd.variable(x), bdd.variable(y))
                for x, y in zip(xs, ys, strict=True)
            )
            self.assertEqual(bdd.node_count(node), expected)

    def test_monotonicity_detection(self) -> None:
        """Monotone formulas are detected; negations are not monotone."""
        rng = random.Random(3)
        for _ in range(20):
            bdd = BDD(NAMES)
            node = build(bdd, random_formula(rng, 4, monotone=True))
            self.assertTrue(bdd.is_monotone(node))
            if node > 1:
                self.assertFalse(bdd.is_monotone(bdd.negate(node)))

    def test_deep_chain_has_no_recursion_limit(self) -> None:
        """A 3000-variable conjunction builds and evaluates iteratively."""
        names = [f"v{i}" for i in range(3000)]
        bdd = BDD(names)
        node = bdd.conjunction(bdd.variable(n) for n in reversed(names))
        self.assertEqual(bdd.node_count(node), 3000)
        p = dict.fromkeys(names, 0.9999)
        self.assertAlmostEqual(bdd.probability(node, p), 0.9999**3000, delta=1e-12)
        dual = bdd.dual(node)
        self.assertEqual(bdd.node_count(dual), 3000)
        family = bdd.minimal_solution_family(dual)
        self.assertEqual(bdd.family_count(family), 3000)


class ProbabilityTests(unittest.TestCase):
    """Exact probabilities and conditional probabilities by enumeration."""

    def test_probability_matches_exact_enumeration(self) -> None:
        """P(f = 1) and P(f = 0) agree with Fraction enumeration to 1e-15."""
        rng = random.Random(5)
        for _ in range(40):
            formula = random_formula(rng, 4, monotone=False)
            p = {n: Fraction(rng.randint(1, 99), 100) for n in NAMES}
            bdd = BDD(NAMES)
            node = build(bdd, formula)
            pf = {n: float(v) for n, v in p.items()}
            for value in (True, False):
                exact = float(enumerate_probability(formula, p, value))
                self.assertAlmostEqual(
                    bdd.probability(node, pf, value=value), exact, delta=2e-15
                )

    def test_variable_effects_match_enumeration(self) -> None:
        """Conditional probabilities and Birnbaum, including skipped levels."""
        rng = random.Random(17)
        for _ in range(40):
            formula = random_formula(rng, 4, monotone=rng.random() < 0.5)
            p = {n: Fraction(rng.randint(1, 99), 100) for n in NAMES}
            pf = {n: float(v) for n, v in p.items()}
            bdd = BDD(NAMES)
            node = build(bdd, formula)
            effects = bdd.variable_effects(node, pf)
            for name in NAMES:
                t1 = enumerate_probability(formula, p, True, {name: 1})
                t0 = enumerate_probability(formula, p, True, {name: 0})
                effect = effects[name]
                self.assertAlmostEqual(effect.true_given_one, float(t1), delta=2e-15)
                self.assertAlmostEqual(effect.true_given_zero, float(t0), delta=2e-15)
                self.assertAlmostEqual(
                    effect.false_given_one, float(1 - t1), delta=2e-15
                )
                self.assertAlmostEqual(
                    effect.false_given_zero, float(1 - t0), delta=2e-15
                )
                self.assertAlmostEqual(effect.birnbaum, float(t1 - t0), delta=2e-15)

    def test_small_failure_probabilities_keep_relative_accuracy(self) -> None:
        """With complements supplied, P(f = 0) of 1e-60 is accurate to 1e-14."""
        bdd = BDD(("x", "y", "z"))
        node = bdd.disjunction(bdd.variable(n) for n in "xyz")
        q = dict.fromkeys("xyz", 1e-20)
        p = dict.fromkeys("xyz", 1.0 - 1e-20)  # rounds to 1.0
        value = bdd.probability(node, p, q, value=False)
        self.assertAlmostEqual(value / 1e-60, 1.0, delta=1e-14)
        effects = bdd.variable_effects(node, p, q)
        self.assertAlmostEqual(effects["x"].false_given_zero / 1e-40, 1.0, delta=1e-14)
        self.assertEqual(effects["x"].false_given_one, 0.0)

    def test_constant_functions(self) -> None:
        """Constants have trivial probabilities and effects."""
        bdd = BDD(("x",))
        self.assertEqual(bdd.probability(BDD.TRUE, {}), 1.0)
        self.assertEqual(bdd.probability(BDD.FALSE, {}), 0.0)
        effect = bdd.variable_effects(BDD.TRUE, {})["x"]
        self.assertEqual((effect.true_given_one, effect.birnbaum), (1.0, 0.0))

    def test_input_validation(self) -> None:
        """Missing, out-of-range, boolean and inconsistent inputs are rejected."""
        bdd = BDD(("x", "y"))
        node = bdd.apply_and(bdd.variable("x"), bdd.variable("y"))
        with self.assertRaises(KeyError):
            bdd.probability(node, {"x": 0.5})
        with self.assertRaises(ValueError):
            bdd.probability(node, {"x": 0.5, "y": 1.5})
        with self.assertRaises(TypeError):
            bdd.probability(node, {"x": 0.5, "y": True})
        with self.assertRaises(ValueError):
            bdd.probability(node, {"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.4})
        with self.assertRaises(TypeError):
            bdd.variable(3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bdd.variable("")
        with self.assertRaises(TypeError):
            bdd.restrict(node, "x", 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bdd.negate(10_000)
        with self.assertRaises(TypeError):
            bdd.negate(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BDD(("x", "x"))
        with self.assertRaises(ValueError):
            bdd.evaluate(node, {"x": 2, "y": 1})
        with self.assertRaises(KeyError):
            bdd.variable_effects(node, {"x": 0.5, "y": 0.5}, variables=("z",))

    def test_node_limit(self) -> None:
        """The node guard raises instead of exhausting memory."""
        bdd = BDD(max_nodes=6)
        with self.assertRaises(ValueError):
            bdd.conjunction(bdd.variable(f"v{i}") for i in range(10))


class MinimalSolutionTests(unittest.TestCase):
    """Rauzy's minimal solutions against brute-force minimal true sets."""

    def test_minimal_solutions_match_brute_force(self) -> None:
        """Random monotone formulas: minsol equals minimal true sets."""
        rng = random.Random(23)
        for _ in range(40):
            formula = random_formula(rng, 4, monotone=True)
            bdd = BDD(NAMES)
            node = build(bdd, formula)
            found = set(bdd.family_sets(bdd.minimal_solution_family(node)))
            true_sets = [
                frozenset(n for n in NAMES if state[n])
                for state in states()
                if evaluate_formula(formula, state)
            ]
            minimal = {s for s in true_sets if not any(t < s for t in true_sets)}
            self.assertEqual(found, minimal)
            family = bdd.minimal_solution_family(node)
            self.assertEqual(bdd.family_count(family), len(minimal))
            # The family function recovers the original monotone function.
            self.assertEqual(bdd.family_function(family), node)
            for name in NAMES:
                onset = set(bdd.family_sets(bdd.family_onset(family, name)))
                self.assertEqual(onset, {s for s in minimal if name in s})

    def test_non_monotone_is_rejected_and_limit_enforced(self) -> None:
        """Minimal solutions need monotone input; enumeration has a guard."""
        bdd = BDD(("x", "y"))
        node = bdd.apply_and(bdd.variable("x"), bdd.negate(bdd.variable("y")))
        with self.assertRaises(ValueError):
            bdd.minimal_solution_family(node)
        family = bdd.minimal_solution_family(
            bdd.apply_or(bdd.variable("x"), bdd.variable("y"))
        )
        with self.assertRaises(ValueError):
            bdd.family_sets(family, limit=1)


def mixed_moment_independent(
    mean: Fraction, variance: Fraction, j: int, k: int
) -> Fraction:
    """Return E[p^j (1 - p)^k] for j + k <= 2 from mean and variance."""
    table = {
        (0, 0): Fraction(1),
        (1, 0): mean,
        (0, 1): 1 - mean,
        (2, 0): mean * mean + variance,
        (1, 1): mean - mean * mean - variance,
        (0, 2): (1 - mean) ** 2 + variance,
    }
    return table[(j, k)]


class MomentTests(unittest.TestCase):
    """Exact epistemic moments against a rational double enumeration."""

    def oracle(self, formula, means, variances):
        """Return E[R] and Var[R] by summing over pairs of states exactly."""
        assignments = list(states())
        true_states = [s for s in assignments if evaluate_formula(formula, s)]
        mean = Fraction(0)
        for s in true_states:
            term = Fraction(1)
            for n in NAMES:
                term *= means[n] if s[n] else 1 - means[n]
            mean += term
        second = Fraction(0)
        for s in true_states:
            for t in true_states:
                term = Fraction(1)
                for n in NAMES:
                    j = s[n] + t[n]
                    term *= mixed_moment_independent(means[n], variances[n], j, 2 - j)
                second += term
        return mean, second - mean * mean

    def test_moments_match_rational_enumeration(self) -> None:
        """E[R] and Var[R] agree with the exact rational oracle."""
        rng = random.Random(29)
        for _ in range(15):
            formula = random_formula(rng, 4, monotone=rng.random() < 0.7)
            means = {n: Fraction(rng.randint(5, 95), 100) for n in NAMES}
            variances = {
                n: means[n] * (1 - means[n]) * Fraction(rng.randint(0, 9), 10)
                for n in NAMES
            }
            bdd = BDD(NAMES)
            node = build(bdd, formula)
            mf = {n: float(v) for n, v in means.items()}
            vf = {n: float(v) for n, v in variances.items()}
            result = bdd.moments(node, mf, vf)
            mean, variance = self.oracle(formula, means, variances)
            self.assertAlmostEqual(result.mean, float(mean), delta=1e-14)
            self.assertAlmostEqual(result.complement_mean, float(1 - mean), delta=1e-14)
            self.assertAlmostEqual(result.variance, float(variance), delta=1e-14)
            blocks = {n: n for n in NAMES}
            laws = {
                n: (
                    lambda j, k, m=means[n], v=variances[n]: float(
                        mixed_moment_independent(m, v, j, k)
                    )
                )
                for n in NAMES
            }
            grouped = bdd.grouped_moments(node, blocks, laws)
            self.assertAlmostEqual(grouped.mean, float(mean), delta=1e-14)
            self.assertAlmostEqual(grouped.variance, float(variance), delta=1e-13)

    def test_moment_validation(self) -> None:
        """Variances above m(1-m) and non-contiguous blocks are rejected."""
        bdd = BDD(("x", "y", "z"))
        node = bdd.conjunction(bdd.variable(n) for n in "xyz")
        with self.assertRaises(ValueError):
            bdd.moments(node, dict.fromkeys("xyz", 0.5), dict.fromkeys("xyz", 0.3))
        with self.assertRaises(KeyError):
            bdd.moments(node, dict.fromkeys("xy", 0.5), dict.fromkeys("xyz", 0.1))
        with self.assertRaises(ValueError):
            bdd.grouped_moments(
                node,
                {"x": "g", "y": "h", "z": "g"},
                {"g": lambda j, k: 0.25, "h": lambda j, k: 0.25},
            )
        with self.assertRaises(KeyError):
            bdd.grouped_moments(node, {"x": "g", "y": "g", "z": "g"}, {})

    def test_zero_variance_gives_point_value(self) -> None:
        """With no epistemic variance the mean is R(m) and the variance 0."""
        bdd = BDD(NAMES)
        node = build(bdd, ("or", ("and", ("var", "a"), ("var", "b")), ("var", "c")))
        means = dict.fromkeys(NAMES, 0.7)
        result = bdd.moments(node, means, dict.fromkeys(NAMES, 0.0))
        self.assertAlmostEqual(result.mean, 1 - (1 - 0.49) * 0.3, delta=1e-15)
        self.assertEqual(result.variance, 0.0)


if __name__ == "__main__":
    unittest.main()
