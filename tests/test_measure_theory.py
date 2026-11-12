"""Exact-arithmetic tests for the finite measure-theoretic bridge (ADR-0071)."""

from __future__ import annotations

import math
import re
import unittest
from fractions import Fraction
from itertools import product
from pathlib import Path

from _documentation import run_document

from probstat.probability import (
    CompletenessCheck,
    FiniteMeasurableSpace,
    FiniteMeasure,
    FiniteProbabilityMeasure,
    FiniteSampleSpace,
    FubiniCheck,
    check_completeness,
    is_complete_statistic,
    is_sigma_algebra,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
FOUNDATIONS = REPOSITORY_ROOT / "docs" / "foundations"
MEASURE_GUIDE = FOUNDATIONS / "measure-theoretic-probability.md"
INTERPRETATION_GUIDE = FOUNDATIONS / "frequentist-and-bayesian-interpretation.md"
BLOCK_PATTERN = re.compile(r"^```(\w+)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
DECIMAL_PATTERN = re.compile(r"(?<![\w.])\d+\.\d{3,}(?![\w.])")
FACES = tuple(range(1, 7))


def fair_die() -> FiniteMeasure[int]:
    """Return the exact uniform measure on six faces.

    Returns:
        The fair-die measure on the power set.

    References:
        [EQ-MEAS-002]. Test fixture only.

    """
    return FiniteMeasure.from_point_masses({face: Fraction(1, 6) for face in FACES})


def loaded_die() -> FiniteMeasure[int]:
    """Return an exact loaded-die measure with a heavy six.

    Returns:
        Masses ``1/10`` on faces one to five and ``1/2`` on six.

    References:
        [EQ-MEAS-002]. Test fixture only.

    """
    masses = {face: Fraction(1, 10) for face in FACES[:-1]}
    masses[6] = Fraction(1, 2)
    return FiniteMeasure.from_point_masses(masses)


def bernoulli_family(
    trials: int, probabilities: list[Fraction] | list[float]
) -> dict[object, dict[tuple[int, ...], Fraction | float]]:
    """Return i.i.d. Bernoulli laws of a full sample, one per probability.

    Args:
        trials: Number of trials ``n``.
        probabilities: Success probabilities.

    Returns:
        Parameter-to-distribution mapping over ``{0, 1}^n``.

    References:
        [EQ-MEAS-010]. Test fixture only.

    """
    samples = list(product((0, 1), repeat=trials))
    return {
        p: {s: p ** sum(s) * (1 - p) ** (trials - sum(s)) for s in samples}
        for p in probabilities
    }


def guide_pairs(path: Path) -> list[tuple[str, str]]:
    """Return ``(code, expected_output)`` pairs from a guide.

    Args:
        path: Markdown guide whose Python blocks are followed by text blocks.

    Returns:
        Ordered pairs of source and documented standard output.

    References:
        Project documentation contract. No equation is evaluated.

    """
    blocks = BLOCK_PATTERN.findall(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for index, (language, body) in enumerate(blocks):
        if language == "python":
            if index + 1 >= len(blocks) or blocks[index + 1][0] != "text":
                raise AssertionError(f"{path.name}: block {index} lacks output")
            pairs.append((body, blocks[index + 1][1]))
    return pairs


def run_guide(path: Path) -> tuple[list[tuple[str, str]], dict[str, object]]:
    """Execute every Python block of a guide in one shared namespace.

    Args:
        path: Markdown guide.

    Returns:
        Pairs of (actual, documented) outputs and the final namespace.

    The guide is executed once per test run by the shared documentation
    harness (``tests/_documentation.py``, ADR-0095), whose cached result is
    reused here.

    References:
        Project documentation contract [REF-PYTHON-AST]. No equation is
        evaluated.

    """
    guide_pairs(path)
    run = run_document(path)
    if run.failures:
        raise AssertionError("\n\n".join(run.failures))
    outputs = [
        (printed.rstrip(), (documented or "").rstrip())
        for printed, documented in run.outputs
    ]
    return outputs, run.namespace


class SigmaAlgebraTests(unittest.TestCase):
    """Generated sigma-algebras, atoms, and closure checks."""

    def test_generated_sizes_are_powers_of_two_in_the_atom_count(self) -> None:
        """``|sigma(G)| = 2^k`` and the enumeration is a sigma-algebra."""
        cases = (
            (None, 6),
            ((), 1),
            (({2, 4, 6},), 2),
            (({2, 4, 6}, {1, 2, 3}), 4),
            (({1}, {2}, {3}, {4}, {5}), 6),
            (({1, 2}, {2, 3}, {5, 6}), 5),
        )
        for generators, atoms in cases:
            with self.subTest(generators=generators):
                space = FiniteMeasurableSpace(FACES, generators)
                self.assertEqual(len(space.atoms), atoms)
                events = list(space.events())
                self.assertEqual(len(events), 2**atoms)
                self.assertEqual(len(set(events)), 2**atoms)
                self.assertEqual(space.event_count, 2**atoms)
                self.assertTrue(is_sigma_algebra(FACES, events))
                for generator in generators or ():
                    self.assertTrue(space.is_measurable(generator))

    def test_generated_sigma_algebra_is_the_smallest_containing_generators(
        self,
    ) -> None:
        """Every sigma-algebra containing the generators contains ``sigma(G)``."""
        generators = ({1, 2}, {2, 3})
        generated = set(FiniteMeasurableSpace(FACES, generators).events())
        universe = frozenset(FACES)
        closure = {frozenset(g) for g in generators} | {universe}
        while True:
            grown = closure | {universe - a for a in closure}
            grown |= {a | b for a in grown for b in grown}
            if grown == closure:
                break
            closure = grown
        self.assertEqual(closure, generated)
        bigger = FiniteMeasurableSpace(FACES, [*generators, {6}])
        self.assertTrue(generated <= set(bigger.events()))
        self.assertTrue(
            FiniteMeasurableSpace(FACES, generators).is_sub_sigma_algebra_of(bigger)
        )

    def test_closure_check_rejects_incomplete_families(self) -> None:
        """Missing complements or unions are detected."""
        self.assertTrue(is_sigma_algebra(range(3), [set(), {0}, {1, 2}, {0, 1, 2}]))
        self.assertFalse(is_sigma_algebra(range(3), [set(), {0}, {0, 1, 2}]))
        self.assertFalse(is_sigma_algebra(range(3), [set(), {0}, {1}, {1, 2}]))
        self.assertFalse(
            is_sigma_algebra(
                range(4), [set(), {0}, {1}, {1, 2, 3}, {0, 2, 3}, {0, 1, 2, 3}]
            )
        )
        with self.assertRaises(ValueError):
            is_sigma_algebra(range(3), [{7}])

    def test_measurability_and_atoms(self) -> None:
        """Measurable sets are unions of atoms."""
        space = FiniteMeasurableSpace(FACES, [{2, 4, 6}, {1, 2, 3}])
        self.assertEqual(space.atom_of(3), frozenset({1, 3}))
        self.assertTrue(space.is_measurable({1, 3, 5}))
        self.assertFalse(space.is_measurable({1}))
        self.assertTrue(space.is_measurable(()))
        event = FiniteSampleSpace(FACES)
        self.assertTrue(space.is_measurable(event.certain_event))
        with self.assertRaises(ValueError):
            space.is_measurable({9})
        with self.assertRaises(ValueError):
            space.atom_of(9)

    def test_partitions_maps_and_sub_sigma_algebras(self) -> None:
        """Partition and map constructors and refinement order."""
        parity = FiniteMeasurableSpace.generated_by_map(FACES, lambda f: f % 2)
        cells = FiniteMeasurableSpace.from_partition(FACES, [{1, 3, 5}, {2, 4, 6}])
        self.assertEqual(parity, cells)
        self.assertEqual(hash(parity), hash(cells))
        full = FiniteMeasurableSpace.power_set(FACES)
        trivial = FiniteMeasurableSpace.trivial(FACES)
        self.assertTrue(trivial.is_sub_sigma_algebra_of(parity))
        self.assertTrue(parity.is_sub_sigma_algebra_of(full))
        self.assertFalse(full.is_sub_sigma_algebra_of(parity))
        with self.assertRaises(ValueError):
            parity.is_sub_sigma_algebra_of(FiniteMeasurableSpace(range(6)))

    def test_invalid_spaces_raise(self) -> None:
        """Empty, repeated, unhashable, or foreign inputs are rejected."""
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace([])
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace([1, 1])
        with self.assertRaises(TypeError):
            FiniteMeasurableSpace([[1]])
        with self.assertRaises(TypeError):
            FiniteMeasurableSpace("abc")
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace(FACES, [{7}])
        with self.assertRaises(TypeError):
            FiniteMeasurableSpace(FACES, [5])
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace.from_partition(FACES, [{1, 2}, {2, 3, 4, 5, 6}])
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace.from_partition(FACES, [{1, 2}, {3}])
        with self.assertRaises(ValueError):
            FiniteMeasurableSpace.from_partition(FACES, [set(), set(FACES)])


class MeasurableMapTests(unittest.TestCase):
    """Measurable maps, functions, and image measures."""

    def test_non_measurable_maps_are_rejected(self) -> None:
        """A map splitting a source atom is not measurable."""
        parity = FiniteMeasurableSpace.generated_by_map(FACES, lambda f: f % 2)
        target = FiniteMeasurableSpace(["low", "high"])
        low_high = {f: "low" if f <= 3 else "high" for f in FACES}
        self.assertFalse(parity.is_measurable_map(low_high, target))
        measure = fair_die().restrict(parity)
        with self.assertRaises(ValueError):
            measure.pushforward(low_high, target)
        coarse_target = FiniteMeasurableSpace.trivial(["low", "high"])
        self.assertTrue(parity.is_measurable_map(low_high, coarse_target))
        with self.assertRaises(ValueError):
            parity.is_measurable_map(lambda f: "other", target)
        self.assertFalse(parity.is_measurable_function(lambda f: f))
        self.assertTrue(parity.is_measurable_function(lambda f: (-1) ** f))

    def test_pushforward_is_the_law_of_the_map(self) -> None:
        """``(mu o f^-1)(B) = mu(f^-1(B))`` for every target event."""
        measure = loaded_die()
        target = FiniteMeasurableSpace(range(3))
        mapping = {f: f % 3 for f in FACES}
        law = measure.pushforward(mapping, target)
        for event in target.events():
            self.assertEqual(
                law.measure_of(event),
                measure.measure_of(measure.space.preimage(mapping, event)),
            )
        self.assertEqual(law.total_mass, 1)


class MeasureAndIntegralTests(unittest.TestCase):
    """Measures from atom masses and the Lebesgue integral."""

    def test_measure_is_additive_and_exact(self) -> None:
        """Event measures are exact sums of atom masses."""
        measure = loaded_die()
        self.assertIsInstance(measure.total_mass, Fraction)
        self.assertTrue(measure.is_probability)
        events = list(measure.space.events())
        for first in events[:16]:
            for second in events[:16]:
                if first.isdisjoint(second):
                    self.assertEqual(
                        measure.measure_of(first | second),
                        measure.measure_of(first) + measure.measure_of(second),
                    )
        self.assertEqual(measure.measure_of({5, 6}), Fraction(3, 5))
        self.assertEqual(FiniteMeasure.counting(FACES).measure_of({1, 2}), 2)

    def test_coarse_measures_reject_non_measurable_events(self) -> None:
        """Subsets splitting an atom have no measure."""
        space = FiniteMeasurableSpace(FACES, [{1, 2, 3}])
        measure = FiniteMeasure(
            space, {frozenset({1, 2, 3}): 2, frozenset({4, 5, 6}): 1}
        )
        self.assertFalse(measure.is_probability)
        self.assertEqual(measure.measure_of({1, 2, 3}), 2)
        with self.assertRaises(ValueError):
            measure.measure_of({1})
        with self.assertRaises(ValueError):
            measure.integrate(lambda f: f)

    def test_invalid_measures_raise(self) -> None:
        """Keys must be the atoms and masses non-negative finite reals."""
        space = FiniteMeasurableSpace([0, 1])
        with self.assertRaises(ValueError):
            FiniteMeasure(space, {frozenset({0}): 1})
        with self.assertRaises(ValueError):
            FiniteMeasure(space, {frozenset({0}): -1, frozenset({1}): 2})
        with self.assertRaises(TypeError):
            FiniteMeasure(space, {frozenset({0}): True, frozenset({1}): 0})
        with self.assertRaises(ValueError):
            FiniteMeasure(space, {frozenset({0}): math.inf, frozenset({1}): 0})
        with self.assertRaises(TypeError):
            FiniteMeasure(space, {frozenset({0}): "1", frozenset({1}): 0})
        with self.assertRaises(ValueError):
            FiniteMeasure(space, {frozenset({0, 1}): 1})
        with self.assertRaises(TypeError):
            FiniteMeasure("space", {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FiniteMeasure.from_point_masses({0: 1, 1: -1})

    def test_bridge_from_finite_probability_measure(self) -> None:
        """A ``FiniteProbabilityMeasure`` converts with equal probabilities."""
        sample = FiniteSampleSpace(["a", "b", "c"])
        legacy = FiniteProbabilityMeasure(sample, {"a": 0.25, "b": 0.25, "c": 0.5})
        measure = FiniteMeasure.from_probability_measure(legacy)
        self.assertTrue(measure.is_probability)
        self.assertEqual(measure.measure_of({"a", "c"}), 0.75)
        self.assertEqual(
            measure.integrate({"a": 1, "b": 2, "c": 3}),
            legacy.expectation({"a": 1, "b": 2, "c": 3}.__getitem__),
        )
        with self.assertRaises(TypeError):
            FiniteMeasure.from_probability_measure(measure)  # type: ignore[arg-type]

    def test_integrals_equal_finite_sums(self) -> None:
        """``integral f dmu`` equals ``sum f(omega) m(omega)`` exactly."""
        masses = {face: Fraction(face, 21) for face in FACES}
        measure = FiniteMeasure.from_point_masses(masses)
        for function in (lambda f: f, lambda f: f * f - 3, lambda f: Fraction(1, f)):
            expected = sum((function(f) * masses[f] for f in FACES), Fraction(0))
            self.assertEqual(measure.integrate(function), expected)
        identity = {f: f for f in FACES}
        layer_cake = [(1, {f for f in FACES if f >= k}) for k in FACES]
        self.assertEqual(
            measure.simple_integral(layer_cake), measure.integrate(identity)
        )
        self.assertEqual(
            measure.integrate_over(identity, {2, 4, 6}),
            sum((f * masses[f] for f in (2, 4, 6)), Fraction(0)),
        )
        self.assertEqual(measure.simple_integral([]), 0)
        with self.assertRaises(ValueError):
            measure.integrate({1: 1})

    def test_float_inputs_are_accepted(self) -> None:
        """Floats give float results with tolerance-based comparisons."""
        measure = FiniteMeasure.from_point_masses({0: 0.1, 1: 0.2, 2: 0.7})
        self.assertTrue(measure.is_probability)
        self.assertIsInstance(measure.total_mass, float)
        self.assertAlmostEqual(measure.integrate(lambda w: w), 1.6, places=14)


class ProductMeasureTests(unittest.TestCase):
    """Product measures and the Fubini-Tonelli theorem."""

    def test_product_masses_and_fubini(self) -> None:
        """All three integrals coincide exactly for several functions."""
        die = loaded_die()
        coin = FiniteMeasure.from_point_masses(
            {"H": Fraction(1, 3), "T": Fraction(2, 3)}
        )
        joint = die.product(coin)
        self.assertEqual(len(joint.space.atoms), 12)
        self.assertEqual(joint.measure_of({(6, "H")}), Fraction(1, 6))
        self.assertEqual(
            joint.measure_of({(f, c) for f in (1, 2) for c in "HT"}),
            die.measure_of({1, 2}),
        )
        functions = (
            lambda pair: pair[0],
            lambda pair: pair[0] ** 2 * (3 if pair[1] == "H" else -1),
            lambda pair: Fraction(1, pair[0]) - (pair[1] == "T"),
        )
        for function in functions:
            check = die.fubini_check(function, coin)
            self.assertIsInstance(check, FubiniCheck)
            self.assertTrue(check.agree)
            self.assertEqual(check.product_integral, check.outer_first)
            self.assertEqual(check.product_integral, check.outer_second)
            brute = sum(
                (
                    function((f, c)) * joint.measure_of({(f, c)})
                    for f in FACES
                    for c in "HT"
                ),
                Fraction(0),
            )
            self.assertEqual(check.product_integral, brute)

    def test_coarse_products_require_product_measurability(self) -> None:
        """A function splitting a product atom is rejected."""
        parity = fair_die().restrict(
            FiniteMeasurableSpace.generated_by_map(FACES, lambda f: f % 2)
        )
        coin = FiniteMeasure.from_point_masses(
            {"H": Fraction(1, 2), "T": Fraction(1, 2)}
        )
        self.assertEqual(len(parity.product(coin).space.atoms), 4)
        self.assertTrue(
            parity.fubini_check(lambda p: p[0] % 2 + (p[1] == "H"), coin).agree
        )
        with self.assertRaises(ValueError):
            parity.fubini_check(lambda p: p[0], coin)
        with self.assertRaises(TypeError):
            parity.product("coin")  # type: ignore[arg-type]


class RadonNikodymTests(unittest.TestCase):
    """Absolute continuity, densities, and the chain rule."""

    def test_derivative_reproduces_the_measure(self) -> None:
        """``nu(A) = integral_A (d nu / d mu) dmu`` for every event."""
        mu, nu = fair_die(), loaded_die()
        self.assertTrue(nu.is_absolutely_continuous_wrt(mu))
        density = nu.radon_nikodym_derivative(mu)
        self.assertEqual(density[6], 3)
        self.assertEqual(density[1], Fraction(3, 5))
        for event in mu.space.events():
            self.assertEqual(nu.measure_of(event), mu.integrate_over(density, event))
        self.assertEqual(mu.with_density(density).atom_masses, nu.atom_masses)

    def test_chain_rule_and_null_sets(self) -> None:
        """``d lambda / d mu = (d lambda / d nu)(d nu / d mu)`` mu-a.e."""
        mu = fair_die()
        nu = FiniteMeasure.from_point_masses(
            {1: Fraction(1, 2), 2: Fraction(1, 4), 3: Fraction(1, 4), 4: 0, 5: 0, 6: 0}
        )
        lam = FiniteMeasure.from_point_masses(
            {1: 0, 2: Fraction(1, 3), 3: Fraction(2, 3), 4: 0, 5: 0, 6: 0}
        )
        outer = lam.radon_nikodym_derivative(nu)
        inner = nu.radon_nikodym_derivative(mu)
        direct = lam.radon_nikodym_derivative(mu)
        for face in FACES:
            self.assertEqual(outer[face] * inner[face], direct[face])
        self.assertEqual(inner[4], 0)
        self.assertFalse(mu.is_absolutely_continuous_wrt(nu))
        with self.assertRaises(ValueError):
            mu.radon_nikodym_derivative(nu)
        counting = FiniteMeasure.counting(FACES)
        pmf = loaded_die().radon_nikodym_derivative(counting)
        self.assertEqual(pmf[6], Fraction(1, 2))

    def test_invalid_densities_and_spaces(self) -> None:
        """Densities must be non-negative, measurable, on one space."""
        mu = fair_die()
        with self.assertRaises(ValueError):
            mu.with_density(lambda f: f - 2)
        coarse = mu.restrict(FiniteMeasurableSpace.trivial(FACES))
        with self.assertRaises(ValueError):
            coarse.with_density(lambda f: f)
        with self.assertRaises(ValueError):
            mu.radon_nikodym_derivative(coarse)
        with self.assertRaises(TypeError):
            mu.radon_nikodym_derivative("mu")  # type: ignore[arg-type]


class ConditionalExpectationTests(unittest.TestCase):
    """Conditional expectation given a sub-sigma-algebra."""

    def setUp(self) -> None:
        """Build a die, three nested sigma-algebras, and a random variable.

        References:
            [EQ-MEAS-008]. Test fixture only.

        """
        self.measure = loaded_die()
        self.fine = FiniteMeasurableSpace(FACES, [{2, 4, 6}, {1, 2, 3}])
        self.parity = FiniteMeasurableSpace.generated_by_map(FACES, lambda f: f % 2)
        self.trivial = FiniteMeasurableSpace.trivial(FACES)
        self.value = {face: face * face for face in FACES}

    def test_defining_property_holds_exactly(self) -> None:
        """``integral_A E[X | G] dmu = integral_A X dmu`` for all ``A in G``."""
        for sub in (self.fine, self.parity, self.trivial):
            expectation = self.measure.conditional_expectation(self.value, sub)
            self.assertTrue(sub.is_measurable_function(expectation))
            for event in sub.events():
                self.assertEqual(
                    self.measure.integrate_over(expectation, event),
                    self.measure.integrate_over(self.value, event),
                )
        even = self.measure.conditional_expectation(self.value, self.parity)[2]
        self.assertEqual(even, Fraction(4 + 16 + 180, 7))

    def test_tower_property_and_known_factors(self) -> None:
        """Tower property and taking out what is known hold exactly."""
        self.assertTrue(
            self.measure.verify_tower_property(self.value, self.fine, self.parity)
        )
        self.assertTrue(
            self.measure.verify_tower_property(self.value, self.parity, self.trivial)
        )
        nested = self.measure.conditional_expectation(
            self.measure.conditional_expectation(self.value, self.fine), self.parity
        )
        self.assertEqual(
            dict(nested),
            dict(self.measure.conditional_expectation(self.value, self.parity)),
        )
        self.assertTrue(
            self.measure.verify_taking_out_what_is_known(
                self.value, lambda f: 2 + f % 2, self.parity
            )
        )
        with self.assertRaises(ValueError):
            self.measure.verify_taking_out_what_is_known(
                self.value, lambda f: f, self.parity
            )
        with self.assertRaises(ValueError):
            self.measure.verify_tower_property(self.value, self.parity, self.fine)

    def test_null_atoms_and_invalid_sub_sigma_algebras(self) -> None:
        """Null atoms take the a.e. convention; non-subalgebras are rejected."""
        measure = FiniteMeasure.from_point_masses(
            {1: Fraction(1, 2), 2: 0, 3: Fraction(1, 2), 4: 0, 5: 0, 6: 0}
        )
        expectation = measure.conditional_expectation(self.value, self.parity)
        self.assertEqual(expectation[2], 0)
        self.assertEqual(expectation[1], 5)
        coarse = self.measure.restrict(self.parity)
        with self.assertRaises(ValueError):
            coarse.conditional_expectation(lambda f: f % 2, self.fine)
        with self.assertRaises(TypeError):
            self.measure.conditional_expectation(self.value, "G")  # type: ignore[arg-type]


class TotalVariationTests(unittest.TestCase):
    """Total-variation distance against an exhaustive event search."""

    def test_total_variation_equals_supremum_over_events(self) -> None:
        """``(1/2) sum |P - Q|`` equals ``max_A |P(A) - Q(A)|``."""
        p, q = fair_die(), loaded_die()
        search = max(abs(p.measure_of(e) - q.measure_of(e)) for e in p.space.events())
        self.assertEqual(p.total_variation_distance(q), search)
        self.assertEqual(p.total_variation_distance(q), Fraction(1, 3))
        self.assertEqual(p.total_variation_distance(p), 0)
        with self.assertRaises(ValueError):
            FiniteMeasure.counting(FACES).total_variation_distance(p)


class CompletenessTests(unittest.TestCase):
    """Completeness of statistics for finite parametric families."""

    def test_binomial_count_is_complete_with_n_plus_one_parameters(self) -> None:
        """The count is complete iff at least ``n + 1`` distinct ``p``."""
        for n in range(1, 5):
            grid = [Fraction(k, n + 2) for k in range(1, n + 2)]
            with self.subTest(n=n):
                full = check_completeness(bernoulli_family(n, grid), sum)
                self.assertIsInstance(full, CompletenessCheck)
                self.assertTrue(full.is_complete)
                self.assertTrue(full.exact)
                self.assertEqual(full.rank, n + 1)
                self.assertIsNone(full.witness)
                short = check_completeness(bernoulli_family(n, grid[:-1]), sum)
                self.assertFalse(short.is_complete)
                self.assertEqual(short.rank, n)
                self.assert_unbiased_zero(bernoulli_family(n, grid[:-1]), sum, short)

    def assert_unbiased_zero(
        self,
        family: dict[object, dict[tuple[int, ...], Fraction | float]],
        statistic: object,
        result: CompletenessCheck,
    ) -> None:
        """Assert that the witness is non-zero with zero expectation.

        Args:
            family: Distributions used for the check.
            statistic: The statistic as a callable.
            result: Completeness result with a witness.

        References:
            [EQ-MEAS-010]. Test helper only.

        """
        assert result.witness is not None
        assert callable(statistic)
        self.assertTrue(any(value != 0 for value in result.witness.values()))
        for distribution in family.values():
            mean = sum(
                (
                    result.witness[statistic(sample)] * probability
                    for sample, probability in distribution.items()
                ),
                Fraction(0),
            )
            self.assertEqual(mean, 0)

    def test_full_sample_is_not_complete(self) -> None:
        """``X_1 - X_2`` is an unbiased estimator of zero (Lehmann-Casella 1.6)."""
        grid = [Fraction(k, 7) for k in range(1, 7)]
        family = bernoulli_family(2, grid)
        result = check_completeness(family, lambda sample: sample)
        self.assertFalse(result.is_complete)
        self.assertEqual(result.rank, 3)
        self.assertEqual(len(result.statistic_values), 4)
        self.assert_unbiased_zero(family, lambda sample: sample, result)
        self.assertFalse(is_complete_statistic(family, lambda sample: sample))
        self.assertTrue(is_complete_statistic(family, sum))

    def test_float_family_and_uncharged_values(self) -> None:
        """Floats use a tolerance; values no member charges are ignored."""
        floats = bernoulli_family(3, [0.2, 0.4, 0.6, 0.8])
        result = check_completeness(floats, sum)
        self.assertTrue(result.is_complete)
        self.assertFalse(result.exact)
        degenerate = {"a": {0: Fraction(1, 2), 1: Fraction(1, 2), 2: 0}}
        check = check_completeness(degenerate, lambda w: w)
        self.assertEqual(check.statistic_values, (0, 1))
        self.assertFalse(check.is_complete)
        self.assertTrue(is_complete_statistic([{0: 1, 1: 0}], lambda w: w))

    def test_invalid_families_raise(self) -> None:
        """Empty, mismatched, negative, or unnormalized families are rejected."""
        with self.assertRaises(ValueError):
            check_completeness([], sum)
        with self.assertRaises(ValueError):
            check_completeness([{0: 1}, {1: 1}], lambda w: w)
        with self.assertRaises(ValueError):
            check_completeness([{0: Fraction(3, 2), 1: Fraction(-1, 2)}], lambda w: w)
        with self.assertRaises(ValueError):
            check_completeness([{0: Fraction(1, 2), 1: Fraction(1, 3)}], lambda w: w)
        with self.assertRaises(TypeError):
            check_completeness("family", sum)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            check_completeness([[0.5, 0.5]], sum)  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            check_completeness([{0: 1}], lambda w: w, tolerance=-1.0)


class GuideTests(unittest.TestCase):
    """The two ADR-0071 guides print what they claim."""

    def test_measure_guide_snippets_print_documented_output(self) -> None:
        """Every snippet of the measure guide reproduces its output."""
        outputs, _ = run_guide(MEASURE_GUIDE)
        self.assertGreaterEqual(len(outputs), 8)
        for number, (actual, expected) in enumerate(outputs):
            with self.subTest(snippet=number):
                self.assertEqual(actual, expected)

    def test_interpretation_guide_numbers_are_reproduced_by_code(self) -> None:
        """Every quoted value in the interpretation guide is computed."""
        outputs, namespace = run_guide(INTERPRETATION_GUIDE)
        self.assertGreaterEqual(len(outputs), 6)
        for number, (actual, expected) in enumerate(outputs):
            with self.subTest(snippet=number):
                self.assertEqual(actual, expected)
        printed = "\n".join(actual for actual, _ in outputs)
        printed_numbers = set(DECIMAL_PATTERN.findall(printed))
        text = INTERPRETATION_GUIDE.read_text(encoding="utf-8")
        prose = BLOCK_PATTERN.sub("", text)
        quoted = set(DECIMAL_PATTERN.findall(prose))
        self.assertGreater(len(quoted), 15)
        self.assertEqual(sorted(quoted - printed_numbers), [])
        result = namespace["result"]
        coverage = namespace["coverage"]
        wilson_bounds = namespace["wilson_bounds"]
        exact_bounds = namespace["exact_bounds"]
        prior_averaged = namespace["prior_averaged"]
        credible = namespace["credible"]
        bf01 = namespace["bf01"]
        assert callable(coverage) and callable(prior_averaged)
        exact_wilson = coverage(wilson_bounds, 0.05)
        distance = abs(result.coverage - exact_wilson) / result.coverage_se  # type: ignore[attr-defined]
        self.assertEqual(f"{distance:.1f}", "1.2")
        assert isinstance(bf01, float)
        self.assertTrue(1.0 < 1.0 / bf01 < 3.0)
        for p in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            self.assertGreaterEqual(coverage(exact_bounds, p), 0.95)
        self.assertAlmostEqual(prior_averaged(credible), 0.95, places=8)


if __name__ == "__main__":
    unittest.main()
