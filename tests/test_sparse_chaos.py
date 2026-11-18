"""Tests for sparse polynomial chaos by hybrid LAR and OMP."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Normal
from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import householder_qr, inverse_spd
from probstat.modeling import (
    ISHIGAMI_FIRST_ORDER,
    ISHIGAMI_TOTAL,
    JointInput,
    RandomInput,
    flood_model,
    ishigami_model,
)
from probstat.propagation import make_design
from probstat.surrogates import (
    SPARSE_CHAOS_METHODS,
    fit_polynomial_chaos,
    fit_sparse_polynomial_chaos,
    hermite_orthonormal,
    least_angle_regression,
    multi_indices,
    polynomial_chaos_from_samples,
    sparse_polynomial_chaos_from_samples,
)

DIMENSION = 10
# Exact sparse truth in the orthonormal Hermite basis: 4 non-constant terms of
# degree <= 3 among the 285 non-constant terms of the total-degree-3 set.
TRUTH = {
    (0,) * DIMENSION: 2.0,
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0): 1.5,
    (0, 0, 2, 0, 0, 0, 0, 0, 0, 0): -1.0,
    (0, 0, 0, 0, 1, 0, 0, 1, 0, 0): 0.8,
    (0, 0, 0, 0, 0, 0, 3, 0, 0, 0): 0.5,
}


def _sparse_truth(x: tuple[float, ...]) -> float:
    """Evaluate the exact sparse Hermite expansion ``TRUTH`` at ``x``."""
    total = 0.0
    for alpha, coefficient in TRUTH.items():
        product = 1.0
        for value, a in zip(x, alpha, strict=True):
            if a:
                product *= hermite_orthonormal(value, a)[a]
        total += coefficient * product
    return total


def _normal_inputs(dimension: int) -> JointInput:
    """Return ``dimension`` independent standard normals."""
    return JointInput.independent(
        *(RandomInput(f"x{i}", Normal(0.0, 1.0)) for i in range(dimension))
    )


def _lar_example() -> tuple[list[list[float]], list[float]]:
    """Return a deterministic 20 x 6 regression problem with correlated columns."""
    rows: list[list[float]] = []
    values: list[float] = []
    for i in range(20):
        row = [math.sin(0.7 * (i + 1) * (j + 1) + 0.3 * j) for j in range(6)]
        row[3] += 0.6 * row[0]
        rows.append(row)
        values.append(
            3.0 * row[0] - 2.0 * row[3] + row[5] + 0.5 * math.cos(1.3 * i) + 4.0
        )
    return rows, values


class LeastAngleRegressionTests(unittest.TestCase):
    """The LAR path agrees with an independent implementation."""

    def test_path_matches_independent_reference(self) -> None:
        """Order, correlation levels, and knot coefficients match a numpy LARS.

        The pinned values come from a separate numpy implementation of
        Efron et al. (2004), eqs. 2.4-2.13 (signed active matrix, explicit
        inverse of its Gram matrix), on centred unit-norm columns; it agrees
        with this implementation to 4e-15.
        """
        rows, values = _lar_example()
        path = least_angle_regression(rows, values)
        self.assertEqual(path.order, (0, 4, 3, 5, 1, 2))
        expected_levels = (
            5.537179326738593,
            3.936429954443062,
            3.1118999595036945,
            2.488891396731965,
            0.7652554310492132,
            0.02690345197425878,
        )
        for got, want in zip(path.max_correlations, expected_levels, strict=False):
            self.assertAlmostEqual(got, want, delta=1e-11)
        self.assertLess(path.max_correlations[-1], 1e-12)
        expected_knots = {
            2: (2.444046986828826, 0.0, 0.0, 0.0, -0.8432976145332958, 0.0),
            4: (
                7.8342315796269055,
                0.0,
                0.0,
                -5.5199293329684265,
                -0.18394494408193662,
                1.9595707476233115,
            ),
            6: (
                9.578543216286404,
                0.6934576842863238,
                -0.07986731835412532,
                -7.336032530754482,
                0.09516171011555181,
                2.9003784969560362,
            ),
        }
        for knot, coefficients in expected_knots.items():
            for got, want in zip(path.coefficients[knot], coefficients, strict=True):
                self.assertAlmostEqual(got, want, delta=1e-10)

    def test_final_knot_is_the_least_squares_fit(self) -> None:
        """With a step limit the path ends at least squares on the entered columns."""
        rows, values = _lar_example()
        path = least_angle_regression(rows, values, max_steps=3)
        self.assertEqual(path.order, (0, 4, 3))
        standardized = [
            [(row[j] - path.means[j]) / path.scales[j] for j in path.order]
            for row in rows
        ]
        center = math.fsum(values) / len(values)
        solution = householder_qr(tuple(map(tuple, standardized))).solve(
            tuple(v - center for v in values)
        )
        for j, value in zip(path.order, solution, strict=True):
            self.assertAlmostEqual(path.coefficients[-1][j], value, delta=1e-10)
        self.assertLess(path.max_correlations[-1], 1e-12)

    def test_constant_column_never_enters(self) -> None:
        """A column constant on the rows is excluded, and bad shapes are refused."""
        rows, values = _lar_example()
        padded = [[1.0, *row] for row in rows]
        path = least_angle_regression(padded, values)
        self.assertEqual(path.excluded, (0,))
        self.assertEqual(path.order, (1, 5, 4, 6, 2, 3))
        with self.assertRaises(ValueError):
            least_angle_regression(rows, values[:-1])
        with self.assertRaises(ValueError):
            least_angle_regression(rows, values, max_steps=-1)


class CorrectedLeaveOneOutTests(unittest.TestCase):
    """The closed-form errors agree with brute-force refits."""

    def test_full_fit_matches_brute_force_refits(self) -> None:
        """``loo_error`` from refits; ``corrected_loo_error`` from an explicit trace."""
        inputs = _normal_inputs(2)
        unit = make_design("lhs", 25, 2, seed=resolve_stream(3))
        points = inputs.sample_design(unit)
        values = [math.exp(0.3 * p[0]) + p[0] * p[1] ** 2 for p in points]
        expansion = polynomial_chaos_from_samples(inputs, points, values, degree=3)
        count, terms = len(points), expansion.size
        rows = [expansion.basis_values(p) for p in points]
        squared = []
        for left_out in range(count):
            kept = [i for i in range(count) if i != left_out]
            factor = householder_qr(tuple(rows[i] for i in kept))
            coefficients = factor.solve(tuple(values[i] for i in kept))
            prediction = math.fsum(
                c * v for c, v in zip(coefficients, rows[left_out], strict=True)
            )
            squared.append((values[left_out] - prediction) ** 2)
        mean = math.fsum(values) / count
        variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
        brute_loo = math.fsum(squared) / count / variance
        self.assertAlmostEqual(expansion.loo_error / brute_loo, 1.0, delta=1e-9)
        gram = tuple(
            tuple(math.fsum(r[a] * r[b] for r in rows) for b in range(terms))
            for a in range(terms)
        )
        trace = math.fsum(inverse_spd(gram)[j][j] for j in range(terms))
        factor = count / (count - terms) * (1.0 + trace)
        self.assertAlmostEqual(
            expansion.corrected_loo_error / (brute_loo * factor), 1.0, delta=1e-9
        )
        self.assertEqual(expansion.candidate_size, terms)
        self.assertIsNone(expansion.selection)

    def test_nested_path_matches_the_final_refit(self) -> None:
        """The incremental path value equals the Householder refit's corrected LOO."""
        inputs = _normal_inputs(3)
        unit = make_design("lhs", 40, 3, seed=resolve_stream(5))
        points = inputs.sample_design(unit)
        values = [math.sin(p[0]) + 0.5 * p[1] * p[2] + 0.1 * p[2] ** 3 for p in points]
        for method in SPARSE_CHAOS_METHODS:
            expansion = sparse_polynomial_chaos_from_samples(
                inputs, points, values, degrees=(4,), method=method
            )
            chosen = expansion.selection.chosen
            self.assertAlmostEqual(
                chosen.loo_path[chosen.selected_terms] / expansion.corrected_loo_error,
                1.0,
                delta=1e-8,
            )
            self.assertEqual(
                set(expansion.indices[1:]),
                set(chosen.entry_order[: chosen.selected_terms]),
            )
            self.assertEqual(len(chosen.loo_path), len(chosen.entry_order) + 1)
            self.assertEqual(expansion.indices[0], (0, 0, 0))


class SparseRecoveryTests(unittest.TestCase):
    """An exact sparse expansion is recovered from fewer runs than terms."""

    def test_exact_sparse_polynomial_with_fewer_runs_than_terms(self) -> None:
        """Ten inputs, 4 active terms of degree <= 3, N = 60 < P = 286."""
        inputs = _normal_inputs(DIMENSION)
        unit = make_design("lhs", 60, DIMENSION, seed=resolve_stream(1))
        points = inputs.sample_design(unit)
        values = [_sparse_truth(p) for p in points]
        for method in SPARSE_CHAOS_METHODS:
            expansion = sparse_polynomial_chaos_from_samples(
                inputs, points, values, degrees=range(1, 5), method=method
            )
            self.assertEqual(expansion.degree, 3)
            self.assertEqual(expansion.candidate_size, 286)
            self.assertEqual(set(expansion.indices), set(TRUTH))
            for alpha, coefficient in zip(
                expansion.indices, expansion.coefficients, strict=True
            ):
                self.assertAlmostEqual(coefficient, TRUTH[alpha], delta=1e-12)
            self.assertLess(expansion.corrected_loo_error, 1e-20)
            self.assertEqual(expansion.selection.method, method)
            self.assertEqual(
                set(expansion.selection.entry_order[:4]), set(TRUTH) - {(0,) * 10}
            )


class BenchmarkTests(unittest.TestCase):
    """Sparse fits of the Ishigami and flood benchmarks."""

    def test_ishigami_sobol_indices_with_fewer_runs(self) -> None:
        """200 runs (the full fit of the guide used 288) give indices within 0.005."""
        model = ishigami_model()
        expansion = fit_sparse_polynomial_chaos(
            model,
            size=200,
            degrees=range(2, 13),
            q_norms=(0.75,),
            basis="legendre",
            seed=1,
        )
        self.assertEqual(model.evaluations, 200)
        self.assertLess(expansion.corrected_loo_error, 1e-4)
        self.assertLess(expansion.size, 200)
        sobol = expansion.sobol_indices()
        for name, first, total in zip(
            sobol.names, ISHIGAMI_FIRST_ORDER, ISHIGAMI_TOTAL, strict=True
        ):
            self.assertAlmostEqual(sobol.first_order[name], first, delta=0.005)
            self.assertAlmostEqual(sobol.total_order[name], total, delta=0.005)
        self.assertAlmostEqual(expansion.mean, 3.5, delta=0.01)

    def test_flood_overflow_matches_sampled_sobol_indices(self) -> None:
        """150 runs, Hermite basis on the Nataf transform of the flood inputs.

        Reference values: ``sobol_analysis(flood_model(), "overflow", 4096,
        seed=2026, replicates=8)`` in docs/foundations/sensitivity-analysis.md
        (half-widths at most 0.003).
        """
        expansion = fit_sparse_polynomial_chaos(
            flood_model(),
            "overflow",
            size=150,
            degrees=range(1, 6),
            q_norms=(0.75,),
            seed=1,
        )
        self.assertLess(expansion.corrected_loo_error, 0.01)
        sobol = expansion.sobol_indices()
        reference_first = {"Q": 0.3440, "Hd": 0.2844, "Zv": 0.1897, "Ks": 0.1338}
        reference_total = {"Q": 0.3523, "Hd": 0.2843, "Zv": 0.1902, "Ks": 0.1420}
        for name, value in reference_first.items():
            self.assertAlmostEqual(sobol.first_order[name], value, delta=0.02)
        for name, value in reference_total.items():
            self.assertAlmostEqual(sobol.total_order[name], value, delta=0.02)
        self.assertLess(sobol.total_order["L"] + sobol.total_order["B"], 0.005)

    def test_omp_sanity_on_flood(self) -> None:
        """OMP reaches a comparable error and the same input ranking."""
        expansion = fit_sparse_polynomial_chaos(
            flood_model(),
            "overflow",
            size=150,
            degrees=range(1, 6),
            q_norms=(0.75,),
            method="omp",
            seed=1,
        )
        self.assertEqual(expansion.selection.method, "omp")
        self.assertLess(expansion.corrected_loo_error, 0.01)
        first = expansion.sobol_indices().first_order
        ranking = sorted(first, key=first.__getitem__, reverse=True)
        self.assertEqual(ranking[:5], ["Q", "Hd", "Zv", "Ks", "Cb"])


class AdaptivityAndValidationTests(unittest.TestCase):
    """Degree adaptivity, early stopping, and argument checks."""

    def test_full_path_without_early_stopping(self) -> None:
        """``early_stop=False`` ranks terms up to ``N - 2`` on every set."""
        inputs = _normal_inputs(2)
        unit = make_design("lhs", 30, 2, seed=resolve_stream(2))
        points = inputs.sample_design(unit)
        values = [math.cos(p[0]) * (1.0 + p[1]) for p in points]
        expansion = sparse_polynomial_chaos_from_samples(
            inputs, points, values, degrees=(2, 6), early_stop=False
        )
        degrees = [c.degree for c in expansion.selection.candidates]
        self.assertEqual(degrees, [2, 6])
        self.assertEqual(len(expansion.selection.candidates[0].entry_order), 5)
        self.assertEqual(len(expansion.selection.candidates[1].entry_order), 27)
        best = expansion.selection.chosen
        self.assertEqual(
            best.corrected_loo_error,
            min(c.corrected_loo_error for c in expansion.selection.candidates),
        )

    def test_existing_fit_reports_corrected_error(self) -> None:
        """The full fit gains a corrected LOO error larger than the plain one."""
        expansion = fit_polynomial_chaos(
            ishigami_model(), degree=4, basis="legendre", seed=1
        )
        self.assertGreater(expansion.corrected_loo_error, expansion.loo_error)
        self.assertEqual(expansion.candidate_size, expansion.size)

    def test_invalid_arguments_are_refused(self) -> None:
        """Unknown methods, empty grids, bad norms, and tiny designs raise."""
        inputs = _normal_inputs(2)
        points = [(0.1 * i, -0.2 * i) for i in range(10)]
        values = [p[0] + p[1] ** 2 for p in points]
        cases = (
            {"degrees": (2,), "method": "lasso"},
            {"degrees": ()},
            {"degrees": (2,), "q_norms": (1.5,)},
            {"degrees": (2,), "q_norms": ()},
            {"degrees": (-1,)},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sparse_polynomial_chaos_from_samples(inputs, points, values, **kwargs)
        with self.assertRaises(ValueError):
            sparse_polynomial_chaos_from_samples(
                inputs, points, [1.0] * 10, degrees=(2,)
            )
        with self.assertRaises(ValueError):
            fit_sparse_polynomial_chaos(ishigami_model(), size=2, degrees=(2,))

    def test_hyperbolic_enumeration_matches_brute_force(self) -> None:
        """The pruned enumeration equals filtering the full total-degree set."""
        for dimension, degree, q in ((3, 6, 0.5), (4, 5, 0.75), (5, 4, 0.4)):
            full = multi_indices(dimension, degree, q_norm=1.0)
            expected = tuple(
                alpha
                for alpha in full
                if math.fsum(a**q for a in alpha if a) ** (1.0 / q)
                <= degree + 1e-12 * degree
            )
            self.assertEqual(multi_indices(dimension, degree, q_norm=q), expected)


if __name__ == "__main__":
    unittest.main()
