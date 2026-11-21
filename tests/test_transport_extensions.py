"""Tests for sliced and unbalanced optimal transport.

The exact one-dimensional distance between discrete laws is compared with
the transportation simplex and with the quadrature-based quantile distance;
the sliced distance with its one-dimensional special case, the translation
identity ``SW_2^2 = |t|^2/d``, and metric properties. Unbalanced Sinkhorn is
checked through the first-order conditions of its stated objective, the
equality of primal and dual values, direct perturbations of the plan, the
balanced limit against the library's Sinkhorn, and a closed-form one-atom
problem.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.foundations import RandomStream
from probstat.information import (
    discrete_wasserstein,
    empirical_wasserstein_1d,
    quantile_wasserstein,
    sinkhorn,
    sliced_wasserstein,
    squared_euclidean_cost_matrix,
    unbalanced_sinkhorn,
)


class _EmpiricalLaw:
    """Equal-weight empirical law exposing a left-continuous quantile."""

    def __init__(self, values: list[float]) -> None:
        """Store the sorted atoms."""
        self._sorted = sorted(values)

    def quantile(self, probability: float) -> float:
        """Return the smallest atom whose cumulative weight reaches ``u``."""
        n = len(self._sorted)
        index = min(n - 1, max(0, math.ceil(probability * n) - 1))
        return self._sorted[index]


class EmpiricalWassersteinTests(unittest.TestCase):
    """Exact one-dimensional Wasserstein distance between discrete laws."""

    def test_equal_sizes_by_hand(self) -> None:
        """Sorted matching: atoms 0, 1, 3 against 1, 2, 5."""
        x, y = [3.0, 0.0, 1.0], [5.0, 1.0, 2.0]
        self.assertAlmostEqual(empirical_wasserstein_1d(x, y, p=1), 4.0 / 3.0)
        self.assertAlmostEqual(empirical_wasserstein_1d(x, y), math.sqrt(2.0))
        self.assertAlmostEqual(
            empirical_wasserstein_1d(x, y, p=3), (10.0 / 3.0) ** (1.0 / 3.0)
        )

    def test_agrees_with_the_transportation_simplex(self) -> None:
        """Unequal sizes and weights match the exact linear programme."""
        rng = random.Random(1)
        for p in (1.0, 2.0, 2.5):
            for _ in range(5):
                x = [rng.gauss(0.0, 1.0) for _ in range(7)]
                y = [rng.gauss(0.5, 2.0) for _ in range(5)]
                wx = [rng.random() + 0.05 for _ in x]
                wy = [rng.random() + 0.05 for _ in y]
                wx[2] = 0.0
                ax = [w / sum(wx) for w in wx]
                ay = [w / sum(wy) for w in wy]
                self.assertAlmostEqual(
                    empirical_wasserstein_1d(
                        x, y, p=p, source_weights=wx, target_weights=wy
                    ),
                    discrete_wasserstein(x, ax, y, ay, p=p),
                    places=9,
                )
                self.assertAlmostEqual(
                    empirical_wasserstein_1d(x, y, p=p),
                    discrete_wasserstein(x, [1 / 7] * 7, y, [0.2] * 5, p=p),
                    places=9,
                )

    def test_agrees_with_the_quantile_quadrature(self) -> None:
        """The library's quantile quadrature of the step laws is reproduced.

        Quadrature of a step function converges slowly, hence the loose
        tolerance; this is why the sliced distance sorts instead.
        """
        rng = random.Random(2)
        x = [rng.gauss(0.0, 1.0) for _ in range(6)]
        y = [rng.gauss(1.0, 1.5) for _ in range(6)]
        expected = quantile_wasserstein(
            _EmpiricalLaw(x), _EmpiricalLaw(y), p=2.0, method="simpson", intervals=8192
        )
        self.assertAlmostEqual(empirical_wasserstein_1d(x, y), expected, delta=2e-3)

    def test_validation(self) -> None:
        """Empty inputs, ``p < 1``, and wrong weight counts raise."""
        with self.assertRaises(ValueError):
            empirical_wasserstein_1d([], [1.0])
        with self.assertRaises(ValueError):
            empirical_wasserstein_1d([1.0], [1.0], p=0.5)
        with self.assertRaises(ValueError):
            empirical_wasserstein_1d([1.0, 2.0], [1.0], source_weights=[1.0])
        with self.assertRaises(ValueError):
            empirical_wasserstein_1d([1.0, 2.0], [1.0], source_weights=[0.0, 0.0])


class SlicedWassersteinTests(unittest.TestCase):
    """Sliced Wasserstein distance."""

    def test_one_dimension_is_exact(self) -> None:
        """In ``d = 1`` the result equals the exact ``W_p`` with no error."""
        rng = random.Random(3)
        x = [rng.gauss(0.0, 1.0) for _ in range(9)]
        y = [rng.gauss(1.0, 2.0) for _ in range(6)]
        for p in (1.0, 2.0):
            result = sliced_wasserstein(x, [(v,) for v in y], p=p)
            exact = discrete_wasserstein(x, [1 / 9] * 9, y, [1 / 6] * 6, p=p)
            self.assertTrue(result.exact)
            self.assertEqual(result.standard_error, 0.0)
            self.assertEqual(result.distance_standard_error, 0.0)
            self.assertAlmostEqual(result.distance, exact, places=9)
            self.assertAlmostEqual(result.power, exact**p, places=9)
            self.assertAlmostEqual(
                result.distance, empirical_wasserstein_1d(x, y, p=p), places=13
            )

    def test_translation_identity(self) -> None:
        """For ``Y = X + t`` the estimate is ``|t|^2/d`` within its error."""
        rng = random.Random(4)
        for dimension, shift in ((2, (3.0, -4.0)), (5, (1.0, 0.0, -2.0, 0.5, 3.0))):
            x = [
                tuple(rng.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(12)
            ]
            y = [tuple(c + s for c, s in zip(point, shift, strict=True)) for point in x]
            expected = sum(s * s for s in shift) / dimension
            result = sliced_wasserstein(x, y, directions=2000, seed=5)
            self.assertFalse(result.exact)
            self.assertGreater(result.standard_error, 0.0)
            self.assertLess(abs(result.power - expected), 4.0 * result.standard_error)
            self.assertLess(result.standard_error, 0.05 * expected)
            self.assertAlmostEqual(
                result.distance_standard_error,
                result.standard_error / (2.0 * result.distance),
                places=14,
            )

    def test_translation_identity_holds_per_direction(self) -> None:
        """The estimate equals the mean of ``(theta't)^2`` over the directions."""
        x = [(0.0, 0.0, 0.0), (1.0, 2.0, 0.5), (-1.0, 0.3, 2.0)]
        shift = (0.5, -1.0, 2.0)
        y = [tuple(c + s for c, s in zip(point, shift, strict=True)) for point in x]
        result = sliced_wasserstein(x, y, directions=50, seed=9)
        generator = RandomStream(9).child("sliced-directions").generator()
        total = 0.0
        for _ in range(50):
            raw = [generator.gauss(0.0, 1.0) for _ in range(3)]
            norm = math.hypot(*raw)
            total += sum(r * s for r, s in zip(raw, shift, strict=True)) ** 2 / norm**2
        self.assertAlmostEqual(result.power, total / 50, places=12)

    def test_symmetry_identity_and_reproducibility(self) -> None:
        """Symmetric in its arguments, zero on equal samples, seeded."""
        rng = random.Random(6)
        x = [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(10)]
        y = [(rng.gauss(1.0, 1.0), rng.gauss(0.0, 2.0)) for _ in range(14)]
        forward = sliced_wasserstein(x, y, p=1.5, directions=64, seed=7)
        backward = sliced_wasserstein(y, x, p=1.5, directions=64, seed=7)
        self.assertAlmostEqual(forward.power, backward.power, places=13)
        self.assertEqual(
            forward, sliced_wasserstein(x, y, p=1.5, directions=64, seed=7)
        )
        other = sliced_wasserstein(x, y, p=1.5, directions=64, seed=8)
        self.assertNotEqual(other.power, forward.power)
        same = sliced_wasserstein(x, list(reversed(x)), directions=64, seed=7)
        self.assertEqual(same.distance, 0.0)
        self.assertEqual(same.standard_error, 0.0)
        self.assertEqual(same.distance_standard_error, 0.0)
        self.assertGreater(forward.distance, 0.0)

    def test_bounded_by_the_wasserstein_distance(self) -> None:
        """Every projection is 1-Lipschitz, so ``SW_p <= W_p``."""
        rng = random.Random(8)
        x = [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(8)]
        y = [(rng.gauss(2.0, 1.0), rng.gauss(-1.0, 0.5)) for _ in range(8)]
        exact = discrete_wasserstein(x, [0.125] * 8, y, [0.125] * 8, p=2.0)
        result = sliced_wasserstein(x, y, directions=400, seed=1)
        self.assertLess(result.distance, exact)
        self.assertGreater(result.distance, 0.4 * exact)

    def test_weights(self) -> None:
        """A doubled weight equals a duplicated atom."""
        x = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]
        y = [(1.0, 1.0), (2.0, -1.0)]
        weighted = sliced_wasserstein(
            x, y, directions=40, seed=3, source_weights=[2.0, 1.0, 1.0]
        )
        duplicated = sliced_wasserstein([x[0], *x], y, directions=40, seed=3)
        self.assertAlmostEqual(weighted.power, duplicated.power, places=13)

    def test_validation(self) -> None:
        """Mismatched dimensions and bad settings raise."""
        with self.assertRaises(ValueError):
            sliced_wasserstein([(0.0, 1.0)], [(0.0, 1.0, 2.0)])
        with self.assertRaises(ValueError):
            sliced_wasserstein([(0.0, 1.0)], [(0.0, 1.0)], directions=1)
        with self.assertRaises(ValueError):
            sliced_wasserstein([(0.0, 1.0)], [(0.0, 1.0)], p=0.5)
        with self.assertRaises(ValueError):
            sliced_wasserstein([], [(0.0, 1.0)])
        with self.assertRaises(ValueError):
            sliced_wasserstein([(0.0, 1.0)], [(0.0, 1.0)], source_weights=[1.0, 1.0])


def _problem(
    seed: int, m: int = 5, n: int = 6
) -> tuple[list[float], list[float], list[list[float]]]:
    """Return random weights of different total mass and a cost matrix."""
    rng = random.Random(seed)
    a = [rng.random() + 0.05 for _ in range(m)]
    b = [2.0 * rng.random() + 0.05 for _ in range(n)]
    cost = [[3.0 * rng.random() for _ in range(n)] for _ in range(m)]
    return a, b, cost


def _objective(
    plan: list[list[float]],
    a: list[float],
    b: list[float],
    cost: list[list[float]],
    eps: float,
    rho: float,
) -> float:
    """Evaluate the stated primal objective literally."""

    def kl(p: list[float], q: list[float]) -> float:
        """Return the generalized KL divergence of positive vectors."""
        return sum(u * math.log(u / v) - u + v for u, v in zip(p, q, strict=True))

    m, n = len(a), len(b)
    rows = [sum(row) for row in plan]
    columns = [sum(plan[i][j] for i in range(m)) for j in range(n)]
    flat = [plan[i][j] for i in range(m) for j in range(n)]
    reference = [a[i] * b[j] for i in range(m) for j in range(n)]
    linear = sum(cost[i][j] * plan[i][j] for i in range(m) for j in range(n))
    return linear + eps * kl(flat, reference) + rho * kl(rows, a) + rho * kl(columns, b)


class UnbalancedSinkhornTests(unittest.TestCase):
    """Unbalanced entropic transport."""

    def test_one_atom_closed_form(self) -> None:
        """One source and one target atom: a scalar fixed point.

        With ``P = a b exp((f + g - C)/eps)`` and ``f = g`` by symmetry of
        the conditions when ``a = b = 1``, ``f = -rho log P`` and
        ``log P = (2 f - C)/eps`` give ``log P = -C/(eps + 2 rho)``.
        """
        result = unbalanced_sinkhorn(
            [1.0], [1.0], [[1.2]], epsilon=0.5, marginal_penalty=2.0
        )
        self.assertTrue(result.converged)
        expected = math.exp(-1.2 / 4.5)
        self.assertAlmostEqual(result.plan[0][0], expected, places=9)
        self.assertAlmostEqual(result.transported_mass, expected, places=9)
        self.assertAlmostEqual(
            result.source_potential[0], -2.0 * math.log(expected), places=8
        )
        value = (
            1.2 * expected
            + 0.5 * (expected * math.log(expected) - expected + 1.0)
            + 4.0 * (expected * math.log(expected) - expected + 1.0)
        )
        self.assertAlmostEqual(result.primal_objective, value, places=9)

    def test_first_order_conditions_hold(self) -> None:
        """The stationarity equations of the primal objective vanish."""
        for rho in (0.1, 1.0, 10.0):
            a, b, cost = _problem(1)
            eps = 0.2
            result = unbalanced_sinkhorn(
                a, b, cost, epsilon=eps, marginal_penalty=rho, tolerance=1e-12
            )
            self.assertTrue(result.converged)
            self.assertLess(result.fixed_point_residual, 1e-11)
            self.assertLess(result.optimality_residual, 1e-9)
            rows, columns = result.source_marginal, result.target_marginal
            worst = 0.0
            for i in range(len(a)):
                for j in range(len(b)):
                    gradient = (
                        cost[i][j]
                        + eps * math.log(result.plan[i][j] / (a[i] * b[j]))
                        + rho * math.log(rows[i] / a[i])
                        + rho * math.log(columns[j] / b[j])
                    )
                    worst = max(worst, abs(gradient))
            self.assertLess(worst, 1e-9)

    def test_reported_quantities_are_consistent(self) -> None:
        """Marginals, mass, cost, and objective match literal evaluation."""
        a, b, cost = _problem(2)
        result = unbalanced_sinkhorn(a, b, cost, epsilon=0.3, marginal_penalty=1.5)
        plan = [list(row) for row in result.plan]
        m, n = len(a), len(b)
        for i in range(m):
            self.assertAlmostEqual(result.source_marginal[i], sum(plan[i]), places=14)
        for j in range(n):
            self.assertAlmostEqual(
                result.target_marginal[j], sum(plan[i][j] for i in range(m)), places=14
            )
        self.assertAlmostEqual(result.transported_mass, sum(map(sum, plan)), places=13)
        self.assertAlmostEqual(
            result.transport_cost,
            sum(cost[i][j] * plan[i][j] for i in range(m) for j in range(n)),
            places=13,
        )
        self.assertAlmostEqual(
            result.primal_objective, _objective(plan, a, b, cost, 0.3, 1.5), places=11
        )
        self.assertEqual((result.epsilon, result.marginal_penalty), (0.3, 1.5))
        self.assertEqual(len(result.trace), result.iterations)

    def test_duality_gap_vanishes_and_plan_is_a_minimizer(self) -> None:
        """Primal equals dual, and random perturbations raise the objective."""
        a, b, cost = _problem(3)
        eps, rho = 0.2, 1.0
        result = unbalanced_sinkhorn(
            a, b, cost, epsilon=eps, marginal_penalty=rho, tolerance=1e-12
        )
        self.assertAlmostEqual(result.duality_gap, 0.0, delta=1e-10)
        early = unbalanced_sinkhorn(
            a, b, cost, epsilon=eps, marginal_penalty=rho, max_iterations=2
        )
        self.assertFalse(early.converged)
        self.assertGreater(early.duality_gap, 1e-6)
        self.assertLess(early.dual_objective, result.dual_objective)
        self.assertGreater(early.primal_objective, result.primal_objective)
        base = _objective([list(r) for r in result.plan], a, b, cost, eps, rho)
        rng = random.Random(4)
        for scale in (1e-1, 1e-2):
            for _ in range(50):
                perturbed = [
                    [v * math.exp(scale * rng.gauss(0.0, 1.0)) for v in row]
                    for row in result.plan
                ]
                self.assertGreater(_objective(perturbed, a, b, cost, eps, rho), base)

    def test_balanced_limit(self) -> None:
        """As ``rho`` grows the plan tends to the balanced Sinkhorn plan."""
        a, b, cost = _problem(5)
        for mass in (1.0, 2.5):
            pa = [mass * v / sum(a) for v in a]
            pb = [mass * v / sum(b) for v in b]
            balanced = sinkhorn(pa, pb, cost, epsilon=0.2, tolerance=1e-13)
            self.assertTrue(balanced.converged)
            errors = []
            for rho in (1e2, 1e4, 1e6):
                result = unbalanced_sinkhorn(
                    pa, pb, cost, epsilon=0.2, marginal_penalty=rho, tolerance=1e-12
                )
                self.assertTrue(result.converged)
                errors.append(
                    max(
                        abs(result.plan[i][j] - balanced.plan[i][j])
                        for i in range(len(a))
                        for j in range(len(b))
                    )
                )
            self.assertLess(errors[1], errors[0] / 50.0)
            self.assertLess(errors[2], errors[1] / 50.0)
            self.assertLess(errors[2], 1e-5)
            self.assertAlmostEqual(result.transported_mass, mass, places=5)

    def test_unequal_masses_and_the_role_of_rho(self) -> None:
        """Mass lies between nothing and the larger total; it grows with rho."""
        points_x = [(0.0,), (1.0,), (2.0,)]
        points_y = [(0.5,), (1.5,), (6.0,)]
        cost = squared_euclidean_cost_matrix(points_x, points_y)
        a, b = [1.0, 1.0, 1.0], [0.5, 0.5, 4.0]
        masses = []
        for rho in (0.01, 1.0, 100.0):
            result = unbalanced_sinkhorn(a, b, cost, epsilon=0.1, marginal_penalty=rho)
            self.assertTrue(result.converged)
            masses.append(result.transported_mass)
        self.assertTrue(masses[0] < masses[1] < masses[2])
        self.assertTrue(3.0 < masses[2] < 5.0)
        # A cheap penalty lets the far target atom keep almost none of its mass.
        cheap = unbalanced_sinkhorn(a, b, cost, epsilon=0.1, marginal_penalty=1.0)
        self.assertLess(cheap.target_marginal[2], 0.05 * b[2])
        self.assertGreater(cheap.target_marginal[0], 0.5 * b[0])

    def test_translation_step_does_not_change_the_solution(self) -> None:
        """Both variants reach the same plan; the step needs fewer iterations."""
        a, b, cost = _problem(6)
        with_step = unbalanced_sinkhorn(
            a, b, cost, epsilon=0.2, marginal_penalty=10.0, tolerance=1e-12
        )
        plain = unbalanced_sinkhorn(
            a,
            b,
            cost,
            epsilon=0.2,
            marginal_penalty=10.0,
            tolerance=1e-12,
            translation_step=False,
        )
        self.assertTrue(with_step.converged and plain.converged)
        self.assertLess(with_step.iterations, plain.iterations)
        for i in range(len(a)):
            for j in range(len(b)):
                self.assertAlmostEqual(with_step.plan[i][j], plain.plan[i][j], places=9)
            self.assertAlmostEqual(
                with_step.source_potential[i], plain.source_potential[i], places=8
            )

    def test_plain_iteration_is_the_stated_scaling_update(self) -> None:
        """One log-domain step equals ``u = (1/K(b v))^lambda`` in scalings."""
        a, b, cost = _problem(7, 3, 4)
        eps, rho = 0.5, 2.0
        power = rho / (rho + eps)
        kernel = [[math.exp(-c / eps) for c in row] for row in cost]
        u = [1.0] * 3
        v = [1.0] * 4
        for _ in range(3):
            v = [
                (1.0 / sum(kernel[i][j] * a[i] * u[i] for i in range(3))) ** power
                for j in range(4)
            ]
            u = [
                (1.0 / sum(kernel[i][j] * b[j] * v[j] for j in range(4))) ** power
                for i in range(3)
            ]
        result = unbalanced_sinkhorn(
            a,
            b,
            cost,
            epsilon=eps,
            marginal_penalty=rho,
            max_iterations=3,
            tolerance=1e-300,
            translation_step=False,
        )
        self.assertFalse(result.converged)
        for i in range(3):
            self.assertAlmostEqual(
                result.source_potential[i], eps * math.log(u[i]), places=12
            )
            for j in range(4):
                self.assertAlmostEqual(
                    result.plan[i][j],
                    a[i] * u[i] * kernel[i][j] * v[j] * b[j],
                    places=12,
                )

    def test_counting_measure_convention_is_a_cost_shift(self) -> None:
        """The counting-measure objective is this one with a shifted cost.

        The scaling iteration ``u = (a/(K v))^lambda``, ``v = (b/(K'u))^lambda``
        solves the problem with entropy relative to the counting measure.
        Its plan differs from the library plan (reference ``a b'``) and
        equals the library plan for the cost ``C_ij + eps log(a_i b_j)``.
        """
        a, b = (0.5, 1.5, 0.2), (2.0, 0.3)
        cost = ((0.1, 1.0), (0.7, 0.2), (0.4, 0.9))
        eps, rho = 0.3, 0.8
        damping = rho / (rho + eps)
        kernel = [[math.exp(-c / eps) for c in row] for row in cost]
        u, v = [1.0] * 3, [1.0] * 2
        for _ in range(2000):
            u = [
                (a[i] / sum(kernel[i][j] * v[j] for j in range(2))) ** damping
                for i in range(3)
            ]
            v = [
                (b[j] / sum(kernel[i][j] * u[i] for i in range(3))) ** damping
                for j in range(2)
            ]
        counting = [[u[i] * kernel[i][j] * v[j] for j in range(2)] for i in range(3)]
        library = unbalanced_sinkhorn(
            a, b, cost, epsilon=eps, marginal_penalty=rho, tolerance=1e-13
        )
        shifted_cost = [
            [cost[i][j] + eps * math.log(a[i] * b[j]) for j in range(2)]
            for i in range(3)
        ]
        shifted = unbalanced_sinkhorn(
            a, b, shifted_cost, epsilon=eps, marginal_penalty=rho, tolerance=1e-13
        )
        self.assertAlmostEqual(library.transported_mass, 2.009034, delta=1e-5)
        self.assertAlmostEqual(sum(map(sum, counting)), 1.968920, delta=1e-5)
        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(shifted.plan[i][j], counting[i][j], delta=1e-10)

    def test_translation_step_on_very_unequal_masses_and_zero_weights(self) -> None:
        """The translation step keeps the fixed point and never lowers the dual.

        Random problems with total masses from 1e-3 to 1e3, some zero
        weights, and ``rho/eps`` up to 1e4: the plans with and without the
        step agree, and the dual objective after 1, 2, 3, 5, 8 iterations is
        non-decreasing.
        """
        rng = random.Random(2018)
        for case in range(12):
            m, n = rng.randint(2, 4), rng.randint(2, 4)
            scale_a = 10.0 ** rng.uniform(-3.0, 3.0)
            scale_b = 10.0 ** rng.uniform(-3.0, 3.0)
            a = [scale_a * rng.random() for _ in range(m)]
            b = [scale_b * rng.random() for _ in range(n)]
            a[rng.randrange(m)] = 0.0 if m > 2 else a[0]
            b[rng.randrange(n)] = 0.0 if n > 2 else b[0]
            cost = [[rng.random() for _ in range(n)] for _ in range(m)]
            eps = 10.0 ** rng.uniform(-1.5, 0.0)
            rho = 10.0 ** rng.uniform(-1.0, 2.5)
            with_step = unbalanced_sinkhorn(
                a, b, cost, epsilon=eps, marginal_penalty=rho, max_iterations=20000
            )
            plain = unbalanced_sinkhorn(
                a,
                b,
                cost,
                epsilon=eps,
                marginal_penalty=rho,
                max_iterations=20000,
                translation_step=False,
            )
            with self.subTest(case=case):
                self.assertTrue(with_step.converged)
                self.assertLess(
                    abs(with_step.duality_gap),
                    1e-9 * max(1.0, abs(with_step.primal_objective)),
                )
                if plain.converged:
                    largest = max(max(row) for row in plain.plan)
                    for row_a, row_b in zip(with_step.plan, plain.plan, strict=True):
                        for x, y in zip(row_a, row_b, strict=True):
                            self.assertAlmostEqual(x, y, delta=1e-5 * largest)
                previous = -math.inf
                for count in (1, 2, 3, 5, 8):
                    partial = unbalanced_sinkhorn(
                        a,
                        b,
                        cost,
                        epsilon=eps,
                        marginal_penalty=rho,
                        max_iterations=count,
                        tolerance=1e-300,
                    )
                    self.assertGreaterEqual(
                        partial.dual_objective,
                        previous - 1e-9 * max(1.0, abs(previous)),
                    )
                    previous = partial.dual_objective

    def test_zero_weights_and_small_epsilon(self) -> None:
        """Zero-mass atoms get no plan mass; small ``eps`` does not underflow."""
        cost = [[0.0, 50.0, 200.0], [50.0, 0.0, 50.0]]
        result = unbalanced_sinkhorn(
            [1.0, 0.0], [0.5, 0.0, 2.0], cost, epsilon=0.01, marginal_penalty=5.0
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.plan[1], (0.0, 0.0, 0.0))
        self.assertEqual(result.target_marginal[1], 0.0)
        self.assertTrue(math.isfinite(result.primal_objective))
        self.assertGreater(result.transported_mass, 0.0)
        self.assertAlmostEqual(result.duality_gap, 0.0, delta=1e-8)

    def test_validation(self) -> None:
        """Bad shapes and non-positive parameters raise."""
        cost = [[0.0, 1.0], [1.0, 0.0]]
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0], [1.0, 1.0], cost, epsilon=1.0, marginal_penalty=1.0
            )
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0, 1.0], [1.0, 1.0], cost, epsilon=0.0, marginal_penalty=1.0
            )
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0, 1.0], [1.0, 1.0], cost, epsilon=1.0, marginal_penalty=0.0
            )
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0, -1.0], [1.0, 1.0], cost, epsilon=1.0, marginal_penalty=1.0
            )
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0, 1.0],
                [1.0, 1.0],
                cost,
                epsilon=1.0,
                marginal_penalty=1.0,
                tolerance=0.0,
            )
        with self.assertRaises(ValueError):
            unbalanced_sinkhorn(
                [1.0, 1.0],
                [1.0, 1.0],
                cost,
                epsilon=1.0,
                marginal_penalty=1.0,
                max_iterations=0,
            )
        with self.assertRaises(TypeError):
            unbalanced_sinkhorn(
                [1.0, 1.0],
                [1.0, 1.0],
                cost,
                epsilon=1.0,
                marginal_penalty=1.0,
                max_iterations=2.5,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
