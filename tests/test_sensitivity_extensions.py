"""Tests of rank-based, moment-independent, kernel and active-subspace tools."""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.distributions import Normal, Poisson, Uniform
from probstat.linalg import matmul, symmetric_eigen
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.benchmarks import (
    ISHIGAMI_FIRST_ORDER,
    ishigami,
    ishigami_model,
)
from probstat.numerics import AutodiffUnsupportedError, admath
from probstat.sensitivity import (
    active_subspace,
    active_subspace_from_gradients,
    borgonovo_delta,
    chatterjee_indices,
    chatterjee_xi,
    estimate_sobol_indices,
    kernel_sobol_indices,
    model_kernel_sobol_indices,
    model_rank_sobol_indices,
    pawn_indices,
    rank_sobol_indices,
)
from probstat.sensitivity.active_subspace import _subspace_distance
from probstat.uq import QuantityOfInterest


def _ishigami_sample(
    n: int, rng: random.Random
) -> tuple[list[list[float]], list[float]]:
    """Return an i.i.d. Ishigami input-output sample."""
    rows = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(n)]
    return rows, [ishigami(*row) for row in rows]


def _linear_sample(
    n: int, rng: random.Random, weights: tuple[float, ...]
) -> tuple[list[list[float]], list[float]]:
    """Return a sample of ``Y = sum_i a_i X_i`` with standard normal inputs."""
    rows = [[rng.gauss(0.0, 1.0) for _ in weights] for _ in range(n)]
    return rows, [
        math.fsum(a * x for a, x in zip(weights, row, strict=True)) for row in rows
    ]


def _normal_pdf(y: float, mean: float, sd: float) -> float:
    """Return the normal density."""
    return math.exp(-0.5 * ((y - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


def _simpson_weights(count: int, step: float) -> list[float]:
    """Return composite Simpson weights for an odd number of nodes."""
    return [
        (1.0 if j in (0, count - 1) else 4.0 if j % 2 else 2.0) * step / 3.0
        for j in range(count)
    ]


def _exact_gaussian_delta(a1: float, a2: float) -> float:
    """Return delta_1 of ``Y = a1 X1 + a2 X2`` by two-dimensional quadrature.

    ``Y ~ N(0, a1^2 + a2^2)`` and ``Y | X1 = x ~ N(a1 x, a2^2)``.
    """
    total_sd = math.hypot(a1, a2)
    nx, ny = 121, 601
    hx = 14.0 / (nx - 1)
    result = 0.0
    for wx, i in zip(_simpson_weights(nx, hx), range(nx), strict=True):
        x = -7.0 + i * hx
        low = min(-8.0 * total_sd, a1 * x - 8.0 * abs(a2))
        high = max(8.0 * total_sd, a1 * x + 8.0 * abs(a2))
        hy = (high - low) / (ny - 1)
        inner = math.fsum(
            wy
            * abs(
                _normal_pdf(low + j * hy, 0.0, total_sd)
                - _normal_pdf(low + j * hy, a1 * x, abs(a2))
            )
            for wy, j in zip(_simpson_weights(ny, hy), range(ny), strict=True)
        )
        result += wx * _normal_pdf(x, 0.0, 1.0) * inner
    return 0.5 * result


class RankSobolTests(unittest.TestCase):
    """Rank-based first-order Sobol indices."""

    def test_matches_the_published_formula_literally(self) -> None:
        """The centred implementation equals the uncentred cyclic formula."""
        rng = random.Random(11)
        rows, y = _ishigami_sample(40, rng)
        result = rank_sobol_indices(rows, y)
        n = len(y)
        mean = sum(y) / n
        for i in range(3):
            ordered = [y[k] for k in sorted(range(n), key=lambda k, i=i: rows[k][i])]
            lag = sum(ordered[j] * ordered[(j + 1) % n] for j in range(n)) / n
            expected = (lag - mean**2) / (sum(v * v for v in y) / n - mean**2)
            self.assertAlmostEqual(result.first_order[i], expected, places=12)
        self.assertAlmostEqual(
            result.output_variance, statistics.pvariance(y), places=10
        )
        self.assertIsNone(result.standard_errors)
        self.assertEqual(result.evaluations, 0)

    def test_ishigami_indices_within_replicate_error_and_converging(self) -> None:
        """Replicate means match the exact indices; the error shrinks with n."""
        rng = random.Random(1)
        errors = {}
        for n, replicates in ((500, 16), (4000, 16)):
            estimates = [
                rank_sobol_indices(*_ishigami_sample(n, rng)).first_order
                for _ in range(replicates)
            ]
            squared = 0.0
            for i, exact in enumerate(ISHIGAMI_FIRST_ORDER):
                column = [e[i] for e in estimates]
                mean = statistics.fmean(column)
                se = statistics.stdev(column) / math.sqrt(replicates)
                self.assertLess(abs(mean - exact), 4.0 * se + 5e-4)
                squared += statistics.fmean((v - exact) ** 2 for v in column)
            errors[n] = math.sqrt(squared / 3.0)
        # sqrt(n) rate: eight times the rows give about 1/sqrt(8) the error.
        self.assertLess(errors[4000], 0.55 * errors[500])
        self.assertLess(errors[4000], 0.03)

    def test_linear_gaussian_model_closed_form(self) -> None:
        """``Y = sum a_i X_i`` gives ``S_i = a_i^2 / sum a_j^2``."""
        weights = (2.0, 1.0, 0.5)
        total = sum(a * a for a in weights)
        rng = random.Random(3)
        estimates = [
            rank_sobol_indices(*_linear_sample(2000, rng, weights)).first_order
            for _ in range(12)
        ]
        for i, a in enumerate(weights):
            column = [e[i] for e in estimates]
            se = statistics.stdev(column) / math.sqrt(len(column))
            self.assertLess(abs(statistics.fmean(column) - a * a / total), 4.0 * se)

    def test_block_standard_error_tracks_replicate_spread(self) -> None:
        """The disjoint-block standard error is close to the replicate sd."""
        rng = random.Random(5)
        results = [
            rank_sobol_indices(*_ishigami_sample(2000, rng), blocks=10)
            for _ in range(24)
        ]
        for i in range(3):
            spread = statistics.stdev(r.first_order[i] for r in results)
            reported = statistics.fmean(
                r.standard_errors[i]
                for r in results
                if r.standard_errors  # type: ignore[index]
            )
            self.assertLess(abs(reported / spread - 1.0), 0.35)
            block = statistics.fmean(r.block_means[i] for r in results)  # type: ignore[index]
            self.assertLess(abs(block - ISHIGAMI_FIRST_ORDER[i]), 0.03)
        self.assertEqual(results[0].blocks, 10)
        self.assertEqual(results[0].block_size, 200)

    def test_bootstrap_resampling_is_biased(self) -> None:
        """Resampled rows inflate the index of an input without effect."""
        rng = random.Random(8)
        rows, y = _ishigami_sample(2000, rng)
        clean = rank_sobol_indices(rows, y).first_order[2]
        draw = [rng.randrange(2000) for _ in range(2000)]
        resampled = rank_sobol_indices(
            [rows[k] for k in draw], [y[k] for k in draw], seed=1
        )
        self.assertLess(abs(clean), 0.08)
        self.assertGreater(resampled.first_order[2], 0.25)
        self.assertEqual(resampled.tied_inputs, ("x0", "x1", "x2"))

    def test_model_wrapper_is_reproducible_and_counts_runs(self) -> None:
        """One Monte Carlo sample serves all inputs."""
        first = model_rank_sobol_indices(ishigami_model(), "y", 1500, seed=4, blocks=5)
        second = model_rank_sobol_indices(ishigami_model(), "y", 1500, seed=4, blocks=5)
        self.assertEqual(first, second)
        self.assertEqual(first.evaluations, 1500)
        self.assertEqual(first.input_names, ("x1", "x2", "x3"))
        self.assertEqual(first.ranking()[0], "x2")
        self.assertEqual(set(first.as_mapping()), {"x1", "x2", "x3"})
        for value, exact in zip(first.first_order, ISHIGAMI_FIRST_ORDER, strict=True):
            self.assertLess(abs(value - exact), 0.12)

    def test_tied_inputs_are_reported_and_seeded(self) -> None:
        """A discrete input is ordered at random within ties, reproducibly."""
        rng = random.Random(2)
        rows = [[float(rng.randrange(4)), rng.random()] for _ in range(400)]
        y = [row[0] + row[1] for row in rows]
        first = rank_sobol_indices(rows, y, seed=9)
        self.assertEqual(first.tied_inputs, ("x0",))
        self.assertEqual(first, rank_sobol_indices(rows, y, seed=9))
        # Var(E[Y | X0]) / Var(Y) = (15/12) / (15/12 + 1/12)
        self.assertLess(abs(first.first_order[0] - 15.0 / 16.0), 0.05)

    def test_validation(self) -> None:
        """Invalid samples and settings are rejected."""
        rows = [[float(k)] for k in range(20)]
        y = [float(k % 3) for k in range(20)]
        with self.assertRaisesRegex(ValueError, "eight"):
            rank_sobol_indices(rows[:5], y[:5])
        with self.assertRaisesRegex(ValueError, "constant"):
            rank_sobol_indices(rows, [1.0] * 20)
        with self.assertRaisesRegex(ValueError, "blocks"):
            rank_sobol_indices(rows, y, blocks=1)
        with self.assertRaisesRegex(ValueError, "blocks"):
            rank_sobol_indices(rows, y, blocks=3)
        with self.assertRaisesRegex(ValueError, "same positive length"):
            rank_sobol_indices(rows, y[:-1])
        with self.assertRaisesRegex(ValueError, "names"):
            rank_sobol_indices(rows, y, names=("a", "b"))
        with self.assertRaises(ValueError):
            rank_sobol_indices([[math.nan]] * 20, y)


class ChatterjeeTests(unittest.TestCase):
    """Chatterjee's rank correlation."""

    def test_monotone_relation_has_the_exact_finite_value(self) -> None:
        """A strictly monotone relation gives ``1 - 3 / (n + 1)``."""
        for n in (5, 10, 101):
            x = [float(k) for k in range(n)]
            exact = 1.0 - 3.0 / (n + 1)
            self.assertEqual(chatterjee_xi(x, x), exact)
            self.assertEqual(chatterjee_xi(x, [-(v**3) for v in x]), exact)
            self.assertEqual(chatterjee_xi(x, [math.exp(0.1 * v) for v in x]), exact)

    def test_tie_free_formula_and_hand_computed_example(self) -> None:
        """The general formula reduces to ``1 - 3 sum |dr| / (n^2 - 1)``."""
        # Sorted by x the y-ranks are 2, 4, 1, 3, 5: jumps 2 + 3 + 2 + 2 = 9.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [0.2, 0.9, 0.1, 0.5, 1.4]
        self.assertEqual(chatterjee_xi(x, y), 1.0 - 3.0 * 9.0 / 24.0)
        rng = random.Random(6)
        n = 57
        xs = [rng.random() for _ in range(n)]
        ys = [rng.random() for _ in range(n)]
        order = sorted(range(n), key=xs.__getitem__)
        ranks = [sorted(ys).index(ys[k]) + 1 for k in order]
        jumps = sum(abs(ranks[j + 1] - ranks[j]) for j in range(n - 1))
        self.assertEqual(chatterjee_xi(xs, ys), 1.0 - 3 * jumps / (n * n - 1))

    def test_tied_output_step_function(self) -> None:
        """A two-level function of x gives ``1 - 2 / n`` by hand."""
        # r = n/2 then n, l = n then n/2: sum |dr| = n/2 and
        # sum l (n - l) = n^3 / 8, so xi = 1 - n (n/2) / (n^3 / 4).
        for n in (20, 64):
            x = [float(k) for k in range(n)]
            y = [0.0] * (n // 2) + [1.0] * (n // 2)
            self.assertAlmostEqual(chatterjee_xi(x, y), 1.0 - 2.0 / n, places=14)

    def test_independence_null_variance_is_two_fifths(self) -> None:
        """``n Var(xi_n)`` is near 2/5 and the asymptotic test holds its level."""
        rng = random.Random(10)
        n, replicates = 200, 1500
        values = []
        rejections = 0
        x = [[float(k)] for k in range(n)]
        for _ in range(replicates):
            y = [rng.random() for _ in range(n)]
            result = chatterjee_indices(x, y)
            values.append(result.xi[0])
            rejections += result.p_values[0] < 0.05
        # Monte Carlo se of the variance estimate: 0.4 sqrt(2 / 1500) = 0.015.
        self.assertLess(abs(n * statistics.pvariance(values) - 0.4), 0.05)
        self.assertLess(abs(statistics.fmean(values)), 4.0 * math.sqrt(0.4 / n / 1500))
        self.assertLess(abs(rejections / replicates - 0.05), 0.02)

    def test_detects_nonmonotone_dependence(self) -> None:
        """A symmetric dependence with zero correlation is detected."""
        rng = random.Random(12)
        rows = [[rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)] for _ in range(600)]
        y = [row[0] ** 2 + 0.05 * rng.gauss(0.0, 1.0) for row in rows]
        result = chatterjee_indices(rows, y, names=("a", "b"))
        self.assertGreater(result.xi[0], 0.5)
        self.assertLess(result.p_values[0], 1e-6)
        self.assertGreater(result.p_values[1], 0.01)
        self.assertEqual(result.dependent(), ("a",))
        self.assertEqual(result.test, "asymptotic")
        self.assertFalse(result.output_ties)
        self.assertEqual(result.as_mapping()["a"], result.xi[0])
        # the p-value is 1 - Phi(xi sqrt(n / 0.4))
        z = result.xi[1] * math.sqrt(600 / 0.4)
        self.assertAlmostEqual(
            result.p_values[1], 0.5 * math.erfc(z / math.sqrt(2.0)), places=14
        )

    def test_permutation_test_with_tied_output(self) -> None:
        """Ties in the output need the permutation test, which is seeded."""
        rng = random.Random(13)
        rows = [[rng.random(), rng.random()] for _ in range(300)]
        y = [float(row[0] > 0.4) for row in rows]
        with self.assertRaisesRegex(ValueError, "permutation"):
            chatterjee_indices(rows, y)
        result = chatterjee_indices(
            rows, y, test="permutation", permutations=199, seed=3
        )
        self.assertTrue(result.output_ties)
        self.assertEqual(result.permutations, 199)
        self.assertEqual(result.p_values[0], 1.0 / 200.0)
        self.assertGreater(result.p_values[1], 0.02)
        self.assertEqual(
            result,
            chatterjee_indices(rows, y, test="permutation", permutations=199, seed=3),
        )

    def test_validation(self) -> None:
        """Invalid samples and settings are rejected."""
        rows = [[float(k)] for k in range(12)]
        y = [float((7 * k) % 12) for k in range(12)]
        with self.assertRaisesRegex(ValueError, "three"):
            chatterjee_xi([1.0, 2.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "constant"):
            chatterjee_xi([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "constant"):
            chatterjee_indices(rows, [2.0] * 12)
        with self.assertRaisesRegex(ValueError, "test must"):
            chatterjee_indices(rows, y, test="exact")
        with self.assertRaisesRegex(ValueError, "permutations"):
            chatterjee_indices(rows, y, test="permutation", permutations=0)
        with self.assertRaisesRegex(ValueError, "eight"):
            chatterjee_indices(rows[:6], y[:6])
        with self.assertRaisesRegex(ValueError, "level"):
            chatterjee_indices(rows, y).dependent(level=1.5)


class ActiveSubspaceTests(unittest.TestCase):
    """Active subspaces from gradients and from a model."""

    def test_ridge_function_gives_rank_one_matrix(self) -> None:
        """``f = h(a'x)`` has ``W_1 = a / |a|`` exactly."""
        a = (1.0, -2.0, 0.5, 0.0)
        norm = math.sqrt(sum(v * v for v in a))
        rng = random.Random(2)
        points = [[rng.gauss(0.0, 1.0) for _ in a] for _ in range(40)]
        projections = [sum(p * q for p, q in zip(a, x, strict=True)) for x in points]
        gradients = [[math.cos(t) * p for p in a] for t in projections]
        values = [math.sin(t) for t in projections]
        result = active_subspace_from_gradients(
            gradients, points=points, values=values, bootstrap=30, seed=1
        )
        self.assertEqual(result.dimension, 1)
        self.assertEqual(result.dimension_rule, "largest-gap")
        sign = -1.0  # the largest component of a / |a| is negative
        for w, v in zip(result.eigenvectors[0], a, strict=True):
            self.assertAlmostEqual(w, sign * v / norm, places=12)
        expected = norm**2 * statistics.fmean(math.cos(t) ** 2 for t in projections)
        self.assertAlmostEqual(result.eigenvalues[0], expected, places=10)
        for value in result.eigenvalues[1:]:
            self.assertLess(value, 1e-12)
        self.assertAlmostEqual(result.explained_fraction(), 1.0, places=12)
        for score, v in zip(result.activity_scores, a, strict=True):
            self.assertAlmostEqual(score, expected * v * v / norm**2, places=10)
        assert result.subspace_errors is not None
        self.assertLess(result.subspace_errors[0], 1e-6)
        # the sufficient summary lies on the curve f = sin(sign |a| y)
        for y, f in result.sufficient_summary():
            self.assertAlmostEqual(f, math.sin(sign * norm * y), places=10)
        self.assertAlmostEqual(
            result.active_variables(points[0])[0], sign * projections[0] / norm
        )
        self.assertEqual(result.gradient_method, "given")

    def test_quadratic_form_gives_the_squared_matrix(self) -> None:
        """``f = x'Ax/2`` with ``x ~ N(0, I)`` has ``C = A^2``."""
        a = ((2.0, 0.5, 0.0), (0.5, 1.0, 0.3), (0.0, 0.3, 0.2))
        rng = random.Random(4)
        count = 3000
        gradients = [
            [sum(a[i][j] * x[j] for j in range(3)) for i in range(3)]
            for x in ([rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(count))
        ]
        result = active_subspace_from_gradients(gradients, bootstrap=60, seed=2)
        squared = matmul(a, a)
        for i in range(3):
            for j in range(3):
                products = [g[i] * g[j] for g in gradients]
                se = statistics.stdev(products) / math.sqrt(count)
                self.assertLess(abs(result.matrix[i][j] - squared[i][j]), 4.0 * se)
                self.assertEqual(result.matrix[i][j], result.matrix[j][i])
        exact = sorted(symmetric_eigen(squared)[0], reverse=True)
        assert result.eigenvalue_intervals is not None
        for value, target, (low, high) in zip(
            result.eigenvalues, exact, result.eigenvalue_intervals, strict=True
        ):
            self.assertLess(low, value)
            self.assertLess(value, high)
            # exact value within the interval widened by its own half-width
            self.assertLess(low - 0.5 * (high - low), target)
            self.assertGreater(high + 0.5 * (high - low), target)
            self.assertLess(abs(value / target - 1.0), 0.1)
        self.assertEqual(
            result.gaps,
            tuple(result.eigenvalues[k] - result.eigenvalues[k + 1] for k in range(2)),
        )
        assert result.subspace_errors is not None
        self.assertLess(max(result.subspace_errors), 0.05)
        self.assertEqual(
            result, active_subspace_from_gradients(gradients, bootstrap=60, seed=2)
        )

    def test_full_dimension_scores_are_mean_squared_derivatives(self) -> None:
        """With ``k = d`` the activity scores equal ``mean (df/dx_i)^2``."""
        rng = random.Random(7)
        gradients = [[rng.gauss(0.0, s) for s in (1.0, 3.0, 0.2)] for _ in range(50)]
        result = active_subspace_from_gradients(gradients, dimension=3)
        self.assertEqual(result.dimension_rule, "user")
        for i, score in enumerate(result.activity_scores):
            self.assertAlmostEqual(
                score, statistics.fmean(g[i] ** 2 for g in gradients), places=10
            )
        self.assertAlmostEqual(
            sum(result.eigenvalues),
            statistics.fmean(sum(v * v for v in g) for g in gradients),
            places=10,
        )
        self.assertEqual(result.as_mapping()["x1"], result.activity_scores[1])
        # orthonormal eigenvectors
        for i in range(3):
            for j in range(3):
                dot = sum(
                    p * q
                    for p, q in zip(
                        result.eigenvectors[i], result.eigenvectors[j], strict=True
                    )
                )
                self.assertAlmostEqual(dot, float(i == j), places=12)

    def test_subspace_distance_is_the_sine_of_the_largest_angle(self) -> None:
        """Known principal angles are recovered."""
        e1, e2, e3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        angle = 0.3
        tilted = (math.cos(angle), math.sin(angle), 0.0)
        self.assertEqual(_subspace_distance((e1,), (e1,)), 0.0)
        self.assertEqual(_subspace_distance((e1,), (e2,)), 1.0)
        self.assertAlmostEqual(_subspace_distance((e1,), (tilted,)), math.sin(angle))
        # span(e1, e3) against span(tilted, e3): one angle 0, one angle 0.3
        self.assertAlmostEqual(
            _subspace_distance((e1, e3), (tilted, e3)), math.sin(angle), places=12
        )
        # the same plane in a rotated orthonormal basis
        normal = (-math.sin(angle), math.cos(angle), 0.0)
        self.assertAlmostEqual(
            _subspace_distance((e1, e2), (tilted, normal)), 0.0, places=7
        )

    def test_model_gradients_in_standard_normal_space(self) -> None:
        """Autodiff and finite differences agree with the closed-form matrix."""

        def response(p):  # type: ignore[no-untyped-def]
            """Return ``3 a + 2 b`` (accepts dual numbers)."""
            return {"y": 3.0 * p["a"] + 2.0 * p["b"] + 0.0 * admath.exp(p["a"])}

        def build() -> ProbabilisticModel:
            """Build the two-input test model."""
            inputs = JointInput.independent(
                RandomInput("a", Normal(1.0, 2.0)), RandomInput("b", Uniform(0.0, 1.0))
            )
            return ProbabilisticModel(
                inputs, response, (QuantityOfInterest("y", "response"),), name="lin"
            )

        size = 2000
        auto = active_subspace(build(), "y", size, seed=5, bootstrap=20)
        self.assertEqual(auto.gradient_method, "autodiff")
        self.assertEqual(auto.space, "standard-normal")
        self.assertEqual(auto.evaluations, size)
        self.assertIsNone(auto.gradient_fallback_reason)
        # g(u) = 3 (1 + 2 u1) + 2 Phi(u2): grad = (6, 2 phi(u2)), and
        # E[phi(U)] = 1 / (2 sqrt(pi)), E[phi(U)^2] = 1 / (2 pi sqrt(3)).
        mean_phi = 1.0 / (2.0 * math.sqrt(math.pi))
        mean_phi2 = 1.0 / (2.0 * math.pi * math.sqrt(3.0))
        self.assertAlmostEqual(auto.matrix[0][0], 36.0, places=9)
        self.assertLess(abs(auto.matrix[0][1] - 12.0 * mean_phi), 0.08)
        self.assertLess(abs(auto.matrix[1][1] - 4.0 * mean_phi2), 0.01)
        self.assertEqual(auto.dimension, 1)
        self.assertGreater(auto.eigenvectors[0][0], 0.99)
        finite = active_subspace(
            build(), "y", 200, seed=5, gradient="finite-difference"
        )
        self.assertEqual(finite.evaluations, 200 * 5)
        self.assertEqual(finite.gradient_method, "finite-difference")
        short = active_subspace(build(), "y", 200, seed=5)
        self.assertEqual(short.points, finite.points)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    short.matrix[i][j], finite.matrix[i][j], delta=1e-6
                )
        self.assertEqual(len(short.sufficient_summary()), 200)
        self.assertEqual(short.values, finite.values)

    def test_fallbacks_and_refusals(self) -> None:
        """Models without dual support fall back; discrete inputs are refused."""

        def response(p):  # type: ignore[no-untyped-def]
            """Return a response that needs ``math.exp``."""
            return {"y": math.exp(0.3 * p["a"]) + p["b"]}

        qoi = (QuantityOfInterest("y", "response"),)
        laws = (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0)))
        model = ProbabilisticModel(JointInput.independent(*laws), response, qoi)
        result = active_subspace(model, "y", 30, seed=1)
        self.assertEqual(result.gradient_method, "finite-difference")
        self.assertIn("Dual", str(result.gradient_fallback_reason))
        with self.assertRaises(AutodiffUnsupportedError):
            active_subspace(model, "y", 30, seed=1, gradient="autodiff")
        dependent = ProbabilisticModel(
            JointInput.from_spearman(laws, ((1.0, 0.5), (0.5, 1.0))), response, qoi
        )
        correlated = active_subspace(dependent, "y", 30, seed=1)
        self.assertEqual(correlated.gradient_method, "finite-difference")
        self.assertIn("dependent", str(correlated.gradient_fallback_reason))
        with self.assertRaises(AutodiffUnsupportedError):
            active_subspace(dependent, "y", 30, seed=1, gradient="autodiff")
        discrete = ProbabilisticModel(
            JointInput.independent(laws[0], RandomInput("b", Poisson(3.0))),
            response,
            qoi,
        )
        with self.assertRaisesRegex(ValueError, "continuous"):
            active_subspace(discrete, "y", 30, seed=1)

    def test_validation(self) -> None:
        """Invalid gradients and settings are rejected."""
        gradients = [[1.0, 2.0], [0.5, -1.0], [0.0, 1.0]]
        with self.assertRaisesRegex(ValueError, "two gradients"):
            active_subspace_from_gradients(gradients[:1])
        with self.assertRaisesRegex(ValueError, "rectangular"):
            active_subspace_from_gradients([[1.0, 2.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "dimension"):
            active_subspace_from_gradients(gradients, dimension=3)
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            active_subspace_from_gradients(gradients, bootstrap=-1)
        with self.assertRaisesRegex(ValueError, "confidence"):
            active_subspace_from_gradients(gradients, confidence=1.0)
        with self.assertRaisesRegex(ValueError, "together"):
            active_subspace_from_gradients(gradients, points=gradients)
        with self.assertRaisesRegex(ValueError, "match"):
            active_subspace_from_gradients(
                gradients, points=gradients, values=[1.0, 2.0]
            )
        with self.assertRaisesRegex(ValueError, "names"):
            active_subspace_from_gradients(gradients, names=("a",))
        result = active_subspace_from_gradients(gradients)
        with self.assertRaisesRegex(ValueError, "no sample points"):
            result.sufficient_summary()
        with self.assertRaisesRegex(ValueError, "length"):
            result.active_variables([1.0])
        self.assertIsNone(result.eigenvalue_intervals)
        single = active_subspace_from_gradients([[1.0], [2.0]])
        self.assertEqual(single.dimension, 1)
        self.assertEqual(single.gaps, ())
        with self.assertRaisesRegex(ValueError, "size"):
            active_subspace(ishigami_model(), "y", 1, seed=1)
        with self.assertRaisesRegex(ValueError, "gradient must"):
            active_subspace(ishigami_model(), "y", 10, seed=1, gradient="exact")
        with self.assertRaisesRegex(ValueError, "relative_step"):
            active_subspace(ishigami_model(), "y", 10, seed=1, relative_step=0.0)


class MomentIndependentTests(unittest.TestCase):
    """Borgonovo's delta and PAWN from given data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Draw one linear Gaussian sample with an input without effect."""
        rng = random.Random(21)
        cls.rows = [[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(4000)]
        cls.y = [2.0 * row[0] + row[1] for row in cls.rows]

    def test_delta_of_a_linear_gaussian_model(self) -> None:
        """Both density estimators reproduce the quadrature value within bias."""
        exact = (_exact_gaussian_delta(2.0, 1.0), _exact_gaussian_delta(1.0, 2.0))
        # independent check of the quadrature: delta is scale invariant
        self.assertAlmostEqual(exact[0], _exact_gaussian_delta(4.0, 2.0), places=6)
        self.assertAlmostEqual(exact[0], 0.5055, places=3)
        histogram = borgonovo_delta(self.rows, self.y, permutations=30, seed=1)
        kernel = borgonovo_delta(self.rows[:1500], self.y[:1500], density="kde")
        for result, tolerance in ((histogram, 0.04), (kernel, 0.05)):
            for value, target in zip(result.delta[:2], exact, strict=True):
                self.assertLess(abs(value - target), tolerance)
        # documented direction of the bias: the class average and the
        # smoothing lower the estimate of the strong input
        self.assertLess(histogram.delta[0], exact[0])
        self.assertLess(kernel.delta[0], exact[0])
        assert histogram.corrected is not None and histogram.p_values is not None
        self.assertLess(histogram.corrected[1], exact[1])
        self.assertGreater(histogram.delta[1], exact[1])
        self.assertEqual(histogram.p_values[0], 1.0 / 31.0)
        self.assertEqual(histogram.p_values[1], 1.0 / 31.0)
        self.assertEqual(histogram.density, "histogram")
        self.assertEqual(histogram.partitions, 16)
        self.assertEqual(histogram.bins, 8)
        self.assertEqual(kernel.bins, 0)
        self.assertIsNone(kernel.corrected)
        self.assertTrue(math.isnan(kernel.null_level))

    def test_input_without_effect_sits_at_the_permutation_null(self) -> None:
        """The raw indices of an irrelevant input equal the null level."""
        delta = borgonovo_delta(self.rows, self.y, permutations=40, seed=2)
        pawn = pawn_indices(self.rows, self.y, permutations=200, seed=2)
        kde = borgonovo_delta(
            self.rows[:800], self.y[:800], density="kde", permutations=15, seed=2
        )
        for result, raw in ((delta, delta.delta), (pawn, pawn.index), (kde, kde.delta)):
            assert result.corrected is not None and result.p_values is not None
            self.assertGreater(raw[2], 0.5 * result.null_level)
            self.assertLess(abs(result.corrected[2]), 0.35 * result.null_level)
            self.assertGreater(result.p_values[2], 0.05)
            self.assertGreater(result.null_level, 0.02)
            self.assertGreater(result.corrected[0], result.corrected[1])
            self.assertGreater(result.corrected[1], 3.0 * abs(result.corrected[2]))
            self.assertAlmostEqual(
                result.corrected[0], raw[0] - result.null_level, places=14
            )

    def test_invariance_under_monotone_transformations(self) -> None:
        """Histogram delta and PAWN depend on the output through its order."""
        rows, y = self.rows[:1000], self.y[:1000]
        increasing = [math.exp(0.5 * v) for v in y]
        decreasing = [-(v**3) for v in y]
        self.assertEqual(
            borgonovo_delta(rows, y).delta, borgonovo_delta(rows, increasing).delta
        )
        for statistic in ("median", "mean", "max"):
            reference = pawn_indices(rows, y, statistic=statistic)
            self.assertEqual(
                reference.ks_distances,
                pawn_indices(rows, increasing, statistic=statistic).ks_distances,
            )
            self.assertEqual(
                reference.index,
                pawn_indices(rows, decreasing, statistic=statistic).index,
            )

    def test_pawn_distances_against_brute_force(self) -> None:
        """Class KS distances equal a direct evaluation of both ECDFs."""
        rng = random.Random(23)
        n, classes = 60, 4
        rows = [[rng.random()] for _ in range(n)]
        # rounded outputs create ties
        y = [round(row[0] + rng.gauss(0.0, 0.4), 1) for row in rows]
        result = pawn_indices(rows, y, partitions=classes, statistic="max")
        order = sorted(range(n), key=lambda k: rows[k][0])
        expected = []
        for m in range(classes):
            members = [y[k] for k in order[m * n // classes : (m + 1) * n // classes]]
            best = 0.0
            for t in sorted(set(y)):
                whole = sum(v <= t for v in y) / n
                part = sum(v <= t for v in members) / len(members)
                best = max(best, abs(whole - part))
            expected.append(best)
        for value, target in zip(result.ks_distances[0], expected, strict=True):
            self.assertAlmostEqual(value, target, places=12)
        self.assertAlmostEqual(result.index[0], max(expected), places=12)
        self.assertAlmostEqual(
            pawn_indices(rows, y, partitions=classes, statistic="mean").index[0],
            statistics.fmean(expected),
            places=12,
        )
        self.assertAlmostEqual(
            pawn_indices(rows, y, partitions=classes).index[0],
            statistics.median(expected),
            places=12,
        )
        self.assertEqual(result.as_mapping(), {"x0": result.index[0]})

    def test_histogram_delta_against_brute_force(self) -> None:
        """The histogram estimator equals its definition on a small sample."""
        rng = random.Random(25)
        n, classes, bins = 80, 4, 4
        rows = [[rng.random()] for _ in range(n)]
        y = [row[0] ** 2 + 0.2 * rng.random() for row in rows]
        result = borgonovo_delta(rows, y, partitions=classes, bins=bins)
        ordered = sorted(y)
        label = [ordered.index(v) * bins // n for v in y]
        pooled = [label.count(b) / n for b in range(bins)]
        order = sorted(range(n), key=lambda k: rows[k][0])
        expected = 0.0
        for m in range(classes):
            members = [label[k] for k in order[m * 20 : (m + 1) * 20]]
            expected += (
                0.25
                * 0.5
                * sum(abs(pooled[b] - members.count(b) / 20) for b in range(bins))
            )
        self.assertAlmostEqual(result.delta[0], expected, places=12)
        self.assertEqual(result.as_mapping(), {"x0": result.delta[0]})
        self.assertGreater(result.delta[0], 0.4)

    def test_validation(self) -> None:
        """Invalid samples and settings are rejected."""
        rows, y = self.rows[:200], self.y[:200]
        with self.assertRaisesRegex(ValueError, "rows per class"):
            borgonovo_delta(rows, y, partitions=40)
        with self.assertRaisesRegex(ValueError, "rows per class"):
            pawn_indices(rows, y, partitions=1)
        with self.assertRaisesRegex(ValueError, "integer"):
            pawn_indices(rows, y, partitions=2.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "density"):
            borgonovo_delta(rows, y, density="spline")
        with self.assertRaisesRegex(ValueError, "bins"):
            borgonovo_delta(rows, y, bins=1)
        with self.assertRaisesRegex(ValueError, "grid"):
            borgonovo_delta(rows, y, density="kde", grid=4)
        with self.assertRaisesRegex(ValueError, "permutations"):
            borgonovo_delta(rows, y, permutations=-1)
        with self.assertRaisesRegex(ValueError, "statistic"):
            pawn_indices(rows, y, statistic="min")
        with self.assertRaisesRegex(ValueError, "constant"):
            borgonovo_delta(rows, [1.0] * 200)
        with self.assertRaisesRegex(ValueError, "constant"):
            pawn_indices(rows, [1.0] * 200)


class KernelSobolTests(unittest.TestCase):
    """Kernel (MMD) pick-freeze indices."""

    @classmethod
    def setUpClass(cls) -> None:
        """Draw two base matrices for the Ishigami function."""
        rng = random.Random(31)
        cls.a = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(300)]
        cls.b = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(300)]

    @staticmethod
    def model(row: tuple[float, ...]) -> float:
        """Evaluate the Ishigami function on a row."""
        return ishigami(*row)

    def test_linear_kernel_reproduces_the_janon_sobol_estimator(self) -> None:
        """``k(y, y') = y y'`` gives the library's pick-freeze estimate."""
        library = estimate_sobol_indices(self.model, self.a, self.b, estimator="janon")
        named = kernel_sobol_indices(self.model, self.a, self.b, kernel="linear")
        custom = kernel_sobol_indices(
            self.model, self.a, self.b, kernel=lambda first, second: first * second
        )
        for i in range(3):
            self.assertAlmostEqual(
                named.indices[i], library.first_order[i], delta=1e-12
            )
            self.assertAlmostEqual(
                custom.indices[i], library.first_order[i], delta=1e-12
            )
        self.assertEqual(named.kernel, "linear")
        self.assertEqual(custom.kernel, "custom")
        self.assertTrue(math.isnan(named.bandwidth))
        self.assertEqual(named.evaluations, 300 * 4)
        # an independent uncentred formula on the same outputs
        yb = [self.model(tuple(row)) for row in self.b]
        for i in range(3):
            yi = [
                self.model(tuple(ra[:i] + [rb[i]] + ra[i + 1 :]))
                for ra, rb in zip(self.a, self.b, strict=True)
            ]
            centre = (statistics.fmean(yb) + statistics.fmean(yi)) / 2.0
            numerator = (
                statistics.fmean(p * q for p, q in zip(yb, yi, strict=True)) - centre**2
            )
            denominator = (
                statistics.fmean(
                    (p * p + q * q) / 2.0 for p, q in zip(yb, yi, strict=True)
                )
                - centre**2
            )
            self.assertAlmostEqual(
                named.indices[i], numerator / denominator, delta=1e-11
            )

    def test_gaussian_kernel_against_direct_double_sum(self) -> None:
        """The estimator equals its definition with all ordered pairs."""
        a, b = self.a[:25], self.b[:25]
        result = kernel_sobol_indices(self.model, a, b, bandwidth=2.0)
        self.assertEqual(result.bandwidth, 2.0)

        def kernel(first: float, second: float) -> float:
            """Evaluate the Gaussian kernel with bandwidth 2."""
            return math.exp(-((first - second) ** 2) / 8.0)

        yb = [self.model(tuple(row)) for row in b]
        for i in range(3):
            yi = [
                self.model(tuple(ra[:i] + [rb[i]] + ra[i + 1 :]))
                for ra, rb in zip(a, b, strict=True)
            ]
            pooled = yb + yi
            t1 = statistics.fmean(kernel(p, q) for p, q in zip(yb, yi, strict=True))
            t0 = statistics.fmean(kernel(p, q) for p in pooled for q in pooled)
            self.assertAlmostEqual(result.indices[i], (t1 - t0) / (1.0 - t0), places=12)

    def test_function_of_one_input_has_index_one(self) -> None:
        """``Y = g(X_1)`` gives exactly one for ``X_1`` and about zero else."""
        result = kernel_sobol_indices(lambda row: math.sin(row[0]), self.a, self.b)
        self.assertAlmostEqual(result.indices[0], 1.0, places=12)
        self.assertLess(abs(result.indices[1]), 0.15)
        self.assertEqual(result.indices[1], result.indices[2])
        self.assertEqual(result.kernel, "gaussian")
        self.assertGreater(result.bandwidth, 0.0)
        self.assertEqual(result.ranking()[0], "x0")
        self.assertEqual(set(result.as_mapping()), {"x0", "x1", "x2"})

    def test_gaussian_kernel_sees_a_pure_scale_effect(self) -> None:
        """``Y = X_1 X_2`` has zero first-order Sobol but positive MMD index."""
        rng = random.Random(33)
        a = [[rng.gauss(0.0, 1.0) for _ in range(2)] for _ in range(500)]
        b = [[rng.gauss(0.0, 1.0) for _ in range(2)] for _ in range(500)]

        def product(row: tuple[float, ...]) -> float:
            """Return the product of both inputs."""
            return row[0] * row[1]

        linear = kernel_sobol_indices(product, a, b, kernel="linear")
        gaussian = kernel_sobol_indices(product, a, b)
        std = kernel_sobol_indices(product, a, b, bandwidth="std")
        for i in range(2):
            self.assertLess(abs(linear.indices[i]), 0.2)
            self.assertGreater(gaussian.indices[i], 0.08)
            self.assertGreater(std.indices[i], 0.03)
        self.assertNotEqual(std.bandwidth, gaussian.bandwidth)

    def test_model_wrapper(self) -> None:
        """The wrapper draws the design, counts runs, and is reproducible."""
        first = model_kernel_sobol_indices(ishigami_model(), "y", 200, seed=3)
        second = model_kernel_sobol_indices(ishigami_model(), "y", 200, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first.evaluations, 200 * 4)
        self.assertEqual(first.input_names, ("x1", "x2", "x3"))
        self.assertGreater(first.indices[1], first.indices[2])
        linear = model_kernel_sobol_indices(
            ishigami_model(), "y", 512, seed=3, design="sobol", kernel="linear"
        )
        for value, exact in zip(linear.indices, ISHIGAMI_FIRST_ORDER, strict=True):
            self.assertLess(abs(value - exact), 0.08)

    def test_validation(self) -> None:
        """Invalid designs, kernels, and bandwidths are rejected."""
        a, b = self.a[:10], self.b[:10]
        with self.assertRaises(TypeError):
            kernel_sobol_indices(3.0, a, b)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "equal shape"):
            kernel_sobol_indices(self.model, a, b[:9])
        with self.assertRaisesRegex(ValueError, "four rows"):
            kernel_sobol_indices(self.model, a[:3], b[:3])
        with self.assertRaisesRegex(ValueError, "kernel must"):
            kernel_sobol_indices(self.model, a, b, kernel="matern")
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            kernel_sobol_indices(self.model, a, b, bandwidth="scott")
        with self.assertRaisesRegex(ValueError, "positive"):
            kernel_sobol_indices(self.model, a, b, bandwidth=-1.0)
        with self.assertRaisesRegex(ValueError, "names"):
            kernel_sobol_indices(self.model, a, b, names=("a",))
        with self.assertRaisesRegex(ValueError, "separate"):
            kernel_sobol_indices(lambda row: 1.0, a, b, kernel="linear")
        with self.assertRaisesRegex(ValueError, "size"):
            model_kernel_sobol_indices(ishigami_model(), "y", 2, seed=1)
        with self.assertRaisesRegex(ValueError, "design"):
            model_kernel_sobol_indices(ishigami_model(), "y", 8, seed=1, design="grid")
        laws = (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0)))
        dependent = ProbabilisticModel(
            JointInput.from_spearman(laws, ((1.0, 0.5), (0.5, 1.0))),
            lambda p: {"y": p["a"] + p["b"]},
            (QuantityOfInterest("y", "sum"),),
        )
        with self.assertRaisesRegex(ValueError, "independent"):
            model_kernel_sobol_indices(dependent, "y", 8, seed=1)


if __name__ == "__main__":
    unittest.main()
