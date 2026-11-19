"""Tests for Tikhonov, truncated SVD, and the parameter-choice rules (ADR-0110).

Exact answers are used where they exist: the backward heat problem has a
closed-form SVD, Tikhonov regularization with ``L = I`` equals a Gaussian
posterior mean computed independently from the normal equations, and the
L-curve curvature is compared with finite differences of the log norms.
Seeded noise is drawn with ``random.Random``; parameter choices are judged by
bands around the error-optimal ``lambda``, never by pinned digits.
"""

from __future__ import annotations

import math
import unittest

from probstat.foundations import RandomStream
from probstat.inverse import (
    DiscreteIllPosedProblem,
    backward_heat_problem,
    discrepancy_principle,
    generalized_cross_validation,
    l_curve,
    nonlinear_tikhonov,
    picard_coefficients,
    second_derivative_problem,
    tikhonov,
    truncated_svd,
)
from probstat.inverse.regularization import _curvature, _Curve
from probstat.linalg import (
    matmul,
    matvec,
    pseudo_inverse,
    singular_value_decomposition,
    solve_spd,
    transpose,
)


def noisy_data(
    problem: DiscreteIllPosedProblem, relative_level: float, seed: int
) -> tuple[tuple[float, ...], float]:
    """Return ``b_exact + e`` with i.i.d. Gaussian ``e`` and the norm ``||e||``.

    The standard deviation is ``relative_level ||b_exact|| / sqrt(m)``.

    Args:
        problem: Test problem.
        relative_level: Expected ``||e|| / ||b_exact||``.
        seed: Root seed of the noise stream.

    Returns:
        The noisy data and the realized noise norm.

    References:
        [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

    """
    exact = problem.exact_data
    scale = relative_level * norm(exact) / math.sqrt(len(exact))
    generator = RandomStream(seed).child("noise").generator()
    noise = [generator.gauss(0.0, scale) for _ in exact]
    return tuple(b + e for b, e in zip(exact, noise, strict=True)), norm(noise)


def norm(vector: tuple[float, ...] | list[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(value * value for value in vector))


def relative_error(estimate: tuple[float, ...], exact: tuple[float, ...]) -> float:
    """Return ``||estimate - exact|| / ||exact||``.

    References:
        [REF-HANSEN-2008, Section 2.5]. No equation is evaluated.

    """
    return norm([a - b for a, b in zip(estimate, exact, strict=True)]) / norm(exact)


def svd_tikhonov(svd, data, regularization):  # type: ignore[no-untyped-def]
    """Return ``sum_i f_i (u_i'b / sigma_i) v_i`` from an SVD (filter form).

    References:
        [REF-HANSEN-2008, Section 2.4]. Equation [EQ-INV-REG-002].

    """
    k = len(svd.singular_values)
    beta = [
        math.fsum(svd.u[i][j] * data[i] for i in range(len(data))) for j in range(k)
    ]
    weights = [
        sigma / (sigma * sigma + regularization * regularization) * b
        for sigma, b in zip(svd.singular_values, beta, strict=True)
    ]
    return tuple(
        math.fsum(svd.v[row][j] * weights[j] for j in range(k))
        for row in range(len(svd.v))
    )


def optimal_regularization(problem, data, svd):  # type: ignore[no-untyped-def]
    """Return the ``lambda`` minimizing the true error on a fine log grid.

    References:
        [REF-HANSEN-2008, Section 2.5]. No equation is evaluated.

    """
    best_lambda, best_error = math.nan, math.inf
    for index in range(301):
        lam = 10.0 ** (-6.0 + 6.0 * index / 300)
        error = relative_error(svd_tikhonov(svd, data, lam), problem.exact_solution)
        if error < best_error:
            best_lambda, best_error = lam, error
    return best_lambda


def first_difference(size: int) -> tuple[tuple[float, ...], ...]:
    """Return the ``(size - 1) x size`` first-difference matrix.

    References:
        [REF-HANSEN-2008, Section 2.2]. No equation is evaluated.

    """
    return tuple(
        tuple(-1.0 if j == i else 1.0 if j == i + 1 else 0.0 for j in range(size))
        for i in range(size - 1)
    )


class TestProblemTests(unittest.TestCase):
    """The test problems are what their docstrings claim."""

    def test_backward_heat_has_the_stated_svd(self) -> None:
        """Sine vectors diagonalize the second-difference matrix; SVD agrees."""
        n = 12
        problem = backward_heat_problem(n, time=0.01)
        h = 1.0 / (n + 1)
        for k in range(1, n + 1):
            vector = [
                math.sqrt(2 * h) * math.sin(j * k * math.pi * h)
                for j in range(1, n + 1)
            ]
            mu = 4.0 / h**2 * math.sin(k * math.pi * h / 2.0) ** 2
            for j in range(n):
                left = vector[j - 1] if j > 0 else 0.0
                right = vector[j + 1] if j < n - 1 else 0.0
                applied = (2.0 * vector[j] - left - right) / h**2
                self.assertAlmostEqual(applied, mu * vector[j], delta=1e-10 * mu)
        assert problem.singular_values is not None
        computed = singular_value_decomposition(problem.operator).singular_values
        for value, exact in zip(computed, problem.singular_values, strict=True):
            self.assertAlmostEqual(value, exact, delta=1e-14)
        self.assertEqual(problem.exact_solution[3], 6.75 * (4 * h) * (1 - 4 * h) ** 2)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(
                    problem.operator[i][j], problem.operator[j][i], delta=1e-15
                )

    def test_second_derivative_data_converge_to_the_continuous_data(self) -> None:
        """``A x`` approaches ``g(s) = (s^3 - s) / 6`` at the rate ``h^2``."""
        errors = []
        for n in (16, 32, 64):
            problem = second_derivative_problem(n)
            errors.append(
                max(
                    abs(b - (s**3 - s) / 6.0)
                    for b, s in zip(problem.exact_data, problem.grid, strict=True)
                )
            )
        for coarse, fine in zip(errors, errors[1:], strict=False):
            self.assertGreater(coarse / fine, 3.8)
            self.assertLess(coarse / fine, 4.2)
        problem = second_derivative_problem(4)
        self.assertEqual(problem.operator[0][1], 0.25 * (0.125 * (0.375 - 1.0)))
        self.assertEqual(problem.operator[1][0], problem.operator[0][1])
        self.assertIsNone(problem.singular_values)

    def test_invalid_problem_arguments(self) -> None:
        """Sizes and times are validated."""
        with self.assertRaises(ValueError):
            backward_heat_problem(1)
        with self.assertRaises(TypeError):
            second_derivative_problem(8.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            backward_heat_problem(8, time=0.0)


class TikhonovTests(unittest.TestCase):
    """Tikhonov solutions, filter factors, and the Bayesian cross-check."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one noisy heat problem and its SVD.

        References:
            [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

        """
        cls.problem = backward_heat_problem(16, time=0.005)
        cls.data, cls.noise_norm = noisy_data(cls.problem, 1e-2, 11)
        cls.svd = singular_value_decomposition(cls.problem.operator)

    def test_equals_gaussian_posterior_mean(self) -> None:
        """``L = I`` gives the posterior mean for prior ``N(0, s^2/lambda^2 I)``.

        With noise variance ``s^2`` the posterior precision is
        ``(A'A + lambda^2 I) / s^2`` and the mean
        ``(A'A + lambda^2 I)^-1 A'b``, solved here by Cholesky.
        """
        operator = self.problem.operator
        for lam in (3e-3, 3e-2, 0.3):
            precision = matmul(transpose(operator), operator)
            precision = tuple(
                tuple(v + (lam * lam if i == j else 0.0) for j, v in enumerate(row))
                for i, row in enumerate(precision)
            )
            mean = solve_spd(precision, matvec(transpose(operator), self.data))
            result = tikhonov(operator, self.data, lam, decomposition=self.svd)
            self.assertLess(relative_error(result.solution, mean), 1e-10)
            self.assertEqual(result.regularization, lam)

    def test_filter_factors_and_svd_form(self) -> None:
        """QR solution equals the filter-factor expansion; factors are exact."""
        lam = 0.02
        result = tikhonov(self.problem.operator, self.data, lam)
        assert result.filter_factors is not None
        for f, sigma in zip(
            result.filter_factors, self.svd.singular_values, strict=True
        ):
            self.assertAlmostEqual(f, sigma**2 / (sigma**2 + lam**2), delta=1e-13)
        factors = result.filter_factors
        self.assertTrue(all(a >= b for a, b in zip(factors, factors[1:], strict=False)))
        self.assertTrue(all(0.0 <= f <= 1.0 for f in factors))
        expansion = svd_tikhonov(self.svd, self.data, lam)
        self.assertLess(relative_error(result.solution, expansion), 1e-11)
        residual = [
            a - b
            for a, b in zip(
                matvec(self.problem.operator, result.solution), self.data, strict=False
            )
        ]
        self.assertAlmostEqual(result.residual_norm, norm(residual), delta=1e-14)
        self.assertAlmostEqual(result.solution_norm, norm(result.solution), delta=1e-14)

    def test_general_penalty_matches_normal_equations(self) -> None:
        """A first-difference penalty solves ``(A'A + lambda^2 L'L) x = A'b``."""
        operator = self.problem.operator
        penalty = first_difference(16)
        lam = 0.05
        result = tikhonov(operator, self.data, lam, penalty=penalty)
        self.assertIsNone(result.filter_factors)
        gram = matmul(transpose(operator), operator)
        roughness = matmul(transpose(penalty), penalty)
        system = tuple(
            tuple(a + lam * lam * b for a, b in zip(row, other, strict=True))
            for row, other in zip(gram, roughness, strict=True)
        )
        expected = solve_spd(system, matvec(transpose(operator), self.data))
        self.assertLess(relative_error(result.solution, expected), 1e-9)
        self.assertAlmostEqual(
            result.solution_norm, norm(matvec(penalty, result.solution)), delta=1e-12
        )

    def test_zero_regularization_is_least_squares(self) -> None:
        """``lambda = 0`` reproduces the pseudo-inverse solution of a tall matrix."""
        operator = ((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0))
        data = (1.0, 2.0, 2.0, 4.0)
        result = tikhonov(operator, data, 0.0)
        expected = matvec(pseudo_inverse(operator), data)
        for a, b in zip(result.solution, expected, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)
        assert result.filter_factors is not None
        self.assertEqual(result.filter_factors, (1.0, 1.0))

    def test_rank_deficient_stack_is_refused(self) -> None:
        """A shared null space of ``A`` and ``L`` raises instead of repairing."""
        operator = ((1.0, 1.0), (2.0, 2.0))
        penalty = ((1.0, -1.0),)
        with self.assertRaises(ValueError):
            tikhonov(operator, (1.0, 2.0), 0.5, penalty=((1.0, 1.0),))
        result = tikhonov(operator, (1.0, 2.0), 0.5, penalty=penalty)
        self.assertAlmostEqual(result.solution[0], result.solution[1], delta=1e-14)
        with self.assertRaises(ValueError):
            tikhonov(operator, (1.0, 2.0), 0.0)

    def test_tikhonov_approaches_truncated_svd_across_a_gap(self) -> None:
        """With ``sigma_k >> lambda`` and ``sigma_(k+1) << lambda^2`` Tikhonov is TSVD.

        A discarded component contributes ``sigma b / (sigma^2 + lambda^2)``,
        about ``sigma / lambda^2``, so ``lambda >> sigma_(k+1)`` is not enough.
        """
        values = (1.0, 0.5, 1e-12, 1e-13)
        operator = tuple(
            tuple(values[i] if i == j else 0.0 for j in range(4)) for i in range(4)
        )
        data = (1.0, 1.0, 1.0, 1.0)
        tsvd = truncated_svd(operator, data, 2)
        regularized = tikhonov(operator, data, 1e-3)
        self.assertLess(relative_error(regularized.solution, tsvd.solution), 1e-5)
        self.assertEqual(tsvd.filter_factors, (1.0, 1.0, 0.0, 0.0))
        full = truncated_svd(operator, data, 4)
        unregularized = tikhonov(operator, data, 0.0)
        self.assertLess(relative_error(unregularized.solution, full.solution), 1e-12)

    def test_large_lambda_limit(self) -> None:
        """``lambda^2 x_lambda -> A'b`` as ``lambda -> inf``."""
        lam = 1e4
        result = tikhonov(self.problem.operator, self.data, lam, decomposition=self.svd)
        target = matvec(transpose(self.problem.operator), self.data)
        scaled = tuple(lam * lam * value for value in result.solution)
        self.assertLess(relative_error(scaled, target), 1e-7)

    def test_invalid_tikhonov_arguments(self) -> None:
        """Shapes and parameters are validated."""
        operator = self.problem.operator
        with self.assertRaises(ValueError):
            tikhonov(operator, self.data[:-1], 0.1)
        with self.assertRaises(ValueError):
            tikhonov(operator, self.data, -0.1)
        with self.assertRaises(ValueError):
            tikhonov(operator, self.data, math.inf)
        with self.assertRaises(ValueError):
            tikhonov(operator, self.data, 0.1, penalty=((1.0, 2.0),))
        wrong = singular_value_decomposition(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            tikhonov(operator, self.data, 0.1, decomposition=wrong)


class TruncatedSVDAndPicardTests(unittest.TestCase):
    """Truncated SVD and the discrete Picard coefficients."""

    def test_truncated_svd_matches_truncated_pseudo_inverse(self) -> None:
        """``x_k = A_k^+ b`` with the residual from the discarded coefficients."""
        problem = backward_heat_problem(12, time=0.005)
        data, _ = noisy_data(problem, 1e-3, 5)
        svd = singular_value_decomposition(problem.operator)
        k = 6
        result = truncated_svd(problem.operator, data, k, decomposition=svd)
        rcond = 0.5 * (svd.singular_values[k - 1] + svd.singular_values[k])
        rcond /= svd.singular_values[0]
        expected = matvec(pseudo_inverse(problem.operator, rcond=rcond), data)
        self.assertLess(relative_error(result.solution, expected), 1e-12)
        residual = [
            a - b
            for a, b in zip(
                matvec(problem.operator, result.solution), data, strict=False
            )
        ]
        self.assertAlmostEqual(result.residual_norm, norm(residual), delta=1e-13)
        self.assertAlmostEqual(result.solution_norm, norm(result.solution), delta=1e-12)
        self.assertEqual(result.rank, k)

    def test_truncated_svd_validation(self) -> None:
        """Ranks outside ``1..k`` or reaching a zero singular value raise."""
        operator = ((1.0, 0.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            truncated_svd(operator, (1.0, 1.0), 0)
        with self.assertRaises(ValueError):
            truncated_svd(operator, (1.0, 1.0), 2)
        with self.assertRaises(ValueError):
            truncated_svd(operator, (1.0, 1.0), 3)
        with self.assertRaises(TypeError):
            truncated_svd(operator, (1.0, 1.0), 1.0)  # type: ignore[arg-type]
        result = truncated_svd(operator, (1.0, 1.0), 1)
        self.assertEqual(result.solution, (1.0, 0.0))
        self.assertAlmostEqual(result.residual_norm, 1.0, delta=1e-15)

    def test_picard_coefficients_of_exact_data(self) -> None:
        """For exact heat data ``|u_i'b| / sigma_i = |v_i'x|`` (sine coefficients)."""
        n = 16
        problem = backward_heat_problem(n, time=0.005)
        picard = picard_coefficients(problem.operator, problem.exact_data)
        h = 1.0 / (n + 1)
        assert problem.singular_values is not None
        for index in range(8):
            k = index + 1
            sine = [
                math.sqrt(2 * h) * math.sin(j * k * math.pi * h)
                for j in range(1, n + 1)
            ]
            coefficient = abs(
                math.fsum(
                    a * b for a, b in zip(sine, problem.exact_solution, strict=False)
                )
            )
            self.assertAlmostEqual(
                picard.singular_values[index],
                problem.singular_values[index],
                delta=1e-14,
            )
            self.assertAlmostEqual(
                picard.solution_coefficients[index], coefficient, delta=1e-9
            )
            self.assertAlmostEqual(
                picard.data_coefficients[index],
                problem.singular_values[index] * coefficient,
                delta=1e-14,
            )

    def test_picard_zero_singular_values(self) -> None:
        """A zero singular value gives ``inf`` or ``nan`` explicitly."""
        operator = ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        picard = picard_coefficients(operator, (4.0, 1.0, 0.0))
        self.assertEqual(picard.singular_values[0], 2.0)
        self.assertAlmostEqual(picard.solution_coefficients[0], 2.0, delta=1e-15)
        ratios = picard.solution_coefficients[1:]
        self.assertEqual(sum(1 for value in ratios if math.isinf(value)), 1)
        self.assertEqual(sum(1 for value in ratios if math.isnan(value)), 1)


class CurveTests(unittest.TestCase):
    """Closed-form curve quantities against finite differences and QR paths."""

    def assert_curvature_matches_finite_differences(
        self, curve: _Curve, lam: float
    ) -> None:
        """Compare the closed form with central differences in ``log lambda``.

        References:
            [REF-HANSEN-OLEARY-1993, Section 5.1]. Equation [EQ-INV-REG-004].

        """
        step = 1e-4

        def coordinates(t: float) -> tuple[float, float]:
            """Return ``(log ||r||, log ||L x||)`` at ``lambda = exp(t)``.

            References:
                [REF-HANSEN-OLEARY-1993, Section 4.2]. No equation is evaluated.

            """
            point = curve.point(math.exp(t))
            return 0.5 * math.log(point.rho), 0.5 * math.log(point.eta)

        t = math.log(lam)
        (x0, y0), (x1, y1), (x2, y2) = (
            coordinates(t - step),
            coordinates(t),
            coordinates(t + step),
        )
        dx, dy = (x2 - x0) / (2 * step), (y2 - y0) / (2 * step)
        ddx, ddy = (x2 - 2 * x1 + x0) / step**2, (y2 - 2 * y1 + y0) / step**2
        numeric = (dx * ddy - ddx * dy) / (dx * dx + dy * dy) ** 1.5
        exact = _curvature(curve.point(lam), lam)
        self.assertAlmostEqual(exact, numeric, delta=1e-4 * max(1.0, abs(numeric)))

    def test_curvature_and_derivative_identity(self) -> None:
        """Curvature equals finite differences; ``rho' = -lambda^2 eta'``."""
        problem = backward_heat_problem(16, time=0.005)
        data, _ = noisy_data(problem, 1e-2, 3)
        identity_curve = _Curve(problem.operator, data, None, None)
        general_curve = _Curve(problem.operator, data, first_difference(16), None)
        for curve in (identity_curve, general_curve):
            for lam in (3e-3, 3e-2, 0.3):
                with self.subTest(general=curve.penalty is not None, lam=lam):
                    self.assert_curvature_matches_finite_differences(curve, lam)
                    step = 1e-6 * lam
                    upper, lower = curve.point(lam + step), curve.point(lam - step)
                    rho_derivative = (upper.rho - lower.rho) / (2 * step)
                    eta_derivative = curve.point(lam).eta_derivative
                    self.assertAlmostEqual(
                        rho_derivative,
                        -lam * lam * eta_derivative,
                        delta=1e-5 * abs(lam * lam * eta_derivative),
                    )
                    numeric_eta = (upper.eta - lower.eta) / (2 * step)
                    self.assertAlmostEqual(
                        eta_derivative, numeric_eta, delta=1e-5 * abs(numeric_eta)
                    )

    def test_identity_penalty_through_qr_matches_svd_path(self) -> None:
        """Passing ``L = I`` explicitly (QR path) reproduces the SVD path."""
        problem = backward_heat_problem(12, time=0.005)
        data, _ = noisy_data(problem, 1e-2, 4)
        grid = tuple(10.0 ** (-3 + 2.5 * i / 9) for i in range(10))
        identity = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(12)) for i in range(12)
        )
        for rule in (generalized_cross_validation, l_curve):
            with self.subTest(rule=rule.__name__):
                svd_path = rule(problem.operator, data, grid=grid)
                qr_path = rule(problem.operator, data, grid=grid, penalty=identity)
                for a, b in zip(svd_path.criterion, qr_path.criterion, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-8 * max(1.0, abs(a)))
                for a, b in zip(
                    svd_path.residual_norms, qr_path.residual_norms, strict=False
                ):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                self.assertAlmostEqual(
                    svd_path.regularization,
                    qr_path.regularization,
                    delta=1e-5 * svd_path.regularization,
                )


class ParameterChoiceTests(unittest.TestCase):
    """L-curve, GCV, and the discrepancy principle on seeded problems."""

    def test_rules_land_near_the_error_optimal_lambda(self) -> None:
        """On ``deriv2`` (``n = 32``, 1% noise) every rule is within a factor 4.

        With the noise of ``noisy_data`` and seeds 1-6, the observed ratios to the
        error-optimal ``lambda`` were 0.48-0.75 (L-curve), 0.44-1.04 (GCV), and
        1.18-2.33 (discrepancy).
        """
        problem = second_derivative_problem(32)
        svd = singular_value_decomposition(problem.operator)
        for seed in (1, 2, 3):
            data, delta = noisy_data(problem, 1e-2, seed)
            best = optimal_regularization(problem, data, svd)
            choices = (
                l_curve(problem.operator, data, decomposition=svd),
                generalized_cross_validation(problem.operator, data, decomposition=svd),
                discrepancy_principle(problem.operator, data, delta, decomposition=svd),
            )
            for choice in choices:
                with self.subTest(seed=seed, method=choice.method):
                    ratio = choice.regularization / best
                    self.assertGreater(ratio, 0.25)
                    self.assertLess(ratio, 4.0)
                    self.assertTrue(choice.converged)
                    self.assertFalse(choice.at_boundary)
                    self.assertEqual(len(choice.grid), len(choice.criterion))
                    self.assertAlmostEqual(
                        choice.solution.regularization, choice.regularization, delta=0.0
                    )

    def test_discrepancy_principle_hits_the_target(self) -> None:
        """``||r(lambda)|| = tau delta`` to ``1e-9`` relative, for both penalties."""
        problem = backward_heat_problem(16, time=0.005)
        svd = singular_value_decomposition(problem.operator)
        for seed in (1, 2, 3):
            data, delta = noisy_data(problem, 1e-2, seed)
            best = optimal_regularization(problem, data, svd)
            choice = discrepancy_principle(
                problem.operator, data, delta, decomposition=svd
            )
            self.assertAlmostEqual(
                choice.solution.residual_norm, delta, delta=1e-9 * delta
            )
            self.assertGreater(choice.regularization / best, 1.0 / 3.0)
            self.assertLess(choice.regularization / best, 3.0)
            safer = discrepancy_principle(
                problem.operator, data, delta, safety_factor=1.1, decomposition=svd
            )
            self.assertGreater(safer.regularization, choice.regularization)
            self.assertAlmostEqual(
                safer.solution.residual_norm, 1.1 * delta, delta=1e-9 * delta
            )
        general = discrepancy_principle(
            problem.operator, data, delta, penalty=first_difference(16), points=25
        )
        self.assertAlmostEqual(
            general.solution.residual_norm, delta, delta=1e-8 * delta
        )

    def test_discrepancy_principle_rejects_unattainable_targets(self) -> None:
        """A target above ``||b||`` or below the least-squares residual raises."""
        problem = backward_heat_problem(8, time=0.005)
        data, delta = noisy_data(problem, 1e-2, 7)
        with self.assertRaises(ValueError):
            discrepancy_principle(problem.operator, data, 2.0 * norm(data))
        tall = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            discrepancy_principle(tall, (1.0, 1.0, 0.0), 1e-3)
        with self.assertRaises(ValueError):
            discrepancy_principle(problem.operator, data, 0.0)
        with self.assertRaises(ValueError):
            discrepancy_principle(problem.operator, data, delta, safety_factor=0.9)

    def test_grid_handling(self) -> None:
        """User grids are validated; an optimum at the grid end is flagged."""
        problem = second_derivative_problem(16)
        data, _ = noisy_data(problem, 1e-2, 2)
        narrow = (1.0, 2.0, 4.0)
        choice = generalized_cross_validation(problem.operator, data, grid=narrow)
        self.assertTrue(choice.at_boundary)
        self.assertEqual(choice.grid, narrow)
        with self.assertRaises(ValueError):
            l_curve(problem.operator, data, grid=(1.0, 2.0))
        with self.assertRaises(ValueError):
            l_curve(problem.operator, data, grid=(1.0, 1.0, 2.0))
        with self.assertRaises(ValueError):
            l_curve(problem.operator, data, points=2)
        with self.assertRaises(ValueError):
            generalized_cross_validation(((0.0, 0.0), (0.0, 0.0)), (1.0, 1.0))

    def test_general_penalty_choices_run(self) -> None:
        """GCV and the L-curve with a difference penalty pick an interior lambda."""
        problem = second_derivative_problem(16)
        data, _ = noisy_data(problem, 1e-2, 2)
        penalty = first_difference(16)
        for rule in (generalized_cross_validation, l_curve):
            with self.subTest(rule=rule.__name__):
                choice = rule(problem.operator, data, penalty=penalty, points=20)
                self.assertFalse(choice.at_boundary)
                self.assertIsNone(choice.solution.filter_factors)
                self.assertLess(
                    relative_error(choice.solution.solution, problem.exact_solution),
                    0.2,
                )


class NonlinearTikhonovTests(unittest.TestCase):
    """Nonlinear Tikhonov by Levenberg-Marquardt."""

    def test_linear_residual_reproduces_tikhonov(self) -> None:
        """For ``r(x) = A x - b`` and ``x_bar = 0`` it equals ``tikhonov``."""
        problem = backward_heat_problem(8, time=0.005)
        data, _ = noisy_data(problem, 1e-2, 1)
        operator = problem.operator

        def residual(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``A x - b``.

            References:
                [REF-HANSEN-2008, Section 2.2]. No equation is evaluated.

            """
            return tuple(a - b for a, b in zip(matvec(operator, x), data, strict=True))

        lam = 0.05
        result = nonlinear_tikhonov(
            residual,
            (0.5,) * 8,
            lam,
            prior_mean=(0.0,) * 8,
            jacobian=lambda _: operator,
        )
        expected = tikhonov(operator, data, lam)
        self.assertTrue(result.converged)
        self.assertLess(relative_error(result.solution, expected.solution), 1e-7)
        self.assertAlmostEqual(result.residual_norm, expected.residual_norm, delta=1e-8)
        self.assertAlmostEqual(result.penalty_norm, expected.solution_norm, delta=1e-7)

    def test_exponential_model_stationarity_and_shrinkage(self) -> None:
        """Stationarity holds, and a larger ``lambda`` shrinks toward ``x_bar``."""
        times = tuple(0.25 * i for i in range(12))
        truth = (2.0, 0.7)
        generator = RandomStream(42).child("noise").generator()
        observed = tuple(
            truth[0] * math.exp(-truth[1] * t) + generator.gauss(0.0, 0.02)
            for t in times
        )
        prior = (1.5, 0.5)

        def residual(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``a exp(-k t_i) - y_i``.

            References:
                [REF-HANSEN-2008, Section 2.2]. No equation is evaluated.

            """
            return tuple(
                x[0] * math.exp(-x[1] * t) - y
                for t, y in zip(times, observed, strict=True)
            )

        def jacobian(x: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
            """Return the analytic Jacobian of ``residual``.

            References:
                [REF-NOCEDAL-WRIGHT-2006, Section 10.1]. No equation is evaluated.

            """
            return tuple(
                (math.exp(-x[1] * t), -x[0] * t * math.exp(-x[1] * t)) for t in times
            )

        penalties = []
        for lam in (0.0, 0.1, 1.0, 10.0):
            result = nonlinear_tikhonov(
                residual,
                (1.0, 1.0),
                lam,
                prior_mean=prior,
                jacobian=jacobian,
                tolerance=1e-14,
            )
            self.assertTrue(result.converged)
            r = residual(result.solution)
            j = jacobian(result.solution)
            for column in range(2):
                gradient = math.fsum(j[i][column] * r[i] for i in range(len(times)))
                gradient += lam * lam * (result.solution[column] - prior[column])
                self.assertAlmostEqual(gradient, 0.0, delta=1e-7)
            penalties.append(result.penalty_norm)
        self.assertTrue(
            all(a > b for a, b in zip(penalties, penalties[1:], strict=False))
        )
        unpenalized = nonlinear_tikhonov(residual, (1.0, 1.0), 0.0, jacobian=jacobian)
        self.assertAlmostEqual(unpenalized.solution[0], truth[0], delta=0.1)
        self.assertAlmostEqual(unpenalized.solution[1], truth[1], delta=0.1)
        default_center = nonlinear_tikhonov(
            residual, (1.5, 0.5), 1.0, jacobian=jacobian
        )
        self.assertEqual(default_center.prior_mean, (1.5, 0.5))
        with_prior = nonlinear_tikhonov(
            residual, (1.5, 0.5), 1.0, prior_mean=prior, jacobian=jacobian
        )
        for a, b in zip(default_center.solution, with_prior.solution, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6)
        with_numeric = nonlinear_tikhonov(residual, (1.5, 0.5), 1.0, prior_mean=prior)
        for a, b in zip(with_numeric.solution, with_prior.solution, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6)

    def test_nonlinear_validation(self) -> None:
        """Shapes and the parameter are validated."""

        def residual(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``x``.

            References:
                [REF-HANSEN-2008, Section 2.2]. No equation is evaluated.

            """
            return x

        with self.assertRaises(ValueError):
            nonlinear_tikhonov(residual, (), 1.0)
        with self.assertRaises(ValueError):
            nonlinear_tikhonov(residual, (1.0,), -1.0)
        with self.assertRaises(ValueError):
            nonlinear_tikhonov(residual, (1.0,), 1.0, prior_mean=(0.0, 0.0))
        with self.assertRaises(ValueError):
            nonlinear_tikhonov(residual, (1.0,), 1.0, penalty=((1.0, 1.0),))


if __name__ == "__main__":
    unittest.main()
