"""Tests for Householder QR, linear models, GLMs, and penalized regression.

Reference values are computed independently inside each test: exact rational
normal equations with :class:`fractions.Fraction`, a hand-written Newton
iteration for logistic regression, closed-form Poisson maximum-likelihood
estimates for factor designs, and explicit sandwich and KKT formulas.
Stochastic checks use fixed seeds and binomial-error tolerances.
"""

from __future__ import annotations

import math
import time
import unittest
import warnings
from fractions import Fraction

from probstat.foundations import RandomStream
from probstat.inference import (
    GLMConvergenceWarning,
    RankDeficientDesignError,
    SeparationError,
    SeparationWarning,
    cross_validate_elastic_net,
    design_matrix,
    elastic_net_path,
    fit_elastic_net,
    fit_glm,
    fit_linear_model,
    fit_ridge,
    fit_simple_linear_regression,
    glm_family,
    kfold_assignment,
    student_t_quantile,
    variance_inflation_factors,
)
from probstat.linalg import householder_qr, solve_spd, symmetric_eigen

FractionMatrix = list[list[Fraction]]


def exact_inverse(matrix: FractionMatrix) -> FractionMatrix:
    """Invert a square rational matrix by Gauss-Jordan elimination.

    Args:
        matrix: Non-singular square matrix of fractions.

    Returns:
        The exact inverse.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.2]. Test helper; no new equation.

    """
    size = len(matrix)
    work = [
        row[:] + [Fraction(int(i == j)) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(r for r in range(column, size) if work[r][column] != 0)
        work[column], work[pivot] = work[pivot], work[column]
        factor = work[column][column]
        work[column] = [value / factor for value in work[column]]
        for r in range(size):
            if r != column and work[r][column] != 0:
                scale = work[r][column]
                work[r] = [
                    a - scale * b for a, b in zip(work[r], work[column], strict=True)
                ]
    return [row[size:] for row in work]


def exact_matmul(left: FractionMatrix, right: FractionMatrix) -> FractionMatrix:
    """Multiply two rational matrices exactly.

    Args:
        left: ``m x k`` matrix.
        right: ``k x n`` matrix.

    Returns:
        The ``m x n`` product.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. Test helper; no new equation.

    """
    return [
        [
            sum((left[i][k] * right[k][j] for k in range(len(right))), Fraction(0))
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def exact_transpose(matrix: FractionMatrix) -> FractionMatrix:
    """Transpose a rational matrix.

    Args:
        matrix: Rectangular matrix.

    Returns:
        Its transpose.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. Test helper; no new equation.

    """
    return [list(column) for column in zip(*matrix, strict=True)]


# Small three-column design used for exact rational checks.
DESIGN_TEXT = [
    ("1", "1", "2"),
    ("1", "2", "1"),
    ("1", "3", "5"),
    ("1", "4", "3"),
    ("1", "5", "6"),
    ("1", "6", "4"),
    ("1", "7", "8"),
    ("1", "8", "7"),
]
RESPONSE_TEXT = ["3.1", "4.0", "8.2", "6.9", "11.4", "9.8", "14.1", "15.3"]


class ExactOLS:
    """Exact rational OLS quantities for the shared small design.

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapters 3-6]. Test helper.

    """

    def __init__(self, weights: list[Fraction] | None = None) -> None:
        """Compute every exact quantity once.

        Args:
            weights: Optional exact case weights.

        References:
            [EQ-LM-OLS-001]. Test helper; no new equation.

        """
        self.x = [[Fraction(v) for v in row] for row in DESIGN_TEXT]
        self.y = [Fraction(v) for v in RESPONSE_TEXT]
        n = len(self.y)
        self.w = weights or [Fraction(1)] * n
        xt = exact_transpose(self.x)
        xtw = [[xt[j][i] * self.w[i] for i in range(n)] for j in range(len(xt))]
        self.gram_inverse = exact_inverse(exact_matmul(xtw, self.x))
        xty = exact_matmul(xtw, [[v] for v in self.y])
        self.beta = [row[0] for row in exact_matmul(self.gram_inverse, xty)]
        self.fitted = [
            sum((b * v for b, v in zip(self.beta, row, strict=True)), Fraction(0))
            for row in self.x
        ]
        self.residuals = [a - b for a, b in zip(self.y, self.fitted, strict=True)]
        self.n, self.p = n, len(self.beta)
        self.rss = sum(
            (wi * e * e for wi, e in zip(self.w, self.residuals, strict=True)),
            Fraction(0),
        )
        self.sigma2 = self.rss / (n - self.p)
        mean = sum(
            (wi * v for wi, v in zip(self.w, self.y, strict=True)), Fraction(0)
        ) / sum(self.w, Fraction(0))
        self.tss = sum(
            (wi * (v - mean) ** 2 for wi, v in zip(self.w, self.y, strict=True)),
            Fraction(0),
        )
        self.leverage = [self.w[i] * self.quadratic(self.x[i]) for i in range(n)]

    def quadratic(self, row: list[Fraction]) -> Fraction:
        """Return ``x' (X'WX)^-1 x`` exactly.

        Args:
            row: Design row.

        Returns:
            The exact quadratic form.

        References:
            [EQ-LM-PRED-001]. Test helper; no new equation.

        """
        size = len(row)
        return sum(
            (
                row[i] * self.gram_inverse[i][j] * row[j]
                for i in range(size)
                for j in range(size)
            ),
            Fraction(0),
        )


class HouseholderQRTests(unittest.TestCase):
    """The QR factorization reproduces the matrix with orthonormal Q."""

    def test_factorization_reconstructs_matrix(self) -> None:
        """Q'Q = I, R is upper triangular, and QR = A."""
        a = (
            (2.0, -1.0, 0.5),
            (1.0, 3.0, 2.0),
            (0.0, 1.0, -4.0),
            (4.0, 0.5, 1.0),
            (-2.0, 2.0, 3.0),
        )
        qr = householder_qr(a)
        self.assertTrue(qr.full_rank)
        for i in range(3):
            for j in range(3):
                gram = math.fsum(row[i] * row[j] for row in qr.q)
                self.assertAlmostEqual(gram, float(i == j), places=13)
                if j < i:
                    self.assertEqual(qr.r[i][j], 0.0)
        for i, row in enumerate(a):
            for j in range(3):
                value = math.fsum(qr.q[i][k] * qr.r[k][j] for k in range(3))
                self.assertAlmostEqual(value, row[j], places=13)

    def test_rank_deficiency_is_flagged_not_hidden(self) -> None:
        """A column equal to a combination of earlier ones is reported."""
        a = ((1.0, 2.0, 3.0), (1.0, 0.0, 1.0), (1.0, 5.0, 6.0), (1.0, 1.0, 2.0))
        qr = householder_qr(a)
        self.assertEqual(qr.rank_deficient, (2,))
        with self.assertRaises(ValueError):
            qr.solve((1.0, 2.0, 3.0, 4.0))
        with self.assertRaises(ValueError):
            householder_qr(((1.0, 2.0, 3.0),))


class OrdinaryLeastSquaresTests(unittest.TestCase):
    """OLS output matches exact rational normal-equation arithmetic."""

    def setUp(self) -> None:
        """Fit the shared small design."""
        self.exact = ExactOLS()
        self.fit = fit_linear_model(
            [[float(v) for v in row] for row in self.exact.x],
            [float(v) for v in self.exact.y],
            column_names=["(Intercept)", "a", "b"],
        )

    def test_coefficients_and_fit_measures(self) -> None:
        """Coefficients, RSS, R^2, adjusted R^2, and F agree with exact values."""
        e, f = self.exact, self.fit
        for got, want in zip(f.coefficients, e.beta, strict=True):
            self.assertAlmostEqual(got, float(want), places=12)
        self.assertAlmostEqual(f.residual_sum_squares, float(e.rss), places=12)
        self.assertAlmostEqual(f.residual_variance, float(e.sigma2), places=12)
        r2 = 1 - e.rss / e.tss
        self.assertAlmostEqual(f.r_squared, float(r2), places=12)
        adjusted = 1 - (1 - r2) * Fraction(e.n - 1, e.n - e.p)
        self.assertAlmostEqual(f.adjusted_r_squared, float(adjusted), places=12)
        f_stat = ((e.tss - e.rss) / (e.p - 1)) / e.sigma2
        self.assertAlmostEqual(f.f_statistic, float(f_stat), places=9)
        self.assertEqual(f.f_df, (2, 5))
        self.assertTrue(f.has_intercept)
        self.assertAlmostEqual(f.coefficient("b"), float(e.beta[2]), places=12)

    def test_standard_errors_and_t_tests(self) -> None:
        """SEs are sqrt(s^2 (X'X)^-1_jj); p-values are two-sided t on n-p df."""
        e, f = self.exact, self.fit
        for j in range(e.p):
            se = math.sqrt(float(e.sigma2 * e.gram_inverse[j][j]))
            self.assertAlmostEqual(f.standard_errors[j], se, places=12)
            self.assertAlmostEqual(f.t_values[j], float(e.beta[j]) / se, places=9)
            quantile = student_t_quantile(1 - f.p_values[j] / 2, 5)
            self.assertAlmostEqual(quantile, abs(f.t_values[j]), places=6)
        lower, upper = f.coefficient_intervals(0.9)[1]
        half = student_t_quantile(0.95, 5) * f.standard_errors[1]
        self.assertAlmostEqual(lower, f.coefficients[1] - half, places=12)
        self.assertAlmostEqual(upper, f.coefficients[1] + half, places=12)

    def test_leverage_residuals_and_cooks_distance(self) -> None:
        """Diagnostics match exact leverage, studentization, and Cook formulas."""
        e, f = self.exact, self.fit
        self.assertAlmostEqual(math.fsum(f.leverage), e.p, places=12)
        for i in range(e.n):
            h = e.leverage[i]
            self.assertAlmostEqual(f.leverage[i], float(h), places=12)
            r = float(e.residuals[i]) / math.sqrt(float(e.sigma2 * (1 - h)))
            self.assertAlmostEqual(f.standardized_residuals[i], r, places=10)
            deleted = (e.rss - e.residuals[i] ** 2 / (1 - h)) / (e.n - e.p - 1)
            t = float(e.residuals[i]) / math.sqrt(float(deleted * (1 - h)))
            self.assertAlmostEqual(f.studentized_residuals[i], t, places=10)
            cook = e.residuals[i] ** 2 * h / (e.p * e.sigma2 * (1 - h) ** 2)
            self.assertAlmostEqual(f.cooks_distance[i], float(cook), places=10)

    def test_hc0_and_hc3_match_explicit_sandwich(self) -> None:
        """HC0/HC3 equal (X'X)^-1 X' diag(omega) X (X'X)^-1 computed exactly."""
        e, f = self.exact, self.fit
        for kind in ("HC0", "HC3"):
            omega = [
                e.residuals[i] ** 2 / ((1 - e.leverage[i]) ** 2 if kind == "HC3" else 1)
                for i in range(e.n)
            ]
            xt = exact_transpose(e.x)
            meat = exact_matmul(
                [[xt[j][i] * omega[i] for i in range(e.n)] for j in range(e.p)],
                e.x,
            )
            sandwich = exact_matmul(exact_matmul(e.gram_inverse, meat), e.gram_inverse)
            got = f.robust_covariance(kind)
            for i in range(e.p):
                for j in range(e.p):
                    self.assertAlmostEqual(got[i][j], float(sandwich[i][j]), places=11)
            errors = f.robust_standard_errors(kind)
            self.assertAlmostEqual(
                errors[0], math.sqrt(float(sandwich[0][0])), places=11
            )
        hc1 = f.robust_covariance("HC1")
        hc0 = f.robust_covariance("HC0")
        self.assertAlmostEqual(hc1[1][1], hc0[1][1] * 8 / 5, places=12)
        with self.assertRaises(ValueError):
            f.robust_covariance("HC9")  # type: ignore[arg-type]

    def test_confidence_and_prediction_intervals(self) -> None:
        """Interval half-widths use s^2 x0'(X'X)^-1 x0 (+ s^2 for prediction)."""
        e, f = self.exact, self.fit
        x0 = [Fraction(1), Fraction(9, 2), Fraction(7, 2)]
        q = student_t_quantile(0.975, 5)
        mean = sum((b * v for b, v in zip(e.beta, x0, strict=True)), Fraction(0))
        quad = e.quadratic(x0)
        rows = [[float(v) for v in x0]]
        ci = f.interval(rows, kind="confidence")
        pi = f.interval(rows, kind="prediction")
        self.assertAlmostEqual(ci.mean[0], float(mean), places=12)
        self.assertAlmostEqual(
            ci.upper[0] - ci.mean[0], q * math.sqrt(float(e.sigma2 * quad)), places=11
        )
        self.assertAlmostEqual(
            pi.upper[0] - pi.mean[0],
            q * math.sqrt(float(e.sigma2 * (1 + quad))),
            places=11,
        )
        self.assertEqual(f.predict(rows), ci.mean)
        with self.assertRaises(ValueError):
            f.interval(rows, kind="other")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            f.predict([[1.0, 2.0]])

    def test_gaussian_log_likelihood_and_aic(self) -> None:
        """The log-likelihood uses the MLE variance RSS/n and k = p + 1."""
        e, f = self.exact, self.fit
        rss = float(e.rss)
        ll = -0.5 * e.n * (math.log(2 * math.pi * rss / e.n) + 1)
        self.assertAlmostEqual(f.log_likelihood, ll, places=12)
        self.assertAlmostEqual(f.aic, 2 * 4 - 2 * ll, places=11)
        self.assertAlmostEqual(f.bic, 4 * math.log(8) - 2 * ll, places=11)


class TextbookAndWeightedTests(unittest.TestCase):
    """Anscombe's first data set and weighted least squares."""

    def test_anscombe_quartet_first_set(self) -> None:
        """Slope 0.500, intercept 3.00, R^2 0.67, matching exact arithmetic."""
        x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        y_text = [
            "8.04",
            "6.95",
            "7.58",
            "8.81",
            "8.33",
            "9.96",
            "7.24",
            "4.26",
            "10.84",
            "4.82",
            "5.68",
        ]
        y = [Fraction(v) for v in y_text]
        n = len(x)
        x_bar = Fraction(sum(x), n)
        y_bar = sum(y, Fraction(0)) / n
        sxx = sum((Fraction(v) - x_bar) ** 2 for v in x)
        sxy = sum(
            ((Fraction(a) - x_bar) * (b - y_bar) for a, b in zip(x, y, strict=True)),
            Fraction(0),
        )
        slope = sxy / sxx
        intercept = y_bar - slope * x_bar
        fit = fit_linear_model([[1.0, float(v)] for v in x], [float(v) for v in y])
        self.assertAlmostEqual(fit.coefficients[1], float(slope), places=13)
        self.assertAlmostEqual(fit.coefficients[0], float(intercept), places=12)
        self.assertAlmostEqual(fit.coefficients[1], 0.500, places=3)
        self.assertAlmostEqual(fit.coefficients[0], 3.00, places=2)
        self.assertAlmostEqual(fit.r_squared, 0.67, places=2)
        simple = fit_simple_linear_regression(x, [float(v) for v in y])
        self.assertAlmostEqual(simple.slope, fit.coefficients[1], places=12)
        self.assertAlmostEqual(simple.r_squared, fit.r_squared, places=12)

    def test_weighted_least_squares_exact_and_replication(self) -> None:
        """WLS solves X'WX b = X'Wy and equals OLS on replicated rows."""
        weights = [Fraction(v) for v in (1, 2, 1, 3, 1, 2, 1, 1)]
        exact = ExactOLS(weights)
        rows = [[float(v) for v in row] for row in exact.x]
        response = [float(v) for v in exact.y]
        fit = fit_linear_model(rows, response, weights=[float(w) for w in weights])
        for got, want in zip(fit.coefficients, exact.beta, strict=True):
            self.assertAlmostEqual(got, float(want), places=12)
        self.assertAlmostEqual(fit.residual_sum_squares, float(exact.rss), places=12)
        for i in range(exact.n):
            self.assertAlmostEqual(fit.leverage[i], float(exact.leverage[i]), places=12)
        replicated_rows: list[list[float]] = []
        replicated_y: list[float] = []
        for row, value, weight in zip(rows, response, weights, strict=True):
            replicated_rows.extend([row] * int(weight))
            replicated_y.extend([value] * int(weight))
        ols = fit_linear_model(replicated_rows, replicated_y)
        for got, want in zip(fit.coefficients, ols.coefficients, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        with self.assertRaises(ValueError):
            fit_linear_model(rows, response, weights=[1.0] * 7 + [0.0])

    def test_prediction_interval_coverage_simulation(self) -> None:
        """Seeded simulation: 95% PI and CI coverage within binomial error."""
        generator = RandomStream(20261012).child("coverage").generator()
        xs = [i / 14 for i in range(15)]
        rows = [[1.0, x, x * x] for x in xs]
        beta = (1.0, 2.0, -0.5)
        x0 = (1.0, 0.3, 0.09)
        true_mean = math.fsum(b * v for b, v in zip(beta, x0, strict=True))
        replications = 400
        pi_hits = 0
        ci_hits = 0
        for _ in range(replications):
            y = [
                math.fsum(b * v for b, v in zip(beta, row, strict=True))
                + generator.gauss(0.0, 0.7)
                for row in rows
            ]
            fit = fit_linear_model(rows, y)
            new_y = true_mean + generator.gauss(0.0, 0.7)
            pi = fit.interval([x0], kind="prediction")
            ci = fit.interval([x0], kind="confidence")
            pi_hits += pi.lower[0] <= new_y <= pi.upper[0]
            ci_hits += ci.lower[0] <= true_mean <= ci.upper[0]
        tolerance = 3.5 * math.sqrt(0.95 * 0.05 / replications)
        self.assertLess(abs(pi_hits / replications - 0.95), tolerance)
        self.assertLess(abs(ci_hits / replications - 0.95), tolerance)


class DesignAndRankTests(unittest.TestCase):
    """Design-matrix encoding, VIFs, and explicit rank-deficiency errors."""

    def test_categorical_treatment_coding(self) -> None:
        """Indicators drop the reference; coefficients are group contrasts."""
        data = {
            "dose": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "site": ["b", "a", "c", "a", "c", "b", "c", "b", "a"],
        }
        design = design_matrix(data, categorical=["site"])
        self.assertEqual(
            design.column_names, ("(Intercept)", "dose", "site[b]", "site[c]")
        )
        self.assertEqual(design.rows[0], (1.0, 1.0, 1.0, 0.0))
        chosen = design_matrix(data, categorical=["site"], reference={"site": "c"})
        self.assertEqual(chosen.column_names[2:], ("site[a]", "site[b]"))
        cell = design_matrix(
            {"site": data["site"]}, intercept=False, categorical=["site"]
        )
        self.assertEqual(cell.column_names, ("site[a]", "site[b]", "site[c]"))
        new = design.transform({"dose": [2.5], "site": ["c"]})
        self.assertEqual(new, ((1.0, 2.5, 0.0, 1.0),))
        with self.assertRaises(ValueError):
            design.transform({"dose": [1.0], "site": ["z"]})
        with self.assertRaises(KeyError):
            design_matrix(data, categorical=["missing"])
        with self.assertRaises(ValueError):
            design_matrix(data, categorical=["site"], reference={"site": "q"})

    def test_factor_only_fit_reproduces_group_means(self) -> None:
        """With only a factor, fitted values are the group means."""
        groups = ["a", "a", "b", "b", "b", "c", "c"]
        y = [1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 2.5]
        fit = fit_linear_model(design_matrix({"g": groups}, categorical=["g"]), y)
        self.assertAlmostEqual(fit.coefficients[0], 2.0, places=12)
        self.assertAlmostEqual(fit.coefficients[1], 6.0 - 2.0, places=12)
        self.assertAlmostEqual(fit.coefficients[2], 2.25 - 2.0, places=12)
        self.assertEqual(fit.f_df, (2, 4))

    def test_variance_inflation_two_predictors(self) -> None:
        """With two predictors, VIF = 1 / (1 - r^2) for their correlation r."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        x2 = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0]
        n = len(x1)
        m1, m2 = sum(x1) / n, sum(x2) / n
        sxy = sum((a - m1) * (b - m2) for a, b in zip(x1, x2, strict=True))
        r2 = sxy**2 / (sum((a - m1) ** 2 for a in x1) * sum((b - m2) ** 2 for b in x2))
        vifs = variance_inflation_factors(design_matrix({"x1": x1, "x2": x2}))
        self.assertAlmostEqual(vifs["x1"], 1 / (1 - r2), places=12)
        self.assertAlmostEqual(vifs["x2"], 1 / (1 - r2), places=12)
        collinear = variance_inflation_factors(
            [[1.0, a, b, a + b] for a, b in zip(x1, x2, strict=True)]
        )
        self.assertTrue(all(math.isinf(v) for v in collinear.values()))

    def test_rank_deficient_design_raises(self) -> None:
        """Duplicated or dummy-trap columns raise; a ridge penalty is explicit."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.5, 2.9, 4.2, 5.1]
        with self.assertRaises(RankDeficientDesignError) as caught:
            fit_linear_model(
                [[1.0, v, 2.0 * v] for v in x], y, column_names=["c", "x", "twice"]
            )
        self.assertIn("twice", str(caught.exception))
        trap = [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        with self.assertRaises(RankDeficientDesignError):
            fit_linear_model(trap, y)
        with self.assertRaises(RankDeficientDesignError):
            fit_glm(trap, [1.0, 0.0, 1.0, 1.0, 0.0], family="binomial")
        with self.assertRaises(RankDeficientDesignError):
            fit_ridge([[v, 2.0 * v] for v in x], y, penalty=0.0)
        ridge = fit_ridge([[v, 2.0 * v] for v in x], y, penalty=0.5)
        self.assertAlmostEqual(
            ridge.standardized_coefficients[0],
            ridge.standardized_coefficients[1],
            places=9,
        )
        self.assertAlmostEqual(
            ridge.coefficients[0], 2.0 * ridge.coefficients[1], places=9
        )
        with self.assertRaises(ValueError):
            fit_linear_model([[1.0, 2.0], [1.0, 3.0]], [1.0, 2.0])


def logistic_newton(
    x: list[float], y: list[int]
) -> tuple[list[float], list[float], float]:
    """Solve two-parameter logistic regression by an explicit Newton iteration.

    Args:
        x: Predictor values.
        y: Binary responses.

    Returns:
        Coefficients, standard errors from the inverse information, deviance.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.4.1]. Test helper.

    """
    b0 = b1 = 0.0
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for xi, yi in zip(x, y, strict=True):
            p = 1.0 / (1.0 + math.exp(-(b0 + b1 * xi)))
            g0 += yi - p
            g1 += (yi - p) * xi
            v = p * (1.0 - p)
            h00 += v
            h01 += v * xi
            h11 += v * xi * xi
        det = h00 * h11 - h01 * h01
        b0 += (h11 * g0 - h01 * g1) / det
        b1 += (-h01 * g0 + h00 * g1) / det
    h00 = h01 = h11 = 0.0
    deviance = 0.0
    for xi, yi in zip(x, y, strict=True):
        p = 1.0 / (1.0 + math.exp(-(b0 + b1 * xi)))
        v = p * (1.0 - p)
        h00 += v
        h01 += v * xi
        h11 += v * xi * xi
        deviance -= 2.0 * (yi * math.log(p) + (1 - yi) * math.log(1.0 - p))
    det = h00 * h11 - h01 * h01
    return [b0, b1], [math.sqrt(h11 / det), math.sqrt(h00 / det)], deviance


class GeneralizedLinearModelTests(unittest.TestCase):
    """IRLS reproduces independent Newton solutions and deviance identities."""

    X = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    Y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]

    def test_logistic_matches_independent_newton(self) -> None:
        """Coefficients, Wald SEs, deviance, and AIC match a hand Newton solve."""
        beta, errors, deviance = logistic_newton(self.X, self.Y)
        fit = fit_glm([[1.0, v] for v in self.X], self.Y, family="binomial")
        self.assertTrue(fit.converged)
        self.assertFalse(fit.separation_detected)
        for j in range(2):
            self.assertAlmostEqual(fit.coefficients[j], beta[j], places=9)
            self.assertAlmostEqual(fit.standard_errors[j], errors[j], places=8)
        self.assertAlmostEqual(fit.deviance, deviance, places=10)
        self.assertAlmostEqual(fit.aic, deviance + 4.0, places=10)
        self.assertEqual(fit.statistic_distribution, "normal")
        self.assertEqual(fit.dispersion, 1.0)
        null = -2 * (6 * math.log(0.6) + 4 * math.log(0.4))
        self.assertAlmostEqual(fit.null_deviance, null, places=9)
        self.assertAlmostEqual(
            math.fsum(r * r for r in fit.deviance_residuals), fit.deviance, places=10
        )
        prediction = fit.predict([[1.0, 2.0]])[0]
        self.assertAlmostEqual(
            prediction, 1 / (1 + math.exp(-(beta[0] + 2 * beta[1]))), places=9
        )

    def test_probit_score_equations(self) -> None:
        """At the probit MLE, sum (y-mu) phi(eta) x / (mu(1-mu)) = 0."""
        fit = fit_glm(
            [[1.0, v] for v in self.X], self.Y, family="binomial", link="probit"
        )
        for j in range(2):
            score = math.fsum(
                (y - mu)
                * math.exp(-0.5 * eta * eta)
                / math.sqrt(2 * math.pi)
                / (mu * (1 - mu))
                * (1.0 if j == 0 else x)
                for x, y, mu, eta in zip(
                    self.X, self.Y, fit.fitted_values, fit.linear_predictor, strict=True
                )
            )
            self.assertAlmostEqual(score, 0.0, places=8)

    def test_grouped_binomial_equals_expanded_bernoulli(self) -> None:
        """Trials as prior weights give the same coefficients and SEs."""
        levels = [1.0, 2.0, 3.0, 4.0]
        trials = [5, 6, 5, 4]
        successes = [1, 2, 4, 3]
        grouped = fit_glm(
            [[1.0, v] for v in levels],
            [s / t for s, t in zip(successes, trials, strict=True)],
            family="binomial",
            weights=trials,
        )
        rows: list[list[float]] = []
        ys: list[int] = []
        for v, t, s in zip(levels, trials, successes, strict=True):
            rows.extend([[1.0, v]] * t)
            ys.extend([1] * s + [0] * (t - s))
        expanded = fit_glm(rows, ys, family="binomial")
        for j in range(2):
            self.assertAlmostEqual(
                grouped.coefficients[j], expanded.coefficients[j], places=9
            )
            self.assertAlmostEqual(
                grouped.standard_errors[j], expanded.standard_errors[j], places=8
            )

    def test_poisson_factor_model_deviance_identities(self) -> None:
        """Group means are the MLE; deviance, Pearson, AIC follow by hand."""
        groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        counts = [2, 3, 1, 4, 6, 8, 5, 7, 0, 1, 2, 1]
        fit = fit_glm(
            design_matrix({"g": groups}, categorical=["g"]), counts, family="poisson"
        )
        means = {"a": 2.5, "b": 6.5, "c": 1.0}
        mu = [means[g] for g in groups]
        for got, want in zip(fit.fitted_values, mu, strict=True):
            self.assertAlmostEqual(got, want, places=10)
        self.assertAlmostEqual(
            math.fsum(y - m for y, m in zip(counts, fit.fitted_values, strict=True)),
            0.0,
            places=10,
        )
        deviance = 2 * math.fsum(
            y * math.log(y / m) for y, m in zip(counts, mu, strict=True) if y > 0
        )
        self.assertAlmostEqual(fit.deviance, deviance, places=10)
        pearson = math.fsum((y - m) ** 2 / m for y, m in zip(counts, mu, strict=True))
        self.assertAlmostEqual(fit.pearson_chi2, pearson, places=10)
        grand = sum(counts) / len(counts)
        null = 2 * math.fsum(y * math.log(y / grand) for y in counts if y > 0)
        self.assertAlmostEqual(fit.null_deviance, null, places=9)
        loglik = math.fsum(
            (y * math.log(m) if y else 0.0) - m - math.lgamma(y + 1)
            for y, m in zip(counts, mu, strict=True)
        )
        self.assertAlmostEqual(fit.log_likelihood, loglik, places=10)
        self.assertAlmostEqual(fit.aic, -2 * loglik + 6, places=9)
        self.assertEqual((fit.df_residual, fit.df_null), (9, 11))

    def test_poisson_offset_gives_group_rates(self) -> None:
        """With log-exposure offsets, the fitted rate is total count/exposure."""
        groups = ["a", "a", "a", "b", "b", "b"]
        exposure = [1.0, 2.0, 0.5, 3.0, 1.0, 2.0]
        counts = [3, 5, 1, 2, 1, 1]
        fit = fit_glm(
            design_matrix({"g": groups}, categorical=["g"]),
            counts,
            family="poisson",
            offset=[math.log(t) for t in exposure],
        )
        rate_a, rate_b = 9 / 3.5, 4 / 6.0
        self.assertAlmostEqual(math.exp(fit.coefficients[0]), rate_a, places=10)
        self.assertAlmostEqual(math.exp(sum(fit.coefficients)), rate_b, places=10)

    def test_gamma_score_equations_and_dispersion(self) -> None:
        """Gamma inverse- and log-link fits solve their score equations."""
        x = [float(i) for i in range(9)]
        y = [2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
        rows = [[1.0, v] for v in x]
        inverse = fit_glm(rows, y, family="gamma")
        log = fit_glm(rows, y, family="gamma", link="log")
        for j in range(2):
            canonical = math.fsum(
                (yi - mi) * row[j]
                for yi, mi, row in zip(y, inverse.fitted_values, rows, strict=True)
            )
            self.assertAlmostEqual(canonical, 0.0, places=8)
            score = math.fsum(
                (yi - mi) / mi * row[j]
                for yi, mi, row in zip(y, log.fitted_values, rows, strict=True)
            )
            self.assertAlmostEqual(score, 0.0, places=9)
        pearson = math.fsum(
            ((yi - mi) / mi) ** 2 for yi, mi in zip(y, log.fitted_values, strict=True)
        )
        self.assertAlmostEqual(log.dispersion, pearson / 7, places=12)
        self.assertEqual(log.statistic_distribution, "t")
        self.assertEqual(log.parameter_count, 3)

    def test_gaussian_glm_equals_weighted_least_squares(self) -> None:
        """Gaussian-identity IRLS reproduces WLS coefficients, SEs, and AIC."""
        exact = ExactOLS()
        rows = [[float(v) for v in row] for row in exact.x]
        y = [float(v) for v in exact.y]
        weights = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0]
        glm = fit_glm(rows, y, weights=weights)
        lm = fit_linear_model(rows, y, weights=weights)
        for j in range(3):
            self.assertAlmostEqual(glm.coefficients[j], lm.coefficients[j], places=11)
            self.assertAlmostEqual(
                glm.standard_errors[j], lm.standard_errors[j], places=11
            )
        self.assertAlmostEqual(glm.deviance, lm.residual_sum_squares, places=11)
        self.assertAlmostEqual(glm.dispersion, lm.residual_variance, places=11)
        self.assertAlmostEqual(glm.aic, lm.aic, places=10)

    def test_separation_is_detected(self) -> None:
        """Complete and quasi-complete separation raise or warn explicitly."""
        complete = [[1.0, v] for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
        quasi = [[1.0, v] for v in (1.0, 2.0, 3.0, 3.0, 4.0, 5.0)]
        y = [0, 0, 0, 1, 1, 1]
        for link in ("logit", "probit"):
            with self.assertRaises(SeparationError):
                fit_glm(complete, y, family="binomial", link=link)
            with self.assertRaises(SeparationError):
                fit_glm(quasi, y, family="binomial", link=link)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_glm(quasi, y, family="binomial", on_separation="warn")
        self.assertTrue(any(issubclass(w.category, SeparationWarning) for w in caught))
        self.assertTrue(fit.separation_detected)
        self.assertEqual(fit.separated_observations, (0, 1, 4, 5))
        overlapping = fit_glm(complete, [0, 1, 0, 1, 0, 1], family="binomial")
        self.assertFalse(overlapping.separation_detected)

    def test_convergence_warning_and_input_errors(self) -> None:
        """Iteration limits warn; invalid families, links, and data raise."""
        rows = [[1.0, v] for v in self.X]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_glm(rows, self.Y, family="binomial", max_iterations=1)
        self.assertFalse(fit.converged)
        self.assertTrue(
            any(issubclass(w.category, GLMConvergenceWarning) for w in caught)
        )
        with self.assertRaises(ValueError):
            glm_family("poisson", "logit")
        with self.assertRaises(ValueError):
            glm_family("tweedie")
        with self.assertRaises(ValueError):
            fit_glm(rows, [2] + self.Y[1:], family="binomial")
        with self.assertRaises(ValueError):
            fit_glm(rows, [-1] + self.Y[1:], family="poisson")
        with self.assertRaises(ValueError):
            fit_glm(rows, [0.0] + [1.0] * 9, family="gamma")
        with self.assertRaises(ValueError):
            fit_glm(rows, self.Y, family="binomial", on_separation="ignore")


def regularization_data() -> tuple[list[list[float]], list[float]]:
    """Return seeded correlated predictors with a sparse true coefficient vector.

    Returns:
        Predictor rows and responses.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4]. Test helper.

    """
    generator = RandomStream(7).child("penalized").generator()
    beta = [2.0, -1.5, 0.0, 0.0, 1.0, 0.0]
    rows: list[list[float]] = []
    y: list[float] = []
    for _ in range(50):
        z = generator.gauss(0.0, 1.0)
        row = [0.6 * z + 0.8 * generator.gauss(0.0, 1.0) for _ in range(6)]
        rows.append(row)
        y.append(
            1.0
            + math.fsum(b * v for b, v in zip(beta, row, strict=True))
            + generator.gauss(0.0, 1.0)
        )
    return rows, y


def standardize(
    rows: list[list[float]], y: list[float]
) -> tuple[list[list[float]], list[float], list[float]]:
    """Centre and scale columns to unit population SD; centre the response.

    Args:
        rows: Predictor rows.
        y: Responses.

    Returns:
        Standardized columns, centred response, and column scales.

    References:
        [EQ-PEN-STD-001]. Test helper; no new equation.

    """
    n = len(rows)
    columns: list[list[float]] = []
    scales: list[float] = []
    for j in range(len(rows[0])):
        column = [row[j] for row in rows]
        mean = sum(column) / n
        scale = math.sqrt(sum((v - mean) ** 2 for v in column) / n)
        columns.append([(v - mean) / scale for v in column])
        scales.append(scale)
    y_mean = sum(y) / n
    return columns, [v - y_mean for v in y], scales


class PenalizedRegressionTests(unittest.TestCase):
    """Ridge closed form, coordinate-descent KKT conditions, and CV folds."""

    def setUp(self) -> None:
        """Generate the shared seeded data set."""
        self.rows, self.y = regularization_data()

    def test_ridge_at_zero_penalty_is_ols(self) -> None:
        """Ridge with lambda = 0 reproduces OLS and has p degrees of freedom."""
        ridge = fit_ridge(self.rows, self.y, penalty=0.0)
        ols = fit_linear_model([[1.0, *row] for row in self.rows], self.y)
        self.assertAlmostEqual(ridge.intercept, ols.coefficients[0], places=11)
        for got, want in zip(ridge.coefficients, ols.coefficients[1:], strict=True):
            self.assertAlmostEqual(got, want, places=11)
        self.assertAlmostEqual(ridge.effective_degrees_of_freedom, 6.0, places=11)
        self.assertAlmostEqual(
            ridge.residual_sum_squares, ols.residual_sum_squares, places=10
        )

    def test_ridge_closed_form_and_effective_df(self) -> None:
        """Ridge solves (X'X + lambda I) b = X'y; df = sum d^2/(d^2 + lambda)."""
        lam = 7.5
        columns, yc, scales = standardize(self.rows, self.y)
        gram = tuple(
            tuple(
                math.fsum(a * b for a, b in zip(ci, cj, strict=True))
                + (lam if i == j else 0.0)
                for j, cj in enumerate(columns)
            )
            for i, ci in enumerate(columns)
        )
        rhs = tuple(
            math.fsum(a * b for a, b in zip(c, yc, strict=True)) for c in columns
        )
        expected = solve_spd(gram, rhs)
        ridge = fit_ridge(self.rows, self.y, penalty=lam)
        for got, want, scale in zip(ridge.coefficients, expected, scales, strict=True):
            self.assertAlmostEqual(got * scale, want, places=10)
        plain = tuple(
            tuple(
                math.fsum(a * b for a, b in zip(ci, cj, strict=True)) for cj in columns
            )
            for ci in columns
        )
        eigenvalues, _ = symmetric_eigen(plain)
        df = math.fsum(d / (d + lam) for d in eigenvalues)
        self.assertAlmostEqual(ridge.effective_degrees_of_freedom, df, places=9)
        self.assertLess(df, 6.0)

    def test_elastic_net_alpha_zero_equals_ridge(self) -> None:
        """Coordinate descent at alpha = 0 equals ridge with penalty n lambda."""
        lam = 0.3
        net = fit_elastic_net(self.rows, self.y, penalty=lam, alpha=0.0)
        ridge = fit_ridge(self.rows, self.y, penalty=lam * len(self.y))
        self.assertTrue(net.converged)
        for got, want in zip(net.coefficients, ridge.coefficients, strict=True):
            self.assertAlmostEqual(got, want, places=8)

    def test_lasso_and_elastic_net_kkt_conditions(self) -> None:
        """Stationarity holds on the active set; |gradient| <= lambda alpha off it."""
        columns, yc, scales = standardize(self.rows, self.y)
        n = len(yc)
        for alpha, lam in ((1.0, 0.15), (0.5, 0.2)):
            fit = fit_elastic_net(self.rows, self.y, penalty=lam, alpha=alpha)
            b = [c * s for c, s in zip(fit.coefficients, scales, strict=True)]
            residual = [
                yc[i] - math.fsum(b[j] * columns[j][i] for j in range(6))
                for i in range(n)
            ]
            inactive = 0
            for j in range(6):
                gradient = (
                    math.fsum(x * r for x, r in zip(columns[j], residual, strict=True))
                    / n
                    - lam * (1 - alpha) * b[j]
                )
                if b[j] != 0.0:
                    self.assertAlmostEqual(
                        gradient, lam * alpha * math.copysign(1.0, b[j]), places=8
                    )
                else:
                    inactive += 1
                    self.assertLessEqual(abs(gradient), lam * alpha + 1e-10)
            self.assertGreater(inactive, 0)
            self.assertIn("x0", fit.active_set)

    def test_path_starts_at_lambda_max(self) -> None:
        """The first path point is all zero; just below it one predictor enters."""
        path = elastic_net_path(self.rows, self.y, n_penalties=10)
        self.assertEqual(path[0].active_set, ())
        self.assertGreater(path[0].penalty, path[-1].penalty)
        below = fit_elastic_net(self.rows, self.y, penalty=0.999 * path[0].penalty)
        self.assertEqual(len(below.active_set), 1)
        self.assertEqual(len(path[-1].active_set), 6)
        with self.assertRaises(ValueError):
            elastic_net_path(self.rows, self.y, alpha=0.0)

    def test_cross_validation_is_seeded_and_selects_signal(self) -> None:
        """Folds are balanced and reproducible; CV keeps the true predictors."""
        labels = kfold_assignment(23, 5, seed=3)
        sizes = sorted(labels.count(k) for k in range(5))
        self.assertEqual(sizes, [4, 4, 5, 5, 5])
        self.assertEqual(labels, kfold_assignment(23, 5, seed=RandomStream(3)))
        self.assertNotEqual(labels, kfold_assignment(23, 5, seed=4))
        first = cross_validate_elastic_net(
            self.rows, self.y, n_penalties=15, folds=5, seed=11
        )
        second = cross_validate_elastic_net(
            self.rows, self.y, n_penalties=15, folds=5, seed=11
        )
        self.assertEqual(first.mean_error, second.mean_error)
        self.assertGreaterEqual(first.penalty_1se, first.penalty_min)
        minimum = min(first.mean_error)
        self.assertEqual(
            first.mean_error[first.penalties.index(first.penalty_min)], minimum
        )
        for name in ("x0", "x1", "x4"):
            self.assertIn(name, first.best.active_set)
        self.assertEqual(len(first.fold_errors), 5)
        with self.assertRaises(ValueError):
            kfold_assignment(5, 6)

    def test_suite_runs_quickly(self) -> None:
        """A full lasso path on the shared data takes well under a second."""
        start = time.perf_counter()
        elastic_net_path(self.rows, self.y, n_penalties=20)
        self.assertLess(time.perf_counter() - start, 2.0)


if __name__ == "__main__":
    unittest.main()
