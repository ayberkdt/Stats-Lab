"""Splines, LOESS, isotonic and kernel regression, and additive models (ADR-0102).

Oracles: closed forms (Marsden's identity, polynomial reproduction, the
min-max formula of isotonic regression, explicit weighted means and
weighted least squares), the exact restricted likelihood of the equivalent
mixed model computed independently with dense algebra, values computed
offline with SciPy 1.14.1 and NumPy 2.2.6 (``make_smoothing_spline``,
``BSpline.design_matrix``, brute-force hat matrices, and
``minimize_scalar``; the calls are quoted in each test), and a seeded
simulation judged by its Monte Carlo standard error.

References:
    [REF-DE-BOOR-1972], [REF-EILERS-MARX-1996], [REF-REINSCH-1967],
    [REF-GREEN-SILVERMAN-1994], [REF-CRAVEN-WAHBA-1979], [REF-WOOD-2011],
    [REF-CLEVELAND-1979], [REF-AYER-ETAL-1955], [REF-NADARAYA-1964],
    [REF-BUJA-HASTIE-TIBSHIRANI-1989], [REF-FRIEDMAN-2001], [REF-NYCHKA-1988].

"""

from __future__ import annotations

import itertools
import math
import statistics
import unittest

from probstat.foundations import resolve_stream
from probstat.inference import (
    backfit_additive_model,
    bspline_basis,
    bspline_design_matrix,
    difference_penalty,
    fit_additive_model,
    fit_linear_model,
    fit_pspline,
    fit_simple_linear_regression,
    fit_smoothing_spline,
    isotonic_regression,
    kernel_regression,
    loess,
    partial_dependence,
    uniform_knots,
)
from probstat.linalg import (
    Matrix,
    cholesky,
    cholesky_solve,
    inverse_spd,
    log_det_spd,
    matmul,
    symmetric_eigen,
    transpose,
)

# Thirty irregularly spaced points: x_i = (i + 0.37 ((7 i) mod 5)/4)/30 rounded
# to six decimals, y_i = sin(2 pi x_i) + N(0, 0.25^2) noise rounded to four.
X = [0.0, 0.0395, 0.079, 0.103083, 0.142583, 0.166667, 0.206167, 0.245667,
     0.26975, 0.30925, 0.333333, 0.372833, 0.412333, 0.436417, 0.475917, 0.5,
     0.5395, 0.579, 0.603083, 0.642583, 0.666667, 0.706167, 0.745667, 0.76975,
     0.80925, 0.833333, 0.872833, 0.912333, 0.936417, 0.975917]  # fmt: skip
Y = [0.2165, 0.4467, -0.0181, 0.5666, 0.9228, 0.9729, 0.7056, 0.8106, 1.0579,
     0.6234, 0.8139, 0.2924, 0.5255, 0.0567, 0.1725, 0.3444, 0.1079, -0.181,
     -1.0397, -0.6379, -1.0154, -1.3085, -0.9526, -0.8079, -0.3606, -0.9385,
     -0.3777, -1.4625, -0.4882, -0.1592]  # fmt: skip


def close(test: unittest.TestCase, a: float, b: float, rel: float = 1e-9) -> None:
    """Assert ``|a - b| <= rel max(1, |b|)``.

    References:
        [REF-PYTHON-UNITTEST]. No mathematical equation is evaluated.

    """
    test.assertLessEqual(abs(a - b), rel * max(1.0, abs(b)), (a, b))


class BSplineBasisTests(unittest.TestCase):
    """De Boor recursion against closed forms and SciPy.

    References:
        [EQ-SMOOTH-001], [REF-DE-BOOR-1972].

    """

    knots = (0.0, 0.0, 0.0, 0.0, 0.2, 0.45, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0)

    def test_values_match_scipy(self) -> None:
        """``scipy.interpolate.BSpline.design_matrix([x], knots, 3)`` (SciPy 1.14.1).

        References:
            [EQ-SMOOTH-001], [REF-DE-BOOR-1972].

        """
        expected = {
            0.1: (0.125, 0.6219135802469136, 0.2308641975308642, 0.022222222222222227,
                  0.0, 0.0, 0.0, 0.0),
            0.47: (0.0, 0.0, 0.0036000000000000107, 0.5608285714285716,
                   0.43474025974025954, 0.0008311688311688265, 0.0, 0.0),
            0.93: (0.0, 0.0, 0.0, 0.0, 0.006236363636363623, 0.11675363636363623,
                   0.6023849999999998, 0.27462500000000023),
        }  # fmt: skip
        for x, values in expected.items():
            for got, want in zip(bspline_basis(x, self.knots), values, strict=True):
                self.assertAlmostEqual(got, want, places=14)

    def test_partition_of_unity_and_marsden_identity(self) -> None:
        """``sum B_i = 1`` and ``(x - tau)^3 = sum_i prod_r (t_(i+r) - tau) B_i(x)``.

        Marsden's identity reproduces every cubic exactly, so its two sides
        must agree to rounding for several ``tau``.

        References:
            [EQ-SMOOTH-001], [REF-DE-BOOR-1972].

        """
        t = self.knots
        for x in (0.0, 0.13, 0.2, 0.33, 0.5, 0.77, 0.999, 1.0):
            row = bspline_basis(x, t)
            self.assertAlmostEqual(math.fsum(row), 1.0, places=15)
            for tau in (-0.3, 0.25, 1.4):
                psi = [math.prod(t[i + r] - tau for r in (1, 2, 3)) for i in range(8)]
                lhs = (x - tau) ** 3
                rhs = math.fsum(p * b for p, b in zip(psi, row, strict=True))
                self.assertAlmostEqual(lhs, rhs, places=13)

    def test_domain_and_validation(self) -> None:
        """Points outside the domain and bad knot vectors raise.

        References:
            [EQ-SMOOTH-001], [REF-DE-BOOR-1972].

        """
        with self.assertRaises(ValueError):
            bspline_basis(1.01, self.knots)
        with self.assertRaises(ValueError):
            bspline_basis(0.5, (0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0))
        with self.assertRaises(ValueError):
            uniform_knots(1.0, 1.0)
        knots = uniform_knots(0.0, 1.0, 4, 3)
        self.assertEqual(len(knots), 11)
        self.assertEqual(len(bspline_design_matrix([0.0, 1.0], knots)[0]), 7)

    def test_difference_penalty(self) -> None:
        """``D_2'D_2`` has the second-difference stencil and annihilates lines.

        References:
            [EQ-SMOOTH-002], [REF-EILERS-MARX-1996].

        """
        s = difference_penalty(5, 2)
        self.assertEqual(s[0][:3], (1.0, -2.0, 1.0))
        self.assertEqual(s[2], (1.0, -4.0, 6.0, -4.0, 1.0))
        line = [2.0 + 3.0 * k for k in range(5)]
        for row in s:
            self.assertAlmostEqual(
                math.fsum(a * b for a, b in zip(row, line, strict=True)), 0.0
            )
        with self.assertRaises(ValueError):
            difference_penalty(3, 3)


def mixed_model_reml(
    design: list[list[float]], penalty: Matrix, y: list[float], lam: float
) -> float:
    """Return the exact restricted log-likelihood of the equivalent mixed model.

    ``beta = U0 a + U+ b`` with ``S = U diag(d) U'``, ``b ~ N(0, (phi/lambda)
    diag(d+)^-1)`` and ``y ~ N(X U0 a, phi (I + Z Z'/lambda))``,
    ``Z = X U+ diag(d+)^-1/2``; ``phi`` is profiled out (Patterson-Thompson
    restricted likelihood, dense algebra, independent of the library's
    penalized-least-squares formula).

    References:
        [EQ-SMOOTH-005], [REF-WOOD-2011].

    """
    values, vectors = symmetric_eigen(penalty)
    k = len(values)
    tol = 1e-9 * max(values)
    null = [j for j in range(k) if abs(values[j]) <= tol]
    rng_ = [j for j in range(k) if abs(values[j]) > tol]
    n = len(y)
    f = [
        [math.fsum(row[m] * vectors[m][j] for m in range(k)) for j in null]
        for row in design
    ]
    z = [
        [
            math.fsum(row[m] * vectors[m][j] for m in range(k)) / math.sqrt(values[j])
            for j in rng_
        ]
        for row in design
    ]
    gram = [
        [math.fsum(a * b for a, b in zip(z[i], z[j], strict=True)) for j in range(n)]
        for i in range(n)
    ]
    sigma = [
        [(1.0 if i == j else 0.0) + gram[i][j] / lam for j in range(n)]
        for i in range(n)
    ]
    lower = cholesky(sigma)
    sinv_f = [
        cholesky_solve(lower, tuple(row[c] for row in f)) for c in range(len(null))
    ]
    width = len(null)
    ftsf = [
        [math.fsum(f[i][a] * sinv_f[b][i] for i in range(n)) for b in range(width)]
        for a in range(width)
    ]
    sinv_y = cholesky_solve(lower, tuple(y))
    fty = [math.fsum(f[i][a] * sinv_y[i] for i in range(n)) for a in range(len(null))]
    a_hat = cholesky_solve(cholesky(ftsf), tuple(fty))
    resid = [
        y[i] - math.fsum(f[i][a] * a_hat[a] for a in range(len(null))) for i in range(n)
    ]
    quad = math.fsum(
        r * s for r, s in zip(resid, cholesky_solve(lower, tuple(resid)), strict=True)
    )
    dof = n - len(null)
    phi = quad / dof
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(n))
    return -0.5 * (log_det + log_det_spd(ftsf) + dof * math.log(phi) + dof)


class PSplineTests(unittest.TestCase):
    """P-splines: exact limits, NumPy values, REML, and band coverage.

    References:
        [EQ-SMOOTH-002], [EQ-SMOOTH-003], [EQ-SMOOTH-004], [EQ-SMOOTH-005],
        [EQ-SMOOTH-006], [REF-EILERS-MARX-1996], [REF-WOOD-2011].

    """

    def test_fixed_lambda_matches_numpy(self) -> None:
        """EDF, GCV, a fitted value, and a standard error at ``lambda = 0.1, 10``.

        Offline: ``B = BSpline.design_matrix(x, knots, 3)`` with the knots of
        ``uniform_knots(min x, max x, 10, 3)``, ``S = D'D`` from
        ``numpy.diff(numpy.eye(13), n=2, axis=0)``, ``A = B'B + lambda S``,
        ``beta = numpy.linalg.solve(A, B'y)``, ``edf = tr(A^-1 B'B)``,
        ``V = RSS/(n - edf) A^-1`` (NumPy 2.2.6, SciPy 1.14.1).

        References:
            [EQ-SMOOTH-002], [EQ-SMOOTH-003], [EQ-SMOOTH-004], [EQ-SMOOTH-006].

        """
        cases = {
            0.1: (7.701048410945681, 0.13316050132181625, 0.1036942232022804,
                  0.14963280479209595),
            10.0: (3.571610821864557, 0.16337466050521665, -0.012083172974993759,
                   0.11264522632798779),
        }  # fmt: skip
        for lam, (edf, gcv, mid, se) in cases.items():
            fit = fit_pspline(X, Y, segments=10, smoothing=lam)
            close(self, fit.edf, edf)
            close(self, fit.criterion, gcv)
            band = fit.band([0.5])
            close(self, band.estimate[0], mid)
            close(self, band.standard_error[0], se)
            self.assertEqual(fit.method, "fixed")

    def test_limits_of_the_smoothing_parameter(self) -> None:
        """``lambda = 0`` is regression on the basis; large ``lambda`` is the line.

        With a second-order penalty the limit is the least-squares line, so
        ``fit_simple_linear_regression`` is the oracle; the EDF tends to 2.

        References:
            [EQ-SMOOTH-002], [EQ-SMOOTH-003].

        """
        free = fit_pspline(X, Y, segments=6, smoothing=0.0)
        self.assertAlmostEqual(free.edf, 9.0, places=9)
        stiff = fit_pspline(X, Y, segments=10, smoothing=1e10)
        self.assertAlmostEqual(stiff.edf, 2.0, places=5)
        line = fit_simple_linear_regression(X, Y)
        for x, fitted in zip(X, stiff.fitted, strict=True):
            self.assertAlmostEqual(fitted, line.intercept + line.slope * x, places=5)

    def test_polynomials_in_the_null_space_are_reproduced(self) -> None:
        """A quadratic is fitted exactly for every ``lambda`` with ``d = 3``.

        References:
            [EQ-SMOOTH-002], [REF-EILERS-MARX-1996].

        """
        y = [1.0 - 2.0 * x + 4.0 * x * x for x in X]
        for lam in (0.0, 1.0, 1e6):
            fit = fit_pspline(X, y, segments=8, penalty_order=3, smoothing=lam)
            for got, want in zip(fit.fitted, y, strict=True):
                self.assertAlmostEqual(got, want, places=8)
            self.assertAlmostEqual(fit.predict([0.25])[0], 1.0 - 0.5 + 0.25, places=8)

    def test_reml_matches_the_mixed_model_likelihood(self) -> None:
        """REML differences equal restricted-likelihood differences; argmax agrees.

        References:
            [EQ-SMOOTH-005], [REF-WOOD-2011].

        """
        from probstat.inference._penalized import PenalizedProblem

        knots = uniform_knots(min(X), max(X), 8, 3)
        design = bspline_design_matrix(X, knots)
        penalty = difference_penalty(11, 2)
        problem = PenalizedProblem(
            design, tuple(Y), (1.0,) * 30, (penalty,), (9,), (tuple(range(11)),)
        )
        lams = (0.01, 0.3, 5.0)
        ours = [
            problem.criterion("reml", (lam,), problem.solve((lam,))) for lam in lams
        ]
        theirs = [
            mixed_model_reml([list(r) for r in design], penalty, Y, lam) for lam in lams
        ]
        for i in (1, 2):
            self.assertAlmostEqual(
                ours[i] - ours[0], -(theirs[i] - theirs[0]), places=9
            )
        fit = fit_pspline(X, Y, segments=8, method="reml")
        best = mixed_model_reml([list(r) for r in design], penalty, Y, fit.smoothing)
        for factor in (0.97, 1.03):
            other = mixed_model_reml(
                [list(r) for r in design], penalty, Y, fit.smoothing * factor
            )
            self.assertLess(other, best)

    def test_band_coverage_across_the_function(self) -> None:
        """Seeded simulation: average pointwise coverage is not below nominal.

        Forty replicates of ``sin(2 pi x) + N(0, 0.3^2)`` at 50 points, REML
        P-splines with 12 segments; the band's average coverage across the
        function must be at least ``0.95 - 3 SE`` (Nychka 1988) and below
        ``0.995``. The run gives about 0.966.

        References:
            [EQ-SMOOTH-006], [REF-NYCHKA-1988].

        """
        generator = resolve_stream(20261112).generator()
        xs = [(i + 0.5) / 50 for i in range(50)]
        truth = [math.sin(2 * math.pi * x) for x in xs]
        cover = []
        for _ in range(40):
            ys = [f + generator.gauss(0.0, 0.3) for f in truth]
            band = fit_pspline(xs, ys, segments=12).band(xs)
            inside = [
                lo <= f <= hi
                for lo, hi, f in zip(band.lower, band.upper, truth, strict=True)
            ]
            cover.append(sum(inside) / 50)
        mean = statistics.fmean(cover)
        error = statistics.stdev(cover) / math.sqrt(len(cover))
        self.assertGreaterEqual(mean, 0.95 - 3.0 * error)
        self.assertLess(mean, 0.995)

    def test_validation(self) -> None:
        """Bad inputs raise ``ValueError``.

        References:
            [EQ-SMOOTH-002]. No further equation is evaluated.

        """
        with self.assertRaises(ValueError):
            fit_pspline(X, Y[:-1])
        with self.assertRaises(ValueError):
            fit_pspline(X, Y, method="aic")
        with self.assertRaises(ValueError):
            fit_pspline(X, Y, smoothing=-1.0)
        fit = fit_pspline(X, Y, segments=8, smoothing=1.0)
        with self.assertRaises(ValueError):
            fit.predict([1.5])
        with self.assertRaises(ValueError):
            fit.band([0.5], level=1.0)


class SmoothingSplineTests(unittest.TestCase):
    """Reinsch smoothing spline against SciPy, brute force, and the mixed model.

    References:
        [EQ-SMOOTH-007], [EQ-SMOOTH-008], [REF-REINSCH-1967],
        [REF-HUTCHINSON-DEHOOG-1985].

    """

    def test_fixed_lambda_matches_scipy(self) -> None:
        """Values, EDF, and leverages at ``lambda = 1e-4`` and ``1e-2``.

        Offline: ``spl = scipy.interpolate.make_smoothing_spline(x, y,
        lam=lam)`` (SciPy 1.14.1) for the values, and the brute-force hat
        matrix ``H[:, k] = make_smoothing_spline(x, e_k, lam=lam)(x)`` for
        the EDF ``tr H`` and the leverages ``H_00`` and ``H_15``.

        References:
            [EQ-SMOOTH-007], [EQ-SMOOTH-008], [REF-GREEN-SILVERMAN-1994].

        """
        cases = {
            1e-4: ((0.19597720029067206, 0.6561161267237959, -0.35834967318904853),
                   0.1443613103014122, 9.267892110526203, 0.7135663247056868,
                   0.27464445440508717),
            1e-2: ((0.5218606461953389, 0.49101645414451944, -0.7623404206437728),
                   -0.010331341755530547, 3.6236645086771153, 0.31017794984381925,
                   0.09036165772483536),
        }  # fmt: skip
        for lam, (values, mid, edf, h0, h15) in cases.items():
            fit = fit_smoothing_spline(X, Y, smoothing=lam)
            for index, want in zip((0, 10, 29), values, strict=True):
                close(self, fit.values[index], want, 1e-9)
            close(self, fit.predict([0.5])[0], mid, 1e-9)
            close(self, fit.edf, edf, 1e-9)
            close(self, fit.leverage[0], h0, 1e-9)
            close(self, fit.leverage[15], h15, 1e-9)

    def test_gcv_minimizer_matches_brute_force(self) -> None:
        """``lambda_GCV`` and ``GCV(lambda_GCV)`` from a brute-force minimization.

        Offline: ``GCV(rho) = n ||y - H y||^2 / (n - tr H)^2`` with the
        brute-force ``H`` above at ``lambda = exp(rho)``, minimized by
        ``scipy.optimize.minimize_scalar(gcv, bracket=..., tol=1e-12)`` from
        a 91-point grid on ``[-16, 2]``: ``lambda = 0.001136379894112462``,
        ``GCV = 0.12698618787639473``, ``edf = 5.520655387855897``.

        References:
            [EQ-SMOOTH-004], [REF-CRAVEN-WAHBA-1979].

        """
        fit = fit_smoothing_spline(X, Y, method="gcv")
        self.assertEqual(fit.method, "gcv")
        close(self, fit.criterion, 0.12698618787639473, 1e-10)
        self.assertAlmostEqual(
            math.log(fit.smoothing), math.log(0.001136379894112462), places=4
        )
        self.assertAlmostEqual(fit.edf, 5.520655387855897, places=4)

    def test_reml_matches_the_mixed_model_likelihood(self) -> None:
        """REML of the Reinsch form equals the mixed-model restricted likelihood.

        The penalty is ``K = Q R^-1 Q'``.

        The dense ``K`` is built from ``Q`` and ``R`` independently of the
        banded algorithm.

        References:
            [EQ-SMOOTH-005], [EQ-SMOOTH-007], [REF-WOOD-2011].

        """
        from probstat.inference.splines import _ReinschSystem

        n = len(X)
        h = [X[i + 1] - X[i] for i in range(n - 1)]
        q = [[0.0] * (n - 2) for _ in range(n)]
        r = [[0.0] * (n - 2) for _ in range(n - 2)]
        for j in range(n - 2):
            q[j][j], q[j + 1][j], q[j + 2][j] = (
                1 / h[j],
                -1 / h[j] - 1 / h[j + 1],
                1 / h[j + 1],
            )
            r[j][j] = (h[j] + h[j + 1]) / 3
            if j + 1 < n - 2:
                r[j][j + 1] = r[j + 1][j] = h[j + 1] / 6
        k = matmul(matmul(q, inverse_spd(r)), transpose(q))
        identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        system = _ReinschSystem(tuple(X), tuple(Y), (1.0,) * n)
        lams = (1e-4, 1e-3, 1e-2)
        ours = [system.criterion("reml", lam, system.solve(lam)) for lam in lams]
        theirs = [mixed_model_reml(identity, k, Y, lam) for lam in lams]
        for i in (1, 2):
            self.assertAlmostEqual(
                ours[i] - ours[0], -(theirs[i] - theirs[0]), places=7
            )
        fit = fit_smoothing_spline(X, Y, method="reml")
        best = mixed_model_reml(identity, k, Y, fit.smoothing)
        for factor in (0.95, 1.05):
            self.assertLess(
                mixed_model_reml(identity, k, Y, fit.smoothing * factor), best
            )

    def test_exact_limits(self) -> None:
        """A straight line is reproduced for every ``lambda``; EDF runs from n to 2.

        References:
            [EQ-SMOOTH-003], [EQ-SMOOTH-007], [REF-GREEN-SILVERMAN-1994].

        """
        line = [3.0 - 2.0 * x for x in X]
        for lam in (1e-6, 1.0, 1e6):
            fit = fit_smoothing_spline(X, line, smoothing=lam)
            for got, want in zip(fit.values, line, strict=True):
                self.assertAlmostEqual(got, want, places=9)
            self.assertAlmostEqual(fit.criterion, 0.0, places=15)
            self.assertAlmostEqual(fit.predict([1.5])[0], 0.0, places=8)
        self.assertGreater(fit_smoothing_spline(X, Y, smoothing=1e-12).edf, 29.99)
        self.assertAlmostEqual(
            fit_smoothing_spline(X, Y, smoothing=1e8).edf, 2.0, places=6
        )

    def test_band_and_validation(self) -> None:
        """The band variance is ``phi H_ii``; ties and short inputs raise.

        References:
            [EQ-SMOOTH-006], [REF-WAHBA-1983].

        """
        fit = fit_smoothing_spline(X, Y, smoothing=1e-3)
        band = fit.band(0.9)
        z = statistics.NormalDist().inv_cdf(0.95)
        self.assertAlmostEqual(
            band.standard_error[3], math.sqrt(fit.scale * fit.leverage[3])
        )
        self.assertAlmostEqual(
            band.upper[3] - band.estimate[3], z * band.standard_error[3]
        )
        shuffled = fit_smoothing_spline(
            list(reversed(X)), list(reversed(Y)), smoothing=1e-3
        )
        self.assertEqual(shuffled.values, fit.values)
        with self.assertRaises(ValueError):
            fit_smoothing_spline([0.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            fit_smoothing_spline([0.0, 1.0], [1.0, 2.0])


class LoessTests(unittest.TestCase):
    """LOESS against closed forms and an explicit weighted least-squares fit.

    References:
        [EQ-SMOOTH-009], [EQ-SMOOTH-010], [REF-CLEVELAND-1979].

    """

    def test_reproduces_polynomials(self) -> None:
        """Degree 1 reproduces lines and degree 2 quadratics, for any span.

        References:
            [EQ-SMOOTH-009], [REF-CLEVELAND-1979].

        """
        line = [1.0 + 2.0 * x for x in X]
        quad = [1.0 + 2.0 * x - 5.0 * x * x for x in X]
        for span in (0.3, 0.75, 1.0):
            fit = loess(X, line, span=span, degree=1)
            for got, want in zip(fit.fitted, line, strict=True):
                self.assertAlmostEqual(got, want, places=11)
            fit2 = loess(X, quad, span=span, degree=2)
            for got, want in zip(fit2.fitted, quad, strict=True):
                self.assertAlmostEqual(got, want, places=10)

    def test_matches_explicit_weighted_least_squares(self) -> None:
        """The fit at one point equals WLS with hand-computed tricube weights.

        References:
            [EQ-SMOOTH-009], [REF-CLEVELAND-1979].

        """
        fit = loess(X, Y, span=0.4, degree=1)
        self.assertEqual(fit.neighbours, 12)
        x0 = X[7]
        h = sorted(abs(x - x0) for x in X)[11]
        weights = [(1 - min(abs(x - x0) / h, 1.0) ** 3) ** 3 for x in X]
        keep = [i for i, w in enumerate(weights) if w > 0.0]
        result = fit_linear_model(
            [[1.0, X[i] - x0] for i in keep], [Y[i] for i in keep],
            weights=[weights[i] for i in keep],
        )  # fmt: skip
        self.assertAlmostEqual(fit.fitted[7], result.coefficients[0], places=12)
        self.assertAlmostEqual(fit.predict([x0])[0], fit.fitted[7], places=14)

    def test_robustness_iterations_remove_an_outlier(self) -> None:
        """With one gross outlier on a line, two bisquare steps recover the line.

        After the first robust step the outlier lies beyond ``6 median|e|``
        and gets weight zero, so the local fits use exact data only.

        References:
            [EQ-SMOOTH-010], [REF-CLEVELAND-1979].

        """
        xs = [i / 49 for i in range(50)]
        ys = [2.0 + 3.0 * x for x in xs]
        ys[20] += 10.0
        plain = loess(xs, ys, span=0.5)
        robust = loess(xs, ys, span=0.5, robustness_iterations=2)
        self.assertGreater(
            max(abs(f - (2 + 3 * x)) for f, x in zip(plain.fitted, xs, strict=True)),
            0.5,
        )
        for f, x in zip(robust.fitted, xs, strict=True):
            self.assertAlmostEqual(f, 2.0 + 3.0 * x, places=10)
        self.assertEqual(robust.robustness_weights[20], 0.0)

    def test_validation(self) -> None:
        """Invalid spans and degrees raise.

        References:
            [EQ-SMOOTH-009]. No further equation is evaluated.

        """
        for kwargs in ({"span": 0.0}, {"span": 1.5}, {"degree": 3}, {"span": 0.02}):
            with self.assertRaises(ValueError):
                loess(X, Y, **kwargs)  # type: ignore[arg-type]


class IsotonicTests(unittest.TestCase):
    """PAVA against the min-max formula and hand cases.

    References:
        [EQ-SMOOTH-011], [REF-AYER-ETAL-1955].

    """

    @staticmethod
    def minmax(y: list[float], w: list[float]) -> list[float]:
        """Return ``max_(j<=i) min_(k>=i) Av_w(y_j..y_k)`` by enumeration.

        References:
            [EQ-SMOOTH-011], [REF-DE-LEEUW-HORNIK-MAIR-2009].

        """
        n = len(y)

        def av(j: int, k: int) -> float:
            """Weighted mean of a block.

            References:
                [EQ-SMOOTH-011]. No further equation is evaluated.

            """
            return math.fsum(w[m] * y[m] for m in range(j, k + 1)) / math.fsum(
                w[j : k + 1]
            )

        return [
            max(min(av(j, k) for k in range(i, n)) for j in range(i + 1))
            for i in range(n)
        ]

    def test_hand_case(self) -> None:
        """``(1, 3, 2, 4, 3.5, 5)`` pools to ``(1, 2.5, 2.5, 3.75, 3.75, 5)``.

        References:
            [EQ-SMOOTH-011], [REF-AYER-ETAL-1955].

        """
        fit = isotonic_regression([1, 3, 2, 4, 3.5, 5])
        self.assertEqual(fit.fitted, (1.0, 2.5, 2.5, 3.75, 3.75, 5.0))
        self.assertEqual(
            [(b.start, b.stop) for b in fit.blocks], [(0, 1), (1, 3), (3, 5), (5, 6)]
        )
        self.assertAlmostEqual(fit.weighted_sse, 0.5 + 0.125)

    def test_matches_minmax_formula_on_random_cases(self) -> None:
        """Two hundred seeded weighted cases agree with the min-max formula.

        References:
            [EQ-SMOOTH-011], [REF-DE-LEEUW-HORNIK-MAIR-2009].

        """
        generator = resolve_stream(102).generator()
        for _ in range(200):
            n = generator.randrange(1, 9)
            y = [round(generator.uniform(-2, 2), 2) for _ in range(n)]
            w = [generator.choice((0.5, 1.0, 2.0, 3.0)) for _ in range(n)]
            got = isotonic_regression(y, weights=w).fitted
            for a, b in zip(got, self.minmax(y, w), strict=True):
                self.assertAlmostEqual(a, b, places=12)
            down = isotonic_regression(y, weights=w, increasing=False).fitted
            for a, b in zip(down, self.minmax([-v for v in y], w), strict=True):
                self.assertAlmostEqual(a, -b, places=12)

    def test_fit_is_monotone_and_optimal_among_monotone_grids(self) -> None:
        """No monotone sequence on a coarse grid has a smaller weighted SSE.

        References:
            [EQ-SMOOTH-011]. No further equation is evaluated.

        """
        y = [0.4, -0.2, 0.9, 0.1]
        fit = isotonic_regression(y)
        self.assertTrue(
            all(a <= b for a, b in zip(fit.fitted, fit.fitted[1:], strict=False))
        )
        levels = [i / 10 for i in range(-5, 11)]
        for seq in itertools.combinations_with_replacement(levels, 4):
            sse = math.fsum((a - b) ** 2 for a, b in zip(seq, y, strict=True))
            self.assertGreaterEqual(sse, fit.weighted_sse - 1e-12)
        with self.assertRaises(ValueError):
            isotonic_regression([1.0, 2.0], weights=[1.0, 0.0])


class KernelRegressionTests(unittest.TestCase):
    """Nadaraya-Watson and local linear against explicit formulas.

    References:
        [EQ-SMOOTH-012], [EQ-SMOOTH-013], [REF-NADARAYA-1964],
        [REF-FAN-GIJBELS-1996].

    """

    def test_nadaraya_watson_is_a_weighted_mean(self) -> None:
        """``m(x0) = sum K_i y_i / sum K_i`` with Gaussian weights.

        References:
            [EQ-SMOOTH-012], [REF-NADARAYA-1964].

        """
        fit = kernel_regression(X, Y, bandwidth=0.08, degree=0)
        weights = [math.exp(-0.5 * ((x - 0.31) / 0.08) ** 2) for x in X]
        expected = math.fsum(
            w * y for w, y in zip(weights, Y, strict=True)
        ) / math.fsum(weights)
        self.assertAlmostEqual(fit.predict([0.31])[0], expected, places=13)

    def test_local_linear_reproduces_lines(self) -> None:
        """Local-linear fits of a line are exact for any bandwidth and kernel.

        References:
            [EQ-SMOOTH-012], [REF-FAN-GIJBELS-1996].

        """
        line = [0.5 - 1.5 * x for x in X]
        for kernel in ("gaussian", "epanechnikov"):
            for h in (0.07, 0.3, 2.0):
                fit = kernel_regression(X, line, bandwidth=h, kernel=kernel)
                for got, want in zip(fit.fitted, line, strict=True):
                    self.assertAlmostEqual(got, want, places=11)

    def test_cross_validation_equals_refitting_without_each_point(self) -> None:
        """``CV(h)`` equals the mean squared error of explicit leave-one-out refits.

        References:
            [EQ-SMOOTH-013], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009].

        """
        for degree in (0, 1):
            fit = kernel_regression(X, Y, degree=degree, candidates=[0.03, 0.05, 0.09])
            for h, score in fit.cv_curve:
                errors = []
                for i in range(len(X)):
                    xs = X[:i] + X[i + 1 :]
                    ys = Y[:i] + Y[i + 1 :]
                    refit = kernel_regression(xs, ys, bandwidth=h, degree=degree)
                    errors.append((Y[i] - refit.predict([X[i]])[0]) ** 2)
                self.assertAlmostEqual(
                    score, math.fsum(errors) / len(errors), places=12
                )
            self.assertEqual(fit.cv_score, min(score for _, score in fit.cv_curve))

    def test_default_grid_and_validation(self) -> None:
        """The default grid has 25 candidates; bad options raise.

        References:
            [EQ-SMOOTH-013], [REF-SILVERMAN-1986].

        """
        fit = kernel_regression(X, Y)
        self.assertEqual(len(fit.cv_curve), 25)
        self.assertIn(fit.bandwidth, [h for h, _ in fit.cv_curve])
        for kwargs in ({"degree": 2}, {"kernel": "box"}, {"bandwidth": 0.0}):
            with self.assertRaises(ValueError):
                kernel_regression(X, Y, **kwargs)  # type: ignore[arg-type]


class AdditiveModelTests(unittest.TestCase):
    """Joint fit, backfitting, and partial dependence.

    References:
        [EQ-GAM-001], [EQ-GAM-002], [EQ-GAM-003], [REF-BUJA-HASTIE-TIBSHIRANI-1989].

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Draw one seeded additive data set.

        References:
            [EQ-GAM-001]. No further equation is evaluated.

        """
        generator = resolve_stream(7).generator()
        n = 80
        cls.a = [generator.uniform(0.0, 1.0) for _ in range(n)]
        cls.b = [generator.uniform(-1.0, 1.0) for _ in range(n)]
        cls.y = [
            math.sin(2 * math.pi * u) + v * v + generator.gauss(0.0, 0.2)
            for u, v in zip(cls.a, cls.b, strict=True)
        ]

    def test_backfitting_converges_to_the_joint_solution(self) -> None:
        """Backfitting with the joint fit's ``lambda_j`` reproduces its fit.

        References:
            [EQ-GAM-002], [REF-BUJA-HASTIE-TIBSHIRANI-1989].

        """
        data = {"a": self.a, "b": self.b}
        joint = fit_additive_model(data, self.y, segments=8)
        lams = {term.name: term.smoothing for term in joint.terms}
        back = backfit_additive_model(data, self.y, smoothing=lams, segments=8)
        for p, q in zip(back.fitted, joint.fitted, strict=True):
            self.assertAlmostEqual(p, q, places=8)
        self.assertAlmostEqual(back.intercept, joint.intercept, places=10)
        grid = [0.1, 0.4, 0.8]
        for p, q in zip(
            back.term("a").evaluate(grid), joint.term("a").evaluate(grid), strict=True
        ):
            self.assertAlmostEqual(p, q, places=8)
        self.assertGreater(back.iterations, 1)

    def test_terms_are_centred_and_intercept_is_the_mean(self) -> None:
        """``sum_i f_j(x_ij) = 0`` and ``alpha = mean(y)`` for unit weights.

        References:
            [EQ-GAM-001], [REF-WOOD-2017].

        """
        fit = fit_additive_model(
            {"a": self.a, "b": self.b}, self.y, segments=8, method="gcv"
        )
        self.assertAlmostEqual(
            math.fsum(fit.term("a").evaluate(self.a)), 0.0, places=10
        )
        self.assertAlmostEqual(
            math.fsum(fit.term("b").evaluate(self.b)), 0.0, places=10
        )
        self.assertAlmostEqual(fit.intercept, statistics.fmean(self.y), places=12)
        self.assertAlmostEqual(
            fit.edf, 1.0 + math.fsum(term.edf for term in fit.terms), places=10
        )

    def test_single_term_model_equals_the_pspline(self) -> None:
        """One term with an intercept spans the same space with the same penalty.

        References:
            [EQ-GAM-001], [EQ-SMOOTH-005], [REF-EILERS-MARX-1996].

        """
        gam = fit_additive_model({"x": X}, Y, segments=10)
        spline = fit_pspline(X, Y, segments=10)
        self.assertAlmostEqual(
            math.log(gam.terms[0].smoothing), math.log(spline.smoothing), places=5
        )
        for p, q in zip(gam.fitted, spline.fitted, strict=True):
            self.assertAlmostEqual(p, q, places=6)
        self.assertAlmostEqual(gam.edf, spline.edf, places=5)

    def test_partial_dependence_of_an_additive_model(self) -> None:
        """``PD_j(v) = alpha + f_j(v)`` exactly for centred additive terms.

        References:
            [EQ-GAM-003], [REF-FRIEDMAN-2001].

        """
        data = {"a": self.a, "b": self.b}
        fit = fit_additive_model(data, self.y, segments=8, smoothing=0.5)
        grid = [0.05, 0.5, 0.95]
        curve = partial_dependence(fit.predict, data, "a", grid)
        for got, f in zip(curve.values, fit.term("a").evaluate(grid), strict=True):
            self.assertAlmostEqual(got, fit.intercept + f, places=12)
        with self.assertRaises(KeyError):
            partial_dependence(fit.predict, data, "c", grid)

    def test_exact_additive_lines_are_reproduced(self) -> None:
        """Noise-free ``1 + 2a - 3b`` is fitted exactly for any fixed ``lambda``.

        References:
            [EQ-GAM-001], [EQ-SMOOTH-002].

        """
        y = [1.0 + 2.0 * u - 3.0 * v for u, v in zip(self.a, self.b, strict=True)]
        fit = fit_additive_model(
            {"a": self.a, "b": self.b}, y, segments=6, smoothing=10.0
        )
        for got, want in zip(fit.fitted, y, strict=True):
            self.assertAlmostEqual(got, want, places=9)
        prediction = fit.predict({"a": [0.5], "b": [0.25]})[0]
        self.assertAlmostEqual(prediction, 1.0 + 1.0 - 0.75, places=9)
        band = fit.term("a").band([0.5])
        self.assertGreaterEqual(band.standard_error[0], 0.0)

    def test_validation(self) -> None:
        """Bad inputs raise.

        References:
            [EQ-GAM-001]. No further equation is evaluated.

        """
        with self.assertRaises(ValueError):
            fit_additive_model({}, self.y)
        with self.assertRaises(ValueError):
            fit_additive_model({"a": self.a}, self.y, penalty_order=0)
        with self.assertRaises(ValueError):
            fit_additive_model({"a": [1.0] * 80}, self.y)
        with self.assertRaises(ValueError):
            backfit_additive_model({"a": self.a}, self.y, smoothing=0.0)
        fit = fit_additive_model({"a": self.a}, self.y, segments=6, smoothing=1.0)
        with self.assertRaises(ValueError):
            fit.predict({"a": [2.0]})


if __name__ == "__main__":
    unittest.main()
