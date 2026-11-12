"""PCA, factor analysis, varimax, discriminant analysis, and k-means (ADR-0102).

Oracles: closed forms (the PCA of a covariance with known eigenvectors,
exact recovery of a population factor structure, Jöreskog's identity
between the concentrated and the full ML discrepancy, varimax of a perfect
simple structure, Gaussian posteriors through
``multivariate.MultivariateNormal``, hand-computed silhouettes in exact
rational arithmetic), exhaustive enumeration of every partition for
k-means, NumPy 2.2.6 eigenvalues computed offline
(``numpy.linalg.eigh(numpy.cov(D.T))``), and a seeded simulation of the LDA
error rate against the Bayes error judged by its binomial standard error.

References:
    [REF-HOTELLING-1933], [REF-JOLLIFFE-2002], [REF-JORESKOG-1967],
    [REF-KAISER-1958], [REF-FISHER-1936], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009],
    [REF-LLOYD-1982], [REF-ROUSSEEUW-1987].

"""

from __future__ import annotations

import importlib.util
import itertools
import math
import unittest
from fractions import Fraction
from statistics import NormalDist

from probstat.foundations import resolve_stream
from probstat.inference import (
    factor_analysis,
    fit_linear_discriminant,
    fit_quadratic_discriminant,
    kmeans,
    principal_components,
    silhouette_samples,
    silhouette_score,
    varimax,
)
from probstat.linalg import inverse_spd, log_det_spd, use_backend
from probstat.multivariate import MultivariateNormal

DATA = [
    [2.5, 2.4, 1.2],
    [0.5, 0.7, 0.3],
    [2.2, 2.9, 1.1],
    [1.9, 2.2, 0.4],
    [3.1, 3.0, 1.9],
    [2.3, 2.7, 0.8],
    [2.0, 1.6, 1.4],
    [1.0, 1.1, 0.2],
]


class PrincipalComponentTests(unittest.TestCase):
    """PCA against closed forms and NumPy.

    References:
        [EQ-MVA-001], [REF-JOLLIFFE-2002].

    """

    def test_known_covariance(self) -> None:
        """Points on the diagonals have eigenvalues ``4s^2/3`` and ``4t^2/3``.

        References:
            [EQ-MVA-001], [REF-HOTELLING-1933].

        """
        s, t = 2.0, 0.5
        rows = [[s, s], [-s, -s], [t, -t], [-t, t]]
        pca = principal_components(rows)
        self.assertAlmostEqual(pca.variances[0], 4 * s * s / 3, places=13)
        self.assertAlmostEqual(pca.variances[1], 4 * t * t / 3, places=13)
        root = 1 / math.sqrt(2)
        for got, want in zip(pca.axes[0], (root, root), strict=True):
            self.assertAlmostEqual(got, want, places=13)
        self.assertAlmostEqual(abs(pca.axes[1][0]), root, places=13)
        self.assertAlmostEqual(pca.axes[1][0], -pca.axes[1][1], places=13)
        self.assertAlmostEqual(pca.explained_variance_ratio[0], 16 / 17, places=13)

    def test_matches_numpy(self) -> None:
        """Eigenvalues and axes match ``numpy.linalg.eigh(numpy.cov(D.T))``.

        Offline with NumPy 2.2.6; axes signed so the largest entry is positive.

        References:
            [EQ-MVA-001], [EQ-LINALG-005].

        """
        values = (1.588615372851066, 0.16445034166473124, 0.021577142627060127)
        axes = (
            (0.6513519736810967, 0.6477842075077948, 0.39511546019516597),
            (0.11787117858745064, -0.6007889134530642, 0.790670011275432),
            (0.7495645346733961, -0.46843174740338034, -0.4676801325526564),
        )
        pca = principal_components(DATA)
        for got, want in zip(pca.variances, values, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for axis, want in zip(pca.axes, axes, strict=True):
            for a, b in zip(axis, want, strict=True):
                self.assertAlmostEqual(a, b, places=10)
        corr = principal_components(DATA, standardize=True)
        for got, want in zip(
            corr.variances,
            (2.614128903134996, 0.35157582207275945, 0.034295274792244965),
            strict=True,
        ):
            self.assertAlmostEqual(got, want, places=12)

    def test_scores_are_uncorrelated_and_reconstruct_the_data(self) -> None:
        """Score variances are the eigenvalues, covariances vanish, rows return.

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        pca = principal_components(DATA, standardize=True)
        n = len(DATA)
        for a in range(3):
            for b in range(3):
                cov = math.fsum(r[a] * r[b] for r in pca.scores) / (n - 1)
                want = pca.variances[a] if a == b else 0.0
                self.assertAlmostEqual(cov, want, places=12)
        for row, back in zip(DATA, pca.reconstruct(pca.scores), strict=True):
            for a, b in zip(row, back, strict=True):
                self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(pca.transform(DATA[:2]), pca.scores[:2])
        loadings = pca.loadings
        self.assertAlmostEqual(
            math.fsum(loadings[0][c] ** 2 for c in range(3)), 1.0, places=12
        )

    def test_retained_components_and_validation(self) -> None:
        """Retaining two components keeps their ratios; bad inputs raise.

        References:
            [EQ-MVA-001]. No further equation is evaluated.

        """
        two = principal_components(DATA, components=2)
        self.assertEqual(len(two.axes), 2)
        self.assertEqual(len(two.scores[0]), 2)
        self.assertLess(two.cumulative_variance_ratio[-1], 1.0)
        with self.assertRaises(ValueError):
            principal_components(DATA, components=4)
        with self.assertRaises(ValueError):
            principal_components([[1.0, 2.0], [1.0, 3.0]], standardize=True)
        with self.assertRaises(ValueError):
            principal_components([[1.0, 2.0], [1.0]])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_dense_backend_gives_the_same_components(self) -> None:
        """The NumPy backend's ``eigh`` gives the same variances and axes.

        References:
            [EQ-MVA-001], [EQ-LINALG-005].

        """
        reference = principal_components(DATA)
        with use_backend("numpy"):
            other = principal_components(DATA)
        for a, b in zip(reference.variances, other.variances, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        for axis_a, axis_b in zip(reference.axes, other.axes, strict=True):
            for a, b in zip(axis_a, axis_b, strict=True):
                self.assertAlmostEqual(a, b, places=10)


LOADINGS = [[0.9, 0.0], [0.8, 0.1], [0.7, 0.0], [0.0, 0.8], [0.1, 0.7], [0.0, 0.6]]


def population_correlation() -> list[list[float]]:
    """Return ``Lambda Lambda' + Psi`` with unit diagonal.

    References:
        [EQ-MVA-002], [REF-LAWLEY-MAXWELL-1971].

    """
    p = len(LOADINGS)
    return [
        [
            math.fsum(LOADINGS[i][k] * LOADINGS[j][k] for k in range(2))
            + (1.0 - math.fsum(v * v for v in LOADINGS[i]) if i == j else 0.0)
            for j in range(p)
        ]
        for i in range(p)
    ]


def simulated_rows(seed: int, n: int) -> list[list[float]]:
    """Draw rows from the two-factor model of :data:`LOADINGS`.

    References:
        [EQ-MVA-002], [REF-LAWLEY-MAXWELL-1971].

    """
    generator = resolve_stream(seed).generator()
    rows = []
    for _ in range(n):
        f = [generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)]
        rows.append(
            [
                math.fsum(lam[k] * f[k] for k in range(2))
                + generator.gauss(0.0, math.sqrt(1.0 - math.fsum(v * v for v in lam)))
                for lam in LOADINGS
            ]
        )
    return rows


class FactorAnalysisTests(unittest.TestCase):
    """ML factor analysis against closed forms and identities.

    References:
        [EQ-MVA-002], [EQ-MVA-004], [REF-JORESKOG-1967].

    """

    def test_population_structure_is_recovered_exactly(self) -> None:
        """Fitting ``R = Lambda Lambda' + Psi`` itself returns ``Psi`` and ``F = 0``.

        References:
            [EQ-MVA-002], [REF-LAWLEY-MAXWELL-1971].

        """
        fit = factor_analysis(
            covariance=population_correlation(), factors=2, observations=200
        )
        self.assertTrue(fit.converged)
        for got, lam in zip(fit.uniquenesses, LOADINGS, strict=True):
            self.assertAlmostEqual(got, 1.0 - math.fsum(v * v for v in lam), places=7)
        self.assertLess(fit.objective, 1e-12)
        self.assertAlmostEqual(fit.p_value, 1.0, places=9)
        self.assertEqual(fit.degrees_of_freedom, 4.0)
        fitted = [
            [math.fsum(a * b for a, b in zip(r, s, strict=True)) for s in fit.loadings]
            for r in fit.loadings
        ]
        target = population_correlation()
        for i in range(6):
            for j in range(6):
                if i != j:
                    self.assertAlmostEqual(fitted[i][j], target[i][j], places=7)

    def test_joreskog_identity_and_stationarity_on_data(self) -> None:
        """``F`` equals ``log|S| - log|R| + tr(S^-1 R) - p`` at the optimum.

        The full discrepancy with ``S = Lambda Lambda' + Psi`` is computed
        independently of the eigenvalue form; at an interior optimum the
        fitted diagonal equals one (Lawley's equation). The statistic and
        p-value follow EQ-MVA-004.

        References:
            [EQ-MVA-002], [EQ-MVA-004], [REF-JORESKOG-1967].

        """
        rows = simulated_rows(5, 300)
        fit = factor_analysis(rows, factors=2)
        self.assertTrue(fit.converged)
        n, p = len(rows), 6
        means = [math.fsum(r[j] for r in rows) / n for j in range(p)]
        cov = [
            [
                math.fsum((r[i] - means[i]) * (r[j] - means[j]) for r in rows) / (n - 1)
                for j in range(p)
            ]
            for i in range(p)
        ]
        corr = [
            [cov[i][j] / math.sqrt(cov[i][i] * cov[j][j]) for j in range(p)]
            for i in range(p)
        ]
        sigma = [
            [
                math.fsum(
                    a * b for a, b in zip(fit.loadings[i], fit.loadings[j], strict=True)
                )
                + (fit.uniquenesses[i] if i == j else 0.0)
                for j in range(p)
            ]
            for i in range(p)
        ]
        inverse = inverse_spd(sigma)
        trace = math.fsum(
            inverse[i][k] * corr[k][i] for i in range(p) for k in range(p)
        )
        full = log_det_spd(sigma) - log_det_spd(corr) + trace - p
        self.assertAlmostEqual(fit.objective, full, places=9)
        for i in range(p):
            self.assertAlmostEqual(sigma[i][i], 1.0, places=6)
        bartlett = (n - 1 - (2 * p + 5) / 6 - 4 / 3) * fit.objective
        self.assertAlmostEqual(fit.statistic, bartlett, places=12)
        self.assertGreater(fit.p_value, 0.0)
        self.assertLessEqual(fit.p_value, 1.0)
        for c in range(2):
            self.assertGreater(math.fsum(row[c] for row in fit.loadings), 0.0)

    def test_rotation_leaves_the_fit_unchanged(self) -> None:
        """Varimax and unrotated loadings give the same ``Lambda Lambda'``.

        References:
            [EQ-MVA-003], [REF-KAISER-1958].

        """
        rows = simulated_rows(11, 200)
        rotated = factor_analysis(rows, factors=2)
        plain = factor_analysis(rows, factors=2, rotation=None)
        for i in range(6):
            for j in range(6):
                a = math.fsum(
                    x * y
                    for x, y in zip(
                        rotated.loadings[i], rotated.loadings[j], strict=True
                    )
                )
                b = math.fsum(
                    x * y
                    for x, y in zip(plain.loadings[i], plain.loadings[j], strict=True)
                )
                self.assertAlmostEqual(a, b, places=6)
        with self.assertRaises(ValueError):
            factor_analysis(rows, factors=4)
        with self.assertRaises(ValueError):
            factor_analysis(factors=1)


class VarimaxTests(unittest.TestCase):
    """Varimax against simple structure and rotation invariance.

    References:
        [EQ-MVA-003], [REF-KAISER-1958].

    """

    simple = [[0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [0.0, 0.8], [0.0, 0.7], [0.0, 0.6]]

    @staticmethod
    def rotate(rows: list[list[float]], angle: float) -> list[list[float]]:
        """Rotate the rows of a two-column matrix.

        References:
            [EQ-MVA-003]. No further equation is evaluated.

        """
        c, s = math.cos(angle), math.sin(angle)
        return [[x * c - y * s, x * s + y * c] for x, y in rows]

    def test_recovers_perfect_simple_structure(self) -> None:
        """A rotated simple structure is rotated back, up to column sign and order.

        References:
            [EQ-MVA-003], [REF-KAISER-1958].

        """
        for normalize in (True, False):
            for angle in (0.3, -0.7, 1.2):
                result = varimax(self.rotate(self.simple, angle), normalize=normalize)
                columns = [[abs(row[c]) for row in result.loadings] for c in range(2)]
                columns.sort(key=lambda col: -col[0])
                for c, col in enumerate(columns):
                    for got, row in zip(col, self.simple, strict=True):
                        self.assertAlmostEqual(got, row[c], places=9)

    def test_rotation_invariance_and_orthogonality(self) -> None:
        """Varimax of ``L`` and of ``L T`` reach the same criterion; ``T'T = I``.

        References:
            [EQ-MVA-003], [REF-KAISER-1958].

        """
        base = varimax(LOADINGS)
        turned = varimax(self.rotate(LOADINGS, 0.9))
        self.assertAlmostEqual(base.criterion, turned.criterion, places=10)
        t = turned.rotation
        for i in range(2):
            for j in range(2):
                dot = math.fsum(t[k][i] * t[k][j] for k in range(2))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=12)
        start = self.rotate(LOADINGS, 0.9)
        product = [
            [math.fsum(r[k] * t[k][j] for k in range(2)) for j in range(2)]
            for r in start
        ]
        for a, b in zip(product, turned.loadings, strict=True):
            for x, y in zip(a, b, strict=True):
                self.assertAlmostEqual(x, y, places=12)


def two_class_data() -> tuple[list[list[float]], list[int]]:
    """Return two classes with exact means and an exact pooled covariance.

    Class 0 has mean ``(0, 0)`` and class 1 mean ``(2, 1)``; each class is
    its mean plus the four points ``(+-1, +-0.5)``, so the pooled covariance
    (divisor ``n - K = 6``) is ``diag(4/6 * 2, 1/6 * 2)``.

    References:
        [EQ-MVA-005]. No further equation is evaluated.

    """
    offsets = [(1.0, 0.5), (1.0, -0.5), (-1.0, 0.5), (-1.0, -0.5)]
    rows = [[dx, dy] for dx, dy in offsets] + [
        [2.0 + dx, 1.0 + dy] for dx, dy in offsets
    ]
    return rows, [0, 0, 0, 0, 1, 1, 1, 1]


class DiscriminantTests(unittest.TestCase):
    """LDA and QDA against Gaussian posteriors and the Bayes error.

    References:
        [EQ-MVA-005], [EQ-MVA-006], [EQ-MVA-007], [REF-FISHER-1936].

    """

    def test_lda_posterior_is_the_gaussian_posterior(self) -> None:
        """``P(k | x)`` equals the normalized ``pi_k N(x; mu_k, Sigma)``.

        References:
            [EQ-MVA-005], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009].

        """
        rows, labels = two_class_data()
        lda = fit_linear_discriminant(rows, labels, priors={0: 0.3, 1: 0.7})
        self.assertAlmostEqual(lda.covariance[0][0], 8 / 6, places=14)
        self.assertAlmostEqual(lda.covariance[1][1], 2 / 6, places=14)
        self.assertAlmostEqual(lda.covariance[0][1], 0.0, places=14)
        laws = [MultivariateNormal(mu, lda.covariance) for mu in lda.means]
        for point in ([0.4, 0.2], [1.5, -0.3], [3.0, 2.0]):
            joint = [
                math.log(pi) + law.log_pdf(point)
                for pi, law in zip((0.3, 0.7), laws, strict=True)
            ]
            top = max(joint)
            weights = [math.exp(v - top) for v in joint]
            for got, w in zip(lda.predict_proba([point])[0], weights, strict=True):
                self.assertAlmostEqual(got, w / sum(weights), places=12)
        direction = lda.directions[0]
        expected = [2.0 / (8 / 6), 1.0 / (2 / 6)]
        ratio = direction[0] / expected[0]
        self.assertAlmostEqual(direction[1], ratio * expected[1], places=12)
        quad = math.fsum(
            direction[i] * lda.covariance[i][j] * direction[j]
            for i in range(2)
            for j in range(2)
        )
        self.assertAlmostEqual(quad, 1.0, places=12)

    def test_qda_posterior_is_the_gaussian_posterior(self) -> None:
        """QDA scores equal ``log pi_k + log N(x; mu_k, Sigma_k)`` up to a constant.

        References:
            [EQ-MVA-006], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009].

        """
        generator = resolve_stream(42).generator()
        rows = [[generator.gauss(0, 1), generator.gauss(0, 2)] for _ in range(30)]
        rows += [[generator.gauss(2, 0.5), generator.gauss(1, 1)] for _ in range(20)]
        labels = ["a"] * 30 + ["b"] * 20
        qda = fit_quadratic_discriminant(rows, labels)
        self.assertEqual(qda.priors, (0.6, 0.4))
        laws = [
            MultivariateNormal(mu, s)
            for mu, s in zip(qda.means, qda.covariances, strict=True)
        ]
        for point in ([0.0, 0.0], [2.0, 1.0], [1.0, -3.0]):
            scores = qda.decision_function([point])[0]
            exact = [
                math.log(pi) + law.log_pdf(point)
                for pi, law in zip(qda.priors, laws, strict=True)
            ]
            self.assertAlmostEqual(
                scores[0] - scores[1], exact[0] - exact[1], places=11
            )

    def test_lda_error_rate_is_close_to_the_bayes_error(self) -> None:
        """Seeded: test error within four binomial standard errors of ``Phi(-Delta/2)``.

        Two unit-covariance classes 2.5 apart with equal priors; 2000
        training and 4000 test points per class, so the estimation excess is
        far below the tolerance.

        References:
            [EQ-MVA-005], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009].

        """
        generator = resolve_stream(2026).generator()

        def draw(count: int, shift: float) -> list[list[float]]:
            """Draw unit-covariance normal points around ``(shift, 0)``.

            References:
                [EQ-MVA-005]. No further equation is evaluated.

            """
            return [
                [generator.gauss(shift, 1), generator.gauss(0, 1)] for _ in range(count)
            ]

        train = draw(2000, 0.0) + draw(2000, 2.5)
        lda = fit_linear_discriminant(train, [0] * 2000 + [1] * 2000)
        test = draw(4000, 0.0) + draw(4000, 2.5)
        truth = [0] * 4000 + [1] * 4000
        errors = sum(p != t for p, t in zip(lda.predict(test), truth, strict=True))
        rate = errors / len(truth)
        bayes = NormalDist().cdf(-1.25)
        se = math.sqrt(bayes * (1 - bayes) / len(truth))
        self.assertLess(abs(rate - bayes), 4 * se)

    def test_three_class_canonical_directions(self) -> None:
        """Directions are ``W``-orthonormal and their ratios are ``a'Ba``.

        References:
            [EQ-MVA-007], [REF-FISHER-1936].

        """
        generator = resolve_stream(9).generator()
        centres = [(0.0, 0.0, 0.0), (3.0, 0.5, 0.0), (0.0, 2.0, 1.0)]
        rows, labels = [], []
        for label, centre in enumerate(centres):
            for _ in range(25):
                rows.append([c + generator.gauss(0, 1) for c in centre])
                labels.append(label)
        lda = fit_linear_discriminant(rows, labels)
        self.assertEqual(len(lda.directions), 2)
        w = lda.covariance
        m = [
            math.fsum(pi * mu[j] for pi, mu in zip(lda.priors, lda.means, strict=True))
            for j in range(3)
        ]
        b = [
            [
                math.fsum(
                    pi * (mu[i] - m[i]) * (mu[j] - m[j])
                    for pi, mu in zip(lda.priors, lda.means, strict=True)
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        for r, a in enumerate(lda.directions):
            for s, c in enumerate(lda.directions):
                inner = math.fsum(
                    a[i] * w[i][j] * c[j] for i in range(3) for j in range(3)
                )
                self.assertAlmostEqual(inner, 1.0 if r == s else 0.0, places=10)
            ratio = math.fsum(a[i] * b[i][j] * a[j] for i in range(3) for j in range(3))
            self.assertAlmostEqual(ratio, lda.discriminant_ratios[r], places=10)
        self.assertGreaterEqual(lda.discriminant_ratios[0], lda.discriminant_ratios[1])
        projected = lda.transform(rows[:1])
        self.assertEqual(len(projected[0]), 2)

    def test_validation(self) -> None:
        """Single classes, bad priors, and singular covariances raise.

        References:
            [EQ-MVA-005], [EQ-MVA-006]. No further equation is evaluated.

        """
        rows, labels = two_class_data()
        with self.assertRaises(ValueError):
            fit_linear_discriminant(rows, [0] * 8)
        with self.assertRaises(ValueError):
            fit_linear_discriminant(rows, labels, priors={0: 1.0})
        with self.assertRaises(ValueError):
            fit_quadratic_discriminant(rows[:3] + rows[4:6], [0, 0, 0, 1, 1])


class KMeansTests(unittest.TestCase):
    """k-means against exhaustive enumeration; silhouettes by hand.

    References:
        [EQ-MVA-008], [EQ-MVA-009], [REF-LLOYD-1982], [REF-ROUSSEEUW-1987].

    """

    points = [
        [0.0, 0.0], [0.4, 0.3], [1.0, 0.1], [0.2, 1.1], [3.0, 3.2],
        [3.4, 2.8], [2.6, 3.9], [5.5, 0.2], [6.1, 0.9],
    ]  # fmt: skip

    def best_partition(self, k: int) -> float:
        """Return the smallest within-cluster sum of squares by enumeration.

        References:
            [EQ-MVA-008]. No further equation is evaluated.

        """
        best = math.inf
        for labels in itertools.product(range(k), repeat=len(self.points)):
            if len(set(labels)) != k:
                continue
            total = 0.0
            for c in range(k):
                members = [
                    p
                    for p, label in zip(self.points, labels, strict=True)
                    if label == c
                ]
                centre = [
                    math.fsum(p[j] for p in members) / len(members) for j in range(2)
                ]
                total += math.fsum(
                    (p[j] - centre[j]) ** 2 for p in members for j in range(2)
                )
            best = min(best, total)
        return best

    def test_restarts_reach_the_global_optimum(self) -> None:
        """With 20 seeded restarts the inertia equals the enumerated optimum.

        References:
            [EQ-MVA-008], [REF-ARTHUR-VASSILVITSKII-2007].

        """
        for k in (2, 3):
            result = kmeans(self.points, k, restarts=20, seed=3)
            self.assertAlmostEqual(result.inertia, self.best_partition(k), places=12)
            self.assertTrue(result.converged)
            self.assertEqual(min(result.restart_inertias), result.inertia)
            for c, centre in enumerate(result.centers):
                members = [
                    p
                    for p, label in zip(self.points, result.labels, strict=True)
                    if label == c
                ]
                for j in range(2):
                    self.assertAlmostEqual(
                        centre[j], math.fsum(p[j] for p in members) / len(members)
                    )
            self.assertEqual(result.predict(self.points), result.labels)

    def test_seeded_runs_are_reproducible(self) -> None:
        """The same seed gives the same partition.

        References:
            [EQ-MVA-008]. No further equation is evaluated.

        """
        a = kmeans(self.points, 3, seed=8)
        b = kmeans(self.points, 3, seed=8)
        self.assertEqual(a.labels, b.labels)
        with self.assertRaises(ValueError):
            kmeans(self.points, 0)
        with self.assertRaises(ValueError):
            kmeans(self.points, 2, restarts=0)

    def test_silhouette_by_hand(self) -> None:
        """Points ``0, 1 | 4, 6`` give ``4/5, 3/4, 3/7, 7/11`` exactly.

        References:
            [EQ-MVA-009], [REF-ROUSSEEUW-1987].

        """
        widths = silhouette_samples([0.0, 1.0, 4.0, 6.0], [0, 0, 1, 1])
        expected = [Fraction(4, 5), Fraction(3, 4), Fraction(3, 7), Fraction(7, 11)]
        for got, want in zip(widths, expected, strict=True):
            self.assertAlmostEqual(got, float(want), places=15)
        self.assertAlmostEqual(
            silhouette_score([0.0, 1.0, 4.0, 6.0], [0, 0, 1, 1]),
            float(sum(expected) / 4),
            places=15,
        )
        self.assertEqual(silhouette_samples([0.0, 1.0, 5.0], [0, 0, 1])[2], 0.0)
        with self.assertRaises(ValueError):
            silhouette_samples([0.0, 1.0], [0, 0])
        with self.assertRaises(ValueError):
            silhouette_samples([0.0, 1.0, 2.0], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
