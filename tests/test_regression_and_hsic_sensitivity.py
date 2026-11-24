"""Regression-based and HSIC sensitivity measures, and their two plots.

Reference values marked "R 4.6.2" were computed offline with base R; the R
code is quoted next to each pin. The tests never call R.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import random
import unittest

from probstat.modeling import flood_model
from probstat.sensitivity import (
    hsic_indices,
    model_hsic_indices,
    model_regression_sensitivity,
    regression_sensitivity,
)
from probstat.statistics import pearson_correlation

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

# Twelve rows (x1, x2, x3, y), y = 1 + 2 x1 - x2 + x3^2 / 2 + N(0, 0.3^2),
# rounded to three decimals (random.Random(7)).
SMALL = (
    (0.324, 0.151, 0.651, 2.043),
    (0.366, 0.058, 0.507, 1.966),
    (0.037, 0.434, 0.07, 0.908),
    (0.827, 0.124, 0.223, 2.725),
    (0.627, 0.948, 0.577, 0.819),
    (0.047, 0.858, 0.29, 0.774),
    (0.144, 0.118, 0.308, 1.294),
    (0.582, 0.639, 0.372, 1.421),
    (0.548, 0.063, 0.06, 2.159),
    (0.428, 0.314, 0.586, 2.15),
    (0.453, 0.3, 0.794, 1.851),
    (0.574, 0.525, 0.875, 1.793),
)
SMALL_X = tuple(row[:3] for row in SMALL)
SMALL_Y = tuple(row[3] for row in SMALL)


def _gaussian_gram(values: list[float]) -> list[list[float]]:
    """Return the Gaussian Gram matrix with the sample-sd bandwidth."""
    n = len(values)
    mean = sum(values) / n
    h2 = sum((v - mean) ** 2 for v in values) / (n - 1)
    return [[math.exp(-((a - b) ** 2) / (2.0 * h2)) for b in values] for a in values]


class RegressionSensitivityTests(unittest.TestCase):
    """SRC, PCC, SRRC, PRCC, tests, and notes."""

    def test_small_sample_matches_r(self) -> None:
        """Coefficients, tests, and rank measures equal base R lm values.

        R 4.6.2::

            d <- read.csv("small.csv"); X <- d[, 1:3]; y <- d$y
            fit <- lm(y ~ x1 + x2 + x3, data = d); s <- summary(fit)
            coef(fit)[-1] * sapply(X, sd) / sd(y)          # SRC
            s$coefficients[-1, 3:4]; s$r.squared; s$fstatistic
            confint(fit, level = 0.9)[-1, ] * sapply(X, sd) / sd(y)
            # PCC: cor of residuals of y and x_i on the other inputs
            # rank versions: lm on lapply(X, rank) and rank(y)
        """
        result = regression_sensitivity(
            SMALL_X, SMALL_Y, names=("x1", "x2", "x3"), rank=True, confidence=0.9
        )
        expected = {
            "src": (0.521288815876332, -0.728256629304666, 0.115864341667837),
            "t_statistics": (3.92854331742356, -5.59059867681723, 0.86522066674074),
            "p_values": (0.00436561660207456, 0.00051587405590475, 0.412110420390684),
            "src_lower": (0.274540462548928, -0.970489753899710, -0.133153414669225),
            "src_upper": (0.768037169203736, -0.486023504709622, 0.364882098004899),
            "pcc": (0.811546493988807, -0.892302209710718, 0.292521260036432),
            "srrc": (0.449000886531778, -0.757472103015516, 0.101140461676461),
            "prcc": (0.615175094645104, -0.792951016358387, 0.170163799590846),
        }
        for field, values in expected.items():
            actual = getattr(result, field)
            for got, want in zip(actual, values, strict=True):
                self.assertAlmostEqual(got, want, places=9, msg=field)
        self.assertAlmostEqual(result.r_squared, 0.868744087922057, places=12)
        self.assertAlmostEqual(result.adjusted_r_squared, 0.819523120892829, places=12)
        self.assertAlmostEqual(result.f_statistic, 17.6498785041379, places=9)
        self.assertAlmostEqual(result.f_p_value, 0.000690964421520279, places=12)
        self.assertEqual(result.f_df, (3, 8))
        self.assertAlmostEqual(result.rank_r_squared, 0.680929940142417, places=12)
        self.assertEqual(result.ranking(), ("x2", "x1", "x3"))
        self.assertEqual(result.significant(), ("x1", "x2"))
        self.assertAlmostEqual(
            result.as_mapping()["x2"], 0.728256629304666**2, places=9
        )

    def test_linear_model_recovers_variance_shares(self) -> None:
        """For y = sum a_i x_i + e, SRC_i^2 ~ a_i^2 Var(X_i) / Var(Y)."""
        rng = random.Random(11)
        coefficients = (3.0, -2.0, 0.5)
        rows = [tuple(rng.uniform(0.0, 1.0) for _ in range(3)) for _ in range(3000)]
        y = [
            sum(a * x for a, x in zip(coefficients, row, strict=True))
            + rng.gauss(0.0, 0.2)
            for row in rows
        ]
        total = sum(a * a / 12.0 for a in coefficients) + 0.04
        result = regression_sensitivity(rows, y)
        for got, a in zip(result.src_squared, coefficients, strict=True):
            self.assertAlmostEqual(got, a * a / 12.0 / total, delta=0.02)
        # Sample correlations of order 1/sqrt(n) between the inputs keep the
        # two apart by a few hundredths (EQ-SENS-SRC-001).
        self.assertAlmostEqual(result.sum_src_squared, result.r_squared, delta=0.05)
        self.assertAlmostEqual(result.r_squared, 1.0 - 0.04 / total, delta=0.01)
        self.assertEqual(result.notes, ())
        self.assertIsNone(result.srrc)
        with self.assertRaises(ValueError):
            result.ranking(rank=True)

    def test_notes_flag_nonlinear_and_monotone_responses(self) -> None:
        """A symmetric quadratic gives a low R^2; exp(5x) favours the ranks."""
        rng = random.Random(5)
        rows = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(500)]
        quadratic = regression_sensitivity(rows, [x * x for x, _ in rows])
        self.assertLess(quadratic.r_squared, 0.1)
        self.assertTrue(
            any("R^2" in note and "< 0.7" in note for note in quadratic.notes)
        )
        monotone = regression_sensitivity(
            rows, [math.exp(5.0 * x) + 0.1 * z for x, z in rows], rank=True
        )
        self.assertGreater(monotone.rank_r_squared, monotone.r_squared + 0.1)
        self.assertTrue(any("monotone" in note for note in monotone.notes))

    def test_correlated_inputs_break_additivity(self) -> None:
        """Correlated inputs make sum SRC^2 differ from R^2."""
        rng = random.Random(3)
        rows = []
        for _ in range(2000):
            z = rng.gauss(0.0, 1.0)
            rows.append((z, 0.9 * z + 0.3 * rng.gauss(0.0, 1.0)))
        y = [a + b for a, b in rows]
        result = regression_sensitivity(rows, y)
        self.assertGreater(result.r_squared, 0.999)
        self.assertLess(result.sum_src_squared, 0.6)
        self.assertTrue(any("correlated" in note for note in result.notes))

    def test_invalid_samples_raise(self) -> None:
        """Degenerate samples and settings are rejected."""
        with self.assertRaises(ValueError):
            regression_sensitivity(SMALL_X[:5], SMALL_Y[:5])
        with self.assertRaises(ValueError):
            regression_sensitivity([(1.0, k) for k in range(10)], list(range(10)))
        with self.assertRaises(ValueError):
            regression_sensitivity(SMALL_X, [1.0] * 12)
        with self.assertRaises(ValueError):
            regression_sensitivity(SMALL_X, SMALL_Y, confidence=1.0)
        with self.assertRaises(ValueError):
            regression_sensitivity(SMALL_X, SMALL_Y, names=("a", "b"))
        with self.assertRaises(ValueError):
            regression_sensitivity(SMALL_X, SMALL_Y[:-1])
        result = regression_sensitivity(SMALL_X, SMALL_Y)
        with self.assertRaises(ValueError):
            result.significant(0.0)

    def test_flood_model_matches_r_on_the_same_sample(self) -> None:
        """Flood overflow and cost indices on 1000 draws equal base R lm.

        The sample (seed 2026, Monte Carlo design) was exported to CSV;
        R 4.6.2::

            fit <- lm(S ~ Q + Ks + Zv + Zm + Hd + Cb + L + B, data = d)
            summary(fit)$r.squared
            coef(fit)[-1] * sapply(d[, 1:8], sd) / sd(d$S)
        """
        overflow = model_regression_sensitivity(
            flood_model(), "overflow", 1000, seed=2026, rank=True
        )
        self.assertEqual(overflow.evaluations, 1000)
        self.assertAlmostEqual(overflow.r_squared, 0.980054945263502, places=9)
        self.assertAlmostEqual(overflow.src[0], 0.57428887894795233, places=9)
        self.assertAlmostEqual(overflow.src[4], -0.53975815956546880, places=9)
        self.assertAlmostEqual(overflow.pcc[1], -0.9290798781995080, places=9)
        self.assertAlmostEqual(overflow.rank_r_squared, 0.948165146330393, places=9)
        self.assertEqual(overflow.notes, ())
        cost = model_regression_sensitivity(flood_model(), "cost", 1000, seed=2026)
        self.assertAlmostEqual(cost.r_squared, 0.686924597779445, places=9)
        self.assertAlmostEqual(cost.src[0], 0.56929392125998513, places=9)
        self.assertTrue(cost.notes)
        with self.assertRaises(KeyError):
            model_regression_sensitivity(flood_model(), "missing", 100, seed=1)
        with self.assertRaises(ValueError):
            model_regression_sensitivity(
                flood_model(), "cost", 100, seed=1, design="grid"
            )


class HSICTests(unittest.TestCase):
    """HSIC estimators, normalization, tests, and target transforms."""

    def test_small_sample_matches_r(self) -> None:
        """Biased/unbiased HSIC, R2_HSIC, and Gamma p-values equal base R.

        R 4.6.2 (n = 12, H = diag(n) - 1/n)::

            gram <- function(v) exp(-outer(v, v, "-")^2 / (2 * sd(v)^2))
            hb <- function(K, L) sum(diag(K %*% H %*% L %*% H)) / n^2
            # hu: Song et al. (2012) with zero-diagonal Gram matrices
            # Gamma test: Gretton et al. (2008), pgamma(n HSIC_b, m^2/v,
            #   scale = n v / m, lower.tail = FALSE)
        """
        biased = hsic_indices(SMALL_X, SMALL_Y, test="gamma")
        for got, want in zip(
            biased.hsic,
            (0.0349804141035602, 0.0437115917081807, 0.0183577252380831),
            strict=False,
        ):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(
            biased.r2_hsic,
            (0.389033092350672, 0.472845707443485, 0.197526243038563),
            strict=False,
        ):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(
            biased.p_values,
            (0.0521872718546866, 0.0174748681327473, 0.406662656287804),
            strict=False,
        ):
            self.assertAlmostEqual(got, want, places=9)
        unbiased = hsic_indices(SMALL_X, SMALL_Y, estimator="unbiased", test="gamma")
        for got, want in zip(
            unbiased.hsic,
            (0.01908642712611315, 0.03415788926763728, 0.00417171991333297),
            strict=False,
        ):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(
            unbiased.r2_hsic,
            (0.2102118897825649, 0.3605352919198909, 0.0434254816433921),
            strict=False,
        ):
            self.assertAlmostEqual(got, want, places=12)
        # The Gamma test always uses the biased statistic.
        self.assertEqual(unbiased.p_values, biased.p_values)
        self.assertEqual(biased.ranking(), ("x1", "x0", "x2"))

    def test_estimators_equal_brute_force_v_and_u_statistics(self) -> None:
        """Closed forms equal the V- and U-statistic definitions (n = 7)."""
        rng = random.Random(2)
        x = [rng.gauss(0.0, 1.0) for _ in range(7)]
        y = [a + rng.gauss(0.0, 1.0) for a in x]
        k, l_ = _gaussian_gram(x), _gaussian_gram(y)
        n = 7
        idx = range(n)
        v_stat = (
            sum(k[i][j] * l_[i][j] for i in idx for j in idx) / n**2
            + sum(map(sum, k)) * sum(map(sum, l_)) / n**4
            - 2.0
            * sum(k[i][j] * l_[i][q] for i in idx for j in idx for q in idx)
            / n**3
        )
        quads = list(itertools.permutations(idx, 4))
        u_stat = sum(
            k[i][j] * l_[i][j] + k[i][j] * l_[q][r] - 2.0 * k[i][j] * l_[i][q]
            for i, j, q, r in quads
        ) / len(quads)
        rows = [[value] for value in x]
        self.assertAlmostEqual(
            hsic_indices(rows, y, test="gamma").hsic[0], v_stat, places=13
        )
        self.assertAlmostEqual(
            hsic_indices(rows, y, estimator="unbiased", test="gamma").hsic[0],
            u_stat,
            places=13,
        )

    def test_square_of_symmetric_input_has_zero_correlation_but_strong_hsic(
        self,
    ) -> None:
        """Y = X^2 with X symmetric: Pearson ~ 0, HSIC rejects independence."""
        rng = random.Random(8)
        half = [rng.uniform(0.0, 1.0) for _ in range(150)]
        x = half + [-value for value in half]
        noise = [rng.uniform(-1.0, 1.0) for _ in range(300)]
        y = [value * value for value in x]
        self.assertLess(abs(pearson_correlation(x, y)), 1e-12)
        rows = list(zip(x, noise, strict=True))
        gamma = hsic_indices(rows, y, test="gamma", names=("x", "noise"))
        self.assertGreater(gamma.r2_hsic[0], 0.3)
        self.assertLess(gamma.r2_hsic[1], 0.05)
        self.assertLess(gamma.p_values[0], 1e-6)
        self.assertGreater(gamma.p_values[1], 0.01)
        permutation = hsic_indices(
            rows, y, test="permutation", permutations=99, seed=4, names=("x", "noise")
        )
        self.assertAlmostEqual(permutation.p_values[0], 0.01)
        self.assertEqual(permutation.permutations, 99)
        self.assertEqual(permutation.independent(0.05), ("noise",))
        self.assertEqual(gamma.as_mapping()["x"], gamma.r2_hsic[0])

    def test_p_values_under_independence_are_roughly_uniform(self) -> None:
        """Gamma and permutation p-values have loose nominal coverage."""
        rng = random.Random(21)
        gamma_p: list[float] = []
        for _ in range(300):
            x = [[rng.gauss(0.0, 1.0)] for _ in range(40)]
            y = [rng.expovariate(1.0) for _ in range(40)]
            gamma_p.append(hsic_indices(x, y, test="gamma").p_values[0])
        self.assertTrue(0.04 <= sum(p < 0.1 for p in gamma_p) / 300 <= 0.18)
        self.assertTrue(0.4 <= sum(gamma_p) / 300 <= 0.6)
        permutation_p: list[float] = []
        for replicate in range(60):
            x = [[rng.random()] for _ in range(25)]
            y = [rng.random() for _ in range(25)]
            result = hsic_indices(x, y, permutations=39, seed=replicate)
            permutation_p.append(result.p_values[0])
        self.assertTrue(0.05 <= sum(p <= 0.2 for p in permutation_p) / 60 <= 0.4)

    def test_unbiased_estimator_is_centred_under_independence(self) -> None:
        """Mean HSIC_u ~ 0 while HSIC_b keeps a positive O(1/n) bias."""
        rng = random.Random(9)
        unbiased: list[float] = []
        biased: list[float] = []
        for _ in range(200):
            x = [[rng.gauss(0.0, 1.0)] for _ in range(20)]
            y = [rng.gauss(0.0, 1.0) for _ in range(20)]
            unbiased.append(
                hsic_indices(x, y, estimator="unbiased", test="gamma").hsic[0]
            )
            biased.append(hsic_indices(x, y, test="gamma").hsic[0])
        mean_u = sum(unbiased) / 200
        spread = math.sqrt(sum((v - mean_u) ** 2 for v in unbiased) / 199)
        self.assertLess(abs(mean_u), 3.0 * spread / math.sqrt(200))
        self.assertGreater(sum(biased) / 200, 3.0 * spread / math.sqrt(200))
        self.assertTrue(all(0.0 <= value for value in biased))

    def test_target_bandwidth_and_validation(self) -> None:
        """Threshold and callable targets, bandwidth rules, and errors."""
        rng = random.Random(4)
        rows = [(rng.random(), rng.random()) for _ in range(200)]
        y = [a + 0.1 * b for a, b in rows]
        target = hsic_indices(rows, y, test="gamma", target=0.8)
        self.assertEqual(target.target, "1{Y > 0.8}")
        self.assertAlmostEqual(
            target.target_fraction, sum(v > 0.8 for v in y) / len(y), places=12
        )
        self.assertLess(target.p_values[0], 1e-6)
        weighted = hsic_indices(rows, y, test="gamma", target=math.sqrt)
        self.assertEqual(weighted.target, "w(Y) = sqrt")
        median = hsic_indices(rows, y, test="gamma", bandwidth="median")
        self.assertGreater(median.bandwidths[0], 0.0)
        explicit = hsic_indices(rows, y, test="gamma", bandwidth=(0.3, 0.3, 0.3))
        self.assertEqual(explicit.bandwidths, (0.3, 0.3))
        self.assertEqual(explicit.output_bandwidth, 0.3)
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, target=5.0)
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, kernel="laplace")
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, estimator="plain")
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, test="bootstrap")
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, permutations=0)
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, bandwidth=(0.3,))
        with self.assertRaises(ValueError):
            hsic_indices(rows, y, bandwidth="silverman")
        with self.assertRaises(ValueError):
            hsic_indices(rows[:5], y[:5])
        with self.assertRaises(ValueError):
            target.independent(1.5)

    def test_model_wrapper_on_the_flood_cost(self) -> None:
        """The dike height drives the cost nonlinearly: small SRC^2, clear HSIC."""
        result = model_hsic_indices(flood_model(), "cost", 300, seed=2026, test="gamma")
        self.assertEqual(result.evaluations, 300)
        self.assertEqual(result.input_names[4], "Hd")
        self.assertLess(result.p_values[4], 1e-4)
        self.assertGreater(result.r2_hsic[4], 4.0 * result.r2_hsic[6])


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class SensitivityPlotTests(unittest.TestCase):
    """Smoke tests of the cobweb and input-scatter figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once and draw one sample."""
        import matplotlib

        matplotlib.use("Agg")
        rng = random.Random(1)
        cls.rows = [(rng.random(), rng.random(), rng.random()) for _ in range(150)]
        cls.y = [3.0 * a - b + 0.1 * c for a, b, c in cls.rows]

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_cobweb_highlights_exceeding_runs(self) -> None:
        """Two line bundles, one axis per input plus the output, a claim title."""
        from probstat.visualization import plot_cobweb
        from probstat.visualization.theme import TITLE_GID

        axes = plot_cobweb(self.rows, ("a", "b", "c"), output=self.y, threshold=1.5)
        self.assertEqual(len(axes.collections), 2)
        labels = [tick.get_text() for tick in axes.get_xticklabels()]
        self.assertEqual(labels, ["a", "b", "c", "Y"])
        title = next(t for t in axes.get_children() if t.get_gid() == TITLE_GID)
        self.assertTrue(title.get_text().startswith("Runs with Y > 1.5 need high a"))
        plain = plot_cobweb(self.rows, ("a", "b", "c"), scale="minmax")
        self.assertEqual(len(plain.collections), 1)
        with self.assertRaises(ValueError):
            plot_cobweb(self.rows, ("a", "b", "c"), output=self.y, threshold=100.0)
        with self.assertRaises(ValueError):
            plot_cobweb(self.rows, ("a", "b", "c"), threshold=1.0)
        with self.assertRaises(ValueError):
            plot_cobweb(self.rows, ("a", "b"))
        with self.assertRaises(ValueError):
            plot_cobweb(self.rows, ("a", "b", "c"), scale="log")

    def test_input_scatter_draws_one_panel_per_input(self) -> None:
        """Panels, smoother lines, units, and the derived title."""
        from probstat.visualization import plot_input_scatter
        from probstat.visualization.theme import TITLE_GID

        figure = plot_input_scatter(
            self.rows, self.y, ("a", "b", "c"), units=("m", "", "s"), columns=2
        )
        self.assertEqual(len(figure.axes), 3)
        self.assertTrue(all(len(ax.get_lines()) == 1 for ax in figure.axes))
        self.assertEqual(figure.axes[0].get_xlabel(), "a [m]")
        self.assertEqual(figure.axes[1].get_xlabel(), "b")
        title = next(
            t for t in figure.axes[0].get_children() if t.get_gid() == TITLE_GID
        )
        self.assertTrue(title.get_text().startswith("Y rises most clearly with a"))
        bare = plot_input_scatter(self.rows, self.y, ("a", "b", "c"), smoother=None)
        self.assertTrue(all(not ax.get_lines() for ax in bare.axes))
        with self.assertRaises(ValueError):
            plot_input_scatter(self.rows, self.y, ("a", "b", "c"), smoother="spline")
        with self.assertRaises(ValueError):
            plot_input_scatter(self.rows, self.y[:-1], ("a", "b", "c"))
        with self.assertRaises(ValueError):
            plot_input_scatter(self.rows, self.y, ("a", "b", "c"), columns=0)
        with self.assertRaises(ValueError):
            plot_input_scatter(self.rows, self.y, ("a", "b", "c"), units=("m",))

    def _tornado(self) -> object:
        """Return a three-input tornado result with one non-monotone input.

        References:
            [EQ-SENS-TORNADO-001]. No equation is evaluated.

        """
        from probstat.sensitivity import tornado_function

        return tornado_function(
            lambda x: 2.0 * x[0] - x[1] + x[2] ** 2,
            low=(-1.0, -1.0, -1.0),
            base=(0.0, 0.0, 0.0),
            high=(1.0, 4.0, 1.0),
            names=("thrust", "drag", "mass"),
        )

    def test_tornado_draws_sorted_bars_with_direct_labels(self) -> None:
        """Rows sorted by swing, two bars per row, hatched high bars, labels.

        References:
            [EQ-SENS-TORNADO-001], [REF-MATPLOTLIB].

        """
        from probstat.visualization import plot_tornado
        from probstat.visualization.theme import TITLE_GID

        axes = plot_tornado(self._tornado(), output_name="Net force", unit="N")
        labels = [tick.get_text() for tick in axes.get_yticklabels()]
        # The top row is the last tick; mass is flagged by its baseline output.
        self.assertEqual(labels, ["mass *", "thrust", "drag"])
        self.assertEqual(len(axes.patches), 6)
        hatched = [patch for patch in axes.patches if patch.get_hatch()]
        self.assertEqual(len(hatched), 3)
        self.assertEqual(axes.get_xlabel(), "Change in Net force from baseline [N]")
        # Every bar starts at zero, so its length is the change it encodes.
        for patch in axes.patches:
            self.assertEqual(
                min(patch.get_x(), patch.get_x() + patch.get_width()),
                min(0.0, patch.get_width()),
            )
        title = next(t for t in axes.get_children() if t.get_gid() == TITLE_GID)
        self.assertTrue(title.get_text().startswith("Drag swings Net force the most"))
        texts = [t.get_text() for t in axes.texts]
        self.assertIn("-1", texts)
        self.assertTrue(any(t.startswith("baseline") for t in texts))
        self.assertEqual(len(axes.lines), 1)

    def test_tornado_text_stays_on_canvas_in_both_themes(self) -> None:
        """Text is at least 9 pt and inside the figure after fitting; dark works.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        from probstat.visualization import plot_tornado
        from probstat.visualization.theme import (
            GRAPHITE,
            chart_theme,
            fit_text_to_canvas,
        )

        for palette in (None, GRAPHITE):
            if palette is None:
                axes = plot_tornado(self._tornado())
            else:
                with chart_theme(palette):
                    axes = plot_tornado(self._tornado())
            figure = axes.figure
            fit_text_to_canvas(figure)
            renderer = figure.canvas.get_renderer()
            tight = figure.get_tightbbox(renderer)
            width, height = figure.get_size_inches()
            self.assertGreaterEqual(tight.x0, -1e-9)
            self.assertGreaterEqual(tight.y0, -1e-9)
            self.assertLessEqual(tight.x1, width + 1e-9)
            self.assertLessEqual(tight.y1, height + 1e-9)
            sizes = [t.get_fontsize() for t in axes.texts]
            sizes += [t.get_fontsize() for t in axes.get_yticklabels()]
            sizes += [t.get_fontsize() for t in axes.get_xticklabels()]
            self.assertGreaterEqual(min(sizes), 9.0)

    def test_tornado_options_and_errors(self) -> None:
        """``top`` limits the rows; invalid arguments are rejected.

        References:
            [EQ-SENS-TORNADO-001].

        """
        from probstat.visualization import plot_tornado

        axes = plot_tornado(self._tornado(), top=2, title="Custom claim")
        self.assertEqual(len(axes.patches), 4)
        self.assertEqual(len(axes.get_yticklabels()), 2)
        with self.assertRaises(TypeError):
            plot_tornado("not a result")
        with self.assertRaises(ValueError):
            plot_tornado(self._tornado(), top=0)
        from probstat.sensitivity import tornado_function

        flat = tornado_function(
            lambda x: 1.0, low=(0.0,), base=(1.0,), high=(2.0,), names=("a",)
        )
        axes = plot_tornado(flat)
        title = next(t for t in axes.get_children() if t.get_gid() == "probstat-title")
        self.assertTrue(title.get_text().startswith("No single input"))


if __name__ == "__main__":
    unittest.main()
