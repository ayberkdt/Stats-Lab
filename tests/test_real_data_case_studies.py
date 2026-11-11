"""Tests for the historical data sets, count models, and the real-data lab.

Oracles: the totals and summaries printed in the primary sources
(Bortkiewicz 1898, page 25; Rutherford and Geiger 1910, page 701; the
Lieblein-Zelen worksheet, page 286); an independent transcription of
Newcomb's series in the order of the MASS documentation; closed-form
maximum-likelihood estimators; the chi-square closed form of Garwood's
interval for two degrees of freedom; a brute-force grid maximization of the
negative binomial likelihood; the published conclusions quoted in each
test; and the tables of the fitting guide, which must equal the lab output.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

from probstat.distributions import NegativeBinomial, Poisson
from probstat.inference import (
    CountFit,
    count_goodness_of_fit,
    fit_gamma_mle,
    fit_negative_binomial_counts,
    fit_poisson_counts,
    fit_weibull_mle,
    poisson_mean_interval,
)
from probstat.statistics import (
    BORTKIEWICZ_HORSE_KICKS,
    DATASETS,
    LIEBLEIN_ZELEN_BALL_BEARINGS,
    NEWCOMB_PASSAGE_TIMES,
    RUTHERFORD_GEIGER_SCINTILLATIONS,
    STUDENT_SLEEP,
    DataColumn,
    DataSet,
    dataset_by_key,
    mean,
    median,
    trimmed_mean,
)
from probstat.visualization.fit_plots import hanging_rootogram_bars

ROOT = Path(__file__).parents[1]
LAB_PATH = ROOT / "examples" / "real_data_fitting_lab.py"
GUIDE = ROOT / "docs" / "foundations" / "uncertainty-modeling-and-fitting.md"
REGISTRY = ROOT / "docs" / "equation-registry.md"
BIBLIOGRAPHY = ROOT / "docs" / "references.md"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

# Newcomb's third series in the order of Staudte and Sheather, transcribed
# from the MASS documentation (REF-R-MASS-NEWCOMB), independently of the
# Stigler Table 5 order stored in the data set.
NEWCOMB_MASS_ORDER = (
    28, -44, 29, 30, 24, 28, 37, 32, 36, 27, 26, 28, 29, 26, 27, 22, 23, 20,
    25, 25, 36, 23, 31, 32, 24, 27, 33, 16, 24, 29, 36, 21, 28, 26, 27, 27,
    32, 25, 28, 24, 40, 21, 31, 32, 28, 26, 30, 27, 26, 24, 32, 29, 34, -2,
    25, 19, 36, 29, 30, 22, 28, 33, 39, 25, 16, 23,
)
# Rutherford and Geiger (1910), page 701, row "Theoretical values", k = 0..12.
RUTHERFORD_THEORETICAL = (54, 210, 407, 525, 508, 394, 254, 140, 68, 29, 11, 4, 1)
# Bortkiewicz (1898), page 25, column "zu erwarten war", k = 0..4 and "5 u. mehr".
BORTKIEWICZ_EXPECTED = (108.7, 66.3, 20.2, 4.1, 0.6, 0.1)


def _load_lab() -> ModuleType:
    """Import the laboratory script as a module.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    spec = importlib.util.spec_from_file_location("real_data_fitting_lab", LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DataIntegrityTests(unittest.TestCase):
    """Bundled values reproduce the totals printed in their sources."""

    def test_bortkiewicz_table(self) -> None:
        """200 corps-years, 122 deaths, mean 0.61 (page 25)."""
        data = BORTKIEWICZ_HORSE_KICKS
        counts = data.expand_frequencies("deaths", "corps_years")
        self.assertEqual(len(counts), 200)
        self.assertEqual(sum(counts), 122)
        self.assertEqual(mean(counts), 0.61)
        self.assertEqual(data.values("corps_years"), (109, 65, 22, 3, 1))

    def test_bortkiewicz_expected_column_is_reproduced(self) -> None:
        """200 Poisson(0.61) probabilities round to the printed column."""
        fit = fit_poisson_counts(
            BORTKIEWICZ_HORSE_KICKS.expand_frequencies("deaths", "corps_years")
        )
        reproduced = tuple(round(value, 1) for value in fit.expected_frequencies(4))
        self.assertEqual(reproduced, BORTKIEWICZ_EXPECTED)

    def test_rutherford_geiger_table(self) -> None:
        """2608 intervals, 10097 particles, average 3.87 (page 701)."""
        counts = RUTHERFORD_GEIGER_SCINTILLATIONS.expand_frequencies(
            "particles", "intervals"
        )
        self.assertEqual(len(counts), 2608)
        self.assertEqual(sum(counts), 10097)
        self.assertEqual(round(mean(counts), 2), 3.87)
        self.assertEqual(counts.count(3), 525)

    def test_rutherford_geiger_theoretical_row(self) -> None:
        """2608 p(k; x_bar) lies within 0.51 of each printed value, k <= 12.

        The printed integers are rounded; the largest gap is the rounding of
        140.50 at ``k = 7``.
        """
        fit = fit_poisson_counts(
            RUTHERFORD_GEIGER_SCINTILLATIONS.expand_frequencies(
                "particles", "intervals"
            )
        )
        expected = fit.expected_frequencies(12)
        for k, printed in enumerate(RUTHERFORD_THEORETICAL):
            with self.subTest(k=k):
                self.assertLess(abs(expected[k] - printed), 0.51)

    def test_newcomb_series(self) -> None:
        """66 values, median 27, same multiset as the MASS listing."""
        values = NEWCOMB_PASSAGE_TIMES.values()
        self.assertEqual(len(values), 66)
        self.assertEqual(median(values), 27.0)
        self.assertEqual(sorted(values), sorted(float(v) for v in NEWCOMB_MASS_ORDER))
        self.assertEqual(sorted(values)[:2], [-44.0, -2.0])

    def test_lieblein_zelen_worksheet(self) -> None:
        """23 bearings in ascending order; worksheet median '68.' (page 286)."""
        values = LIEBLEIN_ZELEN_BALL_BEARINGS.values()
        self.assertEqual(len(values), 23)
        self.assertEqual(list(values), sorted(values))
        self.assertEqual((values[0], values[-1]), (17.88, 173.40))
        self.assertEqual(round(median(values)), 68)

    def test_student_sleep(self) -> None:
        """Ten patients; mean paired difference 1.58 h."""
        first = STUDENT_SLEEP.values("drug_1")
        second = STUDENT_SLEEP.values("drug_2")
        self.assertEqual(STUDENT_SLEEP.rows, 10)
        differences = [b - a for a, b in zip(first, second, strict=True)]
        self.assertAlmostEqual(mean(differences), 1.58, places=12)

    def test_records_are_complete_and_cite_registered_sources(self) -> None:
        """Every record has units, provenance, licence, and known keys."""
        bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
        self.assertEqual(len({data.key for data in DATASETS}), len(DATASETS))
        for data in DATASETS:
            with self.subTest(data=data.key):
                self.assertIs(dataset_by_key(data.key), data)
                self.assertTrue(data.licence.startswith("Public domain."))
                self.assertRegex(data.primary_source, r"[Pp]age \d+|Table \d+")
                for column in data.columns:
                    self.assertTrue(column.unit)
                for key in data.references:
                    self.assertIn(f"## {key}\n", bibliography)

    def test_validation(self) -> None:
        """Malformed records and lookups are rejected."""
        column = DataColumn("x", "unit", (1.0, 2.0))
        base = {
            "key": "k",
            "title": "t",
            "description": "d",
            "primary_source": "p",
            "secondary_source": "s",
            "licence": "l",
            "references": ("REF-NIST-EDA",),
        }
        with self.assertRaises(ValueError):
            DataSet(columns=(column, DataColumn("y", "unit", (1.0,))), **base)
        with self.assertRaises(ValueError):
            DataSet(columns=(column, column), **base)
        with self.assertRaises(ValueError):
            DataSet(columns=(), **base)
        with self.assertRaises(ValueError):
            DataSet(columns=(column,), **{**base, "references": ("NIST",)})
        with self.assertRaises(ValueError):
            DataColumn("x", " ", (1.0,))
        with self.assertRaises(ValueError):
            DataColumn("x", "unit", (math.nan,))
        with self.assertRaises(KeyError):
            dataset_by_key("unknown")
        with self.assertRaises(KeyError):
            BORTKIEWICZ_HORSE_KICKS.column("unknown")
        bad = DataSet(
            columns=(column, DataColumn("f", "units", (1.5, 2.0))), **base
        )
        with self.assertRaises(ValueError):
            bad.expand_frequencies("x", "f")


class CountModelTests(unittest.TestCase):
    """Poisson and negative binomial fits against closed forms and a grid."""

    OVERDISPERSED = (0,) * 30 + (1,) * 20 + (2,) * 12 + (3,) * 8 + (5,) * 6 + (9,) * 4

    def test_poisson_fit_is_the_sample_mean(self) -> None:
        """lambda_hat = x_bar, SE = sqrt(x_bar/n), log L from the library pmf."""
        counts = (0, 1, 1, 2, 4, 0, 3)
        fit = fit_poisson_counts(counts)
        self.assertEqual(fit.mean, sum(counts) / len(counts))
        self.assertAlmostEqual(
            fit.standard_errors[0], math.sqrt(fit.mean / len(counts)), places=15
        )
        law = Poisson(fit.mean)
        self.assertAlmostEqual(
            fit.log_likelihood, math.fsum(law.log_pmf(k) for k in counts), places=10
        )
        self.assertAlmostEqual(fit.aic, 2.0 - 2.0 * fit.log_likelihood, places=10)

    def test_negative_binomial_boundary_without_overdispersion(self) -> None:
        """Divisor-n variance <= mean gives the Poisson limit and AIC + 2."""
        counts = BORTKIEWICZ_HORSE_KICKS.expand_frequencies("deaths", "corps_years")
        poisson = fit_poisson_counts(counts)
        nb = fit_negative_binomial_counts(counts)
        self.assertTrue(nb.at_boundary)
        self.assertTrue(math.isinf(nb.estimates[0]))
        self.assertEqual(nb.log_likelihood, poisson.log_likelihood)
        self.assertAlmostEqual(nb.aic, poisson.aic + 2.0, places=10)

    def test_negative_binomial_matches_grid_maximum(self) -> None:
        """The score root beats every point of a fine (r, mu) grid."""
        counts = self.OVERDISPERSED
        fit = fit_negative_binomial_counts(counts)
        self.assertFalse(fit.at_boundary)
        self.assertEqual(fit.mean, mean(counts))
        size = fit.estimates[0]
        best = -math.inf
        for i in range(-40, 41):
            for j in range(-10, 11):
                trial = CountFit(
                    "negative-binomial",
                    ("size", "mean"),
                    (size * math.exp(i / 40.0), fit.mean * (1.0 + j / 50.0)),
                    (math.nan, math.nan),
                    0.0,
                    len(counts),
                    0.0,
                    0.0,
                    False,
                    "",
                )
                best = max(best, math.fsum(trial.log_pmf(k) for k in counts))
        self.assertGreaterEqual(fit.log_likelihood, best - 1e-9)
        self.assertTrue(math.isfinite(fit.standard_errors[0]))

    def test_negative_binomial_pmf_matches_integer_size_law(self) -> None:
        """For integer size the mass equals NegativeBinomial(r, r/(r+mu))."""
        fit = CountFit(
            "negative-binomial",
            ("size", "mean"),
            (3.0, 2.5),
            (math.nan, math.nan),
            0.0,
            10,
            0.0,
            0.0,
            False,
            "",
        )
        law = NegativeBinomial(3, 3.0 / 5.5)
        for k in range(12):
            with self.subTest(k=k):
                self.assertAlmostEqual(fit.pmf(k), law.pmf(k), places=13)
        self.assertAlmostEqual(math.fsum(fit.expected_frequencies(200)), 10.0)

    def test_profile_expansion_towards_the_poisson_limit(self) -> None:
        """l(r) - l_Poisson ~ n (s_n^2 - x_bar) / (2 r) for large r."""
        counts = self.OVERDISPERSED
        n = len(counts)
        x_bar = mean(counts)
        s2 = math.fsum((k - x_bar) ** 2 for k in counts) / n
        poisson = fit_poisson_counts(counts).log_likelihood
        size = 1e5
        trial = CountFit(
            "negative-binomial",
            ("size", "mean"),
            (size, x_bar),
            (math.nan, math.nan),
            0.0,
            n,
            0.0,
            0.0,
            False,
            "",
        )
        profile = math.fsum(trial.log_pmf(k) for k in counts)
        self.assertAlmostEqual(
            (profile - poisson) * size, n * (s2 - x_bar) / 2.0, delta=1e-2
        )

    def test_garwood_interval_closed_forms(self) -> None:
        """chi2(2) quantiles are -2 log(1 - q): S = 0 and S = 1, n = 1."""
        zero = poisson_mean_interval([0])
        self.assertEqual(zero.lower, 0.0)
        self.assertAlmostEqual(zero.upper, -math.log(0.025), places=9)
        one = poisson_mean_interval([1])
        self.assertAlmostEqual(one.lower, -math.log(0.975), places=9)
        with self.assertRaises(ValueError):
            poisson_mean_interval([1], confidence=1.0)

    def test_pooled_pearson_statistic(self) -> None:
        """Bortkiewicz cells {0}, {1}, {2+} and the hand-computed X^2."""
        counts = BORTKIEWICZ_HORSE_KICKS.expand_frequencies("deaths", "corps_years")
        fit = fit_poisson_counts(counts)
        result = count_goodness_of_fit(counts, fit, n_bootstrap=49, seed=3)
        self.assertEqual(result.cells, ((0, 0), (1, 1), (2, None)))
        self.assertEqual(result.observed, (109, 65, 26))
        p0 = math.exp(-0.61)
        p1 = 0.61 * p0
        expected = (200 * p0, 200 * p1, 200 * (1.0 - p0 - p1))
        hand = math.fsum(
            (o - e) ** 2 / e for o, e in zip(result.observed, expected, strict=True)
        )
        self.assertAlmostEqual(result.statistic, hand, places=10)
        self.assertEqual(result.degrees_of_freedom, 1)
        again = count_goodness_of_fit(counts, fit, n_bootstrap=49, seed=3)
        self.assertEqual(result.bootstrap_p_value, again.bootstrap_p_value)
        with self.assertRaises(ValueError):
            count_goodness_of_fit(counts[:-1], fit, n_bootstrap=9, seed=3)

    def test_invalid_counts(self) -> None:
        """Negative, non-integer, all-zero, and single counts are rejected."""
        for bad in ((0, -1), (0, 0), (3,)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fit_poisson_counts(bad)
        with self.assertRaises(TypeError):
            fit_negative_binomial_counts((1.5, 2))


class RobustAndPlotHelperTests(unittest.TestCase):
    """Trimmed mean and rootogram geometry."""

    def test_trimmed_mean(self) -> None:
        """floor(alpha n) values leave each end; alpha = 0 is the mean."""
        values = (1.0, 2.0, 3.0, 4.0, 100.0)
        self.assertEqual(trimmed_mean(values, 0.0), mean(values))
        self.assertEqual(trimmed_mean(values, 0.2), 3.0)
        self.assertEqual(trimmed_mean(values, 0.1), mean(values))
        with self.assertRaises(ValueError):
            trimmed_mean(values, 0.5)

    def test_hanging_rootogram_bars(self) -> None:
        """Bars hang from sqrt(E) with length sqrt(O)."""
        bars = hanging_rootogram_bars((4.0, 9.0), (9.0, 9.0))
        self.assertEqual(bars, ((1.0, 3.0), (0.0, 3.0)))
        with self.assertRaises(ValueError):
            hanging_rootogram_bars((1.0,), (1.0, 2.0))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
    def test_plot_functions_render(self) -> None:
        """The rootogram and Q-Q plot draw one bar or point per value."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from probstat.distributions import Normal
        from probstat.visualization.fit_plots import plot_qq, plot_rootogram

        axes = plot_rootogram((109, 65, 22, 3, 1), (108.7, 66.3, 20.2, 4.1, 0.6))
        self.assertEqual(len(axes.patches), 5)
        plt.close(axes.figure)
        axes = plot_qq((1.0, 2.0, 3.5, 4.0), Normal(2.5, 1.2))
        self.assertEqual(len(axes.collections[0].get_offsets()), 4)
        plt.close(axes.figure)


class LaboratoryTests(unittest.TestCase):
    """The lab reproduces published conclusions and the guide's tables."""

    lab: ModuleType
    results: Any
    seconds: float

    @classmethod
    def setUpClass(cls) -> None:
        """Run the case studies once with the lab's own seed.

        References:
            [EQ-INF-GOF-PB-001]. No equation is evaluated.

        """
        cls.lab = _load_lab()
        started = time.perf_counter()
        cls.results = cls.lab.run_case_studies()
        cls.seconds = time.perf_counter() - started

    def test_runs_in_under_a_minute(self) -> None:
        """Every fit and bootstrap of the lab takes less than 60 s."""
        self.assertLess(self.seconds, 60.0)

    def test_bortkiewicz_conclusion(self) -> None:
        """Page 25: agreement 'leaves nothing to be desired'; no overdispersion."""
        case = self.results.horse_kicks
        self.assertGreater(case.goodness_of_fit.bootstrap_p_value, 0.05)
        self.assertTrue(case.negative_binomial.at_boundary)
        self.assertLess(case.poisson.aic, case.negative_binomial.aic)

    def test_rutherford_geiger_conclusion(self) -> None:
        """Page 702: excellent accord; at k = 4 the observation is nearly 5% high."""
        case = self.results.scintillations
        self.assertGreater(case.goodness_of_fit.bootstrap_p_value, 0.05)
        self.assertTrue(case.negative_binomial.at_boundary)
        expected = case.poisson.expected_frequencies(14)
        observed = RUTHERFORD_GEIGER_SCINTILLATIONS.values("intervals")
        self.assertTrue(0.04 < observed[4] / expected[4] - 1.0 < 0.05)

    def test_newcomb_conclusions(self) -> None:
        """Heavy tails beat the normal; light trimming improves on the mean.

        Stigler (1977, abstract): "light trimming provides some improvement
        over the sample mean"; 33.02 is the value implied by the accepted
        speed of light.
        """
        case = self.results.light_speed
        families = [candidate.fit.family for candidate in case.candidates]
        self.assertEqual(families, ["StudentT", "Normal"])
        self.assertLess(case.normality.lilliefors.p_value, 0.01)
        student = case.candidates[0].goodness_of_fit
        self.assertGreater(min(student.ks_p_value, student.ad_p_value), 0.05)
        estimates = dict(case.estimates)
        reference = self.lab.LIGHT_SPEED_REFERENCE
        self.assertLess(
            abs(reference - estimates["10% trimmed mean"]),
            abs(reference - estimates["mean"]),
        )
        values = NEWCOMB_PASSAGE_TIMES.values()
        self.assertAlmostEqual(estimates["mean"], math.fsum(values) / 66, places=12)

    def test_bearing_fits_equal_closed_forms(self) -> None:
        """Weibull, gamma, and lognormal estimates equal the direct MLEs."""
        values = LIEBLEIN_ZELEN_BALL_BEARINGS.values()
        fits = {c.fit.family: c.fit for c in self.results.bearings}
        weibull = fit_weibull_mle(values)
        self.assertAlmostEqual(fits["Weibull"].params["shape"], weibull.shape, places=9)
        self.assertAlmostEqual(fits["Weibull"].params["scale"], weibull.scale, places=7)
        gamma = fit_gamma_mle(values)
        self.assertAlmostEqual(fits["Gamma"].params["shape"], gamma.shape, places=9)
        logs = [math.log(value) for value in values]
        log_mean = math.fsum(logs) / 23
        log_sd = math.sqrt(math.fsum((x - log_mean) ** 2 for x in logs) / 23)
        self.assertAlmostEqual(fits["LogNormal"].params["mu"], log_mean, places=12)
        self.assertAlmostEqual(fits["LogNormal"].params["sigma"], log_sd, places=12)

    def test_bearing_families_are_not_distinguished(self) -> None:
        """All three families pass calibrated GOF and lie within 2 AIC units."""
        candidates = self.results.bearings
        aics = [candidate.fit.aic for candidate in candidates]
        self.assertLess(max(aics) - min(aics), 2.0)
        for candidate in candidates:
            with self.subTest(family=candidate.fit.family):
                gof = candidate.goodness_of_fit
                self.assertGreater(min(gof.ks_p_value, gof.ad_p_value), 0.05)

    def test_sleep_paired_t_closed_form(self) -> None:
        """The statistic is d_bar / (s_d / sqrt(n)) with nine degrees of freedom."""
        sleep = self.results.sleep
        first = STUDENT_SLEEP.values("drug_1")
        second = STUDENT_SLEEP.values("drug_2")
        differences = [b - a for a, b in zip(first, second, strict=True)]
        d_bar = math.fsum(differences) / 10
        s_d = math.sqrt(math.fsum((d - d_bar) ** 2 for d in differences) / 9)
        statistic = d_bar / (s_d / math.sqrt(10))
        self.assertAlmostEqual(sleep.statistic, statistic, places=12)
        self.assertEqual(sleep.df, 9.0)
        self.assertLess(sleep.p_value, 0.01)

    def test_guide_tables_equal_lab_output(self) -> None:
        """Every number in the guide's case-study block comes from the lab."""
        text = GUIDE.read_text(encoding="utf-8")
        begin, end = self.lab.DOC_BEGIN, self.lab.DOC_END
        self.assertEqual(text.count(begin), 1)
        block = text[text.index(begin) : text.index(end) + len(end)]
        self.assertEqual(block, self.lab.results_markdown(self.results))

    def test_lab_docstring_contract(self) -> None:
        """Sections, registered EQ and REF keys, and a fixed integer seed."""
        doc = self.lab.__doc__ or ""
        sections = (
            "Definitions:",
            "Assumptions:",
            "Equations:",
            "Diagnostics:",
            "Cost and error:",
            "References:",
        )
        positions = [doc.index(f"\n{section}\n") for section in sections]
        self.assertEqual(positions, sorted(positions))
        equations = doc[positions[2] : positions[3]]
        registry = REGISTRY.read_text(encoding="utf-8")
        for key in re.findall(r"EQ-[A-Z0-9-]+", equations):
            with self.subTest(key=key):
                self.assertIn(f"### {key} ", registry)
        bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
        for key in re.findall(r"REF-[A-Z0-9-]+", doc[positions[5] :]):
            with self.subTest(key=key):
                self.assertIn(f"## {key}\n", bibliography)
        self.assertIsInstance(self.lab.SEED, int)


if __name__ == "__main__":
    unittest.main()
