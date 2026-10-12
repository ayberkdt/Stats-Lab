"""Smoke tests for the validation, missing-data, contingency, and PIT figures."""

from __future__ import annotations

import importlib.util
import random
import unittest
from typing import Any

from probstat.distributions import Normal
from probstat.inference import (
    ConfidenceInterval,
    TwoByTwoMeasures,
    fit_incomplete_multivariate_normal,
    impute_multivariate_normal,
    missing_pattern,
    pool_estimates,
    two_by_two_measures,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.uq import QuantityOfInterest
from probstat.validation import ValidationExperiment
from probstat.workflow import UQStudy

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _hex(colour: Any) -> str:
    """Return an upper-case hex string for a Matplotlib colour."""
    from matplotlib.colors import to_hex

    return str(to_hex(colour)).upper()


def _title(axes: Any) -> str:
    """Return the left (assertion) title without line breaks."""
    return str(axes.get_title(loc="left")).replace("\n", " ")


def _fill_colours(axes: Any) -> set[str]:
    """Return the hex face colours of every collection on the axes."""
    return {
        _hex(colour[:3])
        for collection in axes.collections
        for colour in collection.get_facecolor()
    }


def _texts(axes: Any) -> list[str]:
    """Return every text and annotation string on the axes."""
    return [text.get_text() for text in axes.texts]


def _line(parameters: Any) -> dict[str, float]:
    """Return ``a + b t``."""
    return {"y": parameters["a"] + parameters["b"] * parameters["t"]}


def _validation_study(shift: float) -> UQStudy:
    """Return a study whose validation data are offset by ``shift``."""
    inputs = JointInput.independent(
        RandomInput("a", Normal(1.0, 0.1)), RandomInput("b", Normal(1.0, 0.05))
    )
    model = ProbabilisticModel(
        inputs, _line, (QuantityOfInterest("y", "y"),), fixed={"t": 0.0}
    )
    rng = random.Random(3)
    experiments = tuple(
        ValidationExperiment(
            f"t={t:g}",
            tuple(1.0 + t + shift + rng.gauss(0.0, 0.05) for _ in range(4)),
            {"t": t},
            measurement_uncertainty=0.05,
            provenance="synthetic",
        )
        for t in (0.5, 1.0, 2.0)
    )
    study = UQStudy(model, seed=8)
    study.validate("y", experiments, tolerance=0.2, sample_size=400)
    return study


def _incomplete_rows(n: int, seed: int) -> list[tuple[float | None, ...]]:
    """Return bivariate normal rows with the second value missing at random."""
    full = MultivariateNormal((0.0, 1.0), ((1.0, 0.6), (0.6, 1.0))).sample(
        n, seed=seed
    )
    rng = random.Random(seed)
    return [
        (x[0], None if rng.random() < (0.6 if x[0] > 0.0 else 0.1) else x[1])
        for x in full
    ]


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ValidationFigureTests(unittest.TestCase):
    """The validation stage figure and its place in the study report."""

    def tearDown(self) -> None:
        """Close every figure a test opened."""
        from probstat.visualization.theme import pyplot

        pyplot().close("all")

    def test_title_follows_the_assessment_verdict(self) -> None:
        """Consistent data give a coverage count; biased data name criteria."""
        from probstat.visualization.scoring_plots import plot_validation_predictions
        from probstat.visualization.study import _coverage_from_summary
        from probstat.visualization.theme import PORCELAIN, chart_theme

        for shift in (0.0, 1.0):
            record = _validation_study(shift).latest("validation")
            result = record.result
            with chart_theme(PORCELAIN):
                figure = plot_validation_predictions(
                    result, coverage=_coverage_from_summary(record.summary)
                )
            title = _title(figure.axes[0])
            if result.consistent:
                self.assertTrue(title.startswith("Model is consistent"))
                self.assertIn("of 12 observations", title)
            else:
                self.assertTrue(title.startswith("Model is inconsistent"))
            if shift == 1.0:
                self.assertFalse(result.consistent)
                self.assertIn("area metric exceeds 0.2 in 3 of 3", title)
            self.assertEqual(len(figure.axes), 2)
            main = figure.axes[0]
            self.assertIn(PORCELAIN.model, _fill_colours(main))
            self.assertIn(PORCELAIN.observed, _fill_colours(main))
            coverage_lines = {_hex(line.get_color()) for line in figure.axes[1].lines}
            self.assertIn(PORCELAIN.accent, coverage_lines)

    def test_coverage_records_are_rebuilt_from_the_summary(self) -> None:
        """Both default levels come back with their counts and limits."""
        from probstat.visualization.study import _coverage_from_summary

        summary = _validation_study(0.0).latest("validation").summary
        records = _coverage_from_summary(summary)
        self.assertEqual([r.nominal for r in records], [0.5, 0.9])
        for record in records:
            self.assertEqual(record.size, 12)
            self.assertAlmostEqual(record.rate, record.covered / 12)
            self.assertLessEqual(record.interval.lower, record.interval.upper)
        self.assertEqual(_coverage_from_summary({"coverage[50%]": "x"}), [])

    def test_single_panel_and_errors(self) -> None:
        """Without coverage the figure has one panel; bad levels are refused."""
        from probstat.visualization.scoring_plots import plot_validation_predictions

        result = _validation_study(0.0).latest("validation").result
        figure = plot_validation_predictions(result, pit_mean=0.5)
        self.assertEqual(len(figure.axes), 1)
        with self.assertRaises(ValueError):
            plot_validation_predictions(result, levels=(0.9, 0.5))

    def test_validation_figure_appears_in_the_study_report(self) -> None:
        """The HTML report embeds light and dark validation figures."""
        from probstat.visualization.study import _figures_for, study_html

        study = _validation_study(0.0)
        self.assertEqual(len(_figures_for(study.latest("validation"), study)), 1)
        page = study_html(study)
        section = page[page.index('id="stage-1"') :]
        self.assertIn("validation", section)
        self.assertEqual(section.count("<picture>"), 1)
        self.assertIn('media="(prefers-color-scheme: dark)"', section)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class PitBandTests(unittest.TestCase):
    """Simultaneous PIT band and honest titles."""

    def tearDown(self) -> None:
        """Close every figure a test opened."""
        from probstat.visualization.theme import pyplot

        pyplot().close("all")

    def test_simultaneous_band_is_wider_and_holds_family_wise(self) -> None:
        """All bins of uniform samples stay inside in at least 94% of runs."""
        from probstat.visualization.hierarchical_plots import (
            binomial_band,
            simultaneous_binomial_band,
        )

        low, high = binomial_band(200, 10)
        wide_low, wide_high = simultaneous_binomial_band(200, 10)
        self.assertLess(wide_low, low)
        self.assertGreater(wide_high, high)
        rng = random.Random(11)
        pointwise = simultaneous = 0
        for _ in range(500):
            counts = [0] * 10
            for _ in range(200):
                counts[int(rng.random() * 10)] += 1
            pointwise += any(c < low or c > high for c in counts)
            simultaneous += any(c < wide_low or c > wide_high for c in counts)
        self.assertLessEqual(simultaneous / 500, 0.06)
        self.assertGreater(pointwise / 500, 0.2)
        with self.assertRaises(ValueError):
            simultaneous_binomial_band(200, 1)
        with self.assertRaises(ValueError):
            simultaneous_binomial_band(200, 10, 1.0)

    def test_calibrated_sample_is_not_called_miscalibrated(self) -> None:
        """Two pointwise excursions of a uniform sample stay 'consistent'."""
        from probstat.visualization.hierarchical_plots import (
            binomial_band,
            plot_pit_histogram,
        )

        low, high = binomial_band(400, 20)
        values: list[float] = []
        for seed in range(200):
            rng = random.Random(seed)
            values = [rng.random() for _ in range(400)]
            counts = [0] * 20
            for v in values:
                counts[int(v * 20)] += 1
            if sum(c < low or c > high for c in counts) >= 2:
                break
        title = _title(plot_pit_histogram(values))
        self.assertIn("consistent with calibration", title)
        self.assertNotIn("miscalibration", title)
        legacy = _title(plot_pit_histogram(values, simultaneous=False))
        self.assertIn("pointwise", legacy)
        self.assertNotIn("miscalibration", legacy)
        piled = [0.01] * 50 + [0.99] * 50
        self.assertIn("miscalibration", _title(plot_pit_histogram(piled)))

    def test_bands_are_muted_context(self) -> None:
        """Both bands use muted ink and the uniform level the accent."""
        from probstat.visualization.hierarchical_plots import plot_pit_histogram
        from probstat.visualization.theme import GRAPHITE, chart_theme

        with chart_theme(GRAPHITE):
            axes = plot_pit_histogram([(i + 0.5) / 100 for i in range(100)])
        spans = [p for p in axes.patches if p.get_width() >= 1.0]
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(_hex(p.get_facecolor()) == GRAPHITE.muted for p in spans))
        self.assertIn(GRAPHITE.accent, {_hex(line.get_color()) for line in axes.lines})


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class MissingDataFigureTests(unittest.TestCase):
    """Pattern matrix, imputation diagnostic, and Rubin's-rules summary."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate, fit, and impute one incomplete data set."""
        cls.rows = _incomplete_rows(160, 4)
        fit = fit_incomplete_multivariate_normal(cls.rows)
        cls.imputed = impute_multivariate_normal(
            cls.rows, 5, burn_in=10, start=fit, seed=4
        )
        means = [
            sum(r[1] for r in data) / len(data) for data in cls.imputed.datasets
        ]
        cls.pooled = pool_estimates(means, [0.01] * 5, complete_data_df=159)

    def tearDown(self) -> None:
        """Close every figure a test opened."""
        from probstat.visualization.theme import pyplot

        pyplot().close("all")

    def test_pattern_matrix_counts_and_colours(self) -> None:
        """Title counts complete rows; missing cells use the observed hue."""
        from probstat.visualization.missing_plots import plot_missing_pattern
        from probstat.visualization.theme import PORCELAIN, chart_theme

        pattern = missing_pattern(self.rows)
        with chart_theme(PORCELAIN):
            axes = plot_missing_pattern(pattern, names=("x", "y"))
        title = _title(axes)
        self.assertTrue(
            title.startswith(f"{pattern.complete_cases} of 160 rows"), title
        )
        self.assertIn("y is missing most often", title)
        faces = {_hex(p.get_facecolor()) for p in axes.patches}
        self.assertIn(PORCELAIN.observed, faces)
        labels = " ".join(_texts(axes))
        for count in pattern.pattern_counts:
            self.assertIn(f"{count} rows", labels)
        with self.assertRaises(ValueError):
            plot_missing_pattern(pattern, names=("x",))
        complete = missing_pattern([(1.0, 2.0), (2.0, 3.0)])
        self.assertTrue(_title(plot_missing_pattern(complete)).startswith("No values"))

    def test_imputation_diagnostic_roles_and_title(self) -> None:
        """Observed values take the observed hue, imputations the model hue."""
        from probstat.visualization.missing_plots import plot_imputation_diagnostic
        from probstat.visualization.theme import GRAPHITE, chart_theme

        with chart_theme(GRAPHITE):
            figure = plot_imputation_diagnostic(self.rows, self.imputed)
        self.assertEqual(len(figure.axes), 1)
        colours = _fill_colours(figure.axes[0])
        self.assertIn(GRAPHITE.observed, colours)
        self.assertIn(GRAPHITE.model, colours)
        observed = sorted(r[1] for r in self.rows if r[1] is not None)
        imputed = sorted(
            data[i][1]
            for data in self.imputed.datasets
            for i, r in enumerate(self.rows)
            if r[1] is None
        )
        title = _title(figure.axes[0])
        median_gap = imputed[len(imputed) // 2] - observed[len(observed) // 2]
        if "sit" in title:
            self.assertIn("above" if median_gap > 0 else "below", title)
        with self.assertRaises(ValueError):
            plot_imputation_diagnostic(self.rows[:-1], self.imputed)
        with self.assertRaises(ValueError):
            plot_imputation_diagnostic(self.rows, self.imputed, max_variables=0)

    def test_rubin_summary_titles(self) -> None:
        """The title counts covering intervals, or reports the widening."""
        from probstat.visualization.missing_plots import plot_rubin_summary
        from probstat.visualization.theme import PORCELAIN, chart_theme

        pooled = self.pooled
        others = {"complete cases": (0.4, 0.3, 0.5)}
        truth = pooled.estimate
        with chart_theme(PORCELAIN):
            axes = plot_rubin_summary(pooled, others=others, truth=truth)
        title = _title(axes)
        self.assertIn("Multiple imputation covers the true value", title)
        self.assertIn("complete cases misses it", title)
        self.assertIn(PORCELAIN.accent, {_hex(line.get_color()) for line in axes.lines})
        bare = _title(plot_rubin_summary(pooled))
        self.assertIn("widen the multiple imputation interval", bare)
        with self.assertRaises(ValueError):
            plot_rubin_summary(pooled, others={"bad": (1.0, 2.0, 3.0)})


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ContingencyFigureTests(unittest.TestCase):
    """Adjusted-residual heatmap, diverging scale, and effect forest."""

    def tearDown(self) -> None:
        """Close every figure a test opened."""
        from probstat.visualization.theme import pyplot

        pyplot().close("all")

    def test_diverging_scale_is_two_hues_and_a_neutral_midpoint(self) -> None:
        """Anchors come from the palette and lightness is monotone per arm."""
        from matplotlib.colors import to_rgb

        from probstat.visualization.contingency_plots import (
            _relative_luminance,
            diverging_colormap,
            diverging_ramp,
        )
        from probstat.visualization.theme import GRAPHITE, PORCELAIN

        for palette in (PORCELAIN, GRAPHITE):
            negative, middle, positive = diverging_ramp(palette)
            self.assertEqual(
                (negative, middle, positive),
                (palette.series[2], palette.rule, palette.series[0]),
            )
            red, green, blue = to_rgb(middle)
            self.assertLess(max(red, green, blue) - min(red, green, blue), 0.08)
            colormap = diverging_colormap(palette)
            centre = _relative_luminance(colormap(0.5))
            arms = ((0.5, 0.375, 0.25, 0.125, 0.0), (0.5, 0.625, 0.75, 0.875, 1.0))
            for arm in arms:
                values = [_relative_luminance(colormap(t)) for t in arm]
                steps = [b - a for a, b in zip(values, values[1:], strict=False)]
                direction = 1.0 if values[-1] > centre else -1.0
                self.assertTrue(all(direction * step > 0.0 for step in steps))

    def test_heatmap_titles_and_annotations(self) -> None:
        """Strong association names the largest cell; flagged cells are labelled."""
        from probstat.inference import contingency_residuals
        from probstat.visualization.contingency_plots import plot_adjusted_residuals

        table = [[56, 71, 12, 8], [40, 52, 30, 22], [18, 30, 41, 47]]
        axes = plot_adjusted_residuals(
            table, row_labels=("low", "mid", "high"), row_name="Exposure"
        )
        residuals = contingency_residuals(table)
        flagged = sum(abs(v) > 2.0 for row in residuals.adjusted for v in row)
        title = _title(axes)
        self.assertTrue(title.startswith("High / Column 4 holds more counts"), title)
        self.assertIn(f"{flagged} of 12 cells", title)
        cell_texts = [t for t in _texts(axes) if t.startswith(("+", "-"))]
        self.assertEqual(len(cell_texts), flagged)
        flat = plot_adjusted_residuals([[20, 21], [19, 20]])
        self.assertTrue(_title(flat).startswith("No cell departs"))
        self.assertFalse([t for t in _texts(flat) if t.startswith(("+", "-"))])
        with self.assertRaises(ValueError):
            plot_adjusted_residuals(table, threshold=0.0)
        with self.assertRaises(ValueError):
            plot_adjusted_residuals(table, row_labels=("a", "b"))

    def test_effect_forest_titles(self) -> None:
        """Agreeing, null, and disagreeing measures get different titles."""
        from probstat.visualization.contingency_plots import plot_effect_measures
        from probstat.visualization.theme import PORCELAIN, chart_theme

        with chart_theme(PORCELAIN):
            figure = plot_effect_measures(
                two_by_two_measures([[30, 70], [15, 85]]), groups=("a", "b")
            )
        self.assertIn("every 95% interval excludes no effect", _title(figure.axes[0]))
        accents = {_hex(item.get_color()) for ax in figure.axes for item in ax.lines}
        self.assertIn(PORCELAIN.accent, accents)
        null = plot_effect_measures(two_by_two_measures([[20, 80], [18, 82]]))
        self.assertTrue(_title(null.axes[0]).startswith("No measure separates"))
        base = two_by_two_measures([[30, 70], [15, 85]])
        mixed = TwoByTwoMeasures(
            base.table, base.risk_exposed, base.risk_unexposed, base.risk_difference,
            base.risk_difference_interval, base.risk_ratio,
            ConfidenceInterval(0.9, 3.0, 0.95, "Katz log interval"),
            base.odds_ratio, base.odds_ratio_interval, False, 0.95,
        )  # fmt: skip
        title = _title(plot_effect_measures(mixed).axes[0])
        self.assertIn("The measures disagree", title)
        self.assertIn("the RR interval does not", title)
        zero = plot_effect_measures(two_by_two_measures([[0, 20], [6, 14]]))
        self.assertEqual(len(zero.axes[1].get_yticks()), 1)


if __name__ == "__main__":
    unittest.main()
