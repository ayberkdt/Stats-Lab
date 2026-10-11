"""Smoke tests for the extremes, design, causal, and information figures."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _hex(colour: Any) -> str:
    """Return an upper-case hex string for a Matplotlib colour."""
    from matplotlib.colors import to_hex

    return str(to_hex(colour)).upper()


def _line_colours(axes: Any) -> set[str]:
    """Return the hex colours of every line on the axes."""
    return {_hex(item.get_color()) for item in axes.get_lines()}


def _text_strings(axes: Any) -> list[str]:
    """Return the strings of every annotation on the axes."""
    return [item.get_text() for item in axes.texts]


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class AnalysisFigureTests(unittest.TestCase):
    """Rendering, computed titles, and palette roles of the analysis figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once."""
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_hazen_return_periods(self) -> None:
        """Empirical return periods use ``1/(1 - (i - 1/2)/n)``."""
        from probstat.visualization.extremes_plots import hazen_return_periods

        self.assertEqual(hazen_return_periods(2), (1.0 / 0.75, 1.0 / 0.25))
        with self.assertRaises(ValueError):
            hazen_return_periods(0)

    def test_order_statistic_band_brackets_hazen_positions(self) -> None:
        """Beta order-statistic limits bracket ``(i - 1/2)/n`` and are ordered."""
        from probstat.visualization.extremes_plots import order_statistic_band

        lower, upper = order_statistic_band(20, 0.9)
        for i, (low, high) in enumerate(zip(lower, upper, strict=True), start=1):
            self.assertLess(low, (i - 0.5) / 20)
            self.assertGreater(high, (i - 0.5) / 20)
        self.assertTrue(all(b > a for a, b in zip(lower, lower[1:], strict=False)))
        with self.assertRaises(ValueError):
            order_statistic_band(5, 1.0)

    def test_return_level_plot_titles_headline_and_uses_roles(self) -> None:
        """The title quotes the headline level; colours follow the roles."""
        from probstat.risk import ReturnLevel
        from probstat.visualization import active_palette
        from probstat.visualization.extremes_plots import plot_return_levels

        periods = (2.0, 5.0, 10.0, 50.0, 100.0)
        levels = [
            ReturnLevel(t, 10.0 + math.log(t), 0.5, 9.0 + math.log(t),
                        11.0 + 1.2 * math.log(t), 0.95, "delta")
            for t in periods
        ]  # fmt: skip
        observed = [10.0 + 0.2 * k for k in range(20)]
        axes = plot_return_levels(
            levels, observed, truth=[10.0 + math.log(t) for t in periods]
        )
        palette = active_palette()
        title = axes.get_title(loc="left")
        self.assertIn("The 100-year level is 14.6", title)
        self.assertIn(f"{9.0 + math.log(100.0):.3g}", title)
        colours = _line_colours(axes)
        self.assertIn(palette.model.upper(), colours)
        self.assertIn(palette.accent.upper(), colours)
        self.assertIn(_hex(axes.collections[-1].get_facecolor()[0]), {
            palette.observed.upper()
        })
        self.assertIn("record: 20 years", _text_strings(axes))
        with self.assertRaises(ValueError):
            plot_return_levels(levels[::-1], observed)

    def test_return_level_plot_handles_open_profile_interval(self) -> None:
        """An infinite upper limit is left out of the band and stated."""
        from probstat.risk import ReturnLevel
        from probstat.visualization.extremes_plots import plot_return_levels

        levels = [
            ReturnLevel(2.0, 5.0, math.nan, 4.0, 6.0, 0.95, "profile"),
            ReturnLevel(10.0, 7.0, math.nan, 6.0, 9.0, 0.95, "profile"),
            ReturnLevel(100.0, 9.0, math.nan, 7.0, math.inf, 0.95, "profile"),
        ]
        axes = plot_return_levels(levels, [4.0, 5.0, 6.0], headline_period=100.0)
        self.assertIn("unbounded", axes.get_title(loc="left"))

    def test_mean_residual_life_title_counts_departures(self) -> None:
        """With a GPD overlay the title counts points off the implied line."""
        from probstat.distributions import Exponential
        from probstat.risk import fit_gpd, mean_residual_life, threshold_excesses

        law = Exponential(1.0)
        sample = [law.quantile((k + 0.5) / 400) for k in range(400)]
        mrl = mean_residual_life(sample)
        gpd = fit_gpd(threshold_excesses(sample, 1.0), threshold=1.0)
        from probstat.visualization.extremes_plots import plot_mean_residual_life

        axes = plot_mean_residual_life(mrl, threshold=1.0, gpd=gpd)
        title = axes.get_title(loc="left")
        self.assertTrue(
            title.startswith("Above u = 1") or "thresholds above u = 1" in title
        )
        self.assertTrue(any("GPD line" in text for text in _text_strings(axes)))
        with self.assertRaises(ValueError):
            plot_mean_residual_life(mrl, gpd=gpd)

    def test_tail_quantile_plot_counts_points_outside_band(self) -> None:
        """A sample drawn at model quantiles has no point outside the band."""
        from probstat.distributions import GeneralizedPareto
        from probstat.visualization.extremes_plots import plot_tail_quantiles

        law = GeneralizedPareto(0.0, 1.0, 0.1)
        sample = [law.quantile((k + 0.5) / 60) for k in range(60)]
        axes = plot_tail_quantiles(sample, law)
        self.assertTrue(axes.get_title(loc="left").startswith("0 of 60 tail points"))

    def test_optimal_design_plot_certifies_and_bounds(self) -> None:
        """The title certifies an optimal design and bounds a poor one."""
        from probstat.design import ApproximateDesign, polynomial_basis
        from probstat.visualization import active_palette
        from probstat.visualization.design_plots import plot_optimal_design

        basis = polynomial_basis(2)
        region = [-1.0 + 0.05 * i for i in range(41)]
        optimal = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
        figure = plot_optimal_design(
            optimal, basis, region, comparison=ApproximateDesign.uniform(region)
        )
        top, bottom = figure.axes
        self.assertIn("is D-optimal", top.get_title(loc="left"))
        self.assertIn(active_palette().accent.upper(), _line_colours(bottom))
        poor = ApproximateDesign.uniform((-0.5, 0.0, 0.5))
        title = plot_optimal_design(poor, basis, region).axes[0].get_title(loc="left")
        self.assertIn("D-efficiency is at least", title)
        with self.assertRaises(ValueError):
            plot_optimal_design(ApproximateDesign.uniform(((0.0, 1.0),)), basis, region)

    def test_greedy_design_plot_names_selected_conditions(self) -> None:
        """The title lists the greedy picks and the final EIG."""
        from probstat.design import BayesianDesignProblem, greedy_design
        from probstat.distributions import Normal
        from probstat.inverse import NoiseModel
        from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
        from probstat.uq import QuantityOfInterest
        from probstat.visualization.design_plots import plot_greedy_design

        def line_model(parameters: dict[str, float]) -> dict[str, float]:
            """Return a straight line ``a + b x``."""
            return {"y": parameters["a"] + parameters["b"] * parameters["x"]}

        prior = JointInput.independent(
            RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 2.0))
        )
        model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "y"),))
        problem = BayesianDesignProblem(model, (NoiseModel("y", standard_deviation=1),))
        xs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        result = greedy_design(
            problem, [{"x": x} for x in xs], 2, method="linearized"
        )
        axes = plot_greedy_design(result, xs, factor="x")
        title = axes.get_title(loc="left")
        self.assertIn("picks x = -1, 1", title)
        self.assertIn(f"{result.information_gains[-1]:.2f}", title)
        with self.assertRaises(ValueError):
            plot_greedy_design(result, xs[:3])

    def test_effect_forest_counts_covering_intervals(self) -> None:
        """The forest title counts covering intervals and names the worst miss."""
        from probstat.visualization.causal_plots import plot_effect_estimates

        axes = plot_effect_estimates(
            {
                "naive": (2.8, 2.5, 3.1),
                "AIPW": (2.05, 1.9, 2.2),
                "regression": (2.0, None, None),
            },
            truth=2.0,
        )
        title = axes.get_title(loc="left")
        self.assertEqual(
            title, "1 of 2 intervals cover the true effect 2; naive misses it by 0.80"
        )
        neutral = plot_effect_estimates({"a": (1.0, None, None), "b": (3.0, 2, 4)})
        self.assertIn("range from 1 (a) to 3 (b)", neutral.get_title(loc="left"))
        with self.assertRaises(ValueError):
            plot_effect_estimates({})

    def test_balance_plot_reports_largest_imbalance(self) -> None:
        """The Love-plot title quotes raw and weighted maxima."""
        from probstat.causal import BalanceTable, CovariateBalance
        from probstat.visualization.causal_plots import plot_covariate_balance

        before = BalanceTable(
            (CovariateBalance("x1", 1.0, 0.4, 0.6), CovariateBalance("x2", 0, 0, -0.3)),
            weighted=False,
        )
        after = BalanceTable(
            (CovariateBalance("x1", 0, 0, 0.02), CovariateBalance("x2", 0, 0, 0.15)),
            weighted=True,
        )
        title = plot_covariate_balance(before, after).get_title(loc="left")
        self.assertEqual(
            title, "1 of 2 covariates stay at or above 0.1 SD after weighting"
        )
        good = BalanceTable(
            (CovariateBalance("x1", 0, 0, 0.02), CovariateBalance("x2", 0, 0, -0.05)),
            weighted=True,
        )
        self.assertIn(
            "from 0.60 to 0.05",
            plot_covariate_balance(before, good).get_title(loc="left"),
        )

    def test_e_value_curve_passes_through_the_e_value(self) -> None:
        """The explain-away curve meets the diagonal at the E-value."""
        from probstat.causal import confounding_bias_factor, e_value
        from probstat.visualization.causal_plots import (
            explain_away_curve,
            plot_e_value_contour,
        )

        result = e_value(3.9, lower=1.8, upper=8.7)
        (curve,) = explain_away_curve(3.9, [result.point])
        self.assertAlmostEqual(curve, result.point, places=10)
        for eu in (5.0, 9.0):
            (ud,) = explain_away_curve(3.9, [eu])
            self.assertAlmostEqual(confounding_bias_factor(eu, ud), 3.9, places=10)
        self.assertEqual(explain_away_curve(0.5, [1.5]), (math.inf,))
        axes = plot_e_value_contour(result, limit_ratio=1.8)
        self.assertIn("RR >= 7.26", axes.get_title(loc="left"))
        with self.assertRaises(ValueError):
            plot_e_value_contour(e_value(1.0))

    def test_information_plots_report_computed_numbers(self) -> None:
        """Estimator-check and decomposition titles quote computed values."""
        from probstat.information import mutual_information
        from probstat.visualization.information_plots import (
            plot_entropy_decomposition,
            plot_estimator_check,
        )

        axes = plot_estimator_check(
            (0.0, 0.5), ((0.01, -0.01), (0.2, 0.18)), lambda rho: 0.5 * rho * rho
        )
        self.assertIn("within 0.065 nats", axes.get_title(loc="left"))
        self.assertIn("rho = 0.5", axes.get_title(loc="left"))
        joint = ((0.30, 0.10, 0.05), (0.05, 0.20, 0.30))
        axes = plot_entropy_decomposition(joint)
        info = mutual_information(joint, unit="bits")
        self.assertIn(f"I = {info:.3f} bits", axes.get_title(loc="left"))
        self.assertIn("31%", axes.get_title(loc="left"))
        with self.assertRaises(ValueError):
            plot_estimator_check((0.0,), ((1.0,),), abs)

    def test_figures_render_in_both_palettes(self) -> None:
        """Every analysis figure saves under porcelain and graphite."""
        from probstat.causal import e_value
        from probstat.visualization import GRAPHITE, PORCELAIN, chart_theme, save_figure
        from probstat.visualization.causal_plots import plot_e_value_contour
        from probstat.visualization.information_plots import (
            plot_entropy_decomposition,
        )

        with tempfile.TemporaryDirectory() as directory:
            for palette in (PORCELAIN, GRAPHITE):
                with chart_theme(palette):
                    produced = [
                        plot_e_value_contour(e_value(2.0, lower=1.5, upper=2.5)),
                        plot_entropy_decomposition(((0.25, 0.25), (0.1, 0.4))),
                    ]
                for index, axes in enumerate(produced):
                    self.assertEqual(_hex(axes.figure.get_facecolor()), palette.canvas)
                    path = save_figure(
                        axes.figure, Path(directory) / f"{palette.name}-{index}.png"
                    )
                    self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
