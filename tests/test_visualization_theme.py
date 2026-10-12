"""Smoke tests for the chart theme and the themed plotting functions."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ThemeTests(unittest.TestCase):
    """Theme tokens and rules."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once."""
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_palettes_have_six_distinct_series(self) -> None:
        """Both palettes define six distinct series hues and a separate accent."""
        from probstat.visualization import GRAPHITE, PORCELAIN

        for palette in (PORCELAIN, GRAPHITE):
            self.assertEqual(len(set(palette.series)), 6)
            self.assertNotIn(palette.accent, palette.series)
            self.assertEqual(palette.observed, palette.series[0])
            self.assertEqual(palette.model, palette.series[1])

    def test_continuous_functions_use_two_panels_not_twin_axes(self) -> None:
        """Density and probability live in separate single-axis panels."""
        from probstat.distributions import Exponential
        from probstat.visualization import plot_continuous_functions

        axes = plot_continuous_functions(Exponential(1.2), lower=0.0, upper=4.0)
        figure_axes = axes.figure.axes
        self.assertEqual(len(figure_axes), 2)
        self.assertTrue(all(not ax.spines["right"].get_visible() for ax in figure_axes))

    def test_plots_have_assertion_titles_and_save(self) -> None:
        """Every foundation and UQ plot sets a left title and can be saved."""
        from probstat.distributions import Bernoulli, Normal, Poisson
        from probstat.monte_carlo import convergence_trace
        from probstat.visualization import (
            plot_bernoulli_pmf,
            plot_integer_pmf,
            plot_monte_carlo_convergence,
            plot_normal_pdf,
            save_figure,
        )
        from probstat.visualization.uq import (
            plot_design_comparison,
            plot_method_comparison,
            plot_output_distribution,
            plot_sobol_indices,
        )

        contributions = tuple(4.0 * (i % 4 != 0) for i in range(400))
        produced = [
            plot_bernoulli_pmf(Bernoulli(0.3)),
            plot_integer_pmf(Poisson(3.0), lower=0, upper=10),
            plot_normal_pdf(Normal(0.0, 1.0)),
            plot_monte_carlo_convergence(
                convergence_trace(contributions, (50, 100, 200, 400)),
                reference=3.0,
            ),
            plot_method_comparison((("a", 1.0, 0.9, 1.1), ("b", 1.05, None, None))),
            plot_sobol_indices(("x1", "x2"), (0.3, 0.0), (0.5, 0.4)),
            plot_output_distribution([math.sin(i) for i in range(300)], threshold=0.5),
            plot_design_comparison({"mc": (1.0, 0.2), "sobol": (1.0, 0.01)}, size=64),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, axes in enumerate(produced):
                self.assertTrue(axes.get_title(loc="left"))
                path = save_figure(axes.figure, Path(directory) / f"plot-{index}.png")
                self.assertGreater(path.stat().st_size, 1000)

    def test_new_workflow_plots_render(self) -> None:
        """Inference, filtering, surrogate, field, and risk plots all render."""
        from types import SimpleNamespace

        from probstat.visualization.inference_plots import (
            plot_chain_traces,
            plot_filter_track,
            plot_posterior_marginals,
            plot_surrogate_parity,
        )
        from probstat.visualization.processes_plots import (
            plot_field_realizations,
            plot_kl_spectrum,
        )
        from probstat.visualization.uq import (
            plot_calibration_prediction,
            plot_exceedance_curves,
            plot_form_geometry,
            plot_loss_exceedance,
            plot_reliability_comparison,
        )

        grid = [0.1 * k for k in range(30)]
        wave = [math.sin(x) for x in grid]
        rows = (
            SimpleNamespace(
                method="FORM",
                probability=0.01,
                evaluations=40,
                coefficient_of_variation=None,
            ),
            SimpleNamespace(
                method="MC",
                probability=0.012,
                evaluations=10000,
                coefficient_of_variation=0.09,
            ),
        )
        produced = [
            plot_chain_traces([wave, [w + 0.01 for w in wave]], name="k", rhat=1.0),
            plot_filter_track(grid, wave, [0.1] * 30, truth=wave, name="x"),
            plot_surrogate_parity(wave, [w * 1.01 for w in wave]),
            plot_field_realizations(grid, [wave, [-w for w in wave]]),
            plot_kl_spectrum([0.5, 0.3, 0.1, 0.05, 0.05], 3),
            plot_calibration_prediction(
                grid[:5],
                wave[:5],
                grid,
                wave,
                ([w - 0.1 for w in wave], [w + 0.1 for w in wave]),
                ([w - 0.2 for w in wave], [w + 0.2 for w in wave]),
                calibration_limit=1.0,
            ),
            plot_exceedance_curves({"model": [(x, math.exp(-x)) for x in grid]}),
            plot_form_geometry(lambda a, b: 3.0 - a - b, (1.5, 1.5), 2.1213),
            plot_loss_exceedance(
                [float(k % 7) for k in range(200)], {0.95: 6.0}, {0.95: 6.0}
            ),
            plot_reliability_comparison(rows, reference=0.011),
        ]
        figure = plot_posterior_marginals({"a": wave, "b": grid}, truth={"a": 0.0})
        self.assertEqual(len(figure.axes), 2)
        for axes in produced:
            self.assertTrue(axes.get_title(loc="left"))

    def test_uncertainty_encodings_render(self) -> None:
        """Fan, spaghetti, quantile dots, and ECDF band plots render."""
        from probstat.visualization.uncertainty import (
            dkw_epsilon,
            plot_ecdf_band,
            plot_fan_chart,
            plot_quantile_dots,
            plot_spaghetti,
        )

        self.assertAlmostEqual(
            dkw_epsilon(120, 0.95), math.sqrt(math.log(40.0) / 240.0)
        )
        times = [0.1 * k for k in range(11)]
        paths = [[math.sin(t + 0.3 * j) + 0.1 * j for t in times] for j in range(20)]
        terminal = [path[-1] for path in paths]
        for axes in (
            plot_fan_chart(times, paths),
            plot_spaghetti(times, paths),
            plot_quantile_dots(terminal, dots=20, threshold=1.0),
            plot_ecdf_band(terminal),
        ):
            self.assertTrue(axes.get_title(loc="left"))

    def test_statistical_model_plots_render(self) -> None:
        """Correlogram, forecast, survival, and regression panels render."""
        from probstat.visualization.statistics_plots import (
            plot_correlogram,
            plot_forecast,
            plot_regression_diagnostics,
            plot_survival_curves,
        )

        produced = [
            plot_correlogram([1.0, 0.5, 0.2, 0.05], 0.1, kind="ACF"),
            plot_forecast([1.0, 2.0, 1.5], [1.4, 1.3], [0.4, 0.1], [2.4, 2.5]),
            plot_survival_curves(
                {
                    "a": (
                        (1.0, 2.0, 3.0),
                        (0.8, 0.5, 0.2),
                        (0.6, 0.3, 0.05),
                        (0.95, 0.7, 0.4),
                    )
                }
            ),
        ]
        for axes in produced:
            self.assertTrue(axes.get_title(loc="left"))
        figure = plot_regression_diagnostics(
            [1.0, 2.0, 3.0, 4.0], [0.1, -0.2, 0.3, -0.1]
        )
        self.assertEqual(len(figure.axes), 2)

    def test_distribution_comparison_title_and_limits(self) -> None:
        """Violins carry a data-derived title; too many groups are refused."""
        from probstat.visualization.statistics_plots import (
            plot_distribution_comparison,
        )

        groups = {
            "low": [1.0, 1.2, 0.8, 1.1, 0.9, 1.0],
            "high": [5.0, 6.0, 4.0, 5.5, 4.5, 5.0],
        }
        axes = plot_distribution_comparison(groups, reference=3.0)
        title = axes.get_title(loc="left")
        self.assertTrue(title.startswith("High has the highest median (5)"))
        self.assertIn("5.0x wider than low", title)
        with self.assertRaises(ValueError):
            plot_distribution_comparison({str(i): [0.0, 1.0] for i in range(7)})

    def test_monotone_cubic_interpolates_without_overshoot(self) -> None:
        """Fritsch-Carlson passes through the data and keeps monotone data monotone."""
        from probstat.visualization.theme import monotone_cubic

        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.1, 0.2, 5.0, 5.1]
        xs, ys = monotone_cubic(x, y, samples=20)
        for xi, yi in zip(x, y, strict=True):
            self.assertIn(xi, xs)
            self.assertAlmostEqual(ys[xs.index(xi)], yi, places=12)
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(ys, ys[1:], strict=False)))
        self.assertGreaterEqual(min(ys), 0.0)
        self.assertLessEqual(max(ys), 5.1)
        with self.assertRaises(ValueError):
            monotone_cubic([0.0, 0.0], [1.0, 2.0])

    def test_band_opacity_is_capped_and_area_fades(self) -> None:
        """Bands never exceed 0.18 opacity; the gradient area is clipped."""
        from probstat.visualization.theme import (
            MAX_BAND_ALPHA,
            band,
            gradient_area,
            new_axes,
        )

        axes = new_axes()
        collection = band(
            axes,
            [0.0, 1.0, 2.0],
            [0.0, 0.1, 0.2],
            [1.0, 1.2, 1.1],
            color="#178F6E",
            alpha=0.5,
            smooth=True,
        )
        self.assertLessEqual(collection.get_alpha(), MAX_BAND_ALPHA)
        axes.set_ylim(-1.0, 2.0)
        image = gradient_area(axes, [0.0, 1.0, 2.0], [1.0, 1.5, 0.5], color="#178F6E")
        self.assertIsNotNone(image.get_clip_path())

    def test_dark_theme_changes_canvas(self) -> None:
        """The graphite context paints the dark canvas."""
        from matplotlib.colors import to_hex

        from probstat.distributions import Bernoulli
        from probstat.visualization import GRAPHITE, chart_theme, plot_bernoulli_pmf

        with chart_theme(GRAPHITE):
            axes = plot_bernoulli_pmf(Bernoulli(0.5))
        self.assertEqual(to_hex(axes.figure.get_facecolor()).upper(), GRAPHITE.canvas)


if __name__ == "__main__":
    unittest.main()
