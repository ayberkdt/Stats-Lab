"""Smoke tests for the adaptive-reliability, scoring, and transport figures."""

from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from collections.abc import Callable, Mapping
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


def _scatter_colours(axes: Any) -> set[str]:
    """Return the hex face colours of every collection on the axes."""
    colours: set[str] = set()
    for collection in axes.collections:
        for colour in collection.get_facecolor():
            colours.add(_hex(colour[:3]))
    return colours


def _title(axes: Any) -> str:
    """Return the left (assertion) title without line breaks."""
    return str(axes.get_title(loc="left")).replace("\n", " ")


def _state(function: Callable[[tuple[float, ...]], float], dimension: int) -> Any:
    """Wrap ``G(u)`` on independent standard normal inputs as a limit state."""
    from probstat.distributions import Normal
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.reliability import ModelLimitState
    from probstat.uq import QuantityOfInterest

    names = tuple(f"u{i + 1}" for i in range(dimension))
    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from named inputs."""
        return {"y": function(tuple(p[name] for name in names))}

    model = ProbabilisticModel(inputs, response, (QuantityOfInterest("y", "G"),))
    return ModelLimitState(model, "y", 0.0, "below")


def _two_planes(u: tuple[float, ...]) -> float:
    """Return a series system of two planes at beta 3.0 and 3.2."""
    return min(3.0 - u[0], 3.2 - (-0.6 * u[0] + 0.8 * u[1]))


def _kinked(u: tuple[float, ...]) -> float:
    """Return a kinked limit state that needs a few AK-MCS learning steps."""
    return 2.0 - u[0] + 0.5 * abs(u[1])


def _edge_colours(axes: Any) -> set[str]:
    """Return the hex edge colours of every collection on the axes."""
    return {
        _hex(colour[:3])
        for collection in axes.collections
        for colour in collection.get_edgecolor()
    }


def _grid(
    function: Callable[[tuple[float, ...]], float], extent: float, nodes: int
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate ``G`` on a square grid."""
    axis = [-extent + 2.0 * extent * k / (nodes - 1) for k in range(nodes)]
    return axis, axis, [[function((x, y)) for x in axis] for y in axis]


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ReliabilityFigureTests(unittest.TestCase):
    """Design-point map and AK-MCS figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend and run the reliability methods."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.reliability import (
            ak_mcs,
            find_design_points,
            mixture_importance_sampling,
            multi_point_form,
        )

        state = _state(_two_planes, 2)
        cls.points = find_design_points(state, 3)
        cls.combined = multi_point_form(cls.points)
        cls.mixture = mixture_importance_sampling(state, cls.points, 300, seed=7)
        cls.ak = ak_mcs(
            _state(_kinked, 2),
            400,
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=3,
        )

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_design_point_map_titles_count_and_ratio(self) -> None:
        """The title names both betas and the multi-point/one-point ratio."""
        from probstat.visualization import active_palette
        from probstat.visualization.reliability_plots import plot_design_points

        rng = random.Random(1)
        samples = [(rng.gauss(0.0, 2.0), rng.gauss(0.0, 2.0)) for _ in range(200)]
        failed = [_two_planes(u) <= 0.0 for u in samples]
        axes = plot_design_points(
            self.points,
            limit_state_grid=_grid(_two_planes, 5.0, 41),
            samples=samples,
            failed=failed,
            combined=self.combined,
            mixture=self.mixture,
        )
        ratio = self.combined.probability / self.points.forms[0].probability
        title = _title(axes)
        self.assertIn("2 design points at beta = 3.00, 3.20", title)
        self.assertIn(f"{ratio:.1f}x", title)
        palette = active_palette()
        self.assertIn(palette.model, _line_colours(axes))
        self.assertIn(palette.observed, _scatter_colours(axes))
        self.assertIn(palette.accent, _edge_colours(axes))

    def test_design_point_map_rejects_bad_input(self) -> None:
        """Samples need flags, grids need a sign change, spaces must be 2-D."""
        from probstat.visualization.reliability_plots import plot_design_points

        with self.assertRaises(ValueError):
            plot_design_points(self.points, samples=[(0.0, 0.0)])
        with self.assertRaises(ValueError):
            plot_design_points(
                self.points, limit_state_grid=([0.0, 1.0], [0.0, 1.0], [[1, 1], [1, 1]])
            )
        with self.assertRaises(ValueError):
            plot_design_points([])

    def _population(self) -> list[tuple[float, float]]:
        """Redraw the AK-MCS population from the run's seed."""
        from probstat.foundations.random_streams import resolve_stream

        generator = resolve_stream(3).child("ak-mcs", "population").generator()
        return [
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
            for _ in range(self.ak.population)
        ]

    def test_ak_mcs_classification_reports_calls_and_signs(self) -> None:
        """The title quotes the model calls; training points use the model hue."""
        from probstat.visualization import active_palette
        from probstat.visualization.reliability_plots import (
            plot_ak_mcs_classification,
        )

        population = self._population()
        predicted = sum(1 for u in population if self.ak.surrogate.predict(u) <= 0.0)
        self.assertEqual(predicted, self.ak.predicted_failures)
        axes = plot_ak_mcs_classification(
            self.ak, population, limit_state_grid=_grid(_kinked, 4.0, 21)
        )
        title = _title(axes)
        self.assertIn(f"{self.ak.evaluations} model calls", title)
        self.assertIn(f"{self.ak.population:,} points", title)
        if self.ak.misclassified:
            self.assertIn(f"{self.ak.misclassified} differ from the true sign", title)
        palette = active_palette()
        self.assertIn(palette.model, _edge_colours(axes))
        self.assertIn(palette.accent, _edge_colours(axes))
        with self.assertRaises(ValueError):
            plot_ak_mcs_classification(self.ak, population[:-1])
        with self.assertRaises(ValueError):
            plot_ak_mcs_classification(self.ak, population, learning_threshold=0.0)

    def test_ak_mcs_convergence_has_two_panels_and_threshold(self) -> None:
        """Two panels share x; the bottom one carries the accent threshold."""
        from probstat.visualization import active_palette
        from probstat.visualization.reliability_plots import plot_ak_mcs_convergence

        figure = plot_ak_mcs_convergence(self.ak)
        top, bottom = figure.axes
        self.assertIn(f"{self.ak.added} added points", _title(top))
        self.assertIn(active_palette().accent, _line_colours(bottom))
        self.assertEqual(bottom.get_yscale(), "log")
        self.assertIs(top.get_shared_x_axes().joined(top, bottom), True)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ScoringFigureTests(unittest.TestCase):
    """Reliability diagram, score comparison, coverage, and shift figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend and simulate forecasts."""
        import matplotlib

        matplotlib.use("Agg")
        rng = random.Random(11)
        cls.forecasts = [rng.random() for _ in range(600)]
        cls.reliable = [rng.random() < f for f in cls.forecasts]
        cls.sharp = [
            rng.random() < 0.5 + 0.5 * (f - 0.5) for f in cls.forecasts
        ]

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_reliability_diagram_titles_follow_the_bands(self) -> None:
        """Reliable forecasts pass; flattened outcomes read as overconfident."""
        from probstat.validation import reliability_diagram
        from probstat.visualization import active_palette
        from probstat.visualization.scoring_plots import plot_reliability_diagram

        good = reliability_diagram(
            self.forecasts, self.reliable, bins=5, consistency_level=0.99,
            resamples=300, seed=1,
        )  # fmt: skip
        bad = reliability_diagram(
            self.forecasts, self.sharp, bins=5, consistency_level=0.95,
            resamples=300, seed=1,
        )  # fmt: skip
        figure = plot_reliability_diagram(good)
        top = figure.axes[0]
        if not good.outside_bands:
            self.assertIn("are reliable", _title(top))
        self.assertIn(active_palette().accent, _line_colours(top))
        self.assertIn(active_palette().observed, _line_colours(top))
        title = _title(plot_reliability_diagram(bad, event="rain").axes[0])
        self.assertIn(f"{len(bad.outside_bands)} of 5 bins", title)
        self.assertIn("overconfident", title)
        empty = reliability_diagram([0.1, 0.12], [True, False], bins=5)
        with self.assertRaises(ValueError):
            plot_reliability_diagram(empty)

    def test_score_comparison_reports_paired_difference(self) -> None:
        """The paired difference of the two best forecasters is quoted."""
        from probstat.visualization.scoring_plots import plot_score_comparison

        rng = random.Random(4)
        base = [abs(rng.gauss(0.0, 1.0)) for _ in range(300)]
        scores = {
            "sharp": base,
            "blunt": [b + 0.2 + 0.01 * rng.gauss(0.0, 1.0) for b in base],
            "noisy": [b + 0.5 for b in base],
        }
        axes = plot_score_comparison(scores)
        title = _title(axes)
        self.assertTrue(title.startswith("Sharp has the lowest mean CRPS"))
        self.assertIn("blunt is worse by 0.2", title)
        self.assertEqual(axes.get_yticklabels()[0].get_text(), "sharp")
        with self.assertRaises(ValueError):
            plot_score_comparison({"one": base})
        with self.assertRaises(ValueError):
            plot_score_comparison(scores, confidence=1.0)

    def test_coverage_plot_names_worst_and_consistent(self) -> None:
        """Too-narrow PIT values undercover; uniform ones stay consistent."""
        from probstat.validation import pit_coverage
        from probstat.visualization import active_palette
        from probstat.visualization.scoring_plots import plot_coverage

        rng = random.Random(9)
        uniform = [rng.random() for _ in range(800)]
        narrow = [min(1.0, max(0.0, 0.5 + 2.0 * (u - 0.5))) for u in uniform]
        levels = (0.5, 0.8, 0.9)
        reports = {
            "calibrated": pit_coverage(uniform, levels),
            "narrow": pit_coverage(narrow, levels),
        }
        axes = plot_coverage(reports)
        title = _title(axes)
        self.assertTrue(title.startswith("Narrow covers"))
        if all(r.consistent for r in reports["calibrated"]):
            self.assertIn("calibrated stays consistent", title)
        self.assertIn(active_palette().accent, _line_colours(axes))
        with self.assertRaises(ValueError):
            plot_coverage({})
        with self.assertRaises(ValueError):
            plot_coverage({str(k): reports["narrow"] for k in range(5)})

    def test_shift_plot_marks_the_ks_distance(self) -> None:
        """The marked gap equals the test statistic; a foreign result fails."""
        from probstat.inference import kolmogorov_smirnov_two_sample_test
        from probstat.visualization.scoring_plots import ks_gap, plot_distribution_shift

        rng = random.Random(2)
        first = [rng.gauss(0.0, 1.0) for _ in range(120)]
        second = [rng.gauss(1.0, 1.0) for _ in range(150)]
        result = kolmogorov_smirnov_two_sample_test(first, second)
        _, distance, _, _ = ks_gap(first, second)
        self.assertAlmostEqual(distance, result.statistic, places=12)
        axes = plot_distribution_shift(
            first, second, names=("before", "after"), result=result
        )
        self.assertIn(f"KS distance {distance:.3f}", _title(axes))
        self.assertTrue(_title(axes).startswith("After has shifted"))
        other = kolmogorov_smirnov_two_sample_test(first, first[:60])
        with self.assertRaises(ValueError):
            plot_distribution_shift(first, second, result=other)
        with self.assertRaises(ValueError):
            ks_gap([], second)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class TransportFigureTests(unittest.TestCase):
    """Plan, Sinkhorn, quantile, and geodesic figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend and solve a small 1-D problem."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.information import exact_transport, power_cost_matrix, sinkhorn

        cls.xs = [0.0, 1.0, 2.0, 3.0]
        cls.ys = [0.5, 1.5, 3.5]
        cls.a = [0.25] * 4
        cls.b = [0.3, 0.3, 0.4]
        cost = power_cost_matrix(cls.xs, cls.ys, p=2.0)
        cls.exact = exact_transport(cls.a, cls.b, cost)
        cls.runs = [sinkhorn(cls.a, cls.b, cost, epsilon=e) for e in (1.0, 0.2)]

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plan_lines_are_monotone_for_the_optimal_plan(self) -> None:
        """The exact plan never crosses; a crossing coupling is counted."""
        from probstat.visualization import active_palette
        from probstat.visualization.transport_plots import (
            crossing_pairs,
            plot_transport_lines,
        )

        axes = plot_transport_lines(self.exact, self.xs, self.ys)
        self.assertIn("monotonically", _title(axes))
        self.assertIn(f"cost {self.exact.cost:.3f}", _title(axes))
        self.assertIn(active_palette().observed, _scatter_colours(axes))
        swapped = [[0.0, 0.5], [0.5, 0.0]]
        self.assertEqual(crossing_pairs(swapped, [0.0, 1.0], [0.0, 1.0]), 1)
        self.assertIn("cross", _title(plot_transport_lines(swapped, [0, 1], [0, 1])))
        with self.assertRaises(ValueError):
            plot_transport_lines(self.exact, self.xs[:-1], self.ys)
        with self.assertRaises(ValueError):
            plot_transport_lines([[-1.0]], [0.0], [0.0])

    def test_plan_matrices_count_used_cells(self) -> None:
        """Entropic plans use more cells than the exact plan."""
        from probstat.visualization.transport_plots import (
            plot_plan_matrices,
            used_cells,
        )

        figure = plot_plan_matrices({"exact": self.exact, "eps = 1": self.runs[0]})
        exact_cells = used_cells(self.exact.plan)
        blurred = used_cells(self.runs[0].plan)
        self.assertGreater(blurred, exact_cells)
        title = _title(figure.axes[0])
        self.assertIn(f"{blurred} of 12 cells", title)
        self.assertIn(f"exact uses {exact_cells}", title)
        with self.assertRaises(ValueError):
            plot_plan_matrices({})
        with self.assertRaises(ValueError):
            plot_plan_matrices({"a": [[1.0]], "b": [[0.5, 0.5]]})

    def test_sinkhorn_convergence_reports_iteration_ratio(self) -> None:
        """The title compares the iteration counts of the extreme eps."""
        from probstat.visualization import active_palette
        from probstat.visualization.transport_plots import plot_sinkhorn_convergence

        axes = plot_sinkhorn_convergence(self.runs, tolerance=1e-9)
        title = _title(axes)
        self.assertIn("eps shrinks from 1 to 0.2", title)
        self.assertIn(f"{self.runs[1].iterations:,} iterations", title)
        self.assertEqual(axes.get_yscale(), "log")
        self.assertIn(active_palette().accent, _line_colours(axes))
        with self.assertRaises(ValueError):
            plot_sinkhorn_convergence([])

    def test_quantile_area_equals_w1(self) -> None:
        """For two unit normals one apart W1 = 1 and the curves never cross."""
        from probstat.distributions import Normal
        from probstat.visualization import active_palette
        from probstat.visualization.transport_plots import plot_quantile_wasserstein

        axes = plot_quantile_wasserstein(
            Normal(0.0, 1.0), Normal(1.0, 1.0), names=("N(0,1)", "N(1,1)")
        )
        self.assertTrue(_title(axes).startswith("W1 = 1.000"))
        self.assertIn("lies below", _title(axes))
        self.assertIn(active_palette().series[3], _line_colours(axes))
        crossing = plot_quantile_wasserstein(Normal(0.0, 1.0), Normal(0.0, 2.0))
        self.assertIn("cross at u = 0.50", _title(crossing))
        with self.assertRaises(TypeError):
            plot_quantile_wasserstein(object(), Normal(0.0, 1.0))
        with self.assertRaises(ValueError):
            plot_quantile_wasserstein(Normal(0.0, 1.0), Normal(0.0, 1.0), clip=0.2)

    def test_displacement_midpoint_is_the_average_normal(self) -> None:
        """The midpoint averages means and standard deviations."""
        from probstat.visualization import active_palette
        from probstat.visualization.transport_plots import (
            plot_displacement_interpolation,
        )

        axes = plot_displacement_interpolation(
            (-2.0, 0.5), (3.0, 1.5), names=("A", "B")
        )
        self.assertIn("is N(0.5, 1^2)", _title(axes))
        self.assertIn("bimodal", _title(axes))
        self.assertIn(active_palette().model, _line_colours(axes))
        near = plot_displacement_interpolation((0.0, 1.0), (0.5, 1.0))
        self.assertIn("narrower than the 50/50 mixture", _title(near))
        with self.assertRaises(ValueError):
            plot_displacement_interpolation((0.0, 0.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            plot_displacement_interpolation((0.0, 1.0), (1.0, 1.0), times=(1.0,))


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class BothPalettesTests(unittest.TestCase):
    """Every new figure renders and saves under porcelain and graphite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once."""
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_figures_render_in_both_palettes(self) -> None:
        """Canvas colours follow the palette and the files are non-trivial."""
        from probstat.distributions import Normal
        from probstat.information import exact_transport, power_cost_matrix, sinkhorn
        from probstat.validation import pit_coverage, reliability_diagram
        from probstat.visualization import GRAPHITE, PORCELAIN, chart_theme, save_figure
        from probstat.visualization.scoring_plots import (
            plot_coverage,
            plot_distribution_shift,
            plot_reliability_diagram,
            plot_score_comparison,
        )
        from probstat.visualization.transport_plots import (
            plot_displacement_interpolation,
            plot_plan_matrices,
            plot_quantile_wasserstein,
            plot_sinkhorn_convergence,
            plot_transport_lines,
        )

        rng = random.Random(5)
        forecasts = [rng.random() for _ in range(200)]
        outcomes = [rng.random() < f for f in forecasts]
        diagram = reliability_diagram(forecasts, outcomes, bins=4)
        u = [rng.random() for _ in range(200)]
        cost = power_cost_matrix([0.0, 1.0], [0.5, 2.0], p=2.0)
        plan = exact_transport([0.5, 0.5], [0.5, 0.5], cost)
        run = sinkhorn([0.5, 0.5], [0.5, 0.5], cost, epsilon=0.5)
        with tempfile.TemporaryDirectory() as directory:
            for palette in (PORCELAIN, GRAPHITE):
                with chart_theme(palette):
                    produced = [
                        plot_reliability_diagram(diagram),
                        plot_score_comparison({"a": u, "b": [v + 0.1 for v in u]}),
                        plot_coverage(pit_coverage(u, (0.5, 0.9))),
                        plot_distribution_shift(u[:100], u[100:]),
                        plot_transport_lines(plan, [0.0, 1.0], [0.5, 2.0]),
                        plot_plan_matrices({"exact": plan, "eps": run}),
                        plot_sinkhorn_convergence([run]),
                        plot_quantile_wasserstein(Normal(0, 1), Normal(1, 2)),
                        plot_displacement_interpolation((0.0, 1.0), (4.0, 2.0)),
                    ]
                for index, item in enumerate(produced):
                    figure = getattr(item, "figure", item)
                    self.assertEqual(_hex(figure.get_facecolor()), palette.canvas)
                    path = save_figure(
                        figure, Path(directory) / f"{palette.name}-{index}.png"
                    )
                    self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
