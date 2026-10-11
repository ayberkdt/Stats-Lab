"""Smoke tests for discrepancy, hierarchical, mixture, PPC, and stage figures."""

from __future__ import annotations

import importlib.util
import math
import random
import tempfile
import unittest
from statistics import NormalDist
from typing import Any

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)


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


def _decay_calibration() -> tuple[Any, Any]:
    """Return a small discrepancy calibration and two parameter vectors."""
    from probstat.distributions import LogNormal
    from probstat.inverse import (
        DiscrepancyCalibration,
        DiscrepancyProcess,
        Experiment,
        KnownNoise,
    )
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.uq import QuantityOfInterest

    def simulator(p):
        """Return ``4 exp(-k x)``."""
        return {"y": 4.0 * math.exp(-p["k"] * p["x"])}

    model = ProbabilisticModel(
        JointInput.independent(RandomInput("k", LogNormal(math.log(0.5), 0.5))),
        simulator,
        (QuantityOfInterest("y", "y"),),
    )
    data = tuple(
        Experiment(
            {"y": 4.0 * math.exp(-0.4 * x) + 0.25 * math.sin(2.5 * x)}, {"x": x}
        )
        for x in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    )
    process = DiscrepancyProcess(
        ("x",),
        standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
        length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
    )
    problem = DiscrepancyCalibration(model, data, "y", KnownNoise(0.03), process)
    vectors = (
        (0.39, math.log(0.25), math.log(0.6)),
        (0.41, math.log(0.3), math.log(0.7)),
    )
    return problem, vectors


class MixtureIntervalTests(unittest.TestCase):
    """Exact numerical helpers that do not need Matplotlib."""

    def test_single_component_matches_normal_quantiles(self) -> None:
        """One Gaussian component returns its exact central interval."""
        from probstat.visualization.calibration_plots import (
            gaussian_mixture_interval,
        )

        low, high = gaussian_mixture_interval([1.0], [4.0], 0.9)
        z = NormalDist().inv_cdf(0.95)
        self.assertAlmostEqual(low, 1.0 - 2.0 * z, places=7)
        self.assertAlmostEqual(high, 1.0 + 2.0 * z, places=7)

    def test_weights_and_point_masses(self) -> None:
        """Zero-weight components are ignored; point masses give sample limits."""
        from probstat.visualization.calibration_plots import (
            gaussian_mixture_interval,
        )

        low, high = gaussian_mixture_interval(
            [0.0, 50.0], [1.0, 1.0], 0.95, weights=[1.0, 0.0]
        )
        self.assertAlmostEqual(high, NormalDist().inv_cdf(0.975), places=6)
        self.assertAlmostEqual(low, -high, places=6)
        low, high = gaussian_mixture_interval([2.0, 2.0], [0.0, 0.0], 0.5)
        self.assertAlmostEqual(low, 2.0, places=6)
        self.assertAlmostEqual(high, 2.0, places=6)
        for bad in (
            {"means": [], "variances": []},
            {"means": [0.0], "variances": [-1.0]},
            {"means": [0.0], "variances": [1.0], "probability": 1.0},
            {"means": [0.0], "variances": [1.0], "weights": [0.0]},
        ):
            with self.assertRaises(ValueError):
                gaussian_mixture_interval(**bad)

    def test_binomial_band_for_eight_values_in_four_bins(self) -> None:
        """Binomial(8, 1/4) has central 95% count limits 0 and 5."""
        from probstat.visualization.hierarchical_plots import binomial_band

        self.assertEqual(binomial_band(8, 4, 0.95), (0, 5))
        low, high = binomial_band(1000, 10, 0.95)
        self.assertLess(low, 100)
        self.assertGreater(high, 100)
        with self.assertRaises(ValueError):
            binomial_band(10, 1)

    def test_theta_intervals_bracket_means_and_tau_density_integrates(self) -> None:
        """Exact intervals contain the posterior means; the density has area 1."""
        from probstat.inference import hierarchical_normal_posterior
        from probstat.visualization.hierarchical_plots import (
            tau_density,
            theta_intervals,
        )

        posterior = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S, grid_size=401)
        means, sds = posterior.theta_moments()
        for (low, high), mean, sd in zip(
            theta_intervals(posterior), means, sds, strict=True
        ):
            self.assertLess(low, mean)
            self.assertGreater(high, mean)
            self.assertGreater(high - low, 2.0 * sd)
        density = tau_density(posterior)
        grid = posterior.tau_grid
        area = sum(
            0.5 * (density[i] + density[i + 1]) * (grid[i + 1] - grid[i])
            for i in range(len(grid) - 1)
        )
        self.assertAlmostEqual(area, 1.0, places=9)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class CalibrationFigureTests(unittest.TestCase):
    """Discrepancy figures: computed titles, roles, and errors."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend and build predictions once."""
        import matplotlib

        matplotlib.use("Agg")
        cls.problem, vectors = _decay_calibration()
        cls.xs = tuple(4.0 * i / 10 for i in range(11))
        cls.predictions = cls.problem.predict(
            vectors, tuple({"x": x} for x in cls.xs)
        )

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_prediction_plot_counts_truth_coverage_and_uses_roles(self) -> None:
        """The title counts coverage; corrected = model, simulator = series 4."""
        from probstat.visualization import active_palette
        from probstat.visualization.calibration_plots import (
            plot_discrepancy_predictions,
        )

        truth = [4.0 * math.exp(-0.4 * x) + 0.25 * math.sin(2.5 * x) for x in self.xs]
        axes = plot_discrepancy_predictions(
            self.problem, self.predictions, "x", truth=truth
        )
        palette = active_palette()
        title = axes.get_title(loc="left")
        self.assertIn("covers the truth at", title)
        self.assertIn(f"of {len(self.xs)} points", title)
        colours = _line_colours(axes)
        self.assertIn(palette.model.upper(), colours)
        self.assertIn(palette.series[3].upper(), colours)
        self.assertIn(palette.accent.upper(), colours)
        scatter = axes.collections[-1]
        self.assertEqual(_hex(scatter.get_facecolor()[0]), palette.observed.upper())
        self.assertEqual(len(scatter.get_offsets()), self.problem.observation_count)
        self.assertIn("bias-corrected", _text_strings(axes))

    def test_prediction_plot_without_truth_quotes_largest_shift(self) -> None:
        """Without a truth the title quotes the largest discrepancy mean."""
        from probstat.visualization.calibration_plots import (
            plot_discrepancy_predictions,
        )

        axes = plot_discrepancy_predictions(self.problem, self.predictions, "x")
        worst = max(abs(item.discrepancy_mean) for item in self.predictions)
        self.assertIn(f"{worst:.3g}", axes.get_title(loc="left"))

    def test_prediction_plot_rejects_bad_inputs(self) -> None:
        """Unordered, missing, or misaligned inputs raise ``ValueError``."""
        from probstat.visualization.calibration_plots import (
            plot_discrepancy_estimate,
            plot_discrepancy_predictions,
        )

        with self.assertRaises(ValueError):
            plot_discrepancy_predictions(
                self.problem, tuple(reversed(self.predictions)), "x"
            )
        with self.assertRaises(ValueError):
            plot_discrepancy_predictions(self.problem, self.predictions, "z")
        with self.assertRaises(ValueError):
            plot_discrepancy_estimate(self.predictions, "x", truth=[0.0])
        with self.assertRaises(ValueError):
            plot_discrepancy_estimate(self.predictions[:1], "x")

    def test_discrepancy_estimate_reports_zero_exclusion(self) -> None:
        """The title states where the band excludes zero; zero line is ink."""
        from probstat.visualization import active_palette
        from probstat.visualization.calibration_plots import (
            gaussian_mixture_interval,
            plot_discrepancy_estimate,
        )

        axes = plot_discrepancy_estimate(self.predictions, "x")
        limits = [
            gaussian_mixture_interval(p.discrepancy_means, p.discrepancy_variances)
            for p in self.predictions
        ]
        excluded = sum(low > 0.0 or high < 0.0 for low, high in limits)
        title = axes.get_title(loc="left")
        if excluded:
            self.assertIn(f"{excluded / len(limits):.0%}", title)
        else:
            self.assertIn("contains zero everywhere", title)
        palette = active_palette()
        self.assertIn(palette.ink.upper(), _line_colours(axes))
        self.assertIn(palette.model.upper(), _line_colours(axes))


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class HierarchyFigureTests(unittest.TestCase):
    """Shrinkage, tau, mixture, predictive-check, and PIT figures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend and build the grid posterior."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.inference import hierarchical_normal_posterior

        cls.posterior = hierarchical_normal_posterior(
            SCHOOLS_Y, SCHOOLS_S, grid_size=401
        )

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_shrinkage_plot_quotes_mean_shrinkage(self) -> None:
        """The title quotes the mean of E[B_j | y]; mu is the accent line."""
        from probstat.visualization import active_palette
        from probstat.visualization.hierarchical_plots import plot_shrinkage

        axes = plot_shrinkage(self.posterior, names="ABCDEFGH")
        shrink = self.posterior.expected_shrinkage()
        average = sum(shrink) / len(shrink)
        self.assertIn(f"{average:.0%} of the way to mu", axes.get_title(loc="left"))
        palette = active_palette()
        self.assertIn(palette.accent.upper(), _line_colours(axes))
        labels = [tick.get_text() for tick in axes.get_yticklabels()]
        self.assertEqual(labels[-1], "A")
        with self.assertRaises(ValueError):
            plot_shrinkage(self.posterior, names=("A", "B"))

    def test_tau_plot_compares_draws_with_exact_mean(self) -> None:
        """With draws the title compares the MCMC and exact means."""
        from probstat.visualization import active_palette
        from probstat.visualization.hierarchical_plots import plot_tau_posterior

        draws = [row[1] for row in self.posterior.sample(400, seed=3)]
        axes = plot_tau_posterior(self.posterior, draws=draws)
        title = axes.get_title(loc="left")
        self.assertIn(f"median {self.posterior.tau_quantile(0.5):.3g}", title)
        self.assertIn("MCMC mean", title)
        self.assertIn(active_palette().model.upper(), _line_colours(axes))
        self.assertTrue(axes.patches)
        with self.assertRaises(ValueError):
            plot_tau_posterior(self.posterior, upper_probability=0.2)
        with self.assertRaises(ValueError):
            plot_tau_posterior(self.posterior, draws=[-1.0, 1.0])

    def test_mixture_fit_and_selection_titles(self) -> None:
        """Mixture titles name the components and the selected K."""
        from probstat.inference import (
            GaussianMixture,
            fit_gaussian_mixture,
            select_gaussian_mixture,
        )
        from probstat.visualization import active_palette
        from probstat.visualization.hierarchical_plots import (
            plot_mixture_fit,
            plot_mixture_selection,
        )

        rng = random.Random(4)
        data = [rng.gauss(-3.0, 0.7) for _ in range(80)] + [
            rng.gauss(2.0, 0.7) for _ in range(120)
        ]
        fit = fit_gaussian_mixture(data, 2, seed=1)
        axes = plot_mixture_fit(fit, data)
        title = axes.get_title(loc="left")
        self.assertTrue(title.startswith("2 components centred at"))
        self.assertEqual(_line_colours(axes), {active_palette().model.upper()})
        selection = select_gaussian_mixture(data, (1, 2, 3), seed=1)
        axes = plot_mixture_selection(selection)
        self.assertIn(f"K = {selection.best_components}", axes.get_title(loc="left"))
        self.assertIn(active_palette().accent.upper(), _line_colours(axes))
        flat = GaussianMixture((1.0,), ((0.0, 0.0),), (((1.0, 0.0), (0.0, 1.0)),))
        two_d = fit_gaussian_mixture(flat.sample(30, seed=2), 1)
        with self.assertRaises(ValueError):
            plot_mixture_fit(two_d, data)

    def test_predictive_check_titles_switch_with_p_value(self) -> None:
        """A tail p-value below 0.025 flips the title; observed is the accent."""
        from probstat.inference import PredictiveStatisticCheck
        from probstat.visualization import active_palette
        from probstat.visualization.hierarchical_plots import plot_predictive_check

        replicated = tuple(float(i) for i in range(100))
        central = PredictiveStatisticCheck(
            "max", (50.0,) * 100, replicated, 0.5, 0.51, 0.05, False
        )
        axes = plot_predictive_check(central)
        title = axes.get_title(loc="left")
        self.assertIn("reproduces the observed max: p = 0.50", title)
        self.assertIn(active_palette().accent.upper(), _line_colours(axes))
        extreme = PredictiveStatisticCheck(
            "max", (120.0,) * 100, replicated, 0.0, 1.0, 0.0, False
        )
        self.assertIn("rarely", plot_predictive_check(extreme).get_title(loc="left"))
        realized = PredictiveStatisticCheck(
            "chi2", replicated, tuple(reversed(replicated)), 0.5, 0.5, 0.05, True
        )
        axes = plot_predictive_check(realized)
        self.assertIn("chi2", axes.get_title(loc="left"))
        self.assertIn("T_rep = T_obs", _text_strings(axes))
        with self.assertRaises(ValueError):
            plot_predictive_check(
                PredictiveStatisticCheck("x", (1.0,), (1.0,), 1.0, 1.0, 0.0, False)
            )

    def test_pit_histogram_flags_miscalibration(self) -> None:
        """Uniform PIT values stay inside the band; piled-up values do not."""
        from probstat.visualization.hierarchical_plots import plot_pit_histogram

        uniform = [(i + 0.5) / 100 for i in range(100)]
        title = plot_pit_histogram(uniform).get_title(loc="left")
        self.assertTrue(title.startswith("All 10 PIT bins"))
        piled = [0.02] * 60 + [0.98] * 40
        self.assertIn("miscalibrated", plot_pit_histogram(piled).get_title(loc="left"))
        with self.assertRaises(ValueError):
            plot_pit_histogram([0.2, 1.5])


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class StudyStageFigureTests(unittest.TestCase):
    """Figures of the extremes, information, and design stages."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one small study with every new stage."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.distributions import LogNormal, Normal, Uniform
        from probstat.inverse import NoiseModel
        from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
        from probstat.uq import QuantityOfInterest
        from probstat.workflow import UQStudy

        def margin(p):
            """Return the safety margin ``R - S``."""
            return {"g": p["R"] - p["S"]}

        capacity = ProbabilisticModel(
            JointInput.independent(
                RandomInput("R", LogNormal(math.log(5.0), 0.1)),
                RandomInput("S", Normal(3.0, 0.8)),
            ),
            margin,
            (QuantityOfInterest("g", "margin"),),
            name="capacity",
        )
        cls.study = UQStudy(capacity, seed=7)
        cls.study.propagate(2048, design="mc")
        cls.tail = cls.study.extremes("g", tail="lower", level=0.0)
        cls.study.sensitivity("g", 256, bootstrap=0)
        cls.information = cls.study.information_sensitivity("g", max_points=200)

        def decay(p):
            """Return ``A exp(-k t)``."""
            return {"y": p["A"] * math.exp(-p["k"] * p["t"])}

        decay_model = ProbabilisticModel(
            JointInput.independent(
                RandomInput("A", Uniform(2.0, 8.0)),
                RandomInput("k", Uniform(0.05, 1.0)),
            ),
            decay,
            (QuantityOfInterest("y", "amplitude"),),
            fixed={"t": 0.0},
            name="decay",
        )
        cls.decay = UQStudy(decay_model, seed=7)
        cls.design = cls.decay.design(
            tuple({"t": t} for t in (0.5, 2.0, 4.0, 8.0)),
            2,
            noise=(NoiseModel("y", standard_deviation=0.05),),
            method="linearized",
        )

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_tail_plot_compares_pot_with_empirical_interval(self) -> None:
        """The title states whether POT lies in the Wilson interval."""
        from probstat.visualization import active_palette
        from probstat.visualization.stage_plots import plot_tail_analysis

        axes = plot_tail_analysis(self.tail)
        assert self.tail.empirical is not None
        interval = self.tail.empirical.confidence_interval
        inside = interval.lower <= self.tail.pot_probability <= interval.upper
        title = axes.get_title(loc="left")
        self.assertIn("P(g < 0)", title)
        self.assertIn("inside" if inside else "outside", title)
        self.assertEqual(axes.get_yscale(), "log")
        self.assertIn(active_palette().model.upper(), _line_colours(axes))
        self.assertIn(active_palette().accent.upper(), _line_colours(axes))
        with self.assertRaises(ValueError):
            plot_tail_analysis(self.tail, confidence=1.5)

    def test_information_plot_with_and_without_sobol(self) -> None:
        """One panel alone; two panels with Sobol; mismatched inputs raise."""
        from probstat.visualization.stage_plots import plot_information_sensitivity

        alone = plot_information_sensitivity(self.information)
        self.assertEqual(len(alone.axes), 1)
        top = self.information.ranking()[0]
        self.assertTrue(alone.axes[0].get_title(loc="left").startswith(f"Input {top}"))
        sobol = self.study.latest("sensitivity").result
        paired = plot_information_sensitivity(self.information, sobol=sobol)
        self.assertEqual(len(paired.axes), 2)

        class Other:
            """Sobol stand-in with different input names."""

            input_names = ("a", "b")

        with self.assertRaises(ValueError):
            plot_information_sensitivity(self.information, sobol=Other())

    def test_design_plot_titles_the_chosen_conditions(self) -> None:
        """The title lists the picks and the EIG path end points."""
        from probstat.visualization.stage_plots import plot_design_path

        axes = plot_design_path(self.design)
        selection = self.design.selection
        title = axes.get_title(loc="left")
        for condition in selection.selected:
            self.assertIn(f"t={condition['t']:g}", title)
        self.assertIn(f"{selection.information_gains[-1]:.2f} nats", title)

    def test_study_outputs_include_the_new_stage_figures(self) -> None:
        """save_study_figures and study_html render every new stage."""
        from probstat.visualization.study import save_study_figures, study_html

        with tempfile.TemporaryDirectory() as directory:
            paths = save_study_figures(self.study, directory)
            names = [path.name for path in paths]
            self.assertTrue(any("extremes" in name for name in names))
            self.assertTrue(any("information" in name for name in names))
            paths = save_study_figures(self.decay, directory)
            self.assertTrue(any("design" in path.name for path in paths))
        page = study_html(self.study)
        # propagation, extremes, sensitivity, information: one figure each,
        # light and dark.
        self.assertEqual(page.count("data:image/png;base64,"), 8)
        # The design stage draws its EIG path and, for one-factor candidates,
        # the greedy candidate curves: two figures, light and dark.
        self.assertEqual(study_html(self.decay).count("data:image/png;base64,"), 4)


if __name__ == "__main__":
    unittest.main()
