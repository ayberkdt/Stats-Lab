"""Model-discrepancy laboratory: Kennedy-O'Hagan calibration and noise models.

Part 1 calibrates a decay simulator ``4 exp(-k x)`` against synthetic data from
a process that differs from it by a smooth oscillation. It compares a plain
calibration (no discrepancy) with Kennedy-O'Hagan calibration, and prints the
simulator-only and bias-corrected predictive intervals at held-out inputs.

Part 2 repeats the exercise with a discrepancy that is confounded with ``k``
(a linear drift) to show the Brynjarsdottir-O'Hagan point: the bias-corrected
predictions stay accurate, but what is learned about ``k`` depends on the
discrepancy prior.

Part 3 estimates a log-linear heteroscedastic noise model jointly with a
straight line.

With Matplotlib installed, Part 1 also writes two figures to
``examples/generated/``: the observations with simulator-only and
bias-corrected predictive bands, and the estimated discrepancy ``delta(x)``.

Run with ``PYTHONPATH=src python examples/discrepancy_calibration_lab.py``.
"""

from __future__ import annotations

import importlib.util
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from probstat.distributions import LogNormal, Normal
from probstat.inverse import (
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyPrediction,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    LogLinearNoise,
    NoiseModel,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

TRUE_RATE = 0.4
NOISE = 0.03
DATA_X = tuple(4.0 * i / 14 for i in range(15))
HELD_OUT = (0.65, 1.9, 3.14, 4.5)
FIGURE_GRID = tuple(4.5 * i / 44 for i in range(45))
OUTPUT = Path(__file__).parent / "generated"


def simulator(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the simulator ``4 exp(-k x)``.

    References:
        [REF-KENNEDY-OHAGAN-2001]; a transparent teaching model.

    """
    return {"y": 4.0 * math.exp(-parameters["k"] * parameters["x"])}


def oscillating_truth(x: float) -> float:
    """Return the physical process: the simulator at k=0.4 plus a smooth bias.

    References:
        [REF-KENNEDY-OHAGAN-2001]; synthetic truth for this lab.

    """
    return 4.0 * math.exp(-TRUE_RATE * x) + 0.25 * math.sin(2.5 * x)


def drifting_truth(x: float) -> float:
    """Return a process whose bias ``-0.12 x`` is confounded with ``k``.

    References:
        [REF-BRYNJARSDOTTIR-OHAGAN-2014]; synthetic truth for this lab.

    """
    return 4.0 * math.exp(-TRUE_RATE * x) - 0.12 * x


def simulator_model() -> ProbabilisticModel:
    """Return the simulator with the prior ``k ~ LogNormal(log 0.5, 0.5)``.

    References:
        [REF-KENNEDY-OHAGAN-2001].

    """
    prior = JointInput.independent(RandomInput("k", LogNormal(math.log(0.5), 0.5)))
    return ProbabilisticModel(prior, simulator, (QuantityOfInterest("y", "y"),))


def experiments(truth: Callable[[float], float], seed: int) -> tuple[Experiment, ...]:
    """Return noisy synthetic observations of ``truth`` at the data inputs.

    References:
        [REF-KENNEDY-OHAGAN-2001]; synthetic data for this lab.

    """
    generator = random.Random(seed)
    return tuple(
        Experiment({"y": truth(x) + generator.gauss(0.0, NOISE)}, {"x": x})
        for x in DATA_X
    )


def part_one() -> None:
    """Compare plain and Kennedy-O'Hagan calibration on an oscillating bias.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004].

    """
    print("Part 1: smooth oscillating discrepancy, truth k = 0.4")
    data = experiments(oscillating_truth, 11)
    model = simulator_model()
    plain = CalibrationProblem(
        model, data, (NoiseModel("y", standard_deviation=NOISE),)
    )
    plain_post = plain.sample_posterior(draws=1000, warmup=500, chains=2, seed=3)
    (summary,) = plain_post.summary()
    print(
        f"  plain calibration: k = {summary.mean:.4f} +- "
        f"{summary.standard_deviation:.4f} (90% {summary.quantile(0.05):.4f}"
        f"..{summary.quantile(0.95):.4f})"
    )
    discrepancy = DiscrepancyProcess(
        ("x",),
        standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
        length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
    )
    problem = DiscrepancyCalibration(model, data, "y", KnownNoise(NOISE), discrepancy)
    fit = problem.fit()
    print(
        "  KOH MAP: "
        + ", ".join(
            f"{name} = {value:.4f} (SE {se:.4f})"
            for name, value, se in zip(
                fit.names, fit.estimate, fit.standard_errors, strict=True
            )
        )
    )
    posterior = problem.sample_posterior(draws=1000, warmup=500, chains=2, seed=3)
    for item in posterior.summary():
        print(
            f"  KOH posterior {item.name:12s} mean {item.mean:.4f} sd "
            f"{item.standard_deviation:.4f} R-hat {item.rank_normalized_rhat:.3f} "
            f"ESS {item.effective_sample_size:.0f}"
        )
    predictions = problem.predict(
        posterior, tuple({"x": x} for x in HELD_OUT), max_draws=500
    )
    print("  held-out x  truth    simulator-only 95%      bias-corrected 95%   delta")
    for x, prediction in zip(HELD_OUT, predictions, strict=True):
        true_value = oscillating_truth(x)
        s_low, s_high = prediction.interval(0.95, kind="simulator")
        c_low, c_high = prediction.interval(0.95, kind="corrected")
        s_mark = "covers" if s_low <= true_value <= s_high else "MISSES"
        c_mark = "covers" if c_low <= true_value <= c_high else "MISSES"
        shift = prediction.discrepancy_mean
        print(
            f"  {x:9.2f}  {true_value:.4f}  [{s_low:.3f}, {s_high:.3f}] {s_mark}"
            f"  [{c_low:.3f}, {c_high:.3f}] {c_mark}  {shift:+.3f}"
        )
    grid = problem.predict(
        posterior, tuple({"x": x} for x in FIGURE_GRID), max_draws=300
    )
    save_figures(problem, grid)
    report = problem.identifiability(posterior)
    print(
        f"  identifiability: max |rho(k, log psi)| = "
        f"{report.max_abs_cross_correlation:.3f}, sd ratio k = "
        f"{report.standard_deviation_ratios['k']:.3f}, warning = {report.warning}"
    )


def save_figures(
    problem: DiscrepancyCalibration, grid: Sequence[DiscrepancyPrediction]
) -> None:
    """Write the prediction and discrepancy figures when Matplotlib exists.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-MATPLOTLIB].

    """
    if importlib.util.find_spec("matplotlib") is None:
        print("  (install matplotlib to write the figures)")
        return
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure
    from probstat.visualization.calibration_plots import (
        plot_discrepancy_estimate,
        plot_discrepancy_predictions,
    )

    truth = [oscillating_truth(x) for x in FIGURE_GRID]
    bias = [0.25 * math.sin(2.5 * x) for x in FIGURE_GRID]
    axes = plot_discrepancy_predictions(problem, grid, "x", truth=truth)
    first = save_figure(axes.figure, OUTPUT / "discrepancy-predictions.png")
    plt.close(axes.figure)
    axes = plot_discrepancy_estimate(grid, "x", truth=bias)
    second = save_figure(axes.figure, OUTPUT / "discrepancy-delta.png")
    plt.close(axes.figure)
    print(f"  figures: {first.name}, {second.name} in {OUTPUT}")


def part_two() -> None:
    """Show that theta is learned only as well as the discrepancy prior allows.

    References:
        [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """
    print("\nPart 2: discrepancy -0.12 x confounded with k, truth k = 0.4")
    data = experiments(drifting_truth, 12)
    model = simulator_model()
    plain = CalibrationProblem(
        model, data, (NoiseModel("y", standard_deviation=NOISE),)
    )
    laplace = plain.laplace_posterior()
    print(
        f"  plain calibration: k = {laplace.mode[0]:.4f} +- "
        f"{math.sqrt(laplace.covariance[0][0]):.4f}"
    )
    priors = {
        "vague (sd ~ LogNormal(log 1, 0.75))": LogNormal(0.0, 0.75),
        "tight (sd ~ LogNormal(log 0.05, 0.3))": LogNormal(math.log(0.05), 0.3),
    }
    for label, law in priors.items():
        discrepancy = DiscrepancyProcess(
            ("x",),
            standard_deviation_prior=law,
            length_scale_priors=(LogNormal(math.log(1.5), 0.5),),
        )
        problem = DiscrepancyCalibration(
            model, data, "y", KnownNoise(NOISE), discrepancy
        )
        posterior = problem.sample_posterior(draws=800, warmup=400, chains=2, seed=5)
        (k_summary, *_rest) = posterior.summary()
        (prediction,) = problem.predict(posterior, ({"x": 2.1},), max_draws=400)
        low, high = prediction.interval(0.95, kind="corrected")
        report = problem.identifiability(posterior)
        print(
            f"  KOH {label}: k = {k_summary.mean:.4f} (90% "
            f"{k_summary.quantile(0.05):.4f}..{k_summary.quantile(0.95):.4f}), "
            f"sd ratio {report.standard_deviation_ratios['k']:.3f}, "
            f"max |rho| {report.max_abs_cross_correlation:.3f}; "
            f"corrected 95% at x=2.1 [{low:.3f}, {high:.3f}] "
            f"(truth {drifting_truth(2.1):.3f})"
        )
        if report.warning:
            print(f"    warning: {report.warning}")


def line(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``c0 + c1 x``.

    References:
        [REF-HARVEY-1976].

    """
    return {"y": parameters["c0"] + parameters["c1"] * parameters["x"]}


def covariate(conditions: Mapping[str, float]) -> tuple[float, ...]:
    """Return the log-noise covariate ``g(c) = x``.

    References:
        [REF-HARVEY-1976].

    """
    return (conditions["x"],)


def part_three() -> None:
    """Estimate ``sigma(x) = exp(a + b x)`` jointly with a straight line.

    References:
        [REF-HARVEY-1976].

    """
    print("\nPart 3: heteroscedastic noise sigma(x) = exp(-2 + 0.5 x), n = 200")
    prior = JointInput.independent(
        RandomInput("c0", Normal(0.0, 10.0)), RandomInput("c1", Normal(0.0, 10.0))
    )
    model = ProbabilisticModel(prior, line, (QuantityOfInterest("y", "y"),))
    generator = random.Random(2026)
    data = tuple(
        Experiment(
            {"y": 1.0 + 0.7 * x + generator.gauss(0.0, math.exp(-2.0 + 0.5 * x))},
            {"x": x},
        )
        for x in (4.0 * i / 199 for i in range(200))
    )
    noise = LogLinearNoise(covariate, Normal(0.0, 3.0), (Normal(0.0, 3.0),), ("x",))
    problem = DiscrepancyCalibration(model, data, "y", noise)
    truth = {"c0": 1.0, "c1": 0.7, "noise_a": -2.0, "noise_b[x]": 0.5}
    fit = problem.fit(objective="likelihood")
    for name, value, se in zip(
        fit.names, fit.estimate, fit.standard_errors, strict=True
    ):
        print(
            f"  MLE {name:11s} {value:+.4f} (SE {se:.4f}); truth {truth[name]:+.2f}, "
            f"z = {(value - truth[name]) / se:+.2f}"
        )
    homoscedastic = CalibrationProblem(
        model,
        data,
        (NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.3), 1.0)),),
    )
    constant = homoscedastic.fit_least_squares()
    print(
        f"  constant-noise least squares: c1 = {constant.estimate[1]:+.4f} "
        f"(SE {constant.standard_errors[1]:.4f}), sigma = "
        f"{constant.noise_standard_deviations['y']:.4f}"
    )
    predictions = problem.predict(fit, ({"x": 0.5}, {"x": 3.5}))
    for x, prediction in zip((0.5, 3.5), predictions, strict=True):
        low, high = prediction.interval(0.95, kind="observed")
        print(f"  plug-in 95% observation interval at x = {x}: [{low:.3f}, {high:.3f}]")


def main() -> None:
    """Run the three parts and report the wall time.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-BRYNJARSDOTTIR-OHAGAN-2014],
        [REF-HARVEY-1976].

    """
    start = time.perf_counter()
    part_one()
    part_two()
    part_three()
    print(f"\nwall time {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
