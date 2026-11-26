"""Figure of the large-scale inverse problems guide.

The figure runs the seeded deriv2 example of the guide (ADR-0144): LSQR from
zero on the whitened problem, its data misfit and its true relative error
against the iteration, with the iteration chosen by the discrepancy
principle and the error-optimal iteration that only a synthetic benchmark
can know. The facts are the numbers the caption in the guide quotes.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-PAIGE-SAUNDERS-1982],
    [REF-ENGL-HANKE-NEUBAUER-1996], [REF-MOROZOV-1966].

"""

from __future__ import annotations

import functools
import math
import random
from typing import Any

from probstat.design import (
    exhaustive_linear_design,
    greedy_linear_design,
    linear_design_criteria,
)
from probstat.inverse import (
    LinearInverseProblem,
    correlated_gaussian_benchmark,
    credible_bands,
    deconvolution_benchmark,
    expected_noise_norm,
    krylov_regularization,
    linear_gaussian_posterior,
    second_derivative_problem,
    total_variation_regularized_least_squares,
)
from probstat.linalg import FirstDifferenceOperator, ScaledIdentityCovariance
from probstat.visualization import theme

from . import GuideFigure

SEED = 20261126
WIDE = (7.6, 4.4)


@functools.cache
def _semi_convergence() -> dict[str, Any]:
    """Run the guide's example: deriv2, n = 32, 1% noise, seed 20261126.

    References:
        [EQ-INV-ITER-002], [EQ-LINALG-KRY-007].

    """
    base = second_derivative_problem(32)
    exact = base.exact_data
    scale = 0.01 * math.hypot(*exact) / math.sqrt(len(exact))
    generator = random.Random(SEED)
    data = [b + scale * generator.gauss(0.0, 1.0) for b in exact]
    problem = LinearInverseProblem.create(
        base.operator, data, noise=ScaledIdentityCovariance(32, scale * scale)
    )
    truth = base.exact_solution
    norm = math.hypot(*truth)
    long_run = krylov_regularization(problem, max_iterations=20, truth=truth)
    stopped = krylov_regularization(
        problem, noise_norm=expected_noise_norm(32), truth=truth
    )
    assert long_run.error_history is not None
    assert stopped.error_history is not None
    errors = [value / norm for value in long_run.error_history]
    oracle = min(range(len(errors)), key=errors.__getitem__)
    return {
        "misfits": list(long_run.misfit_history),
        "errors": errors,
        "stop": stopped.iterations,
        "stop_error": stopped.error_history[-1] / norm,
        "oracle": oracle,
        "oracle_error": errors[oracle],
        "delta": 1.01 * expected_noise_norm(32),
    }


def _semi_convergence_figure() -> tuple[Any, dict[str, str]]:
    """Draw misfit and true error of LSQR against the iteration.

    References:
        [EQ-INV-ITER-002], [REF-ENGL-HANKE-NEUBAUER-1996].

    """
    example = _semi_convergence()
    facts = {
        "stop": f"{example['stop']}",
        "stop_error": f"{example['stop_error']:.3f}",
        "oracle": f"{example['oracle']}",
        "oracle_error": f"{example['oracle_error']:.3f}",
        "final_error": f"{example['errors'][-1]:.2f}",
        "target": f"{example['delta']:.2f}",
    }
    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.34}
    )
    palette = theme.active_palette()
    iterations = list(range(len(example["misfits"])))
    model = theme.role_colour("model")
    for axes in (left, right):
        theme.style_axes(axes)
        axes.set_xlim(0.0, iterations[-1])
        axes.grid(axis="x", visible=False)
        axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    theme.line(left, iterations, example["misfits"], color=model)
    theme.points(left, iterations, example["misfits"], color=model, size=14.0)
    left.set_yscale("log")
    left.set_ylim(0.1, 2000.0)
    left.minorticks_off()
    theme.reference_line(left, y=example["delta"], entity="threshold")
    left.annotate(
        f"tau delta = {facts['target']}",
        xy=(iterations[-1], example["delta"]),
        xytext=(0.0, 6.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=theme.ANNOTATION_SIZE,
        color=theme.entity_colour("threshold"),
    )
    theme.axis_labels(left, "LSQR iteration k", "Whitened misfit (log scale)")
    theme.line(right, iterations, example["errors"], color=model)
    theme.points(right, iterations, example["errors"], color=model, size=14.0)
    right.set_ylim(0.0, 2.0)
    stop, oracle = example["stop"], example["oracle"]
    for axes, x_value in ((left, stop), (right, stop)):
        theme.reference_line(axes, x=x_value, color=palette.muted)
    right.annotate(
        f"discrepancy stop\nk = {facts['stop']}: {facts['stop_error']}",
        xy=(stop, example["stop_error"]),
        xytext=(stop + 0.6, 1.45),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.points(
        right,
        [oracle],
        [example["oracle_error"]],
        color=theme.entity_colour("truth"),
        size=48.0,
        marker="D",
    )
    right.annotate(
        f"error-optimal (truth needed)\nk = {facts['oracle']}: {facts['oracle_error']}",
        xy=(oracle, example["oracle_error"]),
        xytext=(10.0, -4.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.axis_labels(right, "LSQR iteration k", "Relative error")
    theme.title_block(
        left,
        "The misfit keeps falling while the error turns up after a few "
        "iterations;\nthe discrepancy principle stops before the noise is fitted",
        "deriv2, n = 32, 1% noise (seed 20261126) · LSQR from zero on the "
        "whitened problem",
    )
    return figure, facts


@functools.cache
def _edges() -> dict[str, Any]:
    """Run the guide's deblurring comparison: L04, n = 64, refinement 2, 2% noise.

    References:
        [EQ-INV-BENCH-001], [EQ-INV-ITER-002], [EQ-INV-SPARSE-001].

    """
    case = deconvolution_benchmark(64, refinement=2, relative_noise=0.02)
    problem = LinearInverseProblem.create(case.operator, case.data, noise=case.noise)
    norm = math.hypot(*case.truth)
    early = krylov_regularization(problem, noise_norm=expected_noise_norm(64))
    tv = total_variation_regularized_least_squares(
        case.operator,
        case.data,
        0.5,
        noise=case.noise,
        gradient=FirstDifferenceOperator(64, spacing=1.0 / 64),
        max_iterations=20000,
    )

    def error(estimate: Any) -> float:
        """Return the relative error against the withheld truth.

        References:
            [EQ-INV-BENCH-001].

        """
        pairs = zip(estimate, case.truth, strict=True)
        return math.hypot(*(a - b for a, b in pairs)) / norm

    return {
        "grid": [(i + 0.5) / 64 for i in range(64)],
        "truth": list(case.truth),
        "lsqr": list(early.estimate),
        "tv": list(tv.solution),
        "lsqr_error": error(early.estimate),
        "lsqr_iterations": early.iterations,
        "tv_error": error(tv.solution),
        "tv_converged": tv.converged,
    }


def _edges_figure() -> tuple[Any, dict[str, str]]:
    """Draw the truth, the early-stopped LSQR and the TV reconstruction.

    References:
        [EQ-INV-SPARSE-001], [REF-RUDIN-OSHER-FATEMI-1992].

    """
    example = _edges()
    if not example["tv_converged"]:
        raise RuntimeError("the TV reconstruction of the guide did not converge")
    facts = {
        "lsqr_error": f"{example['lsqr_error']:.3f}",
        "lsqr_iterations": f"{example['lsqr_iterations']}",
        "tv_error": f"{example['tv_error']:.4f}",
    }
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    grid = example["grid"]
    axes.step(
        grid,
        example["truth"],
        where="mid",
        color=palette.muted,
        linewidth=theme.CONTEXT_WIDTH,
    )
    lsqr_colour = theme.role_colour("series-3")
    tv_colour = theme.role_colour("model")
    theme.line(axes, grid, example["lsqr"], color=lsqr_colour, dashes=theme.FITTED)
    theme.line(axes, grid, example["tv"], color=tv_colour)
    axes.set_xlim(0.0, 1.12)
    axes.set_ylim(-0.25, 1.25)
    axes.grid(axis="x", visible=False)
    axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    theme.direct_labels(
        axes,
        [
            (grid[-1], example["truth"][-1], "truth", palette.muted),
            (grid[-1], example["lsqr"][-1], "LSQR", lsqr_colour),
            (grid[-1], example["tv"][-1], "TV", tv_colour),
        ],
        gap_points=13.0,
    )
    theme.axis_labels(axes, "Position s", "Value x(s)")
    theme.title_block(
        axes,
        f"Total variation keeps the jumps (relative error {facts['tv_error']});\n"
        f"early-stopped LSQR smooths them ({facts['lsqr_error']})",
        "1D deblurring L04, n = 64, data from a twice finer grid, 2% noise · "
        "TV weight 0.5 by PDHG",
    )
    return axes.figure, facts


@functools.cache
def _bands() -> dict[str, Any]:
    """Sample the exact L06 posterior and form pointwise and simultaneous bands.

    References:
        [EQ-INV-LG-001], [EQ-INV-FS-007], [EQ-INV-BENCH-001].

    """
    case = correlated_gaussian_benchmark(
        40, 12, length_scale=0.2, noise_level=0.05, seed=SEED
    )
    posterior = linear_gaussian_posterior(
        case.operator,
        case.data,
        noise=case.noise,
        prior_covariance=case.prior_covariance,
    )
    draws = posterior.sample(2000, seed=SEED)
    bands = credible_bands(draws, 0.9)
    truth = list(case.truth)
    inside_pointwise = sum(
        1
        for t, lo, hi in zip(truth, bands.lower, bands.upper, strict=True)
        if lo <= t <= hi
    )
    inside_simultaneous = all(
        lo <= t <= hi
        for t, lo, hi in zip(
            truth, bands.simultaneous_lower, bands.simultaneous_upper, strict=True
        )
    )
    return {
        "grid": [(i + 0.5) / 40 for i in range(40)],
        "truth": truth,
        "mean": list(posterior.mean),
        "lower": list(bands.lower),
        "upper": list(bands.upper),
        "sim_lower": list(bands.simultaneous_lower),
        "sim_upper": list(bands.simultaneous_upper),
        "factor": bands.scaling_factor,
        "inside_pointwise": inside_pointwise,
        "inside_simultaneous": inside_simultaneous,
        "points": len(truth),
    }


def _bands_figure() -> tuple[Any, dict[str, str]]:
    """Draw the truth, the posterior mean and the two 90% bands.

    References:
        [EQ-INV-FS-007], [REF-STUART-2010].

    """
    example = _bands()
    facts = {
        "factor": f"{example['factor']:.2f}",
        "inside_pointwise": f"{example['inside_pointwise']}",
        "points": f"{example['points']}",
        "simultaneous": "inside" if example["inside_simultaneous"] else "outside",
    }
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    grid = example["grid"]
    model = theme.role_colour("model")
    axes.fill_between(
        grid,
        example["sim_lower"],
        example["sim_upper"],
        color=model,
        alpha=0.14,
        linewidth=0.0,
    )
    axes.fill_between(
        grid,
        example["lower"],
        example["upper"],
        color=model,
        alpha=0.30,
        linewidth=0.0,
    )
    theme.line(axes, grid, example["mean"], color=model)
    truth_colour = theme.entity_colour("truth")
    theme.points(axes, grid, example["truth"], color=truth_colour, size=12.0)
    axes.set_xlim(0.0, 1.2)
    axes.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.grid(axis="x", visible=False)
    axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    last = len(grid) - 1
    theme.direct_labels(
        axes,
        [
            (grid[last], example["sim_upper"][last], "90% simultaneous", model),
            (grid[last], example["upper"][last], "90% pointwise", model),
            (grid[last], example["mean"][last], "posterior mean", model),
            (grid[last], example["truth"][last], "truth", truth_colour),
        ],
        gap_points=13.0,
    )
    theme.axis_labels(axes, "Position s", "Value x(s)")
    theme.title_block(
        axes,
        f"The simultaneous band (c = {facts['factor']} sd) holds the whole truth;\n"
        f"the pointwise band holds {facts['inside_pointwise']} of "
        f"{facts['points']} points",
        "Correlated Gaussian prior L06, n = 40, 12 local averages, noise 0.05 · "
        "2000 exact posterior draws",
    )
    return axes.figure, facts


@functools.cache
def _design() -> dict[str, Any]:
    """Compare greedy, exhaustive and random A-optimal designs.

    References:
        [EQ-DES-LIN-006], [EQ-DES-LIN-001].

    """
    n, candidates = 12, 16
    grid = [(i + 0.5) / n for i in range(n)]
    prior = [[math.exp(-abs(a - b) / 0.25) for b in grid] for a in grid]
    rows = []
    for k in range(candidates):
        centre = (k + 0.5) / candidates
        weights = [math.exp(-0.5 * ((g - centre) / 0.06) ** 2) for g in grid]
        total = math.fsum(weights)
        rows.append([w / total for w in weights])
    noise = [0.05**2] * candidates
    budgets = list(range(1, 6))
    greedy = greedy_linear_design(
        rows, noise, prior_covariance=prior, budget=budgets[-1], criterion="a"
    )
    optimum = []
    for k in budgets:
        best = exhaustive_linear_design(
            rows, noise, prior_covariance=prior, size=k, criterion="a"
        )
        optimum.append(best.value)
    generator = random.Random(SEED)
    medians, lows, highs = [], [], []
    for k in budgets:
        values = sorted(
            linear_design_criteria(
                rows,
                noise,
                generator.sample(range(candidates), k),
                prior_covariance=prior,
            ).a_optimal
            for _ in range(300)
        )
        medians.append(values[150])
        lows.append(values[30])
        highs.append(values[270])
    prior_trace = math.fsum(prior[i][i] for i in range(n))
    return {
        "budgets": budgets,
        "greedy": list(greedy.values),
        "optimum": optimum,
        "median": medians,
        "low": lows,
        "high": highs,
        "prior_trace": prior_trace,
    }


def _design_figure() -> tuple[Any, dict[str, str]]:
    """Draw the posterior-variance trace against the number of sensors.

    References:
        [EQ-DES-LIN-006].

    """
    example = _design()
    gaps = [g - o for g, o in zip(example["greedy"], example["optimum"], strict=True)]
    worst = max(range(len(gaps)), key=gaps.__getitem__)
    facts = {
        "greedy_5": f"{example['greedy'][-1]:.2f}",
        "optimum_5": f"{example['optimum'][-1]:.2f}",
        "median_5": f"{example['median'][-1]:.2f}",
        "prior": f"{example['prior_trace']:.0f}",
        "gap": f"{gaps[worst]:.2f}",
        "gap_k": f"{example['budgets'][worst]}",
        "equal": " and ".join(
            str(k) for k, g in zip(example["budgets"], gaps, strict=True) if g < 1e-9
        ),
    }
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    budgets = example["budgets"]
    model = theme.role_colour("model")
    random_colour = palette.muted
    axes.fill_between(
        budgets,
        example["low"],
        example["high"],
        color=random_colour,
        alpha=0.22,
        linewidth=0.0,
    )
    theme.line(
        axes, budgets, example["median"], color=random_colour, dashes=theme.FITTED
    )
    theme.line(axes, budgets, example["greedy"], color=model)
    axes.scatter(
        budgets,
        example["optimum"],
        s=70.0,
        marker="D",
        facecolors="none",
        edgecolors=theme.entity_colour("truth"),
        linewidths=1.4,
        zorder=5.0,
    )
    axes.set_xlim(0.7, 7.3)
    axes.set_xticks(budgets)
    axes.set_ylim(0.0, None)
    axes.grid(axis="x", visible=False)
    axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    theme.direct_labels(
        axes,
        [
            (
                budgets[-1],
                example["median"][-1],
                "random (median, 10-90%)",
                random_colour,
            ),
            (budgets[-1], example["greedy"][-1], "greedy", model),
            (
                budgets[-1],
                example["optimum"][-1],
                "exhaustive optimum",
                theme.entity_colour("truth"),
            ),
        ],
        gap_points=13.0,
    )
    theme.axis_labels(axes, "Number of sensors k", "Trace of posterior covariance")
    theme.title_block(
        axes,
        f"Greedy sensors reach the exhaustive optimum only at k = {facts['equal']};"
        f"\nworst gap {facts['gap']} at k = {facts['gap_k']}; at k = 5: "
        f"{facts['greedy_5']} vs {facts['optimum_5']} (random {facts['median_5']})",
        f"12 unknowns, prior trace {facts['prior']}, 16 candidate local averages, "
        "noise sd 0.05 · 300 random designs per k",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="large-scale-semi-convergence",
        guide="large-scale-inverse-problems",
        alt=(
            "Two panels against the LSQR iteration: the whitened data misfit on "
            "a log axis with the discrepancy target, and the relative error "
            "with the discrepancy stop and the error-optimal iteration marked"
        ),
        draw=_semi_convergence_figure,
    ),
    GuideFigure(
        name="large-scale-tv-edges",
        guide="large-scale-inverse-problems",
        alt=(
            "A piecewise-constant true signal with its early-stopped LSQR and "
            "total-variation reconstructions from blurred, noisy data"
        ),
        draw=_edges_figure,
    ),
    GuideFigure(
        name="large-scale-posterior-bands",
        guide="large-scale-inverse-problems",
        alt=(
            "A sampled true field with the exact posterior mean and its "
            "pointwise and simultaneous 90% credible bands from posterior draws"
        ),
        draw=_bands_figure,
    ),
    GuideFigure(
        name="large-scale-design-gains",
        guide="large-scale-inverse-problems",
        alt=(
            "Trace of the posterior covariance against the number of sensors for "
            "the greedy A-optimal design, the exhaustive optimum and random designs"
        ),
        draw=_design_figure,
    ),
]
