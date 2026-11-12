"""Figures of the uncertainty-quantification guides (ADR-0105).

One module covers the nine guides of the UQ path: uncertainty modeling and
fitting, propagation, advanced Monte Carlo, sensitivity analysis, surrogate
models, VV-UQ foundations, model validation, forecast calibration, and
conformal prediction. Every ``draw`` function runs a seeded computation of
``probstat`` and returns the figure with the formatted facts its caption
quotes; titles reuse the same strings.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from guide_figures import GuideFigure
from probstat.visualization import theme

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------


def _panels(
    count: int = 1,
    *,
    figsize: tuple[float, float] = (7.2, 4.0),
    sharey: bool = False,
    wspace: float = 0.28,
) -> tuple[Any, list[Any]]:
    """Return a figure with ``count`` themed axes side by side.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    plt = theme.pyplot()
    figure, grid = plt.subplots(1, count, figsize=figsize, sharey=sharey, squeeze=False)
    axes = list(grid[0])
    for ax in axes:
        theme.style_axes(ax)
    if count > 1:
        figure.subplots_adjust(wspace=wspace)
    return figure, axes


def _label(
    ax: Any,
    x: float,
    y: float,
    text: str,
    colour: str,
    *,
    offset: tuple[float, float] = (6.0, 0.0),
    ha: str = "left",
    va: str = "center",
    bold: bool = True,
    boxed: bool = False,
) -> Any:
    """Write a direct label next to a data point in the series colour.

    With ``boxed=True`` the label sits on a canvas-coloured box, which keeps
    its contrast over a shaded area.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    extra: dict[str, Any] = {}
    if boxed:
        extra["bbox"] = {
            "boxstyle": "round,pad=0.18",
            "facecolor": theme.active_palette().canvas,
            "edgecolor": "none",
        }
    return ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold" if bold else "normal",
        color=colour,
        annotation_clip=False,
        **extra,
    )


def _ecdf(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Return the step coordinates of an empirical CDF.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5].

    """
    ordered = sorted(values)
    n = len(ordered)
    return ordered, [(index + 1) / n for index in range(n)]


def _plain_log(axis: Any, *, percent: bool = False) -> None:
    """Write plain numbers (or percentages) on a logarithmic axis.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    if percent:
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{100.0 * v:g}%"))
    else:
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _quiet(function: Callable[[], Any]) -> Any:
    """Run a computation with its diagnostic warnings silenced.

    The warnings (MCMC mixing, kernel conditioning) are part of the guides'
    narratives; silencing them keeps the build output readable and changes
    no number.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function()


def _two(value: float) -> str:
    """Format a number with two decimals and no negative zero.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


# --------------------------------------------------------------------------
# Uncertainty modeling and fitting
# --------------------------------------------------------------------------


def draw_fitting_bootstrap_calibration() -> tuple[Any, dict[str, str]]:
    """Compare naive and bootstrap KS p-values against a fitted normal.

    Eighty normal samples of size 30 are each tested against the normal law
    fitted to them; the naive p-value uses the fully specified KS null, the
    parametric bootstrap refits every one of 39 replicates.

    Equation:
        [EQ-INF-GOF-PB-001], [EQ-INF-GOF-KS-001].

    References:
        [REF-LILLIEFORS-1967], [REF-STEPHENS-1974],
        [REF-DAVISON-HINKLEY-1997, Chapter 4].

    """
    from probstat.distributions import Normal
    from probstat.foundations.random_streams import RandomStream
    from probstat.inference import (
        NORMAL_FAMILY,
        bootstrap_goodness_of_fit,
        fit_normal_mle,
        kolmogorov_smirnov_test,
    )

    root = RandomStream(11)
    replicates, size, refits = 80, 30, 39
    naive: list[float] = []
    calibrated: list[float] = []
    for replicate in range(replicates):
        data = Normal(3.0, 2.0).sample(size, seed=root.child("data", replicate).seed)
        result = bootstrap_goodness_of_fit(
            data,
            NORMAL_FAMILY,
            n_bootstrap=refits,
            seed=root.child("bootstrap", replicate),
        )
        calibrated.append(result.ks_p_value)
        naive.append(kolmogorov_smirnov_test(data, fit_normal_mle(data)).p_value)
    naive_mean = f"{sum(naive) / replicates:.2f}"
    boot_mean = f"{sum(calibrated) / replicates:.2f}"
    naive_low = sum(p < 0.1 for p in naive)
    boot_low = sum(p < 0.1 for p in calibrated)
    facts = {
        "naive_mean": naive_mean,
        "bootstrap_mean": boot_mean,
        "naive_below": f"{naive_low} of {replicates}",
        "bootstrap_below": f"{boot_low} of {replicates}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.4))
    palette = theme.active_palette()
    uniform = theme.role_colour("reference")
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=uniform,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=2,
    )
    for values, colour in ((naive, palette.muted), (calibrated, palette.observed)):
        xs, ys = _ecdf(values)
        ax.step(
            [0.0, *xs, 1.0],
            [0.0, *ys, 1.0],
            where="post",
            color=colour,
            linewidth=theme.LINE_WIDTH,
            zorder=3,
        )
    at = 0.72
    naive_at = sum(p <= at for p in naive) / replicates
    boot_at = sum(p <= 0.3 for p in calibrated) / replicates
    _label(ax, at, naive_at, "naive KS p-values", palette.muted, offset=(8.0, -14.0))
    _label(
        ax,
        0.3,
        boot_at,
        "bootstrap with refitting",
        palette.observed,
        offset=(-10.0, 14.0),
        ha="right",
    )
    _label(ax, 0.86, 0.86, "uniform", uniform, offset=(-8.0, 10.0), ha="right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="both", color=palette.rule, linewidth=0.6)
    theme.axis_labels(ax, "p-value", "Share of samples with a smaller p-value")
    theme.title_block(
        ax,
        f"Naive KS p-values against a fitted normal average {naive_mean}; "
        f"the refitting bootstrap gives {boot_mean}",
        f"Empirical CDF of p-values from {replicates} normal samples of size "
        f"{size}; the null is true",
    )
    return figure, facts


def draw_fitting_bearing_tails() -> tuple[Any, dict[str, str]]:
    """Show three near-tied fits of the bearing lives diverging in the tail.

    Equation:
        [EQ-INF-MLE-GEN-001], [EQ-INF-IC-001], [EQ-INF-GOF-QQ-001].

    References:
        [REF-LIEBLEIN-ZELEN-1956], [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """
    from probstat.inference import (
        GAMMA_FAMILY,
        LOGNORMAL_FAMILY,
        WEIBULL_FAMILY,
        fit_distribution,
    )
    from probstat.statistics.datasets import LIEBLEIN_ZELEN_BALL_BEARINGS

    values = tuple(LIEBLEIN_ZELEN_BALL_BEARINGS.values())
    fits = sorted(
        (
            fit_distribution(values, family)
            for family in (WEIBULL_FAMILY, LOGNORMAL_FAMILY, GAMMA_FAMILY)
        ),
        key=lambda fit: fit.aic,
    )
    spread = f"{fits[-1].aic - fits[0].aic:.1f}"
    lives = {fit.family: fit.distribution.quantile(0.01) for fit in fits}
    low, high = min(lives.values()), max(lives.values())
    facts = {
        "aic_spread": spread,
        "life_low": f"{low:.1f}",
        "life_high": f"{high:.1f}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.4))
    colours = (
        theme.role_colour("model"),
        theme.role_colour("series-3"),
        theme.role_colour("context"),
    )
    dashes = (None, theme.FITTED, theme.PROJECTED)
    markers = ("s", "D", "^")
    floor = 0.0035
    for fit, colour, dash, marker in zip(fits, colours, dashes, markers, strict=True):
        law = fit.distribution
        start = law.quantile(0.01)
        stop = 220.0
        grid = [start * (stop / start) ** (k / 200) for k in range(201)]
        theme.line(ax, grid, [law.cdf(x) for x in grid], color=colour, dashes=dash)
        life = lives[fit.family]
        theme.points(
            ax, [life], [0.01], color=colour, size=38.0, zorder=5, marker=marker
        )
    n = len(values)
    ordered = sorted(values)
    theme.points(
        ax,
        ordered,
        [(i + 0.5) / n for i in range(n)],
        entity="observations",
        size=26.0,
    )
    theme.reference_line(ax, y=0.01, label="1% failed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax.xaxis)
    _plain_log(ax.yaxis, percent=True)
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xlim(4.0, 260.0)
    ax.set_ylim(floor, 1.15)
    order = sorted(fits, key=lambda fit: lives[fit.family])
    for rank, fit in enumerate(order):
        colour = colours[fits.index(fit)]
        _label(
            ax,
            lives[fit.family],
            0.01,
            f"{fit.family} {lives[fit.family]:.1f}",
            colour,
            offset=(0.0, -12.0 - 11.0 * rank),
            ha="center",
            va="top",
        )
    theme.axis_labels(
        ax, "Endurance [millions of revolutions]", "Cumulative probability"
    )
    theme.title_block(
        ax,
        f"Within {spread} AIC units, the three fits put the 1% life between "
        f"{low:.1f} and {high:.1f} million revolutions",
        f"Dots: {n} bearing lives at Hazen positions. Lines: maximum-likelihood "
        "Gamma, LogNormal, and Weibull CDFs",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Uncertainty propagation
# --------------------------------------------------------------------------


def _projectile(parameters: Any) -> dict[str, float]:
    """Return the drag-free range of a projectile for speed and angle.

    Equation:
        [EQ-LAB-PROJECTILE-001].

    References:
        [REF-HALLIDAY-RESNICK-WALKER-2014].

    """
    speed, angle = parameters["v"], parameters["theta"]
    return {"range": speed * speed * math.sin(2.0 * angle) / 9.80665}


def draw_propagation_design_convergence() -> tuple[Any, dict[str, str]]:
    """Plot the spread of replicate means against sample size per design.

    Equation:
        [EQ-MC-ROOTN-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-OWEN-2013, Section 17.2], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    from probstat.distributions import Normal, Uniform
    from probstat.foundations.random_streams import RandomStream
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.propagation import compare_designs
    from probstat.uq import QuantityOfInterest

    model = ProbabilisticModel(
        JointInput(
            (
                RandomInput("v", Normal(25.0, 1.5)),
                RandomInput("theta", Uniform(0.55, 0.85)),
            )
        ),
        _projectile,
        (QuantityOfInterest("range", "projectile range", "m"),),
    )
    root = RandomStream(2026)
    sizes = [2**k for k in range(4, 12)]
    designs = ("mc", "lhs", "sobol")
    replicates = 20
    spread: dict[str, list[float]] = {design: [] for design in designs}
    for size in sizes:
        summary = compare_designs(
            model,
            "range",
            size,
            designs=designs,
            replicates=replicates,
            seed=root.child("designs", size),
        )
        for design in designs:
            spread[design].append(summary[design][1])
    last = len(sizes) - 1
    gains = {d: spread["mc"][last] / spread[d][last] for d in ("lhs", "sobol")}
    logs = [math.log(n) for n in sizes]
    errs = [math.log(e) for e in spread["mc"]]
    mean_x, mean_y = sum(logs) / len(logs), sum(errs) / len(errs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(logs, errs, strict=True))
    slope /= sum((x - mean_x) ** 2 for x in logs)
    facts = {
        "size": f"{sizes[last]:,}",
        "lhs_gain": f"{gains['lhs']:.0f}",
        "sobol_gain": f"{gains['sobol']:.0f}",
        "mc_slope": f"{slope:.2f}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.4))
    palette = theme.active_palette()
    names = theme.CATEGORY_ORDERS["sampling designs"]
    styles = {
        "mc": (names[0], palette.muted, "o"),
        "lhs": (names[1], theme.role_colour("series-3"), "s"),
        "sobol": (names[3], palette.observed, "D"),
    }
    labels = []
    for design in designs:
        name, colour, marker = styles[design]
        theme.line(ax, sizes, spread[design], color=colour)
        theme.points(ax, sizes, spread[design], color=colour, marker=marker)
        labels.append((sizes[last], spread[design][last], name, colour))
    anchor = spread["mc"][0]
    ax.plot(
        [sizes[0], sizes[last]],
        [anchor, anchor * (sizes[0] / sizes[last]) ** 0.5],
        color=palette.muted,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=2,
    )
    rate = spread["sobol"][0] * 1.6
    ax.plot(
        [sizes[0], sizes[last]],
        [rate, rate * sizes[0] / sizes[last]],
        color=palette.muted,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _plain_log(ax.xaxis)
    _plain_log(ax.yaxis)
    ax.set_xticks(sizes)
    labels.append(
        (sizes[last], anchor * (sizes[0] / sizes[last]) ** 0.5, "N^-1/2", palette.muted)
    )
    labels.append((sizes[last], rate * sizes[0] / sizes[last], "N^-1", palette.muted))
    theme.direct_labels(ax, labels)
    theme.axis_labels(
        ax, theme.axis_title("sample_size"), "Spread of the mean range", y_unit="m"
    )
    theme.title_block(
        ax,
        f"At N = {sizes[last]:,}, Latin hypercube points cut the error of the "
        f"mean {gains['lhs']:.0f}-fold and Sobol points {gains['sobol']:.0f}-fold",
        f"Standard deviation of {replicates} replicate means of the projectile "
        "range; grey dashes fall as N^-1/2 and N^-1",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Advanced Monte Carlo
# --------------------------------------------------------------------------


def draw_variance_reduction_budget() -> tuple[Any, dict[str, str]]:
    """Compare five estimators of ``E[U^2]`` at one evaluation budget.

    The setting is the one of ``examples/advanced_monte_carlo_lab.py``.

    Equation:
        [EQ-MC-ANTI-001], [EQ-MC-CONTROL-001], [EQ-MC-STRAT-001],
        [EQ-MC-IS-001], [EQ-MC-EFF-001].

    References:
        [REF-OWEN-2013, Chapters 8 and 9].

    """
    from probstat.monte_carlo import (
        allocate_stratified_budget,
        compare_efficiency,
        estimate_antithetic_uniform,
        estimate_control_variate,
        estimate_importance_sampling,
        estimate_mean,
        estimate_stratified_uniform,
    )

    seed, budget = 2026, 20_000
    generator = random.Random(seed)
    uniforms = tuple(generator.random() for _ in range(budget))
    squares = tuple(value * value for value in uniforms)
    crude = estimate_mean(squares)
    estimates = {
        "crude Monte Carlo": crude,
        "antithetic pairs": estimate_antithetic_uniform(
            lambda u: u * u, pairs=budget // 2, seed=seed
        ),
        "control variate": estimate_control_variate(
            squares, uniforms, known_control_mean=0.5
        ).estimate,
        "stratified (Neyman)": estimate_stratified_uniform(
            lambda u: u * u,
            allocate_stratified_budget((0.08, 0.12, 0.20, 0.35, 0.55), budget),
            seed=seed,
        ),
        "importance sampling": estimate_importance_sampling(
            lambda u: u * u,
            lambda u: 1.0 if 0.0 <= u <= 1.0 else 0.0,
            lambda u: 2.0 * u if 0.0 <= u <= 1.0 else 0.0,
            lambda local: math.sqrt(local.random()),
            sample_size=budget,
            seed=seed,
        ).estimate,
    }
    gains = {
        name: compare_efficiency(crude, value).efficiency_gain
        for name, value in estimates.items()
    }
    best = max(gains, key=lambda name: gains[name])
    short = {
        "antithetic pairs": "antithetic sampling",
        "control variate": "a control variate",
        "stratified (Neyman)": "stratified sampling",
        "importance sampling": "importance sampling",
    }
    facts = {
        "budget": f"{budget:,}",
        "best": short[best],
        "gain": f"{gains[best]:.0f}",
        "crude_se": f"{crude.standard_error:.5f}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    rows = list(range(len(estimates)))[::-1]
    for row, (name, value) in zip(rows, estimates.items(), strict=True):
        interval = value.confidence_interval
        colour = palette.muted if name == "crude Monte Carlo" else palette.observed
        ax.hlines(
            row, interval.lower, interval.upper, color=colour, linewidth=2.4, zorder=2
        )
        theme.points(ax, [value.estimate], [row], color=colour, size=48.0)
        text = (
            f"SE {value.standard_error:.5f}"
            if name == "crude Monte Carlo"
            else f"{gains[name]:,.0f}x as efficient"
        )
        _label(
            ax, interval.upper, row, text, palette.ink, offset=(8.0, 0.0), bold=False
        )
    theme.reference_line(ax, x=1.0 / 3.0, label="exact 1/3", entity="closed form")
    ax.set_yticks(rows)
    ax.set_yticklabels(list(estimates))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    low = min(v.confidence_interval.lower for v in estimates.values())
    high = max(v.confidence_interval.upper for v in estimates.values())
    ax.set_xlim(low - 0.15 * (high - low), high + 0.75 * (high - low))
    theme.axis_labels(ax, "Estimate of E[U^2]")
    theme.title_block(
        ax,
        f"At {budget:,} evaluations, {facts['best']} is {facts['gain']}x as "
        "efficient as crude Monte Carlo",
        "Dot: estimate. Bar: 95% normal interval. Efficiency: ratio of "
        "SE^2 x evaluations to crude Monte Carlo",
    )
    return figure, facts


class _GeometricBrownianMotion:
    """Coupled Euler-Maruyama sampler of ``S_T`` for ``dS = mu S dt + s S dW``.

    Level ``l`` uses ``2^l`` steps on ``[0, 1]`` from ``S_0 = 1``; the coarse
    path reuses the summed Brownian increments of each fine step pair (the
    sampler of ``tests/test_mlmc_ce_line_shapley.py``).

    References:
        [REF-GILES-2008].

    """

    def __init__(self, drift: float, volatility: float) -> None:
        """Store the drift and the volatility.

        References:
            [REF-GILES-2008].

        """
        self.drift = drift
        self.volatility = volatility

    def __call__(
        self, level: int, count: int, stream: Any
    ) -> tuple[list[float], list[float]]:
        """Return ``count`` coupled fine and coarse terminal values.

        Equation:
            [EQ-MC-MLMC-001].

        References:
            [REF-GILES-2008].

        """
        generator = stream.generator()
        steps = 2**level
        h = 1.0 / steps
        root_h = math.sqrt(h)
        mu, sigma = self.drift, self.volatility
        fine: list[float] = []
        coarse: list[float] = []
        for _ in range(count):
            if level == 0:
                fine.append(1.0 + mu + sigma * generator.gauss(0.0, 1.0))
                coarse.append(0.0)
                continue
            s_fine = s_coarse = 1.0
            for _pair in range(steps // 2):
                d1 = root_h * generator.gauss(0.0, 1.0)
                d2 = root_h * generator.gauss(0.0, 1.0)
                s_fine *= 1.0 + mu * h + sigma * d1
                s_fine *= 1.0 + mu * h + sigma * d2
                s_coarse *= 1.0 + 2.0 * mu * h + sigma * (d1 + d2)
            fine.append(s_fine)
            coarse.append(s_coarse)
        return fine, coarse


def draw_mlmc_level_variances() -> tuple[Any, dict[str, str]]:
    """Plot the level variances and sample allocation of an MLMC run.

    Equation:
        [EQ-MC-MLMC-001], [EQ-MC-MLMC-002], [EQ-MC-MLMC-003].

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """
    from probstat.monte_carlo import adaptive_multilevel_monte_carlo

    eps = 0.02
    result = adaptive_multilevel_monte_carlo(
        _GeometricBrownianMotion(1.0, 0.5), eps, seed=1
    )
    levels = [item.level for item in result.levels]
    finest = levels[-1]
    # Exact Var(S_T) of the lognormal terminal value with mu = 1, sigma = 0.5.
    exact_variance = math.exp(2.0) * (math.exp(0.25) - 1.0)
    plain = 2.0 * exact_variance * 2**finest / eps**2
    ratio = plain / result.total_cost
    facts = {
        "beta": f"{result.beta:.2f}",
        "ratio": f"{ratio:.1f}",
        "cost": f"{result.total_cost:,.0f}",
    }

    figure, (left, right) = _panels(2, figsize=(7.6, 4.0), wspace=0.32)
    palette = theme.active_palette()
    fine = [item.fine_variance for item in result.levels]
    corrections = [item.variance for item in result.levels]
    theme.line(left, levels, fine, color=palette.muted)
    theme.points(left, levels, fine, color=palette.muted, marker="s")
    theme.line(left, levels[1:], corrections[1:], color=palette.observed)
    theme.points(left, levels[1:], corrections[1:], color=palette.observed)
    left.set_yscale("log")
    _plain_log(left.yaxis)
    left.set_yticks([0.001, 0.01, 0.1, 1.0])
    left.set_xticks(levels)
    _label(
        left,
        levels[2],
        fine[2],
        "Var P_l",
        palette.muted,
        offset=(0.0, 10.0),
        ha="center",
        va="bottom",
    )
    _label(
        left,
        levels[5],
        corrections[5],
        "Var(P_l - P_l-1)",
        palette.observed,
        offset=(-8.0, -10.0),
        ha="right",
        va="top",
    )
    theme.axis_labels(left, "Level l", "Variance")
    samples = [item.samples for item in result.levels]
    theme.line(right, levels, samples, color=palette.observed)
    theme.points(right, levels, samples, color=palette.observed)
    right.set_yscale("log")
    _plain_log(right.yaxis)
    right.set_xticks(levels)
    theme.axis_labels(right, "Level l", "Samples N_l")
    theme.title_block(
        left,
        f"Level corrections shrink like 2^(-{facts['beta']} l), so MLMC needs "
        f"{facts['ratio']}x less work than plain Monte Carlo",
        f"Euler-Maruyama paths of geometric Brownian motion, RMSE target {eps}; "
        "right: optimal samples per level",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Sensitivity analysis and surrogates
# --------------------------------------------------------------------------


def _ishigami_point(first: float, second: float, third: float) -> float:
    """Evaluate the Ishigami benchmark with ``a = 7`` and ``b = 0.1``.

    Equation:
        [EQ-LAB-ISHIGAMI-001].

    References:
        [REF-SALTELLI-2008, Chapter 4].

    """
    return (
        math.sin(first) + 7.0 * math.sin(second) ** 2 + 0.1 * third**4 * math.sin(first)
    )


def draw_ishigami_sobol_indices() -> tuple[Any, dict[str, str]]:
    """Draw pick-freeze first-order and total Sobol indices of Ishigami.

    The design is the one of ``examples/sensitivity_lab.py`` (two base
    samples of 50 000 rows, seed 2026).

    Equation:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    References:
        [REF-SALTELLI-2010], [REF-SOBOL-1993].

    """
    from probstat.sensitivity import estimate_sobol_indices
    from probstat.visualization.uq import plot_sobol_indices

    generator = random.Random(2026)
    size = 50_000

    def base() -> tuple[tuple[float, ...], ...]:
        """Return one base matrix of independent uniform inputs.

        References:
            [REF-SALTELLI-2010].

        """
        return tuple(
            tuple(generator.uniform(-math.pi, math.pi) for _ in range(3))
            for _ in range(size)
        )

    sample_a = base()
    sample_b = base()
    result = estimate_sobol_indices(
        lambda p: _ishigami_point(p[0], p[1], p[2]), sample_a, sample_b
    )
    s3, st3 = result.first_order[2], result.total_effect[2]
    facts = {"s3": _two(s3), "st3": _two(st3), "size": f"{size:,}"}
    axes = plot_sobol_indices(
        ("x1", "x2", "x3"),
        result.first_order,
        result.total_effect,
        title=f"x3 has almost no main effect (S = {facts['s3']}) yet a total "
        f"effect of {facts['st3']} through its interaction with x1",
        subtitle=f"Pick-freeze estimates from two {size:,}-row designs; grey "
        "segment: variance an input causes only jointly",
    )
    return axes.figure, facts


def draw_gp_variance_beyond_data() -> tuple[Any, dict[str, str]]:
    """Show a GP fitted to 12 runs and its predictive band out to ``x = 12``.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-003], [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """
    from probstat.surrogates import fit_gaussian_process

    def truth(x: float) -> float:
        """Return the target ``sin(x) + 0.3 x``.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return math.sin(x) + 0.3 * x

    runs = [6.0 * index / 11 for index in range(12)]
    process = _quiet(
        lambda: fit_gaussian_process(
            [[x] for x in runs],
            [truth(x) for x in runs],
            kernel="squared_exponential",
            seed=4,
        )
    )
    grid = [12.0 * k / 240 for k in range(241)]
    pairs = [process.predict_with_variance([x]) for x in grid]
    means = [m for m, _ in pairs]
    sds = [math.sqrt(max(v, 0.0)) for _, v in pairs]
    far_mean, far_variance = pairs[-1]
    inside = max(v for x, (_, v) in zip(grid, pairs, strict=True) if x <= 6.0)
    facts = {
        "far_variance": f"{far_variance:.2f}",
        "inside_variance": f"{inside:.0e}",
        "far_error": f"{abs(truth(12.0) - far_mean):.2f}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.2))
    theme.band(
        ax,
        grid,
        [m - 1.96 * s for m, s in zip(means, sds, strict=True)],
        [m + 1.96 * s for m, s in zip(means, sds, strict=True)],
        entity="posterior mean",
    )
    theme.line(ax, grid, [truth(x) for x in grid], entity="truth")
    theme.line(ax, grid, means, entity="posterior mean", label="GP mean")
    theme.points(ax, runs, [truth(x) for x in runs], entity="observations", size=34.0)
    theme.reference_line(ax, x=6.0, label="last run")
    model = theme.entity_colour("posterior mean")
    reference = theme.entity_colour("truth")
    upper = means[-1] + 1.96 * sds[-1]
    theme.direct_labels(
        ax,
        [
            (12.0, upper, "95% band", model),
            (12.0, means[-1], "GP mean", model),
            (12.0, truth(12.0), "truth", reference),
        ],
    )
    ax.set_xlim(0.0, 12.0)
    theme.axis_labels(ax, "Input x", "Response y")
    theme.title_block(
        ax,
        f"Away from the 12 runs the GP variance grows to {facts['far_variance']} "
        f"at x = 12, from below {facts['inside_variance']} inside the design",
        "Squared-exponential GP fitted by maximum likelihood; band: mean "
        "+/- 1.96 predictive sd",
    )
    return figure, facts


def _ishigami_indices() -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """Return the analytic first-order and total Ishigami indices.

    With ``a = 7`` and ``b = 0.1`` the partial variances are
    ``D1 = b pi^4/5 + b^2 pi^8/50 + 1/2``, ``D2 = a^2/8``, ``D3 = 0`` and
    ``D13 = 8 b^2 pi^8/225``.

    Equation:
        [EQ-LAB-ISHIGAMI-001], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    References:
        [REF-SALTELLI-2008, Chapter 4].

    """
    a, b, pi = 7.0, 0.1, math.pi
    d1 = b * pi**4 / 5.0 + b * b * pi**8 / 50.0 + 0.5
    d2 = a * a / 8.0
    d13 = 8.0 * b * b * pi**8 / 225.0
    total = d1 + d2 + d13
    return (d1 / total, d2 / total, 0.0), (
        (d1 + d13) / total,
        d2 / total,
        d13 / total,
    )


def draw_pce_loo_by_degree() -> tuple[Any, dict[str, str]]:
    """Plot the PCE leave-one-out error and the worst index error by degree.

    Equation:
        [EQ-SUR-PCE-004], [EQ-SUR-PCE-006], [EQ-SUR-PCE-008].

    References:
        [REF-BLATMAN-SUDRET-2011], [REF-SUDRET-2008].

    """
    from probstat.distributions import Uniform
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.surrogates import fit_polynomial_chaos
    from probstat.uq import QuantityOfInterest

    model = ProbabilisticModel(
        JointInput(
            tuple(
                RandomInput(name, Uniform(-math.pi, math.pi))
                for name in ("x1", "x2", "x3")
            )
        ),
        lambda p: {"y": _ishigami_point(p["x1"], p["x2"], p["x3"])},
        (QuantityOfInterest("y", "Ishigami response"),),
    )
    exact_first, exact_total = _ishigami_indices()
    degrees = list(range(2, 11))
    loo: list[float] = []
    worst: list[float] = []
    for degree in degrees:
        pce = fit_polynomial_chaos(
            model, "y", degree=degree, basis="legendre", q_norm=0.75, seed=1
        )
        indices = pce.sobol_indices()
        estimated = [indices.first_order[n] for n in indices.names] + [
            indices.total_order[n] for n in indices.names
        ]
        exact = [*exact_first, *exact_total]
        loo.append(pce.loo_error)
        worst.append(max(abs(e - x) for e, x in zip(estimated, exact, strict=True)))
    facts = {
        "loo_low": f"{loo[0]:.2f}",
        "loo_high": f"{loo[-1]:.4f}",
        "index_error": f"{worst[-1]:.4f}",
        "index_error_low": f"{worst[0]:.2f}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    loo_colour = palette.observed
    index_colour = theme.role_colour("series-3")
    theme.line(ax, degrees, loo, color=loo_colour)
    theme.points(ax, degrees, loo, color=loo_colour)
    theme.line(ax, degrees, worst, color=index_colour)
    theme.points(ax, degrees, worst, color=index_colour, marker="s")
    theme.reference_line(ax, y=1.0, label="as bad as the mean")
    ax.set_yscale("log")
    _plain_log(ax.yaxis)
    ax.set_xticks(degrees)
    ax.set_xlim(1.6, 10.4)
    _label(
        ax,
        degrees[3],
        loo[3],
        "LOO error",
        loo_colour,
        offset=(6.0, 8.0),
        ha="left",
        va="bottom",
    )
    _label(
        ax,
        degrees[3],
        worst[3],
        "largest index error",
        index_colour,
        offset=(0.0, -10.0),
        ha="center",
        va="top",
    )
    theme.axis_labels(ax, "PCE degree p (hyperbolic q = 0.75)", "Error")
    theme.title_block(
        ax,
        f"The LOO error falls from {facts['loo_low']} to {facts['loo_high']}, and "
        f"only then are all six Sobol indices within {facts['index_error']}",
        "Legendre PCE of Ishigami on an LHS design of twice as many runs as "
        "terms; index error against the analytic values",
    )
    return figure, facts


# --------------------------------------------------------------------------
# VV-UQ foundations and model validation
# --------------------------------------------------------------------------


def draw_grid_convergence_order() -> tuple[Any, dict[str, str]]:
    """Plot the trapezoid error against step size with the fitted order.

    Equation:
        [EQ-VV-LSQ-001].

    References:
        [REF-ECA-HOEKSTRA-2014], [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    from probstat.numerics.quadrature import composite_trapezoid
    from probstat.uq import least_squares_convergence

    intervals = (8, 12, 16, 24, 32)
    spacings = [1.0 / n for n in intervals]
    values = [
        composite_trapezoid(math.exp, 0.0, 1.0, intervals=n).value for n in intervals
    ]
    fit = least_squares_convergence(spacings, values)
    exact = math.e - 1.0
    errors = [abs(value - exact) for value in values]
    fine = min(spacings)
    facts = {
        "order": f"{fit.order:.5f}",
        "extrapolation_error": f"{abs(fit.extrapolated - exact):.0e}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.2))
    grid = [0.028 * (0.15 / 0.028) ** (k / 100) for k in range(101)]
    model = [abs(fit.fine_error) * (h / fine) ** fit.order for h in grid]
    theme.line(ax, grid, model, color=theme.role_colour("model"), dashes=theme.FITTED)
    theme.points(ax, spacings, errors, entity="observations", size=40.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax.xaxis)
    _plain_log(ax.yaxis)
    ax.set_xticks(spacings)
    ax.set_xticklabels([f"1/{n}" for n in intervals])
    ax.set_yticks([1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ax.set_xlim(0.026, 0.16)
    _label(
        ax,
        spacings[2],
        errors[2],
        f"least-squares fit, p = {facts['order']}",
        theme.role_colour("model"),
        offset=(10.0, -10.0),
        ha="left",
        va="top",
    )
    _label(
        ax,
        spacings[3],
        errors[3],
        "true error",
        theme.entity_colour("observations"),
        offset=(-10.0, 8.0),
        ha="right",
        va="bottom",
    )
    theme.axis_labels(ax, "Step size h", "Error against e - 1")
    theme.title_block(
        ax,
        f"Five trapezoid grids give the observed order {facts['order']} and an "
        f"extrapolated value within {facts['extrapolation_error']} of e - 1",
        "Dots: true error of the composite trapezoid rule for exp on [0, 1]. "
        "Dashes: least-squares error model",
    )
    return figure, facts


def _load_example(name: str) -> Any:
    """Import a laboratory script of ``examples/`` as a module.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    key = f"_guide_example_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "examples" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load examples/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def draw_prediction_components() -> tuple[Any, dict[str, str]]:
    """Compare the forward, inverse, and component predictions at ``t = 10``.

    The numbers come from ``examples/uq_workflow_lab.py`` (seed 2026).

    Equation:
        [EQ-VAL-PRED-001], [EQ-VAL-PRED-002], [EQ-VAL-DISC-001].

    References:
        [REF-ROY-OBERKAMPF-2011].

    """
    lab = _load_example("uq_workflow_lab")
    results = _quiet(lab.run)
    forward = results.forward
    parametric = results.components.parametric_interval
    combined = results.components.combined_interval
    truth = results.truth

    def span(interval: tuple[float, float]) -> str:
        """Format an interval with three decimals.

        References:
            [REF-PROBSTAT-REPORTING]. No equation is evaluated.

        """
        return f"[{interval[0]:.3f}, {interval[1]:.3f}]"

    facts = {
        "truth": f"{truth:.3f}",
        "parametric": span(parametric),
        "combined": span(combined),
    }
    prior = (forward.quantile(0.05), forward.quantile(0.95))
    rows = [
        ("forward: prior push-forward", "5-95%", prior, forward.mean),
        (
            "inverse: calibrated parameters",
            "95%",
            parametric,
            results.components.parametric_mean,
        ),
        (
            "plus model-form and numerical",
            "95%",
            combined,
            results.components.parametric_mean,
        ),
    ]
    figure, (ax,) = _panels(figsize=(7.2, 3.6))
    palette = theme.active_palette()
    positions = list(range(len(rows)))[::-1]
    for position, (_, level, interval, centre) in zip(positions, rows, strict=True):
        covers = interval[0] <= truth <= interval[1]
        colour = palette.observed if covers else palette.muted
        ax.hlines(
            position, interval[0], interval[1], color=colour, linewidth=3.0, zorder=2
        )
        ax.vlines(
            interval,
            position - 0.12,
            position + 0.12,
            color=colour,
            linewidth=1.6,
            zorder=2,
        )
        theme.points(ax, [centre], [position], color=colour, size=40.0)
        _label(
            ax,
            prior[1],
            position,
            f"{level} {span(interval)}",
            palette.ink,
            offset=(14.0, 0.0),
            bold=False,
        )
    theme.reference_line(ax, x=truth, label=f"truth {facts['truth']}", entity="truth")
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0.0, 2.1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    theme.axis_labels(ax, "y at t = 10", x_unit="mg")
    theme.title_block(
        ax,
        f"Calibration sharpens the prediction but misses the truth "
        f"{facts['truth']}; model-form error restores coverage",
        "Dot: mean; bar: interval at t = 10, beyond the validation data at "
        "t = 7, 8, 9; green bars cover the truth",
    )
    return figure, facts


def _spring_assessments() -> tuple[Any, Any]:
    """Assess the spring model against unbiased and +1 mm biased data.

    The setting is the one of ``tests/test_model_validation.py``: ``y = k
    load`` with ``k ~ N(2, 0.1^2)``, loads 1 to 4 with six replicates each,
    measurement sd 0.05 mm, assessment seed 3 with 400 samples.

    Equation:
        [EQ-VAL-AREA-001], [EQ-VAL-UPOOL-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """
    from probstat.distributions import Normal
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.uq import NumericalErrorBudget, QuantityOfInterest
    from probstat.validation import ValidationExperiment, assess_validation

    model = ProbabilisticModel(
        JointInput.independent(RandomInput("k", Normal(2.0, 0.1))),
        lambda p: {"y": p["k"] * p["load"]},
        (QuantityOfInterest("y", "deflection", "mm"),),
        name="spring",
    )

    def experiments(bias: float) -> list[Any]:
        """Return four seeded experiments, optionally shifted by ``bias``.

        References:
            [REF-FERSON-OBERKAMPF-GINZBURG-2008].

        """
        rng = random.Random(11)
        made = []
        for index, load in enumerate((1.0, 2.0, 3.0, 4.0)):
            values = [
                rng.gauss(2.0, 0.1) * load + rng.gauss(0.0, 0.05) + bias
                for _ in range(6)
            ]
            made.append(
                ValidationExperiment(
                    f"test-{index}",
                    tuple(values),
                    {"load": load},
                    measurement_uncertainty=0.05,
                )
            )
        return made

    def assess(bias: float) -> Any:
        """Run the scoped assessment for one data set.

        References:
            [REF-FERSON-OBERKAMPF-GINZBURG-2008].

        """
        return assess_validation(
            model,
            "y",
            experiments(bias),
            tolerance=0.4,
            sample_size=400,
            seed=3,
            numerical_uncertainty=NumericalErrorBudget(discretization=0.01),
        )

    return assess(0.0), assess(1.0)


def draw_validation_area_metric() -> tuple[Any, dict[str, str]]:
    """Shade the area metric of the worst experiment, unbiased and biased.

    Equation:
        [EQ-VAL-AREA-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """
    unbiased, biased = _spring_assessments()
    figure, axes = _panels(2, figsize=(7.6, 4.0), wspace=0.12)
    palette = theme.active_palette()
    model = theme.role_colour("model")
    worst: list[float] = []
    chosen = [
        max(assessment.experiments, key=lambda e: e.area)
        for assessment in (unbiased, biased)
    ]
    every = [
        value
        for item in chosen
        for value in (*item.observable_predictions, *item.experiment.observations)
    ]
    low, high = min(every) - 0.15, max(every) + 0.15
    for ax, item, name in zip(axes, chosen, ("unbiased", "biased +1 mm"), strict=True):
        worst.append(item.area)
        predicted = sorted(item.observable_predictions)
        observed = sorted(item.experiment.observations)
        grid = [low, *sorted({*predicted, *observed}), high]

        def cdf(sample: list[float], x: float) -> float:
            """Return the right-continuous empirical CDF at ``x``.

            References:
                [REF-CASELLA-BERGER-2002, Section 5.5].

            """
            import bisect

            return bisect.bisect_right(sample, x) / len(sample)

        pred = [cdf(predicted, x) for x in grid]
        data = [cdf(observed, x) for x in grid]
        ax.fill_between(
            grid,
            pred,
            data,
            step="post",
            color=palette.muted,
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax.step(
            grid, pred, where="post", color=model, linewidth=theme.LINE_WIDTH, zorder=3
        )
        ax.step(
            grid,
            data,
            where="post",
            color=palette.observed,
            linewidth=theme.LINE_WIDTH,
            zorder=4,
        )
        ax.set_xlim(low, high)
        ax.set_ylim(0.0, 1.05)
        load = item.experiment.conditions["load"]
        ax.annotate(
            f"{name}\nload {load:g}, area {item.area:.2f} mm",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.ink,
        )
        mid_p = predicted[len(predicted) // 2]
        _label(
            ax,
            mid_p,
            cdf(predicted, mid_p),
            "prediction",
            model,
            offset=(8.0, -6.0),
            va="top",
            boxed=True,
        )
        mid_d = observed[1]
        _label(
            ax,
            mid_d,
            cdf(observed, mid_d),
            "data",
            palette.observed,
            offset=(-8.0, 6.0),
            ha="right",
            va="bottom",
            boxed=True,
        )
        theme.axis_labels(ax, "Deflection", None, x_unit="mm")
    axes[0].set_ylabel(theme.axis_title("cumulative_probability"))
    axes[1].tick_params(axis="y", labelleft=False)
    facts = {"unbiased": f"{worst[0]:.2f}", "biased": f"{worst[1]:.2f}"}
    theme.title_block(
        axes[0],
        f"A 1 mm bias raises the largest area metric from {facts['unbiased']} "
        f"to {facts['biased']} mm, past the 0.4 mm tolerance",
        "Shaded: area between the predictive CDF (400 noisy draws) and the "
        "ECDF of six measurements",
    )
    return figure, facts


def draw_validation_u_pooling() -> tuple[Any, dict[str, str]]:
    """Plot the ECDF of pooled u-values for unbiased and biased data.

    Equation:
        [EQ-VAL-UPOOL-001], [EQ-VAL-UPOOL-002].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """
    unbiased, biased = _spring_assessments()
    facts = {
        "unbiased_area": f"{unbiased.u_pooling.area:.3f}",
        "biased_area": f"{biased.u_pooling.area:.3f}",
        "unbiased_p": f"{unbiased.u_pooling.ks.p_value:.3f}",
        "biased_p": f"{biased.u_pooling.ks.p_value:.1e}",
    }
    figure, (ax,) = _panels(figsize=(7.2, 4.4))
    palette = theme.active_palette()
    uniform = theme.role_colour("reference")
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=uniform,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=2,
    )
    colours = (palette.observed, theme.role_colour("series-4"))
    for assessment, colour in zip((unbiased, biased), colours, strict=True):
        xs, ys = _ecdf(assessment.u_pooling.u_values)
        ax.step(
            [0.0, *xs, 1.0],
            [0.0, *ys, 1.0],
            where="post",
            color=colour,
            linewidth=theme.LINE_WIDTH,
            zorder=3,
        )
    _label(ax, 0.62, 0.62, "uniform", uniform, offset=(8.0, -8.0), va="top")
    _label(
        ax,
        0.32,
        sum(u <= 0.32 for u in unbiased.u_pooling.u_values) / 24,
        f"unbiased: area {facts['unbiased_area']}",
        palette.observed,
        offset=(-8.0, 10.0),
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        0.6,
        0.0,
        f"biased +1 mm: area {facts['biased_area']}",
        colours[1],
        offset=(0.0, 12.0),
        ha="center",
        va="bottom",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="both", color=palette.rule, linewidth=0.6)
    theme.axis_labels(ax, "u = F_i(y)", "Share of the 24 pooled observations")
    theme.title_block(
        ax,
        f"Pooled u-values stay uniform for unbiased data (KS p = "
        f"{facts['unbiased_p']}) but pile up at 1 under a 1 mm bias "
        f"(p = {facts['biased_p']})",
        "Each observation mapped through the predictive CDF of its own load; "
        "a calibrated model follows the diagonal",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Forecast calibration
# --------------------------------------------------------------------------


def _forecast_cases() -> tuple[dict[str, list[Any]], list[float]]:
    """Draw the 2000 cases and the three forecasters of the calibration lab.

    Same generator order and seed (20261012) as
    ``examples/forecast_calibration_lab.py``.

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

    """
    from probstat.distributions import Normal

    generator = random.Random(20261012)
    size = 2000
    means = [generator.uniform(-2.0, 2.0) for _ in range(size)]
    scales = [0.5 + generator.random() for _ in range(size)]
    outcomes = [generator.gauss(m, s) for m, s in zip(means, scales, strict=True)]
    centre = math.fsum(outcomes) / size
    spread = math.sqrt(math.fsum((y - centre) ** 2 for y in outcomes) / size)
    laws = {
        "ideal": [Normal(m, s) for m, s in zip(means, scales, strict=True)],
        "overconfident": [
            Normal(m, 0.5 * s) for m, s in zip(means, scales, strict=True)
        ],
        "climatology": [Normal(centre, spread) for _ in means],
    }
    return laws, outcomes


def draw_forecast_pit_histograms() -> tuple[Any, dict[str, str]]:
    """Draw the PIT histograms of three forecasters as small multiples.

    Equation:
        [EQ-VAL-CAL-PIT-001], [EQ-VIZ-PITBAND-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-DAWID-1984].

    """
    from probstat.validation import pit_histogram, pit_values
    from probstat.visualization.hierarchical_plots import binomial_band

    laws, outcomes = _forecast_cases()
    names = ("ideal", "overconfident", "climatology")
    bins = 10
    histograms = {
        name: pit_histogram(pit_values(laws[name], outcomes), bins=bins)
        for name in names
    }
    p_values = {name: histograms[name].chi_square.p_value for name in names}
    counts = {name: histograms[name].counts for name in names}
    n = len(outcomes)
    expected = n / bins
    edge = (counts["overconfident"][0] + counts["overconfident"][-1]) / (2 * expected)
    facts = {
        "ideal_p": f"{p_values['ideal']:.2f}",
        "edge_ratio": f"{edge:.2f}",
        "climatology_p": f"{p_values['climatology']:.1e}",
    }
    low, high = binomial_band(n, bins, 0.95)

    figure, axes = _panels(3, figsize=(7.6, 3.9), sharey=True, wspace=0.14)
    palette = theme.active_palette()
    top = 1.12 * max(max(c) for c in counts.values())
    for ax, name in zip(axes, names, strict=True):
        colour = theme.entity_colour(name)
        centres = [(k + 0.5) / bins for k in range(bins)]
        ax.axhspan(
            low,
            high,
            color=palette.muted,
            alpha=theme.MAX_BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
        ax.bar(
            centres,
            counts[name],
            width=0.86 / bins,
            color=colour,
            alpha=0.45,
            edgecolor=colour,
            linewidth=1.0,
            zorder=2,
        )
        ax.axhline(
            expected,
            color=theme.role_colour("reference"),
            linewidth=theme.REFERENCE_WIDTH,
            linestyle=theme.REFERENCE,
            zorder=3,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, top)
        ax.set_xticks([0.2, 0.5, 0.8])
        p_text = (
            f"p = {p_values[name]:.2f}"
            if p_values[name] >= 0.01
            else f"p = {p_values[name]:.1e}"
        )
        ax.annotate(
            name,
            xy=(0.5, 0.98),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=theme.ANNOTATION_SIZE,
            fontweight="bold",
            color=colour,
        )
        ax.annotate(
            p_text,
            xy=(0.5, 0.91),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.ink,
        )
        theme.axis_labels(ax, theme.axis_title("pit_value"), None)
    axes[0].set_ylabel(theme.axis_title("count"))
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    theme.title_block(
        axes[0],
        f"Only the ideal forecaster has a flat PIT histogram (p = "
        f"{facts['ideal_p']}); the overconfident one puts {facts['edge_ratio']}x "
        "the expected share in the outer bins",
        f"{n:,} cases; dashed line: uniform count, grey band: 95% binomial "
        "range per bin; p from the chi-square test",
    )
    return figure, facts


def draw_forecast_reliability() -> tuple[Any, dict[str, str]]:
    """Draw the reliability diagram of the ideal and overconfident forecasts.

    Equation:
        [EQ-VAL-CAL-REL-001].

    References:
        [REF-BROCKER-SMITH-2007], [REF-MURPHY-1973].

    """
    from probstat.validation import reliability_diagram

    laws, outcomes = _forecast_cases()
    events = [y > 0.0 for y in outcomes]
    diagrams = {
        name: reliability_diagram(
            [law.sf(0.0) for law in laws[name]],
            events,
            bins=5,
            consistency_level=0.95,
            resamples=500,
            seed=20261012,
        )
        for name in ("ideal", "overconfident")
    }
    outside = {name: len(diagrams[name].outside_bands) for name in diagrams}
    used = len([b for b in diagrams["overconfident"].bins if b.count > 0])
    facts = {
        "overconfident_outside": f"{outside['overconfident']} of {used}",
        "ideal_outside": f"{outside['ideal']} of {used}",
    }

    figure, (ax,) = _panels(figsize=(7.2, 4.8))
    palette = theme.active_palette()
    theme.line(ax, [0.0, 1.0], [0.0, 1.0], entity="perfect reliability", zorder=2)
    for shift, name in ((-0.012, "ideal"), (0.012, "overconfident")):
        colour = theme.entity_colour(name)
        items = [b for b in diagrams[name].bins if b.count > 0]
        xs = [b.mean_forecast + shift for b in items]
        ax.vlines(
            xs,
            [b.consistency_lower for b in items],
            [b.consistency_upper for b in items],
            colors=colour,
            linewidth=6.0,
            alpha=0.3,
            zorder=2,
        )
        ys = [b.observed_frequency for b in items]
        theme.line(ax, xs, ys, color=colour, width=theme.CONTEXT_WIDTH)
        theme.points(ax, xs, ys, entity=name, size=40.0)
    ideal = [b for b in diagrams["ideal"].bins if b.count > 0]
    over = [b for b in diagrams["overconfident"].bins if b.count > 0]
    _label(
        ax,
        ideal[3].mean_forecast,
        ideal[3].observed_frequency,
        "ideal",
        theme.entity_colour("ideal"),
        offset=(-12.0, 8.0),
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        over[3].mean_forecast,
        over[3].observed_frequency,
        "overconfident",
        theme.entity_colour("overconfident"),
        offset=(12.0, -8.0),
        va="top",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="both", color=palette.rule, linewidth=0.6)
    theme.axis_labels(ax, "Forecast probability of Y > 0", "Observed frequency")
    theme.title_block(
        ax,
        f"The overconfident forecaster leaves the 95% band in "
        f"{facts['overconfident_outside']} bins; the ideal one in "
        f"{facts['ideal_outside']}",
        "Markers: observed frequency per bin; bars: 95% range if reliable "
        "(500 resamples); dashes: perfect reliability",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------


def _conformal_draw(generator: random.Random, size: int) -> tuple[Any, list[float]]:
    """Draw ``(x, y)`` pairs of the heteroscedastic test process.

    ``x ~ U(0, 3)`` and ``y = sin 2x + 0.3 x + N(0, (0.2 + 0.2 x)^2)``.

    References:
        [REF-LEI-ETAL-2018].

    """
    xs = [generator.uniform(0.0, 3.0) for _ in range(size)]
    ys = [math.sin(2.0 * x) + 0.3 * x + generator.gauss(0.0, 0.2 + 0.2 * x) for x in xs]
    return [[x] for x in xs], ys


def _wrong_line(x: Sequence[float]) -> float:
    """Return the deliberately misspecified predictor ``0.1 + 0.3 x``.

    References:
        [REF-LEI-ETAL-2018].

    """
    return 0.1 + 0.3 * x[0]


def draw_conformal_coverage_staircase() -> tuple[Any, dict[str, str]]:
    """Plot empirical split-conformal coverage against the nominal level.

    Equation:
        [EQ-CONF-001], [EQ-CONF-002], [EQ-CONF-005].

    References:
        [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018].

    """
    from probstat.conformal import conformal_rank, split_conformal_regression

    size, test, replicates = 10, 40, 2000
    alphas = [round(0.05 * k, 2) for k in range(1, 11)]
    covered = {alpha: 0 for alpha in alphas}
    generator = random.Random(20261112)
    for _ in range(replicates):
        x_cal, y_cal = _conformal_draw(generator, size)
        x_new, y_new = _conformal_draw(generator, test)
        for alpha in alphas:
            regressor = split_conformal_regression(
                _wrong_line, x_cal, y_cal, alpha=alpha
            )
            covered[alpha] += sum(
                interval.lower <= y <= interval.upper
                for interval, y in zip(regressor.intervals(x_new), y_new, strict=True)
            )
    rate = {alpha: covered[alpha] / (replicates * test) for alpha in alphas}
    guaranteed = conformal_rank(size, 0.1) / (size + 1)
    # The naive index ceil((1 - alpha) n) = 9 without the (n + 1)/n correction.
    naive = math.ceil(round(0.9 * size, 9)) / (size + 1)
    facts = {
        "empirical": f"{rate[0.1]:.3f}",
        "guaranteed": f"{guaranteed:.3f}",
        "naive_value": f"{naive:.3f}",
    }

    figure, (ax,) = _panels(figsize=(6.4, 5.0))
    palette = theme.active_palette()
    fine = [0.45 + 0.549 * k / 1098 for k in range(1099)]
    stairs = [
        conformal_rank(size, round(1.0 - level, 4)) / (size + 1)
        if conformal_rank(size, round(1.0 - level, 4)) <= size
        else 1.0
        for level in fine
    ]
    ax.plot(
        fine,
        stairs,
        color=theme.role_colour("model"),
        linewidth=theme.CONTEXT_WIDTH,
        drawstyle="steps-post",
        zorder=2,
    )
    ax.plot(
        [0.45, 1.0],
        [0.45, 1.0],
        color=theme.role_colour("reference"),
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=2,
    )
    levels = [1.0 - alpha for alpha in alphas]
    theme.points(ax, levels, [rate[a] for a in alphas], entity="empirical", size=40.0)
    _label(
        ax,
        0.465,
        0.655,
        "guaranteed k/(n+1)",
        theme.role_colour("model"),
        offset=(0.0, 0.0),
        ha="left",
        va="bottom",
    )
    _label(
        ax,
        0.74,
        0.74,
        "nominal 1 - alpha",
        theme.role_colour("reference"),
        offset=(8.0, -8.0),
        va="top",
    )
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="both", color=palette.rule, linewidth=0.6)
    theme.axis_labels(
        ax, theme.axis_title("nominal_coverage"), theme.axis_title("empirical_coverage")
    )
    theme.title_block(
        ax,
        f"With 10 calibration points, 90% intervals cover {facts['empirical']}: "
        f"the guaranteed {facts['guaranteed']}, not the naive {facts['naive_value']}",
        f"Dots: coverage over {replicates:,} calibration sets x {test} test "
        "points; line: k/(n+1), k = ceil((1 - alpha)(n + 1))",
    )
    return figure, facts


def draw_conformal_normalized_scores() -> tuple[Any, dict[str, str]]:
    """Contrast absolute and normalized split-conformal bands on one data set.

    Equation:
        [EQ-CONF-002].

    References:
        [REF-LEI-ETAL-2018].

    """
    from probstat.conformal import empirical_coverage, split_conformal_regression

    generator = random.Random(2026)
    x_cal, y_cal = _conformal_draw(generator, 200)
    x_new, y_new = _conformal_draw(generator, 4000)

    def spread(x: Sequence[float]) -> float:
        """Return the known noise scale ``0.2 + 0.2 x``.

        References:
            [REF-LEI-ETAL-2018].

        """
        return 0.2 + 0.2 * x[0]

    def centre(x: Sequence[float]) -> float:
        """Return the true regression function ``sin 2x + 0.3 x``.

        References:
            [REF-LEI-ETAL-2018].

        """
        return math.sin(2.0 * x[0]) + 0.3 * x[0]

    methods = {
        "absolute residual": split_conformal_regression(
            centre, x_cal, y_cal, alpha=0.1
        ),
        "normalized by 0.2 + 0.2x": split_conformal_regression(
            centre, x_cal, y_cal, alpha=0.1, scale=spread
        ),
    }
    regions = {"low": lambda x: x < 1.0, "high": lambda x: x > 2.0}
    coverage: dict[str, dict[str, float]] = {}
    for name, regressor in methods.items():
        coverage[name] = {}
        for region, inside in regions.items():
            chosen = [(x, y) for x, y in zip(x_new, y_new, strict=True) if inside(x[0])]
            report = empirical_coverage(
                regressor.intervals([x for x, _ in chosen]), [y for _, y in chosen]
            )
            coverage[name][region] = report.coverage
    absolute, normalized = coverage.values()
    facts = {
        "absolute_low": f"{absolute['low']:.0%}",
        "absolute_high": f"{absolute['high']:.0%}",
        "normalized_low": f"{normalized['low']:.0%}",
        "normalized_high": f"{normalized['high']:.0%}",
    }

    figure, axes = _panels(2, figsize=(7.6, 4.0), sharey=True, wspace=0.08)
    palette = theme.active_palette()
    model = theme.role_colour("model")
    grid = [3.0 * k / 120 for k in range(121)]
    shown = list(zip(x_new[:400], y_new[:400], strict=True))
    for ax, (name, regressor) in zip(axes, methods.items(), strict=True):
        ax.scatter(
            [x[0] for x, _ in shown],
            [y for _, y in shown],
            s=6.0,
            color=palette.observed,
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )
        bands = regressor.intervals([[x] for x in grid])
        theme.band(
            ax, grid, [b.lower for b in bands], [b.upper for b in bands], color=model
        )
        theme.line(
            ax,
            grid,
            [b.prediction for b in bands],
            color=model,
            width=theme.CONTEXT_WIDTH,
        )
        theme.line(ax, grid, [b.lower for b in bands], color=model, width=0.9)
        theme.line(ax, grid, [b.upper for b in bands], color=model, width=0.9)
        values = coverage[name]
        ax.annotate(
            f"{name}\ncovers {values['low']:.0%} at x < 1, "
            f"{values['high']:.0%} at x > 2",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.ink,
        )
        ax.set_xlim(0.0, 3.0)
        theme.axis_labels(ax, "Input x", None)
    axes[0].set_ylabel("Response y")
    axes[1].tick_params(axis="y", labelleft=False)
    theme.title_block(
        axes[0],
        f"Absolute scores cover {facts['absolute_low']} where the noise is small "
        f"and {facts['absolute_high']} where it is large; normalized scores give "
        f"{facts['normalized_low']} and {facts['normalized_high']}",
        "90% split-conformal bands around the true mean sin 2x + 0.3x from 200 "
        "calibration points; dots: 400 of 4,000 test points",
    )
    return figure, facts


FIGURES: list[GuideFigure] = [
    GuideFigure(
        name="fitting-bootstrap-calibration",
        guide="uncertainty-modeling-and-fitting",
        alt="Empirical CDFs of naive and bootstrap Kolmogorov-Smirnov p-values "
        "from 80 normal samples tested against fitted normal laws, with the "
        "uniform diagonal",
        draw=draw_fitting_bootstrap_calibration,
    ),
    GuideFigure(
        name="fitting-bearing-tails",
        guide="uncertainty-modeling-and-fitting",
        alt="Log-log plot of 23 ball-bearing lives with fitted Gamma, LogNormal, "
        "and Weibull CDFs and their 1 percent lives marked",
        draw=draw_fitting_bearing_tails,
    ),
    GuideFigure(
        name="propagation-design-convergence",
        guide="uncertainty-propagation",
        alt="Log-log plot of the spread of replicate mean estimates against "
        "sample size for Monte Carlo, Latin hypercube, and Sobol designs",
        draw=draw_propagation_design_convergence,
    ),
    GuideFigure(
        name="variance-reduction-budget",
        guide="advanced-monte-carlo",
        alt="Dot plot of five estimates of the mean of U squared with 95 "
        "percent intervals and the exact value one third",
        draw=draw_variance_reduction_budget,
    ),
    GuideFigure(
        name="mlmc-level-variances",
        guide="advanced-monte-carlo",
        alt="Two panels of a multilevel Monte Carlo run: variances per level "
        "on a log scale and the number of samples per level",
        draw=draw_mlmc_level_variances,
    ),
    GuideFigure(
        name="ishigami-sobol-indices",
        guide="sensitivity-analysis",
        alt="Dumbbell chart of first-order and total Sobol indices of the three "
        "Ishigami inputs on a common zero to one scale",
        draw=draw_ishigami_sobol_indices,
    ),
    GuideFigure(
        name="gp-variance-beyond-data",
        guide="surrogate-models",
        alt="Gaussian-process mean and 95 percent band fitted to 12 runs on 0 "
        "to 6 and extended to 12, with the true function",
        draw=draw_gp_variance_beyond_data,
    ),
    GuideFigure(
        name="pce-loo-by-degree",
        guide="surrogate-models",
        alt="Log-scale plot of the polynomial chaos leave-one-out error and the "
        "largest Sobol index error against degree 2 to 10",
        draw=draw_pce_loo_by_degree,
    ),
    GuideFigure(
        name="grid-convergence-order",
        guide="vv-uq-foundations",
        alt="Log-log plot of trapezoid-rule errors on five step sizes with the "
        "fitted least-squares error model",
        draw=draw_grid_convergence_order,
    ),
    GuideFigure(
        name="prediction-uncertainty-components",
        guide="vv-uq-foundations",
        alt="Dot plot of three prediction intervals at t equal to 10, prior, "
        "calibrated parametric, and with model-form error, against the truth",
        draw=draw_prediction_components,
    ),
    GuideFigure(
        name="validation-area-metric",
        guide="model-validation",
        alt="Two panels each with a predictive CDF, a data ECDF of six "
        "measurements, and the shaded area between them",
        draw=draw_validation_area_metric,
    ),
    GuideFigure(
        name="validation-u-pooling",
        guide="model-validation",
        alt="Empirical CDFs of 24 pooled u-values for unbiased and biased data "
        "with the uniform diagonal",
        draw=draw_validation_u_pooling,
    ),
    GuideFigure(
        name="forecast-pit-histograms",
        guide="forecast-calibration",
        alt="Three PIT histograms for the ideal, overconfident, and "
        "climatological forecasters with the uniform count and binomial band",
        draw=draw_forecast_pit_histograms,
    ),
    GuideFigure(
        name="forecast-reliability-diagram",
        guide="forecast-calibration",
        alt="Reliability diagram of the ideal and overconfident forecasts of "
        "Y above 0 in five bins with consistency bars and the diagonal",
        draw=draw_forecast_reliability,
    ),
    GuideFigure(
        name="conformal-coverage-staircase",
        guide="conformal-prediction",
        alt="Empirical split-conformal coverage at ten nominal levels with the "
        "guaranteed staircase k over n plus 1 and the diagonal",
        draw=draw_conformal_coverage_staircase,
    ),
    GuideFigure(
        name="conformal-normalized-scores",
        guide="conformal-prediction",
        alt="Two panels of heteroscedastic test points with 90 percent "
        "conformal bands from absolute and from normalized scores",
        draw=draw_conformal_normalized_scores,
    ),
]
