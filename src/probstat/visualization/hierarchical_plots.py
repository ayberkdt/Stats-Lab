"""Plots for hierarchical models, Gaussian mixtures, and predictive checks.

The functions take result objects of :mod:`probstat.inference` (ADR-0040)
and never refit. Colour roles follow ADR-0024 and ADR-0044: raw group
estimates are context (muted), posterior quantities and fitted densities use
the model hue, data and simulated replicates the observed hue, and the
accent marks reference values (the pooled mean, an observed test statistic,
the uniform PIT level, the selected number of components).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.distributions import Binomial
from probstat.foundations.validation import coerce_finite_sample
from probstat.inference import (
    GaussianMixtureFit,
    HierarchicalNormalPosterior,
    MixtureSelection,
    PredictiveStatisticCheck,
    pit_uniformity,
)

from .calibration_plots import gaussian_mixture_interval
from .theme import (
    ANNOTATION_SIZE,
    FITTED,
    REFERENCE,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    direct_labels,
    line,
    new_axes,
    points,
    reference_line,
    themed,
    title_block,
)


def theta_intervals(
    posterior: HierarchicalNormalPosterior, probability: float = 0.95
) -> tuple[tuple[float, float], ...]:
    """Return exact central posterior intervals of every group effect.

    Given ``tau`` each ``theta_j`` is normal with the moments of
    ``conditional_theta``; mixing over the quadrature nodes of the gridded
    ``tau`` posterior with their weights gives the exact marginal, whose
    quantiles are found by bisection.

    Args:
        posterior: Gridded posterior of the hierarchical normal model.
        probability: Central probability strictly inside ``(0, 1)``.

    Returns:
        One ``(lower, upper)`` pair per group.

    Equation:
        [EQ-HIER-001], [EQ-HIER-002], [EQ-VIZ-MIXQ-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    largest = max(posterior.weights)
    nodes = [
        (w, t)
        for w, t in zip(posterior.weights, posterior.tau_grid, strict=True)
        if w > 1e-10 * largest
    ]
    moments = [posterior.conditional_theta(t) for _, t in nodes]
    weights = [w for w, _ in nodes]
    return tuple(
        gaussian_mixture_interval(
            [means[j] for means, _ in moments],
            [variances[j] for _, variances in moments],
            probability,
            weights=weights,
        )
        for j in range(posterior.groups)
    )


def tau_density(posterior: HierarchicalNormalPosterior) -> tuple[float, ...]:
    """Return the normalized marginal density of ``tau`` on its grid.

    The stored log density is unnormalized; it is exponentiated relative to
    its maximum and divided by its trapezoid integral over the grid.

    Args:
        posterior: Gridded posterior of the hierarchical normal model.

    Returns:
        Density values aligned with ``posterior.tau_grid``.

    Equation:
        [EQ-HIER-002].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-DAVIS-RABINOWITZ-1984].

    """
    peak = max(posterior.log_density)
    raw = [math.exp(v - peak) for v in posterior.log_density]
    grid = posterior.tau_grid
    area = math.fsum(
        0.5 * (raw[i] + raw[i + 1]) * (grid[i + 1] - grid[i])
        for i in range(len(grid) - 1)
    )
    return tuple(v / area for v in raw)


@themed
def plot_shrinkage(
    posterior: HierarchicalNormalPosterior,
    *,
    names: Sequence[str] | None = None,
    level: float = 0.95,
    unit: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot raw group estimates against their partially pooled posteriors.

    Each row is one group, sorted by its raw estimate: the raw ``y_j +- s_j``
    is muted context above, the posterior mean and exact central interval of
    ``theta_j`` are in the model hue below, and a thin arrow joins them. The
    posterior mean of ``mu`` is the accent reference line.

    Args:
        posterior: Output of :func:`probstat.inference.hierarchical_normal_posterior`.
        names: Group labels (default ``"group 1"``...).
        level: Central probability of the posterior intervals.
        unit: Optional unit of the effects.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``names`` does not match the number of groups or the
            level is invalid.

    Equation:
        [EQ-HIER-001], [EQ-HIER-004], [EQ-VIZ-MIXQ-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-RUBIN-1981], [REF-MATPLOTLIB].

    """
    groups = posterior.groups
    labels = (
        [f"group {j + 1}" for j in range(groups)] if names is None else list(names)
    )
    if len(labels) != groups:
        raise ValueError("names must match the number of groups")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    means, _ = posterior.theta_moments()
    limits = theta_intervals(posterior, level)
    mu, _ = posterior.mu_moments()
    shrink = posterior.expected_shrinkage()
    ys, ss = posterior.estimates, posterior.standard_errors
    order = sorted(range(groups), key=lambda j: (ys[j], j))
    axes = new_axes(ax, figsize=(7.2, 0.5 * groups + 2.2))
    palette = active_palette()
    offset = 0.17
    for row, j in enumerate(order):
        upper_row, lower_row = row + offset, row - offset
        axes.hlines(
            upper_row, ys[j] - ss[j], ys[j] + ss[j], color=palette.muted, linewidth=1.2
        )
        axes.scatter(
            [ys[j]],
            [upper_row],
            s=24.0,
            color=palette.muted,
            edgecolors=palette.canvas,
            linewidths=1.0,
            zorder=3,
        )
        low, high = limits[j]
        axes.hlines(lower_row, low, high, color=palette.model, linewidth=2.2)
        points(axes, [means[j]], [lower_row], color=palette.model, size=40.0)
        if abs(means[j] - ys[j]) > 1e-9:
            axes.annotate(
                "",
                xy=(means[j], lower_row + 0.05),
                xytext=(ys[j], upper_row - 0.05),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.3,head_width=0.15",
                    "color": palette.muted,
                    "linewidth": 0.7,
                    "shrinkA": 3.0,
                    "shrinkB": 4.0,
                },
                zorder=2,
            )
    reference_line(axes, x=mu, label=f"mu = {mu:.3g}")
    axes.set_yticks(range(groups))
    axes.set_yticklabels([labels[j] for j in order])
    axes.set_ylim(-0.7, groups - 0.3)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    top = order[-1]
    pairs = list(zip(ys, ss, strict=True))
    low_edge = min(*(y - s for y, s in pairs), *(lo for lo, _ in limits))
    high_edge = max(*(y + s for y, s in pairs), *(hi for _, hi in limits))
    span = high_edge - low_edge
    axes.set_xlim(low_edge - 0.03 * span, high_edge + 0.22 * span)
    direct_labels(
        axes,
        [
            (ys[top] + ss[top], groups - 1 + offset, "raw +/- 1 s.e.", palette.muted),
            (
                limits[top][1],
                groups - 1 - offset,
                f"posterior {level:.0%}",
                palette.model,
            ),
        ],
        gap_points=9.0,
    )
    axis_labels(axes, "Effect", None, x_unit=unit)
    average = math.fsum(shrink) / groups
    raw_range = max(ys) - min(ys)
    pooled_range = max(means) - min(means)
    title_block(
        axes,
        f"Partial pooling moves the {groups} estimates {average:.0%} of the way "
        f"to mu; their range shrinks from {raw_range:.1f} to {pooled_range:.1f}",
        f"Grey: raw estimate +/- 1 s.e. · colour: posterior mean and {level:.0%} "
        "interval (exact grid) · line: posterior mean of mu",
    )
    return axes


@themed
def plot_tau_posterior(
    posterior: HierarchicalNormalPosterior,
    *,
    draws: Sequence[float] | None = None,
    upper_probability: float = 0.995,
    bins: int = 40,
    ax: Any | None = None,
) -> Any:
    """Plot the exact marginal posterior of ``tau`` with optional MCMC draws.

    The line is the normalized grid density (:func:`tau_density`); NUTS (or
    any) draws of ``tau`` appear as a density histogram in the observed hue
    on the same scale, truncated at the ``upper_probability`` quantile.

    Args:
        posterior: Gridded posterior of the hierarchical normal model.
        draws: Optional sampled ``tau`` values.
        upper_probability: Quantile of ``tau`` that ends the axis.
        bins: Histogram bins for ``draws``.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: For an invalid ``upper_probability`` or ``bins``, or
            negative draws.

    Equation:
        [EQ-HIER-002].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-GELMAN-2006],
        [REF-BETANCOURT-GIROLAMI-2013], [REF-MATPLOTLIB].

    """
    if not 0.5 < upper_probability < 1.0:
        raise ValueError("upper_probability must lie in (0.5, 1)")
    if bins < 2:
        raise ValueError("bins must be at least two")
    end = posterior.tau_quantile(upper_probability)
    density = tau_density(posterior)
    shown = [i for i, t in enumerate(posterior.tau_grid) if t <= end]
    xs = [posterior.tau_grid[i] for i in shown]
    ys = [density[i] for i in shown]
    axes = new_axes(ax)
    palette = active_palette()
    subtitle = "Line: exact grid posterior"
    title_tail = ""
    if draws is not None:
        sample = coerce_finite_sample(draws, name="draws")
        if any(v < 0.0 for v in sample):
            raise ValueError("tau draws must be non-negative")
        width = end / bins
        axes.hist(
            [v for v in sample if v <= end],
            bins=[i * width for i in range(bins + 1)],
            weights=[1.0 / (len(sample) * width)] * sum(v <= end for v in sample),
            color=palette.observed,
            alpha=0.45,
            edgecolor=palette.canvas,
            linewidth=0.6,
            zorder=1,
        )
        mean = math.fsum(sample) / len(sample)
        subtitle += f" · bars: {len(sample):,} MCMC draws (density scale)"
        title_tail = f"; MCMC mean {mean:.3g} vs exact {posterior.tau_mean():.3g}"
    line(axes, xs, ys, color=palette.model, zorder=4.0)
    median = posterior.tau_quantile(0.5)
    upper = posterior.tau_quantile(0.95)
    axes.set_xlim(0.0, end)
    axes.set_ylim(0.0, None)
    axis_labels(axes, "Between-group sd tau", "Posterior density")
    direct_labels(
        axes, [(xs[len(xs) // 2], ys[len(ys) // 2], "exact posterior", palette.model)]
    )
    title_block(
        axes,
        f"tau has posterior median {median:.3g}; 95% of its mass lies below "
        f"{upper:.3g}{title_tail}",
        subtitle + f" · axis ends at the {upper_probability:.1%} quantile",
    )
    return axes


def _normal_pdf(x: float, mean: float, variance: float) -> float:
    """Return the normal density.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3]. Equation [EQ-GMM-001].

    """
    return math.exp(-0.5 * (x - mean) ** 2 / variance) / math.sqrt(
        2.0 * math.pi * variance
    )


@themed
def plot_mixture_fit(
    fit: GaussianMixtureFit,
    data: Sequence[float],
    *,
    bins: int | None = None,
    name: str = "x",
    ax: Any | None = None,
) -> Any:
    """Plot a one-dimensional Gaussian-mixture fit over its data histogram.

    The data are a density histogram in the observed hue; each weighted
    component ``w_k N(mu_k, s_k^2)`` is a thin ``FITTED`` model-hue curve and
    the mixture density their solid sum.

    Args:
        fit: Fit returned by :func:`probstat.inference.fit_gaussian_mixture`
            (or ``MixtureSelection.best``) on one-dimensional data.
        data: The data the mixture was fitted to.
        bins: Histogram bins (default ``ceil(2 n^(1/3))``, Rice rule).
        name: Variable name for the axis.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the mixture is not one-dimensional or fewer than two
            data points are supplied.

    Equation:
        [EQ-GMM-001].

    References:
        [REF-MCLACHLAN-PEEL-2000], [REF-MATPLOTLIB].

    """
    mixture = fit.mixture
    if mixture.dimension != 1:
        raise ValueError("only one-dimensional mixtures can be plotted")
    sample = coerce_finite_sample(data, name="data")
    if len(sample) < 2:
        raise ValueError("at least two data points are required")
    count = bins if bins is not None else math.ceil(2.0 * len(sample) ** (1.0 / 3.0))
    means = [m[0] for m in mixture.means]
    variances = [c[0][0] for c in mixture.covariances]
    reach = [4.0 * math.sqrt(v) for v in variances]
    low = min(min(sample), *(m - r for m, r in zip(means, reach, strict=True)))
    high = max(max(sample), *(m + r for m, r in zip(means, reach, strict=True)))
    grid = [low + (high - low) * i / 400 for i in range(401)]
    axes = new_axes(ax)
    palette = active_palette()
    axes.hist(
        sample,
        bins=count,
        density=True,
        color=palette.observed,
        alpha=0.4,
        edgecolor=palette.canvas,
        linewidth=0.6,
        zorder=1,
    )
    total = [0.0] * len(grid)
    order = sorted(range(mixture.components), key=lambda k: means[k])
    for k in order:
        curve = [
            mixture.weights[k] * _normal_pdf(x, means[k], variances[k]) for x in grid
        ]
        total = [a + b for a, b in zip(total, curve, strict=True)]
        line(axes, grid, curve, color=palette.model, dashes=FITTED, width=1.2)
        peak = mixture.weights[k] * _normal_pdf(means[k], means[k], variances[k])
        axes.annotate(
            f"w = {mixture.weights[k]:.2f}",
            (means[k], peak),
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=palette.model,
        )
    line(axes, grid, total, color=palette.model)
    axes.set_xlim(low, high)
    axes.set_ylim(0.0, max(total) * 1.2)
    axis_labels(axes, name, "Density")
    centres = ", ".join(f"{means[k]:.2f}" for k in order)
    title_block(
        axes,
        f"{mixture.components} components centred at {centres} fit the "
        f"{len(sample)} observations",
        f"Bars: data · solid: mixture density · dashed: weighted components · "
        f"log L = {fit.log_likelihood:.1f}",
    )
    return axes


@themed
def plot_mixture_selection(
    selection: MixtureSelection, *, ax: Any | None = None
) -> Any:
    """Plot the information criterion against the number of components.

    Args:
        selection: Output of :func:`probstat.inference.select_gaussian_mixture`.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If fewer than two candidate fits are present.

    Equation:
        [EQ-GMM-004].

    References:
        [REF-SCHWARZ-1978], [REF-AKAIKE-1974], [REF-KASS-RAFTERY-1995],
        [REF-MATPLOTLIB].

    """
    if len(selection.fits) < 2:
        raise ValueError("at least two candidate fits are required")
    ks = [fit.mixture.components for fit in selection.fits]
    scores = list(selection.scores)
    best = selection.best_index
    axes = new_axes(ax)
    palette = active_palette()
    order = sorted(range(len(ks)), key=lambda i: ks[i])
    line(
        axes,
        [ks[i] for i in order],
        [scores[i] for i in order],
        color=palette.observed,
        width=1.6,
    )
    points(axes, ks, scores, color=palette.observed, size=34.0)
    points(axes, [ks[best]], [scores[best]], color=palette.observed, size=90.0)
    criterion = selection.criterion.upper()
    reference_line(axes, x=float(ks[best]), label=f"{criterion} minimum")
    from matplotlib.ticker import MaxNLocator

    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis_labels(axes, "Number of components K", criterion)
    runner = min((i for i in range(len(ks)) if i != best), key=lambda i: scores[i])
    gap = scores[runner] - scores[best]
    title_block(
        axes,
        f"{criterion} selects K = {ks[best]}, {gap:.1f} below the runner-up "
        f"(K = {ks[runner]})",
        f"Lower is better; a {criterion} gap above 10 is very strong evidence · "
        f"n = {selection.best.sample_size}",
    )
    return axes


@themed
def plot_predictive_check(
    check: PredictiveStatisticCheck,
    *,
    bins: int = 30,
    ax: Any | None = None,
) -> Any:
    """Plot a posterior predictive check of one test quantity.

    A data-only statistic is drawn as the histogram of replicated values with
    the observed value as the accent reference line. A realized discrepancy
    ``T(y, theta)`` is drawn as replicated against observed values per draw
    with the identity line; the upper-tail p-value is the share of points on
    or above it.

    Args:
        check: One record of
            :func:`probstat.inference.posterior_predictive_pvalues`.
        bins: Histogram bins for a data-only statistic.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the check holds fewer than two draws.

    Equation:
        [EQ-PPC-001].

    References:
        [REF-GELMAN-MENG-STERN-1996], [REF-RUBIN-1984], [REF-MATPLOTLIB].

    """
    if len(check.replicated) < 2:
        raise ValueError("at least two replicated values are required")
    axes = new_axes(ax)
    palette = active_palette()
    count = len(check.replicated)
    if check.parameter_dependent:
        axes.scatter(
            check.observed,
            check.replicated,
            s=8.0,
            color=palette.observed,
            alpha=0.5,
            edgecolors="none",
            zorder=3,
        )
        low = min(min(check.observed), min(check.replicated))
        high = max(max(check.observed), max(check.replicated))
        axes.plot(
            [low, high],
            [low, high],
            color=palette.accent,
            linewidth=REFERENCE_WIDTH,
            linestyle=REFERENCE,
            zorder=2,
        )
        axes.grid(axis="both", color=palette.rule, linewidth=0.6)
        axis_labels(
            axes, f"Realized {check.name}  T(y, theta)", "Replicated  T(y_rep, theta)"
        )
        direct_labels(axes, [(high, high, "T_rep = T_obs", palette.accent)])
        subtitle = (
            f"Dots: {count:,} posterior draws · p = share on or above the line · "
            f"two-sided {check.two_sided_p_value:.2f}"
        )
    else:
        observed = check.observed[0]
        axes.hist(
            check.replicated,
            bins=bins,
            color=palette.observed,
            alpha=0.55,
            edgecolor=palette.canvas,
            linewidth=0.6,
            zorder=1,
        )
        reference_line(axes, x=observed, label=f"observed {observed:.3g}")
        low, high = axes.get_xlim()
        if observed > high - 0.02 * (high - low) or observed < low:
            axes.set_xlim(min(low, observed), max(high, observed) + 0.1 * (high - low))
        axis_labels(axes, f"Replicated {check.name}", "Draws")
        subtitle = (
            f"Bars: {count:,} replicated datasets · line: observed value · "
            f"two-sided p = {check.two_sided_p_value:.2f}"
        )
    if check.two_sided_p_value < 0.05:
        title = (
            f"The model rarely reproduces the observed {check.name}: "
            f"p = {check.p_value:.3f}"
        )
    else:
        title = (
            f"The model reproduces the observed {check.name}: "
            f"p = {check.p_value:.2f}"
        )
    title_block(axes, title, subtitle)
    return axes


def binomial_band(count: int, bins: int, confidence: float = 0.95) -> tuple[int, int]:
    """Return pointwise central limits of one bin count under uniformity.

    Each of ``bins`` equal-width bins of ``count`` independent uniform values
    holds ``Binomial(count, 1/bins)`` values; the limits are the smallest
    counts whose CDF reaches ``(1 -+ confidence)/2``.

    Args:
        count: Number of values (positive).
        bins: Number of bins (at least two).
        confidence: Central probability strictly inside ``(0, 1)``.

    Returns:
        Lower and upper count limits.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-VIZ-PITBAND-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.2],
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

    """
    if count < 1 or bins < 2:
        raise ValueError("count must be positive and bins at least two")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    law = Binomial(count, 1.0 / bins)
    tail = 0.5 * (1.0 - confidence)
    cumulative = 0.0
    lower: int | None = None
    for k in range(count + 1):
        cumulative += law.pmf(k)
        if lower is None and cumulative >= tail - 1e-12:
            lower = k
        if cumulative >= 1.0 - tail - 1e-12:
            return (lower if lower is not None else k), k
    return (lower if lower is not None else count), count


@themed
def plot_pit_histogram(
    values: Sequence[float],
    *,
    bins: int | None = None,
    confidence: float = 0.95,
    ax: Any | None = None,
) -> Any:
    """Plot a PIT histogram with the uniform level and a binomial band.

    Under a calibrated model PIT values are uniform, so every bin count is
    ``Binomial(n, 1/B)``; the muted band spans its pointwise central limits
    (:func:`binomial_band`) and the accent line is the expected count
    ``n/B``. Posterior (not leave-one-out) PIT values are pulled toward
    ``1/2``, so a hump in the middle is expected there.

    Args:
        values: PIT values in ``[0, 1]``.
        bins: Number of equal-width bins (default ``max(2, round(sqrt(n)))``
            capped at 20).
        confidence: Pointwise band level.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: For values outside ``[0, 1]``, fewer than two values, or
            invalid ``bins``.

    Equation:
        [EQ-PPC-002], [EQ-VIZ-PITBAND-001], [EQ-INF-GOF-KS-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-DAWID-1984],
        [REF-MATPLOTLIB].

    """
    sample = coerce_finite_sample(values, name="values")
    if len(sample) < 2:
        raise ValueError("at least two PIT values are required")
    if any(v < 0.0 or v > 1.0 for v in sample):
        raise ValueError("PIT values must lie in [0, 1]")
    count = bins if bins is not None else min(20, max(2, round(math.sqrt(len(sample)))))
    if count < 2:
        raise ValueError("bins must be at least two")
    counts = [0] * count
    for v in sample:
        counts[min(int(v * count), count - 1)] += 1
    low, high = binomial_band(len(sample), count, confidence)
    expected = len(sample) / count
    axes = new_axes(ax)
    palette = active_palette()
    width = 1.0 / count
    axes.axhspan(low, high, color=palette.muted, alpha=0.14, linewidth=0.0, zorder=0)
    axes.bar(
        [(i + 0.5) * width for i in range(count)],
        counts,
        width=width * 0.92,
        color=palette.observed,
        alpha=0.75,
        linewidth=0.0,
        zorder=2,
    )
    reference_line(axes, y=expected, label="uniform")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, max(max(counts), high) * 1.3)
    axis_labels(axes, "PIT value", "Count")
    outside = sum(c < low or c > high for c in counts)
    ks = pit_uniformity(sample)
    if outside == 0:
        title = (
            f"All {count} PIT bins stay inside the {confidence:.0%} band of a "
            "calibrated model"
        )
    else:
        title = (
            f"{outside} of {count} PIT bins leave the {confidence:.0%} band: "
            "the predictive distribution is miscalibrated"
        )
    title_block(
        axes,
        title,
        f"Bars: {len(sample)} PIT values · band: binomial counts {low}-{high} "
        f"per bin · KS distance {ks.statistic:.2f}",
    )
    return axes
