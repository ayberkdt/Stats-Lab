"""Plotting helpers that keep mathematical definitions visible.

Every function draws inside :func:`~probstat.visualization.theme.chart_theme`
and follows its rules: semantic colours (observed data, model, reference),
assertion titles with muted subtitles, direct labels instead of legends where
possible, horizontal hairline grids, and no dual y-axes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from probstat.distributions import (
    Bernoulli,
    Categorical,
    Geometric,
    NegativeBinomial,
    Normal,
)
from probstat.foundations import (
    ContinuousProbabilityLaw,
    CumulativeDistribution,
    ProbabilityMassFunction,
    SurvivalFunction,
)
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.monte_carlo import EfficiencyComparison, MonteCarloEstimate
from probstat.probability import (
    FiniteEvent,
    FiniteJointDistribution,
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
)
from probstat.statistics import empirical_cdf
from probstat.uq import observed_order_of_accuracy, richardson_extrapolation

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    FITTED,
    active_palette,
    axis_labels,
    band,
    direct_labels,
    line,
    new_axes,
    note,
    points,
    reference_line,
    sequential_colormap,
    themed,
    title_block,
)


def _lollipops(
    ax: Any, positions: Iterable[float], masses: Iterable[float], colour: str
) -> None:
    """Draw thin stems with rimmed heads for probability masses.

    Args:
        ax: Target axes.
        positions: Horizontal positions.
        masses: Mass at each position.
        colour: Series colour.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    xs = tuple(positions)
    ys = tuple(masses)
    ax.vlines(xs, 0.0, ys, color=colour, linewidth=1.6, zorder=2)
    points(ax, xs, ys, color=colour, size=34.0)


def _value_labels(
    ax: Any, xs: Iterable[float], ys: Iterable[float], fmt: str = "{:.3f}"
) -> None:
    """Write selective value labels above marks in ink colour.

    Args:
        ax: Target axes.
        xs: Horizontal positions.
        ys: Values (also the label anchors).
        fmt: Format string for each value.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(
            fmt.format(y),
            (x, y),
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )


def _integer_locator() -> Any:
    """Return a tick locator that only places integer ticks.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)


def _multiplier_formatter() -> Any:
    """Return a formatter that writes ticks as plain multipliers (``10x``).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{value:g}x")


def _null_formatter() -> Any:
    """Return a formatter that suppresses tick labels.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import NullFormatter

    return NullFormatter()


@themed
def plot_bernoulli_pmf(distribution: Bernoulli, *, ax: Any | None = None) -> Any:
    """Plot and annotate the two masses of a Bernoulli distribution.

    Args:
        distribution: Validated Bernoulli model.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-DIST-BER-001].

    References:
        [REF-ROSS-2019, Section 4.2], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax, figsize=(5.6, 3.8))
    palette = active_palette()
    outcomes = (0, 1)
    masses = tuple(distribution.pmf(value) for value in outcomes)
    axes.bar(
        outcomes,
        masses,
        width=0.5,
        color=(palette.muted, palette.observed),
        edgecolor=palette.canvas,
        linewidth=1.0,
    )
    _value_labels(axes, outcomes, masses)
    axes.set(xticks=outcomes, xticklabels=("failure (0)", "success (1)"))
    axes.set_ylim(0.0, 1.08)
    axis_labels(axes, y="P(X = k)")
    title_block(
        axes,
        f"Success carries probability p = {distribution.p:.3g}",
        "Bernoulli law: all mass sits on the two outcomes 0 and 1",
    )
    return axes


@themed
def plot_integer_pmf(
    distribution: ProbabilityMassFunction[int],
    *,
    lower: int,
    upper: int,
    ax: Any | None = None,
) -> Any:
    """Plot an integer-valued PMF over an explicit finite viewing window.

    The window is explicit because count distributions such as Poisson may have
    infinite support; the subtitle reports the probability mass outside it.

    Args:
        distribution: Integer-valued object exposing ``pmf``.
        lower: Inclusive lower display bound.
        upper: Inclusive upper display bound, not below ``lower``.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If either bound is not an integer.
        ValueError: If ``upper < lower``.

    Equation:
        [EQ-DIST-BIN-001], [EQ-DIST-POIS-001].

    References:
        [REF-ROSS-2019, Chapter 4], [REF-MATPLOTLIB].

    """
    checked_lower = require_non_negative_integer(lower, name="lower")
    checked_upper = require_non_negative_integer(upper, name="upper")
    if checked_upper < checked_lower:
        raise ValueError("upper must not be below lower")
    support = tuple(range(checked_lower, checked_upper + 1))
    masses = tuple(distribution.pmf(value) for value in support)
    axes = new_axes(ax)
    palette = active_palette()
    _lollipops(axes, support, masses, palette.observed)
    mode = support[max(range(len(masses)), key=lambda index: masses[index])]
    outside = max(0.0, 1.0 - math.fsum(masses))
    axes.set_ylim(0.0, max(masses, default=0.0) * 1.15 or 1.0)
    axes.set_xlim(checked_lower - 0.6, checked_upper + 0.6)
    axes.xaxis.set_major_locator(_integer_locator())
    axis_labels(axes, "Count k", "P(X = k)")
    title_block(
        axes,
        f"{type(distribution).__name__} mass peaks at k = {mode}",
        f"Window k = {checked_lower}..{checked_upper}; mass outside the window "
        f"= {outside:.2g}",
    )
    return axes


@themed
def plot_categorical_pmf(
    distribution: Categorical[Any], *, ax: Any | None = None
) -> Any:
    """Plot an unordered categorical PMF without imposing numeric meaning.

    Categories are drawn as horizontal bars in declaration order; their order
    carries no meaning, so no axis implies a scale between labels.

    Args:
        distribution: Validated categorical distribution.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-DIST-CAT-001].

    References:
        [REF-ROSS-2019, Section 4.6], [REF-MATPLOTLIB].

    """
    labels = tuple(str(label) for label in distribution.support.values)
    masses = tuple(distribution.pmf(label) for label in distribution.support.values)
    axes = new_axes(ax, figsize=(6.4, 0.55 * len(labels) + 1.6))
    palette = active_palette()
    positions = tuple(range(len(labels)))[::-1]
    axes.barh(
        positions,
        masses,
        height=0.56,
        color=palette.observed,
        edgecolor=palette.canvas,
        linewidth=1.0,
    )
    for position, mass in zip(positions, masses, strict=True):
        axes.annotate(
            f"{mass:.3f}",
            (mass, position),
            xytext=(6.0, 0.0),
            textcoords="offset points",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    axes.set(yticks=positions, yticklabels=labels, xlim=(0.0, max(masses) * 1.2))
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    axes.spines["left"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axis_labels(axes, "Probability mass")
    title_block(
        axes,
        "Categorical masses on unordered labels",
        "Bar order follows declaration only; labels have no numeric scale",
    )
    return axes


@themed
def plot_waiting_count_pmfs(
    geometric: Geometric,
    negative_binomial: NegativeBinomial,
    *,
    upper: int = 20,
    ax: Any | None = None,
) -> Any:
    """Compare two failures-before-success PMFs on one integer window.

    Args:
        geometric: First-success waiting-count law.
        negative_binomial: Target-success waiting-count law.
        upper: Inclusive non-negative count-window endpoint.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If ``upper`` is not an integer.
        ValueError: If ``upper`` is negative.

    Equation:
        [EQ-DIST-GEOM-001], [EQ-DIST-NBIN-001].

    References:
        [REF-ROSS-2019, Section 4.5], [REF-MATPLOTLIB].

    """
    checked_upper = require_non_negative_integer(upper, name="upper")
    counts = tuple(range(checked_upper + 1))
    geometric_masses = tuple(geometric.pmf(k) for k in counts)
    negative_masses = tuple(negative_binomial.pmf(k) for k in counts)
    axes = new_axes(ax)
    palette = active_palette()
    first, second = palette.series[0], palette.series[2]
    line(axes, counts, geometric_masses, color=first, width=1.6)
    points(axes, counts, geometric_masses, color=first, size=22.0)
    line(axes, counts, negative_masses, color=second, width=1.6)
    points(axes, counts, negative_masses, color=second, size=22.0)
    axes.set_xlim(-0.5, checked_upper + 0.5)
    axes.xaxis.set_major_locator(_integer_locator())
    axes.set_ylim(0.0, 1.12 * max((*geometric_masses, *negative_masses)))
    axis_labels(axes, "Failures k before the target success", "P(X = k)")
    direct_labels(
        axes,
        (
            (counts[-1], geometric_masses[-1], f"Geometric p={geometric.p:g}", first),
            (
                counts[-1],
                negative_masses[-1],
                f"NegBin r={negative_binomial.r}, p={negative_binomial.p:g}",
                second,
            ),
        ),
    )
    title_block(
        axes,
        "Waiting for more successes shifts mass to longer runs",
        "Zero-based convention: k counts failures before the r-th success",
    )
    return axes


@themed
def plot_continuous_functions(
    distribution: ContinuousProbabilityLaw,
    *,
    lower: float,
    upper: float,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot the density above, and CDF with survival below, on a shared x-axis.

    Density can exceed one while CDF and survival are probabilities, so they
    live in two stacked panels with their own single y-axis instead of a
    misleading dual-axis overlay.

    Args:
        distribution: Continuous law that also exposes a direct survival method.
        lower: Finite lower viewing bound.
        upper: Finite upper viewing bound, strictly greater than ``lower``.
        points: Number of equally spaced evaluation points, at least two.
        ax: Optional axes for the density panel; the probability panel is
            appended below it.

    Returns:
        The density axes; its figure also contains the probability panel.

    Raises:
        TypeError: If bounds or points have invalid types, or survival is absent.
        ValueError: If bounds are invalid or fewer than two points are requested.

    Equation:
        [EQ-DIST-UNIF-001], [EQ-DIST-UNIF-002], [EQ-DIST-EXP-001],
        [EQ-DIST-EXP-002], [EQ-DIST-NORM-001], [EQ-DIST-NORM-005].

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3], [REF-MATPLOTLIB].

    """
    if not isinstance(distribution, SurvivalFunction):
        raise TypeError("distribution must expose a survival function")
    checked_lower = require_finite_real(lower, name="lower")
    checked_upper = require_finite_real(upper, name="upper")
    checked_points = require_non_negative_integer(points, name="points")
    if checked_upper <= checked_lower:
        raise ValueError("upper must be greater than lower")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    step = (checked_upper - checked_lower) / (checked_points - 1)
    xs = tuple(checked_lower + index * step for index in range(checked_points))
    densities = tuple(distribution.pdf(x) for x in xs)
    cdf = tuple(distribution.cdf(x) for x in xs)
    survival = tuple(distribution.sf(x) for x in xs)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    density_axes = new_axes(ax, figsize=(7.2, 5.4))
    palette = active_palette()
    probability_axes = make_axes_locatable(density_axes).append_axes(
        "bottom", size="100%", pad=0.35, sharex=density_axes
    )
    new_axes(probability_axes)
    line(density_axes, xs, densities, color=palette.observed)
    density_axes.fill_between(
        xs, 0.0, densities, color=palette.observed, alpha=0.12, linewidth=0.0
    )
    density_axes.set_ylim(0.0, max(densities) * 1.12 or 1.0)
    density_axes.tick_params(axis="x", labelbottom=False, length=0)
    density_axes.spines["bottom"].set_visible(False)
    axis_labels(density_axes, y="Density f(x)")
    line(probability_axes, xs, cdf, color=palette.series[0])
    line(probability_axes, xs, survival, color=palette.series[1])
    probability_axes.set_ylim(0.0, 1.05)
    axis_labels(probability_axes, "x", "Probability")
    direct_labels(
        probability_axes,
        (
            (xs[-1], cdf[-1], "CDF F(x)", palette.series[0]),
            (xs[-1], survival[-1], "survival S(x)", palette.series[1]),
        ),
    )
    title_block(
        density_axes,
        f"{type(distribution).__name__}: density, CDF, and survival",
        "Top: density (not a probability). Bottom: F(x) + S(x) = 1 everywhere",
    )
    return density_axes


@themed
def plot_finite_measure(
    measure: FiniteProbabilityMeasure[Any],
    *,
    highlight: FiniteEvent[Any] | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot outcome masses and optionally highlight one finite event.

    Args:
        measure: Validated finite probability measure.
        highlight: Optional event from the measure's sample space. Highlighted
            outcomes use the data hue; the rest stay muted context.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If ``highlight`` is not a finite event.
        ValueError: If ``highlight`` belongs to another sample space.

    Equation:
        [EQ-PROB-002].

    References:
        [REF-ROSS-2019, Chapter 2], [REF-MATPLOTLIB].

    """
    event_probability: float | None = None
    if highlight is not None:
        event_probability = measure.probability(highlight)
    axes = new_axes(ax)
    palette = active_palette()
    outcomes = tuple(measure.space)
    masses = tuple(measure.weights[outcome] for outcome in outcomes)
    colours = tuple(
        palette.observed if highlight is None or outcome in highlight else palette.rule
        for outcome in outcomes
    )
    positions = tuple(range(len(outcomes)))
    axes.bar(
        positions,
        masses,
        color=colours,
        width=0.6,
        edgecolor=palette.canvas,
        linewidth=1.0,
    )
    _value_labels(axes, positions, masses)
    axes.set(
        xticks=positions,
        xticklabels=tuple(str(outcome) for outcome in outcomes),
        ylim=(0.0, max(masses) * 1.2),
    )
    axis_labels(axes, "Outcome", "Probability mass")
    if event_probability is None:
        title_block(axes, "Finite probability measure", "Mass assigned to each outcome")
    else:
        title_block(
            axes,
            f"P(A) = {event_probability:.3f}: the sum of the highlighted masses",
            "Highlighted bars form event A; grey bars lie outside it",
        )
    return axes


@themed
def plot_random_variable_pmf(
    variable: FiniteRandomVariable[Any, Any], *, ax: Any | None = None
) -> Any:
    """Plot the induced PMF of a finite random variable.

    Args:
        variable: Finite random variable with hashable support values.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-RV-001].

    References:
        [REF-ROSS-2019, Chapters 4-5], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax, figsize=(5.6, 3.8))
    palette = active_palette()
    support = variable.support
    masses = tuple(variable.pmf(value) for value in support)
    positions = tuple(range(len(support)))
    _lollipops(axes, positions, masses, palette.observed)
    _value_labels(axes, positions, masses)
    axes.set(
        xticks=positions,
        xticklabels=tuple(str(value) for value in support),
        ylim=(0.0, max(masses) * 1.22),
        xlim=(-0.6, len(positions) - 0.4),
    )
    axis_labels(axes, f"{variable.name} value", "Probability mass")
    title_block(
        axes,
        f"Induced law of {variable.name}",
        "Each mass is the probability of the outcomes mapped to that value",
    )
    return axes


@themed
def plot_joint_pmf(
    joint: FiniteJointDistribution[Any, Any, Any], *, ax: Any | None = None
) -> Any:
    """Plot a finite joint PMF as an annotated single-hue mass matrix.

    Args:
        joint: Joint law of two finite random variables.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-RV-010].

    References:
        [REF-ROSS-2019, Chapter 6], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax, figsize=(4.8, 4.2))
    palette = active_palette()
    first_support = joint.first.support
    second_support = joint.second.support
    matrix = tuple(
        tuple(joint.joint_pmf(first, second) for first in first_support)
        for second in second_support
    )
    maximum_mass = max(max(row) for row in matrix) or 1.0
    axes.pcolormesh(
        tuple(column - 0.5 for column in range(len(first_support) + 1)),
        tuple(row - 0.5 for row in range(len(second_support) + 1)),
        matrix,
        cmap=sequential_colormap(),
        vmin=0.0,
        vmax=1.6 * maximum_mass,
        edgecolors=palette.canvas,
        linewidth=3.0,
    )
    axes.set_aspect("equal")
    axes.invert_yaxis()
    axes.set(
        xticks=tuple(range(len(first_support))),
        xticklabels=tuple(str(value) for value in first_support),
        yticks=tuple(range(len(second_support))),
        yticklabels=tuple(str(value) for value in second_support),
    )
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(length=0)
    axis_labels(axes, joint.first.name, joint.second.name)
    for row, values in enumerate(matrix):
        for column, mass in enumerate(values):
            axes.text(
                column,
                row,
                f"{mass:.3f}",
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="semibold",
                color=palette.ink,
            )
    title_block(
        axes,
        f"Joint law of {joint.first.name} and {joint.second.name}",
        "Cell value = P(first = column, second = row); darker = more mass",
    )
    return axes


@themed
def plot_characteristic_function(
    variable: FiniteRandomVariable[Any, Any],
    *,
    t_min: float = -3.141592653589793,
    t_max: float = 3.141592653589793,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot real and imaginary parts of a finite characteristic function.

    Args:
        variable: Finite real-valued random variable.
        t_min: Finite lower frequency bound.
        t_max: Finite upper frequency bound, strictly greater than ``t_min``.
        points: Number of equally spaced frequencies, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If bounds, points, or mapped values have invalid types.
        ValueError: If bounds, points, or mapped values have invalid domains.
        OverflowError: If a characteristic-function phase cannot be represented.

    Equation:
        [EQ-GF-003].

    References:
        [REF-ROSS-2019, Section 6.5], [REF-MATPLOTLIB].

    """
    lower = require_finite_real(t_min, name="t_min")
    upper = require_finite_real(t_max, name="t_max")
    checked_points = require_non_negative_integer(points, name="points")
    if upper <= lower:
        raise ValueError("t_max must be greater than t_min")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    step = (upper - lower) / (checked_points - 1)
    frequencies = tuple(lower + index * step for index in range(checked_points))
    values = tuple(variable.characteristic_function(t) for t in frequencies)
    axes = new_axes(ax)
    palette = active_palette()
    real = tuple(value.real for value in values)
    imaginary = tuple(value.imag for value in values)
    # The horizontal hairline grid already marks phi = 0; an extra zero line
    # would strike through the direct labels at the right edge.
    line(axes, frequencies, real, color=palette.series[0])
    line(axes, frequencies, imaginary, color=palette.series[2])
    axes.set_ylim(-1.08, 1.08)
    axis_labels(axes, "Frequency t", "phi(t) component")
    direct_labels(
        axes,
        (
            (frequencies[-1], real[-1], "Re phi(t)", palette.series[0]),
            (frequencies[-1], imaginary[-1], "Im phi(t)", palette.series[2]),
        ),
    )
    largest_imaginary = max(abs(v) for v in imaginary)
    verdict = (
        "is symmetric: phi(t) is real"
        if largest_imaginary < 1e-12
        else f"is not symmetric: |Im phi(t)| reaches {largest_imaginary:.2f}"
    )
    title_block(
        axes,
        f"{variable.name} {verdict}",
        "phi(t) = E[exp(itX)]; |phi(t)| <= 1 with phi(0) = 1",
    )
    return axes


@themed
def plot_normal_pdf(
    distribution: Normal,
    *,
    standard_deviations: float = 4.0,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot a normal density and shade the central one-sigma mass.

    Args:
        distribution: Validated normal model.
        standard_deviations: Positive half-width in units of ``sigma``.
        points: Number of equally spaced evaluation points, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        ValueError: If ``standard_deviations <= 0`` or ``points < 2``.

    Equation:
        [EQ-DIST-NORM-001], [EQ-DIST-NORM-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3], [REF-MATPLOTLIB].

    """
    if standard_deviations <= 0.0:
        raise ValueError("standard_deviations must be positive")
    if points < 2:
        raise ValueError("points must be at least two")
    axes = new_axes(ax)
    palette = active_palette()
    mu, sigma = distribution.mu, distribution.sigma
    lower = mu - standard_deviations * sigma
    step = 2.0 * standard_deviations * sigma / (points - 1)
    xs = tuple(lower + index * step for index in range(points))
    ys = tuple(distribution.pdf(x) for x in xs)
    inner = tuple((x, y) for x, y in zip(xs, ys, strict=True) if abs(x - mu) <= sigma)
    axes.fill_between(
        tuple(x for x, _ in inner),
        0.0,
        tuple(y for _, y in inner),
        color=palette.observed,
        alpha=BAND_ALPHA + 0.06,
        linewidth=0.0,
    )
    line(axes, xs, ys, color=palette.observed)
    axes.set_ylim(0.0, max(ys) * 1.15)
    reference_line(axes, x=mu, label=f"mean = {mu:g}")
    mass = distribution.cdf(mu + sigma) - distribution.cdf(mu - sigma)
    note(
        axes,
        f"{mass:.1%} of the mass lies within one sigma",
        xy=(mu + 0.5 * sigma, distribution.pdf(mu + 0.5 * sigma) * 0.5),
        offset=(46.0, 30.0),
    )
    axis_labels(axes, "x", "Density f(x)")
    title_block(
        axes,
        f"Normal density with mean {mu:g} and sigma {sigma:g}",
        "Shaded: the interval mean +/- one standard deviation",
    )
    return axes


@themed
def plot_empirical_cdf(values: Iterable[float], *, ax: Any | None = None) -> Any:
    """Plot the right-continuous empirical CDF of a finite sample.

    Args:
        values: Non-empty finite real observations.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the sample is empty or contains a non-finite value.

    Equation:
        [EQ-STAT-005].

    References:
        [REF-WASSERMAN-2004, Section 7.3], [REF-MATPLOTLIB].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    axes = new_axes(ax)
    palette = active_palette()
    unique = tuple(dict.fromkeys(sample))
    heights = tuple(empirical_cdf(sample, x) for x in unique)
    span = (unique[-1] - unique[0]) or 1.0
    xs = (unique[0] - 0.08 * span, *unique, unique[-1] + 0.08 * span)
    ys = (0.0, *heights, 1.0)
    axes.step(xs, ys, where="post", color=palette.observed, linewidth=2.0)
    points(axes, unique, heights, color=palette.observed, size=28.0)
    axes.set_ylim(0.0, 1.06)
    axis_labels(axes, "x", "F_n(x)")
    title_block(
        axes,
        f"Empirical CDF of {len(sample)} observations",
        "Right-continuous steps of height (number of ties)/n at each observation",
    )
    return axes


@themed
def plot_fitted_cdf(
    values: Iterable[float],
    distribution: CumulativeDistribution,
    *,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Compare an empirical CDF with a fitted or specified model CDF.

    The largest vertical gap (the Kolmogorov-Smirnov distance) is annotated.
    The plot is diagnostic rather than proof of fit: parameter estimation from
    the same observations changes the calibration of formal tests.

    Args:
        values: Non-empty finite real observations.
        distribution: Object exposing a cumulative distribution function.
        points: Number of model-curve evaluation points, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes containing the empirical step function and model CDF.

    Raises:
        TypeError: If observations, distribution, or points have invalid types.
        ValueError: If observations are invalid or fewer than two points are used.

    Equation:
        [EQ-STAT-005], [EQ-INF-GOF-KS-001].

    References:
        [REF-NIST-GOF], [REF-MATPLOTLIB].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    checked_points = require_non_negative_integer(points, name="points")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    span = sample[-1] - sample[0]
    padding = 0.05 * span if span > 0.0 else max(1.0, abs(sample[0]) * 0.05)
    lower = sample[0] - padding
    upper = sample[-1] + padding
    step = (upper - lower) / (checked_points - 1)
    xs = tuple(lower + index * step for index in range(checked_points))
    unique = tuple(dict.fromkeys(sample))
    empirical = tuple(empirical_cdf(sample, value) for value in unique)
    n = len(sample)
    gap, gap_x, gap_y = 0.0, unique[0], 0.0
    for index, value in enumerate(sample):
        model_value = distribution.cdf(value)
        for candidate in ((index + 1) / n, index / n):
            if abs(candidate - model_value) > gap:
                gap, gap_x, gap_y = abs(candidate - model_value), value, model_value
    axes = new_axes(ax)
    palette = active_palette()
    axes.step(
        (lower, *unique, upper),
        (0.0, *empirical, 1.0),
        where="post",
        color=palette.observed,
        linewidth=2.0,
        zorder=3,
    )
    line(
        axes,
        xs,
        tuple(distribution.cdf(x) for x in xs),
        color=palette.model,
        dashes=FITTED,
    )
    axes.set_ylim(0.0, 1.06)
    axes.set_xlim(lower, upper)
    note(axes, f"largest gap D = {gap:.3f}", xy=(gap_x, gap_y), offset=(30.0, -28.0))
    axis_labels(axes, "x", "Cumulative probability")
    direct_labels(
        axes,
        (
            (upper, 1.0, f"empirical (n={n})", palette.observed),
            (
                upper,
                distribution.cdf(upper),
                f"{type(distribution).__name__} fit",
                palette.model,
            ),
        ),
    )
    title_block(
        axes,
        f"The {type(distribution).__name__} fit stays within D = {gap:.3f} of the data",
        "Solid steps: observed ECDF. Dashed: fitted model CDF",
    )
    return axes


@themed
def plot_refinement_convergence(
    coarse: float,
    medium: float,
    fine: float,
    *,
    refinement_ratio: float = 2.0,
    ax: Any | None = None,
) -> Any:
    """Visualize three-level convergence against Richardson extrapolation.

    Args:
        coarse: Quantity of interest on the coarsest resolution.
        medium: Quantity after one uniform refinement.
        fine: Quantity after a second uniform refinement.
        refinement_ratio: Common resolution ratio greater than one.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Log-log error axes annotated with the observed convergence order.

    Raises:
        TypeError: If any refinement argument is not real.
        ValueError: If refinement assumptions needed by the estimators fail.
        OverflowError: If extrapolation is not representable.

    Equation:
        [EQ-VV-001], [EQ-VV-002].

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-MATPLOTLIB].

    """
    order = observed_order_of_accuracy(
        coarse, medium, fine, refinement_ratio=refinement_ratio
    )
    extrapolated = richardson_extrapolation(
        medium, fine, refinement_ratio=refinement_ratio, order=order
    )
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    steps = (1.0, 1.0 / ratio, 1.0 / (ratio * ratio))
    errors = tuple(abs(value - extrapolated) for value in (coarse, medium, fine))
    axes = new_axes(ax)
    palette = active_palette()
    guide_x = (steps[0] * 1.15, steps[-1] / 1.15)
    guide_y = tuple(errors[0] * (x / steps[0]) ** order for x in guide_x)
    line(axes, guide_x, guide_y, color=palette.model, dashes=FITTED, width=1.4)
    line(axes, steps, errors, color=palette.observed, width=1.6)
    points(axes, steps, errors, color=palette.observed, size=40.0)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks(steps, ("h", f"h/{ratio:g}", f"h/{ratio * ratio:g}"))
    axes.xaxis.set_minor_formatter(_null_formatter())
    axes.invert_xaxis()
    axes.grid(axis="y", which="major", color=palette.rule, linewidth=0.7)
    axis_labels(
        axes, "Relative step size h (finer to the right)", "|QoI(h) - extrapolated|"
    )
    direct_labels(
        axes,
        ((guide_x[-1], guide_y[-1], f"slope p = {order:.2f}", palette.model),),
    )
    title_block(
        axes,
        f"Error falls with observed order p = {order:.2f}",
        f"Errors relative to the Richardson value {extrapolated:.4g}",
    )
    return axes


@themed
def plot_monte_carlo_convergence(
    estimates: Iterable[MonteCarloEstimate],
    *,
    reference: float | None = None,
    reference_label: str = "reference",
    ax: Any | None = None,
) -> Any:
    """Plot nested Monte Carlo estimates and their reported confidence band.

    Args:
        estimates: Non-empty sequence with strictly increasing sample sizes.
        reference: Optional finite analytic/reference value.
        reference_label: Label written next to the reference line.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes with log-sample-size estimate path and confidence band.

    Raises:
        TypeError: If an item is not a Monte Carlo estimate.
        ValueError: If input is empty, sizes do not increase, or reference is
            non-finite.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-SE-001], [EQ-MC-ROOTN-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-MATPLOTLIB].

    """
    trace = tuple(estimates)
    if not trace:
        raise ValueError("estimates must not be empty")
    for index, estimate in enumerate(trace):
        if not isinstance(estimate, MonteCarloEstimate):
            raise TypeError(f"estimates[{index}] must be a MonteCarloEstimate")
        if index and estimate.sample_size <= trace[index - 1].sample_size:
            raise ValueError("estimate sample sizes must be strictly increasing")
    checked_reference = (
        None if reference is None else require_finite_real(reference, name="reference")
    )
    sizes = tuple(estimate.sample_size for estimate in trace)
    centers = tuple(estimate.estimate for estimate in trace)
    lowers = tuple(estimate.confidence_interval.lower for estimate in trace)
    uppers = tuple(estimate.confidence_interval.upper for estimate in trace)
    axes = new_axes(ax)
    palette = active_palette()
    band(axes, sizes, lowers, uppers, color=palette.observed)
    line(axes, sizes, centers, color=palette.observed)
    points(axes, sizes, centers, color=palette.observed)
    axes.set_xscale("log")
    if checked_reference is not None:
        reference_line(axes, y=checked_reference, label=reference_label)
    confidence = trace[-1].confidence_interval.confidence
    axis_labels(axes, "Sample size N", "Estimate")
    width_ratio = (uppers[0] - lowers[0]) / max(uppers[-1] - lowers[-1], 1e-300)
    title_block(
        axes,
        f"The {confidence:.0%} interval narrows {width_ratio:.0f}x as N grows "
        f"{sizes[-1] // sizes[0]}x",
        "Line: running Monte Carlo estimate. Band: its reported confidence interval",
    )
    return axes


@themed
def plot_efficiency_comparison(
    comparisons: Iterable[EfficiencyComparison],
    *,
    ax: Any | None = None,
) -> Any:
    """Plot cost-normalized efficiency gains against a common baseline.

    Strategies are sorted by gain and drawn as dots on a logarithmic axis;
    the accent reference marks parity with the baseline.

    Args:
        comparisons: Non-empty comparisons sharing the same baseline label.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes showing variance-cost gain per strategy. Infinite gains are
        rejected because no finite logarithmic position can represent a
        zero-variance candidate.

    Raises:
        TypeError: If an item is not an efficiency comparison.
        ValueError: If input is empty, baselines differ, or a gain is non-finite.

    Equation:
        [EQ-MC-EFF-001].

    References:
        [REF-OWEN-2013, Chapter 2], [REF-MATPLOTLIB].

    """
    records = tuple(comparisons)
    if not records:
        raise ValueError("comparisons must not be empty")
    for index, comparison in enumerate(records):
        if not isinstance(comparison, EfficiencyComparison):
            raise TypeError(f"comparisons[{index}] must be an EfficiencyComparison")
        if comparison.baseline_label != records[0].baseline_label:
            raise ValueError("all comparisons must share one baseline label")
        if not math.isfinite(comparison.efficiency_gain):
            raise ValueError("efficiency gains must be finite for plotting")
    ordered = sorted(records, key=lambda record: record.efficiency_gain)
    labels = tuple(record.candidate_label for record in ordered)
    gains = tuple(record.efficiency_gain for record in ordered)
    axes = new_axes(ax, figsize=(7.2, 0.55 * len(labels) + 1.9))
    palette = active_palette()
    positions = tuple(range(len(labels)))
    colours = tuple(
        palette.observed if gain >= 1.0 else palette.muted for gain in gains
    )
    axes.hlines(positions, 1.0, gains, color=colours, linewidth=1.6, zorder=2)
    axes.scatter(
        gains,
        positions,
        s=46.0,
        color=colours,
        edgecolors=palette.canvas,
        linewidths=1.2,
        zorder=4,
    )
    for position, gain in zip(positions, gains, strict=True):
        axes.annotate(
            f"x{gain:.3g}",
            (gain, position),
            xytext=(8.0 if gain >= 1.0 else -8.0, 0.0),
            textcoords="offset points",
            ha="left" if gain >= 1.0 else "right",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(_multiplier_formatter())
    axes.xaxis.set_minor_formatter(_null_formatter())
    low, high = min((*gains, 1.0)), max((*gains, 1.0))
    axes.set_xlim(low / 1.8, high * 2.2)
    axes.set_ylim(-0.7, len(labels) - 0.3)
    axes.set(yticks=positions, yticklabels=labels)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", which="major", visible=True, color=palette.rule, linewidth=0.7)
    reference_line(axes, x=1.0, label=f"{records[0].baseline_label} parity")
    axis_labels(axes, "Variance-cost efficiency gain (log scale)")
    best = ordered[-1]
    title_block(
        axes,
        f"{best.candidate_label} is {best.efficiency_gain:.3g}x as efficient as "
        f"{best.baseline_label}",
        "Gain = (baseline variance x cost) / (candidate variance x cost)",
    )
    return axes
