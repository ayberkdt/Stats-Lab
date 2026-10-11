"""Plots for optimal and Bayesian experimental design.

An approximate design is shown as weight lollipops above its standardized
variance function, so the Kiefer-Wolfowitz equivalence bound ``p`` can be read
directly; a greedy Bayesian design is shown as the expected information gain
of every candidate at every step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.design import (
    ApproximateDesign,
    Basis,
    GreedyDesignResult,
    standardized_variances,
)

from .theme import (
    ANNOTATION_SIZE,
    active_palette,
    axis_labels,
    direct_labels,
    line,
    new_axes,
    points,
    pyplot,
    reference_line,
    style_axes,
    themed,
    title_block,
)


def _scalar_points(design: ApproximateDesign) -> list[float]:
    """Return the support of a one-factor design as floats.

    Args:
        design: Approximate design.

    Returns:
        Support coordinates.

    Raises:
        ValueError: If the design has more than one factor.

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

    """
    if design.dimension != 1:
        raise ValueError("only one-factor designs can be plotted")
    return [point[0] for point in design.points]


@themed
def plot_optimal_design(
    design: ApproximateDesign,
    basis: Basis,
    region: Sequence[float],
    *,
    comparison: ApproximateDesign | None = None,
    comparison_label: str = "uniform design",
    factor: str = "x",
) -> Any:
    """Plot design weights above the standardized variance function.

    The top panel shows the support weights as lollipops; the bottom panel
    shows ``d(x) = f(x)' M^-1 f(x)`` over the region with the equivalence
    bound ``p`` (the number of parameters) as a reference line. A design is
    D-optimal exactly when ``max d = p`` (Kiefer-Wolfowitz); otherwise
    ``exp(1 - max d / p)`` bounds its D-efficiency from below.

    Args:
        design: One-factor approximate design (condensed for readability).
        basis: Regression basis.
        region: Points at which ``d(x)`` is evaluated, in increasing order.
        comparison: Optional design drawn in muted ink for context.
        comparison_label: Direct label of the comparison design.
        factor: Name of the design factor.

    Returns:
        The figure with two stacked panels sharing the horizontal axis.

    Raises:
        ValueError: If a design has more than one factor, the region has
            fewer than two points, or an information matrix is singular.

    Equation:
        [EQ-OED-INFO-001], [EQ-OED-KW-001].

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007],
        [REF-MATPLOTLIB].

    """
    grid = [float(x) for x in region]
    if len(grid) < 2:
        raise ValueError("the region needs at least two points")
    support = _scalar_points(design)
    plt = pyplot()
    palette = active_palette()
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.2, 5.6),
        gridspec_kw={"height_ratios": (1.0, 1.6), "hspace": 0.12},
    )
    for axes in (top, bottom):
        style_axes(axes)
    if comparison is not None:
        others = _scalar_points(comparison)
        top.vlines(
            others, 0.0, comparison.weights, colors=palette.muted, linewidth=1.0
        )
    top.vlines(support, 0.0, design.weights, colors=palette.observed, linewidth=2.0)
    points(top, support, design.weights, color=palette.observed, size=46.0)
    for x, w in zip(support, design.weights, strict=True):
        top.annotate(
            f"{w:.3f}",
            (x, w),
            xytext=(0.0, 7.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
            annotation_clip=False,
        )
    top.set_ylim(0.0, max(design.weights) * 1.35)
    axis_labels(top, None, "Weight")
    variances = standardized_variances(design, basis, grid)
    parameters = len(basis((grid[0],)))
    labels = []
    if comparison is not None:
        context = standardized_variances(comparison, basis, grid)
        line(bottom, grid, context, color=palette.muted, width=1.4)
        labels.append((grid[-1], context[-1], comparison_label, palette.muted))
    line(bottom, grid, variances, color=palette.observed)
    on_support = standardized_variances(design, basis, [(x,) for x in support])
    points(bottom, support, on_support, color=palette.observed, size=34.0)
    reference_line(bottom, y=float(parameters))
    bottom.set_ylim(0.0, None)
    labels.append((grid[-1], variances[-1], "this design", palette.observed))
    labels.append(
        (grid[-1], float(parameters), f"bound p = {parameters}", palette.accent)
    )
    right = bottom.get_xlim()[1]
    direct_labels(bottom, [(right, y, text, colour) for _, y, text, colour in labels])
    axis_labels(bottom, factor, "Standardized variance d(x)")
    peak = max(variances)
    peak_at = grid[variances.index(peak)]
    if peak <= parameters * (1.0 + 1e-3):
        title = (
            f"Max d(x) = {peak:.3f} matches p = {parameters}: "
            f"the {len(support)}-point design is D-optimal"
        )
    else:
        bound = math.exp(1.0 - peak / parameters)
        title = (
            f"d(x) peaks at {peak:.2f} > p = {parameters} (x = {peak_at:.3g}); "
            f"D-efficiency is at least {bound:.0%}"
        )
    subtitle = "Top: design weights. Bottom: d(x) over the region; dots: support"
    if comparison is not None:
        subtitle += f"; grey: {comparison_label}"
    title_block(top, title, subtitle)
    return figure


@themed
def plot_greedy_design(
    result: GreedyDesignResult,
    candidates: Sequence[float],
    *,
    factor: str = "Condition",
    unit: str = "nats",
    ax: Any | None = None,
) -> Any:
    """Plot the information each candidate would add at each greedy step.

    Step ``k`` draws ``EIG(selected + [c]) - EIG(selected)`` over the
    candidates ``c``; its maximum (a dot) is the condition chosen at that
    step. Curves fall from step to step as the information already gathered
    grows.

    Args:
        result: Output of :func:`probstat.design.greedy_design`.
        candidates: Scalar value of each candidate condition, in the order
            passed to ``greedy_design``.
        factor: Name of the candidate variable.
        unit: Information unit of the scores.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``candidates`` does not match the scored candidates.

    Equation:
        [EQ-OED-GREEDY-001], [EQ-OED-EIG-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013],
        [REF-MATPLOTLIB].

    """
    xs = [float(c) for c in candidates]
    if not result.candidate_scores or len(result.candidate_scores[0]) != len(xs):
        raise ValueError("candidates must match the scored candidate list")
    axes = new_axes(ax)
    palette = active_palette()
    previous = result.baseline_information_gain
    labels = []
    for step, scores in enumerate(result.candidate_scores):
        colour = palette.series[step % len(palette.series)]
        gains = [score - previous for score in scores]
        line(axes, xs, gains, color=colour, width=1.8)
        chosen = result.indices[step]
        points(axes, [xs[chosen]], [gains[chosen]], color=colour, size=48.0)
        labels.append(
            (
                xs[-1],
                gains[-1],
                f"step {step + 1}: {factor} = {xs[chosen]:g}",
                colour,
            )
        )
        previous = result.information_gains[step]
    axes.set_ylim(0.0, None)
    axis_labels(axes, factor, f"Information added [{unit}]")
    direct_labels(axes, labels)
    picks = ", ".join(f"{xs[i]:g}" for i in result.indices)
    title_block(
        axes,
        f"Greedy design picks {factor} = {picks}; "
        f"EIG reaches {result.information_gains[-1]:.2f} {unit}",
        f"Each curve: EIG gain of one more run per candidate ({result.method}); "
        "dot: chosen",
    )
    return axes
