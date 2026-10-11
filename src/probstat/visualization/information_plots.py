"""Plots for information theory: estimator checks and entropy decompositions.

A sample-based estimator (Kozachenko-Leonenko, KSG) is checked against a
closed form across a parameter, with replicate spread as whiskers; a joint
PMF's entropies are shown as the information diagram drawn with bars, in which
``H(X, Y) = H(X | Y) + I(X; Y) + H(Y | X)``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from probstat.information import (
    InformationUnit,
    conditional_entropy,
    entropy,
    joint_entropy,
    marginals,
    mutual_information,
)

from .theme import (
    ANNOTATION_SIZE,
    REFERENCE,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    direct_labels,
    new_axes,
    themed,
    title_block,
)


@themed
def plot_estimator_check(
    parameters: Sequence[float],
    replicates: Sequence[Sequence[float]],
    exact: Callable[[float], float],
    *,
    estimator: str = "KSG",
    parameter_label: str = "Correlation rho",
    parameter_symbol: str = "rho",
    quantity_label: str = "Mutual information",
    unit: str = "nats",
    sample_size: int | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot replicate estimates against a closed form across a parameter.

    Each parameter value shows the replicate mean (dot) and +/- one replicate
    standard deviation (whisker); the closed form is the accent reference
    curve. The title reports the largest absolute gap between mean and exact
    value and where it occurs.

    Args:
        parameters: Parameter values in increasing order.
        replicates: Estimates per parameter value (at least two each).
        exact: Closed-form value as a function of the parameter; it is drawn
            on a fine grid and evaluated at ``parameters`` for the gaps.
        estimator: Estimator name.
        parameter_label: Horizontal-axis title.
        parameter_symbol: Short parameter name used in the title.
        quantity_label: Vertical-axis title (without unit).
        unit: Unit of the estimates.
        sample_size: Optional sample size per replicate, shown in the subtitle.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the sequences differ in length or a replicate set has
            fewer than two estimates.

    Equation:
        [EQ-INFO-KSG-001], [EQ-INFO-KLE-001].

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004], [REF-MATPLOTLIB].

    """
    if len(parameters) != len(replicates) or not parameters:
        raise ValueError("parameters and replicates must align and be non-empty")
    if any(len(group) < 2 for group in replicates):
        raise ValueError("every parameter value needs at least two replicates")
    axes = new_axes(ax)
    palette = active_palette()
    xs = [float(x) for x in parameters]
    means = [statistics.fmean(group) for group in replicates]
    spreads = [statistics.stdev(group) for group in replicates]
    exact_values = [exact(x) for x in xs]
    fine = [xs[0] + (xs[-1] - xs[0]) * k / 200 for k in range(201)]
    axes.plot(
        fine,
        [exact(x) for x in fine],
        color=palette.accent,
        linewidth=REFERENCE_WIDTH + 0.3,
        linestyle=REFERENCE,
        zorder=2,
    )
    axes.vlines(
        xs,
        [m - s for m, s in zip(means, spreads, strict=True)],
        [m + s for m, s in zip(means, spreads, strict=True)],
        color=palette.observed,
        linewidth=2.2,
        alpha=0.55,
        zorder=3,
    )
    axes.scatter(
        xs,
        means,
        s=46.0,
        color=palette.observed,
        edgecolors=palette.canvas,
        linewidths=1.2,
        zorder=4,
    )
    axis_labels(axes, parameter_label, quantity_label, y_unit=unit)
    direct_labels(
        axes,
        [
            (xs[-1], exact_values[-1], "closed form", palette.accent),
            (
                xs[-1],
                means[-1] - spreads[-1],
                f"{estimator} mean +/- 1 SD",
                palette.observed,
            ),
        ],
    )
    gaps = [abs(m - e) for m, e in zip(means, exact_values, strict=True)]
    worst = max(range(len(gaps)), key=lambda i: gaps[i])
    size_text = f" of n = {sample_size}" if sample_size is not None else ""
    title_block(
        axes,
        f"{estimator} stays within {gaps[worst]:.3f} {unit} of the closed form; "
        f"the largest gap is at {parameter_symbol} = {xs[worst]:g}",
        f"{len(replicates[0])} seeded replicates{size_text} per value; "
        f"typical SD {statistics.median(spreads):.3f} {unit}",
    )
    return axes


@themed
def plot_entropy_decomposition(
    joint: Sequence[Sequence[float]],
    *,
    names: tuple[str, str] = ("X", "Y"),
    unit: InformationUnit = "bits",
    ax: Any | None = None,
) -> Any:
    """Draw the information diagram of a joint PMF as aligned bars.

    Three rows share one scale: ``H(X) = H(X|Y) + I``, ``H(Y) = I + H(Y|X)``
    and ``H(X, Y) = H(X|Y) + I + H(Y|X)``. The mutual information segment uses
    the data hue; conditional entropies are muted.

    Args:
        joint: Joint PMF table with rows indexed by ``X``.
        names: Variable names.
        unit: ``"bits"`` or ``"nats"``.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the table is not a valid PMF.

    Equation:
        [EQ-INFO-H-001], [EQ-INFO-H-002], [EQ-INFO-MI-001].

    References:
        [REF-COVER-THOMAS-2006, Chapter 2], [REF-MATPLOTLIB].

    """
    x_name, y_name = names
    px, py = marginals(joint)
    hx = entropy(px, unit=unit)
    hy = entropy(py, unit=unit)
    hxy = joint_entropy(joint, unit=unit)
    h_y_given_x = conditional_entropy(joint, unit=unit)
    info = mutual_information(joint, unit=unit)
    h_x_given_y = hx - info
    axes = new_axes(ax, figsize=(7.2, 3.6))
    palette = active_palette()
    rows: tuple[tuple[int, str, tuple[tuple[float, float, str], ...]], ...] = (
        (2, f"H({x_name})", ((0.0, h_x_given_y, "cond_x"), (h_x_given_y, info, "mi"))),
        (1, f"H({y_name})", ((h_x_given_y, info, "mi"), (hx, h_y_given_x, "cond_y"))),
        (
            0,
            f"H({x_name}, {y_name})",
            (
                (0.0, h_x_given_y, "cond_x"),
                (h_x_given_y, info, "mi"),
                (hx, h_y_given_x, "cond_y"),
            ),
        ),
    )
    text = {
        "cond_x": f"H({x_name}|{y_name}) {h_x_given_y:.3f}",
        "mi": f"I {info:.3f}",
        "cond_y": f"H({y_name}|{x_name}) {h_y_given_x:.3f}",
    }
    for position, _, segments in rows:
        for start, width, kind in segments:
            colour = palette.observed if kind == "mi" else palette.muted
            alpha = 1.0 if kind == "mi" else (0.35 if kind == "cond_x" else 0.55)
            axes.barh(
                position,
                width,
                left=start,
                height=0.62,
                color=colour,
                alpha=alpha,
                edgecolor=palette.canvas,
                linewidth=1.5,
            )
            if position == 0:
                axes.annotate(
                    text[kind],
                    (start + 0.5 * width, position),
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_SIZE,
                    fontweight="bold",
                    color=palette.canvas if kind == "mi" else palette.ink,
                )
    for (position, _, segments), total in zip(rows, (hx, hy, hxy), strict=True):
        end = segments[-1][0] + segments[-1][1]
        axes.annotate(
            f"= {total:.3f}",
            (end, position),
            xytext=(6.0, 0.0),
            textcoords="offset points",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    axes.set(yticks=(2, 1, 0), yticklabels=[row[1] for row in rows])
    axes.set_xlim(0.0, hxy * 1.12)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    axis_labels(axes, "Entropy", x_unit=unit)
    share = info / hx if hx > 0.0 else math.nan
    title_block(
        axes,
        f"Knowing {y_name} removes {share:.0%} of the uncertainty about {x_name} "
        f"(I = {info:.3f} {unit})",
        "Bars share one scale; colour marks the shared information "
        f"I({x_name}; {y_name})",
    )
    return axes
