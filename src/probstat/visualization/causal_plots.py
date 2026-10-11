"""Plots for causal effect estimation and sensitivity analysis.

Effect estimates from several estimators are compared as a dot-and-whisker
forest against the true (or null) effect, covariate balance before and after
weighting as a dumbbell "Love plot", and robustness to unmeasured confounding
as the E-value contour of confounder strengths that would explain an
association away.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.causal import BalanceTable, EValue

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
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
from .uq import plot_method_comparison


@themed
def plot_effect_estimates(
    estimates: Mapping[str, tuple[float, float | None, float | None]],
    *,
    truth: float | None = None,
    truth_label: str = "true effect",
    quantity_label: str = "Average treatment effect",
    interval_label: str = "95% interval",
    ax: Any | None = None,
) -> Any:
    """Compare effect estimates from several estimators as a forest plot.

    The title is computed from the entries: with a known ``truth`` it counts
    the intervals that cover it and names the estimate farthest from it.

    Args:
        estimates: Label to ``(estimate, lower, upper)``; bounds may be
            ``None`` for estimators without an interval.
        truth: Optional true effect drawn as the accent reference line.
        truth_label: Label of the reference line.
        quantity_label: Value-axis title.
        interval_label: Name of the whiskers in the subtitle.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``estimates`` is empty.

    Equation:
        [EQ-CAUSAL-REG-001], [EQ-CAUSAL-IPW-001], [EQ-CAUSAL-AIPW-001].

    References:
        [REF-HERNAN-ROBINS-2020, Chapters 12 and 13], [REF-MATPLOTLIB].

    """
    if not estimates:
        raise ValueError("estimates must not be empty")
    entries = [(label, *values) for label, values in estimates.items()]
    if truth is not None:
        bounds = [
            (lower, upper)
            for _, _, lower, upper in entries
            if lower is not None and upper is not None
        ]
        covered = sum(lower <= truth <= upper for lower, upper in bounds)
        worst = max(entries, key=lambda e: abs(e[1] - truth))
        title = (
            f"{covered} of {len(bounds)} intervals cover the true effect "
            f"{truth:g}; {worst[0]} misses it by {abs(worst[1] - truth):.2f}"
        )
    else:
        low = min(entries, key=lambda e: e[1])
        high = max(entries, key=lambda e: e[1])
        title = (
            f"Estimates range from {low[1]:.3g} ({low[0]}) to "
            f"{high[1]:.3g} ({high[0]})"
        )
    missing = any(e[2] is None or e[3] is None for e in entries)
    return plot_method_comparison(
        entries,
        reference=truth,
        reference_label=truth_label,
        quantity_label=quantity_label,
        title=title,
        subtitle=f"Dot: estimate. Whisker: {interval_label}"
        + ("; no whisker: no interval" if missing else ""),
        ax=ax,
    )


@themed
def plot_covariate_balance(
    before: BalanceTable,
    after: BalanceTable,
    *,
    threshold: float = 0.1,
    after_label: str = "weighted",
    ax: Any | None = None,
) -> Any:
    """Plot absolute standardized mean differences before and after weighting.

    Each covariate is a dumbbell from its raw imbalance (muted) to its
    weighted imbalance (observed hue); the conventional ``|SMD| = 0.1``
    threshold is the reference line.

    Args:
        before: Unweighted balance table.
        after: Weighted balance table over the same covariates.
        threshold: Reference imbalance.
        after_label: Name of the adjusted sample.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the tables cover different covariates.

    Equation:
        [EQ-CAUSAL-SMD-001].

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 14], [REF-MATPLOTLIB].

    """
    names = [row.name for row in before.covariates]
    if names != [row.name for row in after.covariates]:
        raise ValueError("before and after must cover the same covariates")
    raw = [abs(row.standardized_difference) for row in before.covariates]
    adjusted = [abs(row.standardized_difference) for row in after.covariates]
    order = sorted(range(len(names)), key=lambda i: raw[i])
    axes = new_axes(ax, figsize=(7.2, 0.62 * len(names) + 2.0))
    palette = active_palette()
    for position, index in enumerate(order):
        axes.hlines(
            position,
            min(raw[index], adjusted[index]),
            max(raw[index], adjusted[index]),
            color=palette.rule,
            linewidth=5.0,
            zorder=1,
        )
        points(axes, [raw[index]], [position], color=palette.muted, size=52.0)
        points(axes, [adjusted[index]], [position], color=palette.observed, size=52.0)
        axes.annotate(
            f"{raw[index]:.2f}",
            (raw[index], position),
            xytext=(8.0, 0.0),
            textcoords="offset points",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    top = len(order) - 1
    top_raw, top_adjusted = raw[order[top]], adjusted[order[top]]
    for value, other, text, colour in (
        (top_raw, top_adjusted, "raw", palette.muted),
        (top_adjusted, top_raw, after_label, palette.observed),
    ):
        smaller = value <= other
        axes.annotate(
            text,
            (value, top),
            xytext=(0.0 if smaller else 5.0, 12.0),
            textcoords="offset points",
            ha="left" if smaller else "right",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=colour,
        )
    axes.set(yticks=tuple(range(len(order))), yticklabels=[names[i] for i in order])
    axes.set_ylim(-0.6, len(order) - 0.1)
    axes.set_xlim(0.0, max(max(raw), threshold) * 1.2)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    reference_line(axes, x=threshold, label=f"|SMD| = {threshold:g}")
    axis_labels(axes, "Absolute standardized mean difference")
    worst_raw, worst_adjusted = max(raw), max(adjusted)
    if worst_adjusted < threshold:
        title = (
            f"Weighting cuts the largest imbalance from {worst_raw:.2f} to "
            f"{worst_adjusted:.2f} SD, below {threshold:g}"
        )
    else:
        flagged = sum(value >= threshold for value in adjusted)
        title = (
            f"{flagged} of {len(names)} covariates stay at or above "
            f"{threshold:g} SD after weighting"
        )
    title_block(
        axes,
        title,
        f"Grey: raw sample. Colour: {after_label} sample. Same pooled SD in both",
    )
    return axes


def explain_away_curve(
    risk_ratio: float, strengths: Sequence[float]
) -> tuple[float, ...]:
    """Return confounder-outcome strengths that fully explain a risk ratio.

    Solving the bounding factor ``B = RR_EU RR_UD/(RR_EU + RR_UD - 1) = RR``
    for ``RR_UD`` gives ``RR_UD = RR (RR_EU - 1)/(RR_EU - RR)`` for
    ``RR_EU > RR``; protective ratios are inverted first.

    Args:
        risk_ratio: Observed risk ratio (strictly positive).
        strengths: Treatment-confounder strengths ``RR_EU``.

    Returns:
        ``RR_UD`` per strength (``inf`` where ``RR_EU <= RR``).

    Raises:
        ValueError: If ``risk_ratio`` is not positive.

    Equation:
        [EQ-CAUSAL-BIAS-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    if not risk_ratio > 0.0:
        raise ValueError("risk_ratio must be positive")
    rr = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return tuple(
        rr * (eu - 1.0) / (eu - rr) if eu > rr else math.inf for eu in strengths
    )


@themed
def plot_e_value_contour(
    evalue: EValue,
    *,
    limit_ratio: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot the confounder strengths that would explain an association away.

    The curve holds every pair ``(RR_EU, RR_UD)`` whose bounding factor equals
    the observed risk ratio; any confounder above it could reduce the ratio
    to one. Its point on the diagonal is the E-value. With the confidence
    limit closest to one, a second (muted) curve shows the strengths that
    would move the interval to include one.

    Args:
        evalue: Output of :func:`probstat.causal.e_value`.
        limit_ratio: Optional confidence limit closest to one (risk ratio).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the observed risk ratio is one (nothing to explain).

    Equation:
        [EQ-CAUSAL-EVALUE-001], [EQ-CAUSAL-BIAS-001].

    References:
        [REF-VANDERWEELE-DING-2017], [REF-MATPLOTLIB].

    """
    rr = evalue.risk_ratio if evalue.risk_ratio >= 1.0 else 1.0 / evalue.risk_ratio
    if rr <= 1.0:
        raise ValueError("a risk ratio of one needs no confounding to explain")
    axes = new_axes(ax, figsize=(7.2, 5.2))
    palette = active_palette()
    top = max(3.0 * evalue.point, rr + 2.0)
    count = 240
    strengths = [rr * (top / rr) ** (k / (count - 1)) for k in range(count)][1:]
    curve = explain_away_curve(rr, strengths)
    kept = [(eu, ud) for eu, ud in zip(strengths, curve, strict=True) if ud <= top]
    xs = [eu for eu, _ in kept]
    ys = [ud for _, ud in kept]
    axes.fill_between(
        xs, ys, [top] * len(xs), color=palette.observed, alpha=BAND_ALPHA, linewidth=0.0
    )
    line(axes, xs, ys, color=palette.observed)
    labels = [(xs[-1], ys[-1], f"explains away RR = {rr:.2f}", palette.observed)]
    if limit_ratio is not None and evalue.confidence_limit not in (None, 1.0):
        limit = limit_ratio if limit_ratio >= 1.0 else 1.0 / limit_ratio
        lower_curve = explain_away_curve(limit, strengths)
        pairs = [
            (eu, ud)
            for eu, ud in zip(strengths, lower_curve, strict=True)
            if eu > limit and ud <= top
        ]
        line(
            axes,
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            color=palette.muted,
            width=1.4,
        )
        labels.append(
            (
                pairs[-1][0],
                pairs[-1][1],
                f"moves CI to 1 (limit {limit:.2f})",
                palette.muted,
            )
        )
    points(axes, [evalue.point], [evalue.point], color=palette.observed, size=60.0)
    axes.annotate(
        f"E-value {evalue.point:.2f}",
        (evalue.point, evalue.point),
        xytext=(10.0, 10.0),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=palette.ink,
    )
    axes.plot(
        [1.0, top], [1.0, top], color=palette.rule, linewidth=0.9, zorder=1
    )
    axes.set_xlim(1.0, top)
    axes.set_ylim(1.0, top)
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    axis_labels(
        axes,
        "Treatment-confounder risk ratio RR_EU",
        "Confounder-outcome risk ratio RR_UD",
    )
    direct_labels(axes, labels)
    limit_text = (
        f"; {evalue.confidence_limit:.2f} to move the interval to 1"
        if evalue.confidence_limit is not None
        else ""
    )
    title_block(
        axes,
        f"Explaining away RR = {rr:.2f} needs a confounder with "
        f"RR >= {evalue.point:.2f} on both paths",
        f"Shaded: confounders strong enough to explain it away{limit_text}",
    )
    return axes
