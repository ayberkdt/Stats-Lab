"""Figures for the tail, information, and design stages of a ``UQStudy``.

The functions take the stage result objects of :mod:`probstat.workflow`
(``TailAnalysis``, ``InformationSensitivity``, ``DesignRecommendation``) and
never rerun the model; a POT tail curve is evaluated from the stored fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from probstat.risk.extremes import ReturnLevel
from probstat.workflow import DesignRecommendation, InformationSensitivity, TailAnalysis

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    FITTED,
    PROJECTED,
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


def _oriented(level: ReturnLevel, sign: float) -> tuple[float, float, float]:
    """Return ``(level, lower, upper)`` of a return level in QoI units.

    Args:
        level: Return level of the oriented sample ``sign * Y``.
        sign: ``+1`` for the upper tail, ``-1`` for the lower tail.

    Returns:
        Level and ordered interval limits.

    References:
        [REF-COLES-2001, Section 4.3]. No equation is evaluated.

    """
    if sign > 0.0:
        return level.level, level.lower, level.upper
    return -level.level, -level.upper, -level.lower


def _log_grid(start: float, stop: float, count: int) -> list[float]:
    """Return ``count`` logarithmically spaced values from ``start`` to ``stop``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    ratio = math.log(stop / start)
    return [start * math.exp(ratio * i / (count - 1)) for i in range(count)]


@themed
def plot_tail_analysis(
    tail: TailAnalysis,
    *,
    empirical_levels: Mapping[float, float] | None = None,
    confidence: float = 0.95,
    name: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot the fitted POT tail as exceedance probability against the QoI.

    The vertical axis is the per-run probability ``1/T`` of exceeding (upper
    tail) or undercutting (lower tail) a value, on a log scale. The fitted
    tail ``zeta_u S_GPD`` is evaluated from the stored fit: ``FITTED`` down to
    ``1/n`` (inside the sample) and ``PROJECTED`` below it, with the
    delta-method return-level interval as a horizontal band. Empirical
    quantiles (from the stage summary) are observed-hue dots; a level under
    study is the accent line with its POT probability (model hue) and its
    empirical Wilson interval (observed hue).

    Args:
        tail: Result of ``UQStudy.extremes``.
        empirical_levels: Optional ``{return period: empirical quantile}``.
        confidence: Interval level of the band.
        name: Horizontal-axis title (default: the QoI name).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the confidence is invalid.

    Equation:
        [EQ-EVA-POT-001], [EQ-EVA-POT-002], [EQ-EVA-RATE-001],
        [EQ-MC-PROB-001].

    References:
        [REF-COLES-2001, Section 4.3], [REF-WILSON-1927], [REF-MATPLOTLIB].

    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    fit = tail.fit
    sign = 1.0 if tail.tail == "upper" else -1.0
    relation = ">" if sign > 0.0 else "<"
    count = tail.sample_size
    zeta = fit.rate.probability
    periods = [item.return_period for item in tail.return_levels]
    longest = max([10.0 * count, *periods]) * 2.0
    if tail.level is not None and tail.pot_probability > 0.0:
        longest = max(longest, 2.0 / tail.pot_probability)
    bottom = 1.0 / longest
    grid = _log_grid(1.02 / zeta, longest * 3.0, 90)
    curve = [_oriented(fit.return_level(t, confidence=confidence), sign) for t in grid]
    probabilities = [1.0 / t for t in grid]
    axes = new_axes(ax)
    palette = active_palette()
    finite = [
        (p, low, high)
        for p, (_, low, high) in zip(probabilities, curve, strict=True)
        if math.isfinite(low) and math.isfinite(high)
    ]
    if len(finite) >= 2:
        axes.fill_betweenx(
            [p for p, _, _ in finite],
            [low for _, low, _ in finite],
            [high for _, _, high in finite],
            color=palette.model,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
    levels = [value for value, _, _ in curve]
    split = sum(p >= 1.0 / count for p in probabilities)
    if split:
        line(
            axes, levels[:split], probabilities[:split], color=palette.model,
            dashes=FITTED,
        )  # fmt: skip
    if split < len(levels):
        start = max(split - 1, 0)
        line(
            axes,
            levels[start:],
            probabilities[start:],
            color=palette.model,
            dashes=PROJECTED,
        )
    if empirical_levels:
        shown = {t: v for t, v in empirical_levels.items() if math.isfinite(v)}
        if shown:
            points(
                axes,
                list(shown.values()),
                [1.0 / t for t in shown],
                color=palette.observed,
                size=40.0,
            )
    axes.set_yscale("log")
    reference_line(axes, x=tail.threshold, label="threshold", color=palette.muted)
    reference_line(
        axes, y=1.0 / count, label=f"1 / {count:,} runs", color=palette.muted
    )
    anchor = next(
        (i for i, p in enumerate(probabilities) if p < 4.0 * bottom),
        len(probabilities) - 1,
    )
    labels = [(levels[anchor], probabilities[anchor], "fitted GPD tail", palette.model)]
    if tail.level is not None:
        reference_line(axes, x=tail.level, label=f"level {tail.level:g}")
        if tail.pot_probability > 0.0:
            points(
                axes, [tail.level], [tail.pot_probability], color=palette.model,
                size=46.0, zorder=5.0,
            )  # fmt: skip
        if tail.empirical is not None and tail.empirical.probability > 0.0:
            interval = tail.empirical.confidence_interval
            axes.vlines(
                tail.level,
                max(interval.lower, bottom),
                interval.upper,
                color=palette.observed,
                linewidth=2.4,
                zorder=4,
            )
            points(
                axes, [tail.level], [tail.empirical.probability],
                color=palette.observed, size=40.0, zorder=5.0,
            )  # fmt: skip
    axes.set_ylim(bottom, min(1.0, zeta * 1.6))
    visible = [
        value
        for p, entry in zip(probabilities, curve, strict=True)
        if p >= bottom
        for value in entry
        if math.isfinite(value)
    ]
    visible.append(tail.threshold)
    if tail.level is not None:
        visible.append(tail.level)
    if empirical_levels:
        visible.extend(v for v in empirical_levels.values() if math.isfinite(v))
    left, right = min(visible), max(visible)
    pad = 0.04 * (right - left)
    axes.set_xlim(left - pad, right + pad)
    axis_labels(
        axes,
        name or tail.quantity,
        f"P(Y {relation} x) per run [log scale]",
    )
    direct_labels(axes, labels)
    if tail.level is not None and tail.empirical is not None:
        interval = tail.empirical.confidence_interval
        verdict = (
            "inside" if interval.lower <= tail.pot_probability <= interval.upper
            else "outside"
        )
        title = (
            f"POT puts P({tail.quantity} {relation} {tail.level:g}) at "
            f"{tail.pot_probability:.2e}, {verdict} the empirical "
            f"{interval.confidence:.0%} interval"
        )
    else:
        eligible = [
            item for item in tail.return_levels if item.return_period <= 10 * count
        ]
        item = eligible[-1] if eligible else tail.return_levels[-1]
        value, low, high = _oriented(item, 1.0)
        title = (
            f"The {item.return_period:g}-run {tail.tail} level of {tail.quantity} "
            f"is {value:.3g} [{low:.3g}, {high:.3g}]"
        )
    title_block(
        axes,
        title,
        f"{fit.rate.exceedances} exceedances of {count:,} runs · dashed: fit · "
        f"dotted: beyond the sample · band: {confidence:.0%} delta interval · "
        "dots: empirical",
    )
    return axes


def _lollipop_rows(
    axes: Any, values: list[float], colour: str, *, labels: bool = True
) -> None:
    """Draw horizontal lollipops at rows ``0..n-1``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    rows = list(range(len(values)))
    axes.hlines(rows, 0.0, values, color=colour, linewidth=2.0, zorder=2)
    points(axes, values, rows, color=colour, size=54.0)
    if labels:
        for row, value in zip(rows, values, strict=True):
            axes.annotate(
                f"{value:.2f}",
                (value, row),
                xytext=(8.0, 0.0),
                textcoords="offset points",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color=palette.ink,
            )


@themed
def plot_information_sensitivity(
    result: InformationSensitivity, *, sobol: Any | None = None
) -> Any:
    """Plot the mutual information of each input with the QoI.

    Inputs are rows sorted by mutual information (largest on top), drawn as
    observed-hue lollipops. With Sobol indices of the same QoI a second
    panel on the same rows shows first-order (observed hue, the variance
    analogue of a one-input dependence measure) and total-effect (model hue)
    indices, so moment-independent and variance-based rankings can be read
    side by side without a dual axis.

    Args:
        result: Result of ``UQStudy.information_sensitivity``.
        sobol: Optional Sobol result with ``input_names``, ``first_order``
            and ``total_effect`` (index records with ``estimate``).

    Returns:
        The figure (one or two panels).

    Raises:
        ValueError: If the Sobol inputs differ from the information inputs.

    Equation:
        [EQ-INFO-MI-001], [EQ-INFO-KSG-001], [EQ-SENS-FIRST-001],
        [EQ-SENS-TOTAL-001].

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004], [REF-SALTELLI-2010],
        [REF-MATPLOTLIB].

    """
    names = list(result.input_names)
    mi = dict(zip(names, result.mutual_information, strict=True))
    order = sorted(names, key=lambda n: (mi[n], -names.index(n)))
    plt = pyplot()
    palette = active_palette()
    height = 0.62 * len(names) + 2.3
    if sobol is not None:
        if set(sobol.input_names) != set(names):
            raise ValueError("Sobol inputs must match the information inputs")
        figure, (left, right) = plt.subplots(
            1, 2, sharey=True, figsize=(9.6, height),
            gridspec_kw={"width_ratios": (1.0, 1.0), "wspace": 0.12},
        )  # fmt: skip
        style_axes(right)
    else:
        figure, left = plt.subplots(figsize=(7.2, height))
        right = None
    style_axes(left)
    values = [mi[n] for n in order]
    _lollipop_rows(left, values, palette.observed)
    left.set_yticks(range(len(order)))
    left.set_yticklabels(order)
    left.set_ylim(-0.6, len(order) - 0.4)
    left.set_xlim(0.0, max(max(values), 1e-12) * 1.3)
    left.grid(axis="y", visible=False)
    left.grid(axis="x", color=palette.rule, linewidth=0.7)
    axis_labels(left, f"Mutual information I(X; {result.quantity}) [{result.unit}]")
    top, runner = order[-1], order[-2] if len(order) > 1 else None
    title = f"Input {top} shares the most information with {result.quantity}: " + (
        f"{mi[top]:.2f} {result.unit}"
    )
    if runner is not None and mi[runner] > 0.0:
        title += f", {mi[top] / mi[runner]:.1f}x the next input"
    subtitle = (
        f"KSG estimator, k = {result.k}, {result.sample_size:,} runs; values "
        "are not variance shares"
    )
    if right is not None and sobol is not None:
        first = {
            n: item.estimate
            for n, item in zip(sobol.input_names, sobol.first_order, strict=True)
        }
        total = {
            n: item.estimate
            for n, item in zip(sobol.input_names, sobol.total_effect, strict=True)
        }
        for row, name in enumerate(order):
            right.hlines(
                row, min(first[name], total[name]), max(first[name], total[name]),
                color=palette.rule, linewidth=5.0, zorder=1,
            )  # fmt: skip
        points(right, [first[n] for n in order], list(range(len(order))),
               color=palette.observed, size=54.0)  # fmt: skip
        points(right, [total[n] for n in order], list(range(len(order))),
               color=palette.model, size=54.0)  # fmt: skip
        right.set_xlim(min(0.0, *first.values()) - 0.02, 1.0)
        right.grid(axis="y", visible=False)
        right.grid(axis="x", color=palette.rule, linewidth=0.7)
        right.tick_params(axis="y", length=0, labelleft=False)
        axis_labels(right, "Sobol index (share of variance)")
        lead = order[-1]
        direct_labels(
            right,
            [
                (max(first[lead], total[lead]), len(order) - 1 + 0.28,
                 "total S_T", palette.model),
                (max(first[lead], total[lead]), len(order) - 1 - 0.28,
                 "first-order S", palette.observed),
            ],
            gap_points=10.0,
        )  # fmt: skip
        hidden = [
            n for n in names
            if first[n] < 0.05 and mi[n] >= 0.1 * max(mi.values()) and mi[n] > 0.0
        ]  # fmt: skip
        if hidden:
            name = max(hidden, key=lambda n: mi[n])
            shown = 0.0 if abs(first[name]) < 0.005 else first[name]
            title = (
                f"Input {name} has first-order S = {shown:.2f} yet shares "
                f"{mi[name]:.2f} {result.unit} with {result.quantity}"
            )
        subtitle = (
            f"Left: KSG mutual information (k = {result.k}, "
            f"{result.sample_size:,} runs). Right: Sobol S and S_T"
        )
    title_block(left, title, subtitle)
    return figure


def _format_condition(condition: Mapping[str, float]) -> str:
    """Render a condition mapping as ``name=value`` pairs.

    References:
        Project reporting policy. No equation is evaluated.

    """
    return ", ".join(f"{key}={value:g}" for key, value in condition.items())


@themed
def plot_design_path(
    recommendation: DesignRecommendation,
    *,
    unit: str = "nats",
    ax: Any | None = None,
) -> Any:
    """Plot the expected information gain along a greedy design.

    Step 0 is the baseline design (the calibration experiments, or nothing).
    At step ``k`` the muted dots are the EIG of every candidate added to the
    design so far and the observed-hue path goes through the chosen
    (largest) one, labelled with its condition. The independent re-estimate
    of the full design is a model-hue dot with a two-standard-error whisker.

    Args:
        recommendation: Result of ``UQStudy.design``.
        unit: Information unit of the scores.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the selection is empty.

    Equation:
        [EQ-OED-EIG-001], [EQ-OED-GREEDY-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013],
        [REF-MATPLOTLIB].

    """
    selection = recommendation.selection
    if not selection.selected:
        raise ValueError("the design selected no conditions")
    steps = len(selection.selected)
    path = [selection.baseline_information_gain, *selection.information_gains]
    axes = new_axes(ax)
    palette = active_palette()
    for step, scores in enumerate(selection.candidate_scores, start=1):
        axes.scatter(
            [step] * len(scores),
            scores,
            s=16.0,
            color=palette.muted,
            alpha=0.6,
            edgecolors="none",
            zorder=2,
        )
    line(axes, list(range(steps + 1)), path, color=palette.observed, width=1.8)
    points(axes, list(range(steps + 1)), path, color=palette.observed, size=50.0)
    for step, condition in enumerate(selection.selected, start=1):
        axes.annotate(
            _format_condition(condition),
            (step, path[step]),
            xytext=(0.0, 10.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=palette.observed,
        )
    estimate = recommendation.estimate
    final_x = steps + 0.3
    if estimate.standard_error > 0.0:
        axes.vlines(
            final_x,
            estimate.value - 2.0 * estimate.standard_error,
            estimate.value + 2.0 * estimate.standard_error,
            color=palette.model,
            linewidth=2.0,
            zorder=3,
        )
    points(axes, [final_x], [estimate.value], color=palette.model, size=50.0)
    axes.set_xlim(-0.3, steps + 0.9)
    axes.set_xticks(range(steps + 1))
    base = "baseline" if recommendation.baseline else "no data"
    axes.set_xticklabels([base, *(f"step {k}" for k in range(1, steps + 1))])
    low = min(
        min(min(s) for s in selection.candidate_scores),
        path[0],
        estimate.value - 2.0 * estimate.standard_error,
    )
    high = max(max(path), estimate.value + 2.0 * estimate.standard_error)
    span = max(high - low, 1e-9)
    axes.set_ylim(low - 0.12 * span, high + 0.22 * span)
    axis_labels(axes, "Greedy step", f"Expected information gain [{unit}]")
    direct_labels(
        axes, [(final_x, estimate.value, "re-estimate", palette.model)]
    )
    picks = " then ".join(_format_condition(c) for c in selection.selected)
    title_block(
        axes,
        f"{'Runs' if steps > 1 else 'A run'} at {picks} "
        f"raise{'' if steps > 1 else 's'} the EIG from "
        f"{path[0]:.2f} to {path[-1]:.2f} {unit}",
        f"Grey: every candidate at each step ({selection.method}) · "
        f"re-estimate {estimate.value:.2f} +/- {2.0 * estimate.standard_error:.2f}"
        " (2 SE) on an independent stream",
    )
    return axes
