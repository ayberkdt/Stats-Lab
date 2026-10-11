"""Figures for the stages of a ``UQStudy``.

The functions take the stage result objects of :mod:`probstat.workflow`
(``TailAnalysis``, ``InformationSensitivity``, ``DesignRecommendation``,
``DesignPointAnalysis``), the AK-MCS result of ``active_reliability``, and
the calibration posterior of ``calibrate``. They never rerun the model; a POT
tail curve is evaluated from the stored fit and a FORM hyperplane from the
stored design point and unit normal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.inverse import CalibrationPosterior
from probstat.modeling import JointInput
from probstat.reliability import AKMCSResult
from probstat.risk.extremes import ReturnLevel
from probstat.workflow import (
    DesignPointAnalysis,
    DesignRecommendation,
    InformationSensitivity,
    TailAnalysis,
)

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    FITTED,
    PROJECTED,
    REFERENCE,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    band,
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
    quantiles (from the stage summary), or every stored exceedance at its
    Hazen tail probability ``(k - 1/2)/n`` when the result keeps them, are
    observed-hue dots; a level under study is the accent line with its POT
    probability (model hue) and its empirical Wilson interval (observed hue).

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
    exceedances = list(tail.exceedances)
    if exceedances:
        total = len(exceedances)
        axes.scatter(
            exceedances,
            [(total - i - 0.5) / count for i in range(total)],
            s=12.0,
            color=palette.observed,
            edgecolors="none",
            alpha=0.85,
            zorder=4,
        )
    elif empirical_levels:
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
    visible.extend(exceedances)
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
            "inside"
            if interval.lower <= tail.pot_probability <= interval.upper
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
    dots = "dots: exceedances" if exceedances else "dots: empirical"
    title_block(
        axes,
        title,
        f"{fit.rate.exceedances} exceedances of {count:,} runs · dashed: fit · "
        f"dotted: beyond the sample · band: {confidence:.0%} delta interval · "
        f"{dots}",
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
    direct_labels(axes, [(final_x, estimate.value, "re-estimate", palette.model)])
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


def _projection_axes(alphas: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Return the two coordinates with the largest summed ``alpha_i^2``.

    Args:
        alphas: Unit normals of the design points.

    Returns:
        Two coordinate indices in increasing order.

    References:
        [REF-DERKIUREGHIAN-2005]. No equation is evaluated.

    """
    dimension = len(alphas[0])
    weight = [math.fsum(a[i] ** 2 for a in alphas) for i in range(dimension)]
    first, second = sorted(range(dimension), key=lambda i: (-weight[i], i))[:2]
    return min(first, second), max(first, second)


def _free_angle(occupied: Sequence[float]) -> float:
    """Return the compass angle farthest from every occupied angle.

    Args:
        occupied: Angles in radians of marks already on a circle.

    Returns:
        One of eight compass angles in radians.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    candidates = [math.radians(d) for d in (45, 135, 225, 315, 90, 0, 180, 270)]
    if not occupied:
        return candidates[0]

    def gap(angle: float) -> float:
        """Return the smallest angular distance to an occupied angle.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return min(
            abs(math.remainder(angle - other, 2.0 * math.pi)) for other in occupied
        )

    return max(candidates, key=gap)


@themed
def plot_design_points(
    analysis: DesignPointAnalysis,
    *,
    input_names: Sequence[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot design points and their FORM hyperplanes in standard-normal space.

    Each design point ``u*_k`` is a model-hue dot joined to the origin by a
    muted ray of length ``|beta_k|``; its FORM linearization, the hyperplane
    ``alpha_k . u = alpha_k . u*_k`` through the point, is a ``FITTED``
    model-hue line. The accent circle of radius ``min |beta_k|`` is the
    reference: every point on it is as likely as the most likely failure
    point. Only stored results are drawn (no limit-state evaluations). With
    more than two inputs the plane of the two coordinates with the largest
    summed ``alpha_i^2`` is shown and each hyperplane is cut through its own
    design point.

    Args:
        analysis: Result of ``UQStudy.design_points``.
        input_names: Input names in model order (default: the FORM
            importance-factor keys).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: With fewer than two inputs or misaligned names.

    Equation:
        [EQ-REL-FORM-001], [EQ-REL-MDP-002].

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-HASOFER-LIND-1974],
        [REF-MATPLOTLIB].

    """
    forms = analysis.points.forms
    dimension = len(forms[0].design_point_u)
    if dimension < 2:
        raise ValueError("the standard-normal plot needs at least two inputs")
    names = (
        list(input_names)
        if input_names is not None
        else list(forms[0].importance_factors)
    )
    if len(names) != dimension:
        raise ValueError("input_names must have one name per input")
    first, second = _projection_axes([form.alpha for form in forms])
    betas = [abs(form.beta) for form in forms]
    smallest = min(betas)
    limit = 1.45 * max(max(betas), 1.0)
    axes = new_axes(ax, figsize=(7.6, 6.4))
    palette = active_palette()
    turns = [2.0 * math.pi * i / 240 for i in range(241)]
    axes.plot(
        [smallest * math.cos(t) for t in turns],
        [smallest * math.sin(t) for t in turns],
        color=palette.accent,
        linewidth=REFERENCE_WIDTH,
        linestyle=REFERENCE,
        zorder=2,
    )
    xs: list[float] = []
    ys: list[float] = []
    for form in forms:
        alpha, point = form.alpha, form.design_point_u
        x, y = point[first], point[second]
        xs.append(x)
        ys.append(y)
        axes.plot([0.0, x], [0.0, y], color=palette.muted, linewidth=0.8, zorder=2)
        norm = math.hypot(alpha[first], alpha[second])
        if norm > 1e-9:
            dx, dy = -alpha[second] / norm, alpha[first] / norm
            reach = 3.0 * limit
            line(
                axes,
                [x - reach * dx, x + reach * dx],
                [y - reach * dy, y + reach * dy],
                color=palette.model,
                dashes=FITTED,
                width=1.4,
            )
    points(axes, xs, ys, color=palette.model, size=56.0, zorder=5.0)
    for index, (x, y, beta) in enumerate(zip(xs, ys, betas, strict=True), start=1):
        radius = math.hypot(x, y) or 1.0
        axes.annotate(
            f"u*{index}  beta {beta:.2f}",
            (x, y),
            xytext=(10.0 * x / radius, 10.0 * y / radius),
            textcoords="offset points",
            ha="left" if x >= 0.0 else "right",
            va="bottom" if y >= 0.0 else "top",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=palette.model,
        )
    angle = _free_angle([math.atan2(y, x) for x, y in zip(xs, ys, strict=True)])
    axes.annotate(
        f"beta = {smallest:.2f}",
        (smallest * math.cos(angle), smallest * math.sin(angle)),
        xytext=(6.0 * math.cos(angle), 6.0 * math.sin(angle)),
        textcoords="offset points",
        ha="left" if math.cos(angle) >= 0.0 else "right",
        va="bottom" if math.sin(angle) >= 0.0 else "top",
        fontsize=ANNOTATION_SIZE,
        fontweight="semibold",
        color=palette.accent,
    )
    points(axes, [0.0], [0.0], color=palette.muted, size=18.0)
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal", adjustable="datalim")
    axis_labels(
        axes,
        f"u[{names[first]}] (standard normal)",
        f"u[{names[second]}] (standard normal)",
    )
    system = analysis.system.probability
    single = forms[0].probability
    if len(forms) == 1:
        title = (
            f"One design point at beta = {betas[0]:.2f}: FORM gives Pf = {single:.2e}"
        )
    else:
        title = (
            f"{len(forms)} design points: multi-point FORM gives "
            f"Pf = {system:.2e}, {system / single:.1f}x one point"
        )
    subtitle = "Dots: design points · dashed: FORM hyperplanes · circle: smallest beta"
    if analysis.mixture is not None:
        subtitle += (
            f" · mixture IS {analysis.mixture.probability:.2e} "
            f"(c.o.v. {analysis.mixture.coefficient_of_variation:.2f})"
        )
    if dimension > 2:
        subtitle += f" · plane of {names[first]}, {names[second]}"
    title_block(axes, title, subtitle)
    return axes


@themed
def plot_active_learning(
    result: AKMCSResult, *, learning_threshold: float = 2.0
) -> Any:
    """Plot the AK-MCS estimate and stopping statistic against training runs.

    Top panel: the surrogate estimate ``N_f / N`` after each learning step
    (model hue) with a band of two Monte Carlo standard errors
    ``sqrt(P_f (1 - P_f) / N)`` of the population; crude Monte Carlo on the
    same population, when it was run, is the accent reference. Bottom panel:
    the smallest learning value ``U = |mu| / sigma`` over the population with
    the stopping threshold as the reference line. Two panels share the
    horizontal axis instead of a dual vertical axis.

    Args:
        result: Result of ``UQStudy.active_reliability``.
        learning_threshold: Stopping threshold of ``min U`` used by the run.

    Returns:
        The figure (two panels).

    Raises:
        ValueError: If the history has fewer than two records.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-MATPLOTLIB].

    """
    history = result.history
    if len(history) < 2:
        raise ValueError("the learning history needs at least two records")
    plt = pyplot()
    palette = active_palette()
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.16},
    )
    style_axes(top)
    style_axes(bottom)
    xs = [float(item.training_size) for item in history]
    estimates = [item.probability for item in history]
    errors = [
        2.0 * math.sqrt(max(p * (1.0 - p), 0.0) / item.population)
        for p, item in zip(estimates, history, strict=True)
    ]
    band(
        top,
        xs,
        [max(p - e, 0.0) for p, e in zip(estimates, errors, strict=True)],
        [p + e for p, e in zip(estimates, errors, strict=True)],
        color=palette.model,
    )
    line(top, xs, estimates, color=palette.model, width=1.8)
    top.set_xlim(xs[0], xs[-1])
    labels = [(xs[-1], estimates[-1], "AK-MCS", palette.model)]
    reference = result.reference_probability
    if math.isfinite(reference):
        reference_line(top, y=reference)
        labels.append((xs[-1], reference, "Monte Carlo", palette.accent))
    axis_labels(top, None, "Failure probability estimate")
    direct_labels(top, labels)
    finite = [
        (x, item.min_learning)
        for x, item in zip(xs, history, strict=True)
        if math.isfinite(item.min_learning)
    ]
    tallest = learning_threshold
    if finite:
        line(
            bottom,
            [x for x, _ in finite],
            [u for _, u in finite],
            color=palette.model,
            width=1.6,
        )
        tallest = max(u for _, u in finite)
    ceiling = min(1.1 * tallest, 4.0 * learning_threshold)
    bottom.set_ylim(0.0, max(1.6 * learning_threshold, ceiling))
    reference_line(
        bottom, y=learning_threshold, label=f"stop: U = {learning_threshold:g}"
    )
    axis_labels(bottom, "Training runs of the true model", "Smallest U")
    grown = sum(
        1
        for a, b in zip(history, history[1:], strict=False)
        if b.population > a.population
    )
    if result.converged:
        title = (
            f"AK-MCS settles at Pf = {result.probability:.2e} after "
            f"{result.evaluations} model runs"
        )
    else:
        title = (
            f"AK-MCS stopped at Pf = {result.probability:.2e} after "
            f"{result.evaluations} runs without meeting its stopping rule"
        )
    if math.isfinite(reference):
        title += (
            "; Monte Carlo on the same points agrees"
            if abs(reference - result.probability) <= 1e-12
            else f"; Monte Carlo on the same points gives {reference:.2e}"
        )
    subtitle = (
        f"Band: 2 standard errors on {result.population:,} points · "
        "bottom: min U = |mu| / sigma over the population"
    )
    if grown:
        subtitle += f" · population enlarged {grown}x"
    title_block(top, title, subtitle)
    return figure


@themed
def plot_posterior_marginals(
    posterior: CalibrationPosterior,
    *,
    prior: JointInput | None = None,
    max_panels: int = 4,
) -> Any:
    """Plot the posterior marginal of each calibration parameter.

    One panel per parameter (at most ``max_panels``): a density histogram of
    the pooled MCMC draws in the observed hue (they are simulated draws) and,
    with the prior law, its marginal density as a muted context line. The
    title names the parameter the data narrow most, as its posterior over
    prior standard deviation.

    Args:
        posterior: Result of ``UQStudy.calibrate``.
        prior: The study model's joint input law (the calibration prior).
        max_panels: Maximum number of parameters drawn.

    Returns:
        The figure.

    Raises:
        ValueError: If ``max_panels`` is not positive or the prior lacks a
            parameter.

    Equation:
        [EQ-CAL-NL-001], [EQ-MC-MEAN-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11], [REF-MATPLOTLIB].

    """
    if max_panels < 1:
        raise ValueError("max_panels must be positive")
    names = posterior.parameter_names[:max_panels]
    pooled = posterior.mcmc.pooled()
    count = len(names)
    plt = pyplot()
    palette = active_palette()
    figure, grid = plt.subplots(
        1,
        count,
        figsize=(max(6.4, 3.4 * count), 3.9),
        squeeze=False,
        gridspec_kw={"wspace": 0.3},
    )
    laws: dict[str, Any] = {}
    prior_sds: dict[str, float] = {}
    if prior is not None:
        missing = [name for name in names if name not in prior.names]
        if missing:
            raise ValueError(f"the prior lacks parameters {missing}")
        deviations = prior.standard_deviations()
        for item, deviation in zip(prior.inputs, deviations, strict=True):
            laws[item.name] = item.distribution
            prior_sds[item.name] = deviation
    ratios: dict[str, float] = {}
    means: dict[str, float] = {}
    spreads: dict[str, float] = {}
    lead = grid[0][0]
    for axes, name in zip(grid[0], names, strict=True):
        style_axes(axes)
        column = posterior.mcmc.index(name)
        values = [row[column] for row in pooled]
        mean = math.fsum(values) / len(values)
        spread = math.sqrt(
            math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
        )
        means[name], spreads[name] = mean, spread
        bins = max(12, min(40, int(2.0 * len(values) ** (1.0 / 3.0))))
        axes.hist(
            values,
            bins=bins,
            density=True,
            color=palette.observed,
            alpha=0.55,
            edgecolor=palette.canvas,
            linewidth=0.6,
            zorder=1,
        )
        low, high = min(values), max(values)
        pad = 0.25 * (high - low or 1.0)
        if name in laws:
            xs = [low - pad + (high - low + 2.0 * pad) * i / 200 for i in range(201)]
            density = [float(laws[name].pdf(x)) for x in xs]
            line(axes, xs, density, color=palette.muted, width=1.4)
            if prior_sds[name] > 0.0:
                ratios[name] = spread / prior_sds[name]
        axes.set_xlim(low - pad, high + pad)
        axes.set_ylim(0.0, None)
        axis_labels(axes, name, "Density" if axes is lead else None)
    rhat = max(
        entry.rank_normalized_rhat
        for entry in posterior.mcmc.summary()
        if entry.name in names
    )
    if ratios:
        name = min(ratios, key=lambda n: ratios[n])
        if ratios[name] < 0.8:
            title = (
                f"The data narrow {name} to {ratios[name]:.0%} of its prior "
                f"spread (posterior mean {means[name]:.3g})"
            )
        else:
            title = (
                f"The data barely inform {name}: its posterior spread is "
                f"{ratios[name]:.0%} of the prior"
            )
    else:
        name = names[0]
        title = (
            f"The posterior mean of {name} is {means[name]:.3g} "
            f"(sd {spreads[name]:.3g})"
        )
    subtitle = (
        f"Bars: {len(pooled):,} pooled draws from {len(posterior.mcmc.chains)} "
        f"chains · max R-hat {rhat:.3f}"
    )
    if laws:
        subtitle += " · grey line: prior density"
    title_block(lead, title, subtitle)
    return figure
