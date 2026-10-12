"""Plots for forecast scoring, calibration, coverage, and distribution shift.

The functions take result objects of :mod:`probstat.validation` (ADR-0042)
or per-case numbers and only present them; they never refit a forecast. PIT
histograms are drawn by
:func:`probstat.visualization.hierarchical_plots.plot_pit_histogram`, which
already shows the binomial band of a calibrated forecast.
:func:`plot_validation_predictions` draws a ``ValidationAssessment``
(predictive bands in the model hue, observations in the observed hue).

Colour roles (ADR-0024, ADR-0046): observed frequencies, empirical coverage,
mean scores, and new data use the observed hue; the identity of perfect
calibration or coverage is the accent reference; bands of what a reliable
forecast would produce and reference-period data are muted context.
Several forecasters on one coverage plot take ``series[0]``, ``series[3]``,
``series[4]``, and ``series[5]``; ``series[1]`` and ``series[2]`` are
skipped because the model hue implies a fitted curve and ``series[2]`` is
close to the porcelain accent.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.foundations.validation import coerce_finite_sample
from probstat.inference import TwoSampleKolmogorovSmirnovResult
from probstat.modeling import standard_normal_quantile
from probstat.statistics import quantiles
from probstat.validation import (
    CoverageAtLevel,
    ReliabilityDiagram,
    ValidationAssessment,
)

from ._layout import wrapped_title_block
from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    INNER_BAND_ALPHA,
    MAX_BAND_ALPHA,
    REFERENCE,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    direct_labels,
    line,
    new_axes,
    points,
    pyplot,
    style_axes,
    themed,
)

_COVERAGE_SLOTS = (0, 3, 4, 5)


def _identity(axes: Any, low: float, high: float) -> None:
    """Draw the identity line ``y = x`` as the accent reference.

    Args:
        axes: Target axes.
        low: Start of the line.
        high: End of the line.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    axes.plot(
        [low, high], [low, high], color=active_palette().accent,
        linewidth=REFERENCE_WIDTH, linestyle=REFERENCE, zorder=2,
    )  # fmt: skip


@themed
def plot_reliability_diagram(
    diagram: ReliabilityDiagram, *, event: str = "the event"
) -> Any:
    """Plot observed event frequency against forecast probability.

    The top panel shows each non-empty bin's observed frequency at its mean
    forecast probability (observed hue) over the diagonal of perfect
    reliability (accent). When the diagram carries consistency bands, each
    bin gets a muted vertical bar spanning the frequencies a reliable
    forecaster would produce with the same forecasts. The lower panel,
    sharing the probability axis, shows how many forecasts fall in each bin.

    Args:
        diagram: Output of :func:`probstat.validation.reliability_diagram`.
        event: Short name of the forecast event for the title.

    Returns:
        The figure with two stacked panels.

    Raises:
        ValueError: If fewer than two bins hold forecasts.

    Equation:
        [EQ-VAL-CAL-REL-001].

    References:
        [REF-BROCKER-SMITH-2007], [REF-MURPHY-1973], [REF-HAMILL-2001],
        [REF-MATPLOTLIB].

    """
    used = [item for item in diagram.bins if item.count > 0]
    if len(used) < 2:
        raise ValueError("at least two non-empty bins are required")
    plt = pyplot()
    palette = active_palette()
    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(6.4, 6.6),
        gridspec_kw={"height_ratios": (4.0, 1.0), "hspace": 0.08},
    )  # fmt: skip
    for axes in (top, bottom):
        style_axes(axes)
    banded = diagram.consistency_level is not None
    if banded:
        for item in used:
            top.vlines(
                item.mean_forecast, item.consistency_lower, item.consistency_upper,
                colors=palette.muted, linewidth=7.0, alpha=MAX_BAND_ALPHA, zorder=1,
            )  # fmt: skip
    _identity(top, 0.0, 1.0)
    xs = [item.mean_forecast for item in used]
    ys = [item.observed_frequency for item in used]
    line(top, xs, ys, color=palette.observed, width=1.6)
    points(top, xs, ys, color=palette.observed, size=46.0)
    top.set_xlim(0.0, 1.0)
    top.set_ylim(0.0, 1.0)
    top.grid(axis="both", color=palette.rule, linewidth=0.6)
    direct_labels(top, [(1.0, 1.0, "perfect reliability", palette.accent)])
    axis_labels(top, None, "Observed frequency")
    for item in diagram.bins:
        centre = 0.5 * (item.lower + item.upper)
        bottom.bar(
            centre, item.count, width=0.86 * (item.upper - item.lower),
            color=palette.muted, alpha=0.45, linewidth=0.0, zorder=2,
        )  # fmt: skip
        bottom.annotate(
            f"{item.count:,}", (centre, item.count), xytext=(0.0, 2.0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=ANNOTATION_SIZE, color=palette.ink, annotation_clip=False,
        )  # fmt: skip
    peak = max(item.count for item in diagram.bins)
    bottom.set_ylim(0.0, 1.45 * peak)
    bottom.set_yticks([])
    bottom.grid(axis="y", visible=False)
    axis_labels(bottom, "Forecast probability", "Forecasts")
    total = sum(item.count for item in diagram.bins)
    outside = diagram.outside_bands
    worst = max(abs(item.observed_frequency - item.mean_forecast) for item in used)
    if not banded:
        verb = "stay within" if worst < 0.05 else "depart by up to"
        title = f"Observed frequencies {verb} {worst:.2f} of the forecast probabilities"
    elif not outside:
        title = (
            f"All {len(used)} bins fall inside the {diagram.consistency_level:.0%} "
            f"consistency bands: forecasts of {event} are reliable"
        )
    else:
        pulls = [
            (diagram.bins[i].observed_frequency - diagram.bins[i].mean_forecast)
            * (0.5 - diagram.bins[i].mean_forecast)
            for i in outside
        ]
        if all(p > 0.0 for p in pulls):
            verdict = "overconfident (outcomes are less extreme)"
        elif all(p < 0.0 for p in pulls):
            verdict = "underconfident (outcomes are more extreme)"
        else:
            verdict = "miscalibrated"
        title = (
            f"{len(outside)} of {len(used)} bins leave the "
            f"{diagram.consistency_level:.0%} consistency bands: forecasts of "
            f"{event} are {verdict}"
        )
    parts = [f"dots: observed frequency at the bin's mean forecast ({total:,} cases)"]
    if banded:
        parts.append(
            f"grey bars: range if reliable ({diagram.resamples:,} resamples)"
        )
    parts.append("bottom: forecasts per bin")
    wrapped_title_block(top, title, parts)
    return figure


def _mean_interval(values: Sequence[float], z: float) -> tuple[float, float, float]:
    """Return the mean and its normal-approximation interval.

    Args:
        values: At least two finite numbers.
        z: Standard normal quantile of the interval.

    Returns:
        ``(mean, lower, upper)``.

    Equation:
        [EQ-INF-CI-NMEAN-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 9.2].

    """
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    half = z * math.sqrt(variance / n)
    return mean, mean - half, mean + half


@themed
def plot_score_comparison(
    scores: Mapping[str, Sequence[float]],
    *,
    score_name: str = "CRPS",
    confidence: float = 0.95,
    ax: Any | None = None,
) -> Any:
    """Compare forecasters by their mean score with interval whiskers.

    Rows are sorted so the lowest (best) mean score is on top. Each dot is
    the mean of the per-case scores, each whisker its normal-approximation
    interval (EQ-INF-CI-NMEAN-001). When all forecasters were scored on the
    same cases, the title reports the paired difference between the best and
    the runner-up, whose interval is much narrower than either whisker
    because case difficulty cancels.

    Args:
        scores: Per-case scores of each forecaster (negatively oriented:
            smaller is better).
        score_name: Name of the score for labels and title.
        confidence: Interval level strictly inside ``(0, 1)``.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If fewer than two forecasters, fewer than two cases, a
            non-finite score, or an invalid level is given.

    Equation:
        [EQ-INF-CI-NMEAN-001], [EQ-VAL-SCORE-CRPS-001].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-CASELLA-BERGER-2002, Section 9.2],
        [REF-MATPLOTLIB].

    """
    if len(scores) < 2:
        raise ValueError("at least two forecasters are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    samples = {name: coerce_finite_sample(v, name=name) for name, v in scores.items()}
    if any(len(v) < 2 for v in samples.values()):
        raise ValueError("every forecaster needs at least two scored cases")
    z = standard_normal_quantile(0.5 + 0.5 * confidence)
    summary = {name: _mean_interval(v, z) for name, v in samples.items()}
    order = sorted(summary, key=lambda name: summary[name][0])
    axes = new_axes(ax, figsize=(7.2, 0.55 * len(order) + 2.0))
    palette = active_palette()
    rows = list(range(len(order) - 1, -1, -1))
    for row, name in zip(rows, order, strict=True):
        mean, low, high = summary[name]
        axes.hlines(row, low, high, color=palette.observed, linewidth=2.2, zorder=2)
        points(axes, [mean], [row], color=palette.observed, size=54.0)
        axes.annotate(
            f"{mean:.3f}", (high, row), xytext=(6.0, 0.0),
            textcoords="offset points", ha="left", va="center",
            fontsize=ANNOTATION_SIZE, color=palette.ink,
        )  # fmt: skip
    axes.set_yticks(rows)
    axes.set_yticklabels(order)
    for label in axes.get_yticklabels():
        if label.get_text() == order[0]:
            label.set_fontweight("bold")
    axes.set_ylim(-0.6, len(order) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    low_edge = min(s[1] for s in summary.values())
    high_edge = max(s[2] for s in summary.values())
    span = high_edge - low_edge or 1.0
    axes.set_xlim(low_edge - 0.05 * span, high_edge + 0.16 * span)
    axis_labels(axes, f"Mean {score_name} (lower is better)", None)
    best, runner = order[0], order[1]
    sizes = {len(v) for v in samples.values()}
    if len(sizes) == 1:
        differences = [
            b - a for a, b in zip(samples[best], samples[runner], strict=True)
        ]
        gap, lower, upper = _mean_interval(differences, z)
        if lower > 0.0:
            title = (
                f"{best} has the lowest mean {score_name}; {runner} is worse by "
                f"{gap:.3f} (paired {confidence:.0%} CI {lower:.3f} to {upper:.3f})"
            )
        else:
            title = (
                f"{best} and {runner} are indistinguishable in mean {score_name}: "
                f"paired difference {gap:.3f} ({lower:.3f} to {upper:.3f})"
            )
        cases = f"{sizes.pop():,} common cases"
    else:
        title = (
            f"{best} has the lowest mean {score_name}, "
            f"{summary[runner][0] - summary[best][0]:.3f} below {runner}"
        )
        cases = "unpaired cases"
    wrapped_title_block(
        axes,
        title,
        [f"dots: mean over {cases}", f"bars: {confidence:.0%} interval of the mean"],
    )
    return axes


@themed
def plot_coverage(
    reports: Mapping[str, Sequence[CoverageAtLevel]] | Sequence[CoverageAtLevel],
    *,
    ax: Any | None = None,
) -> Any:
    """Plot empirical against nominal coverage of central intervals.

    Each forecaster is a line of dots at its empirical coverage rate with
    whiskers for the binomial interval stored in each record (Clopper-Pearson
    or Wilson, as computed by :func:`probstat.validation.pit_coverage` or
    :func:`probstat.validation.interval_coverage`). Several forecasters are
    offset slightly along the nominal axis so whiskers do not overlap; the
    identity ``empirical = nominal`` is the accent reference.

    Args:
        reports: Coverage records of one forecaster, or a mapping from
            forecaster name to its records (at most four).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If no records, more than four forecasters, or a
            forecaster with fewer than one record is given.

    Equation:
        [EQ-VAL-CAL-COV-001], [EQ-INF-CI-CP-001], [EQ-INF-CI-WILSON-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-CLOPPER-PEARSON-1934],
        [REF-WILSON-1927], [REF-MATPLOTLIB].

    """
    named: dict[str, list[CoverageAtLevel]] = (
        {name: list(items) for name, items in reports.items()}
        if isinstance(reports, Mapping)
        else {"forecast": list(reports)}
    )
    if not named or any(not items for items in named.values()):
        raise ValueError("every forecaster needs at least one coverage record")
    if len(named) > len(_COVERAGE_SLOTS):
        raise ValueError(f"at most {len(_COVERAGE_SLOTS)} forecasters can be drawn")
    axes = new_axes(ax, figsize=(6.4, 6.2))
    palette = active_palette()
    nominals = sorted({r.nominal for items in named.values() for r in items})
    spacing = min(
        (b - a for a, b in zip(nominals, nominals[1:], strict=False)), default=0.1
    )
    step = min(0.012, 0.15 * spacing)
    low = min(nominals) - max(0.05, spacing)
    high = 1.0
    _identity(axes, low, high)
    labels = []
    for index, (name, items) in enumerate(named.items()):
        colour = palette.series[_COVERAGE_SLOTS[index]]
        shift = step * (index - 0.5 * (len(named) - 1))
        ordered = sorted(items, key=lambda r: r.nominal)
        xs = [r.nominal + shift for r in ordered]
        ys = [r.rate for r in ordered]
        axes.vlines(
            xs,
            [r.interval.lower for r in ordered],
            [r.interval.upper for r in ordered],
            colors=colour,
            linewidth=1.6,
            zorder=3,
        )
        line(axes, xs, ys, color=colour, width=1.1)
        points(axes, xs, ys, color=colour, size=34.0)
        labels.append((high, ys[-1], name, colour))
    axes.set_xlim(low, high)
    bottom = min(r.interval.lower for items in named.values() for r in items)
    axes.set_ylim(min(low, bottom - 0.03), 1.0)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    direct_labels(axes, labels)
    axis_labels(axes, "Nominal coverage", "Empirical coverage")
    gaps = {
        name: max(items, key=lambda r: abs(r.rate - r.nominal))
        for name, items in named.items()
    }
    consistent = [n for n, items in named.items() if all(r.consistent for r in items)]
    worst = max(gaps, key=lambda n: abs(gaps[n].rate - gaps[n].nominal))
    first = next(iter(named.values()))[0]
    level = first.interval.confidence
    method = first.interval.method.removesuffix(" interval")
    if len(consistent) == len(named):
        title = (
            f"Every forecast covers its nominal levels within the {level:.0%} "
            "binomial intervals"
        )
    else:
        record = gaps[worst]
        tail = (
            f"{', '.join(consistent)} "
            f"{'stays' if len(consistent) == 1 else 'stay'} consistent at every level"
            if consistent
            else "no forecast is consistent at every level"
        )
        title = (
            f"{worst} covers {record.rate:.0%} at nominal {record.nominal:.0%}; "
            f"{tail}"
        )
    wrapped_title_block(
        axes,
        title,
        [
            "dots: empirical coverage of central intervals",
            f"bars: {level:.0%} {method} intervals",
            f"n = {first.size:,}",
        ],
    )
    return axes


def ks_gap(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return where two empirical CDFs differ most.

    Both ECDFs are evaluated just after every distinct pooled value (they
    are right-continuous steps), and the first location attaining
    ``D = max |F_m - G_n|`` is returned.

    Args:
        first: Non-empty finite sample.
        second: Non-empty finite sample.

    Returns:
        ``(location, D, F_m(location), G_n(location))``.

    Raises:
        ValueError: If a sample is empty or not finite.

    Equation:
        [EQ-STAT-005], [EQ-INF-GOF-KS2-001].

    References:
        [REF-HODGES-1958], [REF-HOLLANDER-WOLFE-CHICKEN-2014].

    """
    a = sorted(coerce_finite_sample(first, name="first"))
    b = sorted(coerce_finite_sample(second, name="second"))
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    best = (a[0], -1.0, 0.0, 0.0)
    for value in sorted(set(a) | set(b)):
        fa = bisect.bisect_right(a, value) / len(a)
        fb = bisect.bisect_right(b, value) / len(b)
        if abs(fa - fb) > best[1] + 1e-15:
            best = (value, abs(fa - fb), fa, fb)
    return best


def _ecdf_steps(
    values: Sequence[float], low: float, high: float
) -> tuple[list[float], list[float]]:
    """Return step coordinates of an ECDF padded to ``[low, high]``.

    Args:
        values: Sorted sample.
        low: Left end of the drawn range.
        high: Right end of the drawn range.

    Returns:
        Abscissae and ordinates for a post-step line.

    Equation:
        [EQ-STAT-005].

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014].

    """
    n = len(values)
    xs = [low, *values, high]
    ys = [0.0, *((i + 1) / n for i in range(n)), 1.0]
    return xs, ys


@themed
def plot_distribution_shift(
    first: Sequence[float],
    second: Sequence[float],
    *,
    names: tuple[str, str] = ("reference", "new"),
    result: TwoSampleKolmogorovSmirnovResult | None = None,
    variable: str = "Value",
    alpha: float = 0.05,
    ax: Any | None = None,
) -> Any:
    """Plot two empirical CDFs with the Kolmogorov-Smirnov distance marked.

    The reference sample is muted context, the new sample is in the observed
    hue, and an ink segment marks the largest vertical gap ``D`` at its
    location (:func:`ks_gap`). With a test result the title states whether
    the shift is detected at level ``alpha``; the result's statistic must
    equal the plotted ``D``, which ties the test to these samples.

    Args:
        first: Reference sample (for example a calibration period).
        second: New sample (for example a deployment period).
        names: Labels of the two samples.
        result: Optional output of
            :func:`probstat.inference.kolmogorov_smirnov_two_sample_test`.
        variable: Name of the measured quantity.
        alpha: Significance level for the title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If a sample is empty, ``alpha`` is invalid, or the result
            does not belong to the samples.

    Equation:
        [EQ-STAT-005], [EQ-INF-GOF-KS2-001].

    References:
        [REF-HODGES-1958], [REF-HOLLANDER-WOLFE-CHICKEN-2014],
        [REF-MATPLOTLIB].

    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    where, distance, fa, fb = ks_gap(first, second)
    if result is not None and abs(result.statistic - distance) > 1e-9:
        raise ValueError("the test result does not belong to these samples")
    a = sorted(float(v) for v in first)
    b = sorted(float(v) for v in second)
    span = max(a[-1], b[-1]) - min(a[0], b[0]) or 1.0
    low = min(a[0], b[0]) - 0.04 * span
    high = max(a[-1], b[-1]) + 0.04 * span
    axes = new_axes(ax)
    palette = active_palette()
    for sample, colour, width in ((a, palette.muted, 1.6), (b, palette.observed, 2.0)):
        xs, ys = _ecdf_steps(sample, low, high)
        axes.step(xs, ys, where="post", color=colour, linewidth=width, zorder=3)
    axes.plot(
        [where, where], [fa, fb], color=palette.ink, linewidth=1.6,
        marker="_", markersize=9, zorder=4,
    )  # fmt: skip
    middle = 0.5 * (fa + fb)
    upward = middle <= 0.6
    axes.annotate(
        f"D = {distance:.3f} at {where:.2f}",
        xy=(where, middle),
        xytext=(-30.0, 46.0) if upward else (30.0, -46.0),
        textcoords="offset points",
        ha="right" if upward else "left",
        va="center",
        fontsize=ANNOTATION_SIZE,
        color=palette.ink,
        arrowprops={
            "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
            "color": palette.muted,
            "linewidth": 0.8,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
            "connectionstyle": "arc3,rad=-0.15",
        },
    )
    axes.set_xlim(low, high)
    axes.set_ylim(0.0, 1.04)
    direct_labels(
        axes,
        [
            (high, 1.0, f"{names[0]} (n = {len(a)})", palette.muted),
            (high, 1.0, f"{names[1]} (n = {len(b)})", palette.observed),
        ],
    )
    axis_labels(axes, variable, "Empirical CDF")
    kind = "exact" if result is not None and result.exact else "asymptotic"
    if result is None:
        title = (
            f"The ECDFs of {names[0]} and {names[1]} differ by at most "
            f"D = {distance:.3f}"
        )
        tail = "ink: largest gap D"
    elif result.p_value < alpha:
        title = (
            f"{names[1]} has shifted: KS distance {distance:.3f} "
            f"(p = {result.p_value:.1e} < {alpha:g})"
        )
        tail = f"ink: largest gap D · {kind} p-value"
    else:
        title = (
            f"No shift detected between {names[0]} and {names[1]}: "
            f"KS distance {distance:.3f} (p = {result.p_value:.2f})"
        )
        tail = f"ink: largest gap D · {kind} p-value"
    wrapped_title_block(
        axes, title, [f"grey: {names[0]}", f"colour: {names[1]}", tail]
    )
    return axes


def _verdict_title(
    assessment: ValidationAssessment, inside: int, total: int, level: float
) -> str:
    """Return the assertion title of a validation figure from its criteria.

    Args:
        assessment: Validation assessment whose criteria decide the verdict.
        inside: Observations inside the outer predictive intervals.
        total: Number of observations.
        level: Nominal level of the outer intervals.

    Returns:
        ``"consistent ..."`` with the coverage count, or ``"inconsistent: ..."``
        naming every criterion that failed.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapter 12]. No equation is evaluated.

    """
    count = len(assessment.experiments)
    if assessment.consistent:
        return (
            f"Model is consistent with the {count} validation experiment"
            f"{'s' if count != 1 else ''}; {inside} of {total} observations lie "
            f"inside the {level:.0%} predictive intervals"
        )
    reasons: list[str] = []
    for criterion in assessment.criteria:
        if criterion.passed:
            continue
        if criterion.name == "maximum area metric":
            over = sum(
                1 for item in assessment.experiments if not item.within_tolerance
            )
            reasons.append(
                f"area metric exceeds {assessment.tolerance:.3g} in {over} of "
                f"{count} experiments"
            )
        elif criterion.name == "u-pooling KS p-value":
            reasons.append(
                f"pooled PIT rejects uniformity (p = {criterion.value:.2g})"
            )
        else:
            reasons.append(f"{criterion.name} not met")
    return "Model is inconsistent with the validation data: " + " and ".join(reasons)


@themed
def plot_validation_predictions(
    assessment: ValidationAssessment,
    *,
    coverage: Sequence[CoverageAtLevel] = (),
    levels: tuple[float, float] = (0.5, 0.9),
    pit_mean: float | None = None,
) -> Any:
    """Plot each experiment's predictive intervals against its observations.

    One row per validation experiment: the central ``levels`` intervals of
    the observable predictive sample (latent prediction plus measurement
    noise when the assessment includes it) are bands in the model hue with
    the predictive median as a model-hue tick; the measured replicates are
    observed-hue dots. The area metric of each row is printed at the right
    edge, in bold ink when it exceeds the tolerance. The title is the
    assessment's own verdict: the criteria that failed, or the count of
    observations inside the outer intervals. With ``coverage`` records (for
    example the ``coverage[...]`` entries of ``UQStudy.validate``) a second
    panel shows empirical against nominal coverage with their binomial
    intervals and the identity as the accent reference.

    Args:
        assessment: Result of :func:`probstat.validation.assess_validation`
            or ``UQStudy.validate``.
        coverage: Optional coverage records from the PIT values.
        levels: Inner and outer central interval levels in ``(0, 1)``.
        pit_mean: Optional mean PIT value for the subtitle (``1/2`` when
            calibrated).

    Returns:
        The figure (one panel, or two with ``coverage``).

    Raises:
        ValueError: If the assessment has no experiments or ``levels`` are
            not two increasing values inside ``(0, 1)``.

    Equation:
        [EQ-STAT-QUANT-001], [EQ-VAL-AREA-001], [EQ-VAL-CAL-COV-001].

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV], [REF-FERSON-OBERKAMPF-GINZBURG-2008],
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-MATPLOTLIB].

    """
    experiments = assessment.experiments
    if not experiments:
        raise ValueError("the assessment has no experiments")
    inner, outer = float(levels[0]), float(levels[1])
    if not 0.0 < inner < outer < 1.0:
        raise ValueError("levels must be two increasing values inside (0, 1)")
    plt = pyplot()
    palette = active_palette()
    height = max(3.6, 0.62 * len(experiments) + 2.4)
    side: Any | None = None
    if coverage:
        figure, (axes, side) = plt.subplots(
            1,
            2,
            figsize=(10.4, height),
            gridspec_kw={"width_ratios": (2.4, 1.0), "wspace": 0.85},
        )
        style_axes(side)
    else:
        figure, axes = plt.subplots(figsize=(7.6, height))
    style_axes(axes)
    probabilities = (
        0.5 * (1.0 - outer), 0.5 * (1.0 - inner), 0.5,
        0.5 * (1.0 + inner), 0.5 * (1.0 + outer),
    )  # fmt: skip
    rows = list(range(len(experiments) - 1, -1, -1))
    inside = total = 0
    lowest, highest = math.inf, -math.inf
    for row, item in zip(rows, experiments, strict=True):
        q = quantiles(item.observable_predictions, probabilities)
        for low, high, alpha, half in (
            (q[0], q[4], BAND_ALPHA, 0.26),
            (q[1], q[3], INNER_BAND_ALPHA, 0.18),
        ):
            axes.fill_between(
                [low, high], [row - half] * 2, [row + half] * 2,
                color=palette.model, alpha=alpha, linewidth=0.0, zorder=1,
            )  # fmt: skip
        axes.vlines(
            q[2], row - 0.26, row + 0.26, color=palette.model, linewidth=2.0, zorder=3
        )
        values = item.experiment.observations
        spread = [
            0.0 if len(values) == 1 else -0.12 + 0.24 * k / (len(values) - 1)
            for k in range(len(values))
        ]
        points(
            axes, list(values), [row + d for d in spread], color=palette.observed
        )
        inside += sum(1 for v in values if q[0] <= v <= q[4])
        total += len(values)
        lowest = min(lowest, q[0], *values)
        highest = max(highest, q[4], *values)
        failed = not item.within_tolerance
        text = f"area {item.area:.3g}"
        if failed:
            text += f" > {assessment.tolerance:.3g}"
        axes.annotate(
            text, xy=(1.0, row), xycoords=("axes fraction", "data"),
            xytext=(6.0, 0.0), textcoords="offset points", ha="left",
            va="center", fontsize=ANNOTATION_SIZE, annotation_clip=False,
            color=palette.ink if failed else palette.muted,
            fontweight="bold" if failed else "normal",
        )  # fmt: skip
    span = highest - lowest or 1.0
    axes.set_xlim(lowest - 0.04 * span, highest + 0.04 * span)
    axes.set_yticks(rows)
    axes.set_yticklabels([item.experiment.name for item in experiments])
    axes.set_ylim(-0.6, len(experiments) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    axis_labels(axes, assessment.quantity, None, x_unit=assessment.unit)
    if side is not None:
        _coverage_panel(side, coverage)
    noise = "with" if assessment.include_measurement_noise else "without"
    parts = [
        f"bands: central {inner:.0%} and {outer:.0%} predictive intervals "
        f"({assessment.sample_size:,} samples, {noise} measurement noise)",
        "tick: predictive median",
        "dots: observations",
        f"{inside} of {total} inside the {outer:.0%} bands",
    ]
    if pit_mean is not None:
        parts.append(f"PIT mean {pit_mean:.2f} (1/2 if calibrated)")
    if side is not None:
        parts.append("right: coverage with 95% binomial intervals")
    wrapped_title_block(
        axes,
        _verdict_title(assessment, inside, total, outer),
        parts,
        width_inches=figure.get_figwidth(),
    )
    return figure


def _coverage_panel(axes: Any, coverage: Sequence[CoverageAtLevel]) -> None:
    """Draw empirical against nominal coverage on a unit square.

    Args:
        axes: Target axes.
        coverage: Coverage records (one per nominal level).

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    _identity(axes, 0.0, 1.0)
    ordered = sorted(coverage, key=lambda r: r.nominal)
    xs = [r.nominal for r in ordered]
    ys = [r.rate for r in ordered]
    axes.vlines(
        xs, [r.interval.lower for r in ordered], [r.interval.upper for r in ordered],
        colors=palette.observed, linewidth=1.8, zorder=3,
    )  # fmt: skip
    points(axes, xs, ys, color=palette.observed, size=40.0)
    for record in ordered:
        below = record.rate <= record.nominal
        axes.annotate(
            f"{record.covered}/{record.size}", (record.nominal, record.rate),
            xytext=(-7.0, -9.0 if below else 9.0), textcoords="offset points",
            ha="right", va="center", fontsize=ANNOTATION_SIZE, color=palette.ink,
        )  # fmt: skip
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.02)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    axis_labels(axes, "Nominal coverage", "Empirical coverage")
