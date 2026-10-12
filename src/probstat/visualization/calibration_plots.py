"""Plots for Kennedy-O'Hagan calibration with a model-discrepancy term.

The functions take :class:`~probstat.inverse.DiscrepancyCalibration` and the
:class:`~probstat.inverse.DiscrepancyPrediction` records returned by its
``predict`` method; they never refit. Colour roles (ADR-0044):

* observations use the observed hue (``series[0]``);
* the bias-corrected prediction ``eta + delta`` is the model's answer, so it
  takes the model hue (``series[1]``), solid line on its band;
* the simulator-only prediction ``eta`` is the baseline the correction is
  measured against, so it is drawn in muted ink with the ``FITTED`` dash
  (sunumatik chart-aesthetics: context and baseline series take muted ink,
  not a new hue; an extra hue band reads as a third series, the "olive
  blob" failure);
* a supplied truth is a reference curve in the accent colour.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.inverse import DiscrepancyCalibration, DiscrepancyPrediction

from .theme import (
    FITTED,
    INNER_BAND_ALPHA,
    REFERENCE,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    band,
    direct_labels,
    line,
    new_axes,
    points,
    themed,
    title_block,
)


def _normal_cdf(z: float) -> float:
    """Return the standard normal CDF via the complementary error function.

    References:
        [REF-NIST-DLMF]. Equation [EQ-VIZ-MIXQ-001].

    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def gaussian_mixture_interval(
    means: Sequence[float],
    variances: Sequence[float],
    probability: float = 0.95,
    *,
    weights: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Return the central interval of a weighted Gaussian mixture.

    The mixture ``F(q) = sum_s w_s Phi((q - m_s)/sd_s)`` (point masses for
    zero variances) is inverted by bisection at ``(1 -+ probability)/2``. It is
    the exact interval of a posterior that is Gaussian given a parameter and
    mixed over parameter draws or quadrature nodes.

    Args:
        means: Component means.
        variances: Non-negative component variances.
        probability: Central probability strictly inside ``(0, 1)``.
        weights: Non-negative component weights (equal when omitted); they
            are normalized.

    Returns:
        Lower and upper interval limits.

    Raises:
        ValueError: For empty or misaligned inputs, negative variances or
            weights, or an invalid probability.

    Equation:
        [EQ-VIZ-MIXQ-001], [EQ-KOH-PRED-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3], [REF-NIST-DLMF].

    """
    if not means or len(means) != len(variances):
        raise ValueError("means and variances must be non-empty and aligned")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly inside (0, 1)")
    if any(v < 0.0 for v in variances):
        raise ValueError("variances must be non-negative")
    raw = [1.0] * len(means) if weights is None else [float(w) for w in weights]
    if len(raw) != len(means) or any(w < 0.0 for w in raw) or sum(raw) <= 0.0:
        raise ValueError("weights must be non-negative, aligned, and not all zero")
    total = math.fsum(raw)
    largest = max(raw)
    kept = [
        (m, math.sqrt(v), w / total)
        for m, v, w in zip(means, variances, raw, strict=True)
        if w > 1e-12 * largest
    ]

    def cdf(point: float) -> float:
        """Return the mixture CDF at one point.

        References:
            [EQ-VIZ-MIXQ-001]. No new equation is introduced.

        """
        value = 0.0
        for m, sd, w in kept:
            if sd > 0.0:
                value += w * _normal_cdf((point - m) / sd)
            elif point >= m:
                value += w
        return value

    low_start = min(m - 9.0 * sd for m, sd, _ in kept)
    high_start = max(m + 9.0 * sd for m, sd, _ in kept)
    scale = max(1.0, abs(low_start), abs(high_start), high_start - low_start)

    def invert(target: float) -> float:
        """Return the smallest point whose mixture CDF reaches ``target``.

        References:
            [EQ-VIZ-MIXQ-001]. No new equation is introduced.

        """
        low, high = low_start, high_start
        while high - low > 1e-9 * scale:
            middle = 0.5 * (low + high)
            if cdf(middle) < target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    tail = 0.5 * (1.0 - probability)
    return invert(tail), invert(1.0 - tail)


def _axis_values(
    predictions: Sequence[DiscrepancyPrediction], input_name: str
) -> list[float]:
    """Return the prediction abscissae, checking they increase.

    Args:
        predictions: Prediction records.
        input_name: Condition name used as the horizontal axis.

    Returns:
        Condition values in order.

    Raises:
        ValueError: With fewer than two predictions, a missing condition, or
            abscissae that do not increase.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if len(predictions) < 2:
        raise ValueError("at least two predictions are required")
    try:
        xs = [float(item.condition[input_name]) for item in predictions]
    except KeyError as error:
        raise ValueError(f"predictions lack the condition {input_name!r}") from error
    if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
        raise ValueError("predictions must be ordered by increasing condition")
    return xs


def _checked_truth(
    truth: Sequence[float] | None, count: int
) -> list[float] | None:
    """Return the truth as floats aligned with the predictions.

    Raises:
        ValueError: If the length differs from the number of predictions.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if truth is None:
        return None
    values = [float(v) for v in truth]
    if len(values) != count:
        raise ValueError("truth must align with the predictions")
    return values


@themed
def plot_discrepancy_predictions(
    calibration: DiscrepancyCalibration,
    predictions: Sequence[DiscrepancyPrediction],
    input_name: str,
    *,
    truth: Sequence[float] | None = None,
    level: float = 0.95,
    name: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot observations with simulator-only and bias-corrected predictions.

    The simulator-only band is the parameter uncertainty of ``eta(x, theta)``
    (type-7 sample quantiles); the bias-corrected band is the mixture
    interval of ``eta + delta`` (``DiscrepancyPrediction.interval``). With a
    truth the title counts the prediction points whose band covers it;
    otherwise it quotes the largest discrepancy correction.

    Args:
        calibration: Calibration problem that supplied the observations.
        predictions: Records from ``calibration.predict`` at increasing values
            of ``input_name`` (a dense grid draws smooth bands).
        input_name: Condition name of the horizontal axis.
        truth: Optional true process values aligned with ``predictions``.
        level: Central probability of both bands.
        name: Vertical-axis title (default: the calibrated quantity).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: For fewer than two predictions, unordered or missing
            conditions, observations without ``input_name``, a misaligned
            truth, or an invalid level.

    Equation:
        [EQ-KOH-MODEL-001], [EQ-KOH-PRED-001].

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004], [REF-MATPLOTLIB].

    """
    xs = _axis_values(predictions, input_name)
    values = _checked_truth(truth, len(xs))
    try:
        observed_x = [
            float({**calibration.model.fixed, **item.conditions}[input_name])
            for item in calibration.experiments
        ]
    except KeyError as error:
        raise ValueError(f"experiments lack the condition {input_name!r}") from error
    simulator = [item.interval(level, kind="simulator") for item in predictions]
    corrected = [item.interval(level, kind="corrected") for item in predictions]
    axes = new_axes(ax)
    palette = active_palette()
    simulator_colour = palette.muted
    band(
        axes,
        xs,
        [low for low, _ in simulator],
        [high for _, high in simulator],
        color=simulator_colour,
        alpha=INNER_BAND_ALPHA,
    )
    band(
        axes,
        xs,
        [low for low, _ in corrected],
        [high for _, high in corrected],
        color=palette.model,
    )
    simulator_mean = [item.simulator_mean for item in predictions]
    corrected_mean = [item.corrected_mean for item in predictions]
    line(axes, xs, simulator_mean, color=simulator_colour, dashes=FITTED, width=1.6)
    line(axes, xs, corrected_mean, color=palette.model)
    labels = [
        (xs[-1], corrected_mean[-1], "bias-corrected", palette.model),
        (xs[-1], simulator_mean[-1], "simulator only", simulator_colour),
    ]
    if values is not None:
        axes.plot(
            xs,
            values,
            color=palette.accent,
            linewidth=REFERENCE_WIDTH,
            linestyle=REFERENCE,
            zorder=2,
        )
        labels.append((xs[-1], values[-1], "truth", palette.accent))
    points(axes, observed_x, calibration.observed, color=palette.observed, size=26.0)
    axes.set_xlim(min(xs[0], *observed_x), max(xs[-1], *observed_x))
    axis_labels(axes, input_name, name or calibration.quantity)
    direct_labels(axes, labels)
    if values is not None:
        covered = sum(
            low <= v <= high for (low, high), v in zip(corrected, values, strict=True)
        )
        plain = sum(
            low <= v <= high for (low, high), v in zip(simulator, values, strict=True)
        )
        title = (
            f"The bias-corrected {level:.0%} band covers the truth at {covered} of "
            f"{len(xs)} points; the simulator alone at {plain}"
        )
    else:
        shifts = [item.discrepancy_mean for item in predictions]
        worst = max(range(len(shifts)), key=lambda i: abs(shifts[i]))
        title = (
            f"The discrepancy shifts predictions by up to {shifts[worst]:+.3g} "
            f"(at {input_name} = {xs[worst]:.3g})"
        )
    title_block(
        axes,
        title,
        f"Dots: {len(observed_x)} observations · solid: eta + delta · dashed: "
        f"simulator eta · bands: {level:.0%} posterior predictive",
    )
    return axes


@themed
def plot_discrepancy_estimate(
    predictions: Sequence[DiscrepancyPrediction],
    input_name: str,
    *,
    truth: Sequence[float] | None = None,
    level: float = 0.95,
    ax: Any | None = None,
) -> Any:
    """Plot the posterior discrepancy ``delta(x)`` with its band and zero line.

    The band is the interval of the mixture ``S^-1 sum_s N(m_s, v_s)`` of the
    conditional discrepancy moments over parameter draws. The title reports
    the share of the grid where the band excludes zero, i.e. where the data
    show the simulator to be biased.

    Args:
        predictions: Records from ``DiscrepancyCalibration.predict`` at
            increasing values of ``input_name``.
        input_name: Condition name of the horizontal axis.
        truth: Optional true discrepancy aligned with ``predictions`` (truth
            minus the simulator at the true parameter).
        level: Central probability of the band.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: For fewer than two predictions, unordered or missing
            conditions, a misaligned truth, or an invalid level.

    Equation:
        [EQ-KOH-PRED-001], [EQ-VIZ-MIXQ-001].

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-BRYNJARSDOTTIR-OHAGAN-2014],
        [REF-MATPLOTLIB].

    """
    xs = _axis_values(predictions, input_name)
    values = _checked_truth(truth, len(xs))
    limits = [
        gaussian_mixture_interval(
            item.discrepancy_means, item.discrepancy_variances, level
        )
        for item in predictions
    ]
    means = [item.discrepancy_mean for item in predictions]
    axes = new_axes(ax)
    palette = active_palette()
    axes.axhline(0.0, color=palette.ink, linewidth=0.8, zorder=2)
    band(
        axes,
        xs,
        [low for low, _ in limits],
        [high for _, high in limits],
        color=palette.model,
    )
    line(axes, xs, means, color=palette.model)
    labels = [(xs[-1], means[-1], "posterior delta(x)", palette.model)]
    if values is not None:
        axes.plot(
            xs,
            values,
            color=palette.accent,
            linewidth=REFERENCE_WIDTH,
            linestyle=REFERENCE,
            zorder=2,
        )
        labels.append((xs[-1], values[-1], "true delta(x)", palette.accent))
    axes.set_xlim(xs[0], xs[-1])
    axis_labels(axes, input_name, "Discrepancy delta(x)")
    direct_labels(axes, labels)
    excluded = sum(low > 0.0 or high < 0.0 for low, high in limits)
    worst = max(range(len(means)), key=lambda i: abs(means[i]))
    if excluded == 0:
        title = (
            f"The {level:.0%} band contains zero everywhere: the data do not "
            "resolve a simulator bias"
        )
    else:
        title = (
            f"The {level:.0%} band excludes zero over {excluded / len(xs):.0%} of "
            f"the range; the bias peaks at {means[worst]:+.3g}"
        )
    subtitle = (
        f"Line and band: posterior mean and {level:.0%} interval of delta, "
        "including parameter uncertainty"
    )
    if values is not None:
        covered = sum(
            low <= v <= high for (low, high), v in zip(limits, values, strict=True)
        )
        subtitle += f" · covers the true delta at {covered} of {len(xs)} points"
    title_block(axes, title, subtitle)
    return axes
