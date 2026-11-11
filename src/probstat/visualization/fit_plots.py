"""Fit diagnostics: hanging rootograms for counts and Q-Q plots for samples."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from probstat.foundations import coerce_finite_sample
from probstat.inference import qq_points

from .theme import (
    FITTED,
    active_palette,
    axis_labels,
    direct_labels,
    line,
    new_axes,
    note,
    points,
    reference_line,
    source_note,
    themed,
    title_block,
)


def hanging_rootogram_bars(
    observed: Sequence[float], expected: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """Return ``(bottom, top)`` of each hanging-rootogram bar.

    The bar for count ``k`` hangs from the fitted curve ``sqrt(E_k)`` and is
    ``sqrt(O_k)`` long, so its bottom ``sqrt(E_k) - sqrt(O_k)`` sits on zero
    when the fit is exact; departures are read against a horizontal line.

    Args:
        observed: Non-negative observed frequencies.
        expected: Non-negative fitted frequencies of the same length.

    Returns:
        One ``(bottom, top)`` pair per count.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the lengths differ, the input is empty, or a value is
            negative or non-finite.

    Equation:
        [EQ-VIZ-ROOT-001].

    References:
        [REF-KLEIBER-ZEILEIS-2016].

    """
    observed_values = coerce_finite_sample(observed, name="observed")
    expected_values = coerce_finite_sample(expected, name="expected")
    if len(observed_values) != len(expected_values):
        raise ValueError("observed and expected must have the same length")
    if min(observed_values) < 0.0 or min(expected_values) < 0.0:
        raise ValueError("frequencies must be non-negative")
    return tuple(
        (math.sqrt(e) - math.sqrt(o), math.sqrt(e))
        for o, e in zip(observed_values, expected_values, strict=True)
    )


@themed
def plot_rootogram(
    observed: Sequence[float],
    expected: Sequence[float],
    *,
    start: int = 0,
    count_label: str = "Count",
    model_label: str = "fitted model",
    source: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw a hanging rootogram of observed against fitted count frequencies.

    Args:
        observed: Observed frequencies of the counts ``start, start + 1, ...``.
        expected: Fitted frequencies for the same counts.
        start: Count of the first entry.
        count_label: Horizontal axis title.
        model_label: Name of the fitted law used in the label and title.
        source: Optional provenance note under the plot.
        ax: Optional Matplotlib axes.

    Returns:
        Axes with the hanging bars, the fitted root curve, and the zero line.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the frequencies are invalid.

    Equation:
        [EQ-VIZ-ROOT-001].

    References:
        [REF-KLEIBER-ZEILEIS-2016], [REF-MATPLOTLIB].

    """
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("start must be an integer")
    bars = hanging_rootogram_bars(observed, expected)
    xs = tuple(float(start + index) for index in range(len(bars)))
    axes = new_axes(ax)
    palette = active_palette()
    axes.bar(
        xs,
        tuple(top - bottom for bottom, top in bars),
        bottom=tuple(bottom for bottom, _ in bars),
        width=0.72,
        color=palette.sequential[2],
        edgecolor=palette.observed,
        linewidth=1.0,
        zorder=2,
    )
    tops = tuple(top for _, top in bars)
    line(axes, xs, tops, color=palette.model, dashes=FITTED, zorder=4)
    points(axes, xs, tops, color=palette.model, size=22.0, zorder=5)
    reference_line(axes, y=0.0)
    gaps = tuple(abs(bottom) for bottom, _ in bars)
    worst = max(range(len(gaps)), key=gaps.__getitem__)
    lowest = min(min(bottom for bottom, _ in bars), -1.0)
    axes.set_ylim(lowest - 0.6, max(tops) * 1.08)
    axes.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    axes.set_xticks(xs)
    axis_labels(axes, count_label, "Square root of frequency")
    title_block(
        axes,
        f"{model_label}: every bar ends within {gaps[worst]:.2f} of zero",
        "Bars: sqrt(observed) hanging from the dashed curve sqrt(expected)",
    )
    if source:
        source_note(axes, source)
    return axes


@themed
def plot_qq(
    values: Iterable[float],
    distribution: Any,
    *,
    model_label: str | None = None,
    value_label: str = "Observed quantile",
    source: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw a Q-Q plot of a sample against a fitted law with Hazen positions.

    Args:
        values: Finite observations.
        distribution: Law exposing ``quantile(p)``.
        model_label: Name of the fitted law (class name by default).
        value_label: Vertical axis title.
        source: Optional provenance note under the plot.
        ax: Optional Matplotlib axes.

    Returns:
        Axes with the quantile pairs, the identity line, and a note on the
        largest departure.

    Raises:
        TypeError: If the law lacks ``quantile`` or a value is not real.
        ValueError: If the sample is empty or not finite.

    Equation:
        [EQ-INF-GOF-QQ-001].

    References:
        [REF-WILK-GNANADESIKAN-1968], [REF-MATPLOTLIB].

    """
    pairs = qq_points(values, distribution)
    name = model_label or type(distribution).__name__
    theoretical = tuple(pair[0] for pair in pairs)
    empirical = tuple(pair[1] for pair in pairs)
    axes = new_axes(ax)
    palette = active_palette()
    low = min(min(theoretical), min(empirical))
    high = max(max(theoretical), max(empirical))
    pad = 0.05 * (high - low) if high > low else 1.0
    line(
        axes,
        (low - pad, high + pad),
        (low - pad, high + pad),
        color=palette.model,
        dashes=FITTED,
        zorder=2,
    )
    points(axes, theoretical, empirical, color=palette.observed, size=26.0)
    worst = max(range(len(pairs)), key=lambda i: abs(empirical[i] - theoretical[i]))
    axes.set_xlim(low - pad, high + pad)
    axes.set_ylim(low - pad, high + pad)
    axis_labels(axes, f"{name[:1].upper()}{name[1:]} quantile", value_label)
    direct_labels(axes, ((high + pad, high + pad, "y = x", palette.model),))
    gap = empirical[worst] - theoretical[worst]
    note(
        axes,
        f"observed {empirical[worst]:.3g}, fitted {theoretical[worst]:.3g}",
        xy=(theoretical[worst], empirical[worst]),
        offset=(30.0, 16.0) if gap < 0.0 else (30.0, -16.0),
    )
    title_block(
        axes,
        f"The largest departure from the fitted {name} is {abs(gap):.3g}",
        f"Points: ordered data against {name} quantiles at (i - 1/2)/n",
    )
    if source:
        source_note(axes, source)
    return axes
