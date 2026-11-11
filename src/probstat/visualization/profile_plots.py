"""Performance-profile figure for benchmark comparisons (ADR-0077).

A performance profile is the empirical distribution function of a solver's
performance ratios, so it is drawn as a right-continuous step line on a
``log2`` ratio axis with the share of problems on a fixed ``[0, 1]`` axis.
Colour roles (ADR-0024): the (at most four) solvers with the largest area
under their profile take the categorical hues ``series[0]``, ``series[3]``,
``series[4]``, and ``series[5]`` and are labelled at the line end; solvers
with identical ratios share one line and one label; every other solver is
a thin muted context line without a label. Labels are set at the label
size and moved toward the ink until they reach a 4.5:1 contrast with the
canvas, keeping their hue.
"""

from __future__ import annotations

import math
from typing import Any

from probstat.numerics import PerformanceProfile

from ._layout import wrapped_title_block
from .contingency_plots import _relative_luminance
from .theme import (
    LABEL_SIZE,
    LINE_WIDTH,
    active_palette,
    axis_labels,
    direct_labels,
    new_axes,
    themed,
)

_HIGHLIGHT_SLOTS = (0, 3, 4, 5)
_LABEL_CONTRAST = 4.5
_WIDTH_FACTORS = (2.2, 1.5, 0.95, 0.55)


def _legible(colour: str) -> str:
    """Return ``colour`` mixed toward the ink until it reads as small text.

    The hue is kept; the colour moves toward the palette ink in steps of 5 %
    until its contrast ratio against the canvas reaches 4.5:1, the level for
    normal-size text, so a direct label stays legible on both palettes.

    Args:
        colour: Series colour of a line.

    Returns:
        Hex colour for the line's label.

    References:
        [REF-MATPLOTLIB]. No registered equation (display colour arithmetic).

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = active_palette()
    canvas = _relative_luminance(to_rgb(palette.canvas))
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        weight = step / 20.0
        mixed = (
            (1.0 - weight) * base[0] + weight * ink[0],
            (1.0 - weight) * base[1] + weight * ink[1],
            (1.0 - weight) * base[2] + weight * ink[2],
        )
        luminance = _relative_luminance(mixed)
        high, low = max(luminance, canvas), min(luminance, canvas)
        if (high + 0.05) / (low + 0.05) >= _LABEL_CONTRAST:
            return str(to_hex(mixed))
    return palette.ink


def _step_points(
    profile: PerformanceProfile, solver: str, upper: float
) -> tuple[list[float], list[float]]:
    """Return the vertices of one solver's step profile on ``[1, upper]``.

    Args:
        profile: Performance profiles.
        solver: Solver name.
        upper: Right end of the ratio axis.

    Returns:
        Abscissae and ordinates for a ``where="post"`` step line.

    References:
        [EQ-NUM-PROFILE-002], [REF-DOLAN-MORE-2002, Section 2].

    """
    xs = [1.0]
    ys = [profile.value(solver, 1.0)]
    for tau, share in profile.breakpoints(solver):
        if 1.0 < tau <= upper:
            xs.append(tau)
            ys.append(share)
    xs.append(upper)
    ys.append(ys[-1])
    return xs, ys


@themed
def plot_performance_profile(
    profile: PerformanceProfile,
    *,
    highlight: int = 4,
    tau_max: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot Dolan-Moré performance profiles ``rho_s(tau)`` on a ``log2`` axis.

    Args:
        profile: Output of :func:`probstat.numerics.performance_profile`.
        highlight: Number of distinct leading profiles (by area) drawn in
            colour with direct labels, between 1 and 4.
        tau_max: Right end of the ratio axis (default: twice the largest
            finite ratio, at least 4).
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The axes.

    Raises:
        TypeError: If ``profile`` is not a :class:`PerformanceProfile`.
        ValueError: If ``highlight`` is outside ``[1, 4]`` or ``tau_max`` is
            below 2 or not finite.

    Equation:
        [EQ-NUM-PROFILE-002], [EQ-NUM-PROFILE-003].

    References:
        [REF-DOLAN-MORE-2002, Section 2], [REF-MATPLOTLIB].

    """
    if not isinstance(profile, PerformanceProfile):
        raise TypeError("profile must be a PerformanceProfile")
    if isinstance(highlight, bool) or not 1 <= highlight <= len(_HIGHLIGHT_SLOTS):
        raise ValueError("highlight must lie in [1, 4]")
    upper = max(4.0, 2.0 * profile.max_ratio) if tau_max is None else float(tau_max)
    if not math.isfinite(upper) or upper < 2.0:
        raise ValueError("tau_max must be finite and at least 2")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    palette = active_palette()
    axes = new_axes(ax, figsize=(7.2, 4.6))
    ranked = sorted(
        profile.solvers,
        key=lambda s: (-profile.area(s), -profile.wins(s), profile.solvers.index(s)),
    )
    groups: list[list[str]] = []
    rows = {s: profile.ratios[profile.solvers.index(s)] for s in profile.solvers}
    for solver in ranked:
        for group in groups:
            if rows[group[0]] == rows[solver]:
                group.append(solver)
                break
        else:
            groups.append([solver])
    leading = [g for g in groups if profile.solved_fraction(g[0]) > 0.0][:highlight]
    context = [g for g in groups if g not in leading]
    for group in context:
        xs, ys = _step_points(profile, group[0], upper)
        axes.step(xs, ys, where="post", color=palette.muted, linewidth=0.9,
                  alpha=0.55, zorder=2)  # fmt: skip
    labels: list[tuple[float, float, str, str]] = []
    for index, group in enumerate(leading):
        colour = palette.series[_HIGHLIGHT_SLOTS[index]]
        xs, ys = _step_points(profile, group[0], upper)
        # Leading profiles are wider and lower, so a coinciding segment of a
        # later profile shows as a stripe inside the earlier one.
        axes.step(xs, ys, where="post", color=colour,
                  linewidth=LINE_WIDTH * _WIDTH_FACTORS[index],
                  zorder=3 + 0.1 * index)  # fmt: skip
        name = group[0] + (f" (+{len(group) - 1} tied)" if len(group) > 1 else "")
        labels.append((upper, ys[-1], name, _legible(colour)))
    axes.set_xscale("log", base=2)
    ticks = [2.0**k for k in range(0, math.floor(math.log2(upper)) + 1)]
    axes.xaxis.set_major_locator(FixedLocator(ticks))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_xlim(1.0, upper)
    axes.set_ylim(0.0, 1.04)
    total = profile.problems
    if total <= 10:
        axes.set_yticks([k / total for k in range(total + 1)])
        axes.set_yticklabels([f"{k}/{total}" for k in range(total + 1)])
    figure = axes.figure
    if ax is None:
        widest = max((len(label[2]) for label in labels), default=0)
        reserve = (widest * 0.62 * LABEL_SIZE / 72.0 + 0.35) / figure.get_figwidth()
        figure.subplots_adjust(left=0.11, right=max(0.55, 1.0 - reserve), top=0.80)
    before = len(axes.texts)
    direct_labels(axes, labels, gap_points=1.6 * LABEL_SIZE)
    for text in axes.texts[before:]:
        text.set_fontsize(LABEL_SIZE)
    axis_labels(
        axes,
        "Performance ratio tau = cost / cheapest accurate cost (log2 scale)",
        "Share of problems solved",
    )
    if leading:
        best = leading[0]
        title = (
            f"{best[0]} has the largest profile area ({profile.area(best[0]):.2f})"
        )
    else:
        title = "No method solves any problem"
    ties = [" and ".join(group) + " tie" for group in leading if len(group) > 1]
    left = axes.get_position().x0
    wrapped_title_block(
        axes,
        title,
        (
            "Share of problems a method solves within a factor tau of the "
            "cheapest accurate method; area 1 = cheapest everywhere",
            *ties,
            f"{len(context)} other method(s) in grey" if context else "",
        ),
        width_inches=figure.get_figwidth() * (1.0 - left) - 0.2,
    )
    return axes
