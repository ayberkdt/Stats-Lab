"""Publication-grade chart theme with semantic color roles.

The theme encodes a small set of rules so every figure in the library reads as
one system:

* Every hue has one meaning. Observed or simulated data use ``series[0]``,
  model/fit/prediction curves use ``series[1]``, and the accent belongs to
  reference lines (thresholds, limits, exact values) only. Context data use
  muted ink instead of a new hue.
* An uncertainty band takes the hue of its parent series at low opacity and
  no stroke; the parent line is drawn on top of it, markers on top of lines.
* Dash patterns carry epistemic state (fitted, projected, simulated) and are
  never decorative.
* Titles are assertions; a muted subtitle says what the marks are. Direct
  labels at line ends replace legends whenever there is room.
* Horizontal hairline grid only; quiet axes; no glow, shadow, or 3-D effects.

Both palettes were checked with a categorical palette validator for lightness
band, chroma floor, colour-vision-deficiency separation between adjacent
slots, normal-vision separation, and contrast against the canvas.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class ChartPalette:
    """Colour tokens for one chart theme.

    Args:
        name: Palette identifier.
        canvas: Figure and axes background.
        ink: Primary text and strong strokes.
        muted: Secondary text, units, context series.
        rule: Gridlines and hairlines.
        accent: Reference lines and thresholds only.
        series: Six categorical data hues in fixed order.
        sequential: Light-to-dark single-hue ramp for magnitudes.

    References:
        [REF-MATPLOTLIB]. No mathematical equation is evaluated.

    """

    name: str
    canvas: str
    ink: str
    muted: str
    rule: str
    accent: str
    series: tuple[str, str, str, str, str, str]
    sequential: tuple[str, ...]

    @property
    def observed(self) -> str:
        """Return the hue reserved for observed or simulated data.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return self.series[0]

    @property
    def model(self) -> str:
        """Return the hue reserved for model, fit, or prediction curves.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return self.series[1]


PORCELAIN = ChartPalette(
    name="porcelain",
    canvas="#F6F4EF",
    ink="#23262B",
    muted="#656A73",
    rule="#DCD8CF",
    accent="#44599C",
    series=("#178F6E", "#B0642F", "#7D5BC4", "#8C8A1C", "#C04A8A", "#2F7FB5"),
    sequential=("#F6F4EF", "#CFE4DA", "#9BCAB6", "#5EAA8F", "#178F6E", "#0D5A45"),
)

GRAPHITE = ChartPalette(
    name="graphite",
    canvas="#141519",
    ink="#F1F0EB",
    muted="#A6A9B1",
    rule="#303339",
    accent="#9BA9F0",
    series=("#2A9E83", "#B67E2A", "#8A78CF", "#CC5F70", "#3F8BCB", "#6A9638"),
    sequential=("#1E2026", "#1F3F38", "#22604F", "#268266", "#2A9E83", "#7FD3BC"),
)

FONT_STACK = ("Source Sans 3", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")

TITLE_SIZE = 13.0
SUBTITLE_SIZE = 10.0
LABEL_SIZE = 9.5
TICK_SIZE = 9.0
ANNOTATION_SIZE = 8.5
SOURCE_SIZE = 7.5

LINE_WIDTH = 2.0
REFERENCE_WIDTH = 1.2
BAND_ALPHA = 0.16
# sunumatik chart-aesthetics: every uncertainty band at most 0.18 opacity, so
# a nested inner band is drawn at the cap over its outer band.
MAX_BAND_ALPHA = 0.18
INNER_BAND_ALPHA = MAX_BAND_ALPHA
AREA_TOP_ALPHA = 0.26

FITTED: tuple[float, tuple[float, ...]] = (0.0, (5.0, 3.5))
PROJECTED: tuple[float, tuple[float, ...]] = (0.0, (1.0, 3.5))
SIMULATED: tuple[float, tuple[float, ...]] = (0.0, (8.0, 3.0, 1.5, 3.0))
REFERENCE: tuple[float, tuple[float, ...]] = (0.0, (3.5, 3.0))



def _default_palette() -> ChartPalette:
    """Return the palette named by ``PROBSTAT_CHART_THEME`` (default light).

    ``graphite`` (or ``dark``) selects the dark palette; anything else keeps
    the light porcelain palette.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import os

    choice = os.environ.get("PROBSTAT_CHART_THEME", "porcelain").strip().lower()
    return GRAPHITE if choice in {"graphite", "dark"} else PORCELAIN


_ACTIVE: list[ChartPalette] = [_default_palette()]


MATPLOTLIB_MISSING = (
    "probstat figures need the optional Matplotlib dependency: install it with "
    "`pip install probstat-library[visualization]` (or `pip install -e "
    ".[visualization]` from a checkout). HTML study reports still work without "
    "it and omit the figures."
)


def active_palette() -> ChartPalette:
    """Return the palette of the innermost active theme context.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return _ACTIVE[-1]


def pyplot() -> Any:
    """Import and return Matplotlib's plotting interface on demand.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If the optional visualization dependency is missing.

    References:
        [REF-MATPLOTLIB]. This dependency adapter evaluates no equation.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(MATPLOTLIB_MISSING) from error
    return plt


def matplotlib_available() -> bool:
    """Return whether the optional Matplotlib dependency can be imported.

    References:
        [REF-MATPLOTLIB]. This dependency adapter evaluates no equation.

    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _installed_font_names() -> frozenset[str]:
    """Return the family names Matplotlib can resolve on this machine.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib import font_manager

    return frozenset(entry.name for entry in font_manager.fontManager.ttflist)


def available_font_stack() -> list[str]:
    """Return the preferred font families that are actually installed.

    Missing families are dropped so Matplotlib does not warn on every text
    object; ``DejaVu Sans`` (bundled with Matplotlib) is always last.

    Returns:
        Ordered list of installed family names.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    installed = _installed_font_names()
    stack = [name for name in FONT_STACK if name in installed]
    return stack if "DejaVu Sans" in stack else [*stack, "DejaVu Sans"]


def rc_parameters(palette: ChartPalette) -> dict[str, object]:
    """Return Matplotlib rcParams implementing the theme rules.

    Args:
        palette: Colour tokens.

    Returns:
        Mapping suitable for ``matplotlib.rc_context``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from cycler import cycler

    return {
        # Concrete names, not the generic "sans-serif": text resolves its
        # family at draw time, which may happen after the theme has exited.
        "font.family": available_font_stack(),
        "font.sans-serif": list(FONT_STACK),
        "font.size": TICK_SIZE,
        "text.color": palette.ink,
        "figure.facecolor": palette.canvas,
        "figure.figsize": (7.2, 4.3),
        "figure.dpi": 110,
        "savefig.facecolor": palette.canvas,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "axes.facecolor": palette.canvas,
        "axes.edgecolor": palette.muted,
        "axes.linewidth": 0.8,
        "axes.labelcolor": palette.ink,
        "axes.labelsize": LABEL_SIZE,
        "axes.labelweight": "semibold",
        "axes.labelpad": 6.0,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlecolor": palette.ink,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(palette.series)),
        "grid.color": palette.rule,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": palette.muted,
        "ytick.color": palette.muted,
        "xtick.labelcolor": palette.ink,
        "ytick.labelcolor": palette.ink,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "lines.linewidth": LINE_WIDTH,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
        "lines.markersize": 5.5,
        "patch.linewidth": 0.0,
        "legend.frameon": False,
        "legend.fontsize": ANNOTATION_SIZE,
        "legend.labelcolor": palette.ink,
        "legend.handlelength": 1.6,
        "image.cmap": "viridis",
        "mathtext.default": "regular",
    }


@contextmanager
def chart_theme(palette: ChartPalette = PORCELAIN) -> Iterator[ChartPalette]:
    """Activate the chart theme for the enclosed plotting code.

    Args:
        palette: Colour tokens; ``PORCELAIN`` (light) or ``GRAPHITE`` (dark).

    Yields:
        The active palette.

    Raises:
        ImportError: If Matplotlib is not installed (with install advice).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    try:
        import matplotlib
    except ImportError as error:
        raise ImportError(MATPLOTLIB_MISSING) from error

    _ACTIVE.append(palette)
    try:
        with matplotlib.rc_context(rc_parameters(palette)):
            yield palette
    finally:
        _ACTIVE.pop()


def themed(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a plotting function inside the active chart theme.

    Text and artists read rcParams when they are created, so every public
    plotting function is wrapped to create all artists inside the theme.

    Args:
        function: Plotting function.

    Returns:
        Wrapped function with identical signature.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call the wrapped plotting function under the active theme.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        with chart_theme(active_palette()):
            return function(*args, **kwargs)

    return wrapper


def new_axes(
    ax: Any | None = None, *, figsize: tuple[float, float] | None = None
) -> Any:
    """Return ``ax`` or a new themed axes.

    Args:
        ax: Existing axes to reuse.
        figsize: Optional figure size in inches for a new figure.

    Returns:
        Matplotlib axes styled with the active palette.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    plt = pyplot()
    if ax is not None:
        style_axes(ax)
        return ax
    _, axes = plt.subplots(figsize=figsize)
    style_axes(axes)
    return axes


def style_axes(ax: Any) -> None:
    """Apply quiet-axis rules to an axes created outside the theme.

    Args:
        ax: Matplotlib axes.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    ax.set_facecolor(palette.canvas)
    ax.figure.set_facecolor(palette.canvas)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette.muted)
        ax.spines[side].set_linewidth(0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=palette.muted, labelcolor=palette.ink)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=palette.rule, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def title_block(ax: Any, title: str, subtitle: str | None = None) -> None:
    """Draw a left-aligned assertion title with an optional muted subtitle.

    Args:
        ax: Target axes.
        title: Assertion stating what the chart shows.
        subtitle: One line naming the marks (not repeating axis titles).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    title = title[:1].upper() + title[1:]
    ax.set_title(
        title,
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=palette.ink,
        pad=24.0 if subtitle else 10.0,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 8.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=SUBTITLE_SIZE,
            color=palette.muted,
        )


def source_note(ax: Any, text: str) -> None:
    """Add a small muted provenance note under the plot, right-aligned.

    Args:
        ax: Target axes.
        text: Short source or method note.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    ax.annotate(
        text,
        xy=(1.0, 0.0),
        xycoords="axes fraction",
        xytext=(0.0, -34.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=SOURCE_SIZE,
        color=palette.muted,
    )


def axis_labels(
    ax: Any,
    x: str | None = None,
    y: str | None = None,
    *,
    x_unit: str | None = None,
    y_unit: str | None = None,
) -> None:
    """Set axis titles with optional units in brackets.

    Args:
        ax: Target axes.
        x: Horizontal axis title.
        y: Vertical axis title.
        x_unit: Optional horizontal unit.
        y_unit: Optional vertical unit.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if x is not None:
        ax.set_xlabel(f"{x} [{x_unit}]" if x_unit else x)
    if y is not None:
        ax.set_ylabel(f"{y} [{y_unit}]" if y_unit else y)


def band(
    ax: Any,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    color: str,
    alpha: float = BAND_ALPHA,
    label: str | None = None,
    smooth: bool = False,
) -> Any:
    """Draw an uncertainty band in its parent series hue with no stroke.

    The opacity is capped at ``MAX_BAND_ALPHA = 0.18`` (a band in a stronger
    fill reads as a third series). With ``smooth=True`` both edges use the
    same monotone cubic interpolation as a smoothed parent line.

    Args:
        ax: Target axes.
        x: Abscissae.
        lower: Lower band edge.
        upper: Upper band edge.
        color: Parent series colour.
        alpha: Fill opacity, capped at 0.18.
        label: Optional legend label.
        smooth: Interpolate the edges with :func:`monotone_cubic`.

    Returns:
        The Matplotlib collection.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if smooth:
        xs, lower = monotone_cubic(x, lower)
        _, upper = monotone_cubic(x, upper)
        x = xs
    return ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=min(alpha, MAX_BAND_ALPHA),
        linewidth=0.0,
        label=label,
        zorder=1,
    )


def monotone_cubic(
    x: Sequence[float], y: Sequence[float], *, samples: int = 16
) -> tuple[list[float], list[float]]:
    """Return a dense Fritsch-Carlson monotone cubic interpolant of ``(x, y)``.

    The curve passes through every point and never overshoots them: on each
    interval the Hermite slopes are limited so that monotone data stay
    monotone, which is why continuous phenomena can be drawn smooth without
    inventing extrema.

    Args:
        x: Strictly increasing abscissae (at least two).
        y: Ordinates.
        samples: Points per interval in the returned curve.

    Returns:
        Dense abscissae and ordinates, including the original points.

    Raises:
        ValueError: If the lengths differ, fewer than two points are given,
            ``x`` is not strictly increasing, or ``samples < 1``.

    References:
        [REF-FRITSCH-CARLSON-1980]. No equation of the registry is evaluated.

    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("monotone_cubic needs at least two points of equal length")
    if any(b <= a for a, b in zip(xs[:-1], xs[1:], strict=True)):
        raise ValueError("x must be strictly increasing")
    if samples < 1:
        raise ValueError("samples must be positive")
    n = len(xs)
    widths = [xs[i + 1] - xs[i] for i in range(n - 1)]
    secants = [(ys[i + 1] - ys[i]) / widths[i] for i in range(n - 1)]
    slopes = [0.0] * n
    slopes[0], slopes[-1] = secants[0], secants[-1]
    for i in range(1, n - 1):
        if secants[i - 1] * secants[i] <= 0.0:
            slopes[i] = 0.0
        else:
            slopes[i] = 0.5 * (secants[i - 1] + secants[i])
    for i, delta in enumerate(secants):
        if delta == 0.0:
            slopes[i] = slopes[i + 1] = 0.0
            continue
        a, b = slopes[i] / delta, slopes[i + 1] / delta
        radius = a * a + b * b
        if radius > 9.0:
            scale = 3.0 / radius**0.5
            slopes[i], slopes[i + 1] = scale * a * delta, scale * b * delta
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(n - 1):
        h = widths[i]
        for k in range(samples):
            t = k / samples
            h00 = (1 + 2 * t) * (1 - t) ** 2
            h10 = t * (1 - t) ** 2
            h01 = t * t * (3 - 2 * t)
            h11 = t * t * (t - 1)
            out_x.append(xs[i] + t * h)
            out_y.append(
                h00 * ys[i]
                + h10 * h * slopes[i]
                + h01 * ys[i + 1]
                + h11 * h * slopes[i + 1]
            )
    out_x.append(xs[-1])
    out_y.append(ys[-1])
    return out_x, out_y


def gradient_area(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str,
    top_alpha: float = AREA_TOP_ALPHA,
) -> Any:
    """Fill under the primary observed line with a fading gradient.

    The fill is the series hue at ``top_alpha`` (at most 0.26) along the line,
    fading to zero at the plot floor; use it for one series per chart only,
    since two stacked gradients turn into mud.

    Args:
        ax: Target axes (set the y-limits first; the floor is the lower limit).
        x: Abscissae of the line.
        y: Ordinates of the line.
        color: Series colour.
        top_alpha: Opacity at the line, capped at 0.26.

    Returns:
        The Matplotlib image that carries the gradient.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon

    xs, ys = [float(v) for v in x], [float(v) for v in y]
    floor = ax.get_ylim()[0]
    top = max(ys)
    red, green, blue = to_rgb(color)
    strength = min(top_alpha, AREA_TOP_ALPHA)
    steps = 64
    pixels = [
        [[red, green, blue, strength * (k / (steps - 1))]] for k in range(steps)
    ]
    image = ax.imshow(
        pixels,
        aspect="auto",
        extent=(min(xs), max(xs), floor, top),
        origin="lower",
        zorder=0.9,
        interpolation="bilinear",
    )
    outline = Polygon(
        [(xs[0], floor), *zip(xs, ys, strict=True), (xs[-1], floor)],
        closed=True,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(outline)
    image.set_clip_path(outline)
    ax.set_ylim(floor, ax.get_ylim()[1])
    return image


def line(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str,
    dashes: tuple[float, tuple[float, ...]] | None = None,
    width: float = LINE_WIDTH,
    label: str | None = None,
    zorder: float = 3.0,
    smooth: bool = False,
) -> Any:
    """Draw a data or model line with optional epistemic dash pattern.

    With ``smooth=True`` the line is drawn through :func:`monotone_cubic`
    (continuous phenomena are drawn smooth without overshoot; keep the
    default for discrete steps).

    Args:
        ax: Target axes.
        x: Abscissae.
        y: Ordinates.
        color: Series colour.
        dashes: ``FITTED``, ``PROJECTED``, ``SIMULATED`` or ``None`` (solid).
        width: Line width in points.
        label: Optional legend label.
        zorder: Drawing order.
        smooth: Interpolate with :func:`monotone_cubic`.

    Returns:
        The Matplotlib line.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if smooth:
        x, y = monotone_cubic(x, y)
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=width,
        linestyle=dashes if dashes is not None else "-",
        label=label,
        zorder=zorder,
    )[0]


def points(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str,
    size: float = 30.0,
    label: str | None = None,
    zorder: float = 4.0,
) -> Any:
    """Draw markers with a canvas-colour rim so they separate from lines.

    Args:
        ax: Target axes.
        x: Abscissae.
        y: Ordinates.
        color: Series colour.
        size: Marker area in points squared.
        label: Optional legend label.
        zorder: Drawing order.

    Returns:
        The Matplotlib path collection.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return ax.scatter(
        x,
        y,
        s=size,
        color=color,
        edgecolors=active_palette().canvas,
        linewidths=1.2,
        label=label,
        zorder=zorder,
    )


def reference_line(
    ax: Any,
    *,
    y: float | None = None,
    x: float | None = None,
    label: str | None = None,
    color: str | None = None,
) -> Any:
    """Draw a thin dashed reference line in the accent colour with a label.

    Horizontal lines are labelled at the right edge; vertical lines at the top.

    Args:
        ax: Target axes.
        y: Ordinate of a horizontal reference.
        x: Abscissa of a vertical reference.
        label: Short label such as ``"exact value"``.
        color: Override colour (defaults to the accent).

    Returns:
        The Matplotlib line.

    Raises:
        ValueError: If neither or both of ``x`` and ``y`` are given.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if (x is None) == (y is None):
        raise ValueError("supply exactly one of x or y")
    colour = color or active_palette().accent
    if y is not None:
        artist = ax.axhline(
            y, color=colour, linewidth=REFERENCE_WIDTH, linestyle=REFERENCE, zorder=2
        )
        if label:
            ax.annotate(
                label,
                xy=(1.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(4.0, 0.0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color=colour,
                fontweight="semibold",
                annotation_clip=False,
            )
        return artist
    artist = ax.axvline(
        x, color=colour, linewidth=REFERENCE_WIDTH, linestyle=REFERENCE, zorder=2
    )
    if label:
        ax.annotate(
            label,
            xy=(x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4.0, -2.0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE,
            color=colour,
            fontweight="semibold",
            annotation_clip=False,
        )
    return artist


def direct_labels(
    ax: Any,
    items: Sequence[tuple[float, float, str, str]],
    *,
    gap_points: float = 11.0,
) -> None:
    """Place series labels at line ends and push colliding labels apart.

    Args:
        ax: Target axes; call after limits are final.
        items: ``(x_end, y_end, text, colour)`` per series.
        gap_points: Minimum vertical distance between labels in points.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if not items:
        return
    # Force autoscaling first: transData is stale until limits are computed.
    ax.get_xlim()
    ax.get_ylim()
    figure = ax.figure
    to_display = ax.transData.transform
    scale = figure.dpi / 72.0
    placed = sorted(
        ((to_display((x, y))[1], x, y, text, colour) for x, y, text, colour in items),
        key=lambda entry: entry[0],
    )
    positions = [entry[0] for entry in placed]
    minimum = gap_points * scale
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < minimum:
            positions[index] = positions[index - 1] + minimum
    overflow = positions[-1] - to_display((0.0, ax.get_ylim()[1]))[1]
    if overflow > 0.0:
        positions = [value - overflow for value in positions]
        for index in range(len(positions) - 2, -1, -1):
            if positions[index + 1] - positions[index] < minimum:
                positions[index] = positions[index + 1] - minimum
    for (display_y, x, y, text, colour), target in zip(placed, positions, strict=True):
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(8.0, (target - display_y) / scale),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=colour,
            annotation_clip=False,
        )


def note(
    ax: Any,
    text: str,
    *,
    xy: tuple[float, float],
    offset: tuple[float, float] = (24.0, 18.0),
) -> None:
    """Add a muted annotation with a thin curved arrow to an exact point.

    Use at most two per chart; more means the chart is carrying prose.

    Args:
        ax: Target axes.
        text: Short annotation.
        xy: Data coordinates of the annotated point.
        offset: Text offset in points.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        color=palette.muted,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
            "color": palette.muted,
            "linewidth": 0.8,
            "shrinkA": 2.0,
            "shrinkB": 5.0,
            "connectionstyle": "arc3,rad=0.12",
        },
    )


def sequential_colormap(palette: ChartPalette | None = None) -> Any:
    """Return a single-hue light-to-dark colormap for magnitudes.

    Args:
        palette: Palette whose sequential ramp is used (active by default).

    Returns:
        Matplotlib ``LinearSegmentedColormap``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import LinearSegmentedColormap

    ramp = (palette or active_palette()).sequential
    return LinearSegmentedColormap.from_list("probstat-sequential", list(ramp))


def save_figure(figure: Any, path: str | Path, *, dpi: int = 200) -> Path:
    """Save a figure with the theme canvas, tight bounds, and padding.

    Args:
        figure: Matplotlib figure.
        path: Destination file path; parent directories are created.
        dpi: Output resolution.

    Returns:
        The written path.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.25,
    )
    return destination
