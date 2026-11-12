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
# Smallest text on any figure: 9 pt, the floor for figures read on screen
# (ADR-0083); direct labels and source notes use it.
ANNOTATION_SIZE = 9.0
SOURCE_SIZE = 9.0
CANVAS_MARGIN_INCHES = 0.08

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

# Construction lines of a geometric picture (the beta circle of FORM): fine
# dots, distinct from the epistemic patterns above.
GUIDE: tuple[float, tuple[float, ...]] = (0.0, (1.0, 2.5))

# Line widths by role (ADR-0098): every figure of the set uses these four, so
# a line of the same role has the same weight in every figure. Data and model
# lines use ``LINE_WIDTH``; context lines (comparison designs, a reference
# sample, a truth path under its estimate) ``CONTEXT_WIDTH``; many
# overlapping paths (MCMC traces, spaghetti, realizations) ``PATH_WIDTH``;
# reference lines ``REFERENCE_WIDTH``.
CONTEXT_WIDTH = 1.4
PATH_WIDTH = 0.7



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


# --------------------------------------------------------------------------
# The figure-set contract (ADR-0098)
#
# A single figure can follow every rule above and the set can still
# contradict itself: "truth" drawn in the accent in one figure and in muted
# ink in the next, "Failure probability Pf" in one and "Pf estimate" in
# another. The tables below fix the encoding of every recurring entity and
# the name and unit of every recurring quantity; the plotting functions and
# the labs read them instead of choosing hues and axis titles locally.
# ``docs/figures/stats-lab.seti.json`` (light) and
# ``docs/figures/stats-lab-dark.seti.json`` (dark) are the same tables
# written in the datawiz set format, and ``tests/test_figure_set_contract.py``
# keeps both in step.
# --------------------------------------------------------------------------

ROLES = (
    "observed",
    "model",
    "reference",
    "context",
    "series-3",
    "series-4",
    "series-5",
    "series-6",
)
"""Colour roles of ADR-0024: the first four carry meaning, the last four are
the remaining categorical slots for members of one family (chains,
forecasters) that have no observed/model meaning."""


def role_colour(role: str, palette: ChartPalette | None = None) -> str:
    """Return the colour of a semantic role in a palette.

    Args:
        role: One of :data:`ROLES`.
        palette: Palette to read (the active one by default).

    Returns:
        Hexadecimal colour.

    Raises:
        ValueError: If ``role`` is not a known role.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    tokens = palette or active_palette()
    if role == "observed":
        return tokens.observed
    if role == "model":
        return tokens.model
    if role == "reference":
        return tokens.accent
    if role == "context":
        return tokens.muted
    if role in ROLES:
        return tokens.series[int(role.rsplit("-", 1)[1]) - 1]
    raise ValueError(f"unknown colour role {role!r}; expected one of {ROLES}")


@dataclass(frozen=True, slots=True)
class EntityStyle:
    """Fixed encoding of one recurring entity in every figure.

    Args:
        role: Colour role (:data:`ROLES`).
        marker: Matplotlib marker code used when the entity is drawn as points.
        dashes: ``None`` (solid), ``FITTED``, ``PROJECTED``, ``SIMULATED`` or
            ``REFERENCE``.
        width: Line width in points.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """

    role: str
    marker: str = "o"
    dashes: tuple[float, tuple[float, ...]] | None = None
    width: float = LINE_WIDTH

    def colour(self, palette: ChartPalette | None = None) -> str:
        """Return the entity colour in ``palette`` (the active one by default).

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return role_colour(self.role, palette)


_REFERENCE_ENTITY = EntityStyle("reference", "s", REFERENCE, REFERENCE_WIDTH)
_FIT = EntityStyle("model", "s", FITTED)

ENTITY_STYLES: dict[str, EntityStyle] = {
    # Data: observed or simulated values.
    "observations": EntityStyle("observed", "o"),
    "simulated": EntityStyle("observed", "o"),
    "empirical": EntityStyle("observed", "o"),
    "first-order index": EntityStyle("observed", "o"),
    # Model results: fits, posteriors, filters, surrogates, linearizations.
    "total-effect index": EntityStyle("model", "D"),
    "exact posterior": EntityStyle("model", "s"),
    "posterior mean": EntityStyle("model", "s"),
    "filtered mean": EntityStyle("model", "s"),
    "bias-corrected": EntityStyle("model", "s"),
    "forecast mean": EntityStyle("model", "s", PROJECTED),
    "AK-MCS": EntityStyle("model", "s"),
    "FORM": EntityStyle("model", "D", FITTED, CONTEXT_WIDTH),
    "GEV fit": _FIT,
    "GPD fit": _FIT,
    "Gamma fit": _FIT,
    "LogNormal fit": _FIT,
    "Weibull fit": _FIT,
    # Slot 4 (olive, rose in the dark palette) stays far from the accent of
    # the truth line it is drawn beside in both palettes.
    "smoothed": EntityStyle("series-4", "^", FITTED),
    "design point": EntityStyle("model", "o"),
    # Standard-normal maps (FORM geometry, design points, AK-MCS): failed
    # samples are data, the exact limit state is a reference.
    "failed samples": EntityStyle("observed", "o"),
    "limit state": EntityStyle("reference", "s", None, 1.4 * REFERENCE_WIDTH),
    # Reference values: exact, true, or limiting quantities.
    "truth": _REFERENCE_ENTITY,
    "closed form": _REFERENCE_ENTITY,
    "Berry-Esseen bound": _REFERENCE_ENTITY,
    "Monte Carlo reference": _REFERENCE_ENTITY,
    "threshold": _REFERENCE_ENTITY,
    "perfect reliability": _REFERENCE_ENTITY,
    # Context: what the highlighted result is compared with.
    "uniform design": EntityStyle("context", "o", None, CONTEXT_WIDTH),
    "simulator only": EntityStyle("context", "o", FITTED, CONTEXT_WIDTH),
    "calibration": EntityStyle("context", "o", None, CONTEXT_WIDTH),
    "safe samples": EntityStyle("context", "o"),
    "beta circle": EntityStyle("context", "o", GUIDE, CONTEXT_WIDTH),
    # Members of one family without observed/model meaning.
    "chain 1": EntityStyle("series-3", "o", None, PATH_WIDTH),
    "chain 2": EntityStyle("series-4", "o", None, PATH_WIDTH),
    "chain 3": EntityStyle("series-5", "o", None, PATH_WIDTH),
    "chain 4": EntityStyle("series-6", "o", None, PATH_WIDTH),
    "ideal": EntityStyle("observed", "o", None, CONTEXT_WIDTH),
    "ensemble": EntityStyle("model", "s", None, CONTEXT_WIDTH),
    "overconfident": EntityStyle("series-4", "^", None, CONTEXT_WIDTH),
    "climatology": EntityStyle("series-5", "D", None, CONTEXT_WIDTH),
}
"""Entity name (as written in the figure) to its fixed encoding."""


def entity_style(name: str) -> EntityStyle:
    """Return the fixed encoding of a recurring entity.

    A trailing parenthesized qualifier is ignored, so ``"calibration (n =
    300)"`` is drawn like ``"calibration"``.

    Args:
        name: Entity name as written in the figure.

    Returns:
        The entity's :class:`EntityStyle`.

    Raises:
        ValueError: If the entity is not part of the set contract.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    key = name if name in ENTITY_STYLES else name.split(" (", 1)[0].strip()
    try:
        return ENTITY_STYLES[key]
    except KeyError:
        raise ValueError(
            f"{name!r} is not an entity of the figure-set contract (ADR-0098)"
        ) from None


def entity_colour(name: str, palette: ChartPalette | None = None) -> str:
    """Return the colour of a recurring entity in ``palette``.

    Args:
        name: Entity name (see :func:`entity_style`).
        palette: Palette to read (the active one by default).

    Returns:
        Hexadecimal colour.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return entity_style(name).colour(palette)


@dataclass(frozen=True, slots=True)
class Quantity:
    """Name, unit, and axis role of a quantity shared by several figures.

    Args:
        name: Axis title in sentence case, without symbols in parentheses.
        unit: Unit written in brackets after the name, or ``None``.
        axis: ``"x"``, ``"y"``, or ``"free"`` when the quantity is the value
            axis of a dot plot (horizontal) in some figures and of a curve
            (vertical) in others.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """

    name: str
    unit: str | None = None
    axis: str = "x"

    @property
    def label(self) -> str:
        """Return the axis title, ``"Name [unit]"`` or ``"Name"``.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return f"{self.name} [{self.unit}]" if self.unit else self.name


QUANTITIES: dict[str, Quantity] = {
    "count": Quantity("Count", axis="y"),
    "cumulative_probability": Quantity("Cumulative probability", axis="y"),
    "density": Quantity("Density", axis="y"),
    "empirical_coverage": Quantity("Empirical coverage", axis="y"),
    "estimate": Quantity("Estimate", axis="y"),
    "exceedance_probability": Quantity("Exceedance probability", axis="y"),
    "failure_probability": Quantity("Failure probability", axis="free"),
    "kolmogorov_distance": Quantity("Kolmogorov distance", axis="y"),
    "model_evaluations": Quantity("Number of model evaluations", axis="x"),
    "nominal_coverage": Quantity("Nominal coverage", axis="x"),
    "pit_value": Quantity("PIT value", axis="x"),
    "posterior_density": Quantity("Posterior density", axis="y"),
    "probability_mass": Quantity("Probability mass", axis="y"),
    "sample_size": Quantity("Sample size N", axis="x"),
    "smallest_learning_value": Quantity("Smallest learning value U", axis="y"),
    "time": Quantity("Time", axis="x"),
    # A calibrated parameter is the vertical axis of its MCMC trace and the
    # horizontal axis of its posterior histogram.
    "decay_rate": Quantity("k", axis="free"),
    "variance_share": Quantity("Share of output variance", axis="x"),
}
"""Recurring quantities: one name, one unit spelling, one axis role."""

METHOD_NAMES: dict[str, str] = {
    "MC": "Monte Carlo",
    "IS": "Importance sampling",
    "CE-IS": "Cross-entropy importance sampling",
    "Subset": "Subset simulation",
    "Line": "Line sampling",
}
"""Reliability method codes (``reliability.compare_reliability_methods``) to
the names written in figures; FORM and SORM keep their acronyms."""

CATEGORY_ORDERS: dict[str, tuple[str, ...]] = {
    "forecasters": ("ideal", "ensemble", "overconfident", "climatology"),
    "chains": ("chain 1", "chain 2", "chain 3", "chain 4"),
    "reliability methods": (
        "FORM",
        "SORM-Breitung",
        "SORM-HR",
        "Importance sampling",
        "Cross-entropy importance sampling",
        "Subset simulation",
        "Line sampling",
        "Monte Carlo",
    ),
    "sampling designs": (
        "Monte Carlo",
        "Latin hypercube",
        "Halton (RQMC)",
        "Sobol (RQMC)",
    ),
}
"""Families whose members appear in a fixed order in every figure."""


def axis_title(key: str) -> str:
    """Return the axis title of a recurring quantity.

    Args:
        key: Key of :data:`QUANTITIES`.

    Returns:
        ``"Name [unit]"`` or ``"Name"``.

    Raises:
        ValueError: If the quantity is not part of the set contract.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    try:
        return QUANTITIES[key].label
    except KeyError:
        raise ValueError(
            f"{key!r} is not a quantity of the figure-set contract (ADR-0098)"
        ) from None


_SYMBOL_WORDS = frozenset(
    {
        "alpha", "beta", "gamma", "delta", "eps", "epsilon", "theta", "kappa",
        "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "chi",
        "psi", "omega", "log",
    }
)  # fmt: skip


def sentence_case(text: str) -> str:
    """Capitalize the first word of an axis title unless it is a symbol.

    Variable names such as ``deflection`` become ``Deflection``; symbols
    (``x``, ``u_load``, ``eps = 0.5``, Greek names) and words that already
    carry a capital are left as they are.

    Args:
        text: Axis title.

    Returns:
        The title in sentence case.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    head, _, _ = text.partition(" ")
    letters = head.replace("-", "")
    if (
        len(letters) >= 2
        and letters.isalpha()
        and letters.islower()
        and letters.lower() not in _SYMBOL_WORDS
    ):
        return text[:1].upper() + text[1:]
    return text


def share_limits(axes: Sequence[Any], *, axis: str = "y") -> tuple[float, float]:
    """Give compared axes (in one or several figures) one common range.

    Figures that are read against each other (two forecasters' PIT
    histograms, six encodings of one forecast) need a common scale, or equal
    values look unequal (ADR-0098). The union of the current ranges is set
    on every axes.

    Args:
        axes: Axes to align.
        axis: ``"x"`` or ``"y"``.

    Returns:
        The common ``(low, high)`` range.

    Raises:
        ValueError: If ``axes`` is empty or ``axis`` is not ``"x"`` or ``"y"``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not axes:
        raise ValueError("at least one axes is required")
    ranges = [a.get_xlim() if axis == "x" else a.get_ylim() for a in axes]
    low = min(min(r) for r in ranges)
    high = max(max(r) for r in ranges)
    for a in axes:
        (a.set_xlim if axis == "x" else a.set_ylim)(low, high)
    return float(low), float(high)


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

    # Matplotlib >= 3.10 types rc_context keys as a Literal of known rcParams;
    # the mapping is built from those names, so it is passed through as Any.
    parameters: Any = rc_parameters(palette)
    _ACTIVE.append(palette)
    try:
        with matplotlib.rc_context(parameters):
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
            The first letter of the title is capitalized unless the first word
            is an identifier (it contains a digit or an underscore, such as an
            input named ``x3`` or ``igniter_p``), which keeps the spelling the
            axis labels use.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    first_word = title.split(" ", 1)[0]
    if not any(character.isdigit() or character == "_" for character in first_word):
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
    """Set axis titles in sentence case with optional units in brackets.

    Every figure of the set writes a unit as ``"Name [unit]"`` (ADR-0098);
    parentheses are kept for function notation in subtitles, never for units.
    A recurring quantity is passed by its title from :func:`axis_title`.

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
        ax.set_xlabel(sentence_case(f"{x} [{x_unit}]" if x_unit else x))
    if y is not None:
        ax.set_ylabel(sentence_case(f"{y} [{y_unit}]" if y_unit else y))


def band(
    ax: Any,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    color: str | None = None,
    alpha: float = BAND_ALPHA,
    label: str | None = None,
    smooth: bool = False,
    entity: str | None = None,
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
        color: Parent series colour (or give ``entity``).
        alpha: Fill opacity, capped at 0.18.
        label: Optional legend label.
        smooth: Interpolate the edges with :func:`monotone_cubic`.
        entity: Parent entity of the set contract; sets the colour.

    Returns:
        The Matplotlib collection.

    Raises:
        ValueError: If neither ``color`` nor ``entity`` is given.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if entity is not None:
        color = entity_colour(entity)
    if color is None:
        raise ValueError("band needs a color or an entity")
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
    color: str | None = None,
    dashes: tuple[float, tuple[float, ...]] | None = None,
    width: float | None = None,
    label: str | None = None,
    zorder: float = 3.0,
    smooth: bool = False,
    entity: str | None = None,
) -> Any:
    """Draw a data or model line with optional epistemic dash pattern.

    With ``smooth=True`` the line is drawn through :func:`monotone_cubic`
    (continuous phenomena are drawn smooth without overshoot; keep the
    default for discrete steps). With ``entity`` the colour, dash pattern,
    width, and label come from the set contract (:data:`ENTITY_STYLES`), so
    the entity looks the same in every figure; ``label`` may then add a
    qualifier such as ``"calibration (n = 300)"``.

    Args:
        ax: Target axes.
        x: Abscissae.
        y: Ordinates.
        color: Series colour (or give ``entity``).
        dashes: ``FITTED``, ``PROJECTED``, ``SIMULATED`` or ``None`` (solid).
        width: Line width in points (default ``LINE_WIDTH``).
        label: Optional legend label.
        zorder: Drawing order.
        smooth: Interpolate with :func:`monotone_cubic`.
        entity: Entity of the set contract.

    Returns:
        The Matplotlib line.

    Raises:
        ValueError: If neither ``color`` nor ``entity`` is given.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if entity is not None:
        style = entity_style(entity)
        color, dashes, width = style.colour(), style.dashes, style.width
        label = label or entity
    if color is None:
        raise ValueError("line needs a color or an entity")
    if smooth:
        x, y = monotone_cubic(x, y)
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_WIDTH if width is None else width,
        linestyle=dashes if dashes is not None else "-",
        label=label,
        zorder=zorder,
    )[0]


def points(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str | None = None,
    size: float = 30.0,
    label: str | None = None,
    zorder: float = 4.0,
    marker: str | None = None,
    alpha: float | None = None,
    entity: str | None = None,
) -> Any:
    """Draw markers with a canvas-colour rim so they separate from lines.

    Two series that share positions (for example two probability mass
    functions that both tend to zero) stay visible together when they use
    different ``marker`` shapes and a translucent ``alpha``. With ``entity``
    the colour, marker, and label come from the set contract.

    Args:
        ax: Target axes.
        x: Abscissae.
        y: Ordinates.
        color: Series colour (or give ``entity``).
        size: Marker area in points squared.
        label: Optional legend label.
        zorder: Drawing order.
        marker: Matplotlib marker code (default a filled circle).
        alpha: Optional marker opacity in ``(0, 1]``.
        entity: Entity of the set contract.

    Returns:
        The Matplotlib path collection.

    Raises:
        ValueError: If neither ``color`` nor ``entity`` is given.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if entity is not None:
        style = entity_style(entity)
        color, marker = style.colour(), style.marker
        label = label or entity
    if color is None:
        raise ValueError("points needs a color or an entity")
    return ax.scatter(
        x,
        y,
        s=size,
        color=color,
        marker=marker or "o",
        alpha=alpha,
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
    entity: str | None = None,
) -> Any:
    """Draw a thin dashed reference line in the accent colour with a label.

    Horizontal lines are labelled at the right edge; vertical lines at the top.

    Args:
        ax: Target axes.
        y: Ordinate of a horizontal reference.
        x: Abscissa of a vertical reference.
        label: Short label such as ``"exact value"``.
        color: Override colour (defaults to the accent).
        entity: Entity of the set contract (``"truth"``, ``"threshold"``,
            ...); sets the colour and names the line for the set audit.

    Returns:
        The Matplotlib line.

    Raises:
        ValueError: If neither or both of ``x`` and ``y`` are given.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if (x is None) == (y is None):
        raise ValueError("supply exactly one of x or y")
    if entity is not None:
        color = entity_colour(entity)
    colour = color or active_palette().accent
    name = entity or "_reference"
    if y is not None:
        artist = ax.axhline(
            y,
            color=colour,
            linewidth=REFERENCE_WIDTH,
            linestyle=REFERENCE,
            zorder=2,
            label=name,
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
        x,
        color=colour,
        linewidth=REFERENCE_WIDTH,
        linestyle=REFERENCE,
        zorder=2,
        label=name,
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


def ordered_colours(count: int, palette: ChartPalette | None = None) -> list[str]:
    """Return colours for ordered members (steps, decreasing ``eps``).

    Members with an order (greedy design steps, Sinkhorn runs at decreasing
    regularization) are not categories: they take the strong end of the
    single-hue sequential ramp, light to dark, so the order is read from the
    depth and no categorical hue (which means observed or model) is spent
    on them (ADR-0098). Member ``k`` (from 0) always sits at depth
    ``min(1, 0.55 + 0.15 k)``, so step 2 has the same colour in a two-step
    and in a three-step figure, and neighbours stay apart under protan and
    deutan vision. Direct labels of the lightest member are darkened toward
    the ink for legibility by the caller.

    Args:
        count: Number of members (positive).
        palette: Palette whose ramp is used (the active one by default).

    Returns:
        ``count`` hexadecimal colours; members beyond the fourth share the
        darkest.

    Raises:
        ValueError: If ``count`` is not positive.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import to_hex

    if count < 1:
        raise ValueError("count must be positive")
    ramp = sequential_colormap(palette)
    return [to_hex(ramp(min(1.0, 0.55 + 0.15 * index))) for index in range(count)]


def _wrap_wide_titles(figure: Any, renderer: Any) -> None:
    """Break axes titles that are wider than the figure into several lines.

    A title keeps its words in order; the line length in characters is the
    title length scaled by the ratio of the available width to the measured
    width, so the wrapped title fits the canvas.

    Args:
        figure: Matplotlib figure, drawn once so text extents are known.
        renderer: Renderer used to measure text.

    References:
        [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import textwrap

    width = figure.bbox.width
    for ax in figure.axes:
        for name in ("_left_title", "title", "_right_title"):
            text = getattr(ax, name, None)
            if text is None or not text.get_text() or not text.get_visible():
                continue
            extent = text.get_window_extent(renderer)
            available = width - max(0.0, extent.x0) - 4.0
            if extent.width <= available or available <= 0.0:
                continue
            content = " ".join(text.get_text().split())
            characters = max(12, int(len(content) * available / extent.width) - 1)
            text.set_text(textwrap.fill(content, width=characters))


def fit_text_to_canvas(figure: Any) -> None:
    """Enlarge the canvas so every visible artist lies inside the figure box.

    Saving with tight bounds keeps text that sticks out of the figure, but the
    figure itself still reports text outside its box, and a later save with
    other bounds would cut it. Titles wider than the figure are wrapped first;
    any remaining overflow grows the canvas by the overflow plus a small
    margin while every axes, figure text, and figure legend keeps its size
    and its position relative to the plotted data.

    Args:
        figure: Matplotlib figure; modified in place.

    References:
        [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    _wrap_wide_titles(figure, renderer)
    figure.canvas.draw()
    if figure.get_layout_engine() is not None:
        # Freeze the computed layout so the moves below are not undone.
        figure.set_layout_engine("none")
    tight = figure.get_tightbbox(renderer)
    if tight is None:
        return
    width, height = (float(value) for value in figure.get_size_inches())
    margin = CANVAS_MARGIN_INCHES
    left = max(0.0, margin - tight.x0)
    bottom = max(0.0, margin - tight.y0)
    right = max(0.0, tight.x1 + margin - width)
    top = max(0.0, tight.y1 + margin - height)
    if left == bottom == right == top == 0.0:
        return
    new_width = width + left + right
    new_height = height + bottom + top

    def moved(x: float, y: float) -> tuple[float, float]:
        """Map a figure fraction on the old canvas to the new canvas.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return (x * width + left) / new_width, (y * height + bottom) / new_height

    for ax in figure.axes:
        box = ax.get_position()
        x0, y0 = moved(box.x0, box.y0)
        ax.set_position(
            (x0, y0, box.width * width / new_width, box.height * height / new_height)
        )
    for text in figure.texts:
        if text.get_transform() == figure.transFigure:
            text.set_position(moved(*text.get_position()))
    figure.set_size_inches(new_width, new_height, forward=True)
    figure.canvas.draw()


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    dpi: int = 200,
    description: str | None = None,
) -> Path:
    """Save a figure with the theme canvas, tight bounds, and padding.

    The file carries a text alternative (ADR-0078): PNG and SVG files get
    ``Title`` and ``Description`` metadata, PDF files ``Title`` and
    ``Subject``. The text is generated from the figure's titles, subtitle,
    and axes by :func:`probstat.visualization.accessibility.describe_figure`
    unless ``description`` is given.

    Args:
        figure: Matplotlib figure.
        path: Destination file path; parent directories are created.
        dpi: Output resolution.
        description: Text alternative to store instead of the generated one.

    Returns:
        The written path.

    References:
        [REF-MATPLOTLIB], [REF-PNG-SPEC-2003], [REF-W3C-WCAG21-2018].
        No equation is evaluated.

    """
    from .accessibility import image_metadata

    fit_text_to_canvas(figure)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = image_metadata(figure, destination)
    if metadata is not None and description is not None:
        key = "Subject" if "Subject" in metadata else "Description"
        metadata[key] = " ".join(description.split())
    figure.savefig(
        destination,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.25,
        metadata=metadata,
    )
    return destination
