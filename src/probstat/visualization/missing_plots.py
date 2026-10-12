"""Plots for incomplete data: missingness patterns, imputations, Rubin's rules.

The functions take result objects of :mod:`probstat.inference.missing`
(ADR-0047) and only present them; they never refit EM or rerun data
augmentation. Colour roles (ADR-0024, ADR-0052):

* Pattern matrix: missing cells carry the observed hue, because the missing
  entries are the data property the figure is about; observed cells are
  muted context, so the eye goes to the holes.
* Imputation diagnostic: observed values use the observed hue and imputed
  values, which are draws from the fitted model, the model hue.
* Estimates with intervals are data summaries in the observed hue, as in
  every forest plot of the library; a true value is the accent reference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from probstat.inference import MissingPattern, MultipleImputation, PooledEstimate
from probstat.statistics import quantiles

from ._layout import halo, wrapped_title_block
from .theme import (
    ANNOTATION_SIZE,
    active_palette,
    axis_labels,
    points,
    pyplot,
    reference_line,
    style_axes,
    themed,
)

_MAX_PATTERNS = 12
_GOLDEN = 0.6180339887498949


def _names(names: Sequence[str] | None, count: int) -> list[str]:
    """Return variable names, checking their number.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if names is None:
        return [f"x{j}" for j in range(count)]
    if len(names) != count:
        raise ValueError("names must match the number of variables")
    return [str(name) for name in names]


@themed
def plot_missing_pattern(
    pattern: MissingPattern,
    *,
    names: Sequence[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw the missingness patterns as a matrix of observed and missing cells.

    One row per distinct pattern (the most common twelve, most observed
    variables first, as :func:`probstat.inference.missing_pattern` sorts
    them), one column per variable. Missing cells are filled in the observed
    hue and observed cells are muted; each row carries its row count and
    share as a direct label, and each column label states the variable's
    missing fraction.

    Args:
        pattern: Output of :func:`probstat.inference.missing_pattern` (also
            stored on EM and imputation results).
        names: Optional variable names (default ``x0, x1, ...``).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``names`` does not match the number of variables.

    References:
        [REF-LITTLE-RUBIN-2019], [REF-SCHAFER-1997], [REF-MATPLOTLIB].

    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    labels = _names(names, pattern.n_variables)
    palette = active_palette()
    order = sorted(
        range(len(pattern.patterns)),
        key=lambda k: (-sum(pattern.patterns[k]), -pattern.pattern_counts[k]),
    )
    shown = order[:_MAX_PATTERNS]
    rows = len(shown)
    cols = pattern.n_variables
    axes = (
        ax
        if ax is not None
        else pyplot().subplots(figsize=(0.95 * cols + 4.0, 0.42 * rows + 2.4))[1]
    )
    style_axes(axes)
    for position, index in enumerate(shown):
        y = rows - 1 - position
        for j, observed in enumerate(pattern.patterns[index]):
            axes.add_patch(
                Rectangle(
                    (j - 0.5, y - 0.5), 1.0, 1.0,
                    facecolor=to_rgba(palette.muted, 0.22)
                    if observed
                    else palette.observed,
                    edgecolor=palette.canvas, linewidth=2.0,
                )
            )  # fmt: skip
        count = pattern.pattern_counts[index]
        axes.annotate(
            f"{count:,} rows ({count / pattern.n_observations:.0%})",
            xy=(cols - 0.5, y), xytext=(8.0, 0.0), textcoords="offset points",
            ha="left", va="center", fontsize=ANNOTATION_SIZE,
            color=palette.ink, annotation_clip=False,
        )  # fmt: skip
    axes.set_xlim(-0.5, cols - 0.5)
    axes.set_ylim(-0.5, rows - 0.5)
    axes.set_xticks(range(cols))
    axes.set_xticklabels(
        [
            f"{name}\n{fraction:.0%} missing"
            for name, fraction in zip(labels, pattern.missing_fraction, strict=True)
        ]
    )
    axes.set_yticks(range(rows))
    axes.set_yticklabels(
        [
            "complete"
            if all(pattern.patterns[index])
            else f"{cols - sum(pattern.patterns[index])} missing"
            for index in reversed(shown)
        ]
    )
    axes.grid(False)
    axes.tick_params(length=0)
    for side in ("left", "bottom"):
        axes.spines[side].set_visible(False)
    worst = max(range(cols), key=lambda j: pattern.missing_fraction[j])
    complete = pattern.complete_cases
    n = pattern.n_observations
    if pattern.missing_fraction[worst] == 0.0:
        title = f"No values are missing in the {n:,} rows"
    else:
        title = (
            f"{complete:,} of {n:,} rows ({complete / n:.0%}) are complete; "
            f"{labels[worst]} is missing most often "
            f"({pattern.missing_fraction[worst]:.0%} of rows)"
        )
    parts = [
        "rows: missingness patterns",
        "filled: missing",
        "grey: observed",
        f"{pattern.overall_missing_fraction:.0%} of all entries missing",
    ]
    if len(order) > rows:
        parts.append(f"{len(order) - rows} rarer patterns not shown")
    wrapped_title_block(axes, title, parts, width_inches=axes.figure.get_figwidth())
    return axes


def _is_missing(value: object) -> bool:
    """Return whether an entry is ``None`` or ``nan``.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _strip(
    axes: Any, values: Sequence[float], y: float, colour: str, offset: int
) -> tuple[float, float, float, float, float]:
    """Draw a jittered strip with 50% and 90% ranges and the median.

    Returns:
        The 5%, 25%, 50%, 75%, and 95% sample quantiles.

    References:
        [EQ-STAT-QUANT-001], [REF-MATPLOTLIB].

    """
    jitter = [
        y + 0.08 + 0.26 * (((k + offset) * _GOLDEN) % 1.0) for k in range(len(values))
    ]
    axes.scatter(
        values, jitter, s=8.0, color=colour, linewidths=0.0, zorder=2,
        alpha=min(0.6, 12.0 / math.sqrt(len(values))),
    )  # fmt: skip
    q = quantiles(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    axes.hlines(y - 0.14, q[0], q[4], color=colour, linewidth=1.6, zorder=3)
    axes.hlines(y - 0.14, q[1], q[3], color=colour, linewidth=5.0, zorder=3)
    points(axes, [q[2]], [y - 0.14], color=colour, size=36.0, zorder=4)
    return q[0], q[1], q[2], q[3], q[4]


@themed
def plot_imputation_diagnostic(
    rows: Iterable[Iterable[object]],
    imputation: MultipleImputation,
    *,
    names: Sequence[str] | None = None,
    max_variables: int = 4,
) -> Any:
    """Compare observed and imputed values of every incomplete variable.

    One panel per variable with missing entries (at most ``max_variables``,
    the most incomplete first). The top strip holds the observed values
    (observed hue), the bottom strip the imputed values of all ``m``
    completed data sets (model hue); each strip has its 50% and 90% ranges
    and median below it. Under missingness at random the imputed values may
    legitimately differ from the observed ones (they are drawn given the
    other variables), so a shift is not by itself a fault; imputations far
    outside the observed range, or a spread very different from it, are.

    Args:
        rows: The incomplete rows that were imputed (``None``/``nan`` for
            missing entries).
        imputation: Output of :func:`probstat.inference.impute_multivariate_normal`
            for these rows.
        names: Optional variable names.
        max_variables: Largest number of panels (at least one).

    Returns:
        The figure.

    Raises:
        ValueError: If the rows do not match the completed data sets, no
            value is missing, or ``max_variables`` is below one.

    Equation:
        [EQ-MISS-006], [EQ-STAT-QUANT-001].

    References:
        [REF-RUBIN-1987], [REF-SCHAFER-1997], [REF-MATPLOTLIB].

    """
    data = [list(row) for row in rows]
    datasets = imputation.datasets
    if not datasets or len(data) != len(datasets[0]):
        raise ValueError("rows must match the completed data sets")
    if max_variables < 1:
        raise ValueError("max_variables must be at least one")
    width = len(data[0])
    if any(len(row) != width for row in data) or len(datasets[0][0]) != width:
        raise ValueError("rows must match the completed data sets")
    labels = _names(names, width)
    missing = {
        j: [i for i, row in enumerate(data) if _is_missing(row[j])]
        for j in range(width)
    }
    incomplete = sorted(
        (j for j in range(width) if missing[j]), key=lambda j: -len(missing[j])
    )[:max_variables]
    if not incomplete:
        raise ValueError("no value is missing")
    plt = pyplot()
    palette = active_palette()
    figure, panels = plt.subplots(
        len(incomplete), 1, figsize=(7.4, 1.75 * len(incomplete) + 1.7),
        squeeze=False, gridspec_kw={"hspace": 0.7},
    )  # fmt: skip
    shifts: list[tuple[float, str, float, float]] = []
    for axes, j in zip(panels[:, 0], incomplete, strict=True):
        style_axes(axes)
        observed = [float(row[j]) for row in data if not _is_missing(row[j])]  # type: ignore[arg-type]
        imputed = [float(dataset[i][j]) for dataset in datasets for i in missing[j]]
        if len(observed) < 2:
            raise ValueError(f"{labels[j]} needs at least two observed values")
        q_obs = _strip(axes, observed, 1.0, palette.observed, 0)
        q_imp = _strip(axes, imputed, 0.0, palette.model, 7)
        axes.set_yticks([1.0, 0.0])
        axes.set_yticklabels(
            [
                f"observed ({len(observed):,})",
                f"imputed ({len(datasets)} x {len(missing[j]):,})",
            ]
        )
        axes.set_ylim(-0.45, 1.5)
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", color=palette.rule, linewidth=0.7)
        axis_labels(axes, labels[j], None)
        spread = q_obs[3] - q_obs[1] or 1.0
        shift = abs(q_imp[2] - q_obs[2]) / spread
        shifts.append((shift, labels[j], q_imp[2], q_obs[2]))
    size, name, imputed_median, observed_median = max(shifts)
    if size >= 0.1:
        direction = "above" if imputed_median > observed_median else "below"
        title = (
            f"Imputed {name} values sit {abs(imputed_median - observed_median):.2f} "
            f"{direction} the observed ones (medians {imputed_median:.2f} vs "
            f"{observed_median:.2f})"
        )
    else:
        title = (
            "Imputed and observed values share their centre for every variable "
            f"(largest median shift {size:.2f} interquartile ranges, {name})"
        )
    wrapped_title_block(
        panels[0, 0],
        title,
        [
            f"dots: observed values and imputations from m = {len(datasets)} "
            "completed data sets",
            "bars: 50% and 90% ranges, dot: median",
            "a shift is expected when values are missing at random given "
            "other variables",
        ],
        width_inches=figure.get_figwidth(),
    )
    return figure


def _verb(count: int, singular: str, plural: str) -> str:
    """Return the verb form agreeing with ``count`` subjects.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return singular if count == 1 else plural


def _join(items: Sequence[str]) -> str:
    """Join names as ``"a"``, ``"a and b"``, or ``"a, b and c"``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _value_labels(
    axes: Any,
    rows: Sequence[tuple[float, float, float, float]],
    reference: float | None,
) -> None:
    """Print each estimate beside its whisker, away from a reference line.

    A label goes right of the upper limit unless the reference line would
    cross it there; then it goes left of the lower limit. Call after the
    axis limits are final.

    Args:
        axes: Target axes.
        rows: ``(row, estimate, lower, upper)`` per entry.
        reference: Abscissa of a vertical reference line, if any.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    axes.get_xlim()
    scale = axes.figure.dpi / 72.0
    for row, estimate, lower, upper in rows:
        text = f"{estimate:.3f}"
        width = (0.6 * ANNOTATION_SIZE * len(text)) * scale
        right = axes.transData.transform((upper, row))[0] + 8.0 * scale
        centre = axes.transData.transform((estimate, row))[0]
        line_x = (
            axes.transData.transform((reference, row))[0]
            if reference is not None
            else -math.inf
        )
        if not right - 2.0 * scale <= line_x <= right + width + 2.0 * scale:
            anchor, offset, align = (upper, (8.0, 0.0), "left")
        elif abs(line_x - centre) > 0.5 * width + 3.0 * scale:
            anchor, offset, align = (estimate, (0.0, 11.0), "center")
        else:
            anchor, offset, align = (lower, (-8.0, 0.0), "right")
        axes.annotate(
            text, (anchor, row), xytext=offset, textcoords="offset points",
            ha=align, va="center", fontsize=ANNOTATION_SIZE, color=palette.ink,
            path_effects=halo(), zorder=5,
        )  # fmt: skip


@themed
def plot_rubin_summary(
    pooled: PooledEstimate,
    *,
    others: Mapping[str, tuple[float, float, float]] | None = None,
    truth: float | None = None,
    quantity: str = "Estimate",
    label: str = "multiple imputation",
    ax: Any | None = None,
) -> Any:
    """Compare a Rubin's-rules estimate with other analyses of the same data.

    Each row is a dot (estimate) with a whisker (interval). The multiple
    imputation row also shows, as a darker inner bar, the interval that the
    average within-imputation variance ``Ubar`` alone would give with the
    same quantile; the full whisker uses Rubin's total variance
    ``T = Ubar + (1 + 1/m) B``, so the difference is the uncertainty the
    missing values add. A true value, when known (simulations), is the
    accent reference and the title counts the intervals that cover it.

    Args:
        pooled: Output of :func:`probstat.inference.pool_estimates`.
        others: Further analyses as ``{label: (estimate, lower, upper)}``,
            for example complete cases, mean imputation, or EM.
        truth: Optional true value of the estimand.
        quantity: Value-axis title.
        label: Row label of the pooled estimate.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If a value is not finite or an interval is reversed.

    Equation:
        [EQ-MISS-007], [EQ-MISS-008].

    References:
        [REF-RUBIN-1987], [REF-BARNARD-RUBIN-1999], [REF-LITTLE-RUBIN-2019],
        [REF-MATPLOTLIB].

    """
    entries: list[tuple[str, float, float, float]] = [
        (label, pooled.estimate, pooled.lower, pooled.upper)
    ]
    for name, (estimate, lower, upper) in (others or {}).items():
        entries.append((str(name), float(estimate), float(lower), float(upper)))
    for name, estimate, lower, upper in entries:
        if not all(math.isfinite(v) for v in (estimate, lower, upper)):
            raise ValueError(f"{name}: values must be finite")
        if not lower <= estimate <= upper:
            raise ValueError(f"{name}: the interval must contain the estimate")
    if truth is not None and not math.isfinite(truth):
        raise ValueError("truth must be finite")
    palette = active_palette()
    axes = (
        ax
        if ax is not None
        else pyplot().subplots(figsize=(7.4, 0.62 * len(entries) + 2.2))[1]
    )
    style_axes(axes)
    rows = list(range(len(entries) - 1, -1, -1))
    half_total = pooled.upper - pooled.estimate
    within = (
        half_total * math.sqrt(pooled.within_variance / pooled.total_variance)
        if pooled.total_variance > 0.0
        else half_total
    )
    for row, (_, estimate, lower, upper) in zip(rows, entries, strict=True):
        axes.hlines(row, lower, upper, color=palette.observed, linewidth=2.2,
                    alpha=0.55, zorder=2)  # fmt: skip
        if row == rows[0]:
            axes.hlines(row, estimate - within, estimate + within,
                        color=palette.observed, linewidth=5.0, zorder=3)  # fmt: skip
        points(axes, [estimate], [row], color=palette.observed, size=52.0)
    axes.set_yticks(rows)
    axes.set_yticklabels([entry[0] for entry in entries])
    axes.set_ylim(-0.6, len(entries) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    edges = [v for _, _, lower, upper in entries for v in (lower, upper)]
    if truth is not None:
        edges.append(truth)
    low, high = min(edges), max(edges)
    span = high - low or 1.0
    axes.set_xlim(low - 0.05 * span, high + 0.18 * span)
    if truth is not None:
        reference_line(axes, x=truth, label="true value")
    _value_labels(
        axes,
        [
            (row, estimate, lower, upper)
            for row, (_, estimate, lower, upper) in zip(rows, entries, strict=True)
        ],
        truth,
    )
    axis_labels(axes, quantity, None)
    widening = (
        math.sqrt(pooled.total_variance / pooled.within_variance) - 1.0
        if pooled.within_variance > 0.0
        else math.inf
    )
    if truth is not None:
        cover = [n for n, _, lower, upper in entries if lower <= truth <= upper]
        miss = [n for n, _, lower, upper in entries if not lower <= truth <= upper]
        if not miss:
            title = f"Every interval covers the true value {truth:.4g}"
        elif not cover:
            title = f"No interval covers the true value {truth:.4g}"
        else:
            title = (
                f"{_join(cover)} {_verb(len(cover), 'covers', 'cover')} the true "
                f"value {truth:.4g}; {_join(miss)} "
                f"{_verb(len(miss), 'misses', 'miss')} it"
            )
    else:
        title = (
            f"Missing values widen the {label} interval by {widening:.0%} over "
            f"the within-imputation variance alone (FMI "
            f"{pooled.fraction_missing_information:.2f})"
        )
    df = pooled.degrees_of_freedom
    wrapped_title_block(
        axes,
        title,
        [
            f"dots: estimates, whiskers: {pooled.confidence:.0%} intervals",
            "darker bar: Ubar only, full whisker: Rubin's total variance T",
            f"m = {pooled.imputations}, FMI = "
            f"{pooled.fraction_missing_information:.2f}, df = "
            + (f"{df:.0f}" if math.isfinite(df) else "inf"),
        ],
        width_inches=axes.figure.get_figwidth(),
    )
    return axes
