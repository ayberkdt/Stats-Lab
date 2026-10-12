"""Plots for optimal transport: plans, Sinkhorn, quantiles, and geodesics.

The functions take result objects of :mod:`probstat.information` (ADR-0043)
or laws with ``quantile`` methods and only present them; they never re-solve
a transport problem, except that the ``W_1`` value in the quantile figure is
computed by :func:`probstat.information.quantile_wasserstein`.

Colour roles (ADR-0024, ADR-0046): the source measure (first law) takes the
observed hue and the target measure (second law) ``series[3]``, the slot
ADR-0044 gave to a second compared quantity; transported mass, plan entries,
and Sinkhorn runs at decreasing ``eps`` use the single-hue sequential ramp
of the active palette because they are magnitudes or an ordered family;
displacement interpolants and the barycenter are model results (model hue);
the Sinkhorn tolerance and the identity are accent references.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.information import (
    SinkhornResult,
    TransportPlan,
    normal_wasserstein,
    quantile_wasserstein,
)

from ._layout import halo, wrapped_title_block
from .theme import (
    ANNOTATION_SIZE,
    active_palette,
    axis_labels,
    band,
    direct_labels,
    line,
    new_axes,
    pyplot,
    reference_line,
    sequential_colormap,
    style_axes,
    themed,
)

PlanLike = TransportPlan | SinkhornResult | Sequence[Sequence[float]]


def _matrix(plan: PlanLike) -> list[list[float]]:
    """Return a plan's coupling matrix as lists of non-negative floats.

    Args:
        plan: Exact plan, Sinkhorn result, or a matrix.

    Returns:
        The coupling matrix.

    Raises:
        ValueError: If the matrix is empty, ragged, negative, or not finite.

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.3]. No equation is evaluated.

    """
    raw = plan.plan if isinstance(plan, TransportPlan | SinkhornResult) else plan
    matrix = [[float(v) for v in row] for row in raw]
    if not matrix or not matrix[0]:
        raise ValueError("the plan must be a non-empty matrix")
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("the plan must be rectangular")
    if any(not math.isfinite(v) or v < 0.0 for row in matrix for v in row):
        raise ValueError("plan entries must be finite and non-negative")
    if not sum(map(sum, matrix)) > 0.0:
        raise ValueError("the plan must carry positive mass")
    return matrix


def _ramp_word() -> str:
    """Return how the sequential ramp's strong end looks on this canvas.

    The porcelain ramp ends dark on a light canvas, the graphite ramp ends
    light on a dark canvas, so subtitles say "darker" or "brighter".

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import to_rgb

    def luminance(colour: str) -> float:
        """Return the Rec. 709 luma of a hex colour.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        r, g, b = to_rgb(colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    ramp = active_palette().sequential
    return "darker" if luminance(ramp[-1]) < luminance(ramp[0]) else "brighter"


def used_cells(matrix: Sequence[Sequence[float]], share: float = 0.005) -> int:
    """Count plan cells holding at least ``share`` of the total mass.

    Args:
        matrix: Coupling matrix.
        share: Mass fraction in ``(0, 1)`` that makes a cell count as used.

    Returns:
        Number of used cells.

    Raises:
        ValueError: If ``share`` is outside ``(0, 1)``.

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.3]. No equation is evaluated.

    """
    if not 0.0 < share < 1.0:
        raise ValueError("share must lie strictly inside (0, 1)")
    total = math.fsum(v for row in matrix for v in row)
    return sum(1 for row in matrix for v in row if v >= share * total)


def crossing_pairs(
    matrix: Sequence[Sequence[float]],
    source_points: Sequence[float],
    target_points: Sequence[float],
    share: float = 0.005,
) -> int:
    """Count pairs of used plan cells whose transport segments cross.

    Cells ``(i, j)`` and ``(k, l)`` cross when ``x_i < x_k`` and
    ``y_j > y_l``. An optimal plan for a convex cost of ``|x - y|`` on the
    line is monotone, so it has no crossing pairs.

    Args:
        matrix: Coupling matrix.
        source_points: Source atom positions (row order).
        target_points: Target atom positions (column order).
        share: Mass fraction that makes a cell count as used.

    Returns:
        Number of crossing pairs.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.6], [REF-VILLANI-2009].

    """
    total = math.fsum(v for row in matrix for v in row)
    cells = [
        (source_points[i], target_points[j])
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if v >= share * total
    ]
    return sum(
        1
        for a, (x1, y1) in enumerate(cells)
        for x2, y2 in cells[a + 1 :]
        if (x1 - x2) * (y1 - y2) < 0.0
    )


@themed
def plot_transport_lines(
    plan: PlanLike,
    source_points: Sequence[float],
    target_points: Sequence[float],
    *,
    names: tuple[str, str] = ("source", "target"),
    cost: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw a 1-D transport plan as weighted lines between two rows of atoms.

    Source atoms sit on the upper row (observed hue), target atoms on the
    lower row (``series[3]``), each sized by its mass (the plan's row and
    column sums). Every plan entry above 0.5% of the mass is a line from
    ``x_i`` to ``y_j`` whose width and colour depth (sequential ramp) grow
    with ``P_ij``. Crossing lines are counted for the title: optimal plans
    for convex costs on the line never cross.

    Args:
        plan: Exact plan, Sinkhorn result, or coupling matrix.
        source_points: Source atom positions, one per plan row.
        target_points: Target atom positions, one per plan column.
        names: Labels of the two rows.
        cost: Transport cost for the title (default: the plan's own cost).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the positions do not match the plan's shape.

    Equation:
        [EQ-OT-LP-001], [EQ-OT-W1D-001].

    References:
        [REF-PEYRE-CUTURI-2019, Chapter 2], [REF-VILLANI-2009],
        [REF-MATPLOTLIB].

    """
    matrix = _matrix(plan)
    xs = [float(x) for x in source_points]
    ys = [float(y) for y in target_points]
    if len(xs) != len(matrix) or len(ys) != len(matrix[0]):
        raise ValueError("positions must match the plan's rows and columns")
    total = math.fsum(v for row in matrix for v in row)
    source_mass = [math.fsum(row) / total for row in matrix]
    target_mass = [math.fsum(col) / total for col in zip(*matrix, strict=True)]
    axes = new_axes(ax, figsize=(7.4, 4.0))
    palette = active_palette()
    ramp = sequential_colormap()
    peak = max(v for row in matrix for v in row)
    entries = sorted(
        ((v, i, j) for i, row in enumerate(matrix) for j, v in enumerate(row)),
        key=lambda entry: entry[0],
    )
    for entry_mass, i, j in entries:
        if entry_mass < 0.005 * total:
            continue
        depth = entry_mass / peak
        axes.plot(
            [xs[i], ys[j]], [1.0, 0.0], color=ramp(0.35 + 0.65 * depth),
            linewidth=0.6 + 7.0 * depth, solid_capstyle="butt", zorder=2,
        )  # fmt: skip
    largest = max(*source_mass, *target_mass)
    span = max(*xs, *ys) - min(*xs, *ys) or 1.0
    for row_y, positions, masses, colour in (
        (1.0, xs, source_mass, palette.observed),
        (0.0, ys, target_mass, palette.series[3]),
    ):
        axes.scatter(
            positions, [row_y] * len(positions), color=colour,
            s=[40.0 + 260.0 * m / largest for m in masses],
            edgecolors=palette.canvas, linewidths=1.2, zorder=5,
        )  # fmt: skip
        staggered = _stagger(positions, 0.045 * span)
        for x, m, level in zip(positions, masses, staggered, strict=True):
            lift = 13.0 + 11.0 * level
            axes.annotate(
                f"{m:.2f}", (x, row_y), xytext=(0.0, lift if row_y else -lift),
                textcoords="offset points", ha="center",
                va="bottom" if row_y else "top", fontsize=ANNOTATION_SIZE,
                color=palette.ink,
            )  # fmt: skip
    axes.set_yticks([0.0, 1.0])
    axes.set_yticklabels([names[1], names[0]])
    for label, colour in zip(
        axes.get_yticklabels(), (palette.series[3], palette.observed), strict=True
    ):
        label.set_color(colour)
        label.set_fontweight("bold")
    axes.set_ylim(-0.32, 1.32)
    axes.grid(axis="y", visible=False)
    axes.set_xlim(min(*xs, *ys) - 0.06 * span, max(*xs, *ys) + 0.06 * span)
    axis_labels(axes, "Position", None)
    used = used_cells(matrix)
    crossings = crossing_pairs(matrix, xs, ys)
    value = cost
    if value is None and isinstance(plan, TransportPlan):
        value = plan.cost
    if value is None and isinstance(plan, SinkhornResult):
        value = plan.transport_cost
    cost_text = f": cost {value:.3f}" if value is not None else ""
    if crossings == 0:
        title = (
            f"The plan moves mass monotonically through {used} of "
            f"{len(xs) * len(ys)} cells{cost_text}"
        )
    else:
        title = (
            f"{crossings} pairs of transport lines cross: the plan is not "
            f"monotone{cost_text}"
        )
    wrapped_title_block(
        axes,
        title,
        [
            "dots: atoms sized by mass (numbers)",
            f"lines: plan entries, wider and {_ramp_word()} for more mass",
        ],
    )
    return axes


def _stagger(positions: Sequence[float], gap: float) -> list[int]:
    """Return label rows so labels of atoms closer than ``gap`` alternate.

    Args:
        positions: Atom positions.
        gap: Smallest distance at which two labels can share a row.

    Returns:
        ``0`` or ``1`` per position (``1`` moves the label one row out).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    rows = [0] * len(positions)
    for previous, current in zip(order, order[1:], strict=False):
        if positions[current] - positions[previous] < gap:
            rows[current] = 1 - rows[previous]
    return rows


@themed
def plot_plan_matrices(
    plans: Mapping[str, PlanLike],
    *,
    source_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
) -> Any:
    """Show one or more coupling matrices as heatmaps on one colour scale.

    Panels share the sequential single-hue colormap (light to dark with
    mass) and its limits, so an exact plan and entropic plans of the same
    problem compare directly. The title counts cells holding at least 0.5% of
    the mass in the first and last panels (:func:`used_cells`).

    Args:
        plans: Panel title mapped to a plan (at most three, equal shapes).
        source_labels: Optional row labels.
        target_labels: Optional column labels.

    Returns:
        The figure.

    Raises:
        ValueError: If no plan, more than three plans, unequal shapes, or
            label counts that do not match are given.

    Equation:
        [EQ-OT-LP-001], [EQ-OT-SINK-001].

    References:
        [REF-CUTURI-2013], [REF-PEYRE-CUTURI-2019, Section 4.1],
        [REF-MATPLOTLIB].

    """
    if not 1 <= len(plans) <= 3:
        raise ValueError("between one and three plans can be drawn")
    matrices = {name: _matrix(plan) for name, plan in plans.items()}
    shapes = {(len(m), len(m[0])) for m in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("all plans must have the same shape")
    rows, cols = shapes.pop()
    if source_labels is not None and len(source_labels) != rows:
        raise ValueError("source_labels must match the plan rows")
    if target_labels is not None and len(target_labels) != cols:
        raise ValueError("target_labels must match the plan columns")
    plt = pyplot()
    palette = active_palette()
    count = len(matrices)
    figure, panels = plt.subplots(
        1, count, figsize=(3.2 * count + 1.2, 3.9), squeeze=False,
        gridspec_kw={"wspace": 0.12},
    )  # fmt: skip
    peak = max(v for m in matrices.values() for row in m for v in row)
    image = None
    for index, (axes, (name, matrix)) in enumerate(
        zip(panels[0], matrices.items(), strict=True)
    ):
        style_axes(axes)
        image = axes.imshow(
            matrix, cmap=sequential_colormap(), vmin=0.0, vmax=peak,
            aspect="equal", interpolation="nearest",
        )  # fmt: skip
        axes.grid(False)
        axes.set_xticks(range(cols))
        axes.set_yticks(range(rows))
        axes.set_xticklabels(
            target_labels if target_labels is not None else range(1, cols + 1)
        )
        axes.set_yticklabels(
            (source_labels if source_labels is not None else range(1, rows + 1))
            if index == 0
            else []
        )
        for side in ("left", "bottom"):
            axes.spines[side].set_visible(False)
        axes.tick_params(length=0)
        axes.set_xlabel(name, color=palette.ink)
    panels[0][0].set_ylabel("Source atom")
    bar = figure.colorbar(image, ax=list(panels[0]), fraction=0.025, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=palette.muted, labelcolor=palette.ink, length=0)
    bar.set_label("Plan mass", color=palette.ink)
    names = list(matrices)
    first, last = used_cells(matrices[names[0]]), used_cells(matrices[names[-1]])
    if count == 1:
        title = f"The plan places mass in {first} of {rows * cols} cells"
    else:
        title = (
            f"With {names[-1]} the plan spreads over {last} of {rows * cols} "
            f"cells; {names[0]} uses {first}"
        )
    wrapped_title_block(
        panels[0][0],
        title,
        [
            f"cells: plan mass on one shared scale ({_ramp_word()} is more)",
            "rows: source atoms, columns: target atoms",
        ],
        width_inches=figure.get_figwidth(),
    )
    return figure


@themed
def plot_sinkhorn_convergence(
    results: Mapping[str, SinkhornResult] | Sequence[SinkhornResult],
    *,
    tolerance: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot the Sinkhorn marginal error against iteration on a log scale.

    Each run's ``trace`` (row-marginal error after every ``(g, f)`` pair) is
    a line; runs are ordered by ``eps`` and drawn from light to dark in the
    sequential ramp so smaller ``eps`` has more contrast. The optional tolerance is
    the accent reference.

    Args:
        results: Sinkhorn results, or a mapping from label to result
            (default labels ``eps = ...``).
        tolerance: Optional stopping tolerance to mark.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If no result is given or a trace is empty.

    Equation:
        [EQ-OT-SINK-001].

    References:
        [REF-CUTURI-2013], [REF-PEYRE-CUTURI-2019, Section 4.2],
        [REF-MATPLOTLIB].

    """
    named = (
        dict(results)
        if isinstance(results, Mapping)
        else {f"eps = {r.epsilon:g}": r for r in results}
    )
    if not named:
        raise ValueError("at least one Sinkhorn result is required")
    if any(not r.trace for r in named.values()):
        raise ValueError("every result needs a non-empty trace")
    order = sorted(named, key=lambda name: -named[name].epsilon)
    axes = new_axes(ax)
    ramp = sequential_colormap()
    labels = []
    for index, name in enumerate(order):
        result = named[name]
        depth = 0.55 + 0.45 * (index / max(1, len(order) - 1))
        colour = ramp(depth)
        trace = [max(v, 1e-300) for v in result.trace]
        steps = list(range(1, len(trace) + 1))
        line(axes, steps, trace, color=colour, width=1.8)
        labels.append((steps, trace, f"{name}\n{len(trace):,} it", colour))
    axes.set_yscale("log")
    axes.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter

    axes.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes.xaxis.set_minor_formatter(NullFormatter())
    if tolerance is not None:
        reference_line(axes, y=tolerance, label=f"tolerance {tolerance:.0e}")
    floor = min(min(trace) for _, trace, _, _ in labels)
    ceiling = max(max(trace) for _, trace, _, _ in labels)
    if tolerance is not None:
        floor = min(floor, tolerance)
    axes.set_ylim(floor / 300.0, 3.0 * ceiling)
    for steps, trace, text, colour in labels:
        axes.annotate(
            text, (float(steps[-1]), trace[-1]), xytext=(0.0, -8.0),
            textcoords="offset points", ha="center", va="top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=colour,
            linespacing=1.1, path_effects=halo(),
        )  # fmt: skip
    axis_labels(axes, "Iteration", "Row-marginal error")
    fastest, slowest = named[order[0]], named[order[-1]]
    if len(order) == 1:
        title = (
            f"Sinkhorn at eps = {fastest.epsilon:g} reaches "
            f"{fastest.marginal_error:.0e} in {fastest.iterations:,} iterations"
        )
    else:
        ratio = slowest.iterations / max(1, fastest.iterations)
        title = (
            f"Sinkhorn slows {ratio:.0f}x as eps shrinks from {fastest.epsilon:g} "
            f"to {slowest.epsilon:g} ({fastest.iterations:,} to "
            f"{slowest.iterations:,} iterations)"
        )
    failed = [name for name in order if not named[name].converged]
    parts = [
        f"lines: error after each update pair, {_ramp_word()} for smaller eps",
        "labels: iterations to stop",
    ]
    if failed:
        parts.append(f"not converged: {', '.join(failed)}")
    wrapped_title_block(axes, title, parts)
    return axes


@themed
def plot_quantile_wasserstein(
    first: Any,
    second: Any,
    *,
    names: tuple[str, str] = ("first", "second"),
    clip: float = 0.005,
    nodes: int = 401,
    ax: Any | None = None,
) -> Any:
    """Show ``W_1`` as the area between two quantile functions.

    On the line ``W_1 = int_0^1 |Q_0(u) - Q_1(u)| du`` (EQ-OT-W1D-001). The
    two quantile functions are drawn on ``[clip, 1 - clip]`` (first law in
    the observed hue, second in ``series[3]``) with the area between them
    muted; the title's ``W_1`` comes from
    :func:`probstat.information.quantile_wasserstein` over the whole unit
    interval, and the subtitle reports the share of it that the drawn range
    holds.

    Args:
        first: Law with a ``quantile`` method.
        second: Law with a ``quantile`` method.
        names: Labels of the two laws.
        clip: Probability cut at each end of the drawn range in ``(0, 0.1)``.
        nodes: Grid points on the drawn range (at least 11).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        TypeError: If a law has no ``quantile`` method.
        ValueError: If ``clip`` or ``nodes`` is invalid.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.6], [REF-VILLANI-2009],
        [REF-MATPLOTLIB].

    """
    for law in (first, second):
        if not callable(getattr(law, "quantile", None)):
            raise TypeError(f"{type(law).__name__} has no quantile method")
    if not 0.0 < clip < 0.1:
        raise ValueError("clip must lie strictly inside (0, 0.1)")
    if nodes < 11:
        raise ValueError("nodes must be at least 11")
    us = [clip + (1.0 - 2.0 * clip) * k / (nodes - 1) for k in range(nodes)]
    q0 = [float(first.quantile(u)) for u in us]
    q1 = [float(second.quantile(u)) for u in us]
    distance = quantile_wasserstein(first, second, p=1.0)
    gaps = [abs(a - b) for a, b in zip(q0, q1, strict=True)]
    shown = math.fsum(
        0.5 * (gaps[k] + gaps[k + 1]) * (us[k + 1] - us[k]) for k in range(nodes - 1)
    )
    axes = new_axes(ax)
    palette = active_palette()
    band(
        axes, us, [min(a, b) for a, b in zip(q0, q1, strict=True)],
        [max(a, b) for a, b in zip(q0, q1, strict=True)], color=palette.muted,
        alpha=0.22,
    )  # fmt: skip
    line(axes, us, q0, color=palette.observed)
    line(axes, us, q1, color=palette.series[3])
    crossings: list[float] = []
    last: tuple[float, float] | None = None
    scale = max(1e-300, max(abs(a) + abs(b) for a, b in zip(q0, q1, strict=True)))
    for u, a, b in zip(us, q0, q1, strict=True):
        difference = a - b
        if abs(difference) <= 1e-12 * scale:
            continue
        if last is not None and last[1] * difference < 0.0:
            crossings.append(0.5 * (last[0] + u))
        last = (u, difference)
    axes.set_xlim(0.0, 1.0)
    direct_labels(
        axes,
        [(us[-1], q0[-1], names[0], palette.observed),
         (us[-1], q1[-1], names[1], palette.series[3])],
    )  # fmt: skip
    axis_labels(axes, "Probability level u", "Quantile Q(u)")
    if crossings:
        where = ", ".join(f"{u:.2f}" for u in crossings[:3])
        tail = f"; the quantile functions cross at u = {where}"
    elif math.fsum(a - b for a, b in zip(q0, q1, strict=True)) >= 0.0:
        tail = f"; {names[0]} lies above {names[1]} at every drawn level"
    else:
        tail = f"; {names[0]} lies below {names[1]} at every drawn level"
    title = f"W1 = {distance:.3f} is the area between the quantile functions{tail}"
    wrapped_title_block(
        axes,
        title,
        [
            "shaded: |Q0(u) - Q1(u)|",
            f"drawn for u in [{clip:g}, {1 - clip:g}], "
            f"holding {shown / distance:.1%} of W1"
            if distance > 0.0
            else "identical laws",
        ],
    )
    return axes


def _normal_pdf(x: float, mean: float, sd: float) -> float:
    """Return the normal density.

    Equation:
        [EQ-DIST-NORM-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


@themed
def plot_displacement_interpolation(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    times: Sequence[float] = (0.25, 0.5, 0.75),
    names: tuple[str, str] = ("start", "end"),
    ax: Any | None = None,
) -> Any:
    """Draw the ``W_2`` geodesic between two normal laws against mixing.

    The displacement interpolant at time ``t`` is the barycenter with
    weights ``(1 - t, t)``; for normals it is
    ``N((1 - t) mu_0 + t mu_1, ((1 - t) s_0 + t s_1)^2)`` (EQ-OT-BARY-001).
    The end laws are drawn in the observed hue and ``series[3]``, the
    interpolants in the model hue (the midpoint thicker), and the 50/50
    mixture ``(p_0 + p_1)/2`` as a muted contrast: mixing averages
    densities, transport moves mass.

    Args:
        first: ``(mu_0, sigma_0)`` with ``sigma_0 > 0``.
        second: ``(mu_1, sigma_1)`` with ``sigma_1 > 0``.
        times: Interpolation times strictly inside ``(0, 1)``.
        names: Labels of the two end laws.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If a scale is not positive or a time is outside
            ``(0, 1)``.

    Equation:
        [EQ-OT-BARY-001], [EQ-OT-GAUSS-001], [EQ-DIST-NORM-001].

    References:
        [REF-VILLANI-2009], [REF-PEYRE-CUTURI-2019, Section 9.2],
        [REF-ALVAREZ-ESTEBAN-ETAL-2016], [REF-MATPLOTLIB].

    """
    (m0, s0), (m1, s1) = (float(first[0]), float(first[1])), (
        float(second[0]),
        float(second[1]),
    )
    if not (s0 > 0.0 and s1 > 0.0):
        raise ValueError("standard deviations must be positive")
    stops = sorted(float(t) for t in times)
    if any(not 0.0 < t < 1.0 for t in stops):
        raise ValueError("times must lie strictly inside (0, 1)")
    low = min(m0 - 4.0 * s0, m1 - 4.0 * s1)
    high = max(m0 + 4.0 * s0, m1 + 4.0 * s1)
    grid = [low + (high - low) * k / 600 for k in range(601)]
    axes = new_axes(ax)
    palette = active_palette()
    mixture = [0.5 * (_normal_pdf(x, m0, s0) + _normal_pdf(x, m1, s1)) for x in grid]
    line(axes, grid, mixture, color=palette.muted, width=1.4, zorder=2.0)
    labels = []
    for t in stops:
        mean, sd = (1.0 - t) * m0 + t * m1, (1.0 - t) * s0 + t * s1
        curve = [_normal_pdf(x, mean, sd) for x in grid]
        middle = abs(t - 0.5) < 1e-12
        line(
            axes, grid, curve, color=palette.model,
            width=2.4 if middle else 1.2, zorder=3.5 if middle else 3.0,
        )  # fmt: skip
        peak = _normal_pdf(mean, mean, sd)
        axes.annotate(
            f"t = {t:g}", (mean, peak), xytext=(0.0, 5.0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=ANNOTATION_SIZE, color=palette.model,
            fontweight="bold" if middle else "normal",
        )  # fmt: skip
    for (mean, sd), name, colour in (
        ((m0, s0), names[0], palette.observed),
        ((m1, s1), names[1], palette.series[3]),
    ):
        curve = [_normal_pdf(x, mean, sd) for x in grid]
        line(axes, grid, curve, color=colour, width=2.0, zorder=4.0)
        peak = _normal_pdf(mean, mean, sd)
        labels.append((mean + 0.6 * sd, peak, name, colour))
    tops = [max(mixture)] + [_normal_pdf(0.0, 0.0, min(s0, s1))]
    axes.set_ylim(0.0, 1.18 * max(tops))
    axes.set_xlim(low, high)
    for x, y, text, colour in labels:
        axes.annotate(
            text, (x, y), xytext=(6.0, 0.0), textcoords="offset points",
            ha="left", va="center", fontsize=ANNOTATION_SIZE, fontweight="bold",
            color=colour,
        )  # fmt: skip
    peaks = [
        k
        for k in range(1, len(grid) - 1)
        if mixture[k] > mixture[k - 1] and mixture[k] >= mixture[k + 1]
    ]
    k_mix = peaks[-1] if peaks else max(range(len(grid)), key=lambda k: mixture[k])
    axes.annotate(
        "50/50 mixture", (grid[k_mix], mixture[k_mix]), xytext=(0.0, -10.0),
        textcoords="offset points", ha="center", va="top",
        fontsize=ANNOTATION_SIZE, color=palette.muted, fontweight="bold",
        path_effects=halo(),
        zorder=6,
    )  # fmt: skip
    axis_labels(axes, "x", "Density")
    modes = len(peaks)
    mid_mean, mid_sd = 0.5 * (m0 + m1), 0.5 * (s0 + s1)
    contrast = (
        "unlike the bimodal 50/50 mixture"
        if modes > 1
        else (
            "narrower than the 50/50 mixture (sd "
            f"{math.sqrt(0.5 * (s0**2 + s1**2) + 0.25 * (m1 - m0) ** 2):.2f})"
        )
    )
    title = (
        f"The W2 midpoint of {names[0]} and {names[1]} is "
        f"N({mid_mean:.3g}, {mid_sd:.3g}^2), one bump {contrast}"
    )
    wrapped_title_block(
        axes,
        title,
        [
            "colour: geodesic N((1-t) mu0 + t mu1, ((1-t) s0 + t s1)^2)",
            "grey: density average",
            f"W2 = {normal_wasserstein(m0, s0, m1, s1):.3f}",
        ],
    )
    return axes
