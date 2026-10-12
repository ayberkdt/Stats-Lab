"""Plots for multiple design points, mixture sampling, and AK-MCS.

The functions take result objects of :mod:`probstat.reliability` (ADR-0041)
plus arrays computed by the caller, and they never call the limit-state
model: a limit-state contour is drawn from a grid of ``G`` values the caller
evaluated, and failure flags of importance samples are passed in. Only the
stored AK-MCS surrogate (a Gaussian process) is evaluated, to colour the
population by its predicted class.

Colour roles (ADR-0024, ADR-0046): the exact limit state ``G = 0`` is a
reference (accent, solid because it is computed, failure domain filled in
the same hue); design points and FORM linearizations are model results
(model hue, tangents with the ``FITTED`` dash); samples are data (failed in
the observed hue, safe in muted ink); the ``beta`` circle is muted context.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.reliability import (
    AKMCSResult,
    FORMResult,
    MixtureImportanceResult,
    MultipleDesignPointsResult,
    MultiPointFORMResult,
)

from ._layout import halo, wrapped_title_block
from .theme import (
    ANNOTATION_SIZE,
    FITTED,
    REFERENCE_WIDTH,
    active_palette,
    axis_labels,
    band,
    line,
    new_axes,
    points,
    pyplot,
    reference_line,
    source_note,
    style_axes,
    themed,
)

LimitStateGrid = tuple[Sequence[float], Sequence[float], Sequence[Sequence[float]]]


def _check_grid(
    grid: LimitStateGrid,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Validate a limit-state grid ``(xs, ys, values)`` with ``values[i][j]``.

    ``values[i][j]`` is ``G`` at ``(xs[j], ys[i])``, the layout of
    Matplotlib's ``contour``. The grid must contain safe and failed nodes.

    Args:
        grid: Abscissae, ordinates, and limit-state values.

    Returns:
        The grid as lists of floats.

    Raises:
        ValueError: If shapes disagree, a value is not finite, fewer than two
            nodes per axis are given, or ``G`` does not change sign.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    xs = [float(x) for x in grid[0]]
    ys = [float(y) for y in grid[1]]
    values = [[float(v) for v in row] for row in grid[2]]
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("the grid needs at least two nodes per axis")
    if len(values) != len(ys) or any(len(row) != len(xs) for row in values):
        raise ValueError("grid values must have len(ys) rows of len(xs) entries")
    flat = [v for row in values for v in row]
    if not all(math.isfinite(v) for v in flat):
        raise ValueError("grid values must be finite")
    if min(flat) > 0.0 or max(flat) <= 0.0:
        raise ValueError("the grid must contain both safe and failed nodes")
    return xs, ys, values


def _draw_limit_state(axes: Any, grid: LimitStateGrid) -> None:
    """Fill the failure domain and draw ``G = 0`` as the accent reference.

    Args:
        axes: Target axes.
        grid: Validated limit-state grid.

    References:
        [EQ-REL-FORM-001], [REF-MELCHERS-BECK-2018, Chapter 3],
        [REF-MATPLOTLIB].

    """
    xs, ys, values = _check_grid(grid)
    palette = active_palette()
    low = min(v for row in values for v in row)
    axes.contourf(
        xs, ys, values, levels=[low - 1.0, 0.0], colors=[palette.accent],
        alpha=0.08, zorder=0,
    )  # fmt: skip
    axes.contour(
        xs, ys, values, levels=[0.0], colors=[palette.accent],
        linewidths=REFERENCE_WIDTH * 1.4, zorder=2,
    )  # fmt: skip


def _square_map(axes: Any, extent: float, names: tuple[str, str]) -> None:
    """Make a square standard-normal map with a hairline grid on both axes.

    Args:
        axes: Target axes.
        extent: Half-width of the shown square.
        names: Axis names of the two standard-normal variables.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    axes.set_xlim(-extent, extent)
    axes.set_ylim(-extent, extent)
    axes.set_aspect("equal")
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    axes.spines["left"].set_visible(True)
    axes.plot([0.0], [0.0], marker="+", color=palette.muted, markersize=8, zorder=3)
    axis_labels(axes, f"Standard normal {names[0]}", f"Standard normal {names[1]}")


def _forms(
    design_points: MultipleDesignPointsResult | Sequence[FORMResult],
) -> tuple[FORMResult, ...]:
    """Return the FORM results of a design-point search, checked for 2-D.

    Args:
        design_points: Search result or a sequence of FORM results.

    Returns:
        The FORM results.

    Raises:
        ValueError: If there is no design point or the space is not 2-D.

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998]. No equation is evaluated.

    """
    forms = (
        design_points.forms
        if isinstance(design_points, MultipleDesignPointsResult)
        else tuple(design_points)
    )
    if not forms:
        raise ValueError("at least one design point is required")
    if any(len(form.design_point_u) != 2 for form in forms):
        raise ValueError("only two-dimensional standard-normal spaces can be drawn")
    return forms


def _free_angle(angles: Sequence[float]) -> float:
    """Return the angle on the circle farthest from every given angle.

    Args:
        angles: Occupied angles in radians.

    Returns:
        A label angle in radians (``pi/2`` when nothing is occupied); ties
        prefer the top of the circle.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if not angles:
        return math.pi / 2.0
    candidates = [2.0 * math.pi * k / 72 for k in range(72)]

    def clearance(theta: float) -> float:
        """Return the smallest angular distance from ``theta`` to an angle.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return min(
            abs((theta - a + math.pi) % (2.0 * math.pi) - math.pi) for a in angles
        )

    return max(candidates, key=lambda t: (round(clearance(t), 6), math.sin(t)))


@themed
def plot_design_points(
    design_points: MultipleDesignPointsResult | Sequence[FORMResult],
    *,
    limit_state_grid: LimitStateGrid | None = None,
    samples: Sequence[Sequence[float]] | None = None,
    failed: Sequence[bool] | None = None,
    combined: MultiPointFORMResult | None = None,
    mixture: MixtureImportanceResult | None = None,
    names: tuple[str, str] = ("u1", "u2"),
    extent: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Map design points, FORM tangents, and samples in standard-normal space.

    Each design point ``u*_k`` is a model-hue dot joined to the origin by a
    muted radius; its FORM linearization (the line through ``u*_k``
    orthogonal to ``u*_k``) is a ``FITTED`` model-hue line. The dotted muted
    circle has radius ``beta_1``. With ``limit_state_grid`` the exact surface
    ``G = 0`` is the solid accent reference over a light accent failure
    domain. Importance samples (for example drawn from the mixture of
    :func:`probstat.reliability.mixture_importance_sampling`) are small dots:
    failed in the observed hue, safe in muted ink. The caller evaluates
    ``G`` on the grid and the failure flags; this function never calls the
    model.

    Args:
        design_points: Search result or FORM results with 2-D design points.
        limit_state_grid: Optional ``(xs, ys, values)`` with
            ``values[i][j] = G(xs[j], ys[i])``.
        samples: Optional standard-normal sample points.
        failed: Failure flag of each sample (required with ``samples``).
        combined: Optional multi-point FORM result for the title.
        mixture: Optional mixture importance-sampling result for the subtitle.
        names: Names of the two standard-normal variables.
        extent: Half-width of the shown square (default from ``beta``).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the space is not two-dimensional, the grid is invalid,
            or samples and flags do not align.

    Equation:
        [EQ-REL-FORM-001], [EQ-REL-MDP-001], [EQ-REL-MDP-002],
        [EQ-REL-MDP-003].

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-DERKIUREGHIAN-2005],
        [REF-MELCHERS-BECK-2018], [REF-MATPLOTLIB].

    """
    forms = _forms(design_points)
    if (samples is None) != (failed is None):
        raise ValueError("samples and failed must be given together")
    sample_points: list[tuple[float, float]] = []
    flags: list[bool] = []
    if samples is not None and failed is not None:
        sample_points = [(float(p[0]), float(p[1])) for p in samples]
        flags = [bool(f) for f in failed]
        if len(sample_points) != len(flags):
            raise ValueError("failed must hold one flag per sample")
        if any(len(p) != 2 for p in samples):
            raise ValueError("samples must be two-dimensional")
    betas = [form.beta for form in forms]
    half = extent if extent is not None else max(4.0, math.ceil(1.6 * max(betas)))
    axes = new_axes(ax, figsize=(6.8, 6.8))
    palette = active_palette()
    if limit_state_grid is not None:
        _draw_limit_state(axes, limit_state_grid)
    if sample_points:
        safe = [p for p, f in zip(sample_points, flags, strict=True) if not f]
        hit = [p for p, f in zip(sample_points, flags, strict=True) if f]
        for chosen, colour, size, alpha in (
            (safe, palette.muted, 5.0, 0.35),
            (hit, palette.observed, 7.0, 0.65),
        ):
            if chosen:
                axes.scatter(
                    [p[0] for p in chosen], [p[1] for p in chosen], s=size,
                    color=colour, alpha=alpha, edgecolors="none", zorder=1.5,
                    rasterized=True,
                )  # fmt: skip
    angles = [math.atan2(f.design_point_u[1], f.design_point_u[0]) for f in forms]
    theta = [2.0 * math.pi * k / 360 for k in range(361)]
    radius = min(betas)
    axes.plot(
        [radius * math.cos(t) for t in theta],
        [radius * math.sin(t) for t in theta],
        color=palette.muted, linewidth=1.0, linestyle=(0, (1.0, 2.5)), zorder=2,
    )  # fmt: skip
    label_angle = _free_angle(angles)
    axes.annotate(
        f"beta = {radius:.2f}",
        (radius * math.cos(label_angle), radius * math.sin(label_angle)),
        xytext=(-9.0 * math.cos(label_angle), -9.0 * math.sin(label_angle)),
        textcoords="offset points", ha="center", va="center",
        fontsize=ANNOTATION_SIZE, color=palette.muted, fontweight="semibold",
        path_effects=halo(), zorder=6,
    )  # fmt: skip
    reach = 3.0 * half
    for index, form in enumerate(forms):
        u1, u2 = form.design_point_u
        norm = math.hypot(u1, u2)
        n1, n2 = u1 / norm, u2 / norm
        axes.plot([0.0, u1], [0.0, u2], color=palette.muted, linewidth=0.7, zorder=2)
        line(
            axes,
            [u1 - reach * n2, u1 + reach * n2],
            [u2 + reach * n1, u2 - reach * n1],
            color=palette.model, dashes=FITTED, width=1.4, zorder=3.0,
        )  # fmt: skip
        points(axes, [u1], [u2], color=palette.model, size=70.0, zorder=5.0)
        d1, d2 = (n1 - n2) / math.sqrt(2.0), (n2 + n1) / math.sqrt(2.0)
        axes.annotate(
            f"u*{index + 1}",
            (u1, u2),
            xytext=(20.0 * d1, 20.0 * d2),
            textcoords="offset points",
            ha="center", va="center", fontsize=ANNOTATION_SIZE,
            fontweight="bold", color=palette.model, zorder=6,
            path_effects=halo(),
        )  # fmt: skip
    _square_map(axes, half, names)
    count = len(forms)
    single = forms[0].probability
    if max(betas) - min(betas) < 1e-3 * max(1.0, max(betas)):
        head = (
            f"One design point at beta = {betas[0]:.2f}"
            if count == 1
            else f"{count} design points share beta = {betas[0]:.2f}"
        )
    else:
        shown = ", ".join(f"{b:.2f}" for b in betas[:4])
        head = f"{count} design points at beta = {shown}" + (
            ", ..." if count > 4 else ""
        )
    if combined is not None and count > 1:
        title = (
            f"{head}; together they carry {combined.probability / single:.1f}x "
            "the one-point FORM probability"
        )
    else:
        title = f"{head}: FORM gives Pf = {single:.2e}"
    parts = ["dashed: FORM tangents", "dotted: beta circle"]
    if limit_state_grid is not None:
        parts.insert(0, "line: G = 0 (shaded: failure domain)")
    if sample_points:
        parts.append(
            f"dots: {len(sample_points):,} importance samples "
            f"(colour: {sum(flags):,} failed)"
        )
    wrapped_title_block(axes, title, parts)
    estimates = [f"one-point FORM {single:.2e}"]
    if combined is not None and count > 1:
        estimates.append(f"multi-point FORM {combined.probability:.2e}")
    if mixture is not None:
        estimates.append(
            f"mixture IS {mixture.probability:.2e} +/- {mixture.standard_error:.1e}"
        )
    source_note(axes, "Pf: " + " · ".join(estimates))
    return axes


def _population_points(
    population: Sequence[Sequence[float]], expected: int
) -> list[tuple[float, float]]:
    """Return the 2-D AK-MCS population as tuples, checked against its size.

    Args:
        population: Population points in standard-normal space.
        expected: ``AKMCSResult.population``.

    Returns:
        Points as float pairs.

    Raises:
        ValueError: If the size or dimension is wrong.

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

    """
    if len(population) != expected:
        raise ValueError(
            f"population must hold the {expected} points of the AK-MCS run"
        )
    if any(len(p) != 2 for p in population):
        raise ValueError("only two-dimensional populations can be drawn")
    return [(float(p[0]), float(p[1])) for p in population]


@themed
def plot_ak_mcs_classification(
    result: AKMCSResult,
    population: Sequence[Sequence[float]],
    *,
    limit_state_grid: LimitStateGrid | None = None,
    learning_threshold: float = 2.0,
    lowest: int = 12,
    surrogate_nodes: int = 61,
    names: tuple[str, str] = ("u1", "u2"),
    extent: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Show the AK-MCS population by predicted class with the training set.

    The final surrogate of ``result`` is evaluated at every population point:
    predicted failures (``mu <= 0``) are observed-hue dots, predicted safe
    points muted. Training points are model-hue markers (hollow: initial
    design, filled: added by active learning). Population points with
    learning value ``U = |mu| / sigma`` below ``learning_threshold`` are
    ringed in ink; when none is left (the stopping rule holds) the
    ``lowest`` smallest-``U`` points are ringed instead. With
    ``limit_state_grid`` the exact ``G = 0`` is the accent reference and the
    surrogate's zero contour, evaluated on at most ``surrogate_nodes`` nodes
    per axis of the same grid, is a ``FITTED`` model-hue line.

    Args:
        result: Output of :func:`probstat.reliability.ak_mcs` on a 2-D state.
        population: The run's Monte Carlo population (same size and order).
        limit_state_grid: Optional ``(xs, ys, values)`` of the true ``G``.
        learning_threshold: Highlight threshold of ``U`` (positive).
        lowest: Points ringed when no ``U`` is below the threshold.
        surrogate_nodes: Grid nodes per axis for the surrogate contour.
        names: Names of the two standard-normal variables.
        extent: Half-width of the shown square (default from the data).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the population does not match the run, the space is
            not two-dimensional, or an option is invalid.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002], [EQ-SUR-GP-003].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-MATPLOTLIB].

    """
    if not learning_threshold > 0.0:
        raise ValueError("learning_threshold must be positive")
    if lowest < 0 or surrogate_nodes < 2:
        raise ValueError("lowest must be non-negative and surrogate_nodes >= 2")
    surrogate = result.surrogate
    if len(surrogate.points[0]) != 2:
        raise ValueError("only two-dimensional AK-MCS runs can be drawn")
    pool = _population_points(population, result.population)
    means: list[float] = []
    learning: list[float] = []
    for point in pool:
        mean, variance = surrogate.predict_with_variance(point)
        means.append(mean)
        sd = math.sqrt(variance)
        learning.append(abs(mean) / sd if sd > 0.0 else math.inf)
    training = [(float(p[0]), float(p[1])) for p in surrogate.points]
    trained = set(training)
    candidates = [i for i, p in enumerate(pool) if p not in trained]
    flagged = [i for i in candidates if learning[i] < learning_threshold]
    ringed_below = bool(flagged)
    if not flagged:
        flagged = sorted(candidates, key=lambda i: learning[i])[:lowest]
    reach = max(max(abs(c) for p in pool for c in p), 4.0)
    half = extent if extent is not None else math.ceil(reach + 0.25)
    axes = new_axes(ax, figsize=(6.8, 6.8))
    palette = active_palette()
    if limit_state_grid is not None:
        _draw_limit_state(axes, limit_state_grid)
        xs, ys, _ = _check_grid(limit_state_grid)
        step_x = max(1, math.ceil(len(xs) / surrogate_nodes))
        step_y = max(1, math.ceil(len(ys) / surrogate_nodes))
        sx, sy = xs[::step_x], ys[::step_y]
        predicted = [[surrogate.predict((x, y)) for x in sx] for y in sy]
        flat = [v for row in predicted for v in row]
        if min(flat) <= 0.0 < max(flat):
            axes.contour(
                sx, sy, predicted, levels=[0.0], colors=[palette.model],
                linewidths=1.5, linestyles=[FITTED], zorder=3,
            )  # fmt: skip
    safe = [i for i in range(len(pool)) if means[i] > 0.0]
    fail = [i for i in range(len(pool)) if means[i] <= 0.0]
    for chosen, colour, size, alpha in (
        (safe, palette.muted, 3.0, 0.3),
        (fail, palette.observed, 16.0, 0.9),
    ):
        if chosen:
            axes.scatter(
                [pool[i][0] for i in chosen], [pool[i][1] for i in chosen], s=size,
                color=colour, alpha=alpha, edgecolors="none", zorder=1.5,
                rasterized=True,
            )  # fmt: skip
    if flagged:
        axes.scatter(
            [pool[i][0] for i in flagged], [pool[i][1] for i in flagged], s=70.0,
            facecolors="none", edgecolors=palette.ink, linewidths=0.9, zorder=4,
        )  # fmt: skip
    initial = training[: result.initial_size]
    added = training[result.initial_size :]
    axes.scatter(
        [p[0] for p in initial], [p[1] for p in initial], s=46.0, marker="s",
        facecolors=palette.canvas, edgecolors=palette.model, linewidths=1.6,
        zorder=5,
    )  # fmt: skip
    if added:
        axes.scatter(
            [p[0] for p in added], [p[1] for p in added], s=46.0, marker="s",
            color=palette.model, edgecolors=palette.canvas, linewidths=0.8,
            zorder=5,
        )  # fmt: skip
    _square_map(axes, half, names)
    title = (
        f"AK-MCS classifies {result.population:,} points with "
        f"{result.evaluations} model calls"
    )
    if not math.isnan(result.reference_probability):
        title += (
            f"; {result.misclassified} differ from the true sign"
            if result.misclassified
            else "; every predicted sign matches the true limit state"
        )
    if ringed_below:
        ring = f"rings: {len(flagged)} with U < {learning_threshold:g}"
    else:
        smallest = min(learning[i] for i in flagged)
        ring = f"rings: {len(flagged)} lowest U (min {smallest:.2f})"
    parts = [
        f"dots: {len(fail)} predicted failed, rest safe",
        f"squares: {len(initial)} initial + {len(added)} added runs",
        ring,
    ]
    if limit_state_grid is not None:
        parts[0:0] = ["line: true G = 0", "dashed: surrogate"]
    wrapped_title_block(axes, title, parts)
    note = (
        f"Pf = {result.probability:.2e} "
        f"(c.o.v. {result.coefficient_of_variation:.2f})"
    )
    if not math.isnan(result.reference_probability):
        note += f" · same-point MC {result.reference_probability:.2e}"
    source_note(axes, note)
    return axes


@themed
def plot_ak_mcs_convergence(
    result: AKMCSResult, *, learning_threshold: float = 2.0
) -> Any:
    """Plot the AK-MCS estimate and ``min U`` against added training points.

    Two stacked panels share the horizontal axis (no dual axis). Top: the
    surrogate estimate ``P_f = N_f / N`` (model hue) with a band of two
    Monte Carlo standard errors ``2 c.o.v. P_f`` and, when available, the
    same-population Monte Carlo value as the accent reference. Bottom:
    ``min U`` over the population on a log scale with the stopping
    threshold as the accent reference. Records that share a training size
    (a hyperparameter refit or a population enlargement) appear as vertical
    steps.

    Args:
        result: Output of :func:`probstat.reliability.ak_mcs`.
        learning_threshold: Stopping threshold of ``min U`` (positive).

    Returns:
        The figure with two panels.

    Raises:
        ValueError: If the history holds fewer than two records or the
            threshold is not positive.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-MATPLOTLIB].

    """
    if not learning_threshold > 0.0:
        raise ValueError("learning_threshold must be positive")
    history = result.history
    if len(history) < 2:
        raise ValueError("at least two learning records are required")
    plt = pyplot()
    palette = active_palette()
    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(7.2, 5.8),
        gridspec_kw={"height_ratios": (1.25, 1.0), "hspace": 0.14},
    )  # fmt: skip
    for axes in (top, bottom):
        style_axes(axes)
    xs = [float(h.training_size - result.initial_size) for h in history]
    largest = max(h.probability for h in history)
    exponent = math.floor(math.log10(largest)) if largest > 0.0 else 0
    scale = 10.0**exponent
    pf = [h.probability / scale for h in history]
    spread = [
        2.0 * h.coefficient_of_variation * p
        for h, p in zip(history, pf, strict=True)
    ]
    band(
        top, xs, [max(0.0, p - s) for p, s in zip(pf, spread, strict=True)],
        [p + s for p, s in zip(pf, spread, strict=True)], color=palette.model,
    )  # fmt: skip
    line(top, xs, pf, color=palette.model, width=1.8)
    points(top, [xs[-1]], [pf[-1]], color=palette.model, size=40.0)
    if not math.isnan(result.reference_probability):
        reference_line(
            top, y=result.reference_probability / scale,
            label=f"same-point MC {result.reference_probability:.2e}",
        )  # fmt: skip
    top.set_ylim(0.0, None)
    unit = f"x 1e{exponent}" if exponent else None
    axis_labels(top, None, "Pf estimate", y_unit=unit)
    finite = [
        (x, h.min_learning)
        for x, h in zip(xs, history, strict=True)
        if math.isfinite(h.min_learning) and h.min_learning > 0.0
    ]
    line(bottom, [x for x, _ in finite], [u for _, u in finite],
         color=palette.model, width=1.6)  # fmt: skip
    points(bottom, [x for x, _ in finite], [u for _, u in finite],
           color=palette.model, size=18.0)  # fmt: skip
    bottom.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter

    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bottom.yaxis.set_minor_formatter(NullFormatter())
    reference_line(bottom, y=learning_threshold, label=f"U = {learning_threshold:g}")
    axis_labels(bottom, "Training points added by active learning", "min U")
    final = result.min_learning
    if result.converged:
        title = (
            f"AK-MCS stops after {result.added} added points with min U = {final:.2f} "
            f">= {learning_threshold:g}; Pf = {result.probability:.2e}"
        )
    else:
        title = (
            f"AK-MCS did not converge after {result.added} added points "
            f"(min U = {final:.2f}): {result.message}"
        )
    wrapped_title_block(
        top,
        title,
        [
            f"top: estimate with +/- 2 Monte Carlo s.e. (N = {result.population:,})",
            "bottom: smallest learning value U = |mu|/sigma",
            f"{result.evaluations} model calls",
        ],
    )
    return figure
