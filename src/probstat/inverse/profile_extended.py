"""Two-dimensional profiles, bootstrap calibration and ODE identifiability.

Experimental (added in 0.28).

Three extensions of :mod:`probstat.inverse.profile_likelihood`:

* :func:`profile_likelihood_2d` profiles the negative log-likelihood over a
  rectangular grid of *two* interest parameters. The grid is visited along a
  serpentine (boustrophedon) path so that every point is warm-started from a
  grid neighbour; the nuisance block of the supplied estimate is a second
  start. Every point carries its own optimizer status. A point whose
  optimization failed is stored as ``inf`` with status ``"failed"`` and is
  never interpolated or filled. :func:`profile_region_2d` thresholds the
  likelihood ratio at the ``chi2(2)`` quantile (a joint region); cells that did
  not converge are reported as ``"unreliable"``, not as inside or outside.
* :func:`bootstrap_lr_calibration` replaces the asymptotic ``chi2`` threshold
  of a profile likelihood-ratio test by a parametric-bootstrap quantile:
  data are simulated from the fitted model with a seeded stream, the model is
  refitted, and the statistic is recomputed. The empirical quantile is
  reported with a distribution-free Monte Carlo standard error and compared
  with the ``chi2`` threshold. The statistic is the likelihood ratio *for the
  hypothesis that the interest coordinates equal their fitted values*; it is
  not a coverage study of intervals.
* :func:`ode_sensitivity` integrates the forward sensitivity equations
  alongside an ordinary differential equation with the library's fixed-step
  Runge-Kutta solver, assembles the (noise-weighted, scaled) sensitivity
  matrix at the observation times and reports its singular values, a numerical
  rank under an explicit relative tolerance rule and the unidentifiable
  directions.

Scope. All three tools are numerical and local to the supplied estimate or
parameter vector. A rank-deficient sensitivity matrix demonstrates local
non-identifiability at that parameter; a full-rank matrix does *not* prove
structural identifiability (that needs differential-algebraic methods, not
finite numerical evaluation) and small singular values are practical, noise
dependent statements. The ``chi2`` calibration is asymptotic and regular.

References:
    [REF-RAUE-ETAL-2009], [REF-DAVISON-HINKLEY-1997], [REF-SELF-LIANG-1987],
    [REF-HAIRER-NORSETT-WANNER-1993].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from probstat.distributions import ChiSquare
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    singular_value_decomposition,
)
from probstat.numerics import integrate_fixed_step
from probstat.optimization import minimize_bfgs

_EPS = 2.0**-52
_Z95 = 1.959963984540054
_Data = TypeVar("_Data")

OdeRhs = Callable[[float, Sequence[float], Sequence[float]], Sequence[float]]
"""Right-hand side ``f(t, y, theta)`` of a parametrized ODE."""


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _real(value: object, name: str) -> float:
    """Return a real number (infinity allowed) or raise.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} entries must be real numbers")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} entries must not be NaN")
    return result


def _bounds_vector(
    bound: Sequence[float | None] | None, size: int, name: str, fill: float
) -> tuple[float, ...]:
    """Validate an optional per-coordinate bound vector.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """
    if bound is None:
        return (fill,) * size
    items = tuple(bound)
    if len(items) != size:
        raise ValueError(f"{name} must have one entry per parameter")
    return tuple(fill if item is None else _real(item, name) for item in items)


def _safe(
    function: Callable[[Vector], float], counter: list[int]
) -> Callable[[Vector], float]:
    """Wrap ``function``: invalid points give ``inf``; calls are counted.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """

    def wrapped(x: Vector) -> float:
        """Evaluate with accounting.

        References:
            [EQ-INV-PLX-001]; implementation helper.

        """
        counter[0] += 1
        try:
            value = float(function(x))
        except (ValueError, ArithmeticError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    return wrapped


def _minimize(
    objective: Callable[[Vector], float],
    start: Vector,
    box: Sequence[tuple[float | None, float | None]],
    tolerance: float,
    max_iterations: int,
) -> tuple[float, Vector, bool, str] | None:
    """Minimize from one start; ``None`` when the start is infeasible.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """
    if not math.isfinite(objective(start)):
        return None
    try:
        res = minimize_bfgs(
            objective,
            start,
            gradient_tolerance=tolerance,
            max_iterations=max_iterations,
            bounds=box,
        )
    except ValueError:
        return None
    return res.value, res.x, res.converged, res.message


def _box(
    lo: Sequence[float], hi: Sequence[float], which: Sequence[int]
) -> list[tuple[float | None, float | None]]:
    """Return a ``minimize_bfgs`` box for the coordinates ``which``.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """
    return [
        (None if lo[j] == -math.inf else lo[j], None if hi[j] == math.inf else hi[j])
        for j in which
    ]


def _check_grid(values: Sequence[float], name: str, centre: float) -> Vector:
    """Validate a strictly increasing grid that brackets ``centre``.

    References:
        [EQ-INV-PLX-001]; implementation helper.

    """
    grid = as_vector(values, name=name)
    if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError(f"{name} must have at least two strictly increasing values")
    if not grid[0] <= centre <= grid[-1]:
        raise ValueError(f"{name} must bracket the estimate of its parameter")
    return grid


# ---------------------------------------------------------------------------
# 1. two-dimensional profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Profile2DPoint:
    """One grid cell of a two-dimensional profile.

    Args:
        row: Index into the first grid.
        col: Index into the second grid.
        value_a: First interest parameter at this cell.
        value_b: Second interest parameter at this cell.
        profile_value: Minimum of the negative log-likelihood over the
            nuisance block (``inf`` if no finite start existed).
        nuisance: Nuisance optimum (interest coordinates removed).
        converged: Whether the winning optimization met its tolerance.
        evaluations: Objective evaluations spent at this cell.
        start: Winning start: ``"warm"``, ``"estimate"``, ``"fixed"`` (no
            nuisance parameters) or ``"none"`` (failed).
        message: Termination message of the winning optimization.
        order: Position along the serpentine path (0-based).

    References:
        [EQ-INV-PLX-001], [REF-RAUE-ETAL-2009].

    """

    row: int
    col: int
    value_a: float
    value_b: float
    profile_value: float
    nuisance: Vector
    converged: bool
    evaluations: int
    start: str
    message: str
    order: int

    @property
    def status(self) -> str:
        """Return ``"ok"``, ``"not_converged"`` or ``"failed"``.

        Returns:
            ``"failed"`` when the profile value is not finite,
            ``"not_converged"`` when finite but the optimizer did not meet its
            tolerance, else ``"ok"``.

        References:
            [EQ-INV-PLX-001].

        """
        if not math.isfinite(self.profile_value):
            return "failed"
        return "ok" if self.converged else "not_converged"


@dataclass(frozen=True, slots=True)
class Profile2DResult:
    """A two-dimensional profile with baseline and per-cell diagnostics.

    Args:
        indices: The two interest coordinates ``(i, j)``.
        estimate: The supplied estimate.
        nll_at_estimate: Negative log-likelihood at ``estimate``.
        nll_min: ``min(nll_at_estimate, finite profile values)``.
        improved_estimate_found: A profile value is below
            ``nll_at_estimate`` by more than ``improvement_tolerance``.
        improved_estimate: Best full parameter vector when improved, else
            ``None``.
        grid_a: Grid of the first interest parameter.
        grid_b: Grid of the second interest parameter.
        path: Serpentine visiting order as ``(row, col)`` pairs.
        points: Cells in path order.
        likelihood_ratio: ``2 (profile - nll_min)`` as ``len(grid_a)`` rows of
            ``len(grid_b)`` entries; ``inf`` for failed cells.
        n_failed: Cells with no finite profile value.
        n_not_converged: Finite cells whose optimizer did not converge.
        total_evaluations: Objective evaluations, all cells.
        improvement_tolerance: Threshold of the improvement flag.

    References:
        [EQ-INV-PLX-001], [REF-RAUE-ETAL-2009].

    """

    indices: tuple[int, int]
    estimate: Vector
    nll_at_estimate: float
    nll_min: float
    improved_estimate_found: bool
    improved_estimate: Vector | None
    grid_a: Vector
    grid_b: Vector
    path: tuple[tuple[int, int], ...]
    points: tuple[Profile2DPoint, ...]
    likelihood_ratio: tuple[Vector, ...]
    n_failed: int
    n_not_converged: int
    total_evaluations: int
    improvement_tolerance: float

    @property
    def all_ok(self) -> bool:
        """Return whether every cell converged with a finite value.

        Returns:
            ``True`` when there are no failed and no non-converged cells.

        References:
            [EQ-INV-PLX-001].

        """
        return self.n_failed == 0 and self.n_not_converged == 0

    def cell(self, row: int, col: int) -> Profile2DPoint:
        """Return the point at grid indices ``(row, col)``.

        Args:
            row: Index into ``grid_a``.
            col: Index into ``grid_b``.

        Returns:
            The :class:`Profile2DPoint`.

        Raises:
            ValueError: If an index is out of range.

        References:
            [EQ-INV-PLX-001].

        """
        if not (0 <= row < len(self.grid_a) and 0 <= col < len(self.grid_b)):
            raise ValueError("cell index out of range")
        for point in self.points:
            if point.row == row and point.col == col:
                return point
        raise ValueError("cell not found")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ProfileRegion2D:
    """Joint ``chi2(2)`` likelihood-ratio region on a grid.

    Args:
        level: Nominal level.
        threshold: ``chi2_{2, level}`` quantile.
        cells: ``len(grid_a)`` rows of statuses ``"inside"``
            (``LR <= threshold`` and converged), ``"outside"`` (converged,
            ``LR > threshold``) or ``"unreliable"`` (failed or not converged;
            neither inside nor outside is claimed).
        inside_count: Number of inside cells.
        unreliable_count: Number of unreliable cells.
        touches_grid_edge: An inside cell lies on the grid boundary, so the
            region may extend beyond the grid (not a bounded region).
        components: Number of 4-connected inside components.
        area_estimate: Inside-cell count times the cell area for uniform
            grids (first-order, grid-resolution dependent), else ``None``.
        improved_estimate_found: Copied from the profile.

    References:
        [EQ-INV-PLX-002], [REF-RAUE-ETAL-2009].

    """

    level: float
    threshold: float
    cells: tuple[tuple[str, ...], ...]
    inside_count: int
    unreliable_count: int
    touches_grid_edge: bool
    components: int
    area_estimate: float | None
    improved_estimate_found: bool


def profile_likelihood_2d(
    negative_log_likelihood: Callable[[Vector], float],
    estimate: Sequence[float],
    indices: tuple[int, int],
    grid_a: Sequence[float],
    grid_b: Sequence[float],
    *,
    lower: Sequence[float | None] | None = None,
    upper: Sequence[float | None] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
    improvement_tolerance: float = 1e-6,
) -> Profile2DResult:
    """Profile the negative log-likelihood over a grid of two parameters.

    ``nll_prof(a, b) = min_{theta_rest} nll(theta with theta_i = a,
    theta_j = b)``. The grid is walked along a serpentine path: row 0 left to
    right, row 1 right to left, and so on, so consecutive path cells are grid
    neighbours. Each cell is minimized from the previous good cell's nuisance
    optimum (``"warm"``) and from the nuisance block of ``estimate``
    (``"estimate"``); the lower value wins. A failed cell does not replace the
    warm start. Failed and non-converged cells are stored and counted; none is
    interpolated or filled. A value below ``nll(estimate)`` sets
    ``improved_estimate_found`` and the baseline becomes that value (never
    clipped).

    Args:
        negative_log_likelihood: ``theta -> float``; may raise ``ValueError``
            or return a non-finite value for invalid parameters.
        estimate: Supplied estimate of the full parameter vector.
        indices: Two distinct interest coordinates ``(i, j)``.
        grid_a: Strictly increasing values of ``theta_i`` bracketing the
            estimate.
        grid_b: Strictly increasing values of ``theta_j`` bracketing the
            estimate.
        lower: Optional lower bounds per parameter (``None`` for none).
        upper: Optional upper bounds per parameter.
        tolerance: Gradient tolerance of the nuisance optimizations.
        max_iterations: Iteration limit per optimization.
        improvement_tolerance: Absolute decrease (nll units) that counts as
            an improved estimate.

    Returns:
        :class:`Profile2DResult`.

    Raises:
        TypeError: For a non-callable objective or non-real inputs.
        ValueError: For invalid indices, grids or bounds, an estimate outside
            the bounds or with non-finite objective.

    Equation:
        [EQ-INV-PLX-001].

    References:
        [REF-RAUE-ETAL-2009], [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    if not callable(negative_log_likelihood):
        raise TypeError("negative_log_likelihood must be callable")
    theta_hat = as_vector(estimate, name="estimate")
    size = len(theta_hat)
    pair = tuple(indices)
    if len(pair) != 2 or any(
        isinstance(k, bool) or not isinstance(k, int) for k in pair
    ):
        raise TypeError("indices must be two integers")
    ia, ib = pair
    if ia == ib or not (0 <= ia < size and 0 <= ib < size):
        raise ValueError("indices must be two distinct in-range coordinates")
    ga = _check_grid(grid_a, "grid_a", theta_hat[ia])
    gb = _check_grid(grid_b, "grid_b", theta_hat[ib])
    tol = require_finite_real(tolerance, name="tolerance")
    imp_tol = require_finite_real(improvement_tolerance, name="improvement_tolerance")
    if tol <= 0.0 or imp_tol < 0.0:
        raise ValueError("tolerance must be positive, improvement_tolerance >= 0")
    lo = _bounds_vector(lower, size, "lower", -math.inf)
    hi = _bounds_vector(upper, size, "upper", math.inf)
    for j in range(size):
        if not lo[j] < hi[j]:
            raise ValueError("bounds need lower < upper")
        if not lo[j] <= theta_hat[j] <= hi[j]:
            raise ValueError("estimate lies outside the bounds")
    for k, g in ((ia, ga), (ib, gb)):
        if not lo[k] <= g[0] or not g[-1] <= hi[k]:
            raise ValueError("grids must respect the bounds of their parameters")
    nuis = [j for j in range(size) if j not in (ia, ib)]
    box = _box(lo, hi, nuis)
    counter = [0]
    safe = _safe(negative_log_likelihood, counter)
    nll_hat = safe(theta_hat)
    if not math.isfinite(nll_hat):
        raise ValueError("objective is not finite at the supplied estimate")
    nuisance_hat = tuple(theta_hat[j] for j in nuis)

    def assemble(a: float, b: float, nu: Vector) -> Vector:
        """Insert the two interest values into the nuisance vector.

        References:
            [EQ-INV-PLX-001]; implementation helper.

        """
        full = [0.0] * size
        full[ia], full[ib] = a, b
        for pos, j in enumerate(nuis):
            full[j] = nu[pos]
        return tuple(full)

    def point_at(row: int, col: int, order: int, previous: Vector) -> Profile2DPoint:
        """Profile one cell with warm and estimate starts.

        References:
            [EQ-INV-PLX-001]; implementation helper.

        """
        a, b = ga[row], gb[col]
        before = counter[0]
        if not nuis:
            value = safe(assemble(a, b, ()))
            finite = math.isfinite(value)
            return Profile2DPoint(
                row, col, a, b, value, (), finite, counter[0] - before, "fixed",
                "no nuisance parameters" if finite else "objective not finite",
                order,
            )  # fmt: skip

        def inner(nu: Vector) -> float:
            """Objective in the nuisance coordinates.

            References:
                [EQ-INV-PLX-001]; implementation helper.

            """
            return safe(assemble(a, b, nu))

        candidates: list[tuple[str, Vector]] = [("warm", previous)]
        if nuisance_hat != previous:
            candidates.append(("estimate", nuisance_hat))
        best: tuple[float, Vector, bool, str, str] | None = None
        for label, start in candidates:
            out = _minimize(inner, start, box, tol, max_iterations)
            if out is not None and (best is None or out[0] < best[0]):
                best = (out[0], out[1], out[2], out[3], label)
        spent = counter[0] - before
        if best is None:
            return Profile2DPoint(
                row, col, a, b, math.inf, previous, False, spent, "none",
                "no finite start point", order,
            )  # fmt: skip
        return Profile2DPoint(
            row, col, a, b, best[0], best[1], best[2], spent, best[4], best[3], order
        )

    path: list[tuple[int, int]] = []
    for row in range(len(ga)):
        cols = range(len(gb)) if row % 2 == 0 else range(len(gb) - 1, -1, -1)
        path.extend((row, col) for col in cols)
    previous = nuisance_hat
    points: list[Profile2DPoint] = []
    for order, (row, col) in enumerate(path):
        point = point_at(row, col, order, previous)
        points.append(point)
        if math.isfinite(point.profile_value):
            previous = point.nuisance
    finite = [p for p in points if math.isfinite(p.profile_value)]
    if finite:
        best_point = min(finite, key=lambda p: p.profile_value)
        improved = best_point.profile_value < nll_hat - imp_tol
        nll_min = min(nll_hat, best_point.profile_value)
        improved_vec = (
            assemble(best_point.value_a, best_point.value_b, best_point.nuisance)
            if improved
            else None
        )
    else:
        improved, nll_min, improved_vec = False, nll_hat, None
    ratio_rows = [[math.inf] * len(gb) for _ in ga]
    for p in points:
        if math.isfinite(p.profile_value):
            ratio_rows[p.row][p.col] = 2.0 * (p.profile_value - nll_min)
    return Profile2DResult(
        indices=(ia, ib),
        estimate=theta_hat,
        nll_at_estimate=nll_hat,
        nll_min=nll_min,
        improved_estimate_found=improved,
        improved_estimate=improved_vec,
        grid_a=ga,
        grid_b=gb,
        path=tuple(path),
        points=tuple(points),
        likelihood_ratio=tuple(tuple(r) for r in ratio_rows),
        n_failed=sum(1 for p in points if p.status == "failed"),
        n_not_converged=sum(1 for p in points if p.status == "not_converged"),
        total_evaluations=counter[0],
        improvement_tolerance=imp_tol,
    )


def _uniform_step(grid: Vector) -> float | None:
    """Return the common step of a uniform grid or ``None``.

    References:
        [EQ-INV-PLX-002]; implementation helper.

    """
    steps = [b - a for a, b in zip(grid, grid[1:], strict=False)]
    mean = sum(steps) / len(steps)
    if all(abs(s - mean) <= 1e-9 * abs(mean) for s in steps):
        return mean
    return None


def profile_region_2d(result: Profile2DResult, level: float = 0.95) -> ProfileRegion2D:
    """Threshold a 2D profile at the ``chi2(2)`` quantile.

    The joint likelihood-ratio region is ``{(a, b) : 2 (l_max - l_prof(a, b))
    <= chi2_{2, level}}``. Cells whose profile failed or did not converge are
    classed ``"unreliable"``: their likelihood ratio is unknown or an upper
    bound only, so neither membership is claimed. The calibration is
    asymptotic and regular (interior estimate, identifiable pair, smooth
    likelihood, large sample); a region touching the grid edge is reported as
    possibly unbounded within the grid rather than closed off.

    Args:
        result: Output of :func:`profile_likelihood_2d` for a true (not
            penalized) negative log-likelihood.
        level: Confidence level in ``(0, 1)``.

    Returns:
        :class:`ProfileRegion2D`.

    Raises:
        TypeError: If ``result`` has the wrong type.
        ValueError: If ``level`` is outside ``(0, 1)``.

    Equation:
        [EQ-INV-PLX-002].

    References:
        [REF-RAUE-ETAL-2009], [REF-SEBER-WILD-2003].

    """
    if not isinstance(result, Profile2DResult):
        raise TypeError("result must be a Profile2DResult")
    lev = require_finite_real(level, name="level")
    if not 0.0 < lev < 1.0:
        raise ValueError("level must lie in (0, 1)")
    threshold = ChiSquare(2.0).quantile(lev)
    na, nb = len(result.grid_a), len(result.grid_b)
    cells = [["outside"] * nb for _ in range(na)]
    for p in result.points:
        if p.status != "ok":
            cells[p.row][p.col] = "unreliable"
        elif result.likelihood_ratio[p.row][p.col] <= threshold:
            cells[p.row][p.col] = "inside"
    inside = {(r, c) for r in range(na) for c in range(nb) if cells[r][c] == "inside"}
    edge = any(r in (0, na - 1) or c in (0, nb - 1) for r, c in inside)
    seen: set[tuple[int, int]] = set()
    components = 0
    for seed_cell in sorted(inside):
        if seed_cell in seen:
            continue
        components += 1
        stack = [seed_cell]
        seen.add(seed_cell)
        while stack:
            r, c = stack.pop()
            for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nxt in inside and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    ha, hb = _uniform_step(result.grid_a), _uniform_step(result.grid_b)
    area = len(inside) * ha * hb if ha is not None and hb is not None else None
    return ProfileRegion2D(
        level=lev,
        threshold=threshold,
        cells=tuple(tuple(row) for row in cells),
        inside_count=len(inside),
        unreliable_count=sum(row.count("unreliable") for row in cells),
        touches_grid_edge=edge,
        components=components,
        area_estimate=area,
        improved_estimate_found=result.improved_estimate_found,
    )


# ---------------------------------------------------------------------------
# 2. parametric bootstrap calibration of the profile LR threshold
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapCalibrationResult:
    """Bootstrap distribution of the profile likelihood-ratio statistic.

    Args:
        indices: Interest coordinates held at their fitted values.
        dof: Number of interest coordinates (``chi2`` degrees of freedom).
        level: Nominal level.
        replicates: Requested number of simulated data sets.
        valid_replicates: Replicates with both fits converged.
        failed_replicates: Replicates dropped because a fit failed or did not
            converge (never replaced or imputed).
        baseline_refits: Valid replicates in which the constrained fit beat
            the global fit, so the baseline was the constrained value.
        statistics: Sorted bootstrap statistics ``LR*`` of the valid
            replicates.
        quantile_rank: Order-statistic rank ``ceil(level (n + 1))``.
        quantile: Empirical ``level`` quantile (``statistics[rank - 1]``) or
            ``None`` when the rank exceeds the number of valid replicates.
        quantile_ci: Approximate 95 % distribution-free interval for the
            quantile from binomial order-statistic ranks; an end is ``None``
            when its rank lies outside the sample.
        quantile_se: ``(ci_upper - ci_lower) / (2 * 1.96)`` or ``None``.
        chi2_threshold: ``chi2_{dof, level}`` quantile.
        coverage_at_chi2: Fraction of valid statistics ``<= chi2_threshold``
            (the realized level of the asymptotic test).
        coverage_se: Binomial Monte Carlo standard error of that fraction.
        discrepancy_in_se: ``(quantile - chi2_threshold) / quantile_se`` or
            ``None``.
        chi2_within_ci: Whether ``chi2_threshold`` lies inside
            ``quantile_ci`` (``None`` when an end is missing).
        seed: Root seed of the stream.

    References:
        [EQ-INV-PLX-003], [EQ-INV-PLX-004], [REF-DAVISON-HINKLEY-1997].

    """

    indices: tuple[int, ...]
    dof: int
    level: float
    replicates: int
    valid_replicates: int
    failed_replicates: int
    baseline_refits: int
    statistics: Vector
    quantile_rank: int
    quantile: float | None
    quantile_ci: tuple[float | None, float | None]
    quantile_se: float | None
    chi2_threshold: float
    coverage_at_chi2: float
    coverage_se: float
    discrepancy_in_se: float | None
    chi2_within_ci: bool | None
    seed: int


def bootstrap_lr_calibration(
    simulate: Callable[[Vector, random.Random], _Data],
    nll_factory: Callable[[_Data], Callable[[Vector], float]],
    estimate: Sequence[float],
    indices: Sequence[int],
    *,
    replicates: int = 200,
    level: float = 0.95,
    seed: int | RandomStream | None = 0,
    lower: Sequence[float | None] | None = None,
    upper: Sequence[float | None] | None = None,
    refit_starts: Sequence[Sequence[float]] = (),
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> BootstrapCalibrationResult:
    """Calibrate the profile likelihood-ratio threshold by parametric bootstrap.

    For replicate ``b`` (stream ``seed/"replicate"/b``): draw data
    ``D_b ~ model(estimate)``, build ``nll_b = nll_factory(D_b)``, minimize
    ``nll_b`` over all parameters (starts: ``estimate`` and ``refit_starts``)
    for the global value ``g_b``, minimize it with the interest coordinates
    fixed at ``estimate`` for the constrained value ``c_b``, and record
    ``LR*_b = 2 (c_b - min(g_b, c_b))`` (a better constrained value is not
    clipped but taken as baseline and counted in ``baseline_refits``). The
    simulated truth of the interest coordinates is their fitted value, so
    ``LR*`` is the null distribution of the test statistic. Its
    ``level``-quantile (order statistic ``ceil(level (n + 1))``) replaces the
    asymptotic ``chi2_{dof, level}``. A replicate whose global or constrained
    fit did not converge is dropped and counted in ``failed_replicates``; the
    dropped fraction must be inspected before trusting the quantile (dropping
    can bias it).

    The bootstrap quantile has Monte Carlo error, reported through
    distribution-free order-statistic ranks; it also inherits the bias of
    simulating from an estimate rather than the truth.

    Args:
        simulate: ``(theta, rng) -> data``; ``rng`` is a ``random.Random``
            owned by this replicate.
        nll_factory: ``data -> nll`` returning the negative log-likelihood
            callable of a data set (may raise ``ValueError`` or return a
            non-finite value for invalid parameters).
        estimate: Fitted parameter vector used as the simulation truth.
        indices: Distinct interest coordinates fixed under the null.
        replicates: Number of simulated data sets (at least 20).
        level: Level in ``(0, 1)``.
        seed: Integer seed, :class:`RandomStream` or ``None`` (not
            reproducible).
        lower: Optional lower bounds per parameter.
        upper: Optional upper bounds per parameter.
        refit_starts: Extra full-length starts of the global fit.
        tolerance: Gradient tolerance of the fits.
        max_iterations: Iteration limit per fit.

    Returns:
        :class:`BootstrapCalibrationResult`.

    Raises:
        TypeError: For non-callables or non-real inputs.
        ValueError: For invalid indices, bounds, level or replicate count, or
            an estimate outside the bounds.

    Equation:
        [EQ-INV-PLX-003], [EQ-INV-PLX-004].

    References:
        [REF-DAVISON-HINKLEY-1997], [REF-SELF-LIANG-1987],
        [REF-RAUE-ETAL-2009].

    """
    if not callable(simulate) or not callable(nll_factory):
        raise TypeError("simulate and nll_factory must be callable")
    theta_hat = as_vector(estimate, name="estimate")
    size = len(theta_hat)
    idx = tuple(indices)
    if not idx or any(isinstance(k, bool) or not isinstance(k, int) for k in idx):
        raise TypeError("indices must be a non-empty sequence of integers")
    if len(set(idx)) != len(idx) or any(not 0 <= k < size for k in idx):
        raise ValueError("indices must be distinct and in range")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("replicates must be an integer")
    if replicates < 20:
        raise ValueError("replicates must be at least 20")
    lev = require_finite_real(level, name="level")
    if not 0.0 < lev < 1.0:
        raise ValueError("level must lie in (0, 1)")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    lo = _bounds_vector(lower, size, "lower", -math.inf)
    hi = _bounds_vector(upper, size, "upper", math.inf)
    for j in range(size):
        if not lo[j] < hi[j]:
            raise ValueError("bounds need lower < upper")
        if not lo[j] <= theta_hat[j] <= hi[j]:
            raise ValueError("estimate lies outside the bounds")
    extra = tuple(as_vector(s, name="refit_starts") for s in refit_starts)
    if any(len(s) != size for s in extra):
        raise ValueError("each refit start needs one entry per parameter")
    nuis = [j for j in range(size) if j not in idx]
    nuisance_hat = tuple(theta_hat[j] for j in nuis)
    stream = resolve_stream(seed)

    def constrained(nll: Callable[[Vector], float]) -> float | None:
        """Return the minimum with the interest coordinates fixed, or ``None``.

        References:
            [EQ-INV-PLX-003]; implementation helper.

        """
        if not nuis:
            value = nll(theta_hat)
            return value if math.isfinite(value) else None

        def inner(nu: Vector) -> float:
            """Objective in the nuisance coordinates.

            References:
                [EQ-INV-PLX-003]; implementation helper.

            """
            full = list(theta_hat)
            for pos, j in enumerate(nuis):
                full[j] = nu[pos]
            return nll(tuple(full))

        out = _minimize(inner, nuisance_hat, _box(lo, hi, nuis), tol, max_iterations)
        return out[0] if out is not None and out[2] else None

    def global_min(nll: Callable[[Vector], float]) -> float | None:
        """Return the best converged global minimum over the starts, or ``None``.

        References:
            [EQ-INV-PLX-003]; implementation helper.

        """
        best: float | None = None
        for start in (theta_hat, *extra):
            out = _minimize(nll, start, _box(lo, hi, range(size)), tol, max_iterations)
            if out is not None and out[2] and (best is None or out[0] < best):
                best = out[0]
        return best

    stats: list[float] = []
    failed = 0
    refits = 0
    for b in range(replicates):
        rng = stream.child("replicate", b).generator()
        data = simulate(theta_hat, rng)
        nll = _safe(nll_factory(data), [0])
        c_val, g_val = constrained(nll), global_min(nll)
        if c_val is None or g_val is None:
            failed += 1
            continue
        if c_val < g_val:
            refits += 1
        stats.append(2.0 * (c_val - min(c_val, g_val)))
    stats.sort()
    n = len(stats)
    dof = len(idx)
    chi2_thr = ChiSquare(float(dof)).quantile(lev)
    rank = math.ceil(lev * (n + 1)) if n else 0
    quantile = stats[rank - 1] if 1 <= rank <= n else None
    ci: tuple[float | None, float | None] = (None, None)
    se: float | None = None
    if n:
        half = _Z95 * math.sqrt(n * lev * (1.0 - lev))
        r_lo = max(1, math.floor(n * lev - half))
        r_hi = math.ceil(n * lev + half) + 1
        ci = (stats[r_lo - 1], stats[r_hi - 1] if r_hi <= n else None)
        if ci[1] is not None:
            se = (ci[1] - ci[0]) / (2.0 * _Z95)
    cover = sum(1 for s in stats if s <= chi2_thr) / n if n else math.nan
    cover_se = math.sqrt(cover * (1.0 - cover) / n) if n else math.nan
    ci_lo, ci_hi = ci
    discrepancy = (
        (quantile - chi2_thr) / se
        if quantile is not None and se is not None and se > 0.0
        else None
    )
    within = (
        ci_lo <= chi2_thr <= ci_hi if ci_lo is not None and ci_hi is not None else None
    )
    return BootstrapCalibrationResult(
        indices=idx,
        dof=dof,
        level=lev,
        replicates=replicates,
        valid_replicates=n,
        failed_replicates=failed,
        baseline_refits=refits,
        statistics=tuple(stats),
        quantile_rank=rank,
        quantile=quantile,
        quantile_ci=ci,
        quantile_se=se,
        chi2_threshold=chi2_thr,
        coverage_at_chi2=cover,
        coverage_se=cover_se,
        discrepancy_in_se=discrepancy,
        chi2_within_ci=within,
        seed=stream.seed,
    )


# ---------------------------------------------------------------------------
# 3. ODE forward sensitivities and identifiability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ODESensitivityResult:
    """Sensitivity matrix of an ODE model and its identifiability diagnosis.

    Args:
        times: Observation times.
        states: Full state at each observation time.
        observations: ``C y`` at each observation time (one tuple per time).
        sensitivity: Raw sensitivity matrix ``d(C y)/d theta`` with
            ``len(times) * q`` rows ordered time-major, one column per
            parameter.
        scaled_sensitivity: ``sensitivity`` with rows divided by the noise
            standard deviation of their observable and columns multiplied by
            the parameter scale; the matrix whose singular values are used.
        singular_values: All ``p`` singular values (descending; zeros are
            exact when the matrix is rank deficient or has fewer rows).
        right_singular_vectors: ``p x p`` matrix ``V`` (columns are the
            parameter-space directions, in scaled coordinates).
        relative_tolerance: The rank rule ``sigma_k > relative_tolerance *
            sigma_1``.
        rank_tolerance: Absolute cut ``relative_tolerance * sigma_1``.
        rank: Numerical rank under that rule.
        rank_gap: ``sigma_rank / sigma_{rank+1}`` (``inf`` when the next
            value is zero or there is none); a gap near 1 means the rank
            decision is fragile.
        identifiable: ``rank == p`` (local, numerical; not structural).
        null_directions: Unit vectors of the numerically unidentifiable
            subspace (columns ``rank..p-1`` of ``V``), in scaled coordinates;
            a basis only, since equal singular values leave it arbitrary.
        std_error_scale: ``1 / sigma_k`` per direction (``inf`` for the
            numerically null ones): the standard error of the data along
            ``V[:, k]`` when the noise weights are the true standard
            deviations.
        condition_number: ``sigma_1 / sigma_p`` (``inf`` when ``sigma_p`` is
            zero).
        steps_per_interval: RK4 steps per observation interval.

    References:
        [EQ-INV-PLX-005], [EQ-INV-PLX-006].

    """

    times: Vector
    states: tuple[Vector, ...]
    observations: tuple[Vector, ...]
    sensitivity: Matrix
    scaled_sensitivity: Matrix
    singular_values: Vector
    right_singular_vectors: Matrix
    relative_tolerance: float
    rank_tolerance: float
    rank: int
    rank_gap: float
    identifiable: bool
    null_directions: tuple[Vector, ...]
    std_error_scale: Vector
    condition_number: float
    steps_per_interval: int


def _fd_jacobian(
    function: Callable[[Vector], Sequence[float]], point: Vector, rows: int
) -> list[list[float]]:
    """Central-difference Jacobian of a vector function (``rows`` outputs).

    References:
        [EQ-INV-PLX-005]; implementation helper.

    """
    jac = [[0.0] * len(point) for _ in range(rows)]
    for k in range(len(point)):
        h = _EPS ** (1.0 / 3.0) * max(1.0, abs(point[k]))
        up = list(point)
        dn = list(point)
        up[k] += h
        dn[k] -= h
        f_up, f_dn = function(tuple(up)), function(tuple(dn))
        for i in range(rows):
            jac[i][k] = (f_up[i] - f_dn[i]) / (2.0 * h)
    return jac


def ode_sensitivity(
    rhs: OdeRhs,
    theta: Sequence[float],
    y0: Sequence[float],
    times: Sequence[float],
    *,
    t0: float = 0.0,
    observation: Sequence[Sequence[float]] | None = None,
    initial_sensitivity: Sequence[Sequence[float]] | None = None,
    jacobian_y: Callable[[float, Vector, Vector], Sequence[Sequence[float]]]
    | None = None,
    jacobian_theta: Callable[[float, Vector, Vector], Sequence[Sequence[float]]]
    | None = None,
    steps_per_interval: int = 20,
    noise_std: float | Sequence[float] = 1.0,
    parameter_scale: Sequence[float] | None = None,
    relative_tolerance: float = 1e-8,
) -> ODESensitivityResult:
    """Integrate forward sensitivities and diagnose local identifiability.

    For ``y' = f(t, y, theta)`` with ``S = dy/dtheta`` the sensitivity
    equations ``S' = (df/dy) S + df/dtheta``, ``S(t0) = dy0/dtheta`` are
    integrated together with ``y`` as one augmented system of dimension
    ``n (1 + p)`` by the classical Runge-Kutta method on each interval between
    consecutive times of ``(t0, *times)`` (``steps_per_interval`` steps each).
    The Jacobians are analytic when ``jacobian_y`` / ``jacobian_theta`` are
    given and central finite differences of ``rhs`` otherwise.

    With observation matrix ``C`` the stacked matrix ``d(C y(t_k))/d theta``
    is divided row-wise by the noise standard deviations and column-wise
    scaled by ``parameter_scale``. Its singular values ``sigma_1 >= ... >=
    sigma_p`` give the numerical rank ``#{k : sigma_k > relative_tolerance
    sigma_1}``; right singular vectors beyond the rank span the numerically
    unidentifiable subspace, in the scaled coordinates. The tolerance rule is
    explicit and relative; it must exceed the integration error ratio (RK4 is
    fourth order in the step), and the finite-difference Jacobian default
    carries an error near ``1e-10``, so ``1e-8`` is a floor, not a guarantee.

    This is local to ``theta``: rank deficiency shows local non-identifiability
    at that point; full rank proves neither structural nor practical
    identifiability (small ``sigma_k`` means a large error ``1/sigma_k`` along
    ``V[:, k]``, which is a noise-dependent, practical statement).

    Args:
        rhs: ``f(t, y, theta) -> sequence`` of length ``n``.
        theta: Parameter vector (``p`` entries).
        y0: Initial state at ``t0`` (``n`` entries; independent of ``theta``
            unless ``initial_sensitivity`` says otherwise).
        times: Strictly increasing observation times, all greater than ``t0``.
        t0: Initial time.
        observation: ``q x n`` matrix ``C`` (default identity).
        initial_sensitivity: ``n x p`` matrix ``dy0/dtheta`` (default zeros).
        jacobian_y: ``(t, y, theta) -> n x n`` matrix ``df/dy``.
        jacobian_theta: ``(t, y, theta) -> n x p`` matrix ``df/dtheta``.
        steps_per_interval: RK4 steps per observation interval (positive).
        noise_std: Positive scalar or length-``q`` observation standard
            deviations.
        parameter_scale: Positive length-``p`` column scales (default ones).
        relative_tolerance: Relative rank tolerance in ``(0, 1)``.

    Returns:
        :class:`ODESensitivityResult`.

    Raises:
        TypeError: For non-callable ``rhs`` or non-real inputs.
        ValueError: For invalid shapes, times, tolerances or scales, a
            right-hand side of the wrong length, or a state that becomes
            non-finite during integration.

    Equation:
        [EQ-INV-PLX-005], [EQ-INV-PLX-006].

    References:
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1],
        [REF-SEBER-WILD-2003].

    """
    if not callable(rhs):
        raise TypeError("rhs must be callable")
    th = as_vector(theta, name="theta")
    y_start = as_vector(y0, name="y0")
    p, n = len(th), len(y_start)
    t_start = require_finite_real(t0, name="t0")
    obs_times = as_vector(times, name="times")
    previous = t_start
    for value in obs_times:
        if not value > previous:
            raise ValueError("times must be strictly increasing and greater than t0")
        previous = value
    if isinstance(steps_per_interval, bool) or not isinstance(steps_per_interval, int):
        raise TypeError("steps_per_interval must be an integer")
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be positive")
    rtol = require_finite_real(relative_tolerance, name="relative_tolerance")
    if not 0.0 < rtol < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    c_mat = (
        tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))
        if observation is None
        else as_matrix(observation, name="observation")
    )
    q = len(c_mat)
    if any(len(row) != n for row in c_mat):
        raise ValueError("observation must have one column per state")
    s0 = (
        tuple((0.0,) * p for _ in range(n))
        if initial_sensitivity is None
        else as_matrix(initial_sensitivity, name="initial_sensitivity")
    )
    if len(s0) != n or any(len(row) != p for row in s0):
        raise ValueError("initial_sensitivity must be n x p")
    if isinstance(noise_std, int | float) and not isinstance(noise_std, bool):
        sigmas = (float(noise_std),) * q
    else:
        sigmas = as_vector(noise_std, name="noise_std")  # type: ignore[arg-type]
    if len(sigmas) != q or any(not s > 0.0 for s in sigmas):
        raise ValueError("noise_std must be positive, scalar or one per observable")
    scale = (
        (1.0,) * p
        if parameter_scale is None
        else as_vector(parameter_scale, name="parameter_scale")
    )
    if len(scale) != p or any(not s > 0.0 for s in scale):
        raise ValueError("parameter_scale must be positive, one per parameter")

    def drift(t: float, y: Vector) -> Vector:
        """Evaluate ``f`` with length validation.

        References:
            [EQ-INV-PLX-005]; implementation helper.

        """
        out = tuple(float(v) for v in rhs(t, y, th))
        if len(out) != n:
            raise ValueError("rhs must return one entry per state")
        return out

    def jac_y(t: float, y: Vector) -> Sequence[Sequence[float]]:
        """Return ``df/dy`` (analytic or central differences).

        References:
            [EQ-INV-PLX-005]; implementation helper.

        """
        if jacobian_y is not None:
            return jacobian_y(t, y, th)
        return _fd_jacobian(lambda v: drift(t, v), y, n)

    def jac_theta(t: float, y: Vector) -> Sequence[Sequence[float]]:
        """Return ``df/dtheta`` (analytic or central differences).

        References:
            [EQ-INV-PLX-005]; implementation helper.

        """
        if jacobian_theta is not None:
            return jacobian_theta(t, y, th)
        return _fd_jacobian(lambda v: tuple(rhs(t, y, v)), th, n)

    def augmented(t: float, z: Sequence[float]) -> Vector:
        """Right-hand side of ``(y, S)`` with ``S`` stored row-major.

        References:
            [EQ-INV-PLX-005]; implementation helper.

        """
        y = tuple(z[:n])
        jy, jt = jac_y(t, y), jac_theta(t, y)
        slope = list(drift(t, y))
        for i in range(n):
            for k in range(p):
                slope.append(
                    math.fsum(jy[i][m] * z[n + m * p + k] for m in range(n)) + jt[i][k]
                )
        return tuple(slope)

    z: Vector = y_start + tuple(v for row in s0 for v in row)
    now = t_start
    states: list[Vector] = []
    observed: list[Vector] = []
    raw_rows: list[Vector] = []
    for target in obs_times:
        traj = integrate_fixed_step(
            augmented,
            now,
            z,
            step=(target - now) / steps_per_interval,
            steps=steps_per_interval,
            method="rk4",
        )
        z = tuple(traj.final_state)
        now = target
        y_now = z[:n]
        states.append(y_now)
        observed.append(
            tuple(math.fsum(c_mat[i][j] * y_now[j] for j in range(n)) for i in range(q))
        )
        for i in range(q):
            raw_rows.append(
                tuple(
                    math.fsum(c_mat[i][m] * z[n + m * p + k] for m in range(n))
                    for k in range(p)
                )
            )
    sens: Matrix = tuple(raw_rows)
    weights = tuple(sigmas[r % q] for r in range(len(sens)))
    scaled: Matrix = tuple(
        tuple(sens[r][k] * scale[k] / weights[r] for k in range(p))
        for r in range(len(sens))
    )
    padded = scaled + tuple((0.0,) * p for _ in range(max(0, p - len(scaled))))
    svd = singular_value_decomposition(padded)
    sigma = tuple(svd.singular_values)
    cut = rtol * sigma[0]
    rank = sum(1 for s in sigma if s > cut)
    v_cols = tuple(tuple(svd.v[i][k] for i in range(p)) for k in range(p))
    gap = (
        sigma[rank - 1] / sigma[rank]
        if 0 < rank < p and sigma[rank] > 0.0
        else math.inf
    )
    return ODESensitivityResult(
        times=obs_times,
        states=tuple(states),
        observations=tuple(observed),
        sensitivity=sens,
        scaled_sensitivity=scaled,
        singular_values=sigma,
        right_singular_vectors=tuple(svd.v[i][:p] for i in range(p)),
        relative_tolerance=rtol,
        rank_tolerance=cut,
        rank=rank,
        rank_gap=gap,
        identifiable=rank == p,
        null_directions=v_cols[rank:],
        std_error_scale=tuple(
            1.0 / sigma[k] if k < rank else math.inf for k in range(p)
        ),
        condition_number=sigma[0] / sigma[-1] if sigma[-1] > 0.0 else math.inf,
        steps_per_interval=steps_per_interval,
    )
