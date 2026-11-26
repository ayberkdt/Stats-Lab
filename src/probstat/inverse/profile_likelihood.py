"""Profile likelihood, Wilks intervals and nuisance-corrected information.

Experimental (added in 0.28).

For a negative log-likelihood ``nll(theta)`` and a target coordinate
``theta_i``, the *profile* is ``nll_prof(t) = min_{theta_-i} nll(t, theta_-i)``.
:func:`profile_likelihood` evaluates it on a user grid by continuation: it
starts at the grid point nearest the supplied estimate and walks outward in
both directions, warm-starting every nuisance optimization from the previous
optimum, and also tries the nuisance block of the supplied estimate as a
second start; extra user starts are used only when neither converged.
Nothing is clipped: a profile value below ``nll(estimate)`` is reported as
``improved_estimate_found`` and the likelihood ratio is measured against the
better baseline.

:func:`profile_interval` applies Wilks' calibration
``2 (l_max - l_prof(t)) <= chi2_{1, level}``. The calibration is asymptotic
and *regular*: it is unreliable at parameter-space boundaries, under weak
(practical) or structural non-identifiability, for multimodal likelihoods and
for small samples. A profile that does not cross the threshold inside the grid
gives an endpoint ``None`` with status ``"unbounded_within_grid"`` (the
interval may be genuinely unbounded or merely longer than the grid; the grid
alone cannot tell) and never a finite fabricated limit. The profile of a
penalized or MAP objective is not a confidence profile and must not be passed
here. Numerical profiles cannot prove structural identifiability.

:func:`efficient_information` forms the Schur complement of the nuisance
block by Cholesky solves; :func:`delta_method_variance` and
:func:`functional_estimability` handle linear functionals of the parameters.

References:
    [REF-RAUE-ETAL-2009], [REF-SEBER-WILD-2003], [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_solve,
    singular_value_decomposition,
)
from probstat.optimization import minimize_bfgs

_EPS = 2.0**-52


@dataclass(frozen=True, slots=True)
class ProfilePoint:
    """One grid point of a profile.

    Args:
        value: Fixed value ``t`` of the target parameter.
        profile_value: ``min`` of the negative log-likelihood over the
            nuisance parameters (``inf`` if no finite start was found).
        nuisance: Nuisance optimum (target removed), a tuple.
        converged: Whether the winning optimization met its tolerance (always
            ``True`` when there is no nuisance parameter and the value is
            finite).
        evaluations: Objective evaluations spent at this point, all starts.
        start: Label of the winning start: ``"warm"``, ``"estimate"``,
            ``"extra[k]"`` or ``"none"`` / ``"fixed"``.
        message: Termination message of the winning optimization.

    References:
        [EQ-INV-PROF-001], [REF-RAUE-ETAL-2009].

    """

    value: float
    profile_value: float
    nuisance: Vector
    converged: bool
    evaluations: int
    start: str
    message: str


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodResult:
    """A computed profile with its baseline and diagnostics.

    Args:
        index: Target coordinate.
        estimate: The user-supplied estimate.
        nll_at_estimate: Negative log-likelihood at ``estimate``.
        nll_min: Baseline ``min(nll_at_estimate, all profile values)``.
        improved_estimate_found: ``True`` when some profile value is below
            ``nll_at_estimate`` by more than ``improvement_tolerance``; the
            supplied estimate is then not a global minimum.
        improved_estimate: Full parameter vector of the best profile point
            (target and nuisance) when improved, else ``None``.
        points: Profile points in grid order.
        likelihood_ratio: ``2 (profile_value - nll_min)`` per point.
        all_converged: Whether every point converged.
        total_evaluations: Objective evaluations, all points.
        improvement_tolerance: Threshold used for the improvement flag.

    References:
        [EQ-INV-PROF-001], [EQ-INV-PROF-002], [REF-RAUE-ETAL-2009].

    """

    index: int
    estimate: Vector
    nll_at_estimate: float
    nll_min: float
    improved_estimate_found: bool
    improved_estimate: Vector | None
    points: tuple[ProfilePoint, ...]
    likelihood_ratio: tuple[float, ...]
    all_converged: bool
    total_evaluations: int
    improvement_tolerance: float

    @property
    def grid(self) -> Vector:
        """Return the grid values of the target parameter.

        Returns:
            Grid values in increasing order.

        References:
            [EQ-INV-PROF-001].

        """
        return tuple(p.value for p in self.points)

    @property
    def profile_values(self) -> Vector:
        """Return the profile negative log-likelihood per grid point.

        Returns:
            Profile values in grid order.

        References:
            [EQ-INV-PROF-001].

        """
        return tuple(p.profile_value for p in self.points)


@dataclass(frozen=True, slots=True)
class ProfileInterval:
    """Wilks likelihood-ratio interval from a grid profile.

    Args:
        level: Nominal level.
        threshold: ``chi2_{1, level}`` quantile.
        lower: Lower endpoint or ``None``.
        upper: Upper endpoint or ``None``.
        lower_status: ``"bounded"``, ``"unbounded_within_grid"`` or
            ``"undetermined_nonfinite"`` (the profile is not finite at the
            neighbouring grid point, so no interpolation is possible).
        upper_status: Same for the upper side.
        connected: ``False`` when further grid points beyond a crossing fall
            back below the threshold, i.e. the set is not one interval
            (multimodality); the reported interval is the component around
            the baseline.
        other_regions: Grid values (outside the component) below threshold.
        all_converged: Whether every profile point converged.
        improved_estimate_found: Copied from the profile result.

    References:
        [EQ-INV-PROF-003], [REF-RAUE-ETAL-2009].

    """

    level: float
    threshold: float
    lower: float | None
    upper: float | None
    lower_status: str
    upper_status: str
    connected: bool
    other_regions: Vector
    all_converged: bool
    improved_estimate_found: bool


@dataclass(frozen=True, slots=True)
class EstimabilityResult:
    """Whether a linear functional is determined by the data.

    Args:
        estimable: ``True`` when the null-space component of ``c`` is within
            ``tolerance`` relative to ``||c||``.
        null_component_norm: Norm of the component of ``c`` in null(M).
        relative_null_component: That norm divided by ``||c||``.
        rank: Numerical rank of the matrix.
        null_dimension: ``n - rank``.
        rank_tolerance: Absolute singular-value cut used for the rank.
        tolerance: Relative tolerance used for the decision.

    References:
        [EQ-INV-PROF-005], [REF-SEBER-WILD-2003].

    """

    estimable: bool
    null_component_norm: float
    relative_null_component: float
    rank: int
    null_dimension: int
    rank_tolerance: float
    tolerance: float


def _invalid_to_inf(
    function: Callable[[Vector], float], counter: list[int]
) -> Callable[[Vector], float]:
    """Wrap ``function`` so invalid points give ``inf`` and calls are counted.

    References:
        [EQ-INV-PROF-001]; implementation helper.

    """

    def wrapped(x: Vector) -> float:
        """Evaluate with accounting.

        References:
            [EQ-INV-PROF-001]; implementation helper.

        """
        counter[0] += 1
        try:
            value = float(function(x))
        except (ValueError, ArithmeticError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    return wrapped


def _bounds_vector(
    bound: Sequence[float | None] | None, size: int, name: str, fill: float
) -> tuple[float, ...]:
    """Validate an optional per-coordinate bound vector.

    References:
        [EQ-INV-PROF-001]; implementation helper.

    """
    if bound is None:
        return (fill,) * size
    items = tuple(bound)
    if len(items) != size:
        raise ValueError(f"{name} must have one entry per parameter")
    out: list[float] = []
    for item in items:
        if item is None:
            out.append(fill)
            continue
        out.append(_real(item, name))
    return tuple(out)


def _real(value: object, name: str) -> float:
    """Return a real number (infinity allowed) or raise.

    References:
        [EQ-INV-PROF-001]; implementation helper.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} entries must be real numbers")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} entries must not be NaN")
    return result


def profile_likelihood(
    negative_log_likelihood: Callable[[Vector], float],
    estimate: Sequence[float],
    index: int,
    grid: Sequence[float],
    *,
    lower: Sequence[float | None] | None = None,
    upper: Sequence[float | None] | None = None,
    optimizer: str = "bfgs",
    starts: Sequence[Sequence[float]] = (),
    tolerance: float = 1e-6,
    max_iterations: int = 200,
    improvement_tolerance: float = 1e-6,
) -> ProfileLikelihoodResult:
    """Compute a profile negative log-likelihood by two-sided continuation.

    ``nll_prof(t) = min_{theta_-i} nll(theta with theta_i = t)``. The walk
    starts at the grid point nearest ``estimate[index]`` and proceeds to both
    ends. At each point the nuisance minimization is started from the
    previous point's optimum (``"warm"``) and from the nuisance block of
    ``estimate`` (``"estimate"``); the lower value wins. Only if neither
    converged are the ``starts`` tried. A profile value below
    ``nll(estimate) - improvement_tolerance`` sets
    ``improved_estimate_found``; the baseline becomes that smaller value
    (never clipped), and the better full parameter vector is returned.

    Args:
        negative_log_likelihood: ``theta -> float``; may raise ``ValueError``
            or return a non-finite value for invalid parameters.
        estimate: Supplied maximum-likelihood estimate (re-checked here).
        index: Target coordinate.
        grid: Strictly increasing target values bracketing ``estimate[index]``.
        lower: Optional lower bounds, one entry per parameter (``None`` for
            none); only nuisance bounds are optimized under, but the grid must
            respect the target's bounds.
        upper: Optional upper bounds.
        optimizer: Only ``"bfgs"`` (central-difference BFGS with box
            transform) is supported.
        starts: Extra nuisance start vectors (nuisance coordinates only) used
            as a multistart fallback.
        tolerance: Gradient tolerance of the nuisance optimizations.
        max_iterations: Iteration limit per optimization.
        improvement_tolerance: Absolute decrease (in nll units) that counts
            as an improved estimate.

    Returns:
        :class:`ProfileLikelihoodResult`.

    Raises:
        TypeError: For a non-callable objective or non-real inputs.
        ValueError: For invalid index, grid, bounds, an estimate outside the
            bounds or with non-finite objective, or an unknown optimizer.

    Equation:
        [EQ-INV-PROF-001], [EQ-INV-PROF-002].

    References:
        [REF-RAUE-ETAL-2009], [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    if not callable(negative_log_likelihood):
        raise TypeError("negative_log_likelihood must be callable")
    if optimizer != "bfgs":
        raise ValueError("optimizer must be 'bfgs'")
    theta_hat = as_vector(estimate, name="estimate")
    size = len(theta_hat)
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < size:
        raise ValueError("index out of range")
    values = as_vector(grid, name="grid")
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("grid must have at least two strictly increasing values")
    if not values[0] <= theta_hat[index] <= values[-1]:
        raise ValueError("grid must bracket the estimate of the target parameter")
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
    if not lo[index] <= values[0] or not values[-1] <= hi[index]:
        raise ValueError("grid must respect the bounds of the target parameter")
    nuis = [j for j in range(size) if j != index]
    extra_starts = tuple(as_vector(s, name="starts") for s in starts)
    for s in extra_starts:
        if len(s) != len(nuis):
            raise ValueError("each extra start needs one entry per nuisance parameter")
    box = [
        (None if lo[j] == -math.inf else lo[j], None if hi[j] == math.inf else hi[j])
        for j in nuis
    ]

    counter = [0]
    safe = _invalid_to_inf(negative_log_likelihood, counter)
    nll_hat = safe(theta_hat)
    if not math.isfinite(nll_hat):
        raise ValueError("objective is not finite at the supplied estimate")
    nuisance_hat = tuple(theta_hat[j] for j in nuis)

    def assemble(t: float, nu: Vector) -> Vector:
        """Insert the target value into the nuisance vector.

        References:
            [EQ-INV-PROF-001]; implementation helper.

        """
        full = list(nu)
        full.insert(index, t)
        return tuple(full)

    def solve_from(t: float, start: Vector) -> tuple[float, Vector, bool, str] | None:
        """Minimize over the nuisance block from one start; ``None`` if infeasible.

        References:
            [EQ-INV-PROF-001]; implementation helper.

        """

        def inner(nu: Vector) -> float:
            """Objective in the nuisance coordinates.

            References:
                [EQ-INV-PROF-001]; implementation helper.

            """
            return safe(assemble(t, nu))

        if not math.isfinite(inner(start)):
            return None
        try:
            res = minimize_bfgs(
                inner,
                start,
                gradient_tolerance=tol,
                max_iterations=max_iterations,
                bounds=box,
            )
        except ValueError:
            return None
        return res.value, res.x, res.converged, res.message

    def point_at(t: float, previous: Vector) -> ProfilePoint:
        """Profile one grid value with warm start, estimate start, extras.

        References:
            [EQ-INV-PROF-001]; implementation helper.

        """
        before = counter[0]
        if not nuis:
            value = safe((t,))
            finite = math.isfinite(value)
            return ProfilePoint(
                t, value, (), finite, counter[0] - before, "fixed",
                "no nuisance parameters" if finite else "objective not finite",
            )  # fmt: skip
        candidates: list[tuple[str, Vector]] = [("warm", previous)]
        if nuisance_hat != previous:
            candidates.append(("estimate", nuisance_hat))
        best: tuple[float, Vector, bool, str, str] | None = None
        for label, start in candidates:
            out = solve_from(t, start)
            if out is not None and (best is None or out[0] < best[0]):
                best = (out[0], out[1], out[2], out[3], label)
        if best is None or not best[2]:
            for k, start in enumerate(extra_starts):
                out = solve_from(t, start)
                if out is not None and (best is None or out[0] < best[0]):
                    best = (out[0], out[1], out[2], out[3], f"extra[{k}]")
        spent = counter[0] - before
        if best is None:
            return ProfilePoint(
                t, math.inf, previous, False, spent, "none",
                "no finite start point",
            )  # fmt: skip
        return ProfilePoint(t, best[0], best[1], best[2], spent, best[4], best[3])

    nearest = min(range(len(values)), key=lambda k: abs(values[k] - theta_hat[index]))
    slots: list[ProfilePoint | None] = [None] * len(values)
    for direction in (1, -1):
        previous = nuisance_hat
        k = nearest if direction == 1 else nearest - 1
        while 0 <= k < len(values):
            point = point_at(values[k], previous)
            slots[k] = point
            if math.isfinite(point.profile_value):
                previous = point.nuisance
            k += direction
    points = tuple(p for p in slots if p is not None)
    best_point = min(points, key=lambda p: p.profile_value)
    improved = best_point.profile_value < nll_hat - imp_tol
    nll_min = min(nll_hat, best_point.profile_value)
    ratio = tuple(2.0 * (p.profile_value - nll_min) for p in points)
    return ProfileLikelihoodResult(
        index=index,
        estimate=theta_hat,
        nll_at_estimate=nll_hat,
        nll_min=nll_min,
        improved_estimate_found=improved,
        improved_estimate=(
            assemble(best_point.value, best_point.nuisance) if improved else None
        ),
        points=points,
        likelihood_ratio=ratio,
        all_converged=all(p.converged for p in points),
        total_evaluations=counter[0],
        improvement_tolerance=imp_tol,
    )


def profile_interval(
    result: ProfileLikelihoodResult, level: float = 0.95
) -> ProfileInterval:
    """Wilks likelihood-ratio interval by linear interpolation on the grid.

    The set ``{t : 2 (l_max - l_prof(t)) <= chi2_{1, level}}`` is intersected
    with the grid. Starting from the grid point of smallest profile value, each
    side is followed until the first grid point above the threshold; the
    endpoint is linearly interpolated in the likelihood ratio between that
    point and its inner neighbour. Because the ratio is typically convex, the
    chord lies above it and the interpolated interval is slightly narrower than
    the exact one; refine the grid to control this. If no grid point on a side
    exceeds the threshold the endpoint is ``None`` with status
    ``"unbounded_within_grid"``. If the neighbouring ratio is not finite the
    status is ``"undetermined_nonfinite"``.

    The calibration needs regular asymptotics: it fails at boundaries, under
    weak identifiability, for multimodal likelihoods and for small samples.
    Grid points below the threshold outside the connected component are
    reported in ``other_regions`` and flagged by ``connected=False``.

    Args:
        result: Output of :func:`profile_likelihood` for a true (not
            penalized) negative log-likelihood.
        level: Confidence level in ``(0, 1)``.

    Returns:
        :class:`ProfileInterval`.

    Raises:
        TypeError: If ``result`` has the wrong type.
        ValueError: If ``level`` is outside ``(0, 1)``.

    Equation:
        [EQ-INV-PROF-003].

    References:
        [REF-RAUE-ETAL-2009], [REF-SEBER-WILD-2003].

    """
    if not isinstance(result, ProfileLikelihoodResult):
        raise TypeError("result must be a ProfileLikelihoodResult")
    lev = require_finite_real(level, name="level")
    if not 0.0 < lev < 1.0:
        raise ValueError("level must lie in (0, 1)")
    threshold = ChiSquare(1.0).quantile(lev)
    grid = result.grid
    ratio = result.likelihood_ratio
    n = len(grid)
    centre = min(range(n), key=lambda k: ratio[k])

    def walk(step: int) -> tuple[float | None, str, int]:
        """Follow one side; return endpoint, status and last inside index.

        References:
            [EQ-INV-PROF-003]; implementation helper.

        """
        k = centre
        while 0 <= k + step < n:
            nxt = k + step
            if not ratio[nxt] <= threshold:
                if not math.isfinite(ratio[nxt]):
                    return None, "undetermined_nonfinite", k
                frac = (threshold - ratio[k]) / (ratio[nxt] - ratio[k])
                return grid[k] + frac * (grid[nxt] - grid[k]), "bounded", k
            k = nxt
        return None, "unbounded_within_grid", k

    lo, lo_status, lo_in = walk(-1)
    hi, hi_status, hi_in = walk(1)
    others = tuple(
        grid[k] for k in range(n) if not lo_in <= k <= hi_in and ratio[k] <= threshold
    )
    return ProfileInterval(
        level=lev,
        threshold=threshold,
        lower=lo,
        upper=hi,
        lower_status=lo_status,
        upper_status=hi_status,
        connected=not others,
        other_regions=others,
        all_converged=result.all_converged,
        improved_estimate_found=result.improved_estimate_found,
    )


def _symmetric_matrix(values: Sequence[Sequence[float]], name: str) -> Matrix:
    """Validate a square, numerically symmetric matrix.

    References:
        [EQ-INV-PROF-004]; implementation helper.

    """
    mat = as_matrix(values, name=name, square=True)
    scale = max(abs(x) for row in mat for x in row) or 1.0
    n = len(mat)
    for i in range(n):
        for j in range(i):
            if abs(mat[i][j] - mat[j][i]) > 1e-10 * scale:
                raise ValueError(f"{name} must be symmetric")
    return mat


def efficient_information(
    fisher: Sequence[Sequence[float]], target_indices: Sequence[int]
) -> Matrix:
    """Return the nuisance-corrected information ``F_tt - F_tn F_nn^-1 F_nt``.

    The Schur complement is formed with a Cholesky factor of ``F_nn`` and one
    triangular solve per target column; no inverse is formed. The result is
    the inverse of the target block of ``F^-1`` and is symmetrized
    ``(S + S')/2`` to remove rounding asymmetry only.

    Args:
        fisher: Symmetric information matrix (Fisher or observed).
        target_indices: Distinct target coordinates.

    Returns:
        The efficient information, a ``|t| x |t|`` matrix.

    Raises:
        ValueError: If ``fisher`` is not symmetric, the indices are invalid,
            or ``F_nn`` is not positive definite (nuisance parameters not
            estimable from the data; no generalized inverse is used).

    Equation:
        [EQ-INV-PROF-004].

    References:
        [REF-SEBER-WILD-2003], [REF-NOCEDAL-WRIGHT-2006].

    """
    mat = _symmetric_matrix(fisher, "fisher")
    n = len(mat)
    idx = tuple(target_indices)
    if not idx:
        raise ValueError("target_indices must be non-empty")
    if any(isinstance(i, bool) or not isinstance(i, int) for i in idx):
        raise TypeError("target_indices must be integers")
    if len(set(idx)) != len(idx) or any(not 0 <= i < n for i in idx):
        raise ValueError("target_indices must be distinct and in range")
    nu = tuple(j for j in range(n) if j not in set(idx))
    f_tt = [[mat[i][k] for k in idx] for i in idx]
    if not nu:
        return tuple(tuple(row) for row in f_tt)
    f_nn = tuple(tuple(mat[i][k] for k in nu) for i in nu)
    try:
        lower = cholesky(f_nn)
    except ValueError as exc:
        raise ValueError(
            "the nuisance block F_nn is singular or indefinite: the nuisance "
            "parameters are not estimable from this information, so the "
            "efficient information is not defined without further assumptions"
        ) from exc
    # columns of F_nt solved against F_nn
    solved = [cholesky_solve(lower, tuple(mat[i][j] for j in nu)) for i in idx]
    out = [
        [
            f_tt[a][b]
            - math.fsum(mat[idx[a]][j] * solved[b][p] for p, j in enumerate(nu))
            for b in range(len(idx))
        ]
        for a in range(len(idx))
    ]
    sym = tuple(
        tuple(0.5 * (out[a][b] + out[b][a]) for b in range(len(idx)))
        for a in range(len(idx))
    )
    return sym


def delta_method_variance(
    gradient: Sequence[float], covariance: Sequence[Sequence[float]]
) -> float:
    """Return the delta-method variance ``c' C c`` of a smooth functional.

    Args:
        gradient: Gradient ``c`` of the functional at the estimate.
        covariance: Symmetric covariance ``C`` of the estimator.

    Returns:
        ``c' C c`` (not clipped; a clearly negative value means ``C`` is not
        positive semidefinite and raises).

    Raises:
        ValueError: On shape mismatch, asymmetric ``C``, or a negative
            quadratic form beyond rounding.

    Equation:
        [EQ-INV-PROF-005].

    References:
        [REF-SEBER-WILD-2003].

    """
    c = as_vector(gradient, name="gradient")
    cov = _symmetric_matrix(covariance, "covariance")
    if len(cov) != len(c):
        raise ValueError("gradient and covariance dimensions differ")
    cc = [math.fsum(cov[i][j] * c[j] for j in range(len(c))) for i in range(len(c))]
    value = math.fsum(c[i] * cc[i] for i in range(len(c)))
    magnitude = math.fsum(abs(c[i] * cc[i]) for i in range(len(c)))
    if value < -1e-12 * magnitude:
        raise ValueError("covariance is not positive semidefinite along gradient")
    return value


def functional_estimability(
    functional: Sequence[float],
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
) -> EstimabilityResult:
    """Test whether ``c' theta`` is determined by a Jacobian or information.

    ``c' theta`` is locally estimable iff ``c`` is orthogonal to the null space
    of ``M`` (``M`` a Jacobian ``J`` or a Fisher matrix ``J'J``, which share
    the null space). The rank comes from the SVD with cut
    ``max(m, n) eps sigma_max``; the null-space component is
    ``c - V_r V_r' c``. This is local and linear, in the given coordinates; it
    proves nothing about structural identifiability.

    Args:
        functional: Nonzero coefficient vector ``c``.
        matrix: ``m x n`` Jacobian or ``n x n`` information matrix.
        tolerance: Relative tolerance on ``||null component|| / ||c||``.

    Returns:
        :class:`EstimabilityResult`.

    Raises:
        ValueError: On a zero ``c``, shape mismatch or invalid tolerance.

    Equation:
        [EQ-INV-PROF-006].

    References:
        [REF-SEBER-WILD-2003], [REF-NOCEDAL-WRIGHT-2006].

    """
    c = as_vector(functional, name="functional")
    mat = as_matrix(matrix, name="matrix")
    n = len(mat[0])
    if len(c) != n:
        raise ValueError("functional length must equal the number of columns")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative")
    norm_c = math.sqrt(math.fsum(x * x for x in c))
    if norm_c == 0.0:
        raise ValueError("functional must be nonzero")
    svd = singular_value_decomposition(mat)
    sigma = svd.singular_values
    cut = max(len(mat), n) * _EPS * (sigma[0] if sigma else 0.0)
    rank = sum(1 for s in sigma if s > cut)
    coeffs = [math.fsum(svd.v[i][k] * c[i] for i in range(n)) for k in range(rank)]
    resid = [
        c[i] - math.fsum(svd.v[i][k] * coeffs[k] for k in range(rank)) for i in range(n)
    ]
    null_norm = math.sqrt(math.fsum(x * x for x in resid))
    relative = null_norm / norm_c
    return EstimabilityResult(
        estimable=relative <= tol,
        null_component_norm=null_norm,
        relative_null_component=relative,
        rank=rank,
        null_dimension=n - rank,
        rank_tolerance=cut,
        tolerance=tol,
    )
