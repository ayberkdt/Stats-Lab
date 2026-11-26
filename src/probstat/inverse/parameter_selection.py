"""Choosing the Tikhonov parameter without the truth: one path, several criteria.

:func:`regularization_path` solves the affine, whitened Tikhonov problem of
[EQ-INV-LIN-001] on a user grid of ``lambda`` values and records, for each
``lambda``, the whitened misfit ``||r||``, the penalty ``||L (x - x_ref)||``,
the degrees of freedom ``df = tr(H_lambda)`` of the whitened influence matrix
(exact from the stacked QR, or a Hutchinson estimate with its Monte Carlo
standard error for matrix-free problems), and the criteria
[EQ-INV-SEL-001]:

- GCV: ``m ||r||^2 / (m - df)^2``;
- UPRE (Stein / Mallows, whitened UNIT noise): ``||r||^2 + 2 df - m``,
  an unbiased estimate of the PREDICTION risk ``E||W A (x_lambda - x_true)||^2``
  in observation space, not of the parameter error;
- the discrepancy gap ``||r|| - tau delta``;
- the discrete L-curve curvature of ``(log ||r||, log ||L(x - x_ref)||)``;
- the log marginal likelihood (evidence) when ``lambda^2 L'L`` is the
  precision of a PROPER Gaussian prior centred at ``x_ref`` (``L`` square
  and invertible), with all normalizing constants [EQ-INV-SEL-002].

:func:`select_regularization` picks a value by one criterion and reports
boundary choices (the optimum sits at the grid end, so the grid should be
extended), unattainable discrepancy targets and flat criteria instead of
fabricating a value. :func:`grouped_cross_validation` scores held-out GROUPS
of observations (sensors, experiments, time blocks) by refitting; it refuses
correlated noise across groups, for which the held-out score would leak.

When a synthetic truth is supplied, the error-optimal ``lambda`` is stored as
an unattainable benchmark; it never enters the selection.

Experimental (added in 0.28).

References:
    [REF-GOLUB-HEATH-WAHBA-1979], [REF-STEIN-1981], [REF-MALLOWS-1973],
    [REF-VOGEL-2002], [REF-HANSEN-OLEARY-1993], [REF-MOROZOV-1966],
    [REF-HUTCHINSON-1989], [REF-HANSEN-2008, Section 2.7].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    DiagonalCovariance,
    Vector,
    householder_qr,
    lsqr,
    materialize,
)

from .linear_problem import (
    LinearInverseProblem,
    _finite_vector,
    solve_regularized,
)

_CRITERIA = ("gcv", "upre", "discrepancy", "l_curve", "evidence")


@dataclass(frozen=True, slots=True)
class RegularizationPath:
    """Solutions and criteria of affine Tikhonov on a grid of ``lambda`` values.

    Attributes:
        regularizations: Increasing positive grid.
        estimates: ``x_lambda`` for each grid value.
        misfits: Whitened ``||W (A x - y)||``.
        penalty_norms: ``||L (x - x_ref)||``.
        degrees_of_freedom: ``tr(H_lambda)``.
        degrees_of_freedom_errors: Monte Carlo standard errors of ``df``
            (zeros for the exact dense path).
        gcv: GCV values (``inf`` where ``m - df <= 0``).
        upre: UPRE values for whitened unit noise.
        discrepancy_gaps: ``||r|| - tau delta`` (``None`` without
            ``noise_norm``).
        curvature: Discrete L-curve curvature (``nan`` at the two ends and
            where a norm vanishes).
        log_evidence: Log marginal likelihoods (``None`` unless ``L`` is
            square and invertible).
        errors: ``||x_lambda - x_true||`` for a synthetic truth, else
            ``None``.
        trace_method: ``"exact"`` or ``"hutchinson"``.
        observations: ``m``.
        target: ``tau delta`` or ``None``.

    References:
        [EQ-INV-SEL-001], [EQ-INV-SEL-002].

    """

    regularizations: Vector
    estimates: tuple[Vector, ...]
    misfits: Vector
    penalty_norms: Vector
    degrees_of_freedom: Vector
    degrees_of_freedom_errors: Vector
    gcv: Vector
    upre: Vector
    discrepancy_gaps: Vector | None
    curvature: Vector
    log_evidence: Vector | None
    errors: Vector | None
    trace_method: str
    observations: int
    target: float | None

    def oracle_index(self) -> int | None:
        """Return the error-minimizing grid index (benchmark only; needs a truth).

        References:
            [EQ-INV-SEL-001].

        """
        if self.errors is None:
            return None
        return min(range(len(self.errors)), key=self.errors.__getitem__)


def logarithmic_grid(low: float, high: float, points: int = 40) -> Vector:
    """Return ``points`` logarithmically spaced values from ``low`` to ``high``.

    Args:
        low: Positive lower end.
        high: Upper end ``> low``.
        points: At least 3.

    Returns:
        Increasing grid.

    References:
        [REF-HANSEN-2008, Section 2.7]. No equation is evaluated.

    """
    if not (math.isfinite(low) and math.isfinite(high)) or not 0.0 < low < high:
        raise ValueError("need 0 < low < high, both finite")
    if isinstance(points, bool) or not isinstance(points, int) or points < 3:
        raise ValueError("points must be an integer of at least 3")
    a, b = math.log(low), math.log(high)
    return tuple(math.exp(a + (b - a) * i / (points - 1)) for i in range(points))


def _discrete_curvature(lambdas: Vector, rho: Vector, eta: Vector) -> Vector:
    """Return the curvature of ``(log rho, log eta)`` in ``log lambda``.

    ``kappa = (x'y'' - x''y') / (x'^2 + y'^2)^(3/2)`` with ``x = log rho``,
    ``y = log eta`` [EQ-INV-SEL-001]. Non-uniform central differences;
    ``nan`` at the ends and where a norm is zero.

    References:
        [EQ-INV-SEL-001], [REF-HANSEN-OLEARY-1993].

    """
    n = len(lambdas)
    out = [math.nan] * n
    t = [math.log(v) for v in lambdas]
    if any(r <= 0.0 for r in rho) or any(e <= 0.0 for e in eta):
        valid = [r > 0.0 and e > 0.0 for r, e in zip(rho, eta, strict=True)]
    else:
        valid = [True] * n
    x = [math.log(r) if r > 0.0 else math.nan for r in rho]
    y = [math.log(e) if e > 0.0 else math.nan for e in eta]
    for i in range(1, n - 1):
        if not (valid[i - 1] and valid[i] and valid[i + 1]):
            continue
        h1, h2 = t[i] - t[i - 1], t[i + 1] - t[i]
        dx = (x[i + 1] - x[i - 1]) / (h1 + h2)
        dy = (y[i + 1] - y[i - 1]) / (h1 + h2)
        ddx = (
            2.0
            * (h1 * x[i + 1] - (h1 + h2) * x[i] + h2 * x[i - 1])
            / (h1 * h2 * (h1 + h2))
        )
        ddy = (
            2.0
            * (h1 * y[i + 1] - (h1 + h2) * y[i] + h2 * y[i - 1])
            / (h1 * h2 * (h1 + h2))
        )
        denominator = (dx * dx + dy * dy) ** 1.5
        out[i] = (dx * ddy - ddx * dy) / denominator if denominator > 0.0 else math.nan
    return tuple(out)


def _log_abs_diagonal(upper: Sequence[Sequence[float]]) -> float:
    """Return ``sum_i log |R_ii|``.

    References:
        [EQ-INV-SEL-002].

    """
    return math.fsum(math.log(abs(upper[i][i])) for i in range(len(upper)))


def regularization_path(
    problem: LinearInverseProblem,
    regularizations: Sequence[float],
    *,
    trace: str = "exact",
    probes: int = 20,
    seed: int | RandomStream | None = 0,
    noise_norm: float | None = None,
    tau: float = 1.0,
    truth: Sequence[float] | None = None,
    inner_tolerance: float = 1e-10,
) -> RegularizationPath:
    """Evaluate affine Tikhonov and the selection criteria on a ``lambda`` grid.

    ``trace="exact"`` factors the materialized stack ``[W A; lambda L]`` by
    Householder QR for every ``lambda`` and takes ``df = ||Q_1||_F^2``.
    ``trace="hutchinson"`` is matrix-free: the solution uses LSQR and
    ``df = tr(H)`` is estimated with the SAME Rademacher probes ``z_k`` at
    every ``lambda`` (common random numbers), each needing one LSQR solve of
    ``[W A; lambda L] w = [z_k; 0]``; the inner tolerance biases the estimate
    slightly and is reported via ``inner_tolerance``. The evidence is only
    computed on the exact path.

    The criteria assume whitened UNIT noise (``W`` from the true noise
    covariance). Under a misspecified covariance they are biased.

    Args:
        problem: Linear inverse problem.
        regularizations: Positive, strictly increasing grid (at least 3).
        trace: ``"exact"`` or ``"hutchinson"``.
        probes: Hutchinson probes.
        seed: Probe stream.
        noise_norm: ``delta`` for the discrepancy gaps.
        tau: ``tau >= 1`` (``1`` by default here, as the gap is reported).
        truth: Optional synthetic truth (benchmark errors only).
        inner_tolerance: LSQR tolerance of the matrix-free path.

    Returns:
        :class:`RegularizationPath`.

    Raises:
        ValueError: For an invalid grid or option.

    References:
        [EQ-INV-SEL-001], [EQ-INV-SEL-002], [REF-GOLUB-HEATH-WAHBA-1979],
        [REF-VOGEL-2002], [REF-HUTCHINSON-1989].

    """
    grid = tuple(float(v) for v in regularizations)
    if len(grid) < 3 or any(not math.isfinite(v) or v <= 0.0 for v in grid):
        raise ValueError(
            "regularizations must be at least three positive finite values"
        )
    if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError("regularizations must be strictly increasing")
    if trace not in ("exact", "hutchinson"):
        raise ValueError("trace must be 'exact' or 'hutchinson'")
    m, n = problem.shape
    true_x = None if truth is None else _finite_vector(truth, n, "truth")
    target = None
    if noise_norm is not None:
        if (
            not math.isfinite(noise_norm)
            or noise_norm <= 0.0
            or not math.isfinite(tau)
            or tau < 1.0
        ):
            raise ValueError("noise_norm must be positive and tau >= 1")
        target = tau * float(noise_norm)
    estimates, misfits, penalties, dofs, dof_errors, evidences = [], [], [], [], [], []
    square_invertible = problem.penalty.shape[0] == n
    log_det_ltl: float | None = None
    if trace == "exact" and square_invertible:
        l_factor = householder_qr(materialize(problem.penalty), rank_tolerance=1e-12)
        if l_factor.full_rank:
            log_det_ltl = 2.0 * _log_abs_diagonal(l_factor.r)
    log_det_gamma = 0.0 if problem.noise is None else problem.noise.log_det()
    stream = resolve_stream(seed)
    probe_vectors: list[list[float]] = []
    if trace == "hutchinson":
        if isinstance(probes, bool) or not isinstance(probes, int) or probes < 2:
            raise ValueError("probes must be an integer of at least 2")
        for k in range(probes):
            generator = stream.child("df_probe", k).generator()
            probe_vectors.append(
                [1.0 if generator.random() < 0.5 else -1.0 for _ in range(m)]
            )
    for lam in grid:
        if trace == "exact":
            solution = solve_regularized(problem, lam, method="qr")
            dof = solution.degrees_of_freedom
            assert dof is not None
            dof_error = 0.0
            if log_det_ltl is not None:
                stacked = materialize(problem.stacked_operator(lam))
                factor = householder_qr(stacked, rank_tolerance=1e-14)
                log_det_normal = 2.0 * _log_abs_diagonal(factor.r)
                log_det_prior_precision = 2.0 * n * math.log(lam) + log_det_ltl
                two_phi = solution.misfit**2 + lam * lam * solution.penalty_norm**2
                evidences.append(
                    -0.5
                    * (
                        two_phi
                        + log_det_normal
                        - log_det_prior_precision
                        + m * math.log(2.0 * math.pi)
                        + log_det_gamma
                    )
                )
        else:
            solution = solve_regularized(
                problem, lam, method="lsqr", tolerance=inner_tolerance
            )
            stacked_op = problem.stacked_operator(lam)
            top = problem.whitened_operator()
            samples = []
            pad = (0.0,) * problem.penalty.shape[0]
            for z in probe_vectors:
                w = lsqr(
                    stacked_op,
                    tuple(z) + pad,
                    atol=inner_tolerance,
                    btol=inner_tolerance,
                ).solution
                samples.append(
                    math.fsum(a * b for a, b in zip(z, top.matvec(w), strict=True))
                )
            dof = math.fsum(samples) / len(samples)
            variance = math.fsum((s - dof) ** 2 for s in samples) / (len(samples) - 1)
            dof_error = math.sqrt(variance / len(samples))
        estimates.append(solution.estimate)
        misfits.append(solution.misfit)
        penalties.append(solution.penalty_norm)
        dofs.append(dof)
        dof_errors.append(dof_error)
    gcv = tuple(
        m * r * r / (m - d) ** 2 if m - d > 0.0 else math.inf
        for r, d in zip(misfits, dofs, strict=True)
    )
    upre = tuple(r * r + 2.0 * d - m for r, d in zip(misfits, dofs, strict=True))
    gaps = None if target is None else tuple(r - target for r in misfits)
    curvature = _discrete_curvature(grid, tuple(misfits), tuple(penalties))
    errors = None
    if true_x is not None:
        errors = tuple(
            math.hypot(*(a - b for a, b in zip(x, true_x, strict=True)))
            for x in estimates
        )
    return RegularizationPath(
        regularizations=grid,
        estimates=tuple(estimates),
        misfits=tuple(misfits),
        penalty_norms=tuple(penalties),
        degrees_of_freedom=tuple(dofs),
        degrees_of_freedom_errors=tuple(dof_errors),
        gcv=gcv,
        upre=upre,
        discrepancy_gaps=gaps,
        curvature=curvature,
        log_evidence=tuple(evidences) if evidences else None,
        errors=errors,
        trace_method=trace,
        observations=m,
        target=target,
    )


@dataclass(frozen=True, slots=True)
class RegularizationSelection:
    """A ``lambda`` chosen from a path by one criterion, with its caveats.

    Attributes:
        criterion: ``"gcv"``, ``"upre"``, ``"discrepancy"``, ``"l_curve"`` or
            ``"evidence"``.
        index: Grid index chosen, or ``None`` when no valid choice exists.
        regularization: The chosen ``lambda`` or ``None``.
        estimate: The solution at that ``lambda`` or ``None``.
        status: ``"interior"`` (a proper interior optimum or crossing),
            ``"boundary"`` (the optimum is at a grid end: extend the grid),
            ``"unattainable"`` (the discrepancy target is never met),
            ``"always_met"`` (the target is met on the whole grid),
            ``"flat"`` (the criterion is constant to rounding), or
            ``"undefined"`` (no finite criterion value).
        values: The criterion on the grid.

    References:
        [EQ-INV-SEL-001].

    """

    criterion: str
    index: int | None
    regularization: float | None
    estimate: Vector | None
    status: str
    values: Vector


def select_regularization(
    path: RegularizationPath, criterion: str
) -> RegularizationSelection:
    """Select ``lambda`` from a path by one criterion without fabricating a value.

    GCV and UPRE are minimized, the evidence and the L-curve curvature
    maximized, over the grid. The discrepancy rule takes the LARGEST grid
    ``lambda`` with ``||r|| <= tau delta`` (the most regularized solution
    consistent with the noise level) [EQ-INV-SEL-001].

    Args:
        path: :class:`RegularizationPath`.
        criterion: One of ``"gcv"``, ``"upre"``, ``"discrepancy"``,
            ``"l_curve"``, ``"evidence"``.

    Returns:
        :class:`RegularizationSelection`.

    Raises:
        ValueError: If the criterion is unknown or not available on the path.

    References:
        [EQ-INV-SEL-001], [EQ-INV-SEL-002], [REF-MOROZOV-1966],
        [REF-HANSEN-OLEARY-1993].

    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    grid = path.regularizations
    last = len(grid) - 1
    if criterion == "discrepancy":
        if path.discrepancy_gaps is None:
            raise ValueError("the path was computed without noise_norm")
        values = path.discrepancy_gaps
        feasible = [i for i, g in enumerate(values) if g <= 0.0]
        if not feasible:
            return RegularizationSelection(
                criterion, None, None, None, "unattainable", values
            )
        index = max(feasible)
        status = "always_met" if len(feasible) == len(values) else "interior"
        return RegularizationSelection(
            criterion, index, grid[index], path.estimates[index], status, values
        )
    if criterion == "evidence":
        if path.log_evidence is None:
            raise ValueError(
                "the evidence needs the exact path and a square invertible L"
            )
        values = path.log_evidence
        sign = -1.0
    elif criterion == "l_curve":
        values = path.curvature
        sign = -1.0
    else:
        values = path.gcv if criterion == "gcv" else path.upre
        sign = 1.0
    finite = [i for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return RegularizationSelection(criterion, None, None, None, "undefined", values)
    index = min(finite, key=lambda i: sign * values[i])
    spread = max(values[i] for i in finite) - min(values[i] for i in finite)
    scale = max(abs(values[i]) for i in finite)
    if spread <= 1e-12 * max(scale, 1e-300):
        status = "flat"
    elif index in (0, last) or (
        criterion == "l_curve"
        and index in (1, last - 1)
        and index in (min(finite), max(finite))
    ):
        status = "boundary"
    else:
        status = "interior"
    return RegularizationSelection(
        criterion, index, grid[index], path.estimates[index], status, values
    )


@dataclass(frozen=True, slots=True)
class CrossValidationPath:
    """Grouped cross-validation scores on a ``lambda`` grid.

    Attributes:
        regularizations: The grid.
        scores: Mean held-out whitened squared error per observation.
        fold_scores: Per-group scores (rows: groups, columns: grid).
        selected_index: Minimizing index.
        boundary: Whether the minimum is at a grid end.

    References:
        [EQ-INV-SEL-003].

    """

    regularizations: Vector
    scores: Vector
    fold_scores: tuple[Vector, ...]
    selected_index: int
    boundary: bool


def grouped_cross_validation(
    problem: LinearInverseProblem,
    regularizations: Sequence[float],
    groups: Sequence[int],
) -> CrossValidationPath:
    """Leave-one-group-out cross-validation of affine Tikhonov by refitting.

    For each group ``g`` the problem is refitted without its observations and
    scored by ``||W_g (A_g x_(-g) - y_g)||^2 / m_g`` [EQ-INV-SEL-003]. Groups
    must be independent under the noise model: only a diagonal noise
    covariance (or none) is accepted, because with correlated noise across
    groups the marginal held-out score leaks training information.

    Args:
        problem: Linear inverse problem with ``noise`` ``None`` or
            :class:`~probstat.linalg.DiagonalCovariance`.
        regularizations: Positive increasing grid.
        groups: Group label per observation (at least two groups).

    Returns:
        :class:`CrossValidationPath`.

    Raises:
        ValueError: For correlated noise, fewer than two groups or bad
            labels.

    References:
        [EQ-INV-SEL-003], [REF-HANSEN-2008, Section 2.7].

    """
    m, n = problem.shape
    labels = list(groups)
    if len(labels) != m:
        raise ValueError("groups must have one label per observation")
    if problem.noise is not None and not isinstance(problem.noise, DiagonalCovariance):
        raise ValueError("grouped cross-validation needs independent (diagonal) noise")
    distinct = sorted(set(labels))
    if len(distinct) < 2:
        raise ValueError("at least two groups are needed")
    grid = tuple(float(v) for v in regularizations)
    if len(grid) < 2 or any(not math.isfinite(v) or v <= 0.0 for v in grid):
        raise ValueError("regularizations must be positive and finite")
    dense = materialize(problem.operator)
    deviations = (
        (1.0,) * m if problem.noise is None else problem.noise.standard_deviations
    )
    fold_scores = []
    for g in distinct:
        train = [i for i in range(m) if labels[i] != g]
        test = [i for i in range(m) if labels[i] == g]
        sub_problem = LinearInverseProblem.create(
            [dense[i] for i in train],
            [problem.data[i] for i in train],
            noise=None
            if problem.noise is None
            else DiagonalCovariance([deviations[i] ** 2 for i in train]),
            penalty=problem.penalty,
            reference=problem.reference,
        )
        row = []
        for lam in grid:
            x = solve_regularized(sub_problem, lam, method="svd").estimate
            errors = [
                (
                    math.fsum(a * b for a, b in zip(dense[i], x, strict=True))
                    - problem.data[i]
                )
                / deviations[i]
                for i in test
            ]
            row.append(math.fsum(e * e for e in errors) / len(test))
        fold_scores.append(tuple(row))
    scores = tuple(
        math.fsum(f[j] for f in fold_scores) / len(fold_scores)
        for j in range(len(grid))
    )
    index = min(range(len(grid)), key=scores.__getitem__)
    return CrossValidationPath(
        grid, scores, tuple(fold_scores), index, index in (0, len(grid) - 1)
    )
