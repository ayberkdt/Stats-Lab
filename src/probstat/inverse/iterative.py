"""Iterative regularization: Landweber, early-stopped Krylov and hybrid LSQR.

For the whitened data problem ``B z = b`` with ``B = W A``,
``b = W (y - A x_ref)`` and ``x = x_ref + z``, iteration count plays the role
of the regularization parameter: early iterates from ``z = 0`` keep the
large singular components and suppress the small ones, while late iterates
fit the noise (semi-convergence: the residual keeps falling while the error
grows). Three methods are provided:

- :func:`landweber`: ``z_(k+1) = z_k + omega B'(b - B z_k)`` with
  ``0 < omega < 2 / ||B||^2`` [EQ-INV-ITER-001]. The step needs an UPPER
  bound on ``||B||``; the lower estimate of
  :func:`probstat.linalg.power_iteration` is rejected unless the caller
  supplies it with an explicit safety factor.
- :func:`krylov_regularization`: CGLS, LSQR or LSMR on ``B z = b`` stopped by
  the discrepancy principle ``||W (A x_k - y)|| <= tau delta``
  [EQ-INV-ITER-002] on the DATA residual (never on an augmented residual).
- :func:`hybrid_lsqr`: Golub-Kahan bidiagonalization
  ``B V_k = U_(k+1) T_k`` with full reorthogonalization and a Tikhonov
  problem ``min ||T_k q - beta e_1||^2 + lambda_k^2 ||q||^2`` solved on the
  small projected system at every step, with ``lambda_k`` chosen by
  projected GCV or the projected discrepancy principle [EQ-INV-ITER-003].
  Projected GCV is a criterion on the PROJECTED problem; it is not the
  full-problem GCV of :func:`probstat.inverse.generalized_cross_validation`.

When a synthetic truth is passed, the error history and the error-optimal
iteration are recorded as an UNATTAINABLE oracle for benchmarking only; they
never influence the stopping rule.

Experimental (added in 0.28).

References:
    [REF-LANDWEBER-1951], [REF-ENGL-HANKE-NEUBAUER-1996],
    [REF-HANSEN-2008, Section 2.8], [REF-PAIGE-SAUNDERS-1982],
    [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-MOROZOV-1966].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import (
    CountingOperator,
    KrylovResult,
    OperatorCounts,
    Vector,
    cgls,
    householder_qr,
    lsmr,
    lsqr,
    singular_value_decomposition,
)

from .linear_problem import LinearInverseProblem, _finite_vector

_METHODS = ("cgls", "lsqr", "lsmr")


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.hypot(*vector) if vector else 0.0


def expected_noise_norm(observations: int) -> float:
    """Return ``sqrt(m)``, the approximate norm of ``m`` whitened unit-variance errors.

    ``E||e||^2 = m`` for ``e ~ N(0, I_m)``, so ``delta = sqrt(m)`` is the
    natural noise level of the whitened discrepancy principle
    [EQ-INV-ITER-002]. It is NOT the noise standard deviation.

    Args:
        observations: ``m >= 1``.

    Returns:
        ``sqrt(m)``.

    References:
        [EQ-INV-ITER-002], [REF-HANSEN-2008, Section 2.7.5].

    """
    if (
        isinstance(observations, bool)
        or not isinstance(observations, int)
        or observations < 1
    ):
        raise ValueError("observations must be a positive integer")
    return math.sqrt(observations)


@dataclass(frozen=True, slots=True)
class IterativeRegularizationResult:
    """Result of an iteratively regularized linear inversion.

    Attributes:
        estimate: ``x_k = x_ref + z_k`` at the stopping iteration.
        iterations: Stopping iteration ``k``.
        stop_reason: ``"discrepancy"``, ``"max_iterations"`` or the Krylov
            solver's own reason (``"normal_tolerance"``, ``"breakdown"``...).
        converged: Whether a stopping RULE fired (discrepancy or tolerance);
            ``False`` for ``"max_iterations"`` and ``"breakdown"``.
        misfit_history: Whitened data misfit ``||W (A x_k - y)||``, ``k >= 0``.
        error_history: ``||x_k - x_true||`` when a synthetic truth was given,
            else ``None``.
        oracle_iteration: Error-minimizing iteration (benchmark only), else
            ``None``.
        target: ``tau delta`` used for the discrepancy rule, or ``None``.
        method: ``"landweber"``, ``"cgls"``, ``"lsqr"``, ``"lsmr"`` or
            ``"hybrid_lsqr"``.
        counts: Applications of ``A`` (forward, transpose).
        regularization_history: ``lambda_k`` of the hybrid method, else
            ``None``.
        krylov: Underlying Krylov result, when one was used.
        gcv_weight_history: ``omega_k`` used by the weighted-GCV rule of the
            hybrid method (``None`` for every other rule).

    References:
        [EQ-INV-ITER-001], [EQ-INV-ITER-002], [EQ-INV-ITER-003],
        [EQ-INV-ITER-004].

    """

    estimate: Vector
    iterations: int
    stop_reason: str
    converged: bool
    misfit_history: Vector
    error_history: Vector | None
    oracle_iteration: int | None
    target: float | None
    method: str
    counts: OperatorCounts
    regularization_history: Vector | None
    krylov: KrylovResult | None
    gcv_weight_history: Vector | None = None


def _target(noise_norm: float | None, tau: float) -> float | None:
    """Validate ``tau >= 1`` and ``delta > 0`` and return ``tau delta``.

    References:
        [EQ-INV-ITER-002].

    """
    if noise_norm is None:
        return None
    delta = float(noise_norm)
    t = float(tau)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("noise_norm must be finite and positive")
    if not math.isfinite(t) or t < 1.0:
        raise ValueError("tau must be finite and at least one")
    return t * delta


def _counted_problem(
    problem: LinearInverseProblem,
) -> tuple[LinearInverseProblem, CountingOperator]:
    """Return a copy of the problem whose operator counts applications.

    References:
        [EQ-INV-LIN-001].

    """
    if not isinstance(problem, LinearInverseProblem):
        raise TypeError("problem must be a LinearInverseProblem")
    counted = CountingOperator(problem.operator)
    return (
        LinearInverseProblem(
            counted, problem.data, problem.noise, problem.penalty, problem.reference
        ),
        counted,
    )


def _errors(
    history: list[Vector], truth: Vector | None
) -> tuple[Vector | None, int | None]:
    """Return the error history and its argmin (oracle) for a synthetic truth.

    References:
        [EQ-INV-ITER-002].

    """
    if truth is None:
        return None, None
    errors = tuple(
        _norm([a - b for a, b in zip(x, truth, strict=True)]) for x in history
    )
    return errors, min(range(len(errors)), key=errors.__getitem__)


def landweber(
    problem: LinearInverseProblem,
    *,
    step: float | None = None,
    operator_norm_bound: float | None = None,
    noise_norm: float | None = None,
    tau: float = 1.01,
    max_iterations: int = 500,
    truth: Sequence[float] | None = None,
) -> IterativeRegularizationResult:
    """Run the Landweber iteration on the whitened problem with discrepancy stopping.

    ``z_(k+1) = z_k + omega B'(b - B z_k)`` from ``z_0 = 0`` [EQ-INV-ITER-001].
    Exactly one of ``step`` (``omega``) or ``operator_norm_bound`` (a
    CERTIFIED upper bound ``||W A|| <= N``, giving ``omega = 1 / N^2``) is
    required; ``omega`` must satisfy ``0 < omega < 2 / N^2`` when the bound is
    known. A power-iteration estimate is a lower bound and must be inflated
    by the caller before use.

    Args:
        problem: Linear inverse problem (penalty ignored; the iteration is
            the regularization).
        step: ``omega > 0``.
        operator_norm_bound: Upper bound ``N`` on ``||W A||``.
        noise_norm: ``delta`` (whitened); enables the discrepancy rule.
        tau: ``tau >= 1``.
        max_iterations: Iteration cap.
        truth: Optional synthetic truth for the oracle error history.

    Returns:
        :class:`IterativeRegularizationResult`.

    Raises:
        ValueError: If neither or both step arguments are given, or
            ``omega`` violates the bound.

    References:
        [EQ-INV-ITER-001], [EQ-INV-ITER-002], [REF-LANDWEBER-1951],
        [REF-ENGL-HANKE-NEUBAUER-1996].

    """
    if (step is None) == (operator_norm_bound is None):
        raise ValueError("pass exactly one of step or operator_norm_bound")
    if operator_norm_bound is not None:
        bound = float(operator_norm_bound)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("operator_norm_bound must be finite and positive")
        omega = 1.0 / (bound * bound)
    else:
        omega = float(step)  # type: ignore[arg-type]
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("step must be finite and positive")
    target = _target(noise_norm, tau)
    counted_problem, counted = _counted_problem(problem)
    true_x = None if truth is None else _finite_vector(truth, problem.shape[1], "truth")
    operator = counted_problem.whitened_operator()
    b = counted_problem.stacked_data(0.0)
    n = problem.shape[1]
    z = [0.0] * n
    residual = list(b)
    misfits = [_norm(residual)]
    iterates = [problem.reference]
    reason = "max_iterations"
    if target is not None and misfits[0] <= target:
        reason = "discrepancy"
    else:
        for _ in range(1, max_iterations + 1):
            gradient = operator.rmatvec(residual)
            z = [zi + omega * gi for zi, gi in zip(z, gradient, strict=True)]
            fitted = operator.matvec(z)
            residual = [bi - fi for bi, fi in zip(b, fitted, strict=True)]
            misfit = _norm(residual)
            if not math.isfinite(misfit) or misfit > 1e3 * misfits[0] + 1e-300:
                reason = "divergence"
                misfits.append(misfit)
                iterates.append(
                    tuple(a + c for a, c in zip(z, problem.reference, strict=True))
                )
                break
            misfits.append(misfit)
            iterates.append(
                tuple(a + c for a, c in zip(z, problem.reference, strict=True))
            )
            if target is not None and misfit <= target:
                reason = "discrepancy"
                break
    errors, oracle = _errors(iterates, true_x)
    return IterativeRegularizationResult(
        estimate=iterates[-1],
        iterations=len(iterates) - 1,
        stop_reason=reason,
        converged=reason == "discrepancy",
        misfit_history=tuple(misfits),
        error_history=errors,
        oracle_iteration=oracle,
        target=target,
        method="landweber",
        counts=counted.counts,
        regularization_history=None,
        krylov=None,
    )


def krylov_regularization(
    problem: LinearInverseProblem,
    *,
    method: str = "lsqr",
    noise_norm: float | None = None,
    tau: float = 1.01,
    max_iterations: int = 200,
    truth: Sequence[float] | None = None,
) -> IterativeRegularizationResult:
    """Early-stopped CGLS, LSQR or LSMR with the discrepancy principle.

    The Krylov solver runs on ``W A z = W (y - A x_ref)`` from ``z = 0``;
    after iteration ``k`` the whitened data misfit is tested against
    ``tau delta`` [EQ-INV-ITER-002]. Without ``noise_norm`` the solver runs to
    its own tolerance or the cap, which for noisy data over-fits.

    Args:
        problem: Linear inverse problem (penalty ignored).
        method: ``"cgls"``, ``"lsqr"`` or ``"lsmr"``.
        noise_norm: ``delta`` in whitened units (e.g.
            :func:`expected_noise_norm`).
        tau: ``tau >= 1``.
        max_iterations: Iteration cap.
        truth: Optional synthetic truth for the oracle error history.

    Returns:
        :class:`IterativeRegularizationResult` with ``krylov`` set.

    References:
        [EQ-INV-ITER-002], [EQ-LINALG-KRY-005], [EQ-LINALG-KRY-007],
        [REF-HANSEN-2008, Section 2.8], [REF-ENGL-HANKE-NEUBAUER-1996].

    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    target = _target(noise_norm, tau)
    counted_problem, counted = _counted_problem(problem)
    true_x = None if truth is None else _finite_vector(truth, problem.shape[1], "truth")
    operator = counted_problem.whitened_operator()
    b = counted_problem.stacked_data(0.0)
    iterates: list[Vector] = [problem.reference]
    misfits = [_norm(b)]
    reference = problem.reference

    def rule(k: int, z: Vector, residual: float) -> bool:
        """Record the iterate and test the discrepancy principle.

        References:
            [EQ-INV-ITER-002].

        """
        iterates.append(tuple(a + c for a, c in zip(z, reference, strict=True)))
        misfits.append(residual)
        return target is not None and residual <= target

    if target is not None and misfits[0] <= target:
        errors, oracle = _errors(iterates, true_x)
        return IterativeRegularizationResult(
            reference,
            0,
            "discrepancy",
            True,
            tuple(misfits),
            errors,
            oracle,
            target,
            method,
            counted.counts,
            None,
            None,
        )
    solver = lsqr if method == "lsqr" else lsmr
    if method == "cgls":
        result = cgls(
            operator,
            b,
            residual_tolerance=1e-14,
            normal_tolerance=1e-14,
            max_iterations=max_iterations,
            callback=rule,
        )
    else:
        result = solver(
            operator,
            b,
            atol=1e-14,
            btol=1e-14,
            max_iterations=max_iterations,
            callback=rule,
        )
    reason = "discrepancy" if result.stop_reason == "callback" else result.stop_reason
    errors, oracle = _errors(iterates, true_x)
    return IterativeRegularizationResult(
        estimate=iterates[-1],
        iterations=result.iterations,
        stop_reason=reason,
        converged=reason not in ("max_iterations", "breakdown"),
        misfit_history=tuple(misfits),
        error_history=errors,
        oracle_iteration=oracle,
        target=target,
        method=method,
        counts=counted.counts,
        regularization_history=None,
        krylov=result,
    )


def _projected_tikhonov(
    bidiagonal: list[list[float]],
    beta: float,
    lam: float,
    svd_cache: tuple[Vector, list[list[float]], list[list[float]]],
    omega: float = 1.0,
) -> tuple[list[float], float, float]:
    """Solve ``min ||T q - beta e_1||^2 + lambda^2 ||q||^2`` from the SVD of ``T``.

    Returns ``q``, the projected residual norm and the (weighted) projected
    GCV value ``||(I - T T_lambda^#) beta e_1||^2 / (k + 1 - omega sum f_i)^2``
    (``omega = 1``: standard GCV).

    References:
        [EQ-INV-ITER-003], [EQ-INV-ITER-004].

    """
    sigma, u, v = svd_cache
    rows = len(bidiagonal)
    k = len(sigma)
    g = [beta * u[0][i] for i in range(k)]  # U'(beta e_1)
    filters = [s * s / (s * s + lam * lam) if s > 0.0 else 0.0 for s in sigma]
    coefficients = [
        f * gi / s if s > 0.0 else 0.0
        for f, gi, s in zip(filters, g, sigma, strict=True)
    ]
    q = [math.fsum(v[j][i] * coefficients[i] for i in range(k)) for j in range(k)]
    # Residual of the projected problem: beta e1 - T q.
    residual = [
        (beta if r == 0 else 0.0) - math.fsum(bidiagonal[r][j] * q[j] for j in range(k))
        for r in range(rows)
    ]
    rnorm = _norm(residual)
    denominator = rows - omega * math.fsum(filters)
    gcv = rnorm * rnorm / (denominator * denominator) if denominator > 0.0 else math.inf
    return q, rnorm, gcv


def _projected_general(
    bidiagonal: list[list[float]],
    beta: float,
    lam: float,
    penalty_r: list[list[float]],
    omega: float = 1.0,
) -> tuple[list[float], float, float]:
    """Solve ``min ||T q - beta e_1||^2 + lambda^2 ||R q||^2`` by QR.

    ``R`` is the triangular factor of ``L V_k`` (so ``||R q|| = ||L V_k q||``).
    The projected GCV value uses ``tr(H) = ||Q_1||_F^2`` of the top block of
    the stacked QR factor of ``[T; lambda R]`` [EQ-INV-ITER-003], weighted
    as ``k + 1 - omega tr(H)`` [EQ-INV-ITER-004].

    References:
        [EQ-INV-ITER-003], [EQ-INV-ITER-004], [EQ-LINALG-009].

    """
    rows = len(bidiagonal)
    k = len(bidiagonal[0])
    stacked = [list(row) for row in bidiagonal] + [
        [lam * value for value in row] for row in penalty_r
    ]
    factor = householder_qr(tuple(tuple(row) for row in stacked), rank_tolerance=1e-14)
    rhs = (beta,) + (0.0,) * (len(stacked) - 1)
    q = list(factor.solve(rhs))
    residual = [
        (beta if r == 0 else 0.0) - math.fsum(bidiagonal[r][j] * q[j] for j in range(k))
        for r in range(rows)
    ]
    rnorm = _norm(residual)
    trace = math.fsum(factor.q[r][j] ** 2 for r in range(rows) for j in range(k))
    denominator = rows - omega * trace
    gcv = rnorm * rnorm / (denominator * denominator) if denominator > 0.0 else math.inf
    return q, rnorm, gcv


def _upper_factor(columns: list[list[float]]) -> list[list[float]]:
    """Return a ``k``-column matrix ``R`` with ``||R q|| = ||[c_1..c_k] q||``.

    Uses the triangular factor of Householder QR when there are at least
    ``k`` rows, and the matrix itself otherwise.

    References:
        [EQ-LINALG-009].

    """
    k = len(columns)
    rows = len(columns[0])
    matrix = [[columns[j][i] for j in range(k)] for i in range(rows)]
    if rows < k:
        return matrix
    factor = householder_qr(tuple(tuple(row) for row in matrix), rank_tolerance=1e-14)
    return [list(row) for row in factor.r]


def _choose_lambda(
    bidiagonal: list[list[float]],
    beta: float,
    rule: str,
    target: float | None,
    fixed: float | None,
    penalty_r: list[list[float]] | None = None,
    omega: float = 1.0,
) -> float:
    """Choose ``lambda_k`` on the projected problem.

    ``"gcv"`` minimizes projected GCV over a logarithmic grid of 60 points
    spanning the projected singular values, ``"wgcv"`` the weighted GCV with
    weight ``omega`` on the same grid; ``"discrepancy"`` takes the
    largest grid ``lambda`` whose projected residual is at most ``target``
    (the smallest grid value when none is); ``"fixed"`` returns ``fixed``.

    References:
        [EQ-INV-ITER-003], [EQ-INV-ITER-002], [EQ-INV-ITER-004].

    """
    if rule == "fixed":
        assert fixed is not None
        return fixed
    svd = singular_value_decomposition(tuple(tuple(row) for row in bidiagonal))
    sigma = svd.singular_values
    cache = (sigma, [list(r) for r in svd.u], [list(r) for r in svd.v])
    positive = [s for s in sigma if s > 0.0]
    if not positive:
        return 0.0
    high, low = max(positive), max(min(positive), max(positive) * 1e-12)
    if penalty_r is not None:
        r_sigma = [
            s
            for s in singular_value_decomposition(
                tuple(tuple(row) for row in penalty_r)
            ).singular_values
            if s > 0.0
        ]
        if r_sigma:
            high = high / max(min(r_sigma), max(r_sigma) * 1e-12)
            low = low / max(r_sigma)
    grid = [
        math.exp(
            math.log(low * 1e-2)
            + (math.log(high * 10.0) - math.log(low * 1e-2)) * i / 59
        )
        for i in range(60)
    ]

    def evaluate(lam: float) -> tuple[list[float], float, float]:
        """Solve the projected problem for one ``lambda``.

        References:
            [EQ-INV-ITER-003].

        """
        if penalty_r is None:
            return _projected_tikhonov(bidiagonal, beta, lam, cache, omega)
        return _projected_general(bidiagonal, beta, lam, penalty_r, omega)

    if rule in ("gcv", "wgcv"):
        values = [evaluate(lam)[2] for lam in grid]
        return grid[min(range(60), key=values.__getitem__)]
    assert target is not None
    feasible = [lam for lam in grid if evaluate(lam)[1] <= target]
    return max(feasible) if feasible else grid[0]


def _adaptive_gcv_weight(
    bidiagonal: list[list[float]],
    beta: float,
) -> float | None:
    """Return ``omega_hat_k`` of the adaptive weighted-GCV rule.

    ``omega_hat_k`` makes ``lambda = delta_min(T_k)`` a stationary point of
    ``G(omega, lambda) = rho(lambda) / (k + 1 - omega F(lambda))^2`` with
    ``rho = sum_i h_i^2 g_i^2 + g_(k+1)^2``, ``h_i = lambda^2 /
    (delta_i^2 + lambda^2)``, ``F = sum_i delta_i^2 / (delta_i^2 +
    lambda^2)`` and ``g = U'(beta e_1)``. ``dG/dlambda = 0`` is linear in
    ``omega``: ``omega_hat = (k + 1) rho' / (rho' F + 2 rho S)`` with
    ``S = sum_i dh_i/dlambda = sum_i 2 lambda delta_i^2 / (delta_i^2 +
    lambda^2)^2`` [EQ-INV-ITER-004]. Returns ``None`` when ``T_k`` has a zero
    singular value or the expression is undefined.

    References:
        [EQ-INV-ITER-004], [REF-CHUNG-NAGY-OLEARY-2008, Section 5.4].

    """
    svd = singular_value_decomposition(tuple(tuple(row) for row in bidiagonal))
    sigma = svd.singular_values
    k = len(sigma)
    if not sigma or min(sigma) <= 0.0:
        return None
    lam = min(sigma)
    g = [beta * svd.u[0][i] for i in range(k)]
    tail = max(beta * beta - math.fsum(gi * gi for gi in g), 0.0)
    h = [lam * lam / (s * s + lam * lam) for s in sigma]
    dh = [2.0 * lam * s * s / (s * s + lam * lam) ** 2 for s in sigma]
    rho = math.fsum(hi * hi * gi * gi for hi, gi in zip(h, g, strict=True)) + tail
    d_rho = math.fsum(
        2.0 * hi * di * gi * gi for hi, di, gi in zip(h, dh, g, strict=True)
    )
    filters = math.fsum(1.0 - hi for hi in h)
    slope = math.fsum(dh)
    denominator = d_rho * filters + 2.0 * rho * slope
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return (k + 1) * d_rho / denominator


def hybrid_lsqr(
    problem: LinearInverseProblem,
    *,
    rule: str = "gcv",
    noise_norm: float | None = None,
    tau: float = 1.01,
    regularization: float | None = None,
    max_iterations: int = 50,
    stabilization_tolerance: float = 1e-3,
    truth: Sequence[float] | None = None,
    gcv_weight: float | str = "adaptive",
) -> IterativeRegularizationResult:
    """Hybrid LSQR: bidiagonalize ``W A`` and regularize the projected problem.

    After ``k`` Golub-Kahan steps with full reorthogonalization,
    ``W A V_k = U_(k+1) T_k``; ``q_k`` solves the projected Tikhonov problem
    ``min ||T_k q - beta e_1||^2 + lambda_k^2 ||q||^2`` and
    ``x_k = x_ref + V_k q_k`` [EQ-INV-ITER-003]. For a general penalty
    ``L`` the projected penalty is ``||L V_k q||`` (one application of ``L``
    per step; ``L V_k`` is reduced to a ``k x k`` triangular factor), not
    ``||q||``. With orthonormal bases the
    projected residual equals ``||W (A x_k - y)||``. ``lambda_k`` follows
    ``rule``: ``"gcv"`` (projected GCV), ``"wgcv"`` (weighted projected
    GCV, ``G(omega, lambda) = k ||r||^2 / (k + 1 - omega tr H)^2``, which
    counters the over-smoothing of projected GCV for ``omega < 1``
    [EQ-INV-ITER-004]), ``"discrepancy"`` (needs
    ``noise_norm``) or ``"fixed"`` (needs ``regularization``; then ``x_k``
    tends to the Tikhonov solution of [EQ-INV-LIN-001]). For ``"wgcv"``,
    ``gcv_weight`` is a fixed ``omega`` in ``(0, 1]`` or ``"adaptive"``:
    ``omega_k`` is the mean of ``omega_hat_1 .. omega_hat_k``, each making
    ``lambda = delta_min(T_j)`` a stationary point of the weighted GCV
    function (Chung, Nagy and O'Leary, Section 5.4; identity penalty only).
    The adaptive values are not clipped; they are reported in
    ``gcv_weight_history``. The iteration
    stops when
    ``lambda_k`` and ``||x_k||`` change by less than
    ``stabilization_tolerance`` relatively for two consecutive steps, at a
    breakdown (invariant subspace), or at the cap. Memory is
    ``O(k (m + n))``.

    Args:
        problem: Linear inverse problem; its penalty ``L`` and reference
            ``x_ref`` are used.
        rule: ``"gcv"``, ``"discrepancy"`` or ``"fixed"``.
        noise_norm: ``delta`` for the discrepancy rule.
        tau: ``tau >= 1``.
        regularization: ``lambda`` for the fixed rule.
        max_iterations: Cap on ``k`` (at most ``min(m, n)``).
        stabilization_tolerance: Relative stabilization threshold.
        truth: Optional synthetic truth for the oracle error history.
        gcv_weight: ``omega`` of the ``"wgcv"`` rule: a float in ``(0, 1]``
            or ``"adaptive"`` (ignored by the other rules).

    Returns:
        :class:`IterativeRegularizationResult` with
        ``regularization_history``.

    Raises:
        ValueError: For an invalid rule or missing rule parameter.

    References:
        [EQ-INV-ITER-003], [EQ-INV-ITER-004], [EQ-LINALG-KRY-006],
        [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-GOLUB-KAHAN-1965],
        [REF-CHUNG-NAGY-OLEARY-2008].

    """
    from probstat.linalg import IdentityOperator

    if rule not in ("gcv", "wgcv", "discrepancy", "fixed"):
        raise ValueError("rule must be 'gcv', 'wgcv', 'discrepancy' or 'fixed'")
    general = not isinstance(problem.penalty, IdentityOperator)
    adaptive = False
    omega = 1.0
    if rule == "wgcv":
        if gcv_weight == "adaptive":
            if general:
                raise ValueError(
                    "the adaptive weighted-GCV rule needs the identity penalty; "
                    "pass a fixed gcv_weight for a general L"
                )
            adaptive = True
        elif isinstance(gcv_weight, (int, float)) and not isinstance(gcv_weight, bool):
            omega = float(gcv_weight)
            if not math.isfinite(omega) or not 0.0 < omega <= 1.0:
                raise ValueError("a fixed gcv_weight must lie in (0, 1]")
        else:
            raise ValueError("gcv_weight must be a float in (0, 1] or 'adaptive'")
    omega_hats: list[float] = []
    omegas: list[float] = []
    penalty_columns: list[list[float]] = []
    target = _target(noise_norm, tau)
    if rule == "discrepancy" and target is None:
        raise ValueError("the discrepancy rule needs noise_norm")
    fixed: float | None = None
    if rule == "fixed":
        if (
            regularization is None
            or not math.isfinite(regularization)
            or regularization < 0.0
        ):
            raise ValueError(
                "the fixed rule needs a finite non-negative regularization"
            )
        fixed = float(regularization)
    counted_problem, counted = _counted_problem(problem)
    true_x = None if truth is None else _finite_vector(truth, problem.shape[1], "truth")
    operator = counted_problem.whitened_operator()
    b = list(counted_problem.stacked_data(0.0))
    m, n = problem.shape
    limit = min(max_iterations, m, n)
    beta = _norm(b)
    iterates: list[Vector] = [problem.reference]
    misfits = [beta]
    lambdas: list[float] = []
    if beta == 0.0:
        errors, oracle = _errors(iterates, true_x)
        return IterativeRegularizationResult(
            problem.reference,
            0,
            "exact",
            True,
            tuple(misfits),
            errors,
            oracle,
            target,
            "hybrid_lsqr",
            counted.counts,
            (),
            None,
        )
    u_basis = [[value / beta for value in b]]
    v_basis: list[list[float]] = []
    alphas: list[float] = []
    betas: list[float] = []
    reason = "max_iterations"
    stable_steps = 0
    previous_lambda = previous_norm = None
    v = list(operator.rmatvec(u_basis[0]))
    for k in range(1, limit + 1):
        for _ in range(2):
            for basis in v_basis:
                c = math.fsum(a * bb for a, bb in zip(v, basis, strict=True))
                v = [a - c * bb for a, bb in zip(v, basis, strict=True)]
        alpha = _norm(v)
        if alpha == 0.0:
            reason = "breakdown"
            break
        v = [value / alpha for value in v]
        v_basis.append(v)
        alphas.append(alpha)
        u = [
            a - alpha * c for a, c in zip(operator.matvec(v), u_basis[-1], strict=True)
        ]
        for _ in range(2):
            for basis in u_basis:
                c = math.fsum(a * bb for a, bb in zip(u, basis, strict=True))
                u = [a - c * bb for a, bb in zip(u, basis, strict=True)]
        beta_next = _norm(u)
        betas.append(beta_next)
        bidiagonal = [[0.0] * k for _ in range(k + 1)]
        for j in range(k):
            bidiagonal[j][j] = alphas[j]
            bidiagonal[j + 1][j] = betas[j]
        penalty_r: list[list[float]] | None = None
        if general:
            penalty_columns.append(list(problem.penalty.matvec(v)))
            penalty_r = _upper_factor(penalty_columns)
        if adaptive:
            omega_hat = _adaptive_gcv_weight(bidiagonal, beta)
            if omega_hat is not None:
                omega_hats.append(omega_hat)
            omega = math.fsum(omega_hats) / len(omega_hats) if omega_hats else 1.0
        if rule == "wgcv":
            omegas.append(omega)
        lam = _choose_lambda(bidiagonal, beta, rule, target, fixed, penalty_r, omega)
        if penalty_r is None:
            svd = singular_value_decomposition(tuple(tuple(row) for row in bidiagonal))
            cache = (
                svd.singular_values,
                [list(r) for r in svd.u],
                [list(r) for r in svd.v],
            )
            q, projected_residual, _ = _projected_tikhonov(bidiagonal, beta, lam, cache)
        else:
            q, projected_residual, _ = _projected_general(
                bidiagonal, beta, lam, penalty_r
            )
        z = [math.fsum(v_basis[j][i] * q[j] for j in range(k)) for i in range(n)]
        x = tuple(a + c for a, c in zip(z, problem.reference, strict=True))
        iterates.append(x)
        misfits.append(projected_residual)
        lambdas.append(lam)
        znorm = _norm(z)
        if previous_lambda is not None and previous_norm is not None:
            lam_change = abs(lam - previous_lambda) / max(lam, previous_lambda, 1e-300)
            norm_change = abs(znorm - previous_norm) / max(znorm, previous_norm, 1e-300)
            stable_steps = (
                stable_steps + 1
                if max(lam_change, norm_change) <= stabilization_tolerance
                else 0
            )
            if stable_steps >= 2:
                reason = "stabilized"
                break
        previous_lambda, previous_norm = lam, znorm
        if beta_next == 0.0:
            reason = "breakdown_exact"
            break
        u_basis.append([value / beta_next for value in u])
        v = list(operator.rmatvec(u_basis[-1]))
        v = [a - beta_next * c for a, c in zip(v, v_basis[-1], strict=True)]
    errors, oracle = _errors(iterates, true_x)
    return IterativeRegularizationResult(
        estimate=iterates[-1],
        iterations=len(iterates) - 1,
        stop_reason=reason,
        converged=reason in ("stabilized", "breakdown_exact"),
        misfit_history=tuple(misfits),
        error_history=errors,
        oracle_iteration=oracle,
        target=target,
        method="hybrid_lsqr",
        counts=counted.counts,
        regularization_history=tuple(lambdas),
        krylov=None,
        gcv_weight_history=tuple(omegas) if rule == "wgcv" else None,
    )
