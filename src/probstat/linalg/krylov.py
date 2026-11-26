"""Matrix-free Krylov solvers and randomized spectral estimators.

Least squares ``min ||b - A x||^2 + damp^2 ||x||^2`` is solved by three
methods that only apply ``A`` and ``A'``:

- :func:`cgls`: conjugate gradients on the normal equations without forming
  ``A'A`` [EQ-LINALG-KRY-005];
- :func:`lsqr`: Golub-Kahan bidiagonalization with plane rotations
  [EQ-LINALG-KRY-006], [EQ-LINALG-KRY-007];
- :func:`lsmr`: MINRES on the normal equations from the same
  bidiagonalization, whose ``||A'r||`` decreases monotonically
  [EQ-LINALG-KRY-008].

:func:`conjugate_gradient` solves an SPD system. The spectral helpers are
:func:`power_iteration` (a LOWER estimate of ``||A||_2``, not a certified
bound), :func:`lanczos_eigenpairs` (Ritz pairs of a symmetric operator with
residual norms), :func:`hutchinson_trace` (Rademacher trace estimate with a
standard error) and :func:`randomized_svd` (randomized range finder with
oversampling and power passes, reporting singular-triplet residuals).

Every solver wraps its operator in a :class:`CountingOperator` and reports
the exact numbers of forward and transpose applications. Stopping reasons
are explicit strings; a breakdown or an exhausted budget is never reported as
convergence. Random probes come from :class:`~probstat.foundations.RandomStream`
children, so results are reproducible for an integer seed.

Experimental (added in 0.28).

References:
    [REF-HESTENES-STIEFEL-1952], [REF-PAIGE-SAUNDERS-1982],
    [REF-FONG-SAUNDERS-2011], [REF-GOLUB-KAHAN-1965], [REF-LANCZOS-1950],
    [REF-HUTCHINSON-1989], [REF-HALKO-MARTINSSON-TROPP-2011],
    [REF-GOLUB-VANLOAN-2013, Chapters 10-11].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream

from .dense import Matrix, Vector, symmetric_eigen
from .operators import (
    CountingOperator,
    LinearOperator,
    OperatorCounts,
    _vector,
    require_operator,
)
from .qr import householder_qr
from .svd import singular_value_decomposition

IterationCallback = Callable[[int, Vector, float], bool]
"""Called as ``callback(k, x_k, residual_norm_k)``; return ``True`` to stop."""


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.hypot(*vector) if vector else 0.0


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the compensated inner product.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _axpy(alpha: float, x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Return ``alpha x + y``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return [alpha * a + b for a, b in zip(x, y, strict=True)]


def _scale(alpha: float, x: Sequence[float]) -> list[float]:
    """Return ``alpha x``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return [alpha * a for a in x]


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """Return a stable plane rotation ``(c, s, r)`` with ``[c s; -s c][a; b] = [r; 0]``.

    References:
        [REF-PAIGE-SAUNDERS-1982, Section 4], [EQ-LINALG-KRY-007].

    """
    if b == 0.0:
        return (1.0 if a >= 0.0 else -1.0), 0.0, abs(a)
    if a == 0.0:
        return 0.0, (1.0 if b >= 0.0 else -1.0), abs(b)
    r = math.hypot(a, b)
    return a / r, b / r, r


def _require_positive_int(value: object, name: str) -> int:
    """Validate a positive integer.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _require_tolerance(value: object, name: str, *, allow_zero: bool = True) -> float:
    """Validate a finite non-negative tolerance.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class KrylovResult:
    """Result of a matrix-free least-squares or SPD solve.

    Attributes:
        solution: Final iterate ``x_k``.
        iterations: Number of completed iterations ``k``.
        stop_reason: ``"zero_rhs"``, ``"exact"`` (``b = A x_0`` or ``A'r_0 = 0``
            at the start), ``"residual_tolerance"``, ``"normal_tolerance"``,
            ``"condition_limit"``, ``"callback"``, ``"breakdown"`` or
            ``"max_iterations"``.
        converged: ``True`` for the first five reasons (tolerance met or exact
            solution); ``False`` for ``"breakdown"`` and ``"max_iterations"``;
            ``True`` for ``"callback"`` (the caller's rule decided).
        residual_norm: Data residual ``||b - A x_k||`` (recursive estimate,
            except where ``true_residual_norm`` is reported).
        augmented_residual_norm: ``sqrt(||b - A x||^2 + damp^2 ||x||^2)``.
        normal_residual_norm: ``||A'(b - A x) - damp^2 x||`` (recursive
            estimate for LSQR/LSMR).
        solution_norm: ``||x_k||``.
        true_residual_norm: ``||b - A x_k||`` recomputed once at the end
            (one extra forward application, counted).
        operator_norm_estimate: Frobenius-type estimate of ``||[A; damp I]||``
            from the bidiagonalization (``nan`` for CG/CGLS).
        condition_estimate: Estimate of the condition number (``nan`` when
            not available); a lower bound, not a certificate.
        residual_history: Recursive data residual norms, starting at
            ``k = 0``.
        normal_residual_history: Recursive normal residual norms, starting at
            ``k = 0``.
        counts: Forward and transpose applications, including the final
            residual check.
        damping: The damping ``damp`` (``0`` for CG/CGLS without damping).
        method: ``"cg"``, ``"cgls"``, ``"lsqr"`` or ``"lsmr"``.

    References:
        [REF-PAIGE-SAUNDERS-1982], [REF-FONG-SAUNDERS-2011],
        [REF-HESTENES-STIEFEL-1952].

    """

    solution: Vector
    iterations: int
    stop_reason: str
    converged: bool
    residual_norm: float
    augmented_residual_norm: float
    normal_residual_norm: float
    solution_norm: float
    true_residual_norm: float
    operator_norm_estimate: float
    condition_estimate: float
    residual_history: Vector
    normal_residual_history: Vector
    counts: OperatorCounts
    damping: float
    method: str


_CONVERGED_REASONS = frozenset(
    {
        "zero_rhs",
        "exact",
        "residual_tolerance",
        "normal_tolerance",
        "condition_limit",
        "callback",
    }
)


def _setup(
    operator: LinearOperator,
    data: Sequence[float],
    initial: Sequence[float] | None,
) -> tuple[CountingOperator, Vector, list[float]]:
    """Wrap the operator, validate ``b`` and the start ``x_0``.

    References:
        [EQ-LINALG-OP-001].

    """
    counted = CountingOperator(require_operator(operator))
    rows, columns = counted.shape
    if rows == 0 or columns == 0:
        raise ValueError("operator must have non-empty dimensions")
    b = _vector(data, rows, "data")
    x0 = (
        [0.0] * columns
        if initial is None
        else list(_vector(initial, columns, "initial"))
    )
    return counted, b, x0


def _finish(
    counted: CountingOperator,
    b: Vector,
    x: Sequence[float],
    *,
    iterations: int,
    reason: str,
    residual: float,
    normal: float,
    damping: float,
    norm_estimate: float,
    condition: float,
    residuals: list[float],
    normals: list[float],
    method: str,
) -> KrylovResult:
    """Assemble a :class:`KrylovResult` with one counted true-residual check.

    References:
        [REF-PAIGE-SAUNDERS-1982, Section 6].

    """
    solution = tuple(x)
    ax = counted.matvec(solution)
    true_residual = _norm([bi - ai for bi, ai in zip(b, ax, strict=True)])
    xnorm = _norm(solution)
    return KrylovResult(
        solution=solution,
        iterations=iterations,
        stop_reason=reason,
        converged=reason in _CONVERGED_REASONS,
        residual_norm=residual,
        augmented_residual_norm=math.hypot(residual, damping * xnorm),
        normal_residual_norm=normal,
        solution_norm=xnorm,
        true_residual_norm=true_residual,
        operator_norm_estimate=norm_estimate,
        condition_estimate=condition,
        residual_history=tuple(residuals),
        normal_residual_history=tuple(normals),
        counts=counted.counts,
        damping=damping,
        method=method,
    )


def cgls(
    operator: LinearOperator,
    data: Sequence[float],
    *,
    damping: float = 0.0,
    initial: Sequence[float] | None = None,
    residual_tolerance: float = 1e-10,
    normal_tolerance: float = 1e-10,
    max_iterations: int | None = None,
    callback: IterationCallback | None = None,
) -> KrylovResult:
    """Solve ``min ||b - A x||^2 + damp^2 ||x||^2`` by CGLS.

    With ``r = b - A x``, ``s = A'r - damp^2 x`` and ``p = s`` initially, each
    step sets ``q = A p``, ``alpha = ||s||^2 / (||q||^2 + damp^2 ||p||^2)``,
    ``x += alpha p``, ``r -= alpha q``, ``s' = A'r - damp^2 x``,
    ``beta = ||s'||^2 / ||s||^2`` and ``p = s' + beta p`` [EQ-LINALG-KRY-005].
    Stops when ``||r|| <= residual_tolerance ||b||`` or
    ``||s|| <= normal_tolerance ||s_0||``. From ``x_0 = 0`` the iterates stay
    in ``range(A')`` and converge to the minimum-norm solution of a
    rank-deficient problem. Every 50 iterations the residual is recomputed
    from ``x`` to control recursive drift (counted).

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``b`` of length ``m``.
        damping: Non-negative ``damp``.
        initial: Start ``x_0``; zero by default.
        residual_tolerance: Relative tolerance on ``||r||``.
        normal_tolerance: Relative tolerance on the normal residual.
        max_iterations: Iteration cap (default ``2 n``).
        callback: Optional ``callback(k, x_k, ||r_k||)`` returning ``True`` to
            stop with reason ``"callback"``.

    Returns:
        :class:`KrylovResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension or tolerance is invalid.

    References:
        [EQ-LINALG-KRY-005], [REF-HESTENES-STIEFEL-1952],
        [REF-HANSEN-2008, Section 2.8].

    """
    counted, b, x = _setup(operator, data, initial)
    damp = _require_tolerance(damping, "damping")
    rtol = _require_tolerance(residual_tolerance, "residual_tolerance")
    ntol = _require_tolerance(normal_tolerance, "normal_tolerance")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    damp2 = damp * damp
    bnorm = _norm(b)
    ax = counted.matvec(x)
    r = [bi - ai for bi, ai in zip(b, ax, strict=True)]
    s = _axpy(-damp2, x, counted.rmatvec(r))
    rnorm, snorm = _norm(r), _norm(s)
    residuals, normals = [rnorm], [snorm]
    nan = math.nan
    if bnorm == 0.0 and _norm(x) == 0.0:
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason="zero_rhs",
            residual=0.0,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=residuals,
            normals=normals,
            method="cgls",
        )
    if rnorm == 0.0 or snorm == 0.0:
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason="exact",
            residual=rnorm,
            normal=snorm,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=residuals,
            normals=normals,
            method="cgls",
        )
    s0 = snorm
    p = list(s)
    gamma = snorm * snorm
    reason = "max_iterations"
    iterations = 0
    for k in range(1, limit + 1):
        q = counted.matvec(p)
        denominator = _dot(q, q) + damp2 * _dot(p, p)
        if denominator <= 0.0 or not math.isfinite(denominator):
            reason = "breakdown"
            break
        alpha = gamma / denominator
        x = _axpy(alpha, p, x)
        if k % 50 == 0:
            ax = counted.matvec(x)
            r = [bi - ai for bi, ai in zip(b, ax, strict=True)]
        else:
            r = _axpy(-alpha, q, r)
        s = _axpy(-damp2, x, counted.rmatvec(r))
        rnorm, snorm = _norm(r), _norm(s)
        residuals.append(rnorm)
        normals.append(snorm)
        iterations = k
        if callback is not None and callback(k, tuple(x), rnorm):
            reason = "callback"
            break
        if rnorm <= rtol * bnorm:
            reason = "residual_tolerance"
            break
        if snorm <= ntol * s0:
            reason = "normal_tolerance"
            break
        gamma_new = snorm * snorm
        beta = gamma_new / gamma
        gamma = gamma_new
        p = _axpy(beta, p, s)
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=rnorm,
        normal=snorm,
        damping=damp,
        norm_estimate=nan,
        condition=nan,
        residuals=residuals,
        normals=normals,
        method="cgls",
    )


def lsqr(
    operator: LinearOperator,
    data: Sequence[float],
    *,
    damping: float = 0.0,
    initial: Sequence[float] | None = None,
    atol: float = 1e-10,
    btol: float = 1e-10,
    condition_limit: float = 1e8,
    max_iterations: int | None = None,
    callback: IterationCallback | None = None,
) -> KrylovResult:
    """Solve ``min ||b - A x||^2 + damp^2 ||x||^2`` by LSQR.

    Golub-Kahan bidiagonalization ``beta_1 u_1 = b - A x_0``,
    ``alpha_1 v_1 = A'u_1``, ``beta_(k+1) u_(k+1) = A v_k - alpha_k u_k``,
    ``alpha_(k+1) v_(k+1) = A'u_(k+1) - beta_(k+1) v_k``
    [EQ-LINALG-KRY-006] is combined with plane rotations that eliminate the
    damping and the subdiagonal [EQ-LINALG-KRY-007]. The stopping tests of
    Paige and Saunders are used: ``||r|| <= btol ||b|| + atol ||A|| ||x||``
    (compatible system), ``||A'r|| <= atol ||A|| ||r||`` (least squares) and
    ``cond(A) >= condition_limit``. Workspace is ``O(m + n)``; no basis is
    stored.

    Args:
        operator: ``A`` or a dense matrix.
        data: ``b``.
        damping: Non-negative ``damp``.
        initial: Start ``x_0`` (the method solves for ``x - x_0``).
        atol: Relative tolerance on ``A`` (see above).
        btol: Relative tolerance on ``b``.
        condition_limit: Stop when the condition estimate exceeds this
            (``> 1``).
        max_iterations: Iteration cap (default ``2 n``).
        callback: Optional ``callback(k, x_k, ||r_k||)``.

    Returns:
        :class:`KrylovResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension or tolerance is invalid.

    References:
        [EQ-LINALG-KRY-006], [EQ-LINALG-KRY-007], [REF-PAIGE-SAUNDERS-1982],
        [REF-GOLUB-KAHAN-1965].

    """
    counted, b, x0 = _setup(operator, data, initial)
    damp = _require_tolerance(damping, "damping")
    a_tol = _require_tolerance(atol, "atol")
    b_tol = _require_tolerance(btol, "btol")
    conlim = _require_tolerance(condition_limit, "condition_limit")
    if conlim <= 1.0:
        raise ValueError("condition_limit must exceed one")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    nan = math.nan
    ax0 = counted.matvec(x0)
    u = [bi - ai for bi, ai in zip(b, ax0, strict=True)]
    beta = _norm(u)
    bnorm = _norm(b)
    x = list(x0)
    if beta == 0.0:
        reason = "zero_rhs" if bnorm == 0.0 and _norm(x0) == 0.0 else "exact"
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason=reason,
            residual=0.0,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[0.0],
            normals=[0.0],
            method="lsqr",
        )
    u = _scale(1.0 / beta, u)
    v = list(counted.rmatvec(u))
    alpha = _norm(v)
    if alpha == 0.0:
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason="exact",
            residual=beta,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[beta],
            normals=[0.0],
            method="lsqr",
        )
    v = _scale(1.0 / alpha, v)
    w = list(v)
    z = [0.0] * n
    phibar, rhobar = beta, alpha
    anorm2 = 0.0
    ddnorm = 0.0
    res2 = 0.0
    residuals, normals = [beta], [alpha * beta]
    reason = "max_iterations"
    iterations = 0
    rnorm = beta
    arnorm = alpha * beta
    anorm = acond = nan
    damp2 = damp * damp
    for k in range(1, limit + 1):
        av = counted.matvec(v)
        u = _axpy(-alpha, u, av)
        beta = _norm(u)
        if beta > 0.0:
            u = _scale(1.0 / beta, u)
        anorm2 += alpha * alpha + beta * beta + damp2
        anorm = math.sqrt(anorm2)
        if beta > 0.0:
            atu = counted.rmatvec(u)
            v = _axpy(-beta, v, atu)
            alpha = _norm(v)
            if alpha > 0.0:
                v = _scale(1.0 / alpha, v)
        else:
            alpha = 0.0
        # Eliminate the damping parameter (rotation of [rhobar; damp]).
        cs1, sn1, rhobar1 = _sym_ortho(rhobar, damp)
        psi = sn1 * phibar
        phibar = cs1 * phibar
        # Eliminate the subdiagonal beta_(k+1).
        cs, sn, rho = _sym_ortho(rhobar1, beta)
        if rho == 0.0:
            reason = "breakdown"
            break
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        t1 = phi / rho
        t2 = -theta / rho
        dk = _scale(1.0 / rho, w)
        z = _axpy(t1, w, z)
        w = _axpy(t2, w, v)
        ddnorm += _dot(dk, dk)
        x = [a + c for a, c in zip(x0, z, strict=True)]
        res2 += psi * psi
        augmented = math.sqrt(phibar * phibar + res2)
        znorm = _norm(z)
        rnorm = math.sqrt(max(augmented * augmented - damp2 * znorm * znorm, 0.0))
        arnorm = alpha * abs(tau)
        acond = anorm * math.sqrt(ddnorm)
        residuals.append(rnorm)
        normals.append(arnorm)
        iterations = k
        if callback is not None and callback(k, tuple(x), rnorm):
            reason = "callback"
            break
        if augmented <= b_tol * bnorm + a_tol * anorm * znorm:
            reason = "residual_tolerance"
            break
        if augmented > 0.0 and arnorm <= a_tol * anorm * augmented:
            reason = "normal_tolerance"
            break
        if augmented == 0.0 or arnorm == 0.0:
            reason = "normal_tolerance"
            break
        if acond >= conlim:
            reason = "condition_limit"
            break
        if alpha == 0.0:
            reason = "normal_tolerance"
            break
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=rnorm,
        normal=arnorm,
        damping=damp,
        norm_estimate=anorm,
        condition=acond,
        residuals=residuals,
        normals=normals,
        method="lsqr",
    )


def lsmr(
    operator: LinearOperator,
    data: Sequence[float],
    *,
    damping: float = 0.0,
    initial: Sequence[float] | None = None,
    atol: float = 1e-10,
    btol: float = 1e-10,
    condition_limit: float = 1e8,
    max_iterations: int | None = None,
    callback: IterationCallback | None = None,
) -> KrylovResult:
    """Solve ``min ||b - A x||^2 + damp^2 ||x||^2`` by LSMR.

    LSMR applies MINRES to the normal equations
    ``(A'A + damp^2 I) x = A'b`` through the bidiagonalization of
    [EQ-LINALG-KRY-006], so that ``||A'r_k - damp^2 x_k||`` is monotone
    [EQ-LINALG-KRY-008]. The residual norm is estimated by the recurrences of
    Fong and Saunders; stopping tests mirror :func:`lsqr`.

    Args:
        operator: ``A`` or a dense matrix.
        data: ``b``.
        damping: Non-negative ``damp``.
        initial: Start ``x_0``.
        atol: Relative tolerance on ``A``.
        btol: Relative tolerance on ``b``.
        condition_limit: Stop when the condition estimate exceeds this.
        max_iterations: Iteration cap (default ``2 n``).
        callback: Optional ``callback(k, x_k, ||r_k||)``.

    Returns:
        :class:`KrylovResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension or tolerance is invalid.

    References:
        [EQ-LINALG-KRY-008], [REF-FONG-SAUNDERS-2011].

    """
    counted, b, x0 = _setup(operator, data, initial)
    damp = _require_tolerance(damping, "damping")
    a_tol = _require_tolerance(atol, "atol")
    b_tol = _require_tolerance(btol, "btol")
    conlim = _require_tolerance(condition_limit, "condition_limit")
    if conlim <= 1.0:
        raise ValueError("condition_limit must exceed one")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    nan = math.nan
    ax0 = counted.matvec(x0)
    u = [bi - ai for bi, ai in zip(b, ax0, strict=True)]
    beta = _norm(u)
    bnorm = _norm(b)
    if beta == 0.0:
        reason = "zero_rhs" if bnorm == 0.0 and _norm(x0) == 0.0 else "exact"
        return _finish(
            counted,
            b,
            list(x0),
            iterations=0,
            reason=reason,
            residual=0.0,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[0.0],
            normals=[0.0],
            method="lsmr",
        )
    u = _scale(1.0 / beta, u)
    v = list(counted.rmatvec(u))
    alpha = _norm(v)
    if alpha == 0.0:
        return _finish(
            counted,
            b,
            list(x0),
            iterations=0,
            reason="exact",
            residual=beta,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[beta],
            normals=[0.0],
            method="lsmr",
        )
    v = _scale(1.0 / alpha, v)
    zetabar = alpha * beta
    alphabar = alpha
    rho = rhobar = cbar = 1.0
    sbar = 0.0
    h = list(v)
    hbar = [0.0] * n
    z = [0.0] * n
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0
    norm_a2 = alpha * alpha
    maxrbar = 0.0
    minrbar = math.inf
    residuals, normals = [beta], [alpha * beta]
    reason = "max_iterations"
    iterations = 0
    normr = beta
    normar = alpha * beta
    norm_a = cond_a = nan
    x = list(x0)
    for k in range(1, limit + 1):
        u = _axpy(-alpha, u, counted.matvec(v))
        beta = _norm(u)
        if beta > 0.0:
            u = _scale(1.0 / beta, u)
            v = _axpy(-beta, v, counted.rmatvec(u))
            alpha = _norm(v)
            if alpha > 0.0:
                v = _scale(1.0 / alpha, v)
        else:
            alpha = 0.0
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        if rho == 0.0:
            reason = "breakdown"
            break
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        if rhobar == 0.0:
            reason = "breakdown"
            break
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        factor = thetabar * rho / (rhoold * rhobarold)
        hbar = _axpy(-factor, hbar, h)
        z = _axpy(zeta / (rho * rhobar), hbar, z)
        h = _axpy(-thetanew / rho, h, v)
        # Residual-norm recurrences.
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d += betacheck * betacheck
        augmented = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        norm_a2 += beta * beta
        norm_a = math.sqrt(norm_a2)
        norm_a2 += alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if k > 1:
            minrbar = min(minrbar, rhobarold)
        cond_a = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        x = [a + c2 for a, c2 in zip(x0, z, strict=True)]
        znorm = _norm(z)
        normr = math.sqrt(max(augmented * augmented - damp * damp * znorm * znorm, 0.0))
        residuals.append(normr)
        normals.append(normar)
        iterations = k
        if callback is not None and callback(k, tuple(x), normr):
            reason = "callback"
            break
        if augmented <= b_tol * bnorm + a_tol * norm_a * znorm:
            reason = "residual_tolerance"
            break
        if augmented == 0.0 or normar <= a_tol * norm_a * augmented:
            reason = "normal_tolerance"
            break
        if cond_a >= conlim:
            reason = "condition_limit"
            break
        if alpha == 0.0:
            reason = "normal_tolerance"
            break
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=normr,
        normal=normar,
        damping=damp,
        norm_estimate=norm_a,
        condition=cond_a,
        residuals=residuals,
        normals=normals,
        method="lsmr",
    )


def conjugate_gradient(
    operator: LinearOperator,
    data: Sequence[float],
    *,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    callback: IterationCallback | None = None,
) -> KrylovResult:
    """Solve ``M x = b`` for a symmetric positive-definite operator ``M`` by CG.

    ``r = b - M x``, ``p = r``; ``alpha = r'r / p'M p``, ``x += alpha p``,
    ``r -= alpha M p``, ``beta = r_new'r_new / r'r``, ``p = r + beta p``
    [EQ-LINALG-KRY-004]. Only ``matvec`` is used; symmetry is assumed and
    not checked. A non-positive curvature ``p'M p <= 0`` stops with reason
    ``"breakdown"`` (the operator is not positive definite on the Krylov
    space).

    Args:
        operator: Square SPD operator ``M``.
        data: ``b``.
        initial: Start ``x_0``.
        tolerance: Relative tolerance ``||r|| <= tol ||b||``.
        max_iterations: Iteration cap (default ``2 n``).
        callback: Optional ``callback(k, x_k, ||r_k||)``.

    Returns:
        :class:`KrylovResult`; ``normal_residual_norm`` equals the residual
        norm.

    Raises:
        ValueError: If the operator is not square or an argument is invalid.

    References:
        [EQ-LINALG-KRY-004], [REF-HESTENES-STIEFEL-1952].

    """
    counted, b, x = _setup(operator, data, initial)
    if counted.shape[0] != counted.shape[1]:
        raise ValueError("conjugate_gradient needs a square operator")
    tol = _require_tolerance(tolerance, "tolerance")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    nan = math.nan
    bnorm = _norm(b)
    r = [bi - ai for bi, ai in zip(b, counted.matvec(x), strict=True)]
    rnorm = _norm(r)
    residuals = [rnorm]
    if rnorm == 0.0:
        reason = "zero_rhs" if bnorm == 0.0 else "exact"
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason=reason,
            residual=0.0,
            normal=0.0,
            damping=0.0,
            norm_estimate=nan,
            condition=nan,
            residuals=residuals,
            normals=residuals,
            method="cg",
        )
    p = list(r)
    gamma = rnorm * rnorm
    reason = "max_iterations"
    iterations = 0
    for k in range(1, limit + 1):
        q = counted.matvec(p)
        curvature = _dot(p, q)
        if curvature <= 0.0 or not math.isfinite(curvature):
            reason = "breakdown"
            break
        alpha = gamma / curvature
        x = _axpy(alpha, p, x)
        r = _axpy(-alpha, q, r)
        rnorm = _norm(r)
        residuals.append(rnorm)
        iterations = k
        if callback is not None and callback(k, tuple(x), rnorm):
            reason = "callback"
            break
        if rnorm <= tol * bnorm:
            reason = "residual_tolerance"
            break
        gamma_new = rnorm * rnorm
        p = _axpy(gamma_new / gamma, p, r)
        gamma = gamma_new
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=rnorm,
        normal=rnorm,
        damping=0.0,
        norm_estimate=nan,
        condition=nan,
        residuals=residuals,
        normals=residuals,
        method="cg",
    )


@dataclass(frozen=True, slots=True)
class SpectralNormEstimate:
    """Power-iteration estimate of ``||A||_2``.

    Attributes:
        estimate: ``sqrt(v'A'A v)`` at the final unit vector ``v``; a LOWER
            bound on ``||A||_2`` up to rounding, never a certified upper
            bound.
        iterations: Iterations performed.
        converged: Whether the relative change fell below the tolerance.
        relative_change: Last relative change of the estimate.
        vector: Final unit right vector ``v``.
        counts: Forward and transpose applications.

    References:
        [EQ-LINALG-KRY-001], [REF-GOLUB-VANLOAN-2013, Section 8.2].

    """

    estimate: float
    iterations: int
    converged: bool
    relative_change: float
    vector: Vector
    counts: OperatorCounts


def _gaussian_vector(stream: RandomStream, size: int) -> list[float]:
    """Return ``size`` standard normal draws from a stream.

    References:
        [REF-PROBSTAT-API]. No equation is evaluated.

    """
    generator = stream.generator()
    return [generator.gauss(0.0, 1.0) for _ in range(size)]


def power_iteration(
    operator: LinearOperator,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    seed: int | RandomStream | None = 0,
) -> SpectralNormEstimate:
    """Estimate ``||A||_2`` by power iteration on ``A'A``.

    ``v_(k+1) = A'A v_k / ||A'A v_k||`` from a Gaussian start, with estimate
    ``||A v_k||`` [EQ-LINALG-KRY-001]. The estimate increases towards
    ``sigma_1`` but is a lower bound: a step size ``omega < 2 / ||A||^2`` set
    from it is NOT guaranteed safe. Repeated leading singular values do not
    slow convergence of the value; a gap ``sigma_2 / sigma_1`` near one does.

    Args:
        operator: ``A``.
        max_iterations: Positive iteration cap.
        tolerance: Relative change threshold.
        seed: Seed or stream for the start vector.

    Returns:
        :class:`SpectralNormEstimate`; ``0`` for the zero operator.

    References:
        [EQ-LINALG-KRY-001], [REF-GOLUB-VANLOAN-2013, Section 8.2].

    """
    counted = CountingOperator(require_operator(operator))
    limit = _require_positive_int(max_iterations, "max_iterations")
    tol = _require_tolerance(tolerance, "tolerance")
    n = counted.shape[1]
    v = _gaussian_vector(resolve_stream(seed).child("power_iteration"), n)
    vnorm = _norm(v)
    v = _scale(1.0 / vnorm, v)
    estimate = 0.0
    change = math.inf
    converged = False
    iterations = 0
    for k in range(1, limit + 1):
        av = counted.matvec(v)
        new_estimate = _norm(av)
        iterations = k
        if new_estimate == 0.0:
            return SpectralNormEstimate(0.0, k, True, 0.0, tuple(v), counted.counts)
        change = abs(new_estimate - estimate) / new_estimate
        estimate = new_estimate
        if change <= tol and k > 1:
            converged = True
            break
        w = counted.rmatvec(av)
        wnorm = _norm(w)
        if wnorm == 0.0:
            break
        v = _scale(1.0 / wnorm, w)
    return SpectralNormEstimate(
        estimate, iterations, converged, change, tuple(v), counted.counts
    )


@dataclass(frozen=True, slots=True)
class LanczosResult:
    """Ritz pairs of a symmetric operator from ``k`` Lanczos steps.

    Attributes:
        eigenvalues: Ritz values in DESCENDING order.
        eigenvectors: Ritz vectors as rows (one per Ritz value), unit norm.
        residual_norms: ``||M y - theta y|| = beta_k |e_k's|`` for each pair.
        steps: Lanczos steps performed.
        breakdown: Whether an invariant subspace was found early
            (``beta_j`` below tolerance); the Ritz values are then exact
            eigenvalues of ``M`` restricted to that subspace.
        counts: Operator applications.

    References:
        [EQ-LINALG-KRY-002], [REF-LANCZOS-1950].

    """

    eigenvalues: Vector
    eigenvectors: Matrix
    residual_norms: Vector
    steps: int
    breakdown: bool
    counts: OperatorCounts


def lanczos_eigenpairs(
    operator: LinearOperator,
    steps: int,
    *,
    reorthogonalize: bool = True,
    seed: int | RandomStream | None = 0,
    breakdown_tolerance: float = 1e-12,
) -> LanczosResult:
    """Return Ritz pairs of a symmetric operator after ``steps`` Lanczos steps.

    ``beta_(j+1) q_(j+1) = M q_j - alpha_j q_j - beta_j q_(j-1)`` builds the
    tridiagonal ``T_k = Q_k'M Q_k`` [EQ-LINALG-KRY-002]; its eigenpairs
    ``(theta_i, s_i)`` give Ritz vectors ``y_i = Q_k s_i`` with residual
    ``beta_(k+1) |e_k's_i|``. Full reorthogonalization (default) stores the
    ``k`` basis vectors (``O(k n)`` memory) and removes the loss of
    orthogonality that otherwise produces spurious copies of converged
    eigenvalues. Symmetry of ``M`` is assumed, not checked.

    Args:
        operator: Square symmetric operator.
        steps: Number of steps ``k`` (at most ``n``).
        reorthogonalize: Whether to reorthogonalize against all previous
            vectors (twice, classical Gram-Schmidt).
        seed: Seed or stream for the start vector.
        breakdown_tolerance: Relative threshold on ``beta`` for an
            invariant subspace.

    Returns:
        :class:`LanczosResult`.

    Raises:
        ValueError: If the operator is not square or ``steps`` is invalid.

    References:
        [EQ-LINALG-KRY-002], [REF-LANCZOS-1950],
        [REF-GOLUB-VANLOAN-2013, Section 10.1].

    """
    counted = CountingOperator(require_operator(operator))
    n = counted.shape[0]
    if counted.shape[1] != n:
        raise ValueError("lanczos_eigenpairs needs a square operator")
    k_max = _require_positive_int(steps, "steps")
    if k_max > n:
        raise ValueError("steps must not exceed the dimension")
    tol = _require_tolerance(breakdown_tolerance, "breakdown_tolerance")
    q = _gaussian_vector(resolve_stream(seed).child("lanczos"), n)
    q = _scale(1.0 / _norm(q), q)
    basis: list[list[float]] = [q]
    alphas: list[float] = []
    betas: list[float] = []
    previous = [0.0] * n
    beta = 0.0
    breakdown = False
    scale = 0.0
    for j in range(k_max):
        w = list(counted.matvec(basis[j]))
        alpha = _dot(w, basis[j])
        alphas.append(alpha)
        w = _axpy(-alpha, basis[j], w)
        w = _axpy(-beta, previous, w)
        if reorthogonalize:
            for _ in range(2):
                for vector in basis:
                    w = _axpy(-_dot(w, vector), vector, w)
        beta = _norm(w)
        scale = max(scale, abs(alpha), beta)
        if j == k_max - 1:
            betas.append(beta)
            break
        if beta <= tol * max(scale, 1e-300):
            breakdown = True
            betas.append(beta)
            break
        betas.append(beta)
        previous = basis[j]
        basis.append(_scale(1.0 / beta, w))
    k = len(alphas)
    tridiagonal = tuple(
        tuple(
            alphas[i] if i == j else (betas[min(i, j)] if abs(i - j) == 1 else 0.0)
            for j in range(k)
        )
        for i in range(k)
    )
    values, vectors = symmetric_eigen(tridiagonal)
    order = sorted(range(k), key=lambda index: values[index], reverse=True)
    last_beta = betas[k - 1]
    ritz_values = tuple(values[i] for i in order)
    ritz_vectors = []
    residual_norms = []
    for i in order:
        coefficients = [vectors[row][i] for row in range(k)]
        y = [0.0] * n
        for c, vector in zip(coefficients, basis, strict=False):
            y = _axpy(c, vector, y)
        ynorm = _norm(y)
        ritz_vectors.append(tuple(_scale(1.0 / ynorm, y)) if ynorm > 0.0 else tuple(y))
        residual_norms.append(abs(last_beta * coefficients[-1]))
    return LanczosResult(
        eigenvalues=ritz_values,
        eigenvectors=tuple(ritz_vectors),
        residual_norms=tuple(residual_norms),
        steps=k,
        breakdown=breakdown,
        counts=counted.counts,
    )


@dataclass(frozen=True, slots=True)
class TraceEstimate:
    """Hutchinson estimate of ``tr(M)``.

    Attributes:
        estimate: Mean of ``z_k'M z_k`` over the probes.
        standard_error: Sample standard deviation over the probes divided by
            ``sqrt(s)`` (``nan`` for one probe).
        probes: Number ``s`` of independent Rademacher probes.
        samples: The individual ``z_k'M z_k``.
        counts: Operator applications (``s`` forward).

    References:
        [EQ-LINALG-KRY-003], [REF-HUTCHINSON-1989].

    """

    estimate: float
    standard_error: float
    probes: int
    samples: Vector
    counts: OperatorCounts


def hutchinson_trace(
    operator: LinearOperator,
    *,
    probes: int = 30,
    seed: int | RandomStream | None = 0,
) -> TraceEstimate:
    """Estimate ``tr(M)`` with independent Rademacher probes.

    ``tr(M) ~= (1/s) sum_k z_k'M z_k`` with ``z_k`` uniform on ``{-1, 1}^n``
    is unbiased with variance ``2 (||M||_F^2 - sum_i M_ii^2) / s`` for a
    symmetric ``M`` [EQ-LINALG-KRY-003]; it is exact for a diagonal ``M``.
    Probe ``k`` is drawn from the stream child ``("hutchinson", k)``, so the
    same seed gives the same probes for every operator (common random numbers
    across a parameter path).

    Args:
        operator: Square operator.
        probes: Positive number of probes ``s``.
        seed: Seed or stream.

    Returns:
        :class:`TraceEstimate`.

    Raises:
        ValueError: If the operator is not square.

    References:
        [EQ-LINALG-KRY-003], [REF-HUTCHINSON-1989].

    """
    counted = CountingOperator(require_operator(operator))
    n = counted.shape[0]
    if counted.shape[1] != n:
        raise ValueError("hutchinson_trace needs a square operator")
    count = _require_positive_int(probes, "probes")
    stream = resolve_stream(seed)
    samples = []
    for k in range(count):
        generator = stream.child("hutchinson", k).generator()
        z = [1.0 if generator.random() < 0.5 else -1.0 for _ in range(n)]
        samples.append(_dot(z, counted.matvec(z)))
    mean = math.fsum(samples) / count
    if count > 1:
        variance = math.fsum((value - mean) ** 2 for value in samples) / (count - 1)
        error = math.sqrt(variance / count)
    else:
        error = math.nan
    return TraceEstimate(mean, error, count, tuple(samples), counted.counts)


@dataclass(frozen=True, slots=True)
class RandomizedSVD:
    """Approximate truncated SVD ``A ~= U diag(sigma) V'`` from a random range.

    Attributes:
        u: ``m x k`` left vectors as rows of length ``k``.
        singular_values: ``k`` approximate singular values, descending.
        v: ``n x k`` right vectors as rows of length ``k``.
        right_residuals: ``||A v_i - sigma_i u_i||``.
        left_residuals: ``||A'u_i - sigma_i v_i||``.
        sample_size: Range sample size ``l = k + oversampling`` (capped).
        power_iterations: Power passes ``q``.
        counts: Operator applications.

    References:
        [EQ-LINALG-KRY-009], [REF-HALKO-MARTINSSON-TROPP-2011].

    """

    u: Matrix
    singular_values: Vector
    v: Matrix
    right_residuals: Vector
    left_residuals: Vector
    sample_size: int
    power_iterations: int
    counts: OperatorCounts


def _orthonormal_columns(columns: list[list[float]]) -> list[list[float]]:
    """Return an orthonormal basis (as columns) of the span of ``columns``.

    Uses Householder QR of the tall matrix whose columns are given.

    References:
        [EQ-LINALG-009], [REF-GOLUB-VANLOAN-2013, Section 5.2].

    """
    rows = len(columns[0])
    matrix = tuple(tuple(column[i] for column in columns) for i in range(rows))
    factor = householder_qr(matrix)
    return [[factor.q[i][j] for i in range(rows)] for j in range(len(columns))]


def randomized_svd(
    operator: LinearOperator,
    rank: int,
    *,
    oversampling: int = 10,
    power_iterations: int = 2,
    seed: int | RandomStream | None = 0,
) -> RandomizedSVD:
    """Return a rank-``k`` approximate SVD by the randomized range finder.

    ``Y = A Omega`` with a Gaussian ``n x l`` test matrix, ``l = k + p``;
    ``q`` power passes ``Y <- A (A'Q)`` with re-orthonormalization by QR after
    every application; then ``B = Q'A`` (computed as ``A'Q``), the small SVD
    ``B = U_B S V'`` and ``U = Q U_B`` [EQ-LINALG-KRY-009]. The reported
    residuals are a posteriori checks; the rank is not certified, and
    singular values below the noise of the sample are not accurate.

    Args:
        operator: ``A``.
        rank: Target rank ``k`` (positive).
        oversampling: Extra samples ``p >= 0``.
        power_iterations: Power passes ``q >= 0``.
        seed: Seed or stream for ``Omega``.

    Returns:
        :class:`RandomizedSVD`; ``l`` is capped at ``min(m, n)``.

    Raises:
        ValueError: If ``rank`` exceeds ``min(m, n)`` or an argument is invalid.

    References:
        [EQ-LINALG-KRY-009], [REF-HALKO-MARTINSSON-TROPP-2011, Algorithms 4.4
        and 5.1].

    """
    counted = CountingOperator(require_operator(operator))
    m, n = counted.shape
    k = _require_positive_int(rank, "rank")
    if k > min(m, n):
        raise ValueError("rank must not exceed min(m, n)")
    if (
        isinstance(oversampling, bool)
        or not isinstance(oversampling, int)
        or oversampling < 0
    ):
        raise ValueError("oversampling must be a non-negative integer")
    if (
        isinstance(power_iterations, bool)
        or not isinstance(power_iterations, int)
        or power_iterations < 0
    ):
        raise ValueError("power_iterations must be a non-negative integer")
    sample = min(k + oversampling, m, n)
    stream = resolve_stream(seed)
    omega = [
        _gaussian_vector(stream.child("randomized_svd", j), n) for j in range(sample)
    ]
    q_columns = _orthonormal_columns([list(counted.matvec(column)) for column in omega])
    for _ in range(power_iterations):
        z_columns = _orthonormal_columns(
            [list(counted.rmatvec(column)) for column in q_columns]
        )
        q_columns = _orthonormal_columns(
            [list(counted.matvec(column)) for column in z_columns]
        )
    bt_columns = [
        counted.rmatvec(column) for column in q_columns
    ]  # columns of B' (n x l)
    bt = tuple(tuple(column[i] for column in bt_columns) for i in range(n))
    small = singular_value_decomposition(bt)  # B' = U_s S V_s'  =>  B = V_s S U_s'
    sigma = small.singular_values[:k]
    v_rows = tuple(tuple(small.u[i][j] for j in range(k)) for i in range(n))
    u_rows = tuple(
        tuple(
            math.fsum(q_columns[c][i] * small.v[c][j] for c in range(sample))
            for j in range(k)
        )
        for i in range(m)
    )
    right, left = [], []
    for j in range(k):
        vj = [v_rows[i][j] for i in range(n)]
        uj = [u_rows[i][j] for i in range(m)]
        av = counted.matvec(vj)
        atu = counted.rmatvec(uj)
        right.append(_norm([a - sigma[j] * b for a, b in zip(av, uj, strict=True)]))
        left.append(_norm([a - sigma[j] * b for a, b in zip(atu, vj, strict=True)]))
    return RandomizedSVD(
        u=u_rows,
        singular_values=tuple(sigma),
        v=v_rows,
        right_residuals=tuple(right),
        left_residuals=tuple(left),
        sample_size=sample,
        power_iterations=power_iterations,
        counts=counted.counts,
    )
