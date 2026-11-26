"""Matrix-free likelihood-informed subspaces and resolution probes.

Experimental (added in 0.28).

The dense routines of :mod:`probstat.inverse.function_space_extended`
(:func:`~probstat.inverse.function_space_extended.likelihood_informed_subspace`,
:func:`~probstat.inverse.function_space_extended.resolution_operator`) form
``r x r`` or ``n x n`` matrices and refuse beyond ``max_dimension``. This
module reaches the same quantities with operator actions only.

**Matrix-free likelihood-informed subspace.** With the prior factor
``C0 = P P'``, the noise covariance ``Gamma`` and Jacobians ``J_k`` (any
:class:`~probstat.linalg.LinearOperator`, or a pair of Jacobian-vector and
vector-Jacobian callbacks wrapped by :func:`jvp_vjp_operator`), the
prior-preconditioned Gauss-Newton Hessian
``H = sum_k w_k P' J_k' Gamma^-1 J_k P`` is applied as an ``r x r`` operator
at the cost of one ``P``, one ``P'``, one ``Gamma^-1`` and one ``J_k``/``J_k'``
pair per point [EQ-INV-LIS-014]. :func:`matrix_free_lis` computes its leading
eigenpairs by Lanczos or by a randomized range finder, enlarging the computed
set until an explicit rule is decided: the rank is the number of eigenvalues
above ``threshold`` (capped by ``max_rank``), and the rule is trusted only if
a computed eigenvalue lies at or below the threshold (the threshold is
bracketed), the retained eigenpairs have small residuals and no eigenvalue
interval ``[theta - ||r||, theta + ||r||]`` contains the threshold
[EQ-INV-LIS-015]. Every pair carries its true residual
``||H y - theta y||`` (one extra application); for symmetric ``H`` an
eigenvalue of ``H`` lies within that distance of ``theta``, which is the
reported eigenvalue error bound. The subspace error against another basis is
measured by principal angles computed from sines, :func:`subspace_angles`
[EQ-INV-LIS-016]. In exact arithmetic a single-start Lanczos run sees each
distinct eigenvalue once (a circulant operator has every nonzero eigenvalue
twice); in floating point the copies may or may not appear, so the multiplicity
it reports is not reliable and the default method is the randomized one,
whose block of start vectors resolves multiplicities by construction
[EQ-INV-LIS-015].

**Resolution probing.** The resolution operator
``R = C G' S^-1 G = P (I + H)^-1 P' G' Gamma^-1 G`` (``H = P'G'Gamma^-1 G P``)
needs, for each column or row, one solve with the whitened posterior
precision ``I + H`` by conjugate gradients, that is one posterior-covariance
application ``Sigma = P (I + H)^-1 P'`` [EQ-INV-LIS-017]. Columns ``R e_j``
and rows ``R' e_i`` are probed this way, and the averaging kernel of a
functional ``l`` (a point value or a local average, from
:func:`~probstat.inverse.function_space_extended.point_functional` and
:func:`~probstat.inverse.function_space_extended.local_average_functional`)
is ``a = M^-1 R' l`` with the mass matrix ``M`` applied through conjugate
gradients as well [EQ-INV-LIS-018]. The effective number of resolved degrees
of freedom is ``tr R = tr((I + H)^-1 H) = sum_i lambda_i / (1 + lambda_i)``,
estimated by Hutchinson probes of the symmetric whitened operator (or of
``R`` itself) with a standard error [EQ-INV-LIS-019]. Every conjugate-gradient
solve is recorded as a :class:`SolveRecord`; a solve that did not reach its
tolerance is reported in ``all_converged`` and ``nonconverged_count``, never
hidden, and its vector is still returned so the caller can see the effect.

Limitations: the eigenpairs are those of the computed Krylov or random
subspace, not certified to be the leading ones (an eigenvalue absent from the
start vectors is not found); a rank decided with ``converged = False`` must
not be used silently; the nonlinear LIS is again an average of Jacobians at
the supplied points; the kernel centroid and width use user-supplied
coefficient positions and are only meaningful for nodal bases; the Hutchinson
estimate has the Monte-Carlo error that it reports.

References:
    [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014],
    [REF-SPANTINI-ETAL-2015], [REF-HALKO-MARTINSSON-TROPP-2011],
    [REF-LANCZOS-1950], [REF-HUTCHINSON-1989], [REF-HESTENES-STIEFEL-1952],
    [REF-TARANTOLA-2005], [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.function_space import _positive_int, _positive_real
from probstat.inverse.function_space_extended import (
    CovarianceFactor,
    _dot,
    _require_factor,
    _require_noise,
    _weights,
)
from probstat.linalg import (
    CallbackOperator,
    CountingOperator,
    LinearOperator,
    Matrix,
    OperatorCounts,
    TraceEstimate,
    Vector,
    as_vector,
    conjugate_gradient,
    hutchinson_trace,
    lanczos_eigenpairs,
    require_operator,
    singular_value_decomposition,
    symmetric_eigen,
)

__all__ = [
    "MatrixFreeLIS",
    "MatrixFreeResolution",
    "KernelProbe",
    "ResolutionProbe",
    "SolveRecord",
    "jvp_vjp_operator",
    "matrix_free_lis",
    "probe_resolution",
    "subspace_angles",
]

LISMethod = Literal["lanczos", "randomized"]
TraceMethod = Literal["whitened", "coefficient"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    Args:
        vector: Vector.

    Returns:
        ``sqrt(v'v)``.

    References:
        [EQ-INV-LIS-015].

    """
    return math.sqrt(math.fsum(a * a for a in vector))


def jvp_vjp_operator(
    jvp: Callable[[Vector], Sequence[float]],
    vjp: Callable[[Vector], Sequence[float]],
    shape: tuple[int, int],
) -> LinearOperator:
    """Wrap a Jacobian-vector and a vector-Jacobian product as an operator.

    Args:
        jvp: ``v -> J v`` for ``v`` of length ``n``.
        vjp: ``u -> J' u`` for ``u`` of length ``m``.
        shape: ``(m, n)``.

    Returns:
        A :class:`~probstat.linalg.CallbackOperator`. The adjoint identity
        ``u' (J v) = (J' u)' v`` is not checked here; use
        :func:`probstat.linalg.adjoint_test`.

    Raises:
        TypeError: If a callback is not callable or ``shape`` is malformed.

    References:
        [EQ-INV-LIS-014].

    """
    return CallbackOperator(shape, jvp, vjp, name="jvp/vjp Jacobian")


def _total(counts: Sequence[OperatorCounts]) -> OperatorCounts:
    """Return the sum of operator counts.

    Args:
        counts: Counts to add.

    Returns:
        Total forward and transpose applications.

    References:
        [EQ-INV-LIS-014].

    """
    return OperatorCounts(
        sum(c.forward for c in counts), sum(c.transpose for c in counts)
    )


# ---------------------------------------------------------------------------
# 1. Matrix-free LIS
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatrixFreeLIS:
    """Leading eigenpairs of the prior-preconditioned Hessian, matrix-free.

    Attributes:
        eigenvalues: Computed eigenvalue approximations (Rayleigh quotients)
            in descending order; only the leading ones are computed.
        eigenvectors: Unit whitened directions ``y_i`` (rows of length ``r``),
            one per computed eigenvalue.
        residual_norms: True residuals ``||H y_i - theta_i y_i||``; an
            eigenvalue of ``H`` lies within this distance of ``theta_i``.
        rank: Number of retained directions (eigenvalues above the threshold,
            at most ``max_rank``).
        basis: The ``rank`` retained whitened directions.
        parameter_directions: ``P y_i`` for the retained directions.
        threshold: The eigenvalue threshold.
        method: ``"lanczos"`` or ``"randomized"``.
        converged: Whether the rank rule was decided (see ``stop_reason``).
        stop_reason: ``"threshold_bracketed"``, ``"rank_cap"``,
            ``"invariant_subspace"``, ``"full_dimension"`` (all with
            ``converged = True``) or ``"max_steps"`` (``False``).
        capped: ``True`` if ``max_rank`` removed directions above the
            threshold.
        bracketed: ``True`` if a computed eigenvalue is at or below the
            threshold.
        ambiguous_count: Number of computed pairs whose interval
            ``[theta - residual, theta + residual]`` contains the threshold.
        max_retained_residual: Largest residual among the retained pairs.
        gap: ``theta_rank - theta_(rank+1)`` (``theta_rank`` if nothing
            follows, ``0`` for rank zero).
        subspace_size: Number of eigenpairs computed in the last round.
        rounds: Number of enlargement rounds used.
        hessian_applications: Total applications of ``H`` (including the
            residual checks).
        jacobian_counts: Forward and transpose applications summed over all
            Jacobians.
        factor_counts: Forward and transpose applications of ``P``.
        seconds: Wall-clock time of the computation.
        point_count: Number of Jacobians averaged.
        dimension: ``r``, the whitened dimension.
        prior_factor: ``P``.

    References:
        [EQ-INV-LIS-014], [EQ-INV-LIS-015].

    """

    eigenvalues: Vector
    eigenvectors: Matrix
    residual_norms: Vector
    rank: int
    basis: Matrix
    parameter_directions: Matrix
    threshold: float
    method: str
    converged: bool
    stop_reason: str
    capped: bool
    bracketed: bool
    ambiguous_count: int
    max_retained_residual: float
    gap: float
    subspace_size: int
    rounds: int
    hessian_applications: int
    jacobian_counts: OperatorCounts
    factor_counts: OperatorCounts
    seconds: float
    point_count: int
    dimension: int
    prior_factor: LinearOperator


def _hessian_operator(
    jacobians: Sequence[LinearOperator],
    weights: Sequence[float],
    noise: CovarianceFactor,
    factor: LinearOperator,
    applications: list[int],
) -> LinearOperator:
    """Return ``H = P' (sum_k w_k J_k' Gamma^-1 J_k) P`` as an operator.

    Args:
        jacobians: The (counting) Jacobian operators.
        weights: Normalized point weights.
        noise: Noise covariance factor.
        factor: The (counting) prior factor ``P``.
        applications: One-element list incremented per application.

    Returns:
        Symmetric positive semi-definite ``r x r`` operator.

    References:
        [EQ-INV-LIS-014].

    """
    r = factor.shape[1]
    n = factor.shape[0]

    def apply(vector: Vector) -> Vector:
        """Apply ``H``.

        References:
            [EQ-INV-LIS-014].

        """
        applications[0] += 1
        point = factor.matvec(vector)
        total = [0.0] * n
        for weight, jacobian in zip(weights, jacobians, strict=True):
            if weight == 0.0:
                continue
            back = jacobian.rmatvec(noise.solve(jacobian.matvec(point)))
            for i in range(n):
                total[i] += weight * back[i]
        return factor.rmatvec(tuple(total))

    return CallbackOperator((r, r), apply, apply, name="prior-preconditioned Hessian")


def _orthonormalize(columns: list[list[float]]) -> tuple[list[list[float]], bool]:
    """Return an orthonormal basis of the span of ``columns`` (twice-MGS).

    Args:
        columns: Vectors to orthonormalize.

    Returns:
        ``(basis, deficient)``; columns whose remainder is below
        ``1e-12`` of the largest input norm are dropped and ``deficient``
        is then ``True`` (the span has smaller dimension than requested).

    References:
        [EQ-INV-LIS-015], [REF-HALKO-MARTINSSON-TROPP-2011].

    """
    scale = max((_norm(c) for c in columns), default=0.0)
    basis: list[list[float]] = []
    deficient = False
    for column in columns:
        v = list(column)
        for _ in range(2):
            for q in basis:
                c = _dot(v, q)
                v = [a - c * b for a, b in zip(v, q, strict=True)]
        length = _norm(v)
        if length <= 1e-12 * scale:
            deficient = True
            continue
        basis.append([a / length for a in v])
    return basis, deficient


def _randomized_ritz(
    operator: LinearOperator,
    size: int,
    oversampling: int,
    power_iterations: int,
    stream: RandomStream,
) -> tuple[list[list[float]], bool]:
    """Return the top ``size`` Ritz vectors of a randomized range finder.

    A Gaussian block of ``size + oversampling`` columns is multiplied by the
    symmetric ``H``, ``power_iterations`` times re-orthonormalized, and the
    Rayleigh-Ritz problem ``Q'HQ`` is solved densely (Halko, Martinsson and
    Tropp, Algorithm 5.3); this uses only products with ``H``.

    Args:
        operator: Symmetric positive semi-definite operator.
        size: Number of Ritz vectors returned.
        oversampling: Extra block columns.
        power_iterations: Power passes.
        stream: Random stream for the Gaussian block.

    Returns:
        ``(vectors, deficient)`` with ``deficient`` true if the range of
        the sample had smaller dimension than the block (an invariant
        subspace of ``H`` was found).

    References:
        [EQ-INV-LIS-015], [REF-HALKO-MARTINSSON-TROPP-2011].

    """
    r = operator.shape[0]
    block = min(size + oversampling, r)
    omega = []
    for j in range(block):
        generator = stream.child("matrix_free_lis", j).generator()
        omega.append([generator.gauss(0.0, 1.0) for _ in range(r)])
    q, deficient = _orthonormalize([list(operator.matvec(tuple(c))) for c in omega])
    for _ in range(power_iterations):
        q, again = _orthonormalize([list(operator.matvec(tuple(c))) for c in q])
        deficient = deficient or again
    if not q:
        raise ValueError("the Hessian applied to the random block is zero")
    products = [operator.matvec(tuple(c)) for c in q]
    k = len(q)
    small = tuple(
        tuple(
            0.5 * (_dot(q[i], products[j]) + _dot(q[j], products[i])) for j in range(k)
        )
        for i in range(k)
    )
    values, vectors = symmetric_eigen(small)
    order = sorted(range(k), key=lambda i: values[i], reverse=True)[:size]
    ritz = []
    for i in order:
        y = [0.0] * r
        for c in range(k):
            coefficient = vectors[c][i]
            y = [a + coefficient * b for a, b in zip(y, q[c], strict=True)]
        ritz.append(y)
    return ritz, deficient


def _rayleigh_pairs(
    operator: LinearOperator, vectors: Sequence[Sequence[float]]
) -> tuple[list[float], list[list[float]], list[float]]:
    """Return sorted Rayleigh quotients, unit vectors and true residuals.

    Args:
        operator: Symmetric operator ``H``.
        vectors: Candidate unit vectors.

    Returns:
        ``(theta, vectors, residuals)`` sorted by descending ``theta``, with
        ``theta_i = y_i' H y_i`` and residual ``||H y_i - theta_i y_i||``.

    References:
        [EQ-INV-LIS-015].

    """
    entries = []
    for vector in vectors:
        length = _norm(vector)
        y = [v / length for v in vector]
        hy = operator.matvec(tuple(y))
        theta = _dot(y, hy)
        residual = _norm([a - theta * b for a, b in zip(hy, y, strict=True)])
        entries.append((theta, y, residual))
    entries.sort(key=lambda item: item[0], reverse=True)
    return (
        [e[0] for e in entries],
        [e[1] for e in entries],
        [e[2] for e in entries],
    )


def matrix_free_lis(
    jacobians: Sequence[object],
    *,
    noise: object,
    prior_factor: object,
    threshold: float,
    method: LISMethod = "randomized",
    max_rank: int | None = None,
    weights: Sequence[float] | None = None,
    initial_size: int | None = None,
    max_size: int | None = None,
    residual_tolerance: float = 1e-8,
    oversampling: int = 10,
    power_iterations: int = 2,
    seed: int | RandomStream | None = 0,
) -> MatrixFreeLIS:
    """Return the LIS of the averaged Gauss-Newton Hessian without dense matrices.

    The eigenpairs of ``H = sum_k w_k K_k' K_k`` (``K_k = W J_k P``) are
    computed on a subspace of size ``s`` (Lanczos steps or randomized-SVD
    rank), which doubles until the rule is decided or ``max_size`` is reached
    [EQ-INV-LIS-015]. The rule is decided when (i) the threshold is
    bracketed (a computed eigenvalue is ``<= threshold``), or the Krylov space
    became invariant, or ``s = r``, or the cap ``max_rank`` is met by
    eigenvalues above the threshold; (ii) every retained pair has residual at
    most ``residual_tolerance * theta_1``; and (iii) no pair's residual
    interval contains the threshold.

    Args:
        jacobians: Non-empty sequence of ``m x n`` operators or dense matrices
            (see :func:`jvp_vjp_operator` for callback pairs).
        noise: Noise covariance factor of dimension ``m``.
        prior_factor: ``P`` (``n x r``) with ``C0 = P P'`` (operator, matrix
            or covariance factor).
        threshold: Positive finite eigenvalue threshold.
        method: ``"randomized"`` (default; sees multiplicities) or
            ``"lanczos"`` (single start vector; eigenvalue multiplicity is
            not reliably reported).
        max_rank: Optional cap on the rank.
        weights: Optional nonnegative point weights (normalized).
        initial_size: First subspace size (default ``max_rank + 5`` or 16,
            capped at ``r``).
        max_size: Largest subspace size (default ``min(r, 256)``).
        residual_tolerance: Relative residual tolerance for retained pairs.
        oversampling: Extra random samples for the randomized method.
        power_iterations: Power passes for the randomized method.
        seed: Seed or stream (same start vectors in every round).

    Returns:
        :class:`MatrixFreeLIS`; ``converged = False`` means the rank is not
        decided and must not be used as is.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree or a numeric argument is invalid.

    References:
        [EQ-INV-LIS-014], [EQ-INV-LIS-015],
        [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014],
        [REF-HALKO-MARTINSSON-TROPP-2011], [REF-LANCZOS-1950].

    """
    started = time.perf_counter()
    operators = [require_operator(j, name="jacobian") for j in jacobians]
    if not operators:
        raise ValueError("jacobians must be non-empty")
    shape = operators[0].shape
    if any(op.shape != shape for op in operators):
        raise ValueError("all jacobians must have the same shape")
    m, n = shape
    checked_noise = _require_noise(noise, m)
    factor = _require_factor(prior_factor, n)
    r = factor.shape[1]
    level = _positive_real(threshold, "threshold")
    if method not in ("lanczos", "randomized"):
        raise ValueError("method must be 'lanczos' or 'randomized'")
    cap = None if max_rank is None else _positive_int(max_rank, "max_rank")
    tol = _positive_real(residual_tolerance, "residual_tolerance")
    start = (
        min(r, 16 if cap is None else cap + 5)
        if initial_size is None
        else min(r, _positive_int(initial_size, "initial_size"))
    )
    limit = (
        min(r, 256) if max_size is None else min(r, _positive_int(max_size, "max_size"))
    )
    start = min(start, limit)
    w = _weights(weights, len(operators))
    counted = [CountingOperator(op) for op in operators]
    counted_factor = CountingOperator(factor)
    applications = [0]
    hessian = _hessian_operator(counted, w, checked_noise, counted_factor, applications)

    stream = resolve_stream(seed)
    size = start
    rounds = 0
    while True:
        rounds += 1
        breakdown = False
        if method == "lanczos":
            lanczos = lanczos_eigenpairs(hessian, size, seed=seed)
            raw = [list(v) for v in lanczos.eigenvectors]
            breakdown = lanczos.breakdown
        else:
            raw, breakdown = _randomized_ritz(
                hessian, size, oversampling, power_iterations, stream
            )
        theta, vectors, residuals = _rayleigh_pairs(hessian, raw)
        top = theta[0]
        above = sum(1 for t in theta if t > level)
        rank = above if cap is None else min(above, cap)
        bracketed = above < len(theta)
        cap_met = cap is not None and above >= cap and len(theta) > cap
        full = len(theta) >= r
        retained_residual = max(residuals[:rank], default=0.0)
        residual_ok = top <= 0.0 or retained_residual <= tol * top
        ambiguous = sum(
            1 for t, res in zip(theta, residuals, strict=True) if abs(t - level) <= res
        )
        if breakdown:
            reason = "invariant_subspace"
        elif full:
            reason = "full_dimension"
        elif cap_met:
            reason = "rank_cap"
        elif bracketed:
            reason = "threshold_bracketed"
        else:
            reason = ""
        decided = reason != "" and residual_ok and ambiguous == 0
        if decided:
            break
        if size >= limit:
            reason = "max_steps"
            break
        size = min(2 * size, limit)
    converged = reason != "max_steps"
    if rank == 0:
        gap = 0.0
    elif rank == len(theta):
        gap = theta[rank - 1]
    else:
        gap = theta[rank - 1] - theta[rank]
    kept = vectors[:rank]
    return MatrixFreeLIS(
        eigenvalues=tuple(theta),
        eigenvectors=tuple(tuple(v) for v in vectors),
        residual_norms=tuple(residuals),
        rank=rank,
        basis=tuple(tuple(v) for v in kept),
        parameter_directions=tuple(counted_factor.matvec(tuple(v)) for v in kept),
        threshold=level,
        method=method,
        converged=converged,
        stop_reason=reason,
        capped=cap is not None and above > cap,
        bracketed=bracketed,
        ambiguous_count=ambiguous,
        max_retained_residual=retained_residual,
        gap=gap,
        subspace_size=len(theta),
        rounds=rounds,
        hessian_applications=applications[0],
        jacobian_counts=_total([c.counts for c in counted]),
        factor_counts=counted_factor.counts,
        seconds=time.perf_counter() - started,
        point_count=len(operators),
        dimension=r,
        prior_factor=factor,
    )


def subspace_angles(
    basis_a: Sequence[Sequence[float]], basis_b: Sequence[Sequence[float]]
) -> Vector:
    """Return the principal angles between two orthonormal row bases.

    The sines of the angles of ``span(B)`` against ``span(A)`` are the
    singular values of ``E = B - (B A') A`` (the component of ``B`` outside
    ``span(A)``), which resolves small angles to full relative accuracy
    where ``acos`` of a cosine would lose half the digits
    [EQ-INV-LIS-016]. The cosines are the singular values of ``B A'``, and
    ``atan2(sine, cosine)`` is accurate near both ``0`` and ``pi/2``. With
    equal sizes these are the principal angles.

    Args:
        basis_a: ``ka`` orthonormal rows of length ``r``.
        basis_b: ``kb`` orthonormal rows of length ``r``.

    Returns:
        ``kb`` angles in radians, ascending (``pi/2`` appears for directions
        of ``B`` orthogonal to ``span(A)``, e.g. when ``kb > ka``).

    Raises:
        ValueError: If a basis is empty, the lengths differ, or a basis is
            not orthonormal to ``1e-6``.

    References:
        [EQ-INV-LIS-016], [REF-GOLUB-VANLOAN-2013].

    """
    a_rows = [as_vector(row, name="basis_a row") for row in basis_a]
    b_rows = [as_vector(row, name="basis_b row") for row in basis_b]
    if not a_rows or not b_rows:
        raise ValueError("bases must be non-empty")
    r = len(a_rows[0])
    if any(len(v) != r for v in a_rows + b_rows):
        raise ValueError("all basis vectors must have the same length")
    for name, rows in (("basis_a", a_rows), ("basis_b", b_rows)):
        for i, ri in enumerate(rows):
            for j, rj in enumerate(rows):
                target = 1.0 if i == j else 0.0
                if abs(_dot(ri, rj) - target) > 1e-6:
                    raise ValueError(f"{name} is not orthonormal")
    residual_columns = []
    for b in b_rows:
        e = list(b)
        for a in a_rows:
            c = _dot(a, b)
            e = [x - c * y for x, y in zip(e, a, strict=True)]
        residual_columns.append(e)
    tall = tuple(
        tuple(residual_columns[j][i] for j in range(len(b_rows))) for i in range(r)
    )
    sines = sorted(singular_value_decomposition(tall).singular_values)
    overlap = tuple(tuple(_dot(a_row, b_row) for a_row in a_rows) for b_row in b_rows)
    cosines = sorted(singular_value_decomposition(overlap).singular_values)
    cosines = [0.0] * (len(b_rows) - len(cosines)) + cosines
    cosines.reverse()
    return tuple(sorted(math.atan2(s, c) for s, c in zip(sines, cosines, strict=True)))


# ---------------------------------------------------------------------------
# 2. Resolution probing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveRecord:
    """Outcome of one conjugate-gradient solve.

    Attributes:
        iterations: Iterations performed.
        stop_reason: Stop reason of the Krylov routine.
        converged: Whether the tolerance was met.
        relative_residual: True ``||b - A x|| / ||b||`` (``0`` for a zero
            right-hand side).

    References:
        [EQ-INV-LIS-017].

    """

    iterations: int
    stop_reason: str
    converged: bool
    relative_residual: float


class MatrixFreeResolution:
    """Action of ``R = P (I + H)^-1 P' G' Gamma^-1 G`` and of its transpose.

    ``R`` is the derivative of the posterior mean with respect to the true
    coefficients; each application costs one conjugate-gradient solve with
    the whitened posterior precision ``I + H`` [EQ-INV-LIS-017]. Records of
    all solves accumulate in :attr:`records`.

    References:
        [EQ-INV-LIS-017], [REF-TARANTOLA-2005].

    """

    def __init__(
        self,
        forward: object,
        *,
        noise: object,
        prior_factor: object,
        cg_tolerance: float = 1e-10,
        cg_max_iterations: int | None = None,
    ) -> None:
        """Store the problem and the solver settings.

        Args:
            forward: ``G`` (``m x n``), coefficients to data.
            noise: Noise covariance factor of dimension ``m``.
            prior_factor: ``P`` (``n x r``) with ``C = P P'``.
            cg_tolerance: Relative residual tolerance of the solves.
            cg_max_iterations: Iteration cap per solve (default ``2 r``).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If shapes or numeric settings are invalid.

        References:
            [EQ-INV-LIS-017].

        """
        self._g = CountingOperator(require_operator(forward, name="forward"))
        m, n = self._g.shape
        self._noise = _require_noise(noise, m)
        self._p = CountingOperator(_require_factor(prior_factor, n))
        self._tolerance = _positive_real(cg_tolerance, "cg_tolerance")
        self._max_iterations = (
            None
            if cg_max_iterations is None
            else _positive_int(cg_max_iterations, "cg_max_iterations")
        )
        self.records: list[SolveRecord] = []
        self.precision = self._precision_operator()

    @property
    def dimension(self) -> int:
        """Return ``n``, the coefficient dimension.

        References:
            [EQ-INV-LIS-017].

        """
        return self._g.shape[1]

    @property
    def whitened_dimension(self) -> int:
        """Return ``r``, the number of columns of the prior factor.

        References:
            [EQ-INV-LIS-017].

        """
        return self._p.shape[1]

    @property
    def forward_counts(self) -> OperatorCounts:
        """Return the applications of ``G`` so far.

        References:
            [EQ-INV-LIS-017].

        """
        return self._g.counts

    @property
    def factor_counts(self) -> OperatorCounts:
        """Return the applications of ``P`` so far.

        References:
            [EQ-INV-LIS-017].

        """
        return self._p.counts

    def _gram(self, point: Vector) -> Vector:
        """Return ``G' Gamma^-1 G point``.

        Args:
            point: Vector of length ``n``.

        Returns:
            The vector.

        References:
            [EQ-INV-LIS-014].

        """
        return self._g.rmatvec(self._noise.solve(self._g.matvec(point)))

    def _precision_operator(self) -> LinearOperator:
        """Return ``I + H`` on the whitened space.

        Returns:
            The SPD operator.

        References:
            [EQ-INV-LIS-017].

        """
        r = self._p.shape[1]

        def apply(vector: Vector) -> Vector:
            """Apply ``I + P' G' Gamma^-1 G P``.

            References:
                [EQ-INV-LIS-017].

            """
            back = self._p.rmatvec(self._gram(self._p.matvec(vector)))
            return tuple(a + b for a, b in zip(vector, back, strict=True))

        return CallbackOperator((r, r), apply, apply, name="whitened precision")

    def solve_precision(self, rhs: Sequence[float]) -> Vector:
        """Solve ``(I + H) z = rhs`` by CG and record the outcome.

        Args:
            rhs: Vector of length ``r``.

        Returns:
            The solution ``z`` (also when the solve did not converge; see
            :attr:`records`).

        References:
            [EQ-INV-LIS-017], [REF-HESTENES-STIEFEL-1952].

        """
        b = tuple(rhs)
        result = conjugate_gradient(
            self.precision,
            b,
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
        )
        bnorm = _norm(b)
        relative = result.true_residual_norm / bnorm if bnorm > 0.0 else 0.0
        self.records.append(
            SolveRecord(
                iterations=result.iterations,
                stop_reason=result.stop_reason,
                converged=result.converged,
                relative_residual=relative,
            )
        )
        return result.solution

    def apply(self, vector: Sequence[float]) -> Vector:
        """Return ``R x``: how the estimate responds to a true coefficient ``x``.

        Args:
            vector: ``x`` of length ``n``.

        Returns:
            ``P (I + H)^-1 P' G' Gamma^-1 G x``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LIS-017].

        """
        x = as_vector(vector, name="vector")
        if len(x) != self.dimension:
            raise ValueError(f"vector has length {len(x)}, expected {self.dimension}")
        z = self.solve_precision(self._p.rmatvec(self._gram(x)))
        return self._p.matvec(z)

    def apply_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``R' l = G' Gamma^-1 G P (I + H)^-1 P' l``.

        Args:
            vector: ``l`` of length ``n`` (a functional of the coefficients).

        Returns:
            ``R' l``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LIS-017].

        """
        ell = as_vector(vector, name="vector")
        if len(ell) != self.dimension:
            raise ValueError(f"vector has length {len(ell)}, expected {self.dimension}")
        z = self.solve_precision(self._p.rmatvec(ell))
        return self._gram(self._p.matvec(z))

    def as_operator(self) -> LinearOperator:
        """Return ``R`` as an ``n x n`` operator (each action runs a CG solve).

        Returns:
            Operator with ``matvec = R x`` and ``rmatvec = R' l``.

        References:
            [EQ-INV-LIS-017].

        """
        n = self.dimension
        return CallbackOperator(
            (n, n), self.apply, self.apply_transpose, name="resolution operator"
        )

    def whitened_trace_operator(self) -> LinearOperator:
        """Return the symmetric operator ``(I + H)^-1 H`` on the whitened space.

        Its trace equals ``tr R`` [EQ-INV-LIS-019].

        Returns:
            ``r x r`` operator.

        References:
            [EQ-INV-LIS-019].

        """
        r = self.whitened_dimension

        def apply(vector: Vector) -> Vector:
            """Apply ``(I + H)^-1 H``.

            References:
                [EQ-INV-LIS-019].

            """
            h_vector = self._p.rmatvec(self._gram(self._p.matvec(vector)))
            return self.solve_precision(h_vector)

        return CallbackOperator((r, r), apply, apply, name="(I+H)^-1 H")


@dataclass(frozen=True, slots=True)
class KernelProbe:
    """Averaging kernel of one functional from operator actions.

    Attributes:
        coefficients: ``a = M^-1 R' l``.
        row: ``R' l`` (before the mass solve).
        integral: ``m' a`` if an integral vector was given, else ``nan``.
        target_integral: ``l' c`` with ``c = M^-1 m`` if an integral vector
            was given, else ``nan``.
        centroid: ``sum_j x_j |a_j| / sum_j |a_j|`` over the supplied
            coefficient positions, else ``nan``.
        rms_width: ``sqrt(sum_j (x_j - centroid)^2 |a_j| / sum_j |a_j|)``,
            else ``nan``.
        records: Solve records of this kernel (resolution solve first, then
            the mass solve if a mass operator was given).

    References:
        [EQ-INV-LIS-018].

    """

    coefficients: Vector
    row: Vector
    integral: float
    target_integral: float
    centroid: float
    rms_width: float
    records: tuple[SolveRecord, ...]


@dataclass(frozen=True, slots=True)
class ResolutionProbe:
    """Columns, rows, averaging kernels and trace of ``R`` from operator actions.

    Attributes:
        column_indices: Probed column indices.
        columns: ``R e_j`` for each index.
        column_records: Solve record of each column.
        row_indices: Probed row indices.
        rows: ``R' e_i`` (row ``i`` of ``R``) for each index.
        row_records: Solve record of each row.
        kernels: One :class:`KernelProbe` per functional.
        trace: Hutchinson estimate of ``tr R`` (``None`` if no probes were
            requested); its ``standard_error`` is the Monte-Carlo error.
        trace_method: ``"whitened"`` or ``"coefficient"``.
        solve_count: Total number of CG solves (including mass solves of
            kernels and the trace probes).
        total_cg_iterations: Total CG iterations over all recorded solves.
        max_relative_residual: Largest true relative residual over all
            solves of the resolution precision and the mass operator.
        nonconverged_count: Number of solves that did not converge.
        all_converged: ``nonconverged_count == 0``.
        forward_counts: Applications of ``G``.
        factor_counts: Applications of ``P``.
        seconds: Wall-clock time.

    References:
        [EQ-INV-LIS-017], [EQ-INV-LIS-018], [EQ-INV-LIS-019].

    """

    column_indices: tuple[int, ...]
    columns: Matrix
    column_records: tuple[SolveRecord, ...]
    row_indices: tuple[int, ...]
    rows: Matrix
    row_records: tuple[SolveRecord, ...]
    kernels: tuple[KernelProbe, ...]
    trace: TraceEstimate | None
    trace_method: str
    solve_count: int
    total_cg_iterations: int
    max_relative_residual: float
    nonconverged_count: int
    all_converged: bool
    forward_counts: OperatorCounts
    factor_counts: OperatorCounts
    seconds: float


def _indices(values: Sequence[int], size: int, name: str) -> tuple[int, ...]:
    """Validate a sequence of coefficient indices.

    Args:
        values: Candidate indices.
        size: Dimension ``n``.
        name: Name for messages.

    Returns:
        The indices as a tuple.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If an index is out of range.

    References:
        [EQ-INV-LIS-017].

    """
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integers")
        if value < 0 or value >= size:
            raise ValueError(f"{name} entry {value} is outside [0, {size})")
        out.append(value)
    return tuple(out)


def probe_resolution(
    forward: object,
    *,
    noise: object,
    prior_factor: object,
    column_indices: Sequence[int] = (),
    row_indices: Sequence[int] = (),
    functionals: Sequence[Sequence[float]] = (),
    mass: object | None = None,
    integral_vector: Sequence[float] | None = None,
    positions: Sequence[float] | None = None,
    trace_probes: int = 30,
    trace_method: TraceMethod = "whitened",
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
    seed: int | RandomStream | None = 0,
) -> ResolutionProbe:
    """Probe the resolution operator through operator actions only.

    Columns ``R e_j``, rows ``R' e_i``, mass-weighted averaging kernels
    ``a = M^-1 R' l`` and a Hutchinson estimate of ``tr R`` are computed
    with one conjugate-gradient solve per column, row, kernel and trace
    probe; no ``n x n`` matrix is formed [EQ-INV-LIS-017],
    [EQ-INV-LIS-018], [EQ-INV-LIS-019]. A solve that fails to reach
    ``cg_tolerance`` is recorded and counted in ``nonconverged_count``.

    Args:
        forward: ``G`` (``m x n``).
        noise: Noise covariance factor of dimension ``m``.
        prior_factor: ``P`` (``n x r``) with ``C = P P'``.
        column_indices: Columns of ``R`` to extract.
        row_indices: Rows of ``R`` to extract.
        functionals: Coefficient vectors ``l`` (point values from
            :func:`~probstat.inverse.function_space_extended.point_functional`,
            local averages from
            :func:`~probstat.inverse.function_space_extended.local_average_functional`).
        mass: SPD mass matrix ``M`` of the basis as an operator or matrix
            (default the identity, an orthonormal coefficient basis).
        integral_vector: ``m_j = int phi_j`` (enables ``integral`` and
            ``target_integral`` of the kernels).
        positions: Coefficient locations ``x_j`` for the kernel centroid and
            width (nodal bases only).
        trace_probes: Number of Hutchinson probes (``0`` skips the trace).
        trace_method: ``"whitened"`` probes the symmetric ``r x r`` operator
            ``(I+H)^-1 H``; ``"coefficient"`` probes ``R`` itself.
        cg_tolerance: Relative CG tolerance.
        cg_max_iterations: CG iteration cap (default ``2 r``).
        seed: Seed or stream for the probes.

    Returns:
        :class:`ResolutionProbe`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes or indices are invalid.

    References:
        [EQ-INV-LIS-017], [EQ-INV-LIS-018], [EQ-INV-LIS-019],
        [REF-HUTCHINSON-1989], [REF-TARANTOLA-2005].

    """
    started = time.perf_counter()
    resolution = MatrixFreeResolution(
        forward,
        noise=noise,
        prior_factor=prior_factor,
        cg_tolerance=cg_tolerance,
        cg_max_iterations=cg_max_iterations,
    )
    n = resolution.dimension
    cols = _indices(column_indices, n, "column_indices")
    rws = _indices(row_indices, n, "row_indices")
    probes = trace_probes
    if isinstance(probes, bool) or not isinstance(probes, int) or probes < 0:
        raise ValueError("trace_probes must be a non-negative integer")
    if trace_method not in ("whitened", "coefficient"):
        raise ValueError("trace_method must be 'whitened' or 'coefficient'")
    mass_operator = None if mass is None else require_operator(mass, name="mass")
    if mass_operator is not None and mass_operator.shape != (n, n):
        raise ValueError(f"mass has shape {mass_operator.shape}, expected {(n, n)}")
    integral = (
        None if integral_vector is None else as_vector(integral_vector, name="integral")
    )
    if integral is not None and len(integral) != n:
        raise ValueError(f"integral_vector has length {len(integral)}, expected {n}")
    where = None if positions is None else as_vector(positions, name="positions")
    if where is not None and len(where) != n:
        raise ValueError(f"positions has length {len(where)}, expected {n}")
    ells = [as_vector(f, name="functional") for f in functionals]
    if any(len(ell) != n for ell in ells):
        raise ValueError(f"every functional must have length {n}")

    unit = [0.0] * n

    def unit_vector(index: int) -> Vector:
        """Return the ``index``-th unit vector.

        References:
            [EQ-INV-LIS-017].

        """
        v = list(unit)
        v[index] = 1.0
        return tuple(v)

    columns = []
    column_records = []
    for j in cols:
        columns.append(resolution.apply(unit_vector(j)))
        column_records.append(resolution.records[-1])
    rows = []
    row_records = []
    for i in rws:
        rows.append(resolution.apply_transpose(unit_vector(i)))
        row_records.append(resolution.records[-1])

    mass_records: list[SolveRecord] = []

    def mass_solve(rhs: Vector) -> tuple[Vector, SolveRecord]:
        """Return ``M^-1 rhs`` by CG with its record (identity if no mass).

        References:
            [EQ-INV-LIS-018].

        """
        if mass_operator is None:
            return rhs, SolveRecord(0, "identity", True, 0.0)
        result = conjugate_gradient(
            mass_operator,
            rhs,
            tolerance=cg_tolerance,
            max_iterations=cg_max_iterations,
        )
        bnorm = _norm(rhs)
        record = SolveRecord(
            result.iterations,
            result.stop_reason,
            result.converged,
            result.true_residual_norm / bnorm if bnorm > 0.0 else 0.0,
        )
        mass_records.append(record)
        return result.solution, record

    constant: Vector | None = None
    if integral is not None and ells:
        constant, _ = mass_solve(integral)
    kernels = []
    for ell in ells:
        row = resolution.apply_transpose(ell)
        recs = [resolution.records[-1]]
        coefficients, mass_record = mass_solve(row)
        if mass_operator is not None:
            recs.append(mass_record)
        total = math.nan if integral is None else _dot(integral, coefficients)
        target = math.nan if constant is None else _dot(ell, constant)
        centroid = math.nan
        width = math.nan
        if where is not None:
            mass_abs = math.fsum(abs(a) for a in coefficients)
            if mass_abs > 0.0:
                centroid = (
                    math.fsum(
                        x * abs(a) for x, a in zip(where, coefficients, strict=True)
                    )
                    / mass_abs
                )
                width = math.sqrt(
                    math.fsum(
                        (x - centroid) ** 2 * abs(a)
                        for x, a in zip(where, coefficients, strict=True)
                    )
                    / mass_abs
                )
        kernels.append(
            KernelProbe(
                coefficients=coefficients,
                row=row,
                integral=total,
                target_integral=target,
                centroid=centroid,
                rms_width=width,
                records=tuple(recs),
            )
        )

    trace: TraceEstimate | None = None
    if probes > 0:
        if trace_method == "whitened":
            trace_operator = resolution.whitened_trace_operator()
        else:
            trace_operator = resolution.as_operator()
        trace = hutchinson_trace(trace_operator, probes=probes, seed=seed)

    solves = resolution.records + mass_records
    nonconverged = sum(1 for s in solves if not s.converged)
    return ResolutionProbe(
        column_indices=cols,
        columns=tuple(columns),
        column_records=tuple(column_records),
        row_indices=rws,
        rows=tuple(rows),
        row_records=tuple(row_records),
        kernels=tuple(kernels),
        trace=trace,
        trace_method=trace_method,
        solve_count=len(solves),
        total_cg_iterations=sum(s.iterations for s in solves),
        max_relative_residual=max((s.relative_residual for s in solves), default=0.0),
        nonconverged_count=nonconverged,
        all_converged=nonconverged == 0,
        forward_counts=resolution.forward_counts,
        factor_counts=resolution.factor_counts,
        seconds=time.perf_counter() - started,
    )
