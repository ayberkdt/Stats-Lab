"""Spectral diagnostics: what an observation operator can and cannot resolve.

:func:`spectral_diagnostics` takes the direct SVD of the WHITENED operator
``B = W A`` (never ``B'B``, which squares the conditioning) and reports the
singular values, the numerical rank for an explicit cutoff, the condition
numbers, an orthonormal basis of the resolved (row-space) subspace and of the
FULL nullspace (completed, since a thin SVD of a wide matrix omits ``n - m``
null directions), the noise amplification ``1 / sigma_i``, and the discrete
Picard coefficients ``|u_i'W y|`` together with the part of the data outside
the range of ``B`` [EQ-INV-SPEC-001]. A numerical cutoff is not a scientific
detection threshold: a direction with ``sigma_i`` far above the cutoff may
still be invisible against the noise when ``sigma_i |c_i| << 1`` in whitened
units, which :attr:`SpectralDiagnostics.detectable` makes explicit for a
given coefficient size.

:func:`generalized_svd` computes the GSVD of a pair ``(A, L)`` with a
full-column-rank stack by the QR-plus-SVD construction (restricted scope, not
a general LAPACK GSVD) and :func:`gsvd_tikhonov` evaluates general-form
Tikhonov solutions from it for any ``lambda`` in ``O(n^2)`` work
[EQ-INV-SPEC-002], [EQ-INV-SPEC-003].

Experimental (added in 0.28).

References:
    [REF-HANSEN-2008, Sections 2.1-2.4], [REF-VAN-LOAN-1976],
    [REF-GOLUB-VANLOAN-2013, Sections 6.1 and 8.7].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    householder_qr,
    materialize,
    require_operator,
    singular_value_decomposition,
)
from probstat.linalg.operators import LinearOperator
from probstat.linalg.svd import _complete_orthonormal

from .linear_problem import CovarianceFactor, _finite_vector

_DEFAULT_LIMIT = 4_000_000


@dataclass(frozen=True, slots=True)
class SpectralDiagnostics:
    """Singular spectrum, rank, subspaces and Picard quantities of ``B = W A``.

    Attributes:
        singular_values: ``sigma_1 >= ... >= sigma_k``, ``k = min(m, n)``.
        cutoff: Absolute threshold ``max(atol, rtol sigma_1)``; ``sigma_i``
            above it count towards the rank.
        rank: Numerical rank ``r``.
        condition_number: ``sigma_1 / sigma_k``; ``inf`` when ``sigma_k = 0``
            or when ``n > m`` (some directions are always unobserved).
        effective_condition_number: ``sigma_1 / sigma_r`` on the resolved
            subspace.
        nullspace_dimension: ``n - r``.
        resolved_basis: ``r`` orthonormal vectors (rows) spanning the
            resolved subspace ``span(v_1..v_r)``.
        nullspace_basis: ``n - r`` orthonormal vectors (rows) spanning its
            orthogonal complement, including the completion of a thin SVD.
        noise_amplification: ``1 / sigma_i`` for ``i <= r`` (whitened units:
            a unit noise component along ``u_i`` moves ``x`` by ``1/sigma_i``
            along ``v_i``).
        data_coefficients: ``|u_i'W y|`` (``None`` without data).
        solution_coefficients: ``|u_i'W y| / sigma_i`` for ``sigma_i > 0``,
            ``inf`` for ``sigma_i = 0 < |u_i'W y|`` and ``nan`` for ``0/0``
            (``None`` without data).
        out_of_range_norm: ``||(I - U U')W y||``, the data component no
            ``x`` can fit (``None`` without data).
        left_vectors: ``U`` as rows of length ``k`` (``m`` rows).
        right_vectors: ``V`` as rows of length ``k`` (``n`` rows).

    References:
        [EQ-INV-SPEC-001], [REF-HANSEN-2008, Sections 2.1 and 2.4].

    """

    singular_values: Vector
    cutoff: float
    rank: int
    condition_number: float
    effective_condition_number: float
    nullspace_dimension: int
    resolved_basis: Matrix
    nullspace_basis: Matrix
    noise_amplification: Vector
    data_coefficients: Vector | None
    solution_coefficients: Vector | None
    out_of_range_norm: float | None
    left_vectors: Matrix
    right_vectors: Matrix

    def null_projection(self, vector: Sequence[float]) -> Vector:
        """Return ``P_null x = x - V_r V_r' x``.

        Args:
            vector: ``x`` of length ``n``.

        Returns:
            The component of ``x`` invisible to the data at this cutoff.

        References:
            [EQ-INV-SPEC-001].

        """
        n = len(self.right_vectors)
        x = _finite_vector(vector, n, "vector")
        result = list(x)
        for basis in self.resolved_basis:
            c = math.fsum(a * b for a, b in zip(basis, x, strict=True))
            result = [r - c * b for r, b in zip(result, basis, strict=True)]
        return tuple(result)

    def detectable(self, coefficient: float = 1.0) -> tuple[bool, ...]:
        """Return whether a coefficient of size ``c`` along ``v_i`` exceeds unit noise.

        A change ``c v_i`` of the parameters changes the whitened data by
        ``c sigma_i u_i``; with unit whitened noise per component it is
        detectable (heuristically, one standard deviation) iff
        ``|c| sigma_i >= 1``. This is a documented heuristic, not a test.

        Args:
            coefficient: Coefficient size ``c`` in parameter units.

        Returns:
            One flag per singular value.

        References:
            [EQ-INV-SPEC-001].

        """
        size = abs(float(coefficient))
        return tuple(size * s >= 1.0 for s in self.singular_values)


def spectral_diagnostics(
    operator: LinearOperator | Sequence[Sequence[float]],
    *,
    data: Sequence[float] | None = None,
    noise: CovarianceFactor | None = None,
    rtol: float | None = None,
    atol: float = 0.0,
    max_dense_entries: int = _DEFAULT_LIMIT,
) -> SpectralDiagnostics:
    """Return the spectral diagnostics of the whitened operator ``W A``.

    Args:
        operator: ``A`` (dense or operator; materialized, dense oracle).
        data: Optional ``y`` for the Picard coefficients.
        noise: Optional covariance factor (``W = I`` when omitted).
        rtol: Relative cutoff (default ``max(m, n) * 2^-52``).
        atol: Absolute cutoff (whitened units); the cutoff is
            ``max(atol, rtol sigma_1)``.
        max_dense_entries: Size guard; larger problems need the Krylov or
            randomized estimators of :mod:`probstat.linalg.krylov`.

    Returns:
        :class:`SpectralDiagnostics`.

    Raises:
        ValueError: If dimensions disagree or the operator is too large.

    References:
        [EQ-INV-SPEC-001], [REF-HANSEN-2008, Section 2.4],
        [REF-GOLUB-VANLOAN-2013, Section 8.6].

    """
    a = require_operator(operator)
    m, n = a.shape
    dense = materialize(a, max_entries=max_dense_entries)
    if noise is not None:
        if noise.dimension != m:
            raise ValueError("noise dimension must equal the number of observations")
        columns = [noise.whiten([dense[i][j] for i in range(m)]) for j in range(n)]
        dense = tuple(tuple(columns[j][i] for j in range(n)) for i in range(m))
    svd = singular_value_decomposition(dense)
    sigma = svd.singular_values
    k = len(sigma)
    relative = max(m, n) * 2.0**-52 if rtol is None else float(rtol)
    if (
        not math.isfinite(relative)
        or relative < 0.0
        or not math.isfinite(atol)
        or atol < 0
    ):
        raise ValueError("rtol and atol must be finite and non-negative")
    cutoff = max(float(atol), relative * (sigma[0] if sigma else 0.0))
    rank = sum(1 for s in sigma if s > cutoff)
    v_columns = [[svd.v[j][i] for j in range(n)] for i in range(k)]
    full = [list(col) for col in v_columns[:rank]] + [
        [0.0] * n for _ in range(n - rank)
    ]
    _complete_orthonormal(full, list(range(rank, n)), n)
    condition = math.inf if (n > m or sigma[-1] == 0.0) else sigma[0] / sigma[-1]
    effective = sigma[0] / sigma[rank - 1] if rank > 0 else math.inf
    data_coefficients = solution_coefficients = None
    out_of_range = None
    if data is not None:
        y = _finite_vector(data, m, "data")
        wy = y if noise is None else noise.whiten(y)
        beta = [math.fsum(svd.u[r][i] * wy[r] for r in range(m)) for i in range(k)]
        data_coefficients = tuple(abs(b) for b in beta)
        solution_coefficients = tuple(
            abs(b) / s if s > 0.0 else (math.inf if b != 0.0 else math.nan)
            for b, s in zip(beta, sigma, strict=True)
        )
        projected = [
            math.fsum(svd.u[r][i] * beta[i] for i in range(k)) for r in range(m)
        ]
        out_of_range = math.hypot(
            *(a_ - b_ for a_, b_ in zip(wy, projected, strict=True))
        )
    return SpectralDiagnostics(
        singular_values=sigma,
        cutoff=cutoff,
        rank=rank,
        condition_number=condition,
        effective_condition_number=effective,
        nullspace_dimension=n - rank,
        resolved_basis=tuple(tuple(col) for col in full[:rank]),
        nullspace_basis=tuple(tuple(col) for col in full[rank:]),
        noise_amplification=tuple(1.0 / s for s in sigma[:rank]),
        data_coefficients=data_coefficients,
        solution_coefficients=solution_coefficients,
        out_of_range_norm=out_of_range,
        left_vectors=svd.u,
        right_vectors=svd.v,
    )


@dataclass(frozen=True, slots=True)
class GeneralizedSVD:
    """GSVD ``A X = U C``, ``L X = Z S`` of a pair with full-rank stack ``[A; L]``.

    Attributes:
        c: Cosines ``c_i`` in ``[0, 1]``.
        s: Sines ``s_i`` in ``[0, 1]`` with ``c_i^2 + s_i^2 = 1`` (computed as
            column norms of ``Q_L V``, not as ``sqrt(1 - c^2)``).
        generalized_values: ``gamma_i = c_i / s_i`` (``inf`` where
            ``s_i = 0``), in the order of ``c``.
        x: ``X = R^-1 V`` (``n x n``) as rows.
        u: ``U`` (``m x n``) as rows; columns with ``c_i = 0`` are zero.
        z: ``Z`` (``p x n``) as rows; columns with ``s_i = 0`` are zero.

    References:
        [EQ-INV-SPEC-002], [REF-VAN-LOAN-1976], [REF-HANSEN-2008, Section 2.1].

    """

    c: Vector
    s: Vector
    generalized_values: Vector
    x: Matrix
    u: Matrix
    z: Matrix


def generalized_svd(
    matrix: Sequence[Sequence[float]],
    penalty: Sequence[Sequence[float]],
    *,
    rank_tolerance: float = 1e-12,
) -> GeneralizedSVD:
    """Return the GSVD of ``(A, L)`` by QR of the stack and SVD of its top block.

    With ``[A; L] = [Q_A; Q_L] R`` (thin QR, full column rank required) and
    the SVD ``Q_A = U C V'``, ``X = R^-1 V`` gives ``A X = U C`` and
    ``L X = Q_L V = Z S`` with ``Z'Z = I`` on the columns where ``s_i > 0``,
    because ``Q_A'Q_A + Q_L'Q_L = I`` [EQ-INV-SPEC-002]. Restricted scope:
    ``m >= n`` and a full-column-rank stack (``null(A) ∩ null(L) = {0}``);
    other cases raise rather than return a partial decomposition.

    Args:
        matrix: ``A`` (``m x n``, ``m >= n``).
        penalty: ``L`` (``p x n``).
        rank_tolerance: Relative rank threshold of the stacked QR.

    Returns:
        :class:`GeneralizedSVD`.

    Raises:
        ValueError: If ``m < n`` or the stack is rank deficient.

    References:
        [EQ-INV-SPEC-002], [REF-VAN-LOAN-1976],
        [REF-GOLUB-VANLOAN-2013, Section 8.7].

    """
    a = [list(_finite_vector(row, len(matrix[0]), "matrix row")) for row in matrix]
    n = len(a[0])
    l_rows = [list(_finite_vector(row, n, "penalty row")) for row in penalty]
    m, p = len(a), len(l_rows)
    if m < n:
        raise ValueError("generalized_svd needs m >= n (restricted scope)")
    factor = householder_qr(
        tuple(tuple(r) for r in a + l_rows), rank_tolerance=rank_tolerance
    )
    if not factor.full_rank:
        raise ValueError(
            "the stack [A; L] is rank deficient: null(A) and null(L) intersect"
        )
    q_a = factor.q[:m]
    q_l = factor.q[m:]
    svd = singular_value_decomposition(q_a)
    c = tuple(min(value, 1.0) for value in svd.singular_values)
    v = svd.v
    qlv = [
        [math.fsum(q_l[r][k] * v[k][j] for k in range(n)) for j in range(n)]
        for r in range(p)
    ]
    s = tuple(
        math.hypot(*(qlv[r][j] for r in range(p))) if p else 0.0 for j in range(n)
    )
    tiny = 1e-14
    z = tuple(
        tuple(qlv[r][j] / s[j] if s[j] > tiny else 0.0 for j in range(n))
        for r in range(p)
    )
    u = tuple(
        tuple(svd.u[r][j] if c[j] > tiny else 0.0 for j in range(n)) for r in range(m)
    )
    # X = R^-1 V by back substitution, column by column.
    upper = factor.r
    x_cols = []
    for j in range(n):
        column = [v[i][j] for i in range(n)]
        sol = [0.0] * n
        for i in range(n - 1, -1, -1):
            sol[i] = (
                column[i] - math.fsum(upper[i][k] * sol[k] for k in range(i + 1, n))
            ) / upper[i][i]
        x_cols.append(sol)
    x = tuple(tuple(x_cols[j][i] for j in range(n)) for i in range(n))
    gamma = tuple(
        ci / si if si > tiny else math.inf for ci, si in zip(c, s, strict=True)
    )
    return GeneralizedSVD(c=c, s=s, generalized_values=gamma, x=x, u=u, z=z)


def gsvd_tikhonov(
    decomposition: GeneralizedSVD,
    data: Sequence[float],
    regularization: float,
) -> Vector:
    """Return ``argmin ||A x - b||^2 + lambda^2 ||L x||^2`` from a GSVD.

    With ``x = X y``: ``y_i = c_i u_i'b / (c_i^2 + lambda^2 s_i^2)``
    [EQ-INV-SPEC-003]; directions with ``c_i = 0`` get ``y_i = 0``. The
    filter factor of mode ``i`` is ``gamma_i^2 / (gamma_i^2 + lambda^2)``.

    Args:
        decomposition: :class:`GeneralizedSVD` of ``(A, L)``.
        data: ``b`` (length ``m``).
        regularization: ``lambda >= 0`` (``0`` requires every ``c_i > 0``,
            i.e. full-column-rank ``A``).

    Returns:
        ``x_lambda``.

    Raises:
        ValueError: If ``lambda`` is negative or ``lambda = 0`` with a zero
            ``c_i``.

    References:
        [EQ-INV-SPEC-003], [REF-HANSEN-2008, Section 2.1].

    """
    lam = float(regularization)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    m = len(decomposition.u)
    b = _finite_vector(data, m, "data")
    n = len(decomposition.c)
    coefficients = []
    for j in range(n):
        cj, sj = decomposition.c[j], decomposition.s[j]
        denominator = cj * cj + lam * lam * sj * sj
        if cj <= 1e-14:
            coefficients.append(0.0)
            continue
        if denominator == 0.0:
            raise ValueError("lambda = 0 needs a full-column-rank A")
        beta = math.fsum(decomposition.u[r][j] * b[r] for r in range(m))
        coefficients.append(cj * beta / denominator)
    return tuple(
        math.fsum(decomposition.x[i][j] * coefficients[j] for j in range(n))
        for i in range(n)
    )
