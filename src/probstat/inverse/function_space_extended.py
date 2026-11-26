"""Likelihood-informed subspaces, functional resolution and non-sine priors.

Experimental (added in 0.28).

Three extensions of the function-space layer
(:mod:`probstat.inverse.function_space`,
:mod:`probstat.inverse.linear_gaussian`):

**Likelihood-informed subspace (LIS).** With the prior factor ``C0 = P P'``
and noise whitening ``W``, the prior-preconditioned Gauss-Newton Hessian at a
parameter point ``x_k`` is ``H_k = P' J_k' Gamma^-1 J_k P`` (``J_k`` the
Jacobian). The LIS is spanned by the leading eigenvectors of the average
``H = sum_k w_k H_k``, equivalently the generalized eigenproblem
``J' Gamma^-1 J v = lambda C0^-1 v`` with ``v = P w`` [EQ-INV-LIS-001]. The
rank is the number of eigenvalues above an explicit threshold
[EQ-INV-LIS-002]; for a nonlinear model the Jacobians are evaluated at
user-supplied points (posterior samples or a fixed design) and the result is
a Gauss-Newton average, **not** a global statement: it depends on the points
and neglects the second-order term of the misfit, and a sample spread is
reported. For a linear-Gaussian problem the likelihood is replaced by its
restriction to the subspace, which gives a Gaussian posterior in closed form
[EQ-INV-LIS-003] whose error against the exact posterior is computed densely
[EQ-INV-LIS-004]; the error is zero (to rounding) when the subspace contains
the range of the whitened adjoint, i.e. the rank equals the data-informed
rank.

**Functional resolution.** The resolution operator
``R = (G' Gamma^-1 G + C^-1)^-1 G' Gamma^-1 G`` of a coefficient-space
problem is evaluated without inverting ``C`` as ``R = C G' S^-1 G`` with
``S = G C G' + Gamma`` [EQ-INV-LIS-005]. For a field ``f = sum_j x_j phi_j``
with mass matrix ``M`` the averaging kernel of a functional ``l(f) = l' x`` is
the element of the span of the basis ``a = sum_j (M^-1 R' l)_j phi_j`` with
``l(f_hat - f_0) = int a f`` for every ``f`` in the span [EQ-INV-LIS-006];
its integral is ``l' R c`` with ``c = M^-1 int phi``, which equals ``l(1)``
exactly when the method resolves constants. Centroid, absolute-value width
and the Backus-Gilbert spread summarize the kernel [EQ-INV-LIS-007].

**Non-sine priors.** :class:`WhittleMaternPrior1D` discretizes the precision
``Q = h (kappa^2 - Laplacian)^alpha / amplitude^2`` for ``alpha`` in
``{1, 2}`` by second differences with Dirichlet (node-centred) or Neumann
(cell-centred, mirror ghost cell) boundary conditions; ``Q`` is banded and is
sampled through a banded Cholesky factor [EQ-INV-LIS-008], [EQ-INV-LIS-009].
The continuum covariance is the Matern covariance of the whole line,
``exp(-kappa r) / (2 kappa)`` for ``alpha = 1`` and
``(1 + kappa r) exp(-kappa r) / (4 kappa^3)`` for ``alpha = 2``, and the
covariance with a boundary condition is its method-of-images sum
[EQ-INV-LIS-010]. The measured boundary effect is reported by
:meth:`WhittleMaternPrior1D.boundary_effect` [EQ-INV-LIS-011].
:class:`CosineBasis` and :class:`ShiftedLegendreBasis` are bases with exact
mass matrices [EQ-INV-LIS-012], [EQ-INV-LIS-013].

Limitations: the dense LIS, resolution and projection-error routines are for
small problems (they refuse beyond ``max_dimension``); the LIS of a
nonlinear model is an empirical Gauss-Newton average; kernel width measures
are quadrature approximations when the kernel changes sign or has jumps
between quadrature panels; the SPDE prior is restricted to one dimension and
integer ``alpha``.

References:
    [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014],
    [REF-LINDGREN-RUE-LINDSTROM-2011], [REF-SPANTINI-ETAL-2015],
    [REF-TARANTOLA-2005], [REF-KAIPIO-SOMERSALO-2007],
    [REF-RASMUSSEN-WILLIAMS-2006], [REF-STUART-2010].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inverse.function_space import (
    HatBasis,
    PiecewiseConstantBasis,
    _composite_rule,
    _IntervalBasis,
    _positive_int,
    _positive_real,
)
from probstat.linalg import (
    BlockDiagonalCovariance,
    CallbackOperator,
    DenseCovariance,
    DiagonalCovariance,
    LinearOperator,
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    log_det_spd,
    materialize,
    matmul,
    require_operator,
    solve_spd,
    symmetric_eigen,
    transpose,
)

__all__ = [
    "AveragingKernel",
    "BoundaryEffect",
    "CosineBasis",
    "LISProjectedPosterior",
    "LISProjectionError",
    "LikelihoodInformedSubspace",
    "ResolutionOperator",
    "ShiftedLegendreBasis",
    "WhittleMaternPrior1D",
    "averaging_kernel",
    "cosine_spectral_prior",
    "likelihood_informed_subspace",
    "lis_linear_posterior",
    "lis_projection_error",
    "local_average_functional",
    "matern_covariance",
    "nonlinear_lis",
    "point_functional",
    "resolution_kernel_surface",
    "resolution_operator",
]

CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance
Boundary = Literal["dirichlet", "neumann"]

_RANK_TOLERANCE = 1e-12
_PSD_ROUNDOFF = 1e-10
_ORTHONORMALITY_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Euclidean inner product with compensated summation.

    Args:
        left: First vector.
        right: Second vector of the same length.

    Returns:
        ``sum_i left_i right_i``.

    References:
        [EQ-INV-LIS-001].

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _unit(size: int, index: int) -> Vector:
    """Return the ``index``-th unit vector of length ``size``.

    Args:
        size: Vector length.
        index: Position of the one.

    Returns:
        The unit vector.

    References:
        [EQ-INV-LIS-001].

    """
    return tuple(1.0 if i == index else 0.0 for i in range(size))


def _require_noise(noise: object, size: int) -> CovarianceFactor:
    """Return ``noise`` after checking type and dimension.

    Args:
        noise: Candidate covariance factor.
        size: Required dimension.

    Returns:
        The covariance factor.

    Raises:
        TypeError: If ``noise`` is not a covariance factor.
        ValueError: If the dimension differs from ``size``.

    References:
        [EQ-INV-LIS-001].

    """
    if not isinstance(
        noise, (DiagonalCovariance, DenseCovariance, BlockDiagonalCovariance)
    ):
        raise TypeError("noise must be a probstat.linalg covariance factor")
    if noise.dimension != size:
        raise ValueError(f"noise has dimension {noise.dimension}, expected {size}")
    return noise


def _require_factor(prior_factor: object, size: int) -> LinearOperator:
    """Return the prior factor ``P`` (``size x r``) with ``C0 = P P'``.

    Args:
        prior_factor: Covariance factor, operator or dense matrix.
        size: Required number of rows.

    Returns:
        The factor as a linear operator.

    Raises:
        TypeError: If the argument has the wrong type.
        ValueError: If the dimension is wrong.

    References:
        [EQ-INV-LIS-001].

    """
    if isinstance(
        prior_factor, (DiagonalCovariance, DenseCovariance, BlockDiagonalCovariance)
    ):
        if prior_factor.dimension != size:
            raise ValueError(
                f"prior has dimension {prior_factor.dimension}, expected {size}"
            )
        return prior_factor.factor_operator()
    factor = require_operator(prior_factor, name="prior_factor")
    if factor.shape[0] != size:
        raise ValueError(f"prior_factor has {factor.shape[0]} rows, expected {size}")
    if factor.shape[1] < 1:
        raise ValueError("prior_factor must have at least one column")
    return factor


def _whitened_matrix(
    forward: LinearOperator, noise: CovarianceFactor, factor: LinearOperator
) -> Matrix:
    """Return the dense ``K = W A P`` (``m x r``) with ``W' W = Gamma^-1``.

    Args:
        forward: ``A`` (``m x n``).
        noise: Noise covariance factor.
        factor: ``P`` (``n x r``).

    Returns:
        ``K`` as rows.

    References:
        [EQ-INV-LIS-001].

    """
    r = factor.shape[1]
    columns = [
        noise.whiten(forward.matvec(factor.matvec(_unit(r, j)))) for j in range(r)
    ]
    return transpose(tuple(columns))


def _gram(k_matrix: Matrix) -> Matrix:
    """Return the symmetrized ``K' K``.

    Args:
        k_matrix: Dense ``K``.

    Returns:
        ``r x r`` symmetric positive semidefinite matrix.

    References:
        [EQ-INV-LIS-001].

    """
    gram = matmul(transpose(k_matrix), k_matrix)
    n = len(gram)
    return tuple(
        tuple(0.5 * (gram[i][j] + gram[j][i]) for j in range(n)) for i in range(n)
    )


def _check_dimension(size: int, limit: int, name: str) -> None:
    """Refuse dense work above a stated limit.

    Args:
        size: Dimension of the dense problem.
        limit: Allowed maximum.
        name: Description for the message.

    Raises:
        ValueError: If ``size`` exceeds ``limit``.

    References:
        [EQ-INV-LIS-001].

    """
    if size > limit:
        raise ValueError(
            f"{name} dimension {size} exceeds max_dimension={limit}; "
            "this dense routine is for small problems"
        )


def _descending_eigen(matrix: Matrix) -> tuple[Vector, Matrix]:
    """Return eigenvalues in descending order and eigenvectors as rows.

    Args:
        matrix: Symmetric positive semidefinite matrix.

    Returns:
        ``(values, vectors)`` with ``vectors[i]`` the eigenvector of
        ``values[i]``; negative values within roundoff of zero are set to
        zero because the matrix is positive semidefinite by construction.

    Raises:
        ValueError: If an eigenvalue is negative beyond roundoff.

    References:
        [EQ-INV-LIS-001].

    """
    values, vectors = symmetric_eigen(matrix)
    n = len(values)
    order = list(range(n - 1, -1, -1))
    descending = [values[i] for i in order]
    scale = max(descending[0], 1.0)
    if descending[-1] < -_PSD_ROUNDOFF * scale:
        raise ValueError("negative eigenvalue of the PSD Hessian beyond roundoff")
    # K'K is positive semidefinite: roundoff-level negatives are set to zero.
    descending = [max(v, 0.0) for v in descending]
    rows = tuple(tuple(vectors[r][i] for r in range(n)) for i in order)
    return tuple(descending), rows


# ---------------------------------------------------------------------------
# 1. Likelihood-informed subspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LikelihoodInformedSubspace:
    """Leading eigenspace of the averaged prior-preconditioned Hessian.

    Attributes:
        eigenvalues: All ``r`` eigenvalues ``lambda_i`` of ``H`` in
            descending order.
        basis: ``basis[i]`` is the whitened direction ``w_i`` (length ``r``)
            of the ``rank`` retained eigenpairs, orthonormal.
        parameter_directions: ``P w_i`` (length ``n``): the generalized
            eigenvectors ``v`` of ``J' Gamma^-1 J v = lambda C0^-1 v``.
        rank: Number of retained directions.
        threshold: The eigenvalue threshold used.
        informed_rank: Number of eigenvalues above ``1e-12 lambda_1`` (the
            numerically data-informed rank of the averaged Hessian).
        capped: ``True`` if ``max_rank`` removed directions above the
            threshold.
        gap: ``lambda_rank - lambda_{rank+1}`` (``lambda_rank`` if the
            subspace is everything); a small gap means the subspace is
            sensitive to the points and to rounding.
        point_count: Number of Jacobians averaged.
        relative_sample_spread: ``sqrt(sum_k w_k ||H_k - H||_F^2) / ||H||_F``
            (zero for one point or a zero Hessian): the variability of the
            Hessian across the points.
        relative_standard_error: ``relative_sample_spread / sqrt(count)``,
            meaningful only if the points are independent draws.
        prior_factor: ``P``.

    References:
        [EQ-INV-LIS-001], [EQ-INV-LIS-002].

    """

    eigenvalues: Vector
    basis: Matrix
    parameter_directions: Matrix
    rank: int
    threshold: float
    informed_rank: int
    capped: bool
    gap: float
    point_count: int
    relative_sample_spread: float
    relative_standard_error: float
    prior_factor: LinearOperator

    def project(self, vector: Sequence[float]) -> Vector:
        """Return ``Pi z = sum_i w_i (w_i' z)`` for a whitened vector.

        Args:
            vector: Vector ``z`` of length ``r``.

        Returns:
            The orthogonal projection onto the LIS.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LIS-003].

        """
        z = as_vector(vector, name="vector")
        r = self.prior_factor.shape[1]
        if len(z) != r:
            raise ValueError(f"vector has length {len(z)}, expected {r}")
        out = [0.0] * r
        for direction in self.basis:
            c = _dot(direction, z)
            out = [a + c * b for a, b in zip(out, direction, strict=True)]
        return tuple(out)


def _weights(weights: Sequence[float] | None, count: int) -> list[float]:
    """Return normalized nonnegative point weights.

    Args:
        weights: Optional weights.
        count: Number of points.

    Returns:
        Weights summing to one.

    Raises:
        ValueError: If a weight is negative or the sum is not positive.

    References:
        [EQ-INV-LIS-001].

    """
    if weights is None:
        return [1.0 / count] * count
    values = [require_finite_real(w, name="weight") for w in weights]
    if len(values) != count:
        raise ValueError(f"weights have length {len(values)}, expected {count}")
    if any(w < 0.0 for w in values):
        raise ValueError("weights must be nonnegative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    return [w / total for w in values]


def likelihood_informed_subspace(
    jacobians: Sequence[object],
    *,
    noise: object,
    prior_factor: object,
    threshold: float,
    max_rank: int | None = None,
    weights: Sequence[float] | None = None,
    max_dimension: int = 400,
) -> LikelihoodInformedSubspace:
    """Return the LIS of the averaged Gauss-Newton Hessian.

    ``H = sum_k w_k K_k' K_k`` with ``K_k = W J_k P`` is formed densely and
    diagonalized; the rank is the number of eigenvalues strictly above
    ``threshold`` (an eigenvalue ``lambda`` is the ratio of posterior to
    prior precision gain along its direction, so ``threshold = 1`` keeps
    directions where the data at least double the prior precision).

    Args:
        jacobians: Non-empty sequence of ``m x n`` operators (or dense
            matrices): the Jacobian at each point, all of the same shape.
        noise: Noise covariance factor of dimension ``m``.
        prior_factor: ``P`` (``n x r``), an operator, matrix or covariance
            factor with ``C0 = P P'``.
        threshold: Positive finite eigenvalue threshold.
        max_rank: Optional cap on the rank (``capped`` reports whether it
            removed directions above the threshold).
        weights: Optional nonnegative point weights, normalized to sum to
            one (default equal).
        max_dimension: Refuse ``r`` larger than this (dense eigensolver).

    Returns:
        :class:`LikelihoodInformedSubspace`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree, the threshold or ``max_rank`` is
            invalid, or ``r`` exceeds ``max_dimension``.

    References:
        [EQ-INV-LIS-001], [EQ-INV-LIS-002],
        [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014],
        [REF-SPANTINI-ETAL-2015].

    """
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
    _check_dimension(r, _positive_int(max_dimension, "max_dimension"), "whitened")
    level = _positive_real(threshold, "threshold")
    cap: int | None = None
    if max_rank is not None:
        cap = _positive_int(max_rank, "max_rank")
    w = _weights(weights, len(operators))
    hessians = [_gram(_whitened_matrix(op, checked_noise, factor)) for op in operators]
    average = [
        [math.fsum(w[k] * hessians[k][i][j] for k in range(len(w))) for j in range(r)]
        for i in range(r)
    ]
    matrix = tuple(tuple(row) for row in average)
    values, vectors = _descending_eigen(matrix)
    informed = sum(1 for v in values if v > _RANK_TOLERANCE * values[0])
    above = sum(1 for v in values if v > level)
    rank = above if cap is None else min(above, cap)
    norm = math.sqrt(math.fsum(a * a for row in matrix for a in row))
    if len(operators) > 1 and norm > 0.0:
        variance = math.fsum(
            w[k]
            * math.fsum(
                (hessians[k][i][j] - matrix[i][j]) ** 2
                for i in range(r)
                for j in range(r)
            )
            for k in range(len(w))
        )
        spread = math.sqrt(variance) / norm
    else:
        spread = 0.0
    if rank == 0:
        gap = 0.0
    elif rank == r:
        gap = values[rank - 1]
    else:
        gap = values[rank - 1] - values[rank]
    kept = vectors[:rank]
    return LikelihoodInformedSubspace(
        eigenvalues=values,
        basis=tuple(kept),
        parameter_directions=tuple(factor.matvec(v) for v in kept),
        rank=rank,
        threshold=level,
        informed_rank=informed,
        capped=cap is not None and above > cap,
        gap=gap,
        point_count=len(operators),
        relative_sample_spread=spread,
        relative_standard_error=spread / math.sqrt(len(operators)),
        prior_factor=factor,
    )


def nonlinear_lis(
    jacobian: Callable[[Vector], object],
    points: Sequence[Sequence[float]],
    *,
    noise: object,
    prior_factor: object,
    threshold: float,
    max_rank: int | None = None,
    weights: Sequence[float] | None = None,
    max_dimension: int = 400,
) -> LikelihoodInformedSubspace:
    """Return the empirical LIS of a nonlinear model over a set of points.

    The Jacobian of the forward model is evaluated at every point (posterior
    samples, or a fixed design over the prior) and averaged as in
    :func:`likelihood_informed_subspace`. This is a Gauss-Newton average:
    it is not the LIS of the exact posterior, depends on the points, and
    must not be presented as a global subspace; compare
    ``relative_sample_spread`` and repeat with other points.

    Args:
        jacobian: Callable mapping a parameter vector to its ``m x n``
            Jacobian (operator or matrix).
        points: Non-empty sequence of parameter vectors.
        noise: Noise covariance factor.
        prior_factor: Prior factor ``P``.
        threshold: Positive eigenvalue threshold.
        max_rank: Optional cap on the rank.
        weights: Optional point weights.
        max_dimension: Dense dimension limit.

    Returns:
        :class:`LikelihoodInformedSubspace`.

    Raises:
        TypeError: If ``jacobian`` is not callable.
        ValueError: If ``points`` is empty or inputs are invalid.

    References:
        [EQ-INV-LIS-001], [EQ-INV-LIS-002],
        [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014].

    """
    if not callable(jacobian):
        raise TypeError("jacobian must be callable")
    pts = [as_vector(p, name="point") for p in points]
    if not pts:
        raise ValueError("points must be non-empty")
    return likelihood_informed_subspace(
        [jacobian(p) for p in pts],
        noise=noise,
        prior_factor=prior_factor,
        threshold=threshold,
        max_rank=max_rank,
        weights=weights,
        max_dimension=max_dimension,
    )


@dataclass(frozen=True, slots=True)
class LISProjectedPosterior:
    """Gaussian posterior with the likelihood restricted to a subspace.

    The likelihood ``L(z)`` of the whitened coordinate is replaced by
    ``L(Pi z)`` with ``Pi = W_r W_r'`` the projector onto the subspace;
    the posterior precision is ``I + Pi H Pi`` and the mean
    ``(I + Pi H Pi)^-1 Pi K' W r``.

    Attributes:
        mean: ``mu + P z_mean``.
        whitened_mean: ``z_mean``.
        whitened_covariance: Dense ``r x r`` covariance
            ``I - W_r (I - (I + W_r' H W_r)^-1) W_r'`` of ``z``.
        rank: Subspace rank.
        prior_mean: ``mu``.
        prior_factor: ``P``.

    References:
        [EQ-INV-LIS-003].

    """

    mean: Vector
    whitened_mean: Vector
    whitened_covariance: Matrix
    rank: int
    prior_mean: Vector
    prior_factor: LinearOperator

    def covariance_matrix(self, *, max_entries: int = 1_000_000) -> Matrix:
        """Return the dense ``n x n`` covariance ``P Sigma_z P'``.

        Args:
            max_entries: Limit on ``n r`` for materializing ``P``.

        Returns:
            The covariance of ``x``.

        Raises:
            ValueError: If ``P`` is too large to materialize.

        References:
            [EQ-INV-LIS-003].

        """
        p = materialize(self.prior_factor, max_entries=max_entries)
        return matmul(matmul(p, self.whitened_covariance), transpose(p))


def _problem(
    operator: object,
    data: Sequence[float],
    noise: object,
    prior_factor: object,
    prior_mean: Sequence[float] | None,
    max_dimension: int,
) -> tuple[LinearOperator, Vector, Vector, Matrix, Vector, Matrix, LinearOperator]:
    """Return the dense whitened ingredients of a linear-Gaussian problem.

    Args:
        operator: ``A``.
        data: ``y``.
        noise: Noise factor.
        prior_factor: ``P``.
        prior_mean: ``mu`` or ``None``.
        max_dimension: Dense limit on ``r``.

    Returns:
        ``(A, y, mu, H, b, K, P)`` with ``H = K'K`` and ``b = K' W (y - A mu)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree.

    References:
        [EQ-INV-LIS-003].

    """
    forward = require_operator(operator, name="operator")
    m, n = forward.shape
    y = as_vector(data, name="data")
    if len(y) != m:
        raise ValueError(f"data has length {len(y)}, expected {m}")
    checked_noise = _require_noise(noise, m)
    factor = _require_factor(prior_factor, n)
    _check_dimension(factor.shape[1], max_dimension, "whitened")
    if prior_mean is None:
        mu: Vector = (0.0,) * n
    else:
        mu = as_vector(prior_mean, name="prior_mean")
        if len(mu) != n:
            raise ValueError(f"prior_mean has length {len(mu)}, expected {n}")
    k = _whitened_matrix(forward, checked_noise, factor)
    residual = tuple(a - b for a, b in zip(y, forward.matvec(mu), strict=True))
    whitened_residual = checked_noise.whiten(residual)
    b = tuple(
        math.fsum(k[i][j] * whitened_residual[i] for i in range(m))
        for j in range(factor.shape[1])
    )
    return forward, y, mu, _gram(k), b, k, factor


def _projected_moments(
    h: Matrix, b: Vector, basis: Matrix
) -> tuple[Vector, Matrix, Matrix]:
    """Return mean, covariance and the reduced Hessian of the LIS posterior.

    Args:
        h: Dense ``H = K'K`` (``r x r``).
        b: ``K' W (y - A mu)``.
        basis: Orthonormal rows ``w_i`` spanning the subspace.

    Returns:
        ``(z_mean, Sigma_z, H_k)`` with ``H_k = W_r' H W_r``.

    Raises:
        ValueError: If the basis is not orthonormal.

    References:
        [EQ-INV-LIS-003].

    """
    r = len(h)
    k = len(basis)
    for i in range(k):
        for j in range(k):
            expected = 1.0 if i == j else 0.0
            if abs(_dot(basis[i], basis[j]) - expected) > _ORTHONORMALITY_TOLERANCE:
                raise ValueError("subspace basis is not orthonormal")
    if k == 0:
        return (
            (0.0,) * r,
            tuple(tuple(1.0 if i == j else 0.0 for j in range(r)) for i in range(r)),
            (),
        )
    hw = [[_dot(h[i], basis[j]) for j in range(k)] for i in range(r)]
    reduced = tuple(
        tuple(math.fsum(basis[i][a] * hw[a][j] for a in range(r)) for j in range(k))
        for i in range(k)
    )
    shifted = tuple(
        tuple(reduced[i][j] + (1.0 if i == j else 0.0) for j in range(k))
        for i in range(k)
    )
    inverse = inverse_spd(shifted)
    projected_b = tuple(_dot(basis[i], b) for i in range(k))
    coeff = [_dot(inverse[i], projected_b) for i in range(k)]
    mean = tuple(math.fsum(coeff[i] * basis[i][a] for i in range(k)) for a in range(r))
    covariance = tuple(
        tuple(
            (1.0 if a == c else 0.0)
            - math.fsum(
                basis[i][a] * ((1.0 if i == j else 0.0) - inverse[i][j]) * basis[j][c]
                for i in range(k)
                for j in range(k)
            )
            for c in range(r)
        )
        for a in range(r)
    )
    return mean, covariance, reduced


def lis_linear_posterior(
    operator: object,
    data: Sequence[float],
    *,
    noise: object,
    prior_factor: object,
    prior_mean: Sequence[float] | None = None,
    subspace: LikelihoodInformedSubspace,
    max_dimension: int = 400,
) -> LISProjectedPosterior:
    """Return the LIS-projected posterior of a linear-Gaussian problem.

    The subspace may come from any set of Jacobians (for a linear model all
    equal to ``A``, or an average over points); the posterior is exact for
    the likelihood restricted to the subspace. If the subspace contains the
    range of ``K' = P' A' W'`` the result equals the exact posterior.

    Args:
        operator: ``A`` (``m x n``).
        data: Observation ``y``.
        noise: Noise covariance factor.
        prior_factor: ``P`` (must be the factor used to build the subspace).
        prior_mean: ``mu`` (default zeros).
        subspace: A :class:`LikelihoodInformedSubspace`.
        max_dimension: Dense dimension limit.

    Returns:
        :class:`LISProjectedPosterior`.

    Raises:
        TypeError: If ``subspace`` has the wrong type.
        ValueError: If dimensions disagree or the basis is not orthonormal.

    References:
        [EQ-INV-LIS-003], [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014].

    """
    if not isinstance(subspace, LikelihoodInformedSubspace):
        raise TypeError("subspace must be a LikelihoodInformedSubspace")
    _, _, mu, h, b, _, factor = _problem(
        operator, data, noise, prior_factor, prior_mean, max_dimension
    )
    if factor.shape[1] != subspace.prior_factor.shape[1]:
        raise ValueError("prior_factor does not match the subspace")
    z_mean, covariance, _ = _projected_moments(h, b, subspace.basis)
    mean = tuple(a + c for a, c in zip(mu, factor.matvec(z_mean), strict=True))
    return LISProjectedPosterior(
        mean=mean,
        whitened_mean=z_mean,
        whitened_covariance=covariance,
        rank=subspace.rank,
        prior_mean=mu,
        prior_factor=factor,
    )


@dataclass(frozen=True, slots=True)
class LISProjectionError:
    """Error of the LIS-projected posterior against the exact posterior.

    All covariance errors are for the whitened coordinate ``z``; the mean
    errors are given in both ``z`` and ``x``.

    Attributes:
        rank: Subspace rank.
        mean_error_whitened: ``||z_exact - z_lis||_2``.
        mean_error: ``||x_exact - x_lis||_2``.
        covariance_frobenius_error: ``||Sigma_exact - Sigma_lis||_F``.
        covariance_operator_error: spectral norm of the difference.
        trace_difference: ``tr(Sigma_lis - Sigma_exact)``.
        kl_exact_to_lis: ``KL(N_exact || N_lis)`` in nats.
        omitted_spectrum_trace: ``sum_{i>rank} lambda_i / (1 + lambda_i)``
            over the eigenvalues of ``H`` beyond the leading ``rank``; equal
            to ``trace_difference`` when the subspace is the leading
            eigen-subspace of this ``H``, otherwise only a reference value.
        omitted_spectrum_bound: ``lambda_{rank+1} / (1 + lambda_{rank+1})``,
            the corresponding operator-norm value.
        exact_mean: Exact whitened posterior mean.
        exact_covariance: Exact whitened posterior covariance
            ``(I + H)^-1``.

    References:
        [EQ-INV-LIS-004].

    """

    rank: int
    mean_error_whitened: float
    mean_error: float
    covariance_frobenius_error: float
    covariance_operator_error: float
    trace_difference: float
    kl_exact_to_lis: float
    omitted_spectrum_trace: float
    omitted_spectrum_bound: float
    exact_mean: Vector
    exact_covariance: Matrix


def lis_projection_error(
    operator: object,
    data: Sequence[float],
    *,
    noise: object,
    prior_factor: object,
    prior_mean: Sequence[float] | None = None,
    subspace: LikelihoodInformedSubspace,
    max_dimension: int = 400,
) -> LISProjectionError:
    """Return the exact error of the LIS-projected linear-Gaussian posterior.

    The exact whitened posterior ``N((I + H)^-1 b, (I + H)^-1)`` is formed
    densely and compared with the projected one. The Kullback-Leibler
    divergence uses ``S_lis^-1 = I + Pi H Pi``:
    ``2 KL = tr(S_lis^-1 S) - r + dm' S_lis^-1 dm + log det S_lis - log det S``.
    For the leading eigen-subspace of ``H`` the covariance difference
    ``Sigma_exact - Sigma_lis`` is ``-sum_{i>k} lambda_i / (1 + lambda_i)
    v_i v_i'``, so ``trace_difference = tr(Sigma_lis - Sigma_exact)``
    equals ``omitted_spectrum_trace``.

    Args:
        operator: ``A``.
        data: ``y``.
        noise: Noise factor.
        prior_factor: ``P``.
        prior_mean: ``mu``.
        subspace: The subspace to test.
        max_dimension: Dense dimension limit.

    Returns:
        :class:`LISProjectionError`.

    Raises:
        TypeError: If ``subspace`` has the wrong type.
        ValueError: If dimensions disagree.

    References:
        [EQ-INV-LIS-004], [REF-CUI-MARTIN-MARZOUK-SOLONEN-SPANTINI-2014].

    """
    if not isinstance(subspace, LikelihoodInformedSubspace):
        raise TypeError("subspace must be a LikelihoodInformedSubspace")
    _, _, _, h, b, _, factor = _problem(
        operator, data, noise, prior_factor, prior_mean, max_dimension
    )
    r = len(h)
    if r != subspace.prior_factor.shape[1]:
        raise ValueError("prior_factor does not match the subspace")
    shifted = tuple(
        tuple(h[i][j] + (1.0 if i == j else 0.0) for j in range(r)) for i in range(r)
    )
    exact_cov = inverse_spd(shifted)
    exact_mean = solve_spd(shifted, b)
    lis_mean, lis_cov, reduced = _projected_moments(h, b, subspace.basis)
    k = subspace.rank
    delta = tuple(a - c for a, c in zip(exact_mean, lis_mean, strict=True))
    diff = tuple(
        tuple(
            0.5 * (exact_cov[i][j] - lis_cov[i][j] + exact_cov[j][i] - lis_cov[j][i])
            for j in range(r)
        )
        for i in range(r)
    )
    frobenius = math.sqrt(math.fsum(v * v for row in diff for v in row))
    # relative Jacobi tolerance 1e-10 of ||diff||_F bounds the eigenvalue error;
    # the default 1e-14 is unattainable when diff is pure roundoff.
    values, _ = symmetric_eigen(diff, tolerance=1e-10)
    operator_norm = max(abs(v) for v in values)
    trace_difference = math.fsum(lis_cov[i][i] - exact_cov[i][i] for i in range(r))
    # 2 KL = tr(S_lis^-1 S) - r + dm' S_lis^-1 dm + logdet S_lis - logdet S.
    basis = subspace.basis
    projected_delta = [_dot(basis[i], delta) for i in range(k)]
    quad = _dot(delta, delta) + math.fsum(
        projected_delta[i] * reduced[i][j] * projected_delta[j]
        for i in range(k)
        for j in range(k)
    )
    trace_term = math.fsum(exact_cov[i][i] for i in range(r))
    if k:
        sw = [[_dot(exact_cov[a], basis[j]) for j in range(k)] for a in range(r)]
        small = [
            [math.fsum(basis[i][a] * sw[a][j] for a in range(r)) for j in range(k)]
            for i in range(k)
        ]
        trace_term += math.fsum(
            reduced[i][j] * small[j][i] for i in range(k) for j in range(k)
        )
        shifted_reduced = tuple(
            tuple(reduced[i][j] + (1.0 if i == j else 0.0) for j in range(k))
            for i in range(k)
        )
        logdet_lis = -log_det_spd(shifted_reduced)
    else:
        logdet_lis = 0.0
    logdet_exact = -log_det_spd(shifted)
    kl = 0.5 * (trace_term - r + quad + logdet_lis - logdet_exact)
    spectrum, _ = _descending_eigen(h)
    omitted = spectrum[k:]
    dm_x = factor.matvec(delta)
    return LISProjectionError(
        rank=k,
        mean_error_whitened=math.sqrt(_dot(delta, delta)),
        mean_error=math.sqrt(_dot(dm_x, dm_x)),
        covariance_frobenius_error=frobenius,
        covariance_operator_error=operator_norm,
        trace_difference=trace_difference,
        kl_exact_to_lis=kl,
        omitted_spectrum_trace=math.fsum(v / (1.0 + v) for v in omitted),
        omitted_spectrum_bound=(omitted[0] / (1.0 + omitted[0]) if omitted else 0.0),
        exact_mean=exact_mean,
        exact_covariance=exact_cov,
    )


# ---------------------------------------------------------------------------
# 2. Functional resolution with mass matrices
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolutionOperator:
    """Coefficient-space resolution operator and its function-space data.

    Attributes:
        matrix: ``R = C G' S^-1 G`` (``n x n``), the derivative of the
            posterior mean with respect to the true coefficients.
        mass_matrix: ``M`` of the basis.
        integral_vector: ``m_j = int phi_j ds``.
        constant_coefficients: ``c = M^-1 m``, the ``L2`` projection of the
            constant one onto the span of the basis (the constant itself
            if the basis contains it).
        constant_defect: ``||R c - c||_M / ||c||_M``: zero iff the method
            resolves constants exactly.
        trace: ``tr R``, the effective number of resolved degrees of
            freedom.
        length: Interval length.
        panels: Quadrature panels used for the integral vector.

    References:
        [EQ-INV-LIS-005], [EQ-INV-LIS-006].

    """

    matrix: Matrix
    mass_matrix: Matrix
    integral_vector: Vector
    constant_coefficients: Vector
    constant_defect: float
    trace: float
    length: float
    panels: int


def _default_panels(basis: _IntervalBasis) -> int:
    """Return a quadrature panel count aligned with the basis cells.

    Args:
        basis: Interval basis.

    Returns:
        A multiple of the number of cells for piecewise bases, 200
        otherwise; panel edges then coincide with the kinks and jumps.

    References:
        [EQ-INV-LIS-006].

    """
    n = basis.dimension
    if isinstance(basis, PiecewiseConstantBasis):
        return n * math.ceil(200 / n)
    if isinstance(basis, HatBasis):
        return (n + 1) * math.ceil(200 / (n + 1))
    return 200


def resolution_operator(
    basis: _IntervalBasis,
    operator: object,
    *,
    noise: object,
    prior_factor: object,
    panels: int | None = None,
    max_dimension: int = 400,
) -> ResolutionOperator:
    """Return ``R = C G' S^-1 G`` with function-space bookkeeping.

    ``R`` equals ``(G' Gamma^-1 G + C^-1)^-1 G' Gamma^-1 G`` (matrix
    inversion lemma) without inverting ``C``; ``S = G C G' + Gamma`` is
    factored by Cholesky and no pseudo-inverse is used.

    Args:
        basis: Interval basis (supplies ``M``, the integral of the basis and
            the interval).
        operator: ``G`` (``m x n`` with ``n = basis.dimension``), coefficient
            to data.
        noise: Noise covariance factor of dimension ``m``.
        prior_factor: ``P`` (``n x r``) of the coefficient prior ``C = P P'``.
        panels: Number of quadrature panels for the constant's integral
            (default aligned with the basis cells).
        max_dimension: Refuse ``n`` or ``r`` above this.

    Returns:
        :class:`ResolutionOperator`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree, a dimension exceeds
            ``max_dimension`` or ``S`` is not positive definite.

    References:
        [EQ-INV-LIS-005], [EQ-INV-LIS-006], [REF-TARANTOLA-2005],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    forward = require_operator(operator, name="operator")
    m, n = forward.shape
    if n != basis.dimension:
        raise ValueError(f"operator has {n} columns, basis has {basis.dimension}")
    limit = _positive_int(max_dimension, "max_dimension")
    checked_noise = _require_noise(noise, m)
    factor = _require_factor(prior_factor, n)
    _check_dimension(max(n, factor.shape[1]), limit, "coefficient")
    g = materialize(forward)
    p = materialize(factor)
    gp = matmul(g, p)
    gamma = materialize(checked_noise.covariance_operator())
    gpt = transpose(gp)
    s_matrix = matmul(gp, gpt)
    s_dense = tuple(
        tuple(0.5 * (s_matrix[i][j] + s_matrix[j][i]) + gamma[i][j] for j in range(m))
        for i in range(m)
    )
    lower = cholesky(s_dense)
    columns = [
        cholesky_solve(lower, tuple(g[i][j] for i in range(m))) for j in range(n)
    ]
    solved = transpose(tuple(columns))
    r_matrix = matmul(p, matmul(gpt, solved))
    count = (
        _default_panels(basis) if panels is None else _positive_int(panels, "panels")
    )
    ones = basis.integral_operator([lambda _s: 1.0], count)
    integral = tuple(ones.matrix[0])
    mass = basis.mass_matrix()
    constant = solve_spd(mass, integral)
    resolved = tuple(_dot(row, constant) for row in r_matrix)
    defect_vec = tuple(a - c for a, c in zip(resolved, constant, strict=True))

    def mass_norm(v: Sequence[float]) -> float:
        """Return ``sqrt(v' M v)``.

        References:
            [EQ-INV-LIS-006].

        """
        return math.sqrt(max(math.fsum(v[i] * _dot(mass[i], v) for i in range(n)), 0.0))

    return ResolutionOperator(
        matrix=r_matrix,
        mass_matrix=mass,
        integral_vector=integral,
        constant_coefficients=constant,
        constant_defect=mass_norm(defect_vec) / mass_norm(constant),
        trace=math.fsum(r_matrix[i][i] for i in range(n)),
        length=basis.length,
        panels=count,
    )


def point_functional(basis: _IntervalBasis, point: float) -> Vector:
    """Return the coefficient vector of the point functional ``f -> f(s0)``.

    Args:
        basis: Interval basis.
        point: Location ``s0`` in ``[0, L]``.

    Returns:
        ``(phi_j(s0))_j``.

    Raises:
        ValueError: If the point is outside ``[0, L]``.

    References:
        [EQ-INV-LIS-006].

    """
    return tuple(basis.evaluation_operator([point]).matrix[0])


def local_average_functional(
    basis: _IntervalBasis,
    weight: Callable[[float], float],
    panels: int | None = None,
) -> Vector:
    """Return the coefficients of ``f -> int w(s) f(s) ds``.

    Args:
        basis: Interval basis.
        weight: Weight ``w`` (for a local average, integrating to one).
        panels: Quadrature panels (default aligned with the basis cells).

    Returns:
        ``(int w phi_j ds)_j``.

    Raises:
        TypeError: If ``weight`` is not callable.
        ValueError: If a weight value is not finite.

    References:
        [EQ-INV-LIS-006], [REF-DAVIS-RABINOWITZ-1984].

    """
    count = (
        _default_panels(basis) if panels is None else _positive_int(panels, "panels")
    )
    return tuple(basis.integral_operator([weight], count).matrix[0])


@dataclass(frozen=True, slots=True)
class AveragingKernel:
    """Averaging kernel of a functional in function space.

    The kernel is ``a(s) = sum_j coefficients_j phi_j(s)``;
    ``l(f_hat - f_0) = int a (f - f_0) ds`` for all ``f`` in the basis span.

    Attributes:
        coefficients: ``a = M^-1 R' l``.
        integral: ``int a ds = m' a = l' R c`` (exact).
        target_integral: ``l' c`` (``l(1)`` if the basis contains the
            constants): the integral of a perfectly resolving kernel.
        centroid: ``int s |a| ds / int |a| ds``.
        rms_width: ``sqrt(int (s - centroid)^2 |a| ds / int |a| ds)``.
        center: Location used for the spread.
        backus_gilbert_spread: ``12 int (s - center)^2 a^2 ds / (int a)^2``
            (length units; a box of width ``w`` centred at ``center`` gives
            ``w``). ``nan`` if the integral is zero.
        negative_mass_fraction: ``int a^- ds / int |a| ds``.

    References:
        [EQ-INV-LIS-006], [EQ-INV-LIS-007].

    """

    coefficients: Vector
    integral: float
    target_integral: float
    centroid: float
    rms_width: float
    center: float
    backus_gilbert_spread: float
    negative_mass_fraction: float


def averaging_kernel(
    resolution: ResolutionOperator,
    basis: _IntervalBasis,
    functional: Sequence[float],
    *,
    center: float | None = None,
    panels: int | None = None,
) -> AveragingKernel:
    """Return the averaging kernel of a functional and its width measures.

    Args:
        resolution: Result of :func:`resolution_operator` for the same basis.
        basis: The basis.
        functional: Coefficient vector ``l`` of the functional (from
            :func:`point_functional` or :func:`local_average_functional`).
        center: Location for the Backus-Gilbert spread (default the
            centroid of ``|a|``).
        panels: Quadrature panels for the width integrals (default aligned
            with the basis cells).

    Returns:
        :class:`AveragingKernel`.

    Raises:
        ValueError: If lengths disagree or the center is outside ``[0, L]``.

    References:
        [EQ-INV-LIS-006], [EQ-INV-LIS-007], [REF-TARANTOLA-2005].

    """
    n = basis.dimension
    ell = as_vector(functional, name="functional")
    if len(ell) != n or len(resolution.matrix) != n:
        raise ValueError("functional, resolution and basis dimensions disagree")
    transposed = transpose(resolution.matrix)
    rhs = tuple(_dot(row, ell) for row in transposed)
    a = solve_spd(resolution.mass_matrix, rhs)
    count = (
        _default_panels(basis) if panels is None else _positive_int(panels, "panels")
    )
    nodes, weights = _composite_rule(basis.length, count)
    values = basis.evaluate(a, nodes)
    mass_abs = math.fsum(w * abs(v) for w, v in zip(weights, values, strict=True))
    negative = math.fsum(w * max(-v, 0.0) for w, v in zip(weights, values, strict=True))
    if mass_abs > 0.0:
        centroid = (
            math.fsum(
                w * s * abs(v) for w, s, v in zip(weights, nodes, values, strict=True)
            )
            / mass_abs
        )
        width = math.sqrt(
            math.fsum(
                w * (s - centroid) ** 2 * abs(v)
                for w, s, v in zip(weights, nodes, values, strict=True)
            )
            / mass_abs
        )
    else:
        centroid = math.nan
        width = math.nan
    if center is None:
        spread_center = centroid
    else:
        spread_center = require_finite_real(center, name="center")
        if spread_center < 0.0 or spread_center > basis.length:
            raise ValueError("center must lie in [0, L]")
    integral = _dot(resolution.integral_vector, a)
    if integral != 0.0 and not math.isnan(spread_center):
        second = math.fsum(
            w * (s - spread_center) ** 2 * v * v
            for w, s, v in zip(weights, nodes, values, strict=True)
        )
        spread = 12.0 * second / integral**2
    else:
        spread = math.nan
    return AveragingKernel(
        coefficients=a,
        integral=integral,
        target_integral=_dot(ell, resolution.constant_coefficients),
        centroid=centroid,
        rms_width=width,
        center=spread_center,
        backus_gilbert_spread=spread,
        negative_mass_fraction=negative / mass_abs if mass_abs > 0.0 else math.nan,
    )


def resolution_kernel_surface(
    resolution: ResolutionOperator,
    basis: _IntervalBasis,
    row_points: Sequence[float],
    column_points: Sequence[float],
) -> Matrix:
    """Return ``a(s_i, s'_j) = phi(s_i)' R M^-1 phi(s'_j)``.

    The row ``i`` is the averaging kernel of the point functional at
    ``s_i`` evaluated at the points ``s'_j``.

    Args:
        resolution: Result of :func:`resolution_operator`.
        basis: The basis.
        row_points: Points ``s_i`` of the estimate.
        column_points: Points ``s'_j`` of the truth.

    Returns:
        Matrix of kernel values.

    Raises:
        ValueError: If a point is outside ``[0, L]`` or dimensions disagree.

    References:
        [EQ-INV-LIS-006].

    """
    n = basis.dimension
    if len(resolution.matrix) != n:
        raise ValueError("resolution and basis dimensions disagree")
    phi_rows = basis.evaluation_operator(row_points).matrix
    phi_cols = basis.evaluation_operator(column_points).matrix
    left = matmul(tuple(tuple(r) for r in phi_rows), resolution.matrix)
    solved = [solve_spd(resolution.mass_matrix, tuple(row)) for row in phi_cols]
    return tuple(
        tuple(_dot(left[i], solved[j]) for j in range(len(solved)))
        for i in range(len(left))
    )


# ---------------------------------------------------------------------------
# 3. Non-sine priors
# ---------------------------------------------------------------------------


def matern_covariance(
    distance: float, kappa: float, alpha: int, amplitude: float = 1.0
) -> float:
    """Return the whole-line covariance of ``N(0, a^2 (kappa^2 - Delta)^-alpha)``.

    ``alpha = 1``: ``a^2 exp(-kappa r) / (2 kappa)``; ``alpha = 2``:
    ``a^2 (1 + kappa r) exp(-kappa r) / (4 kappa^3)`` (the Matern covariances
    with smoothness ``alpha - 1/2``).

    Args:
        distance: Separation ``r`` (sign ignored).
        kappa: Inverse correlation length, positive.
        alpha: ``1`` or ``2``.
        amplitude: Amplitude ``a`` (positive).

    Returns:
        The covariance value.

    Raises:
        TypeError: If ``alpha`` is not an integer.
        ValueError: If ``alpha`` is not 1 or 2 or a parameter is invalid.

    References:
        [EQ-INV-LIS-010], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-LINDGREN-RUE-LINDSTROM-2011].

    """
    r = abs(require_finite_real(distance, name="distance"))
    k = _positive_real(kappa, "kappa")
    a = _positive_real(amplitude, "amplitude")
    if isinstance(alpha, bool) or not isinstance(alpha, int):
        raise TypeError("alpha must be an integer")
    if alpha == 1:
        return a * a * math.exp(-k * r) / (2.0 * k)
    if alpha == 2:
        return a * a * (1.0 + k * r) * math.exp(-k * r) / (4.0 * k**3)
    raise ValueError("alpha must be 1 or 2")


def _band_cholesky(bands: list[list[float]], width: int) -> list[list[float]]:
    """Return the banded Cholesky factor ``L`` of a banded SPD matrix.

    Storage: ``bands[i][d] = Q[i, i - d]`` for ``d = 0..width``; the factor
    uses the same layout, ``L[i][d] = L[i, i - d]``.

    Args:
        bands: Lower band of ``Q`` (entries with ``i - d < 0`` ignored).
        width: Half bandwidth.

    Returns:
        The lower band of ``L``.

    Raises:
        ValueError: If a pivot is not positive and finite.

    References:
        [EQ-INV-LIS-009], [REF-GOLUB-VANLOAN-2013].

    """
    n = len(bands)
    low = [[0.0] * (width + 1) for _ in range(n)]
    for j in range(n):
        for i in range(j, min(j + width, n - 1) + 1):
            s = bands[i][i - j]
            for k in range(max(0, i - width), j):
                s -= low[i][i - k] * low[j][j - k]
            if i == j:
                if not math.isfinite(s) or s <= 0.0:
                    raise ValueError("precision matrix is not positive definite")
                low[j][0] = math.sqrt(s)
            else:
                low[i][i - j] = s / low[j][0]
    return low


def _band_forward(low: list[list[float]], width: int, rhs: Sequence[float]) -> Vector:
    """Solve ``L y = rhs`` by forward substitution on the band.

    Args:
        low: Banded factor from :func:`_band_cholesky`.
        width: Half bandwidth.
        rhs: Right-hand side.

    Returns:
        ``y``.

    References:
        [EQ-INV-LIS-009].

    """
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        s = rhs[i]
        for d in range(1, min(width, i) + 1):
            s -= low[i][d] * y[i - d]
        y[i] = s / low[i][0]
    return tuple(y)


def _band_backward(low: list[list[float]], width: int, rhs: Sequence[float]) -> Vector:
    """Solve ``L' u = rhs`` by back substitution on the band.

    Args:
        low: Banded factor from :func:`_band_cholesky`.
        width: Half bandwidth.
        rhs: Right-hand side.

    Returns:
        ``u``.

    References:
        [EQ-INV-LIS-009].

    """
    n = len(low)
    u = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for d in range(1, min(width, n - 1 - i) + 1):
            s -= low[i + d][d] * u[i + d]
        u[i] = s / low[i][0]
    return tuple(u)


@dataclass(frozen=True, slots=True)
class BoundaryEffect:
    """Measured effect of the boundary on a discrete Matern prior.

    Attributes:
        boundary: ``"dirichlet"`` or ``"neumann"``.
        stationary_variance: Whole-line variance ``C(0)``.
        boundary_variance_ratio: Discrete variance at the node nearest the
            boundary divided by ``C(0)``.
        exact_boundary_variance_ratio: The same ratio for the continuum
            covariance with the boundary condition (method of images) at that
            node's position.
        center_variance_ratio: Discrete variance at the central node over
            ``C(0)``.
        decay_distance: Largest distance of a node from its nearest
            boundary whose variance differs from ``C(0)`` by more than
            ``tolerance`` (relative); beyond it the stationary covariance
            applies to that accuracy.
        interior_reached: ``True`` if the central node is within
            ``tolerance`` of stationarity.
        tolerance: The relative tolerance used.
        center_column_error: ``max_j |C_h(c, j) - Matern(s_c - s_j)| / C(0)``
            for the central node ``c``: discretization plus boundary error
            of the covariance column away from the boundary.
        boundary_column_error: The same for the node nearest the boundary.
        discretization_error: ``max |C_h(c, j) - C_images(s_c, s_j)| / C(0)``
            over the central and the boundary column: the pure
            discretization error against the continuum covariance that
            includes the boundary condition.

    References:
        [EQ-INV-LIS-011].

    """

    boundary: str
    stationary_variance: float
    boundary_variance_ratio: float
    exact_boundary_variance_ratio: float
    center_variance_ratio: float
    decay_distance: float
    interior_reached: bool
    tolerance: float
    center_column_error: float
    boundary_column_error: float
    discretization_error: float


class WhittleMaternPrior1D:
    """Discrete Whittle-Matern prior ``N(0, a^2 (kappa^2 - Delta)^-alpha)``.

    The precision is ``Q = h L_h^alpha / a^2`` with
    ``L_h = kappa^2 I - D2`` and ``D2`` the second difference divided by
    ``h^2``. ``boundary="dirichlet"`` places ``n`` nodes at ``s_i = i h``,
    ``i = 1..n``, ``h = L / (n + 1)``, with the field zero at ``0`` and ``L``;
    ``boundary="neumann"`` uses ``n`` cells with centres
    ``s_i = (i - 1/2) h``, ``h = L / n``, and the mirror ghost cell, so the
    discrete field has zero normal derivative at ``0`` and ``L``. ``Q`` has
    half bandwidth ``alpha`` and is factored by a banded Cholesky ``Q = L L'``;
    ``u = L^-T xi`` has covariance ``Q^-1`` [EQ-INV-LIS-008], [EQ-INV-LIS-009].
    Interior covariances approximate the Matern covariance
    :func:`matern_covariance` up to ``O(h^2)`` and the boundary effect;
    both are measured by :meth:`boundary_effect` [EQ-INV-LIS-010],
    [EQ-INV-LIS-011].

    Args:
        n: Number of nodes (at least 4).
        length: Interval length ``L``.
        kappa: Inverse correlation length.
        alpha: ``1`` (exponential covariance) or ``2``.
        boundary: ``"dirichlet"`` or ``"neumann"``.
        amplitude: Amplitude ``a``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-LIS-008], [EQ-INV-LIS-009], [REF-LINDGREN-RUE-LINDSTROM-2011].

    """

    def __init__(
        self,
        n: int,
        length: float,
        kappa: float,
        *,
        alpha: int = 1,
        boundary: Boundary = "neumann",
        amplitude: float = 1.0,
    ) -> None:
        """Build and factor the banded precision.

        Args:
            n: Number of nodes (at least 4).
            length: Interval length.
            kappa: Inverse correlation length.
            alpha: ``1`` or ``2``.
            boundary: ``"dirichlet"`` or ``"neumann"``.
            amplitude: Amplitude ``a``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range.

        References:
            [EQ-INV-LIS-008], [EQ-INV-LIS-009].

        """
        self._n = _positive_int(n, "n")
        if self._n < 4:
            raise ValueError("n must be at least 4")
        self._length = _positive_real(length, "length")
        self._kappa = _positive_real(kappa, "kappa")
        self._amplitude = _positive_real(amplitude, "amplitude")
        if isinstance(alpha, bool) or not isinstance(alpha, int):
            raise TypeError("alpha must be an integer")
        if alpha not in (1, 2):
            raise ValueError("alpha must be 1 or 2")
        if boundary not in ("dirichlet", "neumann"):
            raise ValueError("boundary must be 'dirichlet' or 'neumann'")
        self._alpha = alpha
        self._boundary: Boundary = boundary
        if boundary == "dirichlet":
            self._h = self._length / (self._n + 1)
            self._nodes = tuple((i + 1) * self._h for i in range(self._n))
        else:
            self._h = self._length / self._n
            self._nodes = tuple((i + 0.5) * self._h for i in range(self._n))
        self._bands = self._precision_bands()
        self._factor = _band_cholesky(self._bands, alpha)

    @property
    def dimension(self) -> int:
        """Return the number of nodes.

        References:
            [EQ-INV-LIS-008].

        """
        return self._n

    @property
    def spacing(self) -> float:
        """Return the grid spacing ``h``.

        References:
            [EQ-INV-LIS-008].

        """
        return self._h

    @property
    def nodes(self) -> Vector:
        """Return the node locations ``s_i``.

        References:
            [EQ-INV-LIS-008].

        """
        return self._nodes

    @property
    def bandwidth(self) -> int:
        """Return the half bandwidth of the precision (``alpha``).

        References:
            [EQ-INV-LIS-008].

        """
        return self._alpha

    @property
    def boundary(self) -> str:
        """Return the boundary condition name.

        References:
            [EQ-INV-LIS-008].

        """
        return self._boundary

    def _precision_bands(self) -> list[list[float]]:
        """Return the lower band ``bands[i][d] = Q[i, i-d]``.

        Returns:
            Band storage for :func:`_band_cholesky`.

        References:
            [EQ-INV-LIS-008].

        """
        n, h = self._n, self._h
        k2 = self._kappa**2
        off = -1.0 / (h * h)
        diag = [k2 + 2.0 / (h * h)] * n
        if self._boundary == "neumann":
            diag[0] = k2 + 1.0 / (h * h)
            diag[-1] = k2 + 1.0 / (h * h)
        scale = h / self._amplitude**2
        if self._alpha == 1:
            return [[scale * diag[i], scale * off if i >= 1 else 0.0] for i in range(n)]
        bands: list[list[float]] = []
        for i in range(n):
            d0 = diag[i] ** 2
            if i > 0:
                d0 += off * off
            if i < n - 1:
                d0 += off * off
            d1 = off * (diag[i] + diag[i - 1]) if i >= 1 else 0.0
            d2 = off * off if i >= 2 else 0.0
            bands.append([scale * d0, scale * d1, scale * d2])
        return bands

    def precision_matrix(self) -> Matrix:
        """Return the dense precision ``Q`` (for small ``n``; tests, oracles).

        Returns:
            ``n x n`` matrix assembled from the band.

        References:
            [EQ-INV-LIS-008].

        """
        n, w = self._n, self._alpha
        rows = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for d in range(0, min(w, i) + 1):
                rows[i][i - d] = self._bands[i][d]
                rows[i - d][i] = self._bands[i][d]
        return tuple(tuple(r) for r in rows)

    def log_det_precision(self) -> float:
        """Return ``log det Q = 2 sum log L_ii``.

        Returns:
            The log determinant.

        References:
            [EQ-INV-LIS-009].

        """
        return 2.0 * math.fsum(math.log(row[0]) for row in self._factor)

    def solve_precision(self, rhs: Sequence[float]) -> Vector:
        """Return ``Q^-1 rhs`` by two banded triangular solves.

        Args:
            rhs: Vector of length ``n``.

        Returns:
            The solution.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LIS-009].

        """
        v = as_vector(rhs, name="rhs")
        if len(v) != self._n:
            raise ValueError(f"rhs has length {len(v)}, expected {self._n}")
        return _band_backward(
            self._factor, self._alpha, _band_forward(self._factor, self._alpha, v)
        )

    def covariance_column(self, index: int) -> Vector:
        """Return the exact discrete covariance column ``Q^-1 e_index``.

        Args:
            index: Node index in ``0..n-1``.

        Returns:
            The column.

        Raises:
            ValueError: If the index is out of range.

        References:
            [EQ-INV-LIS-009].

        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self._n:
            raise ValueError("index out of range")
        return self.solve_precision(_unit(self._n, index))

    def marginal_variances(self) -> Vector:
        """Return the exact diagonal of ``Q^-1`` (``n`` banded solves).

        Returns:
            Discrete marginal variances.

        References:
            [EQ-INV-LIS-009].

        """
        return tuple(self.covariance_column(i)[i] for i in range(self._n))

    def prior_factor(self) -> CallbackOperator:
        """Return ``P = L^-T`` (``n x n``) with ``P P' = Q^-1``.

        The operator applies banded solves only; it can be passed as
        ``prior_factor`` to the linear-Gaussian routines without forming
        any dense covariance.

        Returns:
            A square operator with ``matvec = L^-T`` and ``rmatvec = L^-1``.

        References:
            [EQ-INV-LIS-009].

        """
        low, w = self._factor, self._alpha

        def forward(z: Vector) -> Vector:
            """Return ``L^-T z``.

            References:
                [EQ-INV-LIS-009].

            """
            return _band_backward(low, w, z)

        def adjoint(v: Vector) -> Vector:
            """Return ``L^-1 v``.

            References:
                [EQ-INV-LIS-009].

            """
            return _band_forward(low, w, v)

        return CallbackOperator(
            (self._n, self._n), forward, adjoint, name="Whittle-Matern factor"
        )

    def sample(
        self, count: int, seed: int | RandomStream | None = 0
    ) -> tuple[Vector, ...]:
        """Draw samples ``u = L^-T xi``, ``xi ~ N(0, I)``.

        Args:
            count: Number of samples.
            seed: Integer seed or stream.

        Returns:
            ``count`` vectors of length ``n`` with covariance ``Q^-1``.

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count`` is not positive.

        References:
            [EQ-INV-LIS-009].

        """
        draws = _positive_int(count, "count")
        generator = resolve_stream(seed).child("whittle_matern").generator()
        out: list[Vector] = []
        for _ in range(draws):
            xi = tuple(generator.gauss(0.0, 1.0) for _ in range(self._n))
            out.append(_band_backward(self._factor, self._alpha, xi))
        return tuple(out)

    def stationary_covariance(self, distance: float) -> float:
        """Return the whole-line Matern covariance at ``distance``.

        Args:
            distance: Separation.

        Returns:
            :func:`matern_covariance` with this prior's parameters.

        References:
            [EQ-INV-LIS-010].

        """
        return matern_covariance(distance, self._kappa, self._alpha, self._amplitude)

    def continuum_covariance(self, s: float, t: float) -> float:
        """Return the continuum covariance with the boundary condition.

        Method of images: ``sum_k [C(s - t + 2kL) +- C(s + t + 2kL)]`` with
        ``+`` for Neumann and ``-`` for Dirichlet, summed until the terms
        are below ``1e-18`` of ``C(0)``.

        Args:
            s: First point in ``[0, L]``.
            t: Second point in ``[0, L]``.

        Returns:
            The covariance ``C_BC(s, t)``.

        Raises:
            ValueError: If a point is outside ``[0, L]`` or the image sum
                would need more than ``10**5`` terms.

        References:
            [EQ-INV-LIS-010].

        """
        a = require_finite_real(s, name="s")
        b = require_finite_real(t, name="t")
        length = self._length
        if not (0.0 <= a <= length and 0.0 <= b <= length):
            raise ValueError("points must lie in [0, L]")
        terms = math.ceil(42.0 / (2.0 * self._kappa * length)) + 2
        if terms > 100_000:
            raise ValueError("kappa * L too small for the image sum")
        sign = 1.0 if self._boundary == "neumann" else -1.0
        total = 0.0
        for k in range(-terms, terms + 1):
            shift = 2.0 * k * length
            total += self.stationary_covariance(a - b + shift)
            total += sign * self.stationary_covariance(a + b + shift)
        return total

    def boundary_effect(self, tolerance: float = 0.01) -> BoundaryEffect:
        """Measure the boundary effect and the discretization error.

        Args:
            tolerance: Relative tolerance on the variance profile.

        Returns:
            :class:`BoundaryEffect`.

        Raises:
            ValueError: If ``tolerance`` is not in ``(0, 1)``.

        References:
            [EQ-INV-LIS-011].

        """
        tol = _positive_real(tolerance, "tolerance")
        if tol >= 1.0:
            raise ValueError("tolerance must be below 1")
        n = self._n
        var0 = self.stationary_covariance(0.0)
        variances = self.marginal_variances()
        distance = [min(s, self._length - s) for s in self._nodes]
        violating = [
            d
            for d, v in zip(distance, variances, strict=True)
            if abs(v / var0 - 1.0) > tol
        ]
        centre = n // 2
        edge = 0
        center_col = self.covariance_column(centre)
        edge_col = self.covariance_column(edge)
        s_c, s_e = self._nodes[centre], self._nodes[edge]
        center_err = max(
            abs(center_col[j] - self.stationary_covariance(s_c - self._nodes[j]))
            for j in range(n)
        )
        edge_err = max(
            abs(edge_col[j] - self.stationary_covariance(s_e - self._nodes[j]))
            for j in range(n)
        )
        disc = max(
            max(
                abs(center_col[j] - self.continuum_covariance(s_c, self._nodes[j]))
                for j in range(n)
            ),
            max(
                abs(edge_col[j] - self.continuum_covariance(s_e, self._nodes[j]))
                for j in range(n)
            ),
        )
        return BoundaryEffect(
            boundary=self._boundary,
            stationary_variance=var0,
            boundary_variance_ratio=variances[edge] / var0,
            exact_boundary_variance_ratio=self.continuum_covariance(s_e, s_e) / var0,
            center_variance_ratio=variances[centre] / var0,
            decay_distance=max(violating) if violating else 0.0,
            interior_reached=abs(variances[centre] / var0 - 1.0) <= tol,
            tolerance=tol,
            center_column_error=center_err / var0,
            boundary_column_error=edge_err / var0,
            discretization_error=disc / var0,
        )


class CosineBasis(_IntervalBasis):
    """Orthonormal cosine basis on ``[0, L]`` (Neumann eigenfunctions).

    ``phi_0 = 1 / sqrt(L)`` and ``phi_k = sqrt(2 / L) cos(k pi s / L)``.

    ``k = 0..n-1``; the functions are the Neumann eigenfunctions of the
    Laplacian, contain the constants and are orthonormal in ``L2(0, L)``, so
    the mass matrix is the identity.

    Args:
        n: Number of functions.
        length: Interval length ``L``.

    References:
        [EQ-INV-LIS-012].

    """

    def _row(self, point: float) -> list[float]:
        """Return ``phi_k(point)`` for ``k = 0..n-1``.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            Basis values.

        References:
            [EQ-INV-LIS-012].

        """
        scale = math.sqrt(2.0 / self._length)
        omega = math.pi * point / self._length
        row = [scale * math.cos(k * omega) for k in range(self._n)]
        row[0] = 1.0 / math.sqrt(self._length)
        return row

    def mass_matrix(self) -> Matrix:
        """Return the identity.

        Returns:
            Dense ``n x n`` identity.

        References:
            [EQ-INV-LIS-012].

        """
        return tuple(
            tuple(1.0 if i == j else 0.0 for j in range(self._n))
            for i in range(self._n)
        )


class ShiftedLegendreBasis(_IntervalBasis):
    """Shifted Legendre polynomials ``P_k(2 s / L - 1)``, ``k = 0..n-1``.

    ``int_0^L P_j P_k ds = L delta_jk / (2 k + 1)``. With
    ``normalized=True`` the functions are scaled by ``sqrt((2k + 1) / L)``
    and the mass matrix is the identity; with ``normalized=False`` it is
    ``diag(L / (2k + 1))``.

    Args:
        n: Number of polynomials (degrees ``0..n-1``).
        length: Interval length ``L``.
        normalized: Whether to scale to unit ``L2`` norm.

    References:
        [EQ-INV-LIS-013].

    """

    def __init__(self, n: int, length: float, *, normalized: bool = True) -> None:
        """Validate and store the dimension, length and scaling.

        Args:
            n: Number of polynomials.
            length: Interval length.
            normalized: Whether to normalize.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``n`` or ``length`` is invalid.

        References:
            [EQ-INV-LIS-013].

        """
        super().__init__(n, length)
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        self._normalized = normalized

    def _row(self, point: float) -> list[float]:
        """Return the polynomial values at ``point`` by the three-term recurrence.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            Basis values.

        References:
            [EQ-INV-LIS-013].

        """
        t = 2.0 * point / self._length - 1.0
        values = [1.0]
        if self._n > 1:
            values.append(t)
        for k in range(1, self._n - 1):
            values.append(((2 * k + 1) * t * values[k] - k * values[k - 1]) / (k + 1))
        if self._normalized:
            values = [
                math.sqrt((2 * k + 1) / self._length) * v for k, v in enumerate(values)
            ]
        return values

    def mass_matrix(self) -> Matrix:
        """Return the exact diagonal Gram matrix.

        Returns:
            ``n x n`` identity if normalized, else ``diag(L / (2k + 1))``.

        References:
            [EQ-INV-LIS-013].

        """
        diag = [
            1.0 if self._normalized else self._length / (2 * k + 1)
            for k in range(self._n)
        ]
        return tuple(
            tuple(diag[i] if i == j else 0.0 for j in range(self._n))
            for i in range(self._n)
        )


def cosine_spectral_prior(
    n: int, length: float, kappa: float, *, s: float = 1.0, amplitude: float = 1.0
) -> DiagonalCovariance:
    """Return the coefficient covariance of the Neumann spectral prior.

    The covariance operator ``a^2 (kappa^2 - Delta_N)^-s`` has the cosine
    functions as eigenfunctions with eigenvalues
    ``a^2 (kappa^2 + (k pi / L)^2)^-s``, ``k = 0..n-1``; in the
    :class:`CosineBasis` the coefficients are independent with these
    variances, independent of ``n`` (the same prior at every truncation).

    Args:
        n: Number of modes.
        length: Interval length.
        kappa: Inverse correlation length.
        s: Smoothness exponent.
        amplitude: Amplitude ``a``.

    Returns:
        A diagonal covariance of dimension ``n``.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If a parameter is invalid.

    References:
        [EQ-INV-LIS-012], [EQ-INV-LIS-010], [REF-STUART-2010].

    """
    count = _positive_int(n, "n")
    big_l = _positive_real(length, "length")
    k2 = _positive_real(kappa, "kappa") ** 2
    exponent = _positive_real(s, "s")
    a2 = _positive_real(amplitude, "amplitude") ** 2
    return DiagonalCovariance(
        [a2 * (k2 + (k * math.pi / big_l) ** 2) ** (-exponent) for k in range(count)]
    )
