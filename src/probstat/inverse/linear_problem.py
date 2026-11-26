"""Generalized linear inverse problems: GLS, affine Tikhonov, constraints, TLS.

A :class:`LinearInverseProblem` holds the operator ``A`` (dense or
matrix-free), the data ``y``, a Gaussian noise covariance ``Gamma`` (as a
covariance factor with whitening ``W``, ``W'W = Gamma^-1``), a penalty operator
``L`` and a reference (prior centre) ``x_ref``. The regularized estimate
minimizes

``Phi(x) = 1/2 ||W (A x - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2``

[EQ-INV-LIN-001]. With ``z = x - x_ref`` it is the least-squares solution of
the stacked system ``[W A; lambda L] z = [W (y - A x_ref); 0]``, solved by
Householder QR (unique solutions only), by SVD (explicitly selected
minimum-``||x - x_ref||`` solution) or matrix-free by LSQR, LSMR or CGLS.
The legacy :func:`probstat.inverse.tikhonov` is the special case ``W = I``,
``x_ref = 0`` and gives the same answer (tested).

Further baselines: equality-constrained least squares by the nullspace method
[EQ-INV-LIN-003], non-negative least squares by the Lawson-Hanson active-set
algorithm [EQ-INV-LIN-004], box-constrained regularized least squares by
projected gradients [EQ-INV-LIN-005], and total least squares for isotropic
errors in variables [EQ-INV-LIN-006]. :func:`regularized_factorization`
factors the stacked operator once and reuses it for many data vectors
[EQ-INV-LIN-007]. :func:`linear_estimator_analysis`
returns the resolution matrix, the data influence matrix, the degrees of
freedom and the sampling covariance ``M Gamma M'`` of the fixed linear
estimator, which is NOT the Bayesian posterior covariance in general
[EQ-INV-LIN-002].

Nothing here adds jitter, clips an unconstrained solution, or silently
switches to a pseudo-inverse: a non-unique problem raises unless the
minimum-norm SVD route is requested explicitly.

Experimental (added in 0.28).

References:
    [REF-HANSEN-2008], [REF-LAWSON-HANSON-1995], [REF-GOLUB-VANLOAN-1980],
    [REF-VAN-HUFFEL-VANDEWALLE-1991], [REF-TARANTOLA-2005],
    [REF-GOLUB-VANLOAN-2013, Chapters 5-6].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import (
    BlockDiagonalCovariance,
    CountingOperator,
    DenseCovariance,
    DiagonalCovariance,
    HouseholderQR,
    IdentityOperator,
    KrylovResult,
    LinearOperator,
    Matrix,
    OperatorCounts,
    ScaledOperator,
    Vector,
    VStackOperator,
    cgls,
    householder_qr,
    lsmr,
    lsqr,
    materialize,
    require_operator,
    singular_value_decomposition,
)
from probstat.linalg.operators import ComposedOperator
from probstat.linalg.svd import _complete_orthonormal

CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance
"""Noise covariance factors accepted by this module."""

_DENSE_LIMIT = 4_000_000
_RANK_TOLERANCE = 1e-12


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


def _finite_vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Validate a finite vector of a given length.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a sequence of real numbers") from error
    if len(vector) != size:
        raise ValueError(f"{name} has length {len(vector)}, expected {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _require_lambda(value: object) -> float:
    """Validate ``lambda >= 0``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("regularization must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    return result


def _whitening(noise: CovarianceFactor | None, size: int) -> LinearOperator:
    """Return ``W`` as an operator (identity when ``noise`` is ``None``).

    References:
        [EQ-LINALG-COV-001].

    """
    if noise is None:
        return IdentityOperator(size)
    if noise.dimension != size:
        raise ValueError("noise dimension must equal the number of observations")
    return noise.whitening_operator()


@dataclass(frozen=True, slots=True)
class LinearInverseProblem:
    """A linear inverse problem ``y = A x + e``, ``e ~ N(0, Gamma)``, with a penalty.

    Use :meth:`create` to build a validated instance.

    Attributes:
        operator: ``A`` (``m x n``) as a linear operator.
        data: ``y`` (length ``m``).
        noise: Covariance factor of ``Gamma``; ``None`` means ``Gamma = I``.
        penalty: ``L`` (``p x n``); identity by default.
        reference: ``x_ref`` (length ``n``); zero by default. It is the
            centre of the penalty, distinct from the solver's start vector.

    References:
        [EQ-INV-LIN-001], [REF-TARANTOLA-2005].

    """

    operator: LinearOperator
    data: Vector
    noise: CovarianceFactor | None
    penalty: LinearOperator
    reference: Vector

    @classmethod
    def create(
        cls,
        operator: LinearOperator | Sequence[Sequence[float]],
        data: Sequence[float],
        *,
        noise: CovarianceFactor | None = None,
        penalty: LinearOperator | Sequence[Sequence[float]] | None = None,
        reference: Sequence[float] | None = None,
    ) -> LinearInverseProblem:
        """Validate and build a problem.

        Args:
            operator: ``A`` as a :class:`~probstat.linalg.LinearOperator` or a
                dense matrix.
            data: Finite ``y`` of length ``m``.
            noise: Optional covariance factor of dimension ``m``.
            penalty: Optional ``L`` with ``n`` columns (identity by default).
            reference: Optional ``x_ref`` of length ``n`` (zero by default).

        Returns:
            :class:`LinearInverseProblem`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If dimensions disagree or values are not finite.

        References:
            [EQ-INV-LIN-001].

        """
        a = require_operator(operator, name="operator")
        m, n = a.shape
        if m == 0 or n == 0:
            raise ValueError("operator must have non-empty dimensions")
        y = _finite_vector(data, m, "data")
        if noise is not None:
            if not hasattr(noise, "whiten") or not hasattr(noise, "dimension"):
                raise TypeError(
                    "noise must be a covariance factor from probstat.linalg"
                )
            if noise.dimension != m:
                raise ValueError(
                    "noise dimension must equal the number of observations"
                )
        l_operator = (
            IdentityOperator(n)
            if penalty is None
            else require_operator(penalty, name="penalty")
        )
        if l_operator.shape[1] != n:
            raise ValueError("penalty must have as many columns as the operator")
        x_ref = (
            (0.0,) * n
            if reference is None
            else _finite_vector(reference, n, "reference")
        )
        return cls(a, y, noise, l_operator, x_ref)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(m, n)``.

        References:
            [EQ-INV-LIN-001].

        """
        return self.operator.shape

    def whiten(self, residual: Sequence[float]) -> Vector:
        """Return ``W r``.

        References:
            [EQ-LINALG-COV-001].

        """
        if self.noise is None:
            return _finite_vector(residual, self.shape[0], "residual")
        return self.noise.whiten(residual)

    def whitened_operator(self) -> LinearOperator:
        """Return ``W A`` as an operator (``A`` itself when ``Gamma = I``).

        References:
            [EQ-INV-LIN-001], [EQ-LINALG-OP-002].

        """
        if self.noise is None:
            return self.operator
        return ComposedOperator([_whitening(self.noise, self.shape[0]), self.operator])

    def stacked_operator(self, regularization: float) -> LinearOperator:
        """Return ``B = [W A; lambda L]`` (``W A`` alone for ``lambda = 0``).

        References:
            [EQ-INV-LIN-001], [EQ-LINALG-OP-002].

        """
        lam = _require_lambda(regularization)
        top = self.whitened_operator()
        if lam == 0.0:
            return top
        return VStackOperator([top, ScaledOperator(self.penalty, lam)])

    def stacked_data(self, regularization: float) -> Vector:
        """Return ``b = [W (y - A x_ref); 0]`` matching :meth:`stacked_operator`.

        References:
            [EQ-INV-LIN-001].

        """
        lam = _require_lambda(regularization)
        shifted = [
            yi - ai
            for yi, ai in zip(
                self.data, self.operator.matvec(self.reference), strict=True
            )
        ]
        top = self.whiten(shifted)
        if lam == 0.0:
            return top
        return top + (0.0,) * self.penalty.shape[0]

    def misfit(self, estimate: Sequence[float]) -> float:
        """Return the whitened data misfit ``||W (A x - y)||``.

        References:
            [EQ-INV-LIN-001].

        """
        x = _finite_vector(estimate, self.shape[1], "estimate")
        residual = [
            ai - yi for ai, yi in zip(self.operator.matvec(x), self.data, strict=True)
        ]
        return _norm(self.whiten(residual))

    def penalty_norm(self, estimate: Sequence[float]) -> float:
        """Return ``||L (x - x_ref)||``.

        References:
            [EQ-INV-LIN-001].

        """
        x = _finite_vector(estimate, self.shape[1], "estimate")
        return _norm(
            self.penalty.matvec([a - b for a, b in zip(x, self.reference, strict=True)])
        )

    def objective(self, estimate: Sequence[float], regularization: float) -> float:
        """Return ``Phi(x) = 1/2 ||W (A x - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2``.

        References:
            [EQ-INV-LIN-001].

        """
        lam = _require_lambda(regularization)
        misfit = self.misfit(estimate)
        penalty = self.penalty_norm(estimate)
        return 0.5 * misfit * misfit + 0.5 * lam * lam * penalty * penalty

    def gradient(self, estimate: Sequence[float], regularization: float) -> Vector:
        """Return ``A'W'W (A x - y) + lambda^2 L'L (x - x_ref)``.

        References:
            [EQ-INV-LIN-001].

        """
        lam = _require_lambda(regularization)
        x = _finite_vector(estimate, self.shape[1], "estimate")
        residual = [
            ai - yi for ai, yi in zip(self.operator.matvec(x), self.data, strict=True)
        ]
        weighted = residual if self.noise is None else self.noise.solve(residual)
        data_part = self.operator.rmatvec(weighted)
        penalty_part = self.penalty.rmatvec(
            self.penalty.matvec([a - b for a, b in zip(x, self.reference, strict=True)])
        )
        return tuple(
            d + lam * lam * p for d, p in zip(data_part, penalty_part, strict=True)
        )


@dataclass(frozen=True, slots=True)
class LinearInverseSolution:
    """Regularized solution of a :class:`LinearInverseProblem` for one ``lambda``.

    Attributes:
        estimate: ``x_lambda``.
        regularization: ``lambda``.
        objective: ``Phi(x_lambda)`` of [EQ-INV-LIN-001].
        misfit: Whitened misfit ``||W (A x - y)||``.
        residual_norm: Raw residual ``||A x - y||`` (data units).
        penalty_norm: ``||L (x - x_ref)||``.
        stationarity: ``||grad Phi(x_lambda)||``; small at an exact solve.
        method: ``"qr"``, ``"svd"``, ``"lsqr"``, ``"lsmr"`` or ``"cgls"``.
        status: ``"unique"`` (full-rank stack), ``"minimum_norm"`` (SVD route
            on a rank-deficient stack: the solution with the smallest
            ``||x - x_ref||``), ``"iterative"`` (converged Krylov solve) or
            ``"not_converged"``.
        rank: Numerical rank of the stacked operator for the dense routes
            (``None`` for iterative routes).
        degrees_of_freedom: ``tr(H)`` of the whitened influence matrix for the
            dense routes (``None`` otherwise).
        counts: Operator applications of ``A`` (forward, transpose) by the
            solver, including diagnostics.
        krylov: The Krylov solver result for iterative routes.

    References:
        [EQ-INV-LIN-001], [EQ-INV-LIN-002].

    """

    estimate: Vector
    regularization: float
    objective: float
    misfit: float
    residual_norm: float
    penalty_norm: float
    stationarity: float
    method: str
    status: str
    rank: int | None
    degrees_of_freedom: float | None
    counts: OperatorCounts
    krylov: KrylovResult | None


_METHODS = ("qr", "svd", "lsqr", "lsmr", "cgls")


def solve_regularized(
    problem: LinearInverseProblem,
    regularization: float,
    *,
    method: str = "qr",
    rank_tolerance: float = _RANK_TOLERANCE,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    max_dense_entries: int = _DENSE_LIMIT,
) -> LinearInverseSolution:
    """Return the affine, weighted Tikhonov solution of [EQ-INV-LIN-001].

    The increment ``z = x - x_ref`` solves ``min ||B z - b||`` with
    ``B = [W A; lambda L]`` and ``b = [W (y - A x_ref); 0]``.

    - ``"qr"``: Householder QR of the materialized ``B``; raises when ``B``
      is numerically rank deficient (the solution is not unique).
    - ``"svd"``: SVD of ``B``; for a rank-deficient ``B`` returns the
      solution with the smallest ``||x - x_ref||`` (note: not the smallest
      ``||x||`` unless ``x_ref = 0``) and reports ``status="minimum_norm"``.
    - ``"lsqr"``, ``"lsmr"``, ``"cgls"``: matrix-free Krylov solves of the
      same stacked problem from ``z = 0`` (minimum-norm in ``z`` for a
      consistent rank-deficient stack); only operator actions are used.

    Args:
        problem: The inverse problem.
        regularization: ``lambda >= 0`` (``0`` gives generalized least
            squares).
        method: One of ``"qr"``, ``"svd"``, ``"lsqr"``, ``"lsmr"``, ``"cgls"``.
        rank_tolerance: Relative rank threshold for the dense routes.
        tolerance: Relative tolerance of the iterative routes.
        max_iterations: Iteration cap of the iterative routes.
        max_dense_entries: Refuse to materialize ``B`` beyond this size.

    Returns:
        :class:`LinearInverseSolution`.

    Raises:
        ValueError: For an invalid argument, a rank-deficient stack with
            ``method="qr"``, an underdetermined dense stack with
            ``method="qr"``, or a matrix too large to materialize.

    References:
        [EQ-INV-LIN-001], [EQ-INV-REG-001], [REF-HANSEN-2008, Section 2.7.1],
        [REF-PAIGE-SAUNDERS-1982].

    """
    if not isinstance(problem, LinearInverseProblem):
        raise TypeError("problem must be a LinearInverseProblem")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    lam = _require_lambda(regularization)
    counted = CountingOperator(problem.operator)
    counted_problem = LinearInverseProblem(
        counted, problem.data, problem.noise, problem.penalty, problem.reference
    )
    stacked = counted_problem.stacked_operator(lam)
    rhs = counted_problem.stacked_data(lam)
    n = problem.shape[1]
    rank: int | None = None
    dof: float | None = None
    krylov: KrylovResult | None = None
    if method in ("qr", "svd"):
        rows = stacked.shape[0]
        if rows * n > max_dense_entries:
            raise ValueError(
                "problem too large for a dense route; use lsqr, lsmr or cgls"
            )
        dense = materialize(stacked, max_entries=max_dense_entries)
        m_data = problem.shape[0]
        if method == "qr":
            if rows < n:
                raise ValueError(
                    "the stacked system is underdetermined; the solution is not "
                    "unique (use method='svd' for the minimum-norm solution)"
                )
            factor = householder_qr(dense, rank_tolerance=rank_tolerance)
            if not factor.full_rank:
                raise ValueError(
                    "the stacked operator [W A; lambda L] is rank deficient: the "
                    "solution is not unique (use method='svd' for the minimum-norm "
                    "solution or increase lambda)"
                )
            increment = factor.solve(rhs)
            rank = n
            dof = math.fsum(
                factor.q[i][j] ** 2 for i in range(m_data) for j in range(n)
            )
            status = "unique"
        else:
            svd = singular_value_decomposition(dense)
            sigma = svd.singular_values
            threshold = rank_tolerance * (sigma[0] if sigma else 0.0)
            kept = [i for i, s in enumerate(sigma) if s > threshold and s > 0.0]
            rank = len(kept)
            coefficients = {
                i: math.fsum(svd.u[r][i] * rhs[r] for r in range(rows)) / sigma[i]
                for i in kept
            }
            increment = tuple(
                math.fsum(svd.v[j][i] * coefficients[i] for i in kept) for j in range(n)
            )
            dof = math.fsum(svd.u[r][i] ** 2 for r in range(m_data) for i in kept)
            status = "unique" if rank == n else "minimum_norm"
    else:
        solver = lsqr if method == "lsqr" else lsmr
        if method == "cgls":
            krylov = cgls(
                stacked,
                rhs,
                residual_tolerance=tolerance,
                normal_tolerance=tolerance,
                max_iterations=max_iterations,
            )
        else:
            krylov = solver(
                stacked,
                rhs,
                atol=tolerance,
                btol=tolerance,
                max_iterations=max_iterations,
            )
        increment = krylov.solution
        status = "iterative" if krylov.converged else "not_converged"
    estimate = tuple(a + b for a, b in zip(increment, problem.reference, strict=True))
    fitted = counted.matvec(estimate)
    raw = [f - y for f, y in zip(fitted, problem.data, strict=True)]
    misfit = _norm(problem.whiten(raw))
    penalty_norm = problem.penalty_norm(estimate)
    weighted = raw if problem.noise is None else problem.noise.solve(raw)
    data_gradient = counted.rmatvec(weighted)
    penalty_gradient = problem.penalty.rmatvec(
        problem.penalty.matvec(
            [a - b for a, b in zip(estimate, problem.reference, strict=True)]
        )
    )
    gradient = [
        d + lam * lam * p for d, p in zip(data_gradient, penalty_gradient, strict=True)
    ]
    return LinearInverseSolution(
        estimate=estimate,
        regularization=lam,
        objective=0.5 * misfit * misfit + 0.5 * lam * lam * penalty_norm * penalty_norm,
        misfit=misfit,
        residual_norm=_norm(raw),
        penalty_norm=penalty_norm,
        stationarity=_norm(gradient),
        method=method,
        status=status,
        rank=rank,
        degrees_of_freedom=dof,
        counts=counted.counts,
        krylov=krylov,
    )


@dataclass(frozen=True, slots=True)
class RegularizedFactorization:
    """Reusable QR factorization of ``B = [W A; lambda L]`` for many data vectors.

    The stacked operator of [EQ-INV-LIN-001] does not depend on ``y`` or
    ``x_ref``, so one Householder QR serves every right-hand side
    ``b = [W (y - A x_ref); 0]`` [EQ-INV-LIN-007]. Each :meth:`solve` costs
    one forward application of ``A`` (for ``A x_ref``), one whitening and an
    ``O(n (m + p))`` triangular solve, instead of a new factorization.

    Attributes:
        operator: ``A``.
        noise: Covariance factor of ``Gamma`` (``None`` for ``Gamma = I``).
        penalty_rows: Number of rows ``p`` of ``L``.
        regularization: ``lambda``.
        shape: ``(m, n)``.
        factor_counts: Operator applications of ``A`` spent on materializing
            ``B`` (once).
        factor: The Householder QR of ``B``.

    References:
        [EQ-INV-LIN-001], [EQ-INV-LIN-007], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """

    operator: LinearOperator
    noise: CovarianceFactor | None
    penalty_rows: int
    regularization: float
    shape: tuple[int, int]
    factor_counts: OperatorCounts
    factor: HouseholderQR

    def solve(
        self, data: Sequence[float], reference: Sequence[float] | None = None
    ) -> Vector:
        """Return ``x_lambda`` for data ``y`` and reference ``x_ref``.

        Args:
            data: ``y`` (length ``m``).
            reference: ``x_ref`` (length ``n``; zero by default).

        Returns:
            The regularized estimate, identical (to rounding) to
            :func:`solve_regularized` with ``method="qr"``.

        Raises:
            TypeError: If a vector is not numeric.
            ValueError: If a length is wrong or a value is not finite.

        References:
            [EQ-INV-LIN-007].

        """
        m, n = self.shape
        y = _finite_vector(data, m, "data")
        x_ref = (
            (0.0,) * n
            if reference is None
            else _finite_vector(reference, n, "reference")
        )
        shifted = [
            yi - ai for yi, ai in zip(y, self.operator.matvec(x_ref), strict=True)
        ]
        top = shifted if self.noise is None else list(self.noise.whiten(shifted))
        rhs = tuple(top) + (
            (0.0,) * self.penalty_rows if self.regularization > 0.0 else ()
        )
        increment = self.factor.solve(rhs)
        return tuple(a + b for a, b in zip(increment, x_ref, strict=True))

    def solve_many(
        self,
        datasets: Sequence[Sequence[float]],
        reference: Sequence[float] | None = None,
    ) -> tuple[Vector, ...]:
        """Return :meth:`solve` for every data vector, sharing ``x_ref``.

        Args:
            datasets: Data vectors ``y_1, ..., y_K``.
            reference: Common ``x_ref`` (zero by default).

        Returns:
            One estimate per data vector.

        Raises:
            ValueError: If a data vector is invalid.

        References:
            [EQ-INV-LIN-007].

        """
        return tuple(self.solve(y, reference) for y in datasets)


def regularized_factorization(
    problem: LinearInverseProblem,
    regularization: float,
    *,
    rank_tolerance: float = _RANK_TOLERANCE,
    max_dense_entries: int = _DENSE_LIMIT,
) -> RegularizedFactorization:
    """Factor ``B = [W A; lambda L]`` once for repeated solves [EQ-INV-LIN-007].

    Only ``problem.operator``, ``problem.noise`` and ``problem.penalty`` are
    used; the data and reference of ``problem`` are not part of the factor.

    Args:
        problem: Supplies ``A``, ``Gamma`` and ``L``.
        regularization: ``lambda >= 0``.
        rank_tolerance: Relative rank threshold of the QR.
        max_dense_entries: Refuse to materialize ``B`` beyond this size.

    Returns:
        :class:`RegularizedFactorization`.

    Raises:
        TypeError: If ``problem`` is not a :class:`LinearInverseProblem`.
        ValueError: If ``B`` is underdetermined, rank deficient (the solution
            is not unique) or too large.

    References:
        [EQ-INV-LIN-001], [EQ-INV-LIN-007], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    if not isinstance(problem, LinearInverseProblem):
        raise TypeError("problem must be a LinearInverseProblem")
    lam = _require_lambda(regularization)
    counted = CountingOperator(problem.operator)
    counted_problem = LinearInverseProblem(
        counted, problem.data, problem.noise, problem.penalty, problem.reference
    )
    stacked = counted_problem.stacked_operator(lam)
    rows = stacked.shape[0]
    n = problem.shape[1]
    if rows * n > max_dense_entries:
        raise ValueError("problem too large for a dense factorization")
    if rows < n:
        raise ValueError(
            "the stacked system is underdetermined; the solution is not unique"
        )
    dense = materialize(stacked, max_entries=max_dense_entries)
    factor = householder_qr(dense, rank_tolerance=rank_tolerance)
    if not factor.full_rank:
        raise ValueError(
            "the stacked operator [W A; lambda L] is rank deficient: the solution "
            "is not unique (increase lambda or use solve_regularized(method='svd'))"
        )
    return RegularizedFactorization(
        operator=problem.operator,
        noise=problem.noise,
        penalty_rows=problem.penalty.shape[0],
        regularization=lam,
        shape=problem.shape,
        factor_counts=counted.counts,
        factor=factor,
    )


@dataclass(frozen=True, slots=True)
class LinearEstimatorAnalysis:
    """Linear-estimator diagnostics of affine Tikhonov for one ``lambda``.

    The estimate is ``x = x_ref + M (y - A x_ref)`` with gain
    ``M = (A'Gamma^-1 A + lambda^2 L'L)^-1 A'Gamma^-1`` [EQ-INV-LIN-002].

    Attributes:
        gain: ``M`` (``n x m``).
        resolution: ``R = M A`` (``n x n``); row ``i`` is the averaging
            kernel of ``x_i``, column ``j`` the point-spread function of a
            unit perturbation of ``x_j``.
        influence: Whitened data influence matrix ``H = W A M W^-1``
            (``m x m``), a symmetric projector-like matrix.
        degrees_of_freedom: ``tr(H) = tr(R)``.
        leverages: ``diag(H)``.
        estimator_covariance: ``M Gamma M'``, the sampling covariance of the
            estimator under repeated noise for a FIXED truth.
        bayesian_covariance: ``(A'Gamma^-1 A + lambda^2 L'L)^-1``; this is
            the posterior covariance only if ``lambda^2 L'L`` is the precision
            of a proper Gaussian prior centred at ``x_ref``.
        regularization: ``lambda``.

    References:
        [EQ-INV-LIN-002], [REF-HANSEN-2008], [REF-TARANTOLA-2005].

    """

    gain: Matrix
    resolution: Matrix
    influence: Matrix
    degrees_of_freedom: float
    leverages: Vector
    estimator_covariance: Matrix
    bayesian_covariance: Matrix
    regularization: float

    def bias(self, truth: Sequence[float], reference: Sequence[float]) -> Vector:
        """Return the expected error ``E[x] - x_true = (R - I)(x_true - x_ref)``.

        Only computable for a synthetic problem with a known truth.

        Args:
            truth: ``x_true``.
            reference: ``x_ref`` used by the estimator.

        Returns:
            The bias vector.

        References:
            [EQ-INV-LIN-002].

        """
        n = len(self.resolution)
        t = _finite_vector(truth, n, "truth")
        r = _finite_vector(reference, n, "reference")
        delta = [a - b for a, b in zip(t, r, strict=True)]
        return tuple(
            math.fsum(self.resolution[i][j] * delta[j] for j in range(n)) - delta[i]
            for i in range(n)
        )

    def risk(self, truth: Sequence[float], reference: Sequence[float]) -> float:
        """Return the mean squared error ``tr(M Gamma M') + ||bias||^2``.

        References:
            [EQ-INV-LIN-002].

        """
        b = self.bias(truth, reference)
        n = len(self.resolution)
        return math.fsum(self.estimator_covariance[i][i] for i in range(n)) + _dot(b, b)


def _upper_inverse(upper: Matrix) -> list[list[float]]:
    """Return ``R^-1`` of an upper-triangular matrix by back substitution.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    n = len(upper)
    inverse = [[0.0] * n for _ in range(n)]
    for column in range(n):
        for i in range(n - 1, -1, -1):
            total = (1.0 if i == column else 0.0) - math.fsum(
                upper[i][k] * inverse[k][column] for k in range(i + 1, n)
            )
            inverse[i][column] = total / upper[i][i]
    return inverse


def linear_estimator_analysis(
    problem: LinearInverseProblem,
    regularization: float,
    *,
    rank_tolerance: float = _RANK_TOLERANCE,
    max_dense_entries: int = _DENSE_LIMIT,
) -> LinearEstimatorAnalysis:
    """Return the resolution, influence and covariance of the affine Tikhonov estimator.

    From the thin QR factorization ``[W A; lambda L] = [Q_1; Q_2] R`` (no
    explicit normal-equation inverse): ``M = R^-1 Q_1'W``,
    ``R_res = M A``, ``H = Q_1 Q_1'``, ``M Gamma M' = R^-1 Q_1'Q_1 R^-T`` and
    ``(B'B)^-1 = R^-1 R^-T`` [EQ-INV-LIN-002]. Dense and intended for
    moderate ``n``.

    Args:
        problem: The inverse problem.
        regularization: ``lambda >= 0``.
        rank_tolerance: Relative rank threshold of the QR factorization.
        max_dense_entries: Size guard.

    Returns:
        :class:`LinearEstimatorAnalysis`.

    Raises:
        ValueError: If the stacked operator is rank deficient or too large.

    References:
        [EQ-INV-LIN-002], [REF-HANSEN-2008], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    lam = _require_lambda(regularization)
    m, n = problem.shape
    stacked = problem.stacked_operator(lam)
    if stacked.shape[0] * n > max_dense_entries or stacked.shape[0] < n:
        raise ValueError(
            "stacked operator too large or underdetermined for the dense analysis"
        )
    factor = householder_qr(
        materialize(stacked, max_entries=max_dense_entries),
        rank_tolerance=rank_tolerance,
    )
    if not factor.full_rank:
        raise ValueError(
            "the stacked operator is rank deficient; the estimator is not unique"
        )
    r_inv = _upper_inverse(factor.r)
    q1 = [factor.q[i] for i in range(m)]
    # M_w = R^-1 Q_1' maps whitened data to the estimate; M = M_w W.
    gain_whitened = [
        [math.fsum(r_inv[i][k] * q1[r][k] for k in range(n)) for r in range(m)]
        for i in range(n)
    ]
    if problem.noise is None:
        gain = gain_whitened
    else:
        w_t = problem.noise.whiten_transpose
        # Row i of M = (W' row_i(M_w)')'.
        gain = [list(w_t(row)) for row in gain_whitened]
    columns_a = materialize(problem.operator, max_entries=max_dense_entries)
    resolution = tuple(
        tuple(
            math.fsum(gain[i][r] * columns_a[r][j] for r in range(m)) for j in range(n)
        )
        for i in range(n)
    )
    influence = tuple(
        tuple(math.fsum(q1[a][k] * q1[b][k] for k in range(n)) for b in range(m))
        for a in range(m)
    )
    q1tq1 = [
        [math.fsum(q1[r][a] * q1[r][b] for r in range(m)) for b in range(n)]
        for a in range(n)
    ]
    left = [
        [math.fsum(r_inv[i][k] * q1tq1[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    estimator_covariance = tuple(
        tuple(math.fsum(left[i][k] * r_inv[j][k] for k in range(n)) for j in range(n))
        for i in range(n)
    )
    bayesian = tuple(
        tuple(math.fsum(r_inv[i][k] * r_inv[j][k] for k in range(n)) for j in range(n))
        for i in range(n)
    )
    dof = math.fsum(influence[i][i] for i in range(m))
    return LinearEstimatorAnalysis(
        gain=tuple(tuple(row) for row in gain),
        resolution=resolution,
        influence=influence,
        degrees_of_freedom=dof,
        leverages=tuple(influence[i][i] for i in range(m)),
        estimator_covariance=estimator_covariance,
        bayesian_covariance=bayesian,
        regularization=lam,
    )


@dataclass(frozen=True, slots=True)
class ConstrainedSolution:
    """Solution of a constrained least-squares problem.

    Attributes:
        estimate: ``x``.
        objective: ``1/2 ||W (A x - y)||^2 (+ penalty when used)``.
        misfit: ``||W (A x - y)||``.
        status: ``"optimal"``, ``"max_iterations"`` or ``"stagnation"``.
        converged: Whether the optimality test passed.
        iterations: Iterations of the active-set or projected-gradient loop
            (``0`` for direct methods).
        optimality: The KKT residual used for the test: the projected-gradient
            norm, or for NNLS ``max(max_(free) |w_i|, max_(active) w_i^+)``,
            or for equality constraints the reduced gradient norm.
        multipliers: Lagrange multipliers (equality constraints) or the dual
            vector ``w = -grad`` (NNLS, bounds); see the function docstring
            for the sign convention.
        active: Indices of active inequality constraints (bound at a limit).
        constraint_residual: ``||C x - d||`` for equality constraints, else 0.

    References:
        [EQ-INV-LIN-003], [EQ-INV-LIN-004], [EQ-INV-LIN-005],
        [REF-LAWSON-HANSON-1995].

    """

    estimate: Vector
    objective: float
    misfit: float
    status: str
    converged: bool
    iterations: int
    optimality: float
    multipliers: Vector
    active: tuple[int, ...]
    constraint_residual: float


def _dense_whitened(
    operator: LinearOperator | Sequence[Sequence[float]],
    data: Sequence[float],
    noise: CovarianceFactor | None,
    max_entries: int,
) -> tuple[list[list[float]], list[float], int, int]:
    """Return the materialized ``W A`` and ``W y``.

    References:
        [EQ-LINALG-COV-001].

    """
    a = require_operator(operator)
    m, n = a.shape
    y = _finite_vector(data, m, "data")
    dense = materialize(a, max_entries=max_entries)
    if noise is None:
        return [list(row) for row in dense], list(y), m, n
    if noise.dimension != m:
        raise ValueError("noise dimension must equal the number of observations")
    columns = [noise.whiten([dense[i][j] for i in range(m)]) for j in range(n)]
    wa = [[columns[j][i] for j in range(n)] for i in range(m)]
    return wa, list(noise.whiten(y)), m, n


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Return a dense matrix-vector product.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return [
        math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix
    ]


def _rmatvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Return a dense transpose-vector product.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    columns = len(matrix[0])
    return [
        math.fsum(matrix[i][j] * vector[i] for i in range(len(matrix)))
        for j in range(columns)
    ]


def equality_constrained_least_squares(
    operator: LinearOperator | Sequence[Sequence[float]],
    data: Sequence[float],
    constraints: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    noise: CovarianceFactor | None = None,
    feasibility_tolerance: float = 1e-10,
    rank_tolerance: float = _RANK_TOLERANCE,
    max_dense_entries: int = _DENSE_LIMIT,
) -> ConstrainedSolution:
    """Solve ``min ||W (A x - y)||`` subject to ``C x = d`` exactly (nullspace method).

    With the SVD ``C = U_C S V_C'`` of rank ``r``, the minimum-norm particular
    solution is ``x_p = C^+ d`` and ``Z`` spans ``null(C)`` (the completed
    columns of ``V_C``). Then ``x = x_p + Z z`` with ``z`` the least-squares
    solution of ``W A Z z = W (y - A x_p)`` by QR [EQ-INV-LIN-003]. The
    multipliers ``mu`` solve ``C'mu = A'W'W (y - A x)`` in the least-squares
    sense (stationarity of the Lagrangian ``1/2||W(Ax-y)||^2 - mu'(Cx-d)``).
    Redundant (linearly dependent but consistent) constraints are accepted;
    inconsistent constraints raise. No penalty is used in place of the
    equality.

    Args:
        operator: ``A`` (dense or operator; materialized).
        data: ``y``.
        constraints: ``C`` (``p x n``).
        targets: ``d`` (length ``p``).
        noise: Optional covariance factor.
        feasibility_tolerance: Relative tolerance on ``||C x_p - d||``.
        rank_tolerance: Relative rank threshold for ``C`` and ``W A Z``.
        max_dense_entries: Size guard.

    Returns:
        :class:`ConstrainedSolution` with ``multipliers = mu``.

    Raises:
        ValueError: If the constraints are inconsistent, the reduced problem
            is rank deficient (non-unique solution), or shapes disagree.

    References:
        [EQ-INV-LIN-003], [REF-LAWSON-HANSON-1995],
        [REF-GOLUB-VANLOAN-2013, Section 6.2].

    """
    wa, wy, m, n = _dense_whitened(operator, data, noise, max_dense_entries)
    c_rows = [list(_finite_vector(row, n, "constraint row")) for row in constraints]
    if not c_rows:
        raise ValueError("constraints must be non-empty")
    d = list(_finite_vector(targets, len(c_rows), "targets"))
    svd = singular_value_decomposition(tuple(tuple(row) for row in c_rows))
    sigma = svd.singular_values
    threshold = rank_tolerance * sigma[0] if sigma and sigma[0] > 0.0 else 0.0
    kept = [i for i, s in enumerate(sigma) if s > threshold]
    p = len(c_rows)
    coefficients = [
        math.fsum(svd.u[r][i] * d[r] for r in range(p)) / sigma[i] for i in kept
    ]
    x_p = [
        math.fsum(svd.v[j][i] * c for i, c in zip(kept, coefficients, strict=True))
        for j in range(n)
    ]
    c_norm = sigma[0] if sigma else 0.0
    gap = _norm([a - b for a, b in zip(_matvec(c_rows, x_p), d, strict=True)])
    if gap > feasibility_tolerance * (_norm(d) + c_norm * _norm(x_p) + 1e-300):
        raise ValueError(
            f"the equality constraints are inconsistent (||C x_p - d|| = {gap:.3e})"
        )
    v_columns = [[svd.v[j][i] for j in range(n)] for i in kept]
    rank = len(kept)
    columns = v_columns + [[0.0] * n for _ in range(n - rank)]
    _complete_orthonormal(columns, list(range(rank, n)), n)
    z_basis = columns[rank:]
    if n - rank == 0:
        x = x_p
    else:
        waz = [
            [math.fsum(wa[i][j] * basis[j] for j in range(n)) for basis in z_basis]
            for i in range(m)
        ]
        rhs = [a - b for a, b in zip(wy, _matvec(wa, x_p), strict=True)]
        if m < len(z_basis):
            raise ValueError(
                "the reduced problem is underdetermined; the solution is not unique"
            )
        factor = householder_qr(
            tuple(tuple(row) for row in waz), rank_tolerance=rank_tolerance
        )
        if not factor.full_rank:
            raise ValueError(
                "the reduced problem W A Z is rank deficient; "
                "the solution is not unique"
            )
        z = factor.solve(tuple(rhs))
        x = [
            x_p[j] + math.fsum(z[k] * z_basis[k][j] for k in range(len(z_basis)))
            for j in range(n)
        ]
    residual_w = [a - b for a, b in zip(_matvec(wa, x), wy, strict=True)]
    gradient = _rmatvec(wa, residual_w)  # A'W'W (A x - y)
    # Stationarity: grad = C'mu  ->  least squares for mu via SVD of C'.
    mu_coefficients = {
        i: math.fsum(svd.v[j][i] * gradient[j] for j in range(n)) / sigma[i]
        for i in kept
    }
    mu = [math.fsum(svd.u[r][i] * mu_coefficients[i] for i in kept) for r in range(p)]
    reduced = _norm(
        [g - c for g, c in zip(gradient, _rmatvec(c_rows, mu), strict=True)]
    )
    misfit = _norm(residual_w)
    return ConstrainedSolution(
        estimate=tuple(x),
        objective=0.5 * misfit * misfit,
        misfit=misfit,
        status="optimal",
        converged=True,
        iterations=0,
        optimality=reduced,
        multipliers=tuple(mu),
        active=tuple(range(p)),
        constraint_residual=_norm(
            [a - b for a, b in zip(_matvec(c_rows, x), d, strict=True)]
        ),
    )


def nonnegative_least_squares(
    operator: LinearOperator | Sequence[Sequence[float]],
    data: Sequence[float],
    *,
    noise: CovarianceFactor | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    max_dense_entries: int = _DENSE_LIMIT,
) -> ConstrainedSolution:
    """Solve ``min ||W (A x - y)||`` subject to ``x >= 0`` by Lawson-Hanson NNLS.

    The active-set algorithm moves the index with the largest positive dual
    ``w = (W A)'(W y - W A x)`` into the passive set ``P``, solves the
    unconstrained least squares on ``P`` and, when that leaves the feasible
    region, steps back along the segment to the first variable that hits
    zero [EQ-INV-LIN-004]. KKT conditions at the solution: ``x >= 0``,
    ``w_i <= 0`` where ``x_i = 0`` and ``w_i = 0`` where ``x_i > 0``. The
    reported ``optimality`` is ``max(max_P |w_i|, max_(not P) max(w_i, 0))``
    relative to ``||(W A)'W y||``. Clipping an unconstrained solution is NOT
    this optimum in general.

    Args:
        operator: ``A`` (materialized).
        data: ``y``.
        noise: Optional covariance factor.
        tolerance: Relative dual tolerance.
        max_iterations: Outer-iteration cap (default ``3 n``).
        max_dense_entries: Size guard.

    Returns:
        :class:`ConstrainedSolution` with ``multipliers = w`` and ``active``
        the indices with ``x_i = 0``.

    References:
        [EQ-INV-LIN-004], [REF-LAWSON-HANSON-1995, Chapter 23].

    """
    wa, wy, m, n = _dense_whitened(operator, data, noise, max_dense_entries)
    limit = 3 * n if max_iterations is None else int(max_iterations)
    if limit < 1:
        raise ValueError("max_iterations must be positive")
    x = [0.0] * n
    passive: list[int] = []
    scale = _norm(_rmatvec(wa, wy)) or 1.0

    def dual(point: Sequence[float]) -> list[float]:
        """Return ``w = (W A)'(W y - W A x)``.

        References:
            [EQ-INV-LIN-004].

        """
        residual = [b - a for a, b in zip(_matvec(wa, point), wy, strict=True)]
        return _rmatvec(wa, residual)

    def solve_passive(indices: list[int]) -> list[float] | None:
        """Least squares restricted to the passive columns (``None`` if rank deficient).

        References:
            [EQ-INV-LIN-004], [EQ-LINALG-009].

        """
        if len(indices) > m:
            return None
        sub = tuple(tuple(wa[i][j] for j in indices) for i in range(m))
        factor = householder_qr(sub, rank_tolerance=_RANK_TOLERANCE)
        if not factor.full_rank:
            return None
        return list(factor.solve(tuple(wy)))

    status = "max_iterations"
    iterations = 0
    w = dual(x)
    for outer in range(1, limit + 1):
        iterations = outer
        candidates = [
            j for j in range(n) if j not in passive and w[j] > tolerance * scale
        ]
        if not candidates:
            status = "optimal"
            break
        best = max(candidates, key=lambda j: (w[j], -j))
        passive.append(best)
        inner_guard = 0
        while True:
            inner_guard += 1
            if inner_guard > 3 * n + 3:
                status = "stagnation"
                break
            values = solve_passive(passive)
            if values is None:
                passive.remove(best)
                status = "stagnation"
                break
            s = [0.0] * n
            for index, value in zip(passive, values, strict=True):
                s[index] = value
            if all(s[j] > 0.0 for j in passive):
                x = s
                break
            ratios = [
                x[j] / (x[j] - s[j])
                for j in passive
                if s[j] <= 0.0 and x[j] - s[j] > 0.0
            ]
            alpha = min(ratios) if ratios else 0.0
            x = [xi + alpha * (si - xi) for xi, si in zip(x, s, strict=True)]
            passive = [
                j
                for j in passive
                if x[j] > tolerance * max(1.0, max(abs(v) for v in x))
            ]
            for j in range(n):
                if j not in passive:
                    x[j] = 0.0
        if status == "stagnation":
            break
        w = dual(x)
    w = dual(x)
    free_violation = max((abs(w[j]) for j in passive), default=0.0)
    bound_violation = max(
        (max(w[j], 0.0) for j in range(n) if j not in passive), default=0.0
    )
    optimality = max(free_violation, bound_violation) / scale
    converged = status == "optimal" and optimality <= max(tolerance, 1e-8) * 10
    misfit = _norm([a - b for a, b in zip(_matvec(wa, x), wy, strict=True)])
    return ConstrainedSolution(
        estimate=tuple(x),
        objective=0.5 * misfit * misfit,
        misfit=misfit,
        status=status,
        converged=converged,
        iterations=iterations,
        optimality=optimality,
        multipliers=tuple(w),
        active=tuple(j for j in range(n) if j not in passive),
        constraint_residual=0.0,
    )


def bounded_least_squares(
    problem: LinearInverseProblem,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    regularization: float = 0.0,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 5000,
) -> ConstrainedSolution:
    """Minimize ``Phi(x)`` of [EQ-INV-LIN-001] on the box ``l <= x <= u``.

    ``x_(k+1) = P(x_k - t_k grad Phi(x_k))`` with ``P`` the projection on the
    box and ``t_k`` chosen by backtracking until the sufficient-decrease
    condition ``Phi(x+) <= Phi(x) + grad'(x+ - x) + ||x+ - x||^2 / (2 t)``
    holds [EQ-INV-LIN-005]; the step grows by ``1.5`` after an accepted step.
    Matrix-free (two operator applications per trial). Optimality is the
    projected-gradient norm ``||x - P(x - grad Phi(x))||`` relative to its
    value at the start; the multipliers returned are ``-grad Phi``
    (non-positive at an active lower bound, non-negative at an active upper
    bound, zero at a free variable). Bounds and
    the penalty centre ``x_ref`` are in the same physical coordinates.

    Args:
        problem: The inverse problem.
        lower: Lower bounds (use ``-inf`` for none).
        upper: Upper bounds (use ``inf`` for none), ``lower <= upper``.
        regularization: ``lambda >= 0``.
        initial: Start (projected onto the box); ``x_ref`` projected by
            default.
        tolerance: Relative projected-gradient tolerance.
        max_iterations: Iteration cap.

    Returns:
        :class:`ConstrainedSolution`.

    Raises:
        ValueError: If the bounds are inconsistent or not finite where
            required.

    References:
        [EQ-INV-LIN-005], [REF-NOCEDAL-WRIGHT-2006, Section 16.7],
        [REF-BECK-TEBOULLE-2009].

    """
    n = problem.shape[1]
    lam = _require_lambda(regularization)
    lo = [float(v) for v in lower]
    hi = [float(v) for v in upper]
    if len(lo) != n or len(hi) != n:
        raise ValueError("bounds must have length n")
    if any(
        math.isnan(a) or math.isnan(b) or a > b for a, b in zip(lo, hi, strict=True)
    ):
        raise ValueError("bounds must satisfy lower <= upper (and not be NaN)")
    if any(a == math.inf or b == -math.inf for a, b in zip(lo, hi, strict=True)):
        raise ValueError("bounds are infeasible")

    def project(point: Sequence[float]) -> list[float]:
        """Project onto the box.

        References:
            [EQ-INV-LIN-005].

        """
        return [min(max(v, a), b) for v, a, b in zip(point, lo, hi, strict=True)]

    start = (
        problem.reference if initial is None else _finite_vector(initial, n, "initial")
    )
    x = project(start)
    value = problem.objective(x, lam)
    grad = list(problem.gradient(x, lam))

    def projected_gradient(point: list[float], gradient: list[float]) -> float:
        """Return ``||x - P(x - g)||``.

        References:
            [EQ-INV-LIN-005].

        """
        moved = project([p - g for p, g in zip(point, gradient, strict=True)])
        return _norm([p - q for p, q in zip(point, moved, strict=True)])

    initial_measure = projected_gradient(x, grad) or 1.0
    step = 1.0 / max(_norm(grad), 1.0)
    status = "max_iterations"
    iterations = 0
    measure = initial_measure
    for k in range(1, max_iterations + 1):
        iterations = k
        accepted = False
        for _ in range(60):
            trial = project([p - step * g for p, g in zip(x, grad, strict=True)])
            delta = [a - b for a, b in zip(trial, x, strict=True)]
            trial_value = problem.objective(trial, lam)
            bound = value + _dot(grad, delta) + _dot(delta, delta) / (2.0 * step)
            if trial_value <= bound + 1e-15 * abs(value):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "stagnation"
            break
        x = trial
        value = trial_value
        grad = list(problem.gradient(x, lam))
        measure = projected_gradient(x, grad)
        if measure <= tolerance * initial_measure:
            status = "optimal"
            break
        step *= 1.5
    misfit = problem.misfit(x)
    active = tuple(j for j in range(n) if x[j] == lo[j] or x[j] == hi[j])
    return ConstrainedSolution(
        estimate=tuple(x),
        objective=value,
        misfit=misfit,
        status=status,
        converged=status == "optimal",
        iterations=iterations,
        optimality=measure / initial_measure,
        multipliers=tuple(-g for g in grad),
        active=active,
        constraint_residual=0.0,
    )


@dataclass(frozen=True, slots=True)
class TotalLeastSquaresSolution:
    """Total least-squares solution for isotropic errors in ``A`` and ``y``.

    Attributes:
        estimate: ``x_TLS = -v_(1:n) / v_(n+1)``.
        smallest_singular_value: ``sigma_(n+1)([A, y])``; the Frobenius norm
            of the smallest correction ``[dA, dy]``.
        operator_smallest_singular_value: ``sigma_n(A)``.
        gap: ``sigma_n(A) - sigma_(n+1)([A, y])``; positive gap is the
            Golub-Van Loan condition for a unique TLS solution.
        correction_norm: Equal to ``smallest_singular_value``.

    References:
        [EQ-INV-LIN-006], [REF-GOLUB-VANLOAN-1980].

    """

    estimate: Vector
    smallest_singular_value: float
    operator_smallest_singular_value: float
    gap: float
    correction_norm: float


def total_least_squares(
    matrix: Sequence[Sequence[float]],
    data: Sequence[float],
    *,
    gap_tolerance: float = 1e-12,
) -> TotalLeastSquaresSolution:
    """Return the total least-squares solution from the SVD of ``[A, y]``.

    Minimizes ``||[dA, dy]||_F`` subject to ``(A + dA) x = y + dy``. With the
    right singular vector ``v`` of the smallest singular value of ``[A, y]``,
    ``x = -v_(1:n) / v_(n+1)`` [EQ-INV-LIN-006]. Assumption: errors in ``A``
    and ``y`` are independent, zero-mean and of equal variance in the given
    units; rescale columns first otherwise. TLS is not uniformly better than
    ordinary least squares and is more sensitive to noise when the gap is
    small.

    Args:
        matrix: ``A`` (``m x n`` with ``m >= n + 1``).
        data: ``y``.
        gap_tolerance: Relative threshold on ``sigma_n(A) - sigma_(n+1)``.

    Returns:
        :class:`TotalLeastSquaresSolution`.

    Raises:
        ValueError: If ``m < n + 1``, the gap condition fails (no unique TLS
            solution) or ``v_(n+1) = 0`` (nongeneric problem).

    References:
        [EQ-INV-LIN-006], [REF-GOLUB-VANLOAN-1980],
        [REF-VAN-HUFFEL-VANDEWALLE-1991].

    """
    a = [list(row) for row in matrix]
    if not a or not a[0]:
        raise ValueError("matrix must be non-empty")
    m, n = len(a), len(a[0])
    y = _finite_vector(data, m, "data")
    if m < n + 1:
        raise ValueError("total least squares needs m >= n + 1")
    augmented = tuple(
        tuple(_finite_vector(row, n, "row")) + (yi,)
        for row, yi in zip(a, y, strict=True)
    )
    svd = singular_value_decomposition(augmented)
    sigma_aug = svd.singular_values
    sigma_a = singular_value_decomposition(
        tuple(tuple(row) for row in a)
    ).singular_values
    smallest = sigma_aug[n]
    gap = sigma_a[n - 1] - smallest
    if gap <= gap_tolerance * max(sigma_a[0], 1e-300):
        raise ValueError(
            "sigma_n(A) does not exceed sigma_(n+1)([A, y]); "
            "the TLS solution is not unique"
        )
    v = [svd.v[j][n] for j in range(n + 1)]
    if v[n] == 0.0:
        raise ValueError("nongeneric total least-squares problem (v_(n+1) = 0)")
    return TotalLeastSquaresSolution(
        estimate=tuple(-v[j] / v[n] for j in range(n)),
        smallest_singular_value=smallest,
        operator_smallest_singular_value=sigma_a[n - 1],
        gap=gap,
        correction_norm=smallest,
    )
