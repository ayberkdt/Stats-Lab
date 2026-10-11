"""Closed-form and quadrature Wasserstein distances and barycenters.

* Gaussian laws: ``W_2^2 = |m0 - m1|^2 + B(S0, S1)^2`` with the Bures metric
  ``B^2 = tr S0 + tr S1 - 2 tr (S0^(1/2) S1 S0^(1/2))^(1/2)``; matrix square
  roots use the symmetric Jacobi eigendecomposition of :mod:`probstat.linalg`.
* One-dimensional laws: ``W_p^p = int_0^1 |Q0(u) - Q1(u)|^p du`` for any
  objects with ``quantile`` (and optionally ``isf``), by Simpson quadrature
  after the substitution ``u = Phi(z)``.
* Barycenters: the ``W_2`` barycenter of 1-D laws averages quantile
  functions; the barycenter of Gaussians solves the fixed-point equation of
  Alvarez-Esteban et al. (2016).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    matmul,
    symmetric_eigen,
    symmetrize,
)
from probstat.numerics import tanh_sinh

from ._quantile_quadrature import quantile_at, quantile_expectation

_PSD_TOLERANCE = 1e-10
_WEIGHT_TOLERANCE = 1e-9


def _symmetric(matrix: Iterable[Iterable[object]], *, name: str) -> Matrix:
    """Validate a symmetric matrix and return its symmetrized copy.

    Args:
        matrix: Candidate square matrix.
        name: Argument name.

    Returns:
        ``(A + A')/2`` after a relative symmetry check of ``1e-10``.

    Raises:
        ValueError: If the matrix is not square or not symmetric.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    return symmetrize(matrix, name=name, relative_tolerance=1e-10)


def _psd_power(matrix: Matrix, power: float, *, name: str) -> Matrix:
    """Return ``A^power`` of a symmetric positive-semidefinite matrix.

    Eigenvalues above ``-1e-10 max(1, |lambda|_max)`` are clipped to zero;
    a negative power requires every eigenvalue to be positive.

    Args:
        matrix: Symmetric matrix.
        power: Real exponent, typically ``1/2`` or ``-1/2``.
        name: Argument name for error messages.

    Returns:
        ``V diag(lambda^power) V'``.

    Raises:
        ValueError: If the matrix is not positive semidefinite, or is
            singular when ``power < 0``.

    Equation:
        [EQ-OT-GAUSS-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    values, vectors = symmetric_eigen(matrix)
    scale = max(1.0, max(abs(value) for value in values))
    if min(values) < -_PSD_TOLERANCE * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    clipped = [max(value, 0.0) for value in values]
    if power < 0.0 and min(clipped) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    powered = [value**power if value > 0.0 else 0.0 for value in clipped]
    size = len(matrix)
    return tuple(
        tuple(
            math.fsum(vectors[i][k] * powered[k] * vectors[j][k] for k in range(size))
            for j in range(size)
        )
        for i in range(size)
    )


def _sym(matrix: Matrix) -> Matrix:
    """Return the symmetric part ``(A + A')/2``.

    Args:
        matrix: Square matrix.

    Returns:
        Symmetric part, removing rounding asymmetry from products.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    size = len(matrix)
    return tuple(
        tuple(0.5 * (matrix[i][j] + matrix[j][i]) for j in range(size))
        for i in range(size)
    )


def bures_distance(
    covariance0: Iterable[Iterable[object]], covariance1: Iterable[Iterable[object]]
) -> float:
    """Return the Bures distance between positive-semidefinite matrices.

    ``B(S0, S1)^2 = tr S0 + tr S1 - 2 tr (S0^(1/2) S1 S0^(1/2))^(1/2)``.
    It is the covariance part of the Gaussian ``W_2`` distance.

    Args:
        covariance0: Symmetric positive-semidefinite matrix.
        covariance1: Symmetric positive-semidefinite matrix of equal size.

    Returns:
        ``B(S0, S1) >= 0``.

    Raises:
        ValueError: If a matrix is invalid or the sizes differ.

    Equation:
        [EQ-OT-GAUSS-001].

    References:
        [REF-PEYRE-CUTURI-2019, Remark 2.31], [REF-DOWSON-LANDAU-1982].

    """
    s0 = _symmetric(covariance0, name="covariance0")
    s1 = _symmetric(covariance1, name="covariance1")
    if len(s0) != len(s1):
        raise ValueError("covariances must have equal size")
    root0 = _psd_power(s0, 0.5, name="covariance0")
    _psd_power(s1, 0.5, name="covariance1")
    middle = _sym(matmul(matmul(root0, s1), root0))
    cross = _psd_power(middle, 0.5, name="S0^(1/2) S1 S0^(1/2)")
    size = len(s0)
    squared = math.fsum(s0[i][i] + s1[i][i] - 2.0 * cross[i][i] for i in range(size))
    return math.sqrt(max(squared, 0.0))


def gaussian_wasserstein(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
) -> float:
    """Return ``W_2(N(m0, S0), N(m1, S1)) = sqrt(|m0 - m1|^2 + B(S0, S1)^2)``.

    Args:
        mean0: Mean of the first law.
        covariance0: Symmetric positive-semidefinite covariance.
        mean1: Mean of the second law.
        covariance1: Symmetric positive-semidefinite covariance.

    Returns:
        The Wasserstein-2 distance.

    Raises:
        ValueError: If dimensions disagree or a covariance is invalid.

    Equation:
        [EQ-OT-GAUSS-001].

    References:
        [REF-DOWSON-LANDAU-1982], [REF-OLKIN-PUKELSHEIM-1982],
        [REF-PEYRE-CUTURI-2019, Remark 2.31].

    """
    m0 = as_vector(mean0, name="mean0")
    m1 = as_vector(mean1, name="mean1")
    s0 = as_matrix(covariance0, name="covariance0", square=True)
    if not (len(m0) == len(m1) == len(s0)):
        raise ValueError("means and covariances must share one dimension")
    shift = math.fsum((x - y) ** 2 for x, y in zip(m0, m1, strict=True))
    bures = bures_distance(s0, covariance1)
    return math.sqrt(shift + bures * bures)


def normal_wasserstein(mu0: float, sigma0: float, mu1: float, sigma1: float) -> float:
    """Return ``W_2 = sqrt((mu0 - mu1)^2 + (sigma0 - sigma1)^2)`` for two normals.

    Args:
        mu0: Mean of the first law.
        sigma0: Non-negative standard deviation of the first law.
        mu1: Mean of the second law.
        sigma1: Non-negative standard deviation of the second law.

    Returns:
        The Wasserstein-2 distance.

    Raises:
        ValueError: If a standard deviation is negative.

    Equation:
        [EQ-OT-GAUSS-001].

    References:
        [REF-PEYRE-CUTURI-2019, Remark 2.31].

    """
    m0 = require_finite_real(mu0, name="mu0")
    m1 = require_finite_real(mu1, name="mu1")
    s0 = require_finite_real(sigma0, name="sigma0")
    s1 = require_finite_real(sigma1, name="sigma1")
    if s0 < 0.0 or s1 < 0.0:
        raise ValueError("standard deviations must be non-negative")
    return math.hypot(m0 - m1, s0 - s1)


def quantile_wasserstein(
    law0: object,
    law1: object,
    *,
    p: float = 2.0,
    method: str = "tanh-sinh",
    intervals: int = 512,
    z_max: float = 8.0,
) -> float:
    """Return ``W_p`` between 1-D laws from their quantile functions.

    ``W_p^p = int_0^1 |Q0(u) - Q1(u)|^p du``. The default ``"tanh-sinh"``
    method integrates in ``u`` on ``(0, 1/2)`` and ``(1/2, 1)`` with nodes
    reaching about ``1e-300`` from either end, so heavy-tailed laws with a
    barely finite ``W_p`` (for example Student t with 2.2 degrees of freedom
    against a normal at ``p = 2``) converge to rounding level. The
    ``"simpson"`` method integrates ``|Q0(Phi(z)) - Q1(Phi(z))|^p phi(z)``
    on ``[-z_max, z_max]``; its truncation omits probability
    ``2 Phi(-z_max)`` and, for heavy tails, a mass that can reach several per
    cent. Upper-tail nodes use ``isf`` when the objects provide it.

    Args:
        law0: Object with ``quantile`` (for example a distribution or a
            :class:`QuantileBarycenter`).
        law1: Second object with ``quantile``.
        p: Exponent ``p >= 1``.
        method: ``"tanh-sinh"`` (default) or ``"simpson"``.
        intervals: Positive even number of Simpson subintervals
            (``"simpson"`` only).
        z_max: Positive truncation point in the normal variable
            (``"simpson"`` only).

    Returns:
        The Wasserstein-``p`` distance.

    Raises:
        TypeError: If a law has no ``quantile``.
        ValueError: If ``p < 1``, the method is unknown, or the quadrature
            settings are invalid.

    Equation:
        [EQ-OT-W1D-001], [EQ-VAL-QUAD-DE-001].

    References:
        [REF-PEYRE-CUTURI-2019, Remark 2.30], [REF-VILLANI-2009, Chapter 6],
        [REF-TAKAHASI-MORI-1974].

    """
    exponent = require_finite_real(p, name="p")
    if exponent < 1.0:
        raise ValueError("p must be at least 1")
    if method == "tanh-sinh":

        def quantile(law: object, u: float, uc: float) -> float:
            """Return ``Q(u)``, using ``isf(1 - u)`` in the upper half.

            References:
                [EQ-OT-W1D-001], [REF-PEYRE-CUTURI-2019, Remark 2.30].

            """
            function = getattr(law, "quantile", None)
            if not callable(function):
                raise TypeError(f"{type(law).__name__} has no quantile method")
            if u <= 0.5:
                return float(function(u))
            isf = getattr(law, "isf", None)
            return float(isf(uc)) if callable(isf) else float(function(u))

        def by_u(u: float, uc: float) -> float:
            """Return ``|Q0(u) - Q1(u)|^p``.

            References:
                [EQ-OT-W1D-001], [REF-PEYRE-CUTURI-2019, Remark 2.30].

            """
            difference = quantile(law0, u, uc) - quantile(law1, u, uc)
            return float(abs(difference) ** exponent)

        total = tanh_sinh(by_u, 0.0, 0.5).value + tanh_sinh(by_u, 0.5, 1.0).value
        return float(max(total, 0.0) ** (1.0 / exponent))
    if method != "simpson":
        raise ValueError("method must be 'tanh-sinh' or 'simpson'")

    def integrand(z: float) -> float:
        """Return ``|Q0 - Q1|^p`` at ``u = Phi(z)``.

        References:
            [EQ-OT-W1D-001], [REF-PEYRE-CUTURI-2019, Remark 2.30].

        """
        return float(abs(quantile_at(law0, z) - quantile_at(law1, z)) ** exponent)

    value = quantile_expectation(integrand, intervals=intervals, z_max=z_max)
    return float(max(value, 0.0) ** (1.0 / exponent))


def _simplex_weights(weights: Iterable[object] | None, count: int) -> Vector:
    """Validate barycentric weights, defaulting to uniform.

    Args:
        weights: Non-negative weights summing to one, or ``None``.
        count: Number of components.

    Returns:
        Validated weights.

    Raises:
        ValueError: If the weights are negative, mis-sized, or do not sum to
            one within ``1e-9``.

    References:
        [REF-ALVAREZ-ESTEBAN-ETAL-2016]. No equation is evaluated.

    """
    if count == 0:
        raise ValueError("at least one law is required")
    if weights is None:
        return tuple(1.0 / count for _ in range(count))
    checked = as_vector(weights, name="weights")
    if len(checked) != count:
        raise ValueError("weights must have one entry per law")
    if any(weight < 0.0 for weight in checked):
        raise ValueError("weights must be non-negative")
    if abs(math.fsum(checked) - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError("weights must sum to one")
    return checked


@dataclass(frozen=True, slots=True)
class QuantileBarycenter:
    """The ``W_2`` barycenter of 1-D laws, ``Q(u) = sum_k w_k Q_k(u)``.

    Args:
        laws: Component objects with ``quantile`` (and optionally ``isf`` and
            ``mean``).
        weights: Non-negative weights summing to one.

    References:
        [REF-ALVAREZ-ESTEBAN-ETAL-2016], [REF-PEYRE-CUTURI-2019, Section 9.2].

    """

    laws: tuple[object, ...]
    weights: Vector

    def quantile(self, probability: float) -> float:
        """Return the barycenter quantile ``sum_k w_k Q_k(u)``.

        Args:
            probability: Probability in ``[0, 1]``.

        Returns:
            The weighted average of the component quantiles.

        Equation:
            [EQ-OT-BARY-001].

        References:
            [REF-PEYRE-CUTURI-2019, Section 9.2].

        """
        return math.fsum(
            weight * float(law.quantile(probability))  # type: ignore[attr-defined]
            for law, weight in zip(self.laws, self.weights, strict=True)
        )

    def isf(self, probability: float) -> float:
        """Return the barycenter upper quantile ``sum_k w_k Q_k(1 - u)``.

        Components without ``isf`` use ``quantile(1 - u)``.

        Args:
            probability: Upper-tail probability in ``[0, 1]``.

        Returns:
            The weighted average of the component upper quantiles.

        Equation:
            [EQ-OT-BARY-001].

        References:
            [REF-PEYRE-CUTURI-2019, Section 9.2].

        """
        terms = []
        for law, weight in zip(self.laws, self.weights, strict=True):
            isf = getattr(law, "isf", None)
            if callable(isf):
                terms.append(weight * float(isf(probability)))
            else:
                terms.append(
                    weight * float(law.quantile(1.0 - probability))  # type: ignore[attr-defined]
                )
        return math.fsum(terms)

    @property
    def mean(self) -> float:
        """Return ``sum_k w_k E[X_k]``, since ``E[X] = int_0^1 Q(u) du``.

        Returns:
            The barycenter mean.

        Raises:
            AttributeError: If a component has no ``mean``.

        Equation:
            [EQ-OT-BARY-001].

        References:
            [REF-PEYRE-CUTURI-2019, Section 9.2].

        """
        return math.fsum(
            weight * float(law.mean)  # type: ignore[attr-defined]
            for law, weight in zip(self.laws, self.weights, strict=True)
        )


def quantile_barycenter(
    laws: Sequence[object], weights: Iterable[object] | None = None
) -> QuantileBarycenter:
    """Return the ``W_2`` barycenter of 1-D laws by averaging quantiles.

    On the real line ``W_2`` is the ``L^2`` distance between quantile
    functions, so ``argmin_nu sum_k w_k W_2(nu, mu_k)^2`` has quantile
    ``Q = sum_k w_k Q_k``. Two normal laws give the normal law with averaged
    mean and averaged standard deviation.

    Args:
        laws: Objects with ``quantile``.
        weights: Non-negative weights summing to one (default uniform).

    Returns:
        A :class:`QuantileBarycenter`.

    Raises:
        TypeError: If a law has no ``quantile``.
        ValueError: If the weights are invalid.

    Equation:
        [EQ-OT-BARY-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 9.2], [REF-ALVAREZ-ESTEBAN-ETAL-2016].

    """
    components = tuple(laws)
    checked = _simplex_weights(weights, len(components))
    for law in components:
        if not callable(getattr(law, "quantile", None)):
            raise TypeError(f"{type(law).__name__} has no quantile method")
    return QuantileBarycenter(components, checked)


@dataclass(frozen=True, slots=True)
class GaussianBarycenter:
    """The ``W_2`` barycenter of Gaussian laws.

    Args:
        mean: Barycenter mean ``sum_k w_k m_k``.
        covariance: Fixed point ``S`` of the barycenter equation.
        iterations: Number of fixed-point steps.
        converged: Whether the relative step fell below the tolerance.
        residual: Final relative fixed-point residual
            ``|sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2) - S|_F / |S|_F``.
        trace: Relative step ``|S_new - S|_F / |S|_F`` at every iteration.

    References:
        [REF-ALVAREZ-ESTEBAN-ETAL-2016].

    """

    mean: Vector
    covariance: Matrix
    iterations: int
    converged: bool
    residual: float
    trace: Vector


def _frobenius(matrix: Matrix) -> float:
    """Return the Frobenius norm.

    Args:
        matrix: Matrix.

    Returns:
        ``sqrt(sum_ij A_ij^2)``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.3]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


def _difference(left: Matrix, right: Matrix) -> Matrix:
    """Return ``left - right``.

    Args:
        left: Matrix.
        right: Matrix of the same shape.

    Returns:
        Entrywise difference.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(
        tuple(a - b for a, b in zip(row_l, row_r, strict=True))
        for row_l, row_r in zip(left, right, strict=True)
    )


def _barycenter_map(
    covariance: Matrix, components: Sequence[Matrix], w: Vector
) -> Matrix:
    """Return ``sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2)``.

    Args:
        covariance: Current positive-definite iterate ``S``.
        components: Component covariances ``S_k``.
        w: Barycentric weights.

    Returns:
        The weighted sum of matrix square roots.

    Equation:
        [EQ-OT-BARY-002].

    References:
        [REF-ALVAREZ-ESTEBAN-ETAL-2016].

    """
    root = _psd_power(covariance, 0.5, name="covariance")
    size = len(covariance)
    total = [[0.0] * size for _ in range(size)]
    for weight, component in zip(w, components, strict=True):
        if weight == 0.0:
            continue
        inner = _sym(matmul(matmul(root, component), root))
        square_root = _psd_power(inner, 0.5, name="S^(1/2) S_k S^(1/2)")
        for i in range(size):
            for j in range(size):
                total[i][j] += weight * square_root[i][j]
    return tuple(tuple(row) for row in total)


def gaussian_barycenter(
    means: Sequence[Iterable[object]],
    covariances: Sequence[Iterable[Iterable[object]]],
    weights: Iterable[object] | None = None,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 500,
) -> GaussianBarycenter:
    """Return the ``W_2`` barycenter of Gaussian laws by fixed-point iteration.

    The barycenter is ``N(sum_k w_k m_k, S)`` where ``S`` solves
    ``S = sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2)``. Starting from
    ``S = sum_k w_k S_k``, the iteration
    ``S <- S^(-1/2) (sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2))^2 S^(-1/2)``
    of Alvarez-Esteban et al. (2016) is applied until the relative
    Frobenius step is below ``tolerance``. For commuting covariances the
    fixed point is ``(sum_k w_k S_k^(1/2))^2``.

    Args:
        means: Component mean vectors of one dimension ``d``.
        covariances: Component symmetric positive-definite covariances.
        weights: Non-negative weights summing to one (default uniform).
        tolerance: Positive relative stopping tolerance.
        max_iterations: Positive iteration limit.

    Returns:
        Barycenter mean and covariance with convergence diagnostics.

    Raises:
        ValueError: If inputs are invalid or a covariance is not positive
            definite.

    Equation:
        [EQ-OT-BARY-002].

    References:
        [REF-ALVAREZ-ESTEBAN-ETAL-2016].

    """
    mean_vectors = [as_vector(mean, name="means") for mean in means]
    components = [_symmetric(c, name="covariances") for c in covariances]
    count = len(mean_vectors)
    if count == 0 or len(components) != count:
        raise ValueError("means and covariances must be non-empty and paired")
    w = _simplex_weights(weights, count)
    size = len(mean_vectors[0])
    if any(len(m) != size for m in mean_vectors) or any(
        len(c) != size for c in components
    ):
        raise ValueError("means and covariances must share one dimension")
    for component in components:
        _psd_power(component, -0.5, name="covariances")
    stop = require_finite_real(tolerance, name="tolerance")
    if not stop > 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    mean = tuple(
        math.fsum(weight * m[i] for weight, m in zip(w, mean_vectors, strict=True))
        for i in range(size)
    )
    covariance: Matrix = tuple(
        tuple(
            math.fsum(weight * c[i][j] for weight, c in zip(w, components, strict=True))
            for j in range(size)
        )
        for i in range(size)
    )
    trace: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007
        mapped = _barycenter_map(covariance, components, w)
        inverse_root = _psd_power(covariance, -0.5, name="covariance")
        updated = _sym(
            matmul(matmul(inverse_root, matmul(mapped, mapped)), inverse_root)
        )
        step = _frobenius(_difference(updated, covariance)) / max(
            _frobenius(covariance), 1e-300
        )
        trace.append(step)
        covariance = updated
        if step <= stop:
            converged = True
            break
    residual = _frobenius(
        _difference(_barycenter_map(covariance, components, w), covariance)
    ) / max(_frobenius(covariance), 1e-300)
    return GaussianBarycenter(
        mean=mean,
        covariance=covariance,
        iterations=iterations,
        converged=converged,
        residual=residual,
        trace=tuple(trace),
    )
