"""Function-space representation of unknown fields on an interval.

Experimental (added in 0.28).

An unknown field ``f`` on ``[0, L]`` is represented as ``f(s) = sum_j x_j
phi_j(s)`` with coefficients ``x`` and a basis ``phi_j`` [EQ-INV-FS-001].
Three bases are provided, each with an exact Gram (mass) matrix
[EQ-INV-FS-002]:

- :class:`PiecewiseConstantBasis`: ``n`` equal cells, ``phi_j`` the indicator
  of cell ``j``; the mass matrix is ``h I``.
- :class:`HatBasis`: nodal piecewise linear hats at the ``n`` *interior*
  nodes ``s_j = j h``, ``h = L / (n + 1)``; the field vanishes at ``0`` and
  ``L`` (zero Dirichlet ends). The mass matrix is tridiagonal
  ``h / 6 * [1 4 1]``.
- :class:`SineBasis`: ``phi_k = sqrt(2 / L) sin(k pi s / L)``, orthonormal in
  ``L2(0, L)``; the mass matrix is the identity.

Coefficients and values are **not** interchangeable: for a non-orthonormal
basis the discrete ``l2`` norm of ``x`` differs from the ``L2`` norm of ``f``
by the mass matrix, ``||f||^2 = x' M x``. Observation functionals
``y_i = int K_i(s) f(s) ds`` become a dense operator with entries
``A_ij = int K_i phi_j ds`` by composite five-point Gauss-Legendre
quadrature [EQ-INV-FS-003]. The quadrature grid is chosen independently of
the basis mesh (a panel count or explicit breakpoints on ``[0, L]``); the
rule is exact for polynomials up to degree 9 per panel, so the result is
exact for polynomial kernels against basis functions only when no panel
contains a kink or jump of the basis; otherwise it converges as the panels
are refined (the independence of the quadrature from the basis is the point
of a functional inverse experiment, not an inverse crime).

:class:`SpectralGaussianPrior` is the Gaussian measure on functions with
covariance operator ``C = a^2 (kappa^2 - Laplacian)^(-s)`` under Dirichlet
boundary conditions. Its eigenfunctions are the sine functions and its
eigenvalues ``lambda_k = a^2 (kappa^2 + (k pi / L)^2)^(-s)`` do not depend on
any mesh or truncation, so the coefficient variances of ``n = 16`` and
``n = 64`` modes agree on their common part [EQ-INV-FS-004]. The covariance
is trace class on ``L2`` in one dimension if and only if ``s > 1/2``; the
truncation error is controlled by the remaining trace, bounded in closed
form [EQ-INV-FS-005]. Karhunen-Loeve draws and pointwise variances follow
[EQ-INV-FS-006]. :func:`credible_bands` gives empirical pointwise and
simultaneous bands [EQ-INV-FS-007].

Limitations: truncation to a finite number of modes is an approximation of
the continuum prior; no statement about the infinite-dimensional posterior is
proved here. Bands are empirical and finite-sample.
"""

from __future__ import annotations

import abc
import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import DenseOperator, DiagonalCovariance, Matrix, Vector

__all__ = [
    "CredibleBands",
    "HatBasis",
    "PiecewiseConstantBasis",
    "SineBasis",
    "SpectralGaussianPrior",
    "credible_bands",
]

# Five-point Gauss-Legendre rule on [-1, 1] in closed form [REF-GOLUB-WELSCH-1969].
_ROOT = math.sqrt(10.0 / 7.0)
_GL_X1 = math.sqrt(5.0 - 2.0 * _ROOT) / 3.0
_GL_X2 = math.sqrt(5.0 + 2.0 * _ROOT) / 3.0
_GL_W1 = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
_GL_W2 = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
_GL_NODES = (-_GL_X2, -_GL_X1, 0.0, _GL_X1, _GL_X2)
_GL_WEIGHTS = (_GL_W2, _GL_W1, 128.0 / 225.0, _GL_W1, _GL_W2)


def _positive_int(value: object, name: str) -> int:
    """Validate a positive integer.

    Args:
        value: Candidate.
        name: Name for messages.

    Returns:
        The integer.

    Raises:
        TypeError: If not an integer.
        ValueError: If not positive.

    References:
        [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_real(value: object, name: str) -> float:
    """Validate a strictly positive finite real.

    Args:
        value: Candidate.
        name: Name for messages.

    Returns:
        The value as float.

    Raises:
        TypeError: If not real.
        ValueError: If not finite and positive.

    References:
        [REF-PROBSTAT-VALIDATION].

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


class _IntervalBasis(abc.ABC):
    """Common behaviour of bases on ``[0, L]``.

    References:
        [EQ-INV-FS-001], [REF-STUART-2010].

    """

    def __init__(self, n: int, length: float) -> None:
        """Validate and store the dimension and the interval length.

        Args:
            n: Number of basis functions (at least one).
            length: Interval length ``L > 0``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``n < 1`` or ``length`` is not positive finite.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        self._n = _positive_int(n, "n")
        self._length = _positive_real(length, "length")

    @property
    def dimension(self) -> int:
        """Return the number ``n`` of basis functions.

        References:
            [EQ-INV-FS-001].

        """
        return self._n

    @property
    def length(self) -> float:
        """Return the interval length ``L``.

        References:
            [EQ-INV-FS-001].

        """
        return self._length

    @abc.abstractmethod
    def _row(self, point: float) -> list[float]:
        """Return ``(phi_1(s), ..., phi_n(s))`` at one point.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            The ``n`` basis values.

        References:
            [EQ-INV-FS-001].

        """

    @abc.abstractmethod
    def mass_matrix(self) -> Matrix:
        """Return the exact Gram matrix ``M_ij = int phi_i phi_j ds``.

        Returns:
            Dense ``n x n`` symmetric positive definite matrix.

        References:
            [EQ-INV-FS-002].

        """

    def _checked_points(self, points: Sequence[float]) -> list[float]:
        """Validate evaluation points in ``[0, L]``.

        Args:
            points: Non-empty sequence of points.

        Returns:
            The points as floats.

        Raises:
            ValueError: If empty or a point lies outside ``[0, L]``.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        values = [require_finite_real(p, name="point") for p in points]
        if not values:
            raise ValueError("points must be non-empty")
        for p in values:
            if p < 0.0 or p > self._length:
                raise ValueError(f"point {p} lies outside [0, {self._length}]")
        return values

    def _matrix(self, points: Sequence[float]) -> list[list[float]]:
        """Return the matrix ``Phi_ij = phi_j(s_i)``.

        Args:
            points: Evaluation points.

        Returns:
            List of rows.

        References:
            [EQ-INV-FS-001].

        """
        return [self._row(p) for p in self._checked_points(points)]

    def evaluate(
        self, coefficients: Sequence[float], points: Sequence[float]
    ) -> Vector:
        """Return the field values ``f(s_i) = sum_j x_j phi_j(s_i)``.

        Args:
            coefficients: Coefficient vector of length ``n``.
            points: Points in ``[0, L]``.

        Returns:
            Field values at ``points``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong, entries are non-finite or a
                point is outside ``[0, L]``.

        References:
            [EQ-INV-FS-001], [REF-STUART-2010].

        """
        x = [require_finite_real(c, name="coefficient") for c in coefficients]
        if len(x) != self._n:
            raise ValueError(f"coefficients have length {len(x)}, expected {self._n}")
        return tuple(
            math.fsum(a * b for a, b in zip(row, x, strict=True))
            for row in self._matrix(points)
        )

    def evaluation_operator(self, points: Sequence[float]) -> DenseOperator:
        """Return the point-evaluation operator ``x -> (f(s_i))``.

        Args:
            points: Points in ``[0, L]``.

        Returns:
            Dense ``len(points) x n`` operator with entries ``phi_j(s_i)``.

        Raises:
            ValueError: If ``points`` is empty or outside ``[0, L]``.

        References:
            [EQ-INV-FS-001], [REF-STUART-2010].

        """
        return DenseOperator(self._matrix(points), name="basis evaluation")

    def integral_operator(
        self,
        kernels: Sequence[Callable[[float], float]],
        quadrature_points: int | Sequence[float] = 200,
    ) -> DenseOperator:
        """Return ``A_ij = int_0^L K_i(s) phi_j(s) ds`` by Gauss-Legendre.

        The integral uses a composite five-point Gauss-Legendre rule on a
        quadrature grid that is independent of the basis mesh.

        Args:
            kernels: Non-empty sequence of callables ``K_i(s) -> float``.
            quadrature_points: Either an integer number of equal panels, or
                strictly increasing breakpoints ``0 = b_0 < ... < b_P = L``.
                Each panel receives five nodes.

        Returns:
            Dense ``len(kernels) x n`` operator.

        Raises:
            TypeError: If a kernel is not callable or the grid has a wrong
                type.
            ValueError: If the grid is invalid or a kernel value is not
                finite.

        References:
            [EQ-INV-FS-003], [REF-GOLUB-WELSCH-1969],
            [REF-DAVIS-RABINOWITZ-1984].

        """
        kernel_list = list(kernels)
        if not kernel_list:
            raise ValueError("kernels must be non-empty")
        if not all(callable(k) for k in kernel_list):
            raise TypeError("every kernel must be callable")
        nodes, weights = _composite_rule(self._length, quadrature_points)
        phi_columns = list(zip(*(self._row(s) for s in nodes), strict=True))
        rows: list[list[float]] = []
        for kernel in kernel_list:
            weighted = [
                w * require_finite_real(kernel(s), name="kernel value")
                for s, w in zip(nodes, weights, strict=True)
            ]
            rows.append(
                [math.fsum(map(operator.mul, weighted, c)) for c in phi_columns]
            )
        return DenseOperator(rows, name="integral operator")


def _composite_rule(
    length: float, grid: int | Sequence[float]
) -> tuple[list[float], list[float]]:
    """Return nodes and weights of the composite five-point rule on ``[0, L]``.

    Args:
        length: Interval length.
        grid: Panel count or breakpoints.

    Returns:
        Nodes and weights.

    Raises:
        TypeError: If ``grid`` has a wrong type.
        ValueError: If the grid is invalid.

    References:
        [EQ-INV-FS-003], [REF-GOLUB-WELSCH-1969].

    """
    if isinstance(grid, int) and not isinstance(grid, bool):
        panels = _positive_int(grid, "quadrature_points")
        breaks = [length * i / panels for i in range(panels + 1)]
    elif isinstance(grid, bool) or isinstance(grid, (str, bytes)):
        raise TypeError("quadrature_points must be an integer or a sequence")
    else:
        breaks = [require_finite_real(b, name="breakpoint") for b in grid]
        if len(breaks) < 2:
            raise ValueError("breakpoints need at least two entries")
        tol = 1e-12 * length
        if abs(breaks[0]) > tol or abs(breaks[-1] - length) > tol:
            raise ValueError("breakpoints must start at 0 and end at L")
        if any(b2 <= b1 for b1, b2 in zip(breaks, breaks[1:], strict=False)):
            raise ValueError("breakpoints must be strictly increasing")
        breaks[0], breaks[-1] = 0.0, length
    nodes: list[float] = []
    weights: list[float] = []
    for a, b in zip(breaks, breaks[1:], strict=False):
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        for x, w in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            nodes.append(mid + half * x)
            weights.append(half * w)
    return nodes, weights


class PiecewiseConstantBasis(_IntervalBasis):
    """Piecewise constant basis on ``n`` equal cells of ``[0, L]``.

    ``phi_j`` is the indicator of ``[j h, (j + 1) h)`` with ``h = L / n``
    (right-continuous; ``s = L`` belongs to the last cell). The mass matrix is
    ``h I``.

    Args:
        n: Number of cells.
        length: Interval length ``L``.

    References:
        [EQ-INV-FS-001], [EQ-INV-FS-002].

    """

    def _row(self, point: float) -> list[float]:
        """Return the indicator row of the cell containing ``point``.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            Basis values.

        References:
            [EQ-INV-FS-001].

        """
        h = self._length / self._n
        cell = min(int(point / h), self._n - 1)
        row = [0.0] * self._n
        row[cell] = 1.0
        return row

    def mass_matrix(self) -> Matrix:
        """Return the diagonal mass matrix ``h I``.

        Returns:
            Dense ``n x n`` matrix.

        References:
            [EQ-INV-FS-002].

        """
        h = self._length / self._n
        return tuple(
            tuple(h if i == j else 0.0 for j in range(self._n)) for i in range(self._n)
        )


class HatBasis(_IntervalBasis):
    """Nodal piecewise linear basis at interior nodes (zero Dirichlet ends).

    Nodes are ``s_j = j h`` for ``j = 1..n`` with ``h = L / (n + 1)``;
    ``phi_j`` is the hat with ``phi_j(s_i) = delta_ij``, vanishing outside
    ``(s_{j-1}, s_{j+1})``. There are no basis functions at ``0`` and ``L``,
    so every represented field is zero there. Coefficients are the nodal
    field values. The mass matrix is tridiagonal with diagonal ``4 h / 6`` and
    off-diagonals ``h / 6``.

    Args:
        n: Number of interior nodes.
        length: Interval length ``L``.

    References:
        [EQ-INV-FS-001], [EQ-INV-FS-002].

    """

    @property
    def spacing(self) -> float:
        """Return the node spacing ``h = L / (n + 1)``.

        References:
            [EQ-INV-FS-002].

        """
        return self._length / (self._n + 1)

    @property
    def nodes(self) -> Vector:
        """Return the interior nodes ``h, 2h, ..., n h``.

        References:
            [EQ-INV-FS-001].

        """
        h = self.spacing
        return tuple(h * (j + 1) for j in range(self._n))

    def _row(self, point: float) -> list[float]:
        """Return the (at most two) non-zero hat values at ``point``.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            Basis values.

        References:
            [EQ-INV-FS-001].

        """
        t = point / self.spacing
        i = min(int(t), self._n + 1)
        frac = t - i
        row = [0.0] * self._n
        if 1 <= i <= self._n:
            row[i - 1] = 1.0 - frac
        if 1 <= i + 1 <= self._n:
            row[i] = frac
        return row

    def mass_matrix(self) -> Matrix:
        """Return the tridiagonal mass matrix ``h / 6 [1 4 1]``.

        Returns:
            Dense ``n x n`` matrix.

        References:
            [EQ-INV-FS-002].

        """
        h = self.spacing
        n = self._n

        def entry(i: int, j: int) -> float:
            """Return one mass-matrix entry.

            Args:
                i: Row index.
                j: Column index.

            Returns:
                ``4h/6`` on the diagonal, ``h/6`` next to it, else zero.

            References:
                [EQ-INV-FS-002].

            """
            if i == j:
                return 4.0 * h / 6.0
            return h / 6.0 if abs(i - j) == 1 else 0.0

        return tuple(tuple(entry(i, j) for j in range(n)) for i in range(n))


class SineBasis(_IntervalBasis):
    """Orthonormal sine basis ``phi_k = sqrt(2 / L) sin(k pi s / L)``.

    ``k = 1..n``; the functions are orthonormal in ``L2(0, L)`` and satisfy
    zero Dirichlet conditions. The mass matrix is the identity, so coefficient
    and ``L2`` norms agree.

    Args:
        n: Number of modes.
        length: Interval length ``L``.

    References:
        [EQ-INV-FS-001], [EQ-INV-FS-002].

    """

    def _row(self, point: float) -> list[float]:
        """Return ``phi_k(point)`` for ``k = 1..n``.

        Args:
            point: Point in ``[0, L]``.

        Returns:
            Basis values.

        References:
            [EQ-INV-FS-001].

        """
        scale = math.sqrt(2.0 / self._length)
        omega = math.pi * point / self._length
        return [scale * math.sin((k + 1) * omega) for k in range(self._n)]

    def mass_matrix(self) -> Matrix:
        """Return the identity.

        Returns:
            Dense ``n x n`` identity.

        References:
            [EQ-INV-FS-002].

        """
        return tuple(
            tuple(1.0 if i == j else 0.0 for j in range(self._n))
            for i in range(self._n)
        )


class SpectralGaussianPrior:
    """Gaussian prior with covariance ``a^2 (kappa^2 - Laplacian)^(-s)``.

    Dirichlet boundary conditions on ``[0, L]``. The eigenpairs are the sine
    functions ``phi_k`` with ``lambda_k = a^2 (kappa^2 + (k pi / L)^2)^(-s)``
    [EQ-INV-FS-004]. With ``s > 1/2`` the covariance is trace class on
    ``L2`` [EQ-INV-FS-005]; larger ``s`` gives smoother draws. The
    coefficient variances are properties of the continuum prior and do not
    depend on the number of retained modes.

    Args:
        length: Interval length ``L > 0``.
        amplitude: Amplitude ``a > 0`` (units of the field times
            ``L^(s - 1/2)``-type scaling; see the eigenvalue formula).
        kappa: Inverse length scale ``kappa >= 0``.
        smoothness: Exponent ``s``; must exceed ``1/2``.
        modes: Default number of retained modes (at least one).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a parameter is outside its range, in particular
            ``smoothness <= 1/2`` (not trace class in one dimension).

    References:
        [EQ-INV-FS-004], [EQ-INV-FS-005], [REF-STUART-2010],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """

    def __init__(
        self,
        length: float,
        amplitude: float,
        kappa: float,
        smoothness: float,
        modes: int,
    ) -> None:
        """Validate and store the prior parameters.

        Args:
            length: Interval length.
            amplitude: Amplitude ``a``.
            kappa: Inverse length scale.
            smoothness: Exponent ``s > 1/2``.
            modes: Default number of modes.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a parameter is out of range.

        References:
            [EQ-INV-FS-004], [EQ-INV-FS-005].

        """
        self.length = _positive_real(length, "length")
        self.amplitude = _positive_real(amplitude, "amplitude")
        self.kappa = require_finite_real(kappa, name="kappa")
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        self.smoothness = require_finite_real(smoothness, name="smoothness")
        if self.smoothness <= 0.5:
            raise ValueError(
                "smoothness must exceed 1/2 for a trace-class covariance in 1D"
            )
        self.modes = _positive_int(modes, "modes")

    def _modes(self, modes: int | None) -> int:
        """Resolve an optional mode count.

        Args:
            modes: Count or ``None`` for the default.

        Returns:
            The count.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        return self.modes if modes is None else _positive_int(modes, "modes")

    def eigenvalue(self, k: int) -> float:
        """Return ``lambda_k`` for mode ``k >= 1``.

        Args:
            k: Mode index.

        Returns:
            ``a^2 (kappa^2 + (k pi / L)^2)^(-s)``.

        Raises:
            TypeError: If ``k`` is not an integer.
            ValueError: If ``k < 1``.

        References:
            [EQ-INV-FS-004].

        """
        k = _positive_int(k, "k")
        omega = k * math.pi / self.length
        return self.amplitude**2 * math.pow(self.kappa**2 + omega**2, -self.smoothness)

    def coefficient_covariance(self, modes: int | None = None) -> DiagonalCovariance:
        """Return the covariance of the sine-basis coefficients.

        The ``k``-th variance is ``lambda_k``, independent of ``modes``.

        Args:
            modes: Number of modes (default: the constructor value).

        Returns:
            Diagonal covariance for ``SineBasis(modes, L)`` coefficients.

        Raises:
            ValueError: If ``modes < 1``.

        References:
            [EQ-INV-FS-004].

        """
        count = self._modes(modes)
        return DiagonalCovariance([self.eigenvalue(k) for k in range(1, count + 1)])

    def trace(self, modes: int | None = None) -> float:
        """Return the partial trace ``sum_{k<=modes} lambda_k``.

        Args:
            modes: Number of modes (default: the constructor value).

        Returns:
            The partial sum, which equals the expected squared ``L2`` norm of
            the truncated field.

        Raises:
            ValueError: If ``modes < 1``.

        References:
            [EQ-INV-FS-005].

        """
        count = self._modes(modes)
        return math.fsum(self.eigenvalue(k) for k in range(1, count + 1))

    def remaining_trace_bound(self, modes: int | None = None) -> float:
        """Return an upper bound on ``sum_{k>modes} lambda_k``.

        Since ``lambda_k`` decreases in ``k``, the tail is at most
        ``int_N^inf lambda(x) dx <= a^2 (L / pi)^(2s) N^(1 - 2s) / (2s - 1)``
        with ``N = modes``.

        Args:
            modes: Number of retained modes (default: the constructor value).

        Returns:
            The bound; also bounds the squared ``L2`` truncation error in
            expectation and, times ``2 / L``, the pointwise variance error.

        Raises:
            ValueError: If ``modes < 1``.

        References:
            [EQ-INV-FS-005].

        """
        n = self._modes(modes)
        s = self.smoothness
        return (
            self.amplitude**2
            * math.pow(self.length / math.pi, 2.0 * s)
            * math.pow(float(n), 1.0 - 2.0 * s)
            / (2.0 * s - 1.0)
        )

    def pointwise_variance(
        self, points: Sequence[float], modes: int | None = None
    ) -> Vector:
        """Return ``Var f(s) = sum_k lambda_k phi_k(s)^2`` at ``points``.

        The truncation error at any point is at most
        ``(2 / L) * remaining_trace_bound(modes)``.

        Args:
            points: Points in ``[0, L]``.
            modes: Number of modes (default: the constructor value).

        Returns:
            Variances of the truncated prior field.

        Raises:
            ValueError: If ``modes < 1`` or a point is outside ``[0, L]``.

        References:
            [EQ-INV-FS-006].

        """
        count = self._modes(modes)
        basis = SineBasis(count, self.length)
        lam = [self.eigenvalue(k) for k in range(1, count + 1)]
        return tuple(
            math.fsum(lk * v * v for lk, v in zip(lam, row, strict=True))
            for row in basis._matrix(points)
        )

    def sample(
        self,
        count: int,
        seed: int | RandomStream | None,
        points: Sequence[float],
        modes: int | None = None,
    ) -> tuple[Vector, ...]:
        """Draw Karhunen-Loeve samples evaluated at ``points``.

        ``f = sum_k sqrt(lambda_k) xi_k phi_k`` with ``xi_k`` independent
        standard normal.

        Args:
            count: Number of draws.
            seed: Integer seed or :class:`RandomStream`.
            points: Evaluation points in ``[0, L]``.
            modes: Number of modes (default: the constructor value).

        Returns:
            ``count`` field vectors, one value per point.

        Raises:
            TypeError: If ``count`` or ``seed`` has a wrong type.
            ValueError: If ``count < 1``, ``modes < 1`` or a point is outside.

        References:
            [EQ-INV-FS-006], [REF-STUART-2010].

        """
        n_draws = _positive_int(count, "count")
        n_modes = self._modes(modes)
        rng = resolve_stream(seed).generator()
        phi = SineBasis(n_modes, self.length)._matrix(points)
        scale = [math.sqrt(self.eigenvalue(k)) for k in range(1, n_modes + 1)]
        draws: list[Vector] = []
        for _ in range(n_draws):
            coeff = [sc * rng.gauss(0.0, 1.0) for sc in scale]
            draws.append(
                tuple(
                    math.fsum(a * b for a, b in zip(row, coeff, strict=True))
                    for row in phi
                )
            )
        return tuple(draws)


@dataclass(frozen=True, slots=True)
class CredibleBands:
    """Empirical pointwise and simultaneous bands for sampled fields.

    Attributes:
        level: Nominal probability content.
        sample_count: Number of samples ``N``.
        mean: Pointwise sample mean.
        standard_deviation: Pointwise sample standard deviation (``N - 1``).
        lower: Pointwise lower quantile at ``(1 - level) / 2``.
        upper: Pointwise upper quantile at ``(1 + level) / 2``.
        simultaneous_lower: ``mean - c sd``.
        simultaneous_upper: ``mean + c sd``.
        scaling_factor: ``c``, the empirical ``level`` quantile of
            ``max_j |f - mean|_j / sd_j`` over the samples.
        scaling_factor_interval: Approximate 95% order-statistic interval for
            ``c`` (normal approximation to the binomial rank), expressing the
            Monte Carlo uncertainty of the scaling factor.

    References:
        [EQ-INV-FS-007].

    """

    level: float
    sample_count: int
    mean: Vector
    standard_deviation: Vector
    lower: Vector
    upper: Vector
    simultaneous_lower: Vector
    simultaneous_upper: Vector
    scaling_factor: float
    scaling_factor_interval: tuple[float, float]


def _quantile(sorted_values: Sequence[float], p: float) -> float:
    """Return the type-7 (linear interpolation) quantile.

    Args:
        sorted_values: Ascending values.
        p: Probability in ``[0, 1]``.

    Returns:
        The quantile.

    References:
        [EQ-INV-FS-007].

    """
    position = (len(sorted_values) - 1) * p
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    frac = position - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def credible_bands(samples: Sequence[Sequence[float]], level: float) -> CredibleBands:
    """Compute pointwise and simultaneous empirical bands from field samples.

    Pointwise: equal-tail sample quantiles. Simultaneous: ``mean +- c sd`` with
    ``c`` the empirical ``level`` quantile of ``T_j = max_i |f_ji - mean_i| /
    sd_i``. Both are empirical and finite-sample; their accuracy is limited by
    the number of samples, and ``c`` is reported with an order-statistic
    interval.

    Args:
        samples: At least two field vectors of equal length on a common grid.
        level: Probability content in ``(0, 1)``.

    Returns:
        :class:`CredibleBands`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than two samples, ragged or non-finite samples,
            ``level`` outside ``(0, 1)``, or a grid point where the sample
            standard deviation is zero (drop such points, for example
            Dirichlet boundary points).

    References:
        [EQ-INV-FS-007], [REF-STUART-2010].

    """
    p = require_finite_real(level, name="level")
    if not 0.0 < p < 1.0:
        raise ValueError("level must lie in (0, 1)")
    data = [[require_finite_real(v, name="sample value") for v in s] for s in samples]
    n = len(data)
    if n < 2:
        raise ValueError("at least two samples are required")
    m = len(data[0])
    if m == 0 or any(len(row) != m for row in data):
        raise ValueError("samples must be non-empty and of equal length")
    columns = [[row[i] for row in data] for i in range(m)]
    mean = [math.fsum(c) / n for c in columns]
    sd = [
        math.sqrt(math.fsum((v - mu) ** 2 for v in c) / (n - 1))
        for c, mu in zip(columns, mean, strict=True)
    ]
    if any(s == 0.0 for s in sd):
        raise ValueError("a grid point has zero sample standard deviation")
    lower = [_quantile(sorted(c), 0.5 * (1.0 - p)) for c in columns]
    upper = [_quantile(sorted(c), 0.5 * (1.0 + p)) for c in columns]
    stats = sorted(
        max(abs(v - mu) / s for v, mu, s in zip(row, mean, sd, strict=True))
        for row in data
    )
    factor = _quantile(stats, p)
    half = 1.96 * math.sqrt(n * p * (1.0 - p))
    k_low = max(0, math.floor(n * p - half))
    k_high = min(n - 1, math.ceil(n * p + half))
    return CredibleBands(
        level=p,
        sample_count=n,
        mean=tuple(mean),
        standard_deviation=tuple(sd),
        lower=tuple(lower),
        upper=tuple(upper),
        simultaneous_lower=tuple(
            mu - factor * s for mu, s in zip(mean, sd, strict=True)
        ),
        simultaneous_upper=tuple(
            mu + factor * s for mu, s in zip(mean, sd, strict=True)
        ),
        scaling_factor=factor,
        scaling_factor_interval=(stats[k_low], stats[k_high]),
    )
