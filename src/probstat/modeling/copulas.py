"""Copulas: the dependence part of a joint input law.

By Sklar's theorem a joint CDF with continuous marginals ``F_i`` is
``F(x) = C(F_1(x_1), ..., F_d(x_d))`` for a unique copula ``C``, and the joint
density is ``f(x) = c(u) prod_i f_i(x_i)`` with ``u_i = F_i(x_i)``. A
:class:`~probstat.modeling.JointInput` combines named marginals with one of
the copulas defined here:

- :class:`IndependenceCopula` (``C = prod u_i``);
- :class:`GaussianCopula` (the Nataf model; the default when a latent
  correlation matrix is given);
- :class:`StudentTCopula` (elliptical with symmetric tail dependence);
- the exchangeable Archimedean copulas :class:`ClaytonCopula` (lower-tail
  dependence), :class:`GumbelCopula` (upper-tail dependence) and
  :class:`FrankCopula` (no tail dependence), in any dimension where the
  generator is completely monotone; the bivariate Frank copula also admits
  negative dependence.

Every copula provides its log-density on the open unit cube, a CDF where one
is tractable, pairwise Kendall's tau and tail-dependence coefficients in
closed form, the Rosenblatt transform ``w_k = C(u_k | u_1, ..., u_{k-1})``
with its inverse (the conditional-inverse construction), a sampler from its
stochastic representation (normal variance mixture or Marshall-Olkin
frailty), and its lower-dimensional margins.

The Rosenblatt transform depends on the order of the variables. Any order
gives an exact isoprobabilistic map, but the standard-normal image of a
limit-state surface changes with the order, so FORM/SORM approximations
(not exact probabilities) can differ between orders. For the Gaussian copula
the Rosenblatt and Nataf maps coincide for every order up to a rotation of
the standard space, so FORM is order-invariant there; for the other copulas
it is not.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from statistics import NormalDist
from typing import ClassVar

from probstat.distributions import StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    dot,
    inverse_spd,
    log_det_spd,
    matvec,
    solve_lower,
    symmetric_eigen,
    symmetrize,
)
from probstat.numerics import gauss_kronrod
from probstat.optimization import brent_root

_NORMAL = NormalDist()
_TINY = 1e-300
_ONE_MINUS = 1.0 - 2.0**-53
_MAX_ARCHIMEDEAN_DIMENSION = 100
_FRANK_THETA_LIMIT = 700.0


def _normal_cdf(z: float) -> float:
    """Return ``Phi(z)`` through the complementary error function.

    Equation:
        [EQ-DIST-NORM-002].

    References:
        [REF-NIST-DLMF, Equation 7.2.2].

    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _normal_quantile(probability: float) -> float:
    """Return ``Phi^-1(p)`` for ``p`` clamped into the open unit interval.

    Equation:
        [EQ-DIST-NORM-006].

    References:
        [REF-PYTHON-STATISTICS].

    """
    return _NORMAL.inv_cdf(min(max(probability, _TINY), _ONE_MINUS))


def _open(probability: float) -> float:
    """Clamp a probability into ``[1e-300, 1 - 2^-53]``.

    References:
        Project validation policy. No equation is evaluated.

    """
    return min(max(probability, _TINY), _ONE_MINUS)


def _require_dimension(value: object, *, minimum: int) -> int:
    """Validate an integer dimension of at least ``minimum``.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        Project validation policy. No equation is evaluated.

    """
    size = require_non_negative_integer(value, name="dimension")
    if size < minimum:
        raise ValueError(f"dimension must be at least {minimum}")
    return size


def _unit_point(u: Sequence[float], dimension: int, *, closed: bool = False) -> Vector:
    """Validate a point of the open (or closed) unit cube.

    Raises:
        ValueError: If the length is wrong or a coordinate is outside.

    References:
        Project validation policy. No equation is evaluated.

    """
    point = as_vector(u, name="u")
    if len(point) != dimension:
        raise ValueError(f"point length must equal the copula dimension {dimension}")
    for value in point:
        inside = 0.0 <= value <= 1.0 if closed else 0.0 < value < 1.0
        if not inside:
            interval = "[0, 1]" if closed else "(0, 1)"
            raise ValueError(f"copula coordinates must lie in {interval}")
    return point


def _pair(i: int, j: int, dimension: int) -> tuple[int, int]:
    """Validate a pair of distinct coordinate indices.

    Raises:
        ValueError: If an index is out of range or the indices coincide.

    References:
        Project validation policy. No equation is evaluated.

    """
    first = require_non_negative_integer(i, name="i")
    second = require_non_negative_integer(j, name="j")
    if first >= dimension or second >= dimension or first == second:
        raise ValueError("i and j must be distinct indices below the dimension")
    return first, second


def _indices(indices: Sequence[int], dimension: int) -> tuple[int, ...]:
    """Validate a non-empty sequence of distinct coordinate indices.

    Raises:
        ValueError: If it is empty, repeats, or leaves the range.

    References:
        Project validation policy. No equation is evaluated.

    """
    chosen = tuple(require_non_negative_integer(i, name="index") for i in indices)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("indices must be a non-empty set of distinct indices")
    if max(chosen) >= dimension:
        raise ValueError("an index exceeds the copula dimension")
    return chosen


def _validated_correlation(matrix: Matrix) -> tuple[Matrix, Matrix]:
    """Validate a correlation matrix and return it with its Cholesky factor.

    Raises:
        ValueError: If the matrix is not a positive-definite correlation.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    checked = symmetrize(matrix, name="correlation")
    size = len(checked)
    for i in range(size):
        if abs(checked[i][i] - 1.0) > 1e-12:
            raise ValueError("correlation matrix must have a unit diagonal")
        for j in range(size):
            if abs(checked[i][j]) > 1.0:
                raise ValueError("correlation entries must lie in [-1, 1]")
    try:
        lower = cholesky(checked)
    except ValueError as error:
        raise ValueError("correlation must be positive definite") from error
    return checked, lower


def _log_sum_exp(values: Sequence[float]) -> float:
    """Return ``log sum exp(v_i)`` without overflow.

    References:
        [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

    """
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def _positive_uniform(generator: random.Random) -> float:
    """Return a uniform variate strictly inside ``(0, 1)``.

    References:
        [REF-DEVROYE-1986, Section II.2].

    """
    value = generator.random()
    while value == 0.0:
        value = generator.random()
    return value


class Copula(ABC):
    """Abstract ``d``-dimensional copula on the unit hypercube.

    Subclasses define ``dimension`` and implement the log-density, the
    Rosenblatt transform and its inverse, pairwise dependence measures, and
    margins. The default sampler maps independent uniforms through the
    inverse Rosenblatt transform; families with a stochastic representation
    override it.

    References:
        [REF-NELSEN-2006, Chapter 2], [REF-JOE-2014].

    """

    family: ClassVar[str] = "copula"
    dimension: int

    @abstractmethod
    def log_density(self, u: Sequence[float]) -> float:
        """Return the log copula density at a point of the open unit cube.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            ``log c(u)``.

        Equation:
            [EQ-MOD-COP-SKLAR-001].

        References:
            [REF-NELSEN-2006, Section 2.3].

        """

    def density(self, u: Sequence[float]) -> float:
        """Return the copula density ``c(u)``.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Non-negative density.

        Equation:
            [EQ-MOD-COP-SKLAR-001].

        References:
            [REF-NELSEN-2006, Section 2.3].

        """
        return math.exp(self.log_density(u))

    def cdf(self, u: Sequence[float]) -> float:
        """Return ``C(u)`` on the closed unit cube.

        Boundary rules hold for every copula: ``C = 0`` when a coordinate is
        zero, and coordinates equal to one drop out (the margin of the
        remaining coordinates is evaluated).

        Args:
            u: Point in ``[0, 1]^d``.

        Returns:
            Copula CDF value.

        Raises:
            NotImplementedError: If the family has no tractable CDF in the
                required dimension.

        Equation:
            [EQ-MOD-COP-SKLAR-001].

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        point = _unit_point(u, self.dimension, closed=True)
        if any(value == 0.0 for value in point):
            return 0.0
        keep = tuple(i for i, value in enumerate(point) if value < 1.0)
        if not keep:
            return 1.0
        if len(keep) == 1:
            return point[keep[0]]
        sub = self if len(keep) == self.dimension else self.marginal(keep)
        return min(1.0, max(0.0, sub._interior_cdf(tuple(point[i] for i in keep))))

    def _interior_cdf(self, u: Vector) -> float:
        """Return ``C(u)`` for ``u`` in the open cube (override when tractable).

        Raises:
            NotImplementedError: Always, unless overridden.

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        raise NotImplementedError(
            f"the {self.family} copula has no tractable CDF in dimension {len(u)}"
        )

    @property
    def has_cdf(self) -> bool:
        """Return whether :meth:`cdf` is available in every dimension.

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return False

    @abstractmethod
    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return Kendall's tau of coordinates ``i`` and ``j`` in closed form.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Population Kendall's tau in ``[-1, 1]``.

        Equation:
            [EQ-MOD-COP-TAU-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """

    @abstractmethod
    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return the lower and upper tail-dependence coefficients of a pair.

        ``lambda_L = lim_{q->0} P(U_j <= q | U_i <= q)`` and
        ``lambda_U = lim_{q->1} P(U_j > q | U_i > q)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            ``(lambda_L, lambda_U)``.

        Equation:
            [EQ-MOD-COP-TAIL-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """

    @abstractmethod
    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Map a copula point to independent uniforms (Rosenblatt transform).

        ``w_1 = u_1`` and ``w_k = C(u_k | u_1, ..., u_{k-1})``. The map depends
        on the coordinate order.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Independent uniform coordinates ``w``.

        Equation:
            [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-HOHENBICHLER-RACKWITZ-1981].

        """

    @abstractmethod
    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Map independent uniforms to a copula point (conditional inverse).

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            Copula point ``u`` with ``rosenblatt(u) = w``; coordinates are
            clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-DEVROYE-1986, Chapter XI].

        """

    @abstractmethod
    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the copula of the coordinates ``indices`` (in that order).

        Args:
            indices: Distinct coordinate indices.

        Returns:
            Margin copula of dimension ``len(indices)``.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """

    def _draw(self, generator: random.Random) -> Vector:
        """Draw one copula point by the conditional-inverse method.

        References:
            [REF-DEVROYE-1986, Chapter XI].

        """
        return self.inverse_rosenblatt(
            tuple(_positive_uniform(generator) for _ in range(self.dimension))
        )

    def sample(
        self, size: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw independent copula points from the stochastic representation.

        Args:
            size: Positive number of draws.
            seed: Integer seed or stream (``None`` is not reproducible).

        Returns:
            Points of the unit hypercube.

        Raises:
            ValueError: If ``size`` is not positive.

        Equation:
            [EQ-MOD-COP-ARCH-003], [EQ-MOD-COP-T-001].

        References:
            [REF-MARSHALL-OLKIN-1988], [REF-HOFERT-2008],
            [REF-MCNEIL-FREY-EMBRECHTS-2015].

        """
        count = require_non_negative_integer(size, name="size")
        if count < 1:
            raise ValueError("size must be positive")
        generator = resolve_stream(seed).generator()
        return tuple(self._draw(generator) for _ in range(count))


@dataclass(frozen=True)
class IndependenceCopula(Copula):
    """The independence copula ``C(u) = prod_i u_i``.

    Args:
        dimension: Positive dimension.

    References:
        [REF-NELSEN-2006, Chapter 2].

    """

    dimension: int
    family: ClassVar[str] = "independence"

    def __post_init__(self) -> None:
        """Validate the dimension.

        Raises:
            ValueError: If the dimension is not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(
            self, "dimension", _require_dimension(self.dimension, minimum=1)
        )

    def log_density(self, u: Sequence[float]) -> float:
        """Return ``log c(u) = 0`` on the open cube.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Zero.

        Equation:
            [EQ-MOD-COP-SKLAR-001].

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        _unit_point(u, self.dimension)
        return 0.0

    def _interior_cdf(self, u: Vector) -> float:
        """Return ``prod_i u_i``.

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        return math.prod(u)

    @property
    def has_cdf(self) -> bool:
        """Return ``True``: the CDF is the product of the coordinates.

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return True

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return zero.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            ``0``.

        Equation:
            [EQ-MOD-COP-TAU-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 0.0

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return ``(0, 0)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Zero tail dependence.

        Equation:
            [EQ-MOD-COP-TAIL-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 0.0, 0.0

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return ``u`` unchanged.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            ``u``.

        Equation:
            [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952].

        """
        return _unit_point(u, self.dimension)

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Return ``w`` unchanged.

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            ``w``.

        Equation:
            [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952].

        """
        return _unit_point(w, self.dimension)

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the independence copula of the selected dimension.

        Args:
            indices: Distinct coordinate indices.

        Returns:
            Independence copula.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        return IndependenceCopula(len(_indices(indices, self.dimension)))


def _bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """Return ``P(Z_1 <= a, Z_2 <= b)`` for a standard bivariate normal pair.

    Uses Plackett's identity ``d Phi_2 / d rho = phi_2`` integrated from the
    independent case by adaptive Gauss-Kronrod quadrature.

    Raises:
        ArithmeticError: If the quadrature does not converge.

    Equation:
        [EQ-MOD-COP-GAUSS-001].

    References:
        [REF-PLACKETT-1954], [REF-PIESSENS-QUADPACK-1983, Section 2.2].

    """
    base = _normal_cdf(a) * _normal_cdf(b)
    if rho == 0.0:
        return base

    def integrand(r: float) -> float:
        """Return the bivariate normal density at ``(a, b)`` with correlation ``r``.

        References:
            [EQ-MOD-COP-GAUSS-001], [REF-PLACKETT-1954].

        """
        one_minus = (1.0 - r) * (1.0 + r)
        exponent = -(a * a - 2.0 * a * b * r + b * b) / (2.0 * one_minus)
        return math.exp(exponent) / (2.0 * math.pi * math.sqrt(one_minus))

    low, high = (0.0, rho) if rho > 0.0 else (rho, 0.0)
    result = gauss_kronrod(integrand, low, high, abs_tol=1e-15, rel_tol=1e-12)
    if not result.converged:
        raise ArithmeticError("bivariate normal CDF quadrature did not converge")
    value = base + (result.value if rho > 0.0 else -result.value)
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class GaussianCopula(Copula):
    """Gaussian (Nataf) copula with latent correlation matrix ``R``.

    ``c(u) = det(R)^(-1/2) exp(-z'(R^-1 - I) z / 2)`` with ``z_i = Phi^-1(u_i)``.
    Its Rosenblatt transform is ``w = Phi(L^-1 z)`` with ``R = L L'``, the
    same map :class:`~probstat.modeling.JointInput` has always used.

    Args:
        correlation: Symmetric positive-definite unit-diagonal matrix.

    References:
        [REF-NELSEN-2006, Section 2.3], [REF-LIU-DERKIUREGHIAN-1986].

    """

    correlation: Matrix
    dimension: int = field(init=False)
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _precision_minus_identity: Matrix = field(init=False, repr=False, compare=False)
    _log_det: float = field(init=False, repr=False, compare=False)
    family: ClassVar[str] = "gaussian"

    def __post_init__(self) -> None:
        """Validate the correlation matrix and cache its factors.

        Raises:
            ValueError: If the matrix is not a positive-definite correlation.

        Equation:
            [EQ-LINALG-001].

        References:
            [REF-NELSEN-2006, Section 2.3].

        """
        matrix, lower = _validated_correlation(self.correlation)
        size = len(matrix)
        precision = inverse_spd(matrix)
        object.__setattr__(self, "correlation", matrix)
        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(
            self,
            "_precision_minus_identity",
            tuple(
                tuple(precision[i][j] - (1.0 if i == j else 0.0) for j in range(size))
                for i in range(size)
            ),
        )
        object.__setattr__(self, "_log_det", log_det_spd(matrix))

    def log_density(self, u: Sequence[float]) -> float:
        """Return the Gaussian-copula log-density.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            ``-0.5 z'(R^-1 - I) z - 0.5 log det R``.

        Equation:
            [EQ-MOD-COPULA-001].

        References:
            [REF-NELSEN-2006, Section 2.3].

        """
        point = _unit_point(u, self.dimension)
        z = tuple(_normal_quantile(value) for value in point)
        quadratic = dot(z, matvec(self._precision_minus_identity, z))
        return -0.5 * quadratic - 0.5 * self._log_det

    def _interior_cdf(self, u: Vector) -> float:
        """Return the bivariate Gaussian-copula CDF by Plackett's identity.

        Raises:
            NotImplementedError: For dimensions above two.

        Equation:
            [EQ-MOD-COP-GAUSS-001].

        References:
            [REF-PLACKETT-1954].

        """
        if len(u) != 2:
            raise NotImplementedError(
                "the Gaussian copula CDF is implemented for two dimensions only"
            )
        return _bivariate_normal_cdf(
            _normal_quantile(u[0]), _normal_quantile(u[1]), self.correlation[0][1]
        )

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return ``tau = (2 / pi) arcsin(rho_ij)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Equation:
            [EQ-MOD-COP-TAU-001].

        References:
            [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-JOE-2014].

        """
        a, b = _pair(i, j, self.dimension)
        return 2.0 / math.pi * math.asin(self.correlation[a][b])

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return ``(0, 0)``: the Gaussian copula is tail independent for ``|rho| < 1``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Zero tail dependence.

        Equation:
            [EQ-MOD-COP-TAIL-001].

        References:
            [REF-MCNEIL-FREY-EMBRECHTS-2015].

        """
        _pair(i, j, self.dimension)
        return 0.0, 0.0

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return ``w = Phi(L^-1 Phi^-1(u))``.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Independent uniforms.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-NATAF-001].

        References:
            [REF-ROSENBLATT-1952], [REF-LEBRUN-DUTFOY-2009].

        """
        point = _unit_point(u, self.dimension)
        latent = tuple(_normal_quantile(value) for value in point)
        return tuple(_normal_cdf(y) for y in solve_lower(self._lower, latent))

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Return ``u = Phi(L Phi^-1(w))``.

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            Copula point.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-NATAF-001].

        References:
            [REF-ROSENBLATT-1952], [REF-LEBRUN-DUTFOY-2009].

        """
        point = _unit_point(w, self.dimension)
        latent = matvec(self._lower, tuple(_normal_quantile(value) for value in point))
        return tuple(_open(_normal_cdf(z)) for z in latent)

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the Gaussian copula of a coordinate subset (sub-matrix of ``R``).

        Args:
            indices: Distinct coordinate indices.

        Returns:
            Gaussian copula (independence copula for one index).

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        chosen = _indices(indices, self.dimension)
        if len(chosen) == 1:
            return IndependenceCopula(1)
        return GaussianCopula(
            tuple(tuple(self.correlation[a][b] for b in chosen) for a in chosen)
        )


@dataclass(frozen=True)
class StudentTCopula(Copula):
    """Student-t copula with correlation ``R`` and ``nu`` degrees of freedom.

    ``c(u) = t_{nu,R}(x) / prod_i t_nu(x_i)`` with ``x_i = T_nu^-1(u_i)``.
    Pairs have symmetric tail dependence
    ``lambda = 2 T_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho)))``, which vanishes
    only as ``nu -> inf`` (the Gaussian copula).

    Args:
        correlation: Symmetric positive-definite unit-diagonal matrix.
        nu: Finite positive degrees of freedom (need not be an integer).

    References:
        [REF-DEMARTA-MCNEIL-2005], [REF-MCNEIL-FREY-EMBRECHTS-2015].

    """

    correlation: Matrix
    nu: float
    dimension: int = field(init=False)
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _log_det: float = field(init=False, repr=False, compare=False)
    _laws: tuple[StudentT, ...] = field(init=False, repr=False, compare=False)
    family: ClassVar[str] = "student-t"

    def __post_init__(self) -> None:
        """Validate parameters and cache the factor and conditional laws.

        Raises:
            TypeError: If ``nu`` is not real.
            ValueError: If ``nu <= 0`` or the matrix is not a correlation.

        Equation:
            [EQ-LINALG-001].

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        nu = require_finite_real(self.nu, name="nu")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        matrix, lower = _validated_correlation(self.correlation)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "correlation", matrix)
        object.__setattr__(self, "dimension", len(matrix))
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_log_det", log_det_spd(matrix))
        object.__setattr__(
            self, "_laws", tuple(StudentT(nu + k) for k in range(len(matrix)))
        )

    def _scores(self, point: Vector) -> Vector:
        """Return ``x_i = T_nu^-1(u_i)``.

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        law = self._laws[0]
        return tuple(float(law.quantile(value)) for value in point)

    def log_density(self, u: Sequence[float]) -> float:
        """Return the t-copula log-density.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            ``log t_{nu,R}(x) - sum_i log t_nu(x_i)``.

        Equation:
            [EQ-MOD-COP-T-001].

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        point = _unit_point(u, self.dimension)
        x = self._scores(point)
        y = solve_lower(self._lower, x)
        quadratic = dot(y, y)
        nu = self.nu
        d = self.dimension
        constant = (
            math.lgamma(0.5 * (nu + d))
            + (d - 1) * math.lgamma(0.5 * nu)
            - d * math.lgamma(0.5 * (nu + 1.0))
            - 0.5 * self._log_det
        )
        marginal = math.fsum(math.log1p(value * value / nu) for value in x)
        return (
            constant
            - 0.5 * (nu + d) * math.log1p(quadratic / nu)
            + 0.5 * (nu + 1.0) * marginal
        )

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return ``tau = (2 / pi) arcsin(rho_ij)`` (any elliptical copula).

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Equation:
            [EQ-MOD-COP-TAU-001].

        References:
            [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-DEMARTA-MCNEIL-2005].

        """
        a, b = _pair(i, j, self.dimension)
        return 2.0 / math.pi * math.asin(self.correlation[a][b])

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return the symmetric t-copula tail-dependence coefficient twice.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            ``(lambda, lambda)``.

        Equation:
            [EQ-MOD-COP-T-003].

        References:
            [REF-DEMARTA-MCNEIL-2005], [REF-MCNEIL-FREY-EMBRECHTS-2015].

        """
        a, b = _pair(i, j, self.dimension)
        rho = self.correlation[a][b]
        argument = math.sqrt((self.nu + 1.0) * (1.0 - rho) / (1.0 + rho))
        value = 2.0 * float(StudentT(self.nu + 1.0).cdf(-argument))
        return value, value

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return the sequential conditional-t Rosenblatt transform.

        With ``x = L y`` and ``q_k = sum_{j<k} y_j^2``,
        ``T_k = y_k sqrt((nu + k)/(nu + q_k))`` is ``t_{nu+k}`` and independent
        of the earlier coordinates, so ``w_k = T_{nu+k}(T_k)``.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Independent uniforms.

        Equation:
            [EQ-MOD-COP-T-002], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-DEMARTA-MCNEIL-2005].

        """
        point = _unit_point(u, self.dimension)
        y = solve_lower(self._lower, self._scores(point))
        result: list[float] = []
        accumulated = 0.0
        for k, value in enumerate(y):
            statistic = value * math.sqrt((self.nu + k) / (self.nu + accumulated))
            law = self._laws[k]
            if statistic > 0.0:
                result.append(1.0 - float(law.sf(statistic)))
            else:
                result.append(float(law.cdf(statistic)))
            accumulated += value * value
        return tuple(result)

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Invert the sequential conditional-t transform.

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            Copula point.

        Equation:
            [EQ-MOD-COP-T-002], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-DEMARTA-MCNEIL-2005].

        """
        point = _unit_point(w, self.dimension)
        y: list[float] = []
        accumulated = 0.0
        for k, value in enumerate(point):
            statistic = float(self._laws[k].quantile(value))
            scaled = statistic * math.sqrt((self.nu + accumulated) / (self.nu + k))
            y.append(scaled)
            accumulated += scaled * scaled
        x = matvec(self._lower, tuple(y))
        return tuple(_open(float(self._laws[0].cdf(value))) for value in x)

    def _draw(self, generator: random.Random) -> Vector:
        """Draw ``U = T_nu(L Z / sqrt(W / nu))`` with ``W ~ chi^2_nu``.

        Equation:
            [EQ-MOD-COP-T-001].

        References:
            [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-DEMARTA-MCNEIL-2005].

        """
        z = tuple(generator.gauss(0.0, 1.0) for _ in range(self.dimension))
        mixing = generator.gammavariate(0.5 * self.nu, 2.0)
        while mixing <= 0.0:
            mixing = generator.gammavariate(0.5 * self.nu, 2.0)
        factor = math.sqrt(self.nu / mixing)
        law = self._laws[0]
        return tuple(
            _open(float(law.cdf(value * factor))) for value in matvec(self._lower, z)
        )

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the t copula of a coordinate subset (same ``nu``).

        Args:
            indices: Distinct coordinate indices.

        Returns:
            Student-t copula (independence copula for one index).

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        chosen = _indices(indices, self.dimension)
        if len(chosen) == 1:
            return IndependenceCopula(1)
        return StudentTCopula(
            tuple(tuple(self.correlation[a][b] for b in chosen) for a in chosen),
            self.nu,
        )


class ArchimedeanCopula(Copula):
    """Exchangeable Archimedean copula ``C(u) = psi(sum_i phi(u_i))``.

    ``psi`` is the generator (a Laplace transform for the families here) and
    ``phi = psi^-1``. The density is ``|psi^(d)(t)| prod_i |phi'(u_i)|`` and
    the conditional law is
    ``C(u_k | u_<k) = psi^(k-1)(t_k) / psi^(k-1)(t_{k-1})`` with
    ``t_k = sum_{i<=k} phi(u_i)``. Samples use the Marshall-Olkin frailty
    representation ``U_i = psi(E_i / V)`` with ``V`` the frailty whose
    Laplace transform is ``psi``.

    References:
        [REF-NELSEN-2006, Chapter 4], [REF-MARSHALL-OLKIN-1988],
        [REF-HOFERT-2008], [REF-JOE-2014].

    """

    theta: float
    dimension: int

    @abstractmethod
    def psi(self, t: float) -> float:
        """Return the generator ``psi(t)`` for ``t >= 0``.

        Args:
            t: Non-negative argument.

        Returns:
            Generator value in ``(0, 1]``.

        Equation:
            [EQ-MOD-COP-ARCH-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """

    @abstractmethod
    def phi(self, u: float) -> float:
        """Return the inverse generator ``phi(u)`` for ``u`` in ``(0, 1]``.

        Args:
            u: Probability.

        Returns:
            Non-negative value.

        Equation:
            [EQ-MOD-COP-ARCH-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """

    @abstractmethod
    def _log_abs_psi_derivative(self, order: int, t: float) -> float:
        """Return ``log |psi^(order)(t)|`` for ``order >= 1``.

        References:
            [REF-NELSEN-2006, Chapter 4].

        """

    @abstractmethod
    def _log_abs_phi_prime(self, u: float) -> float:
        """Return ``log |phi'(u)|``.

        References:
            [REF-NELSEN-2006, Chapter 4].

        """

    @abstractmethod
    def _frailty(self, generator: random.Random) -> float:
        """Draw the frailty ``V`` whose Laplace transform is ``psi``.

        References:
            [REF-MARSHALL-OLKIN-1988], [REF-HOFERT-2008].

        """

    def _phi_checked(self, u: float) -> float:
        """Return ``phi(u)`` and reject arguments that underflow to zero.

        Raises:
            ArithmeticError: If ``phi(u)`` underflows (``u`` too close to one).

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        value = self.phi(u)
        if not value > 0.0:
            raise ArithmeticError(
                f"phi(u) underflows for u={u!r} under the {self.family} copula"
            )
        return value

    def log_density(self, u: Sequence[float]) -> float:
        """Return ``log |psi^(d)(t)| + sum_i log |phi'(u_i)|``, ``t = sum phi(u_i)``.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Log copula density.

        Equation:
            [EQ-MOD-COP-ARCH-001].

        References:
            [REF-NELSEN-2006, Chapter 4], [REF-JOE-2014].

        """
        point = _unit_point(u, self.dimension)
        total = math.fsum(self._phi_checked(value) for value in point)
        return self._log_abs_psi_derivative(self.dimension, total) + math.fsum(
            self._log_abs_phi_prime(value) for value in point
        )

    def _interior_cdf(self, u: Vector) -> float:
        """Return ``psi(sum_i phi(u_i))``.

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return self.psi(math.fsum(self.phi(value) for value in u))

    @property
    def has_cdf(self) -> bool:
        """Return ``True``: the Archimedean CDF is closed form.

        References:
            [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

        """
        return True

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return ``w_k = psi^(k-1)(t_k) / psi^(k-1)(t_{k-1})``.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Independent uniforms.

        Equation:
            [EQ-MOD-COP-ARCH-002], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-HOFERT-2008].

        """
        point = _unit_point(u, self.dimension)
        result = [point[0]]
        previous = self._phi_checked(point[0])
        for order in range(1, self.dimension):
            current = previous + self._phi_checked(point[order])
            log_ratio = self._log_abs_psi_derivative(
                order, current
            ) - self._log_abs_psi_derivative(order, previous)
            result.append(min(1.0, math.exp(log_ratio)))
            previous = current
        return tuple(result)

    def _conditional_step(self, order: int, previous: float, log_w: float) -> float:
        """Solve ``log|psi^(order)(t + s)| - log|psi^(order)(t)| = log w`` for ``s``.

        The left side decreases from zero to ``-inf`` in ``s``, so a bracket
        on ``y = log s`` always exists; Brent's method solves it.

        Raises:
            ArithmeticError: If no bracket is found in the floating range.

        Equation:
            [EQ-MOD-COP-ARCH-002].

        References:
            [REF-BRENT-1973, Chapter 4], [REF-HOFERT-2008].

        """
        base = self._log_abs_psi_derivative(order, previous)

        def residual(y: float) -> float:
            """Return the log-ratio residual at ``s = exp(y)``.

            References:
                [EQ-MOD-COP-ARCH-002], [REF-HOFERT-2008].

            """
            return (
                self._log_abs_psi_derivative(order, previous + math.exp(y))
                - base
                - log_w
            )

        high = max(0.0, math.log1p(previous))
        while residual(high) > 0.0:
            high += 2.0
            if high > 700.0:
                raise ArithmeticError("conditional inverse has no bracket")
        low = high - 2.0
        while residual(low) <= 0.0:
            low -= 2.0
            if low < -745.0:
                raise ArithmeticError("conditional inverse has no bracket")
        return math.exp(brent_root(residual, low, high, tolerance=1e-15).root)

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Invert the Archimedean Rosenblatt transform coordinate by coordinate.

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            Copula point.

        Equation:
            [EQ-MOD-COP-ARCH-002], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-ROSENBLATT-1952], [REF-HOFERT-2008].

        """
        point = _unit_point(w, self.dimension)
        result = [point[0]]
        previous = self._phi_checked(point[0])
        for order in range(1, self.dimension):
            step = self._conditional_step(order, previous, math.log(point[order]))
            result.append(_open(self.psi(step)))
            previous += step
        return tuple(result)

    def _draw(self, generator: random.Random) -> Vector:
        """Draw ``U_i = psi(E_i / V)`` (Marshall-Olkin frailty construction).

        Equation:
            [EQ-MOD-COP-ARCH-003].

        References:
            [REF-MARSHALL-OLKIN-1988], [REF-HOFERT-2008].

        """
        frailty = self._frailty(generator)
        return tuple(
            _open(self.psi(generator.expovariate(1.0) / frailty))
            for _ in range(self.dimension)
        )

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the same family and parameter in the selected dimension.

        Args:
            indices: Distinct coordinate indices.

        Returns:
            Archimedean copula (independence copula for one index).

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        chosen = _indices(indices, self.dimension)
        if len(chosen) == 1:
            return IndependenceCopula(1)
        return type(self)(self.theta, len(chosen))  # type: ignore[call-arg]


@dataclass(frozen=True)
class ClaytonCopula(ArchimedeanCopula):
    """Clayton copula: ``psi(t) = (1 + t)^(-1/theta)``, ``theta > 0``.

    Kendall's tau is ``theta / (theta + 2)``; lower-tail dependence
    ``2^(-1/theta)``, no upper-tail dependence. The frailty is
    ``Gamma(1/theta, 1)``. Valid in every dimension.

    Args:
        theta: Positive dependence parameter.
        dimension: Dimension of at least two.

    References:
        [REF-NELSEN-2006, Chapters 4-5], [REF-HOFERT-2008].

    """

    theta: float
    dimension: int = 2
    family: ClassVar[str] = "clayton"

    def __post_init__(self) -> None:
        """Validate ``theta > 0`` and the dimension.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If ``theta <= 0`` or the dimension is invalid.

        References:
            [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

        """
        theta = require_finite_real(self.theta, name="theta")
        if theta <= 0.0:
            raise ValueError("Clayton theta must be positive")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(
            self,
            "dimension",
            _require_dimension(self.dimension, minimum=2),
        )
        if self.dimension > _MAX_ARCHIMEDEAN_DIMENSION:
            raise ValueError("Archimedean dimension is limited to 100")

    def psi(self, t: float) -> float:
        """Return ``(1 + t)^(-1/theta)``.

        Args:
            t: Non-negative argument.

        Returns:
            Generator value.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return math.exp(-math.log1p(t) / self.theta)

    def phi(self, u: float) -> float:
        """Return ``u^(-theta) - 1``.

        Args:
            u: Probability in ``(0, 1]``.

        Returns:
            Inverse generator value.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return math.expm1(-self.theta * math.log(u))

    def _log_abs_psi_derivative(self, order: int, t: float) -> float:
        """Return ``sum_{j<k} log(1/theta + j) - (1/theta + k) log(1 + t)``.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-HOFERT-2008].

        """
        alpha = 1.0 / self.theta
        return math.fsum(math.log(alpha + j) for j in range(order)) - (
            alpha + order
        ) * math.log1p(t)

    def _log_abs_phi_prime(self, u: float) -> float:
        """Return ``log theta - (theta + 1) log u``.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return math.log(self.theta) - (self.theta + 1.0) * math.log(u)

    def _conditional_step(self, order: int, previous: float, log_w: float) -> float:
        """Solve the Clayton conditional equation in closed form.

        ``s = (1 + t) expm1(-log w / (1/theta + k))``.

        Equation:
            [EQ-MOD-COP-CLAYTON-001], [EQ-MOD-COP-ARCH-002].

        References:
            [REF-HOFERT-2008].

        """
        return (1.0 + previous) * math.expm1(-log_w / (1.0 / self.theta + order))

    def _frailty(self, generator: random.Random) -> float:
        """Draw ``V ~ Gamma(1/theta, 1)``.

        Equation:
            [EQ-MOD-COP-ARCH-003].

        References:
            [REF-MARSHALL-OLKIN-1988], [REF-HOFERT-2008].

        """
        value = generator.gammavariate(1.0 / self.theta, 1.0)
        while value <= 0.0:
            value = generator.gammavariate(1.0 / self.theta, 1.0)
        return value

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return ``theta / (theta + 2)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return self.theta / (self.theta + 2.0)

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return ``(2^(-1/theta), 0)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Lower and upper tail-dependence coefficients.

        Equation:
            [EQ-MOD-COP-CLAYTON-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 2.0 ** (-1.0 / self.theta), 0.0


def _gumbel_coefficients(alpha: float, order: int) -> tuple[tuple[int, float], ...]:
    """Return ``(m, a_km)`` with ``|psi^(k)(t)| = e^(-t^a) sum_m a_km t^(m a - k)``.

    Derived by differentiating ``e^{-t^alpha} t^{m alpha - k}``: the
    recursion ``a_{k+1,m+1} += alpha a_{k,m}`` and
    ``a_{k+1,m} += (k - m alpha) a_{k,m}`` keeps every coefficient
    non-negative for ``alpha <= 1``, so the sum has no cancellation.

    Equation:
        [EQ-MOD-COP-GUMBEL-001].

    References:
        [REF-HOFERT-MACHLER-MCNEIL-2012].

    """
    current: dict[int, float] = {0: 1.0}
    for k in range(order):
        updated: dict[int, float] = {}
        for m, value in current.items():
            updated[m + 1] = updated.get(m + 1, 0.0) + alpha * value
            updated[m] = updated.get(m, 0.0) + (k - m * alpha) * value
        current = updated
    return tuple((m, value) for m, value in sorted(current.items()) if value > 0.0)


@dataclass(frozen=True)
class GumbelCopula(ArchimedeanCopula):
    """Gumbel (Gumbel-Hougaard) copula: ``psi(t) = exp(-t^(1/theta))``, ``theta >= 1``.

    Kendall's tau is ``1 - 1/theta``; upper-tail dependence
    ``2 - 2^(1/theta)``, no lower-tail dependence. The frailty is a positive
    stable law with Laplace transform ``exp(-t^(1/theta))``. ``theta = 1`` is
    independence. Valid in every dimension.

    Args:
        theta: Dependence parameter of at least one.
        dimension: Dimension of at least two.

    References:
        [REF-NELSEN-2006, Chapters 4-5], [REF-HOFERT-2008].

    """

    theta: float
    dimension: int = 2
    _coefficients: tuple[tuple[tuple[int, float], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    family: ClassVar[str] = "gumbel"

    def __post_init__(self) -> None:
        """Validate ``theta >= 1`` and cache derivative coefficients.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If ``theta < 1`` or the dimension is invalid.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        theta = require_finite_real(self.theta, name="theta")
        if theta < 1.0:
            raise ValueError("Gumbel theta must be at least one")
        dimension = _require_dimension(self.dimension, minimum=2)
        if dimension > _MAX_ARCHIMEDEAN_DIMENSION:
            raise ValueError("Archimedean dimension is limited to 100")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(
            self,
            "_coefficients",
            tuple(_gumbel_coefficients(1.0 / theta, k) for k in range(dimension + 1)),
        )

    def psi(self, t: float) -> float:
        """Return ``exp(-t^(1/theta))``.

        Args:
            t: Non-negative argument.

        Returns:
            Generator value.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return math.exp(-(t ** (1.0 / self.theta)))

    def phi(self, u: float) -> float:
        """Return ``(-log u)^theta``.

        Args:
            u: Probability in ``(0, 1]``.

        Returns:
            Inverse generator value.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return float((-math.log(u)) ** self.theta)

    def _log_abs_psi_derivative(self, order: int, t: float) -> float:
        """Return ``-t^a + log sum_m a_{k,m} t^{m a - k}`` with ``a = 1/theta``.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-HOFERT-MACHLER-MCNEIL-2012].

        """
        alpha = 1.0 / self.theta
        log_t = math.log(t)
        terms = [
            math.log(value) + (m * alpha - order) * log_t
            for m, value in self._coefficients[order]
        ]
        return -math.exp(alpha * log_t) + _log_sum_exp(terms)

    def _log_abs_phi_prime(self, u: float) -> float:
        """Return ``log theta + (theta - 1) log(-log u) - log u``.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        minus_log = -math.log(u)
        return (
            math.log(self.theta)
            + (self.theta - 1.0) * math.log(minus_log)
            - math.log(u)
        )

    def _frailty(self, generator: random.Random) -> float:
        """Draw a positive stable variate with Laplace transform ``exp(-t^a)``.

        Kanter's representation with ``U ~ Uniform(0, pi)``, ``E ~ Exp(1)``:
        ``V = sin(a U) sin((1-a) U)^((1-a)/a) / (sin(U)^(1/a) E^((1-a)/a))``.

        Equation:
            [EQ-MOD-COP-GUMBEL-002].

        References:
            [REF-KANTER-1975], [REF-CHAMBERS-MALLOWS-STUCK-1976],
            [REF-HOFERT-2008].

        """
        alpha = 1.0 / self.theta
        if alpha == 1.0:
            return 1.0
        angle = math.pi * _positive_uniform(generator)
        exponential = generator.expovariate(1.0)
        while exponential <= 0.0:
            exponential = generator.expovariate(1.0)
        log_value = (
            math.log(math.sin(alpha * angle))
            + (1.0 - alpha) / alpha * math.log(math.sin((1.0 - alpha) * angle))
            - math.log(math.sin(angle)) / alpha
            - (1.0 - alpha) / alpha * math.log(exponential)
        )
        return math.exp(log_value)

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return ``1 - 1/theta``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 1.0 - 1.0 / self.theta

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return ``(0, 2 - 2^(1/theta))``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Lower and upper tail-dependence coefficients.

        Equation:
            [EQ-MOD-COP-GUMBEL-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 0.0, 2.0 - 2.0 ** (1.0 / self.theta)


def _bernoulli_numbers(count: int) -> tuple[Fraction, ...]:
    """Return ``B_0, ..., B_{count-1}`` (``B_1 = -1/2``) as exact fractions.

    Uses ``sum_{j=0}^{m} C(m+1, j) B_j = 0`` for ``m >= 1``.

    Equation:
        [EQ-MOD-COP-DEBYE-001].

    References:
        [REF-NIST-DLMF, Section 24.5].

    """
    numbers = [Fraction(1)]
    for m in range(1, count):
        total = sum(
            (Fraction(math.comb(m + 1, j)) * numbers[j] for j in range(m)),
            Fraction(0),
        )
        numbers.append(-total / (m + 1))
    return tuple(numbers)


_BERNOULLI = _bernoulli_numbers(42)
_DEBYE_SERIES = tuple(
    float(_BERNOULLI[2 * k] / ((2 * k + 1) * math.factorial(2 * k)))
    for k in range(1, 21)
)


def _debye_minus_one_over_x(x: float) -> float:
    """Return ``(D_1(x) - 1) / x`` without cancellation near ``x = 0``.

    ``D_1(x) = (1/x) int_0^x t / (e^t - 1) dt`` (first Debye function). For
    ``|x| < 2`` the Bernoulli series of ``t/(e^t - 1)`` (radius ``2 pi``) is
    summed; for ``x >= 2`` ``int_0^x = pi^2/6 - sum_k e^{-kx}(x/k + 1/k^2)``;
    for ``x <= -2`` ``D_1(x) = D_1(-x) - x/2``.

    Equation:
        [EQ-MOD-COP-DEBYE-001].

    References:
        [REF-NIST-DLMF, Section 24.2], [REF-NELSEN-2006,
        Chapter 5].

    """
    if abs(x) < 2.0:
        return -0.25 + math.fsum(
            coefficient * x ** (2 * k + 1)
            for k, coefficient in enumerate(_DEBYE_SERIES)
        )
    magnitude = abs(x)
    tail = []
    k = 1
    while True:
        term = math.exp(-k * magnitude) * (magnitude / k + 1.0 / (k * k))
        tail.append(term)
        if term < 1e-18 * magnitude or k > 200:
            break
        k += 1
    debye = (math.pi**2 / 6.0 - math.fsum(tail)) / magnitude
    if x < 0.0:
        debye += magnitude / 2.0
    return (debye - 1.0) / x


def _eulerian_rows(count: int) -> tuple[tuple[int, ...], ...]:
    """Return Eulerian-number rows ``A(n, j)`` for ``n = 0, ..., count - 1``.

    ``A(n, j) = (j + 1) A(n-1, j) + (n - j) A(n-1, j-1)`` with ``A(0, 0) = 1``.

    Equation:
        [EQ-MOD-COP-FRANK-001].

    References:
        [REF-NIST-DLMF, Section 26.14].

    """
    rows: list[tuple[int, ...]] = [(1,)]
    for n in range(1, count):
        previous = rows[-1]
        row = []
        for j in range(n):
            left = (j + 1) * previous[j] if j < len(previous) else 0
            right = (n - j) * previous[j - 1] if 0 < j <= len(previous) else 0
            row.append(left + right)
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True)
class FrankCopula(ArchimedeanCopula):
    """Frank copula: ``psi(t) = -log(1 - (1 - e^-theta) e^-t) / theta``.

    Kendall's tau is ``1 + 4 (D_1(theta) - 1) / theta`` with the first Debye
    function ``D_1``; there is no tail dependence. For ``d = 2`` any
    ``theta != 0`` is valid (negative values give negative dependence); for
    ``d >= 3`` the generator is completely monotone only for ``theta > 0``.
    The frailty is logarithmic with parameter ``1 - e^-theta``. ``|theta|``
    is limited to 700 to avoid overflow.

    Args:
        theta: Non-zero dependence parameter.
        dimension: Dimension of at least two.

    References:
        [REF-NELSEN-2006, Chapters 4-5], [REF-HOFERT-2008],
        [REF-KEMP-1981].

    """

    theta: float
    dimension: int = 2
    _eulerian: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    family: ClassVar[str] = "frank"

    def __post_init__(self) -> None:
        """Validate the parameter range for the dimension.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If ``theta`` is zero, out of range, or negative for
                ``d >= 3``.

        Equation:
            [EQ-MOD-COP-FRANK-001].

        References:
            [REF-NELSEN-2006, Chapter 4], [REF-HOFERT-2008].

        """
        theta = require_finite_real(self.theta, name="theta")
        dimension = _require_dimension(self.dimension, minimum=2)
        if dimension > _MAX_ARCHIMEDEAN_DIMENSION:
            raise ValueError("Archimedean dimension is limited to 100")
        if theta == 0.0 or abs(theta) > _FRANK_THETA_LIMIT:
            raise ValueError("Frank theta must be non-zero with |theta| <= 700")
        if dimension > 2 and theta < 0.0:
            raise ValueError("Frank theta must be positive in dimension three or more")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_eulerian", _eulerian_rows(dimension))

    def psi(self, t: float) -> float:
        """Return ``-log1p(-(1 - e^-theta) e^-t) / theta``.

        Args:
            t: Non-negative argument.

        Returns:
            Generator value.

        Equation:
            [EQ-MOD-COP-FRANK-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return -math.log1p(math.expm1(-self.theta) * math.exp(-t)) / self.theta

    def phi(self, u: float) -> float:
        """Return ``-log(expm1(-theta u) / expm1(-theta))``.

        For ``u > 1/2`` the form ``-log1p(e^-theta expm1(theta (1-u)) / expm1(-theta))``
        keeps relative accuracy as ``u -> 1``.

        Args:
            u: Probability in ``(0, 1]``.

        Returns:
            Inverse generator value.

        Equation:
            [EQ-MOD-COP-FRANK-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        theta = self.theta
        if u > 0.5:
            return -math.log1p(
                math.exp(-theta) * math.expm1(theta * (1.0 - u)) / math.expm1(-theta)
            )
        return -math.log(math.expm1(-theta * u) / math.expm1(-theta))

    def _log_abs_psi_derivative(self, order: int, t: float) -> float:
        """Return ``log |Li_{1-k}(x)| - log |theta|`` with ``x = (1 - e^-theta) e^-t``.

        ``Li_{-n}(x) = x P_n(x) / (1 - x)^(n+1)`` with the Eulerian polynomial
        ``P_n``; ``1 - x = -expm1(-t) + e^(-theta-t)`` is evaluated stably.

        Equation:
            [EQ-MOD-COP-FRANK-001].

        References:
            [REF-NIST-DLMF, Sections 25.12 and 26.14], [REF-HOFERT-2008].

        """
        n = order - 1
        theta = self.theta
        c = -math.expm1(-theta)
        x = c * math.exp(-t)
        one_minus_x = -math.expm1(-t) + math.exp(-theta - t)
        polynomial = math.fsum(
            coefficient * x**j for j, coefficient in enumerate(self._eulerian[n])
        )
        return (
            math.log(abs(c))
            - t
            + math.log(abs(polynomial))
            - (n + 1) * math.log(one_minus_x)
            - math.log(abs(theta))
        )

    def _log_abs_phi_prime(self, u: float) -> float:
        """Return ``log |theta| - log |expm1(theta u)|``.

        Equation:
            [EQ-MOD-COP-FRANK-001].

        References:
            [REF-NELSEN-2006, Chapter 4].

        """
        return math.log(abs(self.theta)) - math.log(abs(math.expm1(self.theta * u)))

    def _frailty(self, generator: random.Random) -> float:
        """Draw a logarithmic variate with parameter ``p = 1 - e^-theta``.

        Kemp's algorithm LK: return 1 when ``V > p``; otherwise with
        ``q = 1 - (1 - p)^U`` return ``floor(1 + log V / log q)`` if
        ``V < q^2``, 1 if ``V > q``, and 2 otherwise.

        Raises:
            ValueError: If ``theta < 0`` (only the bivariate Frank copula
                admits it, and it has no frailty representation).

        Equation:
            [EQ-MOD-COP-FRANK-002].

        References:
            [REF-KEMP-1981], [REF-DEVROYE-1986, Chapter X].

        """
        if self.theta < 0.0:
            raise ValueError("negative-theta Frank copula has no frailty")
        p = -math.expm1(-self.theta)
        v = _positive_uniform(generator)
        if v >= p:
            return 1.0
        q = -math.expm1(-self.theta * _positive_uniform(generator))
        if q <= 0.0:
            return 1.0
        if v < q * q:
            return float(math.floor(1.0 + math.log(v) / math.log(q)))
        if v > q:
            return 1.0
        return 2.0

    def _draw(self, generator: random.Random) -> Vector:
        """Draw by frailty for ``theta > 0`` and by conditional inversion otherwise.

        Equation:
            [EQ-MOD-COP-ARCH-003], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-MARSHALL-OLKIN-1988], [REF-DEVROYE-1986, Chapter XI].

        """
        if self.theta < 0.0:
            return Copula._draw(self, generator)
        return ArchimedeanCopula._draw(self, generator)

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return ``1 + 4 (D_1(theta) - 1) / theta``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Equation:
            [EQ-MOD-COP-FRANK-001], [EQ-MOD-COP-DEBYE-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 1.0 + 4.0 * _debye_minus_one_over_x(self.theta)

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return ``(0, 0)``.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Zero tail dependence.

        Equation:
            [EQ-MOD-COP-TAIL-001].

        References:
            [REF-NELSEN-2006, Chapter 5].

        """
        _pair(i, j, self.dimension)
        return 0.0, 0.0


def frank_tau(theta: float) -> float:
    """Return Kendall's tau of the Frank copula with parameter ``theta``.

    Args:
        theta: Real parameter; ``0`` gives ``0`` (independence limit).

    Returns:
        ``1 + 4 (D_1(theta) - 1) / theta``.

    Raises:
        TypeError: If ``theta`` is not real.

    Equation:
        [EQ-MOD-COP-FRANK-001], [EQ-MOD-COP-DEBYE-001].

    References:
        [REF-NELSEN-2006, Chapter 5].

    """
    value = require_finite_real(theta, name="theta")
    if value == 0.0:
        return 0.0
    return 1.0 + 4.0 * _debye_minus_one_over_x(value)


@dataclass(frozen=True, slots=True)
class NearestCorrelation:
    """Result of a nearest-correlation-matrix repair.

    Args:
        matrix: Repaired unit-diagonal positive-definite matrix.
        original: Symmetric input matrix.
        distance: Frobenius distance between ``matrix`` and ``original``.
        iterations: Alternating-projection iterations used.
        converged: Whether the relative change fell below the tolerance.
        minimum_eigenvalue: Smallest eigenvalue of ``matrix``.
        repaired: ``False`` when the input was already acceptable.

    References:
        [REF-HIGHAM-2002].

    """

    matrix: Matrix
    original: Matrix
    distance: float
    iterations: int
    converged: bool
    minimum_eigenvalue: float
    repaired: bool


def _frobenius(first: Matrix, second: Matrix) -> float:
    """Return the Frobenius norm of ``first - second``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.3].

    """
    return math.sqrt(
        math.fsum(
            (a - b) ** 2
            for row_a, row_b in zip(first, second, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
    )


def _eigen_floor_projection(matrix: Matrix, floor: float) -> tuple[Matrix, float]:
    """Project a symmetric matrix onto ``{X : lambda_min(X) >= floor}``.

    Returns:
        Projected matrix and the smallest original eigenvalue.

    Equation:
        [EQ-MOD-COP-NCM-001].

    References:
        [REF-HIGHAM-2002].

    """
    values, vectors = symmetric_eigen(matrix)
    size = len(matrix)
    clipped = tuple(max(value, floor) for value in values)
    projected = tuple(
        tuple(
            math.fsum(vectors[i][k] * clipped[k] * vectors[j][k] for k in range(size))
            for j in range(size)
        )
        for i in range(size)
    )
    return projected, min(values)


def nearest_correlation_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    eigenvalue_floor: float = 1e-8,
    tolerance: float = 1e-12,
    max_iterations: int = 2000,
) -> NearestCorrelation:
    """Return the nearest correlation matrix by Higham's alternating projections.

    Dykstra-corrected alternating projections between the unit-diagonal
    matrices and the matrices with every eigenvalue at least
    ``eigenvalue_floor`` (both convex) converge to the Frobenius-nearest
    matrix in their intersection. A floor above zero keeps the result
    strictly positive definite so that it factorizes. A matrix whose
    smallest eigenvalue already reaches the floor is returned unchanged.

    Args:
        matrix: Symmetric matrix with unit diagonal (for example pairwise
            correlation estimates that are not jointly positive definite).
        eigenvalue_floor: Minimum eigenvalue of the result, in ``[0, 1)``.
        tolerance: Relative change at which the iteration stops.
        max_iterations: Iteration limit.

    Returns:
        Repaired matrix with distance, iteration count and convergence flag.

    Raises:
        ValueError: If the matrix is not symmetric or settings are invalid.

    Equation:
        [EQ-MOD-COP-NCM-001].

    References:
        [REF-HIGHAM-2002].

    """
    original = symmetrize(matrix, name="matrix")
    floor = require_finite_real(eigenvalue_floor, name="eigenvalue_floor")
    if not 0.0 <= floor < 1.0:
        raise ValueError("eigenvalue_floor must lie in [0, 1)")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit < 1:
        raise ValueError("max_iterations must be positive")
    size = len(original)
    values, _ = symmetric_eigen(original)
    unit_diagonal = all(abs(original[i][i] - 1.0) <= 1e-12 for i in range(size))
    if unit_diagonal and min(values) >= floor:
        return NearestCorrelation(original, original, 0.0, 0, True, min(values), False)
    correction = tuple(tuple(0.0 for _ in range(size)) for _ in range(size))
    current = original
    converged = False
    iterations = 0
    zero = tuple(tuple(0.0 for _ in range(size)) for _ in range(size))
    for iterations in range(1, limit + 1):  # noqa: B007
        shifted = tuple(
            tuple(current[i][j] - correction[i][j] for j in range(size))
            for i in range(size)
        )
        projected, _ = _eigen_floor_projection(shifted, floor)
        correction = tuple(
            tuple(projected[i][j] - shifted[i][j] for j in range(size))
            for i in range(size)
        )
        updated = tuple(
            tuple(1.0 if i == j else projected[i][j] for j in range(size))
            for i in range(size)
        )
        change = _frobenius(updated, current) / _frobenius(updated, zero)
        current = updated
        if change <= tol:
            converged = True
            break
    symmetric = tuple(
        tuple(0.5 * (current[i][j] + current[j][i]) for j in range(size))
        for i in range(size)
    )
    minimum = min(symmetric_eigen(symmetric)[0])
    return NearestCorrelation(
        symmetric,
        original,
        _frobenius(symmetric, original),
        iterations,
        converged,
        minimum,
        True,
    )


__all__ = [
    "ArchimedeanCopula",
    "ClaytonCopula",
    "Copula",
    "FrankCopula",
    "GaussianCopula",
    "GumbelCopula",
    "IndependenceCopula",
    "NearestCorrelation",
    "StudentTCopula",
    "frank_tau",
    "nearest_correlation_matrix",
]
