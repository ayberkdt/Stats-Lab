"""Structured latent supports: simplex, ordered vector, correlation, covariance.

A structured site holds a vector or a matrix whose support is a manifold of
lower dimension than its number of entries (the simplex) or an open subset
of the free entries (ordered vectors, Cholesky factors of correlation
matrices, correlation and covariance matrices). Each support has a smooth
bijection from ``R^m`` onto its interior together with the log absolute
Jacobian determinant of the map from the ``m`` unconstrained coordinates to
the ``m`` free coordinates on which the site's density is defined (the
transforms of the Stan Reference Manual 2.36, "Constraint Transforms"):

========================  ==========================  =====================
support                   free coordinates            unconstrained ``m``
========================  ==========================  =====================
simplex (``K``)           ``x_1 .. x_(K-1)``          ``K - 1``
ordered (``n``)           ``x_1 .. x_n``              ``n``
Cholesky correlation      strictly lower ``L_ij``     ``d (d - 1) / 2``
correlation (``d x d``)   strictly lower ``Omega_ij`` ``d (d - 1) / 2``
covariance (``d x d``)    lower ``Sigma_ij, i >= j``  ``d (d + 1) / 2``
========================  ==========================  =====================

Every map is written with :class:`~probstat.numerics.admath` and operators
only, so it is transparent to forward- and reverse-mode automatic
differentiation. Log-densities of the library laws on these supports
(Dirichlet, LKJ, Wishart, inverse Wishart, ordered i.i.d. vectors) are
evaluated here from the quantities the transform already holds (logarithms
of simplex entries and of Cholesky diagonals), without the float validation
of the distribution classes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from probstat.distributions import RealSupport
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, cholesky
from probstat.multivariate import (
    Dirichlet,
    InverseWishart,
    LKJCholesky,
    LKJCorrelation,
    Wishart,
    multivariate_log_gamma,
)
from probstat.multivariate.dirichlet import log_multivariate_beta
from probstat.numerics import admath

from ._errors import ModelSpecificationError

StructureKind = Literal[
    "simplex", "ordered", "cholesky_correlation", "correlation", "covariance"
]
_KINDS = ("simplex", "ordered", "cholesky_correlation", "correlation", "covariance")
_MATRIX_KINDS = ("cholesky_correlation", "correlation", "covariance")
_LOG_TWO = math.log(2.0)


def _softplus(t: Any) -> Any:
    """Return ``log(1 + exp(t))`` without overflow (dual-transparent).

    ``t + log1p(exp(-t))`` for ``t > 0`` and ``log1p(exp(t))`` otherwise:
    both branches are smooth, so the derivative at ``t = 0`` is the exact
    ``1/2``. A form written with ``|t|`` took the derivative 0 that
    automatic differentiation assigns to ``|t|`` at its kink, which is the
    last stick of the simplex map at the PPL start ``u = 0``.

    References:
        [EQ-PPL-TRANSFORM-003], [REF-STAN-REFERENCE-MANUAL-2-36].

    """
    if t > 0.0:
        return t + admath.log1p(admath.exp(-t))
    return admath.log1p(admath.exp(t))


def _log_one_minus_tanh_squared(y: Any) -> Any:
    """Return ``log(1 - tanh(y)^2) = -2 log cosh(y)`` without overflow.

    ``log cosh y = |y| + log1p(exp(-2|y|)) - log 2``.

    References:
        [EQ-PPL-TRANSFORM-005], [REF-STAN-REFERENCE-MANUAL-2-36].

    """
    a = abs(y)
    return 2.0 * _LOG_TWO - 2.0 * a - 2.0 * admath.log1p(admath.exp(-2.0 * a))


@dataclass(frozen=True, slots=True)
class StructuredValue:
    """A constrained value with its log-Jacobian and transform by-products.

    Args:
        value: Vector (tuple) or matrix (tuple of rows).
        log_jacobian: ``log |det dT/du|`` onto the free coordinates.
        log_entries: Logarithms of the simplex entries (simplex only).
        factor: Lower Cholesky factor of a matrix value (matrix kinds).
        log_diagonal: ``log L_ii`` of that factor (matrix kinds).

    References:
        [EQ-PPL-TRANSFORM-003], [EQ-PPL-TRANSFORM-005],
        [EQ-PPL-TRANSFORM-006].

    """

    value: Any
    log_jacobian: Any
    log_entries: tuple[Any, ...] = ()
    factor: tuple[tuple[Any, ...], ...] = ()
    log_diagonal: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class StructuredSupport:
    """A structured support and its bijection from unconstrained space.

    ``kind`` is ``"simplex"`` (``K`` non-negative entries summing to one,
    stick-breaking), ``"ordered"`` (``n`` strictly increasing reals,
    cumulative exponentials; with ``lower`` the first entry exceeds it),
    ``"cholesky_correlation"`` (lower-triangular ``d x d`` factor with unit
    rows and positive diagonal, tanh-transformed canonical partial
    correlations), ``"correlation"`` (``L L'`` of such a factor), or
    ``"covariance"`` (symmetric positive definite, Cholesky factor with a
    log diagonal). A duck-typed distribution may set its ``support``
    attribute to an instance to be used as a structured latent site.

    Args:
        kind: Support family.
        dimension: ``K`` (simplex), ``n`` (ordered), or the matrix order ``d``.
        lower: Lower bound of an ordered vector, or ``None`` for ``R``.

    Raises:
        ValueError: If the kind is unknown, the dimension too small, or
            ``lower`` is given for a kind other than ``"ordered"``.

    References:
        [REF-STAN-REFERENCE-MANUAL-2-36], [REF-CARPENTER-ETAL-2017].

    """

    kind: StructureKind
    dimension: int
    lower: float | None = None

    def __post_init__(self) -> None:
        """Validate the kind, the dimension, and the bound.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        if self.kind not in _KINDS:
            raise ValueError(f"unknown structured support {self.kind!r}")
        size = require_non_negative_integer(self.dimension, name="dimension")
        minimum = 1 if self.kind == "ordered" else 2
        if size < minimum:
            raise ValueError(f"a {self.kind} support needs dimension >= {minimum}")
        if self.lower is not None:
            if self.kind != "ordered":
                raise ValueError("only an ordered support takes a lower bound")
            if not math.isfinite(self.lower):
                raise ValueError("the lower bound must be finite")

    # -- sizes and names ---------------------------------------------------

    @property
    def is_matrix(self) -> bool:
        """Return whether values are ``d x d`` matrices.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.kind in _MATRIX_KINDS

    @property
    def free_size(self) -> int:
        """Return the number ``m`` of unconstrained coordinates.

        References:
            [REF-STAN-REFERENCE-MANUAL-2-36]. No equation is evaluated.

        """
        d = self.dimension
        if self.kind == "simplex":
            return d - 1
        if self.kind == "ordered":
            return d
        if self.kind == "covariance":
            return d * (d + 1) // 2
        return d * (d - 1) // 2

    @property
    def constrained_size(self) -> int:
        """Return the number of reported entries (``K``, ``n``, or ``d^2``).

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.dimension * self.dimension if self.is_matrix else self.dimension

    def flat_names(self, name: str) -> tuple[str, ...]:
        """Return entry names ``name[i]`` or ``name[i,j]`` (row-major).

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        d = self.dimension
        if self.is_matrix:
            return tuple(f"{name}[{i},{j}]" for i in range(d) for j in range(d))
        return tuple(f"{name}[{i}]" for i in range(d))

    def free_names(self, name: str) -> tuple[str, ...]:
        """Return names of the unconstrained coordinates.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        d = self.dimension
        if self.kind == "simplex":
            return tuple(f"stick_logit({name})[{k}]" for k in range(d - 1))
        if self.kind == "ordered":
            if self.lower is None:
                first = f"{name}[0]"
            elif self.lower == 0.0:
                first = f"log({name}[0])"
            else:
                first = f"log({name}[0] - {self.lower:g})"
            return (first,) + tuple(
                f"log({name}[{k}] - {name}[{k - 1}])" for k in range(1, d)
            )
        if self.kind == "covariance":
            return tuple(
                f"log(chol({name})[{i},{i}])" if i == j else f"chol({name})[{i},{j}]"
                for i in range(d)
                for j in range(i + 1)
            )
        return tuple(
            f"atanh_cpc({name})[{i},{j}]" for i in range(d) for j in range(i)
        )

    # -- flattening --------------------------------------------------------

    def flatten(self, value: Any) -> tuple[float, ...]:
        """Return the entries of a value as floats (row-major for matrices).

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        if self.is_matrix:
            return tuple(float(entry) for row in value for entry in row)
        return tuple(float(entry) for entry in value)

    def unflatten(self, flat: Sequence[float]) -> Any:
        """Rebuild a vector or matrix value from :meth:`flatten` output.

        Raises:
            ValueError: If the length does not match.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        values = tuple(float(v) for v in flat)
        if len(values) != self.constrained_size:
            raise ValueError("flat value does not match the structured support")
        if not self.is_matrix:
            return values
        d = self.dimension
        return tuple(values[i * d : (i + 1) * d] for i in range(d))

    def checked_value(self, value: object, site: str) -> Any:
        """Validate the shape of a fixed value and return it as floats.

        Raises:
            ModelSpecificationError: If the shape does not match.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        d = self.dimension
        try:
            if self.is_matrix:
                rows = tuple(tuple(float(v) for v in row) for row in value)  # type: ignore[attr-defined]
                if len(rows) != d or any(len(row) != d for row in rows):
                    raise ValueError
                return rows
            vector = tuple(float(v) for v in value)  # type: ignore[attr-defined]
            if len(vector) != d:
                raise ValueError
            return vector
        except (TypeError, ValueError):
            shape = f"{d} x {d} matrix" if self.is_matrix else f"vector of length {d}"
            raise ModelSpecificationError(
                f"site {site!r}: value must be a {shape}"
            ) from None

    def free_coordinates(self, value: Any) -> tuple[Any, ...]:
        """Return the free coordinates of a value (the density's arguments).

        Simplex: the first ``K - 1`` entries; ordered: every entry; Cholesky
        correlation and correlation: the strictly lower entries row by row;
        covariance: the lower entries including the diagonal row by row.

        References:
            [REF-STAN-REFERENCE-MANUAL-2-36]. No equation is evaluated.

        """
        d = self.dimension
        if self.kind == "simplex":
            return tuple(value[: d - 1])
        if self.kind == "ordered":
            return tuple(value)
        if self.kind == "covariance":
            return tuple(value[i][j] for i in range(d) for j in range(i + 1))
        return tuple(value[i][j] for i in range(d) for j in range(i))

    # -- the bijection -----------------------------------------------------

    def forward(self, u: Sequence[Any]) -> StructuredValue:
        """Map ``m`` unconstrained coordinates to a value and its log-Jacobian.

        Args:
            u: Unconstrained coordinates (floats, duals, or reverse-mode
                variables), of length :attr:`free_size`.

        Returns:
            The constrained value, ``log |det J|`` onto the free coordinates,
            and the by-products used by the structured log-densities.

        Raises:
            ValueError: If the length of ``u`` is wrong.

        Equation:
            [EQ-PPL-TRANSFORM-003], [EQ-PPL-TRANSFORM-004],
            [EQ-PPL-TRANSFORM-005], [EQ-PPL-TRANSFORM-006].

        References:
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        if len(u) != self.free_size:
            raise ValueError("unconstrained slice has the wrong length")
        if self.kind == "simplex":
            return self._simplex(u)
        if self.kind == "ordered":
            return self._ordered(u)
        if self.kind == "covariance":
            return self._covariance(u)
        return self._correlation(u)

    def _simplex(self, u: Sequence[Any]) -> StructuredValue:
        """Stick-breaking map onto the ``K``-simplex.

        ``z_k = logistic(y_k - log(K - k))``, ``x_k = r_(k-1) z_k``,
        ``r_k = r_(k-1) (1 - z_k)``, ``x_K = r_(K-1)``; ``log |J| =
        sum_k [log z_k + log(1 - z_k) + log r_(k-1)]`` (one-based ``k``).

        References:
            [EQ-PPL-TRANSFORM-003],
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        K = self.dimension
        log_remaining: Any = 0.0
        log_entries: list[Any] = []
        jacobian: Any = 0.0
        for index, y in enumerate(u):
            t = y - math.log(K - index - 1)
            log_z = -_softplus(-t)
            log_one_minus_z = -_softplus(t)
            log_entries.append(log_remaining + log_z)
            jacobian = jacobian + log_z + log_one_minus_z + log_remaining
            log_remaining = log_remaining + log_one_minus_z
        log_entries.append(log_remaining)
        value = tuple(admath.exp(entry) for entry in log_entries)
        return StructuredValue(value, jacobian, tuple(log_entries))

    def _ordered(self, u: Sequence[Any]) -> StructuredValue:
        """Cumulative-exponential map onto strictly increasing vectors.

        ``x_1 = y_1`` (or ``a + exp(y_1)``), ``x_k = x_(k-1) + exp(y_k)``;
        ``log |J| = sum_(k >= 2) y_k`` (plus ``y_1`` with a lower bound).

        References:
            [EQ-PPL-TRANSFORM-004],
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        first = u[0]
        if self.lower is None:
            current: Any = first
            jacobian: Any = 0.0
        else:
            current = self.lower + admath.exp(first)
            jacobian = first
        values = [current]
        for y in u[1:]:
            current = current + admath.exp(y)
            jacobian = jacobian + y
            values.append(current)
        return StructuredValue(tuple(values), jacobian)

    def _correlation(self, u: Sequence[Any]) -> StructuredValue:
        """Tanh-CPC map onto Cholesky factors (and correlation matrices).

        Row ``i``: ``z_ij = tanh(y_ij)``,
        ``L_ij = z_ij sqrt(r_ij)`` with ``r_ij = 1 - sum_(j'<j) L_ij'^2``,
        ``L_ii = sqrt(r_ii)``; ``log |J| = sum log(1 - z^2)
        + (1/2) sum_(i>j) log r_ij``. The correlation kind adds the
        Jacobian of ``L -> L L'``, ``sum_k (d - k) log L_kk``.

        References:
            [EQ-PPL-TRANSFORM-005],
            [REF-STAN-REFERENCE-MANUAL-2-36],
            [REF-STAN-FUNCTIONS-REFERENCE-2-36].

        """
        d = self.dimension
        rows: list[tuple[Any, ...]] = []
        log_diagonal: list[Any] = []
        jacobian: Any = 0.0
        offset = 0
        for i in range(d):
            log_remaining: Any = 0.0
            row: list[Any] = []
            for _ in range(i):
                y = u[offset]
                offset += 1
                z = admath.tanh(y)
                row.append(z * admath.exp(0.5 * log_remaining))
                jacobian = jacobian + 0.5 * log_remaining
                log_one_minus = _log_one_minus_tanh_squared(y)
                jacobian = jacobian + log_one_minus
                log_remaining = log_remaining + log_one_minus
            log_diagonal.append(0.5 * log_remaining)
            row.append(admath.exp(0.5 * log_remaining) if i else 1.0)
            row.extend([0.0] * (d - i - 1))
            rows.append(tuple(row))
        factor = tuple(rows)
        if self.kind == "cholesky_correlation":
            return StructuredValue(
                factor, jacobian, factor=factor, log_diagonal=tuple(log_diagonal)
            )
        for i in range(1, d):
            jacobian = jacobian + (d - 1 - i) * log_diagonal[i]
        return StructuredValue(
            _gram(factor, unit_diagonal=True),
            jacobian,
            factor=factor,
            log_diagonal=tuple(log_diagonal),
        )

    def _covariance(self, u: Sequence[Any]) -> StructuredValue:
        """Cholesky map with a log diagonal onto covariance matrices.

        ``L_ii = exp(y_ii)``, ``L_ij = y_ij`` (``i > j``), ``Sigma = L L'``;
        ``log |J| = d log 2 + sum_k (d - k + 2) y_kk`` (one-based ``k``).

        References:
            [EQ-PPL-TRANSFORM-006],
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        d = self.dimension
        rows: list[tuple[Any, ...]] = []
        log_diagonal: list[Any] = []
        jacobian: Any = d * _LOG_TWO
        offset = 0
        for i in range(d):
            row: list[Any] = []
            for j in range(i + 1):
                y = u[offset]
                offset += 1
                if j == i:
                    row.append(admath.exp(y))
                    log_diagonal.append(y)
                    jacobian = jacobian + (d - i + 1) * y
                else:
                    row.append(y)
            row.extend([0.0] * (d - i - 1))
            rows.append(tuple(row))
        factor = tuple(rows)
        return StructuredValue(
            _gram(factor, unit_diagonal=False),
            jacobian,
            factor=factor,
            log_diagonal=tuple(log_diagonal),
        )

    def inverse(self, value: Any) -> tuple[float, ...]:
        """Map a value in the interior of the support to ``R^m``.

        Args:
            value: Vector or matrix of floats strictly inside the support.

        Returns:
            Unconstrained coordinates.

        Raises:
            ValueError: If the value is outside the interior of the support.

        Equation:
            [EQ-PPL-TRANSFORM-003], [EQ-PPL-TRANSFORM-004],
            [EQ-PPL-TRANSFORM-005], [EQ-PPL-TRANSFORM-006].

        References:
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        d = self.dimension
        if self.kind == "simplex":
            x = tuple(float(v) for v in value)
            if len(x) != d or any(v <= 0.0 for v in x):
                raise ValueError("value must be a point in the open simplex")
            if abs(math.fsum(x) - 1.0) > 1e-10:
                raise ValueError("simplex entries must sum to one")
            result = []
            for k in range(d - 1):
                remaining = math.fsum(x[k:])
                z = x[k] / remaining
                result.append(math.log(z) - math.log1p(-z) + math.log(d - k - 1))
            return tuple(result)
        if self.kind == "ordered":
            x = tuple(float(v) for v in value)
            if len(x) != d:
                raise ValueError("value has the wrong length")
            if any(b <= a for a, b in zip(x, x[1:], strict=False)):
                raise ValueError("value must be strictly increasing")
            if self.lower is None:
                first = x[0]
            else:
                if not x[0] > self.lower:
                    raise ValueError("first entry must exceed the lower bound")
                first = math.log(x[0] - self.lower)
            return (first,) + tuple(
                math.log(b - a) for a, b in zip(x, x[1:], strict=False)
            )
        matrix = tuple(tuple(float(v) for v in row) for row in value)
        if len(matrix) != d or any(len(row) != d for row in matrix):
            raise ValueError("value has the wrong order")
        if self.kind == "covariance":
            factor = cholesky(matrix)
            return tuple(
                math.log(factor[i][j]) if i == j else factor[i][j]
                for i in range(d)
                for j in range(i + 1)
            )
        factor = matrix if self.kind == "cholesky_correlation" else cholesky(matrix)
        result = []
        for i in range(d):
            remaining = 1.0
            for j in range(i):
                z = factor[i][j] / math.sqrt(remaining)
                if not -1.0 < z < 1.0:
                    raise ValueError("value is not inside the correlation support")
                result.append(math.atanh(z))
                remaining *= 1.0 - z * z
        return tuple(result)


def _gram(factor: tuple[tuple[Any, ...], ...], *, unit_diagonal: bool) -> Matrix:
    """Return ``L L'`` for a lower-triangular factor (dual-transparent).

    With ``unit_diagonal`` the diagonal is set to exactly ``1`` (it is ``1``
    for a factor with unit rows).

    References:
        [EQ-PPL-TRANSFORM-005], [EQ-PPL-TRANSFORM-006].

    """
    d = len(factor)
    lower: list[list[Any]] = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            if unit_diagonal and i == j:
                lower[i][j] = 1.0
                continue
            lower[i][j] = admath.fsum(factor[i][k] * factor[j][k] for k in range(j + 1))
    return tuple(
        tuple(lower[i][j] if j <= i else lower[j][i] for j in range(d))
        for i in range(d)
    )


# ---------------------------------------------------------------------------
# Ordered i.i.d. vectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderedVector:
    """The law of the sorted vector of ``size`` i.i.d. draws from ``base``.

    Its density on strictly increasing vectors is
    ``n! prod_i f(x_i)`` (the joint density of the order statistics), so
    ``OrderedVector(Normal(0, 5), 2)`` is a proper prior for the ordered
    component means of a mixture. The base law must be continuous with
    support ``R`` or ``(a, inf)``.

    Args:
        base: Continuous base distribution with ``support``, ``log_pdf``, and
            ``sample``.
        size: Vector length ``n >= 2``.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size < 2``.

    References:
        [REF-DAVID-NAGARAJA-2003], [REF-STAN-REFERENCE-MANUAL-2-36].

    """

    base: Any
    size: int

    def __post_init__(self) -> None:
        """Validate the length.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        count = require_non_negative_integer(self.size, name="size")
        if count < 2:
            raise ValueError("an ordered vector needs size >= 2")

    @property
    def support(self) -> StructuredSupport:
        """Return the ordered support (bounded below if the base law is).

        Raises:
            ModelSpecificationError: If the base support is not ``R`` or
                ``(a, inf)``.

        References:
            [EQ-PPL-TRANSFORM-004], [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        support = getattr(self.base, "support", None)
        if not isinstance(support, RealSupport) or support.upper is not None:
            raise ModelSpecificationError(
                "OrderedVector needs a continuous base law on R or (a, inf)"
            )
        return StructuredSupport("ordered", self.size, support.lower)

    def log_pdf(self, value: Sequence[float]) -> float:
        """Evaluate ``log n! + sum_i log f(x_i)`` (``-inf`` if not increasing).

        Raises:
            ValueError: If the length is wrong.

        Equation:
            [EQ-PPL-TRANSFORM-004].

        References:
            [REF-DAVID-NAGARAJA-2003].

        """
        x = tuple(value)
        if len(x) != self.size:
            raise ValueError("value has the wrong length")
        if any(b <= a for a, b in zip(x, x[1:], strict=False)):
            return -math.inf
        return math.lgamma(self.size + 1.0) + math.fsum(
            float(self.base.log_pdf(v)) for v in x
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[float, ...], ...]:
        """Draw sorted vectors of i.i.d. base draws.

        References:
            [REF-DAVID-NAGARAJA-2003]. No equation is evaluated.

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            tuple(
                sorted(
                    float(v)
                    for v in self.base.sample(self.size, seed=generator.getrandbits(63))
                )
            )
            for _ in range(count)
        )


# ---------------------------------------------------------------------------
# Dispatch from distributions to structures and densities
# ---------------------------------------------------------------------------


def structure_for(distribution: object) -> StructuredSupport | None:
    """Return the structured support of a distribution, or ``None``.

    Dirichlet laws live on the simplex, :class:`~probstat.multivariate.LKJCholesky`
    on Cholesky factors, :class:`~probstat.multivariate.LKJCorrelation` on
    correlation matrices, Wishart and inverse-Wishart laws on covariance
    matrices, and :class:`OrderedVector` on ordered vectors; any other object
    whose ``support`` is a :class:`StructuredSupport` uses that support.

    References:
        [REF-STAN-REFERENCE-MANUAL-2-36]. No equation is evaluated.

    """
    if isinstance(distribution, Dirichlet):
        return StructuredSupport("simplex", distribution.dimension)
    if isinstance(distribution, LKJCholesky):
        return StructuredSupport("cholesky_correlation", distribution.dimension)
    if isinstance(distribution, LKJCorrelation):
        return StructuredSupport("correlation", distribution.dimension)
    if isinstance(distribution, Wishart | InverseWishart):
        return StructuredSupport("covariance", distribution.dimension)
    if isinstance(distribution, OrderedVector):
        return distribution.support
    if isinstance(distribution, list | tuple):
        return None
    support = getattr(distribution, "support", None)
    return support if isinstance(support, StructuredSupport) else None


def _lower_solve_squared_norm(
    lower: Sequence[Sequence[Any]], right: Sequence[Sequence[Any]]
) -> Any:
    """Return ``||L^-1 R||_F^2`` by forward substitution (dual-transparent).

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1]. No new equation is introduced.

    """
    d = len(lower)
    total: list[Any] = []
    for column in range(len(right[0])):
        solution: list[Any] = []
        for i in range(d):
            partial = right[i][column]
            for k in range(i):
                partial = partial - lower[i][k] * solution[k]
            solution.append(partial / lower[i][i])
        total.extend(s * s for s in solution)
    return admath.fsum(total)


def structured_log_density(
    distribution: Any,
    structure: StructuredSupport,
    point: StructuredValue,
    element_log_density: Callable[[Any, Any], Any],
) -> Any:
    """Return the log-density of a structured site at a transformed point.

    The library laws are evaluated from the transform's by-products with
    ``admath`` so that the result is differentiable; other distributions are
    evaluated by their own ``log_pdf``.

    Args:
        distribution: The site's distribution.
        structure: Its structured support.
        point: Output of :meth:`StructuredSupport.forward`.
        element_log_density: Scalar log-density used for the elements of an
            :class:`OrderedVector`.

    Returns:
        The log-density (float, dual, or reverse-mode variable).

    Equation:
        [EQ-DIST-DIR-001], [EQ-DIST-LKJ-001], [EQ-DIST-LKJ-002],
        [EQ-DIST-WISH-001], [EQ-DIST-IWISH-001].

    References:
        [REF-BISHOP-2006, Appendix B], [REF-ANDERSON-2003, Section 7.2],
        [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    if isinstance(distribution, Dirichlet):
        alpha = distribution.alpha
        return admath.fsum(
            (a - 1.0) * log_x for a, log_x in zip(alpha, point.log_entries, strict=True)
        ) - log_multivariate_beta(alpha)
    if isinstance(distribution, LKJCholesky | LKJCorrelation):
        d = structure.dimension
        eta = distribution.eta
        log_c = distribution.log_normalizer
        if isinstance(distribution, LKJCholesky):
            return admath.fsum(
                (d - k - 1 + 2.0 * eta - 2.0) * point.log_diagonal[k]
                for k in range(1, d)
            ) - log_c
        return (eta - 1.0) * 2.0 * admath.fsum(point.log_diagonal) - log_c
    if isinstance(distribution, Wishart | InverseWishart):
        p = structure.dimension
        nu = distribution.dof
        scale_factor = cholesky(distribution.scale)
        log_det_scale = 2.0 * math.fsum(math.log(scale_factor[i][i]) for i in range(p))
        log_det_value = 2.0 * admath.fsum(point.log_diagonal)
        constant = 0.5 * nu * p * _LOG_TWO + multivariate_log_gamma(0.5 * nu, p)
        if isinstance(distribution, Wishart):
            trace = _lower_solve_squared_norm(scale_factor, point.factor)
            return (
                0.5 * (nu - p - 1.0) * log_det_value
                - 0.5 * trace
                - 0.5 * nu * log_det_scale
                - constant
            )
        trace = _lower_solve_squared_norm(point.factor, scale_factor)
        return (
            0.5 * nu * log_det_scale
            - 0.5 * (nu + p + 1.0) * log_det_value
            - 0.5 * trace
            - constant
        )
    if isinstance(distribution, OrderedVector):
        return math.lgamma(distribution.size + 1.0) + admath.fsum(
            element_log_density(distribution.base, x) for x in point.value
        )
    method = getattr(distribution, "log_pdf", None)
    if method is None:
        raise ModelSpecificationError(
            f"distribution {type(distribution).__name__} has no log_pdf"
        )
    return cast(Any, method(point.value))
