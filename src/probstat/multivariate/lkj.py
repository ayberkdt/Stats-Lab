"""The LKJ law on correlation matrices and on their Cholesky factors.

``Omega ~ LKJ(eta)`` on ``d x d`` correlation matrices has density
``det(Omega)^(eta - 1) / c_d(eta)`` with respect to Lebesgue measure on the
``d (d - 1) / 2`` strictly lower entries; ``eta = 1`` is uniform on the set
of correlation matrices. Writing ``Omega = L L'`` with ``L`` lower triangular
with positive diagonal gives the law of the Cholesky factor, with density
``prod_(k=2..d) L_kk^(d - k + 2 eta - 2) / c_d(eta)`` on the strictly lower
entries of ``L``. Draws use independent canonical partial correlations
``z_ij ~ 2 Beta(b_j, b_j) - 1`` (vine method), which also give the
normalizing constant as a product of Beta integrals.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, as_matrix, cholesky

from ._support import log_gamma_variate

CORRELATION_TOLERANCE = 1e-9
"""Absolute tolerance on unit diagonals, unit row norms, and symmetry."""


def _checked_parameters(dimension: object, eta: object) -> tuple[int, float]:
    """Validate the matrix order ``d >= 2`` and the shape ``eta > 0``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``d < 2`` or ``eta`` is not finite and positive.

    References:
        [EQ-DIST-LKJ-001], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    order = require_non_negative_integer(dimension, name="dimension")
    if order < 2:
        raise ValueError("dimension must be at least 2")
    shape = require_finite_real(eta, name="eta")
    if shape <= 0.0:
        raise ValueError("eta must be positive")
    return order, shape


def cpc_shape(dimension: int, column: int, eta: float) -> float:
    """Return the Beta shape ``b_j = eta + (d - 2 - j) / 2`` of column ``j``.

    The canonical partial correlation in column ``j`` (zero-based) of a
    ``d x d`` LKJ(eta) matrix is ``2 B - 1`` with ``B ~ Beta(b_j, b_j)``.

    Args:
        dimension: Matrix order ``d``.
        column: Zero-based column ``j`` with ``0 <= j <= d - 2``.
        eta: Shape ``eta > 0``.

    Returns:
        The Beta shape ``b_j``.

    Equation:
        [EQ-DIST-LKJ-003].

    References:
        [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    return eta + 0.5 * (dimension - 2 - column)


def lkj_log_normalizer(dimension: int, eta: float) -> float:
    """Return ``log c_d(eta)``, the log normalizing constant of LKJ(eta).

    ``c_d(eta) = prod_(k=1..d-1) [2^(2 b - 1) B(b, b)]^(d - k)`` with
    ``b = eta + (d - 1 - k) / 2``; equivalently
    ``2^(sum_k (2 eta - 2 + d - k)(d - k)) prod_k B(b, b)^(d - k)``. For
    ``d = 2`` it is ``int_(-1)^1 (1 - r^2)^(eta - 1) dr``; for ``d = 3`` and
    ``eta = 1`` it is ``pi^2 / 2``, the volume of the 3 x 3 correlation
    matrices.

    Args:
        dimension: Matrix order ``d >= 2``.
        eta: Shape ``eta > 0``.

    Returns:
        ``log c_d(eta)``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``d < 2`` or ``eta <= 0``.

    Equation:
        [EQ-DIST-LKJ-001].

    References:
        [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    order, shape = _checked_parameters(dimension, eta)
    terms: list[float] = []
    for k in range(1, order):
        b = shape + 0.5 * (order - 1 - k)
        log_beta = 2.0 * math.lgamma(b) - math.lgamma(2.0 * b)
        terms.append((order - k) * ((2.0 * b - 1.0) * math.log(2.0) + log_beta))
    return math.fsum(terms)


def _beta_symmetric(shape: float, generator: random.Random) -> float:
    """Draw ``2 B - 1`` with ``B ~ Beta(shape, shape)`` from two gamma variates.

    References:
        [EQ-DIST-LKJ-003], [REF-DEVROYE-1986, Chapter IX].

    """
    first = log_gamma_variate(shape, generator)
    second = log_gamma_variate(shape, generator)
    peak = max(first, second)
    a = math.exp(first - peak)
    b = math.exp(second - peak)
    return (a - b) / (a + b)


def _cholesky_from_cpcs(dimension: int, cpcs: dict[tuple[int, int], float]) -> Matrix:
    """Build the Cholesky factor of a correlation matrix from its CPCs.

    Row ``i``: ``L_ij = z_ij sqrt(1 - sum_(j' < j) L_ij'^2)`` for ``j < i`` and
    ``L_ii = sqrt(1 - sum_(j < i) L_ij^2)``.

    References:
        [EQ-PPL-TRANSFORM-005], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    rows: list[tuple[float, ...]] = []
    for i in range(dimension):
        row = [0.0] * dimension
        remaining = 1.0
        for j in range(i):
            z = cpcs[(i, j)]
            row[j] = z * math.sqrt(remaining)
            remaining *= 1.0 - z * z
        row[i] = math.sqrt(remaining)
        rows.append(tuple(row))
    return tuple(rows)


def _gram(lower: Matrix) -> Matrix:
    """Return ``L L'`` with an exact unit diagonal for a unit-row-norm ``L``.

    References:
        [EQ-DIST-LKJ-002], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(lower)
    rows = []
    for i in range(size):
        rows.append(
            tuple(
                1.0
                if i == j
                else math.fsum(lower[i][k] * lower[j][k] for k in range(min(i, j) + 1))
                for j in range(size)
            )
        )
    return tuple(rows)


def _draw_factors(
    dimension: int, eta: float, size: int, seed: int | RandomStream | None
) -> tuple[Matrix, ...]:
    """Draw ``size`` LKJ Cholesky factors by independent Beta CPCs.

    References:
        [EQ-DIST-LKJ-003], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

    """
    count = require_non_negative_integer(size, name="size")
    generator = resolve_stream(seed).generator()
    draws: list[Matrix] = []
    for _ in range(count):
        cpcs = {
            (i, j): _beta_symmetric(cpc_shape(dimension, j, eta), generator)
            for i in range(dimension)
            for j in range(i)
        }
        draws.append(_cholesky_from_cpcs(dimension, cpcs))
    return tuple(draws)


@dataclass(frozen=True, slots=True)
class LKJCholesky:
    """The LKJ(eta) law of the Cholesky factor ``L`` of a correlation matrix.

    The support is the set of lower-triangular ``d x d`` matrices with a
    positive diagonal and unit-norm rows (``L L'`` has a unit diagonal); the
    density is taken on the ``d (d - 1) / 2`` strictly lower entries.

    Args:
        dimension: Matrix order ``d >= 2``.
        eta: Shape ``eta > 0`` (``1`` is uniform over correlation matrices;
            larger values concentrate on the identity).

    References:
        [REF-LEWANDOWSKI-KUROWICKA-JOE-2009],
        [REF-STAN-FUNCTIONS-REFERENCE-2-36].

    """

    dimension: int
    eta: float = 1.0

    def __post_init__(self) -> None:
        """Validate the order and the shape.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If ``d < 2`` or ``eta <= 0``.

        References:
            [EQ-DIST-LKJ-002], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        order, shape = _checked_parameters(self.dimension, self.eta)
        object.__setattr__(self, "dimension", order)
        object.__setattr__(self, "eta", shape)

    @property
    def log_normalizer(self) -> float:
        """Return ``log c_d(eta)``.

        References:
            [EQ-DIST-LKJ-001], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        return lkj_log_normalizer(self.dimension, self.eta)

    def log_pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate the log-density at a Cholesky factor of a correlation matrix.

        ``log f(L) = sum_(k=2..d) (d - k + 2 eta - 2) log L_kk - log c_d``
        (one-based ``k``). A lower-triangular matrix with unit-norm rows but a
        non-positive diagonal entry is outside the support (``-inf``).

        Args:
            value: Lower-triangular ``d x d`` matrix.

        Returns:
            ``log f(L)``.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the order is wrong, an entry above the diagonal is
                non-zero, or a row does not have unit norm.

        Equation:
            [EQ-DIST-LKJ-002].

        References:
            [REF-STAN-FUNCTIONS-REFERENCE-2-36],
            [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        lower = as_matrix(value, name="value")
        d = self.dimension
        if len(lower) != d or any(len(row) != d for row in lower):
            raise ValueError("value order must equal the distribution dimension")
        for i in range(d):
            if any(lower[i][j] != 0.0 for j in range(i + 1, d)):
                raise ValueError("value must be lower triangular")
            norm = math.fsum(entry * entry for entry in lower[i][: i + 1])
            if abs(norm - 1.0) > CORRELATION_TOLERANCE:
                raise ValueError("rows of a correlation Cholesky factor have unit norm")
        if any(lower[i][i] <= 0.0 for i in range(d)):
            return -math.inf
        return math.fsum(
            (d - k - 1 + 2.0 * self.eta - 2.0) * math.log(lower[k][k])
            for k in range(1, d)
        ) - self.log_normalizer

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Matrix, ...]:
        """Draw reproducible Cholesky factors by independent Beta CPCs.

        Args:
            size: Number of factors, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of lower-triangular factors.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-LKJ-003], [EQ-PPL-TRANSFORM-005].

        References:
            [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        return _draw_factors(self.dimension, self.eta, size, seed)


@dataclass(frozen=True, slots=True)
class LKJCorrelation:
    """The LKJ(eta) law on ``d x d`` correlation matrices.

    Args:
        dimension: Matrix order ``d >= 2``.
        eta: Shape ``eta > 0``.

    References:
        [REF-LEWANDOWSKI-KUROWICKA-JOE-2009],
        [REF-STAN-FUNCTIONS-REFERENCE-2-36].

    """

    dimension: int
    eta: float = 1.0

    def __post_init__(self) -> None:
        """Validate the order and the shape.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If ``d < 2`` or ``eta <= 0``.

        References:
            [EQ-DIST-LKJ-001], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        order, shape = _checked_parameters(self.dimension, self.eta)
        object.__setattr__(self, "dimension", order)
        object.__setattr__(self, "eta", shape)

    @property
    def log_normalizer(self) -> float:
        """Return ``log c_d(eta)``.

        References:
            [EQ-DIST-LKJ-001], [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        return lkj_log_normalizer(self.dimension, self.eta)

    def log_pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate ``(eta - 1) log det(Omega) - log c_d`` at a correlation matrix.

        A symmetric matrix with unit diagonal that is not positive definite
        is outside the support (``-inf``).

        Args:
            value: Symmetric ``d x d`` matrix with unit diagonal.

        Returns:
            ``log f(Omega)``.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the order is wrong, the matrix is not symmetric,
                or a diagonal entry differs from one.

        Equation:
            [EQ-DIST-LKJ-001].

        References:
            [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        matrix = as_matrix(value, name="value")
        d = self.dimension
        if len(matrix) != d or any(len(row) != d for row in matrix):
            raise ValueError("value order must equal the distribution dimension")
        for i in range(d):
            if abs(matrix[i][i] - 1.0) > CORRELATION_TOLERANCE:
                raise ValueError("a correlation matrix has a unit diagonal")
            for j in range(i):
                if abs(matrix[i][j] - matrix[j][i]) > CORRELATION_TOLERANCE:
                    raise ValueError("a correlation matrix is symmetric")
        try:
            lower = cholesky(matrix)
        except ValueError:
            return -math.inf
        log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(d))
        return (self.eta - 1.0) * log_det - self.log_normalizer

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Matrix, ...]:
        """Draw reproducible correlation matrices ``L L'`` from LKJ Cholesky factors.

        Args:
            size: Number of matrices, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of correlation matrices.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-LKJ-003].

        References:
            [REF-LEWANDOWSKI-KUROWICKA-JOE-2009].

        """
        return tuple(
            _gram(factor)
            for factor in _draw_factors(self.dimension, self.eta, size, seed)
        )
