"""Validation and matrix helpers shared by the multivariate probability laws.

The helpers build on :mod:`probstat.linalg` (Cholesky factors, triangular
solves) and add only what the vector- and matrix-valued laws need in common:
positive parameter vectors, count vectors, symmetric positive-definite
parameters with their Cholesky factor, log-determinants and traces of
``A^-1 B`` from a factor, and the log-space gamma variates used by the
Dirichlet and Wishart samplers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from numbers import Integral

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    symmetrize,
)

SIMPLEX_TOLERANCE = 1e-10
"""Absolute tolerance on ``|sum(x) - 1|`` for a point or vector on the simplex."""


def positive_vector(
    values: Iterable[object], *, name: str, minimum_length: int = 1
) -> Vector:
    """Validate a vector of finite, strictly positive components.

    Args:
        values: Candidate components.
        name: Argument name used in error messages.
        minimum_length: Smallest admissible number of components.

    Returns:
        Tuple of positive floats.

    Raises:
        TypeError: If ``values`` is not iterable or holds a non-real entry.
        ValueError: If an entry is non-finite or not positive, or the vector
            is shorter than ``minimum_length``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    vector = as_vector(values, name=name)
    if len(vector) < minimum_length:
        raise ValueError(f"{name} must have at least {minimum_length} components")
    for index, value in enumerate(vector):
        if value <= 0.0:
            raise ValueError(f"{name}[{index}] must be positive")
    return vector


def probability_vector(values: Iterable[object], *, name: str) -> Vector:
    """Validate a probability vector with at least two categories.

    Components must lie in ``[0, 1]`` and sum to one within
    :data:`SIMPLEX_TOLERANCE`; the returned vector is divided by its
    compensated sum so that it sums to one to rounding.

    Args:
        values: Candidate category probabilities.
        name: Argument name used in error messages.

    Returns:
        Normalized probability vector.

    Raises:
        TypeError: If ``values`` is not iterable or holds a non-real entry.
        ValueError: If there are fewer than two entries, an entry is outside
            ``[0, 1]``, or the entries do not sum to one.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35]. No equation is
        evaluated.

    """
    vector = as_vector(values, name=name)
    if len(vector) < 2:
        raise ValueError(f"{name} must have at least two categories")
    for index, value in enumerate(vector):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{index}] must lie in [0, 1]")
    total = math.fsum(vector)
    if abs(total - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"{name} must sum to one")
    return tuple(value / total for value in vector)


def count_vector(
    values: Iterable[object], *, name: str, length: int
) -> tuple[int, ...]:
    """Validate a vector of category counts of known length.

    Integers and integral floats are accepted; booleans and non-integral
    values are rejected, as for univariate discrete laws (ADR-0059).
    Negative entries are returned unchanged so that a mass function can
    report them as outside the support.

    Args:
        values: Candidate counts.
        name: Argument name used in error messages.
        length: Required number of categories.

    Returns:
        Tuple of Python integers.

    Raises:
        TypeError: If ``values`` is not iterable or holds a non-integral entry.
        ValueError: If the length differs from ``length``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    if len(raw) != length:
        raise ValueError(f"{name} must have {length} entries")
    counts: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise TypeError(f"{name}[{index}] must be an integer")
        if isinstance(value, Integral):
            counts.append(int(value))
        elif isinstance(value, float) and value.is_integer():
            counts.append(int(value))
        else:
            raise TypeError(f"{name}[{index}] must be an integer")
    return tuple(counts)


def spd_matrix(
    values: Iterable[Iterable[object]], *, name: str
) -> tuple[Matrix, Matrix]:
    """Validate a symmetric positive-definite matrix and factor it.

    Args:
        values: Square finite matrix, symmetric to relative ``1e-12``.
        name: Argument name used in error messages.

    Returns:
        ``(A, L)``: the averaged symmetric matrix and its lower Cholesky factor.

    Raises:
        TypeError: If rows or entries have invalid types.
        ValueError: If the matrix is empty, ragged, not square, not
            symmetric, or not positive definite.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    matrix = symmetrize(values, name=name)
    try:
        lower = cholesky(matrix)
    except ValueError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix, lower


def log_det_from_factor(lower: Matrix) -> float:
    """Return ``log det A`` from the lower Cholesky factor of ``A``.

    Args:
        lower: Lower Cholesky factor with positive diagonal.

    Returns:
        ``2 sum_i log L_ii``.

    Equation:
        [EQ-LINALG-006].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    return 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))


def trace_of_solve(lower: Matrix, matrix: Matrix) -> float:
    """Return ``tr(A^-1 B)`` from the Cholesky factor of ``A``.

    Column ``j`` of ``A^-1 B`` is ``A^-1 b_j``; only its ``j``-th entry
    enters the trace. No inverse is formed.

    Args:
        lower: Lower Cholesky factor of the symmetric positive-definite ``A``.
        matrix: Square ``B`` of the same order.

    Returns:
        The trace ``sum_j (A^-1 b_j)_j``.

    Equation:
        [EQ-LINALG-002], [EQ-LINALG-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    size = len(lower)
    return math.fsum(
        cholesky_solve(lower, tuple(matrix[row][column] for row in range(size)))[column]
        for column in range(size)
    )


def lower_inverse(lower: Matrix) -> Matrix:
    """Return ``L^-1`` for a non-singular lower-triangular ``L``.

    Args:
        lower: Lower-triangular matrix with non-zero diagonal.

    Returns:
        The lower-triangular inverse, built row by row by forward
        substitution against the unit vectors.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    size = len(lower)
    inverse = [[0.0] * size for _ in range(size)]
    for column in range(size):
        for row in range(column, size):
            target = 1.0 if row == column else 0.0
            residual = target - math.fsum(
                lower[row][inner] * inverse[inner][column]
                for inner in range(column, row)
            )
            inverse[row][column] = residual / lower[row][row]
    return tuple(tuple(row) for row in inverse)


def gram_lower(lower: Matrix) -> Matrix:
    """Return the symmetric product ``L L'`` of a lower-triangular ``L``.

    Args:
        lower: Lower-triangular matrix.

    Returns:
        Exactly symmetric ``L L'`` with compensated inner products.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(lower)
    result = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = math.fsum(
                lower[row][inner] * lower[column][inner] for inner in range(column + 1)
            )
            result[row][column] = value
            result[column][row] = value
    return tuple(tuple(row) for row in result)


def lower_product(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two lower-triangular matrices of the same order.

    Args:
        left: Lower-triangular ``A``.
        right: Lower-triangular ``B``.

    Returns:
        Lower-triangular ``A B``; entry ``(i, j)`` sums ``k = j..i``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    size = len(left)
    return tuple(
        tuple(
            math.fsum(
                left[row][inner] * right[inner][column]
                for inner in range(column, row + 1)
            )
            if column <= row
            else 0.0
            for column in range(size)
        )
        for row in range(size)
    )


def scaled(matrix: Matrix, factor: float) -> Matrix:
    """Multiply every entry of a matrix by one real factor.

    Infinite and NaN factors are allowed so that undefined or divergent
    moments can be represented entrywise.

    Args:
        matrix: Rectangular matrix.
        factor: Real multiplier.

    Returns:
        ``factor * matrix``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(tuple(factor * value for value in row) for row in matrix)


def divergent_matrix(size: int) -> Matrix:
    """Return a moment matrix with ``inf`` diagonal and ``nan`` off-diagonal.

    This is the convention for a second moment whose diagonal integrals
    diverge while the off-diagonal products are not absolutely integrable
    (ADR-0059: divergent integrals are ``inf``, undefined ones ``nan``).

    Args:
        size: Matrix order.

    Returns:
        The ``size`` by ``size`` marker matrix.

    References:
        [REF-KOTZ-NADARAJAH-2004, Chapter 1]. No equation is evaluated.

    """
    return tuple(
        tuple(math.inf if row == column else math.nan for column in range(size))
        for row in range(size)
    )


def nan_matrix(size: int) -> Matrix:
    """Return a ``size`` by ``size`` matrix of ``nan`` (undefined moment).

    Args:
        size: Matrix order.

    Returns:
        Matrix whose every entry is ``nan``.

    References:
        Project moment convention [REF-PROBSTAT-API]. No equation is evaluated.

    """
    return tuple(tuple(math.nan for _ in range(size)) for _ in range(size))


def positive_real(value: object, *, name: str) -> float:
    """Validate a finite, strictly positive real number.

    Args:
        value: Candidate value.
        name: Argument name used in error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is non-finite or not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def log_gamma_variate(shape: float, generator: random.Random) -> float:
    """Return ``log G`` for ``G ~ Gamma(shape, 1)`` without underflow.

    For ``shape >= 1`` the standard-library variate is used directly. For
    ``shape < 1`` the boost ``G_a = G_(a+1) U^(1/a)`` with ``U`` uniform on
    ``(0, 1]`` is applied in log space, so that very small shapes give a
    finite logarithm even when ``G`` itself underflows.

    Args:
        shape: Finite positive shape ``a``.
        generator: Call-local pseudo-random generator.

    Returns:
        Natural logarithm of one gamma variate.

    Equation:
        [EQ-DIST-DIR-006].

    References:
        [REF-MARSAGLIA-TSANG-2000], [REF-DEVROYE-1986, Chapter IX].

    """
    if shape < 1.0:
        boosted = generator.gammavariate(shape + 1.0, 1.0)
        uniform = 1.0 - generator.random()
        return math.log(boosted) + math.log(uniform) / shape
    return math.log(generator.gammavariate(shape, 1.0))


def chi_square_variate(degrees: float, generator: random.Random) -> float:
    """Return one chi-square variate with real ``degrees > 0``.

    Args:
        degrees: Degrees of freedom.
        generator: Call-local pseudo-random generator.

    Returns:
        ``2 G`` with ``G ~ Gamma(degrees / 2, 1)``.

    References:
        [REF-DEVROYE-1986, Chapter IX]. The chi-square law is the gamma law
        with shape ``nu/2`` and scale 2; no new equation is evaluated.

    """
    return 2.0 * generator.gammavariate(0.5 * degrees, 1.0)
