"""Data-driven model reduction and system identification.

Experimental (added in 0.28).

Five related tools that turn snapshot data of a dynamical system into a small
model, all in pure Python on top of :mod:`probstat.linalg`:

1. **Proper orthogonal decomposition** (:func:`fit_pod`): the leading right
   singular vectors of the snapshot matrix span the best rank-``r`` subspace
   in the mean-square sense. The rank follows an explicit rule: either a
   given ``rank`` or the smallest ``r`` whose singular values capture a
   stated fraction ``energy`` of ``sum sigma_i^2`` [EQ-SUR-ROM-001]. The
   Frobenius reconstruction error equals the discarded tail energy
   [EQ-SUR-ROM-002].
2. **Intrusive POD-Galerkin** (:func:`galerkin_reduced_model`): when the
   full-order operators of ``x' = c + A x + Q(x, x) + B u`` are available the
   reduced operators are the projections ``V' A V`` and so on
   [EQ-SUR-ROM-003].
3. **Operator inference** (:func:`fit_operator_inference`): the same reduced
   operators are learned from projected snapshots and time derivatives by
   Tikhonov-regularized least squares, without access to the full-order
   operators [EQ-SUR-ROM-004]. :func:`select_operator_inference_regularization`
   picks the Tikhonov weight by held-out trajectory error and rejects
   candidates whose simulated reduced model diverges [EQ-SUR-ROM-005].
   Reduced models are stepped by classical Runge-Kutta with an explicit
   divergence status instead of a silent overflow [EQ-SUR-ROM-006].
4. **Dynamic mode decomposition** (:func:`fit_dmd`, exact DMD with rank
   truncation) [EQ-SUR-ROM-007] and **extended DMD** (:func:`fit_edmd`) with
   a polynomial, radial-basis or custom dictionary [EQ-SUR-ROM-008]. The
   eigenvalues of the non-symmetric reduced operator come from a complex
   shifted QR iteration [EQ-SUR-ROM-012].
5. **SINDy** (:func:`fit_sindy`): sparse regression on a library of candidate
   terms by sequentially thresholded least squares with an explicit threshold
   [EQ-SUR-ROM-010], derivative estimates by finite differences or a local
   polynomial fit [EQ-SUR-ROM-009], and :func:`fit_ensemble_sindy`, which
   bootstraps the data (or the library) to report inclusion probabilities and
   coefficient intervals [EQ-SUR-ROM-011].

Assumptions and failure reporting:
    Snapshots are real, finite and sampled at a uniform step where a step is
    needed. Nothing is regularized silently: a rank-deficient regression
    raises ``ValueError`` unless an explicit ridge weight is given. A reduced
    model that blows up reports ``completed=False`` with a ``stop_reason``;
    operator inference and DMD give no stability guarantee, and a stable
    full-order system can produce an unstable learned model.

Common pitfalls:
    - Operator inference from noisy or finite-difference derivatives is
      biased; the noise-free recovery of the Galerkin operators holds only
      when the trajectory lies in the span of the basis and the data matrix
      has full column rank.
    - SINDy thresholds act on coefficients in the units of the library
      terms, so a threshold chosen for one scaling of the state is not
      transferable to another.
    - Inclusion probabilities of ensemble SINDy are properties of the
      resampling scheme, not posterior probabilities.

Guide: docs/research/fragments/reduced-order.md

References:
    [REF-BRUNTON-PROCTOR-KUTZ-2016] (SINDy), [REF-SCHMID-2010] (DMD),
    [REF-TU-ETAL-2014] (exact DMD), [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015]
    (EDMD), [REF-PEHERSTORFER-WILLCOX-2016] (operator inference),
    [REF-FASEL-ETAL-2022] (ensemble SINDy), [REF-SIROVICH-1987] (snapshot
    POD), [REF-ECKART-YOUNG-1936] (best low-rank approximation),
    [REF-GOLUB-VANLOAN-2013] (least squares, eigenvalues).

"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    householder_qr,
    singular_value_decomposition,
    solve_spd,
)

__all__ = [
    "DMDResult",
    "EDMDResult",
    "EnsembleSINDyResult",
    "FeatureDictionary",
    "OperatorInferenceResult",
    "PODBasis",
    "ROMTrajectory",
    "ReducedModel",
    "RegularizationSelection",
    "SINDyResult",
    "custom_dictionary",
    "eigen_general",
    "finite_difference_derivatives",
    "fit_dmd",
    "fit_edmd",
    "fit_ensemble_sindy",
    "fit_operator_inference",
    "fit_pod",
    "fit_sindy",
    "galerkin_reduced_model",
    "local_polynomial_derivatives",
    "polynomial_dictionary",
    "quadratic_features",
    "rbf_dictionary",
    "select_operator_inference_regularization",
]

ComplexVector = tuple[complex, ...]
Bilinear = Callable[[Vector, Vector], Vector]

# Computational choices (not from the literature): QR iteration gives up after
# this many sweeps per eigenvalue, and inverse iteration shifts the eigenvalue
# by this multiple of the Frobenius norm so the shifted matrix is not exactly
# singular.
_MAX_QR_SWEEPS = 100
_INVERSE_ITERATION_SHIFT = 1e-10
_INVERSE_ITERATION_STEPS = 3
_EPS = 2.0**-52


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _integer(value: object, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    Args:
        value: Candidate integer (booleans are rejected).
        name: Argument name for messages.
        minimum: Smallest allowed value.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is below ``minimum``.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    Args:
        value: Candidate real number.
        name: Argument name for messages.

    Returns:
        The validated value.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite and positive.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative float.

    Args:
        value: Candidate real number.
        name: Argument name for messages.

    Returns:
        The validated value.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is negative or non-finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the compensated inner product of two equal-length sequences.

    Args:
        left: First sequence.
        right: Second sequence.

    Returns:
        ``sum_i left_i right_i``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _frobenius(rows: Sequence[Sequence[float]]) -> float:
    """Return the Frobenius norm of a matrix given as rows.

    Args:
        rows: Rows of real numbers.

    Returns:
        ``sqrt(sum_ij a_ij^2)``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 2]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(v * v for row in rows for v in row))


def _transpose(rows: Sequence[Sequence[float]]) -> Matrix:
    """Return the transpose of a non-empty rectangular matrix.

    Args:
        rows: Rows of equal length.

    Returns:
        Tuple of column tuples.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    return tuple(tuple(column) for column in zip(*rows, strict=True))


def _snapshot_matrix(values: object, name: str, *, minimum: int = 1) -> Matrix:
    """Validate time-ordered snapshots as a matrix with one snapshot per row.

    Args:
        values: Iterable of equal-length state vectors.
        name: Argument name for messages.
        minimum: Smallest allowed number of snapshots.

    Returns:
        Validated tuple of state tuples.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the data are empty, ragged, non-finite or too short.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    matrix = as_matrix(values, name=name)  # type: ignore[arg-type]
    if len(matrix) < minimum:
        raise ValueError(f"{name} needs at least {minimum} snapshots")
    return matrix


def _ridge_least_squares(
    design: Matrix, targets: Matrix, ridge: Sequence[float]
) -> Matrix:
    """Solve ``min ||D X - T||^2 + sum_j ridge_j ||X_j.||^2`` by Householder QR.

    The ridge enters as extra rows ``sqrt(ridge_j) e_j'`` appended to the
    design, so the conditioning is that of ``D`` and not of ``D'D``.

    Args:
        design: ``K x p`` design matrix.
        targets: ``K x q`` right-hand sides, one per column.
        ridge: ``p`` non-negative Tikhonov weights.

    Returns:
        ``p x q`` coefficient matrix.

    Raises:
        ValueError: If the (augmented) design is rank deficient or has more
            columns than rows.

    Equation:
        [EQ-SUR-ROM-004].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.3], [REF-TIKHONOV-1963].

    """
    p = len(design[0])
    extra = [j for j in range(p) if ridge[j] > 0.0]
    rows = list(design)
    for j in extra:
        row = [0.0] * p
        row[j] = math.sqrt(ridge[j])
        rows.append(tuple(row))
    factor = householder_qr(tuple(rows))
    if not factor.full_rank:
        raise ValueError(
            "regression is rank deficient: add data or use a positive "
            "regularization weight"
        )
    q = len(targets[0])
    pad = (0.0,) * len(extra)
    columns = []
    for k in range(q):
        rhs = tuple(row[k] for row in targets) + pad
        columns.append(factor.solve(rhs))
    return tuple(tuple(columns[k][j] for k in range(q)) for j in range(p))


# ---------------------------------------------------------------------------
# Polynomial features and dictionaries
# ---------------------------------------------------------------------------


def quadratic_features(state: Sequence[float]) -> Vector:
    """Return the compact quadratic features ``a_i a_j`` for ``i <= j``.

    The Kronecker product ``a (x) a`` repeats every cross term; keeping only
    ``i <= j`` removes the non-uniqueness of the quadratic operator
    [EQ-SUR-ROM-003]. The order is ``(0,0), (0,1), ..., (0,r-1), (1,1), ...``.

    Args:
        state: Reduced state ``a`` of length ``r``.

    Returns:
        Tuple of ``r (r + 1) / 2`` products.

    Equation:
        [EQ-SUR-ROM-003].

    References:
        [REF-PEHERSTORFER-WILLCOX-2016].

    """
    return tuple(
        state[i] * state[j] for i in range(len(state)) for j in range(i, len(state))
    )


def _monomial(exponents: tuple[int, ...], state: Vector) -> float:
    """Return ``prod_i state_i^exponents_i``.

    Args:
        exponents: Non-negative integer exponent per coordinate.
        state: State vector.

    Returns:
        The monomial value.

    References:
        [EQ-SUR-ROM-010]. Monomials are the SINDy library terms.

    """
    value = 1.0
    for x, e in zip(state, exponents, strict=True):
        if e:
            value *= x**e
    return value


def _gaussian_bump(center: Vector, width: float, state: Vector) -> float:
    """Return ``exp(-||state - center||^2 / (2 width^2))``.

    Args:
        center: Radial-basis centre.
        width: Positive length scale.
        state: State vector.

    Returns:
        The Gaussian radial basis function value.

    References:
        [EQ-SUR-ROM-008]. Radial dictionary of EDMD.

    """
    squared = math.fsum((x - c) ** 2 for x, c in zip(state, center, strict=True))
    return math.exp(-squared / (2.0 * width * width))


def _constant_one(state: Vector) -> float:
    """Return the constant observable ``1``.

    Args:
        state: Ignored state vector.

    Returns:
        ``1.0``.

    References:
        [EQ-SUR-ROM-008]. The constant observable has Koopman eigenvalue one.

    """
    return 1.0


def _exponent_tuples(dimension: int, degree: int) -> list[tuple[int, ...]]:
    """List exponent tuples of total degree at most ``degree``, graded order.

    Args:
        dimension: Number of coordinates.
        degree: Maximum total degree.

    Returns:
        Exponent tuples ordered by total degree, then reverse-lexicographic.

    References:
        [EQ-SUR-ROM-010]. Library of polynomial candidate terms.

    """
    result: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        stack: list[tuple[int, ...]] = [()]
        for position in range(dimension):
            nxt: list[tuple[int, ...]] = []
            for prefix in stack:
                used = sum(prefix)
                if position == dimension - 1:
                    nxt.append((*prefix, total - used))
                else:
                    nxt.extend((*prefix, e) for e in range(total - used, -1, -1))
            stack = nxt
        result.extend(stack)
    return result


@dataclass(frozen=True, slots=True)
class FeatureDictionary:
    """Finite dictionary of scalar observables or library terms.

    Used as the EDMD dictionary ``psi(x) = (psi_1(x), ..., psi_N(x))``
    [EQ-SUR-ROM-008] and as the SINDy library ``Theta`` [EQ-SUR-ROM-010].

    Attributes:
        dimension: State dimension the functions accept.
        functions: The ``N`` scalar functions of a state vector.
        names: Human-readable name of each function.

    References:
        [EQ-SUR-ROM-008], [EQ-SUR-ROM-010].

    """

    dimension: int
    functions: tuple[Callable[[Vector], float], ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the dimension and the matching of functions and names.

        Raises:
            TypeError: If the dimension is not an integer.
            ValueError: If the dictionary is empty or names mismatch.

        References:
            [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

        """
        _integer(self.dimension, "dimension", 1)
        if not self.functions:
            raise ValueError("dictionary needs at least one function")
        if len(self.functions) != len(self.names):
            raise ValueError("functions and names must have the same length")

    @property
    def size(self) -> int:
        """Return the number ``N`` of dictionary functions.

        Returns:
            Number of functions.

        References:
            [EQ-SUR-ROM-008]. No equation is evaluated.

        """
        return len(self.functions)

    def __call__(self, state: Sequence[float]) -> Vector:
        """Evaluate every function at one state.

        Args:
            state: State vector of length ``dimension``.

        Returns:
            The lifted vector ``psi(state)``.

        Raises:
            ValueError: If the state has the wrong length.

        References:
            [EQ-SUR-ROM-008].

        """
        x = as_vector(state, name="state")
        if len(x) != self.dimension:
            raise ValueError("state length must equal the dictionary dimension")
        return tuple(float(f(x)) for f in self.functions)

    def evaluate(self, states: Sequence[Sequence[float]]) -> Matrix:
        """Evaluate the dictionary at every row of a snapshot matrix.

        Args:
            states: Snapshots, one state per row.

        Returns:
            ``m x N`` matrix whose row ``k`` is ``psi(states[k])``.

        Raises:
            ValueError: If a state has the wrong length or a value is not
                finite.

        References:
            [EQ-SUR-ROM-008], [EQ-SUR-ROM-010].

        """
        rows = tuple(self(state) for state in states)
        return as_matrix(rows, name="dictionary values")


def polynomial_dictionary(
    dimension: int, degree: int, *, include_constant: bool = True
) -> FeatureDictionary:
    """Return all monomials of total degree at most ``degree``.

    Args:
        dimension: State dimension.
        degree: Maximum total degree (at least one).
        include_constant: Whether the constant term is included.

    Returns:
        Dictionary of ``C(dimension + degree, degree)`` monomials (one fewer
        without the constant), named like ``"x0^2 x1"``.

    Raises:
        TypeError: If an integer argument is not an integer.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-SUR-ROM-010].

    References:
        [REF-BRUNTON-PROCTOR-KUTZ-2016], [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015].

    """
    n = _integer(dimension, "dimension", 1)
    d = _integer(degree, "degree", 1)
    exponents = _exponent_tuples(n, d)
    if not include_constant:
        exponents = exponents[1:]
    names = []
    for e in exponents:
        parts = [f"x{i}" if k == 1 else f"x{i}^{k}" for i, k in enumerate(e) if k]
        names.append(" ".join(parts) if parts else "1")
    functions = tuple(partial(_monomial, e) for e in exponents)
    return FeatureDictionary(n, functions, tuple(names))


def rbf_dictionary(
    centers: Sequence[Sequence[float]],
    width: float,
    *,
    include_constant: bool = True,
    include_linear: bool = True,
) -> FeatureDictionary:
    """Return Gaussian radial basis functions, optionally with ``1`` and ``x``.

    The observables are ``exp(-||x - c_k||^2 / (2 width^2))``; the constant
    and the coordinates are appended because the state must lie in the span of
    the dictionary for EDMD to reconstruct it.

    Args:
        centers: RBF centres, one per row.
        width: Positive length scale.
        include_constant: Add the constant observable first.
        include_linear: Add the coordinates ``x_i`` after the bumps.

    Returns:
        Dictionary of the requested functions.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If centres are empty or ragged or ``width`` is not positive.

    Equation:
        [EQ-SUR-ROM-008].

    References:
        [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015].

    """
    grid = as_matrix(centers, name="centers")
    w = _positive(width, "width")
    dimension = len(grid[0])
    functions: list[Callable[[Vector], float]] = []
    names: list[str] = []
    if include_constant:
        functions.append(_constant_one)
        names.append("1")
    for index, center in enumerate(grid):
        functions.append(partial(_gaussian_bump, center, w))
        names.append(f"rbf{index}")
    if include_linear:
        for i in range(dimension):
            functions.append(
                partial(_monomial, tuple(int(i == j) for j in range(dimension)))
            )
            names.append(f"x{i}")
    return FeatureDictionary(dimension, tuple(functions), tuple(names))


def custom_dictionary(
    dimension: int,
    functions: Sequence[Callable[[Vector], float]],
    names: Sequence[str],
) -> FeatureDictionary:
    """Wrap user functions of a state vector as a dictionary.

    Args:
        dimension: State dimension.
        functions: Scalar functions ``psi_k(x)``.
        names: One name per function.

    Returns:
        The dictionary.

    Raises:
        ValueError: If the lists are empty or of different length.

    Equation:
        [EQ-SUR-ROM-008].

    References:
        [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015].

    """
    return FeatureDictionary(dimension, tuple(functions), tuple(names))


# ---------------------------------------------------------------------------
# General (non-symmetric) real eigenproblem
# ---------------------------------------------------------------------------


def _hessenberg(matrix: list[list[float]]) -> list[list[float]]:
    """Reduce a real square matrix to upper Hessenberg form by reflections.

    Args:
        matrix: Square matrix (modified in place).

    Returns:
        The Hessenberg matrix, similar to the input.

    Equation:
        [EQ-SUR-ROM-012].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 7.4].

    """
    n = len(matrix)
    a = matrix
    for k in range(n - 2):
        x = [a[i][k] for i in range(k + 1, n)]
        alpha = math.hypot(*x)
        if alpha == 0.0:
            continue
        alpha = -math.copysign(alpha, x[0])
        x[0] -= alpha
        scale = math.hypot(*x)
        if scale == 0.0:
            continue
        v = [value / scale for value in x]
        for j in range(n):
            s = math.fsum(v[i] * a[k + 1 + i][j] for i in range(len(v)))
            for i, vi in enumerate(v):
                a[k + 1 + i][j] -= 2.0 * vi * s
        for i in range(n):
            s = math.fsum(a[i][k + 1 + j] * v[j] for j in range(len(v)))
            for j, vj in enumerate(v):
                a[i][k + 1 + j] -= 2.0 * s * vj
        for i in range(k + 2, n):
            a[i][k] = 0.0
    return a


def _givens(a: complex, b: complex) -> tuple[float, complex]:
    """Return ``(c, s)`` with ``[[c, s], [-conj(s), c]] (a, b)' = (r, 0)'``.

    Args:
        a: Upper component.
        b: Lower component to eliminate.

    Returns:
        Real cosine and complex sine of the unitary rotation.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.1.8].
        Used by [EQ-SUR-ROM-012].

    """
    abs_b = abs(b)
    if abs_b == 0.0:
        return 1.0, 0j
    abs_a = abs(a)
    if abs_a == 0.0:
        return 0.0, b.conjugate() / abs_b
    r = math.hypot(abs_a, abs_b)
    return abs_a / r, (a / abs_a) * b.conjugate() / r


def _qr_eigenvalues(h: list[list[complex]]) -> list[complex]:
    """Return the eigenvalues of a complex Hessenberg matrix by shifted QR.

    Wilkinson-shifted QR sweeps with Givens rotations and deflation of
    negligible subdiagonal entries; an exceptional shift is used every tenth
    sweep on one eigenvalue.

    Args:
        h: Upper Hessenberg matrix (modified in place).

    Returns:
        The ``n`` eigenvalues.

    Raises:
        ValueError: If a sweep limit is exceeded.

    Equation:
        [EQ-SUR-ROM-012].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 7.5].

    """
    n = len(h)
    values = [0j] * n
    norm = math.sqrt(math.fsum(abs(v) ** 2 for row in h for v in row)) or 1.0
    hi = n - 1
    sweeps = 0
    while hi >= 0:
        low = hi
        while low > 0:
            local = abs(h[low - 1][low - 1]) + abs(h[low][low]) or norm
            if abs(h[low][low - 1]) <= _EPS * local:
                h[low][low - 1] = 0j
                break
            low -= 1
        if low == hi:
            values[hi] = h[hi][hi]
            hi -= 1
            sweeps = 0
            continue
        sweeps += 1
        if sweeps > _MAX_QR_SWEEPS:
            raise ValueError("QR iteration did not converge")
        a, b = h[hi - 1][hi - 1], h[hi - 1][hi]
        c, d = h[hi][hi - 1], h[hi][hi]
        half = (a - d) / 2.0
        root = cmath.sqrt(half * half + b * c)
        mu1 = (a + d) / 2.0 + root
        mu2 = (a + d) / 2.0 - root
        mu = mu1 if abs(mu1 - d) <= abs(mu2 - d) else mu2
        if sweeps % 10 == 0:
            mu += abs(c) + abs(h[hi - 1][hi - 2] if hi >= 2 else 0.0)
        for i in range(low, hi + 1):
            h[i][i] -= mu
        rotations = []
        for k in range(low, hi):
            ca, sa = _givens(h[k][k], h[k + 1][k])
            rotations.append((ca, sa))
            for j in range(k, hi + 1):
                top, bottom = h[k][j], h[k + 1][j]
                h[k][j] = ca * top + sa * bottom
                h[k + 1][j] = -sa.conjugate() * top + ca * bottom
        for offset, (ca, sa) in enumerate(rotations):
            k = low + offset
            for i in range(low, min(k + 2, hi) + 1):
                left, right = h[i][k], h[i][k + 1]
                h[i][k] = ca * left + sa.conjugate() * right
                h[i][k + 1] = -sa * left + ca * right
        for i in range(low, hi + 1):
            h[i][i] += mu
    return values


def _complex_solve(matrix: list[list[complex]], rhs: list[complex]) -> list[complex]:
    """Solve a complex square system by Gaussian elimination with pivoting.

    Args:
        matrix: Square matrix (copied).
        rhs: Right-hand side.

    Returns:
        The solution vector.

    Raises:
        ValueError: If the matrix is exactly singular.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.4]. Used by [EQ-SUR-ROM-012].

    """
    n = len(rhs)
    a = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(a[i][k]))
        if abs(a[pivot][k]) == 0.0:
            raise ValueError("matrix is singular")
        a[k], a[pivot] = a[pivot], a[k]
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            if factor != 0:
                for j in range(k, n + 1):
                    a[i][j] -= factor * a[k][j]
    x = [0j] * n
    for i in range(n - 1, -1, -1):
        tail = sum((a[i][j] * x[j] for j in range(i + 1, n)), 0j)
        x[i] = (a[i][n] - tail) / a[i][i]
    return x


def eigen_general(
    matrix: Sequence[Sequence[float]],
) -> tuple[ComplexVector, tuple[ComplexVector, ...]]:
    """Return eigenvalues and unit eigenvectors of a real square matrix.

    Eigenvalues come from Hessenberg reduction and Wilkinson-shifted complex
    QR iteration; each eigenvector from three steps of inverse iteration
    with the eigenvalue shifted by ``1e-10 ||A||_F`` and normalized to unit
    2-norm with its largest component real and positive. For a defective or
    nearly defective matrix the returned vectors of a repeated eigenvalue
    are close to parallel; this is reported by the caller through the
    conditioning of the eigenvector matrix, not hidden here.

    Args:
        matrix: Finite real square matrix.

    Returns:
        ``(eigenvalues, eigenvectors)`` with eigenvalues sorted by decreasing
        modulus (ties by argument) and ``eigenvectors[i]`` belonging to
        ``eigenvalues[i]``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is not square or the iteration fails.

    Equation:
        [EQ-SUR-ROM-012].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 7.4-7.5], [REF-WILKINSON-1965].

    """
    checked = as_matrix(matrix, name="matrix", square=True)
    n = len(checked)
    reduced = _hessenberg([list(row) for row in checked])
    complex_h = [[complex(v) for v in row] for row in reduced]
    eigenvalues = _qr_eigenvalues(complex_h)
    eigenvalues.sort(key=lambda z: (-abs(z), cmath.phase(z)))
    scale = _frobenius(checked) or 1.0
    vectors: list[ComplexVector] = []
    for value in eigenvalues:
        shifted = value + _INVERSE_ITERATION_SHIFT * scale
        system = [
            [complex(checked[i][j]) - (shifted if i == j else 0j) for j in range(n)]
            for i in range(n)
        ]
        v = [complex(1.0 + 0.37 * ((i * 7) % 5), 0.0) for i in range(n)]
        for _ in range(_INVERSE_ITERATION_STEPS):
            v = _complex_solve(system, v)
            length = math.sqrt(math.fsum(abs(z) ** 2 for z in v))
            v = [z / length for z in v]
        anchor = max(v, key=abs)
        phase = anchor.conjugate() / abs(anchor)
        vectors.append(tuple(z * phase for z in v))
    return tuple(eigenvalues), tuple(vectors)


# ---------------------------------------------------------------------------
# Proper orthogonal decomposition
# ---------------------------------------------------------------------------


def _select_rank(
    singular_values: Vector,
    numerical_rank: int,
    rank: int | None,
    energy: float | None,
    *,
    default_numerical: bool,
) -> int:
    """Apply the explicit truncation rule to a spectrum.

    Args:
        singular_values: Descending singular values.
        numerical_rank: Numerical rank of the snapshot matrix.
        rank: Requested rank, or ``None``.
        energy: Requested energy fraction in ``(0, 1]``, or ``None``.
        default_numerical: Allow neither argument and use the numerical rank.

    Returns:
        The selected rank ``r``.

    Raises:
        ValueError: If both or (when not allowed) neither rule is given, or
            the request is out of range.

    Equation:
        [EQ-SUR-ROM-001].

    References:
        [REF-SIROVICH-1987], [REF-ECKART-YOUNG-1936].

    """
    if rank is not None and energy is not None:
        raise ValueError("give either rank or energy, not both")
    if rank is None and energy is None:
        if not default_numerical:
            raise ValueError("give rank or energy explicitly")
        return numerical_rank
    if rank is not None:
        r = _integer(rank, "rank", 1)
        if r > numerical_rank:
            raise ValueError(
                f"rank {r} exceeds the numerical rank {numerical_rank} of the data"
            )
        return r
    fraction = require_finite_real(energy, name="energy")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy must lie in (0, 1]")
    squares = [s * s for s in singular_values]
    total = math.fsum(squares)
    if total <= 0.0:
        raise ValueError("snapshot data have zero energy")
    cumulative = 0.0
    for index, value in enumerate(squares):
        cumulative += value
        if cumulative >= fraction * total:
            return min(index + 1, numerical_rank)
    return numerical_rank


@dataclass(frozen=True, slots=True)
class PODBasis:
    """Proper orthogonal decomposition basis of a snapshot set.

    The state is approximated as ``x ~ mean + V a`` with orthonormal modes
    ``V = [v_1 ... v_r]`` and reduced coordinates ``a = V'(x - mean)``
    [EQ-SUR-ROM-001].

    Attributes:
        modes: ``modes[j]`` is the ``j``-th POD mode (length ``n``).
        singular_values: All singular values of the (centred) snapshot
            matrix, descending.
        mean: Subtracted mean (zeros when ``center=False``).
        rank: Number ``r`` of retained modes.
        energy_fractions: Cumulative energy ``sum_{i<=k} sigma_i^2 /
            sum sigma_i^2`` for ``k = 1..len(singular_values)``.
        centered: Whether the snapshot mean was removed.

    References:
        [EQ-SUR-ROM-001], [EQ-SUR-ROM-002], [REF-SIROVICH-1987].

    """

    modes: tuple[Vector, ...]
    singular_values: Vector
    mean: Vector
    rank: int
    energy_fractions: Vector
    centered: bool

    @property
    def state_dimension(self) -> int:
        """Return the full-order dimension ``n``.

        Returns:
            Length of a state vector.

        References:
            [EQ-SUR-ROM-001]. No equation is evaluated.

        """
        return len(self.mean)

    @property
    def energy_captured(self) -> float:
        """Return the energy fraction captured by the retained modes.

        Returns:
            ``energy_fractions[rank - 1]``.

        Equation:
            [EQ-SUR-ROM-001].

        References:
            [REF-SIROVICH-1987].

        """
        return self.energy_fractions[self.rank - 1]

    @property
    def truncation_error(self) -> float:
        """Return the Frobenius error ``sqrt(sum_{i>r} sigma_i^2)``.

        By the Eckart-Young theorem this equals the Frobenius norm of the
        training reconstruction error [EQ-SUR-ROM-002].

        Returns:
            The discarded singular-value tail.

        Equation:
            [EQ-SUR-ROM-002].

        References:
            [REF-ECKART-YOUNG-1936].

        """
        tail = self.singular_values[self.rank :]
        return math.sqrt(math.fsum(s * s for s in tail))

    def project(self, state: Sequence[float]) -> Vector:
        """Return the reduced coordinates ``a = V'(x - mean)``.

        Args:
            state: Full-order state.

        Returns:
            Vector of length ``rank``.

        Raises:
            ValueError: If the state has the wrong length.

        Equation:
            [EQ-SUR-ROM-001].

        References:
            [REF-SIROVICH-1987].

        """
        x = as_vector(state, name="state")
        if len(x) != self.state_dimension:
            raise ValueError("state length must equal the basis dimension")
        centred = tuple(a - b for a, b in zip(x, self.mean, strict=True))
        return tuple(_dot(mode, centred) for mode in self.modes[: self.rank])

    def project_derivative(self, derivative: Sequence[float]) -> Vector:
        """Return ``V' x'``; the mean does not enter a derivative.

        Args:
            derivative: Full-order time derivative.

        Returns:
            Reduced derivative of length ``rank``.

        Raises:
            ValueError: If the derivative has the wrong length.

        Equation:
            [EQ-SUR-ROM-004].

        References:
            [REF-PEHERSTORFER-WILLCOX-2016].

        """
        x = as_vector(derivative, name="derivative")
        if len(x) != self.state_dimension:
            raise ValueError("derivative length must equal the basis dimension")
        return tuple(_dot(mode, x) for mode in self.modes[: self.rank])

    def lift(self, coefficients: Sequence[float]) -> Vector:
        """Return ``mean + V a``.

        Args:
            coefficients: Reduced state of length ``rank``.

        Returns:
            Full-order approximation.

        Raises:
            ValueError: If the coefficient length is not ``rank``.

        Equation:
            [EQ-SUR-ROM-001].

        References:
            [REF-SIROVICH-1987].

        """
        a = as_vector(coefficients, name="coefficients")
        if len(a) != self.rank:
            raise ValueError("coefficients length must equal the basis rank")
        return tuple(
            self.mean[i] + math.fsum(a[j] * self.modes[j][i] for j in range(self.rank))
            for i in range(self.state_dimension)
        )

    def reconstruct(self, state: Sequence[float]) -> Vector:
        """Return the projection ``mean + V V'(x - mean)`` of a state.

        Args:
            state: Full-order state.

        Returns:
            The rank-``r`` reconstruction.

        Equation:
            [EQ-SUR-ROM-001].

        References:
            [REF-SIROVICH-1987].

        """
        return self.lift(self.project(state))

    def reconstruction_error(
        self, snapshots: Sequence[Sequence[float]], *, relative: bool = False
    ) -> float:
        """Return the Frobenius projection error of a snapshot set.

        Args:
            snapshots: States, one per row.
            relative: Divide by the Frobenius norm of the centred snapshots.

        Returns:
            ``||X - reconstruct(X)||_F`` (optionally relative).

        Raises:
            ValueError: If the relative norm is zero.

        Equation:
            [EQ-SUR-ROM-002].

        References:
            [REF-ECKART-YOUNG-1936].

        """
        data = _snapshot_matrix(snapshots, "snapshots")
        residual = [
            [a - b for a, b in zip(x, self.reconstruct(x), strict=True)] for x in data
        ]
        error = _frobenius(residual)
        if not relative:
            return error
        reference = _frobenius(
            [[a - b for a, b in zip(x, self.mean, strict=True)] for x in data]
        )
        if reference == 0.0:
            raise ValueError("centred snapshots are identically zero")
        return error / reference


def fit_pod(
    snapshots: Sequence[Sequence[float]],
    *,
    rank: int | None = None,
    energy: float | None = None,
    center: bool = False,
) -> PODBasis:
    """Compute a POD basis from snapshots by the SVD.

    With the snapshot matrix ``X`` (``n x m``, one column per snapshot,
    optionally minus the mean) and ``X = U S W'``, the modes are the columns
    of ``U`` and the rank is the given ``rank`` or the smallest ``r`` with
    ``sum_{i<=r} sigma_i^2 >= energy sum_i sigma_i^2`` [EQ-SUR-ROM-001].

    Args:
        snapshots: Time-ordered states, one per row.
        rank: Number of modes to keep.
        energy: Energy fraction in ``(0, 1]`` to capture. Exactly one of
            ``rank`` and ``energy`` must be given.
        center: Subtract the snapshot mean before the SVD.

    Returns:
        :class:`PODBasis`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the rule is missing, both rules are given, the rank
            exceeds the numerical rank or the data are degenerate.

    Equation:
        [EQ-SUR-ROM-001], [EQ-SUR-ROM-002].

    References:
        [REF-SIROVICH-1987], [REF-ECKART-YOUNG-1936],
        [REF-GOLUB-VANLOAN-2013, Section 8.6].

    """
    data = _snapshot_matrix(snapshots, "snapshots")
    m, n = len(data), len(data[0])
    if center:
        mean = tuple(math.fsum(row[j] for row in data) / m for j in range(n))
    else:
        mean = (0.0,) * n
    work = tuple(tuple(a - b for a, b in zip(row, mean, strict=True)) for row in data)
    svd = singular_value_decomposition(work)
    sigma = svd.singular_values
    r = _select_rank(sigma, svd.rank(), rank, energy, default_numerical=False)
    squares = [s * s for s in sigma]
    total = math.fsum(squares)
    cumulative = []
    running = 0.0
    for value in squares:
        running += value
        cumulative.append(running / total)
    modes = tuple(tuple(svd.v[i][j] for i in range(n)) for j in range(len(sigma)))
    return PODBasis(modes[:r], sigma, mean, r, tuple(cumulative), center)


# ---------------------------------------------------------------------------
# Reduced models and simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ROMTrajectory:
    """Time-stepping result of a reduced model.

    Attributes:
        times: Grid times reached.
        states: Reduced states at those times (the first is the initial one).
        completed: ``True`` if all requested steps were taken.
        stop_reason: ``"completed"`` or a description of the divergence.

    References:
        [EQ-SUR-ROM-006].

    """

    times: Vector
    states: tuple[Vector, ...]
    completed: bool
    stop_reason: str


@dataclass(frozen=True, slots=True)
class ReducedModel:
    """Reduced model ``a' = c + A a + H q(a) + B u`` with ``q`` compact.

    Attributes:
        constant: Constant vector ``c`` (length ``r``; zeros if absent).
        linear: ``r x r`` linear operator ``A``.
        quadratic: ``r x r (r + 1) / 2`` operator ``H`` acting on
            :func:`quadratic_features`, or ``None``.
        input_matrix: ``r x p`` input operator ``B``, or ``None``.

    References:
        [EQ-SUR-ROM-003], [EQ-SUR-ROM-004].

    """

    constant: Vector
    linear: Matrix
    quadratic: Matrix | None
    input_matrix: Matrix | None

    @property
    def dimension(self) -> int:
        """Return the reduced dimension ``r``.

        Returns:
            Length of the reduced state.

        References:
            [EQ-SUR-ROM-003]. No equation is evaluated.

        """
        return len(self.constant)

    def rhs(
        self, state: Sequence[float], inputs: Sequence[float] | None = None
    ) -> Vector:
        """Evaluate the reduced right-hand side.

        Args:
            state: Reduced state ``a``.
            inputs: Input ``u`` if the model has an input operator.

        Returns:
            ``c + A a + H q(a) + B u``.

        Raises:
            ValueError: If a length is wrong or an input is missing.

        Equation:
            [EQ-SUR-ROM-003].

        References:
            [REF-PEHERSTORFER-WILLCOX-2016].

        """
        a = tuple(state)
        if len(a) != self.dimension:
            raise ValueError("state length must equal the reduced dimension")
        out = [
            self.constant[i] + _dot(self.linear[i], a) for i in range(self.dimension)
        ]
        if self.quadratic is not None:
            q = quadratic_features(a)
            for i in range(self.dimension):
                out[i] += _dot(self.quadratic[i], q)
        if self.input_matrix is not None:
            if inputs is None:
                raise ValueError("this model needs an input vector")
            u = tuple(inputs)
            if len(u) != len(self.input_matrix[0]):
                raise ValueError("input length does not match the input operator")
            for i in range(self.dimension):
                out[i] += _dot(self.input_matrix[i], u)
        return tuple(out)

    def simulate(
        self,
        initial: Sequence[float],
        *,
        step: float,
        steps: int,
        input_function: Callable[[float], Sequence[float]] | None = None,
        t0: float = 0.0,
        blowup_limit: float = 1e8,
    ) -> ROMTrajectory:
        """Integrate the reduced model by classical Runge-Kutta.

        The run stops, and reports it, as soon as a state is non-finite or
        its Euclidean norm exceeds ``blowup_limit``; no state is clipped.

        Args:
            initial: Reduced initial state.
            step: Positive step ``h``.
            steps: Number of steps (non-negative).
            input_function: ``u(t)`` for models with an input operator.
            t0: Initial time.
            blowup_limit: Norm above which the run is declared diverged.

        Returns:
            :class:`ROMTrajectory`; ``completed`` is ``False`` on divergence.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-SUR-ROM-006].

        References:
            [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

        """
        h = _positive(step, "step")
        n_steps = _integer(steps, "steps", 0)
        limit = _positive(blowup_limit, "blowup_limit")
        time0 = require_finite_real(t0, name="t0")
        state = as_vector(initial, name="initial")
        if len(state) != self.dimension:
            raise ValueError("initial length must equal the reduced dimension")

        def evaluate(t: float, a: Vector) -> Vector:
            """Return the reduced right-hand side at time ``t``.

            Args:
                t: Time.
                a: Reduced state.

            Returns:
                ``rhs(a, u(t))``.

            References:
                [EQ-SUR-ROM-006].

            """
            u = None if input_function is None else input_function(t)
            return self.rhs(a, u)

        times = [time0]
        states = [state]
        for k in range(n_steps):
            t = time0 + k * h
            try:
                k1 = evaluate(t, state)
                k2 = evaluate(t + h / 2, _axpy(state, k1, h / 2))
                k3 = evaluate(t + h / 2, _axpy(state, k2, h / 2))
                k4 = evaluate(t + h, _axpy(state, k3, h))
                state = tuple(
                    s + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
                    for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
                )
            except OverflowError:
                return ROMTrajectory(
                    tuple(times), tuple(states), False, f"overflow at step {k + 1}"
                )
            size = math.sqrt(math.fsum(v * v for v in state))
            if not math.isfinite(size) or size > limit:
                return ROMTrajectory(
                    tuple(times),
                    tuple(states),
                    False,
                    f"diverged at step {k + 1}: norm {size!r} exceeds {limit!r}",
                )
            times.append(time0 + (k + 1) * h)
            states.append(state)
        return ROMTrajectory(tuple(times), tuple(states), True, "completed")


def _axpy(x: Vector, y: Vector, a: float) -> Vector:
    """Return ``x + a y`` componentwise.

    Args:
        x: First vector.
        y: Second vector.
        a: Scalar.

    Returns:
        The combination.

    References:
        [EQ-SUR-ROM-006]. Runge-Kutta stage arithmetic.

    """
    return tuple(p + a * q for p, q in zip(x, y, strict=True))


def galerkin_reduced_model(
    basis: PODBasis,
    *,
    linear: Sequence[Sequence[float]] | None = None,
    quadratic: Bilinear | None = None,
    constant: Sequence[float] | None = None,
    input_matrix: Sequence[Sequence[float]] | None = None,
) -> ReducedModel:
    """Project a full-order quadratic system onto a POD basis (Galerkin).

    For ``x' = c + A x + Q(x, x) + B u`` with a bilinear ``Q`` and
    ``x = m + V a`` the reduced model has [EQ-SUR-ROM-003]

    - constant ``V'(c + A m + Q(m, m))``,
    - linear columns ``V'(A v_j + Q(m, v_j) + Q(v_j, m))``,
    - compact quadratic columns ``V'(Q(v_i, v_j) + Q(v_j, v_i))`` for
      ``i < j`` and ``V' Q(v_i, v_i)`` for ``i = j``,
    - input operator ``V' B``.

    Args:
        basis: POD basis ``(m, V)``.
        linear: Full-order ``n x n`` matrix ``A`` (zero if omitted).
        quadratic: Bilinear map ``Q(x, y)`` returning a vector of length
            ``n`` (the quadratic term is ``Q(x, x)``); it need not be
            symmetric.
        constant: Full-order constant ``c`` (zero if omitted).
        input_matrix: Full-order ``n x p`` matrix ``B``.

    Returns:
        :class:`ReducedModel`.

    Raises:
        ValueError: If a shape does not match the basis.

    Equation:
        [EQ-SUR-ROM-003].

    References:
        [REF-PEHERSTORFER-WILLCOX-2016], [REF-SIROVICH-1987].

    """
    n, r = basis.state_dimension, basis.rank
    modes = basis.modes[:r]
    a_full = None if linear is None else as_matrix(linear, name="linear")
    if a_full is not None and (len(a_full) != n or len(a_full[0]) != n):
        raise ValueError("linear must be n x n")
    c_full = (0.0,) * n if constant is None else as_vector(constant, name="constant")
    if len(c_full) != n:
        raise ValueError("constant must have length n")
    mean = basis.mean

    def apply_a(x: Vector) -> Vector:
        """Return ``A x`` (zero when no linear operator is given).

        Args:
            x: Full-order vector.

        Returns:
            The product.

        References:
            [EQ-SUR-ROM-003].

        """
        if a_full is None:
            return (0.0,) * n
        return tuple(_dot(row, x) for row in a_full)

    def bilinear(x: Vector, y: Vector) -> Vector:
        """Return ``Q(x, y)`` (zero when no quadratic term is given).

        Args:
            x: First full-order vector.
            y: Second full-order vector.

        Returns:
            The bilinear value, validated for length.

        Raises:
            ValueError: If the callback returns the wrong length.

        References:
            [EQ-SUR-ROM-003].

        """
        if quadratic is None:
            return (0.0,) * n
        value = as_vector(quadratic(x, y), name="quadratic(x, y)")
        if len(value) != n:
            raise ValueError("quadratic must return a vector of length n")
        return value

    def project(x: Vector) -> Vector:
        """Return ``V' x``.

        Args:
            x: Full-order vector.

        Returns:
            Reduced vector.

        References:
            [EQ-SUR-ROM-003].

        """
        return tuple(_dot(mode, x) for mode in modes)

    drift = tuple(
        c + a + q
        for c, a, q in zip(c_full, apply_a(mean), bilinear(mean, mean), strict=True)
    )
    const_r = project(drift)
    columns = []
    for mode in modes:
        cross = tuple(
            a + q1 + q2
            for a, q1, q2 in zip(
                apply_a(mode), bilinear(mean, mode), bilinear(mode, mean), strict=True
            )
        )
        columns.append(project(cross))
    linear_r = tuple(tuple(columns[j][i] for j in range(r)) for i in range(r))
    quad_r: Matrix | None = None
    if quadratic is not None:
        quad_cols = []
        for i in range(r):
            for j in range(i, r):
                if i == j:
                    value = bilinear(modes[i], modes[i])
                else:
                    value = tuple(
                        p + q
                        for p, q in zip(
                            bilinear(modes[i], modes[j]),
                            bilinear(modes[j], modes[i]),
                            strict=True,
                        )
                    )
                quad_cols.append(project(value))
        quad_r = tuple(tuple(col[k] for col in quad_cols) for k in range(r))
    input_r: Matrix | None = None
    if input_matrix is not None:
        b_full = as_matrix(input_matrix, name="input_matrix")
        if len(b_full) != n:
            raise ValueError("input_matrix must have n rows")
        b_cols = _transpose(b_full)
        projected = [project(col) for col in b_cols]
        input_r = tuple(tuple(col[k] for col in projected) for k in range(r))
    return ReducedModel(const_r, linear_r, quad_r, input_r)


# ---------------------------------------------------------------------------
# Operator inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorInferenceResult:
    """Reduced model learned by operator inference.

    Attributes:
        model: The learned :class:`ReducedModel`.
        regularization: Tikhonov weight on the constant, linear and input
            columns.
        quadratic_regularization: Tikhonov weight on the quadratic columns.
        relative_residual: ``||D O' - R||_F / ||R||_F`` of the regression.
        condition_number: Condition number of the (unregularized) data
            matrix ``D``; ``inf`` if it is numerically singular.
        samples: Number of ``(state, derivative)`` pairs used.
        parameters: Number of learned columns per equation.

    References:
        [EQ-SUR-ROM-004].

    """

    model: ReducedModel
    regularization: float
    quadratic_regularization: float
    relative_residual: float
    condition_number: float
    samples: int
    parameters: int


def _optional_rows(
    values: Sequence[Sequence[float]] | None, name: str, count: int
) -> Matrix | None:
    """Validate optional per-sample input rows.

    Args:
        values: Input rows or ``None``.
        name: Argument name for messages.
        count: Required number of rows.

    Returns:
        Validated rows or ``None``.

    Raises:
        ValueError: If the row count differs.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if values is None:
        return None
    rows = as_matrix(values, name=name)
    if len(rows) != count:
        raise ValueError(f"{name} must have one row per snapshot")
    return rows


def fit_operator_inference(
    basis: PODBasis,
    snapshots: Sequence[Sequence[float]],
    derivatives: Sequence[Sequence[float]],
    *,
    inputs: Sequence[Sequence[float]] | None = None,
    constant: bool = False,
    quadratic: bool = False,
    regularization: float = 0.0,
    quadratic_regularization: float | None = None,
) -> OperatorInferenceResult:
    """Learn reduced operators from projected snapshots by least squares.

    With reduced states ``a_k = V'(x_k - m)`` and reduced derivatives
    ``r_k = V' x'_k`` the data matrix has rows
    ``d_k = [1, a_k', q(a_k)', u_k']`` (blocks optional) and the operators
    solve ``min_O sum_k ||O d_k - r_k||^2 + lambda_1 ||O_1||^2 +
    lambda_2 ||O_2||^2`` with ``lambda_2`` acting on the quadratic block
    only [EQ-SUR-ROM-004]. When the data are noise free, lie in the span of
    the basis and ``D`` has full column rank, ``lambda = 0`` returns the
    intrusive Galerkin operators of [EQ-SUR-ROM-003] up to rounding.

    Args:
        basis: POD basis; the data are projected with it.
        snapshots: Full-order states ``x_k``, one per row (several
            trajectories may be concatenated when exact derivatives are given).
        derivatives: Full-order time derivatives ``x'_k``, one per row.
        inputs: Inputs ``u_k``, one per row, if the model is input-driven.
        constant: Learn a constant vector ``c``.
        quadratic: Learn a compact quadratic operator ``H``.
        regularization: Tikhonov weight ``lambda_1 >= 0``.
        quadratic_regularization: Weight ``lambda_2``; defaults to
            ``regularization``.

    Returns:
        :class:`OperatorInferenceResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree, the regression has more parameters
            than samples, or it is rank deficient at the given weights.

    Equation:
        [EQ-SUR-ROM-004].

    References:
        [REF-PEHERSTORFER-WILLCOX-2016], [REF-TIKHONOV-1963],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    states = _snapshot_matrix(snapshots, "snapshots")
    ders = _snapshot_matrix(derivatives, "derivatives")
    if len(ders) != len(states):
        raise ValueError("snapshots and derivatives must have the same length")
    u_rows = _optional_rows(inputs, "inputs", len(states))
    lam1 = _non_negative(regularization, "regularization")
    lam2 = (
        lam1
        if quadratic_regularization is None
        else _non_negative(quadratic_regularization, "quadratic_regularization")
    )
    r = basis.rank
    reduced = tuple(basis.project(x) for x in states)
    targets = tuple(basis.project_derivative(d) for d in ders)
    rows = []
    for k, a in enumerate(reduced):
        row: list[float] = []
        if constant:
            row.append(1.0)
        row.extend(a)
        if quadratic:
            row.extend(quadratic_features(a))
        if u_rows is not None:
            row.extend(u_rows[k])
        rows.append(tuple(row))
    design = tuple(rows)
    p = len(design[0])
    if p > len(design):
        raise ValueError("more parameters than samples: add data or reduce the model")
    n_q = r * (r + 1) // 2
    ridge: list[float] = []
    if constant:
        ridge.append(lam1)
    ridge.extend([lam1] * r)
    if quadratic:
        ridge.extend([lam2] * n_q)
    if u_rows is not None:
        ridge.extend([lam1] * len(u_rows[0]))
    coefficients = _ridge_least_squares(design, targets, ridge)
    operator = _transpose(coefficients)  # r x p, row i for equation i
    offset = 0
    c_vec = (0.0,) * r
    if constant:
        c_vec = tuple(row[0] for row in operator)
        offset = 1
    a_mat = tuple(row[offset : offset + r] for row in operator)
    offset += r
    h_mat: Matrix | None = None
    if quadratic:
        h_mat = tuple(row[offset : offset + n_q] for row in operator)
        offset += n_q
    b_mat: Matrix | None = None
    if u_rows is not None:
        b_mat = tuple(row[offset:] for row in operator)
    fitted = tuple(
        tuple(_dot(design[k], coefficients_col) for coefficients_col in operator)
        for k in range(len(design))
    )
    residual = _frobenius(
        [
            [f - t for f, t in zip(fr, tr, strict=True)]
            for fr, tr in zip(fitted, targets, strict=True)
        ]
    )
    scale = _frobenius(targets)
    relative = residual / scale if scale > 0.0 else residual
    singular = singular_value_decomposition(design).singular_values
    smallest = singular[-1]
    cond = math.inf if smallest <= 0.0 else singular[0] / smallest
    model = ReducedModel(c_vec, a_mat, h_mat, b_mat)
    return OperatorInferenceResult(model, lam1, lam2, relative, cond, len(design), p)


@dataclass(frozen=True, slots=True)
class RegularizationSelection:
    """Outcome of the held-out trajectory rule for the Tikhonov weight.

    Attributes:
        selected: The chosen weight.
        candidates: Candidate weights in the order given.
        scores: Validation trajectory error of each candidate (``inf`` for
            a candidate whose simulation diverged).
        completed: Whether each candidate's simulation completed.
        result: Operator-inference fit with ``selected`` on all samples.

    References:
        [EQ-SUR-ROM-005].

    """

    selected: float
    candidates: Vector
    scores: Vector
    completed: tuple[bool, ...]
    result: OperatorInferenceResult


class _SampledInput:
    """Piecewise-linear interpolant of equally spaced input samples.

    References:
        [EQ-SUR-ROM-005]. Input signal of the validation simulation.

    """

    def __init__(self, samples: Matrix, step: float, t0: float) -> None:
        """Store the samples.

        Args:
            samples: Input rows ``u_k`` at times ``t0 + k step``.
            step: Sample spacing.
            t0: Time of the first sample.

        References:
            [EQ-SUR-ROM-005]. No equation is evaluated.

        """
        self.samples = samples
        self.step = step
        self.t0 = t0

    def __call__(self, t: float) -> Vector:
        """Return the interpolated input at time ``t`` (clamped at the ends).

        Args:
            t: Time.

        Returns:
            Interpolated input vector.

        References:
            [EQ-SUR-ROM-005].

        """
        position = (t - self.t0) / self.step
        last = len(self.samples) - 1
        position = min(max(position, 0.0), float(last))
        lower = min(int(math.floor(position)), max(last - 1, 0))
        frac = position - lower
        a = self.samples[lower]
        b = self.samples[min(lower + 1, last)]
        return tuple((1.0 - frac) * x + frac * y for x, y in zip(a, b, strict=True))


def select_operator_inference_regularization(
    basis: PODBasis,
    snapshots: Sequence[Sequence[float]],
    derivatives: Sequence[Sequence[float]],
    *,
    step: float,
    candidates: Sequence[float],
    inputs: Sequence[Sequence[float]] | None = None,
    constant: bool = False,
    quadratic: bool = False,
    validation_fraction: float = 0.25,
    blowup_limit: float = 1e8,
) -> RegularizationSelection:
    """Choose the Tikhonov weight by held-out trajectory error.

    Declared rule [EQ-SUR-ROM-005] for ONE time-ordered trajectory with
    uniform ``step``: each candidate ``lambda`` is fitted on the leading
    ``1 - validation_fraction`` of the samples (the same weight on every
    block), the reduced model is simulated from the first reduced snapshot
    over the whole record, and the score is the relative Frobenius error
    ``||a_sim - a_data||_F / ||a_data||_F`` on the trailing validation
    samples. A candidate whose simulation diverges (or whose regression is
    rank deficient) scores ``inf``. The smallest finite score wins and ties
    go to the earlier candidate. The selected weight is then refitted on all
    samples.

    Args:
        basis: POD basis.
        snapshots: One trajectory, uniformly spaced in time.
        derivatives: Matching time derivatives.
        step: Spacing of the snapshots.
        candidates: Non-negative candidate weights.
        inputs: Input samples, interpolated linearly in time.
        constant: Learn a constant vector.
        quadratic: Learn a quadratic operator.
        validation_fraction: Trailing fraction in ``(0, 1)`` held out.
        blowup_limit: Norm above which a simulation counts as diverged.

    Returns:
        :class:`RegularizationSelection`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or EVERY candidate fails.

    Equation:
        [EQ-SUR-ROM-005].

    References:
        [REF-PEHERSTORFER-WILLCOX-2016], [REF-TIKHONOV-1963].

    """
    states = _snapshot_matrix(snapshots, "snapshots", minimum=4)
    ders = _snapshot_matrix(derivatives, "derivatives")
    h = _positive(step, "step")
    grid = as_vector(candidates, name="candidates")
    if not grid:
        raise ValueError("candidates must be non-empty")
    if any(c < 0.0 for c in grid):
        raise ValueError("candidates must be non-negative")
    frac = require_finite_real(validation_fraction, name="validation_fraction")
    if not 0.0 < frac < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    m = len(states)
    split = min(max(int(round(m * (1.0 - frac))), 2), m - 1)
    u_rows = _optional_rows(inputs, "inputs", m)
    reduced = tuple(basis.project(x) for x in states)
    signal = None if u_rows is None else _SampledInput(u_rows, h, 0.0)
    reference = _frobenius(reduced[split:]) or 1.0
    scores: list[float] = []
    flags: list[bool] = []
    for weight in grid:
        try:
            fit = fit_operator_inference(
                basis,
                states[:split],
                ders[:split],
                inputs=None if u_rows is None else u_rows[:split],
                constant=constant,
                quadratic=quadratic,
                regularization=weight,
            )
        except ValueError:
            scores.append(math.inf)
            flags.append(False)
            continue
        run = fit.model.simulate(
            reduced[0],
            step=h,
            steps=m - 1,
            input_function=signal,
            blowup_limit=blowup_limit,
        )
        flags.append(run.completed)
        if not run.completed:
            scores.append(math.inf)
            continue
        error = _frobenius(
            [
                [p - q for p, q in zip(sim, obs, strict=True)]
                for sim, obs in zip(run.states[split:], reduced[split:], strict=True)
            ]
        )
        scores.append(error / reference)
    best = min(range(len(grid)), key=lambda i: (scores[i], i))
    if not math.isfinite(scores[best]):
        raise ValueError("every candidate regularization failed or diverged")
    final = fit_operator_inference(
        basis,
        states,
        ders,
        inputs=u_rows,
        constant=constant,
        quadratic=quadratic,
        regularization=grid[best],
    )
    return RegularizationSelection(grid[best], grid, tuple(scores), tuple(flags), final)


# ---------------------------------------------------------------------------
# Dynamic mode decomposition
# ---------------------------------------------------------------------------


def _complex_matrix_apply(
    matrix: Sequence[Sequence[float]], vector: Sequence[complex]
) -> list[complex]:
    """Return ``matrix @ vector`` for a real matrix and a complex vector.

    Args:
        matrix: Real matrix.
        vector: Complex vector.

    Returns:
        The product.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    return [
        sum((m * v for m, v in zip(row, vector, strict=True)), 0j) for row in matrix
    ]


@dataclass(frozen=True, slots=True)
class DMDResult:
    """Exact dynamic mode decomposition of a snapshot sequence.

    Attributes:
        eigenvalues: Eigenvalues ``lambda_i`` of the reduced operator
            (discrete-time, one per ``step``), sorted by decreasing modulus.
        modes: Exact DMD modes ``phi_i`` (length ``n``).
        projected_modes: Projected modes ``U w_i``.
        amplitudes: Amplitudes ``b`` with ``x_0 ~ sum_i b_i U w_i``.
        singular_values: Singular values of the first-snapshot matrix.
        rank: Truncation rank ``r``.
        reduced_operator: Real ``r x r`` operator ``A~``.
        basis: The ``r`` left singular vectors ``U`` (each length ``n``).
        step: Sampling interval.

    References:
        [EQ-SUR-ROM-007].

    """

    eigenvalues: ComplexVector
    modes: tuple[ComplexVector, ...]
    projected_modes: tuple[ComplexVector, ...]
    amplitudes: ComplexVector
    singular_values: Vector
    rank: int
    reduced_operator: Matrix
    basis: tuple[Vector, ...]
    step: float

    @property
    def continuous_eigenvalues(self) -> ComplexVector:
        """Return ``log(lambda_i) / step``, the continuous-time eigenvalues.

        Returns:
            Complex growth-rate and frequency pairs.

        Raises:
            ValueError: If an eigenvalue is exactly zero.

        Equation:
            [EQ-SUR-ROM-007].

        References:
            [REF-SCHMID-2010].

        """
        if any(z == 0 for z in self.eigenvalues):
            raise ValueError("a zero eigenvalue has no logarithm")
        return tuple(cmath.log(z) / self.step for z in self.eigenvalues)

    def forecast(self, initial: Sequence[float], steps: int) -> tuple[Vector, ...]:
        """Iterate the reduced operator from an initial state.

        The state is projected, advanced ``steps`` times by ``A~`` and lifted
        back with ``U`` [EQ-SUR-ROM-007].

        Args:
            initial: Full-order initial state.
            steps: Number of steps (non-negative).

        Returns:
            ``steps + 1`` full-order states ``x_0 ... x_steps``.

        Raises:
            ValueError: If the state has the wrong length.

        Equation:
            [EQ-SUR-ROM-007].

        References:
            [REF-SCHMID-2010], [REF-TU-ETAL-2014].

        """
        n_steps = _integer(steps, "steps", 0)
        x0 = as_vector(initial, name="initial")
        if len(x0) != len(self.basis[0]):
            raise ValueError("initial length must equal the state dimension")
        a = tuple(_dot(u, x0) for u in self.basis)
        out = []
        for _ in range(n_steps + 1):
            out.append(
                tuple(
                    math.fsum(a[j] * self.basis[j][i] for j in range(self.rank))
                    for i in range(len(x0))
                )
            )
            a = tuple(_dot(row, a) for row in self.reduced_operator)
        return tuple(out)


def fit_dmd(
    snapshots: Sequence[Sequence[float]],
    *,
    rank: int | None = None,
    energy: float | None = None,
    step: float = 1.0,
) -> DMDResult:
    """Fit exact DMD ``x_{k+1} ~ A x_k`` with an SVD rank truncation.

    With ``X = [x_0 .. x_{m-2}]``, ``Y = [x_1 .. x_{m-1}]`` and the rank-``r``
    SVD ``X ~ U S W'`` the reduced operator is ``A~ = U' Y W S^-1``; its
    eigenpairs ``A~ z = lambda z`` give the exact modes
    ``phi = Y W S^-1 z`` as in [EQ-SUR-ROM-007].

    Args:
        snapshots: Time-ordered states, one per row (at least three).
        rank: Truncation rank, or ``None``.
        energy: Energy fraction rule of [EQ-SUR-ROM-001], or ``None``. If both
            are ``None`` the numerical rank of the data is used.
        step: Sampling interval.

    Returns:
        :class:`DMDResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the data are too short or degenerate, the rank is out
            of range, or the eigenproblem fails.

    Equation:
        [EQ-SUR-ROM-007].

    References:
        [REF-SCHMID-2010], [REF-TU-ETAL-2014],
        [REF-GOLUB-VANLOAN-2013, Section 8.6].

    """
    data = _snapshot_matrix(snapshots, "snapshots", minimum=3)
    h = _positive(step, "step")
    first, second = data[:-1], data[1:]
    svd = singular_value_decomposition(first)
    sigma = svd.singular_values
    r = _select_rank(sigma, svd.rank(), rank, energy, default_numerical=True)
    n = len(data[0])
    basis = tuple(tuple(svd.v[i][j] for i in range(n)) for j in range(r))
    time_modes = tuple(tuple(svd.u[k][j] for k in range(len(first))) for j in range(r))
    # A~ = U' Y W S^-1; entry (i, j) = u_i' Y w_j / sigma_j.
    y_w = [
        tuple(
            math.fsum(second[k][a] * time_modes[j][k] for k in range(len(second)))
            for a in range(n)
        )
        for j in range(r)
    ]
    reduced = tuple(
        tuple(_dot(basis[i], y_w[j]) / sigma[j] for j in range(r)) for i in range(r)
    )
    values, vectors = eigen_general(reduced)
    projected = []
    exact = []
    for z in vectors:
        projected.append(
            tuple(sum((basis[j][i] * z[j] for j in range(r)), 0j) for i in range(n))
        )
        weights = [z[j] / sigma[j] for j in range(r)]
        mode = [sum((y_w[j][i] * weights[j] for j in range(r)), 0j) for i in range(n)]
        exact.append(tuple(mode))
    w_matrix = [[vectors[j][i] for j in range(r)] for i in range(r)]
    a0 = [complex(_dot(basis[i], data[0])) for i in range(r)]
    amplitudes = tuple(_complex_solve(w_matrix, a0))
    return DMDResult(
        values,
        tuple(exact),
        tuple(projected),
        amplitudes,
        sigma,
        r,
        reduced,
        basis,
        h,
    )


# ---------------------------------------------------------------------------
# Extended DMD
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EDMDResult:
    """Extended DMD (Koopman) approximation on a dictionary.

    Attributes:
        dictionary: The observables ``psi``.
        koopman_matrix: ``N x N`` matrix ``K`` with
            ``psi(y)' ~ psi(x)' K`` for ``y`` the successor of ``x``.
        eigenvalues: Eigenvalues of ``K``, sorted by decreasing modulus.
        eigenvectors: ``eigenvectors[i]`` is ``xi_i`` with
            ``K xi_i = lambda_i xi_i``; the eigenfunction is
            ``phi_i(x) = psi(x)' xi_i``.
        state_map: ``N x n`` matrix ``C`` with ``x ~ C' psi(x)``.
        relative_residual: ``||Psi_X K - Psi_Y||_F / ||Psi_Y||_F``.
        regularization: Tikhonov weight used.

    References:
        [EQ-SUR-ROM-008].

    """

    dictionary: FeatureDictionary
    koopman_matrix: Matrix
    eigenvalues: ComplexVector
    eigenvectors: tuple[ComplexVector, ...]
    state_map: Matrix
    relative_residual: float
    regularization: float

    def eigenfunction(self, index: int, state: Sequence[float]) -> complex:
        """Evaluate the ``index``-th approximate Koopman eigenfunction.

        Args:
            index: Eigenpair index.
            state: State vector.

        Returns:
            ``psi(state)' xi_index``.

        Raises:
            IndexError: If ``index`` is out of range.

        Equation:
            [EQ-SUR-ROM-008].

        References:
            [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015].

        """
        lifted = self.dictionary(state)
        xi = self.eigenvectors[index]
        return sum((p * c for p, c in zip(lifted, xi, strict=True)), 0j)

    def predict(self, state: Sequence[float], steps: int) -> tuple[Vector, ...]:
        """Forecast by iterating the Koopman matrix in the lifted space.

        Args:
            state: Initial full-order state.
            steps: Number of steps (non-negative).

        Returns:
            ``steps + 1`` states ``C' psi(x) K^k``; the error grows with
            ``k`` unless the dictionary spans an invariant subspace.

        Equation:
            [EQ-SUR-ROM-008].

        References:
            [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015].

        """
        n_steps = _integer(steps, "steps", 0)
        z = self.dictionary(state)
        n_state = len(self.state_map[0])
        out = []
        for _ in range(n_steps + 1):
            out.append(
                tuple(
                    math.fsum(z[j] * self.state_map[j][i] for j in range(len(z)))
                    for i in range(n_state)
                )
            )
            z = tuple(
                math.fsum(z[i] * self.koopman_matrix[i][j] for i in range(len(z)))
                for j in range(len(z))
            )
        return tuple(out)


def fit_edmd(
    current: Sequence[Sequence[float]],
    successor: Sequence[Sequence[float]],
    dictionary: FeatureDictionary,
    *,
    regularization: float = 0.0,
) -> EDMDResult:
    """Fit extended DMD by (ridge) least squares on a dictionary.

    Solves ``min_K ||Psi_X K - Psi_Y||_F^2 + lambda ||K||_F^2`` with
    ``Psi_X[k] = psi(x_k)'`` and ``Psi_Y[k] = psi(y_k)'`` [EQ-SUR-ROM-008];
    the Koopman eigenvalues are those of ``K``. A second regression gives the
    linear map from observables back to the state.

    Args:
        current: States ``x_k``, one per row.
        successor: States ``y_k = F(x_k)`` (one step later), one per row.
        dictionary: Observables.
        regularization: Tikhonov weight ``lambda >= 0``.

    Returns:
        :class:`EDMDResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree, the dictionary dimension does not
            match the states, or the regression is rank deficient at the
            given weight.

    Equation:
        [EQ-SUR-ROM-008].

    References:
        [REF-WILLIAMS-KEVREKIDIS-ROWLEY-2015], [REF-SCHMID-2010].

    """
    x = _snapshot_matrix(current, "current")
    y = _snapshot_matrix(successor, "successor")
    if len(x) != len(y) or len(x[0]) != len(y[0]):
        raise ValueError("current and successor must have the same shape")
    if not isinstance(dictionary, FeatureDictionary):
        raise TypeError("dictionary must be a FeatureDictionary")
    if dictionary.dimension != len(x[0]):
        raise ValueError("dictionary dimension must equal the state dimension")
    lam = _non_negative(regularization, "regularization")
    psi_x = dictionary.evaluate(x)
    psi_y = dictionary.evaluate(y)
    size = dictionary.size
    ridge = [lam] * size
    koopman = _ridge_least_squares(psi_x, psi_y, ridge)
    state_map = _ridge_least_squares(psi_x, x, ridge)
    predicted = tuple(
        tuple(
            math.fsum(row[i] * koopman[i][j] for i in range(size)) for j in range(size)
        )
        for row in psi_x
    )
    scale = _frobenius(psi_y)
    residual = _frobenius(
        [
            [p - q for p, q in zip(a, b, strict=True)]
            for a, b in zip(predicted, psi_y, strict=True)
        ]
    )
    values, vectors = eigen_general(koopman)
    return EDMDResult(
        dictionary,
        koopman,
        values,
        vectors,
        state_map,
        residual / scale if scale > 0.0 else residual,
        lam,
    )


# ---------------------------------------------------------------------------
# Derivative estimation
# ---------------------------------------------------------------------------


def finite_difference_derivatives(
    snapshots: Sequence[Sequence[float]], step: float
) -> Matrix:
    """Estimate time derivatives by second-order finite differences.

    Interior points use ``(x_{k+1} - x_{k-1}) / (2 h)``; the end points use
    the one-sided formulas ``(-3 x_0 + 4 x_1 - x_2) / (2 h)`` and
    ``(3 x_{m-1} - 4 x_{m-2} + x_{m-3}) / (2 h)`` [EQ-SUR-ROM-009]. The
    truncation error is ``O(h^2)`` and white noise of standard deviation
    ``s`` in the states gives derivative noise of about ``s / (sqrt(2) h)``.

    Args:
        snapshots: Uniformly spaced states, one per row (at least three).
        step: Spacing ``h``.

    Returns:
        Derivative estimates, one row per snapshot.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If there are fewer than three snapshots or ``step`` is
            not positive.

    Equation:
        [EQ-SUR-ROM-009].

    References:
        [REF-BRUNTON-PROCTOR-KUTZ-2016], [REF-GOLUB-VANLOAN-2013].

    """
    x = _snapshot_matrix(snapshots, "snapshots", minimum=3)
    h = _positive(step, "step")
    m = len(x)
    rows = []
    for k in range(m):
        if k == 0:
            rows.append(
                tuple(
                    (-3.0 * a + 4.0 * b - c) / (2.0 * h)
                    for a, b, c in zip(x[0], x[1], x[2], strict=True)
                )
            )
        elif k == m - 1:
            rows.append(
                tuple(
                    (3.0 * a - 4.0 * b + c) / (2.0 * h)
                    for a, b, c in zip(x[-1], x[-2], x[-3], strict=True)
                )
            )
        else:
            rows.append(
                tuple(
                    (b - a) / (2.0 * h) for a, b in zip(x[k - 1], x[k + 1], strict=True)
                )
            )
    return tuple(rows)


def local_polynomial_derivatives(
    snapshots: Sequence[Sequence[float]],
    step: float,
    *,
    window: int,
    polyorder: int = 3,
) -> Matrix:
    """Estimate derivatives by local polynomial least squares (smoothing).

    Around each sample ``k`` a polynomial of degree ``polyorder`` is fitted by
    least squares to the ``window`` nearest samples (a window shifted inwards
    at the ends) and its slope at ``t_k`` is returned [EQ-SUR-ROM-009]. This
    is the Savitzky-Golay derivative; it averages out noise at the price of
    bias where the signal varies faster than the polynomial resolves.

    Args:
        snapshots: Uniformly spaced states, one per row.
        step: Spacing ``h``.
        window: Number of samples per fit (at least ``polyorder + 1``, at
            most the number of snapshots).
        polyorder: Polynomial degree (at least one).

    Returns:
        Derivative estimates, one row per snapshot.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the window, order or step is out of range.

    Equation:
        [EQ-SUR-ROM-009].

    References:
        [REF-SAVITZKY-GOLAY-1964], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    x = _snapshot_matrix(snapshots, "snapshots", minimum=2)
    h = _positive(step, "step")
    order = _integer(polyorder, "polyorder", 1)
    width = _integer(window, "window", order + 1)
    m = len(x)
    if width > m:
        raise ValueError("window cannot exceed the number of snapshots")
    half = width // 2
    n = len(x[0])
    rows = []
    for k in range(m):
        start = min(max(k - half, 0), m - width)
        taus = [(start + j - k) * h for j in range(width)]
        # Scale by the window half-width for conditioning.
        spread = max(abs(t) for t in taus) or 1.0
        design = tuple(tuple((t / spread) ** p for p in range(order + 1)) for t in taus)
        factor = householder_qr(design)
        slope = []
        for c in range(n):
            coefficients = factor.solve(tuple(x[start + j][c] for j in range(width)))
            slope.append(coefficients[1] / spread)
        rows.append(tuple(slope))
    return tuple(rows)


# ---------------------------------------------------------------------------
# SINDy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SINDyResult:
    """Sparse identification of nonlinear dynamics.

    Attributes:
        coefficients: ``coefficients[j][i]`` multiplies library term ``i`` in
            the equation of state ``j``.
        dictionary: The library of candidate terms.
        threshold: Hard threshold used.
        iterations: Number of thresholding passes per equation.
        converged: Whether the support of each equation stopped changing
            within the iteration limit.
        relative_residuals: ``||x'_j - Theta xi_j|| / ||x'_j||`` per equation.

    References:
        [EQ-SUR-ROM-010].

    """

    coefficients: Matrix
    dictionary: FeatureDictionary
    threshold: float
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]
    relative_residuals: Vector

    def support(self) -> tuple[tuple[int, ...], ...]:
        """Return the indices of non-zero coefficients per equation.

        Returns:
            One tuple of term indices per state.

        References:
            [EQ-SUR-ROM-010]. No equation is evaluated.

        """
        return tuple(
            tuple(i for i, c in enumerate(row) if c != 0.0) for row in self.coefficients
        )

    def rhs(self, t: float, state: Sequence[float]) -> Vector:
        """Evaluate the identified vector field ``Theta(x) Xi``.

        Args:
            t: Time (ignored; the model is autonomous).
            state: State vector.

        Returns:
            The identified time derivative, usable as an ODE right-hand side.

        Equation:
            [EQ-SUR-ROM-010].

        References:
            [REF-BRUNTON-PROCTOR-KUTZ-2016].

        """
        lifted = self.dictionary(state)
        return tuple(_dot(row, lifted) for row in self.coefficients)

    def equations(self, digits: int = 4) -> tuple[str, ...]:
        """Return the identified equations as readable strings.

        Args:
            digits: Significant digits of the coefficients.

        Returns:
            One string ``"x0' = 10 x1 + -10 x0"`` per state.

        References:
            [EQ-SUR-ROM-010]. No equation is evaluated.

        """
        out = []
        for j, row in enumerate(self.coefficients):
            terms = [
                f"{c:.{digits}g} {self.dictionary.names[i]}"
                for i, c in enumerate(row)
                if c != 0.0
            ]
            out.append(f"x{j}' = " + (" + ".join(terms) if terms else "0"))
        return tuple(out)


def _gram(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the Gram matrix of a set of column vectors.

    Args:
        columns: ``N`` columns of equal length.

    Returns:
        The symmetric ``N x N`` matrix of inner products.

    References:
        [EQ-SUR-ROM-010]. Normal equations of the active-set regressions.

    """
    size = len(columns)
    gram = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = math.fsum(
                a * b for a, b in zip(columns[i], columns[j], strict=True)
            )
            gram[i][j] = gram[j][i] = value
    return gram


def _stlsq(
    gram: list[list[float]],
    moments: list[float],
    available: list[bool],
    *,
    threshold: float,
    ridge: float,
    max_iterations: int,
    normalize: bool,
) -> tuple[list[float], int, bool]:
    """Run sequentially thresholded least squares for one equation.

    Args:
        gram: Gram matrix ``Theta' Theta`` of the library columns.
        moments: ``Theta' x'`` for this equation.
        available: Columns that may enter (zero columns may not).
        threshold: Hard threshold on coefficients in library units.
        ridge: Tikhonov weight on the unit-norm columns.
        max_iterations: Maximum number of thresholding passes.
        normalize: Scale the columns to unit norm for the solve.

    Returns:
        ``(coefficients, passes, converged)``.

    Raises:
        ValueError: If an active-set regression is rank deficient.

    Equation:
        [EQ-SUR-ROM-010].

    References:
        [REF-BRUNTON-PROCTOR-KUTZ-2016].

    """
    size = len(moments)
    scale = [
        math.sqrt(gram[i][i]) if normalize and gram[i][i] > 0.0 else 1.0
        for i in range(size)
    ]

    def solve(active: list[int]) -> list[float]:
        """Return the coefficient vector of the regression on ``active``.

        Args:
            active: Indices of the columns kept.

        Returns:
            Full-length coefficient list, zero off the active set.

        Raises:
            ValueError: If the active Gram matrix is not positive definite.

        References:
            [EQ-SUR-ROM-010].

        """
        coefficients = [0.0] * size
        if not active:
            return coefficients
        system = tuple(
            tuple(
                gram[i][j] / (scale[i] * scale[j]) + (ridge if i == j else 0.0)
                for j in active
            )
            for i in active
        )
        rhs = tuple(moments[i] / scale[i] for i in active)
        try:
            solution = solve_spd(system, rhs)
        except ValueError as error:
            raise ValueError(
                "SINDy regression is rank deficient on the active set: reduce "
                "the library or use a positive ridge weight"
            ) from error
        for position, i in enumerate(active):
            coefficients[i] = solution[position] / scale[i]
        return coefficients

    active = [i for i in range(size) if available[i] and gram[i][i] > 0.0]
    for passes in range(1, max_iterations + 1):
        coefficients = solve(active)
        keep = [i for i in active if abs(coefficients[i]) >= threshold]
        if keep == active:
            return coefficients, passes, True
        active = keep
        if not active:
            return [0.0] * size, passes, True
    return solve(active), max_iterations, False


def _sindy_from_columns(
    columns: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    available: list[bool],
    *,
    threshold: float,
    ridge: float,
    max_iterations: int,
    normalize: bool,
) -> tuple[list[list[float]], list[int], list[bool]]:
    """Run STLSQ for every equation given library columns and target columns.

    Args:
        columns: ``N`` library columns ``Theta_i`` over the samples.
        targets: One derivative column per state.
        available: Library columns allowed to enter.
        threshold: Hard threshold.
        ridge: Tikhonov weight.
        max_iterations: Pass limit.
        normalize: Column scaling flag.

    Returns:
        Coefficients per equation, passes per equation, convergence flags.

    Equation:
        [EQ-SUR-ROM-010].

    References:
        [REF-BRUNTON-PROCTOR-KUTZ-2016].

    """
    gram = _gram(columns)
    coefficients, passes, flags = [], [], []
    for target in targets:
        moments = [
            math.fsum(a * b for a, b in zip(col, target, strict=True))
            for col in columns
        ]
        c, p, f = _stlsq(
            gram,
            moments,
            available,
            threshold=threshold,
            ridge=ridge,
            max_iterations=max_iterations,
            normalize=normalize,
        )
        coefficients.append(c)
        passes.append(p)
        flags.append(f)
    return coefficients, passes, flags


def _prepare_sindy(
    states: Sequence[Sequence[float]],
    derivatives: Sequence[Sequence[float]],
    dictionary: FeatureDictionary,
    threshold: float,
    ridge: float,
    max_iterations: int,
) -> tuple[Matrix, Matrix, float, float, int]:
    """Validate common SINDy arguments and evaluate the library.

    Args:
        states: States, one per row.
        derivatives: Derivatives, one per row.
        dictionary: Library.
        threshold: Hard threshold.
        ridge: Tikhonov weight.
        max_iterations: Pass limit.

    Returns:
        ``(library_matrix, derivative_matrix, threshold, ridge, passes)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree or a value is out of range.

    References:
        [REF-PROBSTAT-VALIDATION]. Argument checks of [EQ-SUR-ROM-010].

    """
    x = _snapshot_matrix(states, "states")
    d = _snapshot_matrix(derivatives, "derivatives")
    if len(x) != len(d) or len(x[0]) != len(d[0]):
        raise ValueError("states and derivatives must have the same shape")
    if not isinstance(dictionary, FeatureDictionary):
        raise TypeError("dictionary must be a FeatureDictionary")
    if dictionary.dimension != len(x[0]):
        raise ValueError("dictionary dimension must equal the state dimension")
    thr = _non_negative(threshold, "threshold")
    lam = _non_negative(ridge, "ridge")
    passes = _integer(max_iterations, "max_iterations", 1)
    return dictionary.evaluate(x), d, thr, lam, passes


def fit_sindy(
    states: Sequence[Sequence[float]],
    derivatives: Sequence[Sequence[float]],
    dictionary: FeatureDictionary,
    *,
    threshold: float,
    ridge: float = 0.0,
    max_iterations: int = 10,
    normalize: bool = True,
) -> SINDyResult:
    """Identify a sparse vector field by sequentially thresholded least squares.

    For each state ``j`` the regression ``x'_j ~ Theta xi_j`` is solved on the
    library columns, coefficients with ``|xi| < threshold`` are set to zero
    and removed, and the regression is repeated on the survivors until the
    support is stable or ``max_iterations`` passes elapsed [EQ-SUR-ROM-010].
    The threshold is applied to coefficients in the units of the library
    terms. The active-set regressions use the normal equations of the
    unit-norm columns, which squares their condition number; a singular
    active set raises instead of being regularized.

    Args:
        states: States ``x_k``, one per row.
        derivatives: Derivative estimates (see
            :func:`finite_difference_derivatives`), one per row.
        dictionary: Library of candidate terms.
        threshold: Hard threshold ``>= 0``.
        ridge: Optional Tikhonov weight on the unit-norm columns.
        max_iterations: Maximum number of thresholding passes.
        normalize: Scale library columns to unit norm for the solves.

    Returns:
        :class:`SINDyResult`; ``converged`` flags equations whose support did
        not stabilize.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree, a value is out of range, or an active
            set is rank deficient.

    Equation:
        [EQ-SUR-ROM-010].

    References:
        [REF-BRUNTON-PROCTOR-KUTZ-2016].

    """
    theta, ders, thr, lam, passes = _prepare_sindy(
        states, derivatives, dictionary, threshold, ridge, max_iterations
    )
    columns = _transpose(theta)
    targets = _transpose(ders)
    coefficients, counts, flags = _sindy_from_columns(
        columns,
        targets,
        [True] * len(columns),
        threshold=thr,
        ridge=lam,
        max_iterations=passes,
        normalize=normalize,
    )
    residuals = []
    for j, target in enumerate(targets):
        fitted = [
            math.fsum(coefficients[j][i] * columns[i][k] for i in range(len(columns)))
            for k in range(len(target))
        ]
        norm = math.sqrt(math.fsum(v * v for v in target))
        miss = math.sqrt(
            math.fsum((a - b) ** 2 for a, b in zip(fitted, target, strict=True))
        )
        residuals.append(miss / norm if norm > 0.0 else miss)
    return SINDyResult(
        tuple(tuple(row) for row in coefficients),
        dictionary,
        thr,
        tuple(counts),
        tuple(flags),
        tuple(residuals),
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated sample quantile (R type 7).

    Args:
        sorted_values: Non-empty ascending sample.
        probability: Level in ``[0, 1]``.

    Returns:
        The quantile.

    References:
        [REF-EFRON-TIBSHIRANI-1993]. Percentile intervals of [EQ-SUR-ROM-011].

    """
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return (1.0 - frac) * sorted_values[lower] + frac * sorted_values[upper]


@dataclass(frozen=True, slots=True)
class EnsembleSINDyResult:
    """Ensemble (bootstrap) SINDy summary.

    Attributes:
        dictionary: The library.
        inclusion_probability: ``p[j][i]``: fraction of the successful
            ensemble members, among those whose library contained term ``i``,
            in which the term is non-zero in equation ``j``.
        coefficient_mean: Mean over members of the coefficient (zeros count).
        coefficient_median: Median over members (zeros count).
        interval_lower: Lower percentile bound over members (zeros count).
        interval_upper: Upper percentile bound over members (zeros count).
        conditional_median: Median over members that include the term
            (zero if never included).
        selected: ``conditional_median`` where the inclusion probability is at
            least ``inclusion_cutoff``, zero elsewhere.
        inclusion_cutoff: Probability cut-off used for ``selected``.
        interval_level: Central coverage of the interval, e.g. 0.95.
        models: Number of ensemble members requested.
        failed: Number of members dropped because a regression was rank
            deficient.
        mode: Resampling scheme: ``"data"``, ``"library"`` or ``"both"``.

    References:
        [EQ-SUR-ROM-011].

    """

    dictionary: FeatureDictionary
    inclusion_probability: Matrix
    coefficient_mean: Matrix
    coefficient_median: Matrix
    interval_lower: Matrix
    interval_upper: Matrix
    conditional_median: Matrix
    selected: Matrix
    inclusion_cutoff: float
    interval_level: float
    models: int
    failed: int
    mode: str


def fit_ensemble_sindy(
    states: Sequence[Sequence[float]],
    derivatives: Sequence[Sequence[float]],
    dictionary: FeatureDictionary,
    *,
    threshold: float,
    models: int = 100,
    mode: str = "data",
    library_fraction: float = 0.8,
    inclusion_cutoff: float = 0.5,
    interval_level: float = 0.95,
    ridge: float = 0.0,
    max_iterations: int = 10,
    seed: int | RandomStream | None = 0,
) -> EnsembleSINDyResult:
    """Run SINDy on resampled data and summarize the ensemble.

    Each member draws a bootstrap sample of the rows with replacement
    (``mode="data"``), a random subset of ``ceil(library_fraction N)`` library
    terms without replacement (``"library"``), or both, and runs STLSQ with
    the common threshold [EQ-SUR-ROM-011]. Inclusion probabilities are the
    fractions of members (among those that offered the term) in which a
    coefficient is non-zero. Percentile intervals use all members, counting
    an excluded term as the coefficient zero; they are bootstrap spreads, not
    credible intervals.

    Args:
        states: States, one per row.
        derivatives: Derivative estimates, one per row.
        dictionary: Library.
        threshold: Hard threshold of every member.
        models: Number of ensemble members (at least two).
        mode: ``"data"``, ``"library"`` or ``"both"``.
        library_fraction: Fraction of terms offered per member in the
            library modes, in ``(0, 1]``.
        inclusion_cutoff: Inclusion probability needed for ``selected``.
        interval_level: Central coverage in ``(0, 1)`` of the intervals.
        ridge: Tikhonov weight of the members.
        max_iterations: STLSQ pass limit.
        seed: Seed or :class:`RandomStream` (one child stream per member).

    Returns:
        :class:`EnsembleSINDyResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or every member fails.

    Equation:
        [EQ-SUR-ROM-011].

    References:
        [REF-FASEL-ETAL-2022], [REF-BRUNTON-PROCTOR-KUTZ-2016],
        [REF-EFRON-TIBSHIRANI-1993].

    """
    theta, ders, thr, lam, passes = _prepare_sindy(
        states, derivatives, dictionary, threshold, ridge, max_iterations
    )
    count = _integer(models, "models", 2)
    if mode not in ("data", "library", "both"):
        raise ValueError("mode must be 'data', 'library' or 'both'")
    fraction = require_finite_real(library_fraction, name="library_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("library_fraction must lie in (0, 1]")
    cutoff = require_finite_real(inclusion_cutoff, name="inclusion_cutoff")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("inclusion_cutoff must lie in [0, 1]")
    level = require_finite_real(interval_level, name="interval_level")
    if not 0.0 < level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")
    stream = resolve_stream(seed)
    samples = len(theta)
    columns = _transpose(theta)
    targets_full = _transpose(ders)
    size, n_state = len(columns), len(targets_full)
    offered_count = max(1, math.ceil(fraction * size))
    draws: list[list[list[float]]] = []  # per member: n_state x size
    offered: list[list[bool]] = []
    failed = 0
    for member in range(count):
        rng = stream.child("ensemble-sindy", member).generator()
        if mode in ("data", "both"):
            rows = [rng.randrange(samples) for _ in range(samples)]
            sampled = [[col[k] for k in rows] for col in columns]
            sampled_targets = [[t[k] for k in rows] for t in targets_full]
        else:
            sampled = [list(col) for col in columns]
            sampled_targets = [list(t) for t in targets_full]
        if mode in ("library", "both"):
            chosen = set(rng.sample(range(size), offered_count))
            available = [i in chosen for i in range(size)]
        else:
            available = [True] * size
        try:
            coefficients, _, _ = _sindy_from_columns(
                sampled,
                sampled_targets,
                available,
                threshold=thr,
                ridge=lam,
                max_iterations=passes,
                normalize=True,
            )
        except ValueError:
            failed += 1
            continue
        draws.append(coefficients)
        offered.append(available)
    if not draws:
        raise ValueError("every ensemble member failed (rank-deficient regressions)")
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    prob, mean, median, lower, upper, cond, picked = ([] for _ in range(7))
    for j in range(n_state):
        rows_p, rows_m, rows_d, rows_l, rows_u, rows_c, rows_s = ([] for _ in range(7))
        for i in range(size):
            values = sorted(m[j][i] for m in draws)
            seen = [m[j][i] for m, o in zip(draws, offered, strict=True) if o[i]]
            included = sorted(v for v in seen if v != 0.0)
            p = len(included) / len(seen) if seen else 0.0
            c = _quantile(included, 0.5) if included else 0.0
            rows_p.append(p)
            rows_m.append(math.fsum(values) / len(values))
            rows_d.append(_quantile(values, 0.5))
            rows_l.append(_quantile(values, lo))
            rows_u.append(_quantile(values, hi))
            rows_c.append(c)
            rows_s.append(c if p >= cutoff and p > 0.0 else 0.0)
        prob.append(tuple(rows_p))
        mean.append(tuple(rows_m))
        median.append(tuple(rows_d))
        lower.append(tuple(rows_l))
        upper.append(tuple(rows_u))
        cond.append(tuple(rows_c))
        picked.append(tuple(rows_s))
    return EnsembleSINDyResult(
        dictionary,
        tuple(prob),
        tuple(mean),
        tuple(median),
        tuple(lower),
        tuple(upper),
        tuple(cond),
        tuple(picked),
        cutoff,
        level,
        count,
        failed,
        mode,
    )
