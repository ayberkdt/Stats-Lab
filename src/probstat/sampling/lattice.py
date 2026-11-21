"""Rank-1 lattice rules: points, random shifts, and generating vectors.

A rank-1 lattice with ``n`` points and an integer generating vector ``z`` is
the point set ``x_k = frac(k z / n)``, ``k = 0, ..., n - 1``. It is a finite
additive group modulo one, which is what makes it exact for every
trigonometric monomial ``exp(2 pi i h.x)`` whose frequency ``h`` is not in the
dual lattice ``{h : h.z = 0 (mod n)}`` and what reduces the quality criterion
below from a double to a single sum. A Cranley-Patterson rotation
``frac(x_k + Delta)`` with ``Delta ~ U[0, 1)^s`` makes every point uniform on
the cube, so the equal-weight rule is unbiased.

The quality of ``z`` is measured here by one explicit criterion,
[EQ-SAMP-LATTICE-002]: the root mean square, over the random shift, of the
worst-case integration error in the unit ball of the reproducing-kernel
Hilbert space with the product kernel

``K(x, y) = prod_j (1 + gamma_j (B_1(x_j) B_1(y_j) + B_2(|x_j - y_j|) / 2))``,

``B_1(x) = x - 1/2``, ``B_2(x) = x^2 - x + 1/6``. Per coordinate this is the
kernel of the inner product
``<f, g> = (int f)(int g) + (1 / gamma) int f' g'`` (the "unanchored" Sobolev
space of functions with a square-integrable first derivative). Averaging
``K(frac(x + Delta), frac(y + Delta))`` over the shift gives the
shift-invariant kernel ``prod_j (1 + gamma_j B_2(frac(x_j - y_j)))``, from
which the criterion follows; ADR-0118 writes the derivation out and the tests
check it against a Monte Carlo average over shifts. Weight conventions differ
between authors by constant factors; the statement above is the one this
module implements.

``cbc_generating_vector`` chooses ``z`` one component at a time (greedy, not a
global optimum beyond two dimensions); ``korobov_generating_vector`` searches
vectors of the form ``(1, a, a^2, ...)``. Both are the naive ``O(s n^2)``
searches and require a prime ``n``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real

from .designs import SamplingDesign

_TINY = 2.0**-53


def _is_prime(value: int) -> bool:
    """Return whether an integer is prime, by trial division.

    References:
        Used to validate lattice sizes for [EQ-SAMP-LATTICE-003]. No equation
        is evaluated.

    """
    if value < 2:
        return False
    if value < 4:
        return True
    if value % 2 == 0:
        return False
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _require_positive_integer(value: object, *, name: str) -> int:
    """Validate and return a positive integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 1``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _require_prime_size(size: object) -> int:
    """Validate a prime number of lattice points.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size`` is not prime.

    References:
        [EQ-SAMP-LATTICE-003]: for prime ``n`` every candidate component in
        ``{1, ..., n - 1}`` is coprime to ``n``.

    """
    checked = _require_positive_integer(size, name="size")
    if not _is_prime(checked):
        raise ValueError(f"size must be a prime number, got {checked}")
    return checked


def _generating_vector(size: int, generating_vector: Sequence[int]) -> tuple[int, ...]:
    """Validate a generating vector and reduce it modulo ``size``.

    Raises:
        TypeError: If a component is not an integer.
        ValueError: If the vector is empty or a component shares a factor
            with ``size``.

    References:
        [EQ-SAMP-LATTICE-001].

    """
    components = tuple(generating_vector)
    if not components:
        raise ValueError("generating_vector must have at least one component")
    reduced = []
    for index, component in enumerate(components):
        if isinstance(component, bool) or not isinstance(component, int):
            raise TypeError(f"generating_vector[{index}] must be an integer")
        if math.gcd(component, size) != 1:
            raise ValueError(
                f"generating_vector[{index}] = {component} is not coprime to "
                f"size = {size}; that coordinate would take fewer than "
                f"{size} distinct values"
            )
        reduced.append(component % size)
    return tuple(reduced)


def _product_weights(
    weights: Sequence[float] | None, dimension: int
) -> tuple[float, ...]:
    """Validate product weights ``gamma_j`` (default: all one).

    Raises:
        ValueError: If the length is not ``dimension`` or a weight is not
            finite and positive.

    References:
        [EQ-SAMP-LATTICE-002].

    """
    if weights is None:
        return (1.0,) * dimension
    values = tuple(
        require_finite_real(weight, name=f"weights[{index}]")
        for index, weight in enumerate(weights)
    )
    if len(values) != dimension:
        raise ValueError(
            f"weights must have one entry per dimension ({dimension}), "
            f"got {len(values)}"
        )
    if any(value <= 0.0 for value in values):
        raise ValueError("weights must be positive")
    return values


def _kernel_table(size: int, weight: float) -> list[float]:
    """Return ``1 + gamma B_2(m / n)`` for ``m = 0, ..., n - 1``.

    References:
        [EQ-SAMP-LATTICE-002].

    """
    table = []
    for residue in range(size):
        x = residue / size
        table.append(1.0 + weight * (x * x - x + 1.0 / 6.0))
    return table


def _mean_minus_one(products: Sequence[float]) -> float:
    """Return ``mean(products) - 1`` with compensated summation.

    References:
        [EQ-SAMP-LATTICE-002].

    """
    return math.fsum(value - 1.0 for value in products) / len(products)


def rank1_lattice(
    size: int,
    generating_vector: Sequence[int],
    *,
    shift: bool = False,
    seed: int | RandomStream | None = None,
) -> SamplingDesign:
    """Return the rank-1 lattice ``x_k = frac(k z / n)``, optionally shifted.

    Without a shift the points are the exact rationals ``(k z_j mod n) / n``
    and the first point is the origin. With ``shift=True`` one vector
    ``Delta ~ U[0, 1)^s`` is drawn and every point becomes
    ``frac(x_k + Delta)`` (a Cranley-Patterson rotation); each point is then
    uniform on the cube and coordinates are kept inside ``(0, 1)``.

    Args:
        size: Positive number of points ``n``.
        generating_vector: Integers ``z_j`` coprime to ``n`` (reduced modulo
            ``n``); its length is the dimension.
        shift: Apply one random shift modulo one to all points.
        seed: Seed or stream of the shift; ignored without a shift.

    Returns:
        Design with ``n`` points, method ``"lattice"``.

    Raises:
        TypeError: If ``size`` or a component of ``z`` is not an integer.
        ValueError: If ``size < 1``, ``z`` is empty, or a component is not
            coprime to ``n``.

    Equation:
        [EQ-SAMP-LATTICE-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-CRANLEY-PATTERSON-1976], [REF-L-ECUYER-2018].

    """
    n = _require_positive_integer(size, name="size")
    z = _generating_vector(n, generating_vector)
    if not shift:
        points = tuple(tuple(((k * zj) % n) / n for zj in z) for k in range(n))
        return SamplingDesign(points, "lattice", False, None)
    stream = resolve_stream(seed)
    generator = stream.generator()
    delta = [generator.random() for _ in z]
    shifted = tuple(
        tuple(
            min(max((((k * zj) % n) / n + dj) % 1.0, _TINY), 1.0 - _TINY)
            for zj, dj in zip(z, delta, strict=True)
        )
        for k in range(n)
    )
    return SamplingDesign(shifted, "lattice", True, stream.seed)


def lattice_squared_error(
    size: int,
    generating_vector: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Return the shift-averaged squared worst-case error of a lattice rule.

    ``e^2(z) = -1 + (1/n) sum_{k=0}^{n-1} prod_j (1 + gamma_j B_2(frac(k z_j / n)))``
    with ``B_2(x) = x^2 - x + 1/6``. It equals
    ``E_Delta sup_{|f| <= 1} (Q_Delta f - int f)^2`` for the randomly shifted
    lattice rule ``Q_Delta`` and the unit ball of the unanchored Sobolev
    space described in the module docstring, and also the squared worst-case
    error of the unshifted lattice in the space with the shift-invariant
    kernel ``prod_j (1 + gamma_j B_2(frac(x_j - y_j)))``. In one dimension it
    is ``gamma / (6 n^2)`` for every admissible ``z``.

    Args:
        size: Positive number of points ``n`` (need not be prime).
        generating_vector: Integers coprime to ``n``.
        weights: Positive product weights ``gamma_j``; all one by default.

    Returns:
        The non-negative criterion value ``e^2(z)``.

    Raises:
        TypeError: If ``size`` or a component of ``z`` is not an integer.
        ValueError: If the inputs are outside their domains.

    Equation:
        [EQ-SAMP-LATTICE-002].

    References:
        [REF-CRANLEY-PATTERSON-1976] (random shifts); derivation in
        ADR-0118.

    """
    n = _require_positive_integer(size, name="size")
    z = _generating_vector(n, generating_vector)
    gammas = _product_weights(weights, len(z))
    products = [1.0] * n
    for zj, gamma in zip(z, gammas, strict=True):
        table = _kernel_table(n, gamma)
        products = [product * table[(k * zj) % n] for k, product in enumerate(products)]
    return max(_mean_minus_one(products), 0.0)


@dataclass(frozen=True, slots=True)
class LatticeRule:
    """A constructed rank-1 lattice rule with its quality criterion.

    Args:
        size: Prime number of points ``n``.
        generating_vector: The chosen ``z`` with components in
            ``{1, ..., n - 1}``.
        weights: Product weights ``gamma_j`` of the criterion.
        squared_error: ``e^2(z)`` of [EQ-SAMP-LATTICE-002] in the full
            dimension.
        squared_error_by_dimension: ``e^2`` of the leading ``1, 2, ..., s``
            components.
        construction: ``"cbc"`` or ``"korobov"``.
        multiplier: The Korobov multiplier ``a`` (``None`` for CBC).
        candidates_evaluated: Number of candidate vectors whose criterion was
            computed; each costs ``O(n)`` per component.

    References:
        [EQ-SAMP-LATTICE-002], [EQ-SAMP-LATTICE-003], [EQ-SAMP-LATTICE-004].

    """

    size: int
    generating_vector: tuple[int, ...]
    weights: tuple[float, ...]
    squared_error: float
    squared_error_by_dimension: tuple[float, ...]
    construction: str
    multiplier: int | None
    candidates_evaluated: int

    @property
    def dimension(self) -> int:
        """Return the dimension ``s`` of the rule.

        References:
            [EQ-SAMP-LATTICE-001].

        """
        return len(self.generating_vector)

    @property
    def worst_case_error(self) -> float:
        """Return the root ``e(z)`` of the shift-averaged squared error.

        References:
            [EQ-SAMP-LATTICE-002].

        """
        return math.sqrt(self.squared_error)

    def points(
        self, *, shift: bool = False, seed: int | RandomStream | None = None
    ) -> SamplingDesign:
        """Return the lattice points of this rule, optionally shifted.

        Args:
            shift: Apply a Cranley-Patterson random shift.
            seed: Seed or stream of the shift.

        Returns:
            The design of :func:`rank1_lattice` for this rule.

        Equation:
            [EQ-SAMP-LATTICE-001].

        References:
            [REF-CRANLEY-PATTERSON-1976].

        """
        return rank1_lattice(self.size, self.generating_vector, shift=shift, seed=seed)


@lru_cache(maxsize=64)
def _cbc(
    size: int, gammas: tuple[float, ...]
) -> tuple[tuple[int, ...], tuple[float, ...], int]:
    """Run the greedy component-by-component search for prime ``size``.

    Returns:
        The vector, the criterion after every component, and the number of
        candidates evaluated.

    References:
        [EQ-SAMP-LATTICE-003].

    """
    products = [1.0] * size
    vector: list[int] = []
    history: list[float] = []
    evaluated = 0
    half = max(1, size // 2)
    for gamma in gammas:
        table = _kernel_table(size, gamma)
        best_value = math.inf
        best_component = 1
        # B_2 is symmetric about 1/2, so z and n - z give the same value;
        # the first component is always 1 (every choice gives the same e^2).
        candidates = range(1, half + 1) if vector else range(1, 2)
        for candidate in candidates:
            value = math.fsum(
                product * table[(k * candidate) % size]
                for k, product in enumerate(products)
            )
            evaluated += 1
            if value < best_value:
                best_value = value
                best_component = candidate
        vector.append(best_component)
        products = [
            product * table[(k * best_component) % size]
            for k, product in enumerate(products)
        ]
        history.append(max(_mean_minus_one(products), 0.0))
    return tuple(vector), tuple(history), evaluated


def cbc_generating_vector(
    size: int, dimension: int, *, weights: Sequence[float] | None = None
) -> LatticeRule:
    """Construct a generating vector component by component (greedy).

    With ``z_1 = 1``, each further component is
    ``z_d = argmin_{1 <= z <= n-1} e^2(z_1, ..., z_{d-1}, z)`` for the
    criterion of [EQ-SAMP-LATTICE-002], earlier components held fixed. Ties
    go to the smallest ``z`` (``z`` and ``n - z`` always tie, so only
    ``z <= n / 2`` is searched). The search is exhaustive in one and two
    dimensions, where it attains the minimum of ``e^2`` over all vectors, and
    a greedy heuristic beyond: it need not find the global minimum. This is
    the naive implementation with ``O(s n^2)`` operations; results for
    recently used ``(n, weights)`` pairs are cached.

    Args:
        size: Prime number of points ``n``.
        dimension: Positive dimension ``s``.
        weights: Positive product weights ``gamma_j``; all one by default.
            The result depends on their order: components are chosen in the
            order given.

    Returns:
        The constructed rule with ``e^2`` after every component.

    Raises:
        TypeError: If ``size`` or ``dimension`` is not an integer.
        ValueError: If ``size`` is not prime, ``dimension < 1``, or the
            weights are invalid.

    Equation:
        [EQ-SAMP-LATTICE-003], [EQ-SAMP-LATTICE-002].

    References:
        [REF-L-ECUYER-2018] (lattice rules in randomized quasi-Monte Carlo);
        the search as implemented is specified in ADR-0118.

    """
    n = _require_prime_size(size)
    s = _require_positive_integer(dimension, name="dimension")
    gammas = _product_weights(weights, s)
    vector, history, evaluated = _cbc(n, gammas)
    return LatticeRule(n, vector, gammas, history[-1], history, "cbc", None, evaluated)


def korobov_vector(size: int, dimension: int, multiplier: int) -> tuple[int, ...]:
    """Return the Korobov vector ``(1, a, a^2, ..., a^(s-1)) mod n``.

    Args:
        size: Number of points ``n >= 2``.
        dimension: Positive dimension ``s``.
        multiplier: Integer ``a`` coprime to ``n``.

    Returns:
        The generating vector with components in ``{1, ..., n - 1}``.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If ``n < 2``, ``s < 1``, or ``a`` is not coprime to ``n``.

    Equation:
        [EQ-SAMP-LATTICE-004].

    References:
        [REF-L-ECUYER-2018].

    """
    n = _require_positive_integer(size, name="size")
    s = _require_positive_integer(dimension, name="dimension")
    if n < 2:
        raise ValueError("size must be at least 2")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        raise TypeError("multiplier must be an integer")
    if math.gcd(multiplier, n) != 1:
        raise ValueError(f"multiplier {multiplier} is not coprime to size {n}")
    vector = [1]
    for _ in range(s - 1):
        vector.append((vector[-1] * multiplier) % n)
    return tuple(vector)


def korobov_generating_vector(
    size: int, dimension: int, *, weights: Sequence[float] | None = None
) -> LatticeRule:
    """Search the Korobov vectors ``(1, a, ..., a^(s-1))`` for the best ``a``.

    Every multiplier ``1 <= a <= n / 2`` is tried (``a`` and ``n - a`` give
    the same criterion) and the one with the smallest ``e^2`` of
    [EQ-SAMP-LATTICE-002] in the full dimension is kept; ties go to the
    smallest ``a``. The search is exhaustive over the Korobov family, which
    is a subset of all generating vectors, at ``O(s n^2)`` operations.

    Args:
        size: Prime number of points ``n``.
        dimension: Positive dimension ``s``.
        weights: Positive product weights ``gamma_j``; all one by default.

    Returns:
        The best Korobov rule, with its multiplier.

    Raises:
        TypeError: If ``size`` or ``dimension`` is not an integer.
        ValueError: If ``size`` is not prime, ``dimension < 1``, or the
            weights are invalid.

    Equation:
        [EQ-SAMP-LATTICE-004], [EQ-SAMP-LATTICE-002].

    References:
        [REF-L-ECUYER-2018].

    """
    n = _require_prime_size(size)
    s = _require_positive_integer(dimension, name="dimension")
    gammas = _product_weights(weights, s)
    tables = [_kernel_table(n, gamma) for gamma in gammas]
    best_value = math.inf
    best_multiplier = 1
    candidates = range(1, max(1, n // 2) + 1)
    for multiplier in candidates:
        vector = korobov_vector(n, s, multiplier)
        products = [1.0] * n
        for zj, table in zip(vector, tables, strict=True):
            products = [
                product * table[(k * zj) % n] for k, product in enumerate(products)
            ]
        value = math.fsum(products)
        if value < best_value:
            best_value = value
            best_multiplier = multiplier
    chosen = korobov_vector(n, s, best_multiplier)
    history = tuple(
        lattice_squared_error(n, chosen[:d], weights=gammas[:d])
        for d in range(1, s + 1)
    )
    return LatticeRule(
        n,
        chosen,
        gammas,
        history[-1],
        history,
        "korobov",
        best_multiplier,
        len(candidates),
    )
