"""Space-filling sampling designs on the open unit hypercube.

Designs return points in ``(0, 1)^d``. Mapping them to physical inputs is the
job of a joint input law (``probstat.modeling``), which applies inverse
marginal CDFs and a dependence structure. Randomized designs take an integer
seed or a :class:`~probstat.foundations.RandomStream`.

Sobol points use the Joe-Kuo ``new-joe-kuo-6.21201`` direction numbers for
up to 1024 dimensions (:mod:`probstat.sampling._joe_kuo`, which carries their
licence). Unscrambled Sobol points are shifted to the centres of their
finest dyadic cells by default, so no coordinate is ``0`` (the origin would
map to ``-inf`` or a far-tail value under an inverse CDF); randomized
variants are a digital shift or Owen's nested uniform scrambling. Randomized
Halton points use independent uniform digit permutations at every digit
position (zero is permuted too), which makes every point uniform on the unit
cube. Latin hypercubes can be optimized for the Morris-Mitchell maximin
criterion or given a target rank correlation by the Iman-Conover method.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions.normal import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import cholesky, solve_lower

from ._joe_kuo import JOE_KUO_MAX_DIMENSION, joe_kuo_table

Point = tuple[float, ...]


def _first_primes(count: int) -> tuple[int, ...]:
    """Return the first ``count`` primes by trial division.

    References:
        [REF-HALTON-1960]. No equation is evaluated.

    """
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


HALTON_MAX_DIMENSION = 1024
_PRIMES = _first_primes(HALTON_MAX_DIMENSION)

# Primitive-polynomial degree s, inner-coefficient code a, and initial odd
# direction integers m_1..m_s for Sobol dimensions 2..1024 (dimension 1 is the
# van der Corput sequence in base two), from Joe and Kuo (2008).
_SOBOL_TABLE: tuple[tuple[int, int, tuple[int, ...]], ...] = joe_kuo_table()
SOBOL_MAX_DIMENSION = JOE_KUO_MAX_DIMENSION
_SOBOL_BITS = 52
SCRAMBLES = ("shift", "owen")


class SobolBalanceWarning(UserWarning):
    """Warn that a Sobol design size is not a power of two.

    Only blocks of ``2^m`` consecutive points (starting at a multiple of
    ``2^m``) inherit the ``(t, m, s)``-net balance of the sequence.

    References:
        [REF-OWEN-2013, Section 15.7].

    """


@dataclass(frozen=True, slots=True)
class SamplingDesign:
    """An immutable point set on the unit hypercube with provenance.

    Args:
        points: Points in ``(0, 1)^d``.
        method: Short design name such as ``"lhs"`` or ``"sobol"``.
        randomized: Whether a random scramble, shift, or permutation was used.
        seed: Seed of the randomization stream, if any.

    References:
        [REF-OWEN-2013, Chapters 10 and 17].

    """

    points: tuple[Point, ...]
    method: str
    randomized: bool
    seed: int | None = None

    @property
    def size(self) -> int:
        """Return the number of points.

        References:
            [REF-OWEN-2013, Chapter 10].

        """
        return len(self.points)

    @property
    def dimension(self) -> int:
        """Return the hypercube dimension.

        References:
            [REF-OWEN-2013, Chapter 10].

        """
        return len(self.points[0]) if self.points else 0


def _check_size_dimension(size: int, dimension: int) -> None:
    """Validate positive integer design size and dimension.

    Args:
        size: Number of points.
        dimension: Hypercube dimension.

    Raises:
        TypeError: If either value is not an integer.
        ValueError: If either value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    for value, name in ((size, "size"), (dimension, "dimension")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")


def _open_unit(value: float) -> float:
    """Clamp a coordinate into the open interval ``(0, 1)``.

    Inverse CDFs are infinite at 0 and 1, so design points are nudged inward
    by the smallest representable amount.

    Args:
        value: Coordinate in ``[0, 1]``.

    Returns:
        Coordinate in ``(0, 1)``.

    References:
        [REF-OWEN-2013, Section 17.1]. No equation is evaluated.

    """
    tiny = 2.0**-53
    return min(max(value, tiny), 1.0 - tiny)


def monte_carlo_design(
    size: int, dimension: int, *, seed: int | RandomStream | None = None
) -> SamplingDesign:
    """Return independent uniform points (plain Monte Carlo).

    Args:
        size: Positive number of points.
        dimension: Positive hypercube dimension.
        seed: Integer seed or stream.

    Returns:
        IID uniform design.

    Equation:
        [EQ-MC-MEAN-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    _check_size_dimension(size, dimension)
    stream = resolve_stream(seed)
    generator = stream.generator()
    points = tuple(
        tuple(_open_unit(generator.random()) for _ in range(dimension))
        for _ in range(size)
    )
    return SamplingDesign(points, "mc", True, stream.seed)


def latin_hypercube(
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
    centered: bool = False,
) -> SamplingDesign:
    """Return a Latin hypercube design with one point per marginal stratum.

    Args:
        size: Positive number of points and strata per axis.
        dimension: Positive hypercube dimension.
        seed: Integer seed or stream for the permutations and jitter.
        centered: Place points at stratum midpoints instead of jittering.

    Returns:
        Design whose every one-dimensional projection has exactly one point in
        each interval ``[k/n, (k+1)/n)``.

    Equation:
        [EQ-SAMP-LHS-001].

    References:
        [REF-MCKAY-BECKMAN-CONOVER-1979], [REF-OWEN-2013, Section 10.3].

    """
    _check_size_dimension(size, dimension)
    stream = resolve_stream(seed)
    generator = stream.generator()
    columns: list[list[float]] = []
    for _ in range(dimension):
        permutation = list(range(size))
        generator.shuffle(permutation)
        columns.append(
            [
                _open_unit((stratum + (0.5 if centered else generator.random())) / size)
                for stratum in permutation
            ]
        )
    points = tuple(tuple(column[row] for column in columns) for row in range(size))
    return SamplingDesign(points, "lhs", True, stream.seed)


def radical_inverse(
    index: int, base: int, permutation: tuple[int, ...] | None = None
) -> float:
    """Return the (optionally digit-permuted) radical inverse of an integer.

    Args:
        index: Non-negative integer.
        base: Integer base at least two.
        permutation: Optional permutation of ``range(base)`` fixing zero.

    Returns:
        ``sum_k pi(d_k) base^-(k+1)`` for base-``b`` digits ``d_k``.

    Raises:
        ValueError: If arguments are out of range.

    Equation:
        [EQ-SAMP-HALTON-001].

    References:
        [REF-HALTON-1960], [REF-OWEN-2013, Section 15.4].

    """
    if index < 0 or base < 2:
        raise ValueError("index must be non-negative and base at least two")
    result = 0.0
    factor = 1.0 / base
    value = index
    while value > 0:
        value, digit = divmod(value, base)
        result += (permutation[digit] if permutation else digit) * factor
        factor /= base
    return result


def _digit_count(base: int) -> int:
    """Return ``K`` with ``base^-K <= 2^-53``: digits carried by scrambled Halton.

    References:
        [REF-OWEN-1995]. No equation is evaluated.

    """
    count = 1
    while base**count < 2**53:
        count += 1
    return count


def _scrambled_radical_inverse(
    index: int, base: int, permutations: Sequence[Sequence[int]]
) -> float:
    """Return ``sum_k pi_k(d_k) base^-k`` over ``K`` digits (leading zeros included).

    The ``K = len(permutations)`` digits include the infinitely many leading
    zeros of the expansion up to the float resolution, so a permutation that
    moves zero changes every digit position below the index's length too.

    References:
        [EQ-SAMP-HALTON-002], [REF-MATOUSEK-1998].

    """
    result = 0.0
    factor = 1.0 / base
    value = index
    for permutation in permutations:
        value, digit = divmod(value, base)
        result += permutation[digit] * factor
        factor /= base
    return result


def halton(
    size: int,
    dimension: int,
    *,
    skip: int = 1,
    seed: int | RandomStream | None = None,
    scramble: bool = False,
    shift: bool = False,
) -> SamplingDesign:
    """Return a Halton low-discrepancy design, optionally randomized.

    ``scramble=True`` applies an independent, uniformly random permutation of
    ``{0, ..., b-1}`` at every digit position of every axis, including the
    leading zero digits up to ``b^-K <= 2^-53``, plus a uniform jitter below
    the last digit. Each scrambled point is then exactly uniform on
    ``(0, 1)^d`` (so the estimator is unbiased) while the van der Corput
    stratification of every axis is kept: points sharing their first ``k``
    digits still share them after scrambling. (Earlier releases fixed zero
    in the permutation, which leaves the leading zeros unscrambled and
    biases the mean towards zero.) ``shift=True`` adds a Cranley-Patterson
    rotation modulo one.

    Args:
        size: Positive number of points.
        dimension: Dimension at most 1024 (one prime base per axis).
        skip: Number of leading indices to skip; the default drops the origin.
        seed: Seed for scrambling or shifting.
        scramble: Apply random digit permutations (zero is permuted too).
        shift: Apply a Cranley-Patterson random shift modulo one per axis.

    Returns:
        Halton design in ``(0, 1)^d``.

    Raises:
        ValueError: If the dimension exceeds the prime table or ``skip < 0``.

    Equation:
        [EQ-SAMP-HALTON-001], [EQ-SAMP-HALTON-002], [EQ-SAMP-RQMC-001].

    References:
        [REF-HALTON-1960], [REF-MATOUSEK-1998], [REF-CRANLEY-PATTERSON-1976],
        [REF-OWEN-2013, Sections 15.4 and 17.2].

    """
    _check_size_dimension(size, dimension)
    if dimension > len(_PRIMES):
        raise ValueError(f"halton supports at most {len(_PRIMES)} dimensions")
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise ValueError("skip must be a non-negative integer")
    randomized = scramble or shift
    stream = resolve_stream(seed) if randomized else None
    generator = stream.generator() if stream else None
    tables: list[list[list[int]] | None] = []
    shifts: list[float] = []
    for axis in range(dimension):
        base = _PRIMES[axis]
        if scramble and generator is not None:
            permutations: list[list[int]] = []
            for _ in range(_digit_count(base)):
                digits = list(range(base))
                generator.shuffle(digits)
                permutations.append(digits)
            tables.append(permutations)
        else:
            tables.append(None)
        shifts.append(generator.random() if shift and generator is not None else 0.0)
    points: list[Point] = []
    for index in range(skip, skip + size):
        point: list[float] = []
        for axis in range(dimension):
            base = _PRIMES[axis]
            table = tables[axis]
            if table is None:
                value = radical_inverse(index, base)
            else:
                assert generator is not None
                jitter = generator.random() * float(base) ** -len(table)
                value = _scrambled_radical_inverse(index, base, table) + jitter
            point.append(_open_unit((value + shifts[axis]) % 1.0))
        points.append(tuple(point))
    return SamplingDesign(
        tuple(points), "halton", randomized, stream.seed if stream else None
    )


def _sobol_directions(dimension: int) -> list[list[int]]:
    """Return 52-bit Sobol direction integers ``v_k`` for each dimension.

    Args:
        dimension: Number of dimensions, at most ``SOBOL_MAX_DIMENSION``.

    Returns:
        Direction integers, already shifted to the left of a 52-bit word.

    Equation:
        [EQ-SAMP-SOBOL-001].

    References:
        [REF-JOE-KUO-2008], [REF-BRATLEY-FOX-1988].

    """
    directions: list[list[int]] = [
        [1 << (_SOBOL_BITS - 1 - k) for k in range(_SOBOL_BITS)]
    ]
    for degree, code, initial in _SOBOL_TABLE[: dimension - 1]:
        m = list(initial)
        for k in range(degree, _SOBOL_BITS):
            value = m[k - degree] ^ (m[k - degree] << degree)
            for j in range(1, degree):
                if (code >> (degree - 1 - j)) & 1:
                    value ^= m[k - j] << j
            m.append(value)
        directions.append([m[k] << (_SOBOL_BITS - 1 - k) for k in range(_SOBOL_BITS)])
    return directions


def _owen_scramble(
    values: list[int], bits: int, generator: object, tree: dict[tuple[int, int], int]
) -> list[int]:
    """Apply Owen's nested uniform scrambling to ``bits``-digit integers.

    Digit ``k`` (from the most significant) is flipped by a random bit that
    depends on the digits above it; flip bits are drawn lazily from the
    seeded generator and stored in ``tree`` keyed by ``(k, prefix)``. The
    ``52 - bits`` low digits, which are zero in the unscrambled points and
    sit below distinct prefixes, become independent uniform bits.

    References:
        [EQ-SAMP-OWEN-001], [REF-OWEN-1995].

    """
    draw = generator.getrandbits  # type: ignore[attr-defined]
    low_bits = _SOBOL_BITS - bits
    scrambled: list[int] = []
    for value in values:
        top = value >> low_bits
        result = 0
        for k in range(bits):
            prefix = top >> (bits - k)
            key = (k, prefix)
            flip = tree.get(key)
            if flip is None:
                flip = draw(1)
                tree[key] = flip
            digit = ((top >> (bits - 1 - k)) & 1) ^ flip
            result = (result << 1) | digit
        tail = draw(low_bits) if low_bits > 0 else 0
        scrambled.append((result << low_bits) | tail)
    return scrambled


def sobol(
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
    scramble: bool | str = False,
    skip: int = 0,
    centered: bool | None = None,
) -> SamplingDesign:
    """Return a Sobol digital-net design in Gray-code order (up to 1024 dimensions).

    Powers of two preserve the net balance properties; any other size emits
    a :class:`SobolBalanceWarning` (the points are valid but less uniform).

    Randomization (``scramble``):

    * ``False``: deterministic points. With ``centered`` (the default for
      unscrambled points) each coordinate ``k / 2^m`` (``2^m`` the smallest
      power of two ``>= skip + size``) is moved to the centre
      ``(k + 1/2) / 2^m`` of its cell: the first point is no longer the
      origin, the elementary-interval balance is unchanged, and the
      coordinate mean of ``2^m`` points is exactly ``1/2``. ``centered=False``
      returns the raw points (the origin is then clamped to ``2^-53``).
    * ``True`` or ``"shift"``: a random digital shift (XOR) per axis, which
      keeps every balance property and makes each point uniform.
    * ``"owen"``: Owen's nested uniform scrambling, which keeps the net
      property, makes each point uniform, and gives the variance reduction of
      scrambled nets for smooth integrands.

    Args:
        size: Positive number of points.
        dimension: Dimension at most ``SOBOL_MAX_DIMENSION`` (1024).
        seed: Seed for the randomization.
        scramble: ``False``, ``True``/``"shift"``, or ``"owen"``.
        skip: Number of leading Gray-code points to skip.
        centered: Centre unscrambled points in their cells (default: true
            when unscrambled; ignored for scrambled points).

    Returns:
        Sobol design in ``(0, 1)^d``.

    Raises:
        ValueError: If the dimension is unsupported, ``skip < 0``, or the
            scramble is unknown.

    Equation:
        [EQ-SAMP-SOBOL-001], [EQ-SAMP-RQMC-001], [EQ-SAMP-OWEN-001].

    References:
        [REF-SOBOL-1967], [REF-JOE-KUO-2008], [REF-OWEN-1995],
        [REF-OWEN-2013, Section 15.7].

    """
    _check_size_dimension(size, dimension)
    if dimension > SOBOL_MAX_DIMENSION:
        raise ValueError(f"sobol supports at most {SOBOL_MAX_DIMENSION} dimensions")
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise ValueError("skip must be a non-negative integer")
    if scramble is True:
        mode = "shift"
    elif scramble is False:
        mode = "none"
    elif scramble in SCRAMBLES:
        mode = str(scramble)
    else:
        raise ValueError("scramble must be False, True, 'shift', or 'owen'")
    if size & (size - 1):
        warnings.warn(
            f"sobol size {size} is not a power of two; only 2^m points keep the "
            "net balance properties",
            SobolBalanceWarning,
            stacklevel=2,
        )
    total = skip + size
    bits = max(1, (total - 1).bit_length())
    if bits > _SOBOL_BITS:
        raise ValueError("too many Sobol points for 52-bit direction numbers")
    directions = _sobol_directions(dimension)
    stream = resolve_stream(seed) if mode != "none" else None
    generator = stream.generator() if stream else None
    shifts = [
        generator.getrandbits(_SOBOL_BITS)
        if (generator is not None and mode == "shift")
        else 0
        for _ in range(dimension)
    ]
    centre = 0
    if mode == "none" and (centered is None or centered):
        centre = 1 << (_SOBOL_BITS - bits - 1)
    state = [0] * dimension
    raw: list[list[int]] = []
    for index in range(total):
        if index >= skip:
            raw.append(list(state))
        changed = ((index + 1) & -(index + 1)).bit_length() - 1
        for axis in range(dimension):
            state[axis] ^= directions[axis][changed]
    if mode == "owen":
        assert generator is not None
        columns = []
        for axis in range(dimension):
            tree: dict[tuple[int, int], int] = {}
            columns.append(
                _owen_scramble([row[axis] for row in raw], bits, generator, tree)
            )
        raw = [[columns[axis][i] for axis in range(dimension)] for i in range(size)]
    scale = 2.0**-_SOBOL_BITS
    points = tuple(
        tuple(
            _open_unit(((row[axis] ^ shifts[axis]) + centre) * scale)
            for axis in range(dimension)
        )
        for row in raw
    )
    return SamplingDesign(
        points, "sobol", mode != "none", stream.seed if stream else None
    )


def _pair_distances(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the symmetric matrix of Euclidean distances.

    References:
        [REF-MORRIS-MITCHELL-1995]. No equation is evaluated.

    """
    count = len(points)
    distances = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            value = math.dist(points[i], points[j])
            distances[i][j] = distances[j][i] = value
    return distances


def maximin_criterion(design: SamplingDesign, *, power: int = 15) -> float:
    """Return the Morris-Mitchell criterion ``phi_p = (sum_(i<j) d_ij^-p)^(1/p)``.

    Smaller is better; as ``p -> inf`` it orders designs by their minimum
    pairwise distance (then by how often it occurs).

    Args:
        design: Design with at least two distinct points.
        power: Positive integer exponent ``p``.

    Returns:
        ``phi_p`` (``inf`` if two points coincide).

    Equation:
        [EQ-SAMP-MAXIMIN-001].

    References:
        [REF-MORRIS-MITCHELL-1995].

    """
    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    total = 0.0
    points = design.points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.dist(points[i], points[j])
            if d == 0.0:
                return math.inf
            total += d**-power
    return math.pow(total, 1.0 / power)


def maximin_latin_hypercube(
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
    iterations: int = 2000,
    power: int = 15,
) -> SamplingDesign:
    """Return a Latin hypercube improved for the Morris-Mitchell maximin criterion.

    Starting from :func:`latin_hypercube` (stream ``child("start")``), a
    seeded stochastic search swaps the values of two random rows in one
    random column (which preserves the Latin property) and keeps the swap
    when ``phi_p`` [EQ-SAMP-MAXIMIN-001] decreases. Each proposal costs
    ``O(n)`` distance updates.

    Args:
        size: Number of points (at least two).
        dimension: Positive dimension.
        seed: Integer seed or stream.
        iterations: Number of proposed swaps.
        power: Exponent ``p`` of the criterion.

    Returns:
        Latin hypercube design with method ``"lhs-maximin"``.

    Raises:
        ValueError: If ``size < 2`` or ``iterations < 0``.

    Equation:
        [EQ-SAMP-LHS-001], [EQ-SAMP-MAXIMIN-001].

    References:
        [REF-MORRIS-MITCHELL-1995], [REF-MCKAY-BECKMAN-CONOVER-1979].

    """
    _check_size_dimension(size, dimension)
    if size < 2:
        raise ValueError("a maximin design needs at least two points")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 0
    ):
        raise ValueError("iterations must be a non-negative integer")
    stream = resolve_stream(seed)
    start = latin_hypercube(size, dimension, seed=stream.child("start"))
    rows = [list(point) for point in start.points]
    distances = _pair_distances(rows)
    terms = [
        [0.0 if i == j else distances[i][j] ** -power for j in range(size)]
        for i in range(size)
    ]
    generator = stream.child("swaps").generator()
    for _ in range(iterations):
        axis = generator.randrange(dimension)
        first, second = generator.sample(range(size), 2)
        rows[first][axis], rows[second][axis] = rows[second][axis], rows[first][axis]
        changed = (first, second)
        new_terms: dict[tuple[int, int], float] = {}
        delta = 0.0
        for i in changed:
            for j in range(size):
                if j == i or (j in changed and j < i):
                    continue
                d = math.dist(rows[i], rows[j])
                value = d**-power if d > 0.0 else math.inf
                new_terms[(i, j)] = value
                delta += value - terms[i][j]
        if delta < 0.0:
            for (i, j), value in new_terms.items():
                terms[i][j] = terms[j][i] = value
        else:
            rows[first][axis], rows[second][axis] = (
                rows[second][axis],
                rows[first][axis],
            )
    points = tuple(tuple(row) for row in rows)
    return SamplingDesign(points, "lhs-maximin", True, stream.seed)


def _ranks(values: Sequence[float]) -> list[int]:
    """Return the 0-based rank of each value (ties broken by position).

    References:
        [REF-IMAN-CONOVER-1982]. No equation is evaluated.

    """
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def iman_conover(
    design: SamplingDesign,
    correlation: Sequence[Sequence[float]],
    *,
    seed: int | RandomStream | None = None,
) -> SamplingDesign:
    """Rearrange each column of a design to induce a target rank correlation.

    Iman and Conover (1982): build a score matrix ``S`` whose columns are
    random permutations of the van der Waerden scores
    ``Phi^-1(i/(n+1))``, factor its sample correlation ``E = Q Q'`` and the
    target ``C = P P'``, and form ``T = S (Q^-1)' P'``, whose correlation is
    ``C``. Each design column is then reordered to have the ranks of the
    matching column of ``T``. Only the pairing of values changes, so every
    one-dimensional projection (for example the strata of a Latin
    hypercube) is preserved exactly; the achieved Spearman correlation is
    close to ``C`` but not equal to it.

    Args:
        design: Design with at least ``d + 2`` points.
        correlation: Symmetric positive-definite ``d x d`` target matrix with
            unit diagonal.
        seed: Seed of the score permutations.

    Returns:
        Design with method ``"<method>-iman-conover"``.

    Raises:
        ValueError: If the matrix has the wrong shape, is not a correlation
            matrix, or is not positive definite.

    Equation:
        [EQ-SAMP-IC-001].

    References:
        [REF-IMAN-CONOVER-1982].

    """
    size, dimension = design.size, design.dimension
    target = tuple(tuple(float(v) for v in row) for row in correlation)
    if len(target) != dimension or any(len(row) != dimension for row in target):
        raise ValueError("correlation must be a d x d matrix")
    for i in range(dimension):
        if abs(target[i][i] - 1.0) > 1e-12:
            raise ValueError("correlation must have a unit diagonal")
        for j in range(i):
            if abs(target[i][j] - target[j][i]) > 1e-12:
                raise ValueError("correlation must be symmetric")
    if size < dimension + 2:
        raise ValueError("need at least dimension + 2 points")
    p_factor = cholesky(target)
    standard = Normal()
    scores = [standard.quantile((i + 1) / (size + 1)) for i in range(size)]
    generator = resolve_stream(seed).child("iman-conover").generator()
    columns: list[list[float]] = []
    for _ in range(dimension):
        column = list(scores)
        generator.shuffle(column)
        columns.append(column)
    means = [math.fsum(c) / size for c in columns]
    sample = [
        [
            math.fsum(
                (columns[a][k] - means[a]) * (columns[b][k] - means[b])
                for k in range(size)
            )
            for b in range(dimension)
        ]
        for a in range(dimension)
    ]
    scale = [math.sqrt(sample[a][a]) for a in range(dimension)]
    empirical = tuple(
        tuple(sample[a][b] / (scale[a] * scale[b]) for b in range(dimension))
        for a in range(dimension)
    )
    q_factor = cholesky(empirical)
    transformed_columns: list[list[float]] = [[0.0] * size for _ in range(dimension)]
    for k in range(size):
        row = tuple(columns[a][k] for a in range(dimension))
        whitened = solve_lower(q_factor, row)
        for a in range(dimension):
            transformed_columns[a][k] = math.fsum(
                p_factor[a][b] * whitened[b] for b in range(a + 1)
            )
    reordered: list[list[float]] = []
    for a in range(dimension):
        sorted_values = sorted(point[a] for point in design.points)
        ranks = _ranks(transformed_columns[a])
        reordered.append([sorted_values[rank] for rank in ranks])
    points = tuple(
        tuple(reordered[a][k] for a in range(dimension)) for k in range(size)
    )
    return SamplingDesign(points, f"{design.method}-iman-conover", True, design.seed)


def star_discrepancy_bound(design: SamplingDesign, *, grid: int = 16) -> float:
    """Return a grid-based lower bound on the star discrepancy of a design.

    The exact star discrepancy is NP-hard; this evaluates the local
    discrepancy at anchored boxes whose corners lie on a regular grid, which
    gives a lower bound useful for comparing designs of equal size.

    Args:
        design: Design with dimension at most four.
        grid: Positive number of grid cells per axis.

    Returns:
        ``max_box |#points in box / n - volume(box)|`` over grid boxes.

    Raises:
        ValueError: If the dimension exceeds four or ``grid`` is not positive.

    Equation:
        [EQ-SAMP-DISC-001].

    References:
        [REF-OWEN-2013, Section 15.2].

    """
    if design.dimension > 4:
        raise ValueError("grid discrepancy is limited to four dimensions")
    if grid < 1:
        raise ValueError("grid must be positive")
    corners = [(k + 1) / grid for k in range(grid)]
    worst = 0.0
    count = design.size

    def visit(prefix: tuple[float, ...]) -> None:
        """Evaluate every anchored box extending ``prefix``.

        References:
            [EQ-SAMP-DISC-001], [REF-OWEN-2013, Section 15.2].

        """
        nonlocal worst
        if len(prefix) == design.dimension:
            inside = sum(
                all(p < c for p, c in zip(point, prefix, strict=True))
                for point in design.points
            )
            volume = math.prod(prefix)
            worst = max(worst, abs(inside / count - volume))
            return
        for corner in corners:
            visit((*prefix, corner))

    visit(())
    return worst
