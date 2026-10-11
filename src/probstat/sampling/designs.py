"""Space-filling sampling designs on the open unit hypercube.

Designs return points in ``(0, 1)^d``. Mapping them to physical inputs is the
job of a joint input law (``probstat.modeling``), which applies inverse
marginal CDFs and a dependence structure. Randomized designs take an integer
seed or a :class:`~probstat.foundations.RandomStream`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream

Point = tuple[float, ...]

_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
)

# Primitive-polynomial degree s, inner-coefficient code a, and initial odd
# direction integers m_1..m_s for Sobol dimensions 2..16 (dimension 1 is the
# van der Corput sequence in base two).
_SOBOL_TABLE: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 1, 3, 3)),
    (4, 4, (1, 3, 5, 13)),
    (5, 2, (1, 1, 5, 5, 17)),
    (5, 4, (1, 1, 5, 5, 5)),
    (5, 7, (1, 1, 7, 11, 19)),
    (5, 11, (1, 1, 5, 1, 1)),
    (5, 13, (1, 1, 1, 3, 11)),
    (5, 14, (1, 3, 5, 5, 31)),
    (6, 1, (1, 3, 3, 9, 7, 49)),
    (6, 13, (1, 1, 1, 15, 21, 21)),
    (6, 16, (1, 3, 1, 13, 27, 49)),
)
SOBOL_MAX_DIMENSION = len(_SOBOL_TABLE) + 1
_SOBOL_BITS = 52


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

    Args:
        size: Positive number of points.
        dimension: Dimension at most 50 (one prime base per axis).
        skip: Number of leading indices to skip; the default drops the origin.
        seed: Seed for scrambling or shifting.
        scramble: Apply a random digit permutation per axis (zero fixed).
        shift: Apply a Cranley-Patterson random shift modulo one per axis.

    Returns:
        Halton design in ``(0, 1)^d``.

    Raises:
        ValueError: If the dimension exceeds the prime table or ``skip < 0``.

    Equation:
        [EQ-SAMP-HALTON-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-HALTON-1960], [REF-CRANLEY-PATTERSON-1976],
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
    permutations: list[tuple[int, ...] | None] = []
    shifts: list[float] = []
    for axis in range(dimension):
        base = _PRIMES[axis]
        if scramble and generator is not None:
            tail = list(range(1, base))
            generator.shuffle(tail)
            permutations.append((0, *tail))
        else:
            permutations.append(None)
        shifts.append(generator.random() if shift and generator is not None else 0.0)
    points = tuple(
        tuple(
            _open_unit(
                (
                    radical_inverse(index, _PRIMES[axis], permutations[axis])
                    + shifts[axis]
                )
                % 1.0
            )
            for axis in range(dimension)
        )
        for index in range(skip, skip + size)
    )
    return SamplingDesign(points, "halton", randomized, stream.seed if stream else None)


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


def sobol(
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
    scramble: bool = False,
    skip: int = 0,
) -> SamplingDesign:
    """Return a Sobol digital-net design in Gray-code order.

    Powers of two preserve the net balance properties; other sizes are valid
    but less uniform. ``scramble=True`` applies a random digital shift (XOR),
    which keeps every elementary-interval balance of the unscrambled net while
    making the estimator unbiased.

    Args:
        size: Positive number of points.
        dimension: Dimension at most ``SOBOL_MAX_DIMENSION`` (16).
        seed: Seed for the digital shift.
        scramble: Apply a random digital shift per axis.
        skip: Number of leading Gray-code points to skip.

    Returns:
        Sobol design in ``(0, 1)^d``.

    Raises:
        ValueError: If the dimension is unsupported or ``skip < 0``.

    Equation:
        [EQ-SAMP-SOBOL-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-SOBOL-1967], [REF-JOE-KUO-2008], [REF-OWEN-2013, Section 15.7].

    """
    _check_size_dimension(size, dimension)
    if dimension > SOBOL_MAX_DIMENSION:
        raise ValueError(f"sobol supports at most {SOBOL_MAX_DIMENSION} dimensions")
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise ValueError("skip must be a non-negative integer")
    directions = _sobol_directions(dimension)
    stream = resolve_stream(seed) if scramble else None
    generator = stream.generator() if stream else None
    shifts = [
        generator.getrandbits(_SOBOL_BITS) if generator is not None else 0
        for _ in range(dimension)
    ]
    state = [0] * dimension
    scale = 2.0**-_SOBOL_BITS
    points: list[Point] = []
    for index in range(skip + size):
        if index >= skip:
            points.append(
                tuple(
                    _open_unit((state[axis] ^ shifts[axis]) * scale)
                    for axis in range(dimension)
                )
            )
        changed = ((index + 1) & -(index + 1)).bit_length() - 1
        for axis in range(dimension):
            state[axis] ^= directions[axis][changed]
    return SamplingDesign(
        tuple(points), "sobol", scramble, stream.seed if stream else None
    )


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
