"""Shannon entropy and divergences for finite probability mass functions.

A PMF is a finite sequence of non-negative probabilities summing to one
within ``1e-9``; a joint PMF is a rectangular table whose rows index ``X``
and whose columns index ``Y``. The convention ``0 log 0 = 0`` is used
throughout, and every quantity is returned in nats or bits.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable

from probstat.foundations import require_finite_real

from ._units import InformationUnit, unit_factor

_SUM_TOLERANCE = 1e-9
_SMALLEST_NORMAL = sys.float_info.min
_LOG_TWO = math.log(2.0)
Table = tuple[tuple[float, ...], ...]


def _log_ratio(numerator: float, denominator: float) -> float:
    """Return ``log(numerator/denominator)`` for positive arguments.

    The quotient is used when it is a finite normal float (one rounding);
    when it would overflow or underflow, ``log numerator - log denominator``
    is returned instead, so extreme but valid probabilities never produce
    ``inf``, ``0`` or a division error.

    Args:
        numerator: Positive finite value.
        denominator: Positive finite value.

    Returns:
        Natural logarithm of the ratio.

    References:
        [REF-COVER-THOMAS-2006, Section 2.3]. Floating-point evaluation of
        the log-ratio in the divergences; no equation is evaluated.

    """
    ratio = numerator / denominator
    if _SMALLEST_NORMAL <= ratio < math.inf:
        return math.log(ratio)
    return math.log(numerator) - math.log(denominator)


def _pmf(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    """Validate a finite PMF.

    Args:
        values: Candidate probabilities.
        name: Argument name for error messages.

    Returns:
        Probabilities as floats.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the PMF is empty, has a negative or non-finite entry,
            or does not sum to one within ``1e-9``.

    References:
        [REF-COVER-THOMAS-2006, Section 2.1]. No equation is evaluated.

    """
    probabilities = tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not probabilities:
        raise ValueError(f"{name} must not be empty")
    if any(p < 0.0 for p in probabilities):
        raise ValueError(f"{name} entries must be non-negative")
    if abs(math.fsum(probabilities) - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to one")
    return probabilities


def _joint(table: Iterable[Iterable[object]]) -> Table:
    """Validate a rectangular joint PMF table.

    Args:
        table: Rows of joint probabilities ``p(x, y)``.

    Returns:
        Validated table.

    Raises:
        ValueError: If the table is ragged, empty, negative, or not
            normalized.

    References:
        [REF-COVER-THOMAS-2006, Section 2.2]. No equation is evaluated.

    """
    rows = tuple(
        tuple(require_finite_real(value, name="joint") for value in row)
        for row in table
    )
    if not rows or not rows[0]:
        raise ValueError("joint must be a non-empty table")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("joint must be rectangular")
    flat = [value for row in rows for value in row]
    if any(value < 0.0 for value in flat):
        raise ValueError("joint entries must be non-negative")
    if abs(math.fsum(flat) - 1.0) > _SUM_TOLERANCE:
        raise ValueError("joint must sum to one")
    return rows


def marginals(joint: Iterable[Iterable[object]]) -> tuple[tuple[float, ...], ...]:
    """Return the marginal PMFs ``(p_X, p_Y)`` of a joint table.

    Args:
        joint: Joint PMF with rows indexing ``X``.

    Returns:
        Pair of row sums and column sums.

    Raises:
        ValueError: If the table is invalid.

    Equation:
        [EQ-INFO-H-002].

    References:
        [REF-COVER-THOMAS-2006, Section 2.2].

    """
    rows = _joint(joint)
    row_sums = tuple(math.fsum(row) for row in rows)
    column_sums = tuple(
        math.fsum(row[j] for row in rows) for j in range(len(rows[0]))
    )
    return row_sums, column_sums


def _plogp_sum(values: Iterable[float]) -> float:
    """Return ``-sum p log p`` with ``0 log 0 = 0``.

    Args:
        values: Non-negative probabilities.

    Returns:
        Entropy in nats.

    Equation:
        [EQ-INFO-H-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.1].

    """
    return -math.fsum(p * math.log(p) for p in values if p > 0.0)


def entropy(pmf: Iterable[object], *, unit: InformationUnit = "nats") -> float:
    """Return the Shannon entropy ``H(X) = -sum p log p``.

    Args:
        pmf: Finite PMF.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Entropy, between ``0`` and ``log |support|``.

    Raises:
        ValueError: If the PMF or unit is invalid.

    Equation:
        [EQ-INFO-H-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.1].

    """
    factor = unit_factor(unit)
    return max(_plogp_sum(_pmf(pmf, name="pmf")), 0.0) * factor


def joint_entropy(
    joint: Iterable[Iterable[object]], *, unit: InformationUnit = "nats"
) -> float:
    """Return ``H(X, Y) = -sum p(x, y) log p(x, y)``.

    Args:
        joint: Joint PMF table.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Joint entropy.

    Raises:
        ValueError: If the table or unit is invalid.

    Equation:
        [EQ-INFO-H-002].

    References:
        [REF-COVER-THOMAS-2006, Section 2.2].

    """
    factor = unit_factor(unit)
    rows = _joint(joint)
    return max(_plogp_sum(v for row in rows for v in row), 0.0) * factor


def conditional_entropy(
    joint: Iterable[Iterable[object]], *, unit: InformationUnit = "nats"
) -> float:
    """Return ``H(Y | X) = -sum p(x, y) log(p(x, y)/p(x))``.

    Rows index the conditioning variable ``X``. The chain rule
    ``H(X, Y) = H(X) + H(Y | X)`` holds to rounding.

    Args:
        joint: Joint PMF table with rows indexing ``X``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Conditional entropy.

    Raises:
        ValueError: If the table or unit is invalid.

    Equation:
        [EQ-INFO-H-002].

    References:
        [REF-COVER-THOMAS-2006, Section 2.2].

    """
    factor = unit_factor(unit)
    rows = _joint(joint)
    terms: list[float] = []
    for row in rows:
        row_mass = math.fsum(row)
        terms.extend(p * math.log(p / row_mass) for p in row if p > 0.0)
    return max(-math.fsum(terms), 0.0) * factor


def mutual_information(
    joint: Iterable[Iterable[object]], *, unit: InformationUnit = "nats"
) -> float:
    """Return ``I(X; Y) = sum p(x, y) log(p(x, y)/(p(x) p(y)))``.

    ``I = H(X) + H(Y) - H(X, Y) >= 0`` with equality iff ``X`` and ``Y`` are
    independent; rounding-level negative sums are returned as ``0``. When
    ``p(x) p(y)`` underflows, each term is evaluated as
    ``p ((log p - log p(x)) - log p(y))``.

    Args:
        joint: Joint PMF table.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Mutual information.

    Raises:
        ValueError: If the table or unit is invalid.

    Equation:
        [EQ-INFO-MI-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.4].

    """
    factor = unit_factor(unit)
    rows = _joint(joint)
    p_x, p_y = marginals(rows)
    terms: list[float] = []
    for i, row in enumerate(rows):
        for j, p in enumerate(row):
            if p <= 0.0:
                continue
            product = p_x[i] * p_y[j]
            if product >= _SMALLEST_NORMAL:
                terms.append(p * _log_ratio(p, product))
            else:
                terms.append(p * ((math.log(p) - math.log(p_x[i])) - math.log(p_y[j])))
    return max(math.fsum(terms), 0.0) * factor


def _pair(
    p: Iterable[object], q: Iterable[object]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate two PMFs on the same finite support.

    Args:
        p: First PMF.
        q: Second PMF.

    Returns:
        Validated pair.

    Raises:
        ValueError: If either PMF is invalid or the lengths differ.

    References:
        [REF-COVER-THOMAS-2006, Section 2.3]. No equation is evaluated.

    """
    first, second = _pmf(p, name="p"), _pmf(q, name="q")
    if len(first) != len(second):
        raise ValueError("p and q must have the same length")
    return first, second


def kl_divergence(
    p: Iterable[object], q: Iterable[object], *, unit: InformationUnit = "nats"
) -> float:
    """Return the Kullback-Leibler divergence ``D(p || q) = sum p log(p/q)``.

    A quotient ``p_i/q_i`` that would overflow (for example ``q_i`` close to
    the smallest subnormal) is replaced by ``log p_i - log q_i``, so the
    divergence stays finite whenever every ``q_i > 0`` where ``p_i > 0``.

    Args:
        p: PMF of the reference law.
        q: PMF of the approximating law on the same support.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Divergence ``>= 0``; ``inf`` when ``q_i = 0 < p_i`` for some ``i``.

    Raises:
        ValueError: If the PMFs or unit are invalid.

    Equation:
        [EQ-INFO-KL-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.3].

    """
    factor = unit_factor(unit)
    first, second = _pair(p, q)
    terms: list[float] = []
    for a, b in zip(first, second, strict=True):
        if a == 0.0:
            continue
        if b == 0.0:
            return math.inf
        terms.append(a * _log_ratio(a, b))
    return max(math.fsum(terms), 0.0) * factor


def cross_entropy(
    p: Iterable[object], q: Iterable[object], *, unit: InformationUnit = "nats"
) -> float:
    """Return the cross-entropy ``H(p, q) = -sum p log q = H(p) + D(p || q)``.

    Args:
        p: Reference PMF.
        q: Model PMF on the same support.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Cross-entropy (``inf`` when ``q_i = 0 < p_i``).

    Raises:
        ValueError: If the PMFs or unit are invalid.

    Equation:
        [EQ-INFO-KL-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.3].

    """
    factor = unit_factor(unit)
    first, second = _pair(p, q)
    terms: list[float] = []
    for a, b in zip(first, second, strict=True):
        if a == 0.0:
            continue
        if b == 0.0:
            return math.inf
        terms.append(-a * math.log(b))
    return math.fsum(terms) * factor


def jensen_shannon_divergence(
    p: Iterable[object], q: Iterable[object], *, unit: InformationUnit = "nats"
) -> float:
    """Return ``JS(p, q) = D(p || m)/2 + D(q || m)/2`` with ``m = (p + q)/2``.

    The divergence is symmetric, finite, and bounded by ``log 2`` (one bit);
    its square root is a metric. Where the midpoint ``(p_i + q_i)/2``
    underflows, ``log m_i = log(p_i + q_i) - log 2`` is used instead.

    Args:
        p: First PMF.
        q: Second PMF on the same support.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Jensen-Shannon divergence in ``[0, log 2]``.

    Raises:
        ValueError: If the PMFs or unit are invalid.

    Equation:
        [EQ-INFO-JS-001].

    References:
        [REF-LIN-1991], [REF-COVER-THOMAS-2006, Section 2.3].

    """
    factor = unit_factor(unit)
    first, second = _pair(p, q)

    def term(a: float, b: float) -> float:
        """Return ``a log(a/m)`` with ``m = (a + b)/2`` and ``0 log 0 = 0``.

        References:
            [EQ-INFO-KL-001], [REF-COVER-THOMAS-2006, Section 2.3].

        """
        if a <= 0.0:
            return 0.0
        middle = 0.5 * (a + b)
        if middle >= _SMALLEST_NORMAL:
            return a * _log_ratio(a, middle)
        return a * ((math.log(a) - math.log(a + b)) + _LOG_TWO)

    total = math.fsum(
        term(a, b) + term(b, a) for a, b in zip(first, second, strict=True)
    )
    value = 0.5 * total
    return min(max(value, 0.0), math.log(2.0)) * factor
