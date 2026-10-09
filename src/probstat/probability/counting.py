"""Exact integer counting operations with explicit mathematical provenance."""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.foundations.validation import require_non_negative_integer


def factorial(n: int) -> int:
    """Return the number of orderings of ``n`` distinct objects.

    Args:
        n: Number of distinct objects, with ``n >= 0``.

    Returns:
        ``n!`` as an exact integer. By convention, ``0! == 1``.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative.

    Equation:
        [EQ-COMB-001].

    References:
        [REF-ROSS-2019, Section 1.2].

    """
    return math.factorial(require_non_negative_integer(n, name="n"))


def permutations(n: int, r: int) -> int:
    """Return ordered selections of ``r`` objects from ``n`` distinct objects.

    Args:
        n: Population size, with ``n >= 0``.
        r: Selection size, with ``0 <= r <= n``.

    Returns:
        The exact permutation count ``P(n, r)``.

    Raises:
        TypeError: If either argument is not an integer.
        ValueError: If either argument is negative or ``r > n``.

    Equation:
        [EQ-COMB-002].

    References:
        [REF-ROSS-2019, Section 1.4].

    """
    checked_n = require_non_negative_integer(n, name="n")
    checked_r = require_non_negative_integer(r, name="r")
    if checked_r > checked_n:
        raise ValueError("r must not exceed n")
    return math.perm(checked_n, checked_r)


def combinations(n: int, r: int) -> int:
    """Return unordered selections of ``r`` objects from ``n`` distinct objects.

    Args:
        n: Population size, with ``n >= 0``.
        r: Selection size, with ``0 <= r <= n``.

    Returns:
        The exact binomial coefficient ``C(n, r)``.

    Raises:
        TypeError: If either argument is not an integer.
        ValueError: If either argument is negative or ``r > n``.

    Equation:
        [EQ-COMB-003].

    References:
        [REF-ROSS-2019, Section 1.5].

    """
    checked_n = require_non_negative_integer(n, name="n")
    checked_r = require_non_negative_integer(r, name="r")
    if checked_r > checked_n:
        raise ValueError("r must not exceed n")
    return math.comb(checked_n, checked_r)


def multiplication_rule(stage_counts: Iterable[int]) -> int:
    """Return the count for a sequence of finite independent choices.

    "Independent" here means that every choice at a stage exposes the stated
    number of choices at the next stage; it is a counting condition, not a
    probabilistic-independence assertion.

    Args:
        stage_counts: Non-empty iterable of non-negative choice counts.

    Returns:
        The exact product of the stage counts.

    Raises:
        TypeError: If a stage count is not an integer.
        ValueError: If the iterable is empty or a count is negative.

    Equation:
        [EQ-COMB-004].

    References:
        [REF-ROSS-2019, Section 1.1].

    """
    counts = tuple(
        require_non_negative_integer(value, name=f"stage_counts[{index}]")
        for index, value in enumerate(stage_counts)
    )
    if not counts:
        raise ValueError("stage_counts must contain at least one stage")
    return math.prod(counts)

