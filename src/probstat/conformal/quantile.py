"""Finite-sample conformal quantiles and the marginal coverage they guarantee.

Split conformal prediction ranks ``n`` calibration scores and uses the
``k``-th smallest, ``k = ceil((1 - alpha)(n + 1))``. This is the empirical
``(1 - alpha)(n + 1)/n`` quantile: the ``(n + 1)/n`` inflation accounts for the
unseen test point, which is exchangeable with the calibration points. When
``k > n`` the quantile is ``+inf`` and every interval is the whole line; that
is the price of asking for ``alpha < 1/(n + 1)``.

The index is computed in exact rational arithmetic from the decimal
representation of ``alpha`` (``Fraction(str(alpha))``), so ``alpha = 0.1``
and ``n = 9`` give exactly ``k = 9`` rather than a value perturbed by binary
rounding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction

from probstat.foundations.validation import require_finite_real


def _alpha_fraction(alpha: float) -> Fraction:
    """Validate ``alpha`` and return it as the rational its decimal text names.

    Args:
        alpha: Miscoverage level strictly inside ``(0, 1)``.

    Returns:
        ``Fraction(str(alpha))``.

    Raises:
        TypeError: If ``alpha`` is not real.
        ValueError: If ``alpha`` is not strictly inside ``(0, 1)``.

    References:
        [REF-ANGELOPOULOS-BATES-2023]. No equation is evaluated.

    """
    level = require_finite_real(alpha, name="alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    return Fraction(str(level))


def _check_size(size: int) -> int:
    """Validate a positive calibration-set size.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 1:
        raise ValueError("size must be positive")
    return size


def conformal_rank(size: int, alpha: float) -> int:
    """Return the conformal order-statistic index ``k = ceil((1-alpha)(n+1))``.

    Args:
        size: Number ``n`` of calibration scores.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        ``k`` in ``1..n+1``; ``k = n + 1`` means the quantile is infinite.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``size < 1`` or ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-001].

    References:
        [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
        [REF-ANGELOPOULOS-BATES-2023].

    """
    n = _check_size(size)
    level = _alpha_fraction(alpha)
    rank = math.ceil((n + 1) * (1 - level))
    return max(1, min(rank, n + 1))


def lower_conformal_rank(size: int, alpha: float) -> int:
    """Return ``floor(alpha (n + 1))``, the lower jackknife+ order index.

    Args:
        size: Number ``n`` of leave-one-out residuals.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Index in ``0..n``; ``0`` means the lower limit is ``-inf``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``size < 1`` or ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-004].

    References:
        [REF-BARBER-ETAL-2021].

    """
    n = _check_size(size)
    level = _alpha_fraction(alpha)
    return max(0, min(math.floor(level * (n + 1)), n))


def conformal_quantile(scores: Iterable[float], alpha: float) -> float:
    """Return the finite-sample conformal quantile of calibration scores.

    ``q_hat = s_(k)``, the ``k``-th smallest score with
    ``k = ceil((1 - alpha)(n + 1))``, or ``+inf`` when ``k = n + 1``.

    Args:
        scores: Finite calibration scores (at least one).
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Conformal quantile, possibly ``+inf``.

    Raises:
        TypeError: If a score is not real.
        ValueError: If there is no score, a score is not finite, or ``alpha``
            is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-001].

    References:
        [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
        [REF-ANGELOPOULOS-BATES-2023].

    """
    ordered = sorted(require_finite_real(value, name="score") for value in scores)
    if not ordered:
        raise ValueError("at least one calibration score is required")
    rank = conformal_rank(len(ordered), alpha)
    return math.inf if rank > len(ordered) else ordered[rank - 1]


def split_conformal_coverage(size: int, alpha: float) -> float:
    """Return the marginal coverage ``k / (n + 1)`` of split conformal sets.

    For exchangeable calibration and test points, ``P(Y in C(X)) >= k/(n+1)
    >= 1 - alpha``, with equality ``k/(n+1)`` when scores are almost surely
    distinct; then also ``k/(n+1) < 1 - alpha + 1/(n+1)``. The probability is
    over both the calibration set and the test point.

    Args:
        size: Number ``n`` of calibration scores.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        ``min(k, n + 1) / (n + 1)`` (``1`` when the quantile is infinite).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``size < 1`` or ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-005].

    References:
        [REF-LEI-ETAL-2018], [REF-ANGELOPOULOS-BATES-2023].

    """
    n = _check_size(size)
    return conformal_rank(n, alpha) / (n + 1)
