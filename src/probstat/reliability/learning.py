"""Learning functions and stopping rules of active-learning reliability methods.

A surrogate with a Gaussian predictive law ``G(x) ~ N(mu(x), sigma^2(x))``
rates every candidate point by a learning function; the next true-model call
goes to the best-rated candidate, and learning stops when the best rating
says that no candidate is worth a call. Three functions are provided:

- ``"U"`` (Echard et al. 2011): ``U = |mu| / sigma``, smallest is best;
  stop when ``min U >= 2``.
- ``"EFF"`` (Bichon et al. 2008): the expected feasibility function with
  ``epsilon = 2 sigma``, largest is best; stop when ``max EFF <= 0.001``.
- ``"H"`` (Lv, Lu, and Wang 2015): the information entropy of the predictive
  law on the band ``|G| <= 2 sigma``, largest is best; stop when
  ``max H <= 0.5``.

The functions are independent of how ``mu`` and ``sigma`` were obtained, so
AK-MCS, a subset-simulation variant, or any other candidate pool can share
them. A second stopping rule bounds the failure probability itself
(:func:`probability_bounds`, :func:`bounds_satisfied`).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType

from probstat.foundations.validation import require_finite_real
from probstat.modeling import standard_normal_cdf

__all__ = [
    "BOUNDS_QUANTILE",
    "LEARNING_FUNCTIONS",
    "LEARNING_THRESHOLDS",
    "STOPPING_RULES",
    "bounds_satisfied",
    "learning_value",
    "probability_bounds",
    "select_learning_point",
    "stopping_satisfied",
]

LEARNING_FUNCTIONS: tuple[str, ...] = ("U", "EFF", "H")
"""Names accepted by :func:`learning_value`."""

LEARNING_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {"U": 2.0, "EFF": 1e-3, "H": 0.5}
)
"""Published stopping value of each learning function (``min U >= 2``,
``max EFF <= 0.001``, ``max H <= 0.5``)."""

STOPPING_RULES: tuple[str, ...] = ("learning", "bounds")
"""Stopping rules: the learning-function threshold or the ``P_f`` bounds."""

BOUNDS_QUANTILE = 1.96
"""Standard-normal quantile of the ``mu -/+ k sigma`` bounds (``k = 1.96``)."""

_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _require_learning(name: object) -> str:
    """Validate a learning-function name.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not isinstance(name, str) or name not in LEARNING_FUNCTIONS:
        raise ValueError(f"learning must be one of {LEARNING_FUNCTIONS}")
    return name


def _density(z: float) -> float:
    """Return the standard normal density ``phi(z)``.

    References:
        [EQ-DIST-NORM-001], [REF-NIST-DLMF, Equation 7.2.2].

    """
    return math.exp(-0.5 * z * z) / _SQRT_TWO_PI


def _expected_feasibility(mean: float, std: float) -> float:
    """Return Bichon's ``EFF`` with ``epsilon = 2 sigma`` (``0`` when ``sigma = 0``).

    References:
        [EQ-REL-AL-EFF-001], [REF-BICHON-ETAL-2008].

    """
    if std <= 0.0:
        return 0.0
    epsilon = 2.0 * std
    t = -mean / std
    low = (-epsilon - mean) / std
    high = (epsilon - mean) / std
    cdf_t, cdf_low, cdf_high = (
        standard_normal_cdf(t),
        standard_normal_cdf(low),
        standard_normal_cdf(high),
    )
    return (
        mean * (2.0 * cdf_t - cdf_low - cdf_high)
        - std * (2.0 * _density(t) - _density(low) - _density(high))
        + epsilon * (cdf_high - cdf_low)
    )


def _entropy(mean: float, std: float) -> float:
    """Return Lv, Lu, and Wang's ``H`` (``0`` when ``sigma = 0``).

    References:
        [EQ-REL-AL-H-001], [REF-LV-LU-WANG-2015].

    """
    if std <= 0.0:
        return 0.0
    ratio = mean / std
    upper = 2.0 - ratio
    lower = -2.0 - ratio
    mass = standard_normal_cdf(upper) - standard_normal_cdf(lower)
    tails = 0.5 * (upper * _density(upper) - lower * _density(lower))
    return abs((math.log(_SQRT_TWO_PI * std) + 0.5) * mass - tails)


def learning_value(name: str, mean: float, std: float) -> float:
    """Return a learning function at one candidate with ``G ~ N(mean, std^2)``.

    ``U = |mu| / sigma`` [EQ-REL-AKMCS-001] (``0`` for ``sigma = mu = 0`` and
    ``inf`` for ``sigma = 0 != mu``); ``EFF`` [EQ-REL-AL-EFF-001] with
    ``epsilon = 2 sigma`` and ``H`` [EQ-REL-AL-H-001] are ``0`` when
    ``sigma = 0``. ``EFF`` is in the units of ``G`` and ``H`` contains
    ``ln sigma``, so their thresholds depend on how ``G`` is scaled; ``U`` is
    scale-free.

    Args:
        name: ``"U"``, ``"EFF"``, or ``"H"``.
        mean: Predictive mean ``mu`` (finite).
        std: Predictive standard deviation ``sigma >= 0``.

    Returns:
        Learning value; for ``U`` smaller is more informative, for ``EFF``
        and ``H`` larger is.

    Raises:
        ValueError: If ``name`` is unknown, ``mean`` is not finite, or ``std``
            is negative or not finite.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-BICHON-ETAL-2008],
        [REF-LV-LU-WANG-2015].

    """
    function = _require_learning(name)
    mu = require_finite_real(mean, name="mean")
    sigma = require_finite_real(std, name="std")
    if sigma < 0.0:
        raise ValueError("std must be non-negative")
    if function == "U":
        if sigma == 0.0:
            return 0.0 if mu == 0.0 else math.inf
        return abs(mu) / sigma
    if function == "EFF":
        return _expected_feasibility(mu, sigma)
    return _entropy(mu, sigma)


def _value_from_variance(function: str, mean: float, variance: float) -> float:
    """Return a learning value from a variance without input validation.

    For ``U`` this is exactly ``|mu| / sqrt(var)`` (the arithmetic of AK-MCS
    before the learning functions were generalized), with ``var <= 0``
    treated as ``sigma = 0``.

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001].

    """
    if function == "U":
        if variance <= 0.0:
            return 0.0 if mean == 0.0 else math.inf
        return abs(mean) / math.sqrt(variance)
    std = math.sqrt(variance) if variance > 0.0 else 0.0
    if function == "EFF":
        return _expected_feasibility(mean, std)
    return _entropy(mean, std)


def select_learning_point(
    name: str,
    means: Sequence[float],
    variances: Sequence[float],
    excluded: Collection[int] = (),
) -> tuple[int, float]:
    """Return the most informative candidate and its learning value.

    Candidates are scanned in index order and ties keep the first index:
    ``argmin U`` or ``argmax EFF`` / ``argmax H`` over the indices not in
    ``excluded`` (typically candidates already added to the training set).

    Args:
        name: ``"U"``, ``"EFF"``, or ``"H"``.
        means: Predictive means of the candidates.
        variances: Predictive variances of the candidates (same length).
        excluded: Indices that may not be selected.

    Returns:
        ``(index, value)``; ``index`` is ``-1`` when no candidate is left,
        with ``value = inf`` for ``U`` and ``0`` for ``EFF`` and ``H``.

    Raises:
        ValueError: If ``name`` is unknown or the lengths differ.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-BICHON-ETAL-2008],
        [REF-LV-LU-WANG-2015].

    """
    function = _require_learning(name)
    if len(means) != len(variances):
        raise ValueError("means and variances must have the same length")
    best_index = -1
    if function == "U":
        # Inlined _value_from_variance("U", ...): same arithmetic, no call.
        best = math.inf
        sqrt = math.sqrt
        for index, (mean, variance) in enumerate(zip(means, variances, strict=True)):
            if index in excluded:
                continue
            if variance <= 0.0:
                value = 0.0 if mean == 0.0 else math.inf
            else:
                value = abs(mean) / sqrt(variance)
            if value < best:
                best, best_index = value, index
        return best_index, best
    best = -math.inf
    for index, (mean, variance) in enumerate(zip(means, variances, strict=True)):
        if index in excluded:
            continue
        value = _value_from_variance(function, mean, variance)
        if value > best:
            best, best_index = value, index
    return best_index, (best if best_index >= 0 else 0.0)


def stopping_satisfied(name: str, value: float, threshold: float | None = None) -> bool:
    """Return whether the best learning value allows learning to stop.

    ``min U >= threshold`` for ``U``; ``max EFF <= threshold`` and
    ``max H <= threshold`` for the others. ``threshold=None`` uses
    :data:`LEARNING_THRESHOLDS`.

    Args:
        name: ``"U"``, ``"EFF"``, or ``"H"``.
        value: Best learning value over the candidates (``min U`` or
            ``max EFF`` / ``max H``).
        threshold: Positive stopping value, or ``None`` for the default.

    Returns:
        ``True`` when the stopping condition holds.

    Raises:
        ValueError: If ``name`` is unknown or ``threshold`` is not positive.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-BICHON-ETAL-2008],
        [REF-LV-LU-WANG-2015], [REF-DU-XU-2023].

    """
    function = _require_learning(name)
    if threshold is None:
        limit = LEARNING_THRESHOLDS[function]
    else:
        limit = require_finite_real(threshold, name="threshold")
        if limit <= 0.0:
            raise ValueError("threshold must be positive")
    if function == "U":
        return value >= limit
    return value <= limit


def probability_bounds(
    means: Sequence[float],
    variances: Sequence[float],
    quantile: float = BOUNDS_QUANTILE,
) -> tuple[float, float, float]:
    """Return ``(P_f^-, P_f^0, P_f^+)`` of a candidate population.

    ``P_f^0 = #{mu_i <= 0} / N``, ``P_f^- = #{mu_i + k sigma_i <= 0} / N``
    and ``P_f^+ = #{mu_i - k sigma_i <= 0} / N`` with ``k = quantile``
    [EQ-REL-AL-BOUNDS-001]: the points whose sign is uncertain at level
    ``k`` are counted as safe for the lower and as failed for the upper
    bound.

    Args:
        means: Predictive means of the population.
        variances: Predictive variances (same length; negatives count as 0).
        quantile: Positive quantile ``k`` (default 1.96).

    Returns:
        Lower bound, estimate, and upper bound, with
        ``P_f^- <= P_f^0 <= P_f^+``.

    Raises:
        ValueError: If the population is empty, the lengths differ, or
            ``quantile`` is not positive.

    Equation:
        [EQ-REL-AL-BOUNDS-001].

    References:
        [REF-SCHOBI-SUDRET-MARELLI-2017].

    """
    k = require_finite_real(quantile, name="quantile")
    if k <= 0.0:
        raise ValueError("quantile must be positive")
    if len(means) != len(variances):
        raise ValueError("means and variances must have the same length")
    if not means:
        raise ValueError("the population must not be empty")
    lower = estimate = upper = 0
    for mean, variance in zip(means, variances, strict=True):
        spread = k * math.sqrt(variance) if variance > 0.0 else 0.0
        if mean + spread <= 0.0:
            lower += 1
        if mean <= 0.0:
            estimate += 1
        if mean - spread <= 0.0:
            upper += 1
    count = len(means)
    return lower / count, estimate / count, upper / count


def bounds_satisfied(
    lower: float, estimate: float, upper: float, tolerance: float
) -> bool:
    """Return whether ``(P_f^+ - P_f^-) / P_f^0 <= tolerance``.

    With ``P_f^0 = 0`` the rule holds only when ``P_f^+ = 0`` as well (no
    point can be failed at the chosen level).

    Args:
        lower: ``P_f^-``.
        estimate: ``P_f^0``.
        upper: ``P_f^+``.
        tolerance: Positive relative width.

    Returns:
        ``True`` when the relative width of the bounds is within tolerance.

    Raises:
        ValueError: If ``tolerance`` is not positive or the bounds are not
            ordered.

    Equation:
        [EQ-REL-AL-BOUNDS-001].

    References:
        [REF-SCHOBI-SUDRET-MARELLI-2017].

    """
    width = require_finite_real(tolerance, name="tolerance")
    if width <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 0.0 <= lower <= estimate <= upper <= 1.0:
        raise ValueError("bounds must satisfy 0 <= lower <= estimate <= upper <= 1")
    if estimate == 0.0:
        return upper == 0.0
    return upper - lower <= width * estimate
