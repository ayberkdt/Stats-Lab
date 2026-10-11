"""Statistical distances and divergences beyond Kullback-Leibler.

For finite PMFs ``p`` and ``q`` on one support:

* total variation ``TV = sum |p - q| / 2``;
* Bhattacharyya coefficient ``BC = sum sqrt(p q)``, distance ``-log BC``;
* Hellinger distance ``H = sqrt(sum (sqrt p - sqrt q)^2 / 2) = sqrt(1 - BC)``
  in ``[0, 1]`` (some texts omit the ``1/2``, giving ``sqrt 2`` times this);
* Renyi divergence ``D_alpha = log(sum p^alpha q^(1 - alpha)) / (alpha - 1)``
  for ``alpha`` in ``[0, inf]``, with the KL divergence at ``alpha = 1``;
* the alpha-divergence ``D_A = (sum p^alpha q^(1 - alpha) - 1) /
  (alpha (alpha - 1))``, with ``KL(p || q)`` at ``alpha = 1`` and
  ``KL(q || p)`` at ``alpha = 0``.

For Gaussian laws all of these follow from the affinity
``A_alpha = int p^alpha q^(1 - alpha)``, which is available in closed form;
total variation has closed forms for univariate normals and for a shared
covariance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.foundations import require_finite_real
from probstat.linalg import as_vector, cholesky, cholesky_solve, solve_lower

from ._units import InformationUnit, unit_factor
from .differential import _log_det_from_lower, _spd, kl_multivariate_normal
from .discrete import _pmf, kl_divergence

_SQRT_TWO = math.sqrt(2.0)

# exp(x) overflows a double above about 709.78.
_LOG_MAX = 709.0


def _pair(
    p: Iterable[object], q: Iterable[object]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate two PMFs on one finite support.

    Args:
        p: First PMF.
        q: Second PMF.

    Returns:
        Both PMFs as tuples.

    Raises:
        ValueError: If a PMF is invalid or the lengths differ.

    References:
        [REF-COVER-THOMAS-2006, Section 2.1]. No equation is evaluated.

    """
    first = _pmf(p, name="p")
    second = _pmf(q, name="q")
    if len(first) != len(second):
        raise ValueError("p and q must have equal length")
    return first, second


def total_variation_distance(p: Iterable[object], q: Iterable[object]) -> float:
    """Return ``TV(p, q) = sum_i |p_i - q_i| / 2 = max_A |P(A) - Q(A)|``.

    Args:
        p: First PMF.
        q: Second PMF on the same support.

    Returns:
        Distance in ``[0, 1]``.

    Raises:
        ValueError: If the PMFs are invalid.

    Equation:
        [EQ-DIV-TV-001].

    References:
        [REF-TSYBAKOV-2009, Chapter 2].

    """
    first, second = _pair(p, q)
    return min(
        1.0, 0.5 * math.fsum(abs(a - b) for a, b in zip(first, second, strict=True))
    )


def bhattacharyya_coefficient(p: Iterable[object], q: Iterable[object]) -> float:
    """Return ``BC(p, q) = sum_i sqrt(p_i q_i)`` in ``[0, 1]``.

    Args:
        p: First PMF.
        q: Second PMF on the same support.

    Returns:
        The Bhattacharyya coefficient (affinity).

    Raises:
        ValueError: If the PMFs are invalid.

    Equation:
        [EQ-DIV-HELL-001].

    References:
        [REF-KAILATH-1967].

    """
    first, second = _pair(p, q)
    return min(
        1.0, math.fsum(math.sqrt(a * b) for a, b in zip(first, second, strict=True))
    )


def bhattacharyya_distance(p: Iterable[object], q: Iterable[object]) -> float:
    """Return ``D_B(p, q) = -log BC(p, q)`` (``inf`` for disjoint supports).

    Args:
        p: First PMF.
        q: Second PMF on the same support.

    Returns:
        Bhattacharyya distance in nats; it equals ``D_{1/2}/2`` (Renyi).

    Raises:
        ValueError: If the PMFs are invalid.

    Equation:
        [EQ-DIV-HELL-001].

    References:
        [REF-KAILATH-1967].

    """
    coefficient = bhattacharyya_coefficient(p, q)
    return math.inf if coefficient == 0.0 else max(0.0, -math.log(coefficient))


def hellinger_distance(p: Iterable[object], q: Iterable[object]) -> float:
    """Return ``H(p, q) = sqrt(sum_i (sqrt p_i - sqrt q_i)^2 / 2)`` in ``[0, 1]``.

    Args:
        p: First PMF.
        q: Second PMF on the same support.

    Returns:
        The Hellinger distance; ``H^2 = 1 - BC``.

    Raises:
        ValueError: If the PMFs are invalid.

    Equation:
        [EQ-DIV-HELL-001].

    References:
        [REF-TSYBAKOV-2009, Chapter 2].

    """
    first, second = _pair(p, q)
    squared = 0.5 * math.fsum(
        (math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(first, second, strict=True)
    )
    return math.sqrt(min(1.0, squared))


def _order(alpha: float, *, allow_infinity: bool) -> float:
    """Validate a divergence order.

    Args:
        alpha: Candidate order.
        allow_infinity: Whether ``+inf`` is accepted.

    Returns:
        The order as a float.

    Raises:
        TypeError: If the order is not real.
        ValueError: If the order is NaN or an unaccepted infinity.

    References:
        [REF-VAN-ERVEN-HARREMOES-2014]. No equation is evaluated.

    """
    if isinstance(alpha, bool) or not isinstance(alpha, int | float):
        raise TypeError("alpha must be real")
    value = float(alpha)
    if math.isnan(value):
        raise ValueError("alpha must not be NaN")
    if math.isinf(value) and not (allow_infinity and value > 0.0):
        raise ValueError("alpha must be finite")
    return value


def _log_power_sum(
    first: tuple[float, ...], second: tuple[float, ...], alpha: float
) -> float:
    """Return ``log sum_i p_i^alpha q_i^(1 - alpha)`` with limiting conventions.

    The sum is formed by log-sum-exp, so large orders whose individual terms
    overflow (``alpha = 200`` with ``q_i = 0.01``) still give a finite
    logarithm.

    Terms with ``p_i = q_i = 0`` vanish; a term with ``q_i = 0 < p_i`` is
    ``+inf`` when ``alpha > 1`` and zero when ``alpha < 1``; a term with
    ``p_i = 0 < q_i`` is ``+inf`` when ``alpha < 0`` and zero when
    ``alpha > 0``.

    Args:
        first: PMF ``p``.
        second: PMF ``q``.
        alpha: Finite order other than ``0`` and ``1``.

    Returns:
        ``log A_alpha``, possibly ``+inf`` (or ``-inf`` for an empty sum).

    Equation:
        [EQ-DIV-RENYI-001].

    References:
        [REF-VAN-ERVEN-HARREMOES-2014], [REF-CICHOCKI-CRUCES-AMARI-2011].

    """
    terms: list[float] = []
    for a, b in zip(first, second, strict=True):
        if a == 0.0 and b == 0.0:
            continue
        if b == 0.0:
            if alpha > 1.0:
                return math.inf
            continue
        if a == 0.0:
            if alpha < 0.0:
                return math.inf
            continue
        terms.append(alpha * math.log(a) + (1.0 - alpha) * math.log(b))
    if not terms:
        return -math.inf
    top = max(terms)
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


def renyi_divergence(
    p: Iterable[object],
    q: Iterable[object],
    alpha: float,
    *,
    unit: InformationUnit = "nats",
) -> float:
    """Return the Renyi divergence ``D_alpha(p || q)`` of order ``alpha``.

    ``D_alpha = log(sum p^alpha q^(1 - alpha)) / (alpha - 1)`` for
    ``alpha`` in ``(0, 1) U (1, inf)``; ``D_0 = -log Q(p > 0)``;
    ``D_1 = KL(p || q)``; ``D_inf = log max_i p_i/q_i``. It is
    non-decreasing in ``alpha`` and ``D_{1/2} = 2 D_B``.

    Args:
        p: First PMF.
        q: Second PMF on the same support.
        alpha: Order in ``[0, inf]``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Divergence ``>= 0``, possibly ``inf``.

    Raises:
        ValueError: If the PMFs, order, or unit are invalid.

    Equation:
        [EQ-DIV-RENYI-001].

    References:
        [REF-VAN-ERVEN-HARREMOES-2014].

    """
    factor = unit_factor(unit)
    order = _order(alpha, allow_infinity=True)
    if order < 0.0:
        raise ValueError("alpha must be non-negative")
    first, second = _pair(p, q)
    if order == 1.0:
        return kl_divergence(first, second, unit=unit)
    if order == 0.0:
        mass = math.fsum(b for a, b in zip(first, second, strict=True) if a > 0.0)
        return math.inf if mass == 0.0 else max(0.0, -math.log(mass)) * factor
    if math.isinf(order):
        ratios = []
        for a, b in zip(first, second, strict=True):
            if a > 0.0:
                if b == 0.0:
                    return math.inf
                ratios.append(a / b)
        return max(0.0, math.log(max(ratios))) * factor
    log_affinity = _log_power_sum(first, second, order)
    if math.isinf(log_affinity):
        return math.inf
    return max(0.0, log_affinity / (order - 1.0)) * factor


def alpha_divergence(p: Iterable[object], q: Iterable[object], alpha: float) -> float:
    """Return the alpha-divergence ``D_A^(alpha)(p || q)`` for PMFs.

    ``D_A = (sum p^alpha q^(1 - alpha) - 1) / (alpha (alpha - 1))`` for
    ``alpha`` not in ``{0, 1}``, ``KL(p || q)`` at ``alpha = 1`` and
    ``KL(q || p)`` at ``alpha = 0`` (all in nats). ``alpha = 1/2`` gives
    ``4 (1 - BC) = 4 H^2``. It relates to the Renyi divergence by
    ``D_alpha = log(1 + alpha (alpha - 1) D_A) / (alpha - 1)``.

    Args:
        p: First PMF.
        q: Second PMF on the same support.
        alpha: Any real order.

    Returns:
        Divergence ``>= 0``, possibly ``inf``.

    Raises:
        ValueError: If the PMFs or order are invalid.

    Equation:
        [EQ-DIV-ALPHA-001].

    References:
        [REF-CICHOCKI-CRUCES-AMARI-2011, Equation 25].

    """
    order = _order(alpha, allow_infinity=False)
    first, second = _pair(p, q)
    if order == 1.0:
        return kl_divergence(first, second)
    if order == 0.0:
        return kl_divergence(second, first)
    log_affinity = _log_power_sum(first, second, order)
    if log_affinity == math.inf or log_affinity > _LOG_MAX:
        return math.inf
    return max(0.0, math.expm1(log_affinity) / (order * (order - 1.0)))


def _gaussian_inputs(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
]:
    """Validate two Gaussian laws of one dimension.

    Args:
        mean0: First mean.
        covariance0: First symmetric positive-definite covariance.
        mean1: Second mean.
        covariance1: Second symmetric positive-definite covariance.

    Returns:
        Validated means and covariances.

    Raises:
        ValueError: If dimensions disagree or a covariance is invalid.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    m0 = as_vector(mean0, name="mean0")
    m1 = as_vector(mean1, name="mean1")
    s0 = _spd(covariance0, name="covariance0")
    s1 = _spd(covariance1, name="covariance1")
    if not (len(m0) == len(m1) == len(s0) == len(s1)):
        raise ValueError("means and covariances must share one dimension")
    return m0, s0, m1, s1


def _gaussian_log_affinity(
    m0: tuple[float, ...],
    s0: tuple[tuple[float, ...], ...],
    m1: tuple[float, ...],
    s1: tuple[tuple[float, ...], ...],
    alpha: float,
) -> float:
    """Return ``log int N0^alpha N1^(1 - alpha)`` or ``+inf`` if it diverges.

    With ``S_alpha = alpha S1 + (1 - alpha) S0`` and ``d = m0 - m1``:
    ``log A = -alpha (1 - alpha) d' S_alpha^-1 d / 2 - (log det S_alpha
    - (1 - alpha) log det S0 - alpha log det S1) / 2``. The integral is
    finite iff ``S_alpha`` is positive definite, which always holds for
    ``alpha`` in ``[0, 1]``.

    Args:
        m0: First mean.
        s0: First covariance.
        m1: Second mean.
        s1: Second covariance.
        alpha: Finite order.

    Returns:
        The log-affinity.

    Equation:
        [EQ-DIV-GAUSS-001].

    References:
        [REF-GIL-ALAJAJI-LINDER-2013], [REF-KAILATH-1967].

    """
    size = len(m0)
    mixed = tuple(
        tuple(alpha * s1[i][j] + (1.0 - alpha) * s0[i][j] for j in range(size))
        for i in range(size)
    )
    try:
        lower = cholesky(mixed)
    except ValueError:
        return math.inf
    difference = tuple(a - b for a, b in zip(m0, m1, strict=True))
    whitened = solve_lower(lower, difference)
    mahalanobis = math.fsum(value * value for value in whitened)
    log_det = (
        _log_det_from_lower(lower)
        - (1.0 - alpha) * _log_det_from_lower(cholesky(s0))
        - alpha * _log_det_from_lower(cholesky(s1))
    )
    return -0.5 * alpha * (1.0 - alpha) * mahalanobis - 0.5 * log_det


def renyi_divergence_gaussian(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
    alpha: float,
    *,
    unit: InformationUnit = "nats",
) -> float:
    """Return ``D_alpha(N(m0, S0) || N(m1, S1))`` in closed form.

    ``D_alpha = alpha d' S_alpha^-1 d / 2 - (log det S_alpha
    - (1 - alpha) log det S0 - alpha log det S1) / (2 (alpha - 1))`` with
    ``S_alpha = alpha S1 + (1 - alpha) S0`` and ``d = m0 - m1``; ``inf`` when
    ``alpha > 1`` and ``S_alpha`` is not positive definite; the KL divergence
    at ``alpha = 1``.

    Args:
        mean0: Mean of the first law.
        covariance0: Symmetric positive-definite covariance of the first law.
        mean1: Mean of the second law.
        covariance1: Symmetric positive-definite covariance of the second law.
        alpha: Positive finite order.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Divergence ``>= 0``, possibly ``inf``.

    Raises:
        ValueError: If inputs are invalid or ``alpha <= 0``.

    Equation:
        [EQ-DIV-GAUSS-001].

    References:
        [REF-GIL-ALAJAJI-LINDER-2013], [REF-VAN-ERVEN-HARREMOES-2014].

    """
    factor = unit_factor(unit)
    order = _order(alpha, allow_infinity=False)
    if not order > 0.0:
        raise ValueError("alpha must be positive")
    m0, s0, m1, s1 = _gaussian_inputs(mean0, covariance0, mean1, covariance1)
    if order == 1.0:
        return kl_multivariate_normal(m0, s0, m1, s1, unit=unit)
    log_affinity = _gaussian_log_affinity(m0, s0, m1, s1, order)
    if math.isinf(log_affinity):
        return math.inf
    return max(0.0, log_affinity / (order - 1.0)) * factor


def alpha_divergence_gaussian(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
    alpha: float,
) -> float:
    """Return the alpha-divergence between Gaussian laws in closed form.

    ``D_A = (A_alpha - 1) / (alpha (alpha - 1))`` with the Gaussian affinity
    ``A_alpha``; ``KL(N0 || N1)`` at ``alpha = 1`` and ``KL(N1 || N0)`` at
    ``alpha = 0`` (nats).

    Args:
        mean0: Mean of the first law.
        covariance0: Symmetric positive-definite covariance of the first law.
        mean1: Mean of the second law.
        covariance1: Symmetric positive-definite covariance of the second law.
        alpha: Any real order.

    Returns:
        Divergence ``>= 0``, possibly ``inf``.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-DIV-ALPHA-001], [EQ-DIV-GAUSS-001].

    References:
        [REF-CICHOCKI-CRUCES-AMARI-2011, Equation 25],
        [REF-GIL-ALAJAJI-LINDER-2013].

    """
    order = _order(alpha, allow_infinity=False)
    m0, s0, m1, s1 = _gaussian_inputs(mean0, covariance0, mean1, covariance1)
    if order == 1.0:
        return kl_multivariate_normal(m0, s0, m1, s1)
    if order == 0.0:
        return kl_multivariate_normal(m1, s1, m0, s0)
    log_affinity = _gaussian_log_affinity(m0, s0, m1, s1, order)
    if math.isinf(log_affinity) or log_affinity > _LOG_MAX:
        return math.inf
    return max(0.0, math.expm1(log_affinity) / (order * (order - 1.0)))


def bhattacharyya_distance_gaussian(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
) -> float:
    """Return the Bhattacharyya distance between Gaussian laws.

    ``D_B = d' S^-1 d / 8 + (log det S - (log det S0 + log det S1)/2) / 2``
    with ``S = (S0 + S1)/2`` and ``d = m0 - m1``.

    Args:
        mean0: Mean of the first law.
        covariance0: Symmetric positive-definite covariance of the first law.
        mean1: Mean of the second law.
        covariance1: Symmetric positive-definite covariance of the second law.

    Returns:
        Distance ``>= 0`` in nats.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-DIV-GAUSS-001].

    References:
        [REF-KAILATH-1967].

    """
    m0, s0, m1, s1 = _gaussian_inputs(mean0, covariance0, mean1, covariance1)
    return max(0.0, -_gaussian_log_affinity(m0, s0, m1, s1, 0.5))


def hellinger_distance_gaussian(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
) -> float:
    """Return ``H = sqrt(1 - exp(-D_B))`` between Gaussian laws.

    Args:
        mean0: Mean of the first law.
        covariance0: Symmetric positive-definite covariance of the first law.
        mean1: Mean of the second law.
        covariance1: Symmetric positive-definite covariance of the second law.

    Returns:
        Hellinger distance in ``[0, 1]`` (``H^2 = 1 - BC`` convention).

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-DIV-GAUSS-001].

    References:
        [REF-KAILATH-1967], [REF-TSYBAKOV-2009, Chapter 2].

    """
    distance = bhattacharyya_distance_gaussian(mean0, covariance0, mean1, covariance1)
    return math.sqrt(max(0.0, -math.expm1(-distance)))


def _phi(x: float) -> float:
    """Return the standard normal CDF ``erfc(-x/sqrt 2)/2``.

    Args:
        x: Abscissa.

    Returns:
        ``Phi(x)``.

    References:
        [REF-NIST-DLMF, Section 7.2]. No new equation is evaluated.

    """
    return 0.5 * math.erfc(-x / _SQRT_TWO)


def total_variation_normal(
    mu0: float, sigma0: float, mu1: float, sigma1: float
) -> float:
    """Return the exact total variation between two univariate normal laws.

    By Scheffe's identity ``TV = int_{p0 > p1} (p0 - p1)``. Equal standard
    deviations give ``erf(|mu0 - mu1| / (2 sqrt 2 sigma))``. Otherwise the
    densities cross at the two roots ``r1 < r2`` of
    ``log p0(x) = log p1(x)`` (a quadratic), and
    ``TV = |(F0(r2) - F0(r1)) - (F1(r2) - F1(r1))|``.

    Args:
        mu0: Mean of the first law.
        sigma0: Positive standard deviation of the first law.
        mu1: Mean of the second law.
        sigma1: Positive standard deviation of the second law.

    Returns:
        Distance in ``[0, 1]``.

    Raises:
        ValueError: If a standard deviation is not positive.

    Equation:
        [EQ-DIV-GAUSS-002].

    References:
        [REF-TSYBAKOV-2009, Chapter 2].

    """
    m0 = require_finite_real(mu0, name="mu0")
    m1 = require_finite_real(mu1, name="mu1")
    s0 = require_finite_real(sigma0, name="sigma0")
    s1 = require_finite_real(sigma1, name="sigma1")
    if not (s0 > 0.0 and s1 > 0.0):
        raise ValueError("standard deviations must be positive")
    if s0 == s1:
        return math.erf(abs(m0 - m1) / (2.0 * _SQRT_TWO * s0))
    a = 0.5 / (s1 * s1) - 0.5 / (s0 * s0)
    b = m0 / (s0 * s0) - m1 / (s1 * s1)
    c = 0.5 * m1 * m1 / (s1 * s1) - 0.5 * m0 * m0 / (s0 * s0) + math.log(s1 / s0)
    root = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    q = -0.5 * (b + math.copysign(root, b))
    first = q / a
    second = c / q if q != 0.0 else -first
    r1, r2 = sorted((first, second))
    mass0 = _phi((r2 - m0) / s0) - _phi((r1 - m0) / s0)
    mass1 = _phi((r2 - m1) / s1) - _phi((r1 - m1) / s1)
    return min(1.0, abs(mass0 - mass1))


def total_variation_gaussian_shared_covariance(
    mean0: Iterable[object],
    mean1: Iterable[object],
    covariance: Iterable[Iterable[object]],
) -> float:
    """Return ``TV = 2 Phi(Delta/2) - 1 = erf(Delta / (2 sqrt 2))``.

    ``Delta`` is the Mahalanobis distance of the means under the shared
    covariance; the log-likelihood ratio is linear, so the problem reduces
    to two unit-variance normals ``Delta`` apart.

    Args:
        mean0: First mean.
        mean1: Second mean.
        covariance: Shared symmetric positive-definite covariance.

    Returns:
        Distance in ``[0, 1]``.

    Raises:
        ValueError: If dimensions disagree or the covariance is invalid.

    Equation:
        [EQ-DIV-GAUSS-002].

    References:
        [REF-TSYBAKOV-2009, Chapter 2].

    """
    m0 = as_vector(mean0, name="mean0")
    m1 = as_vector(mean1, name="mean1")
    matrix = _spd(covariance, name="covariance")
    if not (len(m0) == len(m1) == len(matrix)):
        raise ValueError("means and covariance must share one dimension")
    difference = tuple(a - b for a, b in zip(m0, m1, strict=True))
    solved = cholesky_solve(cholesky(matrix), difference)
    delta = math.sqrt(
        max(0.0, math.fsum(x * y for x, y in zip(difference, solved, strict=True)))
    )
    return math.erf(delta / (2.0 * _SQRT_TWO))
