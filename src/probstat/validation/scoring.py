"""Proper scoring rules for probabilistic forecasts.

A scoring rule ``S(F, y)`` assigns a penalty (here: smaller is better) to a
forecast distribution ``F`` once the outcome ``y`` is known. It is *proper*
if the expected score under ``G`` is minimized by forecasting ``F = G``, so a
forecaster cannot gain by misreporting. The module provides:

* CRPS for ensembles (exact ``O(m log m)`` sorted formula and the fair,
  ensemble-size-unbiased version), closed forms for Normal, log-normal,
  logistic, exponential, and uniform laws, and a quadrature fallback for any
  law with ``cdf`` and ``quantile``;
* the logarithmic score and the Dawid-Sebastiani score;
* the energy and variogram scores for multivariate ensembles;
* the interval score, weighted interval score, and pinball (quantile) loss
  for quantile forecasts;
* the Brier score with the Murphy decomposition and the two within-bin terms
  that make it exact for any binning.

Every score is negatively oriented. Averages over many forecast cases are
left to the caller, because the cases usually carry their own metadata.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

from probstat.distributions import Exponential, Logistic, LogNormal, Normal, Uniform
from probstat.foundations import (
    CumulativeDistribution,
    LogProbabilityDensity,
    LogProbabilityMassFunction,
    ProbabilityDensity,
    ProbabilityMassFunction,
    QuantileFunction,
    coerce_finite_sample,
    require_finite_real,
    require_probability,
)
from probstat.numerics.double_exponential import tanh_sinh, warn_if_unreliable

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)

CrpsMethod = Literal["auto", "closed-form", "quadrature"]


def _phi(z: float) -> float:
    """Return the standard normal density.

    References:
        [REF-GNEITING-RAFTERY-2007].

    """
    return _INV_SQRT_TWO_PI * math.exp(-0.5 * z * z)


def _big_phi(z: float) -> float:
    """Return the standard normal CDF via ``erfc`` (accurate in the left tail).

    References:
        [REF-GNEITING-RAFTERY-2007].

    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _positive(value: object, *, name: str) -> float:
    """Validate a finite, strictly positive real parameter.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def crps_ensemble(
    members: Sequence[float], observation: float, *, fair: bool = False
) -> float:
    """Return the CRPS of an ensemble (empirical) forecast.

    ``CRPS(F_m, y) = int (F_m(x) - 1{x >= y})^2 dx
    = (1/m) sum |x_i - y| - (1/(2 m^2)) sum_i sum_j |x_i - x_j|``.
    The double sum is evaluated in ``O(m log m)`` from the order statistics,
    ``sum_i sum_j |x_i - x_j| = 2 sum_k (2k - m - 1) x_(k)``, after shifting
    every member by ``y`` (the score is shift invariant), so a large common
    offset costs no precision.

    The *fair* version divides the spread term by ``2 m (m - 1)`` instead.
    It is the CRPS one expects for the distribution the ensemble members
    were drawn from: its expectation over exchangeable members does not
    depend on ``m``, whereas the ordinary ensemble CRPS is too large by
    ``E|X - X'| / (2m)`` on average.

    Args:
        members: Non-empty finite ensemble (at least two members if fair).
        observation: Finite verifying value ``y``.
        fair: Use the fair (unbiased spread) estimator.

    Returns:
        Non-negative score in the unit of ``y`` (smaller is better). The
        fair score can be slightly negative for an ensemble that is very
        close to ``y``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If inputs are empty or non-finite, or ``fair`` is
            requested with fewer than two members.

    Equation:
        [EQ-VAL-SCORE-CRPS-001], [EQ-VAL-SCORE-CRPS-002],
        [EQ-VAL-SCORE-CRPS-003].

    References:
        [REF-MATHESON-WINKLER-1976], [REF-HERSBACH-2000],
        [REF-GNEITING-RAFTERY-2007], [REF-FERRO-2014].

    """
    y = require_finite_real(observation, name="observation")
    shifted = sorted(
        value - y for value in coerce_finite_sample(members, name="members")
    )
    m = len(shifted)
    if fair and m < 2:
        raise ValueError("the fair CRPS needs at least two members")
    accuracy = math.fsum(abs(value) for value in shifted) / m
    spread = math.fsum((2 * k - m - 1) * value for k, value in enumerate(shifted, 1))
    denominator = m * (m - 1) if fair else m * m
    return accuracy - spread / denominator


def crps_normal(observation: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the closed-form CRPS of ``Normal(mu, sigma)``.

    ``CRPS = sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mu) / sigma``.

    Args:
        observation: Finite verifying value.
        mu: Finite mean.
        sigma: Positive standard deviation.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``sigma <= 0``.

    Equation:
        [EQ-VAL-SCORE-CRPS-004].

    References:
        [REF-GNEITING-RAFTERY-WESTVELD-GOLDMAN-2005],
        [REF-GNEITING-RAFTERY-2007].

    """
    y = require_finite_real(observation, name="observation")
    location = require_finite_real(mu, name="mu")
    scale = _positive(sigma, name="sigma")
    z = (y - location) / scale
    return scale * (z * (2.0 * _big_phi(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRT_PI)


def crps_lognormal(observation: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the closed-form CRPS of ``LogNormal(mu, sigma)``.

    For ``y > 0`` with ``z = (ln y - mu) / sigma``,
    ``CRPS = y (2 Phi(z) - 1) - 2 exp(mu + sigma^2/2)
    [Phi(z - sigma) + Phi(sigma / sqrt 2) - 1]``. For ``y <= 0`` the same
    expression holds with ``Phi(z) = Phi(z - sigma) = 0``, giving
    ``-y + 2 exp(mu + sigma^2/2) [1 - Phi(sigma / sqrt 2)]``.

    Args:
        observation: Finite verifying value (any sign).
        mu: Finite log-scale location.
        sigma: Positive log-scale standard deviation.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``sigma <= 0``.

    Equation:
        [EQ-VAL-SCORE-CRPS-005].

    References:
        [REF-BARAN-LERCH-2015], [REF-JORDAN-KRUGER-LERCH-2019].

    """
    y = require_finite_real(observation, name="observation")
    location = require_finite_real(mu, name="mu")
    scale = _positive(sigma, name="sigma")
    mean = math.exp(location + 0.5 * scale * scale)
    half = _big_phi(scale / math.sqrt(2.0))
    if y <= 0.0:
        return -y + 2.0 * mean * (1.0 - half)
    z = (math.log(y) - location) / scale
    return y * (2.0 * _big_phi(z) - 1.0) - 2.0 * mean * (
        _big_phi(z - scale) + half - 1.0
    )


def crps_logistic(observation: float, mu: float = 0.0, s: float = 1.0) -> float:
    """Return the closed-form CRPS of ``Logistic(mu, s)``.

    ``CRPS = s [z - 2 ln F(z) - 1]`` with ``z = (y - mu) / s`` and the
    standard logistic CDF ``F``; ``ln F(z) = -log1p(exp(-z))`` is evaluated
    stably on both tails.

    Args:
        observation: Finite verifying value.
        mu: Finite location.
        s: Positive scale.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``s <= 0``.

    Equation:
        [EQ-VAL-SCORE-CRPS-006].

    References:
        [REF-JORDAN-KRUGER-LERCH-2019].

    """
    y = require_finite_real(observation, name="observation")
    location = require_finite_real(mu, name="mu")
    scale = _positive(s, name="s")
    z = (y - location) / scale
    log_cdf = -math.log1p(math.exp(-z)) if z >= 0.0 else z - math.log1p(math.exp(z))
    return scale * (z - 2.0 * log_cdf - 1.0)


def crps_exponential(observation: float, rate: float = 1.0) -> float:
    """Return the closed-form CRPS of ``Exponential(rate)``.

    With ``lambda = rate``: ``CRPS = y + 2 exp(-lambda y)/lambda
    - 3/(2 lambda)`` for ``y >= 0`` and ``|y| + 1/(2 lambda)`` for ``y < 0``
    (from ``E|X - y| - E|X - X'|/2`` with ``E|X - X'| = 1/lambda``).

    Args:
        observation: Finite verifying value (any sign).
        rate: Positive rate.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``rate <= 0``.

    Equation:
        [EQ-VAL-SCORE-CRPS-007].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-JORDAN-KRUGER-LERCH-2019].

    """
    y = require_finite_real(observation, name="observation")
    lam = _positive(rate, name="rate")
    if y < 0.0:
        return -y + 0.5 / lam
    return y + 2.0 * math.exp(-lam * y) / lam - 1.5 / lam


def crps_uniform(observation: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return the closed-form CRPS of ``Uniform(lower, upper)``.

    With ``w = upper - lower``: ``E|X - X'| = w/3``, and
    ``E|X - y| = ((y - lower)^2 + (upper - y)^2) / (2w)`` inside the support
    and ``|y - (lower + upper)/2|`` outside, so
    ``CRPS = E|X - y| - w/6``.

    Args:
        observation: Finite verifying value.
        lower: Finite lower bound.
        upper: Finite upper bound greater than ``lower``.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``upper <= lower``.

    Equation:
        [EQ-VAL-SCORE-CRPS-008].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-JORDAN-KRUGER-LERCH-2019].

    """
    y = require_finite_real(observation, name="observation")
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    if b <= a:
        raise ValueError("upper must exceed lower")
    width = b - a
    if a <= y <= b:
        distance = ((y - a) ** 2 + (b - y) ** 2) / (2.0 * width)
    else:
        distance = abs(y - 0.5 * (a + b))
    return distance - width / 6.0


def crps_quadrature(
    distribution: CumulativeDistribution,
    observation: float,
    *,
    tolerance: float = 1.0e-11,
) -> float:
    """Return the CRPS of a continuous law by quantile-scale quadrature.

    Uses ``CRPS(F, y) = 2 int_0^1 QS_u(F^-1(u), y) du`` with the pinball
    loss ``QS_u``, split at ``u_0 = F(y)``:
    ``2 [int_0^u0 u (y - Q(u)) du + int_u0^1 (1 - u)(Q(u) - y) du]``. Both
    pieces are smooth inside and integrable at the ends, which the tanh-sinh
    rule handles; upper-tail quantiles use ``isf`` when the law has one.
    Nodes closer than ``1e-15`` to ``0`` or ``1`` are omitted, so laws whose
    quantile function grows faster than any power near the ends are not
    supported (the CRPS of a law without a finite mean is infinite anyway).

    Args:
        distribution: Continuous law exposing ``cdf`` and ``quantile``.
        observation: Finite verifying value.
        tolerance: Relative stopping tolerance of the quadrature.

    Returns:
        Non-negative score, accurate to roughly ``tolerance``.

    Raises:
        TypeError: If the law lacks ``cdf`` or ``quantile``.
        ValueError: If the observation or a returned value is non-finite.

    Equation:
        [EQ-VAL-SCORE-CRPS-009], [EQ-VAL-QUAD-DE-001].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-TAKAHASI-MORI-1974].

    """
    if not isinstance(distribution, CumulativeDistribution) or not isinstance(
        distribution, QuantileFunction
    ):
        raise TypeError("distribution must expose cdf(x) and quantile(p)")
    y = require_finite_real(observation, name="observation")
    law = cast(QuantileFunction, distribution)
    isf = getattr(distribution, "isf", None)

    def quantile(u: float, complement: float) -> float:
        """Return ``Q(u)`` using ``isf(1 - u)`` in the upper half when possible.

        References:
            [EQ-VAL-SCORE-CRPS-009].

        """
        if u > 0.5 and callable(isf):
            return float(isf(complement))
        return float(law.quantile(u))

    split = require_probability(distribution.cdf(y), name="distribution.cdf(y)")
    total = 0.0
    if split > 0.0:
        total += warn_if_unreliable(
            tanh_sinh(
                lambda u, uc: u * (y - quantile(u, uc)), 0.0, split, tolerance=tolerance
            ),
            "CRPS lower part",
        )
    if split < 1.0:
        total += warn_if_unreliable(
            tanh_sinh(
                lambda u, uc: uc * (quantile(u, uc) - y),
                split,
                1.0,
                tolerance=tolerance,
            ),
            "CRPS upper part",
        )
    return max(0.0, 2.0 * total)


def crps_distribution(
    distribution: CumulativeDistribution,
    observation: float,
    *,
    method: CrpsMethod = "auto",
) -> float:
    """Return the CRPS of a distribution object, closed form when available.

    ``method="auto"`` uses the closed form for :class:`Normal`,
    :class:`LogNormal`, :class:`Logistic`, :class:`Exponential`, and
    :class:`Uniform`, and :func:`crps_quadrature` otherwise.
    ``"closed-form"`` raises for other families; ``"quadrature"`` always
    integrates (useful as an independent check).

    Args:
        distribution: Forecast law.
        observation: Finite verifying value.
        method: ``"auto"``, ``"closed-form"``, or ``"quadrature"``.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If the law lacks the needed methods.
        ValueError: If ``method`` is unknown, or ``"closed-form"`` is
            requested for a family without one.

    Equation:
        [EQ-VAL-SCORE-CRPS-004] to [EQ-VAL-SCORE-CRPS-009].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-JORDAN-KRUGER-LERCH-2019].

    """
    if method not in ("auto", "closed-form", "quadrature"):
        raise ValueError("method must be 'auto', 'closed-form', or 'quadrature'")
    if method != "quadrature":
        if isinstance(distribution, Normal):
            return crps_normal(observation, distribution.mu, distribution.sigma)
        if isinstance(distribution, LogNormal):
            return crps_lognormal(observation, distribution.mu, distribution.sigma)
        if isinstance(distribution, Logistic):
            return crps_logistic(observation, distribution.mu, distribution.s)
        if isinstance(distribution, Exponential):
            return crps_exponential(observation, distribution.rate)
        if isinstance(distribution, Uniform):
            return crps_uniform(observation, distribution.lower, distribution.upper)
        if method == "closed-form":
            raise ValueError(f"no closed-form CRPS for {type(distribution).__name__}")
    return crps_quadrature(distribution, observation)


# ---------------------------------------------------------------------------
# Logarithmic and Dawid-Sebastiani scores
# ---------------------------------------------------------------------------


def log_score(distribution: object, observation: float) -> float:
    """Return the logarithmic (ignorance) score ``-log f(y)``.

    The law's ``log_pdf`` is preferred, then ``pdf``, then ``log_pmf`` and
    ``pmf`` for discrete laws (the observation is then passed unchanged).

    Args:
        distribution: Law exposing a density or mass function.
        observation: Finite verifying value.

    Returns:
        Score in nats; ``+inf`` where the forecast assigns zero density.

    Raises:
        TypeError: If the law exposes none of the supported methods.
        ValueError: If the observation is non-finite.

    Equation:
        [EQ-VAL-SCORE-LOG-001].

    References:
        [REF-GNEITING-RAFTERY-2007].

    """
    y = require_finite_real(observation, name="observation")
    if isinstance(distribution, LogProbabilityDensity):
        return -float(distribution.log_pdf(y))
    if isinstance(distribution, ProbabilityDensity):
        density = float(distribution.pdf(y))
        return math.inf if density <= 0.0 else -math.log(density)
    value: object = observation
    if isinstance(observation, float) and observation.is_integer():
        value = int(observation)
    if isinstance(distribution, LogProbabilityMassFunction):
        return -float(distribution.log_pmf(value))
    if isinstance(distribution, ProbabilityMassFunction):
        mass = float(distribution.pmf(value))
        return math.inf if mass <= 0.0 else -math.log(mass)
    raise TypeError("distribution must expose log_pdf, pdf, log_pmf, or pmf")


def dawid_sebastiani_score(observation: float, mean: float, variance: float) -> float:
    """Return the Dawid-Sebastiani score ``(y - mu)^2 / s^2 + ln s^2``.

    It depends on the forecast only through its first two moments and is
    proper relative to them; it equals twice the Gaussian log score up to an
    additive constant. For an ensemble pass its sample mean and variance.

    Args:
        observation: Finite verifying value.
        mean: Finite predictive mean.
        variance: Positive predictive variance.

    Returns:
        Score (may be negative).

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``variance <= 0``.

    Equation:
        [EQ-VAL-SCORE-DSS-001].

    References:
        [REF-DAWID-SEBASTIANI-1999], [REF-GNEITING-RAFTERY-2007].

    """
    y = require_finite_real(observation, name="observation")
    mu = require_finite_real(mean, name="mean")
    var = _positive(variance, name="variance")
    return (y - mu) ** 2 / var + math.log(var)


# ---------------------------------------------------------------------------
# Multivariate scores
# ---------------------------------------------------------------------------


def _vectors(
    members: Sequence[Sequence[float]], observation: Sequence[float]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Validate an ensemble of equal-length vectors and the observed vector.

    References:
        Project validation policy. No equation is evaluated.

    """
    y = coerce_finite_sample(observation, name="observation")
    vectors = tuple(coerce_finite_sample(row, name="members") for row in members)
    if not vectors:
        raise ValueError("members must contain at least one vector")
    if any(len(row) != len(y) for row in vectors):
        raise ValueError("every member must have the observation's dimension")
    return vectors, y


def _euclidean(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Euclidean distance between two equal-length vectors.

    References:
        [REF-GNEITING-RAFTERY-2007].

    """
    return math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(first, second, strict=True))
    )


def energy_score(
    members: Sequence[Sequence[float]],
    observation: Sequence[float],
    *,
    fair: bool = False,
) -> float:
    """Return the energy score of a multivariate ensemble.

    ``ES = (1/m) sum ||x_i - y|| - (1/(2 m^2)) sum_i sum_j ||x_i - x_j||``
    (Euclidean norm), the multivariate generalization of the ensemble CRPS;
    in one dimension it equals :func:`crps_ensemble`. The fair version uses
    ``2 m (m - 1)`` in the spread term. Cost ``O(m^2 d)``.

    Args:
        members: ``m`` vectors of dimension ``d``.
        observation: Observed vector of dimension ``d``.
        fair: Use the fair spread denominator (needs ``m >= 2``).

    Returns:
        Score in the unit of the vectors.

    Raises:
        TypeError: If a value is not real.
        ValueError: If shapes disagree, a value is non-finite, or ``fair`` is
            requested with one member.

    Equation:
        [EQ-VAL-SCORE-ES-001].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-GNEITING-STANBERRY-ETAL-2008],
        [REF-FERRO-2014].

    """
    vectors, y = _vectors(members, observation)
    m = len(vectors)
    if fair and m < 2:
        raise ValueError("the fair energy score needs at least two members")
    accuracy = math.fsum(_euclidean(row, y) for row in vectors) / m
    pairs = math.fsum(
        _euclidean(vectors[i], vectors[j]) for i in range(m) for j in range(i + 1, m)
    )
    denominator = m * (m - 1) if fair else m * m
    return accuracy - pairs / denominator


def variogram_score(
    members: Sequence[Sequence[float]],
    observation: Sequence[float],
    *,
    order: float = 0.5,
    weights: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return the variogram score of order ``p`` of a multivariate ensemble.

    ``VS_p = sum_(i<j) w_ij (|y_i - y_j|^p - (1/m) sum_k |x_ki - x_kj|^p)^2``.
    This sums each unordered pair once, i.e. half of the double sum over all
    ``(i, j)`` with symmetric weights. The score targets the dependence
    structure and is proper but not strictly proper (it cannot see a common
    shift of all components).

    Args:
        members: ``m`` vectors of dimension ``d >= 2``.
        observation: Observed vector of dimension ``d``.
        order: Positive variogram order ``p`` (``0.5`` is the usual choice).
        weights: Optional non-negative ``d x d`` weights (only ``i < j`` are
            used); all ones by default.

    Returns:
        Non-negative score.

    Raises:
        TypeError: If a value is not real.
        ValueError: If shapes disagree, ``d < 2``, ``order <= 0``, or a
            weight is negative.

    Equation:
        [EQ-VAL-SCORE-VS-001].

    References:
        [REF-SCHEUERER-HAMILL-2015].

    """
    vectors, y = _vectors(members, observation)
    p = _positive(order, name="order")
    d = len(y)
    if d < 2:
        raise ValueError("the variogram score needs dimension at least two")
    if weights is not None:
        grid = tuple(coerce_finite_sample(row, name="weights") for row in weights)
        if len(grid) != d or any(len(row) != d for row in grid):
            raise ValueError("weights must be d x d")
        if any(value < 0.0 for row in grid for value in row):
            raise ValueError("weights must be non-negative")
    m = len(vectors)
    terms: list[float] = []
    for i in range(d):
        for j in range(i + 1, d):
            weight = 1.0 if weights is None else grid[i][j]
            expected = math.fsum(abs(row[i] - row[j]) ** p for row in vectors) / m
            terms.append(weight * (abs(y[i] - y[j]) ** p - expected) ** 2)
    return math.fsum(terms)


# ---------------------------------------------------------------------------
# Quantile and interval forecasts
# ---------------------------------------------------------------------------


def pinball_loss(quantile: float, observation: float, level: float) -> float:
    """Return the pinball (quantile) loss of a level-``tau`` quantile forecast.

    ``QS_tau(q, y) = (1{y < q} - tau)(q - y)``; it is minimized in
    expectation by the true ``tau``-quantile. ``2 int_0^1 QS_tau dtau`` is the
    CRPS.

    Args:
        quantile: Finite forecast quantile ``q``.
        observation: Finite verifying value ``y``.
        level: Quantile level ``tau`` in ``[0, 1]``.

    Returns:
        Non-negative loss.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or ``level`` is outside
            ``[0, 1]``.

    Equation:
        [EQ-VAL-SCORE-QS-001].

    References:
        [REF-GNEITING-RAFTERY-2007].

    """
    q = require_finite_real(quantile, name="quantile")
    y = require_finite_real(observation, name="observation")
    tau = require_probability(level, name="level")
    indicator = 1.0 if y < q else 0.0
    return (indicator - tau) * (q - y)


def _alpha(value: object) -> float:
    """Validate a miscoverage level strictly inside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    alpha = require_finite_real(value, name="alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    return alpha


def interval_score(
    lower: float, upper: float, observation: float, alpha: float
) -> float:
    """Return the interval score of a central ``(1 - alpha)`` prediction interval.

    ``IS_alpha(l, u; y) = (u - l) + (2/alpha)(l - y) 1{y < l}
    + (2/alpha)(y - u) 1{y > u}``: width plus a penalty proportional to the
    miss. It is proper for the ``alpha/2`` and ``1 - alpha/2`` quantiles, and
    ``(alpha/2) IS_alpha = QS_(alpha/2)(l, y) + QS_(1-alpha/2)(u, y)``.

    Args:
        lower: Finite lower endpoint.
        upper: Finite upper endpoint, not below ``lower``.
        observation: Finite verifying value.
        alpha: Miscoverage level in ``(0, 1)`` (``0.1`` for 90 % intervals).

    Returns:
        Non-negative score in the unit of ``y``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite, ``upper < lower``, or ``alpha``
            is outside ``(0, 1)``.

    Equation:
        [EQ-VAL-SCORE-IS-001].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-BRACHER-RAY-GNEITING-REICH-2021].

    """
    low = require_finite_real(lower, name="lower")
    high = require_finite_real(upper, name="upper")
    y = require_finite_real(observation, name="observation")
    a = _alpha(alpha)
    if high < low:
        raise ValueError("upper must not be below lower")
    score = high - low
    if y < low:
        score += 2.0 / a * (low - y)
    elif y > high:
        score += 2.0 / a * (y - high)
    return score


@dataclass(frozen=True, slots=True)
class WeightedIntervalScore:
    """Weighted interval score and its additive decomposition.

    Args:
        score: ``WIS = dispersion + overprediction + underprediction``.
        dispersion: Weighted interval widths (sharpness penalty).
        overprediction: Weighted misses with ``y`` below the forecast.
        underprediction: Weighted misses with ``y`` above the forecast.
        levels: Number ``K`` of central intervals used.

    References:
        [REF-BRACHER-RAY-GNEITING-REICH-2021].

    """

    score: float
    dispersion: float
    overprediction: float
    underprediction: float
    levels: int


def weighted_interval_score(
    observation: float,
    median: float,
    intervals: Sequence[tuple[float, float, float]],
) -> WeightedIntervalScore:
    """Return the weighted interval score of a quantile forecast.

    ``WIS = (1/(K + 1/2)) [ (1/2)|y - m| + sum_k (alpha_k/2) IS_(alpha_k) ]``
    over ``K`` central intervals ``(alpha_k, l_k, u_k)`` and the median
    ``m``. Since ``(alpha/2) IS_alpha`` is the sum of the pinball losses at
    ``alpha/2`` and ``1 - alpha/2`` and ``|y - m|/2 = QS_(1/2)(m, y)``, the
    WIS is twice the mean pinball loss over the ``2K + 1`` quantile levels;
    for many evenly spread levels it approximates the CRPS.

    Args:
        observation: Finite verifying value.
        median: Finite forecast median.
        intervals: ``(alpha, lower, upper)`` triples, ``alpha`` in ``(0, 1)``.

    Returns:
        Total score and its dispersion / over- / underprediction parts.

    Raises:
        TypeError: If a value is not real.
        ValueError: If an interval is invalid.

    Equation:
        [EQ-VAL-SCORE-WIS-001], [EQ-VAL-SCORE-IS-001].

    References:
        [REF-BRACHER-RAY-GNEITING-REICH-2021].

    """
    y = require_finite_real(observation, name="observation")
    centre = require_finite_real(median, name="median")
    triples = tuple(intervals)
    k = len(triples)
    dispersion: list[float] = []
    over: list[float] = [0.5 * max(0.0, centre - y)]
    under: list[float] = [0.5 * max(0.0, y - centre)]
    for alpha_value, lower, upper in triples:
        a = _alpha(alpha_value)
        low = require_finite_real(lower, name="lower")
        high = require_finite_real(upper, name="upper")
        if high < low:
            raise ValueError("upper must not be below lower")
        dispersion.append(0.5 * a * (high - low))
        over.append(max(0.0, low - y))
        under.append(max(0.0, y - high))
    norm = k + 0.5
    parts = (
        math.fsum(dispersion) / norm,
        math.fsum(over) / norm,
        math.fsum(under) / norm,
    )
    return WeightedIntervalScore(math.fsum(parts), parts[0], parts[1], parts[2], k)


# ---------------------------------------------------------------------------
# Binary events: Brier score and decomposition
# ---------------------------------------------------------------------------


def _binary(values: Sequence[object]) -> tuple[float, ...]:
    """Validate binary outcomes given as bools or the numbers 0 and 1.

    References:
        Project validation policy. No equation is evaluated.

    """
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            result.append(1.0 if value else 0.0)
            continue
        if not isinstance(value, Real) or float(value) not in (0.0, 1.0):
            raise ValueError(f"outcomes[{index}] must be 0, 1, False, or True")
        result.append(float(value))
    if not result:
        raise ValueError("outcomes must not be empty")
    return tuple(result)


def _probability_forecasts(
    probabilities: Sequence[float], outcomes: Sequence[object]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate paired probability forecasts and binary outcomes.

    References:
        Project validation policy. No equation is evaluated.

    """
    forecasts = tuple(
        require_probability(p, name="probabilities") for p in probabilities
    )
    observed = _binary(outcomes)
    if len(forecasts) != len(observed):
        raise ValueError("probabilities and outcomes must have equal length")
    return forecasts, observed


def brier_score(probabilities: Sequence[float], outcomes: Sequence[object]) -> float:
    """Return the mean Brier score ``(1/N) sum (f_i - o_i)^2``.

    Args:
        probabilities: Forecast probabilities of the event, in ``[0, 1]``.
        outcomes: Event indicators (bools or 0/1).

    Returns:
        Score in ``[0, 1]``.

    Raises:
        TypeError: If a forecast is not real.
        ValueError: If lengths differ, inputs are empty, or values are invalid.

    Equation:
        [EQ-VAL-SCORE-BS-001].

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-MURPHY-1973].

    """
    forecasts, observed = _probability_forecasts(probabilities, outcomes)
    return math.fsum(
        (f - o) ** 2 for f, o in zip(forecasts, observed, strict=True)
    ) / len(forecasts)


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy decomposition of the Brier score with within-bin terms.

    ``score = reliability - resolution + uncertainty
    + within_bin_variance - within_bin_covariance`` holds exactly (up to
    rounding) for any binning. When every bin holds one forecast value (the
    default grouping), both within-bin terms vanish and the classical
    three-term identity is exact.

    Args:
        score: Mean Brier score.
        reliability: ``(1/N) sum_k n_k (fbar_k - obar_k)^2`` (smaller is
            better).
        resolution: ``(1/N) sum_k n_k (obar_k - obar)^2`` (larger is better).
        uncertainty: ``obar (1 - obar)``.
        within_bin_variance: ``(1/N) sum_k sum_(i in k) (f_i - fbar_k)^2``.
        within_bin_covariance: ``(2/N) sum_k sum_(i in k) (f_i - fbar_k)
            (o_i - obar_k)``.
        bin_edges: Lower and upper edge of each non-empty bin.
        counts: Forecasts per non-empty bin.
        mean_forecasts: ``fbar_k`` per non-empty bin.
        observed_frequencies: ``obar_k`` per non-empty bin.

    References:
        [REF-MURPHY-1973], [REF-STEPHENSON-COELHO-JOLLIFFE-2008].

    """

    score: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float
    bin_edges: tuple[tuple[float, float], ...]
    counts: tuple[int, ...]
    mean_forecasts: tuple[float, ...]
    observed_frequencies: tuple[float, ...]

    @property
    def identity_residual(self) -> float:
        """Return ``score - (REL - RES + UNC + WBV - WBC)`` (rounding only).

        References:
            [EQ-VAL-SCORE-BSDEC-002].

        """
        return self.score - (
            self.reliability
            - self.resolution
            + self.uncertainty
            + self.within_bin_variance
            - self.within_bin_covariance
        )


def _bin_index(value: float, edges: Sequence[float]) -> int:
    """Return the bin of ``value`` for half-open bins, the last one closed.

    References:
        Project validation policy. No equation is evaluated.

    """
    last = len(edges) - 2
    for index in range(last):
        if value < edges[index + 1]:
            return index
    return last


def brier_decomposition(
    probabilities: Sequence[float],
    outcomes: Sequence[object],
    *,
    bins: int | Sequence[float] | None = None,
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution, and uncertainty.

    Forecasts are grouped by distinct value (``bins=None``), by ``bins``
    equal-width bins on ``[0, 1]``, or by explicit increasing edges spanning
    ``[0, 1]``. Bins are ``[e_k, e_(k+1))`` except the last, which is closed.
    Reliability uses the bin-mean forecast; the two within-bin terms of
    Stephenson, Coelho and Jolliffe (2008) restore the exact identity when a
    bin holds different forecast values.

    Args:
        probabilities: Forecast probabilities in ``[0, 1]``.
        outcomes: Event indicators (bools or 0/1).
        bins: ``None``, a positive bin count, or explicit edges.

    Returns:
        Decomposition with per-bin counts, mean forecasts, and frequencies.

    Raises:
        TypeError: If a value has an invalid type.
        ValueError: If inputs are invalid or edges do not span ``[0, 1]``.

    Equation:
        [EQ-VAL-SCORE-BSDEC-001], [EQ-VAL-SCORE-BSDEC-002].

    References:
        [REF-MURPHY-1973], [REF-STEPHENSON-COELHO-JOLLIFFE-2008].

    """
    forecasts, observed = _probability_forecasts(probabilities, outcomes)
    total = len(forecasts)
    groups: dict[int | float, list[int]] = {}
    edges_of: dict[int | float, tuple[float, float]] = {}
    if bins is None:
        for index, value in enumerate(forecasts):
            groups.setdefault(value, []).append(index)
            edges_of[value] = (value, value)
    else:
        edges = reliability_bin_edges(bins)
        for index, value in enumerate(forecasts):
            slot = _bin_index(value, edges)
            groups.setdefault(slot, []).append(index)
            edges_of[slot] = (edges[slot], edges[slot + 1])
    climatology = math.fsum(observed) / total
    reliability: list[float] = []
    resolution: list[float] = []
    variance: list[float] = []
    covariance: list[float] = []
    keys = sorted(groups)
    counts: list[int] = []
    means: list[float] = []
    freqs: list[float] = []
    for key in keys:
        members = groups[key]
        n_k = len(members)
        f_bar = math.fsum(forecasts[i] for i in members) / n_k
        o_bar = math.fsum(observed[i] for i in members) / n_k
        reliability.append(n_k * (f_bar - o_bar) ** 2)
        resolution.append(n_k * (o_bar - climatology) ** 2)
        variance.append(math.fsum((forecasts[i] - f_bar) ** 2 for i in members))
        covariance.append(
            math.fsum((forecasts[i] - f_bar) * (observed[i] - o_bar) for i in members)
        )
        counts.append(n_k)
        means.append(f_bar)
        freqs.append(o_bar)
    return BrierDecomposition(
        brier_score(forecasts, observed),
        math.fsum(reliability) / total,
        math.fsum(resolution) / total,
        climatology * (1.0 - climatology),
        math.fsum(variance) / total,
        2.0 * math.fsum(covariance) / total,
        tuple(edges_of[key] for key in keys),
        tuple(counts),
        tuple(means),
        tuple(freqs),
    )


def reliability_bin_edges(bins: int | Sequence[float]) -> tuple[float, ...]:
    """Return validated probability-bin edges spanning ``[0, 1]``.

    Args:
        bins: Positive number of equal-width bins, or explicit strictly
            increasing edges starting at ``0`` and ending at ``1``.

    Returns:
        Edge tuple of length ``bins + 1``.

    Raises:
        TypeError: If ``bins`` has an invalid type.
        ValueError: If the count is not positive or the edges are invalid.

    References:
        [REF-BROCKER-SMITH-2007]. No equation is evaluated.

    """
    if isinstance(bins, bool):
        raise TypeError("bins must be an integer or a sequence of edges")
    if isinstance(bins, int):
        if bins < 1:
            raise ValueError("bins must be positive")
        return tuple(k / bins for k in range(bins + 1))
    edges = tuple(require_probability(e, name="bins") for e in bins)
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("bin edges must start at 0 and end at 1")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("bin edges must be strictly increasing")
    return edges
