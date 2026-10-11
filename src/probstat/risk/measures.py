"""Risk measures on loss samples: expected loss, VaR, CVaR, and exceedance.

Losses are oriented so that larger values are worse. Every sample routine
accepts optional non-negative weights, so the same code serves equally weighted
Monte Carlo output, importance-sampling output, and finite scenario sets
("risk = scenarios x probabilities x consequences").

Conventions:

* ``value_at_risk`` is the *lower* ``alpha``-quantile of the loss law,
  ``VaR_alpha = inf{l : F(l) >= alpha}``; for equal weights this is the
  Hyndman-Fan type-1 quantile (inverse empirical CDF), never an interpolated
  value.
* ``conditional_value_at_risk`` uses the Rockafellar-Uryasev definition
  ``CVaR_alpha = VaR_alpha + E[(L - VaR_alpha)^+] / (1 - alpha)``, which is
  correct for laws with atoms and for finite samples with ties: it averages
  exactly the worst ``1 - alpha`` probability mass, splitting the atom at
  ``VaR_alpha`` when necessary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import (
    coerce_finite_sample,
    require_finite_real,
    require_probability,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import standard_normal_cdf, standard_normal_quantile
from probstat.statistics import quantile

_CUMULATIVE_TOLERANCE = 1e-10
MEASURES = ("var", "cvar", "expected_loss")


def _require_level(alpha: object, *, name: str = "alpha") -> float:
    """Validate a risk level strictly inside ``(0, 1)``.

    Args:
        alpha: Candidate confidence level of the risk measure.
        name: Parameter name used in error messages.

    Returns:
        The level as a float.

    Raises:
        TypeError: If ``alpha`` is not real.
        ValueError: If ``alpha`` is not strictly between zero and one.

    References:
        [REF-ROCKAFELLAR-URYASEV-2002]. No equation is evaluated.

    """
    level = require_probability(alpha, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return level


def _sorted_weighted(
    losses: Iterable[float], weights: Iterable[float] | None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate losses and weights, normalize weights, and sort by loss.

    Args:
        losses: Non-empty finite losses.
        weights: Optional non-negative finite weights with a positive sum.

    Returns:
        Ascending losses and their normalized weights.

    Raises:
        ValueError: If the sample is empty, lengths differ, a weight is
            negative, or the weights sum to zero.

    References:
        [REF-ROCKAFELLAR-URYASEV-2002]. No equation is evaluated.

    """
    values = coerce_finite_sample(losses, name="losses")
    size = len(values)
    if weights is None:
        normalized = (1.0 / size,) * size
    else:
        raw = coerce_finite_sample(weights, name="weights")
        if len(raw) != size:
            raise ValueError("weights must have the same length as losses")
        if any(weight < 0.0 for weight in raw):
            raise ValueError("weights must be non-negative")
        total = math.fsum(raw)
        if total <= 0.0:
            raise ValueError("weights must have a positive sum")
        normalized = tuple(weight / total for weight in raw)
    order = sorted(range(size), key=values.__getitem__)
    return (
        tuple(values[index] for index in order),
        tuple(normalized[index] for index in order),
    )


def _lower_quantile(
    values: Sequence[float], weights: Sequence[float], alpha: float
) -> float:
    """Return ``inf{l : F(l) >= alpha}`` for a sorted weighted sample.

    Args:
        values: Ascending losses.
        weights: Normalized weights aligned with ``values``.
        alpha: Level in ``(0, 1)``.

    Returns:
        Lower ``alpha``-quantile; cumulative sums are compared with a
        ``1e-10`` tolerance to absorb floating-point rounding.

    References:
        [EQ-RISK-VAR-001], [REF-ROCKAFELLAR-URYASEV-2002].

    """
    cumulative = 0.0
    for value, weight in zip(values, weights, strict=True):
        cumulative += weight
        if cumulative >= alpha - _CUMULATIVE_TOLERANCE:
            return value
    return values[-1]


def _cvar_sorted(
    values: Sequence[float], weights: Sequence[float], alpha: float
) -> float:
    """Return the Rockafellar-Uryasev CVaR of a sorted weighted sample.

    Args:
        values: Ascending losses.
        weights: Normalized weights aligned with ``values``.
        alpha: Level in ``(0, 1)``.

    Returns:
        ``VaR + E[(L - VaR)^+] / (1 - alpha)``.

    References:
        [EQ-RISK-CVAR-001], [REF-ROCKAFELLAR-URYASEV-2002].

    """
    var = _lower_quantile(values, weights, alpha)
    excess = math.fsum(
        weight * (value - var)
        for value, weight in zip(values, weights, strict=True)
        if value > var
    )
    return var + excess / (1.0 - alpha)


def expected_loss(
    losses: Iterable[float], *, weights: Iterable[float] | None = None
) -> float:
    """Return the (weighted) mean loss ``E[L]``.

    Args:
        losses: Non-empty finite losses.
        weights: Optional non-negative weights (normalized internally).

    Returns:
        Expected loss.

    Raises:
        ValueError: If the sample or weights are invalid.

    Equation:
        [EQ-RISK-EL-001].

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2].

    """
    values, normalized = _sorted_weighted(losses, weights)
    return math.fsum(v * w for v, w in zip(values, normalized, strict=True))


def value_at_risk(
    losses: Iterable[float],
    alpha: float,
    *,
    weights: Iterable[float] | None = None,
) -> float:
    """Return the lower ``alpha``-quantile of the loss distribution.

    Args:
        losses: Non-empty finite losses (larger is worse).
        alpha: Level strictly inside ``(0, 1)``, e.g. ``0.95``.
        weights: Optional non-negative weights (normalized internally).

    Returns:
        ``VaR_alpha = inf{l : sum_{L_i <= l} w_i >= alpha}``. With equal
        weights this is the order statistic ``L_(ceil(n alpha))``.

    Raises:
        ValueError: If ``alpha`` or the sample is invalid.

    Equation:
        [EQ-RISK-VAR-001].

    References:
        [REF-ROCKAFELLAR-URYASEV-2002], [REF-HYNDMAN-FAN-1996].

    """
    level = _require_level(alpha)
    values, normalized = _sorted_weighted(losses, weights)
    return _lower_quantile(values, normalized, level)


def conditional_value_at_risk(
    losses: Iterable[float],
    alpha: float,
    *,
    weights: Iterable[float] | None = None,
) -> float:
    """Return the conditional value-at-risk (expected shortfall) of losses.

    The Rockafellar-Uryasev formula averages exactly the worst ``1 - alpha``
    probability mass. When an atom (a tie) straddles ``VaR_alpha`` only the
    needed fraction of it is counted, so the result is a coherent risk
    measure for every finite sample.

    Args:
        losses: Non-empty finite losses (larger is worse).
        alpha: Level strictly inside ``(0, 1)``.
        weights: Optional non-negative weights (normalized internally).

    Returns:
        ``CVaR_alpha = VaR_alpha + E[(L - VaR_alpha)^+] / (1 - alpha)``.

    Raises:
        ValueError: If ``alpha`` or the sample is invalid.

    Equation:
        [EQ-RISK-CVAR-001].

    References:
        [REF-ROCKAFELLAR-URYASEV-2000], [REF-ROCKAFELLAR-URYASEV-2002],
        [REF-ACERBI-TASCHE-2002], [REF-ARTZNER-ETAL-1999].

    """
    level = _require_level(alpha)
    values, normalized = _sorted_weighted(losses, weights)
    return _cvar_sorted(values, normalized, level)


def expected_shortfall(
    losses: Iterable[float],
    alpha: float,
    *,
    weights: Iterable[float] | None = None,
) -> float:
    """Return expected shortfall, identical to :func:`conditional_value_at_risk`.

    Args:
        losses: Non-empty finite losses.
        alpha: Level strictly inside ``(0, 1)``.
        weights: Optional non-negative weights.

    Returns:
        Expected shortfall at level ``alpha``.

    Equation:
        [EQ-RISK-CVAR-001].

    References:
        [REF-ACERBI-TASCHE-2002], [REF-ROCKAFELLAR-URYASEV-2002].

    """
    return conditional_value_at_risk(losses, alpha, weights=weights)


def exceedance_probability(
    losses: Iterable[float],
    level: float,
    *,
    weights: Iterable[float] | None = None,
) -> float:
    """Return the (weighted) empirical probability ``P(L > level)``.

    Args:
        losses: Non-empty finite losses.
        level: Finite loss threshold; equality does not count as exceedance.
        weights: Optional non-negative weights.

    Returns:
        Exceedance probability in ``[0, 1]``.

    Equation:
        [EQ-RISK-EXC-001].

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-BAKER-BRADLEY-STAFFORD-2021].

    """
    threshold = require_finite_real(level, name="level")
    values, normalized = _sorted_weighted(losses, weights)
    probability = math.fsum(
        w for v, w in zip(values, normalized, strict=True) if v > threshold
    )
    return min(1.0, probability)


def loss_exceedance_curve(
    losses: Iterable[float],
    levels: Iterable[float] | None = None,
    *,
    weights: Iterable[float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return the loss exceedance curve ``l -> P(L > l)``.

    Args:
        losses: Non-empty finite losses.
        levels: Finite loss levels; defaults to the distinct observed losses.
        weights: Optional non-negative weights.

    Returns:
        Pairs ``(l, P(L > l))`` sorted by ``l``; the curve is non-increasing.

    Equation:
        [EQ-RISK-EXC-001].

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-BAKER-BRADLEY-STAFFORD-2021].

    """
    values, normalized = _sorted_weighted(losses, weights)
    if levels is None:
        grid = tuple(sorted(set(values)))
    else:
        grid = tuple(sorted(coerce_finite_sample(levels, name="levels")))
    curve: list[tuple[float, float]] = []
    index = 0
    below = 0.0
    for level in grid:
        while index < len(values) and values[index] <= level:
            below += normalized[index]
            index += 1
        curve.append((level, max(0.0, min(1.0, 1.0 - below))))
    return tuple(curve)


def return_period(annual_rate: float) -> float:
    """Return the mean recurrence interval ``T = 1 / lambda`` in years.

    Args:
        annual_rate: Positive mean annual exceedance frequency ``lambda``.

    Returns:
        Return period in years.

    Raises:
        ValueError: If the rate is not positive and finite.

    Equation:
        [EQ-RISK-RP-001].

    References:
        [REF-BAKER-BRADLEY-STAFFORD-2021].

    """
    rate = require_finite_real(annual_rate, name="annual_rate")
    if rate <= 0.0:
        raise ValueError("annual_rate must be positive")
    return 1.0 / rate


def poisson_exceedance_probability(annual_rate: float, years: float = 1.0) -> float:
    """Return ``P(at least one exceedance in t years) = 1 - exp(-lambda t)``.

    Args:
        annual_rate: Non-negative mean annual exceedance frequency.
        years: Non-negative exposure time.

    Returns:
        Probability of one or more events under a homogeneous Poisson model.

    Raises:
        ValueError: If the rate or exposure is negative or non-finite.

    Equation:
        [EQ-RISK-RP-001].

    References:
        [REF-BAKER-BRADLEY-STAFFORD-2021].

    """
    rate = require_finite_real(annual_rate, name="annual_rate")
    exposure = require_finite_real(years, name="years")
    if rate < 0.0 or exposure < 0.0:
        raise ValueError("annual_rate and years must be non-negative")
    return -math.expm1(-rate * exposure)


def _equal_weight_measure(
    ordered: Sequence[float], measure: str, alpha: float
) -> float:
    """Evaluate a risk measure on an ascending, equally weighted sample.

    This fast path avoids re-validating bootstrap resamples. VaR is the order
    statistic ``L_(ceil(n alpha))`` (with the same ``1e-10`` cumulative
    tolerance as the weighted path) and CVaR adds the mean excess over it.

    Args:
        ordered: Ascending finite losses.
        measure: One of :data:`MEASURES`.
        alpha: Level in ``(0, 1)``.

    Returns:
        Measure value.

    References:
        [EQ-RISK-VAR-001], [EQ-RISK-CVAR-001], [EQ-RISK-EL-001],
        [REF-ROCKAFELLAR-URYASEV-2002].

    """
    size = len(ordered)
    if measure == "expected_loss":
        return math.fsum(ordered) / size
    index = max(0, math.ceil(size * (alpha - _CUMULATIVE_TOLERANCE)) - 1)
    var = ordered[min(index, size - 1)]
    if measure == "var":
        return var
    excess = math.fsum(value - var for value in ordered if value > var)
    return var + excess / (size * (1.0 - alpha))


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile bootstrap interval for one sample risk measure.

    Args:
        measure: ``"var"``, ``"cvar"``, or ``"expected_loss"``.
        alpha: Risk level (ignored for expected loss but recorded).
        estimate: Measure evaluated on the original sample.
        lower: Lower percentile-interval endpoint.
        upper: Upper percentile-interval endpoint.
        standard_error: Standard deviation of the bootstrap replicates.
        confidence: Nominal two-sided confidence.
        replicates: Number of bootstrap resamples.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 13].

    """

    measure: str
    alpha: float
    estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence: float
    replicates: int


def bootstrap_risk_interval(
    losses: Iterable[float],
    alpha: float = 0.95,
    *,
    measure: str = "cvar",
    confidence: float = 0.95,
    replicates: int = 1000,
    seed: int | RandomStream | None = None,
) -> BootstrapInterval:
    """Return a seeded nonparametric percentile bootstrap interval.

    Each replicate resamples the (equally weighted) losses with replacement
    and re-evaluates the measure. Percentile intervals for extreme-tail VaR
    and CVaR undercover when few losses lie beyond ``VaR_alpha``; inspect
    ``n (1 - alpha)`` before trusting the interval.

    Args:
        losses: Finite losses with at least two observations.
        alpha: Risk level strictly inside ``(0, 1)``.
        measure: One of :data:`MEASURES`.
        confidence: Two-sided confidence strictly inside ``(0, 1)``.
        replicates: Number of resamples, at least 20.
        seed: Integer seed or stream; resampling uses the child stream
            ``("risk", "bootstrap", measure)``.

    Returns:
        Bootstrap interval record.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-RISK-BOOT-001], [EQ-RISK-VAR-001], [EQ-RISK-CVAR-001].

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 13], [REF-HYNDMAN-FAN-1996].

    """
    sample = coerce_finite_sample(losses, name="losses")
    if len(sample) < 2:
        raise ValueError("at least two losses are required")
    level = _require_level(alpha)
    coverage = _require_level(confidence, name="confidence")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("replicates must be an integer")
    if replicates < 20:
        raise ValueError("replicates must be at least 20")
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    generator = resolve_stream(seed).child("risk", "bootstrap", measure).generator()
    size = len(sample)
    replicate_values = [
        _equal_weight_measure(sorted(generator.choices(sample, k=size)), measure, level)
        for _ in range(replicates)
    ]
    center = math.fsum(replicate_values) / replicates
    spread = math.sqrt(
        math.fsum((value - center) ** 2 for value in replicate_values)
        / (replicates - 1)
    )
    tail = 0.5 * (1.0 - coverage)
    return BootstrapInterval(
        measure,
        level,
        _equal_weight_measure(sorted(sample), measure, level),
        quantile(replicate_values, tail),
        quantile(replicate_values, 1.0 - tail),
        spread,
        coverage,
        replicates,
    )


def _standard_normal_pdf(z: float) -> float:
    """Return the standard normal density ``phi(z)``.

    Args:
        z: Finite argument.

    Returns:
        Density value.

    References:
        [EQ-DIST-NORM-001], [REF-NIST-DLMF].

    """
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _require_scale(sigma: float) -> float:
    """Validate a positive finite scale parameter.

    Args:
        sigma: Candidate scale.

    Returns:
        The scale as a float.

    Raises:
        ValueError: If ``sigma`` is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    scale = require_finite_real(sigma, name="sigma")
    if scale <= 0.0:
        raise ValueError("sigma must be positive")
    return scale


def normal_value_at_risk(mu: float, sigma: float, alpha: float) -> float:
    """Return ``VaR_alpha`` of a ``Normal(mu, sigma^2)`` loss.

    Args:
        mu: Finite mean loss.
        sigma: Positive loss standard deviation.
        alpha: Level strictly inside ``(0, 1)``.

    Returns:
        ``mu + sigma Phi^-1(alpha)``.

    Equation:
        [EQ-RISK-NORMAL-001].

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2].

    """
    location = require_finite_real(mu, name="mu")
    return location + _require_scale(sigma) * standard_normal_quantile(
        _require_level(alpha)
    )


def normal_conditional_value_at_risk(mu: float, sigma: float, alpha: float) -> float:
    """Return ``CVaR_alpha`` of a ``Normal(mu, sigma^2)`` loss.

    Args:
        mu: Finite mean loss.
        sigma: Positive loss standard deviation.
        alpha: Level strictly inside ``(0, 1)``.

    Returns:
        ``mu + sigma phi(Phi^-1(alpha)) / (1 - alpha)``.

    Equation:
        [EQ-RISK-NORMAL-001].

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2],
        [REF-ROCKAFELLAR-URYASEV-2000].

    """
    location = require_finite_real(mu, name="mu")
    level = _require_level(alpha)
    z = standard_normal_quantile(level)
    return location + _require_scale(sigma) * _standard_normal_pdf(z) / (1.0 - level)


def lognormal_value_at_risk(mu: float, sigma: float, alpha: float) -> float:
    """Return ``VaR_alpha`` of a loss with ``ln L ~ Normal(mu, sigma^2)``.

    Args:
        mu: Finite log-location.
        sigma: Positive log-scale.
        alpha: Level strictly inside ``(0, 1)``.

    Returns:
        ``exp(mu + sigma Phi^-1(alpha))``.

    Equation:
        [EQ-RISK-LOGNORMAL-001].

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2].

    """
    location = require_finite_real(mu, name="mu")
    return math.exp(
        location
        + _require_scale(sigma) * standard_normal_quantile(_require_level(alpha))
    )


def lognormal_conditional_value_at_risk(mu: float, sigma: float, alpha: float) -> float:
    """Return ``CVaR_alpha`` of a loss with ``ln L ~ Normal(mu, sigma^2)``.

    Args:
        mu: Finite log-location.
        sigma: Positive log-scale.
        alpha: Level strictly inside ``(0, 1)``.

    Returns:
        ``exp(mu + sigma^2 / 2) Phi(sigma - Phi^-1(alpha)) / (1 - alpha)``.

    Equation:
        [EQ-RISK-LOGNORMAL-001].

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2].

    """
    location = require_finite_real(mu, name="mu")
    scale = _require_scale(sigma)
    level = _require_level(alpha)
    z = standard_normal_quantile(level)
    return (
        math.exp(location + 0.5 * scale * scale)
        * standard_normal_cdf(scale - z)
        / (1.0 - level)
    )
