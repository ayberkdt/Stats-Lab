"""E-values: sensitivity of an observed association to unmeasured confounding.

The E-value of a risk ratio ``RR`` (EQ-CAUSAL-EVALUE-001) is the minimum
strength of association, on the risk-ratio scale, that an unmeasured
confounder would need with both the treatment and the outcome, conditional on
the measured covariates, to fully explain away the observed association. It is
the common value ``E`` at which the bounding factor
``B = RR_UD RR_EU / (RR_UD + RR_EU - 1)`` (EQ-CAUSAL-BIAS-001) equals ``RR``.

Odds ratios and standardized mean differences can be mapped approximately to
the risk-ratio scale first (EQ-CAUSAL-RRAPPROX-001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real


def _positive(value: object, *, name: str) -> float:
    """Validate a finite, strictly positive ratio.

    Args:
        value: Candidate ratio.
        name: Argument name for error messages.

    Returns:
        The ratio as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite and positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return checked


def _e_value_of(ratio: float) -> float:
    """Return ``RR + sqrt(RR (RR - 1))`` after inverting protective ratios.

    Args:
        ratio: Positive risk ratio.

    Returns:
        E-value (at least one).

    Equation:
        [EQ-CAUSAL-EVALUE-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    rr = ratio if ratio >= 1.0 else 1.0 / ratio
    return rr + math.sqrt(rr * (rr - 1.0))


@dataclass(frozen=True, slots=True)
class EValue:
    """E-values for a risk-ratio point estimate and its confidence limit.

    Attributes:
        risk_ratio: Observed risk ratio.
        point: E-value of the point estimate.
        confidence_limit: E-value of the confidence limit closest to the null
            (``1.0`` when the interval contains one); ``None`` when no
            interval was supplied.

    References:
        [REF-VANDERWEELE-DING-2017].

    """

    risk_ratio: float
    point: float
    confidence_limit: float | None


def e_value(
    risk_ratio: float, *, lower: float | None = None, upper: float | None = None
) -> EValue:
    """Compute the E-value of a risk ratio and, optionally, of its interval.

    For ``RR >= 1``, ``E = RR + sqrt(RR (RR - 1))``; a protective ``RR < 1``
    is inverted first. The interval E-value uses the limit closest to one and
    equals one if the interval covers one.

    Args:
        risk_ratio: Observed (adjusted) risk ratio, strictly positive.
        lower: Optional lower confidence limit.
        upper: Optional upper confidence limit. For ``RR >= 1`` only the lower
            limit is used; for ``RR < 1`` only the upper limit.

    Returns:
        :class:`EValue`.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is not positive, the limits do not bracket the
            estimate, or the needed limit is missing while the other is given.

    Equation:
        [EQ-CAUSAL-EVALUE-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    rr = _positive(risk_ratio, name="risk_ratio")
    lo = None if lower is None else _positive(lower, name="lower")
    hi = None if upper is None else _positive(upper, name="upper")
    if lo is not None and lo > rr:
        raise ValueError("lower must not exceed risk_ratio")
    if hi is not None and hi < rr:
        raise ValueError("upper must not be below risk_ratio")
    limit_value: float | None = None
    if lo is not None or hi is not None:
        limit = lo if rr >= 1.0 else hi
        if limit is None:
            needed = "lower" if rr >= 1.0 else "upper"
            raise ValueError(f"the {needed} limit is required for this risk ratio")
        crosses = limit <= 1.0 if rr >= 1.0 else limit >= 1.0
        limit_value = 1.0 if crosses else _e_value_of(limit)
    return EValue(rr, _e_value_of(rr), limit_value)


def confounding_bias_factor(
    exposure_confounder_rr: float, confounder_outcome_rr: float
) -> float:
    """Return the maximal bias factor of an unmeasured confounder.

    ``B = RR_EU RR_UD / (RR_EU + RR_UD - 1)``: dividing an observed risk ratio
    by ``B`` gives the most it could be reduced by a confounder with those two
    associations. ``B(E, E) = RR`` when ``E`` is the E-value of ``RR``.

    Args:
        exposure_confounder_rr: ``RR_EU >= 1``, treatment-confounder strength.
        confounder_outcome_rr: ``RR_UD >= 1``, confounder-outcome strength.

    Returns:
        Bias factor ``B >= 1``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a strength is below one or not finite.

    Equation:
        [EQ-CAUSAL-BIAS-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    eu = _positive(exposure_confounder_rr, name="exposure_confounder_rr")
    ud = _positive(confounder_outcome_rr, name="confounder_outcome_rr")
    if eu < 1.0 or ud < 1.0:
        raise ValueError("association strengths must be at least one")
    return eu * ud / (eu + ud - 1.0)


def risk_ratio_from_odds_ratio(odds_ratio: float, *, common_outcome: bool) -> float:
    """Approximate a risk ratio from an odds ratio for E-value purposes.

    A rare outcome (below roughly 15 percent) gives ``RR ~ OR``; a common
    outcome uses ``RR ~ sqrt(OR)``.

    Args:
        odds_ratio: Strictly positive odds ratio.
        common_outcome: Whether the outcome is common.

    Returns:
        Approximate risk ratio.

    Raises:
        TypeError: If the odds ratio is not real.
        ValueError: If it is not finite and positive.

    Equation:
        [EQ-CAUSAL-RRAPPROX-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    checked = _positive(odds_ratio, name="odds_ratio")
    return math.sqrt(checked) if common_outcome else checked


def risk_ratio_from_standardized_difference(difference: float) -> float:
    """Approximate a risk ratio from a standardized mean difference ``d``.

    Uses ``RR ~ exp(0.91 d)`` for a continuous outcome.

    Args:
        difference: Standardized mean difference (mean difference divided by
            the outcome standard deviation).

    Returns:
        Approximate risk ratio.

    Raises:
        TypeError: If ``difference`` is not real.
        ValueError: If it is not finite.

    Equation:
        [EQ-CAUSAL-RRAPPROX-001].

    References:
        [REF-VANDERWEELE-DING-2017].

    """
    return math.exp(0.91 * require_finite_real(difference, name="difference"))
