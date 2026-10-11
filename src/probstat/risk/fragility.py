"""Lognormal fragility curves, maximum-likelihood fitting, and hazard convolution.

A fragility function gives the conditional failure probability at an
intensity measure ``IM = a``. The lognormal form
``P(F | a) = Phi(ln(a / theta) / beta)`` has median capacity ``theta`` and
log-standard deviation (dispersion) ``beta``. Parameters are fitted by
maximizing the binomial likelihood of failure observations, and the annual
failure frequency follows by integrating the fragility against the slope of a
hazard curve ``lambda(a)`` (mean annual frequency of ``IM > a``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample, require_finite_real
from probstat.linalg import Vector
from probstat.modeling import standard_normal_cdf, standard_normal_quantile
from probstat.optimization import minimize_bfgs

_LOG_PARAMETER_LIMIT = 50.0


def _require_positive(value: float, *, name: str) -> float:
    """Validate a positive finite real number.

    Args:
        value: Candidate value.
        name: Parameter name for error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _log_standard_normal_cdf(z: float) -> float:
    """Return ``ln Phi(z)`` without underflow in the lower tail.

    Args:
        z: Finite argument.

    Returns:
        Natural log of the standard normal CDF; for ``z < -30`` the leading
        Mills-ratio asymptote is used.

    References:
        [EQ-DIST-NORM-002], [REF-NIST-DLMF, Equation 7.12.1].

    """
    if z > -30.0:
        return math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
    return -0.5 * z * z - math.log(-z) - 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class LognormalFragility:
    """Lognormal fragility function ``Phi(ln(a / theta) / beta)``.

    Args:
        median: Median capacity ``theta > 0`` (IM with 50% failure).
        dispersion: Logarithmic standard deviation ``beta > 0``.

    References:
        [REF-SHINOZUKA-ETAL-2000], [REF-BAKER-2015].

    """

    median: float
    dispersion: float

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is not positive and finite.

        References:
            [REF-BAKER-2015]. No equation is evaluated.

        """
        object.__setattr__(
            self, "median", _require_positive(self.median, name="median")
        )
        object.__setattr__(
            self, "dispersion", _require_positive(self.dispersion, name="dispersion")
        )

    def probability(self, intensity: float) -> float:
        """Return ``P(failure | IM = intensity)``.

        Args:
            intensity: Finite intensity; non-positive values give zero.

        Returns:
            Conditional failure probability.

        Equation:
            [EQ-RISK-FRAG-001].

        References:
            [REF-BAKER-2015].

        """
        a = require_finite_real(intensity, name="intensity")
        if a <= 0.0:
            return 0.0
        return standard_normal_cdf(math.log(a / self.median) / self.dispersion)

    def intensity_at(self, probability: float) -> float:
        """Return the intensity with a given conditional failure probability.

        Args:
            probability: Level strictly inside ``(0, 1)``.

        Returns:
            ``theta exp(beta Phi^-1(p))``.

        Equation:
            [EQ-RISK-FRAG-001].

        References:
            [REF-BAKER-2015].

        """
        return self.median * math.exp(
            self.dispersion * standard_normal_quantile(probability)
        )


@dataclass(frozen=True, slots=True)
class FragilityFit:
    """Maximum-likelihood lognormal fragility fit.

    Args:
        fragility: Fitted fragility function.
        log_likelihood: Maximized binomial log-likelihood (without the
            constant binomial coefficients).
        observations: Number of intensity levels or records used.
        trials: Total number of Bernoulli trials.
        converged: Whether the optimizer met its tolerance.
        iterations: Optimizer iterations.
        evaluations: Objective evaluations, including finite differences.

    References:
        [REF-BAKER-2015], [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """

    fragility: LognormalFragility
    log_likelihood: float
    observations: int
    trials: int
    converged: bool
    iterations: int
    evaluations: int


def fit_lognormal_fragility(
    intensities: Iterable[float],
    failures: Iterable[int | bool],
    *,
    trials: Iterable[int] | None = None,
) -> FragilityFit:
    """Fit ``theta`` and ``beta`` by maximum likelihood (Baker 2015).

    Without ``trials`` each observation is one binary outcome (``1`` or
    ``True`` = failure). With ``trials`` each entry of ``failures`` is the
    number of failures among ``trials[j]`` analyses at ``intensities[j]``
    (multiple-stripe data). The negative mean log-likelihood is minimized by
    BFGS over ``(ln theta, ln beta)``, which keeps both parameters positive.

    Args:
        intensities: Positive finite intensity levels.
        failures: Binary outcomes or failure counts aligned with intensities.
        trials: Optional positive trial counts per intensity.

    Returns:
        Fragility fit.

    Raises:
        ValueError: If inputs are inconsistent, fewer than two distinct
            intensities exist, data contain no failure or no survival, or
            failures and survivals are completely separated by intensity (the
            MLE does not exist; ``beta -> 0``).

    Equation:
        [EQ-RISK-FRAG-001], [EQ-RISK-FRAG-MLE-001], [EQ-OPT-BFGS-001].

    References:
        [REF-BAKER-2015], [REF-SHINOZUKA-ETAL-2000].

    """
    levels = coerce_finite_sample(intensities, name="intensities")
    if any(a <= 0.0 for a in levels):
        raise ValueError("intensities must be positive")
    raw_failures = tuple(failures)
    if trials is None:
        counts = tuple(
            _binary_outcome(value, index) for index, value in enumerate(raw_failures)
        )
        totals = (1,) * len(counts)
    else:
        totals = tuple(
            _count(value, f"trials[{i}]", minimum=1) for i, value in enumerate(trials)
        )
        counts = tuple(
            _count(value, f"failures[{i}]") for i, value in enumerate(raw_failures)
        )
        if len(totals) != len(counts):
            raise ValueError("trials must have the same length as failures")
        if any(z > n for z, n in zip(counts, totals, strict=True)):
            raise ValueError("failures cannot exceed trials")
    if len(counts) != len(levels):
        raise ValueError("failures must have the same length as intensities")
    if len(set(levels)) < 2:
        raise ValueError("at least two distinct intensities are required")
    failed = [a for a, z in zip(levels, counts, strict=True) if z > 0]
    survived = [a for a, z, n in zip(levels, counts, totals, strict=True) if z < n]
    if not failed or not survived:
        raise ValueError("data must contain at least one failure and one survival")
    if max(survived) < min(failed):
        raise ValueError(
            "failures and survivals are completely separated; the MLE does not exist"
        )
    log_levels = tuple(math.log(a) for a in levels)
    total_trials = sum(totals)

    def negative_mean_log_likelihood(parameters: Vector) -> float:
        """Return the negative binomial log-likelihood per trial.

        References:
            [EQ-RISK-FRAG-MLE-001], [REF-BAKER-2015].

        """
        log_theta, log_beta = parameters
        if abs(log_theta) > 700.0 or abs(log_beta) > _LOG_PARAMETER_LIMIT:
            raise ValueError("fragility parameters left the representable range")
        beta = math.exp(log_beta)
        terms: list[float] = []
        for x, z, n in zip(log_levels, counts, totals, strict=True):
            standardized = (x - log_theta) / beta
            if z:
                terms.append(z * _log_standard_normal_cdf(standardized))
            if n - z:
                terms.append((n - z) * _log_standard_normal_cdf(-standardized))
        return -math.fsum(terms) / total_trials

    mean_log = math.fsum(log_levels) / len(log_levels)
    spread = math.sqrt(
        math.fsum((x - mean_log) ** 2 for x in log_levels) / len(log_levels)
    )
    start = (mean_log, math.log(spread if spread > 0.05 else 0.5))
    result = minimize_bfgs(negative_mean_log_likelihood, start, gradient_tolerance=1e-7)
    log_theta, log_beta = result.x
    return FragilityFit(
        LognormalFragility(math.exp(log_theta), math.exp(log_beta)),
        -result.value * total_trials,
        len(levels),
        total_trials,
        result.converged,
        result.iterations,
        result.evaluations,
    )


def _binary_outcome(value: object, index: int) -> int:
    """Validate one binary failure indicator.

    Args:
        value: ``0``, ``1``, ``False``, or ``True``.
        index: Position for error messages.

    Returns:
        ``0`` or ``1``.

    Raises:
        ValueError: If the value is not binary.

    References:
        [REF-BAKER-2015]. No equation is evaluated.

    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"failures[{index}] must be 0, 1, False, or True")


def _count(value: object, name: str, *, minimum: int = 0) -> int:
    """Validate a non-negative integer count.

    Args:
        value: Candidate count.
        name: Field name for error messages.
        minimum: Smallest admissible value.

    Returns:
        The count.

    Raises:
        ValueError: If the value is not an integer of at least ``minimum``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class HazardConvolution:
    """Annual failure frequency from a fragility and a hazard curve.

    Args:
        annual_frequency: Mean annual failure frequency ``lambda_F``.
        contributions: ``(a_mid, P(F | a_mid) * delta lambda)`` per interval,
            the disaggregation of ``lambda_F`` by intensity.
        tail_contribution: ``P(F | a_max) lambda(a_max)`` for intensities
            beyond the last hazard level (zero if excluded).

    References:
        [REF-BAKER-BRADLEY-STAFFORD-2021], [REF-CORNELL-ETAL-2002].

    """

    annual_frequency: float
    contributions: tuple[tuple[float, float], ...]
    tail_contribution: float

    @property
    def return_period(self) -> float:
        """Return ``1 / lambda_F`` (``inf`` when the frequency is zero).

        Equation:
            [EQ-RISK-RP-001].

        References:
            [REF-BAKER-BRADLEY-STAFFORD-2021].

        """
        if self.annual_frequency <= 0.0:
            return math.inf
        return 1.0 / self.annual_frequency


def annual_failure_frequency(
    fragility: LognormalFragility | Callable[[float], float],
    hazard: Sequence[tuple[float, float]],
    *,
    include_tail: bool = True,
) -> HazardConvolution:
    """Integrate a fragility against a hazard curve (risk convolution).

    ``lambda_F = int P(F | a) |d lambda(a)|`` is approximated by
    ``sum_i P(F | sqrt(a_i a_{i+1})) (lambda(a_i) - lambda(a_{i+1}))`` with
    the fragility evaluated at the geometric midpoint of each hazard interval.
    Events below the first hazard level are not counted, so the first level
    must lie where the fragility is negligible.

    Args:
        fragility: Lognormal fragility or callable ``a -> P(F | a)``.
        hazard: At least two ``(a_i, lambda_i)`` pairs with strictly
            increasing positive ``a_i`` and non-increasing ``lambda_i >= 0``.
        include_tail: Add ``P(F | a_max) lambda(a_max)`` for ``IM > a_max``
            (a lower bound for that tail when the fragility increases).

    Returns:
        Convolution result with disaggregation.

    Raises:
        ValueError: If the hazard curve is invalid or the fragility returns a
            value outside ``[0, 1]``.

    Equation:
        [EQ-RISK-CONV-001].

    References:
        [REF-BAKER-BRADLEY-STAFFORD-2021], [REF-CORNELL-ETAL-2002].

    """
    curve = tuple(
        (
            require_finite_real(a, name="hazard level"),
            require_finite_real(rate, name="hazard rate"),
        )
        for a, rate in hazard
    )
    if len(curve) < 2:
        raise ValueError("hazard needs at least two points")
    for (a0, r0), (a1, r1) in zip(curve, curve[1:], strict=False):
        if a0 <= 0.0 or not a1 > a0:
            raise ValueError("hazard levels must be positive and strictly increasing")
        if r0 < 0.0 or r1 < 0.0 or r1 > r0:
            raise ValueError("hazard rates must be non-negative and non-increasing")
    evaluate = (
        fragility.probability
        if isinstance(fragility, LognormalFragility)
        else fragility
    )

    def conditional(a: float) -> float:
        """Evaluate and validate the conditional failure probability.

        References:
            [EQ-RISK-FRAG-001], [REF-BAKER-2015].

        """
        value = require_finite_real(evaluate(a), name="fragility value")
        if not 0.0 <= value <= 1.0:
            raise ValueError("fragility must return a probability in [0, 1]")
        return value

    contributions = tuple(
        (math.sqrt(a0 * a1), conditional(math.sqrt(a0 * a1)) * (r0 - r1))
        for (a0, r0), (a1, r1) in zip(curve, curve[1:], strict=False)
    )
    tail = conditional(curve[-1][0]) * curve[-1][1] if include_tail else 0.0
    total = math.fsum(c for _, c in contributions) + tail
    return HazardConvolution(total, contributions, tail)


def power_law_failure_frequency(
    k0: float, k: float, fragility: LognormalFragility
) -> float:
    """Return the closed-form ``lambda_F`` for a power-law hazard.

    For ``lambda(a) = k0 a^-k`` and a lognormal fragility,
    ``lambda_F = k0 theta^-k exp(k^2 beta^2 / 2)``; this is an oracle for
    :func:`annual_failure_frequency`.

    Args:
        k0: Positive hazard scale.
        k: Positive hazard log-slope.
        fragility: Lognormal fragility.

    Returns:
        Annual failure frequency.

    Equation:
        [EQ-RISK-CONV-002].

    References:
        [REF-CORNELL-ETAL-2002].

    """
    scale = _require_positive(k0, name="k0")
    slope = _require_positive(k, name="k")
    theta, beta = fragility.median, fragility.dispersion
    return scale * math.pow(theta, -slope) * math.exp(0.5 * slope * slope * beta * beta)
