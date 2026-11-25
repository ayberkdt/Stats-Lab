"""Normal-theory tolerance factors, bounds, and intervals.

A *tolerance bound* answers the requirement "at least a fraction ``p`` of the
population stays below ``B`` with confidence ``gamma``" from a sample of ``n``
values. Under the **assumption that the values are independent draws of a
normal law** the bound is ``B = xbar + k s`` with the sample mean ``xbar``,
the sample standard deviation ``s``, and a factor ``k(n, p, gamma)`` that
makes the statement exact:

- One-sided, exact: ``k = t'_(gamma; n-1, delta) / sqrt(n)`` with the
  gamma-quantile of a noncentral ``t`` law with ``n - 1`` degrees of freedom
  and noncentrality ``delta = z_p sqrt(n)`` ([EQ-PROP-TOL-001]). The
  noncentral ``t`` distribution function is evaluated here as an integral
  over the chi distribution of ``s / sigma`` ([EQ-PROP-TOL-002]); it is
  private to this module and no new distribution family is added.
- Two-sided, exact: the interval ``xbar +- k s`` contains at least ``p`` of
  the law with confidence ``gamma`` when ``k`` solves the integral equation
  over the standardized deviation ``z`` of the sample mean
  ``2 int_0^inf phi(z) P(chi2_(n-1) >= (n-1) r(z)^2 / k^2) dz = gamma``,
  where ``r(z)`` is the half-width that makes an interval centred at
  ``z / sqrt(n)`` hold content ``p`` ([EQ-PROP-TOL-003]).
- Two-sided, Howe (1969): ``k = z_((1+p)/2) sqrt((n-1)(1 + 1/n) /
  chi2_(1-gamma; n-1))`` ([EQ-PROP-TOL-004]), a closed form that never
  needs the integral. Its error against the exact factor at ``p = gamma =
  0.95`` is about +2.1 % at ``n = 3``, +0.33 % at ``n = 5``, -0.34 % at
  ``n = 10``, -0.21 % at ``n = 30``, -0.05 % at ``n = 100`` and below
  0.002 % at ``n = 1000`` (positive: Howe is conservative; negative: it is
  too small, so the stated confidence is not quite reached). The exact
  factor is the default and Howe's is a named alternative whose error the
  tests quantify.

**Normality is an assumption, not a result of this module.** A tolerance
bound from ``xbar + k s`` is only as valid as the normal model of the tail
it extrapolates to; a skewed or heavy-tailed output makes the stated
confidence meaningless. :func:`probstat.propagation.wilks_bound` gives the
distribution-free alternative that needs no model but more samples (59 for
a 95 %/95 % upper bound), and :func:`normality_price` reports both bounds
for the same sample so the price of dropping normality is visible.

Numerical notes: the noncentral ``t`` integral is evaluated to a relative
tolerance of ``1e-13`` and the factors agree with base R's ``qt`` to about
``1e-11`` where R itself is accurate (R's ``qt`` with ``ncp`` warns about
lost precision for large noncentrality, for example ``n = 1000``, where
this module's value is the one that satisfies the defining integral). The
two-sided equation
cancels two normal probabilities, so ``p`` closer to one than ``1e-10`` is
rejected rather than silently degraded.

Assumptions: independent draws; normal law (the user's modelling choice);
the same ``n`` and the same standard deviation estimate as the data.

References:
    [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-ODEH-OWEN-1980],
    [REF-NIST-SEMATECH-HANDBOOK], [REF-WILKS-1941].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare, Normal
from probstat.foundations.validation import require_finite_real
from probstat.numerics.gauss_kronrod import K15_NODES, K15_WEIGHTS, gauss_kronrod
from probstat.optimization import brent_root
from probstat.statistics import mean, variance

from .wilks import WilksBound, wilks_bound, wilks_sample_size

FACTOR_SIDES = ("one-sided", "two-sided")
FACTOR_METHODS = ("exact", "howe")
BOUND_SIDES = ("upper", "lower")

_STANDARD_NORMAL = Normal(0.0, 1.0)
_TAIL = 1.0e-17  # probability mass cut from each end of the chi integral
_Z_MAX = 8.5  # tail mass beyond 8.5 is ~1e-17: the sample-mean deviation is cut here
_Z_PANELS = 17  # panels of width 0.5 on [0, _Z_MAX]
_MAX_CONTENT = 1.0 - 1.0e-10


@dataclass(frozen=True, slots=True)
class NormalToleranceBound:
    """A one-sided normal-theory tolerance bound.

    Args:
        bound: ``xbar + k s`` (upper) or ``xbar - k s`` (lower).
        factor: Tolerance factor ``k``.
        mean: Sample mean ``xbar``.
        standard_deviation: Sample standard deviation ``s`` (divisor
            ``n - 1``).
        size: Number of values ``n``.
        coverage: Population fraction ``p`` the bound is meant to cover.
        confidence: Confidence ``gamma`` of the statement.
        sided: ``"upper"`` or ``"lower"``.

    References:
        [EQ-PROP-TOL-001], [REF-KRISHNAMOORTHY-MATHEW-2009].

    """

    bound: float
    factor: float
    mean: float
    standard_deviation: float
    size: int
    coverage: float
    confidence: float
    sided: str


@dataclass(frozen=True, slots=True)
class NormalToleranceInterval:
    """A two-sided normal-theory tolerance interval ``xbar +- k s``.

    Args:
        lower: ``xbar - k s``.
        upper: ``xbar + k s``.
        factor: Tolerance factor ``k``.
        method: ``"exact"`` or ``"howe"``.
        mean: Sample mean ``xbar``.
        standard_deviation: Sample standard deviation ``s``.
        size: Number of values ``n``.
        coverage: Population fraction ``p`` the interval is meant to cover.
        confidence: Confidence ``gamma`` of the statement.

    References:
        [EQ-PROP-TOL-003], [EQ-PROP-TOL-004], [REF-HOWE-1969].

    """

    lower: float
    upper: float
    factor: float
    method: str
    mean: float
    standard_deviation: float
    size: int
    coverage: float
    confidence: float


@dataclass(frozen=True, slots=True)
class NormalityPrice:
    """The normal-theory and distribution-free upper bounds side by side.

    Args:
        normal: Normal-theory bound ``xbar + k s`` (or the lower analogue).
        wilks: Wilks order-statistic bound, ``None`` when the sample is too
            small for the requested guarantee.
        wilks_required_size: Smallest sample size for a first-order Wilks
            bound with the same level, so a user sees what dropping
            normality costs in runs.
        difference: ``wilks.bound - normal.bound`` for an upper bound
            (``normal.bound - wilks.bound`` for a lower bound), so a
            positive value means the distribution-free bound is the more
            conservative one; ``None`` when no Wilks bound exists.

    References:
        [REF-WILKS-1941], [EQ-PROP-TOL-001], [EQ-PROP-WILKS-001].

    """

    normal: NormalToleranceBound
    wilks: WilksBound | None
    wilks_required_size: int
    difference: float | None


def _level(value: float, *, name: str) -> float:
    """Return a finite level strictly inside ``(0, 1)``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is outside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    level = require_finite_real(value, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return level


def _sample_size(value: object) -> int:
    """Return a sample size of at least two, rejecting booleans.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is smaller than two.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("size must be an integer")
    if value < 2:
        raise ValueError("size must be at least two")
    return value


def _choice(value: str, allowed: tuple[str, ...], *, name: str) -> str:
    """Return ``value`` if it is one of ``allowed``.

    Raises:
        ValueError: If the value is not allowed.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def _normal_cdf(x: float) -> float:
    """Return ``Phi(x)`` through ``erfc`` (accurate in both tails).

    References:
        [REF-NIST-DLMF-ELEMENTARY]. No new equation.

    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _normal_pdf(x: float) -> float:
    """Return the standard normal density.

    References:
        [REF-NIST-DLMF-ELEMENTARY]. No new equation.

    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _z(probability: float) -> float:
    """Return the standard normal quantile.

    References:
        [REF-NIST-DLMF-ELEMENTARY]. No new equation.

    """
    return float(_STANDARD_NORMAL.quantile(probability))


def _noncentral_t_cdf(t: float, df: int, delta: float) -> float:
    """Return ``P(T' <= t)`` for ``T' = (Z + delta) / sqrt(V / df)``.

    Conditional on ``s = sqrt(V / df)`` the event is ``Z <= t s - delta``,
    so ``P = int_0^inf Phi(t s - delta) f_S(s) ds`` with the chi density
    ``f_S(s) = 2 df s f_V(df s^2)`` ([EQ-PROP-TOL-002]). The integral runs
    over the central range of ``S`` that holds all but ``2e-17`` of its
    mass and is evaluated by adaptive Gauss-Kronrod quadrature.

    References:
        [EQ-PROP-TOL-002].

    """
    chi = ChiSquare(float(df))
    low = math.sqrt(float(chi.quantile(_TAIL)) / df)
    high = math.sqrt(float(chi.isf(_TAIL)) / df)

    def integrand(s: float) -> float:
        """Return ``Phi(t s - delta) f_S(s)``.

        References:
            [EQ-PROP-TOL-002]. No new equation.

        """
        return _normal_cdf(t * s - delta) * 2.0 * df * s * float(chi.pdf(df * s * s))

    result = gauss_kronrod(integrand, low, high, abs_tol=1.0e-14, rel_tol=1.0e-13)
    return min(1.0, max(0.0, result.value))


def _noncentral_t_quantile(probability: float, df: int, delta: float) -> float:
    """Return the ``probability``-quantile of the noncentral ``t`` law.

    The start point ``delta + z_q`` (exact as ``df -> infinity``) is widened
    geometrically to a sign-changing bracket and refined by Brent's method.

    References:
        [EQ-PROP-TOL-001], [EQ-PROP-TOL-002].

    """
    guess = delta + _z(probability)
    step = max(1.0, 0.25 * abs(guess))
    low = high = guess
    while _noncentral_t_cdf(low, df, delta) > probability:
        low -= step
        step *= 2.0
    step = max(1.0, 0.25 * abs(guess))
    while _noncentral_t_cdf(high, df, delta) < probability:
        high += step
        step *= 2.0
    root = brent_root(
        lambda t: _noncentral_t_cdf(t, df, delta) - probability,
        low,
        high,
        tolerance=1.0e-13 * max(1.0, abs(guess)),
    )
    return float(root.root)


def _one_sided_factor(n: int, coverage: float, confidence: float) -> float:
    """Return the exact one-sided factor ``t'(gamma; n-1, z_p sqrt(n)) / sqrt(n)``.

    References:
        [EQ-PROP-TOL-001].

    """
    delta = _z(coverage) * math.sqrt(n)
    return _noncentral_t_quantile(confidence, n - 1, delta) / math.sqrt(n)


def _centred_content_radius(shift: float, coverage: float, upper: float) -> float:
    """Return ``r`` with ``Phi(shift + r) - Phi(shift - r) = coverage``.

    ``upper`` is a bracket end that already holds the content.

    References:
        [EQ-PROP-TOL-003].

    """
    root = brent_root(
        lambda r: _normal_cdf(shift + r) - _normal_cdf(shift - r) - coverage,
        0.0,
        upper,
        tolerance=1.0e-14,
    )
    return float(root.root)


def _two_sided_rule(n: int, coverage: float) -> list[tuple[float, float]]:
    """Return ``(weight phi(z), r(z)^2)`` over a composite K15 rule on ``[0, 8.5]``.

    The radii do not depend on ``k``, so one rule serves the whole root
    search for the factor.

    References:
        [EQ-PROP-TOL-003].

    """
    if coverage > _MAX_CONTENT:
        raise ValueError("coverage is too close to one for the two-sided factor")
    base = _z(0.5 * (1.0 + coverage))
    width = _Z_MAX / _Z_PANELS
    full_nodes = [
        (node, weight) for node, weight in zip(K15_NODES, K15_WEIGHTS, strict=True)
    ]
    rule: list[tuple[float, float]] = []
    for panel in range(_Z_PANELS):
        centre = (panel + 0.5) * width
        for node, weight in full_nodes:
            offsets = (0.0,) if node == 0.0 else (-node, node)
            for offset in offsets:
                z = centre + 0.5 * width * offset
                shift = z / math.sqrt(n)
                radius = _centred_content_radius(shift, coverage, base + shift)
                rule.append((0.5 * width * weight * _normal_pdf(z), radius * radius))
    return rule


def _two_sided_confidence(
    k: float, rule: list[tuple[float, float]], chi: ChiSquare, nu: int
) -> float:
    """Return ``2 int_0^inf phi(z) P(chi2_nu >= nu r(z)^2 / k^2) dz``.

    References:
        [EQ-PROP-TOL-003].

    """
    total = 0.0
    for weight, radius_squared in rule:
        total += weight * float(chi.sf(nu * radius_squared / (k * k)))
    return min(1.0, 2.0 * total)


def _two_sided_exact_factor(n: int, coverage: float, confidence: float) -> float:
    """Return the exact two-sided factor by root search on the integral equation.

    References:
        [EQ-PROP-TOL-003].

    """
    nu = n - 1
    chi = ChiSquare(float(nu))
    rule = _two_sided_rule(n, coverage)
    base = _z(0.5 * (1.0 + coverage))

    def residual(k: float) -> float:
        """Return confidence at ``k`` minus the target.

        References:
            [EQ-PROP-TOL-003]. No new equation.

        """
        return _two_sided_confidence(k, rule, chi, nu) - confidence

    low = 0.25 * base
    while residual(low) > 0.0:
        low *= 0.5
    high = 2.0 * base
    while residual(high) < 0.0:
        high *= 2.0
    root = brent_root(residual, low, high, tolerance=1.0e-13 * high)
    return float(root.root)


def _howe_factor(n: int, coverage: float, confidence: float) -> float:
    """Return Howe's two-sided approximation.

    References:
        [EQ-PROP-TOL-004], [REF-HOWE-1969].

    """
    nu = n - 1
    lower_quantile = float(ChiSquare(float(nu)).quantile(1.0 - confidence))
    z = _z(0.5 * (1.0 + coverage))
    return z * math.sqrt(nu * (1.0 + 1.0 / n) / lower_quantile)


def normal_tolerance_factor(
    size: int,
    coverage: float,
    confidence: float,
    *,
    sided: str = "one-sided",
    method: str = "exact",
) -> float:
    """Return the normal-theory tolerance factor ``k(n, p, gamma)``.

    **Normality of the sampled law is assumed.** One-sided: the bound
    ``xbar + k s`` exceeds the ``p``-quantile (``xbar - k s`` is below the
    ``(1 - p)``-quantile) with probability ``gamma``, ``k = t'_(gamma; n-1, z_p sqrt(n)) /
    sqrt(n)``. Two-sided: ``[xbar - k s, xbar + k s]`` contains at least a
    fraction ``p`` of the law with probability ``gamma``. ``method="exact"``
    solves the defining equation; ``method="howe"`` (two-sided only) is the
    closed-form approximation of Howe (1969), whose factor is slightly too
    large for ``n <= 5`` and slightly too small from about ``n = 10`` on
    (module docstring gives the measured errors). For one-sided factors only the exact method
    exists. The factors decrease to ``z_p`` (one-sided) or
    ``z_((1+p)/2)`` (two-sided) as ``n`` grows.

    For a distribution-free alternative see
    :func:`probstat.propagation.wilks_bound`; compare the two with
    :func:`normality_price`.

    Args:
        size: Sample size ``n`` (integer, at least two).
        coverage: Population fraction ``p`` strictly inside ``(0, 1)``;
            below ``1 - 1e-10`` for the two-sided factor.
        confidence: Confidence ``gamma`` strictly inside ``(0, 1)``.
        sided: ``"one-sided"`` or ``"two-sided"``.
        method: ``"exact"`` or ``"howe"``.

    Returns:
        The factor ``k``.

    Raises:
        TypeError: If ``size`` is not an integer or a level is not real.
        ValueError: If a level is out of range, ``size < 2``, an option is
            unknown, or ``method="howe"`` is requested with ``one-sided``.

    Equation:
        [EQ-PROP-TOL-001], [EQ-PROP-TOL-003], [EQ-PROP-TOL-004].

    References:
        [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009],
        [REF-NIST-SEMATECH-HANDBOOK].

    """
    n = _sample_size(size)
    p = _level(coverage, name="coverage")
    gamma = _level(confidence, name="confidence")
    _choice(sided, FACTOR_SIDES, name="sided")
    _choice(method, FACTOR_METHODS, name="method")
    if sided == "one-sided":
        if method != "exact":
            raise ValueError("the one-sided factor is available only as 'exact'")
        return _one_sided_factor(n, p, gamma)
    if method == "howe":
        return _howe_factor(n, p, gamma)
    return _two_sided_exact_factor(n, p, gamma)


def _moments(values: Sequence[float]) -> tuple[int, float, float]:
    """Return ``(n, mean, sample standard deviation)`` of finite values.

    Raises:
        TypeError: If a value is not real.
        ValueError: If fewer than two values are given or one is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    sample = [require_finite_real(value, name="value") for value in values]
    if len(sample) < 2:
        raise ValueError("values must contain at least two values")
    return (
        len(sample),
        float(mean(sample)),
        math.sqrt(float(variance(sample, ddof=1))),
    )


def normal_tolerance_bound(
    values: Sequence[float],
    coverage: float,
    confidence: float,
    *,
    sided: str = "upper",
) -> NormalToleranceBound:
    """Return the exact one-sided normal tolerance bound of a sample.

    Under the **assumption of a normal law**, ``xbar + k s`` (upper) lies
    above the ``p``-quantile with probability ``gamma``; ``xbar - k s``
    (lower) lies below the ``(1 - p)``-quantile, so at least a fraction ``p``
    of the law lies above it.

    Args:
        values: Independent finite sample of at least two values.
        coverage: Population fraction ``p`` strictly inside ``(0, 1)``.
        confidence: Confidence ``gamma`` strictly inside ``(0, 1)``.
        sided: ``"upper"`` or ``"lower"``.

    Returns:
        Bound, factor, and the sample moments.

    Raises:
        TypeError: If a value or level is not real.
        ValueError: If fewer than two values, a non-finite value, a level
            outside ``(0, 1)``, or an unknown ``sided``.

    Equation:
        [EQ-PROP-TOL-001].

    References:
        [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-NIST-SEMATECH-HANDBOOK].

    """
    p = _level(coverage, name="coverage")
    gamma = _level(confidence, name="confidence")
    _choice(sided, BOUND_SIDES, name="sided")
    n, centre, spread = _moments(values)
    factor = _one_sided_factor(n, p, gamma)
    sign = 1.0 if sided == "upper" else -1.0
    return NormalToleranceBound(
        bound=centre + sign * factor * spread,
        factor=factor,
        mean=centre,
        standard_deviation=spread,
        size=n,
        coverage=p,
        confidence=gamma,
        sided=sided,
    )


def normal_tolerance_interval(
    values: Sequence[float],
    coverage: float,
    confidence: float,
    *,
    method: str = "exact",
) -> NormalToleranceInterval:
    """Return the two-sided normal tolerance interval ``xbar +- k s``.

    **Normality of the sampled law is assumed.** The interval contains at
    least a fraction ``coverage`` of the law with probability
    ``confidence``. ``method="howe"`` uses Howe's closed form instead of the
    exact factor.

    Args:
        values: Independent finite sample of at least two values.
        coverage: Population fraction ``p`` strictly inside ``(0, 1)``.
        confidence: Confidence ``gamma`` strictly inside ``(0, 1)``.
        method: ``"exact"`` or ``"howe"``.

    Returns:
        The interval with its factor and the sample moments.

    Raises:
        TypeError: If a value or level is not real.
        ValueError: If fewer than two values, a non-finite value, a level
            out of range, or an unknown ``method``.

    Equation:
        [EQ-PROP-TOL-003], [EQ-PROP-TOL-004].

    References:
        [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009].

    """
    _choice(method, FACTOR_METHODS, name="method")
    n, centre, spread = _moments(values)
    factor = normal_tolerance_factor(
        n, coverage, confidence, sided="two-sided", method=method
    )
    return NormalToleranceInterval(
        lower=centre - factor * spread,
        upper=centre + factor * spread,
        factor=factor,
        method=method,
        mean=centre,
        standard_deviation=spread,
        size=n,
        coverage=float(coverage),
        confidence=float(confidence),
    )


def normal_tolerance_sample_size(
    coverage: float,
    confidence: float,
    max_factor: float,
    *,
    sided: str = "one-sided",
    method: str = "exact",
    max_size: int = 100_000,
) -> int:
    """Return the smallest ``n`` whose tolerance factor is at most ``max_factor``.

    The factor decreases in ``n`` towards ``z_p`` (one-sided) or
    ``z_((1+p)/2)`` (two-sided), so a target below that limit is
    unattainable. The answer is found by doubling and bisection. It answers
    "how many samples before the statistical allowance ``k s`` shrinks to
    ``max_factor`` standard deviations?"

    Args:
        coverage: Population fraction ``p`` strictly inside ``(0, 1)``.
        confidence: Confidence ``gamma`` strictly inside ``(0, 1)``.
        max_factor: Largest acceptable factor, above the limit value.
        sided: ``"one-sided"`` or ``"two-sided"``.
        method: ``"exact"`` or ``"howe"`` (two-sided only).
        max_size: Largest ``n`` considered (at least 2).

    Returns:
        The smallest sample size meeting the target.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a level is out of range, the target is not above the
            limit factor, or it needs more than ``max_size`` values.

    Equation:
        [EQ-PROP-TOL-001], [EQ-PROP-TOL-003].

    References:
        [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-ODEH-OWEN-1980].

    """
    p = _level(coverage, name="coverage")
    gamma = _level(confidence, name="confidence")
    target = require_finite_real(max_factor, name="max_factor")
    _choice(sided, FACTOR_SIDES, name="sided")
    _choice(method, FACTOR_METHODS, name="method")
    limit = _z(p) if sided == "one-sided" else _z(0.5 * (1.0 + p))
    if target <= limit:
        raise ValueError(
            f"max_factor must exceed the limit factor {limit:.6g} reached as n grows"
        )
    cap = _sample_size(max_size)

    def passes(n: int) -> bool:
        """Return whether the factor at ``n`` meets the target.

        References:
            [EQ-PROP-TOL-001]. No new equation.

        """
        return (
            normal_tolerance_factor(n, p, gamma, sided=sided, method=method) <= target
        )

    low, high = 1, 2
    while not passes(high):
        if high >= cap:
            raise ValueError(f"more than {cap} values are needed")
        low, high = high, min(2 * high, cap)
    while high - low > 1:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle
    return high


def normality_price(
    values: Sequence[float],
    coverage: float,
    confidence: float,
    *,
    sided: str = "upper",
) -> NormalityPrice:
    """Report the normal tolerance bound and the Wilks bound for one sample.

    Both bounds carry the same statement: "at least a fraction ``coverage``
    of the output stays on the safe side with probability ``confidence``".
    For ``sided="upper"`` both bound the ``coverage`` quantile from above; for
    ``sided="lower"`` both bound the ``1 - coverage`` quantile from below
    (the Wilks order statistic is therefore taken at level ``1 - coverage``).
    The normal bound relies on the normal model; the Wilks bound is an order
    statistic and needs no model. The difference is the price of dropping
    the normality assumption; the required Wilks sample size is the price in
    runs. When the sample is too small for any Wilks bound, ``wilks`` is
    ``None`` and only the size is reported.

    Args:
        values: Independent finite sample of at least two values.
        coverage: Population fraction ``p`` strictly inside ``(0, 1)``.
        confidence: Confidence ``gamma`` strictly inside ``(0, 1)``.
        sided: ``"upper"`` or ``"lower"``.

    Returns:
        Both bounds, their difference, and the Wilks sample size.

    Raises:
        TypeError: If a value or level is not real.
        ValueError: If the sample or a level is invalid.

    Equation:
        [EQ-PROP-TOL-001], [EQ-PROP-WILKS-001].

    References:
        [REF-WILKS-1941], [REF-KRISHNAMOORTHY-MATHEW-2009].

    """
    normal = normal_tolerance_bound(values, coverage, confidence, sided=sided)
    # The Wilks functions take the quantile LEVEL. An upper bound covers the
    # ``coverage`` quantile from above; a lower bound ``xbar - k s`` covers the
    # ``1 - coverage`` quantile from below.
    level = coverage if sided == "upper" else 1.0 - coverage
    needed = wilks_sample_size(level, confidence, sided=sided)
    distribution_free: WilksBound | None = None
    gap: float | None = None
    if normal.size >= needed:
        distribution_free = wilks_bound(values, level, confidence, sided=sided)
        if sided == "upper":
            gap = distribution_free.bound - normal.bound
        else:
            gap = normal.bound - distribution_free.bound
    return NormalityPrice(
        normal=normal,
        wilks=distribution_free,
        wilks_required_size=needed,
        difference=gap,
    )


__all__ = [
    "NormalToleranceBound",
    "NormalToleranceInterval",
    "NormalityPrice",
    "normal_tolerance_bound",
    "normal_tolerance_factor",
    "normal_tolerance_interval",
    "normal_tolerance_sample_size",
    "normality_price",
]
