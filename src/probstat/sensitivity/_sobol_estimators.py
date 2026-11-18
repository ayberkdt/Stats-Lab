"""Pick-freeze estimators of first-order and total Sobol indices.

All estimators read the same three output blocks of a pick-freeze design
with ``N`` rows: ``y_A = f(A)``, ``y_B = f(B)``, and ``y_i = f(A_B^(i))``,
where ``A_B^(i)`` is ``A`` with column ``i`` taken from ``B``. The pair
``(y_B, y_i)`` shares only ``x_i`` and gives the first-order index; the pair
``(y_A, y_i)`` shares every input except ``x_i`` and gives the total index.
The estimators differ only in how they combine these blocks, so one design
can be scored by all of them at no extra model cost.

The default ``"saltelli2010"`` estimator is evaluated by the calling modules
themselves (so its floating-point results stay bit-for-bit unchanged); this
module holds the alternatives, whose formulas follow the R package
``sensitivity`` ([REF-R-SENSITIVITY]): ``sobol2002``, ``sobol2007``,
``soboljansen``, ``sobolmartinez``, and the first-order estimator of
``sobolEff`` (Janon et al. 2014).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from probstat.distributions import Normal

SOBOL_ESTIMATORS: tuple[str, ...] = (
    "saltelli2010",
    "saltelli2002",
    "sobol2007",
    "jansen",
    "martinez",
    "janon",
)
"""Names accepted by the ``estimator`` argument of the Sobol functions."""

ASYMPTOTIC_ESTIMATORS: tuple[str, ...] = ("martinez", "janon")
"""Estimators with closed-form asymptotic confidence intervals."""


def check_estimator(estimator: object) -> str:
    """Return a validated Sobol estimator name.

    Args:
        estimator: Candidate name.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If the name is not one of :data:`SOBOL_ESTIMATORS`.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not isinstance(estimator, str) or estimator not in SOBOL_ESTIMATORS:
        raise ValueError(f"estimator must be one of {SOBOL_ESTIMATORS}")
    return estimator


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence.

    References:
        [REF-PYTHON-MATH]. No sensitivity equation is evaluated.

    """
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance (denominator ``n - 1``).

    References:
        [REF-PYTHON-MATH]. No sensitivity equation is evaluated.

    """
    centre = _mean(values)
    return math.fsum((v - centre) ** 2 for v in values) / (len(values) - 1)


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Pearson correlation from centred products.

    Raises:
        ValueError: If either sequence is constant.

    Equation:
        [EQ-SENS-EST-004].

    References:
        [REF-BAUDIN-ETAL-2016].

    """
    m1, m2 = _mean(first), _mean(second)
    d1 = [v - m1 for v in first]
    d2 = [v - m2 for v in second]
    s11 = math.fsum(v * v for v in d1)
    s22 = math.fsum(v * v for v in d2)
    if s11 <= 0.0 or s22 <= 0.0:
        raise ValueError("the Martinez estimator requires non-constant outputs")
    return math.fsum(a * b for a, b in zip(d1, d2, strict=True)) / math.sqrt(s11 * s22)


def _janon_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Janon et al. closed-index estimate for one output pair.

    Both outputs are centred on the mean of the pooled pair ``m``; the ratio
    ``sum (y - m)(y' - m) / (sum ((y - m)^2 + (y' - m)^2) / 2)`` is
    algebraically identical to the published uncentred form.

    Raises:
        ValueError: If the pooled pair is constant.

    Equation:
        [EQ-SENS-EST-006].

    References:
        [REF-JANON-ETAL-2014].

    """
    centre = 0.5 * (_mean(first) + _mean(second))
    numerator = math.fsum(
        (a - centre) * (b - centre) for a, b in zip(first, second, strict=True)
    )
    denominator = 0.5 * math.fsum(
        (a - centre) ** 2 + (b - centre) ** 2
        for a, b in zip(first, second, strict=True)
    )
    if denominator <= 0.0:
        raise ValueError("the Janon estimator requires non-constant outputs")
    return numerator / denominator


def closed_index(
    estimator: str,
    ya: Sequence[float],
    yb: Sequence[float],
    column: Sequence[float],
    variance: float,
) -> float:
    """Return the first-order (closed) index of the inputs ``column`` shares with B.

    ``column`` holds ``f`` on ``A`` with the columns of a group ``u`` taken
    from ``B``; the result estimates ``Var(E[Y | X_u]) / Var(Y)``. With a
    single column it is the first-order index ``S_i``.

    Args:
        estimator: Name other than ``"saltelli2010"``.
        ya: Outputs on A.
        yb: Outputs on B.
        column: Outputs on the hybrid matrix.
        variance: Sample variance of ``ya`` (denominator ``n - 1``).

    Returns:
        Unclipped closed-index estimate.

    Raises:
        ValueError: If the estimator is unknown here or a pair is constant.

    Equation:
        [EQ-SENS-EST-001], [EQ-SENS-EST-002], [EQ-SENS-EST-003],
        [EQ-SENS-EST-004], [EQ-SENS-EST-006].

    References:
        [REF-SALTELLI-2002], [REF-SOBOL-ETAL-2007], [REF-JANSEN-1999],
        [REF-BAUDIN-ETAL-2016], [REF-JANON-ETAL-2014], [REF-R-SENSITIVITY].

    """
    n = len(ya)
    if estimator == "saltelli2002":
        product = math.fsum(b * c for b, c in zip(yb, column, strict=True))
        squared_mean = math.fsum(a * b for a, b in zip(ya, yb, strict=True)) / n
        return (product / (n - 1) - squared_mean) / variance
    if estimator == "sobol2007":
        return (
            math.fsum(b * (c - a) for a, b, c in zip(ya, yb, column, strict=True))
            / (n - 1)
            / variance
        )
    if estimator == "jansen":
        residual = math.fsum((b - c) ** 2 for b, c in zip(yb, column, strict=True)) / (
            2 * n - 1
        )
        return (variance - residual) / variance
    if estimator == "martinez":
        return _correlation(yb, column)
    if estimator == "janon":
        return _janon_ratio(yb, column)
    raise ValueError(f"no closed-index rule for estimator {estimator!r}")


def total_index(
    estimator: str,
    ya: Sequence[float],
    yb: Sequence[float],
    column: Sequence[float],
    variance: float,
) -> float:
    """Return the total-effect index of the input that ``column`` takes from B.

    Args:
        estimator: Name other than ``"saltelli2010"``.
        ya: Outputs on A.
        yb: Outputs on B (unused by the total formulas, kept for symmetry).
        column: Outputs on ``A_B^(i)``.
        variance: Sample variance of ``ya`` (denominator ``n - 1``).

    Returns:
        Unclipped total-effect estimate.

    Raises:
        ValueError: If the estimator is unknown here or a pair is constant.

    Equation:
        [EQ-SENS-EST-001], [EQ-SENS-EST-002], [EQ-SENS-EST-003],
        [EQ-SENS-EST-004], [EQ-SENS-EST-006].

    References:
        [REF-SALTELLI-2002], [REF-SOBOL-ETAL-2007], [REF-JANSEN-1999],
        [REF-BAUDIN-ETAL-2016], [REF-JANON-ETAL-2014], [REF-R-SENSITIVITY].

    """
    del yb
    n = len(ya)
    if estimator == "saltelli2002":
        product = math.fsum(a * c for a, c in zip(ya, column, strict=True))
        return 1.0 - (product / (n - 1) - _mean(ya) ** 2) / variance
    if estimator == "sobol2007":
        return (
            math.fsum(a * (a - c) for a, c in zip(ya, column, strict=True))
            / (n - 1)
            / variance
        )
    if estimator == "jansen":
        return (
            math.fsum((a - c) ** 2 for a, c in zip(ya, column, strict=True))
            / (2 * n - 1)
            / variance
        )
    if estimator == "martinez":
        return 1.0 - _correlation(ya, column)
    if estimator == "janon":
        return 1.0 - _janon_ratio(ya, column)
    raise ValueError(f"no total-index rule for estimator {estimator!r}")


def alternative_indices(
    estimator: str,
    ya: Sequence[float],
    yb: Sequence[float],
    yab: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], float]:
    """Return first-order and total indices of every input for one estimator.

    Args:
        estimator: Name other than ``"saltelli2010"``.
        ya: Outputs on A (at least two rows).
        yb: Outputs on B.
        yab: Outputs on ``A_B^(i)``, one sequence per input.

    Returns:
        First-order list, total-effect list, and the sample variance of
        ``ya`` (denominator ``n - 1``) used as the reported output variance.

    Raises:
        ValueError: If the output variance is zero or a pair is constant.

    Equation:
        [EQ-SENS-EST-001], [EQ-SENS-EST-002], [EQ-SENS-EST-003],
        [EQ-SENS-EST-004], [EQ-SENS-EST-006].

    References:
        [REF-R-SENSITIVITY], [REF-SALTELLI-2002], [REF-SOBOL-ETAL-2007],
        [REF-JANSEN-1999], [REF-BAUDIN-ETAL-2016], [REF-JANON-ETAL-2014].

    """
    variance = _sample_variance(ya)
    if variance <= 0.0:
        raise ValueError("Sobol indices require positive output variance")
    first = [closed_index(estimator, ya, yb, column, variance) for column in yab]
    total = [total_index(estimator, ya, yb, column, variance) for column in yab]
    return first, total, variance


def _fisher_interval(
    correlation: float, size: int, confidence: float
) -> tuple[float, float]:
    """Return the Fisher-z interval ``tanh(atanh(r) -+ z / sqrt(n - 3))``.

    Raises:
        ValueError: If fewer than four rows are available.

    Equation:
        [EQ-SENS-EST-005].

    References:
        [REF-FISHER-1921], [REF-BAUDIN-ETAL-2016].

    """
    if size < 4:
        raise ValueError("Fisher-z intervals need at least four rows")
    quantile = Normal(0.0, 1.0).quantile(0.5 + 0.5 * confidence)
    clipped = max(-1.0 + 1e-15, min(1.0 - 1e-15, correlation))
    centre = math.atanh(clipped)
    half = quantile / math.sqrt(size - 3)
    return math.tanh(centre - half), math.tanh(centre + half)


def _janon_interval(
    first: Sequence[float], second: Sequence[float], confidence: float
) -> tuple[float, float, float]:
    """Return a Janon estimate with its asymptotic normal interval.

    With ``m`` the pooled pair mean, ``a = y - m``, ``b = y' - m``, ``T`` the
    estimate and ``U = a b - T (a^2 + b^2) / 2``, the standard error is
    ``sd(U) / (sqrt(N) s^2)`` where ``s^2`` is the pooled pair variance.

    Raises:
        ValueError: If the pooled pair is constant.

    Equation:
        [EQ-SENS-EST-007].

    References:
        [REF-JANON-ETAL-2014].

    """
    estimate = _janon_ratio(first, second)
    n = len(first)
    centre = 0.5 * (_mean(first) + _mean(second))
    a = [v - centre for v in first]
    b = [v - centre for v in second]
    pair_variance = 0.5 * math.fsum(x * x + y * y for x, y in zip(a, b, strict=True))
    pair_variance /= n
    influence = [
        x * y - 0.5 * estimate * (x * x + y * y) for x, y in zip(a, b, strict=True)
    ]
    spread = _sample_variance(influence)
    error = math.sqrt(spread / n) / pair_variance
    quantile = Normal(0.0, 1.0).quantile(0.5 + 0.5 * confidence)
    return estimate, estimate - quantile * error, estimate + quantile * error


def asymptotic_intervals(
    estimator: str,
    ya: Sequence[float],
    yb: Sequence[float],
    yab: Sequence[Sequence[float]],
    confidence: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return ``(estimate, lower, upper)`` triples from asymptotic theory.

    ``"martinez"`` uses Fisher's z transform of each correlation
    ([EQ-SENS-EST-005]); the total-effect bounds are ``1 - `` the reversed
    correlation bounds. ``"janon"`` uses the asymptotic normal law of the
    Janon estimator ([EQ-SENS-EST-007]) on the pairs ``(y_B, y_i)`` and
    ``(y_A, y_i)``. Both assume independent, identically distributed rows.

    Args:
        estimator: ``"martinez"`` or ``"janon"``.
        ya: Outputs on A.
        yb: Outputs on B.
        yab: Outputs on ``A_B^(i)``, one sequence per input.
        confidence: Nominal level in ``(0, 1)``.

    Returns:
        First-order and total-effect triples, one per input.

    Raises:
        ValueError: For another estimator, too few rows, or constant outputs.

    Equation:
        [EQ-SENS-EST-004], [EQ-SENS-EST-005], [EQ-SENS-EST-006],
        [EQ-SENS-EST-007].

    References:
        [REF-BAUDIN-ETAL-2016], [REF-FISHER-1921], [REF-JANON-ETAL-2014].

    """
    first: list[tuple[float, float, float]] = []
    total: list[tuple[float, float, float]] = []
    if estimator == "martinez":
        for column in yab:
            rho = _correlation(yb, column)
            low, high = _fisher_interval(rho, len(ya), confidence)
            first.append((rho, low, high))
            rho_total = _correlation(ya, column)
            low, high = _fisher_interval(rho_total, len(ya), confidence)
            total.append((1.0 - rho_total, 1.0 - high, 1.0 - low))
        return first, total
    if estimator == "janon":
        for column in yab:
            first.append(_janon_interval(yb, column, confidence))
            closed, low, high = _janon_interval(ya, column, confidence)
            total.append((1.0 - closed, 1.0 - high, 1.0 - low))
        return first, total
    raise ValueError(
        f"asymptotic intervals are available for {ASYMPTOTIC_ESTIMATORS} only"
    )
