"""Change-point detection: CUSUM test for a mean shift and PELT segmentation.

**CUSUM.** For ``x_1..x_n`` with partial sums of deviations
``S_k = sum_{i<=k} (x_i - x_bar)``, the statistic
``B = max_k |S_k| / (sigma_hat sqrt(n))`` converges under the null of a
constant mean to the supremum of a Brownian bridge, whose tail is the
Kolmogorov series ``P(B > b) = 2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 b^2)``
[EQ-TS-CP-001]; the arg-max estimates the location of a single shift.

**PELT.** Segmentation by penalized cost minimizes
``sum_i C(y_(tau_i+1 : tau_(i+1))) + beta m`` over the number ``m`` and the
positions of the change points [EQ-TS-CP-002]. The optimal-partitioning
recursion ``F(t) = min_tau [F(tau) + C(y_(tau+1:t)) + beta]`` is exact;
PELT (Killick, Fearnhead and Eckley 2012) discards a candidate ``tau`` once
``F(tau) + C(y_(tau+1:t)) + K > F(t)``, which never removes the optimum when
``C(a:b) + C(b+1:c) + K <= C(a:c)`` for all splits. The Gaussian costs are
``-2 log``-likelihoods, for which ``K = 0`` holds; the MBIC segment term
``log n_i`` needs ``K = -log n``. With a minimum segment length ``L`` a
pruned candidate stays usable for ``L - 1`` further steps, because the
pruning argument needs a following segment of at least ``L`` points. The
result therefore equals optimal partitioning exactly, at a cost that is
linear in ``n`` when the number of changes grows with ``n``.

Costs and penalties follow the source of R's ``changepoint`` package
(``cost_general_functions.c`` and ``penalty_decision.R``): normal mean
change with known variance ``C = sum (x - x_bar)^2 /
sigma^2``; normal variance change with known mean ``C = n (log 2 pi +
log(sum (x - mu)^2 / n) + 1)``; normal mean-and-variance change
``C = n (log 2 pi + log sigma_hat^2 + 1)``; BIC penalty
``(d + 1) log n`` and MBIC ``(d + 2) log n`` plus ``log n_i`` per segment,
with ``d = 1`` changed parameter for mean or variance and ``d = 2`` for both
[EQ-TS-CP-003].
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)

from .unit_root import newey_west_variance

Vector = tuple[float, ...]
_MODELS = {"mean": 1, "variance": 1, "meanvar": 2}
_VARIANCE_FLOOR = 1e-11


def kolmogorov_sf(value: float) -> float:
    """Return ``P(sup_t |B(t)| > b)`` for a Brownian bridge ``B``.

    Uses ``2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 b^2)`` for ``b >= 1`` and the
    Jacobi-transformed ``1 - (sqrt(2 pi)/b) sum_{j>=1} exp(-(2j-1)^2 pi^2 /
    (8 b^2))`` below, both of which converge in a few terms.

    Args:
        value: Threshold ``b``.

    Returns:
        Tail probability in ``[0, 1]``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite.

    Equation:
        [EQ-INF-GOF-KS-002], [EQ-TS-CP-001].

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test],
        [REF-PLOBERGER-KRAMER-1992].

    """
    b = require_finite_real(value, name="value")
    if b <= 0.0:
        return 1.0
    if b < 1.0:
        factor = math.sqrt(2.0 * math.pi) / b
        total = math.fsum(
            math.exp(-((2 * j - 1) ** 2) * math.pi**2 / (8.0 * b * b))
            for j in range(1, 60)
        )
        return min(1.0, max(0.0, 1.0 - factor * total))
    terms: list[float] = []
    for j in range(1, 200):
        term = 2.0 * math.exp(-2.0 * j * j * b * b)
        terms.append(term if j % 2 else -term)
        if term < 1e-300:
            break
    return min(1.0, max(0.0, math.fsum(terms)))


@dataclass(frozen=True, slots=True)
class CUSUMResult:
    """CUSUM test for a single shift in the mean.

    Args:
        statistic: ``B = max_k |S_k| / (sigma_hat sqrt(n))``.
        p_value: Asymptotic Kolmogorov tail probability of ``B``.
        location: ``k_hat``: the estimated shift is after observation
            ``k_hat`` (the first ``k_hat`` values form the first segment).
        path: ``S_k / (sigma_hat sqrt(n))`` for ``k = 1..n-1``.
        scale: ``sigma_hat`` (the square root of the long-run variance when
            ``lags > 0``).

    References:
        [REF-PAGE-1954], [REF-PLOBERGER-KRAMER-1992].

    """

    statistic: float
    p_value: float
    location: int
    path: Vector
    scale: float


def cusum_test(series: Iterable[float], *, lags: int = 0) -> CUSUMResult:
    """Test for a single change in the mean with the CUSUM of deviations.

    ``sigma_hat^2`` is the Newey-West long-run variance of ``x_t - x_bar``
    with truncation lag ``lags`` (``0`` gives the variance with divisor
    ``n``, appropriate for independent observations; use a positive lag for
    serially correlated data, whose positive autocorrelation otherwise
    inflates the false-alarm rate).

    Args:
        series: Observations ``x_1..x_n`` (``n >= 3``).
        lags: Newey-West truncation lag.

    Returns:
        Statistic, asymptotic p-value, estimated location, and the
        normalized CUSUM path.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the series is too short or constant.

    Equation:
        [EQ-TS-CP-001], [EQ-TS-KPSS-001].

    References:
        [REF-PAGE-1954], [REF-PLOBERGER-KRAMER-1992], [REF-NEWEY-WEST-1987].

    """
    data = coerce_finite_sample(series, name="series")
    n = len(data)
    if n < 3:
        raise ValueError("series must contain at least three observations")
    centre = math.fsum(data) / n
    deviations = [x - centre for x in data]
    variance = newey_west_variance(
        deviations, require_non_negative_integer(lags, name="lags")
    )
    if variance <= 0.0:
        raise ValueError("series must not be constant")
    scale = math.sqrt(variance)
    norm = scale * math.sqrt(n)
    partial = 0.0
    path: list[float] = []
    for value in deviations[:-1]:
        partial += value
        path.append(partial / norm)
    location = max(range(n - 1), key=lambda k: abs(path[k])) + 1
    statistic = abs(path[location - 1])
    return CUSUMResult(
        statistic, kolmogorov_sf(statistic), location, tuple(path), scale
    )


@dataclass(frozen=True, slots=True)
class Segment:
    """One segment of a segmentation.

    Args:
        start: First index (0-based, inclusive).
        stop: End index (exclusive), so the data are ``series[start:stop]``.
        mean: Segment mean (the known mean for the variance model).
        variance: Segment variance with divisor ``stop - start`` (the known
            variance for the mean model).

    References:
        [REF-KILLICK-2012].

    """

    start: int
    stop: int
    mean: float
    variance: float


@dataclass(frozen=True)
class ChangePointResult:
    """Optimal segmentation under a penalized Gaussian cost.

    Args:
        changepoints: ``tau_1 < ... < tau_m``: each is the number of
            observations before a change (R ``changepoint``'s ``cpts``
            without the final ``n``).
        segments: The ``m + 1`` segments.
        model: ``"mean"``, ``"variance"``, or ``"meanvar"``.
        penalty: Penalty ``beta`` per change point.
        objective: Minimized ``sum C + beta m`` (including the MBIC segment
            terms).
        min_segment_length: Minimum segment length ``L``.
        cost_evaluations: Segment costs evaluated (``n (n + 1) / 2`` for
            optimal partitioning without pruning).

    References:
        [REF-KILLICK-2012], [REF-ZHANG-SIEGMUND-2007].

    """

    changepoints: tuple[int, ...]
    segments: tuple[Segment, ...]
    model: str
    penalty: float
    objective: float
    min_segment_length: int
    cost_evaluations: int


def _median(values: list[float]) -> float:
    """Return the median of a non-empty list.

    Args:
        values: Values.

    Returns:
        The median.

    References:
        [REF-HAMPEL-1974]. No equation is evaluated.

    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def robust_noise_variance(series: Iterable[float]) -> float:
    """Estimate the noise variance from first differences by the MAD.

    ``sigma_hat = 1.4826 MAD(x_t - x_(t-1)) / sqrt(2)``: differencing removes
    piecewise-constant means except at the few change points, which the
    median absolute deviation ignores, and ``1.4826`` makes the MAD
    consistent for the normal standard deviation.

    Args:
        series: Observations (``n >= 3``).

    Returns:
        ``sigma_hat^2`` (positive).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the series is too short or the estimate is zero.

    Equation:
        [EQ-TS-CP-003].

    References:
        [REF-HAMPEL-1974].

    """
    data = coerce_finite_sample(series, name="series")
    if len(data) < 3:
        raise ValueError("series must contain at least three observations")
    differences = [data[t] - data[t - 1] for t in range(1, len(data))]
    centre = _median(differences)
    mad = _median([abs(d - centre) for d in differences])
    sigma = 1.482602218505602 * mad / math.sqrt(2.0)
    if sigma <= 0.0:
        raise ValueError("the robust noise scale is zero; pass variance explicitly")
    return sigma * sigma


def pelt(
    series: Iterable[float],
    *,
    model: str = "meanvar",
    penalty: str | float = "mbic",
    min_segment_length: int | None = None,
    variance: float | None = None,
    mean: float | None = None,
) -> ChangePointResult:
    """Find multiple change points by PELT with a penalized Gaussian cost.

    Args:
        series: Observations ``x_1..x_n``.
        model: ``"mean"`` (shifts in the mean, known variance),
            ``"variance"`` (shifts in the variance, known mean), or
            ``"meanvar"`` (shifts in both).
        penalty: ``"bic"``, ``"mbic"``, or a non-negative number ``beta``.
        min_segment_length: Minimum segment length ``L`` (default ``1`` for
            ``"mean"`` and ``2`` otherwise).
        variance: Known noise variance for ``"mean"`` (default
            :func:`robust_noise_variance`); pass ``1.0`` on standardized data
            to reproduce R ``changepoint::cpt.mean``.
        mean: Known mean for ``"variance"`` (default the sample mean).

    Returns:
        Change points, segments with their means and variances, and the
        minimized objective.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid or the series is shorter than
            ``2 L``.

    Equation:
        [EQ-TS-CP-002], [EQ-TS-CP-003].

    References:
        [REF-KILLICK-2012], [REF-KILLICK-ECKLEY-2014],
        [REF-ZHANG-SIEGMUND-2007].

    """
    data = coerce_finite_sample(series, name="series")
    if model not in _MODELS:
        raise ValueError("model must be 'mean', 'variance', or 'meanvar'")
    n = len(data)
    length = (
        (1 if model == "mean" else 2)
        if min_segment_length is None
        else require_non_negative_integer(min_segment_length, name="min_segment_length")
    )
    if length < 1 or (model != "mean" and length < 2):
        raise ValueError(
            "min_segment_length must be at least 1 (2 for variance models)"
        )
    if n < 2 * length:
        raise ValueError("series must contain at least 2 * min_segment_length values")
    changed = _MODELS[model]
    mbic = False
    if penalty == "bic":
        beta = (changed + 1) * math.log(n)
    elif penalty == "mbic":
        beta = (changed + 2) * math.log(n)
        mbic = True
    elif isinstance(penalty, str):
        raise ValueError("penalty must be 'bic', 'mbic', or a number")
    else:
        beta = require_finite_real(penalty, name="penalty")
        if beta < 0.0:
            raise ValueError("penalty must be non-negative")
    centre = math.fsum(data) / n
    overall = math.fsum((x - centre) ** 2 for x in data) / n
    if model == "mean":
        sigma2 = (
            robust_noise_variance(data)
            if variance is None
            else require_finite_real(variance, name="variance")
        )
        if sigma2 <= 0.0:
            raise ValueError("variance must be positive")
    else:
        sigma2 = 1.0
    known_mean = centre if mean is None else require_finite_real(mean, name="mean")
    shifted = [x - (known_mean if model == "variance" else centre) for x in data]
    first = [0.0] * (n + 1)
    second = [0.0] * (n + 1)
    for i, x in enumerate(shifted):
        first[i + 1] = first[i] + x
        second[i + 1] = second[i] + x * x
    floor = _VARIANCE_FLOOR * max(overall, 1e-300)
    log_two_pi = math.log(2.0 * math.pi)

    def cost(a: int, b: int) -> float:
        """Return the segment cost of ``series[a:b]``.

        References:
            [EQ-TS-CP-003], [REF-KILLICK-2012].

        """
        size = b - a
        total = first[b] - first[a]
        squares = second[b] - second[a]
        if model == "mean":
            value = max(squares - total * total / size, 0.0) / sigma2
        elif model == "variance":
            value = size * (log_two_pi + math.log(max(squares / size, floor)) + 1.0)
        else:
            spread = max((squares - total * total / size) / size, floor)
            value = size * (log_two_pi + math.log(spread) + 1.0)
        if mbic:
            value += math.log(size)
        return value

    slack = -math.log(n) if mbic else 0.0
    best = [math.inf] * (n + 1)
    best[0] = -beta
    previous = [0] * (n + 1)
    candidates: list[list[float]] = [[0, math.inf]]
    evaluations = 0
    for t in range(length, n + 1):
        newcomer = t - length
        if newcomer >= length:
            candidates.append([newcomer, math.inf])
        candidates = [item for item in candidates if item[1] >= t]
        values: list[float] = []
        winner_value = math.inf
        winner = 0
        for item in candidates:
            tau = int(item[0])
            value = best[tau] + cost(tau, t)
            evaluations += 1
            values.append(value)
            if value + beta < winner_value:
                winner_value = value + beta
                winner = tau
        best[t] = winner_value
        previous[t] = winner
        for item, value in zip(candidates, values, strict=True):
            if value + slack > winner_value and item[1] == math.inf:
                item[1] = t + length - 1
    points: list[int] = []
    end = n
    while end > 0:
        start = previous[end]
        if start > 0:
            points.append(start)
        end = start
    points.reverse()
    edges = [0, *points, n]
    segments: list[Segment] = []
    for a, b in zip(edges, edges[1:], strict=False):
        chunk = data[a:b]
        segment_mean = math.fsum(chunk) / len(chunk)
        if model == "mean":
            segment_variance = sigma2
        elif model == "variance":
            segment_mean = known_mean
            segment_variance = math.fsum((x - known_mean) ** 2 for x in chunk) / len(
                chunk
            )
        else:
            segment_variance = math.fsum((x - segment_mean) ** 2 for x in chunk) / len(
                chunk
            )
        segments.append(Segment(a, b, segment_mean, segment_variance))
    return ChangePointResult(
        changepoints=tuple(points),
        segments=tuple(segments),
        model=model,
        penalty=beta,
        objective=best[n],
        min_segment_length=length,
        cost_evaluations=evaluations,
    )
