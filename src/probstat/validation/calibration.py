"""Calibration and sharpness diagnostics for probabilistic forecasts.

A forecast is *probabilistically calibrated* when its probability integral
transform (PIT) ``u = F(y)`` is ``Uniform(0, 1)`` over many cases, and its
central ``beta`` intervals then contain the outcome a fraction ``beta`` of
the time. Calibration alone is cheap (a climatological forecast is
calibrated), so it is reported next to *sharpness*: the narrower the
calibrated forecast, the better. The functions here return the raw evidence:

* :func:`pit_values` for distribution objects and :func:`ensemble_pit` for
  ensembles (randomized over the rank, so a calibrated ensemble gives
  exactly uniform values, ties included);
* :func:`pit_histogram` with a Pearson chi-square uniformity test and simple
  shape indices (a U shape means an overconfident, too narrow forecast);
* :func:`pit_coverage` and :func:`interval_coverage` with exact
  Clopper-Pearson or Wilson intervals for the empirical coverage;
* :func:`reliability_diagram` for binary probability forecasts, with
  optional resampled consistency bands;
* :func:`sharpness_summary` for interval widths and predictive spread.

The chi-square test assumes independent cases. Forecasts that overlap in
time or space (serially dependent PIT values) make it anti-conservative.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations import (
    CumulativeDistribution,
    HasMeanAndVariance,
    QuantileFunction,
    RandomStream,
    coerce_finite_sample,
    require_finite_real,
    require_probability,
    resolve_stream,
)
from probstat.foundations.validation import require_non_negative_integer
from probstat.inference import (
    ChiSquareTestResult,
    ConfidenceInterval,
    chi_square_goodness_of_fit,
    clopper_pearson_interval,
    wilson_score_interval,
)
from probstat.statistics import median, quantile

from .scoring import _probability_forecasts, reliability_bin_edges

IntervalMethod = Literal["clopper-pearson", "wilson"]
DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)


# ---------------------------------------------------------------------------
# PIT values
# ---------------------------------------------------------------------------


def pit_values(
    distributions: Sequence[CumulativeDistribution], observations: Sequence[float]
) -> tuple[float, ...]:
    """Return ``u_i = F_i(y_i)`` for continuous forecast laws.

    Args:
        distributions: One law with a ``cdf`` per case.
        observations: Finite verifying values, one per case.

    Returns:
        PIT values in ``[0, 1]``.

    Raises:
        TypeError: If a law lacks ``cdf`` or a value is not real.
        ValueError: If lengths differ, inputs are empty, or a CDF value is
            not a probability.

    Equation:
        [EQ-VAL-CAL-PIT-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

    """
    laws = tuple(distributions)
    values = coerce_finite_sample(observations, name="observations")
    if len(laws) != len(values):
        raise ValueError("distributions and observations must have equal length")
    result: list[float] = []
    for law, y in zip(laws, values, strict=True):
        if not isinstance(law, CumulativeDistribution):
            raise TypeError("every distribution must expose cdf(x)")
        result.append(require_probability(law.cdf(y), name="distribution.cdf(y)"))
    return tuple(result)


def ensemble_pit(
    ensembles: Sequence[Sequence[float]],
    observations: Sequence[float],
    *,
    seed: int | RandomStream | None = None,
) -> tuple[float, ...]:
    """Return randomized PIT values of ensemble forecasts.

    With ``r = #{x_k < y}`` and ``s = #{x_k = y}`` among ``m`` members, the
    value is ``u = (r + V (s + 1)) / (m + 1)`` with ``V ~ Uniform(0, 1)``.
    If ``y`` and the members are exchangeable, the rank of ``y`` among the
    ``m + 1`` values (ties broken at random) is uniform on ``0..m`` and
    ``u`` is then exactly ``Uniform(0, 1)``. This is the continuous version
    of the verification-rank (Talagrand) histogram; the plain ECDF value
    ``r/m`` would never be uniform for a finite ensemble.

    Args:
        ensembles: One non-empty finite ensemble per case.
        observations: Finite verifying values, one per case.
        seed: Integer seed or stream for ``V`` (child ``"ensemble-pit"``).

    Returns:
        PIT values in ``[0, 1]``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If lengths differ or inputs are empty or non-finite.

    Equation:
        [EQ-VAL-CAL-PIT-002].

    References:
        [REF-HAMILL-2001], [REF-GNEITING-BALABDAOUI-RAFTERY-2007],
        [REF-CZADO-GNEITING-HELD-2009].

    """
    cases = tuple(ensembles)
    values = coerce_finite_sample(observations, name="observations")
    if len(cases) != len(values):
        raise ValueError("ensembles and observations must have equal length")
    generator = resolve_stream(seed).child("ensemble-pit").generator()
    result: list[float] = []
    for members, y in zip(cases, values, strict=True):
        ordered = sorted(coerce_finite_sample(members, name="ensemble"))
        below = bisect.bisect_left(ordered, y)
        ties = bisect.bisect_right(ordered, y) - below
        result.append((below + generator.random() * (ties + 1)) / (len(ordered) + 1))
    return tuple(result)


# ---------------------------------------------------------------------------
# PIT histogram
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PitHistogram:
    """Equal-width PIT histogram with a chi-square uniformity test.

    Args:
        counts: PIT values per bin (the last bin includes ``u = 1``).
        edges: ``bins + 1`` equally spaced edges on ``[0, 1]``.
        chi_square: Pearson test of the counts against equal probabilities
            (``bins - 1`` degrees of freedom); valid for independent cases.

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-HAMILL-2001].

    """

    counts: tuple[int, ...]
    edges: tuple[float, ...]
    chi_square: ChiSquareTestResult

    @property
    def sample_size(self) -> int:
        """Return the number of PIT values.

        References:
            [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

        """
        return sum(self.counts)

    @property
    def edge_ratio(self) -> float:
        """Return the outer-bin share relative to its uniform expectation.

        ``(c_1 + c_K) / (2 N / K)``: about ``1`` when calibrated, above ``1``
        for a U shape (overconfident, underdispersed forecasts), below ``1``
        for a hump (underconfident, overdispersed forecasts).

        References:
            [REF-HAMILL-2001], [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

        """
        bins = len(self.counts)
        return (self.counts[0] + self.counts[-1]) / (2.0 * self.sample_size / bins)

    @property
    def mean(self) -> float:
        """Return the mean PIT bin centre (``0.5`` when unbiased).

        Values below ``0.5`` indicate outcomes systematically below the
        forecast (an upward-biased forecast), above ``0.5`` the reverse.

        References:
            [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

        """
        centres = (
            0.5 * (a + b) for a, b in zip(self.edges, self.edges[1:], strict=False)
        )
        weighted = math.fsum(c * n for c, n in zip(centres, self.counts, strict=True))
        return weighted / self.sample_size


def pit_histogram(u_values: Sequence[float], *, bins: int = 10) -> PitHistogram:
    """Bin PIT values and test them for uniformity.

    Args:
        u_values: PIT values in ``[0, 1]``.
        bins: Number of equal-width bins (at least two).

    Returns:
        Counts, edges, and the Pearson chi-square test. The test emits
        :class:`~probstat.inference.SmallExpectedCountWarning` when
        ``N / bins < 5``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the sample is empty, a value is outside ``[0, 1]``,
            or ``bins < 2``.

    Equation:
        [EQ-VAL-CAL-HIST-001], [EQ-INF-CHI2-GOF-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-AGRESTI-2013, Chapter 1].

    """
    count = require_non_negative_integer(bins, name="bins")
    if count < 2:
        raise ValueError("bins must be at least two")
    values = tuple(
        require_probability(u, name="u_values")
        for u in coerce_finite_sample(u_values, name="u_values")
    )
    counts = [0] * count
    for u in values:
        counts[min(int(u * count), count - 1)] += 1
    edges = tuple(k / count for k in range(count + 1))
    return PitHistogram(tuple(counts), edges, chi_square_goodness_of_fit(counts))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageAtLevel:
    """Empirical coverage of central intervals at one nominal level.

    Args:
        nominal: Nominal central coverage ``beta``.
        covered: Cases whose outcome lies inside the interval.
        size: Number of cases.
        rate: ``covered / size``.
        interval: Binomial confidence interval for the true coverage rate.

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-WILSON-1927].

    """

    nominal: float
    covered: int
    size: int
    rate: float
    interval: ConfidenceInterval

    @property
    def consistent(self) -> bool:
        """Return whether the nominal level lies inside the rate's interval.

        References:
            [REF-CLOPPER-PEARSON-1934]. No equation is evaluated.

        """
        return self.interval.lower <= self.nominal <= self.interval.upper


def _coverage(
    nominal: float,
    covered: int,
    size: int,
    method: IntervalMethod,
    confidence: float,
) -> CoverageAtLevel:
    """Build one coverage record with the requested binomial interval.

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-WILSON-1927].

    """
    if method == "clopper-pearson":
        interval = clopper_pearson_interval(covered, size, confidence=confidence)
    elif method == "wilson":
        interval = wilson_score_interval(covered, size, confidence=confidence)
    else:
        raise ValueError("method must be 'clopper-pearson' or 'wilson'")
    return CoverageAtLevel(nominal, covered, size, covered / size, interval)


def _level(value: object) -> float:
    """Validate a nominal coverage strictly inside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    level = require_finite_real(value, name="level")
    if not 0.0 < level < 1.0:
        raise ValueError("levels must lie strictly inside (0, 1)")
    return level


def pit_coverage(
    u_values: Sequence[float],
    levels: Sequence[float] = DEFAULT_LEVELS,
    *,
    method: IntervalMethod = "clopper-pearson",
    confidence: float = 0.95,
) -> tuple[CoverageAtLevel, ...]:
    """Return the coverage of central intervals at several nominal levels.

    The central ``beta`` interval ``[F^-1((1-beta)/2), F^-1((1+beta)/2)]``
    contains ``y`` exactly when ``(1-beta)/2 <= u <= (1+beta)/2`` for the
    PIT value ``u``, so every level is read off one set of PIT values.

    Args:
        u_values: PIT values in ``[0, 1]``.
        levels: Nominal central coverages in ``(0, 1)``.
        method: ``"clopper-pearson"`` (exact, conservative) or ``"wilson"``.
        confidence: Level of the binomial interval for each rate.

    Returns:
        One :class:`CoverageAtLevel` per level, in the given order.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value or level is outside its domain.

    Equation:
        [EQ-VAL-CAL-COV-001], [EQ-INF-CI-CP-001], [EQ-INF-CI-WILSON-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-CLOPPER-PEARSON-1934],
        [REF-WILSON-1927].

    """
    values = tuple(
        require_probability(u, name="u_values")
        for u in coerce_finite_sample(u_values, name="u_values")
    )
    reports: list[CoverageAtLevel] = []
    for raw in levels:
        beta = _level(raw)
        low, high = 0.5 * (1.0 - beta), 0.5 * (1.0 + beta)
        covered = sum(1 for u in values if low <= u <= high)
        reports.append(_coverage(beta, covered, len(values), method, confidence))
    return tuple(reports)


def interval_coverage(
    lowers: Sequence[float],
    uppers: Sequence[float],
    observations: Sequence[float],
    *,
    nominal: float,
    method: IntervalMethod = "clopper-pearson",
    confidence: float = 0.95,
) -> CoverageAtLevel:
    """Return the empirical coverage of explicit prediction intervals.

    Args:
        lowers: Lower interval endpoints.
        uppers: Upper interval endpoints (not below the lowers).
        observations: Verifying values.
        nominal: Nominal coverage in ``(0, 1)``.
        method: ``"clopper-pearson"`` or ``"wilson"``.
        confidence: Level of the binomial interval for the rate.

    Returns:
        Coverage count, rate, and its binomial interval.

    Raises:
        TypeError: If a value is not real.
        ValueError: If lengths differ or an interval is reversed.

    Equation:
        [EQ-VAL-CAL-COV-001].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-WILSON-1927].

    """
    low = coerce_finite_sample(lowers, name="lowers")
    high = coerce_finite_sample(uppers, name="uppers")
    ys = coerce_finite_sample(observations, name="observations")
    if not len(low) == len(high) == len(ys):
        raise ValueError("lowers, uppers, and observations must have equal length")
    if any(h < lo for lo, h in zip(low, high, strict=True)):
        raise ValueError("every upper endpoint must be at least the lower one")
    covered = sum(1 for lo, h, y in zip(low, high, ys, strict=True) if lo <= y <= h)
    return _coverage(_level(nominal), covered, len(ys), method, confidence)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability diagram.

    Args:
        lower: Lower bin edge.
        upper: Upper bin edge.
        count: Forecasts in the bin.
        mean_forecast: Mean forecast probability in the bin (``nan`` if
            empty).
        observed_frequency: Event frequency in the bin (``nan`` if empty).
        consistency_lower: Lower consistency bound for the frequency under
            reliability (``nan`` when not computed or empty).
        consistency_upper: Upper consistency bound (``nan`` likewise).

    References:
        [REF-BROCKER-SMITH-2007].

    """

    lower: float
    upper: float
    count: int
    mean_forecast: float
    observed_frequency: float
    consistency_lower: float
    consistency_upper: float


@dataclass(frozen=True, slots=True)
class ReliabilityDiagram:
    """Reliability-diagram data for binary probability forecasts.

    Args:
        bins: Per-bin counts, mean forecasts, frequencies, and bands.
        consistency_level: Central level of the bands, ``None`` if absent.
        resamples: Resamples used for the bands (``0`` if absent).
        seed: Seed of the band resampling, ``None`` if absent.

    References:
        [REF-BROCKER-SMITH-2007], [REF-MURPHY-1973].

    """

    bins: tuple[ReliabilityBin, ...]
    consistency_level: float | None
    resamples: int
    seed: int | None

    @property
    def outside_bands(self) -> tuple[int, ...]:
        """Return indices of non-empty bins whose frequency leaves its band.

        References:
            [REF-BROCKER-SMITH-2007]. No equation is evaluated.

        """
        return tuple(
            index
            for index, item in enumerate(self.bins)
            if item.count
            and not math.isnan(item.consistency_lower)
            and not (
                item.consistency_lower
                <= item.observed_frequency
                <= item.consistency_upper
            )
        )


def reliability_diagram(
    probabilities: Sequence[float],
    outcomes: Sequence[object],
    *,
    bins: int | Sequence[float] = 10,
    consistency_level: float | None = None,
    resamples: int = 1000,
    seed: int | RandomStream | None = None,
) -> ReliabilityDiagram:
    """Return observed event frequencies against binned forecast probabilities.

    Bins are ``[e_k, e_(k+1))`` with the last one closed. When
    ``consistency_level`` is given, each bin gets a band for the observed
    frequency expected *if the forecasts were reliable*: outcomes are
    redrawn as ``Bernoulli(f_i)`` for the forecasts actually in the bin,
    ``resamples`` times, and the central type-7 quantiles of the resampled
    frequencies are reported. This keeps the forecasts fixed; Bröcker and
    Smith (2007) additionally resample the forecast values, which widens the
    bands slightly when a bin's forecasts are spread out.

    Args:
        probabilities: Forecast probabilities in ``[0, 1]``.
        outcomes: Event indicators (bools or 0/1).
        bins: Number of equal-width bins or explicit edges spanning ``[0, 1]``.
        consistency_level: Optional central band level in ``(0, 1)``.
        resamples: Positive number of resamples for the bands.
        seed: Integer seed or stream (child ``"reliability-consistency"``).

    Returns:
        Per-bin data and band metadata.

    Raises:
        TypeError: If a value has an invalid type.
        ValueError: If inputs are invalid.

    Equation:
        [EQ-VAL-CAL-REL-001].

    References:
        [REF-BROCKER-SMITH-2007], [REF-MURPHY-1973].

    """
    forecasts, observed = _probability_forecasts(probabilities, outcomes)
    edges = reliability_bin_edges(bins)
    last = len(edges) - 2
    members: list[list[int]] = [[] for _ in range(last + 1)]
    for index, value in enumerate(forecasts):
        members[min(bisect.bisect_right(edges, value) - 1, last)].append(index)
    level: float | None = None
    draws = 0
    seed_used: int | None = None
    generator = None
    if consistency_level is not None:
        level = _level(consistency_level)
        draws = require_non_negative_integer(resamples, name="resamples")
        if draws < 1:
            raise ValueError("resamples must be positive")
        stream = resolve_stream(seed)
        seed_used = stream.seed
        generator = stream.child("reliability-consistency").generator()
    result: list[ReliabilityBin] = []
    for k, indices in enumerate(members):
        n_k = len(indices)
        if n_k == 0:
            result.append(
                ReliabilityBin(
                    edges[k], edges[k + 1], 0, math.nan, math.nan, math.nan, math.nan
                )
            )
            continue
        f_bar = math.fsum(forecasts[i] for i in indices) / n_k
        o_bar = math.fsum(observed[i] for i in indices) / n_k
        band_low = band_high = math.nan
        if generator is not None and level is not None:
            frequencies = [
                sum(1 for i in indices if generator.random() < forecasts[i]) / n_k
                for _ in range(draws)
            ]
            band_low = quantile(frequencies, 0.5 * (1.0 - level))
            band_high = quantile(frequencies, 0.5 * (1.0 + level))
        result.append(
            ReliabilityBin(
                edges[k], edges[k + 1], n_k, f_bar, o_bar, band_low, band_high
            )
        )
    return ReliabilityDiagram(tuple(result), level, draws, seed_used)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SharpnessSummary:
    """Average spread of a set of forecasts.

    Args:
        levels: Nominal central levels.
        mean_widths: Mean central-interval width per level.
        median_widths: Median central-interval width per level.
        mean_standard_deviation: Mean predictive standard deviation
            (``nan`` if a law exposes no variance).
        size: Number of forecasts.

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

    """

    levels: tuple[float, ...]
    mean_widths: tuple[float, ...]
    median_widths: tuple[float, ...]
    mean_standard_deviation: float
    size: int


def sharpness_summary(
    forecasts: Sequence[Sequence[float] | QuantileFunction],
    levels: Sequence[float] = (0.5, 0.9),
) -> SharpnessSummary:
    """Summarize forecast spread by central-interval widths and dispersion.

    Each forecast is either an ensemble (widths from type-7 sample quantiles,
    standard deviation with ``ddof = 1``, zero for one member) or a law with
    ``quantile`` (and ``variance`` when available). Sharpness is a property
    of the forecasts alone; read it together with calibration.

    Args:
        forecasts: Ensembles or laws, one per case.
        levels: Nominal central levels in ``(0, 1)``.

    Returns:
        Mean and median widths per level and mean predictive spread.

    Raises:
        TypeError: If a forecast is neither an ensemble nor a quantile law.
        ValueError: If inputs are empty or a level is invalid.

    Equation:
        [EQ-VAL-CAL-SHARP-001], [EQ-STAT-QUANT-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-HYNDMAN-FAN-1996].

    """
    cases = tuple(forecasts)
    if not cases:
        raise ValueError("forecasts must not be empty")
    checked = tuple(_level(beta) for beta in levels)
    widths: list[list[float]] = [[] for _ in checked]
    spreads: list[float] = []
    for forecast in cases:
        if isinstance(forecast, QuantileFunction):
            for slot, beta in zip(widths, checked, strict=True):
                slot.append(
                    float(forecast.quantile(0.5 * (1.0 + beta)))
                    - float(forecast.quantile(0.5 * (1.0 - beta)))
                )
            if isinstance(forecast, HasMeanAndVariance):
                spreads.append(math.sqrt(float(forecast.variance)))
            else:
                spreads.append(math.nan)
            continue
        if isinstance(forecast, str | bytes):
            raise TypeError("a forecast must be an ensemble or a quantile law")
        members = coerce_finite_sample(forecast, name="ensemble")
        for slot, beta in zip(widths, checked, strict=True):
            slot.append(
                quantile(members, 0.5 * (1.0 + beta))
                - quantile(members, 0.5 * (1.0 - beta))
            )
        m = len(members)
        if m < 2:
            spreads.append(0.0)
        else:
            centre = math.fsum(members) / m
            spreads.append(
                math.sqrt(math.fsum((x - centre) ** 2 for x in members) / (m - 1))
            )
    mean_widths = tuple(math.fsum(slot) / len(slot) for slot in widths)
    median_widths = tuple(median(slot) for slot in widths)
    mean_sd = math.fsum(spreads) / len(spreads)
    return SharpnessSummary(checked, mean_widths, median_widths, mean_sd, len(cases))
