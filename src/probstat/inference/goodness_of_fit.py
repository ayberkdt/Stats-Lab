"""One-sample empirical-distribution goodness-of-fit diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import cast

from probstat.foundations import (
    CumulativeDistribution,
    SurvivalFunction,
    coerce_finite_sample,
)
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)

_KS_EXACT_MAX_N = 100
_KS_SMIRNOV_SWITCH = 1e-7
_KS_EXACT_LABEL = "exact finite-n one-sample KS"
_KS_ASYMPTOTIC_LABEL = "Stephens-corrected asymptotic one-sample KS"


@dataclass(frozen=True, slots=True)
class KolmogorovSmirnovResult:
    """One-sample Kolmogorov-Smirnov statistic and null p-value.

    Args:
        statistic: Supremum empirical-CDF distance in ``[0, 1]``.
        p_value: Null-tail probability ``P(D_n >= d)`` in ``[0, 1]``; exact
            for ``n <= 100`` and asymptotic beyond.
        sample_size: Positive number of observations.
        method: Label identifying the p-value computation.

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test],
        [REF-NOE-1972].

    """

    statistic: float
    p_value: float
    sample_size: int
    method: str = _KS_ASYMPTOTIC_LABEL

    def __post_init__(self) -> None:
        """Validate probability-valued fields, sample size, and method label.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a field has an invalid domain.

        References:
            [EQ-INF-GOF-KS-001], [REF-NIST-GOF].

        """
        statistic = require_probability(self.statistic, name="statistic")
        p_value = require_probability(self.p_value, name="p_value")
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        if sample_size == 0:
            raise ValueError("sample_size must be positive")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        if not self.method.strip():
            raise ValueError("method must not be empty")
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "sample_size", sample_size)


@dataclass(frozen=True, slots=True)
class AndersonDarlingResult:
    """One-sample Anderson-Darling statistic for a fully specified CDF.

    Critical values and p-values depend on the null family and whether its
    parameters were estimated. This generic result intentionally reports no
    universal p-value.

    Args:
        statistic: Non-negative tail-weighted discrepancy, possibly infinite.
        sample_size: Positive number of observations.

    References:
        [REF-NIST-GOF, Anderson-Darling Test].

    """

    statistic: float
    sample_size: int

    def __post_init__(self) -> None:
        """Validate non-negative discrepancy and positive sample size.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the statistic is NaN/negative or sample size is zero.

        References:
            [EQ-INF-GOF-AD-001], [REF-NIST-GOF].

        """
        if isinstance(self.statistic, bool) or not isinstance(self.statistic, Real):
            raise TypeError("statistic must be a real number")
        statistic = float(self.statistic)
        if math.isnan(statistic) or statistic < 0.0:
            raise ValueError("statistic must be non-negative and not NaN")
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        if sample_size == 0:
            raise ValueError("sample_size must be positive")
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "sample_size", sample_size)


def kolmogorov_smirnov_test(
    values: Iterable[object],
    distribution: CumulativeDistribution,
) -> KolmogorovSmirnovResult:
    """Run a one-sample KS test against a fully specified continuous CDF.

    For ``n <= 100`` the p-value ``P(D_n >= d)`` is the exact finite-sample
    value from :func:`kolmogorov_exact_sf`. The asymptotic Kolmogorov law
    with Stephens' correction is used only for larger samples; it would give
    ``p >= 0.097`` for every single observation, whereas the exact value for
    ``n = 1`` is ``2 (1 - d)`` when ``d >= 1/2``. If distribution parameters
    were fitted from the same sample, the ordinary null calibration is
    invalid; use a parametric bootstrap or a family-specific correction
    instead.

    Args:
        values: Non-empty iterable of finite real observations.
        distribution: Object exposing a valid cumulative distribution method.

    Returns:
        KS statistic, p-value, sample size, and method label.

    Raises:
        TypeError: If inputs violate the iterable or CDF protocol.
        ValueError: If observations or returned probabilities are invalid.

    Equation:
        [EQ-INF-GOF-KS-001], [EQ-INF-GOF-KS-002], [EQ-INF-GOF-KS-003],
        [EQ-INF-GOF-KS-004].

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test],
        [REF-NOE-1972], [REF-BIRNBAUM-TINGEY-1951],
        [REF-MARSAGLIA-TSANG-WANG-2003].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    count = len(sample)
    lower_gap = 0.0
    upper_gap = 0.0
    for rank, value in enumerate(sample, start=1):
        probability = _cdf_probability(distribution, value)
        lower_gap = max(lower_gap, rank / count - probability)
        upper_gap = max(upper_gap, probability - (rank - 1) / count)
    statistic = max(lower_gap, upper_gap)
    if count <= _KS_EXACT_MAX_N:
        return KolmogorovSmirnovResult(
            statistic,
            kolmogorov_exact_sf(count, statistic),
            count,
            _KS_EXACT_LABEL,
        )
    root_count = math.sqrt(count)
    corrected = (root_count + 0.12 + 0.11 / root_count) * statistic
    return KolmogorovSmirnovResult(
        statistic,
        _kolmogorov_survival(corrected),
        count,
    )


def kolmogorov_exact_sf(count: int, statistic: float) -> float:
    """Return the exact two-sided one-sample KS tail ``P(D_n >= d)``.

    ``D_n`` is the statistic of ``n`` independent ``Uniform(0, 1)`` values
    (equivalently of any continuous, fully specified null). With the exact
    one-sided tail ``p_+ = P(D_n^+ >= d)`` of
    :func:`smirnov_one_sided_sf`, the value is ``2 p_+`` when ``d >= 1/2``
    (the events ``D_n^+ >= d`` and ``D_n^- >= d`` are then disjoint) and when
    ``p_+ <= 1e-7`` (the first event is decreasing and the second increasing
    in the observations, so by the Harris-FKG inequality their overlap is at
    most ``p_+^2 <= 1e-14``). Otherwise ``1 - P(D_n < d)`` from
    :func:`_kolmogorov_noe_cdf` is returned, accurate to about ``1e-14`` in
    absolute terms (the regression tests compare it with the
    Marsaglia-Tsang-Wang matrix formula evaluated in exact rationals).

    Args:
        count: Positive sample size ``n`` (practical up to a few hundred).
        statistic: Observed ``d`` in ``[0, 1]``.

    Returns:
        Exact null tail probability in ``[0, 1]``.

    Raises:
        ValueError: If ``count`` is not positive or ``statistic`` is NaN.

    Equation:
        [EQ-INF-GOF-KS-003], [EQ-INF-GOF-KS-004].

    References:
        [REF-NOE-1972], [REF-BIRNBAUM-TINGEY-1951],
        [REF-ESARY-PROSCHAN-WALKUP-1967], [REF-MARSAGLIA-TSANG-WANG-2003].

    """
    if count < 1:
        raise ValueError("count must be positive")
    if statistic != statistic:
        raise ValueError("statistic must not be NaN")
    if statistic <= 0.5 / count:
        return 1.0
    if statistic >= 1.0:
        return 0.0
    one_sided = smirnov_one_sided_sf(count, statistic)
    if statistic >= 0.5 or one_sided <= _KS_SMIRNOV_SWITCH:
        return min(1.0, 2.0 * one_sided)
    return min(1.0, max(0.0, 1.0 - _kolmogorov_noe_cdf(count, statistic)))


def smirnov_one_sided_sf(count: int, statistic: float) -> float:
    """Return the exact one-sided tail ``P(D_n^+ >= d)`` (Birnbaum-Tingey).

    ``P(D_n^+ >= d) = d sum_(j=0)^(floor(n(1-d))) C(n, j) (1 - d - j/n)^(n-j)
    (d + j/n)^(j-1)``; all terms are positive, so the sum (accumulated in
    log space) keeps relative accuracy far into the tail.

    Args:
        count: Positive sample size ``n``.
        statistic: Observed ``d`` in ``[0, 1]``.

    Returns:
        One-sided tail probability in ``[0, 1]``.

    Equation:
        [EQ-INF-GOF-KS-003].

    References:
        [REF-BIRNBAUM-TINGEY-1951].

    """
    if statistic <= 0.0:
        return 1.0
    if statistic >= 1.0:
        return 0.0
    n = count
    log_terms: list[float] = []
    for j in range(n + 1):
        base = 1.0 - statistic - j / n
        if base < 0.0 or (base == 0.0 and j < n):
            break
        log_base_part = 0.0 if base == 0.0 else (n - j) * math.log(base)
        log_terms.append(
            math.lgamma(n + 1.0)
            - math.lgamma(j + 1.0)
            - math.lgamma(n - j + 1.0)
            + log_base_part
            + (j - 1) * math.log(statistic + j / n)
        )
    peak = max(log_terms)
    total = math.fsum(math.exp(term - peak) for term in log_terms)
    return min(1.0, statistic * total * math.exp(peak))


def _kolmogorov_noe_cdf(count: int, statistic: float) -> float:
    """Return ``P(D_n < d)`` by Noé's recursion over the boundary crossings.

    ``D_n < d`` holds exactly when the counting process ``N(t)`` of the
    uniform sample satisfies ``N(i/n - d) <= i - 1`` and
    ``N((i-1)/n + d) >= i`` for every ``i``. Merging these boundary times
    ``0 = t_0 < t_1 < ... < t_M = 1``, the multinomial law of the counts in
    the gaps gives the recursion
    ``Q_(m+1)(c') = sum_(c <= c') Q_m(c) (t_(m+1) - t_m)^(c'-c)/(c'-c)!``
    restricted to the admissible counts, and ``P(D_n < d) = n! Q_M(n)``.
    All terms are non-negative; the vector is rescaled at every step and the
    scale is carried in logarithms.

    Args:
        count: Positive sample size ``n``.
        statistic: Observed ``d`` with ``1/(2n) < d < 1``.

    Returns:
        Lower-tail probability in ``[0, 1]``.

    Equation:
        [EQ-INF-GOF-KS-004].

    References:
        [REF-NOE-1972].

    """
    n = count
    bounds: dict[float, list[int]] = {}
    for index in range(1, n + 1):
        upper_time = index / n - statistic
        if 0.0 < upper_time < 1.0:
            entry = bounds.setdefault(upper_time, [0, n])
            entry[1] = min(entry[1], index - 1)
        lower_time = (index - 1) / n + statistic
        if 0.0 < lower_time < 1.0:
            entry = bounds.setdefault(lower_time, [0, n])
            entry[0] = max(entry[0], index)
    schedule = sorted(bounds.items())
    schedule.append((1.0, [n, n]))
    weights = [1.0] + [0.0] * n
    low = high = 0
    log_scale = 0.0
    previous = 0.0
    for time, (lower_count, upper_count) in schedule:
        gap = time - previous
        new_low = max(lower_count, low)
        new_high = upper_count
        if new_low > new_high:
            return 0.0
        poisson = [1.0]
        for step in range(1, new_high - low + 1):
            poisson.append(poisson[-1] * gap / step)
        updated = [0.0] * (n + 1)
        peak = 0.0
        for target in range(new_low, new_high + 1):
            value = math.fsum(
                weights[source] * poisson[target - source]
                for source in range(low, min(high, target) + 1)
            )
            updated[target] = value
            peak = max(peak, value)
        if peak == 0.0:
            return 0.0
        for target in range(new_low, new_high + 1):
            updated[target] /= peak
        log_scale += math.log(peak)
        weights, low, high = updated, new_low, new_high
        previous = time
    final = weights[n]
    if final <= 0.0:
        return 0.0
    return min(1.0, math.exp(math.lgamma(n + 1.0) + log_scale + math.log(final)))


def anderson_darling_test(
    values: Iterable[object],
    distribution: CumulativeDistribution,
) -> AndersonDarlingResult:
    """Compute the one-sample Anderson-Darling statistic.

    Args:
        values: Non-empty iterable of finite real observations.
        distribution: Fully specified continuous null CDF. A direct survival
            method is used for upper tails when available.

    Returns:
        Tail-weighted discrepancy and sample size; no universal p-value.

    Raises:
        TypeError: If inputs violate the iterable or CDF protocol.
        ValueError: If observations or returned probabilities are invalid.

    Equation:
        [EQ-INF-GOF-AD-001].

    References:
        [REF-NIST-GOF, Anderson-Darling Test].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    count = len(sample)
    weighted_logs: list[float] = []
    has_survival = isinstance(distribution, SurvivalFunction)
    for rank, lower_value in enumerate(sample, start=1):
        upper_value = sample[-rank]
        lower_probability = _cdf_probability(distribution, lower_value)
        if has_survival:
            survival = cast(SurvivalFunction, distribution)
            upper_probability = require_probability(
                survival.sf(upper_value),
                name="distribution.sf(x)",
            )
        else:
            upper_probability = 1.0 - _cdf_probability(distribution, upper_value)
        if lower_probability == 0.0 or upper_probability == 0.0:
            return AndersonDarlingResult(math.inf, count)
        weighted_logs.append(
            (2 * rank - 1)
            * (math.log(lower_probability) + math.log(upper_probability))
        )
    statistic = -count - math.fsum(weighted_logs) / count
    return AndersonDarlingResult(max(0.0, statistic), count)


def _cdf_probability(distribution: CumulativeDistribution, value: float) -> float:
    """Evaluate and validate one null-CDF probability.

    Args:
        distribution: Cumulative distribution object.
        value: Finite sample observation.

    Returns:
        Validated probability in ``[0, 1]``.

    Raises:
        TypeError: If the result is not real.
        ValueError: If the result is non-finite or outside ``[0, 1]``.

    References:
        [EQ-INF-GOF-KS-001], [REF-NIST-GOF].

    """
    return require_probability(distribution.cdf(value), name="distribution.cdf(x)")


def _kolmogorov_survival(argument: float) -> float:
    """Evaluate the limiting Kolmogorov survival function stably.

    Args:
        argument: Non-negative corrected root-sample statistic.

    Returns:
        Limiting upper-tail probability in ``[0, 1]``.

    Equation:
        [EQ-INF-GOF-KS-002].

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test].

    """
    if argument <= 0.0:
        return 1.0
    if argument < 1.18:
        factor = math.sqrt(2.0 * math.pi) / argument
        terms = (
            math.exp(
                -((2 * index - 1) ** 2)
                * math.pi**2
                / (8.0 * argument**2)
            )
            for index in range(1, 1000)
        )
        cumulative = factor * math.fsum(terms)
        return min(1.0, max(0.0, 1.0 - cumulative))
    alternating: list[float] = []
    for index in range(1, 1000):
        term = 2.0 * math.exp(-2.0 * index**2 * argument**2)
        alternating.append(term if index % 2 else -term)
        if term < 1e-16:
            break
    return min(1.0, max(0.0, math.fsum(alternating)))


_KS2_EXACT_MAX_PRODUCT = 250_000
_KS2_EXACT_LABEL = "exact two-sample KS (lattice-path count, conditional on ties)"
_KS2_ASYMPTOTIC_LABEL = "asymptotic two-sample KS (limiting Kolmogorov law)"


@dataclass(frozen=True, slots=True)
class TwoSampleKolmogorovSmirnovResult:
    """Two-sample Kolmogorov-Smirnov (Smirnov) statistic and null p-value.

    Args:
        statistic: ``D = sup_x |F_m(x) - G_n(x)|`` in ``[0, 1]``.
        p_value: Null probability of a statistic at least ``D`` under
            exchangeability of the pooled sample; exact (conditional on the
            observed tie pattern) or asymptotic as recorded by ``method``.
        first_size: Positive size ``m`` of the first sample.
        second_size: Positive size ``n`` of the second sample.
        method: Label identifying the p-value computation.
        exact: Whether the p-value was computed by exact enumeration.

    References:
        [REF-HODGES-1958], [REF-HOLLANDER-WOLFE-CHICKEN-2014].

    """

    statistic: float
    p_value: float
    first_size: int
    second_size: int
    method: str
    exact: bool

    def __post_init__(self) -> None:
        """Validate probability fields and sample sizes.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a probability or a size is outside its domain.

        References:
            [EQ-INF-GOF-KS2-001], [REF-HODGES-1958].

        """
        object.__setattr__(
            self, "statistic", require_probability(self.statistic, name="statistic")
        )
        object.__setattr__(
            self, "p_value", require_probability(self.p_value, name="p_value")
        )
        for name in ("first_size", "second_size"):
            size = require_non_negative_integer(getattr(self, name), name=name)
            if size == 0:
                raise ValueError(f"{name} must be positive")


def kolmogorov_smirnov_two_sample_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    exact: bool | None = None,
) -> TwoSampleKolmogorovSmirnovResult:
    """Test whether two samples come from the same continuous distribution.

    The statistic ``D = max_z |F_m(z) - G_n(z)|`` is evaluated at every
    distinct pooled value. It is kept as the exact integer
    ``K = m n D = max |i n - j m|`` over the ECDF step points, so the
    comparison "at least as extreme" involves no floating-point tolerance.

    The exact p-value counts the monotone lattice paths from ``(0, 0)`` to
    ``(m, n)`` (one step per pooled order statistic, right for the first
    sample, up for the second) that reach ``|i n - j m| >= K`` at a point
    where the pooled ordering is not inside a block of tied values. All
    ``C(m + n, m)`` paths are equally likely under exchangeability, so the
    ratio is exact; with ties it is exact conditionally on the observed tie
    pattern (the permutation distribution). The counts are exact Python
    integers and the final ratio is rounded once, so tiny p-values keep full
    relative accuracy. For large samples the limiting law
    ``P(sqrt(m n/(m + n)) D > t) -> 2 sum_k (-1)^(k-1) exp(-2 k^2 t^2)`` is
    used without small-sample correction.

    Args:
        first: Non-empty finite sample of size ``m``.
        second: Non-empty finite sample of size ``n``.
        exact: ``True`` forces the exact count, ``False`` the asymptotic law,
            ``None`` (default) uses the exact count when ``m n <= 250000``.

    Returns:
        Statistic, p-value, sizes, and method label.

    Raises:
        TypeError: If a value is not real or ``exact`` is not a bool/None.
        ValueError: If a sample is empty or contains a non-finite value.

    Equation:
        [EQ-INF-GOF-KS2-001], [EQ-INF-GOF-KS2-002], [EQ-INF-GOF-KS2-003].

    References:
        [REF-HODGES-1958], [REF-HOLLANDER-WOLFE-CHICKEN-2014],
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test].

    """
    a = sorted(coerce_finite_sample(first, name="first"))
    b = sorted(coerce_finite_sample(second, name="second"))
    if exact is not None and not isinstance(exact, bool):
        raise TypeError("exact must be a bool or None")
    m, n = len(a), len(b)
    pooled = sorted([(value, 0) for value in a] + [(value, 1) for value in b])
    block_ends: set[int] = set()
    largest = 0
    i = j = 0
    for position, (value, label) in enumerate(pooled):
        if label == 0:
            i += 1
        else:
            j += 1
        last = position + 1 == m + n
        if last or pooled[position + 1][0] != value:
            block_ends.add(position + 1)
            largest = max(largest, abs(i * n - j * m))
    statistic = largest / (m * n)
    use_exact = m * n <= _KS2_EXACT_MAX_PRODUCT if exact is None else exact
    if use_exact:
        p_value = _smirnov_exact_tail(m, n, largest, block_ends)
        return TwoSampleKolmogorovSmirnovResult(
            statistic, p_value, m, n, _KS2_EXACT_LABEL, True
        )
    effective = math.sqrt(m * n / (m + n))
    return TwoSampleKolmogorovSmirnovResult(
        statistic,
        _kolmogorov_survival(effective * statistic),
        m,
        n,
        _KS2_ASYMPTOTIC_LABEL,
        False,
    )


def _smirnov_exact_tail(
    m: int, n: int, threshold: int, block_ends: set[int]
) -> float:
    """Return the exact share of lattice paths that reach the threshold.

    ``inside[j]`` holds, for the current column ``i``, the number of paths
    from ``(0, 0)`` to ``(i, j)`` that never satisfied ``|i' n - j' m| >=
    threshold`` at a checked step ``i' + j'`` in ``block_ends``.

    Args:
        m: First sample size.
        n: Second sample size.
        threshold: Observed integer statistic ``K = m n D``.
        block_ends: Pooled positions at which the ECDFs are compared.

    Returns:
        ``1 - inside / C(m + n, m)`` rounded once to the nearest float.

    Equation:
        [EQ-INF-GOF-KS2-002].

    References:
        [REF-HODGES-1958].

    """
    if threshold == 0:
        return 1.0

    def allowed(row: int, column: int) -> bool:
        """Return whether the lattice point stays strictly inside the band.

        References:
            [EQ-INF-GOF-KS2-002].

        """
        if row + column not in block_ends:
            return True
        return abs(row * n - column * m) < threshold

    inside = [0] * (n + 1)
    inside[0] = 1
    for column in range(1, n + 1):
        inside[column] = inside[column - 1] if allowed(0, column) else 0
    for row in range(1, m + 1):
        inside[0] = inside[0] if allowed(row, 0) else 0
        for column in range(1, n + 1):
            total = inside[column] + inside[column - 1]
            inside[column] = total if allowed(row, column) else 0
    paths = math.comb(m + n, m)
    return float(Fraction(paths - inside[n], paths))
