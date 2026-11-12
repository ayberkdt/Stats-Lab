"""Classical k-sample, blocked, variance, normality, and post hoc tests.

The rank tests here extend :mod:`.rank_tests` to several groups:
Kruskal-Wallis for independent samples (with an exact permutation law for
small designs) and Friedman for complete blocks, with Dunn's pairwise
follow-up. The sign test is the exact binomial test of the signs of paired
differences. Levene's and the Brown-Forsythe tests compare spreads by an
ANOVA of absolute deviations from the group means or medians; Bartlett's
test is the normal-theory likelihood-ratio test of equal variances and is
not robust to non-normality. Shapiro-Wilk follows Royston's algorithm
AS R94. Cochran's Q compares several matched binary outcomes. Tukey's HSD
compares all pairs of means with the studentized range distribution
(Tukey-Kramer for unequal sizes).

A p-value is the probability under the null hypothesis of a statistic at
least as extreme as observed; it is not the probability that the null is
true.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import require_finite_real, require_probability
from probstat.statistics import median, ranks

from ._hypothesis import Alternative, check_alternative, check_level, clamp_probability
from .confidence import ConfidenceInterval
from .hypothesis_tests import BinomialTestResult, binomial_test, one_way_anova
from .multiple_testing import AdjustmentMethod, MultipleTestingResult, adjust_p_values
from .sampling_distributions import chi_square_sf, standard_normal_sf
from .studentized_range import studentized_range_quantile, studentized_range_sf

_KRUSKAL_EXACT_LIMIT = 100_000


@dataclass(frozen=True, slots=True)
class ClassicalTestResult:
    """Result of a classical test with a scalar statistic.

    Args:
        statistic: Test statistic (``H``, ``Q``, ``W``, ``F``, ``T``).
        p_value: Probability under the null of a statistic at least as
            extreme as observed; not the probability that the null is true.
        df: Degrees of freedom of the reference law (empty when the law has
            none, as for Shapiro-Wilk or an exact permutation law).
        method: Test label, naming the reference law.
        exact: Whether the p-value comes from an exact null law.
        sample_sizes: Group sizes, or ``(blocks, treatments)`` for blocked
            designs, or ``(n,)``.
        effect_size: Descriptive effect size where one is standard
            (``epsilon^2`` for Kruskal-Wallis, Kendall's ``W`` for Friedman),
            otherwise ``None``.

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014], [REF-LEHMANN-ROMANO-2005, Chapter 3].

    """

    statistic: float
    p_value: float
    df: tuple[float, ...]
    method: str
    exact: bool
    sample_sizes: tuple[int, ...]
    effect_size: float | None = None

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-LEHMANN-ROMANO-2005, Chapter 3]. No equation is evaluated.

        """
        object.__setattr__(
            self, "p_value", require_probability(self.p_value, name="p_value")
        )


def _groups(
    groups: Sequence[Iterable[object]], minimum: int = 1
) -> tuple[tuple[float, ...], ...]:
    """Validate at least two groups of at least ``minimum`` observations.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    samples = tuple(
        coerce_finite_sample(group, name=f"groups[{index}]")
        for index, group in enumerate(groups)
    )
    if len(samples) < 2:
        raise ValueError("at least two groups are required")
    for index, sample in enumerate(samples):
        if len(sample) < minimum:
            raise ValueError(f"groups[{index}] needs at least {minimum} observations")
    return samples


def _tie_sum(values: Iterable[float]) -> float:
    """Return ``sum (t^3 - t)`` over groups of equal values.

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 6]. No new equation.

    """
    return float(sum(t**3 - t for t in Counter(values).values()))


def _kruskal_exact_p(
    doubled_ranks: Sequence[int], sizes: Sequence[int], observed: int, lcm: int
) -> float:
    """Return ``P(T >= T_obs)`` over all assignments of ranks to groups.

    ``T = sum_g (lcm / n_g) S_g^2`` with ``S_g`` the sum of doubled mid-ranks
    of group ``g``; ``H`` is increasing in ``T`` for a fixed tie pattern, so
    the tail of ``T`` is the exact permutation p-value of ``H``.

    References:
        [EQ-INF-KW-001], [REF-KRUSKAL-WALLIS-1952].

    """
    k = len(sizes)
    weights = [lcm // n for n in sizes]
    count = 0
    extreme = 0

    def walk(remaining: tuple[int, ...], group: int, partial: int) -> None:
        """Assign the next group's ranks and count extreme leaves.

        References:
            [EQ-INF-KW-001].

        """
        nonlocal count, extreme
        if group == k - 1:
            last = sum(doubled_ranks[i] for i in remaining)
            value = partial + weights[group] * last * last
            count += 1
            if value >= observed:
                extreme += 1
            return
        for chosen in itertools.combinations(remaining, sizes[group]):
            picked = set(chosen)
            subtotal = sum(doubled_ranks[i] for i in chosen)
            walk(
                tuple(i for i in remaining if i not in picked),
                group + 1,
                partial + weights[group] * subtotal * subtotal,
            )

    walk(tuple(range(len(doubled_ranks))), 0, 0)
    return extreme / count


def kruskal_wallis_test(
    *groups: Iterable[object],
    method: str = "auto",
    max_exact: int = _KRUSKAL_EXACT_LIMIT,
) -> ClassicalTestResult:
    """Run the Kruskal-Wallis test of equal distributions for ``k`` groups.

    ``H = [12/(N(N+1)) sum_g R_g^2/n_g - 3(N+1)] / C`` with mid-ranks of the
    pooled sample and the tie correction ``C = 1 - sum(t^3 - t)/(N^3 - N)``.
    ``method="exact"`` (and ``"auto"`` when the number of distinct
    assignments ``N!/prod n_g!`` is at most ``max_exact``) enumerates every
    assignment of the observed mid-ranks to groups, which is the exact
    permutation law given the ties; otherwise ``H`` is referred to
    chi-square with ``k - 1`` degrees of freedom.

    Args:
        *groups: At least two non-empty samples.
        method: ``"auto"``, ``"exact"``, or ``"chi-square"``.
        max_exact: Largest number of assignments enumerated by ``"auto"``.

    Returns:
        ``H``, its p-value, and ``epsilon^2 = H/(N - 1)``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If fewer than two groups, an empty group, all
            observations tied, or an unknown method is given.

    Equation:
        [EQ-INF-KW-001].

    References:
        [REF-KRUSKAL-WALLIS-1952], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 6].

    """
    samples = _groups(groups)
    if method not in ("auto", "exact", "chi-square"):
        raise ValueError("method must be 'auto', 'exact', or 'chi-square'")
    sizes = tuple(len(sample) for sample in samples)
    pooled = [value for sample in samples for value in sample]
    n = len(pooled)
    if n < 3:
        raise ValueError("at least three observations are required")
    pooled_ranks = ranks(pooled)
    correction = 1.0 - _tie_sum(pooled) / (n**3 - n)
    if correction <= 0.0:
        raise ValueError("all observations are tied; the test is undefined")
    sums: list[float] = []
    start = 0
    for size in sizes:
        sums.append(math.fsum(pooled_ranks[start : start + size]))
        start += size
    raw = 12.0 / (n * (n + 1)) * math.fsum(
        s * s / m for s, m in zip(sums, sizes, strict=True)
    ) - 3.0 * (n + 1)
    statistic = max(0.0, raw / correction)
    assignments = math.factorial(n)
    for size in sizes:
        assignments //= math.factorial(size)
    use_exact = method == "exact" or (method == "auto" and assignments <= max_exact)
    k = len(sizes)
    effect = statistic / (n - 1)
    if use_exact:
        doubled = [round(2.0 * r) for r in pooled_ranks]
        lcm = math.lcm(*sizes)
        observed = 0
        start = 0
        for size in sizes:
            group_sum = sum(doubled[start : start + size])
            observed += (lcm // size) * group_sum * group_sum
            start += size
        p_value = _kruskal_exact_p(doubled, sizes, observed, lcm)
        return ClassicalTestResult(
            statistic,
            clamp_probability(p_value),
            (),
            f"Kruskal-Wallis test (exact permutation law, {assignments} assignments)",
            True,
            sizes,
            effect,
        )
    return ClassicalTestResult(
        statistic,
        chi_square_sf(statistic, k - 1),
        (float(k - 1),),
        "Kruskal-Wallis test (chi-square approximation, tie-corrected)",
        False,
        sizes,
        effect,
    )


def _blocks(
    blocks: Iterable[Iterable[object]], name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate a rectangular table of at least two rows and two columns.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(
        coerce_finite_sample(row, name=f"{name}[{index}]")
        for index, row in enumerate(blocks)
    )
    if len(rows) < 2:
        raise ValueError(f"{name} needs at least two rows")
    width = len(rows[0])
    if width < 2:
        raise ValueError(f"{name} needs at least two columns")
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must have equal lengths")
    return rows


def friedman_test(blocks: Iterable[Iterable[object]]) -> ClassicalTestResult:
    """Run the Friedman test of equal treatment effects in complete blocks.

    Each row (block) holds one observation per treatment (column). Values
    are mid-ranked within each block and
    ``Q = [12/(n k (k+1)) sum_j R_j^2 - 3 n (k+1)] / C`` with
    ``C = 1 - sum_blocks sum(t^3 - t) / (n (k^3 - k))``, referred to
    chi-square with ``k - 1`` degrees of freedom.

    Args:
        blocks: ``n >= 2`` rows of ``k >= 2`` measurements.

    Returns:
        ``Q``, its chi-square p-value, and Kendall's ``W = Q/(n(k-1))``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the table is not rectangular, too small, or every
            block is completely tied.

    Equation:
        [EQ-INF-FRIEDMAN-001].

    References:
        [REF-FRIEDMAN-1937], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 7].

    """
    rows = _blocks(blocks, "blocks")
    n, k = len(rows), len(rows[0])
    column_sums = [0.0] * k
    ties = 0.0
    for row in rows:
        for j, rank in enumerate(ranks(row)):
            column_sums[j] += rank
        ties += _tie_sum(row)
    correction = 1.0 - ties / (n * (k**3 - k))
    if correction <= 0.0:
        raise ValueError("every block is completely tied; the test is undefined")
    raw = 12.0 / (n * k * (k + 1)) * math.fsum(r * r for r in column_sums) - 3.0 * n * (
        k + 1
    )
    statistic = max(0.0, raw / correction)
    return ClassicalTestResult(
        statistic,
        chi_square_sf(statistic, k - 1),
        (float(k - 1),),
        "Friedman test (chi-square approximation, tie-corrected)",
        False,
        (n, k),
        statistic / (n * (k - 1)),
    )


def sign_test(
    first: Iterable[object],
    second: Iterable[object] | None = None,
    *,
    null_value: float = 0.0,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> BinomialTestResult:
    """Run the exact sign test for a median (or a median paired difference).

    Differences ``d = first - second - null_value`` (or
    ``first - null_value``) equal to zero are discarded; under the null the
    number of positive differences among the ``m`` that remain is
    ``Binomial(m, 1/2)``. The test needs no symmetry, unlike the Wilcoxon
    signed-rank test.

    Args:
        first: First measurements, or the only sample.
        second: Optional paired second measurements of equal length.
        null_value: Hypothesized median of the differences.
        alternative: ``"greater"`` means differences tend to exceed the null.
        confidence: Level of the Clopper-Pearson interval for ``P(d > 0)``.

    Returns:
        :class:`BinomialTestResult` with ``successes`` the positive count and
        ``trials`` the number of non-zero differences.

    Raises:
        TypeError: If observations are not real.
        ValueError: If lengths differ or every difference is zero.

    Equation:
        [EQ-INF-SIGN-001], [EQ-INF-BINOM-001].

    References:
        [REF-DIXON-MOOD-1946], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 3].

    """
    a = coerce_finite_sample(first, name="first")
    if second is None:
        raw = list(a)
    else:
        b = coerce_finite_sample(second, name="second")
        if len(a) != len(b):
            raise ValueError("paired samples must have equal lengths")
        raw = [x - y for x, y in zip(a, b, strict=True)]
    shift = require_finite_real(null_value, name="null_value")
    differences = [d - shift for d in raw if d - shift != 0.0]
    if not differences:
        raise ValueError("all differences are zero; the test is undefined")
    positive = sum(1 for d in differences if d > 0.0)
    result = binomial_test(
        positive,
        len(differences),
        null_probability=0.5,
        alternative=check_alternative(alternative),
        confidence=confidence,
    )
    return replace(result, method="exact sign test")


def levene_test(*groups: Iterable[object], center: str = "mean") -> ClassicalTestResult:
    """Run Levene's test (or Brown-Forsythe's) of equal variances.

    The one-way ANOVA F statistic of ``z_ij = |x_ij - c_i|`` with ``c_i``
    the group mean (Levene) or median (Brown-Forsythe), referred to
    ``F(k - 1, N - k)``. The median-centred version keeps its level for
    skewed and heavy-tailed data.

    Args:
        *groups: At least two groups, each with at least two observations.
        center: ``"mean"`` (Levene) or ``"median"`` (Brown-Forsythe).

    Returns:
        ``F``, its p-value, and ``df = (k - 1, N - k)``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If a group is too small, every absolute deviation is
            constant within groups, or ``center`` is unknown.

    Equation:
        [EQ-INF-LEVENE-001], [EQ-INF-ANOVA-001].

    References:
        [REF-BROWN-FORSYTHE-1974].

    """
    samples = _groups(groups, minimum=2)
    if center not in ("mean", "median"):
        raise ValueError("center must be 'mean' or 'median'")
    label = (
        "Levene test (mean-centred)"
        if center == "mean"
        else "Brown-Forsythe test (median-centred Levene)"
    )
    centres = [
        statistics.fmean(sample) if center == "mean" else median(sample)
        for sample in samples
    ]
    deviations = [
        [abs(x - c) for x in sample] for sample, c in zip(samples, centres, strict=True)
    ]
    anova = one_way_anova(*deviations)
    return ClassicalTestResult(
        anova.statistic,
        anova.p_value,
        (float(anova.df_between), float(anova.df_within)),
        label,
        False,
        anova.group_sizes,
    )


def brown_forsythe_test(*groups: Iterable[object]) -> ClassicalTestResult:
    """Run the Brown-Forsythe test: Levene's test centred on group medians.

    Args:
        *groups: At least two groups, each with at least two observations.

    Returns:
        ``F``, its p-value, and ``df = (k - 1, N - k)``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If a group is too small or every absolute deviation is
            constant within groups.

    Equation:
        [EQ-INF-LEVENE-001].

    References:
        [REF-BROWN-FORSYTHE-1974].

    """
    return levene_test(*groups, center="median")


def bartlett_test(*groups: Iterable[object]) -> ClassicalTestResult:
    """Run Bartlett's test of equal variances for normal groups.

    ``T = [(N - k) ln s_p^2 - sum (n_i - 1) ln s_i^2] / C`` with
    ``C = 1 + (sum 1/(n_i - 1) - 1/(N - k)) / (3(k - 1))``, referred to
    chi-square with ``k - 1`` degrees of freedom. Sensitive to
    non-normality: a small p-value can reflect heavy tails rather than
    unequal variances.

    Args:
        *groups: At least two groups, each with at least two observations
            and positive variance.

    Returns:
        ``T``, its p-value, and ``df = (k - 1,)``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If a group is too small or has zero variance.

    Equation:
        [EQ-INF-BARTLETT-001].

    References:
        [REF-BARTLETT-1937].

    """
    samples = _groups(groups, minimum=2)
    k = len(samples)
    dof = [len(sample) - 1 for sample in samples]
    variances = [statistics.variance(sample) for sample in samples]
    if any(v <= 0.0 for v in variances):
        raise ValueError("every group needs a positive variance")
    total_dof = sum(dof)
    pooled = math.fsum(d * v for d, v in zip(dof, variances, strict=True)) / total_dof
    numerator = total_dof * math.log(pooled) - math.fsum(
        d * math.log(v) for d, v in zip(dof, variances, strict=True)
    )
    correction = 1.0 + (math.fsum(1.0 / d for d in dof) - 1.0 / total_dof) / (
        3.0 * (k - 1)
    )
    statistic = max(0.0, numerator / correction)
    return ClassicalTestResult(
        statistic,
        chi_square_sf(statistic, k - 1),
        (float(k - 1),),
        "Bartlett test (chi-square approximation)",
        False,
        tuple(len(sample) for sample in samples),
    )


_SW_C1 = (0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
_SW_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_SW_C3 = (0.544, -0.39978, 0.025054, -6.714e-4)
_SW_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_SW_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_SW_C6 = (-0.4803, -0.082676, 0.0030302)
_SW_G = (-2.273, 0.459)
_SW_SMALL = 1e-19


def _poly(coefficients: Sequence[float], x: float) -> float:
    """Evaluate ``c_0 + c_1 x + ... + c_m x^m`` by Horner's rule.

    References:
        [EQ-INF-SW-001], [REF-ROYSTON-1995].

    """
    value = 0.0
    for c in reversed(coefficients):
        value = value * x + c
    return value


def shapiro_wilk_coefficients(size: int) -> tuple[float, ...]:
    """Return Royston's approximate Shapiro-Wilk coefficients ``a_1 .. a_{n/2}``.

    With ``m_i = Phi^-1((i - 3/8)/(n + 1/4))`` and ``u = 1/sqrt(n)``, the two
    largest coefficients are polynomial corrections
    ``a_1 = -m_1/||m|| + poly(C1, u)`` and (``n > 5``)
    ``a_2 = -m_2/||m|| + poly(C2, u)``; the others are ``-m_i/phi`` with
    ``phi`` chosen so that ``sum a_i^2 = 1``. ``n = 3`` gives
    ``a_1 = sqrt(1/2)``. The polynomial coefficients are those of
    algorithm AS R94.

    Args:
        size: Sample size ``3 <= n <= 5000``.

    Returns:
        The ``floor(n/2)`` positive coefficients, largest first.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size`` is outside ``[3, 5000]``.

    Equation:
        [EQ-INF-SW-001].

    References:
        [REF-ROYSTON-1992], [REF-ROYSTON-1995].

    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if not 3 <= size <= 5000:
        raise ValueError("Shapiro-Wilk needs 3 <= n <= 5000")
    n = size
    half = n // 2
    if n == 3:
        return (math.sqrt(0.5),)
    normal = NormalDist()
    m = [normal.inv_cdf((i - 0.375) / (n + 0.25)) for i in range(1, half + 1)]
    summ2 = 2.0 * math.fsum(v * v for v in m)
    ssumm2 = math.sqrt(summ2)
    rsn = 1.0 / math.sqrt(n)
    a1 = _poly(_SW_C1, rsn) - m[0] / ssumm2
    a = [0.0] * half
    a[0] = a1
    if n > 5:
        first = 2
        a2 = -m[1] / ssumm2 + _poly(_SW_C2, rsn)
        fac = math.sqrt(
            (summ2 - 2.0 * m[0] ** 2 - 2.0 * m[1] ** 2)
            / (1.0 - 2.0 * a1**2 - 2.0 * a2**2)
        )
        a[1] = a2
    else:
        first = 1
        fac = math.sqrt((summ2 - 2.0 * m[0] ** 2) / (1.0 - 2.0 * a1**2))
    for i in range(first, half):
        a[i] = -m[i] / fac
    return tuple(a)


def shapiro_wilk_test(values: Iterable[object]) -> ClassicalTestResult:
    """Run the Shapiro-Wilk test of normality (Royston's algorithm AS R94).

    ``W`` is the squared correlation between the ordered sample and the
    antisymmetric coefficient vector of :func:`shapiro_wilk_coefficients`,
    computed as ``1 - w1`` with ``w1`` formed without cancellation. The
    p-value is exact for ``n = 3``
    (``(6/pi)(asin(sqrt W) - pi/3)``); for ``4 <= n <= 11`` the statistic
    ``-ln(gamma - ln(1 - W))``, ``gamma = -2.273 + 0.459 n``, and for
    ``n >= 12`` the statistic ``ln(1 - W)`` are referred to normal laws
    with polynomial mean and log standard deviation (AS R94).

    Args:
        values: Sample of size ``3 <= n <= 5000`` with positive range.

    Returns:
        ``W`` and its upper-tail p-value (small ``W`` rejects normality).

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the size is outside ``[3, 5000]`` or all values are
            equal.

    Equation:
        [EQ-INF-SW-001].

    References:
        [REF-SHAPIRO-WILK-1965], [REF-ROYSTON-1992], [REF-ROYSTON-1995].

    """
    sample = sorted(coerce_finite_sample(values, name="values"))
    n = len(sample)
    coefficients = shapiro_wilk_coefficients(n)
    spread = sample[-1] - sample[0]
    if spread <= 0.0:
        raise ValueError("all observations are equal; W is undefined")
    full = [0.0] * n
    for i, value in enumerate(coefficients):
        full[i] = -value
        full[n - 1 - i] = value
    scaled = [x / spread for x in sample]
    mean_a = math.fsum(full) / n
    mean_x = math.fsum(scaled) / n
    da = [v - mean_a for v in full]
    dx = [v - mean_x for v in scaled]
    ssa = math.fsum(v * v for v in da)
    ssx = math.fsum(v * v for v in dx)
    sax = math.fsum(u * v for u, v in zip(da, dx, strict=True))
    root = math.sqrt(ssa * ssx)
    w1 = (root - sax) * (root + sax) / (ssa * ssx)
    statistic = 1.0 - w1
    if n == 3:
        p_value = max(
            0.0, 6.0 / math.pi * (math.asin(math.sqrt(statistic)) - math.pi / 3.0)
        )
    else:
        y = math.log(w1)
        if n <= 11:
            gamma = _poly(_SW_G, n)
            if y >= gamma:
                p_value = _SW_SMALL
            else:
                y = -math.log(gamma - y)
                mean = _poly(_SW_C3, n)
                sd = math.exp(_poly(_SW_C4, n))
                p_value = standard_normal_sf((y - mean) / sd)
        else:
            log_n = math.log(n)
            mean = _poly(_SW_C5, log_n)
            sd = math.exp(_poly(_SW_C6, log_n))
            p_value = standard_normal_sf((y - mean) / sd)
    return ClassicalTestResult(
        statistic,
        clamp_probability(p_value),
        (),
        "Shapiro-Wilk test (Royston AS R94)",
        n == 3,
        (n,),
    )


def cochran_q_test(table: Iterable[Iterable[object]]) -> ClassicalTestResult:
    """Run Cochran's Q test of equal success rates in matched binary data.

    Rows are subjects and columns are ``k`` conditions with entries 0 or 1.
    With column totals ``C_j``, row totals ``R_i``, and ``T = sum C_j``,
    ``Q = (k - 1)(k sum C_j^2 - T^2)/(k T - sum R_i^2)``, referred to
    chi-square with ``k - 1`` degrees of freedom. Rows with all zeros or all
    ones carry no information and cancel. For ``k = 2`` it equals McNemar's
    statistic without continuity correction.

    Args:
        table: At least two rows of ``k >= 2`` binary outcomes.

    Returns:
        ``Q``, its p-value, and ``sample_sizes = (subjects, conditions)``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If an entry is not 0 or 1, the table is not rectangular,
            or no row is mixed.

    Equation:
        [EQ-INF-COCHRANQ-001].

    References:
        [REF-COCHRAN-1950].

    """
    rows = _blocks(table, "table")
    if any(value not in (0.0, 1.0) for row in rows for value in row):
        raise ValueError("table entries must be 0 or 1")
    n, k = len(rows), len(rows[0])
    column_totals = [sum(row[j] for row in rows) for j in range(k)]
    row_totals = [sum(row) for row in rows]
    total = sum(row_totals)
    denominator = k * total - sum(r * r for r in row_totals)
    if denominator <= 0.0:
        raise ValueError("no row has both outcomes; Q is undefined")
    statistic = (
        (k - 1) * (k * sum(c * c for c in column_totals) - total * total) / denominator
    )
    return ClassicalTestResult(
        statistic,
        chi_square_sf(statistic, k - 1),
        (float(k - 1),),
        "Cochran Q test (chi-square approximation)",
        False,
        (n, k),
    )


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """One pairwise comparison of a post hoc procedure.

    Args:
        first: Index of the first group.
        second: Index of the second group.
        difference: Mean difference (Tukey) or mean-rank difference (Dunn),
            first minus second.
        standard_error: Standard error of ``difference``.
        statistic: Studentized range ``q`` (Tukey) or ``z`` (Dunn).
        p_value: Unadjusted p-value (Dunn) or the family-wise p-value from the
            studentized range (Tukey).
        adjusted_p_value: p-value adjusted for the family of comparisons.
        rejected: Whether the adjusted p-value is at most ``alpha``.
        confidence_interval: Simultaneous interval for the difference
            (Tukey), or ``None``.

    References:
        [REF-HOCHBERG-TAMHANE-1987], [REF-DUNN-1964].

    """

    first: int
    second: int
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    adjusted_p_value: float
    rejected: bool
    confidence_interval: ConfidenceInterval | None


@dataclass(frozen=True, slots=True)
class PairwiseComparisonResult:
    """All pairwise comparisons of a post hoc procedure.

    Args:
        comparisons: One entry per pair ``i < j`` in lexicographic order.
        method: Procedure label.
        alpha: Family-wise (or FDR) level of the rejection decisions.
        adjustment: Multiple-testing adjustment (Dunn), or ``None`` (Tukey,
            whose p-values are already simultaneous).
        df: Error degrees of freedom (Tukey), or ``None``.

    References:
        [REF-HOCHBERG-TAMHANE-1987], [REF-DUNN-1964].

    """

    comparisons: tuple[PairwiseComparison, ...]
    method: str
    alpha: float
    adjustment: MultipleTestingResult | None
    df: float | None

    def comparison(self, first: int, second: int) -> PairwiseComparison:
        """Return the comparison of groups ``first`` and ``second``.

        Args:
            first: Group index.
            second: Another group index.

        Returns:
            The matching comparison (stored with ``first < second``).

        Raises:
            KeyError: If no such pair exists.

        References:
            [REF-HOCHBERG-TAMHANE-1987]. No equation is evaluated.

        """
        low, high = min(first, second), max(first, second)
        for item in self.comparisons:
            if (item.first, item.second) == (low, high):
                return item
        raise KeyError((first, second))


def tukey_hsd_test(
    *groups: Iterable[object], confidence: float = 0.95
) -> PairwiseComparisonResult:
    """Compare all pairs of group means by Tukey's HSD (Tukey-Kramer).

    With the pooled within-group mean square ``MSE`` on ``nu = N - k``
    degrees of freedom, each pair has
    ``q_ij = |x_bar_i - x_bar_j| / sqrt(MSE/2 (1/n_i + 1/n_j))``, p-value
    ``P(Q_{k, nu} > q_ij)`` and simultaneous interval
    ``x_bar_i - x_bar_j +- q_{1-alpha; k, nu} sqrt(MSE/2 (1/n_i + 1/n_j))``.
    For equal sizes the family-wise error rate is exactly ``alpha`` under
    normal errors with a common variance; for unequal sizes the
    Tukey-Kramer intervals are conservative.

    Args:
        *groups: At least two groups with ``N > k`` observations in total.
        confidence: Simultaneous confidence level ``1 - alpha``.

    Returns:
        :class:`PairwiseComparisonResult` with ``df = N - k``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If too few observations are given or the within-group
            variation is zero.

    Equation:
        [EQ-INF-TUKEY-001], [EQ-INF-SRANGE-001].

    References:
        [REF-KRAMER-1956], [REF-HOCHBERG-TAMHANE-1987].

    """
    samples = _groups(groups)
    level = check_level(confidence, name="confidence")
    k = len(samples)
    total = sum(len(sample) for sample in samples)
    if total <= k:
        raise ValueError("total observations must exceed the number of groups")
    means = [statistics.fmean(sample) for sample in samples]
    within = math.fsum(
        (x - mean) ** 2
        for sample, mean in zip(samples, means, strict=True)
        for x in sample
    )
    if within <= 0.0:
        raise ValueError("within-group variation is zero")
    df = total - k
    mse = within / df
    critical = studentized_range_quantile(level, k, df)
    alpha = 1.0 - level
    comparisons: list[PairwiseComparison] = []
    for i, j in itertools.combinations(range(k), 2):
        difference = means[i] - means[j]
        error = math.sqrt(0.5 * mse * (1.0 / len(samples[i]) + 1.0 / len(samples[j])))
        q = abs(difference) / error
        p_value = clamp_probability(studentized_range_sf(q, k, df))
        comparisons.append(
            PairwiseComparison(
                i,
                j,
                difference,
                error,
                q,
                p_value,
                p_value,
                p_value <= alpha,
                ConfidenceInterval(
                    difference - critical * error,
                    difference + critical * error,
                    level,
                    "Tukey-Kramer simultaneous interval",
                ),
            )
        )
    return PairwiseComparisonResult(
        tuple(comparisons), "Tukey HSD (Tukey-Kramer)", alpha, None, float(df)
    )


def dunn_test(
    *groups: Iterable[object],
    adjustment: AdjustmentMethod = "holm",
    alpha: float = 0.05,
) -> PairwiseComparisonResult:
    """Compare all pairs of groups after Kruskal-Wallis by Dunn's test.

    With pooled mid-ranks, mean ranks ``R_bar_i`` and
    ``z_ij = (R_bar_i - R_bar_j) / sqrt((N(N+1)/12 - sum(t^3 - t)/(12(N - 1)))
    (1/n_i + 1/n_j))``, two-sided normal p-values are adjusted for the
    ``k(k-1)/2`` comparisons by :func:`adjust_p_values`.

    Args:
        *groups: At least two non-empty groups.
        adjustment: ``"bonferroni"``, ``"holm"``, ``"bh"``, or ``"by"``.
        alpha: Level of the rejection decisions.

    Returns:
        :class:`PairwiseComparisonResult` with the adjustment record.

    Raises:
        TypeError: If observations are not real.
        ValueError: If fewer than two groups are given, all observations
            are tied, or an option is invalid.

    Equation:
        [EQ-INF-DUNN-001], [EQ-INF-MT-001], [EQ-INF-MT-002].

    References:
        [REF-DUNN-1964], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 6].

    """
    samples = _groups(groups)
    level = check_level(alpha, name="alpha")
    pooled = [value for sample in samples for value in sample]
    n = len(pooled)
    pooled_ranks = ranks(pooled)
    variance = (
        n * (n + 1) / 12.0 - _tie_sum(pooled) / (12.0 * (n - 1)) if n > 1 else 0.0
    )
    if variance <= 0.0:
        raise ValueError("all observations are tied; the test is undefined")
    mean_ranks: list[float] = []
    start = 0
    for sample in samples:
        mean_ranks.append(
            math.fsum(pooled_ranks[start : start + len(sample)]) / len(sample)
        )
        start += len(sample)
    pairs = list(itertools.combinations(range(len(samples)), 2))
    rows: list[tuple[int, int, float, float, float, float]] = []
    for i, j in pairs:
        difference = mean_ranks[i] - mean_ranks[j]
        error = math.sqrt(variance * (1.0 / len(samples[i]) + 1.0 / len(samples[j])))
        z = difference / error
        rows.append(
            (i, j, difference, error, z, min(1.0, 2.0 * standard_normal_sf(abs(z))))
        )
    adjusted = adjust_p_values([row[5] for row in rows], method=adjustment, alpha=level)
    comparisons = tuple(
        PairwiseComparison(i, j, d, e, z, p, q, flag, None)
        for (i, j, d, e, z, p), q, flag in zip(
            rows, adjusted.adjusted_p_values, adjusted.rejected, strict=True
        )
    )
    return PairwiseComparisonResult(
        comparisons, f"Dunn test ({adjusted.method} adjustment)", level, adjusted, None
    )
