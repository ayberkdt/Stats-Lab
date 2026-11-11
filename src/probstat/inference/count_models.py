"""Maximum-likelihood count models with calibrated goodness of fit.

:func:`fit_poisson_counts` and :func:`fit_negative_binomial_counts` fit the
Poisson law and the negative binomial law with a real size parameter to
non-negative integer counts. The negative binomial maximum sits on the
Poisson boundary (size ``r -> infinity``) when the sample is not
over-dispersed; the fit then reports that boundary instead of a large finite
number. :func:`poisson_mean_interval` is Garwood's exact interval for the
Poisson mean, and :func:`count_goodness_of_fit` pools sparse cells and
calibrates Pearson's statistic by a parametric bootstrap that refits every
replicate, because the asymptotic chi-square reference ignores estimation
and sparse-cell effects.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from probstat.distributions import ChiSquare
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.optimization import brent_root

from .confidence import ConfidenceInterval
from .fitting import information_criteria
from .sampling_distributions import chi_square_sf

_FAMILIES = ("poisson", "negative-binomial")
_TAIL_EPSILON = 1e-15
_SIZE_FLOOR = 1e-10
_SIZE_CEILING = 1e12


@dataclass(frozen=True, slots=True)
class CountFit:
    """Maximum-likelihood fit of a count family.

    Args:
        family: ``"poisson"`` or ``"negative-binomial"``.
        parameter_names: ``("mean",)`` or ``("size", "mean")``.
        estimates: Estimates in the order of ``parameter_names``; the size is
            ``inf`` when the maximum is on the Poisson boundary.
        standard_errors: Observed-information standard errors (``nan`` where
            undefined, e.g. at the boundary).
        log_likelihood: Maximized log-likelihood.
        sample_size: Number of counted units ``n``.
        aic: ``2k - 2 log L`` with ``k = len(estimates)``.
        bic: ``k log n - 2 log L``.
        at_boundary: Whether the negative binomial maximum is the Poisson
            limit ``r -> infinity``.
        message: How the estimate was obtained.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002], [EQ-INF-IC-001],
        [EQ-INF-IC-003].

    """

    family: str
    parameter_names: tuple[str, ...]
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]
    log_likelihood: float
    sample_size: int
    aic: float
    bic: float
    at_boundary: bool
    message: str

    @property
    def params(self) -> dict[str, float]:
        """Return estimates keyed by parameter name.

        References:
            [REF-CASELLA-BERGER-2002, Section 7.2]. No equation is evaluated.

        """
        return dict(zip(self.parameter_names, self.estimates, strict=True))

    @property
    def mean(self) -> float:
        """Return the fitted mean count.

        References:
            [EQ-INF-COUNT-001], [EQ-INF-COUNT-002].

        """
        return self.estimates[-1]

    def log_pmf(self, k: int) -> float:
        """Return the fitted log probability of the count ``k``.

        Args:
            k: Count.

        Returns:
            ``log P(X = k)``, or ``-inf`` for a negative count.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-POIS-001], [EQ-INF-COUNT-002].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be an integer")
        count = int(k)
        if count < 0:
            return -math.inf
        return _log_pmf(count, self._size(), self.mean)

    def pmf(self, k: int) -> float:
        """Return the fitted probability of the count ``k``.

        Args:
            k: Count.

        Returns:
            ``P(X = k)``.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-POIS-001], [EQ-INF-COUNT-002].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        value = self.log_pmf(k)
        return 0.0 if value == -math.inf else math.exp(value)

    def expected_frequencies(self, max_count: int) -> tuple[float, ...]:
        """Return ``n P(X = k)`` for ``k = 0..max_count`` and the upper tail.

        Args:
            max_count: Largest count listed individually.

        Returns:
            ``max_count + 2`` values; the last is ``n P(X > max_count)``.

        Raises:
            TypeError: If ``max_count`` is not an integer.
            ValueError: If ``max_count`` is negative.

        Equation:
            [EQ-INF-COUNT-001], [EQ-INF-COUNT-002].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        top = require_non_negative_integer(max_count, name="max_count")
        masses = [self.pmf(k) for k in range(top + 1)]
        tail = max(0.0, 1.0 - math.fsum(masses))
        return tuple(self.sample_size * value for value in (*masses, tail))

    def _size(self) -> float:
        """Return the negative binomial size, ``inf`` for the Poisson law.

        References:
            [EQ-INF-COUNT-002]. No equation is evaluated.

        """
        return math.inf if self.family == "poisson" else self.estimates[0]


@dataclass(frozen=True, slots=True)
class CountGoodnessOfFit:
    """Pearson goodness of fit of a count model on pooled cells.

    Args:
        family: Fitted family.
        cells: ``(low, high)`` count ranges; ``high`` is ``None`` for the
            open upper cell.
        observed: Observed units per cell.
        expected: Fitted expected units per cell.
        statistic: Pearson ``X^2`` on the pooled cells.
        degrees_of_freedom: ``cells - 1 - parameters`` (may be below one).
        asymptotic_p_value: Chi-square tail at ``degrees_of_freedom``, ``nan``
            when the degrees of freedom are below one.
        bootstrap_p_value: ``(1 + #{X*^2 >= X^2}) / (1 + refits)``.
        n_bootstrap: Requested replicates.
        refits: Replicates that entered the bootstrap p-value.
        failed_refits: Replicates whose refit failed.

    References:
        [EQ-INF-CHI2-GOF-001], [EQ-INF-GOF-PB-001], [REF-COCHRAN-1954].

    """

    family: str
    cells: tuple[tuple[int, int | None], ...]
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    statistic: float
    degrees_of_freedom: int
    asymptotic_p_value: float
    bootstrap_p_value: float
    n_bootstrap: int
    refits: int
    failed_refits: int


def _counts(values: Iterable[object]) -> tuple[int, ...]:
    """Validate non-negative integer counts with a positive total.

    Args:
        values: Counts, one per unit.

    Returns:
        Tuple of counts.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative, there are fewer than two units, or
            every count is zero.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    counts = tuple(
        require_non_negative_integer(value, name=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if len(counts) < 2:
        raise ValueError("at least two counted units are required")
    if sum(counts) == 0:
        raise ValueError("at least one count must be positive")
    return counts


def _table(counts: tuple[int, ...]) -> dict[int, int]:
    """Return the frequency table ``{k: f_k}`` of a count sample.

    References:
        [EQ-INF-COUNT-001]. No equation is evaluated.

    """
    table: dict[int, int] = {}
    for count in counts:
        table[count] = table.get(count, 0) + 1
    return table


def _log_pmf(k: int, size: float, mean: float) -> float:
    """Return the Poisson (``size = inf``) or negative binomial log mass.

    Args:
        k: Non-negative count.
        size: Negative binomial size ``r > 0`` or ``inf``.
        mean: Mean ``mu > 0``.

    Returns:
        Log probability of ``k``.

    Equation:
        [EQ-DIST-POIS-001], [EQ-INF-COUNT-002].

    References:
        [REF-ROSS-2019, Chapter 4].

    """
    if math.isinf(size):
        return k * math.log(mean) - mean - math.lgamma(k + 1.0)
    return (
        math.lgamma(k + size)
        - math.lgamma(size)
        - math.lgamma(k + 1.0)
        + size * math.log(size / (size + mean))
        + k * math.log(mean / (size + mean))
    )


def _log_likelihood(table: dict[int, int], size: float, mean: float) -> float:
    """Return ``sum_k f_k log p(k)`` for a frequency table.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002].

    """
    return math.fsum(
        frequency * _log_pmf(k, size, mean) for k, frequency in table.items()
    )


def _size_score(table: dict[int, int], size: float, mean: float, n: int) -> float:
    """Return the profile score ``d l(r) / dr`` of the negative binomial.

    Args:
        table: Frequency table.
        size: Size ``r > 0``.
        mean: Sample mean (the profile maximizer of the mean).
        n: Number of units.

    Returns:
        ``sum_k f_k sum_{j<k} 1/(r+j) + n log(r / (r + mean))``.

    Equation:
        [EQ-INF-COUNT-002].

    References:
        [REF-LEVIN-REEDS-1977].

    """
    harmonic = math.fsum(
        frequency * math.fsum(1.0 / (size + j) for j in range(k))
        for k, frequency in table.items()
    )
    return harmonic + n * (math.log(size) - math.log(size + mean))


def _ml_variance(table: dict[int, int], mean: float, n: int) -> float:
    """Return the divisor-``n`` sample variance of a frequency table.

    References:
        [EQ-STAT-002]. The divisor is ``n`` (maximum-likelihood convention).

    """
    return (
        math.fsum(frequency * (k - mean) ** 2 for k, frequency in table.items()) / n
    )


def _build_fit(
    family: str,
    names: tuple[str, ...],
    estimates: tuple[float, ...],
    errors: tuple[float, ...],
    log_likelihood: float,
    n: int,
    at_boundary: bool,
    message: str,
) -> CountFit:
    """Assemble a :class:`CountFit` with its information criteria.

    References:
        [EQ-INF-IC-001], [EQ-INF-IC-003].

    """
    aic, _, bic = information_criteria(log_likelihood, len(estimates), n)
    return CountFit(
        family=family,
        parameter_names=names,
        estimates=estimates,
        standard_errors=errors,
        log_likelihood=log_likelihood,
        sample_size=n,
        aic=aic,
        bic=bic,
        at_boundary=at_boundary,
        message=message,
    )


def fit_poisson_counts(values: Iterable[object]) -> CountFit:
    """Fit a Poisson law to counts by maximum likelihood.

    The estimator is the sample mean, ``lambda_hat = sum x_i / n``, with
    observed-information standard error ``sqrt(lambda_hat / n)``.

    Args:
        values: Non-negative integer counts, one per unit (at least two units
            and a positive total).

    Returns:
        Fit with ``parameter_names == ("mean",)``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If the counts are invalid.

    Equation:
        [EQ-INF-COUNT-001], [EQ-INF-IC-001], [EQ-INF-IC-003].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2], [REF-ROSS-2019, Chapter 4].

    """
    counts = _counts(values)
    n = len(counts)
    table = _table(counts)
    mean = sum(counts) / n
    return _build_fit(
        "poisson",
        ("mean",),
        (mean,),
        (math.sqrt(mean / n),),
        _log_likelihood(table, math.inf, mean),
        n,
        False,
        "closed-form maximum-likelihood estimate (sample mean)",
    )


def fit_negative_binomial_counts(values: Iterable[object]) -> CountFit:
    """Fit a negative binomial law with real size by maximum likelihood.

    The law has mean ``mu`` and variance ``mu + mu^2 / r``. For fixed ``r``
    the likelihood is maximized by ``mu = x_bar``; the size solves the profile
    score equation by Brent's method. When the divisor-``n`` variance does
    not exceed the mean, the profile likelihood increases towards the
    Poisson limit and the fit reports ``size = inf`` with ``at_boundary``.

    Args:
        values: Non-negative integer counts, one per unit (at least two units
            and a positive total).

    Returns:
        Fit with ``parameter_names == ("size", "mean")``; ``k = 2`` parameters
        enter the information criteria even at the boundary.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If the counts are invalid.
        ArithmeticError: If the score root cannot be bracketed.

    Equation:
        [EQ-INF-COUNT-002], [EQ-INF-IC-001], [EQ-INF-IC-003].

    References:
        [REF-LEVIN-REEDS-1977], [REF-ROSS-2019, Chapter 4],
        [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    counts = _counts(values)
    n = len(counts)
    table = _table(counts)
    mean = sum(counts) / n
    variance = _ml_variance(table, mean, n)
    names = ("size", "mean")
    if variance <= mean:
        return _build_fit(
            "negative-binomial",
            names,
            (math.inf, mean),
            (math.nan, math.sqrt(mean / n)),
            _log_likelihood(table, math.inf, mean),
            n,
            True,
            "maximum on the Poisson boundary (size -> infinity): the "
            "divisor-n variance does not exceed the mean",
        )
    lower = upper = mean * mean / (variance - mean)
    while _size_score(table, lower, mean, n) <= 0.0:
        lower *= 0.5
        if lower < _SIZE_FLOOR:
            raise ArithmeticError("could not bracket the negative binomial size")
    while _size_score(table, upper, mean, n) >= 0.0:
        upper *= 2.0
        if upper > _SIZE_CEILING:
            raise ArithmeticError("could not bracket the negative binomial size")
    root = brent_root(
        lambda size: _size_score(table, size, mean, n),
        lower,
        upper,
        tolerance=1e-12 * upper,
    )
    size = root.root
    curvature = _size_curvature(table, size, mean, n)
    size_error = math.sqrt(-1.0 / curvature) if curvature < 0.0 else math.nan
    mean_error = math.sqrt((mean + mean * mean / size) / n)
    return _build_fit(
        "negative-binomial",
        names,
        (size, mean),
        (size_error, mean_error),
        _log_likelihood(table, size, mean),
        n,
        False,
        f"profile-score root by Brent's method ({root.iterations} iterations)",
    )


def _size_curvature(table: dict[int, int], size: float, mean: float, n: int) -> float:
    """Return ``d^2 l / dr^2`` of the full log-likelihood at ``mu = x_bar``.

    At the maximum the size and mean are orthogonal (the cross derivative
    vanishes at ``mu = x_bar``), so ``-1 / curvature`` is the size variance.

    Args:
        table: Frequency table.
        size: Size ``r > 0``.
        mean: Sample mean.
        n: Number of units.

    Returns:
        Second derivative with respect to ``r``.

    Equation:
        [EQ-INF-COUNT-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """
    harmonic = math.fsum(
        frequency * math.fsum(1.0 / (size + j) ** 2 for j in range(k))
        for k, frequency in table.items()
    )
    return -harmonic + n * (1.0 / size - 1.0 / (size + mean))


def poisson_mean_interval(
    values: Iterable[object], confidence: float = 0.95
) -> ConfidenceInterval:
    """Return Garwood's exact two-sided interval for a Poisson mean.

    With total ``S = sum x_i`` over ``n`` units and ``alpha = 1 - confidence``,
    ``lower = chi2_{alpha/2}(2S) / (2n)`` (zero when ``S = 0``) and
    ``upper = chi2_{1-alpha/2}(2S + 2) / (2n)``.

    Args:
        values: Non-negative integer counts, one per unit.
        confidence: Level strictly inside ``(0, 1)``.

    Returns:
        Interval for the mean count per unit.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If the counts or the level are invalid.

    Equation:
        [EQ-INF-COUNT-003].

    References:
        [REF-GARWOOD-1936].

    """
    counts = tuple(
        require_non_negative_integer(value, name=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if not counts:
        raise ValueError("at least one counted unit is required")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    total = sum(counts)
    n = len(counts)
    alpha = 1.0 - level
    lower = (
        0.0 if total == 0 else ChiSquare(2.0 * total).quantile(alpha / 2.0) / (2.0 * n)
    )
    upper = ChiSquare(2.0 * total + 2.0).quantile(1.0 - alpha / 2.0) / (2.0 * n)
    return ConfidenceInterval(lower, upper, level, "Garwood exact Poisson interval")


def _refit(family: str, counts: tuple[int, ...]) -> CountFit:
    """Refit the named family to a bootstrap replicate.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002].

    """
    if family == "poisson":
        return fit_poisson_counts(counts)
    return fit_negative_binomial_counts(counts)


def _pooled_cells(
    expected: tuple[float, ...], min_expected: float
) -> list[tuple[int, int | None]]:
    """Merge adjacent count cells until each expected frequency is large enough.

    Args:
        expected: ``n P(X = k)`` for ``k = 0..K`` followed by ``n P(X > K)``.
        min_expected: Minimum expected frequency per pooled cell.

    Returns:
        Contiguous ranges ``(low, high)``; the last is open (``high=None``).

    References:
        [REF-COCHRAN-1954]. No equation is evaluated.

    """
    top = len(expected) - 2
    groups: list[list[int]] = []
    current: list[int] = []
    running = 0.0
    for k in range(top + 2):
        current.append(k)
        running += expected[k]
        if running >= min_expected and k <= top:
            groups.append(current)
            current, running = [], 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    if len(groups) < 2:
        raise ValueError("too few expected units to form two cells")
    cells: list[tuple[int, int | None]] = []
    for index, group in enumerate(groups):
        high: int | None = None if index == len(groups) - 1 else group[-1]
        cells.append((group[0], high))
    return cells


def _cell_counts(
    table: dict[int, int], cells: list[tuple[int, int | None]]
) -> tuple[int, ...]:
    """Return observed units per pooled cell.

    References:
        [EQ-INF-CHI2-GOF-001]. No equation is evaluated.

    """
    return tuple(
        sum(
            frequency
            for k, frequency in table.items()
            if k >= low and (high is None or k <= high)
        )
        for low, high in cells
    )


def _cell_expectations(
    fit: CountFit, cells: list[tuple[int, int | None]]
) -> tuple[float, ...]:
    """Return fitted expected units per pooled cell (open last cell).

    References:
        [EQ-INF-CHI2-GOF-001].

    """
    values: list[float] = []
    for low, high in cells:
        if high is None:
            below = math.fsum(fit.pmf(k) for k in range(low))
            values.append(fit.sample_size * max(0.0, 1.0 - below))
        else:
            values.append(
                fit.sample_size * math.fsum(fit.pmf(k) for k in range(low, high + 1))
            )
    return tuple(values)


def _pearson(observed: tuple[int, ...], expected: tuple[float, ...]) -> float:
    """Return ``sum (O - E)^2 / E`` over cells with positive expectation.

    References:
        [EQ-INF-CHI2-GOF-001].

    """
    return math.fsum(
        (o - e) ** 2 / e for o, e in zip(observed, expected, strict=True) if e > 0.0
    )


def _cumulative(fit: CountFit) -> list[float]:
    """Return the fitted CDF table up to a tail mass below ``1e-15``.

    References:
        [EQ-DIST-POIS-002]. Cumulative sums of the fitted mass.

    """
    table: list[float] = []
    total = 0.0
    k = 0
    while total < 1.0 - _TAIL_EPSILON and k < 100_000:
        total += fit.pmf(k)
        table.append(total)
        k += 1
    return table


def count_goodness_of_fit(
    values: Iterable[object],
    fit: CountFit,
    *,
    n_bootstrap: int = 999,
    seed: int | RandomStream | None = None,
    min_expected: float = 5.0,
) -> CountGoodnessOfFit:
    """Pearson goodness of fit for a count model with a bootstrap p-value.

    Cells ``{0}, {1}, ...`` and the open upper tail are merged from the left
    until every pooled cell expects at least ``min_expected`` units (the
    remainder joins the last cell). The asymptotic p-value uses
    ``cells - 1 - k`` degrees of freedom; the bootstrap draws ``n`` counts
    from the fitted law by inversion, refits the family, and recomputes
    ``X^2`` on the same cells with the refitted expectations.

    Args:
        values: The counts the model was fitted to.
        fit: Fit from :func:`fit_poisson_counts` or
            :func:`fit_negative_binomial_counts` on ``values``.
        n_bootstrap: Positive number of parametric-bootstrap replicates.
        seed: Integer seed or :class:`RandomStream`; ``None`` is not
            reproducible.
        min_expected: Positive minimum expected units per pooled cell.

    Returns:
        Cells, observed and expected units, ``X^2``, and both p-values.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the fit does not match the counts or settings are
            invalid.
        ArithmeticError: If every bootstrap refit fails.

    Equation:
        [EQ-INF-CHI2-GOF-001], [EQ-INF-GOF-PB-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-COCHRAN-1954], [REF-DAVISON-HINKLEY-1997, Chapter 4].

    """
    counts = _counts(values)
    if not isinstance(fit, CountFit):
        raise TypeError("fit must be a CountFit")
    if fit.family not in _FAMILIES or fit.sample_size != len(counts):
        raise ValueError("fit does not belong to these counts")
    if not math.isclose(fit.mean, sum(counts) / len(counts), rel_tol=1e-12):
        raise ValueError("fit does not belong to these counts")
    replicates = require_non_negative_integer(n_bootstrap, name="n_bootstrap")
    if replicates == 0:
        raise ValueError("n_bootstrap must be positive")
    threshold = require_finite_real(min_expected, name="min_expected")
    if threshold <= 0.0:
        raise ValueError("min_expected must be positive")
    table = _table(counts)
    cells = _pooled_cells(fit.expected_frequencies(max(table)), threshold)
    observed = _cell_counts(table, cells)
    expected = _cell_expectations(fit, cells)
    statistic = _pearson(observed, expected)
    df = len(cells) - 1 - len(fit.estimates)
    asymptotic = chi_square_sf(statistic, df) if df >= 1 else math.nan
    stream = resolve_stream(seed).child("count-gof", fit.family)
    cumulative = _cumulative(fit)
    exceed = refits = failed = 0
    for index in range(replicates):
        generator = stream.child(index).generator()
        replicate = tuple(
            min(bisect.bisect_right(cumulative, generator.random()), len(cumulative))
            for _ in counts
        )
        try:
            refitted = _refit(fit.family, replicate)
        except (ValueError, ArithmeticError):
            failed += 1
            continue
        value = _pearson(
            _cell_counts(_table(replicate), cells),
            _cell_expectations(refitted, cells),
        )
        refits += 1
        exceed += value >= statistic
    if refits == 0:
        raise ArithmeticError("every bootstrap refit failed")
    return CountGoodnessOfFit(
        family=fit.family,
        cells=tuple(cells),
        observed=observed,
        expected=expected,
        statistic=statistic,
        degrees_of_freedom=df,
        asymptotic_p_value=asymptotic,
        bootstrap_p_value=(1 + exceed) / (1 + refits),
        n_bootstrap=replicates,
        refits=refits,
        failed_refits=failed,
    )
