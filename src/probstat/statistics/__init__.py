"""Descriptive statistics, one-pass streaming summaries, and historical data sets.

Sample summaries computed with ``math.fsum`` and centred two-pass formulas,
order and rank statistics, kernel density estimates, constant-memory
streaming accumulators that merge across chunks, and a few classic data
sets with their provenance.

Main entry points:
    ``variance``: sample variance with an explicit ``ddof``.
    ``quantile``: Hyndman-Fan type 7 (default) or type 1 sample quantile.
    ``median_absolute_deviation``: robust scale, normal-consistent by
        default.
    ``pearson_correlation``: linear association of paired samples.
    ``gaussian_kde``: Gaussian kernel density estimate with a rule-of-thumb
        bandwidth.
    ``RunningMoments``: mergeable one-pass mean, variance, skewness, and
        kurtosis.
    ``P2Quantile``: fixed-memory streaming estimate of one quantile.

Assumptions:
    Samples are non-empty sequences of finite real numbers (booleans and
    non-finite values are rejected). The summaries describe the data; an
    estimator interpretation (``ddof=1`` unbiased under i.i.d. sampling with
    finite variance) is the caller's choice. Streaming results equal the
    batch results up to rounding, except ``P2Quantile``, which is an
    approximation that depends on arrival order.

Common pitfalls:
    - ``variance``, ``standard_deviation``, ``covariance``, and
      ``z_scores`` default to ``ddof=0`` (divide by ``n``), whereas the
      standard-library ``statistics.variance`` divides by ``n - 1``. Pass
      ``ddof=1`` for the unbiased sample variance.
    - ``RunningMoments.variance()`` also defaults to ``ddof=0``, but
      ``RunningMoments.standard_error()`` uses ``ddof=1``.
    - ``median_absolute_deviation`` multiplies by about ``1.4826`` unless
      ``normal_consistent=False``; ``skewness`` and ``excess_kurtosis`` are
      the uncorrected sample moment ratios (``g1``, ``g2``).
    - ``trimmed_mean`` trims ``floor(proportion * n)`` values from each end
      with ``proportion=0.1`` by default, so it equals the plain mean for
      fewer than ten observations.
    - ``quantile`` supports only ``method=7`` and ``method=1``, and
      ``P2Quantile`` cannot be merged (only the moment, covariance, and
      exceedance accumulators have ``merge``).

Guide: docs/foundations/descriptive-statistics.md

References:
    [REF-NIST-EDA] (descriptive statistics), [REF-HYNDMAN-FAN-1996]
    (sample quantiles), [REF-WELFORD-1962] (one-pass moments),
    [REF-JAIN-CHLAMTAC-1985] (P-square quantile), [REF-VITTER-1985]
    (reservoir sampling).

"""

from .datasets import (
    BORTKIEWICZ_HORSE_KICKS,
    DATASETS,
    LIEBLEIN_ZELEN_BALL_BEARINGS,
    NEWCOMB_PASSAGE_TIMES,
    RUTHERFORD_GEIGER_SCINTILLATIONS,
    STUDENT_SLEEP,
    DataColumn,
    DataSet,
    dataset_by_key,
)
from .density import KernelDensity, gaussian_kde, kde_bandwidth
from .descriptive import (
    covariance,
    empirical_cdf,
    mean,
    pearson_correlation,
    standard_deviation,
    variance,
    z_scores,
)
from .order import (
    excess_kurtosis,
    interquartile_range,
    median,
    median_absolute_deviation,
    quantile,
    quantiles,
    ranks,
    skewness,
    spearman_correlation,
    trimmed_mean,
)
from .streaming import (
    ExceedanceCounter,
    P2Quantile,
    ReservoirSampler,
    RunningCovariance,
    RunningMoments,
    StreamingRunResult,
    WeightedRunningMoments,
    run_until,
)

__all__ = [
    "BORTKIEWICZ_HORSE_KICKS",
    "DATASETS",
    "DataColumn",
    "DataSet",
    "dataset_by_key",
    "LIEBLEIN_ZELEN_BALL_BEARINGS",
    "NEWCOMB_PASSAGE_TIMES",
    "RUTHERFORD_GEIGER_SCINTILLATIONS",
    "STUDENT_SLEEP",
    "trimmed_mean",
    "covariance",
    "empirical_cdf",
    "ExceedanceCounter",
    "excess_kurtosis",
    "gaussian_kde",
    "interquartile_range",
    "kde_bandwidth",
    "KernelDensity",
    "mean",
    "median",
    "median_absolute_deviation",
    "P2Quantile",
    "pearson_correlation",
    "quantile",
    "quantiles",
    "ranks",
    "ReservoirSampler",
    "run_until",
    "RunningCovariance",
    "RunningMoments",
    "skewness",
    "spearman_correlation",
    "standard_deviation",
    "StreamingRunResult",
    "variance",
    "WeightedRunningMoments",
    "z_scores",
]
