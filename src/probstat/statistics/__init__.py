"""Descriptive statistics, one-pass streaming summaries, and historical data sets.

Sample summaries computed with ``math.fsum`` and centred two-pass formulas,
order and rank statistics, kernel density estimates, constant-memory
streaming accumulators that merge across chunks, and a few classic data
sets with their provenance.

Main entry points:
    ``variance``: sample variance with an explicit ``ddof``.
    ``quantile``: sample quantile of Hyndman-Fan types 1-9 (type 7 by
        default).
    ``skewness`` / ``excess_kurtosis``: shape statistics in the ``g``
        (default), ``G``, or ``b`` convention of Joanes and Gill.
    ``median_absolute_deviation``: robust scale, normal-consistent by
        default.
    ``pearson_correlation``: linear association of paired samples.
    ``gaussian_kde``: Gaussian kernel density estimate with a rule-of-thumb
        or cross-validated bandwidth.
    ``select_kde_bandwidth``: bandwidth by leave-one-out likelihood, unbiased
        (least-squares) or biased cross-validation.
    ``gaussian_kde_multivariate``: multivariate Gaussian KDE with kernel
        covariance ``h^2 R``, Silverman, Scott, or likelihood factor.
    ``RunningMoments``: mergeable one-pass mean, variance, skewness, and
        kurtosis.
    ``P2Quantile``: fixed-memory streaming estimate of one quantile.
    ``median_of_means`` / ``catoni_mean``: mean estimators with
        sub-Gaussian-type deviations under a finite variance only.
    ``ledoit_wolf_covariance``: linear shrinkage of a sample covariance
        matrix towards a scaled identity.
    ``chatterjee_correlation``: rank coefficient of functional dependence
        with an asymptotic independence test.
    ``distance_correlation``: distance covariance and correlation of
        vector observations with a permutation test.

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
      ``normal_consistent=False``; ``skewness`` and ``excess_kurtosis``
      default to the uncorrected sample moment ratios (``g1``, ``g2``).
    - ``trimmed_mean`` trims ``floor(proportion * n)`` values from each end
      with ``proportion=0.1`` by default, so it equals the plain mean for
      fewer than ten observations.
    - ``skewness`` and ``excess_kurtosis`` default to ``type=1`` while R's
      ``e1071`` defaults to type 3; ``quantile`` defaults to type 7 while
      SAS uses type 2.
    - ``P2Quantile`` cannot be merged (only the moment, covariance, and
      exceedance accumulators have ``merge``).
    - ``median_of_means`` and ``catoni_mean`` are tuned to one confidence
      level and are biased for skewed laws at a fixed tuning; the default
      scale of ``catoni_mean`` plugs in the sample variance, a heuristic.
    - ``chatterjee_correlation(x, y)`` is not symmetric, equals
      ``(n - 2)/(n + 1)`` rather than 1 for a monotone relation, and
      rejects ties in ``y``.
    - ``distance_correlation`` is positive for independent samples (the
      squared distance covariance is biased upwards); use the permutation
      test before reading it as dependence.
    - ``ledoit_wolf_covariance`` uses the divisor ``n`` and centres the
      data by default.

Guide: docs/foundations/descriptive-statistics.md

References:
    [REF-NIST-EDA] (descriptive statistics), [REF-HYNDMAN-FAN-1996]
    (sample quantiles), [REF-JOANES-GILL-1998] (skewness and kurtosis),
    [REF-WELFORD-1962] (one-pass moments),
    [REF-JAIN-CHLAMTAC-1985] (P-square quantile), [REF-VITTER-1985]
    (reservoir sampling), [REF-LUGOSI-MENDELSON-2019], [REF-CATONI-2012]
    (robust means), [REF-LEDOIT-WOLF-2004] (shrinkage covariance),
    [REF-CHATTERJEE-2021], [REF-SZEKELY-RIZZO-BAKIROV-2007] (dependence).

"""

from .bandwidth_selection import (
    BandwidthSelection,
    kde_bcv,
    kde_loo_log_likelihood,
    kde_ucv,
    oversmoothed_bandwidth,
    select_kde_bandwidth,
)
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
from .dependence import (
    ChatterjeeResult,
    DistanceCorrelationResult,
    chatterjee_correlation,
    distance_correlation,
    distance_correlation_bivariate_normal,
)
from .descriptive import (
    covariance,
    empirical_cdf,
    mean,
    pearson_correlation,
    standard_deviation,
    variance,
    z_scores,
)
from .multivariate_density import (
    MultivariateKernelDensity,
    gaussian_kde_multivariate,
    multivariate_kde_factor,
    multivariate_kde_loo_log_likelihood,
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
from .robust_mean import (
    CatoniMeanResult,
    MedianOfMeansResult,
    catoni_influence,
    catoni_mean,
    catoni_scale,
    median_of_means,
    median_of_means_block_count,
)
from .shrinkage import LedoitWolfResult, ledoit_wolf_covariance
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
    "BandwidthSelection",
    "gaussian_kde_multivariate",
    "kde_bcv",
    "kde_loo_log_likelihood",
    "kde_ucv",
    "MultivariateKernelDensity",
    "multivariate_kde_factor",
    "multivariate_kde_loo_log_likelihood",
    "oversmoothed_bandwidth",
    "select_kde_bandwidth",
    "CatoniMeanResult",
    "ChatterjeeResult",
    "DistanceCorrelationResult",
    "LedoitWolfResult",
    "MedianOfMeansResult",
    "catoni_influence",
    "catoni_mean",
    "catoni_scale",
    "chatterjee_correlation",
    "distance_correlation",
    "distance_correlation_bivariate_normal",
    "ledoit_wolf_covariance",
    "median_of_means",
    "median_of_means_block_count",
]
