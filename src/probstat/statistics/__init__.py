"""Descriptive statistics for finite samples and one-pass streaming summaries."""

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
