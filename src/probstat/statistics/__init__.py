"""Descriptive statistics for finite, univariate and paired samples."""

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

__all__ = [
    "covariance",
    "empirical_cdf",
    "excess_kurtosis",
    "interquartile_range",
    "mean",
    "median",
    "median_absolute_deviation",
    "pearson_correlation",
    "quantile",
    "quantiles",
    "ranks",
    "skewness",
    "spearman_correlation",
    "standard_deviation",
    "variance",
    "z_scores",
]
