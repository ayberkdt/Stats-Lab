"""Distribution-free conformal prediction intervals for any point predictor.

``quantile`` holds the finite-sample conformal quantile and its coverage
guarantee; ``regression`` provides split conformal (absolute and normalized
scores), conformalized quantile regression, and jackknife+; ``coverage``
measures empirical coverage on test data; ``surrogate`` calibrates error bars
for a PCE or GP emulator against its true probabilistic model.
"""

from .coverage import CoverageReport, empirical_coverage
from .quantile import (
    conformal_quantile,
    conformal_rank,
    lower_conformal_rank,
    split_conformal_coverage,
)
from .regression import (
    ConformalizedQuantileRegressor,
    JackknifePlusRegressor,
    PredictionInterval,
    SplitConformalRegressor,
    as_predictor,
    conformalized_quantile_regression,
    jackknife_plus_regression,
    split_conformal_regression,
)
from .surrogate import (
    SurrogateConformalResult,
    VarianceSurrogate,
    conformalize_surrogate,
)

__all__ = [
    "ConformalizedQuantileRegressor",
    "CoverageReport",
    "JackknifePlusRegressor",
    "PredictionInterval",
    "SplitConformalRegressor",
    "SurrogateConformalResult",
    "VarianceSurrogate",
    "as_predictor",
    "conformal_quantile",
    "conformal_rank",
    "conformalize_surrogate",
    "conformalized_quantile_regression",
    "empirical_coverage",
    "jackknife_plus_regression",
    "lower_conformal_rank",
    "split_conformal_coverage",
    "split_conformal_regression",
]
