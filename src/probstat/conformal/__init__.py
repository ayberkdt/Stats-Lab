"""Distribution-free conformal prediction intervals for any point predictor.

``quantile`` holds the finite-sample conformal quantile and its coverage
guarantee; ``regression`` provides split conformal (absolute and normalized
scores), conformalized quantile regression, and jackknife+; ``coverage``
measures empirical coverage on test data; ``surrogate`` calibrates error bars
for a PCE or GP emulator against its true probabilistic model; ``adaptive``
provides adaptive conformal inference for sequential prediction under
distribution shift.

Main entry points:
    ``split_conformal_regression``: split conformal intervals with absolute
        or normalized scores.
    ``conformalized_quantile_regression``: conformal correction of lower
        and upper quantile predictors.
    ``jackknife_plus_regression``: jackknife+ intervals from ``n``
        leave-one-out refits.
    ``conformalize_surrogate``: error bars for a PCE or GP emulator of a
        probabilistic model.
    ``empirical_coverage``: coverage rate with a Clopper-Pearson interval.
    ``adaptive_conformal_intervals``: adaptive conformal inference for a
        sequence under drift.

Assumptions:
    Calibration and test pairs are exchangeable, and the predictor (and any
    scale) was fitted without the calibration data. The guarantee is
    marginal: it averages over calibration sets and test points, not over
    one input. Adaptive conformal inference drops exchangeability and
    bounds only the long-run miscoverage frequency.

Common pitfalls:
    - ``alpha`` is the miscoverage: ``alpha=0.1`` asks for 90% intervals.
    - When ``ceil((1 - alpha)(n + 1)) > n`` the quantile is ``+inf``; with
      ``alpha = 0.1`` this happens for fewer than nine calibration points.
    - Calibration inputs must be random draws; Latin hypercube or Sobol
      designs are not exchangeable with a random test input.
    - Jackknife+ guarantees only ``1 - 2 alpha``.
    - Normalized scores with a vanishing GP standard deviation make
      ``conformalize_surrogate`` raise; choose ``scale_offset`` before
      seeing the calibration data.

Guide: docs/foundations/conformal-prediction.md

References:
    [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
    [REF-BARBER-ETAL-2021], [REF-ANGELOPOULOS-BATES-2023],
    [REF-GIBBS-CANDES-2021].

"""

from .adaptive import (
    AdaptiveConformalPredictor,
    AdaptiveConformalResult,
    adaptive_conformal_intervals,
)
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
    "AdaptiveConformalPredictor",
    "AdaptiveConformalResult",
    "adaptive_conformal_intervals",
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
