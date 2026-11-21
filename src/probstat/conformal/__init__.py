"""Distribution-free conformal prediction intervals for any point predictor.

``quantile`` holds the finite-sample conformal quantile and its coverage
guarantee; ``regression`` provides split conformal (absolute and normalized
scores), conformalized quantile regression, and jackknife+; ``coverage``
measures empirical coverage on test data; ``surrogate`` calibrates error bars
for a PCE or GP emulator against its true probabilistic model; ``adaptive``
provides adaptive conformal inference for sequential prediction under
distribution shift. Extensions beyond exchangeable split conformal:
``weighted`` (covariate shift with a known likelihood ratio, and fixed
data-independent weights), ``risk_control`` (conformal risk control for
monotone bounded losses), ``pvalues`` (conformal p-values and outlier
detection with FDR control), ``grouped`` (Mondrian, group-conditional
calibration), ``cross`` (CV+), and ``online`` (quantile tracking with fixed
or decaying steps).

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
    ``weighted_split_conformal_regression``: split conformal under
        covariate shift with known likelihood-ratio weights.
    ``nonexchangeable_split_conformal_regression``: fixed weights in
        ``[0, 1]`` with a documented, non-computable coverage gap.
    ``conformal_risk_control``: smallest grid parameter whose expected
        monotone loss is at most ``alpha``.
    ``conformal_p_values``, ``conformal_outlier_detection``: conformal
        p-values and Benjamini-Hochberg outlier detection.
    ``mondrian_conformal_regression``: one calibration quantile per group.
    ``cv_plus_regression``: CV+ intervals from ``K`` fold fits.
    ``online_quantile_intervals``: quantile tracking with fixed or
        decaying step sizes.

Assumptions:
    Calibration and test pairs are exchangeable, and the predictor (and any
    scale) was fitted without the calibration data. The guarantee is
    marginal: it averages over calibration sets and test points, not over
    one input. Adaptive conformal inference drops exchangeability and
    bounds only the long-run miscoverage frequency, as does online
    quantile tracking. Weighted conformal prediction assumes covariate
    shift with the likelihood ratio known; fixed-weight nonexchangeable
    conformal loses a coverage gap that cannot be computed from data.
    Mondrian coverage is conditional on the group only. Conformal p-values
    and the FDR control built on them are valid marginally over the
    calibration set. CV+ guarantees ``1 - 2 alpha`` minus a slack given in
    its source.

Common pitfalls:
    - ``alpha`` is the miscoverage: ``alpha=0.1`` asks for 90% intervals.
    - When ``ceil((1 - alpha)(n + 1)) > n`` the quantile is ``+inf``; with
      ``alpha = 0.1`` this happens for fewer than nine calibration points.
    - Calibration inputs must be random draws; Latin hypercube or Sobol
      designs are not exchangeable with a random test input.
    - Jackknife+ guarantees only ``1 - 2 alpha``.
    - Weighted conformal needs ``dP_test/dP_train``; estimated weights void
      the exact guarantee, and large weights give infinite intervals.
    - Conformal risk control needs losses that are non-increasing in
      ``lambda``; the function raises rather than repairing them.
    - A Mondrian group with fewer than ``ceil(1/alpha) - 1`` calibration
      points gets infinite intervals.
    - Normalized scores with a vanishing GP standard deviation make
      ``conformalize_surrogate`` raise; choose ``scale_offset`` before
      seeing the calibration data.

Guide: docs/foundations/conformal-prediction.md

References:
    [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
    [REF-BARBER-ETAL-2021], [REF-ANGELOPOULOS-BATES-2023],
    [REF-GIBBS-CANDES-2021], [REF-TIBSHIRANI-ETAL-2019],
    [REF-BARBER-ETAL-2023], [REF-ANGELOPOULOS-ETAL-2024],
    [REF-BATES-ETAL-2023], [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
    [REF-ANGELOPOULOS-BARBER-BATES-2024].

"""

from .adaptive import (
    AdaptiveConformalPredictor,
    AdaptiveConformalResult,
    adaptive_conformal_intervals,
)
from .coverage import CoverageReport, empirical_coverage
from .cross import CVPlusRegressor, cv_plus_regression
from .grouped import (
    MondrianConformalRegressor,
    MondrianGroup,
    mondrian_conformal_regression,
)
from .online import (
    OnlineQuantileResult,
    OnlineQuantileTracker,
    online_quantile_intervals,
)
from .pvalues import (
    ConformalOutlierResult,
    conformal_outlier_detection,
    conformal_p_values,
)
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
from .risk_control import ConformalRiskControlResult, conformal_risk_control
from .surrogate import (
    SurrogateConformalResult,
    VarianceSurrogate,
    conformalize_surrogate,
)
from .weighted import (
    WeightedConformalRegressor,
    nonexchangeable_conformal_quantile,
    nonexchangeable_coverage_gap,
    nonexchangeable_split_conformal_regression,
    weighted_conformal_quantile,
    weighted_split_conformal_regression,
)

__all__ = [
    "AdaptiveConformalPredictor",
    "AdaptiveConformalResult",
    "CVPlusRegressor",
    "ConformalOutlierResult",
    "ConformalRiskControlResult",
    "ConformalizedQuantileRegressor",
    "CoverageReport",
    "JackknifePlusRegressor",
    "MondrianConformalRegressor",
    "MondrianGroup",
    "OnlineQuantileResult",
    "OnlineQuantileTracker",
    "PredictionInterval",
    "SplitConformalRegressor",
    "SurrogateConformalResult",
    "VarianceSurrogate",
    "WeightedConformalRegressor",
    "adaptive_conformal_intervals",
    "as_predictor",
    "conformal_outlier_detection",
    "conformal_p_values",
    "conformal_quantile",
    "conformal_rank",
    "conformal_risk_control",
    "conformalize_surrogate",
    "conformalized_quantile_regression",
    "cv_plus_regression",
    "empirical_coverage",
    "jackknife_plus_regression",
    "lower_conformal_rank",
    "mondrian_conformal_regression",
    "nonexchangeable_conformal_quantile",
    "nonexchangeable_coverage_gap",
    "nonexchangeable_split_conformal_regression",
    "online_quantile_intervals",
    "split_conformal_coverage",
    "split_conformal_regression",
    "weighted_conformal_quantile",
    "weighted_split_conformal_regression",
]
