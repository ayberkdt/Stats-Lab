"""Causal graphs, structural causal models, effect estimation, and sensitivity.

- :mod:`.graphs`: :class:`CausalGraph`, d-separation, backdoor and frontdoor
  criteria, minimal adjustment sets, and ``do`` graph surgery.
- :mod:`.discrete`: finite structural causal models, truncated factorization,
  and adjustment formulas on joint tables.
- :mod:`.effects`: regression adjustment, IPW, AIPW, bootstrap intervals, and
  covariate balance.
- :mod:`.sensitivity`: E-values for unmeasured confounding.
- :mod:`.omitted_variable_bias`: partial-R-squared sensitivity of a
  regression coefficient to an omitted confounder (robustness values).
- :mod:`.double_ml`: cross-fitted double/debiased machine learning for the
  partially linear model and the cross-fitted AIPW average effect.
- :mod:`.difference_in_differences`: 2x2 difference-in-differences and
  group-time effects for staggered adoption with explicit aggregations.
- :mod:`.synthetic_control`: simplex-constrained synthetic control weights
  and in-space placebo ranks.
- :mod:`.instrumental`: two-stage least squares with robust standard
  errors and first-stage F statistics.

Main entry points:
    ``CausalGraph``: acyclic causal graph with d-separation and graph
        surgery.
    ``minimal_adjustment_sets``: minimal backdoor adjustment sets of a
        graph.
    ``DiscreteCausalModel``: finite structural causal model with exact
        interventional distributions.
    ``estimate_ipw``: inverse-probability-weighted ATE (stabilized by
        default).
    ``estimate_aipw``: doubly robust augmented IPW estimate of the ATE.
    ``bootstrap_effect``: bootstrap interval that resamples whole units
        and refits every model (pass ``seed`` for reproducibility).
    ``e_value``: E-value of a risk ratio and of its confidence limit.
    ``sensitivity_statistics``: partial R-squared and robustness values of
        a regression coefficient; ``adjust_for_confounder`` gives the
        estimate a confounder of stated strength would leave.
    ``dml_partially_linear``: cross-fitted double ML estimate of ``theta``
        in ``Y = theta D + g(X) + e`` with user-supplied learners.
    ``dml_average_treatment_effect``: cross-fitted AIPW average effect.
    ``did_two_by_two``: canonical 2x2 difference-in-differences.
    ``group_time_effects``: ``ATT(g, t)`` under staggered adoption, with
        ``event_study()`` and ``overall()`` aggregations.
    ``synthetic_control``: synthetic control weights and effect path;
        ``synthetic_control_placebo`` ranks it among in-space placebos.
    ``two_stage_least_squares``: instrumental-variable regression.

Assumptions:
    The graph encodes the true causal structure. Effect estimates assume
    conditional exchangeability given the covariates (no unmeasured
    confounding), positivity (``0 < e(X) < 1``), consistency, and a binary
    treatment; the outcome models are linear and the propensity model
    logistic. Double ML replaces the parametric models by learners whose
    errors must shrink fast enough, and still needs unconfoundedness.
    Difference-in-differences needs parallel trends and no anticipation;
    synthetic control needs a stable convex donor combination;
    instrumental variables need relevance, exclusion and exchangeability
    of the instrument. All standard errors of these estimators are
    asymptotic.

Common pitfalls:
    - Adjusting for a collider or its descendant opens a backdoor path
      (M-bias); let ``minimal_adjustment_sets`` choose the set.
    - ``estimate_regression_adjustment`` reports ``standard_error = nan``;
      use ``bootstrap_effect`` for an interval.
    - ``estimate_ipw`` defaults to stabilized (Hajek) weights, its standard
      error treats the propensity scores as known, and any ``trim``
      changes the estimand to the overlap population.
    - The confidence-limit E-value is ``1`` whenever the interval covers
      one; double robustness does not protect against unmeasured
      confounding.
    - Double ML with ``folds=1`` is not cross-fitted (the result says so);
      a slowly converging learner leaves bias the interval does not show.
    - A static two-way fixed-effects coefficient is not an average of the
      ``ATT(g, t)`` under staggered adoption with dynamic effects.
    - The synthetic-control placebo p-value is a rank among ``J + 1``
      units, not a sampling-based p-value.
    - Robustness values say how strong a confounder would have to be, not
      whether one exists.
    - Two-stage least squares is unreliable with weak instruments; read
      ``first_stage_f``.

Guide: docs/foundations/causal-inference.md

References:
    [REF-PEARL-2009], [REF-HERNAN-ROBINS-2020], [REF-ROSENBAUM-RUBIN-1983],
    [REF-ROBINS-ROTNITZKY-ZHAO-1994], [REF-VANDERWEELE-DING-2017],
    [REF-CHERNOZHUKOV-ET-AL-2018], [REF-CALLAWAY-SANTANNA-2021],
    [REF-ROTH-ET-AL-2023], [REF-CINELLI-HAZLETT-2020],
    [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

"""

from .difference_in_differences import (
    AggregatedEffect,
    DifferenceInDifferencesResult,
    GroupTimeEffect,
    GroupTimeEffects,
    did_two_by_two,
    group_time_effects,
    panel_from_long,
)
from .discrete import (
    AdjustmentComparison,
    DiscreteCausalModel,
    JointTable,
    backdoor_adjustment,
    compare_naive_and_adjusted,
    frontdoor_adjustment,
)
from .double_ml import (
    DoubleMLResult,
    dml_average_treatment_effect,
    dml_partially_linear,
    knn_learner,
    linear_learner,
    logistic_learner,
    ridge_learner,
)
from .effects import (
    BalanceTable,
    CovariateBalance,
    EffectEstimate,
    PositivityWarning,
    PropensityModel,
    bootstrap_effect,
    estimate_aipw,
    estimate_ipw,
    estimate_regression_adjustment,
    fit_propensity_scores,
    standardized_mean_differences,
)
from .graphs import (
    CausalGraph,
    CyclicGraphError,
    minimal_adjustment_sets,
    satisfies_backdoor,
    satisfies_frontdoor,
)
from .instrumental import InstrumentalVariablesResult, two_stage_least_squares
from .omitted_variable_bias import (
    ConfounderAdjustment,
    SensitivityStatistics,
    adjust_for_confounder,
    omitted_variable_bias,
    partial_r2,
    robustness_value,
    sensitivity_statistics,
)
from .sensitivity import (
    EValue,
    confounding_bias_factor,
    e_value,
    risk_ratio_from_odds_ratio,
    risk_ratio_from_standardized_difference,
)
from .synthetic_control import (
    PlaceboInference,
    SimplexWeights,
    SyntheticControlResult,
    project_to_simplex,
    synthetic_control,
    synthetic_control_placebo,
    synthetic_control_weights,
)

__all__ = [
    "AdjustmentComparison",
    "AggregatedEffect",
    "BalanceTable",
    "CausalGraph",
    "ConfounderAdjustment",
    "CovariateBalance",
    "CyclicGraphError",
    "DifferenceInDifferencesResult",
    "DiscreteCausalModel",
    "DoubleMLResult",
    "EValue",
    "EffectEstimate",
    "GroupTimeEffect",
    "GroupTimeEffects",
    "InstrumentalVariablesResult",
    "JointTable",
    "PlaceboInference",
    "PositivityWarning",
    "PropensityModel",
    "SensitivityStatistics",
    "SimplexWeights",
    "SyntheticControlResult",
    "adjust_for_confounder",
    "backdoor_adjustment",
    "bootstrap_effect",
    "compare_naive_and_adjusted",
    "confounding_bias_factor",
    "did_two_by_two",
    "dml_average_treatment_effect",
    "dml_partially_linear",
    "e_value",
    "estimate_aipw",
    "estimate_ipw",
    "estimate_regression_adjustment",
    "fit_propensity_scores",
    "frontdoor_adjustment",
    "group_time_effects",
    "knn_learner",
    "linear_learner",
    "logistic_learner",
    "minimal_adjustment_sets",
    "omitted_variable_bias",
    "panel_from_long",
    "partial_r2",
    "project_to_simplex",
    "ridge_learner",
    "risk_ratio_from_odds_ratio",
    "risk_ratio_from_standardized_difference",
    "robustness_value",
    "satisfies_backdoor",
    "satisfies_frontdoor",
    "sensitivity_statistics",
    "standardized_mean_differences",
    "synthetic_control",
    "synthetic_control_placebo",
    "synthetic_control_weights",
    "two_stage_least_squares",
]
