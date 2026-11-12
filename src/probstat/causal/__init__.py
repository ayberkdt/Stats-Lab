"""Causal graphs, structural causal models, effect estimation, and sensitivity.

- :mod:`.graphs`: :class:`CausalGraph`, d-separation, backdoor and frontdoor
  criteria, minimal adjustment sets, and ``do`` graph surgery.
- :mod:`.discrete`: finite structural causal models, truncated factorization,
  and adjustment formulas on joint tables.
- :mod:`.effects`: regression adjustment, IPW, AIPW, bootstrap intervals, and
  covariate balance.
- :mod:`.sensitivity`: E-values for unmeasured confounding.

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

Assumptions:
    The graph encodes the true causal structure. Effect estimates assume
    conditional exchangeability given the covariates (no unmeasured
    confounding), positivity (``0 < e(X) < 1``), consistency, and a binary
    treatment; the outcome models are linear and the propensity model
    logistic.

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

Guide: docs/foundations/causal-inference.md

References:
    [REF-PEARL-2009], [REF-HERNAN-ROBINS-2020], [REF-ROSENBAUM-RUBIN-1983],
    [REF-ROBINS-ROTNITZKY-ZHAO-1994], [REF-VANDERWEELE-DING-2017].

"""

from .discrete import (
    AdjustmentComparison,
    DiscreteCausalModel,
    JointTable,
    backdoor_adjustment,
    compare_naive_and_adjusted,
    frontdoor_adjustment,
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
from .sensitivity import (
    EValue,
    confounding_bias_factor,
    e_value,
    risk_ratio_from_odds_ratio,
    risk_ratio_from_standardized_difference,
)

__all__ = [
    "AdjustmentComparison",
    "BalanceTable",
    "CausalGraph",
    "CovariateBalance",
    "CyclicGraphError",
    "DiscreteCausalModel",
    "EValue",
    "EffectEstimate",
    "JointTable",
    "PositivityWarning",
    "PropensityModel",
    "backdoor_adjustment",
    "bootstrap_effect",
    "compare_naive_and_adjusted",
    "confounding_bias_factor",
    "e_value",
    "estimate_aipw",
    "estimate_ipw",
    "estimate_regression_adjustment",
    "fit_propensity_scores",
    "frontdoor_adjustment",
    "minimal_adjustment_sets",
    "risk_ratio_from_odds_ratio",
    "risk_ratio_from_standardized_difference",
    "satisfies_backdoor",
    "satisfies_frontdoor",
    "standardized_mean_differences",
]
