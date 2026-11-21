"""Survival analysis for right-censored time-to-event data.

``data`` holds the censored sample and its risk-set table; ``nonparametric``
provides Kaplan-Meier (Greenwood variance, log-log intervals), Nelson-Aalen,
survival quantiles, restricted mean survival time, and the k-sample log-rank
test; ``parametric`` fits exponential, Weibull, and lognormal
accelerated-failure-time models by censored maximum likelihood; ``cox`` fits
proportional-hazards regression with Breslow or Efron tie handling;
``competing_risks`` provides the Aalen-Johansen cumulative incidence function
with a delta-method variance; ``restricted_mean`` provides the restricted
mean survival time as a function and its two-sample difference and ratio.
"""

from .competing_risks import (
    CumulativeIncidenceEstimate,
    cumulative_incidence,
    cumulative_incidence_curves,
)
from .cox import (
    CoxModelResult,
    PartialLikelihood,
    cox_partial_likelihood,
    fit_cox_model,
)
from .data import RiskSetRow, SurvivalData, survival_data
from .nonparametric import (
    KaplanMeierEstimate,
    LogRankResult,
    NelsonAalenEstimate,
    RestrictedMeanSurvival,
    SurvivalQuantile,
    kaplan_meier,
    log_rank_test,
    nelson_aalen,
)
from .parametric import ParametricSurvivalFit, fit_parametric_survival
from .restricted_mean import (
    RestrictedMeanComparison,
    compare_restricted_means,
    restricted_mean_survival,
)

__all__ = [
    "CoxModelResult",
    "CumulativeIncidenceEstimate",
    "KaplanMeierEstimate",
    "LogRankResult",
    "NelsonAalenEstimate",
    "ParametricSurvivalFit",
    "PartialLikelihood",
    "RestrictedMeanComparison",
    "RestrictedMeanSurvival",
    "RiskSetRow",
    "SurvivalData",
    "SurvivalQuantile",
    "compare_restricted_means",
    "cox_partial_likelihood",
    "cumulative_incidence",
    "cumulative_incidence_curves",
    "fit_cox_model",
    "fit_parametric_survival",
    "kaplan_meier",
    "log_rank_test",
    "nelson_aalen",
    "restricted_mean_survival",
    "survival_data",
]
