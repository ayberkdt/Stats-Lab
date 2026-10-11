"""Model-validation evidence kept distinct from verification and prediction.

The package owns discrepancy records, validation-experiment records and their
applicability domain, validation metrics (area metric, u-pooling, Gaussian
residuals, reliability metric, ASME V&V 20 comparison, Bayes factor), the
scoped :func:`assess_validation` report, a :class:`CredibilityRecord` that
keeps verification, validation, and prediction evidence separate, and
:func:`predict_with_uncertainty_components`, which reports parametric,
numerical, and model-form prediction uncertainty as separate components.
"""

from .assessment import (
    ExperimentAssessment,
    ValidationAssessment,
    ValidationCriterion,
    assess_predictions,
    assess_validation,
    predict_at_conditions,
)
from .credibility import CredibilityRecord, PredictionStatement, VerificationEvidence
from .discrepancy import AdditiveGaussianDiscrepancy
from .metrics import (
    AsmeComparison,
    BayesFactorResult,
    MahalanobisResult,
    ReliabilityMetricResult,
    UPoolingResult,
    ValidationAlternative,
    area_metric,
    asme_validation_comparison,
    budget_standard_uncertainty,
    interpret_log_bayes_factor,
    mahalanobis_metric,
    probability_integral_transform,
    reliability_metric,
    silverman_bandwidth,
    u_pooling,
    uniform_area,
    validation_bayes_factor,
)
from .prediction import (
    ComponentPrediction,
    DiscrepancyAtCondition,
    DiscrepancyEstimate,
    DiscrepancyExtrapolation,
    NumericalTreatment,
    estimate_discrepancy,
    predict_with_uncertainty_components,
)
from .records import DomainCheck, ValidationDomain, ValidationExperiment

__all__ = [
    "AdditiveGaussianDiscrepancy",
    "AsmeComparison",
    "ComponentPrediction",
    "BayesFactorResult",
    "CredibilityRecord",
    "DiscrepancyAtCondition",
    "DiscrepancyEstimate",
    "DiscrepancyExtrapolation",
    "DomainCheck",
    "ExperimentAssessment",
    "MahalanobisResult",
    "NumericalTreatment",
    "PredictionStatement",
    "ReliabilityMetricResult",
    "UPoolingResult",
    "ValidationAlternative",
    "ValidationAssessment",
    "ValidationCriterion",
    "ValidationDomain",
    "ValidationExperiment",
    "VerificationEvidence",
    "area_metric",
    "asme_validation_comparison",
    "assess_predictions",
    "assess_validation",
    "budget_standard_uncertainty",
    "estimate_discrepancy",
    "interpret_log_bayes_factor",
    "mahalanobis_metric",
    "predict_at_conditions",
    "predict_with_uncertainty_components",
    "probability_integral_transform",
    "reliability_metric",
    "silverman_bandwidth",
    "u_pooling",
    "uniform_area",
    "validation_bayes_factor",
]
