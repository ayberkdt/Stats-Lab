"""Model-validation evidence kept distinct from verification and prediction.

The package owns discrepancy records, validation-experiment records and their
applicability domain, validation metrics (area metric, u-pooling, Gaussian
residuals, reliability metric, ASME V&V 20 comparison, Bayes factor), the
scoped :func:`assess_validation` report, and a :class:`CredibilityRecord` that
keeps verification, validation, and prediction evidence separate.
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
from .records import DomainCheck, ValidationDomain, ValidationExperiment

__all__ = [
    "AdditiveGaussianDiscrepancy",
    "AsmeComparison",
    "BayesFactorResult",
    "CredibilityRecord",
    "DomainCheck",
    "ExperimentAssessment",
    "MahalanobisResult",
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
    "interpret_log_bayes_factor",
    "mahalanobis_metric",
    "predict_at_conditions",
    "probability_integral_transform",
    "reliability_metric",
    "silverman_bandwidth",
    "u_pooling",
    "uniform_area",
    "validation_bayes_factor",
]
