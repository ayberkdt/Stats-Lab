"""Model-validation evidence kept distinct from verification and prediction.

The package owns discrepancy records, validation-experiment records and their
applicability domain, validation metrics (area metric, u-pooling, Gaussian
residuals, reliability metric, ASME V&V 20 comparison, Bayes factor), the
scoped :func:`assess_validation` report, a :class:`CredibilityRecord` that
keeps verification, validation, and prediction evidence separate, and
:func:`predict_with_uncertainty_components`, which reports parametric,
numerical, and model-form prediction uncertainty as separate components.

Probabilistic-forecast evaluation lives in three further modules:
:mod:`~probstat.validation.scoring` (CRPS, log, Dawid-Sebastiani, energy,
variogram, interval, weighted interval, pinball, and Brier scores),
:mod:`~probstat.validation.calibration` (PIT values and histograms, interval
coverage, reliability diagrams, sharpness), and
:mod:`~probstat.validation.shift` (energy-distance and kernel-MMD permutation
tests and Wasserstein distances for distribution shift).

Model-form uncertainty across competing models (stacking and pseudo-BMA
weights of predictive distributions, and the within/between-model variance
split of the averaged prediction) lives in
:mod:`~probstat.validation.model_averaging`.

Main entry points:
    ``ValidationExperiment``: measured values, conditions, measurement
    uncertainty, and provenance of one experiment.
    ``assess_validation``: scoped verdict for one quantity at a stated
    tolerance, with every metric as evidence.
    ``area_metric``: area between the predictive CDF and the data ECDF.
    ``asme_validation_comparison``: ASME V&V 20 model-error interval.
    ``ValidationDomain``: applicability domain and extrapolation check.
    ``CredibilityRecord``: verification, validation, and prediction evidence
    kept separate.
    ``crps_ensemble``: continuous ranked probability score of an ensemble.
    ``pit_histogram``: PIT histogram with a uniformity test and shape indices.

Assumptions:
    A verdict holds only for the stated quantity, tolerance, and validation
    domain. Replicate observations of one experiment are conditionally
    independent, and the default data uncertainty ``sigma_m / sqrt(n)``
    assumes no shared systematic error. Validation data are separate from
    calibration data. The PIT, coverage, and permutation tests assume
    independent (exchangeable) cases; ``pit_values`` assumes continuous
    forecast laws. Scores are negatively oriented: smaller is better.

Common pitfalls:
    - ``ValidationExperiment`` defaults to ``measurement_uncertainty=0.0``,
      so the data are treated as exact until a measurement uncertainty is
      declared.
    - The argument order differs between scores: ``crps_normal(observation,
      mu, sigma)`` takes the observation first, while ``crps_ensemble``
      (members, observation) and ``log_score`` (distribution, observation)
      take it second.
    - ``crps_ensemble`` defaults to ``fair=False``, which is biased upward by
      ``E|X - X'| / (2m)`` for an ``m``-member ensemble.
    - ``ensemble_pit`` returns a randomized rank and defaults to
      ``seed=None``, so its values differ between runs unless a seed is
      given.
    - The ASME V&V 20 interval estimates the model error; it is not a
      pass/fail test. ``error_resolved`` says only whether ``|E|`` exceeds
      ``coverage_factor`` times the validation uncertainty.

Guide: docs/foundations/model-validation.md

References:
    [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-ASME-VV20-2009],
    [REF-OBERKAMPF-ROY-2025], [REF-GNEITING-RAFTERY-2007],
    [REF-KASS-RAFTERY-1995].

"""

from .assessment import (
    ExperimentAssessment,
    ValidationAssessment,
    ValidationCriterion,
    assess_predictions,
    assess_validation,
    predict_at_conditions,
)
from .calibration import (
    CoverageAtLevel,
    PitHistogram,
    ReliabilityBin,
    ReliabilityDiagram,
    SharpnessSummary,
    ensemble_pit,
    interval_coverage,
    pit_coverage,
    pit_histogram,
    pit_values,
    reliability_diagram,
    sharpness_summary,
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
from .model_averaging import (
    ModelAveragedPrediction,
    ModelWeights,
    model_averaged_prediction,
    pseudo_bma_weights,
    stacking_weights,
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
from .scoring import (
    BrierDecomposition,
    WeightedIntervalScore,
    brier_decomposition,
    brier_score,
    crps_distribution,
    crps_ensemble,
    crps_exponential,
    crps_logistic,
    crps_lognormal,
    crps_normal,
    crps_quadrature,
    crps_uniform,
    dawid_sebastiani_score,
    energy_score,
    interval_score,
    log_score,
    pinball_loss,
    reliability_bin_edges,
    variogram_score,
    weighted_interval_score,
)
from .shift import (
    TwoSamplePermutationResult,
    WassersteinEstimate,
    distribution_wasserstein_distance,
    energy_distance,
    energy_distance_test,
    median_heuristic_bandwidth,
    mmd_squared,
    mmd_test,
    normal_wasserstein_distance,
    wasserstein_distance,
)

__all__ = [
    "ModelAveragedPrediction",
    "ModelWeights",
    "model_averaged_prediction",
    "pseudo_bma_weights",
    "stacking_weights",
    "AdditiveGaussianDiscrepancy",
    "AsmeComparison",
    "BayesFactorResult",
    "BrierDecomposition",
    "ComponentPrediction",
    "CoverageAtLevel",
    "CredibilityRecord",
    "DiscrepancyAtCondition",
    "DiscrepancyEstimate",
    "DiscrepancyExtrapolation",
    "DomainCheck",
    "ExperimentAssessment",
    "MahalanobisResult",
    "NumericalTreatment",
    "PitHistogram",
    "PredictionStatement",
    "ReliabilityBin",
    "ReliabilityDiagram",
    "ReliabilityMetricResult",
    "SharpnessSummary",
    "TwoSamplePermutationResult",
    "UPoolingResult",
    "ValidationAlternative",
    "ValidationAssessment",
    "ValidationCriterion",
    "ValidationDomain",
    "ValidationExperiment",
    "VerificationEvidence",
    "WassersteinEstimate",
    "WeightedIntervalScore",
    "area_metric",
    "asme_validation_comparison",
    "assess_predictions",
    "assess_validation",
    "brier_decomposition",
    "brier_score",
    "budget_standard_uncertainty",
    "crps_distribution",
    "crps_ensemble",
    "crps_exponential",
    "crps_logistic",
    "crps_lognormal",
    "crps_normal",
    "crps_quadrature",
    "crps_uniform",
    "dawid_sebastiani_score",
    "distribution_wasserstein_distance",
    "energy_distance",
    "energy_distance_test",
    "energy_score",
    "ensemble_pit",
    "estimate_discrepancy",
    "interpret_log_bayes_factor",
    "interval_coverage",
    "interval_score",
    "log_score",
    "mahalanobis_metric",
    "median_heuristic_bandwidth",
    "mmd_squared",
    "mmd_test",
    "normal_wasserstein_distance",
    "pinball_loss",
    "pit_coverage",
    "pit_histogram",
    "pit_values",
    "predict_at_conditions",
    "predict_with_uncertainty_components",
    "probability_integral_transform",
    "reliability_bin_edges",
    "reliability_diagram",
    "reliability_metric",
    "sharpness_summary",
    "silverman_bandwidth",
    "u_pooling",
    "uniform_area",
    "validation_bayes_factor",
    "variogram_score",
    "wasserstein_distance",
    "weighted_interval_score",
]
