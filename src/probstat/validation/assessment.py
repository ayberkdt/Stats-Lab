"""End-to-end validation assessment of one quantity of interest.

:func:`assess_validation` propagates a :class:`~probstat.modeling.
ProbabilisticModel` at every experiment's conditions with one root seed,
computes every metric of :mod:`probstat.validation.metrics`, checks the
applicability domain, and returns a structured :class:`ValidationAssessment`.

The verdict is deliberately narrow: the model is reported as *consistent* or
*inconsistent* with the given data at a stated area-metric tolerance and
significance level, within the stated validation domain. The assessment never
states that a model is valid in general, and predictions outside the domain
are flagged as extrapolation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
)
from probstat.modeling import ProbabilisticModel
from probstat.propagation import propagate
from probstat.uq import NumericalErrorBudget

from .metrics import (
    AsmeComparison,
    BayesFactorResult,
    MahalanobisResult,
    ReliabilityMetricResult,
    UPoolingResult,
    ValidationAlternative,
    area_metric,
    asme_validation_comparison,
    interpret_log_bayes_factor,
    mahalanobis_metric,
    probability_integral_transform,
    reliability_metric,
    u_pooling,
    validation_bayes_factor,
)
from .records import DomainCheck, ValidationDomain, ValidationExperiment
from .scoring import crps_ensemble


def predict_at_conditions(
    model: ProbabilisticModel,
    quantity: str,
    conditions: Mapping[str, float],
    size: int,
    *,
    design: str = "mc",
    seed: int | RandomStream | None = None,
) -> tuple[float, ...]:
    """Propagate a model with experiment conditions fixed and return samples.

    The conditions are merged over ``model.fixed``; they must not name a
    random input of the model (that would silently change its law).

    Args:
        model: Probabilistic model.
        quantity: QoI name.
        conditions: Named deterministic settings for this prediction.
        size: Number of propagated samples (at least two).
        design: Sampling design name accepted by
            :func:`probstat.propagation.propagate`.
        seed: Seed or stream for the propagation.

    Returns:
        Predictive samples of the latent QoI.

    Raises:
        KeyError: If the quantity is unknown.
        ValueError: If a condition shadows a random input or ``size < 2``.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4].

    """
    model.quantity(quantity)
    merged = dict(model.fixed)
    merged.update(conditions)
    conditioned = dataclasses.replace(model, fixed=merged)
    study = propagate(conditioned, size, design=design, seed=seed)
    return study[quantity].values


@dataclass(frozen=True, slots=True)
class ExperimentAssessment:
    """All validation metrics for one experiment.

    Args:
        experiment: The validation experiment.
        latent_predictions: Model predictive samples without measurement noise.
        observable_predictions: Samples compared with data (latent plus seeded
            measurement noise when requested, else the latent samples).
        area: Area metric between ``observable_predictions`` and the data.
        u_values: ``F_i(y)`` for each observation.
        reliability: Reliability metric at the declared tolerance.
        asme: ASME V&V 20 comparison of the predictive mean with the data mean.
        bayes: Bayes factor against the declared alternative.
        standardized_residuals: ``(y - mu) / sqrt(s^2 + sigma_m^2)`` for each
            observation.
        within_tolerance: Whether ``area <= tolerance``.

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-ASME-VV20-2009].

    """

    experiment: ValidationExperiment
    latent_predictions: tuple[float, ...]
    observable_predictions: tuple[float, ...]
    area: float
    u_values: tuple[float, ...]
    reliability: ReliabilityMetricResult
    asme: AsmeComparison
    bayes: BayesFactorResult
    standardized_residuals: tuple[float, ...]
    within_tolerance: bool

    @property
    def predictive_mean(self) -> float:
        """Return the mean of the latent predictive samples.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return math.fsum(self.latent_predictions) / len(self.latent_predictions)

    @property
    def crps(self) -> float:
        """Return the mean ensemble CRPS of the observations.

        Each observation is scored against ``observable_predictions`` with
        :func:`~probstat.validation.scoring.crps_ensemble`; the mean is in
        QoI units and rewards calibration and sharpness jointly.

        References:
            [EQ-VAL-SCORE-CRPS-002], [REF-GNEITING-RAFTERY-2007],
            [REF-HERSBACH-2000].

        """
        scores = [
            crps_ensemble(self.observable_predictions, y)
            for y in self.experiment.observations
        ]
        return math.fsum(scores) / len(scores)


@dataclass(frozen=True, slots=True)
class ValidationCriterion:
    """One explicit acceptance criterion and its outcome.

    Args:
        name: Short criterion name.
        value: Observed statistic.
        threshold: Threshold the statistic is compared with.
        passed: Whether the criterion is met.
        description: Exact rule applied.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapter 12].

    """

    name: str
    value: float
    threshold: float
    passed: bool
    description: str


@dataclass(frozen=True, slots=True)
class ValidationAssessment:
    """Structured validation evidence for one QoI and its narrow verdict.

    Args:
        model_name: Model label.
        quantity: QoI name.
        unit: Optional QoI unit.
        tolerance: Area-metric tolerance in QoI units.
        significance: Significance level for the u-pooling KS criterion.
        experiments: Per-experiment assessments.
        u_pooling: Pooled PIT evidence.
        mahalanobis: Gaussian-summary residual evidence over all observations.
        log_bayes_factor: Sum of per-experiment log Bayes factors.
        alternative: Declared alternative of the Bayes factors.
        domain: Validation domain spanned by the experiment conditions.
        prediction_checks: Domain checks of requested prediction conditions.
        criteria: Criteria that define the verdict.
        sample_size: Predictive samples per experiment.
        seed: Root seed of every random draw.
        include_measurement_noise: Whether observable predictions include
            seeded measurement noise.

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV], [REF-ROY-OBERKAMPF-2011].

    """

    model_name: str
    quantity: str
    unit: str | None
    tolerance: float
    significance: float
    experiments: tuple[ExperimentAssessment, ...]
    u_pooling: UPoolingResult
    mahalanobis: MahalanobisResult
    log_bayes_factor: float
    alternative: ValidationAlternative
    domain: ValidationDomain
    prediction_checks: tuple[DomainCheck, ...]
    criteria: tuple[ValidationCriterion, ...]
    sample_size: int
    seed: int
    include_measurement_noise: bool

    @property
    def consistent(self) -> bool:
        """Return whether every criterion passed.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapter 12].

        """
        return all(criterion.passed for criterion in self.criteria)

    @property
    def verdict(self) -> str:
        """Return the narrow, scoped verdict sentence.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapter 12], [REF-ROY-OBERKAMPF-2011].

        """
        word = "consistent" if self.consistent else "inconsistent"
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"Model '{self.model_name}', QoI '{self.quantity}': {word} with the "
            f"{len(self.experiments)} validation experiment(s) at area-metric "
            f"tolerance {self.tolerance:.4g}{unit} and u-pooling significance "
            f"{self.significance:g}, within the validation domain "
            f"[{self.domain.describe()}]. This statement is conditional on these "
            "data, this QoI, these tolerances, and this domain; it makes no "
            "general claim about the model."
        )

    def summary(self) -> str:
        """Return a multi-line human-readable report.

        Returns:
            Verdict, criteria, per-experiment metrics, pooled metrics, and
            applicability notes.

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV]. No equation is evaluated.

        """
        lines = [self.verdict, "", "Criteria:"]
        for criterion in self.criteria:
            mark = "met" if criterion.passed else "NOT met"
            lines.append(
                f"  - {criterion.name}: {criterion.value:.4g} vs "
                f"{criterion.threshold:.4g} ({mark}); {criterion.description}"
            )
        lines.append("Per-experiment evidence:")
        for item in self.experiments:
            lines.append(
                f"  - {item.experiment.name}: area={item.area:.4g}, "
                f"CRPS={item.crps:.4g}, "
                f"reliability(eps={item.reliability.tolerance:.3g})="
                f"{item.reliability.estimate:.3f}, "
                f"log B={item.bayes.log_bayes_factor:.3g}"
            )
            lines.append(f"      ASME V&V 20: {item.asme.statement()}")
        lines.append(
            f"Pooled: u-pooling area={self.u_pooling.area:.4g}, KS D="
            f"{self.u_pooling.ks.statistic:.4g} (p={self.u_pooling.ks.p_value:.3g}, "
            f"n={self.u_pooling.count}); Mahalanobis D^2="
            f"{self.mahalanobis.squared_distance:.4g} on "
            f"{self.mahalanobis.degrees_of_freedom} dof "
            f"(p={self.mahalanobis.p_value:.3g}, Gaussian summary)."
        )
        lines.append(
            f"Bayes factor: log B={self.log_bayes_factor:.4g} "
            f"({interpret_log_bayes_factor(self.log_bayes_factor)}) against the "
            f"alternative: {self.alternative.description}."
        )
        lines.append(f"Validation domain: {self.domain.describe()}.")
        if self.prediction_checks:
            lines.append("Intended-use prediction conditions:")
            for check in self.prediction_checks:
                values = ", ".join(f"{k}={v:.6g}" for k, v in check.condition.items())
                lines.append(f"  - ({values}): {check.note}")
        lines.append(
            f"Reproducibility: seed={self.seed}, {self.sample_size} predictive "
            "samples per experiment, measurement noise "
            f"{'included' if self.include_measurement_noise else 'excluded'} in "
            "observable predictions."
        )
        return "\n".join(lines)


def assess_predictions(
    quantity: str,
    experiments: Sequence[ValidationExperiment],
    predictions: Sequence[Sequence[float]],
    *,
    tolerance: float,
    significance: float = 0.05,
    reliability_tolerance: float | None = None,
    numerical_uncertainty: float | NumericalErrorBudget = 0.0,
    include_measurement_noise: bool = True,
    alternative: ValidationAlternative | None = None,
    prediction_conditions: Sequence[Mapping[str, float]] = (),
    distance_threshold: float = 0.25,
    coverage_factor: float = 2.0,
    seed: int | RandomStream | None = None,
    model_name: str = "model",
    unit: str | None = None,
) -> ValidationAssessment:
    """Assess user-supplied predictive samples against validation experiments.

    Verdict criteria (all must hold for "consistent"):

    1. every per-experiment area metric is ``<= tolerance``; and
    2. the u-pooling KS p-value is ``>= significance``.

    ASME V&V 20, reliability, Mahalanobis, and Bayes-factor results are
    reported as evidence but do not change the verdict.

    Args:
        quantity: QoI name (for reporting).
        experiments: Validation experiments.
        predictions: One sequence of latent predictive samples per experiment.
        tolerance: Positive area-metric tolerance in QoI units.
        significance: KS significance level in ``(0, 1)``.
        reliability_tolerance: ``epsilon`` of the reliability metric; defaults
            to ``tolerance``.
        numerical_uncertainty: ``u_num`` or a verification error budget for the
            ASME comparison.
        include_measurement_noise: Add seeded ``N(0, sigma_m^2)`` noise to the
            predictive samples before comparing distributions with data.
        alternative: Bayes-factor alternative; defaults to
            ``ValidationAlternative.inflated(5.0)``.
        prediction_conditions: Intended-use conditions to check against the
            validation domain.
        distance_threshold: Scaled distance flag for domain checks.
        coverage_factor: ``k`` of the ASME model-error interval.
        seed: Root seed or stream for measurement-noise draws.
        model_name: Model label for the report.
        unit: Optional QoI unit.

    Returns:
        Structured assessment with :meth:`ValidationAssessment.summary`.

    Raises:
        TypeError: If an experiment has the wrong type.
        ValueError: If lengths disagree or a numeric argument is invalid.

    Equation:
        [EQ-VAL-AREA-001], [EQ-VAL-UPOOL-001], [EQ-VAL-UPOOL-002],
        [EQ-VAL-MAHAL-001], [EQ-VAL-REL-001], [EQ-VAL-ASME-001],
        [EQ-VAL-BF-001], [EQ-VAL-DOM-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-ASME-VV20-2009],
        [REF-REBBA-MAHADEVAN-2008], [REF-KASS-RAFTERY-1995],
        [REF-OBERKAMPF-ROY-2025, Part IV].

    """
    items = tuple(experiments)
    samples = tuple(tuple(p) for p in predictions)
    if not items:
        raise ValueError("at least one experiment is required")
    if len(samples) != len(items):
        raise ValueError("predictions must have one sample set per experiment")
    domain = ValidationDomain.from_experiments(items)
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    alpha = require_finite_real(significance, name="significance")
    if not 0.0 < alpha < 1.0:
        raise ValueError("significance must lie strictly between zero and one")
    epsilon = tol if reliability_tolerance is None else reliability_tolerance
    threshold = require_non_negative_finite_real(
        distance_threshold, name="distance_threshold"
    )
    chosen = ValidationAlternative.inflated(5.0) if alternative is None else alternative
    if not isinstance(chosen, ValidationAlternative):
        raise TypeError("alternative must be a ValidationAlternative")
    root = resolve_stream(seed)

    assessed: list[ExperimentAssessment] = []
    all_observed: list[float] = []
    all_means: list[float] = []
    all_variances: list[float] = []
    for index, (experiment, latent) in enumerate(zip(items, samples, strict=True)):
        if len(latent) < 2:
            raise ValueError(f"experiment {experiment.name!r} needs >= 2 samples")
        sigma_m = experiment.measurement_uncertainty
        if include_measurement_noise and sigma_m > 0.0:
            generator = root.child("validation", "noise", index).generator()
            observable = tuple(v + generator.gauss(0.0, sigma_m) for v in latent)
        else:
            observable = latent
        size = len(latent)
        mean = math.fsum(latent) / size
        variance = math.fsum((v - mean) ** 2 for v in latent) / (size - 1)
        predictive_var = variance + sigma_m * sigma_m
        if predictive_var <= 0.0:
            raise ValueError(
                f"experiment {experiment.name!r} has zero predictive variance and "
                "zero measurement uncertainty; Gaussian metrics are undefined"
            )
        area = area_metric(observable, experiment.observations)
        u_values = tuple(
            probability_integral_transform(observable, y)
            for y in experiment.observations
        )
        standardized = tuple(
            (y - mean) / math.sqrt(predictive_var) for y in experiment.observations
        )
        all_observed.extend(experiment.observations)
        all_means.extend([mean] * experiment.size)
        all_variances.extend([predictive_var] * experiment.size)
        assessed.append(
            ExperimentAssessment(
                experiment,
                latent,
                observable,
                area,
                u_values,
                reliability_metric(observable, experiment.observations, epsilon),
                asme_validation_comparison(
                    mean,
                    experiment.mean,
                    numerical_uncertainty=numerical_uncertainty,
                    input_uncertainty=math.sqrt(variance),
                    data_uncertainty=experiment.data_standard_uncertainty,
                    coverage_factor=coverage_factor,
                ),
                validation_bayes_factor(
                    latent,
                    experiment.observations,
                    alternative=chosen,
                    noise_standard_deviation=sigma_m,
                ),
                standardized,
                area <= tol,
            )
        )
    pooled = u_pooling(
        [
            (item.observable_predictions, item.experiment.observations)
            for item in assessed
        ]
    )
    count = len(all_observed)
    covariance = tuple(
        tuple(all_variances[i] if i == j else 0.0 for j in range(count))
        for i in range(count)
    )
    mahalanobis = mahalanobis_metric(all_observed, all_means, covariance)
    log_bf = math.fsum(item.bayes.log_bayes_factor for item in assessed)
    worst = max(item.area for item in assessed)
    criteria = (
        ValidationCriterion(
            "maximum area metric",
            worst,
            tol,
            worst <= tol,
            "every per-experiment area metric must be <= the tolerance",
        ),
        ValidationCriterion(
            "u-pooling KS p-value",
            pooled.ks.p_value,
            alpha,
            pooled.ks.p_value >= alpha,
            "pooled PIT values must not reject Uniform(0, 1) at the significance",
        ),
    )
    checks = tuple(
        domain.check(condition, distance_threshold=threshold)
        for condition in prediction_conditions
    )
    return ValidationAssessment(
        model_name,
        quantity,
        unit,
        tol,
        alpha,
        tuple(assessed),
        pooled,
        mahalanobis,
        log_bf,
        chosen,
        domain,
        checks,
        criteria,
        min(len(s) for s in samples),
        root.seed,
        include_measurement_noise,
    )


def assess_validation(
    model: ProbabilisticModel,
    quantity: str,
    experiments: Sequence[ValidationExperiment],
    *,
    tolerance: float,
    sample_size: int = 2000,
    design: str = "mc",
    seed: int | RandomStream | None = None,
    significance: float = 0.05,
    reliability_tolerance: float | None = None,
    numerical_uncertainty: float | NumericalErrorBudget = 0.0,
    include_measurement_noise: bool = True,
    alternative: ValidationAlternative | None = None,
    prediction_conditions: Sequence[Mapping[str, float]] = (),
    distance_threshold: float = 0.25,
    coverage_factor: float = 2.0,
) -> ValidationAssessment:
    """Propagate a model at every experiment's conditions and assess it.

    Each experiment ``i`` is predicted with the stream
    ``root.child("validation", "prediction", i)``; measurement noise uses
    ``root.child("validation", "noise", i)``. One integer seed therefore
    reproduces the entire assessment. See :func:`assess_predictions` for the
    verdict criteria.

    Args:
        model: Probabilistic model whose joint input law expresses the input
            uncertainty to propagate.
        quantity: QoI name to validate.
        experiments: Validation experiments; their conditions are passed to
            the model as fixed inputs.
        tolerance: Positive area-metric tolerance in QoI units.
        sample_size: Predictive samples per experiment (at least two).
        design: Sampling design name.
        seed: Root seed or stream.
        significance: KS significance level.
        reliability_tolerance: Reliability-metric ``epsilon``.
        numerical_uncertainty: ``u_num`` or a verification error budget.
        include_measurement_noise: Add seeded measurement noise to predictions.
        alternative: Bayes-factor alternative.
        prediction_conditions: Intended-use conditions to check.
        distance_threshold: Scaled distance flag for domain checks.
        coverage_factor: ``k`` of the ASME interval.

    Returns:
        Structured validation assessment.

    Raises:
        TypeError: If ``model`` is not a :class:`ProbabilisticModel`.
        KeyError: If the quantity is unknown.
        ValueError: If inputs are invalid (see :func:`assess_predictions`).

    Equation:
        [EQ-UQ-001], [EQ-VAL-AREA-001], [EQ-VAL-UPOOL-002], [EQ-VAL-ASME-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-ROY-OBERKAMPF-2011],
        [REF-OBERKAMPF-ROY-2025, Part IV].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    model.quantity(quantity)
    items = tuple(experiments)
    if not items:
        raise ValueError("at least one experiment is required")
    root = resolve_stream(seed)
    predictions = [
        predict_at_conditions(
            model,
            quantity,
            experiment.conditions,
            sample_size,
            design=design,
            seed=root.child("validation", "prediction", index),
        )
        for index, experiment in enumerate(items)
    ]
    unit = model.quantity(quantity).unit
    return assess_predictions(
        quantity,
        items,
        predictions,
        tolerance=tolerance,
        significance=significance,
        reliability_tolerance=reliability_tolerance,
        numerical_uncertainty=numerical_uncertainty,
        include_measurement_noise=include_measurement_noise,
        alternative=alternative,
        prediction_conditions=prediction_conditions,
        distance_threshold=distance_threshold,
        coverage_factor=coverage_factor,
        seed=root,
        model_name=model.name,
        unit=unit,
    )
