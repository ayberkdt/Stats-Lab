"""Forward versus inverse problems in one audited VV-UQ workflow.

A first-order decay ``dy/dt = -k y`` is solved numerically with Heun's
second-order method; the step ``h`` is a numerical setting of the simulation.
"Reality" contains a small constant offset the model lacks, so the model has
genuine model-form error. The script

1. defines the QoI and characterizes the uncertain inputs;
2. verifies the solution with a three-grid GCI (numerical uncertainty);
3. solves the **forward** problem: propagates the prior input law;
4. solves the **inverse** problem: calibrates ``A`` and ``k`` from data;
5. validates the calibrated model on separate data and estimates the
   model-form discrepancy; and
6. predicts the QoI outside the validation domain with parametric,
   numerical, and model-form uncertainty reported separately,

recording every step in a :class:`~probstat.uq.UQWorkflowRecord`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from probstat.distributions import Uniform
from probstat.foundations.random_streams import RandomStream
from probstat.inverse import (
    CalibrationPosterior,
    CalibrationProblem,
    Experiment,
    NoiseModel,
    predict,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import OutputDistribution
from probstat.uq import (
    QuantityOfInterest,
    SolutionVerification,
    StepStatus,
    UncertaintyNature,
    UncertaintySource,
    UncertaintySourceKind,
    UQWorkflowRecord,
    WorkflowStep,
    refinement_study,
)
from probstat.validation import (
    ComponentPrediction,
    DiscrepancyEstimate,
    DiscrepancyExtrapolation,
    ValidationAssessment,
    ValidationExperiment,
    assess_predictions,
    estimate_discrepancy,
    predict_with_uncertainty_components,
)
from probstat.workflow import UQStudy

TRUE_A = 5.0
TRUE_K = 0.3
OFFSET = 0.1
NOISE = 0.05
STEP = 0.25
CALIBRATION_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
VALIDATION_TIMES = (7.0, 8.0, 9.0)
PREDICTION_TIME = 10.0
QOI = QuantityOfInterest("y", "decaying amount at time t", "mg")


def heun_decay(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Integrate ``dy/dt = -k y`` from ``y(0) = A`` to ``t`` with Heun's method.

    The number of steps is ``n = ceil(t / h)`` with the effective step
    ``t / n``; Heun's method is second-order accurate.

    Args:
        parameters: ``A``, ``k``, time ``t``, and step ``h``.

    Returns:
        Mapping with the QoI ``y``.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 6-8]; Heun (explicit trapezoidal)
        method as a second-order teaching scheme.

    """
    amount = parameters["A"]
    rate = parameters["k"]
    time = parameters["t"]
    steps = max(1, math.ceil(time / parameters["h"] - 1e-12))
    dt = time / steps
    for _ in range(steps):
        slope = -rate * amount
        predicted = amount + dt * slope
        amount += 0.5 * dt * (slope - rate * predicted)
    return {"y": amount}


def reality(time: float) -> float:
    """Return the true process, which has an offset the model lacks.

    Args:
        time: Time.

    Returns:
        ``A exp(-k t) + offset`` at the true parameters.

    References:
        Synthetic teaching system. No library equation is evaluated.

    """
    return TRUE_A * math.exp(-TRUE_K * time) + OFFSET


@dataclass(frozen=True)
class WorkflowResults:
    """Key results of the worked example.

    Args:
        record: Audited workflow record.
        verification: Solution-verification statement.
        forward: Prior (forward) output distribution at the prediction time.
        posterior: Calibration posterior (inverse problem).
        assessment: Validation assessment of the calibrated model.
        estimate: Discrepancy estimate from the validation comparisons.
        components: Component prediction at the prediction time.
        truth: True value of the process at the prediction time.
        study: The UQStudy that ran the forward and inverse stages.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """

    record: UQWorkflowRecord
    verification: SolutionVerification
    forward: OutputDistribution
    posterior: CalibrationPosterior
    assessment: ValidationAssessment
    estimate: DiscrepancyEstimate
    components: ComponentPrediction
    truth: float
    study: UQStudy


def run(seed: int = 2026, *, draws: int = 600) -> WorkflowResults:
    """Run the complete workflow and return its key results.

    Args:
        seed: Root seed of every random draw.
        draws: Posterior draws per chain (warm-up uses the same number).

    Returns:
        Typed results of every step.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-SMITH-UQ-2024, Chapters 1 and 8].

    """
    root = RandomStream(seed)
    record = UQWorkflowRecord.start("decay prediction at t=10", (QOI,))
    record = record.update(
        WorkflowStep.QOI_DEFINITION,
        StepStatus.COMPLETE,
        evidence=(f"QoI {QOI.name!r} [{QOI.unit}] at t={PREDICTION_TIME:g}",),
    )

    inputs = JointInput.independent(
        RandomInput("A", Uniform(3.0, 7.0), unit="mg"),
        RandomInput("k", Uniform(0.1, 0.6), unit="1/s"),
    )
    sources = (
        UncertaintySource(
            "A",
            "initial amount, imperfectly known",
            UncertaintyNature.EPISTEMIC,
            UncertaintySourceKind.PARAMETRIC,
            "mg",
        ),
        UncertaintySource(
            "k",
            "decay rate, imperfectly known",
            UncertaintyNature.EPISTEMIC,
            UncertaintySourceKind.PARAMETRIC,
            "1/s",
        ),
    )
    record = record.update(
        WorkflowStep.INPUT_CHARACTERIZATION,
        StepStatus.COMPLETE,
        evidence=tuple(
            f"{s.name}: {s.nature.value} {s.kind.value} ({s.description})"
            for s in sources
        ),
    )

    study_refinement = refinement_study(
        heun_decay,
        QOI,
        {"A": 5.0, "k": 0.35, "t": PREDICTION_TIME},
        (4 * STEP, 2 * STEP, STEP),
        spacing_name="h",
        model_name="heun-decay",
    )
    verification = study_refinement.verification
    record = record.update(
        WorkflowStep.VERIFICATION,
        StepStatus.COMPLETE,
        evidence=(verification.summary(),),
    )

    model = ProbabilisticModel(
        inputs,
        heun_decay,
        (QOI,),
        name="heun-decay",
        fixed={"t": PREDICTION_TIME, "h": STEP},
    )
    study = UQStudy(model, seed=root.child("study"))
    forward = study.propagate(1024, design="sobol")

    noise = root.child("data").generator()
    calibration = tuple(
        Experiment({"y": reality(t) + noise.gauss(0.0, NOISE)}, {"t": t})
        for t in CALIBRATION_TIMES
    )
    noise_models = (NoiseModel("y", standard_deviation=NOISE),)
    posterior = study.calibrate(
        calibration, noise_models, draws=draws, warmup=draws, chains=2
    )

    validation_data = tuple(
        ValidationExperiment(
            f"t={t:g}",
            [reality(t) + noise.gauss(0.0, NOISE) for _ in range(4)],
            {"t": t},
            measurement_uncertainty=NOISE,
            provenance="synthetic validation campaign (separate from calibration)",
        )
        for t in VALIDATION_TIMES
    )
    problem = CalibrationProblem(model, calibration, noise_models)
    validation_predictions = predict(
        problem,
        posterior,
        conditions=tuple(dict(item.conditions) for item in validation_data),
        max_draws=400,
        seed=root.child("validation-prediction"),
    )
    assessment = assess_predictions(
        QOI.name,
        validation_data,
        [item.latent for item in validation_predictions.predictions],
        tolerance=0.1,
        numerical_uncertainty=verification.standard_uncertainty("uniform"),
        prediction_conditions=({"t": PREDICTION_TIME},),
        seed=root.child("validation"),
        model_name="calibrated heun-decay",
        unit=QOI.unit,
    )
    estimate = estimate_discrepancy(assessment)
    record = record.update(
        WorkflowStep.VALIDATION,
        StepStatus.COMPLETE,
        evidence=(
            assessment.verdict,
            f"sigma_delta,0={estimate.discrepancy.standard_deviation:.4g} from "
            f"{estimate.count} ASME comparisons (mean error {estimate.mean_error:.4g})",
        ),
    )

    prediction_draws = study.predict(
        conditions=({"t": PREDICTION_TIME},), max_draws=1000
    )
    extrapolation = DiscrepancyExtrapolation(
        estimate.discrepancy, assessment.domain, growth_rate=1.0
    )
    components = predict_with_uncertainty_components(
        QOI.name,
        {"t": PREDICTION_TIME},
        prediction_draws.predictions[0].latent,
        numerical=verification,
        numerical_treatment="interval",
        discrepancy=extrapolation,
        seed=root.child("components"),
    )
    record = record.record_stages(study.records)
    record = record.update(
        WorkflowStep.PREDICTION,
        StepStatus.COMPLETE,
        evidence=(components.uncertainty_sources(),),
    )
    return WorkflowResults(
        record,
        verification,
        forward[QOI.name],
        posterior,
        assessment,
        estimate,
        components,
        reality(PREDICTION_TIME),
        study,
    )


def main() -> None:
    """Print the numbers quoted in the VV-UQ foundations guide.

    References:
        [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

    """
    print(
        "Running verification, forward propagation, calibration, validation, "
        "and prediction (about 2 s)...",
        flush=True,
    )
    results = run()
    print(results.verification.summary())
    forward = results.forward
    print(
        f"forward (prior): mean {forward.mean:.4f}, 5-95% "
        f"[{forward.quantile(0.05):.4f}, {forward.quantile(0.95):.4f}]"
    )
    for entry in results.posterior.summary():
        print(
            f"posterior {entry.name}: mean {entry.mean:.4f}, sd "
            f"{entry.standard_deviation:.4f}, rhat {entry.rank_normalized_rhat:.3f}"
        )
    print(results.assessment.verdict)
    estimate = results.estimate
    print(
        f"discrepancy: sigma_0={estimate.discrepancy.standard_deviation:.4f}, "
        f"mean error {estimate.mean_error:.4f}, truncated {estimate.truncated}"
    )
    print(results.components.summary())
    print(f"truth at t={PREDICTION_TIME:g}: {results.truth:.4f}")
    print(results.record.summary())
    directions = ", ".join(d.value for d in results.record.directions)
    low, high = results.components.combined_interval
    covered = "covers" if low <= results.truth <= high else "MISSES"
    print(
        f"Result: combined 95% interval [{low:.4f}, {high:.4f}] {covered} the "
        f"truth {results.truth:.4f}; directions: {directions}; gaps: "
        f"{len(results.record.gaps)}. Next step: add validation data nearer "
        f"t={PREDICTION_TIME:g} to replace the assumed growth rate kappa."
    )


if __name__ == "__main__":
    main()
