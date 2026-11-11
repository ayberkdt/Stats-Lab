"""One ``UQStudy`` that runs every stage and writes one HTML report.

The model is a decay ``y(x) = A exp(-k x)`` with uncertain amplitude ``A``
and rate ``k``, observed at a condition ``x``. A second QoI,
``deviation = |y(1) - 4 exp(-0.5)|``, measures how far the value at ``x = 1``
strays from its nominal value; large deviations happen on both sides, so its
failure domain has two design points. One study object, one root seed:

1. ``propagate`` and ``extremes`` (POT upper tail of ``y`` at ``x = 1``).
2. ``design_points`` and ``active_reliability`` on ``deviation >= 0.9``.
3. ``calibrate`` against noisy observations of a process that differs from
   the simulator by a smooth oscillation, then ``predict`` and ``design``.
4. ``calibrate_with_discrepancy`` (Kennedy-O'Hagan) on the same data, then
   ``predict`` again: simulator-only against bias-corrected intervals.
5. ``validate`` with the CRPS, PIT, and coverage evidence of ADR-0045.

With Matplotlib installed the report ``uq-study-complete.html`` (light and
dark figures) is written to ``examples/generated/``.

Run with ``PYTHONPATH=src python examples/uq_study_complete_lab.py``.

Definitions:
    ``y(x) = A exp(-k x)``: the simulator, priors ``A ~ N(4, 0.4)`` and
        ``k ~ LogNormal(log 0.5, 0.25)``.
    ``deviation``: ``|y(1) - 4 exp(-0.5)|``; failure is ``deviation >= 0.9``.
    ``zeta(x) = 4.2 exp(-0.45 x) + 0.15 sin(2 x)``: the synthetic physical
        process that generates the 11 calibration and 20 validation
        observations (noise standard deviation 0.03).
    ``delta(x)``: Kennedy-O'Hagan discrepancy, a Gaussian process with
        squared-exponential covariance and lognormal hyperpriors.
    design point: a local minimum-distance point of the failure domain in
        standard-normal space; this domain has two.

Assumptions:
    - Observation noise is Gaussian, independent, and known (0.03).
    - Plain calibration assumes the simulator is exact, which is false here
      by construction; the discrepancy calibration models the bias with a
      stationary Gaussian process.
    - The tail of ``y`` above its 0.95 quantile follows a generalized Pareto
      law; AK-MCS assumes its Gaussian-process surrogate classifies the
      4000-point population correctly once the learning criterion is met.

Equations:
    EQ-UQ-001: the named forward model and the posterior pushed through it.
    EQ-MC-MEAN-001: propagated means.
    EQ-EVA-GPD-001: GPD likelihood of the threshold excesses.
    EQ-EVA-POT-001: POT return levels.
    EQ-EVA-POT-002: delta-method return-level intervals.
    EQ-REL-MDP-001: bulged limit state for successive design points.
    EQ-REL-MDP-002: multi-point FORM series approximation.
    EQ-REL-MDP-003: mixture importance sampling over the design points.
    EQ-REL-AKMCS-001: AK-MCS learning function and stopping rule.
    EQ-REL-AKMCS-002: AK-MCS population estimate and coefficient of
        variation.
    EQ-CAL-NL-001: nonlinear observation model of the plain calibration.
    EQ-MCMC-AM-001: adaptive Metropolis sampling of both posteriors.
    EQ-MCMC-RHAT-002: rank-normalized R-hat in the calibration summaries.
    EQ-OED-EIG-001: expected information gain of new conditions.
    EQ-OED-GREEDY-001: greedy selection of the new conditions.
    EQ-OED-LAP-001: Laplace approximation of the EIG.
    EQ-KOH-MODEL-001: Kennedy-O'Hagan observation model.
    EQ-KOH-POST-001: posterior of parameters and hyperparameters.
    EQ-KOH-IDENT-001: parameter-discrepancy confounding diagnostics.
    EQ-KOH-PRED-001: simulator-only and bias-corrected predictions.
    EQ-VAL-AREA-001: area validation metric.
    EQ-VAL-SCORE-CRPS-002: ensemble CRPS.
    EQ-VAL-CAL-PIT-002: randomized ensemble PIT.
    EQ-VAL-CAL-COV-001: interval coverage with Clopper-Pearson intervals.

Diagnostics:
    - Two design points: the multi-point FORM probability (about 0.020)
      against the single-point value (0.0087), confirmed by mixture
      importance sampling and by AK-MCS, which matches Monte Carlo on the
      same population with no misclassified point.
    - Plain-calibration prediction intervals that exclude the truth at both
      held-out conditions, against bias-corrected intervals that contain it.
    - The identifiability check of the discrepancy calibration.
    - R-hat and ESS warnings: with 2 chains of 1000 draws several
      parameters exceed R-hat 1.01, so posterior summaries are indicative
      rather than converged.
    - The validation verdict, CRPS, PIT mean, and 50% and 90% coverage with
      their Clopper-Pearson intervals.

Cost and error:
    - 140 184 forward-model evaluations in total. Per stage: propagation
      4096; extremes 0 (reuses the propagated sample); design points 4044;
      AK-MCS 4026 (26 training runs plus the Monte Carlo comparison on the
      same 4000-point population); plain calibration 34 441; prediction
      2000; design 25 472; discrepancy calibration 45 105; corrected
      prediction 13 000; validation 8000.
    - AK-MCS reports a coefficient of variation of 0.11 for its estimate;
      mixture importance sampling 0.037.
    - Return levels beyond the 4096-run sample are extrapolations; the EIG
      carries a Monte Carlo standard error (0.023 nats) and Laplace error.

References:
    REF-SMITH-UQ-2024: uncertainty quantification workflow.
    REF-COLES-2001: peaks-over-threshold modelling.
    REF-DERKIUREGHIAN-DAKESSIAN-1998: multiple design points.
    REF-ECHARD-GAYTON-LEMAIRE-2011: AK-MCS.
    REF-HAARIO-SAKSMAN-TAMMINEN-2001: adaptive Metropolis.
    REF-HUAN-MARZOUK-2013: Bayesian experimental design.
    REF-KENNEDY-OHAGAN-2001: calibration with model discrepancy.
    REF-BRYNJARSDOTTIR-OHAGAN-2014: identifiability of the discrepancy.
    REF-GNEITING-RAFTERY-2007: proper scoring rules.
    REF-FERSON-OBERKAMPF-GINZBURG-2008: area validation metric.
    REF-CLOPPER-PEARSON-1934: exact binomial intervals.

"""

from __future__ import annotations

import importlib.util
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import LogNormal, Normal
from probstat.inverse import (
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    NoiseModel,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest
from probstat.validation import ValidationExperiment
from probstat.workflow import DiscrepancyStudyPrediction, UQStudy

NOMINAL = 4.0 * math.exp(-0.5)
NOISE = 0.03
DATA_X = tuple(0.4 * i for i in range(11))
VALIDATION_X = (0.5, 1.5, 2.5, 3.5)
CANDIDATES = tuple({"x": x} for x in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0))
OUTPUT = Path(__file__).parent / "generated"
# Root seed of the study; the observation and validation data use 11 and 5.
SEED = 2026


def decay(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the decay value at ``x`` and the deviation at ``x = 1``.

    Args:
        parameters: ``A``, ``k``, and the condition ``x``.

    Returns:
        ``y`` at ``x`` and ``deviation`` at ``x = 1``.

    References:
        [REF-KENNEDY-OHAGAN-2001]; a transparent teaching model.

    """
    amplitude, rate = parameters["A"], parameters["k"]
    return {
        "y": amplitude * math.exp(-rate * parameters["x"]),
        "deviation": abs(amplitude * math.exp(-rate) - NOMINAL),
    }


def truth(x: float) -> float:
    """Return the physical process: the simulator plus a smooth bias.

    References:
        [REF-KENNEDY-OHAGAN-2001]; synthetic truth for this lab.

    """
    return 4.2 * math.exp(-0.45 * x) + 0.15 * math.sin(2.0 * x)


def model() -> ProbabilisticModel:
    """Return the decay model with priors ``A ~ N(4, 0.4)``, ``k ~ LN``.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """
    inputs = JointInput.independent(
        RandomInput("A", Normal(4.0, 0.4)),
        RandomInput("k", LogNormal(math.log(0.5), 0.25)),
    )
    return ProbabilisticModel(
        inputs,
        decay,
        (
            QuantityOfInterest("y", "decay value"),
            QuantityOfInterest("deviation", "deviation from nominal at x = 1"),
        ),
        name="complete",
        fixed={"x": 1.0},
    )


def observations(seed: int) -> tuple[Experiment, ...]:
    """Return noisy observations of the truth at the data inputs.

    References:
        [REF-KENNEDY-OHAGAN-2001]; synthetic data for this lab.

    """
    generator = random.Random(seed)
    return tuple(
        Experiment({"y": truth(x) + generator.gauss(0.0, NOISE)}, {"x": x})
        for x in DATA_X
    )


def validation_data(seed: int) -> tuple[ValidationExperiment, ...]:
    """Return four replicated validation experiments of the truth.

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV]; synthetic data for this lab.

    """
    generator = random.Random(seed)
    return tuple(
        ValidationExperiment(
            f"x={x:g}",
            tuple(truth(x) + generator.gauss(0.0, NOISE) for _ in range(5)),
            {"x": x},
            measurement_uncertainty=NOISE,
            provenance="synthetic replicate campaign",
        )
        for x in VALIDATION_X
    )


def run(seed: int = SEED) -> UQStudy:
    """Run every stage of one study.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-KENNEDY-OHAGAN-2001].

    """
    study = UQStudy(model(), seed=seed)
    study.propagate(4096, design="mc")
    study.extremes(
        "y", threshold_quantile=0.95, return_periods=(100.0, 1000.0), reuse=True
    )
    study.design_points(
        "deviation",
        0.9,
        failure="above",
        max_points=3,
        importance_samples=2000,
        start=(0.5, 0.5),
    )
    study.active_reliability(
        "deviation",
        0.9,
        failure="above",
        population=4000,
        target_cov=0.15,
        compare_with_monte_carlo=True,
    )
    data = observations(11)
    study.calibrate(
        data,
        (NoiseModel("y", standard_deviation=NOISE),),
        draws=1000,
        warmup=500,
        chains=2,
    )
    study.predict(("y",), conditions=({"x": 2.0}, {"x": 5.0}))
    study.design(CANDIDATES, 2, method="laplace", samples=32)
    process = DiscrepancyProcess(
        ("x",),
        standard_deviation_prior=LogNormal(math.log(0.15), 0.75),
        length_scale_priors=(LogNormal(math.log(0.6), 0.5),),
    )
    study.calibrate_with_discrepancy(
        data, "y", KnownNoise(NOISE), process, draws=1000, warmup=500, chains=2
    )
    study.predict(conditions=({"x": 2.0}, {"x": 5.0}), include_noise=True)
    study.validate("y", validation_data(5), tolerance=0.3, sample_size=2000)
    return study


def write_report(study: UQStudy) -> None:
    """Write the HTML report when Matplotlib is installed.

    References:
        [REF-MATPLOTLIB].

    """
    if importlib.util.find_spec("matplotlib") is None:
        print("(install matplotlib to write the HTML report)")
        return
    from probstat.visualization.study import write_study_html

    path = write_study_html(study, OUTPUT / f"uq-study-{study.name}.html")
    print(f"  report: {path}")


def main() -> None:
    """Run the study, print its report, then the headline numbers.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """
    started = time.perf_counter()
    print("Running one study through every stage (about 20 s)...")
    study = run()
    print(study.report())
    print("Headline numbers")
    points = study.latest("design_points").summary
    print(
        f"  design points: {points['design points']:.0f}, multi-point FORM "
        f"{points['pf[multi-point FORM]']:.3e} against single-point "
        f"{points['pf[single-point FORM]']:.3e}"
    )
    active = study.latest("active_reliability").summary
    print(
        f"  AK-MCS {active['pf[AK-MCS]']:.3e} with {active['training runs']:.0f} "
        f"runs; Monte Carlo on the same population "
        f"{active['pf[MC, same population]']:.3e}"
    )
    plain = study.records[[r.stage for r in study.records].index("prediction")]
    koh = study.latest("prediction")
    assert isinstance(koh.result, DiscrepancyStudyPrediction)
    for index, x in enumerate((2.0, 5.0)):
        key = f"y#{index}"
        print(
            f"  x = {x:g}: truth {truth(x):.3f}; plain 95% "
            f"[{plain.summary[f'q_low[{key}]']:.3f}, "
            f"{plain.summary[f'q_high[{key}]']:.3f}]; simulator-only "
            f"[{koh.summary[f'simulator q_low[{key}]']:.3f}, "
            f"{koh.summary[f'simulator q_high[{key}]']:.3f}]; bias-corrected "
            f"[{koh.summary[f'corrected q_low[{key}]']:.3f}, "
            f"{koh.summary[f'corrected q_high[{key}]']:.3f}]"
        )
    identifiability = study.latest("discrepancy_calibration").summary
    print(f"  identifiability: {identifiability['identifiability']}")
    validation = study.latest("validation").summary
    print(
        f"  validation consistent: {validation['consistent']}; mean CRPS "
        f"{validation['crps.mean']:.4f}; PIT mean {validation['pit.mean']:.3f}; "
        f"90% coverage {validation['coverage[90%]']:.2f} "
        f"({validation['coverage[90%].interval']})"
    )
    print(f"  forward-model evaluations: {study.total_evaluations:,}")
    write_report(study)
    print(f"wall time {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
