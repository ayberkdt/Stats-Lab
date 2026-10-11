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


def run(seed: int = 2026) -> UQStudy:
    """Run every stage of one study.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-KENNEDY-OHAGAN-2001].

    """
    study = UQStudy(model(), seed=seed)
    study.propagate(4096, design="mc")
    study.extremes("y", threshold_quantile=0.95, return_periods=(100.0, 1000.0))
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
