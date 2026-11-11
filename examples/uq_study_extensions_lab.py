"""Tail, information, and design stages of one ``UQStudy`` report.

Three short studies exercise the stages added in ADR-0038:

1. A cantilever tip deflection: the ``extremes`` stage fits a
   peaks-over-threshold model to the propagated sample (no new runs) and
   compares the POT probability of exceeding the deflection limit with the
   empirical fraction.
2. The Ishigami function: ``information_sensitivity`` next to Sobol indices.
   ``x3`` has a zero first-order index but a positive total index; its mutual
   information with the output is also positive.
3. An exponential decay calibrated on two early-time observations: the
   ``design`` stage picks the next measurement times by expected information
   gain.

With Matplotlib installed, every stage figure is written to
``examples/generated/`` and each study gets a self-contained HTML report
(``uq-study-<name>.html``) with light and dark figures. Run with
``PYTHONPATH=src python examples/uq_study_extensions_lab.py``.

Definitions:
    ``delta``: cantilever tip deflection ``4 P L^3 / (E b h^3)``; the limit
        is 0.075 m.
    threshold ``u``: the 0.95 sample quantile of the deflection; excesses
        over ``u`` are modelled by a generalized Pareto law.
    ``N``-run return level: the deflection exceeded once in ``N`` model runs
        on average.
    ``I(X_i; Y)``: mutual information between one Ishigami input and the
        output, in nats.
    EIG: expected information gain of a set of new measurement times about
        the decay parameters ``(A, k)``.

Assumptions:
    - The cantilever inputs are independent (lognormal load, normal modulus
      and thickness); excesses over the 0.95 quantile follow the GPD limit
      law closely enough for extrapolation to ``10^4`` runs.
    - The Ishigami inputs are independent and uniform on ``[-pi, pi]``.
    - The decay observations carry independent Gaussian noise with standard
      deviation 0.05; the uniform priors on ``A`` and ``k`` are the study's
      input law; the Laplace EIG treats the model as locally linear.

Equations:
    EQ-UQ-001: the named forward models.
    EQ-LAB-ISHIGAMI-001: the Ishigami benchmark.
    EQ-MC-PROB-001: empirical exceedance fraction of the propagated sample.
    EQ-EVA-GPD-001: GPD likelihood of the threshold excesses.
    EQ-EVA-RATE-001: threshold exceedance rate.
    EQ-EVA-POT-001: POT return levels.
    EQ-EVA-POT-002: delta-method return-level intervals.
    EQ-SENS-FIRST-001: first-order Sobol indices.
    EQ-SENS-TOTAL-001: total-effect indices.
    EQ-SENS-BOOT-001: bootstrap intervals of the indices.
    EQ-INFO-MI-001: mutual information.
    EQ-INFO-KSG-001: KSG nearest-neighbour estimator of it.
    EQ-CAL-NL-001: nonlinear observation model of the calibration.
    EQ-MCMC-AM-001: adaptive Metropolis sampling of the posterior.
    EQ-MCMC-RHAT-002: rank-normalized R-hat in the calibration summary.
    EQ-OED-EIG-001: expected information gain.
    EQ-OED-GREEDY-001: greedy selection of the new times.
    EQ-OED-LAP-001: Laplace approximation of the EIG.

Diagnostics:
    - The POT probability of exceeding the limit against the empirical
      fraction and its Wilson interval (ratio 0.98).
    - Return levels with intervals next to the empirical quantiles in the
      study report.
    - ``x3``: first-order index near zero, positive total index and mutual
      information, which shows dependence through interactions.
    - R-hat of every calibrated parameter (about 1.01) before the design
      stage uses the posterior.
    - The EIG standard error next to the selected design.

Cost and error:
    - Cantilever: 20 000 Monte Carlo runs; the extremes stage reuses them
      (no new runs). Return levels beyond about ``2 x 10^4`` runs are
      extrapolations whose intervals widen accordingly.
    - Ishigami: 5120 runs for the Sobol indices (bootstrap of 100
      resamples) and 800 fresh runs for the KSG estimator, whose bias at
      ``n = 800`` and ``k = 3`` is not corrected.
    - Decay: 16 138 runs for the posterior (4 chains of 1000 warm-up and
      1000 kept draws) and 15 872 for the design stage; the EIG carries a
      Monte Carlo standard error of 0.14 nats and Laplace approximation
      error.

References:
    REF-COLES-2001: peaks-over-threshold modelling.
    REF-WILSON-1927: score interval of the empirical fraction.
    REF-SALTELLI-2010: Sobol index estimators.
    REF-KRASKOV-STOGBAUER-GRASSBERGER-2004: KSG mutual information.
    REF-HAARIO-SAKSMAN-TAMMINEN-2001: adaptive Metropolis.
    REF-HUAN-MARZOUK-2013: Bayesian experimental design by EIG.
    REF-CHALONER-VERDINELLI-1995: Bayesian experimental design.

"""

from __future__ import annotations

import importlib.util
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.inverse import Experiment, NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.sensitivity import ModelSobolResult
from probstat.uq import QuantityOfInterest
from probstat.workflow import (
    DesignRecommendation,
    InformationSensitivity,
    TailAnalysis,
    UQStudy,
)

LENGTH = 2.0
WIDTH = 0.10
LIMIT = 0.075
TIMES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
OUTPUT = Path(__file__).parent / "generated"
# Root seed shared by the three studies.
SEED = 2026


def cantilever(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the tip deflection of a rectangular cantilever under end load.

    ``delta = 4 P L^3 / (E b h^3)`` for load ``P`` [N], modulus ``E`` [Pa],
    width ``b`` and thickness ``h`` [m].

    Args:
        parameters: ``load``, ``modulus``, and ``thickness``.

    Returns:
        Mapping with ``deflection`` in metres.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 1]; Euler-Bernoulli beam teaching model.

    """
    deflection = (
        4.0
        * parameters["load"]
        * LENGTH**3
        / (parameters["modulus"] * WIDTH * parameters["thickness"] ** 3)
    )
    return {"deflection": deflection}


def ishigami(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Evaluate the Ishigami benchmark with ``a = 7`` and ``b = 0.1``.

    Args:
        parameters: ``x1``, ``x2``, ``x3``, each uniform on ``[-pi, pi]``.

    Returns:
        Mapping with the response ``y``.

    Equation:
        [EQ-LAB-ISHIGAMI-001].

    References:
        [REF-SALTELLI-2008, Chapter 4].

    """
    x1, x2, x3 = parameters["x1"], parameters["x2"], parameters["x3"]
    return {"y": math.sin(x1) + 7.0 * math.sin(x2) ** 2 + 0.1 * x3**4 * math.sin(x1)}


def decay(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the amplitude ``A exp(-k t)`` at measurement time ``t``.

    Args:
        parameters: Amplitude ``A``, rate ``k``, and time ``t``.

    Returns:
        Mapping with the amplitude ``y``.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SMITH-UQ-2024, Chapter 8]; first-order decay teaching model.

    """
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


@dataclass(frozen=True)
class LabResults:
    """Results of the three studies.

    Args:
        beam: Cantilever study.
        tail: Tail analysis of the deflection.
        benchmark: Ishigami study.
        sobol: Sobol indices of the Ishigami response.
        information: Mutual information of the Ishigami response.
        decay: Decay study.
        design: Recommended new measurement times.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    beam: UQStudy
    tail: TailAnalysis
    benchmark: UQStudy
    sobol: ModelSobolResult
    information: InformationSensitivity
    decay: UQStudy
    design: DesignRecommendation


def run(seed: int = SEED) -> LabResults:
    """Run the three studies from one seed.

    Args:
        seed: Root seed shared by the three studies.

    Returns:
        Study objects and stage results.

    References:
        [REF-COLES-2001, Section 4.3], [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004],
        [REF-HUAN-MARZOUK-2013].

    """
    beam_model = ProbabilisticModel(
        JointInput.independent(
            RandomInput("load", LogNormal(7.6, 0.2), unit="N"),
            RandomInput("modulus", Normal(70e9, 5e9), unit="Pa"),
            RandomInput("thickness", Normal(0.06, 0.002), unit="m"),
        ),
        cantilever,
        (QuantityOfInterest("deflection", "tip deflection", "m"),),
        name="cantilever",
    )
    beam = UQStudy(beam_model, seed=seed)
    beam.propagate(20_000, design="mc")
    tail = beam.extremes(
        "deflection",
        threshold_quantile=0.95,
        level=LIMIT,
        return_periods=(100.0, 1000.0, 10_000.0),
        reuse=True,
    )

    uniform = Uniform(-math.pi, math.pi)
    benchmark = UQStudy(
        ProbabilisticModel(
            JointInput.independent(
                *(RandomInput(name, uniform) for name in ("x1", "x2", "x3"))
            ),
            ishigami,
            (QuantityOfInterest("y", "Ishigami response"),),
            name="ishigami",
        ),
        seed=seed,
    )
    sobol = benchmark.sensitivity("y", 1024, bootstrap=100)
    information = benchmark.information_sensitivity("y", max_points=800, size=800)

    decay_model = ProbabilisticModel(
        JointInput.independent(
            RandomInput("A", Uniform(2.0, 8.0)), RandomInput("k", Uniform(0.05, 1.0))
        ),
        decay,
        (QuantityOfInterest("y", "amplitude"),),
        fixed={"t": 0.0},
        name="decay",
    )
    observations = ((0.0, 5.04), (1.0, 5.0 * math.exp(-0.3) - 0.03))
    experiments = tuple(Experiment({"y": y}, {"t": t}) for t, y in observations)
    noise = (NoiseModel("y", standard_deviation=0.05),)
    decay_study = UQStudy(decay_model, seed=seed)
    decay_study.calibrate(experiments, noise, draws=1000, warmup=1000, chains=4)
    design = decay_study.design(
        tuple({"t": t} for t in TIMES), 2, method="laplace", samples=64
    )
    return LabResults(beam, tail, benchmark, sobol, information, decay_study, design)


def write_reports(results: LabResults) -> None:
    """Write the stage figures and one HTML report per study.

    References:
        [REF-SMITH-UQ-2024, Chapter 1], [REF-MATPLOTLIB].

    """
    if importlib.util.find_spec("matplotlib") is None:
        print("(install matplotlib to write the figures and HTML reports)")
        return
    from probstat.visualization.study import save_study_figures, write_study_html

    for study in (results.beam, results.benchmark, results.decay):
        figures = save_study_figures(study, OUTPUT)
        report = write_study_html(study, OUTPUT / f"uq-study-{study.name}.html")
        print(f"  {study.name}: {len(figures)} figures and {report.name}")


def main() -> None:
    """Print the three study reports, then the headline numbers last.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """
    started = time.perf_counter()
    print("Running the cantilever, Ishigami, and decay studies (about 12 s)...")
    results = run()
    for study in (results.beam, results.benchmark, results.decay):
        print(study.report())
    tail = results.tail
    assert tail.empirical is not None
    interval = tail.empirical.confidence_interval
    print("Headline numbers")
    print(
        f"  P(deflection > {LIMIT} m): POT {tail.pot_probability:.3e}, "
        f"empirical {tail.empirical.probability:.3e} "
        f"[{interval.lower:.3e}, {interval.upper:.3e}]"
    )
    for item in tail.return_levels:
        print(
            f"  {item.return_period:>6g}-run level: {item.level * 1e3:.1f} mm "
            f"[{item.lower * 1e3:.1f}, {item.upper * 1e3:.1f}]"
        )
    for name, first, total, mi in zip(
        results.sobol.input_names,
        results.sobol.first_order,
        results.sobol.total_effect,
        results.information.mutual_information,
        strict=True,
    ):
        print(
            f"  {name}: S {first.estimate:.3f}, ST {total.estimate:.3f}, "
            f"MI {mi:.3f} nats"
        )
    selection = results.design.selection
    times = ", ".join(f"{condition['t']:g}" for condition in selection.selected)
    estimate = results.design.estimate
    print(
        f"Next measurement times: t = {times} "
        f"(EIG {estimate.value:.2f} +/- {estimate.standard_error:.2f} nats)."
    )
    write_reports(results)
    print(f"wall time {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
