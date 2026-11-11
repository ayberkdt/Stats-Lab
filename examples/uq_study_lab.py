"""One model, one seed, one report: a complete UQStudy.

A cantilever beam tip deflection depends on load, Young's modulus, and
thickness. ``UQStudy`` propagates the inputs, ranks them, estimates the
probability of exceeding a deflection limit with several methods, prices the
exceedance risk, fits a polynomial-chaos emulator, and writes a Markdown report
plus one figure per stage. Run with
``PYTHONPATH=src python examples/uq_study_lab.py``; the Markdown and HTML
reports and the stage figures are written to ``examples/generated/``.

Definitions:
    ``delta = 4 P L^3 / (E b h^3)``: tip deflection for load ``P``, modulus
        ``E``, thickness ``h``, length ``L = 2`` m, and width ``b = 0.1`` m.
    failure: ``delta >= 0.075`` m, the deflection limit.
    loss: linear loss above the limit, ``2e6`` per metre of excess.
    stage: one recorded analysis of the study (propagation, sensitivity,
        reliability, risk, emulation) with its own child seed and cost.

Assumptions:
    - Load is lognormal, modulus and thickness are normal, and the three are
      independent; the normal thickness has negligible mass at or below 0.
    - The Euler-Bernoulli formula is the true model; there is no model-form
      error and no data, so every result is a forward (prior) result.
    - FORM and SORM assume a single dominant design point; the sampling
      methods do not.

Equations:
    EQ-UQ-001: the named forward model.
    EQ-MC-MEAN-001: mean deflection from the propagated sample.
    EQ-SENS-FIRST-001: first-order Sobol indices.
    EQ-SENS-TOTAL-001: total-effect indices and the ranking.
    EQ-SENS-BOOT-001: bootstrap intervals of the indices.
    EQ-REL-PF-001: failure probability.
    EQ-REL-FORM-002: FORM index and probability.
    EQ-REL-SORM-002: Breitung SORM correction.
    EQ-REL-SORM-003: Hohenbichler-Rackwitz SORM correction.
    EQ-REL-ISDP-001: importance sampling at the design point.
    EQ-REL-SUBSET-001: subset simulation.
    EQ-RISK-LOSS-001: linear loss function.
    EQ-RISK-EL-001: expected loss.
    EQ-RISK-VAR-001: value-at-risk.
    EQ-RISK-CVAR-001: conditional value-at-risk.
    EQ-SUR-PCE-005: least-squares PCE fit of the emulation stage.

Diagnostics:
    - Six reliability methods on one limit state: agreement of FORM, SORM,
      importance sampling, subset simulation, and crude Monte Carlo
      (``7.6e-3`` to ``8.9e-3``) is the evidence that the probability is
      right; the crude Monte Carlo row is the unbiased reference.
    - The emulator Monte Carlo probability against the true-model rows.
    - The Markdown and HTML reports list every stage with its seed, cost,
      warnings, and figure.

Cost and error:
    - True-model evaluations per stage: propagation 4096, sensitivity 5120
      (1024 rows, five blocks), reliability 24 829 (FORM 49, SORM 18 more,
      importance sampling 2000, subset simulation 2762, crude MC 20 000),
      risk 20 000, emulation 70; 54 115 in total.
    - Crude MC with 20 000 runs at ``P_f`` near ``8e-3`` has a relative
      standard error of about 8%; every other row lies within two of its
      standard errors. FORM and SORM carry approximation error instead.
    - The emulator study spends 200 000 surrogate evaluations and no
      true-model runs; its probability carries emulator error on top of a
      relative Monte Carlo error of about 2.5%.

References:
    REF-SMITH-UQ-2024: uncertainty quantification workflow.
    REF-MELCHERS-BECK-2018: structural reliability methods.
    REF-SALTELLI-2010: Sobol index estimators.
    REF-AU-BECK-2001: subset simulation.
    REF-ROCKAFELLAR-URYASEV-2002: value-at-risk and conditional value-at-risk.
    REF-SUDRET-2007: polynomial chaos emulators.

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import LogNormal, Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.risk import linear_loss
from probstat.uq import QuantityOfInterest
from probstat.visualization.study import save_study_figures, write_study_html
from probstat.workflow import UQStudy

LENGTH = 2.0
WIDTH = 0.10
LIMIT = 0.075
# Root seed of the study; every stage derives its own child stream.
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
    load = parameters["load"]
    modulus = parameters["modulus"]
    thickness = parameters["thickness"]
    deflection = 4.0 * load * LENGTH**3 / (modulus * WIDTH * thickness**3)
    return {"deflection": deflection}


def main() -> None:
    """Run every study stage and write the report and figures.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-MATPLOTLIB].

    """
    inputs = JointInput.independent(
        RandomInput("load", LogNormal(7.6, 0.2), unit="N"),
        RandomInput("modulus", Normal(70e9, 5e9), unit="Pa"),
        RandomInput("thickness", Normal(0.06, 0.002), unit="m"),
    )
    model = ProbabilisticModel(
        inputs,
        cantilever,
        (QuantityOfInterest("deflection", "tip deflection", "m"),),
        name="cantilever",
    )
    study = UQStudy(model, seed=SEED)
    forward = study.propagate(4096, design="sobol")
    print(f"mean deflection={forward['deflection'].mean * 1e3:.3f} mm")
    sobol = study.sensitivity("deflection", 1024, bootstrap=200)
    print(f"ranking={sobol.ranking()}")
    comparison = study.reliability(
        "deflection", LIMIT, failure="above", monte_carlo_samples=20_000
    )
    for row in comparison.rows:
        print(f"{row.method:14s} pf={row.probability:.3e} runs={row.evaluations}")
    study.risk("deflection", linear_loss(LIMIT, 2.0e6), 20_000)
    emulated = study.emulate("deflection", method="pce", degree=4)
    cheap = emulated.reliability(
        "deflection",
        LIMIT,
        failure="above",
        methods=("mc",),
        monte_carlo_samples=200_000,
    )
    print(f"emulator MC pf={cheap.row('MC').probability:.3e} (200,000 emulator runs)")

    output = Path(__file__).parent / "generated"
    output.mkdir(parents=True, exist_ok=True)
    report = study.report()
    (output / "cantilever-study-report.md").write_text(report, encoding="utf-8")
    paths = save_study_figures(study, output)
    page = write_study_html(study, output / "cantilever-study.html")
    print(f"HTML report: {page.name}")
    print(f"report: {len(report.splitlines())} lines; figures: {len(paths)}")
    print(f"true-model evaluations: {study.total_evaluations:,}")


if __name__ == "__main__":
    main()
