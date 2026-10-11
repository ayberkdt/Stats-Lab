"""One model, one seed, one report: a complete UQStudy.

A cantilever beam tip deflection depends on load, Young's modulus, and
thickness. ``UQStudy`` propagates the inputs, ranks them, estimates the
probability of exceeding a deflection limit with several methods, prices the
exceedance risk, fits a polynomial-chaos emulator, and writes a Markdown report
plus one figure per stage.
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
    study = UQStudy(model, seed=2026)
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
