"""One model, many questions: the central ProbabilisticModel laboratory.

Defines a cantilever-bracket model once and asks it every forward question:
propagation under four sampling designs, output distribution and exceedance,
Sobol and Morris sensitivity. Later labs reuse the same object for
reliability, risk, and calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import compare_designs, propagate
from probstat.risk import assess_risk, custom_loss
from probstat.sensitivity import morris_screening, sobol_analysis
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.uq import (
    plot_design_comparison,
    plot_loss_exceedance,
    plot_output_distribution,
    plot_sobol_indices,
)


def bracket(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the safety margin of an idealized inclined bracket.

    Args:
        parameters: ``load`` and ``strength`` in kN, ``angle`` in radians.

    Returns:
        Mapping with the capacity-minus-demand ``margin`` in kN.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 1]; a transparent teaching model.

    """
    demand = parameters["load"] / math.cos(parameters["angle"])
    return {"margin": parameters["strength"] - demand}


def build_model() -> ProbabilisticModel:
    """Return the bracket model with correlated load and strength.

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-SMITH-UQ-2024, Chapter 3].

    """
    inputs = JointInput.from_pearson(
        (
            RandomInput("load", LogNormal(0.9, 0.25), unit="kN"),
            RandomInput("strength", Normal(4.0, 0.4), unit="kN"),
            RandomInput("angle", Uniform(0.0, math.pi / 4), unit="rad"),
        ),
        ((1.0, 0.3, 0.0), (0.3, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    return ProbabilisticModel(
        inputs,
        bracket,
        (QuantityOfInterest("margin", "capacity minus demand", "kN"),),
        name="bracket",
    )


def main() -> None:
    """Run the forward analyses and write explanatory figures.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-5], [REF-SALTELLI-2010], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    model = build_model()
    output = Path(__file__).parent / "generated"

    study = propagate(model, 4096, design="sobol", seed=2026)
    margin = study["margin"]
    print(f"mean margin={margin.mean:.4f} kN, sd={margin.standard_deviation:.4f} kN")
    levels = (0.05, 0.5, 0.95)
    print(f"quantiles {levels}={[round(margin.quantile(p), 4) for p in levels]}")
    failure = 1.0 - margin.exceedance(0.0).probability
    print(f"P(margin <= 0)={failure:.5f}")
    print(f"rank correlations={dict(study.rank_correlations('margin'))}")

    axes = plot_output_distribution(
        margin.values,
        name="Safety margin",
        unit="kN",
        threshold=0.0,
        failure_below=True,
    )
    save_figure(axes.figure, output / "central-model-margin-distribution.png")
    plt.close(axes.figure)

    designs = compare_designs(model, "margin", 256, replicates=10, seed=2026)
    for name, (center, spread) in designs.items():
        print(f"design={name:6s} mean={center:.5f} spread={spread:.2e}")
    axes = plot_design_comparison(designs, size=256)
    save_figure(axes.figure, output / "central-model-design-comparison.png")
    plt.close(axes.figure)

    independent = ProbabilisticModel(
        JointInput.independent(*model.inputs.inputs),
        bracket,
        model.quantities,
        name="bracket-independent",
    )
    sobol = sobol_analysis(independent, "margin", 4096, seed=2026, bootstrap=200)
    for name, first, total in zip(
        sobol.input_names, sobol.first_order, sobol.total_effect, strict=True
    ):
        print(
            f"{name:9s} S={first.estimate:.3f} [{first.lower:.3f}, {first.upper:.3f}] "
            f"ST={total.estimate:.3f} [{total.lower:.3f}, {total.upper:.3f}]"
        )
    print(f"Sobol evaluations={sobol.evaluations}")
    morris = morris_screening(independent, "margin", 40, seed=2026)
    print(f"Morris ranking={morris.ranking()}, evaluations={morris.evaluations}")
    axes = plot_sobol_indices(
        sobol.input_names,
        tuple(index.estimate for index in sobol.first_order),
        tuple(index.estimate for index in sobol.total_effect),
        title=f"{sobol.ranking()[0].capitalize()} dominates the margin variance",
        subtitle="Independent-input variant; interactions are small for this model",
    )
    save_figure(axes.figure, output / "central-model-sobol.png")
    plt.close(axes.figure)

    def repair_cost(margin: float) -> float:
        """Return repair cost in kEUR: 40 per kN of negative margin plus 25.

        References:
            [REF-KAPLAN-GARRICK-1981]; a teaching consequence model.

        """
        return 25.0 + 40.0 * (-margin) if margin < 0.0 else 0.0

    risk = assess_risk(
        model,
        "margin",
        custom_loss(repair_cost, "repair cost in kEUR"),
        20_000,
        design="lhs",
        seed=2026,
        levels=(0.95, 0.99),
    )
    print(f"expected loss={risk.expected_loss.estimate:.3f} kEUR")
    print(f"P(loss > 0)={risk.probability_of_loss:.4f}")
    for level in (0.95, 0.99):
        print(
            f"VaR{level:.0%}={risk.value_at_risk[level]:.2f} "
            f"CVaR{level:.0%}={risk.conditional_value_at_risk[level]:.2f}"
        )
    axes = plot_loss_exceedance(
        risk.losses,
        risk.value_at_risk,
        risk.conditional_value_at_risk,
        unit="kEUR",
    )
    save_figure(axes.figure, output / "central-model-loss-exceedance.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
