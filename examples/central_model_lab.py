"""One model, many questions: the central ProbabilisticModel laboratory.

Defines a cantilever-bracket model once and asks it every forward question:
propagation under four sampling designs, output distribution and exceedance,
Sobol and Morris sensitivity. Later labs reuse the same object for
reliability, risk, and calibration. The same model then prices a repair-cost
loss (expected loss, VaR, CVaR) and is checked by FORM against SORM,
importance sampling, subset simulation, and crude Monte Carlo. Run with
``PYTHONPATH=src python examples/central_model_lab.py``; seven figures are
written to ``examples/generated/``.

Definitions:
    ``margin = strength - load / cos(angle)``: capacity minus demand in kN;
    failure is ``margin <= 0``.
    Inputs: ``load ~ LogNormal(0.9, 0.25)`` kN, ``strength ~ Normal(4, 0.4)``
    kN, ``angle ~ Uniform(0, pi/4)`` rad; load and strength have Pearson
    correlation 0.3 through a Gaussian copula.
    ``S_i``, ``S_Ti``: first-order and total Sobol indices; ``mu*``: Morris
    mean absolute elementary effect.
    Loss: repair cost ``25 + 40 (-margin)`` kEUR when the margin is negative.
    ``beta``: FORM reliability index at the design point ``u*``.

Assumptions:
    - The Gaussian copula with the Nataf transformation represents the
      input dependence; the latent correlation is solved from the Pearson
      value.
    - Sobol indices and Morris screening assume independent inputs, so they
      are computed on an independent-input variant of the model.
    - FORM and SORM linearize or curve-fit the limit state at one design
      point; their errors are judged against the sampling methods.

Equations:
    EQ-UQ-001: forward model ``Y = g(X)``.
    EQ-MOD-NATAF-002: latent correlation from the Pearson correlation.
    EQ-MOD-NATAF-001: Nataf map between physical and standard-normal space.
    EQ-SAMP-INV-001: inverse-CDF mapping of design points.
    EQ-SAMP-RQMC-001: randomized quasi-Monte Carlo designs.
    EQ-STAT-QUANT-001: sample quantiles of the margin.
    EQ-PROP-THRESH-001: exceedance probability of the margin.
    EQ-SENS-FIRST-001: first-order Sobol index.
    EQ-SENS-TOTAL-001: total Sobol index.
    EQ-SENS-BOOT-001: bootstrap intervals of the Sobol indices.
    EQ-SENS-MORRIS-001: Morris elementary effects.
    EQ-RISK-LOSS-001: loss sample from the custom repair-cost function.
    EQ-RISK-EL-001: expected loss.
    EQ-RISK-VAR-001: value at risk.
    EQ-RISK-CVAR-001: conditional value at risk.
    EQ-REL-MLS-001: limit state from the margin QoI.
    EQ-REL-HLRF-001: HL-RF design-point search.
    EQ-REL-FORM-002: FORM probability and importance factors.
    EQ-REL-SORM-002: Breitung SORM correction.
    EQ-REL-SORM-003: Hohenbichler-Rackwitz SORM correction.
    EQ-REL-ISDP-001: importance sampling at the design point.
    EQ-REL-SUBSET-001: subset simulation.
    EQ-REL-PF-001: crude Monte Carlo failure probability.
    EQ-REL-BETA-001: generalized reliability index of every method.

Diagnostics:
    - Four sampling designs are replicated 10 times at 256 points; the
      spread of their means shows the variance reduction of LHS and RQMC.
    - Sobol indices carry 200-replicate bootstrap intervals; ``S`` close to
      ``S_T`` shows small interactions.
    - The reliability table puts every method beside crude Monte Carlo with
      its run count and c.o.v.; FORM reports convergence.

Cost and error:
    - Propagation: 4096 Sobol points; designs: 10 x 256 per design; Sobol
      indices: 20 480 runs; Morris: 160 runs; risk: 20 000 LHS runs; FORM:
      56 runs; the printed run counts are exact.
    - Sampling estimates carry Monte Carlo error (the printed c.o.v. for
      IS, subset simulation, and MC); FORM and SORM carry a deterministic
      approximation error, visible as their gap to the sampling estimates.

References:
    REF-SMITH-UQ-2024: forward UQ with a probabilistic model.
    REF-LIU-DERKIUREGHIAN-1986: the Nataf model.
    REF-L-ECUYER-2018: randomized quasi-Monte Carlo.
    REF-SALTELLI-2010: Sobol index estimators.
    REF-MORRIS-1991: elementary-effects screening.
    REF-ROCKAFELLAR-URYASEV-2002: conditional value at risk.
    REF-KAPLAN-GARRICK-1981: quantitative risk.
    REF-HASOFER-LIND-1974: the reliability index.
    REF-RACKWITZ-FIESSLER-1978: the HL-RF iteration.
    REF-BREITUNG-1984: asymptotic SORM.
    REF-AU-BECK-2001: subset simulation.
    REF-MELCHERS-BECK-2018: structural reliability.

"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import compare_designs, propagate
from probstat.reliability import (
    ModelLimitState,
    compare_reliability_methods,
    form_hlrf,
)
from probstat.risk import assess_risk, custom_loss
from probstat.sensitivity import morris_screening, sobol_analysis
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.uq import (
    plot_design_comparison,
    plot_form_geometry,
    plot_loss_exceedance,
    plot_output_distribution,
    plot_reliability_comparison,
    plot_sobol_indices,
)

# Seed shared by every sampling analysis of the bracket model.
SEED = 2026


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

    study = propagate(model, 4096, design="sobol", seed=SEED)
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

    designs = compare_designs(model, "margin", 256, replicates=10, seed=SEED)
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
    sobol = sobol_analysis(independent, "margin", 4096, seed=SEED, bootstrap=200)
    for name, first, total in zip(
        sobol.input_names, sobol.first_order, sobol.total_effect, strict=True
    ):
        print(
            f"{name:9s} S={first.estimate:.3f} [{first.lower:.3f}, {first.upper:.3f}] "
            f"ST={total.estimate:.3f} [{total.lower:.3f}, {total.upper:.3f}]"
        )
    print(f"Sobol evaluations={sobol.evaluations}")
    morris = morris_screening(independent, "margin", 40, seed=SEED)
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
        seed=SEED,
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

    state = ModelLimitState(model, "margin", 0.0, failure="below")
    form = form_hlrf(state)
    print(
        f"FORM beta={form.beta:.4f} pf={form.probability:.5f} "
        f"evaluations={form.evaluations} converged={form.converged}"
    )
    print(f"importance factors={dict(form.importance_factors)}")
    comparison = compare_reliability_methods(state, seed=SEED)
    for row in comparison.rows:
        print(
            f"{row.method:10s} pf={row.probability:.5f} beta={row.beta:.3f} "
            f"runs={row.evaluations} cov={row.coefficient_of_variation}"
        )
    third = form.design_point_u[2]
    generator = random.Random(SEED)
    cloud = [
        (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)) for _ in range(1500)
    ]
    axes = plot_form_geometry(
        lambda u1, u2: state.standard((u1, u2, third)),
        form.design_point_u[:2],
        form.beta,
        axis_names=("u_load", "u_strength"),
        samples=cloud,
    )
    save_figure(axes.figure, output / "central-model-form-geometry.png")
    plt.close(axes.figure)
    mc_row = comparison.row("MC")
    axes = plot_reliability_comparison(
        comparison.rows,
        reference=mc_row.probability,
        reference_label="crude MC",
    )
    save_figure(axes.figure, output / "central-model-reliability-methods.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
