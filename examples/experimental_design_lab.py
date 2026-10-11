"""Experimental design laboratory: D-optimal weights and greedy Bayesian runs.

The classical half finds the approximate D-optimal design of a quadratic
regression on ``[-1, 1]`` and certifies it with the Kiefer-Wolfowitz
equivalence theorem, against a uniform grid design for context. The Bayesian
half plans three runs of an exponential-decay experiment ``A exp(-k t)`` with
an uncertain rate by greedy maximization of the expected information gain.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from probstat.design import (
    ApproximateDesign,
    BayesianDesignProblem,
    d_efficiency,
    d_optimal_design,
    greedy_design,
    polynomial_basis,
)
from probstat.distributions import LogNormal, Normal
from probstat.inverse import NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.design_plots import (
    plot_greedy_design,
    plot_optimal_design,
)


def decay_model(parameters: dict[str, float]) -> dict[str, float]:
    """Return the exponential-decay signal ``A exp(-k t)``.

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007]. No new equation is introduced.

    """
    return {
        "y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])
    }


def main() -> None:
    """Compute both designs, print their certificates, and write two figures.

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-CHALONER-VERDINELLI-1995],
        [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    output = Path(__file__).parent / "generated"

    basis = polynomial_basis(2)
    grid = [round(-1.0 + 0.1 * i, 10) for i in range(21)]
    result = d_optimal_design(grid, basis)
    design = result.design.condensed(min_weight=1e-3, merge_distance=0.05)
    uniform = ApproximateDesign.uniform(grid)
    print(
        f"D-optimal quadratic design after {result.iterations} multiplicative "
        f"iterations: support {[p[0] for p in design.points]}, "
        f"weights {[round(w, 4) for w in design.weights]}"
    )
    print(
        f"  max d = {result.equivalence.max_variance:.6f} (p = 3), "
        f"efficiency bound {result.equivalence.efficiency_lower_bound:.6f}; "
        f"uniform grid D-efficiency {d_efficiency(uniform, design, basis):.3f}"
    )
    region = [-1.0 + 0.01 * i for i in range(201)]
    figure = plot_optimal_design(design, basis, region, comparison=uniform)
    save_figure(figure, output / "design-d-optimal.png")
    plt.close(figure)

    prior = JointInput.independent(
        RandomInput("A", Normal(3.0, 0.3)),
        RandomInput("k", LogNormal(math.log(0.5), 0.2)),
    )
    model = ProbabilisticModel(prior, decay_model, (QuantityOfInterest("y", "y"),))
    problem = BayesianDesignProblem(model, (NoiseModel("y", standard_deviation=0.1),))
    times = [0.25 * i for i in range(25)]
    greedy = greedy_design(
        problem, [{"t": t} for t in times], 3, method="laplace", samples=8, seed=2026
    )
    print(
        f"Greedy Laplace design: t = {[c['t'] for c in greedy.selected]}, "
        f"EIG path {[round(g, 4) for g in greedy.information_gains]} nats, "
        f"{greedy.model_evaluations} model evaluations"
    )
    axes = plot_greedy_design(greedy, times, factor="t")
    save_figure(axes.figure, output / "design-greedy-eig.png")
    plt.close(axes.figure)
    print(f"elapsed {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
