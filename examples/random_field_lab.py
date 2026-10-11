"""Random-field laboratory: a bar whose stiffness varies along its length.

The log-stiffness is a Gaussian random field with a Matern-5/2 covariance.
A truncated Karhunen-Loeve expansion turns the field into a handful of
independent standard-normal coefficients, which become ordinary inputs of a
ProbabilisticModel; propagation and FORM then run unchanged.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path

from probstat.modeling import JointInput, ProbabilisticModel, karhunen_loeve_inputs
from probstat.processes import CovarianceKernel, karhunen_loeve_interval
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.processes_plots import (
    plot_field_realizations,
    plot_kl_spectrum,
)
from probstat.visualization.uq import plot_output_distribution
from probstat.workflow import UQStudy

MEAN_LOG_STIFFNESS = math.log(200.0)
LOAD = 1.0


def main() -> None:
    """Build the field model, propagate, run FORM, and write figures.

    References:
        [REF-GHANEM-SPANOS-1991], [REF-SUDRET-DERKIUREGHIAN-2000],
        [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    kernel = CovarianceKernel("matern52", variance=0.3**2, length_scale=0.3)
    expansion = karhunen_loeve_interval(
        kernel, 0.0, 1.0, 40, mean=0.0, variance_fraction=0.97
    )
    print(
        f"KL terms={expansion.terms}, explained variance="
        f"{expansion.explained_variance_fraction:.4f}"
    )
    inputs, field = karhunen_loeve_inputs(expansion, prefix="xi")
    weights = expansion.weights

    def bar(parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Return the tip displacement ``int P / E(x) dx`` of a unit bar.

        Args:
            parameters: KL coefficients ``xi_k``.

        Returns:
            Mapping with ``tip`` displacement.

        Equation:
            [EQ-MOD-RF-001].

        References:
            [REF-SUDRET-DERKIUREGHIAN-2000]; teaching model.

        """
        log_field = field(parameters)
        return {
            "tip": math.fsum(
                w * LOAD / math.exp(MEAN_LOG_STIFFNESS + g)
                for w, g in zip(weights, log_field, strict=True)
            )
        }

    model = ProbabilisticModel(
        JointInput.independent(*inputs),
        bar,
        (QuantityOfInterest("tip", "tip displacement", "mm"),),
        name="random-bar",
    )
    study = UQStudy(model, seed=2026)
    forward = study.propagate(4096, design="sobol")
    tip = forward["tip"]
    print(f"tip mean={tip.mean * 1e3:.4f}e-3, sd={tip.standard_deviation * 1e3:.4f}e-3")
    limit = tip.quantile(0.999)
    form = form_hlrf(ModelLimitState(model, "tip", limit, failure="above"))
    print(f"FORM beta={form.beta:.3f}, pf={form.probability:.2e} (target 1e-3)")
    leading = sorted(form.importance_factors.items(), key=lambda kv: -kv[1])[:3]
    print(f"leading modes={leading}")

    output = Path(__file__).parent / "generated"
    grid = [k / 199 for k in range(200)]
    generator = random.Random(2026)
    realizations = []
    for _ in range(30):
        xi = [generator.gauss(0.0, 1.0) for _ in range(expansion.terms)]
        realizations.append([expansion.realize_at(xi, x) for x in grid])
    axes = plot_field_realizations(
        grid,
        realizations,
        standard_deviation=[0.3] * len(grid),
        name="log-stiffness fluctuation",
    )
    save_figure(axes.figure, output / "random-field-realizations.png")
    plt.close(axes.figure)
    axes = plot_kl_spectrum(expansion.spectrum, expansion.terms)
    save_figure(axes.figure, output / "random-field-spectrum.png")
    plt.close(axes.figure)
    axes = plot_output_distribution(
        [value * 1e3 for value in tip.values], name="Tip displacement x 1e3"
    )
    save_figure(axes.figure, output / "random-field-tip.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
