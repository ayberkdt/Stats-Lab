"""Surrogate laboratory: emulate the Ishigami function with polynomial chaos.

A degree-8 Legendre PCE is trained on a Latin-hypercube design, validated on
unseen points, and its coefficients give Sobol indices analytically. The same
emulator is then plugged back into a ProbabilisticModel and a UQStudy, so
every forward analysis can run on it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.surrogates import fit_polynomial_chaos
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.inference_plots import plot_surrogate_parity
from probstat.visualization.uq import plot_sobol_indices
from probstat.workflow import UQStudy

ANALYTIC_FIRST = (0.3139, 0.4424, 0.0)
ANALYTIC_TOTAL = (0.5576, 0.4424, 0.2437)


def ishigami(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Evaluate the Ishigami benchmark with a = 7 and b = 0.1.

    Args:
        parameters: Inputs ``x1``, ``x2``, ``x3`` on ``[-pi, pi]``.

    Returns:
        Mapping with output ``y``.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SALTELLI-2008, Chapter 4]; standard sensitivity benchmark.

    """
    x1, x2, x3 = parameters["x1"], parameters["x2"], parameters["x3"]
    return {"y": math.sin(x1) + 7.0 * math.sin(x2) ** 2 + 0.1 * x3**4 * math.sin(x1)}


def main() -> None:
    """Fit, validate, and reuse the emulator; write figures.

    References:
        [REF-SUDRET-2007, Chapter 4], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )
    model = ProbabilisticModel(
        inputs, ishigami, (QuantityOfInterest("y", "Ishigami output"),), name="ishigami"
    )
    expansion = fit_polynomial_chaos(
        model, "y", degree=8, basis="legendre", size=600, seed=2026
    )
    print(f"terms={expansion.size}, training runs={len(expansion.design)}")
    print(f"leave-one-out error={expansion.loo_error:.3e}")
    print(f"PCE mean={expansion.mean:.5f} (exact 3.5)")
    print(f"PCE variance={expansion.variance:.4f} (exact 13.8446)")
    indices = expansion.sobol_indices()
    for index, name in enumerate(indices.names):
        print(
            f"{name}: S={indices.first_order[name]:.4f} (exact {ANALYTIC_FIRST[index]})"
            f" ST={indices.total_order[name]:.4f} (exact {ANALYTIC_TOTAL[index]})"
        )

    validation = inputs.sample(400, seed=7)
    truth = [model.evaluate(point)["y"] for point in validation]
    predicted = [expansion.predict(point) for point in validation]
    output = Path(__file__).parent / "generated"
    axes = plot_surrogate_parity(truth, predicted, label="PCE")
    save_figure(axes.figure, output / "surrogate-parity.png")
    plt.close(axes.figure)
    axes = plot_sobol_indices(
        indices.names,
        tuple(indices.first_order[name] for name in indices.names),
        tuple(indices.total_order[name] for name in indices.names),
        subtitle="Read off the PCE coefficients: no extra model runs",
    )
    save_figure(axes.figure, output / "surrogate-sobol.png")
    plt.close(axes.figure)

    study = UQStudy(model, seed=2026, name="ishigami")
    emulated = study.emulate("y", method="pce", degree=8, size=600)
    forward = emulated.propagate(16_384, design="sobol")
    print(
        f"emulated propagation: mean={forward['y'].mean:.5f}, "
        f"P(y > 10)={forward['y'].exceedance(10.0).probability:.5f} "
        f"using {study.total_evaluations} true-model runs"
    )


if __name__ == "__main__":
    main()
