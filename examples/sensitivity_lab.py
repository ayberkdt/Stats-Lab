"""Reproducible UQ-5 laboratory for local and Sobol sensitivity analysis.

The Ishigami function is differentiated at one point by central differences
and decomposed globally by Saltelli pick-freeze Sobol estimators. Run with
``PYTHONPATH=src python examples/sensitivity_lab.py``; a first-versus-total
index figure is written to ``examples/generated/``.

Definitions:
    ``Y = f(X)``: Ishigami response with ``a = 7`` and ``b = 0.1``.
    ``S_i``: first-order Sobol index, the variance share of ``X_i`` alone.
    ``S_Ti``: total-effect index, every variance share involving ``X_i``.
    ``S_ij``: second-order interaction index of a pair.
    fix candidate: an input whose total effect is at most the screening
        threshold 0.02.

Assumptions:
    - The three inputs are independent and uniform on ``[-pi, pi]``, the
      condition under which the ANOVA decomposition is unique.
    - The step ``h = 1e-5`` is small enough for the central-difference
      truncation error and large enough to avoid round-off dominance.

Equations:
    EQ-LAB-ISHIGAMI-001: the Ishigami benchmark function.
    EQ-SENS-LOCAL-001: central finite-difference gradient.
    EQ-SENS-ANOVA-001: functional ANOVA decomposition behind the indices.
    EQ-SENS-FIRST-001: first-order Sobol index.
    EQ-SENS-SECOND-001: second-order interaction index.
    EQ-SENS-TOTAL-001: total-effect index and the ranking.
    EQ-SENS-DECISION-001: total-effect screening rule.

Diagnostics:
    - The estimates can be compared with the analytic Ishigami indices
      ``S = (0.3139, 0.4424, 0)`` and ``S_T = (0.5576, 0.4424, 0.2437)``.
    - ``S_3`` near zero with ``S_T3`` near 0.25 and ``S_13`` near 0.23
      identifies ``X_3`` as acting only through its interaction with ``X_1``.
    - The empty fix-candidate set shows that no input can be frozen at the
      0.02 threshold.

Cost and error:
    - Local analysis: 6 model evaluations; truncation error ``O(h^2)``.
    - Global analysis: two base samples of 50 000 rows and their hybrids,
      400 000 model evaluations in total. The indices carry Monte Carlo
      error of order ``1 / sqrt(50 000)`` (no interval is printed); the
      first-order estimate of ``X_1`` is 0.007 below its analytic value.

References:
    REF-SALTELLI-2008: global sensitivity analysis primer.
    REF-SALTELLI-2010: pick-freeze estimators for first and total indices.
    REF-SOBOL-1993: Sobol variance decomposition.
    REF-FORNBERG-1988: finite-difference formulas.

"""

from __future__ import annotations

import math
import random
from pathlib import Path

from probstat.sensitivity import central_difference_gradient, estimate_sobol_indices
from probstat.visualization import save_figure
from probstat.visualization.uq import plot_sobol_indices

# Seed of the two base samples of the Sobol design.
SEED = 2026


def ishigami(parameters: tuple[float, ...]) -> float:
    """Evaluate the three-input Ishigami interaction benchmark.

    Args:
        parameters: Three coordinates independently uniform on ``[-pi, pi]``.

    Returns:
        Ishigami response with conventional ``a=7`` and ``b=0.1``.

    Equation:
        [EQ-LAB-ISHIGAMI-001].

    References:
        [REF-SALTELLI-2008, Chapter 4].

    """
    first, second, third = parameters
    return (
        math.sin(first)
        + 7.0 * math.sin(second) ** 2
        + 0.1 * third**4 * math.sin(first)
    )


def main() -> None:
    """Run local and global analyses and save a first-versus-total plot.

    References:
        [EQ-SENS-LOCAL-001], [EQ-SENS-FIRST-001], [EQ-SENS-SECOND-001],
        [EQ-SENS-TOTAL-001], [EQ-SENS-DECISION-001],
        [REF-SALTELLI-2008], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    local = central_difference_gradient(ishigami, (0.2, 0.5, -1.0), (1e-5,) * 3)
    generator = random.Random(SEED)
    sample_size = 50_000
    sample_a = tuple(
        tuple(generator.uniform(-math.pi, math.pi) for _ in range(3))
        for _ in range(sample_size)
    )
    sample_b = tuple(
        tuple(generator.uniform(-math.pi, math.pi) for _ in range(3))
        for _ in range(sample_size)
    )
    global_result = estimate_sobol_indices(ishigami, sample_a, sample_b)
    decision = global_result.decision(threshold=0.02)

    print(f"local gradient={local.gradient}, evaluations={local.evaluations}")
    print(f"first-order={global_result.first_order}")
    print(f"total-effect={global_result.total_effect}")
    print(f"S_13={global_result.second_order[0][2]:.6f}")
    print(f"ranking={global_result.ranking()}")
    print(f"fix candidates={decision.fix_candidates}")
    print(f"model evaluations={global_result.evaluations}")

    axes = plot_sobol_indices(
        # The input names of every Ishigami figure of the set (ADR-0098).
        ("x1", "x2", "x3"),
        global_result.first_order,
        global_result.total_effect,
    )
    save_figure(
        axes.figure, Path(__file__).parent / "generated" / "ishigami-sobol-indices.png"
    )
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
