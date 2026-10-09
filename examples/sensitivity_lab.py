"""Reproducible UQ-5 laboratory for local and Sobol sensitivity analysis."""

from __future__ import annotations

import math
import random
from pathlib import Path

from probstat.sensitivity import central_difference_gradient, estimate_sobol_indices


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
    generator = random.Random(2026)
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

    positions = tuple(range(3))
    width = 0.36
    figure, axes = plt.subplots()
    axes.bar(
        tuple(position - width / 2 for position in positions),
        global_result.first_order,
        width,
        label="first-order",
        color="#4C78A8",
    )
    axes.bar(
        tuple(position + width / 2 for position in positions),
        global_result.total_effect,
        width,
        label="total-effect",
        color="#F58518",
    )
    axes.set_xticks(positions, ("X1", "X2", "X3"))
    axes.set(ylabel="Sobol index", title="Ishigami global sensitivity")
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.legend()
    figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "ishigami-sobol-indices.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
