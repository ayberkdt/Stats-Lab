"""Reproducible UQ-3 laboratory for variance reduction and efficiency."""

from __future__ import annotations

import math
import random
from pathlib import Path

from probstat.monte_carlo import (
    allocate_stratified_budget,
    compare_efficiency,
    estimate_antithetic_uniform,
    estimate_conditional_expectation,
    estimate_control_variate,
    estimate_importance_sampling,
    estimate_mean,
    estimate_stratified_uniform,
)
from probstat.visualization import plot_efficiency_comparison


def main() -> None:
    """Compare five estimators of ``E[U^2]`` under one evaluation budget.

    A second experiment estimates ``P(X+Y>1)`` by raw indicators and by the
    exact conditional expectation ``P(Y>1-X | X)=X``. The script prints
    uncertainty, cost, efficiency, allocation, and importance-weight diagnostics
    and saves a cost-normalized comparison figure.

    References:
        [EQ-MC-ANTI-001], [EQ-MC-CONTROL-001], [EQ-MC-COND-001],
        [EQ-MC-STRAT-001], [EQ-MC-IS-001], [EQ-MC-NEYMAN-001],
        [EQ-MC-EFF-001], [REF-OWEN-2013], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    budget = 20_000
    generator = random.Random(2026)
    uniforms = tuple(generator.random() for _ in range(budget))
    squares = tuple(value * value for value in uniforms)
    crude = estimate_mean(squares)
    antithetic = estimate_antithetic_uniform(
        lambda value: value * value,
        pairs=budget // 2,
        seed=2026,
    )
    control = estimate_control_variate(
        squares,
        uniforms,
        known_control_mean=0.5,
    )
    allocations = allocate_stratified_budget(
        (0.08, 0.12, 0.20, 0.35, 0.55),
        budget,
    )
    stratified = estimate_stratified_uniform(
        lambda value: value * value,
        allocations,
        seed=2026,
    )
    importance = estimate_importance_sampling(
        lambda value: value * value,
        lambda value: 1.0 if 0.0 <= value <= 1.0 else 0.0,
        lambda value: 2.0 * value if 0.0 <= value <= 1.0 else 0.0,
        lambda local_generator: math.sqrt(local_generator.random()),
        sample_size=budget,
        seed=2026,
    )

    comparisons = (
        compare_efficiency(
            crude,
            antithetic,
            baseline_label="crude uniform MC",
            candidate_label="antithetic",
        ),
        compare_efficiency(
            crude,
            control.estimate,
            baseline_label="crude uniform MC",
            candidate_label="control variate",
        ),
        compare_efficiency(
            crude,
            stratified,
            baseline_label="crude uniform MC",
            candidate_label="stratified",
        ),
        compare_efficiency(
            crude,
            importance,
            baseline_label="crude uniform MC",
            candidate_label="importance",
        ),
    )

    conditional_generator = random.Random(17)
    first = tuple(conditional_generator.random() for _ in range(budget))
    second = tuple(conditional_generator.random() for _ in range(budget))
    raw_probability = estimate_mean(
        float(left + right > 1.0)
        for left, right in zip(first, second, strict=True)
    )
    conditional_probability = estimate_conditional_expectation(first)
    conditional_gain = compare_efficiency(
        raw_probability,
        conditional_probability,
        baseline_label="raw indicator",
        candidate_label="conditional MC",
    )

    print("Target: E[U^2] = 1/3")
    print(
        f"crude={crude.estimate:.6f}, SE={crude.standard_error:.6f}, "
        f"evaluations={crude.evaluations}"
    )
    print(
        f"antithetic={antithetic.estimate:.6f}, "
        f"SE={antithetic.standard_error:.6f}, evaluations={antithetic.evaluations}"
    )
    print(
        f"control={control.estimate.estimate:.6f}, "
        f"SE={control.estimate.standard_error:.6f}, beta={control.coefficient:.6f}"
    )
    print(
        f"stratified={stratified.estimate:.6f}, "
        f"SE={stratified.standard_error:.6f}, allocations={allocations}"
    )
    print(
        f"importance={importance.estimate.estimate:.6f}, "
        f"SE={importance.standard_error:.6f}, "
        f"weight ESS={importance.effective_sample_size:.1f}, "
        f"max normalized weight={importance.max_normalized_weight:.6f}"
    )
    for comparison in comparisons:
        print(
            f"{comparison.candidate_label} efficiency gain: "
            f"{comparison.efficiency_gain:.3f}x"
        )
    print(
        f"conditional MC probability={conditional_probability.estimate:.6f}, "
        f"efficiency gain={conditional_gain.efficiency_gain:.3f}x"
    )

    axes = plot_efficiency_comparison(comparisons)
    axes.figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    axes.figure.savefig(output / "advanced-monte-carlo-efficiency.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
