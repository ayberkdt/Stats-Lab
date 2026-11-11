"""Reproducible UQ-3 laboratory for variance reduction and efficiency.

Five estimators of ``E[U^2] = 1/3`` for ``U ~ Uniform(0, 1)`` share one
budget of 20 000 evaluations: crude Monte Carlo, antithetic pairs, a control
variate, stratified sampling with Neyman-type allocation, and importance
sampling. A second experiment estimates ``P(X + Y > 1)`` for independent
uniforms by raw indicators and by the conditional expectation
``P(Y > 1 - X | X) = X``. Run with
``PYTHONPATH=src python examples/advanced_monte_carlo_lab.py``; the
cost-normalized comparison figure is written to ``examples/generated/``.

Definitions:
    ``f(u) = u^2``: the integrand; the exact target is ``1/3``.
    Antithetic pair: ``(f(U) + f(1 - U)) / 2`` with one uniform per pair.
    Control variate: ``U`` itself, whose mean ``1/2`` is known exactly.
    Strata: five equal-width subintervals of ``[0, 1]`` with pilot standard
    deviations ``0.08, 0.12, 0.20, 0.35, 0.55``.
    Importance density: ``q(u) = 2u`` on ``[0, 1]``, sampled as ``sqrt(V)``.
    Efficiency gain: ratio of ``SE^2 x cost`` of the baseline to that of the
    candidate.

Assumptions:
    - All draws are independent; the antithetic estimator needs a monotone
      integrand, which ``u^2`` is, for its variance to fall.
    - The control mean is exact, so the fitted coefficient adds only a
      second-order bias.
    - Every estimator is charged its function evaluations; the cost of
      generating uniforms is ignored.

Equations:
    EQ-MC-MEAN-001: crude sample mean.
    EQ-MC-SE-001: standard error ``s / sqrt(N)``.
    EQ-MC-ANTI-001: antithetic estimator.
    EQ-MC-CONTROL-001: control-variate estimator with its fitted ``beta``.
    EQ-MC-NEYMAN-001: stratum allocation ``n_h proportional to W_h sigma_h``.
    EQ-MC-STRAT-001: stratified estimator and variance.
    EQ-MC-IS-001: importance-sampling identity ``E_p[g] = E_q[g w]``.
    EQ-MC-ESS-001: effective sample size of the importance weights.
    EQ-MC-COND-001: conditional Monte Carlo by the tower property.
    EQ-MC-EFF-001: work-normalized efficiency gain.

Diagnostics:
    - Every estimate is printed with its standard error against the exact
      value 1/3 (or 1/2 for ``P(X + Y > 1)``).
    - Importance sampling prints the weight ESS and the largest normalized
      weight; a weight dominating the sum would make its SE unreliable.
    - The printed allocation shows the strata that receive most samples.

Cost and error:
    - Each estimator uses 20 000 integrand evaluations (10 000 antithetic
      pairs count as 20 000); the conditional experiment draws 20 000 pairs.
    - The printed standard errors are the Monte Carlo errors; the efficiency
      gains are ratios of estimated variances and so carry sampling error of
      their own.

References:
    REF-OWEN-2013: variance reduction and efficiency.
    REF-ROBERT-CASELLA-2004: Monte Carlo estimators and errors.

"""

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
from probstat.visualization import plot_efficiency_comparison, save_figure

# Seed of the shared uniform sample and of the antithetic, stratified, and
# importance-sampling streams.
SEED = 2026


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
    generator = random.Random(SEED)
    uniforms = tuple(generator.random() for _ in range(budget))
    squares = tuple(value * value for value in uniforms)
    crude = estimate_mean(squares)
    antithetic = estimate_antithetic_uniform(
        lambda value: value * value,
        pairs=budget // 2,
        seed=SEED,
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
        seed=SEED,
    )
    importance = estimate_importance_sampling(
        lambda value: value * value,
        lambda value: 1.0 if 0.0 <= value <= 1.0 else 0.0,
        lambda value: 2.0 * value if 0.0 <= value <= 1.0 else 0.0,
        lambda local_generator: math.sqrt(local_generator.random()),
        sample_size=budget,
        seed=SEED,
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
    save_figure(axes.figure, output / "advanced-monte-carlo-efficiency.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
