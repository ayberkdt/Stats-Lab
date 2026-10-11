"""Reproducible UQ-4 laboratory comparing reliability estimators."""

from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import NormalDist

from probstat.reliability import (
    LimitState,
    estimate_directional_failure_probability,
    estimate_failure_probability,
    estimate_weighted_failure_probability,
    linear_normal_form,
)
from probstat.visualization import save_figure
from probstat.visualization.uq import plot_method_comparison


def main() -> None:
    """Compare direct MC, affine FORM, directional, and rare-event importance MC.

    References:
        [EQ-REL-PF-001], [EQ-REL-IS-001], [EQ-REL-FORM-001],
        [EQ-REL-DIR-001], [REF-MELCHERS-BECK-2018], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    state = LimitState(lambda vector: 3.0 - vector[0] - vector[1], "sum margin", 2)
    generator = random.Random(2026)
    samples = tuple(
        (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
        for _ in range(100_000)
    )
    direct = estimate_failure_probability(state, samples)
    form = linear_normal_form(3.0, (-1.0, -1.0))

    def radial_boundary(direction: tuple[float, ...]) -> float | None:
        """Return the affine failure-entry radius on one unit ray.

        References:
            [EQ-REL-DIR-001], [REF-MELCHERS-BECK-2018, Chapter 6].

        """
        projection = direction[0] + direction[1]
        return None if projection <= 0.0 else 3.0 / projection

    directional = estimate_directional_failure_probability(
        radial_boundary,
        dimension=2,
        directions=30_000,
        seed=2026,
    )

    rare_state = LimitState(lambda vector: 5.0 - vector[0], "five-sigma margin", 1)
    proposal = random.Random(2027)
    rare_samples = tuple((proposal.gauss(3.0, 1.0),) for _ in range(100_000))
    likelihood_ratios = tuple(
        math.exp(-0.5 * value[0] ** 2 + 0.5 * (value[0] - 3.0) ** 2)
        for value in rare_samples
    )
    rare = estimate_weighted_failure_probability(
        rare_state,
        rare_samples,
        likelihood_ratios,
    )

    print(f"FORM beta={form.reliability_index:.6f}")
    print(f"FORM design point={form.design_point}")
    print(f"direct MC Pf={direct.failure.probability:.8f}, N={direct.evaluations}")
    print(
        f"directional Pf={directional.failure_probability:.8f}, "
        f"directions={directional.directions}"
    )
    print(f"FORM/exact affine Pf={form.failure_probability:.8f}")
    print(
        f"five-sigma importance Pf={rare.estimate.estimate:.10e}, "
        f"reference={NormalDist().cdf(-5.0):.10e}, "
        f"proposal failures={rare.observed_failures}"
    )

    axes = plot_method_comparison(
        (
            (
                "direct Monte Carlo",
                direct.failure.probability,
                direct.failure.confidence_interval.lower,
                direct.failure.confidence_interval.upper,
            ),
            (
                "directional sampling",
                directional.failure_probability,
                directional.estimate.confidence_interval.lower,
                directional.estimate.confidence_interval.upper,
            ),
            ("affine FORM (exact here)", form.failure_probability, None, None),
        ),
        reference=form.failure_probability,
        reference_label="exact Pf",
        quantity_label="Failure probability Pf",
        title="Sampling methods bracket the exact failure probability",
        subtitle="Dot: estimate. Whisker: 95% interval. FORM is exact for an "
        "affine limit state",
    )
    save_figure(
        axes.figure,
        Path(__file__).parent / "generated" / "reliability-method-comparison.png",
    )
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
