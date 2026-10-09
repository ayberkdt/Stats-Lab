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

    labels = ("direct MC", "directional", "affine FORM")
    values = (
        direct.failure.probability,
        directional.failure_probability,
        form.failure_probability,
    )
    errors = (
        direct.failure.confidence_interval.upper - direct.failure.probability,
        directional.estimate.confidence_interval.upper
        - directional.failure_probability,
        0.0,
    )
    figure, axes = plt.subplots()
    axes.bar(
        labels,
        values,
        yerr=errors,
        capsize=5,
        color=("#4C78A8", "#54A24B", "#F58518"),
    )
    axes.axhline(form.failure_probability, color="#E45756", linestyle="--")
    axes.set(ylabel="Failure probability", title="Reliability method comparison")
    axes.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "reliability-method-comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
