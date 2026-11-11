"""Reproducible UQ-4 laboratory comparing reliability estimators.

An affine limit state ``g(u) = 3 - u_1 - u_2`` in two independent standard
normals has an exact failure probability, so direct Monte Carlo, FORM, and
directional sampling can be judged against it. A one-dimensional five-sigma
margin ``g(u) = 5 - u`` shows importance sampling where direct sampling would
see no failures. Run with ``PYTHONPATH=src python examples/reliability_lab.py``;
the method-comparison figure is written to ``examples/generated/``.

Definitions:
    ``g(u)``: limit-state function; failure is ``g(u) <= 0``.
    ``P_f``: failure probability ``P(g(U) <= 0)``.
    ``beta``: Hasofer-Lind reliability index, the distance from the origin to
        the design point in standard-normal space.
    ``q``: importance proposal ``N(3, 1)`` for the five-sigma margin.

Assumptions:
    - Inputs are independent standard normals, so FORM is exact for the
      affine limit state and directional sampling integrates the radial
      chi-square law exactly along each ray.
    - The importance proposal covers the failure region ``u >= 5`` and the
      likelihood ratios ``phi(u) / phi(u - 3)`` are exact.

Equations:
    EQ-REL-PF-001: failure probability as an indicator expectation.
    EQ-MC-PROB-001: direct Monte Carlo event probability and its standard
        error.
    EQ-REL-FORM-001: affine FORM index, design point, and ``Phi(-beta)``.
    EQ-REL-DIR-001: directional estimator with the radial chi-square tail.
    EQ-REL-IS-001: importance-weighted failure estimator.

Diagnostics:
    - Every sampling estimate is compared with the exact affine value
      ``Phi(-3 / sqrt(2))``; the figure shows each 95% interval against it.
    - The five-sigma estimate is compared with ``Phi(-5)``, and the number of
      proposal samples that fail (2194 of 100 000) shows the proposal reaches
      the failure region.

Cost and error:
    - Direct Monte Carlo: 100 000 limit-state evaluations; relative standard
      error about ``sqrt((1 - P_f) / (N P_f))``, roughly 2.4% here.
    - Directional sampling: 30 000 directions, one root per direction; its
      variance is lower than direct sampling for this smooth boundary.
    - FORM: closed form, no sampling error, exact for an affine ``g``.
    - Importance sampling: 100 000 proposal draws; the estimate has a
      relative standard error of about 2.5% (not printed) and lies 1.8
      standard errors below ``Phi(-5)``, inside its 95% interval.

References:
    REF-MELCHERS-BECK-2018: structural reliability methods.
    REF-OWEN-2013: Monte Carlo and importance sampling.
    REF-WILSON-1927: score interval for a binomial proportion.

"""

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

# Seed of the direct and directional samples; the proposal uses SEED + 1.
SEED = 2026


def main() -> None:
    """Compare direct MC, affine FORM, directional, and rare-event importance MC.

    References:
        [EQ-REL-PF-001], [EQ-REL-IS-001], [EQ-REL-FORM-001],
        [EQ-REL-DIR-001], [REF-MELCHERS-BECK-2018], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    state = LimitState(lambda vector: 3.0 - vector[0] - vector[1], "sum margin", 2)
    generator = random.Random(SEED)
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
        seed=SEED,
    )

    rare_state = LimitState(lambda vector: 5.0 - vector[0], "five-sigma margin", 1)
    proposal = random.Random(SEED + 1)
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
