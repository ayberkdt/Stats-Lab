"""Reproducible UQ-2 laboratory for integration and forward propagation.

Three experiments: ``pi`` from the quarter-circle hit probability of 20 000
uniform points, the integral of ``x^2`` on ``[0, 1]`` by Simpson's rule and
by Monte Carlo, and the range of an idealized projectile with uncertain
speed and launch angle propagated by 5000 samples. Run with
``PYTHONPATH=src python examples/basic_monte_carlo_lab.py``; the ``pi``
convergence figure is written to ``examples/generated/``.

Definitions:
    Hit indicator: ``I(U1^2 + U2^2 <= 1)`` for independent uniforms; its mean
    is ``pi / 4``.
    Convergence trace: the running estimate and standard error at
    increasing sample sizes.
    Projectile model: ``R = v^2 sin(2 theta) / g`` with
    ``v ~ Normal(25, 1.5)`` m/s and ``theta ~ Uniform(0.55, 0.85)`` rad.
    Exceedance probability: ``P(R > 65 m)``.

Assumptions:
    - Draws are independent and the contributions have finite variance, so
      the error falls as ``1 / sqrt(N)``.
    - The projectile model ignores drag and launches and lands at one height.
    - Speed and angle are independent.

Equations:
    EQ-MC-PROB-001: hit-probability estimator and its standard error.
    EQ-MC-MEAN-001: sample mean of the ``4 I`` contributions.
    EQ-MC-SE-001: standard error ``s / sqrt(N)``.
    EQ-MC-ROOTN-001: the ``1 / sqrt(N)`` rate shown by the trace.
    EQ-MC-INT-001: integral as ``(b - a) E[f(U)]``.
    EQ-NUM-SIMP-001: composite Simpson rule with 100 subintervals.
    EQ-LAB-PROJECTILE-001: projectile range model.
    EQ-UQ-001: forward model ``Y = g(X)`` propagated by sampling.
    EQ-PROP-THRESH-001: exceedance probability ``P(Y > S)``.

Diagnostics:
    - The ``pi`` estimate carries a Wilson interval on the hit probability,
      and the convergence figure shows the trace against the exact value.
    - Simpson's rule is exact for ``x^2`` and serves as the reference for the
      Monte Carlo integral and its standard error.

Cost and error:
    - ``pi``: 20 000 points; integral: 101 integrand evaluations (Simpson)
      against 20 000 (Monte Carlo); projectile: 5000 model evaluations.
    - Monte Carlo numbers carry the printed standard error; Simpson's rule
      has zero truncation error for a quadratic integrand, so only rounding
      remains.

References:
    REF-ROBERT-CASELLA-2004: Monte Carlo estimation and error.
    REF-WILSON-1927: the score interval for a proportion.
    REF-DAVIS-RABINOWITZ-1984: Simpson's rule.
    REF-HALLIDAY-RESNICK-WALKER-2014: projectile motion.
    REF-SMITH-UQ-2024: forward propagation.

"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import Normal, Uniform
from probstat.monte_carlo import (
    convergence_trace,
    estimate_mean,
    estimate_probability,
    estimate_uniform_integral,
)
from probstat.numerics import composite_simpson
from probstat.propagation import propagate_samples
from probstat.uq import QuantityOfInterest
from probstat.visualization import plot_monte_carlo_convergence, save_figure

# Seed of the quarter-circle points and of the Monte Carlo integral.
SEED = 2026


def projectile_model(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return idealized projectile range for speed and launch angle.

    Args:
        parameters: Named ``speed`` and radian ``angle`` values.

    Returns:
        Mapping containing horizontal ``range`` in meters.

    Equation:
        [EQ-LAB-PROJECTILE-001].

    References:
        [REF-HALLIDAY-RESNICK-WALKER-2014, projectile motion]; used only as a
        transparent teaching model for [EQ-UQ-001].

    """
    speed = parameters["speed"]
    angle = parameters["angle"]
    return {"range": speed**2 * math.sin(2.0 * angle) / 9.81}


def main() -> None:
    """Run pi, integration, and projectile-propagation experiments.

    The lab prints deterministic cost and uncertainty metadata and writes a
    convergence figure. Every pseudo-random stream is call-local and seeded.

    References:
        [EQ-MC-INT-001], [EQ-MC-ROOTN-001], [EQ-UQ-001],
        [EQ-PROP-THRESH-001], [REF-ROBERT-CASELLA-2004], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    generator = random.Random(SEED)
    inside = tuple(
        generator.random() ** 2 + generator.random() ** 2 <= 1.0
        for _ in range(20_000)
    )
    pi_probability = estimate_probability(inside)
    pi_contributions = tuple(4.0 if event else 0.0 for event in inside)
    pi_estimate = estimate_mean(pi_contributions)
    pi_trace = convergence_trace(
        pi_contributions,
        (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 20_000),
    )

    quadrature = composite_simpson(lambda x: x**2, 0.0, 1.0, intervals=100)
    monte_carlo = estimate_uniform_integral(
        lambda x: x**2,
        0.0,
        1.0,
        sample_size=20_000,
        seed=SEED,
    )

    sample_size = 5_000
    speeds = Normal(25.0, 1.5).sample(sample_size, seed=310)
    angles = Uniform(0.55, 0.85).sample(sample_size, seed=311)
    parameter_samples = tuple(
        {"speed": speed, "angle": angle}
        for speed, angle in zip(speeds, angles, strict=True)
    )
    range_qoi = QuantityOfInterest(
        "range",
        "Idealized horizontal range",
        "m",
    )
    propagation = propagate_samples(
        projectile_model,
        parameter_samples,
        (range_qoi,),
        thresholds={"range": 65.0},
    )
    range_summary = propagation.summaries["range"]

    print(f"pi estimate: {pi_estimate.estimate:.6f}")
    print(
        "pi Wilson interval from quarter-circle probability: "
        f"[{4*pi_probability.confidence_interval.lower:.6f}, "
        f"{4*pi_probability.confidence_interval.upper:.6f}]"
    )
    print(
        f"integral x^2 quadrature: {quadrature.value:.6f} "
        f"({quadrature.evaluations} evaluations)"
    )
    print(
        f"integral x^2 Monte Carlo: {monte_carlo.estimate:.6f} "
        f"(SE={monte_carlo.standard_error:.6f}, "
        f"{monte_carlo.evaluations} evaluations)"
    )
    print(
        f"projectile range mean={range_summary.mean:.6f} m, "
        f"variance={range_summary.variance:.6f} m^2"
    )
    assert range_summary.exceedance is not None
    print(
        f"P(range > 65 m)={range_summary.exceedance.probability:.6f}, "
        f"model evaluations={propagation.model_evaluations}"
    )

    axes = plot_monte_carlo_convergence(
        pi_trace, reference=math.pi, reference_label="exact pi"
    )
    axes.figure.tight_layout()
    output = Path(__file__).parent / "generated"
    save_figure(axes.figure, output / "monte-carlo-pi-convergence.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
