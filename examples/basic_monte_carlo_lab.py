"""Reproducible UQ-2 laboratory for integration and forward propagation."""

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
from probstat.visualization import plot_monte_carlo_convergence


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

    generator = random.Random(2026)
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
        seed=2026,
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

    axes = plot_monte_carlo_convergence(pi_trace, reference=math.pi)
    axes.figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    axes.figure.savefig(output / "monte-carlo-pi-convergence.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
