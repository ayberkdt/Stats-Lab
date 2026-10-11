"""Reproducible UQ-6 linear-Gaussian Bayesian calibration laboratory."""

from __future__ import annotations

import math
import random
from pathlib import Path

from probstat.inference import posterior_predictive, posterior_predictive_check
from probstat.inverse import calibrate_linear_gaussian
from probstat.multivariate import MultivariateNormal
from probstat.validation import AdditiveGaussianDiscrepancy
from probstat.visualization import save_figure
from probstat.visualization.uq import plot_calibration_prediction


def generate_measurements(
    inputs: tuple[float, ...], *, seed: int
) -> tuple[float, ...]:
    """Generate seeded measurements from a documented teaching scenario.

    Args:
        inputs: Finite experiment settings.
        seed: Call-local pseudo-random seed.

    Returns:
        Responses from ``1.25 + 0.82*x`` with Gaussian scale ``0.35``.

    Equation:
        [EQ-LAB-CAL-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 3].

    """
    generator = random.Random(seed)
    return tuple(1.25 + 0.82 * value + generator.gauss(0.0, 0.35) for value in inputs)


def main() -> None:
    """Calibrate, diagnose, predict, check, and save an explanatory plot.

    References:
        [EQ-REG-OLS-001], [EQ-BAYES-LIN-POST-001], [EQ-BAYES-PRED-001],
        [EQ-BAYES-PPC-001], [REF-GELMAN-ETAL-2013], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    calibration_inputs = tuple(float(index) for index in range(8))
    observations = generate_measurements(calibration_inputs, seed=2026)
    prior = MultivariateNormal(
        (0.0, 0.0),
        ((25.0, 0.0), (0.0, 4.0)),
    )
    result = calibrate_linear_gaussian(
        calibration_inputs,
        observations,
        observation_variance=0.35**2,
        prior=prior,
    )
    discrepancy = AdditiveGaussianDiscrepancy(
        0.15,
        "Residual linear-model inadequacy retained for prediction.",
        "Teaching system for inputs from 0 through 9.",
    )
    grid = tuple(index * 0.05 for index in range(181))
    design = result.design_rows(grid)
    latent = posterior_predictive(
        result.posterior,
        design,
        include_observation_noise=False,
        discrepancy_variance=discrepancy.variance,
    )
    future = posterior_predictive(
        result.posterior,
        design,
        include_observation_noise=True,
        discrepancy_variance=discrepancy.variance,
    )
    calibration_predictive = posterior_predictive(
        result.posterior,
        result.design_rows(calibration_inputs),
        include_observation_noise=True,
        discrepancy_variance=discrepancy.variance,
    )
    check = posterior_predictive_check(
        calibration_predictive,
        observations,
        draws=2_000,
        seed=2026,
    )

    print(
        "OLS intercept/slope="
        f"({result.regression.intercept:.5f}, {result.regression.slope:.5f})"
    )
    print(f"posterior mean={result.posterior.mean}")
    print(f"posterior covariance={result.posterior.covariance}")
    print(f"design condition number={result.identifiability.condition_number:.5f}")
    print(f"posterior-predictive tail probability={check.tail_probability:.5f}")

    latent_scale = tuple(math.sqrt(value) for value in latent.parameter_variance)
    future_scale = tuple(
        math.sqrt(future.covariance[index][index]) for index in range(len(grid))
    )
    latent_lower = tuple(
        center - 1.96 * scale
        for center, scale in zip(latent.mean, latent_scale, strict=True)
    )
    latent_upper = tuple(
        center + 1.96 * scale
        for center, scale in zip(latent.mean, latent_scale, strict=True)
    )
    future_lower = tuple(
        center - 1.96 * scale
        for center, scale in zip(future.mean, future_scale, strict=True)
    )
    future_upper = tuple(
        center + 1.96 * scale
        for center, scale in zip(future.mean, future_scale, strict=True)
    )

    axes = plot_calibration_prediction(
        calibration_inputs,
        observations,
        grid,
        latent.mean,
        (latent_lower, latent_upper),
        (future_lower, future_upper),
        calibration_limit=max(calibration_inputs),
    )
    save_figure(
        axes.figure,
        Path(__file__).parent / "generated" / "bayesian-calibration-prediction.png",
    )
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
