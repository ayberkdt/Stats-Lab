"""Reproducible UQ-6 linear-Gaussian Bayesian calibration laboratory."""

from __future__ import annotations

import math
import random
from pathlib import Path

from probstat.inference import posterior_predictive, posterior_predictive_check
from probstat.inverse import calibrate_linear_gaussian
from probstat.multivariate import MultivariateNormal
from probstat.validation import AdditiveGaussianDiscrepancy


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

    figure, axes = plt.subplots()
    axes.fill_between(
        grid,
        future_lower,
        future_upper,
        color="#F58518",
        alpha=0.18,
        label="future observation: 95% marginal band",
    )
    axes.fill_between(
        grid,
        latent_lower,
        latent_upper,
        color="#4C78A8",
        alpha=0.3,
        label="parameter uncertainty: 95% marginal band",
    )
    axes.plot(grid, latent.mean, color="#2F4B7C", label="posterior mean response")
    axes.scatter(
        calibration_inputs,
        observations,
        color="#E45756",
        marker="o",
        zorder=3,
        label="calibration measurements",
    )
    axes.axvline(
        max(calibration_inputs),
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="calibration-domain boundary",
    )
    axes.set(
        xlabel="Input x",
        ylabel="Response y",
        title="Linear-Gaussian calibration and prediction",
    )
    axes.legend(fontsize="small")
    figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "bayesian-calibration-prediction.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
