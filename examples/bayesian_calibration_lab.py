"""Reproducible UQ-6 linear-Gaussian Bayesian calibration laboratory.

Eight synthetic measurements of ``1.25 + 0.82 x + N(0, 0.35^2)`` at
``x = 0, ..., 7`` calibrate the intercept and slope of a straight line under
a Gaussian prior. The exact conjugate posterior is compared with ordinary
least squares, a posterior-predictive check tests the fit, and latent and
observable predictive bands with an explicit model-discrepancy variance are
drawn up to ``x = 9``, beyond the calibrated range. Run with
``PYTHONPATH=src python examples/bayesian_calibration_lab.py``; the figure is
written to ``examples/generated/``.

Definitions:
    ``theta = (a, b)``: intercept and slope with prior
    ``N((0, 0), diag(25, 4))``.
    ``H``: design matrix with rows ``(1, x_i)``.
    Discrepancy: an additive Gaussian model-inadequacy term with standard
    deviation 0.15 retained for prediction.
    Latent band: uncertainty of the mean response (parameters and
    discrepancy); future band: adds the observation noise.
    Posterior-predictive tail probability: the share of replicated data sets
    whose Mahalanobis discrepancy exceeds that of the observed data.

Assumptions:
    - The observation noise is Gaussian with known variance ``0.35^2`` and
      independent across measurements.
    - The model is linear in ``theta``, so the posterior is exactly Gaussian.
    - The discrepancy is independent of the noise and of ``theta``; bands
      beyond ``x = 7`` are extrapolation and rest on that assumption.

Equations:
    EQ-LAB-CAL-001: synthetic data-generating process.
    EQ-REG-OLS-001: least-squares intercept and slope for comparison.
    EQ-BAYES-LIN-POST-001: conjugate Gaussian posterior.
    EQ-BAYES-PRED-001: Gaussian posterior predictive with discrepancy.
    EQ-CAL-DISC-001: additive discrepancy variance in the predictive.
    EQ-BAYES-PPC-001: posterior-predictive tail probability.

Diagnostics:
    - The design condition number flags poorly identified parameters.
    - The posterior-predictive tail probability (2000 replicates) checks the
      calibrated model against the observations; values near 0 or 1
      indicate misfit.
    - The figure separates latent from observable uncertainty and marks the
      end of the calibration range.

Cost and error:
    - The posterior and predictive moments are closed-form, so they carry
      only rounding error; the eight observations are the whole data cost.
    - The predictive check uses 2000 seeded replicates; its tail probability
      has Monte Carlo error of about ``sqrt(p (1 - p) / 2000)``.
    - Bands are ``+-1.96`` predictive standard deviations (95% for a
      Gaussian predictive).

References:
    REF-GELMAN-ETAL-2013: conjugate posteriors and predictive checks.
    REF-MONTGOMERY-PECK-VINING-2021: least-squares regression.
    REF-KENNEDY-OHAGAN-2001: model discrepancy in calibration.

"""

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

# Seed of the synthetic measurements and of the posterior-predictive check.
SEED = 2026


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
    observations = generate_measurements(calibration_inputs, seed=SEED)
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
        seed=SEED,
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
