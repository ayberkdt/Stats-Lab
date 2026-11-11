"""Sequential Bayesian estimation laboratory: tracking an unmeasured velocity.

A constant-velocity target is observed through noisy positions only. The
Kalman filter, RTS smoother, bootstrap particle filter, and ensemble Kalman
filter estimate the hidden velocity; their log-likelihoods are compared.

Definitions:
    ``x_k = (position, velocity)``: hidden state at step ``k`` (80 steps of
        length 0.5).
    ``y_k``: noisy position measurement with standard deviation 0.8.
    ``q``: white-noise acceleration intensity (0.4) that sets the process
        covariance ``Q`` of the discretized constant-velocity model.
    filtered and smoothed means: ``E[x_k | y_1:k]`` and ``E[x_k | y_1:T]``.
    log-likelihood: ``log p(y_1:T)`` from the prediction-error decomposition.

Assumptions:
    - The data are simulated from the same linear-Gaussian model that the
      filters use, so the Kalman filter and RTS smoother are exact and serve
      as the reference for the two sampling filters.
    - Process noise, measurement noise, and the initial state are mutually
      independent Gaussians.

Equations:
    EQ-FILT-SSM-001: linear-Gaussian state-space model that simulates the
        track and defines the filters' model.
    EQ-FILT-BAYES-001: Bayesian prediction-update recursion every filter
        approximates.
    EQ-FILT-KF-001: Kalman prediction step.
    EQ-FILT-KF-002: Kalman gain and mean update.
    EQ-FILT-KF-003: Joseph-form covariance update.
    EQ-FILT-KF-004: prediction-error decomposition behind the printed Kalman
        log-likelihood.
    EQ-FILT-RTS-001: Rauch-Tung-Striebel backward smoothing pass.
    EQ-FILT-PF-001: bootstrap particle weights and likelihood increments.
    EQ-FILT-PF-002: effective sample size that triggers resampling.
    EQ-FILT-PF-003: systematic resampling.
    EQ-DA-ENKF-001: perturbed-observation ensemble Kalman analysis.
    EQ-DA-ENKF-002: ensemble mean and covariance.
    EQ-LINALG-007: linear covariance propagation ``Cov(HX) = H P H'`` that
        the tracking model and the plotted bands rely on.

Diagnostics:
    - The particle log-likelihood is compared with the exact Kalman value;
      a gap much larger than its Monte Carlo spread signals weight
      degeneracy.
    - Mean absolute filter errors of position and velocity against the
      simulated truth; the velocity is never observed, so its error shows
      what the dynamics alone recover.
    - The figures overlay the filtered mean with its 95% band
      (``m +/- 1.96 s``), the smoothed track, and the truth; the figure
      title states the fraction of steps with the truth inside the band,
      which should be near 95%.

Cost and error:
    - The Kalman filter and smoother are exact up to rounding for this
      model; their cost is a fixed number of 2 x 2 matrix operations per
      step.
    - The particle filter uses 2000 particles and the ensemble Kalman filter
      200 members; both log-likelihood estimates carry Monte Carlo error of
      order ``N**-0.5`` and change with ``SEED``.

References:
    REF-KALMAN-1960: the Kalman filter.
    REF-RAUCH-TUNG-STRIEBEL-1965: fixed-interval smoothing.
    REF-SARKKA-2013: Bayesian filtering and smoothing.
    REF-GORDON-SALMOND-SMITH-1993: bootstrap particle filter.
    REF-DOUCET-JOHANSEN-2011: particle filtering tutorial.
    REF-EVENSEN-1994: ensemble Kalman filter.

"""

from __future__ import annotations

import math
from pathlib import Path

from probstat.data_assimilation import ensemble_kalman_filter
from probstat.filtering import (
    LinearGaussianStateSpace,
    bootstrap_particle_filter,
    kalman_filter,
    rts_smoother,
    simulate,
)
from probstat.visualization import save_figure
from probstat.visualization.inference_plots import plot_filter_track

# Root seed of the simulated track and of both sampling filters.
SEED = 2026


def tracking_model(
    step: float = 0.5, intensity: float = 0.4
) -> LinearGaussianStateSpace:
    """Return the discretized constant-velocity (white-acceleration) model.

    Args:
        step: Sampling interval.
        intensity: Continuous white-noise acceleration intensity ``q``.

    Returns:
        Linear-Gaussian state-space model with position-only observations.

    Equation:
        [EQ-LINALG-007].

    References:
        [REF-SARKKA-2013, Section 4.3].

    """
    q = intensity
    return LinearGaussianStateSpace(
        transition_matrix=((1.0, step), (0.0, 1.0)),
        process_covariance=(
            (q * step**3 / 3.0, q * step**2 / 2.0),
            (q * step**2 / 2.0, q * step),
        ),
        observation_matrix=((1.0, 0.0),),
        observation_covariance=((0.8**2,),),
        initial_mean=(0.0, 1.0),
        initial_covariance=((1.0, 0.0), (0.0, 1.0)),
    )


def main() -> None:
    """Run the filters and write tracking figures.

    References:
        [REF-SARKKA-2013], [REF-GORDON-SALMOND-SMITH-1993], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    model = tracking_model()
    steps = 80
    simulation = simulate(model, steps, seed=SEED)
    states = simulation.states[-steps:]
    observations = simulation.observations
    times = tuple(0.5 * (k + 1) for k in range(steps))

    filtered = kalman_filter(model, observations)
    smoothed = rts_smoother(filtered)
    particle = bootstrap_particle_filter(model, observations, particles=2000, seed=SEED)
    ensemble = ensemble_kalman_filter(
        model, [list(y) for y in observations], ensemble_size=200, seed=SEED
    )
    print(f"Kalman log-likelihood={math.fsum(filtered.log_likelihood_terms):.4f}")
    print(f"particle log-likelihood={math.fsum(particle.log_likelihood_terms):.4f}")
    enkf_terms = getattr(ensemble, "log_likelihood_terms", None)
    if enkf_terms is not None:
        print(f"EnKF log-likelihood={math.fsum(enkf_terms):.4f}")

    filtered_states = filtered.filtered[-steps:]
    smoothed_states = smoothed.smoothed[-steps:]
    output = Path(__file__).parent / "generated"
    for component, name, show_data in ((0, "position", True), (1, "velocity", False)):
        axes = plot_filter_track(
            times,
            [state.mean[component] for state in filtered_states],
            [
                math.sqrt(state.covariance[component][component])
                for state in filtered_states
            ],
            observations=[y[0] for y in observations] if show_data else None,
            truth=[state[component] for state in states],
            smoothed=(
                [state.mean[component] for state in smoothed_states],
                [
                    math.sqrt(state.covariance[component][component])
                    for state in smoothed_states
                ],
            ),
            name=name,
        )
        save_figure(axes.figure, output / f"filtering-{name}.png")
        plt.close(axes.figure)
        errors = [
            abs(state.mean[component] - truth[component])
            for state, truth in zip(filtered_states, states, strict=True)
        ]
        print(f"{name}: mean absolute filter error={math.fsum(errors) / steps:.4f}")


if __name__ == "__main__":
    main()
