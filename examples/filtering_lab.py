"""Sequential Bayesian estimation laboratory: tracking an unmeasured velocity.

A constant-velocity target is observed through noisy positions only. The
Kalman filter, RTS smoother, bootstrap particle filter, and ensemble Kalman
filter estimate the hidden velocity; their log-likelihoods are compared.
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
    simulation = simulate(model, steps, seed=2026)
    states = simulation.states[-steps:]
    observations = simulation.observations
    times = tuple(0.5 * (k + 1) for k in range(steps))

    filtered = kalman_filter(model, observations)
    smoothed = rts_smoother(filtered)
    particle = bootstrap_particle_filter(model, observations, particles=2000, seed=2026)
    ensemble = ensemble_kalman_filter(
        model, [list(y) for y in observations], ensemble_size=200, seed=2026
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
