"""Backward-then-forward laboratory: calibrate a decay model, then predict.

Synthetic measurements of an exponential decay are generated from known
parameters. ``UQStudy.calibrate`` samples the posterior by adaptive MCMC in
the prior's standard-normal space; ``UQStudy.predict`` pushes the posterior
through the same model to forecast beyond the measured time window.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path

from probstat.distributions import Uniform
from probstat.inverse import Experiment, NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest
from probstat.visualization import save_figure
from probstat.visualization.inference_plots import (
    plot_chain_traces,
    plot_posterior_marginals,
)
from probstat.visualization.uq import plot_calibration_prediction
from probstat.workflow import UQStudy

TRUE_AMPLITUDE = 5.0
TRUE_RATE = 0.3
NOISE = 0.12


def decay(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the amplitude ``A exp(-k t)`` at time ``t``.

    Args:
        parameters: Amplitude ``A``, rate ``k``, and time ``t``.

    Returns:
        Mapping with the observable ``y``.

    Equation:
        [EQ-CAL-NL-001].

    References:
        [REF-SEBER-WILD-2003, Chapter 1]; a transparent teaching model.

    """
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def main() -> None:
    """Calibrate, diagnose, predict, and write figures.

    References:
        [REF-HAARIO-SAKSMAN-TAMMINEN-2001], [REF-VEHTARI-ETAL-2021],
        [REF-GELMAN-ETAL-2013, Chapter 11], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    prior = JointInput.independent(
        RandomInput("A", Uniform(1.0, 10.0)), RandomInput("k", Uniform(0.01, 1.5))
    )
    model = ProbabilisticModel(
        prior, decay, (QuantityOfInterest("y", "decayed amplitude"),), fixed={"t": 0.0}
    )
    generator = random.Random(2026)
    times = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    measured = tuple(
        TRUE_AMPLITUDE * math.exp(-TRUE_RATE * t) + generator.gauss(0.0, NOISE)
        for t in times
    )
    experiments = tuple(
        Experiment({"y": y}, {"t": t}) for y, t in zip(measured, times, strict=True)
    )
    study = UQStudy(model, seed=2026, name="decay")
    posterior = study.calibrate(
        experiments, (NoiseModel("y", standard_deviation=NOISE),), draws=1500
    )
    for entry in posterior.summary():
        print(
            f"{entry.name}: mean={entry.mean:.4f} sd={entry.standard_deviation:.4f} "
            f"R-hat={entry.rank_normalized_rhat:.4f} "
            f"ESS={entry.effective_sample_size:.0f}"
        )
    grid = tuple(0.25 * k for k in range(49))
    prediction = study.predict(
        conditions=tuple({"t": t} for t in grid), include_noise=True, max_draws=800
    )
    print(f"forward evaluations in the study={study.total_evaluations}")
    print(study.report().splitlines()[0])

    output = Path(__file__).parent / "generated"
    names = posterior.parameter_names
    chains = posterior.mcmc.chains
    summaries = {entry.name: entry for entry in posterior.summary()}
    axes = plot_chain_traces(
        [[sample[1] for sample in chain.samples] for chain in chains],
        name="k (decay rate)",
        rhat=summaries["k"].rank_normalized_rhat,
        truth=TRUE_RATE,
    )
    save_figure(axes.figure, output / "calibration-trace-k.png")
    plt.close(axes.figure)
    pooled = {
        name: [sample[index] for chain in chains for sample in chain.samples]
        for index, name in enumerate(names)
    }
    figure = plot_posterior_marginals(
        pooled, truth={"A": TRUE_AMPLITUDE, "k": TRUE_RATE}
    )
    save_figure(figure, output / "calibration-marginals.png")
    plt.close(figure)
    latent = [item.latent_quantiles for item in prediction.predictions]
    observed = [
        item.observed_quantiles or item.latent_quantiles
        for item in prediction.predictions
    ]
    axes = plot_calibration_prediction(
        times,
        measured,
        grid,
        [item.mean for item in prediction.predictions],
        ([q[0] for q in latent], [q[-1] for q in latent]),
        ([q[0] for q in observed], [q[-1] for q in observed]),
        calibration_limit=max(times),
        x_label="Time t",
        y_label="Amplitude y",
        confidence=prediction.predictions[0].probabilities[-1]
        - prediction.predictions[0].probabilities[0],
        title="The calibrated decay predicts beyond the data with honest bands",
    )
    save_figure(axes.figure, output / "calibration-forecast.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
