"""Backward-then-forward laboratory: calibrate a decay model, then predict.

Synthetic measurements of an exponential decay are generated from known
parameters. ``UQStudy.calibrate`` samples the posterior by adaptive MCMC in
the prior's standard-normal space; ``UQStudy.predict`` pushes the posterior
through the same model to forecast beyond the measured time window.

Definitions:
    ``y(t) = A exp(-k t)``: forward model with amplitude ``A`` and rate
        ``k``; nine measurements at ``t`` from 0 to 6.
    prior: independent ``A ~ Uniform(1, 10)`` and ``k ~ Uniform(0.01,
        1.5)``, mapped to standard-normal space by the Nataf transformation.
    posterior: ``p(A, k | y)`` sampled by adaptive Metropolis.
    R-hat and ESS: rank-normalized split potential scale reduction and
        effective sample size of each parameter.
    latent and observed predictions: ``y*`` from posterior draws without
        and with measurement noise.

Assumptions:
    - Measurement errors are independent ``N(0, 0.12^2)`` with known
      standard deviation; the model has no discrepancy term, which is
      correct here because the data come from the model itself.
    - The chains have converged when R-hat is close to 1 and ESS is
      several hundred; the posterior is then summarized by its draws.
    - Forecasts beyond ``t = 6`` extrapolate the same model form.

Equations:
    EQ-CAL-NL-001: nonlinear calibration observation model.
    EQ-CAL-LIK-001: Gaussian measurement log-likelihood.
    EQ-CAL-POST-001: log posterior of the calibration problem.
    EQ-CAL-STD-001: posterior in the prior's standard-normal space.
    EQ-MOD-NATAF-001: Nataf map between the prior and standard normals.
    EQ-MCMC-AM-001: adaptive Metropolis proposal covariance.
    EQ-MCMC-MH-001: Metropolis-Hastings acceptance.
    EQ-MCMC-RHAT-002: rank-normalized split R-hat in the summary and trace.
    EQ-MCMC-ESS-001: effective sample size in the summary.
    EQ-CAL-PRED-001: posterior predictive draws and quantile intervals.
    EQ-STAT-QUANT-001: sample quantiles of the marginals and predictions.

Diagnostics:
    - Posterior means against the true ``A = 5`` and ``k = 0.3``, with R-hat
      and ESS per parameter.
    - The trace figure of ``k`` shows mixing of the chains and the truth;
      the marginal figure overlays the true values on the posterior
      histograms.
    - The forecast figure separates latent (model) and observed (model plus
      noise) bands and marks the end of the calibration window.

Cost and error:
    - Four chains of 1000 warm-up and 1500 kept draws, plus 800 posterior
      draws pushed through 49 prediction times; the study prints its total
      forward-model evaluations.
    - Posterior means carry Monte Carlo error of about ``sd / sqrt(ESS)``;
      prediction quantiles carry the error of 800 draws.

References:
    REF-SEBER-WILD-2003: nonlinear regression.
    REF-KENNEDY-OHAGAN-2001: Bayesian calibration of computer models.
    REF-HAARIO-SAKSMAN-TAMMINEN-2001: adaptive Metropolis.
    REF-VEHTARI-ETAL-2021: rank-normalized R-hat and ESS.
    REF-GELMAN-ETAL-2013: posterior predictive distributions.

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
# Seed of the synthetic measurements and root seed of the study.
SEED = 2026


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
    generator = random.Random(SEED)
    times = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    measured = tuple(
        TRUE_AMPLITUDE * math.exp(-TRUE_RATE * t) + generator.gauss(0.0, NOISE)
        for t in times
    )
    experiments = tuple(
        Experiment({"y": y}, {"t": t}) for y, t in zip(measured, times, strict=True)
    )
    study = UQStudy(model, seed=SEED, name="decay")
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
