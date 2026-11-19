"""State-space learning laboratory: parameters of an AR(1) observed with noise.

Run with ``PYTHONPATH=src python examples/state_space_learning_lab.py``. The
static parameters of a scalar AR(1) state observed with noise are estimated
four ways: maximum likelihood by BFGS on the Kalman likelihood, EM, and
particle marginal Metropolis-Hastings (PMMH) for the AR coefficient against
the exact posterior by quadrature. Then the three particle filters are
compared by the spread of their likelihood estimates, and the FFBS particle
smoother by its distance from the exact RTS smoother. With Matplotlib three
figures are written to ``examples/generated/``. Every random step is seeded;
the run takes about a minute and a half in pure Python.

Definitions:
    Model: ``x_k = phi x_(k-1) + w_k``, ``w_k ~ N(0, q)``, ``y_k = x_k + v_k``,
        ``v_k ~ N(0, r)``, ``x_0 ~ N(0, 1)``; truth ``phi = 0.8``, ``q = 1``,
        ``r = 0.5``, 100 simulated steps.
    ``l(theta)``: the exact Kalman log-likelihood ``log p(y_1:T | theta)``.
    EM trace: ``l`` after each EM iteration; its shortfall is the gap to the
        maximum found by BFGS.
    PMMH: random-walk Metropolis-Hastings on ``phi`` (``q``, ``r`` known,
        uniform prior on ``(-1, 1)``) with a particle estimate of the
        likelihood of the first 40 observations.
    likelihood spread: standard deviation of ``log p_hat(y_1:T)`` over
        independently seeded filter runs; the bias is the mean of
        ``log p_hat - l``.

Assumptions:
    - The data come from the model being fitted, so the Kalman filter is
      exact and every particle method has an exact reference.
    - The variances are fitted on the log scale; the MLE and EM agree only up
      to their stopping tolerances on a flat likelihood.
    - PMMH targets the exact posterior only because the particle likelihood
      is unbiased and the current estimate is stored, never recomputed; its
      Monte Carlo error is reported as an MCSE from the effective sample
      size.
    - FFBS needs a positive process variance and a filter run that stored
      its particle history.

Equations:
    EQ-FILT-SSM-001: linear-Gaussian state-space model that simulates the data.
    EQ-FILT-KF-004: prediction-error decomposition of the exact likelihood.
    EQ-FILT-RTS-001: RTS smoother, the exact smoothing reference.
    EQ-FILT-MLE-001: filter maximum likelihood and observed information.
    EQ-FILT-EM-001: EM M-step for the linear-Gaussian model.
    EQ-FILT-PMMH-001: particle marginal Metropolis-Hastings acceptance.
    EQ-FILT-PF-001: bootstrap particle weights and likelihood increments.
    EQ-FILT-APF-001: auxiliary particle filter weights and likelihood.
    EQ-FILT-OPT-001: locally optimal proposal for a linear observation.
    EQ-FILT-FFBS-001: forward-filtering backward-smoothing weights.
    EQ-NUM-SIMP-001: Simpson quadrature of the exact posterior of ``phi``.

Diagnostics:
    - The EM trace must never decrease (printed as its smallest step) and
      must end at the BFGS maximum; its log-likelihood shortfall is drawn on
      a logarithmic axis, where linear convergence is a straight line.
    - The MLE prints its observed-information standard errors.
    - PMMH prints its acceptance rate, ESS, R-hat, and the distance of its
      posterior mean from the exact one in MCSE units.
    - Each particle filter prints the mean and standard deviation of its
      log-likelihood estimate over the replications and the mean of
      ``exp(log p_hat - l)``, whose expectation is one.
    - FFBS prints its largest distance from the RTS mean in RTS standard
      deviations.

Cost and error:
    - MLE: one Kalman filter per BFGS function evaluation; EM: one filter
      and smoother per iteration (tolerance ``1e-10``).
    - PMMH: 2 chains x (200 warm-up + 1200 draws), 20 particles of the
      optimal proposal on 40 observations, so about 2800 particle filters.
    - Likelihood spread: 40 replications of 100 particles per filter on the
      40 observations; the standard deviations carry a relative error of
      about ``1/sqrt(2 x 40) = 11 %``.
    - FFBS: one bootstrap filter with 500 particles and 200 backward
      trajectories; its error is Monte Carlo error of one run.

References:
    REF-SHUMWAY-STOFFER-1982: EM for state-space models.
    REF-DURBIN-KOOPMAN-2012: likelihood-based state-space inference.
    REF-ANDRIEU-DOUCET-HOLENSTEIN-2010: particle MCMC and PMMH.
    REF-PITT-SHEPHARD-1999: the auxiliary particle filter.
    REF-DOUCET-GODSILL-ANDRIEU-2000: the optimal proposal.
    REF-GODSILL-DOUCET-WEST-2004: forward-filtering backward-smoothing.
    REF-DOUCET-JOHANSEN-2011: particle filtering and smoothing tutorial.
    REF-SARKKA-2013: Bayesian filtering and smoothing.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from probstat.filtering import (
    LinearGaussianStateSpace,
    auxiliary_particle_filter,
    bootstrap_particle_filter,
    ffbs_particle_smoother,
    fit_state_space_mle,
    kalman_filter,
    linear_gaussian_em,
    optimal_proposal_particle_filter,
    particle_marginal_metropolis_hastings,
    rts_smoother,
    simulate,
)
from probstat.numerics import composite_simpson

# Root seed: the data use SEED, filter replication i uses SEED + 1 + i, the
# PMMH run SEED + 1000, and the smoothing run SEED + 2000 and SEED + 2001.
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
TRUTH = (0.8, 1.0, 0.5)
STEPS = 100
SHORT = 40
REPLICATIONS = 40
PARTICLES = 100


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def ar1(phi: float, q: float, r: float) -> LinearGaussianStateSpace:
    """Return the scalar AR(1)-plus-noise model with ``x_0 ~ N(0, 1)``.

    References:
        [EQ-FILT-SSM-001], [REF-DURBIN-KOOPMAN-2012, Chapter 3].

    """
    return LinearGaussianStateSpace(
        ((phi,),), ((q,),), ((1.0,),), ((r,),), (0.0,), ((1.0,),)
    )


def from_log_scale(theta: Sequence[float]) -> LinearGaussianStateSpace:
    """Build the model from ``(phi, log q, log r)``.

    References:
        [EQ-FILT-SSM-001]. No other equation is evaluated.

    """
    return ar1(theta[0], math.exp(theta[1]), math.exp(theta[2]))


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def estimation_section(data: Sequence[tuple[float, ...]]) -> dict[str, Any]:
    """Fit ``(phi, q, r)`` by BFGS on the Kalman likelihood and by EM.

    References:
        [EQ-FILT-MLE-001], [EQ-FILT-EM-001], [EQ-FILT-KF-004],
        [REF-SHUMWAY-STOFFER-1982], [REF-DURBIN-KOOPMAN-2012, Chapter 7].

    """
    truth_loglik = kalman_filter(ar1(*TRUTH), data).log_likelihood
    mle = fit_state_space_mle(
        from_log_scale,
        (0.5, math.log(0.5), 0.0),
        data,
        names=("phi", "log_q", "log_r"),
    )
    em = linear_gaussian_em(
        ar1(0.5, 0.5, 1.0),
        data,
        tolerance=1e-10,
        estimate=("transition", "process_covariance", "observation_covariance"),
    )
    trace = em.log_likelihood_trace
    steps = [b - a for a, b in zip(trace, trace[1:], strict=False)]
    phi, q, r = (
        em.model.transition_matrix[0][0],
        em.model.process_covariance[0][0],
        em.model.observation_covariance[0][0],
    )
    errors = mle.standard_errors or (math.nan,) * 3
    print("1. Maximum likelihood and EM (AR(1) plus noise, 100 steps)")
    print(f"   log-likelihood at the truth (0.8, 1, 0.5): {truth_loglik:.5f}")
    print(
        f"   BFGS: phi={mle.parameters[0]:.5f} q={math.exp(mle.parameters[1]):.5f} "
        f"r={math.exp(mle.parameters[2]):.5f} log-likelihood={mle.log_likelihood:.5f}"
    )
    print(
        f"   BFGS standard errors: phi {errors[0]:.3f}, log q {errors[1]:.3f}, "
        f"log r {errors[2]:.3f}"
    )
    print(
        f"   EM:   phi={phi:.5f} q={q:.5f} r={r:.5f} "
        f"log-likelihood={trace[-1]:.5f} after {em.iterations} iterations "
        f"(converged={em.converged})"
    )
    print(
        f"   EM trace from {trace[0]:.4f}; smallest step {min(steps):.2e} "
        f"(never negative: {min(steps) >= -1e-9})\n"
    )
    return {"mle": mle, "em": em}


def exact_posterior(
    data: Sequence[tuple[float, ...]],
) -> tuple[Callable[[float], float], float, float]:
    """Return the exact posterior density of ``phi`` and its mean and sd.

    The prior is uniform on ``(-1, 1)``, ``q`` and ``r`` are known, and the
    posterior is normalized by Simpson quadrature (400 intervals) of the
    Kalman likelihood.

    References:
        [EQ-FILT-KF-004], [EQ-NUM-SIMP-001], [REF-SARKKA-2013, Chapter 12].

    """
    peak = kalman_filter(ar1(TRUTH[0], *TRUTH[1:]), data).log_likelihood

    def unnormalized(phi: float) -> float:
        """Return ``p(y | phi) / p(y | 0.8)`` on the prior support.

        References:
            [EQ-FILT-KF-004]. No other equation is evaluated.

        """
        value = kalman_filter(ar1(phi, *TRUTH[1:]), data).log_likelihood
        return math.exp(value - peak)

    grid = 400
    mass = composite_simpson(unnormalized, -1.0, 1.0, intervals=grid).value
    first = composite_simpson(
        lambda phi: phi * unnormalized(phi), -1.0, 1.0, intervals=grid
    ).value
    second = composite_simpson(
        lambda phi: phi * phi * unnormalized(phi), -1.0, 1.0, intervals=grid
    ).value
    mean = first / mass
    return (
        lambda phi: unnormalized(phi) / mass,
        mean,
        math.sqrt(second / mass - mean * mean),
    )


def pmmh_section(data: Sequence[tuple[float, ...]]) -> dict[str, Any]:
    """Sample ``phi`` by PMMH and compare with the exact posterior.

    References:
        [EQ-FILT-PMMH-001], [EQ-FILT-OPT-001],
        [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010].

    """
    density, mean, sd = exact_posterior(data)

    def log_prior(theta: Sequence[float]) -> float:
        """Return the log density of the uniform prior on ``(-1, 1)``.

        References:
            [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010]. No equation is evaluated.

        """
        return 0.0 if -1.0 < theta[0] < 1.0 else -math.inf

    result = particle_marginal_metropolis_hastings(
        lambda theta: ar1(theta[0], *TRUTH[1:]),
        log_prior,
        (0.5,),
        data,
        particles=20,
        draws=1200,
        warmup=200,
        chains=2,
        proposal_covariance=((0.06,),),
        seed=SEED + 1000,
        particle_filter="optimal",
        names=("phi",),
        check_convergence=False,
    )
    summary = result.summary()[0]
    draws = [value for chain in result.chain_columns(0) for value in chain]
    distance = (summary.mean - mean) / summary.mcse_mean
    print("2. PMMH for phi (first 40 observations, q and r known, uniform prior)")
    print(f"   exact posterior: mean {mean:.4f}, sd {sd:.4f} (Simpson, 400 intervals)")
    print(
        f"   PMMH: mean {summary.mean:.4f} +/- {summary.mcse_mean:.4f} (MCSE), "
        f"sd {summary.standard_deviation:.4f}; {distance:+.2f} MCSE from exact"
    )
    print(
        f"   acceptance {result.acceptance_rate:.3f}, bulk ESS "
        f"{summary.effective_sample_size:.0f} of {len(draws)} draws, R-hat "
        f"{summary.rank_normalized_rhat:.3f}\n"
    )
    return {
        "density": density,
        "mean": mean,
        "sd": sd,
        "draws": draws,
        "summary": summary,
    }


def likelihood_section(data: Sequence[tuple[float, ...]]) -> dict[str, Any]:
    """Compare the spread of three particle likelihood estimates.

    References:
        [EQ-FILT-PF-001], [EQ-FILT-APF-001], [EQ-FILT-OPT-001],
        [REF-PITT-SHEPHARD-1999], [REF-DOUCET-GODSILL-ANDRIEU-2000],
        [REF-DOUCET-JOHANSEN-2011, Section 4].

    """
    model = ar1(*TRUTH)
    exact = kalman_filter(model, data).log_likelihood
    filters = {
        "bootstrap": bootstrap_particle_filter,
        "auxiliary": auxiliary_particle_filter,
        "optimal proposal": optimal_proposal_particle_filter,
    }
    print(
        f"3. Particle likelihoods ({REPLICATIONS} runs, N = {PARTICLES}, first "
        f"40 observations; exact {exact:.4f})"
    )
    print("   filter              mean log p_hat   sd     mean exp(log p_hat - l)")
    spreads: dict[str, list[float]] = {}
    for name, run in filters.items():
        values = [
            run(model, data, particles=PARTICLES, seed=SEED + 1 + i).log_likelihood
            for i in range(REPLICATIONS)
        ]
        ratios = [math.exp(value - exact) for value in values]
        spreads[name] = values
        print(
            f"   {name:18s}  {statistics.fmean(values):.4f}      "
            f"{statistics.stdev(values):.4f}  {statistics.fmean(ratios):.3f} +/- "
            f"{statistics.stdev(ratios) / math.sqrt(REPLICATIONS):.3f}"
        )
    print()
    return {"exact": exact, "values": spreads}


def smoothing_section(data: Sequence[tuple[float, ...]]) -> dict[str, Any]:
    """Compare the FFBS particle smoother with the exact RTS smoother.

    References:
        [EQ-FILT-FFBS-001], [EQ-FILT-RTS-001], [REF-GODSILL-DOUCET-WEST-2004].

    """
    model = ar1(*TRUTH)
    exact = rts_smoother(kalman_filter(model, data))
    particle = bootstrap_particle_filter(
        model, data, particles=500, seed=SEED + 2000, store_history=True
    )
    smoother = ffbs_particle_smoother(
        model, particle, trajectories=200, seed=SEED + 2001
    )
    rts_mean = [state.mean[0] for state in exact.smoothed]
    rts_sd = [math.sqrt(state.covariance[0][0]) for state in exact.smoothed]
    ffbs_mean = [mean[0] for mean in smoother.means]
    ffbs_sd = [math.sqrt(cov[0][0]) for cov in smoother.covariances]
    gaps = [abs(a - b) / s for a, b, s in zip(ffbs_mean, rts_mean, rts_sd, strict=True)]
    worst = max(range(len(gaps)), key=lambda k: gaps[k])
    ratio = statistics.fmean(a / b for a, b in zip(ffbs_sd, rts_sd, strict=True))
    print("4. FFBS (500 particles, 200 trajectories) against the RTS smoother")
    print(
        f"   largest |FFBS mean - RTS mean| = {gaps[worst]:.2f} RTS sd (at k = "
        f"{worst}); mean of the gaps {statistics.fmean(gaps):.2f} RTS sd"
    )
    print(f"   mean ratio of FFBS to RTS standard deviation {ratio:.3f}\n")
    return {
        "observations": [y[0] for y in data],
        "rts": (rts_mean, rts_sd),
        "ffbs": (ffbs_mean, ffbs_sd),
        "mean_gap": statistics.fmean(gaps),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def save(figure: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def em_figure(estimation: dict[str, Any]) -> str:
    """Draw the EM log-likelihood shortfall against the iteration.

    References:
        [EQ-FILT-EM-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    trace = estimation["em"].log_likelihood_trace
    maximum = max(estimation["mle"].log_likelihood, max(trace))
    shortfall = [max(maximum - value, 1e-12) for value in trace]
    # Per-iteration contraction of the gap between iterations 40 and 140.
    rate = (shortfall[140] / shortfall[40]) ** (1.0 / 100.0)
    iterations = list(range(len(trace)))
    axes = theme.new_axes(figsize=(7.2, 4.2))
    theme.line(axes, iterations, shortfall, color=theme.role_colour("model"))
    axes.set_yscale("log")
    axes.set_xlim(0.0, iterations[-1])
    axes.set_yticks([10.0**k for k in range(-6, 2, 2)])
    axes.minorticks_off()
    axes.set_ylim(min(shortfall) / 2.0, 2.0 * max(shortfall))
    theme.axis_labels(axes, "EM iteration", "Log-likelihood below the maximum")
    theme.title_block(
        axes,
        f"EM reaches the BFGS maximum after {estimation['em'].iterations} "
        "iterations, closing a fixed fraction of the gap per step",
        f"Gap to the BFGS log-likelihood {estimation['mle'].log_likelihood:.5f}. "
        f"Straight line = linear convergence: each step keeps {rate:.1%} of the gap",
    )
    return save(axes.figure, "state-space-em-trace.png")


def pmmh_figure(pmmh: dict[str, Any]) -> str:
    """Draw the PMMH histogram of ``phi`` against the exact posterior density.

    References:
        [EQ-FILT-PMMH-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    draws = pmmh["draws"]
    low, high = 0.2, 1.0
    bins = 32
    width = (high - low) / bins
    counts = [0] * bins
    for value in draws:
        index = int((value - low) / width)
        if 0 <= index < bins:
            counts[index] += 1
    heights = [count / (len(draws) * width) for count in counts]
    axes.bar(
        [low + (i + 0.5) * width for i in range(bins)],
        heights,
        width=width * 0.92,
        color=theme.role_colour("observed"),
        alpha=0.35,
        linewidth=0.0,
    )
    grid = [low + (high - low) * i / 200 for i in range(201)]
    style = theme.entity_style("exact posterior")
    theme.line(axes, grid, [pmmh["density"](x) for x in grid], color=style.colour())
    summary = pmmh["summary"]
    axes.annotate(
        "exact posterior",
        xy=(0.5, pmmh["density"](0.5)),
        xytext=(-10.0, 4.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=style.colour(),
    )
    axes.annotate(
        f"PMMH draws (bars): mean {summary.mean:.3f} +/- {summary.mcse_mean:.3f}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    axes.set_xlim(low, high)
    axes.set_ylim(0.0, max(heights) * 1.25)
    theme.axis_labels(axes, "AR coefficient phi", "Posterior density")
    theme.title_block(
        axes,
        f"PMMH with 20 particles matches the exact posterior of phi: mean "
        f"{summary.mean:.3f} against {pmmh['mean']:.3f}",
        f"Exact: Kalman likelihood by Simpson quadrature. PMMH: 2 chains x "
        f"{len(draws) // 2} draws, ESS {summary.effective_sample_size:.0f}, "
        f"difference {(summary.mean - pmmh['mean']) / summary.mcse_mean:.1f} MCSE",
    )
    return save(axes.figure, "state-space-pmmh-posterior.png")


def smoothing_figure(smoothing: dict[str, Any]) -> str:
    """Draw the FFBS mean against the RTS mean and its 95 % band.

    References:
        [EQ-FILT-FFBS-001], [EQ-FILT-RTS-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    rts_mean, rts_sd = smoothing["rts"]
    ffbs_mean, _ = smoothing["ffbs"]
    steps = list(range(len(rts_mean)))
    axes = theme.new_axes(figsize=(7.2, 4.2))
    model_colour = theme.role_colour("model")
    theme.band(
        axes,
        steps,
        [m - 1.96 * s for m, s in zip(rts_mean, rts_sd, strict=True)],
        [m + 1.96 * s for m, s in zip(rts_mean, rts_sd, strict=True)],
        color=model_colour,
    )
    theme.line(axes, steps, rts_mean, color=model_colour)
    observations = smoothing["observations"]
    theme.points(
        axes,
        steps[1:],
        observations,
        color=theme.entity_colour("observations"),
        size=22.0,
        alpha=0.8,
    )
    smoothed = theme.entity_style("smoothed")
    theme.line(
        axes,
        steps,
        ffbs_mean,
        color=smoothed.colour(),
        dashes=smoothed.dashes,
        width=theme.CONTEXT_WIDTH,
        zorder=4.0,
    )
    theme.direct_labels(
        axes,
        [
            (steps[-1], rts_mean[-1], "RTS (exact)", model_colour),
            (steps[-1], ffbs_mean[-1], "FFBS", smoothed.colour()),
        ],
        gap_points=13.0,
    )
    axes.set_xlim(0.0, steps[-1])
    theme.axis_labels(axes, "Time step k", "Smoothed state x_k")
    theme.title_block(
        axes,
        f"The FFBS particle smoother stays within {smoothing['mean_gap']:.2f} RTS "
        "standard deviations of the exact smoother on average",
        "Band: exact 95% smoothing interval. Dots: observations. FFBS: 500 "
        "particles, 200 backward trajectories",
    )
    return save(axes.figure, "state-space-ffbs-smoother.png")


def main() -> None:
    """Run every section, write the figures, and print the elapsed time.

    References:
        [REF-SHUMWAY-STOFFER-1982], [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010].

    """
    start = time.perf_counter()
    simulation = simulate(ar1(*TRUTH), STEPS, seed=SEED)
    data = [tuple(y) for y in simulation.observations]
    short = data[:SHORT]
    estimation = estimation_section(data)
    pmmh = pmmh_section(short)
    likelihood_section(short)
    smoothing = smoothing_section(short)
    if has_matplotlib():
        written = (
            em_figure(estimation),
            pmmh_figure(pmmh),
            smoothing_figure(smoothing),
        )
        print("figures: " + ", ".join(written))
    else:
        print("Matplotlib is not installed; figures skipped")
    print(f"elapsed {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
