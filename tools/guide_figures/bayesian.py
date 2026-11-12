"""Figures of the Bayesian, calibration, generative, and transport guides.

Each figure runs a seeded computation of the library and returns the facts
its caption quotes (ADR-0105). The guides are hierarchical-models,
probabilistic-modeling, probabilistic-programming,
nonlinear-calibration-and-mcmc, inverse-problems-and-calibration,
model-discrepancy, generative-models, and optimal-transport. MCMC runs are
short and seeded; their convergence diagnostics are drawn or quoted, never
hidden.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-GELMAN-ETAL-2013],
    [REF-VEHTARI-ETAL-2021], [REF-KENNEDY-OHAGAN-2001],
    [REF-PAPAMAKARIOS-ETAL-2021], [REF-PEYRE-CUTURI-2019].

"""

from __future__ import annotations

import functools
import math
import random
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from probstat.distributions import (
    Beta,
    Binomial,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    Uniform,
)
from probstat.generative import fit_flow, mean_log_likelihood, real_nvp_flow
from probstat.inference import (
    ConvergenceWarning,
    GaussianMixture,
    fit_multivariate_normal_mle,
    hierarchical_normal_posterior,
    posterior_predictive,
    posterior_predictive_check,
    ppl,
    random_walk_metropolis,
    rank_normalized_rhat,
    select_gaussian_mixture,
    split_rhat,
)
from probstat.information import (
    exact_transport,
    quantile_wasserstein,
    sinkhorn,
    squared_euclidean_cost_matrix,
)
from probstat.inverse import (
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    NoiseModel,
    calibrate_linear_gaussian,
    predict,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.propagation import compare_designs
from probstat.statistics import pearson_correlation, quantile
from probstat.uq import QuantityOfInterest
from probstat.validation import AdditiveGaussianDiscrepancy
from probstat.visualization import theme
from probstat.visualization.calibration_plots import plot_discrepancy_predictions
from probstat.visualization.hierarchical_plots import (
    plot_mixture_fit,
    plot_shrinkage,
)
from probstat.visualization.transport_plots import plot_quantile_wasserstein
from probstat.visualization.uq import (
    plot_calibration_prediction,
    plot_design_comparison,
)

from . import GuideFigure

FIGSIZE = (7.2, 4.0)
WIDE = (7.6, 4.0)

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
SCHOOLS = ("A", "B", "C", "D", "E", "F", "G", "H")


def _retitle(axes: Any, title: str) -> None:
    """Replace the assertion title a library plot wrote, keeping its subtitle.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    axes.set_title(
        title[:1].upper() + title[1:],
        loc="left",
        fontsize=theme.TITLE_SIZE,
        fontweight="bold",
        color=theme.active_palette().ink,
        pad=24.0,
    )


def _check_title(axes: Any, facts: dict[str, str]) -> None:
    """Fail when a fact string is missing from the title the plot wrote.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    title = " ".join(axes.get_title(loc="left").split())
    for key, value in facts.items():
        if value not in title:
            raise AssertionError(f"fact {key}={value!r} not in title {title!r}")


def _quiet(function: Callable[[], Any]) -> Any:
    """Run a deliberately short or badly mixing sampler without its warnings.

    The diagnostics the warnings report are drawn or quoted in the figure.

    References:
        [REF-VEHTARI-ETAL-2021]. No equation is evaluated.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return function()


# --------------------------------------------------------------------------
# hierarchical-models
# --------------------------------------------------------------------------


def _schools_shrinkage() -> tuple[Any, dict[str, str]]:
    """Eight schools: raw estimates against partially pooled posteriors.

    References:
        [EQ-HIER-001], [EQ-HIER-004], [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    posterior = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S)
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=(7.2, 5.2))
    plot_shrinkage(posterior, names=SCHOOLS, ax=axes)
    means, _ = posterior.theta_moments()
    shrink = posterior.expected_shrinkage()
    mu, _ = posterior.mu_moments()
    facts = {
        "shrinkage": f"{math.fsum(shrink) / len(shrink):.0%}",
        "raw_range": f"{max(SCHOOLS_Y) - min(SCHOOLS_Y):.1f}",
        "pooled_range": f"{max(means) - min(means):.1f}",
    }
    _check_title(axes, facts)
    facts["mu"] = f"{mu:.3g}"
    return figure, facts


def _mixture_fit() -> tuple[Any, dict[str, str]]:
    """BIC-selected Gaussian mixture over 300 draws of a three-component law.

    References:
        [EQ-GMM-001], [EQ-GMM-004], [REF-MCLACHLAN-PEEL-2000],
        [REF-SCHWARZ-1978].

    """
    truth = GaussianMixture(
        (0.3, 0.5, 0.2),
        ((-3.0,), (0.5,), (4.0,)),
        (((0.64,),), ((1.0,),), ((0.36,),)),
    )
    data = [point[0] for point in truth.sample(300, seed=21)]
    selection = select_gaussian_mixture(
        data, (1, 2, 3, 4, 5), seed=5, tolerance=1e-8, max_iterations=300
    )
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_mixture_fit(selection.best, data, ax=axes)
    mixture = selection.best.mixture
    centres = sorted(mean[0] for mean in mixture.means)
    facts = {
        "components": f"K = {selection.best_components}",
        "centres": ", ".join(f"{c:.2f}" for c in centres),
    }
    _retitle(
        axes,
        f"BIC selects {facts['components']} for 300 draws; the fitted centres "
        f"{facts['centres']} sit near the true -3, 0.5, 4",
    )
    return figure, facts


# --------------------------------------------------------------------------
# probabilistic-programming
# --------------------------------------------------------------------------

BETA_A, BETA_B, TRIALS, SUCCESSES = 2.0, 3.0, 20, 13
NORMAL_DATA = (4.1, 5.3, 3.8, 6.0, 4.9, 5.5, 4.4, 5.1, 6.2, 3.9)
NORMAL_SIGMA, PRIOR_MEAN, PRIOR_SD = 1.2, 0.0, 3.0


def _beta_binomial_model(m: Any) -> None:
    """Beta(2, 3) prior with 13 successes in 20 binomial trials.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2].

    """
    p = m.sample("p", Beta(BETA_A, BETA_B))
    m.observe("k", Binomial(TRIALS, p), SUCCESSES)


def _normal_model(m: Any) -> None:
    """Model a normal mean with known scale; conditioning supplies the data.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2].

    """
    mu = m.sample("mu", Normal(PRIOR_MEAN, PRIOR_SD))
    m.observe("y", Normal(mu, NORMAL_SIGMA), None, shape=len(NORMAL_DATA))


@functools.lru_cache(maxsize=1)
def _beta_binomial_draws() -> tuple[tuple[float, ...], Any]:
    """Return the lab's NUTS draws of ``p`` (seed 7) and their summary.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-CARPENTER-ETAL-2017].

    """
    result = ppl.sample(_beta_binomial_model, draws=1000, warmup=300, chains=2, seed=7)
    draws = tuple(float(p) for p in result["p"])  # type: ignore[union-attr]
    return draws, result.summary()[0]


def _ppl_beta_binomial() -> tuple[Any, dict[str, str]]:
    """NUTS draws of a probabilistic program against the exact Beta posterior.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2], [REF-HOFFMAN-GELMAN-2014].

    """
    draws, summary = _beta_binomial_draws()
    prior = Beta(BETA_A, BETA_B)
    exact = Beta(BETA_A + SUCCESSES, BETA_B + TRIALS - SUCCESSES)
    facts = {
        "nuts_mean": f"{summary.mean:.4f}",
        "exact_mean": f"{exact.mean:.4f}",
        "rhat": f"{summary.rank_normalized_rhat:.3f}",
    }
    axes = theme.new_axes(figsize=FIGSIZE)
    palette = theme.active_palette()
    width = 0.025
    axes.hist(
        draws,
        bins=[i * width for i in range(int(1.0 / width) + 1)],
        density=True,
        color=palette.observed,
        alpha=0.45,
        edgecolor=palette.canvas,
        linewidth=0.6,
        zorder=1,
    )
    grid = [0.002 + 0.996 * i / 400 for i in range(401)]
    exact_pdf = [exact.pdf(x) for x in grid]
    prior_pdf = [prior.pdf(x) for x in grid]
    theme.line(
        axes, grid, prior_pdf, color=theme.role_colour("context"),
        dashes=theme.FITTED, width=theme.CONTEXT_WIDTH,
    )  # fmt: skip
    theme.line(axes, grid, exact_pdf, entity="exact posterior", zorder=4.0)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, None)
    peak = max(range(len(grid)), key=lambda i: exact_pdf[i])
    axes.annotate(
        "exact Beta(15, 10)", (grid[peak], exact_pdf[peak]), xytext=(10.0, 4.0),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=theme.ANNOTATION_SIZE, fontweight="bold",
        color=theme.entity_colour("exact posterior"),
    )  # fmt: skip
    top = max(range(len(grid)), key=lambda i: prior_pdf[i])
    axes.annotate(
        "prior Beta(2, 3)", (grid[top], prior_pdf[top]), xytext=(-8.0, 6.0),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=theme.ANNOTATION_SIZE, fontweight="bold",
        color=theme.role_colour("context"),
    )  # fmt: skip
    theme.axis_labels(axes, "Success probability p", theme.axis_title("density"))
    theme.title_block(
        axes,
        f"NUTS on the logit scale recovers the exact posterior: mean "
        f"{facts['nuts_mean']} against {facts['exact_mean']}",
        f"Bars: {len(draws):,} NUTS draws (2 chains, R-hat {facts['rhat']}) · "
        "solid: exact posterior · dashed: prior",
    )
    return axes.figure, facts


@functools.lru_cache(maxsize=1)
def _engine_summaries() -> tuple[tuple[str, Any], ...]:
    """Run five engines on the conditioned normal program (seeds 1-5).

    The lab settings, except HMC with 200 warm-up and 300 kept draws per
    chain to keep the figure fast. Cached so both palettes share one run.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-KUCUKELBIR-ETAL-2017].

    """
    model = ppl.condition(_normal_model, {"y": NORMAL_DATA})
    fits = _quiet(
        lambda: (
            ("Laplace", ppl.laplace(model, draws=2000, seed=1)),
            ("NUTS", ppl.sample(model, draws=1000, warmup=500, chains=2, seed=2)),
            (
                "HMC",
                ppl.sample(
                    model, method="hmc", draws=300, warmup=200, chains=2, seed=3
                ),
            ),
            (
                "Metropolis",
                ppl.sample(
                    model,
                    method="metropolis",
                    draws=4000,
                    warmup=1000,
                    chains=2,
                    seed=4,
                ),
            ),
            ("ADVI", ppl.fit_advi(model, draws=4000, seed=5)),
        )
    )
    return tuple((label, fit.summary()[0]) for label, fit in fits)


def _ppl_engines() -> tuple[Any, dict[str, str]]:
    """Five inference engines on one conjugate program against the closed form.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2], [REF-HOFFMAN-GELMAN-2014],
        [REF-KUCUKELBIR-ETAL-2017].

    """
    precision = 1.0 / PRIOR_SD**2 + len(NORMAL_DATA) / NORMAL_SIGMA**2
    exact_mean = (
        PRIOR_MEAN / PRIOR_SD**2 + math.fsum(NORMAL_DATA) / NORMAL_SIGMA**2
    ) / precision
    exact_sd = precision**-0.5
    rows = list(_engine_summaries())
    gap = max(abs(s.mean - exact_mean) for _, s in rows)
    facts = {"exact": f"{exact_mean:.4f}", "gap": f"{gap:.3f}"}
    axes = theme.new_axes(figsize=(7.2, 3.8))
    palette = theme.active_palette()
    for edge in (exact_mean - exact_sd, exact_mean + exact_sd):
        axes.axvline(
            edge, color=theme.entity_colour("closed form"), linewidth=1.0,
            linestyle=theme.GUIDE, zorder=2,
        )  # fmt: skip
    theme.reference_line(axes, x=exact_mean, entity="closed form")
    positions = list(range(len(rows)))[::-1]
    for position, (_, s) in zip(positions, rows, strict=True):
        axes.hlines(
            position, s.mean - s.standard_deviation, s.mean + s.standard_deviation,
            color=palette.model, linewidth=2.2, zorder=3,
        )  # fmt: skip
        theme.points(axes, [s.mean], [position], color=palette.model, size=46.0)
    axes.set_yticks(positions)
    axes.set_yticklabels([label for label, _ in rows])
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    axes.annotate(
        f"closed form {facts['exact']} +/- {exact_sd:.4f}", xy=(exact_mean, 1.0),
        xycoords=("data", "axes fraction"), xytext=(6.0, -2.0),
        textcoords="offset points", ha="left", va="top",
        fontsize=theme.ANNOTATION_SIZE, fontweight="semibold",
        color=theme.entity_colour("closed form"),
    )  # fmt: skip
    theme.axis_labels(axes, "Posterior mean of mu, +/- 1 posterior sd")
    theme.title_block(
        axes,
        f"Five engines agree with the closed-form posterior mean {facts['exact']} "
        f"to within {facts['gap']}",
        "Dots and bars: each engine's posterior mean +/- 1 sd · dotted: exact "
        "mean +/- 1 sd",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# nonlinear-calibration-and-mcmc
# --------------------------------------------------------------------------


def _decay(parameters: Any) -> dict[str, float]:
    """Return ``A exp(-k t)``.

    References:
        [EQ-CAL-NL-001], [REF-SEBER-WILD-2003, Chapter 1].

    """
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


@functools.lru_cache(maxsize=1)
def _decay_problem() -> tuple[Any, Any, tuple[float, ...], tuple[float, ...]]:
    """Return the decay calibration of the guide, its posterior, and its data.

    Section 2 data (seed 7, 21 times, noise 0.1); section 8 sampler settings
    (400 warm-up, 800 kept draws) with 2 chains, seed 1; two chain hues
    stay apart under colour-vision deficiency, four do not.

    References:
        [EQ-CAL-STD-001], [EQ-MCMC-AM-001], [REF-HAARIO-SAKSMAN-TAMMINEN-2001].

    """
    prior = JointInput.independent(
        RandomInput("A", Uniform(1.0, 10.0)),
        RandomInput("k", LogNormal(math.log(0.5), 1.0)),
    )
    model = ProbabilisticModel(prior, _decay, (QuantityOfInterest("y", "signal"),))
    rng = random.Random(7)
    times = tuple(0.5 * i for i in range(21))
    values = tuple(5.0 * math.exp(-0.3 * t) + rng.gauss(0.0, 0.1) for t in times)
    experiments = tuple(
        Experiment({"y": y}, {"t": t}) for t, y in zip(times, values, strict=True)
    )
    problem = CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.1),)
    )
    posterior = _quiet(
        lambda: problem.sample_posterior(draws=800, warmup=400, chains=2, seed=1)
    )
    return problem, posterior, times, values


def _bimodal(x: Sequence[float]) -> float:
    """Log density of ``(N(-5, 1) + N(5, 1)) / 2`` up to a constant.

    References:
        [REF-GELMAN-ETAL-2013, Section 11.4]. No registry equation.

    """
    a, b = -0.5 * (x[0] + 5.0) ** 2, -0.5 * (x[0] - 5.0) ** 2
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


def _trace_convergence() -> tuple[Any, dict[str, str]]:
    """Mixed chains of the decay rate beside two chains stuck in two modes.

    References:
        [EQ-MCMC-RHAT-001], [EQ-MCMC-RHAT-002], [REF-VEHTARI-ETAL-2021].

    """
    _, posterior, _, _ = _decay_problem()
    index = posterior.parameter_names.index("k")
    decay_chains = [[s[index] for s in c.samples] for c in posterior.mcmc.chains]
    rhat_k = {e.name: e for e in posterior.summary()}["k"].rank_normalized_rhat
    stuck = _quiet(
        lambda: random_walk_metropolis(
            _bimodal,
            [(-5.0,), (5.0,)],
            proposal_covariance=((0.25,),),
            draws=800,
            chains=2,
            seed=4,
        )
    )
    mode_chains = [[s[0] for s in c.samples] for c in stuck.chains]
    facts = {
        "rhat_k": f"{rhat_k:.3f}",
        "split": f"{split_rhat(mode_chains):.2f}",
        "rank": f"{rank_normalized_rhat(mode_chains):.2f}",
    }
    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.28}
    )
    members = theme.CATEGORY_ORDERS["chains"]
    for axes, chains in ((left, decay_chains), (right, mode_chains)):
        theme.style_axes(axes)
        for number, chain in enumerate(chains):
            style = theme.entity_style(members[number])
            axes.plot(
                range(len(chain)), chain, color=style.colour(),
                linewidth=theme.PATH_WIDTH, alpha=0.8,
            )  # fmt: skip
        axes.set_xlim(0.0, len(chains[0]) - 1.0)
    theme.reference_line(left, y=0.3, label="truth", entity="truth")
    low = min(min(chain) for chain in decay_chains)
    high = max(max(chain) for chain in decay_chains)
    left.set_ylim(low - 0.05 * (high - low), high + 0.4 * (high - low))
    right.set_ylim(-9.0, 9.0)
    theme.axis_labels(left, "Iteration after warm-up", theme.axis_title("decay_rate"))
    theme.axis_labels(right, "Iteration", "x")
    left.annotate(
        f"{len(decay_chains)} chains, R-hat {facts['rhat_k']}", xy=(1.0, 1.0),
        xycoords="axes fraction", xytext=(0.0, -2.0), textcoords="offset points",
        ha="right", va="top", fontsize=theme.ANNOTATION_SIZE,
        color=theme.active_palette().ink,
    )  # fmt: skip
    right.annotate(
        f"2 chains, R-hat {facts['split']} (split), {facts['rank']} (rank)",
        xy=(1.0, 0.5), xycoords="axes fraction", xytext=(0.0, 0.0),
        textcoords="offset points", ha="right", va="center",
        fontsize=theme.ANNOTATION_SIZE, color=theme.active_palette().ink,
    )  # fmt: skip
    theme.title_block(
        left,
        f"Overlapping chains give R-hat {facts['rhat_k']} for the decay rate; "
        f"two chains trapped in separate modes give {facts['split']}",
        "Left: decay rate k, one colour per chain · right: random-walk "
        "Metropolis on modes at -5 and +5, one chain started in each",
    )
    return figure, facts


def _decay_prediction() -> tuple[Any, dict[str, str]]:
    """Latent and observation bands of the calibrated decay up to t = 12.

    References:
        [EQ-CAL-PRED-001], [REF-GELMAN-ETAL-2013, Section 6.3].

    """
    problem, posterior, times, values = _decay_problem()
    grid = tuple(0.25 * i for i in range(49))
    prediction = predict(
        problem,
        posterior,
        conditions=tuple({"t": t} for t in grid),
        include_noise=True,
        max_draws=1600,
        seed=3,
    )
    items = prediction.predictions
    latent = [item.interval(0.95) for item in items]
    observed = [item.interval(0.95, observed=True) for item in items]
    last_latent, last_observed = latent[-1], observed[-1]
    ratio = (last_observed[1] - last_observed[0]) / (last_latent[1] - last_latent[0])
    facts = {
        "ratio": f"{ratio:.0f}",
        "latent": f"({last_latent[0]:.3f}, {last_latent[1]:.3f})",
        "observed": f"({last_observed[0]:.3f}, {last_observed[1]:.3f})",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_calibration_prediction(
        times,
        values,
        grid,
        [item.mean for item in items],
        ([low for low, _ in latent], [high for _, high in latent]),
        ([low for low, _ in observed], [high for _, high in observed]),
        calibration_limit=max(times),
        x_label=theme.axis_title("time"),
        y_label="Signal y",
        title=(
            f"At t = 12, beyond the data, a new measurement is {facts['ratio']} "
            "times as uncertain as the mean response"
        ),
        ax=axes,
    )
    return figure, facts


# --------------------------------------------------------------------------
# inverse-problems-and-calibration
# --------------------------------------------------------------------------

LINE_PRIOR = ((0.0, 0.0), ((25.0, 0.0), (0.0, 4.0)))


@functools.lru_cache(maxsize=1)
def _line_calibration() -> tuple[Any, tuple[float, ...], tuple[float, ...]]:
    """Calibrate the lab's straight line: eight points of 1.25 + 0.82 x.

    References:
        [EQ-LAB-CAL-001], [EQ-BAYES-LIN-POST-001], [REF-GELMAN-ETAL-2013].

    """
    inputs = tuple(float(i) for i in range(8))
    generator = random.Random(2026)
    observations = tuple(1.25 + 0.82 * x + generator.gauss(0.0, 0.35) for x in inputs)
    result = calibrate_linear_gaussian(
        inputs,
        observations,
        observation_variance=0.35**2,
        prior=MultivariateNormal(*LINE_PRIOR),
    )
    return result, inputs, observations


def _line_band(
    mean: Sequence[float], covariance: Sequence[Sequence[float]], x: float
) -> tuple[float, float]:
    """Return the mean and 1.96-sd half-width of ``a + b x`` under a Gaussian.

    References:
        [EQ-BAYES-PRED-001]. Linear map of a Gaussian vector.

    """
    centre = mean[0] + mean[1] * x
    variance = covariance[0][0] + 2.0 * x * covariance[0][1] + x * x * covariance[1][1]
    return centre, 1.96 * math.sqrt(variance)


def _prior_to_posterior() -> tuple[Any, dict[str, str]]:
    """Draw the 95% band of the line before and after the eight measurements.

    References:
        [EQ-BAYES-LIN-POST-001], [EQ-BAYES-PRED-001],
        [REF-GELMAN-ETAL-2013, Section 14.2].

    """
    result, inputs, observations = _line_calibration()
    prior_mean, prior_cov = LINE_PRIOR
    post = result.posterior
    grid = [7.0 * i / 140 for i in range(141)]
    bands = {
        "prior": [_line_band(prior_mean, prior_cov, x) for x in grid],
        "posterior": [_line_band(post.mean, post.covariance, x) for x in grid],
    }
    probe = 3.5
    prior_half = _line_band(prior_mean, prior_cov, probe)[1]
    post_half = _line_band(post.mean, post.covariance, probe)[1]
    facts = {
        "prior": f"+/-{prior_half:.1f}",
        "posterior": f"+/-{post_half:.2f}",
        "ratio": f"{prior_half / post_half:.0f}",
    }
    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=WIDE, sharey=True, gridspec_kw={"wspace": 0.12}
    )
    palette = theme.active_palette()
    for axes, key in ((left, "prior"), (right, "posterior")):
        theme.style_axes(axes)
        centre = [c for c, _ in bands[key]]
        colour = palette.muted if key == "prior" else palette.model
        theme.band(
            axes, grid, [c - h for c, h in bands[key]],
            [c + h for c, h in bands[key]], color=colour, alpha=theme.MAX_BAND_ALPHA,
        )  # fmt: skip
        theme.line(axes, grid, centre, color=colour)
        axes.set_xlim(0.0, 7.0)
        axes.annotate(
            f"{key}: 95% band {facts[key]} at x = {probe:g}", xy=(0.03, 0.97),
            xycoords="axes fraction", ha="left", va="top",
            fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=colour,
        )  # fmt: skip
    theme.points(right, inputs, observations, entity="observations", size=30.0)
    theme.axis_labels(left, "Input x", "Response a + b x")
    theme.axis_labels(right, "Input x", None)
    theme.title_block(
        left,
        f"Eight measurements narrow the 95% band of the line at x = 3.5 from "
        f"{facts['prior']} to {facts['posterior']}, {facts['ratio']} times",
        "Bands: 95% of a + b x under the Gaussian prior (left) and the exact "
        "conjugate posterior (right) · dots: measurements",
    )
    return figure, facts


def _line_prediction() -> tuple[Any, dict[str, str]]:
    """Parameter band and future-observation band of the calibrated line.

    References:
        [EQ-BAYES-PRED-001], [EQ-CAL-DISC-001], [EQ-BAYES-PPC-001],
        [REF-GELMAN-ETAL-2013, Section 3.3].

    """
    result, inputs, observations = _line_calibration()
    discrepancy = AdditiveGaussianDiscrepancy(
        0.15,
        "Residual linear-model inadequacy retained for prediction.",
        "Teaching system for inputs from 0 through 9.",
    )
    grid = tuple(i * 0.05 for i in range(181))
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
    check = posterior_predictive_check(
        posterior_predictive(
            result.posterior,
            result.design_rows(inputs),
            include_observation_noise=True,
            discrepancy_variance=discrepancy.variance,
        ),
        observations,
        draws=2000,
        seed=2026,
    )
    inner = [math.sqrt(v) for v in latent.parameter_variance]
    outer = [math.sqrt(future.covariance[i][i]) for i in range(len(grid))]
    ratio = outer[-1] / inner[-1]
    facts = {
        "ratio": f"{ratio:.1f}",
        "tail": f"{check.tail_probability:.3f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_calibration_prediction(
        inputs,
        observations,
        grid,
        latent.mean,
        (
            [m - 1.96 * s for m, s in zip(latent.mean, inner, strict=True)],
            [m + 1.96 * s for m, s in zip(latent.mean, inner, strict=True)],
        ),
        (
            [m - 1.96 * s for m, s in zip(future.mean, outer, strict=True)],
            [m + 1.96 * s for m, s in zip(future.mean, outer, strict=True)],
        ),
        calibration_limit=max(inputs),
        title=(
            f"At x = 9 the band for a future measurement is {facts['ratio']} "
            "times as wide as the band for the mean response"
        ),
        ax=axes,
    )
    # Headroom keeps the top direct label clear of the calibration-end label.
    axes.set_ylim(0.0, 10.6)
    return figure, facts


# --------------------------------------------------------------------------
# model-discrepancy
# --------------------------------------------------------------------------

DISCREPANCY_X = tuple(4.0 * i / 14 for i in range(15))


def _biased_truth(x: float) -> float:
    """Return the true process ``4 exp(-0.4 x) + 0.25 sin(2.5 x)``.

    References:
        [EQ-KOH-MODEL-001]. Teaching scenario of the guide.

    """
    return 4.0 * math.exp(-0.4 * x) + 0.25 * math.sin(2.5 * x)


@functools.lru_cache(maxsize=1)
def _discrepancy_runs() -> tuple[Any, Any, Any]:
    """Return the KOH problem, its posterior, and the plain posterior.

    Guide section 4: 15 points with noise 0.03 (seed 11), 2 chains of 1000
    draws after 500 warm-up (seed 3).

    References:
        [EQ-KOH-MARG-001], [EQ-KOH-POST-001], [REF-KENNEDY-OHAGAN-2001].

    """
    prior = JointInput.independent(RandomInput("k", LogNormal(math.log(0.5), 0.5)))
    model = ProbabilisticModel(
        prior,
        lambda p: {"y": 4.0 * math.exp(-p["k"] * p["x"])},
        (QuantityOfInterest("y", "y"),),
    )
    rng = random.Random(11)
    experiments = tuple(
        Experiment({"y": _biased_truth(x) + rng.gauss(0.0, 0.03)}, {"x": x})
        for x in DISCREPANCY_X
    )
    process = DiscrepancyProcess(
        ("x",),
        kernel="squared_exponential",
        standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
        length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
    )
    problem = DiscrepancyCalibration(model, experiments, "y", KnownNoise(0.03), process)
    plain = CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.03),)
    )
    posterior, plain_posterior = _quiet(
        lambda: (
            problem.sample_posterior(draws=1000, warmup=500, chains=2, seed=3),
            plain.sample_posterior(draws=1000, warmup=500, chains=2, seed=3),
        )
    )
    return problem, posterior, plain_posterior


def _discrepancy_bands() -> tuple[Any, dict[str, str]]:
    """Simulator-only and bias-corrected 95% bands against the true process.

    References:
        [EQ-KOH-PRED-001], [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004].

    """
    problem, posterior, _ = _discrepancy_runs()
    grid = tuple(4.5 * i / 44 for i in range(45))
    predictions = problem.predict(
        posterior, tuple({"x": x} for x in grid), max_draws=300
    )
    truth = [_biased_truth(x) for x in grid]
    corrected = [item.interval(0.95, kind="corrected") for item in predictions]
    simulator = [item.interval(0.95, kind="simulator") for item in predictions]
    facts = {
        "corrected": str(
            sum(lo <= v <= hi for (lo, hi), v in zip(corrected, truth, strict=True))
        ),
        "simulator": str(
            sum(lo <= v <= hi for (lo, hi), v in zip(simulator, truth, strict=True))
        ),
        "points": str(len(grid)),
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_discrepancy_predictions(problem, predictions, "x", truth=truth, ax=axes)
    _check_title(axes, facts)
    return figure, facts


def _discrepancy_rate() -> tuple[Any, dict[str, str]]:
    """Posterior intervals of k with and without a discrepancy term.

    References:
        [EQ-KOH-POST-001], [EQ-CAL-POST-001], [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """
    _, posterior, plain_posterior = _discrepancy_runs()

    def pooled(result: Any) -> list[float]:
        """Return the pooled draws of ``k``.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        column = result.mcmc.names.index("k")
        return [s[column] for chain in result.mcmc.chains for s in chain.samples]

    rows = []
    for label, result, entity in (
        ("no discrepancy", plain_posterior, "simulator only"),
        ("Kennedy-O'Hagan", posterior, "bias-corrected"),
    ):
        draws = sorted(pooled(result))
        mean = math.fsum(draws) / len(draws)
        sd = math.sqrt(math.fsum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
        rows.append(
            (label, entity, mean, sd, quantile(draws, 0.05), quantile(draws, 0.95))
        )
    plain, koh = rows
    facts = {
        "ratio": f"{koh[3] / plain[3]:.0f}",
        "plain": f"{plain[4]:.4f}..{plain[5]:.4f}",
        "koh": f"{koh[4]:.4f}..{koh[5]:.4f}",
    }
    axes = theme.new_axes(figsize=(7.2, 3.2))
    positions = [1.0, 0.0]
    for position, (_, entity, mean, _, low, high) in zip(positions, rows, strict=True):
        colour = theme.entity_colour(entity)
        axes.hlines(position, low, high, color=colour, linewidth=3.0, zorder=3)
        theme.points(axes, [mean], [position], color=colour, size=56.0)
        axes.annotate(
            f"90% {low:.4f}..{high:.4f}", (high, position), xytext=(8.0, 0.0),
            textcoords="offset points", ha="left", va="center",
            fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=colour,
        )  # fmt: skip
    theme.reference_line(axes, x=0.4, label="true k = 0.4", entity="truth")
    axes.set_yticks(positions)
    axes.set_yticklabels([row[0] for row in rows])
    axes.set_ylim(-0.6, 1.6)
    axes.set_xlim(0.30, 0.54)
    palette = theme.active_palette()
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    theme.axis_labels(axes, "Decay rate k")
    theme.title_block(
        axes,
        f"Without a discrepancy term the posterior of k is {facts['ratio']} times "
        "narrower and misses the true value",
        "Dots: posterior means · bars: central 90% intervals · "
        "2 chains of 1,000 draws each",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# generative-models
# --------------------------------------------------------------------------


def _banana(size: int, seed: int) -> list[tuple[float, float]]:
    """Draw from ``x1 ~ N(0, 1)``, ``x2 | x1 ~ N(x1^2 / 2 - 1, 0.35^2)``.

    References:
        [REF-PAPAMAKARIOS-ETAL-2021, Section 2]. Teaching target of ADR-0074.

    """
    rng = random.Random(seed)
    points = []
    for _ in range(size):
        a = rng.gauss(0.0, 1.0)
        points.append((a, 0.5 * a * a - 1.0 + 0.35 * rng.gauss(0.0, 1.0)))
    return points


def _banana_log_density(x: Sequence[float]) -> float:
    """Return the exact log density of the banana target.

    References:
        [EQ-DIST-NORM-001]. Product of two normal densities.

    """
    return Normal(0.0, 1.0).log_pdf(x[0]) + Normal(0.5 * x[0] ** 2 - 1.0, 0.35).log_pdf(
        x[1]
    )


def _hdr_levels(values: Sequence[float], cell: float) -> list[float]:
    """Return density levels whose upper sets hold 90% and 50% of the mass.

    References:
        [REF-PROBSTAT-REPORTING]. Grid approximation of highest-density
        regions; no registry equation.

    """
    ordered = sorted(values, reverse=True)
    total = math.fsum(ordered) * cell
    levels: list[float] = []
    for share in (0.9, 0.5):
        running = 0.0
        for value in ordered:
            running += value * cell
            if running >= share * total:
                levels.append(value)
                break
    return levels


@functools.lru_cache(maxsize=1)
def _banana_fits() -> tuple[Any, Any, list[Any], list[Any]]:
    """Fit the guide's two-coupling flow and one Gaussian to banana data.

    Cached so both palettes share one fit.

    References:
        [EQ-GEN-FLOW-007], [REF-PAPAMAKARIOS-ETAL-2021, Section 2.3.1].

    """
    data, heldout = _banana(300, 5), _banana(300, 6)
    fit = fit_flow(
        data,
        real_nvp_flow(2, couplings=2, hidden=(8,)),
        iterations=300,
        learning_rate=0.02,
        heldout=heldout,
        seed=3,
    )
    return fit, fit_multivariate_normal_mle(data), data, heldout


def _flow_versus_gaussian() -> tuple[Any, dict[str, str]]:
    """Density regions of one Gaussian and a RealNVP flow over banana data.

    References:
        [EQ-GEN-FLOW-001], [EQ-GEN-FLOW-007], [EQ-GEN-DIAG-002],
        [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017].

    """
    fit, gaussian, data, heldout = _banana_fits()
    scores = {
        "flow": fit.heldout_mean_log_likelihood,
        "gaussian": mean_log_likelihood(gaussian.log_pdf, heldout).estimate,
        "truth": mean_log_likelihood(_banana_log_density, heldout).estimate,
    }
    facts = {key: f"{value:.2f}" for key, value in scores.items()}
    xs = [-3.2 + 6.4 * i / 63 for i in range(64)]
    ys = [-2.4 + 6.4 * j / 63 for j in range(64)]
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    plt = theme.pyplot()
    figure, panels = plt.subplots(
        1, 2, figsize=WIDE, sharey=True, gridspec_kw={"wspace": 0.08}
    )
    palette = theme.active_palette()
    for axes, key, log_density in (
        (panels[0], "gaussian", gaussian.log_pdf),
        (panels[1], "flow", fit.log_density),
    ):
        theme.style_axes(axes)
        grid = [[math.exp(log_density((x, y))) for x in xs] for y in ys]
        levels = _hdr_levels([v for row in grid for v in row], cell)
        axes.scatter(
            [p[0] for p in data], [p[1] for p in data], s=9.0,
            color=palette.observed, alpha=0.55, linewidths=0.0, zorder=2,
        )  # fmt: skip
        axes.contour(
            xs, ys, grid, levels=levels, colors=[palette.model],
            linewidths=[1.2, 2.2], zorder=3,
        )  # fmt: skip
        name = "one Gaussian" if key == "gaussian" else "two-coupling flow"
        axes.annotate(
            f"{name}: held-out {facts[key]}", xy=(0.5, 0.98),
            xycoords="axes fraction", ha="center", va="top",
            fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=palette.model,
        )  # fmt: skip
        axes.set_xlim(xs[0], xs[-1])
        axes.set_ylim(ys[0], ys[-1])
    theme.axis_labels(panels[0], "x1", "x2")
    theme.axis_labels(panels[1], "x1", None)
    theme.title_block(
        panels[0],
        f"A flow bends to the banana: held-out log-likelihood {facts['flow']} "
        f"against {facts['gaussian']} for one Gaussian ({facts['truth']} for the "
        "truth)",
        "Dots: 300 training points · thin and thick contours: regions holding "
        "90% and 50% of each fitted density",
    )
    return figure, facts


# --------------------------------------------------------------------------
# probabilistic-modeling
# --------------------------------------------------------------------------


def _latent_and_physical() -> tuple[Any, dict[str, str]]:
    """Latent normals and physical lognormals of one Gaussian-copula input.

    References:
        [EQ-MOD-NATAF-001], [EQ-MOD-NATAF-002], [REF-LIU-DERKIUREGHIAN-1986].

    """
    joint = JointInput.from_pearson(
        (
            RandomInput("x1", LogNormal(0.0, 0.5)),
            RandomInput("x2", LogNormal(0.0, 0.8)),
        ),
        ((1.0, 0.6), (0.6, 1.0)),
    )
    assert joint.correlation is not None
    latent_rho = joint.correlation[0][1]
    draws = joint.sample(30000, seed=1)
    physical = pearson_correlation([d[0] for d in draws], [d[1] for d in draws])
    shown = draws[:1500]
    latent = [joint.to_latent(point) for point in shown]
    facts = {"latent": f"{latent_rho:.3f}", "physical": f"{physical:.3f}"}
    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.28}
    )
    palette = theme.active_palette()
    for axes, xs, ys in (
        (left, [z[0] for z in latent], [z[1] for z in latent]),
        (right, [p[0] for p in shown], [p[1] for p in shown]),
    ):
        theme.style_axes(axes)
        axes.scatter(
            xs, ys, s=7.0, color=palette.observed, alpha=0.45, linewidths=0.0,
            zorder=2,
        )  # fmt: skip
    left.set_xlim(-4.0, 4.0)
    left.set_ylim(-4.0, 4.0)
    right.set_xlim(0.0, 5.0)
    right.set_ylim(0.0, 9.0)
    left.annotate(
        f"latent correlation {facts['latent']}", xy=(0.03, 0.97),
        xycoords="axes fraction", ha="left", va="top",
        fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=palette.ink,
    )  # fmt: skip
    right.annotate(
        f"Pearson correlation {facts['physical']}", xy=(0.97, 0.97),
        xycoords="axes fraction", ha="right", va="top",
        fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=palette.ink,
    )  # fmt: skip
    theme.axis_labels(left, "Latent normal z1", "Latent normal z2")
    theme.axis_labels(
        right, "Input x1, LogNormal(0, 0.5)", "Input x2, LogNormal(0, 0.8)"
    )
    outside = sum(p[0] > 5.0 or p[1] > 9.0 for p in shown)
    theme.title_block(
        left,
        f"A latent correlation of {facts['latent']} gives the two lognormal "
        f"inputs a Pearson correlation of {facts['physical']}, against the "
        "target 0.6",
        "1,500 joint draws in the latent normal space (left) and the physical "
        f"space (right, {outside} beyond the axes) · correlation from 30,000 draws",
    )
    return figure, facts


def _ishigami() -> ProbabilisticModel:
    """Return the Ishigami function on three uniform inputs.

    References:
        [REF-SALTELLI-2008].

    """
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )

    def function(p: Any) -> dict[str, float]:
        """Evaluate the Ishigami output.

        References:
            [REF-SALTELLI-2008].

        """
        return {
            "y": math.sin(p["x1"])
            + 7.0 * math.sin(p["x2"]) ** 2
            + 0.1 * p["x3"] ** 4 * math.sin(p["x1"])
        }

    return ProbabilisticModel(
        inputs, function, (QuantityOfInterest("y", "Ishigami output"),)
    )


def _design_comparison() -> tuple[Any, dict[str, str]]:
    """Replicate spread of the Ishigami mean for four sampling designs.

    References:
        [EQ-SAMP-RQMC-001], [REF-OWEN-2013, Section 17.2].

    """
    summary = compare_designs(_ishigami(), "y", 256, replicates=8, seed=1)
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_design_comparison(summary, size=256, ax=axes)
    facts = {
        "gain": f"{summary['mc'][1] / summary['sobol'][1]:.0f}x",
        "mc": f"{summary['mc'][1]:.3g}",
        "sobol": f"{summary['sobol'][1]:.3g}",
    }
    _check_title(axes, {"gain": facts["gain"]})
    return figure, facts


# --------------------------------------------------------------------------
# optimal-transport
# --------------------------------------------------------------------------


def _plane_coupling() -> tuple[Any, dict[str, str]]:
    """Exact and entropic couplings of the lab's 6 + 6 points in the plane.

    References:
        [EQ-OT-LP-001], [EQ-OT-SIMPLEX-001], [EQ-OT-SINK-001],
        [REF-PEYRE-CUTURI-2019, Chapters 3-4], [REF-CUTURI-2013].

    """
    rng = random.Random(2026)
    xs = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(6)]
    ys = [(rng.gauss(1.5, 1), rng.gauss(0.5, 1)) for _ in range(6)]
    weights = [1.0 / 6.0] * 6
    cost = squared_euclidean_cost_matrix(xs, ys)
    exact = exact_transport(weights, weights, cost)
    epsilon = 0.5
    soft = sinkhorn(weights, weights, cost, epsilon=epsilon, max_iterations=50_000)
    used = sum(v >= 0.005 for row in soft.plan for v in row)
    facts = {
        "cost": f"{exact.cost:.3f}",
        "epsilon": f"eps = {epsilon:g}",
        "cells": f"{used} of 36",
        "gap": f"{soft.transport_cost - exact.cost:.3f}",
    }
    plt = theme.pyplot()
    figure, panels = plt.subplots(
        1, 2, figsize=WIDE, sharex=True, sharey=True, gridspec_kw={"wspace": 0.08}
    )
    palette = theme.active_palette()
    ramp = theme.sequential_colormap()
    for axes, plan in ((panels[0], exact.plan), (panels[1], soft.plan)):
        theme.style_axes(axes)
        peak = max(v for row in plan for v in row)
        entries = sorted(
            (v, i, j) for i, row in enumerate(plan) for j, v in enumerate(row)
        )
        for mass, i, j in entries:
            if mass < 0.005:
                continue
            depth = mass / peak
            axes.plot(
                [xs[i][0], ys[j][0]], [xs[i][1], ys[j][1]],
                color=ramp(0.35 + 0.65 * depth), linewidth=0.6 + 4.0 * depth,
                solid_capstyle="round", zorder=2,
            )  # fmt: skip
        theme.points(
            axes, [p[0] for p in xs], [p[1] for p in xs], color=palette.observed,
            size=58.0, zorder=5,
        )  # fmt: skip
        theme.points(
            axes, [p[0] for p in ys], [p[1] for p in ys], color=palette.model,
            size=58.0, marker="s", zorder=5,
        )  # fmt: skip
    panels[0].annotate(
        f"exact: cost {facts['cost']}", xy=(0.03, 0.03), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=theme.ANNOTATION_SIZE, fontweight="bold",
        color=palette.ink,
    )  # fmt: skip
    panels[1].annotate(
        f"Sinkhorn, {facts['epsilon']}: {facts['cells']} pairs", xy=(0.03, 0.03),
        xycoords="axes fraction", ha="left", va="bottom",
        fontsize=theme.ANNOTATION_SIZE, fontweight="bold", color=palette.ink,
    )  # fmt: skip
    panels[0].set_ylim(-2.5, 2.1)
    # The ramp runs toward the ink: darker on the light canvas, brighter on dark.
    from matplotlib.colors import to_rgb

    depth_word = "darker" if sum(to_rgb(palette.canvas)) > 1.5 else "brighter"
    theme.axis_labels(panels[0], "First coordinate", "Second coordinate")
    theme.axis_labels(panels[1], "First coordinate", None)
    theme.title_block(
        panels[0],
        f"The exact plan pairs each point once at cost {facts['cost']}; Sinkhorn at "
        f"{facts['epsilon']} spreads mass over {facts['cells']} pairs and costs "
        f"{facts['gap']} more",
        "Circles: source points · squares: target points · lines: plan "
        f"entries, wider and {depth_word} for more mass",
    )
    return figure, facts


def _quantile_w1() -> tuple[Any, dict[str, str]]:
    """W1 between Gamma(2, 1) and Exp(0.8) as the area between quantiles.

    References:
        [EQ-OT-W1D-001], [REF-PEYRE-CUTURI-2019, Section 2.6].

    """
    first, second = Gamma(2.0, 1.0), Exponential(0.8)
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_quantile_wasserstein(first, second, names=("Gamma(2, 1)", "Exp(0.8)"), ax=axes)
    facts = {"w1": f"{quantile_wasserstein(first, second, p=1.0):.3f}"}
    _check_title(axes, facts)
    return figure, facts


FIGURES = [
    GuideFigure(
        name="hierarchical-schools-shrinkage",
        guide="hierarchical-models",
        alt=(
            "Eight school effects, each drawn as a raw estimate with one standard "
            "error above its posterior mean and 95% interval, joined by arrows"
        ),
        draw=_schools_shrinkage,
    ),
    GuideFigure(
        name="hierarchical-mixture-fit",
        guide="hierarchical-models",
        alt=(
            "Histogram of 300 observations with the fitted Gaussian-mixture "
            "density and its weighted components"
        ),
        draw=_mixture_fit,
    ),
    GuideFigure(
        name="ppl-beta-binomial-posterior",
        guide="probabilistic-programming",
        alt=(
            "Histogram of NUTS draws of a success probability over the exact "
            "Beta posterior density and the dashed Beta prior"
        ),
        draw=_ppl_beta_binomial,
    ),
    GuideFigure(
        name="ppl-engines-closed-form",
        guide="probabilistic-programming",
        alt=(
            "Posterior mean and one standard deviation of a normal mean from five "
            "inference engines beside the closed-form value"
        ),
        draw=_ppl_engines,
    ),
    GuideFigure(
        name="calibration-trace-convergence",
        guide="nonlinear-calibration-and-mcmc",
        alt=(
            "Two trace panels: two MCMC chains of a decay rate, and two "
            "random-walk chains on a target with two separated modes"
        ),
        draw=_trace_convergence,
    ),
    GuideFigure(
        name="calibration-decay-prediction",
        guide="nonlinear-calibration-and-mcmc",
        alt=(
            "Decay measurements with the calibrated mean curve, its parameter "
            "band and its wider observation band extended to t = 12"
        ),
        draw=_decay_prediction,
    ),
    GuideFigure(
        name="inverse-prior-to-posterior",
        guide="inverse-problems-and-calibration",
        alt=(
            "Two panels with the 95% band of a straight line under the prior and "
            "under the posterior, with the eight measurements"
        ),
        draw=_prior_to_posterior,
    ),
    GuideFigure(
        name="inverse-calibration-prediction",
        guide="inverse-problems-and-calibration",
        alt=(
            "Eight measurements with the posterior mean line, its parameter band "
            "and its future-observation band extended to x = 9"
        ),
        draw=_line_prediction,
    ),
    GuideFigure(
        name="discrepancy-bias-corrected-bands",
        guide="model-discrepancy",
        alt=(
            "Observations, true process, simulator-only band and bias-corrected "
            "band of an exponential-decay simulator over the input x"
        ),
        draw=_discrepancy_bands,
    ),
    GuideFigure(
        name="discrepancy-rate-intervals",
        guide="model-discrepancy",
        alt=(
            "Posterior mean and 90% interval of the decay rate k with and without "
            "a discrepancy term, beside the true value"
        ),
        draw=_discrepancy_rate,
    ),
    GuideFigure(
        name="generative-flow-versus-gaussian",
        guide="generative-models",
        alt=(
            "Two panels of banana-shaped training points with the 50% and 90% "
            "density contours of one Gaussian and of a normalizing flow"
        ),
        draw=_flow_versus_gaussian,
    ),
    GuideFigure(
        name="modeling-latent-physical-correlation",
        guide="probabilistic-modeling",
        alt=(
            "Two scatter panels of the same joint draws: latent standard normals "
            "and the two lognormal physical inputs"
        ),
        draw=_latent_and_physical,
    ),
    GuideFigure(
        name="modeling-design-comparison",
        guide="probabilistic-modeling",
        alt=(
            "Spread of replicate mean estimates of the Ishigami function for four "
            "sampling designs on a logarithmic axis"
        ),
        draw=_design_comparison,
    ),
    GuideFigure(
        name="transport-plane-coupling",
        guide="optimal-transport",
        alt=(
            "Two panels of six source and six target points in the plane joined by "
            "the exact and by an entropic transport plan"
        ),
        draw=_plane_coupling,
    ),
    GuideFigure(
        name="transport-quantile-w1",
        guide="optimal-transport",
        alt=(
            "Quantile functions of a Gamma and an exponential law with the area "
            "between them shaded"
        ),
        draw=_quantile_w1,
    ),
]
