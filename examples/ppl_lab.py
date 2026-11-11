"""Probabilistic-programming laboratory: models as functions, checked exactly.

Four seeded experiments, printed as numbers:

1. Eight schools with a half-Cauchy prior on tau, written as a PPL model
   (non-centred), against the exact grid posterior; the centred form for
   comparison.
2. A conjugate normal-normal model: closed form against Laplace, NUTS, HMC,
   adaptive Metropolis, and ADVI run on the same model function.
3. A beta-binomial model: the logit transform and its Jacobian checked by
   integrating the unconstrained density, and NUTS against the exact Beta.
4. Prior and posterior predictive simulation and posterior predictive
   p-values for eight schools.

With Matplotlib installed one figure is written to ``examples/generated/``:
the non-centred NUTS draws of ``tau`` over the exact grid posterior.

Definitions:
    model function: a Python function that declares latent sites with
        ``m.sample`` and observed sites with ``m.observe``; tracing it gives
        the joint log-density.
    unconstrained space: coordinates obtained by a log bijection for
        ``tau > 0`` and a logit bijection for ``0 < p < 1``, with the
        log-Jacobian added to the density.
    eight schools: ``y_j ~ N(theta_j, s_j)``, ``theta_j ~ N(mu, tau)``,
        ``mu ~ N(0, 5)``, ``tau ~ half-Cauchy(5)``.
    non-centred form: ``theta_j = mu + tau z_j`` with ``z_j ~ N(0, 1)``.
    MCSE, ESS, R-hat: Monte Carlo standard error, effective sample size,
        and rank-normalized split R-hat of a posterior summary.
    posterior predictive p-value: share of replicated data sets whose test
        statistic is at least the observed one.

Assumptions:
    - The exact references are trustworthy: the eight-schools grid posterior
      integrates ``tau`` numerically, and the normal-normal and
      beta-binomial posteriors are conjugate closed forms.
    - Chains are long enough for their MCSE to describe the remaining error;
      the convergence warnings printed for the short centred run say when
      this fails.
    - Laplace and ADVI are Gaussian approximations; they are exact only for
      the normal-normal model.

Equations:
    EQ-PPL-JOINT-001: joint log-density on the unconstrained space.
    EQ-PPL-TRANSFORM-001: log bijection for ``tau``.
    EQ-PPL-TRANSFORM-002: logit bijection for ``p``.
    EQ-PPL-NONCENTERED-001: non-centred reparameterization of ``theta``.
    EQ-PPL-PREDICTIVE-001: prior and posterior predictive simulation.
    EQ-HIER-001: conditional posteriors of the hierarchical normal model.
    EQ-HIER-002: exact marginal posterior of ``tau`` on a grid.
    EQ-HMC-H-001: Hamiltonian with a Gaussian momentum.
    EQ-HMC-LEAPFROG-001: leapfrog integrator.
    EQ-NUTS-UTURN-001: no-U-turn termination.
    EQ-HMC-BFMI-001: energy Bayesian fraction of missing information.
    EQ-MCMC-AM-001: adaptive Metropolis proposal.
    EQ-LAPLACE-001: Laplace approximation.
    EQ-VI-ELBO-001: evidence lower bound maximized by ADVI.
    EQ-MCMC-RHAT-002: rank-normalized split R-hat.
    EQ-MCMC-ESS-001: effective sample size.
    EQ-MCMC-MCSE-001: Monte Carlo standard error of a posterior mean.
    EQ-PPC-001: posterior predictive p-value.

Diagnostics:
    - Every sampled mean is printed beside its exact value and its MCSE;
      a gap larger than about two MCSE signals a sampler or model error.
    - Divergences, R-hat, ESS, E-BFMI, and tree-depth hits contrast the
      non-centred and centred eight-schools posteriors (the funnel).
    - The integral of the unconstrained beta-binomial density equals the
      exact evidence, which checks the logit Jacobian.
    - Predictive p-values near 0 or 1 flag a test statistic the model cannot
      reproduce; the figure compares the sampled ``tau`` with its exact law.

Cost and error:
    - Non-centred NUTS: 2 chains x (300 warm-up + 300 draws); centred:
      2 x (150 + 150) with tree depth at most 6.
    - Normal-normal: Laplace 2,000 draws, NUTS and HMC 2 x (500 + 1,000),
      adaptive Metropolis 2 x (1,000 + 4,000), ADVI 4,000 draws.
    - Beta-binomial: a 6,001-point trapezoid on ``[-30, 30]`` and NUTS
      2 x (300 + 1,000).
    - Predictive: 2,000 prior and 600 posterior predictive data sets; each
      p-value carries the printed MCSE.
    - The NUTS timing and the final line print wall-clock times.

References:
    REF-GELMAN-ETAL-2013: hierarchical models and predictive checks.
    REF-GELMAN-2006: half-Cauchy priors for variance parameters.
    REF-HOFFMAN-GELMAN-2014: the no-U-turn sampler.
    REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007: centred and non-centred forms.
    REF-CARPENTER-ETAL-2017: constrained-space transforms in Stan.
    REF-BINGHAM-ETAL-2019: models as programs (Pyro).
    REF-VEHTARI-ETAL-2021: rank-normalized R-hat and ESS.
    REF-GELMAN-MENG-STERN-1996: posterior predictive p-values.
    REF-MATPLOTLIB: rendering.

"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.distributions import Beta, Binomial, Exponential, Normal, RealSupport
from probstat.inference import half_cauchy_log_prior, hierarchical_normal_posterior, ppl
from probstat.inference.ppl import ModelHandle
from probstat.visualization import save_figure
from probstat.visualization.hierarchical_plots import plot_tau_posterior
from probstat.visualization.theme import matplotlib_available

# Seed of both eight-schools NUTS runs.
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
CAUCHY_SCALE = 5.0
MU_PRIOR = (0.0, 5.0)

NORMAL_DATA = (4.1, 5.3, 3.8, 6.0, 4.9, 5.5, 4.4, 5.1, 6.2, 3.9)
NORMAL_SIGMA = 1.2
PRIOR_MEAN, PRIOR_SD = 0.0, 3.0

BETA_A, BETA_B, TRIALS, SUCCESSES = 2.0, 3.0, 20, 13


@dataclass(frozen=True)
class HalfCauchy:
    """User-defined half-Cauchy law: any object with the three methods works.

    References:
        [REF-GELMAN-2006].

    """

    scale: float

    @property
    def support(self) -> RealSupport:
        """Return ``[0, inf)``.

        References:
            [REF-GELMAN-2006]. No equation is evaluated.

        """
        return RealSupport(0.0, None, lower_closed=True)

    def log_pdf(self, x: float) -> float:
        """Return ``log(2 / (pi A)) - log(1 + (x / A)^2)`` for ``x >= 0``.

        References:
            [REF-GELMAN-2006].

        """
        if x < 0.0:
            return -math.inf
        return math.log(2.0 / (math.pi * self.scale)) - math.log1p(
            (x / self.scale) ** 2
        )

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw ``A tan(pi U / 2)`` with ``U`` uniform (inverse CDF).

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        uniform = Exponential(1.0).sample(size, seed=seed)
        return tuple(
            self.scale * math.tan(0.5 * math.pi * (1.0 - math.exp(-e))) for e in uniform
        )


def schools(m: ModelHandle, *, noncentered: bool = True) -> None:
    """Eight-schools model with a half-Cauchy prior on ``tau``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    mu = m.sample("mu", Normal(*MU_PRIOR))
    tau = m.sample("tau", HalfCauchy(CAUCHY_SCALE))
    theta = m.sample(
        "theta",
        Normal(mu, tau),
        shape=8,
        reparam="noncentered" if noncentered else None,
    )
    m.observe("y", ppl.elementwise(Normal, theta, SCHOOLS_S), SCHOOLS_Y)


def centered_schools(m: ModelHandle) -> None:
    """Centred version of :func:`schools`.

    References:
        [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007].

    """
    schools(m, noncentered=False)


def normal_model(m: ModelHandle) -> None:
    """Model a normal mean with known scale; :func:`ppl.condition` adds data.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2].

    """
    mu = m.sample("mu", Normal(PRIOR_MEAN, PRIOR_SD))
    m.observe("y", Normal(mu, NORMAL_SIGMA), None, shape=len(NORMAL_DATA))


def beta_binomial(m: ModelHandle) -> None:
    """Beta prior with a binomial likelihood.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2].

    """
    p = m.sample("p", Beta(BETA_A, BETA_B))
    m.observe("k", Binomial(TRIALS, p), SUCCESSES)


def exact_schools_posterior() -> Any:
    """Return the exact grid posterior of eight schools with a half-Cauchy prior.

    References:
        [EQ-HIER-002], [REF-GELMAN-ETAL-2013, Chapter 5], [REF-GELMAN-2006].

    """
    return hierarchical_normal_posterior(
        SCHOOLS_Y,
        SCHOOLS_S,
        mu_prior=MU_PRIOR,
        log_tau_prior=half_cauchy_log_prior(CAUCHY_SCALE),
    )


def tau_figure(posterior: ppl.PosteriorSamples) -> Any:
    """Draw the sampled ``tau`` over its exact grid posterior.

    Args:
        posterior: Non-centred NUTS draws of eight schools.

    Returns:
        Populated Matplotlib axes.

    References:
        [EQ-HIER-002], [REF-GELMAN-2006], [REF-MATPLOTLIB].

    """
    draws = [float(value) for value in posterior["tau"]]  # type: ignore[arg-type]
    return plot_tau_posterior(exact_schools_posterior(), draws=draws)


def write_figures(posterior: ppl.PosteriorSamples) -> None:
    """Write the ``tau`` figure when Matplotlib is installed.

    Args:
        posterior: Non-centred NUTS draws of eight schools.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if not matplotlib_available():
        return
    import matplotlib.pyplot as plt

    axes = tau_figure(posterior)
    save_figure(axes.figure, OUTPUT / "ppl-eight-schools-tau.png")
    plt.close(axes.figure)


def experiment_eight_schools() -> ppl.PosteriorSamples:
    """Compare the PPL posterior of eight schools with the exact grid.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-HOFFMAN-GELMAN-2014].

    """
    print("1. Eight schools, half-Cauchy(5) prior on tau, normal(0, 5) on mu")
    exact = exact_schools_posterior()
    program = ppl.trace(schools)
    print(
        f"   unconstrained coordinates: {', '.join(program.unconstrained_names[:3])}"
        f", ... (d = {program.dimension})"
    )
    print("   running non-centred NUTS: 2 chains x 600 iterations ...", flush=True)
    start = time.perf_counter()
    result = ppl.sample(schools, draws=300, warmup=300, chains=2, seed=SEED)
    elapsed = time.perf_counter() - start
    summaries = {s.name: s for s in result.summary()}
    mu_mean, mu_sd = exact.mu_moments()
    rows = (
        ("mu", mu_mean, mu_sd),
        ("tau", exact.tau_mean(), exact.tau_sd()),
    )
    print("   quantity   exact mean  NUTS mean   MCSE  exact sd   NUTS sd   ESS")
    for name, mean, sd in rows:
        s = summaries[name]
        print(
            f"   {name:<9} {mean:11.3f} {s.mean:10.3f} {s.mcse_mean:6.3f}"
            f" {sd:9.3f} {s.standard_deviation:9.3f} {s.effective_sample_size:5.0f}"
        )
    rhat = max(summaries[name].rank_normalized_rhat for name, _, _ in rows)
    theta_means, _ = exact.theta_moments()
    first = summaries["theta[0]"]
    print(
        f"   theta[0]   exact {theta_means[0]:.3f}  NUTS {first.mean:.3f}"
        f" (MCSE {first.mcse_mean:.3f})"
    )
    hamiltonian = result.hamiltonian
    assert hamiltonian is not None
    print(
        f"   non-centred: {hamiltonian.divergences} divergences, max R-hat "
        f"{rhat:.3f}, {elapsed:.1f} s (finite-difference gradients)"
    )
    print("   running centred NUTS: 2 chains x 300 iterations ...", flush=True)
    centered = ppl.sample(
        centered_schools,
        draws=150,
        warmup=150,
        chains=2,
        seed=SEED,
        max_tree_depth=6,
    )
    centered_h = centered.hamiltonian
    assert centered_h is not None
    tau_c = {s.name: s for s in centered.summary()}["tau"]
    hits = sum(item.max_tree_depth_hits for item in centered_h.statistics)
    print(
        f"   centred (tree depth <= 6): {centered_h.divergences} divergences, "
        f"{hits} of 300 draws hit the depth limit, tau ESS "
        f"{tau_c.effective_sample_size:.0f}, mean {tau_c.mean:.3f}"
    )
    return result


def experiment_normal() -> None:
    """Run every engine on one conjugate model and compare with the closed form.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2].

    """
    print("\n2. Normal-normal model (data supplied by condition)")
    model = ppl.condition(normal_model, {"y": NORMAL_DATA})
    precision = 1.0 / PRIOR_SD**2 + len(NORMAL_DATA) / NORMAL_SIGMA**2
    mean = (
        PRIOR_MEAN / PRIOR_SD**2 + math.fsum(NORMAL_DATA) / NORMAL_SIGMA**2
    ) / precision
    sd = precision**-0.5
    print(f"   closed form       mean {mean:.4f}  sd {sd:.4f}")
    print("   running Laplace, NUTS, HMC, Metropolis, and ADVI ...", flush=True)
    fits = (
        ("Laplace", ppl.laplace(model, draws=2000, seed=1)),
        ("NUTS", ppl.sample(model, draws=1000, warmup=500, chains=2, seed=2)),
        (
            "HMC",
            ppl.sample(model, method="hmc", draws=1000, warmup=500, chains=2, seed=3),
        ),
        (
            "Metropolis",
            ppl.sample(
                model, method="metropolis", draws=4000, warmup=1000, chains=2, seed=4
            ),
        ),
        ("ADVI", ppl.fit_advi(model, draws=4000, seed=5)),
    )
    for label, fit in fits:
        s = fit.summary()[0]
        print(
            f"   {label:<17} mean {s.mean:.4f}  sd {s.standard_deviation:.4f}"
            f"  (MCSE {s.mcse_mean:.4f})"
        )
    laplace_fit = fits[0][1]
    assert laplace_fit.point is not None
    print(f"   Laplace mode {laplace_fit.point['mu']:.6f} (exact {mean:.6f})")


def experiment_beta_binomial() -> None:
    """Check the logit Jacobian and compare NUTS with the exact Beta posterior.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2], [REF-CARPENTER-ETAL-2017].

    """
    print("\n3. Beta(2, 3) prior, 13 successes in 20 trials")
    program = ppl.trace(beta_binomial)
    step = 0.01
    values = [math.exp(program.log_density((-30.0 + step * i,))) for i in range(6001)]
    integral = step * (math.fsum(values) - 0.5 * (values[0] + values[-1]))
    posterior = Beta(BETA_A + SUCCESSES, BETA_B + TRIALS - SUCCESSES)
    evidence = math.comb(TRIALS, SUCCESSES) * math.exp(
        math.lgamma(posterior.alpha)
        + math.lgamma(posterior.beta)
        - math.lgamma(posterior.alpha + posterior.beta)
        - math.lgamma(BETA_A)
        - math.lgamma(BETA_B)
        + math.lgamma(BETA_A + BETA_B)
    )
    print(f"   integral of exp(log p(u)) over u: {integral:.10f}")
    print(f"   exact evidence p(k):              {evidence:.10f}")
    print("   running NUTS: 2 chains x 1300 iterations ...", flush=True)
    result = ppl.sample(beta_binomial, draws=1000, warmup=300, chains=2, seed=7)
    s = result.summary()[0]
    print(
        f"   exact Beta(15, 10) mean {posterior.mean:.4f}  sd "
        f"{math.sqrt(posterior.variance):.4f}"
    )
    print(
        f"   NUTS               mean {s.mean:.4f}  sd {s.standard_deviation:.4f}"
        f"  (MCSE {s.mcse_mean:.4f})"
    )
    draws = [float(p) for p in result["p"]]  # type: ignore[arg-type]
    print(f"   draws in ({min(draws):.4f}, {max(draws):.4f}); all inside (0, 1)")


def experiment_predictive(posterior: ppl.PosteriorSamples) -> None:
    """Simulate from the prior and posterior predictive distributions.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 6], [REF-GELMAN-MENG-STERN-1996].

    """
    print("\n4. Predictive simulation for eight schools")
    print("   simulating 2000 prior and 600 posterior predictive datasets ...")
    prior = ppl.prior_predictive(schools, 2000, seed=11)
    spread = [max(y) - min(y) for y in prior["y"]]  # type: ignore[arg-type]
    print(
        f"   prior predictive range of y: median {statistics.median(spread):.1f},"
        f" 90% quantile {statistics.quantiles(spread, n=10)[-1]:.1f}"
        f" (observed {max(SCHOOLS_Y) - min(SCHOOLS_Y):.1f})"
    )
    replicated = ppl.posterior_predictive(schools, posterior, seed=12)
    print(
        f"   posterior predictive draws: {replicated.draws} replicates of"
        f" length {len(replicated['y'][0])}"
    )  # type: ignore[arg-type]
    check = ppl.predictive_pvalues(
        schools,
        posterior,
        "y",
        statistics={"max": max, "min": min, "sd": statistics.stdev},
        seed=13,
    )
    for item in check.checks:
        print(
            f"   p-value T = {item.name:<4} {item.p_value:.3f} (MCSE {item.mcse:.3f})"
        )


def main() -> None:
    """Run the four experiments.

    References:
        Project laboratory policy. No equation is evaluated.

    """
    start = time.perf_counter()
    posterior = experiment_eight_schools()
    experiment_normal()
    experiment_beta_binomial()
    experiment_predictive(posterior)
    write_figures(posterior)
    print(
        f"\nDone in {time.perf_counter() - start:.1f} s: compare each exact value "
        "with its PPL estimate and MCSE above. Next: read "
        "docs/foundations/probabilistic-programming.md."
    )


if __name__ == "__main__":
    main()
