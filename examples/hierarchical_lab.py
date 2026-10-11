"""Hierarchical-model laboratory: EM, mixtures, eight schools, and PPCs.

Five seeded experiments, printed as numbers:

1. Gaussian-mixture selection: BIC over K = 1..5 on three-component data.
2. Censored normal data: EM against two naive fixes for detection limits.
3. Eight schools: exact grid posterior against NUTS (non-centered).
4. Posterior predictive checks on the eight-schools NUTS draws.
5. Beta-binomial empirical Bayes shrinkage.

With Matplotlib installed the lab also writes seven figures to
``examples/generated/``: a one-dimensional mixture fit and its BIC curve, the
eight-schools shrinkage plot, the posterior of tau with the NUTS draws, two
predictive checks, and the PIT histogram.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from probstat.inference import (
    GaussianMixture,
    HierarchicalNormalPosterior,
    HierarchicalNormalSamples,
    PosteriorPredictivePValues,
    fit_beta_binomial,
    fit_censored_normal,
    hierarchical_normal_posterior,
    pit_values,
    posterior_predictive_pvalues,
    sample_hierarchical_normal,
    select_gaussian_mixture,
)

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
SCHOOLS = ("A", "B", "C", "D", "E", "F", "G", "H")
OUTPUT = Path(__file__).parent / "generated"


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def save(axes: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    figure = getattr(axes, "figure", axes)
    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def mixture_figures() -> None:
    """Fit and select a one-dimensional mixture and draw both figures.

    References:
        [EQ-GMM-001], [EQ-GMM-004], [REF-MCLACHLAN-PEEL-2000],
        [REF-SCHWARZ-1978].

    """
    from probstat.visualization.hierarchical_plots import (
        plot_mixture_fit,
        plot_mixture_selection,
    )

    truth = GaussianMixture(
        (0.3, 0.5, 0.2),
        ((-3.0,), (0.5,), (4.0,)),
        (((0.64,),), ((1.0,),), ((0.36,),)),
    )
    data = [point[0] for point in truth.sample(300, seed=21)]
    selection = select_gaussian_mixture(
        data, (1, 2, 3, 4, 5), seed=5, tolerance=1e-8, max_iterations=300
    )
    names = (
        save(plot_mixture_fit(selection.best, data), "hierarchical-mixture-fit.png"),
        save(plot_mixture_selection(selection), "hierarchical-mixture-bic.png"),
    )
    print(
        f"   1-D mixture (n = 300, true K = 3): BIC selects K = "
        f"{selection.best_components}; figures {', '.join(names)}"
    )


def schools_figures(
    grid: HierarchicalNormalPosterior,
    samples: HierarchicalNormalSamples,
    check: PosteriorPredictivePValues,
    pit: tuple[float, ...],
) -> None:
    """Draw the shrinkage, tau, predictive-check, and PIT figures.

    References:
        [EQ-HIER-001], [EQ-HIER-002], [EQ-PPC-001], [EQ-PPC-002],
        [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    from probstat.visualization.hierarchical_plots import (
        plot_pit_histogram,
        plot_predictive_check,
        plot_shrinkage,
        plot_tau_posterior,
    )

    tau_draws = [v for chain in samples.draws.chain_columns("tau") for v in chain]
    names = (
        save(plot_shrinkage(grid, names=SCHOOLS), "hierarchical-shrinkage.png"),
        save(plot_tau_posterior(grid, draws=tau_draws), "hierarchical-tau.png"),
        save(plot_predictive_check(check["max"]), "hierarchical-ppc-max.png"),
        save(plot_predictive_check(check["chi2"]), "hierarchical-ppc-chi2.png"),
        save(plot_pit_histogram(pit, bins=4), "hierarchical-pit.png"),
    )
    print(f"   figures: {', '.join(names)}")


def mixture_selection() -> int:
    """Select the number of mixture components by BIC.

    References:
        [EQ-GMM-001], [EQ-GMM-002], [EQ-GMM-004], [REF-MCLACHLAN-PEEL-2000],
        [REF-SCHWARZ-1978].

    """
    truth = GaussianMixture(
        (0.3, 0.5, 0.2),
        ((-4.0, 0.0), (0.0, 3.0), (4.0, -1.0)),
        (
            ((1.0, 0.3), (0.3, 0.5)),
            ((0.6, 0.0), (0.0, 1.5)),
            ((0.8, -0.4), (-0.4, 1.0)),
        ),
    )
    print("1. Gaussian mixture, n = 400, true K = 3 (seed 11); fitting K = 1..5 ...")
    data = truth.sample(400, seed=11)
    selection = select_gaussian_mixture(
        data, (1, 2, 3, 4, 5), seed=5, tolerance=1e-7, max_iterations=150
    )
    print(f"{'K':>3}{'log L':>11}{'params':>8}{'BIC':>10}{'iters':>7}{'monotone':>10}")
    for fit, score in zip(selection.fits, selection.scores, strict=True):
        print(
            f"{fit.mixture.components:>3}{fit.log_likelihood:>11.2f}"
            f"{fit.parameter_count:>8}{score:>10.2f}{fit.iterations:>7}"
            f"{str(fit.monotone):>10}"
        )
    best = selection.best.mixture
    print(f"   BIC selects K = {selection.best_components}")
    for k in range(best.components):
        mean = ", ".join(f"{v:6.3f}" for v in best.means[k])
        print(f"   component {k}: weight {best.weights[k]:.3f}, mean ({mean})")
    if has_matplotlib():
        mixture_figures()
    print()
    return selection.best_components


def censored_normal() -> float:
    """Compare EM with naive treatments of detection limits.

    References:
        [EQ-EM-CENS-001], [EQ-EM-CENS-002], [REF-DEMPSTER-LAIRD-RUBIN-1977].

    """
    rng = random.Random(5)
    raw = [rng.gauss(10.0, 2.0) for _ in range(300)]
    low, high = 8.0, 13.0
    intervals = [
        (-math.inf, low) if v < low else (high, math.inf) if v > high else (v, v)
        for v in raw
    ]
    fit = fit_censored_normal(intervals)
    exact = [a for a, b in intervals if a == b]
    substituted = [min(max(v, low), high) for v in raw]
    print("2. Normal(10, 2) data, censored below 8 and above 13 (seed 5)")
    print(f"   censored values: {fit.censored_count} of {len(raw)}")
    print(f"{'method':<28}{'mean':>8}{'sd':>8}")
    print(
        f"{'drop censored values':<28}{statistics.fmean(exact):>8.3f}"
        f"{statistics.pstdev(exact):>8.3f}"
    )
    print(
        f"{'substitute the limit':<28}{statistics.fmean(substituted):>8.3f}"
        f"{statistics.pstdev(substituted):>8.3f}"
    )
    print(
        f"{'EM (censored likelihood)':<28}{fit.mean:>8.3f}"
        f"{fit.standard_deviation:>8.3f}"
    )
    print(
        f"   EM iterations {fit.em.iterations}, monotone {fit.em.monotone}, "
        f"log L {fit.log_likelihood:.4f}"
    )
    print()
    return fit.standard_deviation


def eight_schools() -> float:
    """Compare the exact grid posterior with NUTS and run predictive checks.

    References:
        [EQ-HIER-001], [EQ-HIER-002], [EQ-HIER-003], [EQ-HIER-004],
        [EQ-PPC-001], [EQ-PPC-002], [REF-GELMAN-ETAL-2013, Chapter 5],
        [REF-BETANCOURT-GIROLAMI-2013], [REF-GELMAN-MENG-STERN-1996].

    """
    print("3. Eight schools, uniform priors on mu and tau")
    print("   running NUTS: 4 chains x (500 warm-up + 1000 draws), about 8 s ...")
    grid = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S)
    started = time.perf_counter()
    samples = sample_hierarchical_normal(
        SCHOOLS_Y, SCHOOLS_S, draws=1000, warmup=500, chains=4, seed=2026
    )
    elapsed = time.perf_counter() - started
    summaries = {s.name: s for s in samples.draws.summary()}
    mu_mean, mu_sd = grid.mu_moments()
    theta_means, theta_sds = grid.theta_moments()
    shrink = grid.expected_shrinkage()
    print(
        f"   grid: {len(grid.tau_grid)} points up to tau = {grid.tau_grid[-1]:.0f}; "
        f"NUTS: 4 x 1000 draws in {elapsed:.1f} s, "
        f"{samples.hamiltonian.divergences} divergences"
    )
    print(
        f"{'quantity':<10}{'grid mean':>10}{'grid sd':>9}{'NUTS mean':>11}"
        f"{'NUTS sd':>9}{'MCSE':>7}{'z':>7}"
    )
    rows = [("mu", mu_mean, mu_sd, None), ("tau", grid.tau_mean(), grid.tau_sd(), None)]
    rows += [
        (f"theta_{j + 1}", theta_means[j], theta_sds[j], shrink[j]) for j in range(8)
    ]
    largest_z = 0.0
    for name, mean, sd, _ in rows:
        summary = summaries[name]
        z = (summary.mean - mean) / summary.mcse_mean
        largest_z = max(largest_z, abs(z))
        print(
            f"{name:<10}{mean:>10.3f}{sd:>9.3f}{summary.mean:>11.3f}"
            f"{summary.standard_deviation:>9.3f}{summary.mcse_mean:>7.3f}"
            f"{z:>7.2f}"
        )
    print("   E[B_j | y]:", ", ".join(f"{factor:.3f}" for factor in shrink))
    quantiles = ", ".join(
        f"{p:.2f}: {grid.tau_quantile(p):.2f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95)
    )
    print(f"   grid tau quantiles {quantiles}")
    print()

    def simulate(theta: tuple[float, ...], rng: random.Random) -> list[float]:
        """Replicate the eight estimates given (mu, tau, theta_1..theta_8).

        References:
            [EQ-PPC-001]. No new equation is introduced.

        """
        return [rng.gauss(t, s) for t, s in zip(theta[2:], SCHOOLS_S, strict=True)]

    check = posterior_predictive_pvalues(
        samples.draws,
        simulate,
        SCHOOLS_Y,
        statistics={
            "max": max,
            "min": min,
            "mean": statistics.fmean,
            "sd": statistics.stdev,
        },
        discrepancies={
            "chi2": lambda y, t: math.fsum(
                ((v - m) / s) ** 2 for v, m, s in zip(y, t[2:], SCHOOLS_S, strict=True)
            )
        },
        keep_replicates=True,
        seed=4,
    )
    print(f"4. Posterior predictive checks on {check.draws} NUTS draws (seed 4)")
    print(f"{'statistic':<10}{'p (>=)':>8}{'MCSE':>8}{'two-sided':>11}")
    for item in check.checks:
        print(
            f"{item.name:<10}{item.p_value:>8.3f}{item.mcse:>8.3f}"
            f"{item.two_sided_p_value:>11.3f}"
        )
    pit = pit_values(SCHOOLS_Y, check.replicates)
    print("   PIT values:", ", ".join(f"{v:.2f}" for v in pit))
    if has_matplotlib():
        schools_figures(grid, samples, check, pit)
    print()
    return largest_z


def beta_binomial() -> None:
    """Fit a beta-binomial hierarchy by empirical Bayes.

    References:
        [EQ-HIER-BB-001], [REF-MORRIS-1983], [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    rng = random.Random(17)
    trials = [rng.randint(20, 80) for _ in range(60)]
    successes = []
    for n in trials:
        p = rng.betavariate(4.0, 12.0)
        successes.append(sum(rng.random() < p for _ in range(n)))
    fit = fit_beta_binomial(successes, trials)
    print("5. Beta-binomial empirical Bayes, 60 groups from Beta(4, 12) (seed 17)")
    print(
        f"   alpha = {fit.alpha:.3f}, beta = {fit.beta:.3f}, prior mean "
        f"{fit.prior_mean:.4f} (true 0.25), kappa = {fit.alpha + fit.beta:.2f}"
    )
    for j in range(3):
        print(
            f"   group {j}: {successes[j]}/{trials[j]} = "
            f"{successes[j] / trials[j]:.3f} -> {fit.posterior_means[j]:.3f} "
            f"(shrinkage {fit.shrinkage[j]:.3f})"
        )


def main() -> None:
    """Run the five experiments and print a one-line conclusion.

    References:
        Project laboratory policy. No equation is evaluated.

    """
    started = time.perf_counter()
    chosen = mixture_selection()
    sd = censored_normal()
    largest_z = eight_schools()
    beta_binomial()
    print(
        f"\nDone in {time.perf_counter() - started:.1f} s: BIC chose K = {chosen}, "
        f"censored-EM sd = {sd:.3f} (true 2), grid vs NUTS max |z| = "
        f"{largest_z:.2f}. Next: docs/foundations/hierarchical-models.md"
    )


if __name__ == "__main__":
    main()
