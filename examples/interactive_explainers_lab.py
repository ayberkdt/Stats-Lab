"""Interactive explanations laboratory: three offline HTML explainers.

Writes the central-limit-theorem, Bayesian-updating, and Monte Carlo error
explainers of ``probstat.visualization.interactive`` (ADR-0085) to
``examples/generated/`` in English and in Turkish, prints the numbers each
page opens with, and saves one static themed figure (the convergence trace
of the first Monte Carlo seed) so the run also leaves a figure with an
embedded text description. Run with
``PYTHONPATH=src python examples/interactive_explainers_lab.py`` and open
the ``*-explainer.html`` files in any browser; they need no network.

Definitions:
    explainer: a single HTML file with inline CSS, inline SVG charts, and a
        short script that selects among states precomputed in Python.
    ``K_n``: exact Kolmogorov distance ``sup_z |F_n(z) - Phi(z)|`` of the
        standardized mean of ``n`` Exponential(1) summands to the normal law.
    Berry-Esseen bound: ``C rho / (sigma^3 sqrt(n))`` with ``C = 0.4748``.
    posterior: ``Beta(a + x, b + n - x)`` after ``x`` successes in ``n``
        Bernoulli trials under a ``Beta(a, b)`` prior.
    credible and Wilson intervals: equal-tailed posterior quantiles and the
        frequentist score interval for the same proportion.
    Monte Carlo trace: running mean of ``4 I(U1^2 + U2^2 <= 1)`` with its
        standard error and normal-approximation interval at checkpoints.

Assumptions:
    - The summands of the CLT page are i.i.d. with a finite third moment,
      so the Berry-Esseen bound applies; ``F_n`` is a gamma law, so ``K_n``
      is exact, not simulated.
    - The Bernoulli trials and the Monte Carlo points are independent draws
      from child streams of the root ``SEED``; the Wilson and Monte Carlo
      intervals are large-sample approximations.
    - The pages select among precomputed states only; the script evaluates
      no probability.

Equations:
    EQ-LIM-001: Berry-Esseen bound with Shevtsova's constant.
    EQ-LIM-002: exact law of the standardized exponential mean.
    EQ-LIM-003: Kolmogorov distance to the normal law.
    EQ-LIM-010: third absolute central moment of the summands.
    EQ-REL-DATA-001: Beta posterior from binomial data.
    EQ-DIST-BETA-001: Beta density of the prior, likelihood, and posterior.
    EQ-DIST-INV-001: Beta quantiles of the credible interval.
    EQ-VIZ-LIK-001: binomial likelihood scaled to unit area.
    EQ-INF-CI-WILSON-001: Wilson score interval.
    EQ-MC-MEAN-001: Monte Carlo sample mean.
    EQ-MC-SE-001: Monte Carlo standard error.
    EQ-MC-ROOTN-001: the ``1 / sqrt(N)`` rate of the band.
    EQ-RNG-SPLITMIX-001: child seeds of the Bernoulli and Monte Carlo runs.

Diagnostics:
    - CLT page: the exact distance stays below the Berry-Esseen bound at
      every ``n`` and the printed ratio shows how conservative the bound is.
    - Bayesian page: the credible and Wilson intervals are printed side by
      side after all trials, with whether each contains the true ``p``.
      With ``SEED = 2026`` the 40 trials give 20 successes, a tail draw
      (``P(X >= 20) = 0.0063`` for ``p = 0.3``), so both 95% intervals
      exclude ``p``: a 95% procedure misses in about one sample in twenty,
      and the page shows such a sample.
    - Monte Carlo page: each seed's interval is checked against ``pi`` and
      the spread of the seeds is printed; the half-width roughly halves for
      every fourfold increase in ``N``.
    - Every page passes the UX law audit of ADR-0085 and embeds its raw
      numbers as JSON, which the regression test compares with fresh
      library calls.

Cost and error:
    - CLT: 23 sample sizes, each with an exact distance and 161 exact
      distribution-function values; no simulation error.
    - Bayesian: 40 Bernoulli draws; posterior and Wilson intervals are
      exact functions of the counts.
    - Monte Carlo: 4 seeds of 50,000 points (400,000 uniforms); the 95%
      half-width at ``N = 50,000`` is about 0.014.
    - The six pages take a few seconds to compute and occupy about
      1.1 MB in total.

References:
    REF-SHEVTSOVA-2011: the Berry-Esseen constant.
    REF-CASELLA-BERGER-2002: limit theorems.
    REF-GELMAN-ETAL-2013: Beta-Binomial updating.
    REF-WILSON-1927: the score interval.
    REF-ROBERT-CASELLA-2004: Monte Carlo error.
    REF-W3C-WAI-ARIA-12-2023: live regions and toggle buttons.
    REF-W3C-WCAG21-2018: text alternatives and contrast.
    REF-UNICODE-CLDR: Turkish number symbols and plural categories.

"""

from __future__ import annotations

import math
from pathlib import Path

from probstat.visualization import plot_monte_carlo_convergence, save_figure
from probstat.visualization.interactive import (
    bayesian_updating_explainer,
    clt_explainer,
    monte_carlo_error_explainer,
    pi_estimate_traces,
)

# Root seed of the Bernoulli trials and of the Monte Carlo runs.
SEED = 2026


def main() -> None:
    """Write the explainers, print their opening numbers, and save one figure.

    References:
        [EQ-LIM-003], [EQ-REL-DATA-001], [EQ-INF-CI-WILSON-001],
        [EQ-MC-ROOTN-001], [REF-W3C-WAI-ARIA-12-2023], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    output = Path(__file__).parent / "generated"
    for locale in ("en", "tr"):
        suffix = ".html" if locale == "en" else f".{locale}.html"
        pages = (
            clt_explainer(locale=locale),
            bayesian_updating_explainer(seed=SEED, locale=locale),
            monte_carlo_error_explainer(seed=SEED, locale=locale),
        )
        for page in pages:
            path = page.write(output / f"{page.name}{suffix}")
            print(f"wrote {path.name} ({path.stat().st_size:,} bytes)")
    clt, bayes, monte_carlo = pages

    sizes = clt.data["sizes"]
    for n in (1, 30, 1000):
        index = sizes.index(n)
        distance = clt.data["kolmogorov_normal"][index]
        bound = clt.data["berry_esseen"][index]
        print(
            f"CLT n={n}: exact distance {distance:.5f}, Berry-Esseen bound "
            f"{bound:.5f} (ratio {bound / distance:.1f})"
        )
    final = bayes.data["states"][-1]
    truth = bayes.data["probability"]
    for name in ("credible_interval", "wilson_interval"):
        lower, upper = final[name]
        verdict = "contains" if lower <= truth <= upper else "excludes"
        print(f"Bayes {name.replace('_', ' ')} {verdict} the true p = {truth:g}")
    print(
        f"Bayes after {final['trials']} trials ({final['successes']} successes): "
        f"posterior Beta({final['posterior'][0]:g}, {final['posterior'][1]:g}), "
        f"credible [{final['credible_interval'][0]:.4f}, "
        f"{final['credible_interval'][1]:.4f}], Wilson "
        f"[{final['wilson_interval'][0]:.4f}, {final['wilson_interval'][1]:.4f}]"
    )
    for run in monte_carlo.data["runs"]:
        lower, upper = run["lower"][-1], run["upper"][-1]
        covers = "covers" if lower <= math.pi <= upper else "misses"
        print(
            f"MC seed {run['run']}: N={monte_carlo.data['checkpoints'][-1]:,} "
            f"estimate {run['estimate'][-1]:.5f}, 95% interval [{lower:.5f}, "
            f"{upper:.5f}] {covers} pi"
        )

    traces = pi_estimate_traces(seed=SEED)
    axes = plot_monte_carlo_convergence(
        traces[0], reference=math.pi, reference_label="exact pi"
    )
    axes.figure.tight_layout()
    save_figure(axes.figure, output / "interactive-monte-carlo-seed-1.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
