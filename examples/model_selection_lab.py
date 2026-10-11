"""Model-selection laboratory: rank candidate families for positive data.

Seeded synthetic Weibull data are fitted by maximum likelihood with six
candidate families. The candidates are ranked by AIC with Akaike weights, and
each receives parametric-bootstrap Kolmogorov-Smirnov and Anderson-Darling
p-values that account for the parameters being estimated from the same data.
The winner's fitted CDF is drawn against the empirical CDF.
"""

from __future__ import annotations

import time
from pathlib import Path

from probstat.distributions import Weibull
from probstat.inference import select_distribution
from probstat.visualization import plot_fitted_cdf, save_figure

CANDIDATES = ("Weibull", "Gamma", "LogNormal", "Gumbel", "Normal", "Exponential")
SAMPLE_SIZE = 250
BOOTSTRAP_REPLICATES = 99
ALPHA = 0.05


def main() -> None:
    """Fit, rank, calibrate, print one table, and save the winner's CDF.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-INF-IC-001], [EQ-INF-IC-004],
        [EQ-INF-GOF-PB-001], [REF-BURNHAM-ANDERSON-2002, Chapter 2],
        [REF-STEPHENS-1974], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    truth = Weibull(shape=1.6, scale=40.0)
    data = truth.sample(SAMPLE_SIZE, seed=2026)
    print(
        f"Data: {SAMPLE_SIZE} draws from Weibull(shape=1.6, scale=40), seed 2026."
    )
    print(
        f"Fitting {len(CANDIDATES)} candidates; {BOOTSTRAP_REPLICATES} bootstrap "
        "refits each (takes a few seconds)..."
    )
    started = time.perf_counter()
    result = select_distribution(
        data, CANDIDATES, n_bootstrap=BOOTSTRAP_REPLICATES, alpha=ALPHA, seed=2026
    )
    elapsed = time.perf_counter() - started

    print()
    print(
        f"{'Rank':>4}  {'Family':<12}{'AIC':>9}{'dAIC':>8}{'Weight':>8}"
        f"{'KS p':>7}{'AD p':>7}"
    )
    for candidate in result.candidates:
        gof = candidate.goodness_of_fit
        ks_p = f"{gof.ks_p_value:.2f}" if gof is not None else "-"
        ad_p = f"{gof.ad_p_value:.2f}" if gof is not None else "-"
        print(
            f"{candidate.rank:>4}  {candidate.fit.family:<12}"
            f"{candidate.fit.aic:>9.1f}{candidate.delta_aic:>8.1f}"
            f"{candidate.akaike_weight:>8.3f}{ks_p:>7}{ad_p:>7}"
        )
    print(f"(p-values: parametric bootstrap with refits; {elapsed:.1f} s)")

    best = result.best
    print()
    print(f"Winner {best.fit.family} (true family: Weibull):")
    for name, value, error in zip(
        best.fit.parameter_names,
        best.fit.estimates,
        best.fit.standard_errors,
        strict=True,
    ):
        print(f"  {name:<6} = {value:8.3f}  (SE {error:.3f})")
    for message in result.warnings:
        print(f"WARNING: {message}")

    output = Path(__file__).parent / "generated" / "model-selection-cdf.png"
    axes = plot_fitted_cdf(data, best.fit.distribution)
    save_figure(axes.figure, output)
    plt.close(axes.figure)

    check = best.goodness_of_fit
    verdict = (
        "is not rejected"
        if check is not None and not check.rejected(ALPHA)
        else "is REJECTED"
    )
    print()
    print(
        f"Result: {best.fit.family} wins with Akaike weight "
        f"{best.akaike_weight:.3f} and {verdict} at alpha={ALPHA}. "
        f"Next: open {output.relative_to(Path(__file__).parent.parent)}"
    )


if __name__ == "__main__":
    main()
