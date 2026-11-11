"""Reproducible UQ-1 laboratory for lognormal fitting and diagnostics.

Run with ``PYTHONPATH=src python examples/statistical_fitting_lab.py``. A
seeded sample of 250 values from ``LogNormal(mu = 0.4, sigma = 0.35)`` is
fitted by maximum likelihood; the fit is checked with the Kolmogorov-Smirnov
and Anderson-Darling statistics, first with the textbook p-value (which
assumes the law was fixed in advance) and then with p-values calibrated by a
parametric bootstrap that refits the law to every replicate. The fitted CDF
is drawn against the empirical CDF in ``examples/generated/``.

Definitions:
    ``X ~ LogNormal(mu, sigma)``: ``log X`` is normal with mean ``mu`` and
        standard deviation ``sigma``.
    ``F_n``: empirical CDF of the sample; ``F_hat``: fitted lognormal CDF.
    ``D``: Kolmogorov-Smirnov distance ``sup |F_n - F_hat|``.
    ``A^2``: Anderson-Darling statistic, a tail-weighted EDF distance.
    parametric-bootstrap p-value: share of replicate statistics at least as
        large as the observed one, each replicate drawn from ``F_hat`` and
        tested against its own refitted law.

Assumptions:
    - The observations are independent draws from one positive law.
    - The ordinary KS p-value is valid only for a fully specified null; with
      ``mu`` and ``sigma`` estimated from the same data it is conservative,
      so it is printed as an illustration only.
    - The bootstrap p-values assume the fitted law is close enough to the
      truth for its statistic distribution to stand in for the null law.
    - The log-mean interval is the large-sample normal ``z`` interval.

Equations:
    EQ-DIST-LN-003: lognormal quantile, used to draw the seeded sample.
    EQ-INF-MLE-LN-001: closed-form lognormal maximum-likelihood estimates.
    EQ-INF-GOF-KS-001: one-sample Kolmogorov-Smirnov statistic.
    EQ-INF-GOF-AD-001: Anderson-Darling statistic.
    EQ-INF-GOF-PB-001: parametric-bootstrap p-value with refitted parameters.
    EQ-INF-CI-NMEAN-001: normal interval for the mean of ``log X``.
    EQ-STAT-005: empirical CDF drawn in the figure.

Diagnostics:
    - Fitted ``(mu, sigma)`` against the generating ``(0.4, 0.35)``.
    - ``D`` and ``A^2`` with bootstrap-calibrated p-values; a small p-value
      (for example below 0.05) is evidence against the lognormal family.
      The gap between the ordinary and the calibrated KS p-value shows how
      much fitting the parameters shrinks the statistic.
    - The fitted-CDF figure shows where the two CDFs separate most.

Cost and error:
    - 250 observations; the MLE standard errors are about
      ``sigma / sqrt(250) = 0.022`` for ``mu`` and ``sigma / sqrt(500) = 0.016``
      for ``sigma``.
    - 199 bootstrap replicates, each a draw of 250 values, a refit, and two
      EDF statistics; the smallest attainable p-value is ``1/200 = 0.005``
      and a p-value ``p`` carries a Monte Carlo standard error of about
      ``sqrt(p (1 - p) / 199)``.
    - The run takes a few seconds; every number is fixed by ``SEED``.

References:
    REF-NIST-GOF: Kolmogorov-Smirnov and Anderson-Darling tests.
    REF-STEPHENS-1974: EDF statistics with estimated parameters.
    REF-LILLIEFORS-1967: the KS test when parameters are estimated.
    REF-DAVISON-HINKLEY-1997: parametric bootstrap tests.
    REF-CASELLA-BERGER-2002: maximum likelihood and the normal interval.
    REF-MATPLOTLIB: rendering.

"""

from __future__ import annotations

import math
from pathlib import Path

from probstat.distributions import LogNormal
from probstat.inference import (
    LOGNORMAL_FAMILY,
    anderson_darling_test,
    bootstrap_goodness_of_fit,
    fit_lognormal_mle,
    kolmogorov_smirnov_test,
    normal_mean_confidence_interval,
)
from probstat.visualization import plot_fitted_cdf, save_figure

# Seed of the sample and, through a child stream, of the bootstrap replicates.
SEED = 2026
BOOTSTRAP_REPLICATES = 199


def main() -> None:
    """Fit seeded positive data, report diagnostics, and render a CDF plot.

    This lab deliberately reports that the ordinary KS p-value assumes a fully
    specified null and is not calibrated after fitting on the same data. The
    value is shown as an illustrative diagnostic, then replaced by
    parametric-bootstrap p-values that refit the law to every replicate.

    References:
        [EQ-INF-MLE-LN-001], [EQ-INF-GOF-KS-001], [EQ-INF-GOF-AD-001],
        [EQ-INF-GOF-PB-001], [EQ-INF-CI-NMEAN-001], [REF-NIST-GOF],
        [REF-STEPHENS-1974], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    generating_law = LogNormal(mu=0.4, sigma=0.35)
    observations = generating_law.sample(250, seed=SEED)
    fitted = fit_lognormal_mle(observations)
    ks_result = kolmogorov_smirnov_test(observations, fitted)
    ad_result = anderson_darling_test(observations, fitted)
    log_mean_interval = normal_mean_confidence_interval(
        tuple(math.log(value) for value in observations)
    )

    print(f"fitted log-mu: {fitted.mu:.6f}")
    print(f"fitted log-sigma: {fitted.sigma:.6f}")
    print(f"KS D: {ks_result.statistic:.6f}")
    print(
        "KS illustrative p-value (not fitted-null calibrated): "
        f"{ks_result.p_value:.6f}"
    )
    print(f"Anderson-Darling A^2: {ad_result.statistic:.6f}")
    print(
        "Approximate 95% interval for log-mean: "
        f"[{log_mean_interval.lower:.6f}, {log_mean_interval.upper:.6f}]"
    )

    calibrated = bootstrap_goodness_of_fit(
        observations,
        LOGNORMAL_FAMILY,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    print(
        f"Parametric bootstrap ({calibrated.refits} refits, "
        f"{calibrated.failed_refits} failed):"
    )
    print(
        f"  KS D={calibrated.ks_statistic:.6f}  "
        f"calibrated p={calibrated.ks_p_value:.4f}"
    )
    print(
        f"  AD A^2={calibrated.ad_statistic:.6f}  "
        f"calibrated p={calibrated.ad_p_value:.4f}"
    )

    axes = plot_fitted_cdf(observations, fitted)
    axes.figure.tight_layout()
    output = Path(__file__).parent / "generated"
    save_figure(axes.figure, output / "lognormal-fit-cdf.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
