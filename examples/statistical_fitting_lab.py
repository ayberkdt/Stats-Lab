"""Reproducible UQ-1 laboratory for lognormal fitting and diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

from probstat.distributions import LogNormal
from probstat.inference import (
    anderson_darling_test,
    fit_lognormal_mle,
    kolmogorov_smirnov_test,
    normal_mean_confidence_interval,
)
from probstat.visualization import plot_fitted_cdf


def main() -> None:
    """Fit seeded positive data, report diagnostics, and render a CDF plot.

    This lab deliberately reports that the ordinary KS p-value assumes a fully
    specified null and is not calibrated after fitting on the same data. The
    value is shown as an illustrative diagnostic, not a valid fitted-null test.

    References:
        [EQ-INF-MLE-LN-001], [EQ-INF-GOF-KS-001], [EQ-INF-GOF-AD-001],
        [EQ-INF-CI-NMEAN-001], [REF-NIST-GOF], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    generating_law = LogNormal(mu=0.4, sigma=0.35)
    observations = generating_law.sample(250, seed=2026)
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

    axes = plot_fitted_cdf(observations, fitted)
    axes.figure.tight_layout()
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    axes.figure.savefig(output / "lognormal-fit-cdf.png", dpi=160)
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
