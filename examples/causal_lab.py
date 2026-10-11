"""Causal inference laboratory: confounded data, four estimators, E-values.

Seeded observational data with a known average treatment effect of 2 are
analysed by the naive contrast, regression adjustment, stabilized IPW, and
AIPW; covariate balance is compared before and after weighting; and the
AIPW estimate is mapped to an approximate risk ratio whose E-value says how
strong unmeasured confounding would have to be to explain it away.
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

from probstat.causal import (
    bootstrap_effect,
    e_value,
    estimate_aipw,
    estimate_ipw,
    estimate_regression_adjustment,
    risk_ratio_from_standardized_difference,
    standardized_mean_differences,
)
from probstat.foundations import RandomStream
from probstat.visualization import save_figure
from probstat.visualization.causal_plots import (
    plot_covariate_balance,
    plot_e_value_contour,
    plot_effect_estimates,
)

TRUE_EFFECT = 2.0


def simulate(
    size: int, stream: RandomStream
) -> tuple[list[tuple[float, float, float]], list[int], list[float]]:
    """Draw confounded data with a known average treatment effect of two.

    ``X1, X2, X3 ~ N(0, 1)``; ``logit e = -0.3 + 0.8 X1 - 0.5 X2 + 0.3 X3``;
    ``Y = 1 + 2 T + 1.5 X1 + X2 + 0.4 X3 + 0.5 T X1 + N(0, 1)``, so
    ``E[Y(1) - Y(0)] = 2 + 0.5 E[X1] = 2``.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 13]. No equation is evaluated.

    """
    rng = stream.generator()
    rows, treatment, outcome = [], [], []
    for _ in range(size):
        x1, x2, x3 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        score = 1.0 / (1.0 + math.exp(-(-0.3 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3)))
        t = 1 if rng.random() < score else 0
        y = (
            1.0
            + TRUE_EFFECT * t
            + 1.5 * x1
            + x2
            + 0.4 * x3
            + 0.5 * t * x1
            + rng.gauss(0.0, 1.0)
        )
        rows.append((x1, x2, x3))
        treatment.append(t)
        outcome.append(y)
    return rows, treatment, outcome


def main() -> None:
    """Estimate the effect four ways, check balance, and write three figures.

    References:
        [REF-HERNAN-ROBINS-2020, Chapters 12 and 13],
        [REF-VANDERWEELE-DING-2017], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    output = Path(__file__).parent / "generated"
    root = RandomStream(2026)
    rows, treatment, outcome = simulate(1200, root.child("data"))
    treated = [y for y, t in zip(outcome, treatment, strict=True) if t]
    control = [y for y, t in zip(outcome, treatment, strict=True) if not t]
    naive = statistics.fmean(treated) - statistics.fmean(control)
    naive_se = math.sqrt(
        statistics.variance(treated) / len(treated)
        + statistics.variance(control) / len(control)
    )
    regression = estimate_regression_adjustment(rows, treatment, outcome)
    regression_ci = bootstrap_effect(
        rows,
        treatment,
        outcome,
        method="regression",
        replicates=100,
        seed=root.child("bootstrap"),
    )
    ipw = estimate_ipw(rows, treatment, outcome, stabilized=True)
    aipw = estimate_aipw(rows, treatment, outcome)
    print(f"{len(treated)} treated of {len(outcome)}; true ATE {TRUE_EFFECT}")
    print(f"  naive          {naive:.4f} (SE {naive_se:.4f})")
    print(
        f"  regression     {regression.estimate:.4f} bootstrap CI "
        f"({regression_ci.interval.lower:.4f}, {regression_ci.interval.upper:.4f})"
    )
    print(f"  IPW stabilized {ipw.estimate:.4f} (SE {ipw.standard_error:.4f})")
    print(f"  AIPW           {aipw.estimate:.4f} (SE {aipw.standard_error:.4f})")
    z = 1.959963984540054
    estimates = {
        "naive difference": (naive, naive - z * naive_se, naive + z * naive_se),
        "regression adjustment": (
            regression.estimate,
            regression_ci.interval.lower,
            regression_ci.interval.upper,
        ),
        "IPW, stabilized": (ipw.estimate, *ipw.confidence_interval()),
        "AIPW": (aipw.estimate, *aipw.confidence_interval()),
    }
    axes = plot_effect_estimates(
        estimates,
        truth=TRUE_EFFECT,
        interval_label="95% Wald interval (regression: bootstrap percentile)",
    )
    save_figure(axes.figure, output / "causal-effect-estimates.png")
    plt.close(axes.figure)

    names = ("x1", "x2", "x3")
    before = standardized_mean_differences(rows, treatment, names=names)
    after = standardized_mean_differences(
        rows, treatment, weights=ipw.weights, names=names
    )
    for raw, weighted in zip(before.covariates, after.covariates, strict=True):
        print(
            f"  SMD {raw.name}: raw {raw.standardized_difference:+.3f}, "
            f"weighted {weighted.standardized_difference:+.3f}"
        )
    axes = plot_covariate_balance(before, after)
    save_figure(axes.figure, output / "causal-covariate-balance.png")
    plt.close(axes.figure)

    spread = statistics.stdev(outcome)
    low, high = aipw.confidence_interval()
    rr = risk_ratio_from_standardized_difference(aipw.estimate / spread)
    rr_low = risk_ratio_from_standardized_difference(low / spread)
    rr_high = risk_ratio_from_standardized_difference(high / spread)
    evalue = e_value(rr, lower=rr_low, upper=rr_high)
    print(
        f"  approximate RR {rr:.3f} ({rr_low:.3f}, {rr_high:.3f}); "
        f"E-value {evalue.point:.3f}, CI limit {evalue.confidence_limit:.3f}"
    )
    axes = plot_e_value_contour(evalue, limit_ratio=rr_low)
    save_figure(axes.figure, output / "causal-e-value.png")
    plt.close(axes.figure)
    print(f"elapsed {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
