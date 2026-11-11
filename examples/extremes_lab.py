"""Extreme value laboratory: block maxima, threshold choice, and tail fit.

Forty "years" of daily data are drawn from a generalized Pareto law with
known shape ``xi = 0.1``, so the true return levels are known exactly. The
annual maxima are fitted by a GEV law and their return levels shown with
profile-likelihood intervals against the truth; the mean-residual-life plot
supports a POT threshold, and a Q-Q plot with order-statistic bands checks the
GPD fit of the excesses. Run with
``PYTHONPATH=src python examples/extremes_lab.py``; three figures are
written to ``examples/generated/``.

Definitions:
    Block maximum ``M_k``: the largest of the 365 daily values of year ``k``.
    GEV ``(mu, sigma, xi)``: generalized extreme value law fitted to the 40
    annual maxima.
    ``T``-year return level ``z_T``: the level exceeded by an annual maximum
    with probability ``1/T``.
    POT: peaks over the threshold ``u = 3.5``; the excesses ``X - u`` are
    fitted by a generalized Pareto law ``GPD(sigma_u, xi)``.
    Mean residual life ``e(u) = E[X - u | X > u]``, linear in ``u`` above a
    valid GPD threshold.

Assumptions:
    - Daily values are independent and identically distributed
      ``GPD(0, 1, 0.1)``, so no declustering is needed and the true return
      levels follow exactly from ``F^365``.
    - Forty blocks are enough for the asymptotic GEV approximation of the
      annual maxima, and ``u = 3.5`` is high enough for the GPD
      approximation of the excesses.
    - Profile-likelihood and delta-method intervals rely on large-sample
      likelihood theory.

Equations:
    EQ-DIST-GPD-002: GPD quantile used to simulate the daily series.
    EQ-EVA-BM-001: block maxima and the GEV likelihood.
    EQ-INF-MLE-GEN-001: numerical maximum-likelihood fit.
    EQ-EVA-RL-001: GEV return level.
    EQ-EVA-PROF-001: profile-likelihood return-level interval.
    EQ-EVA-MRL-001: mean residual life with its interval.
    EQ-EVA-GPD-001: threshold excesses and the GPD likelihood.
    EQ-EVA-RATE-001: exceedance rate per year.
    EQ-EVA-POT-001: POT return level.
    EQ-EVA-POT-002: delta-method POT return-level interval.
    EQ-VIZ-QQBAND-001: pointwise order-statistic band of the Q-Q plot.
    EQ-RNG-SPLITMIX-001: seeded child stream of the daily series.

Diagnostics:
    - Every return level is printed with its interval beside the exact
      level; the figure shows the fitted curve against the empirical maxima.
    - The mean-residual-life plot shows linearity above the chosen
      threshold.
    - The Q-Q plot of the excesses against the fitted GPD has a pointwise
      band; about 5% of points fall outside a 95% band by chance.

Cost and error:
    - 14 600 simulated days, 40 annual maxima, 676 threshold excesses; the
      fits are deterministic given the data.
    - Return levels carry sampling error, shown by the interval widths; the
      100-year GEV interval is wide and skewed because 40 maxima barely
      reach that return period.

References:
    REF-COLES-2001: extreme value modelling.
    REF-PICKANDS-1975: the generalized Pareto approximation.
    REF-WILK-GNANADESIKAN-1968: probability plots.

"""

from __future__ import annotations

import time
from pathlib import Path

from probstat.distributions import GeneralizedPareto
from probstat.foundations import RandomStream
from probstat.risk import (
    block_maxima,
    fit_gev,
    fit_pot,
    mean_residual_life,
)
from probstat.visualization import save_figure
from probstat.visualization.extremes_plots import (
    plot_mean_residual_life,
    plot_return_levels,
    plot_tail_quantiles,
)

# Root seed; the daily series draws from its "daily-series" child stream.
SEED = 2026
YEARS = 40
DAYS = 365
DAILY_LAW = GeneralizedPareto(0.0, 1.0, 0.1)
THRESHOLD = 3.5
PERIODS = (
    1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0,
    60.0, 100.0, 150.0, 200.0,
)  # fmt: skip


def true_return_level(period: float) -> float:
    """Return the exact level whose annual exceedance probability is ``1/T``.

    The annual maximum of ``DAYS`` independent daily values has CDF ``F^DAYS``,
    so its ``1 - 1/T`` quantile is ``F^-1((1 - 1/T)^(1/DAYS))``.

    References:
        [REF-COLES-2001, Section 3.1].

    """
    return DAILY_LAW.quantile((1.0 - 1.0 / period) ** (1.0 / DAYS))


def main() -> None:
    """Simulate the series, fit both tail models, and write three figures.

    References:
        [REF-COLES-2001, Chapters 3 and 4], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    output = Path(__file__).parent / "generated"
    rng = RandomStream(SEED).child("daily-series").generator()
    series = [DAILY_LAW.quantile(rng.random()) for _ in range(YEARS * DAYS)]

    maxima = block_maxima(series, DAYS)
    gev = fit_gev(maxima)
    print(
        f"GEV fit to {len(maxima)} annual maxima: mu={gev.mu:.3f}, "
        f"sigma={gev.sigma:.3f}, xi={gev.xi:.3f}"
    )
    levels = [gev.profile_return_level(period) for period in PERIODS]
    for entry in levels:
        if entry.return_period in (10.0, 100.0):
            print(
                f"  {entry.return_period:g}-year level {entry.level:.3f} "
                f"profile CI ({entry.lower:.3f}, {entry.upper:.3f}); "
                f"true {true_return_level(entry.return_period):.3f}"
            )
    axes = plot_return_levels(
        levels,
        maxima,
        truth=[true_return_level(period) for period in PERIODS],
        headline_period=100.0,
        name="Annual maximum",
    )
    save_figure(axes.figure, output / "extremes-return-levels.png")
    plt.close(axes.figure)

    mrl = mean_residual_life(series)
    pot = fit_pot(series, THRESHOLD, observations_per_year=DAYS)
    print(
        f"POT at u={THRESHOLD}: {pot.rate.exceedances} excesses, "
        f"sigma={pot.gpd.sigma:.3f}, xi={pot.gpd.xi:.3f}"
    )
    for period in (10.0, 100.0):
        entry = pot.return_level(period)
        print(
            f"  POT {period:g}-year level {entry.level:.3f} delta CI "
            f"({entry.lower:.3f}, {entry.upper:.3f})"
        )
    axes = plot_mean_residual_life(mrl, threshold=THRESHOLD, gpd=pot.gpd)
    save_figure(axes.figure, output / "extremes-mean-residual-life.png")
    plt.close(axes.figure)

    exceedances = [x for x in series if x > THRESHOLD]
    axes = plot_tail_quantiles(
        exceedances, pot.gpd.distribution, name="daily value above u"
    )
    save_figure(axes.figure, output / "extremes-tail-qq.png")
    plt.close(axes.figure)
    print(f"elapsed {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
