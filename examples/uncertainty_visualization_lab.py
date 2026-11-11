"""Uncertainty-visualization laboratory: one forecast, six honest encodings.

A demand forecast follows geometric Brownian motion. The same uncertainty is
shown as a fan chart, as individual paths, as a quantile dot plot with a
capacity limit, as an empirical CDF with a distribution-free band, as an
exceedance curve, and as a histogram with quantiles. Each encoding answers a
different reader question; none adds or hides information.

Run with ``PYTHONPATH=src python examples/uncertainty_visualization_lab.py``.
The six figures are written to ``examples/generated/``; the printed numbers
state how precise the simulation behind them is.

Definitions:
    ``S_t``: demand at time ``t``, a geometric Brownian motion with drift
        0.05, volatility 0.25, and ``S_0 = 100`` on ``0 <= t <= 4``.
    path: one simulated trajectory on 48 equal steps; 400 paths form the
        forecast ensemble.
    exceedance probability: ``P(S_4 > 160)``, the share of paths above the
        capacity at the horizon.
    fan chart: nested central quantile bands of ``S_t`` over time.
    quantile dot plot: 50 equally likely outcomes, the sample quantiles at
        ``(k + 1/2) / 50``.
    DKW band: a distribution-free simultaneous band around the empirical CDF.

Assumptions:
    - The paths are independent and simulated exactly from the log-normal
      transition law, so there is no time-discretization error.
    - The 400 terminal values are an i.i.d. sample of ``S_4``; Monte Carlo
      error is the only error in the printed estimates.
    - The DKW band holds for any continuous law but is simultaneous at 95%,
      so it is wider than a pointwise interval.

Equations:
    EQ-RNG-SPLITMIX-001: seeded, spawned random streams, one per path.
    EQ-PROC-GBM-001: exact solution used to simulate each path.
    EQ-PROC-GBM-002: exact mean ``S_0 exp(0.05 t)`` printed for comparison.
    EQ-MC-MEAN-001: Monte Carlo mean of the terminal demand.
    EQ-MC-SE-001: its estimated standard error.
    EQ-PROP-THRESH-001: strict exceedance probability of the capacity.
    EQ-INF-CI-WILSON-001: Wilson interval for that probability.
    EQ-STAT-QUANT-001: sample quantiles behind the fan chart, dots, and
        histogram markers.
    EQ-STAT-005: empirical CDF and exceedance curve.
    EQ-STAT-DKW-001: half-width of the DKW band.

Diagnostics:
    - The simulated terminal mean is compared with the exact mean; the gap
      should be within about two standard errors.
    - The Wilson interval says how many paths are needed before the
      exceedance probability is known to the precision a decision needs.
    - The DKW half-width at ``n = 120`` is printed and drawn: an empirical
      CDF from 120 paths is uncertain by that much everywhere at once.

Cost and error:
    - 400 paths of 48 steps, 19,200 exact transition draws.
    - The terminal mean carries the printed standard error (about 3.2); the
      exceedance probability has a Wilson half-width of about 0.04.
    - The DKW half-width ``sqrt(log(2 / 0.05) / (2 n))`` is 0.124 at
      ``n = 120`` and falls as ``n^(-1/2)``.

References:
    REF-SPIEGELHALTER-PEARSON-SHORT-2011: visualizing uncertainty about the
        future (fan charts, spaghetti plots).
    REF-KAY-ETAL-2016: quantile dot plots for frequency framing.
    REF-MASSART-1990: the tight constant of the DKW inequality.
    REF-GLASSERMAN-2003: exact simulation of geometric Brownian motion.
    REF-WILSON-1927: score interval for a binomial probability.
    REF-ROBERT-CASELLA-2004: Monte Carlo means and standard errors.
    REF-MATPLOTLIB: rendering.

"""

from __future__ import annotations

from pathlib import Path

from probstat.foundations import RandomStream
from probstat.inference import wilson_score_interval
from probstat.monte_carlo import estimate_mean
from probstat.processes import GeometricBrownianMotion
from probstat.propagation import OutputDistribution
from probstat.visualization import save_figure
from probstat.visualization.uncertainty import (
    dkw_epsilon,
    plot_ecdf_band,
    plot_fan_chart,
    plot_quantile_dots,
    plot_spaghetti,
)
from probstat.visualization.uq import plot_exceedance_curves, plot_output_distribution

CAPACITY = 160.0
# Root seed of the 400 spawned path streams.
SEED = 2026


def main() -> None:
    """Simulate the forecast and write one figure per encoding.

    References:
        [REF-SPIEGELHALTER-PEARSON-SHORT-2011], [REF-KAY-ETAL-2016],
        [REF-MASSART-1990], [REF-WILSON-1927], [EQ-INF-CI-WILSON-001],
        [EQ-STAT-DKW-001], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    process = GeometricBrownianMotion(drift=0.05, volatility=0.25)
    root = RandomStream(SEED)
    paths = [
        process.path(4.0, 48, initial=100.0, seed=stream)
        for stream in root.spawn(400, label="demand")
    ]
    times = paths[0].times
    values = [path.values for path in paths]
    terminal = [path.values[-1] for path in paths]
    exceed = sum(value > CAPACITY for value in terminal) / len(terminal)
    exact_mean = process.mean(100.0, 4.0)
    print(f"terminal mean={sum(terminal) / len(terminal):.2f} (exact {exact_mean:.2f})")
    print(f"P(demand > capacity at t=4)={exceed:.3f}")
    mean_estimate = estimate_mean(terminal)
    exceed_count = sum(value > CAPACITY for value in terminal)
    exceed_interval = wilson_score_interval(exceed_count, len(terminal))
    print(f"  terminal mean standard error={mean_estimate.standard_error:.2f}")
    print(
        "  95% Wilson interval for the exceedance probability="
        f"[{exceed_interval.lower:.3f}, {exceed_interval.upper:.3f}]"
    )
    print(f"  DKW 95% band half-width at n=120: {dkw_epsilon(120):.3f}")

    output = Path(__file__).parent / "generated"
    figures = {
        "uncertainty-fan-chart": plot_fan_chart(times, values, name="Demand"),
        "uncertainty-spaghetti": plot_spaghetti(times, values[:60], name="Demand"),
        "uncertainty-quantile-dots": plot_quantile_dots(
            terminal, dots=50, threshold=CAPACITY, name="Demand at t = 4"
        ),
        "uncertainty-ecdf-band": plot_ecdf_band(
            terminal[:120], name="Demand at t = 4 (first 120 paths)"
        ),
        "uncertainty-exceedance": plot_exceedance_curves(
            {
                "simulated": OutputDistribution(
                    "demand", tuple(terminal), estimate_mean(terminal)
                ).exceedance_curve(80)
            },
            threshold=CAPACITY,
            x_label="Demand at t = 4",
        ),
        "uncertainty-histogram": plot_output_distribution(
            terminal, name="Demand at t = 4", threshold=CAPACITY
        ),
    }
    for name, axes in figures.items():
        save_figure(axes.figure, output / f"{name}.png")
        plt.close(axes.figure)
    print(f"wrote {len(figures)} encodings of one forecast")


if __name__ == "__main__":
    main()
