"""Uncertainty-visualization laboratory: one forecast, six honest encodings.

A demand forecast follows geometric Brownian motion. The same uncertainty is
shown as a fan chart, as individual paths, as a quantile dot plot with a
capacity limit, as an empirical CDF with a distribution-free band, as an
exceedance curve, and as a histogram with quantiles. Each encoding answers a
different reader question; none adds or hides information.
"""

from __future__ import annotations

from pathlib import Path

from probstat.foundations import RandomStream
from probstat.monte_carlo import estimate_mean
from probstat.processes import GeometricBrownianMotion
from probstat.propagation import OutputDistribution
from probstat.visualization import save_figure
from probstat.visualization.uncertainty import (
    plot_ecdf_band,
    plot_fan_chart,
    plot_quantile_dots,
    plot_spaghetti,
)
from probstat.visualization.uq import plot_exceedance_curves, plot_output_distribution

CAPACITY = 160.0


def main() -> None:
    """Simulate the forecast and write one figure per encoding.

    References:
        [REF-SPIEGELHALTER-PEARSON-SHORT-2011], [REF-KAY-ETAL-2016],
        [REF-MASSART-1990], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    process = GeometricBrownianMotion(drift=0.05, volatility=0.25)
    root = RandomStream(2026)
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
