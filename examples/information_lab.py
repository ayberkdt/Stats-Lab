"""Information theory laboratory: entropy decomposition and a KSG check.

A two-by-three joint PMF is split into ``H(X|Y)``, ``I(X; Y)`` and
``H(Y|X)``, and the KSG mutual-information estimator is checked against the
bivariate-normal closed form ``-log(1 - rho^2)/2`` over a range of
correlations with seeded replicates.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from probstat.foundations import RandomStream
from probstat.information import ksg_mutual_information, mutual_information
from probstat.visualization import save_figure
from probstat.visualization.information_plots import (
    plot_entropy_decomposition,
    plot_estimator_check,
)

JOINT = ((0.30, 0.10, 0.05), (0.05, 0.20, 0.30))
RHOS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9)
REPLICATES = 6
SAMPLE_SIZE = 300


def gaussian_mutual_information(rho: float) -> float:
    """Return ``-log(1 - rho^2)/2``, the MI of a standard bivariate normal.

    References:
        [REF-COVER-THOMAS-2006, Chapter 8].

    """
    return -0.5 * math.log1p(-rho * rho) + 0.0


def main() -> None:
    """Compute the decomposition and the estimator check; write two figures.

    References:
        [REF-COVER-THOMAS-2006, Chapter 2],
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    output = Path(__file__).parent / "generated"
    print(f"I(X; Y) of the table: {mutual_information(JOINT, unit='bits'):.5f} bits")
    axes = plot_entropy_decomposition(JOINT)
    save_figure(axes.figure, output / "information-decomposition.png")
    plt.close(axes.figure)

    root = RandomStream(2026)
    estimates: list[list[float]] = []
    exact = [gaussian_mutual_information(rho) for rho in RHOS]
    for rho in RHOS:
        group = []
        for replicate in range(REPLICATES):
            rng = root.child("ksg", f"{rho}", replicate).generator()
            xs = [rng.gauss(0.0, 1.0) for _ in range(SAMPLE_SIZE)]
            ys = [
                rho * x + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0) for x in xs
            ]
            group.append(ksg_mutual_information(xs, ys))
        estimates.append(group)
        mean = sum(group) / len(group)
        print(
            f"  rho={rho:.1f}: KSG mean {mean:.4f} nats, "
            f"exact {exact[len(estimates) - 1] + 0.0:.4f}"
        )
    axes = plot_estimator_check(
        RHOS, estimates, gaussian_mutual_information, sample_size=SAMPLE_SIZE
    )
    save_figure(axes.figure, output / "information-ksg-check.png")
    plt.close(axes.figure)
    print(f"elapsed {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
