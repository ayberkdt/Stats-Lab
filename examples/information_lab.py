"""Information theory laboratory: entropy decomposition and a KSG check.

A two-by-three joint PMF is split into ``H(X|Y)``, ``I(X; Y)`` and
``H(Y|X)``, and the KSG mutual-information estimator is checked against the
bivariate-normal closed form ``-log(1 - rho^2)/2`` over a range of
correlations with seeded replicates.

Definitions:
    ``p(x, y)``: the two-by-three joint PMF ``JOINT`` with rows indexing
        ``X``.
    ``H(X)``, ``H(X, Y)``, ``H(Y | X)``: Shannon, joint, and conditional
        entropies.
    ``I(X; Y)``: mutual information, printed in bits for the table and in
        nats for the Gaussian check.
    ``rho``: correlation of a standard bivariate normal pair.
    KSG estimate: Kraskov-Stogbauer-Grassberger k-nearest-neighbour
        estimator of ``I(X; Y)`` from a sample (``k = 3``).

Assumptions:
    - The joint PMF sums to one; the decomposition is exact.
    - Each KSG replicate is an independent sample of ``SAMPLE_SIZE`` pairs
      from the bivariate normal with the stated ``rho``; continuous data
      have no ties.
    - The reference value ``-log(1 - rho^2)/2`` is the closed-form mutual
      information of a standard bivariate normal, computed in the lab.

Equations:
    EQ-INFO-H-001: Shannon entropy of each marginal.
    EQ-INFO-H-002: joint and conditional entropy (chain rule).
    EQ-INFO-MI-001: mutual information of the table.
    EQ-INFO-KSG-001: KSG estimator in the Gaussian check.
    EQ-RNG-SPLITMIX-001: SplitMix64 child streams, one per ``(rho,
        replicate)``.

Diagnostics:
    - The entropy-decomposition figure shows that ``H(X|Y) + I(X; Y) +
      H(Y|X)`` fills ``H(X, Y)``.
    - Replicate mean against the exact value per ``rho``; the figure draws
      +/- one replicate standard deviation and its title names the largest
      absolute gap. The estimate at ``rho = 0`` may be slightly negative,
      which reveals the estimator's finite-sample bias.

Cost and error:
    - 6 correlations x 6 replicates x 300 pairs; each replicate mean has a
      standard error of about the replicate standard deviation over
      ``sqrt(6)``, and the KSG estimator has a bias that does not vanish
      with more replicates, only with larger samples.
    - The final line prints the elapsed wall-clock time, which varies
      between runs; every other number is fixed by ``SEED``.

References:
    REF-COVER-THOMAS-2006: entropy and mutual information.
    REF-KRASKOV-STOGBAUER-GRASSBERGER-2004: the KSG estimator.
    REF-STEELE-LEA-FLOOD-2014: SplitMix64 streams.

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
# Root seed of the KSG replicate streams.
SEED = 2026


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

    root = RandomStream(SEED)
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
