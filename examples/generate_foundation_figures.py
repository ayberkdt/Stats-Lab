"""Generate the first educational probability and statistics figures."""

from __future__ import annotations

from pathlib import Path

from probstat.distributions import Bernoulli, Normal
from probstat.probability import (
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
)
from probstat.visualization import (
    plot_bernoulli_pmf,
    plot_empirical_cdf,
    plot_finite_measure,
    plot_joint_pmf,
    plot_normal_pdf,
    plot_random_variable_pmf,
)


def main() -> None:
    """Render deterministic foundation plots into ``examples/generated``.

    The function writes six PNG images and closes their figures so repeated
    runs do not leak GUI resources.

    References:
        [EQ-PROB-002], [EQ-RV-001], [EQ-RV-010], [EQ-DIST-BER-001],
        [EQ-DIST-NORM-001], [EQ-STAT-005], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)

    die = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
    fair_die = FiniteProbabilityMeasure.uniform(die)
    even = die.event((2, 4, 6))
    parity = FiniteRandomVariable.from_function(
        fair_die,
        lambda outcome: "even" if outcome % 2 == 0 else "odd",
        name="Parity",
    )
    coin_pairs = FiniteSampleSpace(("HH", "HT", "TH", "TT"))
    fair_coin_pairs = FiniteProbabilityMeasure.uniform(coin_pairs)
    first_coin = FiniteRandomVariable.from_function(
        fair_coin_pairs, lambda outcome: outcome[0], name="First coin"
    )
    second_coin = FiniteRandomVariable.from_function(
        fair_coin_pairs, lambda outcome: outcome[1], name="Second coin"
    )

    figures = (
        (
            plot_finite_measure(fair_die, highlight=even),
            "finite-probability-measure.png",
        ),
        (plot_random_variable_pmf(parity), "random-variable-pmf.png"),
        (plot_joint_pmf(first_coin.joint_with(second_coin)), "joint-pmf.png"),
        (plot_bernoulli_pmf(Bernoulli(0.65)), "bernoulli-pmf.png"),
        (plot_normal_pdf(Normal(mu=0.0, sigma=1.0)), "normal-pdf.png"),
        (
            plot_empirical_cdf((1.1, 1.3, 1.3, 1.8, 2.4, 2.9)),
            "empirical-cdf.png",
        ),
    )
    for axes, filename in figures:
        axes.figure.tight_layout()
        axes.figure.savefig(output / filename, dpi=160)
        plt.close(axes.figure)


if __name__ == "__main__":
    main()

