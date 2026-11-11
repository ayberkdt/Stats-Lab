"""Generate the first educational probability and statistics figures.

Run with ``PYTHONPATH=src python examples/generate_foundation_figures.py``.
Thirteen figures are written to ``examples/generated/``: a finite
probability measure with a highlighted event, a random-variable PMF, a joint
PMF, a characteristic function, seven distribution plots, an empirical CDF,
and a grid-refinement convergence plot. Nothing is printed.

Definitions:
    ``Omega``, ``P``: finite sample space (a die, two coins) and its uniform
        probability measure.
    random variable ``X``: a mapping of outcomes (parity, roll value, coin
        faces) whose law is the pushforward of ``P``.
    PMF, CDF, survival function: mass, cumulative, and upper-tail functions
        of the plotted laws (Bernoulli, binomial, categorical, Poisson,
        geometric, negative binomial, exponential, normal).
    ``phi_X(t)``: characteristic function of the die roll.
    ``F_n``: empirical CDF of a seeded exponential sample.
    observed order ``p``: convergence order from three grid solutions.

Assumptions:
    - Every law is exact and finite-parameter; all figures except the
      empirical CDF are deterministic.
    - The empirical-CDF sample is drawn independently from
      ``Exponential(1.25)`` by inverse transformation with ``SEED``.
    - The refinement plot assumes uniform refinement in the asymptotic range
      (values 1.16, 1.04, 1.01 at ratio 2).

Equations:
    EQ-PROB-002: probability of an event on a finite space.
    EQ-RV-001: pushforward PMF of a finite random variable.
    EQ-RV-010: joint PMF of two random variables.
    EQ-GF-003: characteristic function.
    EQ-DIST-BER-001: Bernoulli PMF.
    EQ-DIST-BIN-001: binomial PMF.
    EQ-DIST-CAT-001: categorical PMF.
    EQ-DIST-POIS-001: Poisson PMF.
    EQ-DIST-GEOM-001: geometric PMF.
    EQ-DIST-NBIN-001: negative-binomial PMF.
    EQ-DIST-EXP-001: exponential density.
    EQ-DIST-EXP-002: exponential CDF and survival function.
    EQ-DIST-EXP-005: exponential quantile used to draw the seeded sample.
    EQ-DIST-NORM-001: normal density.
    EQ-STAT-005: empirical CDF.
    EQ-VV-001: observed order of convergence.
    EQ-VV-002: Richardson extrapolation.

Diagnostics:
    - Integer PMFs drawn on a finite window (binomial, Poisson) report the
      probability mass outside the window in their subtitle.
    - The empirical CDF is right-continuous with jumps of ``1/n``.
    - The refinement plot draws the errors of the three grid values against
      the Richardson-extrapolated value on log-log axes, annotated with the
      observed order.

Cost and error:
    - All laws are evaluated in closed form; the only random quantity is
      the 12-point empirical-CDF sample, whose distance to the true CDF is
      of order ``n**-0.5``.
    - Every figure is written at 160 dpi; the run takes a few seconds.

References:
    REF-KOLMOGOROV-1956: axioms of probability.
    REF-ROSS-2019: discrete and continuous distributions.
    REF-CASELLA-BERGER-2002: normal and exponential laws.
    REF-DEVROYE-1986: inverse-transform sampling.
    REF-WASSERMAN-2004: the empirical CDF.
    REF-ROACHE-1998: grid-refinement verification.
    REF-MATPLOTLIB: rendering.

"""

from __future__ import annotations

from pathlib import Path

from probstat.distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    Exponential,
    Geometric,
    NegativeBinomial,
    Normal,
    Poisson,
)
from probstat.probability import (
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
)
from probstat.visualization import (
    plot_bernoulli_pmf,
    plot_categorical_pmf,
    plot_characteristic_function,
    plot_continuous_functions,
    plot_empirical_cdf,
    plot_finite_measure,
    plot_integer_pmf,
    plot_joint_pmf,
    plot_normal_pdf,
    plot_random_variable_pmf,
    plot_refinement_convergence,
    plot_waiting_count_pmfs,
    save_figure,
)

# Seed of the sample drawn for the empirical-CDF figure.
SEED = 2026


def main() -> None:
    """Render the foundation plots into ``examples/generated``.

    The function writes thirteen PNG images and closes their figures so repeated
    runs do not leak GUI resources. Only the empirical-CDF sample is random,
    and it is drawn with ``SEED`` [EQ-DIST-EXP-005], [REF-DEVROYE-1986].

    References:
        [EQ-PROB-002], [EQ-RV-001], [EQ-RV-010], [EQ-GF-003],
        [EQ-DIST-BER-001], [EQ-DIST-BIN-001], [EQ-DIST-CAT-001],
        [EQ-DIST-POIS-001], [EQ-DIST-GEOM-001], [EQ-DIST-NBIN-001],
        [EQ-DIST-EXP-001], [EQ-DIST-EXP-002],
        [EQ-DIST-NORM-001], [EQ-STAT-005], [EQ-VV-001], [EQ-VV-002],
        [REF-MATPLOTLIB].

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
    roll = FiniteRandomVariable.from_function(fair_die, float, name="Die roll")
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
        (
            plot_characteristic_function(roll),
            "characteristic-function.png",
        ),
        (plot_bernoulli_pmf(Bernoulli(0.65)), "bernoulli-pmf.png"),
        (
            plot_integer_pmf(Binomial(12, 0.35), lower=0, upper=12),
            "binomial-pmf.png",
        ),
        (
            plot_categorical_pmf(
                Categorical({"red": 0.50, "green": 0.20, "blue": 0.30})
            ),
            "categorical-pmf.png",
        ),
        (
            plot_integer_pmf(Poisson(4.0), lower=0, upper=12),
            "poisson-pmf-window.png",
        ),
        (
            plot_waiting_count_pmfs(
                Geometric(0.35), NegativeBinomial(4, 0.55), upper=18
            ),
            "waiting-count-pmfs.png",
        ),
        (
            plot_continuous_functions(
                Exponential(1.25), lower=-0.5, upper=4.0
            ),
            "exponential-functions.png",
        ),
        (plot_normal_pdf(Normal(mu=0.0, sigma=1.0)), "normal-pdf.png"),
        (
            plot_empirical_cdf(Exponential(1.25).sample(12, seed=SEED)),
            "empirical-cdf.png",
        ),
        (
            plot_refinement_convergence(1.16, 1.04, 1.01),
            "refinement-convergence.png",
        ),
    )
    for axes, filename in figures:
        axes.figure.tight_layout()
        save_figure(axes.figure, output / filename, dpi=160)
        plt.close(axes.figure)


if __name__ == "__main__":
    main()

