"""Limit theorems and estimator properties laboratory.

Part 1 answers limit-theorem questions exactly: Kolmogorov distances of the
standardized mean to ``Phi`` from the binomial and gamma laws, the
Berry-Esseen bound, the Edgeworth correction, tail bounds against exact
tails, and delta-method variances. Part 2 shows the same phenomena by
simulation and states the Monte Carlo error. Part 3 evaluates estimators:
exact risks, Cramer-Rao bounds, Rao-Blackwellization, and a simulated
check of every exact formula with its Monte Carlo standard error.

Run with ``PYTHONPATH=src python examples/limit_theorems_lab.py``. With
Matplotlib installed one figure is written to ``examples/generated/``: the
exact Kolmogorov distance of standardized exponential means to ``Phi`` and to
the Edgeworth approximation, under the Berry-Esseen bound.

Definitions:
    ``Xbar_n``: mean of ``n`` i.i.d. summands (Bernoulli(p), Exponential(1),
        Poisson(1), or N(0, 1)); ``Z_n = sqrt(n) (Xbar_n - mu) / sigma``.
    Kolmogorov distance: ``sup_x |P(Z_n <= x) - G(x)|`` for a reference CDF
        ``G`` (``Phi`` or its one-term Edgeworth correction).
    tail bound: an upper bound on ``P(|Xbar_n - mu| > t)`` (Chebyshev,
        Hoeffding, Bernstein) compared with the exact tail.
    risk: bias, variance, and mean squared error of an estimator;
        efficiency: Cramer-Rao lower bound divided by the MSE.
    Rao-Blackwellization: conditioning a crude unbiased estimator on a
        sufficient statistic.

Assumptions:
    - Summands are independent and identically distributed with finite
      third absolute moment, so the Berry-Esseen bound applies.
    - Part 1 and the estimator risks of Part 3 are exact: the laws of the
      sums are binomial or gamma, so no simulation error enters.
    - Hoeffding's bound needs bounded summands (Bernoulli here); the delta
      method is a first-order large-sample approximation.
    - Simulated checks use independent seeded replications; their Monte
      Carlo standard errors are printed beside the estimates.

Equations:
    EQ-LIM-001: Berry-Esseen bound with Shevtsova's constant 0.4748.
    EQ-LIM-002: exact laws of the standardized mean and exact tails.
    EQ-LIM-003: Kolmogorov distance to the normal law.
    EQ-LIM-004: one-term Edgeworth expansion.
    EQ-LIM-005: Chebyshev bound.
    EQ-LIM-006: Hoeffding bound.
    EQ-LIM-007: Bernstein bound.
    EQ-LIM-008: delta method against exact small-sample variances.
    EQ-LIM-009: simulation estimates of the weak law, running means, and
        the CLT distance.
    EQ-LIM-010: third central moments of the summands.
    EQ-EST-001: Monte Carlo bias, variance, and MSE with standard errors.
    EQ-EST-002: paired relative efficiency of Rao-Blackwellization.
    EQ-EST-004: Cramer-Rao lower bound.
    EQ-EST-005: normal-variance divisor family ``SS / d``.
    EQ-EST-006: Uniform(0, theta) maximum estimators.
    EQ-EST-007: Rao-Blackwellized Poisson zero probability.
    EQ-EST-010: exact moments of the exponential-rate MLE.

Diagnostics:
    - Exact distance against the Berry-Esseen bound: the printed ratio shows
      how conservative the bound is (well below 1 for every row).
    - Edgeworth against normal: the correction reduces the distance for the
      skewed exponential summands at every ``n``; the figure shows the gap.
    - Every simulated quantity is printed next to its exact value and its
      Monte Carlo standard error; agreement within about two standard
      errors checks both the simulation and the exact formula.
    - Efficiency below 1 marks an estimator that does not reach the
      Cramer-Rao bound at finite ``n``.

Cost and error:
    - Part 1 is exact and costs milliseconds.
    - Part 2: 2,000 replications per weak-law point, 300 running-mean paths
      of length 1,000, and 5,000 replications per CLT distance; a simulated
      probability ``p`` carries a standard error of ``sqrt(p (1 - p) / R)``
      and the KS distance a sampling noise of order ``R^(-1/2)``.
    - Part 3: 5,000 replications per simulated estimator check; the MSE and
      bias standard errors are printed.
    - The final line prints the wall-clock time of the run.

References:
    REF-SHEVTSOVA-2011: the Berry-Esseen constant.
    REF-HALL-1992: Edgeworth expansions.
    REF-HOEFFDING-1963: Hoeffding's inequality.
    REF-BOUCHERON-LUGOSI-MASSART-2013: Bernstein's inequality.
    REF-CASELLA-BERGER-2002: limit theorems, the delta method, and
        estimator risk.
    REF-LEHMANN-CASELLA-1998: Cramer-Rao bounds and Rao-Blackwellization.
    REF-MATPLOTLIB: rendering.

"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from probstat.inference import (
    evaluate_estimator,
    mle_efficiency,
    normal_variance_estimator_risk,
    poisson_zero_indicator,
    poisson_zero_probability_risks,
    poisson_zero_rao_blackwell,
    uniform_maximum_risks,
)
from probstat.monte_carlo import (
    simulate_clt_distance,
    simulate_running_means,
    simulate_weak_law,
)
from probstat.probability import (
    bernoulli_summand,
    clt_error_profile,
    compare_tail_bounds,
    delta_method_exponential_reciprocal,
    delta_method_normal_square,
    exponential_summand,
    kolmogorov_distance_to_normal,
    sample_mean_two_sided_tail,
)
from probstat.visualization import active_palette, chart_theme, save_figure
from probstat.visualization.theme import (
    REFERENCE,
    REFERENCE_WIDTH,
    axis_labels,
    direct_labels,
    line,
    matplotlib_available,
    new_axes,
    points,
    title_block,
)

# Seed of every simulated check in Parts 2 and 3.
SEED = 20261012
OUTPUT = Path(__file__).parent / "generated"
CLT_SIZES = (2, 5, 10, 30, 100)


def exponential_draw(rng: random.Random) -> float:
    """Draw one Exponential(1) summand.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    return rng.expovariate(1.0)


def poisson_sample(n: int, rng: random.Random) -> list[float]:
    """Draw ``n`` Poisson(1) counts by Knuth's product-of-uniforms method.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.2].

    """
    limit = math.exp(-1.0)
    sample = []
    for _ in range(n):
        count, product = 0, rng.random()
        while product > limit:
            count += 1
            product *= rng.random()
        sample.append(float(count))
    return sample


def normal_sample(n: int, rng: random.Random) -> list[float]:
    """Draw ``n`` N(0, 1) observations.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def unbiased_variance(sample: Sequence[float]) -> float:
    """Return ``SS / (n - 1)``.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.2].

    """
    center = math.fsum(sample) / len(sample)
    return math.fsum((x - center) ** 2 for x in sample) / (len(sample) - 1)


def part_one() -> None:
    """Print exact limit-theorem results.

    References:
        [REF-SHEVTSOVA-2011], [REF-HALL-1992, Chapter 2],
        [REF-HOEFFDING-1963].

    """
    print("== Part 1: exact results ==")
    print("Bernoulli means: exact Kolmogorov distance vs Berry-Esseen (C=0.4748)")
    print(f"{'p':>5} {'n':>5} {'exact':>9} {'bound':>9} {'ratio':>6}")
    for p in (0.5, 0.1):
        for row in clt_error_profile(bernoulli_summand(p), (10, 100, 1000)):
            ratio = row.kolmogorov_normal / row.berry_esseen
            print(
                f"{p:5.2f} {row.n:5d} {row.kolmogorov_normal:9.5f} "
                f"{row.berry_esseen:9.5f} {ratio:6.3f}"
            )
    print("\nExponential means: normal vs Edgeworth vs Berry-Esseen")
    print(f"{'n':>5} {'Phi':>9} {'Edgeworth':>10} {'bound':>9}")
    for row in clt_error_profile(exponential_summand(1.0), CLT_SIZES):
        print(
            f"{row.n:5d} {row.kolmogorov_normal:9.5f} "
            f"{row.kolmogorov_edgeworth or 0.0:10.6f} {row.berry_esseen:9.5f}"
        )
    print("\nBernoulli(0.3), n=100: exact tails vs bounds")
    header = ("t", "exact1", "Hoeff", "Bern", "exact2", "Cheb")
    print(" ".join(f"{h:>10}" for h in header))
    for t in (0.05, 0.1, 0.2):
        row = compare_tail_bounds(bernoulli_summand(0.3), 100, t)
        print(
            f"{t:10.2f} {row.exact_upper:10.3e} {row.hoeffding or 0.0:10.3e} "
            f"{row.bernstein or 0.0:10.3e} {row.exact_two_sided:10.3e} "
            f"{row.chebyshev:10.4f}"
        )
    print("\nDelta method: exact vs first-order variance")
    for n in (5, 20, 100):
        square = delta_method_normal_square(1.0, 1.0, n)
        reciprocal = delta_method_exponential_reciprocal(1.0, n)
        print(
            f"n={n:4d}  Var(Xbar^2) exact/delta = {square.ratio:6.3f}   "
            f"Var(1/Xbar) exact/delta = {reciprocal.ratio:6.3f}"
        )


def part_two() -> None:
    """Print simulation demonstrations next to exact answers.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5].

    """
    print("\n== Part 2: simulation (Exponential(1) summands, eps = 0.2) ==")
    points = simulate_weak_law(
        exponential_draw, 1.0, 0.2, (10, 50, 200), replications=2000, seed=SEED
    )
    summand = exponential_summand(1.0)
    for point in points:
        exact = sample_mean_two_sided_tail(summand, point.n, 0.2)
        print(
            f"P(|Xbar_{point.n} - 1| > 0.2): simulated {point.probability:.4f} "
            f"+/- {point.standard_error:.4f}, exact {exact:.4f}"
        )
    paths = simulate_running_means(
        exponential_draw, 1.0, 0.2, (10, 100, 1000), paths=300, seed=SEED
    )
    for m, freq in zip(paths.checkpoints, paths.sup_exceedance, strict=True):
        print(f"P(max_(m<=k<=1000) |Xbar_k - 1| > 0.2), m={m:4d}: {freq:.3f}")
    for n in (5, 30):
        sim = simulate_clt_distance(
            exponential_draw, 1.0, 1.0, n, replications=5000, seed=SEED
        )
        exact = kolmogorov_distance_to_normal(summand, n)
        print(
            f"KS(Z_{n}, Phi): simulated {sim.ks_distance:.4f} (noise scale "
            f"{sim.sampling_scale:.4f}), exact {exact:.4f}"
        )


def part_three() -> None:
    """Print exact estimator risks and their simulated checks.

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3], [REF-LEHMANN-CASELLA-1998,
        Chapter 2].

    """
    print("\n== Part 3: estimators ==")
    n = 10
    for d in (n - 1, n, n + 1):
        risk = normal_variance_estimator_risk(1.0, n, d)
        print(
            f"SS/{d:2d}: bias {risk.bias:+.4f} variance {risk.variance:.4f} "
            f"MSE {risk.mse:.4f}"
        )
    for risk in uniform_maximum_risks(1.0, n):
        print(
            f"Uniform(0,1) {risk.label:>16}: bias {risk.bias:+.4f} "
            f"MSE {risk.mse:.5f}"
        )
    for family, value in (("poisson", 2.0), ("normal_variance", 1.0)):
        record = mle_efficiency(family, n, value)  # type: ignore[arg-type]
        print(
            f"MLE {family:>16}: variance {record.variance:.4f} CRLB "
            f"{record.crlb:.4f} efficiency CRLB/MSE {record.efficiency:.3f}"
        )
    for size in (5, 50, 500):
        record = mle_efficiency("exponential_rate", size, 1.0)
        print(f"MLE exponential_rate n={size:3d}: efficiency {record.efficiency:.4f}")
    risks = poisson_zero_probability_risks(1.0, n)
    print(
        f"P(X=0), lambda=1, n={n}: crude var {risks.crude.variance:.5f}, "
        f"Rao-Blackwell var {risks.rao_blackwellized.variance:.5f}, "
        f"CRLB {risks.crlb:.5f}"
    )
    rb = evaluate_estimator(
        poisson_zero_rao_blackwell,
        poisson_sample,
        math.exp(-1.0),
        n,
        5000,
        seed=SEED,
        reference=poisson_zero_indicator,
    )
    print(
        f"  simulated RB: bias {rb.bias:+.5f} +/- {rb.bias_se:.5f}, variance "
        f"{rb.variance:.5f} +/- {rb.variance_se:.5f}, efficiency vs crude "
        f"{rb.relative_efficiency or 0.0:.2f} +/- "
        f"{rb.relative_efficiency_se or 0.0:.2f}"
    )
    s2 = evaluate_estimator(unbiased_variance, normal_sample, 1.0, n, 5000, seed=SEED)
    exact = normal_variance_estimator_risk(1.0, n, n - 1)
    print(
        f"  simulated SS/(n-1): MSE {s2.mse:.4f} +/- {s2.mse_se:.4f} "
        f"(exact {exact.mse:.4f})"
    )


def clt_error_figure() -> Any:
    """Draw the exact CLT error of exponential means against its bound.

    Lines show the exact Kolmogorov distance of ``Z_n`` to ``Phi`` (observed
    hue) and to the one-term Edgeworth approximation (model hue) on log-log
    axes; the Berry-Esseen bound is the accent reference. The title states
    the values at the largest ``n``.

    Returns:
        Populated Matplotlib axes.

    References:
        [EQ-LIM-001], [EQ-LIM-003], [EQ-LIM-004], [REF-SHEVTSOVA-2011],
        [REF-HALL-1992], [REF-MATPLOTLIB].

    """
    from matplotlib.ticker import FuncFormatter, NullLocator

    rows = clt_error_profile(exponential_summand(1.0), CLT_SIZES)
    sizes = [float(row.n) for row in rows]
    normal = [row.kolmogorov_normal for row in rows]
    edgeworth = [row.kolmogorov_edgeworth or math.nan for row in rows]
    bound = [row.berry_esseen for row in rows]
    palette = active_palette()
    ax = new_axes()
    line(
        ax, sizes, bound, color=palette.accent, dashes=REFERENCE, width=REFERENCE_WIDTH
    )
    line(ax, sizes, normal, color=palette.observed)
    points(ax, sizes, normal, color=palette.observed)
    line(ax, sizes, edgeworth, color=palette.model)
    points(ax, sizes, edgeworth, color=palette.model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [f"{row.n}" for row in rows])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_labels(ax, "Number of summands n", "Kolmogorov distance")
    direct_labels(
        ax,
        (
            (sizes[-1], bound[-1], "Berry-Esseen bound", palette.accent),
            (sizes[-1], normal[-1], "to the normal law", palette.observed),
            (sizes[-1], edgeworth[-1], "to Edgeworth", palette.model),
        ),
    )
    title_block(
        ax,
        f"At n = {rows[-1].n} the normal law is off by {normal[-1]:.3f}, "
        f"Edgeworth by {edgeworth[-1]:.4f}",
        "Exact distances for standardized means of Exponential(1) summands",
    )
    return ax


def write_figures() -> None:
    """Write the CLT error figure when Matplotlib is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if not matplotlib_available():
        return
    import matplotlib.pyplot as plt

    with chart_theme():
        ax = clt_error_figure()
        save_figure(ax.figure, OUTPUT / "limit-theorems-clt-error.png")
        plt.close(ax.figure)


def main() -> None:
    """Run the three parts and report the elapsed time.

    References:
        Project example policy. No equation is evaluated.

    """
    start = time.perf_counter()
    part_one()
    part_two()
    part_three()
    write_figures()
    elapsed = time.perf_counter() - start
    print(
        f"\nDone in {elapsed:.1f} s. Next: docs/foundations/"
        "limit-theorems-and-estimators.md explains each table."
    )


if __name__ == "__main__":
    main()
