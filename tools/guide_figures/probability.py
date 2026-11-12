"""Figures of the probability foundation guides (ADR-0105).

Guides: probability, random variables, measure-theoretic probability,
discrete distributions, waiting times and continuous laws, the continuous
distribution catalog, multivariate laws, limit theorems and estimators, and
information theory. Every figure evaluates the library's own exact laws,
bounds, or seeded computations; nothing is copied from the guide text.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-ROSS-2019].

"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from guide_figures import GuideFigure
from probstat.distributions import (
    Binomial,
    Exponential,
    Gumbel,
    Logistic,
    Normal,
    Poisson,
    StudentT,
    Weibull,
)
from probstat.inference.estimator_properties import (
    minimum_mse_variance_divisor,
    normal_variance_estimator_risk,
)
from probstat.information import entropy as shannon_entropy
from probstat.information import (
    jensen_shannon_divergence,
    kl_divergence,
    marginals,
    mutual_information,
    total_variation_distance,
)
from probstat.multivariate import Dirichlet, dirichlet_posterior
from probstat.probability import (
    FiniteMeasurableSpace,
    FiniteMeasure,
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
    bernoulli_summand,
    clt_error_profile,
    compare_tail_bounds,
    exponential_summand,
)
from probstat.visualization import plot_random_variable_pmf, theme
from probstat.visualization.information_plots import plot_entropy_decomposition

EULER_GAMMA = 0.5772156649015329


def _axes(
    ncols: int = 1,
    *,
    figsize: tuple[float, float] = (7.2, 4.0),
    sharey: bool = False,
) -> tuple[Any, list[Any]]:
    """Return a themed figure with ``ncols`` axes in one row.

    References:
        [REF-MATPLOTLIB].

    """
    plt = theme.pyplot()
    figure, grid = plt.subplots(
        1, ncols, figsize=figsize, sharey=sharey, squeeze=False
    )
    axes = list(grid[0])
    for ax in axes:
        theme.style_axes(ax)
    return figure, axes


def _heading(ax: Any, text: str) -> None:
    """Write a bold panel heading inside the top-left corner of an axes.

    References:
        [REF-MATPLOTLIB].

    """
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=theme.active_palette().ink,
    )


def _label(ax: Any, x: float, y: float, text: str, colour: str, **kw: Any) -> None:
    """Write a direct label in a series colour next to a data point.

    References:
        [REF-MATPLOTLIB].

    """
    ax.annotate(
        text,
        (x, y),
        xytext=kw.pop("offset", (6.0, 0.0)),
        textcoords="offset points",
        ha=kw.pop("ha", "left"),
        va=kw.pop("va", "center"),
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=colour,
        **kw,
    )


def _luminance(colour: str) -> float:
    """Return the WCAG relative luminance of a colour.

    References:
        [REF-W3C-WCAG21-2018].

    """
    from matplotlib.colors import to_rgb

    def channel(value: float) -> float:
        """Linearize one sRGB channel.

        References:
            [REF-W3C-WCAG21-2018].

        """
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(v) for v in to_rgb(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _legible(colour: str) -> str:
    """Move a light series colour toward the ink until text reaches 4.5:1.

    The lightest members of :func:`theme.ordered_colours` are too pale for
    9 pt labels on the canvas; their labels keep the hue and gain contrast.

    References:
        [REF-W3C-WCAG21-2018], [REF-MATPLOTLIB].

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = theme.active_palette()
    canvas = _luminance(palette.canvas)
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        share = step / 20
        mixed = to_hex(tuple(b + share * (i - b) for b, i in zip(base, ink,
                                                                  strict=True)))
        light, dark = sorted((_luminance(mixed), canvas), reverse=True)
        if (light + 0.05) / (dark + 0.05) >= 4.5:
            return mixed
    return palette.ink


def _sci(value: float) -> str:
    """Format a small probability as ``2.87e-7`` (three significant digits).

    References:
        [REF-PROBSTAT-REPORTING].

    """
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _log_ticks(ax: Any, low: int, high: int, *, axis: str = "y") -> None:
    """Put decade ticks ``10^low .. 10^high`` written as powers of ten.

    References:
        [REF-MATPLOTLIB].

    """
    from matplotlib.ticker import FixedLocator, NullLocator

    ticks = [10.0**k for k in range(low, high + 1)]
    labels = [f"{10**k:g}" if 0 <= k <= 3 else f"$10^{{{k}}}$" for k in
              range(low, high + 1)]
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_minor_locator(NullLocator())
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(labels)


# ---------------------------------------------------------------------------
# probability.md
# ---------------------------------------------------------------------------


def _posterior_after_positive(
    prevalence: float, sensitivity: float, false_positive: float
) -> float:
    """Return ``P(condition | positive)`` from a four-outcome finite measure.

    References:
        [EQ-PROB-008], [REF-ROSS-2019, Chapter 3].

    """
    space = FiniteSampleSpace(("ill, +", "ill, -", "well, +", "well, -"))
    measure = FiniteProbabilityMeasure(
        space,
        {
            "ill, +": prevalence * sensitivity,
            "ill, -": prevalence * (1.0 - sensitivity),
            "well, +": (1.0 - prevalence) * false_positive,
            "well, -": 1.0 - prevalence * sensitivity
            - prevalence * (1.0 - sensitivity)
            - (1.0 - prevalence) * false_positive,
        },
    )
    condition = space.event(("ill, +", "ill, -"))
    positive = space.event(("ill, +", "well, +"))
    return measure.bayes_probability(condition, positive)


def draw_base_rate() -> tuple[Any, dict[str, str]]:
    """Posterior probability after a positive test against the base rate.

    References:
        [EQ-PROB-007], [EQ-PROB-008], [REF-ROSS-2019, Chapter 3].

    """
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    grid = [10.0 ** (-3.0 + 2.7 * k / 160) for k in range(161)]
    common = [_posterior_after_positive(x, 0.95, 0.05) for x in grid]
    better = [_posterior_after_positive(x, 0.95, 0.01) for x in grid]
    at_one = _posterior_after_positive(0.01, 0.95, 0.05)
    at_one_better = _posterior_after_positive(0.01, 0.95, 0.01)
    facts = {"posterior": f"{at_one:.0%}", "better": f"{at_one_better:.0%}"}
    theme.line(ax, grid, grid, color=palette.muted, dashes=theme.REFERENCE,
               width=theme.REFERENCE_WIDTH)
    theme.line(ax, grid, better, color=palette.muted, width=theme.CONTEXT_WIDTH)
    theme.line(ax, grid, common, color=theme.role_colour("model"))
    theme.points(ax, [0.01, 0.01], [at_one, at_one_better],
                 color=theme.role_colour("model"), size=40.0)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.1%", "1%", "10%", "50%"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _label(ax, 0.01, at_one, f"{facts['posterior']} at 1% prevalence",
           theme.role_colour("model"), offset=(8.0, -2.0))
    _label(ax, 0.01, at_one_better, facts["better"], palette.muted,
           offset=(-8.0, 0.0), ha="right")
    theme.direct_labels(
        ax,
        (
            (0.5, common[-1], "5% false positives", theme.role_colour("model")),
            (0.5, better[-1], "1% false positives", palette.muted),
        ),
    )
    _label(ax, 0.16, 0.16, "no test: posterior = prevalence", palette.muted,
           offset=(6.0, -8.0), va="top")
    theme.axis_labels(ax, "Prevalence of the condition",
                      "P(condition | positive test)")
    theme.title_block(
        ax,
        f"At 1% prevalence a positive result means only a {facts['posterior']} "
        "chance of the condition",
        "Bayes' rule for a test with 95% sensitivity; grey: 1% false positives",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# random-variables.md
# ---------------------------------------------------------------------------


def draw_dice_sum() -> tuple[Any, dict[str, str]]:
    """Induced PMF of the sum of two fair dice, a many-to-one mapping.

    References:
        [EQ-RV-001], [EQ-RV-002], [REF-ROSS-2019, Chapters 4-5].

    """
    pairs = FiniteSampleSpace(
        tuple((a, b) for a in range(1, 7) for b in range(1, 7))
    )
    fair = FiniteProbabilityMeasure.uniform(pairs)
    total = FiniteRandomVariable.from_function(
        fair, lambda pair: pair[0] + pair[1], name="Sum"
    )
    seven = total.preimage(7)
    facts = {
        "outcomes": str(len(pairs.outcomes)),
        "values": str(len(total.support)),
        "seven": str(len(seven.members)),
        "mass": f"{total.pmf(7):.3f}",
    }
    ax = plot_random_variable_pmf(total, ax=theme.new_axes(figsize=(7.2, 4.0)))
    subtitle = "Each mass is the probability of the outcomes mapped to that value"
    for text in list(ax.texts):
        if text.get_text() == subtitle:
            text.remove()
    ax.set_xlabel("Sum of the two faces")
    theme.title_block(
        ax,
        f"{facts['outcomes']} equally likely outcomes collapse onto "
        f"{facts['values']} sums; 7 collects {facts['seven']} of them "
        f"(mass {facts['mass']})",
        "Each mass is the probability of the dice pairs that add up to that sum",
    )
    return ax.figure, facts


def draw_zero_covariance() -> tuple[Any, dict[str, str]]:
    """Joint support of X and Y = X^2: zero covariance, full dependence.

    References:
        [EQ-RV-008], [EQ-RV-012], [EQ-RV-013], [REF-ROSS-2019, Chapter 6].

    """
    space = FiniteSampleSpace((-2, -1, 0, 1, 2))
    uniform = FiniteProbabilityMeasure.uniform(space)
    x = FiniteRandomVariable.from_function(uniform, lambda v: v, name="X")
    y = FiniteRandomVariable.from_function(uniform, lambda v: v * v, name="Y")
    joint = y.joint_with(x)
    covariance = x.joint_with(y).covariance
    given_two = joint.conditional_pmf_first_given_second(4, given_second=2)
    facts = {
        "covariance": f"{abs(covariance):.2f}",
        "marginal": f"{y.pmf(4):.1f}",
        "conditional": f"{given_two:.1f}",
    }
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    model = theme.role_colour("model")
    curve = [-2.3 + 4.6 * k / 200 for k in range(201)]
    theme.line(ax, curve, [v * v for v in curve], color=palette.muted,
               width=theme.CONTEXT_WIDTH, zorder=2)
    support = list(space.outcomes)
    support.sort()
    theme.points(ax, support, [v * v for v in support], entity="observations",
                 size=90.0)
    mean_y = y.mean
    slope = covariance / x.variance
    theme.line(ax, [-2.3, 2.3], [mean_y + slope * -2.3, mean_y + slope * 2.3],
               color=model, dashes=theme.FITTED)
    _label(ax, 2.3, mean_y, f"best linear fit: slope {abs(slope):.2f}", model,
           offset=(6.0, 0.0))
    for value in support:
        # Labels sit on the inner side of the parabola, away from the curve.
        side = 0.0 if value == 0 else (-1.0 if value > 0 else 1.0)
        ax.annotate(
            f"{joint.joint_pmf(value * value, value):.1f}",
            (value, value * value),
            xytext=(10.0 * side, 10.0 if value == 0 else 0.0),
            textcoords="offset points",
            ha={0.0: "center", 1.0: "left", -1.0: "right"}[side],
            va="bottom" if value == 0 else "center",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.ink,
        )
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.6, 5.2)
    ax.set_xticks(support)
    theme.axis_labels(ax, "X", "Y = X²")
    theme.title_block(
        ax,
        f"Y = X² depends completely on X, yet Cov(X, Y) = {facts['covariance']}",
        f"Dots: the joint law, labelled with their mass; P(Y = 4) = "
        f"{facts['marginal']} but P(Y = 4 | X = 2) = {facts['conditional']}",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# measure-theoretic-probability.md
# ---------------------------------------------------------------------------


def draw_conditional_expectation() -> tuple[Any, dict[str, str]]:
    """``E[X | G]`` of a die face for three nested sigma-algebras.

    References:
        [EQ-MEAS-001], [EQ-MEAS-008], [REF-BILLINGSLEY-1995, Section 34].

    """
    faces = tuple(range(1, 7))
    fair = FiniteMeasure.from_point_masses({face: Fraction(1, 6) for face in faces})
    value = {face: face for face in faces}
    mean = fair.integrate(value)

    def kept_variance(conditional: dict[int, Any]) -> float:
        """Return ``Var E[X | G]``, the variance the information keeps.

        References:
            [EQ-MEAS-004], [EQ-MEAS-008].

        """
        return float(fair.integrate({f: (conditional[f] - mean) ** 2 for f in faces}))

    algebras = (
        ("face known: 6 atoms", FiniteMeasurableSpace.power_set(faces)),
        ("parity and face <= 3: 4 atoms",
         FiniteMeasurableSpace(faces, [{2, 4, 6}, {1, 2, 3}])),
        ("parity only: 2 atoms",
         FiniteMeasurableSpace.generated_by_map(faces, lambda face: face % 2)),
    )
    conditionals = [dict(fair.conditional_expectation(value, g)) for _, g in algebras]
    kept = [kept_variance(c) for c in conditionals]
    facts = {
        "full": f"{kept[0]:.2f}",
        "info": f"{kept[1]:.2f}",
        "parity": f"{kept[2]:.2f}",
    }
    figure, axes = _axes(3, figsize=(7.6, 3.9), sharey=True)
    palette = theme.active_palette()
    model = theme.role_colour("model")
    for ax, (heading, _), conditional, share in zip(
        axes, algebras, conditionals, kept, strict=True
    ):
        levels = [float(conditional[f]) for f in faces]
        ax.vlines(faces, faces, levels, color=palette.muted,
                  linewidth=theme.PATH_WIDTH * 1.4, zorder=2)
        ax.hlines(levels, [f - 0.32 for f in faces], [f + 0.32 for f in faces],
                  color=model, linewidth=3.0, zorder=3)
        theme.points(ax, faces, faces, color=palette.muted, size=26.0)
        _heading(ax, f"{heading}\nVar E[X | G] = {share:.2f}")
        ax.set_xticks(faces)
        ax.set_xlim(0.4, 6.6)
        ax.set_ylim(0.0, 8.6)
        ax.set_yticks(range(0, 8, 2))
        theme.axis_labels(ax, "Face")
    theme.axis_labels(axes[0], y="Value")
    figure.subplots_adjust(wspace=0.12, left=0.08, right=0.98, top=0.78, bottom=0.14)
    theme.title_block(
        axes[0],
        f"Coarser information averages X over larger atoms: the variance kept "
        f"falls from {facts['full']} to {facts['info']} to {facts['parity']}",
        "Fair die; grey dots: the face value X; bars: E[X | G] on each atom of G",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# discrete-distributions.md
# ---------------------------------------------------------------------------


def _binomial_poisson_tv(n: int, lam: float) -> float:
    """Return the total variation between ``Binomial(n, lam/n)`` and Poisson.

    The last cell collects the Poisson mass beyond ``n``, where the binomial
    has none, so both vectors are complete PMFs.

    References:
        [EQ-DIST-BIN-001], [EQ-DIST-POIS-001], [EQ-DIV-TV-001].

    """
    binomial = Binomial(n=n, p=lam / n)
    poisson = Poisson(lam=lam)
    b = [binomial.pmf(k) for k in range(n + 1)] + [0.0]
    q = [poisson.pmf(k) for k in range(n + 1)] + [poisson.sf(n)]
    return total_variation_distance(b, q)


def draw_binomial_to_poisson() -> tuple[Any, dict[str, str]]:
    """Binomial(n, 4/n) masses converge to Poisson(4) in total variation.

    References:
        [EQ-DIST-BIN-001], [EQ-DIST-POIS-001], [EQ-DIV-TV-001],
        [REF-ROSS-2019, Section 4.7].

    """
    lam = 4.0
    sizes = (5, 10, 20, 50, 100, 200, 500, 1000)
    distances = [_binomial_poisson_tv(n, lam) for n in sizes]
    facts = {
        "tv10": f"{distances[1]:.3f}",
        "tv100": f"{distances[4]:.3f}",
    }
    figure, (left, right) = _axes(2, figsize=(7.6, 4.0))
    ks = list(range(0, 12))
    observed = theme.role_colour("observed")
    reference = theme.entity_colour("closed form")
    binomial = [Binomial(n=10, p=lam / 10).pmf(k) for k in ks]
    poisson = [Poisson(lam=lam).pmf(k) for k in ks]
    left.vlines([k - 0.14 for k in ks], 0.0, binomial, color=observed,
                linewidth=2.6, zorder=2)
    theme.points(left, [k - 0.14 for k in ks], binomial, color=observed, size=26.0)
    left.vlines([k + 0.14 for k in ks], 0.0, poisson, color=reference,
                linewidth=1.4, zorder=2)
    theme.points(left, [k + 0.14 for k in ks], poisson, entity="closed form",
                 size=26.0)
    left.set_ylim(0.0, 0.36)
    left.set_xlim(-0.7, 11.7)
    left.set_xticks(range(0, 12, 2))
    _heading(left, "Masses at each count")
    _label(left, 6.6, 0.215, "Binomial(10, 0.4)", observed)
    _label(left, 6.6, 0.185, "Poisson(4)", reference)
    theme.axis_labels(left, "Count k", theme.axis_title("probability_mass"))
    model = theme.role_colour("model")
    theme.line(right, sizes, distances, color=model)
    theme.points(right, sizes, distances, color=model, size=30.0)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(3.0, 2000.0)
    right.set_ylim(5e-4, 3.0)
    _log_ticks(right, 0, 3, axis="x")
    _log_ticks(right, -3, -1)
    right.set_ylim(5e-4, 3.0)
    for index in (1, 4):
        _label(right, sizes[index], distances[index],
               f"{distances[index]:.3f} at n = {sizes[index]}", model,
               offset=(6.0, 6.0), va="bottom")
    _heading(right, "Total variation to Poisson(4)")
    theme.axis_labels(right, "Number of trials n", "Total variation distance")
    figure.subplots_adjust(wspace=0.32, left=0.09, right=0.97, top=0.78, bottom=0.15)
    theme.title_block(
        left,
        f"Binomial(n, 4/n) approaches Poisson(4): the total variation\n"
        f"falls from {facts['tv10']} at n = 10 to {facts['tv100']} at n = 100",
        "Left: masses at n = 10; right: TV = sum |p - q| / 2 over all counts, "
        "against n",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# waiting-and-continuous-distributions.md
# ---------------------------------------------------------------------------


def draw_memorylessness() -> tuple[Any, dict[str, str]]:
    """Conditional survival after waiting ``s``: exponential against Weibull.

    References:
        [EQ-DIST-EXP-001], [EQ-DIST-WEIB-004], [REF-KLEIN-MOESCHBERGER-2003].

    """
    exponential = Exponential(rate=1.0)
    weibull = Weibull(2.0, 1.0 / math.gamma(1.5))
    ages = (0.0, 1.0, 2.0)
    grid = [3.0 * k / 150 for k in range(151)]

    def conditional(law: Any, s: float, t: float) -> float:
        """Return ``P(X > s + t | X > s)`` from the direct survival function.

        References:
            [EQ-DIST-EXP-001].

        """
        return float(law.sf(s + t) / law.sf(s))

    exp_one = [conditional(exponential, s, 1.0) for s in ages]
    weib_one = [conditional(weibull, s, 1.0) for s in ages]
    facts = {
        "exponential": f"{exp_one[0]:.3f}",
        "weibull0": f"{weib_one[0]:.3f}",
        "weibull2": f"{weib_one[2]:.3f}",
    }
    figure, axes = _axes(2, figsize=(7.6, 4.0), sharey=True)
    shades = theme.ordered_colours(3)
    for ax, law, name in ((axes[0], exponential, "Exponential, constant hazard"),
                          (axes[1], weibull, "Weibull shape 2, rising hazard")):
        for colour, s in zip(shades, ages, strict=True):
            theme.line(ax, grid, [conditional(law, s, t) for t in grid],
                       color=colour)
        values = [conditional(law, s, 1.0) for s in ages]
        for colour, value in zip(shades, values, strict=True):
            theme.points(ax, [1.0], [value], color=colour, size=30.0, zorder=5)
        _heading(ax, name)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 1.3)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        theme.axis_labels(ax, "Additional waiting time t")

    def key(ax: Any, row: int, text: str, colour: str) -> None:
        """Write one line of a colour key in the empty upper right.

        References:
            [REF-MATPLOTLIB].

        """
        ax.text(0.97, 0.72 - 0.075 * row, text, transform=ax.transAxes,
                ha="right", va="center", fontsize=theme.ANNOTATION_SIZE,
                fontweight="bold", color=_legible(colour))

    key(axes[0], 0, "at t = 1, after waiting", theme.active_palette().ink)
    key(axes[0], 1, f"s = 0, 1, 2: all {facts['exponential']}", shades[-1])
    key(axes[1], 0, "at t = 1, after waiting", theme.active_palette().ink)
    for row, (colour, s, value) in enumerate(zip(shades, ages, weib_one,
                                                 strict=True), start=1):
        key(axes[1], row, f"s = {s:g}: {value:.3f}", colour)
    theme.axis_labels(axes[0], y="P(X > s + t | X > s)")
    figure.subplots_adjust(wspace=0.1, left=0.1, right=0.97, top=0.78, bottom=0.15)
    theme.title_block(
        axes[0],
        f"Only the exponential forgets elapsed time: P(X > s + 1 | X > s)\n"
        f"stays {facts['exponential']} for every s, while the Weibull's falls "
        f"from {facts['weibull0']} to {facts['weibull2']}",
        "Survival after already waiting s = 0, 1, 2 (faint to strong shade); "
        "both laws have mean 1",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# continuous-distribution-catalog.md
# ---------------------------------------------------------------------------


def draw_tails() -> tuple[Any, dict[str, str]]:
    """Upper tails of four laws with mean 0 and variance 1 on a log scale.

    References:
        [EQ-DIST-INV-001], [REF-JKB-1994], [REF-JKB-1995].

    """
    root6 = math.sqrt(6.0) / math.pi
    laws = (
        ("Normal", Normal(0.0, 1.0), "context"),
        ("Logistic", Logistic(0.0, math.sqrt(3.0) / math.pi), "series-6"),
        ("Student t, 5 df", StudentT(5.0, 0.0, math.sqrt(3.0 / 5.0)), "series-5"),
        ("Gumbel", Gumbel(-EULER_GAMMA * root6, root6), "series-4"),
    )
    at_five = {name: float(law.sf(5.0)) for name, law, _ in laws}
    ratio = at_five["Gumbel"] / at_five["Normal"]
    facts = {
        "normal": _sci(at_five["Normal"]),
        "gumbel": _sci(at_five["Gumbel"]),
        "ratio": f"{ratio:,.0f}",
    }
    figure, (ax,) = _axes()
    grid = [6.0 * k / 240 for k in range(241)]
    labels = []
    for name, law, role in laws:
        colour = theme.role_colour(role)
        theme.line(ax, grid, [float(law.sf(x)) for x in grid], color=colour)
        theme.points(ax, [5.0], [at_five[name]], color=colour, size=34.0, zorder=5)
        labels.append((6.0, float(law.sf(6.0)), name, colour))
    ax.set_yscale("log")
    ax.set_xlim(0.0, 6.0)
    _log_ticks(ax, -10, 0)
    ax.set_yticks([10.0**k for k in range(-10, 1, 2)])
    ax.set_yticklabels([f"$10^{{{k}}}$" if k else "1" for k in range(-10, 1, 2)])
    ax.set_ylim(1e-10, 1.0)
    ax.axvline(5.0, color=theme.active_palette().rule, linewidth=1.0, zorder=1)
    theme.direct_labels(ax, labels)
    theme.axis_labels(ax, "x", "P(X > x)")
    theme.title_block(
        ax,
        f"With equal mean and variance, P(X > 5) still ranges from "
        f"{facts['normal']} (normal) to {facts['gumbel']} (Gumbel), "
        f"a factor of {facts['ratio']}",
        "Survival functions sf(x) of four laws with mean 0 and variance 1; "
        "dots at x = 5",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# multivariate-laws.md
# ---------------------------------------------------------------------------

_COUNTS = (12, 5, 3)


def draw_dirichlet_update() -> tuple[Any, dict[str, str]]:
    """Prior and posterior marginals of a Dirichlet-categorical update.

    References:
        [EQ-CONJ-DIR-001], [EQ-CONJ-DIR-002], [EQ-DIST-DIR-005],
        [REF-GELMAN-ETAL-2013, Section 3.4].

    """
    prior = Dirichlet((1.0, 1.0, 1.0))
    result = dirichlet_posterior(prior, _COUNTS)
    shares = result.predictive_probabilities
    facts = {f"p{i + 1}": f"{p:.3f}" for i, p in enumerate(shares)}
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    grid = [k / 400 for k in range(1, 400)]
    flat = prior.marginal(0)
    theme.line(ax, grid, [flat.pdf(x) for x in grid], color=palette.muted,
               dashes=theme.FITTED, width=theme.CONTEXT_WIDTH)
    roles = ("series-4", "series-5", "series-6")
    names = ("category 1", "category 2", "category 3")
    for index, (role, name, count) in enumerate(zip(roles, names, _COUNTS,
                                                     strict=True)):
        colour = theme.role_colour(role)
        marginal = result.posterior.marginal(index)
        density = [marginal.pdf(x) for x in grid]
        theme.line(ax, grid, density, color=colour)
        share = shares[index]
        theme.points(ax, [share], [marginal.pdf(share)], color=colour, size=34.0)
        peak = max(density)
        _label(ax, grid[density.index(peak)], peak,
               f"{name}: {count} of 20, mean {share:.3f}", colour,
               offset=(8.0, 4.0), va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 6.4)
    theme.axis_labels(ax, "Category probability", theme.axis_title("density"))
    theme.title_block(
        ax,
        f"Counts (12, 5, 3) move a flat Dirichlet prior to posterior shares "
        f"{facts['p1']}, {facts['p2']} and {facts['p3']}",
        "Posterior marginal densities, dots at the predictive probability; "
        "dashed grey: the prior marginal Beta(1, 2)",
    )
    return figure, facts


def draw_dirichlet_multinomial() -> tuple[Any, dict[str, str]]:
    """Posterior predictive count of category 1 against the plug-in binomial.

    References:
        [EQ-CONJ-DIR-002], [EQ-DIST-DIRMULT-002], [EQ-DIST-MULT-003],
        [REF-MOSIMANN-1962].

    """
    result = dirichlet_posterior(Dirichlet((1.0, 1.0, 1.0)), _COUNTS)
    future = 10
    predictive = result.predictive(future)
    share = result.predictive_probabilities[0]
    plug_in = Binomial(n=future, p=share)
    ks = list(range(future + 1))
    mixed = [
        math.fsum(
            predictive.pmf((k, j, future - k - j)) for j in range(future - k + 1)
        )
        for k in ks
    ]
    binomial = [plug_in.pmf(k) for k in ks]

    def variance(masses: list[float]) -> float:
        """Return the variance of a PMF on ``0..future``.

        References:
            [EQ-RV-007].

        """
        mean = math.fsum(k * m for k, m in zip(ks, masses, strict=True))
        return math.fsum((k - mean) ** 2 * m for k, m in zip(ks, masses, strict=True))

    v_mixed, v_binomial = variance(mixed), variance(binomial)
    facts = {
        "mixed": f"{v_mixed:.2f}",
        "binomial": f"{v_binomial:.2f}",
        "ratio": f"{v_mixed / v_binomial:.3f}",
    }
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    model = theme.role_colour("model")
    ax.vlines([k - 0.15 for k in ks], 0.0, mixed, color=model, linewidth=2.4,
              zorder=3)
    theme.points(ax, [k - 0.15 for k in ks], mixed, color=model, size=26.0)
    ax.vlines([k + 0.15 for k in ks], 0.0, binomial, color=palette.muted,
              linewidth=2.4, zorder=2)
    theme.points(ax, [k + 0.15 for k in ks], binomial, color=palette.muted,
                 size=26.0)
    ax.set_xticks(ks)
    ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(0.0, max(binomial) * 1.3)
    # The upper left is empty: every mass below a count of 4 is under 0.1.
    _label(ax, -0.5, 0.27, f"posterior predictive, variance {facts['mixed']}",
           model, offset=(0.0, 0.0))
    _label(ax, -0.5, 0.245,
           f"plug-in Binomial(10, {share:.3f}), variance {facts['binomial']}",
           palette.muted, offset=(0.0, 0.0))
    theme.axis_labels(ax, "Category 1 count in the next 10 trials",
                      theme.axis_title("probability_mass"))
    theme.title_block(
        ax,
        f"Averaging over the posterior widens the next-10 count: variance "
        f"{facts['mixed']} against {facts['binomial']}, a factor of "
        f"{facts['ratio']}",
        "Dirichlet-multinomial predictive (colour) and the multinomial with the "
        "posterior mean plugged in (grey)",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# limit-theorems-and-estimators.md
# ---------------------------------------------------------------------------


def draw_clt_rates() -> tuple[Any, dict[str, str]]:
    """Exact Kolmogorov distances of exponential sums against Berry-Esseen.

    References:
        [EQ-LIM-001], [EQ-LIM-003], [EQ-LIM-004], [REF-SHEVTSOVA-2011],
        [REF-HALL-1992, Chapter 2].

    """
    sizes = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000)
    rows = clt_error_profile(exponential_summand(1.0), sizes)
    at = {row.n: row for row in rows}
    edgeworth_100 = at[100].kolmogorov_edgeworth
    assert edgeworth_100 is not None
    facts = {
        "normal": f"{at[100].kolmogorov_normal:.3g}",
        "edgeworth": f"{edgeworth_100:.3g}",
        "bound": f"{at[100].berry_esseen:.3g}",
    }
    figure, (ax,) = _axes()
    normal = [row.kolmogorov_normal for row in rows]
    edgeworth = [float(row.kolmogorov_edgeworth or 0.0) for row in rows]
    bound = [row.berry_esseen for row in rows]
    observed, model = theme.role_colour("observed"), theme.role_colour("model")
    theme.line(ax, sizes, bound, entity="Berry-Esseen bound")
    theme.line(ax, sizes, normal, color=observed)
    theme.points(ax, sizes, normal, color=observed, size=24.0)
    theme.line(ax, sizes[1:], edgeworth[1:], color=model)
    theme.points(ax, sizes[1:], edgeworth[1:], color=model, marker="s", size=22.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1000.0)
    ax.set_ylim(3e-5, 2.0)
    _log_ticks(ax, 0, 3, axis="x")
    _log_ticks(ax, -4, 0)
    theme.direct_labels(
        ax,
        (
            (1000.0, bound[-1], "Berry-Esseen bound", theme.entity_colour(
                "Berry-Esseen bound")),
            (1000.0, normal[-1], "sup |F_n - Phi|", observed),
            (1000.0, edgeworth[-1], "sup |F_n - Edgeworth|", model),
        ),
    )
    bound_colour = theme.entity_colour("Berry-Esseen bound")
    theme.points(ax, [100.0], [at[100].berry_esseen], color=bound_colour, size=26.0)
    for value, colour in ((at[100].kolmogorov_normal, observed),
                          (edgeworth_100, model),
                          (at[100].berry_esseen, bound_colour)):
        _label(ax, 100.0, value, f"n = 100: {value:.3g}", colour,
               offset=(-6.0, -8.0), ha="right", va="top")
    theme.axis_labels(ax, "Number of summands n", theme.axis_title(
        "kolmogorov_distance"))
    theme.title_block(
        ax,
        f"The normal error of exponential sums falls like 1/sqrt(n), the Edgeworth "
        f"error like 1/n: {facts['normal']} and {facts['edgeworth']} at n = 100, "
        f"under a bound of {facts['bound']}",
        "Exact Kolmogorov distance of the standardized mean of Exponential(1) draws",
    )
    return figure, facts


def draw_tail_bounds() -> tuple[Any, dict[str, str]]:
    """Exact sample-mean tail of Bernoulli(0.3) data against three bounds.

    References:
        [EQ-LIM-005], [EQ-LIM-006], [EQ-LIM-007], [REF-HOEFFDING-1963].

    """
    summand = bernoulli_summand(0.3)
    n = 100
    ts = [0.005 + 0.001 * k for k in range(246)]
    rows = [compare_tail_bounds(summand, n, t) for t in ts]
    marked = compare_tail_bounds(summand, n, 0.2)
    assert marked.hoeffding is not None
    facts = {
        "exact": _sci(marked.exact_upper),
        "hoeffding": f"{marked.hoeffding / marked.exact_upper:.0f}",
        "chebyshev": f"{marked.chebyshev / marked.exact_upper:,.0f}",
    }
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    series = (
        ("Chebyshev", [r.chebyshev for r in rows], palette.muted),
        ("Bernstein", [float(r.bernstein or 1.0) for r in rows],
         theme.role_colour("series-3")),
        ("Hoeffding", [float(r.hoeffding or 1.0) for r in rows],
         theme.role_colour("model")),
    )
    labels = []
    for name, values, colour in series:
        theme.line(ax, ts, values, color=colour)
        labels.append((ts[-1], values[-1], name, colour))
    exact = [r.exact_upper for r in rows]
    theme.line(ax, ts, exact, color=theme.role_colour("observed"))
    labels.append((ts[-1], exact[-1], "exact P(mean - 0.3 >= t)",
                   theme.role_colour("observed")))
    theme.points(ax, [0.2, 0.2, 0.2],
                 [marked.exact_upper, marked.hoeffding, marked.chebyshev],
                 color=palette.ink, size=24.0, zorder=6)
    ax.axvline(0.2, color=palette.rule, linewidth=1.0, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.25)
    _log_ticks(ax, -8, 0)
    ax.set_ylim(1e-8, 1.5)
    theme.direct_labels(ax, labels)
    theme.axis_labels(ax, "Deviation t of the sample mean")
    ax.set_ylabel("Tail probability")
    theme.title_block(
        ax,
        f"At t = 0.2 the exact tail is {facts['exact']}: Hoeffding overstates it "
        f"{facts['hoeffding']}-fold and Chebyshev {facts['chebyshev']}-fold",
        "Mean of n = 100 Bernoulli(0.3) draws; Chebyshev bounds the two-sided "
        "tail, hence this one too",
    )
    return figure, facts


def draw_variance_divisor() -> tuple[Any, dict[str, str]]:
    """Exact bias, variance, and MSE of ``SS/d`` against the divisor.

    References:
        [EQ-EST-001], [EQ-EST-005], [REF-CASELLA-BERGER-2002, Section 7.3.1].

    """
    n = 10
    best = minimum_mse_variance_divisor(n)
    grid = [6.0 + 11.0 * k / 220 for k in range(221)]
    risks = [normal_variance_estimator_risk(1.0, n, d) for d in grid]
    unbiased = normal_variance_estimator_risk(1.0, n, n - 1)
    optimal = normal_variance_estimator_risk(1.0, n, best)
    facts = {
        "best": str(best),
        "unbiased": f"{unbiased.mse:.3f}",
        "optimal": f"{optimal.mse:.3f}",
    }
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    model = theme.role_colour("model")
    squared_bias = [r.bias * r.bias for r in risks]
    variances = [r.variance for r in risks]
    mses = [r.mse for r in risks]
    theme.line(ax, grid, squared_bias, color=theme.role_colour("series-3"),
               width=theme.CONTEXT_WIDTH)
    theme.line(ax, grid, variances, color=palette.muted, width=theme.CONTEXT_WIDTH)
    theme.line(ax, grid, mses, color=model)
    theme.points(ax, [n - 1, best], [unbiased.mse, optimal.mse], color=model,
                 size=40.0, zorder=5)
    _label(ax, n - 1, unbiased.mse, f"d = n - 1: {facts['unbiased']}", model,
           offset=(-6.0, -6.0), ha="right", va="top")
    _label(ax, best, optimal.mse, f"d = n + 1: {facts['optimal']}", model,
           offset=(0.0, 12.0), ha="center", va="bottom")
    theme.direct_labels(
        ax,
        (
            (grid[-1], mses[-1], "MSE", model),
            (grid[-1], variances[-1], "variance", palette.muted),
            (grid[-1], squared_bias[-1], "squared bias",
             theme.role_colour("series-3")),
        ),
    )
    ax.set_xlim(6.0, 17.0)
    ax.set_ylim(0.0, 0.8)
    theme.axis_labels(ax, "Divisor d of the sum of squares SS", "Risk")
    theme.title_block(
        ax,
        f"Dividing by n + 1 = {facts['best']} instead of n - 1 accepts bias and "
        f"cuts the MSE of the variance estimate from {facts['unbiased']} to "
        f"{facts['optimal']}",
        "Exact risk of SS/d for n = 10 normal observations with variance 1",
    )
    return figure, facts


# ---------------------------------------------------------------------------
# information-theory.md
# ---------------------------------------------------------------------------

_TABLE = ((0.30, 0.10, 0.05), (0.05, 0.20, 0.30))


def draw_entropy_decomposition() -> tuple[Any, dict[str, str]]:
    """Information diagram of the guide's worked joint table, in bits.

    References:
        [EQ-INFO-H-001], [EQ-INFO-H-002], [EQ-INFO-MI-001],
        [REF-COVER-THOMAS-2006, Chapter 2].

    """
    px, _ = marginals(_TABLE)
    info = mutual_information(_TABLE, unit="bits")
    share = info / shannon_entropy(px, unit="bits")
    facts = {"share": f"{share:.0%}", "information": f"{info:.3f}"}
    ax = plot_entropy_decomposition(_TABLE, unit="bits")
    return ax.figure, facts


def draw_kl_asymmetry() -> tuple[Any, dict[str, str]]:
    """KL divergence in both directions and Jensen-Shannon, binary laws.

    References:
        [EQ-INFO-KL-001], [EQ-INFO-JS-001], [REF-COVER-THOMAS-2006,
        Section 2.3].

    """
    p = (0.5, 0.5)
    grid = [0.5 + 0.48 * k / 120 for k in range(121)]
    forward = [kl_divergence(p, (q, 1.0 - q)) for q in grid]
    backward = [kl_divergence((q, 1.0 - q), p) for q in grid]
    js = [jensen_shannon_divergence(p, (q, 1.0 - q)) for q in grid]
    marked = (0.9, 0.1)
    facts = {
        "forward": f"{kl_divergence(p, marked):.3f}",
        "backward": f"{kl_divergence(marked, p):.3f}",
        "js": f"{jensen_shannon_divergence(p, marked):.3f}",
    }
    figure, (ax,) = _axes()
    palette = theme.active_palette()
    model, third = theme.role_colour("model"), theme.role_colour("series-3")
    curves = (
        (f"D(p || q), {facts['forward']} at 0.9", forward, model, theme.LINE_WIDTH),
        (f"D(q || p), {facts['backward']} at 0.9", backward, third,
         theme.LINE_WIDTH),
        (f"Jensen-Shannon, {facts['js']} at 0.9", js, palette.muted,
         theme.CONTEXT_WIDTH),
    )
    for _, values, colour, width in curves:
        theme.line(ax, grid, values, color=colour, width=width)
    ax.axvline(0.9, color=palette.rule, linewidth=1.0, zorder=1)
    for key, colour in (("forward", model), ("backward", third),
                        ("js", palette.muted)):
        theme.points(ax, [0.9], [float(facts[key])], color=colour, size=30.0,
                     zorder=5)
    theme.direct_labels(
        ax, [(grid[-1], values[-1], name, colour) for name, values, colour, _ in
             curves]
    )
    ax.set_xlim(0.5, 0.98)
    ax.set_ylim(0.0, 1.4)
    theme.axis_labels(ax, "Mass q puts on the first outcome", "Divergence",
                      y_unit="nats")
    theme.title_block(
        ax,
        f"KL divergence is not symmetric: for q = (0.9, 0.1), D(p || q) = "
        f"{facts['forward']} but D(q || p) = {facts['backward']} nats",
        "p = (0.5, 0.5) against q = (q, 1 - q); Jensen-Shannon compares both "
        "with their average",
    )
    return figure, facts


FIGURES = [
    GuideFigure(
        name="probability-base-rate-posterior",
        guide="probability",
        alt="Line chart of the posterior probability of a condition after a "
        "positive test against its prevalence on a log axis, for two tests",
        draw=draw_base_rate,
    ),
    GuideFigure(
        name="random-variables-two-dice-sum",
        guide="random-variables",
        alt="Lollipop chart of the probability mass of the sum of two fair dice "
        "at each value from 2 to 12",
        draw=draw_dice_sum,
    ),
    GuideFigure(
        name="random-variables-zero-covariance",
        guide="random-variables",
        alt="Scatter of the five support points of X and Y equal to X squared, "
        "with the parabola and the flat best linear fit",
        draw=draw_zero_covariance,
    ),
    GuideFigure(
        name="measure-conditional-expectation-atoms",
        guide="measure-theoretic-probability",
        alt="Three panels of die faces with the conditional expectation of the "
        "face given three nested sigma-algebras drawn as bars",
        draw=draw_conditional_expectation,
    ),
    GuideFigure(
        name="discrete-binomial-to-poisson",
        guide="discrete-distributions",
        alt="Left: binomial and Poisson masses at counts 0 to 11; right: total "
        "variation distance against the number of trials on log axes",
        draw=draw_binomial_to_poisson,
    ),
    GuideFigure(
        name="waiting-memorylessness",
        guide="waiting-and-continuous-distributions",
        alt="Two panels of conditional survival curves after waiting 0, 1, and "
        "2 time units, for an exponential and a Weibull law",
        draw=draw_memorylessness,
    ),
    GuideFigure(
        name="catalog-tails-equal-variance",
        guide="continuous-distribution-catalog",
        alt="Log-scale survival functions of normal, logistic, Student t, and "
        "Gumbel laws with mean 0 and variance 1 from 0 to 6",
        draw=draw_tails,
    ),
    GuideFigure(
        name="multivariate-dirichlet-update",
        guide="multivariate-laws",
        alt="Densities of the flat prior marginal and of three posterior "
        "category marginals after a Dirichlet update",
        draw=draw_dirichlet_update,
    ),
    GuideFigure(
        name="multivariate-dirichlet-multinomial-spread",
        guide="multivariate-laws",
        alt="Paired stems of two probability mass functions of a count from 0 "
        "to 10: predictive and plug-in binomial",
        draw=draw_dirichlet_multinomial,
    ),
    GuideFigure(
        name="limit-clt-error-rates",
        guide="limit-theorems-and-estimators",
        alt="Log-log lines of the normal and Edgeworth Kolmogorov errors and "
        "the Berry-Esseen bound against the number of summands",
        draw=draw_clt_rates,
    ),
    GuideFigure(
        name="limit-tail-bounds",
        guide="limit-theorems-and-estimators",
        alt="Log-scale exact upper tail of a sample mean and the Chebyshev, "
        "Bernstein, and Hoeffding bounds against the deviation",
        draw=draw_tail_bounds,
    ),
    GuideFigure(
        name="limit-variance-divisor-risk",
        guide="limit-theorems-and-estimators",
        alt="Curves of squared bias, variance, and mean squared error of a "
        "variance estimator against its divisor",
        draw=draw_variance_divisor,
    ),
    GuideFigure(
        name="information-entropy-decomposition",
        guide="information-theory",
        alt="Three aligned bars splitting H(X), H(Y), and H(X, Y) of a joint "
        "table into conditional entropies and mutual information",
        draw=draw_entropy_decomposition,
    ),
    GuideFigure(
        name="information-kl-asymmetry",
        guide="information-theory",
        alt="Curves of the two KL divergences and the Jensen-Shannon divergence "
        "between a fair coin and a coin with bias q",
        draw=draw_kl_asymmetry,
    ),
]
