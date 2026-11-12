"""Figures of the statistical-inference guides (ADR-0105).

One module for the nine guides on descriptive statistics, tests and the
bootstrap, frequentist and Bayesian interpretation, regression and GLMs,
count and categorical regression, robust and quantile regression, smoothing
and additive models, multivariate analysis, and mixed models. Every figure
runs a seeded computation of ``probstat`` and returns the strings its
caption quotes.

Figure brief (message; chart type; why):

- ``descriptive-statistics-skewed-ecdf``: in a right-skewed sample the mean
  lies right of the median; empirical CDF with mean and median rules; the
  ECDF reads both centres and the share below the mean on one scale.
- ``hypothesis-testing-interval-coverage``: the t interval covers the mean
  about 95 times in 100, the z interval with ``s`` fewer; two panels of 100
  replicate intervals; misses are counted by eye.
- ``hypothesis-testing-bootstrap-bca``: BCa shifts a skewed statistic's
  interval; replicate histogram over interval bars; the shift is read
  against the bootstrap distribution itself.
- ``frequentist-bayesian-exact-coverage``: only Clopper-Pearson keeps 95%
  coverage for every ``p``; three small multiples of exact coverage; the
  saw-tooth dips are the message and need their own panel each.
- ``frequentist-bayesian-posterior-intervals``: the credible and Wilson
  intervals nearly coincide at ``n = 20``; posterior density over interval
  bars; same scale for the two kinds of interval.
- ``regression-logistic-band``: the Wald band is asymmetric on the
  probability scale; fitted curve with band and 0/1 data; the inverse link
  is the point.
- ``regression-lasso-path``: cross-validation keeps a correlated null
  predictor; coefficient paths on a log penalty axis; standard lasso view.
- ``count-regression-zero-frequencies``: Poisson misses the zeros, the
  zero-inflated model does not; observed frequency bars with expected
  points; frequencies per count are compared at the same positions.
- ``count-regression-ordinal-probabilities``: one odds ratio shifts all
  three category curves; two panels of fitted probabilities with observed
  shares; proportional odds is visible as parallel shifts.
- ``robust-regression-leverage``: five far points capture least squares
  and Huber, not Theil-Sen; scatter with three lines; leverage is spatial.
- ``quantile-regression-fan``: regression quantiles fan out under growing
  spread; scatter with five quantile lines; the fan is the finding.
- ``smoothing-pspline-reml``: REML sits between a wiggle and a line; three
  panels of fits at three penalties; equal axes make roughness comparable.
- ``smoothing-additive-terms``: each additive term recovers its own shape;
  two panels of partial effects with partial residuals; one shape per term.
- ``multivariate-pca-scaling``: the unit of a variable decides its PCA
  weight; dot plot of first-axis weights; few values, exact positions.
- ``multivariate-lda-regions``: LDA splits the plane with straight lines;
  decision regions with training points; a classifier is a partition.
- ``mixed-models-blup-shrinkage``: small groups are shrunk more; dumbbell
  of raw deviations and BLUPs by group size; the arrow length is shrinkage.
- ``mixed-models-growth-lines``: subjects scatter around the population
  line; subject lines with data and the fixed-effect line; random slopes
  are seen as a spread of slopes.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from probstat.distributions import Beta, Binomial, Exponential, LogNormal, Poisson
from probstat.foundations import resolve_stream
from probstat.inference import (
    bootstrap_confidence_interval,
    clopper_pearson_interval,
    cross_validate_elastic_net,
    elastic_net_path,
    fit_additive_model,
    fit_glm,
    fit_linear_discriminant,
    fit_linear_mixed_model,
    fit_negative_binomial_regression,
    fit_ordinal_logistic,
    fit_poisson_regression,
    fit_pspline,
    fit_quantile_regression,
    fit_robust_linear_model,
    fit_simple_linear_regression,
    fit_zero_inflated,
    normal_mean_confidence_interval,
    principal_components,
    random_effect,
    student_t_mean_confidence_interval,
    theil_sen_slope,
    wilson_score_interval,
)
from probstat.statistics import empirical_cdf
from probstat.statistics import mean as sample_mean
from probstat.statistics import median as sample_median
from probstat.visualization import theme
from probstat.visualization.accessibility import legible_colour

from . import GuideFigure

FIGSIZE = (7.2, 4.0)


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------


def _figure(
    panels: int = 1,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    sharey: bool = False,
    sharex: bool = False,
    rows: int = 1,
    height_ratios: Sequence[float] | None = None,
) -> tuple[Any, list[Any]]:
    plt = theme.pyplot()
    gridspec: dict[str, Any] = {}
    if height_ratios is not None:
        gridspec["height_ratios"] = list(height_ratios)
    figure, axes = plt.subplots(
        rows,
        panels,
        figsize=figsize,
        sharey=sharey,
        sharex=sharex,
        squeeze=False,
        gridspec_kw=gridspec or None,
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        theme.style_axes(ax)
    return figure, flat


def _halo() -> list[Any]:
    from matplotlib import patheffects

    return [
        patheffects.withStroke(linewidth=3.0, foreground=theme.active_palette().canvas)
    ]


def _text(
    ax: Any,
    x: float,
    y: float,
    text: str,
    colour: str,
    *,
    ha: str = "left",
    va: str = "center",
    offset: tuple[float, float] = (0.0, 0.0),
    coords: str | tuple[str, str] = "data",
    bold: bool = True,
) -> Any:
    return ax.annotate(
        text,
        xy=(x, y),
        xycoords=coords,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="semibold" if bold else "normal",
        color=_ink(colour),
        path_effects=_halo(),
        annotation_clip=False,
        zorder=8,
    )


def _ink(colour: str) -> str:
    """Return a series colour darkened toward the ink until it reads as text."""
    from matplotlib.colors import to_hex

    palette = theme.active_palette()
    return legible_colour(to_hex(colour).upper(), palette.canvas, palette.ink)


def _title(ax: Any, title: str, subtitle: str) -> None:
    """Draw the title block and keep it out of the subplot layout."""
    theme.title_block(ax, title, subtitle)
    for artist in theme.title_artists(ax):
        artist.set_in_layout(False)


def _finish(figure: Any, wspace: float | None = None) -> None:
    """Lay out the axes below a title block of about 0.7 inch.

    Side-by-side panels get fixed margins and the given ``wspace``; a single
    axes or a stacked pair is laid out tightly.
    """
    height = float(figure.get_size_inches()[1])
    top = 1.0 - 0.72 / height
    if wspace is None:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, top))
    else:
        figure.subplots_adjust(
            left=0.09, right=0.975, bottom=0.15, top=top, wspace=wspace
        )


def _panel_header(ax: Any, text: str) -> None:
    _text(
        ax,
        0.0,
        1.0,
        text,
        theme.active_palette().ink,
        va="top",
        offset=(4.0, -2.0),
        coords="axes fraction",
    )


def _interval_rows(
    ax: Any,
    rows: Sequence[tuple[str, float, float, str]],
    *,
    decimals: int = 2,
    brackets: str = "()",
) -> None:
    """Draw labelled horizontal interval bars, one per row, top to bottom."""
    palette = theme.active_palette()
    for index, (name, lower, upper, colour) in enumerate(rows):
        y = len(rows) - 1 - index
        ax.hlines(y, lower, upper, color=colour, linewidth=4.0, zorder=3)
        ax.plot([lower, upper], [y, y], "|", color=colour, markersize=11, zorder=3)
        bounds = f"{brackets[0]}{lower:.{decimals}f}, {upper:.{decimals}f}{brackets[1]}"
        _text(ax, upper, y, f"{name} {bounds}", colour, offset=(8, 0))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["bottom"].set_color(palette.muted)


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------


def _skewed_ecdf() -> tuple[Any, dict[str, str]]:
    law = LogNormal(0.0, 0.8)
    sample = sorted(law.sample(200, seed=2026))
    centre = sample_mean(sample)
    middle = sample_median(sample)
    below = empirical_cdf(sample, centre)
    facts = {
        "mean": f"{centre:.2f}",
        "median": f"{middle:.2f}",
        "below": f"{below:.0%}",
    }
    figure, (ax,) = _figure()
    heights = [(index + 1) / len(sample) for index in range(len(sample))]
    xs = [0.0, *sample, sample[-1] * 1.03]
    ys = [0.0, *heights, 1.0]
    ax.step(xs, ys, where="post", color=theme.entity_colour("empirical"), linewidth=2.0)
    grid = [sample[-1] * 1.03 * k / 300 for k in range(1, 301)]
    theme.line(ax, grid, [law.cdf(v) for v in grid], entity="closed form")
    model = theme.role_colour("model")
    context = theme.role_colour("context")
    ax.axvline(centre, color=model, linewidth=theme.REFERENCE_WIDTH, zorder=2)
    ax.axvline(middle, color=context, linewidth=theme.REFERENCE_WIDTH, zorder=2)
    ax.axhline(0.5, color=context, linewidth=0.8, linestyle=theme.GUIDE, zorder=2)
    theme.points(ax, [centre], [below], color=model, size=40.0)
    theme.points(ax, [middle], [0.5], color=context, size=40.0)
    _text(
        ax,
        centre,
        below,
        f"mean {facts['mean']}: {facts['below']} below",
        model,
        offset=(12, -4),
        va="top",
    )
    _text(
        ax, middle, 0.5, f"median {facts['median']}", context, offset=(10, -8), va="top"
    )
    _text(
        ax,
        sample[-1],
        1.0,
        "empirical CDF",
        theme.entity_colour("empirical"),
        ha="right",
        offset=(0, -14),
        va="top",
    )
    _text(
        ax,
        4.2,
        law.cdf(4.2),
        "LogNormal(0, 0.8) CDF",
        theme.entity_colour("closed form"),
        offset=(6, -6),
        va="top",
        bold=False,
    )
    ax.set_xlim(0.0, sample[-1] * 1.03)
    ax.set_ylim(0.0, 1.04)
    theme.axis_labels(ax, "Observed value", theme.axis_title("cumulative_probability"))
    _title(
        ax,
        f"The mean {facts['mean']} sits right of the median {facts['median']}; "
        f"{facts['below']} of the sample lies below the mean",
        "Steps: empirical CDF of 200 seeded LogNormal(0, 0.8) draws. "
        "Dashed: the law's CDF",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Hypothesis testing and the bootstrap
# --------------------------------------------------------------------------


def _interval_coverage() -> tuple[Any, dict[str, str]]:
    generator = resolve_stream(97).generator()
    samples = [[generator.gauss(0.0, 1.0) for _ in range(6)] for _ in range(100)]
    rules = (
        ("Student-t interval", student_t_mean_confidence_interval),
        ("z interval with s", normal_mean_confidence_interval),
    )
    figure, axes = _figure(2, sharey=True)
    hit_colour = theme.role_colour("context")
    miss_colour = theme.role_colour("model")
    counts: list[int] = []
    for ax, (name, rule) in zip(axes, rules, strict=True):
        intervals = [rule(sample) for sample in samples]
        covered = [iv.lower <= 0.0 <= iv.upper for iv in intervals]
        counts.append(sum(covered))
        for index, (iv, hit) in enumerate(zip(intervals, covered, strict=True)):
            ax.vlines(
                index + 1,
                iv.lower,
                iv.upper,
                color=hit_colour if hit else miss_colour,
                linewidth=1.0 if hit else 2.0,
                zorder=3 if hit else 4,
            )
        theme.reference_line(ax, y=0.0, entity="truth")
        _panel_header(ax, f"{name}: {sum(covered)} of 100 cover")
        ax.set_xlim(0, 101)
        ax.set_xticks([1, 25, 50, 75, 100])
        theme.axis_labels(ax, "Replicate sample")
    facts = {"t_hits": str(counts[0]), "z_hits": str(counts[1])}
    axes[0].set_ylim(-3.2, 3.9)
    theme.axis_labels(axes[0], y="Interval for the mean")
    _text(axes[1], 101, 0.0, "true mean", theme.entity_colour("truth"), offset=(4, 0))
    _title(
        axes[0],
        f"Over 100 samples of n = 6 the t interval covers the true mean "
        f"{facts['t_hits']} times, the z interval with s only {facts['z_hits']}",
        "Each line: a 95% interval from one seeded N(0, 1) sample. "
        "Highlighted: intervals that miss 0",
    )
    _finish(figure, wspace=0.08)
    return figure, facts


def _bootstrap_bca() -> tuple[Any, dict[str, str]]:
    data = Exponential(1.0).sample(25, seed=7)
    percentile = bootstrap_confidence_interval(
        statistics.fmean, data, method="percentile", replicates=2000, seed=7
    )
    bca = bootstrap_confidence_interval(
        statistics.fmean, data, method="bca", replicates=2000, seed=7
    )
    student = student_t_mean_confidence_interval(data)
    estimate = bca.bootstrap.estimate
    facts = {
        "bca": f"({bca.lower:.2f}, {bca.upper:.2f})",
        "percentile": f"({percentile.lower:.2f}, {percentile.upper:.2f})",
        "acceleration": f"{bca.acceleration:.3f}",
        "estimate": f"{estimate:.2f}",
    }
    figure, (top, bottom) = _figure(
        1, rows=2, sharex=True, height_ratios=(3.0, 1.25), figsize=(7.2, 4.4)
    )
    palette = theme.active_palette()
    top.hist(
        bca.bootstrap.replicates,
        bins=40,
        density=True,
        color=palette.sequential[2],
        edgecolor=palette.observed,
        linewidth=0.6,
        zorder=2,
    )
    top.set_ylim(0.0, top.get_ylim()[1] * 1.12)
    for ax in (top, bottom):
        theme.reference_line(ax, x=1.0, entity="truth")
        ax.axvline(
            estimate, color=palette.ink, linewidth=theme.REFERENCE_WIDTH, zorder=2
        )
    _text(
        top,
        1.0,
        1.0,
        "true mean 1",
        theme.entity_colour("truth"),
        coords=("data", "axes fraction"),
        va="top",
        offset=(5, -2),
    )
    _text(
        top,
        estimate,
        1.0,
        f"sample mean {facts['estimate']}",
        palette.ink,
        coords=("data", "axes fraction"),
        va="top",
        ha="right",
        offset=(-5, -2),
    )
    _interval_rows(
        bottom,
        (
            (
                "percentile",
                percentile.lower,
                percentile.upper,
                theme.role_colour("context"),
            ),
            ("BCa", bca.lower, bca.upper, theme.role_colour("model")),
            ("Student t", student.lower, student.upper, theme.role_colour("series-3")),
        ),
    )
    top.set_xlim(0.25, 1.75)
    theme.axis_labels(top, y=theme.axis_title("density"))
    theme.axis_labels(bottom, "Bootstrap replicate of the mean")
    top.tick_params(labelbottom=False)
    _title(
        top,
        f"On a skewed sample of 25, BCa moves the 95% interval right to {facts['bca']}",
        "Bars: 2000 bootstrap means of seeded Exp(1) data. "
        f"Acceleration a = {facts['acceleration']}",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Frequentist and Bayesian interpretation
# --------------------------------------------------------------------------


def _binomial_rules(n: int) -> dict[str, list[tuple[float, float]]]:
    credible = []
    for k in range(n + 1):
        law = Beta(1 + k, 1 + n - k)
        credible.append((law.quantile(0.025), law.quantile(0.975)))
    wilson = [
        (iv.lower, iv.upper)
        for iv in (wilson_score_interval(k, n) for k in range(n + 1))
    ]
    exact = [
        (iv.lower, iv.upper)
        for iv in (clopper_pearson_interval(k, n) for k in range(n + 1))
    ]
    return {"Wilson": wilson, "Clopper-Pearson": exact, "credible": credible}


def _exact_coverage() -> tuple[Any, dict[str, str]]:
    n = 20
    rules = _binomial_rules(n)
    grid = [k / 400 for k in range(1, 400)]
    curves: dict[str, list[float]] = {}
    for name, bounds in rules.items():
        values = []
        for p in grid:
            law = Binomial(n, p)
            values.append(
                math.fsum(
                    law.pmf(k)
                    for k in range(n + 1)
                    if bounds[k][0] <= p <= bounds[k][1]
                )
            )
        curves[name] = values
    lows = {name: min(values) for name, values in curves.items()}
    facts = {
        "wilson": f"{lows['Wilson']:.1%}",
        "exact": f"{lows['Clopper-Pearson']:.1%}",
        "credible": f"{lows['credible']:.1%}",
    }
    figure, axes = _figure(3, sharey=True)
    colour = theme.role_colour("model")
    for ax, (name, values) in zip(axes, curves.items(), strict=True):
        theme.line(ax, grid, values, color=colour, width=1.2)
        theme.reference_line(ax, y=0.95, entity="threshold")
        worst = values.index(min(values))
        theme.points(ax, [grid[worst]], [values[worst]], color=colour, size=36.0)
        _text(
            ax,
            grid[worst],
            values[worst],
            f"min {min(values):.1%}",
            colour,
            offset=(8, -2),
            ha="left" if grid[worst] < 0.6 else "right",
        )
        _panel_header(ax, name)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        theme.axis_labels(ax, "True p")
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    axes[0].yaxis.set_major_formatter(_percent)
    _text(axes[2], 1.0, 0.95, "95%", theme.entity_colour("threshold"), offset=(4, 0))
    theme.axis_labels(axes[0], y="Exact coverage")
    _title(
        axes[0],
        f"At n = 20 only Clopper-Pearson keeps 95% coverage for every p "
        f"(min {facts['exact']}); the credible interval falls to "
        f"{facts['credible']}",
        "Exact coverage of three 95% interval rules for a binomial proportion, "
        "p on a grid of step 1/400",
    )
    _finish(figure, wspace=0.16)
    return figure, facts


def _percent(value: float, _position: Any = None) -> str:
    return f"{value:.0%}"


def _posterior_intervals() -> tuple[Any, dict[str, str]]:
    n, x = 20, 14
    posterior = Beta(1 + x, 1 + n - x)
    wilson = wilson_score_interval(x, n)
    exact = clopper_pearson_interval(x, n)
    lower, upper = posterior.quantile(0.025), posterior.quantile(0.975)
    facts = {
        "credible": f"[{lower:.4f}, {upper:.4f}]",
        "wilson": f"[{wilson.lower:.4f}, {wilson.upper:.4f}]",
        "above_half": f"{posterior.sf(0.5):.4f}",
    }
    figure, (top, bottom) = _figure(
        1, rows=2, sharex=True, height_ratios=(3.0, 1.25), figsize=(7.2, 4.4)
    )
    grid = [k / 400 for k in range(1, 400)]
    density = [posterior.pdf(v) for v in grid]
    model = theme.role_colour("model")
    theme.line(top, grid, density, color=model)
    inside = [v for v in grid if lower <= v <= upper]
    theme.band(
        top,
        inside,
        [0.0] * len(inside),
        [posterior.pdf(v) for v in inside],
        color=model,
    )
    theme.reference_line(top, x=0.5, entity="threshold")
    _text(
        top,
        0.5,
        1.0,
        "p = 0.5",
        theme.entity_colour("threshold"),
        coords=("data", "axes fraction"),
        va="top",
        ha="right",
        offset=(-5, -2),
    )
    _text(
        top,
        0.66,
        posterior.pdf(0.66) * 0.45,
        f"P(p > 0.5 | x) = {facts['above_half']}",
        model,
        ha="center",
    )
    top.set_ylim(0.0, max(density) * 1.15)
    _interval_rows(
        bottom,
        (
            ("credible", lower, upper, model),
            ("Wilson", wilson.lower, wilson.upper, theme.role_colour("observed")),
            ("Clopper-Pearson", exact.lower, exact.upper, theme.role_colour("context")),
        ),
        decimals=4,
        brackets="[]",
    )
    top.set_xlim(0.2, 1.0)
    theme.axis_labels(top, y=theme.axis_title("posterior_density"))
    theme.axis_labels(bottom, "Success probability p")
    top.tick_params(labelbottom=False)
    _title(
        top,
        f"At 14 of 20 the 95% credible interval {facts['credible']} nearly "
        f"matches Wilson {facts['wilson']}",
        "Curve: Beta(15, 7) posterior under a uniform prior. Bars: three 95% "
        "intervals from the same data",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Regression and GLMs
# --------------------------------------------------------------------------


def _logistic_band() -> tuple[Any, dict[str, str]]:
    x = [0.5 * k for k in range(1, 11)]
    y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    rows = [(1.0, v) for v in x]
    logit = fit_glm(rows, y, family="binomial")
    probit = fit_glm(rows, y, family="binomial", link="probit")
    b0, b1 = logit.coefficients
    cov = logit.covariance
    z = statistics.NormalDist().inv_cdf(0.975)

    def expit(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def limits(v: float) -> tuple[float, float]:
        eta = b0 + b1 * v
        se = math.sqrt(cov[0][0] + 2.0 * v * cov[0][1] + v * v * cov[1][1])
        return expit(eta - z * se), expit(eta + z * se)

    half = -b0 / b1
    low, high = limits(half)
    facts = {
        "slope": f"{b1:.3f}",
        "half": f"{half:.2f}",
        "band": f"{low:.2f} to {high:.2f}",
    }
    figure, (ax,) = _figure()
    grid = [k / 100 for k in range(0, 551)]
    bands = [limits(v) for v in grid]
    model = theme.role_colour("model")
    theme.band(ax, grid, [b[0] for b in bands], [b[1] for b in bands], color=model)
    theme.line(ax, grid, logit.predict([(1.0, v) for v in grid]), color=model)
    theme.line(
        ax,
        grid,
        probit.predict([(1.0, v) for v in grid]),
        color=theme.role_colour("context"),
        dashes=theme.FITTED,
        width=theme.CONTEXT_WIDTH,
    )
    theme.points(ax, x, y, entity="observations", size=42.0)
    ax.vlines(
        half,
        low,
        high,
        color=model,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=4,
    )
    theme.points(ax, [half], [0.5], color=model, size=36.0)
    _text(
        ax,
        half,
        high,
        f"x = {facts['half']}: band {facts['band']}",
        model,
        offset=(-6, 8),
        ha="right",
        va="bottom",
    )
    _text(
        ax,
        5.5,
        logit.predict([(1.0, 5.5)])[0],
        "logit fit",
        model,
        offset=(4, -12),
        ha="right",
        va="top",
    )
    _text(
        ax,
        5.5,
        probit.predict([(1.0, 5.5)])[0],
        "probit fit",
        theme.role_colour("context"),
        offset=(4, 6),
        ha="right",
        va="bottom",
        bold=False,
    )
    _text(
        ax,
        0.5,
        0.0,
        "observed 0/1 outcomes",
        theme.entity_colour("observations"),
        offset=(10, 12),
        va="bottom",
    )
    ax.set_xlim(0.0, 5.5)
    ax.set_ylim(-0.06, 1.06)
    theme.axis_labels(ax, "Predictor x", "Probability of y = 1")
    _title(
        ax,
        f"The logistic fit crosses p = 0.5 at x = {facts['half']}, where its 95% "
        f"band still spans {facts['band']}",
        "Band: Wald 95% interval of the linear predictor mapped through the "
        "inverse logit; dashed: probit fit",
    )
    _finish(figure)
    return figure, facts


def _regularization_data() -> tuple[list[list[float]], list[float]]:
    generator = resolve_stream(7).child("penalized").generator()
    beta = [2.0, -1.5, 0.0, 0.0, 1.0, 0.0]
    rows: list[list[float]] = []
    y: list[float] = []
    for _ in range(50):
        z = generator.gauss(0.0, 1.0)
        row = [0.6 * z + 0.8 * generator.gauss(0.0, 1.0) for _ in range(6)]
        rows.append(row)
        y.append(
            1.0
            + math.fsum(b * v for b, v in zip(beta, row, strict=True))
            + generator.gauss(0.0, 1.0)
        )
    return rows, y


def _lasso_path() -> tuple[Any, dict[str, str]]:
    rows, y = _regularization_data()
    path = elastic_net_path(rows, y, n_penalties=80)
    cv = cross_validate_elastic_net(rows, y, n_penalties=15, folds=5, seed=11)
    facts = {
        "lambda_max": f"{path[0].penalty:.4f}",
        "lambda_min": f"{cv.penalty_min:.4f}",
        "lambda_1se": f"{cv.penalty_1se:.4f}",
    }
    figure, (ax,) = _figure()
    penalties = [fit.penalty for fit in path]
    names = path[0].column_names
    signals = {"x0", "x1", "x4"}
    labels = []
    for index, name in enumerate(names):
        values = [fit.standardized_coefficients[index] for fit in path]
        colour = theme.role_colour("model" if name in signals else "context")
        if name == "x3":
            colour = theme.role_colour("observed")
        theme.line(
            ax,
            penalties,
            values,
            color=colour,
            width=theme.LINE_WIDTH if name in signals | {"x3"} else theme.CONTEXT_WIDTH,
        )
        labels.append((penalties[-1], values[-1], name, colour))
    ax.set_xscale("log")
    ax.set_xlim(penalties[0] * 1.15, penalties[-1])
    ax.axhline(0.0, color=theme.active_palette().muted, linewidth=0.8, zorder=1)
    for value, text in ((cv.penalty_min, "CV minimum"), (cv.penalty_1se, "1-SE rule")):
        theme.reference_line(ax, x=value, entity="threshold")
        _text(
            ax,
            value,
            -0.55,
            text,
            theme.entity_colour("threshold"),
            coords="data",
            va="center",
            offset=(4, 0),
        )
    theme.direct_labels(ax, labels, gap_points=13.0)
    ax.set_xticks([1.0, 0.1, 0.01])
    ax.set_xticklabels(["1", "0.1", "0.01"])
    ax.minorticks_off()
    theme.axis_labels(
        ax, "Lasso penalty lambda (log scale, decreasing)", "Standardized coefficient"
    )
    _title(
        ax,
        f"Cross-validation at lambda = {facts['lambda_min']} keeps the three true "
        "signals and the correlated null predictor x3",
        f"Lasso paths from lambda_max = {facts['lambda_max']} down; true "
        "non-zero slopes x0, x1, x4; seeded n = 50",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Count and categorical regression
# --------------------------------------------------------------------------


def _zero_frequencies() -> tuple[Any, dict[str, str]]:
    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] * 4
    y = [0, 0, 1, 0, 3, 0, 6, 0, 2, 0, 2, 0, 5, 4,
         0, 0, 0, 3, 4, 6, 0, 1, 0, 2, 0, 3, 0, 9]  # fmt: skip
    rows = [(1.0, v) for v in x]
    poisson = fit_poisson_regression(rows, y)
    inflated = fit_zero_inflated(rows, y)
    negbin = fit_negative_binomial_regression(rows, y)
    counts = list(range(max(y) + 1))
    observed = [sum(v == k for v in y) for k in counts]
    expected_poisson = [
        math.fsum(Poisson(mu).pmf(k) for mu in poisson.fitted_values) for k in counts
    ]
    pis = [
        1.0
        / (
            1.0
            + math.exp(
                -(inflated.zero_coefficients[0] + inflated.zero_coefficients[1] * v)
            )
        )
        for v in x
    ]
    mus = [
        math.exp(inflated.count_coefficients[0] + inflated.count_coefficients[1] * v)
        for v in x
    ]
    expected_zip = [
        math.fsum(
            (pi if k == 0 else 0.0) + (1.0 - pi) * Poisson(mu).pmf(k)
            for pi, mu in zip(pis, mus, strict=True)
        )
        for k in counts
    ]
    theta = negbin.theta

    def nb_pmf(k: int, mu: float) -> float:
        return math.exp(
            math.lgamma(k + theta)
            - math.lgamma(theta)
            - math.lgamma(k + 1)
            + theta * math.log(theta / (theta + mu))
            + k * math.log(mu / (theta + mu))
        )

    expected_nb = [
        math.fsum(nb_pmf(k, mu) for mu in negbin.fitted_values) for k in counts
    ]
    facts = {
        "observed": str(observed[0]),
        "poisson": f"{expected_poisson[0]:.2f}",
        "zip": f"{expected_zip[0]:.2f}",
        "nb": f"{expected_nb[0]:.2f}",
    }
    figure, (ax,) = _figure()
    palette = theme.active_palette()
    ax.bar(
        counts,
        observed,
        width=0.7,
        color=palette.sequential[2],
        edgecolor=palette.observed,
        linewidth=1.0,
        zorder=2,
    )
    series = (
        ("Poisson", expected_poisson, theme.role_colour("context"), "o", theme.FITTED),
        ("NB2", expected_nb, theme.role_colour("series-3"), "^", theme.PROJECTED),
        ("zero-inflated Poisson", expected_zip, theme.role_colour("model"), "s", None),
    )
    for shift, (_, values, colour, marker, dashes) in zip(
        (-0.2, 0.0, 0.2), series, strict=True
    ):
        shifted = [k + shift for k in counts]
        theme.line(
            ax,
            shifted,
            values,
            color=colour,
            dashes=dashes,
            width=theme.CONTEXT_WIDTH,
            zorder=4,
        )
        theme.points(
            ax, shifted, values, color=colour, marker=marker, size=30.0, zorder=5
        )
    _text(
        ax,
        0,
        observed[0],
        f"observed {facts['observed']}",
        palette.observed,
        offset=(0, 6),
        ha="center",
        va="bottom",
    )
    _text(
        ax,
        1.3,
        expected_poisson[0],
        f"Poisson: {facts['poisson']} zeros",
        theme.role_colour("context"),
    )
    _text(
        ax,
        1.3,
        expected_nb[0],
        f"NB2: {facts['nb']} zeros",
        theme.role_colour("series-3"),
    )
    _text(
        ax,
        1.3,
        expected_zip[0],
        f"zero-inflated Poisson: {facts['zip']} zeros",
        theme.role_colour("model"),
    )
    ax.set_xticks(counts)
    ax.set_xlim(-0.6, counts[-1] + 0.6)
    ax.set_ylim(0.0, max(observed) * 1.18)
    theme.axis_labels(ax, "Count y", "Number of observations")
    _title(
        ax,
        f"Poisson regression expects {facts['poisson']} zeros where "
        f"{facts['observed']} occur; the zero-inflated model expects {facts['zip']}",
        "Bars: observed frequencies of 28 counts. Markers: expected frequencies, "
        "summed over the fitted laws",
    )
    _finish(figure)
    return figure, facts


def _ordinal_probabilities() -> tuple[Any, dict[str, str]]:
    dose = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3] * 2  # fmt: skip
    treated = [0] * 24 + [1] * 24
    response = [
        "low", "low", "low", "mid", "low", "mid", "low", "mid", "low", "mid",
        "high", "low", "mid", "mid", "low", "high", "mid", "low", "mid", "high",
        "mid", "high", "low", "high", "low", "mid", "low", "mid", "mid", "high",
        "mid", "low", "high", "mid", "mid", "high", "high", "mid", "high", "mid",
        "high", "low", "high", "high", "mid", "high", "high", "mid",
    ]  # fmt: skip
    rows = [(float(d), float(t)) for d, t in zip(dose, treated, strict=True)]
    levels = ["low", "mid", "high"]
    fit = fit_ordinal_logistic(
        rows, response, levels=levels, column_names=["dose", "treated"]
    )
    ratios = fit.odds_ratios()
    facts = {"dose": f"{ratios[0][0]:.3f}", "treated": f"{ratios[1][0]:.2f}"}
    figure, axes = _figure(2, sharey=True)
    colours = theme.ordered_colours(3)
    grid = [k / 40 for k in range(0, 121)]
    for ax, group in zip(axes, (0, 1), strict=True):
        probabilities = fit.predict_proba([(v, float(group)) for v in grid])
        labels = []
        for index, level in enumerate(levels):
            values = [p[index] for p in probabilities]
            theme.line(ax, grid, values, color=colours[index])
            shares = []
            for d in range(4):
                cell = [
                    r
                    for r, dd, tt in zip(response, dose, treated, strict=True)
                    if dd == d and tt == group
                ]
                shares.append(sum(r == level for r in cell) / len(cell))
            theme.points(ax, list(range(4)), shares, color=colours[index], size=30.0)
            labels.append((grid[-1], values[-1], f"P({level})", _ink(colours[index])))
        theme.direct_labels(ax, labels)
        _panel_header(ax, "untreated" if group == 0 else "treated")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.15, 3.15)
        theme.axis_labels(ax, "Dose")
    axes[0].set_ylim(0.0, 0.8)
    theme.axis_labels(axes[0], y="Category probability")
    _title(
        axes[0],
        f"Each dose step multiplies the odds of a higher category by {facts['dose']}",
        "Curves: proportional-odds fit. Points: observed shares among the six "
        "patients of each cell",
    )
    _finish(figure, wspace=0.3)
    return figure, facts


# --------------------------------------------------------------------------
# Robust and quantile regression
# --------------------------------------------------------------------------


def _leverage_lines() -> tuple[Any, dict[str, str]]:
    x = [float(i) for i in range(15)] + [100.0] * 5
    y = [3.0 * v + 1.0 + (0.1 if i % 2 else -0.1) for i, v in enumerate(x[:15])]
    y += [-1000.0] * 5
    ols = fit_simple_linear_regression(x, y)
    huber = fit_robust_linear_model([(1.0, v) for v in x], y)
    sen = theil_sen_slope(x, y)
    facts = {
        "ols": f"{ols.slope:.3f}",
        "huber": f"{huber.coefficients[1]:.3f}",
        "sen": f"{sen.slope:.2f}",
    }
    figure, (ax,) = _figure()
    grid = [0.0, 105.0]
    model = theme.role_colour("model")
    context = theme.role_colour("context")
    theme.points(ax, x[:15], y[:15], entity="observations", size=30.0)
    theme.points(
        ax, x[15:], y[15:], color=theme.role_colour("series-3"), size=60.0, marker="X"
    )
    theme.line(
        ax,
        grid,
        [ols.intercept + ols.slope * v for v in grid],
        color=context,
        width=theme.CONTEXT_WIDTH,
    )
    theme.line(
        ax,
        grid,
        [huber.coefficients[0] + huber.coefficients[1] * v for v in grid],
        color=theme.role_colour("series-4"),
        dashes=theme.FITTED,
        width=theme.CONTEXT_WIDTH,
    )
    theme.line(ax, grid, [sen.intercept + sen.slope * v for v in grid], color=model)
    _text(
        ax,
        105,
        sen.intercept + sen.slope * 105,
        f"Theil-Sen, slope {facts['sen']}",
        model,
        ha="right",
        offset=(0, 10),
        va="bottom",
    )
    _text(
        ax,
        60,
        ols.intercept + ols.slope * 60,
        f"least squares {facts['ols']} and Huber {facts['huber']}",
        context,
        offset=(8, 4),
        va="bottom",
    )
    _text(
        ax,
        100,
        -1000,
        "5 planted points at x = 100",
        theme.role_colour("series-3"),
        ha="right",
        offset=(-12, 0),
    )
    _text(
        ax,
        14,
        y[14],
        "15 points on y = 3x + 1",
        theme.entity_colour("observations"),
        offset=(10, -6),
        va="top",
    )
    ax.set_xlim(-3.0, 108.0)
    ax.set_ylim(-1120.0, 420.0)
    theme.axis_labels(ax, "Predictor x", "Response y")
    _title(
        ax,
        f"Five far points drag least squares to slope {facts['ols']} and Huber "
        f"to {facts['huber']}; Theil-Sen keeps {facts['sen']}",
        "A quarter of the points sit at one leverage position; lines: three "
        "slope estimators",
    )
    _finish(figure)
    return figure, facts


def _quantile_fan() -> tuple[Any, dict[str, str]]:
    generator = resolve_stream(2027).generator()
    x = [generator.uniform(0.0, 10.0) for _ in range(300)]
    y = [2.0 + 0.5 * v + (0.3 + 0.25 * v) * generator.gauss(0.0, 1.0) for v in x]
    design = [(1.0, v) for v in x]
    taus = (0.1, 0.25, 0.5, 0.75, 0.9)
    fits = {tau: fit_quantile_regression(design, y, quantile=tau) for tau in taus}
    normal = statistics.NormalDist()
    facts = {
        "upper": f"{fits[0.9].coefficients[1]:.2f}",
        "lower": f"{fits[0.1].coefficients[1]:.2f}",
        "true_upper": f"{0.5 + 0.25 * normal.inv_cdf(0.9):.2f}",
        "true_lower": f"{0.5 + 0.25 * normal.inv_cdf(0.1):.2f}",
    }
    figure, (ax,) = _figure()
    theme.points(
        ax, x, y, color=theme.role_colour("context"), size=12.0, alpha=0.55, zorder=2
    )
    shades = theme.ordered_colours(3)
    depth = {0.1: 0, 0.9: 0, 0.25: 1, 0.75: 1, 0.5: 2}
    labels = []
    for tau, fit in fits.items():
        b0, b1 = fit.coefficients
        colour = shades[depth[tau]]
        theme.line(
            ax,
            [0.0, 10.0],
            [b0, b0 + 10.0 * b1],
            color=colour,
            width=theme.LINE_WIDTH if tau == 0.5 else theme.CONTEXT_WIDTH + 0.4,
        )
        labels.append(
            (10.0, b0 + 10.0 * b1, f"tau = {tau:g}, slope {b1:.2f}", _ink(colour))
        )
    ax.set_xlim(0.0, 10.0)
    theme.direct_labels(ax, labels)
    theme.axis_labels(ax, "Predictor x", "Response y")
    _title(
        ax,
        f"As the spread grows with x, the 0.9 and 0.1 regression quantiles "
        f"fan out: slopes {facts['upper']} and {facts['lower']}",
        "300 seeded points, y = 2 + 0.5x + (0.3 + 0.25x) e; lines: quantile "
        "regression at five tau",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Smoothing and additive models
# --------------------------------------------------------------------------


def _pspline_smoothness() -> tuple[Any, dict[str, str]]:
    generator = resolve_stream(2026).generator()
    x = [i / 60 for i in range(61)]
    y = [math.sin(2 * math.pi * v) + generator.gauss(0.0, 0.3) for v in x]
    rough = fit_pspline(x, y, segments=20, smoothing=1e-4)
    reml = fit_pspline(x, y, segments=20)
    stiff = fit_pspline(x, y, segments=20, smoothing=1e4)
    facts = {
        "reml": f"{reml.edf:.2f}",
        "rough": f"{rough.edf:.2f}",
        "stiff": f"{stiff.edf:.2f}",
    }
    figure, axes = _figure(3, sharey=True)
    grid = [k / 300 for k in range(301)]
    truth = [math.sin(2 * math.pi * v) for v in grid]
    panels = (
        (rough, f"lambda = 1e-4: {facts['rough']} df", theme.role_colour("context")),
        (reml, f"REML: {facts['reml']} df", theme.role_colour("model")),
        (stiff, f"lambda = 1e4: {facts['stiff']} df", theme.role_colour("context")),
    )
    for ax, (fit, header, colour) in zip(axes, panels, strict=True):
        theme.points(ax, x, y, entity="observations", size=12.0, zorder=2)
        theme.line(ax, grid, truth, entity="truth")
        if fit is reml:
            band = fit.band(grid)
            theme.band(ax, grid, band.lower, band.upper, color=colour)
        theme.line(ax, grid, fit.predict(grid), color=colour)
        _panel_header(ax, header)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        theme.axis_labels(ax, "x")
    axes[0].set_ylim(-1.9, 1.8)
    theme.axis_labels(axes[0], y="y")
    _text(
        axes[1],
        0.75,
        -1.0,
        "true sin(2 pi x)",
        theme.entity_colour("truth"),
        offset=(0, -14),
        ha="center",
        va="top",
        bold=False,
    )
    _title(
        axes[0],
        f"REML picks {facts['reml']} effective degrees of freedom, between a "
        f"{facts['rough']}-df wiggle and a {facts['stiff']}-df line",
        "Cubic P-splines with 20 segments on 61 seeded points; band: 95% "
        "pointwise Bayesian band",
    )
    _finish(figure, wspace=0.16)
    return figure, facts


def _additive_terms() -> tuple[Any, dict[str, str]]:
    generator = resolve_stream(102).generator()
    a = [generator.uniform(0.0, 1.0) for _ in range(100)]
    b = [generator.uniform(-1.0, 1.0) for _ in range(100)]
    response = [
        math.sin(2 * math.pi * u) + v * v + generator.gauss(0.0, 0.2)
        for u, v in zip(a, b, strict=True)
    ]
    data = {"a": a, "b": b}
    gam = fit_additive_model(data, response, segments=10)
    terms = {term.name: term for term in gam.terms}
    facts = {"a": f"{terms['a'].edf:.2f}", "b": f"{terms['b'].edf:.2f}"}
    figure, axes = _figure(2, sharey=True)
    model = theme.role_colour("model")
    shapes = {
        "a": (a, lambda u: math.sin(2 * math.pi * u), (0.0, 1.0), "sin(2 pi a)"),
        "b": (b, lambda v: v * v, (-1.0, 1.0), "b squared"),
    }
    for ax, name in zip(axes, ("a", "b"), strict=True):
        values, shape, _, truth_name = shapes[name]
        other = "b" if name == "a" else "a"
        partial = [
            r - gam.intercept - e
            for r, e in zip(response, terms[other].evaluate(data[other]), strict=True)
        ]
        low, high = min(values), max(values)
        grid = [low + (high - low) * k / 200 for k in range(201)]
        centre = statistics.fmean(shape(v) for v in values)
        band = terms[name].band(grid)
        theme.points(ax, values, partial, entity="observations", size=12.0, zorder=2)
        theme.band(ax, grid, band.lower, band.upper, color=model)
        theme.line(ax, grid, [shape(v) - centre for v in grid], entity="truth")
        theme.line(ax, grid, terms[name].evaluate(grid), color=model)
        _panel_header(ax, f"term f({name}): {facts[name]} df")
        ax.set_xlim(*shapes[name][2])
        span = shapes[name][2]
        ax.set_xticks([span[0] + (span[1] - span[0]) * k / 4 for k in range(5)])
        theme.axis_labels(ax, f"Predictor {name}")

    axes[0].set_ylim(-1.8, 1.9)
    theme.axis_labels(axes[0], y="Partial effect")
    _title(
        axes[0],
        f"The additive model recovers both shapes, with {facts['a']} and "
        f"{facts['b']} effective degrees of freedom",
        "Solid: centred P-spline terms with 95% bands; dashed: true shapes. "
        "Points: partial residuals",
    )
    _finish(figure, wspace=0.16)
    return figure, facts


# --------------------------------------------------------------------------
# Multivariate analysis
# --------------------------------------------------------------------------


def _pca_rows() -> tuple[Any, list[list[float]]]:
    generator = resolve_stream(1936).generator()
    rows = []
    for _ in range(200):
        size = generator.gauss(0.0, 1.0)
        rows.append([
            170 + 8 * size + generator.gauss(0.0, 3.0),
            70 + 9 * size + generator.gauss(0.0, 5.0),
            1.0 + 0.01 * size + 0.02 * generator.gauss(0.0, 1.0),
        ])  # fmt: skip
    return generator, rows


def _pca_scaling() -> tuple[Any, dict[str, str]]:
    _, rows = _pca_rows()
    raw = principal_components(rows)
    scaled = principal_components(rows, standardize=True)
    facts = {
        "raw_length": f"{raw.axes[0][2]:.3f}",
        "scaled_length": f"{scaled.axes[0][2]:.3f}",
        "raw_share": f"{raw.explained_variance_ratio[0]:.1%}",
        "scaled_share": f"{scaled.explained_variance_ratio[0]:.1%}",
    }
    figure, (ax,) = _figure(figsize=(7.2, 3.6))
    names = ("height [cm]", "weight [kg]", "length [m]")
    positions = (2, 1, 0)
    context = theme.role_colour("context")
    model = theme.role_colour("model")

    upper = [pos + 0.14 for pos in positions]
    lower = [pos - 0.14 for pos in positions]
    theme.points(ax, raw.axes[0], upper, color=context, size=60.0, marker="o")
    theme.points(ax, scaled.axes[0], lower, color=model, size=60.0, marker="s")
    _text(
        ax,
        raw.axes[0][2],
        upper[2],
        f"covariance PCA {facts['raw_length']}",
        context,
        offset=(10, 0),
    )
    _text(
        ax,
        scaled.axes[0][2],
        lower[2],
        f"correlation PCA {facts['scaled_length']}",
        model,
        offset=(10, 0),
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.6, 2.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.active_palette().rule, linewidth=0.7)
    theme.axis_labels(ax, "Weight on the first principal axis")
    _title(
        ax,
        f"Measured in metres, length gets weight {facts['raw_length']} on the "
        f"first axis; standardized, it gets {facts['scaled_length']}",
        f"First principal axis of 200 seeded rows: covariance PCA "
        f"({facts['raw_share']} of variance), correlation PCA "
        f"({facts['scaled_share']})",
    )
    _finish(figure)
    return figure, facts


def _lda_regions() -> tuple[Any, dict[str, str]]:
    generator, _ = _pca_rows()
    pattern = [[0.9, 0.0], [0.8, 0.1], [0.7, 0.0], [0.0, 0.8], [0.1, 0.7], [0.0, 0.6]]
    for _ in range(300):
        for _ in range(2):
            generator.gauss(0.0, 1.0)
        for _ in pattern:
            generator.gauss(0.0, 1.0)
    centres = {"a": (0.0, 0.0), "b": (2.5, 0.5), "c": (0.5, 2.5)}
    train: list[list[float]] = []
    labels: list[str] = []
    for label, (cx, cy) in centres.items():
        for _ in range(60):
            train.append(
                [cx + generator.gauss(0.0, 1.0), cy + generator.gauss(0.0, 1.0)]
            )
            labels.append(label)
    lda = fit_linear_discriminant(train, labels)
    errors = sum(p != t for p, t in zip(lda.predict(train), labels, strict=True))
    facts = {"errors": str(errors), "rate": f"{errors / len(train):.1%}"}
    from matplotlib.colors import ListedColormap, to_rgb

    figure, (ax,) = _figure(figsize=(7.2, 4.6))
    palette = theme.active_palette()
    roles = {"a": "series-3", "b": "series-5", "c": "series-6"}
    colours = {k: theme.role_colour(v) for k, v in roles.items()}
    low_x, high_x, low_y, high_y = -3.0, 5.5, -3.0, 5.5
    steps = 240
    grid = [
        [low_x + (high_x - low_x) * i / steps, low_y + (high_y - low_y) * j / steps]
        for j in range(steps + 1)
        for i in range(steps + 1)
    ]
    order = list(centres)
    predicted = [order.index(p) for p in lda.predict(grid)]
    image = [
        predicted[j * (steps + 1) : (j + 1) * (steps + 1)] for j in range(steps + 1)
    ]
    canvas = to_rgb(palette.canvas)
    tints = [
        tuple(
            0.82 * c + 0.18 * f for c, f in zip(canvas, to_rgb(colours[k]), strict=True)
        )
        for k in order
    ]
    ax.imshow(
        image,
        origin="lower",
        extent=(low_x, high_x, low_y, high_y),
        cmap=ListedColormap(tints),
        vmin=-0.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    markers = {"a": "o", "b": "s", "c": "^"}
    for label in order:
        xs = [r[0] for r, t in zip(train, labels, strict=True) if t == label]
        ys = [r[1] for r, t in zip(train, labels, strict=True) if t == label]
        theme.points(
            ax, xs, ys, color=colours[label], marker=markers[label], size=24.0, zorder=3
        )
        cx, cy = centres[label]
        _text(ax, cx, cy, f"class {label}", palette.ink, ha="center", offset=(0, 0))
    ax.set_xlim(low_x, high_x)
    ax.set_ylim(low_y, high_y)
    ax.grid(False)
    theme.axis_labels(ax, "First measurement", "Second measurement")
    _title(
        ax,
        f"LDA splits the plane with straight boundaries and misclassifies "
        f"{facts['errors']} of 180 training points ({facts['rate']})",
        "Shaded: predicted class. Markers: 60 seeded points per class around "
        "the labelled centres",
    )
    _finish(figure)
    return figure, facts


# --------------------------------------------------------------------------
# Mixed models
# --------------------------------------------------------------------------


def _blup_shrinkage() -> tuple[Any, dict[str, str]]:
    generator = resolve_stream(2028).generator()
    sizes = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30)
    group: list[int] = []
    y: list[float] = []
    for index, size in enumerate(sizes):
        effect = generator.gauss(0.0, 1.0)
        for _ in range(size):
            group.append(index)
            y.append(10.0 + effect + generator.gauss(0.0, 2.0))
    fit = fit_linear_mixed_model(
        [[1.0]] * len(y), y, random_effect(group, name="group")
    )
    intercept = fit.coefficients[0]
    sigma_b = fit.variance_components[0].standard_deviations[0]
    sigma = fit.residual_sd
    estimates = fit.random_effects[0]
    raw = [
        statistics.fmean(v for v, g in zip(y, group, strict=True) if g == index)
        - intercept
        for index in range(len(sizes))
    ]
    blups = [values[0] for values in estimates.values]
    kept = [size * sigma_b**2 / (size * sigma_b**2 + sigma**2) for size in sizes]
    facts = {
        "small": f"{kept[0]:.0%}",
        "large": f"{kept[-1]:.0%}",
        "sigma_b": f"{sigma_b:.2f}",
        "sigma": f"{sigma:.2f}",
    }
    figure, (ax,) = _figure(figsize=(7.2, 4.6))
    context = theme.role_colour("context")
    model = theme.role_colour("model")
    observed = theme.entity_colour("observations")
    positions = list(range(len(sizes)))
    for pos, start, end in zip(positions, raw, blups, strict=True):
        ax.annotate(
            "",
            xy=(end, pos),
            xytext=(start, pos),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.4,head_width=0.2",
                "color": context,
                "linewidth": 1.0,
                "shrinkA": 4.0,
                "shrinkB": 4.0,
            },
            zorder=2,
        )
    theme.points(ax, raw, positions, color=observed, size=40.0)
    theme.points(ax, blups, positions, color=model, size=40.0, marker="s")
    ax.axvline(
        0.0,
        color=theme.entity_colour("threshold"),
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=theme.REFERENCE,
        zorder=1,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([f"n = {size}" for size in sizes])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.active_palette().rule, linewidth=0.7)
    top = len(sizes) - 1
    _text(
        ax,
        raw[0],
        0,
        f"raw mean, keeps {facts['small']} as BLUP",
        observed,
        offset=(10, 0) if raw[0] > blups[0] else (-10, 0),
        ha="left" if raw[0] > blups[0] else "right",
    )
    _text(
        ax,
        blups[top],
        top,
        f"BLUP keeps {facts['large']}",
        model,
        offset=(10, 0) if blups[top] > raw[top] else (-10, 0),
        ha="left" if blups[top] > raw[top] else "right",
    )
    _text(
        ax,
        0.0,
        1.0,
        f"fixed intercept {intercept:.2f}",
        theme.entity_colour("threshold"),
        coords=("data", "axes fraction"),
        offset=(5, -3),
        va="top",
    )
    span = max(abs(v) for v in raw) * 1.25
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.7, top + 1.4)

    theme.axis_labels(ax, "Deviation from the fixed intercept", "Group size")
    _title(
        ax,
        f"A group of 2 keeps {facts['small']} of its raw deviation as its BLUP, "
        f"a group of 30 keeps {facts['large']}",
        f"Arrows: raw group mean to BLUP; REML sigma_b = {facts['sigma_b']}, "
        f"sigma = {facts['sigma']}, 12 seeded groups",
    )
    _finish(figure)
    return figure, facts


def _growth_lines() -> tuple[Any, dict[str, str]]:
    subject = [s for s in range(6) for _ in range(5)]
    week = [float(w) for _ in range(6) for w in range(5)]
    score = [
        50.1, 52.3, 55.0, 56.2, 58.9,
        47.5, 48.1, 49.6, 50.0, 51.8,
        53.0, 56.4, 59.1, 62.5, 65.2,
        49.2, 51.0, 52.2, 54.9, 55.7,
        51.4, 52.0, 54.7, 55.1, 57.3,
        46.8, 49.5, 51.9, 54.0, 56.6,
    ]  # fmt: skip
    design = [[1.0, w] for w in week]
    slopes = random_effect(subject, slopes={"week": week}, name="subject")
    fit = fit_linear_mixed_model(
        design, score, slopes, column_names=["(Intercept)", "week"]
    )
    b0, b1 = fit.coefficients
    components = fit.variance_components[0]
    facts = {
        "slope": f"{b1:.2f}",
        "slope_sd": f"{components.standard_deviations[1]:.2f}",
        "correlation": f"{components.correlations[0][1]:.2f}",
    }
    figure, (ax,) = _figure()
    context = theme.role_colour("context")
    model = theme.role_colour("model")
    observed = theme.entity_colour("observations")
    weeks = [0.0, 4.0]
    for index, effect in enumerate(fit.random_effects[0].values):
        values = [score[5 * index + k] for k in range(5)]
        theme.points(
            ax, [0, 1, 2, 3, 4], values, color=observed, size=22.0, alpha=0.8, zorder=3
        )
        theme.line(
            ax,
            weeks,
            [b0 + effect[0] + (b1 + effect[1]) * w for w in weeks],
            color=context,
            width=theme.CONTEXT_WIDTH,
            zorder=2,
        )
    theme.line(ax, weeks, [b0 + b1 * w for w in weeks], color=model, zorder=4)
    _text(
        ax,
        4.0,
        b0 + 4.0 * b1,
        f"population: {facts['slope']} per week",
        model,
        offset=(8, 0),
    )
    _text(ax, 4.0, 65.2, "subject lines (BLUP)", context, offset=(8, 0), bold=False)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.2, 4.2)
    theme.axis_labels(ax, "Week", "Score")
    _title(
        ax,
        f"Subject slopes scatter around the population slope {facts['slope']} "
        f"per week with standard deviation {facts['slope_sd']}",
        "Points: six subjects over five weeks. Lines: REML fixed effect and "
        "subject-specific BLUP lines",
    )
    _finish(figure)
    return figure, facts


FIGURES: list[GuideFigure] = [
    GuideFigure(
        name="descriptive-statistics-skewed-ecdf",
        guide="descriptive-statistics",
        alt=(
            "Empirical CDF of 200 right-skewed draws with the law's CDF and "
            "vertical lines at the sample mean and median"
        ),
        draw=_skewed_ecdf,
    ),
    GuideFigure(
        name="hypothesis-testing-interval-coverage",
        guide="hypothesis-testing-and-bootstrap",
        alt=(
            "Two panels of 100 replicate 95% intervals for a normal mean, "
            "Student-t and z with s, with the misses highlighted"
        ),
        draw=_interval_coverage,
    ),
    GuideFigure(
        name="hypothesis-testing-bootstrap-bca",
        guide="hypothesis-testing-and-bootstrap",
        alt=(
            "Histogram of 2000 bootstrap means of a skewed sample above bars "
            "for the percentile, BCa, and Student-t intervals"
        ),
        draw=_bootstrap_bca,
    ),
    GuideFigure(
        name="frequentist-bayesian-exact-coverage",
        guide="frequentist-and-bayesian-interpretation",
        alt=(
            "Three panels of exact coverage against the true proportion for the "
            "Wilson, Clopper-Pearson, and credible intervals at n = 20"
        ),
        draw=_exact_coverage,
    ),
    GuideFigure(
        name="frequentist-bayesian-posterior-intervals",
        guide="frequentist-and-bayesian-interpretation",
        alt=(
            "Beta posterior density for 14 successes in 20 trials above bars "
            "for the credible, Wilson, and Clopper-Pearson intervals"
        ),
        draw=_posterior_intervals,
    ),
    GuideFigure(
        name="regression-logistic-band",
        guide="regression-and-glms",
        alt=(
            "Binary outcomes against a predictor with the fitted logistic curve, "
            "its 95% band, and a probit curve"
        ),
        draw=_logistic_band,
    ),
    GuideFigure(
        name="regression-lasso-path",
        guide="regression-and-glms",
        alt=(
            "Six lasso coefficient paths against a decreasing log penalty with "
            "the cross-validation choices marked"
        ),
        draw=_lasso_path,
    ),
    GuideFigure(
        name="count-regression-zero-frequencies",
        guide="count-and-categorical-regression",
        alt=(
            "Bars of observed count frequencies with expected frequencies under "
            "Poisson, NB2, and zero-inflated Poisson fits"
        ),
        draw=_zero_frequencies,
    ),
    GuideFigure(
        name="count-regression-ordinal-probabilities",
        guide="count-and-categorical-regression",
        alt=(
            "Two panels of fitted low, mid, and high category probabilities "
            "against dose with observed shares per cell"
        ),
        draw=_ordinal_probabilities,
    ),
    GuideFigure(
        name="robust-regression-leverage",
        guide="robust-and-quantile-regression",
        alt=(
            "Scatter of 15 points on a line and 5 far points with least-squares, "
            "Huber, and Theil-Sen lines"
        ),
        draw=_leverage_lines,
    ),
    GuideFigure(
        name="quantile-regression-fan",
        guide="robust-and-quantile-regression",
        alt=(
            "Scatter with spread growing in x and five regression quantile lines "
            "from tau 0.1 to 0.9"
        ),
        draw=_quantile_fan,
    ),
    GuideFigure(
        name="smoothing-pspline-reml",
        guide="smoothing-and-additive-models",
        alt=(
            "Three panels of P-spline fits to noisy sine data at a small, a "
            "REML-chosen, and a large penalty"
        ),
        draw=_pspline_smoothness,
    ),
    GuideFigure(
        name="smoothing-additive-terms",
        guide="smoothing-and-additive-models",
        alt=(
            "Two panels of fitted additive-model terms with bands, partial "
            "residuals, and the true centred shapes"
        ),
        draw=_additive_terms,
    ),
    GuideFigure(
        name="multivariate-pca-scaling",
        guide="multivariate-analysis",
        alt=(
            "Dot plot of the first principal-axis weights of height, weight, and "
            "length under covariance and correlation PCA"
        ),
        draw=_pca_scaling,
    ),
    GuideFigure(
        name="multivariate-lda-regions",
        guide="multivariate-analysis",
        alt=(
            "Plane shaded by the class LDA predicts, with the training points of "
            "three classes as markers"
        ),
        draw=_lda_regions,
    ),
    GuideFigure(
        name="mixed-models-blup-shrinkage",
        guide="mixed-models",
        alt=(
            "Dumbbell chart of raw group-mean deviations and BLUPs for 12 groups "
            "ordered by size"
        ),
        draw=_blup_shrinkage,
    ),
    GuideFigure(
        name="mixed-models-growth-lines",
        guide="mixed-models",
        alt=(
            "Scores of six subjects over five weeks with subject BLUP lines and "
            "the population line"
        ),
        draw=_growth_lines,
    ),
]
