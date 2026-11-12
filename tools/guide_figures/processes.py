"""Figures of the process, design, and effect-estimation guides (ADR-0105).

Twelve figures for nine guides: time series (forecast intervals, GARCH
volatility, PELT change points), stochastic processes (strong convergence of
SDE schemes, Karhunen-Loeve fields), filtering (the unmeasured velocity),
causal inference (double robustness), the visual language (the encoding
rules on one fit), missing data (interval coverage), experimental design
(the equivalence theorem), survival analysis (censoring), and extreme values
(delta against profile return-level intervals). Each ``draw`` runs a seeded
library computation and returns the figure with the facts its caption
quotes; computations that take seconds are cached, so the light and the dark
run share one result.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import functools
import math
import random
import statistics
from typing import Any

from guide_figures import GuideFigure
from probstat.causal import (
    estimate_aipw,
    estimate_ipw,
    estimate_regression_adjustment,
)
from probstat.design import (
    ApproximateDesign,
    d_optimal_design,
    polynomial_basis,
    standardized_variances,
)
from probstat.distributions import GeneralizedExtremeValue
from probstat.filtering import (
    LinearGaussianStateSpace,
    kalman_filter,
    rts_smoother,
    simulate,
)
from probstat.foundations import RandomStream
from probstat.inference import (
    complete_cases,
    fit_incomplete_multivariate_normal,
    fit_linear_model,
    impute_multivariate_normal,
    mean_impute,
    pool_estimates,
    student_t_quantile,
)
from probstat.inference.survival import (
    fit_parametric_survival,
    kaplan_meier,
    survival_data,
)
from probstat.multivariate import MultivariateNormal
from probstat.processes import (
    CovarianceKernel,
    GeometricBrownianMotion,
    ScalarSDE,
    karhunen_loeve_interval,
    strong_convergence_study,
)
from probstat.risk import fit_gev, gev_return_level
from probstat.timeseries import (
    fit_arima,
    fit_garch,
    pelt,
    simulate_arima,
    simulate_arma,
    simulate_garch,
)
from probstat.visualization import theme
from probstat.visualization.causal_plots import plot_effect_estimates
from probstat.visualization.design_plots import plot_optimal_design
from probstat.visualization.extremes_plots import hazen_return_periods
from probstat.visualization.inference_plots import plot_filter_track

FIGSIZE = (7.2, 4.0)


def _restate(axes: Any, title: str, subtitle: str | None = None) -> None:
    """Replace the computed title (and subtitle) of a library plot.

    Library plots compute their own assertion; a guide figure states the
    finding with the fact strings its caption quotes, so the title text is
    replaced in place (font, colour, and padding are kept) and the subtitle
    annotation is rewritten.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    heading = axes.title if axes.get_title() else axes._left_title
    old = heading.get_text()
    heading.set_text(title[:1].upper() + title[1:])
    if subtitle is None:
        return
    for text in axes.texts:
        if text.get_text() and text.get_text() != old and text.xyann == (0.0, 8.0):
            text.set_text(subtitle)
            return


def _panel_heading(axes: Any, text: str) -> None:
    """Name a panel in semibold ink at the top left inside the axes.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    axes.text(
        0.02,
        0.97,
        text,
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=theme.LABEL_SIZE,
        fontweight="semibold",
        color=theme.active_palette().ink,
    )


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------


def _forecast_panel(axes: Any, series: Any, forecast: Any, shown: int) -> None:
    """Draw the last observations and a forecast with its 95% band.

    References:
        [EQ-TS-FC-001], [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    n = len(series)
    past = list(range(n - shown + 1, n + 1))
    future = list(range(n, n + len(forecast.mean) + 1))
    last = series[-1]
    theme.line(axes, past, series[-shown:], entity="observations")
    theme.band(
        axes,
        future,
        [last, *forecast.lower],
        [last, *forecast.upper],
        entity="forecast mean",
    )
    theme.line(axes, future, [last, *forecast.mean], entity="forecast mean")
    theme.direct_labels(
        axes,
        [
            (future[-1], forecast.upper[-1], "95%", palette.model),
            (future[-1], forecast.mean[-1], "forecast", palette.model),
        ],
    )


def _draw_forecast() -> tuple[Any, dict[str, str]]:
    """Random walk against AR(1): forecast intervals over 20 steps.

    References:
        [EQ-TS-FC-001], [REF-BROCKWELL-DAVIS-2016].

    """
    walk = simulate_arima((), 1, (), 300, seed=110)
    walk_forecast = fit_arima(walk, (0, 1, 0)).forecast(20)
    ar = simulate_arma((0.6,), (), 400, mean=2.0, seed=111)
    ar_fit = fit_arima(ar, (1, 0, 0))
    ar_forecast = ar_fit.forecast(20)

    def widening(forecast: Any) -> float:
        return float(forecast.standard_errors[-1] / forecast.standard_errors[0])

    facts = {
        "walk": f"{widening(walk_forecast):.2f}x",
        "ar": f"{widening(ar_forecast):.2f}x",
        "phi": f"{ar_fit.ar[0]:.4f}",
    }
    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE)
    for axes, series, forecast, name in (
        (left, walk, walk_forecast, "Random walk, ARIMA(0, 1, 0)"),
        (right, ar, ar_forecast, f"AR(1), phi = {facts['phi']}"),
    ):
        theme.style_axes(axes)
        _forecast_panel(axes, series, forecast, 60)
        low, high = axes.get_ylim()
        axes.set_ylim(low, high + 0.18 * (high - low))
        _panel_heading(axes, name)
        theme.axis_labels(axes, "Time index")
    theme.axis_labels(left, None, "Value")
    theme.title_block(
        left,
        f"Over 20 steps the random-walk interval widens {facts['walk']}, "
        f"the AR(1) interval only {facts['ar']}",
        "Solid: last 60 observations. Dotted line and band: forecast mean "
        "and 95% interval",
    )
    figure.subplots_adjust(wspace=0.38)
    return figure, facts


def _draw_garch() -> tuple[Any, dict[str, str]]:
    """Fitted GARCH(1, 1) volatility against the simulated one.

    References:
        [EQ-TS-GARCH-001], [EQ-TS-GARCH-002], [REF-BOLLERSLEV-1986].

    """
    simulation = simulate_garch(0.05, (0.1,), (0.85,), 1500, mean=0.02, seed=2026)
    fit = fit_garch(simulation.series)
    true_sd = [math.sqrt(v) for v in simulation.conditional_variances]
    fitted_sd = [math.sqrt(v) for v in fit.conditional_variances]
    facts = {
        "correlation": f"{statistics.correlation(true_sd, fitted_sd):.3f}",
        "persistence": f"{fit.persistence:.3f}",
    }
    times = list(range(1, len(simulation.series) + 1))
    plt = theme.pyplot()
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.2, 4.6),
        gridspec_kw={"height_ratios": (1.0, 1.25), "hspace": 0.12},
    )
    for axes in (top, bottom):
        theme.style_axes(axes)
    palette = theme.active_palette()
    theme.line(
        top,
        times,
        simulation.series,
        color=palette.observed,
        width=theme.PATH_WIDTH,
    )
    theme.axis_labels(top, None, "Return")
    theme.line(bottom, times, fitted_sd, color=palette.model, label="fitted")
    theme.line(bottom, times, true_sd, entity="truth", zorder=4)
    bottom.set_xlim(0, len(times) * 1.0)
    theme.axis_labels(bottom, "Time index", "Conditional sd")
    right = bottom.get_xlim()[1]
    theme.direct_labels(
        bottom,
        [
            (right, true_sd[-1], "true", theme.entity_colour("truth")),
            (right, fitted_sd[-1], "fitted", palette.model),
        ],
        gap_points=12.0,
    )
    theme.title_block(
        top,
        f"Fitted GARCH(1, 1) volatility follows the simulated clusters: "
        f"correlation {facts['correlation']} with the true sigma_t",
        f"Top: 1500 simulated returns. Bottom: conditional sd sigma_t, "
        f"fitted persistence alpha + beta = {facts['persistence']}",
    )
    return figure, facts


def _draw_pelt() -> tuple[Any, dict[str, str]]:
    """PELT segmentation of a series with a mean and a variance change.

    References:
        [EQ-TS-CP-002], [EQ-TS-CP-003], [REF-KILLICK-2012].

    """
    generator = random.Random(2012)
    series = (
        [generator.gauss(0.0, 1.0) for _ in range(300)]
        + [generator.gauss(3.0, 1.0) for _ in range(300)]
        + [generator.gauss(3.0, 3.0) for _ in range(300)]
    )
    found = pelt(series, model="meanvar", penalty="mbic")
    facts = {
        "changes": " and ".join(str(tau) for tau in found.changepoints),
        "first_variance": f"{found.segments[1].variance:.2f}",
        "last_variance": f"{found.segments[-1].variance:.2f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    theme.style_axes(axes)
    palette = theme.active_palette()
    theme.points(
        axes,
        list(range(len(series))),
        series,
        color=palette.observed,
        size=10.0,
        alpha=0.8,
    )
    for segment in found.segments:
        xs = [float(segment.start), float(segment.stop - 1)]
        spread = 2.0 * math.sqrt(segment.variance)
        theme.band(
            axes,
            xs,
            [segment.mean - spread] * 2,
            [segment.mean + spread] * 2,
            color=palette.model,
        )
        theme.line(axes, xs, [segment.mean] * 2, color=palette.model, zorder=5)
    for tau in (300, 600):
        theme.reference_line(axes, x=tau - 0.5, entity="truth")
    axes.set_xlim(0, len(series))
    theme.axis_labels(axes, "Time index", "Value")
    theme.note(
        axes,
        "true changes",
        xy=(
            599.5,
            axes.get_ylim()[0] + 0.1 * (axes.get_ylim()[1] - axes.get_ylim()[0]),
        ),
        offset=(-96.0, 0.0),
    )
    theme.title_block(
        axes,
        f"PELT places the changes at {facts['changes']} and finds the variance "
        f"rising from {facts['first_variance']} to {facts['last_variance']}",
        "Dots: 900 observations. Line and band: segment mean +/- 2 sd "
        "(PELT, mean-and-variance cost, MBIC penalty)",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Stochastic processes and random fields
# --------------------------------------------------------------------------


def _draw_sde_convergence() -> tuple[Any, dict[str, str]]:
    """Strong error of Euler-Maruyama and Milstein on Higham's GBM.

    References:
        [EQ-PROC-SDE-003], [EQ-PROC-SDE-004], [REF-HIGHAM-2001].

    """
    sde = ScalarSDE.geometric_brownian_motion(2.0, 1.0)
    gbm = GeometricBrownianMotion(2.0, 1.0)

    def exact(brownian: float) -> float:
        return gbm.exact(1.0, 1.0, brownian)

    studies = {
        "Euler-Maruyama": strong_convergence_study(
            sde, 1.0, 1.0, exact, paths=500, seed=1
        ),
        "Milstein": strong_convergence_study(
            sde, 1.0, 1.0, exact, paths=500, scheme="milstein", seed=1
        ),
    }
    facts = {
        "euler": f"{studies['Euler-Maruyama'].order:.3f}",
        "milstein": f"{studies['Milstein'].order:.3f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    theme.style_axes(axes)
    palette = theme.active_palette()
    labels = []
    styles = {
        "Euler-Maruyama": (theme.role_colour("series-5"), "o", 0.5, "slope 1/2"),
        "Milstein": (theme.role_colour("series-6"), "s", 1.0, "slope 1"),
    }
    for name, study in studies.items():
        colour, marker, order, guide_label = styles[name]
        h = list(study.step_sizes)
        errors = list(study.errors)
        axes.errorbar(
            h,
            errors,
            yerr=[2.0 * s for s in study.standard_errors],
            fmt="none",
            ecolor=colour,
            elinewidth=1.2,
            capsize=0,
            zorder=3,
        )
        theme.line(axes, h, errors, color=colour, zorder=3)
        theme.points(axes, h, errors, color=colour, marker=marker, size=40.0)
        anchor = errors[0] / h[0] ** order
        theme.line(
            axes,
            h,
            [anchor * x**order for x in h],
            color=palette.muted,
            dashes=theme.GUIDE,
            width=theme.CONTEXT_WIDTH,
            zorder=2,
        )
        labels.append((h[0], errors[0], name, colour))
        axes.annotate(
            guide_label,
            xy=(h[-1], anchor * h[-1] ** order),
            xytext=(-6.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.muted,
        )
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.set_xlim(1 / 300, 1 / 10)
    steps = studies["Milstein"].step_sizes
    axes.set_xticks(list(steps))
    axes.set_xticklabels([f"1/{round(1 / step)}" for step in steps])
    axes.set_yticks([0.1, 0.2, 0.5, 1.0])
    axes.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    axes.minorticks_off()
    theme.direct_labels(axes, labels, gap_points=14.0)
    theme.axis_labels(axes, "Step size h", "Strong error E|X_N - X(T)|")
    theme.title_block(
        axes,
        f"Milstein converges at order {facts['milstein']} and Euler-Maruyama "
        f"at {facts['euler']}, close to the theoretical 1 and 1/2",
        "Markers: mean error over 500 Brownian paths, whiskers +/- 2 Monte "
        "Carlo s.e. Grey dots: theoretical slopes",
    )
    return figure, facts


def _draw_kl_fields() -> tuple[Any, dict[str, str]]:
    """Realizations of a rough and a smooth Gaussian field on [0, 1].

    References:
        [EQ-PROC-KL-001], [EQ-PROC-KL-003], [REF-GHANEM-SPANOS-1991].

    """
    plt = theme.pyplot()
    figure, panels = plt.subplots(1, 2, sharey=True, figsize=FIGSIZE)
    palette = theme.active_palette()
    grid = [k / 200.0 for k in range(201)]
    facts: dict[str, str] = {}
    for axes, family, heading in (
        (panels[0], "exponential", "Exponential kernel"),
        (panels[1], "squared_exponential", "Squared-exponential kernel"),
    ):
        theme.style_axes(axes)
        kl = karhunen_loeve_interval(
            CovarianceKernel(family, 1.0, 0.3), 0.0, 1.0, 30, variance_fraction=0.95
        )
        facts[family] = f"{kl.terms} modes"
        nodes = [float(point[0]) for point in kl.points]
        spread = [2.0 * math.sqrt(v) for v in kl.pointwise_variance()]
        theme.band(
            axes,
            nodes,
            [-s for s in spread],
            spread,
            color=palette.observed,
            alpha=0.12,
        )
        generator = RandomStream(41).child("guide", family).generator()
        for index in range(12):
            xi = [generator.gauss(0.0, 1.0) for _ in range(kl.terms)]
            values = [kl.realize_at(xi, x) for x in grid]
            if index < 2:
                theme.line(axes, grid, values, color=palette.series[2 + index])
            else:
                theme.line(
                    axes,
                    grid,
                    values,
                    color=palette.muted,
                    width=theme.PATH_WIDTH,
                    zorder=2,
                ).set_alpha(0.6)
        axes.set_xlim(0.0, 1.0)
        _panel_heading(axes, f"{heading}: {facts[family]}")
        theme.axis_labels(axes, "Location x")
    low, high = panels[0].get_ylim()
    panels[0].set_ylim(low, high + 0.22 * (high - low))
    theme.axis_labels(panels[0], None, "Field value")
    theme.title_block(
        panels[0],
        f"The rough exponential field needs {facts['exponential']} for 95% of "
        f"its variance, the smooth one {facts['squared_exponential']}",
        "Lines: 12 truncated KL realizations, two highlighted. Band: +/- 2 sd. "
        "Length scale 0.3",
    )
    figure.subplots_adjust(wspace=0.08)
    return figure, facts


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def _draw_hidden_velocity() -> tuple[Any, dict[str, str]]:
    """Draw the filtered and smoothed estimates of a never-measured velocity.

    References:
        [EQ-FILT-KF-001], [EQ-FILT-RTS-001], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    q, r = 0.01, 4.0
    model = LinearGaussianStateSpace(
        ((1.0, 1.0), (0.0, 1.0)),
        ((q / 3.0, q / 2.0), (q / 2.0, q)),
        ((1.0, 0.0),),
        ((r,),),
        (0.0, 1.0),
        ((1.0, 0.0), (0.0, 0.5)),
    )
    simulation = simulate(model, 150, seed=11)
    filtered = kalman_filter(model, simulation.observations)
    smoothed = rts_smoother(filtered)
    truth = [state[1] for state in simulation.states[1:]]
    filter_states = filtered.filtered[1:]
    smooth_states = smoothed.smoothed[1:]

    def rmse(states: Any) -> float:
        squares = [
            (state.mean[1] - value) ** 2
            for state, value in zip(states, truth, strict=True)
        ]
        return math.sqrt(math.fsum(squares) / len(squares))

    estimate = [state.mean[1] for state in filter_states]
    deviation = [state.standard_deviations[1] for state in filter_states]
    inside = sum(
        abs(m - t) <= 1.96 * s
        for m, s, t in zip(estimate, deviation, truth, strict=True)
    )
    facts = {
        "filter": f"{rmse(filter_states):.3f}",
        "smoother": f"{rmse(smooth_states):.3f}",
        "coverage": f"{inside / len(truth):.0%}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    plot_filter_track(
        list(range(1, len(truth) + 1)),
        estimate,
        deviation,
        truth=truth,
        smoothed=(
            [state.mean[1] for state in smooth_states],
            [state.standard_deviations[1] for state in smooth_states],
        ),
        name="velocity",
        ax=axes,
    )
    _restate(
        axes,
        f"Velocity is never measured, yet the filter tracks it with RMSE "
        f"{facts['filter']} and the smoother with {facts['smoother']}",
        f"Band: filtered 95% interval, which holds the truth at "
        f"{facts['coverage']} of 150 steps. Only position is observed (r = 4)",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Causal inference
# --------------------------------------------------------------------------


def _confounded(
    n: int, seed: int
) -> tuple[list[tuple[float, float]], list[int], list[float]]:
    """Draw the confounded data of the guide (true ATE 2).

    ``X1, X2 ~ N(0, 1)``, ``logit e = -0.3 + 0.8 X1 - 0.5 X2``,
    ``Y = 1 + 2 T + 1.5 X1 + X2 + 0.5 T X1 + N(0, 1)``; the same recipe and
    stream as ``simulate`` in ``tests/test_causal_inference.py``.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 13]. No equation is evaluated.

    """
    generator = RandomStream(seed).child("causal-simulation").generator()
    rows: list[tuple[float, float]] = []
    treatment: list[int] = []
    outcome: list[float] = []
    for _ in range(n):
        x1, x2 = generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)
        score = 1.0 / (1.0 + math.exp(-(-0.3 + 0.8 * x1 - 0.5 * x2)))
        t = 1 if generator.random() < score else 0
        y = 1.0 + 2.0 * t + 1.5 * x1 + x2 + 0.5 * t * x1 + generator.gauss(0.0, 1.0)
        rows.append((x1, x2))
        treatment.append(t)
        outcome.append(y)
    return rows, treatment, outcome


def _draw_double_robustness() -> tuple[Any, dict[str, str]]:
    """Effect estimates with correct and with misspecified working models.

    References:
        [EQ-CAUSAL-REG-001], [EQ-CAUSAL-SW-001], [EQ-CAUSAL-AIPW-001],
        [REF-HERNAN-ROBINS-2020, Chapter 13].

    """
    rows, t, y = _confounded(2000, 20261012)
    only_x2 = [(row[1],) for row in rows]
    treated = [yi for yi, ti in zip(y, t, strict=True) if ti]
    control = [yi for yi, ti in zip(y, t, strict=True) if not ti]
    naive = math.fsum(treated) / len(treated) - math.fsum(control) / len(control)

    def entry(result: Any) -> tuple[float, float | None, float | None]:
        if math.isnan(result.standard_error):
            return (result.estimate, None, None)
        lower, upper = result.confidence_interval(0.95)
        return (result.estimate, lower, upper)

    aipw = estimate_aipw(rows, t, y)
    both_wrong = estimate_aipw(only_x2, t, y)
    one_wrong = (
        estimate_aipw(rows, t, y, outcome_covariates=only_x2),
        estimate_aipw(rows, t, y, propensity_covariates=only_x2),
    )
    estimates = {
        "naive difference": (naive, None, None),
        "regression adjustment": entry(estimate_regression_adjustment(rows, t, y)),
        "IPW, stabilized": entry(estimate_ipw(rows, t, y, stabilized=True)),
        "AIPW": entry(aipw),
        "regression, X1 omitted": entry(estimate_regression_adjustment(only_x2, t, y)),
        "IPW, X1 omitted": entry(estimate_ipw(only_x2, t, y, stabilized=True)),
        "AIPW, outcome omits X1": entry(one_wrong[0]),
        "AIPW, propensity omits X1": entry(one_wrong[1]),
        "AIPW, both omit X1": entry(both_wrong),
    }
    worst_one = max(abs(result.estimate - 2.0) for result in one_wrong)
    facts = {
        "naive": f"{naive:.2f}",
        "one_wrong": f"{worst_one:.2f}",
        "both_wrong": f"{both_wrong.estimate:.2f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    plot_effect_estimates(estimates, truth=2.0, truth_label="true ATE 2", ax=axes)
    _restate(
        axes,
        f"AIPW stays within {facts['one_wrong']} of the true effect 2 when one "
        f"model omits X1; with both wrong it reports {facts['both_wrong']}",
        "Dot: estimate on 2000 confounded units. Whisker: 95% interval; no "
        "whisker: no interval",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Visual language
# --------------------------------------------------------------------------


def _draw_encoding_rules() -> tuple[Any, dict[str, str]]:
    """Observed, fitted, projected, and true lines on one regression.

    References:
        [EQ-LM-PRED-001], [REF-MATPLOTLIB].

    """
    generator = RandomStream(24).child("visual-language").generator()
    xs = sorted(6.0 * generator.random() for _ in range(25))
    ys = [1.0 + 0.5 * x + generator.gauss(0.0, 0.6) for x in xs]
    fit = fit_linear_model([(1.0, x) for x in xs], ys, column_names=("1", "x"))
    inside = [xs[0] + (xs[-1] - xs[0]) * k / 60 for k in range(61)]
    beyond = [xs[-1] + (10.0 - xs[-1]) * k / 40 for k in range(41)]
    grid = inside + beyond[1:]
    mean = fit.interval([(1.0, x) for x in grid], level=0.95)
    widths = [u - lo for lo, u in zip(mean.lower, mean.upper, strict=True)]
    narrowest = min(widths[: len(inside)])
    facts = {"widening": f"{widths[-1] / narrowest:.1f}x", "n": f"{len(xs)}"}
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    theme.style_axes(axes)
    palette = theme.active_palette()
    theme.band(axes, grid, mean.lower, mean.upper, color=palette.model)
    count = len(inside)
    theme.line(
        axes, grid[:count], mean.mean[:count], color=palette.model, dashes=theme.FITTED
    )
    theme.line(
        axes,
        grid[count - 1 :],
        mean.mean[count - 1 :],
        color=palette.model,
        dashes=theme.PROJECTED,
    )
    truth = [1.0 + 0.5 * x for x in (0.0, 10.0)]
    theme.line(axes, [0.0, 10.0], truth, entity="truth", zorder=2)
    theme.points(axes, xs, ys, entity="observations")
    theme.reference_line(axes, x=xs[-1], color=palette.muted)
    axes.set_xlim(0.0, 10.0)
    theme.axis_labels(axes, "x", "y")
    theme.direct_labels(
        axes,
        [
            (10.0, mean.upper[-1], "95% band", palette.model),
            (10.0, mean.mean[-1], "projected", palette.model),
            (10.0, truth[-1], "truth", theme.entity_colour("truth")),
        ],
    )
    theme.note(
        axes,
        "fitted (dashed)",
        xy=(inside[count // 2], mean.mean[count // 2]),
        offset=(30.0, -60.0),
    )
    theme.note(
        axes, "last observation", xy=(xs[-1], axes.get_ylim()[0]), offset=(14.0, 18.0)
    )
    theme.title_block(
        axes,
        f"Past the last observation the mean turns dotted and its 95% band "
        f"grows {facts['widening']} wider by x = 10",
        f"Dots: {facts['n']} observations (data hue). Model hue: least-squares "
        "mean and its band. Accent dashes: the true line",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Missing data
# --------------------------------------------------------------------------

_MISSING_MEAN = (1.0, 2.0, -1.0)
_MISSING_COVARIANCE = ((1.0, 0.6, 0.3), (0.6, 1.5, 0.4), (0.3, 0.4, 0.8))


def _incomplete_sample(n: int, seed: int) -> tuple[tuple[float | None, ...], ...]:
    """Draw the MAR/MCAR sample of ``examples/missing_data_lab.py``.

    ``x1`` is missing with probability ``logistic(-0.5 + 2 (x0 - 1))`` (MAR
    given ``x0``) and ``x2`` with probability 0.2 (MCAR).

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    full = MultivariateNormal(_MISSING_MEAN, _MISSING_COVARIANCE).sample(n, seed=seed)
    generator = random.Random(seed + 10_000)
    rows: list[tuple[float | None, ...]] = []
    for x in full:
        drop_1 = generator.random() < 1.0 / (1.0 + math.exp(0.5 - 2.0 * (x[0] - 1.0)))
        drop_2 = generator.random() < 0.2
        rows.append((x[0], None if drop_1 else x[1], None if drop_2 else x[2]))
    return tuple(rows)


def _mean_interval(values: list[float]) -> tuple[float, float, float]:
    """Return the mean and the 95% t interval of a column.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No registered equation.

    """
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    half = student_t_quantile(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


@functools.cache
def _coverage_study(reps: int) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Intervals for E[x1] by three methods over repeated MAR samples.

    Per sample (``n = 80``, seed ``5000 + rep``): the complete-case and
    mean-imputation t intervals, and multiple imputation (``m = 5``,
    burn-in 20, stream ``rep``) pooled by Rubin's rules with Barnard-Rubin
    degrees of freedom; the data seeds are those of the lab's coverage study,
    which uses ``m = 10``.

    References:
        [EQ-MISS-005], [EQ-MISS-007], [EQ-MISS-008], [REF-RUBIN-1987].

    """
    out: dict[str, list[tuple[float, float, float]]] = {
        "complete cases": [],
        "mean imputation": [],
        "multiple imputation": [],
    }
    for rep in range(reps):
        rows = _incomplete_sample(80, 5000 + rep)
        cases = [float(row[1]) for row in complete_cases(rows) if row[1] is not None]
        out["complete cases"].append(_mean_interval(cases))
        imputed_means = [float(row[1]) for row in mean_impute(rows)]
        out["mean imputation"].append(_mean_interval(imputed_means))
        fit = fit_incomplete_multivariate_normal(rows, standard_errors=False)
        imputed = impute_multivariate_normal(
            rows, 5, burn_in=20, start=fit, seed=RandomStream(rep)
        )
        per_set = [_mean_interval([float(r[1]) for r in d]) for d in imputed.datasets]
        variances = [
            ((upper - est) / student_t_quantile(0.975, 79)) ** 2
            for est, _, upper in per_set
        ]
        pooled = pool_estimates(
            [est for est, _, _ in per_set], variances, complete_data_df=79
        )
        out["multiple imputation"].append((pooled.estimate, pooled.lower, pooled.upper))
    return {key: tuple(value) for key, value in out.items()}


def _draw_missing_coverage() -> tuple[Any, dict[str, str]]:
    """Coverage of three interval methods for E[x1] under MAR.

    References:
        [EQ-MISS-007], [REF-LITTLE-RUBIN-2019].

    """
    reps = 60
    study = _coverage_study(reps)
    truth = _MISSING_MEAN[1]
    facts: dict[str, str] = {}
    plt = theme.pyplot()
    figure, panels = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(7.2, 4.4))
    palette = theme.active_palette()
    miss_colour = theme.role_colour("series-5")
    names = {
        "complete cases": "Complete cases",
        "mean imputation": "Mean imputation",
        "multiple imputation": "Multiple imputation",
    }
    for axes, (key, intervals) in zip(panels, study.items(), strict=True):
        theme.style_axes(axes)
        ordered = sorted(intervals, key=lambda item: item[0])
        hits = sum(lower <= truth <= upper for _, lower, upper in ordered)
        facts[key] = f"{hits / reps:.0%}"
        for index, (estimate, lower, upper) in enumerate(ordered, start=1):
            covered = lower <= truth <= upper
            colour = palette.muted if covered else miss_colour
            axes.plot(
                [lower, upper],
                [index, index],
                color=colour,
                linewidth=theme.CONTEXT_WIDTH if covered else 1.8,
                solid_capstyle="butt",
                zorder=3,
            )
            axes.plot(
                [estimate], [index], marker="o", markersize=2.6, color=colour, zorder=4
            )
        theme.line(axes, [truth, truth], [0.0, reps + 1.0], entity="truth", zorder=5)
        axes.grid(axis="y", visible=False)
        axes.tick_params(axis="y", labelleft=False)
        _panel_heading(axes, f"{names[key]}\ncovers {facts[key]}")
        theme.axis_labels(axes, "Mean of x1")
    panels[0].set_ylim(0, reps + 11)
    theme.title_block(
        panels[0],
        f"95% intervals for the mean of x1 cover it in {facts['complete cases']} "
        f"of samples with complete cases, {facts['mean imputation']} with mean "
        f"imputation, {facts['multiple imputation']} with multiple imputation",
        f"One row per MAR sample ({reps} samples, n = 80; m = 5 imputations), "
        "sorted by estimate. Coloured: misses the true mean 2",
    )
    figure.subplots_adjust(wspace=0.12)
    return figure, facts


# --------------------------------------------------------------------------
# Experimental design
# --------------------------------------------------------------------------


def _draw_equivalence() -> tuple[Any, dict[str, str]]:
    """D-optimal quadratic design against the uniform grid design.

    References:
        [EQ-OED-KW-001], [EQ-OED-MULT-001], [REF-KIEFER-WOLFOWITZ-1960].

    """
    basis = polynomial_basis(2)
    grid = [(round(-1.0 + 0.1 * k, 10),) for k in range(21)]
    result = d_optimal_design(grid, basis)
    design = result.design.condensed(min_weight=1e-3, merge_distance=0.05)
    uniform = ApproximateDesign.uniform(grid)
    region = [(-1.0 + 2.0 * k / 400,) for k in range(401)]
    optimal_peak = max(standardized_variances(design, basis, region))
    uniform_variances = standardized_variances(uniform, basis, region)
    uniform_peak = max(uniform_variances)
    facts = {
        "optimal": f"{optimal_peak:.3f}",
        "uniform": f"{uniform_peak:.2f}",
        "bound": f"{math.exp(1.0 - uniform_peak / 3.0):.3f}",
    }
    figure = plot_optimal_design(
        design, basis, [x for (x,) in region], comparison=uniform
    )
    _restate(
        figure.axes[0],
        f"Three points with weight 1/3 hold max d(x) at {facts['optimal']} = p; "
        f"the uniform 21-point design peaks at {facts['uniform']}",
        "Top: design weights; grey: uniform design. Bottom: standardized variance d(x)",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Survival analysis
# --------------------------------------------------------------------------


def _draw_censoring() -> tuple[Any, dict[str, str]]:
    """Kaplan-Meier and a censored Weibull fit against ignoring censoring.

    References:
        [EQ-SURV-KM-001], [EQ-SURV-AFT-001], [REF-KAPLAN-MEIER-1958].

    """
    generator = RandomStream(2026).generator()
    times: list[float] = []
    events: list[bool] = []
    for _ in range(500):
        lifetime = 5.0 * (-math.log(1.0 - generator.random())) ** (1.0 / 1.7)
        censor = 14.0 * generator.random()
        times.append(min(lifetime, censor))
        events.append(lifetime <= censor)
    data = survival_data(times, events)
    km = kaplan_meier(data)
    naive = kaplan_meier(survival_data(times, [True] * len(times)))
    weibull = fit_parametric_survival(data, "weibull")
    shape, _, scale, _ = weibull.weibull_parameters()
    weibull_median = scale * math.log(2.0) ** (1.0 / shape)
    facts = {
        "censored": f"{data.censored_count}",
        "km": f"{km.median().estimate:.2f}",
        "naive": f"{naive.median().estimate:.2f}",
        "weibull": f"{weibull_median:.2f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    theme.style_axes(axes)
    palette = theme.active_palette()
    xs = [0.0, *km.times]
    axes.fill_between(
        xs,
        [1.0, *km.lower],
        [1.0, *km.upper],
        step="post",
        color=palette.observed,
        alpha=theme.BAND_ALPHA,
        linewidth=0.0,
    )
    axes.step(
        xs, [1.0, *km.survival], where="post", color=palette.observed, linewidth=2.0
    )
    axes.step(
        [0.0, *naive.times],
        [1.0, *naive.survival],
        where="post",
        color=palette.muted,
        linewidth=theme.CONTEXT_WIDTH,
    )
    grid = [14.0 * k / 200 for k in range(201)]
    fitted = [weibull.survival(t) for t in grid]
    theme.line(axes, grid, fitted, entity="Weibull fit")
    theme.reference_line(axes, y=0.5, entity="threshold")
    axes.set_xlim(0.0, 14.0)
    axes.set_ylim(0.0, 1.03)
    theme.axis_labels(axes, "Time", "Survival probability")

    def crossing(curve_times: Any, curve: Any, level: float) -> float:
        return float(
            next(t for t, s in zip(curve_times, curve, strict=True) if s <= level)
        )

    for text, colour, (x, y), side in (
        (
            "Kaplan-Meier",
            palette.observed,
            (crossing(km.times, km.survival, 0.85), 0.85),
            1.0,
        ),
        (
            "Weibull fit",
            theme.entity_colour("Weibull fit"),
            (crossing(grid, fitted, 0.65), 0.65),
            1.0,
        ),
        (
            "censoring ignored",
            palette.muted,
            (crossing(naive.times, naive.survival, 0.3), 0.3),
            -1.0,
        ),
    ):
        axes.annotate(
            text,
            xy=(x, y),
            xytext=(10.0 * side, 0.0),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=theme.ANNOTATION_SIZE,
            fontweight="bold",
            color=colour,
        )
    theme.note(axes, "median", xy=(0.6, 0.5), offset=(10.0, -16.0))
    theme.title_block(
        axes,
        f"Ignoring {facts['censored']} censored times pulls the median from "
        f"{facts['km']} to {facts['naive']}; the Weibull fit gives "
        f"{facts['weibull']}",
        "Step and band: Kaplan-Meier, 95% log-log. Grey: every time as an "
        "event. 500 Weibull(1.7, 5) lifetimes",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Extreme values
# --------------------------------------------------------------------------

_PROFILE_PERIODS = (2.0, 5.0, 10.0, 30.0, 100.0)


@functools.cache
def _gev_study() -> tuple[Any, tuple[float, ...], tuple[Any, ...]]:
    """Fit the 80 seeded GEV maxima and profile five return levels.

    References:
        [EQ-EVA-RL-002], [EQ-EVA-PROF-001], [REF-COLES-2001, Section 3.3].

    """
    maxima = GeneralizedExtremeValue(10.0, 2.0, 0.1).sample(80, seed=3)
    fit = fit_gev(maxima)
    profiles = tuple(fit.profile_return_level(period) for period in _PROFILE_PERIODS)
    return fit, tuple(maxima), profiles


def _draw_return_levels() -> tuple[Any, dict[str, str]]:
    """GEV return levels with the delta band and profile intervals.

    References:
        [EQ-EVA-RL-001], [EQ-EVA-RL-002], [EQ-EVA-PROF-001], [REF-COLES-2001].

    """
    fit, maxima, profiles = _gev_study()
    record = len(maxima)
    periods = [10.0 ** (0.03 + 2.37 * k / 120) for k in range(121)]
    delta = [fit.return_level(period) for period in periods]
    hundred_delta = fit.return_level(100.0)
    hundred_profile = profiles[-1]
    facts = {
        "profile": f"{hundred_profile.lower:.2f} to {hundred_profile.upper:.2f}",
        "delta": f"{hundred_delta.lower:.2f} to {hundred_delta.upper:.2f}",
        "truth": f"{gev_return_level(10.0, 2.0, 0.1, 100.0):.2f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(figsize=FIGSIZE)
    theme.style_axes(axes)
    palette = theme.active_palette()
    model = theme.entity_colour("GEV fit")
    theme.band(
        axes,
        periods,
        [entry.lower for entry in delta],
        [entry.upper for entry in delta],
        color=model,
    )
    split = sum(period <= record for period in periods)
    levels = [entry.level for entry in delta]
    theme.line(axes, periods[:split], levels[:split], entity="GEV fit")
    theme.line(
        axes,
        periods[split - 1 :],
        levels[split - 1 :],
        color=model,
        dashes=theme.PROJECTED,
    )
    truth = [gev_return_level(10.0, 2.0, 0.1, period) for period in periods]
    theme.line(axes, periods, truth, entity="truth", zorder=2)
    for entry in profiles:
        axes.plot(
            [entry.return_period] * 2,
            [entry.lower, entry.upper],
            color=palette.ink,
            linewidth=1.6,
            zorder=4,
            solid_capstyle="butt",
        )
        for end in (entry.lower, entry.upper):
            axes.plot(
                [entry.return_period / 1.06, entry.return_period * 1.06],
                [end, end],
                color=palette.ink,
                linewidth=1.6,
                zorder=4,
            )
    theme.points(
        axes, hazen_return_periods(record), sorted(maxima), entity="observations"
    )
    axes.set_xscale("log")
    axes.set_xlim(1.0, periods[-1])
    axes.set_xticks([2, 5, 10, 20, 50, 100, 200])
    axes.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    axes.minorticks_off()
    theme.reference_line(
        axes, x=float(record), color=palette.muted, label="record: 80 years"
    )
    theme.axis_labels(axes, "Return period", "Return level", x_unit="years")
    right = axes.get_xlim()[1]
    theme.direct_labels(
        axes,
        [
            (right, delta[-1].upper, "delta 95%", model),
            (right, levels[-1], "GEV fit", model),
            (right, truth[-1], "truth", theme.entity_colour("truth")),
        ],
    )
    theme.note(
        axes,
        "profile 95%",
        xy=(100.0, hundred_profile.upper),
        offset=(-78.0, 4.0),
    )
    theme.title_block(
        axes,
        f"The 100-year profile interval, {facts['profile']}, is skewed upward; "
        f"the symmetric delta interval is {facts['delta']}",
        "Dots: 80 annual maxima. Band: delta-method 95% interval. Whiskers: "
        "profile-likelihood 95%",
    )
    return figure, facts


FIGURES: list[GuideFigure] = [
    GuideFigure(
        name="time-series-forecast-intervals",
        guide="time-series",
        alt=(
            "Two panels: the last 60 values of a random walk and of an AR(1) "
            "series, each followed by a 20-step forecast with its 95% band"
        ),
        draw=_draw_forecast,
    ),
    GuideFigure(
        name="time-series-garch-volatility",
        guide="time-series",
        alt=(
            "Simulated GARCH returns above the true and the fitted conditional "
            "standard deviation over 1500 time steps"
        ),
        draw=_draw_garch,
    ),
    GuideFigure(
        name="time-series-pelt-segments",
        guide="time-series",
        alt=(
            "A 900-point series with three segments, their PELT means and "
            "two-standard-deviation bands, and the true change points"
        ),
        draw=_draw_pelt,
    ),
    GuideFigure(
        name="stochastic-processes-strong-convergence",
        guide="stochastic-processes-and-random-fields",
        alt=(
            "Strong error of the Euler-Maruyama and Milstein schemes against "
            "the step size on log axes with theoretical slope lines"
        ),
        draw=_draw_sde_convergence,
    ),
    GuideFigure(
        name="stochastic-processes-kl-fields",
        guide="stochastic-processes-and-random-fields",
        alt=(
            "Two panels of truncated Karhunen-Loeve realizations on the unit "
            "interval, for an exponential and a squared-exponential kernel"
        ),
        draw=_draw_kl_fields,
    ),
    GuideFigure(
        name="filtering-hidden-velocity",
        guide="filtering-and-data-assimilation",
        alt=(
            "True velocity, Kalman-filtered velocity with its 95% band, and the "
            "smoothed velocity over 150 time steps"
        ),
        draw=_draw_hidden_velocity,
    ),
    GuideFigure(
        name="causal-double-robustness",
        guide="causal-inference",
        alt=(
            "Forest plot of nine average-treatment-effect estimates with 95% "
            "intervals against the true effect of 2"
        ),
        draw=_draw_double_robustness,
    ),
    GuideFigure(
        name="visual-language-encoding-rules",
        guide="visual-language",
        alt=(
            "Regression observations with a dashed fitted mean, a dotted "
            "projection, its 95% band, and the dashed true line"
        ),
        draw=_draw_encoding_rules,
    ),
    GuideFigure(
        name="missing-data-interval-coverage",
        guide="missing-data",
        alt=(
            "Three panels of 95% intervals for the mean of x1 over 60 samples: "
            "complete cases, mean imputation, and multiple imputation"
        ),
        draw=_draw_missing_coverage,
    ),
    GuideFigure(
        name="experimental-design-equivalence",
        guide="experimental-design",
        alt=(
            "Design weights above the standardized variance function of the "
            "D-optimal quadratic design and of the uniform grid design"
        ),
        draw=_draw_equivalence,
    ),
    GuideFigure(
        name="survival-censoring-bias",
        guide="survival-analysis",
        alt=(
            "Kaplan-Meier curve with its 95% band, a fitted Weibull survival "
            "curve, and the curve that treats every time as an event"
        ),
        draw=_draw_censoring,
    ),
    GuideFigure(
        name="extreme-value-return-levels",
        guide="extreme-value-analysis",
        alt=(
            "GEV return-level curve against the return period on a log axis "
            "with the delta band, profile intervals, maxima, and truth"
        ),
        draw=_draw_return_levels,
    ),
]
