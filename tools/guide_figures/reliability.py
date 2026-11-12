"""Figures of the reliability, systems, risk, and cost guides (ADR-0105).

Every figure runs a seeded computation of ``probstat`` and states its
finding in the title: FORM geometry and the method comparisons of the
structural-reliability guides, design points and AK-MCS learning, exact
system reliability and importance, Markov availability and standby,
loss exceedance, the risk register and FMECA, the cost S-curve and age
replacement, and the phased lander mission of the systems-engineering
guide.

References:
    [REF-MELCHERS-BECK-2018], [REF-DERKIUREGHIAN-2005],
    [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-RAUSAND-HOYLAND-2004],
    [REF-ROCKAFELLAR-URYASEV-2002], [REF-MIL-STD-1629A-1980],
    [REF-NASA-CEH-2015], [REF-BARLOW-PROSCHAN-1965],
    [REF-ZANG-SUN-TRIVEDI-1999], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Callable, Mapping
from typing import Any

from guide_figures import GuideFigure
from probstat.cost import (
    CostElement,
    LifeCycleCostModel,
    age_replacement_cost_rate,
    life_cycle_cost_risk,
    optimize_age_replacement,
)
from probstat.distributions import LogNormal, Normal, Weibull
from probstat.inference.confidence import wilson_score_interval
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.reliability import (
    LimitState,
    ModelLimitState,
    ak_mcs,
    compare_reliability_methods,
    estimate_directional_failure_probability,
    estimate_failure_probability,
    failure_rate_estimate,
    find_design_points,
    form_hlrf,
    linear_normal_form,
    mixture_importance_sampling,
    multi_point_form,
    sorm,
)
from probstat.risk import (
    DiscreteImpact,
    FailureMode,
    Fmeca,
    Risk,
    RiskRegister,
    assess_risk,
)
from probstat.systems import (
    BetaFactorGroup,
    ExponentialComponent,
    Phase,
    PhasedMission,
    SystemModel,
    SystemStructure,
    k_out_of_n,
    parallel,
    series,
    single_repairable_unit,
    standby_system,
)
from probstat.uq import QuantityOfInterest
from probstat.visualization.reliability_plots import (
    plot_ak_mcs_convergence,
    plot_design_points,
)
from probstat.visualization.theme import (
    ANNOTATION_SIZE,
    CATEGORY_ORDERS,
    METHOD_NAMES,
    active_palette,
    axis_labels,
    axis_title,
    direct_labels,
    entity_colour,
    line,
    new_axes,
    ordered_colours,
    points,
    pyplot,
    reference_line,
    role_colour,
    style_axes,
    title_block,
)
from probstat.visualization.uq import (
    plot_form_geometry,
    plot_loss_exceedance,
    plot_method_comparison,
)

SQRT2 = math.sqrt(2.0)
_Y = (QuantityOfInterest("y", "limit-state value"),)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _retitle(axes: Any, title: str) -> None:
    """Replace the left-aligned assertion title, keeping its style and pad.

    Library plots compute a neutral title; the guide states its own finding
    with the facts its caption quotes. Only the text changes, so the
    subtitle placement of the plot is kept.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    heading = axes._left_title  # the artist set by title_block (loc="left")
    heading.set_text(title[:1].upper() + title[1:])


def _standard_normal_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on independent standard-normal inputs as a limit state.

    References:
        [EQ-REL-MLS-001], [REF-MELCHERS-BECK-2018, Chapter 3].

    """
    names = tuple(f"u{index + 1}" for index in range(dimension))
    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )

    def response(values: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from named inputs.

        References:
            [EQ-REL-MLS-001]. No further equation is evaluated.

        """
        return {"y": function(tuple(values[name] for name in names))}

    model = ProbabilisticModel(inputs, response, _Y)
    return ModelLimitState(model, "y", 0.0, "below")


def _four_branch(u: tuple[float, ...]) -> float:
    """Return the four-branch series system with ``k = 6``.

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011] (benchmark used there).

    """
    x1, x2 = u
    s = 0.1 * (x1 - x2) ** 2
    return min(
        3.0 + s - (x1 + x2) / SQRT2,
        3.0 + s + (x1 + x2) / SQRT2,
        (x1 - x2) + 6.0 / SQRT2,
        (x2 - x1) + 6.0 / SQRT2,
    )


def _grid(
    function: Callable[[tuple[float, ...]], float], extent: float, nodes: int
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate ``G`` on a square grid for the limit-state contour.

    References:
        [EQ-REL-MLS-001]. No further equation is evaluated.

    """
    axis = [-extent + 2.0 * extent * k / (nodes - 1) for k in range(nodes)]
    return axis, axis, [[function((x, y)) for x in axis] for y in axis]


def _dot_axes(axes: Any) -> None:
    """Style a horizontal dot plot: grid on the value axis, rows unruled.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    palette = active_palette()
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=palette.rule, linewidth=0.7)
    axes.tick_params(axis="y", length=0)
    axes.tick_params(axis="x", length=0)


def _plain_log(axis: Any, subs: tuple[float, ...] = (1.0, 2.0, 5.0)) -> None:
    """Label a logarithmic axis with plain numbers (no ``10^k`` notation).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def plain(value: float, _: object) -> str:
        """Return a positional decimal such as ``0.00005`` (no E notation).

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return f"{value:.12f}".rstrip("0").rstrip(".")

    axis.set_major_locator(LogLocator(base=10.0, subs=subs))
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


def _legible(colour: str, minimum: float = 4.5) -> str:
    """Darken (or lighten) a series colour toward the ink until text is legible.

    The WCAG contrast of the colour against the canvas is raised to at least
    ``minimum`` by mixing it with the ink colour in steps of 5%.

    References:
        [REF-W3C-WCAG21-2018]. No equation of the registry is evaluated.

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = active_palette()

    def luminance(rgb: tuple[float, float, float]) -> float:
        """Return the WCAG relative luminance of an sRGB colour.

        References:
            [REF-W3C-WCAG21-2018].

        """
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    canvas = luminance(to_rgb(palette.canvas))
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        weight = step / 20
        mixed = tuple(
            (1 - weight) * b + weight * i for b, i in zip(base, ink, strict=True)
        )
        value = luminance(mixed)  # type: ignore[arg-type]
        light, dark = max(value, canvas), min(value, canvas)
        if (light + 0.05) / (dark + 0.05) >= minimum:
            return str(to_hex(mixed))
    return palette.ink


def _figure(width: float = 7.2, height: float = 4.0) -> Any:
    """Return a themed single axes of the guide-figure size.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return new_axes(figsize=(width, height))


# --------------------------------------------------------------------------
# reliability-analysis
# --------------------------------------------------------------------------

_AFFINE_SEED = 2026


@functools.lru_cache(maxsize=1)
def _affine_runs() -> tuple[Any, Any, Any, tuple[tuple[float, float], ...]]:
    """Run direct MC, affine FORM, and directional sampling on ``3 - u1 - u2``.

    References:
        [EQ-REL-PF-001], [EQ-REL-FORM-001], [EQ-REL-DIR-001],
        [REF-MELCHERS-BECK-2018, Chapters 3-6].

    """
    state = LimitState(lambda u: 3.0 - u[0] - u[1], "sum margin", 2)
    generator = random.Random(_AFFINE_SEED)
    samples = tuple(
        (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)) for _ in range(100_000)
    )
    direct = estimate_failure_probability(state, samples)
    form = linear_normal_form(3.0, (-1.0, -1.0))

    def radial_boundary(direction: tuple[float, ...]) -> float | None:
        """Return the affine failure-entry radius on one unit ray.

        References:
            [EQ-REL-DIR-001].

        """
        projection = direction[0] + direction[1]
        return None if projection <= 0.0 else 3.0 / projection

    directional = estimate_directional_failure_probability(
        radial_boundary, dimension=2, directions=30_000, seed=_AFFINE_SEED
    )
    return direct, form, directional, samples


def draw_form_design_point() -> tuple[Any, dict[str, str]]:
    """FORM geometry of the affine limit state with Monte Carlo samples.

    References:
        [EQ-REL-FORM-001], [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    _, form, _, samples = _affine_runs()
    beta = form.reliability_index
    u1, u2 = form.design_point
    facts = {
        "beta": f"{beta:.2f}",
        "point": f"({u1:.2f}, {u2:.2f})",
        "pf": f"{form.failure_probability:.5f}",
    }
    axes = plot_form_geometry(
        lambda a, b: 3.0 - a - b,
        form.design_point,
        beta,
        samples=samples[:2000],
        extent=4.0,
        grid=81,
    )
    _retitle(
        axes,
        f"The design point {facts['point']} lies beta = {facts['beta']} from "
        f"the origin, so FORM gives Pf = {facts['pf']}",
    )
    return axes.figure, facts


def draw_method_intervals() -> tuple[Any, dict[str, str]]:
    """Direct and directional sampling intervals against the exact value.

    References:
        [EQ-REL-PF-001], [EQ-REL-DIR-001], [REF-WILSON-1927].

    """
    direct, form, directional, _ = _affine_runs()
    mc = direct.failure.confidence_interval
    dr = directional.estimate.confidence_interval
    ratio = (mc.upper - mc.lower) / (dr.upper - dr.lower)
    facts = {
        "ratio": f"{ratio:.1f}",
        "exact": f"{form.failure_probability:.5f}",
        "directional": f"{directional.failure_probability:.5f}",
    }
    axes = plot_method_comparison(
        (
            (
                f"Monte Carlo ({direct.evaluations:,} draws)",
                direct.failure.probability,
                mc.lower,
                mc.upper,
            ),
            (
                f"directional sampling ({directional.directions:,} rays)",
                directional.failure_probability,
                dr.lower,
                dr.upper,
            ),
            ("FORM (exact for a plane)", form.failure_probability, None, None),
        ),
        reference=form.failure_probability,
        reference_label="exact Pf",
        quantity_label=axis_title("failure_probability"),
        title=f"30,000 directional rays give a 95% interval {facts['ratio']}x "
        "narrower than 100,000 direct draws",
        subtitle=f"Dot: estimate · whisker: 95% interval (Wilson for Monte Carlo) "
        f"· g = 3 - u1 - u2, exact Pf = {facts['exact']}",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# advanced-reliability
# --------------------------------------------------------------------------


def _margin_state() -> ModelLimitState:
    """Return the correlated lognormal margin ``R - S <= 0`` of Section 7.

    References:
        [EQ-REL-MLS-001], [EQ-MOD-NATAF-001].

    """
    laws = {"R": LogNormal(1.2, 0.15), "S": LogNormal(0.3, 0.25)}
    inputs = JointInput(
        tuple(RandomInput(name, laws[name]) for name in "RS"),
        ((1.0, 0.4), (0.4, 1.0)),
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": p["R"] - p["S"]},
        (QuantityOfInterest("y", "margin"),),
    )
    return ModelLimitState(model, "y", 0.0, failure="below")


def draw_method_comparison() -> tuple[Any, dict[str, str]]:
    """FORM, SORM, IS, subset simulation, and MC on one model limit state.

    References:
        [EQ-REL-FORM-002], [EQ-REL-SORM-002], [EQ-REL-ISDP-001],
        [EQ-REL-SUBSET-001], [REF-MELCHERS-BECK-2018, Chapters 5-6].

    """
    comparison = compare_reliability_methods(
        _margin_state(), monte_carlo_samples=20_000, seed=2026
    )
    rows = {METHOD_NAMES.get(row.method, row.method): row for row in comparison.rows}
    order = [name for name in CATEGORY_ORDERS["reliability methods"] if name in rows]
    entries: list[tuple[str, float, float | None, float | None]] = []
    for name in order:
        row = rows[name]
        label = f"{name} ({row.evaluations:,} calls)"
        cov = row.coefficient_of_variation
        if name == "Monte Carlo":
            failures = round(row.probability * row.evaluations)
            interval = wilson_score_interval(failures, row.evaluations)
            entries.append((label, row.probability, interval.lower, interval.upper))
        elif cov is not None and math.isfinite(cov):
            spread = 1.96 * cov * row.probability
            entries.append(
                (
                    label,
                    row.probability,
                    row.probability - spread,
                    row.probability + spread,
                )
            )
        else:
            entries.append((label, row.probability, None, None))
    importance = rows["Importance sampling"]
    crude = rows["Monte Carlo"]
    facts = {
        "is_calls": f"{importance.evaluations:,}",
        "is_cov": f"{importance.coefficient_of_variation:.3f}",
        "mc_calls": f"{crude.evaluations:,}",
        "mc_cov": f"{crude.coefficient_of_variation:.2f}",
    }
    axes = plot_method_comparison(
        entries,
        reference=rows["FORM"].probability,
        reference_label="exact Pf (plane in u)",
        log_scale=True,
        quantity_label=axis_title("failure_probability"),
        title=f"Importance sampling reaches c.o.v. {facts['is_cov']} with "
        f"{facts['is_calls']} calls; {facts['mc_calls']} Monte Carlo draws "
        f"leave c.o.v. {facts['mc_cov']}",
        subtitle="Dot: estimate · whisker: 95% interval (1.96 c.o.v.; Wilson for "
        "Monte Carlo) · correlated lognormal R - S",
    )
    _plain_log(axes.xaxis)
    return axes.figure, facts


@functools.lru_cache(maxsize=1)
def _curvature_sweep() -> tuple[
    tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], float
]:
    """Exact, Breitung, and Hohenbichler-Rackwitz ``P_f`` against ``kappa``.

    The parabolic limit state ``G = 3 - u2 + kappa u1^2 / 2`` has the exact
    value ``E[Phi(-(3 + kappa U^2 / 2))]`` by Gauss-Hermite quadrature.

    References:
        [EQ-REL-SORM-001], [EQ-REL-SORM-002], [EQ-REL-SORM-003],
        [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988].

    """
    kappas = tuple(-0.25 + 0.05 * k for k in range(16))
    nodes, weights = gauss_hermite(60)
    exact: list[float] = []
    breitung: list[float] = []
    hohenbichler: list[float] = []
    form_value = math.nan
    for kappa in kappas:
        exact.append(
            math.fsum(
                w * standard_normal_cdf(-(3.0 + 0.5 * kappa * z * z))
                for z, w in zip(nodes, weights, strict=True)
            )
        )
        state = _standard_normal_state(
            lambda u, c=kappa: 3.0 - u[1] + 0.5 * c * u[0] ** 2, 2
        )
        result = sorm(state, form_result=form_hlrf(state, start=(0.0, 1.0)))
        breitung.append(result.breitung_probability)
        hohenbichler.append(result.hohenbichler_rackwitz_probability)
        form_value = result.form.probability
    return kappas, tuple(exact), tuple(breitung), tuple(hohenbichler), form_value


def draw_sorm_curvature() -> tuple[Any, dict[str, str]]:
    """FORM, SORM, and the exact value against the curvature of ``G = 0``.

    References:
        [EQ-REL-SORM-002], [EQ-REL-SORM-003], [REF-BREITUNG-1984],
        [REF-HOHENBICHLER-RACKWITZ-1988].

    """
    kappas, exact, breitung, hohenbichler, form_value = _curvature_sweep()
    at = min(range(len(kappas)), key=lambda k: abs(kappas[k] - 0.2))
    form_error = form_value / exact[at] - 1.0
    hr_error = hohenbichler[at] / exact[at] - 1.0
    facts = {
        "form_error": f"{100.0 * form_error:.0f}%",
        "hr_error": f"{100.0 * hr_error:.1f}%",
    }
    axes = _figure(7.2, 4.2)
    palette = active_palette()
    xs = list(kappas)
    breitung_colour = role_colour("series-3")
    hr_colour = role_colour("series-4")
    # Breitung and HR nearly coincide for kappa > 0: the wider line is drawn
    # first so the narrower one stays visible on top of it.
    line(axes, xs, list(breitung), color=breitung_colour, width=3.2, zorder=2.6)
    line(axes, xs, list(hohenbichler), color=hr_colour, width=1.6, zorder=2.8)
    line(axes, xs, [form_value] * len(xs), entity="FORM", zorder=3.0)
    line(axes, xs, list(exact), entity="closed form", zorder=3.2)
    axes.set_yscale("log")
    _plain_log(axes.yaxis, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0))
    axes.set_xlim(xs[0], xs[-1])
    axes.axvline(0.2, color=palette.muted, linewidth=0.8, zorder=1)
    axes.annotate(
        "kappa = +0.2",
        xy=(0.2, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4.0, -2.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color=palette.muted,
    )
    axis_labels(
        axes,
        "Principal curvature kappa of G = 0 at the design point",
        axis_title("failure_probability"),
    )
    direct_labels(
        axes,
        [
            (xs[-1], exact[-1], "exact (quadrature)", entity_colour("closed form")),
            (xs[-1], form_value, "FORM", entity_colour("FORM")),
            (xs[-1], breitung[-1], "SORM-Breitung", breitung_colour),
            (xs[-1], hohenbichler[-1], "SORM-HR", hr_colour),
        ],
    )
    title_block(
        axes,
        f"FORM ignores curvature: at kappa = +0.2 it overstates Pf by "
        f"{facts['form_error']}, SORM-HR by {facts['hr_error']}",
        "G = 3 - u2 + kappa u1^2 / 2 · beta = 3 for every kappa · SORM evaluated "
        "at 16 curvatures",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# adaptive-reliability
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _four_branch_design_points() -> tuple[Any, Any, Any]:
    """Find the four design points and combine them (multi-point FORM, IS).

    References:
        [EQ-REL-MDP-001], [EQ-REL-MDP-002], [EQ-REL-MDP-003],
        [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """
    state = _standard_normal_state(_four_branch, 2)
    # G has a kink at the origin, where the central-difference gradient
    # vanishes, so the first FORM run starts off-centre.
    first = form_hlrf(state, start=(0.3, 0.1))
    found = find_design_points(state, 6, first=first)
    combined = multi_point_form(found, samples=20_000, seed=1)
    mixture = mixture_importance_sampling(state, found, 4000, seed=3)
    return found, combined, mixture


def draw_design_points() -> tuple[Any, dict[str, str]]:
    """Four design points, their FORM tangents, and mixture samples.

    References:
        [EQ-REL-MDP-001], [EQ-REL-MDP-003],
        [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """
    found, combined, mixture = _four_branch_design_points()
    generator = random.Random(3)
    samples = []
    for _ in range(2000):
        cx, cy = mixture.centers[generator.randrange(len(mixture.centers))]
        samples.append((generator.gauss(cx, 1.0), generator.gauss(cy, 1.0)))
    failed = [_four_branch(u) <= 0.0 for u in samples]
    single = found.forms[0].probability
    facts = {
        "count": f"{len(found.forms)}",
        "beta": f"{found.betas[0]:.2f}",
        "mixture": f"{mixture.probability:.2e}",
        "ratio": f"{mixture.probability / single:.1f}x",
    }
    axes = plot_design_points(
        found,
        limit_state_grid=_grid(_four_branch, 5.0, 161),
        samples=samples,
        failed=failed,
        combined=combined,
        mixture=mixture,
        extent=5.0,
    )
    _retitle(
        axes,
        f"{facts['count']} design points share beta = {facts['beta']}; mixture "
        f"sampling gives Pf = {facts['mixture']}, {facts['ratio']} the "
        "one-point FORM value",
    )
    return axes.figure, facts


@functools.lru_cache(maxsize=1)
def _ak_mcs_run() -> Any:
    """Run AK-MCS on the four-branch system with the same-population MC check.

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    return ak_mcs(
        _standard_normal_state(_four_branch, 2),
        10_000,
        target_cov=1.0,
        compare_with_monte_carlo=True,
        seed=1,
    )


def draw_ak_mcs_learning() -> tuple[Any, dict[str, str]]:
    """AK-MCS estimate and smallest learning value against model calls.

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    result = _ak_mcs_run()
    facts = {
        "calls": f"{result.evaluations}",
        "pf": f"{result.probability:.2e}",
        "population": f"{result.population:,}",
    }
    figure = plot_ak_mcs_convergence(result)
    _retitle(
        figure.axes[0],
        f"AK-MCS reaches the Monte Carlo value Pf = {facts['pf']} with "
        f"{facts['calls']} model calls instead of {facts['population']}",
    )
    return figure, facts


# --------------------------------------------------------------------------
# system-reliability
# --------------------------------------------------------------------------


def draw_k_out_of_n() -> tuple[Any, dict[str, str]]:
    """System reliability of 3-of-3, 2-of-3, and 1-of-3 against ``p``.

    References:
        [EQ-SYS-STRUCT-001], [EQ-SYS-STRUCT-002], [EQ-SYS-BDD-001],
        [REF-RAUSAND-HOYLAND-2004].

    """
    names = ("A", "B", "C")
    structures = {k: SystemStructure(k_out_of_n(k, *names)) for k in (3, 2, 1)}

    def reliability(k: int, p: float) -> float:
        """Return the exact k-out-of-3 reliability at component value ``p``.

        References:
            [EQ-SYS-BDD-001].

        """
        return structures[k].reliability(dict.fromkeys(names, p))

    low, high = 0.05, 0.95
    for _ in range(60):  # bisection of R_2of3(p) - p on (0.05, 0.95)
        middle = 0.5 * (low + high)
        if reliability(2, middle) < middle:
            low = middle
        else:
            high = middle
    crossing = 0.5 * (low + high)
    facts = {
        "crossing": f"{crossing:.2f}",
        "two": f"{reliability(2, 0.9):.3f}",
        "series": f"{reliability(3, 0.9):.3f}",
        "parallel": f"{reliability(1, 0.9):.3f}",
    }
    axes = _figure(7.2, 4.4)
    palette = active_palette()
    grid = [k / 100 for k in range(101)]
    axes.plot(grid, grid, color=palette.muted, linewidth=1.2, zorder=2)
    colours = dict(zip((1, 2, 3), ordered_colours(3), strict=True))
    for k in (1, 2, 3):
        line(axes, grid, [reliability(k, p) for p in grid], color=colours[k])
    # Each label sits on the empty side of its own curve.
    placements = (
        ("parallel (1 of 3)", colours[1], (0.30, reliability(1, 0.30)), (-8, 14)),
        ("2 of 3", colours[2], (0.25, reliability(2, 0.25)), (10, -14)),
        ("series (3 of 3)", colours[3], (0.80, reliability(3, 0.80)), (8, -14)),
        ("one component", palette.muted, (0.55, 0.55), (8, -6)),
    )
    for text, colour, anchor, offset in placements:
        axes.annotate(
            text, xy=anchor, xytext=offset, textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] > 0 else "top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=_legible(colour),
        )  # fmt: skip
    points(axes, [crossing], [crossing], color=colours[2], size=40.0, zorder=5)
    axes.annotate(
        f"p = {facts['crossing']}", xy=(crossing, crossing), xytext=(-8.0, 8.0),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=ANNOTATION_SIZE, color=palette.ink,
    )  # fmt: skip
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.02)
    axis_labels(axes, "Component reliability p", "System reliability")
    title_block(
        axes,
        f"2-out-of-3 beats one unit only for p > {facts['crossing']}; at p = 0.9 "
        f"it gives {facts['two']}",
        "Exact BDD evaluation, three identical independent components · at p = "
        f"0.9: series {facts['series']}, parallel {facts['parallel']}",
    )
    return axes.figure, facts


def _cooling_loop() -> SystemModel:
    """Return the cooling loop of Section 10 with the pump beta-factor CCF.

    References:
        [EQ-SYS-CCF-001], [REF-MOSLEH-ETAL-1998].

    """
    structure = series(parallel("P1", "P2"), "V", k_out_of_n(2, "S1", "S2", "S3"), "C")
    rates = {
        "P1": 1e-4, "P2": 1e-4, "V": 2e-5, "S1": 5e-5, "S2": 5e-5,
        "S3": 5e-5, "C": 1e-5,
    }  # fmt: skip
    components = {name: ExponentialComponent(rate) for name, rate in rates.items()}
    return SystemModel(
        structure, components, ccf_groups=[BetaFactorGroup(("P1", "P2"), 0.05)]
    )


def draw_importance() -> tuple[Any, dict[str, str]]:
    """Birnbaum and criticality importance of every basic event at 1000 h.

    References:
        [EQ-SYS-IMP-001], [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

    """
    measures = sorted(
        _cooling_loop().importance(1000.0).values(),
        key=lambda m: (m.criticality or 0.0, m.birnbaum),
    )
    by_name = {m.component: m for m in measures}
    ccf = by_name["CCF[P1,P2]"]
    valve = by_name["V"]
    facts = {
        "ccf_birnbaum": f"{ccf.birnbaum:.3f}",
        "ccf_criticality": f"{ccf.criticality or 0.0:.3f}",
        "valve_criticality": f"{valve.criticality or 0.0:.3f}",
    }
    axes = _figure(7.2, 4.4)
    _dot_axes(axes)
    birnbaum_colour = role_colour("series-3")
    critical_colour = role_colour("series-4")
    rows = list(range(len(measures)))
    # The two measures of one event are dodged vertically, so equal values
    # (S1: Birnbaum = criticality) stay visible side by side.
    dodge = 0.14
    points(
        axes, [m.birnbaum for m in measures], [r + dodge for r in rows],
        color=birnbaum_colour, size=46.0, marker="o",
    )  # fmt: skip
    points(
        axes, [m.criticality or 0.0 for m in measures], [r - dodge for r in rows],
        color=critical_colour, size=46.0, marker="D",
    )  # fmt: skip
    axes.set_yticks(rows, [m.component for m in measures])
    axes.set_xlim(0.0, 1.08)
    axes.set_ylim(-0.6, len(rows) - 0.4)
    top = len(rows) - 1
    axes.annotate(
        "Birnbaum dR/dp", xy=(measures[top].birnbaum, top + dodge),
        xytext=(0.0, 9.0),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=ANNOTATION_SIZE, fontweight="bold",
        color=_legible(birnbaum_colour),
    )  # fmt: skip
    axes.annotate(
        "criticality", xy=(measures[top].criticality or 0.0, top + dodge),
        xytext=(0.0, 9.0), textcoords="offset points", ha="center",
        va="bottom", fontsize=ANNOTATION_SIZE, fontweight="bold",
        color=_legible(critical_colour),
    )  # fmt: skip
    axis_labels(axes, "Importance measure at t = 1000 h", "Basic event")
    title_block(
        axes,
        f"The pump CCF event has Birnbaum {facts['ccf_birnbaum']}, like a single "
        f"point of failure, but criticality {facts['ccf_criticality']}",
        "Cooling loop with beta = 0.05 on the pumps · rows sorted by criticality "
        "(share of system failures)",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# markov-reliability
# --------------------------------------------------------------------------


def draw_availability_reliability() -> tuple[Any, dict[str, str]]:
    """Point availability and reliability of one repairable unit.

    References:
        [EQ-SYS-MKV-001], [EQ-SYS-MKV-005], [EQ-PROC-CTMC-002].

    """
    unit = single_repairable_unit(0.001, 0.1)
    times = [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0] + [100.0 * k for k in range(1, 31)]
    availability = [value.value for value in unit.availability_curve(times)]
    reliability = [value.value for value in unit.reliability_curve(times)]
    steady = unit.steady_state_availability()
    facts = {
        "availability": f"{unit.availability(1000.0).value:.6f}",
        "reliability": f"{unit.reliability(1000.0).value:.6f}",
    }
    axes = _figure()
    up = role_colour("series-4")
    survive = role_colour("series-3")
    line(axes, times, availability, color=up)
    line(axes, times, reliability, color=survive)
    up_1000 = unit.availability(1000.0).value
    survive_1000 = unit.reliability(1000.0).value
    points(axes, [1000.0], [up_1000], color=up, size=36.0)
    points(axes, [1000.0], [survive_1000], color=survive, size=36.0)
    axes.annotate(
        f"R(1000) = {facts['reliability']}", xy=(1000.0, survive_1000),
        xytext=(10.0, 10.0), textcoords="offset points", ha="left", va="bottom",
        fontsize=ANNOTATION_SIZE, color=survive, fontweight="semibold",
    )  # fmt: skip
    axes.set_xlim(0.0, 3000.0)
    axes.set_ylim(0.0, 1.05)
    axis_labels(axes, axis_title("time"), "Probability", x_unit="h")
    direct_labels(
        axes,
        [
            (times[-1], availability[-1], "availability A(t)", up),
            (times[-1], reliability[-1], "reliability R(t)", survive),
        ],
    )
    title_block(
        axes,
        f"At 1000 h the unit is up with probability {facts['availability']}, yet "
        f"it has run without failure with probability only {facts['reliability']}",
        f"lambda = 0.001/h, mu = 0.1/h · A(t) settles at mu/(lambda + mu) = "
        f"{steady:.6f} within about 50 h",
    )
    return axes.figure, facts


def draw_standby() -> tuple[Any, dict[str, str]]:
    """Reliability of cold, warm, and hot standby with imperfect switching.

    References:
        [EQ-SYS-MKV-007], [EQ-PROC-CTMC-002].

    """
    times = [50.0 * k for k in range(61)]
    rates = (0.0, 0.0005, 0.001)
    names = ("cold", "warm", "hot")
    colours = ordered_colours(3)
    curves = []
    at_1000 = []
    for rate in rates:
        model = standby_system(1, 0.001, rate, 0.98)
        curves.append([value.value for value in model.reliability_curve(times)])
        at_1000.append(model.reliability(1000.0).value)
    facts = {
        "cold": f"{at_1000[0]:.6f}",
        "hot": f"{at_1000[2]:.6f}",
        "single": f"{math.exp(-1.0):.3f}",
    }
    axes = _figure()
    palette = active_palette()
    single = [math.exp(-0.001 * t) for t in times]
    axes.plot(times, single, color=palette.muted, linewidth=1.4, zorder=2)
    items = [(times[-1], single[-1], "one unit, no spare", palette.muted)]
    for name, rate, colour, values, value in zip(
        names, rates, colours, curves, at_1000, strict=True
    ):
        line(axes, times, values, color=colour)
        points(axes, [1000.0], [value], color=colour, size=30.0)
        items.append(
            (times[-1], values[-1], f"{name} (lambda_s = {rate:g})", _legible(colour))
        )
    axes.axvline(1000.0, color=palette.rule, linewidth=1.0, zorder=1)
    axes.set_xlim(0.0, 3000.0)
    axes.set_ylim(0.0, 1.03)
    axis_labels(axes, axis_title("time"), "Reliability R(t)", x_unit="h")
    direct_labels(axes, items, gap_points=12.0)
    title_block(
        axes,
        f"A cold spare lifts R(1000) to {facts['cold']}; a hot spare that ages "
        f"while waiting reaches only {facts['hot']}",
        "One active unit at lambda = 0.001/h, one spare, switch success 0.98 · "
        f"without a spare R(1000) = {facts['single']}",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# risk-analysis
# --------------------------------------------------------------------------


def draw_loss_exceedance() -> tuple[Any, dict[str, str]]:
    """Loss exceedance curve of ``max(Y, 0)`` with VaR and CVaR markers.

    References:
        [EQ-RISK-EXC-001], [EQ-RISK-VAR-001], [EQ-RISK-CVAR-001],
        [REF-ROCKAFELLAR-URYASEV-2002].

    """
    model = ProbabilisticModel(
        JointInput.independent(RandomInput("X", Normal(0.0, 1.0))),
        lambda p: {"Y": p["X"]},
        (QuantityOfInterest("Y", "response"),),
    )
    assessment = assess_risk(
        model, "Y", lambda y: max(y, 0.0), 20_000, design="lhs", seed=5
    )
    var = assessment.value_at_risk[0.95]
    cvar = assessment.conditional_value_at_risk[0.95]
    facts = {
        "var": f"{var:.5f}",
        "cvar": f"{cvar:.5f}",
        "mean": f"{assessment.expected_loss.estimate:.6f}",
    }
    axes = plot_loss_exceedance(
        assessment.losses,
        {0.95: var, 0.99: assessment.value_at_risk[0.99]},
        {0.95: cvar, 0.99: assessment.conditional_value_at_risk[0.99]},
    )
    levels = [0.02 + 0.05 * k for k in range(80)]
    line(
        axes,
        levels,
        [standard_normal_cdf(-level) for level in levels],
        entity="closed form",
        zorder=2.5,
    )
    reference_line(
        axes,
        x=assessment.expected_loss.estimate,
        label=f"expected loss {facts['mean']}",
    )
    axes.set_xlim(0.0, 4.2)
    axes.annotate(
        "closed form 1 - Phi(l)", xy=(3.9, standard_normal_cdf(-3.9)),
        xytext=(-4.0, -10.0), textcoords="offset points", ha="right", va="top",
        fontsize=ANNOTATION_SIZE, color=entity_colour("closed form"),
        fontweight="semibold",
    )  # fmt: skip
    _retitle(
        axes,
        f"At 95% the VaR is {facts['var']} and the worst 5% of losses average "
        f"{facts['cvar']}",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# risk-register-and-fmeca
# --------------------------------------------------------------------------


def _register() -> RiskRegister:
    """Return the three-risk register of the worked example.

    References:
        [EQ-RISK-REG-001], [EQ-RISK-REG-003].

    """
    return RiskRegister(
        (
            Risk(
                "supplier", 0.3, 100.0, mitigation_cost=5.0, mitigated_probability=0.1
            ),
            Risk(
                "test",
                0.2,
                DiscreteImpact((50.0, 200.0), (0.5, 0.5)),
                mitigation_cost=10.0,
                mitigated_impact=60.0,
            ),
            Risk("late", 0.5, 40.0, category="schedule"),
        )
    )


def draw_register_exposure() -> tuple[Any, dict[str, str]]:
    """Exact exceedance of the total exposure before and after mitigation.

    References:
        [EQ-RISK-REG-002], [EQ-RISK-REG-004].

    """
    register = _register()
    before = register.exposure_distribution()
    after = register.exposure_distribution(mitigated=True)
    cost = sum(risk.mitigation_cost for risk in register.risks)
    facts = {
        "p90": f"{before.quantile(0.9):.0f}",
        "p90_mitigated": f"{after.quantile(0.9):.0f}",
        "mean": f"{before.mean:.2f}",
        "mean_mitigated": f"{after.mean:.2f}",
    }
    axes = _figure()
    palette = active_palette()
    colours = {"unmitigated": palette.muted, "mitigated": role_colour("series-3")}
    for name, law in (("unmitigated", before), ("mitigated", after)):
        # T >= 0, so P(T > x) starts at P(T > 0) and steps down at each total.
        xs = [*law.values, 360.0]
        ys = [*(law.exceedance(v) for v in law.values), 0.0]
        axes.step(xs, ys, where="post", color=colours[name], linewidth=2.0, zorder=3)
        points(axes, [law.quantile(0.9)], [0.1], color=colours[name], size=40.0)
    reference_line(axes, y=0.1, label="10% (P90)", entity="threshold")
    # Each name sits just above a flat step of its own curve, where no other
    # line passes (checked against the step values of both curves).
    for text, xy, colour in (
        ("unmitigated", (52.0, before.exceedance(60.0) + 0.015), palette.muted),
        ("mitigated", (42.0, after.exceedance(50.0) + 0.015), colours["mitigated"]),
    ):
        axes.annotate(
            text, xy=xy, ha="left", va="bottom", fontsize=ANNOTATION_SIZE,
            fontweight="bold", color=_legible(colour),
        )  # fmt: skip
    for law, colour, offset in (
        (after, colours["mitigated"], -6.0),
        (before, palette.muted, 6.0),
    ):
        axes.annotate(
            f"P90 = {law.quantile(0.9):.0f}", xy=(law.quantile(0.9), 0.1),
            xytext=(offset, 4.0), textcoords="offset points",
            ha="right" if offset < 0 else "left", va="bottom",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=_legible(colour),
        )  # fmt: skip
    axes.set_xlim(0.0, 360.0)
    axes.set_ylim(0.0, 0.8)
    axis_labels(axes, "Total exposure T", axis_title("exceedance_probability"))
    title_block(
        axes,
        f"Two mitigations costing {cost:.0f} cut the P90 exposure from "
        f"{facts['p90']} to {facts['p90_mitigated']}",
        f"Exact P(T > x) by convolution of three independent risks · mean "
        f"{facts['mean']} before, {facts['mean_mitigated']} after",
    )
    return axes.figure, facts


def draw_fmeca() -> tuple[Any, dict[str, str]]:
    """Criticality number and RPN of the same three failure modes.

    References:
        [EQ-RISK-FMECA-001], [EQ-RISK-FMECA-003], [REF-MIL-STD-1629A-1980],
        [REF-BOWLES-2003].

    """
    sheet = Fmeca(
        (
            FailureMode(
                "pump", "seal leak", 3, 2e-5, 0.6, 0.5, 1000.0,
                severity_rank=5, occurrence_rank=6, detection_rank=3,
            ),
            FailureMode(
                "pump", "seizure", 2, 2e-5, 0.4, 1.0, 1000.0,
                severity_rank=8, occurrence_rank=3, detection_rank=4,
            ),
            FailureMode(
                "valve", "fails closed", 2, 5e-6, 0.7, 1.0, 1000.0,
                severity_rank=8, occurrence_rank=2, detection_rank=6,
            ),
        )
    )  # fmt: skip
    rows = sorted(sheet.mode_criticality(), key=lambda pair: pair[1])
    labels = [f"{mode.item}: {mode.mode}" for mode, _ in rows]
    criticality = [1e3 * value for _, value in rows]
    rpn = [mode.rpn or 0 for mode, _ in rows]
    by_label = dict(zip(labels, rows, strict=True))
    seizure = by_label["pump: seizure"][1]
    valve = by_label["valve: fails closed"][1]
    facts = {
        "rpn": f"{by_label['pump: seizure'][0].rpn}",
        "ratio": f"{seizure / valve:.1f}x",
    }
    plt = pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, sharey=True, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.12}
    )
    palette = active_palette()
    colour = role_colour("series-3")
    positions = list(range(len(rows)))
    for axes, values, label, fmt in (
        (left, criticality, "Criticality C_m [per 1000 missions]", "{:.1f}"),
        (right, rpn, "RPN = S x O x D", "{:.0f}"),
    ):
        style_axes(axes)
        _dot_axes(axes)
        axes.hlines(positions, 0.0, values, color=colour, linewidth=2.0, zorder=2)
        points(axes, values, positions, color=colour, size=50.0)
        for position, value in zip(positions, values, strict=True):
            axes.annotate(
                fmt.format(value), xy=(value, position), xytext=(8.0, 0.0),
                textcoords="offset points", ha="left", va="center",
                fontsize=ANNOTATION_SIZE, color=palette.ink,
            )  # fmt: skip
        axes.set_xlim(0.0, 1.25 * max(values))
        axes.set_xlabel(label)
    left.set_yticks(positions, labels)
    left.set_ylim(-0.6, len(rows) - 0.4)
    right.tick_params(axis="y", labelleft=False)
    title_block(
        left,
        f"RPN ties pump seizure and valve fails-closed at {facts['rpn']}, though "
        f"their criticality differs {facts['ratio']}",
        "Same three failure modes · left: C_m = beta alpha lambda_p t (MIL-STD-1629A)"
        " · right: product of ordinal ranks",
    )
    return figure, facts


# --------------------------------------------------------------------------
# cost-analysis
# --------------------------------------------------------------------------


def draw_cost_s_curve() -> tuple[Any, dict[str, str]]:
    """Monte Carlo and analytic S-curves of the life-cycle cost.

    References:
        [EQ-COST-MOM-001], [EQ-COST-MOM-002], [EQ-COST-SCURVE-001],
        [REF-NASA-CEH-2015].

    """
    model = LifeCycleCostModel(
        (
            CostElement.spread(
                "development", Normal(1000, 100), {0: 0.4, 1: 0.6},
                category="research_development",
            ),
            CostElement.one_time(
                "production", LogNormal(6.0, 0.2), 2, category="acquisition"
            ),
            CostElement.recurring(
                "operations", Normal(50, 5), 3, 7, escalation=0.02,
                category="operation",
            ),
            CostElement.one_time("disposal", 80.0, 8, category="disposal"),
        ),
        discount_rate=0.05,
    )  # fmt: skip
    result = life_cycle_cost_risk(
        model, [[1, 0.5, 0], [0.5, 1, 0.3], [0, 0.3, 1]], samples=20_000, seed=2026
    )
    row80 = next(row for row in result.rows if abs(row.level - 0.8) < 1e-9)
    facts = {
        "p80": f"{row80.monte_carlo:.2f}",
        "p80_independent": f"{row80.analytic_independent:.2f}",
        "point": f"{result.point_estimate:.2f}",
        "reserve": f"{row80.contingency_monte_carlo:.2f}",
    }
    low, high = 1150.0, 2150.0
    costs = [low + (high - low) * k / 240 for k in range(241)]
    axes = _figure()
    palette = active_palette()
    simulated = entity_colour("simulated")
    fitted = entity_colour("LogNormal fit")
    independent = result.independent_law
    if independent is not None:
        axes.plot(
            costs, [independent.cdf(c) for c in costs], color=palette.muted,
            linewidth=1.4, zorder=2,
        )  # fmt: skip
    line(axes, costs, [result.monte_carlo_cdf(c) for c in costs], entity="simulated")
    line(
        axes, costs, [result.analytic_cdf(c) for c in costs], entity="LogNormal fit",
        zorder=3.5,
    )  # fmt: skip
    reference_line(axes, y=0.8, label="P80", entity="threshold")
    level = result.monte_carlo_cdf(result.point_estimate)
    accent = entity_colour("threshold")
    axes.vlines(
        result.point_estimate, 0.0, level, color=accent, linewidth=1.2,
        linestyle=(0, (3.5, 3.0)), zorder=2,
    )  # fmt: skip
    axes.annotate(
        f"point estimate {facts['point']}", xy=(result.point_estimate, 0.04),
        xytext=(6.0, 0.0), textcoords="offset points", ha="left", va="bottom",
        fontsize=ANNOTATION_SIZE, color=accent, fontweight="semibold",
    )  # fmt: skip
    points(axes, [row80.monte_carlo], [0.8], entity="simulated", size=44.0)
    points(axes, [row80.analytic_independent], [0.8], color=palette.muted, size=36.0)
    axes.annotate(
        facts["p80"], xy=(row80.monte_carlo, 0.8), xytext=(10.0, -12.0),
        textcoords="offset points", ha="left", va="top",
        fontsize=ANNOTATION_SIZE, fontweight="bold", color=_legible(simulated),
    )  # fmt: skip
    axes.annotate(
        facts["p80_independent"], xy=(row80.analytic_independent, 0.8),
        xytext=(-8.0, 10.0), textcoords="offset points", ha="right",
        va="bottom", fontsize=ANNOTATION_SIZE, fontweight="bold",
        color=palette.muted,
    )  # fmt: skip
    for text, xy, colour in (
        ("Monte Carlo, correlated", (1760.0, 0.50), simulated),
        ("lognormal with the same moments", (1760.0, 0.42), fitted),
        ("correlation ignored", (1760.0, 0.34), palette.muted),
    ):
        axes.annotate(
            text, xy=xy, ha="left", va="center", fontsize=ANNOTATION_SIZE,
            fontweight="bold", color=_legible(colour),
        )  # fmt: skip
    axes.set_xlim(low, high)
    axes.set_ylim(0.0, 1.02)
    axis_labels(
        axes, "Life-cycle cost, present value", axis_title("cumulative_probability")
    )
    title_block(
        axes,
        f"Ignoring correlation sets the P80 budget at {facts['p80_independent']} "
        f"instead of {facts['p80']}",
        f"20,000 correlated draws (seed 2026) · P80 reserve over the point "
        f"estimate: {facts['reserve']}",
    )
    return axes.figure, facts


def draw_age_replacement() -> tuple[Any, dict[str, str]]:
    """Long-run cost rate of age replacement against the replacement age.

    References:
        [EQ-COST-AGE-001], [REF-BARLOW-PROSCHAN-1965].

    """
    law = Weibull(2.5, 1000.0)
    optimum = optimize_age_replacement(law, 1000.0, 10_000.0)
    ages = [60.0 * 1.05**k for k in range(78)]
    rates = [age_replacement_cost_rate(law, age, 1000.0, 10_000.0) for age in ages]
    facts = {
        "age": f"{optimum.interval:.1f}",
        "rate": f"{optimum.cost_rate:.3f}",
        "run_to_failure": f"{optimum.run_to_failure_cost_rate:.3f}",
        "savings": f"{100.0 * optimum.savings:.0f}%",
    }
    axes = _figure()
    palette = active_palette()
    model = role_colour("model")
    line(axes, ages, rates, color=model)
    reference_line(
        axes, y=optimum.run_to_failure_cost_rate, label="run to failure",
        entity="threshold",
    )  # fmt: skip
    points(axes, [optimum.interval], [optimum.cost_rate], color=model, size=60.0)
    axes.annotate(
        f"T* = {facts['age']} h, {facts['rate']} per h",
        xy=(optimum.interval, optimum.cost_rate), xytext=(16.0, -18.0),
        textcoords="offset points", ha="left", va="top",
        fontsize=ANNOTATION_SIZE, fontweight="bold", color=model,
    )  # fmt: skip
    axes.set_xlim(0.0, ages[-1])
    axes.set_ylim(0.0, 1.15 * max(rates[0], optimum.run_to_failure_cost_rate))
    axes.annotate(
        "replace too early: preventive cost dominates", xy=(ages[0], rates[0]),
        xytext=(10.0, 0.0), textcoords="offset points", ha="left",
        va="center", fontsize=ANNOTATION_SIZE, color=palette.muted,
    )  # fmt: skip
    axis_labels(axes, "Replacement age T", "Cost rate [per hour]", x_unit="h")
    title_block(
        axes,
        f"Replacing at age {facts['age']} h costs {facts['rate']} per hour, "
        f"{facts['savings']} less than running to failure ({facts['run_to_failure']})",
        "Weibull(shape 2.5, scale 1000 h) lifetime · c_p = 1000, c_f = 10 000 · "
        "line: (c_p R(T) + c_f F(T)) / E[cycle length]",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# systems-engineering-guide
# --------------------------------------------------------------------------


def _lander_components(dormant: bool, phases: int, used: set[str]) -> dict[str, Any]:
    """Return the lander components, with or without dormant cruise rates.

    The computers fail at the Jeffreys posterior mean of 3 failures in
    1.2 million hours; the engine and the radar get one model per phase
    (the cruise model at 1% of the active rate when ``dormant``).

    References:
        [EQ-REL-DATA-002], [EQ-SYS-COMP-001], [REF-ZANG-SUN-TRIVEDI-1999].

    """
    computer = failure_rate_estimate(3, 1.2e6).posterior.mean
    cruise = 0.01 if dormant else 1.0
    models: dict[str, Any] = {
        "cpu1": ExponentialComponent(computer),
        "cpu2": ExponentialComponent(computer),
        "cpu3": ExponentialComponent(computer),
        "engine": tuple(
            ExponentialComponent(5e-4 * factor) for factor in (1.0, cruise, 1.0)
        )[:phases],
        "radar": tuple(
            ExponentialComponent(2e-4 * factor) for factor in (1.0, cruise, 1.0)
        )[:phases],
        "transponder": ExponentialComponent(1e-4),
    }
    return {name: model for name, model in models.items() if name in used}


def draw_phased_dormancy() -> tuple[Any, dict[str, str]]:
    """Cumulative mission reliability at the end of launch, cruise, and landing.

    References:
        [EQ-SYS-PMS-001], [REF-ZANG-SUN-TRIVEDI-1999].

    """
    cpus = ("cpu1", "cpu2", "cpu3")
    phases = (
        Phase("launch", 2.0, series("engine", k_out_of_n(3, *cpus))),
        Phase("cruise", 717.0, series(k_out_of_n(1, *cpus), "transponder")),
        Phase("landing", 1.0, series("engine", "radar", k_out_of_n(2, *cpus))),
    )
    used = {"engine", "radar", "transponder", *cpus}
    results = {
        dormant: PhasedMission(
            phases, _lander_components(dormant, len(phases), used)
        ).evaluate()
        for dormant in (True, False)
    }
    dormant_curve = [1.0, *results[True].cumulative_reliability]
    active_curve = [1.0, *results[False].cumulative_reliability]
    facts = {
        "dormant": f"{results[True].reliability:.4f}",
        "active": f"{results[False].reliability:.4f}",
        "cruise": f"{results[True].cumulative_reliability[1]:.4f}",
    }
    axes = _figure()
    palette = active_palette()
    positions = [0, 1, 2, 3]
    colours = {True: role_colour("series-4"), False: role_colour("series-3")}
    names = {
        True: "engine and radar dormant in cruise",
        False: "active rates throughout",
    }
    items = []
    for dormant, values in ((False, active_curve), (True, dormant_curve)):
        colour = colours[dormant]
        line(axes, positions, values, color=colour, zorder=3 if dormant else 2.5)
        # The curves share their first three values; the active curve only
        # marks its own landing value so no marker hides another.
        shown = positions if dormant else positions[-1:]
        points(
            axes,
            shown,
            values[-len(shown) :],
            color=colour,
            size=40.0,
            marker="o" if dormant else "s",
        )
        items.append(
            (
                positions[-1],
                values[-1],
                f"{names[dormant]}: {values[-1]:.4f}",
                _legible(colour),
            )
        )
    axes.annotate(
        f"{facts['cruise']}", xy=(2, dormant_curve[2]), xytext=(0.0, 10.0),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=ANNOTATION_SIZE, color=palette.ink,
    )  # fmt: skip
    axes.set_xticks(
        positions,
        [
            "start",
            "end of launch\n(2 h)",
            "end of cruise\n(719 h)",
            "end of landing\n(720 h)",
        ],
    )
    axes.set_xlim(-0.15, 3.15)
    axes.set_ylim(0.5, 1.03)
    axis_labels(axes, "Mission timeline", "Mission reliability so far")
    direct_labels(axes, items, gap_points=13.0)
    title_block(
        axes,
        f"Ageing the idle engine and radar through cruise drops mission "
        f"reliability at landing from {facts['dormant']} to {facts['active']}",
        "Exact phased-mission BDD, non-repairable components · cruise needs only "
        "1 of 3 computers and the transponder",
    )
    return axes.figure, facts


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

FIGURES: list[GuideFigure] = [
    GuideFigure(
        "reliability-form-design-point",
        "reliability-analysis",
        "Standard-normal plane with the affine limit state, failed and safe "
        "Monte Carlo samples, the beta circle, and the FORM design point",
        draw_form_design_point,
    ),
    GuideFigure(
        "reliability-method-intervals",
        "reliability-analysis",
        "Dot plot of direct Monte Carlo, directional sampling, and FORM failure "
        "probabilities with 95% intervals and the exact value",
        draw_method_intervals,
    ),
    GuideFigure(
        "advanced-reliability-method-comparison",
        "advanced-reliability",
        "Dot plot on a log axis of FORM, SORM, importance sampling, subset "
        "simulation, and Monte Carlo failure probabilities with intervals",
        draw_method_comparison,
    ),
    GuideFigure(
        "advanced-reliability-sorm-curvature",
        "advanced-reliability",
        "Failure probability against limit-state curvature for the exact value, "
        "FORM, and the Breitung and Hohenbichler-Rackwitz SORM formulas",
        draw_sorm_curvature,
    ),
    GuideFigure(
        "adaptive-reliability-design-points",
        "adaptive-reliability",
        "Standard-normal map of the four-branch limit state with four design "
        "points, FORM tangents, the beta circle, and mixture samples",
        draw_design_points,
    ),
    GuideFigure(
        "adaptive-reliability-ak-mcs-learning",
        "adaptive-reliability",
        "Two panels against model calls: the AK-MCS failure probability with "
        "its band and the smallest learning value with the stop threshold",
        draw_ak_mcs_learning,
    ),
    GuideFigure(
        "system-reliability-k-out-of-n",
        "system-reliability",
        "System reliability of series, 2-out-of-3, and parallel structures of "
        "three components against the component reliability",
        draw_k_out_of_n,
    ),
    GuideFigure(
        "system-reliability-importance",
        "system-reliability",
        "Dot plot of Birnbaum and criticality importance for each basic event "
        "of the cooling loop with common-cause failure",
        draw_importance,
    ),
    GuideFigure(
        "markov-availability-vs-reliability",
        "markov-reliability",
        "Point availability and reliability of one repairable unit over 3000 "
        "hours of operation",
        draw_availability_reliability,
    ),
    GuideFigure(
        "markov-standby-reliability",
        "markov-reliability",
        "Reliability over time of cold, warm, and hot standby pairs and of a "
        "single unit without a spare",
        draw_standby,
    ),
    GuideFigure(
        "risk-loss-exceedance",
        "risk-analysis",
        "Loss exceedance curve on a log probability axis with VaR and CVaR "
        "markers, the closed-form tail, and the expected loss",
        draw_loss_exceedance,
    ),
    GuideFigure(
        "risk-register-exposure",
        "risk-register-and-fmeca",
        "Exceedance probability of the total register exposure before and "
        "after mitigation with the P90 values marked",
        draw_register_exposure,
    ),
    GuideFigure(
        "fmeca-criticality-vs-rpn",
        "risk-register-and-fmeca",
        "Two lollipop panels of three failure modes: criticality number on the "
        "left and risk priority number on the right",
        draw_fmeca,
    ),
    GuideFigure(
        "cost-risk-s-curve",
        "cost-analysis",
        "Cumulative probability of the life-cycle cost from Monte Carlo, the "
        "moment-matched lognormal, and the independent approximation",
        draw_cost_s_curve,
    ),
    GuideFigure(
        "cost-age-replacement",
        "cost-analysis",
        "Long-run cost rate of age replacement against the replacement age with "
        "the optimum and the run-to-failure rate",
        draw_age_replacement,
    ),
    GuideFigure(
        "systems-engineering-phased-dormancy",
        "systems-engineering-guide",
        "Mission reliability over launch, cruise, and landing with dormant and "
        "with active failure rates for the engine and radar",
        draw_phased_dormancy,
    ),
]

__all__ = ["FIGURES"]
