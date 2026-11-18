"""Risk-analysis laboratory: a rare failure probability and the guarantees around it.

The four-branch series system (``k = 7``) on two independent standard
normals fails with probability about ``2.22e-3``. Its failure set has four
design points, two at distance 3 and two at distance 3.5, so it separates
methods that look at one point from methods that sample the whole space.
The laboratory estimates the probability by crude Monte Carlo, FORM from two
starting points, multi-point FORM, directional simulation, subset
simulation, and moving particles; it evaluates the ideal cost of
conditioned sampling; and it closes with two guarantees a safety case needs:
a Wilks upper bound on an output quantile and the probability that a
simulation certifies an acceptable failure probability. Run with
``PYTHONPATH=src python examples/risk_analysis_lab.py``; figures are written
to ``examples/generated/``.

Definitions:
    ``g(u) = min(3 + (u1 - u2)^2 / 10 - |u1 + u2| / sqrt(2),
        k / sqrt(2) - |u1 - u2|)``: four-branch limit state, failure when
        ``g <= 0``; ``k = 7``.
    ``P_f = P(g(U) <= 0)`` with ``U`` standard normal in two dimensions;
        reference ``2.2245e-3`` from ``2 x 10^8`` crude Monte Carlo draws.
    ``beta``: distance from the origin to a design point (most probable
        failure point) in standard-normal space.
    c.o.v.: coefficient of variation, standard deviation of an estimator
        over its mean.
    moving particles: ``N`` particles, each move replaces the particle with
        the largest ``g`` by a conditional draw below that level; after
        ``M`` moves the estimate is ``(1 - 1/N)^M``.
    Wilks bound: the order statistic ``Y_(k)`` of ``n`` runs that exceeds
        the ``a``-quantile with probability at least ``beta``.
    certification: deciding ``P_f <= alpha`` from ``n`` runs with a
        false-certification probability of at most ``beta`` when
        ``P_f >= alpha``.

Assumptions:
    - Inputs are independent standard normals, so the standard-normal space
      is the input space and the radial law of directional simulation is the
      chi law with two degrees of freedom.
    - FORM linearizes ``g`` at one design point; ``Phi(-beta)`` is exact only
      for a plane. The origin lies on the kinks of ``|u1 + u2|`` and
      ``|u1 - u2|``, where the finite-difference gradient vanishes.
    - The Poisson interval of moving particles and the splitting formulas
      assume exact conditional sampling; the Markov chains used in practice
      only approximate it.
    - Wilks bounds and certification probabilities need independent runs of
      a deterministic model; they cover sampling error, not errors in the
      input law or the model.

Equations:
    EQ-REL-WAARTS-001: four-branch series limit state.
    EQ-REL-PF-001: failure probability as an indicator expectation.
    EQ-MC-PROB-001: crude Monte Carlo estimate with its Wilson interval.
    EQ-REL-HLRF-001: improved HL-RF iteration of FORM.
    EQ-REL-FORM-002: ``Phi(-beta)`` at a converged design point.
    EQ-REL-MDP-001: search for further design points with bulges.
    EQ-REL-MDP-002: multi-point FORM, the union of tangent half-spaces.
    EQ-REL-DIR-002: directional simulation with every root of each ray.
    EQ-REL-SUBSET-001: subset simulation as a product of conditional
        probabilities.
    EQ-REL-MP-001: moving-particles estimator and its ideal variance.
    EQ-REL-MP-002: exact Poisson interval of moving particles.
    EQ-REL-PCN-001: Crank-Nicolson conditional move.
    EQ-REL-SPLIT-001: c.o.v. of ideal splitting with ``M`` levels.
    EQ-REL-SPLIT-002: optimal level count and level probability.
    EQ-PROP-WILKS-001: Wilks rank of an upper quantile bound.
    EQ-PROP-WILKS-002: Wilks sample size of order ``m``.
    EQ-PROP-WILKS-003: two-sided distribution-free tolerance interval.
    EQ-REL-CERT-001: normal certification margin.
    EQ-REL-CERT-002: exact certification probability as a binomial CDF.
    EQ-REL-CERT-003: certification probability of moving particles.

Diagnostics:
    - Every estimate is compared with the reference ``2.2245e-3``; the
      method-comparison figure draws each 95% interval against it.
    - FORM is run from the origin (it stops with an error), from
      ``(0.5, 0.5)`` and from ``(1, -1)``; the two runs find different design
      points, which is the sign that one point does not describe the set.
    - Moving particles: the number of moves is compared with its ideal mean
      ``-N ln P_f`` and the Markov-chain acceptance rate is printed.
    - Wilks: over 1000 replicates of 1000 normal draws, the share of empirical
      0.95-quantiles and of Wilks bounds that fall below the true quantile is
      compared with 0.5 and with ``1 - 0.9567``.
    - Certification: the false-certification probability at ``P_f = alpha``
      is printed for both rules; it must stay at or below 0.05 for the
      Clopper-Pearson rule.

Cost and error:
    - Crude Monte Carlo: 188 000 model calls over seven sample sizes; at
      ``n = 100 000`` the c.o.v. is about 6%. A 10% c.o.v. needs about
      45 000 calls.
    - FORM: 10 calls from ``(0.5, 0.5)``; the design-point search 1775 calls;
      multi-point FORM adds no model call.
    - Directional simulation: 200 rays, 4507 calls, c.o.v. 0.11. Subset
      simulation: 1000 samples per level, 2580 calls, c.o.v. 0.23. Moving
      particles: ``N + 20 M`` calls, 12 560 for ``N = 100``.
    - Wilks replicates: 10^6 normal draws; the reported shares carry a
      binomial standard error of at most 0.016.
    - Certification probabilities are exact binomial and Poisson tails; no
      sampling error. The whole laboratory runs in about one minute; it
      prints its wall-clock time.

References:
    REF-WAARTS-2000: the four-branch benchmark.
    REF-MELCHERS-BECK-2018: structural reliability methods.
    REF-HASOFER-LIND-1974: the reliability index.
    REF-DERKIUREGHIAN-DAKESSIAN-1998: multiple design points.
    REF-BJERAGER-1988: directional simulation.
    REF-AU-BECK-2001: subset simulation.
    REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011: moving particles.
    REF-WALTER-2015: moving particles and its certification.
    REF-LAGNOUX-2006: optimal splitting.
    REF-WILKS-1941: order-statistic tolerance limits.
    REF-NUTT-WALLIS-2004: Wilks' method in safety analysis.
    REF-CLOPPER-PEARSON-1934: exact binomial bounds.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import dataclasses
import functools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from probstat.distributions import Normal
from probstat.foundations import RandomStream
from probstat.modeling import (
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    four_branch,
    four_branch_model,
)
from probstat.propagation import (
    wilks_bound,
    wilks_rank,
    wilks_sample_size,
    wilks_tolerance_interval,
)
from probstat.reliability import (
    ModelLimitState,
    certification_probability,
    certification_sample_size,
    directional_simulation,
    find_design_points,
    form_hlrf,
    model_monte_carlo,
    moving_particles,
    moving_particles_certification_probability,
    multi_point_form,
    optimal_splitting,
    splitting_coefficient_of_variation,
    subset_simulation,
)
from probstat.visualization import save_figure, theme
from probstat.visualization.uq import plot_method_comparison

SEED = 2026
K = 7.0
REFERENCE = FOUR_BRANCH_REFERENCE_PROBABILITY
MONTE_CARLO_SIZES = (1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000)
DIRECTIONS = 200
PARTICLES = (100, 200)
MARKOV_STEPS = 20
SPLITTING_BUDGET = 10_000
WILKS_RUNS = 1000
WILKS_REPLICATES = 1000
LEVEL = 0.95
RISK = 1e-3
RISK_RATIOS = (0.1, 0.5, 0.9)
OUTPUT = Path(__file__).parent / "generated"
MAP_EXTENT = 5.0


# --------------------------------------------------------------------------
# The limit state, with an optional record of every evaluated point
# --------------------------------------------------------------------------


def four_branch_state(
    record: list[tuple[float, float]] | None = None,
) -> ModelLimitState:
    """Return the four-branch limit state ``g <= 0`` with ``k = K``.

    With ``record`` the library model is wrapped so that every evaluated
    input point is appended to the list before the model runs; the values
    returned are unchanged.

    Args:
        record: Optional list that receives ``(u1, u2)`` of every call.

    Returns:
        Model limit state with failure below the threshold 0.

    References:
        [EQ-REL-WAARTS-001], [REF-WAARTS-2000].

    """
    model = four_branch_model(K)
    if record is not None:
        forward = model.model

        def recorded(parameters: Mapping[str, float]) -> Any:
            """Append the evaluated point, then run the library model.

            References:
                [EQ-REL-WAARTS-001], [REF-WAARTS-2000].

            """
            record.append((float(parameters["u1"]), float(parameters["u2"])))
            return forward(parameters)

        model = dataclasses.replace(model, model=recorded)
    return ModelLimitState(model, "g", 0.0, "below")


def _ratio(value: float) -> str:
    """Format an estimate as a multiple of the reference probability.

    References:
        [REF-WAARTS-2000]. No equation is evaluated.

    """
    return f"{value / REFERENCE:.2f}"


# --------------------------------------------------------------------------
# Estimating the failure probability
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def crude_monte_carlo() -> tuple[Any, ...]:
    """Run crude Monte Carlo at every size of ``MONTE_CARLO_SIZES``.

    Returns:
        One model Monte Carlo result per size, each from its own child seed.

    References:
        [EQ-REL-PF-001], [EQ-MC-PROB-001], [REF-MELCHERS-BECK-2018].

    """
    root = RandomStream(SEED).child("crude-monte-carlo")
    return tuple(
        model_monte_carlo(four_branch_state(), size, seed=root.child(size))
        for size in MONTE_CARLO_SIZES
    )


@dataclass(frozen=True)
class FormStudy:
    """FORM attempts and the multi-point correction on the four-branch set.

    Args:
        origin_message: Error raised when FORM starts at the origin.
        inner: FORM result started at ``(0.5, 0.5)``.
        outer: FORM result started at ``(1, -1)``.
        design_points: All design points found.
        multi_point: Multi-point FORM probability of their union.

    References:
        [EQ-REL-HLRF-001], [EQ-REL-MDP-002], [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """

    origin_message: str
    inner: Any
    outer: Any
    design_points: Any
    multi_point: Any


@functools.lru_cache(maxsize=1)
def form_study() -> FormStudy:
    """Run FORM from three starts, then find and combine every design point.

    References:
        [EQ-REL-HLRF-001], [EQ-REL-FORM-002], [EQ-REL-MDP-001],
        [EQ-REL-MDP-002], [REF-HASOFER-LIND-1974].

    """
    try:
        form_hlrf(four_branch_state())
    except ValueError as error:
        origin_message = str(error)
    else:  # pragma: no cover - the kink at the origin always stops FORM
        origin_message = "FORM converged from the origin"
    inner = form_hlrf(four_branch_state(), start=(0.5, 0.5))
    outer = form_hlrf(four_branch_state(), start=(1.0, -1.0))
    state = four_branch_state()
    found = find_design_points(state, 4, first=form_hlrf(state, start=(0.1, 0.0)))
    combined = multi_point_form(found, seed=SEED)
    return FormStudy(origin_message, inner, outer, found, combined)


@dataclass(frozen=True)
class DirectionalStudy:
    """Directional simulation with the rays it scanned.

    Args:
        result: Library result (estimate, standard error, roots per ray).
        directions: Unit direction of every ray, in sampling order.

    References:
        [EQ-REL-DIR-002], [REF-BJERAGER-1988].

    """

    result: Any
    directions: tuple[tuple[float, float], ...]


@functools.lru_cache(maxsize=1)
def directional_study() -> DirectionalStudy:
    """Run directional simulation and recover each ray's direction.

    The direction of a ray is read from the first point the method evaluated
    on it (the first grid radius); the evaluation record therefore links
    every root reported by the library to its ray.

    References:
        [EQ-REL-DIR-002], [REF-BJERAGER-1988].

    """
    record: list[tuple[float, float]] = []
    result = directional_simulation(four_branch_state(record), DIRECTIONS, seed=SEED)
    first_radius = result.radius_max / result.grid
    directions: list[tuple[float, float]] = []
    for u1, u2 in record:
        if abs(math.hypot(u1, u2) - first_radius) < 1e-9 * first_radius:
            directions.append((u1 / first_radius, u2 / first_radius))
    if len(directions) != result.rays:
        raise RuntimeError("could not match every ray to its evaluations")
    return DirectionalStudy(result, tuple(directions))


@functools.lru_cache(maxsize=1)
def subset_study() -> Any:
    """Run subset simulation with 1000 samples per level and ``p0 = 0.1``.

    References:
        [EQ-REL-SUBSET-001], [REF-AU-BECK-2001].

    """
    return subset_simulation(four_branch_state(), samples_per_level=1000, seed=SEED)


@dataclass(frozen=True)
class ParticleStudy:
    """One moving-particles run with its initial particles.

    Args:
        result: Library result (estimate, Poisson interval, levels).
        initial: The ``N`` initial particles (first ``N`` evaluations).

    References:
        [EQ-REL-MP-001], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """

    result: Any
    initial: tuple[tuple[float, float], ...]


@functools.cache
def particle_study(particles: int) -> ParticleStudy:
    """Run moving particles with ``MARKOV_STEPS`` chain steps per move.

    Args:
        particles: Number of particles ``N``.

    References:
        [EQ-REL-MP-001], [EQ-REL-MP-002], [EQ-REL-PCN-001],
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015].

    """
    record: list[tuple[float, float]] = []
    result = moving_particles(
        four_branch_state(record),
        particles,
        steps=MARKOV_STEPS,
        correlation=0.8,
        seed=SEED,
    )
    return ParticleStudy(result, tuple(record[:particles]))


@dataclass(frozen=True)
class SplittingStudy:
    """Ideal splitting with a fixed budget at the reference probability.

    Args:
        optimum: Library optimum (integer and continuous level counts).
        levels: Level counts ``M`` of the c.o.v. curve.
        cov: Exact c.o.v. for each level count.
        equal_cost_ratio: c.o.v. of crude Monte Carlo over that of moving
            particles at the expected moving-particles cost with ``Q`` steps.

    References:
        [EQ-REL-SPLIT-001], [EQ-REL-SPLIT-002], [REF-LAGNOUX-2006].

    """

    optimum: Any
    levels: tuple[int, ...]
    cov: tuple[float, ...]
    equal_cost_ratio: float


@functools.lru_cache(maxsize=1)
def splitting_study() -> SplittingStudy:
    """Return the c.o.v. of ideal splitting against the number of levels.

    References:
        [EQ-REL-SPLIT-001], [EQ-REL-SPLIT-002], [EQ-REL-MP-001],
        [REF-LAGNOUX-2006], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """
    optimum = optimal_splitting(REFERENCE, SPLITTING_BUDGET)
    levels = tuple(range(1, 16))
    cov = tuple(
        splitting_coefficient_of_variation(REFERENCE, m, SPLITTING_BUDGET)
        for m in levels
    )
    # Equal expected cost N (1 - Q ln p): crude MC squared c.o.v.
    # (1 - p) / (p N (1 - Q ln p)) against -ln(p) / N for moving particles.
    log_p = math.log(REFERENCE)
    crude = (1.0 - REFERENCE) / (REFERENCE * (1.0 - MARKOV_STEPS * log_p))
    ratio = math.sqrt(crude / -log_p)
    return SplittingStudy(optimum, levels, cov, ratio)


# --------------------------------------------------------------------------
# Guarantees: Wilks bounds and certification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WilksStudy:
    """Wilks ranks and sample sizes with a replicate check on normal draws.

    Args:
        rank: Wilks rank of the 0.95/0.95 upper bound for ``WILKS_RUNS``.
        sizes: First- to fourth-order 0.95/0.95 sample sizes.
        median_size: Sample size of a 0.95-confidence bound on the median.
        tolerance: Two-sided 0.95/0.95 tolerance interval of one replicate.
        quantile: True 0.95-quantile of the standard normal law.
        empirical: Type-7 empirical 0.95-quantile of every replicate.
        bounds: Wilks 0.95/0.95 upper bound of every replicate.

    References:
        [EQ-PROP-WILKS-001], [EQ-PROP-WILKS-002], [REF-WILKS-1941].

    """

    rank: Any
    sizes: tuple[int, ...]
    median_size: int
    tolerance: Any
    quantile: float
    empirical: tuple[float, ...]
    bounds: tuple[float, ...]

    def below(self, values: Sequence[float]) -> float:
        """Return the share of ``values`` strictly below the true quantile.

        References:
            [EQ-PROP-WILKS-001], [REF-WILKS-1941].

        """
        return sum(value < self.quantile for value in values) / len(values)


@functools.lru_cache(maxsize=1)
def wilks_study() -> WilksStudy:
    """Compute Wilks ranks and sizes, then check them on normal replicates.

    References:
        [EQ-PROP-WILKS-001], [EQ-PROP-WILKS-002], [EQ-PROP-WILKS-003],
        [REF-WILKS-1941], [REF-NUTT-WALLIS-2004].

    """
    rank = wilks_rank(WILKS_RUNS, LEVEL, LEVEL)
    sizes = tuple(wilks_sample_size(LEVEL, LEVEL, order=m) for m in (1, 2, 3, 4))
    median_size = wilks_sample_size(0.5, LEVEL)
    law = Normal(0.0, 1.0)
    streams = RandomStream(SEED).spawn(WILKS_REPLICATES, label="wilks")
    empirical: list[float] = []
    bounds: list[float] = []
    tolerance = None
    for stream in streams:
        draws = law.sample(WILKS_RUNS, seed=stream)
        bound = wilks_bound(draws, LEVEL, LEVEL)
        empirical.append(bound.empirical_quantile)
        bounds.append(bound.bound)
        if tolerance is None:
            tolerance = wilks_tolerance_interval(draws, LEVEL, LEVEL)
    return WilksStudy(
        rank,
        sizes,
        median_size,
        tolerance,
        NormalDist().inv_cdf(LEVEL),
        tuple(empirical),
        tuple(bounds),
    )


@dataclass(frozen=True)
class CertificationStudy:
    """Certification probability against the scaled sample size ``n alpha``.

    Args:
        scaled: Grid of ``n alpha`` values.
        curves: ``(ratio, rule) -> P(certify)`` on the grid, with
            ``P_f = ratio * alpha``.
        sizes: ``(ratio, rule) -> n alpha`` from which certification has a
            probability of at least 0.95.
        false_certification: ``rule -> P(certify)`` at ``P_f = alpha = 0.01``
            with ``n = 3000``.
        particles: ``alpha -> (moving-particles certification, crude Monte
            Carlo sample size)`` for ``P_f = alpha / 10``, 32 particles, and
            10 calls per move, at ``alpha = RISK`` and ``alpha = 1e-6``.

    References:
        [EQ-REL-CERT-002], [EQ-REL-CERT-003], [REF-CLOPPER-PEARSON-1934].

    """

    scaled: tuple[float, ...]
    curves: Mapping[tuple[float, str], tuple[float, ...]]
    sizes: Mapping[tuple[float, str], float]
    false_certification: Mapping[str, float]
    particles: Mapping[float, tuple[Any, int]]


@functools.lru_cache(maxsize=1)
def certification_study() -> CertificationStudy:
    """Evaluate certification probabilities and sample sizes at ``alpha = RISK``.

    References:
        [EQ-REL-CERT-001], [EQ-REL-CERT-002], [EQ-REL-CERT-003],
        [REF-CLOPPER-PEARSON-1934], [REF-WALTER-2015].

    """
    rules = ("normal", "clopper-pearson")
    sizes_n = sorted(
        {max(1, round(10 ** (k / 60 - 0.4) / RISK)) for k in range(0, 238)}
    )
    scaled = tuple(n * RISK for n in sizes_n)
    curves: dict[tuple[float, str], tuple[float, ...]] = {}
    sizes: dict[tuple[float, str], float] = {}
    for ratio in RISK_RATIOS:
        for rule in rules:
            curves[ratio, rule] = tuple(
                certification_probability(n, ratio * RISK, RISK, rule=rule).probability
                for n in sizes_n
            )
            needed = certification_sample_size(ratio * RISK, RISK, rule=rule)
            sizes[ratio, rule] = needed.size * RISK
    # At alpha = 0.01 and n = 3000 the normal margin admits 21 failures and the
    # exact rule 20, so the two false-certification probabilities differ.
    false = {
        rule: certification_probability(3000, 0.01, 0.01, rule=rule).probability
        for rule in rules
    }
    particles = {
        alpha: (
            moving_particles_certification_probability(32, alpha / 10.0, alpha),
            certification_sample_size(alpha / 10.0, alpha).size,
        )
        for alpha in (RISK, 1e-6)
    }
    return CertificationStudy(scaled, curves, sizes, false, particles)


# --------------------------------------------------------------------------
# Figures: each returns the figure and the formatted facts it states
# --------------------------------------------------------------------------


def _plain_log(axis: Any, subs: tuple[float, ...] = (1.0,)) -> None:
    """Write plain numbers on a logarithmic axis at ``subs`` times powers of 10.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10.0, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def _legible(colour: str, minimum: float = 4.5) -> str:
    """Mix a series colour toward the ink until text in it reaches ``minimum``.

    The WCAG contrast against the canvas is raised in steps of 5% of ink, so
    a label keeps the hue of its series and stays readable.

    References:
        [REF-W3C-WCAG21-2018]. No equation of the registry is evaluated.

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = theme.active_palette()

    def luminance(rgb: Sequence[float]) -> float:
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
        mixed = [(1 - weight) * b + weight * i for b, i in zip(base, ink, strict=True)]
        value = luminance(mixed)
        light, dark = max(value, canvas), min(value, canvas)
        if (light + 0.05) / (dark + 0.05) >= minimum:
            return str(to_hex(mixed))
    return palette.ink


def _label(
    ax: Any,
    xy: tuple[float, float],
    text: str,
    colour: str,
    *,
    offset: tuple[float, float] = (6.0, 0.0),
    ha: str = "left",
    va: str = "center",
) -> None:
    """Write a bold direct label in a legible shade of its series colour.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    ax.annotate(
        text,
        xy,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=_legible(colour),
        annotation_clip=False,
        zorder=6,
    )


def _grid_values(extent: float, nodes: int) -> tuple[list[float], list[list[float]]]:
    """Evaluate ``g`` on a square grid for contour lines.

    References:
        [EQ-REL-WAARTS-001]. No further equation is evaluated.

    """
    axis = [-extent + 2.0 * extent * index / (nodes - 1) for index in range(nodes)]
    return axis, [[four_branch(x, y, K) for x in axis] for y in axis]


def _limit_state_map(ax: Any, extent: float = MAP_EXTENT) -> None:
    """Shade the failure domain, draw ``g = 0``, and square the axes.

    References:
        [EQ-REL-WAARTS-001], [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    axis, values = _grid_values(extent, 161)
    reference = theme.entity_colour("limit state")
    low = min(min(row) for row in values)
    ax.contourf(
        axis, axis, values, levels=[low - 1.0, 0.0], colors=[reference],
        alpha=0.08, zorder=0,
    )  # fmt: skip
    ax.contour(
        axis, axis, values, levels=[0.0], colors=[reference],
        linewidths=theme.entity_style("limit state").width, zorder=2,
    )  # fmt: skip
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.grid(axis="both", color=palette.rule, linewidth=0.6)
    ax.plot([0.0], [0.0], marker="+", color=palette.muted, markersize=8, zorder=3)
    theme.axis_labels(ax, "Standard normal u1", "Standard normal u2")


def _inside(direction: tuple[float, float], radius: float, extent: float) -> float:
    """Return ``radius`` shortened so that ``radius * direction`` stays in frame.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    limits = [extent / abs(c) for c in direction if abs(c) > 1e-12]
    return min([radius, *limits])


def figure_crude_monte_carlo(
    results: Sequence[Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw crude Monte Carlo estimates and Wilson intervals against ``n``.

    References:
        [EQ-MC-PROB-001], [REF-MELCHERS-BECK-2018].

    """
    runs = crude_monte_carlo() if results is None else tuple(results)
    last = runs[-1]
    facts = {
        "n": f"{last.evaluations:,}",
        "pf": f"{last.failure.probability:.2e}",
        "cov": f"{last.coefficient_of_variation:.3f}",
    }
    ax = theme.new_axes(figsize=(7.2, 4.0))
    palette = theme.active_palette()
    dense = [10 ** (2.9 + 2.2 * k / 110) for k in range(111)]
    spread = [1.96 * math.sqrt(REFERENCE * (1.0 - REFERENCE) / n) for n in dense]
    theme.band(
        ax,
        dense,
        [max(REFERENCE - s, 0.0) for s in spread],
        [REFERENCE + s for s in spread],
        color=palette.muted,
    )
    theme.reference_line(
        ax, y=REFERENCE, label=f"reference {REFERENCE:.4g}",
        entity="Monte Carlo reference",
    )  # fmt: skip
    colour = theme.entity_colour("simulated")
    for run in runs:
        interval = run.failure.confidence_interval
        ax.vlines(run.evaluations, interval.lower, interval.upper, color=colour,
                  linewidth=1.6, zorder=3)  # fmt: skip
    theme.points(
        ax,
        [run.evaluations for run in runs],
        [run.failure.probability for run in runs],
        entity="simulated",
    )
    ax.set_xscale("log")
    _plain_log(ax.xaxis)
    ax.set_xlim(800, 125_000)
    ax.set_ylim(0.0, 0.0075)
    theme.axis_labels(ax, theme.axis_title("model_evaluations"), "Estimated Pf")
    theme.title_block(
        ax,
        f"Crude Monte Carlo needs {facts['n']} calls for Pf = {facts['pf']} "
        f"with c.o.v. {facts['cov']}",
        "Dot: estimate · whisker: 95% Wilson interval · grey band: reference "
        "+/- 1.96 standard errors",
    )
    return ax.figure, facts


def figure_method_comparison() -> tuple[Any, dict[str, str]]:
    """Draw every estimator of the laboratory against the reference (log axis).

    References:
        [EQ-REL-FORM-002], [EQ-REL-MDP-002], [EQ-REL-DIR-002],
        [EQ-REL-SUBSET-001], [EQ-REL-MP-002], [REF-MELCHERS-BECK-2018].

    """
    crude = crude_monte_carlo()[-1]
    forms = form_study()
    directional = directional_study().result
    subset = subset_study()
    particles = particle_study(PARTICLES[0]).result
    facts = {
        "form": f"{forms.inner.probability:.2e}",
        "form_ratio": _ratio(forms.inner.probability),
        "multi": f"{forms.multi_point.probability:.2e}",
    }

    def spread(estimate: float, cov: float) -> tuple[float, float]:
        """Return the normal 95% interval ``estimate (1 +/- 1.96 cov)``.

        References:
            [EQ-MC-SE-001], [REF-MELCHERS-BECK-2018].

        """
        return estimate * (1.0 - 1.96 * cov), estimate * (1.0 + 1.96 * cov)

    interval = crude.failure.confidence_interval
    entries: list[tuple[str, float, float | None, float | None]] = [
        (f"FORM, one point ({forms.inner.evaluations} calls)",
         forms.inner.probability, None, None),
        (f"multi-point FORM ({forms.design_points.evaluations:,} calls)",
         forms.multi_point.probability, None, None),
        (f"directional simulation ({directional.evaluations:,} calls)",
         directional.probability, *directional.confidence_interval),
        (f"subset simulation ({subset.evaluations:,} calls)", subset.probability,
         *spread(subset.probability, subset.coefficient_of_variation)),
        (f"moving particles ({particles.evaluations:,} calls)",
         particles.probability, *particles.confidence_interval),
        (f"Monte Carlo ({crude.evaluations:,} calls)", crude.failure.probability,
         interval.lower, interval.upper),
    ]  # fmt: skip
    ax = plot_method_comparison(
        entries,
        reference=REFERENCE,
        reference_label="reference Pf",
        log_scale=True,
        quantity_label=theme.axis_title("failure_probability"),
        title=f"One-point FORM gives {facts['form']}, {facts['form_ratio']} of "
        f"the truth; four points overshoot to {facts['multi']}",
        subtitle="Dot: estimate · whisker: 95% interval (Wilson, Poisson, or "
        "1.96 c.o.v.) · four-branch limit state, k = 7",
    )
    _plain_log(ax.xaxis, subs=(1.0, 1.5, 2.0, 3.0, 5.0))
    return ax.figure, facts


def figure_moving_particles(
    study: ParticleStudy | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw initial and final particles and two level sets passed on the way.

    References:
        [EQ-REL-MP-001], [EQ-REL-MP-002],
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """
    run = particle_study(PARTICLES[0]) if study is None else study
    result = run.result
    lower, upper = result.confidence_interval
    facts = {
        "particles": f"{result.particles}",
        "moves": f"{result.moves}",
        "pf": f"{result.probability:.2e}",
        "interval": f"[{lower:.2e}, {upper:.2e}]",
    }
    ax = theme.new_axes(figsize=(6.4, 6.6))
    palette = theme.active_palette()
    _limit_state_map(ax)
    axis, values = _grid_values(MAP_EXTENT, 161)
    picks = (result.moves // 3, (2 * result.moves) // 3)
    for index in picks:
        ax.contour(axis, axis, values, levels=[result.levels[index]],
                   colors=[palette.muted], linewidths=0.9, zorder=1.5)  # fmt: skip
    ax.scatter(
        [p[0] for p in run.initial], [p[1] for p in run.initial], s=14.0,
        color=theme.entity_colour("safe samples"), alpha=0.7, edgecolors="none",
        zorder=3,
    )  # fmt: skip
    final = result.final_particles
    theme.points(
        ax, [p[0] for p in final], [p[1] for p in final],
        entity="failed samples", size=22.0, zorder=4,
    )  # fmt: skip
    shown = " and ".join(f"{result.levels[index]:.2f}" for index in picks)
    theme.title_block(
        ax,
        f"{facts['moves']} moves push {facts['particles']} particles past g = 0: "
        f"Pf = {facts['pf']}, 95% Poisson interval {facts['interval']}",
        f"Grey: initial particles · green: final · thin lines: levels g = {shown}",
    )
    return ax.figure, facts


def figure_directional_rays(
    study: DirectionalStudy | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw the rays of directional simulation and their roots.

    References:
        [EQ-REL-DIR-002], [REF-BJERAGER-1988].

    """
    data = directional_study() if study is None else study
    result = data.result
    facts = {
        "rays": f"{result.rays}",
        "pf": f"{result.probability:.2e}",
        "cov": f"{result.coefficient_of_variation:.2f}",
        "calls": f"{result.evaluations:,}",
    }
    ax = theme.new_axes(figsize=(6.4, 6.6))
    palette = theme.active_palette()
    _limit_state_map(ax)
    xs: list[float] = []
    ys: list[float] = []
    for (a1, a2), roots in zip(data.directions, result.roots, strict=True):
        reach = _inside((a1, a2), roots[0] if roots else result.radius_max, MAP_EXTENT)
        ax.plot([0.0, reach * a1], [0.0, reach * a2], color=palette.muted,
                linewidth=0.5, alpha=0.6, zorder=1.2)  # fmt: skip
        for radius in roots:
            if max(abs(radius * a1), abs(radius * a2)) <= MAP_EXTENT:
                xs.append(radius * a1)
                ys.append(radius * a2)
    theme.points(ax, xs, ys, entity="failed samples", size=16.0, zorder=4)
    theme.title_block(
        ax,
        f"{facts['rays']} rays, one root each, give Pf = {facts['pf']} with "
        f"c.o.v. {facts['cov']} for {facts['calls']} calls",
        "Grey: ray to its root · green: root r, weight exp(-r^2/2) · shaded: failure",
    )
    return ax.figure, facts


def figure_splitting_levels(
    study: SplittingStudy | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw the c.o.v. of ideal splitting against the number of levels.

    References:
        [EQ-REL-SPLIT-001], [EQ-REL-SPLIT-002], [REF-LAGNOUX-2006].

    """
    data = splitting_study() if study is None else study
    optimum = data.optimum
    facts = {
        "levels": f"{optimum.levels}",
        "cov": f"{optimum.coefficient_of_variation:.3f}",
        "crude": f"{optimum.monte_carlo_coefficient_of_variation:.3f}",
        "level_probability": f"{optimum.level_probability:.2f}",
    }
    ax = theme.new_axes(figsize=(7.2, 4.0))
    palette = theme.active_palette()
    colour = theme.role_colour("model")
    best = theme.role_colour("observed")
    theme.line(ax, list(data.levels), list(data.cov), color=colour)
    theme.points(ax, list(data.levels), list(data.cov), color=colour, size=22.0)
    theme.points(ax, [optimum.levels], [optimum.coefficient_of_variation],
                 color=best, size=70.0, zorder=5)  # fmt: skip
    _label(
        ax,
        (optimum.levels, optimum.coefficient_of_variation),
        f"optimum: M = {facts['levels']}, P_m = {facts['level_probability']}, "
        f"c.o.v. {facts['cov']}",
        best,
        offset=(4.0, 28.0),
        va="bottom",
    )
    ax.annotate(
        f"M = 1 is crude Monte Carlo: {facts['crude']}",
        (1, data.cov[0]),
        xytext=(10.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    moving = optimum.moving_particles_coefficient_of_variation
    theme.reference_line(
        ax, y=moving, label=f"ideal moving particles {moving:.3f}",
        entity="closed form",
    )  # fmt: skip
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.0, 0.24)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    theme.axis_labels(ax, "Number of splitting levels M", "Coefficient of variation")
    theme.title_block(
        ax,
        f"With {SPLITTING_BUDGET:,} ideal draws, {facts['levels']} levels cut "
        f"the c.o.v. from {facts['crude']} to {facts['cov']}",
        f"Exact c.o.v. of ideal splitting at Pf = {REFERENCE:.4g}, budget split "
        "equally over the levels",
    )
    return ax.figure, facts


def figure_wilks_quantiles(
    study: WilksStudy | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw the empirical CDFs of the empirical quantile and the Wilks bound.

    References:
        [EQ-PROP-WILKS-001], [EQ-STAT-QUANT-001], [REF-WILKS-1941].

    """
    data = wilks_study() if study is None else study
    empirical_below = data.below(data.empirical)
    wilks_below = data.below(data.bounds)
    facts = {
        "empirical": f"{100.0 * empirical_below:.1f}%",
        "wilks": f"{100.0 * wilks_below:.1f}%",
        "rank": f"{data.rank.rank}",
        "replicates": f"{len(data.bounds):,}",
    }
    ax = theme.new_axes(figsize=(7.2, 4.0))
    palette = theme.active_palette()
    observed = theme.role_colour("observed")
    model = theme.role_colour("model")
    curves = {}
    for key, values, colour in (
        ("empirical", data.empirical, observed),
        ("wilks", data.bounds, model),
    ):
        ordered = sorted(values)
        heights = [(index + 1) / len(ordered) for index in range(len(ordered))]
        ax.step(ordered, heights, where="post", color=colour,
                linewidth=theme.LINE_WIDTH, zorder=3)  # fmt: skip
        curves[key] = ordered
    truth = theme.entity_colour("truth")
    ax.axvline(data.quantile, color=truth, linewidth=theme.REFERENCE_WIDTH,
               linestyle=theme.REFERENCE, zorder=2)  # fmt: skip
    ax.annotate(
        f"true q = {data.quantile:.3f}",
        (data.quantile, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-4.0, -2.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="semibold",
        color=truth,
    )
    for share, colour in ((empirical_below, observed), (wilks_below, model)):
        ax.plot([data.quantile], [share], marker="o", color=colour,
                markersize=6, zorder=5)  # fmt: skip
    _label(ax, (data.quantile, empirical_below), f"{facts['empirical']} below q",
           observed, offset=(-8.0, 0.0), ha="right")  # fmt: skip
    _label(ax, (data.quantile, wilks_below), f"{facts['wilks']} below q", model,
           offset=(34.0, 0.0))  # fmt: skip
    low_empirical = curves["empirical"][len(curves["empirical"]) * 3 // 10]
    _label(ax, (low_empirical, 0.3), "empirical 0.95-quantile", observed,
           offset=(-10.0, 0.0), ha="right")  # fmt: skip
    high_wilks = curves["wilks"][len(curves["wilks"]) * 3 // 4]
    _label(ax, (high_wilks, 0.75), f"Wilks bound Y_({facts['rank']})", model,
           offset=(10.0, 0.0))  # fmt: skip
    ax.set_xlim(1.3, 2.08)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="x", color=palette.rule, linewidth=0.6)
    theme.axis_labels(
        ax, "Estimate of the 0.95-quantile", theme.axis_title("cumulative_probability")
    )
    theme.title_block(
        ax,
        f"The empirical quantile falls below the truth in {facts['empirical']} of "
        f"replicates, the Wilks bound in {facts['wilks']}",
        f"{facts['replicates']} replicates of {WILKS_RUNS:,} standard normal "
        f"draws · Wilks rank {facts['rank']} gives confidence "
        f"{data.rank.achieved_confidence:.3f}",
    )
    return ax.figure, facts


def figure_certification(
    study: CertificationStudy | None = None,
) -> tuple[Any, dict[str, str]]:
    """Draw the certification probability against ``n alpha``.

    References:
        [EQ-REL-CERT-002], [REF-CLOPPER-PEARSON-1934].

    """
    data = certification_study() if study is None else study
    sizes = [data.sizes[ratio, "normal"] for ratio in RISK_RATIOS]
    facts = {
        "small": f"{sizes[0]:.1f}",
        "half": f"{sizes[1]:.1f}",
        "near": f"{sizes[2]:,.0f}",
    }
    ax = theme.new_axes(figsize=(7.2, 4.4))
    colours = theme.ordered_colours(len(RISK_RATIOS))
    for ratio, colour in zip(RISK_RATIOS, colours, strict=True):
        theme.line(ax, list(data.scaled), list(data.curves[ratio, "normal"]),
                   color=colour, width=1.6)  # fmt: skip
        theme.line(ax, list(data.scaled), list(data.curves[ratio, "clopper-pearson"]),
                   color=colour, width=1.0, dashes=theme.FITTED)  # fmt: skip
    theme.reference_line(ax, y=0.95, label="power 0.95", entity="threshold")
    for size, colour in zip(sizes, colours, strict=True):
        ax.plot([size], [0.95], marker="o", color=colour, markersize=6, zorder=5)
    # Labels sit in empty regions next to each curve (left of the first jump,
    # right of the rising parts), where no other curve passes.
    _label(ax, (2.4, 0.85), "Pf = 0.1 alpha", colours[0], offset=(0.0, 0.0),
           ha="right")  # fmt: skip
    _label(ax, (32.0, 0.6), "Pf = 0.5 alpha", colours[1], offset=(0.0, 0.0))
    _label(ax, (420.0, 0.42), "Pf = 0.9 alpha", colours[2], offset=(0.0, 0.0))
    ax.set_xscale("log")
    _plain_log(ax.xaxis)
    ax.set_xlim(0.4, data.scaled[-1])
    ax.set_ylim(0.0, 1.03)
    theme.axis_labels(ax, "Number of runs times acceptable risk, n alpha",
                      "Probability of certifying")  # fmt: skip
    theme.title_block(
        ax,
        f"Certifying Pf = 0.1, 0.5, 0.9 alpha with power 0.95 needs n alpha = "
        f"{facts['small']}, {facts['half']}, {facts['near']}",
        "Solid: normal margin · dashed: Clopper-Pearson rule · dot: n from which "
        f"P stays above 0.95 · beta = 0.05, alpha = {RISK:g}",
    )
    return ax.figure, facts


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _save(name: str, draw: Callable[[], tuple[Any, dict[str, str]]]) -> None:
    """Draw one figure, save it to ``OUTPUT``, and close it.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    figure, _ = draw()
    path = save_figure(figure, OUTPUT / f"{name}.png")
    theme.pyplot().close(figure)
    print(f"  figure: {path.relative_to(Path(__file__).parent)}")


def main() -> None:
    """Run every step of the laboratory and print its numbers.

    References:
        [EQ-REL-PF-001], [EQ-REL-MP-001], [EQ-PROP-WILKS-001],
        [EQ-REL-CERT-002], [REF-MELCHERS-BECK-2018], [REF-WILKS-1941].

    """
    import matplotlib

    matplotlib.use("Agg")
    start = time.perf_counter()
    print(f"Four-branch limit state, k = {K:g}; reference Pf = {REFERENCE:.4e}")

    print("\n1. Crude Monte Carlo")
    for run in crude_monte_carlo():
        interval = run.failure.confidence_interval
        print(
            f"  n = {run.evaluations:>7,}: Pf = {run.failure.probability:.3e}, "
            f"95% Wilson [{interval.lower:.3e}, {interval.upper:.3e}], "
            f"c.o.v. {run.coefficient_of_variation:.3f}"
        )
    needed = (1.0 - REFERENCE) / (REFERENCE * 0.1**2)
    print(f"  calls for a 10% c.o.v. at the reference: {needed:,.0f}")

    print("\n2. FORM and multi-point FORM")
    forms = form_study()
    print(f"  start at the origin: {forms.origin_message}")
    for label, result in (("(0.5, 0.5)", forms.inner), ("(1, -1)", forms.outer)):
        u1, u2 = result.design_point_u
        print(
            f"  start at {label}: beta = {result.beta:.4f} at ({u1:.3f}, {u2:.3f}), "
            f"Phi(-beta) = {result.probability:.3e} "
            f"({_ratio(result.probability)} of the reference), "
            f"{result.evaluations} calls"
        )
    betas = ", ".join(f"{beta:.3f}" for beta in forms.design_points.betas)
    print(f"  design points: beta = {betas} ({forms.design_points.evaluations} calls)")
    two_and_two = 2.0 * NormalDist().cdf(-3.0) + 2.0 * NormalDist().cdf(-3.5)
    print(
        f"  multi-point FORM: {forms.multi_point.probability:.3e} "
        f"({_ratio(forms.multi_point.probability)} of the reference; "
        f"2 Phi(-3) + 2 Phi(-3.5) = {two_and_two:.3e})"
    )

    print("\n3. Sampling around the design points")
    directional = directional_study().result
    print(
        f"  directional simulation: Pf = {directional.probability:.3e}, "
        f"SE {directional.standard_error:.2e}, c.o.v. "
        f"{directional.coefficient_of_variation:.3f}, {directional.rays} rays, "
        f"{directional.evaluations} calls, roots per ray "
        f"{min(map(len, directional.roots))} to {max(map(len, directional.roots))}"
    )
    subset = subset_study()
    thresholds = ", ".join(f"{value:.3f}" for value in subset.thresholds)
    print(
        f"  subset simulation: Pf = {subset.probability:.3e}, c.o.v. "
        f"{subset.coefficient_of_variation:.3f}, thresholds {thresholds}, "
        f"{subset.evaluations} calls"
    )
    for count in PARTICLES:
        result = particle_study(count).result
        lower, upper = result.confidence_interval
        expected = -count * math.log(REFERENCE)
        sigmas = (result.probability - REFERENCE) / (
            REFERENCE * math.sqrt(math.expm1(-math.log(REFERENCE) / count))
        )
        print(
            f"  moving particles N = {count}: Pf = {result.probability:.3e}, "
            f"95% Poisson [{lower:.3e}, {upper:.3e}], M = {result.moves} moves "
            f"(ideal mean {expected:.0f}), acceptance {result.acceptance_rate:.2f}, "
            f"{result.evaluations} calls, {sigmas:+.1f} ideal SD from the reference"
        )

    print("\n4. Optimal splitting (ideal conditional sampling)")
    splitting = splitting_study()
    optimum = splitting.optimum
    print(
        f"  budget {SPLITTING_BUDGET:,}: M = {optimum.levels} levels of "
        f"probability {optimum.level_probability:.3f}, c.o.v. "
        f"{optimum.coefficient_of_variation:.4f}; continuous optimum "
        f"M = {optimum.continuous_levels:.2f} (P_m = "
        f"{optimum.continuous_level_probability:.4f})"
    )
    print(
        f"  same budget: crude MC c.o.v. "
        f"{optimum.monte_carlo_coefficient_of_variation:.4f}, ideal moving "
        f"particles {optimum.moving_particles_coefficient_of_variation:.4f}"
    )
    print(
        f"  at equal expected cost with {MARKOV_STEPS} chain steps per move, "
        f"crude MC c.o.v. is {splitting.equal_cost_ratio:.2f} times that of "
        "moving particles"
    )

    print("\n5. Wilks bounds on a quantile")
    wilks = wilks_study()
    print(
        f"  0.95/0.95 sample sizes for orders 1-4: "
        f"{', '.join(map(str, wilks.sizes))}; median bound: {wilks.median_size} runs"
    )
    print(
        f"  n = {WILKS_RUNS}: rank {wilks.rank.rank} ({wilks.rank.order}th largest), "
        f"achieved confidence {wilks.rank.achieved_confidence:.4f}"
    )
    tolerance = wilks.tolerance
    print(
        f"  two-sided 0.95/0.95 tolerance interval drops m = {tolerance.lower_rank} "
        f"values per end: [{tolerance.lower:.3f}, {tolerance.upper:.3f}], "
        f"confidence {tolerance.achieved_confidence:.4f}"
    )
    print(
        f"  {WILKS_REPLICATES} replicates: empirical quantile below "
        f"q = {wilks.quantile:.4f} in {wilks.below(wilks.empirical):.3f}, "
        f"Wilks bound below q in {wilks.below(wilks.bounds):.3f} "
        f"(designed: at most {1 - wilks.rank.achieved_confidence:.4f})"
    )

    print(f"\n6. Certification against alpha = {RISK:g}, beta = 0.05")
    certification = certification_study()
    for rule, value in certification.false_certification.items():
        print(
            f"  false certification, Pf = alpha = 0.01, n = 3000, {rule} rule: "
            f"{value:.4f} (target at most 0.05)"
        )
    for ratio in RISK_RATIOS:
        normal = certification.sizes[ratio, "normal"]
        exact = certification.sizes[ratio, "clopper-pearson"]
        print(
            f"  Pf = {ratio:g} alpha: power 0.95 from n alpha = {normal:.2f} "
            f"(normal) or {exact:.2f} (Clopper-Pearson)"
        )
    for alpha, (particles, crude_size) in certification.particles.items():
        print(
            f"  moving particles, N = {particles.particles}, alpha = {alpha:g}, "
            f"Pf = alpha / 10: P(certify) = {particles.probability:.3f} for "
            f"{particles.expected_calls:,.0f} expected calls; crude Monte Carlo "
            f"needs {crude_size:,} runs for power 0.95"
        )

    print("\nFigures")
    _save("risk-crude-monte-carlo", figure_crude_monte_carlo)
    _save("risk-method-comparison", figure_method_comparison)
    _save("risk-moving-particles", figure_moving_particles)
    _save("risk-directional-rays", figure_directional_rays)
    _save("risk-splitting-levels", figure_splitting_levels)
    _save("risk-wilks-quantiles", figure_wilks_quantiles)
    _save("risk-certification", figure_certification)
    print(f"\nWall-clock time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
