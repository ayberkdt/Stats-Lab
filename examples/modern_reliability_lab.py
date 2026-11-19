"""Modern reliability laboratory: cost against accuracy of rare-event methods.

Run with ``PYTHONPATH=src python examples/modern_reliability_lab.py``. All
numbers are printed. When Matplotlib is installed five figures are written to
``examples/generated/``: one cost-against-accuracy chart per comparison
problem, the zero contours of three surrogates inside AK-MCS, and the AK-MCS
population of two initial designs on the four-branch system. Every stochastic
step is seeded, so repeated runs print identical values. The run takes about
two to three minutes in pure Python; the seed counts below are chosen for
that budget, not for converged statistics.

Definitions:
    ``u``: independent standard-normal input vector; every limit state is
    evaluated in this space.
    ``G(u)``: limit-state function; failure is ``G(u) <= 0`` and
    ``P_f = P(G(U) <= 0)``.
    Problems: the four-branch series system with ``k = 7`` (2 inputs, four
    failure regions, ``P_f = 2.2245e-3``), the hyperplane
    ``G = 3.5 - sum(u)/10`` in ``d = 100`` (``P_f = Phi(-3.5)``), and the
    hyperplane ``G = 4.5 - (u1 + u2)/sqrt 2`` (``P_f = Phi(-4.5) = 3.4e-6``).
    Calls: evaluations of the true limit state read from the model's counter;
    surrogate evaluations are free.
    Mean relative error: ``mean_s |P_s / P_f - 1|`` over the seeds ``s``.
    Observed c.o.v.: sample standard deviation of the estimates over the
    seeds divided by their mean; reported c.o.v.: the mean of the c.o.v. each
    run reports about itself (runs that found no failure report an infinite
    c.o.v. and are left out of that mean).

Assumptions:
    - Inputs are independent standard normals, so every reference is exact,
      a quadrature, or a recorded crude Monte Carlo run of ``2 x 10^8`` draws.
    - Subset simulation, moving particles, and AK-SS use Markov chains whose
      reported c.o.v. ignores the correlation between levels and states; the
      observed c.o.v. over seeds shows how far that holds.
    - AK-MCS and AK-SS report the sampling error of their population or last
      level only; misclassification by the surrogate is not part of it.
    - The surrogate comparison uses a smooth limit state that is linear in
      ``u2`` (reference by Gauss-Hermite quadrature over ``u1``); it favours
      polynomial trends, which the text says where it matters.

Equations:
    EQ-REL-MLS-001: model limit state on the standard-normal inputs.
    EQ-REL-WAARTS-001: four-branch series limit state.
    EQ-REL-HLRF-001: HL-RF iteration of FORM.
    EQ-REL-FORM-002: FORM probability ``Phi(-beta)``.
    EQ-REL-SUBSET-001: subset-simulation product of conditional levels.
    EQ-REL-SUBSET-002: component-wise modified Metropolis step.
    EQ-REL-SUBSET-003: subset-simulation c.o.v.
    EQ-REL-SUBSET-005: adaptive conditional sampling (aCS).
    EQ-REL-HMCSS-001: rejection-sampling Hamiltonian step.
    EQ-REL-MP-001: moving-particles estimator ``(1 - 1/N)^M``.
    EQ-REL-MP-002: exact Poisson interval of moving particles.
    EQ-REL-PCN-001: conditional Crank-Nicolson move.
    EQ-REL-DIR-001: directional standard-normal probability.
    EQ-REL-DIR-002: directional simulation with several roots per ray.
    EQ-REL-AKMCS-001: AK-MCS learning function ``U`` and stopping rule.
    EQ-REL-AKMCS-002: AK-MCS population estimate and c.o.v.
    EQ-REL-AKSS-001: AK-SS active learning on surrogate subset levels.
    EQ-REL-AKSS-002: AK-SS stop on the last level's failure-count bounds.
    EQ-SUR-GP-003: GP predictive mean and variance.
    EQ-SUR-PCK-001: PC-Kriging trend and kriging residual.
    EQ-SUR-DE-001: deep-ensemble member likelihood.
    EQ-SUR-DE-002: deep-ensemble mixture prediction and variance split.
    EQ-MOD-GH-001: Gauss-Hermite reference value of the surrogate problem.

Diagnostics:
    - Every method row prints calls, mean relative error, observed c.o.v.,
      and reported c.o.v.; an observed c.o.v. well above the reported one
      means the method's own error bar is too small.
    - Subset kernels print their mean acceptance rate in ``d = 100``.
    - Moving particles prints its predicted cost ``N (1 - steps ln P_f)``
      next to the observed mean.
    - AK-MCS runs on the four-branch system and the surrogate study count
      misclassified population points against the true model on the same
      population (those check calls are not counted as cost).

Cost and error:
    - Sampling methods: subset simulation with ``N = 500`` per level and
      ``p0 = 0.1``; moving particles with 50 particles and 5 steps per move;
      directional simulation with 100 directions (50 in ``d = 100``). Ten
      seeds per method on the two-input problems, three in ``d = 100``.
    - AK-MCS: population ``10^4``, three seeds on the four-branch system, one
      on the ``3.4e-6`` case. AK-SS: ``N = 300`` per level (a smaller level
      than the sampling runs, to keep the lab within its time budget), three
      seeds on both two-input problems. Neither runs in ``d = 100``.
    - With three to ten seeds the observed c.o.v. is itself uncertain by
      roughly 25 to 50 percent; the mean relative error mixes bias and
      spread.
    - Surrogate study: population 5000 and one seed per surrogate.

References:
    REF-AU-BECK-2001: subset simulation with modified Metropolis.
    REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015: aCS kernels.
    REF-WANG-BROCCARDO-SONG-2019: Hamiltonian subset simulation.
    REF-COTTER-ROBERTS-STUART-WHITE-2013: the Crank-Nicolson move.
    REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011: moving particles.
    REF-WALTER-2015: the moving-particles estimator and its variance.
    REF-BJERAGER-1988: directional simulation.
    REF-DITLEVSEN-MADSEN-1996: directional simulation background.
    REF-ECHARD-GAYTON-LEMAIRE-2011: AK-MCS and the four-branch benchmark.
    REF-HUANG-CHEN-ZHU-2016: AK-SS.
    REF-WAARTS-2000: the four-branch function.
    REF-SCHOBI-SUDRET-WIART-2015: PC-Kriging.
    REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017: deep ensembles.
    REF-RASMUSSEN-WILLIAMS-2006: Gaussian-process regression.
    REF-MORRIS-MITCHELL-1995: space-filling designs.
    REF-MELCHERS-BECK-2018: structural reliability background.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.distributions import Normal
from probstat.foundations.random_streams import resolve_stream
from probstat.modeling import (
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch,
    four_branch_model,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.reliability import (
    ModelLimitState,
    ak_mcs,
    ak_ss,
    directional_simulation,
    form_hlrf,
    moving_particles,
    subset_simulation,
)
from probstat.surrogates import (
    ProbabilisticSurrogate,
    fit_deep_ensemble,
    fit_pc_kriging,
)
from probstat.uq import QuantityOfInterest

# Root seed: run ``i`` of a method uses the seed ``SEED + i``.
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
Y = (QuantityOfInterest("g", "limit-state value"),)
SUBSET_SAMPLES = 500
AKSS_SAMPLES = 300
PARTICLES, PARTICLE_STEPS = 50, 5
AK_POPULATION = 10_000
ACCURATE = 0.25


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def save(figure: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on independent standard normal inputs as a limit state.

    References:
        [EQ-REL-MLS-001], [REF-MELCHERS-BECK-2018, Chapter 3].

    """
    names = tuple(f"u{i + 1}" for i in range(dimension))
    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from named inputs.

        References:
            [EQ-REL-MLS-001]. No further equation is evaluated.

        """
        return {"g": function(tuple(p[name] for name in names))}

    return ModelLimitState(ProbabilisticModel(inputs, response, Y), "g", 0.0, "below")


def hyperplane(beta: float, dimension: int) -> Callable[[], ModelLimitState]:
    """Return a builder of ``G = beta - sum(u) / sqrt(d)`` (``P_f = Phi(-beta)``).

    References:
        [EQ-REL-MLS-001], [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    root = math.sqrt(dimension)
    return lambda: make_state(lambda u: beta - math.fsum(u) / root, dimension)


def four_branch_state() -> ModelLimitState:
    """Return the four-branch system with ``k = 7`` as a limit state.

    References:
        [EQ-REL-WAARTS-001], [REF-WAARTS-2000].

    """
    return ModelLimitState(four_branch_model(7.0), "g", 0.0, "below")


# --------------------------------------------------------------------------
# Runs and summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One run of one method: estimate, cost, and self-reported c.o.v.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5]. No equation is evaluated.

    """

    probability: float
    calls: int
    reported_cov: float | None
    detail: float = math.nan
    check_calls: int = 0


@dataclass(frozen=True)
class Summary:
    """Mean error, spread, and cost of one method over its seeds.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5]. No equation is evaluated.

    """

    method: str
    runs: tuple[Run, ...]
    reference: float
    failure: str = ""

    @property
    def mean_calls(self) -> float:
        """Return the mean number of true model calls.

        References:
            [REF-MELCHERS-BECK-2018]. No equation is evaluated.

        """
        return statistics.fmean(run.calls for run in self.runs)

    @property
    def mean_relative_error(self) -> float:
        """Return ``mean |P_s / P_f - 1|`` over the runs.

        References:
            [REF-MELCHERS-BECK-2018]. No equation is evaluated.

        """
        return statistics.fmean(
            abs(run.probability / self.reference - 1.0) for run in self.runs
        )

    @property
    def observed_cov(self) -> float:
        """Return the sample c.o.v. of the estimates (``nan`` for one run).

        References:
            [REF-MELCHERS-BECK-2018]. No equation is evaluated.

        """
        values = [run.probability for run in self.runs]
        mean = statistics.fmean(values)
        if len(values) < 2 or mean <= 0.0:
            return math.nan
        return statistics.stdev(values) / mean

    @property
    def reported_cov(self) -> float:
        """Return the mean self-reported c.o.v. (``nan`` when none is reported).

        References:
            [REF-MELCHERS-BECK-2018]. No equation is evaluated.

        """
        values = [
            run.reported_cov
            for run in self.runs
            if run.reported_cov is not None and math.isfinite(run.reported_cov)
        ]
        return statistics.fmean(values) if values else math.nan


Runner = Callable[[ModelLimitState, int], Run]


def run_form(state: ModelLimitState, seed: int) -> Run:
    """Run FORM by HL-RF (deterministic; the seed is unused).

    References:
        [EQ-REL-HLRF-001], [EQ-REL-FORM-002], [REF-MELCHERS-BECK-2018].

    """
    del seed
    result = form_hlrf(state)
    return Run(result.probability, result.evaluations, None)


def run_subset(sampler: str) -> Runner:
    """Return a runner of subset simulation with one Markov-chain kernel.

    References:
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-002], [EQ-REL-SUBSET-003],
        [EQ-REL-SUBSET-005], [EQ-REL-HMCSS-001], [REF-AU-BECK-2001].

    """

    def run(state: ModelLimitState, seed: int) -> Run:
        """Run subset simulation and keep its mean acceptance rate.

        References:
            [EQ-REL-SUBSET-001], [REF-AU-BECK-2001].

        """
        result = subset_simulation(
            state, samples_per_level=SUBSET_SAMPLES, sampler=sampler, seed=seed
        )
        rates = [rate for rate in result.acceptance_rates if math.isfinite(rate)]
        return Run(
            result.probability,
            result.evaluations,
            result.coefficient_of_variation,
            statistics.fmean(rates) if rates else math.nan,
        )

    return run


def run_particles(state: ModelLimitState, seed: int) -> Run:
    """Run moving particles with 50 particles and 5 Markov steps per move.

    References:
        [EQ-REL-MP-001], [EQ-REL-MP-002], [EQ-REL-PCN-001],
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """
    result = moving_particles(state, PARTICLES, steps=PARTICLE_STEPS, seed=seed)
    return Run(
        result.probability,
        result.evaluations,
        result.coefficient_of_variation,
        result.acceptance_rate,
    )


def run_directional(directions: int) -> Runner:
    """Return a runner of directional simulation with ``directions`` rays.

    References:
        [EQ-REL-DIR-001], [EQ-REL-DIR-002], [REF-BJERAGER-1988].

    """

    def run(state: ModelLimitState, seed: int) -> Run:
        """Run directional simulation.

        References:
            [EQ-REL-DIR-002], [REF-BJERAGER-1988].

        """
        result = directional_simulation(state, directions, seed=seed)
        return Run(
            result.probability, result.evaluations, result.coefficient_of_variation
        )

    return run


def run_ak_mcs(state: ModelLimitState, seed: int) -> Run:
    """Run AK-MCS on a ``10^4`` population with the same-population MC check.

    ``detail`` holds the number of misclassified population points (the
    check's calls are not counted).

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    result = ak_mcs(
        state,
        AK_POPULATION,
        max_population=AK_POPULATION,
        target_cov=1.0,
        compare_with_monte_carlo=True,
        seed=seed,
    )
    return Run(
        result.probability,
        result.evaluations,
        result.coefficient_of_variation,
        float(result.misclassified),
        result.reference_evaluations,
    )


def run_ak_ss(state: ModelLimitState, seed: int) -> Run:
    """Run AK-SS with ``N = 300`` samples per level; ``detail`` is the rounds.

    References:
        [EQ-REL-AKSS-001], [EQ-REL-AKSS-002], [REF-HUANG-CHEN-ZHU-2016].

    """
    result = ak_ss(state, samples_per_level=AKSS_SAMPLES, seed=seed)
    return Run(
        result.probability,
        result.evaluations,
        result.coefficient_of_variation,
        float(result.rounds),
    )


@dataclass(frozen=True)
class Case:
    """One comparison problem with its reference and per-method seed counts.

    References:
        [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """

    key: str
    label: str
    build: Callable[[], ModelLimitState]
    reference: float
    plan: tuple[tuple[str, Runner, int], ...]


def cases() -> tuple[Case, ...]:
    """Return the three comparison problems and the runs planned on each.

    References:
        [EQ-REL-WAARTS-001], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """

    def sampling(seeds: int, directions: int) -> tuple[tuple[str, Runner, int], ...]:
        """Return the sampling-method plan with ``seeds`` runs per method.

        References:
            [REF-AU-BECK-2001]. No equation is evaluated.

        """
        return (
            ("Subset (MMA)", run_subset("mma"), seeds),
            ("Subset (aCS)", run_subset("acs"), seeds),
            ("Subset (HMC)", run_subset("hmc"), seeds),
            ("Moving particles", run_particles, seeds),
            ("Directional simulation", run_directional(directions), seeds),
        )

    return (
        Case(
            "four-branch",
            "four-branch system, k = 7",
            four_branch_state,
            FOUR_BRANCH_REFERENCE_PROBABILITY,
            (("FORM", run_form, 1),)
            + sampling(10, 100)
            + (("AK-MCS", run_ak_mcs, 3), ("AK-SS", run_ak_ss, 3)),
        ),
        Case(
            "linear-100d",
            "hyperplane, d = 100, beta = 3.5",
            hyperplane(3.5, 100),
            standard_normal_cdf(-3.5),
            (("FORM", run_form, 1),) + sampling(3, 50),
        ),
        Case(
            "linear-rare",
            "hyperplane, d = 2, beta = 4.5",
            hyperplane(4.5, 2),
            standard_normal_cdf(-4.5),
            (("FORM", run_form, 1),)
            + sampling(10, 100)
            + (("AK-MCS", run_ak_mcs, 1), ("AK-SS", run_ak_ss, 3)),
        ),
    )


def run_case(case: Case) -> list[Summary]:
    """Run every planned method of one case over its seeds.

    A method that raises (FORM at the kink of the four-branch system) is
    kept as a summary without runs and its message.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5-6]. No further equation is
        evaluated.

    """
    summaries: list[Summary] = []
    for method, runner, count in case.plan:
        runs: list[Run] = []
        failure = ""
        for index in range(count):
            state = case.build()
            try:
                run = runner(state, SEED + index)
            except ValueError as error:
                failure = str(error).split(";")[0]
                break
            if run.calls + run.check_calls != state.model.evaluations:
                raise RuntimeError(f"{method}: reported calls differ from the counter")
            runs.append(run)
        summaries.append(Summary(method, tuple(runs), case.reference, failure))
    return summaries


def print_case(case: Case, summaries: Sequence[Summary]) -> None:
    """Print the comparison table of one case.

    References:
        [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """
    print(f"   reference P_f = {case.reference:.4e}")
    print(
        "   method                  runs  calls (mean)  rel. error  "
        "obs. c.o.v.  rep. c.o.v."
    )
    for summary in summaries:
        if not summary.runs:
            print(f"   {summary.method:<23s} fails: {summary.failure}")
            continue
        print(
            f"   {summary.method:<23s} {len(summary.runs):>4d}  "
            f"{summary.mean_calls:>12,.0f}  {summary.mean_relative_error:>10.3f}  "
            f"{summary.observed_cov:>11.3f}  {summary.reported_cov:>11.3f}"
        )


def cheapest_accurate(summaries: Sequence[Summary]) -> Summary | None:
    """Return the cheapest method whose mean relative error is within 25 %.

    References:
        [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """
    accurate = [s for s in summaries if s.runs and s.mean_relative_error <= ACCURATE]
    return min(accurate, key=lambda s: s.mean_calls) if accurate else None


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _plain_log(axis: Any, subs: tuple[float, ...] = (1.0, 3.0)) -> None:
    """Label a logarithmic axis with plain numbers (no ``10^k`` notation).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def plain(value: float, _: object) -> str:
        """Return a positional decimal with thousands separators.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        if value >= 1.0:
            return f"{value:,.0f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    axis.set_major_locator(LogLocator(base=10.0, subs=subs))
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


#: Colour role and marker of each method: the sampling family takes the
#: observed hue and the extra slots, the model-based methods the model hue.
METHOD_STYLES: dict[str, tuple[str, str]] = {
    "FORM": ("model", "D"),
    "Subset (MMA)": ("observed", "o"),
    "Subset (aCS)": ("observed", "s"),
    "Subset (HMC)": ("observed", "^"),
    "Moving particles": ("series-3", "v"),
    "Directional simulation": ("series-6", "P"),
    "AK-MCS": ("model", "X"),
    "AK-SS": ("model", "h"),
}


def label_points(
    axes: Any, items: Sequence[tuple[float, float, str]], *, cluster: float = 50.0
) -> None:
    """Label points to their right, pushing apart labels of nearby points.

    Points closer than ``cluster`` points horizontally form one group; the
    labels of a group are spread vertically by at least 13 points, and a
    displaced label gets a thin leader line to its point.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    palette = theme.active_palette()
    axes.get_xlim()
    axes.get_ylim()
    scale = axes.figure.dpi / 72.0
    to_display = axes.transData.transform
    entries = sorted(
        ((*to_display((x, y)), x, y, text) for x, y, text in items),
        key=lambda entry: entry[0],
    )
    groups: list[list[tuple[float, float, float, float, str]]] = []
    for entry in entries:
        if groups and entry[0] - groups[-1][-1][0] < cluster * scale:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    top = to_display((0.0, axes.get_ylim()[1]))[1]
    minimum = 13.0 * scale
    for group in groups:
        group.sort(key=lambda entry: entry[1])
        targets = [entry[1] for entry in group]
        for index in range(1, len(targets)):
            targets[index] = max(targets[index], targets[index - 1] + minimum)
        overflow = targets[-1] - top
        if overflow > 0.0:
            targets = [value - overflow for value in targets]
        right = max(entry[0] for entry in group)
        for (display_x, display_y, x, y, text), target in zip(
            group, targets, strict=True
        ):
            dx = (right - display_x) / scale + 10.0
            dy = (target - display_y) / scale
            leader = abs(dy) > 4.0 or dx > 14.0
            axes.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=theme.ANNOTATION_SIZE,
                color=palette.ink,
                annotation_clip=False,
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": palette.muted,
                        "linewidth": 0.7,
                        "shrinkA": 1.0,
                        "shrinkB": 5.0,
                        "relpos": (0.0, 0.5),
                    }
                    if leader
                    else None
                ),
            )


FLOOR_PERCENT = 0.1


def cost_figure(case: Case, summaries: Sequence[Summary]) -> str:
    """Plot mean relative error against mean model calls, one marker per method.

    Errors below 0.1 % (FORM on a hyperplane is exact) are drawn at 0.1 %
    and their label says so.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.4))
    items: list[tuple[float, float, str]] = []
    shown_values: list[float] = []
    for summary in summaries:
        if not summary.runs:
            continue
        role, marker = METHOD_STYLES[summary.method]
        error = 100.0 * summary.mean_relative_error
        shown = max(error, FLOOR_PERCENT)
        shown_values.append(shown)
        theme.points(
            axes,
            [summary.mean_calls],
            [shown],
            color=theme.role_colour(role),
            marker=marker,
            size=70.0,
        )
        text = summary.method
        if error < FLOOR_PERCENT:
            text += " (exact, below 0.1%)"
        elif summary.method == "AK-MCS" and summary.mean_relative_error >= 0.999:
            text += " (no failure found)"
        items.append((summary.mean_calls, shown, text))
    axes.set_xscale("log")
    axes.set_yscale("log")
    low, high = axes.get_xlim()
    axes.set_xlim(low / 1.5, high * 8.0)
    axes.set_ylim(min(shown_values) / 2.0, max(shown_values) * 2.5)
    _plain_log(axes.xaxis)
    _plain_log(axes.yaxis, (1.0, 2.0, 5.0))
    theme.axis_labels(
        axes, theme.axis_title("model_evaluations"), "Mean relative error [%]"
    )
    best = cheapest_accurate(summaries)
    if best is None:
        title = f"No method reaches {ACCURATE:.0%} mean error on the {case.label}"
    else:
        title = (
            f"{best.method} is the cheapest within {ACCURATE:.0%} mean error on the "
            f"{case.label}: {best.mean_calls:,.0f} calls"
        )
    seeds = sorted({len(s.runs) for s in summaries if s.runs})
    theme.title_block(
        axes,
        title,
        "Marker: one method, mean over "
        + (f"{seeds[0]}" if seeds[0] == seeds[-1] else f"{seeds[0]} to {seeds[-1]}")
        + " seeds · lower left is better",
    )
    label_points(axes, items)
    return save(axes.figure, f"modern-reliability-cost-{case.key}.png")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def comparison_section() -> dict[str, list[Summary]]:
    """Run the cost-against-accuracy study on the three problems.

    References:
        [EQ-REL-SUBSET-001], [EQ-REL-MP-001], [EQ-REL-DIR-002],
        [EQ-REL-AKMCS-002], [EQ-REL-AKSS-001], [REF-AU-BECK-2001].

    """
    print("1. Cost against accuracy on three problems")
    results: dict[str, list[Summary]] = {}
    for case in cases():
        start = time.perf_counter()
        print(f"\n   {case.label}")
        summaries = run_case(case)
        print_case(case, summaries)
        results[case.key] = summaries
        if has_matplotlib():
            print(f"   figure: {cost_figure(case, summaries)}")
        print(f"   ({time.perf_counter() - start:.1f} s)")
    return results


def lessons_section(results: Mapping[str, Sequence[Summary]]) -> None:
    """Read the study: kernels in d = 100, AK-MCS against AK-SS, MP and DS.

    References:
        [EQ-REL-SUBSET-005], [EQ-REL-HMCSS-001], [EQ-REL-MP-001],
        [EQ-REL-DIR-001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WALTER-2015].

    """
    print("\n2. What the study says")
    by = {key: {s.method: s for s in rows} for key, rows in results.items()}
    print("   Subset kernels in d = 100 (mean acceptance of the conditional levels):")
    for method in ("Subset (MMA)", "Subset (aCS)", "Subset (HMC)"):
        summary = by["linear-100d"][method]
        acceptance = statistics.fmean(run.detail for run in summary.runs)
        print(
            f"     {method:<13s} acceptance {acceptance:.2f}  "
            f"rel. error {summary.mean_relative_error:.3f}  "
            f"obs./rep. c.o.v. {summary.observed_cov:.2f}/{summary.reported_cov:.2f}"
        )
    directional = by["linear-100d"]["Directional simulation"]
    estimates = ", ".join(f"{run.probability:.1e}" for run in directional.runs)
    print(
        f"   Directional simulation in d = 100: estimates {estimates} against "
        f"{directional.reference:.2e} (rel. error "
        f"{directional.mean_relative_error:.3f}); in d = 2 (beta = 4.5): "
        f"{by['linear-rare']['Directional simulation'].mean_relative_error:.3f}."
    )
    print(
        "   The radial integral is exact on every ray, but in d = 100 a ray"
        " crosses the plane at r = beta / cos(angle to the design direction),"
        " mostly far beyond the radii near 10 where chi_100 has its mass; the"
        " few rays close to the design direction carry the probability, and 50"
        " random rays rarely include one."
    )
    for key in ("four-branch", "linear-rare"):
        summary = by[key]["Moving particles"]
        predicted = PARTICLES * (
            1.0 - PARTICLE_STEPS * math.log(results_reference(results, key))
        )
        print(
            f"   Moving particles, {key}: calls {summary.mean_calls:,.0f} against "
            f"N (1 - steps ln P_f) = {predicted:,.0f}; obs./rep. c.o.v. "
            f"{summary.observed_cov:.2f}/{summary.reported_cov:.2f}"
        )
    rare_mcs = by["linear-rare"]["AK-MCS"]
    rare_ss = by["linear-rare"]["AK-SS"]
    reference = results_reference(results, "linear-rare")
    needed = (1.0 - reference) / (reference * 0.3**2)
    print(
        f"   AK-MCS at P_f = {reference:.2e} with a 10^4 population: "
        f"P = {rare_mcs.runs[0].probability:.1e} after {rare_mcs.runs[0].calls} "
        f"calls; a c.o.v. of 0.3 needs about {needed:,.0f} candidates."
    )
    print(
        f"   AK-SS on the same case: {rare_ss.mean_calls:.0f} calls on average, "
        f"rel. error {rare_ss.mean_relative_error:.3f}, "
        f"mean rounds {statistics.fmean(r.detail for r in rare_ss.runs):.1f}."
    )
    branch = by["four-branch"]
    misclassified = [int(run.detail) for run in branch["AK-MCS"].runs]
    print(
        f"   Four-branch: AK-MCS {branch['AK-MCS'].mean_calls:.0f} calls "
        f"(misclassified per run {misclassified}), AK-SS "
        f"{branch['AK-SS'].mean_calls:.0f} calls, subset (MMA) "
        f"{branch['Subset (MMA)'].mean_calls:,.0f} calls. FORM: "
        f"{branch['FORM'].failure or 'ran'}."
    )


def results_reference(results: Mapping[str, Sequence[Summary]], key: str) -> float:
    """Return the reference failure probability of one case.

    References:
        [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """
    return results[key][0].reference


# --------------------------------------------------------------------------
# Surrogate choice inside AK-MCS
# --------------------------------------------------------------------------

WAVE_BETA, WAVE_KAPPA, WAVE_AMPLITUDE = 2.0, 0.5, 0.5


def wavy_margin(u: Sequence[float]) -> float:
    """Return ``G = h(u1) - u2`` with ``h(z) = 2 + 0.25 z^2 + 0.5 sin 2z``.

    References:
        [EQ-REL-MLS-001]. No further equation is evaluated.

    """
    return wavy_level(u[0]) - u[1]


def wavy_level(z: float) -> float:
    """Return the failure level ``h(z)`` of the surrogate-study limit state.

    References:
        [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """
    return WAVE_BETA + 0.5 * WAVE_KAPPA * z * z + WAVE_AMPLITUDE * math.sin(2.0 * z)


def wavy_reference() -> float:
    """Return ``P_f = E[Phi(-h(Z))]`` by 60-node Gauss-Hermite quadrature.

    References:
        [EQ-MOD-GH-001], [REF-MELCHERS-BECK-2018].

    """
    nodes, weights = gauss_hermite(60)
    return math.fsum(
        w * standard_normal_cdf(-wavy_level(z))
        for z, w in zip(nodes, weights, strict=True)
    )


class EpistemicEnsemble:
    """Deep ensemble that reports only its epistemic (member) variance.

    For a deterministic limit state the aleatoric part measures misfit, not
    noise; the member disagreement plays the role of a GP's latent variance.

    References:
        [EQ-SUR-DE-002], [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017].

    """

    def __init__(self, ensemble: Any) -> None:
        """Keep the fitted ensemble.

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017]. No equation is
            evaluated.

        """
        self.ensemble = ensemble

    def predict(self, x: Sequence[float]) -> float:
        """Return the ensemble mean.

        References:
            [EQ-SUR-DE-002].

        """
        return float(self.ensemble.predict(x))

    def predict_with_variance(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the mean and the epistemic variance.

        References:
            [EQ-SUR-DE-002].

        """
        mean, variance = self.ensemble.predict_with_variance(x, include_aleatoric=False)
        return float(mean), float(variance)


def surrogate_factories(
    inputs: JointInput,
) -> tuple[tuple[str, Any], ...]:
    """Return the GP (default incremental path), PC-Kriging, and ensemble hooks.

    References:
        [EQ-SUR-GP-003], [EQ-SUR-PCK-001], [EQ-SUR-DE-001],
        [REF-SCHOBI-SUDRET-WIART-2015].

    """

    def pc_kriging(
        points: Sequence[Sequence[float]],
        values: Sequence[float],
        previous: ProbabilisticSurrogate | None,
    ) -> ProbabilisticSurrogate:
        """Fit PC-Kriging (Hermite trend up to degree 3) on the training set.

        References:
            [EQ-SUR-PCK-001], [REF-SCHOBI-SUDRET-WIART-2015].

        """
        del previous
        return fit_pc_kriging(
            (inputs, points, values), degrees=range(1, 4), restarts=1, seed=SEED
        )

    def ensemble(
        points: Sequence[Sequence[float]],
        values: Sequence[float],
        previous: ProbabilisticSurrogate | None,
    ) -> ProbabilisticSurrogate:
        """Fit three one-layer networks and keep their epistemic variance.

        References:
            [EQ-SUR-DE-001], [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017].

        """
        del previous
        fitted = fit_deep_ensemble(
            points,
            values,
            members=3,
            hidden=(16,),
            iterations=300,
            learning_rate=0.01,
            seed=SEED,
        )
        return EpistemicEnsemble(fitted)

    return (("GP", None), ("PC-Kriging", pc_kriging), ("Deep ensemble", ensemble))


def surrogate_section() -> None:
    """Compare GP, PC-Kriging, and a deep ensemble inside AK-MCS on one problem.

    References:
        [EQ-REL-AKMCS-001], [EQ-SUR-GP-003], [EQ-SUR-PCK-001],
        [EQ-SUR-DE-002], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    print("\n3. Surrogate choice inside AK-MCS: G = 2 + 0.25 u1^2 + 0.5 sin 2u1 - u2")
    reference = wavy_reference()
    print(f"   Gauss-Hermite reference P_f = {reference:.4e}; population 5000")
    print("   surrogate      calls  added  P_f        MC same pop.  wrong  stop")
    runs: list[tuple[str, Any]] = []
    for name, factory in surrogate_factories(make_state(wavy_margin, 2).model.inputs):
        start = time.perf_counter()
        result = ak_mcs(
            make_state(wavy_margin, 2),
            5000,
            max_population=5000,
            target_cov=1.0,
            max_added=15,
            surrogate=factory,
            compare_with_monte_carlo=True,
            seed=SEED,
        )
        print(
            f"   {name:<13s} {result.evaluations:>5d}  {result.added:>5d}  "
            f"{result.probability:.4e}  {result.reference_probability:.4e}    "
            f"{result.misclassified:>5d}  "
            f"{'converged' if result.converged else 'max_added reached'} "
            f"({time.perf_counter() - start:.1f} s)"
        )
        runs.append((name, result))
    print(
        "   U = |mu| / sigma stops only when sigma is honest near G = 0: "
        + ", ".join(
            f"{name} {'stopped by U' if result.converged else 'hit max_added'} "
            f"with {result.misclassified} wrong"
            for name, result in runs
        )
        + ". The ensemble's member spread is not calibrated."
    )
    if has_matplotlib():
        print(f"   figure: {surrogate_figure(runs)}")


def surrogate_figure(runs: Sequence[tuple[str, Any]]) -> str:
    """Draw the true ``G = 0`` and each surrogate's zero contour.

    References:
        [EQ-REL-MLS-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.6))
    xs = [-4.0 + 8.0 * k / 80 for k in range(81)]
    ys = [-1.0 + 6.0 * k / 60 for k in range(61)]
    truth = theme.entity_colour("limit state")
    inside = [x for x in xs if wavy_level(x) <= 5.0]
    axes.plot(
        inside, [wavy_level(x) for x in inside], color=truth, linewidth=4.0, zorder=2
    )
    items = [(inside[-1], wavy_level(inside[-1]), "true G = 0")]
    styles = (
        ("model", (0, (5.0, 2.0))),
        ("series-5", (0, (6.0, 2.0, 1.5, 2.0))),
        ("series-4", (0, (1.2, 1.6))),
    )
    for (name, result), (role, dashes) in zip(runs, styles, strict=True):
        grid = [[result.surrogate.predict((x, y)) for x in xs] for y in ys]
        contour = axes.contour(
            xs,
            ys,
            grid,
            levels=[0.0],
            colors=[theme.role_colour(role)],
            linewidths=2.0,
            linestyles=[dashes],
            zorder=3,
        )
        ends = [
            (float(point[0]), float(point[1]))
            for segment in contour.allsegs[0]
            for point in segment
        ]
        if ends:
            x, y = max(ends)
            items.append((x, y, f"{name}: {result.misclassified} wrong"))
    axes.set_xlim(-4.0, 6.8)
    axes.set_ylim(-1.0, 5.0)
    theme.axis_labels(axes, "Standard normal u1", "Standard normal u2")
    fewest = min(result.misclassified for _, result in runs)
    best = [name for name, result in runs if result.misclassified == fewest]
    worst = max(runs, key=lambda item: item[1].misclassified)
    verb = "classifies" if len(best) == 1 else "classify"
    theme.title_block(
        axes,
        f"{' and '.join(best)} {verb} the population with {fewest} errors; "
        f"the {worst[0].lower()} leaves {worst[1].misclassified}",
        "Line: zero contour of each surrogate mean when AK-MCS stopped · "
        "failure lies above the line",
    )
    label_points(axes, items, cluster=200.0)
    return save(axes.figure, "modern-reliability-surrogates.png")


# --------------------------------------------------------------------------
# Multiple failure regions: the blind spot of a central design
# --------------------------------------------------------------------------

BLIND_SEED = SEED + 1


def population_of(result: Any, seed: int) -> list[tuple[float, float]]:
    """Redraw the AK-MCS population of one run from its child stream.

    References:
        [EQ-REL-AKMCS-002], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    generator = resolve_stream(seed).child("ak-mcs", "population").generator()
    return [
        (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
        for _ in range(result.population)
    ]


def blind_spot_section() -> None:
    """AK-MCS on the four-branch system with a central and a box design.

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-WAARTS-001], [REF-MORRIS-MITCHELL-1995],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    print(f"\n4. Four failure regions and the initial design (seed {BLIND_SEED})")
    print("   design                       calls  P_f        MC same pop.  wrong")
    runs = []
    for label, options in (
        ("LHS through Phi^-1", {"design": "lhs", "design_bound": None}),
        ("maximin LHS in [-5, 5]^2", {"design": "lhs-maximin"}),
    ):
        result = ak_mcs(
            four_branch_state(),
            AK_POPULATION,
            max_population=AK_POPULATION,
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=BLIND_SEED,
            **options,  # type: ignore[arg-type]
        )
        print(
            f"   {label:<28s} {result.evaluations:>5d}  {result.probability:.4e}  "
            f"{result.reference_probability:.4e}    {result.misclassified:>5d}  "
            f"{'converged' if result.converged else result.message}"
        )
        runs.append((label, result))
    print(
        f"   The central design missed {runs[0][1].misclassified} of the "
        f"{round(runs[0][1].reference_probability * AK_POPULATION)} failed population"
        " points: where no training point lies, the surrogate predicts safe with a"
        " small sigma, so U proposes no point there."
    )
    if has_matplotlib():
        print(f"   figure: {blind_spot_figure(runs)}")


def blind_spot_figure(runs: Sequence[tuple[str, Any]]) -> str:
    """Show both AK-MCS populations by predicted class side by side.

    References:
        [EQ-REL-AKMCS-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme
    from probstat.visualization.reliability_plots import plot_ak_mcs_classification

    plt = theme.pyplot()
    figure, panels = plt.subplots(1, 2, figsize=(9.6, 5.2), sharey=True)
    axis = [-5.0 + 10.0 * k / 120 for k in range(121)]
    grid = (axis, axis, [[four_branch(x, y, 7.0) for x in axis] for y in axis])
    for ax, (label, result) in zip(panels, runs, strict=True):
        theme.style_axes(ax)
        plot_ak_mcs_classification(
            result,
            population_of(result, BLIND_SEED),
            limit_state_grid=grid,
            extent=5.0,
            ax=ax,
        )
        ax.set_xlabel(f"u1 · {label}, {result.evaluations} calls")
    for artist in theme.title_artists(panels[1]):
        artist.set_text("")
        artist.set_visible(False)
    panels[1].set_ylabel("")
    central, box = runs[0][1], runs[1][1]
    theme.retitle(
        panels[0],
        f"A central design misses "
        f"{'all four' if central.predicted_failures == 0 else 'some'} failure "
        f"regions ({central.misclassified} points wrong); the box design finds "
        f"them ({box.misclassified} wrong)",
        "Grey: failure domain · squares: training points (filled: added) · "
        "dots: population by predicted class",
    )
    return save(figure, "modern-reliability-blind-spot.png")


def main() -> None:
    """Run every section and print the elapsed time.

    References:
        [REF-AU-BECK-2001], [REF-ECHARD-GAYTON-LEMAIRE-2011],
        [REF-HUANG-CHEN-ZHU-2016].

    """
    start = time.perf_counter()
    results = comparison_section()
    lessons_section(results)
    surrogate_section()
    blind_spot_section()
    print(f"\nelapsed {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
