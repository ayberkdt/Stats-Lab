"""Figure of the method-development guide.

The figure runs the seeded example of the guide (ADR-0131): the derived
estimator of ``E[integral_0^1 exp(Z x) dx]`` is a control-variate-corrected
level-2 trapezoid rule plus a randomized-truncation correction that is
evaluated with probability ``q``. It shows the measured work-normalized
variance at five values of ``q`` against the prediction
``(V + m / q)(c + q c')``. The facts are the numbers the caption quotes.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-RHEE-GLYNN-2015],
    [REF-GLASSERMAN-2003].

"""

from __future__ import annotations

import functools
import math
from typing import Any

from probstat.foundations import RandomStream
from probstat.methodlab import (
    Draw,
    control_variate_estimator,
    estimator_from_function,
    geometric_levels,
    rank_by_efficiency,
    single_term_estimator,
    sum_of_estimators,
    summarize,
    thinned_estimator,
)
from probstat.visualization import theme

from . import GuideFigure

PROBABILITIES = (0.05, 0.1, 0.25, 0.5, 1.0)
REPLICATES = 20_000


def _trapezoid(z: float, level: int) -> float:
    """Return the trapezoid rule for ``exp(z x)`` on ``[0, 1]``.

    References:
        [EQ-NUM-TRAP-001].

    """
    n = 2**level
    inner = math.fsum(math.exp(z * i / n) for i in range(1, n))
    return (0.5 * (1.0 + math.exp(z)) + inner) / n


def _difference(level: int, stream: RandomStream) -> Draw:
    """Return one coupled difference of successive trapezoid levels.

    References:
        [EQ-MLAB-RT-001].

    """
    z = stream.generator().gauss(0.0, 1.0)
    value = _trapezoid(z, level + 3) - _trapezoid(z, level + 2)
    return Draw(value, 3.0 * 2 ** (level + 2) + 2.0)


@functools.cache
def _example() -> dict[str, Any]:
    """Run the guide's example and the prediction of its efficiency.

    References:
        [EQ-MLAB-THIN-001], [EQ-MLAB-CV-001], [EQ-MLAB-WNV-001].

    """
    base = estimator_from_function(
        "trapezoid[2]", lambda g: _trapezoid(g.gauss(0.0, 1.0), 2), cost=5.0
    )
    control = estimator_from_function(
        "level0", lambda g: _trapezoid(g.gauss(0.0, 1.0), 0), cost=0.0
    )
    base_cv, _ = control_variate_estimator(
        "base+cv",
        base,
        (control,),
        ((1.0 + math.exp(0.5)) / 2.0,),
        pilot_size=2000,
        seed=1,
    )
    correction = single_term_estimator("correction", _difference, geometric_levels(0.2))
    candidates = [
        sum_of_estimators(f"q={q}", (base_cv, thinned_estimator("rare", correction, q)))
        for q in PROBABILITIES
    ]
    ranking = rank_by_efficiency(candidates, replicates=REPLICATES, seed=3)
    measured = {
        float(s.name[2:]): (s.work_normalized_variance, error)
        for s, error in zip(ranking.summaries, ranking.log_standard_errors, strict=True)
    }
    base_summary = summarize(base_cv, REPLICATES, 4)
    correction_summary = summarize(correction, REPLICATES, 4)
    variance = base_summary.variance
    second = correction_summary.variance + correction_summary.mean**2
    mean_square = correction_summary.mean**2
    cost = correction_summary.mean_cost

    def predicted(q: float) -> float:
        """Return ``(V + m/q - mu'^2)(c + q c')``.

        References:
            [EQ-MLAB-THIN-001], [EQ-MLAB-SUM-001], [EQ-MLAB-WNV-001].

        """
        return (variance + second / q - mean_square) * (5.0 + q * cost)

    grid = [10.0 ** (-1.6 + 1.6 * i / 200) for i in range(201)]
    curve = [predicted(q) for q in grid]
    best = min(range(len(grid)), key=lambda i: curve[i])
    return {
        "measured": measured,
        "grid": grid,
        "curve": curve,
        "best": grid[best],
        "separated": ranking.separated(),
    }


def _thinning_efficiency() -> tuple[Any, dict[str, str]]:
    """Draw measured and predicted work-normalized variance against ``q``.

    References:
        [EQ-MLAB-WNV-001], [EQ-MLAB-THIN-001], [REF-RHEE-GLYNN-2015].

    """
    example = _example()
    measured = example["measured"]
    facts = {
        "full": f"{measured[1.0][0]:.3f}",
        "tenth": f"{measured[0.1][0]:.3f}",
        "optimum": f"{example['best']:.3f}",
        "replicates": f"{REPLICATES}",
    }
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    theme.line(
        axes, example["grid"], example["curve"], color=theme.role_colour("model")
    )
    observed = theme.role_colour("observed")
    for q in PROBABILITIES:
        value, log_error = measured[q]
        low = value * math.exp(-2.576 * log_error)
        high = value * math.exp(2.576 * log_error)
        theme.line(axes, [q, q], [low, high], color=observed)
    theme.points(
        axes,
        list(PROBABILITIES),
        [measured[q][0] for q in PROBABILITIES],
        color=observed,
        size=40.0,
    )
    axes.set_xscale("log")
    axes.set_xlim(0.025, 1.3)
    axes.set_ylim(0.0, 0.7)
    theme.reference_line(axes, x=example["best"], color=palette.muted)
    axes.annotate(
        f"predicted minimum at q = {facts['optimum']}",
        xy=(example["best"], 0.06),
        xytext=(6.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    axes.grid(axis="x", visible=False)
    theme.axis_labels(
        axes, "Evaluation probability q of the correction", "Variance x mean cost"
    )
    theme.title_block(
        axes,
        "Evaluating the correction one time in ten lowers the work\n"
        "for a given accuracy and leaves the expectation unchanged",
        f"Points: {REPLICATES} replicates, 99% intervals (seed 3) · "
        "line: predicted from moments",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="methodlab-thinning-efficiency",
        guide="method-development",
        alt=(
            "Work-normalized variance of the derived estimator against the "
            "evaluation probability of its correction term on a logarithmic "
            "horizontal axis: measured points with intervals and the predicted "
            "curve"
        ),
        draw=_thinning_efficiency,
    ),
]
