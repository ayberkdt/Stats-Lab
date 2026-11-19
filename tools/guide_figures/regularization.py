"""Figures of the regularization and ill-posed problems guide.

The figures run the seeded backward heat example of the guide (ADR-0110):
the Picard plot of the noisy data, and Tikhonov solutions on a grid of
regularization parameters, their true relative errors, and the parameters
chosen by the L-curve, generalized cross-validation, and the discrepancy
principle. The facts are the numbers
the caption in the guide quotes.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-HANSEN-2008],
    [REF-HANSEN-OLEARY-1993], [REF-MOROZOV-1966].

"""

from __future__ import annotations

import functools
import math
from typing import Any

from probstat.foundations import RandomStream
from probstat.inverse import (
    backward_heat_problem,
    discrepancy_principle,
    generalized_cross_validation,
    l_curve,
    picard_coefficients,
    tikhonov,
)
from probstat.linalg import singular_value_decomposition
from probstat.visualization import theme

from . import GuideFigure


def _norm(vector: Any) -> float:
    """Return the Euclidean norm.

    References:
        [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(value * value for value in vector))


@functools.cache
def _noisy_heat() -> tuple[Any, Any, tuple[float, ...], float, float]:
    """Return the heat problem, its SVD, noisy data, noise sd, and noise norm.

    The guide's data: ``n = 32``, diffusion time 0.005, 1% white noise from
    seed 2026 (child stream ``noise``).

    References:
        [EQ-INV-REG-PROB-001], [EQ-LINALG-SVD-001], [REF-HANSEN-2008].

    """
    problem = backward_heat_problem(32, time=0.005)
    svd = singular_value_decomposition(problem.operator)
    exact_data = problem.exact_data
    scale = 0.01 * _norm(exact_data) / math.sqrt(len(exact_data))
    generator = RandomStream(2026).child("noise").generator()
    noise = [generator.gauss(0.0, scale) for _ in exact_data]
    data = tuple(b + e for b, e in zip(exact_data, noise, strict=True))
    return problem, svd, data, scale, _norm(noise)


@functools.cache
def _heat_example() -> dict[str, Any]:
    """Run the guide's example: heat problem, 1% noise, seed 2026.

    References:
        [EQ-INV-REG-001], [EQ-INV-REG-003], [EQ-INV-REG-004], [EQ-INV-REG-005].

    """
    problem, svd, data, _, delta = _noisy_heat()

    def error(solution: Any) -> float:
        """Return the relative error against the exact initial state.

        References:
            [REF-HANSEN-2008, Section 2.5]. No equation is evaluated.

        """
        exact = problem.exact_solution
        difference = [a - b for a, b in zip(solution, exact, strict=True)]
        return _norm(difference) / _norm(exact)

    # The guide's scan: 601 log-spaced values in [1e-5, 1]; drawn from 1e-4.
    grid = [10.0 ** (-5.0 + 5.0 * index / 600) for index in range(120, 601)]
    errors = [
        error(tikhonov(problem.operator, data, lam, decomposition=svd).solution)
        for lam in grid
    ]
    best = min(range(len(grid)), key=lambda index: errors[index])
    choices = {
        "L-curve": l_curve(problem.operator, data, decomposition=svd),
        "GCV": generalized_cross_validation(problem.operator, data, decomposition=svd),
        "discrepancy": discrepancy_principle(
            problem.operator, data, delta, decomposition=svd
        ),
    }
    chosen = {
        name: (choice.regularization, error(choice.solution.solution))
        for name, choice in choices.items()
    }
    return {
        "grid": grid,
        "errors": errors,
        "best": (grid[best], errors[best]),
        "chosen": chosen,
    }


def _parameter_choice() -> tuple[Any, dict[str, str]]:
    """Draw the true Tikhonov error against lambda with the three choices.

    References:
        [EQ-INV-REG-002], [EQ-INV-REG-003], [EQ-INV-REG-004], [EQ-INV-REG-005],
        [REF-HANSEN-2008, Section 2.9].

    """
    example = _heat_example()
    grid, errors = example["grid"], example["errors"]
    best_lambda, best_error = example["best"]
    chosen = example["chosen"]
    facts = {
        "best": f"{best_error:.4f}",
        "best_lambda": f"{best_lambda:.3g}",
        "discrepancy": f"{chosen['discrepancy'][1]:.4f}",
        "gcv": f"{chosen['GCV'][1]:.4f}",
        "lcurve": f"{chosen['L-curve'][1]:.3f}",
    }
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    theme.line(axes, grid, errors, color=theme.role_colour("model"))
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(1e-4, 1.0)
    axes.set_ylim(1e-2, 40.0)
    theme.reference_line(axes, x=best_lambda, label="error-optimal", entity="truth")
    styles = (
        ("L-curve", "series-3", "D"),
        ("GCV", "series-4", "^"),
        ("discrepancy", "series-5", "o"),
    )
    for name, role, marker in styles:
        lam, value = chosen[name]
        colour = theme.role_colour(role)
        theme.points(axes, [lam], [value], color=colour, size=64.0, marker=marker)
        axes.annotate(
            f"{name}: {value:.3g}",
            (lam, value),
            xytext=(-9.0, -12.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=theme.ANNOTATION_SIZE,
            fontweight="bold",
            color=palette.ink,
        )
    axes.grid(axis="x", visible=False)
    axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    theme.axis_labels(axes, "Regularization parameter lambda", "Relative error")
    theme.title_block(
        axes,
        "The discrepancy principle lands near the error minimum;\n"
        "GCV and the L-curve regularize too little",
        "Backward heat problem, n = 32, 1% noise (seed 2026) · line: "
        "true error of the Tikhonov solution",
    )
    return axes.figure, facts


def _picard_plot() -> tuple[Any, dict[str, str]]:
    """Draw the Picard plot of the noisy backward heat data.

    References:
        [EQ-INV-REG-007], [EQ-LINALG-SVD-001], [REF-HANSEN-2008, Section 2.4].

    """
    problem, svd, data, noise_sd, _ = _noisy_heat()
    picard = picard_coefficients(problem.operator, data, decomposition=svd)
    sigma = picard.singular_values
    beta = picard.data_coefficients
    ratio = picard.solution_coefficients
    turn = min(range(len(ratio)), key=lambda i: ratio[i]) + 1
    tail = beta[turn:]
    facts = {
        "noise": f"{noise_sd:.4f}",
        "turn": f"{turn}",
        "tail": f"{math.fsum(tail) / len(tail):.4f}",
        "smallest": f"{sigma[-1]:.1e}",
    }
    index = list(range(1, len(sigma) + 1))
    axes = theme.new_axes(figsize=(7.2, 4.2))
    palette = theme.active_palette()
    series = (
        ("sigma_i", sigma, palette.muted, "o", theme.CONTEXT_WIDTH, None),
        ("|u_i'b|", beta, theme.role_colour("observed"), "s", theme.LINE_WIDTH, None),
        (
            "|u_i'b| / sigma_i",
            ratio,
            theme.role_colour("model"),
            "^",
            theme.LINE_WIDTH,
            theme.FITTED,
        ),
    )
    labels = []
    for name, values, colour, marker, width, dashes in series:
        theme.line(axes, index, values, color=colour, width=width, dashes=dashes)
        theme.points(axes, index, values, color=colour, marker=marker, size=22.0)
        labels.append((index[-1], values[-1], name, colour))
    axes.set_yscale("log")
    axes.set_xlim(0.5, len(sigma) + 0.5)
    axes.set_ylim(1e-11, 1e9)
    axes.set_yticks([1e-10, 1e-6, 1e-2, 1e2, 1e6])
    axes.minorticks_off()
    theme.reference_line(axes, y=noise_sd, entity="threshold")
    axes.annotate(
        f"noise sd s = {facts['noise']}",
        xy=(1.0, noise_sd),
        xycoords=("data", "data"),
        xytext=(0.0, -8.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        color=theme.entity_colour("threshold"),
    )
    theme.reference_line(axes, x=turn + 0.5, color=palette.muted)
    axes.annotate(
        f"noise takes over after i = {turn}",
        xy=(turn + 0.5, 1e7),
        xytext=(5.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.direct_labels(axes, labels, gap_points=13.0)
    axes.grid(axis="x", visible=False)
    theme.axis_labels(axes, "Index i", "Coefficient (log scale)")
    theme.title_block(
        axes,
        f"The data coefficients fall faster than the singular values up to "
        f"i = {turn},\nthen flatten at the noise level and the ratios explode",
        "Backward heat problem, n = 32, 1% noise (seed 2026) · the Picard "
        "condition holds only while |u_i'b| falls faster than sigma_i",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="regularization-parameter-choice",
        guide="regularization-and-ill-posed-problems",
        alt=(
            "Relative error of the Tikhonov solution against the regularization "
            "parameter on log axes, with the error-optimal parameter and the "
            "parameters chosen by the L-curve, GCV, and the discrepancy principle"
        ),
        draw=_parameter_choice,
    ),
    GuideFigure(
        name="regularization-picard-plot",
        guide="regularization-and-ill-posed-problems",
        alt=(
            "Singular values, data coefficients, and their ratios of the backward "
            "heat problem against the index on a log axis, with the noise level"
        ),
        draw=_picard_plot,
    ),
]
