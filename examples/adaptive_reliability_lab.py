"""Adaptive reliability laboratory: CE guard, lines, design points, AK-MCS.

Run with ``PYTHONPATH=src python examples/adaptive_reliability_lab.py``. All
numbers are printed. When Matplotlib is installed three figures are written
to ``examples/generated/``: the four-branch design-point map, the AK-MCS
population by predicted class, and the AK-MCS learning curve. Every
stochastic step is seeded, so repeated runs print identical values.
"""

from __future__ import annotations

import importlib.util
import math
import random
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from probstat.distributions import Normal
from probstat.foundations.random_streams import resolve_stream
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.reliability import (
    ModelLimitState,
    ak_mcs,
    bivariate_normal_cdf,
    cross_entropy_importance_sampling,
    find_design_points,
    form_hlrf,
    line_sampling,
    mixture_importance_sampling,
    model_monte_carlo,
    multi_point_form,
)
from probstat.uq import QuantityOfInterest

SQRT2 = math.sqrt(2.0)
Y = (QuantityOfInterest("y", "limit-state value"),)
OUTPUT = Path(__file__).parent / "generated"


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def save(figure_or_axes: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    figure = getattr(figure_or_axes, "figure", figure_or_axes)
    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def limit_state_grid(
    function: Callable[[tuple[float, ...]], float], extent: float, nodes: int
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate ``G`` on a square grid for the limit-state contour.

    References:
        [EQ-REL-MLS-001]. No further equation is evaluated.

    """
    axis = [-extent + 2.0 * extent * k / (nodes - 1) for k in range(nodes)]
    return axis, axis, [[function((x, y)) for x in axis] for y in axis]


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
        return {"y": function(tuple(p[name] for name in names))}

    return ModelLimitState(ProbabilisticModel(inputs, response, Y), "y", 0.0, "below")


def four_branch(u: tuple[float, ...]) -> float:
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


def cross_entropy_section() -> None:
    """Show the CE covariance collapse, its warning, and the floor fix.

    References:
        [EQ-REL-CE-002], [EQ-REL-CE-003], [REF-RUBINSTEIN-KROESE-2004].

    """
    print("1. CE-IS covariance adaptation on G = 4 - (u1 + u2)/sqrt2")
    exact = standard_normal_cdf(-4.0)
    print(f"   exact Pf = {exact:.4e}")
    print("   seed  floor   Pf          Pf/exact  c.o.v.  z      lam_min  flag")
    for seed in (1, 2, 3):
        for floor in (None, 0.55):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = cross_entropy_importance_sampling(
                    make_state(lambda u: 4.0 - (u[0] + u[1]) / SQRT2, 2),
                    samples_per_level=1000,
                    adapt_covariance=True,
                    covariance_floor=floor,
                    seed=seed,
                )
            z = (
                (result.probability - exact) / result.standard_error
                if result.standard_error > 0.0
                else math.nan
            )
            label = "none" if floor is None else f"{floor:.2f}"
            print(
                f"   {seed:<5d} {label:<6s}  {result.probability:.4e}  "
                f"{result.probability / exact:7.3f}  "
                f"{result.coefficient_of_variation:6.3f}  {z:6.2f}  "
                f"{result.min_covariance_eigenvalue:7.4f}  "
                f"{result.weights_degenerate}"
            )


def line_sampling_section() -> None:
    """Line sampling on a plane, a parabola, and a band with rootless lines.

    References:
        [EQ-REL-LINE-001], [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004].

    """
    print("\n2. Line sampling")
    beta = 3.5
    alpha = 1.0 / math.sqrt(3.0)
    plane = make_state(lambda u: beta - alpha * (u[0] + u[1] + u[2]), 3)
    result = line_sampling(plane, 30, seed=11)
    exact = standard_normal_cdf(-beta)
    print(
        f"   plane d=3, beta=3.5: Pf={result.probability:.6e} exact={exact:.6e} "
        f"c.o.v.={result.coefficient_of_variation:.1e} "
        f"calls={result.evaluations}+{result.form_evaluations} (FORM)"
    )
    kappa = 0.3
    nodes, weights = gauss_hermite(60)
    exact = math.fsum(
        w * standard_normal_cdf(-(3.0 + 0.5 * kappa * z * z))
        for z, w in zip(nodes, weights, strict=True)
    )
    parabola = make_state(lambda u: 3.0 - u[1] + 0.5 * kappa * u[0] ** 2, 2)
    result = line_sampling(parabola, 200, seed=5)
    print(
        f"   parabola kappa=0.3: Pf={result.probability:.5e} "
        f"quadrature={exact:.5e} SE={result.standard_error:.2e} "
        f"z={(result.probability - exact) / result.standard_error:.2f} "
        f"calls={result.evaluations + result.form_evaluations}"
    )
    band = make_state(lambda u: 3.0 - u[1] if abs(u[0]) < 1.0 else 5.0 + u[1] ** 2, 2)
    result = line_sampling(band, 400, direction=(0.0, 1.0), seed=2)
    exact = (2.0 * standard_normal_cdf(1.0) - 1.0) * standard_normal_cdf(-3.0)
    rootless = sum(1 for c in result.distances if c == math.inf)
    print(
        f"   safe band: Pf={result.probability:.5e} exact={exact:.5e} "
        f"SE={result.standard_error:.2e}; {rootless}/400 lines without a root "
        "contribute exactly 0"
    )


def design_point_section() -> None:
    """Multiple design points, multi-point FORM, and mixture IS.

    References:
        [EQ-REL-MDP-001], [EQ-REL-MDP-002], [EQ-REL-MDP-003],
        [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """
    print("\n3. Multiple design points")
    b1, b2, a2 = 3.0, 3.2, (-0.6, 0.8)
    state = make_state(
        lambda u: min(b1 - u[0], b2 - (a2[0] * u[0] + a2[1] * u[1])), 2
    )
    points = find_design_points(state, 4)
    exact = (
        standard_normal_cdf(-b1)
        + standard_normal_cdf(-b2)
        - bivariate_normal_cdf(-b1, -b2, a2[0])
    )
    combined = multi_point_form(points)
    mixture = mixture_importance_sampling(state, points, 2000, seed=7)
    print(
        f"   two planes: betas={tuple(round(b, 6) for b in points.betas)} "
        f"calls={points.evaluations} ({points.message})"
    )
    print(
        f"   exact={exact:.6e}  FORM(first point)={points.forms[0].probability:.6e}"
        f"  multi-point FORM={combined.probability:.6e} ({combined.method})"
    )
    print(
        f"   mixture IS (2000): {mixture.probability:.4e} +- "
        f"{mixture.standard_error:.1e} "
        f"z={(mixture.probability - exact) / mixture.standard_error:.2f}"
    )
    state = make_state(four_branch, 2)
    # G has a kink at the origin (two branches tie), where the central
    # difference gradient vanishes, so the first FORM run starts off-centre.
    first = form_hlrf(state, start=(0.3, 0.1))
    points = find_design_points(state, 6, first=first)
    combined = multi_point_form(points, samples=20_000, seed=1)
    mixture = mixture_importance_sampling(state, points, 4000, seed=3)
    crude = model_monte_carlo(state, 200_000, seed=4)
    print(
        f"   four-branch: {len(points.forms)} design points, betas="
        f"{tuple(round(b, 4) for b in points.betas)}, calls={points.evaluations}"
    )
    print(
        f"   multi-point FORM={combined.probability:.4e} ({combined.method}, "
        f"SE {combined.standard_error:.1e}); mixture IS (4000)="
        f"{mixture.probability:.4e} +- {mixture.standard_error:.1e}; "
        f"crude MC (2e5)={crude.probability:.4e} +- {crude.standard_error:.1e}"
    )
    if has_matplotlib():
        from probstat.visualization.reliability_plots import plot_design_points

        rng = random.Random(3)
        centres = mixture.centers
        samples = []
        for _ in range(2000):
            cx, cy = centres[rng.randrange(len(centres))]
            samples.append((rng.gauss(cx, 1.0), rng.gauss(cy, 1.0)))
        failed = [four_branch(u) <= 0.0 for u in samples]
        axes = plot_design_points(
            points,
            limit_state_grid=limit_state_grid(four_branch, 5.0, 161),
            samples=samples,
            failed=failed,
            combined=combined,
            mixture=mixture,
            extent=5.0,
        )
        print(f"   figure: {save(axes, 'adaptive-reliability-design-points.png')}")


def ak_mcs_section() -> None:
    """AK-MCS on the four-branch system against MC on the same population.

    References:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    print("\n4. AK-MCS on the four-branch system (k = 6)")
    print("   seed  N      calls  added  Pf_AK       Pf_MC(same)  wrong  c.o.v.")
    first = None
    for seed in (1, 2):
        result = ak_mcs(
            make_state(four_branch, 2),
            10_000,
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=seed,
        )
        print(
            f"   {seed:<5d} {result.population:<6d} {result.evaluations:<6d} "
            f"{result.added:<6d} {result.probability:.4e}  "
            f"{result.reference_probability:.4e}   {result.misclassified:<5d}  "
            f"{result.coefficient_of_variation:.3f}"
        )
        first = first or result
    if has_matplotlib() and first is not None:
        ak_mcs_figures(first, seed=1)
    result = ak_mcs(
        make_state(four_branch, 2),
        5000,
        target_cov=0.15,
        population_growth=5000,
        compare_with_monte_carlo=True,
        seed=1,
    )
    print(
        f"   start N=5000, c.o.v. target 0.15: N grew to {result.population}, "
        f"calls={result.evaluations}, Pf_AK={result.probability:.4e}, "
        f"Pf_MC(same)={result.reference_probability:.4e}, "
        f"wrong={result.misclassified}, "
        f"c.o.v.={result.coefficient_of_variation:.3f} ({result.message})"
    )


def ak_mcs_figures(result: Any, *, seed: int) -> None:
    """Draw the AK-MCS population map and learning curve of one run.

    ``ak_mcs`` does not store its population, so it is redrawn from the same
    child stream (``"ak-mcs"``, ``"population"``) of the run's seed; the
    number of points the final surrogate predicts as failed is printed next
    to the run's own count as a check that the redraw is identical.

    References:
        [EQ-REL-AKMCS-001], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    from probstat.visualization.reliability_plots import (
        plot_ak_mcs_classification,
        plot_ak_mcs_convergence,
    )

    generator = resolve_stream(seed).child("ak-mcs", "population").generator()
    population = [
        (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
        for _ in range(result.population)
    ]
    redrawn = sum(1 for u in population if result.surrogate.predict(u) <= 0.0)
    print(
        f"   redrawn population: {redrawn} predicted failures "
        f"(run reported {result.predicted_failures})"
    )
    axes = plot_ak_mcs_classification(
        result,
        population,
        limit_state_grid=limit_state_grid(four_branch, 5.0, 121),
        extent=5.0,
    )
    print(f"   figure: {save(axes, 'adaptive-reliability-ak-mcs-population.png')}")
    figure = plot_ak_mcs_convergence(result)
    print(f"   figure: {save(figure, 'adaptive-reliability-ak-mcs-convergence.png')}")


def main() -> None:
    """Run every section and print the elapsed time.

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-DERKIUREGHIAN-DAKESSIAN-1998],
        [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004].

    """
    start = time.perf_counter()
    cross_entropy_section()
    line_sampling_section()
    design_point_section()
    ak_mcs_section()
    print(f"\nelapsed {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
