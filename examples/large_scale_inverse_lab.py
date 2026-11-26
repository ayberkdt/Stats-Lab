"""Large-scale inverse laboratory: matrix-free deblurring and a Gaussian posterior.

Run with ``PYTHONPATH=src python examples/large_scale_inverse_lab.py``. The
lab deblurs the benchmark L04 three ways, reads a correlated Gaussian
benchmark as an exact posterior, prints a table of relative errors and
operator-application counts, and, with Matplotlib installed, writes two
figures to ``examples/generated/``. Every number comes from the run; the only
constant that fixes the data is ``SEED``.

Definitions:
    L04: one-dimensional Gaussian deblurring on ``[0, 1]`` with ``n`` cells,
        a piecewise-constant truth ``x`` and data ``b = A x + e``,
        ``e ~ N(0, s^2 I)``. With ``refinement = 2`` the data are generated
        on a grid twice as fine and cell-averaged, so ``A`` is a slightly
        wrong model of the data (no inverse crime).
    ``A``: the matrix-free convolution operator; only ``A v`` and ``A' v`` are
        ever applied. ``CountingOperator`` records both counts.
    relative error: ``||x_est - x|| / ||x||`` against the withheld truth of
        the solver's grid.
    early-stopped LSQR: the iterate of the first iteration whose whitened
        residual is at most ``tau sqrt(m)`` (discrepancy principle).
    path: affine Tikhonov solutions ``x_lambda`` on a log grid of ``lambda``,
        with GCV, UPRE and the discrepancy gap evaluated on the same grid.
    total variation: the minimizer of ``1/2 ||A x - b||^2 + alpha TV(x)``
        (primal-dual hybrid gradient); ``alpha`` is the largest grid value
        whose whitened residual is at most ``tau sqrt(m)``.
    L06: a linear Gaussian problem with prior ``x ~ N(0, C0)`` (exponential
        kernel) and local-average observations; its posterior is exact.

Assumptions:
    - The noise is white with known standard deviation; the discrepancy
      target ``sqrt(m)`` is the expected whitened noise norm, not the
      realized one, so a single noise draw may stop one iteration early or
      late.
    - The truth is piecewise constant, which suits total variation; the
      ranking of the three methods is a statement about this seed and this
      problem, not a universal ordering.
    - The deblurring operator carries a discretization error from the
      refinement, so no method can reach zero error even without noise.
    - For L06 the prior used by the posterior is the prior the truth was
      drawn from, so the posterior band is calibrated by construction.

Equations:
    EQ-INV-BENCH-001: benchmark data generation and the inverse-crime flag.
    EQ-INV-LIN-001: weighted affine Tikhonov functional behind the path.
    EQ-INV-ITER-002: discrepancy principle as the LSQR stopping rule.
    EQ-INV-SEL-001: GCV, UPRE and discrepancy gap on the path.
    EQ-INV-SPARSE-003: discrete total variation.
    EQ-INV-LG-001: linear Gaussian posterior mean and covariance.
    EQ-INV-LG-002: log evidence of the data.

Diagnostics:
    - The LSQR run prints its stop reason and iteration count; the
      discrepancy rule must have fired before the iteration cap.
    - Each criterion on the path prints its status (``interior`` means a
      proper optimum on the grid; ``boundary`` would mean the grid is too
      short).
    - The total-variation scan prints the whitened residual and the
      ``converged`` flag of every ``alpha``; a solve that hit the iteration
      cap is shown as such.
    - The Gaussian part checks the data-space and parameter-space solvers
      against each other (maximum mean difference and evidence difference,
      both rounding-sized) and reports the fraction of grid points inside
      the two-standard-deviation band.

Cost and error:
    - Cost is the number of applications of ``A`` and ``A'`` seen by the
      counting wrapper, printed per method. The path materializes the
      stacked operator for an exact trace, so its count is much larger than
      the Krylov count and is shared by the GCV, UPRE and discrepancy rows;
      the total-variation count is the whole ``alpha`` scan.
    - Errors are for one noise realization; across seeds the order of the
      methods can change.
    - The grid, the ``alpha`` list, and the iteration caps limit the
      attainable accuracy and are part of the printed result.

References:
    REF-GAZZOLA-HANSEN-NAGY-2019: benchmark deblurring problems.
    REF-HANSEN-2008: semi-convergence and the discrepancy principle.
    REF-PAIGE-SAUNDERS-1982: LSQR.
    REF-GOLUB-HEATH-WAHBA-1979: generalized cross-validation.
    REF-RUDIN-OSHER-FATEMI-1992: total variation regularization.
    REF-CHAMBOLLE-POCK-2011: primal-dual hybrid gradient.
    REF-TARANTOLA-2005: the linear Gaussian posterior.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.inverse import (
    LinearInverseProblem,
    correlated_gaussian_benchmark,
    deconvolution_benchmark,
    expected_noise_norm,
    krylov_regularization,
    linear_gaussian_posterior,
    logarithmic_grid,
    regularization_path,
    select_regularization,
    total_variation_regularized_least_squares,
)
from probstat.linalg import CountingOperator

# Root seed of the benchmark noise (L04) and of the truth and noise of L06.
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
SIZE = 48
REFINEMENT = 2
LAMBDA_GRID = logarithmic_grid(1e-3, 1e2, 24)
ALPHA_GRID = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1)
TAU = 1.01
TV_ITERATIONS = 3000
BAND = 2.0


@dataclass(frozen=True)
class Row:
    """One line of the result table.

    References:
        [EQ-INV-BENCH-001].

    """

    method: str
    choice: str
    status: str
    error: float
    forward: int
    transpose: int


@dataclass(frozen=True)
class GaussianPart:
    """Summary of the exact posterior of the correlated Gaussian benchmark.

    References:
        [EQ-INV-LG-001], [EQ-INV-LG-002].

    """

    grid: tuple[float, ...]
    truth: tuple[float, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    data: tuple[float, ...]
    log_evidence: float
    prior_error: float
    posterior_error: float
    inside: int
    solver_mean_difference: float
    solver_evidence_difference: float


@dataclass(frozen=True)
class LabResult:
    """Everything the lab prints and plots.

    References:
        [EQ-INV-BENCH-001].

    """

    grid: tuple[float, ...]
    truth: tuple[float, ...]
    estimates: dict[str, tuple[float, ...]]
    rows: tuple[Row, ...]
    lsqr_iterations: int
    lsqr_stop: str
    lsqr_oracle_iteration: int | None
    path_statuses: dict[str, str]
    path_lambdas: dict[str, float | None]
    tv_scan: tuple[tuple[float, float, bool, str, float], ...]
    tv_alpha: float | None
    noise_level: float
    gaussian: GaussianPart


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def relative_error(estimate: Sequence[float], truth: Sequence[float]) -> float:
    """Return ``||estimate - truth|| / ||truth||``.

    References:
        [REF-HANSEN-2008, Section 2.5]. No equation is evaluated.

    """
    top = math.fsum((a - b) ** 2 for a, b in zip(estimate, truth, strict=True))
    bottom = math.fsum(b * b for b in truth)
    return math.sqrt(top / bottom)


def deblurring_part(size: int = SIZE) -> dict[str, Any]:
    """Solve L04 by early-stopped LSQR, the Tikhonov path, and total variation.

    Args:
        size: Number of cells ``n``.

    Returns:
        The benchmark, the rows of the table, and per-method diagnostics.

    References:
        [EQ-INV-BENCH-001], [EQ-INV-LIN-001], [EQ-INV-ITER-002],
        [EQ-INV-SEL-001], [EQ-INV-SPARSE-003].

    """
    bench = deconvolution_benchmark(size, refinement=REFINEMENT, seed=SEED)
    truth = bench.truth
    m = len(bench.data)
    target = expected_noise_norm(m)
    rows: list[Row] = []
    estimates: dict[str, tuple[float, ...]] = {}

    # (a) early-stopped LSQR with the discrepancy principle
    counted = CountingOperator(bench.operator)
    problem = LinearInverseProblem.create(counted, bench.data, noise=bench.noise)
    lsqr = krylov_regularization(
        problem,
        method="lsqr",
        noise_norm=target,
        tau=TAU,
        max_iterations=200,
        truth=truth,
    )
    estimates["LSQR"] = lsqr.estimate
    rows.append(
        Row(
            "LSQR",
            f"k = {lsqr.iterations}",
            lsqr.stop_reason,
            relative_error(lsqr.estimate, truth),
            counted.counts.forward,
            counted.counts.transpose,
        )
    )

    # (b) regularization path with three selection rules
    counted = CountingOperator(bench.operator)
    problem = LinearInverseProblem.create(counted, bench.data, noise=bench.noise)
    path = regularization_path(
        problem, LAMBDA_GRID, noise_norm=target, tau=1.0, truth=truth
    )
    statuses: dict[str, str] = {}
    lambdas: dict[str, float | None] = {}
    for criterion in ("gcv", "upre", "discrepancy"):
        choice = select_regularization(path, criterion)
        statuses[criterion] = choice.status
        lambdas[criterion] = choice.regularization
        if choice.estimate is None:
            continue
        name = f"Tikhonov {criterion.upper()}"
        estimates[name] = choice.estimate
        rows.append(
            Row(
                name,
                f"lambda = {choice.regularization:.3g}",
                choice.status,
                relative_error(choice.estimate, truth),
                counted.counts.forward,
                counted.counts.transpose,
            )
        )

    # (c) total variation, alpha by the discrepancy principle on a short list
    counted = CountingOperator(bench.operator)
    scan: list[tuple[float, float, bool, str, float]] = []
    results = []
    for alpha in ALPHA_GRID:
        result = total_variation_regularized_least_squares(
            counted, bench.data, alpha, max_iterations=TV_ITERATIONS
        )
        misfit = math.sqrt(2.0 * result.data_misfit) / bench.noise_level
        scan.append((alpha, misfit, result.converged, result.stop_reason, 0.0))
        results.append(result)
    feasible = [i for i, entry in enumerate(scan) if entry[1] <= TAU * target]
    pick = max(feasible) if feasible else 0
    chosen = results[pick]
    estimates["TV"] = chosen.solution
    rows.append(
        Row(
            "Total variation",
            f"alpha = {ALPHA_GRID[pick]:g}",
            chosen.stop_reason if feasible else "discrepancy_unattainable",
            relative_error(chosen.solution, truth),
            counted.counts.forward,
            counted.counts.transpose,
        )
    )
    scan = [
        (alpha, misfit, converged, reason, relative_error(r.solution, truth))
        for (alpha, misfit, converged, reason, _), r in zip(scan, results, strict=True)
    ]
    return {
        "bench": bench,
        "rows": tuple(rows),
        "estimates": estimates,
        "lsqr": lsqr,
        "statuses": statuses,
        "lambdas": lambdas,
        "scan": tuple(scan),
        "alpha": ALPHA_GRID[pick] if feasible else None,
    }


def gaussian_part() -> GaussianPart:
    """Compute the exact posterior of the correlated Gaussian benchmark L06.

    Returns:
        Posterior mean, marginal standard deviations, log evidence, and the
        agreement of the two solver paths.

    References:
        [EQ-INV-LG-001], [EQ-INV-LG-002], [EQ-INV-BENCH-001].

    """
    bench = correlated_gaussian_benchmark(seed=SEED)
    posterior = linear_gaussian_posterior(
        bench.operator,
        bench.data,
        noise=bench.noise,
        prior_covariance=bench.prior_covariance,
    )
    other = linear_gaussian_posterior(
        bench.operator,
        bench.data,
        noise=bench.noise,
        prior_covariance=bench.prior_covariance,
        method="parameter_space",
    )
    variances = other.marginal_variances()
    sd = tuple(math.sqrt(v) for v in variances)
    n = len(bench.truth)
    inside = sum(
        abs(mean - x) <= BAND * s
        for mean, x, s in zip(posterior.mean, bench.truth, sd, strict=True)
    )
    return GaussianPart(
        grid=tuple(i / (n - 1) for i in range(n)),
        truth=bench.truth,
        mean=posterior.mean,
        sd=sd,
        data=bench.data,
        log_evidence=posterior.log_evidence,
        prior_error=relative_error((0.0,) * n, bench.truth),
        posterior_error=relative_error(posterior.mean, bench.truth),
        inside=inside,
        solver_mean_difference=max(
            abs(a - b) for a, b in zip(posterior.mean, other.mean, strict=True)
        ),
        solver_evidence_difference=abs(posterior.log_evidence - other.log_evidence),
    )


def compute(size: int = SIZE) -> LabResult:
    """Run both parts and return the numbers the lab reports.

    Args:
        size: Number of deblurring cells.

    Returns:
        :class:`LabResult`.

    References:
        [EQ-INV-BENCH-001], [EQ-INV-LG-001].

    """
    deblur = deblurring_part(size)
    bench = deblur["bench"]
    n = len(bench.truth)
    lsqr = deblur["lsqr"]
    return LabResult(
        grid=tuple((i + 0.5) / n for i in range(n)),
        truth=bench.truth,
        estimates=deblur["estimates"],
        rows=deblur["rows"],
        lsqr_iterations=lsqr.iterations,
        lsqr_stop=lsqr.stop_reason,
        lsqr_oracle_iteration=lsqr.oracle_iteration,
        path_statuses=deblur["statuses"],
        path_lambdas=deblur["lambdas"],
        tv_scan=deblur["scan"],
        tv_alpha=deblur["alpha"],
        noise_level=bench.noise_level,
        gaussian=gaussian_part(),
    )


def report(result: LabResult) -> None:
    """Print the deblurring diagnostics, the Gaussian summary, and the table.

    References:
        [EQ-INV-ITER-002], [EQ-INV-SEL-001], [EQ-INV-LG-002].

    """
    n = len(result.truth)
    print(f"1. Deblurring L04 (n = {n}, refinement {REFINEMENT}, seed {SEED})")
    print(
        f"   LSQR: stop reason {result.lsqr_stop} at iteration "
        f"{result.lsqr_iterations}; error-minimizing iteration "
        f"{result.lsqr_oracle_iteration}"
    )
    for criterion, status in result.path_statuses.items():
        value = result.path_lambdas[criterion]
        shown = "none" if value is None else f"{value:.3g}"
        print(f"   path {criterion:11s} lambda = {shown:>8s}  status {status}")
    print("   alpha     whitened misfit  converged  stop reason     error")
    for alpha, misfit, converged, reason, error in result.tv_scan:
        print(
            f"   {alpha:<8g}  {misfit:>13.3f}  {converged!s:>9}  {reason:<14s}"
            f"  {error:.4f}"
        )
    gaussian = result.gaussian
    print()
    print(f"2. Gaussian posterior L06 (n = {len(gaussian.truth)}, seed {SEED})")
    print(
        f"   relative error: prior mean {gaussian.prior_error:.4f}, "
        f"posterior mean {gaussian.posterior_error:.4f}"
    )
    print(
        f"   marginal sd {min(gaussian.sd):.4f} to {max(gaussian.sd):.4f}; "
        f"{gaussian.inside} of {len(gaussian.truth)} grid points inside "
        f"+-{BAND:g} sd"
    )
    print(f"   log evidence {gaussian.log_evidence:.4f}")
    print(
        "   data-space against parameter-space: max |mean difference| "
        f"{gaussian.solver_mean_difference:.1e}, |evidence difference| "
        f"{gaussian.solver_evidence_difference:.1e}"
    )
    print()
    print("3. Table: relative error and applications of A (forward / transpose)")
    print("   method                choice          status                 error"
          "   forward  transpose")  # fmt: skip
    for row in result.rows:
        print(
            f"   {row.method:<21s} {row.choice:<15s} {row.status:<20s} "
            f"{row.error:>8.4f} {row.forward:>8d} {row.transpose:>10d}"
        )
    print(
        f"   Gaussian posterior mean (L06)                          "
        f"{gaussian.posterior_error:>8.4f}"
    )
    print()


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


def deblurring_figure(result: LabResult) -> str:
    """Draw the truth and the three reconstructions of L04.

    References:
        [EQ-INV-BENCH-001], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.4))
    truth_style = theme.entity_style("truth")
    theme.line(
        axes,
        result.grid,
        result.truth,
        color=truth_style.colour(),
        dashes=truth_style.dashes,
        width=truth_style.width,
    )
    styles = (
        ("LSQR", "series-3", theme.CONTEXT_WIDTH, None),
        ("Tikhonov DISCREPANCY", "series-4", theme.CONTEXT_WIDTH, theme.SIMULATED),
        ("TV", "model", theme.LINE_WIDTH, None),
    )
    errors = {row.method: row.error for row in result.rows}
    names = {
        "LSQR": "LSQR",
        "Tikhonov DISCREPANCY": "Tikhonov DISCREPANCY",
        "TV": "Total variation",
    }
    labels = [
        (result.grid[-1], result.truth[-1], "truth", truth_style.colour()),
    ]
    for key, role, width, dashes in styles:
        estimate = result.estimates[key]
        colour = theme.role_colour(role)
        theme.line(
            axes, result.grid, estimate, color=colour, width=width, dashes=dashes
        )
        labels.append(
            (
                result.grid[-1],
                estimate[-1],
                f"{key} ({errors[names[key]]:.3f})",
                colour,
            )
        )
    axes.set_xlim(0.0, 1.0)
    theme.direct_labels(axes, labels, gap_points=13.0)
    theme.axis_labels(axes, "Position s", "Signal x(s)")
    best = min(result.rows, key=lambda row: row.error)
    theme.title_block(
        axes,
        f"{best.method} has the smallest error on this seed ({best.error:.1%})",
        "Labels: method (relative error). Deblurring L04 with a 2x finer "
        "data grid and 1% noise",
    )
    return save(axes.figure, "large-scale-inverse-deblurring.png")


def posterior_figure(result: LabResult) -> str:
    """Draw the exact posterior band of L06 against the truth.

    References:
        [EQ-INV-LG-001], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    gaussian = result.gaussian
    axes = theme.new_axes(figsize=(7.2, 4.4))
    low = [m - BAND * s for m, s in zip(gaussian.mean, gaussian.sd, strict=True)]
    high = [m + BAND * s for m, s in zip(gaussian.mean, gaussian.sd, strict=True)]
    theme.band(axes, gaussian.grid, low, high, color=theme.role_colour("model"))
    style = theme.entity_style("truth")
    theme.line(
        axes,
        gaussian.grid,
        gaussian.truth,
        color=style.colour(),
        dashes=style.dashes,
        width=style.width,
    )
    mean_colour = theme.role_colour("model")
    theme.line(axes, gaussian.grid, gaussian.mean, color=mean_colour)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(
        min(low + list(gaussian.truth)) - 0.2, max(high + list(gaussian.truth)) + 0.2
    )
    theme.direct_labels(
        axes,
        [
            (gaussian.grid[-1], gaussian.mean[-1], "posterior mean", mean_colour),
            (gaussian.grid[-1], gaussian.truth[-1], "truth", style.colour()),
        ],
        gap_points=13.0,
    )
    theme.axis_labels(axes, "Position s", "Field x(s)")
    theme.title_block(
        axes,
        f"The exact posterior holds the truth at {gaussian.inside} of "
        f"{len(gaussian.truth)} points",
        f"Band: posterior mean +-{BAND:g} marginal sd. Relative error "
        f"{gaussian.posterior_error:.3f} against {gaussian.prior_error:.3f} "
        f"for the prior mean; log evidence {gaussian.log_evidence:.2f}",
    )
    return save(axes.figure, "large-scale-inverse-posterior.png")


def main() -> None:
    """Run both parts, print the table, and write the figures.

    References:
        [EQ-INV-BENCH-001], [EQ-INV-LG-001].

    """
    result = compute()
    report(result)
    if not has_matplotlib():
        print("Matplotlib is not installed; figures skipped")
        return
    written = (deblurring_figure(result), posterior_figure(result))
    print("figures: " + ", ".join(written))


if __name__ == "__main__":
    main()
