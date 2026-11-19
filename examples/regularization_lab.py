"""Regularization laboratory: the backward heat problem from noisy data.

Run with ``PYTHONPATH=src python examples/regularization_lab.py``. Every
number is printed; with Matplotlib installed two figures are written to
``examples/generated/``: the reconstructions at the chosen regularization
parameters against the exact initial temperature, and the L-curve with the
three parameter choices. The noise is drawn from ``SEED``, so repeated runs
print identical values; the run takes about half a minute.

Definitions:
    ``A``: the ``32 x 32`` backward heat operator (diffusion time 0.005)
        with singular values ``sigma_i`` and SVD ``A = U diag(sigma) V'``.
    ``x``: the exact initial temperature ``(27/4) s (1 - s)^2``; ``b = A x +
        e`` with white noise ``e`` of standard deviation ``s`` equal to 1 %
        of the root-mean-square of ``A x``; ``delta = ||e||``.
    ``x_lambda``: the Tikhonov solution minimizing
        ``||A x - b||^2 + lambda^2 ||x||^2``; ``x_k``: the truncated-SVD
        solution of rank ``k``.
    relative error: ``||x_est - x|| / ||x||`` against the exact solution.
    error-optimal ``lambda``: the minimizer of the relative error over a
        601-point log grid on ``[1e-5, 1]``, which needs the truth and is
        only a yardstick.
    posterior: the Gaussian posterior of ``x`` under the prior
        ``N(0, tau^2 I)`` and the noise ``N(0, s^2 I)``, ``lambda = s / tau``.

Assumptions:
    - The noise is white, Gaussian, and of known standard deviation ``s``;
      the discrepancy principle is given the realized norm ``delta``.
    - The operator is exact (no model error); the only error source is the
      data noise, so the exact solution is a valid reference.
    - The Bayesian reading takes the zero-mean isotropic prior that makes
      the posterior mean equal to the Tikhonov solution; its band is only as
      good as that prior.
    - The nonlinear example fits ``y = a exp(-k t)`` with a penalty centered
      at ``(1.5, 0.5)``; Levenberg-Marquardt returns a local minimizer.

Equations:
    EQ-INV-REG-PROB-001: discrete backward heat operator with exact SVD.
    EQ-LINALG-SVD-001: one-sided Jacobi SVD of ``A`` reused by every method.
    EQ-INV-REG-007: discrete Picard coefficients ``|u_i'b|`` and their ratios.
    EQ-INV-REG-006: truncated-SVD solution.
    EQ-INV-REG-001: Tikhonov functional and its stacked least-squares form.
    EQ-INV-REG-002: Tikhonov filter factors.
    EQ-INV-REG-003: generalized cross-validation function.
    EQ-INV-REG-004: L-curve curvature.
    EQ-INV-REG-005: discrepancy principle (also used for the nonlinear fit).
    EQ-OPT-LM-001: Levenberg-Marquardt step of the nonlinear Tikhonov fit.

Diagnostics:
    - The Picard section prints where ``|u_i'b|`` stops decaying and meets
      the noise level ``s``: beyond that index regularization must filter.
    - Every parameter choice is printed with its ``lambda`` and relative
      error next to the error-optimal value; ``at_boundary`` is printed for
      the curve-based rules.
    - The posterior mean is compared with the Tikhonov solution (maximum
      absolute difference, which should be rounding); the 95 % band reports
      the fraction of grid points whose exact value it contains.
    - The nonlinear fit prints its convergence flag and its residual norm
      against the discrepancy target.

Cost and error:
    - One ``32 x 32`` Jacobi SVD (about 0.2 s) is shared by all linear
      methods; each Tikhonov solve is one Householder QR of the stacked
      ``64 x 32`` matrix (about 0.03 s), so the 601-point error scan and the
      301 solves of the L-curve figure take most of the run time.
    - All numbers are for one noise realization (``SEED``); across seeds the
      ratio of chosen to error-optimal ``lambda`` varies, see
      ``docs/foundations/regularization-and-ill-posed-problems.md``.
    - The posterior band coverage is a pointwise fraction over 32 correlated
      grid points of one realization, not a coverage probability.

References:
    REF-HANSEN-2008: discrete ill-posed problems, filter factors, Picard plot.
    REF-TIKHONOV-1963: Tikhonov regularization.
    REF-MOROZOV-1966: the discrepancy principle.
    REF-CRAVEN-WAHBA-1979: generalized cross-validation.
    REF-HANSEN-OLEARY-1993: the L-curve and its curvature.
    REF-TARANTOLA-2005: the Gaussian (Bayesian) reading of Tikhonov.
    REF-HESTENES-1958: one-sided Jacobi SVD.
    REF-MORE-1978: Levenberg-Marquardt.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.foundations import RandomStream
from probstat.inverse import (
    DiscreteIllPosedProblem,
    RegularizationParameterChoice,
    backward_heat_problem,
    discrepancy_principle,
    generalized_cross_validation,
    l_curve,
    nonlinear_tikhonov,
    picard_coefficients,
    tikhonov,
    truncated_svd,
)
from probstat.linalg import SingularValueDecomposition, singular_value_decomposition

# Root seed of the data noise (child stream "noise") and of the nonlinear
# example (child stream "decay").
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
SIZE = 32
DIFFUSION_TIME = 0.005
NOISE_FRACTION = 0.01
GRID = tuple(10.0 ** (-5.0 + 5.0 * index / 600) for index in range(601))
Z95 = 1.959963984540054


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(value * value for value in vector))


@dataclass(frozen=True)
class NoisyProblem:
    """The heat problem, its SVD, and one seeded noisy data vector.

    References:
        [REF-HANSEN-2008, Sections 2.1 and 2.3].

    """

    problem: DiscreteIllPosedProblem
    svd: SingularValueDecomposition
    data: tuple[float, ...]
    noise_sd: float
    noise_norm: float

    def relative_error(self, estimate: Sequence[float]) -> float:
        """Return ``||estimate - x|| / ||x||`` against the exact solution.

        References:
            [REF-HANSEN-2008, Section 2.5]. No equation is evaluated.

        """
        exact = self.problem.exact_solution
        return norm([a - b for a, b in zip(estimate, exact, strict=True)]) / norm(exact)

    def tikhonov_error(self, lam: float) -> float:
        """Return the relative error of the Tikhonov solution at ``lam``.

        References:
            [EQ-INV-REG-001], [REF-TIKHONOV-1963].

        """
        solution = tikhonov(
            self.problem.operator, self.data, lam, decomposition=self.svd
        ).solution
        return self.relative_error(solution)


def noisy_problem() -> NoisyProblem:
    """Build the heat problem and add 1 % white noise from ``SEED``.

    References:
        [EQ-INV-REG-PROB-001], [EQ-LINALG-SVD-001], [REF-HANSEN-2008].

    """
    problem = backward_heat_problem(SIZE, time=DIFFUSION_TIME)
    svd = singular_value_decomposition(problem.operator)
    exact_data = problem.exact_data
    scale = NOISE_FRACTION * norm(exact_data) / math.sqrt(len(exact_data))
    generator = RandomStream(SEED).child("noise").generator()
    noise = [generator.gauss(0.0, scale) for _ in exact_data]
    data = tuple(b + e for b, e in zip(exact_data, noise, strict=True))
    return NoisyProblem(problem, svd, data, scale, norm(noise))


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def picard_section(case: NoisyProblem) -> int:
    """Print the Picard coefficients and the index where noise takes over.

    Returns:
        The (1-based) index at which ``|u_i'b| / sigma_i`` is smallest.

    References:
        [EQ-INV-REG-007], [REF-HANSEN-2008, Section 2.4].

    """
    picard = picard_coefficients(
        case.problem.operator, case.data, decomposition=case.svd
    )
    print("1. Picard coefficients (backward heat, n = 32, 1 % noise)")
    print(f"   condition number {case.svd.condition_number:.3g}")
    print(
        f"   noise sd s = {case.noise_sd:.4f}, noise norm delta = {case.noise_norm:.4f}"
    )
    print("   i   sigma_i      |u_i'b|      |u_i'b|/sigma_i")
    for i in (1, 4, 7, 10, 11, 12, 13, 16, 32):
        print(
            f"   {i:2d}  {picard.singular_values[i - 1]:.3e}  "
            f"{picard.data_coefficients[i - 1]:.3e}  "
            f"{picard.solution_coefficients[i - 1]:.3e}"
        )
    ratios = picard.solution_coefficients
    turn = min(range(len(ratios)), key=lambda i: ratios[i]) + 1
    tail = picard.data_coefficients[turn:]
    tail_mean = math.fsum(tail) / len(tail)
    print(
        f"   |u_i'b| / sigma_i is smallest at index {turn} and grows after it; "
        f"beyond it the mean |u_i'b| is {tail_mean:.4f}, against "
        f"s sqrt(2/pi) = {case.noise_sd * math.sqrt(2.0 / math.pi):.4f} for "
        "pure noise\n"
    )
    return turn


def filtering_section(case: NoisyProblem) -> dict[str, Any]:
    """Compare the best truncated SVD and the best Tikhonov solution.

    References:
        [EQ-INV-REG-001], [EQ-INV-REG-002], [EQ-INV-REG-006],
        [REF-HANSEN-2008, Sections 2.4 and 2.7].

    """
    operator, data, svd = case.problem.operator, case.data, case.svd
    naive = tikhonov(operator, data, 0.0, decomposition=svd)
    tsvd_errors = [
        case.relative_error(
            truncated_svd(operator, data, k, decomposition=svd).solution
        )
        for k in range(1, SIZE + 1)
    ]
    best_rank = min(range(SIZE), key=lambda i: tsvd_errors[i]) + 1
    errors = [case.tikhonov_error(lam) for lam in GRID]
    best = min(range(len(GRID)), key=lambda i: errors[i])
    optimal = tikhonov(operator, data, GRID[best], decomposition=svd)
    factors = optimal.filter_factors or ()
    print("2. Filtering: truncated SVD against Tikhonov")
    print(
        "   lambda = 0 (no regularization): relative error "
        f"{case.relative_error(naive.solution):.3e}"
    )
    print(
        f"   best TSVD rank k = {best_rank}: relative error "
        f"{tsvd_errors[best_rank - 1]:.4f}"
    )
    print(
        f"   best Tikhonov lambda = {GRID[best]:.3g} (601-point grid): relative "
        f"error {errors[best]:.4f}"
    )
    print(
        "   Tikhonov filter factors f_4..f_9 at that lambda: "
        + ", ".join(f"{f:.3f}" for f in factors[3:9])
        + "\n"
    )
    return {
        "best_rank": best_rank,
        "best_lambda": GRID[best],
        "best_error": errors[best],
        "errors": errors,
        "optimal": optimal,
    }


def parameter_section(
    case: NoisyProblem, best_lambda: float, best_error: float
) -> dict[str, RegularizationParameterChoice]:
    """Choose lambda by the discrepancy principle, GCV, and the L-curve.

    References:
        [EQ-INV-REG-003], [EQ-INV-REG-004], [EQ-INV-REG-005],
        [REF-MOROZOV-1966], [REF-CRAVEN-WAHBA-1979], [REF-HANSEN-OLEARY-1993].

    """
    operator, data, svd = case.problem.operator, case.data, case.svd
    choices = {
        "discrepancy": discrepancy_principle(
            operator, data, case.noise_norm, decomposition=svd
        ),
        "GCV": generalized_cross_validation(operator, data, decomposition=svd),
        "L-curve": l_curve(operator, data, decomposition=svd),
    }
    print("3. Choosing lambda")
    print(f"   error-optimal  lambda={best_lambda:.3g}  error={best_error:.4f}")
    for name, choice in choices.items():
        error = case.relative_error(choice.solution.solution)
        print(
            f"   {name:13s}  lambda={choice.regularization:.3g}  error={error:.4f}  "
            f"lambda/optimal={choice.regularization / best_lambda:.2f}  "
            f"at_boundary={choice.at_boundary}"
        )
    print()
    return choices


def posterior_sd(case: NoisyProblem, lam: float) -> list[float]:
    """Return pointwise posterior standard deviations for prior sd ``s / lam``.

    The posterior covariance is ``s^2 (A'A + lam^2 I)^-1
    = s^2 V diag(1 / (sigma_i^2 + lam^2)) V'``.

    References:
        [EQ-LINALG-SVD-001], [REF-TARANTOLA-2005, Chapter 3].

    """
    svd = case.svd
    s2 = case.noise_sd**2
    weights = [1.0 / (sigma * sigma + lam * lam) for sigma in svd.singular_values]
    return [
        math.sqrt(
            s2 * math.fsum(row[i] * row[i] * weights[i] for i in range(len(weights)))
        )
        for row in svd.v
    ]


def log_evidence(case: NoisyProblem, lam: float) -> float:
    """Return ``log p(b | lambda)`` for ``b ~ N(0, s^2 I + (s/lam)^2 A A')``.

    In the left singular basis the covariance is diagonal with entries
    ``s^2 (1 + sigma_i^2 / lam^2)``.

    References:
        [EQ-LINALG-SVD-001], [REF-TARANTOLA-2005, Chapter 3].

    """
    s2 = case.noise_sd**2
    svd = case.svd
    beta = [
        math.fsum(svd.u[r][j] * case.data[r] for r in range(len(case.data)))
        for j in range(len(svd.singular_values))
    ]
    terms = []
    for sigma, coefficient in zip(svd.singular_values, beta, strict=True):
        variance = s2 * (1.0 + sigma * sigma / (lam * lam))
        terms.append(
            coefficient * coefficient / variance + math.log(2.0 * math.pi * variance)
        )
    return -0.5 * math.fsum(terms)


def bayesian_section(
    case: NoisyProblem, lam: float, best_lambda: float
) -> dict[str, Any]:
    """Read Tikhonov as a posterior mean and add the posterior band.

    References:
        [EQ-INV-REG-001], [REF-TARANTOLA-2005, Chapter 3],
        [REF-HANSEN-2008, Section 2.9].

    """
    operator, data, svd = case.problem.operator, case.data, case.svd
    solution = tikhonov(operator, data, lam, decomposition=svd).solution
    # Posterior mean V diag(sigma / (sigma^2 + lam^2)) U'b, independent of QR.
    beta = [
        math.fsum(svd.u[r][j] * data[r] for r in range(len(data)))
        for j in range(len(svd.singular_values))
    ]
    coefficients = [
        sigma * b / (sigma * sigma + lam * lam)
        for sigma, b in zip(svd.singular_values, beta, strict=True)
    ]
    mean = [
        math.fsum(row[i] * coefficients[i] for i in range(len(coefficients)))
        for row in svd.v
    ]
    difference = max(abs(a - b) for a, b in zip(mean, solution, strict=True))
    sds = posterior_sd(case, lam)
    exact = case.problem.exact_solution
    inside = sum(
        abs(m - x) <= Z95 * sd for m, x, sd in zip(mean, exact, sds, strict=True)
    )
    # Empirical Bayes: maximize the marginal likelihood over the scan grid.
    evidences = [log_evidence(case, value) for value in GRID]
    top = max(range(len(GRID)), key=lambda i: evidences[i])
    marginal = GRID[top]
    print("4. Bayesian reading (prior N(0, tau^2 I), lambda = s / tau)")
    print(
        f"   at the discrepancy lambda = {lam:.3g}: prior sd tau = "
        f"{case.noise_sd / lam:.4f}"
    )
    print(f"   max |posterior mean - Tikhonov solution| = {difference:.1e}")
    print(
        f"   posterior sd ranges {min(sds):.4f} to {max(sds):.4f}; the 95 % band "
        f"holds the exact value at {inside} of {len(exact)} grid points"
    )
    print(
        f"   maximum marginal likelihood lambda = {marginal:.3g} "
        f"(lambda/optimal={marginal / best_lambda:.2f}), relative error "
        f"{case.tikhonov_error(marginal):.4f}\n"
    )
    return {"mean": mean, "sd": sds, "inside": inside, "marginal": marginal}


def nonlinear_section() -> None:
    """Fit an exponential decay by nonlinear Tikhonov with Morozov's lambda.

    References:
        [EQ-INV-REG-001], [EQ-INV-REG-005], [EQ-OPT-LM-001],
        [REF-MOROZOV-1966], [REF-MORE-1978].

    """
    times = tuple(0.25 * i for i in range(12))
    noise_sd = 0.02
    generator = RandomStream(SEED).child("decay").generator()
    observed = tuple(
        2.0 * math.exp(-0.7 * t) + generator.gauss(0.0, noise_sd) for t in times
    )

    def residual(x: Sequence[float]) -> tuple[float, ...]:
        """Return model minus data for ``y = a exp(-k t)``.

        References:
            [REF-HANSEN-2008, Section 2.2]. No equation is evaluated.

        """
        return tuple(
            x[0] * math.exp(-x[1] * t) - y for t, y in zip(times, observed, strict=True)
        )

    prior = (1.5, 0.5)
    target = noise_sd * math.sqrt(len(times))

    def fit(lam: float) -> Any:
        """Return the nonlinear Tikhonov fit at ``lam``.

        References:
            [EQ-INV-REG-001], [REF-MORE-1978].

        """
        return nonlinear_tikhonov(residual, (1.0, 1.0), lam, prior_mean=prior)

    print("5. Nonlinear Tikhonov: y = a exp(-k t), truth (2.0, 0.7), prior (1.5, 0.5)")
    for lam in (0.0, 0.1, 1.0, 10.0):
        result = fit(lam)
        print(
            f"   lambda={lam:<5g} (a, k)=({result.solution[0]:.3f}, "
            f"{result.solution[1]:.3f})  ||r||={result.residual_norm:.4f}  "
            f"converged={result.least_squares.converged}"
        )
    # Discrepancy principle by bisection in log lambda: ||r(lambda)|| = s sqrt(m).
    low, high = math.log(1e-4), math.log(10.0)
    for _ in range(50):
        middle = 0.5 * (low + high)
        if fit(math.exp(middle)).residual_norm < target:
            low = middle
        else:
            high = middle
    chosen = fit(math.exp(0.5 * (low + high)))
    print(
        f"   discrepancy target s sqrt(m) = {target:.4f}: lambda = "
        f"{chosen.regularization:.3g}, (a, k)=({chosen.solution[0]:.3f}, "
        f"{chosen.solution[1]:.3f}), ||r|| = {chosen.residual_norm:.4f}\n"
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


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


def reconstruction_figure(
    case: NoisyProblem,
    choices: dict[str, RegularizationParameterChoice],
    posterior: dict[str, Any],
) -> str:
    """Draw the exact solution, the three reconstructions, and the band.

    References:
        [REF-MATPLOTLIB], [REF-TARANTOLA-2005]. No equation is evaluated.

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.4))
    grid = list(case.problem.grid)
    exact = list(case.problem.exact_solution)
    mean, sds = posterior["mean"], posterior["sd"]
    theme.band(
        axes,
        grid,
        [m - Z95 * s for m, s in zip(mean, sds, strict=True)],
        [m + Z95 * s for m, s in zip(mean, sds, strict=True)],
        color=theme.role_colour("model"),
    )
    # Line pattern as well as hue tells the three choices apart.
    styles = (
        ("L-curve", "series-3", theme.PATH_WIDTH * 2.0, None),
        ("GCV", "series-4", theme.CONTEXT_WIDTH, theme.SIMULATED),
        ("discrepancy", "model", theme.LINE_WIDTH, None),
    )
    labels = []
    for name, role, width, dashes in styles:
        solution = choices[name].solution.solution
        colour = theme.role_colour(role)
        theme.line(axes, grid, solution, color=colour, width=width, dashes=dashes)
        error = case.relative_error(solution)
        labels.append((grid[-1], solution[-1], f"{name} ({error:.3f})", colour))
    style = theme.entity_style("truth")
    theme.line(
        axes, grid, exact, color=style.colour(), dashes=style.dashes, width=style.width
    )
    labels.append((grid[-1], exact[-1], "exact", style.colour()))
    axes.set_xlim(0.0, 1.0)
    low = min(m - Z95 * s for m, s in zip(mean, sds, strict=True))
    high = max(m + Z95 * s for m, s in zip(mean, sds, strict=True))
    axes.set_ylim(low - 0.05, high + 0.05)
    theme.direct_labels(axes, labels, gap_points=13.0)
    theme.axis_labels(axes, "Position s", "Initial temperature x(s)")
    disc = case.relative_error(choices["discrepancy"].solution.solution)
    theme.title_block(
        axes,
        f"The discrepancy choice recovers the initial state to {disc:.1%}; "
        "GCV and the L-curve leave noise in",
        f"Labels: method (relative error). Band: 95% posterior interval at the "
        f"discrepancy lambda, holding the exact value at {posterior['inside']} "
        f"of {len(exact)} points",
    )
    return save(axes.figure, "regularization-reconstructions.png")


def _plain_log(axis: Any, ticks: Sequence[float]) -> None:
    """Put plain-number ticks on a logarithmic axis.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FixedLocator, NullLocator

    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_ticklabels([f"{value:g}" for value in ticks])


def l_curve_figure(
    case: NoisyProblem,
    choices: dict[str, RegularizationParameterChoice],
    best_lambda: float,
) -> str:
    """Draw the corner of the L-curve with the three choices and the optimum.

    References:
        [EQ-INV-REG-004], [REF-HANSEN-OLEARY-1993], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    operator, data, svd = case.problem.operator, case.data, case.svd
    grid = [10.0 ** (-4.0 + 3.7 * i / 300) for i in range(301)]
    solutions = [tikhonov(operator, data, lam, decomposition=svd) for lam in grid]
    axes = theme.new_axes(figsize=(7.2, 4.4))
    palette = theme.active_palette()
    theme.line(
        axes,
        [s.residual_norm for s in solutions],
        [s.solution_norm for s in solutions],
        color=palette.muted,
        width=theme.CONTEXT_WIDTH,
    )
    optimal = tikhonov(operator, data, best_lambda, decomposition=svd)
    marks = [
        (
            "L-curve corner",
            choices["L-curve"].solution,
            theme.role_colour("series-3"),
            "D",
        ),
        ("GCV", choices["GCV"].solution, theme.role_colour("series-4"), "^"),
        (
            "discrepancy",
            choices["discrepancy"].solution,
            theme.role_colour("model"),
            "o",
        ),
        ("error-optimal", optimal, theme.entity_colour("truth"), "s"),
    ]
    # Labels stacked in a column above the corner, each with a leader line.
    label_heights = (60.0, 34.0, 19.0, 10.5)
    for (name, solution, colour, marker), height in zip(
        marks, label_heights, strict=True
    ):
        point = (solution.residual_norm, solution.solution_norm)
        theme.points(
            axes, [point[0]], [point[1]], color=colour, marker=marker, size=60.0,
            zorder=5.0,
        )  # fmt: skip
        axes.annotate(
            f"{name}: lambda = {solution.regularization:.2g}, "
            f"error {case.relative_error(solution.solution):.3f}",
            point,
            xytext=(1.08 * case.noise_norm, height),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=theme.ANNOTATION_SIZE,
            color=palette.ink,
            arrowprops={
                "arrowstyle": "-",
                "color": palette.muted,
                "linewidth": 0.7,
                "shrinkA": 2.0,
                "shrinkB": 5.0,
                "relpos": (0.0, 0.5),
            },
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(0.025, 0.25)
    axes.set_ylim(2.5, 120.0)
    _plain_log(axes.xaxis, (0.03, 0.05, 0.1, 0.2))
    _plain_log(axes.yaxis, (3.0, 5.0, 10.0, 20.0, 50.0, 100.0))
    theme.reference_line(axes, x=case.noise_norm, color=palette.muted)
    axes.annotate(
        f"noise norm {case.noise_norm:.4f}",
        (case.noise_norm, 2.5),
        xytext=(4.0, 6.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.muted,
    )
    theme.axis_labels(axes, "Residual norm ||A x - b||", "Solution norm ||x||")
    theme.title_block(
        axes,
        "The L-curve corner sits left of the noise norm; the discrepancy "
        "principle sits on it",
        "Grey: Tikhonov solutions for lambda from 1e-4 to 0.5 (corner region) "
        "· backward heat problem, n = 32, 1% noise",
    )
    return save(axes.figure, "regularization-l-curve.png")


def main() -> None:
    """Run every section and write the figures.

    References:
        [REF-HANSEN-2008], [REF-TARANTOLA-2005].

    """
    case = noisy_problem()
    picard_section(case)
    filtering = filtering_section(case)
    choices = parameter_section(case, filtering["best_lambda"], filtering["best_error"])
    posterior = bayesian_section(
        case, choices["discrepancy"].regularization, filtering["best_lambda"]
    )
    nonlinear_section()
    if not has_matplotlib():
        print("Matplotlib is not installed; figures skipped")
        return
    written = (
        reconstruction_figure(case, choices, posterior),
        l_curve_figure(case, choices, filtering["best_lambda"]),
    )
    print("figures: " + ", ".join(written))


if __name__ == "__main__":
    main()
