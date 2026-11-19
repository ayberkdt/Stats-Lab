"""Field inversion laboratory: a log-permeability field from pressure data.

Run with ``PYTHONPATH=src python examples/field_inversion_lab.py``. A 1-D
steady Darcy problem is solved by a finite-volume scheme and a tridiagonal
(Thomas) solve written below; its log-permeability is a Gaussian random field
entered through a truncated Karhunen-Loeve expansion. Seven noisy pressure
readings are inverted four ways: preconditioned Crank-Nicolson (pCN),
adaptive Metropolis, ES-MDA, and ensemble Kalman inversion (EKI) with
discrepancy stopping. Then the inverse crime is measured, and the pCN and
random-walk acceptance rates are compared as the truncation grows. With
Matplotlib two figures are written to ``examples/generated/``. Every random
step is seeded; the run takes about a minute and a half in pure Python.

Definitions:
    Darcy problem: ``-(k(x) u'(x))' = 1`` on ``[0, 1]`` with ``u(0) = u(1) =
        0``; ``u`` is the pressure and ``k = exp(g)`` the permeability.
    Discretization: ``n`` equal cells with ``k`` constant per cell; at node
        ``i`` the flux balance ``(k_(i-1/2) (u_i - u_(i-1)) - k_(i+1/2)
        (u_(i+1) - u_i)) / h^2 = 1`` gives a tridiagonal system, solved by
        the Thomas algorithm. The inversion grid has 64 cells.
    Prior: ``g`` is a zero-mean Gaussian field with Matern-3/2 covariance
        (standard deviation 1, length scale 0.2); ``g = sum_k sqrt(lambda_k)
        xi_k phi_k`` with ``M = 12`` Nystrom modes on the 64 cell midpoints
        and ``xi_k ~ N(0, 1)`` (``RandomFieldInput`` with ``exp``).
    Truth: one exact draw of the same Gaussian field on a 256-cell grid
        (Cholesky factor of the full covariance, no truncation); the data are
        its pressures at ``x = 1/8, ..., 7/8`` plus white noise of standard
        deviation 5 % of the largest of them.
    Coarse truth: the mean of the fine log-permeability over each coarse
        cell; RMSE: root mean square over the 64 cells of an estimate minus
        the coarse truth; coverage: fraction of cells whose coarse truth lies
        in the pointwise 90 % band (5 % and 95 % quantiles).

Assumptions:
    - The noise is Gaussian with known standard deviation; every method uses
      the same likelihood.
    - The prior law (kernel, variance, length scale) is the law that drew
      the truth; only the discretization and the truncation differ. A wrong
      kernel would add a model error that no section below measures.
    - The 64-cell solver is accurate enough: the discretization error at the
      observation points is printed against the noise level.
    - MCMC bands are trusted only as far as their R-hat and ESS allow; the
      ES-MDA ensemble is a linearized approximation, and the EKI ensemble is
      not a posterior sample.

Equations:
    EQ-PROC-KER-003: Matern-3/2 covariance of the log-permeability.
    EQ-PROC-KL-001: truncated Karhunen-Loeve expansion of the prior.
    EQ-PROC-KL-002: Nystrom discretization of the KL eigenproblem.
    EQ-PROC-QUAD-001: midpoint rule behind the Nystrom nodes and weights.
    EQ-MOD-RF-001: map from the KL coefficients to the permeability field.
    EQ-LINALG-001: Cholesky factor that draws the untruncated truth.
    EQ-CAL-LIK-001: Gaussian likelihood of the seven pressure readings.
    EQ-CAL-STD-001: posterior in the prior's standard-normal space.
    EQ-MCMC-PCN-001: pCN proposal and acceptance.
    EQ-MCMC-PCN-002: warm-up adaptation of the pCN step.
    EQ-MCMC-AM-001: adaptive Metropolis proposal covariance.
    EQ-MCMC-MH-001: random-walk Metropolis acceptance (truncation study).
    EQ-MCMC-ESS-002: bulk effective sample size of each coefficient.
    EQ-MCMC-RHAT-002: rank-normalized R-hat of each coefficient.
    EQ-INV-ESMDA-001: ES-MDA update with inflated noise.
    EQ-INV-EKI-001: ensemble Kalman inversion update.
    EQ-INV-EKI-002: discrepancy-principle stopping of EKI.

Diagnostics:
    - The discretization error of the 64-cell solver at the observation
      points, in noise standard deviations, and the best RMSE any field in
      the 12-mode span can reach (projection of the coarse truth).
    - pCN and adaptive Metropolis print acceptance, the smallest bulk ESS,
      and the largest R-hat over the 12 coefficients; pCN prints its frozen
      steps ``beta``.
    - Every method prints its forward-model runs, the RMSE of its mean field,
      the coverage of its 90 % band, and the mean band width.
    - The inverse-crime section prints the ES-MDA RMSE with data from the
      inversion's own model against the honest data. With ``SEED`` the
      64-cell solver is off by at most 0.013 noise standard deviations and
      the 12 modes keep 0.9942 of the variance, so the crime moves the RMSE
      only from 0.377 to 0.362: here the honest set-up costs little, and the
      crime would matter with a coarser solver or fewer modes, where it hides
      a model error the noise model does not carry.
    - With ``SEED`` pCN mixes worse than adaptive Metropolis (smallest bulk
      ESS 14 against 89, from more forward runs): seven precise readings make
      the posterior much narrower than the prior in the leading modes, the
      regime in which the isotropic pCN move is slow. Its advantage is the
      truncation section: fixed-step acceptance rates of pCN and random-walk
      Metropolis for ``M = 12, 32, 64``.

Cost and error:
    - One forward run is one Thomas solve on 64 cells plus the KL sum; the
      counts include the Laplace start of both MCMC runs.
    - pCN: 4 chains x (1000 warm-up + 4000 draws); adaptive Metropolis: 4 x
      (1000 + 3000); ES-MDA: 200 members x 5 evaluations; EKI: 50 members,
      at most 20 updates. MCMC bands carry Monte Carlo error that the ESS
      quantifies; one data set and one truth (``SEED``) are shown.
    - Coverage is a fraction over 64 strongly correlated cells of one truth,
      not a coverage probability.
    - Truncation study: one chain of 300 warm-up and 1200 draws per sampler
      and ``M``, from the prior mean; acceptance rates carry the error of one
      short run.

References:
    REF-COTTER-ROBERTS-STUART-WHITE-2013: pCN for functions.
    REF-HAIRER-STUART-VOLLMER-2014: dimension-independent pCN spectral gap.
    REF-HAARIO-SAKSMAN-TAMMINEN-2001: adaptive Metropolis.
    REF-EMERICK-REYNOLDS-2013: ES-MDA.
    REF-IGLESIAS-LAW-STUART-2013: ensemble Kalman inversion.
    REF-IGLESIAS-2016: discrepancy stopping of EKI.
    REF-KAIPIO-SOMERSALO-2007: discretization, truncation, and the inverse
        crime.
    REF-GHANEM-SPANOS-1991: Karhunen-Loeve representation of random fields.
    REF-RASMUSSEN-WILLIAMS-2006: Matern kernels and the Nystrom method.
    REF-GOLUB-VANLOAN-2013: tridiagonal (Thomas) elimination.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import statistics
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.foundations import RandomStream
from probstat.inference import (
    ConvergenceWarning,
    bulk_effective_sample_size,
    preconditioned_crank_nicolson,
    random_walk_metropolis,
    rank_normalized_rhat,
)
from probstat.inverse import (
    CalibrationProblem,
    Experiment,
    NoiseModel,
    ensemble_kalman_inversion,
    ensemble_smoother_mda,
)
from probstat.linalg import cholesky, matvec
from probstat.modeling import ProbabilisticModel, RandomFieldInput
from probstat.processes import (
    CovarianceKernel,
    KarhunenLoeveExpansion,
    covariance_matrix,
    karhunen_loeve_interval,
)
from probstat.statistics import quantile
from probstat.uq import QuantityOfInterest

# Root seed: child streams "truth" (field), "noise" (data), and the samplers
# and ensembles use SEED itself or SEED + 1 (truncation study).
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
KERNEL = CovarianceKernel("matern32", variance=1.0, length_scale=0.2)
COARSE, FINE = 64, 256
TERMS = 12
STUDY_TERMS = (12, 32, 64)
OBSERVED = tuple(j / 8 for j in range(1, 8))
NOISE_FRACTION = 0.05


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


# --------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------


def solve_pressure(permeability: Sequence[float]) -> list[float]:
    """Solve ``-(k u')' = 1``, ``u(0) = u(1) = 0`` on equal cells.

    ``permeability[i]`` is ``k`` on cell ``i``; the unknowns are the ``n - 1``
    interior nodal pressures of the flux-balance (finite-volume) scheme,
    found by the Thomas algorithm (forward elimination, back substitution).

    Returns:
        Nodal pressures ``u_0..u_n`` including the two boundary zeros.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.3]. No registered equation is
        evaluated.

    """
    n = len(permeability)
    load = 1.0 / (n * n)
    size = n - 1
    upper = [0.0] * size
    rhs = [0.0] * size
    for i in range(size):
        west, east = permeability[i], permeability[i + 1]
        diagonal = west + east
        if i > 0:
            diagonal += west * upper[i - 1]
            rhs[i] = (load + west * rhs[i - 1]) / diagonal
        else:
            rhs[i] = load / diagonal
        upper[i] = -east / diagonal
    pressure = [0.0] * size
    pressure[-1] = rhs[-1]
    for i in range(size - 2, -1, -1):
        pressure[i] = rhs[i] - upper[i] * pressure[i + 1]
    return [0.0, *pressure, 0.0]


def observe(pressure: Sequence[float]) -> tuple[float, ...]:
    """Return the nodal pressures at the observation points ``j / 8``.

    References:
        [REF-KAIPIO-SOMERSALO-2007]. No equation is evaluated.

    """
    n = len(pressure) - 1
    return tuple(pressure[round(x * n)] for x in OBSERVED)


def rms(values: Sequence[float]) -> float:
    """Return the root mean square.

    References:
        [REF-KAIPIO-SOMERSALO-2007]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


# --------------------------------------------------------------------------
# Truth, data, and the inverse problem
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """The fine truth, its coarse view, and the noisy data.

    References:
        [REF-KAIPIO-SOMERSALO-2007].

    """

    fine: tuple[float, ...]
    coarse: tuple[float, ...]
    clean: tuple[float, ...]
    data: tuple[float, ...]
    noise_sd: float
    discretization: tuple[float, ...]


def draw_truth() -> Truth:
    """Draw the untruncated field on 256 cells and simulate the data.

    References:
        [EQ-PROC-KER-003], [EQ-LINALG-001], [REF-KAIPIO-SOMERSALO-2007].

    """
    centres = [(i + 0.5) / FINE for i in range(FINE)]
    lower = cholesky(covariance_matrix(KERNEL, centres))
    generator = RandomStream(SEED).child("truth").generator()
    fine = matvec(lower, [generator.gauss(0.0, 1.0) for _ in range(FINE)])
    clean = observe(solve_pressure([math.exp(v) for v in fine]))
    noise_sd = NOISE_FRACTION * max(abs(v) for v in clean)
    noise = RandomStream(SEED).child("noise").generator()
    data = tuple(v + noise.gauss(0.0, noise_sd) for v in clean)
    ratio = FINE // COARSE
    coarse = tuple(
        math.fsum(fine[ratio * i : ratio * (i + 1)]) / ratio for i in range(COARSE)
    )
    # Discretization error alone: the coarse solver run on the exact
    # series (harmonic-mean) permeability of each coarse cell.
    harmonic = [
        ratio / math.fsum(math.exp(-v) for v in fine[ratio * i : ratio * (i + 1)])
        for i in range(COARSE)
    ]
    coarse_clean = observe(solve_pressure(harmonic))
    discretization = tuple(
        (a - b) / noise_sd for a, b in zip(coarse_clean, clean, strict=True)
    )
    return Truth(tuple(fine), coarse, clean, data, noise_sd, discretization)


def truncate(expansion: KarhunenLoeveExpansion, terms: int) -> KarhunenLoeveExpansion:
    """Return the leading ``terms`` modes of a computed expansion.

    References:
        [EQ-PROC-KL-001], [REF-GHANEM-SPANOS-1991, Chapter 2].

    """
    return KarhunenLoeveExpansion(
        expansion.kernel,
        expansion.points,
        expansion.weights,
        expansion.mean,
        expansion.eigenvalues[:terms],
        expansion.eigenvectors[:terms],
        expansion.spectrum,
    )


def projection(
    expansion: KarhunenLoeveExpansion, field: Sequence[float]
) -> list[float]:
    """Return the weighted least-squares projection of a field on the modes.

    References:
        [EQ-PROC-KL-001], [EQ-PROC-QUAD-001].

    """
    coefficients = [
        math.fsum(
            w * g * p for w, g, p in zip(expansion.weights, field, mode, strict=True)
        )
        for mode in expansion.eigenvectors
    ]
    return [
        math.fsum(
            c * mode[i]
            for c, mode in zip(coefficients, expansion.eigenvectors, strict=True)
        )
        for i in range(len(field))
    ]


@dataclass
class Inversion:
    """A calibration problem on the coarse grid with a forward-run counter.

    References:
        [EQ-CAL-LIK-001], [EQ-MOD-RF-001].

    """

    field: RandomFieldInput
    problem: CalibrationProblem
    runs: list[int]

    def log_permeability(self, coefficients: Sequence[float]) -> tuple[float, ...]:
        """Return the Gaussian field ``g`` for KL coefficients.

        References:
            [EQ-PROC-KL-001].

        """
        return self.field.expansion.realize(coefficients)

    def counted(self, run: Callable[[], Any]) -> tuple[Any, int]:
        """Run a method and return its result and its forward-model runs.

        References:
            [REF-COTTER-ROBERTS-STUART-WHITE-2013]. No equation is evaluated.

        """
        before = self.runs[0]
        result = run()
        return result, self.runs[0] - before


def build_inversion(
    expansion: KarhunenLoeveExpansion, data: Sequence[float], noise_sd: float
) -> Inversion:
    """Declare the pressure-data calibration over the KL coefficients.

    References:
        [EQ-MOD-RF-001], [EQ-CAL-LIK-001], [REF-GHANEM-SPANOS-1991].

    """
    field = RandomFieldInput(expansion, "xi", math.exp, "log-permeability")
    runs = [0]
    names = tuple(f"p{j + 1}" for j in range(len(OBSERVED)))

    def forward(parameters: Mapping[str, float]) -> dict[str, float]:
        """Return the pressures at the observation points.

        References:
            [EQ-MOD-RF-001]. The solver evaluates no registered equation.

        """
        runs[0] += 1
        pressure = solve_pressure(field.field(parameters))
        return dict(zip(names, observe(pressure), strict=True))

    model = ProbabilisticModel(
        field.joint_input(),
        forward,
        tuple(
            QuantityOfInterest(name, f"pressure at x = {x}")
            for name, x in zip(names, OBSERVED, strict=True)
        ),
    )
    problem = CalibrationProblem(
        model,
        (Experiment(dict(zip(names, data, strict=True)), label="pressures"),),
        tuple(NoiseModel(name, standard_deviation=noise_sd) for name in names),
    )
    return Inversion(field, problem, runs)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSummary:
    """Pointwise mean and 90 % band of ``g`` from members or draws.

    References:
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """

    name: str
    mean: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    runs: int
    note: str

    def rmse(self, truth: Sequence[float]) -> float:
        """Return the RMSE of the mean field against a reference.

        References:
            [REF-KAIPIO-SOMERSALO-2007]. No equation is evaluated.

        """
        return rms([a - b for a, b in zip(self.mean, truth, strict=True)])

    def coverage(self, truth: Sequence[float]) -> float:
        """Return the fraction of cells whose reference lies in the band.

        References:
            [REF-KAIPIO-SOMERSALO-2007]. No equation is evaluated.

        """
        inside = sum(
            low <= value <= high
            for low, value, high in zip(self.lower, truth, self.upper, strict=True)
        )
        return inside / len(truth)

    def width(self) -> float:
        """Return the mean width of the 90 % band.

        References:
            [REF-KAIPIO-SOMERSALO-2007]. No equation is evaluated.

        """
        return statistics.fmean(
            b - a for a, b in zip(self.lower, self.upper, strict=True)
        )


def summarize(
    inversion: Inversion,
    members: Sequence[Sequence[float]],
    name: str,
    runs: int,
    note: str,
) -> FieldSummary:
    """Map coefficient vectors to fields and take the pointwise summaries.

    References:
        [EQ-PROC-KL-001], [EQ-MOD-RF-001].

    """
    fields = [inversion.log_permeability(member) for member in members]
    cells = range(len(fields[0]))
    columns = [[f[i] for f in fields] for i in cells]
    return FieldSummary(
        name,
        tuple(statistics.fmean(column) for column in columns),
        tuple(quantile(column, 0.05) for column in columns),
        tuple(quantile(column, 0.95) for column in columns),
        runs,
        note,
    )


def thinned(draws: Sequence[Sequence[float]], keep: int) -> list[Sequence[float]]:
    """Return about ``keep`` equally spaced draws.

    References:
        [REF-HAARIO-SAKSMAN-TAMMINEN-2001]. No equation is evaluated.

    """
    step = max(1, len(draws) // keep)
    return list(draws[::step])


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def setup_section(truth: Truth, full: KarhunenLoeveExpansion) -> KarhunenLoeveExpansion:
    """Print the data, the truncation, and the inverse-crime diagnostics.

    References:
        [EQ-PROC-KL-002], [EQ-PROC-KL-001], [REF-KAIPIO-SOMERSALO-2007].

    """
    expansion = truncate(full, TERMS)
    best = projection(expansion, truth.coarse)
    print("1. Darcy problem, prior, and data")
    print(
        "   pressures at x = 1/8..7/8: "
        + ", ".join(f"{v:.4f}" for v in truth.clean)
        + f"; noise sd {truth.noise_sd:.4f}"
    )
    print(
        f"   KL on {COARSE} midpoints: {TERMS} modes keep "
        f"{expansion.explained_variance_fraction:.4f} of the prior variance"
    )
    print(
        "   discretization error of the 64-cell solver at the data points: "
        f"at most {max(abs(v) for v in truth.discretization):.3f} noise sd"
    )
    floor = rms([a - b for a, b in zip(truth.coarse, best, strict=True)])
    print(
        f"   RMS of the coarse truth {rms(truth.coarse):.3f}; best RMSE in the "
        f"{TERMS}-mode span (projection) {floor:.3f}\n"
    )
    return expansion


def methods_section(inversion: Inversion, truth: Truth) -> list[FieldSummary]:
    """Invert the data with pCN, adaptive Metropolis, ES-MDA, and EKI.

    References:
        [EQ-MCMC-PCN-001], [EQ-MCMC-PCN-002], [EQ-MCMC-AM-001],
        [EQ-INV-ESMDA-001], [EQ-INV-EKI-001], [EQ-INV-EKI-002],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-EMERICK-REYNOLDS-2013],
        [REF-IGLESIAS-2016].

    """
    problem = inversion.problem
    summaries = []
    seconds: list[float] = []
    print("2. Four inversions of the same data (12 KL coefficients)")
    for method, draws in (("pcn", 4000), ("adaptive-metropolis", 3000)):
        with warnings.catch_warnings():
            # Short chains are part of the comparison; their ESS and R-hat
            # are printed instead of raised.
            warnings.simplefilter("ignore", ConvergenceWarning)
            began = time.perf_counter()
            posterior, runs = inversion.counted(
                lambda method=method, draws=draws: problem.sample_posterior(
                    method=method, draws=draws, warmup=1000, chains=4, seed=SEED
                )
            )
            seconds.append(time.perf_counter() - began)
        # Bulk ESS and rank-normalized R-hat only: the full summary adds
        # tail and quantile diagnostics that cost more than the chains.
        columns = [posterior.mcmc.chain_columns(name) for name in posterior.mcmc.names]
        ess = min(bulk_effective_sample_size(chains) for chains in columns)
        rhat = max(rank_normalized_rhat(chains) for chains in columns)
        label = "pCN" if method == "pcn" else "adaptive Metropolis"
        note = (
            f"acceptance {posterior.mcmc.acceptance_rate:.3f}, min bulk ESS "
            f"{ess:.0f}, max R-hat {rhat:.3f}"
        )
        print(f"   {label}: {note}")
        if method == "pcn":
            steps = [
                math.sqrt(chain.proposal_covariance[0][0])
                for chain in posterior.standard_space.chains
            ]
            print(
                "   pCN frozen beta per chain: " + ", ".join(f"{b:.3f}" for b in steps)
            )
        short = f"ESS {ess:.0f}, R-hat {rhat:.2f}"
        summaries.append(
            summarize(
                inversion, thinned(posterior.mcmc.pooled(), 2000), label, runs, short
            )
        )
    began = time.perf_counter()
    smoother, runs = inversion.counted(
        lambda: ensemble_smoother_mda(
            problem, ensemble_size=200, inflation_coefficients=(4.0,) * 4, seed=SEED
        )
    )
    seconds.append(time.perf_counter() - began)
    summaries.append(summarize(inversion, smoother.ensemble, "ES-MDA", runs, "N = 200"))
    began = time.perf_counter()
    eki, runs = inversion.counted(
        lambda: ensemble_kalman_inversion(
            problem, ensemble_size=50, iterations=20, seed=SEED, stopping="discrepancy"
        )
    )
    seconds.append(time.perf_counter() - began)
    note = (
        f"N = 50, stopped by {eki.stopped_by} after {eki.updates} updates "
        f"(misfit {eki.misfit_history[-1]:.2f} <= {eki.discrepancy_threshold:.2f})"
    )
    print(f"   EKI: {note}")
    summaries.append(
        summarize(inversion, eki.ensemble, "EKI", runs, f"{eki.updates} updates")
    )
    print(
        "   method               runs   seconds   RMSE of mean   90% coverage   "
        "band width"
    )
    for item, elapsed in zip(summaries, seconds, strict=True):
        print(
            f"   {item.name:19s} {item.runs:6d}   {elapsed:7.1f}   "
            f"{item.rmse(truth.coarse):12.3f}   {item.coverage(truth.coarse):12.3f}   "
            f"{item.width():10.3f}"
        )
    print(f"   prior (mean 0)      {0:6d}   {0.0:7.1f}   {rms(truth.coarse):12.3f}\n")
    return summaries


def inverse_crime_section(
    expansion: KarhunenLoeveExpansion, truth: Truth, honest: FieldSummary
) -> None:
    """Compare ES-MDA on honest data with data from the inversion's own model.

    The criminal data come from the 64-cell solver applied to the projection
    of the coarse truth on the 12 modes, with the same noise realization.

    References:
        [EQ-INV-ESMDA-001], [REF-KAIPIO-SOMERSALO-2007].

    """
    best = projection(expansion, truth.coarse)
    clean = observe(solve_pressure([math.exp(v) for v in best]))
    noise = [d - c for d, c in zip(truth.data, truth.clean, strict=True)]
    data = [c + e for c, e in zip(clean, noise, strict=True)]
    inversion = build_inversion(expansion, data, truth.noise_sd)
    smoother = ensemble_smoother_mda(
        inversion.problem,
        ensemble_size=200,
        inflation_coefficients=(4.0,) * 4,
        seed=SEED,
    )
    crime = summarize(inversion, smoother.ensemble, "ES-MDA", 0, "")
    print("3. The inverse crime (ES-MDA, same noise, same seed)")
    print(
        "   honest data (256-cell untruncated truth): RMSE "
        f"{honest.rmse(truth.coarse):.3f} against the coarse truth, coverage "
        f"{honest.coverage(truth.coarse):.3f}"
    )
    print(
        f"   criminal data (64-cell model of the 12-mode truth): RMSE "
        f"{crime.rmse(best):.3f} against its own truth, coverage "
        f"{crime.coverage(best):.3f}\n"
    )


def truncation_section(
    full: KarhunenLoeveExpansion, truth: Truth
) -> dict[str, tuple[float, ...]]:
    """Measure fixed-step pCN and random-walk acceptance as ``M`` grows.

    pCN uses ``beta = 0.2`` and the random walk the isotropic step 0.2 in
    the same standard-normal coordinates; neither adapts.

    References:
        [EQ-MCMC-PCN-001], [EQ-MCMC-MH-001],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-HAIRER-STUART-VOLLMER-2014].

    """
    pcn_rates, walk_rates = [], []
    print("4. Acceptance against the truncation (fixed steps 0.2, one chain each)")
    for terms in STUDY_TERMS:
        expansion = truncate(full, terms)
        inversion = build_inversion(expansion, truth.data, truth.noise_sd)
        problem = inversion.problem
        start = tuple(0.0 for _ in range(terms))
        pcn = preconditioned_crank_nicolson(
            problem.standard_log_likelihood, start, draws=1200, warmup=300, chains=1,
            step=0.2, adapt=False, seed=SEED + 1, check_convergence=False,
        )  # fmt: skip
        covariance = [
            [0.2**2 if i == j else 0.0 for j in range(terms)] for i in range(terms)
        ]
        walk = random_walk_metropolis(
            problem.standard_log_posterior, start, proposal_covariance=covariance,
            draws=1200, warmup=300, chains=1, seed=SEED + 1, check_convergence=False,
        )  # fmt: skip
        pcn_rates.append(pcn.acceptance_rate)
        walk_rates.append(walk.acceptance_rate)
        print(
            f"   M = {terms:2d} (variance kept "
            f"{expansion.explained_variance_fraction:.4f}): "
            f"pCN {pcn.acceptance_rate:.3f}, random walk {walk.acceptance_rate:.3f}"
        )
    print(
        f"   pCN acceptance stays between {min(pcn_rates):.3f} and "
        f"{max(pcn_rates):.3f} as M grows {STUDY_TERMS[0]} -> {STUDY_TERMS[-1]}; "
        f"the random walk falls by a factor {walk_rates[0] / walk_rates[-1]:.1f}\n"
    )
    return {"pcn": tuple(pcn_rates), "walk": tuple(walk_rates)}


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


def field_figure(truth: Truth, summaries: Sequence[FieldSummary]) -> str:
    """Draw each method's mean and 90 % band against the true field.

    References:
        [EQ-PROC-KL-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    plt = theme.pyplot()
    figure, grid = plt.subplots(
        2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.18, "wspace": 0.2},
    )  # fmt: skip
    palette = theme.active_palette()
    cells = [(i + 0.5) / COARSE for i in range(COARSE)]
    fine = [(i + 0.5) / FINE for i in range(FINE)]
    style = theme.entity_style("truth")
    model = theme.role_colour("model")
    low = min(min(s.lower) for s in summaries)
    high = max(max(s.upper) for s in summaries)
    bottom = low - 0.5
    for axes, summary in zip(grid.flat, summaries, strict=True):
        theme.style_axes(axes)
        theme.band(axes, cells, summary.lower, summary.upper, color=model)
        theme.line(axes, cells, summary.mean, color=model)
        theme.line(
            axes, fine, truth.fine, color=style.colour(), dashes=style.dashes,
            width=style.width,
        )  # fmt: skip
        # Observation points as ticks on the floor of each panel.
        theme.points(
            axes, list(OBSERVED), [bottom + 0.12] * len(OBSERVED),
            color=palette.muted, marker="^", size=30.0,
        )  # fmt: skip
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(bottom, high + 2.4)
        axes.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axes.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        axes.text(
            0.03, 0.97,
            f"{summary.name} · {summary.runs} runs\n{summary.note} · "
            f"RMSE {summary.rmse(truth.coarse):.2f}\nband covers "
            f"{summary.coverage(truth.coarse):.0%} of cells",
            transform=axes.transAxes, ha="left", va="top",
            fontsize=theme.ANNOTATION_SIZE, fontweight="semibold", color=palette.ink,
        )  # fmt: skip
    for axes in grid[1]:
        theme.axis_labels(axes, "Position x", None)
    for axes in grid[:, 0]:
        theme.axis_labels(axes, None, "Log-permeability g(x)")
    errors = [s.rmse(truth.coarse) for s in summaries]
    theme.title_block(
        grid[0][0],
        "All four methods recover the large-scale field\n"
        f"(RMSE {min(errors):.2f}-{max(errors):.2f}); the ensembles use "
        f"{summaries[3].runs}-{summaries[2].runs} runs",
        "Band: 90% pointwise (EKI: ensemble spread) · dashed: true field, 256 "
        "cells · triangles: the 7 readings",
    )
    return save(figure, "field-inversion-posteriors.png")


def acceptance_figure(rates: dict[str, tuple[float, ...]]) -> str:
    """Draw fixed-step pCN and random-walk acceptance against ``M``.

    References:
        [EQ-MCMC-PCN-001], [EQ-MCMC-MH-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    axes = theme.new_axes(figsize=(7.2, 4.0))
    palette = theme.active_palette()
    terms = list(STUDY_TERMS)
    rows = (
        ("pCN, beta = 0.2", rates["pcn"], theme.role_colour("model"), "s", None),
        (
            "random walk, step 0.2",
            rates["walk"],
            theme.role_colour("context"),
            "o",
            theme.FITTED,
        ),
    )
    labels = []
    for name, values, colour, marker, dashes in rows:
        theme.line(axes, terms, values, color=colour, dashes=dashes)
        theme.points(axes, terms, values, color=colour, marker=marker, size=46.0)
        labels.append((terms[-1], values[-1], name, colour))
    axes.set_xscale("log", base=2)
    axes.set_xticks(terms)
    axes.set_xticklabels([str(t) for t in terms])
    axes.minorticks_off()
    axes.set_xlim(9.0, 100.0)
    top = max(max(rates["pcn"]), max(rates["walk"])) * 1.25
    axes.set_ylim(0.0, top)
    ticks = [0.05 * k for k in range(int(top / 0.05) + 1)]
    axes.set_yticks(ticks)
    axes.set_yticklabels([f"{t:.2f}" for t in ticks])
    axes.grid(axis="x", visible=False)
    axes.grid(axis="y", color=palette.rule, linewidth=0.7)
    theme.direct_labels(axes, labels, gap_points=13.0)
    theme.axis_labels(axes, "KL truncation M", "Acceptance rate")
    theme.title_block(
        axes,
        f"Adding prior-dominated modes leaves pCN near {rates['pcn'][-1]:.2f}; the "
        f"random walk falls from {rates['walk'][0]:.2f} to {rates['walk'][-1]:.2f}",
        "Darcy inversion, same data · one chain per point, 300 warm-up + 1200 "
        f"draws from the prior mean, seed {SEED + 1}",
    )
    return save(axes.figure, "field-inversion-acceptance.png")


def main() -> None:
    """Run every section, write the figures, and print the elapsed time.

    References:
        [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-KAIPIO-SOMERSALO-2007].

    """
    start = time.perf_counter()
    truth = draw_truth()
    full = karhunen_loeve_interval(
        KERNEL, 0.0, 1.0, COARSE, rule="midpoint", variance_fraction=1.0
    )
    expansion = setup_section(truth, full)
    inversion = build_inversion(expansion, truth.data, truth.noise_sd)
    summaries = methods_section(inversion, truth)
    inverse_crime_section(expansion, truth, summaries[2])
    rates = truncation_section(full, truth)
    if has_matplotlib():
        written = (field_figure(truth, summaries), acceptance_figure(rates))
        print("figures: " + ", ".join(written))
    else:
        print("Matplotlib is not installed; figures skipped")
    print(f"elapsed {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
