"""Tempered SMC, a pCN dimension study and quadrature references for posteriors.

This module closes three gaps around the prior-whitened targets of
:mod:`probstat.inverse.posterior_targets`. It contains no new Markov kernel:
every sampler is the library's own
(:func:`probstat.inference.sequential_monte_carlo`,
:func:`probstat.inference.preconditioned_crank_nicolson`,
:func:`probstat.inference.random_walk_metropolis`).

Tempered SMC adapter
    :func:`tempered_smc_posterior` runs the library's tempered SMC on a prior
    sampler and a log-likelihood and returns the particles in physical
    coordinates (and in the sampling coordinates), the log-evidence estimate
    ``log Z_hat = sum_t log mean_i w_{t,i}`` ([EQ-INV-SMP-001]), the
    temperature schedule, the ESS of every stage and the number of resampling
    steps. The library resamples at *every* stage (the adaptive rule fixes the
    ESS at the threshold), so ``resample_count`` equals the number of stages.
    :func:`tempered_smc_from_target` does the same for a
    :class:`~probstat.inverse.posterior_targets.WhitenedGaussianPriorTarget`
    (pCN or random-walk mutation) and adds the likelihood normalizing
    constant, so that ``log Z_hat`` is the marginal likelihood ``log p(y)``.
    :func:`repeat_tempered_smc` repeats a run under distinct child seeds and
    summarizes the run-to-run spread of the evidence and of the posterior
    mean ([EQ-INV-SMP-002]). Numerical failures of the library (all particles
    with zero likelihood, too many stages, a singular particle covariance, a
    non-finite likelihood) are returned as ``converged=False`` with a
    ``stop_reason``; invalid arguments raise.

pCN dimension study
    :func:`pcn_dimension_study` discretises one linear-Gaussian
    function-space problem (prior coefficients with decay ``k^-decay``, a
    fixed number of point observations) at several dimensions ``n`` and runs
    pCN and a Gaussian random-walk Metropolis sampler with the *same* proposal
    scale ``beta`` and no adaptation, returning a structured table of measured
    acceptance rates ([EQ-INV-SMP-003]). The pCN acceptance depends on the
    likelihood ratio only; the random-walk acceptance also contains the prior
    ratio, whose log has a variance that grows like ``beta^2 n``. What is
    *measured* is reported; the expected pattern (pCN stable, random walk
    degrading) is not assumed by the code, and nothing is claimed for
    nonlinear targets.

Quadrature reference
    :func:`quadrature_reference` integrates an unnormalized log density on a
    bounded box (one or two dimensions) with a composite tensor
    Gauss-Legendre rule and returns the normalizing constant, mean,
    covariance, marginal quantiles and grid probabilities ([EQ-INV-SMP-004],
    [EQ-INV-SMP-005]). Truncation of the box is *estimated* (geometric
    extrapolation of the outer panel masses), the doubled-panel refinement
    error is reported, and warnings are attached when the estimates are not
    small. :func:`compare_to_reference` validates samples against the
    reference (moments, quantiles and the occupation of named regions such as
    the modes) and lists every failure; a sampler that misses a mode is
    reported as failed, never silently accepted ([EQ-INV-SMP-006]).
    :func:`square_bimodal_target` and :func:`banana_target` build two standard
    nonlinear test posteriors.

Experimental (added in 0.28).

References:
    [REF-DELMORAL-DOUCET-JASRA-2006], [REF-LEE-WHITELEY-2018],
    [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-ROBERTS-ROSENTHAL-2001],
    [REF-DAVIS-RABINOWITZ-1984], [REF-GOLUB-WELSCH-1969].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.inference import (
    MCMCResult,
    preconditioned_crank_nicolson,
    random_walk_metropolis,
    sequential_monte_carlo,
)
from probstat.linalg import (
    DenseOperator,
    DiagonalCovariance,
    DiagonalOperator,
    Matrix,
    Vector,
    as_vector,
)

from .gauss_newton import CallbackForward
from .linear_gaussian import linear_gaussian_posterior
from .posterior_targets import WhitenedGaussianPriorTarget

PriorSampler = Callable[[random.Random], Sequence[float]]
LogFunction = Callable[[Vector], float]
Region = tuple[str, Callable[[Vector], bool]]

_NUMERICAL_FAILURES = (
    "fewer than two particles",
    "tempering needed more than",
    "not positive definite",
    "NaN or +inf",
)
_MAX_GRID_POINTS = 2_000_000


# ---------------------------------------------------------------------------
# Tempered SMC adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemperedSMCPosterior:
    """Outcome of a tempered SMC run on an inverse-problem posterior.

    Attributes:
        converged: ``True`` when the run reached temperature one.
        stop_reason: ``"reached beta = 1"`` or the failure message of the
            library (then every field below is empty or NaN).
        samples: Equally weighted particles in physical coordinates.
        latent_samples: The same particles in the sampling coordinates
            (identical to ``samples`` when no map was given).
        names: Physical coordinate names.
        log_evidence: ``log Z_hat`` ([EQ-INV-SMP-001]); NaN on failure.
        log_evidence_standard_error: Lee-Whiteley delta-method standard error
            of ``log Z_hat`` for one run (NaN when unavailable); it excludes
            the bias of a poorly mixing kernel.
        temperatures: ``0 = beta_0 < ... < beta_T = 1``.
        ess_history: ESS of the incremental weights of every stage.
        resample_count: Number of resampling steps (one per stage).
        acceptance_rates: Mutation acceptance rate per stage.
        proposal_scales: Random-walk factor or pCN ``beta`` per stage.
        log_evidence_increments: ``log mean_i w_{t,i}`` per stage.
        mutation: ``"rwm"`` or ``"pcn"``.
        evaluations: Log-likelihood evaluations.

    References:
        [EQ-INV-SMP-001], [REF-DELMORAL-DOUCET-JASRA-2006].

    """

    converged: bool
    stop_reason: str
    samples: tuple[Vector, ...]
    latent_samples: tuple[Vector, ...]
    names: tuple[str, ...]
    log_evidence: float
    log_evidence_standard_error: float
    temperatures: tuple[float, ...]
    ess_history: tuple[float, ...]
    resample_count: int
    acceptance_rates: tuple[float, ...]
    proposal_scales: tuple[float, ...]
    log_evidence_increments: tuple[float, ...]
    mutation: str
    evaluations: int

    def mean(self) -> Vector:
        """Return the mean of the physical samples.

        Returns:
            The componentwise sample mean.

        Raises:
            ValueError: If the run failed and there are no samples.

        References:
            [EQ-INV-SMP-001].

        """
        if not self.samples:
            raise ValueError("the run failed; there are no samples")
        count = len(self.samples)
        return tuple(
            math.fsum(sample[i] for sample in self.samples) / count
            for i in range(len(self.samples[0]))
        )

    def variance(self) -> Vector:
        """Return the componentwise sample variance (divisor ``N``).

        Returns:
            The marginal variances of the physical samples.

        Raises:
            ValueError: If the run failed and there are no samples.

        References:
            [EQ-INV-SMP-001].

        """
        centre = self.mean()
        count = len(self.samples)
        return tuple(
            math.fsum((sample[i] - centre[i]) ** 2 for sample in self.samples) / count
            for i in range(len(centre))
        )


def _failed_result(
    message: str, mutation: str, names: tuple[str, ...]
) -> TemperedSMCPosterior:
    """Return the empty failed result carrying ``message``.

    References:
        [EQ-INV-SMP-001].

    """
    return TemperedSMCPosterior(
        False,
        message,
        (),
        (),
        names,
        math.nan,
        math.nan,
        (),
        (),
        0,
        (),
        (),
        (),
        mutation,
        0,
    )


def tempered_smc_posterior(
    prior_sampler: PriorSampler,
    log_likelihood: LogFunction,
    *,
    particles: int,
    log_prior: LogFunction | None = None,
    mutation: str = "pcn",
    mutation_steps: int = 5,
    ess_fraction: float = 0.5,
    resampling: str = "multinomial",
    max_stages: int = 200,
    to_physical: Callable[[Vector], Sequence[float]] | None = None,
    names: Sequence[str] | None = None,
    seed: int | RandomStream | None = None,
) -> TemperedSMCPosterior:
    """Run the library's tempered SMC on a posterior and report the schedule.

    ``log_likelihood`` must contain every normalizing constant of the data
    density if ``log_evidence`` is to be the marginal likelihood
    ([EQ-INV-SMP-001]). ``mutation="pcn"`` assumes the sampling coordinates
    have an exactly ``N(0, I)`` prior (``log_prior`` must then be ``None``);
    ``"rwm"`` needs ``log_prior``. Failures that the library raises as
    ``ValueError`` for numerical reasons are returned with
    ``converged=False``.

    Args:
        prior_sampler: ``generator -> x`` draws from the prior.
        log_likelihood: ``log L(x)`` (``-inf`` allowed).
        particles: Number of particles (at least two).
        log_prior: Prior log density for ``mutation="rwm"``.
        mutation: ``"pcn"`` (default) or ``"rwm"``.
        mutation_steps: Markov moves per particle and stage.
        ess_fraction: ESS target fraction in ``(0, 1)``.
        resampling: ``"multinomial"`` or ``"systematic"``.
        max_stages: Stage limit.
        to_physical: Optional map from sampling to physical coordinates.
        names: Physical coordinate names (default ``x0, x1, ...``).
        seed: Integer seed or stream.

    Returns:
        :class:`TemperedSMCPosterior`.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid options or ``particles < 2``.

    References:
        [EQ-INV-SMP-001], [REF-DELMORAL-DOUCET-JASRA-2006],
        [REF-LEE-WHITELEY-2018].

    """
    if not callable(prior_sampler) or not callable(log_likelihood):
        raise TypeError("prior_sampler and log_likelihood must be callable")
    if to_physical is not None and not callable(to_physical):
        raise TypeError("to_physical must be callable")
    count = require_non_negative_integer(particles, name="particles")
    if count < 2:
        raise ValueError("particles must be at least 2")
    stream = resolve_stream(seed)
    physical_map = to_physical if to_physical is not None else (lambda z: z)
    label: tuple[str, ...] = ()
    try:
        result = sequential_monte_carlo(
            prior_sampler,
            log_prior,
            log_likelihood,
            particles=count,
            mutation=mutation,
            mutation_steps=mutation_steps,
            ess_fraction=ess_fraction,
            resampling=resampling,
            max_stages=max_stages,
            seed=stream,
        )
    except ValueError as error:
        if any(phrase in str(error) for phrase in _NUMERICAL_FAILURES):
            return _failed_result(f"smc failed: {error}", mutation, label)
        raise
    latent = result.particles
    physical = tuple(
        as_vector(physical_map(z), name="to_physical output") for z in latent
    )
    dimension = len(physical[0])
    label = (
        tuple(names) if names is not None else tuple(f"x{i}" for i in range(dimension))
    )
    if len(label) != dimension or len(set(label)) != dimension:
        raise ValueError("names must be distinct and match the physical dimension")
    return TemperedSMCPosterior(
        True,
        "reached beta = 1",
        physical,
        latent,
        label,
        result.log_evidence,
        result.log_evidence_standard_error,
        result.temperatures,
        result.effective_sample_sizes,
        len(result.effective_sample_sizes),
        result.acceptance_rates,
        result.proposal_scales,
        result.log_evidence_increments,
        result.mutation,
        result.evaluations,
    )


def tempered_smc_from_target(
    target: WhitenedGaussianPriorTarget,
    *,
    particles: int,
    mutation: str = "pcn",
    mutation_steps: int = 5,
    ess_fraction: float = 0.5,
    resampling: str = "multinomial",
    max_stages: int = 200,
    seed: int | RandomStream | None = None,
) -> TemperedSMCPosterior:
    """Run tempered SMC on a prior-whitened target (prior ``N(0, I)``).

    The likelihood handed to the sampler is
    ``target.log_likelihood(z) + target.log_likelihood_constant`` so that the
    returned evidence is ``log p(y)`` including the data-density constants.
    Particles are drawn from the standard normal prior of ``z`` and mapped
    back with ``target.to_physical``.

    Args:
        target: The whitened target.
        particles: Number of particles (at least two).
        mutation: ``"pcn"`` (default) or ``"rwm"``.
        mutation_steps: Markov moves per particle and stage.
        ess_fraction: ESS target fraction.
        resampling: ``"multinomial"`` or ``"systematic"``.
        max_stages: Stage limit.
        seed: Integer seed or stream.

    Returns:
        :class:`TemperedSMCPosterior` (samples in ``x``, latent in ``z``).

    Raises:
        TypeError: If ``target`` is not a ``WhitenedGaussianPriorTarget``.
        ValueError: For invalid options.

    References:
        [EQ-INV-SMP-001], [REF-DELMORAL-DOUCET-JASRA-2006].

    """
    if not isinstance(target, WhitenedGaussianPriorTarget):
        raise TypeError("target must be a WhitenedGaussianPriorTarget")
    constant = target.log_likelihood_constant
    dimension = target.dimension

    def prior(generator: random.Random) -> Vector:
        """Draw ``z ~ N(0, I)``.

        References:
            [EQ-INV-SMP-001].

        """
        return tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))

    def likelihood(z: Vector) -> float:
        """Return the normalized log likelihood at ``z``.

        References:
            [EQ-INV-SMP-001].

        """
        return target.log_likelihood(z) + constant

    return tempered_smc_posterior(
        prior,
        likelihood,
        particles=particles,
        log_prior=target.log_prior if mutation == "rwm" else None,
        mutation=mutation,
        mutation_steps=mutation_steps,
        ess_fraction=ess_fraction,
        resampling=resampling,
        max_stages=max_stages,
        to_physical=target.to_physical,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class SMCRepeatSummary:
    """Run-to-run spread of repeated tempered SMC runs.

    Attributes:
        runs: Number of runs attempted.
        failed: Number of runs that returned ``converged=False``.
        log_evidences: ``log Z_hat`` of each converged run.
        log_evidence_mean: Mean of ``log_evidences``.
        log_evidence_std: Sample standard deviation across runs (NaN for one
            converged run).
        log_evidence_mean_se: ``std / sqrt(runs converged)``.
        mean_estimates: Posterior-mean estimate of each converged run.
        mean_of_means: Componentwise average of ``mean_estimates``.
        mean_std: Componentwise across-run standard deviation.
        mean_se: ``mean_std / sqrt(runs converged)``.
        variance_of_means: Average over runs of the within-run variance.

    References:
        [EQ-INV-SMP-002].

    """

    runs: int
    failed: int
    log_evidences: tuple[float, ...]
    log_evidence_mean: float
    log_evidence_std: float
    log_evidence_mean_se: float
    mean_estimates: tuple[Vector, ...]
    mean_of_means: Vector
    mean_std: Vector
    mean_se: Vector
    variance_of_means: Vector


def repeat_tempered_smc(
    run: Callable[[RandomStream], TemperedSMCPosterior],
    *,
    runs: int,
    seed: int | RandomStream | None = 0,
) -> SMCRepeatSummary:
    """Repeat a tempered SMC under distinct child seeds and summarize.

    The spread across independent runs is the honest Monte Carlo error of
    ``log Z_hat`` and of the posterior mean, including the effect of the
    adaptive schedule that the one-run Lee-Whiteley estimate does not cover
    ([EQ-INV-SMP-002]).

    Args:
        run: ``stream -> TemperedSMCPosterior``; it must use ``stream`` as
            its seed.
        runs: Number of repetitions (at least two).
        seed: Integer seed or stream for the children ``("run", i)``.

    Returns:
        :class:`SMCRepeatSummary`.

    Raises:
        TypeError: If ``run`` is not callable or returns the wrong type.
        ValueError: If ``runs < 2`` or no run converged.

    References:
        [EQ-INV-SMP-002], [REF-DELMORAL-DOUCET-JASRA-2006].

    """
    if not callable(run):
        raise TypeError("run must be callable")
    count = require_non_negative_integer(runs, name="runs")
    if count < 2:
        raise ValueError("runs must be at least 2")
    stream = resolve_stream(seed)
    results: list[TemperedSMCPosterior] = []
    for index in range(count):
        outcome = run(stream.child("run", index))
        if not isinstance(outcome, TemperedSMCPosterior):
            raise TypeError("run must return a TemperedSMCPosterior")
        results.append(outcome)
    good = [r for r in results if r.converged]
    if not good:
        raise ValueError("no run converged: " + results[0].stop_reason)
    n = len(good)
    evidences = tuple(r.log_evidence for r in good)
    centre = math.fsum(evidences) / n
    spread = (
        math.sqrt(math.fsum((e - centre) ** 2 for e in evidences) / (n - 1))
        if n > 1
        else math.nan
    )
    means = tuple(r.mean() for r in good)
    dimension = len(means[0])
    grand = tuple(math.fsum(m[i] for m in means) / n for i in range(dimension))
    deviation = tuple(
        (
            math.sqrt(math.fsum((m[i] - grand[i]) ** 2 for m in means) / (n - 1))
            if n > 1
            else math.nan
        )
        for i in range(dimension)
    )
    within = tuple(
        math.fsum(r.variance()[i] for r in good) / n for i in range(dimension)
    )
    return SMCRepeatSummary(
        count,
        count - n,
        evidences,
        centre,
        spread,
        spread / math.sqrt(n),
        means,
        grand,
        deviation,
        tuple(d / math.sqrt(n) for d in deviation),
        within,
    )


# ---------------------------------------------------------------------------
# Example nonlinear targets
# ---------------------------------------------------------------------------


def square_bimodal_target(
    data: float = 4.0, noise_sd: float = 0.5, prior_sd: float = 2.0
) -> WhitenedGaussianPriorTarget:
    """Return the posterior of ``y = x^2 + eps``, ``x ~ N(0, prior_sd^2)``.

    For ``y > 0`` the sign of ``x`` is not identifiable and the posterior is
    symmetric bimodal with modes near ``+-sqrt(y)`` (physical) and the
    barrier at ``x = 0``. The whitened coordinate is ``z = x / prior_sd``.

    Args:
        data: Observed ``y``.
        noise_sd: Noise standard deviation (positive).
        prior_sd: Prior standard deviation (positive).

    Returns:
        A one-dimensional :class:`WhitenedGaussianPriorTarget`.

    Raises:
        ValueError: If a scale is not positive and finite.

    References:
        [EQ-INV-SMP-004], [REF-STUART-2010].

    """
    if not (math.isfinite(noise_sd) and noise_sd > 0.0):
        raise ValueError("noise_sd must be positive and finite")
    if not (math.isfinite(prior_sd) and prior_sd > 0.0):
        raise ValueError("prior_sd must be positive and finite")

    def value(x: Vector) -> Vector:
        """Return ``(x^2,)``.

        References:
            [EQ-INV-SMP-004].

        """
        return (x[0] * x[0],)

    def jvp(x: Vector, v: Vector) -> Vector:
        """Return ``(2 x v,)``.

        References:
            [EQ-INV-SMP-004].

        """
        return (2.0 * x[0] * v[0],)

    return WhitenedGaussianPriorTarget(
        CallbackForward(1, 1, value, jvp, jvp),
        (data,),
        noise=DiagonalCovariance([noise_sd**2]),
        prior_factor=DenseOperator(((prior_sd,),)),
    )


def banana_target(
    data: tuple[float, float] = (1.0, 0.5),
    noise_sd: tuple[float, float] = (0.5, 0.3),
    prior_sd: float = 2.0,
) -> WhitenedGaussianPriorTarget:
    """Return the 2D banana posterior of ``y = (x1, x2 + x1^2) + eps``.

    The prior is ``N(0, prior_sd^2 I)``; the second observation is tight, so
    the posterior bends along ``x2 = y2 - x1^2``.

    Args:
        data: Observation ``(y1, y2)``.
        noise_sd: Noise standard deviations (positive).
        prior_sd: Prior standard deviation (positive).

    Returns:
        A two-dimensional :class:`WhitenedGaussianPriorTarget`.

    Raises:
        ValueError: If a scale is not positive and finite.

    References:
        [EQ-INV-SMP-004], [REF-STUART-2010].

    """
    if not all(math.isfinite(s) and s > 0.0 for s in (*noise_sd, prior_sd)):
        raise ValueError("noise_sd and prior_sd must be positive and finite")

    def value(x: Vector) -> Vector:
        """Return ``(x1, x2 + x1^2)``.

        References:
            [EQ-INV-SMP-004].

        """
        return (x[0], x[1] + x[0] * x[0])

    def jvp(x: Vector, v: Vector) -> Vector:
        """Return ``J(x) v``.

        References:
            [EQ-INV-SMP-004].

        """
        return (v[0], v[1] + 2.0 * x[0] * v[0])

    def vjp(x: Vector, w: Vector) -> Vector:
        """Return ``J(x)' w``.

        References:
            [EQ-INV-SMP-004].

        """
        return (w[0] + 2.0 * x[0] * w[1], w[1])

    return WhitenedGaussianPriorTarget(
        CallbackForward(2, 2, value, jvp, vjp),
        data,
        noise=DiagonalCovariance([noise_sd[0] ** 2, noise_sd[1] ** 2]),
        prior_factor=DenseOperator(((prior_sd, 0.0), (0.0, prior_sd))),
    )


# ---------------------------------------------------------------------------
# pCN dimension study
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DimensionStudyRow:
    """One discretisation level of :func:`pcn_dimension_study`.

    Attributes:
        dimension: Number of whitened coefficients ``n``.
        beta: Shared proposal scale (pCN ``beta``; random-walk step).
        pcn_acceptance: Measured pCN acceptance rate.
        rwm_acceptance: Measured random-walk acceptance rate (same scale).
        pcn_evaluations: Likelihood evaluations of the pCN run.
        rwm_evaluations: Density evaluations of the random-walk run.
        draws: Kept draws per chain (same for both samplers).
        chains: Chains per sampler.
        exact_mode_mean: Exact posterior mean of the first coefficient.
        pcn_mode_mean: pCN estimate of it.
        rwm_mode_mean: Random-walk estimate of it.

    References:
        [EQ-INV-SMP-003].

    """

    dimension: int
    beta: float
    pcn_acceptance: float
    rwm_acceptance: float
    pcn_evaluations: int
    rwm_evaluations: int
    draws: int
    chains: int
    exact_mode_mean: float
    pcn_mode_mean: float
    rwm_mode_mean: float


@dataclass(frozen=True, slots=True)
class DimensionStudy:
    """Structured result of :func:`pcn_dimension_study`.

    Attributes:
        rows: One row per dimension, in the order requested.
        observations: Number of point observations.
        noise_sd: Observation noise standard deviation.
        decay: Prior coefficient decay exponent.

    References:
        [EQ-INV-SMP-003].

    """

    rows: tuple[DimensionStudyRow, ...]
    observations: int
    noise_sd: float
    decay: float

    def as_table(self) -> str:
        """Return the measured acceptance rates as a plain-text table.

        Returns:
            One header line and one line per dimension.

        References:
            [EQ-INV-SMP-003].

        """
        lines = ["   n   beta  pCN accept  RWM accept  pCN evals  RWM evals"]
        for row in self.rows:
            lines.append(
                f"{row.dimension:4d}  {row.beta:5.2f}  {row.pcn_acceptance:10.3f}"
                f"  {row.rwm_acceptance:10.3f}  {row.pcn_evaluations:9d}"
                f"  {row.rwm_evaluations:9d}"
            )
        return "\n".join(lines)


def _discretised_target(
    n: int, observations: int, noise_sd: float, decay: float
) -> tuple[WhitenedGaussianPriorTarget, Matrix, Vector]:
    """Return the whitened target, the matrix ``A`` and the data.

    The field is ``u(s) = sum_k x_k cos(k pi s)`` with prior std
    ``k^-decay`` of ``x_k`` and ``observations`` evaluations at
    ``s_j = j / (observations + 1)``; the truth has ``x_1 = 1, x_2 = 0.5``.

    References:
        [EQ-INV-SMP-003].

    """
    rows: Matrix = tuple(
        tuple(math.cos(k * math.pi * j / (observations + 1)) for k in range(1, n + 1))
        for j in range(1, observations + 1)
    )
    prior_std = tuple(k**-decay for k in range(1, n + 1))
    truth = tuple((1.0 if k == 1 else 0.5 if k == 2 else 0.0) for k in range(1, n + 1))
    data = tuple(
        math.fsum(a * b for a, b in zip(row, truth, strict=True)) for row in rows
    )
    model = CallbackForward(
        n,
        observations,
        lambda x: tuple(
            math.fsum(a * b for a, b in zip(row, x, strict=True)) for row in rows
        ),
        jacobian=lambda x: rows,
    )
    target = WhitenedGaussianPriorTarget(
        model,
        data,
        noise=DiagonalCovariance([noise_sd**2] * observations),
        prior_factor=DiagonalOperator(prior_std),
    )
    return target, rows, data


def pcn_dimension_study(
    dimensions: Sequence[int] = (8, 32, 128),
    *,
    beta: float = 0.3,
    draws: int = 300,
    warmup: int = 100,
    chains: int = 2,
    observations: int = 4,
    noise_sd: float = 0.3,
    decay: float = 1.0,
    seed: int | RandomStream | None = 0,
) -> DimensionStudy:
    """Measure pCN and random-walk acceptance on a refined linear-Gaussian problem.

    For each ``n`` the same continuous problem is discretised
    (:func:`_discretised_target`) and sampled twice with the same fixed
    scale ``beta`` and no adaptation: pCN on ``log_likelihood`` and
    Gaussian random-walk Metropolis ``z' = z + beta xi`` on the full
    ``log_density`` (prior included) ([EQ-INV-SMP-003]). Starts are draws from
    the prior; ``warmup`` iterations are discarded, so the reported rates
    describe the near-stationary regime of a finite run. Acceptance rates are
    measured, not asserted: the table is the result.

    Args:
        dimensions: Discretisation sizes ``n >= 1``.
        beta: Proposal scale in ``(0, 1]``.
        draws: Kept draws per chain.
        warmup: Discarded iterations per chain.
        chains: Chains per sampler.
        observations: Number of point observations.
        noise_sd: Observation noise standard deviation.
        decay: Prior coefficient decay exponent (non-negative).
        seed: Integer seed or stream.

    Returns:
        :class:`DimensionStudy`.

    Raises:
        ValueError: For an invalid dimension, ``beta``, noise or decay.

    References:
        [EQ-INV-SMP-003], [REF-COTTER-ROBERTS-STUART-WHITE-2013],
        [REF-ROBERTS-ROSENTHAL-2001].

    """
    sizes = tuple(require_non_negative_integer(n, name="dimension") for n in dimensions)
    if not sizes or min(sizes) < 1:
        raise ValueError("dimensions must be positive integers")
    if not (math.isfinite(beta) and 0.0 < beta <= 1.0):
        raise ValueError("beta must lie in (0, 1]")
    if not (math.isfinite(noise_sd) and noise_sd > 0.0):
        raise ValueError("noise_sd must be positive and finite")
    if not (math.isfinite(decay) and decay >= 0.0):
        raise ValueError("decay must be non-negative and finite")
    stream = resolve_stream(seed)
    rows: list[DimensionStudyRow] = []
    for n in sizes:
        target, matrix, data = _discretised_target(n, observations, noise_sd, decay)
        level = stream.child("dimension", n)
        generator = level.child("starts").generator()
        starts = tuple(
            tuple(generator.gauss(0.0, 1.0) for _ in range(n)) for _ in range(chains)
        )
        pcn = preconditioned_crank_nicolson(
            target.log_likelihood,
            starts,
            draws=draws,
            warmup=warmup,
            chains=chains,
            step=beta,
            adapt=False,
            seed=level.child("pcn"),
            check_convergence=False,
        )
        identity = tuple(
            tuple(beta * beta if i == j else 0.0 for j in range(n)) for i in range(n)
        )
        rwm = random_walk_metropolis(
            target.log_density,
            starts,
            proposal_covariance=identity,
            draws=draws,
            warmup=warmup,
            chains=chains,
            seed=level.child("rwm"),
            check_convergence=False,
        )
        exact = linear_gaussian_posterior(
            matrix,
            data,
            noise=DiagonalCovariance([noise_sd**2] * observations),
            prior_covariance=DiagonalCovariance(
                [k ** (-2.0 * decay) for k in range(1, n + 1)]
            ),
        )
        rows.append(
            DimensionStudyRow(
                n,
                beta,
                pcn.acceptance_rate,
                rwm.acceptance_rate,
                pcn.evaluations,
                rwm.evaluations,
                draws,
                chains,
                exact.mean[0],
                _first_mean(pcn),
                _first_mean(rwm),
            )
        )
    return DimensionStudy(tuple(rows), observations, noise_sd, decay)


def _first_mean(result: MCMCResult) -> float:
    """Return the pooled mean of the first whitened coordinate.

    References:
        [EQ-INV-SMP-003].

    """
    return result.mean()[0]


# ---------------------------------------------------------------------------
# Quadrature reference
# ---------------------------------------------------------------------------


def _gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Newton iteration on ``P_order`` from the Chebyshev initial guesses
    [REF-GOLUB-WELSCH-1969]; the weights are ``2 / ((1 - x^2) P'(x)^2)``.

    References:
        [EQ-INV-SMP-004], [REF-DAVIS-RABINOWITZ-1984].

    """
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(order):
        x = math.cos(math.pi * (i + 0.75) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            p_prev, p_curr = 1.0, x
            for k in range(2, order + 1):
                p_prev, p_curr = (
                    p_curr,
                    ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k,
                )
            derivative = order * (x * p_curr - p_prev) / (x * x - 1.0)
            step = p_curr / derivative
            x -= step
            if abs(step) < 1e-15:
                break
        p_prev, p_curr = 1.0, x
        for k in range(2, order + 1):
            p_prev, p_curr = p_curr, ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        derivative = order * (x * p_curr - p_prev) / (x * x - 1.0)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    pairs = sorted(zip(nodes, weights, strict=True))
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def _composite_grid(
    lower: float, upper: float, panels: int, order: int
) -> tuple[list[float], list[float]]:
    """Return composite Gauss-Legendre nodes and weights on ``[lower, upper]``.

    References:
        [EQ-INV-SMP-004].

    """
    base_x, base_w = _gauss_legendre(order)
    width = (upper - lower) / panels
    xs: list[float] = []
    ws: list[float] = []
    for p in range(panels):
        centre = lower + (p + 0.5) * width
        for t, w in zip(base_x, base_w, strict=True):
            xs.append(centre + 0.5 * width * t)
            ws.append(0.5 * width * w)
    return xs, ws


def _quadrature_pass(
    log_density: LogFunction,
    bounds: tuple[tuple[float, float], ...],
    panels: tuple[int, ...],
    order: int,
) -> tuple[
    list[list[float]], list[list[float]], list[tuple[Vector, float, float]], float
]:
    """Evaluate the density on the tensor grid.

    Returns the per-dimension nodes, weights, a list of ``(point, w, mass)``
    with ``mass = w * exp(log f - shift)`` and the shift.

    References:
        [EQ-INV-SMP-004].

    """
    axes = [
        _composite_grid(lo, hi, p, order)
        for (lo, hi), p in zip(bounds, panels, strict=True)
    ]
    total = 1
    for axis in axes:
        total *= len(axis[0])
    if total > _MAX_GRID_POINTS:
        raise ValueError(f"grid of {total} points exceeds {_MAX_GRID_POINTS}")
    raw: list[tuple[Vector, float, float]] = []
    if len(axes) == 1:
        for x, w in zip(axes[0][0], axes[0][1], strict=True):
            raw.append(((x,), w, _checked(log_density((x,)))))
    else:
        for x, wx in zip(axes[0][0], axes[0][1], strict=True):
            for y, wy in zip(axes[1][0], axes[1][1], strict=True):
                raw.append(((x, y), wx * wy, _checked(log_density((x, y)))))
    finite = [value for _, _, value in raw if value != -math.inf]
    if not finite:
        raise ValueError("the density is zero everywhere on the box")
    shift = max(finite)
    cells = [
        (point, w, w * math.exp(value - shift) if value != -math.inf else 0.0)
        for point, w, value in raw
    ]
    return [a[0] for a in axes], [a[1] for a in axes], cells, shift


def _checked(value: float) -> float:
    """Return ``float(value)`` or raise for NaN and ``+inf``.

    References:
        [EQ-INV-SMP-004].

    """
    number = float(value)
    if math.isnan(number) or number == math.inf:
        raise ValueError("log_density returned NaN or +inf")
    return number


@dataclass(frozen=True, slots=True)
class QuadratureReference:
    """Deterministic reference of a one- or two-dimensional density.

    Attributes:
        dimension: 1 or 2.
        bounds: Box ``((lo, hi), ...)``.
        panels: Panels per dimension.
        order: Gauss-Legendre points per panel.
        log_normalizer: ``log integral exp(log_density)`` over the box.
        mean: Posterior mean.
        covariance: Posterior covariance.
        quantiles: Per dimension, ``(probability, value)`` pairs of the
            marginal ([EQ-INV-SMP-005]).
        truncation_mass_estimate: Heuristic estimate of the mass outside the
            box (geometric extrapolation of the outer panel masses of each
            marginal); an estimate, not a bound.
        boundary_density_ratio: Largest ratio of the density on the box
            boundary nodes to the density maximum.
        refinement_log_error: ``|log Z(2 panels) - log Z(panels)|``.
        refinement_mean_error: Largest change of the mean under doubled
            panels, in units of the posterior standard deviation.
        evaluations: Density evaluations of the main pass.
        grid_points: Quadrature points (main pass).
        grid_probabilities: Normalized probability mass of each point.
        warnings: Messages when the truncation or refinement estimates are
            not small.

    References:
        [EQ-INV-SMP-004], [EQ-INV-SMP-005].

    """

    dimension: int
    bounds: tuple[tuple[float, float], ...]
    panels: tuple[int, ...]
    order: int
    log_normalizer: float
    mean: Vector
    covariance: Matrix
    quantiles: tuple[tuple[tuple[float, float], ...], ...]
    truncation_mass_estimate: float
    boundary_density_ratio: float
    refinement_log_error: float
    refinement_mean_error: float
    evaluations: int
    grid_points: tuple[Vector, ...]
    grid_probabilities: tuple[float, ...]
    warnings: tuple[str, ...]

    def std(self) -> Vector:
        """Return the marginal standard deviations.

        Returns:
            ``sqrt`` of the covariance diagonal.

        References:
            [EQ-INV-SMP-004].

        """
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(self.dimension))

    def region_mass(self, predicate: Callable[[Vector], bool]) -> float:
        """Return the reference probability of ``{x : predicate(x)}``.

        The sum is over grid points, so it is spectrally accurate only when
        the region boundary coincides with panel boundaries (for example
        ``x > 0`` on a box symmetric about zero with an even number of
        panels); otherwise the error is of the order of one panel's mass.

        Args:
            predicate: Region indicator on points.

        Returns:
            The probability mass of the region.

        References:
            [EQ-INV-SMP-006].

        """
        return math.fsum(
            p
            for point, p in zip(self.grid_points, self.grid_probabilities, strict=True)
            if predicate(point)
        )


def _marginal_nodes(
    cells: list[tuple[Vector, float]], axis: int, nodes: list[float]
) -> list[float]:
    """Return the marginal mass of every node of ``axis`` (unnormalized).

    References:
        [EQ-INV-SMP-005].

    """
    index = {x: i for i, x in enumerate(nodes)}
    mass = [0.0] * len(nodes)
    for point, m in cells:
        mass[index[point[axis]]] += m
    return mass


def _panel_quantile(
    xs: list[float],
    ws: list[float],
    masses: list[float],
    order: int,
    interval: tuple[float, float],
    probability: float,
) -> float:
    """Return the marginal quantile of ``probability`` by bisection.

    The marginal density at a node is ``mass / weight``. Within a panel it
    is the degree ``order - 1`` interpolant of the nodal values (exact for
    polynomial densities of that degree), integrated by Gauss-Legendre
    [EQ-INV-SMP-005].

    References:
        [EQ-INV-SMP-005], [REF-DAVIS-RABINOWITZ-1984].

    """
    panels = len(xs) // order
    width = (interval[1] - interval[0]) / panels
    target = probability * math.fsum(masses)
    densities = [m / w for m, w in zip(masses, ws, strict=True)]
    base_x, base_w = _gauss_legendre(order)
    cumulative = 0.0
    chosen = panels - 1
    for p in range(panels):
        panel_mass = math.fsum(masses[p * order : (p + 1) * order])
        if cumulative + panel_mass >= target:
            chosen = p
            break
        cumulative += panel_mass
    node_x = xs[chosen * order : (chosen + 1) * order]
    node_f = densities[chosen * order : (chosen + 1) * order]
    start = interval[0] + chosen * width

    def interpolant(x: float) -> float:
        """Evaluate the Lagrange interpolant of the panel density at ``x``.

        References:
            [EQ-INV-SMP-005].

        """
        value = 0.0
        for i in range(order):
            term = node_f[i]
            for j in range(order):
                if j != i:
                    term *= (x - node_x[j]) / (node_x[i] - node_x[j])
            value += term
        return value

    def partial(x: float) -> float:
        """Return the integral of the interpolant from the panel start to ``x``.

        References:
            [EQ-INV-SMP-005].

        """
        half_length = 0.5 * (x - start)
        middle = 0.5 * (x + start)
        return math.fsum(
            half_length * w * interpolant(middle + half_length * t)
            for t, w in zip(base_x, base_w, strict=True)
        )

    lo, hi = start, start + width
    for _ in range(80):
        middle = 0.5 * (lo + hi)
        if cumulative + partial(middle) < target:
            lo = middle
        else:
            hi = middle
    return 0.5 * (lo + hi)


def _tail_estimate(panel_masses: list[float]) -> tuple[float, bool]:
    """Return the extrapolated mass beyond both ends and a flag for no decay.

    References:
        [EQ-INV-SMP-005].

    """
    total = 0.0
    no_decay = False
    for outer, inner in (
        (panel_masses[0], panel_masses[1] if len(panel_masses) > 1 else 0.0),
        (panel_masses[-1], panel_masses[-2] if len(panel_masses) > 1 else 0.0),
    ):
        if outer <= 0.0:
            continue
        ratio = outer / inner if inner > 0.0 else math.inf
        if ratio < 1.0:
            total += outer * ratio / (1.0 - ratio)
        else:
            total += outer
            no_decay = True
    return total, no_decay


def quadrature_reference(
    log_density: LogFunction,
    bounds: Sequence[tuple[float, float]],
    *,
    panels: int | Sequence[int] = 32,
    order: int = 8,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    refine_check: bool = True,
    refinement_tolerance: float = 1e-6,
) -> QuadratureReference:
    """Integrate an unnormalized log density on a box by tensor Gauss-Legendre.

    The box is split into ``panels`` panels per dimension with ``order``
    Gauss-Legendre points each ([EQ-INV-SMP-004]). The result holds ``log Z``,
    mean, covariance and marginal quantiles ([EQ-INV-SMP-005]), the
    heuristic truncation mass estimate, and, with ``refine_check``, the change
    under doubled panels. The reference is only as good as the box and the
    resolution; read ``warnings``.

    Args:
        log_density: ``x -> log f(x)`` (tuple of length 1 or 2); ``-inf``
            allowed; include every constant that should enter ``log Z``.
        bounds: ``[(lo, hi)]`` or ``[(lo1, hi1), (lo2, hi2)]``.
        panels: Panels per dimension (int or one per dimension).
        order: Gauss-Legendre points per panel (2 to 40).
        quantiles: Marginal probabilities in ``(0, 1)``.
        refine_check: Also run the doubled-panel pass.
        refinement_tolerance: Largest tolerated change of ``log Z`` under
            doubled panels before a warning is attached.

    Returns:
        :class:`QuadratureReference`.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid bounds, panels or order, a grid larger than
            two million points, a density that is zero on the whole box, or a
            NaN or ``+inf`` density value.

    References:
        [EQ-INV-SMP-004], [EQ-INV-SMP-005], [REF-DAVIS-RABINOWITZ-1984].

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    box = tuple((float(lo), float(hi)) for lo, hi in bounds)
    dimension = len(box)
    if dimension not in (1, 2):
        raise ValueError("only one- and two-dimensional boxes are supported")
    if any(not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) for lo, hi in box):
        raise ValueError("bounds must be finite with lo < hi")
    if isinstance(panels, int):
        panel_tuple = (panels,) * dimension
    else:
        panel_tuple = tuple(panels)
    if len(panel_tuple) != dimension or any(
        not isinstance(p, int) or p < 1 for p in panel_tuple
    ):
        raise ValueError("panels must be positive integers, one per dimension")
    if not isinstance(order, int) or not 2 <= order <= 40:
        raise ValueError("order must be an integer in [2, 40]")
    levels = tuple(float(q) for q in quantiles)
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError("quantile probabilities must lie in (0, 1)")

    axes_x, axes_w, cells, shift = _quadrature_pass(
        log_density, box, panel_tuple, order
    )
    total = math.fsum(m for _, _, m in cells)
    probabilities = tuple(m / total for _, _, m in cells)
    points = tuple(point for point, _, _ in cells)
    mean = tuple(
        math.fsum(p[i] * q for p, q in zip(points, probabilities, strict=True))
        for i in range(dimension)
    )
    covariance = tuple(
        tuple(
            math.fsum(
                (p[i] - mean[i]) * (p[j] - mean[j]) * q
                for p, q in zip(points, probabilities, strict=True)
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    normalized_cells = [(p, m / total) for p, _, m in cells]
    tails = 0.0
    no_decay = False
    quantile_rows: list[tuple[tuple[float, float], ...]] = []
    boundary_ratio = 0.0
    maximum_density = max(m / w for _, w, m in cells)
    for axis in range(dimension):
        masses = _marginal_nodes(normalized_cells, axis, axes_x[axis])
        panel_masses = [
            math.fsum(masses[k * order : (k + 1) * order])
            for k in range(panel_tuple[axis])
        ]
        tail, flat = _tail_estimate(panel_masses)
        tails += tail
        no_decay = no_decay or flat
        quantile_rows.append(
            tuple(
                (
                    q,
                    _panel_quantile(
                        axes_x[axis], axes_w[axis], masses, order, box[axis], q
                    ),
                )
                for q in levels
            )
        )
        for p, w, m in cells:
            if p[axis] in (axes_x[axis][0], axes_x[axis][-1]):
                boundary_ratio = max(boundary_ratio, (m / w) / maximum_density)
    log_normalizer = shift + math.log(total)
    refinement_log = math.nan
    refinement_mean = math.nan
    if refine_check:
        doubled = tuple(2 * p for p in panel_tuple)
        _, _, cells2, shift2 = _quadrature_pass(log_density, box, doubled, order)
        total2 = math.fsum(m for _, _, m in cells2)
        refinement_log = abs((shift2 + math.log(total2)) - log_normalizer)
        mean2 = tuple(
            math.fsum(p[i] * m for p, _, m in cells2) / total2 for i in range(dimension)
        )
        deviation = tuple(math.sqrt(covariance[i][i]) for i in range(dimension))
        refinement_mean = max(
            abs(mean2[i] - mean[i]) / deviation[i] if deviation[i] > 0.0 else 0.0
            for i in range(dimension)
        )
    messages: list[str] = []
    if tails > 1e-6:
        messages.append(
            f"estimated truncation mass {tails:.2e} exceeds 1e-6; enlarge the box"
        )
    if no_decay:
        messages.append("an outer panel mass does not decay; the box is too small")
    if refine_check and refinement_log > refinement_tolerance:
        messages.append(
            f"doubling the panels changed log Z by {refinement_log:.2e}; "
            "increase panels or order"
        )
    return QuadratureReference(
        dimension,
        box,
        panel_tuple,
        order,
        log_normalizer,
        mean,
        covariance,
        tuple(quantile_rows),
        tails,
        boundary_ratio,
        refinement_log,
        refinement_mean,
        len(cells),
        points,
        probabilities,
        tuple(messages),
    )


def reference_for_target(
    target: WhitenedGaussianPriorTarget,
    bounds: Sequence[tuple[float, float]],
    **options: object,
) -> QuadratureReference:
    """Return the quadrature reference of the whitened posterior of ``z``.

    The integrand is ``target.log_density(z) + target.log_likelihood_constant
    = log [p(y | z) N(z; 0, I)]``, whose integral over all of ``R^r`` is the
    evidence ``log p(y)``, so ``log_normalizer`` is directly comparable with
    the SMC evidence ([EQ-INV-SMP-004]).

    Args:
        target: The whitened target (dimension 1 or 2).
        bounds: Box in ``z`` coordinates.
        **options: Passed to :func:`quadrature_reference`.

    Returns:
        :class:`QuadratureReference` in ``z`` coordinates.

    Raises:
        TypeError: If ``target`` is not a ``WhitenedGaussianPriorTarget``.
        ValueError: As :func:`quadrature_reference`.

    References:
        [EQ-INV-SMP-004].

    """
    if not isinstance(target, WhitenedGaussianPriorTarget):
        raise TypeError("target must be a WhitenedGaussianPriorTarget")
    constant = target.log_likelihood_constant
    return quadrature_reference(
        lambda z: target.log_density(z) + constant,
        bounds,
        **options,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Comparison with a reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegionReport:
    """Occupation of one named region compared with the reference.

    Attributes:
        name: Region label.
        reference_mass: Reference probability of the region.
        sample_fraction: Fraction of the samples inside.
        status: ``"ok"``, ``"under-occupied"``, ``"over-occupied"`` or
            ``"missed"`` (sample fraction below a tenth of the reference
            mass).

    References:
        [EQ-INV-SMP-006].

    """

    name: str
    reference_mass: float
    sample_fraction: float
    status: str


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """Samples compared with a :class:`QuadratureReference`.

    Attributes:
        mean_error_sd: Per dimension, ``|mean_sample - mean_ref| / sd_ref``.
        std_ratio: Per dimension, ``sd_sample / sd_ref``.
        quantile_error_sd: Per dimension, the largest quantile error over the
            reference levels in units of ``sd_ref``.
        regions: One :class:`RegionReport` per region.
        failures: Human-readable reasons the comparison failed (empty when
            it passed).
        passed: ``not failures``.

    References:
        [EQ-INV-SMP-006].

    """

    mean_error_sd: Vector
    std_ratio: Vector
    quantile_error_sd: Vector
    regions: tuple[RegionReport, ...]
    failures: tuple[str, ...]
    passed: bool


def _sample_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the type-7 sample quantile of sorted values.

    References:
        [EQ-INV-SMP-006].

    """
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (position - low) * (
        sorted_values[high] - sorted_values[low]
    )


def compare_to_reference(
    samples: Sequence[Sequence[float]],
    reference: QuadratureReference,
    *,
    regions: Sequence[Region] = (),
    mean_tolerance: float = 0.25,
    std_tolerance: float = 0.3,
    quantile_tolerance: float = 0.35,
    occupation_tolerance: float = 0.15,
) -> ReferenceComparison:
    """Compare samples with a quadrature reference and list every failure.

    Moments and quantiles are compared in units of the reference standard
    deviation; named regions (for example the two modes) are compared by
    occupation. The tolerances are screening thresholds that must be chosen
    from the Monte Carlo error of the run; a sampler that never visits a
    region with reference mass above ``occupation_tolerance`` is reported as
    ``"missed"`` and fails the comparison ([EQ-INV-SMP-006]).

    Args:
        samples: Draws (pooled), same dimension as the reference.
        reference: The reference.
        regions: ``(name, predicate)`` pairs.
        mean_tolerance: Largest allowed mean error in reference sd units.
        std_tolerance: Largest allowed ``|std ratio - 1|``.
        quantile_tolerance: Largest allowed quantile error in sd units.
        occupation_tolerance: Largest allowed ``|fraction - mass|``.

    Returns:
        :class:`ReferenceComparison`.

    Raises:
        ValueError: If there are fewer than two samples or the dimensions
            disagree.

    References:
        [EQ-INV-SMP-006].

    """
    draws = [as_vector(s, name="sample") for s in samples]
    if len(draws) < 2:
        raise ValueError("at least two samples are required")
    d = reference.dimension
    if any(len(s) != d for s in draws):
        raise ValueError(f"samples must have dimension {d}")
    count = len(draws)
    sd_ref = reference.std()
    means = tuple(math.fsum(s[i] for s in draws) / count for i in range(d))
    sds = tuple(
        math.sqrt(math.fsum((s[i] - means[i]) ** 2 for s in draws) / count)
        for i in range(d)
    )
    mean_error = tuple(abs(means[i] - reference.mean[i]) / sd_ref[i] for i in range(d))
    ratio = tuple(sds[i] / sd_ref[i] for i in range(d))
    quantile_error: list[float] = []
    for i in range(d):
        ordered = sorted(s[i] for s in draws)
        quantile_error.append(
            max(
                abs(_sample_quantile(ordered, p) - value) / sd_ref[i]
                for p, value in reference.quantiles[i]
            )
        )
    reports: list[RegionReport] = []
    failures: list[str] = []
    for name, predicate in regions:
        mass = reference.region_mass(predicate)
        fraction = sum(1 for s in draws if predicate(s)) / count
        if fraction < 0.1 * mass and mass > occupation_tolerance:
            status = "missed"
        elif abs(fraction - mass) <= occupation_tolerance:
            status = "ok"
        elif fraction < mass:
            status = "under-occupied"
        else:
            status = "over-occupied"
        reports.append(RegionReport(name, mass, fraction, status))
        if status != "ok":
            failures.append(
                f"region {name!r}: reference mass {mass:.3f}, "
                f"sample fraction {fraction:.3f} ({status})"
            )
    for i in range(d):
        if mean_error[i] > mean_tolerance:
            failures.append(f"mean of coordinate {i} off by {mean_error[i]:.3f} sd")
        if abs(ratio[i] - 1.0) > std_tolerance:
            failures.append(f"std ratio of coordinate {i} is {ratio[i]:.3f}")
        if quantile_error[i] > quantile_tolerance:
            failures.append(
                f"quantile error of coordinate {i} is {quantile_error[i]:.3f} sd"
            )
    return ReferenceComparison(
        mean_error,
        ratio,
        tuple(quantile_error),
        tuple(reports),
        tuple(failures),
        not failures,
    )


def chain_region_fractions(
    result: MCMCResult, predicate: Callable[[Vector], bool]
) -> tuple[float, ...]:
    """Return, per chain, the fraction of draws inside a region.

    Chains that never leave one mode show up as fractions near 0 or 1; the
    pooled fraction of such a run can look balanced by accident.

    Args:
        result: Multi-chain MCMC result.
        predicate: Region indicator.

    Returns:
        One fraction per chain.

    Raises:
        TypeError: If ``result`` is not an ``MCMCResult``.

    References:
        [EQ-INV-SMP-006].

    """
    if not isinstance(result, MCMCResult):
        raise TypeError("result must be an MCMCResult")
    return tuple(
        sum(1 for s in chain.samples if predicate(s)) / len(chain.samples)
        for chain in result.chains
    )
