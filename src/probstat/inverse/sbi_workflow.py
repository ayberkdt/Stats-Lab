"""Simulation-based inference workflow for inverse problems (ABC, NPE, NRE).

Experimental (added in 0.28): the API may change before it is declared stable.
Every learned or simulated posterior in this module is *approximate*; each
result says so in its ``approximate`` or ``status`` field and nothing here
certifies a posterior.

The module wraps, and does not reimplement, the library's likelihood-free
tools: :func:`probstat.inference.abc_rejection` and
:func:`probstat.inference.abc_smc`, the Gaussian neural posterior
:func:`probstat.inference.fit_amortized_gaussian_posterior`, the ratio
estimator :func:`probstat.generative.fit_ratio_estimator`, and
:func:`probstat.inference.simulation_based_calibration`. It adds

* a **simulator contract** (prior sampler, simulator with an explicit random
  stream, optional summary) with work accounting and simulator-failure
  tracking, and train / validation / test splits drawn from separate child
  streams (:class:`SimulatorContract`, :class:`WorkLedger`,
  :func:`make_splits`);
* a **Gaussian synthetic likelihood** with finite-simulation covariance, its
  Monte Carlo variance, and a reported (not jittered) singular covariance
  (:func:`synthetic_log_likelihood`); it is the likelihood of a *Gaussian
  model of the summaries*, not the likelihood of the simulator [EQ-INV-SBI-002];
* a **tolerance study** on a linear-Gaussian model whose posterior is known
  in closed form, with a sufficient and an insufficient summary
  (:func:`abc_tolerance_study`, [EQ-INV-SBI-001]);
* a **bimodal comparison** on ``y = theta^2 + noise`` with an exact
  quadrature reference, ABC, the library NPE and NRE, and the prior as a
  baseline (:func:`run_bimodal_comparison`, [EQ-INV-SBI-003],
  [EQ-INV-SBI-004]);
* a **Gaussian score test** that checks, in closed form, that the posterior
  score at diffusion time zero is the prior score plus the likelihood
  gradient and that naive addition at a noisy time is not exact
  (:func:`noisy_score_comparison`, [EQ-INV-SBI-005], [EQ-INV-SBI-006]);
* a **plug-and-play proximal gradient iteration** with a user denoiser and a
  fixed-point residual (:func:`plug_and_play_proximal_gradient`,
  [EQ-INV-SBI-007]);
* a declared **out-of-distribution heuristic** on the summaries
  (:func:`fit_summary_ood_detector`, [EQ-INV-SBI-008]).

Added in the second round:

* **diffusion posterior sampling with exact scores** for a linear-Gaussian
  problem (no network): variance-preserving forward noising, the closed-form
  noisy-time posterior score, and Euler-Maruyama of the reverse SDE or Euler
  of the probability-flow ODE, with a measured convergence study against the
  exact posterior and a clearly labelled *approximate* DPS-style variant
  (:func:`diffusion_posterior_sampling`, :func:`diffusion_convergence_study`,
  [EQ-INV-SBI-009], [EQ-INV-SBI-010], [EQ-INV-SBI-011]);
* an optional **flow-matching row** in the bimodal comparison
  (``BimodalConfig(include_flow_matching=True)``, :func:`train_flow_matching`,
  [EQ-INV-SBI-012]); it is a sample-only method here, so the comparison uses
  no density, evidence, or Metropolis-Hastings correction for it;
* an explicit **training policy**: only prior simulations are accepted
  (``training_policy="prior"``); proposal-based (sequential) training is
  refused with :class:`ProposalTrainingRefused`, never silently run without
  the proposal-shift correction (:func:`check_training_policy`,
  [EQ-INV-SBI-013]).

Deferred (not implemented here): diffusion posterior sampling with *learned*
scores, sequential/proposal-corrected training, and deep-image-prior or
unrolled networks. The analytic score tests are the prerequisite the roadmap
asks for before any neural claim.

Failure handling: a non-finite or raising simulation raises
:class:`SimulationFailure` and is counted in the :class:`WorkLedger`; a
training exception is returned as ``status="failed"``; zero ABC acceptance
is returned as ``status="zero_acceptance"``; a singular synthetic-likelihood
covariance is returned as ``status="singular_covariance"``. Nothing is
clipped, jittered, or replaced silently.
"""

from __future__ import annotations

import bisect
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.generative import (
    FlowMatchingPosterior,
    RatioEstimator,
    fit_flow_matching_posterior,
    fit_ratio_estimator,
)
from probstat.inference import (
    ABCResult,
    AmortizedGaussianPosterior,
    abc_rejection,
    abc_smc,
    fit_amortized_gaussian_posterior,
    simulation_based_calibration,
)
from probstat.information import empirical_wasserstein_1d
from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_matrix,
    as_vector,
    cholesky,
    identity,
    inverse_spd,
    log_det_spd,
    matmul,
    matvec,
    scale_matrix,
    solve_lower,
    symmetrize,
    transpose,
)

__all__ = [
    "ABCOutcome",
    "BimodalComparison",
    "BimodalConfig",
    "DiffusionConvergence",
    "DiffusionSamplingResult",
    "GaussianScoreProblem",
    "GridDensity",
    "LinearGaussianCase",
    "LinearGaussianDiffusionProblem",
    "MethodReport",
    "NoisyScoreComparison",
    "OODSummary",
    "PREDECLARED_SEEDS",
    "PlugAndPlayResult",
    "ProposalTrainingRefused",
    "SimulationDataset",
    "SimulationFailure",
    "SimulationSplits",
    "SimulatorContract",
    "SummaryOODDetector",
    "SyntheticLikelihoodGrid",
    "SyntheticLikelihoodResult",
    "ToleranceRow",
    "ToleranceStudy",
    "TrainedModel",
    "VPSchedule",
    "WorkLedger",
    "abc_tolerance_study",
    "bimodal_contract",
    "bimodal_exact_posterior",
    "check_training_policy",
    "crps_ensemble_value",
    "diffusion_convergence_study",
    "diffusion_posterior_sampling",
    "draw_dataset",
    "fit_summary_ood_detector",
    "gaussian_proximal_denoiser",
    "linear_gaussian_case",
    "linear_gaussian_contract",
    "make_splits",
    "noisy_score_comparison",
    "plug_and_play_proximal_gradient",
    "posterior_score_t0",
    "run_abc_rejection",
    "run_abc_smc",
    "run_bimodal_comparison",
    "simulate_summary",
    "slow_bimodal_config",
    "synthetic_likelihood_grid",
    "synthetic_log_likelihood",
    "train_flow_matching",
    "train_npe",
    "train_nre",
]

PREDECLARED_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 67, 71, 89)
"""Seeds fixed before any study is run; studies default to these."""

_FAILURE_POLICIES = ("raise", "discard")
_PROPOSAL_POLICIES = ("proposal", "sequential", "snpe", "truncated_proposal")
_PIVOT_RATIO = 1e-7  # Cholesky pivot ratio below which a covariance is singular
_MAX_ATTEMPTS_PER_PAIR = 50

PriorSampler = Callable[[random.Random], Sequence[float]]
DataSimulator = Callable[[Vector, random.Random], Sequence[float]]
SummaryMap = Callable[[Vector], Sequence[float]]
LogPrior = Callable[[Sequence[float]], float]
Distance = Callable[[Vector, Vector], float]
Denoiser = Callable[[Vector, float], Sequence[float]]


class SimulationFailure(ValueError):
    """A simulator or summary call failed or returned a non-finite value.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """


@dataclass(slots=True)
class WorkLedger:
    """Mutable count of simulator calls and failures.

    Attributes:
        simulator_calls: Number of simulator invocations (including failed
            ones).
        failures: Number of invocations that raised :class:`SimulationFailure`.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """

    simulator_calls: int = 0
    failures: int = 0

    @property
    def failure_rate(self) -> float:
        """Return ``failures / simulator_calls`` (zero before any call).

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        return self.failures / self.simulator_calls if self.simulator_calls else 0.0


@dataclass(frozen=True, slots=True)
class SimulatorContract:
    """A prior sampler, a simulator with an explicit stream, and a summary.

    Args:
        prior_sampler: ``generator -> theta``.
        simulator: ``(theta, generator) -> data``; observation noise is drawn
            from ``generator`` only.
        summary: Optional ``data -> summary``; ``None`` uses the data itself.
        log_prior: Optional prior log-density (needed by ABC-SMC and by grid
            posteriors); ``-inf`` outside the support.
        name: Label used in reports.

    References:
        [EQ-INV-SBI-001], [REF-PRITCHARD-ETAL-1999].

    """

    prior_sampler: PriorSampler
    simulator: DataSimulator
    summary: SummaryMap | None = None
    log_prior: LogPrior | None = None
    name: str = "simulator"

    def __post_init__(self) -> None:
        """Validate the callables.

        Raises:
            TypeError: If a required callable is not callable.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        if not callable(self.prior_sampler) or not callable(self.simulator):
            raise TypeError("prior_sampler and simulator must be callable")
        if self.summary is not None and not callable(self.summary):
            raise TypeError("summary must be callable or None")
        if self.log_prior is not None and not callable(self.log_prior):
            raise TypeError("log_prior must be callable or None")

    def bind(
        self, ledger: WorkLedger | None
    ) -> tuple[PriorSampler, Callable[[Sequence[float], random.Random], Vector]]:
        """Return library-compatible ``(prior_sampler, summary_simulator)``.

        The returned simulator validates its output and records every call
        in ``ledger``; it is what the library ABC/NPE/NRE routines receive.

        Args:
            ledger: Ledger to update, or ``None`` for no accounting.

        Returns:
            The prior sampler and a ``(theta, generator) -> summary`` callable.

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """

        def simulator(theta: Sequence[float], rng: random.Random) -> Vector:
            """Return the validated summary of one simulation.

            References:
                [EQ-INV-SBI-003]. No equation is evaluated.

            """
            return simulate_summary(self, tuple(theta), rng, ledger)

        return self.prior_sampler, simulator


def simulate_summary(
    contract: SimulatorContract,
    theta: Vector,
    generator: random.Random,
    ledger: WorkLedger | None = None,
) -> Vector:
    """Simulate once and return the validated summary vector.

    Args:
        contract: Simulator contract.
        theta: Parameter vector.
        generator: Stream used for the observation noise.
        ledger: Optional work ledger (calls and failures are counted).

    Returns:
        The finite, non-empty summary tuple.

    Raises:
        SimulationFailure: If the simulator or summary raises
            ``ValueError``/``ArithmeticError`` or returns a non-finite value.
        ValueError: If the summary is empty (a contract violation).

    References:
        [EQ-INV-SBI-003].

    """
    if ledger is not None:
        ledger.simulator_calls += 1
    try:
        data = tuple(float(v) for v in contract.simulator(theta, generator))
        if not all(math.isfinite(v) for v in data) or not data:
            raise SimulationFailure("simulator returned an empty or non-finite output")
        values = (
            data
            if contract.summary is None
            else tuple(float(v) for v in contract.summary(data))
        )
        if not values:
            raise ValueError("summary must be non-empty")
        if not all(math.isfinite(v) for v in values):
            raise SimulationFailure("summary returned a non-finite value")
    except SimulationFailure:
        if ledger is not None:
            ledger.failures += 1
        raise
    except ArithmeticError as error:
        if ledger is not None:
            ledger.failures += 1
        raise SimulationFailure(f"simulation raised {error!r}") from error
    return values


@dataclass(frozen=True, slots=True)
class SimulationDataset:
    """Simulated ``(theta, summary)`` pairs with their failure record.

    Attributes:
        label: Split name.
        parameters: Parameter vectors.
        summaries: Summary vectors.
        attempts: Simulator calls made (including discarded failures).
        failures: Discarded failed calls.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """

    label: str
    parameters: tuple[Vector, ...]
    summaries: tuple[Vector, ...]
    attempts: int
    failures: int

    @property
    def size(self) -> int:
        """Return the number of kept pairs.

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        return len(self.parameters)

    @property
    def failure_rate(self) -> float:
        """Return ``failures / attempts``.

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        return self.failures / self.attempts if self.attempts else 0.0


def draw_dataset(
    contract: SimulatorContract,
    count: int,
    stream: RandomStream,
    *,
    label: str = "dataset",
    failure_policy: str = "raise",
    max_failure_fraction: float = 0.5,
    ledger: WorkLedger | None = None,
) -> SimulationDataset:
    """Draw ``count`` prior-predictive pairs from child streams of ``stream``.

    Pair ``n`` uses ``stream.child("simulation", n)`` for the prior and the
    simulator, the convention of the library's own training routines, so the
    pairs equal the library's training set for the same stream (without
    failures). Under ``failure_policy="discard"`` a failed pair is redrawn on
    ``stream.child("simulation", n, attempt)``; the retained pairs then
    follow the prior-predictive law *conditional on success*, which changes
    the target and is reported through ``failures``.

    Args:
        contract: Simulator contract.
        count: Number of pairs (positive).
        stream: Random stream.
        label: Label stored in the result.
        failure_policy: ``"raise"`` (default) or ``"discard"``.
        max_failure_fraction: Largest tolerated ``failures / attempts`` in
            ``[0, 1)`` under ``"discard"``.
        ledger: Optional work ledger.

    Returns:
        The dataset.

    Raises:
        ValueError: For invalid options, a pair that fails
            ``50`` times, or an exceeded failure fraction.
        SimulationFailure: Under ``"raise"`` at the first failure.

    References:
        [EQ-INV-SBI-003].

    """
    size = require_non_negative_integer(count, name="count")
    if size == 0:
        raise ValueError("count must be positive")
    if failure_policy not in _FAILURE_POLICIES:
        raise ValueError(f"failure_policy must be one of {_FAILURE_POLICIES}")
    fraction = require_finite_real(max_failure_fraction, name="max_failure_fraction")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("max_failure_fraction must lie in [0, 1)")
    prior, simulator = contract.bind(ledger)
    thetas: list[Vector] = []
    summaries: list[Vector] = []
    attempts = 0
    failures = 0
    for index in range(size):
        for attempt in range(_MAX_ATTEMPTS_PER_PAIR):
            child = (
                stream.child("simulation", index)
                if attempt == 0
                else stream.child("simulation", index, attempt)
            )
            rng = child.generator()
            theta = as_vector(prior(rng), name="prior draw")
            attempts += 1
            try:
                summary = simulator(theta, rng)
            except SimulationFailure:
                if failure_policy == "raise":
                    raise
                failures += 1
                continue
            thetas.append(theta)
            summaries.append(summary)
            break
        else:
            raise ValueError(f"pair {index} failed {_MAX_ATTEMPTS_PER_PAIR} times")
    if failures > fraction * attempts:
        raise ValueError(
            f"failure fraction {failures / attempts:.3f} exceeds {fraction:.3f}"
        )
    return SimulationDataset(label, tuple(thetas), tuple(summaries), attempts, failures)


@dataclass(frozen=True, slots=True)
class SimulationSplits:
    """Train, validation and test datasets from separate child streams.

    Attributes:
        train: Training pairs (``stream.child("split", "train")``).
        validation: Validation pairs.
        test: Held-out test pairs.
        train_stream: The stream the library fitters must receive so that they
            simulate exactly ``train``.
        seed: Root seed of the splits.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """

    train: SimulationDataset
    validation: SimulationDataset
    test: SimulationDataset
    train_stream: RandomStream
    seed: int


def make_splits(
    contract: SimulatorContract,
    *,
    train: int,
    validation: int,
    test: int,
    seed: int | RandomStream,
    failure_policy: str = "raise",
    ledger: WorkLedger | None = None,
) -> SimulationSplits:
    """Draw independent train / validation / test sets.

    Args:
        contract: Simulator contract.
        train: Training size (at least two).
        validation: Validation size (positive).
        test: Test size (positive).
        seed: Integer seed or stream (``None`` is not accepted: splits must
            be reproducible).
        failure_policy: Passed to :func:`draw_dataset`.
        ledger: Optional ledger charged with all three draws.

    Returns:
        The three datasets and the training stream.

    Raises:
        TypeError: If ``seed`` is ``None``.
        ValueError: For invalid sizes.

    References:
        [EQ-INV-SBI-003].

    """
    if seed is None:
        raise TypeError("make_splits needs an explicit seed")
    root = resolve_stream(seed)
    if require_non_negative_integer(train, name="train") < 2:
        raise ValueError("train must be at least 2")
    streams = {name: root.child("split", name) for name in ("train", "val", "test")}
    sets = [
        draw_dataset(
            contract,
            size,
            streams[key],
            label=label,
            failure_policy=failure_policy,
            ledger=ledger,
        )
        for label, key, size in (
            ("train", "train", train),
            ("validation", "val", validation),
            ("test", "test", test),
        )
    ]
    return SimulationSplits(sets[0], sets[1], sets[2], streams["train"], root.seed)


# --------------------------------------------------------------------------
# Synthetic likelihood
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticLikelihoodResult:
    """Gaussian synthetic log-likelihood at one parameter (approximate).

    Attributes:
        log_likelihood: Mean over valid replicates of the plug-in Gaussian
            log density of the observed summary; ``nan`` if any replicate had
            a singular covariance.
        log_likelihood_variance: Sample variance of the per-replicate
            estimates (Monte Carlo variance of one estimate), or ``None``
            with fewer than two replicates.
        mean: Simulated summary mean of the first replicate.
        covariance: Its unbiased sample covariance.
        simulations: Simulations per replicate.
        replicates: Independent replicates.
        status: ``"ok"`` or ``"singular_covariance"``.
        approximate: Always ``True``: a Gaussian model of the summaries with
            plug-in moments, not the simulator likelihood.

    References:
        [EQ-INV-SBI-002], [REF-WOOD-2010].

    """

    log_likelihood: float
    log_likelihood_variance: float | None
    mean: Vector
    covariance: Matrix
    simulations: int
    replicates: int
    status: str
    approximate: bool = True


def _gaussian_log_density(
    observed: Vector, mean: Vector, covariance: Matrix
) -> float | None:
    """Return the Gaussian log density or ``None`` for a singular covariance.

    References:
        [EQ-INV-SBI-002].

    """
    try:
        lower = cholesky(covariance)
        log_det = log_det_spd(covariance)
    except ValueError:
        return None
    pivots = [lower[i][i] for i in range(len(lower))]
    if min(pivots) < _PIVOT_RATIO * max(pivots):
        return None
    z = solve_lower(lower, tuple(o - m for o, m in zip(observed, mean, strict=True)))
    quad = math.fsum(v * v for v in z)
    return -0.5 * (quad + log_det + len(observed) * math.log(2.0 * math.pi))


def _moments(summaries: Sequence[Vector]) -> tuple[Vector, Matrix]:
    """Return the sample mean and unbiased sample covariance.

    References:
        [EQ-INV-SBI-002].

    """
    n = len(summaries)
    d = len(summaries[0])
    mean = tuple(math.fsum(s[k] for s in summaries) / n for k in range(d))
    cov = tuple(
        tuple(
            math.fsum((s[i] - mean[i]) * (s[j] - mean[j]) for s in summaries) / (n - 1)
            for j in range(d)
        )
        for i in range(d)
    )
    return mean, cov


def synthetic_log_likelihood(
    contract: SimulatorContract,
    theta: Sequence[float],
    observed_summary: Sequence[float],
    *,
    simulations: int,
    replicates: int = 1,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
) -> SyntheticLikelihoodResult:
    """Return the Gaussian synthetic log-likelihood of Wood (2010).

    For each replicate ``simulations`` summaries are drawn at ``theta`` and
    ``log N(s_obs; mean, S)`` is evaluated with the sample mean and the
    unbiased sample covariance ``S``. The estimate is *approximate*: it is
    the likelihood of a Gaussian model of the summaries with plug-in moments
    (biased as an estimate of the log-likelihood for few simulations), not
    the likelihood of the simulator. The Monte Carlo variance is the sample
    variance across ``replicates`` independent replicates. A covariance that
    is not numerically positive definite (a Cholesky pivot below ``1e-7``
    times the largest one, or fewer simulations than summary dimension plus
    one) is reported with
    ``status="singular_covariance"``; no jitter or pseudo-inverse is used.

    Args:
        contract: Simulator contract.
        theta: Parameter vector.
        observed_summary: Observed summary of the contract's summary width.
        simulations: Simulations per replicate (at least two).
        replicates: Independent replicates (positive).
        seed: Integer seed or stream.
        ledger: Optional work ledger.

    Returns:
        :class:`SyntheticLikelihoodResult`.

    Raises:
        ValueError: For invalid counts or a summary width mismatch.
        SimulationFailure: If a simulation fails.

    Equation:
        [EQ-INV-SBI-002].

    References:
        [REF-WOOD-2010].

    """
    n = require_non_negative_integer(simulations, name="simulations")
    reps = require_non_negative_integer(replicates, name="replicates")
    if n < 2 or reps < 1:
        raise ValueError("simulations must be at least 2 and replicates positive")
    point = as_vector(theta, name="theta")
    target = as_vector(observed_summary, name="observed_summary")
    stream = resolve_stream(seed)
    estimates: list[float] = []
    first: tuple[Vector, Matrix] | None = None
    singular = False
    for r in range(reps):
        sims = []
        for i in range(n):
            rng = stream.child("replicate", r, "simulation", i).generator()
            summary = simulate_summary(contract, point, rng, ledger)
            if len(summary) != len(target):
                raise ValueError("simulated and observed summaries differ in width")
            sims.append(summary)
        moments = _moments(sims)
        if first is None:
            first = moments
        value = (
            None
            if n <= len(target)
            else _gaussian_log_density(target, moments[0], moments[1])
        )
        if value is None:
            singular = True
        else:
            estimates.append(value)
    assert first is not None
    variance = statistics.variance(estimates) if len(estimates) >= 2 else None
    return SyntheticLikelihoodResult(
        log_likelihood=math.nan if singular else math.fsum(estimates) / reps,
        log_likelihood_variance=None if singular else variance,
        mean=first[0],
        covariance=first[1],
        simulations=n,
        replicates=reps,
        status="singular_covariance" if singular else "ok",
    )


@dataclass(frozen=True, slots=True)
class SyntheticLikelihoodGrid:
    """Synthetic-likelihood posterior of a scalar parameter on a grid.

    Attributes:
        grid: Parameter grid.
        density: Normalized density (empty unless ``status == "ok"``).
        mean: Posterior mean (``nan`` unless ``status == "ok"``).
        standard_deviation: Posterior standard deviation (``nan`` likewise).
        mean_log_likelihood_variance: Average Monte Carlo variance of the
            per-point log-likelihood estimate (``None`` without replicates).
        singular_points: Grid indices with a singular covariance.
        status: ``"ok"`` or ``"singular_covariance"``.
        approximate: Always ``True``.

    References:
        [EQ-INV-SBI-002], [REF-WOOD-2010].

    """

    grid: Vector
    density: Vector
    mean: float
    standard_deviation: float
    mean_log_likelihood_variance: float | None
    singular_points: tuple[int, ...]
    status: str
    approximate: bool = True


def synthetic_likelihood_grid(
    contract: SimulatorContract,
    observed_summary: Sequence[float],
    grid: Sequence[float],
    *,
    simulations: int,
    replicates: int = 1,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
) -> SyntheticLikelihoodGrid:
    """Tabulate ``p(theta) L_s(theta)`` on a grid of a scalar parameter.

    ``L_s`` is the Gaussian synthetic likelihood of
    :func:`synthetic_log_likelihood`; the contract must define ``log_prior``.
    Grid point ``i`` uses its own child stream. If any point has a singular
    covariance the whole posterior is reported as singular (no density).

    Args:
        contract: Contract with ``log_prior``.
        observed_summary: Observed summary.
        grid: At least three strictly increasing finite values.
        simulations: Simulations per grid point and replicate.
        replicates: Replicates per grid point.
        seed: Integer seed or stream.
        ledger: Optional ledger.

    Returns:
        :class:`SyntheticLikelihoodGrid`.

    Raises:
        ValueError: If ``log_prior`` is missing, the grid is invalid, or the
            prior vanishes on the whole grid.

    Equation:
        [EQ-INV-SBI-002].

    References:
        [REF-WOOD-2010].

    """
    if contract.log_prior is None:
        raise ValueError("the contract needs log_prior")
    points = as_vector(grid, name="grid")
    if len(points) < 3 or any(b <= a for a, b in zip(points, points[1:], strict=False)):
        raise ValueError("grid needs at least three strictly increasing values")
    stream = resolve_stream(seed)
    logs: list[float] = []
    variances: list[float] = []
    singular: list[int] = []
    for index, point in enumerate(points):
        prior = float(contract.log_prior((point,)))
        if prior == -math.inf:
            logs.append(-math.inf)
            continue
        result = synthetic_log_likelihood(
            contract,
            (point,),
            observed_summary,
            simulations=simulations,
            replicates=replicates,
            seed=stream.child("grid", index),
            ledger=ledger,
        )
        if result.status != "ok":
            singular.append(index)
            logs.append(-math.inf)
            continue
        logs.append(prior + result.log_likelihood)
        if result.log_likelihood_variance is not None:
            variances.append(result.log_likelihood_variance)
    mean_variance = math.fsum(variances) / len(variances) if variances else None
    if singular:
        return SyntheticLikelihoodGrid(
            points, (), math.nan, math.nan, mean_variance, tuple(singular),
            "singular_covariance",
        )  # fmt: skip
    peak = max(logs)
    if peak == -math.inf:
        raise ValueError("the prior vanishes on the whole grid")
    grid_density = GridDensity.from_log(points, logs)
    return SyntheticLikelihoodGrid(
        points,
        grid_density.density,
        grid_density.mean(),
        grid_density.standard_deviation(),
        mean_variance,
        (),
        "ok",
    )


# --------------------------------------------------------------------------
# Tabulated densities and sample metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridDensity:
    """A normalized one-dimensional density tabulated on an increasing grid.

    Attributes:
        grid: Increasing grid.
        density: Density values with trapezoidal integral one.

    References:
        [EQ-INV-SBI-004]. No new equation is introduced.

    """

    grid: Vector
    density: Vector

    @classmethod
    def from_log(cls, grid: Sequence[float], logs: Sequence[float]) -> GridDensity:
        """Normalize unnormalized log values by the trapezoidal rule.

        Args:
            grid: Increasing grid.
            logs: Unnormalized log density (``-inf`` allowed).

        Returns:
            The normalized density.

        Raises:
            ValueError: If every value is ``-inf`` or the mass is zero.

        References:
            [EQ-INV-SBI-004].

        """
        peak = max(logs)
        if peak == -math.inf or not math.isfinite(peak):
            raise ValueError("the density vanishes or is not finite on the grid")
        raw = [math.exp(v - peak) for v in logs]
        mass = math.fsum(
            0.5 * (grid[i + 1] - grid[i]) * (raw[i] + raw[i + 1])
            for i in range(len(grid) - 1)
        )
        if mass <= 0.0:
            raise ValueError("the density has zero mass on the grid")
        return cls(tuple(grid), tuple(v / mass for v in raw))

    def weights(self) -> Vector:
        """Return trapezoidal quadrature weights times the density.

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        g = self.grid
        n = len(g)
        w = []
        for i in range(n):
            left = g[i] - g[i - 1] if i > 0 else 0.0
            right = g[i + 1] - g[i] if i < n - 1 else 0.0
            w.append(0.5 * (left + right) * self.density[i])
        total = math.fsum(w)
        return tuple(v / total for v in w)

    def mean(self) -> float:
        """Return the mean by the quadrature weights.

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        return math.fsum(w * x for w, x in zip(self.weights(), self.grid, strict=True))

    def standard_deviation(self) -> float:
        """Return the standard deviation by the quadrature weights.

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        m = self.mean()
        return math.sqrt(
            math.fsum(
                w * (x - m) ** 2 for w, x in zip(self.weights(), self.grid, strict=True)
            )
        )

    def mass_in(self, low: float, high: float) -> float:
        """Return the quadrature mass in ``[low, high]``.

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        return math.fsum(
            w
            for w, x in zip(self.weights(), self.grid, strict=True)
            if low <= x <= high
        )

    def log_density_at(self, x: float) -> float:
        """Return the log of the linearly interpolated density.

        Args:
            x: Evaluation point.

        Returns:
            ``log`` of the interpolant; ``-inf`` outside the grid or where it
            is zero.

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        g = self.grid
        if x < g[0] or x > g[-1]:
            return -math.inf
        i = min(max(bisect.bisect_right(g, x) - 1, 0), len(g) - 2)
        t = (x - g[i]) / (g[i + 1] - g[i])
        value = (1.0 - t) * self.density[i] + t * self.density[i + 1]
        return math.log(value) if value > 0.0 else -math.inf

    def sample(self, size: int, generator: random.Random) -> list[float]:
        """Draw by inverting the piecewise-linear trapezoidal CDF.

        Args:
            size: Number of draws.
            generator: Random generator.

        Returns:
            ``size`` draws (constant density inside each cell).

        References:
            [EQ-INV-SBI-004]. No new equation is introduced.

        """
        g = self.grid
        cdf = [0.0]
        for i in range(len(g) - 1):
            cdf.append(
                cdf[-1]
                + 0.5 * (g[i + 1] - g[i]) * (self.density[i] + self.density[i + 1])
            )
        total = cdf[-1]
        out = []
        for _ in range(size):
            u = generator.random() * total
            i = min(max(bisect.bisect_right(cdf, u) - 1, 0), len(g) - 2)
            span = cdf[i + 1] - cdf[i]
            t = (u - cdf[i]) / span if span > 0.0 else 0.5
            out.append(g[i] + t * (g[i + 1] - g[i]))
        return out


def crps_ensemble_value(draws: Sequence[float], truth: float) -> float:
    """Return the ensemble CRPS ``E|X - y| - E|X - X'| / 2``.

    Uses the sorted-sample identity ``sum_ij |x_i - x_j| = 2 sum_i (2i - m + 1)
    x_(i)`` (V-statistic, including ``i = j``).

    Args:
        draws: Posterior draws (non-empty, finite).
        truth: Realized value.

    Returns:
        The score (lower is better; it is a proper scoring rule).

    Raises:
        ValueError: If ``draws`` is empty or not finite.

    Equation:
        [EQ-INV-SBI-003].

    References:
        [REF-TALTS-ETAL-2018]. CRPS itself is a standard proper score.

    """
    xs = sorted(require_finite_real(v, name="draw") for v in draws)
    if not xs:
        raise ValueError("draws must be non-empty")
    y = require_finite_real(truth, name="truth")
    m = len(xs)
    first = math.fsum(abs(x - y) for x in xs) / m
    pair = math.fsum((2 * i - m + 1) * x for i, x in enumerate(xs))
    return first - pair / (m * m)


def _covered(draws: Sequence[float], truth: float, level: float) -> bool:
    """Return whether ``truth`` lies in the central ``level`` interval.

    References:
        [EQ-INV-SBI-003].

    """
    xs = sorted(draws)
    m = len(xs)
    alpha = 0.5 * (1.0 - level)
    low = xs[min(m - 1, math.floor(alpha * m))]
    high = xs[max(0, min(m - 1, math.ceil((1.0 - alpha) * m) - 1))]
    return low <= truth <= high


# --------------------------------------------------------------------------
# Training wrappers
# --------------------------------------------------------------------------


class ProposalTrainingRefused(ValueError):
    """Raised when proposal-based (sequential) training is requested.

    The training routines here learn from prior-predictive pairs only. A
    network trained on pairs ``theta ~ q(theta)`` with a proposal ``q``
    different from the prior targets the *proposal posterior*
    ``q(theta | y) ∝ p(theta | y) q(theta) / p(theta)``, not ``p(theta | y)``
    [EQ-INV-SBI-013]; correcting that shift needs a method-specific
    validated procedure that this module does not provide, so the request is
    refused rather than silently run with the shift left in.

    References:
        [EQ-INV-SBI-013], [REF-PAPAMAKARIOS-MURRAY-2016].

    """


def check_training_policy(policy: str) -> str:
    """Accept only the prior-simulation training policy.

    The roadmap's permitted option for sequential inference is to restrict
    training to prior simulations. ``"prior"`` is the only accepted value;
    proposal-based names (``"proposal"``, ``"sequential"``, ``"snpe"``,
    ``"truncated_proposal"``) raise :class:`ProposalTrainingRefused`.

    Args:
        policy: Requested policy.

    Returns:
        ``"prior"``.

    Raises:
        ProposalTrainingRefused: For a proposal-based policy.
        ValueError: For an unknown policy name.
        TypeError: If ``policy`` is not a string.

    Equation:
        [EQ-INV-SBI-013].

    References:
        [REF-PAPAMAKARIOS-MURRAY-2016].

    """
    if not isinstance(policy, str):
        raise TypeError("training_policy must be a string")
    if policy == "prior":
        return policy
    if policy in _PROPOSAL_POLICIES:
        raise ProposalTrainingRefused(
            f"training_policy {policy!r} refused: only prior simulations are "
            "supported; proposal-based training targets the proposal posterior "
            "and its correction is not implemented (no silent proposal shift)"
        )
    raise ValueError("training_policy must be 'prior' (the only supported policy)")


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A library NPE/NRE fit with its provenance and status.

    Attributes:
        kind: ``"npe"`` or ``"nre"``.
        model: The library object (``None`` when training failed).
        status: ``"ok"``, ``"loss_not_decreased"``, ``"nonfinite_loss"`` or
            ``"failed"``.
        message: Error text when ``status == "failed"``.
        simulations: Training simulations.
        iterations: Adam iterations.
        hidden_layers: Hidden widths.
        batch_size: Mini-batch size.
        learning_rate: Adam step size.
        seed: Seed of the training stream.
        initial_loss: Mean loss over the first tenth of the iterations.
        final_loss: Mean loss over the last tenth.
        train_seconds: Wall-clock seconds (informational, not reproducible).
        approximate: Always ``True``.
        training_policy: Always ``"prior"`` (see :func:`check_training_policy`).

    References:
        [EQ-INF-NPE-001], [EQ-GEN-NRE-001], [EQ-INV-SBI-013],
        [REF-PAPAMAKARIOS-MURRAY-2016], [REF-HERMANS-BEGY-LOUPPE-2020].

    """

    kind: str
    model: object | None
    status: str
    message: str
    simulations: int
    iterations: int
    hidden_layers: tuple[int, ...]
    batch_size: int
    learning_rate: float
    seed: int
    initial_loss: float | None
    final_loss: float | None
    train_seconds: float
    approximate: bool = True
    training_policy: str = "prior"


def _loss_windows(losses: Sequence[float]) -> tuple[float, float]:
    """Return the mean of the first and of the last tenth of ``losses``.

    References:
        [EQ-INF-NPE-001]. No new equation is introduced.

    """
    k = max(1, len(losses) // 10)
    return statistics.fmean(losses[:k]), statistics.fmean(losses[-k:])


def _train(
    kind: str,
    contract: SimulatorContract,
    simulations: int,
    *,
    hidden_layers: Sequence[int],
    iterations: int,
    batch_size: int,
    learning_rate: float,
    seed: int | RandomStream,
    ledger: WorkLedger | None,
    training_policy: str = "prior",
    sigma_min: float = 0.0,
) -> TrainedModel:
    """Fit an NPE, NRE or flow-matching posterior and classify the outcome.

    References:
        [EQ-INF-NPE-001], [EQ-GEN-NRE-001], [EQ-INV-SBI-012], [EQ-INV-SBI-013].

    """
    check_training_policy(training_policy)
    stream = resolve_stream(seed)
    prior, simulator = contract.bind(ledger)
    widths = tuple(hidden_layers)
    started = time.perf_counter()
    model: object | None = None
    status = "ok"
    message = ""
    initial: float | None = None
    final: float | None = None
    try:
        if kind == "npe":
            fitted = fit_amortized_gaussian_posterior(
                prior,
                simulator,
                simulations,
                hidden_layers=widths,
                iterations=iterations,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=stream,
            )
            initial, final = _loss_windows(fitted.losses)
            model = fitted
        elif kind == "fm":
            flow = fit_flow_matching_posterior(
                prior,
                simulator,
                simulations,
                hidden_layers=widths,
                sigma_min=sigma_min,
                iterations=iterations,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=stream,
            )
            initial, final = flow.initial_loss, flow.final_loss
            model = flow
        else:
            ratio = fit_ratio_estimator(
                prior,
                simulator,
                simulations,
                hidden_layers=widths,
                iterations=iterations,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=stream,
            )
            initial, final = ratio.initial_loss, ratio.final_loss
            model = ratio
    except SimulationFailure:
        raise
    except (ValueError, ArithmeticError) as error:
        status, message, model = "failed", str(error), None
    if model is not None:
        assert initial is not None and final is not None
        if not (math.isfinite(initial) and math.isfinite(final)):
            status = "nonfinite_loss"
        elif final >= initial:
            status = "loss_not_decreased"
    return TrainedModel(
        kind,
        model,
        status,
        message,
        simulations,
        iterations,
        widths,
        batch_size,
        learning_rate,
        stream.seed,
        initial,
        final,
        time.perf_counter() - started,
    )


def train_npe(
    contract: SimulatorContract,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (16,),
    iterations: int = 500,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
    training_policy: str = "prior",
) -> TrainedModel:
    """Train the library's diagonal-Gaussian NPE and record its status.

    The family is a diagonal Gaussian: it cannot represent a multimodal
    posterior, whatever the training budget. Training exceptions
    (``ValueError``/``ArithmeticError`` other than :class:`SimulationFailure`)
    are returned as ``status="failed"``.

    Args:
        contract: Simulator contract.
        simulations: Training pairs.
        hidden_layers: Hidden widths.
        iterations: Adam iterations.
        batch_size: Mini-batch size.
        learning_rate: Adam step size.
        seed: Integer seed or training stream (e.g. ``splits.train_stream``).
        ledger: Optional work ledger.
        training_policy: Must be ``"prior"``; proposal-based training is
            refused (:func:`check_training_policy`).

    Returns:
        :class:`TrainedModel` whose ``model`` is an
        :class:`~probstat.inference.AmortizedGaussianPosterior` when
        training succeeded.

    Raises:
        SimulationFailure: If a training simulation fails.
        ProposalTrainingRefused: For a proposal-based ``training_policy``.

    Equation:
        [EQ-INF-NPE-001].

    References:
        [REF-PAPAMAKARIOS-MURRAY-2016].

    """
    return _train(
        "npe",
        contract,
        simulations,
        hidden_layers=hidden_layers,
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        ledger=ledger,
        training_policy=training_policy,
    )


def train_nre(
    contract: SimulatorContract,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (16, 16),
    iterations: int = 500,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
    training_policy: str = "prior",
) -> TrainedModel:
    """Train the library's ratio estimator and record its status.

    Args:
        contract: Simulator contract.
        simulations: Training pairs.
        hidden_layers: Hidden widths.
        iterations: Adam iterations.
        batch_size: Mini-batch size (at least two).
        learning_rate: Adam step size.
        seed: Integer seed or training stream.
        ledger: Optional work ledger.
        training_policy: Must be ``"prior"``; proposal-based training is
            refused (:func:`check_training_policy`).

    Returns:
        :class:`TrainedModel` whose ``model`` is a
        :class:`~probstat.generative.RatioEstimator` when training succeeded.

    Raises:
        SimulationFailure: If a training simulation fails.
        ProposalTrainingRefused: For a proposal-based ``training_policy``.

    Equation:
        [EQ-GEN-NRE-001].

    References:
        [REF-HERMANS-BEGY-LOUPPE-2020].

    """
    return _train(
        "nre",
        contract,
        simulations,
        hidden_layers=hidden_layers,
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        ledger=ledger,
        training_policy=training_policy,
    )


def train_flow_matching(
    contract: SimulatorContract,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (16,),
    iterations: int = 150,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    sigma_min: float = 0.0,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
    training_policy: str = "prior",
) -> TrainedModel:
    """Train the library's flow-matching posterior and record its status.

    Wraps :func:`probstat.generative.fit_flow_matching_posterior`. With the
    training stream of :func:`make_splits` the fitter simulates exactly the
    train split (it uses ``stream.child("simulation", n)``), so it sees the
    same pairs as :func:`train_nre`. The model generates samples by
    integrating a learned velocity field; this module uses it as a
    sample-only method and applies no evidence or Metropolis-Hastings
    correction to it.

    Args:
        contract: Simulator contract.
        simulations: Training pairs.
        hidden_layers: Hidden widths of the velocity network.
        iterations: Adam iterations.
        batch_size: Mini-batch size.
        learning_rate: Adam step size.
        sigma_min: Path width in ``[0, 1)`` (standardized units).
        seed: Integer seed or training stream.
        ledger: Optional work ledger.
        training_policy: Must be ``"prior"``.

    Returns:
        :class:`TrainedModel` whose ``model`` is a
        :class:`~probstat.generative.FlowMatchingPosterior` when training
        succeeded.

    Raises:
        SimulationFailure: If a training simulation fails.
        ProposalTrainingRefused: For a proposal-based ``training_policy``.

    Equation:
        [EQ-INV-SBI-012].

    References:
        [REF-LIPMAN-ETAL-2023], [REF-WILDBERGER-ETAL-2023].

    """
    return _train(
        "fm",
        contract,
        simulations,
        hidden_layers=hidden_layers,
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        ledger=ledger,
        training_policy=training_policy,
        sigma_min=sigma_min,
    )


# --------------------------------------------------------------------------
# ABC wrappers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ABCOutcome:
    """Result of an ABC run with explicit failure status.

    Attributes:
        status: ``"ok"``, ``"zero_acceptance"`` or ``"insufficient_acceptance"``.
        parameters: Accepted parameter vectors (empty unless ``"ok"``).
        weights: Normalized weights (equal for rejection).
        tolerance: Final tolerance (``nan`` when not ``"ok"``).
        simulations: Simulator calls charged.
        accepted: Number of candidates that met the tolerance.
        result: The library :class:`~probstat.inference.ABCResult`, if any.
        approximate: Always ``True``: the ABC target depends on the summary
            and the tolerance.

    References:
        [EQ-INV-SBI-001], [REF-PRITCHARD-ETAL-1999].

    """

    status: str
    parameters: tuple[Vector, ...]
    weights: tuple[float, ...]
    tolerance: float
    simulations: int
    accepted: int
    result: ABCResult | None
    approximate: bool = True


def run_abc_rejection(
    contract: SimulatorContract,
    observed_summary: Sequence[float],
    *,
    draws: int,
    tolerance: float | None = None,
    quantile: float | None = None,
    max_simulations: int | None = None,
    distance: Distance | None = None,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
) -> ABCOutcome:
    """Run :func:`probstat.inference.abc_rejection` and report zero acceptance.

    In tolerance mode the library raises when the budget is exhausted; this
    wrapper counts acceptances and returns ``status="zero_acceptance"`` or
    ``"insufficient_acceptance"`` instead. Invalid summaries or distances
    still raise ``ValueError`` from the library.

    Args:
        contract: Simulator contract.
        observed_summary: Observed summary.
        draws: Accepted draws wanted.
        tolerance: Fixed tolerance (exclusive with ``quantile``).
        quantile: Accepted fraction in ``(0, 1]``.
        max_simulations: Tolerance-mode budget.
        distance: Distance (Euclidean when ``None``).
        seed: Integer seed or stream.
        ledger: Optional work ledger.

    Returns:
        :class:`ABCOutcome`.

    Raises:
        ValueError: For invalid options, summaries, or distances.
        SimulationFailure: If a simulation fails.

    Equation:
        [EQ-INV-SBI-001].

    References:
        [REF-PRITCHARD-ETAL-1999].

    """
    prior, simulator = contract.bind(ledger)
    local = WorkLedger()
    counted = {"accepted": 0}

    def measured(simulated: Vector, observed: Vector) -> float:
        """Return the distance and count candidates within the tolerance.

        References:
            [EQ-INV-SBI-001]. No new equation is introduced.

        """
        base = distance if distance is not None else _euclidean
        value = float(base(simulated, observed))
        if math.isnan(value) or value < 0.0:
            raise ValueError("distance must return a non-negative number")
        if tolerance is not None and value <= tolerance:
            counted["accepted"] += 1
        return value

    def counting(theta: Sequence[float], rng: random.Random) -> Vector:
        """Simulate through the contract and count calls locally.

        References:
            [EQ-INV-SBI-001]. No new equation is introduced.

        """
        local.simulator_calls += 1
        return simulator(theta, rng)

    try:
        result = abc_rejection(
            counting,
            prior,
            observed_summary,
            measured,
            draws=draws,
            tolerance=tolerance,
            quantile=quantile,
            max_simulations=max_simulations,
            seed=seed,
        )
    except ValueError as error:
        if tolerance is None or isinstance(error, SimulationFailure):
            raise
        accepted = counted["accepted"]
        budget = 1000 * draws if max_simulations is None else max_simulations
        if accepted >= draws or local.simulator_calls < budget:
            raise
        status = "zero_acceptance" if accepted == 0 else "insufficient_acceptance"
        return ABCOutcome(
            status, (), (), math.nan, local.simulator_calls, accepted, None
        )
    return ABCOutcome(
        "ok",
        result.parameters,
        result.weights,
        result.tolerance,
        result.simulations,
        len(result.parameters),
        result,
    )


def run_abc_smc(
    contract: SimulatorContract,
    observed_summary: Sequence[float],
    *,
    draws: int,
    generations: int = 4,
    quantile: float = 0.5,
    final_tolerance: float | None = None,
    max_simulations: int | None = None,
    distance: Distance | None = None,
    seed: int | RandomStream,
    ledger: WorkLedger | None = None,
) -> ABCOutcome:
    """Run :func:`probstat.inference.abc_smc` through the simulator contract.

    The importance weights and kernel normalization are those of the library
    (checked by its own tests); this wrapper adds the contract's failure and
    work accounting. The contract must define ``log_prior``.

    Args:
        contract: Simulator contract with ``log_prior``.
        observed_summary: Observed summary.
        draws: Population size.
        generations: Maximum generations.
        quantile: Schedule quantile in ``(0, 1)``.
        final_tolerance: Optional stopping tolerance.
        max_simulations: Total budget.
        distance: Distance (Euclidean when ``None``).
        seed: Integer seed or stream.
        ledger: Optional work ledger.

    Returns:
        :class:`ABCOutcome` with weighted final population.

    Raises:
        ValueError: If ``log_prior`` is missing or the library rejects an
            option.

    Equation:
        [EQ-INV-SBI-001].

    References:
        [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

    """
    if contract.log_prior is None:
        raise ValueError("the contract needs log_prior for ABC-SMC")
    prior, simulator = contract.bind(ledger)
    result = abc_smc(
        simulator,
        prior,
        contract.log_prior,
        observed_summary,
        distance,
        draws=draws,
        generations=generations,
        quantile=quantile,
        final_tolerance=final_tolerance,
        max_simulations=max_simulations,
        seed=seed,
    )
    return ABCOutcome(
        "ok",
        result.parameters,
        result.weights,
        result.tolerance,
        result.simulations,
        len(result.parameters),
        result,
    )


def _euclidean(simulated: Vector, observed: Vector) -> float:
    """Return the Euclidean distance of two summaries.

    References:
        [REF-PRITCHARD-ETAL-1999]. No equation is evaluated.

    """
    if len(simulated) != len(observed):
        raise ValueError("summaries must have the same length")
    return math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(simulated, observed, strict=True))
    )


# --------------------------------------------------------------------------
# (a) Linear-Gaussian tolerance study
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearGaussianCase:
    """``theta ~ N(0, tau^2)``, ``y_i = theta + N(0, sigma^2)``, ``i <= n``.

    Attributes:
        n_obs: Number of observations ``n``.
        noise_sd: ``sigma``.
        prior_sd: ``tau``.
        observed: The fixed observed data.

    References:
        [EQ-INV-SBI-001]. Conjugate normal model.

    """

    n_obs: int
    noise_sd: float
    prior_sd: float
    observed: Vector

    def exact_full(self) -> tuple[float, float]:
        """Return the exact posterior ``(mean, variance)`` given all data.

        References:
            [EQ-INV-SBI-001].

        """
        precision = 1.0 / self.prior_sd**2 + self.n_obs / self.noise_sd**2
        mean = math.fsum(self.observed) / self.noise_sd**2 / precision
        return mean, 1.0 / precision

    def exact_for_summary(self, summary: str) -> tuple[float, float]:
        """Return the exact posterior given only the named summary.

        ``"mean"`` is sufficient (equals :meth:`exact_full`); ``"first"``
        (``y_1`` alone) is not, and its posterior is the limit of ABC as the
        tolerance tends to zero.

        Args:
            summary: ``"mean"`` or ``"first"``.

        Returns:
            ``(mean, variance)``.

        Raises:
            ValueError: For an unknown summary.

        References:
            [EQ-INV-SBI-001].

        """
        if summary == "mean":
            return self.exact_full()
        if summary == "first":
            precision = 1.0 / self.prior_sd**2 + 1.0 / self.noise_sd**2
            return self.observed[0] / self.noise_sd**2 / precision, 1.0 / precision
        raise ValueError("summary must be 'mean' or 'first'")

    def observed_summary(self, summary: str) -> Vector:
        """Return the observed value of the named summary.

        References:
            [EQ-INV-SBI-001]. No new equation is introduced.

        """
        if summary == "mean":
            return (math.fsum(self.observed) / self.n_obs,)
        if summary == "first":
            return (self.observed[0],)
        raise ValueError("summary must be 'mean' or 'first'")


def linear_gaussian_case(
    *,
    n_obs: int = 5,
    noise_sd: float = 1.0,
    prior_sd: float = 1.0,
    theta_true: float = 0.5,
    data_seed: int = 2026,
) -> LinearGaussianCase:
    """Generate the fixed observed data of the linear-Gaussian case.

    Args:
        n_obs: Number of observations (at least two).
        noise_sd: Positive noise standard deviation.
        prior_sd: Positive prior standard deviation.
        theta_true: Generating parameter.
        data_seed: Seed of the observed data.

    Returns:
        :class:`LinearGaussianCase`.

    Raises:
        ValueError: For invalid arguments.

    References:
        [EQ-INV-SBI-001].

    """
    n = require_non_negative_integer(n_obs, name="n_obs")
    sigma = require_finite_real(noise_sd, name="noise_sd")
    tau = require_finite_real(prior_sd, name="prior_sd")
    truth = require_finite_real(theta_true, name="theta_true")
    if n < 2 or sigma <= 0.0 or tau <= 0.0:
        raise ValueError("need n_obs >= 2 and positive noise_sd and prior_sd")
    rng = RandomStream(data_seed).child("observed").generator()
    data = tuple(truth + sigma * rng.gauss(0.0, 1.0) for _ in range(n))
    return LinearGaussianCase(n, sigma, tau, data)


def linear_gaussian_contract(
    case: LinearGaussianCase, summary: str
) -> SimulatorContract:
    """Return the contract of the linear-Gaussian case for a named summary.

    Args:
        case: The case.
        summary: ``"mean"`` (sufficient), ``"first"`` (insufficient) or
            ``"full"`` (all data).

    Returns:
        A contract with a Gaussian ``log_prior``.

    Raises:
        ValueError: For an unknown summary.

    References:
        [EQ-INV-SBI-001].

    """
    if summary not in ("mean", "first", "full"):
        raise ValueError("summary must be 'mean', 'first' or 'full'")
    n, sigma, tau = case.n_obs, case.noise_sd, case.prior_sd

    def prior(rng: random.Random) -> Vector:
        """Draw ``theta ~ N(0, tau^2)``.

        References:
            [EQ-INV-SBI-001].

        """
        return (rng.gauss(0.0, tau),)

    def simulator(theta: Vector, rng: random.Random) -> Vector:
        """Draw ``n`` observations ``N(theta, sigma^2)``.

        References:
            [EQ-INV-SBI-001].

        """
        return tuple(theta[0] + sigma * rng.gauss(0.0, 1.0) for _ in range(n))

    def log_prior(theta: Sequence[float]) -> float:
        """Return the log prior up to a constant.

        References:
            [EQ-INV-SBI-001].

        """
        return -0.5 * (theta[0] / tau) ** 2

    maps: dict[str, SummaryMap | None] = {
        "mean": lambda y: (math.fsum(y) / len(y),),
        "first": lambda y: (y[0],),
        "full": None,
    }
    return SimulatorContract(prior, simulator, maps[summary], log_prior, summary)


@dataclass(frozen=True, slots=True)
class ToleranceRow:
    """One ABC run of the tolerance study.

    Attributes:
        summary: ``"mean"`` (sufficient) or ``"first"`` (insufficient).
        tolerance: Tolerance ``epsilon``.
        seed: Predeclared seed.
        simulations: Simulator calls.
        acceptance_rate: Accepted fraction.
        mean: Sample mean of the accepted draws.
        standard_deviation: Sample standard deviation.
        w1_to_full: Wasserstein-1 distance to the exact full-data posterior.
        w1_to_summary: Wasserstein-1 distance to the exact posterior given
            only the summary (the ``epsilon -> 0`` limit).

    References:
        [EQ-INV-SBI-001].

    """

    summary: str
    tolerance: float
    seed: int
    simulations: int
    acceptance_rate: float
    mean: float
    standard_deviation: float
    w1_to_full: float
    w1_to_summary: float


@dataclass(frozen=True, slots=True)
class ToleranceStudy:
    """Rows of the tolerance study with exact references.

    Attributes:
        rows: One row per (summary, tolerance, seed).
        case: The linear-Gaussian case.
        draws: Accepted draws per run.
        seeds: Predeclared seeds.
        tolerances: Predeclared tolerances (decreasing).
        reference_full: Exact ``(mean, variance)`` given all data.
        reference_first: Exact ``(mean, variance)`` given ``y_1`` only.

    References:
        [EQ-INV-SBI-001].

    """

    rows: tuple[ToleranceRow, ...]
    case: LinearGaussianCase
    draws: int
    seeds: tuple[int, ...]
    tolerances: tuple[float, ...]
    reference_full: tuple[float, float]
    reference_first: tuple[float, float]

    def aggregate(self) -> tuple[dict[str, float | int | str], ...]:
        """Return per (summary, tolerance) means over seeds.

        Each record has ``summary``, ``tolerance``, ``w1_to_full``,
        ``w1_to_summary``, ``mean_simulations``, ``mean_sd`` (all averaged
        over seeds) and ``seeds_improved``: the number of seeds whose
        ``w1_to_summary`` at this tolerance is below that at the previous
        (larger) tolerance. Improvement is a statement about seeds, not a
        guarantee for any single sample.

        References:
            [EQ-INV-SBI-001].

        """
        out: list[dict[str, float | int | str]] = []
        for summary in sorted({r.summary for r in self.rows}):
            previous: dict[int, float] = {}
            for eps in self.tolerances:
                group = [
                    r for r in self.rows if r.summary == summary and r.tolerance == eps
                ]
                improved = sum(
                    1 for r in group if r.seed in previous
                    and r.w1_to_summary < previous[r.seed]
                )  # fmt: skip
                out.append(
                    {
                        "summary": summary,
                        "tolerance": eps,
                        "w1_to_full": statistics.fmean(r.w1_to_full for r in group),
                        "w1_to_summary": statistics.fmean(
                            r.w1_to_summary for r in group
                        ),
                        "mean_simulations": statistics.fmean(
                            r.simulations for r in group
                        ),
                        "mean_sd": statistics.fmean(
                            r.standard_deviation for r in group
                        ),
                        "seeds_improved": improved if previous else -1,
                    }
                )
                previous = {r.seed: r.w1_to_summary for r in group}
        return tuple(out)


def _w1_to_normal(samples: Sequence[float], mean: float, variance: float) -> float:
    """Return ``W_1`` between samples and ``N(mean, variance)`` (1000 atoms).

    References:
        [EQ-INV-SBI-001].

    """
    dist = statistics.NormalDist(mean, math.sqrt(variance))
    m = 1000
    atoms = [dist.inv_cdf((i + 0.5) / m) for i in range(m)]
    return empirical_wasserstein_1d(samples, atoms, p=1.0)


DEFAULT_TOLERANCES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


def abc_tolerance_study(
    *,
    tolerances: Sequence[float] = DEFAULT_TOLERANCES,
    seeds: Sequence[int] = PREDECLARED_SEEDS,
    draws: int = 100,
    summaries: Sequence[str] = ("mean", "first"),
    case: LinearGaussianCase | None = None,
) -> ToleranceStudy:
    """Run ABC rejection over tolerances and seeds against the exact posterior.

    For each seed the same child stream is used at every tolerance (common
    random numbers: the candidate sequence is identical, only the acceptance
    threshold changes). With the sufficient summary ``mean`` the ABC target
    tends to the exact full-data posterior as ``epsilon -> 0``; with the
    insufficient summary ``first`` it tends to the (wider) posterior given
    ``y_1`` alone, so ``w1_to_full`` stays bounded away from zero however
    small the tolerance. Per-seed errors are noisy and need not decrease
    monotonically; the trend is read from :meth:`ToleranceStudy.aggregate`.

    Args:
        tolerances: Strictly decreasing positive tolerances.
        seeds: Seeds (default :data:`PREDECLARED_SEEDS`).
        draws: Accepted draws per run (at least 10).
        summaries: Subset of ``("mean", "first")``.
        case: Optional :class:`LinearGaussianCase` (default fixed case).

    Returns:
        :class:`ToleranceStudy`.

    Raises:
        ValueError: For invalid settings.

    Equation:
        [EQ-INV-SBI-001].

    References:
        [REF-PRITCHARD-ETAL-1999], [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

    """
    eps_list = tuple(require_finite_real(e, name="tolerance") for e in tolerances)
    if not eps_list or any(e <= 0 for e in eps_list):
        raise ValueError("tolerances must be positive")
    if any(b >= a for a, b in zip(eps_list, eps_list[1:], strict=False)):
        raise ValueError("tolerances must be strictly decreasing")
    if require_non_negative_integer(draws, name="draws") < 10:
        raise ValueError("draws must be at least 10")
    if not seeds or any(s not in range(0, 2**62) for s in seeds):
        raise ValueError("seeds must be non-empty non-negative integers")
    if not set(summaries) <= {"mean", "first"} or not summaries:
        raise ValueError("summaries must be a non-empty subset of mean/first")
    problem = case if case is not None else linear_gaussian_case()
    full = problem.exact_full()
    first = problem.exact_for_summary("first")
    rows: list[ToleranceRow] = []
    for summary in summaries:
        contract = linear_gaussian_contract(problem, summary)
        observed = problem.observed_summary(summary)
        for eps in eps_list:
            for seed in seeds:
                ledger = WorkLedger()
                outcome = run_abc_rejection(
                    contract,
                    observed,
                    draws=draws,
                    tolerance=eps,
                    seed=RandomStream(seed).child("study", summary),
                    ledger=ledger,
                )
                if outcome.status != "ok":
                    raise ValueError(
                        f"ABC {outcome.status} at tolerance {eps} (seed {seed})"
                    )
                xs = [p[0] for p in outcome.parameters]
                rows.append(
                    ToleranceRow(
                        summary,
                        eps,
                        seed,
                        outcome.simulations,
                        outcome.accepted / outcome.simulations,
                        statistics.fmean(xs),
                        statistics.stdev(xs),
                        _w1_to_normal(xs, *full),
                        _w1_to_normal(xs, *problem.exact_for_summary(summary)),
                    )
                )
    return ToleranceStudy(
        tuple(rows),
        problem,
        draws,
        tuple(seeds),
        eps_list,
        full,
        first,
    )


# --------------------------------------------------------------------------
# (b) Bimodal x^2 comparison
# --------------------------------------------------------------------------


def bimodal_contract(
    noise_sd: float = 0.25, half_width: float = 3.0
) -> SimulatorContract:
    """Return the contract ``theta ~ U(-h, h)``, ``y = theta^2 + N(0, s^2)``.

    For ``y > 0`` the exact posterior has two modes near ``+-sqrt(y)``.

    Args:
        noise_sd: Positive noise standard deviation ``s``.
        half_width: Positive half-width ``h`` of the uniform prior.

    Returns:
        The contract (summary = the scalar ``y``).

    Raises:
        ValueError: For non-positive arguments.

    Equation:
        [EQ-INV-SBI-004].

    References:
        [REF-TARANTOLA-2005].

    """
    s = require_finite_real(noise_sd, name="noise_sd")
    h = require_finite_real(half_width, name="half_width")
    if s <= 0.0 or h <= 0.0:
        raise ValueError("noise_sd and half_width must be positive")

    def prior(rng: random.Random) -> Vector:
        """Draw ``theta ~ U(-h, h)``.

        References:
            [EQ-INV-SBI-004].

        """
        return (rng.uniform(-h, h),)

    def simulator(theta: Vector, rng: random.Random) -> Vector:
        """Return ``theta^2`` plus Gaussian noise.

        References:
            [EQ-INV-SBI-004].

        """
        return (theta[0] ** 2 + s * rng.gauss(0.0, 1.0),)

    def log_prior(theta: Sequence[float]) -> float:
        """Return ``0`` inside ``[-h, h]`` and ``-inf`` outside.

        References:
            [EQ-INV-SBI-004].

        """
        return 0.0 if abs(theta[0]) <= h else -math.inf

    return SimulatorContract(prior, simulator, None, log_prior, "bimodal-square")


def bimodal_exact_posterior(
    y: float,
    *,
    noise_sd: float = 0.25,
    half_width: float = 3.0,
    points: int = 2001,
) -> GridDensity:
    """Return the quadrature posterior ``p(theta | y) ∝ N(y; theta^2, s^2)``.

    Args:
        y: Observation.
        noise_sd: Noise standard deviation.
        half_width: Prior half-width.
        points: Grid points on ``[-h, h]`` (at least 101).

    Returns:
        The posterior tabulated on a uniform grid (trapezoidal
        normalization; discretization error ``O(h^2)`` for smooth modes).

    Raises:
        ValueError: For invalid arguments.

    Equation:
        [EQ-INV-SBI-004].

    References:
        [REF-TARANTOLA-2005].

    """
    value = require_finite_real(y, name="y")
    s = require_finite_real(noise_sd, name="noise_sd")
    h = require_finite_real(half_width, name="half_width")
    n = require_non_negative_integer(points, name="points")
    if s <= 0.0 or h <= 0.0 or n < 101:
        raise ValueError("need positive noise_sd and half_width and points >= 101")
    grid = tuple(-h + 2.0 * h * i / (n - 1) for i in range(n))
    logs = [-0.5 * ((value - t * t) / s) ** 2 for t in grid]
    return GridDensity.from_log(grid, logs)


@dataclass(frozen=True, slots=True)
class BimodalConfig:
    """Predeclared budgets and seeds of :func:`run_bimodal_comparison`.

    Attributes:
        seed: Root seed (all splits and methods derive child streams).
        noise_sd: Observation noise.
        half_width: Prior half-width.
        n_train: Training simulations of the learned methods.
        n_validation: Validation pairs.
        n_test: Held-out test pairs.
        n_sbc: SBC replicates.
        sbc_draws: Posterior draws per SBC replicate.
        eval_draws: Posterior draws per test observation.
        abc_simulations: ABC simulations per posterior (quantile 0.1 kept).
        npe_hidden: NPE hidden widths.
        npe_iterations: NPE Adam iterations.
        nre_hidden: NRE hidden widths.
        nre_iterations: NRE Adam iterations.
        nre_grid_points: Grid points of the NRE posterior.
        headline_y: Headline observation (modes at ``+-sqrt(y)``).
        mode_radius: Radius of the mode windows.
        min_y_modes: Smallest ``y`` entering the mode-occupation metrics.
        ood_y: Out-of-distribution observation (outside the training range).
        ood_quantile: Training-score quantile of the OOD threshold.
        include_flow_matching: Add a flow-matching row ``"fm"`` (sample-only;
            default ``False`` so existing tables are unchanged).
        fm_hidden: Flow-matching velocity-network hidden widths.
        fm_iterations: Flow-matching Adam iterations.
        fm_steps: ODE steps per flow-matching draw.
        fm_method: ODE method of the draws (``"euler"``, ``"heun"``, ``"rk4"``).

    References:
        [EQ-INV-SBI-003], [EQ-INV-SBI-004], [EQ-INV-SBI-012].

    """

    seed: int = 20260101
    noise_sd: float = 0.25
    half_width: float = 3.0
    n_train: int = 200
    n_validation: int = 20
    n_test: int = 16
    n_sbc: int = 40
    sbc_draws: int = 19
    eval_draws: int = 100
    abc_simulations: int = 300
    npe_hidden: tuple[int, ...] = (16,)
    npe_iterations: int = 80
    nre_hidden: tuple[int, ...] = (16, 16)
    nre_iterations: int = 100
    nre_grid_points: int = 61
    headline_y: float = 1.0
    mode_radius: float = 0.3
    min_y_modes: float = 0.5
    ood_y: float = 12.0
    ood_quantile: float = 0.99
    include_flow_matching: bool = False
    fm_hidden: tuple[int, ...] = (16,)
    fm_iterations: int = 150
    fm_steps: int = 8
    fm_method: str = "heun"

    def __post_init__(self) -> None:
        """Validate sizes.

        Raises:
            ValueError: If a count is too small or a scale is not positive.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        for name in (
            "n_train", "n_validation", "n_test", "n_sbc", "sbc_draws",
            "eval_draws", "abc_simulations", "npe_iterations",
            "nre_iterations", "nre_grid_points",
        ):  # fmt: skip
            require_non_negative_integer(getattr(self, name), name=name)
        if self.n_train < 2 or self.n_test < 1 or self.n_validation < 1:
            raise ValueError("need n_train >= 2 and positive validation/test sizes")
        if self.n_sbc < 2 or self.sbc_draws < 1 or self.eval_draws < 2:
            raise ValueError("need n_sbc >= 2, sbc_draws >= 1, eval_draws >= 2")
        if self.abc_simulations < 20 or self.nre_grid_points < 3:
            raise ValueError("need abc_simulations >= 20 and nre_grid_points >= 3")
        if self.noise_sd <= 0.0 or self.half_width <= 0.0 or self.mode_radius <= 0.0:
            raise ValueError("noise_sd, half_width, mode_radius must be positive")
        require_non_negative_integer(self.fm_iterations, name="fm_iterations")
        require_non_negative_integer(self.fm_steps, name="fm_steps")
        if self.fm_iterations < 1 or self.fm_steps < 1:
            raise ValueError("fm_iterations and fm_steps must be positive")
        if self.fm_method not in ("euler", "heun", "rk4"):
            raise ValueError("fm_method must be 'euler', 'heun' or 'rk4'")


def slow_bimodal_config() -> BimodalConfig:
    """Return the larger-budget configuration of the slow tier.

    Returns:
        A :class:`BimodalConfig` with 2000 training simulations, 1500 Adam
        iterations per network, 60 test pairs and 100 SBC replicates.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """
    return BimodalConfig(
        n_train=2000,
        n_validation=40,
        n_test=60,
        n_sbc=100,
        sbc_draws=49,
        eval_draws=200,
        abc_simulations=1000,
        npe_hidden=(16,),
        npe_iterations=1500,
        nre_hidden=(16, 16),
        nre_iterations=1500,
        nre_grid_points=121,
    )


@dataclass(frozen=True, slots=True)
class MethodReport:
    """Measured held-out metrics of one posterior method (approximate).

    Attributes:
        method: ``"exact"``, ``"abc"``, ``"npe"``, ``"nre"``, ``"fm"``
            (flow matching, optional) or ``"prior"``.
        status: ``"ok"``, ``"reference"``, ``"baseline"`` or a training
            failure status.
        approximate: ``False`` only for the quadrature reference.
        training_simulations: Simulator calls spent on training.
        simulations_per_posterior: Simulator calls per new observation.
        evaluation_simulations: Simulator calls measured during the test and
            headline evaluation.
        sbc_simulations: Simulator calls measured in SBC.
        train_seconds: Training wall-clock seconds (informational).
        mean_abs_error: Mean over test observations of ``|mean - exact mean|``.
        sd_abs_error: Mean of ``|sd - exact sd|``.
        w1_to_exact: Mean Wasserstein-1 distance to the exact posterior.
        crps: Mean ensemble CRPS of the true parameter.
        coverage90: Fraction of test pairs whose true ``theta`` lies in the
            central 90% interval of the draws.
        test_mean_log_density: Mean ``log q(theta_true | y)`` (``None`` for
            sample-only methods; ``-inf`` if any density is zero).
        validation_mean_log_density: The same on the validation split.
        mode_mass_positive: Mean mass within the radius of ``+sqrt(y)``.
        mode_mass_negative: Mean mass within the radius of ``-sqrt(y)``.
        mode_mass_error: Mean over both modes of the absolute difference to
            the exact mode mass.
        predictive_misfit: Mean over test observations of
            ``sqrt(E_draws (theta^2 - y)^2)`` (noise-free predictive misfit).
        headline_mean: Posterior mean at the headline observation.
        headline_sd: Posterior standard deviation there.
        sbc_min_p_value: Smallest SBC chi-square p-value.
        sbc_calibrated: Bonferroni decision at level 0.01.
        ratio_normalization: Mean ``int p r d theta`` (NRE only).
        notes: Caveats.

    References:
        [EQ-INV-SBI-003], [EQ-INV-SBI-004].

    """

    method: str
    status: str
    approximate: bool
    training_simulations: int
    simulations_per_posterior: int
    evaluation_simulations: int
    sbc_simulations: int
    train_seconds: float
    mean_abs_error: float | None
    sd_abs_error: float | None
    w1_to_exact: float | None
    crps: float | None
    coverage90: float | None
    test_mean_log_density: float | None
    validation_mean_log_density: float | None
    mode_mass_positive: float | None
    mode_mass_negative: float | None
    mode_mass_error: float | None
    predictive_misfit: float | None
    headline_mean: float | None
    headline_sd: float | None
    sbc_min_p_value: float | None
    sbc_calibrated: bool | None
    ratio_normalization: float | None
    notes: str


@dataclass(frozen=True, slots=True)
class OODSummary:
    """Declared OOD heuristic applied to the comparison.

    Attributes:
        threshold: Training-score quantile used as the flag threshold.
        test_flag_fraction: Fraction of in-distribution test summaries flagged.
        ood_observation: The out-of-distribution observation.
        ood_score: Its squared Mahalanobis score.
        ood_flagged: Whether it was flagged.
        npe_at_ood: ``(mean, sd)`` of the NPE at the OOD observation (the
            network itself gives no warning), or ``None``.
        heuristic: Label: the score is a heuristic, not a guarantee.

    References:
        [EQ-INV-SBI-008].

    """

    threshold: float
    test_flag_fraction: float
    ood_observation: float
    ood_score: float
    ood_flagged: bool
    npe_at_ood: tuple[float, float] | None
    heuristic: str


@dataclass(frozen=True, slots=True)
class BimodalComparison:
    """Structured result of :func:`run_bimodal_comparison`.

    Attributes:
        config: The predeclared configuration.
        reports: One :class:`MethodReport` per method.
        split_sizes: ``(train, validation, test)`` pair counts.
        split_simulations: Simulator calls spent generating the splits.
        ood: The OOD heuristic summary.
        headline_exact_modes: Exact mode masses at the headline observation.
        training_status: Status of each trained model.

    References:
        [EQ-INV-SBI-003], [EQ-INV-SBI-004].

    """

    config: BimodalConfig
    reports: tuple[MethodReport, ...]
    split_sizes: tuple[int, int, int]
    split_simulations: int
    ood: OODSummary
    headline_exact_modes: tuple[float, float]
    training_status: dict[str, str] = field(default_factory=dict)

    def table(self) -> tuple[dict[str, object], ...]:
        """Return one dictionary per method (the cost/accuracy/coverage table).

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        return tuple(
            {name: getattr(r, name) for name in MethodReport.__slots__}
            for r in self.reports
        )


class _Posterior:
    """Per-observation posterior handle of one method.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """

    def __init__(
        self,
        sampler: Callable[[int, RandomStream], list[float]],
        log_density: Callable[[float], float] | None = None,
    ) -> None:
        """Store the sampler and optional log density.

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        self.sample = sampler
        self.log_density = log_density


class _Method:
    """A named posterior method with its accounting.

    References:
        [EQ-INV-SBI-003]. No equation is evaluated.

    """

    def __init__(
        self,
        name: str,
        factory: Callable[[float, WorkLedger], _Posterior],
        *,
        per_posterior: int = 0,
    ) -> None:
        """Store the factory and zero the counters.

        References:
            [EQ-INV-SBI-003]. No equation is evaluated.

        """
        self.name = name
        self.factory = factory
        self.per_posterior = per_posterior
        self.normalizations: list[float] = []
        self.training = 0
        self.seconds = 0.0
        self.status = "ok"


def _mode_masses(
    draws: Sequence[float], y: float, radius: float
) -> tuple[float, float]:
    """Return the sample mass within ``radius`` of ``+-sqrt(y)``.

    References:
        [EQ-INV-SBI-004].

    """
    root = math.sqrt(max(y, 0.0))
    m = len(draws)
    return (
        sum(1 for x in draws if abs(x - root) <= radius) / m,
        sum(1 for x in draws if abs(x + root) <= radius) / m,
    )


def _build_methods(
    config: BimodalConfig,
    contract: SimulatorContract,
    npe: TrainedModel,
    nre: TrainedModel,
    fm: TrainedModel | None = None,
) -> list[_Method]:
    """Return the posterior methods of the comparison (five, six with ``fm``).

    References:
        [EQ-INV-SBI-003], [EQ-GEN-NRE-002].

    """
    s, h = config.noise_sd, config.half_width

    def exact(y: float, _ledger: WorkLedger) -> _Posterior:
        """Return the quadrature posterior handle.

        References:
            [EQ-INV-SBI-004].

        """
        grid = bimodal_exact_posterior(y, noise_sd=s, half_width=h)
        return _Posterior(
            lambda size, st: grid.sample(size, st.generator()), grid.log_density_at
        )

    def prior(_y: float, _ledger: WorkLedger) -> _Posterior:
        """Return the prior-only baseline (ignores the data).

        References:
            [EQ-INV-SBI-003].

        """

        def sampler(size: int, st: RandomStream) -> list[float]:
            """Draw from the prior with one generator.

            References:
                [EQ-INV-SBI-003].

            """
            rng = st.generator()
            return [rng.uniform(-h, h) for _ in range(size)]

        return _Posterior(
            sampler, lambda x: -math.log(2.0 * h) if abs(x) <= h else -math.inf
        )

    def abc(y: float, ledger: WorkLedger) -> _Posterior:
        """Return an ABC-rejection handle (runs on each ``sample`` call).

        References:
            [EQ-INV-SBI-001].

        """

        def sampler(size: int, st: RandomStream) -> list[float]:
            """Run ABC with a fixed simulation budget and keep the closest.

            References:
                [EQ-INV-SBI-001].

            """
            budget = config.abc_simulations
            kept = min(size, max(2, round(0.1 * budget)))
            out = run_abc_rejection(
                contract,
                (y,),
                draws=kept,
                quantile=kept / budget,
                seed=st,
                ledger=ledger,
            )
            xs = [p[0] for p in out.parameters]
            if kept == size:
                return xs
            rng = st.child("resample").generator()
            return [xs[rng.randrange(len(xs))] for _ in range(size)]

        return _Posterior(sampler)

    methods = [
        _Method("exact", exact),
        _Method("abc", abc, per_posterior=config.abc_simulations),
        _Method("prior", prior),
    ]
    if isinstance(npe.model, AmortizedGaussianPosterior):
        model = npe.model

        def npe_posterior(y: float, _ledger: WorkLedger) -> _Posterior:
            """Return the Gaussian NPE handle at ``y``.

            References:
                [EQ-INF-NPE-001].

            """
            return _Posterior(
                lambda size, st: [v[0] for v in model.sample((y,), size, seed=st)],
                lambda x: model.log_density((x,), (y,)),
            )

        m = _Method("npe", npe_posterior)
        m.training, m.seconds, m.status = npe.simulations, npe.train_seconds, npe.status
        methods.append(m)
    else:
        m = _Method("npe", lambda y, _l: _Posterior(lambda n, st: []))
        m.status = npe.status
        methods.append(m)
    if isinstance(nre.model, RatioEstimator):
        ratio = nre.model
        grid = tuple(-h + 2.0 * h * i / (config.nre_grid_points - 1) for i in
                     range(config.nre_grid_points))  # fmt: skip

        def log_prior(theta: Sequence[float]) -> float:
            """Return the log of the uniform prior up to a constant.

            References:
                [EQ-INV-SBI-004].

            """
            return 0.0 if abs(theta[0]) <= h else -math.inf

        def nre_posterior(y: float, _ledger: WorkLedger) -> _Posterior:
            """Return the NRE grid-posterior handle at ``y``.

            References:
                [EQ-GEN-NRE-002].

            """
            posterior = ratio.posterior_on_grid((y,), log_prior, grid)
            method.normalizations.append(posterior.ratio_normalization)
            density = GridDensity(posterior.grid, posterior.density)
            return _Posterior(
                lambda size, st: density.sample(size, st.generator()),
                density.log_density_at,
            )

        method = _Method("nre", nre_posterior)
        method.training, method.seconds = nre.simulations, nre.train_seconds
        method.status = nre.status
        methods.append(method)
    else:
        method = _Method("nre", lambda y, _l: _Posterior(lambda n, st: []))
        method.status = nre.status
        methods.append(method)
    if fm is not None:
        flow_method = _Method("fm", lambda y, _l: _Posterior(lambda n, st: []))
        flow_method.training, flow_method.seconds = fm.simulations, fm.train_seconds
        flow_method.status = fm.status
        if isinstance(fm.model, FlowMatchingPosterior):
            flow = fm.model

            def fm_posterior(y: float, _ledger: WorkLedger) -> _Posterior:
                """Return the sample-only flow-matching handle at ``y``.

                No density is attached: the comparison applies no evidence or
                Metropolis-Hastings correction to this method.

                References:
                    [EQ-INV-SBI-012].

                """
                return _Posterior(
                    lambda size, st: [
                        v[0]
                        for v in flow.sample(
                            (y,),
                            size,
                            steps=config.fm_steps,
                            method=config.fm_method,
                            seed=st,
                        )
                    ]
                )

            flow_method.factory = fm_posterior
        methods.append(flow_method)
    return methods


def _evaluate_method(
    method: _Method,
    config: BimodalConfig,
    contract: SimulatorContract,
    splits: SimulationSplits,
    root: RandomStream,
) -> MethodReport:
    """Evaluate one method on the validation, test and SBC streams.

    References:
        [EQ-INV-SBI-003], [EQ-INV-SBI-004].

    """
    failed = method.status in ("failed", "nonfinite_loss")
    if failed:
        return MethodReport(
            method.name, method.status, True, 0, 0, 0, 0, method.seconds,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
            "training failed; no metrics",
        )  # fmt: skip
    s, h, name = config.noise_sd, config.half_width, method.name
    eval_ledger = WorkLedger()
    errs, sds, w1s, scores, hits, preds = [], [], [], [], [], []
    logs: list[float] = []
    plus, minus, plus_ref, minus_ref = [], [], [], []
    for i, (theta, summary) in enumerate(
        zip(splits.test.parameters, splits.test.summaries, strict=True)
    ):
        y = summary[0]
        ref = bimodal_exact_posterior(y, noise_sd=s, half_width=h)
        post = method.factory(y, eval_ledger)
        draws = post.sample(config.eval_draws, root.child("eval", name, i))
        errs.append(abs(statistics.fmean(draws) - ref.mean()))
        sds.append(abs(statistics.pstdev(draws) - ref.standard_deviation()))
        w1s.append(
            empirical_wasserstein_1d(
                draws, ref.grid, p=1.0, target_weights=ref.weights()
            )
        )
        scores.append(crps_ensemble_value(draws, theta[0]))
        hits.append(1.0 if _covered(draws, theta[0], 0.9) else 0.0)
        preds.append(math.sqrt(statistics.fmean((x * x - y) ** 2 for x in draws)))
        if post.log_density is not None:
            logs.append(post.log_density(theta[0]))
        if y >= config.min_y_modes:
            root_y = math.sqrt(y)
            mp, mm = _mode_masses(draws, y, config.mode_radius)
            plus.append(mp)
            minus.append(mm)
            plus_ref.append(
                ref.mass_in(root_y - config.mode_radius, root_y + config.mode_radius)
            )
            minus_ref.append(
                ref.mass_in(-root_y - config.mode_radius, -root_y + config.mode_radius)
            )
    validation_logs: list[float] = []
    for theta, summary in zip(
        splits.validation.parameters, splits.validation.summaries, strict=True
    ):
        post = method.factory(summary[0], eval_ledger)
        if post.log_density is None:
            break
        validation_logs.append(post.log_density(theta[0]))
    headline = method.factory(config.headline_y, eval_ledger).sample(
        max(config.eval_draws, 2), root.child("headline", name)
    )
    sbc_ledger = WorkLedger()

    def sbc_simulator(theta: Sequence[float], rng: random.Random) -> Vector:
        """Simulate the SBC observation through the contract.

        References:
            [EQ-INV-SBI-003].

        """
        return simulate_summary(contract, tuple(theta), rng, sbc_ledger)

    def sbc_posterior(data: object, stream: RandomStream) -> list[tuple[float, ...]]:
        """Return posterior draws for one SBC observation.

        References:
            [EQ-INV-SBI-003].

        """
        assert isinstance(data, tuple)
        return [
            (x,)
            for x in method.factory(data[0], sbc_ledger).sample(
                config.sbc_draws, stream
            )
        ]

    calibration = simulation_based_calibration(
        contract.prior_sampler,
        sbc_simulator,
        sbc_posterior,
        replicates=config.n_sbc,
        seed=root.child("sbc", name),
    )
    sbc_cost = sbc_ledger.simulator_calls
    if name == "exact":
        status, approx = "reference", False
    elif name == "prior":
        status, approx = "baseline", True
    else:
        status, approx = method.status, True
    notes = {
        "exact": "quadrature reference, CDF-inversion draws (Monte Carlo noise floor)",
        "abc": "rejection with a fixed simulation budget; closest draws kept",
        "npe": "diagonal Gaussian family: cannot be bimodal",
        "nre": "grid posterior from the classifier logit; normalization diagnostic",
        "prior": "ignores the data; passes SBC by construction",
        "fm": (
            "flow matching, sample-only: no normalized density is used, so no "
            "evidence or Metropolis-Hastings correction"
        ),
    }[name]
    mean = statistics.fmean
    return MethodReport(
        name, status, approx, method.training, method.per_posterior,
        eval_ledger.simulator_calls, sbc_cost, method.seconds,
        mean(errs), mean(sds), mean(w1s), mean(scores), mean(hits),
        (math.fsum(logs) / len(logs)) if logs else None,
        (math.fsum(validation_logs) / len(validation_logs))
        if validation_logs
        else None,
        mean(plus) if plus else None, mean(minus) if minus else None,
        (
            0.5 * mean(abs(a - b) for a, b in zip(plus, plus_ref, strict=True))
            + 0.5 * mean(abs(a - b) for a, b in zip(minus, minus_ref, strict=True))
        ) if plus else None,
        mean(preds), statistics.fmean(headline), statistics.pstdev(headline),
        calibration.min_p_value(), calibration.calibrated(0.01),
        mean(method.normalizations) if method.normalizations else None, notes,
    )  # fmt: skip


def run_bimodal_comparison(config: BimodalConfig | None = None) -> BimodalComparison:
    """Compare exact quadrature, ABC, NPE, NRE, (flow matching) and the prior.

    The model is ``y = theta^2 + noise``. Flow matching is added as method
    ``"fm"`` when ``config.include_flow_matching`` is true; it is trained on
    the same train split as the NRE, is sample-only (no density, so no
    evidence or Metropolis-Hastings correction) and so reports
    ``test_mean_log_density = None``.

    All inference methods see the same simulator contract and the same
    predeclared splits; every number in the returned table is measured by
    this call. The reference is exact quadrature of
    ``p(theta | y) ∝ N(y; theta^2, s^2)`` on ``[-h, h]``. Metrics are
    averaged over the held-out test pairs: error of posterior mean and
    standard deviation, Wasserstein-1 distance to the exact posterior,
    ensemble CRPS of the true parameter, 90% central coverage, mean log
    density (where the method has one), mass in windows around the two
    modes ``+-sqrt(y)`` (observations with ``y >= min_y_modes``), a
    noise-free predictive misfit, and SBC. SBC averages over the prior and
    is passed by the prior-only baseline, so it is not a measure of
    informativeness. ABC cost is charged per posterior; NPE/NRE cost is the
    one-off training budget. The OOD heuristic is the Mahalanobis score of
    :func:`fit_summary_ood_detector` fitted on the training summaries.

    Args:
        config: Predeclared budgets and seeds (default fast configuration).

    Returns:
        :class:`BimodalComparison`.

    Raises:
        SimulationFailure: If a simulation fails.

    Equation:
        [EQ-INV-SBI-003], [EQ-INV-SBI-004].

    References:
        [REF-TALTS-ETAL-2018], [REF-PAPAMAKARIOS-MURRAY-2016],
        [REF-HERMANS-BEGY-LOUPPE-2020].

    """
    cfg = config if config is not None else BimodalConfig()
    contract = bimodal_contract(cfg.noise_sd, cfg.half_width)
    root = RandomStream(cfg.seed)
    split_ledger = WorkLedger()
    splits = make_splits(
        contract,
        train=cfg.n_train,
        validation=cfg.n_validation,
        test=cfg.n_test,
        seed=root.child("splits"),
        ledger=split_ledger,
    )
    npe = train_npe(
        contract,
        cfg.n_train,
        hidden_layers=cfg.npe_hidden,
        iterations=cfg.npe_iterations,
        seed=splits.train_stream,
    )
    nre = train_nre(
        contract,
        cfg.n_train,
        hidden_layers=cfg.nre_hidden,
        iterations=cfg.nre_iterations,
        seed=splits.train_stream,
    )
    fm: TrainedModel | None = None
    if cfg.include_flow_matching:
        fm = train_flow_matching(
            contract,
            cfg.n_train,
            hidden_layers=cfg.fm_hidden,
            iterations=cfg.fm_iterations,
            seed=splits.train_stream,
        )
    methods = _build_methods(cfg, contract, npe, nre, fm)
    reports = tuple(_evaluate_method(m, cfg, contract, splits, root) for m in methods)
    detector = fit_summary_ood_detector(
        splits.train.summaries, quantile=cfg.ood_quantile
    )
    flagged = [detector.is_out_of_distribution(s) for s in splits.test.summaries]
    npe_ood: tuple[float, float] | None = None
    if isinstance(npe.model, AmortizedGaussianPosterior):
        means, sds = npe.model.moments((cfg.ood_y,))
        npe_ood = (means[0], sds[0])
    ood = OODSummary(
        detector.threshold,
        sum(flagged) / len(flagged),
        cfg.ood_y,
        detector.score((cfg.ood_y,)),
        detector.is_out_of_distribution((cfg.ood_y,)),
        npe_ood,
        "Mahalanobis heuristic on summaries; not a universal detector",
    )
    ref = bimodal_exact_posterior(
        cfg.headline_y, noise_sd=cfg.noise_sd, half_width=cfg.half_width
    )
    root_y = math.sqrt(cfg.headline_y)
    modes = (
        ref.mass_in(root_y - cfg.mode_radius, root_y + cfg.mode_radius),
        ref.mass_in(-root_y - cfg.mode_radius, -root_y + cfg.mode_radius),
    )
    return BimodalComparison(
        cfg,
        reports,
        (splits.train.size, splits.validation.size, splits.test.size),
        split_ledger.simulator_calls,
        ood,
        modes,
        {"npe": npe.status, "nre": nre.status}
        | ({"fm": fm.status} if fm is not None else {}),
    )


# --------------------------------------------------------------------------
# OOD heuristic
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummaryOODDetector:
    """Mahalanobis out-of-distribution heuristic on summaries (heuristic).

    Attributes:
        mean: Training summary mean.
        factor: Cholesky factor of the training sample covariance.
        threshold: Empirical ``quantile`` of the training scores.
        quantile: The quantile level used for ``threshold``.
        training_size: Number of training summaries.
        heuristic: Fixed label: the flag is a heuristic only.

    References:
        [EQ-INV-SBI-008].

    """

    mean: Vector
    factor: Matrix
    threshold: float
    quantile: float
    training_size: int
    heuristic: str = "heuristic: no coverage guarantee, detects only distant summaries"

    def score(self, summary: Sequence[float]) -> float:
        """Return the squared Mahalanobis distance to the training summaries.

        Args:
            summary: Summary vector of the training width.

        Returns:
            ``(s - m)' S^-1 (s - m)``.

        Raises:
            ValueError: If the width is wrong or an entry is not finite.

        Equation:
            [EQ-INV-SBI-008].

        References:
            [REF-PROBSTAT-VALIDATION]. The score is a heuristic.

        """
        s = as_vector(summary, name="summary")
        if len(s) != len(self.mean):
            raise ValueError("summary has the wrong width")
        z = solve_lower(
            self.factor, tuple(a - b for a, b in zip(s, self.mean, strict=True))
        )
        return math.fsum(v * v for v in z)

    def is_out_of_distribution(self, summary: Sequence[float]) -> bool:
        """Return whether the score exceeds the training-quantile threshold.

        By construction about ``1 - quantile`` of in-distribution summaries
        are flagged; far-away summaries are flagged, nearby shifts (for
        example in directions the simulator never produces but with a small
        Mahalanobis score) are not.

        Args:
            summary: Summary vector.

        Returns:
            ``True`` if flagged.

        Equation:
            [EQ-INV-SBI-008].

        References:
            [REF-PROBSTAT-VALIDATION]. The flag is a heuristic.

        """
        return self.score(summary) > self.threshold


def fit_summary_ood_detector(
    summaries: Sequence[Sequence[float]], *, quantile: float = 0.99
) -> SummaryOODDetector:
    """Fit the Mahalanobis heuristic on training summaries.

    Args:
        summaries: Training summaries (more than ``d + 1`` rows).
        quantile: Threshold level in ``(0, 1)``.

    Returns:
        :class:`SummaryOODDetector`.

    Raises:
        ValueError: If there are too few rows, the width varies, or the
            sample covariance is singular (reported, not regularized).

    Equation:
        [EQ-INV-SBI-008].

    References:
        [REF-PROBSTAT-VALIDATION]. The score is a heuristic.

    """
    level = require_finite_real(quantile, name="quantile")
    if not 0.0 < level < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    rows = [as_vector(s, name="summary") for s in summaries]
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("summaries must be non-empty with one width")
    d = len(rows[0])
    if len(rows) <= d + 1:
        raise ValueError("need more than d + 1 training summaries")
    mean, cov = _moments(rows)
    try:
        factor = cholesky(cov)
    except ValueError as error:
        raise ValueError("training summary covariance is singular") from error
    detector = SummaryOODDetector(mean, factor, math.inf, level, len(rows))
    scores = sorted(detector.score(r) for r in rows)
    index = min(len(scores) - 1, math.ceil(level * len(scores)) - 1)
    return SummaryOODDetector(mean, factor, scores[index], level, len(rows))


# --------------------------------------------------------------------------
# Gaussian score test
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GaussianScoreProblem:
    """Scalar Gaussian prior and linear-Gaussian likelihood.

    ``theta ~ N(prior_mean, prior_sd^2)``, ``y = a theta + N(0, noise_sd^2)``.

    Attributes:
        prior_mean: Prior mean ``m0``.
        prior_sd: Prior standard deviation ``s0 > 0``.
        coefficient: Forward coefficient ``a``.
        noise_sd: Noise standard deviation ``sigma > 0``.
        observation: Observed ``y``.

    References:
        [EQ-INV-SBI-005].

    """

    prior_mean: float
    prior_sd: float
    coefficient: float
    noise_sd: float
    observation: float

    def __post_init__(self) -> None:
        """Validate the scales.

        Raises:
            ValueError: If a standard deviation is not positive.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        for name in (
            "prior_mean",
            "prior_sd",
            "coefficient",
            "noise_sd",
            "observation",
        ):
            require_finite_real(getattr(self, name), name=name)
        if self.prior_sd <= 0.0 or self.noise_sd <= 0.0:
            raise ValueError("prior_sd and noise_sd must be positive")

    def posterior(self) -> tuple[float, float]:
        """Return the exact posterior ``(mean, variance)``.

        References:
            [EQ-INV-SBI-005].

        """
        precision = 1.0 / self.prior_sd**2 + self.coefficient**2 / self.noise_sd**2
        numerator = (
            self.prior_mean / self.prior_sd**2
            + self.coefficient * self.observation / self.noise_sd**2
        )
        return numerator / precision, 1.0 / precision


def posterior_score_t0(
    problem: GaussianScoreProblem, theta: float
) -> tuple[float, float]:
    """Return the posterior score at ``t = 0`` two ways.

    The first value is ``grad log p(theta) + grad log p(y | theta)``
    (``-(theta - m0)/s0^2 + a (y - a theta)/sigma^2``); the second is
    ``-(theta - mu) / v`` from the closed-form Gaussian posterior. They are
    equal because ``p(theta | y) ∝ p(theta) p(y | theta)``.

    Args:
        problem: The Gaussian problem.
        theta: Evaluation point.

    Returns:
        ``(prior_plus_likelihood, closed_form)``.

    Equation:
        [EQ-INV-SBI-005].

    References:
        [REF-SONG-ETAL-2021].

    """
    x = require_finite_real(theta, name="theta")
    p = problem
    additive = (
        -(x - p.prior_mean) / p.prior_sd**2
        + p.coefficient * (p.observation - p.coefficient * x) / p.noise_sd**2
    )
    mu, v = p.posterior()
    return additive, -(x - mu) / v


@dataclass(frozen=True, slots=True)
class NoisyScoreComparison:
    """Scores of the noisy-time posterior at ``theta_t`` (Gaussian case).

    The forward map is ``theta_t = alpha theta_0 + s eps``.

    Attributes:
        alpha: Signal coefficient.
        noise_scale: Noise standard deviation ``s``.
        theta_t: Evaluation point.
        exact: Exact score ``-(theta_t - alpha mu)/(alpha^2 v + s^2)`` of the
            marginal ``p_t(theta_t | y)``.
        naive: ``grad log p_t(theta_t) + grad_theta_t log p(y | theta_t)``
            with the clean-state likelihood evaluated at the noisy state.
        tweedie: Prior score plus the likelihood gradient evaluated at the
            Tweedie mean ``E[theta_0 | theta_t]``, ignoring its variance
            (the approximation of diffusion posterior sampling).
        exact_guided: Prior score plus the gradient of the exact noisy
            likelihood ``p_t(y | theta_t)``; equals ``exact`` in this case.
        naive_error: ``naive - exact``.
        tweedie_error: ``tweedie - exact``.
        exact_guided_error: ``exact_guided - exact`` (rounding level).

    References:
        [EQ-INV-SBI-006], [REF-SONG-ETAL-2021].

    """

    alpha: float
    noise_scale: float
    theta_t: float
    exact: float
    naive: float
    tweedie: float
    exact_guided: float
    naive_error: float
    tweedie_error: float
    exact_guided_error: float


def noisy_score_comparison(
    problem: GaussianScoreProblem, alpha: float, noise_scale: float, theta_t: float
) -> NoisyScoreComparison:
    """Compare exact and approximate posterior scores at a noisy state.

    With ``alpha = 1`` and ``noise_scale = 0`` the state is clean and
    ``naive``, ``tweedie`` and ``exact`` coincide
    (:func:`posterior_score_t0`). For ``noise_scale > 0`` naive addition is
    generally not exact: the likelihood must be the *noisy* likelihood
    ``p_t(y | theta_t) = N(y; a m_t, a^2 V_t + sigma^2)`` with
    ``m_t = E[theta_0 | theta_t]`` and ``V_t = Var[theta_0 | theta_t]``,
    which is closed form only in conjugate cases.

    Args:
        problem: The Gaussian problem.
        alpha: Non-zero signal coefficient.
        noise_scale: Noise standard deviation ``s >= 0``.
        theta_t: Noisy state.

    Returns:
        :class:`NoisyScoreComparison`.

    Raises:
        ValueError: If ``alpha == 0`` or ``noise_scale < 0``.

    Equation:
        [EQ-INV-SBI-006].

    References:
        [REF-SONG-ETAL-2021].

    """
    a_t = require_finite_real(alpha, name="alpha")
    s = require_finite_real(noise_scale, name="noise_scale")
    x = require_finite_real(theta_t, name="theta_t")
    if a_t == 0.0 or s < 0.0:
        raise ValueError("alpha must be non-zero and noise_scale non-negative")
    p = problem
    mu, v = p.posterior()
    s0sq, m0, a, sig2 = p.prior_sd**2, p.prior_mean, p.coefficient, p.noise_sd**2
    marginal_var = a_t**2 * s0sq + s * s
    prior_score = -(x - a_t * m0) / marginal_var
    exact = -(x - a_t * mu) / (a_t**2 * v + s * s)
    naive = prior_score + a * (p.observation - a * x) / sig2
    if s == 0.0:
        mean_t, var_t, slope = x / a_t, 0.0, 1.0 / a_t
    else:
        precision = 1.0 / s0sq + a_t**2 / (s * s)
        var_t = 1.0 / precision
        mean_t = (m0 / s0sq + a_t * x / (s * s)) * var_t
        slope = (a_t / (s * s)) * var_t
    tweedie = prior_score + a * (p.observation - a * mean_t) / sig2 * slope
    guided = (
        prior_score + a * (p.observation - a * mean_t) / (a * a * var_t + sig2) * slope
    )
    return NoisyScoreComparison(
        a_t, s, x, exact, naive, tweedie, guided,
        naive - exact, tweedie - exact, guided - exact,
    )  # fmt: skip


# --------------------------------------------------------------------------
# Plug-and-play proximal gradient
# --------------------------------------------------------------------------


def gaussian_proximal_denoiser(prior_precision: float) -> Denoiser:
    """Return the exact proximal map of ``g(x) = (mu / 2) |x|^2``.

    ``prox_{tau g}(v) = v / (1 + tau mu)``. Used as the denoiser it makes
    :func:`plug_and_play_proximal_gradient` solve the Tikhonov problem
    ``min (1/2)|Ax - b|^2 + (mu/2)|x|^2``.

    Args:
        prior_precision: ``mu >= 0``.

    Returns:
        A ``(v, step) -> x`` denoiser.

    Raises:
        ValueError: If ``mu`` is negative.

    Equation:
        [EQ-INV-SBI-007].

    References:
        [REF-BOYD-ETAL-2011].

    """
    mu = require_finite_real(prior_precision, name="prior_precision")
    if mu < 0.0:
        raise ValueError("prior_precision must be non-negative")

    def denoiser(v: Vector, step: float) -> Vector:
        """Return ``v / (1 + step mu)``.

        References:
            [EQ-INV-SBI-007].

        """
        return tuple(value / (1.0 + step * mu) for value in v)

    return denoiser


@dataclass(frozen=True, slots=True)
class PlugAndPlayResult:
    """Outcome of a plug-and-play proximal gradient run.

    Attributes:
        solution: Last iterate.
        iterations: Iterations performed.
        converged: Whether the fixed-point residual met the tolerance.
        stop_reason: ``"tolerance"``, ``"max_iterations"`` or
            ``"non_finite"``.
        residual: Final fixed-point residual ``|x_k - x_{k+1}| / step``.
        residual_history: Residual per iteration.
        step: Gradient step ``tau``.
        denoiser_calls: Number of denoiser evaluations.
        lipschitz_lower_bound: Largest observed ratio
            ``|D(u') - D(u)| / |u' - u|`` over consecutive iterates (a *lower*
            bound on the denoiser's Lipschitz constant; above one it voids the
            non-expansiveness assumption of the convergence theory).
        interpretation: Fixed label: the limit is a fixed point of
            ``D (I - tau grad f)``, a MAP solution only if ``D`` is the
            proximal map of a convex penalty.

    References:
        [EQ-INV-SBI-007], [REF-VENKATAKRISHNAN-BOUMAN-WOHLBERG-2013].

    """

    solution: Vector
    iterations: int
    converged: bool
    stop_reason: str
    residual: float
    residual_history: tuple[float, ...]
    step: float
    denoiser_calls: int
    lipschitz_lower_bound: float
    interpretation: str = (
        "fixed point of D(I - tau grad f); MAP only if D is a convex proximal map"
    )


def plug_and_play_proximal_gradient(
    operator: Sequence[Sequence[float]],
    data: Sequence[float],
    denoiser: Denoiser,
    *,
    step: float | None = None,
    initial: Sequence[float] | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> PlugAndPlayResult:
    """Iterate ``x <- D(x - tau A'(A x - b), tau)`` with a user denoiser.

    The data term is ``f(x) = |A x - b|^2 / 2``. If ``D`` is the proximal map
    of ``tau g`` for a closed convex ``g`` and ``tau <= 1/|A|_2^2``, this is
    the proximal gradient method and the limit minimizes ``f + g``. For a
    general denoiser no objective is implied: the limit (if any) is only a
    fixed point, and convergence needs documented assumptions such as
    non-expansiveness of ``D``; the reported ``lipschitz_lower_bound`` can
    refute but never certify them. The default step is ``1 / |A|_F^2``
    (``|A|_F^2 >= |A|_2^2``), which is safe and conservative.

    Args:
        operator: Dense ``m x n`` matrix ``A``.
        data: Data ``b`` of length ``m``.
        denoiser: ``(v, step) -> x`` callback.
        step: Positive step ``tau`` (default ``1 / |A|_F^2``).
        initial: Starting point (default zeros).
        max_iterations: Positive iteration cap.
        tolerance: Residual tolerance relative to ``1 + |x|``.

    Returns:
        :class:`PlugAndPlayResult`.

    Raises:
        TypeError: If ``denoiser`` is not callable.
        ValueError: If shapes disagree, a setting is invalid, or the denoiser
            returns the wrong length.

    Equation:
        [EQ-INV-SBI-007].

    References:
        [REF-VENKATAKRISHNAN-BOUMAN-WOHLBERG-2013], [REF-BOYD-ETAL-2011].

    """
    if not callable(denoiser):
        raise TypeError("denoiser must be callable")
    a = as_matrix(operator, name="operator")
    b = as_vector(data, name="data")
    m, n = len(a), len(a[0])
    if len(b) != m:
        raise ValueError("data length must match the operator rows")
    frob = math.fsum(v * v for row in a for v in row)
    if frob == 0.0:
        raise ValueError("operator must be non-zero")
    tau = 1.0 / frob if step is None else require_finite_real(step, name="step")
    if tau <= 0.0:
        raise ValueError("step must be positive")
    cap = require_non_negative_integer(max_iterations, name="max_iterations")
    tol = require_finite_real(tolerance, name="tolerance")
    if cap == 0 or tol < 0.0:
        raise ValueError("max_iterations must be positive and tolerance non-negative")
    x = as_vector(initial, name="initial") if initial is not None else (0.0,) * n
    if len(x) != n:
        raise ValueError("initial must have the operator's column count")
    at = tuple(zip(*a, strict=True))
    history: list[float] = []
    calls = 0
    lipschitz = 0.0
    previous: tuple[Vector, Vector] | None = None
    stop = "max_iterations"
    for _ in range(cap):
        resid = tuple(r - d for r, d in zip(matvec(a, x), b, strict=True))
        grad = matvec(at, resid)
        point = tuple(xi - tau * g for xi, g in zip(x, grad, strict=True))
        out = tuple(float(v) for v in denoiser(point, tau))
        calls += 1
        if len(out) != n:
            raise ValueError("denoiser must return a vector of the same length")
        if not all(math.isfinite(v) for v in out):
            stop = "non_finite"
            break
        if previous is not None:
            gap = math.sqrt(
                math.fsum((p - q) ** 2 for p, q in zip(point, previous[0], strict=True))
            )
            if gap > 0.0:
                moved = math.sqrt(
                    math.fsum(
                        (p - q) ** 2 for p, q in zip(out, previous[1], strict=True)
                    )
                )
                lipschitz = max(lipschitz, moved / gap)
        previous = (point, out)
        change = math.sqrt(math.fsum((p - q) ** 2 for p, q in zip(out, x, strict=True)))
        history.append(change / tau)
        x = out
        if change / tau <= tol * (1.0 + math.sqrt(math.fsum(v * v for v in x))):
            stop = "tolerance"
            break
    return PlugAndPlayResult(
        x,
        len(history),
        stop == "tolerance",
        stop,
        history[-1] if history else math.nan,
        tuple(history),
        tau,
        calls,
        lipschitz,
    )


# --------------------------------------------------------------------------
# Diffusion posterior sampling in the linear-Gaussian case (exact scores)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VPSchedule:
    """Variance-preserving forward noising with a linear ``beta`` schedule.

    The forward SDE is ``dx = -(beta(t)/2) x dt + sqrt(beta(t)) dW`` on
    ``[0, horizon]`` with ``beta(t) = beta_min + (beta_max - beta_min) t /
    horizon``. The transition law is ``x_t | x_0 ~ N(alpha(t) x_0,
    s(t)^2 I)`` with ``alpha(t) = exp(-(1/2) int_0^t beta)`` and
    ``s(t)^2 = 1 - alpha(t)^2`` (the variance-preserving SDE of Song et al.)
    [EQ-INV-SBI-009]. This module uses the variance-preserving form only.

    Attributes:
        beta_min: ``beta(0) > 0``.
        beta_max: ``beta(horizon) >= beta_min``.
        horizon: Final time ``T > 0``.

    References:
        [EQ-INV-SBI-009], [REF-SONG-ETAL-2021].

    """

    beta_min: float = 0.1
    beta_max: float = 20.0
    horizon: float = 1.0

    def __post_init__(self) -> None:
        """Validate the schedule.

        Raises:
            ValueError: If a value is not finite or ``0 < beta_min <=
                beta_max`` and ``horizon > 0`` fail.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        for name in ("beta_min", "beta_max", "horizon"):
            require_finite_real(getattr(self, name), name=name)
        if not 0.0 < self.beta_min <= self.beta_max or self.horizon <= 0.0:
            raise ValueError("need 0 < beta_min <= beta_max and horizon > 0")

    def beta(self, t: float) -> float:
        """Return ``beta(t)``.

        References:
            [EQ-INV-SBI-009].

        """
        return self.beta_min + (self.beta_max - self.beta_min) * t / self.horizon

    def alpha(self, t: float) -> float:
        """Return the signal coefficient ``alpha(t)``.

        References:
            [EQ-INV-SBI-009].

        """
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (
            t * t / self.horizon
        )
        return math.exp(-0.5 * integral)

    def noise_variance(self, t: float) -> float:
        """Return ``s(t)^2 = 1 - alpha(t)^2`` (zero at ``t = 0``).

        References:
            [EQ-INV-SBI-009].

        """
        a = self.alpha(t)
        return (1.0 - a) * (1.0 + a)


@dataclass(frozen=True, slots=True)
class LinearGaussianDiffusionProblem:
    """Gaussian prior and linear-Gaussian likelihood in ``n`` dimensions.

    ``theta ~ N(m0, C0)``, ``y = A theta + e``, ``e ~ N(0, Gamma)``, with
    the closed-form Gaussian posterior (conjugacy). It provides the exact
    posterior at diffusion time zero and, via :class:`VPSchedule`, at every
    noisy time.

    Attributes:
        prior_mean: ``m0`` (length ``n``).
        prior_covariance: SPD ``C0`` (``n x n``).
        forward: ``A`` (``m x n``).
        noise_covariance: SPD ``Gamma`` (``m x m``).
        observation: ``y`` (length ``m``).

    References:
        [EQ-INV-SBI-005], [EQ-INV-SBI-009].

    """

    prior_mean: Vector
    prior_covariance: Matrix
    forward: Matrix
    noise_covariance: Matrix
    observation: Vector

    def __post_init__(self) -> None:
        """Validate shapes and positive definiteness.

        Raises:
            ValueError: If a shape disagrees or a covariance is not
                symmetric positive definite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        m0 = as_vector(self.prior_mean, name="prior_mean")
        a = as_matrix(self.forward, name="forward")
        y = as_vector(self.observation, name="observation")
        c0 = symmetrize(self.prior_covariance, name="prior_covariance")
        gamma = symmetrize(self.noise_covariance, name="noise_covariance")
        if len(c0) != len(m0) or len(a[0]) != len(m0):
            raise ValueError("prior_mean, prior_covariance, forward disagree in n")
        if len(a) != len(y) or len(gamma) != len(y):
            raise ValueError("forward, noise_covariance, observation disagree in m")
        cholesky(c0)
        cholesky(gamma)
        object.__setattr__(self, "prior_mean", m0)
        object.__setattr__(self, "forward", a)
        object.__setattr__(self, "observation", y)
        object.__setattr__(self, "prior_covariance", c0)
        object.__setattr__(self, "noise_covariance", gamma)

    @property
    def dimension(self) -> int:
        """Return the parameter dimension ``n``.

        References:
            [EQ-INV-SBI-009]. No equation is evaluated.

        """
        return len(self.prior_mean)

    def information(self) -> tuple[Matrix, Vector]:
        """Return ``(B, b) = (A' Gamma^-1 A, A' Gamma^-1 y)``.

        References:
            [EQ-INV-SBI-005].

        """
        a = self.forward
        weighted = matmul(transpose(a), inverse_spd(self.noise_covariance))
        return matmul(weighted, a), matvec(weighted, self.observation)

    def posterior(self) -> tuple[Vector, Matrix]:
        """Return the exact posterior ``(mean, covariance)``.

        ``Sigma_p = (C0^-1 + B)^-1`` and ``mu = Sigma_p (C0^-1 m0 + b)``.

        References:
            [EQ-INV-SBI-005].

        """
        b_mat, b_vec = self.information()
        c0_inv = inverse_spd(self.prior_covariance)
        cov = symmetrize(inverse_spd(add(c0_inv, b_mat)))
        rhs = tuple(
            u + v for u, v in zip(matvec(c0_inv, self.prior_mean), b_vec, strict=True)
        )
        return matvec(cov, rhs), cov


def _scaled_identity_plus(matrix: Matrix, scale: float, shift: float) -> Matrix:
    """Return ``scale * matrix + shift * I``.

    References:
        [EQ-INV-SBI-009]. No equation is evaluated.

    """
    return add(scale_matrix(matrix, scale), scale_matrix(identity(len(matrix)), shift))


def _affine_score(
    problem: LinearGaussianDiffusionProblem,
    schedule: VPSchedule,
    t: float,
    score: str,
) -> tuple[Matrix, Vector]:
    """Return ``(S, c)`` with ``score(x, t) = S x + c`` for the named score.

    ``"exact"`` is the score of the noisy posterior marginal
    ``N(alpha mu, alpha^2 Sigma_p + s^2 I)`` [EQ-INV-SBI-009].
    ``"dps_approximate"`` is the prior score at the noisy time plus the
    likelihood gradient at the prior Tweedie mean [EQ-INV-SBI-011].

    References:
        [EQ-INV-SBI-009], [EQ-INV-SBI-011].

    """
    alpha = schedule.alpha(t)
    s2 = schedule.noise_variance(t)
    if score == "exact":
        mu, cov = problem.posterior()
        gain = inverse_spd(_scaled_identity_plus(cov, alpha * alpha, s2))
        return scale_matrix(gain, -1.0), matvec(gain, tuple(alpha * v for v in mu))
    if score != "dps_approximate":
        raise ValueError("score must be 'exact' or 'dps_approximate'")
    c0, m0 = problem.prior_covariance, problem.prior_mean
    b_mat, b_vec = problem.information()
    k_inv = inverse_spd(_scaled_identity_plus(c0, alpha * alpha, s2))
    jac = scale_matrix(matmul(c0, k_inv), alpha)  # d E[theta_0 | x_t] / d x_t
    jac_t = transpose(jac)
    quad = matmul(matmul(jac_t, b_mat), jac)
    slope = scale_matrix(add(k_inv, quad), -1.0)
    shift_lik = tuple(
        bv - u + alpha * w
        for bv, u, w in zip(
            b_vec, matvec(b_mat, m0), matvec(matmul(b_mat, jac), m0), strict=True
        )
    )
    offset = tuple(
        alpha * u + v
        for u, v in zip(matvec(k_inv, m0), matvec(jac_t, shift_lik), strict=True)
    )
    return slope, offset


@dataclass(frozen=True, slots=True)
class DiffusionSamplingResult:
    """Reverse-time sampler output compared with the exact posterior.

    The *deterministic* fields (``discretized_*``, ``mean_error``,
    ``covariance_error``, ``w2_per_coordinate``) are the exact Gaussian
    moments of the discretized sampler obtained by propagating mean and
    covariance through the affine Euler steps; they contain no Monte Carlo
    noise. The *sample* fields are measured from ``n_samples`` paths and
    carry Monte Carlo error of order ``n_samples**-1/2``.

    Attributes:
        steps: Number of Euler steps.
        sampler: ``"sde"`` (Euler-Maruyama of the reverse SDE) or ``"ode"``
            (Euler of the probability-flow ODE).
        score: ``"exact"`` or ``"dps_approximate"``.
        approximate: ``True`` unless the exact score is used; with the exact
            score the only error is the time discretization (and the
            initialization if ``"standard_normal"``).
        initialization: ``"exact_marginal"`` or ``"standard_normal"``.
        exact_mean: Exact posterior mean.
        exact_covariance: Exact posterior covariance.
        discretized_mean: Mean of the discretized sampler (deterministic).
        discretized_covariance: Covariance of the discretized sampler.
        mean_error: ``|discretized_mean - exact_mean|_2``.
        covariance_error: Frobenius norm of the covariance difference.
        w2_per_coordinate: Per-coordinate Wasserstein-2 distance between the
            discretized and the exact one-dimensional Gaussian marginals
            (an upper bound of the Wasserstein-1 distance).
        n_samples: Number of sample paths.
        samples: The draws (empty if ``n_samples == 0``).
        sample_mean: Sample mean (``None`` if no samples).
        sample_covariance: Sample covariance (``None`` if fewer than two).
        sample_mean_error: ``|sample_mean - exact_mean|_2``.
        sample_covariance_error: Frobenius error of the sample covariance.
        w1_per_coordinate: Per-coordinate Wasserstein-1 distance between the
            samples and the exact marginal (``None`` if no samples).
        score_evaluations: Score evaluations (``steps`` per path).
        label: Fixed text naming the approximation.

    References:
        [EQ-INV-SBI-010], [EQ-INV-SBI-011], [REF-SONG-ETAL-2021].

    """

    steps: int
    sampler: str
    score: str
    approximate: bool
    initialization: str
    exact_mean: Vector
    exact_covariance: Matrix
    discretized_mean: Vector
    discretized_covariance: Matrix
    mean_error: float
    covariance_error: float
    w2_per_coordinate: Vector
    n_samples: int
    samples: tuple[Vector, ...]
    sample_mean: Vector | None
    sample_covariance: Matrix | None
    sample_mean_error: float | None
    sample_covariance_error: float | None
    w1_per_coordinate: Vector | None
    score_evaluations: int
    label: str


def _euclid(v: Sequence[float]) -> float:
    """Return the Euclidean norm of ``v``.

    References:
        [EQ-INV-SBI-010]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(x * x for x in v))


def _frobenius_difference(a: Matrix, b: Matrix) -> float:
    """Return the Frobenius norm of ``a - b``.

    References:
        [EQ-INV-SBI-010]. No equation is evaluated.

    """
    return math.sqrt(
        math.fsum(
            (x - y) ** 2
            for ra, rb in zip(a, b, strict=True)
            for x, y in zip(ra, rb, strict=True)
        )
    )


def diffusion_posterior_sampling(
    problem: LinearGaussianDiffusionProblem,
    *,
    steps: int,
    sampler: str = "sde",
    score: str = "exact",
    n_samples: int = 500,
    schedule: VPSchedule | None = None,
    initialization: str = "exact_marginal",
    seed: int | RandomStream = 0,
) -> DiffusionSamplingResult:
    """Sample a linear-Gaussian posterior by reverse-time diffusion.

    The forward process is the variance-preserving noising of
    :class:`VPSchedule`. With a Gaussian prior and a linear-Gaussian
    likelihood the noisy-time marginal of the *posterior*,
    ``N(alpha mu, alpha^2 Sigma_p + s^2 I)``, is Gaussian and its score is
    available exactly, so no network is involved [EQ-INV-SBI-009]. Samples
    are produced by ``steps`` explicit Euler steps from ``t = T`` to ``t = 0``
    on a uniform grid, using the score at the start of each step:

    * ``sampler="sde"``: Euler-Maruyama of the reverse-time SDE
      ``x <- x + h (beta x / 2 + beta score) + sqrt(beta h) z``;
    * ``sampler="ode"``: Euler of the probability-flow ODE
      ``x <- x + h (beta x / 2 + beta score / 2)``;

    [EQ-INV-SBI-010]. Because the exact score is affine in ``x``, the law of
    the discretized sampler is Gaussian and its mean and covariance are
    propagated exactly; the result reports these noise-free moments and
    their error against the exact posterior next to the sample-based
    metrics (mean, covariance, per-coordinate Wasserstein-1 distance).

    ``score="dps_approximate"`` replaces the exact score by the DPS-style
    approximation: the *prior* score at the noisy time plus the gradient of
    the log-likelihood ``log N(y; A m_t(x), Gamma)`` evaluated at the prior
    Tweedie mean ``m_t(x) = E[theta_0 | theta_t = x]`` with the Jacobian
    ``d m_t / d x`` (the variance of ``theta_0 | theta_t`` is ignored)
    [EQ-INV-SBI-011]. It coincides with the exact score at ``t = 0`` and is
    *not* exact for ``t > 0``; its bias does not vanish as ``steps`` grows
    and the result is flagged ``approximate``. The gradient-norm step
    normalization of the original DPS algorithm is not used.

    Args:
        problem: The linear-Gaussian problem.
        steps: Number of Euler steps (positive).
        sampler: ``"sde"`` or ``"ode"``.
        score: ``"exact"`` or ``"dps_approximate"``.
        n_samples: Number of sample paths (``0`` for moments only).
        schedule: Variance-preserving schedule (default ``beta`` from 0.1 to
            20 on ``[0, 1]``).
        initialization: ``"exact_marginal"`` starts from the exact noisy
            marginal at ``T`` (isolating the discretization error);
            ``"standard_normal"`` starts from ``N(0, I)``, which adds an
            error of order ``alpha(T)``.
        seed: Integer seed or stream (path ``i`` uses the child stream
            ``("diffusion", i)``).

    Returns:
        :class:`DiffusionSamplingResult`.

    Raises:
        TypeError: If ``problem`` or ``schedule`` has the wrong type.
        ValueError: For invalid options.

    Equation:
        [EQ-INV-SBI-009], [EQ-INV-SBI-010], [EQ-INV-SBI-011].

    References:
        [REF-SONG-ETAL-2021], [REF-ANDERSON-1982], [REF-CHUNG-ETAL-2023].

    """
    if not isinstance(problem, LinearGaussianDiffusionProblem):
        raise TypeError("problem must be a LinearGaussianDiffusionProblem")
    sched = VPSchedule() if schedule is None else schedule
    if not isinstance(sched, VPSchedule):
        raise TypeError("schedule must be a VPSchedule")
    n_steps = require_non_negative_integer(steps, name="steps")
    count = require_non_negative_integer(n_samples, name="n_samples")
    if n_steps < 1:
        raise ValueError("steps must be positive")
    if sampler not in ("sde", "ode"):
        raise ValueError("sampler must be 'sde' or 'ode'")
    if score not in ("exact", "dps_approximate"):
        raise ValueError("score must be 'exact' or 'dps_approximate'")
    if initialization not in ("exact_marginal", "standard_normal"):
        raise ValueError("initialization must be 'exact_marginal' or 'standard_normal'")
    n = problem.dimension
    mu, cov = problem.posterior()
    horizon = sched.horizon
    h = horizon / n_steps
    if initialization == "exact_marginal":
        a_t, s2_t = sched.alpha(horizon), sched.noise_variance(horizon)
        start_mean = tuple(a_t * v for v in mu)
        start_cov = _scaled_identity_plus(cov, a_t * a_t, s2_t)
    else:
        start_mean = (0.0,) * n
        start_cov = identity(n)
    eye = identity(n)
    maps: list[tuple[Matrix, Vector, float]] = []
    for k in range(n_steps):
        t = horizon - k * h
        beta = sched.beta(t)
        slope, offset = _affine_score(problem, sched, t, score)
        weight = beta if sampler == "sde" else 0.5 * beta
        transition = add(
            scale_matrix(eye, 1.0 + 0.5 * h * beta),
            scale_matrix(slope, h * weight),
        )
        maps.append(
            (
                transition,
                tuple(h * weight * v for v in offset),
                beta * h if sampler == "sde" else 0.0,
            )
        )
    d_mean, d_cov = start_mean, start_cov
    for transition, shift, noise in maps:
        d_mean = tuple(
            u + v for u, v in zip(matvec(transition, d_mean), shift, strict=True)
        )
        d_cov = add(
            matmul(matmul(transition, d_cov), transpose(transition)),
            scale_matrix(eye, noise),
        )
    w2 = tuple(
        math.hypot(d_mean[i] - mu[i], math.sqrt(d_cov[i][i]) - math.sqrt(cov[i][i]))
        for i in range(n)
    )
    stream = resolve_stream(seed)
    lower = cholesky(start_cov)
    draws: list[Vector] = []
    for i in range(count):
        rng = stream.child("diffusion", i).generator()
        z = tuple(rng.gauss(0.0, 1.0) for _ in range(n))
        x = tuple(m + v for m, v in zip(start_mean, matvec(lower, z), strict=True))
        for transition, shift, noise in maps:
            moved = tuple(
                u + v for u, v in zip(matvec(transition, x), shift, strict=True)
            )
            if noise > 0.0:
                amp = math.sqrt(noise)
                moved = tuple(v + amp * rng.gauss(0.0, 1.0) for v in moved)
            x = moved
        draws.append(x)
    s_mean: Vector | None = None
    s_cov: Matrix | None = None
    s_mean_err: float | None = None
    s_cov_err: float | None = None
    w1: Vector | None = None
    if count >= 1:
        s_mean = tuple(math.fsum(x[j] for x in draws) / count for j in range(n))
        s_mean_err = _euclid(tuple(a - b for a, b in zip(s_mean, mu, strict=True)))
        w1 = tuple(
            _w1_to_normal([x[j] for x in draws], mu[j], cov[j][j]) for j in range(n)
        )
    if count >= 2:
        _, s_cov = _moments(draws)
        s_cov_err = _frobenius_difference(s_cov, cov)
    return DiffusionSamplingResult(
        n_steps,
        sampler,
        score,
        score != "exact",
        initialization,
        mu,
        cov,
        d_mean,
        d_cov,
        _euclid(tuple(a - b for a, b in zip(d_mean, mu, strict=True))),
        _frobenius_difference(d_cov, cov),
        w2,
        count,
        tuple(draws),
        s_mean,
        s_cov,
        s_mean_err,
        s_cov_err,
        w1,
        n_steps * count,
        (
            "exact score; error is time discretization only"
            if score == "exact"
            else "APPROXIMATE: prior score + likelihood gradient at the Tweedie "
            "mean (DPS-style); biased for t > 0"
        ),
    )


@dataclass(frozen=True, slots=True)
class DiffusionConvergence:
    """Results of :func:`diffusion_posterior_sampling` over several step counts.

    Attributes:
        results: One result per step count (increasing).
        mean_orders: Observed order ``log(e_i / e_{i+1}) / log(N_{i+1} /
            N_i)`` of the deterministic mean error between consecutive step
            counts (``nan`` where an error is zero).
        covariance_orders: The same for the covariance error.
        approximate: ``True`` if the score is not exact.

    References:
        [EQ-INV-SBI-010], [EQ-INV-SBI-011].

    """

    results: tuple[DiffusionSamplingResult, ...]
    mean_orders: tuple[float, ...]
    covariance_orders: tuple[float, ...]
    approximate: bool


def _observed_orders(
    steps: Sequence[int], errors: Sequence[float]
) -> tuple[float, ...]:
    """Return observed convergence orders between consecutive step counts.

    References:
        [EQ-INV-SBI-010]. No equation is evaluated.

    """
    return tuple(
        math.log(errors[i] / errors[i + 1]) / math.log(steps[i + 1] / steps[i])
        if errors[i] > 0.0 and errors[i + 1] > 0.0
        else math.nan
        for i in range(len(steps) - 1)
    )


def diffusion_convergence_study(
    problem: LinearGaussianDiffusionProblem,
    *,
    steps: Sequence[int] = (10, 20, 40, 80, 160),
    sampler: str = "sde",
    score: str = "exact",
    n_samples: int = 0,
    schedule: VPSchedule | None = None,
    initialization: str = "exact_marginal",
    seed: int | RandomStream = 0,
) -> DiffusionConvergence:
    """Measure the error of the diffusion sampler as the step count grows.

    With ``score="exact"`` the deterministic mean and covariance errors are
    expected to decrease at first order in the step size (Euler-Maruyama has
    weak order one; explicit Euler for the ODE has order one); the observed
    orders are reported, not assumed. With ``score="dps_approximate"`` the
    errors instead approach a positive floor, the bias of the approximation.

    Args:
        problem: The linear-Gaussian problem.
        steps: Strictly increasing positive step counts (at least two).
        sampler: ``"sde"`` or ``"ode"``.
        score: ``"exact"`` or ``"dps_approximate"``.
        n_samples: Sample paths per step count (``0``: moments only).
        schedule: Variance-preserving schedule.
        initialization: See :func:`diffusion_posterior_sampling`.
        seed: Integer seed or stream.

    Returns:
        :class:`DiffusionConvergence`.

    Raises:
        ValueError: If ``steps`` is not strictly increasing with two or more
            positive entries, or an option is invalid.

    Equation:
        [EQ-INV-SBI-010].

    References:
        [REF-SONG-ETAL-2021], [REF-KLOEDEN-PLATEN-1992].

    """
    grid = tuple(require_non_negative_integer(s, name="steps") for s in steps)
    if (
        len(grid) < 2
        or grid[0] < 1
        or any(b <= a for a, b in zip(grid, grid[1:], strict=False))
    ):
        raise ValueError("steps must be strictly increasing with >= 2 positive entries")
    results = tuple(
        diffusion_posterior_sampling(
            problem,
            steps=k,
            sampler=sampler,
            score=score,
            n_samples=n_samples,
            schedule=schedule,
            initialization=initialization,
            seed=seed,
        )
        for k in grid
    )
    return DiffusionConvergence(
        results,
        _observed_orders(grid, [r.mean_error for r in results]),
        _observed_orders(grid, [r.covariance_error for r in results]),
        score != "exact",
    )
