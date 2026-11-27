"""One entry point for the research-grade inverse toolbox: ``InverseStudy``.

Experimental (added in 0.28).

:class:`InverseStudy` drives the inverse-problem modules of
:mod:`probstat.inverse` end to end on one problem and one root seed, and
turns the numbers into a reproducible Markdown report, in the way
:class:`~probstat.workflow.UQStudy` does for forward uncertainty analysis.

Problems
    * Linear-Gaussian: a :class:`~probstat.inverse.LinearInverseProblem`
      (``y = A x + e``, ``e ~ N(0, Gamma)``) plus a Gaussian prior
      ``x ~ N(mu, P P')``. By default the prior is ``N(x_ref, I)``, which is
      the Bayesian reading of the problem's own penalty only when that
      penalty is the identity; any other penalty needs an explicit
      ``prior_factor``. The exact posterior
      (:func:`~probstat.inverse.linear_gaussian_posterior`) is the reference
      of :meth:`InverseStudy.compare`.
    * Nonlinear: a :class:`~probstat.inverse.NonlinearBayesProblem`, or a
      :class:`~probstat.inverse.DifferentiableForward` together with ``data``,
      ``noise`` and ``prior_factor``.

Stages (each optional, repeatable, with its own child seed stream
``root.child(stage, k)`` for its ``k``-th run, its own evaluation counts and
wall time, and its own report section)
    * :meth:`~InverseStudy.spectrum`: singular values of the prior-whitened
      Jacobian ``W J P``, numerical rank, directions the data inform
      [EQ-WF-INVSTUDY-001].
    * :meth:`~InverseStudy.regularize` (linear only): Tikhonov path and the
      GCV, UPRE, discrepancy, L-curve and evidence choices.
    * :meth:`~InverseStudy.map` and :meth:`~InverseStudy.laplace`: the MAP,
      low-rank Laplace covariance, marginal variances, Laplace log-evidence.
    * :meth:`~InverseStudy.sample`: pCN, generalized pCN, DILI-style,
      randomize-then-optimize, tempered SMC or transitional MCMC, with ESS,
      acceptance and the Monte Carlo standard error of the mean
      [EQ-WF-INVSTUDY-003].
    * :meth:`~InverseStudy.resolution`: matrix-free averaging kernels and
      likelihood-informed subspace rank.
    * :meth:`~InverseStudy.compare`: sample moments against the exact
      posterior (linear), a quadrature reference (at most two whitened
      parameters) or user moments [EQ-WF-INVSTUDY-004], [EQ-WF-INVSTUDY-005].
    * :meth:`~InverseStudy.report`: the Markdown report.

Common pitfalls:
    - Evaluation counts are model work. For a nonlinear problem they are the
      calls of ``G``, ``J v`` and ``J' w`` (``value``, ``jvp``, ``vjp``); for a
      linear problem the forward count merges ``A x`` and ``A v`` and the
      adjoint count is ``A' w``. A stage that needs a missing prerequisite
      (the Laplace approximation for ``"gpcn"`` and ``"dili"``) runs it as its
      own recorded stage first, so its cost appears there.
    - The exact reference of a linear problem and the quadrature reference
      are computed on first use inside :meth:`~InverseStudy.compare` and are
      charged to that stage; reference cost is not sampler cost.
    - The default report omits wall times so that two runs with the same
      seed give byte-identical text; pass ``include_timings=True`` for them.
    - Chains are single chains. The reported ESS is the split-chain estimate
      of one chain and the MCSE is ``sd / sqrt(ESS)``; there is no
      between-chain R-hat. Particle methods (``"smc"``, ``"tmcmc"``) report
      the naive iid ``sd / sqrt(N)``, which understates their error.
    - A comparison that passes says the sampler agrees with the reference at
      the stated tolerances for the stored sample, not that the model is
      right for real data.

Guide: docs/foundations/inverse-problems-and-calibration.md

References:
    [REF-KAIPIO-SOMERSALO-2007], [REF-BUI-THANH-ETAL-2013],
    [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-GEYER-1992],
    [REF-VEHTARI-ETAL-2021].

"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inference import effective_sample_size
from probstat.inference.transitional_mcmc import transitional_mcmc
from probstat.inverse import (
    ChainResult,
    DifferentiableForward,
    EvaluationCounts,
    GaussianLinearPosterior,
    LaplaceApproximation,
    LinearInverseProblem,
    MapEstimate,
    NonlinearBayesProblem,
    compute_map,
    dili_chain,
    gpcn_chain,
    laplace_approximation,
    linear_gaussian_posterior,
    logarithmic_grid,
    matrix_free_lis,
    pcn_chain,
    probe_resolution,
    quadrature_reference,
    randomize_then_optimize,
    regularization_path,
    sample_moments,
    select_regularization,
    spectral_diagnostics,
    tempered_smc_posterior,
)
from probstat.linalg import (
    CountingOperator,
    IdentityOperator,
    LinearOperator,
    ScaledIdentityCovariance,
    Vector,
    as_vector,
    materialize,
    singular_value_decomposition,
)

__all__ = [
    "REFERENCE_KINDS",
    "SAMPLE_METHODS",
    "SELECTION_RULES",
    "InverseStageRecord",
    "InverseStudy",
    "ReportTable",
    "ResolutionResult",
    "SampleSummary",
    "SpectrumResult",
    "StudyComparison",
]

SAMPLE_METHODS = ("pcn", "gpcn", "dili", "rto", "smc", "tmcmc")
SELECTION_RULES = ("gcv", "upre", "discrepancy", "l_curve", "evidence")
REFERENCE_KINDS = ("auto", "exact", "quadrature", "moments")

_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHOWN = 12
_DEFAULT_RULES = ("gcv", "upre", "discrepancy", "l_curve")
_CHAIN_METHODS = ("pcn", "gpcn", "dili", "rto")
_PARTICLE_METHODS = ("smc", "tmcmc")
_QUADRATURE_HALF_WIDTH = 8.0
_DENSE_LIMIT = 2000


# --------------------------------------------------------------------------
# validation and formatting helpers
# --------------------------------------------------------------------------
def _integer(value: object, name: str, minimum: int) -> int:
    """Return ``value`` as an integer at least ``minimum``.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(value: object, name: str, *, positive: bool = False) -> float:
    """Return ``value`` as a finite real, positive if requested.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite (or not positive when requested).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return number


def _fmt(value: float) -> str:
    """Return a stable six-significant-digit text of a float.

    References:
        Project reporting policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def _fmt_vector(values: Sequence[float], limit: int = 4) -> str:
    """Return a bracketed text of the first ``limit`` entries.

    References:
        Project reporting policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    shown = ", ".join(_fmt(v) for v in values[:limit])
    suffix = ", ..." if len(values) > limit else ""
    return f"[{shown}{suffix}]"


def _unit(size: int, index: int) -> Vector:
    """Return the ``index``-th unit vector of length ``size``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    return tuple(1.0 if i == index else 0.0 for i in range(size))


def _spread_indices(size: int, count: int) -> tuple[int, ...]:
    """Return up to ``count`` distinct indices spread over ``range(size)``.

    References:
        Project reporting policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    k = min(count, size)
    if k <= 1:
        return (0,)
    return tuple(sorted({round(i * (size - 1) / (k - 1)) for i in range(k)}))


def _series_ess(samples: Sequence[Vector], index: int) -> float:
    """Return the split ESS of coordinate ``index`` of one chain.

    ``nan`` for fewer than eight draws or a constant series.

    References:
        [EQ-WF-INVSTUDY-003], [REF-VEHTARI-ETAL-2021].

    """
    if len(samples) < 8:
        return math.nan
    return effective_sample_size([[s[index] for s in samples]])


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReportTable:
    """A small pre-formatted table of one report section.

    Attributes:
        title: Table heading.
        header: Column names.
        rows: Rows of already formatted cell texts.

    References:
        Project reporting policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """

    title: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class InverseStageRecord:
    """One stage run of an :class:`InverseStudy`, with cost and provenance.

    Attributes:
        stage: Stage name (``"spectrum"``, ``"regularize"``, ``"map"``,
            ``"laplace"``, ``"sample"``, ``"resolution"``, ``"compare"``).
        run: Zero-based run index of this stage; the stage stream is
            ``root.child(stage, run)``.
        label: Human-readable description.
        summary: Headline numbers of the stage.
        evaluations: Model work spent by the stage (see the module notes on
            how linear and nonlinear counts are defined).
        seconds: Wall time of the stage.
        seed: Seed of the stage stream.
        settings: Arguments that define the stage.
        tables: Pre-formatted report tables.
        warnings: Convergence, accuracy and approximation warnings.
        result: The full result object of the stage.

    References:
        [EQ-WF-INVSTUDY-002], [REF-KAIPIO-SOMERSALO-2007].

    """

    stage: str
    run: int
    label: str
    summary: Mapping[str, float | str]
    evaluations: EvaluationCounts
    seconds: float
    seed: int
    settings: Mapping[str, object]
    tables: tuple[ReportTable, ...]
    warnings: tuple[str, ...]
    result: Any = field(repr=False, compare=False)

    @property
    def work(self) -> int:
        """Return ``value + jvp + vjp`` of the stage.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return self.evaluations.work


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Spectrum of the prior-whitened Jacobian ``K = W J P``.

    Attributes:
        singular_values: ``sigma_1 >= ... >= sigma_k`` of ``K``.
        eigenvalues: ``lambda_i = sigma_i^2``, the eigenvalues of the
            prior-preconditioned Gauss-Newton Hessian.
        rank: Numerical rank of ``K``.
        detectable: Number of ``lambda_i`` above ``threshold``.
        threshold: The eigenvalue threshold.
        point: ``"prior mean"`` or ``"MAP"``, where ``J`` was evaluated.
        variance_reduction: ``lambda_i / (1 + lambda_i)`` along each direction.
        noise_whitened_rank: Rank of ``W A`` alone (linear problems), else
            ``None``.
        noise_whitened_condition: ``sigma_1 / sigma_k`` of ``W A`` (linear),
            else ``None``.
        out_of_range_norm: ``||(I - U U') W y||`` (linear), else ``None``.

    References:
        [EQ-WF-INVSTUDY-001], [REF-BUI-THANH-ETAL-2013].

    """

    singular_values: Vector
    eigenvalues: Vector
    rank: int
    detectable: int
    threshold: float
    point: str
    variance_reduction: Vector
    noise_whitened_rank: int | None
    noise_whitened_condition: float | None
    out_of_range_norm: float | None


@dataclass(frozen=True, slots=True)
class SampleSummary:
    """Posterior samples with their Monte Carlo diagnostics.

    Attributes:
        method: One of :data:`SAMPLE_METHODS`.
        asymptotically_exact: Whether the sampler targets the posterior
            exactly as the run length grows (chains and Metropolis RTO) or
            is a finite-particle approximation (SMC, TMCMC).
        converged: Whether the sampler finished (particle methods can fail).
        stop_reason: The sampler's status text.
        samples: Stored physical samples (after burn-in).
        mean: Sample mean of ``x``.
        variance: Sample variance of ``x`` (divisor ``N - 1``).
        ess: Split-chain ESS per coordinate (``nan`` for particle methods).
        mcse: Monte Carlo standard error of each mean.
        mcse_kind: ``"ess"`` (``sd / sqrt(ESS)``) or ``"iid-naive"``
            (``sd / sqrt(N)``, understated for dependent particles).
        acceptance_rate: Acceptance fraction (``nan`` if not defined).
        log_evidence: Evidence estimate of SMC or TMCMC (``nan`` otherwise).
        work: Model work of the sampler.
        ess_per_work: ``min ESS / work`` (``nan`` for particle methods).
        details: The sampler's own result object.

    References:
        [EQ-WF-INVSTUDY-003], [REF-GEYER-1992], [REF-VEHTARI-ETAL-2021].

    """

    method: str
    asymptotically_exact: bool
    converged: bool
    stop_reason: str
    samples: tuple[Vector, ...]
    mean: Vector
    variance: Vector
    ess: Vector
    mcse: Vector
    mcse_kind: str
    acceptance_rate: float
    log_evidence: float
    work: int
    ess_per_work: float
    details: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Matrix-free resolution probes and LIS rank at one linearization point.

    Attributes:
        indices: Probed parameter indices ``j``.
        diagonal: ``R_jj`` for each index (``1`` is perfect resolution).
        peak_index: ``argmax_i |R_ij|`` for each index.
        peak_value: ``R_{peak, j}``.
        trace: Hutchinson estimate of ``tr R`` (``nan`` without probes).
        trace_standard_error: Its Monte Carlo standard error.
        lis_rank: Number of likelihood-informed directions.
        lis_threshold: The LIS eigenvalue threshold.
        lis_converged: Whether the LIS rank was decided.
        lis_stop_reason: The eigensolver's stop reason.
        solves: Number of conjugate-gradient solves.
        all_converged: Whether every solve met its tolerance.
        point: ``"prior mean"``, ``"MAP"`` or ``"linear"``.

    References:
        [EQ-INV-LIS-017], [EQ-INV-LIS-018], [EQ-INV-LIS-019],
        [REF-TARANTOLA-2005].

    """

    indices: tuple[int, ...]
    diagonal: Vector
    peak_index: tuple[int, ...]
    peak_value: Vector
    trace: float
    trace_standard_error: float
    lis_rank: int
    lis_threshold: float
    lis_converged: bool
    lis_stop_reason: str
    solves: int
    all_converged: bool
    point: str


@dataclass(frozen=True, slots=True)
class StudyComparison:
    """Sample moments compared with a reference.

    Attributes:
        method: The sample stage that was compared.
        reference_kind: ``"exact"``, ``"quadrature"`` or ``"moments"``.
        reference_mean: Reference mean of ``x``.
        reference_variance: Reference marginal variances of ``x``.
        mean_error_sd: ``|mean_sample - mean_ref| / sd_ref`` per coordinate.
        mean_error_mcse: ``|mean_sample - mean_ref| / MCSE`` per coordinate
            (``nan`` where the MCSE is not available).
        variance_ratio: ``var_sample / var_ref`` per coordinate.
        mean_tolerance_sd: Allowed mean error in reference standard
            deviations (before the MCSE allowance).
        variance_tolerance: Allowed ``|ratio - 1|``.
        mcse_multiplier: Multiple of the MCSE added to the mean allowance.
        passed: Whether every coordinate is within its tolerances.
        failures: Human-readable reasons for each violation.
        laplace_mean_error_sd: Laplace mean error in reference sd, if the
            Laplace stage ran, else ``None``.
        laplace_variance_ratio: Laplace marginal variance over the reference
            variance, if the Laplace stage ran, else ``None``.
        reference_log_evidence: Reference ``log p(y)`` (``nan`` if unknown).
        reference_note: Accuracy note of the reference.

    References:
        [EQ-WF-INVSTUDY-004], [EQ-WF-INVSTUDY-005].

    """

    method: str
    reference_kind: str
    reference_mean: Vector
    reference_variance: Vector
    mean_error_sd: Vector
    mean_error_mcse: Vector
    variance_ratio: Vector
    mean_tolerance_sd: float
    variance_tolerance: float
    mcse_multiplier: float
    passed: bool
    failures: tuple[str, ...]
    laplace_mean_error_sd: Vector | None
    laplace_variance_ratio: Vector | None
    reference_log_evidence: float
    reference_note: str


@dataclass(frozen=True, slots=True)
class _Reference:
    """Internal reference moments of ``x``.

    References:
        [EQ-WF-INVSTUDY-004].

    """

    kind: str
    mean: Vector
    variance: Vector
    log_evidence: float
    note: str


@dataclass(frozen=True, slots=True)
class _Start:
    """Internal stage clock and counter snapshot.

    References:
        [EQ-WF-INVSTUDY-002].

    """

    name: str
    run: int
    stream: RandomStream
    counts: EvaluationCounts
    clock: float


class _LinearForward:
    """``G(x) = A x`` through a counting operator, as a differentiable forward.

    References:
        [EQ-WF-INVSTUDY-002].

    """

    __slots__ = ("_operator",)

    def __init__(self, operator: LinearOperator) -> None:
        """Store the operator.

        Args:
            operator: The (counting) operator ``A``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        self._operator = operator

    @property
    def input_dimension(self) -> int:
        """Return ``n``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return self._operator.shape[1]

    @property
    def output_dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return self._operator.shape[0]

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``A x``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return tuple(self._operator.matvec(x))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``A v``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return tuple(self._operator.matvec(v))

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``A' w``.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return tuple(self._operator.rmatvec(w))


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------
class InverseStudy:
    """Run the inverse-problem stages on one problem and write one report.

    Args:
        problem: A :class:`~probstat.inverse.LinearInverseProblem`, a
            :class:`~probstat.inverse.NonlinearBayesProblem`, or a
            :class:`~probstat.inverse.DifferentiableForward` (which also needs
            ``data``, ``noise`` and ``prior_factor``).
        seed: Root seed (non-negative integer) or
            :class:`~probstat.foundations.random_streams.RandomStream`. The
            ``k``-th run of a stage uses ``root.child(stage, k)``, so a result
            depends only on the root seed and the run index of its stage.
        label: Title of the report.
        data: Observation ``y`` (forward-model input only).
        noise: Covariance factor of the noise (forward-model input only).
        prior_factor: ``P`` with prior covariance ``P P'``. For a linear
            problem it defaults to the identity if the problem's penalty is
            the identity; for a forward model it is required.
        prior_mean: ``mu`` (default: the problem's ``reference`` for a linear
            problem, zeros for a forward model).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If arguments are inconsistent (extra inputs together with
            a ready problem, a missing prior for a non-identity penalty, or
            a negative seed).

    References:
        [EQ-WF-INVSTUDY-002], [REF-KAIPIO-SOMERSALO-2007].

    """

    def __init__(
        self,
        problem: object,
        *,
        seed: int | RandomStream = 0,
        label: str = "Inverse study",
        data: Sequence[float] | None = None,
        noise: object | None = None,
        prior_factor: object | None = None,
        prior_mean: Sequence[float] | None = None,
    ) -> None:
        """Validate the problem and build the shared Bayesian problem.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If arguments are inconsistent.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        if isinstance(seed, bool) or not isinstance(seed, int | RandomStream):
            raise TypeError("seed must be an integer or a RandomStream")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        self._root = resolve_stream(seed)
        self._label = label.strip()
        self._runs: dict[str, int] = {}
        self._records: list[InverseStageRecord] = []
        self._map: MapEstimate | None = None
        self._laplace: LaplaceApproximation | None = None
        self._exact: GaussianLinearPosterior | None = None
        self._counted: CountingOperator | None = None
        self._linear: LinearInverseProblem | None = None
        self._sigma1: float | None = None
        self._quadrature: dict[str, _Reference] = {}
        extra = (data, noise, prior_factor, prior_mean)
        if isinstance(problem, LinearInverseProblem):
            if data is not None or noise is not None:
                raise ValueError("data and noise belong to the LinearInverseProblem")
            self.kind = "linear"
            m, n = problem.shape
            counted = CountingOperator(problem.operator)
            self._counted = counted
            self._linear = LinearInverseProblem(
                counted, problem.data, problem.noise, problem.penalty, problem.reference
            )
            if prior_factor is None:
                if not isinstance(problem.penalty, IdentityOperator):
                    raise ValueError(
                        "the penalty is not the identity: pass prior_factor to "
                        "state the Gaussian prior"
                    )
                factor: object = IdentityOperator(n)
            else:
                factor = prior_factor
            mean = problem.reference if prior_mean is None else prior_mean
            covariance: object = (
                problem.noise
                if problem.noise is not None
                else ScaledIdentityCovariance(m, 1.0)
            )
            self._bayes = NonlinearBayesProblem(
                _LinearForward(counted),
                problem.data,
                noise=covariance,
                prior_factor=factor,
                prior_mean=mean,
            )
        elif isinstance(problem, NonlinearBayesProblem):
            if any(item is not None for item in extra):
                raise ValueError(
                    "data, noise, prior_factor and prior_mean belong to the "
                    "NonlinearBayesProblem"
                )
            self.kind = "nonlinear"
            self._bayes = problem
        elif isinstance(problem, DifferentiableForward):
            if data is None or noise is None or prior_factor is None:
                raise ValueError("a forward model needs data, noise and prior_factor")
            self.kind = "nonlinear"
            self._bayes = NonlinearBayesProblem(
                problem,
                data,
                noise=noise,
                prior_factor=prior_factor,
                prior_mean=prior_mean,
            )
        else:
            raise TypeError(
                "problem must be a LinearInverseProblem, a NonlinearBayesProblem "
                "or a DifferentiableForward"
            )

    # ----- basic accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        """Return the integer seed of the root stream.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return self._root.seed

    @property
    def problem(self) -> NonlinearBayesProblem:
        """Return the Bayesian problem all stages share.

        For a linear problem this is the linear forward ``A x`` wrapped as a
        :class:`~probstat.inverse.NonlinearBayesProblem`.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return self._bayes

    @property
    def records(self) -> tuple[InverseStageRecord, ...]:
        """Return all stage records in execution order.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return tuple(self._records)

    @property
    def evaluations(self) -> EvaluationCounts:
        """Return the model work summed over all recorded stages.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        return EvaluationCounts(
            sum(r.evaluations.value for r in self._records),
            sum(r.evaluations.jvp for r in self._records),
            sum(r.evaluations.vjp for r in self._records),
        )

    def latest(self, stage: str) -> InverseStageRecord:
        """Return the most recent record of ``stage``.

        Args:
            stage: Stage name.

        Returns:
            The record.

        Raises:
            ValueError: If the stage has not been run.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        for record in reversed(self._records):
            if record.stage == stage:
                return record
        raise ValueError(f"no {stage!r} stage has been run")

    # ----- stage plumbing --------------------------------------------------
    def _snapshot(self) -> EvaluationCounts:
        """Return the current model-work counters.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        if self._counted is not None:
            counts = self._counted.counts
            return EvaluationCounts(counts.forward, 0, counts.transpose)
        return self._bayes.counts

    def _begin(self, name: str) -> _Start:
        """Open a stage: take its child stream, counters and clock.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        run = self._runs.get(name, 0)
        self._runs[name] = run + 1
        return _Start(
            name,
            run,
            self._root.child(name, run),
            self._snapshot(),
            time.perf_counter(),
        )

    def _finish(
        self,
        start: _Start,
        label: str,
        summary: Mapping[str, float | str],
        result: Any,
        *,
        settings: Mapping[str, object],
        tables: Sequence[ReportTable] = (),
        warnings: Sequence[str] = (),
    ) -> InverseStageRecord:
        """Close a stage and append its record.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        record = InverseStageRecord(
            stage=start.name,
            run=start.run,
            label=label,
            summary=MappingProxyType(dict(summary)),
            evaluations=self._snapshot().minus(start.counts),
            seconds=time.perf_counter() - start.clock,
            seed=start.stream.seed,
            settings=MappingProxyType(dict(settings)),
            tables=tuple(tables),
            warnings=tuple(warnings),
            result=result,
        )
        self._records.append(record)
        return record

    def _coordinate_table(
        self, title: str, columns: Mapping[str, Sequence[float]]
    ) -> ReportTable:
        """Return a per-coordinate table of at most ``_SHOWN`` rows.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        n = self._bayes.physical_dimension
        rows = tuple(
            (str(i), *(_fmt(values[i]) for values in columns.values()))
            for i in range(min(n, _SHOWN))
        )
        suffix = "" if n <= _SHOWN else f" (first {_SHOWN} of {n})"
        return ReportTable(title + suffix, ("i", *columns.keys()), rows)

    def _require_linear(self, stage: str) -> None:
        """Raise unless the study is linear.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        if self.kind != "linear":
            raise ValueError(f"{stage}() needs a linear problem")

    def _exact_posterior(self) -> GaussianLinearPosterior:
        """Return (and cache) the exact linear-Gaussian posterior.

        References:
            [EQ-INV-LG-001], [EQ-WF-INVSTUDY-004].

        """
        if self._exact is None:
            assert self._counted is not None
            bayes = self._bayes
            method = (
                "parameter_space" if bayes.dimension <= _DENSE_LIMIT else "data_space"
            )
            self._exact = linear_gaussian_posterior(
                self._counted,
                bayes.data,
                noise=bayes.noise,
                prior_mean=bayes.prior_mean,
                prior_factor=bayes.prior_factor,
                method=method,  # type: ignore[arg-type]
            )
        return self._exact

    def _linearization(self) -> tuple[Vector, str]:
        """Return the whitened linearization point and its name.

        References:
            [EQ-WF-INVSTUDY-001].

        """
        if self.kind == "linear":
            return (0.0,) * self._bayes.dimension, "linear"
        if self._map is not None:
            return self._map.whitened, "MAP"
        return (0.0,) * self._bayes.dimension, "prior mean"

    # ----- spectrum --------------------------------------------------------
    def spectrum(
        self, *, threshold: float = 1.0, max_dimension: int = 400
    ) -> InverseStageRecord:
        """Return the record of the prior-whitened Jacobian spectrum.

        Forms ``K = W J P`` densely (``r`` Jacobian actions) at the prior mean
        or, if :meth:`map` ran, at the MAP, and takes its singular values
        ``sigma_i``. The eigenvalues ``lambda_i = sigma_i^2`` of the
        prior-preconditioned Gauss-Newton Hessian say how much each prior-
        whitened direction is informed: the posterior variance along it is
        ``1 / (1 + lambda_i)`` of the prior's [EQ-WF-INVSTUDY-001]. A linear
        problem adds the singular spectrum of ``W A`` alone (rank,
        condition number, Picard out-of-range norm).

        Args:
            threshold: ``lambda`` above which a direction counts as
                detectable (``> 0``; ``1`` means the data halve the prior
                variance).
            max_dimension: Largest ``r`` for the dense route.

        Returns:
            The stage record; ``record.result`` is a :class:`SpectrumResult`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``threshold <= 0`` or ``r`` exceeds
                ``max_dimension`` (use :meth:`laplace` or :meth:`resolution`,
                which are matrix-free).

        References:
            [EQ-WF-INVSTUDY-001], [REF-BUI-THANH-ETAL-2013],
            [REF-HANSEN-2008].

        """
        tau = _real(threshold, "threshold", positive=True)
        cap = _integer(max_dimension, "max_dimension", 1)
        bayes = self._bayes
        if bayes.dimension > cap:
            raise ValueError(
                f"r = {bayes.dimension} exceeds max_dimension = {cap}; use the "
                "matrix-free laplace() or resolution() stages"
            )
        start = self._begin("spectrum")
        point, where = self._linearization()
        k_matrix = bayes.whitened_jacobian_dense(point)
        sigma = tuple(singular_value_decomposition(k_matrix).singular_values)
        eigenvalues = tuple(s * s for s in sigma)
        cutoff = max(bayes.data_dimension, bayes.dimension) * 2.0**-52
        top = sigma[0] if sigma else 0.0
        rank = sum(1 for s in sigma if s > cutoff * top) if top > 0.0 else 0
        detectable = sum(1 for v in eigenvalues if v > tau)
        reduction = tuple(v / (1.0 + v) for v in eigenvalues)
        warnings: list[str] = []
        whitened_rank: int | None = None
        whitened_cond: float | None = None
        out_of_range: float | None = None
        summary: dict[str, float | str] = {
            "linearization point": where,
            "singular values": float(len(sigma)),
            "sigma_1": top,
            "numerical rank": float(rank),
            "detectable directions": float(detectable),
            "threshold lambda": tau,
        }
        if self.kind == "linear":
            assert self._counted is not None and self._linear is not None
            diag = spectral_diagnostics(
                self._counted, data=bayes.data, noise=self._linear.noise
            )
            whitened_rank = diag.rank
            whitened_cond = diag.condition_number
            out_of_range = diag.out_of_range_norm
            summary["rank of W A"] = float(diag.rank)
            summary["condition of W A"] = diag.condition_number
            if out_of_range is not None:
                summary["out-of-range data norm"] = out_of_range
        else:
            warnings.append(
                f"Jacobian evaluated at the {where}; the nonlinear spectrum is a "
                "local statement and changes with the linearization point"
            )
        if rank < min(bayes.data_dimension, bayes.dimension):
            warnings.append(
                f"K has numerical rank {rank} below min(m, r) = "
                f"{min(bayes.data_dimension, bayes.dimension)}: some prior "
                "directions are not informed by the data at all"
            )
        result = SpectrumResult(
            sigma,
            eigenvalues,
            rank,
            detectable,
            tau,
            where,
            reduction,
            whitened_rank,
            whitened_cond,
            out_of_range,
        )
        rows = tuple(
            (str(i), _fmt(sigma[i]), _fmt(eigenvalues[i]), _fmt(reduction[i]))
            for i in range(min(len(sigma), _SHOWN))
        )
        table = ReportTable(
            "Whitened singular values"
            + ("" if len(sigma) <= _SHOWN else f" (first {_SHOWN} of {len(sigma)})"),
            ("i", "sigma_i", "lambda_i = sigma_i^2", "variance reduction"),
            rows,
        )
        return self._finish(
            start,
            "Spectrum of the prior-whitened Jacobian",
            summary,
            result,
            settings={"threshold": tau, "max_dimension": cap},
            tables=(table,),
            warnings=warnings,
        )

    # ----- regularization --------------------------------------------------
    def regularize(
        self,
        *,
        rules: Sequence[str] = _DEFAULT_RULES,
        grid: Sequence[float] | None = None,
        points: int = 25,
        trace: str = "exact",
        probes: int = 20,
    ) -> InverseStageRecord:
        """Return the record of the Tikhonov path and the parameter rules.

        Evaluates the affine Tikhonov solution of the problem's own penalty
        ``L`` on a grid of ``lambda`` and applies the selection rules to the
        path. This is the deterministic view; it is independent of the
        Gaussian prior used by the Bayesian stages (the prior corresponds to
        ``lambda = 1`` only for a suitably scaled penalty). The discrepancy
        rule uses the whitened noise norm ``sqrt(m)``.

        Args:
            rules: Subset of :data:`SELECTION_RULES`.
            grid: Increasing positive ``lambda`` values (at least three);
                default ``points`` log-spaced values from ``1e-5 sigma_1`` to
                ``2 sigma_1`` of ``W A``.
            points: Grid size of the default grid (``>= 3``).
            trace: ``"exact"`` or ``"hutchinson"`` degrees of freedom.
            probes: Hutchinson probes.

        Returns:
            The stage record; ``record.result`` is the
            :class:`~probstat.inverse.RegularizationPath`.

        Raises:
            ValueError: If the study is not linear, a rule is unknown, or an
                option is invalid.

        References:
            [EQ-INV-SEL-001], [EQ-INV-SEL-002], [REF-HANSEN-2008],
            [REF-MOROZOV-1966].

        """
        self._require_linear("regularize")
        chosen = tuple(rules)
        if not chosen or any(rule not in SELECTION_RULES for rule in chosen):
            raise ValueError(f"rules must be a non-empty subset of {SELECTION_RULES}")
        count = _integer(points, "points", 3)
        assert self._linear is not None and self._counted is not None
        start = self._begin("regularize")
        if grid is None:
            if self._sigma1 is None:
                self._sigma1 = spectral_diagnostics(
                    self._counted, noise=self._linear.noise
                ).singular_values[0]
            values = logarithmic_grid(1e-5 * self._sigma1, 2.0 * self._sigma1, count)
        else:
            values = as_vector(grid, name="grid")
        path = regularization_path(
            self._linear,
            values,
            trace=trace,
            probes=probes,
            seed=start.stream.child("trace"),
            noise_norm=math.sqrt(self._linear.shape[0]),
        )
        rows: list[tuple[str, ...]] = []
        summary: dict[str, float | str] = {}
        warnings: list[str] = []
        for rule in chosen:
            try:
                pick = select_regularization(path, rule)
            except ValueError as error:
                rows.append((rule, "-", "unavailable", "-", "-", "-"))
                warnings.append(f"rule {rule!r} unavailable: {error}")
                summary[f"lambda ({rule})"] = "unavailable"
                continue
            if pick.index is None or pick.regularization is None:
                rows.append((rule, "-", pick.status, "-", "-", "-"))
                warnings.append(f"rule {rule!r} found no valid lambda ({pick.status})")
                summary[f"lambda ({rule})"] = pick.status
                continue
            i = pick.index
            rows.append(
                (
                    rule,
                    _fmt(pick.regularization),
                    pick.status,
                    _fmt(path.misfits[i]),
                    _fmt(path.penalty_norms[i]),
                    _fmt(path.degrees_of_freedom[i]),
                )
            )
            summary[f"lambda ({rule})"] = pick.regularization
            if pick.status in ("boundary", "always_met", "flat"):
                warnings.append(
                    f"rule {rule!r} status {pick.status!r}: the optimum is not "
                    "interior; extend or refine the grid before trusting lambda"
                )
        summary["grid points"] = float(len(values))
        summary["lambda range"] = f"{_fmt(values[0])} .. {_fmt(values[-1])}"
        table = ReportTable(
            "Rule choices",
            ("rule", "lambda", "status", "misfit", "penalty norm", "dof"),
            tuple(rows),
        )
        return self._finish(
            start,
            "Tikhonov path and regularization-parameter rules",
            summary,
            path,
            settings={
                "rules": ", ".join(chosen),
                "trace": trace,
                "grid": "user" if grid is not None else f"log, {count} points",
            },
            tables=(table,),
            warnings=warnings,
        )

    # ----- MAP and Laplace -------------------------------------------------
    def map(
        self,
        *,
        method: str = "gauss_newton",
        initial: Sequence[float] | None = None,
        max_iterations: int = 100,
        gradient_tolerance: float = 1e-9,
    ) -> InverseStageRecord:
        """Return the record of the MAP point.

        Minimizes ``1/2 ||W (G(x) - y)||^2 + 1/2 ||z||^2`` in whitened
        coordinates with the Gauss-Newton-Krylov or trust-region solver; the
        MAP is stored and reused by :meth:`laplace`, :meth:`sample` and
        :meth:`resolution`. A nonlinear objective may have several minima:
        the result is a local one.

        Args:
            method: ``"gauss_newton"`` or ``"trust_region"``.
            initial: Start in whitened coordinates (default the prior mean).
            max_iterations: Iteration cap.
            gradient_tolerance: Relative gradient tolerance.

        Returns:
            The stage record; ``record.result`` is a
            :class:`~probstat.inverse.MapEstimate`.

        Raises:
            ValueError: For an invalid method or option.

        References:
            [EQ-INV-LAPX-002], [REF-NOCEDAL-WRIGHT-2006].

        """
        start = self._begin("map")
        estimate = compute_map(
            self._bayes,
            method=method,  # type: ignore[arg-type]
            initial=initial,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )
        self._map = estimate
        warnings: list[str] = []
        if not estimate.converged:
            warnings.append(
                f"MAP solver did not converge (stop reason {estimate.stop_reason!r})"
            )
        summary: dict[str, float | str] = {
            "method": estimate.method,
            "objective": estimate.objective,
            "gradient norm": estimate.gradient_norm,
            "iterations": float(estimate.iterations),
            "converged": str(estimate.converged),
            "stop reason": estimate.stop_reason,
        }
        table = self._coordinate_table("MAP estimate", {"x_map": estimate.physical})
        return self._finish(
            start,
            "Maximum a posteriori point",
            summary,
            estimate,
            settings={
                "method": method,
                "max_iterations": max_iterations,
                "gradient_tolerance": gradient_tolerance,
            },
            tables=(table,),
            warnings=warnings,
        )

    def laplace(
        self,
        *,
        threshold: float = 1e-2,
        max_rank: int | None = None,
        lis_method: str = "randomized",
        marginal_probes: int | None = None,
    ) -> InverseStageRecord:
        """Return the record of the Laplace approximation.

        Uses the stored MAP if :meth:`map` ran (otherwise the MAP is computed
        inside this stage and its cost is charged here). The covariance is
        ``P (I + V diag(lambda) V')^-1 P'`` with the dominant eigenpairs of
        the Gauss-Newton Hessian; the log-evidence follows
        [EQ-INV-LAPX-004]. The eigenpairs come from the matrix-free
        likelihood-informed subspace solver and omit directions with
        ``lambda`` below ``threshold``; the omitted part biases the evidence
        upward by at most the reported truncation bound.

        Args:
            threshold: Eigenvalue threshold of the retained subspace.
            max_rank: Optional cap on the retained rank.
            lis_method: ``"randomized"`` or ``"lanczos"``.
            marginal_probes: ``None`` for exact marginal variances (``n``
                adjoint applications of ``P``); otherwise the number of
                Rademacher probes (``>= 2``). Above ``n = 2000`` the default
                switches to 64 probes.

        Returns:
            The stage record; ``record.result`` is a
            :class:`~probstat.inverse.LaplaceApproximation`.

        Raises:
            ValueError: For an invalid option.

        References:
            [EQ-INV-LAPX-003], [EQ-INV-LAPX-004], [EQ-INV-LAPX-005],
            [REF-BUI-THANH-ETAL-2013].

        """
        if lis_method not in ("randomized", "lanczos"):
            raise ValueError("lis_method must be 'randomized' or 'lanczos'")
        start = self._begin("laplace")
        approx = laplace_approximation(
            self._bayes,
            threshold=threshold,
            max_rank=max_rank,
            map_estimate=self._map,
            lis_method=lis_method,  # type: ignore[arg-type]
            seed=start.stream.child("lis"),
        )
        self._laplace = approx
        warnings: list[str] = []
        if self._map is None:
            self._map = approx.map_estimate
            warnings.append("the MAP was computed inside this stage")
        n = self._bayes.physical_dimension
        probes = marginal_probes
        if probes is None and n > _DENSE_LIMIT:
            probes = 64
        variances = approx.marginal_variances(
            probes=probes, seed=start.stream.child("probes")
        )
        sd = tuple(math.sqrt(max(v, 0.0)) for v in variances.values)
        if not approx.converged:
            warnings.append(
                "Laplace approximation not converged "
                f"(MAP converged: {approx.map_estimate.converged}; eigensolver "
                f"{approx.lis_stop_reason!r}); the rank is not decided"
            )
        if self.kind == "nonlinear":
            warnings.append(
                "the Laplace covariance uses the Gauss-Newton Hessian at the MAP: "
                "a Gaussian cannot represent skewness, curvature or multiple modes"
            )
        else:
            warnings.append(
                "exact for a linear model up to the omitted directions with "
                f"lambda below {_fmt(approx.threshold)}"
            )
        bound = approx.log_evidence_truncation_bound
        if math.isnan(bound):
            warnings.append(
                "the eigensolver did not bracket the threshold: the evidence "
                "truncation bound is unavailable"
            )
        summary: dict[str, float | str] = {
            "retained rank": float(approx.rank),
            "eigenvalue threshold": approx.threshold,
            "largest eigenvalue": approx.eigenvalues[0] if approx.eigenvalues else 0.0,
            "log evidence": approx.log_evidence,
            "evidence truncation bound": bound,
            "misfit 1/2||W(G(x_map)-y)||^2": approx.misfit,
            "converged": str(approx.converged),
            "marginal variances": variances.method,
            "MAP work": float(approx.counts_map.work),
            "Hessian work": float(approx.counts_hessian.work),
        }
        columns: dict[str, Sequence[float]] = {
            "mean": approx.mean,
            "sd": sd,
        }
        if variances.method == "probing":
            columns["sd of variance"] = variances.standard_errors
        table = self._coordinate_table("Laplace marginal posterior", columns)
        return self._finish(
            start,
            "Laplace approximation at the MAP",
            summary,
            approx,
            settings={
                "threshold": threshold,
                "max_rank": "none" if max_rank is None else max_rank,
                "lis_method": lis_method,
                "marginal_probes": "exact" if probes is None else probes,
            },
            tables=(table,),
            warnings=warnings,
        )

    # ----- sampling --------------------------------------------------------
    def sample(
        self,
        method: str = "dili",
        *,
        draws: int = 1000,
        burn_in: int = 500,
        particles: int = 500,
        thin: int = 1,
        options: Mapping[str, object] | None = None,
    ) -> InverseStageRecord:
        """Return the record of a posterior sampling run.

        ``"pcn"``, ``"gpcn"`` and ``"dili"`` are the chains of
        :mod:`probstat.inverse` (the last two use the Laplace approximation,
        which runs as its own recorded stage first if :meth:`laplace` has not
        been called); ``"rto"`` is randomize-then-optimize with the
        independence Metropolis correction; ``"smc"`` is tempered SMC and
        ``"tmcmc"`` transitional MCMC. The ESS of every coordinate is the
        split-chain estimate of the stored draws and the MCSE of its mean is
        ``sd / sqrt(ESS)`` [EQ-WF-INVSTUDY-003]. Particle methods give no
        ESS; their MCSE is the iid-naive ``sd / sqrt(N)`` and understates the
        error.

        Args:
            method: One of :data:`SAMPLE_METHODS`.
            draws: Post-burn-in iterations of a chain (``>= 1``); proposals of
                ``"rto"`` (of which ``burn_in`` leading draws are dropped).
            burn_in: Adapted and discarded iterations.
            particles: Particles of ``"smc"`` and ``"tmcmc"`` (``>= 2``).
            thin: Keep every ``thin``-th chain draw.
            options: Extra keyword options of the underlying sampler (for
                example ``{"beta": 0.3}``); unknown names raise ``TypeError``.

        Returns:
            The stage record; ``record.result`` is a :class:`SampleSummary`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``method`` is not in :data:`SAMPLE_METHODS`, a
                count is invalid, or the sampler rejects its inputs (for
                example ``"dili"`` with a Laplace approximation of rank 0).

        References:
            [EQ-INV-LAPX-009], [EQ-INV-LAPX-010], [EQ-WF-INVSTUDY-003],
            [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-GEYER-1992].

        """
        if method not in SAMPLE_METHODS:
            raise ValueError(f"method must be one of {SAMPLE_METHODS}")
        n_draws = _integer(draws, "draws", 1)
        n_burn = _integer(burn_in, "burn_in", 0)
        n_particles = _integer(particles, "particles", 2)
        stride = _integer(thin, "thin", 1)
        extra: dict[str, Any] = dict(options or {})
        prerequisites: list[str] = []
        if method in ("gpcn", "dili") and self._laplace is None:
            self.laplace()
            prerequisites.append("laplace() was run first as its own stage")
        start = self._begin("sample")
        problem = self._bayes
        stream = start.stream
        details: Any
        accept = math.nan
        log_evidence = math.nan
        converged = True
        reason = "completed"
        if method == "pcn":
            details = pcn_chain(
                problem,
                draws=n_draws,
                burn_in=n_burn,
                thin=stride,
                seed=stream,
                **extra,
            )
        elif method == "gpcn":
            assert self._laplace is not None
            details = gpcn_chain(
                problem,
                self._laplace,
                draws=n_draws,
                burn_in=n_burn,
                thin=stride,
                seed=stream,
                **extra,
            )
        elif method == "dili":
            assert self._laplace is not None
            details = dili_chain(
                problem,
                self._laplace,
                draws=n_draws,
                burn_in=n_burn,
                thin=stride,
                seed=stream,
                **extra,
            )
        elif method == "rto":
            details = randomize_then_optimize(
                problem,
                draws=n_draws,
                reference=None if self._map is None else self._map.whitened,
                correction="metropolis",
                seed=stream,
                **extra,
            )
        elif method == "smc":
            constant = problem.log_likelihood_constant
            r = problem.dimension
            details = tempered_smc_posterior(
                lambda g: tuple(g.gauss(0.0, 1.0) for _ in range(r)),
                lambda z: problem.log_likelihood(z) + constant,
                particles=n_particles,
                to_physical=problem.to_physical,
                seed=stream,
                **extra,
            )
        else:
            constant = problem.log_likelihood_constant
            r = problem.dimension
            details = transitional_mcmc(
                lambda g: tuple(g.gauss(0.0, 1.0) for _ in range(r)),
                lambda z: -0.5 * sum(v * v for v in z),
                lambda z: problem.log_likelihood(z) + constant,
                particles=n_particles,
                seed=stream,
                **extra,
            )
        warnings: list[str] = list(prerequisites)
        if isinstance(details, ChainResult):
            raw = details.samples
            accept = details.acceptance_rate
            if details.nonfinite_rejections:
                warnings.append(
                    f"{details.nonfinite_rejections} proposals were rejected for a "
                    "failed forward or adjoint evaluation"
                )
        elif method == "rto":
            raw = tuple(details.samples)[n_burn:]
            accept = details.acceptance_rate
            if details.solver_failures:
                warnings.append(
                    f"{details.solver_failures} of {details.proposals} RTO solves "
                    "failed and were rejected; the chain is biased if this is not tiny"
                )
        elif method == "smc":
            converged = bool(details.converged)
            reason = str(details.stop_reason)
            raw = tuple(details.samples) if converged else ()
            log_evidence = details.log_evidence
            if converged:
                accept = sum(details.acceptance_rates) / len(details.acceptance_rates)
        else:
            converged = bool(details.converged)
            reason = str(details.stop_reason)
            raw = (
                tuple(problem.to_physical(z) for z in details.particles)
                if converged
                else ()
            )
            log_evidence = details.log_evidence
            if converged:
                accept = sum(details.acceptance_rates) / len(details.acceptance_rates)
        if not converged:
            warnings.append(f"sampler did not finish: {reason}")
        if len(raw) < 2 and converged:
            raise ValueError("fewer than two samples were retained; increase draws")
        if raw:
            mean, variance = sample_moments(raw)
        else:
            nan_vector = (math.nan,) * problem.physical_dimension
            mean, variance = nan_vector, nan_vector
        n = problem.physical_dimension
        particle = method in _PARTICLE_METHODS
        if raw and not particle:
            ess = tuple(_series_ess(raw, i) for i in range(n))
            mcse = tuple(
                math.sqrt(variance[i] / ess[i])
                if ess[i] > 0.0 and not math.isnan(ess[i])
                else math.nan
                for i in range(n)
            )
            kind = "ess"
        else:
            ess = (math.nan,) * n
            mcse = tuple(math.sqrt(v / len(raw)) if raw else math.nan for v in variance)
            kind = "iid-naive"
        work = self._snapshot().minus(start.counts).work
        finite = [v for v in ess if not math.isnan(v)]
        min_ess = min(finite) if finite else math.nan
        ess_per_work = min_ess / work if finite and work else math.nan
        exact = method in _CHAIN_METHODS
        if method == "rto":
            warnings.append(
                "RTO with the Metropolis correction is an independence "
                "Metropolis-Hastings chain; consecutive draws repeat when "
                "proposals are rejected"
            )
        if particle:
            warnings.append(
                "particle method: a finite-particle approximation; no ESS is "
                "available and the MCSE shown is the iid-naive sd/sqrt(N), which "
                "understates the error; replicate over seeds"
            )
        elif raw:
            warnings.append(
                "single chain: the split-chain ESS is noisy for short runs and "
                "there is no between-chain R-hat"
            )
        if finite and min_ess < 100.0:
            warnings.append(
                f"smallest ESS is {_fmt(min_ess)} (< 100): moment and MCSE "
                "estimates are unreliable; lengthen the run"
            )
        result = SampleSummary(
            method,
            exact,
            converged,
            reason,
            tuple(raw),
            mean,
            variance,
            ess,
            mcse,
            kind,
            accept,
            log_evidence,
            work,
            ess_per_work,
            details,
        )
        summary: dict[str, float | str] = {
            "method": method,
            "stored samples": float(len(raw)),
            "acceptance": accept,
            "min ESS": min_ess,
            "ESS per work": ess_per_work,
            "work": float(work),
            "max MCSE / sd": max(
                (
                    mcse[i] / math.sqrt(variance[i])
                    for i in range(n)
                    if variance[i] > 0.0 and not math.isnan(mcse[i])
                ),
                default=math.nan,
            ),
            "log evidence": log_evidence,
            "converged": str(converged),
        }
        table = self._coordinate_table(
            "Posterior samples",
            {
                "mean": mean,
                "sd": tuple(math.sqrt(v) for v in variance),
                "ESS": ess,
                "MCSE of mean": mcse,
            },
        )
        return self._finish(
            start,
            f"Posterior samples ({method})",
            summary,
            result,
            settings={
                "method": method,
                "draws": n_draws,
                "burn_in": n_burn,
                "particles": n_particles,
                "thin": stride,
                "options": ", ".join(f"{k}={v}" for k, v in sorted(extra.items()))
                or "defaults",
                "mcse": kind,
            },
            tables=(table,),
            warnings=warnings,
        )

    # ----- resolution ------------------------------------------------------
    def resolution(
        self,
        *,
        indices: Sequence[int] | None = None,
        trace_probes: int = 30,
        lis_threshold: float = 1.0,
        cg_tolerance: float = 1e-10,
    ) -> InverseStageRecord:
        """Return the record of matrix-free resolution probes and LIS rank.

        Probes the resolution operator ``R = P (I + H)^-1 P' J' Gamma^-1 J``
        through conjugate-gradient solves and counts the likelihood-informed
        directions (eigenvalues of ``H`` above ``lis_threshold``) without
        forming an ``n x n`` or ``r x r`` matrix. ``R_jj`` is the averaging
        kernel's value at its own node; ``R_jj`` near 1 and a peak at ``j``
        mean parameter ``j`` is resolved, ``R_jj`` near 0 means the data say
        nothing about it beyond the prior. For a nonlinear problem the
        Jacobian is taken at the MAP if :meth:`map` ran, else at the prior
        mean.

        Args:
            indices: Parameter indices to probe (default up to five spread
                over ``0..n-1``).
            trace_probes: Hutchinson probes for ``tr R`` (``0`` skips it).
            lis_threshold: Eigenvalue threshold of the LIS rank.
            cg_tolerance: Relative CG tolerance.

        Returns:
            The stage record; ``record.result`` is a
            :class:`ResolutionResult`.

        Raises:
            ValueError: If an index is out of range or an option is invalid.

        References:
            [EQ-INV-LIS-014], [EQ-INV-LIS-017], [EQ-INV-LIS-018],
            [EQ-INV-LIS-019], [REF-TARANTOLA-2005].

        """
        bayes = self._bayes
        n = bayes.physical_dimension
        picks = _spread_indices(n, 5) if indices is None else tuple(indices)
        if not picks or any(
            isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < n
            for j in picks
        ):
            raise ValueError(f"indices must be integers in [0, {n - 1}]")
        probes = _integer(trace_probes, "trace_probes", 0)
        tau = _real(lis_threshold, "lis_threshold", positive=True)
        start = self._begin("resolution")
        point, where = self._linearization()
        forward: object
        if self.kind == "linear":
            assert self._counted is not None
            forward = self._counted
        else:
            forward = bayes.jacobian_operator(point)
        probe = probe_resolution(
            forward,
            noise=bayes.noise,
            prior_factor=bayes.prior_factor,
            column_indices=picks,
            trace_probes=probes,
            cg_tolerance=cg_tolerance,
            seed=start.stream.child("trace"),
        )
        lis = matrix_free_lis(
            [forward],
            noise=bayes.noise,
            prior_factor=bayes.prior_factor,
            threshold=tau,
            seed=start.stream.child("lis"),
        )
        diagonal = tuple(
            probe.columns[k][j] for k, j in enumerate(probe.column_indices)
        )
        peaks = tuple(
            max(range(n), key=lambda i, c=column: abs(c[i]))  # type: ignore[misc]
            for column in probe.columns
        )
        peak_value = tuple(
            probe.columns[k][peaks[k]] for k in range(len(probe.columns))
        )
        trace = probe.trace.estimate if probe.trace is not None else math.nan
        trace_se = probe.trace.standard_error if probe.trace is not None else math.nan
        warnings: list[str] = []
        if not probe.all_converged:
            warnings.append(
                f"{probe.nonconverged_count} conjugate-gradient solves did not "
                "reach the tolerance; the probes are not trustworthy"
            )
        if not lis.converged:
            warnings.append(
                f"the LIS rank is not decided (stop reason {lis.stop_reason!r}); "
                "do not use it as is"
            )
        if self.kind == "nonlinear":
            warnings.append(
                f"resolution of the linearization at the {where}; it describes "
                "the local linear estimator, not the nonlinear posterior"
            )
        result = ResolutionResult(
            tuple(probe.column_indices),
            diagonal,
            peaks,
            peak_value,
            trace,
            trace_se,
            lis.rank,
            tau,
            lis.converged,
            lis.stop_reason,
            probe.solve_count,
            probe.all_converged,
            where,
        )
        summary: dict[str, float | str] = {
            "linearization point": where,
            "LIS rank": float(lis.rank),
            "LIS threshold": tau,
            "LIS stop reason": lis.stop_reason,
            "tr R": trace,
            "tr R standard error": trace_se,
            "CG solves": float(probe.solve_count),
            "all solves converged": str(probe.all_converged),
        }
        rows = tuple(
            (
                str(j),
                _fmt(diagonal[k]),
                str(peaks[k]),
                _fmt(peak_value[k]),
            )
            for k, j in enumerate(probe.column_indices)
        )
        table = ReportTable(
            "Averaging kernels (columns of R)",
            ("j", "R_jj", "peak index", "peak value"),
            rows,
        )
        return self._finish(
            start,
            "Resolution and likelihood-informed subspace",
            summary,
            result,
            settings={
                "indices": ", ".join(str(j) for j in picks),
                "trace_probes": probes,
                "lis_threshold": tau,
                "cg_tolerance": cg_tolerance,
            },
            tables=(table,),
            warnings=warnings,
        )

    # ----- comparison ------------------------------------------------------
    def _quadrature_reference(
        self,
        bounds: Sequence[tuple[float, float]] | None,
        options: Mapping[str, object] | None,
    ) -> _Reference:
        """Return the physical moments of a quadrature of the posterior of ``z``.

        With ``x = mu + P z``: ``mean_x = mu + P mean_z`` and
        ``var_x_i = (P C_z P')_ii`` [EQ-WF-INVSTUDY-005]. The reference is
        cached per box and option set; its cost is charged to the first
        :meth:`compare` that needs it.

        Raises:
            ValueError: If ``r > 2``.

        References:
            [EQ-WF-INVSTUDY-005], [EQ-INV-SMP-004], [EQ-INV-SMP-005].

        """
        bayes = self._bayes
        r = bayes.dimension
        if r > 2:
            raise ValueError(
                f"the quadrature reference needs at most two whitened parameters, "
                f"r = {r}; pass reference=(mean, variance) or run a long chain"
            )
        box = (
            tuple(bounds)
            if bounds is not None
            else ((-_QUADRATURE_HALF_WIDTH, _QUADRATURE_HALF_WIDTH),) * r
        )
        settings: dict[str, Any] = {"panels": 16 if r == 2 else 64}
        settings.update(options or {})
        key = repr((box, sorted(settings.items())))
        cached = self._quadrature.get(key)
        if cached is not None:
            return cached
        ref = quadrature_reference(bayes.log_posterior, box, **settings)
        factor = materialize(bayes.prior_factor, max_entries=4_000_000)
        mu = bayes.prior_mean
        n = bayes.physical_dimension
        mean = tuple(
            mu[i] + sum(factor[i][j] * ref.mean[j] for j in range(r)) for i in range(n)
        )
        variance = tuple(
            sum(
                factor[i][j] * ref.covariance[j][k] * factor[i][k]
                for j in range(r)
                for k in range(r)
            )
            for i in range(n)
        )
        log_evidence = (
            ref.log_normalizer + bayes.log_likelihood_constant - 0.5 * r * _LOG_TWO_PI
        )
        note = (
            f"tensor Gauss-Legendre on {box}, panels {ref.panels}, order "
            f"{ref.order}; truncation mass estimate "
            f"{_fmt(ref.truncation_mass_estimate)}, refinement change of log Z "
            f"{_fmt(ref.refinement_log_error)}"
        )
        if ref.warnings:
            note += "; WARNINGS: " + " | ".join(ref.warnings)
        built = _Reference("quadrature", mean, variance, log_evidence, note)
        self._quadrature[key] = built
        return built

    def compare(
        self,
        reference: object = "auto",
        *,
        method: str | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        quadrature_options: Mapping[str, object] | None = None,
        mean_tolerance_sd: float = 0.25,
        variance_tolerance: float = 0.3,
        mcse_multiplier: float = 3.0,
    ) -> InverseStageRecord:
        """Return the record of a comparison of sample moments with a reference.

        Takes the latest converged sample stage (of ``method`` if given) and
        reports per coordinate the mean error in reference standard
        deviations, the mean error in MCSE units, and the variance ratio
        [EQ-WF-INVSTUDY-004]. A coordinate passes when
        ``|mean error| <= mean_tolerance_sd * sd_ref + mcse_multiplier * MCSE``
        and ``|variance ratio - 1| <= variance_tolerance``. The defaults
        match :func:`~probstat.inverse.compare_to_reference`; the variance
        ratio of an ESS-sized sample has a relative error of about
        ``sqrt(2 / ESS)`` for a Gaussian target, so choose
        ``variance_tolerance`` above that. If the Laplace stage ran, the
        Laplace mean and marginal variances are compared with the same
        reference.

        Args:
            reference: ``"auto"`` (exact for a linear problem, quadrature for
                at most two whitened parameters), ``"exact"`` (linear only),
                ``"quadrature"``, or user moments as ``(mean, variance)`` /
                ``{"mean": ..., "variance": ...}`` of the physical ``x``
                (marginal variances).
            method: Compare the latest sample stage of this method.
            bounds: Whitened box of the quadrature (default ``+-8`` per
                coordinate; the whitened posterior is no wider than the
                standard normal prior times the likelihood).
            quadrature_options: Extra options of
                :func:`~probstat.inverse.quadrature_reference`.
            mean_tolerance_sd: Mean tolerance in reference sd (``> 0``).
            variance_tolerance: Variance-ratio tolerance (``> 0``).
            mcse_multiplier: MCSE allowance multiple (``>= 0``).

        Returns:
            The stage record; ``record.result`` is a :class:`StudyComparison`.

        Raises:
            ValueError: If no converged sample stage exists, the reference is
                unknown or unavailable for this problem, user moments have the
                wrong shape or non-positive variances, or an option is
                invalid.

        References:
            [EQ-WF-INVSTUDY-004], [EQ-WF-INVSTUDY-005], [EQ-INV-SMP-006],
            [REF-VEHTARI-ETAL-2021].

        """
        tol_mean = _real(mean_tolerance_sd, "mean_tolerance_sd", positive=True)
        tol_var = _real(variance_tolerance, "variance_tolerance", positive=True)
        multiplier = _real(mcse_multiplier, "mcse_multiplier")
        if multiplier < 0.0:
            raise ValueError("mcse_multiplier must be non-negative")
        if method is not None and method not in SAMPLE_METHODS:
            raise ValueError(f"method must be one of {SAMPLE_METHODS}")
        sample_record: InverseStageRecord | None = None
        for record in reversed(self._records):
            if record.stage != "sample":
                continue
            summary_result = record.result
            if method is not None and summary_result.method != method:
                continue
            if summary_result.converged:
                sample_record = record
                break
        if sample_record is None:
            raise ValueError(
                "compare() needs samples: run sample() first"
                + ("" if method is None else f" with method {method!r}")
                + " and make sure the sampler converged"
            )
        samples: SampleSummary = sample_record.result
        kind: str
        user: tuple[Vector, Vector] | None = None
        if isinstance(reference, str):
            if reference not in REFERENCE_KINDS or reference == "moments":
                raise ValueError(
                    "reference must be 'auto', 'exact', 'quadrature' or moments"
                )
            kind = reference
        else:
            kind = "moments"
            user = self._user_moments(reference)
        if kind == "auto":
            if self.kind == "linear":
                kind = "exact"
            elif self._bayes.dimension <= 2:
                kind = "quadrature"
            else:
                raise ValueError(
                    "no automatic reference for a nonlinear problem with more than "
                    "two parameters: pass reference=(mean, variance)"
                )
        if kind == "exact" and self.kind != "linear":
            raise ValueError("the exact reference needs a linear problem")
        start = self._begin("compare")
        if kind == "exact":
            exact = self._exact_posterior()
            ref = _Reference(
                "exact",
                exact.mean,
                exact.marginal_variances(),
                exact.log_evidence,
                "closed-form linear-Gaussian posterior",
            )
        elif kind == "quadrature":
            ref = self._quadrature_reference(bounds, quadrature_options)
        else:
            assert user is not None
            ref = _Reference(
                "moments", user[0], user[1], math.nan, "user-supplied moments"
            )
        n = self._bayes.physical_dimension
        sd_ref = tuple(math.sqrt(v) for v in ref.variance)
        diff = tuple(abs(samples.mean[i] - ref.mean[i]) for i in range(n))
        mean_error_sd = tuple(diff[i] / sd_ref[i] for i in range(n))
        mean_error_mcse = tuple(
            (
                diff[i] / samples.mcse[i]
                if samples.mcse[i] > 0.0
                else (math.inf if diff[i] > 0.0 else 0.0)
            )
            if not math.isnan(samples.mcse[i])
            else math.nan
            for i in range(n)
        )
        ratio = tuple(samples.variance[i] / ref.variance[i] for i in range(n))
        failures: list[str] = []
        for i in range(n):
            allowance = tol_mean * sd_ref[i] + multiplier * (
                0.0 if math.isnan(samples.mcse[i]) else samples.mcse[i]
            )
            if diff[i] > allowance:
                failures.append(
                    f"coordinate {i}: mean error {_fmt(mean_error_sd[i])} sd exceeds "
                    f"the allowance {_fmt(allowance / sd_ref[i])} sd"
                )
            if abs(ratio[i] - 1.0) > tol_var:
                failures.append(
                    f"coordinate {i}: variance ratio {_fmt(ratio[i])} outside "
                    f"1 +- {_fmt(tol_var)}"
                )
        laplace_error: Vector | None = None
        laplace_ratio: Vector | None = None
        warnings: list[str] = [ref.note] if ref.kind == "quadrature" else []
        if self._laplace is not None:
            lap_var = self._laplace.marginal_variances().values
            laplace_error = tuple(
                abs(self._laplace.mean[i] - ref.mean[i]) / sd_ref[i] for i in range(n)
            )
            laplace_ratio = tuple(lap_var[i] / ref.variance[i] for i in range(n))
        if samples.mcse_kind == "iid-naive":
            warnings.append(
                "the MCSE of a particle method is the iid-naive sd/sqrt(N); the "
                "mean allowance is therefore optimistic"
            )
        if not samples.asymptotically_exact:
            warnings.append(
                "the compared sampler is a finite-particle approximation; a pass "
                "is evidence for this run only"
            )
        passed = not failures
        result = StudyComparison(
            samples.method,
            ref.kind,
            ref.mean,
            ref.variance,
            mean_error_sd,
            mean_error_mcse,
            ratio,
            tol_mean,
            tol_var,
            multiplier,
            passed,
            tuple(failures),
            laplace_error,
            laplace_ratio,
            ref.log_evidence,
            ref.note,
        )
        columns: dict[str, Sequence[float]] = {
            "ref mean": ref.mean,
            "sample mean": samples.mean,
            "error (sd)": mean_error_sd,
            "error (MCSE)": mean_error_mcse,
            "var ratio": ratio,
        }
        if laplace_error is not None and laplace_ratio is not None:
            columns["Laplace error (sd)"] = laplace_error
            columns["Laplace var ratio"] = laplace_ratio
        table = self._coordinate_table("Moment comparison", columns)
        summary: dict[str, float | str] = {
            "sample stage": f"sample #{sample_record.run} ({samples.method})",
            "reference": ref.kind,
            "max mean error (sd)": max(mean_error_sd),
            "max |variance ratio - 1|": max(abs(v - 1.0) for v in ratio),
            "passed": str(passed),
            "reference log evidence": ref.log_evidence,
        }
        if not math.isnan(ref.log_evidence):
            if not math.isnan(samples.log_evidence):
                summary["sampler log evidence error"] = (
                    samples.log_evidence - ref.log_evidence
                )
            if self._laplace is not None:
                summary["Laplace log evidence error"] = (
                    self._laplace.log_evidence - ref.log_evidence
                )
        if failures:
            warnings.extend(failures)
        return self._finish(
            start,
            f"Comparison with the {ref.kind} reference",
            summary,
            result,
            settings={
                "mean_tolerance_sd": tol_mean,
                "variance_tolerance": tol_var,
                "mcse_multiplier": multiplier,
                "reference": ref.kind,
            },
            tables=(table,),
            warnings=warnings,
        )

    def _user_moments(self, reference: object) -> tuple[Vector, Vector]:
        """Return validated user moments ``(mean, variance)``.

        Raises:
            ValueError: For a wrong shape or non-positive variances.
            TypeError: For an unsupported container.

        References:
            [EQ-WF-INVSTUDY-004].

        """
        n = self._bayes.physical_dimension
        if isinstance(reference, Mapping):
            if set(reference) != {"mean", "variance"}:
                raise ValueError("moments mapping needs exactly 'mean' and 'variance'")
            raw_mean, raw_var = reference["mean"], reference["variance"]
        elif isinstance(reference, tuple | list) and len(reference) == 2:
            raw_mean, raw_var = reference
        else:
            raise TypeError(
                "reference must be a string, a (mean, variance) pair or a mapping"
            )
        mean = as_vector(raw_mean, name="reference mean")
        variance = as_vector(raw_var, name="reference variance")
        if len(mean) != n or len(variance) != n:
            raise ValueError(f"reference moments must have length n = {n}")
        if any(v <= 0.0 for v in variance):
            raise ValueError("reference variances must be positive")
        return mean, variance

    # ----- report ----------------------------------------------------------
    def _noise_text(self) -> str:
        """Return a one-line description of the noise model.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        noise: Any = self._bayes.noise
        m = noise.dimension
        shown = min(m, 3)
        variances = [noise.apply(_unit(m, i))[i] for i in range(shown)]
        return (
            f"{type(noise).__name__} (Gaussian, dimension {m}); variances of the "
            f"first {shown} observations {_fmt_vector(variances, 3)}; "
            f"log det Gamma = {_fmt(noise.log_det())}"
        )

    def _prior_text(self) -> str:
        """Return a one-line description of the Gaussian prior.

        References:
            [EQ-WF-INVSTUDY-002].

        """
        bayes = self._bayes
        factor = bayes.prior_factor
        n, r = factor.shape
        shown = min(n, 3)
        sd = [
            math.sqrt(sum(t * t for t in factor.rmatvec(_unit(n, i))))
            for i in range(shown)
        ]
        mean = bayes.prior_mean
        return (
            f"Gaussian N(mu, P P') with P of shape {n} x {r}; prior sd of the "
            f"first {shown} parameters {_fmt_vector(sd, 3)}; mu = "
            f"{_fmt_vector(mean, 3)}"
        )

    def _caveats(self) -> list[str]:
        """Return the caveats that apply to the stages that ran.

        References:
            [EQ-WF-INVSTUDY-002], [REF-BUI-THANH-ETAL-2013].

        """
        ran = {r.stage for r in self._records}
        notes = [
            "All results are conditional on the stated prior and Gaussian noise "
            "model; none of the stages checks the model against real data.",
        ]
        if "laplace" in ran:
            if self.kind == "nonlinear":
                notes.append(
                    "The Laplace approximation is a Gaussian at the MAP built from "
                    "the Gauss-Newton Hessian (second derivatives of the forward "
                    "map are dropped); it cannot represent skewness, banana shapes "
                    "or several modes, and its evidence is only as good as that "
                    "Gaussian."
                )
            notes.append(
                "The Laplace log-evidence omits 1/2 sum log(1 + lambda) over the "
                "unresolved eigenvalues and is biased upward by at most the "
                "reported truncation bound (when it is finite)."
            )
        for record in self._records:
            if record.stage != "sample":
                continue
            method = str(record.summary.get("method", ""))
            if method in _PARTICLE_METHODS:
                notes.append(
                    f"{method}: a finite-particle approximation without a valid "
                    "ESS; the reported MCSE is the iid-naive value and understates "
                    "the error."
                )
            elif method == "rto":
                notes.append(
                    "rto: independence Metropolis-Hastings on randomize-then-"
                    "optimize proposals; exact in the limit, but its efficiency "
                    "depends on the proposal quality and failed solves bias it."
                )
            elif method:
                notes.append(
                    f"{method}: an exact MCMC for the stated target; the run is "
                    "one finite chain, so ESS and MCSE are single-chain estimates."
                )
        if "resolution" in ran and self.kind == "nonlinear":
            notes.append(
                "Resolution is that of the local linearization, not of the "
                "nonlinear posterior."
            )
        if "compare" in ran:
            notes.append(
                "A comparison passes or fails at stated tolerances for the stored "
                "sample; it is a screening check, not a convergence proof."
            )
        return sorted(set(notes), key=notes.index)

    def report(self, *, include_timings: bool = False) -> str:
        """Return the Markdown report of every stage that ran.

        The header records the root seed, the problem sizes, the noise model
        and the prior; then comes one section per stage run in execution
        order (settings, headline numbers, tables, warnings, evaluation
        counts, stage seed), the evaluation ledger and the caveats. Without
        ``include_timings`` the text contains no wall times and is identical
        for two runs with the same seed.

        Args:
            include_timings: Add the wall time of every stage.

        Returns:
            The report text.

        References:
            [EQ-WF-INVSTUDY-002], [REF-KAIPIO-SOMERSALO-2007].

        """
        bayes = self._bayes
        kind_text = (
            "linear-Gaussian" if self.kind == "linear" else "nonlinear forward map"
        )
        lines = [
            f"# {self._label}",
            "",
            f"- Kind: {self.kind} Bayesian inverse problem ({kind_text})",
            f"- Root seed: {self.seed} (stage streams: root.child(stage, run))",
            f"- Sizes: data m = {bayes.data_dimension}, parameters n = "
            f"{bayes.physical_dimension}, whitened coordinates r = {bayes.dimension}",
            f"- Noise model: {self._noise_text()}",
            f"- Prior: {self._prior_text()}",
            "- Stages run: "
            + (
                ", ".join(f"{r.stage} #{r.run}" for r in self._records)
                if self._records
                else "none"
            ),
            "- Wall times: "
            + ("included" if include_timings else "omitted (include_timings=True)"),
        ]
        for index, record in enumerate(self._records, start=1):
            lines += [
                "",
                f"## {index}. {record.stage} #{record.run}: {record.label}",
                "",
            ]
            lines.append(
                f"Seed {record.seed}; evaluations: value {record.evaluations.value}, "
                f"jvp {record.evaluations.jvp}, vjp {record.evaluations.vjp} "
                f"(work {record.work})"
                + (f"; wall time {record.seconds:.3f} s" if include_timings else "")
            )
            if record.settings:
                lines += [
                    "",
                    "Settings: "
                    + "; ".join(f"{k} = {v}" for k, v in record.settings.items()),
                ]
            lines += ["", "| quantity | value |", "| --- | --- |"]
            for key, value in record.summary.items():
                text = _fmt(value) if isinstance(value, float) else str(value)
                lines.append(f"| {key} | {text} |")
            for table in record.tables:
                lines += ["", f"**{table.title}**", ""]
                lines.append("| " + " | ".join(table.header) + " |")
                lines.append("| " + " | ".join("---" for _ in table.header) + " |")
                for row in table.rows:
                    lines.append("| " + " | ".join(row) + " |")
            if record.warnings:
                lines += ["", "Warnings:"]
                lines += [f"- {w}" for w in record.warnings]
        total = self.evaluations
        lines += [
            "",
            "## Evaluation ledger",
            "",
            "| stage | value | jvp | vjp | work |"
            + (" wall time (s) |" if include_timings else ""),
            "| --- | --- | --- | --- | --- |" + (" --- |" if include_timings else ""),
        ]
        for record in self._records:
            entry = (
                f"| {record.stage} #{record.run} | {record.evaluations.value} | "
                f"{record.evaluations.jvp} | {record.evaluations.vjp} | "
                f"{record.work} |"
            )
            if include_timings:
                entry += f" {record.seconds:.3f} |"
            lines.append(entry)
        lines.append(
            f"| total | {total.value} | {total.jvp} | {total.vjp} | {total.work} |"
            + (
                f" {sum(r.seconds for r in self._records):.3f} |"
                if include_timings
                else ""
            )
        )
        if self.kind == "linear":
            lines += [
                "",
                "Linear counts: the value column holds applications of A (the "
                "forward map and its tangent are the same), the vjp column "
                "applications of A'.",
            ]
        lines += ["", "## Caveats", ""]
        lines += [f"- {note}" for note in self._caveats()]
        warned = [r for r in self._records if r.warnings]
        if warned:
            lines += ["", "## Warnings by stage", ""]
            for record in warned:
                lines.append(f"- {record.stage} #{record.run}: {len(record.warnings)}")
        return "\n".join(lines) + "\n"
