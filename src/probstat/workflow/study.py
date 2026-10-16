"""A single study object that runs every UQ analysis on one model.

``UQStudy`` is the integration layer: it owns one
:class:`~probstat.modeling.ProbabilisticModel`, one root random stream, and a
log of every analysis performed. Each stage draws its randomness from a named
child stream, so the whole study is reproducible from one seed, and every
stage records the forward-model evaluations it charged. The study does not
re-implement any method; it calls the domain packages and keeps their results.

Accounting and provenance:

- Every stage record states its status (``"complete"``, ``"aborted"``, or
  ``"budget-exceeded"``), completed and failed evaluations, cache hits, the
  stage settings, any convergence warnings, and the input law it used
  (a label such as ``"prior"`` or ``"posterior #1"`` with a stable hash).
  A stage that raises after spending evaluations is still recorded, so
  :attr:`UQStudy.total_evaluations` never under-states the cost.
- An optional ``evaluation_budget`` refuses a stage whose cost is known and
  too large before it runs, and stops any other stage at the limit.
- :meth:`UQStudy.adopt_posterior` (or :meth:`UQStudy.with_inputs`) replaces
  the input law, so stages after a calibration use the posterior.
- :meth:`UQStudy.to_dict`, :meth:`UQStudy.to_json`, and
  :meth:`UQStudy.save_json` write a JSON record with the library and Python
  versions, platform, UTC creation time, root seed, and every stage.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import math
import platform
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar

import probstat
from probstat.conformal import conformalize_surrogate
from probstat.design import (
    BayesianDesignProblem,
    Condition,
    GreedyDesignResult,
    InformationGainEstimate,
    expected_information_gain,
    greedy_design,
)
from probstat.distributions import Uniform
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import student_t_quantile
from probstat.information import InformationUnit, ksg_mutual_information
from probstat.inverse import (
    CalibrationPosterior,
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyIdentifiability,
    DiscrepancyPosterior,
    DiscrepancyPrediction,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    LogLinearNoise,
    NoiseModel,
    PosteriorPrediction,
    predict,
)
from probstat.linalg import Vector
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.modeling.fitting import (
    JointInputFit,
    fit_joint_input,
)
from probstat.monte_carlo import ProbabilityEstimate, estimate_probability
from probstat.propagation import PropagationStudy, propagate
from probstat.reliability import (
    AKMCSResult,
    FORMResult,
    MethodComparison,
    MixtureImportanceResult,
    ModelLimitState,
    MultipleDesignPointsResult,
    MultiPointFORMResult,
    ak_mcs,
    compare_reliability_methods,
    find_design_points,
    form_hlrf,
    mixture_importance_sampling,
    multi_point_form,
)
from probstat.risk import (
    LossFunction,
    PeaksOverThresholdFit,
    ReturnLevel,
    RiskAssessment,
    assess_risk,
    fit_pot,
)
from probstat.sensitivity import (
    ModelSobolResult,
    MorrisResult,
    ShapleyResult,
    morris_screening,
    shapley_effects,
    sobol_analysis,
)
from probstat.statistics import (
    ExceedanceCounter,
    P2Quantile,
    RunningMoments,
    quantile,
)
from probstat.surrogates import (
    fit_gaussian_process_to_model,
    fit_polynomial_chaos,
    surrogate_probabilistic_model,
)
from probstat.uq import (
    DEFAULT_SAFETY_FACTOR,
    EvaluationBudgetExceeded,
    EvaluationCache,
    ForwardModel,
    FrozenMapping,
    GridConvergenceResult,
    LeastSquaresConvergence,
    NumericalErrorBudget,
    NumericalErrorDistribution,
    QuantityOfInterest,
    SolutionVerification,
    json_safe,
    verify_solution,
)
from probstat.validation import (
    ValidationAssessment,
    ValidationExperiment,
    assess_validation,
    crps_ensemble,
    ensemble_pit,
    pit_coverage,
)

from .chaining import (
    DEFAULT_POSTERIOR_FAMILIES,
    PosteriorMethod,
    input_law_hash,
    posterior_joint_input,
)

STAGE_STATUSES = ("complete", "aborted", "budget-exceeded")


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One analysis stage of a study, with its cost and provenance.

    Args:
        stage: Stage kind such as ``"propagation"`` or ``"reliability"``.
        label: Human-readable description of the stage.
        summary: Headline numbers of the stage.
        evaluations: Completed model evaluations charged to the stage
            (surrogate evaluations in a surrogate study).
        seed: Seed of the stream the stage used.
        result: The full domain result object (``None`` for an aborted
            stage).
        status: ``"complete"``, ``"aborted"`` (the stage raised after
            spending evaluations), or ``"budget-exceeded"``.
        failed_evaluations: Failed simulator attempts during the stage.
        cache_hits: Requests answered from an evaluation cache.
        error: ``"ExceptionType: message"`` of an aborted stage.
        settings: Arguments that define the stage (sample sizes, methods,
            chain settings, ...).
        warnings: Convergence or accuracy warnings raised by the stage.
        input_law: Label of the input law the stage used (``"prior"``,
            ``"posterior #1 (gaussian_copula)"``, ...).
        input_law_hash: Stable hash of that input law.
        evaluation_kind: ``"forward-model"`` or ``"surrogate"``.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    stage: str
    label: str
    summary: Mapping[str, float | str]
    evaluations: int
    seed: int
    result: Any = field(repr=False, compare=False)
    status: str = "complete"
    failed_evaluations: int = 0
    cache_hits: int = 0
    error: str | None = None
    settings: Mapping[str, object] = field(default_factory=FrozenMapping)
    warnings: tuple[str, ...] = ()
    input_law: str = "prior"
    input_law_hash: str = ""
    evaluation_kind: str = "forward-model"

    @property
    def complete(self) -> bool:
        """Return whether the stage finished and produced a result.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        return self.status == "complete"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary of the record (without ``result``).

        Floats stay numbers (non-finite floats become ``"inf"``, ``"-inf"``,
        or ``"nan"``), tuples become lists, and the domain result is named by
        its type only; nothing needs pickle.

        References:
            Project reporting policy. No equation is evaluated.

        """
        return {
            "stage": self.stage,
            "label": self.label,
            "status": self.status,
            "error": self.error,
            "evaluation_kind": self.evaluation_kind,
            "evaluations": self.evaluations,
            "failed_evaluations": self.failed_evaluations,
            "cache_hits": self.cache_hits,
            "seed": self.seed,
            "input_law": {"label": self.input_law, "hash": self.input_law_hash},
            "settings": json_safe(self.settings),
            "warnings": list(self.warnings),
            "summary": json_safe(self.summary),
            "result_type": None
            if self.result is None
            else type(self.result).__name__,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return :meth:`to_dict` as JSON text.

        Args:
            indent: JSON indentation (``None`` for one line).

        References:
            Project reporting policy. No equation is evaluated.

        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass(frozen=True, slots=True)
class TargetedPropagation:
    """Monte Carlo propagation of one QoI sized by a relative-error target.

    A sequential *pilot* chooses the sample size ``n`` (the first batch at
    which the relative standard error of the controlled estimate meets the
    target). With ``final="fresh"`` (default) every reported estimate comes
    from a *fresh, independent* sample of size ``n``: given ``n`` it is an
    ordinary fixed-size Monte Carlo sample, so the estimates are unbiased
    and the intervals have their nominal coverage. With ``final="reuse"``
    the pilot sample itself is reported; that is cheaper but data-dependent
    stopping biases it (it stops preferentially when the running estimate
    happens to look precise) and the intervals under-cover.

    Every summary comes from one-pass, constant-memory accumulators: no model
    output is stored. Fields that do not apply are ``None`` (never NaN).

    Args:
        quantity: QoI name.
        controlled: ``"mean"`` or ``"probability"``: the estimate whose
            relative standard error was driven to the target.
        target_relative_error: Requested relative standard error.
        relative_error: Relative standard error of the reported estimate
            (``inf`` when undefined, e.g. no failures observed).
        converged: Whether the pilot met the target within ``max_samples``.
        sample_size: Size of the sample behind the reported estimates.
        batches: Number of pilot batches drawn.
        mean: Mean of the QoI.
        standard_error: Monte Carlo standard error ``s / sqrt(n)`` of the mean.
        quantiles: ``(probability, estimate)`` pairs from the P² algorithm
            (approximate; exact type-7 quantiles for ``n <= 5``).
        threshold: Failure threshold, or ``None`` in mean mode.
        failure: ``"above"`` (event ``Y > t``, as in
            :meth:`~probstat.propagation.OutputDistribution.exceedance`) or
            ``"below"`` (event ``Y <= t``, its complement).
        probability: Event-probability estimate (``None`` without a
            threshold).
        probability_standard_error: Plug-in Bernoulli standard error
            (``None`` without a threshold).
        probability_interval: Wilson interval at ``confidence`` (``None``
            without a threshold).
        events: Number of events counted.
        confidence: Confidence level of the intervals.
        final: ``"fresh"`` or ``"reuse"``.
        pilot_size: Runs of the sequential pilot.
        pilot_relative_error: Relative error reached by the pilot.
        mean_interval: Student-t interval of the mean at ``confidence``.
        failed_evaluations: Failed simulator runs dropped under the
            ``"skip"``/``"record"`` policies (pilot and final sample).

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962],
        [REF-JAIN-CHLAMTAC-1985], [REF-WILSON-1927], [REF-STEIN-1945].

    """

    quantity: str
    controlled: str
    target_relative_error: float
    relative_error: float
    converged: bool
    sample_size: int
    batches: int
    mean: float
    standard_error: float
    quantiles: tuple[tuple[float, float], ...]
    threshold: float | None
    failure: str
    probability: float | None
    probability_standard_error: float | None
    probability_interval: tuple[float, float] | None
    events: int
    confidence: float = 0.95
    final: str = "fresh"
    pilot_size: int = 0
    pilot_relative_error: float = math.inf
    mean_interval: tuple[float, float] | None = None
    failed_evaluations: int = 0

    @property
    def evaluations(self) -> int:
        """Return completed model runs of pilot and final sample.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

        """
        if self.final == "fresh":
            return self.pilot_size + self.sample_size
        return self.sample_size

    def quantile(self, probability: float) -> float:
        """Return the streaming estimate of one requested quantile level.

        Args:
            probability: A level that was requested from the stage.

        Returns:
            P² estimate of the quantile.

        Raises:
            KeyError: If the level was not requested.

        References:
            [REF-JAIN-CHLAMTAC-1985]. Reads the stored [EQ-STAT-STREAM-005]
            estimate.

        """
        for level, estimate in self.quantiles:
            if level == probability:
                return estimate
        raise KeyError(f"quantile level {probability!r} was not requested")


@dataclass(frozen=True, slots=True)
class DesignPointAnalysis:
    """Several design points of one failure event and their system estimates.

    Args:
        points: Design points found by sequential bulging search.
        system: Multi-point FORM series-system probability with bounds.
        mixture: Mixture importance sampling over all design points, or
            ``None`` when it was not requested.

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-MELCHERS-BECK-2018,
        Chapter 5].

    """

    points: MultipleDesignPointsResult
    system: MultiPointFORMResult
    mixture: MixtureImportanceResult | None


@dataclass(frozen=True, slots=True)
class TailAnalysis:
    """Peaks-over-threshold analysis of one QoI tail, in QoI units.

    Return periods count model runs: the ``T``-run return level is exceeded
    (or, for the lower tail, undercut) with probability ``1/T`` per run.

    Args:
        quantity: QoI name.
        tail: ``"upper"`` or ``"lower"``.
        fit: POT fit on the oriented sample (``-Y`` for the lower tail), one
            observation per model run.
        return_levels: Return levels mapped back to QoI units.
        level: Level whose tail probability was compared, or ``None``.
        pot_probability: POT estimate ``zeta_u S_GPD`` of the tail
            probability of ``level`` (``None`` without a level).
        empirical: Empirical tail probability of ``level`` with a Wilson
            interval, or ``None``.
        sample_size: Number of model outputs analyzed.
        source: Where the sample came from.
        exceedances: Outputs beyond the threshold in QoI units, ordered from
            the least to the most extreme (empty for results built without
            them). The ``k``-th most extreme of ``sample_size`` runs has the
            empirical tail probability ``(k - 1/2) / sample_size``.

    References:
        [REF-COLES-2001, Section 4.3], [REF-WILSON-1927].

    """

    quantity: str
    tail: str
    fit: PeaksOverThresholdFit
    return_levels: tuple[ReturnLevel, ...]
    level: float | None
    pot_probability: float | None
    empirical: ProbabilityEstimate | None
    sample_size: int
    source: str
    exceedances: tuple[float, ...] = field(default=(), repr=False)

    @property
    def threshold(self) -> float:
        """Return the threshold in QoI units.

        References:
            [REF-COLES-2001, Section 4.3]. No equation is evaluated.

        """
        return self.fit.threshold if self.tail == "upper" else -self.fit.threshold


@dataclass(frozen=True, slots=True)
class InformationSensitivity:
    """Mutual information between each input and one QoI.

    Args:
        quantity: QoI name.
        input_names: Input names in model order.
        mutual_information: KSG estimates ``I(X_i; Y)``, one per input.
        k: KSG neighbour order.
        sample_size: Number of model runs used.
        unit: ``"nats"`` or ``"bits"``.
        source: Where the sample came from.

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004],
        [REF-COVER-THOMAS-2006, Chapter 8].

    """

    quantity: str
    input_names: tuple[str, ...]
    mutual_information: tuple[float, ...]
    k: int
    sample_size: int
    unit: str
    source: str

    def as_mapping(self) -> Mapping[str, float]:
        """Return a read-only input-name mapping to mutual information.

        References:
            [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004]. No equation is evaluated.

        """
        return MappingProxyType(
            dict(zip(self.input_names, self.mutual_information, strict=True))
        )

    def ranking(self) -> tuple[str, ...]:
        """Return input names from most to least informative.

        References:
            [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004]. No equation is evaluated.

        """
        order = sorted(
            range(len(self.input_names)),
            key=lambda i: (-self.mutual_information[i], i),
        )
        return tuple(self.input_names[i] for i in order)


@dataclass(frozen=True, slots=True)
class DesignRecommendation:
    """Greedy Bayesian design of new experiments for the study's calibration.

    Args:
        selection: Greedy selection with its EIG path and per-step scores.
        estimate: Independent EIG estimate (with standard error) of the full
            design ``baseline + selected``.
        baseline: Conditions already performed (the calibration experiments)
            or empty.
        candidates: Candidate conditions in the order they were scored, so
            ``selection.candidate_scores[k][i]`` belongs to ``candidates[i]``
            (empty for results built without them).

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013].

    """

    selection: GreedyDesignResult
    estimate: InformationGainEstimate
    baseline: tuple[Condition, ...]
    candidates: tuple[Condition, ...] = ()

    @property
    def design(self) -> tuple[Condition, ...]:
        """Return the full design: baseline followed by the selected conditions.

        References:
            [REF-HUAN-MARZOUK-2013]. No equation is evaluated.

        """
        return self.baseline + self.selection.selected


@dataclass(frozen=True, slots=True)
class DiscrepancyAnalysis:
    """Kennedy-O'Hagan calibration of one QoI with its diagnostics.

    Args:
        calibration: The calibration problem (simulator, data, noise model,
            and discrepancy process).
        posterior: MCMC posterior of ``(theta, sigma_delta, ell, a, b)``.
        identifiability: Theta-discrepancy confounding report.
        level: Central probability of the recorded intervals.
        grid_input: Condition along which ``grid`` was predicted, or
            ``None`` when no grid was requested or the discrepancy has more
            than one input.
        grid: Predictions at evenly spaced values of ``grid_input`` across
            the experiments' range (other conditions at the model defaults).

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """

    calibration: DiscrepancyCalibration
    posterior: DiscrepancyPosterior
    identifiability: DiscrepancyIdentifiability
    level: float
    grid_input: str | None
    grid: tuple[DiscrepancyPrediction, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class DiscrepancyStudyPrediction:
    """Simulator-only and bias-corrected predictions after a KOH calibration.

    Args:
        quantity: Calibrated QoI.
        predictions: One prediction per requested condition.
        level: Central probability of the recorded intervals.
        draws: Posterior draws propagated.
        evaluations: Forward-model evaluations used.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Section 6.3].

    """

    quantity: str
    predictions: tuple[DiscrepancyPrediction, ...]
    level: float
    draws: int
    evaluations: int


def _orient_return_level(level: ReturnLevel, sign: float) -> ReturnLevel:
    """Map a return level of ``sign * Y`` back to QoI units.

    References:
        [REF-COLES-2001, Section 4.3]. No equation is evaluated.

    """
    if sign > 0.0:
        return level
    return ReturnLevel(
        level.return_period,
        -level.level,
        level.standard_error,
        -level.upper,
        -level.lower,
        level.confidence,
        level.method,
    )


def _format_settings(settings: Mapping[str, object]) -> str:
    """Render stage settings as ``name=value`` pairs.

    References:
        Project reporting policy. No equation is evaluated.

    """
    parts = []
    for key, value in settings.items():
        if isinstance(value, float):
            text = f"{value:g}"
        elif isinstance(value, tuple):
            text = "(" + ", ".join(str(item) for item in value) + ")"
        else:
            text = str(value)
        parts.append(f"{key}={text}")
    return "; ".join(parts)


def _converged_text(result: FORMResult) -> str:
    """Return ``"yes"`` or ``"no (<termination reason>)"`` for a FORM search.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5]. No equation is evaluated.

    """
    return "yes" if result.converged else f"no ({result.message})"


def _format_condition(condition: Condition) -> str:
    """Render a condition mapping as ``name=value`` pairs.

    References:
        Project reporting policy. No equation is evaluated.

    """
    if not condition:
        return "(model defaults)"
    return ", ".join(f"{name}={value:g}" for name, value in condition.items())


def _sized(value: object) -> Sequence[Any]:
    """Return ``value`` if it is a list or tuple, else an empty tuple.

    Cost estimates must not consume a generator argument before the stage
    method receives it; an unsized argument simply gives no up-front cost.

    References:
        Project validation policy. No equation is evaluated.

    """
    return value if isinstance(value, tuple | list) else ()


def _setting_value(value: object) -> object:
    """Return a compact, JSON-safe description of one stage argument.

    Scalars and short tuples of scalars are kept; other objects are named by
    type (and length for collections), so settings never embed large data.

    References:
        Project reporting policy. No equation is evaluated.

    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list) and len(value) <= 20:
        if all(
            item is None or isinstance(item, bool | int | float | str)
            for item in value
        ):
            return tuple(value)
    if isinstance(value, tuple | list):
        kinds = sorted({type(item).__name__ for item in value})
        return f"{len(value)} x {'/'.join(kinds) or 'item'}"
    if isinstance(value, Mapping) and len(value) <= 20:
        if all(isinstance(item, bool | int | float | str) for item in value.values()):
            return {str(key): item for key, item in value.items()}
    return f"<{type(value).__name__}>"


@dataclass
class _StageContext:
    """Bookkeeping of the stage that is currently running.

    References:
        [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

    """

    stage: str
    seed: int
    settings: dict[str, object]
    failed_before: int
    hits_before: int
    warnings: list[str] = field(default_factory=list)
    recorded: bool = False


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _stage(
    kind: str,
    cost: Callable[[UQStudy, Mapping[str, Any]], int | None] | None = None,
) -> Callable[
    [Callable[Concatenate[UQStudy, _P], _R]], Callable[Concatenate[UQStudy, _P], _R]
]:
    """Run a study method as one stage: budget, settings, abort records.

    The wrapper binds the call arguments (recorded as stage settings),
    refuses the stage before it runs when ``cost`` gives a known cost above
    the remaining budget, installs the study's cache, executor, and budget
    limit on the model, and records an aborted stage when the method raises
    after spending evaluations.

    Args:
        kind: Stage kind (also the stream name).
        cost: Optional function of the study and bound arguments that returns
            the stage's known minimum cost, or ``None`` when unknown.

    Returns:
        Method decorator.

    References:
        [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

    """

    def decorate(
        method: Callable[Concatenate[UQStudy, _P], _R],
    ) -> Callable[Concatenate[UQStudy, _P], _R]:
        """Wrap one stage method.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self: UQStudy, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            """Run the stage under the study's accounting.

            References:
                [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

            """
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = {
                name: value for name, value in bound.arguments.items() if name != "self"
            }
            settings = {
                name: _setting_value(value) for name, value in arguments.items()
            }
            needed = None if cost is None else cost(self, arguments)
            with self._running(kind, settings, needed):
                return method(self, *args, **kwargs)

        return wrapper

    return decorate


class _Accumulators:
    """Constant-memory accumulators of one QoI for sequential propagation.

    References:
        [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. No equation is evaluated.

    """

    def __init__(
        self, levels: tuple[float, ...], event: Callable[[float], bool] | None
    ) -> None:
        """Create empty accumulators.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        self.moments = RunningMoments()
        self.markers = tuple(P2Quantile(p) for p in levels)
        self.event = event
        self.counter = None if event is None else ExceedanceCounter()

    @property
    def count(self) -> int:
        """Return the number of pushed outputs.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        return self.moments.count

    def push(self, value: float) -> None:
        """Push one model output into every accumulator.

        References:
            [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. Updates
            [EQ-STAT-STREAM-005] markers.

        """
        self.moments.push(value)
        for marker in self.markers:
            marker.push(value)
        if self.counter is not None and self.event is not None:
            self.counter.push(self.event(value))

    def relative_error(self) -> float:
        """Return the relative standard error of the controlled estimate.

        The mean uses ``s / (sqrt(n) |mean|)``; the probability uses
        ``sqrt((1 - p) / (n p))`` with ``inf`` before the first event and the
        half-count-adjusted proportion when every draw is an event.

        Equation:
            [EQ-STAT-STREAM-007], [EQ-MC-PROB-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4].

        """
        if self.moments.count < 2:
            return math.inf
        counter = self.counter
        if counter is None:
            return self.moments.relative_standard_error()
        if counter.events == 0:
            return math.inf
        p = counter.proportion
        if counter.events == counter.trials:
            # All draws are events: (1 - p) / (n p) would be exactly zero.
            p = (counter.events + 0.5) / (counter.trials + 1.0)
        return math.sqrt((1.0 - p) / (counter.trials * p))

    def quantiles(self) -> tuple[tuple[float, float], ...]:
        """Return ``(level, P² estimate)`` pairs.

        References:
            [REF-JAIN-CHLAMTAC-1985]. Reads [EQ-STAT-STREAM-005] estimates.

        """
        return tuple((marker.probability, marker.estimate) for marker in self.markers)


class UQStudy:
    """Run propagation, sensitivity, reliability, and risk on one model.

    Args:
        model: The probabilistic model under study.
        seed: Root seed or stream for every stochastic stage.
        name: Study label (defaults to the model name).
        evaluation_budget: Optional cap on the forward-model runs (completed
            plus failed) charged by all stages. A stage whose cost is known
            and exceeds the remainder is refused before it runs; any other
            stage is stopped at the limit and recorded as
            ``"budget-exceeded"``.
        cache: ``True`` (a new :class:`~probstat.uq.EvaluationCache`) or a
            cache shared by every stage, so identical runs are reused across
            stages. Cache hits are reported, never counted as evaluations.
        executor: Optional :class:`concurrent.futures.Executor` used for
            batch evaluation (propagation and fresh samples, and any library
            code that calls ``evaluate_many``). Results do not depend on it.
        evaluation_kind: ``"forward-model"`` or ``"surrogate"`` (set by
            :meth:`emulate` for the study on the emulator).

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-SUDRET-2007, Chapter 1].

    """

    def __init__(
        self,
        model: ProbabilisticModel,
        *,
        seed: int | RandomStream = 0,
        name: str | None = None,
        evaluation_budget: int | None = None,
        cache: bool | EvaluationCache = False,
        executor: Executor | None = None,
        evaluation_kind: Literal["forward-model", "surrogate"] = "forward-model",
    ) -> None:
        """Bind the model and root stream.

        Raises:
            TypeError: If ``model`` is not a ProbabilisticModel or ``cache``
                has the wrong type.
            ValueError: If the budget is negative or the kind is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        if not isinstance(model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        if evaluation_budget is not None and (
            isinstance(evaluation_budget, bool)
            or not isinstance(evaluation_budget, int)
            or evaluation_budget < 0
        ):
            raise ValueError("evaluation_budget must be a non-negative integer")
        if evaluation_kind not in ("forward-model", "surrogate"):
            raise ValueError("evaluation_kind must be 'forward-model' or 'surrogate'")
        if isinstance(cache, bool):
            store: EvaluationCache | None = EvaluationCache() if cache else None
        elif isinstance(cache, EvaluationCache):
            store = cache
        else:
            raise TypeError("cache must be a bool or an EvaluationCache")
        self._model = model
        self._root = resolve_stream(seed)
        self._name = name or model.name
        self._records: list[StageRecord] = []
        self._problem: CalibrationProblem | None = None
        self._posterior: CalibrationPosterior | None = None
        self._discrepancy: DiscrepancyAnalysis | None = None
        # Which calibration predict() and design() use: the latest one run.
        self._active: Literal["plain", "discrepancy"] | None = None
        # Latest solution verification and its u_num, per QoI.
        self._verifications: dict[str, tuple[SolutionVerification, float]] = {}
        self._budget = evaluation_budget
        self._cache = store
        self._executor = executor
        self._kind = evaluation_kind
        self._contexts: list[_StageContext] = []
        self._children: list[tuple[int, UQStudy]] = []
        self._law_label = "prior"
        self._law_hash = input_law_hash(model.inputs)
        self._law_changes = 0
        self._created = datetime.now(UTC).replace(microsecond=0).isoformat()

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state (the executor is dropped).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        state = dict(self.__dict__)
        state["_executor"] = None
        state["_contexts"] = []
        return state

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Sequence[float]],
        model: ForwardModel,
        quantities: Sequence[QuantityOfInterest],
        *,
        candidates: Sequence[str] | None = None,
        dependence: Literal["spearman", "normal-scores", "independent"] = "spearman",
        seed: int | RandomStream = 0,
        name: str = "model",
        fixed: Mapping[str, Any] | None = None,
    ) -> UQStudy:
        """Fit the input law from data and start a study on the fitted model.

        Each column is assigned the family with the best information criterion
        among ``candidates`` and the dependence is a Gaussian copula fitted
        from ranks. The fit is recorded as the first stage, so the report
        shows which families were chosen and why.

        Args:
            data: Named input observations of equal length.
            model: Forward model consuming the named inputs.
            quantities: Quantities of interest.
            candidates: Candidate family names (all ready-made by default).
            dependence: Copula estimator or ``"independent"``.
            seed: Root seed for the study.
            name: Model and study label.
            fixed: Deterministic parameters passed to every model call.

        Returns:
            A study whose model uses the fitted joint input law.

        Equation:
            [EQ-INF-IC-001], [EQ-MOD-COPULA-FIT-001].

        References:
            [REF-AKAIKE-1974], [REF-GENEST-GHOUDI-RIVEST-1995].

        """
        fit: JointInputFit = fit_joint_input(
            data, candidates=candidates, dependence=dependence
        )
        probabilistic = ProbabilisticModel(
            fit.joint_input, model, tuple(quantities), name=name, fixed=fixed or {}
        )
        study = cls(probabilistic, seed=seed, name=name)
        study._law_label = "fitted from data"
        summary: dict[str, float | str] = {"rows": len(next(iter(data.values())))}
        for column, result in zip(
            fit.joint_input.names, fit.marginal_fits, strict=True
        ):
            summary[f"family[{column}]"] = result.family
            summary[f"aic[{column}]"] = result.aic
        summary["dependence"] = dependence
        study._record(
            "fitting",
            f"input law fitted from {summary['rows']} observations",
            summary,
            0,
            study.stream("fitting"),
            fit,
            settings={
                "candidates": _setting_value(
                    None if candidates is None else tuple(candidates)
                ),
                "dependence": dependence,
            },
        )
        return study

    @property
    def model(self) -> ProbabilisticModel:
        """Return the model under study (with the current input law).

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return self._model

    @property
    def name(self) -> str:
        """Return the study label.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return self._name

    @property
    def root_seed(self) -> int:
        """Return the seed of the root stream.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. No equation is evaluated.

        """
        return self._root.seed

    @property
    def evaluation_kind(self) -> str:
        """Return ``"forward-model"`` or ``"surrogate"``.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        return self._kind

    @property
    def evaluation_budget(self) -> int | None:
        """Return the evaluation budget, or ``None`` without one.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self._budget

    @property
    def input_law(self) -> tuple[str, str]:
        """Return ``(label, hash)`` of the current input law.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self._law_label, self._law_hash

    @property
    def records(self) -> tuple[StageRecord, ...]:
        """Return stage records (complete and aborted) in execution order.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return tuple(self._records)

    @property
    def children(self) -> tuple[tuple[int, UQStudy], ...]:
        """Return ``(stage number, study)`` of surrogate studies from emulate.

        The stage number is the one-based position of the emulation record.

        References:
            [REF-SUDRET-2007, Chapter 4]. No equation is evaluated.

        """
        return tuple(self._children)

    @property
    def total_evaluations(self) -> int:
        """Return the completed evaluations charged by all stages.

        Aborted stages are included. In a surrogate study these are surrogate
        evaluations; :meth:`emulate` children are never added to the parent.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return sum(record.evaluations for record in self._records)

    @property
    def total_failed_evaluations(self) -> int:
        """Return the failed simulator attempts charged by all stages.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return sum(record.failed_evaluations for record in self._records)

    @property
    def remaining_budget(self) -> int | None:
        """Return the runs left under the budget, or ``None`` without one.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if self._budget is None:
            return None
        spent = self.total_evaluations + self.total_failed_evaluations
        return max(0, self._budget - spent)

    def latest(self, stage: str) -> StageRecord:
        """Return the most recent complete record of a stage kind.

        Args:
            stage: Stage kind.

        Returns:
            Latest matching record.

        Raises:
            KeyError: If the stage has not been completed.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        for record in reversed(self._records):
            if record.stage == stage and record.complete:
                return record
        raise KeyError(f"stage {stage!r} has not been run")

    def stream(self, stage: str) -> RandomStream:
        """Return the child stream for the next run of a stage.

        Aborted runs count, so a stage retried after an abort uses new
        randomness.

        Args:
            stage: Stage kind.

        Returns:
            Child stream ``root.child(stage, k)`` for the k-th run of the stage.

        Equation:
            [EQ-RNG-SPLITMIX-001].

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        count = sum(record.stage == stage for record in self._records)
        return self._root.child(stage, count)

    def _require_budget(self, stage: str, cost: int) -> None:
        """Refuse a stage whose known cost exceeds the remaining budget.

        Raises:
            EvaluationBudgetExceeded: If ``cost`` is above the remainder.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        remaining = self.remaining_budget
        if remaining is not None and cost > remaining:
            raise EvaluationBudgetExceeded(
                f"stage {stage!r} needs at least {cost} model evaluations but only "
                f"{remaining} of the study budget of {self._budget} remain; "
                "nothing was run"
            )

    @contextmanager
    def _running(
        self, stage: str, settings: dict[str, object], cost: int | None
    ) -> Iterator[_StageContext]:
        """Run one stage with budget, cache, executor, and abort recording.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        if cost is not None:
            self._require_budget(stage, cost)
        model = self._model
        counter = model.counter
        context = _StageContext(
            stage,
            self.stream(stage).seed,
            settings,
            counter.failed,
            counter.cache_hits,
        )
        count_before = counter.count
        self._contexts.append(context)
        try:
            with (
                counter.limited(self.remaining_budget),
                model.evaluation_context(cache=self._cache, executor=self._executor),
            ):
                yield context
        except BaseException as error:
            spent = counter.count - count_before
            failed = counter.failed - context.failed_before
            if not context.recorded and (spent or failed):
                status = (
                    "budget-exceeded"
                    if isinstance(error, EvaluationBudgetExceeded)
                    else "aborted"
                )
                kind, message = type(error).__name__, str(error)
                self._records.append(
                    StageRecord(
                        stage,
                        f"{stage} stopped by {kind}",
                        FrozenMapping({"status": status, "error": message}),
                        spent,
                        context.seed,
                        None,
                        status=status,
                        failed_evaluations=failed,
                        cache_hits=counter.cache_hits - context.hits_before,
                        error=f"{kind}: {message}",
                        settings=FrozenMapping(settings),
                        warnings=tuple(context.warnings),
                        input_law=self._law_label,
                        input_law_hash=self._law_hash,
                        evaluation_kind=self._kind,
                    )
                )
            raise
        finally:
            self._contexts.pop()

    def _warn(self, message: str) -> None:
        """Attach a warning to the running stage and emit it.

        References:
            Project reporting policy. No equation is evaluated.

        """
        if self._contexts:
            self._contexts[-1].warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _record(
        self,
        stage: str,
        label: str,
        summary: Mapping[str, float | str],
        evaluations: int,
        stream: RandomStream,
        result: Any,
        *,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """Append one complete stage record with its provenance.

        Failed attempts, cache hits, settings, and warnings come from the
        running stage context; a failure fraction is added to the summary
        and the warnings whenever runs failed.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        context = self._contexts[-1] if self._contexts else None
        counter = self._model.counter
        failed = 0 if context is None else counter.failed - context.failed_before
        hits = 0 if context is None else counter.cache_hits - context.hits_before
        values = dict(summary)
        notes = [] if context is None else list(context.warnings)
        if failed:
            attempted = evaluations + failed
            fraction = failed / attempted if attempted else 0.0
            values["failed runs"] = float(failed)
            values["failure fraction"] = fraction
            notes.append(
                f"{failed} of {attempted} model runs failed ({100.0 * fraction:.3g}%) "
                "and were dropped; estimates are biased unless failures are "
                "unrelated to the output"
            )
        chosen = dict(context.settings) if context is not None else {}
        if settings is not None:
            chosen.update(settings)
        self._records.append(
            StageRecord(
                stage,
                label,
                FrozenMapping(values),
                evaluations,
                stream.seed,
                result,
                failed_evaluations=failed,
                cache_hits=hits,
                settings=FrozenMapping(chosen),
                warnings=tuple(notes),
                input_law=self._law_label,
                input_law_hash=self._law_hash,
                evaluation_kind=self._kind,
            )
        )
        if context is not None:
            context.recorded = True


    @_stage(
        "propagation",
        lambda study, args: args["size"] * (args["replicates"] or 1),
    )
    def propagate(
        self,
        size: int = 4096,
        *,
        design: str = "sobol",
        replicates: int | None = None,
        batch: int | None = None,
    ) -> PropagationStudy:
        """Propagate the joint input law through every QoI.

        With ``replicates=r`` the design is randomized ``r`` times (cost
        ``r * size``) and the summary adds Student-t replicate intervals of
        each mean, the honest error bar for LHS and quasi-random designs.

        Args:
            size: Number of input realizations (per replicate).
            design: Sampling design name.
            replicates: Optional number of independent randomizations.
            batch: Chunk size for a vectorised model or executor jobs.

        Returns:
            Propagation study.

        Equation:
            [EQ-UQ-001], [EQ-MC-MEAN-001], [EQ-PROP-RQMC-CI-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 3-4], [REF-L-ECUYER-2018].

        """
        stream = self.stream("propagation")
        study = propagate(
            self._model,
            size,
            design=design,
            seed=stream,
            replicates=replicates,
            batch=batch,
        )
        summary: dict[str, float | str] = {"design": design, "size": size}
        if replicates is not None:
            summary["replicates"] = float(replicates)
        for name, output in study.outputs.items():
            summary[f"{name}.mean"] = output.mean
            summary[f"{name}.std"] = output.standard_deviation
            summary[f"{name}.q05"] = output.quantile(0.05)
            summary[f"{name}.q95"] = output.quantile(0.95)
            if replicates is not None and len(output.replicate_means()) >= 2:
                interval = output.mean_interval(0.95)
                summary[f"{name}.mean 95% low (replicates)"] = interval.lower
                summary[f"{name}.mean 95% high (replicates)"] = interval.upper
        label = f"{design} propagation, {size} samples"
        if replicates is not None:
            label = f"{design} propagation, {replicates} x {size} samples"
        self._record(
            "propagation",
            label,
            summary,
            study.model_evaluations,
            stream,
            study,
        )
        return study

    def _sequential_sample(
        self,
        quantity: str,
        levels: tuple[float, ...],
        event: Callable[[float], bool] | None,
        draw: Callable[[int, int], tuple[Vector, ...]],
        budget: int,
        batch: int,
        stop: Callable[[_Accumulators], bool] | None,
    ) -> tuple[_Accumulators, int, bool]:
        """Push batches of model outputs into accumulators.

        Args:
            quantity: QoI name.
            levels: P² quantile levels.
            event: Event indicator, or ``None`` in mean mode.
            draw: ``draw(k, m)`` returns the ``k``-th batch of ``m`` inputs.
            budget: Maximum number of input points.
            batch: Points per batch.
            stop: Stopping rule checked after each batch (``None``: never).

        Returns:
            Accumulators, batches drawn, and whether ``stop`` fired.

        References:
            [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. Evaluates
            [EQ-STAT-STREAM-007].

        """
        accumulators = _Accumulators(levels, event)
        drawn = 0
        batches = 0
        while drawn < budget:
            count = min(batch, budget - drawn)
            points = draw(batches, count)
            batches += 1
            drawn += count
            outputs = self._model.evaluate_many(points)
            for value in outputs[quantity]:
                accumulators.push(value)
            if stop is not None and stop(accumulators):
                return accumulators, batches, True
        return accumulators, batches, False

    @_stage("sequential_propagation")
    def propagate_until(
        self,
        quantity: str,
        *,
        target_relative_error: float = 0.01,
        threshold: float | None = None,
        failure: Literal["above", "below"] = "above",
        batch: int = 256,
        max_samples: int = 200_000,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        min_samples: int = 100,
        confidence: float = 0.95,
        final: Literal["fresh", "reuse"] = "fresh",
    ) -> TargetedPropagation:
        """Size a Monte Carlo propagation by a relative-error target.

        **Pilot.** Batches of ``batch`` joint input draws (batch ``k`` on the
        child stream ``stream.child("batch", k)``) are evaluated and pushed
        into a :class:`~probstat.statistics.RunningMoments`, one
        :class:`~probstat.statistics.P2Quantile` per level, and, with a
        threshold, an :class:`~probstat.statistics.ExceedanceCounter`. After
        every batch the relative standard error of the controlled estimate is
        checked: ``s / (sqrt(n) |mean|)`` for the mean, or
        ``sqrt((1 - p) / (n p))`` for the event probability. The pilot stops
        at the first batch meeting the target (after ``min_samples``) or at
        ``max_samples`` (``converged=False``).

        **Final sample.** Stopping when the running estimate *looks* precise
        is data-dependent, so the pilot's own estimates are biased and its
        intervals under-cover (optional stopping). With ``final="fresh"``
        (default) the pilot only chooses ``n``; a fresh, independent sample
        of ``n`` draws (stream ``stream.child("final", k)``) gives every
        reported estimate and interval. Given ``n`` it is an ordinary
        fixed-size sample, so the estimates are unbiased and the intervals
        have their nominal coverage; the price is the extra ``n`` runs (at
        most ``2 * max_samples`` in total). ``final="reuse"`` reports the
        pilot itself (the behaviour before version 0.16) and is biased.

        The event matches :meth:`~probstat.propagation.OutputDistribution.exceedance`:
        ``failure="above"`` counts ``Y > t``; ``"below"`` counts the
        complement ``Y <= t``. Mean intervals are Student-t, probability
        intervals Wilson, both at ``confidence``. No output is stored. Under
        a ``"skip"``/``"record"`` failure policy failed runs are dropped and
        reported; ``n`` then counts successful runs of the pilot.

        Args:
            quantity: QoI name.
            target_relative_error: Positive relative standard-error target.
            threshold: Event threshold; switches control to the probability.
            failure: ``"above"`` (``Y > t``) or ``"below"`` (``Y <= t``).
            batch: Draws between checks, at least two.
            max_samples: Pilot draw budget, at least two.
            quantiles: Quantile levels strictly between zero and one.
            min_samples: Draws before the target may be declared met (capped
                at ``max_samples``).
            confidence: Interval confidence strictly inside ``(0, 1)``.
            final: ``"fresh"`` (unbiased, default) or ``"reuse"`` (legacy,
                biased by optional stopping).

        Returns:
            Targeted propagation result; the stage charges pilot plus final
            evaluations.

        Raises:
            TypeError: If numeric arguments have invalid types.
            ValueError: If ``quantity`` is unknown or an argument is out of
                its domain.

        Equation:
            [EQ-UQ-001], [EQ-STAT-STREAM-007], [EQ-MC-SE-001],
            [EQ-MC-PROB-001], [EQ-INF-CI-WILSON-001], [EQ-STAT-STREAM-005],
            [EQ-WF-SEQ-FRESH-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962],
            [REF-JAIN-CHLAMTAC-1985], [REF-WILSON-1927], [REF-STEIN-1945].

        """
        if quantity not in self._model.quantity_names:
            raise ValueError(f"{quantity!r} is not a quantity of the study model")
        target = require_finite_real(
            target_relative_error, name="target_relative_error"
        )
        if target <= 0.0:
            raise ValueError("target_relative_error must be positive")
        size = require_non_negative_integer(batch, name="batch")
        budget = require_non_negative_integer(max_samples, name="max_samples")
        if size < 2:
            raise ValueError("batch must be at least two")
        if budget < 2:
            raise ValueError("max_samples must be at least two")
        if failure not in ("above", "below"):
            raise ValueError("failure must be 'above' or 'below'")
        if final not in ("fresh", "reuse"):
            raise ValueError("final must be 'fresh' or 'reuse'")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise TypeError("min_samples must be an integer")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        level = require_finite_real(confidence, name="confidence")
        if not 0.0 < level < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1")
        minimum = min(min_samples, budget)
        levels = tuple(require_finite_real(p, name="quantile level") for p in quantiles)
        if any(not 0.0 < p < 1.0 for p in levels):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        limit = (
            None
            if threshold is None
            else require_finite_real(threshold, name="threshold")
        )
        event: Callable[[float], bool] | None = None
        if limit is not None:
            cut = limit
            if failure == "above":

                def event(value: float) -> bool:
                    """Return whether ``value > threshold``.

                    References:
                        [EQ-PROP-THRESH-001], [REF-SMITH-UQ-2024, Chapter 4].

                    """
                    return value > cut

            else:

                def event(value: float) -> bool:
                    """Return whether ``value <= threshold``.

                    References:
                        [EQ-PROP-THRESH-001], [REF-SMITH-UQ-2024, Chapter 4].

                    """
                    return value <= cut

        stream = self.stream("sequential_propagation")
        inputs = self._model.inputs
        before = self._model.evaluations

        def stop(accumulators: _Accumulators) -> bool:
            """Return whether the pilot has met the target.

            References:
                [EQ-STAT-STREAM-007], [REF-ROBERT-CASELLA-2004, Chapter 4].

            """
            return (
                accumulators.relative_error() <= target
                and accumulators.count >= minimum
            )

        def pilot_draw(index: int, count: int) -> tuple[Vector, ...]:
            """Return pilot batch ``index``.

            References:
                [REF-DEVROYE-1986, Section II.2]. No equation is evaluated.

            """
            return inputs.sample(count, seed=stream.child("batch", index))

        def final_draw(index: int, count: int) -> tuple[Vector, ...]:
            """Return final-sample batch ``index``.

            References:
                [REF-DEVROYE-1986, Section II.2]. No equation is evaluated.

            """
            return inputs.sample(count, seed=stream.child("final", index))

        pilot, batches, converged = self._sequential_sample(
            quantity, levels, event, pilot_draw, budget, size, stop
        )
        reported = pilot
        if final == "fresh":
            if pilot.count < 2:
                raise ValueError(
                    "the pilot produced fewer than two successful runs; "
                    "no final sample can be sized"
                )
            reported, _, _ = self._sequential_sample(
                quantity, levels, event, final_draw, pilot.count, size, None
            )
        if reported.count < 2:
            raise ValueError("fewer than two successful runs; no estimate is possible")
        relative = reported.relative_error()
        mean = reported.moments.mean
        standard_error = reported.moments.standard_error()
        half = student_t_quantile(0.5 + 0.5 * level, reported.count - 1) * (
            standard_error
        )
        counter = reported.counter
        probability: float | None = None
        probability_se: float | None = None
        interval: tuple[float, float] | None = None
        events = 0
        if counter is not None:
            probability = counter.proportion
            probability_se = counter.standard_error()
            interval = counter.wilson_interval(confidence=level)
            events = counter.events
        spent = self._model.evaluations - before
        result = TargetedPropagation(
            quantity=quantity,
            controlled="mean" if counter is None else "probability",
            target_relative_error=target,
            relative_error=relative,
            converged=converged,
            sample_size=reported.count,
            batches=batches,
            mean=mean,
            standard_error=standard_error,
            quantiles=reported.quantiles(),
            threshold=limit,
            failure=failure,
            probability=probability,
            probability_standard_error=probability_se,
            probability_interval=interval,
            events=events,
            confidence=level,
            final=final,
            pilot_size=pilot.count,
            pilot_relative_error=pilot.relative_error(),
            mean_interval=(mean - half, mean + half),
            failed_evaluations=self._model.failed_evaluations
            - self._contexts[-1].failed_before
            if self._contexts
            else 0,
        )
        percent = f"{100.0 * level:g}%"
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "controlled": result.controlled,
            "target relative error": target,
            "relative error": relative,
            "pilot relative error": result.pilot_relative_error,
            "converged": "yes" if converged else "no (max_samples reached)",
            "samples": float(result.sample_size),
            "pilot samples": float(result.pilot_size),
            "batches": float(batches),
            "final sample": (
                "fresh (independent of the stopping rule)"
                if final == "fresh"
                else "reused pilot (optional-stopping bias)"
            ),
            f"{quantity}.mean": result.mean,
            f"{quantity}.se": result.standard_error,
            f"{quantity}.mean {percent} low": mean - half,
            f"{quantity}.mean {percent} high": mean + half,
        }
        for q_level, estimate in result.quantiles:
            summary[f"{quantity}.q[{q_level:g}] (P2)"] = estimate
        if limit is not None and probability is not None and interval is not None:
            relation = ">" if failure == "above" else "<="
            name = f"P[{quantity} {relation} {limit:g}]"
            summary[name] = probability
            summary[f"{name} se"] = probability_se if probability_se else 0.0
            summary[f"{name} Wilson {percent} lower"] = interval[0]
            summary[f"{name} Wilson {percent} upper"] = interval[1]
        if not converged:
            self._warn(
                f"propagate_until({quantity!r}) reached max_samples={budget} "
                f"before the target relative error {target:g}"
            )
        if final == "reuse":
            self._warn(
                "propagate_until(final='reuse') reports the pilot sample, whose "
                "estimates carry optional-stopping bias; use final='fresh'"
            )
        self._record(
            "sequential_propagation",
            f"Monte Carlo of {quantity} until relative error {target:g} "
            f"on the {result.controlled}",
            summary,
            spent,
            stream,
            result,
        )
        return result

    @_stage(
        "sensitivity",
        lambda study, args: args["size"] * (study.model.dimension + 2),
    )
    def sensitivity(
        self,
        quantity: str,
        size: int = 1024,
        *,
        design: str = "sobol",
        bootstrap: int = 200,
    ) -> ModelSobolResult:
        """Estimate Sobol indices of one QoI with bootstrap intervals.

        Args:
            quantity: QoI name.
            size: Design rows (cost ``size (d + 2)``).
            design: Unit-hypercube design name.
            bootstrap: Bootstrap replicates.

        Returns:
            Sobol result.

        Equation:
            [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [EQ-SENS-BOOT-001].

        References:
            [REF-SALTELLI-2010].

        """
        stream = self.stream("sensitivity")
        result = sobol_analysis(
            self._model, quantity, size, design=design, seed=stream, bootstrap=bootstrap
        )
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, first, total in zip(
            result.input_names, result.first_order, result.total_effect, strict=True
        ):
            summary[f"S[{name}]"] = first.estimate
            summary[f"ST[{name}]"] = total.estimate
        self._record(
            "sensitivity",
            f"Sobol indices of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("shapley")
    def shapley(
        self,
        quantity: str,
        *,
        permutations: int = 200,
        inner_samples: int = 3,
        variance_samples: int = 1000,
    ) -> ShapleyResult:
        """Estimate Shapley effects, valid also for dependent inputs.

        Unlike Sobol indices, Shapley effects share interaction and dependence
        contributions fairly among inputs and always sum to one.

        Args:
            quantity: QoI name.
            permutations: Random permutations when exact enumeration is not
                used (inputs above six).
            inner_samples: Conditional draws per outer sample.
            variance_samples: Draws for the output variance.

        Returns:
            Shapley result with approximate standard errors.

        Equation:
            [EQ-SENS-SHAPLEY-001].

        References:
            [REF-OWEN-2014], [REF-SONG-NELSON-STAUM-2016].

        """
        stream = self.stream("shapley")
        result = shapley_effects(
            self._model,
            quantity,
            permutations=permutations,
            inner_samples=inner_samples,
            variance_samples=variance_samples,
            seed=stream,
        )
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, value in result.as_mapping().items():
            summary[f"Sh[{name}]"] = value
        self._record(
            "shapley",
            f"Shapley effects of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage(
        "screening",
        lambda study, args: args["trajectories"] * (study.model.dimension + 1),
    )
    def screening(self, quantity: str, trajectories: int = 20) -> MorrisResult:
        """Screen inputs of one QoI with Morris elementary effects.

        Args:
            quantity: QoI name.
            trajectories: Number of trajectories (cost ``r (d + 1)``).

        Returns:
            Morris result.

        Equation:
            [EQ-SENS-MORRIS-001].

        References:
            [REF-MORRIS-1991].

        """
        stream = self.stream("screening")
        result = morris_screening(self._model, quantity, trajectories, seed=stream)
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, value in zip(result.input_names, result.mu_star, strict=True):
            summary[f"mu*[{name}]"] = value
        self._record(
            "screening",
            f"Morris screening of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    def limit_state(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
    ) -> ModelLimitState:
        """Return a limit state on one QoI of the study model.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` (low values fail) or ``"above"``.

        Returns:
            Model limit state with ``g <= 0`` meaning failure.

        Equation:
            [EQ-REL-LIMIT-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return ModelLimitState(self._model, quantity, threshold, failure=failure)

    @_stage(
        "reliability",
        lambda study, args: (
            args["monte_carlo_samples"] if "mc" in _sized(args["methods"]) else None
        ),
    )
    def reliability(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        methods: Sequence[str] = ("form", "sorm", "is", "subset", "mc"),
        monte_carlo_samples: int = 10_000,
    ) -> MethodComparison:
        """Compare reliability methods on one QoI threshold.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            methods: Method subset.
            monte_carlo_samples: Crude Monte Carlo sample size.

        Returns:
            Method comparison (also stored with the FORM design point).

        Equation:
            [EQ-REL-PF-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapters 3-5].

        """
        methods = tuple(methods)
        stream = self.stream("reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        # FORM-based rows share one HL-RF search inside the comparison; the
        # search is repeated afterwards only to read its convergence flag.
        # HL-RF is deterministic, so with a cache over the stage every run of
        # the repeat is a cache hit and costs no simulator evaluation.
        local = self._cache if self._cache is not None else EvaluationCache()
        uses_form = bool({"form", "sorm", "is", "line"} & {m.lower() for m in methods})
        with self._model.evaluation_context(cache=local):
            comparison = compare_reliability_methods(
                state,
                methods=methods,
                monte_carlo_samples=monte_carlo_samples,
                seed=stream,
            )
            check = form_hlrf(state) if uses_form else None
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
        }
        for row in comparison.rows:
            summary[f"pf[{row.method}]"] = row.probability
        if check is not None:
            summary["FORM converged"] = _converged_text(check)
            if not check.converged:
                self._warn(
                    f"FORM for {quantity} did not converge ({check.message}); "
                    "FORM, SORM, and importance-sampling rows build on an "
                    "unconverged design point"
                )
        self._record(
            "reliability",
            f"P({quantity} {'<=' if failure == 'below' else '>='} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            comparison,
        )
        return comparison

    @_stage("design_points")
    def design_points(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        max_points: int = 3,
        importance_samples: int = 2000,
        start: Sequence[float] | None = None,
    ) -> DesignPointAnalysis:
        """Find several design points and estimate the system probability.

        A single FORM point underestimates ``P_f`` when the failure domain has
        several important regions. This stage searches up to ``max_points``
        design points, combines them as a series system by multi-point FORM
        (with Ditlevsen bounds), and, unless ``importance_samples`` is zero,
        checks the result by mixture importance sampling centred on every
        point.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            max_points: Maximum number of design points.
            importance_samples: Mixture importance-sampling size (``0`` skips
                it).
            start: Standard-normal starting point of the first HL-RF search;
                needed when the limit-state gradient vanishes at the origin
                (for example at a kink of a series system).

        Returns:
            Design points, multi-point FORM, and the optional mixture estimate.

        Raises:
            ValueError: If ``importance_samples`` is negative.

        Equation:
            [EQ-REL-MDP-001], [EQ-REL-MDP-002], [EQ-REL-MDP-003].

        References:
            [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-MELCHERS-BECK-2018,
            Chapter 5].

        """
        if importance_samples < 0:
            raise ValueError("importance_samples must be non-negative")
        stream = self.stream("design_points")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        first = None if start is None else form_hlrf(state, start=start)
        points = find_design_points(state, max_points, first=first)
        system = multi_point_form(points, seed=stream.child("system"))
        mixture = (
            mixture_importance_sampling(
                state, points, importance_samples, seed=stream.child("mixture")
            )
            if importance_samples
            else None
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
            "design points": float(len(points.forms)),
        }
        for index, beta in enumerate(points.betas, start=1):
            summary[f"beta[{index}]"] = beta
        summary["pf[single-point FORM]"] = points.forms[0].probability
        summary["pf[multi-point FORM]"] = system.probability
        summary["pf bounds (Ditlevsen)"] = (
            f"[{system.bounds.ditlevsen_lower:.4g}, "
            f"{system.bounds.ditlevsen_upper:.4g}]"
        )
        if mixture is not None:
            summary["pf[mixture IS]"] = mixture.probability
            summary["cov[mixture IS]"] = mixture.coefficient_of_variation
        for index, point in enumerate(points.forms, start=1):
            summary[f"FORM converged[{index}]"] = _converged_text(point)
        unconverged = [
            index
            for index, point in enumerate(points.forms, start=1)
            if not point.converged
        ]
        if unconverged:
            self._warn(
                f"design point(s) {unconverged} of {quantity} come from an "
                "unconverged FORM search; multi-point FORM and mixture importance "
                "sampling build on them"
            )
        sign = "<=" if failure == "below" else ">="
        self._record(
            "design_points",
            f"{len(points.forms)} design points of {quantity} {sign} {threshold:g}",
            summary,
            self._model.evaluations - before,
            stream,
            DesignPointAnalysis(points, system, mixture),
        )
        return DesignPointAnalysis(points, system, mixture)

    @_stage(
        "active_reliability",
        lambda study, args: (
            args["population"] if args["compare_with_monte_carlo"] else None
        ),
    )
    def active_reliability(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        population: int = 10_000,
        target_cov: float = 0.1,
        max_added: int = 200,
        compare_with_monte_carlo: bool = False,
    ) -> AKMCSResult:
        """Estimate a failure probability by AK-MCS active learning.

        A Gaussian-process surrogate of the limit state is refined where its
        sign is most uncertain until the learning function ``U`` exceeds 2 on
        the whole Monte Carlo population; the true model is called only for
        the training points. With ``compare_with_monte_carlo`` the whole
        population is also evaluated on the true model (counted separately in
        the result, and in this stage's evaluations).

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            population: Initial Monte Carlo population size.
            target_cov: Target coefficient of variation of the estimate; the
                population grows until it is met (``0.1`` by default, since
                halving it quadruples the population and the surrogate's
                prediction cost).
            max_added: Maximum number of training points added.
            compare_with_monte_carlo: Also run crude Monte Carlo on the same
                population.

        Returns:
            AK-MCS result with probability, true-model evaluations, and the
            final surrogate.

        Equation:
            [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002].

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011].

        """
        stream = self.stream("active_reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        result = ak_mcs(
            state,
            population,
            target_cov=target_cov,
            max_added=max_added,
            compare_with_monte_carlo=compare_with_monte_carlo,
            seed=stream,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
            "pf[AK-MCS]": result.probability,
            "cov": result.coefficient_of_variation,
            "population": float(result.population),
            "training runs": float(result.evaluations),
            "converged": "yes" if result.converged else f"no ({result.message})",
        }
        if compare_with_monte_carlo:
            summary["pf[MC, same population]"] = result.reference_probability
            summary["misclassified"] = float(result.misclassified)
        sign = "<=" if failure == "below" else ">="
        self._record(
            "active_reliability",
            f"AK-MCS P({quantity} {sign} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result

    @_stage("form")
    def form(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
    ) -> FORMResult:
        """Run FORM alone and record design point and importance factors.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.

        Returns:
            FORM result.

        Equation:
            [EQ-REL-FORM-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        stream = self.stream("form")
        result = form_hlrf(self.limit_state(quantity, threshold, failure=failure))
        summary: dict[str, float | str] = {
            "beta": result.beta,
            "pf": result.probability,
        }
        for name, value in result.importance_factors.items():
            summary[f"alpha2[{name}]"] = value
        summary["converged"] = _converged_text(result)
        summary["iterations"] = float(result.iterations)
        if not result.converged:
            self._warn(
                f"FORM for {quantity} did not converge ({result.message}); the "
                "design point and probability are not reliable"
            )
        self._record(
            "form",
            f"FORM for {quantity} threshold {threshold:g}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("risk", lambda study, args: args["size"])
    def risk(
        self,
        quantity: str,
        loss: LossFunction | Callable[[float], float],
        size: int = 20_000,
        *,
        design: str = "lhs",
        levels: Sequence[float] = (0.95, 0.99),
    ) -> RiskAssessment:
        """Assess the loss distribution induced by one QoI.

        Args:
            quantity: QoI name.
            loss: Loss function of the QoI value.
            size: Propagation sample size.
            design: Sampling design name.
            levels: VaR/CVaR confidence levels.

        Returns:
            Risk assessment.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-ROCKAFELLAR-URYASEV-2002], [REF-KAPLAN-GARRICK-1981].

        """
        stream = self.stream("risk")
        result = assess_risk(
            self._model, quantity, loss, size, design=design, seed=stream, levels=levels
        )
        summary: dict[str, float | str] = {
            "expected_loss": result.expected_loss.estimate,
            "P(loss>0)": result.probability_of_loss,
        }
        for level in levels:
            summary[f"VaR{level:g}"] = result.value_at_risk[level]
            summary[f"CVaR{level:g}"] = result.conditional_value_at_risk[level]
        self._record(
            "risk",
            f"loss risk of {quantity}",
            summary,
            result.model_evaluations,
            stream,
            result,
        )
        return result

    @_stage("calibration")
    def calibrate(
        self,
        experiments: Sequence[Experiment],
        noise: Sequence[NoiseModel],
        *,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
    ) -> CalibrationPosterior:
        """Calibrate the model's random inputs against observed experiments.

        The study model's joint input law is the prior over the calibration
        parameters; each experiment supplies observed QoIs and the
        deterministic conditions under which they were measured. Sampling is
        adaptive Metropolis in the prior's standard-normal space.

        Args:
            experiments: Observed experiments.
            noise: One noise model per observed QoI.
            draws: Kept draws per chain.
            warmup: Warm-up iterations per chain.
            chains: Number of chains.

        Returns:
            Posterior with diagnostics; also kept for :meth:`predict`.

        Equation:
            [EQ-CAL-NL-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Chapter 11].

        """
        stream = self.stream("calibration")
        problem = CalibrationProblem(self._model, tuple(experiments), tuple(noise))
        before = self._model.evaluations
        posterior = problem.sample_posterior(
            draws=draws, warmup=warmup, chains=chains, seed=stream
        )
        self._problem, self._posterior = problem, posterior
        self._active = "plain"
        summary: dict[str, float | str] = {"experiments": len(problem.experiments)}
        for entry in posterior.summary():
            summary[f"mean[{entry.name}]"] = entry.mean
            summary[f"sd[{entry.name}]"] = entry.standard_deviation
            summary[f"rhat[{entry.name}]"] = entry.rank_normalized_rhat
        self._record(
            "calibration",
            f"posterior from {len(problem.experiments)} experiments",
            summary,
            max(self._model.evaluations - before, posterior.evaluations),
            stream,
            posterior,
        )
        return posterior

    @_stage("discrepancy_calibration")
    def calibrate_with_discrepancy(
        self,
        experiments: Sequence[Experiment],
        quantity: str,
        noise: KnownNoise | LogLinearNoise,
        discrepancy: DiscrepancyProcess | None,
        *,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        method: Literal["adaptive-metropolis", "hmc", "nuts"] = "adaptive-metropolis",
        level: float = 0.95,
        grid: int = 41,
        grid_draws: int = 200,
        correlation_warning: float = 0.5,
        ratio_warning: float = 0.8,
    ) -> DiscrepancyAnalysis:
        """Calibrate one QoI with a Kennedy-O'Hagan model-discrepancy term.

        The observation model is ``y = eta(x, theta) + delta(x) + eps`` with a
        Gaussian-process discrepancy ``delta`` and known or log-linear noise
        ``eps``. As in :meth:`calibrate`, the study model's joint input law is
        the prior of ``theta``. The chains run on this stage's stream exactly
        as :meth:`~probstat.inverse.DiscrepancyCalibration.sample_posterior`
        with ``seed=stream``. The identifiability report (theta against the
        discrepancy hyperparameters, posterior over prior standard deviation)
        is recorded in the summary. With ``grid > 1`` and a one-input
        discrepancy, simulator-only and bias-corrected predictions on an even
        grid across the experiments' range are stored for figures (their
        model runs are charged to this stage).

        After this stage :meth:`predict` reports simulator-only and
        bias-corrected intervals, and :meth:`design` needs explicit ``noise``
        models (the design problem has no discrepancy term).

        Args:
            experiments: Observed experiments; each must observe ``quantity``.
            quantity: Calibrated QoI.
            noise: :class:`~probstat.inverse.KnownNoise` or
                :class:`~probstat.inverse.LogLinearNoise`.
            discrepancy: Gaussian-process discrepancy, or ``None`` for a
                calibration with the given noise model only.
            draws: Kept draws per chain.
            warmup: Warm-up iterations per chain.
            chains: Number of chains.
            method: ``"adaptive-metropolis"``, ``"hmc"``, or ``"nuts"``.
            level: Central probability of recorded intervals.
            grid: Number of grid predictions (``0`` or ``1`` skips the grid).
            grid_draws: Posterior draws used for the grid predictions.
            correlation_warning: ``|rho|`` threshold of the identifiability
                warning.
            ratio_warning: Standard-deviation ratio threshold of the warning.

        Returns:
            Calibration problem, posterior, identifiability report, and grid
            predictions; also kept for :meth:`predict`.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)`` or ``grid`` is
                negative (and as raised by ``DiscrepancyCalibration``).

        Equation:
            [EQ-KOH-MODEL-001], [EQ-KOH-POST-001], [EQ-KOH-IDENT-001],
            [EQ-KOH-PRED-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
            [REF-BRYNJARSDOTTIR-OHAGAN-2014].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly inside (0, 1)")
        if isinstance(grid, bool) or not isinstance(grid, int) or grid < 0:
            raise ValueError("grid must be a non-negative integer")
        stream = self.stream("discrepancy_calibration")
        calibration = DiscrepancyCalibration(
            self._model, tuple(experiments), quantity, noise, discrepancy
        )
        before = self._model.evaluations
        posterior = calibration.sample_posterior(
            draws=draws, warmup=warmup, chains=chains, seed=stream, method=method
        )
        report = calibration.identifiability(
            posterior,
            correlation_warning=correlation_warning,
            ratio_warning=ratio_warning,
        )
        grid_input: str | None = None
        predictions: tuple[DiscrepancyPrediction, ...] = ()
        if grid > 1 and discrepancy is not None and len(discrepancy.inputs) == 1:
            grid_input = discrepancy.inputs[0]
            observed_x = [
                float({**self._model.fixed, **item.conditions}[grid_input])
                for item in calibration.experiments
            ]
            low, high = min(observed_x), max(observed_x)
            if high > low:
                predictions = calibration.predict(
                    posterior,
                    tuple(
                        {grid_input: low + (high - low) * i / (grid - 1)}
                        for i in range(grid)
                    ),
                    max_draws=grid_draws,
                )
            else:
                grid_input = None
        analysis = DiscrepancyAnalysis(
            calibration, posterior, report, level, grid_input, predictions
        )
        self._discrepancy = analysis
        self._active = "discrepancy"
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "experiments": calibration.observation_count,
            "noise": "known" if isinstance(noise, KnownNoise) else "log-linear",
            "discrepancy": "none" if discrepancy is None else discrepancy.kernel,
            "method": method,
        }
        for entry in posterior.summary():
            summary[f"mean[{entry.name}]"] = entry.mean
            summary[f"sd[{entry.name}]"] = entry.standard_deviation
            summary[f"rhat[{entry.name}]"] = entry.rank_normalized_rhat
        summary["max |rho(theta, psi)|"] = report.max_abs_cross_correlation
        for name, ratio in report.standard_deviation_ratios.items():
            summary[f"sd ratio[{name}]"] = ratio
        summary["identifiability"] = report.warning or "no confounding flagged"
        if predictions:
            shifts = [item.discrepancy_mean for item in predictions]
            worst = max(range(len(shifts)), key=lambda i: abs(shifts[i]))
            summary["largest delta mean"] = shifts[worst]
            summary[f"at {grid_input}"] = float(
                predictions[worst].condition[str(grid_input)]
            )
        self._record(
            "discrepancy_calibration",
            f"Kennedy-O'Hagan posterior of {quantity} from "
            f"{calibration.observation_count} experiments",
            summary,
            self._model.evaluations - before,
            stream,
            analysis,
        )
        return analysis

    def _predict_discrepancy(
        self,
        quantities: Sequence[str] | None,
        conditions: Sequence[Mapping[str, float]],
        include_noise: bool,
        max_draws: int | None,
    ) -> DiscrepancyStudyPrediction:
        """Predict after a Kennedy-O'Hagan calibration and record the stage.

        Raises:
            ValueError: If another QoI is requested, or noise is requested
                with per-experiment known noise (unknown at new conditions).

        References:
            [REF-KENNEDY-OHAGAN-2001]. Equation [EQ-KOH-PRED-001].

        """
        assert self._discrepancy is not None
        analysis = self._discrepancy
        calibration = analysis.calibration
        quantity = calibration.quantity
        if quantities is not None and tuple(quantities) != (quantity,):
            raise ValueError(f"the discrepancy calibration predicts only {quantity!r}")
        targets = tuple(conditions)
        known: tuple[float, ...] | None = None
        if include_noise and isinstance(calibration.noise, KnownNoise):
            shared = calibration.noise.standard_deviations
            if not isinstance(shared, float):
                raise ValueError(
                    "include_noise needs one shared KnownNoise standard deviation"
                )
            known = tuple(shared for _ in targets)
        stream = self.stream("prediction")
        before = self._model.evaluations
        predictions = calibration.predict(
            analysis.posterior,
            targets,
            max_draws=max_draws,
            observation_standard_deviations=known,
        )
        draws = len(predictions[0].simulator)
        result = DiscrepancyStudyPrediction(
            quantity,
            predictions,
            analysis.level,
            draws,
            self._model.evaluations - before,
        )
        summary: dict[str, float | str] = {
            "draws": draws,
            "level": analysis.level,
            "calibration": "Kennedy-O'Hagan",
        }
        for index, item in enumerate(predictions):
            key = f"{quantity}#{index}"
            low, high = item.interval(analysis.level, kind="simulator")
            summary[f"simulator mean[{key}]"] = item.simulator_mean
            summary[f"simulator q_low[{key}]"] = low
            summary[f"simulator q_high[{key}]"] = high
            low, high = item.interval(analysis.level, kind="corrected")
            summary[f"corrected mean[{key}]"] = item.corrected_mean
            summary[f"corrected q_low[{key}]"] = low
            summary[f"corrected q_high[{key}]"] = high
            summary[f"delta mean[{key}]"] = item.discrepancy_mean
            if include_noise and item.noise_variances is not None:
                low, high = item.interval(analysis.level, kind="observed")
                summary[f"observed q_low[{key}]"] = low
                summary[f"observed q_high[{key}]"] = high
        self._record(
            "prediction",
            f"simulator-only and bias-corrected predictions at {len(targets)} "
            "conditions",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("prediction")
    def predict(
        self,
        quantities: Sequence[str] | None = None,
        conditions: Sequence[Mapping[str, float]] = ({},),
        *,
        include_noise: bool = False,
        max_draws: int | None = 1000,
    ) -> PosteriorPrediction | DiscrepancyStudyPrediction:
        """Push the latest posterior through the forward model.

        After :meth:`calibrate` the result is a
        :class:`~probstat.inverse.PosteriorPrediction`. After
        :meth:`calibrate_with_discrepancy` it is a
        :class:`DiscrepancyStudyPrediction` whose summary separates the
        simulator alone ``eta(x, theta)`` from the bias-corrected process
        ``eta + delta`` (and, with ``include_noise``, a new observation).
        The latest calibration of either kind is used.

        Args:
            quantities: QoIs to predict (all observed QoIs by default).
            conditions: Prediction conditions.
            include_noise: Add observation noise to the predictive draws.
            max_draws: Cap on posterior draws propagated.

        Returns:
            Posterior prediction separating latent and observed uncertainty.

        Raises:
            RuntimeError: If no calibration stage has been run.

        Equation:
            [EQ-BAYES-PRED-001], [EQ-KOH-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 1.3], [REF-KENNEDY-OHAGAN-2001].

        """
        if self._active == "discrepancy":
            return self._predict_discrepancy(
                quantities, conditions, include_noise, max_draws
            )
        if self._problem is None or self._posterior is None:
            raise RuntimeError("run calibrate() before predict()")
        stream = self.stream("prediction")
        result = predict(
            self._problem,
            self._posterior,
            quantities,
            conditions,
            include_noise=include_noise,
            max_draws=max_draws,
            seed=stream,
        )
        summary: dict[str, float | str] = {"draws": result.draws}
        for index, item in enumerate(result.predictions):
            summary[f"mean[{item.quantity}#{index}]"] = item.mean
            summary[f"q_low[{item.quantity}#{index}]"] = item.latent_quantiles[0]
            summary[f"q_high[{item.quantity}#{index}]"] = item.latent_quantiles[-1]
        self._record(
            "prediction",
            f"posterior predictive for {len(result.predictions)} targets",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("emulation")
    def emulate(
        self,
        quantity: str,
        *,
        method: Literal["pce", "gp"] = "pce",
        degree: int = 3,
        size: int | None = None,
        basis: str | None = None,
        conformal_alpha: float | None = None,
        conformal_size: int = 200,
    ) -> UQStudy:
        """Fit a surrogate of one QoI and return a study on the surrogate.

        The returned study shares this study's joint input law and draws its
        streams from a child of this study's root, so expensive analyses
        (large propagation, reliability, Sobol) can run on the emulator while
        this study records the training cost and accuracy. The child study is
        kept in :attr:`children`: its stages appear in this study's report
        and JSON record, and its model calls are counted as *surrogate*
        evaluations, never as forward-model evaluations of this study. The
        surrogate's approximation error (LOO error, conformal half-width) is
        in the emulation stage summary and is not propagated automatically.

        Args:
            quantity: QoI to emulate.
            method: Polynomial chaos (``pce``) or Gaussian process (``gp``).
            degree: PCE total degree (ignored for GP).
            size: Training-design size (defaults: PCE chooses from the number
                of terms; GP uses ten points per input).
            basis: PCE basis; by default Legendre when every input is an
                independent uniform law and Hermite (Nataf space) otherwise.
                Hermite polynomials on bounded inputs extrapolate badly in
                the far tails of large quasi-random designs.
            conformal_alpha: If given, calibrate split-conformal error bars of
                the emulator on ``conformal_size`` fresh true-model runs; the
                recorded half-width covers a new output with probability at
                least ``1 - conformal_alpha`` (marginally).
            conformal_size: Number of independent calibration runs.

        Returns:
            New study whose model evaluates the surrogate.

        Raises:
            ValueError: If ``method`` is unknown.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SUDRET-2007, Chapter 4], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        stream = self.stream("emulation")
        before = self._model.evaluations
        accuracy: dict[str, float | str]
        if method == "pce":
            chosen = basis or (
                "legendre"
                if self._model.inputs.is_independent
                and all(
                    isinstance(item.distribution, Uniform)
                    for item in self._model.inputs.inputs
                )
                else "hermite"
            )
            fitted: Any = fit_polynomial_chaos(
                self._model,
                quantity,
                degree=degree,
                basis=chosen,
                size=size,
                seed=stream,
            )
            accuracy = {
                "loo_error": fitted.loo_error,
                "terms": float(fitted.size),
                "basis": chosen,
            }
        elif method == "gp":
            fitted = fit_gaussian_process_to_model(
                self._model,
                quantity,
                size=size or 10 * self._model.dimension,
                seed=stream,
            )
            accuracy = {"log_marginal_likelihood": fitted.log_marginal_likelihood}
        else:
            raise ValueError("method must be pce or gp")
        surrogate = surrogate_probabilistic_model(
            self._model, {quantity: fitted}, name=f"{self._model.name}-{method}"
        )
        summary: dict[str, float | str] = {"method": method, "quantity": quantity}
        summary.update(accuracy)
        if conformal_alpha is not None:
            calibration = conformalize_surrogate(
                self._model,
                fitted,
                quantity=quantity,
                alpha=conformal_alpha,
                size=conformal_size,
                seed=stream.child("conformal"),
            )
            summary["conformal_half_width"] = calibration.regressor.quantile
            summary["conformal_coverage"] = calibration.regressor.guaranteed_coverage
        self._record(
            "emulation",
            f"{method.upper()} surrogate of {quantity}",
            summary,
            self._model.evaluations - before,
            stream,
            fitted,
        )
        child = UQStudy(
            surrogate,
            seed=stream.child("surrogate-study"),
            name=f"{self._name} / {method.upper()} surrogate of {quantity}",
            evaluation_kind="surrogate",
        )
        child._law_label, child._law_hash = self._law_label, self._law_hash
        self._children.append((len(self._records), child))
        return child

    @_stage("verification")
    def verify(
        self,
        quantity: str,
        grid_values: Sequence[float],
        refinement_ratio: float | None = None,
        *,
        grid_sizes: Sequence[float] | None = None,
        safety_factor: float | None = None,
        theoretical_order: float = 2.0,
        asymptotic_tolerance: float = 0.1,
        distribution: NumericalErrorDistribution = "uniform",
    ) -> SolutionVerification:
        """Estimate the numerical (discretization) uncertainty of one QoI.

        The stage runs :func:`~probstat.uq.verify_solution` on QoI values that
        the user computed on a sequence of grids; it adds no numerics. Three
        grids give the Celik et al. GCI (or the oscillation bound), four or
        more the least-squares observed order. The expanded uncertainty ``U``
        is converted to ``u_num`` with ``distribution`` and stored on the
        study, so ``validate(..., numerical_uncertainty="verified")`` folds it
        into the ASME V&V 20 validation uncertainty
        ``u_val = sqrt(u_num^2 + u_input^2 + u_D^2)``. :meth:`predict` does not
        add it; pass the returned statement explicitly as ``numerical=`` to
        ``probstat.validation.predict_with_uncertainty_components`` instead.
        No forward-model evaluations are charged.

        Args:
            quantity: Model QoI the grid solutions belong to.
            grid_values: QoI value on each grid. With ``refinement_ratio`` they
                are ordered fine to coarse; with ``grid_sizes`` they pair with
                the sizes in any order.
            refinement_ratio: Constant ratio ``r > 1`` between consecutive
                grids (spacings ``1, r, r^2, ...``). Give this or
                ``grid_sizes``.
            grid_sizes: Representative grid spacings, one per value.
            safety_factor: ``F_s`` of the three-grid GCI (``None`` uses the
                library default 1.25). Ignored by the four-grid rule, which
                picks 1.25 or 3 from the order credibility.
            theoretical_order: Formal order of the discretization scheme.
            asymptotic_tolerance: Allowed deviation of the asymptotic-range
                ratio (at the theoretical order) from one.
            distribution: ``U`` to ``u_num`` conversion, ``"uniform"``
                (``U / sqrt(3)``) or ``"normal95"`` (``U / 1.96``).

        Returns:
            Solution-verification statement from the library call.

        Raises:
            KeyError: If ``quantity`` is not a model QoI.
            ValueError: If both or neither of ``refinement_ratio`` and
                ``grid_sizes`` are given, the ratio is not above one, or the
                grid data are rejected by :func:`~probstat.uq.verify_solution`.

        Equation:
            [EQ-VV-GCI-004], [EQ-VV-GCI-005], [EQ-VV-LSQ-002],
            [EQ-VV-OSC-001], [EQ-VV-UNUM-001].

        References:
            [REF-CELIK-ETAL-2008], [REF-ROACHE-1998, Chapters 4-5],
            [REF-ECA-HOEKSTRA-2014], [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        self._model.quantity(quantity)
        values = tuple(grid_values)
        if (refinement_ratio is None) == (grid_sizes is None):
            raise ValueError("give exactly one of refinement_ratio and grid_sizes")
        if grid_sizes is None:
            ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
            if ratio <= 1.0:
                raise ValueError("refinement_ratio must be greater than one")
            spacings = tuple(ratio**index for index in range(len(values)))
        else:
            spacings = tuple(grid_sizes)
        stream = self.stream("verification")
        result = verify_solution(
            quantity,
            spacings,
            values,
            theoretical_order=theoretical_order,
            safety_factor=(
                DEFAULT_SAFETY_FACTOR if safety_factor is None else safety_factor
            ),
            asymptotic_tolerance=asymptotic_tolerance,
        )
        u_num = result.standard_uncertainty(distribution)
        self._verifications[quantity] = (result, u_num)
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "method": result.method,
            "grids": result.grid_count,
            "convergence": result.convergence.value,
            "fine_value": result.fine_value,
            "observed_order": (
                "n/a" if result.observed_order is None else result.observed_order
            ),
            "extrapolated": (
                "n/a" if result.extrapolated is None else result.extrapolated
            ),
            "safety_factor": result.safety_factor,
            "U": result.expanded_uncertainty,
            "u_num": u_num,
            "u_num.distribution": distribution,
        }
        study = result.study
        if isinstance(study, GridConvergenceResult):
            ratio_at_theory = study.asymptotic_ratio_at(theoretical_order)
            summary["gci_fine"] = study.gci_fine
            summary["asymptotic_ratio"] = ratio_at_theory
            summary["asymptotic_range"] = str(
                abs(ratio_at_theory - 1.0) <= asymptotic_tolerance
            )
        elif isinstance(study, LeastSquaresConvergence):
            summary["asymptotic_range"] = str(study.credible_order)
        else:
            summary["asymptotic_range"] = "n/a (oscillatory)"
        summary["reliable"] = str(result.reliable)
        summary["warnings"] = "; ".join(result.warnings) or "none"
        self._record(
            "verification",
            f"solution verification of {quantity} on {result.grid_count} grids",
            summary,
            0,
            stream,
            result,
        )
        return result

    def numerical_uncertainty(self, quantity: str) -> float:
        """Return ``u_num`` of the latest :meth:`verify` run for a QoI.

        Args:
            quantity: QoI name.

        Returns:
            Standard numerical uncertainty in QoI units.

        Raises:
            KeyError: If the QoI has not been verified in this study.

        Equation:
            [EQ-VV-UNUM-001].

        References:
            [REF-JCGM-100-2008, Sections 4.3.7 and 6.2].

        """
        if quantity not in self._verifications:
            raise KeyError(f"run verify({quantity!r}, ...) first")
        return self._verifications[quantity][1]

    @_stage(
        "validation",
        lambda study, args: args["sample_size"] * len(_sized(args["experiments"])),
    )
    def validate(
        self,
        quantity: str,
        experiments: Sequence[ValidationExperiment],
        *,
        tolerance: float,
        sample_size: int = 2000,
        significance: float = 0.05,
        coverage_levels: Sequence[float] = (0.5, 0.9),
        numerical_uncertainty: float
        | NumericalErrorBudget
        | Literal["verified"] = 0.0,
    ) -> ValidationAssessment:
        """Compare model predictions with validation experiments.

        The verdict is scoped: consistent or inconsistent with the data at the
        stated tolerance and significance inside the validation domain, never
        a general claim that the model is validated.

        Proper-scoring and calibration evidence is recorded next to the
        verdict without changing its criteria: the mean ensemble CRPS of each
        experiment (and over all observations) in QoI units, and randomized
        ensemble PIT values of every observation (stream ``validation/k/pit``)
        with their mean and the empirical coverage of central predictive
        intervals at ``coverage_levels`` (Clopper-Pearson 95% intervals).
        PIT values of replicates within one experiment share its predictive
        sample, so they are not independent cases.

        ``numerical_uncertainty`` enters only the ASME V&V 20 comparison of
        each experiment (``u_val = sqrt(u_num^2 + u_input^2 + u_D^2)``);
        ``"verified"`` uses the ``u_num`` stored by :meth:`verify` for the
        quantity. The area-metric verdict, u-pooling, CRPS, and PIT do not
        depend on it. A non-zero declaration adds ``u_num`` to the summary.

        Args:
            quantity: QoI compared with the observations.
            experiments: Validation experiments with conditions and data.
            tolerance: Area-metric tolerance in QoI units.
            sample_size: Predictive samples per experiment.
            significance: Significance level of the u-pooling test.
            coverage_levels: Nominal central coverages reported from the PIT
                values (each in ``(0, 1)``).
            numerical_uncertainty: ``u_num``, a verification error budget, or
                ``"verified"`` for the value stored by :meth:`verify`.

        Returns:
            Validation assessment with per-experiment and pooled metrics.

        Raises:
            KeyError: If ``"verified"`` is requested before :meth:`verify`
                was run for the quantity.
            ValueError: If ``numerical_uncertainty`` is another string.

        Equation:
            [EQ-VAL-AREA-001], [EQ-VAL-SCORE-CRPS-002], [EQ-VAL-CAL-PIT-002],
            [EQ-VAL-CAL-COV-001], [EQ-VAL-ASME-001].

        References:
            [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-OBERKAMPF-ROY-2025, Part IV],
            [REF-GNEITING-RAFTERY-2007], [REF-GNEITING-BALABDAOUI-RAFTERY-2007],
            [REF-CLOPPER-PEARSON-1934], [REF-ASME-VV20-2009].

        """
        declared = numerical_uncertainty
        u_num: float | NumericalErrorBudget
        if isinstance(declared, str):
            if declared != "verified":
                raise ValueError("numerical_uncertainty string must be 'verified'")
            u_num = self.numerical_uncertainty(quantity)
        else:
            u_num = declared
        stream = self.stream("validation")
        before = self._model.evaluations
        result = assess_validation(
            self._model,
            quantity,
            experiments,
            tolerance=tolerance,
            sample_size=sample_size,
            significance=significance,
            seed=stream,
            numerical_uncertainty=u_num,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "consistent": str(result.consistent),
            "u_pooling_area": result.u_pooling.area,
            "log_bayes_factor": result.log_bayes_factor,
        }
        for index, item in enumerate(result.experiments, start=1):
            # Index-qualified keys keep every experiment even if names repeat.
            summary[f"area[{index}:{item.experiment.name}]"] = item.area
        u_declared = result.experiments[0].asme.numerical_uncertainty
        if u_declared > 0.0:
            summary["u_num"] = u_declared
            summary["u_num.source"] = (
                "verify stage" if isinstance(declared, str) else "declared"
            )
        ensembles: list[tuple[float, ...]] = []
        observations: list[float] = []
        scores: list[float] = []
        for index, item in enumerate(result.experiments, start=1):
            summary[f"crps[{index}:{item.experiment.name}]"] = item.crps
            for value in item.experiment.observations:
                ensembles.append(item.observable_predictions)
                observations.append(value)
                scores.append(crps_ensemble(item.observable_predictions, value))
        summary["crps.mean"] = math.fsum(scores) / len(scores)
        pit = ensemble_pit(ensembles, observations, seed=stream.child("pit"))
        summary["pit.mean"] = math.fsum(pit) / len(pit)
        for entry in pit_coverage(pit, coverage_levels):
            key = f"coverage[{100.0 * entry.nominal:.6g}%]"
            summary[key] = entry.rate
            summary[f"{key}.interval"] = (
                f"{entry.covered}/{entry.size}, "
                f"[{entry.interval.lower:.3f}, {entry.interval.upper:.3f}]"
            )
        self._record(
            "validation",
            f"validation of {quantity} against {len(result.experiments)} experiments",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result

    def _fresh_cost(self, arguments: Mapping[str, Any], default: int) -> int | None:
        """Return the known cost of a stage that may draw a fresh sample.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if arguments.get("reuse"):
            return None
        size = arguments.get("size")
        return default if size is None else int(size)

    def _output_sample(
        self,
        stream: RandomStream,
        quantity: str,
        size: int | None,
        design: str,
        default_size: int,
        reuse: bool,
    ) -> tuple[PropagationStudy, int, str]:
        """Return a plain Monte Carlo output sample for a sample-based stage.

        By default ``size`` (default ``default_size``) fresh runs are drawn by
        plain Monte Carlo from the stage's own stream, so the result depends
        only on the root seed and the run index of the stage, never on which
        other stages ran before it. With ``reuse=True`` the latest complete
        propagation is reused at no cost *only if* it is a plain Monte Carlo
        sample; LHS and quasi-random samples are not independent draws (the
        KSG and peaks-over-threshold estimators assume independence), so a
        fresh Monte Carlo sample is drawn instead and the source says why.

        Raises:
            ValueError: If ``quantity`` is not a model QoI, ``design`` is not
                ``"mc"``, or both ``size`` and ``reuse`` are given.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if quantity not in self._model.quantity_names:
            raise ValueError(f"{quantity!r} is not a quantity of the study model")
        if design != "mc":
            raise ValueError(
                "sample-based stages need independent draws; only design='mc' "
                "is supported"
            )
        if reuse and size is not None:
            raise ValueError("give either size (fresh sample) or reuse=True")
        reason = ""
        if reuse:
            reason = "no propagation stage to reuse"
            for record in reversed(self._records):
                if record.stage == "propagation" and record.complete:
                    previous: PropagationStudy = record.result
                    if previous.design.method == "mc":
                        return (
                            previous,
                            0,
                            "propagation stage (plain Monte Carlo, stream seed "
                            f"{record.seed})",
                        )
                    reason = (
                        f"latest propagation used {previous.design.method}; "
                        "only plain Monte Carlo samples are reused"
                    )
                    break
        sample = propagate(
            self._model,
            default_size if size is None else size,
            design="mc",
            seed=stream,
        )
        source = "fresh mc sample" + (f" ({reason})" if reason else "")
        return sample, sample.model_evaluations, source

    @_stage("extremes", lambda study, args: study._fresh_cost(args, 4096))
    def extremes(
        self,
        quantity: str,
        *,
        tail: Literal["upper", "lower"] = "upper",
        threshold_quantile: float = 0.95,
        threshold: float | None = None,
        level: float | None = None,
        return_periods: Sequence[float] = (100.0, 1000.0),
        method: Literal["mle", "pwm"] = "mle",
        confidence: float = 0.95,
        size: int | None = None,
        design: str = "mc",
        reuse: bool = False,
    ) -> TailAnalysis:
        """Fit a peaks-over-threshold model to the tail of one QoI.

        The sample is ``size`` (default 4096) fresh plain Monte Carlo runs
        drawn from this stage's stream, so the result does not depend on the
        order of stages. ``reuse=True`` reuses the latest propagation at no
        cost when it is a plain Monte Carlo sample (a quasi-random sample is
        never reused: its points are not independent). Each model run is one
        observation (``observations_per_year = 1``), so a return period ``T``
        is a number of runs and the ``T``-run level has tail probability
        ``1/T``. The lower tail is analyzed as the upper tail of ``-Y``.

        Args:
            quantity: QoI name.
            tail: ``"upper"`` (large values) or ``"lower"`` (small values).
            threshold_quantile: Threshold as the sample quantile of the
                oriented values (used when ``threshold`` is ``None``).
            threshold: Explicit threshold in QoI units.
            level: Optional level whose tail probability is estimated both by
                the POT model and empirically (Wilson interval).
            return_periods: Return periods in model runs (each must exceed
                ``1/zeta_u``).
            method: GPD estimator, ``"mle"`` (with delta intervals) or
                ``"pwm"`` (point estimates only).
            confidence: Two-sided interval confidence.
            size: Fresh sample size (default 4096).
            design: Sampling design of a fresh sample; only ``"mc"``.
            reuse: Reuse the latest plain Monte Carlo propagation.

        Returns:
            Tail analysis in QoI units.

        Raises:
            ValueError: If the quantity, threshold quantile, tail, or level is
                invalid (a level must lie beyond the threshold).

        Equation:
            [EQ-EVA-GPD-001], [EQ-EVA-RATE-001], [EQ-EVA-POT-001],
            [EQ-EVA-POT-002], [EQ-MC-PROB-001].

        References:
            [REF-COLES-2001, Section 4.3], [REF-WILSON-1927].

        """
        if tail not in ("upper", "lower"):
            raise ValueError("tail must be 'upper' or 'lower'")
        if threshold is None and not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie strictly inside (0, 1)")
        stream = self.stream("extremes")
        sample, evaluations, source = self._output_sample(
            stream, quantity, size, design, 4096, reuse
        )
        sign = 1.0 if tail == "upper" else -1.0
        oriented = tuple(sign * value for value in sample[quantity].values)
        cut = (
            quantile(oriented, threshold_quantile)
            if threshold is None
            else sign * threshold
        )
        fit = fit_pot(oriented, cut, observations_per_year=1.0, method=method)
        levels = tuple(
            _orient_return_level(fit.return_level(period, confidence=confidence), sign)
            for period in return_periods
        )
        pot_probability: float | None = None
        empirical: ProbabilityEstimate | None = None
        if level is not None:
            point = sign * level
            if point < cut:
                raise ValueError(
                    "level lies inside the threshold; raise threshold_quantile "
                    "or pass a lower threshold"
                )
            pot_probability = fit.annual_exceedance_rate(point)
            empirical = estimate_probability(
                (value > point for value in oriented), confidence=confidence
            )
        result = TailAnalysis(
            quantity,
            tail,
            fit,
            levels,
            level,
            pot_probability,
            empirical,
            len(oriented),
            source,
            tuple(sign * value for value in sorted(v for v in oriented if v > cut)),
        )
        sigma_se, xi_se = fit.gpd.standard_errors
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "tail": tail,
            "source": source,
            "design": sample.design.method,
            "sample_size": len(oriented),
            "threshold": result.threshold,
            "threshold_rule": (
                f"q{threshold_quantile:g}" if threshold is None else "given"
            ),
            "exceedances": fit.rate.exceedances,
            "zeta_u": fit.rate.probability,
            "sigma": fit.gpd.sigma,
            "sigma_se": sigma_se,
            "xi": fit.gpd.xi,
            "xi_se": xi_se,
        }
        n = len(oriented)
        for item in levels:
            key = f"x[T={item.return_period:g}]"
            summary[key] = item.level
            summary[f"{key}.low"] = item.lower
            summary[f"{key}.high"] = item.upper
            summary[f"empirical {key}"] = (
                sign * quantile(oriented, 1.0 - 1.0 / item.return_period)
                if item.return_period <= n
                else "n/a (return period exceeds the sample size)"
            )
        if level is not None and empirical is not None and pot_probability is not None:
            summary["level"] = level
            summary["P_pot"] = pot_probability
            summary["P_empirical"] = empirical.probability
            summary["P_empirical.low"] = empirical.confidence_interval.lower
            summary["P_empirical.high"] = empirical.confidence_interval.upper
            summary["P_pot/P_empirical"] = (
                pot_probability / empirical.probability
                if empirical.probability > 0.0
                else math.inf
            )
        self._record(
            "extremes",
            f"POT {tail} tail of {quantity}, threshold {result.threshold:g}",
            summary,
            evaluations,
            stream,
            result,
        )
        return result

    @_stage(
        "information",
        lambda study, args: study._fresh_cost(args, args["max_points"]),
    )
    def information_sensitivity(
        self,
        quantity: str,
        *,
        max_points: int = 512,
        k: int = 3,
        unit: InformationUnit = "nats",
        size: int | None = None,
        design: str = "mc",
        reuse: bool = False,
    ) -> InformationSensitivity:
        """Estimate the mutual information between each input and one QoI.

        ``I(X_i; Y)`` is zero exactly when ``Y`` is independent of ``X_i``
        and is invariant under monotone transformations of either, so it
        sees non-monotone and higher-moment dependence that variance-based
        indices can miss, and it needs no independence between inputs. It is
        a moment-independent complement to Sobol indices, not a variance
        share: values are in nats (or bits) and do not sum to anything.

        The sample is ``size`` (default ``max_points``) fresh plain Monte
        Carlo runs drawn from this stage's stream, so the result does not
        depend on the order of stages; ``reuse=True`` uses the first
        ``max_points`` runs of the latest propagation instead when it is a
        plain Monte Carlo sample (quasi-random points are not independent and
        bias the KSG estimator). The KSG estimator costs ``O(n^2)`` per
        input, hence the cap.

        Args:
            quantity: QoI name.
            max_points: Maximum number of runs entering the estimator.
            k: KSG neighbour order.
            unit: ``"nats"`` or ``"bits"``.
            size: Fresh sample size (default ``max_points``).
            design: Sampling design of a fresh sample; only ``"mc"``.
            reuse: Reuse the latest plain Monte Carlo propagation.

        Returns:
            Mutual information per input.

        Raises:
            ValueError: If the quantity or ``max_points`` is invalid, or the
                sample has duplicated (input, output) pairs.

        Equation:
            [EQ-INFO-MI-001], [EQ-INFO-KSG-001].

        References:
            [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004],
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        if isinstance(max_points, bool) or not isinstance(max_points, int):
            raise ValueError("max_points must be an integer")
        if max_points < 2:
            raise ValueError("max_points must be at least two")
        stream = self.stream("information")
        sample, evaluations, source = self._output_sample(
            stream, quantity, size, design, max_points, reuse
        )
        outputs = sample[quantity].values[:max_points]
        count = len(outputs)
        estimates = tuple(
            ksg_mutual_information(
                sample.input_column(name)[:count], outputs, k=k, unit=unit
            )
            for name in sample.input_names
        )
        result = InformationSensitivity(
            quantity, sample.input_names, estimates, k, count, unit, source
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "unit": unit,
            "k": k,
            "sample_size": count,
            "source": source,
        }
        for name, value in zip(sample.input_names, estimates, strict=True):
            summary[f"MI[{name}]"] = value
        summary["ranking"] = " > ".join(result.ranking())
        self._record(
            "information",
            f"mutual information of {quantity} with each input",
            summary,
            evaluations,
            stream,
            result,
        )
        return result

    @_stage("design")
    def design(
        self,
        candidates: Sequence[Condition],
        size: int = 1,
        *,
        noise: Sequence[NoiseModel] | None = None,
        include_experiments: bool = True,
        allow_repeats: bool = True,
        method: str = "laplace",
        samples: int = 32,
        outer: int = 1000,
        inner: int = 200,
        reuse_samples: bool = True,
    ) -> DesignRecommendation:
        """Choose new experimental conditions by expected information gain.

        The design problem reuses the calibration problem of the latest
        :meth:`calibrate` (prior = model input law, its Gaussian noise
        models) or, before any calibration, the ``noise`` models given here.
        With ``include_experiments`` the calibration's experimental conditions
        form the baseline, so each greedy step maximizes the expected
        conditional gain given data at those conditions. The EIG is a
        preposterior quantity under the prior: it does not condition on the
        observed values. Conditions are chosen greedily
        (stream ``design/greedy``); the full design is then re-estimated on an
        independent stream (``design/final``) to report its EIG with a
        standard error. After :meth:`calibrate_with_discrepancy` the
        Kennedy-O'Hagan experiments form the baseline, but ``noise`` is
        required because the design problem has no discrepancy term.

        Args:
            candidates: Candidate conditions (deterministic model inputs).
            size: Number of new conditions to choose.
            noise: Known-scale Gaussian noise models; overrides the
                calibration's when given.
            include_experiments: Use the calibration experiments as baseline.
            allow_repeats: Allow a candidate to be chosen more than once.
            method: ``"linearized"``, ``"laplace"``, or ``"nested"``.
            samples: Prior draws for ``"laplace"``.
            outer: Outer samples for ``"nested"``.
            inner: Inner samples for ``"nested"`` without reuse.
            reuse_samples: Sample reuse for ``"nested"``.

        Returns:
            Greedy selection and the final EIG estimate.

        Raises:
            ValueError: If there is neither a plain calibration problem nor
                ``noise``, or the design arguments are invalid.

        Equation:
            [EQ-OED-EIG-001], [EQ-OED-GREEDY-001], [EQ-OED-LAP-001].

        References:
            [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013].

        """
        plain = self._problem if self._active == "plain" else None
        if noise is not None:
            problem = BayesianDesignProblem(self._model, tuple(noise))
        elif plain is not None:
            problem = BayesianDesignProblem.from_calibration(plain)
        elif self._active == "discrepancy":
            raise ValueError(
                "design() after calibrate_with_discrepancy() needs known-scale "
                "noise models via noise= (the design problem has no discrepancy)"
            )
        else:
            raise ValueError(
                "design() needs a calibration problem: run calibrate() first "
                "or pass known-scale noise models via noise="
            )
        performed: tuple[Experiment, ...] = ()
        if plain is not None:
            performed = plain.experiments
        elif self._active == "discrepancy" and self._discrepancy is not None:
            performed = self._discrepancy.calibration.experiments
        baseline: tuple[Condition, ...] = ()
        if include_experiments:
            baseline = tuple(
                MappingProxyType(dict(item.conditions)) for item in performed
            )
        pool: tuple[Condition, ...] = tuple(
            MappingProxyType(dict(item)) for item in candidates
        )
        stream = self.stream("design")
        before = self._model.evaluations
        selection = greedy_design(
            problem,
            pool,
            size,
            baseline=baseline,
            allow_repeats=allow_repeats,
            method=method,
            samples=samples,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=stream.child("greedy"),
        )
        estimate = expected_information_gain(
            problem,
            baseline + selection.selected,
            method=method,
            samples=samples,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=stream.child("final"),
        )
        result = DesignRecommendation(selection, estimate, baseline, pool)
        summary: dict[str, float | str] = {
            "method": method,
            "candidates": len(pool),
            "baseline_conditions": len(baseline),
            "baseline_eig": selection.baseline_information_gain,
        }
        for step, (condition, gain, increment) in enumerate(
            zip(
                selection.selected,
                selection.information_gains,
                selection.increments,
                strict=True,
            ),
            start=1,
        ):
            summary[f"condition[{step}]"] = _format_condition(condition)
            summary[f"eig[{step}]"] = gain
            summary[f"gain[{step}]"] = increment
        summary["eig"] = estimate.value
        summary["eig_se"] = estimate.standard_error
        summary["eig_bias"] = estimate.bias
        self._record(
            "design",
            f"greedy {method} design of {len(selection.selected)} new conditions",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result

    def with_inputs(
        self, inputs: JointInput, *, label: str | None = None
    ) -> UQStudy:
        """Replace the joint input law used by every later stage.

        The model becomes ``dataclasses.replace(model, inputs=inputs)``: same
        forward map, QoIs, fixed parameters, and failure policy, with its own
        evaluation counter linked to the previous one. Earlier records keep
        the law they used; each record names its law by label and hash. The
        change is recorded as an ``input_law`` stage with no evaluations.

        Args:
            inputs: New joint law over exactly the current input names.
            label: Report label (default ``"user-supplied #k"``).

        Returns:
            This study (for chaining).

        Raises:
            TypeError: If ``inputs`` is not a JointInput.
            ValueError: If the input names differ from the model's.

        References:
            [REF-SMITH-UQ-2024, Chapters 3 and 8]. No equation is evaluated.

        """
        if not isinstance(inputs, JointInput):
            raise TypeError("inputs must be a JointInput")
        if inputs.names != self._model.inputs.names:
            raise ValueError(
                f"input names {list(inputs.names)} must equal the model's "
                f"{list(self._model.inputs.names)} (same order)"
            )
        self._law_changes += 1
        chosen = label or f"user-supplied #{self._law_changes}"
        self._set_inputs(inputs, chosen, {"source": "with_inputs"}, None)
        return self

    def _set_inputs(
        self,
        inputs: JointInput,
        label: str,
        summary: Mapping[str, float | str],
        result: Any,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """Install a new input law and record the change.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        previous_label, previous_hash = self._law_label, self._law_hash
        self._model = dataclasses.replace(self._model, inputs=inputs)
        self._law_label = label
        self._law_hash = input_law_hash(inputs)
        values: dict[str, float | str] = dict(summary)
        values["previous law"] = f"{previous_label} ({previous_hash})"
        values["new law"] = f"{label} ({self._law_hash})"
        for item in inputs.inputs:
            values[f"law[{item.name}]"] = type(item.distribution).__name__
            values[f"mean[{item.name}]"] = item.mean
            values[f"sd[{item.name}]"] = item.standard_deviation
        values["dependence"] = (
            "independent"
            if inputs.is_independent
            else f"{inputs.copula_family.capitalize()} copula"
        )
        self._record(
            "input_law",
            f"input law set to {label}",
            values,
            0,
            self.stream("input_law"),
            inputs if result is None else result,
            settings=settings,
        )

    def adopt_posterior(
        self,
        inputs: Sequence[str] | None = None,
        *,
        method: PosteriorMethod = "gaussian_copula",
        candidates: Sequence[str] = DEFAULT_POSTERIOR_FAMILIES,
        max_draws: int = 4000,
    ) -> JointInput:
        """Use the latest calibration posterior as the input law from now on.

        The posterior is known only through MCMC draws; it is approximated
        by marginals plus a Gaussian copula
        (:func:`~probstat.workflow.chaining.posterior_joint_input`):
        ``"gaussian_copula"`` fits a parametric family per input by AIC among
        ``candidates`` and a Spearman-rank copula; ``"empirical"`` uses the
        interpolated empirical law of the draws (support cut at their range)
        and a normal-score copula. A Gaussian copula cannot represent curved
        or tail dependence, so the approximation is checked in the record:
        it lists posterior and adopted means and standard deviations side by
        side. Inputs not in ``inputs`` keep their prior laws (independent of
        the adopted ones). Later stages, including a new :meth:`calibrate`,
        then use the posterior; calibrating again on the same data would use
        the data twice.

        Args:
            inputs: Calibrated inputs to adopt (default: all model inputs).
            method: ``"gaussian_copula"`` or ``"empirical"``.
            candidates: Candidate family names for ``"gaussian_copula"``.
            max_draws: Posterior draws used after even thinning.

        Returns:
            The adopted joint input law.

        Raises:
            RuntimeError: If no plain :meth:`calibrate` has been run.

        Equation:
            [EQ-INF-IC-001], [EQ-MOD-COPULA-FIT-001], [EQ-WF-EMPIRICAL-001].

        References:
            [REF-GENEST-GHOUDI-RIVEST-1995], [REF-LIU-DERKIUREGHIAN-1986],
            [REF-GELMAN-ETAL-2013, Chapter 1].

        """
        if self._posterior is None:
            raise RuntimeError("run calibrate() before adopt_posterior()")
        draws = self._posterior.draws()
        law = posterior_joint_input(
            self._model.inputs,
            draws,
            inputs,
            method=method,
            candidates=candidates,
            max_draws=max_draws,
        )
        calibration_stage = sum(r.stage == "calibration" for r in self._records)
        self._law_changes += 1
        label = f"posterior #{self._law_changes} ({method})"
        summary: dict[str, float | str] = {
            "source": f"calibration run {calibration_stage}",
            "method": method,
            "draws": float(min(len(draws), max_draws)),
        }
        adopted = list(self._model.inputs.names if inputs is None else inputs)
        for name in adopted:
            column = [float(draw[name]) for draw in draws]
            mean = math.fsum(column) / len(column)
            spread = math.sqrt(
                math.fsum((v - mean) ** 2 for v in column) / (len(column) - 1)
            )
            summary[f"posterior mean[{name}]"] = mean
            summary[f"posterior sd[{name}]"] = spread
        self._set_inputs(
            law,
            label,
            summary,
            law,
            settings={
                "inputs": _setting_value(None if inputs is None else tuple(inputs)),
                "method": method,
                "candidates": _setting_value(tuple(candidates)),
                "max_draws": max_draws,
            },
        )
        return law

    def provenance(self) -> dict[str, object]:
        """Return library, Python, platform, time, and seed provenance.

        References:
            Project reporting policy. No equation is evaluated.

        """
        return {
            "library": "probstat",
            "library_version": probstat.__version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "created_utc": self._created,
            "root_seed": self._root.seed,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the study (no pickle needed).

        It holds provenance, the model (input laws, QoIs, fixed parameters),
        the current input law with its hash, the budget and totals, every
        stage record (:meth:`StageRecord.to_dict`), and the surrogate studies
        of :meth:`emulate`. Domain result objects are named, not serialized.

        References:
            Project reporting policy. No equation is evaluated.

        """
        model = self._model
        return {
            "format": "probstat.UQStudy/1",
            "name": self._name,
            "evaluation_kind": self._kind,
            "provenance": self.provenance(),
            "model": {
                "name": model.name,
                "inputs": [
                    {
                        "name": item.name,
                        "law": repr(item.distribution)
                        if len(repr(item.distribution)) <= 200
                        else type(item.distribution).__name__,
                        "mean": json_safe(item.mean),
                        "sd": json_safe(item.standard_deviation),
                    }
                    for item in model.inputs.inputs
                ],
                "correlation": json_safe(model.inputs.correlation),
                "quantities": [
                    {"name": q.name, "description": q.description, "unit": q.unit}
                    for q in model.quantities
                ],
                "fixed": json_safe(model.fixed),
                "on_failure": model.on_failure,
            },
            "input_law": {"label": self._law_label, "hash": self._law_hash},
            "evaluation_budget": self._budget,
            "totals": {
                "evaluations": self.total_evaluations,
                "failed_evaluations": self.total_failed_evaluations,
                "cache_hits": sum(r.cache_hits for r in self._records),
            },
            "stages": [record.to_dict() for record in self._records],
            "surrogate_studies": [
                {"after_stage": index, "study": child.to_dict()}
                for index, child in self._children
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return :meth:`to_dict` as JSON text (no NaN or infinity tokens).

        Args:
            indent: JSON indentation (``None`` for one line).

        References:
            Project reporting policy. No equation is evaluated.

        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def save_json(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to a file (parents are created).

        Args:
            path: Destination path.

        Returns:
            The written path.

        References:
            Project reporting policy. No equation is evaluated.

        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    def report(self) -> str:
        """Return a Markdown report of every stage with costs and provenance.

        The header states the inputs, QoIs, dependence, current input law
        (label and hash), root seed, evaluation totals (failed runs and cache
        hits separately), the budget, and the library, Python, platform, and
        UTC creation time. Each stage states its status, cost, stream seed,
        input law, settings, and warnings, followed by its summary table.
        Surrogate studies created by :meth:`emulate` follow their emulation
        stage, with surrogate evaluations labelled as such.

        Returns:
            Markdown text: one section per stage with a two-column table.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        return "\n".join(self._report_lines(1))

    def _report_lines(self, level: int) -> list[str]:
        """Return the Markdown lines of this study at a heading level.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        hashes = "#" * level
        forward = self._kind == "forward-model"
        unit = "Forward-model evaluations" if forward else "Surrogate evaluations"
        dependence = (
            "independent"
            if self._model.inputs.is_independent
            else f"{self._model.inputs.copula_family.capitalize()} copula"
        )
        title = "UQ study" if forward else "Surrogate study"
        lines = [
            f"{hashes} {title}: {self._name}",
            "",
            f"- Inputs: {', '.join(self._model.inputs.names)}",
            f"- Quantities of interest: {', '.join(self._model.quantity_names)}",
            f"- Dependence: {dependence}",
            f"- Input law: {self._law_label} (hash {self._law_hash})",
            f"- Root seed: {self._root.seed}",
            f"- {unit}: {self.total_evaluations:,}",
        ]
        failed = self.total_failed_evaluations
        if failed:
            attempts = failed + self.total_evaluations
            lines.append(
                f"- **Failed evaluations: {failed:,} "
                f"({100.0 * failed / attempts:.3g}% of attempts)**"
            )
        hits = sum(record.cache_hits for record in self._records)
        if hits:
            lines.append(f"- Cache hits (not evaluations): {hits:,}")
        if self._budget is not None:
            lines.append(
                f"- Evaluation budget: {self._budget:,} "
                f"({self.remaining_budget:,} remaining)"
            )
        if level == 1:
            info = self.provenance()
            lines.append(
                f"- Software: probstat {info['library_version']} · Python "
                f"{info['python_version']} · {info['platform']}"
            )
            lines.append(f"- Created (UTC): {info['created_utc']}")
        lines.append("")
        children: dict[int, list[UQStudy]] = {}
        for index, child in self._children:
            children.setdefault(index, []).append(child)
        count_label = "Evaluations" if forward else "Surrogate evaluations"
        for index, record in enumerate(self._records, start=1):
            lines.append(f"{hashes}# {index}. {record.stage}: {record.label}")
            lines.append("")
            parts = [f"{count_label}: {record.evaluations:,}"]
            if record.failed_evaluations:
                parts.append(f"failed: {record.failed_evaluations:,}")
            if record.cache_hits:
                parts.append(f"cache hits: {record.cache_hits:,}")
            parts.append(f"stream seed: {record.seed}")
            parts.append(f"input law: {record.input_law} ({record.input_law_hash})")
            if not record.complete:
                parts.append(f"status: **{record.status}**")
            lines.append(" · ".join(parts))
            if record.error:
                lines.extend(["", f"Error: {record.error}"])
            if record.settings:
                lines.extend(["", "Settings: " + _format_settings(record.settings)])
            for message in record.warnings:
                lines.extend(["", f"> **Warning:** {message}"])
            lines.append("")
            lines.append("| Quantity | Value |")
            lines.append("|---|---|")
            for key, value in record.summary.items():
                text = (
                    f"{value:.6g}"
                    if isinstance(value, float) and math.isfinite(value)
                    else str(value)
                )
                lines.append(f"| {key} | {text} |")
            lines.append("")
            for child in children.get(index, []):
                lines.extend(child._report_lines(level + 2))
        return lines
