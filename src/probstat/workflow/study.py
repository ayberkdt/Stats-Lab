"""A single study object that runs every UQ analysis on one model.

``UQStudy`` is the integration layer: it owns one
:class:`~probstat.modeling.ProbabilisticModel`, one root random stream, and a
log of every analysis performed. Each stage draws its randomness from a named
child stream, so the whole study is reproducible from one seed, and every
stage records the forward-model evaluations it charged. The study does not
re-implement any method; it calls the domain packages and keeps their results.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

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
from probstat.modeling import ProbabilisticModel
from probstat.modeling.fitting import JointInputFit, fit_joint_input
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
    ForwardModel,
    GridConvergenceResult,
    LeastSquaresConvergence,
    NumericalErrorBudget,
    NumericalErrorDistribution,
    QuantityOfInterest,
    SolutionVerification,
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


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One completed analysis stage of a study.

    Args:
        stage: Stage kind such as ``"propagation"`` or ``"reliability"``.
        label: Human-readable description of the stage.
        summary: Headline numbers of the stage.
        evaluations: Forward-model evaluations charged to the stage.
        seed: Seed of the stream the stage used.
        result: The full domain result object.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    stage: str
    label: str
    summary: Mapping[str, float | str]
    evaluations: int
    seed: int
    result: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class TargetedPropagation:
    """Monte Carlo propagation of one QoI stopped at a relative-error target.

    Every summary comes from one-pass, constant-memory accumulators: no model
    output is stored, so the memory use does not grow with ``sample_size``.

    Args:
        quantity: QoI name.
        controlled: ``"mean"`` or ``"probability"``: the estimate whose
            relative standard error was driven to the target.
        target_relative_error: Requested relative standard error.
        relative_error: Relative standard error reached by the controlled
            estimate (``inf`` when it is undefined, e.g. no failures yet).
        converged: Whether ``relative_error <= target_relative_error``.
        sample_size: Model runs drawn (one evaluation each).
        batches: Number of batches drawn.
        mean: Running mean of the QoI.
        standard_error: Monte Carlo standard error ``s / sqrt(n)`` of the mean.
        quantiles: ``(probability, estimate)`` pairs from the P² algorithm
            (approximate; exact type-7 quantiles for ``n <= 5``).
        threshold: Failure threshold, or ``None`` in mean mode.
        failure: ``"above"`` (failure when ``Y >= t``) or ``"below"``
            (failure when ``Y <= t``).
        probability: Failure-probability estimate (``nan`` without a
            threshold).
        probability_standard_error: Plug-in Bernoulli standard error
            (``nan`` without a threshold).
        probability_interval: 95% Wilson interval (``(nan, nan)`` without a
            threshold).
        events: Number of failures counted.

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962],
        [REF-JAIN-CHLAMTAC-1985], [REF-WILSON-1927].

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
    probability: float
    probability_standard_error: float
    probability_interval: tuple[float, float]
    events: int

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
            probability of ``level`` (``nan`` without a level).
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
    pot_probability: float
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


def _format_condition(condition: Condition) -> str:
    """Render a condition mapping as ``name=value`` pairs.

    References:
        Project reporting policy. No equation is evaluated.

    """
    if not condition:
        return "(model defaults)"
    return ", ".join(f"{name}={value:g}" for name, value in condition.items())


class UQStudy:
    """Run propagation, sensitivity, reliability, and risk on one model.

    Args:
        model: The probabilistic model under study.
        seed: Root seed or stream for every stochastic stage.
        name: Study label (defaults to the model name).

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-SUDRET-2007, Chapter 1].

    """

    def __init__(
        self,
        model: ProbabilisticModel,
        *,
        seed: int | RandomStream = 0,
        name: str | None = None,
    ) -> None:
        """Bind the model and root stream.

        Raises:
            TypeError: If ``model`` is not a ProbabilisticModel.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        if not isinstance(model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
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
        fixed: Mapping[str, float] | None = None,
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
        )
        return study

    @property
    def model(self) -> ProbabilisticModel:
        """Return the model under study.

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
    def records(self) -> tuple[StageRecord, ...]:
        """Return completed stages in execution order.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return tuple(self._records)

    @property
    def total_evaluations(self) -> int:
        """Return the forward-model evaluations charged by all stages.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return sum(record.evaluations for record in self._records)

    def latest(self, stage: str) -> StageRecord:
        """Return the most recent record of a stage kind.

        Args:
            stage: Stage kind.

        Returns:
            Latest matching record.

        Raises:
            KeyError: If the stage has not been run.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        for record in reversed(self._records):
            if record.stage == stage:
                return record
        raise KeyError(f"stage {stage!r} has not been run")

    def stream(self, stage: str) -> RandomStream:
        """Return the child stream for the next run of a stage.

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

    def _record(
        self,
        stage: str,
        label: str,
        summary: Mapping[str, float | str],
        evaluations: int,
        stream: RandomStream,
        result: Any,
    ) -> None:
        """Append one stage record.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        self._records.append(
            StageRecord(
                stage,
                label,
                MappingProxyType(dict(summary)),
                evaluations,
                stream.seed,
                result,
            )
        )

    def propagate(self, size: int = 4096, *, design: str = "sobol") -> PropagationStudy:
        """Propagate the joint input law through every QoI.

        Args:
            size: Number of input realizations.
            design: Sampling design name.

        Returns:
            Propagation study.

        Equation:
            [EQ-UQ-001], [EQ-MC-MEAN-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 3-4].

        """
        stream = self.stream("propagation")
        study = propagate(self._model, size, design=design, seed=stream)
        summary: dict[str, float | str] = {"design": design, "size": size}
        for name, output in study.outputs.items():
            summary[f"{name}.mean"] = output.mean
            summary[f"{name}.std"] = output.standard_deviation
            summary[f"{name}.q05"] = output.quantile(0.05)
            summary[f"{name}.q95"] = output.quantile(0.95)
        self._record(
            "propagation",
            f"{design} propagation, {size} samples",
            summary,
            study.model_evaluations,
            stream,
            study,
        )
        return study

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
    ) -> TargetedPropagation:
        """Propagate plain Monte Carlo batches until a relative error is reached.

        Batches of ``batch`` joint input draws (the model's own
        :meth:`~probstat.modeling.JointInput.sample`, batch ``k`` on the child
        stream ``stream.child("batch", k)``) are evaluated and pushed into a
        :class:`~probstat.statistics.RunningMoments`, one
        :class:`~probstat.statistics.P2Quantile` per level, and, with a
        threshold, an :class:`~probstat.statistics.ExceedanceCounter`. After
        every batch the relative standard error of the controlled estimate is
        checked: ``s / (sqrt(n) |mean|)`` for the mean, or
        ``sqrt((1 - p) / (n p))`` for the failure probability when
        ``threshold`` is given. Sampling stops at the first batch meeting the
        target or at ``max_samples`` (the last batch is truncated), in which
        case the result reports ``converged=False``. No output is stored.

        Checking once per batch limits, but does not remove, the optimism of
        data-dependent stopping; a mean near zero or a failure probability
        with no observed failures makes the target unreachable.

        Args:
            quantity: QoI name.
            target_relative_error: Positive relative standard-error target.
            threshold: Failure threshold; switches control to the failure
                probability.
            failure: ``"above"`` (failure when ``Y >= t``) or ``"below"``
                (failure when ``Y <= t``), as in :meth:`limit_state`.
            batch: Draws between checks, at least two.
            max_samples: Draw budget, at least two.
            quantiles: Quantile levels strictly between zero and one.

        Returns:
            Targeted propagation result; the stage charges ``sample_size``
            evaluations.

        Raises:
            TypeError: If numeric arguments have invalid types.
            ValueError: If ``quantity`` is unknown or an argument is out of
                its domain.

        Equation:
            [EQ-UQ-001], [EQ-STAT-STREAM-007], [EQ-MC-SE-001],
            [EQ-MC-PROB-001], [EQ-INF-CI-WILSON-001], [EQ-STAT-STREAM-005].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962],
            [REF-JAIN-CHLAMTAC-1985], [REF-WILSON-1927].

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
        levels = tuple(require_finite_real(p, name="quantile level") for p in quantiles)
        if any(not 0.0 < p < 1.0 for p in levels):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        limit = (
            None
            if threshold is None
            else require_finite_real(threshold, name="threshold")
        )

        stream = self.stream("sequential_propagation")
        moments = RunningMoments()
        markers = tuple(P2Quantile(p) for p in levels)
        counter = None if limit is None else ExceedanceCounter()
        before = self._model.evaluations
        relative = math.inf
        converged = False
        batches = 0
        while moments.count < budget:
            draws = min(size, budget - moments.count)
            points = self._model.inputs.sample(
                draws, seed=stream.child("batch", batches)
            )
            batches += 1
            for point in points:
                value = self._model.evaluate(point)[quantity]
                moments.push(value)
                for marker in markers:
                    marker.push(value)
                if counter is not None and limit is not None:
                    counter.push(
                        value >= limit if failure == "above" else value <= limit
                    )
            if counter is None:
                relative = moments.relative_standard_error()
            elif counter.events == 0:
                relative = math.inf
            else:
                p = counter.proportion
                relative = math.sqrt((1.0 - p) / (counter.trials * p))
            if relative <= target:
                converged = True
                break

        if counter is not None:
            probability = counter.proportion
            probability_se = counter.standard_error()
            interval = counter.wilson_interval()
            events = counter.events
        else:
            probability = probability_se = math.nan
            interval = (math.nan, math.nan)
            events = 0
        result = TargetedPropagation(
            quantity=quantity,
            controlled="mean" if counter is None else "probability",
            target_relative_error=target,
            relative_error=relative,
            converged=converged,
            sample_size=moments.count,
            batches=batches,
            mean=moments.mean,
            standard_error=moments.standard_error(),
            quantiles=tuple((m.probability, m.estimate) for m in markers),
            threshold=limit,
            failure=failure,
            probability=probability,
            probability_standard_error=probability_se,
            probability_interval=interval,
            events=events,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "controlled": result.controlled,
            "target relative error": target,
            "relative error": relative,
            "converged": "yes" if converged else "no (max_samples reached)",
            "samples": float(result.sample_size),
            "batches": float(batches),
            f"{quantity}.mean": result.mean,
            f"{quantity}.se": result.standard_error,
        }
        for level, estimate in result.quantiles:
            summary[f"{quantity}.q[{level:g}] (P2)"] = estimate
        if limit is not None:
            relation = ">=" if failure == "above" else "<="
            event = f"P[{quantity} {relation} {limit:g}]"
            summary[event] = probability
            summary[f"{event} se"] = probability_se
            summary[f"{event} Wilson 95% lower"] = interval[0]
            summary[f"{event} Wilson 95% upper"] = interval[1]
        self._record(
            "sequential_propagation",
            f"Monte Carlo of {quantity} until relative error {target:g} "
            f"on the {result.controlled}",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result

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
        stream = self.stream("reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        comparison = compare_reliability_methods(
            state,
            methods=methods,
            monte_carlo_samples=monte_carlo_samples,
            seed=stream,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
        }
        for row in comparison.rows:
            summary[f"pf[{row.method}]"] = row.probability
        self._record(
            "reliability",
            f"P({quantity} {'<=' if failure == 'below' else '>='} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            comparison,
        )
        return comparison

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
        self._record(
            "form",
            f"FORM for {quantity} threshold {threshold:g}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

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
        this study records the training cost and accuracy.

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
        return UQStudy(surrogate, seed=stream.child("surrogate-study"))

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

    def _output_sample(
        self,
        stream: RandomStream,
        quantity: str,
        size: int | None,
        design: str,
        default_size: int,
    ) -> tuple[PropagationStudy, int, str]:
        """Return the latest propagation sample, or a fresh one on ``stream``.

        With ``size=None`` the latest propagation stage is reused at no cost;
        otherwise (or when none exists) ``size`` (default ``default_size``)
        new runs are drawn from ``stream`` and charged to the caller.

        Raises:
            ValueError: If ``quantity`` is not a model QoI.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if quantity not in self._model.quantity_names:
            raise ValueError(f"{quantity!r} is not a quantity of the study model")
        if size is None:
            for record in reversed(self._records):
                if record.stage == "propagation":
                    return (
                        record.result,
                        0,
                        f"propagation stage (stream seed {record.seed})",
                    )
        sample = propagate(
            self._model,
            default_size if size is None else size,
            design=design,
            seed=stream,
        )
        return sample, sample.model_evaluations, f"fresh {design} sample"

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
    ) -> TailAnalysis:
        """Fit a peaks-over-threshold model to the tail of one QoI.

        The sample is the latest propagation (no new runs) unless ``size`` is
        given or no propagation exists, in which case ``size`` (default 4096)
        fresh runs are drawn from this stage's stream. Each model run is one
        observation (``observations_per_year = 1``), so a return period ``T``
        is a number of runs and the ``T``-run level has tail probability
        ``1/T``. The lower tail is analyzed as the upper tail of ``-Y``.

        The rate model assumes independent runs; for LHS or QMC samples the
        binomial intervals are conservative references, not exact.

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
            size: Fresh sample size; ``None`` reuses the latest propagation.
            design: Sampling design of a fresh sample.

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
            stream, quantity, size, design, 4096
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
        pot_probability = math.nan
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
                else math.nan
            )
        if level is not None and empirical is not None:
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

    def information_sensitivity(
        self,
        quantity: str,
        *,
        max_points: int = 512,
        k: int = 3,
        unit: InformationUnit = "nats",
        size: int | None = None,
        design: str = "mc",
    ) -> InformationSensitivity:
        """Estimate the mutual information between each input and one QoI.

        ``I(X_i; Y)`` is zero exactly when ``Y`` is independent of ``X_i``
        and is invariant under monotone transformations of either, so it
        sees non-monotone and higher-moment dependence that variance-based
        indices can miss, and it needs no independence between inputs. It is
        a moment-independent complement to Sobol indices, not a variance
        share: values are in nats (or bits) and do not sum to anything.

        The sample is the first ``max_points`` runs of the latest propagation
        unless ``size`` is given or none exists (then ``size``, default
        ``max_points``, fresh runs are drawn from this stage's stream). The
        KSG estimator costs ``O(n^2)`` per input, hence the cap.

        Args:
            quantity: QoI name.
            max_points: Maximum number of runs entering the estimator.
            k: KSG neighbour order.
            unit: ``"nats"`` or ``"bits"``.
            size: Fresh sample size; ``None`` reuses the latest propagation.
            design: Sampling design of a fresh sample.

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
            stream, quantity, size, design, max_points
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

    def report(self) -> str:
        """Return a Markdown report of every stage with costs and seeds.

        Returns:
            Markdown text: one section per stage with a two-column table.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        dependence = (
            "independent" if self._model.inputs.is_independent else "Gaussian copula"
        )
        lines = [
            f"# UQ study: {self._name}",
            "",
            f"- Inputs: {', '.join(self._model.inputs.names)}",
            f"- Quantities of interest: {', '.join(self._model.quantity_names)}",
            f"- Dependence: {dependence}",
            f"- Root seed: {self._root.seed}",
            f"- Forward-model evaluations: {self.total_evaluations:,}",
            "",
        ]
        for index, record in enumerate(self._records, start=1):
            lines.append(f"## {index}. {record.stage}: {record.label}")
            lines.append("")
            lines.append(
                f"Evaluations: {record.evaluations:,} · stream seed: {record.seed}"
            )
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
        return "\n".join(lines)
