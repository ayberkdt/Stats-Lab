"""Result records of the :class:`~probstat.workflow.study.UQStudy` stages.

Every stage appends one :class:`StageRecord` with its cost and provenance.
Stages whose result is not a single domain object return one of the frozen
records defined here.

References:
    [REF-SMITH-UQ-2024, Chapter 1].

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from probstat.design import Condition, GreedyDesignResult, InformationGainEstimate
from probstat.inverse import (
    DiscrepancyCalibration,
    DiscrepancyIdentifiability,
    DiscrepancyPosterior,
    DiscrepancyPrediction,
)
from probstat.monte_carlo import ProbabilityEstimate
from probstat.reliability import (
    MixtureImportanceResult,
    MultipleDesignPointsResult,
    MultiPointFORMResult,
)
from probstat.risk import PeaksOverThresholdFit, ReturnLevel
from probstat.uq import FrozenMapping, json_safe

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
