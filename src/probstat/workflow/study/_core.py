"""The composed :class:`UQStudy` class.

:class:`UQStudy` inherits its stages from one mixin per domain
(propagation, sensitivity, reliability, calibration, validation, design,
reporting) and adds the constructor, the data-driven constructor, the
read-only properties, record lookup, and input-law replacement.

References:
    [REF-SMITH-UQ-2024, Chapters 1-5], [REF-SUDRET-2007, Chapter 1].

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from datetime import UTC, datetime
from typing import Any, Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.modeling.fitting import JointInputFit, fit_joint_input
from probstat.uq import EvaluationCache, ForwardModel, QuantityOfInterest

from ..chaining import input_law_hash
from ._calibration import _CalibrationStages
from ._context import _setting_value
from ._design import _DesignStages
from ._propagation import _PropagationStages
from ._records import StageRecord
from ._reliability import _ReliabilityStages
from ._reporting import _ReportingStages
from ._sensitivity import _SensitivityStages
from ._validation import _ValidationStages


class UQStudy(
    _PropagationStages,
    _SensitivityStages,
    _ReliabilityStages,
    _CalibrationStages,
    _ValidationStages,
    _DesignStages,
    _ReportingStages,
):
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
        self._records = []
        self._problem = None
        self._posterior = None
        self._discrepancy = None
        # Which calibration predict() and design() use: the latest one run.
        self._active = None
        # Latest solution verification and its u_num, per QoI.
        self._verifications = {}
        self._budget = evaluation_budget
        self._cache = store
        self._executor = executor
        self._kind = evaluation_kind
        self._contexts = []
        self._children = []
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

    def _surrogate_study(
        self, model: ProbabilisticModel, *, seed: RandomStream, name: str
    ) -> UQStudy:
        """Return a new surrogate study on ``model`` (used by :meth:`emulate`).

        References:
            [REF-SUDRET-2007, Chapter 4]. No equation is evaluated.

        """
        return UQStudy(model, seed=seed, name=name, evaluation_kind="surrogate")
