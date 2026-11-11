"""Typed base class and stage decorator shared by every study mixin.

:class:`_StudyBase` declares the instance state that
:class:`~probstat.workflow.study.UQStudy` initializes and implements the
accounting every stage relies on (streams, budget, running context, and
records). :func:`_stage` runs a stage method under that accounting. Each
stage mixin derives from :class:`_StudyBase`, so it type-checks on its own.

References:
    [REF-SMITH-UQ-2024, Chapter 1], [REF-STEELE-LEA-FLOOD-2014].

"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Any, Concatenate, Literal, ParamSpec, Protocol, Self, TypeVar

from probstat.foundations.random_streams import RandomStream
from probstat.inverse import CalibrationPosterior, CalibrationProblem
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.propagation import PropagationStudy, propagate
from probstat.uq import (
    EvaluationBudgetExceeded,
    EvaluationCache,
    FrozenMapping,
    SolutionVerification,
)

from ..chaining import input_law_hash
from ._context import _setting_value, _StageContext
from ._records import DiscrepancyAnalysis, StageRecord

_P = ParamSpec("_P")
_R = TypeVar("_R")
_S = TypeVar("_S", bound="_StudyBase")


class _StageDecorator(Protocol):
    """Type of the decorator returned by :func:`_stage`.

    It keeps the exact signature of the decorated stage method, including
    the mixin type of ``self``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """

    def __call__(
        self, method: Callable[Concatenate[_S, _P], _R], /
    ) -> Callable[Concatenate[_S, _P], _R]:
        """Return the stage-accounting wrapper of ``method``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        ...


def _stage(
    kind: str,
    cost: Callable[[_StudyBase, Mapping[str, Any]], int | None] | None = None,
) -> _StageDecorator:
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
        method: Callable[Concatenate[_S, _P], _R],
    ) -> Callable[Concatenate[_S, _P], _R]:
        """Wrap one stage method.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self: _S, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
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


class _StudyBase:
    """State and stage accounting shared by every stage mixin of a study.

    The class declares the instance attributes that
    :class:`~probstat.workflow.study.UQStudy` initializes and implements the
    machinery every stage relies on: per-stage random streams, the
    evaluation budget, the running-stage context, and record keeping. Each
    stage mixin derives from it, so every mixin type-checks on its own and
    :class:`~probstat.workflow.study.UQStudy` composes them.

    References:
        [REF-SMITH-UQ-2024, Chapter 1], [REF-STEELE-LEA-FLOOD-2014].

    """

    _model: ProbabilisticModel
    _root: RandomStream
    _name: str
    _records: list[StageRecord]
    _problem: CalibrationProblem | None
    _posterior: CalibrationPosterior | None
    _discrepancy: DiscrepancyAnalysis | None
    # Which calibration predict() and design() use: the latest one run.
    _active: Literal["plain", "discrepancy"] | None
    # Latest solution verification and its u_num, per QoI.
    _verifications: dict[str, tuple[SolutionVerification, float]]
    _budget: int | None
    _cache: EvaluationCache | None
    _executor: Executor | None
    _kind: Literal["forward-model", "surrogate"]
    _contexts: list[_StageContext]
    # Surrogate studies created by emulate(), keyed by emulation stage number.
    _children: list[tuple[int, Self]]
    _law_label: str
    _law_hash: str
    _law_changes: int
    _created: str

    @property
    def model(self) -> ProbabilisticModel:
        """Return the model under study (with the current input law).

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return self._model

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

    def _surrogate_study(
        self, model: ProbabilisticModel, *, seed: RandomStream, name: str
    ) -> Self:
        """Return a new study of the composed class on a surrogate model.

        :meth:`~probstat.workflow.study.UQStudy.emulate` builds its child
        study through this hook, so the stage mixins need not import the
        composed class.

        Raises:
            NotImplementedError: Always; the composed study class overrides
                this hook.

        References:
            [REF-SUDRET-2007, Chapter 4]. No equation is evaluated.

        """
        raise NotImplementedError("the composed study class provides this hook")
