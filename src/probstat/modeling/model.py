"""The probabilistic model: one object shared by every UQ workflow.

A :class:`ProbabilisticModel` binds a joint input law, a named forward model
``Y = g(X)``, and the quantities of interest. Propagation, sensitivity,
reliability, calibration, surrogate fitting, and risk analysis all accept this
object, so a study defines its model exactly once.

The model is also the evaluation engine of the library:

- **Counting.** :class:`EvaluationCounter` counts *completed* simulator runs;
  failed attempts and cache hits are counted separately.
- **Failure policy.** ``on_failure`` is ``"raise"`` (default, the original
  exception propagates), ``"skip"``, or ``"record"`` (see
  :mod:`probstat.uq.evaluation`). Batch evaluation drops failed runs and
  reports them; a single :meth:`ProbabilisticModel.evaluate` call cannot drop
  its run and raises :class:`~probstat.uq.ModelEvaluationError` instead.
- **Batch, parallel, cache.** :meth:`ProbabilisticModel.evaluate_many`
  accepts an optional :class:`concurrent.futures.Executor` (results keep
  input order, so outputs do not depend on scheduling), uses a model's
  ``evaluate_batch(list_of_mappings) -> list_of_mappings`` hook when it has
  one, and consults an optional :class:`~probstat.uq.EvaluationCache`.
- **Budget.** A counter ``limit`` stops evaluation with
  :class:`~probstat.uq.EvaluationBudgetExceeded` before the simulator is
  called.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import repeat
from types import MappingProxyType
from typing import Any

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector
from probstat.uq import (
    EvaluationBudgetExceeded,
    EvaluationCache,
    FailedEvaluation,
    FailureLog,
    FailurePolicy,
    ForwardModel,
    FrozenMapping,
    ModelEvaluationError,
    QuantityOfInterest,
    select_outputs,
)
from probstat.uq.evaluation import describe_error, require_failure_policy

from .inputs import JointInput

_Outcome = tuple[str, Any]


@dataclass
class EvaluationCounter:
    """Mutable count of forward-model runs, failures, and cache hits.

    ``count`` holds *completed* evaluations only; failed attempts go to
    ``failed`` and cache hits to ``cache_hits``. A counter may have a
    ``parent``: every increment is also applied to the parent chain, so a
    model derived with :func:`dataclasses.replace` has its own count starting
    at zero while the source model still accounts for every run of the same
    simulator. ``limit`` caps ``count + failed`` of this counter; the check
    walks the parent chain, so a budget set on a source model also binds
    models derived from it.

    Args:
        count: Completed evaluations recorded so far.
        failed: Failed evaluation attempts recorded so far.
        cache_hits: Requests answered from a cache (not evaluations).
        limit: Optional cap on ``count + failed``.
        parent: Optional counter that also receives every increment.

    References:
        [REF-SMITH-UQ-2024, Chapter 3].

    """

    count: int = 0
    failed: int = 0
    cache_hits: int = 0
    limit: int | None = None
    parent: EvaluationCounter | None = field(default=None, repr=False, compare=False)
    _claimed: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def spent(self) -> int:
        """Return completed plus failed evaluations (the budgeted cost).

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.count + self.failed

    def remaining(self) -> int | None:
        """Return the evaluations left under the tightest limit, or ``None``.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        left: int | None = None
        node: EvaluationCounter | None = self
        while node is not None:
            if node.limit is not None:
                room = max(0, node.limit - node.spent)
                left = room if left is None else min(left, room)
            node = node.parent
        return left

    def check(self, amount: int = 1) -> None:
        """Raise before running ``amount`` evaluations that break a limit.

        Args:
            amount: Number of evaluations about to run.

        Raises:
            EvaluationBudgetExceeded: If this counter or an ancestor would
                exceed its limit.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        node: EvaluationCounter | None = self
        while node is not None:
            if node.limit is not None and node.spent + amount > node.limit:
                raise EvaluationBudgetExceeded(
                    f"evaluation budget of {node.limit} runs would be exceeded: "
                    f"{node.spent} spent, {amount} more requested"
                )
            node = node.parent

    def add(self, amount: int = 1) -> None:
        """Record completed evaluations here and in every parent counter.

        Args:
            amount: Non-negative number of new evaluations.

        Raises:
            ValueError: If ``amount`` is negative.
            EvaluationBudgetExceeded: If a limit would be exceeded (nothing
                is recorded then).

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.check(amount)
        node: EvaluationCounter | None = self
        while node is not None:
            node.count += amount
            node = node.parent

    def add_failed(self, amount: int = 1) -> None:
        """Record failed evaluation attempts here and in every parent.

        Args:
            amount: Non-negative number of failed attempts.

        Raises:
            ValueError: If ``amount`` is negative.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if amount < 0:
            raise ValueError("amount must be non-negative")
        node: EvaluationCounter | None = self
        while node is not None:
            node.failed += amount
            node = node.parent

    def add_cache_hits(self, amount: int = 1) -> None:
        """Record requests served from a cache here and in every parent.

        Args:
            amount: Non-negative number of cache hits.

        Raises:
            ValueError: If ``amount`` is negative.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if amount < 0:
            raise ValueError("amount must be non-negative")
        node: EvaluationCounter | None = self
        while node is not None:
            node.cache_hits += amount
            node = node.parent

    @contextmanager
    def limited(self, remaining: int | None) -> Iterator[EvaluationCounter]:
        """Temporarily allow at most ``remaining`` further evaluations.

        Args:
            remaining: Further runs allowed (``None`` leaves the limit as is).

        Yields:
            This counter.

        Raises:
            ValueError: If ``remaining`` is negative.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        previous = self.limit
        if remaining is not None:
            if remaining < 0:
                raise ValueError("remaining must be non-negative")
            proposed = self.spent + remaining
            self.limit = proposed if previous is None else min(previous, proposed)
        try:
            yield self
        finally:
            self.limit = previous


class _EvaluationSettings:
    """Per-model evaluation settings that are not part of the model value.

    Holds the cache and executor installed by
    :meth:`ProbabilisticModel.evaluation_context`. Pickling yields empty
    settings, because executors cannot cross process boundaries.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """

    __slots__ = ("cache", "executor")

    def __init__(self) -> None:
        """Start without cache or executor.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        self.cache: EvaluationCache | None = None
        self.executor: Executor | None = None

    def __reduce__(self) -> tuple[Any, tuple[()]]:
        """Pickle as empty settings.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return (_EvaluationSettings, ())


def _run_job(
    model: Any, parameters: Sequence[Mapping[str, object]], vectorised: bool
) -> list[_Outcome]:
    """Run forward-model calls for one job and tag each outcome.

    The function is module-level so process pools can pickle it. Exceptions
    of the simulator are returned, not raised, so one failed run cannot
    abort the other runs of a parallel batch.

    Args:
        model: Forward model (with ``evaluate_batch`` when ``vectorised``).
        parameters: Complete parameter mappings, one per run.
        vectorised: Call ``model.evaluate_batch`` once for the whole job.

    Returns:
        ``("ok", raw_output)``, ``("error", exception)``, or one
        ``("contract", exception)`` per run when ``evaluate_batch`` breaks
        its contract.

    References:
        [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapter 3].

    """
    if vectorised:
        frozen = [MappingProxyType(dict(item)) for item in parameters]
        try:
            raw = model.evaluate_batch(frozen)
        except Exception as error:
            return [("error", error) for _ in parameters]
        if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
            problem: Exception = TypeError(
                "evaluate_batch must return a sequence of outputs"
            )
            return [("contract", problem) for _ in parameters]
        if len(raw) != len(parameters):
            problem = ValueError(
                f"evaluate_batch returned {len(raw)} outputs for "
                f"{len(parameters)} inputs"
            )
            return [("contract", problem) for _ in parameters]
        return [
            ("error", item) if isinstance(item, BaseException) else ("ok", _plain(item))
            for item in raw
        ]
    outcomes: list[_Outcome] = []
    for item in parameters:
        try:
            outcomes.append(("ok", _plain(model(MappingProxyType(dict(item))))))
        except Exception as error:
            outcomes.append(("error", error))
    return outcomes


def _plain(output: object) -> object:
    """Return a plain dictionary copy of a mapping output (else the output).

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return dict(output) if isinstance(output, Mapping) else output


class EvaluationBatch(FrozenMapping[str, tuple[float, ...]]):
    """Outputs of :meth:`ProbabilisticModel.evaluate_many` with bookkeeping.

    The mapping holds, per QoI, the outputs of the *kept* runs in input
    order; ``indices`` gives their positions in the requested points. Under
    the ``"skip"`` and ``"record"`` policies failed runs are absent and
    listed in ``failures``.

    Args:
        outputs: QoI-name mapping to kept outputs.
        indices: Input positions of the kept outputs.
        failures: Failure records ordered by input position.
        size: Number of requested points.
        model_evaluations: Completed simulator runs (cache hits excluded).
        failed_evaluations: Failed simulator attempts.
        cache_hits: Points answered from the cache or by an identical point
            of the same batch.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """

    __slots__ = (
        "indices",
        "failures",
        "size",
        "model_evaluations",
        "failed_evaluations",
        "cache_hits",
    )
    indices: tuple[int, ...]
    failures: tuple[FailedEvaluation, ...]
    size: int
    model_evaluations: int
    failed_evaluations: int
    cache_hits: int

    def __init__(
        self,
        outputs: Mapping[str, tuple[float, ...]],
        indices: tuple[int, ...],
        failures: tuple[FailedEvaluation, ...],
        size: int,
        model_evaluations: int,
        failed_evaluations: int,
        cache_hits: int,
    ) -> None:
        """Store outputs and counts.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        super().__init__(outputs)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "model_evaluations", model_evaluations)
        object.__setattr__(self, "failed_evaluations", failed_evaluations)
        object.__setattr__(self, "cache_hits", cache_hits)

    @property
    def failure_fraction(self) -> float:
        """Return the fraction of requested points without an output.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return len(self.failures) / self.size if self.size else 0.0

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle through the constructor.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return (
            type(self),
            (
                dict(self.items()),
                self.indices,
                self.failures,
                self.size,
                self.model_evaluations,
                self.failed_evaluations,
                self.cache_hits,
            ),
        )


@dataclass(frozen=True)
class ProbabilisticModel:
    """Joint input law, forward model, and quantities of interest.

    ``dataclasses.replace(model, ...)`` gives the new model its own counter
    (starting at zero) linked to the source counter, so the source still
    accounts for every run of the shared simulator.

    Args:
        inputs: Joint law of the named model inputs.
        model: Deterministic named-input/named-output forward map. It may
            also define ``evaluate_batch(list_of_mappings) ->
            list_of_mappings`` for vectorised evaluation.
        quantities: Non-empty distinctly named quantities of interest.
        name: Short study label.
        fixed: Optional deterministic parameters passed unchanged to every
            call. Floats must be finite; other values (strings, integers,
            Booleans, tuples of solver options) are passed as given.
        counter: Evaluation counter; a counter already owned by another
            model becomes the parent of a fresh one.
        on_failure: Failure policy ``"raise"``, ``"skip"``, or ``"record"``.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-4], [REF-SUDRET-2007, Chapter 1].

    """

    inputs: JointInput
    model: ForwardModel
    quantities: tuple[QuantityOfInterest, ...]
    name: str = "model"
    fixed: Mapping[str, Any] = field(default_factory=dict)
    counter: EvaluationCounter = field(
        default_factory=EvaluationCounter, compare=False, repr=False
    )
    on_failure: FailurePolicy = field(default="raise", kw_only=True, compare=False)
    failures: FailureLog = field(init=False, compare=False, repr=False)
    _settings: _EvaluationSettings = field(init=False, compare=False, repr=False)
    _fixed_items: dict[str, Any] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the model contract once at construction.

        Raises:
            TypeError: If components have invalid types.
            ValueError: If quantities are empty or repeated, a fixed float is
                not finite, a fixed parameter shadows a random input, or the
                failure policy is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if not isinstance(self.inputs, JointInput):
            raise TypeError("inputs must be a JointInput")
        if not callable(self.model):
            raise TypeError("model must be callable")
        quantities = tuple(self.quantities)
        if not quantities:
            raise ValueError("at least one quantity of interest is required")
        for index, quantity in enumerate(quantities):
            if not isinstance(quantity, QuantityOfInterest):
                raise TypeError(f"quantities[{index}] must be a QuantityOfInterest")
        names = [quantity.name for quantity in quantities]
        if len(set(names)) != len(names):
            raise ValueError("quantity names must be distinct")
        if not isinstance(self.fixed, Mapping):
            raise TypeError("fixed must be a mapping")
        fixed: dict[str, Any] = {}
        for key, value in self.fixed.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError("fixed parameter names must be non-empty strings")
            if isinstance(value, float):
                value = require_finite_real(value, name=f"fixed[{key!r}]")
            fixed[key] = value
        overlap = set(fixed) & set(self.inputs.names)
        if overlap:
            raise ValueError(
                f"fixed parameters shadow random inputs: {sorted(overlap)}"
            )
        policy = require_failure_policy(self.on_failure)
        counter = self.counter
        if not isinstance(counter, EvaluationCounter):
            raise TypeError("counter must be an EvaluationCounter")
        if counter._claimed:
            counter = EvaluationCounter(parent=counter)
        counter._claimed = True
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "fixed", FrozenMapping(fixed))
        object.__setattr__(self, "_fixed_items", fixed)
        object.__setattr__(self, "on_failure", policy)
        object.__setattr__(self, "counter", counter)
        object.__setattr__(self, "failures", FailureLog())
        object.__setattr__(self, "_settings", _EvaluationSettings())

    @property
    def dimension(self) -> int:
        """Return the number of random inputs.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.inputs.dimension

    @property
    def quantity_names(self) -> tuple[str, ...]:
        """Return QoI names in declaration order.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return tuple(quantity.name for quantity in self.quantities)

    @property
    def evaluations(self) -> int:
        """Return completed forward-model calls made through this object.

        Failed attempts and cache hits are not included; see
        :attr:`failed_evaluations` and :attr:`cache_hits`.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.counter.count

    @property
    def failed_evaluations(self) -> int:
        """Return failed forward-model attempts made through this object.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.counter.failed

    @property
    def cache_hits(self) -> int:
        """Return requests answered from a cache instead of the simulator.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.counter.cache_hits

    def quantity(self, name: str) -> QuantityOfInterest:
        """Return the declared quantity of interest with a given name.

        Args:
            name: QoI name.

        Returns:
            Matching quantity.

        Raises:
            KeyError: If no quantity has that name.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        for quantity in self.quantities:
            if quantity.name == name:
                return quantity
        raise KeyError(f"unknown quantity of interest {name!r}")

    @contextmanager
    def evaluation_context(
        self,
        *,
        cache: EvaluationCache | None = None,
        executor: Executor | None = None,
    ) -> Iterator[ProbabilisticModel]:
        """Install a cache and/or executor for every evaluation in a block.

        Code that only receives the model (sensitivity, reliability, and
        surrogate fitting call :meth:`evaluate` or :meth:`evaluate_many`)
        then reuses cached runs; :meth:`evaluate_many` also runs in parallel.
        Settings are restored on exit; ``None`` keeps the current setting.

        Args:
            cache: Cache consulted by :meth:`evaluate` and
                :meth:`evaluate_many`.
            executor: Executor used by :meth:`evaluate_many`.

        Yields:
            This model.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        previous = (self._settings.cache, self._settings.executor)
        if cache is not None:
            self._settings.cache = cache
        if executor is not None:
            self._settings.executor = executor
        try:
            yield self
        finally:
            self._settings.cache, self._settings.executor = previous

    def _parameters(self, x: Sequence[float] | Mapping[str, float]) -> dict[str, Any]:
        """Return the complete, validated parameter mapping of one call.

        A mapping must name every random input exactly once and nothing else:
        missing names, unknown names, and fixed-parameter names are rejected.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If names do not match or a value is not finite.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if type(x) is tuple:
            values: dict[str, Any] = dict(self.inputs.as_mapping(x))
            if self._fixed_items:
                values.update(self._fixed_items)
            return values
        names = self.inputs.names
        if isinstance(x, Mapping):
            given = set(x)
            if given != set(names):
                fixed_hits = sorted(str(k) for k in given if k in self.fixed)
                unknown = sorted(
                    str(k) for k in given if k not in names and k not in self.fixed
                )
                missing = [name for name in names if name not in given]
                problems = []
                if fixed_hits:
                    problems.append(
                        f"{fixed_hits} are fixed parameters and cannot be set per "
                        "evaluation (use dataclasses.replace(model, fixed=...))"
                    )
                if unknown:
                    problems.append(f"unknown input names {unknown}")
                if missing:
                    problems.append(f"missing inputs {missing}")
                raise ValueError(
                    f"evaluate() needs exactly the inputs {list(names)}: "
                    + "; ".join(problems)
                )
            values = {name: require_finite_real(x[name], name=name) for name in names}
        else:
            values = dict(self.inputs.as_mapping(x))
        if self._fixed_items:
            values.update(self._fixed_items)
        return values

    def _log_failure(
        self,
        parameters: Mapping[str, Any],
        error: BaseException,
        index: int | None,
        policy: FailurePolicy,
    ) -> FailedEvaluation:
        """Count one failed attempt and add it to the failure log.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        error_type, message = describe_error(error)
        failure = FailedEvaluation(
            index,
            FrozenMapping({} if policy == "skip" else parameters),
            error_type,
            message,
        )
        self.counter.add_failed(1)
        self.failures.add(failure)
        return failure

    def evaluate(
        self,
        x: Sequence[float] | Mapping[str, float],
        *,
        cache: EvaluationCache | None = None,
    ) -> Mapping[str, float]:
        """Evaluate every QoI at one physical input realization.

        A mapping argument must name exactly the random inputs (missing,
        unknown, and fixed-parameter names raise ``ValueError``). On a
        simulator failure the attempt is counted in
        :attr:`failed_evaluations`; under ``"raise"`` the original exception
        propagates, otherwise :class:`~probstat.uq.ModelEvaluationError` is
        raised (one call cannot drop its own run).

        Args:
            x: Physical input vector in model order or a name mapping.
            cache: Optional cache (defaults to the one installed by
                :meth:`evaluation_context`).

        Returns:
            Read-only QoI-name mapping.

        Raises:
            ModelEvaluationError: If the run fails under a dropping policy.
            EvaluationBudgetExceeded: If the counter limit is reached.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 1-3].

        """
        parameters = self._parameters(x)
        store = cache if cache is not None else self._settings.cache
        key: Hashable | None = None
        if store is not None:
            key = store.key(parameters)
            hit = store.get(key, self.quantity_names)
            if hit is not None:
                self.counter.add_cache_hits(1)
                return hit
        self.counter.check(1)
        try:
            outputs = self.model(MappingProxyType(parameters))
            selected = select_outputs(outputs, self.quantities)
        except Exception as error:
            failure = self._log_failure(parameters, error, None, self.on_failure)
            if self.on_failure == "raise":
                raise
            raise ModelEvaluationError(failure) from error
        self.counter.add(1)
        if store is not None:
            store.put(key, selected)
        return selected

    def evaluate_many(
        self,
        points: Iterable[Sequence[float] | Mapping[str, float]],
        *,
        executor: Executor | None = None,
        batch: int | None = None,
        cache: EvaluationCache | None = None,
        on_failure: FailurePolicy | None = None,
    ) -> EvaluationBatch:
        """Evaluate every QoI at many physical input points.

        The whole cost is checked against the counter limit before any run.
        Without an executor, runs happen in input order and the ``"raise"``
        policy stops at the first failure. With an executor, jobs are
        submitted through ``executor.map`` and collected in input order, so
        outputs and counts are identical to sequential evaluation; under
        ``"raise"`` every submitted run is still counted before the first
        failure is re-raised. Process pools need a picklable (module-level)
        forward model.

        If the forward model defines ``evaluate_batch``, it receives lists of
        read-only parameter mappings and must return one output mapping (or
        an exception instance for a failed member) per input; an exception
        raised by ``evaluate_batch`` fails every member of that call.

        With a cache, points already stored and repeated points within the
        batch are not run again; they are counted as cache hits, not as model
        evaluations.

        Args:
            points: Physical input vectors or name mappings.
            executor: Optional :class:`concurrent.futures.Executor` (defaults
                to the one installed by :meth:`evaluation_context`).
            batch: Points per ``evaluate_batch`` call (``None``: all pending
                points in one call); without the hook, points per executor
                job (``None``: one).
            cache: Optional exact-match cache (defaults to the installed one).
            on_failure: Policy for this call (defaults to the model's).

        Returns:
            Read-only mapping from QoI name to kept outputs, with indices,
            failure records, and counts.

        Raises:
            ValueError: If ``batch`` is not a positive integer, a point is
                invalid, or ``evaluate_batch`` returns the wrong number of
                outputs.
            EvaluationBudgetExceeded: If the runs would exceed the limit.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        policy = require_failure_policy(
            self.on_failure if on_failure is None else on_failure
        )
        if batch is not None and (
            isinstance(batch, bool) or not isinstance(batch, int) or batch < 1
        ):
            raise ValueError("batch must be a positive integer or None")
        pool = executor if executor is not None else self._settings.executor
        store = cache if cache is not None else self._settings.cache
        parameters = [self._parameters(point) for point in points]
        size = len(parameters)
        names = self.quantity_names
        results: list[Mapping[str, float] | None] = [None] * size
        keys: list[Hashable | None] = [None] * size
        pending: list[int] = []
        repeats: list[tuple[int, int]] = []
        hits = 0
        if store is not None:
            first: dict[Hashable, int] = {}
            for index, item in enumerate(parameters):
                key = store.key(item)
                keys[index] = key
                hit = store.get(key, names)
                if hit is not None:
                    results[index] = hit
                    hits += 1
                elif key is not None and key in first:
                    repeats.append((index, first[key]))
                else:
                    if key is not None:
                        first[key] = index
                    pending.append(index)
        else:
            pending = list(range(size))
        self.counter.check(len(pending))

        vectorised = callable(getattr(self.model, "evaluate_batch", None))
        width = (batch or len(pending) or 1) if vectorised else (batch or 1)
        jobs = [
            pending[start : start + width] for start in range(0, len(pending), width)
        ]
        forward = self.model

        def direct(index: int) -> list[_Outcome]:
            """Run one point in this thread without defensive copies.

            References:
                [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapter 3].

            """
            try:
                return [("ok", forward(MappingProxyType(parameters[index])))]
            except Exception as error:
                return [("error", error)]

        if pool is None and not vectorised and width == 1:
            outcomes: Iterable[list[_Outcome]] = (direct(job[0]) for job in jobs)
        elif pool is None:
            outcomes = (
                _run_job(self.model, [parameters[i] for i in job], vectorised)
                for job in jobs
            )
        else:
            outcomes = pool.map(
                _run_job,
                repeat(self.model),
                [[parameters[i] for i in job] for job in jobs],
                repeat(vectorised),
            )

        failures: dict[int, FailedEvaluation] = {}
        errors: dict[int, BaseException] = {}
        completed = 0
        first_error: BaseException | None = None
        for job, job_outcomes in zip(jobs, outcomes, strict=True):
            for index, (tag, payload) in zip(job, job_outcomes, strict=True):
                if tag == "contract":
                    raise payload
                if tag == "ok":
                    try:
                        selected = select_outputs(payload, self.quantities)
                    except Exception as error:
                        tag, payload = "error", error
                if tag == "ok":
                    self.counter.add(1)
                    completed += 1
                    results[index] = selected
                    if store is not None:
                        store.put(keys[index], selected)
                    continue
                failures[index] = self._log_failure(
                    parameters[index], payload, index, policy
                )
                errors[index] = payload
                if policy == "raise" and first_error is None:
                    first_error = payload
                    if pool is None:
                        raise payload
        if first_error is not None:
            raise first_error
        for index, source in repeats:
            if results[source] is not None:
                results[index] = results[source]
                hits += 1
                if store is not None:
                    store.hits += 1
                    store.misses -= 1
            else:
                origin = failures[source]
                failures[index] = FailedEvaluation(
                    index, origin.inputs, origin.error_type, origin.message
                )
        if hits:
            self.counter.add_cache_hits(hits)
        kept_pairs = [
            (index, item) for index, item in enumerate(results) if item is not None
        ]
        kept = tuple(index for index, _ in kept_pairs)
        collected = {
            name: tuple(item[name] for _, item in kept_pairs) for name in names
        }
        return EvaluationBatch(
            collected,
            kept,
            tuple(failures[i] for i in sorted(failures)),
            size,
            completed,
            len(errors),
            hits,
        )

    def scalar_function(self, quantity: str) -> Callable[[Sequence[float]], float]:
        """Return ``x -> g_q(x)`` on physical input vectors.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable that counts evaluations through this model.

        Raises:
            KeyError: If the quantity is unknown.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        self.quantity(quantity)

        def function(x: Sequence[float]) -> float:
            """Evaluate one QoI at a physical input vector.

            References:
                [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapter 3].

            """
            return self.evaluate(as_vector(x, name="x"))[quantity]

        return function

    def standard_space_function(
        self, quantity: str
    ) -> Callable[[Sequence[float]], float]:
        """Return ``u -> g_q(T^-1(u))`` on independent standard normals.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable used by FORM, SORM, and subset simulation.

        Equation:
            [EQ-UQ-001], [EQ-MOD-NATAF-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        physical = self.scalar_function(quantity)

        def function(u: Sequence[float]) -> float:
            """Evaluate one QoI at a standard-normal point.

            References:
                [EQ-MOD-NATAF-001], [REF-MELCHERS-BECK-2018, Chapter 5].

            """
            return physical(self.inputs.from_standard_normal(u))

        return function

    def unit_function(self, quantity: str) -> Callable[[Sequence[float]], float]:
        """Return ``p -> g_q(F^-1(p))`` on the unit hypercube.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable on ``(0, 1)^d`` used by pick-freeze Sobol designs.

        Equation:
            [EQ-UQ-001], [EQ-SAMP-INV-001].

        References:
            [REF-SALTELLI-2010].

        """
        physical = self.scalar_function(quantity)

        def function(point: Sequence[float]) -> float:
            """Evaluate one QoI at a unit-hypercube point.

            References:
                [EQ-SAMP-INV-001], [REF-SALTELLI-2010].

            """
            return physical(self.inputs.from_unit(point))

        return function

    def nominal(self) -> Vector:
        """Return the vector of marginal means.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.inputs.means()

