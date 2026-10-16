"""Simulator-run bookkeeping: failure policy, failure log, cache, and budget.

A forward model is an external computation that can fail: it may raise, return
something that is not a mapping, omit a quantity of interest, or return a NaN
or an infinity. This module holds the vocabulary that the central model
(:class:`probstat.modeling.ProbabilisticModel`) uses to treat such runs
explicitly instead of losing a whole stage to one bad run:

- :data:`FailurePolicy` ``"raise"`` (default: the original exception
  propagates), ``"skip"`` (the run is dropped and logged with its error text),
  or ``"record"`` (the run is dropped and logged with its inputs and error
  text). No policy imputes a value; a failed run never enters an estimate.
- :class:`FailedEvaluation` and :class:`FailureLog`: NaN-free records of
  failed attempts, counted separately from completed evaluations.
- :class:`EvaluationCache`: exact-match memoisation keyed by the full
  parameter mapping, so stages can reuse runs; cache hits are not model
  evaluations and are reported separately.
- :class:`FrozenMapping`: a read-only mapping that, unlike
  :class:`types.MappingProxyType`, can be pickled, so models and results can
  cross process boundaries and be saved.

Statistical caution: dropping failed runs conditions every estimate on the
event "the simulator succeeded". The estimates are unbiased for the full
input law only when failures are ignorable, i.e. when failing is independent
of the outputs one would have observed (missing completely at random). When
failures concentrate in a region of the input space (for example a solver
that diverges at extreme loads) the surviving sample is biased, and the
failure fraction should be reported next to every estimate.
"""

from __future__ import annotations

import copyreg
import math
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeVar

FailurePolicy = Literal["raise", "skip", "record"]
FAILURE_POLICIES: tuple[str, ...] = ("raise", "skip", "record")

_K = TypeVar("_K")
_V = TypeVar("_V")


def require_failure_policy(policy: object) -> FailurePolicy:
    """Validate and return a failure policy.

    Args:
        policy: Candidate policy name.

    Returns:
        The policy as a :data:`FailurePolicy` literal.

    Raises:
        ValueError: If ``policy`` is not one of :data:`FAILURE_POLICIES`.

    References:
        Project validation policy. No equation is evaluated.

    """
    if policy == "raise":
        return "raise"
    if policy == "skip":
        return "skip"
    if policy == "record":
        return "record"
    raise ValueError(f"on_failure must be one of {FAILURE_POLICIES}, got {policy!r}")


class FrozenMapping(Mapping[_K, _V]):
    """Read-only, picklable mapping used by results and models.

    Item assignment raises :class:`TypeError` exactly like
    :class:`types.MappingProxyType`; equality follows :class:`Mapping`
    (equal to any mapping with the same items). The mapping owns a private
    copy of its data, so later changes to the source do not leak in.

    Args:
        data: Mapping or iterable of key-value pairs.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """

    __slots__ = ("_data",)
    _data: dict[Any, Any]

    def __init__(self, data: Mapping[_K, _V] | Any = (), /) -> None:
        """Copy the items into a private dictionary.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: _K) -> _V:
        """Return the value stored under ``key``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return self._data[key]  # type: ignore[no-any-return]

    def __iter__(self) -> Iterator[_K]:
        """Iterate over keys in insertion order.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is present.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return key in self._data

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute assignment so the mapping stays read-only.

        Raises:
            AttributeError: Always.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        raise AttributeError("FrozenMapping is read-only")

    def __repr__(self) -> str:
        """Return ``FrozenMapping({...})``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return f"FrozenMapping({self._data!r})"

    def __reduce__(self) -> tuple[Any, tuple[dict[_K, _V]]]:
        """Pickle as the constructor applied to a plain dictionary copy.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return (type(self), (dict(self._data),))


def _rebuild_mapping_proxy(data: dict[Any, Any]) -> MappingProxyType[Any, Any]:
    """Rebuild a pickled read-only mapping proxy over its dictionary copy.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return MappingProxyType(data)


def _reduce_mapping_proxy(proxy: MappingProxyType[Any, Any]) -> tuple[Any, Any]:
    """Pickle a read-only mapping proxy as a proxy over a dictionary copy.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return (_rebuild_mapping_proxy, (dict(proxy),))


# Results throughout the library use ``MappingProxyType`` fields. The standard
# library cannot pickle them, so models, results, and study records could not
# be saved or sent to worker processes. Registering a reducer in the copyreg
# dispatch table (the documented extension point of ``pickle`` and ``copy``)
# makes every such field picklable; unpickling builds a fresh read-only proxy
# over a private dictionary copy. Registration is idempotent.
copyreg.pickle(MappingProxyType, _reduce_mapping_proxy)


@dataclass(frozen=True, slots=True)
class FailedEvaluation:
    """One failed forward-model attempt; contains no NaN placeholder.

    Args:
        index: Position of the attempt in its batch, or ``None`` for a single
            call.
        inputs: Parameter mapping passed to the model (empty under the
            ``"skip"`` policy, which keeps only the error text).
        error_type: Exception class name, for example ``"ZeroDivisionError"``.
        message: Error text.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """

    index: int | None
    inputs: Mapping[str, object]
    error_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary of the failure.

        References:
            Project reporting policy. No equation is evaluated.

        """
        return {
            "index": self.index,
            "inputs": {str(k): json_safe(v) for k, v in self.inputs.items()},
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class FailureLog:
    """Mutable log of failed forward-model attempts.

    ``count`` is exact; at most ``max_records`` records are kept (the first
    ones), so a simulator that fails on every run cannot exhaust memory.

    Args:
        count: Failed attempts so far.
        records: Stored failure records.
        max_records: Maximum number of stored records.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """

    count: int = 0
    records: list[FailedEvaluation] = field(default_factory=list)
    max_records: int = 10_000

    def add(self, failure: FailedEvaluation) -> None:
        """Count one failure and keep its record while space remains.

        Args:
            failure: Failure record.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        self.count += 1
        if len(self.records) < self.max_records:
            self.records.append(failure)


class ModelEvaluationError(RuntimeError):
    """A single forward-model evaluation failed under a non-raising policy.

    Iterative algorithms (FORM, MCMC, pick-freeze Sobol designs) cannot drop
    a run, so under ``"skip"`` or ``"record"`` a failure of a single
    :meth:`~probstat.modeling.ProbabilisticModel.evaluate` call is logged and
    counted, then reported with this exception. Batch evaluation drops the
    run instead.

    Args:
        failure: Record of the failed attempt.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """

    def __init__(self, failure: FailedEvaluation) -> None:
        """Store the failure and build the message.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        super().__init__(
            f"forward-model evaluation failed ({failure.error_type}: "
            f"{failure.message})"
        )
        self.failure = failure

    def __reduce__(self) -> tuple[Any, tuple[FailedEvaluation]]:
        """Pickle through the constructor.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return (type(self), (self.failure,))


class EvaluationBudgetExceeded(RuntimeError):
    """A stage would exceed, or did exceed, the evaluation budget.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """


class EvaluationCache:
    """Exact-match memoisation of validated forward-model outputs.

    Keys are the complete parameter mapping (random inputs and fixed
    parameters) as a sorted tuple of ``(name, value)`` pairs, so a hit
    requires bit-identical inputs. Calls with an unhashable parameter value
    are not cached. Only successful runs are stored; a failed run is attempted
    again on the next request. A cache must only be shared between models with
    the same deterministic forward map.

    Args:
        max_entries: Optional cap on stored entries; once reached, new results
            are not stored (existing entries keep serving hits).

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """

    def __init__(self, max_entries: int | None = None) -> None:
        """Create an empty cache.

        Raises:
            ValueError: If ``max_entries`` is not a positive integer.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if max_entries is not None and (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer or None")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._store: dict[Hashable, Mapping[str, float]] = {}

    @staticmethod
    def key(parameters: Mapping[str, object]) -> Hashable | None:
        """Return the cache key of a parameter mapping, or ``None``.

        Args:
            parameters: Complete parameter mapping passed to the model.

        Returns:
            Hashable key, or ``None`` when a value is unhashable.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        items = tuple(sorted(parameters.items()))
        try:
            hash(items)
        except TypeError:
            return None
        return items

    def get(
        self, key: Hashable | None, required: Iterable[str] = ()
    ) -> Mapping[str, float] | None:
        """Return the stored outputs for ``key`` and count a hit or miss.

        Args:
            key: Key from :meth:`key` (``None`` always misses).
            required: Output names the entry must contain to count as a hit.

        Returns:
            Stored outputs or ``None``.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        found = None if key is None else self._store.get(key)
        if found is not None and any(name not in found for name in required):
            found = None
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: Hashable | None, outputs: Mapping[str, float]) -> None:
        """Store validated outputs under ``key``.

        Args:
            key: Key from :meth:`key` (``None`` is ignored).
            outputs: Validated QoI outputs.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if key is None:
            return
        if self.max_entries is not None and len(self._store) >= self.max_entries:
            return
        self._store[key] = outputs

    def __len__(self) -> int:
        """Return the number of stored entries.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return len(self._store)

    def clear(self) -> None:
        """Remove every entry and reset the hit and miss counts.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        """Return a short summary of the cache state.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return (
            f"EvaluationCache(entries={len(self._store)}, hits={self.hits}, "
            f"misses={self.misses})"
        )


def describe_error(error: BaseException) -> tuple[str, str]:
    """Return the class name and text of an exception.

    Args:
        error: Exception raised by or about a forward-model run.

    Returns:
        ``(error_type, message)``.

    References:
        Project reporting policy. No equation is evaluated.

    """
    return type(error).__name__, str(error)


def json_safe(value: object) -> object:
    """Convert a value to a JSON-safe structure.

    Finite floats and integers stay numbers; non-finite floats become the
    strings ``"inf"``, ``"-inf"``, or ``"nan"``; tuples, lists, and sets
    become lists; mappings become dictionaries with string keys; booleans,
    strings, and ``None`` pass unchanged; anything else becomes its
    ``repr``.

    Args:
        value: Value to convert.

    Returns:
        A structure accepted by ``json.dumps(..., allow_nan=False)``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((json_safe(item) for item in value), key=repr)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, str | int | float):
        return json_safe(enum_value)
    return repr(value)
