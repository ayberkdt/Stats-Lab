"""Batch evaluation contract: algorithms request points, the caller runs them.

An expensive simulator is rarely a Python function of one point. It runs
whole populations at once on a GPU, on a cluster, or in another process.
An adaptive algorithm written as ``for x in points: model(x)`` cannot use
that. This module separates the two roles:

- an algorithm is a *procedure*: a generator that yields an
  :class:`EvaluationRequest` (many points, one label naming the model or
  fidelity) and receives the outputs in the same order;
- the caller owns the simulator and decides how a request is evaluated.

A procedure can be driven in three ways: :func:`run_procedure` with one
batch callable per label, :class:`AskTellSession` when the evaluation
happens outside the Python process (``ask`` -> run the batch anywhere ->
``tell``), or :func:`batch_from_pointwise` to reuse an ordinary function of
one point. All three give bit-identical results because the procedure never
sees how its request was evaluated.

Outputs are duck-typed: any iterable with one entry per requested point is
accepted, so a NumPy array returned by a vectorised simulator works without
this library importing NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

Point = tuple[float, ...]
BatchModel = Callable[[Sequence[Point]], Iterable[Any]]
ResultT = TypeVar("ResultT")
Procedure = Generator["EvaluationRequest", Sequence[Any], ResultT]


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """A set of input points to be evaluated by one named model.

    Attributes:
        points: Input points, one tuple of floats per requested evaluation.
        label: Name of the model, fidelity, or stage that must evaluate the
            points (for example ``"high"`` or ``"low"``).
        stage: Free-text description of why the algorithm asks (for example
            ``"level 3"``); it is diagnostic only.

    References:
        [EQ-BATCH-001], [REF-PEP-342].

    """

    points: tuple[Point, ...]
    label: str = "model"
    stage: str = ""

    def __post_init__(self) -> None:
        """Validate the label and require at least one point.

        Raises:
            TypeError: If ``label`` or ``stage`` is not text.
            ValueError: If no point is requested.

        References:
            [EQ-BATCH-001].

        """
        if not isinstance(self.label, str) or not isinstance(self.stage, str):
            raise TypeError("label and stage must be strings")
        if not self.label:
            raise ValueError("label must be a non-empty string")
        if len(self.points) == 0:
            raise ValueError("an evaluation request needs at least one point")

    @property
    def size(self) -> int:
        """Return the number of requested evaluations.

        Returns:
            Number of points in the request.

        References:
            [EQ-BATCH-001].

        """
        return len(self.points)


def request(
    points: Iterable[Iterable[float]], *, label: str = "model", stage: str = ""
) -> EvaluationRequest:
    """Build a validated :class:`EvaluationRequest` from any nested iterable.

    Args:
        points: Iterable of points; each point is an iterable of finite reals.
        label: Name of the model that must evaluate the points.
        stage: Diagnostic description of the algorithm stage.

    Returns:
        Request holding the points as tuples of floats.

    Raises:
        ValueError: If no point is given, the points differ in length, or a
            coordinate is not finite.

    References:
        [EQ-BATCH-001].

    """
    rows = tuple(tuple(float(value) for value in point) for point in points)
    if rows:
        width = len(rows[0])
        for row in rows:
            if len(row) != width:
                raise ValueError("all requested points must have the same length")
            if not all(math.isfinite(value) for value in row):
                raise ValueError("requested points must be finite")
    return EvaluationRequest(points=rows, label=label, stage=stage)


def scalar_outputs(values: Iterable[Any], count: int) -> tuple[float, ...]:
    """Coerce one scalar output per requested point to a tuple of floats.

    Non-finite outputs are kept: a simulator may legitimately return ``nan``
    or ``inf`` for a failed run, and the algorithm, not the transport layer,
    must decide what that means.

    Args:
        values: Iterable with one real number per requested point (a list, a
            tuple, or a one-dimensional array).
        count: Number of requested points.

    Returns:
        Outputs as floats in request order.

    Raises:
        ValueError: If the number of outputs differs from ``count``.
        TypeError: If an output cannot be converted to ``float``.

    References:
        [EQ-BATCH-001].

    """
    result = tuple(float(value) for value in values)
    if len(result) != count:
        raise ValueError(
            f"expected {count} outputs for {count} requested points, "
            f"received {len(result)}"
        )
    return result


def vector_outputs(
    values: Iterable[Iterable[Any]], count: int, *, width: int | None = None
) -> tuple[Point, ...]:
    """Coerce one output vector per requested point to tuples of floats.

    Args:
        values: Iterable with one iterable of reals per requested point (a
            nested list or a two-dimensional array, one row per point).
        count: Number of requested points.
        width: Required output dimension; ``None`` accepts any common length.

    Returns:
        Output vectors in request order.

    Raises:
        ValueError: If the number of rows differs from ``count``, the rows
            differ in length, or the length differs from ``width``.
        TypeError: If an entry cannot be converted to ``float``.

    References:
        [EQ-BATCH-001].

    """
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if len(rows) != count:
        raise ValueError(
            f"expected {count} output rows for {count} requested points, "
            f"received {len(rows)}"
        )
    expected = len(rows[0]) if width is None and rows else width
    for row in rows:
        if len(row) != expected:
            raise ValueError("every output row must have the same length")
    return rows


def batch_from_pointwise(function: Callable[[Point], Any]) -> BatchModel:
    """Wrap a function of one point as a batch model.

    Args:
        function: Callable that maps one point to one output.

    Returns:
        Batch model that evaluates the points one after another, in order.

    Raises:
        TypeError: If ``function`` is not callable.

    References:
        [EQ-BATCH-001].

    """
    if not callable(function):
        raise TypeError("function must be callable")

    def evaluate(points: Sequence[Point]) -> list[Any]:
        """Evaluate the wrapped function point by point.

        Args:
            points: Requested points.

        Returns:
            Outputs in request order.

        References:
            [EQ-BATCH-001].

        """
        return [function(point) for point in points]

    return evaluate


class AskTellSession(Generic[ResultT]):
    """Drive a procedure by hand: ``ask`` for points, ``tell`` the outputs.

    The session owns no simulator. It is the form to use when a batch is run
    by a job scheduler, another process, or a GPU code with its own driver:

    .. code-block:: python

        session = AskTellSession(procedure)
        while (pending := session.ask()) is not None:
            session.tell(run_somewhere(pending.points, pending.label))
        answer = session.result

    Args:
        procedure: Generator that yields requests and returns the result.

    Attributes:
        requests: Number of requests answered so far.

    References:
        [EQ-BATCH-001], [REF-PEP-342].

    """

    def __init__(self, procedure: Procedure[ResultT]) -> None:
        """Start the procedure and stop at its first request.

        Args:
            procedure: Generator that yields requests and returns the result.

        Raises:
            TypeError: If the procedure yields something that is not an
                :class:`EvaluationRequest`.

        References:
            [EQ-BATCH-001], [REF-PEP-342].

        """
        self._procedure = procedure
        self._pending: EvaluationRequest | None = None
        self._done = False
        self._result: ResultT | None = None
        self._evaluations: dict[str, int] = {}
        self.requests = 0
        self._advance(None)

    def _advance(self, values: Sequence[Any] | None) -> None:
        """Resume the procedure until its next request or its return.

        Args:
            values: Outputs of the pending request, or ``None`` to start.

        Raises:
            TypeError: If the procedure yields a non-request object.

        References:
            [REF-PEP-342].

        """
        try:
            if values is None:
                pending = next(self._procedure)
            else:
                pending = self._procedure.send(values)
        except StopIteration as stop:
            self._pending = None
            self._done = True
            self._result = stop.value
            return
        if not isinstance(pending, EvaluationRequest):
            raise TypeError("a procedure must yield EvaluationRequest objects")
        self._pending = pending

    @property
    def done(self) -> bool:
        """Return whether the procedure has returned its result.

        Returns:
            ``True`` once no request is pending and the result is available.

        References:
            [EQ-BATCH-001].

        """
        return self._done

    @property
    def evaluations(self) -> dict[str, int]:
        """Return the number of answered evaluations per model label.

        Returns:
            New dictionary mapping each label to its evaluation count.

        References:
            [EQ-BATCH-001].

        """
        return dict(self._evaluations)

    def ask(self) -> EvaluationRequest | None:
        """Return the pending request without consuming it.

        Returns:
            The request waiting for outputs, or ``None`` when the procedure
            has finished. Repeated calls return the same request.

        References:
            [EQ-BATCH-001].

        """
        return self._pending

    def tell(self, values: Iterable[Any]) -> None:
        """Answer the pending request and advance the procedure.

        Args:
            values: One output per requested point, in request order.

        Raises:
            RuntimeError: If no request is pending.
            ValueError: If the number of outputs differs from the number of
                requested points.

        References:
            [EQ-BATCH-001], [REF-PEP-342].

        """
        pending = self._pending
        if pending is None:
            raise RuntimeError("no evaluation request is pending")
        outputs = list(values)
        if len(outputs) != pending.size:
            raise ValueError(
                f"request '{pending.label}' asked for {pending.size} outputs, "
                f"received {len(outputs)}"
            )
        self._evaluations[pending.label] = (
            self._evaluations.get(pending.label, 0) + pending.size
        )
        self.requests += 1
        self._advance(outputs)

    @property
    def result(self) -> ResultT:
        """Return the value returned by the finished procedure.

        Returns:
            Result of the procedure.

        Raises:
            RuntimeError: If the procedure still waits for outputs.

        References:
            [EQ-BATCH-001].

        """
        if not self._done:
            raise RuntimeError("the procedure has not finished; answer ask() first")
        return self._result  # type: ignore[return-value]


def run_procedure(
    procedure: Procedure[ResultT],
    models: BatchModel | Mapping[str, BatchModel],
    *,
    max_batch_size: int | None = None,
) -> ResultT:
    """Run a procedure to completion with batch models supplied by the caller.

    Args:
        procedure: Generator that yields requests and returns the result.
        models: One batch model used for every label, or a mapping from
            request label to batch model.
        max_batch_size: Largest number of points passed to a model in one
            call; larger requests are split into consecutive chunks and the
            outputs concatenated in order. ``None`` passes each request whole.

    Returns:
        Result of the procedure.

    Raises:
        KeyError: If a request names a label missing from ``models``.
        ValueError: If ``max_batch_size`` is not a positive integer or a
            model returns the wrong number of outputs.

    References:
        [EQ-BATCH-001], [REF-PEP-342].

    """
    if max_batch_size is not None and (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size < 1
    ):
        raise ValueError("max_batch_size must be a positive integer")
    session = AskTellSession(procedure)
    while True:
        pending = session.ask()
        if pending is None:
            break
        if isinstance(models, Mapping):
            if pending.label not in models:
                raise KeyError(
                    f"no batch model was supplied for label '{pending.label}'"
                )
            model = models[pending.label]
        else:
            model = models
        if max_batch_size is None:
            outputs = list(model(pending.points))
        else:
            outputs = []
            for start in range(0, pending.size, max_batch_size):
                chunk = pending.points[start : start + max_batch_size]
                part = list(model(chunk))
                if len(part) != len(chunk):
                    raise ValueError(
                        f"model '{pending.label}' returned {len(part)} outputs "
                        f"for {len(chunk)} points"
                    )
                outputs.extend(part)
        session.tell(outputs)
    return session.result
