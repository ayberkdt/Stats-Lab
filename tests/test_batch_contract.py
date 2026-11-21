"""Tests of the batch evaluation contract in ``probstat.foundations``."""

from __future__ import annotations

import unittest
from collections.abc import Generator, Sequence
from typing import Any

from probstat.foundations import (
    AskTellSession,
    EvaluationRequest,
    batch_from_pointwise,
    request,
    run_procedure,
    scalar_outputs,
    vector_outputs,
)


def _two_stage() -> Generator[EvaluationRequest, Sequence[Any], float]:
    """Yield two requests; the second depends on the first response."""
    first = yield request([(1.0,), (2.0,), (3.0,)], label="low", stage="pilot")
    centre = sum(scalar_outputs(first, 3)) / 3.0
    second = yield request([(centre,), (centre + 1.0,)], label="high")
    return sum(scalar_outputs(second, 2))


class BatchContractTest(unittest.TestCase):
    """The three drivers agree and the transport validates sizes."""

    def test_run_procedure_matches_hand_computation(self) -> None:
        """Square then double: centre 14/3, result 2*(2*14/3 + 1)."""
        models = {
            "low": lambda points: [p[0] ** 2 for p in points],
            "high": lambda points: [2.0 * p[0] for p in points],
        }
        self.assertAlmostEqual(
            run_procedure(_two_stage(), models), 2.0 * (2.0 * 14.0 / 3.0 + 1.0)
        )

    def test_ask_tell_and_chunking_give_identical_results(self) -> None:
        """Ask-tell, whole batches and chunks of one return the same value."""
        low = batch_from_pointwise(lambda p: p[0] ** 2)
        high = batch_from_pointwise(lambda p: 2.0 * p[0])
        models = {"low": low, "high": high}
        whole = run_procedure(_two_stage(), models)
        chunked = run_procedure(_two_stage(), models, max_batch_size=1)
        session = AskTellSession(_two_stage())
        labels = []
        while (pending := session.ask()) is not None:
            labels.append((pending.label, pending.stage, pending.size))
            session.tell(models[pending.label](pending.points))
        self.assertEqual(whole, chunked)
        self.assertEqual(whole, session.result)
        self.assertEqual(labels, [("low", "pilot", 3), ("high", "", 2)])
        self.assertEqual(session.evaluations, {"low": 3, "high": 2})
        self.assertEqual(session.requests, 2)
        self.assertTrue(session.done)

    def test_ask_is_idempotent_and_result_needs_completion(self) -> None:
        """Asking twice returns the same request; early result access fails."""
        session = AskTellSession(_two_stage())
        self.assertIs(session.ask(), session.ask())
        with self.assertRaises(RuntimeError):
            _ = session.result
        with self.assertRaises(ValueError):
            session.tell([1.0])
        session.tell([1.0, 4.0, 9.0])
        session.tell([0.0, 0.0])
        with self.assertRaises(RuntimeError):
            session.tell([0.0])

    def test_single_model_serves_every_label_and_missing_label_fails(self) -> None:
        """A bare callable answers all labels; a mapping must be complete."""
        self.assertEqual(
            run_procedure(_two_stage(), lambda points: [1.0 for _ in points]), 2.0
        )
        with self.assertRaises(KeyError):
            run_procedure(_two_stage(), {"low": lambda points: [0.0] * len(points)})
        with self.assertRaises(ValueError):
            run_procedure(_two_stage(), lambda points: [0.0], max_batch_size=2)
        with self.assertRaises(ValueError):
            run_procedure(_two_stage(), lambda points: [0.0], max_batch_size=0)

    def test_request_validation(self) -> None:
        """Empty, ragged and non-finite requests are rejected."""
        with self.assertRaises(ValueError):
            request([])
        with self.assertRaises(ValueError):
            request([(1.0, 2.0), (1.0,)])
        with self.assertRaises(ValueError):
            request([(float("nan"),)])
        with self.assertRaises(ValueError):
            request([(1.0,)], label="")
        self.assertEqual(request([[1, 2]]).points, ((1.0, 2.0),))

    def test_output_coercion(self) -> None:
        """Outputs keep non-finite values and enforce shape."""
        self.assertEqual(scalar_outputs(iter([1, 2.5]), 2), (1.0, 2.5))
        self.assertTrue(scalar_outputs([float("nan")], 1)[0] != 0.0)
        with self.assertRaises(ValueError):
            scalar_outputs([1.0], 2)
        self.assertEqual(vector_outputs([[1, 2], [3, 4]], 2), ((1.0, 2.0), (3.0, 4.0)))
        with self.assertRaises(ValueError):
            vector_outputs([[1, 2], [3]], 2)
        with self.assertRaises(ValueError):
            vector_outputs([[1, 2]], 1, width=3)
        with self.assertRaises(ValueError):
            vector_outputs([[1, 2]], 2)

    def test_non_request_yield_is_rejected(self) -> None:
        """A procedure that yields a plain object is a programming error."""

        def bad() -> Generator[Any, Sequence[Any], float]:
            """Yield something that is not a request."""
            yield [1.0]
            return 0.0

        with self.assertRaises(TypeError):
            AskTellSession(bad())
        with self.assertRaises(TypeError):
            batch_from_pointwise(3)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
