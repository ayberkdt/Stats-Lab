"""Model contract and evaluation engine (ADR-0058, workstream W1).

Covers the failure policy, exact counting, strict input names, non-float fixed
parameters, picklability, counters of derived models, batch/parallel/cached
evaluation, evaluation limits, and the ``StrEnum`` vocabulary.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import MappingProxyType

from probstat.distributions import Normal, Uniform
from probstat.modeling import (
    EvaluationBatch,
    EvaluationCounter,
    JointInput,
    ProbabilisticModel,
    RandomInput,
)
from probstat.uq import (
    ConvergenceType,
    EvaluationBudgetExceeded,
    EvaluationCache,
    FailureLog,
    FrozenMapping,
    ModelEvaluationError,
    ProblemDirection,
    QuantityOfInterest,
    StepStatus,
    UncertaintyNature,
    UncertaintySourceKind,
    WorkflowStep,
    evaluate_model,
)

Y = (QuantityOfInterest("y", "response"),)


def fragile(point):
    """Return ``y = x`` but fail for ``x > 0.8`` in four different ways."""
    x = point["x"]
    if x > 0.95:
        raise ZeroDivisionError("solver diverged")
    if x > 0.9:
        return {"y": math.nan}
    if x > 0.85:
        return {"z": x}
    if x > 0.8:
        return 1.0
    return {"y": x}


def scaled(point):
    """Return ``y = gain * x`` with a string option that must arrive intact."""
    if point["mode"] != "linear" or point["order"] != 2 or point["flags"] != (1, 2):
        raise AssertionError("fixed parameters were altered")
    if not isinstance(point["order"], int) or point["verbose"] is not False:
        raise AssertionError("fixed parameter types were altered")
    return {"y": point["gain"] * point["x"]}


def uniform_model(function=fragile, **fixed):
    """Return a one-input model on ``x ~ U(0, 1)``."""
    return ProbabilisticModel(
        JointInput.independent(RandomInput("x", Uniform(0.0, 1.0))),
        function,
        Y,
        fixed=fixed,
    )


class VectorisedLine:
    """Forward model with an ``evaluate_batch`` hook; counts its calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.batch_calls = 0
        self.single_calls = 0

    def __call__(self, point):
        """Evaluate one point."""
        self.single_calls += 1
        return {"y": 2.0 * point["x"]}

    def evaluate_batch(self, points):
        """Evaluate many points in one call (fails at x > 0.9)."""
        self.batch_calls += 1
        return [
            ValueError("bad point") if p["x"] > 0.9 else {"y": 2.0 * p["x"]}
            for p in points
        ]


class FailurePolicyTests(unittest.TestCase):
    """0.1-0.3: failed runs are counted separately and partial outputs kept."""

    POINTS = ((0.1,), (0.97,), (0.5,), (0.92,), (0.88,), (0.82,), (0.3,))

    def test_raise_policy_is_unchanged_and_counts_failure_separately(self) -> None:
        """The original exception propagates; only completed runs are counted."""
        model = uniform_model()
        with self.assertRaises(ZeroDivisionError):
            model.evaluate_many(self.POINTS)
        self.assertEqual(model.evaluations, 1)
        self.assertEqual(model.failed_evaluations, 1)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            model.evaluate((0.93,))
        with self.assertRaisesRegex(ValueError, "missing QoI"):
            model.evaluate((0.86,))
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            model.evaluate((0.81,))
        self.assertEqual(model.evaluations, 1)
        self.assertEqual(model.failed_evaluations, 4)

    def test_skip_drops_failures_and_keeps_every_good_run(self) -> None:
        """Four failure modes are dropped; the three good runs survive."""
        model = uniform_model()
        batch = model.evaluate_many(self.POINTS, on_failure="skip")
        self.assertIsInstance(batch, EvaluationBatch)
        self.assertEqual(batch["y"], (0.1, 0.5, 0.3))
        self.assertEqual(batch.indices, (0, 2, 6))
        self.assertEqual([f.index for f in batch.failures], [1, 3, 4, 5])
        self.assertEqual(
            [f.error_type for f in batch.failures],
            ["ZeroDivisionError", "ValueError", "ValueError", "TypeError"],
        )
        self.assertEqual(dict(batch.failures[0].inputs), {})  # skip keeps no inputs
        self.assertEqual(batch.model_evaluations, 3)
        self.assertEqual(batch.failed_evaluations, 4)
        self.assertAlmostEqual(batch.failure_fraction, 4 / 7)
        self.assertEqual(model.evaluations, 3)
        self.assertEqual(model.failed_evaluations, 4)
        self.assertEqual(model.failures.count, 4)

    def test_record_keeps_nan_free_inputs_and_error_text(self) -> None:
        """Recorded failures carry their inputs and the error message."""
        model = uniform_model()
        batch = model.evaluate_many(self.POINTS, on_failure="record")
        first = batch.failures[0]
        self.assertEqual(dict(first.inputs), {"x": 0.97})
        self.assertEqual(first.message, "solver diverged")
        for failure in batch.failures:
            for value in failure.inputs.values():
                self.assertTrue(math.isfinite(value))
        self.assertEqual(first.to_dict()["error_type"], "ZeroDivisionError")

    def test_single_call_under_skip_raises_model_evaluation_error(self) -> None:
        """One call cannot drop its run, so it reports the logged failure."""
        model = dataclasses.replace(uniform_model(), on_failure="record")
        with self.assertRaises(ModelEvaluationError) as caught:
            model.evaluate((0.99,))
        self.assertEqual(caught.exception.failure.error_type, "ZeroDivisionError")
        self.assertIsInstance(caught.exception.__cause__, ZeroDivisionError)
        self.assertEqual(model.failed_evaluations, 1)
        self.assertEqual(model.evaluations, 0)
        restored = pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual(restored.failure, caught.exception.failure)

    def test_evaluate_model_policies(self) -> None:
        """The low-level contract drops failures under skip and logs them."""
        log = FailureLog()
        self.assertIsNone(
            evaluate_model(fragile, {"x": 0.99}, Y, on_failure="skip", failures=log)
        )
        result = evaluate_model(fragile, {"x": 0.2}, Y, on_failure="skip", failures=log)
        self.assertEqual(dict(result or {}), {"y": 0.2})
        self.assertEqual(log.count, 1)
        with self.assertRaises(ZeroDivisionError):
            evaluate_model(fragile, {"x": 0.99}, Y, failures=log)
        self.assertEqual(log.count, 2)
        # Argument errors are never treated as run failures.
        with self.assertRaises(ValueError):
            evaluate_model(fragile, {"x": math.nan}, Y, on_failure="skip")
        with self.assertRaises(ValueError):
            evaluate_model(fragile, {"x": 0.2}, Y, on_failure="ignore")  # type: ignore[call-overload]

    def test_invalid_policy_is_rejected(self) -> None:
        """Unknown policies fail at construction and per call."""
        with self.assertRaises(ValueError):
            ProbabilisticModel(
                uniform_model().inputs, fragile, Y, on_failure="impute"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            uniform_model().evaluate_many([(0.1,)], on_failure="impute")  # type: ignore[arg-type]


class StrictNameTests(unittest.TestCase):
    """0.16: a mapping must name exactly the random inputs."""

    def test_missing_unknown_and_fixed_names_are_rejected(self) -> None:
        """Each problem is named in the error message."""
        model = uniform_model(scaled, gain=3.0, mode="linear", order=2, flags=(1, 2),
                              verbose=False)  # fmt: skip
        self.assertEqual(model.evaluate({"x": 0.5})["y"], 1.5)
        with self.assertRaisesRegex(ValueError, r"missing inputs \['x'\]"):
            model.evaluate({})
        with self.assertRaisesRegex(ValueError, r"unknown input names \['xx'\]"):
            model.evaluate({"x": 0.5, "xx": 1.0})
        with self.assertRaisesRegex(ValueError, "fixed parameters"):
            model.evaluate({"x": 0.5, "gain": 10.0})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            model.evaluate({"x": math.inf})
        with self.assertRaises(TypeError):
            model.evaluate({"x": "0.5"})  # type: ignore[dict-item]
        self.assertEqual(model.evaluations, 1)

    def test_non_float_fixed_values_pass_unchanged(self) -> None:
        """Strings, integers, Booleans, and tuples reach the model as given."""
        model = uniform_model(scaled, gain=3.0, mode="linear", order=2, flags=(1, 2),
                              verbose=False)  # fmt: skip
        batch = model.evaluate_many([(0.1,), (0.2,)])
        self.assertEqual(batch["y"], (0.30000000000000004, 0.6000000000000001))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            uniform_model(scaled, gain=math.nan)
        with self.assertRaises(TypeError):
            uniform_model(scaled, **{"": 1.0})


class PicklabilityAndCounterTests(unittest.TestCase):
    """P1.3: models and batches pickle; derived models get fresh counters."""

    def test_model_and_batch_round_trip_through_pickle(self) -> None:
        """A module-level forward model makes the whole model picklable."""
        model = uniform_model(scaled, gain=2.0, mode="linear", order=2, flags=(1, 2),
                              verbose=False)  # fmt: skip
        batch = model.evaluate_many([(0.25,), (0.5,)])
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.fixed, model.fixed)
        self.assertEqual(restored.evaluations, 2)
        self.assertEqual(restored.evaluate((0.25,))["y"], 0.5)
        again = pickle.loads(pickle.dumps(batch))
        self.assertEqual(again, batch)
        self.assertEqual(again.indices, (0, 1))
        self.assertEqual(again.model_evaluations, 2)
        self.assertEqual(copy.deepcopy(model.fixed), model.fixed)

    def test_mapping_proxies_pickle_after_import(self) -> None:
        """Legacy MappingProxyType fields of results become picklable."""
        proxy = MappingProxyType({"a": 1.0})
        restored = pickle.loads(pickle.dumps(proxy))
        self.assertIsInstance(restored, MappingProxyType)
        self.assertEqual(dict(restored), {"a": 1.0})

    def test_frozen_mapping_is_read_only(self) -> None:
        """Item and attribute assignment raise like a mapping proxy."""
        frozen = FrozenMapping({"a": 1})
        with self.assertRaises(TypeError):
            frozen["a"] = 2  # type: ignore[index]
        with self.assertRaises(AttributeError):
            frozen._data = {}  # type: ignore[misc]
        self.assertEqual(frozen, {"a": 1})
        self.assertEqual(repr(frozen), "FrozenMapping({'a': 1})")

    def test_replace_gives_a_fresh_counter_linked_to_the_source(self) -> None:
        """The derived model starts at zero; the source still sees its runs."""
        source = uniform_model()
        source.evaluate((0.1,))
        derived = dataclasses.replace(source, name="derived")
        self.assertIsNot(derived.counter, source.counter)
        self.assertEqual(derived.evaluations, 0)
        derived.evaluate((0.2,))
        derived.evaluate((0.3,))
        self.assertEqual(derived.evaluations, 2)
        self.assertEqual(source.evaluations, 3)
        source.evaluate((0.4,))
        self.assertEqual(derived.evaluations, 2)

    def test_counter_increments_only_after_success(self) -> None:
        """0.2: the counter does not move for a run that raised."""
        model = uniform_model()
        with self.assertRaises(ZeroDivisionError):
            model.evaluate((0.99,))
        self.assertEqual(model.counter, EvaluationCounter(0, 1, 0))


class EvaluationEngineTests(unittest.TestCase):
    """P1.5: executors, the batch hook, and the cache."""

    POINTS = tuple((k / 40.0,) for k in range(40))

    def test_thread_pool_matches_sequential_outputs_and_counts(self) -> None:
        """Parallel evaluation is order-preserving and exactly counted."""
        sequential = uniform_model().evaluate_many(self.POINTS, on_failure="skip")
        model = uniform_model()
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = model.evaluate_many(
                self.POINTS, executor=pool, batch=3, on_failure="skip"
            )
        self.assertEqual(parallel, sequential)
        self.assertEqual(parallel.indices, sequential.indices)
        self.assertEqual(
            [f.index for f in parallel.failures], [f.index for f in sequential.failures]
        )
        self.assertEqual(model.evaluations, 33)
        self.assertEqual(model.failed_evaluations, 7)

    def test_thread_pool_raise_policy_counts_every_submitted_run(self) -> None:
        """Under raise, all submitted runs are counted before re-raising."""
        model = uniform_model()
        with ThreadPoolExecutor(max_workers=3) as pool:
            with self.assertRaises(TypeError):
                model.evaluate_many(self.POINTS, executor=pool)
        self.assertEqual(model.evaluations + model.failed_evaluations, 40)
        self.assertEqual(model.failed_evaluations, 7)

    def test_process_pool_with_a_picklable_model(self) -> None:
        """A builtin forward map (``dict``: y is the input) runs in processes."""
        inputs = JointInput.independent(RandomInput("y", Normal(0.0, 1.0)))
        model = ProbabilisticModel(inputs, dict, Y)
        points = tuple((0.1 * k,) for k in range(8))
        try:
            with ProcessPoolExecutor(max_workers=2) as pool:
                batch = model.evaluate_many(points, executor=pool, batch=2)
        except (OSError, PermissionError, NotImplementedError) as error:
            self.skipTest(f"process pools unavailable: {error}")
        self.assertEqual(batch["y"], tuple(p[0] for p in points))
        self.assertEqual(model.evaluations, 8)

    def test_batch_hook_receives_chunks_and_reports_member_failures(self) -> None:
        """``evaluate_batch`` is used in chunks; failed members are dropped."""
        line = VectorisedLine()
        model = uniform_model(line)
        batch = model.evaluate_many(self.POINTS, batch=16, on_failure="skip")
        self.assertEqual(line.batch_calls, 3)
        self.assertEqual(line.single_calls, 0)
        self.assertEqual(len(batch["y"]), 37)
        self.assertEqual([f.index for f in batch.failures], [37, 38, 39])
        self.assertEqual(model.evaluations, 37)
        self.assertEqual(model.failed_evaluations, 3)
        # Single evaluations still use __call__.
        self.assertEqual(model.evaluate((0.25,))["y"], 0.5)
        self.assertEqual(line.single_calls, 1)

    def test_batch_hook_contract_violation_always_raises(self) -> None:
        """A wrong number of outputs is a contract error, not a run failure."""

        class Broken(VectorisedLine):
            """Returns one output too few."""

            def evaluate_batch(self, points):
                """Drop the last output."""
                return super().evaluate_batch(points)[:-1]

        with self.assertRaisesRegex(ValueError, "outputs for"):
            uniform_model(Broken()).evaluate_many(self.POINTS, on_failure="skip")

    def test_cache_hits_are_not_evaluations(self) -> None:
        """Repeated points are served from the cache and reported as hits."""
        model = uniform_model()
        cache = EvaluationCache()
        points = ((0.1,), (0.2,), (0.1,), (0.3,))
        first = model.evaluate_many(points, cache=cache)
        self.assertEqual(first["y"], (0.1, 0.2, 0.1, 0.3))
        self.assertEqual(first.model_evaluations, 3)
        self.assertEqual(first.cache_hits, 1)
        second = model.evaluate_many(points, cache=cache)
        self.assertEqual(second, first)
        self.assertEqual(second.model_evaluations, 0)
        self.assertEqual(second.cache_hits, 4)
        self.assertEqual(model.evaluations, 3)
        self.assertEqual(model.cache_hits, 5)
        self.assertEqual((cache.hits, cache.misses), (5, 3))
        with model.evaluation_context(cache=cache):
            self.assertEqual(model.scalar_function("y")((0.2,)), 0.2)
        self.assertEqual(model.evaluations, 3)
        self.assertEqual(len(cache), 3)

    def test_failed_runs_are_not_cached(self) -> None:
        """A failed run is attempted again on the next request."""
        model = uniform_model()
        cache = EvaluationCache()
        model.evaluate_many([(0.99,)], cache=cache, on_failure="skip")
        model.evaluate_many([(0.99,)], cache=cache, on_failure="skip")
        self.assertEqual(model.failed_evaluations, 2)
        self.assertEqual(len(cache), 0)


class EvaluationLimitTests(unittest.TestCase):
    """Counter limits stop evaluation before the simulator runs."""

    def test_known_cost_is_refused_up_front(self) -> None:
        """A batch above the limit runs nothing."""
        model = uniform_model()
        with model.counter.limited(5):
            with self.assertRaises(EvaluationBudgetExceeded):
                model.evaluate_many([(0.1,)] * 6)
            self.assertEqual(model.evaluations, 0)
            model.evaluate_many([(0.1,)] * 5)
            with self.assertRaises(EvaluationBudgetExceeded):
                model.evaluate((0.1,))
        self.assertIsNone(model.counter.limit)
        model.evaluate((0.1,))
        self.assertEqual(model.evaluations, 6)

    def test_limit_binds_derived_models_and_direct_counter_use(self) -> None:
        """The parent limit applies to replace()d models and counter.add()."""
        model = uniform_model()
        derived = dataclasses.replace(model)
        with model.counter.limited(2):
            derived.evaluate((0.1,))
            derived.counter.add()
            with self.assertRaises(EvaluationBudgetExceeded):
                derived.evaluate((0.2,))
            with self.assertRaises(EvaluationBudgetExceeded):
                model.counter.add()
        self.assertEqual(model.evaluations, 2)
        self.assertEqual(model.counter.remaining(), None)


class StrEnumTests(unittest.TestCase):
    """0.18: StrEnum keeps str(), format(), and value text identical."""

    def test_text_of_every_member_is_its_value(self) -> None:
        """Reports interpolate members directly; the text must be the value."""
        for enum in (
            UncertaintyNature,
            UncertaintySourceKind,
            ConvergenceType,
            WorkflowStep,
            StepStatus,
            ProblemDirection,
        ):
            for member in enum:
                self.assertEqual(str(member), member.value)
                self.assertEqual(f"{member}", member.value)
                self.assertEqual(format(member, ""), member.value)
                self.assertEqual(member, member.value)
                self.assertIsInstance(member, str)


if __name__ == "__main__":
    unittest.main()
