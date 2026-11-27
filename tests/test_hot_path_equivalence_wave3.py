"""Bit-for-bit equivalence of the third wave of performance rewrites.

Covered: the Sobol pick-freeze driver of ``probstat.sensitivity.analysis``
(matrix validation, model evaluation, the Saltelli 2010 / alternative
estimator loops), the fast paths of ``require_non_negative_integer``,
``require_probability`` and ``coerce_finite_sample``, ``select_outputs`` of
``probstat.uq.taxonomy``, the one-sided Jacobi sweep of
``probstat.linalg.svd`` and the ``_axpy`` update of the LSQR driver.

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(the last commit before the rewrites), copied verbatim apart from removed
docstrings and the ``_ref_`` prefix. Every test feeds fixed-seed and
adversarial inputs (NaN, inf, ``-0.0``, ``bool``, ``int``, ``float``
subclasses, wrong lengths, ragged rows) to the current and to the reference
routine and compares floats by ``float.hex``, and exceptions by type and
message; model-call order is compared too. The golden digests at the end were
produced by that revision for whole workloads (Sobol indices, SVD, LSQR).

References:
    [REF-PROBSTAT-VALIDATION]. Equivalence harness only; no equation is
    evaluated.

"""

# ruff: noqa: B023 - every lambda below is called immediately inside its loop
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import math
import operator
import random
import statistics
import unittest
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from numbers import Real
from types import MappingProxyType
from typing import Any

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import (
    Convolution1DOperator,
    DenseOperator,
    lsqr,
    singular_value_decomposition,
)
from probstat.modeling.benchmarks import ishigami
from probstat.sensitivity import analysis as analysis_module
from probstat.sensitivity import estimate_sobol_indices
from probstat.sensitivity._sobol_estimators import (
    SOBOL_ESTIMATORS,
    alternative_indices,
    check_estimator,
    closed_index,
)
from probstat.sensitivity.analysis import SobolResult
from probstat.uq import QuantityOfInterest, select_outputs
from probstat.uq.evaluation import FrozenMapping

krylov_module = importlib.import_module("probstat.linalg.krylov")
svd_module = importlib.import_module("probstat.linalg.svd")

# ---------------------------------------------------------------- references


def _ref_estimate_sobol_indices(
    model: Callable[[tuple[float, ...]], float],
    sample_a: Iterable[Sequence[object]],
    sample_b: Iterable[Sequence[object]],
    *,
    estimator: str = "saltelli2010",
) -> SobolResult:
    if not callable(model):
        raise TypeError("model must be callable")
    estimator = check_estimator(estimator)
    a = _ref_matrix(sample_a, "sample_a")
    b = _ref_matrix(sample_b, "sample_b")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("sample matrices must have equal shape")
    if len(a) < 2:
        raise ValueError("at least two rows are required")
    n, dimension = len(a), len(a[0])
    ya = tuple(_ref_evaluate(model, row) for row in a)
    yb = tuple(_ref_evaluate(model, row) for row in b)
    if estimator != "saltelli2010":
        return _ref_alternative_sobol(model, a, b, ya, yb, estimator)
    variance = statistics.pvariance(ya + yb)
    # Centre f_B before the Saltelli product: adding a constant c to the
    # model adds c * mean(f_ABi - f_A) to the raw estimator, whose variance
    # grows like c^2; centring removes that dependence on the output mean.
    centre = statistics.fmean(ya + yb)
    if variance == 0.0:
        raise ValueError("Sobol indices require positive output variance")
    first: list[float] = []
    total: list[float] = []
    for i in range(dimension):
        values = tuple(
            _ref_evaluate(model, a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a, b, strict=True)
        )
        first.append(
            math.fsum((yb[k] - centre) * (values[k] - ya[k]) for k in range(n))
            / n
            / variance
        )
        total.append(
            0.5 * math.fsum((ya[k] - values[k]) ** 2 for k in range(n)) / n / variance
        )
    second = [[0.0] * dimension for _ in range(dimension)]
    pair_evaluations = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            values = tuple(
                _ref_evaluate(
                    model,
                    tuple(
                        b_row[k] if k in (i, j) else a_row[k] for k in range(dimension)
                    ),
                )
                for a_row, b_row in zip(a, b, strict=True)
            )
            closed = (
                math.fsum((yb[k] - centre) * (values[k] - ya[k]) for k in range(n))
                / n
                / variance
            )
            second[i][j] = second[j][i] = closed - first[i] - first[j]
            pair_evaluations += n
    evaluations = n * (2 + dimension) + pair_evaluations
    return SobolResult(
        tuple(first),
        tuple(total),
        tuple(tuple(row) for row in second),
        variance,
        n,
        evaluations,
    )


def _ref_alternative_sobol(
    model: Callable[[tuple[float, ...]], float],
    a: tuple[tuple[float, ...], ...],
    b: tuple[tuple[float, ...], ...],
    ya: tuple[float, ...],
    yb: tuple[float, ...],
    estimator: str,
) -> SobolResult:
    n, dimension = len(a), len(a[0])
    yab = [
        tuple(
            _ref_evaluate(model, a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a, b, strict=True)
        )
        for i in range(dimension)
    ]
    first, total, variance = alternative_indices(estimator, ya, yb, yab)
    second = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            values = tuple(
                _ref_evaluate(
                    model,
                    tuple(
                        b_row[k] if k in (i, j) else a_row[k] for k in range(dimension)
                    ),
                )
                for a_row, b_row in zip(a, b, strict=True)
            )
            closed = closed_index(estimator, ya, yb, values, variance)
            second[i][j] = second[j][i] = closed - first[i] - first[j]
    pairs = dimension * (dimension - 1) // 2
    return SobolResult(
        tuple(first),
        tuple(total),
        tuple(tuple(row) for row in second),
        variance,
        n,
        n * (2 + dimension) + n * pairs,
    )


def _ref_matrix(
    rows: Iterable[Sequence[object]], name: str
) -> tuple[tuple[float, ...], ...]:
    result = tuple(
        tuple(
            require_finite_real(v, name=f"{name}[{i}][{j}]") for j, v in enumerate(row)
        )
        for i, row in enumerate(rows)
    )
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return result


def _ref_evaluate(
    model: Callable[[tuple[float, ...]], float], row: tuple[float, ...]
) -> float:
    return require_finite_real(model(row), name="model output")


def _ref_require_non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _ref_require_probability(value: object, *, name: str = "p") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _ref_coerce_finite_sample(
    values: Iterable[object], *, name: str = "values"
) -> tuple[float, ...]:
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        observation = float(value)
        if not math.isfinite(observation):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(observation)
    if not result:
        raise ValueError(f"{name} must contain at least one observation")
    return tuple(result)


def _ref_select_outputs(
    outputs: object, quantities: Sequence[QuantityOfInterest]
) -> Mapping[str, float]:
    if type(outputs) is not dict and not isinstance(outputs, Mapping):
        raise TypeError("model output must be a mapping")
    selected: dict[str, float] = {}
    for quantity in quantities:
        if quantity.name not in outputs:
            raise ValueError(f"model output is missing QoI {quantity.name!r}")
        selected[quantity.name] = require_finite_real(
            outputs[quantity.name], name=f"output[{quantity.name!r}]"
        )
    return FrozenMapping(selected)


def _ref_dot(left: list[float], right: list[float]) -> float:
    return math.fsum(map(operator.mul, left, right))


def _ref_one_sided_jacobi(
    columns: list[list[float]], tolerance: float, max_sweeps: int
) -> tuple[list[list[float]], int]:
    size = len(columns)
    v = [[1.0 if i == j else 0.0 for i in range(size)] for j in range(size)]
    norms = [_ref_dot(column, column) for column in columns]
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for p in range(size - 1):
            for q in range(p + 1, size):
                alpha, beta = norms[p], norms[q]
                if alpha == 0.0 or beta == 0.0:
                    continue
                gamma = _ref_dot(columns[p], columns[q])
                if abs(gamma) <= tolerance * math.sqrt(alpha) * math.sqrt(beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.hypot(1.0, zeta))
                c = 1.0 / math.hypot(1.0, t)
                s = c * t
                cp, cq = columns[p], columns[q]
                columns[p] = [c * a - s * b for a, b in zip(cp, cq, strict=True)]
                columns[q] = [s * a + c * b for a, b in zip(cp, cq, strict=True)]
                vp, vq = v[p], v[q]
                v[p] = [c * a - s * b for a, b in zip(vp, vq, strict=True)]
                v[q] = [s * a + c * b for a, b in zip(vp, vq, strict=True)]
                norms[p] = _ref_dot(columns[p], columns[p])
                norms[q] = _ref_dot(columns[q], columns[q])
        if not rotated:
            return v, sweep
    raise ValueError("one-sided Jacobi SVD did not converge")


def _ref_axpy(alpha: float, x: Sequence[float], y: Sequence[float]) -> list[float]:
    return [alpha * a + b for a, b in zip(x, y, strict=True)]


# ------------------------------------------------------------------- helpers


def _hex(value: Any) -> Any:
    """Return a structure of exact bit patterns for floats (recursively)."""
    if isinstance(value, float):
        return (type(value).__name__, value.hex())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, Mapping):
        return {k: _hex(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_hex(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(_hex(getattr(value, f.name)) for f in dataclasses.fields(value))
    return value


def _outcome(call: Callable[[], Any]) -> tuple[Any, ...]:
    """Return ``("ok", bits)`` or ``("err", type name, message)``."""
    try:
        return ("ok", _hex(call()))
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return ("err", type(error).__name__, str(error))


def _flatten(obj: Any, out: list[str]) -> None:
    """Append a canonical text of ``obj`` (floats by ``hex``) to ``out``."""
    if isinstance(obj, float):
        out.append(obj.hex())
    elif isinstance(obj, (bool, int, str)) or obj is None:
        out.append(repr(obj))
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for item in dataclasses.fields(obj):
            _flatten(getattr(obj, item.name), out)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(key, out)
            _flatten(value, out)
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for value in obj:
            _flatten(value, out)
        out.append("]")
    else:
        out.append(f"<{type(obj).__name__}>")


def _digest(obj: Any) -> str:
    out: list[str] = []
    _flatten(obj, out)
    return hashlib.sha256("|".join(out).encode()).hexdigest()[:16]


class _Odd(float):
    """A float subclass: must take the validator path, not the fast path."""


class _OddInt(int):
    """An int subclass: must take the validator path, not the fast path."""


NAN = float("nan")
INF = float("inf")

#: Scalars that probe every branch of the real-number validators.
ADVERSARIAL_SCALARS: tuple[Any, ...] = (
    0.0,
    -0.0,
    1.0,
    0.5,
    1e-300,
    5e-324,
    1.0 - 2.0**-53,
    1.0 + 2.0**-52,
    -1e-12,
    1.5,
    -2.0,
    1e308,
    NAN,
    INF,
    -INF,
    0,
    1,
    2,
    -1,
    10**400,
    -(10**400),
    True,
    False,
    _Odd(0.25),
    _Odd(NAN),
    _Odd(INF),
    _Odd(-0.0),
    _OddInt(1),
    _OddInt(-3),
    Fraction(1, 3),
    Fraction(3, 2),
    "0.5",
    None,
    1j,
    [0.5],
    (0.5,),
    b"1",
)


def _ishigami_model(row: Sequence[float]) -> float:
    return ishigami(*row)


# --------------------------------------------------------------------- tests


class ValidatorFastPathTests(unittest.TestCase):
    """The scalar validators keep results and messages."""

    def test_require_probability(self) -> None:
        """Compare against the original on adversarial scalars."""
        for value in ADVERSARIAL_SCALARS:
            for kwargs in ({}, {"name": "level"}):
                with self.subTest(value=repr(value), kwargs=kwargs):
                    self.assertEqual(
                        _outcome(lambda: require_probability(value, **kwargs)),
                        _outcome(lambda: _ref_require_probability(value, **kwargs)),
                    )

    def test_require_non_negative_integer(self) -> None:
        """Compare against the original on adversarial scalars."""
        for value in ADVERSARIAL_SCALARS:
            with self.subTest(value=repr(value)):
                self.assertEqual(
                    _outcome(lambda: require_non_negative_integer(value, name="n")),
                    _outcome(
                        lambda: _ref_require_non_negative_integer(value, name="n")
                    ),
                )

    def test_require_finite_real_unchanged(self) -> None:
        """Pin the behaviour of the validator the fast paths mimic."""
        results = [
            _outcome(lambda: require_finite_real(v, name="x"))
            for v in ADVERSARIAL_SCALARS
        ]
        self.assertEqual(results[0], ("ok", ("float", (0.0).hex())))
        self.assertEqual(results[12], ("err", "ValueError", "x must be finite"))

    def test_coerce_finite_sample(self) -> None:
        """Compare whole samples and every single-bad-element sample."""
        rng = random.Random(5)
        samples: list[Any] = [
            [rng.gauss(0.0, 3.0) for _ in range(40)],
            (1.0, 2.0, -0.0, 5e-324),
            [],
            (),
            [1.0, True, 2.0],
        ]
        for bad in ADVERSARIAL_SCALARS:
            samples.append([1.0, bad, 2.0])
            samples.append([bad])
        for index, sample in enumerate(samples):
            for kwargs in ({}, {"name": "obs"}):
                with self.subTest(index=index, kwargs=kwargs):
                    self.assertEqual(
                        _outcome(lambda: coerce_finite_sample(sample, **kwargs)),
                        _outcome(lambda: _ref_coerce_finite_sample(sample, **kwargs)),
                    )
        # one-shot iterators and generators
        for make in (
            lambda: iter([1.0, 2.0, 3.5]),
            lambda: (x for x in (0.5, 1, 2)),
            lambda: iter([1.0, NAN]),
        ):
            self.assertEqual(
                _outcome(lambda: coerce_finite_sample(make())),
                _outcome(lambda: _ref_coerce_finite_sample(make())),
            )


class SelectOutputsTests(unittest.TestCase):
    """``select_outputs`` keeps results, messages and the access order."""

    def test_adversarial_outputs(self) -> None:
        """Compare mappings, missing names and every adversarial value."""
        quantities = (
            QuantityOfInterest("y", "response"),
            QuantityOfInterest("z", "load"),
        )
        outputs: list[Any] = [
            {"y": 1.0, "z": 2.5},
            {"z": -0.0, "y": 3.0, "extra": NAN},
            MappingProxyType({"y": 1.0, "z": 2.0}),
            FrozenMapping({"y": 1.0, "z": 2.0}),
            {"y": 1.0},
            {},
            [("y", 1.0)],
            None,
            3.0,
            "y",
        ]
        for bad in ADVERSARIAL_SCALARS:
            outputs.append({"y": 1.0, "z": bad})
            outputs.append({"y": bad, "z": 1.0})
        for index, payload in enumerate(outputs):
            with self.subTest(index=index):
                self.assertEqual(
                    _outcome(lambda: select_outputs(payload, quantities)),
                    _outcome(lambda: _ref_select_outputs(payload, quantities)),
                )

    def test_access_order(self) -> None:
        """The mapping sees the same ``in`` and ``[]`` calls in the same order."""
        quantities = (
            QuantityOfInterest("y", "response"),
            QuantityOfInterest("z", "load"),
        )

        class Spy(dict):  # type: ignore[type-arg]
            """A dict subclass that logs its accesses."""

            def __init__(self, data: dict[str, Any]) -> None:
                super().__init__(data)
                self.log: list[str] = []

            def __contains__(self, key: object) -> bool:
                self.log.append(f"in:{key}")
                return super().__contains__(key)

            def __getitem__(self, key: Any) -> Any:
                self.log.append(f"get:{key}")
                return super().__getitem__(key)

        for data in ({"y": 1.0, "z": 2.0}, {"y": 1.0}, {"y": 1.0, "z": NAN}):
            logs = []
            for function in (select_outputs, _ref_select_outputs):
                spy = Spy(data)
                outcome = _outcome(lambda: function(spy, quantities))
                logs.append((outcome, spy.log))
            self.assertEqual(logs[0], logs[1])


def _fresh(value: Any) -> Any:
    """Call a factory (one-shot iterators need a new object per run)."""
    return value() if callable(value) else value


def _model_log() -> tuple[list[Any], Callable[[Callable[..., Any]], Any]]:
    """Return a call log and a decorator that records every model input."""
    log: list[Any] = []

    def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
        def model(row: Any) -> Any:
            log.append(_hex(row))
            return function(row)

        return model

    return log, wrap


def _design(n: int, d: int, seed: int) -> list[tuple[float, ...]]:
    rng = random.Random(seed)
    return [tuple(rng.uniform(-math.pi, math.pi) for _ in range(d)) for _ in range(n)]


def _linear(row: Sequence[float]) -> float:
    return 1.0 + 2.0 * row[0] - 0.5 * row[1] + row[0] * row[-1] * 0.1


class SobolEquivalenceTests(unittest.TestCase):
    """The Sobol pick-freeze driver is bit-identical to the original."""

    def _same(
        self,
        function: Callable[[Any], Any],
        a: Any,
        b: Any,
        estimator: str = "saltelli2010",
        label: object = "",
    ) -> None:
        """Assert equal outcomes and model-call logs."""
        log_new, wrap_new = _model_log()
        log_ref, wrap_ref = _model_log()
        new = _outcome(
            lambda: estimate_sobol_indices(
                wrap_new(function), _fresh(a), _fresh(b), estimator=estimator
            )
        )
        ref = _outcome(
            lambda: _ref_estimate_sobol_indices(
                wrap_ref(function), _fresh(a), _fresh(b), estimator=estimator
            )
        )
        self.assertEqual(new, ref, label)
        self.assertEqual(log_new, log_ref, label)

    def test_fixed_seed_all_estimators(self) -> None:
        """Ishigami and a linear model, 3 and 4 inputs, every estimator."""
        for estimator in SOBOL_ESTIMATORS:
            for dimension, model in ((3, _ishigami_model), (4, _linear)):
                a = _design(64, dimension, 1)
                b = _design(64, dimension, 2)
                with self.subTest(estimator=estimator, dimension=dimension):
                    self._same(model, a, b, estimator)

    def test_matrix_containers(self) -> None:
        """Lists, generators and mixed row types are accepted alike."""
        a = _design(12, 3, 3)
        b = _design(12, 3, 4)
        for estimator in ("saltelli2010", "jansen"):
            model = _ishigami_model
            self._same(model, [list(r) for r in a], [list(r) for r in b], estimator)
            self._same(model, tuple(a), tuple(b), estimator)
            self._same(model, lambda: iter(a), lambda: iter(b), estimator)
            self._same(
                model,
                lambda: (iter(r) for r in a),
                lambda: (iter(r) for r in b),
                estimator,
            )
            mixed = [list(r) if i % 2 else r for i, r in enumerate(a)]
            self._same(model, mixed, b, estimator)

    def test_adversarial_entries(self) -> None:
        """Every adversarial scalar at several positions of either matrix."""
        base_a = _design(6, 3, 5)
        base_b = _design(6, 3, 6)
        for bad in ADVERSARIAL_SCALARS:
            for row, col in ((0, 0), (2, 1), (5, 2)):
                for target in ("a", "b"):
                    a = [list(r) for r in base_a]
                    b = [list(r) for r in base_b]
                    (a if target == "a" else b)[row][col] = bad
                    self._same(
                        _ishigami_model, a, b, label=(repr(bad), row, col, target)
                    )
        # -0.0, integers and a float subclass in a valid design keep exact results
        a = [list(r) for r in base_a]
        a[1][1] = -0.0
        a[2][0] = 1
        a[3][2] = _Odd(0.5)
        self._same(_ishigami_model, a, base_b)
        self._same(_ishigami_model, a, base_b, "jansen")

    def test_shapes_and_degenerate_inputs(self) -> None:
        """Ragged, empty, single-row and mismatched matrices."""
        a = _design(5, 3, 7)
        b = _design(5, 3, 8)
        cases: list[tuple[Any, Any]] = [
            (a[:1], b[:1]),
            (a, b[:4]),
            (a, [r[:2] for r in b]),
            ([a[0], a[1][:2], *a[2:]], b),
            (a, [b[0][:2], *b[1:]]),
            ([], b),
            (a, []),
            ([[]], [[]]),
            ([(), ()], [(), ()]),
            (3, b),
            ([1.0, 2.0], b),
            ([a[0], 5, *a[2:]], b),
            (a, "abc"),
            (None, None),
        ]
        for index, (left, right) in enumerate(cases):
            for estimator in ("saltelli2010", "sobol2007"):
                self._same(_ishigami_model, left, right, estimator, (index, estimator))
        self._same(_ishigami_model, a, b, "no-such-estimator")
        self._same(_ishigami_model, a, b, 3)  # type: ignore[arg-type]
        self.assertEqual(
            _outcome(lambda: estimate_sobol_indices(3, a, b)),  # type: ignore[arg-type]
            _outcome(lambda: _ref_estimate_sobol_indices(3, a, b)),  # type: ignore[arg-type]
        )

    def test_model_outputs(self) -> None:
        """Every adversarial model response, at several call positions."""
        a = _design(6, 3, 9)
        b = _design(6, 3, 10)
        for bad in ADVERSARIAL_SCALARS:
            for position in (0, 4, 9, 20, 30):
                for estimator in ("saltelli2010", "janon"):
                    outcomes = []
                    for function in (
                        estimate_sobol_indices,
                        _ref_estimate_sobol_indices,
                    ):
                        state = {"n": 0}

                        def model(row: Any, bad: Any = bad, state: Any = state) -> Any:
                            index = state["n"]
                            state["n"] += 1
                            return bad if index == position else _ishigami_model(row)

                        log, wrap = _model_log()
                        outcome = _outcome(
                            lambda: function(wrap(model), a, b, estimator=estimator)
                        )
                        outcomes.append((outcome, log))
                    self.assertEqual(
                        outcomes[0], outcomes[1], (repr(bad), position, estimator)
                    )

    def test_constant_and_failing_models(self) -> None:
        """Zero variance, integer outputs, overflow and a raising model."""
        a = _design(8, 3, 11)
        b = _design(8, 3, 12)
        self._same(lambda x: 2.0, a, b)
        self._same(lambda x: 2.0, a, b, "jansen")
        self._same(lambda x: 3, a, b, "martinez")
        self._same(lambda x: int(x[0] > 0), a, b)
        self._same(lambda x: int(x[0] > 0), a, b, "janon")
        self._same(lambda x: x[0] + 1e300 * x[1], a, b)
        self._same(lambda x: x[0] * 1e200 * x[1] * 1e200, a, b)
        self._same(lambda x: (x[0] * 1e160) + x[1], a, b)

        def failing(row: Any) -> float:
            if row[0] > 2.5:
                raise RuntimeError("boom")
            return float(row[0])

        self._same(failing, a, b)
        self._same(failing, a, b, "jansen")

    def test_matrix_and_evaluate_helpers(self) -> None:
        """``_matrix`` and ``_evaluate`` agree on special row containers."""
        rows_cases: list[Any] = [
            [[1.0, 2.0], [3.0, 4.0]],
            [(1.0, 2.0), [3.0, 4.0]],
            [[1.0, 2.0], [3.0, INF]],
            [[1.0, NAN]],
            [[1, 2], [3, 4]],
            [[1.0, 2.0], [3.0]],
            [[True, 1.0]],
            [[_Odd(1.0), 2.0]],
            [[Fraction(1, 2), 2.0]],
            [],
            [[]],
            lambda: [[1.0, 2.0], iter([3.0, 4.0])],
            [[1.0, 2.0], "ab"],
            [[1.0, 2.0], 7],
            [[-0.0, 0.0]],
        ]
        for index, rows in enumerate(rows_cases):
            with self.subTest(index=index):
                self.assertEqual(
                    _outcome(lambda: analysis_module._matrix(_fresh(rows), "m")),
                    _outcome(lambda: _ref_matrix(_fresh(rows), "m")),
                )
        for value in ADVERSARIAL_SCALARS:
            with self.subTest(value=repr(value)):
                self.assertEqual(
                    _outcome(
                        lambda: analysis_module._evaluate(lambda r: value, (1.0,))
                    ),
                    _outcome(lambda: _ref_evaluate(lambda r: value, (1.0,))),
                )

    def test_alternative_helper_directly(self) -> None:
        """The alternative-estimator driver agrees when called on its own."""
        a = tuple(_design(20, 3, 13))
        b = tuple(_design(20, 3, 14))
        ya = tuple(_ishigami_model(r) for r in a)
        yb = tuple(_ishigami_model(r) for r in b)
        for estimator in SOBOL_ESTIMATORS[1:]:
            self.assertEqual(
                _outcome(
                    lambda: analysis_module._alternative_sobol(
                        _ishigami_model, a, b, ya, yb, estimator
                    )
                ),
                _outcome(
                    lambda: _ref_alternative_sobol(
                        _ishigami_model, a, b, ya, yb, estimator
                    )
                ),
            )


def _jacobi_columns(rows: int, cols: int, seed: int, kind: str) -> list[list[float]]:
    """Return ``cols`` columns of length ``rows`` of the named kind."""
    rng = random.Random(seed)
    out = [[rng.gauss(0.0, 1.0) for _ in range(rows)] for _ in range(cols)]
    if kind == "scaled":
        out = [
            [v * 10.0 ** (j - cols // 2) * 1e3 for v in c] for j, c in enumerate(out)
        ]
    elif kind == "dependent" and cols > 2:
        out[-1] = [x + y for x, y in zip(out[0], out[1], strict=True)]
    elif kind == "duplicate" and cols > 1:
        out[1] = list(out[0])
    elif kind == "zero" and cols > 1:
        out[1] = [0.0] * rows
    elif kind == "negzero":
        out[0] = [-0.0] * rows
    elif kind == "tiny":
        out = [[v * 1e-170 for v in c] for c in out]
    elif kind == "huge":
        out = [[v * 1e150 for v in c] for c in out]
    elif kind == "ints":
        out = [[float(round(v * 3)) for v in c] for c in out]
    return out


def _svd_matrices() -> list[Any]:
    """Return the fixed and adversarial matrices of the SVD digests."""
    rng = random.Random(21)

    def gauss(m: int, n: int) -> list[list[float]]:
        return [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(m)]

    rank_one = [[float(i + 1) * float(j + 2) for j in range(4)] for i in range(6)]
    return [
        gauss(7, 4),
        gauss(4, 7),
        gauss(5, 5),
        gauss(1, 3),
        gauss(3, 1),
        gauss(40, 25),
        rank_one,
        [[0.0] * 3 for _ in range(4)],
        [[-0.0, 1.0], [2.0, 3.0]],
        [[1, 2], [3, 4], [5, 6]],
        [[1.0, NAN], [1.0, 2.0]],
        [[1.0, INF], [1.0, 2.0]],
        [[1.0, 2.0], [3.0]],
        [],
        [[]],
    ]


class JacobiEquivalenceTests(unittest.TestCase):
    """The one-sided Jacobi sweep is bit-identical to the original."""

    def test_direct_calls(self) -> None:
        """Compare ``V``, the sweep count and the in-place columns."""
        kinds = (
            "gauss", "scaled", "dependent", "duplicate", "zero", "negzero",
            "tiny", "huge", "ints",
        )  # fmt: skip
        for rows, cols in ((2, 2), (5, 3), (8, 8), (30, 12), (12, 12), (3, 1)):
            for kind in kinds:
                for tolerance in (1e-15, 1e-8):
                    for sweeps in (60, 1):
                        new_cols = _jacobi_columns(rows, cols, rows * 131 + cols, kind)
                        ref_cols = [list(c) for c in new_cols]
                        new = _outcome(
                            lambda: svd_module._one_sided_jacobi(
                                new_cols, tolerance, sweeps
                            )
                        )
                        ref = _outcome(
                            lambda: _ref_one_sided_jacobi(ref_cols, tolerance, sweeps)
                        )
                        label = (rows, cols, kind, tolerance, sweeps)
                        self.assertEqual(new, ref, label)
                        self.assertEqual(_hex(new_cols), _hex(ref_cols), label)

    def test_public_svd_digest(self) -> None:
        """Whole SVDs (and error texts) equal the digest of the original."""
        self.assertEqual(_golden_svd(), GOLDEN["svd"])


class AxpyEquivalenceTests(unittest.TestCase):
    """The LSQR vector update keeps results and length errors."""

    def test_axpy(self) -> None:
        """Compare on random, special and mismatched vectors."""
        rng = random.Random(33)
        scalars = (0.0, -0.0, 1.0, -1.0, 0.3, 1e300, NAN, INF, 5e-324)
        vectors: list[Any] = [
            [rng.gauss(0.0, 1.0) for _ in range(17)],
            tuple(rng.gauss(0.0, 1.0) for _ in range(17)),
            [-0.0, 0.0, INF, -INF, NAN, 1e308, 5e-324],
            [],
            [1.0],
            [1.0, 2.0],
        ]
        for alpha in scalars:
            for x, y in itertools.product(vectors, vectors):
                with self.subTest(alpha=alpha, nx=len(x), ny=len(y)):
                    self.assertEqual(
                        _outcome(lambda: krylov_module._axpy(alpha, x, y)),
                        _outcome(lambda: _ref_axpy(alpha, x, y)),
                    )

    def test_lsqr_end_to_end(self) -> None:
        """Whole LSQR runs equal the recorded digests of the original."""
        self.assertEqual(_golden_lsqr(), GOLDEN["lsqr"])


# --------------------------------------------------------------------- golden


def _golden_svd() -> str:
    """Return the digest of the SVDs (or error texts) of the fixed matrices."""
    return _digest(
        [_outcome(lambda m=m: singular_value_decomposition(m)) for m in _svd_matrices()]
    )


def _golden_lsqr() -> dict[str, str]:
    """Digest several LSQR runs: plain, damped, warm start, callback, limits."""
    size = 300
    conv = Convolution1DOperator((0.05, 0.2, 0.5, 0.2, 0.05), size)
    truth = tuple(math.sin(2.0 * math.pi * i / 90.0) for i in range(size))
    data = tuple(conv.matvec(truth))
    rng = random.Random(8)
    dense = DenseOperator([[rng.gauss(0.0, 1.0) for _ in range(9)] for _ in range(14)])
    rhs = tuple(rng.gauss(0.0, 1.0) for _ in range(14))
    seen: list[tuple[int, str, str]] = []

    def watch(k: int, x: Sequence[float], rnorm: float) -> bool:
        seen.append((k, x[0].hex(), rnorm.hex()))
        return k >= 4

    runs: dict[str, Callable[[], Any]] = {
        "conv_plain": lambda: lsqr(conv, data, max_iterations=40),
        "conv_damped": lambda: lsqr(conv, data, damping=0.1, max_iterations=25),
        "conv_warm": lambda: lsqr(
            conv, data, initial=tuple(0.5 * v for v in truth), max_iterations=15
        ),
        "dense_full": lambda: lsqr(dense, rhs),
        "dense_loose": lambda: lsqr(dense, rhs, atol=1e-3, btol=1e-3),
        "dense_cond": lambda: lsqr(dense, rhs, condition_limit=2.0),
        "dense_callback": lambda: (lsqr(dense, rhs, callback=watch), list(seen)),
        "dense_zero": lambda: lsqr(dense, tuple(0.0 for _ in rhs)),
        "dense_one": lambda: lsqr(dense, rhs, max_iterations=1),
        "dense_bad": lambda: lsqr(dense, rhs[:-1]),
    }
    return {name: _digest(_outcome(call)) for name, call in runs.items()}


def _golden_sobol() -> dict[str, str]:
    """Digest whole Sobol studies, including the second-order block."""
    out: dict[str, str] = {}
    a = _design(256, 3, 41)
    b = _design(256, 3, 42)
    for estimator in SOBOL_ESTIMATORS:
        out[estimator] = _digest(
            _outcome(
                lambda e=estimator: estimate_sobol_indices(
                    _ishigami_model, a, b, estimator=e
                )
            )
        )
    return out


GOLDEN: dict[str, Any] = {
    "svd": "d76bd5afabef6a92",
    "lsqr": {
        "conv_plain": "9f36adf6de07744b",
        "conv_damped": "59067c3de4bf55dc",
        "conv_warm": "81a7b461214411a9",
        "dense_full": "7929e720153667f6",
        "dense_loose": "7929e720153667f6",
        "dense_cond": "c365a42cf629a014",
        "dense_callback": "ab44739c8b771f03",
        "dense_zero": "5aaa9f26e8bfcf78",
        "dense_one": "f4c7ae8cc608cace",
        "dense_bad": "db39a09a628918b9",
    },
    "sobol": {
        "saltelli2010": "7c0f54be0a330329",
        "saltelli2002": "3493bda7404a4aac",
        "sobol2007": "a94c3d9cb6119b91",
        "jansen": "0cd7cc4986fee919",
        "martinez": "19a34d2d696a0b01",
        "janon": "9dea19e4080cbd8d",
    },
}


class GoldenTests(unittest.TestCase):
    """Whole-workload digests recorded from revision ``315d501``."""

    def test_sobol_studies(self) -> None:
        """Sobol studies of every estimator reproduce the recorded digests."""
        self.assertEqual(_golden_sobol(), GOLDEN["sobol"])


if __name__ == "__main__":
    import sys

    if "--print-golden" in sys.argv:
        import json

        print(
            json.dumps(
                {
                    "svd": _golden_svd(),
                    "lsqr": _golden_lsqr(),
                    "sobol": _golden_sobol(),
                },
                indent=4,
            )
        )
    else:
        unittest.main()
