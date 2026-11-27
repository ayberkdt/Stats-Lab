"""Bit-for-bit equivalence of the fourth wave of performance rewrites.

Covered: the shifted-stencil fast path of ``Convolution1DOperator``, the
slice formulations of ``FirstDifferenceOperator`` and ``GridGradientOperator``,
``materialize``, the validation and the Thomas sweeps of
``TridiagonalFactorization`` (``probstat.inverse.pde_models``), the time
stepping of ``HeatEquation1D`` / ``HeatSource1D`` and the marching and adjoint
loops of ``SlabFluxOperator``.

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(``probstat.linalg.operators``, ``probstat.inverse.pde_models`` and
``probstat.inverse.heat_models``) and of the working-tree version of the
untracked ``probstat.inverse.heat_conduction_inverse`` that existed before this
wave, copied verbatim apart from removed docstrings and the ``_ref_`` prefix.
Every test feeds fixed-seed and adversarial inputs (NaN, inf, ``-0.0``,
``bool``, ``int`` and ``float`` subclasses, huge integers, wrong lengths,
ragged rows, overflowing sums) to the current and to the reference routine and
compares floats by ``float.hex`` and exceptions by type and message. The
golden digests at the end were produced by the reference code for whole
workloads.

References:
    [REF-PROBSTAT-VALIDATION]. Equivalence harness only; no equation is
    evaluated.

"""

# ruff: noqa: B023 - every lambda below is called immediately inside its loop
from __future__ import annotations

import collections
import hashlib
import itertools
import math
import random
import unittest
from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

from probstat.inverse import heat_conduction_inverse as slab_module
from probstat.inverse import pde_models
from probstat.inverse.heat_conduction_inverse import (
    SlabFluxOperator,
    synthetic_ablator_slab,
)
from probstat.inverse.heat_models import HeatEquation1D, HeatSource1D, _HeatOperator
from probstat.inverse.pde_models import (
    EllipticDiffusion1D,
    TridiagonalFactorization,
    _check_tridiagonal,
    _finite_vector,
)
from probstat.linalg import (
    Convolution1DOperator,
    DenseOperator,
    FirstDifferenceOperator,
    GridGradientOperator,
    OperatorBase,
    lsqr,
    materialize,
)
from probstat.linalg.operators import _apply, require_operator

# ---------------------------------------------------------------- references


def _ref_finite_vector(
    name: str, values: Sequence[float], length: int | None
) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        out.append(number)
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must have length {length}, got {len(out)}")
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


class _RefTridiagonal:
    def __init__(
        self,
        lower: Sequence[float],
        diagonal: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        n = _check_tridiagonal(lower, diagonal, upper)
        lo = _ref_finite_vector("lower", lower, n - 1) if n > 1 else []
        di = _ref_finite_vector("diagonal", diagonal, n)
        up = _ref_finite_vector("upper", upper, n - 1) if n > 1 else []
        pivots = [di[0]]
        multipliers: list[float] = []
        for i in range(n - 1):
            if pivots[i] == 0.0:
                raise ValueError(f"zero pivot at row {i}; matrix is singular")
            m = lo[i] / pivots[i]
            multipliers.append(m)
            pivots.append(di[i + 1] - m * up[i])
        if pivots[n - 1] == 0.0 or not all(math.isfinite(p) for p in pivots):
            raise ValueError(f"zero or non-finite pivot at row {n - 1}")
        self._pivots = pivots
        self._multipliers = multipliers
        self._upper = up
        self.order = n

    def solve(self, rhs: Sequence[float]) -> list[float]:
        n = self.order
        y = _ref_finite_vector("rhs", rhs, n)
        for i in range(n - 1):
            y[i + 1] -= self._multipliers[i] * y[i]
        y[n - 1] /= self._pivots[n - 1]
        for i in range(n - 2, -1, -1):
            y[i] = (y[i] - self._upper[i] * y[i + 1]) / self._pivots[i]
        return y

    def solve_transpose(self, rhs: Sequence[float]) -> list[float]:
        n = self.order
        z = _ref_finite_vector("rhs", rhs, n)
        z[0] /= self._pivots[0]
        for i in range(1, n):
            z[i] = (z[i] - self._upper[i - 1] * z[i - 1]) / self._pivots[i]
        for i in range(n - 2, -1, -1):
            z[i] -= self._multipliers[i] * z[i + 1]
        return z


def _ref_solve(fact: TridiagonalFactorization, rhs: Sequence[float]) -> list[float]:
    """Solve with the original sweeps on the stored factors of a factorization."""
    n = fact.order
    y = _ref_finite_vector("rhs", rhs, n)
    for i in range(n - 1):
        y[i + 1] -= fact._multipliers[i] * y[i]
    y[n - 1] /= fact._pivots[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = (y[i] - fact._upper[i] * y[i + 1]) / fact._pivots[i]
    return y


class _RefFirstDifference(OperatorBase):
    """Reference first difference sharing size, spacing and boundary."""

    __slots__ = ("boundary", "spacing")

    def __init__(self, other: FirstDifferenceOperator) -> None:
        self.spacing = other.spacing
        self.boundary = other.boundary
        rows, columns = other.shape
        super().__init__(rows, columns, name=other.name)

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        h = self.spacing
        n = len(vector)
        if self.boundary == "none":
            return tuple((vector[i + 1] - vector[i]) / h for i in range(n - 1))
        if self.boundary == "periodic":
            return tuple((vector[(i + 1) % n] - vector[i]) / h for i in range(n))
        padded = (0.0, *vector, 0.0)
        return tuple((padded[i + 1] - padded[i]) / h for i in range(n + 1))

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        h = self.spacing
        n = self._shape[1]
        result = [0.0] * n
        if self.boundary == "none":
            for i, value in enumerate(vector):
                result[i] -= value / h
                result[i + 1] += value / h
        elif self.boundary == "periodic":
            for i, value in enumerate(vector):
                result[i] -= value / h
                result[(i + 1) % n] += value / h
        else:
            for i, value in enumerate(vector):
                if i < n:
                    result[i] += value / h
                if i >= 1:
                    result[i - 1] -= value / h
        return result


class _RefGrid(OperatorBase):
    """Reference grid gradient sharing shape and spacing."""

    __slots__ = ("columns_count", "rows_count", "spacing")

    def __init__(self, other: GridGradientOperator) -> None:
        self.rows_count = other.rows_count
        self.columns_count = other.columns_count
        self.spacing = other.spacing
        rows, columns = other.shape
        super().__init__(rows, columns, name=other.name)

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        r, c = self.rows_count, self.columns_count
        hy, hx = self.spacing
        dx = [0.0] * (r * c)
        dy = [0.0] * (r * c)
        for i in range(r):
            base = i * c
            for j in range(c):
                k = base + j
                if j + 1 < c:
                    dx[k] = (vector[k + 1] - vector[k]) / hx
                if i + 1 < r:
                    dy[k] = (vector[k + c] - vector[k]) / hy
        return dx + dy

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        r, c = self.rows_count, self.columns_count
        hy, hx = self.spacing
        size = r * c
        result = [0.0] * size
        for i in range(r):
            base = i * c
            for j in range(c):
                k = base + j
                if j + 1 < c:
                    value = vector[k] / hx
                    result[k] -= value
                    result[k + 1] += value
                if i + 1 < r:
                    value = vector[size + k] / hy
                    result[k] -= value
                    result[k + c] += value
        return result


class _RefConvolution(OperatorBase):
    """Reference convolution: the original taps and loops."""

    __slots__ = ("boundary", "kernel", "_taps")

    def __init__(
        self, kernel: Sequence[float], size: int, *, boundary: str = "zero"
    ) -> None:
        weights = tuple(float(value) for value in kernel)
        self.kernel = weights
        self.boundary = boundary
        centre = len(weights) // 2
        taps: list[tuple[tuple[int, int, float], ...]] = []
        for i in range(size):
            row: list[tuple[int, int, float]] = []
            for offset, weight in enumerate(self.kernel):
                j = i + offset - centre
                mapped = self._map(j, size)
                if mapped is not None and weight != 0.0:
                    row.append((i, mapped, weight))
            taps.append(tuple(row))
        self._taps = tuple(taps)
        super().__init__(size, size, name=f"convolution ({boundary})")

    def _map(self, index: int, size: int) -> int | None:
        if 0 <= index < size:
            return index
        if self.boundary == "zero":
            return None
        if self.boundary == "periodic":
            return index % size
        period = 2 * size
        folded = index % period
        return folded if folded < size else period - 1 - folded

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        return tuple(
            math.fsum(weight * vector[j] for _, j, weight in row) for row in self._taps
        )

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        result = [0.0] * self._shape[1]
        for row in self._taps:
            for i, j, weight in row:
                result[j] += weight * vector[i]
        return result


def _ref_materialize(operator: Any) -> tuple[tuple[float, ...], ...]:
    checked = require_operator(operator)
    rows, columns = checked.shape
    if rows == 0 or columns == 0:
        raise ValueError("cannot materialize an operator with an empty dimension")
    if rows * columns > 1_000_000:
        raise ValueError(
            f"materializing {rows} x {columns} exceeds max_entries={1_000_000}"
        )
    columns_list = []
    for j in range(columns):
        unit = [0.0] * columns
        unit[j] = 1.0
        columns_list.append(_apply(checked, unit))
    return tuple(tuple(column[i] for column in columns_list) for i in range(rows))


def _ref_apply_b(op: _HeatOperator, u: list[float]) -> list[float]:
    if op._scheme_name == "implicit_euler":
        return list(u)
    n = op.cells
    k = 0.5 * op._dt * op._c
    out = [0.0] * n
    for i in range(n):
        left = u[i - 1] if i > 0 else 0.0
        right = u[i + 1] if i < n - 1 else 0.0
        out[i] = u[i] - k * (2.0 * u[i] - left - right)
    return out


def _ref_run_forward(
    op: _HeatOperator,
    initial: Sequence[float] | None,
    source: Sequence[float] | None,
    counts: list[int],
) -> list[float]:
    n = op.cells
    u = list(initial) if initial is not None else [0.0] * n
    out = [0.0] * op.shape[0]
    op._observe_add(0, u, out)
    for step in range(1, op._last_step + 1):
        rhs = _ref_apply_b(op, u)
        if source is not None:
            rhs = [r + op._dt * q for r, q in zip(rhs, source, strict=True)]
        u = _ref_solve(op._factorization, rhs)
        counts[2] += 1
        op._observe_add(step, u, out)
    counts[0] += 1
    return out


def _ref_run_transpose(
    op: _HeatOperator, y: Sequence[float], *, source: bool, counts: list[int]
) -> list[float]:
    n = op.cells
    a = [0.0] * n
    total = [0.0] * n
    stop = 1 if source else 0
    for step in range(op._last_step, stop - 1, -1):
        if step < op._last_step:
            a = _ref_apply_b(op, _ref_solve(op._factorization, a))
            counts[3] += 1
        op._observe_transpose_add(step, y, a)
        if source:
            total = [t + v for t, v in zip(total, a, strict=True)]
    counts[1] += 1
    if not source:
        return a
    if op._last_step == 0:
        return total
    solved = _ref_solve(op._factorization, total)
    counts[3] += 1
    return [op._dt * v for v in solved]


def _ref_march(
    op: SlabFluxOperator, flux: Sequence[float]
) -> Iterator[tuple[int, list[float]]]:
    theta = [0.0] * op.cells
    total = op._flux_intervals * op.substeps
    for k in range(1, total + 1):
        rhs = [op._heat_capacity_rate * v for v in theta]
        rhs[0] += flux[(k - 1) // op.substeps]
        theta = _ref_solve(op._factorization, rhs)
        yield k, theta


def _ref_slab_forward(op: SlabFluxOperator, vector: Sequence[float]) -> list[float]:
    out = [0.0] * (op._n_reads * op._n_sensors)
    stride = op.sample_every * op.substeps
    for k, theta in _ref_march(op, vector):
        if k % stride == 0:
            row = (k // stride - 1) * op._n_sensors
            out[row : row + op._n_sensors] = op._sensor_values(theta)
    return out


def _ref_slab_transpose(op: SlabFluxOperator, vector: Sequence[float]) -> list[float]:
    stride = op.sample_every * op.substeps
    total = op._flux_intervals * op.substeps
    result = [0.0] * op._flux_intervals
    nu = [0.0] * op.cells
    for k in range(total, 0, -1):
        lam = [op._heat_capacity_rate * v for v in nu]
        if k % stride == 0:
            row = (k // stride - 1) * op._n_sensors
            for s, (left, weight) in enumerate(op._interp):
                lam[left] += (1.0 - weight) * vector[row + s]
                lam[left + 1] += weight * vector[row + s]
        nu = _ref_solve(op._factorization, lam)
        result[(k - 1) // op.substeps] += nu[0]
    return result


# -------------------------------------------------------------------- helpers


def _hex(value: Any) -> Any:
    """Return ``value`` with every float replaced by its ``float.hex``."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return tuple(_hex(item) for item in value)
    if isinstance(value, dict):
        return {key: _hex(item) for key, item in value.items()}
    return value


def _outcome(call: Callable[[], Any]) -> tuple[Any, ...]:
    """Return ``("ok", hex value)`` or ``("error", type, message)``."""
    try:
        return ("ok", _hex(call()))
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return ("error", type(error), str(error))


def _digest(value: Any) -> str:
    """Return a short stable digest of a nested structure of floats."""
    return hashlib.sha256(repr(_hex(value)).encode()).hexdigest()[:16]


def _floats(rng: random.Random, n: int, scale: float = 1.0) -> list[float]:
    """Return ``n`` values of mixed magnitude, signs and exact zeros."""
    out = []
    for _ in range(n):
        pick = rng.random()
        if pick < 0.08:
            out.append(rng.choice((0.0, -0.0)))
        elif pick < 0.16:
            out.append(rng.choice((1.0, -1.0, 0.5, 2.0)) * scale)
        else:
            out.append(rng.gauss(0.0, 1.0) * 10.0 ** rng.randint(-6, 6) * scale)
    return out


class _FloatSub(float):
    """A ``float`` subclass."""


class _IntSub(int):
    """An ``int`` subclass."""


class _ListSub(list):  # type: ignore[type-arg]
    """A ``list`` subclass."""


def _bad_vectors(n: int) -> list[Any]:
    """Return adversarial candidates for a vector of length ``n``."""
    good = [0.25 * (i + 1) for i in range(n)]
    cases: list[Any] = [
        good,
        tuple(good),
        [int(v * 4) for v in good],
        [*good[:-1], float("nan")],
        [float("inf"), *good[1:]],
        [float("-inf"), *good[1:]],
        [*good[:-1], True],
        [*good[:-1], "1.0"],
        [*good[:-1], None],
        [*good[:-1], Fraction(1, 3)],
        [*good[:-1], Decimal("0.5")],
        [*good[:-1], 10**400],
        [float("nan"), *good[1:-1], 10**400] if n > 1 else [float("nan")],
        [10**400, *good[1:-1], float("nan")] if n > 1 else [10**400],
        [float("nan"), *good[1:-1], "a"] if n > 1 else [float("nan")],
        ["a", *good[1:-1], float("nan")] if n > 1 else ["a"],
        [_FloatSub(v) for v in good],
        [_IntSub(int(v * 4)) for v in good],
        _ListSub(good),
        [*good, 1.0],
        good[:-1],
        [],
        (),
        "abc",
        b"abc",
        None,
        3.5,
        iter(good),
        collections.deque(good),
        range(n),
        [-0.0] * n,
        [1e308] * n,
        [5e-324] * n,
        [0.1] * n,
        [[1.0]] * n,
        {i: 1.0 for i in range(n)},
    ]
    return cases


# ---------------------------------------------------------------- validation


class FiniteVectorTests(unittest.TestCase):
    """``_finite_vector`` keeps every result and every error."""

    def test_adversarial_inputs(self) -> None:
        """Candidates of every kind give the same outcome as the original."""
        for n, length in itertools.product((1, 2, 3, 7), (None, 1, 2, 3, 7, 8)):
            for index, candidate in enumerate(_bad_vectors(n)):
                with self.subTest(n=n, length=length, case=index):
                    if isinstance(candidate, Iterator):
                        pair = list(candidate)
                        first, second = iter(pair), iter(pair)
                    else:
                        first = second = candidate
                    self.assertEqual(
                        _outcome(lambda: _finite_vector("v", first, length)),
                        _outcome(lambda: _ref_finite_vector("v", second, length)),
                    )

    def test_random_vectors(self) -> None:
        """Random lists and tuples are converted to identical floats."""
        rng = random.Random(3)
        for n in (1, 2, 5, 40, 200):
            for _ in range(20):
                values = _floats(rng, n)
                if rng.random() < 0.3:
                    values[rng.randrange(n)] = rng.randint(-5, 5)
                for kind in (list, tuple):
                    self.assertEqual(
                        _outcome(lambda: _finite_vector("v", kind(values), n)),
                        _outcome(lambda: _ref_finite_vector("v", kind(values), n)),
                    )

    def test_result_is_a_new_list(self) -> None:
        """The caller's list is never returned or modified."""
        values = [1.0, 2.0, 3.0]
        out = _finite_vector("v", values, 3)
        self.assertIsNot(out, values)
        out[0] = 9.0
        self.assertEqual(values, [1.0, 2.0, 3.0])


class TridiagonalTests(unittest.TestCase):
    """Factorization, ``solve`` and ``solve_transpose`` of the Thomas solver."""

    def _bands(self, rng: random.Random, n: int) -> tuple[list[float], ...]:
        lower = [
            rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-2, 2) for _ in range(n - 1)
        ]
        upper = [
            rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-2, 2) for _ in range(n - 1)
        ]
        diagonal = [rng.uniform(5.0, 9.0) * 10.0 ** rng.randint(0, 3) for _ in range(n)]
        return lower, diagonal, upper

    def test_factors_and_solves(self) -> None:
        """Pivots, multipliers and both solves equal the original's."""
        rng = random.Random(11)
        for n in (1, 2, 3, 4, 9, 33, 120):
            for _ in range(4):
                lower, diagonal, upper = self._bands(rng, n)
                new = TridiagonalFactorization(lower, diagonal, upper)
                ref = _RefTridiagonal(lower, diagonal, upper)
                self.assertEqual(_hex(new._pivots), _hex(ref._pivots))
                self.assertEqual(_hex(new._multipliers), _hex(ref._multipliers))
                self.assertEqual(_hex(new._upper), _hex(ref._upper))
                self.assertEqual(new.order, ref.order)
                for _ in range(3):
                    rhs = _floats(rng, n)
                    with self.subTest(n=n):
                        self.assertEqual(
                            _outcome(lambda: new.solve(rhs)),
                            _outcome(lambda: ref.solve(rhs)),
                        )
                        self.assertEqual(
                            _outcome(lambda: new.solve_transpose(rhs)),
                            _outcome(lambda: ref.solve_transpose(rhs)),
                        )

    def test_adversarial_right_hand_sides(self) -> None:
        """Wrong types, lengths and non-finite entries raise the same errors."""
        rng = random.Random(5)
        for n in (1, 2, 5):
            lower, diagonal, upper = self._bands(rng, n)
            new = TridiagonalFactorization(lower, diagonal, upper)
            ref = _RefTridiagonal(lower, diagonal, upper)
            for index in range(len(_bad_vectors(n))):
                with self.subTest(n=n, case=index):
                    for method in ("solve", "solve_transpose"):
                        cand_new = _bad_vectors(n)[index]
                        cand_ref = _bad_vectors(n)[index]
                        self.assertEqual(
                            _outcome(lambda: getattr(new, method)(cand_new)),
                            _outcome(lambda: getattr(ref, method)(cand_ref)),
                        )

    def test_overflow_in_the_sweeps(self) -> None:
        """Huge right-hand sides overflow to the same ``inf`` or ``nan``."""
        rng = random.Random(6)
        lower, diagonal, upper = self._bands(rng, 12)
        new = TridiagonalFactorization(lower, diagonal, upper)
        ref = _RefTridiagonal(lower, diagonal, upper)
        for rhs in ([1e308] * 12, [-1e308, 1e308] * 6, [1e-320] * 12, [0.0] * 12):
            self.assertEqual(
                _outcome(lambda: new.solve(rhs)), _outcome(lambda: ref.solve(rhs))
            )
            self.assertEqual(
                _outcome(lambda: new.solve_transpose(rhs)),
                _outcome(lambda: ref.solve_transpose(rhs)),
            )

    def test_construction_errors(self) -> None:
        """Bad bands fail with the original messages."""
        cases: list[tuple[Any, Any, Any]] = [
            ([], [], []),
            ([1.0], [2.0], []),
            ([], [0.0], []),
            ([1.0], [0.0, 1.0], [1.0]),
            ([1.0], [1.0, 1.0], [1.0]),
            ([1.0], [float("nan"), 1.0], [1.0]),
            ([1.0], [1.0, 1.0], [float("inf")]),
            ([1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0]),
            ([1.0], [1e-300, 1.0], [1e300]),
            (["a"], [1.0, 2.0], [1.0]),
            ([True], [1.0, 2.0], [1.0]),
        ]
        for lower, diagonal, upper in cases:
            self.assertEqual(
                _outcome(
                    lambda: _hex_factor(
                        TridiagonalFactorization(lower, diagonal, upper)
                    )
                ),
                _outcome(lambda: _hex_factor(_RefTridiagonal(lower, diagonal, upper))),
            )

    def test_free_functions(self) -> None:
        """``solve_tridiagonal`` and its transpose twin match the original."""
        rng = random.Random(8)
        for n in (1, 2, 6):
            lower, diagonal, upper = self._bands(rng, n)
            rhs = _floats(rng, n)
            ref = _RefTridiagonal(lower, diagonal, upper)
            self.assertEqual(
                _outcome(
                    lambda: pde_models.solve_tridiagonal(lower, diagonal, upper, rhs)
                ),
                _outcome(lambda: tuple(ref.solve(rhs))),
            )
            self.assertEqual(
                _outcome(
                    lambda: pde_models.solve_tridiagonal_transpose(
                        lower, diagonal, upper, rhs
                    )
                ),
                _outcome(lambda: tuple(ref.solve_transpose(rhs))),
            )


def _hex_factor(factorization: Any) -> tuple[Any, ...]:
    """Return the stored factors of a factorization."""
    return (
        factorization._pivots,
        factorization._multipliers,
        factorization._upper,
        factorization.order,
    )


class EllipticTests(unittest.TestCase):
    """``EllipticDiffusion1D`` solves with the reference factorization."""

    def test_state_matches_the_original_solve(self) -> None:
        """The state equals the reference solve of the assembled system."""
        rng = random.Random(21)
        for cells in (2, 3, 17, 80):
            model = EllipticDiffusion1D(
                cells, sensors=(0.2, 0.55), source=1.0, boundary=(0.5, -0.25)
            )
            for _ in range(4):
                x = [rng.uniform(-2.0, 2.0) for _ in range(cells)]
                lower, diagonal, upper, rhs = model.system(x)
                ref = _RefTridiagonal(lower, diagonal, upper)
                self.assertEqual(_hex(model.state(x)), _hex(tuple(ref.solve(rhs))))
                weights = [rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]
                self.assertEqual(
                    _outcome(lambda: model.vjp(x, weights)),
                    _outcome(lambda: _ref_vjp(model, x, weights)),
                )

    def test_bad_arguments(self) -> None:
        """Invalid parameter vectors raise the original errors."""
        model = EllipticDiffusion1D(4, sensors=(0.5,), source=1.0)
        for index in range(len(_bad_vectors(4))):
            self.assertEqual(
                _outcome(lambda: model.value(_bad_vectors(4)[index])),
                _outcome(lambda: _ref_value(model, _bad_vectors(4)[index])),
            )


def _ref_value(model: EllipticDiffusion1D, x: Any) -> tuple[float, ...]:
    """Return the sensor readings with the reference validation of ``x``."""
    key = tuple(_ref_finite_vector("x", x, model.cells))
    system = model.system(key)
    ref = _RefTridiagonal(system[0], system[1], system[2])
    state = ref.solve(system[3])
    hu = model._observe(state)
    return tuple(a + b for a, b in zip(hu, model._offset, strict=True))


def _ref_vjp(
    model: EllipticDiffusion1D, x: Sequence[float], w: Sequence[float]
) -> tuple[float, ...]:
    """Return ``J' w`` with the reference solves."""
    ev = model._evaluate(x)
    weights = _ref_finite_vector("w", w, model.output_dimension)
    n = model.cells
    lower, diagonal, upper, _ = model.system(x)
    ref = _RefTridiagonal(lower, diagonal, upper)
    p = ref.solve_transpose(model._observe_transpose(weights))
    padded = [0.0, *p, 0.0]
    gradient = [0.0] * n
    for f in range(n + 1):
        t = ev.conductance[f] * ev.jump[f] * (padded[f + 1] - padded[f])
        if f >= 1:
            gradient[f - 1] -= t * ev.left_weight[f]
        if f < n:
            gradient[f] -= t * ev.right_weight[f]
    return tuple(gradient)


# ------------------------------------------------------------------ operators


class FirstDifferenceTests(unittest.TestCase):
    """The slice formulation of the 1D difference matches the loops."""

    def test_all_boundaries(self) -> None:
        """Forward and transpose agree for every boundary, size and spacing."""
        rng = random.Random(31)
        for boundary, n, spacing in itertools.product(
            ("none", "periodic", "dirichlet"), (1, 2, 3, 6, 50, 301), (1.0, 0.3, 7.25)
        ):
            if boundary == "none" and n < 2:
                continue
            new = FirstDifferenceOperator(n, spacing=spacing, boundary=boundary)
            ref = _RefFirstDifference(new)
            for _ in range(3):
                x = _floats(rng, n)
                y = _floats(rng, new.shape[0])
                with self.subTest(boundary=boundary, n=n, spacing=spacing):
                    self.assertEqual(
                        _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
                    )
                    self.assertEqual(
                        _outcome(lambda: new.rmatvec(y)),
                        _outcome(lambda: ref.rmatvec(y)),
                    )

    def test_signed_zeros_and_overflow(self) -> None:
        """Signed zeros keep their sign and overflow raises the same error."""
        for boundary in ("none", "periodic", "dirichlet"):
            new = FirstDifferenceOperator(5, spacing=2.0, boundary=boundary)
            ref = _RefFirstDifference(new)
            rows = new.shape[0]
            for pattern in (
                [0.0, -0.0],
                [-0.0],
                [0.0],
                [1e308, -1e308],
                [5e-324, -5e-324],
            ):
                x = [pattern[i % len(pattern)] for i in range(5)]
                y = [pattern[i % len(pattern)] for i in range(rows)]
                self.assertEqual(
                    _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
                )
                self.assertEqual(
                    _outcome(lambda: new.rmatvec(y)), _outcome(lambda: ref.rmatvec(y))
                )

    def test_bad_inputs(self) -> None:
        """Wrong lengths and non-finite entries raise the same errors."""
        for boundary in ("none", "periodic", "dirichlet"):
            new = FirstDifferenceOperator(4, boundary=boundary)
            ref = _RefFirstDifference(new)
            for index in range(len(_bad_vectors(4))):
                for name in ("matvec", "rmatvec"):
                    a, b = _bad_vectors(4)[index], _bad_vectors(4)[index]
                    self.assertEqual(
                        _outcome(lambda: getattr(new, name)(a)),
                        _outcome(lambda: getattr(ref, name)(b)),
                    )


class GridGradientTests(unittest.TestCase):
    """The slice formulation of the 2D gradient matches the raster loops."""

    def test_shapes(self) -> None:
        """Every grid shape, including single rows and columns, agrees."""
        rng = random.Random(41)
        for rows, columns in itertools.product((1, 2, 3, 7, 20), (1, 2, 3, 8, 21)):
            new = GridGradientOperator(rows, columns, spacing=(0.3, 1.7))
            ref = _RefGrid(new)
            for _ in range(3):
                x = _floats(rng, rows * columns)
                y = _floats(rng, 2 * rows * columns)
                with self.subTest(rows=rows, columns=columns):
                    self.assertEqual(
                        _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
                    )
                    self.assertEqual(
                        _outcome(lambda: new.rmatvec(y)),
                        _outcome(lambda: ref.rmatvec(y)),
                    )

    def test_signed_zeros_overflow_and_bad_inputs(self) -> None:
        """Signed zeros, huge values and invalid vectors agree."""
        new = GridGradientOperator(3, 4, spacing=(1.0, 3.0))
        ref = _RefGrid(new)
        for pattern in ([0.0, -0.0], [-0.0], [1e308, -1e308], [5e-324, 1.0]):
            x = [pattern[i % len(pattern)] for i in range(12)]
            y = [pattern[i % len(pattern)] for i in range(24)]
            self.assertEqual(
                _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
            )
            self.assertEqual(
                _outcome(lambda: new.rmatvec(y)), _outcome(lambda: ref.rmatvec(y))
            )
        for index in range(len(_bad_vectors(12))):
            a, b = _bad_vectors(12)[index], _bad_vectors(12)[index]
            self.assertEqual(
                _outcome(lambda: new.matvec(a)), _outcome(lambda: ref.matvec(b))
            )


class ConvolutionTests(unittest.TestCase):
    """The shifted-stencil fast path matches the nested loops."""

    KERNELS = (
        (1.0,),
        (0.5, 0.5),
        (0.25, 0.5, 0.25),
        (0.05, 0.2, 0.5, 0.2, 0.05),
        (0.3, -0.2, 0.0, 0.7),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        tuple(0.1 * (i + 1) * (-1) ** i for i in range(26)),
        (0.0, 0.0),
    )

    def test_kernels_boundaries_and_sizes(self) -> None:
        """All combinations, with and without a usable shifted block."""
        rng = random.Random(51)
        for kernel, boundary, n in itertools.product(
            self.KERNELS, ("zero", "periodic", "reflect"), (1, 2, 5, 9, 10, 11, 40, 205)
        ):
            new = Convolution1DOperator(kernel, n, boundary=boundary)
            ref = _RefConvolution(kernel, n, boundary=boundary)
            self.assertEqual(new._taps, ref._taps)
            for _ in range(2):
                x = _floats(rng, n)
                with self.subTest(kernel=kernel, boundary=boundary, n=n):
                    self.assertEqual(
                        _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
                    )
                    self.assertEqual(
                        _outcome(lambda: new.rmatvec(x)),
                        _outcome(lambda: ref.rmatvec(x)),
                    )

    def test_fast_path_is_used(self) -> None:
        """A long signal does use the shifted block (the test is not vacuous)."""
        operator = Convolution1DOperator((0.05, 0.2, 0.5, 0.2, 0.05), 300)
        self.assertEqual(operator._block, (2, 298))
        short = Convolution1DOperator((0.05, 0.2, 0.5, 0.2, 0.05), 6)
        self.assertEqual(short._block, (0, 0))

    def test_signed_zeros_overflow_and_cancellation(self) -> None:
        """Exact sums, signed zeros and overflow behave as before."""
        kernel = (1.0, -1.0, 1.0, -1.0, 1.0)
        new = Convolution1DOperator(kernel, 30, boundary="reflect")
        ref = _RefConvolution(kernel, 30, boundary="reflect")
        patterns = (
            [1e16, 1.0, -1e16, 1.0, 1e-16],
            [0.0, -0.0],
            [-0.0],
            [1e308, -1e308, 1e308],
            [5e-324, 1e-323],
            [1.0, 1e-17, -1.0, 1e-17],
        )
        for pattern in patterns:
            x = [pattern[i % len(pattern)] for i in range(30)]
            self.assertEqual(
                _outcome(lambda: new.matvec(x)), _outcome(lambda: ref.matvec(x))
            )
            self.assertEqual(
                _outcome(lambda: new.rmatvec(x)), _outcome(lambda: ref.rmatvec(x))
            )

    def test_bad_inputs(self) -> None:
        """Wrong lengths and non-finite entries raise the same errors."""
        new = Convolution1DOperator((0.25, 0.5, 0.25), 20)
        ref = _RefConvolution((0.25, 0.5, 0.25), 20)
        for index in range(len(_bad_vectors(20))):
            for name in ("matvec", "rmatvec"):
                a, b = _bad_vectors(20)[index], _bad_vectors(20)[index]
                self.assertEqual(
                    _outcome(lambda: getattr(new, name)(a)),
                    _outcome(lambda: getattr(ref, name)(b)),
                )

    def test_constructor_errors(self) -> None:
        """Invalid constructor arguments raise the original errors."""
        for kernel, size, boundary in (
            ((), 5, "zero"),
            ((1.0,), 0, "zero"),
            ((1.0,), 5, "wrap"),
            ((float("nan"),), 5, "zero"),
            (("a",), 5, "zero"),
        ):
            outcome = _outcome(
                lambda: Convolution1DOperator(kernel, size, boundary=boundary).shape
            )
            self.assertEqual(outcome[0], "error")

    def test_materialize(self) -> None:
        """``materialize`` returns the same matrix for several operators."""
        rng = random.Random(61)
        dense = DenseOperator([_floats(rng, 4) for _ in range(6)])
        operators: list[Any] = [
            Convolution1DOperator((0.2, 0.6, 0.2), 25, boundary="periodic"),
            FirstDifferenceOperator(9, boundary="dirichlet"),
            GridGradientOperator(3, 4),
            dense,
            _floats_matrix(rng, 5, 3),
        ]
        for operator in operators:
            self.assertEqual(
                _outcome(lambda: materialize(operator)),
                _outcome(lambda: _ref_materialize(operator)),
            )
        for bad in (0, "x", None):
            self.assertEqual(
                _outcome(lambda: materialize(bad)),
                _outcome(lambda: _ref_materialize(bad)),
            )
        self.assertEqual(
            _outcome(lambda: materialize(GridGradientOperator(40, 40), max_entries=10)),
            ("error", ValueError, "materializing 3200 x 1600 exceeds max_entries=10"),
        )


def _floats_matrix(rng: random.Random, rows: int, columns: int) -> list[list[float]]:
    """Return a small dense matrix."""
    return [_floats(rng, columns) for _ in range(rows)]


# ----------------------------------------------------------------- heat models


def _heat_operators() -> list[tuple[str, _HeatOperator]]:
    """Return heat operators of both kinds, schemes and observation modes."""
    out: list[tuple[str, _HeatOperator]] = []
    for kind, scheme, observation in itertools.product(
        (HeatEquation1D, HeatSource1D),
        ("implicit_euler", "crank_nicolson"),
        ("final", "sensors"),
    ):
        extra: dict[str, Any] = {}
        if observation == "sensors":
            extra = {"sensors": (0.15, 0.5, 0.83), "times": (0.0, 0.1, 0.45, 0.8, 1.0)}
        for cells, steps in ((2, 1), (9, 6), (24, 13)):
            operator = kind(
                cells, steps=steps, scheme=scheme, observation=observation, **extra
            )
            out.append((f"{kind.__name__}-{scheme}-{observation}-{cells}", operator))
    return out


class HeatModelTests(unittest.TestCase):
    """Time stepping of the heat operators and their work counters."""

    def test_forward_and_adjoint(self) -> None:
        """Observations, adjoints and counters equal the original loops."""
        rng = random.Random(71)
        for name, op in _heat_operators():
            source = isinstance(op, HeatSource1D)
            for _ in range(3):
                x = _floats(rng, op.shape[1], 1e-3)
                y = _floats(rng, op.shape[0], 1e-3)
                with self.subTest(operator=name):
                    op.reset_counts()
                    counts = [0, 0, 0, 0]
                    got = _outcome(lambda: op.matvec(x))
                    want = _outcome(
                        lambda: _vector_of(
                            op,
                            _ref_run_forward(
                                op,
                                None if source else x,
                                x if source else None,
                                counts,
                            ),
                            op.shape[0],
                        )
                    )
                    self.assertEqual(got, want)
                    got_t = _outcome(lambda: op.rmatvec(y))
                    want_t = _outcome(
                        lambda: _vector_of(
                            op,
                            _ref_run_transpose(op, y, source=source, counts=counts),
                            op.shape[1],
                        )
                    )
                    self.assertEqual(got_t, want_t)
                    snapshot = op.counts
                    self.assertEqual(
                        [
                            snapshot.forward_applications,
                            snapshot.transpose_applications,
                            snapshot.forward_solves,
                            snapshot.transpose_solves,
                        ],
                        counts,
                    )

    def test_overflow_and_bad_inputs(self) -> None:
        """Overflowing sources and invalid vectors raise the same errors."""
        for name, op in _heat_operators()[::3]:
            source = isinstance(op, HeatSource1D)
            n = op.shape[1]
            candidates = [[1e308] * n, [-1e308, 1e308] * n, [0.0] * n, [-0.0] * n]
            for index in range(len(_bad_vectors(n))):
                candidates.append(_bad_vectors(n)[index])
            for index, x in enumerate(candidates):
                if isinstance(x, Iterator):
                    continue
                with self.subTest(operator=name, case=index):
                    counts = [0, 0, 0, 0]
                    self.assertEqual(
                        _outcome(lambda: op.matvec(x)),
                        _outcome(
                            lambda: _vector_of(
                                op,
                                _ref_run_forward(
                                    op,
                                    None if source else _checked(op, x, n),
                                    _checked(op, x, n) if source else None,
                                    counts,
                                ),
                                op.shape[0],
                            )
                        ),
                    )


def _checked(op: OperatorBase, x: Any, n: int) -> tuple[float, ...]:
    """Validate an operator input the way ``matvec`` does."""
    return _operator_vector(x, n, f"{op.name} input")


def _operator_vector(values: Any, size: int, name: str) -> tuple[float, ...]:
    """Mirror of the operator vector validation (unchanged by the wave)."""
    from probstat.linalg.operators import _vector

    return _vector(values, size, name)


def _vector_of(
    op: OperatorBase, values: Sequence[float], size: int
) -> tuple[float, ...]:
    """Validate an operator output the way ``matvec`` does."""
    return _operator_vector(values, size, f"{op.name} output")


# ------------------------------------------------------------------ slab flux


def _slab_operators() -> list[SlabFluxOperator]:
    """Return slab operators with and without a back-face coefficient."""
    out = []
    for back, cells, intervals, sub, every, sensors in (
        (0.0, 30, 12, 3, 1, (0.002, 0.004)),
        (25.0, 12, 8, 2, 2, (0.0021,)),
        (0.0, 2, 4, 1, 1, (0.01,)),
        (5.0, 40, 6, 4, 3, (0.003, 0.009, 0.0151)),
    ):
        out.append(
            SlabFluxOperator(
                synthetic_ablator_slab(back_coefficient=back),
                sensors,
                flux_intervals=intervals,
                interval_length=0.5,
                cells=cells,
                substeps=sub,
                sample_every=every,
            )
        )
    return out


class SlabFluxTests(unittest.TestCase):
    """Marching and adjoint recursion of the slab operator."""

    def test_forward_transpose_and_history(self) -> None:
        """Readings, adjoints, histories and energy balance agree."""
        rng = random.Random(81)
        for op in _slab_operators():
            for _ in range(3):
                flux = _floats(rng, op.shape[1], 1e4)
                data = _floats(rng, op.shape[0], 1e-2)
                with self.subTest(cells=op.cells):
                    self.assertEqual(
                        _outcome(lambda: op.matvec(flux)),
                        _outcome(
                            lambda: _vector_of(
                                op, _ref_slab_forward(op, flux), op.shape[0]
                            )
                        ),
                    )
                    self.assertEqual(
                        _outcome(lambda: op.rmatvec(data)),
                        _outcome(
                            lambda: _vector_of(
                                op, _ref_slab_transpose(op, data), op.shape[1]
                            )
                        ),
                    )
                    reference_history = [(0.0,) * op.cells]
                    for _, theta in _ref_march(op, flux):
                        reference_history.append(tuple(theta))
                    self.assertEqual(
                        _hex(op.field_history(flux)), _hex(tuple(reference_history))
                    )

    def test_overflow_and_bad_inputs(self) -> None:
        """Huge fluxes and invalid vectors raise the same errors."""
        op = _slab_operators()[0]
        n = op.shape[1]
        candidates = [[1e308] * n, [-1e308, 1e308] * n, [0.0] * n, [-0.0] * n]
        candidates += [_bad_vectors(n)[i] for i in range(len(_bad_vectors(n)))]
        for index, flux in enumerate(candidates):
            if isinstance(flux, Iterator):
                continue
            with self.subTest(case=index):
                self.assertEqual(
                    _outcome(lambda: op.matvec(flux)),
                    _outcome(
                        lambda: _vector_of(
                            op,
                            _ref_slab_forward(op, _checked(op, flux, n)),
                            op.shape[0],
                        )
                    ),
                )

    def test_module_unchanged_symbols(self) -> None:
        """The module still exposes the operator the tests rely on."""
        self.assertTrue(hasattr(slab_module, "SlabFluxOperator"))


# --------------------------------------------------------------------- golden


def _golden_operators() -> dict[str, str]:
    """Digest whole workloads of the operators and their Krylov solves."""
    rng = random.Random(91)
    out: dict[str, str] = {}
    size = 400
    conv = Convolution1DOperator((0.05, 0.2, 0.5, 0.2, 0.05), size)
    truth = tuple(math.sin(2.0 * math.pi * i / 90.0) for i in range(size))
    data = tuple(conv.matvec(truth))
    out["lsqr_conv"] = _digest(_outcome(lambda: lsqr(conv, data, max_iterations=40)))
    for boundary in ("periodic", "reflect"):
        wrap = Convolution1DOperator((0.2, 0.3, 0.5), size, boundary=boundary)
        out[f"lsqr_conv_{boundary}"] = _digest(
            _outcome(lambda: lsqr(wrap, tuple(wrap.matvec(truth)), max_iterations=30))
        )
    grid = GridGradientOperator(12, 15, spacing=(0.5, 0.25))
    image = _floats(rng, 12 * 15)
    out["grid"] = _digest((grid.matvec(image), grid.rmatvec(_floats(rng, 2 * 12 * 15))))
    diff = FirstDifferenceOperator(97, spacing=0.5, boundary="periodic")
    out["difference"] = _digest(
        (diff.matvec(_floats(rng, 97)), diff.rmatvec(_floats(rng, 97)))
    )
    out["materialize"] = _digest(
        materialize(Convolution1DOperator((0.25, 0.5, 0.25), 60, boundary="reflect"))
    )
    return out


def _golden_heat() -> dict[str, str]:
    """Digest whole heat-stepping and slab-march workloads."""
    rng = random.Random(92)
    out: dict[str, str] = {}
    heat = HeatSource1D(
        40,
        diffusivity=0.5,
        steps=30,
        scheme="crank_nicolson",
        observation="sensors",
        sensors=(0.2, 0.5, 0.8),
        times=(0.25, 0.5, 0.75, 1.0),
    )
    source = _floats(rng, 40, 1e-3)
    observed = _floats(rng, heat.shape[0], 1e-3)
    out["heat_source"] = _digest((heat.matvec(source), heat.rmatvec(observed)))
    initial = HeatEquation1D(31, steps=20, scheme="implicit_euler")
    out["heat_initial"] = _digest(
        (
            initial.matvec(_floats(rng, 31, 1e-3)),
            initial.rmatvec(_floats(rng, 31, 1e-3)),
        )
    )
    slab = _slab_operators()[0]
    flux = _floats(rng, slab.shape[1], 1e4)
    out["slab"] = _digest(
        (
            slab.matvec(flux),
            slab.rmatvec(_floats(rng, slab.shape[0], 1e-2)),
            slab.field_history(flux)[-1],
            slab.sensitivity_matrix()[-1],
        )
    )
    elliptic = EllipticDiffusion1D(
        60, sensors=(0.2, 0.4, 0.6), source=1.0, boundary=(0.0, 1.0)
    )
    x = [0.3 * math.sin(0.1 * i) for i in range(60)]
    out["elliptic"] = _digest(
        (
            elliptic.value(x),
            elliptic.jvp(x, x),
            elliptic.vjp(x, [0.5, -1.0, 2.0]),
            elliptic.state(x),
        )
    )
    return out


GOLDEN: dict[str, dict[str, str]] = {
    "operators": {
        "lsqr_conv": "f730df8ed7810761",
        "lsqr_conv_periodic": "a121a0dcc9eb0798",
        "lsqr_conv_reflect": "c95b70c7a3a349e7",
        "grid": "392340226b3e36d9",
        "difference": "84d37c8e8e112506",
        "materialize": "7d7da6437163d031",
    },
    "heat": {
        "heat_source": "567098157f0d1c9c",
        "heat_initial": "bddbc98756eabbef",
        "slab": "a6ae9b71d63a2dff",
        "elliptic": "96cac55a0ecf4441",
    },
}


class GoldenTests(unittest.TestCase):
    """Whole-workload digests recorded from the reference implementations."""

    def test_operator_workloads(self) -> None:
        """Convolution solves, gradients and ``materialize`` reproduce."""
        self.assertEqual(_golden_operators(), GOLDEN["operators"])

    def test_heat_workloads(self) -> None:
        """Heat stepping, slab marching and the elliptic model reproduce."""
        self.assertEqual(_golden_heat(), GOLDEN["heat"])


if __name__ == "__main__":
    import sys

    if "--print-golden" in sys.argv:
        import json

        print(
            json.dumps(
                {"operators": _golden_operators(), "heat": _golden_heat()}, indent=4
            )
        )
    else:
        unittest.main()
