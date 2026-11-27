"""Bit-for-bit equivalence of the sixth wave of performance rewrites.

Covered, all in ``probstat.surrogates.gaussian_process``: the row-wise
assembly of the kernel matrix and of the derivative pieces (``_assemble``,
including which operation's error is reported when squares or sums overflow),
and the bulk cross-covariance and ``map(mul, ...)`` sums of
``GaussianProcessRegression.predict_with_variance``. The Cholesky sweep, the
triangular inversion, the log marginal likelihood with its gradient and the
fit-time factorization are pinned against the reference as well, although
this wave leaves their code as it was; the golden digests pin whole
``fit_gaussian_process`` runs (fitted hyperparameters, evaluation counts,
convergence flags, condition numbers, predictions, leave-one-out diagnostics
and warnings).

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(``probstat.surrogates.gaussian_process``), copied verbatim apart from removed
docstrings, type annotations and the ``_ref_`` prefix. Every test feeds
fixed-seed and adversarial inputs (NaN, inf, ``-0.0``, ``bool``, ``int`` and
``float`` subclasses, strings, ragged points, wrong lengths, values whose
squares or sums overflow, non-positive-definite matrices, short rows, an
unknown kernel name) to the current and to the reference routine and compares
floats by ``float.hex`` and exceptions by type, message and cause. The golden
digests at the end were produced by the checkout of that revision
(``--print-golden`` with its ``src`` on ``PYTHONPATH``) for whole fits.

References:
    [REF-PROBSTAT-VALIDATION]. Equivalence harness only; no equation is
    evaluated.

"""

# ruff: noqa: B023 - every lambda below is called immediately inside its loop
from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import unittest
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from operator import mul
from typing import Any

from probstat.foundations.validation import require_finite_real
from probstat.linalg import as_vector, solve_lower
from probstat.surrogates import (
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
)
from probstat.surrogates.gaussian_process import (
    _LOG_TWO_PI,
    _SQRT3,
    _SQRT5,
    KERNELS,
    _accelerated_backend,
    _assemble,
    _dense_likelihood,
    _factorize_dense,
    _fast_cholesky,
    _gradient,
    _inverse_factor_columns,
    _precision_entry,
    _split_parameters,
    _sumprod,
    _triangular_solve,
)

# ---------------------------------------------------------------- references


def _ref_fast_cholesky(matrix):
    count = len(matrix)
    lower = []
    for i in range(count):
        row_i = matrix[i]
        current = []
        for j in range(i):
            row_j = lower[j]
            current.append((row_i[j] - math.fsum(map(mul, current, row_j))) / row_j[j])
        pivot = row_i[i] - math.fsum(map(mul, current, current))
        if not pivot > 0.0:
            raise ValueError("kernel matrix is not positive definite")
        current.append(math.sqrt(pivot))
        lower.append(current)
    return lower


def _ref_inverse_factor_columns(lower):
    count = len(lower)
    columns = []
    for i in range(count):
        row = lower[i]
        inverse_diagonal = 1.0 / row[i]
        for j in range(i):
            column = columns[j]
            column.append(-_sumprod(row[j:i], column) * inverse_diagonal)
        columns.append([inverse_diagonal])
    return columns


def _ref_assemble(points, scales, signal, noise, kernel):
    count = len(points)
    parts = []
    correlation = []
    factors = []
    matrix = []
    squared_exponential = kernel == "squared_exponential"
    matern32 = kernel == "matern32"
    for i in range(count):
        row_parts = []
        row_corr = []
        row_factor = []
        row_matrix = []
        xi = points[i]
        for j in range(i):
            difference = tuple(
                ((a - b) / ell) ** 2
                for a, b, ell in zip(xi, points[j], scales, strict=True)
            )
            squared = math.fsum(difference)
            if squared_exponential:
                value = math.exp(-0.5 * squared)
                factor = value
            elif matern32:
                r = math.sqrt(squared)
                decay = math.exp(-_SQRT3 * r)
                value = (1.0 + _SQRT3 * r) * decay
                factor = 3.0 * decay
            else:
                r = math.sqrt(squared)
                decay = math.exp(-_SQRT5 * r)
                value = (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * decay
                factor = (5.0 / 3.0) * (1.0 + _SQRT5 * r) * decay
            row_parts.append(difference)
            row_corr.append(value)
            row_factor.append(factor)
            row_matrix.append(signal * value)
        row_matrix.append(signal + noise)
        parts.append(row_parts)
        correlation.append(row_corr)
        factors.append(row_factor)
        matrix.append(row_matrix)
    return matrix, parts, correlation, factors


def _ref_log_marginal_likelihood(
    points,
    targets,
    log_parameters,
    *,
    kernel="matern52",
    nugget=1e-10,
    estimate_noise=False,
    relative_nugget=0.0,
):
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(targets, name="targets")
    dimension = len(x[0])
    signal, scales, extra = _split_parameters(log_parameters, dimension, estimate_noise)
    tau = require_finite_real(relative_nugget, name="relative_nugget")
    if tau < 0.0:
        raise ValueError("relative_nugget must be non-negative")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    count = len(x)
    noise = nugget + extra
    if tau > 0.0:
        noise += tau * signal
    matrix, parts, correlation, factors = _ref_assemble(
        x, scales, signal, noise, kernel
    )
    backend = _accelerated_backend()
    if backend is not None:
        try:
            factor = _factorize_dense(backend, matrix)
        except ValueError as error:
            raise ValueError(
                "kernel matrix is not positive definite; increase the nugget"
            ) from error
        value, trace_weight, signal_sum, dense_sums = _dense_likelihood(
            factor, y, parts, correlation, factors, dimension
        )
        return value, _gradient(
            signal, trace_weight, signal_sum, dense_sums, tau, extra, estimate_noise
        )
    try:
        lower = _ref_fast_cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            "kernel matrix is not positive definite; increase the nugget"
        ) from error
    alpha = _triangular_solve(lower, y)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    value = -0.5 * _sumprod(y, alpha)
    value += -0.5 * log_det - 0.5 * count * _LOG_TWO_PI
    columns = _ref_inverse_factor_columns(lower)
    diagonal_weight = [
        alpha[a] * alpha[a] - _sumprod(columns[a], columns[a]) for a in range(count)
    ]
    trace_weight = math.fsum(diagonal_weight)
    signal_sum = 0.0
    axis_sums = [0.0] * dimension
    for b in range(count):
        alpha_b = alpha[b]
        column_b = columns[b]
        row_corr = correlation[b]
        row_factor = factors[b]
        row_parts = parts[b]
        for a in range(b):
            weight = alpha[a] * alpha_b - _sumprod(columns[a][b - a :], column_b)
            signal_sum += weight * row_corr[a]
            scaled_weight = weight * row_factor[a]
            difference = row_parts[a]
            for axis in range(dimension):
                axis_sums[axis] += scaled_weight * difference[axis]
    return value, _gradient(
        signal, trace_weight, signal_sum, axis_sums, tau, extra, estimate_noise
    )


def _ref_correlation(kernel, squared):
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    if kernel == "matern52":
        r = math.sqrt(squared)
        return (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * math.exp(-_SQRT5 * r)
    if kernel == "matern32":
        r = math.sqrt(squared)
        return (1.0 + _SQRT3 * r) * math.exp(-_SQRT3 * r)
    raise ValueError(f"kernel must be one of {KERNELS}")


def _ref_kernel_value(kernel, left, right, length_scales, signal_variance):
    squared = math.fsum(
        ((a - b) / ell) ** 2
        for a, b, ell in zip(left, right, length_scales, strict=True)
    )
    return signal_variance * _ref_correlation(kernel, squared)


def _ref_predict_with_variance(model, x, include_noise=False):
    point = as_vector(x, name="x")
    if len(point) != len(model.points[0]):
        raise ValueError("x must have the training input dimension")
    signal, scales, noise, lower, alpha = model._factorization
    cross = tuple(
        _ref_kernel_value(model.kernel, point, train, scales, signal)
        for train in model.points
    )
    mean = math.fsum(a * b for a, b in zip(cross, alpha, strict=True))
    if model._dense_factor is None:
        whitened = solve_lower(lower, cross)
    else:
        whitened = model._dense_factor.solve_lower(cross)
    variance = signal - math.fsum(v * v for v in whitened)
    if include_noise:
        variance += noise
    return (
        model.output_mean + model.output_scale * mean,
        model.output_scale**2 * max(variance, 0.0),
    )


def _ref_standardized(model):
    """Return the reference ``(signal, scales, noise, lower, alpha)``."""
    dimension = len(model.points[0])
    signal, scales, extra = _split_parameters(
        model.log_parameters, dimension, model.estimate_noise
    )
    noise = model.nugget + extra
    if model.relative_nugget > 0.0:
        noise += model.relative_nugget * signal
    count = len(model.points)
    rows, _, _, _ = _ref_assemble(model.points, scales, signal, noise, model.kernel)
    targets = tuple(
        (value - model.output_mean) / model.output_scale for value in model.values
    )
    factor = _ref_fast_cholesky(rows)
    lower = tuple(tuple(factor[i]) + (0.0,) * (count - 1 - i) for i in range(count))
    return signal, scales, noise, lower, _triangular_solve(factor, targets)


def _ref_condition_number(rows, lower):
    count = len(rows)
    norm_k = max(
        math.fsum(abs(rows[max(a, b)][min(a, b)]) for b in range(count))
        for a in range(count)
    )
    columns = _ref_inverse_factor_columns(lower)
    norm_inverse = max(
        math.fsum(
            abs(_precision_entry(columns, min(a, b), max(a, b))) for b in range(count)
        )
        for a in range(count)
    )
    return norm_k * norm_inverse


# ------------------------------------------------------------------- helpers


def _hex(value: Any) -> Any:
    """Return ``value`` with every float replaced by its ``float.hex``."""
    if isinstance(value, float):
        return value.hex()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _hex(dataclasses.astuple(value))
    if isinstance(value, (list, tuple)):
        return tuple(_hex(item) for item in value)
    return value


def _run(call: Callable[[], Any]) -> tuple[Any, ...]:
    """Return ``("ok", type, hex value)`` or ``("error", type, message, cause)``."""
    try:
        value = call()
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        cause = error.__cause__
        return (
            "error",
            type(error),
            str(error),
            None if cause is None else (type(cause), str(cause)),
        )
    return ("ok", type(value), _hex(value))


def _digest(value: Any) -> str:
    """Return a short stable digest of a nested structure."""
    return hashlib.sha256(repr(_hex(value)).encode()).hexdigest()[:16]


def _points(
    rng: random.Random, n: int, d: int, spread: float = 1.0
) -> list[list[float]]:
    """Return ``n`` points of mixed magnitude with exact zeros and duplicates."""
    out = []
    for _ in range(n):
        if out and rng.random() < 0.1:
            out.append(list(rng.choice(out)))
            continue
        row = []
        for _ in range(d):
            pick = rng.random()
            if pick < 0.06:
                row.append(rng.choice((0.0, -0.0)))
            else:
                row.append(rng.random() * spread)
        out.append(row)
    return out


def _targets(rng: random.Random, pts: Sequence[Sequence[float]]) -> list[float]:
    """Return a smooth standardized-like response with a little noise."""
    return [
        math.sin(3.0 * p[0]) + sum(p) - 0.5 * p[-1] + 0.01 * rng.gauss(0.0, 1.0)
        for p in pts
    ]


class _FloatSub(float):
    """A ``float`` subclass."""


class _IntSub(int):
    """An ``int`` subclass."""


def _theta(d: int, noise: bool, rng: random.Random) -> list[float]:
    """Return a log-hyperparameter vector of the right length."""
    out = [rng.uniform(-1.0, 1.0)]
    out.extend(rng.uniform(-1.5, 0.5) for _ in range(d))
    if noise:
        out.append(rng.uniform(-9.0, -4.0))
    return out


# ------------------------------------------------------------------ assemble


class AssembleEquivalence(unittest.TestCase):
    """``_assemble`` forms the same matrix and derivative pieces as before."""

    def _same(self, points: Any, scales: Any, kernel: str, signal: float = 1.7) -> None:
        reference = _run(lambda: _ref_assemble(points, scales, signal, 1e-6, kernel))
        current = _run(lambda: _assemble(points, scales, signal, 1e-6, kernel))
        self.assertEqual(current, reference)
        bare = _run(
            lambda: _assemble(points, scales, signal, 1e-6, kernel, derivatives=False)
        )
        if reference[0] == "ok":
            self.assertEqual(bare[0], "ok")
            self.assertEqual(bare[2][0], reference[2][0])
            self.assertEqual(bare[2][1:], ((), (), ()))
        else:
            self.assertEqual(bare, reference)

    def test_random_points_and_scales(self) -> None:
        """Mixed points, every kernel, several shapes and scale profiles."""
        rng = random.Random(601)
        shapes = [(1, 1), (2, 3), (6, 2), (15, 4), (9, 1), (12, 7)]
        for kernel in KERNELS:
            for n, d in shapes:
                for profile in ("moderate", "mixed", "extreme"):
                    pts = _points(rng, n, d, spread=rng.choice((1.0, 40.0, 1e-3)))
                    if profile == "moderate":
                        scales = [rng.uniform(0.2, 2.0) for _ in range(d)]
                    elif profile == "mixed":
                        scales = [10.0 ** rng.uniform(-3, 3) for _ in range(d)]
                    else:
                        scales = [math.exp(rng.choice((-25.0, 25.0))) for _ in range(d)]
                    self._same(pts, scales, kernel)

    def test_unknown_kernel_name_uses_the_default_branch(self) -> None:
        """Any other kernel string falls through to the Matern 5/2 formulas."""
        rng = random.Random(602)
        pts = _points(rng, 8, 3)
        self._same(pts, [0.5, 1.5, 0.9], "no_such_kernel")

    def test_exotic_coordinates(self) -> None:
        """ints, bools, fractions, float subclasses, strings, None, nan, inf."""
        scales = [0.7, 1.3]
        base = [[0.1, 0.2], [0.4, 0.9], [0.3, 0.3], [0.8, 0.1]]
        variants: list[Any] = [
            [[0, 1], [2, 3], [1, 1], [4, 0]],
            [[True, False], [False, True], [True, True]],
            [[Fraction(1, 3), 0.5], [0.25, Fraction(2, 7)], [1, 2]],
            [[_FloatSub(0.5), _IntSub(2)], [0.25, 1.0], [3.0, -1.0]],
            [[0.1, "a"], [0.2, 0.3], [0.4, 0.5]],
            [[0.1, None], [0.2, 0.3], [0.4, 0.5]],
            [[math.nan, 0.5], [0.2, 0.3], [0.4, 0.5]],
            [[math.inf, 0.5], [0.2, 0.3], [0.4, 0.5]],
            [[-math.inf, 0.5], [math.inf, 0.3], [0.4, 0.5]],
            [[-0.0, 0.0], [0.0, -0.0], [1e-320, -1e-320]],
            base,
            (tuple(r) for r in base),
        ]
        for kernel in KERNELS:
            for variant in variants:
                if not isinstance(variant, list):
                    variant = list(variant)
                self._same(variant, scales, kernel)

    def test_ragged_and_mismatched_lengths(self) -> None:
        """Length mismatches raise the strict-zip ``ValueError`` unchanged."""
        cases: list[tuple[Any, Any]] = [
            ([[0.1, 0.2], [0.3], [0.5, 0.6]], [1.0, 1.0]),
            ([[0.1, 0.2], [0.3, 0.4, 0.5], [0.5, 0.6]], [1.0, 1.0]),
            ([[0.1, 0.2], [0.3, 0.4]], [1.0]),
            ([[0.1, 0.2], [0.3, 0.4]], [1.0, 1.0, 1.0]),
            ([[0.1], [0.3], [0.4]], []),
            ([[], [], []], [1.0]),
            ([[], [], []], []),
            ([], [1.0, 1.0]),
            ([[0.1, 0.2]], [1.0, 1.0]),
            ([[0.1, 0.2]], [1.0]),
        ]
        for kernel in KERNELS:
            for points, scales in cases:
                self._same(points, scales, kernel)

    def test_failing_scales(self) -> None:
        """Zero, negative, nan and infinite scales fail or propagate alike."""
        rng = random.Random(603)
        pts = _points(rng, 7, 3)
        for kernel in KERNELS:
            for scales in (
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 2.0, 1.0],
                [math.nan, 1.0, 1.0],
                [math.inf, 1.0, 1.0],
                [1.0, 1e-300, 1.0],
                [1.0, 1e-160, 1.0],
            ):
                self._same(pts, scales, kernel)

    def test_overflow_reports_the_reference_failure(self) -> None:
        """The first failing operation of the pair-by-pair order is reported."""
        huge = 1e154
        # Row 2: pair (2, 0) overflows inside ``fsum`` (three squares of 1e308),
        # pair (2, 1) overflows in the square itself. The reference reaches
        # ``fsum`` of pair (2, 0) first.
        cases = [
            [[0.0, 0.0, 0.0], [-huge, 0.0, 0.0], [huge, huge, huge]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1e155, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [huge, huge, huge]],
            [[0.0, 0.0, 0.0], [1e200, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1e150, 1e150, 1e150], [2e154, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1e308, 0.0, 0.0], [-1e308, 0.0, 0.0]],
            [[1.7e308, 0.0, 0.0], [-1.7e308, 0.0, 0.0]],
        ]
        for kernel in KERNELS:
            for points in cases:
                for scales in ([1.0, 1.0, 1.0], [1e-5, 1.0, 3.0], [1e-11] * 3):
                    self._same(points, scales, kernel)

    def test_aliasing_is_absent(self) -> None:
        """The correlation and factor rows are separate lists, as before."""
        rng = random.Random(604)
        pts = _points(rng, 6, 2)
        _, _, correlation, factors = _assemble(
            pts, [0.8, 1.2], 1.0, 0.0, "squared_exponential"
        )
        self.assertEqual(correlation, factors)
        self.assertTrue(
            all(c is not f for c, f in zip(correlation, factors, strict=True))
        )
        correlation[3][0] += 1.0
        self.assertNotEqual(correlation[3][0], factors[3][0])


# ------------------------------------------------------------- linear algebra


class CholeskyEquivalence(unittest.TestCase):
    """``_fast_cholesky`` returns the same factor and fails the same way."""

    def test_kernel_matrices(self) -> None:
        """Kernel matrices of several conditioning, including singular ones."""
        rng = random.Random(611)
        for kernel in KERNELS:
            for n, d in ((1, 1), (2, 2), (9, 2), (24, 3), (30, 1)):
                for noise in (1e-12, 1e-8, 1e-3, 0.0, -1e-3):
                    pts = _points(rng, n, d)
                    scales = [rng.uniform(0.05, 6.0) for _ in range(d)]
                    rows = _ref_assemble(pts, scales, 2.3, noise, kernel)[0]
                    self.assertEqual(
                        _run(lambda: _fast_cholesky(rows)),
                        _run(lambda: _ref_fast_cholesky(rows)),
                    )

    def test_handmade_matrices(self) -> None:
        """Empty, non-positive, nan, inf, short, long and tuple rows."""
        cases: list[Any] = [
            [],
            [[4.0]],
            [[0.0]],
            [[-1.0]],
            [[math.nan]],
            [[math.inf]],
            [[1.0], [2.0, 1.0]],
            [[4.0], [2.0, 5.0], [1.0, 0.5, 3.0]],
            [[1.0], [1.0, 1.0]],
            [[1.0], [math.nan, 2.0]],
            [[1.0], [math.inf, 2.0]],
            [[1e300], [1e300, 1e300]],
            [[1.0], [2.0]],
            [[1.0, 9.0], [2.0, 5.0]],
            [[4.0, 0.0, 0.0], [2.0, 5.0, 0.0], [1.0, 0.5, 3.0]],
            ((4.0,), (2.0, 5.0), (1.0, 0.5, 3.0)),
            [[4.0], [2.0, 5.0], []],
            [[4.0], [], [1.0, 0.5, 3.0]],
            [[4.0], [2.0, "x"]],
            [[4.0], [2.0, None]],
            [[_FloatSub(4.0)], [2.0, _FloatSub(5.0)]],
            [[1e-320], [1e-320, 1e-320]],
            [[-0.0]],
        ]
        for matrix in cases:
            self.assertEqual(
                _run(lambda: _fast_cholesky(matrix)),
                _run(lambda: _ref_fast_cholesky(matrix)),
            )

    def test_random_spd(self) -> None:
        """Random symmetric positive definite matrices of several sizes."""
        rng = random.Random(612)
        for n in (3, 8, 17, 40):
            b = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
            rows = [
                [
                    math.fsum(b[i][k] * b[j][k] for k in range(n))
                    + (0.01 if i == j else 0.0)
                    for j in range(i + 1)
                ]
                for i in range(n)
            ]
            self.assertEqual(
                _run(lambda: _fast_cholesky(rows)),
                _run(lambda: _ref_fast_cholesky(rows)),
            )


class InverseFactorEquivalence(unittest.TestCase):
    """``_inverse_factor_columns`` inverts the factor identically."""

    def test_factors_of_kernel_matrices(self) -> None:
        """Row-list factors and zero-padded square factors."""
        rng = random.Random(621)
        for kernel in KERNELS:
            for n, d in ((1, 1), (2, 1), (7, 2), (22, 3), (35, 2)):
                pts = _points(rng, n, d)
                scales = [rng.uniform(0.1, 4.0) for _ in range(d)]
                rows = _ref_assemble(pts, scales, 1.4, 1e-6, kernel)[0]
                lower = _ref_fast_cholesky(rows)
                padded = tuple(tuple(lower[i]) + (0.0,) * (n - 1 - i) for i in range(n))
                for factor in (lower, padded, tuple(map(tuple, lower))):
                    self.assertEqual(
                        _run(lambda: _inverse_factor_columns(factor)),
                        _run(lambda: _ref_inverse_factor_columns(factor)),
                    )

    def test_degenerate_factors(self) -> None:
        """Empty, zero diagonal, nan, inf, ragged and non-float entries."""
        cases: list[Any] = [
            [],
            [[2.0]],
            [[0.0]],
            [[-0.0]],
            [[math.nan]],
            [[math.inf]],
            [[1.0], [1.0, 0.0]],
            [[1.0], [math.nan, 2.0]],
            [[1.0], [1.0, 2.0], [3.0, 4.0, 5.0]],
            [[1.0], [1.0, 2.0, 9.0], [3.0, 4.0, 5.0]],
            [[1.0], [1.0], [3.0, 4.0, 5.0]],
            [[1.0], [1.0, 2.0], [3.0]],
            [[1.0], [1.0, "x"]],
            [[2], [1, 3]],
            [[Fraction(1, 2)], [1, Fraction(1, 3)]],
            [[1e-300], [1e300, 1e-300], [1e300, 1e300, 1e-300]],
            [[1e300], [1e300, 1e300], [1e300, 1e300, 1e300]],
        ]
        for factor in cases:
            self.assertEqual(
                _run(lambda: _inverse_factor_columns(factor)),
                _run(lambda: _ref_inverse_factor_columns(factor)),
            )


# ---------------------------------------------------------------- likelihood


class LikelihoodEquivalence(unittest.TestCase):
    """The log marginal likelihood and its gradient are unchanged."""

    def _same(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        reference = _run(lambda: _ref_log_marginal_likelihood(*args, **kwargs))
        current = _run(
            lambda: gaussian_process_log_marginal_likelihood(*args, **kwargs)
        )
        self.assertEqual(current, reference)
        return current

    def test_values_and_gradients(self) -> None:
        """Every kernel, with and without estimated noise and relative nugget."""
        rng = random.Random(631)
        statuses = set()
        for kernel in KERNELS:
            for n, d in ((2, 1), (5, 2), (14, 3), (21, 2), (28, 4)):
                for noise in (False, True):
                    for tau in (0.0, 1e-6):
                        pts = _points(rng, n, d)
                        y = _targets(rng, pts)
                        theta = _theta(d, noise, rng)
                        out = self._same(
                            pts,
                            y,
                            theta,
                            kernel=kernel,
                            estimate_noise=noise,
                            relative_nugget=tau,
                            nugget=rng.choice((1e-10, 1e-6, 1e-2)),
                        )
                        statuses.add(out[0])
        self.assertEqual(statuses, {"ok"})

    def test_near_singular_matrices(self) -> None:
        """Duplicates, flat and spiky length scales: accept or reject alike."""
        rng = random.Random(632)
        for kernel in KERNELS:
            for n, d in ((10, 2), (25, 3)):
                for center in (-12.0, -3.0, 0.0, 6.0, 20.0):
                    pts = _points(rng, n, d)
                    y = _targets(rng, pts)
                    theta = [0.0] + [center] * d
                    self._same(pts, y, theta, kernel=kernel, nugget=1e-14)
                    self._same(pts, y, theta, kernel=kernel, nugget=1e-3)

    def test_input_types_and_failures(self) -> None:
        """Bad points, targets, parameters, kernel names and nuggets."""
        rng = random.Random(633)
        pts = _points(rng, 9, 2)
        y = _targets(rng, pts)
        theta = [0.2, -0.4, -0.1]
        int_points = [[int(10 * v) for v in p] for p in pts]
        generator_points = (tuple(p) for p in pts)
        generator_targets = (v for v in y)
        subclass_points = [[_FloatSub(v) for v in p] for p in pts]
        cases: list[tuple[Any, Any, Any, dict[str, Any]]] = [
            (int_points, y, theta, {}),
            (generator_points, generator_targets, theta, {}),
            (subclass_points, [_FloatSub(v) for v in y], theta, {}),
            (pts, y, tuple(theta), {}),
            ([[True, False]] * 3, [1.0, 2.0, 3.0], theta, {}),
            ([*pts[:-1], [math.nan, 0.1]], y, theta, {}),
            ([*pts[:-1], [0.1, math.inf]], y, theta, {}),
            ([*pts[:-1], [0.1, "a"]], y, theta, {}),
            ([*pts[:-1], [0.1]], y, theta, {}),
            ([*pts[:-1], [0.1, 0.2, 0.3]], y, theta, {}),
            ([*pts[:-1], None], y, theta, {}),
            (7, y, theta, {}),
            ([], y, theta, {}),
            ([[]], [1.0], [0.0], {}),
            (pts, y[:-1], theta, {}),
            (pts, [*y, 1.0], theta, {}),
            (pts, [*y[:-1], math.nan], theta, {}),
            (pts, [*y[:-1], "x"], theta, {}),
            (pts, [*y[:-1], 5], theta, {}),
            (pts, [*y[:-1], True], theta, {}),
            (pts, None, theta, {}),
            (pts, y, theta[:-1], {}),
            (pts, y, [*theta, 0.0], {}),
            (pts, y, [0.2, 26.0, 0.0], {}),
            (pts, y, [0.2, -25.5, 0.0], {}),
            (pts, y, [math.nan, 0.0, 0.0], {}),
            (pts, y, theta, {"estimate_noise": True}),
            (pts, y, theta, {"kernel": "rbf"}),
            (pts, y, theta, {"relative_nugget": -1.0}),
            (pts, y, theta, {"relative_nugget": math.nan}),
            (pts, y, theta, {"relative_nugget": "a"}),
            (pts, y, theta, {"nugget": 0.0}),
            (pts, y, theta, {"nugget": -1e-3}),
            (pts, y, theta, {"nugget": math.nan}),
            ([pts[0]] * 5, y[:5], theta, {"nugget": 0.0}),
            ([pts[0]] * 5, y[:5], theta, {"nugget": 1e-12}),
        ]
        for points, targets, parameters, options in cases:
            if hasattr(points, "__next__"):
                points = list(points)
                targets = list(targets)
            self._same(points, targets, parameters, **options)

    def test_single_start_values_are_stable_under_repetition(self) -> None:
        """No state leaks between calls: the same call twice agrees bitwise."""
        rng = random.Random(634)
        pts = _points(rng, 16, 3)
        y = _targets(rng, pts)
        theta = [0.1, -0.5, -0.2, 0.3, -6.0]
        first = _run(
            lambda: gaussian_process_log_marginal_likelihood(
                pts, y, theta, estimate_noise=True
            )
        )
        second = _run(
            lambda: gaussian_process_log_marginal_likelihood(
                pts, y, theta, estimate_noise=True
            )
        )
        self.assertEqual(first, second)


# ------------------------------------------------------------------- predict


def _models() -> list[GaussianProcessRegression]:
    """Return fitted models (fixed hyperparameters) of every kernel."""
    out = []
    rng = random.Random(641)
    for kernel in KERNELS:
        for n, d, noise in ((14, 2, False), (9, 3, True), (6, 1, False)):
            pts = _points(rng, n, d)
            # Distinct points keep the kernel matrix positive definite.
            pts = [[v + 1e-3 * i for v in p] for i, p in enumerate(pts)]
            y = _targets(rng, pts)
            theta = _theta(d, noise, rng)
            out.append(
                fit_gaussian_process(
                    pts,
                    y,
                    kernel=kernel,
                    optimize=False,
                    initial_log_parameters=theta,
                    estimate_noise=noise,
                )
            )
    return out


class PredictEquivalence(unittest.TestCase):
    """``predict_with_variance`` and the fit-time factorization are unchanged."""

    def test_factorization_matches_the_reference(self) -> None:
        """Alpha, the factor, the noise and the condition number agree."""
        for model in _models():
            reference = _ref_standardized(model)
            self.assertEqual(_hex(model._factorization), _hex(reference))
            signal, scales, noise, lower, _ = reference
            rows = _ref_assemble(model.points, scales, signal, noise, model.kernel)[0]
            self.assertEqual(
                model.condition_number.hex(), _ref_condition_number(rows, lower).hex()
            )

    def test_predictions(self) -> None:
        """Random, training, far, tiny and non-float queries, both noise modes."""
        rng = random.Random(642)
        for model in _models():
            d = len(model.points[0])
            queries: list[Any] = [list(p) for p in model.points[:3]]
            queries += [[rng.random() for _ in range(d)] for _ in range(12)]
            queries += [[rng.uniform(-50.0, 50.0) for _ in range(d)] for _ in range(3)]
            queries += [[1e-9 * rng.random() for _ in range(d)] for _ in range(2)]
            queries += [[-0.0] * d, [0.0] * d, [3] * d, [True] * d]
            queries += [tuple(rng.random() for _ in range(d))]
            queries += [[_FloatSub(0.3)] * d, [Fraction(1, 3)] * d]
            queries += [[1e150] * d, [-1e154] * d, [1e200] * d, [1e300] * d]
            for query in queries:
                for noise in (False, True):
                    self.assertEqual(
                        _run(
                            lambda: model.predict_with_variance(
                                query, include_noise=noise
                            )
                        ),
                        _run(lambda: _ref_predict_with_variance(model, query, noise)),
                    )

    def test_invalid_queries(self) -> None:
        """Wrong lengths, nan, inf, strings, None and exhausted iterators."""
        model = _models()[0]
        d = len(model.points[0])
        queries: list[Any] = [
            [],
            [0.5] * (d + 1),
            [0.5] * (d - 1),
            [math.nan] * d,
            [math.inf] * d,
            [-math.inf] * d,
            ["a"] * d,
            [None] * d,
            "ab",
            None,
            7,
            [[0.5]] * d,
            iter([0.4] * d),
            (0.4 for _ in range(d + 2)),
        ]
        for query in queries:
            current_query = list(query) if hasattr(query, "__next__") else query
            self.assertEqual(
                _run(lambda: model.predict_with_variance(current_query)),
                _run(lambda: _ref_predict_with_variance(model, current_query)),
            )

    def test_unknown_kernel_raises_the_reference_error(self) -> None:
        """A model whose kernel name is invalid reports the same error."""
        model = _models()[0]
        broken = dataclasses.replace(model, kernel="no_such_kernel")
        self.assertEqual(
            _run(lambda: broken.predict_with_variance([0.3, 0.4])),
            _run(lambda: _ref_predict_with_variance(broken, [0.3, 0.4])),
        )

    def test_ragged_training_points_raise_the_reference_error(self) -> None:
        """A model built on ragged points reports the same errors."""
        model = _models()[0]
        points = [list(p) for p in model.points]
        points[3] = points[3][:1]
        scales = model._factorization[1]
        reference = _run(lambda: _ref_assemble(points, scales, 1.0, 0.0, model.kernel))
        current = _run(lambda: dataclasses.replace(model, points=tuple(points)))
        self.assertEqual(reference[0], "error")
        self.assertEqual(current[:4], reference[:4])

    def test_leave_one_out_and_aliases(self) -> None:
        """``predict`` is the first component; the LOO diagnostics are stable."""
        model = _models()[1]
        for query in ([0.2, 0.4, 0.6], [0.9, 0.1, 0.5]):
            self.assertEqual(
                model.predict(query).hex(),
                _ref_predict_with_variance(model, query)[0].hex(),
            )
        loo = model.leave_one_out()
        self.assertEqual(len(loo.means), len(model.points))


# -------------------------------------------------------------------- golden


def _outcome(call: Callable[[], Any]) -> Any:
    """Run ``call`` and return its result with the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(call)
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def _fit_digest(model: GaussianProcessRegression, queries: list[list[float]]) -> Any:
    """Return what the fit and its use are expected to reproduce."""
    loo = model.leave_one_out()
    predictions = []
    for query in queries:
        predictions.append(model.predict_with_variance(query))
        predictions.append(model.predict_with_variance(query, include_noise=True))
    return (
        model,
        loo,
        predictions,
        model.condition_number,
        model.standardized_noise_variance,
    )


def _golden_runs() -> dict[str, str]:
    """Digest whole GP fits (optimizer path included)."""
    out: dict[str, str] = {}
    rng = random.Random(651)
    configs: list[tuple[str, int, int, dict[str, Any]]] = [
        (
            "m52_noise_2start",
            18,
            2,
            {"kernel": "matern52", "estimate_noise": True, "restarts": 2, "seed": 1},
        ),
        ("se_3d", 16, 3, {"kernel": "squared_exponential", "restarts": 1, "seed": 2}),
        (
            "m32_reg",
            20,
            1,
            {
                "kernel": "matern32",
                "restarts": 1,
                "seed": 3,
                "regularization": GaussianProcessRegularization(),
            },
        ),
        (
            "m52_prior",
            15,
            2,
            {
                "kernel": "matern52",
                "estimate_noise": True,
                "restarts": 2,
                "seed": 4,
                "regularization": GaussianProcessRegularization(
                    length_scale_prior_sd=1.0
                ),
            },
        ),
        (
            "m52_fixed",
            12,
            2,
            {
                "kernel": "matern52",
                "optimize": False,
                "initial_log_parameters": (0.0, -0.5, -0.5),
            },
        ),
        ("m52_nugget", 14, 2, {"kernel": "matern52", "nugget": 1e-6, "seed": 5}),
    ]
    for name, n, d, options in configs:
        pts = _points(rng, n, d)
        pts = [[v + 1e-3 * i for v in p] for i, p in enumerate(pts)]
        y = _targets(rng, pts)
        queries = [[rng.random() for _ in range(d)] for _ in range(4)]
        out[name] = _digest(
            _outcome(
                lambda: _fit_digest(fit_gaussian_process(pts, y, **options), queries)
            )
        )
    duplicated = [[0.1 * i, 0.05 * (i % 3)] for i in range(8)]
    duplicated += duplicated[:3]
    y_dup = [math.sin(p[0]) + p[1] for p in duplicated]
    out["duplicates_adaptive"] = _digest(
        _outcome(lambda: fit_gaussian_process(duplicated, y_dup, seed=1, restarts=1))
    )
    out["duplicates_strict"] = _digest(
        _outcome(
            lambda: fit_gaussian_process(
                duplicated,
                y_dup,
                seed=1,
                restarts=1,
                adaptive_nugget=False,
                nugget=1e-16,
            )
        )
    )
    out["constant_values"] = _digest(
        _outcome(lambda: fit_gaussian_process([[0.0], [1.0], [2.0]], [1.0, 1.0, 1.0]))
    )
    out["too_few_points"] = _digest(
        _outcome(lambda: fit_gaussian_process([[0.0]], [1.0]))
    )
    out["bad_kernel"] = _digest(
        _outcome(lambda: fit_gaussian_process([[0.0], [1.0]], [1.0, 2.0], kernel="rbf"))
    )
    return out


GOLDEN: dict[str, str] = {
    "m52_noise_2start": "b76e76e92ee0576e",
    "se_3d": "62c0955b9e45efdd",
    "m32_reg": "f0c17a16485bdf3a",
    "m52_prior": "a38591df06525e4c",
    "m52_fixed": "3d2e5cae14188d19",
    "m52_nugget": "101080e53b072083",
    "duplicates_adaptive": "e210651a10d09356",
    "duplicates_strict": "63234b0962215fa1",
    "constant_values": "8e289b99bd407d0b",
    "too_few_points": "4afaaec3d1b60ac0",
    "bad_kernel": "930d685ee4c74fd4",
}


class GoldenTests(unittest.TestCase):
    """Whole-fit digests recorded from the reference revision."""

    def test_fit_workloads(self) -> None:
        """Every fit reproduces the digest of revision ``315d501``."""
        self.assertEqual(_golden_runs(), GOLDEN)


if __name__ == "__main__":
    import sys

    if "--print-golden" in sys.argv:
        import json

        print(json.dumps(_golden_runs(), indent=4))
    else:
        unittest.main()
