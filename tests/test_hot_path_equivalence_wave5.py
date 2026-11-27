"""Bit-for-bit equivalence of the fifth wave of performance rewrites.

Covered: the exact-``float`` fast path of ``_vector`` (and so of every
operator input and output check), the single-validation ``matvec`` /
``rmatvec`` of ``CountingOperator`` over a plain ``OperatorBase``, and the lazy
iterate of ``lsmr`` in ``probstat.linalg.krylov``; the golden digests also pin
whole ``cgls``, ``lsqr``, ``lsmr``, ``conjugate_gradient``,
``lanczos_eigenpairs``, ``power_iteration``, ``hutchinson_trace`` and
``randomized_svd`` runs.

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(``probstat.linalg.operators._vector``, ``CountingOperator`` and
``probstat.linalg.krylov.lsmr``), copied verbatim apart from removed
docstrings, type annotations and the ``_ref_`` prefix. Every test feeds
fixed-seed and adversarial inputs (NaN, inf, ``-0.0``, ``bool``, ``int`` and
``float`` subclasses, huge integers, strings, wrong lengths, overflowing sums,
bad operator outputs, overridden ``matvec``) to the current and to the
reference routine and compares floats by ``float.hex``, and exceptions and
operator counts by type, message and value. The golden digests at the end were
produced by the checkout of that revision (``--print-golden`` with its
``src`` on ``PYTHONPATH``) for whole solver workloads.

References:
    [REF-PROBSTAT-VALIDATION]. Equivalence harness only; no equation is
    evaluated.

"""

# ruff: noqa: B023 - every lambda below is called immediately inside its loop
from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import random
import unittest
from collections.abc import Callable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

from probstat.linalg import (
    CallbackOperator,
    ComposedOperator,
    Convolution1DOperator,
    CountingOperator,
    DenseOperator,
    DiagonalOperator,
    IdentityOperator,
    OperatorBase,
    ScaledOperator,
    TransposedOperator,
    ZeroOperator,
    cgls,
    conjugate_gradient,
    hutchinson_trace,
    lanczos_eigenpairs,
    lsmr,
    lsqr,
    power_iteration,
    randomized_svd,
)
from probstat.linalg.krylov import (
    _axpy,
    _finish,
    _norm,
    _require_positive_int,
    _require_tolerance,
    _scale,
    _setup,
    _sym_ortho,
)
from probstat.linalg.operators import _vector

# ---------------------------------------------------------------- references


def _ref_vector(values, size, name):
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a sequence of real numbers") from error
    if len(vector) != size:
        raise ValueError(f"{name} has length {len(vector)}, expected {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _ref_apply(operator: Any, vector: Any) -> Any:
    if isinstance(operator, OperatorBase):
        return operator.matvec(vector)
    return _ref_vector(operator.matvec(vector), operator.shape[0], "operator output")


def _ref_apply_transpose(operator: Any, vector: Any) -> Any:
    if isinstance(operator, OperatorBase):
        return operator.rmatvec(vector)
    return _ref_vector(
        operator.rmatvec(vector), operator.shape[1], "operator transpose output"
    )


class _RefCounting:
    """The original ``CountingOperator`` (two checks on each side, no shortcut)."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.forward_calls = 0
        self.transpose_calls = 0
        self.shape = tuple(base.shape)
        self.name = f"counted {getattr(base, 'name', 'operator')}"

    @property
    def counts(self) -> tuple[int, int]:
        return (self.forward_calls, self.transpose_calls)

    def matvec(self, vector: Any) -> Any:
        rows, columns = self.shape
        checked = _ref_vector(vector, columns, f"{self.name} input")
        self.forward_calls += 1
        result = _ref_apply(self.base, checked)
        return _ref_vector(result, rows, f"{self.name} output")

    def rmatvec(self, vector: Any) -> Any:
        rows, columns = self.shape
        checked = _ref_vector(vector, rows, f"{self.name} transpose input")
        self.transpose_calls += 1
        result = _ref_apply_transpose(self.base, checked)
        return _ref_vector(result, columns, f"{self.name} transpose output")


def _ref_lsmr(
    operator,
    data,
    *,
    damping=0.0,
    initial=None,
    atol=1e-10,
    btol=1e-10,
    condition_limit=100000000.0,
    max_iterations=None,
    callback=None,
):
    counted, b, x0 = _setup(operator, data, initial)
    damp = _require_tolerance(damping, "damping")
    a_tol = _require_tolerance(atol, "atol")
    b_tol = _require_tolerance(btol, "btol")
    conlim = _require_tolerance(condition_limit, "condition_limit")
    if conlim <= 1.0:
        raise ValueError("condition_limit must exceed one")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    nan = math.nan
    ax0 = counted.matvec(x0)
    u = [bi - ai for bi, ai in zip(b, ax0, strict=True)]
    beta = _norm(u)
    bnorm = _norm(b)
    if beta == 0.0:
        reason = "zero_rhs" if bnorm == 0.0 and _norm(x0) == 0.0 else "exact"
        return _finish(
            counted,
            b,
            list(x0),
            iterations=0,
            reason=reason,
            residual=0.0,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[0.0],
            normals=[0.0],
            method="lsmr",
        )
    u = _scale(1.0 / beta, u)
    v = list(counted.rmatvec(u))
    alpha = _norm(v)
    if alpha == 0.0:
        return _finish(
            counted,
            b,
            list(x0),
            iterations=0,
            reason="exact",
            residual=beta,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[beta],
            normals=[0.0],
            method="lsmr",
        )
    v = _scale(1.0 / alpha, v)
    zetabar = alpha * beta
    alphabar = alpha
    rho = rhobar = cbar = 1.0
    sbar = 0.0
    h = list(v)
    hbar = [0.0] * n
    z = [0.0] * n
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0
    norm_a2 = alpha * alpha
    maxrbar = 0.0
    minrbar = math.inf
    residuals, normals = ([beta], [alpha * beta])
    reason = "max_iterations"
    iterations = 0
    normr = beta
    normar = alpha * beta
    norm_a = cond_a = nan
    x = list(x0)
    for k in range(1, limit + 1):
        u = _axpy(-alpha, u, counted.matvec(v))
        beta = _norm(u)
        if beta > 0.0:
            u = _scale(1.0 / beta, u)
            v = _axpy(-beta, v, counted.rmatvec(u))
            alpha = _norm(v)
            if alpha > 0.0:
                v = _scale(1.0 / alpha, v)
        else:
            alpha = 0.0
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        if rho == 0.0:
            reason = "breakdown"
            break
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        if rhobar == 0.0:
            reason = "breakdown"
            break
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        factor = thetabar * rho / (rhoold * rhobarold)
        hbar = _axpy(-factor, hbar, h)
        z = _axpy(zeta / (rho * rhobar), hbar, z)
        h = _axpy(-thetanew / rho, h, v)
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d += betacheck * betacheck
        augmented = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        norm_a2 += beta * beta
        norm_a = math.sqrt(norm_a2)
        norm_a2 += alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if k > 1:
            minrbar = min(minrbar, rhobarold)
        cond_a = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        x = [a + c2 for a, c2 in zip(x0, z, strict=True)]
        znorm = _norm(z)
        normr = math.sqrt(max(augmented * augmented - damp * damp * znorm * znorm, 0.0))
        residuals.append(normr)
        normals.append(normar)
        iterations = k
        if callback is not None and callback(k, tuple(x), normr):
            reason = "callback"
            break
        if augmented <= b_tol * bnorm + a_tol * norm_a * znorm:
            reason = "residual_tolerance"
            break
        if augmented == 0.0 or normar <= a_tol * norm_a * augmented:
            reason = "normal_tolerance"
            break
        if cond_a >= conlim:
            reason = "condition_limit"
            break
        if alpha == 0.0:
            reason = "normal_tolerance"
            break
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=normr,
        normal=normar,
        damping=damp,
        norm_estimate=norm_a,
        condition=cond_a,
        residuals=residuals,
        normals=normals,
        method="lsmr",
    )


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


class _TupleSub(tuple):  # type: ignore[type-arg]
    """A ``tuple`` subclass."""


class _LyingFloat(float):
    """A ``float`` whose ``__float__`` disagrees with its stored value."""

    def __float__(self) -> float:
        return 7.0


def _bad_vectors(n: int) -> list[Any]:
    """Return adversarial candidates for a vector of length ``n``."""
    good = [0.25 * (i + 1) for i in range(n)]
    nan, inf = float("nan"), float("inf")
    last = good[:-1]
    cases: list[Any] = [
        good,
        tuple(good),
        [int(v * 4) for v in good],
        [*last, nan] if n else [nan],
        [inf, *good[1:]] if n else [inf],
        [-inf, *good[1:]] if n else [-inf],
        [*last, True] if n else [True],
        [*last, "1.0"] if n else ["1.0"],
        [*last, None] if n else [None],
        [*last, Fraction(1, 3)] if n else [Fraction(1, 3)],
        [*last, Decimal("0.5")] if n else [Decimal("0.5")],
        [*last, 10**400] if n else [10**400],
        [nan, *good[1:-1], 10**400] if n > 1 else [nan],
        [10**400, *good[1:-1], nan] if n > 1 else [10**400],
        [nan, *good[1:-1], "a"] if n > 1 else [nan],
        ["a", *good[1:-1], nan] if n > 1 else ["a"],
        [_FloatSub(v) for v in good],
        tuple(_FloatSub(v) for v in good),
        [_IntSub(int(v * 4)) for v in good],
        [_LyingFloat(v) for v in good],
        _ListSub(good),
        _TupleSub(good),
        [*good, 1.0],
        last,
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
        (-0.0,) * n,
        [1e308] * n,
        (1e308,) * n,
        [1e308, -1e308] * (n // 2) + [1e308] * (n % 2),
        [1e308, 1e308, -1e308, -1e308][:n] + [0.5] * max(n - 4, 0),
        [inf, -inf] * (n // 2) + [inf] * (n % 2),
        [nan] * n,
        [5e-324] * n,
        [0.1] * n,
        [[1.0]] * n,
        {i: 1.0 for i in range(n)},
        [float(i) for i in range(n)],
        tuple(float(i) - 1.5 for i in range(n)),
    ]
    return cases


# ---------------------------------------------------------------- validation


class VectorValidationTests(unittest.TestCase):
    """``_vector`` keeps every result and every error."""

    def test_adversarial_inputs(self) -> None:
        """Results (value, type) and errors (type, message, cause) are identical."""
        for n in (0, 1, 2, 3, 6):
            for size in sorted({0, 1, max(n - 1, 0), n, n + 1, 5}):
                new_cases = _bad_vectors(n)
                ref_cases = _bad_vectors(n)
                for index, (new, ref) in enumerate(
                    zip(new_cases, ref_cases, strict=True)
                ):
                    with self.subTest(n=n, size=size, case=index):
                        got = _run(lambda: _vector(new, size, "v"))
                        want = _run(lambda: _ref_vector(ref, size, "v"))
                        self.assertEqual(got, want)

    def test_random_vectors(self) -> None:
        """Random float lists and tuples of many lengths agree bit for bit."""
        rng = random.Random(5)
        for _ in range(300):
            n = rng.randint(0, 40)
            values = _floats(rng, n, 10.0 ** rng.randint(-300, 290))
            for candidate in (values, tuple(values)):
                got = _run(lambda: _vector(candidate, n, "v"))
                want = _run(lambda: _ref_vector(candidate, n, "v"))
                self.assertEqual(got, want)

    def test_result_is_a_tuple_of_floats_and_input_is_untouched(self) -> None:
        """A list is copied to a tuple, an exact tuple is accepted as it is."""
        source = [1.0, -0.0, 2.5]
        result = _vector(source, 3, "v")
        self.assertIs(type(result), tuple)
        self.assertEqual(result, (1.0, -0.0, 2.5))
        source[0] = 99.0
        self.assertEqual(result[0], 1.0)
        self.assertEqual(_vector((), 0, "v"), ())
        self.assertEqual(_vector([], 0, "v"), ())
        mixed = _vector([1, 2.5], 2, "v")
        self.assertEqual([type(v) for v in mixed], [float, float])

    def test_overflowing_totals_are_still_accepted(self) -> None:
        """Finite entries whose sum overflows pass, like the entrywise check."""
        self.assertEqual(_vector([1e308, 1e308], 2, "v"), (1e308, 1e308))
        self.assertEqual(_vector((1.7e308, 1.7e308, -1.7e308), 3, "v")[0], 1.7e308)
        for bad in ([1e308, 1e308, math.inf], [math.nan, 1e308, 1e308]):
            with self.assertRaises(ValueError):
                _vector(bad, 3, "v")


# ------------------------------------------------------------ counting wrapper


class _Foreign:
    """A ``LinearOperator`` that is not an ``OperatorBase``."""

    def __init__(self, matrix: Sequence[Sequence[float]], output: Any = None) -> None:
        self.matrix = matrix
        self.output = output
        self.name = "foreign"

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def matvec(self, vector: Sequence[float]) -> Any:
        if self.output is not None:
            return self.output(self.shape[0])
        return [
            sum(a * b for a, b in zip(row, vector, strict=False)) for row in self.matrix
        ]

    def rmatvec(self, vector: Sequence[float]) -> Any:
        if self.output is not None:
            return self.output(self.shape[1])
        columns = range(self.shape[1])
        return [
            sum(row[j] * v for row, v in zip(self.matrix, vector, strict=False))
            for j in columns
        ]


class _Scripted(OperatorBase):
    """An ``OperatorBase`` whose raw action output is scripted."""

    __slots__ = ("script",)

    def __init__(self, rows: int, columns: int, script: Callable[[int], Any]) -> None:
        super().__init__(rows, columns, name="scripted")
        self.script = script

    def _forward(self, vector: Any) -> Any:
        return self.script(self.shape[0])

    def _transpose(self, vector: Any) -> Any:
        return self.script(self.shape[1])


class _OwnMatvec(OperatorBase):
    """An ``OperatorBase`` that overrides ``matvec`` and ``rmatvec``."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(3, 3, name="own")

    def matvec(self, vector: Any) -> Any:
        return [2 * v for v in vector]

    def rmatvec(self, vector: Any) -> Any:
        return [float("nan")] * 3


class _Patched(OperatorBase):
    """An ``OperatorBase`` with an instance dictionary that can be patched."""

    def __init__(self) -> None:
        super().__init__(3, 3, name="patched")

    def _forward(self, vector: Any) -> Any:
        return [v + 1.0 for v in vector]

    def _transpose(self, vector: Any) -> Any:
        return [v - 1.0 for v in vector]


class _Raising(OperatorBase):
    """An ``OperatorBase`` whose actions raise."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(3, 3, name="raising")

    def _forward(self, vector: Any) -> Any:
        raise RuntimeError("forward failed")

    def _transpose(self, vector: Any) -> Any:
        raise KeyError("transpose failed")


def _outputs() -> dict[str, Callable[[int], Any]]:
    """Return scripted raw outputs of length ``n`` (and wrong ones)."""
    nan, inf = float("nan"), float("inf")
    return {
        "good_list": lambda n: [0.5 * i for i in range(n)],
        "good_tuple": lambda n: tuple(0.5 * i for i in range(n)),
        "ints": lambda n: list(range(n)),
        "nan": lambda n: [nan] * n,
        "inf": lambda n: [0.0] * (n - 1) + [inf],
        "short": lambda n: [0.5] * (n - 1),
        "long": lambda n: [0.5] * (n + 1),
        "strings": lambda n: ["1.5"] * n,
        "words": lambda n: ["a"] * n,
        "none": lambda n: None,
        "scalar": lambda n: 2.5,
        "generator": lambda n: (0.25 * i for i in range(n)),
        "bools": lambda n: [True] * n,
        "float_sub": lambda n: [_FloatSub(0.5)] * n,
        "list_sub": lambda n: _ListSub([0.5] * n),
        "lying": lambda n: [_LyingFloat(0.5)] * n,
        "huge": lambda n: [10**400] * n,
        "overflow_sum": lambda n: [1e308] * n,
        "neg_zero": lambda n: [-0.0] * n,
    }


def _wrapped_bases() -> list[tuple[str, Callable[[], Any]]]:
    """Return named factories of bases to wrap in a ``CountingOperator``."""
    rng = random.Random(41)
    matrix = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(3)) for _ in range(4))
    square = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(3)) for _ in range(3))
    bases: list[tuple[str, Callable[[], Any]]] = [
        ("dense", lambda: DenseOperator(matrix)),
        ("dense_square", lambda: DenseOperator(square)),
        ("identity", lambda: IdentityOperator(3)),
        ("diagonal", lambda: DiagonalOperator((1.0, -2.0, 0.5))),
        ("zero", lambda: ZeroOperator(2, 3)),
        ("conv", lambda: Convolution1DOperator((0.25, 0.5, 0.25), 3)),
        (
            "conv_reflect",
            lambda: Convolution1DOperator((1.0, 2.0), 3, boundary="reflect"),
        ),
        ("scaled", lambda: ScaledOperator(DenseOperator(matrix), 1e200)),
        ("transposed", lambda: TransposedOperator(DenseOperator(matrix))),
        (
            "composed",
            lambda: ComposedOperator([DenseOperator(matrix), DenseOperator(square)]),
        ),
        ("counting", lambda: CountingOperator(DenseOperator(matrix))),
        (
            "counting_twice",
            lambda: CountingOperator(CountingOperator(IdentityOperator(3))),
        ),
        (
            "callback",
            lambda: CallbackOperator(
                (3, 3), lambda v: [x * 3.0 for x in v], lambda v: [x / 3.0 for x in v]
            ),
        ),
        (
            "callback_overflow",
            lambda: CallbackOperator(
                (3, 3), lambda v: [x * 1e308 * 10 for x in v], lambda v: list(v)
            ),
        ),
        ("foreign", lambda: _Foreign(matrix)),
        ("own_matvec", lambda: _OwnMatvec()),
        ("raising", lambda: _Raising()),
    ]
    for key, script in _outputs().items():
        bases.append((f"scripted_{key}", lambda s=script: _Scripted(3, 3, s)))
        bases.append((f"foreign_{key}", lambda s=script: _Foreign(square, s)))
    return bases


def _instance_patched() -> Any:
    """Return an operator whose ``matvec``/``rmatvec`` are set on the instance."""
    base = _Patched()
    base.matvec = lambda v: [float("nan")] * 3  # type: ignore[method-assign]
    base.rmatvec = lambda v: [float("inf")] * 3  # type: ignore[method-assign]
    return base


class CountingOperatorTests(unittest.TestCase):
    """``CountingOperator`` results, errors and counts are unchanged."""

    def _compare(self, name: str, make: Callable[[], Any], reshape: bool) -> None:
        base_new, base_ref = make(), make()
        new = CountingOperator(base_new)
        ref = _RefCounting(base_ref)
        if reshape:
            base_new._shape = (3, 4)
            base_ref._shape = (3, 4)
        rows, columns = new.shape
        self.assertEqual(new.shape, ref.shape)
        self.assertEqual(new.name, ref.name)
        for size, method in ((columns, "matvec"), (rows, "rmatvec")):
            cases_new = _bad_vectors(size)
            cases_ref = _bad_vectors(size)
            for index, (vn, vr) in enumerate(zip(cases_new, cases_ref, strict=True)):
                with self.subTest(base=name, method=method, case=index):
                    got = _run(lambda: getattr(new, method)(vn))
                    want = _run(lambda: getattr(ref, method)(vr))
                    self.assertEqual(got, want)
                    counts = new.counts
                    self.assertEqual((counts.forward, counts.transpose), ref.counts)

    def test_wrapped_operators_and_bad_inputs(self) -> None:
        """Every base kind with every adversarial input (and scripted output)."""
        for name, make in _wrapped_bases():
            self._compare(name, make, reshape=False)

    def test_patched_matvec_and_changed_shape(self) -> None:
        """An instance-level ``matvec`` and a shape change take the old path."""
        self._compare("instance_patched", _instance_patched, reshape=False)
        self._compare("reshaped", _Patched, reshape=True)

    def test_repeated_calls_accumulate_identical_counts(self) -> None:
        """Counts after a mixed call sequence equal the reference's."""
        rng = random.Random(8)
        matrix = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(5)) for _ in range(7))
        base = DenseOperator(matrix)
        new = CountingOperator(base)
        ref = _RefCounting(base)
        for step in range(30):
            if step % 3 == 2:
                vector, method = _floats(rng, 7), "rmatvec"
            else:
                vector, method = _floats(rng, 5), "matvec"
            if step % 7 == 6:
                vector[0] = math.nan
            self.assertEqual(
                _run(lambda: getattr(new, method)(vector)),
                _run(lambda: getattr(ref, method)(vector)),
            )
            self.assertEqual((new.counts.forward, new.counts.transpose), ref.counts)


# ------------------------------------------------------------------ solvers


def _problem(kind: str) -> tuple[Any, tuple[float, ...]]:
    """Return an operator and a data vector."""
    rng = random.Random(15)
    if kind == "dense":
        matrix = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(9)) for _ in range(20))
        return DenseOperator(matrix), tuple(_floats(rng, 20))
    if kind == "convolution":
        operator = Convolution1DOperator((0.1, 0.4, 0.4, 0.1), 60, boundary="reflect")
        truth = tuple(math.sin(0.2 * i) for i in range(60))
        data = tuple(v + 1e-3 * rng.gauss(0.0, 1.0) for v in operator.matvec(truth))
        return operator, data
    if kind == "square":
        matrix = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(8)) for _ in range(8))
        return DenseOperator(matrix), tuple(_floats(rng, 8))
    if kind == "rank_deficient":
        row = tuple(rng.gauss(0.0, 1.0) for _ in range(6))
        matrix = (row, row, tuple(2.0 * v for v in row), (0.0,) * 6, row)
        return DenseOperator(matrix), tuple(_floats(rng, 5))
    return ZeroOperator(4, 4), (1.0, 2.0, 3.0, 4.0)


class LsmrTests(unittest.TestCase):
    """LSMR with its lazily formed iterate against the original loop."""

    def assertSame(self, operator: Any, data: Any, **options: Any) -> None:
        """Assert that ``lsmr`` and the reference agree on one call."""
        got = _run(lambda: lsmr(operator, data, **options))
        want = _run(lambda: _ref_lsmr(operator, data, **options))
        self.assertEqual(got, want)

    def test_results(self) -> None:
        """Every field of the result is bit-identical, with and without extras."""
        rng = random.Random(16)
        for kind in ("dense", "convolution", "square", "rank_deficient", "zero"):
            operator, data = _problem(kind)
            columns = operator.shape[1]
            options: list[dict[str, Any]] = [
                {},
                {"damping": 0.3},
                {"initial": tuple(_floats(rng, columns))},
                {"initial": (-0.0,) * columns, "max_iterations": 1},
                {"initial": (-0.0,) * columns},
                {"max_iterations": 3},
                {"atol": 1e-14, "btol": 1e-14, "condition_limit": 1e12},
                {"condition_limit": 20.0},
                {
                    "damping": 0.1,
                    "initial": tuple(_floats(rng, columns)),
                    "max_iterations": 7,
                },
            ]
            for option in options:
                with self.subTest((kind, tuple(option))):
                    self.assertSame(operator, data, **option)

    def test_callbacks_see_identical_iterates(self) -> None:
        """Callbacks get the same ``x`` at every step and can stop the run."""
        operator, data = _problem("dense")
        initial = (-0.0,) * 9

        def run(solver: Callable[..., Any], stop: int) -> tuple[Any, list[Any]]:
            seen: list[Any] = []

            def callback(k: int, x: Any, rnorm: float) -> bool:
                seen.append((k, _hex(x), rnorm.hex()))
                return k >= stop

            outcome = _run(
                lambda: solver(operator, data, callback=callback, initial=initial)
            )
            return outcome, seen

        for stop in (1, 4, 1000):
            got, got_seen = run(lsmr, stop)
            want, want_seen = run(_ref_lsmr, stop)
            self.assertEqual(got, want)
            self.assertEqual(got_seen, want_seen)

    def test_special_cases_and_errors(self) -> None:
        """Zero data, exact starts, bad arguments and non-finite data agree."""
        operator, data = _problem("dense")
        columns = operator.shape[1]
        self.assertSame(operator, (0.0,) * 20)
        self.assertSame(operator, (0.0,) * 20, initial=(-0.0,) * columns)
        exact = tuple(operator.matvec((1.0,) * columns))
        self.assertSame(operator, exact, initial=(1.0,) * columns)
        self.assertSame(operator, data, damping=-1.0)
        self.assertSame(operator, data, atol=-1.0)
        self.assertSame(operator, data, condition_limit=1.0)
        self.assertSame(operator, data, max_iterations=0)
        self.assertSame(operator, data[:5])
        self.assertSame(operator, [*data[:-1], math.nan])
        self.assertSame(operator, [*data[:-1], "1.5"])
        self.assertSame(operator, [*data[:-1], 10**400])
        self.assertSame(operator, data, initial=(1.0,) * (columns - 1))
        self.assertSame(operator, data, initial=(math.inf,) * columns)
        self.assertSame(operator, data, max_iterations=True)

    def test_random_small_problems(self) -> None:
        """Many tiny random and singular problems, including breakdowns."""
        rng = random.Random(23)
        reasons: set[str] = set()
        for trial in range(80):
            rows, columns = rng.randint(1, 6), rng.randint(1, 6)
            matrix = [
                [rng.gauss(0.0, 1.0) for _ in range(columns)] for _ in range(rows)
            ]
            if trial % 3 == 0 and rows > 1:
                matrix[-1] = list(matrix[0])
            if trial % 5 == 0:
                for row in matrix:
                    row[-1] = 0.0
            operator = DenseOperator(matrix)
            data = tuple(_floats(rng, rows))
            option = {"damping": rng.choice((0.0, 0.0, 0.5)), "max_iterations": 12}
            self.assertSame(operator, data, **option)
            reasons.add(lsmr(operator, data, **option).stop_reason)
        self.assertGreater(len(reasons), 1)


# --------------------------------------------------------------------- golden


def _golden_runs() -> dict[str, str]:
    """Digest whole solver runs through the validated operator paths."""
    out: dict[str, str] = {}
    problems = {kind: _problem(kind) for kind in ("dense", "convolution", "square")}
    spd_rng = random.Random(77)
    b_matrix = [[spd_rng.gauss(0.0, 1.0) for _ in range(12)] for _ in range(12)]
    spd = DenseOperator(
        tuple(
            tuple(
                math.fsum(b_matrix[i][k] * b_matrix[j][k] for k in range(12))
                + (12.0 if i == j else 0.0)
                for j in range(12)
            )
            for i in range(12)
        )
    )
    rhs = tuple(math.cos(0.7 * i) for i in range(12))
    for kind, (operator, data) in problems.items():
        columns = operator.shape[1]
        start = (-0.0,) * columns
        for name, solver in (("cgls", cgls), ("lsqr", lsqr), ("lsmr", lsmr)):
            out[f"{name}_{kind}"] = _digest(_run(lambda: solver(operator, data)))
            out[f"{name}_{kind}_damped"] = _digest(
                _run(lambda: solver(operator, data, damping=0.2, initial=start))
            )
            out[f"{name}_{kind}_short"] = _digest(
                _run(lambda: solver(operator, data, max_iterations=2, initial=start))
            )
            out[f"{name}_{kind}_bad"] = _digest(
                _run(lambda: solver(operator, [*data[:-1], math.nan]))
            )
            seen: list[Any] = []

            def callback(k: int, x: Any, r: float) -> bool:
                seen.append((k, x, r))
                return k >= 3

            out[f"{name}_{kind}_callback"] = _digest(
                (_run(lambda: solver(operator, data, callback=callback)), seen)
            )
    out["cg"] = _digest(_run(lambda: conjugate_gradient(spd, rhs)))
    out["cg_short"] = _digest(
        _run(
            lambda: conjugate_gradient(spd, rhs, max_iterations=3, initial=(-0.0,) * 12)
        )
    )
    out["lanczos"] = _digest(_run(lambda: lanczos_eigenpairs(spd, 7, seed=3)))
    out["lanczos_no_reorth"] = _digest(
        _run(lambda: lanczos_eigenpairs(spd, 5, seed=4, reorthogonalize=False))
    )
    out["power"] = _digest(_run(lambda: power_iteration(problems["dense"][0], seed=2)))
    out["hutchinson"] = _digest(_run(lambda: hutchinson_trace(spd, probes=6, seed=1)))
    out["randomized_svd"] = _digest(
        _run(lambda: randomized_svd(problems["dense"][0], 3, seed=5))
    )
    overflow = CallbackOperator(
        (3, 3), lambda v: [x * 1e300 * 1e10 for x in v], lambda v: list(v)
    )
    out["overflow_operator"] = _digest(
        [_run(lambda: s(overflow, (1.0, 2.0, 3.0))) for s in (cgls, lsqr, lsmr)]
    )
    foreign = _Foreign(((1.0, 2.0), (3.0, 4.0), (5.0, 6.5)))
    out["foreign_operator"] = _digest(
        [_run(lambda: s(foreign, (1.0, 2.0, 3.0))) for s in (cgls, lsqr, lsmr)]
    )
    lying = _Foreign(((1.0, 2.0), (3.0, 4.0)), lambda n: ["x"] * n)
    out["foreign_bad_output"] = _digest(
        [_run(lambda: s(lying, (1.0, 2.0))) for s in (cgls, lsqr, lsmr)]
    )
    return out


GOLDEN: dict[str, str] = {
    "cgls_dense": "20a4b48fa578516f",
    "cgls_dense_damped": "c17fe09797302a51",
    "cgls_dense_short": "b9ff48039157a49a",
    "cgls_dense_bad": "eff3de9cc47282cd",
    "cgls_dense_callback": "b98d58b5459fc22c",
    "lsqr_dense": "79e86bfa4f62fe02",
    "lsqr_dense_damped": "769dfbe48ef72f79",
    "lsqr_dense_short": "a9647542002db80a",
    "lsqr_dense_bad": "eff3de9cc47282cd",
    "lsqr_dense_callback": "c2dbd28b08ff967b",
    "lsmr_dense": "1be04c663b8041fa",
    "lsmr_dense_damped": "101a4eb49f9d5ea8",
    "lsmr_dense_short": "d56cfd9981bd477b",
    "lsmr_dense_bad": "eff3de9cc47282cd",
    "lsmr_dense_callback": "ca52a702f339e26a",
    "cgls_convolution": "7d5f74a8ad20e486",
    "cgls_convolution_damped": "52627f39cacab491",
    "cgls_convolution_short": "02d64a39a7196a48",
    "cgls_convolution_bad": "eff3de9cc47282cd",
    "cgls_convolution_callback": "1c332f29c602b904",
    "lsqr_convolution": "a16de1d02f1c5deb",
    "lsqr_convolution_damped": "7e8b0ea3cffd7610",
    "lsqr_convolution_short": "5c7ed07b6e33cdc6",
    "lsqr_convolution_bad": "eff3de9cc47282cd",
    "lsqr_convolution_callback": "9c5f96f226bf49b6",
    "lsmr_convolution": "b7fa3d69b0e8d129",
    "lsmr_convolution_damped": "560b5305a6ea6110",
    "lsmr_convolution_short": "048a5d1c4b143ff9",
    "lsmr_convolution_bad": "eff3de9cc47282cd",
    "lsmr_convolution_callback": "ff33edaf76074813",
    "cgls_square": "39fcbde46f628f4c",
    "cgls_square_damped": "0751b9157f9757ec",
    "cgls_square_short": "c386e4c899be83ea",
    "cgls_square_bad": "eff3de9cc47282cd",
    "cgls_square_callback": "8260921932eccf76",
    "lsqr_square": "958f1691509b33a7",
    "lsqr_square_damped": "1835399b7d4d22db",
    "lsqr_square_short": "34169567ec461071",
    "lsqr_square_bad": "eff3de9cc47282cd",
    "lsqr_square_callback": "edb5497cede87619",
    "lsmr_square": "0955862b7630e92d",
    "lsmr_square_damped": "724282611217ad57",
    "lsmr_square_short": "b8f2eb8bc24d4805",
    "lsmr_square_bad": "eff3de9cc47282cd",
    "lsmr_square_callback": "59a22a235b33ac3d",
    "cg": "6c9b5c5cb290fa84",
    "cg_short": "4a056c57200b7f55",
    "lanczos": "f8997c3a20bd8594",
    "lanczos_no_reorth": "e5949cf6ed140e4e",
    "power": "5e8a4c0ecaf86f42",
    "hutchinson": "8c3b4e33972809b1",
    "randomized_svd": "a2a61199379bd011",
    "overflow_operator": "d3192bba546143c9",
    "foreign_operator": "7915459486470168",
    "foreign_bad_output": "77a32e6cff803ea9",
}


class GoldenTests(unittest.TestCase):
    """Whole-solver digests recorded from the reference revision."""

    def test_solver_workloads(self) -> None:
        """Every solver run reproduces the digest of revision ``315d501``."""
        self.assertEqual(_golden_runs(), GOLDEN)


if __name__ == "__main__":
    import sys

    if "--print-golden" in sys.argv:
        import json

        print(json.dumps(_golden_runs(), indent=4))
    else:
        unittest.main()
