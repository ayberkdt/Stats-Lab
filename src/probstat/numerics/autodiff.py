"""Forward-mode automatic differentiation with (nested) dual numbers.

A :class:`Dual` carries a value ``a`` and a tangent vector ``b`` and stands
for the truncated Taylor expansion ``a + b . eps`` with ``eps_i eps_j = 0``.
Every arithmetic operation and every elementary function in :class:`admath`
propagates the tangent by the chain rule
``f(a + b eps) = f(a) + f'(a) b eps`` ([EQ-AD-DUAL-001]), so evaluating a
function once on the seeds ``x_i + e_i eps`` returns its value and its whole
gradient (vector-tangent mode, Griewank and Walther 2008, Chapter 3). The
derivatives are exact up to floating-point rounding: there is no step size.

Nesting a dual inside a dual (the value and tangent entries are themselves
duals) differentiates twice; :func:`hessian` uses this forward-over-forward
mode ([EQ-AD-HESSIAN-001]). Each nesting level has its own ``order``; an
operand of lower order is treated as a constant by a higher-order one, which
is correct for the seeds built by this module (do not mix duals created by
independent differentiations of one another).

Status and limits (ADR-0065, experimental inside a stable package)
    * Only code written with ``+ - * / **``, comparisons, ``abs`` and the
      :class:`admath` functions propagates derivatives. ``math.exp(d)``,
      ``float(d)``, ``int(d)``, ``round(d)``, ``math.fsum`` and validators that
      call ``float`` raise :class:`AutodiffUnsupportedError` (a ``TypeError``)
      instead of silently dropping the tangent. Consumers such as the HMC/NUTS
      samplers catch it and fall back to finite differences, recording which
      method they used.
    * Comparisons and ``bool`` use the (innermost) value, so branches select
      a piece of a piecewise function and the derivative of that piece is
      returned. At a kink (``abs`` at 0, ``hypot`` at the origin) the
      derivative is defined as 0; at a pole the float operation raises
      (``ZeroDivisionError``, ``ValueError``) as it does in :mod:`math`.
    * ``digamma`` of a dual needs ``trigamma``; ``trigamma`` of a dual (a third
      derivative of ``lgamma``) is not available and raises.
    * Cost: a vector-tangent evaluation costs about ``d`` times one float
      evaluation in this pure-Python implementation, but makes a *single* call
      of the user function. The Hessian costs about ``d^2`` times a float
      evaluation.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Sequence
from numbers import Real
from typing import Literal, cast

from probstat.foundations.validation import require_finite_real

from .special import digamma as _float_digamma
from .special import trigamma as _float_trigamma

GradientMethod = Literal["auto", "autodiff", "finite-difference"]
GradientCallable = Callable[[tuple[float, ...]], Sequence[float]]
FiniteDifference = Callable[
    [Callable[[tuple[float, ...]], float], tuple[float, ...]], Sequence[float]
]

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_LOG_TWO = math.log(2.0)
_LOG_TEN = math.log(10.0)


class AutodiffUnsupportedError(TypeError):
    """A dual number reached code that cannot propagate derivatives.

    Raised by ``float(dual)`` (and therefore by every :mod:`math` function,
    ``math.fsum`` and float-only validators), by ``int``/``round``/``//``/``%``
    on a dual, and by derivatives this module does not provide. It subclasses
    ``TypeError`` so code that already treats type errors as "unsupported
    input" keeps working.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """


def _unsupported(operation: str) -> AutodiffUnsupportedError:
    """Return the error raised when a dual meets a float-only operation.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    return AutodiffUnsupportedError(
        f"{operation} on a Dual number would discard its derivative. The "
        "differentiated function must use arithmetic operators and the "
        "probstat.numerics.admath functions instead of math.* or float(); "
        "callers with gradient='auto' fall back to finite differences."
    )


def _primal(value: object) -> object:
    """Return the innermost (non-dual) value of a possibly nested dual.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    while isinstance(value, Dual):
        value = value.value
    return value


def _new(value: float | Dual, tangent: tuple[float | Dual, ...], order: int) -> Dual:
    """Build a dual without validation (internal fast path).

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    result = object.__new__(Dual)
    result.value = value
    result.tangent = tangent
    result.order = order
    return result


def _scaled(
    tangent: tuple[float | Dual, ...], factor: float | Dual
) -> tuple[float | Dual, ...]:
    """Return ``factor * tangent`` elementwise.

    References:
        [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    return tuple([factor * entry for entry in tangent])


def _real_pow(base: float | Dual, exponent: float | Dual) -> float | Dual:
    """Return ``base ** exponent``, using ``math.pow`` for two plain reals.

    ``math.pow`` raises ``ValueError`` for a negative base with a non-integer
    exponent where the ``**`` operator would silently return a complex number.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    if isinstance(base, Dual) or isinstance(exponent, Dual):
        return base**exponent
    return math.pow(base, exponent)


def _is_real(value: object) -> bool:
    """Return whether ``value`` is a plain real number usable as a constant.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return isinstance(value, Real)


class Dual:
    """A (possibly nested) dual number ``value + tangent . eps``.

    ``order`` is the nesting depth: 1 when ``value`` and the tangent entries
    are real numbers, ``k + 1`` when they are duals of order ``k`` (entries
    may also be plain reals, meaning a constant at that level). Instances are
    treated as immutable.

    Args:
        value: Real number or dual of order ``k``.
        tangent: Tangent entries (reals, or duals of the same order as
            ``value``); the length is the number of seeded directions.

    Raises:
        TypeError: If an entry is neither real nor a dual of the right order.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """

    __slots__ = ("order", "tangent", "value")

    value: float | Dual
    tangent: tuple[float | Dual, ...]
    order: int

    def __init__(
        self, value: float | Dual, tangent: Iterable[float | Dual] = ()
    ) -> None:
        """Validate and store the value and tangent.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        if isinstance(value, Dual):
            inner = value.order
        elif _is_real(value) and not isinstance(value, bool):
            inner = 0
            value = float(value)
        else:
            raise TypeError("Dual value must be a real number or a Dual")
        entries: list[float | Dual] = []
        for entry in tangent:
            if isinstance(entry, Dual):
                if entry.order != inner:
                    raise TypeError("tangent duals must have the value's order")
                entries.append(entry)
            elif _is_real(entry) and not isinstance(entry, bool):
                entries.append(float(entry))
            else:
                raise TypeError("Dual tangent entries must be real numbers or Duals")
        self.value = value
        self.tangent = tuple(entries)
        self.order = inner + 1

    # -- conversions that would lose the derivative ---------------------------

    def __float__(self) -> float:
        """Refuse conversion to ``float`` (it would drop the tangent).

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("float() or a math.* function")

    def __int__(self) -> int:
        """Refuse conversion to ``int``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("int()")

    def __index__(self) -> int:
        """Refuse use as an index.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("use as an integer index")

    def __complex__(self) -> complex:
        """Refuse conversion to ``complex``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("complex()")

    def __round__(self, ndigits: int | None = None) -> float:
        """Refuse rounding (piecewise constant, derivative discarded).

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("round()")

    def __trunc__(self) -> int:
        """Refuse truncation.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("math.trunc()")

    def __floor__(self) -> int:
        """Refuse ``math.floor``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("math.floor()")

    def __ceil__(self) -> int:
        """Refuse ``math.ceil``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("math.ceil()")

    def __floordiv__(self, other: object) -> Dual:
        """Refuse floor division.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("floor division")

    def __rfloordiv__(self, other: object) -> Dual:
        """Refuse floor division.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("floor division")

    def __mod__(self, other: object) -> Dual:
        """Refuse the modulo operation.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("the % operator")

    def __rmod__(self, other: object) -> Dual:
        """Refuse the modulo operation.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("the % operator")

    # -- representation and comparisons on the value --------------------------

    def __repr__(self) -> str:
        """Return ``Dual(value, tangent)``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return f"Dual({self.value!r}, {self.tangent!r})"

    def __bool__(self) -> bool:
        """Return whether the innermost value is non-zero.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return bool(_primal(self))

    def __hash__(self) -> int:
        """Hash the innermost value (consistent with value equality).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return hash(_primal(self))

    def __eq__(self, other: object) -> bool:
        """Compare innermost values (tangents are ignored).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        if not (isinstance(other, Dual) or _is_real(other)):
            return NotImplemented
        return bool(_primal(self) == _primal(other))

    def __ne__(self, other: object) -> bool:
        """Compare innermost values (tangents are ignored).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        if not (isinstance(other, Dual) or _is_real(other)):
            return NotImplemented
        return bool(_primal(self) != _primal(other))

    def __lt__(self, other: float | Dual) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) < _primal(other))  # type: ignore[operator]

    def __le__(self, other: float | Dual) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) <= _primal(other))  # type: ignore[operator]

    def __gt__(self, other: float | Dual) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) > _primal(other))  # type: ignore[operator]

    def __ge__(self, other: float | Dual) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) >= _primal(other))  # type: ignore[operator]

    # -- arithmetic ------------------------------------------------------------

    def __pos__(self) -> Dual:
        """Return the dual unchanged.

        References:
            [EQ-AD-DUAL-001]. No new equation is introduced.

        """
        return self

    def __neg__(self) -> Dual:
        """Return ``-a - b eps``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        return _new(-self.value, tuple([-t for t in self.tangent]), self.order)

    def __abs__(self) -> Dual:
        """Return ``|a| + sign(a) b eps`` (derivative 0 at ``a = 0``).

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Section 14.1].

        """
        primal = _primal(self)
        if primal > 0.0:  # type: ignore[operator]
            return self
        if primal < 0.0:  # type: ignore[operator]
            return -self
        zero = tuple([0.0 * t for t in self.tangent])
        magnitude = cast("float | Dual", abs(self.value))
        return _new(magnitude, zero, self.order)

    def _constant_side(self, other: object) -> bool:
        """Return whether ``other`` is a constant relative to this dual.

        Unsupported operand types return ``False`` so that the caller can
        return ``NotImplemented``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        if isinstance(other, Dual):
            return other.order < self.order
        return _is_real(other)

    def __add__(self, other: float | Dual) -> Dual:
        """Return ``(a + c) + (b + d) eps``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, Dual) and other.order == self.order:
            if len(other.tangent) != len(self.tangent):
                raise ValueError("Dual tangents have different lengths")
            return _new(
                self.value + other.value,
                tuple(map(operator.add, self.tangent, other.tangent)),
                self.order,
            )
        if self._constant_side(other):
            return _new(self.value + other, self.tangent, self.order)
        if isinstance(other, Dual):
            return other.__radd__(self)
        return NotImplemented

    def __radd__(self, other: float | Dual) -> Dual:
        """Return ``c + (a + b eps)`` for a constant ``c``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self._constant_side(other):
            return _new(other + self.value, self.tangent, self.order)
        return NotImplemented

    def __sub__(self, other: float | Dual) -> Dual:
        """Return ``(a - c) + (b - d) eps``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, Dual) and other.order == self.order:
            if len(other.tangent) != len(self.tangent):
                raise ValueError("Dual tangents have different lengths")
            return _new(
                self.value - other.value,
                tuple(map(operator.sub, self.tangent, other.tangent)),
                self.order,
            )
        if self._constant_side(other):
            return _new(self.value - other, self.tangent, self.order)
        if isinstance(other, Dual):
            return other.__rsub__(self)
        return NotImplemented

    def __rsub__(self, other: float | Dual) -> Dual:
        """Return ``c - (a + b eps)`` for a constant ``c``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self._constant_side(other):
            return _new(
                other - self.value, tuple([-t for t in self.tangent]), self.order
            )
        return NotImplemented

    def __mul__(self, other: float | Dual) -> Dual:
        """Return ``a c + (b c + a d) eps``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, Dual) and other.order == self.order:
            if len(other.tangent) != len(self.tangent):
                raise ValueError("Dual tangents have different lengths")
            a, c = self.value, other.value
            return _new(
                a * c,
                tuple(
                    [
                        b * c + a * d
                        for b, d in zip(self.tangent, other.tangent, strict=True)
                    ]
                ),
                self.order,
            )
        if self._constant_side(other):
            return _new(self.value * other, _scaled(self.tangent, other), self.order)
        if isinstance(other, Dual):
            return other.__rmul__(self)
        return NotImplemented

    def __rmul__(self, other: float | Dual) -> Dual:
        """Return ``c (a + b eps)`` for a constant ``c``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self._constant_side(other):
            return _new(other * self.value, _scaled(self.tangent, other), self.order)
        return NotImplemented

    def __truediv__(self, other: float | Dual) -> Dual:
        """Return ``a / c + (b - (a / c) d) / c eps``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, Dual) and other.order == self.order:
            if len(other.tangent) != len(self.tangent):
                raise ValueError("Dual tangents have different lengths")
            c = other.value
            quotient = self.value / c
            return _new(
                quotient,
                tuple(
                    [
                        (b - quotient * d) / c
                        for b, d in zip(self.tangent, other.tangent, strict=True)
                    ]
                ),
                self.order,
            )
        if self._constant_side(other):
            return _new(
                self.value / other,
                tuple([t / other for t in self.tangent]),
                self.order,
            )
        if isinstance(other, Dual):
            return other.__rtruediv__(self)
        return NotImplemented

    def __rtruediv__(self, other: float | Dual) -> Dual:
        """Return ``c / a - (c / a^2) b eps`` for a constant ``c``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self._constant_side(other):
            quotient = other / self.value
            slope = -quotient / self.value
            return _new(quotient, _scaled(self.tangent, slope), self.order)
        return NotImplemented

    def __pow__(self, other: float | Dual) -> Dual:
        """Return ``a^c`` with the power rule, or ``exp(c log a)`` for a dual ``c``.

        A constant exponent uses ``c a^(c-1) b`` (valid for negative ``a`` and
        integer ``c``); a dual exponent of the same order needs ``a > 0``.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, Dual) and other.order == self.order:
            if len(other.tangent) != len(self.tangent):
                raise ValueError("Dual tangents have different lengths")
            a, c = self.value, other.value
            power = _real_pow(a, c)
            log_a = _log(a)
            ratio = c / a
            return _new(
                power,
                tuple(
                    [
                        power * (d * log_a + ratio * b)
                        for b, d in zip(self.tangent, other.tangent, strict=True)
                    ]
                ),
                self.order,
            )
        if self._constant_side(other):
            if _primal(other) == 0.0:
                zero = tuple([0.0 * t for t in self.tangent])
                return _new(_real_pow(self.value, 0.0), zero, self.order)
            slope = other * _real_pow(self.value, other - 1.0)
            power = _real_pow(self.value, other)
            return _new(power, _scaled(self.tangent, slope), self.order)
        if isinstance(other, Dual):
            return other.__rpow__(self)
        return NotImplemented

    def __rpow__(self, other: float | Dual) -> Dual:
        """Return ``c^(a + b eps) = c^a + c^a log(c) b eps`` for a constant ``c``.

        ``c = 0`` gives ``0`` with derivative ``0`` (for ``a > 0``).

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self._constant_side(other):
            power = _real_pow(other, self.value)
            if _primal(other) == 0.0:
                zero = tuple([0.0 * t for t in self.tangent])
                return _new(power, zero, self.order)
            return _new(power, _scaled(self.tangent, power * _log(other)), self.order)
        return NotImplemented


# ---------------------------------------------------------------------------
# Elementary functions on floats and duals
# ---------------------------------------------------------------------------


def _unary(
    x: Dual,
    function: Callable[[float | Dual], float | Dual],
    derivative: Callable[[float | Dual], float | Dual],
) -> Dual:
    """Apply ``f`` to a dual by the chain rule ``f(a) + f'(a) b eps``.

    References:
        [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    return _new(function(x.value), _scaled(x.tangent, derivative(x.value)), x.order)


def _exp(x: float | Dual) -> float | Dual:
    """Return ``exp(x)``; the derivative is the value itself.

    References:
        [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    if isinstance(x, Dual):
        value = _exp(x.value)
        return _new(value, _scaled(x.tangent, value), x.order)
    return math.exp(x)


def _expm1(x: float | Dual) -> float | Dual:
    """Return ``exp(x) - 1`` accurately; derivative ``exp(x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _expm1, _exp)
    return math.expm1(x)


def _log(x: float | Dual, base: float | Dual | None = None) -> float | Dual:
    """Return ``log(x)`` (natural, or to ``base``); derivative ``1/x``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if base is not None:
        return _log(x) / _log(base)
    if isinstance(x, Dual):
        return _unary(x, _log, lambda v: 1.0 / v)
    return math.log(x)


def _log1p(x: float | Dual) -> float | Dual:
    """Return ``log(1 + x)`` accurately; derivative ``1/(1 + x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _log1p, lambda v: 1.0 / (1.0 + v))
    return math.log1p(x)


def _log2(x: float | Dual) -> float | Dual:
    """Return ``log2(x)``; derivative ``1/(x log 2)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _log2, lambda v: 1.0 / (v * _LOG_TWO))
    return math.log2(x)


def _log10(x: float | Dual) -> float | Dual:
    """Return ``log10(x)``; derivative ``1/(x log 10)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _log10, lambda v: 1.0 / (v * _LOG_TEN))
    return math.log10(x)


def _sqrt(x: float | Dual) -> float | Dual:
    """Return ``sqrt(x)``; derivative ``1/(2 sqrt(x))`` (raises at 0).

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        root = _sqrt(x.value)
        return _new(root, _scaled(x.tangent, 0.5 / root), x.order)
    return math.sqrt(x)


def _sin(x: float | Dual) -> float | Dual:
    """Return ``sin(x)``; derivative ``cos(x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _sin, _cos)
    return math.sin(x)


def _cos(x: float | Dual) -> float | Dual:
    """Return ``cos(x)``; derivative ``-sin(x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _cos, lambda v: -_sin(v))
    return math.cos(x)


def _tan(x: float | Dual) -> float | Dual:
    """Return ``tan(x)``; derivative ``1 + tan(x)^2``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        value = _tan(x.value)
        return _new(value, _scaled(x.tangent, 1.0 + value * value), x.order)
    return math.tan(x)


def _asin(x: float | Dual) -> float | Dual:
    """Return ``asin(x)``; derivative ``1/sqrt(1 - x^2)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _asin, lambda v: 1.0 / _sqrt(1.0 - v * v))
    return math.asin(x)


def _acos(x: float | Dual) -> float | Dual:
    """Return ``acos(x)``; derivative ``-1/sqrt(1 - x^2)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _acos, lambda v: -1.0 / _sqrt(1.0 - v * v))
    return math.acos(x)


def _atan(x: float | Dual) -> float | Dual:
    """Return ``atan(x)``; derivative ``1/(1 + x^2)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _atan, lambda v: 1.0 / (1.0 + v * v))
    return math.atan(x)


def _order(value: object) -> int:
    """Return the dual order of a value (0 for a plain real).

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    return value.order if isinstance(value, Dual) else 0


def _split(
    value: float | Dual, order: int
) -> tuple[float | Dual, tuple[float | Dual, ...] | None]:
    """Return ``(value, tangent)`` at a given order (``None`` for a constant).

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    if isinstance(value, Dual) and value.order == order:
        return value.value, value.tangent
    return value, None


def _atan2(y: float | Dual, x: float | Dual) -> float | Dual:
    """Return ``atan2(y, x)``; gradient ``(x, -y) / (x^2 + y^2)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    order = max(_order(y), _order(x))
    if order == 0:
        return math.atan2(cast(float, y), cast(float, x))
    y_value, y_tangent = _split(y, order)
    x_value, x_tangent = _split(x, order)
    value = _atan2(y_value, x_value)
    radius = x_value * x_value + y_value * y_value
    template = y_tangent if y_tangent is not None else x_tangent
    assert template is not None
    tangent = [0.0 * t for t in template]
    if y_tangent is not None:
        factor = x_value / radius
        tangent = [s + factor * t for s, t in zip(tangent, y_tangent, strict=True)]
    if x_tangent is not None:
        factor = -y_value / radius
        tangent = [s + factor * t for s, t in zip(tangent, x_tangent, strict=True)]
    return _new(value, tuple(tangent), order)


def _sinh(x: float | Dual) -> float | Dual:
    """Return ``sinh(x)``; derivative ``cosh(x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _sinh, _cosh)
    return math.sinh(x)


def _cosh(x: float | Dual) -> float | Dual:
    """Return ``cosh(x)``; derivative ``sinh(x)``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return _unary(x, _cosh, _sinh)
    return math.cosh(x)


def _tanh(x: float | Dual) -> float | Dual:
    """Return ``tanh(x)``; derivative ``1 - tanh(x)^2``.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        value = _tanh(x.value)
        return _new(value, _scaled(x.tangent, 1.0 - value * value), x.order)
    return math.tanh(x)


def _erf(x: float | Dual) -> float | Dual:
    """Return ``erf(x)``; derivative ``2 exp(-x^2) / sqrt(pi)``.

    References:
        [EQ-AD-DUAL-001], [REF-NIST-DLMF, Section 7.10].

    """
    if isinstance(x, Dual):
        return _unary(x, _erf, lambda v: _TWO_OVER_SQRT_PI * _exp(-v * v))
    return math.erf(x)


def _erfc(x: float | Dual) -> float | Dual:
    """Return ``erfc(x)``; derivative ``-2 exp(-x^2) / sqrt(pi)``.

    References:
        [EQ-AD-DUAL-001], [REF-NIST-DLMF, Section 7.10].

    """
    if isinstance(x, Dual):
        return _unary(x, _erfc, lambda v: -_TWO_OVER_SQRT_PI * _exp(-v * v))
    return math.erfc(x)


def _lgamma(x: float | Dual) -> float | Dual:
    """Return ``log |Gamma(x)|``; derivative ``digamma(x)`` (needs ``x > 0``).

    References:
        [EQ-AD-DUAL-001], [EQ-NUM-DIGAMMA-001], [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    if isinstance(x, Dual):
        return _unary(x, _lgamma, _digamma)
    return math.lgamma(x)


def _digamma(x: float | Dual) -> float | Dual:
    """Return ``psi(x)`` for ``x > 0``; derivative ``trigamma(x)``.

    References:
        [EQ-AD-DUAL-001], [EQ-NUM-DIGAMMA-001], [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    if isinstance(x, Dual):
        return _unary(x, _digamma, _trigamma)
    return _float_digamma(x)


def _trigamma(x: float | Dual) -> float | Dual:
    """Return ``psi'(x)`` for ``x > 0`` (floats only; duals raise).

    Raises:
        AutodiffUnsupportedError: For a dual argument (a third derivative of
            ``lgamma`` would be required).

    References:
        [EQ-NUM-DIGAMMA-001], [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    if isinstance(x, Dual):
        raise AutodiffUnsupportedError(
            "trigamma of a Dual (third derivative of lgamma) is not implemented"
        )
    return _float_trigamma(x)


def _hypot(*coordinates: float | Dual) -> float | Dual:
    """Return the Euclidean norm; gradient ``x_i / hypot`` (0 at the origin).

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    order = max((_order(c) for c in coordinates), default=0)
    if order == 0:
        return math.hypot(*cast(tuple[float, ...], coordinates))
    parts = [_split(c, order) for c in coordinates]
    value = _hypot(*(v for v, _ in parts))
    template = next(t for _, t in parts if t is not None)
    tangent: list[float | Dual] = [0.0 * t for t in template]
    if _primal(value) != 0.0:
        for coordinate_value, coordinate_tangent in parts:
            if coordinate_tangent is None:
                continue
            factor = coordinate_value / value
            tangent = [
                s + factor * t for s, t in zip(tangent, coordinate_tangent, strict=True)
            ]
    return _new(value, tuple(tangent), order)


def _fsum(values: Iterable[float | Dual]) -> float | Dual:
    """Return an accurately rounded sum; duals are summed level by level.

    The values (and each tangent component) of the top-order duals are summed
    with ``math.fsum`` recursively; lower-order entries count as constants.

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    items = list(values)
    order = max((_order(v) for v in items), default=0)
    if order == 0:
        return math.fsum(cast(list[float], items))
    parts = [_split(v, order) for v in items]
    tangents = [t for _, t in parts if t is not None]
    length = len(tangents[0])
    if any(len(t) != length for t in tangents):
        raise ValueError("Dual tangents have different lengths")
    value = _fsum(v for v, _ in parts)
    tangent = tuple(_fsum(t[i] for t in tangents) for i in range(length))
    return _new(value, tangent, order)


def _pow(x: float | Dual, y: float | Dual) -> float | Dual:
    """Return ``x ** y`` (``math.pow`` for plain floats).

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual) or isinstance(y, Dual):
        return x**y
    return math.pow(x, y)


def _fabs(x: float | Dual) -> float | Dual:
    """Return ``|x|`` (derivative ``sign(x)``, 0 at 0).

    References:
        [EQ-AD-DUAL-001], [REF-PYTHON-MATH].

    """
    if isinstance(x, Dual):
        return abs(x)
    return math.fabs(x)


def _isfinite(x: float | Dual) -> bool:
    """Return whether the innermost value is finite.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    return math.isfinite(cast(float, _primal(x)))


def _isnan(x: float | Dual) -> bool:
    """Return whether the innermost value is NaN.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    return math.isnan(cast(float, _primal(x)))


def _isinf(x: float | Dual) -> bool:
    """Return whether the innermost value is infinite.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    return math.isinf(cast(float, _primal(x)))


class admath:
    """Drop-in replacements for :mod:`math` that also accept :class:`Dual`.

    ``admath.f(x)`` returns ``math.f(x)`` for a plain float and the dual
    ``f(a) + f'(a) b eps`` for a dual, so a log-density written with
    ``admath`` instead of ``math`` is differentiated exactly by
    :func:`gradient` and :func:`hessian`. Available: ``exp``, ``expm1``,
    ``log`` (optional base), ``log1p``, ``log2``, ``log10``, ``sqrt``,
    ``pow``, ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``,
    ``atan2``, ``sinh``, ``cosh``, ``tanh``, ``erf``, ``erfc``, ``lgamma``,
    ``digamma``, ``trigamma`` (floats only), ``hypot``, ``fabs``, ``fsum``,
    ``isfinite``, ``isnan``, ``isinf`` and the constants ``pi``, ``e``,
    ``tau``, ``inf``, ``nan``. The class is a namespace and is never
    instantiated.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-PYTHON-MATH].

    """

    pi = math.pi
    e = math.e
    tau = math.tau
    inf = math.inf
    nan = math.nan
    exp = staticmethod(_exp)
    expm1 = staticmethod(_expm1)
    log = staticmethod(_log)
    log1p = staticmethod(_log1p)
    log2 = staticmethod(_log2)
    log10 = staticmethod(_log10)
    sqrt = staticmethod(_sqrt)
    pow = staticmethod(_pow)
    sin = staticmethod(_sin)
    cos = staticmethod(_cos)
    tan = staticmethod(_tan)
    asin = staticmethod(_asin)
    acos = staticmethod(_acos)
    atan = staticmethod(_atan)
    atan2 = staticmethod(_atan2)
    sinh = staticmethod(_sinh)
    cosh = staticmethod(_cosh)
    tanh = staticmethod(_tanh)
    erf = staticmethod(_erf)
    erfc = staticmethod(_erfc)
    lgamma = staticmethod(_lgamma)
    digamma = staticmethod(_digamma)
    trigamma = staticmethod(_trigamma)
    hypot = staticmethod(_hypot)
    fabs = staticmethod(_fabs)
    fsum = staticmethod(_fsum)
    isfinite = staticmethod(_isfinite)
    isnan = staticmethod(_isnan)
    isinf = staticmethod(_isinf)

    def __init__(self) -> None:
        """Refuse instantiation; use the class attributes directly.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        raise TypeError("admath is a namespace; call admath.exp(...) directly")


# ---------------------------------------------------------------------------
# Derivative drivers
# ---------------------------------------------------------------------------


def _point(x: Sequence[float], name: str = "x") -> tuple[float, ...]:
    """Validate a non-empty finite evaluation point.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(x, str | bytes) or not isinstance(x, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    point = tuple(require_finite_real(v, name=f"{name}[{i}]") for i, v in enumerate(x))
    if not point:
        raise ValueError(f"{name} must be non-empty")
    return point


def _seeds(point: tuple[float, ...]) -> tuple[Dual, ...]:
    """Return the first-order seeds ``x_i + e_i eps``.

    References:
        [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Section 3.1].

    """
    size = len(point)
    return tuple(
        _new(value, tuple(1.0 if j == i else 0.0 for j in range(size)), 1)
        for i, value in enumerate(point)
    )


def _first_order(output: object, size: int) -> tuple[float, tuple[float, ...]]:
    """Read value and gradient from the output of a first-order evaluation.

    Raises:
        TypeError: If the output is neither real nor a first-order dual.
        ValueError: If the tangent length differs from the dimension.

    References:
        [REF-GRIEWANK-WALTHER-2008, Section 3.1]. No equation is evaluated.

    """
    if isinstance(output, Dual):
        if output.order != 1:
            raise TypeError("function returned a Dual of unexpected order")
        if len(output.tangent) != size:
            raise ValueError("function returned a Dual with a wrong tangent length")
        return cast(float, output.value), cast(tuple[float, ...], output.tangent)
    if _is_real(output) and not isinstance(output, bool):
        return float(cast(float, output)), (0.0,) * size
    raise TypeError(
        f"function must return a real number or a Dual, got {type(output).__name__}"
    )


def value_and_gradient(
    function: Callable[[tuple[Dual, ...]], float | Dual], x: Sequence[float]
) -> tuple[float, tuple[float, ...]]:
    """Return ``f(x)`` and ``grad f(x)`` from one vector-tangent evaluation.

    ``f`` is called once with the seeds ``x_i + e_i eps``; the value of the
    result is ``f(x)`` and its tangent is the gradient, exact up to rounding.

    Args:
        function: Scalar function of a tuple of numbers written with
            arithmetic operators and :class:`admath`.
        x: Finite evaluation point.

    Returns:
        ``(f(x), grad f(x))``. A real (non-dual) result means ``f`` does not
        depend on ``x``; its gradient is zero.

    Raises:
        AutodiffUnsupportedError: If ``f`` converts a dual to ``float`` (for
            example through ``math.exp``).
        TypeError: If ``f`` returns something other than a number or a dual.

    Equation:
        [EQ-AD-DUAL-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    point = _point(x)
    return _first_order(function(_seeds(point)), len(point))


def gradient(
    function: Callable[[tuple[Dual, ...]], float | Dual], x: Sequence[float]
) -> tuple[float, ...]:
    """Return the exact gradient of a scalar function by forward mode.

    Args:
        function: Scalar function written with operators and :class:`admath`.
        x: Finite evaluation point.

    Returns:
        ``grad f(x)``.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support duals.

    Equation:
        [EQ-AD-DUAL-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    return value_and_gradient(function, x)[1]


def derivative(function: Callable[[Dual], float | Dual], x: float) -> float:
    """Return ``f'(x)`` of a scalar function of one variable.

    Args:
        function: Function of one number.
        x: Finite evaluation point.

    Returns:
        Exact derivative up to rounding.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support duals.

    Equation:
        [EQ-AD-DUAL-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    point = require_finite_real(x, name="x")
    return _first_order(function(_new(point, (1.0,), 1)), 1)[1][0]


def jacobian(
    function: Callable[[tuple[Dual, ...]], Sequence[float | Dual]], x: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Return the Jacobian ``J_ij = d f_i / d x_j`` of a vector function.

    One vector-tangent evaluation gives every row.

    Args:
        function: Function returning a sequence of numbers or duals.
        x: Finite evaluation point.

    Returns:
        ``m x d`` Jacobian as a tuple of rows.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support duals.
        TypeError: If the output is not a sequence.

    Equation:
        [EQ-AD-DUAL-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    point = _point(x)
    outputs = function(_seeds(point))
    if isinstance(outputs, str | bytes) or not isinstance(outputs, Sequence):
        raise TypeError("function must return a sequence")
    return tuple(_first_order(item, len(point))[1] for item in outputs)


def _second_order_seeds(point: tuple[float, ...]) -> tuple[Dual, ...]:
    """Return seeds ``(x_i + e_i eps1) + (e_i + 0 eps1) eps2`` for Hessians.

    References:
        [EQ-AD-HESSIAN-001], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

    """
    size = len(point)
    zeros = (0.0,) * size
    seeds: list[Dual] = []
    for i, value in enumerate(point):
        inner_value = _new(value, tuple(1.0 if j == i else 0.0 for j in range(size)), 1)
        outer_tangent = tuple(
            _new(1.0 if j == i else 0.0, zeros, 1) for j in range(size)
        )
        seeds.append(_new(inner_value, outer_tangent, 2))
    return tuple(seeds)


def value_gradient_and_hessian(
    function: Callable[[tuple[Dual, ...]], float | Dual], x: Sequence[float]
) -> tuple[float, tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return ``f(x)``, its gradient and its Hessian by nested forward mode.

    ``f`` is evaluated once on second-order duals; the outer tangent of the
    result holds ``d f / d x_j`` as first-order duals whose tangents are the
    Hessian columns. The returned Hessian is symmetrized,
    ``(H + H') / 2``, which only removes rounding asymmetry.

    Args:
        function: Scalar function written with operators and :class:`admath`
            (twice differentiable at ``x``).
        x: Finite evaluation point.

    Returns:
        ``(f(x), grad f(x), Hessian)``.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support duals or needs a
            derivative this module lacks (``trigamma`` of a dual).
        TypeError: If ``f`` returns an unexpected object.

    Equation:
        [EQ-AD-HESSIAN-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapters 3 and 13].

    """
    point = _point(x)
    size = len(point)
    output = function(_second_order_seeds(point))
    zero_row = (0.0,) * size
    if not isinstance(output, Dual):
        value, _ = _first_order(output, size)
        return value, zero_row, tuple(zero_row for _ in range(size))
    if output.order != 2 or len(output.tangent) != size:
        raise TypeError("function returned a Dual of unexpected shape")
    value, grad = _first_order(output.value, size)
    columns: list[tuple[float, ...]] = []
    for entry in output.tangent:
        if isinstance(entry, Dual):
            columns.append(cast(tuple[float, ...], entry.tangent))
        else:
            columns.append(zero_row)
    hessian_matrix = tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )
    return value, grad, hessian_matrix


def hessian(
    function: Callable[[tuple[Dual, ...]], float | Dual], x: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Return the exact Hessian of a scalar function (forward over forward).

    Args:
        function: Scalar function written with operators and :class:`admath`.
        x: Finite evaluation point.

    Returns:
        Symmetric ``d x d`` Hessian.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support duals.

    Equation:
        [EQ-AD-HESSIAN-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapters 3 and 13].

    """
    return value_gradient_and_hessian(function, x)[2]


# ---------------------------------------------------------------------------
# Gradient oracle with recorded fallback
# ---------------------------------------------------------------------------


class GradientEvaluator:
    """Value-and-gradient oracle: analytic, autodiff, or finite differences.

    ``option`` selects the gradient:

    * ``"auto"`` (default): try forward-mode autodiff; if evaluating the
      function on duals raises a ``TypeError`` (including
      :class:`AutodiffUnsupportedError`) while the plain float evaluation
      succeeds, switch permanently to ``finite_difference`` and record the
      reason. Errors that also occur for floats propagate unchanged.
    * ``"autodiff"``: forward mode only; unsupported functions raise
      :class:`AutodiffUnsupportedError`.
    * ``"finite-difference"``: always use ``finite_difference``.
    * a callable: an analytic gradient ``x -> grad f(x)``.

    The evaluator never counts function calls itself; wrap ``function`` in a
    counter to account for them (one dual evaluation is one call).

    Args:
        function: Scalar function of a tuple of numbers.
        option: Gradient choice (see above).
        finite_difference: ``(f, x) -> grad`` used by the finite-difference
            modes (required for ``"auto"`` and ``"finite-difference"``).

    Raises:
        TypeError: If ``function`` or a callable ``option`` is not callable.
        ValueError: For an unknown option or a missing ``finite_difference``.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-NOCEDAL-WRIGHT-2006,
        Section 8.1].

    """

    __slots__ = (
        "_analytic",
        "_finite_difference",
        "_function",
        "fallback_reason",
        "gradient_evaluations",
        "method",
        "requested",
    )

    def __init__(
        self,
        function: Callable[..., object],
        option: str | GradientCallable = "auto",
        *,
        finite_difference: FiniteDifference | None = None,
    ) -> None:
        """Validate the option and choose the initial method.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not callable(function):
            raise TypeError("function must be callable")
        self._function = function
        self._finite_difference = finite_difference
        self._analytic: GradientCallable | None = None
        self.fallback_reason: str | None = None
        self.gradient_evaluations = 0
        if callable(option):
            self._analytic = option
            self.requested = "analytic"
            self.method = "analytic"
            return
        if option not in ("auto", "autodiff", "finite-difference"):
            raise ValueError(
                "gradient must be 'auto', 'autodiff', 'finite-difference', or a "
                "callable"
            )
        if option != "autodiff" and finite_difference is None:
            raise ValueError("finite_difference is required for this option")
        self.requested = option
        self.method = (
            "finite-difference" if option == "finite-difference" else ("autodiff")
        )

    def _float_value(self, x: tuple[float, ...]) -> float:
        """Evaluate the function on plain floats.

        References:
            Project API policy. No equation is evaluated.

        """
        return float(cast(float, self._function(x)))

    def _dual(self, x: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
        """Evaluate value and gradient on first-order seeds.

        References:
            [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        return _first_order(self._function(_seeds(x)), len(x))

    def value_and_gradient(
        self, x: Sequence[float]
    ) -> tuple[float, tuple[float, ...] | None]:
        """Return ``(f(x), grad f(x))``, with ``None`` for a non-finite ``f(x)``.

        Args:
            x: Evaluation point.

        Returns:
            Value and gradient (``None`` when the value is not finite, so no
            finite-difference work is wasted outside the support).

        Raises:
            AutodiffUnsupportedError: In ``"autodiff"`` mode for a function
                that does not support duals.
            ValueError: If an analytic gradient has the wrong length.

        Equation:
            [EQ-AD-DUAL-001], [EQ-OPT-FD-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3],
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = tuple(float(v) for v in x)
        if self.method == "autodiff":
            try:
                value, grad = self._dual(point)
            except TypeError as error:
                if self.requested == "autodiff":
                    if isinstance(error, AutodiffUnsupportedError):
                        raise
                    raise AutodiffUnsupportedError(
                        f"function does not support Dual numbers: {error}"
                    ) from error
                value = self._float_value(point)
                self.method = "finite-difference"
                self.fallback_reason = f"{type(error).__name__}: {error}"
            else:
                if not math.isfinite(value):
                    return value, None
                self.gradient_evaluations += 1
                return value, grad
        else:
            value = self._float_value(point)
        if not math.isfinite(value):
            return value, None
        self.gradient_evaluations += 1
        if self._analytic is not None:
            grad = tuple(float(g) for g in self._analytic(point))
            if len(grad) != len(point):
                raise ValueError("gradient must match the state dimension")
            return value, grad
        assert self._finite_difference is not None
        evaluate = cast(Callable[[tuple[float, ...]], float], self._float_value)
        return value, tuple(float(g) for g in self._finite_difference(evaluate, point))

    def gradient(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return ``grad f(x)``.

        Args:
            x: Evaluation point with a finite value.

        Returns:
            Gradient.

        Raises:
            ValueError: If ``f(x)`` is not finite.

        Equation:
            [EQ-AD-DUAL-001], [EQ-OPT-FD-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        value, grad = self.value_and_gradient(x)
        if grad is None:
            raise ValueError(f"function value {value} is not finite")
        return grad
