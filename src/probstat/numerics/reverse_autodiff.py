"""Reverse-mode (adjoint) automatic differentiation on an operation tape.

Evaluating a function on :class:`ReverseVariable` inputs records every
elementary operation ``v_i = phi_i(v_j, j < i)`` on a tape together with its
local partial derivatives ``c_ij = d phi_i / d v_j``. One backward sweep over
the tape in reverse order then accumulates the adjoints
``vbar_j += vbar_i c_ij`` from ``ybar = 1`` ([EQ-AD-REVERSE-001]); the
adjoints of the inputs are the gradient. The derivatives are exact up to
floating-point rounding, and the work of the sweep is a small constant
multiple of the work of the recorded evaluation, independent of the number of
inputs (Griewank and Walther 2008, Chapters 3 and 4; Baydin et al. 2018,
Section 3.2). Forward mode (:mod:`probstat.numerics.autodiff`) costs about
``d`` times one evaluation for a gradient in ``R^d``; reverse mode is the
cheaper of the two once ``d`` is more than a few.

Drivers
    :func:`reverse_value_and_gradient`, :func:`reverse_gradient`,
    :func:`vector_jacobian_product` (seed ``ybar = w``, result ``w' J``) and
    :func:`reverse_jacobian` (one recording, one sweep per output).
    :func:`reverse_hessian`, :func:`reverse_value_gradient_and_hessian` and
    :func:`hessian_vector_product` differentiate the sweep itself by forward
    mode: the tape is recorded on first-order :class:`~probstat.numerics.Dual`
    values, so each input adjoint is a dual whose tangent is a Hessian row
    (forward over reverse, [EQ-AD-FOR-001]).

Programs and errors
    The same programs as forward mode are supported: arithmetic operators,
    comparisons, ``abs`` and every :class:`~probstat.numerics.admath`
    function. ``float(v)``, ``int``, ``round``, ``//``, ``%``, the
    :mod:`math` functions and float-only validators raise
    :class:`~probstat.numerics.AutodiffUnsupportedError` (a ``TypeError``),
    so callers that offer a fallback can switch to finite differences.
    Points where the derivative is unbounded or undefined (``sqrt`` at 0,
    ``asin``/``acos`` at +-1, ``0 ** c`` with ``0 < c < 1``, ``atan2`` at the
    origin, a variable exponent of a non-positive base) raise
    :class:`NonDifferentiableError` (a ``ValueError``, which the samplers read
    as "outside the support"). Kinks with bounded one-sided derivatives use
    the forward-mode convention: derivative 0 at ``abs(0)`` and at
    ``hypot(0, ..., 0)`` (an element of the subdifferential).

Implementation notes
    Nodes store parent *indices*, never references to other variables, and
    the backward sweep is a plain loop, so tapes of millions of operations
    neither reach the recursion limit nor build deep reference chains.
    Variables of different tapes cannot be combined (nested reverse mode is
    not supported), and a :class:`~probstat.numerics.Dual` operand is
    rejected rather than treated as a constant.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import cast

from probstat.foundations.validation import require_finite_real

from .autodiff import (
    _LOG_TEN,
    _LOG_TWO,
    _TWO_OVER_SQRT_PI,
    AutodiffUnsupportedError,
    Dual,
    _acos,
    _asin,
    _atan,
    _atan2,
    _cos,
    _cosh,
    _digamma,
    _erf,
    _erfc,
    _exp,
    _expm1,
    _fsum,
    _hypot,
    _lgamma,
    _log,
    _log1p,
    _log2,
    _log10,
    _new,
    _point,
    _primal,
    _real_pow,
    _seeds,
    _sin,
    _sinh,
    _sqrt,
    _tan,
    _tanh,
    _TapeOperand,
    _trigamma,
    _unsupported,
)

Scalar = float | Dual
_allocate = object.__new__
_ONE: tuple[Scalar, ...] = (1.0,)
_MINUS_ONE: tuple[Scalar, ...] = (-1.0,)
_ONE_ONE: tuple[Scalar, ...] = (1.0, 1.0)
_ONE_MINUS_ONE: tuple[Scalar, ...] = (1.0, -1.0)


class NonDifferentiableError(ValueError):
    """A reverse-mode operation was evaluated where its derivative is unbounded.

    Raised during the recording pass, for example by ``admath.sqrt(v)`` at
    ``v = 0``. It subclasses ``ValueError`` so that code mapping numerical
    domain errors to ``-inf`` (the HMC/NUTS targets, the PPL log-density)
    treats the point as outside the support.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 14].

    """


def _not_differentiable(operation: str, point: object) -> NonDifferentiableError:
    """Return the error for an operation without a bounded derivative.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 14]. No equation is evaluated.

    """
    return NonDifferentiableError(
        f"{operation} is not differentiable at {_primal(point)!r} "
        "(unbounded or undefined derivative)"
    )


def _is_constant(value: object) -> bool:
    """Return whether ``value`` is a plain real number (a tape constant).

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    kind = value.__class__
    return kind is float or kind is int or isinstance(value, Real)


def _foreign(other: object) -> object:
    """Reject a dual operand; return ``NotImplemented`` for other types.

    Raises:
        AutodiffUnsupportedError: If ``other`` is a dual number.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    if isinstance(other, Dual):
        raise AutodiffUnsupportedError(
            "a ReverseVariable cannot be combined with a Dual; use "
            "reverse_hessian or hessian_vector_product for second derivatives"
        )
    return NotImplemented


class _Tape:
    """Append-only record of operations: parent indices and local partials.

    Node ``i`` stores the indices of the nodes it was computed from and the
    partial derivatives ``c_ij`` with respect to each of them; inputs have no
    parents. Partials are floats, or first-order duals when the tape is
    recorded for forward-over-reverse second derivatives.

    References:
        [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """

    __slots__ = ("nodes",)

    def __init__(self) -> None:
        """Create an empty tape.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        self.nodes: list[tuple[tuple[int, ...], tuple[Scalar, ...]]] = []

    def __len__(self) -> int:
        """Return the number of recorded nodes.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return len(self.nodes)

    def push(
        self,
        value: Scalar,
        parents: tuple[int, ...],
        partials: tuple[Scalar, ...],
    ) -> ReverseVariable:
        """Append one node and return the variable that refers to it.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        nodes = self.nodes
        result = _allocate(ReverseVariable)
        result.value = value
        result.index = len(nodes)
        result.tape = self
        nodes.append((parents, partials))
        return result

    def sweep(self, seeds: dict[int, Scalar]) -> list[Scalar]:
        """Run the backward sweep from the seeded adjoints.

        The loop visits nodes from the highest seeded index down to 0 and
        adds ``vbar_i c_ij`` to the adjoint of every parent ``j``; adjoints
        that are exactly the float zero are skipped. No recursion is used.

        Args:
            seeds: Initial adjoints by node index (``{output: 1.0}`` for a
                gradient, ``{y_k: w_k}`` for a vector-Jacobian product).

        Returns:
            Adjoints of nodes ``0..max(seeds)``.

        Equation:
            [EQ-AD-REVERSE-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
            Section 3.2].

        """
        if not seeds:
            return []
        top = max(seeds)
        adjoints: list[Scalar] = [0.0] * (top + 1)
        for index, seed in seeds.items():
            adjoints[index] = adjoints[index] + seed
        nodes = self.nodes
        for i in range(top, -1, -1):
            adjoint = adjoints[i]
            if adjoint.__class__ is float and adjoint == 0.0:
                continue
            parents, partials = nodes[i]
            for parent, partial in zip(parents, partials):  # noqa: B905
                adjoints[parent] = adjoints[parent] + adjoint * partial
        return adjoints


class ReverseVariable(_TapeOperand):
    """A scalar recorded on a reverse-mode tape.

    Instances are created by the drivers of this module (the inputs) and by
    arithmetic on other instances; they are immutable. ``value`` is the
    numerical value of the node (a float, or a first-order dual while a
    second-derivative driver runs) and ``index`` its position on the tape.

    Raises:
        TypeError: If constructed directly.

    References:
        [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3],
        [REF-BAYDIN-ETAL-2018, Section 3.2].

    """

    __slots__ = ("index", "tape", "value")

    value: Scalar
    index: int
    tape: _Tape

    def __init__(self) -> None:
        """Refuse direct construction; use the reverse-mode drivers.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        raise TypeError(
            "ReverseVariable instances are created by reverse_gradient and the "
            "other reverse-mode drivers"
        )

    # -- helpers ---------------------------------------------------------------

    def _tape_of(self, other: ReverseVariable) -> _Tape:
        """Return the shared tape of two variables.

        Raises:
            AutodiffUnsupportedError: If the variables belong to different
                tapes (nested reverse mode).

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        if other.tape is not self.tape:
            raise AutodiffUnsupportedError(
                "ReverseVariables from different tapes cannot be combined "
                "(nested reverse mode is not supported)"
            )
        return self.tape

    def _unary(self, value: Scalar, partial: Scalar) -> ReverseVariable:
        """Record ``v = phi(self)`` with local derivative ``partial``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        return self.tape.push(value, (self.index,), (partial,))

    def _admath(self, name: str, arguments: tuple[object, ...]) -> Scalar:
        """Evaluate ``admath.<name>(*arguments)`` on the tape.

        Raises:
            AutodiffUnsupportedError: For a function without a reverse rule.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if name == "atan2":
            return cast(Scalar, _record_atan2(arguments[0], arguments[1]))
        if name == "hypot":
            return cast(Scalar, _record_hypot(arguments))
        if name == "fsum":
            return cast(Scalar, _record_fsum(arguments))
        rule = _UNARY_RULES.get(name)
        if rule is None:
            raise _unsupported(f"admath.{name} in reverse mode")
        return cast(Scalar, rule(self))

    # -- conversions that would lose the derivative ---------------------------

    def __float__(self) -> float:
        """Refuse conversion to ``float`` (it would drop the derivative).

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

    def __floordiv__(self, other: object) -> ReverseVariable:
        """Refuse floor division.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("floor division")

    def __rfloordiv__(self, other: object) -> ReverseVariable:
        """Refuse floor division.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("floor division")

    def __mod__(self, other: object) -> ReverseVariable:
        """Refuse the modulo operation.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("the % operator")

    def __rmod__(self, other: object) -> ReverseVariable:
        """Refuse the modulo operation.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise _unsupported("the % operator")

    # -- representation and comparisons on the value --------------------------

    def __repr__(self) -> str:
        """Return ``ReverseVariable(value, index=i)``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return f"ReverseVariable({self.value!r}, index={self.index})"

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
        """Compare innermost values (derivatives are ignored).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        if not (isinstance(other, ReverseVariable | Dual) or _is_constant(other)):
            return NotImplemented
        return bool(_primal(self) == _primal(other))

    def __ne__(self, other: object) -> bool:
        """Compare innermost values (derivatives are ignored).

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        if not (isinstance(other, ReverseVariable | Dual) or _is_constant(other)):
            return NotImplemented
        return bool(_primal(self) != _primal(other))

    def __lt__(self, other: float | ReverseVariable) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) < _primal(other))  # type: ignore[operator]

    def __le__(self, other: float | ReverseVariable) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) <= _primal(other))  # type: ignore[operator]

    def __gt__(self, other: float | ReverseVariable) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) > _primal(other))  # type: ignore[operator]

    def __ge__(self, other: float | ReverseVariable) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return cast(bool, _primal(self) >= _primal(other))  # type: ignore[operator]

    # -- arithmetic ------------------------------------------------------------

    def __pos__(self) -> ReverseVariable:
        """Return the variable unchanged.

        References:
            [EQ-AD-REVERSE-001]. No new equation is introduced.

        """
        return self

    def __neg__(self) -> ReverseVariable:
        """Record ``-a`` (partial ``-1``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        return self.tape.push(-self.value, (self.index,), _MINUS_ONE)

    def __abs__(self) -> ReverseVariable:
        """Record ``|a|`` (partial ``sign(a)``, 0 at ``a = 0``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 14].

        """
        primal = cast(float, _primal(self))
        if primal > 0.0:
            return self
        if primal < 0.0:
            return -self
        return self._unary(cast(Scalar, abs(self.value)), 0.0)

    def __add__(self, other: float | ReverseVariable) -> ReverseVariable:
        """Record ``a + c`` (partials ``1, 1``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, ReverseVariable):
            tape = self._tape_of(other)
            return tape.push(
                self.value + other.value, (self.index, other.index), _ONE_ONE
            )
        if _is_constant(other):
            return self.tape.push(self.value + other, (self.index,), _ONE)
        return cast(ReverseVariable, _foreign(other))

    def __radd__(self, other: float) -> ReverseVariable:
        """Record ``c + a`` for a constant ``c``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if _is_constant(other):
            return self.tape.push(other + self.value, (self.index,), _ONE)
        return cast(ReverseVariable, _foreign(other))

    def __sub__(self, other: float | ReverseVariable) -> ReverseVariable:
        """Record ``a - c`` (partials ``1, -1``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, ReverseVariable):
            tape = self._tape_of(other)
            return tape.push(
                self.value - other.value, (self.index, other.index), _ONE_MINUS_ONE
            )
        if _is_constant(other):
            return self.tape.push(self.value - other, (self.index,), _ONE)
        return cast(ReverseVariable, _foreign(other))

    def __rsub__(self, other: float) -> ReverseVariable:
        """Record ``c - a`` for a constant ``c``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if _is_constant(other):
            return self.tape.push(other - self.value, (self.index,), _MINUS_ONE)
        return cast(ReverseVariable, _foreign(other))

    def __mul__(self, other: float | ReverseVariable) -> ReverseVariable:
        """Record ``a c`` (partials ``c, a``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, ReverseVariable):
            tape = self._tape_of(other)
            a, c = self.value, other.value
            return tape.push(a * c, (self.index, other.index), (c, a))
        if _is_constant(other):
            return self.tape.push(self.value * other, (self.index,), (other,))
        return cast(ReverseVariable, _foreign(other))

    def __rmul__(self, other: float) -> ReverseVariable:
        """Record ``c a`` for a constant ``c``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if _is_constant(other):
            return self.tape.push(other * self.value, (self.index,), (other,))
        return cast(ReverseVariable, _foreign(other))

    def __truediv__(self, other: float | ReverseVariable) -> ReverseVariable:
        """Record ``q = a / c`` (partials ``1/c, -q/c``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, ReverseVariable):
            tape = self._tape_of(other)
            c = other.value
            quotient = self.value / c
            inverse = 1.0 / c
            return tape.push(
                quotient, (self.index, other.index), (inverse, -quotient * inverse)
            )
        if _is_constant(other):
            return self.tape.push(self.value / other, (self.index,), (1.0 / other,))
        return cast(ReverseVariable, _foreign(other))

    def __rtruediv__(self, other: float) -> ReverseVariable:
        """Record ``q = c / a`` for a constant ``c`` (partial ``-q/a``).

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if _is_constant(other):
            quotient = other / self.value
            return self._unary(quotient, -quotient / self.value)
        return cast(ReverseVariable, _foreign(other))

    def __pow__(self, other: float | ReverseVariable) -> ReverseVariable:
        """Record ``a ** c`` by the power rule (constant ``c``) or ``exp(c log a)``.

        A constant exponent has partial ``c a^(c-1)`` (valid for negative
        ``a`` and integer ``c``; ``c = 0`` gives partial 0). A variable
        exponent has partials ``c a^(c-1)`` and ``a^c log a`` and needs
        ``a > 0``.

        Raises:
            NonDifferentiableError: For ``a = 0`` with ``0 < c < 1`` or
                ``c < 0``, and for a variable exponent with ``a <= 0``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if isinstance(other, ReverseVariable):
            tape = self._tape_of(other)
            a, c = self.value, other.value
            if not cast(float, _primal(a)) > 0.0:
                raise _not_differentiable("x ** y with a variable exponent", a)
            power = _real_pow(a, c)
            return tape.push(
                power,
                (self.index, other.index),
                (c * _real_pow(a, c - 1.0), power * _log(a)),
            )
        if other.__class__ is float or other.__class__ is int:
            a = self.value
            if a.__class__ is float:
                # Fast path for a float value and a float or int exponent.
                if other == 2:
                    return self.tape.push(a * a, (self.index,), (2.0 * a,))
                if other != 0 and not (a == 0.0 and other < 1):
                    return self.tape.push(
                        math.pow(a, other),
                        (self.index,),
                        (other * math.pow(a, other - 1.0),),
                    )
        if _is_constant(other):
            exponent = other
            if exponent == 0.0:
                return self._unary(_real_pow(self.value, 0.0), 0.0)
            if _primal(self) == 0.0 and exponent < 1.0:
                raise _not_differentiable(f"x ** {exponent!r}", self)
            return self._unary(
                _real_pow(self.value, exponent),
                exponent * _real_pow(self.value, exponent - 1.0),
            )
        return cast(ReverseVariable, _foreign(other))

    def __rpow__(self, other: float) -> ReverseVariable:
        """Record ``c ** a`` for a constant ``c`` (partial ``c^a log c``).

        ``c = 0`` gives partial 0 (for ``a > 0``, as in forward mode).

        Raises:
            NonDifferentiableError: For a negative base ``c``.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if _is_constant(other):
            base = other
            if base < 0.0:
                raise _not_differentiable("c ** x with a negative base c", self)
            power = _real_pow(base, self.value)
            if base == 0.0:
                return self._unary(power, 0.0)
            return self._unary(power, power * math.log(base))
        return cast(ReverseVariable, _foreign(other))


# ---------------------------------------------------------------------------
# Elementary-function rules (value and local partial on floats or duals)
# ---------------------------------------------------------------------------


def _rule_exp(x: ReverseVariable) -> ReverseVariable:
    """Record ``exp(x)``; partial ``exp(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    value = _exp(x.value)
    return x._unary(value, value)


def _rule_expm1(x: ReverseVariable) -> ReverseVariable:
    """Record ``exp(x) - 1``; partial ``exp(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_expm1(x.value), _exp(x.value))


def _rule_log(x: ReverseVariable) -> ReverseVariable:
    """Record ``log(x)``; partial ``1/x``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_log(x.value), 1.0 / x.value)


def _rule_log1p(x: ReverseVariable) -> ReverseVariable:
    """Record ``log(1 + x)``; partial ``1/(1 + x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_log1p(x.value), 1.0 / (1.0 + x.value))


def _rule_log2(x: ReverseVariable) -> ReverseVariable:
    """Record ``log2(x)``; partial ``1/(x log 2)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_log2(x.value), 1.0 / (x.value * _LOG_TWO))


def _rule_log10(x: ReverseVariable) -> ReverseVariable:
    """Record ``log10(x)``; partial ``1/(x log 10)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_log10(x.value), 1.0 / (x.value * _LOG_TEN))


def _rule_sqrt(x: ReverseVariable) -> ReverseVariable:
    """Record ``sqrt(x)``; partial ``1/(2 sqrt(x))``.

    Raises:
        NonDifferentiableError: At ``x = 0``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    root = _sqrt(x.value)
    if _primal(root) == 0.0:
        raise _not_differentiable("sqrt", x)
    return x._unary(root, 0.5 / root)


def _rule_sin(x: ReverseVariable) -> ReverseVariable:
    """Record ``sin(x)``; partial ``cos(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_sin(x.value), _cos(x.value))


def _rule_cos(x: ReverseVariable) -> ReverseVariable:
    """Record ``cos(x)``; partial ``-sin(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_cos(x.value), -_sin(x.value))


def _rule_tan(x: ReverseVariable) -> ReverseVariable:
    """Record ``tan(x)``; partial ``1 + tan(x)^2``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    value = _tan(x.value)
    return x._unary(value, 1.0 + value * value)


def _rule_asin(x: ReverseVariable) -> ReverseVariable:
    """Record ``asin(x)``; partial ``1/sqrt(1 - x^2)``.

    Raises:
        NonDifferentiableError: At ``x = +-1``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    value = _asin(x.value)
    if abs(cast(float, _primal(x))) == 1.0:
        raise _not_differentiable("asin", x)
    return x._unary(value, 1.0 / _sqrt(1.0 - x.value * x.value))


def _rule_acos(x: ReverseVariable) -> ReverseVariable:
    """Record ``acos(x)``; partial ``-1/sqrt(1 - x^2)``.

    Raises:
        NonDifferentiableError: At ``x = +-1``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    value = _acos(x.value)
    if abs(cast(float, _primal(x))) == 1.0:
        raise _not_differentiable("acos", x)
    return x._unary(value, -1.0 / _sqrt(1.0 - x.value * x.value))


def _rule_atan(x: ReverseVariable) -> ReverseVariable:
    """Record ``atan(x)``; partial ``1/(1 + x^2)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_atan(x.value), 1.0 / (1.0 + x.value * x.value))


def _rule_sinh(x: ReverseVariable) -> ReverseVariable:
    """Record ``sinh(x)``; partial ``cosh(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_sinh(x.value), _cosh(x.value))


def _rule_cosh(x: ReverseVariable) -> ReverseVariable:
    """Record ``cosh(x)``; partial ``sinh(x)``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    return x._unary(_cosh(x.value), _sinh(x.value))


def _rule_tanh(x: ReverseVariable) -> ReverseVariable:
    """Record ``tanh(x)``; partial ``1 - tanh(x)^2``.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    value = _tanh(x.value)
    return x._unary(value, 1.0 - value * value)


def _rule_erf(x: ReverseVariable) -> ReverseVariable:
    """Record ``erf(x)``; partial ``2 exp(-x^2) / sqrt(pi)``.

    References:
        [EQ-AD-REVERSE-001], [REF-NIST-DLMF, Section 7.10].

    """
    a = x.value
    return x._unary(_erf(a), _TWO_OVER_SQRT_PI * _exp(-a * a))


def _rule_erfc(x: ReverseVariable) -> ReverseVariable:
    """Record ``erfc(x)``; partial ``-2 exp(-x^2) / sqrt(pi)``.

    References:
        [EQ-AD-REVERSE-001], [REF-NIST-DLMF, Section 7.10].

    """
    a = x.value
    return x._unary(_erfc(a), -_TWO_OVER_SQRT_PI * _exp(-a * a))


def _rule_lgamma(x: ReverseVariable) -> ReverseVariable:
    """Record ``log |Gamma(x)|``; partial ``digamma(x)`` (needs ``x > 0``).

    References:
        [EQ-AD-REVERSE-001], [EQ-NUM-DIGAMMA-001],
        [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    return x._unary(_lgamma(x.value), _digamma(x.value))


def _rule_digamma(x: ReverseVariable) -> ReverseVariable:
    """Record ``psi(x)``; partial ``trigamma(x)`` (needs ``x > 0``).

    The forward-over-reverse Hessian of ``digamma`` would need a dual
    ``trigamma`` and raises :class:`AutodiffUnsupportedError`.

    References:
        [EQ-AD-REVERSE-001], [EQ-NUM-DIGAMMA-001],
        [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    return x._unary(_digamma(x.value), _trigamma(x.value))


def _rule_trigamma(x: ReverseVariable) -> ReverseVariable:
    """Refuse ``trigamma`` of a variable (its derivative is not available).

    Raises:
        AutodiffUnsupportedError: Always.

    References:
        [EQ-NUM-DIGAMMA-001], [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    raise AutodiffUnsupportedError(
        "trigamma of a ReverseVariable (its derivative psi'' is not implemented)"
    )


_UNARY_RULES: dict[str, Callable[[ReverseVariable], ReverseVariable]] = {
    "exp": _rule_exp,
    "expm1": _rule_expm1,
    "log": _rule_log,
    "log1p": _rule_log1p,
    "log2": _rule_log2,
    "log10": _rule_log10,
    "sqrt": _rule_sqrt,
    "sin": _rule_sin,
    "cos": _rule_cos,
    "tan": _rule_tan,
    "asin": _rule_asin,
    "acos": _rule_acos,
    "atan": _rule_atan,
    "sinh": _rule_sinh,
    "cosh": _rule_cosh,
    "tanh": _rule_tanh,
    "erf": _rule_erf,
    "erfc": _rule_erfc,
    "lgamma": _rule_lgamma,
    "digamma": _rule_digamma,
    "trigamma": _rule_trigamma,
}


def _split_operands(
    arguments: Sequence[object],
) -> tuple[_Tape, list[Scalar], list[tuple[int, int]]]:
    """Return the tape, the values, and ``(position, tape index)`` of variables.

    Raises:
        AutodiffUnsupportedError: For variables of different tapes or a dual.
        TypeError: For a non-numeric argument.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    tape: _Tape | None = None
    values: list[Scalar] = []
    variables: list[tuple[int, int]] = []
    for position, argument in enumerate(arguments):
        if isinstance(argument, ReverseVariable):
            if tape is None:
                tape = argument.tape
            elif argument.tape is not tape:
                raise AutodiffUnsupportedError(
                    "ReverseVariables from different tapes cannot be combined"
                )
            values.append(argument.value)
            variables.append((position, argument.index))
        elif isinstance(argument, Dual):
            _foreign(argument)
        elif _is_constant(argument):
            values.append(float(cast(float, argument)))
        else:
            raise TypeError(
                f"admath arguments must be numbers, got {type(argument).__name__}"
            )
    assert tape is not None
    return tape, values, variables


def _record_atan2(y: object, x: object) -> ReverseVariable:
    """Record ``atan2(y, x)``; partials ``x/r^2`` and ``-y/r^2``.

    Raises:
        NonDifferentiableError: At the origin.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    tape, (y_value, x_value), variables = _split_operands((y, x))
    radius = x_value * x_value + y_value * y_value
    if _primal(radius) == 0.0:
        raise _not_differentiable("atan2", radius)
    slopes = (x_value / radius, -y_value / radius)
    parents = tuple(index for _, index in variables)
    partials = tuple(slopes[position] for position, _ in variables)
    return tape.push(_atan2(y_value, x_value), parents, partials)


def _record_hypot(coordinates: Sequence[object]) -> ReverseVariable:
    """Record the Euclidean norm; partials ``x_i / hypot`` (0 at the origin).

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    tape, values, variables = _split_operands(coordinates)
    value = _hypot(*values)
    parents = tuple(index for _, index in variables)
    if _primal(value) == 0.0:
        zero: tuple[Scalar, ...] = tuple(0.0 for _ in variables)
        return tape.push(value, parents, zero)
    partials = tuple(values[position] / value for position, _ in variables)
    return tape.push(value, parents, partials)


def _record_fsum(items: Sequence[object]) -> ReverseVariable:
    """Record an accurately rounded sum as one node with unit partials.

    References:
        [EQ-AD-REVERSE-001], [REF-PYTHON-MATH].

    """
    tape, values, variables = _split_operands(items)
    parents = tuple(index for _, index in variables)
    return tape.push(_fsum(values), parents, tuple(1.0 for _ in variables))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

ScalarFunction = Callable[[tuple[ReverseVariable, ...]], object]
VectorFunction = Callable[[tuple[ReverseVariable, ...]], Sequence[object]]


def _record(
    function: Callable[[tuple[ReverseVariable, ...]], object],
    values: Sequence[Scalar],
) -> tuple[_Tape, object]:
    """Create a tape with one input per value and run ``function`` on it.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    tape = _Tape()
    inputs = tuple(tape.push(value, (), ()) for value in values)
    return tape, function(inputs)


def _output(output: object, tape: _Tape) -> tuple[Scalar, int | None]:
    """Return the value of a recorded output and its tape index.

    A plain real output (the function does not depend on its inputs) has
    index ``None``.

    Raises:
        TypeError: For an output that is neither real nor a variable of
            ``tape`` (for example a dual).

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    if isinstance(output, ReverseVariable):
        if output.tape is not tape:
            raise TypeError("function returned a ReverseVariable of another tape")
        return output.value, output.index
    if _is_constant(output) and not isinstance(output, bool):
        return float(cast(float, output)), None
    raise TypeError(
        "function must return a real number or a ReverseVariable, got "
        f"{type(output).__name__}"
    )


def _record_scalar(
    function: ScalarFunction, point: tuple[float, ...]
) -> tuple[float, _Tape, int | None]:
    """Record a scalar function at a float point.

    Returns:
        ``(f(x), tape, output index)``.

    References:
        [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    tape, output = _record(function, point)
    value, index = _output(output, tape)
    return cast(float, value), tape, index


def _input_adjoints(
    tape: _Tape, index: int | None, size: int
) -> tuple[float, ...]:
    """Return the input adjoints of a scalar output (zeros for a constant).

    References:
        [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    if index is None:
        return (0.0,) * size
    adjoints = tape.sweep({index: 1.0})
    return tuple(
        float(cast(float, adjoints[i])) if i < len(adjoints) else 0.0
        for i in range(size)
    )


def reverse_value_and_gradient(
    function: ScalarFunction, x: Sequence[float]
) -> tuple[float, tuple[float, ...]]:
    """Return ``f(x)`` and ``grad f(x)`` from one recording and one sweep.

    ``f`` is called once with :class:`ReverseVariable` inputs; the backward
    sweep from ``ybar = 1`` returns every partial derivative, exact up to
    rounding, at a cost independent of ``d``.

    Args:
        function: Scalar function of a tuple of numbers written with
            arithmetic operators and :class:`~probstat.numerics.admath`.
        x: Finite evaluation point.

    Returns:
        ``(f(x), grad f(x))``. A real (non-variable) result means ``f`` does
        not depend on ``x``; its gradient is zero.

    Raises:
        AutodiffUnsupportedError: If ``f`` converts a variable to ``float``
            (for example through ``math.exp``).
        NonDifferentiableError: At a point with an unbounded derivative.
        TypeError: If ``f`` returns something other than a number or a
            variable of its tape.

    Equation:
        [EQ-AD-REVERSE-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
        Section 3.2].

    """
    point = _point(x)
    value, tape, index = _record_scalar(function, point)
    return value, _input_adjoints(tape, index, len(point))


def reverse_gradient(function: ScalarFunction, x: Sequence[float]) -> tuple[float, ...]:
    """Return the exact gradient of a scalar function by reverse mode.

    Args:
        function: Scalar function written with operators and
            :class:`~probstat.numerics.admath`.
        x: Finite evaluation point.

    Returns:
        ``grad f(x)``.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables.
        NonDifferentiableError: At a point with an unbounded derivative.

    Equation:
        [EQ-AD-REVERSE-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    return reverse_value_and_gradient(function, x)[1]


def _vector_outputs(
    function: VectorFunction, point: tuple[float, ...]
) -> tuple[_Tape, list[tuple[Scalar, int | None]]]:
    """Record a vector function and return its output values and indices.

    Raises:
        TypeError: If the function does not return a sequence.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    tape, outputs = _record(function, point)
    if isinstance(outputs, str | bytes) or not isinstance(outputs, Sequence):
        raise TypeError("function must return a sequence")
    return tape, [_output(item, tape) for item in outputs]


def vector_jacobian_product(
    function: VectorFunction, x: Sequence[float], cotangent: Sequence[float]
) -> tuple[float, ...]:
    """Return ``w' J(x)`` of a vector function from one sweep.

    The adjoint of output ``k`` is seeded with ``w_k``; one backward sweep
    returns ``sum_k w_k grad f_k(x)`` without forming ``J``.

    Args:
        function: Function returning a sequence of numbers or variables.
        x: Finite evaluation point of dimension ``d``.
        cotangent: Weights ``w``, one per output.

    Returns:
        The product ``w' J`` (length ``d``).

    Raises:
        ValueError: If ``cotangent`` does not match the number of outputs or
            is not finite.
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables.

    Equation:
        [EQ-AD-REVERSE-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
        Section 3.2].

    """
    point = _point(x)
    if isinstance(cotangent, str | bytes) or not isinstance(cotangent, Sequence):
        raise TypeError("cotangent must be a sequence of real numbers")
    weights = tuple(
        require_finite_real(w, name=f"cotangent[{i}]") for i, w in enumerate(cotangent)
    )
    tape, outputs = _vector_outputs(function, point)
    if len(weights) != len(outputs):
        raise ValueError(
            f"cotangent has length {len(weights)}, function returned "
            f"{len(outputs)} outputs"
        )
    seeds: dict[int, Scalar] = {}
    for (_, index), weight in zip(outputs, weights, strict=True):
        if index is not None:
            seeds[index] = seeds.get(index, 0.0) + weight
    adjoints = tape.sweep(seeds)
    return tuple(
        float(cast(float, adjoints[i])) if i < len(adjoints) else 0.0
        for i in range(len(point))
    )


def reverse_jacobian(
    function: VectorFunction, x: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Return the Jacobian ``J_ij = d f_i / d x_j`` from one recording.

    The function is evaluated once; each row is one backward sweep seeded at
    one output (``m`` sweeps for ``m`` outputs).

    Args:
        function: Function returning a sequence of numbers or variables.
        x: Finite evaluation point.

    Returns:
        ``m x d`` Jacobian as a tuple of rows.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables.
        TypeError: If the output is not a sequence.

    Equation:
        [EQ-AD-REVERSE-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    point = _point(x)
    tape, outputs = _vector_outputs(function, point)
    return tuple(_input_adjoints(tape, index, len(point)) for _, index in outputs)


def _second_order(
    function: ScalarFunction, inputs: Sequence[Dual], size: int
) -> tuple[Scalar, list[Scalar]]:
    """Record on dual values and return the output value and input adjoints.

    References:
        [EQ-AD-FOR-001], [REF-GRIEWANK-WALTHER-2008, Chapter 5].

    """
    tape, output = _record(function, inputs)
    value, index = _output(output, tape)
    if index is None:
        return value, [0.0] * size
    adjoints = tape.sweep({index: 1.0})
    adjoints.extend([0.0] * max(0, size - len(adjoints)))
    return value, adjoints[:size]


def reverse_value_gradient_and_hessian(
    function: ScalarFunction, x: Sequence[float]
) -> tuple[float, tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return ``f(x)``, its gradient and its Hessian by forward over reverse.

    The tape is recorded on the first-order seeds ``x_i + e_i eps`` and swept
    once; the adjoint of input ``i`` is the dual
    ``d f / d x_i + sum_j H_ij eps_j``, so its tangent is row ``i`` of the
    Hessian. The result is symmetrized, ``(H + H')/2``, which only removes
    rounding asymmetry. The cost is about ``d`` times one gradient.

    Args:
        function: Scalar function written with operators and
            :class:`~probstat.numerics.admath` (twice differentiable at ``x``).
        x: Finite evaluation point.

    Returns:
        ``(f(x), grad f(x), Hessian)``.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables or needs a derivative that is not available (the
            Hessian of ``digamma``).
        NonDifferentiableError: At a point with an unbounded derivative.

    Equation:
        [EQ-AD-FOR-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 5], [REF-PEARLMUTTER-1994].

    """
    point = _point(x)
    size = len(point)
    value, adjoints = _second_order(function, _seeds(point), size)
    zero_row = (0.0,) * size
    rows: list[tuple[float, ...]] = []
    grad: list[float] = []
    for adjoint in adjoints:
        if isinstance(adjoint, Dual):
            grad.append(cast(float, adjoint.value))
            rows.append(cast(tuple[float, ...], adjoint.tangent))
        else:
            grad.append(float(adjoint))
            rows.append(zero_row)
    hessian_matrix = tuple(
        tuple(0.5 * (rows[i][j] + rows[j][i]) for j in range(size))
        for i in range(size)
    )
    return cast(float, _primal(value)), tuple(grad), hessian_matrix


def reverse_hessian(
    function: ScalarFunction, x: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Return the exact Hessian of a scalar function (forward over reverse).

    Args:
        function: Scalar function written with operators and
            :class:`~probstat.numerics.admath`.
        x: Finite evaluation point.

    Returns:
        Symmetric ``d x d`` Hessian.

    Raises:
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables.

    Equation:
        [EQ-AD-FOR-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 5].

    """
    return reverse_value_gradient_and_hessian(function, x)[2]


def hessian_vector_product(
    function: ScalarFunction, x: Sequence[float], direction: Sequence[float]
) -> tuple[float, ...]:
    """Return ``H(x) v`` from one forward-over-reverse sweep.

    The inputs carry the single tangent ``x_i + v_i eps``; the tangents of
    the input adjoints are ``(H v)_i``. The cost is a small multiple of one
    gradient, independent of ``d``, and ``H`` is never formed.

    Args:
        function: Scalar function written with operators and
            :class:`~probstat.numerics.admath`.
        x: Finite evaluation point.
        direction: Vector ``v`` of the same length as ``x``.

    Returns:
        The product ``H v``.

    Raises:
        ValueError: If ``direction`` has the wrong length or is not finite.
        AutodiffUnsupportedError: If ``f`` does not support reverse-mode
            variables.

    Equation:
        [EQ-AD-FOR-001].

    References:
        [REF-PEARLMUTTER-1994], [REF-GRIEWANK-WALTHER-2008, Chapter 5].

    """
    point = _point(x)
    vector = _point(direction, "direction")
    if len(vector) != len(point):
        raise ValueError("direction must have the same length as x")
    inputs = [_new(value, (v,), 1) for value, v in zip(point, vector, strict=True)]
    _, adjoints = _second_order(function, inputs, len(point))
    return tuple(
        cast(float, adjoint.tangent[0]) if isinstance(adjoint, Dual) else 0.0
        for adjoint in adjoints
    )
