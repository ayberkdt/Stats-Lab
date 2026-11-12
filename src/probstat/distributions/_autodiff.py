"""Automatic-differentiation helpers for distribution log-densities (ADR-0090).

The log-density methods (``log_pdf``, ``log_pmf``) of the families accept
:class:`~probstat.numerics.Dual` numbers and reverse-mode
:class:`~probstat.numerics.ReverseVariable` values as parameters and, for
continuous laws, as the evaluation point. Validation reads the innermost
float of such a *variable* (its primal value) and keeps the variable itself,
so the log-density, written with :class:`~probstat.numerics.admath`, carries
exact derivatives ([EQ-AD-DUAL-001], [EQ-AD-REVERSE-001]). For plain floats
every helper returns exactly what the float-only code returned, so float
results are bit-identical.

Where a family evaluates its log-density through a float-only kernel (a
Stirling-series or cancellation-free form for large shapes), the variable
result takes its value from that kernel at the primal parameters and its
derivatives from the closed-form expression ([EQ-DIST-AD-001]).

References:
    [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
    Section 3.2].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import cast

from probstat.foundations.validation import require_finite_real
from probstat.numerics import Dual, ReverseVariable
from probstat.numerics import admath as _admath
from probstat.numerics.autodiff import _primal

VARIABLE_TYPES = (Dual, ReverseVariable)
"""Numeric types that carry derivatives through a log-density."""

_Unary = Callable[[float], float]


class admath:  # noqa: N801 - mirrors the name of probstat.numerics.admath
    """The :class:`probstat.numerics.admath` functions, typed for float code.

    The distribution classes annotate parameters as ``float``; a dual number
    or reverse-mode variable stands in for a float at run time. This
    namespace exposes the same function objects with float signatures, so
    a log-density reads like float code and type-checks. For float
    arguments every function is the :mod:`math` function itself.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-PYTHON-MATH].

    """

    log = staticmethod(cast(_Unary, _admath.log))
    log1p = staticmethod(cast(_Unary, _admath.log1p))
    exp = staticmethod(cast(_Unary, _admath.exp))
    lgamma = staticmethod(cast(_Unary, _admath.lgamma))
    sqrt = staticmethod(cast(_Unary, _admath.sqrt))
    erf = staticmethod(cast(_Unary, _admath.erf))
    erfc = staticmethod(cast(_Unary, _admath.erfc))
    pow = staticmethod(cast(Callable[[float, float], float], _admath.pow))
    fsum = staticmethod(cast(Callable[[Iterable[float]], float], _admath.fsum))
    isfinite = staticmethod(cast(Callable[[float], bool], _admath.isfinite))


def is_variable(value: object) -> bool:
    """Return whether ``value`` is a dual number or a reverse-mode variable.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    return isinstance(value, VARIABLE_TYPES)


def primal(value: object) -> float:
    """Return the innermost float of a float, dual, or reverse-mode variable.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    return cast(float, _primal(value))


def real_argument(value: object, *, name: str) -> float:
    """Validate a finite real argument that may be a variable.

    A float (or other real) is validated and returned exactly as
    :func:`~probstat.foundations.validation.require_finite_real` returns it.
    A variable is validated on its primal value and returned unchanged, so
    its derivatives survive.

    Args:
        value: Candidate real number, dual, or reverse-mode variable.
        name: Argument name used in error messages.

    Returns:
        The float, or the variable itself.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If its (primal) value is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION],
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    if isinstance(value, VARIABLE_TYPES):
        require_finite_real(primal(value), name=name)
        return cast(float, value)
    return require_finite_real(value, name=name)


def with_value(value: float, expression: float) -> float:
    """Return ``value`` carrying the derivatives of ``expression``.

    ``value + (g - g0)`` with ``g0`` the primal of ``g`` has the value
    ``value`` and every derivative of ``g`` ([EQ-DIST-AD-001]). For a float
    ``expression`` the float ``value`` is returned unchanged.

    Args:
        value: Accurate float value of the quantity.
        expression: The same quantity written with ``admath`` on variables.

    Returns:
        ``value`` or a variable with that value.

    Equation:
        [EQ-DIST-AD-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    if not isinstance(expression, VARIABLE_TYPES):
        return value
    return cast(float, (expression - primal(expression)) + value)


def any_variable(*values: object) -> bool:
    """Return whether any of ``values`` is a dual or reverse-mode variable.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    return any(isinstance(value, VARIABLE_TYPES) for value in values)


def transfer_derivatives(accurate: float, closed_form: Callable[[], float]) -> float:
    """Combine an accurate float log-density with closed-form derivatives.

    A non-finite ``accurate`` value (outside the support, or an endpoint
    singularity) is returned as it is; otherwise the result has the value
    ``accurate`` and the derivatives of ``closed_form()``
    ([EQ-DIST-AD-001]).

    Args:
        accurate: Float log-density at the primal values, from the
            float-only implementation.
        closed_form: Zero-argument callable returning the same log-density
            written with ``admath`` on the variables.

    Returns:
        ``accurate`` or a variable with that value.

    Equation:
        [EQ-DIST-AD-001].

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    if accurate in (math.inf, -math.inf) or accurate != accurate:
        return accurate
    return with_value(accurate, closed_form())
