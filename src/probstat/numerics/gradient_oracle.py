"""Value-and-gradient oracle choosing forward mode, reverse mode, or differences.

:class:`GradientEvaluator` is the single place where the inference engines
(HMC, NUTS, ADVI, the probabilistic-programming layer) turn a log-density
into a value and a gradient. It lives in its own module because it uses both
:mod:`probstat.numerics.autodiff` (forward mode, ADR-0065) and
:mod:`probstat.numerics.reverse_autodiff` (reverse mode, ADR-0069), and the
latter builds on the former.

Options (``GRADIENT_OPTIONS``)
    ``"auto"``
        forward mode; if the function rejects dual numbers with a
        ``TypeError`` while its float evaluation succeeds, switch permanently
        to finite differences and record the reason.
    ``"auto-reverse"``
        the same with reverse mode.
    ``"autodiff"`` / ``"forward"``
        forward mode only (raises if unsupported).
    ``"reverse"``
        reverse mode only (raises if unsupported).
    ``"finite-difference"`` / ``"finite"``
        always the injected finite-difference callable.
    a callable
        an analytic gradient.

The recorded ``method`` is ``"autodiff"`` (forward mode, the label used since
ADR-0065), ``"reverse"``, ``"finite-difference"``, or ``"analytic"``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

from .autodiff import AutodiffUnsupportedError, _first_order, _seeds
from .reverse_autodiff import _input_adjoints, _record_scalar

GradientCallable = Callable[[tuple[float, ...]], Sequence[float]]
FiniteDifference = Callable[
    [Callable[[tuple[float, ...]], float], tuple[float, ...]], Sequence[float]
]

GRADIENT_OPTIONS: tuple[str, ...] = (
    "auto",
    "auto-reverse",
    "autodiff",
    "forward",
    "reverse",
    "finite-difference",
    "finite",
)
_ALIASES = {"forward": "autodiff", "finite": "finite-difference"}
_AUTOMATIC = {"auto": "autodiff", "auto-reverse": "reverse"}


class GradientEvaluator:
    """Value-and-gradient oracle: analytic, autodiff, or finite differences.

    ``option`` selects the gradient (see the module docstring for the full
    list): ``"auto"`` (default, forward mode with finite-difference
    fallback), ``"auto-reverse"`` (reverse mode with fallback), ``"autodiff"``
    or ``"forward"`` and ``"reverse"`` (one mode, no fallback),
    ``"finite-difference"`` or ``"finite"``, or a callable analytic gradient.
    A fallback happens only when evaluating the function on dual numbers or
    reverse-mode variables raises a ``TypeError`` (including
    :class:`~probstat.numerics.AutodiffUnsupportedError`) while the plain
    float evaluation succeeds; errors that also occur for floats propagate
    unchanged.

    The evaluator never counts function calls itself; wrap ``function`` in a
    counter to account for them. One dual or one reverse-mode evaluation is
    one call; the backward sweep of reverse mode does not call the function.

    Args:
        function: Scalar function of a tuple of numbers.
        option: Gradient choice (see above).
        finite_difference: ``(f, x) -> grad`` used by the finite-difference
            modes (required for ``"auto"``, ``"auto-reverse"`` and the
            finite-difference options).

    Raises:
        TypeError: If ``function`` or a callable ``option`` is not callable.
        ValueError: For an unknown option or a missing ``finite_difference``.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
        Section 3], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

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
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

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
        if option not in GRADIENT_OPTIONS:
            raise ValueError(
                "gradient must be one of " + ", ".join(map(repr, GRADIENT_OPTIONS))
                + ", or a callable"
            )
        requested = _ALIASES.get(option, option)
        if requested not in ("autodiff", "reverse") and finite_difference is None:
            raise ValueError("finite_difference is required for this option")
        self.requested = requested
        self.method = _AUTOMATIC.get(requested, requested)

    def _float_value(self, x: tuple[float, ...]) -> float:
        """Evaluate the function on plain floats.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return float(cast(float, self._function(x)))

    def _automatic(
        self, x: tuple[float, ...]
    ) -> tuple[float, tuple[float, ...] | None]:
        """Return value and gradient by the current automatic mode.

        The gradient is ``None`` for a non-finite value; reverse mode then
        skips the backward sweep.

        References:
            [EQ-AD-DUAL-001], [EQ-AD-REVERSE-001],
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        if self.method == "reverse":
            value, tape, index = _record_scalar(self._function, x)
            if not math.isfinite(value):
                return value, None
            return value, _input_adjoints(tape, index, len(x))
        value, grad = _first_order(self._function(_seeds(x)), len(x))
        if not math.isfinite(value):
            return value, None
        return value, grad

    def value_and_gradient(
        self, x: Sequence[float]
    ) -> tuple[float, tuple[float, ...] | None]:
        """Return ``(f(x), grad f(x))``, with ``None`` for a non-finite ``f(x)``.

        Args:
            x: Evaluation point.

        Returns:
            Value and gradient (``None`` when the value is not finite, so no
            derivative work is wasted outside the support).

        Raises:
            AutodiffUnsupportedError: In ``"autodiff"``/``"forward"`` or
                ``"reverse"`` mode for a function that does not support it.
            ValueError: If an analytic gradient has the wrong length.

        Equation:
            [EQ-AD-DUAL-001], [EQ-AD-REVERSE-001], [EQ-OPT-FD-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3],
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = tuple(float(v) for v in x)
        if self.method in ("autodiff", "reverse"):
            try:
                value, grad = self._automatic(point)
            except TypeError as error:
                if self.requested in ("autodiff", "reverse"):
                    if isinstance(error, AutodiffUnsupportedError):
                        raise
                    mode = (
                        "Dual numbers"
                        if self.method == "autodiff"
                        else "reverse-mode variables"
                    )
                    raise AutodiffUnsupportedError(
                        f"function does not support {mode}: {error}"
                    ) from error
                value = self._float_value(point)
                self.method = "finite-difference"
                self.fallback_reason = f"{type(error).__name__}: {error}"
            else:
                if grad is None:
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
            [EQ-AD-DUAL-001], [EQ-AD-REVERSE-001], [EQ-OPT-FD-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        value, grad = self.value_and_gradient(x)
        if grad is None:
            raise ValueError(f"function value {value} is not finite")
        return grad
