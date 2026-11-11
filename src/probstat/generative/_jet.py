"""Forward propagation of value, gradient, and Laplacian through a program.

Score matching needs the derivative with respect to the network parameters
of ``|grad_x E|^2 / 2 - Laplacian_x E``, a third-order quantity of the
energy. The reverse-mode tape of :mod:`probstat.numerics` does not nest, so
the input derivatives are propagated forward instead: a :class:`LaplacianJet`
carries ``(f, grad f, Laplacian f)`` of an intermediate value with respect
to the input point, and every operation applies the chain and product rules
of [EQ-GEN-EBM-005]. The components themselves may be reverse-mode tape
variables of the parameters, so one backward sweep of the resulting scalar
returns the exact parameter gradient of the objective.

The jet subclasses the operand hook of :mod:`probstat.numerics.autodiff`, so
:class:`~probstat.numerics.admath` functions dispatch to it and any program
written for automatic differentiation, in particular
:meth:`probstat.numerics.MLP.forward`, runs on jets unchanged.

References:
    [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..numerics import AutodiffUnsupportedError, Dual, ReverseVariable, admath
from ..numerics.autodiff import _TapeOperand
from ._common import primal


def _is_scalar(value: object) -> bool:
    """Return whether ``value`` is a constant with respect to the input point.

    Plain reals and parameter tape variables are constants of the jet.

    References:
        [EQ-GEN-EBM-005]. No equation is evaluated.

    """
    if isinstance(value, bool):
        return False
    return isinstance(value, int | float | ReverseVariable)


class LaplacianJet(_TapeOperand):
    """Value, input gradient, and input Laplacian of one intermediate result.

    ``laplacian`` is ``None`` in gradient-only mode (denoising score
    matching needs no second derivative).

    Args:
        value: Value of the intermediate result.
        gradient: Gradient with respect to the input point.
        laplacian: Sum of the unmixed second derivatives, or ``None``.

    References:
        [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

    """

    __slots__ = ("gradient", "laplacian", "value")

    value: Any
    gradient: tuple[Any, ...]
    laplacian: Any

    def __init__(
        self, value: Any, gradient: tuple[Any, ...], laplacian: Any = None
    ) -> None:
        """Store the components.

        References:
            [EQ-GEN-EBM-005]. No equation is evaluated.

        """
        self.value = value
        self.gradient = gradient
        self.laplacian = laplacian

    # -- seeds ---------------------------------------------------------------

    @classmethod
    def seeds(
        cls, point: Sequence[float], *, laplacian: bool
    ) -> tuple[LaplacianJet, ...]:
        """Return the input jets ``(x_i, e_i, 0)``.

        Args:
            point: Input point.
            laplacian: Whether to propagate the Laplacian.

        Returns:
            One jet per coordinate.

        References:
            [EQ-GEN-EBM-005]. No equation is evaluated.

        """
        size = len(point)
        start = 0.0 if laplacian else None
        return tuple(
            cls(
                float(value),
                tuple(1.0 if j == i else 0.0 for j in range(size)),
                start,
            )
            for i, value in enumerate(point)
        )

    # -- helpers -------------------------------------------------------------

    def _scale(self, factor: Any) -> LaplacianJet:
        """Return ``factor * self`` for a constant factor.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        laplacian = None if self.laplacian is None else self.laplacian * factor
        return LaplacianJet(
            self.value * factor,
            tuple(entry * factor for entry in self.gradient),
            laplacian,
        )

    def _shift(self, constant: Any) -> LaplacianJet:
        """Return ``self + constant`` for a constant.

        References:
            [EQ-GEN-EBM-005]. No new equation is introduced.

        """
        return LaplacianJet(self.value + constant, self.gradient, self.laplacian)

    def _compose(self, value: Any, first: Any, second: Any) -> LaplacianJet:
        """Return the jet of ``phi(self)`` from ``phi``, ``phi'`` and ``phi''``.

        References:
            [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

        """
        gradient = tuple(entry * first for entry in self.gradient)
        if self.laplacian is None:
            return LaplacianJet(value, gradient, None)
        squared: Any = 0.0
        for entry in self.gradient:
            squared = squared + entry * entry
        return LaplacianJet(value, gradient, second * squared + first * self.laplacian)

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: object) -> LaplacianJet:
        """Return the jet of a sum.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if isinstance(other, LaplacianJet):
            laplacian = (
                None
                if self.laplacian is None or other.laplacian is None
                else self.laplacian + other.laplacian
            )
            return LaplacianJet(
                self.value + other.value,
                tuple(
                    a + b for a, b in zip(self.gradient, other.gradient, strict=True)
                ),
                laplacian,
            )
        if _is_scalar(other):
            return self._shift(other)
        return NotImplemented

    def __radd__(self, other: object) -> LaplacianJet:
        """Return the jet of ``constant + self``.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if _is_scalar(other):
            return self._shift(other)
        return NotImplemented

    def __neg__(self) -> LaplacianJet:
        """Return the jet of ``-self``.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        return self._scale(-1.0)

    def __pos__(self) -> LaplacianJet:
        """Return the jet unchanged.

        References:
            [EQ-GEN-EBM-005]. No new equation is introduced.

        """
        return self

    def __sub__(self, other: object) -> LaplacianJet:
        """Return the jet of a difference.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if isinstance(other, LaplacianJet):
            return self + (-other)
        if _is_scalar(other):
            return self._shift(-other)  # type: ignore[operator]
        return NotImplemented

    def __rsub__(self, other: object) -> LaplacianJet:
        """Return the jet of ``constant - self``.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if _is_scalar(other):
            return (-self)._shift(other)
        return NotImplemented

    def __mul__(self, other: object) -> LaplacianJet:
        """Return the jet of a product (product rule).

        ``Laplacian(uv) = u Laplacian v + v Laplacian u + 2 grad u . grad v``.

        References:
            [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

        """
        if isinstance(other, LaplacianJet):
            u, v = self.value, other.value
            gradient = tuple(
                u * b + v * a
                for a, b in zip(self.gradient, other.gradient, strict=True)
            )
            if self.laplacian is None or other.laplacian is None:
                return LaplacianJet(u * v, gradient, None)
            cross: Any = 0.0
            for a, b in zip(self.gradient, other.gradient, strict=True):
                cross = cross + a * b
            return LaplacianJet(
                u * v,
                gradient,
                u * other.laplacian + v * self.laplacian + 2.0 * cross,
            )
        if _is_scalar(other):
            return self._scale(other)
        return NotImplemented

    def __rmul__(self, other: object) -> LaplacianJet:
        """Return the jet of ``constant * self``.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if _is_scalar(other):
            return self._scale(other)
        return NotImplemented

    def __truediv__(self, other: object) -> LaplacianJet:
        """Return the jet of ``self / constant``.

        Raises:
            AutodiffUnsupportedError: For a jet divisor.

        References:
            [EQ-GEN-EBM-005]. Linearity of the derivatives.

        """
        if _is_scalar(other):
            return self._scale(1.0 / other)  # type: ignore[operator]
        raise AutodiffUnsupportedError("a LaplacianJet divides by constants only")

    # -- comparisons (innermost value) ---------------------------------------

    def __lt__(self, other: object) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return primal(self.value) < _value(other)

    def __le__(self, other: object) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return primal(self.value) <= _value(other)

    def __gt__(self, other: object) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return primal(self.value) > _value(other)

    def __ge__(self, other: object) -> bool:
        """Compare innermost values.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return primal(self.value) >= _value(other)

    def __float__(self) -> float:
        """Refuse conversion, which would discard the derivatives.

        Raises:
            AutodiffUnsupportedError: Always.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        raise AutodiffUnsupportedError(
            "float() of a LaplacianJet would discard its derivatives"
        )

    # -- elementary functions ------------------------------------------------

    def _admath(self, name: str, arguments: tuple[object, ...]) -> Any:
        """Evaluate ``admath.<name>`` on the jet by the chain rule.

        Supported: ``tanh``, ``exp``, ``log``, ``log1p``.

        Raises:
            AutodiffUnsupportedError: For another function or extra arguments.

        References:
            [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

        """
        if len(arguments) != 1:
            raise AutodiffUnsupportedError(f"admath.{name} on a LaplacianJet")
        u = self.value
        if name == "tanh":
            t = admath.tanh(u)
            first = 1.0 - t * t
            return self._compose(t, first, -2.0 * t * first)
        if name == "exp":
            e = admath.exp(u)
            return self._compose(e, e, e)
        if name == "log":
            inverse = 1.0 / u
            return self._compose(admath.log(u), inverse, -inverse * inverse)
        if name == "log1p":
            inverse = 1.0 / (1.0 + u)
            return self._compose(admath.log1p(u), inverse, -inverse * inverse)
        raise AutodiffUnsupportedError(f"admath.{name} on a LaplacianJet")


def _value(other: object) -> float:
    """Return the innermost float of a comparison operand.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if isinstance(other, LaplacianJet):
        return primal(other.value)
    if isinstance(other, ReverseVariable | Dual | int | float):
        return primal(other)
    raise TypeError("cannot compare a LaplacianJet with this operand")
