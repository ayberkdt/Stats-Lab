"""Matrix-free Gauss-Newton and iteratively regularized Gauss-Newton inversion.

A differentiable forward map ``G: R^n -> R^m`` is declared through the
structural protocol :class:`DifferentiableForward` (``value``, ``jvp``,
``vjp``); the Jacobian ``J(x)`` is never formed. :class:`CallbackForward` wraps
user callbacks and makes the derivative mode an explicit choice: analytic
``jvp``/``vjp`` callbacks, or a central-difference ``jvp``. There is no
implicit fallback from one mode to the other, and a dense
finite-difference ``vjp`` (``2 n`` forward evaluations) needs
``allow_dense_fd_vjp=True``. :func:`check_derivatives` tests a model with the
directional Taylor remainder and the adjoint dot test.

With the whitened residual ``e = W (G(x) - y)`` (``W' W = Gamma^-1``) the
penalized objective is
``Phi(x) = 1/2 ||W (G(x) - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2``
[EQ-INV-GN-003]. :func:`gauss_newton_krylov` minimizes it with *fixed*
``lambda``: each step solves the linearized stacked least-squares problem
``min_p ||[W J; lambda L] p + [W r; lambda L (x - x_ref)]||`` by LSQR or CGLS
[EQ-INV-GN-004] and a backtracking Armijo line search on the full ``Phi``
[EQ-INV-GN-005]. :func:`iteratively_regularized_gauss_newton` (IRGNM) uses the
same linearized step but *lowers* ``lambda_k = q lambda_{k-1}`` every
iteration, takes full steps and stops by Morozov's discrepancy principle
[EQ-INV-GN-007]; the regularization is the iteration, not a fixed prior
weight. Fixed-``lambda`` Gauss-Newton converges to a stationary point of
``Phi`` for that ``lambda``; IRGNM is a regularization method whose stopping
index plays the role of the parameter.

Assumptions:
    ``G`` is continuously differentiable near the iterates and the supplied
    ``jvp``/``vjp`` are exact adjoints of each other (run
    :func:`check_derivatives`). A forward value that is not finite at a *trial*
    point of the line search is a rejected trial, never clipped; a non-finite
    forward value at the initial point, or non-finite derivative output at an
    accepted point, is reported (``ValueError`` and ``"nonfinite"``).

Common pitfalls:
    - ``reference`` defaults to the zero vector here, unlike
      :func:`probstat.inverse.nonlinear_tikhonov`, which centres the penalty
      at the start ``x0``.
    - Gauss-Newton drops the residual-weighted second-derivative term; it is
      not Newton's method and can converge slowly for large residuals.
    - ``converged`` is ``True`` only for the stop reasons ``"gradient"``,
      ``"step"`` and ``"discrepancy"``; ``"max_iterations"``,
      ``"line_search_failure"`` and ``"nonfinite"`` are failures.
    - A small whitened misfit does not imply a small parameter error for an
      ill-posed ``G``; the discrepancy level is the caller's responsibility.

Experimental (added in 0.28).

References:
    [REF-NOCEDAL-WRIGHT-2006], [REF-ENGL-HANKE-NEUBAUER-1996],
    [REF-TIKHONOV-1963], [REF-MOROZOV-1966], [REF-PAIGE-SAUNDERS-1982].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from probstat.linalg import (
    CallbackOperator,
    IdentityOperator,
    KrylovResult,
    LinearOperator,
    Vector,
    cgls,
    is_covariance_factor,
    lsqr,
    require_operator,
)

_EPS = 2.0**-52
_DEFAULT_FD_STEP: float = float(_EPS ** (1.0 / 3.0))
_STOP_CONVERGED = frozenset({"gradient", "step", "discrepancy"})


class _NonFiniteError(Exception):
    """Internal signal: a derivative action returned non-finite values."""


def _norm(values: Sequence[float]) -> float:
    """Return the Euclidean norm with a correctly rounded sum of squares.

    Args:
        values: Real sequence.

    Returns:
        ``sqrt(sum v_i^2)``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.sqrt(math.fsum(v * v for v in values))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the correctly rounded inner product.

    Args:
        left: First vector.
        right: Second vector of the same length.

    Returns:
        ``sum_i left_i right_i``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _is_finite(values: Sequence[float]) -> bool:
    """Return whether every entry is finite.

    Args:
        values: Real sequence.

    Returns:
        ``True`` when all entries are finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    return all(math.isfinite(v) for v in values)


def _as_vector(values: object, size: int, name: str, *, finite: bool = True) -> Vector:
    """Validate a real vector of a given length.

    Args:
        values: Candidate sequence of reals.
        size: Required length.
        name: Name used in messages.
        finite: Whether non-finite entries are rejected.

    Returns:
        The entries as a tuple of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length is wrong or an entry is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        out.append(float(entry))
    if len(out) != size:
        raise ValueError(f"{name} must have length {size}, got {len(out)}")
    if finite and not _is_finite(out):
        raise ValueError(f"{name} must be finite")
    return tuple(out)


def _positive_int(value: object, name: str) -> int:
    """Validate a strictly positive integer.

    Args:
        value: Candidate.
        name: Name used in messages.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 1``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _real(
    value: object, name: str, *, minimum: float = 0.0, strict: bool = False
) -> float:
    """Validate a finite real bounded below.

    Args:
        value: Candidate.
        name: Name used in messages.
        minimum: Lower bound.
        strict: Whether the bound is strict.

    Returns:
        The float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If non-finite or below the bound.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class ModelCounts:
    """Evaluation counts of a differentiable forward model.

    Attributes:
        value: Calls of the forward map ``G``.
        jvp: Applications of ``J(x) v``.
        vjp: Applications of ``J(x)' w``.
        derivative_value: Forward evaluations spent *inside* central-difference
            derivatives (``2`` per JVP, ``2 n`` per dense VJP); ``0`` for
            analytic derivatives. Total forward cost is
            ``value + derivative_value``.

    References:
        [EQ-INV-GN-002].

    """

    value: int
    jvp: int
    vjp: int
    derivative_value: int


@runtime_checkable
class DifferentiableForward(Protocol):
    """Structural protocol of a differentiable forward map ``G: R^n -> R^m``.

    Implementations provide the dimensions, the value ``G(x)``, the Jacobian
    action ``J(x) v`` (``jvp``) and the adjoint action ``J(x)' w`` (``vjp``,
    Euclidean transpose) without forming ``J``. The adjoint identity
    ``w'(J v) = (J'w)'v`` must hold; verify it with :func:`check_derivatives`.
    ``value`` may return non-finite entries (a failed forward evaluation); the
    solvers treat that as a rejected trial.

    References:
        [EQ-INV-GN-001], [REF-NOCEDAL-WRIGHT-2006].

    """

    @property
    def input_dimension(self) -> int:
        """Return the parameter dimension ``n``.

        References:
            [EQ-INV-GN-001].

        """
        ...

    @property
    def output_dimension(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-GN-001].

        """
        ...

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``G(x)``.

        References:
            [EQ-INV-GN-001].

        """
        ...

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(x) v``.

        References:
            [EQ-INV-GN-001].

        """
        ...

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``J(x)' w``.

        References:
            [EQ-INV-GN-001].

        """
        ...


class CallbackForward:
    """Differentiable forward map defined by user callbacks.

    The derivative mode is chosen explicitly and never inferred:

    * ``derivative="analytic"``: supply EITHER both ``jvp(x, v)`` and
      ``vjp(x, w)`` OR ``jacobian(x)`` (a dense ``m x n`` matrix, from which
      both actions are derived). Missing pieces raise ``ValueError``; there is
      no finite-difference fallback.
    * ``derivative="central_difference"``: ``J(x) v`` is the central
      difference ``(G(x + e v) - G(x - e v)) / (2 e)`` with the scale-aware
      size ``e = step (1 + ||x||_2) / ||v||_2`` (``2`` forward evaluations);
      ``jvp``, ``vjp`` and ``jacobian`` callbacks must NOT be given. The
      adjoint action is refused unless ``allow_dense_fd_vjp=True``, in which
      case it assembles the ``n`` Jacobian columns by central differences
      (``2 n`` forward evaluations per VJP; documented dense cost) and
      returns ``J'w`` [EQ-INV-GN-002].

    ``value`` is returned as given up to a length check, so a failed
    evaluation can return non-finite entries. All evaluations are counted in
    :attr:`counts`.

    References:
        [EQ-INV-GN-001], [EQ-INV-GN-002].

    """

    __slots__ = (
        "_allow_dense_fd_vjp",
        "_counts",
        "_derivative",
        "_jacobian",
        "_jvp",
        "_m",
        "_n",
        "_step",
        "_value",
        "_vjp",
    )

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        value: Callable[[Vector], Sequence[float]],
        jvp: Callable[[Vector, Vector], Sequence[float]] | None = None,
        vjp: Callable[[Vector, Vector], Sequence[float]] | None = None,
        jacobian: Callable[[Vector], Sequence[Sequence[float]]] | None = None,
        *,
        derivative: Literal["analytic", "central_difference"] = "analytic",
        step: float | None = None,
        allow_dense_fd_vjp: bool = False,
    ) -> None:
        """Store the callbacks and fix the derivative mode.

        Args:
            input_dimension: ``n >= 1``.
            output_dimension: ``m >= 1``.
            value: ``x -> G(x)`` (tuple input, sequence output of length ``m``).
            jvp: ``(x, v) -> J(x) v``; analytic mode only.
            vjp: ``(x, w) -> J(x)' w``; analytic mode only.
            jacobian: ``x -> J(x)`` as ``m`` rows of length ``n``; analytic
                mode only, exclusive with ``jvp``/``vjp``.
            derivative: ``"analytic"`` or ``"central_difference"``.
            step: Relative central-difference step (default
                ``eps^(1/3)`` ~ 6e-6); central-difference mode only.
            allow_dense_fd_vjp: Permit the dense ``2 n``-evaluation
                finite-difference VJP; central-difference mode only.

        Raises:
            TypeError: If a callback is not callable or a dimension is not an
                integer.
            ValueError: If the callbacks do not match the derivative mode or
                an argument is out of range.

        References:
            [EQ-INV-GN-001], [EQ-INV-GN-002].

        """
        self._n = _positive_int(input_dimension, "input_dimension")
        self._m = _positive_int(output_dimension, "output_dimension")
        if not callable(value):
            raise TypeError("value must be callable")
        for name, callback in (("jvp", jvp), ("vjp", vjp), ("jacobian", jacobian)):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        if derivative not in ("analytic", "central_difference"):
            raise ValueError("derivative must be 'analytic' or 'central_difference'")
        if derivative == "analytic":
            if step is not None or allow_dense_fd_vjp:
                raise ValueError(
                    "step and allow_dense_fd_vjp apply only to "
                    "derivative='central_difference'"
                )
            if jacobian is not None and (jvp is not None or vjp is not None):
                raise ValueError("give either jacobian or jvp/vjp, not both")
            if jacobian is None and (jvp is None or vjp is None):
                raise ValueError(
                    "analytic mode needs both jvp and vjp (or jacobian); there is "
                    "no finite-difference fallback, choose "
                    "derivative='central_difference' explicitly"
                )
            self._step = _DEFAULT_FD_STEP
        else:
            if jvp is not None or vjp is not None or jacobian is not None:
                raise ValueError(
                    "central_difference mode does not take jvp, vjp or jacobian "
                    "callbacks; use derivative='analytic' to supply them"
                )
            self._step = (
                _DEFAULT_FD_STEP if step is None else _real(step, "step", strict=True)
            )
        self._derivative = derivative
        self._allow_dense_fd_vjp = bool(allow_dense_fd_vjp)
        self._value = value
        self._jvp = jvp
        self._vjp = vjp
        self._jacobian = jacobian
        self._counts = [0, 0, 0, 0]

    @property
    def input_dimension(self) -> int:
        """Return ``n``.

        References:
            [EQ-INV-GN-001].

        """
        return self._n

    @property
    def output_dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-INV-GN-001].

        """
        return self._m

    @property
    def derivative(self) -> str:
        """Return the explicit derivative mode.

        References:
            [EQ-INV-GN-002].

        """
        return self._derivative

    @property
    def step(self) -> float:
        """Return the relative central-difference step (unused if analytic).

        References:
            [EQ-INV-GN-002].

        """
        return self._step

    @property
    def counts(self) -> ModelCounts:
        """Return a snapshot of the evaluation counts.

        References:
            [EQ-INV-GN-002].

        """
        value, jvp, vjp, inner = self._counts
        return ModelCounts(value=value, jvp=jvp, vjp=vjp, derivative_value=inner)

    def _raw_value(self, x: Vector) -> Vector:
        """Call the value callback with a length check (no finiteness check).

        References:
            [EQ-INV-GN-001].

        """
        return _as_vector(self._value(x), self._m, "value(x)", finite=False)

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``G(x)`` (entries may be non-finite for a failed evaluation).

        Args:
            x: Parameter vector of length ``n``.

        Returns:
            ``G(x)`` as a tuple of length ``m``.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or is not finite.

        References:
            [EQ-INV-GN-001].

        """
        point = _as_vector(x, self._n, "x")
        self._counts[0] += 1
        return self._raw_value(point)

    def _fd_directional(self, x: Vector, v: Vector) -> Vector:
        """Return the central difference of ``G`` along ``v`` (counted inside).

        References:
            [EQ-INV-GN-002].

        """
        vnorm = _norm(v)
        if vnorm == 0.0:
            return (0.0,) * self._m
        eps = self._step * (1.0 + _norm(x)) / vnorm
        plus = self._raw_value(tuple(a + eps * b for a, b in zip(x, v, strict=True)))
        minus = self._raw_value(tuple(a - eps * b for a, b in zip(x, v, strict=True)))
        self._counts[3] += 2
        return tuple((p - q) / (2.0 * eps) for p, q in zip(plus, minus, strict=True))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(x) v`` in the declared derivative mode.

        Args:
            x: Linearization point (length ``n``).
            v: Direction (length ``n``).

        Returns:
            ``J(x) v`` (length ``m``); entries may be non-finite if the
            underlying callback returns them.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong or an input is not finite.

        References:
            [EQ-INV-GN-001], [EQ-INV-GN-002].

        """
        point = _as_vector(x, self._n, "x")
        direction = _as_vector(v, self._n, "v")
        self._counts[1] += 1
        if self._derivative == "central_difference":
            return self._fd_directional(point, direction)
        if self._jacobian is not None:
            matrix = self._jacobian(point)
            return tuple(
                _dot(_as_vector(row, self._n, "jacobian row", finite=False), direction)
                for row in matrix
            )
        assert self._jvp is not None
        return _as_vector(self._jvp(point, direction), self._m, "jvp", finite=False)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``J(x)' w`` in the declared derivative mode.

        In central-difference mode this is refused unless the model was built
        with ``allow_dense_fd_vjp=True``; then it costs ``2 n`` forward
        evaluations (the full Jacobian by columns).

        Args:
            x: Linearization point (length ``n``).
            w: Adjoint direction (length ``m``).

        Returns:
            ``J(x)' w`` (length ``n``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite, or the
                dense finite-difference VJP is not allowed.

        References:
            [EQ-INV-GN-001], [EQ-INV-GN-002].

        """
        point = _as_vector(x, self._n, "x")
        weights = _as_vector(w, self._m, "w")
        if self._derivative == "central_difference":
            if not self._allow_dense_fd_vjp:
                raise ValueError(
                    "central-difference vjp forms the Jacobian column by column "
                    f"({2 * self._n} forward evaluations); construct the model "
                    "with allow_dense_fd_vjp=True to accept that cost"
                )
            self._counts[2] += 1
            result: list[float] = []
            for j in range(self._n):
                unit = tuple(1.0 if i == j else 0.0 for i in range(self._n))
                column = self._fd_directional(point, unit)
                result.append(_dot(column, weights))
            return tuple(result)
        self._counts[2] += 1
        if self._jacobian is not None:
            matrix = self._jacobian(point)
            rows = [
                _as_vector(row, self._n, "jacobian row", finite=False) for row in matrix
            ]
            return tuple(
                math.fsum(weights[i] * rows[i][j] for i in range(self._m))
                for j in range(self._n)
            )
        assert self._vjp is not None
        return _as_vector(self._vjp(point, weights), self._n, "vjp", finite=False)


@dataclass(frozen=True, slots=True)
class DerivativeCheck:
    """Result of :func:`check_derivatives`.

    Attributes:
        steps: Step sizes ``h`` swept (as given).
        remainders: ``||G(x + h v) - G(x) - h J(x) v||`` for each ``h``.
        orders: Observed order ``log(r_i / r_(i+1)) / log(h_i / h_(i+1))``
            between consecutive steps (``nan`` if a remainder is zero or
            below the roundoff floor); one entry fewer than ``steps``.
        noise_floor: Roundoff level ``64 eps (||G(x)|| + 1)`` below which a
            remainder carries no order information.
        taylor_passed: Heuristic: the remainder is below the noise floor at
            every step (linear map) or at least two consecutive orders lie in
            ``[1.7, 2.3]``. A wrong Jacobian gives order ~1; a non-smooth
            ``G`` near ``x`` shows no second-order range. A pass is evidence,
            not a proof.
        dot_lhs: ``w'(J v)``.
        dot_rhs: ``(J'w)'v``.
        dot_relative_error: ``|lhs - rhs| / (||w|| ||Jv|| + ||J'w|| ||v||)``
            (``0`` if the denominator is ``0``).
        adjoint_passed: ``dot_relative_error <= dot_tolerance``.
        passed: ``taylor_passed and adjoint_passed`` (heuristic).

    References:
        [EQ-INV-GN-001].

    """

    steps: tuple[float, ...]
    remainders: tuple[float, ...]
    orders: tuple[float, ...]
    noise_floor: float
    taylor_passed: bool
    dot_lhs: float
    dot_rhs: float
    dot_relative_error: float
    adjoint_passed: bool
    passed: bool


def check_derivatives(
    model: DifferentiableForward,
    x: Sequence[float],
    v: Sequence[float],
    w: Sequence[float],
    steps: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
    *,
    dot_tolerance: float = 1e-6,
) -> DerivativeCheck:
    """Check ``jvp`` against a Taylor sweep and ``vjp`` against ``jvp``.

    The directional Taylor remainder
    ``r(h) = ||G(x + h v) - G(x) - h J(x) v||`` is ``O(h^2)`` for a smooth
    ``G`` and an exact JVP, until roundoff (~``eps ||G||``) dominates; an
    observed order near 1 means the JVP is wrong. The adjoint dot test
    compares ``w'(J v)`` with ``(J'w)'v``. The check runs on the model's
    actual computation, so a finite-difference model is tested against its own
    forward map and an analytic model against the same ``G`` (the two are not
    compared with each other here). ``passed`` is a HEURISTIC summary; inspect
    ``remainders`` and ``orders`` near non-smooth points.

    Args:
        model: The :class:`DifferentiableForward` model.
        x: Base point (length ``n``).
        v: Direction (length ``n``, not zero).
        w: Adjoint direction (length ``m``).
        steps: Positive step sizes, ideally decreasing geometrically.
        dot_tolerance: Relative tolerance of the dot test (``> 0``).

    Returns:
        :class:`DerivativeCheck`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a length is wrong, ``v`` is zero, a step is not
            positive, fewer than two steps are given, or the model returns a
            non-finite value at a probe.

    References:
        [EQ-INV-GN-001], [REF-NOCEDAL-WRIGHT-2006].

    """
    if not isinstance(model, DifferentiableForward):
        raise TypeError("model must implement DifferentiableForward")
    n, m = model.input_dimension, model.output_dimension
    point = _as_vector(x, n, "x")
    direction = _as_vector(v, n, "v")
    adjoint = _as_vector(w, m, "w")
    if _norm(direction) == 0.0:
        raise ValueError("v must not be zero")
    hs = tuple(_real(h, "step", strict=True) for h in steps)
    if len(hs) < 2:
        raise ValueError("steps must contain at least two values")
    tol = _real(dot_tolerance, "dot_tolerance", strict=True)
    base = _as_vector(model.value(point), m, "value(x)")
    jv = _as_vector(model.jvp(point, direction), m, "jvp(x, v)")
    remainders: list[float] = []
    for h in hs:
        shifted = tuple(a + h * b for a, b in zip(point, direction, strict=True))
        trial = _as_vector(model.value(shifted), m, "value(x + h v)")
        remainders.append(
            _norm([t - b - h * j for t, b, j in zip(trial, base, jv, strict=True)])
        )
    floor = 64.0 * _EPS * (_norm(base) + 1.0)
    orders: list[float] = []
    for i in range(len(hs) - 1):
        r0, r1 = remainders[i], remainders[i + 1]
        if r0 > floor and r1 > floor and hs[i] != hs[i + 1]:
            orders.append(math.log(r0 / r1) / math.log(hs[i] / hs[i + 1]))
        else:
            orders.append(math.nan)
    second_order = [o for o in orders if 1.7 <= o <= 2.3]
    taylor = all(r <= floor for r in remainders) or len(second_order) >= 2
    jtw = _as_vector(model.vjp(point, adjoint), n, "vjp(x, w)")
    lhs = _dot(adjoint, jv)
    rhs = _dot(jtw, direction)
    scale = _norm(adjoint) * _norm(jv) + _norm(jtw) * _norm(direction)
    relative = 0.0 if scale == 0.0 else abs(lhs - rhs) / scale
    adjoint_ok = relative <= tol
    return DerivativeCheck(
        steps=hs,
        remainders=tuple(remainders),
        orders=tuple(orders),
        noise_floor=floor,
        taylor_passed=taylor,
        dot_lhs=lhs,
        dot_rhs=rhs,
        dot_relative_error=relative,
        adjoint_passed=adjoint_ok,
        passed=taylor and adjoint_ok,
    )


@dataclass(frozen=True, slots=True)
class GaussNewtonIteration:
    """One outer iteration of :func:`gauss_newton_krylov`.

    The state fields describe the iterate ``x_k`` at the start of the
    iteration; the step fields describe the step taken from it. The final
    record (the stopping iterate) has no step.

    Attributes:
        iteration: Index ``k`` (starting at ``0``).
        objective: ``Phi(x_k)``.
        misfit: ``1/2 ||W (G(x_k) - y)||^2``.
        penalty: ``lambda^2/2 ||L (x_k - x_ref)||^2``.
        residual_norm: Raw ``||G(x_k) - y||``.
        gradient_norm: ``||J'W'W r + lambda^2 L'L (x_k - x_ref)||``.
        step_length: Accepted line-search factor ``t`` (``0.0`` if no step).
        step_norm: ``||t p||``.
        rejected_objectives: ``Phi`` at each rejected trial of the line search
            in order (``nan`` for a trial whose forward value was not
            finite), so the history contains the rejected attempts.
        inner_iterations: Krylov iterations of the linearized solve.
        inner_stop_reason: Krylov stop reason (``"none"`` if no solve).

    References:
        [EQ-INV-GN-004], [EQ-INV-GN-005].

    """

    iteration: int
    objective: float
    misfit: float
    penalty: float
    residual_norm: float
    gradient_norm: float
    step_length: float
    step_norm: float
    rejected_objectives: tuple[float, ...]
    inner_iterations: int
    inner_stop_reason: str


@dataclass(frozen=True, slots=True)
class GaussNewtonResult:
    """Result of :func:`gauss_newton_krylov`.

    Attributes:
        estimate: Final iterate ``x``.
        objective: ``Phi(x)``.
        misfit: ``1/2 ||W (G(x) - y)||^2``.
        whitened_residual_norm: ``||W (G(x) - y)||`` (compared with the
            discrepancy level).
        residual_norm: Raw ``||G(x) - y||``.
        penalty: ``lambda^2/2 ||L (x - x_ref)||^2``.
        gradient_norm: Norm of the gradient of ``Phi`` at ``x``.
        iterations: Number of accepted steps.
        converged: ``True`` for ``"gradient"``, ``"step"``, ``"discrepancy"``.
        stop_reason: One of ``"gradient"``, ``"step"``, ``"discrepancy"``,
            ``"max_iterations"``, ``"line_search_failure"`` (no acceptable
            step, including a non-descent inner direction) or
            ``"nonfinite"`` (non-finite derivative action).
        history: One :class:`GaussNewtonIteration` per iterate, the stopping
            iterate last.
        counts: Value/JVP/VJP calls made by the solver;
            ``derivative_value`` is the growth of the model's own count when
            it exposes ``counts``.
        total_inner_iterations: Sum of Krylov iterations.
        regularization: ``lambda``.
        inner_solver: ``"lsqr"`` or ``"cgls"``.
        wall_time: Seconds (benchmark information, not deterministic).

    References:
        [EQ-INV-GN-003], [EQ-INV-GN-004], [EQ-INV-GN-006].

    """

    estimate: Vector
    objective: float
    misfit: float
    whitened_residual_norm: float
    residual_norm: float
    penalty: float
    gradient_norm: float
    iterations: int
    converged: bool
    stop_reason: str
    history: tuple[GaussNewtonIteration, ...]
    counts: ModelCounts
    total_inner_iterations: int
    regularization: float
    inner_solver: str
    wall_time: float


@dataclass(frozen=True, slots=True)
class IRGNMIteration:
    """One iteration of :func:`iteratively_regularized_gauss_newton`.

    Attributes:
        iteration: Index ``k`` (starting at ``0``).
        regularization: ``lambda_k`` used for the step taken from ``x_k``
            (the value that would be used next at the stopping iterate).
        whitened_residual_norm: ``||W (G(x_k) - y)||``.
        misfit: ``1/2 ||W (G(x_k) - y)||^2``.
        penalty: ``lambda_k^2/2 ||L (x_k - x_ref)||^2``.
        step_norm: ``||p_k||`` (``0.0`` for the stopping iterate).
        inner_iterations: Krylov iterations of the step.
        inner_stop_reason: Krylov stop reason (``"none"`` if no solve).

    References:
        [EQ-INV-GN-007].

    """

    iteration: int
    regularization: float
    whitened_residual_norm: float
    misfit: float
    penalty: float
    step_norm: float
    inner_iterations: int
    inner_stop_reason: str


@dataclass(frozen=True, slots=True)
class IRGNMResult:
    """Result of :func:`iteratively_regularized_gauss_newton`.

    Attributes:
        estimate: Final iterate (the first with misfit at most the
            discrepancy level when ``converged``).
        whitened_residual_norm: ``||W (G(x) - y)||``.
        residual_norm: Raw ``||G(x) - y||``.
        discrepancy: The level ``tau delta`` requested.
        iterations: Number of Gauss-Newton steps taken.
        converged: ``True`` only for ``"discrepancy"``.
        stop_reason: ``"discrepancy"``, ``"max_iterations"``,
            ``"regularization_floor"`` or ``"nonfinite"``.
        regularization: ``lambda_k`` that would be used next (the last used
            if no further step).
        history: One :class:`IRGNMIteration` per iterate.
        counts: Value/JVP/VJP calls made by the solver.
        total_inner_iterations: Sum of Krylov iterations.
        inner_solver: ``"lsqr"`` or ``"cgls"``.
        wall_time: Seconds (benchmark information).

    References:
        [EQ-INV-GN-007].

    """

    estimate: Vector
    whitened_residual_norm: float
    residual_norm: float
    discrepancy: float
    iterations: int
    converged: bool
    stop_reason: str
    regularization: float
    history: tuple[IRGNMIteration, ...]
    counts: ModelCounts
    total_inner_iterations: int
    inner_solver: str
    wall_time: float


class _Problem:
    """Validated problem data and counted evaluations shared by both solvers.

    References:
        [EQ-INV-GN-003].

    """

    def __init__(
        self,
        model: DifferentiableForward,
        data: Sequence[float],
        noise: Any,
        penalty: LinearOperator | None,
        reference: Sequence[float] | None,
        initial: Sequence[float] | None,
    ) -> None:
        """Validate and store the problem.

        Args:
            model: The forward model.
            data: Observations ``y``.
            noise: Covariance factor or ``None``.
            penalty: ``L`` or ``None``.
            reference: ``x_ref`` or ``None``.
            initial: Start or ``None``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a dimension is inconsistent.

        References:
            [EQ-INV-GN-003].

        """
        if not isinstance(model, DifferentiableForward):
            raise TypeError("model must implement DifferentiableForward")
        self.model = model
        self.n = _positive_int(model.input_dimension, "input_dimension")
        self.m = _positive_int(model.output_dimension, "output_dimension")
        self.data = _as_vector(data, self.m, "data")
        if noise is not None:
            if not is_covariance_factor(noise):
                raise TypeError("noise must be a probstat.linalg covariance factor")
            if noise.dimension != self.m:
                raise ValueError(
                    f"noise has dimension {noise.dimension}, expected {self.m}"
                )
        self.noise = noise
        if penalty is None:
            self.penalty: LinearOperator = IdentityOperator(self.n)
        else:
            self.penalty = require_operator(penalty, name="penalty")
            if self.penalty.shape[1] != self.n:
                raise ValueError(
                    f"penalty has {self.penalty.shape[1]} columns, expected {self.n}"
                )
        self.reference = (
            (0.0,) * self.n
            if reference is None
            else _as_vector(reference, self.n, "reference")
        )
        self.initial = (
            (0.0,) * self.n
            if initial is None
            else _as_vector(initial, self.n, "initial")
        )
        self.calls = [0, 0, 0]
        self._start = getattr(model, "counts", None)

    def whiten(self, vector: Vector) -> Vector:
        """Return ``W v`` (identity if no noise model).

        References:
            [EQ-INV-GN-003].

        """
        return vector if self.noise is None else tuple(self.noise.whiten(vector))

    def whiten_transpose(self, vector: Vector) -> Vector:
        """Return ``W' v`` (identity if no noise model).

        References:
            [EQ-INV-GN-003].

        """
        if self.noise is None:
            return vector
        return tuple(self.noise.whiten_transpose(vector))

    def residual(self, x: Vector) -> tuple[Vector, bool]:
        """Return ``(G(x) - y, finite)``.

        References:
            [EQ-INV-GN-003].

        """
        self.calls[0] += 1
        value = _as_vector(self.model.value(x), self.m, "value(x)", finite=False)
        r = tuple(a - b for a, b in zip(value, self.data, strict=True))
        return r, _is_finite(r)

    def offset(self, x: Vector) -> Vector:
        """Return ``L (x - x_ref)``.

        References:
            [EQ-INV-GN-003].

        """
        diff = tuple(a - b for a, b in zip(x, self.reference, strict=True))
        return tuple(self.penalty.matvec(diff))

    def jvp(self, x: Vector, v: Vector) -> Vector:
        """Return a counted, finite ``J(x) v``.

        Raises:
            _NonFiniteError: If the model returns non-finite entries.

        References:
            [EQ-INV-GN-004].

        """
        self.calls[1] += 1
        out = _as_vector(self.model.jvp(x, v), self.m, "jvp", finite=False)
        if not _is_finite(out):
            raise _NonFiniteError
        return out

    def vjp(self, x: Vector, w: Vector) -> Vector:
        """Return a counted, finite ``J(x)' w``.

        Raises:
            _NonFiniteError: If the model returns non-finite entries.

        References:
            [EQ-INV-GN-004].

        """
        self.calls[2] += 1
        out = _as_vector(self.model.vjp(x, w), self.n, "vjp", finite=False)
        if not _is_finite(out):
            raise _NonFiniteError
        return out

    def gradient(self, x: Vector, r: Vector, offset: Vector, lam: float) -> Vector:
        """Return ``J'W'W r + lambda^2 L'L (x - x_ref)`` (one VJP).

        References:
            [EQ-INV-GN-003].

        """
        data_part = self.vjp(x, self.whiten_transpose(self.whiten(r)))
        prior_part = self.penalty.rmatvec(offset)
        lam2 = lam * lam
        return tuple(a + lam2 * b for a, b in zip(data_part, prior_part, strict=True))

    def linear_step(
        self,
        x: Vector,
        r: Vector,
        offset: Vector,
        lam: float,
        solver: str,
        tolerance: float,
        max_iterations: int | None,
        condition_limit: float,
    ) -> KrylovResult:
        """Solve the stacked linearized least-squares problem for the step.

        Minimizes ``||[W J; lam L] p + [W r; lam L (x - x_ref)]||`` over ``p``
        by LSQR or CGLS on a callback operator built from ``jvp`` and ``vjp``;
        the Jacobian is never formed [EQ-INV-GN-004].

        References:
            [EQ-INV-GN-004], [EQ-INV-GN-008].

        """
        m, k = self.m, self.penalty.shape[0]
        use_penalty = lam > 0.0

        def forward(p: Vector) -> Sequence[float]:
            """Apply the stacked operator to ``p``.

            References:
                [EQ-INV-GN-004].

            """
            top = self.whiten(self.jvp(x, p))
            if not use_penalty:
                return top
            low = self.penalty.matvec(p)
            return (*top, *(lam * value for value in low))

        def adjoint(u: Vector) -> Sequence[float]:
            """Apply the transpose of the stacked operator to ``u``.

            References:
                [EQ-INV-GN-004].

            """
            head = self.vjp(x, self.whiten_transpose(tuple(u[:m])))
            if not use_penalty:
                return head
            tail = self.penalty.rmatvec(tuple(u[m:]))
            return tuple(a + lam * b for a, b in zip(head, tail, strict=True))

        rows = m + (k if use_penalty else 0)
        operator = CallbackOperator((rows, self.n), forward, adjoint, name="gn_step")
        wr = self.whiten(r)
        rhs = (
            tuple(-a for a in wr)
            if not use_penalty
            else (*(-a for a in wr), *(-lam * a for a in offset))
        )
        if solver == "lsqr":
            return lsqr(
                operator,
                rhs,
                atol=tolerance,
                btol=tolerance,
                condition_limit=condition_limit,
                max_iterations=max_iterations,
            )
        return cgls(
            operator,
            rhs,
            residual_tolerance=0.0,
            normal_tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def counts(self) -> ModelCounts:
        """Return the solver's evaluation counts.

        References:
            [EQ-INV-GN-002].

        """
        inner = 0
        now = getattr(self.model, "counts", None)
        if isinstance(now, ModelCounts) and isinstance(self._start, ModelCounts):
            inner = now.derivative_value - self._start.derivative_value
        return ModelCounts(
            value=self.calls[0],
            jvp=self.calls[1],
            vjp=self.calls[2],
            derivative_value=inner,
        )


def _check_solver(solver: str) -> str:
    """Validate the inner solver name.

    Args:
        solver: Candidate name.

    Returns:
        The name.

    Raises:
        ValueError: If not ``"lsqr"`` or ``"cgls"``.

    References:
        [EQ-INV-GN-004].

    """
    if solver not in ("lsqr", "cgls"):
        raise ValueError("inner_solver must be 'lsqr' or 'cgls'")
    return solver


def _optional_iterations(value: int | None) -> int | None:
    """Validate an optional iteration cap.

    Args:
        value: ``None`` or a positive integer.

    Returns:
        The value.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    return None if value is None else _positive_int(value, "inner_max_iterations")


def gauss_newton_krylov(
    model: DifferentiableForward,
    data: Sequence[float],
    *,
    noise: Any = None,
    regularization: float = 0.0,
    penalty: LinearOperator | None = None,
    reference: Sequence[float] | None = None,
    initial: Sequence[float] | None = None,
    max_iterations: int = 50,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    discrepancy: float | None = None,
    inner_solver: Literal["lsqr", "cgls"] = "lsqr",
    inner_tolerance: float = 1e-8,
    forcing: Literal["constant", "adaptive"] = "constant",
    inner_max_iterations: int | None = None,
    inner_condition_limit: float = 1e12,
    armijo: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 30,
) -> GaussNewtonResult:
    """Minimize ``Phi`` by matrix-free Gauss-Newton with a fixed ``lambda``.

    ``Phi(x) = 1/2 ||W (G(x) - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2``
    [EQ-INV-GN-003] (``W' W = Gamma^-1`` from ``noise``; ``L = I`` and
    ``x_ref = 0`` by default). Each iteration solves the linearized stacked
    least-squares problem
    ``min_p ||[W J; lambda L] p + [W r; lambda L (x - x_ref)]||`` with LSQR or
    CGLS on operators built from ``jvp``/``vjp`` [EQ-INV-GN-004] (never forming
    ``J`` or ``J'J``), then backtracks ``t = 1, f, f^2, ...`` until the Armijo
    condition ``Phi(x + t p) <= Phi(x) + c t g'p`` holds
    [EQ-INV-GN-005], with a roundoff slack ``4 eps |Phi|``. A trial whose
    forward value is not finite is a rejected trial (recorded as ``nan`` in
    ``rejected_objectives``); it is never clipped.

    Stopping, in order each iteration: whitened misfit ``<= discrepancy``
    (``"discrepancy"``, only if a level is given [EQ-INV-GN-006]);
    ``||g|| <= gradient_tolerance max(1, ||g_0||)`` (``"gradient"``); the
    iteration cap (``"max_iterations"``); no acceptable step
    (``"line_search_failure"``); accepted step
    ``||t p|| <= step_tolerance (1 + ||x||)`` (``"step"``); non-finite
    derivative output (``"nonfinite"``). The inner tolerance is the LSQR
    ``atol = btol`` or the CGLS relative normal-residual tolerance; with
    ``forcing="adaptive"`` it is loosened to
    ``min(1/2, max(inner_tolerance, sqrt(||g_k|| / ||g_0||)))``
    [EQ-INV-GN-008]. The rank-deficient or ill-posed case is handled by
    ``lambda > 0`` (the stacked problem then has full column rank when ``L``
    does); with ``lambda = 0`` truncation by the inner iteration is the only
    regularization and is reported, not hidden.

    Args:
        model: Differentiable forward map.
        data: Observations ``y`` (length ``m``).
        noise: Covariance factor of ``probstat.linalg`` (dimension ``m``) or
            ``None`` for unit noise.
        regularization: ``lambda >= 0``.
        penalty: ``L`` as a linear operator with ``n`` columns (default
            identity).
        reference: ``x_ref`` (default zeros; NOT the start).
        initial: Start ``x_0`` (default zeros).
        max_iterations: Outer iteration cap (``>= 1``).
        gradient_tolerance: Relative gradient tolerance (``>= 0``).
        step_tolerance: Relative step tolerance (``>= 0``).
        discrepancy: Optional ``tau delta``.
        inner_solver: ``"lsqr"`` or ``"cgls"``.
        inner_tolerance: Inner tolerance ``eta`` (``> 0``).
        forcing: ``"constant"`` or ``"adaptive"`` inner tolerance.
        inner_max_iterations: Krylov iteration cap per step (default ``2 n``).
        inner_condition_limit: LSQR condition-estimate stop (``> 1``).
        armijo: Sufficient-decrease constant ``c`` in ``(0, 1)``.
        backtracking_factor: ``f`` in ``(0, 1)``.
        max_backtracks: Maximum trial steps per iteration (``>= 1``).

    Returns:
        :class:`GaussNewtonResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension is inconsistent, an option is out of range
            or the forward map is not finite at ``initial``.

    References:
        [EQ-INV-GN-003], [EQ-INV-GN-004], [EQ-INV-GN-005], [EQ-INV-GN-006],
        [EQ-INV-GN-008], [REF-NOCEDAL-WRIGHT-2006].

    """
    started = time.perf_counter()
    problem = _Problem(model, data, noise, penalty, reference, initial)
    lam = _real(regularization, "regularization")
    cap = _positive_int(max_iterations, "max_iterations")
    gtol = _real(gradient_tolerance, "gradient_tolerance")
    stol = _real(step_tolerance, "step_tolerance")
    level = (
        None if discrepancy is None else _real(discrepancy, "discrepancy", strict=True)
    )
    solver = _check_solver(inner_solver)
    eta0 = _real(inner_tolerance, "inner_tolerance", strict=True)
    if forcing not in ("constant", "adaptive"):
        raise ValueError("forcing must be 'constant' or 'adaptive'")
    inner_cap = _optional_iterations(inner_max_iterations)
    conlim = _real(
        inner_condition_limit, "inner_condition_limit", minimum=1.0, strict=True
    )
    c1 = _real(armijo, "armijo", strict=True)
    factor = _real(backtracking_factor, "backtracking_factor", strict=True)
    if c1 >= 1.0 or factor >= 1.0:
        raise ValueError("armijo and backtracking_factor must lie in (0, 1)")
    tries = _positive_int(max_backtracks, "max_backtracks")

    x = problem.initial
    r, finite = problem.residual(x)
    if not finite:
        raise ValueError("forward map is not finite at the initial point")
    history: list[GaussNewtonIteration] = []
    inner_total = 0
    reason = "max_iterations"
    accepted = 0
    g0 = math.nan
    state: tuple[float, float, float, float, float] = (math.nan,) * 5

    def phi_parts(res: Vector, off: Vector) -> tuple[float, float, float]:
        """Return ``(Phi, misfit, penalty)`` for a residual and offset.

        References:
            [EQ-INV-GN-003].

        """
        wr = _norm(problem.whiten(res))
        mis = 0.5 * wr * wr
        pen = 0.5 * lam * lam * _dot(off, off)
        return mis + pen, mis, pen

    for k in range(cap + 1):
        off = problem.offset(x)
        objective, misfit, pen = phi_parts(r, off)
        wnorm = _norm(problem.whiten(r))
        try:
            grad = problem.gradient(x, r, off, lam)
        except _NonFiniteError:
            history.append(
                GaussNewtonIteration(
                    k,
                    objective,
                    misfit,
                    pen,
                    _norm(r),
                    math.nan,
                    0.0,
                    0.0,
                    (),
                    0,
                    "none",
                )
            )
            reason = "nonfinite"
            state = (objective, misfit, wnorm, pen, math.nan)
            break
        gnorm = _norm(grad)
        if k == 0:
            g0 = gnorm
        state = (objective, misfit, wnorm, pen, gnorm)
        stop: str | None = None
        if level is not None and wnorm <= level:
            stop = "discrepancy"
        elif gnorm <= gtol * max(1.0, g0):
            stop = "gradient"
        elif k == cap:
            stop = "max_iterations"
        if stop is not None:
            history.append(
                GaussNewtonIteration(
                    k,
                    objective,
                    misfit,
                    pen,
                    _norm(r),
                    gnorm,
                    0.0,
                    0.0,
                    (),
                    0,
                    "none",
                )
            )
            reason = stop
            break
        eta = eta0
        if forcing == "adaptive" and g0 > 0.0:
            eta = min(0.5, max(eta0, math.sqrt(gnorm / g0)))
        try:
            inner = problem.linear_step(x, r, off, lam, solver, eta, inner_cap, conlim)
        except _NonFiniteError:
            history.append(
                GaussNewtonIteration(
                    k,
                    objective,
                    misfit,
                    pen,
                    _norm(r),
                    gnorm,
                    0.0,
                    0.0,
                    (),
                    0,
                    "none",
                )
            )
            reason = "nonfinite"
            break
        inner_total += inner.iterations
        p = inner.solution
        slope = _dot(grad, p)
        rejected: list[float] = []
        t = 1.0
        taken: tuple[Vector, Vector, float] | None = None
        if math.isfinite(slope) and slope < 0.0 and _is_finite(p):
            slack = 4.0 * _EPS * abs(objective)
            for _ in range(tries):
                trial = tuple(a + t * b for a, b in zip(x, p, strict=True))
                r_trial, ok = problem.residual(trial)
                if not ok:
                    rejected.append(math.nan)
                else:
                    off_trial = problem.offset(trial)
                    phi_trial = phi_parts(r_trial, off_trial)[0]
                    if phi_trial <= objective + c1 * t * slope + slack:
                        taken = (trial, r_trial, t)
                        break
                    rejected.append(phi_trial)
                t *= factor
        if taken is None:
            history.append(
                GaussNewtonIteration(
                    k,
                    objective,
                    misfit,
                    pen,
                    _norm(r),
                    gnorm,
                    0.0,
                    0.0,
                    tuple(rejected),
                    inner.iterations,
                    inner.stop_reason,
                )
            )
            reason = "line_search_failure"
            break
        trial, r_trial, t_taken = taken
        step_norm = t_taken * _norm(p)
        history.append(
            GaussNewtonIteration(
                k,
                objective,
                misfit,
                pen,
                _norm(r),
                gnorm,
                t_taken,
                step_norm,
                tuple(rejected),
                inner.iterations,
                inner.stop_reason,
            )
        )
        x, r = trial, r_trial
        accepted += 1
        if step_norm <= stol * (1.0 + _norm(x)):
            off = problem.offset(x)
            objective, misfit, pen = phi_parts(r, off)
            wnorm = _norm(problem.whiten(r))
            try:
                gnorm = _norm(problem.gradient(x, r, off, lam))
            except _NonFiniteError:
                gnorm = math.nan
            state = (objective, misfit, wnorm, pen, gnorm)
            history.append(
                GaussNewtonIteration(
                    k + 1,
                    objective,
                    misfit,
                    pen,
                    _norm(r),
                    gnorm,
                    0.0,
                    0.0,
                    (),
                    0,
                    "none",
                )
            )
            reason = "step"
            break
    objective, misfit, wnorm, pen, gnorm = state
    return GaussNewtonResult(
        estimate=x,
        objective=objective,
        misfit=misfit,
        whitened_residual_norm=wnorm,
        residual_norm=_norm(r),
        penalty=pen,
        gradient_norm=gnorm,
        iterations=accepted,
        converged=reason in _STOP_CONVERGED,
        stop_reason=reason,
        history=tuple(history),
        counts=problem.counts(),
        total_inner_iterations=inner_total,
        regularization=lam,
        inner_solver=solver,
        wall_time=time.perf_counter() - started,
    )


def iteratively_regularized_gauss_newton(
    model: DifferentiableForward,
    data: Sequence[float],
    *,
    noise: Any = None,
    initial_regularization: float,
    decrease: float = 0.5,
    penalty: LinearOperator | None = None,
    reference: Sequence[float] | None = None,
    initial: Sequence[float] | None = None,
    discrepancy: float,
    max_iterations: int = 50,
    minimum_regularization: float | None = None,
    inner_solver: Literal["lsqr", "cgls"] = "lsqr",
    inner_tolerance: float = 1e-8,
    inner_max_iterations: int | None = None,
    inner_condition_limit: float = 1e12,
) -> IRGNMResult:
    """Run iteratively regularized Gauss-Newton (IRGNM) with discrepancy stopping.

    Starting from ``x_0`` and ``lambda_0``, each iteration takes the FULL step
    ``x_(k+1) = x_k + p_k`` with
    ``p_k = argmin_p ||W (G(x_k) + J_k p - y)||^2
    + lambda_k^2 ||L (x_k + p - x_ref)||^2``
    (the same stacked linearized problem as :func:`gauss_newton_krylov`,
    solved by LSQR/CGLS from ``jvp``/``vjp``), then lowers
    ``lambda_(k+1) = q lambda_k`` with ``q`` in ``(0, 1)`` [EQ-INV-GN-007].
    The iteration stops at the first ``k`` with
    ``||W (G(x_k) - y)|| <= discrepancy`` (Morozov's discrepancy principle,
    ``discrepancy = tau delta`` with ``tau > 1`` and noise level ``delta``
    in the whitened norm) [EQ-INV-GN-006]. In contrast to fixed-``lambda``
    Gauss-Newton there is no line search and no stationary objective: the
    penalty is always centred at ``x_ref`` (not at ``x_k``), the regularization
    is ``lambda_k -> 0`` and the stopping index is the regularization
    parameter; the result depends on ``x_ref`` and ``q``, and iterating past
    the discrepancy index amplifies noise. Convergence theory needs ``G`` to
    satisfy a (tangential cone) condition and ``q`` not too small
    [REF-ENGL-HANKE-NEUBAUER-1996]; this function does not verify it.

    Args:
        model: Differentiable forward map.
        data: Observations ``y``.
        noise: Covariance factor (dimension ``m``) or ``None``.
        initial_regularization: ``lambda_0 > 0``.
        decrease: ``q`` in ``(0, 1)``.
        penalty: ``L`` (default identity).
        reference: ``x_ref`` (default zeros).
        initial: ``x_0`` (default zeros).
        discrepancy: ``tau delta > 0`` (required).
        max_iterations: Cap on Gauss-Newton steps (``>= 1``).
        minimum_regularization: Optional floor; stop with
            ``"regularization_floor"`` when ``lambda_k`` falls below it.
        inner_solver: ``"lsqr"`` or ``"cgls"``.
        inner_tolerance: Inner tolerance (``> 0``).
        inner_max_iterations: Krylov cap per step (default ``2 n``).
        inner_condition_limit: LSQR condition-estimate stop (``> 1``).

    Returns:
        :class:`IRGNMResult`; ``converged`` is ``True`` only for
        ``"discrepancy"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension is inconsistent, an option is out of range
            or the forward map is not finite at ``initial``.

    References:
        [EQ-INV-GN-004], [EQ-INV-GN-006], [EQ-INV-GN-007],
        [REF-ENGL-HANKE-NEUBAUER-1996].

    """
    started = time.perf_counter()
    problem = _Problem(model, data, noise, penalty, reference, initial)
    lam = _real(initial_regularization, "initial_regularization", strict=True)
    q = _real(decrease, "decrease", strict=True)
    if q >= 1.0:
        raise ValueError("decrease must lie in (0, 1)")
    level = _real(discrepancy, "discrepancy", strict=True)
    cap = _positive_int(max_iterations, "max_iterations")
    floor = (
        None
        if minimum_regularization is None
        else _real(minimum_regularization, "minimum_regularization", strict=True)
    )
    solver = _check_solver(inner_solver)
    eta = _real(inner_tolerance, "inner_tolerance", strict=True)
    inner_cap = _optional_iterations(inner_max_iterations)
    conlim = _real(
        inner_condition_limit, "inner_condition_limit", minimum=1.0, strict=True
    )

    x = problem.initial
    r, finite = problem.residual(x)
    if not finite:
        raise ValueError("forward map is not finite at the initial point")
    history: list[IRGNMIteration] = []
    inner_total = 0
    steps = 0
    reason = "max_iterations"
    for k in range(cap + 1):
        off = problem.offset(x)
        wnorm = _norm(problem.whiten(r))
        pen = 0.5 * lam * lam * _dot(off, off)
        if wnorm <= level:
            reason = "discrepancy"
        elif k == cap:
            reason = "max_iterations"
        elif floor is not None and lam < floor:
            reason = "regularization_floor"
        else:
            reason = ""
        if reason:
            history.append(
                IRGNMIteration(k, lam, wnorm, 0.5 * wnorm * wnorm, pen, 0.0, 0, "none")
            )
            break
        try:
            inner = problem.linear_step(x, r, off, lam, solver, eta, inner_cap, conlim)
        except _NonFiniteError:
            history.append(
                IRGNMIteration(k, lam, wnorm, 0.5 * wnorm * wnorm, pen, 0.0, 0, "none")
            )
            reason = "nonfinite"
            break
        inner_total += inner.iterations
        p = inner.solution
        new_x = tuple(a + b for a, b in zip(x, p, strict=True))
        history.append(
            IRGNMIteration(
                k,
                lam,
                wnorm,
                0.5 * wnorm * wnorm,
                pen,
                _norm(p),
                inner.iterations,
                inner.stop_reason,
            )
        )
        new_r, ok = problem.residual(new_x)
        if not ok or not _is_finite(new_x):
            reason = "nonfinite"
            break
        x, r = new_x, new_r
        steps += 1
        lam *= q
    return IRGNMResult(
        estimate=x,
        whitened_residual_norm=_norm(problem.whiten(r)),
        residual_norm=_norm(r),
        discrepancy=level,
        iterations=steps,
        converged=reason == "discrepancy",
        stop_reason=reason,
        regularization=lam,
        history=tuple(history),
        counts=problem.counts(),
        total_inner_iterations=inner_total,
        inner_solver=solver,
        wall_time=time.perf_counter() - started,
    )
