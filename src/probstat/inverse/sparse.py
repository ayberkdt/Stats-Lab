"""Sparse and total-variation regularised least squares by proximal methods.

Two convex inverse problems with a data term
``1/2 ||W (A x - y)||^2``, where ``W = C^-1`` whitens with a covariance
factor ``C`` (``noise``; identity when omitted):

- :func:`l1_regularized_least_squares` (lasso, with optional reference,
  weights, bounds and nonnegativity) solved by ISTA or FISTA
  [EQ-INV-SPARSE-001];
- :func:`total_variation_regularized_least_squares` (anisotropic or isotropic
  total variation) solved by the Chambolle-Pock primal-dual method on the
  stacked operator ``K = [W A; D]`` or by ADMM with a CGLS x-update
  [EQ-INV-SPARSE-003] to [EQ-INV-SPARSE-005].

Coefficient convention. ``alpha`` multiplies an UNSQUARED penalty
(``alpha ||x - x_ref||_1`` or ``alpha TV(x)``), unlike the Tikhonov
parameter of :mod:`probstat.inverse.regularization`, where ``lambda^2``
multiplies a squared norm. The two are not interchangeable: ``alpha`` has the
units of the data-misfit ``1/2||W r||^2`` per unit of ``x``. The value
``alpha_max = lasso_zero_threshold(...)`` is the smallest ``alpha`` for which
the lasso solution equals the reference [EQ-INV-SPARSE-002].

The L1 solution is biased towards the reference; :func:`l1_regularized_least_squares`
can return, as a SEPARATE field, an unpenalised least-squares refit on the
selected support (``debias=True``) [EQ-INV-SPARSE-006], which is not
guaranteed to improve the estimate. Total variation produces staircasing:
smooth ramps are replaced by plateaus.

Experimental (added in 0.28): the API may change before it is declared stable.

References:
    [REF-BECK-TEBOULLE-2009], [REF-CHAMBOLLE-POCK-2011],
    [REF-BOYD-ETAL-2011], [REF-RUDIN-OSHER-FATEMI-1992].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import (
    ComposedOperator,
    CountingOperator,
    FirstDifferenceOperator,
    GridGradientOperator,
    IdentityOperator,
    LinearOperator,
    OperatorCounts,
    ScaledOperator,
    SelectionOperator,
    TransposedOperator,
    Vector,
    VStackOperator,
    cgls,
    is_covariance_factor,
    materialize,
    require_operator,
)
from probstat.optimization.proximal import (
    ADMMResult,
    PDHGResult,
    ProximalGradientResult,
    admm,
    fista,
    ista,
    primal_dual_hybrid_gradient,
    project_box,
    project_linf_ball,
    project_pointwise_l2_ball,
    prox_l1,
    prox_pointwise_l2,
    soft_threshold,
)

_FROBENIUS_LIMIT = 250_000


def _vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Return ``values`` as a finite float tuple of the given length.

    Args:
        values: Candidate vector.
        size: Required length.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length differs or an entry is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a sequence of real numbers")
    items = tuple(values)
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only real numbers")
    result = tuple(float(item) for item in items)
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _real(value: float, name: str) -> float:
    """Return a finite real scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    """Return a finite non-negative scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is negative or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = _real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    """Return a finite strictly positive scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is not strictly positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = _real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    Args:
        vector: Finite vector.

    Returns:
        ``||v||_2``.

    References:
        [EQ-INV-SPARSE-001].

    """
    return math.hypot(*vector)


class _Misfit:
    """Weighted data misfit ``1/2||W(A x - y)||^2`` and its gradient.

    The last evaluated residual is cached so that a value and a gradient at
    the same point cost one forward application of ``A`` in total.

    References:
        [EQ-INV-SPARSE-001].

    """

    __slots__ = ("_cache", "_noise", "operator", "data")

    def __init__(self, operator: LinearOperator, data: Vector, noise: object) -> None:
        """Store the (counted) operator, the data and the covariance factor.

        Args:
            operator: Operator ``A`` (``m x n``).
            data: Data ``y`` of length ``m``.
            noise: ``None`` or a covariance factor of dimension ``m``.

        Raises:
            TypeError: If ``noise`` is not ``None`` or a covariance factor.
            ValueError: If the covariance dimension differs from ``m``.

        References:
            [EQ-INV-SPARSE-001].

        """
        if noise is not None:
            if not is_covariance_factor(noise):
                raise TypeError("noise must be None or a covariance factor")
            if noise.dimension != operator.shape[0]:  # type: ignore[attr-defined]
                raise ValueError("noise dimension must equal the number of data")
        self.operator = operator
        self.data = data
        self._noise = noise
        self._cache: tuple[Vector, Vector, Vector] | None = None

    def whiten(self, vector: Sequence[float]) -> Vector:
        """Return ``W v`` (the identity without a covariance factor).

        References:
            [EQ-INV-SPARSE-001].

        """
        if self._noise is None:
            return tuple(vector)
        return self._noise.whiten(vector)  # type: ignore[attr-defined, no-any-return]

    def whiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``W' v``.

        References:
            [EQ-INV-SPARSE-001].

        """
        if self._noise is None:
            return tuple(vector)
        return self._noise.whiten_transpose(vector)  # type: ignore[attr-defined, no-any-return]

    def whitening_operator(self) -> LinearOperator:
        """Return ``W`` as an operator.

        References:
            [EQ-INV-SPARSE-001].

        """
        if self._noise is None:
            return IdentityOperator(self.operator.shape[0])
        return self._noise.whitening_operator()  # type: ignore[attr-defined, no-any-return]

    def _evaluate(self, point: Vector) -> tuple[Vector, Vector]:
        """Return ``(A x - y, W (A x - y))`` with a one-entry cache.

        References:
            [EQ-INV-SPARSE-001].

        """
        cached = self._cache
        if cached is not None and cached[0] == point:
            return cached[1], cached[2]
        ax = self.operator.matvec(point)
        residual = tuple(a - b for a, b in zip(ax, self.data, strict=True))
        whitened = self.whiten(residual)
        self._cache = (point, residual, whitened)
        return residual, whitened

    def residual(self, point: Vector) -> Vector:
        """Return the raw residual ``A x - y``.

        References:
            [EQ-INV-SPARSE-001].

        """
        return self._evaluate(point)[0]

    def value(self, point: Vector) -> float:
        """Return ``1/2 ||W (A x - y)||^2``.

        References:
            [EQ-INV-SPARSE-001].

        """
        whitened = self._evaluate(point)[1]
        return 0.5 * math.fsum(a * a for a in whitened)

    def gradient(self, point: Vector) -> Vector:
        """Return ``A' W' W (A x - y)``.

        References:
            [EQ-INV-SPARSE-001].

        """
        whitened = self._evaluate(point)[1]
        return tuple(self.operator.rmatvec(self.whiten_transpose(whitened)))


def lasso_zero_threshold(
    operator: LinearOperator,
    data: Sequence[float],
    noise: object = None,
    *,
    reference: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
) -> float:
    """Return the smallest ``alpha`` whose lasso solution is the reference.

    With ``g = A' W' W (y - A x_ref)`` the optimality condition at
    ``x = x_ref`` is ``|g_i| <= alpha w_i`` for every ``i``, so the value is
    ``max_i |g_i| / w_i`` (``max_i |g_i|`` for unit weights; with
    ``x_ref = 0`` this is ``||A' W' W y||_inf``). Any ``alpha`` at or above it
    gives ``x = x_ref`` [EQ-INV-SPARSE-002]. Bounds are not considered.

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        noise: ``None`` or a covariance factor of dimension ``m``.
        reference: Reference ``x_ref`` (zero by default).
        weights: Non-negative per-coefficient weights (default 1); a zero
            weight with a nonzero ``g_i`` gives ``inf``.

    Returns:
        The threshold ``alpha_max`` (``inf`` if a zero weight blocks it).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension or weight is invalid.

    References:
        [EQ-INV-SPARSE-002], [REF-BECK-TEBOULLE-2009].

    """
    a = require_operator(operator, name="operator")
    m, n = a.shape
    y = _vector(data, m, "data")
    ref = (0.0,) * n if reference is None else _vector(reference, n, "reference")
    w = (1.0,) * n if weights is None else _vector(weights, n, "weights")
    if any(item < 0.0 for item in w):
        raise ValueError("weights must be non-negative")
    misfit = _Misfit(a, y, noise)
    grad = misfit.gradient(ref)
    best = 0.0
    for g, wi in zip(grad, w, strict=True):
        if g == 0.0:
            continue
        if wi == 0.0:
            return math.inf
        best = max(best, abs(g) / wi)
    return best


@dataclass(frozen=True, slots=True)
class L1Result:
    """Result of :func:`l1_regularized_least_squares`.

    Attributes:
        solution: Minimiser estimate ``x``.
        objective: ``1/2||W(Ax - y)||^2 + alpha sum w_i |x_i - c_i|`` at the
            solution, recomputed from the solution.
        data_misfit: ``1/2||W(Ax - y)||^2``.
        penalty: ``alpha sum w_i |x_i - c_i|``.
        residual: Raw residual ``A x - y``.
        support: Sorted indices with ``x_i != c_i``.
        alpha: The penalty coefficient (an unsquared multiplier).
        zero_threshold: ``alpha_max`` of :func:`lasso_zero_threshold`.
        optimization: The underlying :class:`ProximalGradientResult`.
        operator_counts: Forward and transpose applications of ``A`` during
            the proximal iteration (excluding the debias refit).
        debiased_solution: Unpenalised least-squares refit on the support
            (``None`` unless requested). Not guaranteed to be better.
        debias_converged: Convergence flag of the refit (``None`` if absent).
        debias_iterations: CGLS iterations of the refit (``0`` if absent).
        converged: Convergence flag of the proximal iteration.
        stop_reason: Stop reason of the proximal iteration.
        method: ``"fista"`` or ``"ista"``.

    References:
        [EQ-INV-SPARSE-001], [EQ-INV-SPARSE-002], [EQ-INV-SPARSE-006].

    """

    solution: Vector
    objective: float
    data_misfit: float
    penalty: float
    residual: Vector
    support: tuple[int, ...]
    alpha: float
    zero_threshold: float
    optimization: ProximalGradientResult
    operator_counts: OperatorCounts
    debiased_solution: Vector | None
    debias_converged: bool | None
    debias_iterations: int
    converged: bool
    stop_reason: str
    method: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def _bounds_pair(
    bounds: tuple[object, object] | None, n: int, nonnegative: bool
) -> tuple[Vector | None, Vector | None]:
    """Combine ``bounds`` and the nonnegativity flag into per-entry bounds.

    Args:
        bounds: ``None`` or ``(lower, upper)``, each a scalar, a sequence or
            ``None``.
        n: Dimension.
        nonnegative: Whether ``x >= 0`` is imposed in addition.

    Returns:
        ``(lower, upper)`` as vectors or ``None`` when unbounded.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the bounds are infeasible or have the wrong length.

    References:
        [EQ-INV-SPARSE-001].

    """
    lower: list[float] = [-math.inf] * n
    upper: list[float] = [math.inf] * n
    if bounds is not None:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError("bounds must be a (lower, upper) pair")
        for target, raw, name in (
            (lower, bounds[0], "lower"),
            (upper, bounds[1], "upper"),
        ):
            if raw is None:
                continue
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = _real(raw, name)
                target[:] = [value] * n
            else:
                target[:] = list(_vector(raw, n, name))  # type: ignore[arg-type]
    if nonnegative:
        lower = [max(item, 0.0) for item in lower]
    if any(a > b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("bounds are infeasible (lower exceeds upper)")
    has_lower = any(item > -math.inf for item in lower)
    has_upper = any(item < math.inf for item in upper)
    return (tuple(lower) if has_lower else None, tuple(upper) if has_upper else None)


def l1_regularized_least_squares(
    operator: LinearOperator,
    data: Sequence[float],
    alpha: float,
    *,
    noise: object = None,
    reference: Sequence[float] | None = None,
    method: str = "fista",
    nonnegative: bool = False,
    bounds: tuple[object, object] | None = None,
    weights: Sequence[float] | None = None,
    lipschitz: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 5000,
    restart: bool = True,
    debias: bool = False,
    debias_tolerance: float = 1e-10,
) -> L1Result:
    """Minimise ``1/2||W(A x - y)||^2 + alpha sum_i w_i |x_i - c_i|``.

    ``alpha`` multiplies an UNSQUARED penalty (compare ``lambda^2`` in
    Tikhonov regularisation). The smooth part has gradient
    ``A' W' W (A x - y)``; the penalty has the prox
    ``clip(c + soft(v - c, t alpha w))`` onto the optional box, so
    ``nonnegative`` and ``bounds`` are exact constraints, not post-hoc
    clipping [EQ-INV-SPARSE-001]. The step is found by backtracking unless
    ``lipschitz`` (a bound on ``||W A||_2^2``) is supplied; nothing relies on a
    spectral estimate. For ``alpha >= alpha_max`` (see
    :func:`lasso_zero_threshold`) the solution equals the reference
    [EQ-INV-SPARSE-002]. With ``alpha = 0`` the problem is plain least
    squares, which proximal gradient solves slowly when ``A`` is ill
    conditioned; with a rank-deficient ``A`` the minimiser is not unique.

    With ``debias=True`` an unpenalised least-squares refit over the selected
    support (coefficients off the support fixed at the reference) is returned
    in ``debiased_solution``; it is unconstrained, so it is refused together
    with ``bounds`` or ``nonnegative`` [EQ-INV-SPARSE-006].

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha: Non-negative penalty coefficient.
        noise: ``None`` or a covariance factor (``W = C^-1``) of dimension
            ``m``.
        reference: Penalty centre ``c`` (zero by default).
        method: ``"fista"`` or ``"ista"``.
        nonnegative: Impose ``x >= 0``.
        bounds: Optional ``(lower, upper)``; each a scalar, a sequence or
            ``None``.
        weights: Non-negative per-coefficient weights ``w`` (default 1).
        lipschitz: Optional bound on the Lipschitz constant of the gradient.
        initial: Start point; must be feasible. Default is the projection of
            the reference onto the box.
        tolerance: Gradient-mapping tolerance.
        max_iterations: Iteration cap.
        restart: Function-value restart for FISTA.
        debias: Also compute the refit on the support.
        debias_tolerance: Normal-residual tolerance of the refit.

    Returns:
        :class:`L1Result`. ``converged`` is ``False`` when the iteration cap
        is hit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``alpha`` is negative or not finite, a dimension or
            bound is invalid, ``method`` is unknown, ``initial`` is
            infeasible, or ``debias`` is combined with bounds.

    References:
        [EQ-INV-SPARSE-001], [EQ-INV-SPARSE-002], [EQ-INV-SPARSE-006],
        [REF-BECK-TEBOULLE-2009], [REF-DAUBECHIES-DEFRISE-DEMOL-2004].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    y = _vector(data, m, "data")
    penalty_scale = _nonnegative(alpha, "alpha")
    if method not in ("fista", "ista"):
        raise ValueError("method must be 'fista' or 'ista'")
    ref = (0.0,) * n if reference is None else _vector(reference, n, "reference")
    w = (1.0,) * n if weights is None else _vector(weights, n, "weights")
    if any(item < 0.0 for item in w):
        raise ValueError("weights must be non-negative")
    lower, upper = _bounds_pair(bounds, n, bool(nonnegative))
    constrained = lower is not None or upper is not None
    if debias and constrained:
        raise ValueError("debias refit is unconstrained; not allowed with bounds")
    counted = CountingOperator(raw)
    misfit = _Misfit(counted, y, noise)
    if initial is None:
        start = project_box(ref, lower, upper)
    else:
        start = _vector(initial, n, "initial")
        if project_box(start, lower, upper) != start:
            raise ValueError("initial violates the bounds")

    def penalty_value(point: Vector) -> float:
        """Return ``alpha sum w_i |x_i - c_i|``.

        References:
            [EQ-INV-SPARSE-001].

        """
        return penalty_scale * math.fsum(
            wi * abs(a - c) for wi, a, c in zip(w, point, ref, strict=True)
        )

    def prox(point: Vector, step: float) -> Vector:
        """Return the prox of ``step * penalty`` on the box.

        References:
            [EQ-INV-SPARSE-001].

        """
        return prox_l1(
            point,
            step * penalty_scale,
            weights=w,
            center=ref,
            lower=lower,
            upper=upper,
        )

    solver = fista if method == "fista" else ista
    kwargs = {"restart": restart} if method == "fista" else {}
    result = solver(
        misfit.value,
        misfit.gradient,
        penalty_value,
        prox,
        start,
        lipschitz=lipschitz,
        tolerance=tolerance,
        max_iterations=max_iterations,
        **kwargs,
    )
    counts = counted.counts
    x = result.solution
    final = _Misfit(raw, y, noise)
    residual = final.residual(x)
    data_misfit = final.value(x)
    penalty = penalty_value(x)
    support = tuple(i for i, (a, c) in enumerate(zip(x, ref, strict=True)) if a != c)

    debiased: Vector | None = None
    debias_ok: bool | None = None
    debias_its = 0
    if debias:
        if not support:
            debiased, debias_ok = ref, True
        else:
            scatter = TransposedOperator(SelectionOperator(n, support))
            reduced = ComposedOperator([final.whitening_operator(), raw, scatter])
            rhs = final.whiten(
                tuple(a - b for a, b in zip(y, raw.matvec(ref), strict=True))
            )
            fit = cgls(
                reduced,
                rhs,
                residual_tolerance=0.0,
                normal_tolerance=_positive(debias_tolerance, "debias_tolerance"),
            )
            lift = scatter.matvec(fit.solution)
            debiased = tuple(a + b for a, b in zip(ref, lift, strict=True))
            debias_ok = fit.converged
            debias_its = fit.iterations
    return L1Result(
        solution=x,
        objective=data_misfit + penalty,
        data_misfit=data_misfit,
        penalty=penalty,
        residual=residual,
        support=support,
        alpha=penalty_scale,
        zero_threshold=lasso_zero_threshold(raw, y, noise, reference=ref, weights=w),
        optimization=result,
        operator_counts=counts,
        debiased_solution=debiased,
        debias_converged=debias_ok,
        debias_iterations=debias_its,
        converged=result.converged,
        stop_reason=result.stop_reason,
        method=method,
    )


# ---------------------------------------------------------------------------
# Total variation
# ---------------------------------------------------------------------------


def _default_components(gradient: LinearOperator) -> int:
    """Return 2 for a grid gradient and 1 for any other difference operator.

    Args:
        gradient: Discrete gradient operator.

    Returns:
        The number of stacked components per point.

    References:
        [EQ-INV-SPARSE-003].

    """
    return 2 if isinstance(gradient, GridGradientOperator) else 1


def _point_weights(weights: Sequence[float] | None, points: int) -> Vector:
    """Return per-point weights (default one).

    Args:
        weights: Non-negative weights or ``None``.
        points: Number of points ``N``.

    Returns:
        Weight vector of length ``N``.

    Raises:
        ValueError: If a weight is negative or the length is wrong.

    References:
        [EQ-INV-SPARSE-003].

    """
    if weights is None:
        return (1.0,) * points
    result = _vector(weights, points, "weights")
    if any(item < 0.0 for item in result):
        raise ValueError("weights must be non-negative")
    return result


def _tv_of_gradient(
    values: Sequence[float],
    components: int,
    weights: Vector,
    isotropic: bool,
) -> float:
    """Return ``sum_i w_i |g_i|`` for a block-stacked gradient ``g``.

    Args:
        values: Block-stacked gradient values.
        components: Number of blocks.
        weights: Per-point weights.
        isotropic: Euclidean (``True``) or absolute-value (``False``) norm.

    Returns:
        The weighted total variation.

    References:
        [EQ-INV-SPARSE-003].

    """
    count = len(weights)
    total = []
    for i in range(count):
        entries = [values[k * count + i] for k in range(components)]
        point = (
            math.hypot(*entries) if isotropic else math.fsum(abs(e) for e in entries)
        )
        total.append(weights[i] * point)
    return math.fsum(total)


def total_variation_value(
    values: Sequence[float],
    gradient: LinearOperator | None = None,
    *,
    isotropic: bool = True,
    components: int | None = None,
    weights: Sequence[float] | None = None,
) -> float:
    """Return the weighted discrete total variation ``sum_i w_i |(D x)_i|``.

    ``|.|`` is the Euclidean norm of the ``components`` gradient entries at a
    grid point (isotropic) or their sum of absolute values (anisotropic);
    for a one-dimensional difference operator both coincide. ``D`` is the
    operator as given, so the boundary rule and the grid spacing of
    :class:`~probstat.linalg.FirstDifferenceOperator` or
    :class:`~probstat.linalg.GridGradientOperator` define the discretisation;
    ``weights`` are area (cell-measure) weights. A constant signal has zero
    total variation [EQ-INV-SPARSE-003].

    Args:
        values: The signal ``x``.
        gradient: ``D`` (default :class:`~probstat.linalg.FirstDifferenceOperator`
            without boundary).
        isotropic: Isotropic (``True``) or anisotropic (``False``) norm.
        components: Blocks of ``D x`` per point (2 for a grid gradient, else 1).
        weights: Non-negative per-point weights (default 1).

    Returns:
        The total variation.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension or weight is invalid.

    References:
        [EQ-INV-SPARSE-003], [REF-RUDIN-OSHER-FATEMI-1992].

    """
    x = tuple(float(a) for a in values)
    d = (
        FirstDifferenceOperator(len(x))
        if gradient is None
        else require_operator(gradient, name="gradient")
    )
    x = _vector(x, d.shape[1], "values")
    c = _default_components(d) if components is None else components
    if isinstance(c, bool) or not isinstance(c, int) or c < 1:
        raise ValueError("components must be a positive integer")
    if d.shape[0] % c != 0:
        raise ValueError("gradient rows must be a multiple of components")
    w = _point_weights(weights, d.shape[0] // c)
    return _tv_of_gradient(d.matvec(x), c, w, isotropic)


def frobenius_norm_bound(
    operator: LinearOperator, *, max_entries: int = _FROBENIUS_LIMIT
) -> float:
    """Return the CERTIFIED upper bound ``||A||_F >= ||A||_2``.

    The matrix is built column by column with
    :func:`~probstat.linalg.materialize`; the bound can exceed ``||A||_2`` by
    up to a factor ``sqrt(rank)``, which only makes primal-dual steps shorter.

    Args:
        operator: ``A``.
        max_entries: Largest ``m n`` that may be materialised.

    Returns:
        ``sqrt(sum a_ij^2)``.

    Raises:
        ValueError: If the operator is too large to materialise.

    References:
        [EQ-INV-SPARSE-008], [REF-GOLUB-VANLOAN-2013].

    """
    matrix = materialize(require_operator(operator), max_entries=max_entries)
    return math.sqrt(math.fsum(a * a for row in matrix for a in row))


def _gradient_bound(gradient: LinearOperator) -> float | None:
    """Return a closed-form bound ``||D||_2 <= b`` for the known stencils.

    One-dimensional differences satisfy ``||D|| <= 2/h`` for every boundary
    rule; the grid gradient satisfies ``||D||^2 <= 4/h_x^2 + 4/h_y^2``.

    Args:
        gradient: Discrete gradient operator.

    Returns:
        The bound, or ``None`` for an unrecognised operator.

    References:
        [EQ-INV-SPARSE-008], [REF-CHAMBOLLE-POCK-2011].

    """
    if isinstance(gradient, FirstDifferenceOperator):
        return 2.0 / gradient.spacing
    if isinstance(gradient, GridGradientOperator):
        hy, hx = gradient.spacing
        return math.sqrt(4.0 / (hx * hx) + 4.0 / (hy * hy))
    return None


@dataclass(frozen=True, slots=True)
class TotalVariationResult:
    """Result of :func:`total_variation_regularized_least_squares`.

    Attributes:
        solution: Estimate ``x``.
        objective: ``1/2||W(Ax - y)||^2 + alpha TV(x)`` recomputed from the
            solution (independent of the solver's own bookkeeping).
        data_misfit: ``1/2||W(Ax - y)||^2``.
        total_variation: ``TV(x)`` (unscaled by ``alpha``).
        penalty: ``alpha TV(x)``.
        residual: Raw residual ``A x - y``.
        alpha: The penalty coefficient (an unsquared multiplier).
        optimization: :class:`~probstat.optimization.PDHGResult` or
            :class:`~probstat.optimization.ADMMResult`.
        method: ``"pdhg"`` or ``"admm"``.
        isotropic: Norm used for the total variation.
        norm_bound: The bound on ``||[W A; D]||_2`` used for PDHG steps
            (``nan`` for ADMM).
        inner_iterations: Total CGLS iterations of the ADMM x-updates (``0``
            for PDHG).
        inner_failures: Number of x-updates whose CGLS did not meet its
            tolerance (``0`` for PDHG).
        inner_tolerance: Relative normal-residual tolerance of the CGLS
            solves (``nan`` for PDHG).
        operator_counts: Applications of ``A`` made by the solver.
        gradient_counts: Applications of ``D`` made by the solver.
        converged: ``True`` only when the outer tolerance was met and no
            inner solve failed.
        stop_reason: Outer stop reason, or ``"inner_solver_failed"``.

    References:
        [EQ-INV-SPARSE-003], [EQ-INV-SPARSE-004], [EQ-INV-SPARSE-005].

    """

    solution: Vector
    objective: float
    data_misfit: float
    total_variation: float
    penalty: float
    residual: Vector
    alpha: float
    optimization: PDHGResult | ADMMResult
    method: str
    isotropic: bool
    norm_bound: float
    inner_iterations: int
    inner_failures: int
    inner_tolerance: float
    operator_counts: OperatorCounts
    gradient_counts: OperatorCounts
    converged: bool
    stop_reason: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


class _TVProxies:
    """Dual and split proximal maps of the weighted total-variation norm.

    References:
        [EQ-INV-SPARSE-003], [EQ-INV-SPARSE-004].

    """

    __slots__ = ("alpha", "components", "isotropic", "radius", "weights")

    def __init__(
        self, alpha: float, components: int, weights: Vector, isotropic: bool
    ) -> None:
        """Store the penalty, the layout and the weights.

        Args:
            alpha: Penalty coefficient.
            components: Blocks per point.
            weights: Per-point weights.
            isotropic: Euclidean (``True``) or absolute-value norm.

        References:
            [EQ-INV-SPARSE-003].

        """
        self.alpha = alpha
        self.components = components
        self.isotropic = isotropic
        self.weights = weights
        self.radius = tuple(alpha * wi for wi in weights)

    def project_dual(self, vector: Sequence[float]) -> Vector:
        """Project onto the dual ball ``|p_i| <= alpha w_i`` (conjugate prox).

        References:
            [EQ-INV-SPARSE-004].

        """
        if self.isotropic and self.components > 1:
            return project_pointwise_l2_ball(
                vector, self.radius, components=self.components
            )
        return project_linf_ball(vector, self.radius * self.components)

    def prox_norm(self, vector: Sequence[float], step: float) -> Vector:
        """Return the prox of ``step alpha sum w_i |.|``.

        References:
            [EQ-INV-SPARSE-005].

        """
        if self.isotropic and self.components > 1:
            return prox_pointwise_l2(
                vector,
                step,
                components=self.components,
                weights=self.radius,
            )
        return soft_threshold(
            vector, tuple(step * r for r in self.radius * self.components)
        )


class _AdmmXUpdate:
    """CGLS x-update of the TV-ADMM: ``min 1/2||W(A x - y)||^2 + rho/2||D x - v||^2``.

    The stacked least-squares problem ``[W A; sqrt(rho) D] x = [W y; sqrt(rho) v]``
    is solved from a zero start so that the relative normal-residual tolerance
    refers to a fixed scale; inner iterations and failures are accumulated.

    References:
        [EQ-INV-SPARSE-005].

    """

    __slots__ = ("data", "failures", "forward", "gradient", "iterations", "tolerance")

    def __init__(
        self,
        forward: LinearOperator,
        gradient: LinearOperator,
        whitened_data: Vector,
        tolerance: float,
    ) -> None:
        """Store the operators, the whitened data and the inner tolerance.

        Args:
            forward: ``W A`` (counted).
            gradient: ``D`` (counted).
            whitened_data: ``W y``.
            tolerance: Relative normal-residual tolerance of CGLS.

        References:
            [EQ-INV-SPARSE-005].

        """
        self.forward = forward
        self.gradient = gradient
        self.data = whitened_data
        self.tolerance = tolerance
        self.iterations = 0
        self.failures = 0

    def __call__(self, v: Vector, rho: float, x_old: Vector) -> Vector:
        """Solve the x-subproblem for the penalty ``rho``.

        References:
            [EQ-INV-SPARSE-005].

        """
        root = math.sqrt(rho)
        stacked = VStackOperator([self.forward, ScaledOperator(self.gradient, root)])
        rhs = self.data + tuple(root * a for a in v)
        fit = cgls(
            stacked,
            rhs,
            residual_tolerance=0.0,
            normal_tolerance=self.tolerance,
            max_iterations=max(200, 4 * stacked.shape[1]),
        )
        self.iterations += fit.iterations
        if not fit.converged:
            self.failures += 1
        return fit.solution


def total_variation_regularized_least_squares(
    operator: LinearOperator,
    data: Sequence[float],
    alpha: float,
    *,
    noise: object = None,
    gradient: LinearOperator | None = None,
    isotropic: bool = True,
    weights: Sequence[float] | None = None,
    components: int | None = None,
    method: str = "pdhg",
    bounds: tuple[object, object] | None = None,
    forward_norm_bound: float | None = None,
    gradient_norm_bound: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance_absolute: float = 1e-8,
    tolerance_relative: float = 1e-6,
    max_iterations: int = 20000,
    step_safety: float = 0.99,
    theta: float = 1.0,
    rho: float = 1.0,
    adapt_rho: bool = False,
    inner_tolerance: float = 1e-10,
) -> TotalVariationResult:
    """Minimise ``1/2||W(A x - y)||^2 + alpha TV(x)`` over an optional box.

    ``TV(x) = sum_i w_i |(D x)_i|`` with the isotropic or anisotropic norm of
    :func:`total_variation_value`; ``alpha`` multiplies an UNSQUARED penalty.
    The default gradient is the one-dimensional forward difference without
    boundary rows; pass a
    :class:`~probstat.linalg.GridGradientOperator` for images.

    ``method="pdhg"`` runs :func:`~probstat.optimization.primal_dual_hybrid_gradient`
    on ``K = [W A; D]`` with ``g(u, v) = 1/2||u - W y||^2 + alpha TV(v)``:
    the dual prox is ``(p - sigma W y) / (1 + sigma)`` on the data block and the
    projection onto the balls ``|q_i| <= alpha w_i`` on the gradient block;
    the primal prox is the box projection [EQ-INV-SPARSE-004]. The step
    condition uses ``||K||^2 <= ||W A||^2 + ||D||^2`` with CERTIFIED bounds:
    ``forward_norm_bound`` (default the Frobenius norm of ``W A``, which needs
    a materialisable operator) and ``gradient_norm_bound`` (default closed
    forms for the two stencil classes; mandatory for any other ``D``)
    [EQ-INV-SPARSE-008].

    ``method="admm"`` splits ``z = D x`` and solves the x-update by CGLS on
    ``[W A; sqrt(rho) D]`` [EQ-INV-SPARSE-005]; the inner tolerance and cost
    are reported. Box constraints are not supported by this x-update and are
    refused. No debiasing refit is offered for total variation.

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha: Non-negative penalty coefficient.
        noise: ``None`` or a covariance factor of dimension ``m``.
        gradient: Discrete gradient ``D``.
        isotropic: Euclidean (``True``) or absolute-value (``False``) norm of
            the gradient at each point (only matters for ``components > 1``).
        weights: Non-negative per-point (area) weights.
        components: Blocks of ``D x`` per point (default 2 for a grid gradient,
            else 1).
        method: ``"pdhg"`` or ``"admm"``.
        bounds: Optional ``(lower, upper)`` box (PDHG only).
        forward_norm_bound: Certified bound on ``||W A||_2`` (PDHG).
        gradient_norm_bound: Certified bound on ``||D||_2`` (PDHG).
        initial: Start point (default zero, projected onto the box).
        tolerance_absolute: Absolute residual tolerance of the outer solver.
        tolerance_relative: Relative residual tolerance of the outer solver.
        max_iterations: Outer iteration cap.
        step_safety: PDHG step factor in ``(0, 1)``.
        theta: PDHG overrelaxation in ``[0, 1]``.
        rho: ADMM penalty parameter.
        adapt_rho: ADMM residual balancing.
        inner_tolerance: Relative normal-residual tolerance of the ADMM
            CGLS x-updates.

    Returns:
        :class:`TotalVariationResult`. ``converged`` is ``False`` when the cap
        is hit or an inner solve failed.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``alpha`` is negative or not finite, a dimension,
            weight or bound is invalid, ``method`` is unknown, a norm bound is
            missing for an unrecognised operator, or ``bounds`` is combined
            with ADMM.

    References:
        [EQ-INV-SPARSE-003], [EQ-INV-SPARSE-004], [EQ-INV-SPARSE-005],
        [EQ-INV-SPARSE-008], [REF-CHAMBOLLE-POCK-2011], [REF-BOYD-ETAL-2011],
        [REF-RUDIN-OSHER-FATEMI-1992].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    y = _vector(data, m, "data")
    scale = _nonnegative(alpha, "alpha")
    if method not in ("pdhg", "admm"):
        raise ValueError("method must be 'pdhg' or 'admm'")
    d_raw = (
        FirstDifferenceOperator(n)
        if gradient is None
        else require_operator(gradient, name="gradient")
    )
    if d_raw.shape[1] != n:
        raise ValueError("gradient must act on the same unknowns as operator")
    c = _default_components(d_raw) if components is None else components
    if isinstance(c, bool) or not isinstance(c, int) or c < 1:
        raise ValueError("components must be a positive integer")
    if d_raw.shape[0] % c != 0:
        raise ValueError("gradient rows must be a multiple of components")
    w = _point_weights(weights, d_raw.shape[0] // c)
    lower, upper = _bounds_pair(bounds, n, False)
    if method == "admm" and (lower is not None or upper is not None):
        raise ValueError("bounds are not supported by the ADMM x-update")
    start = (
        project_box((0.0,) * n, lower, upper)
        if initial is None
        else _vector(initial, n, "initial")
    )
    if project_box(start, lower, upper) != start:
        raise ValueError("initial violates the bounds")

    a_count = CountingOperator(raw)
    d_count = CountingOperator(d_raw)
    misfit = _Misfit(a_count, y, noise)
    proxies = _TVProxies(scale, c, w, isotropic)
    whitened_y = misfit.whiten(y)
    optimization: PDHGResult | ADMMResult
    norm_bound = math.nan
    inner_its = 0
    inner_fail = 0
    inner_tol = math.nan

    if method == "pdhg":
        wa: LinearOperator = (
            a_count
            if noise is None
            else ComposedOperator([misfit.whitening_operator(), a_count])
        )
        fwd_bound = (
            frobenius_norm_bound(
                wa
                if noise is None
                else ComposedOperator([misfit.whitening_operator(), raw])
            )
            if forward_norm_bound is None
            else _positive(forward_norm_bound, "forward_norm_bound")
        )
        if gradient_norm_bound is None:
            known = _gradient_bound(d_raw)
            if known is None:
                raise ValueError("gradient_norm_bound is required for this operator")
            grad_bound = known
        else:
            grad_bound = _positive(gradient_norm_bound, "gradient_norm_bound")
        norm_bound = math.hypot(fwd_bound, grad_bound)
        stacked = VStackOperator([wa, d_count])

        def prox_primal(point: Vector, step: float) -> Vector:
            """Project onto the box (identity when unbounded).

            References:
                [EQ-INV-SPARSE-004].

            """
            return project_box(point, lower, upper)

        def prox_dual(point: Vector, step: float) -> Vector:
            """Return the prox of ``step g*`` for the stacked data/TV term.

            References:
                [EQ-INV-SPARSE-004].

            """
            head = tuple(
                (a - step * b) / (1.0 + step)
                for a, b in zip(point[:m], whitened_y, strict=True)
            )
            return head + proxies.project_dual(point[m:])

        def pdhg_objective(point: Vector, image: Vector) -> float:
            """Return the primal objective from ``K x`` without extra work.

            References:
                [EQ-INV-SPARSE-003].

            """
            fit = 0.5 * math.fsum(
                (a - b) ** 2 for a, b in zip(image[:m], whitened_y, strict=True)
            )
            return fit + scale * _tv_of_gradient(image[m:], c, w, isotropic)

        optimization = primal_dual_hybrid_gradient(
            stacked,
            prox_primal,
            prox_dual,
            start,
            operator_norm_bound=norm_bound,
            step_safety=step_safety,
            theta=theta,
            objective=pdhg_objective,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            max_iterations=max_iterations,
        )
        converged = optimization.converged
        reason = optimization.stop_reason
    else:
        inner_tol = _positive(inner_tolerance, "inner_tolerance")
        wa_admm: LinearOperator = (
            a_count
            if noise is None
            else ComposedOperator([misfit.whitening_operator(), a_count])
        )
        x_update = _AdmmXUpdate(wa_admm, d_count, whitened_y, inner_tol)

        def admm_objective(point: Vector) -> float:
            """Return the true objective at an ADMM iterate.

            References:
                [EQ-INV-SPARSE-003].

            """
            return misfit.value(point) + scale * _tv_of_gradient(
                d_count.matvec(point), c, w, isotropic
            )

        optimization = admm(
            d_count,
            x_update,
            proxies.prox_norm,
            start,
            rho=rho,
            objective=admm_objective,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            adapt_rho=adapt_rho,
            max_iterations=max_iterations,
        )
        inner_its = x_update.iterations
        inner_fail = x_update.failures
        converged = optimization.converged and inner_fail == 0
        reason = optimization.stop_reason
        if optimization.converged and inner_fail > 0:
            reason = "inner_solver_failed"
    a_counts = a_count.counts
    d_counts = d_count.counts

    x = optimization.solution
    final = _Misfit(raw, y, noise)
    data_misfit = final.value(x)
    tv = _tv_of_gradient(d_raw.matvec(x), c, w, isotropic)
    return TotalVariationResult(
        solution=x,
        objective=data_misfit + scale * tv,
        data_misfit=data_misfit,
        total_variation=tv,
        penalty=scale * tv,
        residual=final.residual(x),
        alpha=scale,
        optimization=optimization,
        method=method,
        isotropic=isotropic,
        norm_bound=norm_bound,
        inner_iterations=inner_its,
        inner_failures=inner_fail,
        inner_tolerance=inner_tol,
        operator_counts=a_counts,
        gradient_counts=d_counts,
        converged=converged,
        stop_reason=reason,
    )


__all__ = [
    "L1Result",
    "TotalVariationResult",
    "frobenius_norm_bound",
    "l1_regularized_least_squares",
    "lasso_zero_threshold",
    "total_variation_regularized_least_squares",
    "total_variation_value",
]
