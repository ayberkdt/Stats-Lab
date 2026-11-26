"""Matrix-free trust-region Newton-CG inversion with Gauss-Newton or full Hessian.

For a :class:`~probstat.inverse.gauss_newton.DifferentiableForward` model the
penalized objective is
``Phi(x) = 1/2 ||W (G(x) - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2`` with
gradient ``J'W'W r + lambda^2 L'L (x - x_ref)``, ``r = G(x) - y``
[EQ-INV-NTR-001]. :func:`newton_trust_region_inversion` minimizes it with the
Steihaug trust-region Newton-CG method of
:func:`probstat.optimization.trust_region_newton_cg`; the Hessian of ``Phi`` is
only ever applied to vectors, never formed:

* ``hessian="gauss_newton"``: ``H v = J'W'W J v + lambda^2 L'L v`` through one
  ``jvp`` and one ``vjp`` [EQ-INV-NTR-002]. It is positive semidefinite and
  drops the residual-weighted second derivative.
* ``hessian="full"``: adds the term ``sum_i (W'W r)_i Hess G_i(x) v`` supplied
  by the user through ``hvp_callback(x, v, weighted_residual)`` with
  ``weighted_residual = W'W r`` [EQ-INV-NTR-003]. The full Hessian can be
  indefinite away from the optimum; the trust region (negative-curvature exit
  of Steihaug CG) handles that. There is NO implicit fallback: without a
  callback, ``hvp="central_difference_of_gradient"`` must be requested
  explicitly; it approximates ``Hess Phi v`` by
  ``(grad Phi(x + e v) - grad Phi(x - e v)) / (2 e)`` at the cost of two
  gradient evaluations (two forward values and two ``vjp`` calls) per action
  [EQ-INV-NTR-004].

The trial-step policy is a trust region (ratio of actual to predicted
reduction, radius update, rejection of poor steps) rather than a line search;
a trial whose forward value is not finite is a rejected step, never clipped
[EQ-INV-NTR-005]. ``lambda`` here is the statistical Tikhonov weight of the
objective; the trust radius is numerical globalization and is not a
regularization parameter: do not interpret a small radius as regularization.

An optional keyword-only discrepancy stop (``discrepancy_level`` ``delta`` and
``discrepancy_tau`` ``tau > 1``) ends the run at the first accepted iterate
with ``||W (G(x) - y)|| <= tau delta`` and ``stop_reason="discrepancy"``
[EQ-INV-NLX-007]; it is off by default and then changes nothing. Used with
``lambda`` fixed it is an early stop on the data misfit, not a rule that
chooses ``lambda``.

Assumptions:
    ``G`` is twice continuously differentiable near the iterates for
    ``hessian="full"``; the ``jvp``/``vjp`` are exact adjoints (check with
    :func:`probstat.inverse.gauss_newton.check_derivatives`); an
    ``hvp_callback`` returns the exact residual-weighted second derivative
    (verify it against differences of the gradient, as in the test-suite).

Common pitfalls:
    - ``reference`` defaults to zero (not the start), as in
      :func:`probstat.inverse.gauss_newton.gauss_newton_krylov`.
    - ``converged`` is ``True`` only for the stop reason ``"gradient"`` (and
      ``"discrepancy"`` when ``discrepancy_level`` is given).
    - The gradient test is relative to the initial gradient norm; with a
      large final objective it may be unreachable, and the run then ends with
      ``"radius_collapse"``, reported as not converged.
    - The default trust radii are in parameter units: scale ``initial_radius``
      and ``max_radius`` to the problem.

Experimental (added in 0.28).

References:
    [REF-NOCEDAL-WRIGHT-2006], [REF-STEIHAUG-1983], [REF-TIKHONOV-1963].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from probstat.inverse.gauss_newton import DifferentiableForward
from probstat.linalg import (
    IdentityOperator,
    LinearOperator,
    Vector,
    dot,
    is_covariance_factor,
    norm,
    require_operator,
)
from probstat.optimization.trust_region import (
    TrustRegionIteration,
    trust_region_newton_cg,
)

_EPS = 2.0**-52
_DEFAULT_FD_STEP: float = float(_EPS ** (1.0 / 3.0))
_CACHE_SIZE = 4

HvpCallback = Callable[[Vector, Vector, Vector], Sequence[float]]


@dataclass(frozen=True, slots=True)
class NewtonTrustRegionCounts:
    """Evaluation counts of :func:`newton_trust_region_inversion`.

    Attributes:
        value: Forward evaluations ``G(x)`` made by the solver (cached values
            are not recounted).
        jvp: ``J(x) v`` applications.
        vjp: ``J(x)' w`` applications (gradients and Hessian actions).
        hvp: Hessian actions of ``Phi`` requested by the CG solves (each one
            ``jvp`` + ``vjp`` for Gauss-Newton; plus one callback call for
            ``"full"``; two gradient evaluations for the
            central-difference mode).
        derivative_value: Forward evaluations spent inside a model that
            differentiates by finite differences (``0`` for analytic models).

    References:
        [EQ-INV-NTR-005].

    """

    value: int
    jvp: int
    vjp: int
    hvp: int
    derivative_value: int


@dataclass(frozen=True, slots=True)
class NewtonTrustRegionResult:
    """Result of :func:`newton_trust_region_inversion`.

    Attributes:
        estimate: Final iterate ``x``.
        objective: ``Phi(x)``.
        misfit: ``1/2 ||W (G(x) - y)||^2``.
        penalty: ``lambda^2/2 ||L (x - x_ref)||^2``.
        whitened_residual_norm: ``||W (G(x) - y)||``.
        gradient_norm: ``||grad Phi(x)||``.
        iterations: Accepted steps.
        attempts: Trial steps (accepted plus rejected).
        converged: ``True`` for ``stop_reason`` ``"gradient"`` or
            ``"discrepancy"``.
        stop_reason: ``"gradient"``, ``"discrepancy"`` (only when a
            ``discrepancy_level`` was requested), ``"radius_collapse"``,
            ``"max_iterations"`` or ``"nonfinite"``.
        history: One :class:`~probstat.optimization.TrustRegionIteration` per
            attempt (objective, gradient norm, radius, ``rho``, acceptance)
            plus the stopping record.
        counts: :class:`NewtonTrustRegionCounts`.
        total_inner_iterations: Sum of CG iterations.
        regularization: ``lambda``.
        hessian: ``"gauss_newton"`` or ``"full"``.
        hvp_mode: ``"none"`` (Gauss-Newton), ``"callback"`` or
            ``"central_difference_of_gradient"``.
        radius: Final trust radius.
        wall_time: Seconds (benchmark information, not deterministic).

    References:
        [EQ-INV-NTR-001], [EQ-INV-NTR-005].

    """

    estimate: Vector
    objective: float
    misfit: float
    penalty: float
    whitened_residual_norm: float
    gradient_norm: float
    iterations: int
    attempts: int
    converged: bool
    stop_reason: str
    history: tuple[TrustRegionIteration, ...]
    counts: NewtonTrustRegionCounts
    total_inner_iterations: int
    regularization: float
    hessian: str
    hvp_mode: str
    radius: float
    wall_time: float


def _vector(values: object, size: int, name: str) -> Vector:
    """Validate a finite real vector of a given length.

    Args:
        values: Candidate sequence of reals.
        size: Required length.
        name: Name used in messages.

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
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return tuple(out)


def _raw_vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Return a length-checked float tuple (entries may be non-finite).

    Args:
        values: Sequence returned by a user callback.
        size: Required length.
        name: Name used in messages.

    Returns:
        The entries as a tuple of floats.

    Raises:
        ValueError: If the length is wrong.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    out = tuple(float(v) for v in values)
    if len(out) != size:
        raise ValueError(f"{name} must have length {size}, got {len(out)}")
    return out


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


def newton_trust_region_inversion(
    model: DifferentiableForward,
    data: Sequence[float],
    *,
    noise: Any = None,
    regularization: float = 0.0,
    penalty: LinearOperator | None = None,
    reference: Sequence[float] | None = None,
    initial: Sequence[float] | None = None,
    hessian: Literal["gauss_newton", "full"] = "gauss_newton",
    hvp_callback: HvpCallback | None = None,
    hvp: Literal["callback", "central_difference_of_gradient"] | None = None,
    hvp_step: float | None = None,
    initial_radius: float = 1.0,
    max_radius: float = 1e3,
    eta: float = 0.1,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-8,
    inner_tolerance: float = 0.5,
    forcing: Literal["superlinear", "constant"] = "superlinear",
    inner_max_iterations: int | None = None,
    minimum_radius: float = 1e-14,
    discrepancy_level: float | None = None,
    discrepancy_tau: float = 1.1,
) -> NewtonTrustRegionResult:
    """Minimize ``Phi`` by matrix-free trust-region Newton-CG.

    ``Phi(x) = 1/2 ||W (G(x) - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2`` with
    ``W'W = Gamma^-1`` from ``noise``, ``L = I`` and ``x_ref = 0`` by default
    [EQ-INV-NTR-001]. Every outer attempt applies Steihaug CG to the
    quadratic model of ``Phi`` with the chosen Hessian action inside the trust
    region, then accepts or rejects the step by ``rho`` and updates the
    radius [EQ-INV-NTR-005]; see
    :func:`probstat.optimization.trust_region_newton_cg` for the radius rules,
    forcing term and stop reasons.

    Hessian choices:

    * ``hessian="gauss_newton"`` (default): ``J'W'W J v + lambda^2 L'L v``
      [EQ-INV-NTR-002]; ``hvp_callback`` and ``hvp`` must not be given.
    * ``hessian="full"``: Gauss-Newton action plus
      ``sum_i (W'W r)_i Hess G_i v`` [EQ-INV-NTR-003]. Exactly one source for
      the second-derivative term is required: ``hvp_callback(x, v,
      weighted_residual)`` returning that sum for ``weighted_residual =
      W'W r`` (length ``m``), or ``hvp="central_difference_of_gradient"``,
      which differentiates the gradient of ``Phi`` along ``v`` by central
      differences with step ``e = hvp_step (1 + ||x||) / ||v||`` (default
      ``hvp_step = eps^(1/3)``) at two gradient evaluations per action
      [EQ-INV-NTR-004]. A missing source raises ``ValueError``. At a
      residual ``r = 0`` the two Hessians coincide.

    Args:
        model: Differentiable forward map.
        data: Observations ``y`` (length ``m``).
        noise: Covariance factor of ``probstat.linalg`` (dimension ``m``) or
            ``None`` for unit noise.
        regularization: ``lambda >= 0``.
        penalty: ``L`` with ``n`` columns (default identity).
        reference: ``x_ref`` (default zeros; NOT the start).
        initial: Start ``x_0`` (default zeros).
        hessian: ``"gauss_newton"`` or ``"full"``.
        hvp_callback: ``(x, v, weighted_residual) -> sum_i w_i Hess G_i v``.
        hvp: ``None``/``"callback"`` or ``"central_difference_of_gradient"``.
        hvp_step: Relative step of the central-difference Hessian action
            (``> 0``); only with ``hvp="central_difference_of_gradient"``.
        initial_radius: ``Delta_0 > 0`` (parameter units).
        max_radius: Largest radius ``>= initial_radius``.
        eta: Acceptance threshold in ``[0, 1/4)``.
        max_iterations: Cap on attempts (``>= 1``).
        gradient_tolerance: Relative gradient tolerance (``>= 0``).
        inner_tolerance: Forcing cap or constant (``> 0``).
        forcing: ``"superlinear"`` or ``"constant"``.
        inner_max_iterations: CG cap per attempt (default ``n``).
        minimum_radius: Relative radius floor (``> 0``).
        discrepancy_level: Noise level ``delta > 0`` of the whitened data
            (``None``, the default, disables the test and leaves the
            behaviour unchanged). With a level the run also stops with
            ``stop_reason="discrepancy"`` at the first accepted iterate with
            ``||W (G(x) - y)|| <= tau delta`` (Morozov's discrepancy
            principle, [EQ-INV-NLX-007]); the gradient test, evaluated first
            on the true gradient, keeps priority.
        discrepancy_tau: ``tau > 1`` (default ``1.1``); only used with
            ``discrepancy_level``.

    Returns:
        :class:`NewtonTrustRegionResult`; ``converged`` is ``True`` for
        ``"gradient"`` and ``"discrepancy"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension is inconsistent, an option is out of range,
            the Hessian source is missing or contradictory, or the forward map
            is not finite at ``initial``.

    References:
        [EQ-INV-NTR-001], [EQ-INV-NTR-002], [EQ-INV-NTR-003],
        [EQ-INV-NTR-004], [EQ-INV-NTR-005], [EQ-INV-NLX-007],
        [REF-NOCEDAL-WRIGHT-2006], [REF-STEIHAUG-1983], [REF-MOROZOV-1966].

    """
    started = time.perf_counter()
    if not isinstance(model, DifferentiableForward):
        raise TypeError("model must implement DifferentiableForward")
    n, m = model.input_dimension, model.output_dimension
    y = _vector(data, m, "data")
    if noise is not None:
        if not is_covariance_factor(noise):
            raise TypeError("noise must be a probstat.linalg covariance factor")
        if noise.dimension != m:
            raise ValueError(f"noise has dimension {noise.dimension}, expected {m}")
    if penalty is None:
        operator: LinearOperator = IdentityOperator(n)
    else:
        operator = require_operator(penalty, name="penalty")
        if operator.shape[1] != n:
            raise ValueError(f"penalty has {operator.shape[1]} columns, expected {n}")
    x_ref = (0.0,) * n if reference is None else _vector(reference, n, "reference")
    x0 = (0.0,) * n if initial is None else _vector(initial, n, "initial")
    lam = _real(regularization, "regularization")
    lam2 = lam * lam
    tau = _real(discrepancy_tau, "discrepancy_tau", minimum=1.0, strict=True)
    bound: float | None = None
    if discrepancy_level is not None:
        bound = tau * _real(discrepancy_level, "discrepancy_level", strict=True)
    # [true gradient norm at the start, flagged iterate, its true gradient norm]
    flag: list[Any] = [None, None, math.nan]
    if hessian not in ("gauss_newton", "full"):
        raise ValueError("hessian must be 'gauss_newton' or 'full'")
    if hvp is not None and hvp not in ("callback", "central_difference_of_gradient"):
        raise ValueError("hvp must be 'callback' or 'central_difference_of_gradient'")
    if hvp_callback is not None and not callable(hvp_callback):
        raise TypeError("hvp_callback must be callable")
    fd_step = _DEFAULT_FD_STEP
    if hessian == "gauss_newton":
        if hvp_callback is not None or hvp is not None or hvp_step is not None:
            raise ValueError(
                "hvp_callback, hvp and hvp_step apply only to hessian='full'"
            )
        mode = "none"
    else:
        if hvp == "central_difference_of_gradient":
            if hvp_callback is not None:
                raise ValueError(
                    "give either hvp_callback or hvp='central_difference_of_gradient'"
                )
            mode = "central_difference_of_gradient"
            if hvp_step is not None:
                fd_step = _real(hvp_step, "hvp_step", strict=True)
        else:
            if hvp_callback is None:
                raise ValueError(
                    "hessian='full' needs hvp_callback(x, v, weighted_residual); "
                    "there is no finite-difference fallback, request "
                    "hvp='central_difference_of_gradient' explicitly"
                )
            if hvp_step is not None:
                raise ValueError("hvp_step applies only to the central-difference mode")
            mode = "callback"

    calls = [0, 0, 0]  # value, jvp, vjp
    start_counts = getattr(model, "counts", None)
    cache: dict[Vector, tuple[Vector, Vector, bool]] = {}

    def whiten(vector: Vector) -> Vector:
        """Return ``W v`` (identity without a noise model).

        References:
            [EQ-INV-NTR-001].

        """
        return vector if noise is None else tuple(noise.whiten(vector))

    def whiten_transpose(vector: Vector) -> Vector:
        """Return ``W' v`` (identity without a noise model).

        References:
            [EQ-INV-NTR-001].

        """
        return vector if noise is None else tuple(noise.whiten_transpose(vector))

    def state(x: Vector) -> tuple[Vector, Vector, bool]:
        """Return ``(r, W'W r, finite)`` at ``x``, cached for recent points.

        References:
            [EQ-INV-NTR-001].

        """
        hit = cache.get(x)
        if hit is not None:
            return hit
        calls[0] += 1
        value = _raw_vector(model.value(x), m, "value(x)")
        r = tuple(a - b for a, b in zip(value, y, strict=True))
        finite = all(math.isfinite(v) for v in r)
        wr = whiten_transpose(whiten(r)) if finite else r
        if len(cache) >= _CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[x] = (r, wr, finite)
        return cache[x]

    def prior_offset(x: Vector) -> Vector:
        """Return ``L (x - x_ref)``.

        References:
            [EQ-INV-NTR-001].

        """
        diff = tuple(a - b for a, b in zip(x, x_ref, strict=True))
        return tuple(operator.matvec(diff))

    def parts(x: Vector) -> tuple[float, float, float]:
        """Return ``(Phi, misfit, penalty)`` at ``x`` (``nan`` if not finite).

        References:
            [EQ-INV-NTR-001].

        """
        r, _wr, finite = state(x)
        if not finite:
            return math.nan, math.nan, math.nan
        w = norm(whiten(r))
        off = prior_offset(x)
        mis = 0.5 * w * w
        pen = 0.5 * lam2 * dot(off, off)
        return mis + pen, mis, pen

    def objective(x: Vector) -> float:
        """Return ``Phi(x)`` (``nan`` where the forward value is not finite).

        References:
            [EQ-INV-NTR-001].

        """
        return parts(x)[0]

    def gradient(x: Vector) -> Vector:
        """Return ``J'W'W r + lambda^2 L'L (x - x_ref)`` (one ``vjp``).

        References:
            [EQ-INV-NTR-001].

        """
        _r, wr, finite = state(x)
        if not finite:
            return (math.nan,) * n
        calls[2] += 1
        data_part = _raw_vector(model.vjp(x, wr), n, "vjp")
        prior_part = operator.rmatvec(prior_offset(x))
        return tuple(a + lam2 * b for a, b in zip(data_part, prior_part, strict=True))

    def stopping_gradient(x: Vector) -> Vector:
        """Return the gradient, or zeros where the discrepancy test fires.

        The zero vector makes the driver stop at once; the caller relabels
        that stop and restores the true gradient norm. The test is applied
        only here (accepted iterates), never to the perturbed points of the
        central-difference Hessian action.

        References:
            [EQ-INV-NLX-007].

        """
        g = gradient(x)
        if bound is None:
            return g
        g_norm = norm(g)
        if flag[0] is None:
            flag[0] = g_norm
        r, _wr, finite = state(x)
        if (
            finite
            and g_norm > gradient_tolerance * max(1.0, flag[0])
            and norm(whiten(r)) <= bound
        ):
            flag[1] = x
            flag[2] = g_norm
            return (0.0,) * n
        return g

    def gauss_newton_action(x: Vector, v: Vector) -> Vector:
        """Return ``J'W'W J v + lambda^2 L'L v`` (one ``jvp``, one ``vjp``).

        References:
            [EQ-INV-NTR-002].

        """
        calls[1] += 1
        jv = _raw_vector(model.jvp(x, v), m, "jvp")
        calls[2] += 1
        data_part = _raw_vector(model.vjp(x, whiten_transpose(whiten(jv))), n, "vjp")
        prior_part = operator.rmatvec(tuple(operator.matvec(v)))
        return tuple(a + lam2 * b for a, b in zip(data_part, prior_part, strict=True))

    def full_action(x: Vector, v: Vector) -> Vector:
        """Return the Gauss-Newton action plus the residual-weighted term.

        References:
            [EQ-INV-NTR-003].

        """
        base = gauss_newton_action(x, v)
        assert hvp_callback is not None
        _r, wr, _finite = state(x)
        extra = _raw_vector(hvp_callback(x, v, wr), n, "hvp_callback")
        return tuple(a + b for a, b in zip(base, extra, strict=True))

    def difference_action(x: Vector, v: Vector) -> Vector:
        """Return the central difference of the gradient of ``Phi`` along ``v``.

        References:
            [EQ-INV-NTR-004].

        """
        vnorm = norm(v)
        if vnorm == 0.0:
            return (0.0,) * n
        eps = fd_step * (1.0 + norm(x)) / vnorm
        plus = gradient(tuple(a + eps * b for a, b in zip(x, v, strict=True)))
        minus = gradient(tuple(a - eps * b for a, b in zip(x, v, strict=True)))
        return tuple((p - q) / (2.0 * eps) for p, q in zip(plus, minus, strict=True))

    if mode == "none":
        action: Callable[[Vector, Vector], Vector] = gauss_newton_action
    elif mode == "callback":
        action = full_action
    else:
        action = difference_action

    if not state(x0)[2]:
        raise ValueError("forward map is not finite at the initial point")
    core = trust_region_newton_cg(
        objective,
        stopping_gradient,
        action,
        x0,
        radius=initial_radius,
        max_radius=max_radius,
        eta=eta,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
        inner_tolerance=inner_tolerance,
        forcing=forcing,
        inner_max_iterations=inner_max_iterations,
        minimum_radius=minimum_radius,
    )
    objective_value, misfit, pen = parts(core.x)
    reason = core.stop_reason
    grad_norm = core.gradient_norm
    history = core.history
    if flag[1] is not None and core.x == flag[1] and reason == "gradient":
        reason = "discrepancy"
        grad_norm = float(flag[2])
        history = (*history[:-1], replace(history[-1], gradient_norm=grad_norm))
    r_final = state(core.x)[0]
    derivative_value = 0
    now = getattr(model, "counts", None)
    if now is not None and start_counts is not None:
        derivative_value = int(now.derivative_value - start_counts.derivative_value)
    return NewtonTrustRegionResult(
        estimate=core.x,
        objective=objective_value,
        misfit=misfit,
        penalty=pen,
        whitened_residual_norm=norm(whiten(r_final)),
        gradient_norm=grad_norm,
        iterations=core.iterations,
        attempts=core.attempts,
        converged=reason in ("gradient", "discrepancy"),
        stop_reason=reason,
        history=history,
        counts=NewtonTrustRegionCounts(
            value=calls[0],
            jvp=calls[1],
            vjp=calls[2],
            hvp=core.hessian_actions,
            derivative_value=derivative_value,
        ),
        total_inner_iterations=core.inner_iterations,
        regularization=lam,
        hessian=hessian,
        hvp_mode=mode,
        radius=core.radius,
        wall_time=time.perf_counter() - started,
    )
