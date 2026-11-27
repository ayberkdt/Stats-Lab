"""Latent force models: a linear ODE driven by a Gaussian-process force.

Experimental (added in 0.28).

A latent force model (LFM) is a hybrid of a mechanistic and a data-driven
model. The outputs ``y_d(t)`` obey a known-form linear ordinary differential
equation whose parameters (decay rates, sensitivities, basal levels) are
unknown, and all outputs are driven by one *unknown* time-varying force
``f(t)`` that receives a Gaussian-process prior. Because the ODE is linear the
outputs are jointly Gaussian, so exact regression, the marginal likelihood,
and the posterior of the *force itself* (with uncertainty) follow from the
covariance functions that the ODE induces
[REF-ALVAREZ-LUENGO-LAWRENCE-2013], [REF-RASMUSSEN-WILLIAMS-2006].

Two implementations are provided.

1. **First-order channels, squared-exponential force, closed form.**
   ``dy_d/dt + D_d y_d = B_d + S_d f(t)`` with ``f ~ GP(0, k)``,
   ``k(t, t') = exp(-(t - t')^2 / (2 l^2))`` (unit variance; the amplitude is
   carried by the sensitivities ``S_d``). The output-force and output-output
   covariances are available in closed form through ``erfc`` and exponentials
   [EQ-SUR-LFM-002], [EQ-SUR-LFM-003]; the marginal likelihood has an exact
   gradient (analytic derivative of the closed-form covariance)
   [EQ-SUR-LFM-004] and the force and the outputs are predicted with
   variances [EQ-SUR-LFM-005] (:func:`fit_latent_force_model`,
   :func:`latent_force_model`, :class:`LatentForceModel`). Cost is
   ``O(n^3)`` for ``n`` observations in total.
2. **State-space channels (first or second order), Matern force, exact in
   ``O(n)``.** Each channel is a first-order system or a second-order
   (mass-spring-damper) system [EQ-SUR-LFM-008]; the Matern ``1/2``, ``3/2``
   or ``5/2`` force is realized as a linear SDE [EQ-SUR-LFM-006], so that the
   joint model is a linear Gaussian state-space model and the Kalman filter
   and a fixed-interval smoother give the exact GP marginal likelihood
   and posterior in time linear in the number of time stamps
   [EQ-SUR-LFM-007] (:func:`smooth_latent_force_state_space`,
   :func:`fit_latent_force_state_space`). This is the construction of
   [REF-SARKKA-ALVAREZ-LAWRENCE-2019] (a time-local state-space form of the
   latent force model of [REF-ALVAREZ-LUENGO-LAWRENCE-2013]); the filter is
   implemented here so that the surrogate layer needs no import of the
   filtering package.

Conventions. Time ``t >= 0`` in seconds (any consistent unit), the ODE is
integrated from ``t = 0`` with every output in equilibrium at ``t = 0``
(``y_d(0) = B_d / D_d`` unless ``initial_values`` are given), and the force
prior is stationary on the real line, so only ``f`` on ``[0, t]`` influences
``y(t)``. The force has unit prior variance: the product ``S_d f`` is the
physical input felt by channel ``d`` (for a heat balance, the heat input
divided by the heat capacity), and only these products are identified,
together with a joint sign. No jitter, clipping or pseudo-inverse is ever
applied; a covariance that is not positive definite raises ``ValueError``.
Validity: linear time-invariant ODEs, Gaussian noise, ``D_d l`` of order one
or smaller than ten (a bound enforced by the closed form).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.optimization import minimize_bfgs

from .gp_advanced import _backward, _factor, _forward, _log_det, _precision

_LOG_TWO_PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_SQRT_PI = math.sqrt(math.pi)
_Z95 = 1.959963984540054
_ASYMPTOTIC_START = 10.0
_MAX_DECAY_TIMES_LENGTH = 10.0
_Matrix = list[list[float]]


# ---------------------------------------------------------------------------
# closed-form building blocks (scaled complementary error function)
# ---------------------------------------------------------------------------


def _exp_erfc(exponent: float, x: float) -> float:
    """Return ``exp(exponent) * erfc(x)`` without overflow for large ``x``.

    For ``x >= 10`` the asymptotic series of the scaled function
    ``erfcx(x) = exp(x^2) erfc(x)`` (ten terms, relative error below
    ``1e-13``) is used and the exponents are combined before exponentiation.

    References:
        [EQ-SUR-LFM-002], [EQ-SUR-LFM-003].

    """
    if x < _ASYMPTOTIC_START:
        return math.exp(exponent) * math.erfc(x)
    inv2 = 1.0 / (2.0 * x * x)
    series = 1.0
    term = 1.0
    for k in range(1, 11):
        term *= -(2 * k - 1) * inv2
        series += term
    return math.exp(exponent - x * x) * series / (x * _SQRT_PI)


def _pair_term(exponent: float, a: float, b: float) -> float:
    """Return ``exp(exponent) (erf(a) + erf(b))`` for ``b >= 0``.

    The identity ``erf(a) + erf(b) = erfc(-a) - erfc(b)`` avoids the loss of
    precision of adding ``+1`` and ``-1`` when the pair nearly cancels.

    References:
        [EQ-SUR-LFM-002], [EQ-SUR-LFM-003].

    """
    return _exp_erfc(exponent, -a) - _exp_erfc(exponent, b)


def _force_tilde(t: float, tp: float, d: float, ell: float) -> float:
    """Return ``Cov(y(t), f(tp)) / S`` for a first-order channel.

    ``l sqrt(pi/2) exp(D^2 l^2/2 - D (t - t')) [erf((t - t' - D l^2)/(sqrt2 l))
    + erf((t' + D l^2)/(sqrt2 l))]`` [EQ-SUR-LFM-002].

    References:
        [EQ-SUR-LFM-002].

    """
    l2 = ell * ell
    s = ell * _SQRT2
    exponent = d * d * l2 * 0.5 - d * (t - tp)
    pair = _pair_term(exponent, (t - tp - d * l2) / s, (tp + d * l2) / s)
    return ell * _SQRT_HALF_PI * pair


_Grad = tuple[float, float, float]


def _output_terms(
    t: float, tp: float, d: float, dp: float, ell: float
) -> tuple[_Grad, ...]:
    """Return ``(E, a, b)`` of the four erf pairs of [EQ-SUR-LFM-003].

    The pairs are listed in the order of the formula; the third and fourth
    enter with a minus sign.

    References:
        [EQ-SUR-LFM-003].

    """
    l2 = ell * ell
    s = ell * _SQRT2
    h = l2 * 0.5
    return (
        (dp * dp * h + dp * (t - tp), (tp - t - dp * l2) / s, (t + dp * l2) / s),
        (d * d * h + d * (tp - t), (t - tp - d * l2) / s, (tp + d * l2) / s),
        (dp * dp * h - d * t - dp * tp, (tp - dp * l2) / s, dp * ell / _SQRT2),
        (d * d * h - d * t - dp * tp, (t - d * l2) / s, d * ell / _SQRT2),
    )


def _output_tilde(t: float, tp: float, d: float, dp: float, ell: float) -> float:
    """Return ``Cov(y_d(t), y_d'(tp)) / (S S')`` for two first-order channels.

    The four erf pairs of [EQ-SUR-LFM-003], divided by ``D + D'``.

    References:
        [EQ-SUR-LFM-003].

    """
    t1, t2, t3, t4 = (_pair_term(*term) for term in _output_terms(t, tp, d, dp, ell))
    return (t1 + t2 - t3 - t4) * (ell * _SQRT_HALF_PI / (d + dp))


def _output_tilde_grad(
    t: float, tp: float, d: float, dp: float, ell: float
) -> tuple[float, float, float, float]:
    """Return the covariance of [EQ-SUR-LFM-003] and its three partials.

    The partials are with respect to ``D``, ``D'`` and ``l``. For a pair
    ``e^E (erfc(-a) - erfc(b))`` the derivative is
    ``value dE + (2/sqrt(pi)) (e^(E - a^2) da + e^(E - b^2) db)``; the
    arguments ``E``, ``a``, ``b`` are elementary functions of ``(D, D', l)``
    with the derivatives listed below [EQ-SUR-LFM-004].

    References:
        [EQ-SUR-LFM-004].

    """
    l2 = ell * ell
    r2 = _SQRT2
    q = ell / r2
    pre = ell * _SQRT_HALF_PI / (d + dp)
    terms = _output_terms(t, tp, d, dp, ell)
    # derivatives of (E, a, b) with respect to (D, D', l), term by term
    dexp: tuple[_Grad, ...] = (
        (0.0, dp * l2 + (t - tp), dp * dp * ell),
        (d * l2 + (tp - t), 0.0, d * d * ell),
        (-t, dp * l2 - tp, dp * dp * ell),
        (d * l2 - t, -tp, d * d * ell),
    )
    darg: tuple[_Grad, ...] = (
        (0.0, -q, -(tp - t) / (r2 * l2) - dp / r2),
        (-q, 0.0, -(t - tp) / (r2 * l2) - d / r2),
        (0.0, -q, -tp / (r2 * l2) - dp / r2),
        (-q, 0.0, -t / (r2 * l2) - d / r2),
    )
    dbrg: tuple[_Grad, ...] = (
        (0.0, q, -t / (r2 * l2) + dp / r2),
        (q, 0.0, -tp / (r2 * l2) + d / r2),
        (0.0, q, dp / r2),
        (q, 0.0, d / r2),
    )
    total = 0.0
    grad = [0.0, 0.0, 0.0]
    for index, (exponent, a, b) in enumerate(terms):
        value = _exp_erfc(exponent, -a) - _exp_erfc(exponent, b)
        gauss_a = math.exp(min(exponent - a * a, 700.0))
        gauss_b = math.exp(min(exponent - b * b, 700.0))
        sign = 1.0 if index < 2 else -1.0
        total += sign * value
        for k in range(3):
            grad[k] += sign * (
                value * dexp[index][k]
                + _TWO_OVER_SQRT_PI
                * (gauss_a * darg[index][k] + gauss_b * dbrg[index][k])
            )
    dpre = (-pre / (d + dp), -pre / (d + dp), pre / ell)
    return (
        pre * total,
        dpre[0] * total + pre * grad[0],
        dpre[1] * total + pre * grad[1],
        dpre[2] * total + pre * grad[2],
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _real(value: object, name: str) -> float:
    """Return a finite real number.

    References:
        [REF-PYTHON-DATAMODEL].

    """
    return require_finite_real(value, name=name)


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real number.

    References:
        [REF-PYTHON-DATAMODEL].

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_time(value: object, name: str) -> float:
    """Return a finite time ``t >= 0``.

    References:
        [EQ-SUR-LFM-001].

    """
    number = require_finite_real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative (the ODE starts at t = 0)")
    return number


def _check_scales(decays: Sequence[float], ell: float) -> None:
    """Reject decay-times-length-scale products outside the closed-form range.

    References:
        [EQ-SUR-LFM-003].

    """
    for d in decays:
        if d * ell > _MAX_DECAY_TIMES_LENGTH:
            raise ValueError(
                "decay * lengthscale must not exceed "
                f"{_MAX_DECAY_TIMES_LENGTH:g} (closed form loses precision)"
            )


def _check_series(
    times: Sequence[Sequence[float]], values: Sequence[Sequence[float]]
) -> tuple[tuple[Vector, ...], tuple[Vector, ...]]:
    """Validate per-channel observation times and values.

    References:
        [EQ-SUR-LFM-001].

    """
    if isinstance(times, (str, bytes)) or isinstance(values, (str, bytes)):
        raise TypeError("times and values must be sequences of sequences")
    if len(times) != len(values):
        raise ValueError("times and values must have one entry per channel")
    if not times:
        raise ValueError("at least one channel is required")
    out_t: list[Vector] = []
    out_y: list[Vector] = []
    for c, (t_row, y_row) in enumerate(zip(times, values, strict=True)):
        if len(t_row) != len(y_row):
            raise ValueError(f"channel {c}: times and values differ in length")
        if len(t_row) < 1:
            raise ValueError(f"channel {c}: at least one observation is required")
        out_t.append(tuple(_nonnegative_time(v, f"times[{c}]") for v in t_row))
        out_y.append(tuple(_real(v, f"values[{c}]") for v in y_row))
    return tuple(out_t), tuple(out_y)


def _per_channel(values: Sequence[float], count: int, name: str) -> Vector:
    """Validate one finite real per channel.

    References:
        [EQ-SUR-LFM-001].

    """
    if len(values) != count:
        raise ValueError(f"{name} must have one entry per channel ({count})")
    return tuple(_real(v, name) for v in values)


def _per_channel_positive(values: Sequence[float], count: int, name: str) -> Vector:
    """Validate one positive finite real per channel.

    References:
        [EQ-SUR-LFM-001].

    """
    if len(values) != count:
        raise ValueError(f"{name} must have one entry per channel ({count})")
    return tuple(_positive(v, name) for v in values)


# ---------------------------------------------------------------------------
# public closed-form building blocks
# ---------------------------------------------------------------------------


def lfm_mean(
    t: float, decay: float, basal: float, initial: float | None = None
) -> float:
    """Return the deterministic part of a first-order channel.

    For ``dy/dt + D y = B + S f`` with ``E f = 0``, the mean is
    ``m(t) = B/D + (y(0) - B/D) exp(-D t)``; with ``initial=None`` the
    channel starts in equilibrium, ``y(0) = B/D``, and ``m(t) = B/D``
    [EQ-SUR-LFM-001].

    Args:
        t: Time ``t >= 0``.
        decay: Decay rate ``D > 0`` (1/time).
        basal: Basal input ``B`` (output units per time).
        initial: Initial output ``y(0)`` or ``None`` for equilibrium.

    Returns:
        The mean ``m(t)`` in output units.

    Raises:
        ValueError: If an argument is non-finite, ``t < 0`` or ``D <= 0``.

    Equation:
        [EQ-SUR-LFM-001].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

    """
    time = _nonnegative_time(t, "t")
    rate = _positive(decay, "decay")
    mu = _real(basal, "basal") / rate
    if initial is None:
        return mu
    return mu + (_real(initial, "initial") - mu) * math.exp(-rate * time)


def lfm_force_covariance(
    t: float, t_force: float, decay: float, sensitivity: float, lengthscale: float
) -> float:
    """Return ``Cov(y(t), f(t'))`` for a first-order channel.

    With ``y(t) = S int_0^t exp(-D (t - tau)) f(tau) dtau`` and the unit
    squared-exponential force, completing the square in the exponent gives

    ``S l sqrt(pi/2) exp(D^2 l^2 / 2 - D (t - t'))
    [erf((t - t' - D l^2) / (sqrt(2) l)) + erf((t' + D l^2) / (sqrt(2) l))]``.

    It is zero at ``t = 0`` and tends to ``(S / D) k(t - t')`` for large
    ``D`` [EQ-SUR-LFM-002].

    Args:
        t: Output time ``t >= 0``.
        t_force: Force time ``t' >= 0``.
        decay: Decay ``D > 0``.
        sensitivity: Sensitivity ``S`` (any finite real).
        lengthscale: Force length-scale ``l > 0`` with ``D l <= 10``.

    Returns:
        The covariance in output units times force units.

    Raises:
        ValueError: If an argument is invalid or ``D l > 10``.

    Equation:
        [EQ-SUR-LFM-002].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

    """
    rate = _positive(decay, "decay")
    ell = _positive(lengthscale, "lengthscale")
    _check_scales((rate,), ell)
    value = _force_tilde(
        _nonnegative_time(t, "t"), _nonnegative_time(t_force, "t_force"), rate, ell
    )
    return _real(sensitivity, "sensitivity") * value


def lfm_output_covariance(
    t: float,
    t_other: float,
    decay: float,
    decay_other: float,
    sensitivity: float,
    sensitivity_other: float,
    lengthscale: float,
) -> float:
    """Return ``Cov(y_d(t), y_d'(t'))`` for two first-order channels.

    ``S S' int_0^t int_0^t' exp(-D (t - tau)) exp(-D' (t' - tau'))
    k(tau - tau') dtau' dtau``. Writing ``K = S int_0^t exp(-D (t - tau))
    c'(t', tau) dtau`` with the output-force covariance ``c'`` of
    [EQ-SUR-LFM-002] and integrating the exponential-times-erf terms
    (``int e^(a x) erf(b x + c) dx`` is elementary) gives four erf pairs,

    ``S S' l sqrt(pi/2) / (D + D') * {`` ``e^(D'^2 l^2/2 + D'(t - t'))
    [erf((t' - t - D' l^2)/s) + erf((t + D' l^2)/s)]``
    ``+ e^(D^2 l^2/2 + D(t' - t)) [erf((t - t' - D l^2)/s)
    + erf((t' + D l^2)/s)]``
    ``- e^(D'^2 l^2/2 - D t - D' t') [erf((t' - D' l^2)/s) + erf(D' l/sqrt2)]``
    ``- e^(D^2 l^2/2 - D t - D' t') [erf((t - D l^2)/s) + erf(D l/sqrt2)] }``

    with ``s = sqrt(2) l``. The expression is symmetric under exchanging
    ``(t, D, S)`` with ``(t', D', S')``, vanishes at ``t = 0`` or ``t' = 0``
    and tends to ``S S' k(t - t') / (D D')`` for large decays
    [EQ-SUR-LFM-003].

    Args:
        t: Time of the first output, ``t >= 0``.
        t_other: Time of the second output, ``t' >= 0``.
        decay: Decay ``D > 0`` of the first channel.
        decay_other: Decay ``D' > 0`` of the second channel.
        sensitivity: Sensitivity ``S`` of the first channel.
        sensitivity_other: Sensitivity ``S'`` of the second channel.
        lengthscale: Force length-scale ``l > 0`` with ``D l <= 10``.

    Returns:
        The covariance in squared output units.

    Raises:
        ValueError: If an argument is invalid or ``D l > 10``.

    Equation:
        [EQ-SUR-LFM-003].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

    """
    d1 = _positive(decay, "decay")
    d2 = _positive(decay_other, "decay_other")
    ell = _positive(lengthscale, "lengthscale")
    _check_scales((d1, d2), ell)
    value = _output_tilde(
        _nonnegative_time(t, "t"),
        _nonnegative_time(t_other, "t_other"),
        d1,
        d2,
        ell,
    )
    return (
        _real(sensitivity, "sensitivity")
        * _real(sensitivity_other, "sensitivity_other")
        * value
    )


# ---------------------------------------------------------------------------
# joint covariance, likelihood and gradient
# ---------------------------------------------------------------------------


def _joint_covariance(
    chan_t: Sequence[Vector],
    decays: Vector,
    sens: Vector,
    ell: float,
    want_grad: bool,
) -> tuple[_Matrix, list[_Matrix]]:
    """Assemble the latent covariance and its derivatives.

    The derivative list is ordered ``[log D_d (P), S_d (P), log l]``.

    References:
        [EQ-SUR-LFM-003], [EQ-SUR-LFM-004].

    """
    channels = len(chan_t)
    offsets = [0]
    for row in chan_t:
        offsets.append(offsets[-1] + len(row))
    n = offsets[-1]
    cov = [[0.0] * n for _ in range(n)]
    grads: list[_Matrix] = (
        [[[0.0] * n for _ in range(n)] for _ in range(2 * channels + 1)]
        if want_grad
        else []
    )
    try:
        for c in range(channels):
            for e in range(c, channels):
                scale = sens[c] * sens[e]
                for i, t in enumerate(chan_t[c]):
                    a = offsets[c] + i
                    for j in range(i if c == e else 0, len(chan_t[e])):
                        b = offsets[e] + j
                        tp = chan_t[e][j]
                        if not want_grad:
                            val = _output_tilde(t, tp, decays[c], decays[e], ell)
                            cov[a][b] = cov[b][a] = scale * val
                            continue
                        v, g0, g1, g2 = _output_tilde_grad(
                            t, tp, decays[c], decays[e], ell
                        )
                        cov[a][b] = cov[b][a] = scale * v
                        entries: tuple[tuple[int, float], ...]
                        if c == e:
                            entries = (
                                (c, scale * (g0 + g1) * decays[c]),
                                (channels + c, 2.0 * sens[c] * v),
                            )
                        else:
                            entries = (
                                (c, scale * g0 * decays[c]),
                                (e, scale * g1 * decays[e]),
                                (channels + c, sens[e] * v),
                                (channels + e, sens[c] * v),
                            )
                        for slot, derivative in entries:
                            grads[slot][a][b] = grads[slot][b][a] = derivative
                        grads[2 * channels][a][b] = grads[2 * channels][b][a] = (
                            scale * g2 * ell
                        )
    except OverflowError as error:
        raise ValueError("covariance overflow: parameters outside range") from error
    return cov, grads


def _mean_and_gradient(
    chan_t: Sequence[Vector],
    decays: Vector,
    basals: Vector,
    initial: Vector | None,
) -> tuple[Vector, list[list[float]], list[list[float]]]:
    """Return the stacked mean and its derivatives w.r.t. log D and B.

    References:
        [EQ-SUR-LFM-001].

    """
    mean: list[float] = []
    d_logd: list[list[float]] = []
    d_basal: list[list[float]] = []
    count = len(chan_t)
    n = sum(len(r) for r in chan_t)
    d_logd = [[0.0] * n for _ in range(count)]
    d_basal = [[0.0] * n for _ in range(count)]
    pos = 0
    for c, row in enumerate(chan_t):
        d, b = decays[c], basals[c]
        for t in row:
            decay = math.exp(-d * t)
            if initial is None:
                mean.append(b / d)
                d_basal[c][pos] = 1.0 / d
                d_logd[c][pos] = -b / d
            else:
                y0 = initial[c]
                mean.append(b / d * (1.0 - decay) + y0 * decay)
                d_basal[c][pos] = (1.0 - decay) / d
                d_logd[c][pos] = d * (
                    -b / (d * d) * (1.0 - decay) + b / d * t * decay - y0 * t * decay
                )
            pos += 1
    return tuple(mean), d_logd, d_basal


@dataclass(frozen=True, slots=True)
class LatentForceLikelihood:
    """Marginal likelihood of a first-order latent force model and its gradient.

    Args:
        log_likelihood: ``log p(y | theta)`` [EQ-SUR-LFM-004].
        gradient: Gradient of ``log_likelihood`` with respect to
            ``parameter_names`` (empty if not requested).
        parameter_names: ``log_decay[c]``, ``sensitivity[c]``, ``basal[c]``
            for every channel ``c``, then ``log_lengthscale``, then
            ``log_noise_variance[c]`` for every channel.
        observations: Number of observations ``n``.

    References:
        [EQ-SUR-LFM-004].

    """

    log_likelihood: float
    gradient: Vector
    parameter_names: tuple[str, ...]
    observations: int


def _parameter_names(count: int) -> tuple[str, ...]:
    """Return the ordered parameter names for ``count`` channels.

    References:
        [EQ-SUR-LFM-004].

    """
    return (
        tuple(f"log_decay[{c}]" for c in range(count))
        + tuple(f"sensitivity[{c}]" for c in range(count))
        + tuple(f"basal[{c}]" for c in range(count))
        + ("log_lengthscale",)
        + tuple(f"log_noise_variance[{c}]" for c in range(count))
    )


def _evaluate(
    chan_t: Sequence[Vector],
    chan_y: Sequence[Vector],
    decays: Vector,
    sens: Vector,
    basals: Vector,
    ell: float,
    noise: Vector,
    initial: Vector | None,
    want_grad: bool,
) -> tuple[float, Vector]:
    """Evaluate the log marginal likelihood and optionally its gradient.

    ``log p = -r' K^-1 r / 2 - log|K| / 2 - n log(2 pi) / 2`` and
    ``d log p / d theta = alpha' (dK) alpha / 2 - tr(K^-1 dK) / 2
    + (dm)' alpha`` with ``alpha = K^-1 r`` [EQ-SUR-LFM-004].

    References:
        [EQ-SUR-LFM-004], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    count = len(chan_t)
    _check_scales(decays, ell)
    cov, dcov = _joint_covariance(chan_t, decays, sens, ell, want_grad)
    n = len(cov)
    labels = [c for c, row in enumerate(chan_t) for _ in row]
    for i in range(n):
        cov[i][i] += noise[labels[i]]
    mean, d_logd, d_basal = _mean_and_gradient(chan_t, decays, basals, initial)
    resid = [y - m for y, m in zip((v for r in chan_y for v in r), mean, strict=True)]
    lower = _factor(cov, "latent force covariance")
    alpha = _backward(lower, _forward(lower, resid))
    quad = sum(r * a for r, a in zip(resid, alpha, strict=True))
    value = -0.5 * (quad + _log_det(lower) + n * _LOG_TWO_PI)
    if not want_grad:
        return value, ()
    precision = _precision(lower)

    def trace_term(matrix: _Matrix) -> float:
        """Return ``alpha' dK alpha / 2 - tr(K^-1 dK) / 2``.

        References:
            [EQ-SUR-LFM-004].

        """
        total = 0.0
        for a in range(n):
            row, prow, aa = matrix[a], precision[a], alpha[a]
            total += sum(
                row[b] * (aa * alpha[b] - prow[b]) for b in range(n) if row[b] != 0.0
            )
        return 0.5 * total

    grad: list[float] = []
    for c in range(count):
        grad.append(
            trace_term(dcov[c])
            + sum(m * a for m, a in zip(d_logd[c], alpha, strict=True))
        )
    for c in range(count):
        grad.append(trace_term(dcov[count + c]))
    for c in range(count):
        grad.append(sum(m * a for m, a in zip(d_basal[c], alpha, strict=True)))
    grad.append(trace_term(dcov[2 * count]))
    for c in range(count):
        total = 0.0
        for i in range(n):
            if labels[i] == c:
                total += alpha[i] * alpha[i] - precision[i][i]
        grad.append(0.5 * noise[c] * total)
    return value, tuple(grad)


def latent_force_log_likelihood(
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    decays: Sequence[float],
    sensitivities: Sequence[float],
    basals: Sequence[float],
    lengthscale: float,
    noise_variances: Sequence[float],
    *,
    initial_values: Sequence[float] | None = None,
    gradient: bool = True,
) -> LatentForceLikelihood:
    """Return the log marginal likelihood of first-order channels and its gradient.

    The stacked observations are Gaussian,
    ``y ~ N(m, K + diag(sigma_c^2))`` with the mean of
    :func:`lfm_mean` and the covariance of :func:`lfm_output_covariance`.
    The gradient with respect to every log decay, sensitivity, basal input,
    log length-scale and log noise variance is exact: the closed-form
    covariance is differentiated analytically (the erf pairs with respect to
    the decays and the length-scale; the sensitivities and the noise
    trivially) and contracted with the trace formula
    ``alpha' dK alpha / 2 - tr(K^-1 dK) / 2`` [EQ-SUR-LFM-004].

    Args:
        times: Per channel, observation times (``>= 0``; channels may differ).
        values: Per channel, observed outputs.
        decays: Decays ``D_c > 0``.
        sensitivities: Sensitivities ``S_c``.
        basals: Basal inputs ``B_c``.
        lengthscale: Force length-scale ``l > 0``.
        noise_variances: Observation noise variances ``sigma_c^2 > 0``.
        initial_values: Known initial outputs ``y_c(0)``; ``None`` starts every
            channel in equilibrium (``B_c / D_c``).
        gradient: Whether to compute the gradient.

    Returns:
        A :class:`LatentForceLikelihood`.

    Raises:
        ValueError: If an input is invalid or the covariance is not positive
            definite (no jitter is added).

    Equation:
        [EQ-SUR-LFM-004].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    chan_t, chan_y = _check_series(times, values)
    count = len(chan_t)
    d = _per_channel_positive(decays, count, "decays")
    s = _per_channel(sensitivities, count, "sensitivities")
    b = _per_channel(basals, count, "basals")
    ell = _positive(lengthscale, "lengthscale")
    noise = _per_channel_positive(noise_variances, count, "noise_variances")
    init = (
        None
        if initial_values is None
        else _per_channel(initial_values, count, "initial_values")
    )
    value, grad = _evaluate(chan_t, chan_y, d, s, b, ell, noise, init, gradient)
    return LatentForceLikelihood(
        value, grad, _parameter_names(count), sum(len(r) for r in chan_t)
    )


# ---------------------------------------------------------------------------
# posterior model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentForcePrediction:
    """Gaussian posterior marginals at a set of times.

    Args:
        times: Prediction times.
        mean: Posterior means.
        variance: Posterior variances (of the latent quantity, without
            observation noise unless the prediction asked for it).

    References:
        [EQ-SUR-LFM-005].

    """

    times: Vector
    mean: Vector
    variance: Vector

    @property
    def std(self) -> Vector:
        """Return the posterior standard deviations.

        References:
            [EQ-SUR-LFM-005].

        """
        return tuple(math.sqrt(max(v, 0.0)) for v in self.variance)

    @property
    def lower(self) -> Vector:
        """Return the lower end of the pointwise central 95% interval.

        References:
            [EQ-SUR-LFM-005].

        """
        return tuple(m - _Z95 * s for m, s in zip(self.mean, self.std, strict=True))

    @property
    def upper(self) -> Vector:
        """Return the upper end of the pointwise central 95% interval.

        References:
            [EQ-SUR-LFM-005].

        """
        return tuple(m + _Z95 * s for m, s in zip(self.mean, self.std, strict=True))


@dataclass(frozen=True, slots=True)
class LatentForceModel:
    """A first-order latent force model conditioned on data.

    Posterior of the force and of the outputs given noisy observations
    [EQ-SUR-LFM-005]; build one with :func:`fit_latent_force_model` or
    :func:`latent_force_model`.

    Args:
        times: Training times per channel.
        values: Training outputs per channel.
        decays: Decays ``D_c``.
        sensitivities: Sensitivities ``S_c``.
        basals: Basal inputs ``B_c``.
        lengthscale: Force length-scale ``l``.
        noise_variances: Noise variances ``sigma_c^2``.
        initial_values: Known initial outputs or ``None`` (equilibrium).
        log_marginal_likelihood: Log evidence at these parameters.
        gradient_norm: Euclidean norm of the evidence gradient with respect
            to the free parameters at the solution (``nan`` if not fitted).
        converged: Whether the optimizer met its tolerance (``True`` for
            fixed parameters).
        stop_reason: Optimizer message.
        evaluations: Likelihood evaluations used by the fit.

    References:
        [EQ-SUR-LFM-005].

    """

    times: tuple[Vector, ...]
    values: tuple[Vector, ...]
    decays: Vector
    sensitivities: Vector
    basals: Vector
    lengthscale: float
    noise_variances: Vector
    initial_values: Vector | None
    log_marginal_likelihood: float
    gradient_norm: float
    converged: bool
    stop_reason: str
    evaluations: int
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _alpha: list[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize the training covariance and cache ``K^-1 (y - m)``.

        Raises:
            ValueError: If the covariance is not positive definite.

        Equation:
            [EQ-SUR-LFM-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006].

        """
        cov, _ = _joint_covariance(
            self.times, self.decays, self.sensitivities, self.lengthscale, False
        )
        labels = [c for c, row in enumerate(self.times) for _ in row]
        for i, label in enumerate(labels):
            cov[i][i] += self.noise_variances[label]
        mean, _, _ = _mean_and_gradient(
            self.times, self.decays, self.basals, self.initial_values
        )
        resid = [
            y - m
            for y, m in zip((v for r in self.values for v in r), mean, strict=True)
        ]
        lower = _factor(cov, "latent force covariance")
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_alpha", _backward(lower, _forward(lower, resid)))

    def _train(self) -> list[tuple[int, float]]:
        """Return ``(channel, time)`` of every training observation.

        References:
            [EQ-SUR-LFM-005].

        """
        return [(c, t) for c, row in enumerate(self.times) for t in row]

    def predict_outputs(
        self, channel: int, times: Sequence[float], *, include_noise: bool = False
    ) -> LatentForcePrediction:
        """Predict one output channel.

        ``mu = m_c(t) + k' K^-1 (y - m)`` and
        ``var = K(t, t) - |L^-1 k|^2`` (plus ``sigma_c^2`` if requested), with
        ``k_j = Cov(y_c(t), y_{c_j}(t_j))`` from [EQ-SUR-LFM-003]
        [EQ-SUR-LFM-005].

        Args:
            channel: Channel index.
            times: Times ``>= 0``.
            include_noise: Add the observation noise variance.

        Returns:
            Posterior mean and variance of ``y_c`` (or of a new noisy
            observation).

        Raises:
            ValueError: If the channel or a time is invalid.

        Equation:
            [EQ-SUR-LFM-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006].

        """
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("channel must be an integer")
        if not 0 <= channel < len(self.times):
            raise ValueError("channel index out of range")
        d, s = self.decays[channel], self.sensitivities[channel]
        train = self._train()
        means: list[float] = []
        variances: list[float] = []
        grid = tuple(_nonnegative_time(t, "times") for t in times)
        for t in grid:
            cross = []
            for c, tj in train:
                val = _output_tilde(t, tj, d, self.decays[c], self.lengthscale)
                cross.append(s * self.sensitivities[c] * val)
            prior = _output_tilde(t, t, d, d, self.lengthscale)
            whitened = _forward(self._lower, cross)
            var = s * s * prior - sum(w * w for w in whitened)
            if include_noise:
                var += self.noise_variances[channel]
            init = None if self.initial_values is None else self.initial_values[channel]
            base = lfm_mean(t, d, self.basals[channel], init)
            means.append(
                base + sum(c_ * a for c_, a in zip(cross, self._alpha, strict=True))
            )
            variances.append(var)
        return LatentForcePrediction(grid, tuple(means), tuple(variances))

    def predict_force(
        self, times: Sequence[float], *, channel: int | None = None
    ) -> LatentForcePrediction:
        """Infer the latent force with uncertainty.

        ``E f(t) = c' K^-1 (y - m)`` and ``Var f(t) = 1 - |L^-1 c|^2`` with
        ``c_j = Cov(y_{c_j}(t_j), f(t))`` from [EQ-SUR-LFM-002]
        [EQ-SUR-LFM-005]. Far from the data the posterior reverts to the
        prior ``N(0, 1)``: the band is honest outside the observed window and
        only the force *before* the last observation is informed.

        Args:
            times: Times ``>= 0``.
            channel: If given, return the force felt by that channel,
                ``S_c f(t)`` (mean scaled by ``S_c``, variance by ``S_c^2``);
                otherwise the unit-variance ``f``.

        Returns:
            Posterior mean and variance.

        Raises:
            ValueError: If a time or the channel is invalid.

        Equation:
            [EQ-SUR-LFM-005].

        References:
            [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

        """
        scale = 1.0
        if channel is not None:
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise TypeError("channel must be an integer")
            if not 0 <= channel < len(self.times):
                raise ValueError("channel index out of range")
            scale = self.sensitivities[channel]
        train = self._train()
        grid = tuple(_nonnegative_time(t, "times") for t in times)
        means: list[float] = []
        variances: list[float] = []
        for t in grid:
            cross = []
            for c, tj in train:
                val = _force_tilde(tj, t, self.decays[c], self.lengthscale)
                cross.append(self.sensitivities[c] * val)
            whitened = _forward(self._lower, cross)
            means.append(
                scale * sum(c_ * a for c_, a in zip(cross, self._alpha, strict=True))
            )
            variances.append(scale * scale * (1.0 - sum(w * w for w in whitened)))
        return LatentForcePrediction(grid, tuple(means), tuple(variances))


def latent_force_model(
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    decays: Sequence[float],
    sensitivities: Sequence[float],
    basals: Sequence[float],
    lengthscale: float,
    noise_variances: Sequence[float],
    *,
    initial_values: Sequence[float] | None = None,
) -> LatentForceModel:
    """Condition a first-order latent force model with known parameters.

    Args:
        times: Per channel, observation times (``>= 0``).
        values: Per channel, observed outputs.
        decays: Decays ``D_c > 0``.
        sensitivities: Sensitivities ``S_c``.
        basals: Basal inputs ``B_c``.
        lengthscale: Force length-scale ``l > 0``.
        noise_variances: Noise variances ``sigma_c^2 > 0``.
        initial_values: Known initial outputs or ``None`` (equilibrium).

    Returns:
        The conditioned :class:`LatentForceModel` (``converged=True``,
        ``stop_reason="fixed parameters"``).

    Raises:
        ValueError: If an input is invalid or the covariance is not positive
            definite.

    Equation:
        [EQ-SUR-LFM-005].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

    """
    chan_t, chan_y = _check_series(times, values)
    count = len(chan_t)
    d = _per_channel_positive(decays, count, "decays")
    s = _per_channel(sensitivities, count, "sensitivities")
    b = _per_channel(basals, count, "basals")
    ell = _positive(lengthscale, "lengthscale")
    noise = _per_channel_positive(noise_variances, count, "noise_variances")
    init = (
        None
        if initial_values is None
        else _per_channel(initial_values, count, "initial_values")
    )
    _check_scales(d, ell)
    value, _ = _evaluate(chan_t, chan_y, d, s, b, ell, noise, init, False)
    return LatentForceModel(
        chan_t,
        chan_y,
        d,
        s,
        b,
        ell,
        noise,
        init,
        value,
        math.nan,
        True,
        "fixed parameters",
        0,
    )


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def _default_start(
    chan_t: Sequence[Vector], chan_y: Sequence[Vector]
) -> tuple[Vector, Vector, Vector, float, Vector]:
    """Return heuristic starting parameters from the data scale.

    ``l = span / 6``, ``D = 1 / l``, the equilibrium level ``B / D`` equal to
    the channel mean, the channel standard deviation (``S = std D`` is formed
    by the caller from the decay actually used) and noise ``1%`` of the output
    variance.

    References:
        [EQ-SUR-LFM-004].

    """
    span = max(max(r) for r in chan_t)
    if span <= 0.0:
        raise ValueError("observation times must span a positive interval")
    ell = span / 6.0
    rate = 1.0 / ell
    decays, stds, means, noise = [], [], [], []
    for row in chan_y:
        mean = math.fsum(row) / len(row)
        var = math.fsum((v - mean) ** 2 for v in row) / max(len(row) - 1, 1)
        std = math.sqrt(var) if var > 0.0 else 1.0
        decays.append(rate)
        stds.append(std)
        means.append(mean)
        noise.append(max(0.01 * var, 1e-8))
    return tuple(decays), tuple(stds), tuple(means), ell, tuple(noise)


def fit_latent_force_model(
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    initial_decays: Sequence[float] | None = None,
    initial_sensitivities: Sequence[float] | None = None,
    initial_basals: Sequence[float] | None = None,
    initial_lengthscale: float | None = None,
    initial_noise_variances: Sequence[float] | None = None,
    initial_values: Sequence[float] | None = None,
    fixed: Sequence[str] = (),
    restarts: int = 1,
    seed: int | RandomStream | None = None,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-4,
) -> LatentForceModel:
    """Fit a first-order latent force model by maximizing the log evidence.

    The free parameters are the log decays, the sensitivities, the basal
    inputs, the log length-scale and the log noise variances (any group can
    be held fixed through ``fixed``); all outputs share the one latent force.
    Internally the optimizer uses the equilibrium level ``mu_c = B_c / D_c``
    instead of ``B_c`` (nearly
    orthogonal to the decay, which conditions the problem far better); the
    gradient is converted by the chain rule. The objective is the exact
    log marginal likelihood [EQ-SUR-LFM-004] and its exact gradient, minimized
    with BFGS (:func:`probstat.optimization.minimize_bfgs`). Trial points at
    which the closed form is out of range or the covariance is not positive
    definite make the line search shorten its step.

    Identifiability. ``S_d`` and ``f`` are identified only through the
    products ``S_d f(t)`` (the force has unit prior variance) and a global
    sign (flip ``S`` and ``f``); all starts here have positive ``S`` so the
    sign is fixed by the start. With a single channel, decay and
    length-scale can trade off; several channels sharing the force identify
    them. Evidence maxima can be local: use ``restarts > 1`` (extra starts
    perturb the log parameters by ``N(0, 0.5^2)`` from ``seed``) and inspect
    ``converged``.

    Args:
        times: Per channel, observation times (``>= 0``).
        values: Per channel, observed outputs.
        initial_decays: Start of ``D_c`` (default ``6 / span``).
        initial_sensitivities: Start of ``S_c`` (default ``std_c D_c``).
        initial_basals: Start of ``B_c`` (default ``mean_c D``).
        initial_lengthscale: Start of ``l`` (default ``span / 6``).
        initial_noise_variances: Start (or fixed value) of ``sigma_c^2``
            (default ``1%`` of the sample variance).
        initial_values: Known initial outputs; ``None`` means equilibrium.
        fixed: Parameter groups held at their initial values (any of
            ``"decays"``, ``"sensitivities"``, ``"basals"``,
            ``"lengthscale"``, ``"noise_variances"``); e.g. known thermal
            time constants and a calibrated gain make the force the only
            unknown.
        restarts: Number of optimizer starts (``>= 1``).
        seed: Seed of the restart perturbations.
        max_iterations: BFGS iteration limit per start.
        gradient_tolerance: Absolute gradient-norm tolerance (log evidence).

    Returns:
        The best :class:`LatentForceModel`; ``converged`` and ``stop_reason``
        report the optimizer outcome of the best start.

    Raises:
        TypeError: If ``restarts`` or ``max_iterations`` is not an integer.
        ValueError: If an input is invalid or no start gives a positive
            definite covariance.

    Equation:
        [EQ-SUR-LFM-004].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013], [REF-NOCEDAL-WRIGHT-2006].

    """
    chan_t, chan_y = _check_series(times, values)
    count = len(chan_t)
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise TypeError("restarts must be a positive integer")
    d0, std0, mean0, l0, n0 = _default_start(chan_t, chan_y)
    if initial_decays is not None:
        d0 = _per_channel_positive(initial_decays, count, "initial_decays")
    if initial_sensitivities is not None:
        s0 = _per_channel(initial_sensitivities, count, "initial_sensitivities")
    else:
        s0 = tuple(sd * r for sd, r in zip(std0, d0, strict=True))
    if initial_basals is not None:
        b0 = _per_channel(initial_basals, count, "initial_basals")
    else:
        b0 = tuple(m * r for m, r in zip(mean0, d0, strict=True))
    if initial_lengthscale is not None:
        l0 = _positive(initial_lengthscale, "initial_lengthscale")
    if initial_noise_variances is not None:
        n0 = _per_channel_positive(
            initial_noise_variances, count, "initial_noise_variances"
        )
    init = (
        None
        if initial_values is None
        else _per_channel(initial_values, count, "initial_values")
    )
    groups = {"decays", "sensitivities", "basals", "lengthscale", "noise_variances"}
    held = set(fixed)
    if not held <= groups:
        raise ValueError(
            f"unknown parameter group in fixed; choose from {sorted(groups)}"
        )
    # full vector layout: log D (P), S (P), mu = B/D (P), log l, log noise (P)
    full_start = (
        tuple(math.log(v) for v in d0)
        + s0
        + tuple(b / r for b, r in zip(b0, d0, strict=True))
        + (math.log(l0),)
        + tuple(math.log(v) for v in n0)
    )
    spans = {
        "decays": range(0, count),
        "sensitivities": range(count, 2 * count),
        "basals": range(2 * count, 3 * count),
        "lengthscale": range(3 * count, 3 * count + 1),
        "noise_variances": range(3 * count + 1, 4 * count + 1),
    }
    free_index = [i for name, span in spans.items() if name not in held for i in span]
    free_index.sort()
    if not free_index:
        raise ValueError("every parameter group is fixed; nothing to fit")

    def expand(reduced: Vector) -> Vector:
        """Insert the free coordinates into the fixed full vector.

        References:
            [EQ-SUR-LFM-004].

        """
        full = list(full_start)
        for slot, value in zip(free_index, reduced, strict=True):
            full[slot] = value
        return tuple(full)

    def unpack(theta: Vector) -> tuple[Vector, Vector, Vector, float, Vector]:
        """Split the full vector into model parameters.

        References:
            [EQ-SUR-LFM-004].

        """
        d = tuple(math.exp(v) for v in theta[:count])
        s = tuple(theta[count : 2 * count])
        b = tuple(m * r for m, r in zip(theta[2 * count : 3 * count], d, strict=True))
        ell = math.exp(theta[3 * count])
        noise = tuple(math.exp(v) for v in theta[3 * count + 1 :])
        return d, s, b, ell, noise

    cache: dict[Vector, tuple[float, Vector]] = {}
    evaluations = [0]

    def objective(reduced: Vector) -> tuple[float, Vector]:
        """Return the negative log evidence and its gradient (cached).

        References:
            [EQ-SUR-LFM-004].

        """
        hit = cache.get(reduced)
        if hit is not None:
            return hit
        theta = expand(reduced)
        d, s, b, ell, noise = unpack(theta)
        evaluations[0] += 1
        value, grad = _evaluate(chan_t, chan_y, d, s, b, ell, noise, init, True)
        # the optimizer works with the equilibrium level mu = B / D, which is
        # nearly orthogonal to the decay: B = mu D, so
        # d/dlog D |_mu = d/dlog D |_B + B d/dB and d/dmu = D d/dB
        converted = list(grad)
        for c in range(count):
            converted[c] += b[c] * grad[2 * count + c]
            converted[2 * count + c] = d[c] * grad[2 * count + c]
        result = (-value, tuple(-converted[i] for i in free_index))
        if len(cache) > 8:
            cache.clear()
        cache[reduced] = result
        return result

    start = tuple(full_start[i] for i in free_index)
    generator = resolve_stream(seed).child("lfm-fit").generator()
    best: tuple[float, Vector, bool, str, float] | None = None
    last_error: ValueError | None = None
    for r in range(restarts):
        if r == 0:
            x0 = start
        else:
            x0 = tuple(
                v
                + (generator.gauss(0.0, 0.5) if (i < count or i >= 3 * count) else 0.0)
                for i, v in zip(free_index, start, strict=True)
            )
        try:
            result = minimize_bfgs(
                lambda th: objective(th)[0],
                x0,
                gradient=lambda th: objective(th)[1],
                gradient_tolerance=gradient_tolerance,
                max_iterations=max_iterations,
            )
        except ValueError as error:
            last_error = error
            continue
        if best is None or result.value < best[0]:
            best = (
                result.value,
                result.x,
                result.converged,
                result.message,
                result.gradient_norm,
            )
    if best is None:
        raise ValueError(f"no start gave a valid latent force model: {last_error}")
    d, s, b, ell, noise = unpack(expand(best[1]))
    value, _ = _evaluate(chan_t, chan_y, d, s, b, ell, noise, init, False)
    return LatentForceModel(
        chan_t,
        chan_y,
        d,
        s,
        b,
        ell,
        noise,
        init,
        value,
        best[4],
        best[2],
        best[3],
        evaluations[0],
    )


# ---------------------------------------------------------------------------
# state-space latent force models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentForceChannel:
    """One ODE channel of a state-space latent force model.

    First order: ``y' + D y = S f`` with ``coefficients = (D,)``. Second order
    (normalized mass-spring-damper): ``y'' + 2 zeta omega y' + omega^2 y =
    S f`` with ``coefficients = (omega, zeta)`` [EQ-SUR-LFM-008]. The output
    is observed as ``level + y(t) + noise``; the state starts at zero (the
    channel is in equilibrium at ``level`` at ``t = 0``).

    Args:
        order: ``1`` or ``2``.
        coefficients: ``(D,)`` with ``D > 0`` or ``(omega, zeta)`` with
            ``omega > 0`` and ``zeta >= 0``.
        sensitivity: Nonzero sensitivity ``S``.
        level: Equilibrium output ``B/D`` (output units).
        noise_variance: Observation noise variance (positive).

    Raises:
        ValueError: If a field is out of range.

    References:
        [EQ-SUR-LFM-008].

    """

    order: int
    coefficients: Vector
    sensitivity: float
    level: float = 0.0
    noise_variance: float = 1e-2

    def __post_init__(self) -> None:
        """Validate the channel.

        References:
            [EQ-SUR-LFM-008].

        """
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if len(self.coefficients) != self.order:
            raise ValueError("coefficients must have `order` entries")
        for v in self.coefficients:
            _real(v, "coefficients")
        if self.coefficients[0] <= 0.0:
            raise ValueError("the first coefficient (D or omega) must be positive")
        if self.order == 2 and self.coefficients[1] < 0.0:
            raise ValueError("the damping ratio must be non-negative")
        if _real(self.sensitivity, "sensitivity") == 0.0:
            raise ValueError("sensitivity must be nonzero")
        _real(self.level, "level")
        _positive(self.noise_variance, "noise_variance")


def mass_spring_damper_channel(
    mass: float,
    damping: float,
    stiffness: float,
    gain: float = 1.0,
    *,
    level: float = 0.0,
    noise_variance: float = 1e-2,
) -> LatentForceChannel:
    """Normalize ``m y'' + c y' + k y = g f`` into a second-order channel.

    Dividing by ``m`` gives ``omega = sqrt(k/m)`` (rad/s), the damping ratio
    ``zeta = c / (2 sqrt(k m))`` and the sensitivity ``S = g / m``; SI units:
    ``m`` in kg, ``c`` in N s/m, ``k`` in N/m, ``y`` in m and ``g f`` in N
    [EQ-SUR-LFM-008]. The mass and the gain are identified only through
    ``g/m``.

    Args:
        mass: ``m > 0`` (kg).
        damping: ``c >= 0`` (N s/m).
        stiffness: ``k > 0`` (N/m).
        gain: Force gain ``g`` (nonzero).
        level: Equilibrium output.
        noise_variance: Observation noise variance.

    Returns:
        The normalized :class:`LatentForceChannel`.

    Raises:
        ValueError: If a parameter is out of range.

    Equation:
        [EQ-SUR-LFM-008].

    References:
        [REF-ALVAREZ-LUENGO-LAWRENCE-2013].

    """
    m = _positive(mass, "mass")
    c = _real(damping, "damping")
    k = _positive(stiffness, "stiffness")
    if c < 0.0:
        raise ValueError("damping must be non-negative")
    return LatentForceChannel(
        2,
        (math.sqrt(k / m), c / (2.0 * math.sqrt(k * m))),
        _real(gain, "gain") / m,
        level=level,
        noise_variance=noise_variance,
    )


def matern_state_space(
    smoothness: float, lengthscale: float
) -> tuple[_Matrix, float, _Matrix]:
    """Return the SDE realization ``(F, q, P_inf)`` of a unit-variance Matern GP.

    ``dz = F z dt + L dW`` with ``L = e_last``, spectral density ``q`` of the
    driving white noise and stationary covariance ``P_inf`` of
    ``z = (f, f', ...)``. With ``lam = sqrt(2 nu) / l``: ``nu = 1/2`` gives
    ``F = [-lam]``, ``q = 2 lam``; ``nu = 3/2`` gives
    ``F = [[0, 1], [-lam^2, -2 lam]]``, ``q = 4 lam^3``,
    ``P_inf = diag(1, lam^2)``; ``nu = 5/2`` gives the 3 state companion
    matrix with ``q = 16 lam^5 / 3`` [EQ-SUR-LFM-006]. The covariance
    ``[P_inf exp(F tau)']_(0,0)`` equals the Matern kernel.

    Args:
        smoothness: ``nu`` in ``{0.5, 1.5, 2.5}``.
        lengthscale: ``l > 0``.

    Returns:
        ``(F, q, P_inf)`` as row lists.

    Raises:
        ValueError: If ``nu`` is not supported or ``l <= 0``.

    Equation:
        [EQ-SUR-LFM-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006], [REF-SARKKA-ALVAREZ-LAWRENCE-2019].

    """
    ell = _positive(lengthscale, "lengthscale")
    if smoothness == 0.5:
        lam = 1.0 / ell
        return [[-lam]], 2.0 * lam, [[1.0]]
    if smoothness == 1.5:
        lam = math.sqrt(3.0) / ell
        return (
            [[0.0, 1.0], [-lam * lam, -2.0 * lam]],
            4.0 * lam**3,
            [[1.0, 0.0], [0.0, lam * lam]],
        )
    if smoothness == 2.5:
        lam = math.sqrt(5.0) / ell
        kappa = lam * lam / 3.0
        return (
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3), -3.0 * lam * lam, -3.0 * lam],
            ],
            16.0 * lam**5 / 3.0,
            [[1.0, 0.0, -kappa], [0.0, kappa, 0.0], [-kappa, 0.0, lam**4]],
        )
    raise ValueError("smoothness must be 0.5, 1.5 or 2.5")


def _mm(a: _Matrix, b: _Matrix) -> _Matrix:
    """Multiply two small dense matrices.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    cols = list(zip(*b, strict=True))
    return [
        [sum(x * y for x, y in zip(row, col, strict=True)) for col in cols] for row in a
    ]


def _mt(a: _Matrix) -> _Matrix:
    """Transpose a small dense matrix.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    return [list(col) for col in zip(*a, strict=True)]


def _expm(matrix: _Matrix) -> _Matrix:
    """Matrix exponential by scaling and squaring with an 18-term Taylor series.

    The matrix is scaled to infinity norm at most ``1/2`` (truncation error
    below ``1e-22``) and squared back.

    References:
        [REF-VAN-LOAN-1976], [REF-GOLUB-VANLOAN-2013].

    """
    n = len(matrix)
    norm = max(sum(abs(v) for v in row) for row in matrix)
    squarings = 0 if norm <= 0.5 else math.ceil(math.log2(norm / 0.5))
    factor = 2.0**-squarings
    a = [[v * factor for v in row] for row in matrix]
    result = [[(1.0 if i == j else 0.0) + a[i][j] for j in range(n)] for i in range(n)]
    term = [row[:] for row in a]
    for k in range(2, 19):
        term = [[v / k for v in row] for row in _mm(term, a)]
        for i in range(n):
            for j in range(n):
                result[i][j] += term[i][j]
    for _ in range(squarings):
        result = _mm(result, result)
    return result


@dataclass(frozen=True, slots=True)
class _StateModel:
    """Assembled continuous-time state-space model (internal).

    References:
        [EQ-SUR-LFM-006], [EQ-SUR-LFM-007].

    """

    drift: _Matrix
    diffusion: _Matrix
    initial_covariance: _Matrix
    output_index: tuple[int, ...]
    force_index: int


_FORCE_STATES = {0.5: 1, 1.5: 2, 2.5: 3}


def _assemble_state_model(
    channels: Sequence[LatentForceChannel], lengthscale: float, smoothness: float
) -> _StateModel:
    """Build ``(F, L q L', P0)`` of the joint channel-plus-force model.

    The state is ``(y_1, y_1', ..., z)`` with the force realization ``z`` last;
    the outputs start at zero with no uncertainty and ``z`` starts in its
    stationary law [EQ-SUR-LFM-006], [EQ-SUR-LFM-008].

    References:
        [EQ-SUR-LFM-006], [EQ-SUR-LFM-008].

    """
    force_f, q, force_p = matern_state_space(smoothness, lengthscale)
    nf = len(force_f)
    sizes = [c.order for c in channels]
    fs = sum(sizes)
    n = fs + nf
    drift = [[0.0] * n for _ in range(n)]
    outputs: list[int] = []
    pos = 0
    for ch in channels:
        outputs.append(pos)
        if ch.order == 1:
            drift[pos][pos] = -ch.coefficients[0]
            drift[pos][fs] = ch.sensitivity
        else:
            omega, zeta = ch.coefficients
            drift[pos][pos + 1] = 1.0
            drift[pos + 1][pos] = -omega * omega
            drift[pos + 1][pos + 1] = -2.0 * zeta * omega
            drift[pos + 1][fs] = ch.sensitivity
        pos += ch.order
    for i in range(nf):
        for j in range(nf):
            drift[fs + i][fs + j] = force_f[i][j]
    diffusion = [[0.0] * n for _ in range(n)]
    diffusion[n - 1][n - 1] = q
    p0 = [[0.0] * n for _ in range(n)]
    for i in range(nf):
        for j in range(nf):
            p0[fs + i][fs + j] = force_p[i][j]
    return _StateModel(drift, diffusion, p0, tuple(outputs), fs)


def _discretize(model: _StateModel, step: float) -> tuple[_Matrix, _Matrix]:
    """Return ``A = exp(F dt)`` and ``Q = int exp(F s) G exp(F' s) ds``.

    Q is computed by the Van Loan block exponential of ``[[F, G], [0, -F']]``
    [EQ-SUR-LFM-007].

    References:
        [REF-VAN-LOAN-1976], [EQ-SUR-LFM-007].

    """
    n = len(model.drift)
    big = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            big[i][j] = model.drift[i][j] * step
            big[i][n + j] = model.diffusion[i][j] * step
            big[n + i][n + j] = -model.drift[j][i] * step
    ex = _expm(big)
    a = [row[:n] for row in ex[:n]]
    b = [row[n:] for row in ex[:n]]
    q = _mm(b, _mt(a))
    q = [[0.5 * (q[i][j] + q[j][i]) for j in range(n)] for i in range(n)]
    return a, q


@dataclass(frozen=True, slots=True)
class StateSpaceLatentForceResult:
    """Filtering and smoothing output of a state-space latent force model.

    Args:
        times: Query times (as requested).
        log_marginal_likelihood: Exact log evidence ``log p(y)`` of the Gaussian
            model [EQ-SUR-LFM-007].
        output_mean: Per channel, smoothed means of ``level + y(t)`` at the
            query times (noise-free latent output).
        output_variance: Per channel, smoothed variances.
        force_mean: Smoothed mean of the unit-variance force ``f``.
        force_variance: Smoothed variance of ``f``.
        discretizations: Number of distinct ``(A, Q)`` pairs computed (one
            for a regular grid).
        steps: Number of time stamps processed (the cost is linear in it).
        observations: Number of observations.

    References:
        [EQ-SUR-LFM-007].

    """

    times: Vector
    log_marginal_likelihood: float
    output_mean: tuple[Vector, ...]
    output_variance: tuple[Vector, ...]
    force_mean: Vector
    force_variance: Vector
    discretizations: int
    steps: int
    observations: int

    def force_prediction(self) -> LatentForcePrediction:
        """Return the smoothed force as a :class:`LatentForcePrediction`.

        References:
            [EQ-SUR-LFM-007].

        """
        return LatentForcePrediction(self.times, self.force_mean, self.force_variance)


@dataclass(slots=True)
class _Step:
    """One scalar-observation (or prediction-only) step of the filter.

    References:
        [EQ-SUR-LFM-007].

    """

    mean: list[float]
    cov: _Matrix
    index: int
    innovation: float
    variance: float
    gain: list[float]
    first: bool
    stamp: int
    transition: _Matrix | None = None


def _kalman(
    channels: Sequence[LatentForceChannel],
    lengthscale: float,
    smoothness: float,
    chan_t: Sequence[Vector],
    chan_y: Sequence[Vector],
    query: Vector,
    smooth: bool,
) -> StateSpaceLatentForceResult:
    """Run the Kalman filter and the Bryson-Frazier smoother on the joint model.

    Observations are processed one scalar at a time (the noise is diagonal),
    so no innovation matrix is inverted; the evidence is
    ``sum -(log(2 pi s) + e^2 / s) / 2`` over the innovations ``e`` with
    variances ``s`` [EQ-SUR-LFM-007]. The smoother is the backward recursion
    ``r_(j-1) = z' e_j / s_j + L_j' r_j``,
    ``N_(j-1) = z' z / s_j + L_j' N_j L_j`` with ``L_j = T_j (I - K_j z)``,
    giving ``E x_j = a_j + P_j r_(j-1)`` and
    ``Var x_j = P_j - P_j N_(j-1) P_j`` from the *predicted* moments
    ``(a_j, P_j)`` of the forward pass; unlike the Rauch-Tung-Striebel gain it
    never inverts a predicted covariance, which is numerically singular for
    short steps [EQ-SUR-LFM-007].

    References:
        [EQ-SUR-LFM-007], [REF-SARKKA-ALVAREZ-LAWRENCE-2019].

    """
    model = _assemble_state_model(channels, lengthscale, smoothness)
    n = len(model.drift)
    stamps = sorted({0.0, *query, *(t for row in chan_t for t in row)})
    index = {t: k for k, t in enumerate(stamps)}
    obs: list[list[tuple[int, float]]] = [[] for _ in stamps]
    for c, (t_row, y_row) in enumerate(zip(chan_t, chan_y, strict=True)):
        for t, y in zip(t_row, y_row, strict=True):
            obs[index[t]].append((c, y - channels[c].level))
    cache: dict[float, tuple[_Matrix, _Matrix]] = {}
    mean = [0.0] * n
    cov = [row[:] for row in model.initial_covariance]
    steps: list[_Step] = []
    loglik = 0.0
    for k, t in enumerate(stamps):
        if k > 0:
            key = float(f"{t - stamps[k - 1]:.9g}")
            if key not in cache:
                cache[key] = _discretize(model, key)
            a, q = cache[key]
            mean = [sum(a[i][j] * mean[j] for j in range(n)) for i in range(n)]
            cov = _mm(_mm(a, cov), _mt(a))
            cov = [[cov[i][j] + q[i][j] for j in range(n)] for i in range(n)]
            if smooth:
                steps[-1].transition = a
        first = True
        items: list[tuple[int, float] | None] = list(obs[k]) or [None]
        for item in items:
            if item is None:
                if smooth:
                    steps.append(
                        _Step(mean[:], [r[:] for r in cov], -1, 0.0, 1.0, [], first, k)
                    )
            else:
                c, y = item
                idx = model.output_index[c]
                s = cov[idx][idx] + channels[c].noise_variance
                if s <= 0.0:
                    raise ValueError("non-positive innovation variance")
                innov = y - mean[idx]
                loglik -= 0.5 * (math.log(2.0 * math.pi * s) + innov * innov / s)
                column = [cov[i][idx] for i in range(n)]
                gain = [v / s for v in column]
                if smooth:
                    steps.append(
                        _Step(
                            mean[:], [r[:] for r in cov], idx, innov, s, gain, first, k
                        )
                    )
                mean = [mean[i] + gain[i] * innov for i in range(n)]
                for i in range(n):
                    for j in range(n):
                        cov[i][j] -= column[i] * column[j] / s
                cov = [
                    [0.5 * (cov[i][j] + cov[j][i]) for j in range(n)] for i in range(n)
                ]
            first = False
    count = sum(len(r) for r in chan_t)
    if not smooth:
        return StateSpaceLatentForceResult(
            (), loglik, (), (), (), (), len(cache), len(stamps), count
        )
    wanted = set(query)
    targets = [model.output_index[c] for c in range(len(channels))] + [
        model.force_index
    ]
    found: dict[int, tuple[list[float], list[float]]] = {}
    adjoint = [0.0] * n
    info = [[0.0] * n for _ in range(n)]
    ident = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for step in reversed(steps):
        trans = step.transition if step.transition is not None else ident
        lmat = [row[:] for row in trans]
        if step.index >= 0:
            tk = [sum(trans[i][j] * step.gain[j] for j in range(n)) for i in range(n)]
            for i in range(n):
                lmat[i][step.index] -= tk[i]
        lt = _mt(lmat)
        adjoint = [sum(lt[i][j] * adjoint[j] for j in range(n)) for i in range(n)]
        info = _mm(_mm(lt, info), lmat)
        if step.index >= 0:
            adjoint[step.index] += step.innovation / step.variance
            info[step.index][step.index] += 1.0 / step.variance
        if step.first and stamps[step.stamp] in wanted:
            means, variances = [], []
            for i in targets:
                row = step.cov[i]
                means.append(step.mean[i] + sum(row[j] * adjoint[j] for j in range(n)))
                weighted = [
                    sum(info[a][b] * row[b] for b in range(n)) for a in range(n)
                ]
                variances.append(
                    step.cov[i][i] - sum(row[a] * weighted[a] for a in range(n))
                )
            found[step.stamp] = (means, variances)
    out_m = tuple(
        tuple(channels[c].level + found[index[t]][0][c] for t in query)
        for c in range(len(channels))
    )
    out_v = tuple(
        tuple(found[index[t]][1][c] for t in query) for c in range(len(channels))
    )
    return StateSpaceLatentForceResult(
        tuple(query),
        loglik,
        out_m,
        out_v,
        tuple(found[index[t]][0][-1] for t in query),
        tuple(found[index[t]][1][-1] for t in query),
        len(cache),
        len(stamps),
        count,
    )


def _check_state_space(
    channels: Sequence[LatentForceChannel],
    lengthscale: float,
    smoothness: float,
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
) -> tuple[tuple[Vector, ...], tuple[Vector, ...], float]:
    """Validate the inputs shared by the state-space entry points.

    References:
        [EQ-SUR-LFM-007].

    """
    if not channels or not all(isinstance(c, LatentForceChannel) for c in channels):
        raise TypeError("channels must be a non-empty sequence of LatentForceChannel")
    chan_t, chan_y = _check_series(times, values)
    if len(chan_t) != len(channels):
        raise ValueError("times and values must have one entry per channel")
    if smoothness not in _FORCE_STATES:
        raise ValueError("smoothness must be 0.5, 1.5 or 2.5")
    return chan_t, chan_y, _positive(lengthscale, "lengthscale")


def state_space_latent_force_log_likelihood(
    channels: Sequence[LatentForceChannel],
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    lengthscale: float,
    smoothness: float = 1.5,
) -> float:
    """Return the exact log evidence of a state-space latent force model.

    Kalman filtering of the joint SDE of the channels and the Matern force
    gives ``log p(y) = sum_k log N(e_k; 0, s_k)`` in ``O(n)`` operations
    (``n`` observations), identical to the dense GP evidence for the kernel
    induced by the ODE and the Matern force [EQ-SUR-LFM-007].

    Args:
        channels: The ODE channels.
        times: Per channel, observation times (``>= 0``, any order; shared
            stamps are allowed, channels may be observed at different times).
        values: Per channel, observed outputs.
        lengthscale: Force length-scale ``l > 0``.
        smoothness: Matern ``nu`` in ``{0.5, 1.5, 2.5}``.

    Returns:
        The log marginal likelihood.

    Raises:
        TypeError: If ``channels`` has the wrong type.
        ValueError: If an input is invalid.

    Equation:
        [EQ-SUR-LFM-007].

    References:
        [REF-SARKKA-ALVAREZ-LAWRENCE-2019].

    """
    chan_t, chan_y, ell = _check_state_space(
        channels, lengthscale, smoothness, times, values
    )
    return _kalman(
        channels, ell, smoothness, chan_t, chan_y, (), False
    ).log_marginal_likelihood


def smooth_latent_force_state_space(
    channels: Sequence[LatentForceChannel],
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    query_times: Sequence[float],
    *,
    lengthscale: float,
    smoothness: float = 1.5,
) -> StateSpaceLatentForceResult:
    """Smooth the outputs and the force of a state-space latent force model.

    The Bryson-Frazier smoother after the Kalman filter returns the exact
    posterior marginals of the outputs and of the latent force at ``query_times``
    (inserted as missing-data stamps) [EQ-SUR-LFM-007]. The backward pass is
    the Bryson-Frazier adjoint recursion, which needs no inverse of a
    predicted covariance, so it stays accurate for steps much shorter than the
    dynamics (where the Rauch-Tung-Striebel gain is numerically singular).

    Args:
        channels: The ODE channels.
        times: Per channel, observation times.
        values: Per channel, observed outputs.
        query_times: Times ``>= 0`` of the requested marginals.
        lengthscale: Force length-scale ``l > 0``.
        smoothness: Matern ``nu`` in ``{0.5, 1.5, 2.5}``.

    Returns:
        A :class:`StateSpaceLatentForceResult`.

    Raises:
        ValueError: If an input is invalid or an innovation variance is not
            positive.

    Equation:
        [EQ-SUR-LFM-007].

    References:
        [REF-SARKKA-ALVAREZ-LAWRENCE-2019].

    """
    chan_t, chan_y, ell = _check_state_space(
        channels, lengthscale, smoothness, times, values
    )
    query = tuple(_nonnegative_time(t, "query_times") for t in query_times)
    return _kalman(channels, ell, smoothness, chan_t, chan_y, query, True)


@dataclass(frozen=True, slots=True)
class StateSpaceLatentForceFit:
    """A fitted state-space latent force model.

    Args:
        channels: Channels with fitted coefficients, sensitivities, levels and
            noise variances.
        lengthscale: Fitted force length-scale.
        smoothness: The (fixed) Matern ``nu``.
        log_marginal_likelihood: Evidence at the solution.
        converged: Optimizer outcome.
        stop_reason: Optimizer message.
        evaluations: Filter evaluations (gradient by central differences).

    References:
        [EQ-SUR-LFM-007].

    """

    channels: tuple[LatentForceChannel, ...]
    lengthscale: float
    smoothness: float
    log_marginal_likelihood: float
    converged: bool
    stop_reason: str
    evaluations: int


def fit_latent_force_state_space(
    channels: Sequence[LatentForceChannel],
    times: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    lengthscale: float,
    smoothness: float = 1.5,
    free: Sequence[str] | None = None,
    max_iterations: int = 60,
) -> StateSpaceLatentForceFit:
    """Fit a state-space latent force model by maximizing the Kalman evidence.

    The start ``channels`` and ``lengthscale`` define the initial point. The
    parameter groups that can be freed are ``"coefficients"`` (log ``D`` or
    log ``omega`` and ``zeta`` via ``log(zeta + 1e-3)``-free positive log for
    ``zeta > 0``), ``"sensitivity"``, ``"level"``, ``"noise"`` and
    ``"lengthscale"``; ``free=None`` frees all of them. The gradient is by
    central differences (each evaluation is ``O(n)``), so the cost is about
    ``2 p`` filter runs per iteration for ``p`` free parameters.

    Args:
        channels: Starting channels.
        times: Per channel, observation times.
        values: Per channel, observed outputs.
        lengthscale: Starting force length-scale.
        smoothness: Matern ``nu`` (fixed).
        free: Parameter groups to optimize.
        max_iterations: BFGS iteration limit.

    Returns:
        A :class:`StateSpaceLatentForceFit`.

    Raises:
        ValueError: If an input or a group name is invalid.

    Equation:
        [EQ-SUR-LFM-007].

    References:
        [REF-SARKKA-ALVAREZ-LAWRENCE-2019], [REF-NOCEDAL-WRIGHT-2006].

    """
    chan_t, chan_y, ell0 = _check_state_space(
        channels, lengthscale, smoothness, times, values
    )
    groups = {"coefficients", "sensitivity", "level", "noise", "lengthscale"}
    chosen = groups if free is None else set(free)
    if not chosen <= groups:
        raise ValueError(f"unknown parameter group; choose from {sorted(groups)}")
    base = tuple(channels)
    slots: list[tuple[str, int, int]] = []
    start: list[float] = []
    for c, ch in enumerate(base):
        if "coefficients" in chosen:
            for k, v in enumerate(ch.coefficients):
                if v > 0.0:
                    slots.append(("coef", c, k))
                    start.append(math.log(v))
        if "sensitivity" in chosen:
            slots.append(("sens", c, 0))
            start.append(ch.sensitivity)
        if "level" in chosen:
            slots.append(("level", c, 0))
            start.append(ch.level)
        if "noise" in chosen:
            slots.append(("noise", c, 0))
            start.append(math.log(ch.noise_variance))
    if "lengthscale" in chosen:
        slots.append(("ell", 0, 0))
        start.append(math.log(ell0))
    if not start:
        raise ValueError("no free parameters")

    def build(theta: Vector) -> tuple[tuple[LatentForceChannel, ...], float]:
        """Rebuild channels and length-scale from the free vector.

        References:
            [EQ-SUR-LFM-007].

        """
        coefs = [list(ch.coefficients) for ch in base]
        sens = [ch.sensitivity for ch in base]
        lev = [ch.level for ch in base]
        noi = [ch.noise_variance for ch in base]
        ell = ell0
        for (kind, c, k), v in zip(slots, theta, strict=True):
            if kind == "coef":
                coefs[c][k] = math.exp(v)
            elif kind == "sens":
                sens[c] = v
            elif kind == "level":
                lev[c] = v
            elif kind == "noise":
                noi[c] = math.exp(v)
            else:
                ell = math.exp(v)
        return (
            tuple(
                LatentForceChannel(ch.order, tuple(coefs[c]), sens[c], lev[c], noi[c])
                for c, ch in enumerate(base)
            ),
            ell,
        )

    evaluations = [0]

    def objective(theta: Vector) -> float:
        """Return the negative Kalman log evidence.

        References:
            [EQ-SUR-LFM-007].

        """
        evaluations[0] += 1
        chans, ell = build(theta)
        return -_kalman(
            chans, ell, smoothness, chan_t, chan_y, (), False
        ).log_marginal_likelihood

    result = minimize_bfgs(objective, tuple(start), max_iterations=max_iterations)
    chans, ell = build(result.x)
    return StateSpaceLatentForceFit(
        chans,
        ell,
        smoothness,
        -result.value,
        result.converged,
        result.message,
        evaluations[0],
    )
