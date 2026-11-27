"""Linear thermoacoustic stability of a longitudinal combustor with uncertainty.

Experimental (added in 0.28).

A one-dimensional acoustic network of length ``L`` with a compact flame at
``x_f`` and the Crocco ``n-tau`` heat-release law. The module finds the
complex eigenfrequencies of the network, their exact sensitivities to the
model parameters, the Rayleigh index of a mode, and the probability that a
mode is linearly unstable when the parameters are uncertain.

Model and assumptions (SI units: m, s, Pa, K)
---------------------------------------------

* Time dependence is ``exp(-i omega t)`` with ``omega = 2 pi f + i sigma``.
  The growth rate ``sigma = Im(omega)`` is positive for an unstable mode.
* The mean flow is neglected in the duct (``M = 0`` in the wave equation),
  and the mean temperature is uniform: sound speed ``c`` and density
  ``rho0`` are the same on both sides of the flame, so the flame acts only
  through the jump of the acoustic velocity. This is the textbook
  "uniform-temperature" idealization and holds for low Mach number and a
  small temperature ratio across the flame; it is **not** a validated model
  of any real combustor.
* Distributed losses are a linear friction ``rho0 (u_t + alpha u)`` with
  ``alpha`` the ``damping_rate`` in 1/s. A plane wave then obeys
  ``k^2 = omega (omega + i alpha) / c^2`` and a closed-closed duct without
  flame has the exact modes ``omega = -i alpha / 2 + sqrt(omega_m^2 -
  alpha^2 / 4)``, so ``alpha / 2`` is the decay rate of every mode.
* The flame is compact (``He = omega x_flame / c`` small): pressure is
  continuous and the velocity jumps by ``[u'] = Theta n u'(x_f^-, t - tau)``
  with ``Theta = (gamma - 1) Q / (gamma p u) = DeltaT / T`` the
  dimensionless heat-release strength. ``n`` is the Crocco interaction
  index and ``tau`` the time delay.
* Ends: ``"closed"`` (``u' = 0``), ``"open"`` (``p' = 0``), and at the outlet
  ``"choked"``: the Marble-Candel compact-nozzle condition for a choked
  nozzle with isentropic perturbations, ``rho0 c u' / p' = (gamma - 1) M / 2``
  with ``M`` the mean Mach number at the nozzle entry (the mean flow enters
  only through this boundary admittance; it is neglected in the duct). It
  is valid when the nozzle is acoustically compact, ``omega L_nozzle / c
  << 1``, and when entropy waves are ignored.

The eigenvalue problem is ``F(omega; p) = 0`` with the dispersion function
built from 2 x 2 transfer matrices of the pressure ``p`` and the scaled
velocity ``w = rho0 c u``. ``F`` is analytic in ``omega`` and in every
parameter, so the roots are found by Newton's method and their first
derivatives follow from implicit differentiation,
``d omega / d p = -(dF/dp) / (dF/d omega)``. The partial derivatives are
computed exactly by forward-mode differentiation of the very same function,
which is the low-order analogue of the adjoint eigenvalue sensitivity of
Magri and Juniper.

Guide: docs/foundations/aerospace-propulsion-uq.md

References:
    [REF-SUTTON-BIBLARZ-2017], [REF-RAYLEIGH-1878], [REF-CROCCO-1951],
    [REF-MARBLE-CANDEL-1977], [REF-MAGRI-JUNIPER-2013].

"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.monte_carlo import estimate_probability
from probstat.reliability.model_based import ModelLimitState, form_hlrf
from probstat.surrogates.polynomial_chaos import fit_polynomial_chaos
from probstat.uq import QuantityOfInterest

__all__ = [
    "PARAMETER_NAMES",
    "UNCERTAIN_NAMES",
    "ChamberConfig",
    "EigenvalueSensitivities",
    "InstabilityProbabilityResult",
    "RayleighIndex",
    "ThermoacousticConvergenceError",
    "ThermoacousticMode",
    "acoustic_mode_guess",
    "dispersion_function",
    "eigenvalue_sensitivities",
    "instability_probability",
    "mode_shape",
    "rayleigh_index",
    "solve_mode",
    "thermoacoustic_model",
    "thermoacoustic_modes",
]

EndKind = Literal["closed", "open", "choked"]

PARAMETER_NAMES: tuple[str, ...] = (
    "n",
    "tau",
    "damping_rate",
    "sound_speed",
    "flame_position",
    "heat_release_ratio",
    "length",
)
"""Parameters whose eigenvalue sensitivity :func:`eigenvalue_sensitivities`
returns: interaction index, delay (s), friction rate (1/s), sound speed
(m/s), flame position (m), heat-release strength, and duct length (m)."""

UNCERTAIN_NAMES: tuple[str, ...] = (
    "n",
    "tau",
    "damping_rate",
    "sound_speed",
    "flame_position",
)
"""Parameters that :func:`thermoacoustic_model` can treat as random."""

_NUMERIC_FIELDS = (
    "length",
    "sound_speed",
    "flame_position",
    "n",
    "tau",
    "damping_rate",
    "heat_release_ratio",
    "gamma",
    "nozzle_mach",
)
_SEGMENT_INTERVALS = 400


class ThermoacousticConvergenceError(RuntimeError):
    """Raised by the probabilistic model when a mode cannot be computed.

    References:
        [EQ-AERO-CI-003].

    """


# ---------------------------------------------------------------------------
# Forward-mode dual numbers over the complex field.
# ---------------------------------------------------------------------------


class _Dual:
    """Complex value with one exact directional derivative.

    References:
        [EQ-AERO-CI-004].

    """

    __slots__ = ("slope", "value")

    def __init__(self, value: complex, slope: complex = 0j) -> None:
        """Store the value and its derivative.

        References:
            [EQ-AERO-CI-004].

        """
        self.value = value
        self.slope = slope

    def __add__(self, other: _Num) -> _Dual:
        """Add a dual or complex number.

        References:
            [EQ-AERO-CI-004].

        """
        if isinstance(other, _Dual):
            return _Dual(self.value + other.value, self.slope + other.slope)
        return _Dual(self.value + other, self.slope)

    __radd__ = __add__

    def __sub__(self, other: _Num) -> _Dual:
        """Subtract a dual or complex number.

        References:
            [EQ-AERO-CI-004].

        """
        if isinstance(other, _Dual):
            return _Dual(self.value - other.value, self.slope - other.slope)
        return _Dual(self.value - other, self.slope)

    def __rsub__(self, other: complex) -> _Dual:
        """Subtract this dual from a complex number.

        References:
            [EQ-AERO-CI-004].

        """
        return _Dual(other - self.value, -self.slope)

    def __neg__(self) -> _Dual:
        """Return the negation.

        References:
            [EQ-AERO-CI-004].

        """
        return _Dual(-self.value, -self.slope)

    def __mul__(self, other: _Num) -> _Dual:
        """Multiply by a dual or complex number (product rule).

        References:
            [EQ-AERO-CI-004].

        """
        if isinstance(other, _Dual):
            return _Dual(
                self.value * other.value,
                self.slope * other.value + self.value * other.slope,
            )
        return _Dual(self.value * other, self.slope * other)

    __rmul__ = __mul__

    def __truediv__(self, other: _Num) -> _Dual:
        """Divide by a dual or complex number (quotient rule).

        References:
            [EQ-AERO-CI-004].

        """
        if isinstance(other, _Dual):
            return _Dual(
                self.value / other.value,
                (self.slope * other.value - self.value * other.slope)
                / (other.value * other.value),
            )
        return _Dual(self.value / other, self.slope / other)

    def __rtruediv__(self, other: complex) -> _Dual:
        """Divide a complex number by this dual.

        References:
            [EQ-AERO-CI-004].

        """
        return _Dual(
            other / self.value, -other * self.slope / (self.value * self.value)
        )


_Num = Any  # complex or _Dual; kept loose so operator mixing type-checks.


def _value(number: _Num) -> complex:
    """Return the plain complex value of a dual or complex number.

    References:
        [EQ-AERO-CI-004].

    """
    return number.value if isinstance(number, _Dual) else complex(number)


def _exp(number: _Num) -> _Num:
    """Return ``exp`` of a dual or complex number.

    References:
        [EQ-AERO-CI-004].

    """
    if isinstance(number, _Dual):
        value = cmath.exp(number.value)
        return _Dual(value, value * number.slope)
    return cmath.exp(number)


def _sqrt(number: _Num) -> _Num:
    """Return the principal square root of a dual or complex number.

    References:
        [EQ-AERO-CI-004].

    """
    if isinstance(number, _Dual):
        root = cmath.sqrt(number.value)
        return _Dual(root, number.slope / (2.0 * root))
    return cmath.sqrt(number)


def _sin(number: _Num) -> _Num:
    """Return ``sin`` of a dual or complex number.

    References:
        [EQ-AERO-CI-004].

    """
    if isinstance(number, _Dual):
        return _Dual(cmath.sin(number.value), cmath.cos(number.value) * number.slope)
    return cmath.sin(number)


def _cos(number: _Num) -> _Num:
    """Return ``cos`` of a dual or complex number.

    References:
        [EQ-AERO-CI-004].

    """
    if isinstance(number, _Dual):
        return _Dual(cmath.cos(number.value), -cmath.sin(number.value) * number.slope)
    return cmath.cos(number)


def _cos_and_sinc(q: _Num) -> tuple[_Num, _Num]:
    """Return ``cos(z)`` and ``sin(z)/z`` as entire functions of ``q = z^2``.

    A 12-term series is used for ``|q| < 1`` (truncation below ``1e-25``),
    which also removes the removable singularity at ``z = 0``; otherwise the
    closed forms are evaluated. Both are even in ``z``, so the branch of the
    square root is immaterial.

    References:
        [EQ-AERO-CI-001].

    """
    if abs(_value(q)) < 1.0:
        cosine_term: _Num = 1.0 + 0.0j
        sinc_term: _Num = 1.0 + 0.0j
        cosine = cosine_term
        sinc = sinc_term
        for m in range(1, 13):
            cosine_term = cosine_term * (-q) / float((2 * m) * (2 * m - 1))
            sinc_term = sinc_term * (-q) / float((2 * m) * (2 * m + 1))
            cosine = cosine + cosine_term
            sinc = sinc + sinc_term
        return cosine, sinc
    z = _sqrt(q)
    return _cos(z), _sin(z) / z


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChamberConfig:
    """Longitudinal chamber with a compact flame and an ``n-tau`` law.

    Args:
        length: Duct length ``L`` in m (positive).
        sound_speed: Uniform speed of sound ``c`` in m/s (positive).
        flame_position: Flame location ``x_f`` in m, ``0 < x_f < L``.
        n: Crocco interaction index (dimensionless, any finite sign).
        tau: Flame time delay in s (non-negative).
        damping_rate: Friction rate ``alpha`` in 1/s (non-negative); every
            mode of the flameless closed-closed duct decays at ``alpha/2``.
        heat_release_ratio: ``Theta = (gamma - 1) Q / (gamma p u) = DeltaT/T``
            (dimensionless, non-negative); the flame velocity jump is
            ``Theta n u'(t - tau)``.
        inlet: ``"closed"`` or ``"open"`` end at ``x = 0``.
        outlet: ``"closed"``, ``"open"``, or ``"choked"`` end at ``x = L``.
        gamma: Heat-capacity ratio (``> 1``), used only by the choked nozzle.
        nozzle_mach: Mean Mach number at the nozzle entry in ``[0, 1)``,
            used only by the choked outlet.

    Raises:
        TypeError: If a numeric field is not a real number or an end kind is
            not a string.
        ValueError: If a field is outside its documented range or an end
            kind is unknown.

    Equation:
        [EQ-AERO-CI-001], [EQ-AERO-CI-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [REF-MARBLE-CANDEL-1977].

    """

    length: float
    sound_speed: float
    flame_position: float
    n: float
    tau: float
    damping_rate: float = 0.0
    heat_release_ratio: float = 1.0
    inlet: str = "closed"
    outlet: str = "closed"
    gamma: float = 1.4
    nozzle_mach: float = 0.1

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            TypeError: If a numeric field is not a real number.
            ValueError: If a field is outside its range.

        References:
            [EQ-AERO-CI-001].

        """
        for name in _NUMERIC_FIELDS:
            object.__setattr__(
                self, name, require_finite_real(getattr(self, name), name=name)
            )
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.sound_speed <= 0.0:
            raise ValueError("sound_speed must be positive")
        if not 0.0 < self.flame_position < self.length:
            raise ValueError("flame_position must lie strictly inside (0, length)")
        if self.tau < 0.0:
            raise ValueError("tau must be non-negative")
        if self.damping_rate < 0.0:
            raise ValueError("damping_rate must be non-negative")
        if self.heat_release_ratio < 0.0:
            raise ValueError("heat_release_ratio must be non-negative")
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        if not 0.0 <= self.nozzle_mach < 1.0:
            raise ValueError("nozzle_mach must lie in [0, 1)")
        for label, kind, allowed in (
            ("inlet", self.inlet, ("closed", "open")),
            ("outlet", self.outlet, ("closed", "open", "choked")),
        ):
            if not isinstance(kind, str):
                raise TypeError(f"{label} must be a string")
            if kind not in allowed:
                raise ValueError(f"{label} must be one of {allowed}")

    @property
    def nozzle_admittance(self) -> float:
        """Return the normalized choked-nozzle admittance ``(gamma-1) M / 2``.

        References:
            [EQ-AERO-CI-002], [REF-MARBLE-CANDEL-1977].

        """
        return 0.5 * (self.gamma - 1.0) * self.nozzle_mach

    def end_coefficients(self) -> tuple[float, float, float, float]:
        """Return ``(a0, b0, aL, bL)`` of the end conditions ``a p + b w = 0``.

        Closed: ``(0, 1)``; open: ``(1, 0)``; choked outlet: ``(-zeta, 1)``
        with ``zeta`` the nozzle admittance; ``w = rho0 c u``.

        References:
            [EQ-AERO-CI-002], [REF-MARBLE-CANDEL-1977].

        """
        left = (0.0, 1.0) if self.inlet == "closed" else (1.0, 0.0)
        if self.outlet == "closed":
            right = (0.0, 1.0)
        elif self.outlet == "open":
            right = (1.0, 0.0)
        else:
            right = (-self.nozzle_admittance, 1.0)
        return left[0], left[1], right[0], right[1]


# ---------------------------------------------------------------------------
# Dispersion function and its derivatives.
# ---------------------------------------------------------------------------


def _transfer(omega: _Num, c: _Num, alpha: _Num, x: _Num) -> tuple[_Num, _Num, _Num]:
    """Return ``(C, B, E)`` of the transfer matrix ``[[C, B], [E, C]]`` over ``x``.

    With ``k^2 = omega (omega + i alpha) / c^2``: ``C = cos kx``,
    ``B = i (omega + i alpha) sin(kx)/(k c)`` and ``E = i omega sin(kx)/(k c)``.

    References:
        [EQ-AERO-CI-001].

    """
    q = omega * (omega + 1j * alpha) * x * x / (c * c)
    cosine, sinc = _cos_and_sinc(q)
    sine_term = x * sinc / c
    return cosine, 1j * (omega + 1j * alpha) * sine_term, 1j * omega * sine_term


def _dispersion(
    omega: _Num,
    c: _Num,
    length: _Num,
    flame: _Num,
    n: _Num,
    tau: _Num,
    alpha: _Num,
    theta: _Num,
    ends: tuple[float, float, float, float],
) -> _Num:
    """Evaluate ``F(omega)`` for dual or complex arguments.

    References:
        [EQ-AERO-CI-001].

    """
    a0, b0, a1, b1 = ends
    c1, bb1, e1 = _transfer(omega, c, alpha, flame)
    pressure = c1 * b0 - bb1 * a0
    velocity = e1 * b0 - c1 * a0
    velocity = velocity * (1.0 + theta * n * _exp(1j * omega * tau))
    c2, bb2, e2 = _transfer(omega, c, alpha, length - flame)
    pressure_out = c2 * pressure + bb2 * velocity
    velocity_out = e2 * pressure + c2 * velocity
    return a1 * pressure_out + b1 * velocity_out


def _arguments(omega: complex, config: ChamberConfig) -> dict[str, _Num]:
    """Return the named positional arguments of :func:`_dispersion`.

    References:
        [EQ-AERO-CI-001].

    """
    return {
        "omega": omega,
        "c": config.sound_speed,
        "length": config.length,
        "flame": config.flame_position,
        "n": config.n,
        "tau": config.tau,
        "alpha": config.damping_rate,
        "theta": config.heat_release_ratio,
    }


_ARGUMENT_OF = {
    "omega": "omega",
    "n": "n",
    "tau": "tau",
    "damping_rate": "alpha",
    "sound_speed": "c",
    "flame_position": "flame",
    "heat_release_ratio": "theta",
    "length": "length",
}


def _evaluate(
    omega: complex, config: ChamberConfig, wrt: str | None = None
) -> tuple[complex, complex]:
    """Return ``F`` and its derivative with respect to one named quantity.

    References:
        [EQ-AERO-CI-001], [EQ-AERO-CI-004].

    """
    arguments = _arguments(omega, config)
    if wrt is not None:
        key = _ARGUMENT_OF[wrt]
        arguments[key] = _Dual(complex(arguments[key]), 1.0 + 0.0j)
    result = _dispersion(ends=config.end_coefficients(), **arguments)
    if isinstance(result, _Dual):
        return result.value, result.slope
    return complex(result), 0j


def dispersion_function(omega: complex, config: ChamberConfig) -> complex:
    """Evaluate the complex dispersion function ``F(omega)`` of the chamber.

    ``F(omega) = a_L p(L) + b_L w(L)`` for the solution that satisfies the
    inlet condition and the flame jump ``[w] = Theta n e^{i omega tau} w(x_f^-)``;
    its roots are the eigenfrequencies. For ``alpha = 0`` and a closed-closed
    duct, ``F = i (omega / c) [sin kL + K cos(k (L - x_f)) sin(k x_f)] / k``
    with ``K = Theta n e^{i omega tau}``.

    Args:
        omega: Complex angular frequency in rad/s.
        config: Chamber definition.

    Returns:
        ``F(omega)`` (complex, dimensionless).

    Raises:
        TypeError: If ``omega`` is not a number or ``config`` has the wrong type.

    Equation:
        [EQ-AERO-CI-001].

    References:
        [REF-RAYLEIGH-1878], [REF-CROCCO-1951], [REF-MAGRI-JUNIPER-2013].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    return _evaluate(_as_complex(omega, "omega"), config)[0]


def _as_complex(value: object, name: str) -> complex:
    """Return a finite complex number or raise.

    Raises:
        TypeError: If ``value`` is not a number.
        ValueError: If it is not finite.

    References:
        [EQ-AERO-CI-001].

    """
    if isinstance(value, bool) or not isinstance(value, int | float | complex):
        raise TypeError(f"{name} must be a number")
    number = complex(value)
    if not (math.isfinite(number.real) and math.isfinite(number.imag)):
        raise ValueError(f"{name} must be finite")
    return number


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermoacousticMode:
    """One Newton solution of ``F(omega) = 0``.

    Args:
        mode_index: Index ``m >= 1`` of the acoustic mode the iteration started from.
        initial_guess: Starting value of ``omega`` (rad/s).
        omega: Final complex frequency (rad/s), ``exp(-i omega t)`` convention.
        frequency_hz: ``Re(omega) / 2 pi`` in Hz.
        growth_rate: ``Im(omega)`` in 1/s; positive means unstable.
        converged: Whether the stopping test was met.
        iterations: Newton iterations performed.
        stop_reason: ``"converged"``, ``"maximum iterations reached"``,
            ``"singular derivative"``, ``"non-finite value"`` or
            ``"overflow"``.
        residual: ``|F(omega)|`` at the returned value.
        last_step: Size ``|Delta omega|`` of the last Newton step.

    References:
        [EQ-AERO-CI-003].

    """

    mode_index: int
    initial_guess: complex
    omega: complex
    frequency_hz: float
    growth_rate: float
    converged: bool
    iterations: int
    stop_reason: str
    residual: float
    last_step: float

    @property
    def unstable(self) -> bool:
        """Return ``True`` when the mode converged with ``growth_rate > 0``.

        References:
            [EQ-AERO-CI-003].

        """
        return self.converged and self.growth_rate > 0.0


def acoustic_mode_guess(config: ChamberConfig, mode: int) -> complex:
    """Return the Newton starting value of acoustic mode ``mode``.

    The base value is the lossless, flameless mode ``omega_m = m pi c / L``
    (closed-closed, open-open, closed-choked, and the choked-nozzle limit
    ``M -> 0``) or ``(m - 1/2) pi c / L`` (closed-open); friction shifts it to
    ``-i alpha / 2 + sqrt(omega_m^2 - alpha^2 / 4)``, exact for the flameless
    duct with rigid or open ends.

    Args:
        config: Chamber definition.
        mode: Mode number ``m >= 1``.

    Returns:
        Complex starting frequency in rad/s.

    Raises:
        TypeError: If ``mode`` is not an integer.
        ValueError: If ``mode < 1``.

    Equation:
        [EQ-AERO-CI-003].

    References:
        [REF-SUTTON-BIBLARZ-2017], [REF-CROCCO-1951].

    """
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("mode must be an integer")
    if mode < 1:
        raise ValueError("mode must be at least 1")
    right_open = config.outlet == "open"
    left_open = config.inlet == "open"
    shift = 0.0 if left_open == right_open else 0.5
    base = (mode - shift) * math.pi * config.sound_speed / config.length
    alpha = config.damping_rate
    return -0.5j * alpha + cmath.sqrt(base * base - 0.25 * alpha * alpha)


def solve_mode(
    config: ChamberConfig,
    initial_omega: complex | None = None,
    *,
    mode: int = 1,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> ThermoacousticMode:
    """Solve ``F(omega) = 0`` by Newton's method and report convergence.

    The Newton update is ``omega <- omega - F / F'`` with ``F'`` the exact
    derivative from forward-mode differentiation. The iteration stops with
    ``converged=True`` when ``|Delta omega| <= tolerance max(1, |omega|)``.
    Failures are reported in ``stop_reason`` and never hidden: there is no
    line search, damping, or restart.

    Args:
        config: Chamber definition.
        initial_omega: Starting value (rad/s); defaults to
            :func:`acoustic_mode_guess` of ``mode``.
        mode: Mode number used for the default start and recorded in the result.
        tolerance: Relative step tolerance (positive).
        max_iterations: Iteration cap (positive).

    Returns:
        :class:`ThermoacousticMode` with frequency, growth rate, and status.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a numerical option is not positive or the start is not finite.

    Equation:
        [EQ-AERO-CI-003].

    References:
        [REF-MAGRI-JUNIPER-2013], [REF-CROCCO-1951].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    start = (
        acoustic_mode_guess(config, mode)
        if initial_omega is None
        else _as_complex(initial_omega, "initial_omega")
    )
    omega = start
    status = "maximum iterations reached"
    step_size = math.inf
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        try:
            value, slope = _evaluate(omega, config, "omega")
        except (OverflowError, ValueError):
            status = "overflow"
            break
        if not (cmath.isfinite(value) and cmath.isfinite(slope)):
            status = "non-finite value"
            break
        if slope == 0.0:
            status = "singular derivative"
            break
        step = value / slope
        omega = omega - step
        step_size = abs(step)
        if step_size <= tol * max(1.0, abs(omega)):
            status = "converged"
            break
    try:
        residual = abs(_evaluate(omega, config)[0])
    except (OverflowError, ValueError):
        residual = math.inf
    return ThermoacousticMode(
        mode,
        start,
        omega,
        omega.real / (2.0 * math.pi),
        omega.imag,
        status == "converged",
        iterations,
        status,
        residual,
        step_size,
    )


def thermoacoustic_modes(
    config: ChamberConfig,
    count: int = 3,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> tuple[ThermoacousticMode, ...]:
    """Solve the first ``count`` modes from their acoustic initial guesses.

    Args:
        config: Chamber definition.
        count: Number of modes (positive).
        tolerance: Relative Newton step tolerance.
        max_iterations: Newton iteration cap per mode.

    Returns:
        One :class:`ThermoacousticMode` per mode, in order; check ``converged``.
        Distinct starts may converge to the same root for large ``n``.

    Raises:
        TypeError: If ``count`` is not an integer.
        ValueError: If ``count`` is not positive.

    Equation:
        [EQ-AERO-CI-003].

    References:
        [REF-MAGRI-JUNIPER-2013].

    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")
    return tuple(
        solve_mode(config, mode=m, tolerance=tolerance, max_iterations=max_iterations)
        for m in range(1, count + 1)
    )


# ---------------------------------------------------------------------------
# Sensitivities.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EigenvalueSensitivities:
    """Exact first derivatives of an eigenvalue with respect to parameters.

    Args:
        omega: Eigenvalue (rad/s) at which the derivatives are evaluated.
        d_omega: ``d omega / d p`` (complex) keyed by :data:`PARAMETER_NAMES`.
        d_growth_rate: ``d Im(omega) / d p`` keyed by parameter.
        d_frequency_hz: ``d Re(omega) / (2 pi dp)`` keyed by parameter.
        residual: ``|F(omega)|``; the derivatives are those of a true root only
            when this is small.

    References:
        [EQ-AERO-CI-004].

    """

    omega: complex
    d_omega: Mapping[str, complex]
    d_growth_rate: Mapping[str, float]
    d_frequency_hz: Mapping[str, float]
    residual: float

    def __post_init__(self) -> None:
        """Freeze the mappings.

        References:
            [EQ-AERO-CI-004].

        """
        for name in ("d_omega", "d_growth_rate", "d_frequency_hz"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def eigenvalue_sensitivities(
    config: ChamberConfig, omega: complex
) -> EigenvalueSensitivities:
    """Differentiate an eigenvalue implicitly with respect to all parameters.

    From ``F(omega(p), p) = 0``: ``d omega / d p = -(dF/dp) / (dF/d omega)``.
    Both partial derivatives are exact (forward-mode), so the result matches
    finite differences of re-solved roots to the finite-difference accuracy
    and costs ``8`` evaluations of ``F`` instead of ``2 * 7`` Newton solves.
    The eigenvalue must be simple (``dF/d omega != 0``).

    Args:
        config: Chamber definition.
        omega: A root of ``F`` (e.g. ``ThermoacousticMode.omega``).

    Returns:
        :class:`EigenvalueSensitivities`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``omega`` is not finite or ``dF/d omega`` vanishes
            (a repeated eigenvalue; the derivative does not exist).

    Equation:
        [EQ-AERO-CI-004].

    References:
        [REF-MAGRI-JUNIPER-2013].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    root = _as_complex(omega, "omega")
    value, d_omega_f = _evaluate(root, config, "omega")
    if d_omega_f == 0.0 or not cmath.isfinite(d_omega_f):
        raise ValueError("dF/domega vanishes: the eigenvalue is not simple")
    d_omega: dict[str, complex] = {}
    for name in PARAMETER_NAMES:
        d_omega[name] = -_evaluate(root, config, name)[1] / d_omega_f
    return EigenvalueSensitivities(
        root,
        d_omega,
        {name: d.imag for name, d in d_omega.items()},
        {name: d.real / (2.0 * math.pi) for name, d in d_omega.items()},
        abs(value),
    )


# ---------------------------------------------------------------------------
# Mode shape and Rayleigh index.
# ---------------------------------------------------------------------------


def mode_shape(
    config: ChamberConfig, omega: complex, positions: Sequence[float]
) -> tuple[tuple[complex, complex], ...]:
    """Return the complex pressure ``p`` and scaled velocity ``w = rho0 c u``.

    The shape is propagated from the inlet by the transfer matrices and the
    flame jump, with the arbitrary scale ``(p, w)(0) = (b_0, -a_0)``. At a
    root of ``F`` it also satisfies the outlet condition.

    Args:
        config: Chamber definition.
        omega: Complex frequency in rad/s (normally a root of ``F``).
        positions: Points ``x`` in ``[0, L]`` in m.

    Returns:
        ``(p(x), w(x))`` for each position.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a position is outside ``[0, L]`` or not finite.

    Equation:
        [EQ-AERO-CI-001].

    References:
        [REF-CROCCO-1951], [REF-MAGRI-JUNIPER-2013].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    root = _as_complex(omega, "omega")
    points = [require_finite_real(x, name="position") for x in positions]
    if any(x < 0.0 or x > config.length for x in points):
        raise ValueError("positions must lie in [0, length]")
    return tuple(_shape_at(config, root, x) for x in points)


def _shape_at(
    config: ChamberConfig, omega: complex, x: float
) -> tuple[complex, complex]:
    """Return ``(p, w)`` at one position.

    References:
        [EQ-AERO-CI-001].

    """
    a0, b0, _, _ = config.end_coefficients()
    c = config.sound_speed
    alpha = config.damping_rate
    pressure, velocity = complex(b0), complex(-a0)
    flame = config.flame_position
    first = min(x, flame)
    cc, bb, ee = _transfer(omega, c, alpha, first)
    pressure, velocity = (
        cc * pressure + bb * velocity,
        ee * pressure + cc * velocity,
    )
    if x > flame:
        coupling = config.heat_release_ratio * config.n
        velocity = velocity * (1.0 + coupling * cmath.exp(1j * omega * config.tau))
        cc, bb, ee = _transfer(omega, c, alpha, x - flame)
        pressure, velocity = (
            cc * pressure + bb * velocity,
            ee * pressure + cc * velocity,
        )
    return complex(pressure), complex(velocity)


@dataclass(frozen=True, slots=True)
class RayleighIndex:
    """Rayleigh index and acoustic-energy balance of one mode.

    All growth contributions are in 1/s and are normalized by the mode's
    acoustic energy, so they do not depend on the mode scale.

    Args:
        omega: Eigenvalue (rad/s).
        pressure_at_flame: ``p(x_f)`` for the unscaled shape ``mode_shape``.
        heat_release_jump: ``[w] = Theta n e^{i omega tau} w(x_f^-)`` (the
            flame's velocity source in the same scale).
        phase_cosine: ``Re(conj(p) [w]) / (|p| |[w]|)`` in ``[-1, 1]``;
            positive means heat release in phase with pressure.
        flame_growth_rate: Rayleigh-index contribution of the flame.
        boundary_growth_rate: Net acoustic-energy flux through the ends.
        dissipation_growth_rate: Friction loss ``-alpha ...`` (non-positive).
        balance_growth_rate: Sum of the three contributions: the growth rate
            implied by the acoustic-energy balance.
        growth_rate: ``Im(omega)``.
        balance_error: ``balance_growth_rate - growth_rate``. The balance is
            an identity of the transfer-matrix solution (for any ``omega``, as
            the outlet flux is read from the shape), so this is quadrature
            error only; it is not a test that ``omega`` is a root of ``F``.

    References:
        [EQ-AERO-CI-005].

    """

    omega: complex
    pressure_at_flame: complex
    heat_release_jump: complex
    phase_cosine: float
    flame_growth_rate: float
    boundary_growth_rate: float
    dissipation_growth_rate: float
    balance_growth_rate: float
    growth_rate: float
    balance_error: float


def _simpson(values: Sequence[float], step: float) -> float:
    """Return the composite Simpson integral of equally spaced samples.

    References:
        [EQ-AERO-CI-005].

    """
    odd = math.fsum(values[1:-1:2])
    even = math.fsum(values[2:-1:2])
    return step / 3.0 * (values[0] + values[-1] + 4.0 * odd + 2.0 * even)


def rayleigh_index(config: ChamberConfig, omega: complex) -> RayleighIndex:
    """Compute the Rayleigh index of a mode and the acoustic-energy balance.

    Rayleigh's criterion says heat addition in phase with pressure feeds a
    mode. For the compact flame the power into the acoustic
    energy is ``P = Re(conj(p) [u]) / 2`` with ``[u] = [w] / (rho0 c)``. With
    the acoustic energy ``E = integral (|p|^2/(rho0 c^2) + rho0 |u|^2) / 4``,
    the energy balance ``2 sigma E = P + boundary flux - friction`` gives
    ``sigma = c (Re(conj(p)[w]) + flux terms - (alpha/c) int |w|^2) /
    int (|p|^2 + |w|^2)``, evaluated by Simpson's rule per duct segment.

    Args:
        config: Chamber definition.
        omega: Complex frequency in rad/s (a root of ``F``).

    Returns:
        :class:`RayleighIndex`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``omega`` is not finite or the mode has zero energy.

    Equation:
        [EQ-AERO-CI-005].

    References:
        [REF-RAYLEIGH-1878], [REF-MAGRI-JUNIPER-2013].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    root = _as_complex(omega, "omega")
    flame = config.flame_position
    energy = 0.0
    friction = 0.0
    for lower, upper in ((0.0, flame), (flame, config.length)):
        step = (upper - lower) / _SEGMENT_INTERVALS
        samples = [
            _shape_at(config, root, lower + i * step)
            for i in range(_SEGMENT_INTERVALS + 1)
        ]
        # The sample at x_f belongs to the flame's upstream side for the first
        # segment and is rebuilt downstream for the second (pressure continuous).
        if lower == flame:
            samples[0] = _downstream_at_flame(config, root)
        total = [abs(p) ** 2 + abs(w) ** 2 for p, w in samples]
        loss = [abs(w) ** 2 for _, w in samples]
        energy += _simpson(total, step)
        friction += _simpson(loss, step)
    if energy <= 0.0:
        raise ValueError("the mode has zero energy")
    p_f, w_up = _shape_at(config, root, flame)
    jump = (
        config.heat_release_ratio * config.n * cmath.exp(1j * root * config.tau) * w_up
    )
    p0, w0 = _shape_at(config, root, 0.0)
    p1, w1 = _shape_at(config, root, config.length)
    c = config.sound_speed
    flame_term = c * (p_f.conjugate() * jump).real / energy
    boundary = c * ((p0.conjugate() * w0).real - (p1.conjugate() * w1).real) / energy
    dissipation = -config.damping_rate * friction / energy
    balance = flame_term + boundary + dissipation
    scale = abs(p_f) * abs(jump)
    cosine = (p_f.conjugate() * jump).real / scale if scale > 0.0 else 0.0
    return RayleighIndex(
        root,
        p_f,
        jump,
        cosine,
        flame_term,
        boundary,
        dissipation,
        balance,
        root.imag,
        balance - root.imag,
    )


def _downstream_at_flame(
    config: ChamberConfig, omega: complex
) -> tuple[complex, complex]:
    """Return ``(p, w)`` just downstream of the flame.

    References:
        [EQ-AERO-CI-001].

    """
    p, w = _shape_at(config, omega, config.flame_position)
    coupling = config.heat_release_ratio * config.n
    return p, w * (1.0 + coupling * cmath.exp(1j * omega * config.tau))


# ---------------------------------------------------------------------------
# Probabilistic model and instability probability.
# ---------------------------------------------------------------------------

_MARGIN_SIGMAS = 6.0


class _ChamberResponse:
    """Picklable forward map from named parameters to the mode's growth rate.

    References:
        [EQ-AERO-CI-006].

    """

    def __init__(
        self,
        config: ChamberConfig,
        mode: int,
        tolerance: float,
        max_iterations: int,
        max_mode_shift: float,
    ) -> None:
        """Store the base configuration and solver options.

        References:
            [EQ-AERO-CI-006].

        """
        self.config = config
        self.mode = mode
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_mode_shift = max_mode_shift

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Return ``growth_rate`` and ``frequency_hz`` of the tracked mode.

        Raises:
            ThermoacousticConvergenceError: If Newton does not converge or
                the root left its acoustic neighbourhood (mode jump).

        References:
            [EQ-AERO-CI-006].

        """
        overrides: dict[str, Any] = {k: float(v) for k, v in parameters.items()}
        config = replace(self.config, **overrides)
        found = solve_mode(
            config,
            mode=self.mode,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        if not found.converged:
            raise ThermoacousticConvergenceError(
                f"mode {self.mode}: Newton stopped with {found.stop_reason!r}"
            )
        spacing = math.pi * config.sound_speed / config.length
        if (
            abs(found.omega.real - found.initial_guess.real)
            > self.max_mode_shift * spacing
        ):
            raise ThermoacousticConvergenceError(
                f"mode {self.mode}: root moved by more than "
                f"{self.max_mode_shift} mode spacings (mode jump)"
            )
        return {"growth_rate": found.growth_rate, "frequency_hz": found.frequency_hz}


def thermoacoustic_model(
    config: ChamberConfig,
    relative_std: Mapping[str, float] | None = None,
    *,
    mode: int = 1,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
    max_mode_shift: float = 0.5,
    name: str = "thermoacoustic-chamber",
) -> ProbabilisticModel:
    """Build a :class:`~probstat.modeling.ProbabilisticModel` of one mode.

    The chosen parameters are independent normal variables with mean equal to
    the value in ``config`` and standard deviation ``relative_std * |mean|``;
    the others stay fixed. Outputs are ``growth_rate`` (1/s) and
    ``frequency_hz``. A non-converged Newton solve or a root farther than
    ``max_mode_shift`` mode spacings from the acoustic guess raises
    :class:`ThermoacousticConvergenceError` (evaluation failure policy
    ``"raise"``): such samples are never silently dropped.

    Because the normal law is unbounded, positive parameters and the flame
    position must keep a margin of 6 standard deviations from their bounds
    (``0`` and the duct ends), which limits their relative spread.

    Args:
        config: Mean chamber.
        relative_std: Coefficient of variation per parameter among
            :data:`UNCERTAIN_NAMES`; default ``n: 0.2, tau: 0.05,
            damping_rate: 0.1, sound_speed: 0.02, flame_position: 0.02``.
            A zero entry keeps the parameter fixed, and a default spread is
            skipped for a parameter whose mean is zero (an explicit spread on
            a zero mean raises).
        mode: Acoustic mode tracked (``>= 1``).
        tolerance: Newton step tolerance.
        max_iterations: Newton iteration cap.
        max_mode_shift: Allowed root displacement in mode spacings.
        name: Model label.

    Returns:
        Model whose inputs are the uncertain parameters in
        :data:`UNCERTAIN_NAMES` order, so ``UQStudy``, sensitivity, reliability
        and surrogate tools apply to it directly.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a name is unknown, a spread is negative, ``n`` is zero
            with a positive spread, a margin condition fails, or nothing is
            uncertain.

    Equation:
        [EQ-AERO-CI-006].

    References:
        [REF-MAGRI-JUNIPER-2013], [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(config, ChamberConfig):
        raise TypeError("config must be a ChamberConfig")
    spreads: dict[str, float] = {
        "n": 0.2,
        "tau": 0.05,
        "damping_rate": 0.1,
        "sound_speed": 0.02,
        "flame_position": 0.02,
    }
    requested = set(relative_std) if relative_std is not None else set()
    if relative_std is not None:
        for key, value in relative_std.items():
            if key not in UNCERTAIN_NAMES:
                raise ValueError(f"unknown uncertain parameter {key!r}")
            spreads[key] = require_finite_real(value, name=f"relative_std[{key!r}]")
    tol = require_finite_real(tolerance, name="tolerance")
    shift = require_finite_real(max_mode_shift, name="max_mode_shift")
    if tol <= 0.0 or shift <= 0.0:
        raise ValueError("tolerance and max_mode_shift must be positive")
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 1:
        raise ValueError("mode must be an integer >= 1")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    inputs: list[RandomInput] = []
    units = {
        "n": "1",
        "tau": "s",
        "damping_rate": "1/s",
        "sound_speed": "m/s",
        "flame_position": "m",
    }
    for key in UNCERTAIN_NAMES:
        cv = spreads[key]
        if cv < 0.0:
            raise ValueError(f"relative_std[{key!r}] must be non-negative")
        if cv == 0.0:
            continue
        mean = float(getattr(config, key))
        if mean == 0.0 and key not in requested:
            continue  # a default spread cannot apply to a zero mean
        if mean == 0.0:
            raise ValueError(f"{key} has mean zero, so a relative spread is undefined")
        sd = cv * abs(mean)
        margin = _MARGIN_SIGMAS * sd
        if key == "flame_position":
            if mean - margin <= 0.0 or mean + margin >= config.length:
                raise ValueError(
                    "flame_position spread leaves less than 6 sd of margin"
                )
        elif key != "n" and mean - margin <= 0.0:
            raise ValueError(f"{key} spread leaves less than 6 sd of margin to zero")
        inputs.append(
            RandomInput(key, Normal(mean, sd), f"uncertain {key}", units[key])
        )
    if not inputs:
        raise ValueError("at least one parameter must be uncertain")
    return ProbabilisticModel(
        JointInput.independent(*inputs),
        _ChamberResponse(config, mode, tol, max_iterations, shift),
        (
            QuantityOfInterest("growth_rate", "linear growth rate Im(omega)", "1/s"),
            QuantityOfInterest(
                "frequency_hz", "oscillation frequency Re(omega)/2pi", "Hz"
            ),
        ),
        name=name,
    )


@dataclass(frozen=True, slots=True)
class InstabilityProbabilityResult:
    """Three estimates of ``P(growth_rate > 0)`` and their comparison.

    Args:
        monte_carlo_probability: Direct Monte Carlo estimate.
        monte_carlo_interval: Wilson interval ``(lower, upper)``.
        monte_carlo_samples: Number of exact mode solves.
        mean_growth_rate: Sample mean of the growth rate (1/s).
        std_growth_rate: Sample standard deviation of the growth rate (1/s).
        surrogate_probability: PCE-surrogate estimate (large surrogate sample).
        surrogate_error: Leave-one-out relative error of the PCE.
        surrogate_evaluations: Exact solves used to fit the surrogate.
        form_probability: ``Phi(-beta)`` from FORM on ``g = -growth_rate``.
        form_beta: Signed FORM reliability index (positive: stable at the
            design point is farther than the mean; ``beta < 0``: the mean
            chamber is unstable).
        form_converged: HL-RF convergence flag.
        form_evaluations: Exact solves charged to FORM.
        surrogate_within_interval: Surrogate estimate lies in the Monte Carlo
            interval.
        form_within_interval: FORM estimate lies in the Monte Carlo interval.

    References:
        [EQ-AERO-CI-007].

    """

    monte_carlo_probability: float
    monte_carlo_interval: tuple[float, float]
    monte_carlo_samples: int
    mean_growth_rate: float
    std_growth_rate: float
    surrogate_probability: float
    surrogate_error: float
    surrogate_evaluations: int
    form_probability: float | None
    form_beta: float | None
    form_converged: bool | None
    form_evaluations: int | None
    surrogate_within_interval: bool
    form_within_interval: bool | None


def instability_probability(
    model: ProbabilisticModel,
    *,
    mc_samples: int = 2000,
    pce_degree: int = 3,
    pce_size: int | None = None,
    surrogate_samples: int = 20000,
    confidence: float = 0.95,
    run_form: bool = True,
    seed: int | RandomStream | None = 0,
) -> InstabilityProbabilityResult:
    """Estimate ``P(growth_rate > 0)`` by Monte Carlo, PCE and FORM.

    * Monte Carlo draws ``mc_samples`` joint samples and solves the mode for each.
    * The surrogate route fits a Hermite polynomial chaos of ``growth_rate``
      (:func:`~probstat.surrogates.polynomial_chaos.fit_polynomial_chaos`,
      ``pce_size`` solves) and counts positive predictions on
      ``surrogate_samples`` independent samples.
    * FORM applies
      :func:`~probstat.reliability.model_based.form_hlrf` to ``g = 0 -
      growth_rate`` (failure ``g <= 0`` means unstable).

    FORM is exact only for a linear ``growth_rate`` in standard-normal space;
    when the failure boundary is curved, or the unstable region is not
    connected to one design point, the three estimates disagree and the
    ``*_within_interval`` flags say so. The comparison is reported, not
    forced.

    Args:
        model: Model from :func:`thermoacoustic_model` (needs ``growth_rate``
            and normal inputs).
        mc_samples: Monte Carlo sample size (positive).
        pce_degree: Total degree of the chaos expansion.
        pce_size: Surrogate training solves (default twice the term count).
        surrogate_samples: Number of cheap surrogate evaluations.
        confidence: Wilson interval confidence.
        run_form: Whether to run FORM.
        seed: Integer seed or stream; children ``"mc"``, ``"pce"`` and
            ``"surrogate-mc"`` are independent.

    Returns:
        :class:`InstabilityProbabilityResult`.

    Raises:
        TypeError: If ``model`` is not a :class:`ProbabilisticModel`.
        ValueError: If a count is not positive.
        ThermoacousticConvergenceError: If a mode solve fails for a sample.

    Equation:
        [EQ-AERO-CI-007].

    References:
        [REF-MAGRI-JUNIPER-2013], [REF-RAYLEIGH-1878].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    for label, count in (
        ("mc_samples", mc_samples),
        ("surrogate_samples", surrogate_samples),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{label} must be a positive integer")
    stream = resolve_stream(seed)
    points = model.inputs.sample(mc_samples, seed=stream.child("mc"))
    growth = model.evaluate_many(points)["growth_rate"]
    estimate = estimate_probability((g > 0.0 for g in growth), confidence=confidence)
    interval = (
        estimate.confidence_interval.lower,
        estimate.confidence_interval.upper,
    )
    mean = math.fsum(growth) / len(growth)
    variance = math.fsum((g - mean) ** 2 for g in growth) / max(1, len(growth) - 1)
    expansion = fit_polynomial_chaos(
        model,
        "growth_rate",
        degree=pce_degree,
        basis="hermite",
        size=pce_size,
        seed=stream.child("pce"),
    )
    cheap = model.inputs.sample(surrogate_samples, seed=stream.child("surrogate-mc"))
    surrogate = sum(1 for x in cheap if expansion.predict(x) > 0.0) / surrogate_samples
    form_probability = form_beta = form_converged = form_evaluations = None
    form_within = None
    if run_form:
        form = form_hlrf(ModelLimitState(model, "growth_rate", 0.0, failure="above"))
        form_probability = form.probability
        form_beta = form.beta
        form_converged = form.converged
        form_evaluations = form.evaluations
        form_within = interval[0] <= form.probability <= interval[1]
    return InstabilityProbabilityResult(
        estimate.probability,
        interval,
        mc_samples,
        mean,
        math.sqrt(variance),
        surrogate,
        expansion.loo_error,
        len(expansion.design),
        form_probability,
        form_beta,
        form_converged,
        form_evaluations,
        interval[0] <= surrogate <= interval[1],
        form_within,
    )
