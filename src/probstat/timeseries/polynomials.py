"""Lag polynomials: roots, stationarity / invertibility, and PACF parameterization.

Sign conventions used throughout :mod:`probstat.timeseries`:

* AR polynomial ``phi(z) = 1 - phi_1 z - ... - phi_p z^p``;
* MA polynomial ``theta(z) = 1 + theta_1 z + ... + theta_q z^q``.

An ARMA model is causal (stationary) when every root of ``phi`` lies strictly
outside the unit circle and invertible when every root of ``theta`` does.

Two independent routes answer the question. :func:`polynomial_roots` finds all
complex roots by the Durand-Kerner (Weierstrass) simultaneous iteration with
Newton polishing, so the roots can be reported. The decision itself uses the
Schur-Cohn step-down recursion (the inverse Durbin-Levinson map), which needs
no root finding: ``phi`` has all roots outside the unit circle exactly when
every reflection coefficient it produces lies in ``(-1, 1)``. The forward map
(:func:`pacf_to_coefficients`) is the Barndorff-Nielsen-Schou / Jones /
Monahan reparameterization that turns an unconstrained vector into a
stationary AR polynomial for maximum-likelihood optimization.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_non_negative_integer,
)

Vector = tuple[float, ...]


def _coefficients(values: Sequence[float], name: str) -> Vector:
    """Materialize a possibly empty finite coefficient vector.

    Args:
        values: Candidate coefficients.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats (possibly empty).

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    items = tuple(values)
    if not items:
        return ()
    return coerce_finite_sample(items, name=name)


def _horner(monic: Sequence[complex], z: complex) -> complex:
    """Evaluate ``z^n + a_(n-1) z^(n-1) + ... + a_0`` by Horner's rule.

    Args:
        monic: Lower coefficients ``a_0..a_(n-1)`` in ascending order.
        z: Evaluation point.

    Returns:
        Polynomial value.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    value: complex = 1.0 + 0.0j
    for coefficient in reversed(monic):
        value = value * z + coefficient
    return value


def _horner_derivative(monic: Sequence[complex], z: complex) -> complex:
    """Evaluate the derivative of a monic polynomial by Horner's rule.

    Args:
        monic: Lower coefficients ``a_0..a_(n-1)`` in ascending order.
        z: Evaluation point.

    Returns:
        Derivative value.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    n = len(monic)
    value: complex = complex(n)
    for power in range(n - 1, 0, -1):
        value = value * z + power * monic[power]
    return value


def polynomial_roots(
    coefficients: Sequence[float],
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 2000,
) -> tuple[complex, ...]:
    """Return all complex roots of ``c_0 + c_1 z + ... + c_n z^n``.

    The Durand-Kerner (Weierstrass) iteration
    ``z_k <- z_k - p(z_k) / prod_(j != k) (z_k - z_j)`` updates every root
    estimate simultaneously from points on a circle of radius ``|a_0|^(1/n)``
    (the geometric mean of the root moduli). Each converged root then receives
    Newton steps that are kept only when they reduce ``|p|``. Repeated roots
    converge linearly and are accurate to roughly ``sqrt(eps)``.

    Args:
        coefficients: Ascending real coefficients; trailing zeros are dropped.
        tolerance: Relative step size at which the iteration stops.
        max_iterations: Maximum number of sweeps.

    Returns:
        Roots sorted by modulus, then argument.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If the polynomial is identically zero, or the iteration
            fails to converge.

    Equation:
        [EQ-TS-ROOT-001].

    References:
        [REF-KERNER-1966], [REF-GOLUB-VANLOAN-2013, Section 7.4].

    """
    values = list(_coefficients(coefficients, "coefficients"))
    while values and values[-1] == 0.0:
        values.pop()
    if not values:
        raise ValueError("polynomial must not be identically zero")
    zero_roots = 0
    while values[0] == 0.0:
        values.pop(0)
        zero_roots += 1
    degree = len(values) - 1
    roots: list[complex] = [0j] * zero_roots
    if degree == 0:
        return tuple(roots)
    leading = values[-1]
    monic = [complex(value / leading) for value in values[:-1]]
    if degree == 1:
        roots.append(-monic[0])
        return _sorted_roots(roots)
    radius = abs(monic[0]) ** (1.0 / degree)
    estimates = [
        radius * cmath.exp(1j * (2.0 * math.pi * k / degree + 0.4))
        for k in range(degree)
    ]
    converged = False
    for _ in range(max_iterations):
        largest = 0.0
        for k in range(degree):
            zk = estimates[k]
            denominator: complex = 1.0 + 0.0j
            for j in range(degree):
                if j != k:
                    gap = zk - estimates[j]
                    if gap == 0.0:
                        gap = complex(1e-12 * max(1.0, abs(zk)), 1e-12)
                    denominator *= gap
            step = _horner(monic, zk) / denominator
            estimates[k] = zk - step
            largest = max(largest, abs(step) / max(1.0, abs(estimates[k])))
        if largest <= tolerance:
            converged = True
            break
    if not converged:
        residual = max(abs(_horner(monic, z)) for z in estimates)
        scale = 1.0 + max(abs(a) for a in monic)
        if residual > 1e-8 * scale:
            raise ValueError("Durand-Kerner iteration did not converge")
    for index, root in enumerate(estimates):
        current = root
        current_residual = abs(_horner(monic, current))
        for _ in range(3):
            derivative = _horner_derivative(monic, current)
            if derivative == 0.0:
                break
            candidate = current - _horner(monic, current) / derivative
            candidate_residual = abs(_horner(monic, candidate))
            if candidate_residual >= current_residual:
                break
            current, current_residual = candidate, candidate_residual
        if abs(current.imag) <= 1e-13 * max(1.0, abs(current)):
            current = complex(current.real, 0.0)
        estimates[index] = current
    roots.extend(estimates)
    return _sorted_roots(roots)


def _sorted_roots(roots: Sequence[complex]) -> tuple[complex, ...]:
    """Sort roots by modulus, then argument.

    Args:
        roots: Complex roots.

    Returns:
        Sorted tuple.

    References:
        Project presentation policy. No equation is evaluated.

    """
    return tuple(sorted(roots, key=lambda z: (round(abs(z), 12), cmath.phase(z))))


@dataclass(frozen=True, slots=True)
class RootCheck:
    """Root-based verdict for one lag polynomial.

    Args:
        roots: All complex roots of the lag polynomial.
        minimum_modulus: Smallest root modulus (``inf`` for a constant).
        reflection_coefficients: Schur-Cohn step-down coefficients ``r_1..r_p``
            (``None`` once a coefficient reaches the unit circle).
        satisfied: Whether every root lies strictly outside the unit circle,
            decided by the step-down recursion.

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-MONAHAN-1984].

    """

    roots: tuple[complex, ...]
    minimum_modulus: float
    reflection_coefficients: Vector | None
    satisfied: bool


def coefficients_to_pacf(ar: Sequence[float]) -> Vector | None:
    """Map AR coefficients to reflection (partial autocorrelation) coefficients.

    The Schur-Cohn step-down recursion
    ``phi_(k-1),j = (phi_kj + r_k phi_k,(k-j)) / (1 - r_k^2)`` with
    ``r_k = phi_kk`` inverts :func:`pacf_to_coefficients`.

    Args:
        ar: AR coefficients ``phi_1..phi_p`` (empty for order zero).

    Returns:
        ``(r_1, ..., r_p)`` when every ``|r_k| < 1`` (stationary), otherwise
        ``None``.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    Equation:
        [EQ-TS-ROOT-002], [EQ-TS-PARAM-001].

    References:
        [REF-MONAHAN-1984], [REF-BROCKWELL-DAVIS-2016, Chapter 2].

    """
    current = list(_coefficients(ar, "ar"))
    reflections: list[float] = []
    while current:
        k = len(current)
        r = current[-1]
        if not abs(r) < 1.0:
            return None
        reflections.append(r)
        denominator = 1.0 - r * r
        current = [
            (current[j] + r * current[k - 2 - j]) / denominator for j in range(k - 1)
        ]
    return tuple(reversed(reflections))


def pacf_to_coefficients(partial: Sequence[float]) -> Vector:
    """Map reflection coefficients in ``(-1, 1)`` to stationary AR coefficients.

    The Durbin-Levinson step-up ``phi_kk = r_k``,
    ``phi_kj = phi_(k-1),j - r_k phi_(k-1),(k-j)``.

    Args:
        partial: Partial autocorrelations ``r_1..r_p`` with ``|r_k| < 1``.

    Returns:
        AR coefficients whose polynomial has all roots outside the unit circle.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or outside ``(-1, 1)``.

    Equation:
        [EQ-TS-PARAM-001].

    References:
        [REF-JONES-1980], [REF-MONAHAN-1984].

    """
    reflections = _coefficients(partial, "partial")
    current: list[float] = []
    for r in reflections:
        if not abs(r) < 1.0:
            raise ValueError("partial autocorrelations must lie in (-1, 1)")
        k = len(current) + 1
        current = [current[j] - r * current[k - 2 - j] for j in range(k - 1)] + [r]
    return tuple(current)


def ar_root_check(ar: Sequence[float]) -> RootCheck:
    """Check causality (stationarity) of ``phi(z) = 1 - sum phi_j z^j``.

    Args:
        ar: AR coefficients ``phi_1..phi_p``.

    Returns:
        Roots, minimum modulus, reflection coefficients, and the verdict.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    Equation:
        [EQ-TS-ARMA-001], [EQ-TS-ROOT-001], [EQ-TS-ROOT-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-HAMILTON-1994, Chapter 3].

    """
    phi = _coefficients(ar, "ar")
    polynomial = (1.0,) + tuple(-value for value in phi)
    roots = polynomial_roots(polynomial)
    reflections = coefficients_to_pacf(phi)
    minimum = min((abs(z) for z in roots), default=math.inf)
    return RootCheck(roots, minimum, reflections, reflections is not None)


def ma_root_check(ma: Sequence[float]) -> RootCheck:
    """Check invertibility of ``theta(z) = 1 + sum theta_j z^j``.

    Args:
        ma: MA coefficients ``theta_1..theta_q``.

    Returns:
        Roots, minimum modulus, reflection coefficients, and the verdict.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    Equation:
        [EQ-TS-ARMA-001], [EQ-TS-ROOT-001], [EQ-TS-ROOT-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015, Chapter 3].

    """
    theta = _coefficients(ma, "ma")
    polynomial = (1.0,) + theta
    roots = polynomial_roots(polynomial)
    reflections = coefficients_to_pacf(tuple(-value for value in theta))
    minimum = min((abs(z) for z in roots), default=math.inf)
    return RootCheck(roots, minimum, reflections, reflections is not None)


def is_stationary(ar: Sequence[float]) -> bool:
    """Return whether an AR polynomial is causal (all roots outside |z| = 1).

    Args:
        ar: AR coefficients ``phi_1..phi_p``.

    Returns:
        ``True`` exactly when every step-down coefficient lies in ``(-1, 1)``.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    Equation:
        [EQ-TS-ROOT-002].

    References:
        [REF-MONAHAN-1984], [REF-BROCKWELL-DAVIS-2016, Chapter 3].

    """
    return coefficients_to_pacf(ar) is not None


def is_invertible(ma: Sequence[float]) -> bool:
    """Return whether an MA polynomial is invertible (all roots outside |z| = 1).

    Args:
        ma: MA coefficients ``theta_1..theta_q``.

    Returns:
        ``True`` exactly when every step-down coefficient of ``-theta`` lies in
        ``(-1, 1)``.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    Equation:
        [EQ-TS-ROOT-002].

    References:
        [REF-MONAHAN-1984], [REF-BROCKWELL-DAVIS-2016, Chapter 3].

    """
    theta = _coefficients(ma, "ma")
    return coefficients_to_pacf(tuple(-value for value in theta)) is not None


def psi_weights(ar: Sequence[float], ma: Sequence[float], count: int) -> Vector:
    """Return the MA(infinity) weights ``psi_0..psi_(count-1)`` of an ARMA model.

    ``psi_0 = 1`` and ``psi_j = theta_j + sum_{i=1}^{min(j,p)} phi_i psi_(j-i)``.

    Args:
        ar: AR coefficients ``phi_1..phi_p``.
        ma: MA coefficients ``theta_1..theta_q``.
        count: Number of weights to return (``>= 1``).

    Returns:
        Weights of ``theta(z) / phi(z)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``count`` is zero or a coefficient is non-finite.

    Equation:
        [EQ-TS-ARMA-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015, Chapter 3].

    """
    phi = _coefficients(ar, "ar")
    theta = _coefficients(ma, "ma")
    size = require_non_negative_integer(count, name="count")
    if size < 1:
        raise ValueError("count must be positive")
    weights = [1.0]
    for j in range(1, size):
        value = theta[j - 1] if j <= len(theta) else 0.0
        value += math.fsum(
            phi[i - 1] * weights[j - i] for i in range(1, min(j, len(phi)) + 1)
        )
        weights.append(value)
    return tuple(weights)
