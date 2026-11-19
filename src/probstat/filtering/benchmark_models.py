"""Lorenz-63 and Lorenz-96 test systems as nonlinear state-space models.

Ensemble data assimilation is usually tested on two chaotic toy systems:

* the three-variable convection model of Lorenz (1963)
  [EQ-FILT-LORENZ-001], whose classical parameters ``sigma = 10``,
  ``rho = 28``, ``beta = 8/3`` give the butterfly attractor, and
* the cyclic "atmosphere on a latitude circle" of Lorenz (1996) and
  Lorenz & Emanuel (1998) [EQ-FILT-LORENZ-002], conventionally with
  ``J = 40`` variables, forcing ``F = 8``, and an assimilation interval of
  ``0.05`` time units (six hours).

The factories below wrap them as :class:`NonlinearStateSpace` objects whose
transition is ``substeps`` classical Runge-Kutta steps of size ``step``
([EQ-NUM-ODE-003]) over one assimilation interval, with an observation of a
subset of the components. The default model is perfect (``Q = 0``), so a
twin experiment simulates the truth with the same transition the filter
uses. The transition Jacobian is the exact derivative of the discrete
Runge-Kutta map, obtained by forward-mode automatic differentiation; the
observation Jacobian is the selection matrix. The raw right-hand sides are
exposed for direct integration with
:func:`probstat.numerics.integrate_fixed_step`.

References:
    [REF-LORENZ-1963], [REF-LORENZ-2006], [REF-LORENZ-EMANUEL-1998],
    [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from probstat.linalg import Matrix, Vector
from probstat.numerics import integrate_fixed_step, jacobian

from .state_space import NonlinearStateSpace

LORENZ63_STEP = 0.01
"""Integration step ``0.01`` used by Lorenz (1963)."""

LORENZ96_STEP = 0.05
"""Runge-Kutta step ``0.05`` (six hours) of Lorenz & Emanuel (1998)."""


def lorenz63_rhs(
    t: float,
    state: Sequence[float],
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> tuple[float, float, float]:
    """Return the Lorenz-63 tendency ``(dx/dt, dy/dt, dz/dt)``.

    ``dx/dt = sigma (y - x)``, ``dy/dt = x (rho - z) - y``,
    ``dz/dt = x y - beta z`` [EQ-FILT-LORENZ-001]. The time is ignored (the
    system is autonomous); it is accepted so the function can be passed to
    :func:`probstat.numerics.integrate_fixed_step`. Only arithmetic
    operators are used, so automatic-differentiation variables pass through.

    Args:
        t: Time (unused).
        state: ``(x, y, z)``.
        sigma: Prandtl number ``sigma``.
        rho: Scaled Rayleigh number ``rho`` (Lorenz's ``r``).
        beta: Geometric factor ``beta`` (Lorenz's ``b``).

    Returns:
        The tendency.

    Raises:
        ValueError: If the state does not have three components.

    Equation:
        [EQ-FILT-LORENZ-001].

    References:
        [REF-LORENZ-1963].

    """
    if len(state) != 3:
        raise ValueError("the Lorenz-63 state has three components")
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def lorenz96_rhs(
    t: float, state: Sequence[float], *, forcing: float = 8.0
) -> tuple[float, ...]:
    """Return the Lorenz-96 tendency of every component.

    ``dx_i/dt = (x_(i+1) - x_(i-2)) x_(i-1) - x_i + F`` with cyclic indices
    [EQ-FILT-LORENZ-002]. The time is ignored; only arithmetic operators are
    used, so automatic-differentiation variables pass through.

    Args:
        t: Time (unused).
        state: ``(x_0, ..., x_(J-1))`` with ``J >= 4``.
        forcing: Constant forcing ``F``.

    Returns:
        The tendency.

    Raises:
        ValueError: If the state has fewer than four components.

    Equation:
        [EQ-FILT-LORENZ-002].

    References:
        [REF-LORENZ-2006], [REF-LORENZ-EMANUEL-1998].

    """
    size = len(state)
    if size < 4:
        raise ValueError("the Lorenz-96 state needs at least four components")
    return tuple(
        (state[(i + 1) % size] - state[i - 2]) * state[i - 1] - state[i] + forcing
        for i in range(size)
    )


def _positive_real(value: float, name: str) -> float:
    """Return a finite positive float.

    Raises:
        ValueError: If ``value`` is not finite and positive.

    References:
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1]. No equation is
        evaluated.

    """
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Return an integer of at least ``minimum``.

    Raises:
        ValueError: If ``value`` is not such an integer.

    References:
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _diagonal(size: int, value: float) -> Matrix:
    """Return ``value * I`` of the given size.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(
        tuple(value if i == j else 0.0 for j in range(size)) for i in range(size)
    )


Rhs = Callable[[float, Sequence[float]], Sequence[float]]


def _assemble(
    rhs: Rhs,
    size: int,
    *,
    step: float,
    substeps: int,
    components: tuple[int, ...],
    process_covariance: Sequence[Sequence[float]] | None,
    observation_covariance: Sequence[Sequence[float]] | None,
    initial_mean: Sequence[float],
    initial_covariance: Sequence[Sequence[float]] | None,
) -> NonlinearStateSpace:
    """Wrap a right-hand side as an observed Runge-Kutta state-space model.

    Args:
        rhs: Right-hand side ``f(t, y)``.
        size: State dimension.
        step: Runge-Kutta step.
        substeps: Steps per assimilation interval.
        components: Observed state indices, in observation order.
        process_covariance: ``Q`` or ``None`` for the zero matrix.
        observation_covariance: ``R`` or ``None`` for the identity.
        initial_mean: Prior mean.
        initial_covariance: ``P_0`` or ``None`` for the identity.

    Returns:
        The model.

    References:
        [EQ-NUM-ODE-003], [EQ-FILT-SSM-002],
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    h = _positive_real(step, "step")
    count = _positive_int(substeps, "substeps")
    function = rhs

    def transition(state: Vector) -> Vector:
        """Advance one assimilation interval by classical Runge-Kutta steps.

        References:
            [EQ-NUM-ODE-003], [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

        """
        return integrate_fixed_step(
            function, 0.0, state, step=h, steps=count, method="rk4"
        ).final_state

    def transition_jacobian(state: Vector) -> Matrix:
        """Return the exact Jacobian of the discrete Runge-Kutta map.

        References:
            [EQ-NUM-ODE-003], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """

        def discrete(point: Any) -> Any:
            """Return the Runge-Kutta map of a forward-mode dual point.

            References:
                [EQ-NUM-ODE-003], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

            """
            return integrate_fixed_step(
                function, 0.0, point, step=h, steps=count, method="rk4"
            ).final_state

        return jacobian(discrete, state)

    def observation(state: Vector) -> Vector:
        """Return the observed components.

        References:
            [EQ-FILT-SSM-002].

        """
        return tuple(state[index] for index in components)

    selection = tuple(
        tuple(1.0 if column == index else 0.0 for column in range(size))
        for index in components
    )

    def observation_jacobian(state: Vector) -> Matrix:
        """Return the selection matrix of the observed components.

        References:
            [EQ-FILT-SSM-002].

        """
        return selection

    observed = len(components)
    return NonlinearStateSpace(
        transition,
        observation,
        _diagonal(size, 0.0) if process_covariance is None else process_covariance,
        _diagonal(observed, 1.0)
        if observation_covariance is None
        else observation_covariance,
        initial_mean,
        _diagonal(size, 1.0) if initial_covariance is None else initial_covariance,
        state_dimension=size,
        observation_dimension=observed,
        transition_jacobian_function=transition_jacobian,
        observation_jacobian_function=observation_jacobian,
    )


def lorenz63(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    *,
    step: float = LORENZ63_STEP,
    substeps: int = 1,
    observed_components: Sequence[int] = (0, 1, 2),
    process_covariance: Sequence[Sequence[float]] | None = None,
    observation_covariance: Sequence[Sequence[float]] | None = None,
    initial_mean: Sequence[float] = (0.0, 1.0, 0.0),
    initial_covariance: Sequence[Sequence[float]] | None = None,
) -> NonlinearStateSpace:
    """Return the Lorenz-63 system as an observed state-space model.

    The transition advances [EQ-FILT-LORENZ-001] by ``substeps`` classical
    Runge-Kutta steps of size ``step`` ([EQ-NUM-ODE-003]), so one assimilation
    interval lasts ``step * substeps`` time units. Lorenz (1963) used the
    step ``0.01`` (with a second-order scheme) and the start ``(0, 1, 0)``,
    which are the defaults here.

    Args:
        sigma: Prandtl number ``sigma``.
        rho: Scaled Rayleigh number ``rho``.
        beta: Geometric factor ``beta``.
        step: Runge-Kutta step size.
        substeps: Runge-Kutta steps per assimilation interval.
        observed_components: Distinct state indices observed directly, in
            observation order.
        process_covariance: ``Q``; ``None`` is the zero matrix (perfect model).
        observation_covariance: ``R``; ``None`` is the identity.
        initial_mean: Prior mean ``m_0``.
        initial_covariance: ``P_0``; ``None`` is the identity.

    Returns:
        The model, with the exact Jacobian of the discrete transition.

    Raises:
        ValueError: If a parameter, the step, the number of substeps, the
            observed components, or a covariance is invalid.

    Equation:
        [EQ-FILT-LORENZ-001], [EQ-NUM-ODE-003].

    References:
        [REF-LORENZ-1963], [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    parameters = tuple(float(value) for value in (sigma, rho, beta))
    if not all(map(math.isfinite, parameters)):
        raise ValueError("sigma, rho, and beta must be finite")
    s, r, b = parameters
    components = _components(observed_components, 3)

    def rhs(t: float, state: Sequence[float]) -> tuple[float, float, float]:
        """Return the Lorenz-63 tendency with the chosen parameters.

        References:
            [EQ-FILT-LORENZ-001], [REF-LORENZ-1963].

        """
        return lorenz63_rhs(t, state, sigma=s, rho=r, beta=b)

    return _assemble(
        rhs,
        3,
        step=step,
        substeps=substeps,
        components=components,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
    )


def lorenz96(
    dimension: int = 40,
    forcing: float = 8.0,
    *,
    step: float = LORENZ96_STEP,
    substeps: int = 1,
    observe_every: int = 1,
    process_covariance: Sequence[Sequence[float]] | None = None,
    observation_covariance: Sequence[Sequence[float]] | None = None,
    initial_mean: Sequence[float] | None = None,
    initial_covariance: Sequence[Sequence[float]] | None = None,
) -> NonlinearStateSpace:
    """Return the Lorenz-96 system as an observed state-space model.

    The transition advances [EQ-FILT-LORENZ-002] by ``substeps`` classical
    Runge-Kutta steps of size ``step``; the defaults ``J = 40``, ``F = 8``,
    and one step of ``0.05`` (six hours) are the setting of Lorenz &
    Emanuel (1998). Components ``0, k, 2k, ...`` with ``k = observe_every``
    are observed directly; their grid coordinates are their indices, on a
    periodic domain of length ``J``.

    Args:
        dimension: Number of variables ``J >= 4``.
        forcing: Constant forcing ``F``.
        step: Runge-Kutta step size.
        substeps: Runge-Kutta steps per assimilation interval.
        observe_every: Observe every ``k``-th component, ``1 <= k <= J``.
        process_covariance: ``Q``; ``None`` is the zero matrix (perfect model).
        observation_covariance: ``R``; ``None`` is the identity.
        initial_mean: Prior mean; ``None`` is the equilibrium ``x_i = F``
            with ``x_0`` raised by ``0.01``, which leaves the unstable
            equilibrium; spin the system up before using it as a truth.
        initial_covariance: ``P_0``; ``None`` is the identity.

    Returns:
        The model, with the exact Jacobian of the discrete transition.

    Raises:
        ValueError: If the dimension, forcing, step, number of substeps,
            observation spacing, or a covariance is invalid.

    Equation:
        [EQ-FILT-LORENZ-002], [EQ-NUM-ODE-003].

    References:
        [REF-LORENZ-2006], [REF-LORENZ-EMANUEL-1998],
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    size = _positive_int(dimension, "dimension", minimum=4)
    force = float(forcing)
    if not math.isfinite(force):
        raise ValueError("forcing must be finite")
    spacing = _positive_int(observe_every, "observe_every")
    if spacing > size:
        raise ValueError("observe_every must not exceed the dimension")
    if initial_mean is None:
        mean: Sequence[float] = tuple(
            force + (0.01 if i == 0 else 0.0) for i in range(size)
        )
    else:
        mean = initial_mean

    def rhs(t: float, state: Sequence[float]) -> tuple[float, ...]:
        """Return the Lorenz-96 tendency with the chosen forcing.

        References:
            [EQ-FILT-LORENZ-002], [REF-LORENZ-EMANUEL-1998].

        """
        return lorenz96_rhs(t, state, forcing=force)

    return _assemble(
        rhs,
        size,
        step=step,
        substeps=substeps,
        components=tuple(range(0, size, spacing)),
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        initial_mean=mean,
        initial_covariance=initial_covariance,
    )


def _components(values: Sequence[int], size: int) -> tuple[int, ...]:
    """Validate observed component indices.

    Raises:
        ValueError: If the indices are empty, repeated, or out of range.

    References:
        [EQ-FILT-SSM-002]. No equation is evaluated.

    """
    components = tuple(values)
    if not components:
        raise ValueError("at least one component must be observed")
    for index in components:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("observed components must be integers")
        if not 0 <= index < size:
            raise ValueError(f"observed component {index} is out of range")
    if len(set(components)) != len(components):
        raise ValueError("observed components must be distinct")
    return components
