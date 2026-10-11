"""Consequence (loss) models and model-based risk assessment.

A :class:`LossFunction` maps a quantity-of-interest value to a monetary or
utility loss. :func:`assess_risk` propagates a
:class:`~probstat.modeling.ProbabilisticModel`, applies the loss to one QoI,
and summarizes the resulting loss distribution with expected loss, VaR, CVaR,
an exceedance curve, and the probability of any loss.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real
from probstat.foundations.random_streams import RandomStream
from probstat.modeling import ProbabilisticModel
from probstat.monte_carlo import MonteCarloEstimate, estimate_mean
from probstat.propagation import PropagationStudy, propagate

from .measures import (
    _cvar_sorted,
    _lower_quantile,
    _require_level,
    _sorted_weighted,
    loss_exceedance_curve,
)

LOSS_KINDS = ("step", "linear", "piecewise_linear", "custom")


@dataclass(frozen=True)
class LossFunction:
    """A deterministic consequence model ``c(y)`` from QoI value to loss.

    Args:
        function: Callable returning a finite loss for a finite QoI value.
        kind: One of :data:`LOSS_KINDS`, recorded for reporting.
        description: Human-readable meaning and units of the loss.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

    """

    function: Callable[[float], float]
    kind: str = "custom"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the callable and kind.

        Raises:
            TypeError: If ``function`` is not callable.
            ValueError: If ``kind`` is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not callable(self.function):
            raise TypeError("function must be callable")
        if self.kind not in LOSS_KINDS:
            raise ValueError(f"kind must be one of {LOSS_KINDS}")

    def __call__(self, value: float) -> float:
        """Evaluate the loss at one QoI value.

        Args:
            value: Finite QoI value.

        Returns:
            Finite loss.

        Raises:
            ValueError: If the input or the returned loss is not finite.

        Equation:
            [EQ-RISK-LOSS-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        y = require_finite_real(value, name="value")
        return require_finite_real(self.function(y), name="loss")

    def apply(self, values: Iterable[float]) -> tuple[float, ...]:
        """Evaluate the loss at many QoI values.

        Args:
            values: Finite QoI values.

        Returns:
            Losses in input order.

        Equation:
            [EQ-RISK-LOSS-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        return tuple(self(value) for value in values)


def step_loss(
    threshold: float, consequence: float, *, inclusive: bool = False
) -> LossFunction:
    """Return the all-or-nothing loss ``c 1{y > t}`` (or ``y >= t``).

    Args:
        threshold: Finite QoI threshold ``t``.
        consequence: Finite loss incurred when the threshold is exceeded.
        inclusive: Count ``y == t`` as exceedance.

    Returns:
        Step loss function.

    Equation:
        [EQ-RISK-LOSS-001].

    References:
        [REF-KAPLAN-GARRICK-1981].

    """
    t = require_finite_real(threshold, name="threshold")
    c = require_finite_real(consequence, name="consequence")

    def function(y: float) -> float:
        """Evaluate the step loss.

        References:
            [EQ-RISK-LOSS-001], [REF-KAPLAN-GARRICK-1981].

        """
        exceeded = y >= t if inclusive else y > t
        return c if exceeded else 0.0

    relation = ">=" if inclusive else ">"
    return LossFunction(function, "step", f"{c:g} if y {relation} {t:g}, else 0")


def linear_loss(
    threshold: float, rate: float, *, cap: float | None = None
) -> LossFunction:
    """Return the hinge loss ``min(cap, rate * max(0, y - t))``.

    Args:
        threshold: Finite QoI level where losses start.
        rate: Non-negative loss per unit of exceedance.
        cap: Optional non-negative maximum loss.

    Returns:
        Linear-above-threshold loss function.

    Raises:
        ValueError: If ``rate`` or ``cap`` is negative.

    Equation:
        [EQ-RISK-LOSS-001].

    References:
        [REF-KAPLAN-GARRICK-1981].

    """
    t = require_finite_real(threshold, name="threshold")
    slope = require_finite_real(rate, name="rate")
    if slope < 0.0:
        raise ValueError("rate must be non-negative")
    limit = None if cap is None else require_finite_real(cap, name="cap")
    if limit is not None and limit < 0.0:
        raise ValueError("cap must be non-negative")

    def function(y: float) -> float:
        """Evaluate the capped hinge loss.

        References:
            [EQ-RISK-LOSS-001], [REF-KAPLAN-GARRICK-1981].

        """
        loss = slope * max(0.0, y - t)
        return loss if limit is None else min(limit, loss)

    suffix = "" if limit is None else f", capped at {limit:g}"
    return LossFunction(function, "linear", f"{slope:g} * max(0, y - {t:g}){suffix}")


def piecewise_linear_loss(knots: Sequence[tuple[float, float]]) -> LossFunction:
    """Return a loss interpolated linearly between ``(y_i, loss_i)`` knots.

    Outside the knot range the loss is held constant at the end values.

    Args:
        knots: At least two pairs with strictly increasing ``y_i``.

    Returns:
        Piecewise-linear loss function.

    Raises:
        ValueError: If fewer than two knots are given or ``y_i`` do not
            strictly increase.

    Equation:
        [EQ-RISK-LOSS-001].

    References:
        [REF-KAPLAN-GARRICK-1981].

    """
    points = tuple(
        (
            require_finite_real(x, name="knot level"),
            require_finite_real(loss, name="knot loss"),
        )
        for x, loss in knots
    )
    if len(points) < 2:
        raise ValueError("at least two knots are required")
    for (x0, _), (x1, _) in zip(points, points[1:], strict=False):
        if not x1 > x0:
            raise ValueError("knot levels must strictly increase")

    def function(y: float) -> float:
        """Interpolate the knot table at one QoI value.

        References:
            [EQ-RISK-LOSS-001], [REF-KAPLAN-GARRICK-1981].

        """
        if y <= points[0][0]:
            return points[0][1]
        if y >= points[-1][0]:
            return points[-1][1]
        for (x0, l0), (x1, l1) in zip(points, points[1:], strict=False):
            if y <= x1:
                return l0 + (l1 - l0) * (y - x0) / (x1 - x0)
        return points[-1][1]

    return LossFunction(
        function, "piecewise_linear", f"piecewise linear on {len(points)} knots"
    )


def custom_loss(
    function: Callable[[float], float], description: str = ""
) -> LossFunction:
    """Wrap an arbitrary callable as a validated loss function.

    Args:
        function: Callable returning a finite loss.
        description: Meaning and units of the loss.

    Returns:
        Custom loss function.

    Equation:
        [EQ-RISK-LOSS-001].

    References:
        [REF-KAPLAN-GARRICK-1981].

    """
    return LossFunction(function, "custom", description)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Loss distribution and risk measures of one propagated quantity.

    Args:
        quantity: Assessed QoI name.
        loss_kind: Kind of the applied loss function.
        losses: Losses in propagation-design order.
        expected_loss: Mean loss with standard error and interval (the
            interval assumes independent draws; for LHS/QMC it is a
            conservative Monte Carlo reference).
        value_at_risk: Read-only level-to-VaR mapping (lower quantile).
        conditional_value_at_risk: Read-only level-to-CVaR mapping.
        exceedance_curve: Pairs ``(l, P(L > l))`` on an even loss grid.
        probability_of_loss: Empirical ``P(L > 0)``.
        model_evaluations: Forward-model calls charged to the assessment.
        study: Underlying propagation study.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-ROCKAFELLAR-URYASEV-2002],
        [REF-SMITH-UQ-2024, Chapter 4].

    """

    quantity: str
    loss_kind: str
    losses: tuple[float, ...]
    expected_loss: MonteCarloEstimate
    value_at_risk: Mapping[float, float]
    conditional_value_at_risk: Mapping[float, float]
    exceedance_curve: tuple[tuple[float, float], ...]
    probability_of_loss: float
    model_evaluations: int
    study: PropagationStudy

    def exceedance_probability(self, level: float) -> float:
        """Return the empirical loss exceedance probability ``P(L > level)``.

        Args:
            level: Finite loss level.

        Returns:
            Exceedance probability.

        Equation:
            [EQ-RISK-EXC-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        threshold = require_finite_real(level, name="level")
        count = sum(1 for loss in self.losses if loss > threshold)
        return count / len(self.losses)


def assess_risk(
    model: ProbabilisticModel,
    quantity: str,
    loss: LossFunction | Callable[[float], float],
    size: int,
    *,
    design: str = "mc",
    seed: int | RandomStream | None = None,
    levels: Sequence[float] = (0.9, 0.95, 0.99),
    exceedance_points: int = 50,
) -> RiskAssessment:
    """Propagate a probabilistic model and summarize the risk of one QoI.

    Args:
        model: Probabilistic model shared with other UQ workflows.
        quantity: Name of the QoI whose consequences are assessed.
        loss: Loss function, or a plain callable wrapped as a custom loss.
        size: Number of propagated input realizations, at least two.
        design: Propagation design (``"mc"``, ``"lhs"``, ``"halton"``,
            ``"sobol"``); see :func:`probstat.propagation.propagate`.
        seed: Integer seed or stream for the propagation design.
        levels: Non-empty VaR/CVaR levels strictly inside ``(0, 1)``.
        exceedance_points: Number of evenly spaced loss levels, at least two.

    Returns:
        Risk assessment.

    Raises:
        KeyError: If ``quantity`` is not a declared QoI.
        ValueError: If ``size``, ``levels``, or ``exceedance_points`` are
            invalid.

    Equation:
        [EQ-UQ-001], [EQ-RISK-LOSS-001], [EQ-RISK-EL-001],
        [EQ-RISK-VAR-001], [EQ-RISK-CVAR-001], [EQ-RISK-EXC-001].

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-ROCKAFELLAR-URYASEV-2002],
        [REF-SMITH-UQ-2024, Chapter 4].

    """
    model.quantity(quantity)
    consequence = loss if isinstance(loss, LossFunction) else custom_loss(loss)
    requested = tuple(_require_level(level, name="level") for level in levels)
    if not requested:
        raise ValueError("levels must not be empty")
    if (
        isinstance(exceedance_points, bool)
        or not isinstance(exceedance_points, int)
        or exceedance_points < 2
    ):
        raise ValueError("exceedance_points must be an integer of at least two")
    study = propagate(model, size, design=design, seed=seed)
    losses = consequence.apply(study[quantity].values)
    ordered, weights = _sorted_weighted(losses, None)
    var = {level: _lower_quantile(ordered, weights, level) for level in requested}
    cvar = {level: _cvar_sorted(ordered, weights, level) for level in requested}
    low, high = ordered[0], ordered[-1]
    grid = (
        (low,)
        if low == high
        else tuple(
            low + (high - low) * k / (exceedance_points - 1)
            for k in range(exceedance_points)
        )
    )
    return RiskAssessment(
        quantity,
        consequence.kind,
        losses,
        estimate_mean(losses),
        MappingProxyType(var),
        MappingProxyType(cvar),
        loss_exceedance_curve(losses, grid),
        sum(1 for value in losses if value > 0.0) / len(losses),
        study.model_evaluations,
        study,
    )
