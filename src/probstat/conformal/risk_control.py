"""Conformal risk control: calibrating a parameter to bound an expected loss.

Split conformal prediction bounds the probability of one event, a miss.
Conformal risk control (Angelopoulos, Bates, Fisch, Lei and Schuster 2024)
bounds the expectation of any loss that is bounded and non-increasing in a
tuning parameter ``lambda`` (larger ``lambda`` means larger, more
conservative prediction sets). With calibration losses ``L_1, ..., L_n`` as
functions of ``lambda``, empirical risk ``Rhat_n(lambda) = mean_i
L_i(lambda)``, and ``L_i <= B``,

``lambda_hat = inf{lambda : n/(n+1) Rhat_n(lambda) + B/(n+1) <= alpha}``

satisfies ``E[L_{n+1}(lambda_hat)] <= alpha`` when the ``n + 1`` loss
functions are exchangeable. The expectation is marginal: over the
calibration set and the test point.

This module works on a finite, user-chosen grid of ``lambda`` values and
returns the smallest grid value satisfying the inequality. The guarantee is
unchanged (the proof needs only monotonicity on the set searched over), but
the result can be more conservative than the infimum over a continuum, by at
most one grid step. With the miscoverage loss ``1{s_i > lambda}``, ``B = 1``
and a grid containing the calibration scores, ``lambda_hat`` is exactly the
split conformal quantile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector


@dataclass(frozen=True, slots=True)
class ConformalRiskControlResult:
    """Calibrated parameter and risk curve of conformal risk control.

    Args:
        lambda_hat: Smallest grid value meeting the risk condition, or the
            largest grid value when none does (then ``feasible`` is false).
        index: Position of ``lambda_hat`` in ``lambdas``.
        feasible: Whether some grid value met the condition. When false the
            bound ``E[L(lambda_hat)] <= alpha`` holds only if the loss at
            the largest grid value is at most ``alpha`` for every
            observation.
        empirical_risk: ``Rhat_n(lambda_hat)``.
        adjusted_risk: ``n/(n+1) Rhat_n(lambda_hat) + B/(n+1)``.
        alpha: Target risk level.
        bound: Loss upper bound ``B``.
        size: Number ``n`` of calibration observations.
        lambdas: The increasing grid.
        risks: ``Rhat_n`` at every grid value.

    References:
        [EQ-CONF-008], [REF-ANGELOPOULOS-ETAL-2024].

    """

    lambda_hat: float
    index: int
    feasible: bool
    empirical_risk: float
    adjusted_risk: float
    alpha: float
    bound: float
    size: int
    lambdas: Vector
    risks: Vector

    @property
    def adjusted_risks(self) -> Vector:
        """Return ``n/(n+1) Rhat_n(lambda) + B/(n+1)`` at every grid value.

        References:
            [EQ-CONF-008], [REF-ANGELOPOULOS-ETAL-2024].

        """
        n = self.size
        return tuple((n * risk + self.bound) / (n + 1) for risk in self.risks)


def conformal_risk_control(
    losses: Sequence[Sequence[float]],
    lambdas: Sequence[float],
    *,
    alpha: float = 0.1,
    bound: float = 1.0,
) -> ConformalRiskControlResult:
    """Choose the smallest grid ``lambda`` whose adjusted risk is at most ``alpha``.

    ``losses[i][j]`` is the loss ``L_i(lambda_j)`` of calibration observation
    ``i`` at grid value ``lambda_j``. Every row must be non-increasing along
    the increasing grid and bounded above by ``bound``; both are checked,
    and a violation raises instead of being repaired. The returned
    ``lambda_hat`` is the smallest ``lambda_j`` with
    ``(sum_i L_i(lambda_j) + B) / (n + 1) <= alpha``; the comparison is made
    in exact rational arithmetic on the correctly rounded loss sum and the
    decimal text of ``alpha``.

    If the calibration and test loss functions are exchangeable then
    ``E[L_{n+1}(lambda_hat)] <= alpha``, marginally over the calibration set
    and the test point, for losses evaluated on this grid. If no grid value
    qualifies, the largest one is returned with ``feasible=False``; the
    bound then holds only if the loss at that value never exceeds ``alpha``
    (for example a set-valued predictor that returns everything at
    ``lambda_max``).

    Args:
        losses: ``n`` rows of ``m`` finite losses.
        lambdas: ``m`` strictly increasing finite grid values.
        alpha: Target risk level (finite).
        bound: Upper bound ``B`` of the loss (finite).

    Returns:
        Calibrated parameter, feasibility flag, and the risk curve.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the grid is empty or not strictly increasing, the
            loss table is empty or ragged, a loss exceeds ``bound``, or a
            row increases along the grid.

    Equation:
        [EQ-CONF-008].

    References:
        [REF-ANGELOPOULOS-ETAL-2024].

    """
    grid = tuple(require_finite_real(value, name="lambda") for value in lambdas)
    if not grid:
        raise ValueError("lambdas must not be empty")
    if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError("lambdas must be strictly increasing")
    level = require_finite_real(alpha, name="alpha")
    cap = require_finite_real(bound, name="bound")
    n = len(losses)
    if n == 0:
        raise ValueError("at least one calibration loss row is required")
    table: list[Vector] = []
    for i, row in enumerate(losses):
        values = tuple(require_finite_real(v, name=f"losses[{i}]") for v in row)
        if len(values) != len(grid):
            raise ValueError(f"losses[{i}] must have one value per lambda")
        if max(values) > cap:
            raise ValueError(f"losses[{i}] exceeds bound")
        if any(b > a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError(
                f"losses[{i}] is not non-increasing in lambda; conformal risk "
                "control requires monotone losses"
            )
        table.append(values)
    sums = tuple(math.fsum(row[j] for row in table) for j in range(len(grid)))
    target = Fraction(str(level)) * (n + 1)
    extra = Fraction(cap)
    index = next(
        (j for j, total in enumerate(sums) if Fraction(total) + extra <= target),
        None,
    )
    feasible = index is not None
    chosen = len(grid) - 1 if index is None else index
    return ConformalRiskControlResult(
        lambda_hat=grid[chosen],
        index=chosen,
        feasible=feasible,
        empirical_risk=sums[chosen] / n,
        adjusted_risk=(sums[chosen] + cap) / (n + 1),
        alpha=level,
        bound=cap,
        size=n,
        lambdas=grid,
        risks=tuple(total / n for total in sums),
    )
