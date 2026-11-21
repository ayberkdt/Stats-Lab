"""Mondrian (group-conditional) split conformal regression.

Split conformal coverage is marginal: it can hide over-coverage of one
subpopulation and under-coverage of another. Mondrian conformal prediction
(Vovk, Gammerman and Shafer 2005) partitions the observations into
categories and calibrates each category on its own scores. With a separate
quantile ``q_g`` per group label ``g``, computed from the ``n_g``
calibration points of that group,

``P(Y in C(X) | G = g) >= k_g / (n_g + 1) >= 1 - alpha``,
``k_g = ceil((1 - alpha)(n_g + 1))``,

for every group, provided the calibration and test points *within the
group* are exchangeable. The price is sample size: each group pays the
``+inf`` rule on its own count, so a group with fewer than
``ceil(1/alpha) - 1`` calibration points (or one never seen) gets the whole
line. Coverage is still not conditional on ``x`` within a group.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from probstat.linalg import Vector, as_vector

from .quantile import _alpha_fraction, conformal_quantile, conformal_rank
from .regression import (
    PointPredictor,
    PredictionInterval,
    PredictorLike,
    _calibration,
    _evaluate,
    _positive_scale,
    as_predictor,
)


@dataclass(frozen=True, slots=True)
class MondrianGroup:
    """Calibration summary of one group.

    Args:
        label: Group label.
        size: Number ``n_g`` of calibration points in the group.
        rank: Order index ``k_g = ceil((1 - alpha)(n_g + 1))``.
        quantile: ``k_g``-th smallest group score, ``+inf`` if ``k_g > n_g``.
        guaranteed_coverage: ``min(k_g, n_g + 1) / (n_g + 1)``, the
            group-conditional coverage for continuous scores (``1`` when
            the quantile is infinite).
        scores: Sorted calibration scores of the group.

    References:
        [EQ-CONF-011], [REF-VOVK-GAMMERMAN-SHAFER-2005].

    """

    label: Hashable
    size: int
    rank: int
    quantile: float
    guaranteed_coverage: float
    scores: Vector

    @property
    def finite(self) -> bool:
        """Return whether the group has enough points for a finite quantile.

        References:
            [EQ-CONF-001], [EQ-CONF-011].

        """
        return self.quantile != math.inf


@dataclass(frozen=True)
class MondrianConformalRegressor:
    """Group-conditional split conformal intervals ``mu(x) +/- q_g s(x)``.

    Args:
        predictor: Point predictor ``mu``.
        scale: Positive spread ``s`` for normalized scores, or ``None``.
        alpha: Miscoverage level.
        groups: Per-group summaries in order of first appearance.
        lookup: Position of each label in ``groups``.

    References:
        [EQ-CONF-011], [REF-VOVK-GAMMERMAN-SHAFER-2005].

    """

    predictor: PointPredictor
    scale: PointPredictor | None
    alpha: float
    groups: tuple[MondrianGroup, ...]
    lookup: dict[Hashable, int] = field(repr=False, compare=False)

    @property
    def group_sizes(self) -> dict[Hashable, int]:
        """Return the calibration count ``n_g`` of every group.

        References:
            [EQ-CONF-011]. No equation is evaluated.

        """
        return {group.label: group.size for group in self.groups}

    @property
    def infinite_groups(self) -> tuple[Hashable, ...]:
        """Return the labels whose quantile is ``+inf`` (too few points).

        References:
            [EQ-CONF-001], [EQ-CONF-011].

        """
        return tuple(group.label for group in self.groups if not group.finite)

    def quantile(self, group: Hashable) -> float:
        """Return the calibration quantile ``q_g`` of one group.

        Args:
            group: Group label.

        Returns:
            ``q_g``; ``+inf`` for a small group and for a label with no
            calibration point.

        Equation:
            [EQ-CONF-011].

        References:
            [REF-VOVK-GAMMERMAN-SHAFER-2005].

        """
        position = self.lookup.get(group)
        return math.inf if position is None else self.groups[position].quantile

    def interval(self, x: Sequence[float], group: Hashable) -> PredictionInterval:
        """Return the interval at one input for a point of the given group.

        Args:
            x: Input vector.
            group: Group label of the test point.

        Returns:
            ``[mu(x) - q_g s(x), mu(x) + q_g s(x)]``; the whole line when
            ``q_g`` is infinite.

        Raises:
            ValueError: If the predictor or scale returns an invalid value.

        Equation:
            [EQ-CONF-011].

        References:
            [REF-VOVK-GAMMERMAN-SHAFER-2005].

        """
        point = as_vector(x, name="x")
        center = _evaluate(self.predictor, point, "prediction")
        spread = 1.0 if self.scale is None else _positive_scale(self.scale, point)
        radius = self.quantile(group)
        if radius == math.inf:
            return PredictionInterval(-math.inf, math.inf, center)
        half = radius * spread
        return PredictionInterval(center - half, center + half, center)

    def intervals(
        self, xs: Sequence[Sequence[float]], groups: Sequence[Hashable]
    ) -> tuple[PredictionInterval, ...]:
        """Return intervals at many inputs with their group labels.

        Args:
            xs: Input vectors.
            groups: One label per input.

        Returns:
            One interval per input.

        Raises:
            ValueError: If the lengths differ.

        Equation:
            [EQ-CONF-011].

        References:
            [REF-VOVK-GAMMERMAN-SHAFER-2005].

        """
        if len(xs) != len(groups):
            raise ValueError("xs and groups must have equal length")
        return tuple(self.interval(x, g) for x, g in zip(xs, groups, strict=True))


def mondrian_conformal_regression(
    predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    groups: Sequence[Hashable],
    *,
    alpha: float = 0.1,
    scale: PredictorLike | None = None,
) -> MondrianConformalRegressor:
    """Calibrate one split conformal quantile per group label.

    The group label must be determined by the observation itself (a
    function of ``x``, or a recorded category) and not chosen after seeing
    the scores. Within each group the calibration and test points must be
    exchangeable; then coverage is at least ``1 - alpha`` conditionally on
    the group, marginally over that group's calibration points and test
    inputs. One group reproduces
    :func:`~probstat.conformal.split_conformal_regression`.

    Args:
        predictor: Point predictor fitted without the calibration data.
        x_calibration: Held-out input vectors.
        y_calibration: Held-out responses.
        groups: One hashable label per calibration point.
        alpha: Miscoverage level in ``(0, 1)``.
        scale: Optional positive spread ``s(x)`` for normalized scores.

    Returns:
        Regressor with per-group quantiles and sample sizes; inspect
        ``infinite_groups`` for groups too small for a finite interval.

    Raises:
        TypeError: If a predictor is not callable or a label is unhashable.
        ValueError: If the calibration set is empty or mismatched, or
            ``alpha`` is invalid.

    Equation:
        [EQ-CONF-001], [EQ-CONF-011].

    References:
        [REF-VOVK-GAMMERMAN-SHAFER-2005].

    """
    mu = as_predictor(predictor)
    spread = None if scale is None else as_predictor(scale)
    _alpha_fraction(alpha)
    design, responses = _calibration(x_calibration, y_calibration)
    if len(groups) != len(design):
        raise ValueError("groups must match the number of calibration points")
    by_group: dict[Hashable, list[float]] = {}
    for x, y, label in zip(design, responses, groups, strict=True):
        residual = abs(y - _evaluate(mu, x, "prediction"))
        score = residual if spread is None else residual / _positive_scale(spread, x)
        by_group.setdefault(label, []).append(score)
    summaries: list[MondrianGroup] = []
    for label, scores in by_group.items():
        ordered = tuple(sorted(scores))
        size = len(ordered)
        rank = conformal_rank(size, alpha)
        summaries.append(
            MondrianGroup(
                label,
                size,
                rank,
                conformal_quantile(ordered, alpha),
                rank / (size + 1),
                ordered,
            )
        )
    return MondrianConformalRegressor(
        mu,
        spread,
        float(alpha),
        tuple(summaries),
        {group.label: position for position, group in enumerate(summaries)},
    )
