"""Adaptive conformal inference for sequential prediction under distribution shift.

Split conformal coverage needs exchangeability. When the data-generating law
drifts, a fixed miscoverage level ``alpha`` can under-cover for long
stretches. Adaptive conformal inference (ACI) tracks one parameter instead:
the level ``alpha_t`` at which the conformal quantile of the past scores is
taken. After each observation it moves ``alpha_t`` against the realized
miscoverage,

``alpha_{t+1} = alpha_t + gamma (alpha - err_t)``,

so ``alpha_t`` shrinks after misses (wider sets) and grows after covers
(narrower sets). Whatever the data sequence, even an adversarial one, the
long-run miscoverage frequency satisfies the deterministic bound
``|mean(err_1..err_T) - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) /
(gamma T)``. It is a frequency guarantee over time, not a conditional
coverage guarantee at each step.

Scores are absolute residuals ``|y_t - yhat_t|`` or residuals scaled by a
positive spread ``s_t``. The quantile at level ``1 - alpha_t`` of the score
history ``H`` (optional calibration scores, then every past score, or only
the most recent ``window``) is the conformal order statistic
``k = ceil((1 - alpha_t)(|H| + 1))``: ``+inf`` (the whole line) when
``k > |H|`` or ``alpha_t <= 0``, and ``-inf`` (the empty set) when
``alpha_t >= 1``.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

from .quantile import conformal_rank
from .regression import PredictionInterval


def _level(value: object, name: str) -> float:
    """Validate a miscoverage level strictly inside ``(0, 1)``.

    References:
        [REF-GIBBS-CANDES-2021, Section 2]. No equation is evaluated.

    """
    level = require_finite_real(value, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return level


def _positive(value: object, name: str) -> float:
    """Validate a finite, strictly positive real.

    References:
        [REF-GIBBS-CANDES-2021, Section 1.1]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


class AdaptiveConformalPredictor:
    """Online adaptive conformal intervals ``yhat_t +/- q_t s_t``.

    Call :meth:`interval` before the response is revealed and :meth:`update`
    after; ``update`` scores the response against the interval that was in
    force, records the miss indicator, adds the score to the history, and
    applies the ACI step to ``alpha_t``.

    Args:
        alpha: Target miscoverage level in ``(0, 1)``.
        gamma: Non-negative step size; ``0`` freezes ``alpha_t`` and gives
            ordinary online split conformal prediction.
        calibration_scores: Optional non-negative initial scores.
        window: Optional positive number of most recent scores to keep.
        initial_alpha: Starting level ``alpha_1`` in ``[0, 1]``; defaults to
            ``alpha``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a level, step, score, or window is invalid.

    References:
        [EQ-CONF-ACI-001], [REF-GIBBS-CANDES-2021, Equation 2].

    """

    __slots__ = (
        "_alpha_t",
        "_errors",
        "_gamma",
        "_history",
        "_initial_alpha",
        "_sorted",
        "_target",
        "_window",
    )

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.005,
        *,
        calibration_scores: Iterable[object] = (),
        window: int | None = None,
        initial_alpha: float | None = None,
    ) -> None:
        """Validate settings and seed the score history.

        References:
            [REF-GIBBS-CANDES-2021, Section 2].

        """
        self._target = _level(alpha, "alpha")
        step = require_finite_real(gamma, name="gamma")
        if step < 0.0:
            raise ValueError("gamma must be non-negative")
        self._gamma = step
        if initial_alpha is None:
            start = self._target
        else:
            start = require_finite_real(initial_alpha, name="initial_alpha")
            if not 0.0 <= start <= 1.0:
                raise ValueError("initial_alpha must lie in [0, 1]")
        self._initial_alpha = start
        self._alpha_t = start
        if window is not None:
            window = require_non_negative_integer(window, name="window")
            if window == 0:
                raise ValueError("window must be positive")
        self._window = window
        self._history: deque[float] = deque()
        self._sorted: list[float] = []
        self._errors: list[bool] = []
        for index, score in enumerate(calibration_scores):
            value = require_finite_real(score, name=f"calibration_scores[{index}]")
            if value < 0.0:
                raise ValueError("calibration scores must be non-negative")
            self._remember(value)

    @property
    def alpha(self) -> float:
        """Return the target miscoverage level.

        References:
            [REF-GIBBS-CANDES-2021, Section 2].

        """
        return self._target

    @property
    def gamma(self) -> float:
        """Return the step size.

        References:
            [REF-GIBBS-CANDES-2021, Section 2.1].

        """
        return self._gamma

    @property
    def current_alpha(self) -> float:
        """Return the level ``alpha_t`` used by the next interval.

        References:
            [EQ-CONF-ACI-001].

        """
        return self._alpha_t

    @property
    def errors(self) -> tuple[bool, ...]:
        """Return the miss indicators ``err_1, ..., err_T`` recorded so far.

        References:
            [REF-GIBBS-CANDES-2021, Section 2].

        """
        return tuple(self._errors)

    def radius(self) -> float:
        """Return the score threshold ``q_t``, the ``(1 - alpha_t)`` quantile.

        Returns:
            ``+inf`` for the whole line, ``-inf`` for the empty set, or the
            conformal order statistic of the score history. The order index
            is :func:`~probstat.conformal.conformal_rank` of ``alpha_t``
            (exact rational arithmetic on its decimal text), so a frozen
            level reproduces the split conformal quantile.

        Equation:
            [EQ-CONF-ACI-002].

        References:
            [REF-GIBBS-CANDES-2021, Section 1.1 and Section 4.1].

        """
        level = self._alpha_t
        if level >= 1.0:
            return -math.inf
        size = len(self._sorted)
        if level <= 0.0 or size == 0:
            return math.inf
        # The same exact-rational rank as split conformal prediction, so that
        # gamma = 0 reproduces split_conformal_regression step for step.
        rank = conformal_rank(size, level)
        if rank > size:
            return math.inf
        return self._sorted[rank - 1]

    def interval(self, prediction: float, scale: float = 1.0) -> PredictionInterval:
        """Return the current interval ``prediction +/- q_t scale``.

        Args:
            prediction: Finite point prediction.
            scale: Positive spread for normalized scores (``1`` for absolute
                residuals).

        Returns:
            The interval; ``upper < lower`` encodes the empty set.

        Raises:
            TypeError: If an argument is not real.
            ValueError: If an argument is non-finite or ``scale <= 0``.

        Equation:
            [EQ-CONF-ACI-002].

        References:
            [REF-GIBBS-CANDES-2021, Section 2].

        """
        centre = require_finite_real(prediction, name="prediction")
        spread = _positive(scale, "scale")
        radius = self.radius()
        if radius == -math.inf:
            return PredictionInterval(math.inf, -math.inf, centre)
        if radius == math.inf:
            return PredictionInterval(-math.inf, math.inf, centre)
        return PredictionInterval(
            centre - radius * spread, centre + radius * spread, centre
        )

    def update(self, prediction: float, observation: float, scale: float = 1.0) -> bool:
        """Score one revealed response, record the miss, and step ``alpha_t``.

        Args:
            prediction: The point prediction the interval was built on.
            observation: The revealed response.
            scale: The spread used for that interval.

        Returns:
            ``True`` when the response fell outside the interval (``err_t``).

        Raises:
            TypeError: If an argument is not real.
            ValueError: If an argument is non-finite or ``scale <= 0``.

        Equation:
            [EQ-CONF-ACI-001], [EQ-CONF-ACI-002].

        References:
            [REF-GIBBS-CANDES-2021, Equation 2].

        """
        centre = require_finite_real(prediction, name="prediction")
        response = require_finite_real(observation, name="observation")
        spread = _positive(scale, "scale")
        score = abs(response - centre) / spread
        missed = score > self.radius()
        self._errors.append(missed)
        self._alpha_t += self._gamma * (self._target - (1.0 if missed else 0.0))
        self._remember(score)
        return missed

    def long_run_bound(self) -> float:
        """Return ``(max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)``.

        Returns:
            The deterministic bound on ``|mean(err) - alpha|`` after the
            ``T`` recorded steps; ``inf`` before any step or when
            ``gamma = 0``.

        Equation:
            [EQ-CONF-ACI-003].

        References:
            [REF-GIBBS-CANDES-2021, Proposition 4.1].

        """
        steps = len(self._errors)
        if steps == 0 or self._gamma == 0.0:
            return math.inf
        start = self._initial_alpha
        return (max(start, 1.0 - start) + self._gamma) / (self._gamma * steps)

    def _remember(self, score: float) -> None:
        """Append a score to the history, evicting the oldest beyond the window.

        References:
            [REF-GIBBS-CANDES-2021, Section 2.2]. No equation is evaluated.

        """
        self._history.append(score)
        bisect.insort(self._sorted, score)
        if self._window is not None and len(self._history) > self._window:
            oldest = self._history.popleft()
            del self._sorted[bisect.bisect_left(self._sorted, oldest)]


@dataclass(frozen=True, slots=True)
class AdaptiveConformalResult:
    """Intervals, levels, and miss indicators of one ACI pass over a sequence.

    Args:
        intervals: Interval issued at each step before the response.
        alphas: Level ``alpha_t`` used at each step.
        errors: Miss indicators ``err_t``.
        alpha: Target miscoverage level.
        gamma: Step size.
        long_run_bound: Deterministic bound on ``|miscoverage_rate - alpha|``.

    References:
        [EQ-CONF-ACI-001], [EQ-CONF-ACI-003], [REF-GIBBS-CANDES-2021].

    """

    intervals: tuple[PredictionInterval, ...]
    alphas: tuple[float, ...]
    errors: tuple[bool, ...]
    alpha: float
    gamma: float
    long_run_bound: float

    @property
    def miscoverage_rate(self) -> float:
        """Return the realized miss frequency ``mean(err_t)``.

        References:
            [EQ-CONF-ACI-003].

        """
        return sum(self.errors) / len(self.errors)

    def local_coverage(self, width: int) -> tuple[float, ...]:
        """Return trailing-window coverage frequencies ``1 - mean(err)``.

        Args:
            width: Positive window length; entry ``t`` averages steps
                ``max(0, t - width + 1) .. t``.

        Returns:
            One coverage frequency per step.

        Raises:
            ValueError: If ``width`` is not positive.

        References:
            [REF-GIBBS-CANDES-2021, Equation 4].

        """
        size = require_non_negative_integer(width, name="width")
        if size == 0:
            raise ValueError("width must be positive")
        result: list[float] = []
        misses = 0
        for index, missed in enumerate(self.errors):
            misses += missed
            if index >= size:
                misses -= self.errors[index - size]
            result.append(1.0 - misses / min(index + 1, size))
        return tuple(result)


def adaptive_conformal_intervals(
    predictions: Sequence[object],
    observations: Sequence[object],
    *,
    alpha: float = 0.1,
    gamma: float = 0.005,
    scales: Sequence[object] | None = None,
    calibration_scores: Iterable[object] = (),
    window: int | None = None,
    initial_alpha: float | None = None,
) -> AdaptiveConformalResult:
    """Run adaptive conformal inference over a prediction/response sequence.

    At step ``t`` the interval is formed from the scores of steps
    ``1..t-1`` (plus the optional calibration scores) before ``y_t`` is
    used, then ``alpha_t`` is updated. The point predictions may come from
    any model, including one refitted online on past data.

    Args:
        predictions: Point predictions ``yhat_1, ..., yhat_T``.
        observations: Responses ``y_1, ..., y_T`` of the same length.
        alpha: Target miscoverage level in ``(0, 1)``.
        gamma: Non-negative step size (``0`` gives fixed-level conformal).
        scales: Optional positive spreads ``s_t`` for normalized scores.
        calibration_scores: Optional non-negative initial scores.
        window: Optional number of most recent scores in the quantile.
        initial_alpha: Starting level in ``[0, 1]``; defaults to ``alpha``.

    Returns:
        Intervals, levels, miss indicators, and the long-run bound.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sequences are empty or of unequal length, or a
            setting is invalid.

    Equation:
        [EQ-CONF-ACI-001], [EQ-CONF-ACI-002], [EQ-CONF-ACI-003].

    References:
        [REF-GIBBS-CANDES-2021, Equation 2 and Proposition 4.1].

    """
    steps = len(predictions)
    if steps == 0:
        raise ValueError("predictions must not be empty")
    if len(observations) != steps:
        raise ValueError("predictions and observations must have equal length")
    if scales is not None and len(scales) != steps:
        raise ValueError("scales must match the number of predictions")
    predictor = AdaptiveConformalPredictor(
        alpha,
        gamma,
        calibration_scores=calibration_scores,
        window=window,
        initial_alpha=initial_alpha,
    )
    intervals: list[PredictionInterval] = []
    alphas: list[float] = []
    for index in range(steps):
        prediction = require_finite_real(
            predictions[index], name=f"predictions[{index}]"
        )
        scale = (
            1.0
            if scales is None
            else require_finite_real(scales[index], name=f"scales[{index}]")
        )
        observation = require_finite_real(
            observations[index], name=f"observations[{index}]"
        )
        alphas.append(predictor.current_alpha)
        intervals.append(predictor.interval(prediction, scale))
        predictor.update(prediction, observation, scale)
    return AdaptiveConformalResult(
        intervals=tuple(intervals),
        alphas=tuple(alphas),
        errors=predictor.errors,
        alpha=predictor.alpha,
        gamma=predictor.gamma,
        long_run_bound=predictor.long_run_bound(),
    )
