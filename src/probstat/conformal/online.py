"""Online conformal prediction by quantile tracking, with fixed or decaying steps.

Adaptive conformal inference (:mod:`probstat.conformal.adaptive`) adjusts the
*level* at which a quantile of past scores is taken. Quantile tracking
adjusts the score threshold itself, with no score history:

``err_t = 1{s_t > q_t}``,  ``q_{t+1} = q_t + eta_t (err_t - alpha)``,

an online (sub)gradient step on the pinball loss of level ``1 - alpha``. A
miss raises the threshold by ``eta_t (1 - alpha)``; a cover lowers it by
``eta_t alpha``. A fixed step ``eta_t = eta`` is the quantile tracker of
Angelopoulos, Candes and Tibshirani (2023); its integrator and scorecaster
terms are not implemented here. Steps ``eta_t = eta t^(-decay)`` with
``decay = 1/2 + eps`` are the decaying schedule of Angelopoulos, Barber and
Bates (2024), which lets the threshold settle when the scores are stable
instead of oscillating forever.

Guarantee (long-run only, no assumption on the data). If every score lies in
``[0, b]``, ``q_1`` lies in ``[0, b]``, and the steps are positive and
non-increasing, then for every ``T``

``|mean(err_1..err_T) - alpha| <= (b + eta_1) / (eta_T T)``.

Proof sketch: by induction ``q_t`` stays in
``[-alpha eta_1, b + (1 - alpha) eta_1]`` (below zero every step is a miss,
above ``b`` every step is a cover), and ``err_t - alpha = (q_{t+1} - q_t) /
eta_t``; summation by parts against the non-decreasing sequence ``1/eta_t``
gives the bound. For a fixed step it reads ``(b + eta)/(eta T)``; for
``eta_t = eta t^(-decay)`` it reads ``(b + eta) T^(decay - 1) / eta`` and
tends to zero only for ``decay < 1``. This is a statement about the
frequency of misses over time. It is not a finite-sample coverage
probability for any single step, and it is not conditional coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
)

from .adaptive import _level, _positive
from .regression import PredictionInterval


class OnlineQuantileTracker:
    """Online intervals ``yhat_t +/- q_t s_t`` with a tracked threshold ``q_t``.

    Call :meth:`interval` before the response is revealed and :meth:`update`
    after it.

    Args:
        alpha: Target miscoverage level in ``(0, 1)``.
        step: Positive base step ``eta``, in the units of the scores.
        decay: Exponent in ``eta_t = eta t^(-decay)``, in ``[0, 1]``;
            ``0`` is the fixed-step tracker, ``1/2 + eps`` the decaying
            schedule.
        initial_quantile: Starting threshold ``q_1 >= 0``.
        score_bound: Optional known upper bound ``b`` of the scores; needed
            for :meth:`long_run_bound` and requires ``q_1 <= b``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If a level, step, exponent, or bound is invalid.

    References:
        [EQ-CONF-013], [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
        [REF-ANGELOPOULOS-BARBER-BATES-2024].

    """

    __slots__ = (
        "_bound",
        "_bound_violated",
        "_decay",
        "_errors",
        "_last_step",
        "_quantile",
        "_step",
        "_target",
    )

    def __init__(
        self,
        alpha: float = 0.1,
        step: float = 0.1,
        *,
        decay: float = 0.0,
        initial_quantile: float = 0.0,
        score_bound: float | None = None,
    ) -> None:
        """Validate the settings.

        References:
            [EQ-CONF-013], [REF-ANGELOPOULOS-BARBER-BATES-2024].

        """
        self._target = _level(alpha, "alpha")
        self._step = _positive(step, "step")
        exponent = require_finite_real(decay, name="decay")
        if not 0.0 <= exponent <= 1.0:
            raise ValueError("decay must lie in [0, 1]")
        self._decay = exponent
        self._quantile = require_non_negative_finite_real(
            initial_quantile, name="initial_quantile"
        )
        if score_bound is None:
            self._bound: float | None = None
        else:
            self._bound = _positive(score_bound, "score_bound")
            if self._quantile > self._bound:
                raise ValueError("initial_quantile must not exceed score_bound")
        self._bound_violated = False
        self._errors: list[bool] = []
        self._last_step = self._step

    @property
    def alpha(self) -> float:
        """Return the target miscoverage level.

        References:
            [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023]. No equation is evaluated.

        """
        return self._target

    @property
    def quantile(self) -> float:
        """Return the threshold ``q_t`` used by the next interval.

        References:
            [EQ-CONF-013].

        """
        return self._quantile

    @property
    def errors(self) -> tuple[bool, ...]:
        """Return the miss indicators ``err_1, ..., err_T`` recorded so far.

        References:
            [EQ-CONF-013].

        """
        return tuple(self._errors)

    @property
    def bound_violated(self) -> bool:
        """Return whether a score exceeded the declared ``score_bound``.

        References:
            [EQ-CONF-014]. No equation is evaluated.

        """
        return self._bound_violated

    def step_size(self, index: int) -> float:
        """Return ``eta_t = eta t^(-decay)`` for the one-based step ``t``.

        Args:
            index: Step number ``t >= 1``.

        Returns:
            The step size.

        Raises:
            ValueError: If ``index < 1``.

        Equation:
            [EQ-CONF-013].

        References:
            [REF-ANGELOPOULOS-BARBER-BATES-2024].

        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("index must be a positive integer")
        return self._step * math.pow(index, -self._decay)

    def interval(self, prediction: float, scale: float = 1.0) -> PredictionInterval:
        """Return the current interval ``prediction +/- q_t scale``.

        Args:
            prediction: Finite point prediction.
            scale: Positive spread for normalized scores.

        Returns:
            The interval; a negative ``q_t`` gives ``upper < lower``, the
            empty set.

        Raises:
            TypeError: If an argument is not real.
            ValueError: If an argument is non-finite or ``scale <= 0``.

        Equation:
            [EQ-CONF-013].

        References:
            [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023].

        """
        centre = require_finite_real(prediction, name="prediction")
        half = self._quantile * _positive(scale, "scale")
        return PredictionInterval(centre - half, centre + half, centre)

    def update_score(self, score: float) -> bool:
        """Record one revealed score and step the threshold.

        Args:
            score: Non-negative finite score ``s_t``.

        Returns:
            ``err_t = 1{s_t > q_t}``.

        Raises:
            TypeError: If ``score`` is not real.
            ValueError: If ``score`` is negative or non-finite.

        Equation:
            [EQ-CONF-013].

        References:
            [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
            [REF-ANGELOPOULOS-BARBER-BATES-2024].

        """
        value = require_non_negative_finite_real(score, name="score")
        if self._bound is not None and value > self._bound:
            self._bound_violated = True
        missed = value > self._quantile
        self._errors.append(missed)
        eta = self.step_size(len(self._errors))
        self._last_step = eta
        self._quantile += eta * ((1.0 if missed else 0.0) - self._target)
        return missed

    def update(self, prediction: float, observation: float, scale: float = 1.0) -> bool:
        """Score a revealed response as ``|y - yhat| / scale`` and step.

        Args:
            prediction: The point prediction the interval was built on.
            observation: The revealed response.
            scale: The spread used for that interval.

        Returns:
            ``True`` when the response fell outside the interval.

        Raises:
            TypeError: If an argument is not real.
            ValueError: If an argument is non-finite or ``scale <= 0``.

        Equation:
            [EQ-CONF-013].

        References:
            [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023].

        """
        centre = require_finite_real(prediction, name="prediction")
        response = require_finite_real(observation, name="observation")
        return self.update_score(abs(response - centre) / _positive(scale, "scale"))

    def long_run_bound(self) -> float:
        """Return ``(b + eta_1) / (eta_T T)``, the bound on ``|mean(err) - alpha|``.

        Returns:
            The deterministic bound after the ``T`` recorded steps; ``inf``
            before any step, without a ``score_bound``, or once a score has
            exceeded the declared bound (the bound is then not valid).

        Equation:
            [EQ-CONF-014].

        References:
            [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
            [REF-ANGELOPOULOS-BARBER-BATES-2024].

        """
        steps = len(self._errors)
        if steps == 0 or self._bound is None or self._bound_violated:
            return math.inf
        return (self._bound + self._step) / (self._last_step * steps)


@dataclass(frozen=True, slots=True)
class OnlineQuantileResult:
    """Intervals, thresholds, and miss indicators of one quantile-tracking pass.

    Args:
        intervals: Interval issued at each step before the response.
        quantiles: Threshold ``q_t`` used at each step.
        errors: Miss indicators ``err_t``.
        alpha: Target miscoverage level.
        step: Base step ``eta``.
        decay: Step exponent.
        final_quantile: Threshold ``q_{T+1}`` after the last update.
        long_run_bound: Bound on ``|miscoverage_rate - alpha|``; ``inf``
            when no valid score bound was available.
        bound_valid: Whether a score bound was declared and never exceeded.

    References:
        [EQ-CONF-013], [EQ-CONF-014],
        [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
        [REF-ANGELOPOULOS-BARBER-BATES-2024].

    """

    intervals: tuple[PredictionInterval, ...]
    quantiles: tuple[float, ...]
    errors: tuple[bool, ...]
    alpha: float
    step: float
    decay: float
    final_quantile: float
    long_run_bound: float
    bound_valid: bool

    @property
    def miscoverage_rate(self) -> float:
        """Return the realized miss frequency ``mean(err_t)``.

        References:
            [EQ-CONF-014].

        """
        return sum(self.errors) / len(self.errors)


def online_quantile_intervals(
    predictions: Sequence[object],
    observations: Sequence[object],
    *,
    alpha: float = 0.1,
    step: float = 0.1,
    decay: float = 0.0,
    scales: Sequence[object] | None = None,
    initial_quantile: float = 0.0,
    score_bound: float | None = None,
) -> OnlineQuantileResult:
    """Run quantile-tracking online conformal prediction over a sequence.

    At step ``t`` the interval ``yhat_t +/- q_t s_t`` is issued before
    ``y_t`` is used; then ``q_{t+1} = q_t + eta_t (err_t - alpha)`` with
    ``eta_t = step * t^(-decay)``. The only guarantee is the deterministic
    long-run bound ``|mean(err) - alpha| <= (b + eta_1)/(eta_T T)`` for
    scores in ``[0, b]``; ``step`` should be of the order of the score
    scale, since the bound and the transient both depend on ``b / step``.

    Args:
        predictions: Point predictions ``yhat_1, ..., yhat_T``.
        observations: Responses ``y_1, ..., y_T`` of the same length.
        alpha: Target miscoverage level in ``(0, 1)``.
        step: Positive base step ``eta``.
        decay: Exponent in ``[0, 1]`` (``0`` fixed step; ``1/2 + eps`` the
            decaying schedule; the bound vanishes only for ``decay < 1``).
        scales: Optional positive spreads ``s_t`` for normalized scores.
        initial_quantile: Starting threshold ``q_1 >= 0``.
        score_bound: Optional known bound ``b`` on the scores.

    Returns:
        Intervals, thresholds, miss indicators, and the long-run bound.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sequences are empty or of unequal length, or a
            setting is invalid.

    Equation:
        [EQ-CONF-013], [EQ-CONF-014].

    References:
        [REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
        [REF-ANGELOPOULOS-BARBER-BATES-2024].

    """
    steps = len(predictions)
    if steps == 0:
        raise ValueError("predictions must not be empty")
    if len(observations) != steps:
        raise ValueError("predictions and observations must have equal length")
    if scales is not None and len(scales) != steps:
        raise ValueError("scales must match the number of predictions")
    tracker = OnlineQuantileTracker(
        alpha,
        step,
        decay=decay,
        initial_quantile=initial_quantile,
        score_bound=score_bound,
    )
    intervals: list[PredictionInterval] = []
    quantiles: list[float] = []
    for index in range(steps):
        prediction = require_finite_real(
            predictions[index], name=f"predictions[{index}]"
        )
        observation = require_finite_real(
            observations[index], name=f"observations[{index}]"
        )
        scale = (
            1.0
            if scales is None
            else require_finite_real(scales[index], name=f"scales[{index}]")
        )
        quantiles.append(tracker.quantile)
        intervals.append(tracker.interval(prediction, scale))
        tracker.update(prediction, observation, scale)
    bound = tracker.long_run_bound()
    return OnlineQuantileResult(
        intervals=tuple(intervals),
        quantiles=tuple(quantiles),
        errors=tracker.errors,
        alpha=tracker.alpha,
        step=float(step),
        decay=float(decay),
        final_quantile=tracker.quantile,
        long_run_bound=bound,
        bound_valid=bound != math.inf,
    )
