"""Bayesian online change-point detection by the run-length recursion.

The data are modelled as a product partition: the series splits into
segments, the observations of one segment are exchangeable draws from a
conjugate model whose parameter is drawn afresh from the prior at the start
of each segment, and segments are independent. The *run length* ``r_t`` is
the number of observations among ``x_1..x_t`` that belong to the segment
which will also generate ``x_(t+1)``; ``r_t = 0`` says that ``x_(t+1)`` opens
a new segment. With a constant hazard ``H`` every boundary between two
consecutive observations is a change point with probability ``H``,
independently of everything else, and ``r_0 = 0``.

Adams and MacKay (2007) propagate the joint ``P(r_t, x_1:t)`` forward
[EQ-TS-BOCPD-001]::

    growth:  P(r_t = r + 1, x_1:t) = P(r_(t-1) = r, x_1:(t-1)) pi_r(x_t) (1 - H)
    change:  P(r_t = 0, x_1:t)     = H sum_r P(r_(t-1) = r, x_1:(t-1)) pi_r(x_t)

where ``pi_r(x_t)`` is the posterior predictive density of ``x_t`` given the
``r`` most recent observations. Summing the joint over ``r`` gives the
evidence ``P(x_1:t)``, which equals the sum over all ``2^(t-1)``
segmentations of ``H^m (1 - H)^(t-1-m)`` times the product of the segment
marginal likelihoods (``m`` change points). The recursion is exact for this
model; the work is ``O(T^2)`` predictive evaluations without truncation.

Two consequences of the recursion as published are worth knowing. First,
dividing the two lines shows ``P(r_t = 0 | x_1:t) = H`` for every ``t``: the
decision to reset before ``x_(t+1)`` does not depend on the data seen so far,
so the probability of run length zero carries no information. Evidence for
a change appears a few observations later, as mass on *short* run lengths;
:meth:`OnlineChangePointResult.segment_start_probability` reads it off.
Second, the output is a filtering distribution (it conditions on the past
only), not the smoothed posterior of the segmentation given the whole
series.

Conjugate models [EQ-TS-BOCPD-002], [EQ-TS-BOCPD-003]:

* :class:`NormalInverseGammaPrior`: Gaussian observations with unknown mean
  and variance; the predictive is a Student-t density.
* :class:`GammaPoissonPrior`: Poisson counts with a Gamma prior on the rate;
  the predictive is negative binomial.

``max_run_length`` drops run lengths above a bound after every step. That is
an approximation: the reported evidence then is a lower bound of the exact
one and the result records the largest posterior mass that was discarded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)

Vector = tuple[float, ...]
_State = tuple[float, ...]
_LOG_2PI = math.log(2.0 * math.pi)


def _positive(value: object, *, name: str) -> float:
    """Validate a strictly positive finite real.

    Args:
        value: Candidate number.
        name: Parameter name used in the error message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite or not positive.

    References:
        [REF-ADAMS-MACKAY-2007]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if not number > 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _log_sum_exp(values: list[float]) -> float:
    """Return ``log sum_i exp(v_i)`` without overflow.

    Args:
        values: Log-scale terms; ``-inf`` entries contribute zero.

    Returns:
        The log of the sum, ``-inf`` when every term is ``-inf`` or the list
        is empty.

    References:
        [EQ-TS-BOCPD-001]. Numerical device only.

    """
    peak = max(values, default=-math.inf)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


@dataclass(frozen=True, slots=True)
class NormalInverseGammaPrior:
    """Conjugate prior for Gaussian data with unknown mean and variance.

    ``sigma^2 ~ InverseGamma(shape, scale)`` and
    ``mu | sigma^2 ~ N(mean, sigma^2 / precision_scale)``. After ``n``
    observations with sum ``s`` the parameters are ``kappa_n = kappa_0 + n``,
    ``m_n = (kappa_0 m_0 + s) / kappa_n``, ``alpha_n = alpha_0 + n / 2`` and
    ``beta_n = beta_0 + sum (x_i - x_bar)^2 / 2
    + kappa_0 n (x_bar - m_0)^2 / (2 kappa_n)``; the predictive of the next
    observation is Student-t with ``2 alpha_n`` degrees of freedom, location
    ``m_n`` and squared scale ``beta_n (kappa_n + 1) / (alpha_n kappa_n)``.

    Args:
        mean: Prior mean ``m_0`` of ``mu``.
        precision_scale: ``kappa_0 > 0``, the prior weight of ``m_0`` in
            pseudo-observations.
        shape: ``alpha_0 > 0`` of the inverse-gamma prior on ``sigma^2``.
        scale: ``beta_0 > 0`` of the inverse-gamma prior on ``sigma^2``.

    Raises:
        TypeError: If a parameter is not real.
        ValueError: If a parameter is not finite or a positive parameter is
            not positive.

    References:
        [EQ-TS-BOCPD-002], [REF-MURPHY-2007].

    """

    mean: float = 0.0
    precision_scale: float = 1.0
    shape: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is outside its domain.

        References:
            [REF-MURPHY-2007]. No equation is evaluated.

        """
        object.__setattr__(self, "mean", require_finite_real(self.mean, name="mean"))
        object.__setattr__(
            self,
            "precision_scale",
            _positive(self.precision_scale, name="precision_scale"),
        )
        object.__setattr__(self, "shape", _positive(self.shape, name="shape"))
        object.__setattr__(self, "scale", _positive(self.scale, name="scale"))

    def log_predictive(self, value: float) -> float:
        """Return the log prior predictive density of one observation.

        Args:
            value: Observation ``x``.

        Returns:
            Log density of the Student-t predictive with ``2 alpha_0``
            degrees of freedom.

        Equation:
            [EQ-TS-BOCPD-002].

        References:
            [REF-MURPHY-2007].

        """
        point = require_finite_real(value, name="value")
        return _gaussian_log_predictive(self._state(), point)

    def log_marginal_likelihood(self, values: Iterable[float]) -> float:
        """Return the closed-form log evidence of one segment.

        ``log p(x_1:n) = lgamma(alpha_n) - lgamma(alpha_0)
        + alpha_0 log beta_0 - alpha_n log beta_n
        + (log kappa_0 - log kappa_n) / 2 - (n / 2) log(2 pi)``, computed
        from the batch statistics (not from the one-step predictives).

        Args:
            values: Observations of the segment (may be empty, giving zero).

        Returns:
            Log marginal likelihood.

        Raises:
            TypeError: If an observation is not real.
            ValueError: If an observation is not finite.

        Equation:
            [EQ-TS-BOCPD-002].

        References:
            [REF-MURPHY-2007].

        """
        sample = tuple(
            require_finite_real(value, name=f"values[{index}]")
            for index, value in enumerate(values)
        )
        n = len(sample)
        if n == 0:
            return 0.0
        average = math.fsum(sample) / n
        spread = math.fsum((value - average) ** 2 for value in sample)
        kappa = self.precision_scale + n
        alpha = self.shape + 0.5 * n
        beta = (
            self.scale
            + 0.5 * spread
            + self.precision_scale * n * (average - self.mean) ** 2 / (2.0 * kappa)
        )
        return (
            math.lgamma(alpha)
            - math.lgamma(self.shape)
            + self.shape * math.log(self.scale)
            - alpha * math.log(beta)
            + 0.5 * (math.log(self.precision_scale) - math.log(kappa))
            - 0.5 * n * _LOG_2PI
        )

    def _state(self) -> _State:
        """Return ``(m, kappa, alpha, beta)`` of the prior.

        References:
            [EQ-TS-BOCPD-002]. No equation is evaluated.

        """
        return (self.mean, self.precision_scale, self.shape, self.scale)


@dataclass(frozen=True, slots=True)
class GammaPoissonPrior:
    """Conjugate Gamma prior for the rate of Poisson counts.

    ``lambda ~ Gamma(shape, rate)`` (density proportional to
    ``lambda^(shape - 1) exp(-rate lambda)``). After ``n`` counts with sum
    ``s`` the posterior is ``Gamma(shape + s, rate + n)`` and the predictive
    of the next count is negative binomial,
    ``p(x) = Gamma(a + x) / (Gamma(a) x!) (b / (b + 1))^a (b + 1)^(-x)``
    with the current ``a`` and ``b``.

    Args:
        shape: ``a_0 > 0``.
        rate: ``b_0 > 0``.

    Raises:
        TypeError: If a parameter is not real.
        ValueError: If a parameter is not finite and positive.

    References:
        [EQ-TS-BOCPD-003], [REF-GELMAN-ETAL-2013].

    """

    shape: float = 1.0
    rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is not finite and positive.

        References:
            [REF-GELMAN-ETAL-2013]. No equation is evaluated.

        """
        object.__setattr__(self, "shape", _positive(self.shape, name="shape"))
        object.__setattr__(self, "rate", _positive(self.rate, name="rate"))

    def log_predictive(self, value: float) -> float:
        """Return the log prior predictive probability of one count.

        Args:
            value: Non-negative integer count.

        Returns:
            Log probability under the negative-binomial predictive.

        Raises:
            TypeError: If the count is not real.
            ValueError: If the count is not a non-negative integer.

        Equation:
            [EQ-TS-BOCPD-003].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        return _poisson_log_predictive(self._state(), _count(value, name="value"))

    def log_marginal_likelihood(self, values: Iterable[float]) -> float:
        """Return the closed-form log evidence of one segment of counts.

        ``log p(x_1:n) = lgamma(a_0 + s) - lgamma(a_0) + a_0 log b_0
        - (a_0 + s) log(b_0 + n) - sum_i log(x_i!)`` with ``s = sum x_i``.

        Args:
            values: Non-negative integer counts (may be empty, giving zero).

        Returns:
            Log marginal likelihood.

        Raises:
            TypeError: If a count is not real.
            ValueError: If a count is not a non-negative integer.

        Equation:
            [EQ-TS-BOCPD-003].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        sample = tuple(
            _count(value, name=f"values[{index}]") for index, value in enumerate(values)
        )
        n = len(sample)
        if n == 0:
            return 0.0
        total = math.fsum(sample)
        return (
            math.lgamma(self.shape + total)
            - math.lgamma(self.shape)
            + self.shape * math.log(self.rate)
            - (self.shape + total) * math.log(self.rate + n)
            - math.fsum(math.lgamma(value + 1.0) for value in sample)
        )

    def _state(self) -> _State:
        """Return ``(a, b)`` of the prior.

        References:
            [EQ-TS-BOCPD-003]. No equation is evaluated.

        """
        return (self.shape, self.rate)


def _count(value: object, *, name: str) -> float:
    """Validate a non-negative integer count given as int or integral float.

    Args:
        value: Candidate count.
        name: Name used in error messages.

    Returns:
        The count as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is negative, non-finite or not integral.

    References:
        [EQ-TS-BOCPD-003]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number != math.floor(number):
        raise ValueError(f"{name} must be a non-negative integer count")
    return number


def _gaussian_log_predictive(state: _State, value: float) -> float:
    """Return the Student-t log predictive for a normal-inverse-gamma state.

    Args:
        state: ``(m, kappa, alpha, beta)``.
        value: Observation.

    Returns:
        ``log t_(2 alpha)(x | m, beta (kappa + 1) / (alpha kappa))``.

    References:
        [EQ-TS-BOCPD-002], [REF-MURPHY-2007].

    """
    mean, kappa, alpha, beta = state
    degrees = 2.0 * alpha
    scale_squared = beta * (kappa + 1.0) / (alpha * kappa)
    deviation = (value - mean) ** 2 / scale_squared
    return (
        math.lgamma(0.5 * (degrees + 1.0))
        - math.lgamma(0.5 * degrees)
        - 0.5 * math.log(degrees * math.pi * scale_squared)
        - 0.5 * (degrees + 1.0) * math.log1p(deviation / degrees)
    )


def _gaussian_update(state: _State, value: float) -> _State:
    """Return the normal-inverse-gamma state after one more observation.

    Args:
        state: ``(m, kappa, alpha, beta)``.
        value: Observation.

    Returns:
        ``(m', kappa + 1, alpha + 1/2, beta + kappa (x - m)^2 / (2 (kappa + 1)))``
        with ``m' = (kappa m + x) / (kappa + 1)``.

    References:
        [EQ-TS-BOCPD-002], [REF-MURPHY-2007].

    """
    mean, kappa, alpha, beta = state
    return (
        (kappa * mean + value) / (kappa + 1.0),
        kappa + 1.0,
        alpha + 0.5,
        beta + kappa * (value - mean) ** 2 / (2.0 * (kappa + 1.0)),
    )


def _poisson_log_predictive(state: _State, value: float) -> float:
    """Return the negative-binomial log predictive for a Gamma state.

    Args:
        state: ``(a, b)``.
        value: Count.

    Returns:
        ``lgamma(a + x) - lgamma(a) - lgamma(x + 1) + a log(b / (b + 1))
        - x log(b + 1)``.

    References:
        [EQ-TS-BOCPD-003], [REF-GELMAN-ETAL-2013].

    """
    shape, rate = state
    return (
        math.lgamma(shape + value)
        - math.lgamma(shape)
        - math.lgamma(value + 1.0)
        + shape * (math.log(rate) - math.log1p(rate))
        - value * math.log1p(rate)
    )


def _poisson_update(state: _State, value: float) -> _State:
    """Return the Gamma state after one more count.

    Args:
        state: ``(a, b)``.
        value: Count.

    Returns:
        ``(a + x, b + 1)``.

    References:
        [EQ-TS-BOCPD-003], [REF-GELMAN-ETAL-2013].

    """
    return (state[0] + value, state[1] + 1.0)


@dataclass(frozen=True, slots=True)
class OnlineChangePointResult:
    """Run-length filtering distributions of a series.

    Index ``t`` (zero-based) refers to the state after the first ``t + 1``
    observations. Row ``t`` of ``run_length_posterior`` holds
    ``P(r = k | x_0..x_t)`` for ``k = 0, 1, ...``; a run length ``k >= 1``
    means the current segment started at index ``t - k + 1``.

    Args:
        model: ``"gaussian"`` or ``"poisson"``.
        hazard: Constant change probability ``H`` per boundary.
        run_length_posterior: One row per time with the run-length
            probabilities (rows have ``t + 2`` entries, fewer when truncated).
        map_run_length: Most probable run length per time.
        expected_run_length: Posterior mean run length per time.
        log_evidence_path: ``log P(x_0..x_t)`` per time; its increments are
            the one-step-ahead log predictive densities.
        log_evidence: Final ``log P(x_0..x_(T-1))``; a lower bound of the
            exact value when ``truncated`` is true.
        max_run_length: Truncation bound, or ``None`` for the exact recursion.
        truncated: Whether any run length was actually dropped.
        discarded_mass: Largest posterior probability dropped at one step
            (zero for the exact recursion).
        predictive_evaluations: Number of predictive densities evaluated.

    References:
        [EQ-TS-BOCPD-001], [REF-ADAMS-MACKAY-2007].

    """

    model: str
    hazard: float
    run_length_posterior: tuple[Vector, ...]
    map_run_length: tuple[int, ...]
    expected_run_length: Vector
    log_evidence_path: Vector
    log_evidence: float
    max_run_length: int | None
    truncated: bool
    discarded_mass: float
    predictive_evaluations: int

    @property
    def size(self) -> int:
        """Return the number of observations ``T``.

        References:
            [REF-ADAMS-MACKAY-2007]. No equation is evaluated.

        """
        return len(self.run_length_posterior)

    def run_length_probability(self, time: int, run_length: int) -> float:
        """Return ``P(r_t = k | x_0..x_t)``.

        Args:
            time: Zero-based time index ``t``.
            run_length: Run length ``k >= 0``.

        Returns:
            The filtering probability; zero for run lengths that are
            impossible at ``t`` or were truncated away.

        Raises:
            TypeError: If an argument is not an integer.
            ValueError: If ``time`` is outside the series or an argument is
                negative.

        Equation:
            [EQ-TS-BOCPD-001].

        References:
            [REF-ADAMS-MACKAY-2007].

        """
        index = require_non_negative_integer(time, name="time")
        length = require_non_negative_integer(run_length, name="run_length")
        if index >= self.size:
            raise ValueError("time must be smaller than the series length")
        row = self.run_length_posterior[index]
        return row[length] if length < len(row) else 0.0

    def segment_start_probability(self, delay: int = 1) -> Vector:
        """Return the probability that a segment starts at each index.

        Entry ``s`` is ``P(r_(s + delay - 1) = delay | x_0..x_(s + delay - 1))``:
        the filtering probability, judged after ``delay`` observations of the
        candidate segment, that the current segment began at index ``s``.
        Larger delays give sharper evidence and a later answer. Index ``0``
        always starts a segment (probability one for ``delay = 1`` without
        truncation). Entries whose judging time lies beyond the series are
        ``nan``.

        Args:
            delay: Number of observations ``d >= 1`` of the new segment that
                are used.

        Returns:
            One probability per index.

        Raises:
            TypeError: If ``delay`` is not an integer.
            ValueError: If ``delay`` is smaller than one.

        Equation:
            [EQ-TS-BOCPD-001].

        References:
            [REF-ADAMS-MACKAY-2007].

        """
        lag = require_non_negative_integer(delay, name="delay")
        if lag < 1:
            raise ValueError(
                "delay must be at least one: P(r_t = 0 | x_1:t) equals the hazard"
            )
        values: list[float] = []
        for start in range(self.size):
            judged = start + lag - 1
            if judged >= self.size:
                values.append(math.nan)
                continue
            row = self.run_length_posterior[judged]
            values.append(row[lag] if lag < len(row) else 0.0)
        return tuple(values)

    def map_change_points(self) -> tuple[int, ...]:
        """Return segment starts implied by drops of the MAP run length.

        Whenever the most probable run length fails to grow by one, the
        current segment is taken to have started at ``t - r_t + 1``. This is
        a heuristic reading of the filtering distributions, not the MAP
        segmentation of the series; the same change can be reported at
        slightly different indices while evidence accumulates, so starts are
        returned sorted and without duplicates. Index zero is omitted.

        Returns:
            Zero-based indices of the first observation of each detected
            segment.

        References:
            [EQ-TS-BOCPD-001], [REF-ADAMS-MACKAY-2007].

        """
        starts: set[int] = set()
        previous = 0
        for time, current in enumerate(self.map_run_length):
            if current != previous + 1 and current >= 1:
                start = time - current + 1
                if start > 0:
                    starts.add(start)
            previous = current
        return tuple(sorted(starts))


def bayesian_online_changepoint(
    series: Iterable[float],
    *,
    hazard: float,
    prior: NormalInverseGammaPrior | GammaPoissonPrior | None = None,
    max_run_length: int | None = None,
) -> OnlineChangePointResult:
    """Run the Adams-MacKay run-length recursion with a constant hazard.

    The joint ``P(r_t, x_1:t)`` is propagated in log space by the growth and
    change steps of [EQ-TS-BOCPD-001] with the conjugate predictive of the
    chosen prior. Without truncation the result is exact for the product
    partition model in which every boundary is a change point with
    probability ``hazard``; in particular ``log_evidence`` is the log of the
    sum over all segmentations. ``hazard = 0`` reduces to ordinary conjugate
    updating of one segment.

    Assumptions: observations are conditionally independent within a segment
    and segments are independent given the change points; the hazard is
    known and constant (geometric segment lengths with mean ``1 / hazard``).
    The answer is sensitive to the prior: a prior that is vague relative to
    the data makes new segments expensive and delays detection.

    Args:
        series: Observations; finite reals for the Gaussian model and
            non-negative integers for the Poisson model.
        hazard: Change probability ``H`` in ``[0, 1]`` per boundary.
        prior: :class:`NormalInverseGammaPrior` (default, with its default
            hyperparameters) or :class:`GammaPoissonPrior`.
        max_run_length: Optional bound ``R >= 1``; run lengths above ``R``
            are dropped after each step (an approximation that makes the
            work ``O(T R)``).

    Returns:
        Run-length posteriors, their MAP and mean, the evidence path, and
        truncation diagnostics.

    Raises:
        TypeError: If an observation or argument has the wrong type.
        ValueError: If the series is empty or not finite, a Poisson
            observation is not a non-negative integer, ``hazard`` is outside
            ``[0, 1]``, or ``max_run_length`` is smaller than one.

    Equation:
        [EQ-TS-BOCPD-001], [EQ-TS-BOCPD-002], [EQ-TS-BOCPD-003].

    References:
        [REF-ADAMS-MACKAY-2007], [REF-MURPHY-2007], [REF-GELMAN-ETAL-2013].

    """
    chosen = NormalInverseGammaPrior() if prior is None else prior
    if isinstance(chosen, NormalInverseGammaPrior):
        model = "gaussian"
        data = coerce_finite_sample(series, name="series")
        log_predictive = _gaussian_log_predictive
        update = _gaussian_update
    elif isinstance(chosen, GammaPoissonPrior):
        model = "poisson"
        data = tuple(
            _count(value, name=f"series[{index}]") for index, value in enumerate(series)
        )
        if not data:
            raise ValueError("series must contain at least one observation")
        log_predictive = _poisson_log_predictive
        update = _poisson_update
    else:
        raise TypeError(
            "prior must be a NormalInverseGammaPrior or a GammaPoissonPrior"
        )
    rate = require_finite_real(hazard, name="hazard")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("hazard must lie in [0, 1]")
    bound: int | None = None
    if max_run_length is not None:
        bound = require_non_negative_integer(max_run_length, name="max_run_length")
        if bound < 1:
            raise ValueError("max_run_length must be at least one")
    log_change = math.log(rate) if rate > 0.0 else -math.inf
    log_growth = math.log1p(-rate) if rate < 1.0 else -math.inf

    initial = chosen._state()
    states: list[_State] = [initial]
    log_joint: list[float] = [0.0]
    rows: list[Vector] = []
    map_lengths: list[int] = []
    means: list[float] = []
    evidence_path: list[float] = []
    discarded = 0.0
    truncated = False
    evaluations = 0
    for value in data:
        weighted = [
            mass + log_predictive(state, value)
            for mass, state in zip(log_joint, states, strict=True)
        ]
        evaluations += len(states)
        log_joint = [_log_sum_exp(weighted) + log_change] + [
            mass + log_growth for mass in weighted
        ]
        states = [initial] + [update(state, value) for state in states]
        if bound is not None and len(log_joint) > bound + 1:
            total = _log_sum_exp(log_joint)
            dropped = _log_sum_exp(log_joint[bound + 1 :])
            if dropped > -math.inf:
                truncated = True
                discarded = max(discarded, math.exp(dropped - total))
            del log_joint[bound + 1 :]
            del states[bound + 1 :]
        evidence = _log_sum_exp(log_joint)
        if evidence == -math.inf:
            raise ValueError(
                "the retained run lengths have zero probability; "
                "increase max_run_length or the hazard"
            )
        row = tuple(math.exp(mass - evidence) for mass in log_joint)
        rows.append(row)
        map_lengths.append(max(range(len(row)), key=row.__getitem__))
        means.append(math.fsum(k * p for k, p in enumerate(row)))
        evidence_path.append(evidence)
    return OnlineChangePointResult(
        model,
        rate,
        tuple(rows),
        tuple(map_lengths),
        tuple(means),
        tuple(evidence_path),
        evidence_path[-1],
        bound,
        truncated,
        discarded,
        evaluations,
    )
