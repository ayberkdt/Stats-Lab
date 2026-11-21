"""Off-policy evaluation of a decision policy from logged bandit data.

A log holds ``n`` independent records ``(x_i, a_i, r_i, p_i)``: a context,
the action the logging policy took, the observed reward, and the probability
``p_i = p(a_i | x_i)`` with which the logging policy chose that action. The
value of a different target policy ``pi`` is
``V(pi) = E_x sum_a pi(a | x) E[r | x, a]``; it is estimated without running
``pi``.

With importance weights ``w_i = pi(a_i | x_i) / p_i`` and a reward model
``q(x, a)`` the estimators are

- inverse propensity scoring ``mean(w_i r_i)``, unbiased when the logged
  propensities are the true ones and ``p(a | x) > 0`` wherever
  ``pi(a | x) > 0``;
- self-normalized IPS ``sum(w_i r_i) / sum(w_i)``, a ratio estimator that is
  consistent but biased for finite ``n`` and that shifts by ``c`` when ``c``
  is added to every reward;
- the direct method ``mean(sum_a pi(a | x_i) q(x_i, a))``, which is only as
  good as the reward model;
- the doubly robust estimator
  ``mean(sum_a pi(a | x_i) q(x_i, a) + w_i (r_i - q(x_i, a_i)))``, unbiased
  if either the propensities or the reward model are correct.

Unbiasedness of the doubly robust estimator: with true propensities
``E[w (r - q(x, a)) | x] = sum_a pi(a | x) (m(x, a) - q(x, a))`` where
``m(x, a) = E[r | x, a]``, which cancels the error of the direct term for any
``q``; with ``q = m`` the correction ``w (r - m(x, a))`` has conditional mean
zero for any positive weights. The reward model must not be fitted on the
records it is evaluated on (use a separate sample), otherwise the argument
does not apply.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

TargetPolicy = Callable[[object], Sequence[float]]
RewardModel = Callable[[object, int], float]

_PROBABILITY_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class LoggedBanditData:
    """Validated log of contexts, actions, rewards, and propensities.

    Args:
        contexts: One context per record; any objects the target policy and
            the reward model understand.
        actions: Logged action index in ``0, ..., n_actions - 1`` per record.
        rewards: Finite observed reward per record.
        propensities: Probability in ``(0, 1]`` with which the logging policy
            chose the logged action in that context.
        n_actions: Number of actions (at least two).

    Raises:
        TypeError: If an entry has the wrong type.
        ValueError: If lengths disagree, fewer than two records are given, an
            action is out of range, or a propensity is outside ``(0, 1]``.

    References:
        [REF-DUDIK-LANGFORD-LI-2011].

    """

    contexts: tuple[object, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    propensities: tuple[float, ...]
    n_actions: int

    def __post_init__(self) -> None:
        """Validate and freeze the log.

        References:
            [REF-DUDIK-LANGFORD-LI-2011]. No equation is evaluated.

        """
        arms = require_non_negative_integer(self.n_actions, name="n_actions")
        if arms < 2:
            raise ValueError(f"n_actions must be at least 2, got {arms}")
        contexts = tuple(self.contexts)
        actions = tuple(
            require_non_negative_integer(a, name=f"actions[{i}]")
            for i, a in enumerate(self.actions)
        )
        rewards = tuple(
            require_finite_real(r, name=f"rewards[{i}]")
            for i, r in enumerate(self.rewards)
        )
        propensities = tuple(
            require_finite_real(p, name=f"propensities[{i}]")
            for i, p in enumerate(self.propensities)
        )
        n = len(contexts)
        if not (len(actions) == len(rewards) == len(propensities) == n):
            raise ValueError(
                "contexts, actions, rewards, and propensities must have equal "
                f"lengths, got {n}, {len(actions)}, {len(rewards)}, "
                f"{len(propensities)}"
            )
        if n < 2:
            raise ValueError(f"at least two records are required, got {n}")
        for i, action in enumerate(actions):
            if action >= arms:
                raise ValueError(
                    f"actions[{i}] = {action} is not an action index below {arms}"
                )
        for i, p in enumerate(propensities):
            if not 0.0 < p <= 1.0:
                raise ValueError(f"propensities[{i}] must lie in (0, 1], got {p}")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "propensities", propensities)
        object.__setattr__(self, "n_actions", arms)

    @property
    def size(self) -> int:
        """Return the number of logged records.

        References:
            [REF-DUDIK-LANGFORD-LI-2011]. No equation is evaluated.

        """
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class OffPolicyEstimate:
    """Estimated value of a target policy with its diagnostics.

    Args:
        estimator: ``"ips"``, ``"snips"``, ``"direct"``, or
            ``"doubly_robust"``.
        value: Estimate of ``V(pi)``.
        standard_error: Standard error from the per-record terms. For the
            direct method it reflects only the sampling of contexts, not the
            error of the reward model.
        n: Number of records.
        effective_sample_size: ``(sum w)^2 / sum w^2`` of the (clipped)
            weights, a heuristic diagnostic between ``0`` and ``n``; ``n``
            for the direct method, which uses no weights.
        max_weight: Largest (clipped) importance weight.
        clip: Weight cap used, or ``None``.
        clipped_fraction: Fraction of records whose weight was capped.
        terms: Per-record terms whose mean is ``value`` (for ``"snips"`` the
            terms ``n w_i r_i / sum w``).

    References:
        [EQ-RISK-OPE-001], [EQ-RISK-OPE-002], [EQ-RISK-OPE-003],
        [EQ-RISK-OPE-004], [EQ-RISK-OPE-005].

    """

    estimator: str
    value: float
    standard_error: float
    n: int
    effective_sample_size: float
    max_weight: float
    clip: float | None
    clipped_fraction: float
    terms: tuple[float, ...]

    def interval(self, z: float = 1.959963984540054) -> tuple[float, float]:
        """Return the normal-approximation interval ``value -+ z se``.

        The interval relies on a central limit theorem for the per-record
        terms; with heavy-tailed weights (small effective sample size) its
        coverage can be far below nominal.

        Args:
            z: Positive standard-normal quantile; the default gives 95 %.

        Returns:
            Lower and upper limits.

        Raises:
            ValueError: If ``z`` is not positive.

        References:
            [EQ-RISK-OPE-005].

        """
        quantile = require_finite_real(z, name="z")
        if quantile <= 0.0:
            raise ValueError(f"z must be positive, got {quantile}")
        half = quantile * self.standard_error
        return (self.value - half, self.value + half)


def _require_data(data: object) -> LoggedBanditData:
    """Return ``data`` if it is a :class:`LoggedBanditData`.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not isinstance(data, LoggedBanditData):
        raise TypeError("data must be a LoggedBanditData")
    return data


def _target_probabilities(
    data: LoggedBanditData, target_policy: TargetPolicy
) -> tuple[tuple[float, ...], ...]:
    """Evaluate and validate ``pi(. | x_i)`` for every record.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not callable(target_policy):
        raise TypeError("target_policy must be callable")
    rows: list[tuple[float, ...]] = []
    for i, context in enumerate(data.contexts):
        row = tuple(
            require_finite_real(p, name=f"target_policy(contexts[{i}])[{a}]")
            for a, p in enumerate(target_policy(context))
        )
        if len(row) != data.n_actions:
            raise ValueError(
                f"target_policy(contexts[{i}]) must return {data.n_actions} "
                f"probabilities, got {len(row)}"
            )
        if min(row) < 0.0 or abs(math.fsum(row) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                f"target_policy(contexts[{i}]) must be non-negative "
                "probabilities summing to one"
            )
        rows.append(row)
    return tuple(rows)


def _clip_value(clip: float | None) -> float | None:
    """Validate an optional positive weight cap.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if clip is None:
        return None
    cap = require_finite_real(clip, name="clip")
    if cap <= 0.0:
        raise ValueError(f"clip must be positive, got {cap}")
    return cap


def _weights(
    data: LoggedBanditData,
    probabilities: Sequence[Sequence[float]],
    clip: float | None,
) -> tuple[tuple[float, ...], float]:
    """Return the weights ``min(pi(a_i | x_i) / p_i, clip)`` and clipped share.

    References:
        [EQ-RISK-OPE-001], [EQ-RISK-OPE-005].

    """
    raw = tuple(
        row[action] / p
        for row, action, p in zip(
            probabilities, data.actions, data.propensities, strict=True
        )
    )
    if clip is None:
        return raw, 0.0
    return (
        tuple(min(w, clip) for w in raw),
        sum(1 for w in raw if w > clip) / len(raw),
    )


def _effective_sample_size(weights: Sequence[float]) -> float:
    """Return ``(sum w)^2 / sum w^2``, or ``0`` when all weights vanish.

    References:
        [EQ-MC-ESS-001], [REF-OWEN-2013].

    """
    square = math.fsum(w * w for w in weights)
    if square == 0.0:
        return 0.0
    return math.fsum(weights) ** 2 / square


def _mean_and_error(terms: Sequence[float]) -> tuple[float, float]:
    """Return the mean and ``s / sqrt(n)`` with the ``n - 1`` divisor.

    References:
        [EQ-RISK-OPE-005].

    """
    n = len(terms)
    mean = math.fsum(terms) / n
    variance = math.fsum((t - mean) ** 2 for t in terms) / (n - 1)
    return mean, math.sqrt(variance / n)


def _model_table(
    data: LoggedBanditData, reward_model: RewardModel
) -> tuple[tuple[float, ...], ...]:
    """Evaluate ``q(x_i, a)`` for every record and action.

    References:
        [EQ-RISK-OPE-003]. Costs ``n * n_actions`` model evaluations.

    """
    if not callable(reward_model):
        raise TypeError("reward_model must be callable")
    return tuple(
        tuple(
            require_finite_real(
                reward_model(context, a), name=f"reward_model(contexts[{i}], {a})"
            )
            for a in range(data.n_actions)
        )
        for i, context in enumerate(data.contexts)
    )


def inverse_propensity_score(
    data: LoggedBanditData,
    target_policy: TargetPolicy,
    *,
    clip: float | None = None,
) -> OffPolicyEstimate:
    """Estimate ``V(pi)`` by ``mean(w_i r_i)`` with ``w_i = pi(a_i|x_i) / p_i``.

    Unbiased when the logged propensities are correct and the logging policy
    gives positive probability to every action the target policy can take.
    The variance grows with the weights; the effective sample size reports
    their concentration.

    Args:
        data: The log.
        target_policy: ``pi(context)`` returning one probability per action.
        clip: Optional cap ``M`` replacing ``w_i`` by ``min(w_i, M)``.
            Clipping lowers the variance and introduces bias (downwards for
            non-negative rewards).

    Returns:
        Estimate with standard error ``s / sqrt(n)`` of the terms
        ``w_i r_i``.

    Raises:
        TypeError: If the data or the policy has the wrong type.
        ValueError: If the policy output is not a probability vector or
            ``clip`` is not positive.

    Equation:
        [EQ-RISK-OPE-001], [EQ-RISK-OPE-005].

    References:
        [REF-DUDIK-LANGFORD-LI-2011].

    """
    log = _require_data(data)
    cap = _clip_value(clip)
    weights, clipped = _weights(log, _target_probabilities(log, target_policy), cap)
    terms = tuple(w * r for w, r in zip(weights, log.rewards, strict=True))
    value, error = _mean_and_error(terms)
    return OffPolicyEstimate(
        estimator="ips",
        value=value,
        standard_error=error,
        n=log.size,
        effective_sample_size=_effective_sample_size(weights),
        max_weight=max(weights),
        clip=cap,
        clipped_fraction=clipped,
        terms=terms,
    )


def self_normalized_ips(
    data: LoggedBanditData,
    target_policy: TargetPolicy,
    *,
    clip: float | None = None,
) -> OffPolicyEstimate:
    """Estimate ``V(pi)`` by the ratio ``sum(w_i r_i) / sum(w_i)``.

    The estimate is a weighted average of observed rewards, so it stays in
    their range and shifts by ``c`` when ``c`` is added to every reward. It
    is consistent under the conditions of :func:`inverse_propensity_score`
    but biased for finite ``n``.

    Args:
        data: The log.
        target_policy: ``pi(context)`` returning one probability per action.
        clip: Optional weight cap; introduces bias.

    Returns:
        Estimate with the delta-method standard error
        ``sqrt(sum w_i^2 (r_i - V_hat)^2) / sum w_i``.

    Raises:
        TypeError: If the data or the policy has the wrong type.
        ValueError: If the policy output is not a probability vector,
            ``clip`` is not positive, or every weight is zero (the target
            policy never takes a logged action).

    Equation:
        [EQ-RISK-OPE-002], [EQ-RISK-OPE-005].

    References:
        [REF-OWEN-2013] (self-normalized importance sampling).

    """
    log = _require_data(data)
    cap = _clip_value(clip)
    weights, clipped = _weights(log, _target_probabilities(log, target_policy), cap)
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError(
            "every importance weight is zero: the target policy never takes a "
            "logged action"
        )
    value = math.fsum(w * r for w, r in zip(weights, log.rewards, strict=True)) / total
    error = (
        math.sqrt(
            math.fsum(
                (w * (r - value)) ** 2
                for w, r in zip(weights, log.rewards, strict=True)
            )
        )
        / total
    )
    scale = log.size / total
    return OffPolicyEstimate(
        estimator="snips",
        value=value,
        standard_error=error,
        n=log.size,
        effective_sample_size=_effective_sample_size(weights),
        max_weight=max(weights),
        clip=cap,
        clipped_fraction=clipped,
        terms=tuple(scale * w * r for w, r in zip(weights, log.rewards, strict=True)),
    )


def direct_method(
    data: LoggedBanditData,
    target_policy: TargetPolicy,
    reward_model: RewardModel,
) -> OffPolicyEstimate:
    """Estimate ``V(pi)`` by ``mean(sum_a pi(a | x_i) q(x_i, a))``.

    Uses only the logged contexts and a user-supplied reward model; logged
    actions, rewards, and propensities are ignored. The estimate inherits
    the bias of the model, and the reported standard error does not include
    it.

    Args:
        data: The log.
        target_policy: ``pi(context)`` returning one probability per action.
        reward_model: ``q(context, action)`` returning a predicted reward.

    Returns:
        Estimate with standard error ``s / sqrt(n)`` of the per-context
        model values.

    Raises:
        TypeError: If the data, policy, or model has the wrong type.
        ValueError: If the policy output is not a probability vector or the
            model returns a non-finite value.

    Equation:
        [EQ-RISK-OPE-003], [EQ-RISK-OPE-005].

    References:
        [REF-DUDIK-LANGFORD-LI-2011].

    """
    log = _require_data(data)
    probabilities = _target_probabilities(log, target_policy)
    table = _model_table(log, reward_model)
    terms = tuple(
        math.fsum(p * q for p, q in zip(row, values, strict=True))
        for row, values in zip(probabilities, table, strict=True)
    )
    value, error = _mean_and_error(terms)
    return OffPolicyEstimate(
        estimator="direct",
        value=value,
        standard_error=error,
        n=log.size,
        effective_sample_size=float(log.size),
        max_weight=1.0,
        clip=None,
        clipped_fraction=0.0,
        terms=terms,
    )


def doubly_robust(
    data: LoggedBanditData,
    target_policy: TargetPolicy,
    reward_model: RewardModel,
    *,
    clip: float | None = None,
) -> OffPolicyEstimate:
    """Estimate ``V(pi)`` by the direct method plus a weighted residual.

    The per-record term is
    ``sum_a pi(a | x_i) q(x_i, a) + w_i (r_i - q(x_i, a_i))``. The estimator
    is unbiased if the propensities are correct (for any fixed ``q``) or if
    ``q(x, a) = E[r | x, a]`` (for any positive propensities); a good ``q``
    also reduces the variance relative to IPS. ``q`` must be fixed
    independently of the evaluated records.

    Args:
        data: The log.
        target_policy: ``pi(context)`` returning one probability per action.
        reward_model: ``q(context, action)`` returning a predicted reward.
        clip: Optional weight cap; with a wrong reward model it introduces
            bias.

    Returns:
        Estimate with standard error ``s / sqrt(n)`` of the per-record
        terms.

    Raises:
        TypeError: If the data, policy, or model has the wrong type.
        ValueError: If the policy output is not a probability vector, the
            model returns a non-finite value, or ``clip`` is not positive.

    Equation:
        [EQ-RISK-OPE-004], [EQ-RISK-OPE-005].

    References:
        [REF-DUDIK-LANGFORD-LI-2011].

    """
    log = _require_data(data)
    cap = _clip_value(clip)
    probabilities = _target_probabilities(log, target_policy)
    weights, clipped = _weights(log, probabilities, cap)
    table = _model_table(log, reward_model)
    terms = tuple(
        math.fsum(p * q for p, q in zip(row, values, strict=True))
        + w * (r - values[action])
        for row, values, w, r, action in zip(
            probabilities, table, weights, log.rewards, log.actions, strict=True
        )
    )
    value, error = _mean_and_error(terms)
    return OffPolicyEstimate(
        estimator="doubly_robust",
        value=value,
        standard_error=error,
        n=log.size,
        effective_sample_size=_effective_sample_size(weights),
        max_weight=max(weights),
        clip=cap,
        clipped_fraction=clipped,
        terms=terms,
    )


def logged_bandit_data(
    records: Iterable[tuple[object, int, float, float]], *, n_actions: int
) -> LoggedBanditData:
    """Build a :class:`LoggedBanditData` from ``(x, a, r, p)`` records.

    Args:
        records: Tuples of context, action index, reward, and logging
            propensity.
        n_actions: Number of actions.

    Returns:
        The validated log.

    Raises:
        TypeError: If an entry has the wrong type.
        ValueError: If a record does not have four fields or the log is
            invalid.

    References:
        [REF-DUDIK-LANGFORD-LI-2011]. No equation is evaluated.

    """
    rows: list[tuple[Any, ...]] = [tuple(record) for record in records]
    for i, row in enumerate(rows):
        if len(row) != 4:
            raise ValueError(f"records[{i}] must be (context, action, reward, p)")
    return LoggedBanditData(
        contexts=tuple(row[0] for row in rows),
        actions=tuple(row[1] for row in rows),
        rewards=tuple(row[2] for row in rows),
        propensities=tuple(row[3] for row in rows),
        n_actions=n_actions,
    )
