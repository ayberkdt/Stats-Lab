"""Contextual bandits with linear payoffs: LinUCB and linear Thompson sampling.

In every round the agent sees one feature vector ``x_{t,a}`` in ``R^d`` per
arm, plays one arm, and observes only that arm's reward.

**LinUCB with disjoint linear models** keeps one ridge regression per arm:
``A_a = I + sum x x'`` and ``b_a = sum r x`` over the rounds in which arm
``a`` was played, ``theta_a = A_a^-1 b_a``, and plays the arm with the
largest upper confidence score
``theta_a' x_{t,a} + alpha sqrt(x_{t,a}' A_a^-1 x_{t,a})``. The exploration
weight ``alpha`` is a user parameter.

``A_a^-1`` is maintained directly. For a symmetric positive-definite ``A``
and ``u = A^-1 x``, multiplying out shows
``(A + x x') (A^-1 - u u' / (1 + x' u)) = I + x u' - x u' (1 + x' u) /
(1 + x' u) = I``, so ``(A + x x')^-1 = A^-1 - u u' / (1 + x' u)`` (the
Sherman-Morrison rank-one update), which costs ``O(d^2)`` per round.

**Linear Thompson sampling** assumes one shared parameter: with
``B = I + sum x x'`` and ``f = sum r x`` over all played feature vectors it
draws ``theta_tilde ~ N(B^-1 f, v^2 B^-1)`` and plays
``argmax_a x_{t,a}' theta_tilde``. With ``v = 1`` this is the exact posterior
of a ``N(0, I)`` prior under unit-variance Gaussian rewards. The scale ``v``
is a user parameter; the value used in the regret analysis of the source
depends on the horizon and the confidence level and is not reproduced here.

:func:`simulate_linear_bandit` runs either policy (or uniform random play)
in a known linear environment and reports the cumulative pseudo-regret.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    solve_upper,
    transpose,
)

ContextSampler = Callable[[random.Random], Sequence[Sequence[float]]]


def _positive_integer(value: object, *, name: str) -> int:
    """Validate a positive integer.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    number = require_non_negative_integer(value, name=name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


def _feature(values: Iterable[object], dimension: int, *, name: str) -> Vector:
    """Validate one feature vector of the given dimension.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    vector = tuple(
        require_finite_real(v, name=f"{name}[{j}]") for j, v in enumerate(values)
    )
    if len(vector) != dimension:
        raise ValueError(f"{name} must have dimension {dimension}, got {len(vector)}")
    return vector


def _features(
    contexts: Iterable[Iterable[object]], dimension: int, n_arms: int | None
) -> tuple[Vector, ...]:
    """Validate one feature vector per arm.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(
        _feature(row, dimension, name=f"contexts[{a}]")
        for a, row in enumerate(contexts)
    )
    if n_arms is not None and len(rows) != n_arms:
        raise ValueError(
            f"contexts must hold one feature vector per arm ({n_arms}), got {len(rows)}"
        )
    if len(rows) < 2:
        raise ValueError("contexts must hold at least two arms")
    return rows


def _argmax(values: Sequence[float]) -> int:
    """Return the first index of the largest value.

    References:
        [EQ-RISK-LINUCB-001]. Ties go to the lowest index.

    """
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the inner product of two equally long vectors.

    References:
        [REF-GOLUB-VANLOAN-2013]. No registered equation is evaluated.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _identity(dimension: int) -> list[list[float]]:
    """Return a mutable identity matrix.

    References:
        [REF-GOLUB-VANLOAN-2013]. No registered equation is evaluated.

    """
    return [
        [1.0 if i == j else 0.0 for j in range(dimension)] for i in range(dimension)
    ]


class LinUCB:
    """LinUCB with disjoint linear models (one ridge regression per arm).

    Args:
        n_arms: Number of arms (at least two).
        dimension: Feature dimension ``d``.
        alpha: Non-negative exploration weight; ``0`` gives the greedy
            ridge-regression rule.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``n_arms < 2``, ``dimension < 1``, or ``alpha < 0``.

    References:
        [EQ-RISK-LINUCB-001], [EQ-RISK-LINUCB-002], [REF-LI-ETAL-2010].

    """

    __slots__ = (
        "_design",
        "_inverse",
        "_pulls",
        "_response",
        "alpha",
        "dimension",
        "n_arms",
    )

    def __init__(self, n_arms: int, dimension: int, *, alpha: float = 1.0) -> None:
        """Start every arm at ``A_a = I`` and ``b_a = 0``.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        self.n_arms = _positive_integer(n_arms, name="n_arms")
        if self.n_arms < 2:
            raise ValueError("a bandit needs at least two arms")
        self.dimension = _positive_integer(dimension, name="dimension")
        self.alpha = require_finite_real(alpha, name="alpha")
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        d = self.dimension
        self._design = [_identity(d) for _ in range(self.n_arms)]
        self._inverse = [_identity(d) for _ in range(self.n_arms)]
        self._response = [[0.0] * d for _ in range(self.n_arms)]
        self._pulls = [0] * self.n_arms

    def _arm(self, arm: object) -> int:
        """Validate an arm index.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        index = require_non_negative_integer(arm, name="arm")
        if index >= self.n_arms:
            raise ValueError(f"arm must be below {self.n_arms}, got {index}")
        return index

    @property
    def pulls(self) -> tuple[int, ...]:
        """Return how often each arm was updated.

        References:
            [REF-LI-ETAL-2010]. No equation is evaluated.

        """
        return tuple(self._pulls)

    def design(self, arm: int) -> Matrix:
        """Return ``A_a = I + sum x x'`` accumulated directly.

        Args:
            arm: Arm index.

        Returns:
            The ``d`` by ``d`` design matrix of the arm.

        Raises:
            ValueError: If ``arm`` is out of range.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        return tuple(tuple(row) for row in self._design[self._arm(arm)])

    def inverse_design(self, arm: int) -> Matrix:
        """Return ``A_a^-1`` as maintained by rank-one updates.

        Args:
            arm: Arm index.

        Returns:
            The ``d`` by ``d`` inverse design matrix of the arm.

        Raises:
            ValueError: If ``arm`` is out of range.

        References:
            [EQ-RISK-LINUCB-002].

        """
        return tuple(tuple(row) for row in self._inverse[self._arm(arm)])

    def response(self, arm: int) -> Vector:
        """Return ``b_a = sum r x`` of the arm.

        Args:
            arm: Arm index.

        Returns:
            Vector of length ``d``.

        Raises:
            ValueError: If ``arm`` is out of range.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        return tuple(self._response[self._arm(arm)])

    def theta(self, arm: int) -> Vector:
        """Return the ridge estimate ``theta_a = A_a^-1 b_a``.

        Args:
            arm: Arm index.

        Returns:
            Vector of length ``d``.

        Raises:
            ValueError: If ``arm`` is out of range.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        index = self._arm(arm)
        return tuple(_dot(row, self._response[index]) for row in self._inverse[index])

    def bonus(self, arm: int, context: Iterable[object]) -> float:
        """Return the exploration bonus ``alpha sqrt(x' A_a^-1 x)``.

        Args:
            arm: Arm index.
            context: Feature vector ``x`` of the arm.

        Returns:
            The non-negative bonus.

        Raises:
            ValueError: If ``arm`` is out of range or ``context`` has the
                wrong dimension.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        index = self._arm(arm)
        x = _feature(context, self.dimension, name="context")
        return self.alpha * math.sqrt(max(self._quadratic(index, x), 0.0))

    def _quadratic(self, arm: int, x: Vector) -> float:
        """Return ``x' A_a^-1 x``.

        References:
            [EQ-RISK-LINUCB-001].

        """
        return _dot(x, [_dot(row, x) for row in self._inverse[arm]])

    def scores(self, contexts: Iterable[Iterable[object]]) -> tuple[float, ...]:
        """Return ``theta_a' x_a + alpha sqrt(x_a' A_a^-1 x_a)`` per arm.

        Args:
            contexts: One feature vector per arm.

        Returns:
            Upper confidence scores, one per arm.

        Raises:
            ValueError: If the number of vectors or a dimension is wrong.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        rows = _features(contexts, self.dimension, self.n_arms)
        return tuple(
            _dot(self.theta(a), x)
            + self.alpha * math.sqrt(max(self._quadratic(a, x), 0.0))
            for a, x in enumerate(rows)
        )

    def select(self, contexts: Iterable[Iterable[object]]) -> int:
        """Return the arm with the largest score (lowest index on ties).

        Args:
            contexts: One feature vector per arm.

        Returns:
            The chosen arm index.

        Raises:
            ValueError: If the number of vectors or a dimension is wrong.

        References:
            [EQ-RISK-LINUCB-001], [REF-LI-ETAL-2010].

        """
        return _argmax(self.scores(contexts))

    def update(self, arm: int, context: Iterable[object], reward: float) -> None:
        """Add one observation ``(x, r)`` to the model of ``arm``.

        ``A_a += x x'``, ``b_a += r x``, and
        ``A_a^-1 -= u u' / (1 + x' u)`` with ``u = A_a^-1 x``.

        Args:
            arm: Arm that was played.
            context: Feature vector of that arm.
            reward: Finite observed reward.

        Raises:
            ValueError: If ``arm`` is out of range, ``context`` has the wrong
                dimension, or ``reward`` is not finite.

        References:
            [EQ-RISK-LINUCB-001], [EQ-RISK-LINUCB-002], [REF-LI-ETAL-2010].

        """
        index = self._arm(arm)
        x = _feature(context, self.dimension, name="context")
        r = require_finite_real(reward, name="reward")
        inverse = self._inverse[index]
        u = [_dot(row, x) for row in inverse]
        denominator = 1.0 + _dot(x, u)
        design = self._design[index]
        response = self._response[index]
        for i in range(self.dimension):
            response[i] += r * x[i]
            for j in range(self.dimension):
                design[i][j] += x[i] * x[j]
                inverse[i][j] -= u[i] * u[j] / denominator
        self._pulls[index] += 1


class LinearThompsonSampling:
    """Thompson sampling for one shared linear payoff parameter.

    Args:
        dimension: Feature dimension ``d``.
        v: Positive scale of the sampling covariance ``v^2 B^-1``. ``v = 1``
            samples from the conjugate posterior of a ``N(0, I)`` prior with
            unit-variance Gaussian rewards.
        seed: Integer seed or stream for the parameter draws; ``None`` is
            not reproducible.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``dimension < 1`` or ``v <= 0``.

    References:
        [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

    """

    __slots__ = ("_design", "_response", "_rng", "_updates", "dimension", "v")

    def __init__(
        self,
        dimension: int,
        *,
        v: float = 1.0,
        seed: int | RandomStream | None = None,
    ) -> None:
        """Start at ``B = I`` and ``f = 0``.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        self.dimension = _positive_integer(dimension, name="dimension")
        self.v = require_finite_real(v, name="v")
        if self.v <= 0.0:
            raise ValueError(f"v must be positive, got {self.v}")
        self._design = _identity(self.dimension)
        self._response = [0.0] * self.dimension
        self._updates = 0
        self._rng = resolve_stream(seed).child("linear-thompson").generator()

    @property
    def updates(self) -> int:
        """Return the number of observations absorbed.

        References:
            [REF-AGRAWAL-GOYAL-2013]. No equation is evaluated.

        """
        return self._updates

    def design(self) -> Matrix:
        """Return ``B = I + sum x x'``.

        Returns:
            The ``d`` by ``d`` design matrix.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        return tuple(tuple(row) for row in self._design)

    def posterior_mean(self) -> Vector:
        """Return ``mu_hat = B^-1 f``.

        Returns:
            Vector of length ``d``.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        return cholesky_solve(cholesky(self.design()), tuple(self._response))

    def posterior_covariance(self) -> Matrix:
        """Return the sampling covariance ``v^2 B^-1``.

        Returns:
            The ``d`` by ``d`` covariance matrix.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        scale = self.v * self.v
        return tuple(
            tuple(scale * value for value in row) for row in inverse_spd(self.design())
        )

    def sample_parameter(self) -> Vector:
        """Draw ``theta_tilde ~ N(B^-1 f, v^2 B^-1)``.

        With the Cholesky factor ``B = L L'`` and ``z ~ N(0, I)`` the draw is
        ``B^-1 f + v L'^-1 z``, whose covariance is
        ``v^2 L'^-1 L^-1 = v^2 B^-1``.

        Returns:
            One parameter draw of length ``d``.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        lower = cholesky(self.design())
        mean = cholesky_solve(lower, tuple(self._response))
        z = tuple(self._rng.gauss(0.0, 1.0) for _ in range(self.dimension))
        deviation = solve_upper(transpose(lower), z)
        return tuple(m + self.v * e for m, e in zip(mean, deviation, strict=True))

    def select(self, contexts: Iterable[Iterable[object]]) -> int:
        """Draw a parameter and return ``argmax_a x_a' theta_tilde``.

        Args:
            contexts: One feature vector per arm (at least two arms).

        Returns:
            The chosen arm index (lowest index on ties).

        Raises:
            ValueError: If fewer than two vectors are given or a dimension
                is wrong.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        rows = _features(contexts, self.dimension, None)
        theta = self.sample_parameter()
        return _argmax([_dot(x, theta) for x in rows])

    def update(self, arm: int, context: Iterable[object], reward: float) -> None:
        """Add one observation: ``B += x x'`` and ``f += r x``.

        Args:
            arm: Arm that was played; unused because the parameter is
                shared, kept so that both policies have the same interface.
            context: Feature vector of the played arm.
            reward: Finite observed reward.

        Raises:
            ValueError: If ``context`` has the wrong dimension or ``reward``
                is not finite.

        References:
            [EQ-RISK-LINTS-001], [REF-AGRAWAL-GOYAL-2013].

        """
        del arm
        x = _feature(context, self.dimension, name="context")
        r = require_finite_real(reward, name="reward")
        for i in range(self.dimension):
            self._response[i] += r * x[i]
            for j in range(self.dimension):
                self._design[i][j] += x[i] * x[j]
        self._updates += 1


@dataclass(frozen=True, slots=True)
class LinearBanditRun:
    """Actions, rewards, and regret of one simulated linear bandit run.

    Args:
        policy: ``"linucb"``, ``"linear_thompson"``, or ``"uniform"``.
        actions: Chosen arm per round.
        rewards: Observed noisy reward per round.
        cumulative_regret: Running pseudo-regret
            ``sum_t (max_a m_t(a) - m_t(a_t))`` with the true mean rewards
            ``m_t(a)`` of the environment.
        pulls: Number of rounds each arm was played.

    References:
        [EQ-RISK-LINUCB-003].

    """

    policy: str
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    cumulative_regret: tuple[float, ...]
    pulls: tuple[int, ...]

    @property
    def horizon(self) -> int:
        """Return the number of rounds.

        References:
            [EQ-RISK-LINUCB-003]. No equation is evaluated.

        """
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        """Return the sum of observed rewards.

        References:
            [EQ-RISK-LINUCB-003]. No equation is evaluated.

        """
        return math.fsum(self.rewards)

    @property
    def final_regret(self) -> float:
        """Return the cumulative pseudo-regret after the last round.

        References:
            [EQ-RISK-LINUCB-003].

        """
        return self.cumulative_regret[-1]

    def regret_per_round(self, rounds: int | None = None) -> float:
        """Return the cumulative pseudo-regret after ``rounds`` over ``rounds``.

        A policy that learns has a regret per round that decreases with the
        number of rounds (sublinear regret).

        Args:
            rounds: Number of rounds between ``1`` and the horizon; the
                horizon when omitted.

        Returns:
            Average pseudo-regret per round.

        Raises:
            ValueError: If ``rounds`` is outside ``1, ..., horizon``.

        References:
            [EQ-RISK-LINUCB-003].

        """
        count = self.horizon if rounds is None else rounds
        count = _positive_integer(count, name="rounds")
        if count > self.horizon:
            raise ValueError(f"rounds must be at most {self.horizon}, got {count}")
        return self.cumulative_regret[count - 1] / count


def _unit_contexts(n_arms: int, dimension: int) -> ContextSampler:
    """Return a sampler of independent uniformly distributed unit vectors.

    References:
        [EQ-RISK-LINUCB-003]. A normalized standard-normal vector is uniform
        on the unit sphere.

    """

    def sampler(rng: random.Random) -> tuple[Vector, ...]:
        """Draw one unit feature vector per arm.

        References:
            [EQ-RISK-LINUCB-003].

        """
        rows: list[Vector] = []
        for _ in range(n_arms):
            raw = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
            norm = math.sqrt(math.fsum(v * v for v in raw))
            if norm == 0.0:
                rows.append(tuple(1.0 if j == 0 else 0.0 for j in range(dimension)))
            else:
                rows.append(tuple(v / norm for v in raw))
        return tuple(rows)

    return sampler


def simulate_linear_bandit(
    policy: LinUCB | LinearThompsonSampling | str,
    parameters: Sequence[float] | Sequence[Sequence[float]],
    horizon: int,
    *,
    n_arms: int | None = None,
    noise_sd: float = 1.0,
    context_sampler: ContextSampler | None = None,
    seed: int | RandomStream | None = None,
) -> LinearBanditRun:
    """Run a policy in a known linear environment and record its regret.

    The mean reward of arm ``a`` with features ``x_{t,a}`` is
    ``theta_a' x_{t,a}`` (one parameter row per arm) or ``theta' x_{t,a}``
    (one shared parameter vector); the observed reward adds independent
    ``N(0, noise_sd^2)`` noise. The policy object is updated in place, so
    pass a freshly constructed policy for a reproducible run.

    Args:
        policy: A :class:`LinUCB` or :class:`LinearThompsonSampling`
            instance, or ``"uniform"`` for uniformly random play.
        parameters: A vector (shared parameter; ``n_arms`` is then required)
            or a matrix with one row per arm.
        horizon: Positive number of rounds.
        n_arms: Number of arms; inferred from a parameter matrix.
        noise_sd: Non-negative reward noise standard deviation.
        context_sampler: ``sampler(rng)`` returning one feature vector per
            arm; by default every arm gets an independent uniformly
            distributed unit vector each round.
        seed: Integer seed or stream for contexts, noise, and uniform play
            (separate child streams); ``None`` is not reproducible.

    Returns:
        Actions, rewards, cumulative pseudo-regret, and pull counts.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If shapes disagree, the horizon is not positive, the
            policy does not match the environment, or ``noise_sd < 0``.

    Equation:
        [EQ-RISK-LINUCB-001], [EQ-RISK-LINTS-001], [EQ-RISK-LINUCB-003].

    References:
        [REF-LI-ETAL-2010], [REF-AGRAWAL-GOYAL-2013].

    """
    rounds = _positive_integer(horizon, name="horizon")
    sd = require_finite_real(noise_sd, name="noise_sd")
    if sd < 0.0:
        raise ValueError(f"noise_sd must be non-negative, got {sd}")
    entries = tuple(parameters)
    if not entries:
        raise ValueError("parameters must not be empty")
    if all(isinstance(entry, (tuple, list)) for entry in entries):
        dimension = len(tuple(entries[0]))  # type: ignore[arg-type]
        if dimension == 0:
            raise ValueError("parameters rows must not be empty")
        thetas = tuple(
            _feature(entry, dimension, name=f"parameters[{a}]")  # type: ignore[arg-type]
            for a, entry in enumerate(entries)
        )
        if n_arms is not None and n_arms != len(thetas):
            raise ValueError("n_arms does not match the number of parameter rows")
        arms = len(thetas)
    else:
        if n_arms is None:
            raise ValueError("n_arms is required with a shared parameter vector")
        arms = _positive_integer(n_arms, name="n_arms")
        dimension = len(entries)
        shared = _feature(entries, dimension, name="parameters")
        thetas = (shared,) * arms
    if arms < 2:
        raise ValueError("a bandit needs at least two arms")
    if isinstance(policy, LinUCB):
        name = "linucb"
        if policy.n_arms != arms or policy.dimension != dimension:
            raise ValueError(
                "the LinUCB policy does not match the environment's arms and dimension"
            )
    elif isinstance(policy, LinearThompsonSampling):
        name = "linear_thompson"
        if policy.dimension != dimension:
            raise ValueError(
                "the Thompson policy does not match the environment's dimension"
            )
    elif policy == "uniform":
        name = "uniform"
    else:
        raise ValueError(
            "policy must be a LinUCB, a LinearThompsonSampling, or 'uniform'"
        )
    stream = resolve_stream(seed)
    context_rng = stream.child("linear-bandit", "contexts").generator()
    noise_rng = stream.child("linear-bandit", "noise").generator()
    uniform_rng = stream.child("linear-bandit", "uniform").generator()
    sampler = (
        _unit_contexts(arms, dimension) if context_sampler is None else context_sampler
    )
    actions: list[int] = []
    rewards: list[float] = []
    regret: list[float] = []
    pulls = [0] * arms
    running = 0.0
    for _ in range(rounds):
        contexts = _features(sampler(context_rng), dimension, arms)
        means = [_dot(theta, x) for theta, x in zip(thetas, contexts, strict=True)]
        if isinstance(policy, str):
            action = uniform_rng.randrange(arms)
        else:
            action = policy.select(contexts)
        reward = means[action] + (noise_rng.gauss(0.0, sd) if sd > 0.0 else 0.0)
        if not isinstance(policy, str):
            policy.update(action, contexts[action], reward)
        running += max(means) - means[action]
        actions.append(action)
        rewards.append(reward)
        regret.append(running)
        pulls[action] += 1
    return LinearBanditRun(
        policy=name,
        actions=tuple(actions),
        rewards=tuple(rewards),
        cumulative_regret=tuple(regret),
        pulls=tuple(pulls),
    )
