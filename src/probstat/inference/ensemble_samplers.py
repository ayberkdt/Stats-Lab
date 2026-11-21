"""Ensemble MCMC: the affine-invariant stretch move and parallel tempering.

Both samplers advance several states at once and use the other states to
build moves that a single chain cannot make.

Stretch move (:func:`stretch_move_sampler`, [EQ-INF-STRETCH-001])
    An ensemble of ``n`` walkers targets the product density
    ``pi(x_1) ... pi(x_n)``. To move walker ``x_k``, pick another walker
    ``x_j``, draw ``z`` from the density ``g(z) ∝ 1 / sqrt(z)`` on
    ``[1 / a, a]`` by ``z = ((a - 1) U + 1)^2 / a``, propose
    ``y = x_j + z (x_k - x_j)``, and accept with probability
    ``min(1, z^(d - 1) pi(y) / pi(x_k))`` (Goodman and Weare 2010). The
    proposal uses only differences of walkers, so the sampler is affine
    invariant: sampling ``pi_A(y) = pi(A^-1 (y - b))`` from the mapped start
    with the same random numbers gives exactly the mapped trajectory. Poorly
    scaled or correlated targets therefore cost no more than isotropic ones,
    and there is no step size to tune.

    The walkers are split into two halves that are updated alternately, each
    walker choosing its partner ``x_j`` uniformly from the other half. The
    partner is then never moved during the half-sweep, so each half-sweep is
    a product of valid Goodman-Weare moves conditional on the other half.

    Limits. The ensemble needs more walkers than dimensions and must not
    start in a lower-dimensional affine subspace, which it could never
    leave. The walkers are dependent, so R-hat across walkers is not the
    multi-chain diagnostic it is for independent chains, and the sampler is
    known to degrade in high dimension and does not cross between
    well-separated modes.

Parallel tempering (:func:`parallel_tempering`, [EQ-INF-PT-001])
    Replicas ``x_0, ..., x_T`` target ``pi_i(x) ∝ p(x) L(x)^beta_i`` on a
    ladder ``beta_0 < ... < beta_T = 1``. Each iteration applies a local
    kernel to every replica and then proposes to exchange the states of
    adjacent replicas, accepting the swap of ``i`` and ``i + 1`` with
    probability ``min(1, exp((beta_i - beta_{i+1}) (l(x_{i+1}) - l(x_i))))``,
    ``l = log L`` (Hukushima and Nemoto 1996). The pairs are swept by the
    deterministic even-odd scheme: pairs ``(0,1), (2,3), ...`` at even
    iterations and ``(1,2), (3,4), ...`` at odd ones, the non-reversible
    scheme analysed by Syed, Bouchard-Côté, Deligiannidis and Doucet (2022),
    whose round-trip rate does not collapse as the ladder is refined. The
    result reports the swap rejection rate of every pair and the number of
    round trips (a state that travels from the hottest replica to the
    coldest and back).

    With ``beta_0 = 0`` and a normalized prior the same run gives the
    stepping-stone estimate of the evidence ([EQ-INF-PT-002]),
    ``log Z = sum_i log mean_t exp((beta_{i+1} - beta_i) l(x_i^(t)))``,
    from the identity ``Z(beta') / Z(beta) = E_beta[L^(beta' - beta)]``.
    No standard error is attached: the draws of a replica are autocorrelated
    and the replicas are dependent through the swaps.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector

from .mcmc import (
    LogDensity,
    MarkovChain,
    MCMCResult,
    _checked_log_density,
    _initial_points,
    _names,
    _validate_run,
    _with_convergence_check,
    effective_sample_size,
)

LocalKernel = Callable[[LogDensity, Vector, float, random.Random], Vector]


# ---------------------------------------------------------------------------
# Stretch move
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Kept states of an interacting ensemble of walkers.

    Args:
        walkers: ``walkers[k][t]`` is the state of walker ``k`` at kept
            sweep ``t``.
        log_density: Target log-density at each kept state, same layout.
        names: Coordinate names.
        acceptance_rates: Post-warm-up acceptance rate of each walker.
        stretch: Stretch scale ``a``.
        warmup: Discarded sweeps.
        thin: Thinning interval.
        evaluations: Log-density evaluations including warm-up.
        seed: Seed of the random stream that drove the ensemble.

    References:
        [EQ-INF-STRETCH-001], [REF-GOODMAN-WEARE-2010].

    """

    walkers: tuple[tuple[Vector, ...], ...]
    log_density: tuple[tuple[float, ...], ...]
    names: tuple[str, ...]
    acceptance_rates: tuple[float, ...]
    stretch: float
    warmup: int
    thin: int
    evaluations: int
    seed: int

    @property
    def dimension(self) -> int:
        """Return the state dimension.

        References:
            [REF-GOODMAN-WEARE-2010]. No equation is evaluated.

        """
        return len(self.names)

    @property
    def acceptance_rate(self) -> float:
        """Return the mean post-warm-up acceptance rate over walkers.

        References:
            [EQ-INF-STRETCH-001], [REF-GOODMAN-WEARE-2010].

        """
        return math.fsum(self.acceptance_rates) / len(self.acceptance_rates)

    def columns(self, index: int) -> tuple[tuple[float, ...], ...]:
        """Return one coordinate as one scalar series per walker.

        Args:
            index: Zero-based coordinate index.

        Returns:
            ``columns[k][t]``, usable with the scalar diagnostics of
            :mod:`probstat.inference.mcmc` (read them with the dependence of
            the walkers in mind).

        Raises:
            IndexError: If ``index`` is out of range.

        References:
            [REF-GOODMAN-WEARE-2010]. No equation is evaluated.

        """
        if not 0 <= index < self.dimension:
            raise IndexError("coordinate index out of range")
        return tuple(tuple(state[index] for state in walker) for walker in self.walkers)

    def pooled(self) -> tuple[Vector, ...]:
        """Return all kept states, walker by walker.

        References:
            [REF-GOODMAN-WEARE-2010]. No equation is evaluated.

        """
        return tuple(state for walker in self.walkers for state in walker)

    def mean(self) -> Vector:
        """Return the mean of all kept states.

        References:
            [EQ-MC-MEAN-001], [REF-GOODMAN-WEARE-2010].

        """
        states = self.pooled()
        return tuple(
            math.fsum(state[i] for state in states) / len(states)
            for i in range(self.dimension)
        )

    def standard_deviation(self) -> Vector:
        """Return the sample standard deviation of each coordinate.

        References:
            [EQ-MC-VAR-001], [REF-GOODMAN-WEARE-2010].

        """
        states = self.pooled()
        center = self.mean()
        return tuple(
            math.sqrt(
                math.fsum((state[i] - center[i]) ** 2 for state in states)
                / (len(states) - 1)
            )
            for i in range(self.dimension)
        )

    def effective_sample_size(self, index: int) -> float:
        """Return a heuristic effective sample size of one coordinate.

        The walkers' series are passed to
        :func:`~probstat.inference.effective_sample_size` as if they were
        independent chains. The walkers interact, so the value is an
        approximation whose error is not quantified here; it is adequate for
        comparing runs and for a rough Monte Carlo standard error.

        Args:
            index: Zero-based coordinate index.

        Returns:
            The multi-chain ESS estimate of the pooled draws.

        Raises:
            IndexError: If ``index`` is out of range.
            ValueError: With fewer than four kept sweeps.

        References:
            [EQ-MCMC-ESS-001], [REF-GEYER-1992].

        """
        return effective_sample_size(self.columns(index))


def _walker_matrix(initial: Sequence[Sequence[float]]) -> list[Vector]:
    """Validate the starting ensemble.

    References:
        [REF-GOODMAN-WEARE-2010]. No equation is evaluated.

    """
    walkers = [as_vector(point, name="initial walker") for point in initial]
    if len(walkers) < 2:
        raise ValueError("at least 2 walkers are required")
    dimension = len(walkers[0])
    if dimension == 0 or any(len(walker) != dimension for walker in walkers):
        raise ValueError("walkers must share a positive dimension")
    if len(walkers) <= dimension:
        raise ValueError(
            "the ensemble needs more walkers than dimensions "
            f"({len(walkers)} walkers, dimension {dimension})"
        )
    if len(set(walkers)) < len(walkers):
        raise ValueError("initial walkers must be distinct")
    return walkers


def stretch_move_sampler(
    log_density: LogDensity,
    initial: Sequence[Sequence[float]],
    *,
    draws: int,
    warmup: int = 0,
    thin: int = 1,
    stretch: float = 2.0,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> EnsembleResult:
    """Sample a log-density with the affine-invariant stretch move.

    One sweep updates the first half of the walkers and then the second
    half. Walker ``x_k`` picks a partner ``x_j`` uniformly from the other
    half, draws ``z = ((a - 1) U + 1)^2 / a``, proposes
    ``y = x_j + z (x_k - x_j)``, and accepts when
    ``log U' < (d - 1) log z + log pi(y) - log pi(x_k)``.

    Args:
        log_density: Target ``log pi(x)`` up to a constant; ``-inf`` marks
            points outside the support.
        initial: Starting ensemble, one distinct point per walker, with more
            walkers than dimensions and finite log-density everywhere. The
            points must span the space (not lie in a common hyperplane);
            this is not checked.
        draws: Kept sweeps (each keeps the whole ensemble).
        warmup: Discarded initial sweeps (nothing is adapted).
        thin: Keep every ``thin``-th post-warm-up sweep.
        stretch: Scale ``a > 1`` of the stretch density; 2 is the value used
            by Goodman and Weare.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.

    Returns:
        The kept ensemble with per-walker acceptance rates and the number of
        log-density evaluations. No convergence check is run: the standard
        multi-chain diagnostics assume independent chains.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid run lengths, ``stretch <= 1``, too few or
            coincident walkers, a walker with non-finite log-density, or a
            target returning NaN or ``+inf``.

    Equation:
        [EQ-INF-STRETCH-001].

    References:
        [REF-GOODMAN-WEARE-2010].

    """
    checked = _checked_log_density(log_density)
    draws, warmup, _, thin = _validate_run(draws, warmup, 1, thin)
    scale = require_finite_real(stretch, name="stretch")
    if scale <= 1.0:
        raise ValueError("stretch must be greater than 1")
    states = _walker_matrix(initial)
    count = len(states)
    dimension = len(states[0])
    values = [checked(state) for state in states]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("every initial walker must have a finite log-density")
    evaluations = count
    stream = resolve_stream(seed)
    generator = stream.generator()
    half = count // 2
    halves = (
        (tuple(range(half)), tuple(range(half, count))),
        (tuple(range(half, count)), tuple(range(half))),
    )
    accepted = [0] * count
    kept: list[list[Vector]] = [[] for _ in range(count)]
    trace: list[list[float]] = [[] for _ in range(count)]
    for iteration in range(warmup + draws * thin):
        for active, complement in halves:
            for k in active:
                partner = states[complement[generator.randrange(len(complement))]]
                z = ((scale - 1.0) * generator.random() + 1.0) ** 2 / scale
                proposal = tuple(
                    p + z * (x - p) for p, x in zip(partner, states[k], strict=True)
                )
                value = checked(proposal)
                evaluations += 1
                log_ratio = (dimension - 1) * math.log(z) + value - values[k]
                if math.log(1.0 - generator.random()) < log_ratio:
                    states[k] = proposal
                    values[k] = value
                    if iteration >= warmup:
                        accepted[k] += 1
        if iteration >= warmup and (iteration - warmup + 1) % thin == 0:
            for k in range(count):
                kept[k].append(states[k])
                trace[k].append(values[k])
    checked_names = _names(names, dimension)
    if len(checked_names) != dimension or len(set(checked_names)) != dimension:
        raise ValueError("names must be distinct and match the dimension")
    return EnsembleResult(
        tuple(tuple(walker) for walker in kept),
        tuple(tuple(walker) for walker in trace),
        checked_names,
        tuple(a / (draws * thin) for a in accepted),
        scale,
        warmup,
        thin,
        evaluations,
        stream.seed,
    )


# ---------------------------------------------------------------------------
# Parallel tempering
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParallelTemperingResult:
    """Cold-chain draws and exchange statistics of a parallel-tempering run.

    Args:
        cold: Draws of the replica at ``beta = 1`` as a one-chain
            :class:`MCMCResult` (its ``log_density`` trace holds
            ``log p + log L``; its ``warnings`` hold the convergence
            messages when the check was requested).
        betas: Inverse-temperature ladder ``beta_0 < ... < beta_T = 1``.
        swap_rejection_rates: Post-warm-up fraction of rejected swap
            proposals for each adjacent pair ``(i, i + 1)``.
        local_acceptance_rates: Post-warm-up fraction of local moves that
            changed the state, per temperature.
        round_trips: Number of completed journeys of a state from the
            hottest replica to the coldest and back, after warm-up.
        log_evidence: Stepping-stone estimate of ``log Z`` when
            ``beta_0 = 0``; ``None`` otherwise. It is the marginal
            likelihood only if the prior and the likelihood are normalized.
        log_evidence_increments: ``log mean_t exp((beta_{i+1} - beta_i)
            l(x_i^(t)))`` per pair (always reported).
        mean_log_likelihoods: Post-warm-up mean of ``log L`` at each
            temperature.
        evaluations: Evaluations of the prior-likelihood pair, all replicas
            and warm-up included.
        seed: Seed of the random stream.

    References:
        [EQ-INF-PT-001], [EQ-INF-PT-002], [REF-HUKUSHIMA-NEMOTO-1996],
        [REF-SYED-ETAL-2022].

    """

    cold: MCMCResult
    betas: tuple[float, ...]
    swap_rejection_rates: tuple[float, ...]
    local_acceptance_rates: tuple[float, ...]
    round_trips: int
    log_evidence: float | None
    log_evidence_increments: tuple[float, ...]
    mean_log_likelihoods: tuple[float, ...]
    evaluations: int
    seed: int

    @property
    def swap_acceptance_rates(self) -> tuple[float, ...]:
        """Return ``1 - rejection rate`` for each adjacent pair.

        References:
            [EQ-INF-PT-001], [REF-HUKUSHIMA-NEMOTO-1996].

        """
        return tuple(1.0 - rate for rate in self.swap_rejection_rates)

    @property
    def samples(self) -> tuple[Vector, ...]:
        """Return the kept cold-chain states.

        References:
            [REF-HUKUSHIMA-NEMOTO-1996]. No equation is evaluated.

        """
        return self.cold.chains[0].samples


class _TemperedTarget:
    """Prior and likelihood of one evaluation point, cached per kernel call.

    The local kernel sees a plain log-density ``log p + beta log L``. The
    components of every point it evaluates are remembered so that the
    sampler can read ``log L`` of the returned state without a second
    evaluation.

    References:
        [EQ-INF-PT-001], [REF-HUKUSHIMA-NEMOTO-1996].

    """

    __slots__ = ("beta", "cache", "calls", "log_likelihood", "log_prior")

    def __init__(self, log_prior: LogDensity, log_likelihood: LogDensity) -> None:
        """Store the two callables.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.beta = 1.0
        self.calls = 0
        self.cache: dict[Vector, tuple[float, float]] = {}

    def components(self, point: Vector) -> tuple[float, float]:
        """Return ``(log p(x), log L(x))``, evaluating at most once per call.

        References:
            [EQ-INF-PT-001], [REF-HUKUSHIMA-NEMOTO-1996].

        """
        found = self.cache.get(point)
        if found is not None:
            return found
        self.calls += 1
        prior = self.log_prior(point)
        likelihood = self.log_likelihood(point) if prior > -math.inf else -math.inf
        self.cache[point] = (prior, likelihood)
        return prior, likelihood

    def tempered(self, prior: float, likelihood: float) -> float:
        """Return ``log p + beta log L`` (``-inf`` outside the support).

        References:
            [EQ-INF-PT-001], [REF-HUKUSHIMA-NEMOTO-1996].

        """
        if prior == -math.inf:
            return -math.inf
        if self.beta == 0.0:
            return prior
        return prior + self.beta * likelihood

    def __call__(self, point: Vector) -> float:
        """Evaluate the tempered log-density at the current ``beta``.

        References:
            [EQ-INF-PT-001], [REF-HUKUSHIMA-NEMOTO-1996].

        """
        return self.tempered(*self.components(tuple(point)))


def _random_walk_kernel(scale: float) -> LocalKernel:
    """Return an isotropic random-walk Metropolis kernel with step ``scale``.

    References:
        [EQ-MCMC-MH-001], [REF-METROPOLIS-ETAL-1953].

    """

    def kernel(
        log_density: LogDensity,
        state: Vector,
        current: float,
        generator: random.Random,
    ) -> Vector:
        """Propose ``x + scale xi`` and accept by the Metropolis rule.

        References:
            [EQ-MCMC-MH-001], [REF-METROPOLIS-ETAL-1953].

        """
        proposal = tuple(x + scale * generator.gauss(0.0, 1.0) for x in state)
        if math.log(1.0 - generator.random()) < log_density(proposal) - current:
            return proposal
        return state

    return kernel


def _checked_betas(betas: Sequence[float]) -> tuple[float, ...]:
    """Validate an increasing ladder that ends at one.

    References:
        [EQ-INF-PT-001], [REF-SYED-ETAL-2022].

    """
    ladder = tuple(require_finite_real(beta, name="beta") for beta in betas)
    if len(ladder) < 2:
        raise ValueError("betas needs at least 2 inverse temperatures")
    if ladder[0] < 0.0:
        raise ValueError("betas must be non-negative")
    if any(b <= a for a, b in zip(ladder, ladder[1:], strict=False)):
        raise ValueError("betas must be strictly increasing")
    if ladder[-1] != 1.0:
        raise ValueError("the last beta must be 1 (the target)")
    return ladder


def _log_mean_exp(values: Sequence[float]) -> float:
    """Return ``log((1/n) sum_i exp(v_i))`` without overflow.

    References:
        [EQ-INF-PT-002]. No new equation is introduced.

    """
    top = max(values)
    if not math.isfinite(top):
        return top
    total = math.fsum(math.exp(value - top) for value in values)
    return top + math.log(total / len(values))


def parallel_tempering(
    log_likelihood: LogDensity,
    log_prior: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    betas: Sequence[float],
    draws: int,
    warmup: int = 0,
    thin: int = 1,
    proposal_scale: float | Sequence[float] = 1.0,
    kernels: Sequence[LocalKernel] | None = None,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> ParallelTemperingResult:
    """Sample ``p(x) L(x)`` by parallel tempering with even-odd swaps.

    Replica ``i`` targets ``pi_i(x) ∝ p(x) L(x)^beta_i``. Every iteration
    (a) applies the local kernel of each temperature once and (b) proposes
    the swaps of the pairs ``(i, i + 1)`` with ``i`` even at even
    iterations and ``i`` odd at odd iterations, each accepted with
    probability
    ``min(1, exp((beta_i - beta_{i+1}) (l(x_{i+1}) - l(x_i))))``.
    The kept draws are those of the replica at ``beta = 1``.

    Args:
        log_likelihood: ``l(x) = log L(x)``; the part of the target that is
            tempered. ``-inf`` marks zero likelihood.
        log_prior: ``log p(x)``; the reference that stays at full strength.
            With ``beta_0 = 0`` it must be a proper density, and normalized
            if the evidence is wanted. ``-inf`` marks points outside the
            support.
        initial: One start shared by all replicas, or one start per
            temperature, with finite prior and likelihood.
        betas: Strictly increasing inverse temperatures ending at 1.
        draws: Kept cold-chain draws.
        warmup: Discarded initial iterations (nothing is adapted).
        thin: Keep every ``thin``-th post-warm-up iteration.
        proposal_scale: Standard deviation of the default isotropic
            random-walk Metropolis kernel: one value for all temperatures or
            one per temperature (hot replicas usually take larger steps).
            Ignored when ``kernels`` is given.
        kernels: Optional local kernels, one per temperature. A kernel is
            called as ``kernel(log_density, state, current_value,
            generator)`` and must return the next state of a Markov kernel
            that leaves ``exp(log_density)`` invariant; ``log_density`` is
            the tempered target of that replica.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.
        check_convergence: Run :func:`convergence_warnings` on the cold
            chain and emit each finding as a :class:`ConvergenceWarning`.

    Returns:
        Cold-chain draws, swap rejection rates, local acceptance rates,
        round trips, and the stepping-stone evidence when ``beta_0 = 0``.

    Raises:
        TypeError: If a density or kernel is not callable.
        ValueError: For invalid run lengths, an invalid ladder, a
            non-positive proposal scale, a wrong number of kernels or
            starts, a start with non-finite prior or likelihood, or a
            density returning NaN or ``+inf``.

    Equation:
        [EQ-INF-PT-001], [EQ-INF-PT-002].

    References:
        [REF-HUKUSHIMA-NEMOTO-1996], [REF-SYED-ETAL-2022].

    """
    likelihood = _checked_log_density(log_likelihood)
    prior = _checked_log_density(log_prior)
    draws, warmup, _, thin = _validate_run(draws, warmup, 1, thin)
    ladder = _checked_betas(betas)
    levels = len(ladder)
    scale_top = math.nan
    if kernels is None:
        scales: Vector
        if isinstance(proposal_scale, (int, float)):
            scales = (require_finite_real(proposal_scale, name="proposal_scale"),)
            scales = scales * levels
        else:
            scales = as_vector(proposal_scale, name="proposal_scale")
        if len(scales) != levels:
            raise ValueError("proposal_scale needs one value per temperature")
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("proposal_scale must be positive")
        scale_top = scales[-1] ** 2
        local: tuple[LocalKernel, ...] = tuple(_random_walk_kernel(s) for s in scales)
    else:
        local = tuple(kernels)
        if len(local) != levels:
            raise ValueError("kernels needs one kernel per temperature")
        if not all(callable(kernel) for kernel in local):
            raise TypeError("every kernel must be callable")
    states = list(_initial_points(initial, levels))
    dimension = len(states[0])
    checked_names = _names(names, dimension)

    target = _TemperedTarget(prior, likelihood)
    priors: list[float] = []
    likelihoods: list[float] = []
    for state in states:
        target.cache.clear()
        p, l_value = target.components(state)
        if not (math.isfinite(p) and math.isfinite(l_value)):
            raise ValueError(
                "every initial point must have a finite log-prior and log-likelihood"
            )
        priors.append(p)
        likelihoods.append(l_value)

    stream = resolve_stream(seed)
    generator = stream.generator()
    moved = [0] * levels
    swap_proposals = [0] * (levels - 1)
    swap_rejections = [0] * (levels - 1)
    # Round trips: each state carries a label, 0 = has not visited the
    # hottest replica yet, 1 = left the hottest replica and is on its way to
    # the target, 2 = has since reached the target and is on its way back.
    journey = [1 if index == 0 else 0 for index in range(levels)]
    round_trips = 0
    samples: list[Vector] = []
    trace: list[float] = []
    history: list[list[float]] = [[] for _ in range(levels)]
    total = warmup + draws * thin
    for iteration in range(total):
        keep = iteration >= warmup
        for i in range(levels):
            target.beta = ladder[i]
            target.cache.clear()
            target.cache[states[i]] = (priors[i], likelihoods[i])
            current = target.tempered(priors[i], likelihoods[i])
            new_state = tuple(
                float(x) for x in local[i](target, states[i], current, generator)
            )
            if new_state != states[i]:
                if len(new_state) != dimension:
                    raise ValueError("a kernel returned a state of the wrong dimension")
                priors[i], likelihoods[i] = target.components(new_state)
                states[i] = new_state
                if keep:
                    moved[i] += 1
        for i in range(iteration % 2, levels - 1, 2):
            log_ratio = (ladder[i] - ladder[i + 1]) * (
                likelihoods[i + 1] - likelihoods[i]
            )
            accept = (
                not math.isnan(log_ratio)
                and math.log(1.0 - generator.random()) < log_ratio
            )
            if keep:
                swap_proposals[i] += 1
                swap_rejections[i] += not accept
            if accept:
                states[i], states[i + 1] = states[i + 1], states[i]
                priors[i], priors[i + 1] = priors[i + 1], priors[i]
                likelihoods[i], likelihoods[i + 1] = likelihoods[i + 1], likelihoods[i]
                journey[i], journey[i + 1] = journey[i + 1], journey[i]
        if journey[0] == 2 and keep:
            round_trips += 1
        journey[0] = 1
        if journey[levels - 1] == 1:
            journey[levels - 1] = 2
        if keep:
            for i in range(levels):
                history[i].append(likelihoods[i])
            if (iteration - warmup + 1) % thin == 0:
                samples.append(states[levels - 1])
                trace.append(priors[levels - 1] + likelihoods[levels - 1])

    increments = tuple(
        _log_mean_exp([(ladder[i + 1] - ladder[i]) * value for value in history[i]])
        for i in range(levels - 1)
    )
    chain = MarkovChain(
        tuple(samples),
        tuple(trace),
        moved[levels - 1] / (draws * thin),
        math.nan,
        tuple(
            tuple(scale_top if i == j else 0.0 for j in range(dimension))
            for i in range(dimension)
        ),
        target.calls,
        stream.seed,
    )
    cold = _with_convergence_check(
        MCMCResult((chain,), checked_names, "parallel tempering", warmup, thin),
        check_convergence,
    )
    return ParallelTemperingResult(
        cold,
        ladder,
        tuple(
            rejected / proposed if proposed else math.nan
            for rejected, proposed in zip(swap_rejections, swap_proposals, strict=True)
        ),
        tuple(count / (draws * thin) for count in moved),
        round_trips,
        math.fsum(increments) if ladder[0] == 0.0 else None,
        increments,
        tuple(math.fsum(values) / len(values) for values in history),
        target.calls,
        stream.seed,
    )
