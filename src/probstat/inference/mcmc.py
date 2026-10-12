"""Generic Markov chain Monte Carlo, convergence diagnostics, and Laplace.

Everything here needs only a log-density callable ``log p(x)`` known up to an
additive constant, so the module is reusable by any higher layer (nonlinear
calibration, hierarchical models, or plain teaching examples).

Samplers
    :func:`random_walk_metropolis` uses a fixed Gaussian proposal covariance.
    :func:`adaptive_metropolis` learns that covariance from the chain history
    (Haario, Saksman, and Tamminen, 2001) **only during warm-up**. The proposal
    is frozen when warm-up ends, so every kept draw comes from one
    time-homogeneous Metropolis kernel whose stationary law is the target. The
    kept chain is therefore an ordinary Markov chain and the standard
    diagnostics below apply without the extra ergodicity theory that perpetual
    adaptation would require.

Diagnostics
    Split ``R-hat`` (Gelman and Rubin, 1992; BDA3 Section 11.4), the
    rank-normalized and folded ``R-hat`` of Vehtari et al. (2021), the effective
    sample size from Geyer's initial monotone sequence, the Monte Carlo
    standard error of the mean, and posterior summaries.

Approximation
    :func:`laplace_approximation` finds the mode by BFGS and uses the inverse of
    a finite-difference Hessian of ``-log p`` as a Gaussian covariance.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    inverse_spd,
    log_det_spd,
    matvec,
    solve_lower,
    symmetrize,
)
from probstat.optimization import OptimizationResult, central_hessian, minimize_bfgs
from probstat.statistics import median, quantile, ranks

LogDensity = Callable[[Vector], float]

_STANDARD_NORMAL = NormalDist()
_LOG_TWO_PI = math.log(2.0 * math.pi)
DEFAULT_PROBABILITIES = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkovChain:
    """Kept draws of one Markov chain after warm-up and thinning.

    Args:
        samples: Kept state vectors in iteration order.
        log_density: Target log-density at each kept state.
        acceptance_rate: Fraction of accepted proposals after warm-up.
        warmup_acceptance_rate: Fraction accepted during warm-up (``nan`` when
            there was no warm-up).
        proposal_covariance: Proposal covariance used for every kept draw.
        evaluations: Log-density evaluations spent by this chain.
        seed: Seed of the random stream that drove the chain.

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 7], [REF-GELMAN-ETAL-2013, Ch. 11].

    """

    samples: tuple[Vector, ...]
    log_density: tuple[float, ...]
    acceptance_rate: float
    warmup_acceptance_rate: float
    proposal_covariance: Matrix
    evaluations: int
    seed: int

    def column(self, index: int) -> tuple[float, ...]:
        """Return the draws of one coordinate.

        Args:
            index: Zero-based coordinate index.

        Returns:
            Scalar draws in iteration order.

        Raises:
            IndexError: If ``index`` is out of range.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        if not self.samples or not 0 <= index < len(self.samples[0]):
            raise IndexError("coordinate index out of range")
        return tuple(sample[index] for sample in self.samples)


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    """Posterior summary and convergence evidence for one scalar quantity.

    Args:
        name: Quantity label.
        mean: Pooled posterior mean estimate.
        standard_deviation: Pooled sample standard deviation.
        quantiles: ``(probability, value)`` pairs (type-7 quantiles).
        mcse_mean: Monte Carlo standard error of the mean.
        effective_sample_size: Multi-chain effective sample size.
        split_rhat: Classical split potential scale reduction factor.
        rank_normalized_rhat: Maximum of the bulk and folded (tail)
            rank-normalized split ``R-hat``.

    References:
        [REF-VEHTARI-ETAL-2021], [REF-GELMAN-ETAL-2013, Section 11.5].

    """

    name: str
    mean: float
    standard_deviation: float
    quantiles: tuple[tuple[float, float], ...]
    mcse_mean: float
    effective_sample_size: float
    split_rhat: float
    rank_normalized_rhat: float

    def quantile(self, probability: float) -> float:
        """Return a stored quantile.

        Args:
            probability: One of the summarized probability levels.

        Returns:
            Quantile value.

        Raises:
            KeyError: If the level was not summarized.

        References:
            [EQ-STAT-QUANT-001], [REF-HYNDMAN-FAN-1996].

        """
        for level, value in self.quantiles:
            if abs(level - probability) <= 1e-12:
                return value
        raise KeyError(f"quantile {probability} was not summarized")


@dataclass(frozen=True, slots=True)
class MCMCResult:
    """Several Markov chains targeting the same distribution.

    Args:
        chains: Chains with equal length and dimension.
        names: Distinct coordinate names.
        algorithm: Sampler label.
        warmup: Discarded warm-up iterations per chain.
        thin: Thinning interval applied after warm-up.

    Raises:
        ValueError: If chains are empty or have inconsistent shapes, or names
            are repeated or of the wrong length.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11], [REF-ROBERT-CASELLA-2004, Ch. 7].

    """

    chains: tuple[MarkovChain, ...]
    names: tuple[str, ...]
    algorithm: str
    warmup: int
    thin: int

    def __post_init__(self) -> None:
        """Validate chain shapes and coordinate names.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        chains = tuple(self.chains)
        if not chains:
            raise ValueError("at least one chain is required")
        length = len(chains[0].samples)
        if length == 0:
            raise ValueError("chains must contain at least one kept draw")
        dimension = len(chains[0].samples[0])
        for chain in chains:
            if len(chain.samples) != length:
                raise ValueError("all chains must have the same number of draws")
            if any(len(sample) != dimension for sample in chain.samples):
                raise ValueError("all draws must have the same dimension")
        names = tuple(self.names)
        if len(names) != dimension:
            raise ValueError("names must match the chain dimension")
        if len(set(names)) != len(names):
            raise ValueError("names must be distinct")
        object.__setattr__(self, "chains", chains)
        object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        """Return the state dimension.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7]. No equation is evaluated.

        """
        return len(self.names)

    @property
    def draws_per_chain(self) -> int:
        """Return the number of kept draws in each chain.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        return len(self.chains[0].samples)

    @property
    def acceptance_rate(self) -> float:
        """Return the mean post-warm-up acceptance rate across chains.

        References:
            [REF-ROBERTS-ROSENTHAL-2001]. No equation is evaluated.

        """
        return math.fsum(c.acceptance_rate for c in self.chains) / len(self.chains)

    @property
    def evaluations(self) -> int:
        """Return the total number of log-density evaluations.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7]. No equation is evaluated.

        """
        return sum(chain.evaluations for chain in self.chains)

    def index(self, parameter: int | str) -> int:
        """Resolve a coordinate name or index.

        Args:
            parameter: Coordinate name or zero-based index.

        Returns:
            Zero-based index.

        Raises:
            KeyError: If the name is unknown.
            IndexError: If the index is out of range.

        References:
            Project API policy. No equation is evaluated.

        """
        if isinstance(parameter, str):
            if parameter not in self.names:
                raise KeyError(f"unknown parameter {parameter!r}")
            return self.names.index(parameter)
        if not 0 <= parameter < self.dimension:
            raise IndexError("parameter index out of range")
        return parameter

    def chain_columns(self, parameter: int | str) -> tuple[tuple[float, ...], ...]:
        """Return one coordinate as a tuple of per-chain draw sequences.

        Args:
            parameter: Coordinate name or index.

        Returns:
            Per-chain scalar draws, suitable for the diagnostics functions.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        index = self.index(parameter)
        return tuple(chain.column(index) for chain in self.chains)

    def pooled(self) -> tuple[Vector, ...]:
        """Return all kept draws of all chains, chain after chain.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        return tuple(sample for chain in self.chains for sample in chain.samples)

    def mean(self) -> Vector:
        """Return the pooled posterior mean vector.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7].

        """
        pooled = self.pooled()
        return tuple(
            math.fsum(sample[i] for sample in pooled) / len(pooled)
            for i in range(self.dimension)
        )

    def covariance(self) -> Matrix:
        """Return the pooled unbiased sample covariance matrix.

        Raises:
            ValueError: If fewer than two draws are available.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7].

        """
        pooled = self.pooled()
        if len(pooled) < 2:
            raise ValueError("covariance requires at least two draws")
        center = self.mean()
        return _sample_covariance(pooled, center)

    def summary(
        self, probabilities: Sequence[float] = DEFAULT_PROBABILITIES
    ) -> tuple[ParameterSummary, ...]:
        """Summarize every coordinate with moments, quantiles, and diagnostics.

        Args:
            probabilities: Quantile levels in ``[0, 1]``.

        Returns:
            One :class:`ParameterSummary` per coordinate.

        Raises:
            ValueError: If chains have fewer than four draws.

        Equation:
            [EQ-MCMC-RHAT-001], [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001],
            [EQ-MCMC-MCSE-001].

        References:
            [REF-VEHTARI-ETAL-2021], [REF-GELMAN-ETAL-2013, Section 11.5].

        """
        return tuple(
            summarize_draws(
                self.chain_columns(index), name=name, probabilities=probabilities
            )
            for index, name in enumerate(self.names)
        )

    def transformed(
        self,
        function: Callable[[Vector], Sequence[float]],
        names: Sequence[str],
        *,
        algorithm: str | None = None,
    ) -> MCMCResult:
        """Map every draw through a deterministic function.

        A deterministic function of a Markov chain is a valid sample of the
        push-forward law, so diagnostics can be recomputed on the new scale.
        The stored log-density trace is left on the original sampling scale.

        Args:
            function: Map from a state vector to a new vector.
            names: Names of the output coordinates.
            algorithm: Optional replacement label.

        Returns:
            New result with transformed draws.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7]. No new equation is introduced.

        """
        chains = tuple(
            MarkovChain(
                tuple(
                    as_vector(function(sample), name="transformed sample")
                    for sample in chain.samples
                ),
                chain.log_density,
                chain.acceptance_rate,
                chain.warmup_acceptance_rate,
                chain.proposal_covariance,
                chain.evaluations,
                chain.seed,
            )
            for chain in self.chains
        )
        return MCMCResult(
            chains,
            tuple(names),
            self.algorithm if algorithm is None else algorithm,
            self.warmup,
            self.thin,
        )


@dataclass(frozen=True, slots=True)
class LaplaceApproximation:
    """Gaussian approximation ``N(mode, H^-1)`` of a log-density.

    Args:
        mode: Located maximizer of the log-density.
        covariance: Inverse of the Hessian of ``-log p`` at ``mode``.
        log_density_at_mode: ``log p(mode)``.
        log_normalizer: Laplace estimate of ``log integral exp(log p)``.
        converged: Whether the BFGS mode search met its tolerance.
        evaluations: Log-density evaluations (search plus Hessian).
        optimization: Raw BFGS result.

    References:
        [REF-TIERNEY-KADANE-1986], [REF-GELMAN-ETAL-2013, Section 4.1].

    """

    mode: Vector
    covariance: Matrix
    log_density_at_mode: float
    log_normalizer: float
    converged: bool
    evaluations: int
    optimization: OptimizationResult

    @property
    def dimension(self) -> int:
        """Return the parameter dimension.

        References:
            [REF-TIERNEY-KADANE-1986]. No equation is evaluated.

        """
        return len(self.mode)

    @property
    def standard_deviations(self) -> Vector:
        """Return marginal approximate posterior standard deviations.

        Equation:
            [EQ-LAPLACE-001].

        References:
            [REF-TIERNEY-KADANE-1986].

        """
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(self.dimension))

    def log_pdf(self, x: Sequence[float]) -> float:
        """Return the normalized Gaussian log-density of the approximation.

        Args:
            x: Finite vector.

        Returns:
            ``log N(x; mode, covariance)``.

        Equation:
            [EQ-LAPLACE-001], [EQ-DIST-MVN-001].

        References:
            [REF-TIERNEY-KADANE-1986].

        """
        point = as_vector(x, name="x")
        if len(point) != self.dimension:
            raise ValueError("x must match the approximation dimension")
        lower = cholesky(self.covariance)
        white = solve_lower(
            lower, tuple(a - b for a, b in zip(point, self.mode, strict=True))
        )
        return -0.5 * (
            math.fsum(w * w for w in white)
            + self.dimension * _LOG_TWO_PI
            + log_det_spd(self.covariance)
        )

    def sample(
        self, size: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw independent vectors from the Gaussian approximation.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream.

        Returns:
            Draws ``mode + L z`` with ``L L' = covariance``.

        Equation:
            [EQ-LAPLACE-001], [EQ-BAYES-POST-SAMPLE-001].

        References:
            [REF-DEVROYE-1986, Chapter XI].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        lower = cholesky(self.covariance)
        draws: list[Vector] = []
        for _ in range(count):
            z = tuple(generator.gauss(0.0, 1.0) for _ in range(self.dimension))
            step = matvec(lower, z)
            draws.append(tuple(m + s for m, s in zip(self.mode, step, strict=True)))
        return tuple(draws)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def _checked_log_density(function: LogDensity) -> LogDensity:
    """Wrap a log-density so NaN and ``+inf`` raise instead of misleading.

    References:
        [REF-ROBERT-CASELLA-2004, Section 7.3]. No equation is evaluated.

    """
    if not callable(function):
        raise TypeError("log_density must be callable")

    def checked(point: Vector) -> float:
        """Evaluate and validate one log-density value.

        References:
            [REF-ROBERT-CASELLA-2004, Section 7.3]. No equation is evaluated.

        """
        value = float(function(point))
        if math.isnan(value):
            raise ValueError("log_density returned NaN")
        if value == math.inf:
            raise ValueError("log_density returned +inf")
        return value

    return checked


def _initial_points(
    initial: Sequence[float] | Sequence[Sequence[float]], chains: int
) -> tuple[Vector, ...]:
    """Normalize one shared start or one start per chain.

    References:
        [REF-GELMAN-RUBIN-1992]. No equation is evaluated.

    """
    items: list[object] = list(initial)
    if not items:
        raise ValueError("initial must be non-empty")
    nested = [
        item
        for item in items
        if isinstance(item, Sequence) and not isinstance(item, str)
    ]
    if len(nested) == len(items):
        points = tuple(as_vector(item, name="initial point") for item in nested)
        if len(points) != chains:
            raise ValueError("provide one initial point per chain")
        dimension = len(points[0])
        if dimension == 0 or any(len(point) != dimension for point in points):
            raise ValueError("initial points must share a positive dimension")
        return points
    point = as_vector(items, name="initial")
    return tuple(point for _ in range(chains))


def _checked_covariance(
    covariance: Sequence[Sequence[float]], dimension: int
) -> Matrix:
    """Validate a symmetric positive-definite proposal covariance.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    matrix = symmetrize(covariance, name="proposal_covariance")
    if len(matrix) != dimension:
        raise ValueError("proposal_covariance must match the state dimension")
    try:
        cholesky(matrix)
    except ValueError as error:
        raise ValueError("proposal_covariance must be positive definite") from error
    return matrix


def _average_transpose(matrix: Matrix) -> Matrix:
    """Return ``(A + A') / 2`` to remove rounding asymmetry.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    size = len(matrix)
    return tuple(
        tuple(0.5 * (matrix[i][j] + matrix[j][i]) for j in range(size))
        for i in range(size)
    )


def _sample_covariance(points: Sequence[Vector], center: Vector) -> Matrix:
    """Return the unbiased sample covariance about a given center.

    References:
        [EQ-MC-VAR-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    dimension = len(center)
    count = len(points)
    return tuple(
        tuple(
            math.fsum(
                (point[i] - center[i]) * (point[j] - center[j]) for point in points
            )
            / (count - 1)
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def _run_chain(
    log_density: LogDensity,
    start: Vector,
    *,
    draws: int,
    warmup: int,
    thin: int,
    covariance: Matrix,
    adapt: bool,
    scale: float,
    regularization: float,
    adaptation_start: int,
    adaptation_interval: int,
    stream: RandomStream,
) -> MarkovChain:
    """Run one random-walk Metropolis chain, optionally adapting in warm-up.

    The proposal is ``x' = x + L z`` with ``z ~ N(0, I)`` and ``L L' = C``. The
    move is accepted when ``log U < log p(x') - log p(x)``. With ``adapt``, the
    covariance becomes ``s_d (Cov(history) + eps I)`` during warm-up only.

    References:
        [EQ-MCMC-MH-001], [EQ-MCMC-AM-001], [REF-METROPOLIS-ETAL-1953],
        [REF-HAARIO-SAKSMAN-TAMMINEN-2001].

    """
    generator = stream.generator()
    dimension = len(start)
    evaluations = 0

    def evaluate(point: Vector) -> float:
        """Count and evaluate the target log-density.

        References:
            [EQ-MCMC-MH-001]. No new equation is introduced.

        """
        nonlocal evaluations
        evaluations += 1
        return log_density(point)

    state = start
    current = evaluate(state)
    if not math.isfinite(current):
        raise ValueError("initial point must have a finite log-density")
    lower = cholesky(covariance)
    proposal_covariance = covariance
    running_mean = list(state)
    scatter = [[0.0] * dimension for _ in range(dimension)]
    history = 1
    warm_accepted = 0
    kept_accepted = 0
    samples: list[Vector] = []
    trace: list[float] = []
    total = warmup + draws * thin
    for iteration in range(total):
        z = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
        step = matvec(lower, z)
        proposal = tuple(a + b for a, b in zip(state, step, strict=True))
        proposed = evaluate(proposal)
        accepted = math.log(1.0 - generator.random()) < proposed - current
        if accepted:
            state, current = proposal, proposed
        if iteration < warmup:
            warm_accepted += accepted
            if adapt:
                history += 1
                delta = [state[i] - running_mean[i] for i in range(dimension)]
                for i in range(dimension):
                    running_mean[i] += delta[i] / history
                after = [state[i] - running_mean[i] for i in range(dimension)]
                for i in range(dimension):
                    for j in range(dimension):
                        scatter[i][j] += delta[i] * after[j]
                if (
                    history >= adaptation_start
                    and (iteration + 1) % adaptation_interval == 0
                ):
                    candidate = tuple(
                        tuple(
                            scale
                            * (
                                0.5 * (scatter[i][j] + scatter[j][i]) / (history - 1)
                                + (regularization if i == j else 0.0)
                            )
                            for j in range(dimension)
                        )
                        for i in range(dimension)
                    )
                    try:
                        lower = cholesky(candidate)
                        proposal_covariance = candidate
                    except ValueError:
                        pass
        else:
            kept_accepted += accepted
            if (iteration - warmup + 1) % thin == 0:
                samples.append(state)
                trace.append(current)
    kept_iterations = draws * thin
    return MarkovChain(
        tuple(samples),
        tuple(trace),
        kept_accepted / kept_iterations,
        warm_accepted / warmup if warmup else math.nan,
        proposal_covariance,
        evaluations,
        stream.seed,
    )


def _validate_run(
    draws: int, warmup: int, chains: int, thin: int
) -> tuple[int, int, int, int]:
    """Validate run-length arguments.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked_draws = require_non_negative_integer(draws, name="draws")
    checked_warmup = require_non_negative_integer(warmup, name="warmup")
    checked_chains = require_non_negative_integer(chains, name="chains")
    checked_thin = require_non_negative_integer(thin, name="thin")
    if checked_draws == 0:
        raise ValueError("draws must be positive")
    if checked_chains == 0:
        raise ValueError("chains must be positive")
    if checked_thin == 0:
        raise ValueError("thin must be positive")
    return checked_draws, checked_warmup, checked_chains, checked_thin


def _names(names: Sequence[str] | None, dimension: int) -> tuple[str, ...]:
    """Return coordinate names, defaulting to ``x0, x1, ...``.

    References:
        Project API policy. No equation is evaluated.

    """
    if names is None:
        return tuple(f"x{i}" for i in range(dimension))
    return tuple(names)


def random_walk_metropolis(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    proposal_covariance: Sequence[Sequence[float]],
    draws: int,
    warmup: int = 0,
    chains: int = 1,
    thin: int = 1,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> MCMCResult:
    """Sample a log-density by random-walk Metropolis with a Gaussian proposal.

    Args:
        log_density: Target ``log p(x)`` up to a constant; ``-inf`` marks
            states outside the support (always rejected).
        initial: One start shared by all chains, or one start per chain.
        proposal_covariance: Symmetric positive-definite proposal covariance.
        draws: Kept draws per chain.
        warmup: Discarded initial iterations per chain.
        chains: Number of independent chains, seeded by ``RandomStream.spawn``.
        thin: Keep every ``thin``-th post-warm-up state.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.

    Returns:
        Multi-chain result with acceptance rates and evaluation counts.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid run lengths, covariance, or a start with
            non-finite log-density, or if the target returns NaN/``+inf``.

    Equation:
        [EQ-MCMC-MH-001].

    References:
        [REF-METROPOLIS-ETAL-1953], [REF-HASTINGS-1970],
        [REF-ROBERT-CASELLA-2004, Section 7.5].

    """
    checked = _checked_log_density(log_density)
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    covariance = _checked_covariance(proposal_covariance, dimension)
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_chain(
            checked,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            covariance=covariance,
            adapt=False,
            scale=1.0,
            regularization=0.0,
            adaptation_start=0,
            adaptation_interval=1,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    return MCMCResult(
        results, _names(names, dimension), "random-walk Metropolis", warmup, thin
    )


def adaptive_metropolis(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    draws: int,
    warmup: int,
    chains: int = 4,
    thin: int = 1,
    initial_covariance: Sequence[Sequence[float]] | None = None,
    scale: float | None = None,
    regularization: float = 1e-10,
    adaptation_start: int | None = None,
    adaptation_interval: int = 10,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> MCMCResult:
    """Sample by adaptive Metropolis with adaptation confined to warm-up.

    During warm-up the Gaussian proposal covariance is periodically reset to
    ``s_d (C_t + eps I)``, where ``C_t`` is the running covariance of the
    chain history (Welford update) and ``s_d = 2.38^2 / d`` by default. At the
    end of warm-up the covariance is frozen; kept draws therefore come from a
    fixed Metropolis kernel that leaves the target invariant, which is what
    makes the usual Markov-chain diagnostics valid. Each chain adapts from its
    own history only, so chains remain independent.

    Args:
        log_density: Target ``log p(x)`` up to a constant.
        initial: One start shared by all chains, or one start per chain.
        draws: Kept draws per chain.
        warmup: Adaptation-and-burn-in iterations per chain (positive).
        chains: Number of independent chains.
        thin: Keep every ``thin``-th post-warm-up state.
        initial_covariance: Proposal covariance before adaptation starts;
            defaults to ``0.01 I``.
        scale: Adaptation factor ``s_d``; defaults to ``2.38^2 / d``.
        regularization: ``eps > 0`` keeping the adapted covariance definite.
        adaptation_start: History length before adaptation starts; defaults to
            ``max(20, 10 d)``.
        adaptation_interval: Recompute the covariance every this many steps.
        seed: Integer seed or stream.
        names: Optional coordinate names.

    Returns:
        Multi-chain result; each chain records its frozen proposal covariance.

    Raises:
        ValueError: For invalid arguments or a non-finite starting density.

    Equation:
        [EQ-MCMC-MH-001], [EQ-MCMC-AM-001].

    References:
        [REF-HAARIO-SAKSMAN-TAMMINEN-2001], [REF-ROBERTS-ROSENTHAL-2001],
        [REF-METROPOLIS-ETAL-1953].

    """
    checked = _checked_log_density(log_density)
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    if warmup == 0:
        raise ValueError("adaptive_metropolis needs a positive warmup")
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    if initial_covariance is None:
        covariance = tuple(
            tuple(0.01 if i == j else 0.0 for j in range(dimension))
            for i in range(dimension)
        )
    else:
        covariance = _checked_covariance(initial_covariance, dimension)
    factor = 2.38**2 / dimension if scale is None else scale
    factor = require_finite_real(factor, name="scale")
    if factor <= 0.0:
        raise ValueError("scale must be positive")
    epsilon = require_finite_real(regularization, name="regularization")
    if epsilon <= 0.0:
        raise ValueError("regularization must be positive")
    start_at = (
        max(20, 10 * dimension)
        if adaptation_start is None
        else require_non_negative_integer(adaptation_start, name="adaptation_start")
    )
    start_at = max(start_at, 2)
    interval = require_non_negative_integer(
        adaptation_interval, name="adaptation_interval"
    )
    if interval == 0:
        raise ValueError("adaptation_interval must be positive")
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_chain(
            checked,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            covariance=covariance,
            adapt=True,
            scale=factor,
            regularization=epsilon,
            adaptation_start=start_at,
            adaptation_interval=interval,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    return MCMCResult(
        results, _names(names, dimension), "adaptive Metropolis", warmup, thin
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _validated_chains(
    chains: Sequence[Sequence[float]], *, minimum: int
) -> tuple[tuple[float, ...], ...]:
    """Materialize equal-length finite scalar chains.

    References:
        [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

    """
    materialized = tuple(
        tuple(require_finite_real(value, name="draw") for value in chain)
        for chain in chains
    )
    if not materialized:
        raise ValueError("at least one chain is required")
    length = len(materialized[0])
    if any(len(chain) != length for chain in materialized):
        raise ValueError("all chains must have the same length")
    if length < minimum:
        raise ValueError(f"each chain needs at least {minimum} draws")
    return materialized


def _split_chains(
    chains: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Split each chain into first and second halves (middle draw dropped).

    References:
        [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

    """
    halves: list[tuple[float, ...]] = []
    for chain in chains:
        half = len(chain) // 2
        halves.append(chain[:half])
        halves.append(chain[len(chain) - half :])
    return tuple(halves)


def _potential_scale_reduction(chains: tuple[tuple[float, ...], ...]) -> float:
    """Return ``sqrt(var+ / W)`` for already split chains.

    References:
        [EQ-MCMC-RHAT-001], [REF-GELMAN-RUBIN-1992].

    """
    m = len(chains)
    n = len(chains[0])
    means = [math.fsum(chain) / n for chain in chains]
    variances = [
        math.fsum((value - center) ** 2 for value in chain) / (n - 1)
        for chain, center in zip(chains, means, strict=True)
    ]
    within = math.fsum(variances) / m
    grand = math.fsum(means) / m
    between = n * math.fsum((center - grand) ** 2 for center in means) / (m - 1)
    if within == 0.0:
        return math.nan if between == 0.0 else math.inf
    pooled = (n - 1) / n * within + between / n
    return math.sqrt(pooled / within)


def split_rhat(chains: Sequence[Sequence[float]]) -> float:
    """Return the split potential scale reduction factor ``R-hat``.

    Each chain is split in half so that within-chain drift also inflates the
    statistic. Values near 1 are necessary (not sufficient) for convergence;
    a common rule is ``R-hat < 1.01`` (Vehtari et al.) or ``< 1.1`` (BDA3).

    Args:
        chains: One or more equal-length scalar chains, at least 4 draws each.

    Returns:
        ``sqrt(var+ / W)``; ``nan`` if every draw is identical.

    Raises:
        ValueError: If chains are empty, unequal, too short, or non-finite.

    Equation:
        [EQ-MCMC-RHAT-001].

    References:
        [REF-GELMAN-RUBIN-1992], [REF-GELMAN-ETAL-2013, Section 11.4].

    """
    checked = _validated_chains(chains, minimum=4)
    return _potential_scale_reduction(_split_chains(checked))


def _rank_normalize(
    chains: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Replace pooled draws by normal scores of their average ranks.

    References:
        [EQ-MCMC-RHAT-002], [REF-VEHTARI-ETAL-2021].

    """
    pooled = [value for chain in chains for value in chain]
    size = len(pooled)
    scores = [
        _STANDARD_NORMAL.inv_cdf((rank - 0.375) / (size + 0.25))
        for rank in ranks(pooled)
    ]
    length = len(chains[0])
    return tuple(
        tuple(scores[k * length : (k + 1) * length]) for k in range(len(chains))
    )


def rank_normalized_rhat(chains: Sequence[Sequence[float]]) -> float:
    """Return the rank-normalized, folded split ``R-hat`` of Vehtari et al.

    Draws are split, pooled, replaced by normal scores
    ``Phi^-1((r - 3/8) / (S + 1/4))`` of their average ranks ``r``, and split
    ``R-hat`` is computed (bulk). The same is done for the folded draws
    ``|x - median(x)|`` (tail). The larger of the two is returned, which is
    robust to heavy tails and detects chains that differ only in scale.

    Args:
        chains: One or more equal-length scalar chains, at least 4 draws each.

    Returns:
        ``max(R-hat_bulk, R-hat_tail)``.

    Raises:
        ValueError: If chains are empty, unequal, too short, or non-finite.

    Equation:
        [EQ-MCMC-RHAT-002].

    References:
        [REF-VEHTARI-ETAL-2021].

    """
    checked = _split_chains(_validated_chains(chains, minimum=4))
    bulk = _potential_scale_reduction(_rank_normalize(checked))
    center = median([value for chain in checked for value in chain])
    folded = tuple(tuple(abs(value - center) for value in chain) for chain in checked)
    tail = _potential_scale_reduction(_rank_normalize(folded))
    if math.isnan(bulk) or math.isnan(tail):
        return math.nan
    return max(bulk, tail)


def effective_sample_size(chains: Sequence[Sequence[float]]) -> float:
    """Return the multi-chain effective sample size of a scalar quantity.

    With within-chain autocovariances ``c_{t,m}`` (divisor ``n``), ``W`` the
    mean within-chain variance and ``var+ = (n-1)/n W + B/n``, the combined
    autocorrelation is ``rho_t = 1 - (W - mean_m c_{t,m}) / var+`` and
    ``rho_0 = 1``. Geyer's initial positive sequence adds the pairs
    ``(rho_t, rho_{t+1})``, ``t`` even, while their sum is positive (lags
    ``t < n - 5``), the initial monotone sequence makes the pair sums
    non-increasing, and with the stopping lag ``T``
    ``tau = -1 + 2 sum_{t < T} rho_t + rho_T`` (``rho_T`` only when
    positive: the improved estimate of the reference code, which reduces
    the variance for antithetic chains) and ``ESS = M n / tau``. This is the
    algorithm of the ``.ess`` function of the stan-dev ``posterior``
    package, step for step; for ``n <= 5`` it gives ``tau = 2``.

    Chains are used as given (not split). Antithetic chains can give
    ``ESS > M n``; ``tau`` is bounded below by ``1 / log10(M n)`` (the
    safety bound of the reference code), so the ESS is always positive and
    at most ``M n log10(M n)``.

    Args:
        chains: One or more equal-length scalar chains, at least 4 draws each.

    Returns:
        Effective sample size; ``nan`` if every draw is identical.

    Raises:
        ValueError: If chains are empty, unequal, too short, or non-finite.

    Equation:
        [EQ-MCMC-ESS-001].

    References:
        [REF-GEYER-1992], [REF-VEHTARI-ETAL-2021],
        [REF-BURKNER-ETAL-POSTERIOR, function ``.ess`` in R/convergence.R],
        [REF-GELMAN-ETAL-2013, Section 11.5].

    """
    checked = _validated_chains(chains, minimum=4)
    m = len(checked)
    n = len(checked[0])
    means = [math.fsum(chain) / n for chain in checked]
    centered = [
        [value - center for value in chain]
        for chain, center in zip(checked, means, strict=True)
    ]

    def mean_autocovariance(lag: int) -> float:
        """Return the chain-averaged biased autocovariance at one lag.

        References:
            [EQ-MCMC-ESS-001], [REF-GEYER-1992].

        """
        sums = (sum(map(operator.mul, c[: n - lag], c[lag:])) for c in centered)
        return math.fsum(sums) / (n * m)

    within = mean_autocovariance(0) * n / (n - 1)
    pooled = within * (n - 1) / n
    if m > 1:
        grand = math.fsum(means) / m
        pooled += math.fsum((center - grand) ** 2 for center in means) / (m - 1)
    if pooled <= 0.0:
        return math.nan

    def rho(lag: int) -> float:
        """Return the combined autocorrelation estimate at one lag.

        References:
            [EQ-MCMC-ESS-001], [REF-VEHTARI-ETAL-2021].

        """
        if lag == 0:
            return 1.0
        return 1.0 - (within - mean_autocovariance(lag)) / pooled

    # Geyer's initial positive sequence, exactly as stan-dev/posterior's
    # .ess (R/convergence.R): pairs (rho_t, rho_{t+1}), t even, are added
    # while their sum is positive, scanning t < n - 5; a pair with a
    # negative sum is not stored.
    rho_hat = [0.0] * n
    rho_even = 1.0
    rho_odd = rho(1)
    rho_hat[0] = rho_even
    rho_hat[1] = rho_odd
    t = 0
    while t < n - 5 and rho_even + rho_odd > 0.0:
        t += 2
        rho_even = rho(t)
        rho_odd = rho(t + 1)
        if rho_even + rho_odd >= 0.0:
            rho_hat[t] = rho_even
            rho_hat[t + 1] = rho_odd
    max_t = t
    # The improved estimate uses the even autocorrelation at the stopping
    # lag when it is positive.
    if rho_even > 0.0:
        rho_hat[max_t] = rho_even
    # Geyer's initial monotone sequence: a pair larger than its predecessor
    # is replaced by the predecessor's mean.
    t = 0
    while t <= max_t - 4:
        t += 2
        if rho_hat[t] + rho_hat[t + 1] > rho_hat[t - 2] + rho_hat[t - 1]:
            rho_hat[t] = (rho_hat[t - 2] + rho_hat[t - 1]) / 2.0
            rho_hat[t + 1] = rho_hat[t]
    # tau = -1 + 2 sum_{t < max_t} rho_t + rho_{max_t}: Geyer's truncated sum
    # plus the improved-estimate term, which reduces the variance in the
    # antithetic case. The R code sums rho_hat_t[1:max_t]; for max_t = 0
    # (always when n <= 5) that range is c(1, 0), i.e. rho_0 alone, so
    # tau = -1 + 2 + 1 = 2 then.
    truncated = math.fsum(rho_hat[:max_t]) if max_t > 0 else rho_hat[0]
    tau = -1.0 + 2.0 * truncated + rho_hat[max_t]
    # Safety bound of the reference code: tau >= 1 / log10(M n), so the ESS
    # is positive and at most M n log10(M n) even for strongly antithetic
    # chains (lag-one autocorrelation near -1, as static HMC can produce).
    tau = max(tau, 1.0 / math.log10(m * n))
    return m * n / tau


def monte_carlo_standard_error(chains: Sequence[Sequence[float]]) -> float:
    """Return the Monte Carlo standard error of the posterior-mean estimate.

    Args:
        chains: One or more equal-length scalar chains, at least 4 draws each.

    Returns:
        ``sd / sqrt(ESS)`` using the pooled sample standard deviation.

    Raises:
        ValueError: If chains are empty, unequal, too short, or non-finite.

    Equation:
        [EQ-MCMC-MCSE-001].

    References:
        [REF-GEYER-1992], [REF-VEHTARI-ETAL-2021].

    """
    checked = _validated_chains(chains, minimum=4)
    pooled = [value for chain in checked for value in chain]
    center = math.fsum(pooled) / len(pooled)
    variance = math.fsum((v - center) ** 2 for v in pooled) / (len(pooled) - 1)
    ess = effective_sample_size(checked)
    if math.isnan(ess):
        return math.nan
    return math.sqrt(variance / ess)


def summarize_draws(
    chains: Sequence[Sequence[float]],
    *,
    name: str = "x",
    probabilities: Sequence[float] = DEFAULT_PROBABILITIES,
) -> ParameterSummary:
    """Summarize one scalar quantity from one or more chains.

    Args:
        chains: Equal-length scalar chains, at least 4 draws each.
        name: Quantity label.
        probabilities: Quantile levels in ``[0, 1]``.

    Returns:
        Mean, standard deviation, quantiles, MCSE, ESS, and both ``R-hat``s.

    Raises:
        ValueError: If chains are invalid or a probability is outside
            ``[0, 1]``.

    Equation:
        [EQ-MCMC-RHAT-001], [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001],
        [EQ-MCMC-MCSE-001], [EQ-STAT-QUANT-001].

    References:
        [REF-VEHTARI-ETAL-2021], [REF-GELMAN-ETAL-2013, Section 11.5].

    """
    checked = _validated_chains(chains, minimum=4)
    pooled = sorted(value for chain in checked for value in chain)
    center = math.fsum(pooled) / len(pooled)
    deviation = math.sqrt(
        math.fsum((v - center) ** 2 for v in pooled) / (len(pooled) - 1)
    )
    levels = tuple(require_finite_real(p, name="probability") for p in probabilities)
    quantile_pairs = tuple((p, quantile(pooled, p)) for p in levels)
    ess = effective_sample_size(checked)
    mcse = math.nan if math.isnan(ess) else deviation / math.sqrt(ess)
    return ParameterSummary(
        name,
        center,
        deviation,
        quantile_pairs,
        mcse,
        ess,
        split_rhat(checked),
        rank_normalized_rhat(checked),
    )


# ---------------------------------------------------------------------------
# Laplace approximation
# ---------------------------------------------------------------------------


def laplace_approximation(
    log_density: LogDensity,
    initial: Sequence[float],
    *,
    gradient: Callable[[Vector], Sequence[float]] | None = None,
    relative_step: float = 1e-4,
    max_iterations: int = 500,
) -> LaplaceApproximation:
    """Approximate a log-density by a Gaussian at its mode.

    The mode ``x*`` minimizes ``-log p`` by BFGS; ``H`` is the central
    finite-difference Hessian of ``-log p`` at ``x*``; the approximation is
    ``N(x*, H^-1)`` and
    ``log Z ~= log p(x*) + (d/2) log(2 pi) - (1/2) log det H``.
    It is exact (up to finite-difference rounding) for Gaussian targets.

    Args:
        log_density: Target ``log p(x)`` up to a constant.
        initial: Starting point with finite log-density.
        gradient: Optional analytic gradient of ``log p``.
        relative_step: Hessian finite-difference step relative to
            ``max(1, |x_i|)``.
        max_iterations: BFGS iteration limit.

    Returns:
        Laplace approximation with mode, covariance, log normalizer, and
        convergence flag.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: If the start has non-finite density, or the Hessian at the
            located point is not positive definite (no strict local maximum).

    Equation:
        [EQ-LAPLACE-001], [EQ-OPT-BFGS-001], [EQ-OPT-FD-002].

    References:
        [REF-TIERNEY-KADANE-1986], [REF-GELMAN-ETAL-2013, Section 4.1],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    checked = _checked_log_density(log_density)
    start = as_vector(initial, name="initial")
    if not start:
        raise ValueError("initial must be non-empty")
    if not math.isfinite(checked(start)):
        raise ValueError("initial point must have a finite log-density")

    def negative(point: Vector) -> float:
        """Return ``-log p``; ``+inf`` outside the support raises in BFGS.

        References:
            [EQ-LAPLACE-001]. No new equation is introduced.

        """
        return -checked(point)

    negative_gradient: Callable[[Vector], Sequence[float]] | None = None
    if gradient is not None:
        supplied = gradient

        def negative_gradient(point: Vector) -> Sequence[float]:
            """Return the gradient of ``-log p``.

            References:
                [EQ-LAPLACE-001]. No new equation is introduced.

            """
            return tuple(-value for value in supplied(point))

    optimization = minimize_bfgs(
        negative, start, gradient=negative_gradient, max_iterations=max_iterations
    )
    mode = optimization.x
    hessian, used = central_hessian(negative, mode, relative_step=relative_step)
    try:
        covariance = _average_transpose(inverse_spd(hessian))
        log_det_hessian = log_det_spd(hessian)
    except ValueError as error:
        raise ValueError(
            "Hessian of -log p is not positive definite at the located point"
        ) from error
    value = -optimization.value
    dimension = len(mode)
    log_normalizer = value + 0.5 * dimension * _LOG_TWO_PI - 0.5 * log_det_hessian
    return LaplaceApproximation(
        mode,
        covariance,
        value,
        log_normalizer,
        optimization.converged,
        optimization.evaluations + used + 1,
        optimization,
    )
