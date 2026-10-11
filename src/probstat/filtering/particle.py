"""Bootstrap (sampling-importance-resampling) particle filtering.

The bootstrap filter represents ``p(x_k | y_1:k)`` by weighted samples. Each
step propagates particles through the dynamics with sampled process noise
(the transition prior is the proposal), multiplies the weights by the
observation likelihood, and resamples when the effective sample size falls
below a declared fraction of the particle count. Unlike the Kalman family it
needs no Gaussian or linear assumption about the posterior, but its error
decreases only like ``N^-1/2`` and degenerates in high state or observation
dimension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    solve_lower,
)

from .kalman import Observation, _observation_vector
from .state_space import StateSpaceModel, _affine, _control_plan

_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class ParticleFilterResult:
    """Forward-pass record of a bootstrap particle filter.

    Args:
        means: Weighted particle means for ``k = 0..T``.
        covariances: Weighted particle covariances for ``k = 0..T``.
        effective_sample_sizes: ESS after weighting at ``k = 1..T``.
        resampled: Whether resampling followed step ``k = 1..T``.
        log_likelihood_terms: Estimates of ``log p(y_k | y_1:k-1)``.
        particles: Final particle locations.
        weights: Final normalized weights.
        particle_count: Number of particles ``N``.

    References:
        [REF-GORDON-SALMOND-SMITH-1993], [REF-DOUCET-JOHANSEN-2011].

    """

    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    effective_sample_sizes: tuple[float, ...]
    resampled: tuple[bool, ...]
    log_likelihood_terms: tuple[float, ...]
    particles: tuple[Vector, ...]
    weights: tuple[float, ...]
    particle_count: int

    @property
    def log_likelihood(self) -> float:
        """Return the particle estimate of ``log p(y_1:T)``.

        The likelihood estimate itself is unbiased; its logarithm is biased
        downward by roughly half its variance.

        Equation:
            [EQ-FILT-PF-001].

        References:
            [REF-DOUCET-JOHANSEN-2011].

        """
        return math.fsum(self.log_likelihood_terms)


def _normalized(weights: Sequence[float]) -> tuple[float, ...]:
    """Validate non-negative weights and normalize them to sum to one.

    Args:
        weights: Non-negative finite weights with a positive sum.

    Returns:
        Normalized weights.

    Raises:
        ValueError: If weights are empty, negative, or sum to zero.

    References:
        [REF-DOUCET-JOHANSEN-2011]. No equation is evaluated.

    """
    values = tuple(
        require_finite_real(value, name=f"weights[{index}]")
        for index, value in enumerate(weights)
    )
    if not values:
        raise ValueError("weights must be non-empty")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    return tuple(value / total for value in values)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Return the effective sample size ``1 / sum_i w_i^2``.

    Args:
        weights: Non-negative importance weights (normalized internally).

    Returns:
        ESS between 1 and ``N``.

    Raises:
        ValueError: If the weights are invalid.

    Equation:
        [EQ-FILT-PF-002].

    References:
        [REF-DOUCET-JOHANSEN-2011].

    """
    normalized = _normalized(weights)
    return 1.0 / math.fsum(value * value for value in normalized)


def systematic_resample(weights: Sequence[float], offset: float) -> tuple[int, ...]:
    """Return ancestor indices by systematic resampling.

    Positions ``(offset + j) / N`` for ``j = 0..N-1`` are located in the
    cumulative weights. With ``offset ~ Uniform[0, 1)`` every particle ``i``
    receives ``floor(N w_i)`` or ``ceil(N w_i)`` copies, and the expected
    count is exactly ``N w_i``.

    Args:
        weights: Non-negative weights (normalized internally).
        offset: Single uniform offset in ``[0, 1)``.

    Returns:
        ``N`` ancestor indices in non-decreasing order.

    Raises:
        ValueError: If weights are invalid or the offset is outside ``[0, 1)``.

    Equation:
        [EQ-FILT-PF-003].

    References:
        [REF-KITAGAWA-1996], [REF-DOUCET-JOHANSEN-2011].

    """
    normalized = _normalized(weights)
    shift = require_finite_real(offset, name="offset")
    if not 0.0 <= shift < 1.0:
        raise ValueError("offset must lie in [0, 1)")
    count = len(normalized)
    indices: list[int] = []
    cumulative = normalized[0]
    source = 0
    for j in range(count):
        position = (shift + j) / count
        while position >= cumulative and source < count - 1:
            source += 1
            cumulative += normalized[source]
        indices.append(source)
    return tuple(indices)


def _weighted_moments(
    particles: Sequence[Vector], weights: Sequence[float]
) -> tuple[Vector, Matrix]:
    """Return the weighted mean and covariance of a particle cloud.

    Args:
        particles: Particle locations.
        weights: Normalized weights.

    Returns:
        Mean and covariance ``sum_i w_i (x_i - m)(x_i - m)'``.

    References:
        [REF-DOUCET-JOHANSEN-2011]. No named equation is needed.

    """
    size = len(particles[0])
    mean = tuple(
        math.fsum(w * x[j] for w, x in zip(weights, particles, strict=True))
        for j in range(size)
    )
    covariance = tuple(
        tuple(
            math.fsum(
                w * (x[i] - mean[i]) * (x[j] - mean[j])
                for w, x in zip(weights, particles, strict=True)
            )
            for j in range(size)
        )
        for i in range(size)
    )
    return mean, covariance


def bootstrap_particle_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    *,
    particles: int = 1000,
    resample_threshold: float = 0.5,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> ParticleFilterResult:
    """Run the bootstrap particle filter with ESS-triggered resampling.

    For each step: ``x_k^i = f(x_(k-1)^i) + w_k^i``;
    ``w_k^i ∝ w_(k-1)^i N(y_k; h(x_k^i), R)``; the likelihood increment is
    ``p_hat(y_k | y_1:k-1) = sum_i w_(k-1)^i N(y_k; h(x_k^i), R)`` with
    normalized previous weights; moments are recorded before resampling;
    systematic resampling runs when ``ESS < resample_threshold * N``.

    Args:
        model: Linear or nonlinear additive-Gaussian model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        particles: Number of particles ``N >= 2``.
        resample_threshold: ESS fraction in ``[0, 1]`` that triggers
            resampling; ``1`` resamples whenever weights are unequal and ``0``
            never resamples (pure sequential importance sampling).
        seed: Integer seed or stream (children ``"initial"``,
            ``"propagation"``, ``"resampling"``).
        controls: Optional controls (linear models only).

    Returns:
        Particle moments, ESS history, resampling flags, and log-likelihood.

    Raises:
        ValueError: If arguments are invalid or every weight underflows.

    Equation:
        [EQ-FILT-BAYES-001], [EQ-FILT-PF-001], [EQ-FILT-PF-002],
        [EQ-FILT-PF-003].

    References:
        [REF-GORDON-SALMOND-SMITH-1993], [REF-DOUCET-JOHANSEN-2011],
        [REF-SARKKA-2013, Chapter 7].

    """
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
        raise ValueError("particles must be an integer of at least two")
    threshold = require_finite_real(resample_threshold, name="resample_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("resample_threshold must lie in [0, 1]")
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    plan = _control_plan(controls, len(data))
    stream = resolve_stream(seed)
    initial_gauss = stream.child("initial").generator().gauss
    process_gauss = stream.child("propagation").generator().gauss
    resampling = stream.child("resampling").generator()
    size = model.state_dimension
    initial_factor = cholesky_semidefinite(model.initial_covariance)
    process_factor = cholesky_semidefinite(model.process_covariance)
    noise_factor = cholesky(model.observation_covariance)
    log_normalizer = (
        math.fsum(math.log(noise_factor[i][i]) for i in range(len(noise_factor)))
        + 0.5 * len(noise_factor) * _LOG_TWO_PI
    )

    def draw(mean: Vector, lower: Matrix, gauss_source: str) -> Vector:
        """Return ``mean + L z`` from the named noise generator.

        References:
            [EQ-FILT-SSM-002], [REF-DEVROYE-1986, Chapter XI].

        """
        gauss = initial_gauss if gauss_source == "initial" else process_gauss
        return _affine(lower, tuple(gauss() for _ in range(size)), mean)

    cloud: list[Vector] = [
        draw(model.initial_mean, initial_factor, "initial") for _ in range(particles)
    ]
    weights: tuple[float, ...] = tuple(1.0 / particles for _ in range(particles))
    mean, covariance = _weighted_moments(cloud, weights)
    means = [mean]
    covariances = [covariance]
    ess_history: list[float] = []
    resampled: list[bool] = []
    terms: list[float] = []
    for observation, control in zip(data, plan, strict=True):
        cloud = [
            draw(model._propagate(x, control), process_factor, "process") for x in cloud
        ]
        if observation is None:
            terms.append(0.0)
        else:
            y = _observation_vector(observation, model.observation_dimension)
            log_terms: list[float] = []
            for weight, x in zip(weights, cloud, strict=True):
                residual = tuple(
                    a - b for a, b in zip(y, model._measure(x), strict=True)
                )
                whitened = solve_lower(noise_factor, residual)
                log_g = -0.5 * math.fsum(v * v for v in whitened) - log_normalizer
                log_terms.append(
                    math.log(weight) + log_g if weight > 0.0 else -math.inf
                )
            peak = max(log_terms)
            if not math.isfinite(peak):
                raise ValueError("all particle weights underflowed to zero")
            increment = peak + math.log(
                math.fsum(math.exp(value - peak) for value in log_terms)
            )
            terms.append(increment)
            weights = tuple(math.exp(value - increment) for value in log_terms)
            total = math.fsum(weights)
            weights = tuple(value / total for value in weights)
        mean, covariance = _weighted_moments(cloud, weights)
        means.append(mean)
        covariances.append(covariance)
        ess = effective_sample_size(weights)
        ess_history.append(ess)
        if ess < threshold * particles:
            ancestors = systematic_resample(weights, resampling.random())
            cloud = [cloud[index] for index in ancestors]
            weights = tuple(1.0 / particles for _ in range(particles))
            resampled.append(True)
        else:
            resampled.append(False)
    return ParticleFilterResult(
        tuple(means),
        tuple(covariances),
        tuple(ess_history),
        tuple(resampled),
        tuple(terms),
        tuple(as_vector(x, name="particle") for x in cloud),
        weights,
        particles,
    )
