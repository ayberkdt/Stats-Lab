"""Particle filters with proposals that look at the current observation.

The bootstrap filter proposes from the transition prior and is blind to
``y_k`` until the weighting step. Two classical remedies are implemented:

* the *auxiliary particle filter* of Pitt and Shephard resamples the previous
  particles with first-stage weights ``W_(k-1)^i g(y_k | mu_k^i)``, where
  ``mu_k^i = f(x_(k-1)^i)`` is the predictive mean, and corrects with
  second-stage weights ``g(y_k | x_k^j) / g(y_k | mu_k^(a_j))``;
* the *locally optimal proposal* samples ``p(x_k | x_(k-1), y_k)`` exactly,
  which is Gaussian in closed form when the observation operator is linear
  and both noises are additive Gaussian; its weights ``p(y_k | x_(k-1))``
  do not depend on the new particle.

Both return the same :class:`ParticleFilterResult` as the bootstrap filter,
with an unbiased estimate of ``p(y_1:T)`` (its logarithm is biased downward)
and optional particle histories for :func:`ffbs_particle_smoother`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    identity,
    matmul,
    sandwich,
    transpose,
)

from .kalman import Observation, _observation_vector, gain_from_cross_covariance
from .particle import ParticleFilterResult, _weighted_moments
from .state_space import (
    LinearGaussianStateSpace,
    StateSpaceModel,
    _affine,
    _control_plan,
)

_LOG_TWO_PI = math.log(2.0 * math.pi)


def _require_particles(particles: object) -> int:
    """Validate a particle count.

    Args:
        particles: Candidate count.

    Returns:
        The count, at least two.

    Raises:
        ValueError: If the count is not an integer of at least two.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
        raise ValueError("particles must be an integer of at least two")
    return particles


def _gaussian_log_kernel(lower: Matrix) -> tuple[float, int]:
    """Return the log normalizer ``log((2 pi)^(m/2) |L|)`` and the dimension.

    Args:
        lower: Lower Cholesky factor of a positive-definite covariance.

    Returns:
        Log normalizing constant and dimension ``m``.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapter 1]. No named equation is needed.

    """
    size = len(lower)
    normalizer = math.fsum(math.log(lower[i][i]) for i in range(size))
    return normalizer + 0.5 * size * _LOG_TWO_PI, size


def _whitened_square(lower: Matrix, residual: Sequence[float]) -> float:
    """Return ``r' (L L')^-1 r`` by forward substitution in plain arithmetic.

    Used in per-particle inner loops; the factor was validated once.

    Args:
        lower: Lower Cholesky factor.
        residual: Residual vector.

    Returns:
        Squared Mahalanobis norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1]. No named equation is needed.

    """
    solved: list[float] = []
    total = 0.0
    for i, row in enumerate(lower):
        value = residual[i]
        for j in range(i):
            value -= row[j] * solved[j]
        value /= row[i]
        solved.append(value)
        total += value * value
    return total


def _log_normalize(log_terms: Sequence[float]) -> tuple[float, tuple[float, ...]]:
    """Return ``log sum_i exp(a_i)`` and the normalized weights ``exp(a_i - s)``.

    Args:
        log_terms: Log weights; ``-inf`` entries mean zero weight.

    Returns:
        Log normalizer and normalized weights.

    Raises:
        ValueError: If every weight underflows to zero.

    References:
        [REF-DOUCET-JOHANSEN-2011]. No named equation is needed.

    """
    peak = max(log_terms)
    if not math.isfinite(peak):
        raise ValueError("all particle weights underflowed to zero")
    scaled = [math.exp(value - peak) for value in log_terms]
    total = math.fsum(scaled)
    return peak + math.log(total), tuple(value / total for value in scaled)


def _systematic(weights: Sequence[float], offset: float) -> list[int]:
    """Return systematic-resampling ancestors of internally normalized weights.

    The same algorithm as :func:`probstat.filtering.systematic_resample`
    without re-validating weights produced by the filter itself.

    Args:
        weights: Normalized weights.
        offset: Uniform offset in ``[0, 1)``.

    Returns:
        ``N`` ancestor indices.

    Equation:
        [EQ-FILT-PF-003].

    References:
        [REF-KITAGAWA-1996].

    """
    count = len(weights)
    indices: list[int] = []
    cumulative = weights[0]
    source = 0
    for j in range(count):
        position = (offset + j) / count
        while position >= cumulative and source < count - 1:
            source += 1
            cumulative += weights[source]
        indices.append(source)
    return indices


def _ess(weights: Sequence[float]) -> float:
    """Return ``1 / sum_i w_i^2`` for internally normalized weights.

    Args:
        weights: Normalized weights.

    Returns:
        Effective sample size.

    Equation:
        [EQ-FILT-PF-002].

    References:
        [REF-DOUCET-JOHANSEN-2011].

    """
    return 1.0 / math.fsum(value * value for value in weights)


def _validated_run(
    particles: int,
    observations: Sequence[Observation],
    controls: Sequence[Sequence[float] | None] | None,
) -> tuple[int, tuple[Observation, ...], tuple[Sequence[float] | None, ...]]:
    """Validate the arguments shared by the particle filters.

    Args:
        particles: Particle count.
        observations: Observation sequence.
        controls: Optional controls.

    Returns:
        Checked count, observations, and one control entry per step.

    Raises:
        ValueError: If the count or the observations are invalid.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    count = _require_particles(particles)
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    return count, data, _control_plan(controls, len(data))


def auxiliary_particle_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    *,
    particles: int = 1000,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
    store_history: bool = False,
) -> ParticleFilterResult:
    """Run the auxiliary particle filter of Pitt and Shephard.

    At each observed step, with normalized previous weights ``W^i`` and
    predictive means ``mu^i = f(x_(k-1)^i)``:

    1. first-stage weights ``lambda^i ∝ W^i g(y_k | mu^i)`` with
       ``g(y | x) = N(y; h(x), R)``;
    2. systematic resampling of ancestors ``a_j ~ lambda``;
    3. propagation ``x_k^j = mu^(a_j) + w_k^j``, ``w_k^j ~ N(0, Q)``;
    4. second-stage weights ``w_k^j = g(y_k | x_k^j) / g(y_k | mu^(a_j))``;
    5. likelihood increment
       ``p_hat(y_k | y_1:k-1) = [sum_i W^i g(y_k | mu^i)] (1/N) sum_j w_k^j``.

    The product of the increments is an unbiased estimate of ``p(y_1:T)``.
    The first stage resamples at every observed step; at a missing
    observation the particles are only propagated and the weights kept.
    The point-estimate first stage has light tails: when ``g(y | mu)`` is far
    smaller than ``p(y | x_(k-1))`` for some particle, the second-stage
    weights become variable, which the ESS history records.

    Args:
        model: Linear or nonlinear additive-Gaussian model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        particles: Number of particles ``N >= 2``.
        seed: Integer seed or stream (children ``"initial"``,
            ``"propagation"``, ``"resampling"``).
        controls: Optional controls (linear models only).
        store_history: Keep every weighted cloud for particle smoothing.

    Returns:
        Particle moments, second-stage ESS, resampling flags, and the
        log-likelihood estimate.

    Raises:
        ValueError: If arguments are invalid or every weight underflows.

    Equation:
        [EQ-FILT-APF-001], [EQ-FILT-PF-002], [EQ-FILT-PF-003].

    References:
        [REF-PITT-SHEPHARD-1999], [REF-DOUCET-JOHANSEN-2011, Section 4.2].

    """
    count, data, plan = _validated_run(particles, observations, controls)
    stream = resolve_stream(seed)
    initial_gauss = stream.child("initial").generator().gauss
    process_gauss = stream.child("propagation").generator().gauss
    resampling = stream.child("resampling").generator()
    size = model.state_dimension
    initial_factor = cholesky_semidefinite(model.initial_covariance)
    process_factor = cholesky_semidefinite(model.process_covariance)
    noise_factor = cholesky(model.observation_covariance)
    log_normalizer, _ = _gaussian_log_kernel(noise_factor)
    log_count = math.log(count)

    def log_g(y: Vector, state: Vector) -> float:
        """Return ``log N(y; h(x), R)``.

        References:
            [EQ-FILT-SSM-002], [REF-SARKKA-2013, Chapter 7].

        """
        image = model._measure(state)
        residual = [a - b for a, b in zip(y, image, strict=True)]
        return -0.5 * _whitened_square(noise_factor, residual) - log_normalizer

    def draw(mean: Vector, lower: Matrix, gauss_source: str) -> Vector:
        """Return ``mean + L z`` from the named noise generator.

        References:
            [EQ-FILT-SSM-002], [REF-DEVROYE-1986, Chapter XI].

        """
        gauss = initial_gauss if gauss_source == "initial" else process_gauss
        return _affine(lower, tuple(gauss() for _ in range(size)), mean)

    cloud: list[Vector] = [
        draw(model.initial_mean, initial_factor, "initial") for _ in range(count)
    ]
    weights: tuple[float, ...] = tuple(1.0 / count for _ in range(count))
    mean, covariance = _weighted_moments(cloud, weights)
    means = [mean]
    covariances = [covariance]
    cloud_history: list[tuple[Vector, ...]] = [tuple(cloud)]
    weight_history: list[tuple[float, ...]] = [weights]
    ess_history: list[float] = []
    resampled: list[bool] = []
    terms: list[float] = []
    for observation, control in zip(data, plan, strict=True):
        centers = [model._propagate(x, control) for x in cloud]
        if observation is None:
            cloud = [draw(center, process_factor, "process") for center in centers]
            terms.append(0.0)
            resampled.append(False)
        else:
            y = _observation_vector(observation, model.observation_dimension)
            look_ahead = [log_g(y, center) for center in centers]
            first = [
                math.log(weight) + value if weight > 0.0 else -math.inf
                for weight, value in zip(weights, look_ahead, strict=True)
            ]
            first_log_sum, first_weights = _log_normalize(first)
            ancestors = _systematic(first_weights, resampling.random())
            cloud = [
                draw(centers[index], process_factor, "process") for index in ancestors
            ]
            second = [
                log_g(y, x) - look_ahead[index]
                for x, index in zip(cloud, ancestors, strict=True)
            ]
            second_log_sum, weights = _log_normalize(second)
            terms.append(first_log_sum + second_log_sum - log_count)
            resampled.append(True)
        mean, covariance = _weighted_moments(cloud, weights)
        means.append(mean)
        covariances.append(covariance)
        if store_history:
            cloud_history.append(tuple(cloud))
            weight_history.append(weights)
        ess_history.append(_ess(weights))
    return ParticleFilterResult(
        tuple(means),
        tuple(covariances),
        tuple(ess_history),
        tuple(resampled),
        tuple(terms),
        tuple(as_vector(x, name="particle") for x in cloud),
        weights,
        count,
        tuple(cloud_history) if store_history else None,
        tuple(weight_history) if store_history else None,
    )


def _linear_observation_operator(
    model: StateSpaceModel, observation_matrix: Sequence[Sequence[float]] | None
) -> Matrix:
    """Return the linear observation operator ``H`` of an additive model.

    For a :class:`LinearGaussianStateSpace` the model's own ``H`` is used.
    For a nonlinear model ``observation_matrix`` is required, and the claim
    ``h(x) = H x`` is checked at ``m_0`` and ``m_0 + e_j`` for every unit
    vector ``e_j`` (relative tolerance ``1e-9``).

    Args:
        model: Additive-Gaussian model.
        observation_matrix: ``H`` for a nonlinear model (or ``None``).

    Returns:
        Validated ``m`` by ``n`` observation matrix.

    Raises:
        ValueError: If a nonlinear model has no ``observation_matrix``, the
            matrix is misshapen, or ``h`` is not the linear map ``H x``.

    References:
        [REF-DOUCET-GODSILL-ANDRIEU-2000]. No equation is evaluated.

    """
    if isinstance(model, LinearGaussianStateSpace):
        if observation_matrix is None:
            return model.observation_matrix
    elif observation_matrix is None:
        raise ValueError(
            "the locally optimal proposal is closed-form only for a linear "
            "observation operator; pass observation_matrix=H for a nonlinear "
            "model whose observation_function is h(x) = H x"
        )
    matrix = as_matrix(observation_matrix, name="observation_matrix")
    if len(matrix) != model.observation_dimension or len(matrix[0]) != (
        model.state_dimension
    ):
        raise ValueError(
            "observation_matrix must be observation_dimension by state_dimension"
        )
    base = model.initial_mean
    points = [base] + [
        tuple(value + (1.0 if i == j else 0.0) for i, value in enumerate(base))
        for j in range(model.state_dimension)
    ]
    for point in points:
        image = model._measure(point)
        linear = _affine(matrix, point)
        for a, b in zip(image, linear, strict=True):
            if abs(a - b) > 1e-9 * max(1.0, abs(b)):
                raise ValueError(
                    "observation_function is not the linear map "
                    "observation_matrix @ x; the locally optimal proposal "
                    "needs a linear observation operator"
                )
    return matrix


def optimal_proposal_particle_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    *,
    particles: int = 1000,
    resample_threshold: float = 0.5,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
    observation_matrix: Sequence[Sequence[float]] | None = None,
    store_history: bool = False,
) -> ParticleFilterResult:
    """Run a particle filter with the locally optimal proposal.

    For ``x_k = f(x_(k-1)) + w_k`` and ``y_k = H x_k + v_k`` with Gaussian
    noises, ``p(x_k | x_(k-1), y_k) = N(mu + K (y_k - H mu), Sigma)`` with
    ``mu = f(x_(k-1))``, ``S = H Q H' + R``, ``K = Q H' S^-1``, and
    ``Sigma = (I - K H) Q (I - K H)' + K R K'`` (Joseph form). The
    incremental weight is the predictive density ``p(y_k | x_(k-1)) =
    N(y_k; H mu, S)``, independent of the new particle, so the weight
    variance conditional on the past is zero (the minimum over proposals).
    The likelihood increment ``sum_i W_(k-1)^i N(y_k; H mu^i, S)`` uses the
    normalized previous weights, and systematic resampling runs after
    weighting when ``ESS < resample_threshold * N``, as in the bootstrap
    filter.

    Args:
        model: Linear-Gaussian model, or a nonlinear model whose observation
            map is linear (then ``observation_matrix`` is required).
        observations: Observations ``y_1..y_T`` (``None`` if missing; the
            particles then move with the transition prior).
        particles: Number of particles ``N >= 2``.
        resample_threshold: ESS fraction in ``[0, 1]`` that triggers
            resampling.
        seed: Integer seed or stream (children ``"initial"``,
            ``"propagation"``, ``"resampling"``).
        controls: Optional controls (linear models only).
        observation_matrix: ``H`` of a nonlinear model with a linear
            observation map; checked against ``h`` before filtering.
        store_history: Keep every weighted cloud for particle smoothing.

    Returns:
        Particle moments, ESS history, resampling flags, and the
        log-likelihood estimate.

    Raises:
        ValueError: If arguments are invalid, the observation operator is
            not linear, or every weight underflows.

    Equation:
        [EQ-FILT-OPT-001], [EQ-FILT-PF-002], [EQ-FILT-PF-003].

    References:
        [REF-DOUCET-GODSILL-ANDRIEU-2000],
        [REF-DOUCET-JOHANSEN-2011, Section 4.1].

    """
    count, data, plan = _validated_run(particles, observations, controls)
    threshold = require_finite_real(resample_threshold, name="resample_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("resample_threshold must lie in [0, 1]")
    operator = _linear_observation_operator(model, observation_matrix)
    process = model.process_covariance
    noise = model.observation_covariance
    innovation_covariance = add(sandwich(operator, process), noise)
    innovation_factor = cholesky(innovation_covariance)
    gain = gain_from_cross_covariance(
        matmul(process, transpose(operator)), innovation_covariance
    )
    size = model.state_dimension
    residual_map = add(identity(size), matmul(gain, operator), scale=-1.0)
    proposal_covariance = add(sandwich(residual_map, process), sandwich(gain, noise))
    proposal_factor = cholesky_semidefinite(proposal_covariance)
    process_factor = cholesky_semidefinite(process)
    initial_factor = cholesky_semidefinite(model.initial_covariance)
    log_normalizer, _ = _gaussian_log_kernel(innovation_factor)
    stream = resolve_stream(seed)
    initial_gauss = stream.child("initial").generator().gauss
    process_gauss = stream.child("propagation").generator().gauss
    resampling = stream.child("resampling").generator()

    def draw(mean: Vector, lower: Matrix, gauss_source: str) -> Vector:
        """Return ``mean + L z`` from the named noise generator.

        References:
            [EQ-FILT-SSM-002], [REF-DEVROYE-1986, Chapter XI].

        """
        gauss = initial_gauss if gauss_source == "initial" else process_gauss
        return _affine(lower, tuple(gauss() for _ in range(size)), mean)

    cloud: list[Vector] = [
        draw(model.initial_mean, initial_factor, "initial") for _ in range(count)
    ]
    weights: tuple[float, ...] = tuple(1.0 / count for _ in range(count))
    mean, covariance = _weighted_moments(cloud, weights)
    means = [mean]
    covariances = [covariance]
    cloud_history: list[tuple[Vector, ...]] = [tuple(cloud)]
    weight_history: list[tuple[float, ...]] = [weights]
    ess_history: list[float] = []
    resampled: list[bool] = []
    terms: list[float] = []
    for observation, control in zip(data, plan, strict=True):
        centers = [model._propagate(x, control) for x in cloud]
        if observation is None:
            cloud = [draw(center, process_factor, "process") for center in centers]
            terms.append(0.0)
        else:
            y = _observation_vector(observation, model.observation_dimension)
            innovations = [
                tuple(a - b for a, b in zip(y, _affine(operator, center), strict=True))
                for center in centers
            ]
            log_terms = [
                math.log(weight)
                - 0.5 * _whitened_square(innovation_factor, innovation)
                - log_normalizer
                if weight > 0.0
                else -math.inf
                for weight, innovation in zip(weights, innovations, strict=True)
            ]
            increment, weights = _log_normalize(log_terms)
            terms.append(increment)
            cloud = [
                draw(_affine(gain, innovation, center), proposal_factor, "process")
                for center, innovation in zip(centers, innovations, strict=True)
            ]
        mean, covariance = _weighted_moments(cloud, weights)
        means.append(mean)
        covariances.append(covariance)
        if store_history:
            cloud_history.append(tuple(cloud))
            weight_history.append(weights)
        ess = _ess(weights)
        ess_history.append(ess)
        if ess < threshold * count:
            ancestors = _systematic(weights, resampling.random())
            cloud = [cloud[index] for index in ancestors]
            weights = tuple(1.0 / count for _ in range(count))
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
        count,
        tuple(cloud_history) if store_history else None,
        tuple(weight_history) if store_history else None,
    )
