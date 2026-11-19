"""Forward-filtering backward-smoothing (FFBS) particle smoother.

A stored particle-filter run gives weighted clouds ``{x_k^i, W_k^i}`` that
approximate ``p(x_k | y_1:k)``. Godsill, Doucet, and West draw whole
trajectories from the smoothing law ``p(x_0:T | y_1:T)`` backwards: draw
``x~_T`` from the last cloud, then for ``k = T-1, ..., 0`` draw ``x~_k = x_k^i``
with backward weights ``W_k^i f(x~_(k+1) | x_k^i)``, where ``f`` is the
transition density. Each trajectory costs ``O(N T)``, so ``M`` trajectories
cost ``O(M N T)``; unlike the genealogy of a resampling filter, the drawn
trajectories do not collapse onto a few ancestors at early times.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Matrix, Vector, cholesky

from .particle import ParticleFilterResult, _weighted_moments
from .particle_proposals import _whitened_square
from .state_space import StateSpaceModel, _control_plan


@dataclass(frozen=True, slots=True)
class ParticleSmootherResult:
    """Trajectories drawn from the particle approximation of ``p(x_0:T | y_1:T)``.

    Args:
        trajectories: ``M`` sampled trajectories, each ``x~_0, ..., x~_T``.
        means: Average of the trajectories at ``k = 0..T``.
        covariances: Plug-in covariance ``M^-1 sum_m (x~ - mean)(x~ - mean)'``
            of the trajectories at ``k = 0..T``.
        trajectory_count: Number of trajectories ``M``.

    References:
        [REF-GODSILL-DOUCET-WEST-2004], [REF-DOUCET-JOHANSEN-2011,
        Section 5.2].

    """

    trajectories: tuple[tuple[Vector, ...], ...]
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    trajectory_count: int

    @property
    def variances(self) -> tuple[Vector, ...]:
        """Return the marginal smoothed variances (covariance diagonals).

        References:
            [REF-GODSILL-DOUCET-WEST-2004]. No equation is evaluated.

        """
        return tuple(
            tuple(matrix[i][i] for i in range(len(matrix)))
            for matrix in self.covariances
        )


def _categorical(weights: Sequence[float], uniform: float) -> int:
    """Return the index selected by inverse-CDF lookup of one uniform.

    Args:
        weights: Normalized weights.
        uniform: Uniform draw in ``[0, 1)``.

    Returns:
        Index ``i`` with ``sum_(l<i) W^l <= u < sum_(l<=i) W^l``.

    References:
        [REF-DEVROYE-1986, Chapter III]. No named equation is needed.

    """
    cumulative = 0.0
    last = len(weights) - 1
    for index, weight in enumerate(weights):
        cumulative += weight
        if uniform < cumulative:
            return index
    # Rounding can leave the cumulative sum just below one; take the last
    # index that carries weight.
    while last > 0 and weights[last] <= 0.0:
        last -= 1
    return last


def ffbs_particle_smoother(
    model: StateSpaceModel,
    result: ParticleFilterResult,
    *,
    trajectories: int = 100,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> ParticleSmootherResult:
    """Draw smoothed trajectories by forward filtering, backward sampling.

    With the stored clouds ``{x_k^i, W_k^i}`` of a particle filter run, each
    trajectory starts from ``x~_T ~ sum_i W_T^i delta(x_T^i)`` and then, for
    ``k = T-1, ..., 0``, draws ``x~_k = x_k^i`` with probability
    ``W_(k|k+1)^i ∝ W_k^i N(x~_(k+1); f(x_k^i) + B u_(k+1), Q)``.

    Args:
        model: The model the filter ran on (same ``f``, ``Q``, controls).
        result: Particle-filter result produced with ``store_history=True``.
        trajectories: Number of trajectories ``M >= 2``.
        seed: Integer seed or stream (child ``"backward"``).
        controls: The controls the filter used (linear models only).

    Returns:
        Trajectories with their per-time means and covariances.

    Raises:
        ValueError: If the result has no particle history, its length does
            not match the controls, ``trajectories < 2``, ``Q`` is not
            positive definite (the transition density must exist), or every
            backward weight underflows.

    Equation:
        [EQ-FILT-FFBS-001].

    References:
        [REF-GODSILL-DOUCET-WEST-2004], [REF-DOUCET-JOHANSEN-2011,
        Section 5.2].

    """
    if result.particle_history is None or result.weight_history is None:
        raise ValueError(
            "the particle filter result has no particle history; rerun the "
            "filter with store_history=True"
        )
    if (
        isinstance(trajectories, bool)
        or not isinstance(trajectories, int)
        or trajectories < 2
    ):
        raise ValueError("trajectories must be an integer of at least two")
    clouds = result.particle_history
    cloud_weights = result.weight_history
    steps = len(clouds) - 1
    if steps < 1:
        raise ValueError("the particle history must cover at least one step")
    plan = _control_plan(controls, steps)
    try:
        process_factor = cholesky(model.process_covariance)
    except ValueError as error:
        raise ValueError(
            "forward-filtering backward-smoothing needs a positive-definite "
            "process_covariance (the transition density must exist)"
        ) from error
    if len(clouds[0][0]) != model.state_dimension:
        raise ValueError("the particle history does not match the model dimension")
    centers = [[model._propagate(x, plan[k]) for x in clouds[k]] for k in range(steps)]
    log_weights = [
        [math.log(w) if w > 0.0 else -math.inf for w in weights]
        for weights in cloud_weights
    ]
    generator = resolve_stream(seed).child("backward").generator()
    paths: list[tuple[Vector, ...]] = []
    for _ in range(trajectories):
        index = _categorical(cloud_weights[steps], generator.random())
        state = clouds[steps][index]
        path = [state]
        for k in range(steps - 1, -1, -1):
            backward = [
                log_w
                - 0.5
                * _whitened_square(
                    process_factor,
                    [a - b for a, b in zip(state, center, strict=True)],
                )
                if log_w > -math.inf
                else -math.inf
                for log_w, center in zip(log_weights[k], centers[k], strict=True)
            ]
            peak = max(backward)
            if not math.isfinite(peak):
                raise ValueError("all backward weights underflowed to zero")
            scaled = [math.exp(value - peak) for value in backward]
            total = math.fsum(scaled)
            index = _categorical(
                [value / total for value in scaled], generator.random()
            )
            state = clouds[k][index]
            path.append(state)
        path.reverse()
        paths.append(tuple(path))
    equal = tuple(1.0 / trajectories for _ in range(trajectories))
    means: list[Vector] = []
    covariances: list[Matrix] = []
    for k in range(steps + 1):
        mean, covariance = _weighted_moments([path[k] for path in paths], equal)
        means.append(mean)
        covariances.append(covariance)
    return ParticleSmootherResult(
        tuple(paths), tuple(means), tuple(covariances), trajectories
    )
