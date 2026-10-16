"""Shapley effects for global sensitivity with possibly dependent inputs.

The Shapley effect of input ``i`` shares the output variance among inputs by
the Shapley value of cooperative game theory::

    Sh_i = (1 / Var Y) sum_{J subset of D minus i} w_|J| [c(J + i) - c(J)],
    w_k = k! (d - k - 1)! / d!

Following Song, Nelson, and Staum the cost is ``c(J) = E[Var(Y | X_-J)]``,
which gives the same Shapley values as the closed index
``Var(E[Y | X_J])`` and is estimated by an outer loop over ``X_-J`` and a
short inner loop over ``X_J | X_-J``. Effects are non-negative in the
population, sum to one, and remain meaningful under dependence, unlike Sobol
indices. Conditional sampling uses the model's Gaussian copula: in latent
normal space ``z_J | z_-J`` is Gaussian with Schur-complement covariance.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    identity,
    matvec,
    solve_lower,
    solve_spd,
    symmetrize,
)
from probstat.modeling import ProbabilisticModel

ShapleyMethod = Literal["auto", "exact", "random"]
_EXACT_LIMIT = 6


def _subset_weight(size: int, dimension: int) -> float:
    """Return the Shapley weight ``k! (d - k - 1)! / d!`` for ``|J| = k``.

    References:
        [EQ-SENS-SHAPLEY-001], [REF-OWEN-2014].

    """
    return (
        math.factorial(size)
        * math.factorial(dimension - size - 1)
        / math.factorial(dimension)
    )


def exact_shapley_values(
    dimension: int, value: Callable[[frozenset[int]], float]
) -> tuple[float, ...]:
    """Return exact Shapley values of a set function on ``{0, ..., d - 1}``.

    ``Sh_i = sum_{J not containing i} w_|J| [v(J + i) - v(J)]`` with
    ``w_k = k! (d - k - 1)! / d!``; ``v`` is called once per subset.

    Args:
        dimension: Number of players ``d`` (1 to 20).
        value: Set function ``v``; ``v(frozenset())`` is the empty-coalition
            value.

    Returns:
        Shapley values, summing to ``v(D) - v(empty)``.

    Raises:
        ValueError: If ``dimension`` is outside ``1..20``.

    Equation:
        [EQ-SENS-SHAPLEY-001].

    References:
        [REF-OWEN-2014], [REF-SONG-NELSON-STAUM-2016].

    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValueError("dimension must be an integer")
    if not 1 <= dimension <= 20:
        raise ValueError("dimension must lie between 1 and 20")
    players = range(dimension)
    cache: dict[frozenset[int], float] = {}
    for size in range(dimension + 1):
        for subset in itertools.combinations(players, size):
            key = frozenset(subset)
            cache[key] = require_finite_real(value(key), name="set-function value")
    effects = []
    for i in players:
        others = [j for j in players if j != i]
        total = math.fsum(
            _subset_weight(size, dimension)
            * (cache[frozenset(subset) | {i}] - cache[frozenset(subset)])
            for size in range(dimension)
            for subset in itertools.combinations(others, size)
        )
        effects.append(total)
    return tuple(effects)


def linear_gaussian_shapley(
    coefficients: Sequence[float], covariance: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """Return exact Shapley effects of ``Y = b'X`` for Gaussian ``X``.

    For ``X ~ N(mu, Sigma)`` the closed index is
    ``Var(E[Y | X_J]) = s_J' Sigma_JJ^-1 s_J`` with ``s = Sigma b``, and
    ``Var Y = b' Sigma b``; the Shapley values of this set function are
    normalized by ``Var Y``. In two dimensions this reduces to
    ``Sh_1 = (b1^2 s1^2 (1 - rho^2/2) + rho b1 b2 s1 s2 + b2^2 s2^2 rho^2/2) / V``.

    Args:
        coefficients: Linear coefficients ``b``.
        covariance: Symmetric positive-definite covariance ``Sigma``.

    Returns:
        Shapley effects summing to one.

    Raises:
        ValueError: If dimensions disagree, the covariance is not positive
            definite, or ``Var Y = 0``.

    Equation:
        [EQ-SENS-SHAPLEY-001], [EQ-SENS-SHAPLEY-004].

    References:
        [REF-OWEN-2014], [REF-IOOSS-PRIEUR-2019].

    """
    b = as_vector(coefficients, name="coefficients")
    sigma = symmetrize(covariance, name="covariance")
    dimension = len(b)
    if len(sigma) != dimension:
        raise ValueError("covariance dimension must equal the number of coefficients")
    cholesky(sigma)
    s = matvec(sigma, b)
    variance = math.fsum(bi * si for bi, si in zip(b, s, strict=True))
    if variance <= 0.0:
        raise ValueError("Var(Y) must be positive")

    def closed(subset: frozenset[int]) -> float:
        """Return ``Var(E[Y | X_J])`` for the coalition ``J``.

        References:
            [EQ-SENS-SHAPLEY-004], [REF-RASMUSSEN-WILLIAMS-2006, Appendix A.2].

        """
        if not subset:
            return 0.0
        index = sorted(subset)
        block = tuple(tuple(sigma[i][j] for j in index) for i in index)
        part = tuple(s[i] for i in index)
        solved = solve_spd(block, part)
        return math.fsum(p * q for p, q in zip(part, solved, strict=True))

    raw = exact_shapley_values(dimension, closed)
    return tuple(value / variance for value in raw)


@dataclass(frozen=True, slots=True)
class ShapleyResult:
    """Shapley effects of one QoI of a probabilistic model.

    Args:
        quantity: QoI name.
        input_names: Input names in model order.
        effects: Shapley effects (normalized by the output variance; they sum
            to one by construction).
        standard_errors: Approximate standard errors of the effects (the
            uncertainty of the normalizing variance is ignored).
        variance: Estimated output variance ``Var Y``.
        method: ``"exact"`` (all subsets) or ``"random"`` (random
            permutations).
        permutations: Number of random permutations (``d!`` equivalents for
            ``"exact"`` are not enumerated; 0 is reported).
        outer_samples: Outer samples ``N_O`` per cost estimate.
        inner_samples: Inner samples ``N_I`` per outer sample.
        variance_samples: Samples ``N_V`` used for ``Var Y``.
        evaluations: Forward-model evaluations.

    References:
        [REF-SONG-NELSON-STAUM-2016], [REF-IOOSS-PRIEUR-2019].

    """

    quantity: str
    input_names: tuple[str, ...]
    effects: Vector
    standard_errors: Vector
    variance: float
    method: str
    permutations: int
    outer_samples: int
    inner_samples: int
    variance_samples: int
    evaluations: int

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> Shapley effect``.

        References:
            [EQ-SENS-SHAPLEY-001], [REF-OWEN-2014].

        """
        return MappingProxyType(dict(zip(self.input_names, self.effects, strict=True)))

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending Shapley-effect order.

        References:
            [REF-SONG-NELSON-STAUM-2016].

        """
        order = sorted(range(len(self.effects)), key=lambda i: -self.effects[i])
        return tuple(self.input_names[i] for i in order)


class _ConditionalSampler:
    """Latent Gaussian-copula sampler of ``X_J`` given ``X_-J``.

    References:
        [EQ-SENS-SHAPLEY-003], [REF-RASMUSSEN-WILLIAMS-2006, Appendix A.2].

    """

    __slots__ = ("free", "fixed", "gain", "lower")

    def __init__(self, correlation: Matrix, free: Sequence[int]) -> None:
        """Precompute ``A = R_JK R_KK^-1`` and ``chol(R_JJ - A R_KJ)``.

        Args:
            correlation: Latent correlation ``R``.
            free: Indices ``J`` that are resampled; ``K`` is the complement.

        References:
            [EQ-SENS-SHAPLEY-003], [EQ-LINALG-001].

        """
        dimension = len(correlation)
        self.free = tuple(free)
        self.fixed = tuple(i for i in range(dimension) if i not in set(free))
        block_kk = tuple(
            tuple(correlation[k][m] for m in self.fixed) for k in self.fixed
        )
        gain: list[Vector] = []
        for j in self.free:
            cross = tuple(correlation[k][j] for k in self.fixed)
            gain.append(solve_spd(block_kk, cross) if self.fixed else ())
        self.gain = tuple(gain)
        conditional = tuple(
            tuple(
                correlation[i][j]
                - math.fsum(
                    a * correlation[k][j]
                    for a, k in zip(self.gain[p], self.fixed, strict=True)
                )
                for j in self.free
            )
            for p, i in enumerate(self.free)
        )
        try:
            self.lower = cholesky(conditional)
        except ValueError:
            self.lower = cholesky_semidefinite(conditional)

    def draw(self, latent: Vector, gauss: Callable[[], float]) -> Vector:
        """Return a copy of ``latent`` with ``z_J`` redrawn given ``z_K``.

        References:
            [EQ-SENS-SHAPLEY-003], [EQ-DIST-MVN-005].

        """
        noise = [gauss() for _ in self.free]
        values = list(latent)
        for p, j in enumerate(self.free):
            mean = math.fsum(
                a * latent[k] for a, k in zip(self.gain[p], self.fixed, strict=True)
            )
            values[j] = mean + math.fsum(
                self.lower[p][q] * noise[q] for q in range(p + 1)
            )
        return tuple(values)


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def shapley_effects(
    model: ProbabilisticModel,
    quantity: str,
    *,
    method: ShapleyMethod = "auto",
    permutations: int = 200,
    outer_samples: int | None = None,
    inner_samples: int = 3,
    variance_samples: int = 1000,
    seed: int | RandomStream | None = None,
) -> ShapleyResult:
    """Estimate Shapley effects of a model QoI under the model's joint law.

    ``Var Y`` is estimated from ``N_V`` joint draws. Each cost
    ``c(J) = E[Var(Y | X_-J)]`` is the mean, over ``N_O`` outer draws of
    ``X_-J``, of the unbiased sample variance of ``N_I`` outputs with ``X_J``
    redrawn from its conditional law. With ``method="exact"`` every proper
    non-empty subset is estimated once and combined by the subset formula;
    with ``"random"`` each of ``m`` random permutations estimates the costs of
    its ``d - 1`` leading coalitions afresh (Song-Nelson-Staum Algorithm with
    ``c(empty) = 0`` and ``c(D) = Var Y``). Either way the effects telescope to
    exactly one. ``"auto"`` chooses exact enumeration for ``d <= 6``.

    Conditional laws come from the Gaussian copula: with latent ``z ~ N(0,
    R)``, ``z_J | z_K ~ N(R_JK R_KK^-1 z_K, R_JJ - R_JK R_KK^-1 R_KJ)`` and
    ``x_i = F_i^-1(Phi(z_i))``.

    Args:
        model: Probabilistic model (independent or Gaussian-copula inputs).
        quantity: QoI name.
        method: ``"auto"``, ``"exact"``, or ``"random"``.
        permutations: Random permutations ``m`` for ``"random"``.
        outer_samples: ``N_O`` per cost; default 200 for ``"exact"`` and 1 for
            ``"random"`` (Song-Nelson-Staum's recommendation).
        inner_samples: ``N_I`` per outer sample (at least two; 3 recommended).
        variance_samples: ``N_V`` draws for ``Var Y`` (at least two).
        seed: Root seed or stream.

    Returns:
        Effects, approximate standard errors, variance, and evaluation count.
        Cost is ``N_V + (2^d - 2) N_O N_I`` (exact) or
        ``N_V + m (d - 1) N_O N_I`` (random).

    Raises:
        ValueError: If options are invalid, ``"exact"`` is requested for
            ``d > 20``, the estimated output variance is not positive, or the
            joint law is neither independent nor a Gaussian copula of
            continuous inputs (the conditional sampler is Gaussian).

    Equation:
        [EQ-SENS-SHAPLEY-001], [EQ-SENS-SHAPLEY-002], [EQ-SENS-SHAPLEY-003].

    References:
        [REF-SONG-NELSON-STAUM-2016], [REF-OWEN-2014], [REF-IOOSS-PRIEUR-2019].

    """
    if model.inputs.uses_rosenblatt or (
        model.inputs.has_discrete and not model.inputs.is_independent
    ):
        raise ValueError(
            "Shapley effects need independent inputs or a Gaussian copula of "
            "continuous inputs; the conditional sampler is Gaussian"
        )
    model.quantity(quantity)
    dimension = model.dimension
    if method not in ("auto", "exact", "random"):
        raise ValueError("method must be 'auto', 'exact', or 'random'")
    chosen = method
    if method == "auto":
        chosen = "exact" if dimension <= _EXACT_LIMIT else "random"
    if chosen == "exact" and dimension > 20:
        raise ValueError("exact Shapley enumeration supports at most 20 inputs")
    inner = _require_count(inner_samples, "inner_samples", 2)
    n_variance = _require_count(variance_samples, "variance_samples", 2)
    default_outer = 200 if chosen == "exact" else 1
    outer = _require_count(
        default_outer if outer_samples is None else outer_samples, "outer_samples", 1
    )
    count_permutations = (
        _require_count(permutations, "permutations", 2) if chosen == "random" else 0
    )
    joint = model.inputs
    correlation = (
        joint.correlation if joint.correlation is not None else identity(dimension)
    )
    lower = cholesky(correlation)
    root = resolve_stream(seed)
    function = model.scalar_function(quantity)
    before = model.evaluations

    def evaluate_latent(latent: Vector) -> float:
        """Evaluate the QoI at latent normals ``z``.

        References:
            [EQ-MOD-NATAF-001], [REF-LIU-DERKIUREGHIAN-1986].

        """
        u = latent if joint.is_independent else solve_lower(lower, latent)
        return function(joint.from_standard_normal(u))

    variance_generator = root.child("shapley-variance").generator()
    outputs = []
    for _ in range(n_variance):
        u = tuple(variance_generator.gauss(0.0, 1.0) for _ in range(dimension))
        outputs.append(evaluate_latent(matvec(lower, u)))
    mean_y = math.fsum(outputs) / n_variance
    centered = [(y - mean_y) ** 2 for y in outputs]
    variance = math.fsum(centered) / (n_variance - 1)
    if not variance > 0.0:
        raise ValueError("Shapley effects require positive output variance")
    fourth = math.fsum(c * c for c in centered) / n_variance
    variance_of_variance = max(0.0, fourth - variance * variance) / n_variance

    samplers: dict[frozenset[int], _ConditionalSampler] = {}
    cost_generator = root.child("shapley-costs").generator()

    def gauss() -> float:
        """Return one standard normal variate for the cost loops.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998].

        """
        return cost_generator.gauss(0.0, 1.0)

    def estimate_cost(subset: frozenset[int]) -> tuple[float, float]:
        """Return ``(c_hat(J), Var c_hat(J))`` for ``c(J) = E[Var(Y | X_-J)]``.

        References:
            [EQ-SENS-SHAPLEY-002], [REF-SONG-NELSON-STAUM-2016].

        """
        sampler = samplers.get(subset)
        if sampler is None:
            sampler = _ConditionalSampler(correlation, sorted(subset))
            samplers[subset] = sampler
        variances = []
        for _ in range(outer):
            base = matvec(lower, tuple(gauss() for _ in range(dimension)))
            values = [evaluate_latent(sampler.draw(base, gauss)) for _ in range(inner)]
            center = math.fsum(values) / inner
            variances.append(math.fsum((v - center) ** 2 for v in values) / (inner - 1))
        estimate = math.fsum(variances) / outer
        if outer < 2:
            return estimate, math.nan
        spread = math.fsum((v - estimate) ** 2 for v in variances) / (outer - 1)
        return estimate, spread / outer

    effects: list[float]
    errors: list[float]
    if chosen == "exact":
        costs: dict[frozenset[int], float] = {frozenset(): 0.0}
        cost_variances: dict[frozenset[int], float] = {frozenset(): 0.0}
        full = frozenset(range(dimension))
        costs[full] = variance
        cost_variances[full] = variance_of_variance
        for size in range(1, dimension):
            for subset in itertools.combinations(range(dimension), size):
                key = frozenset(subset)
                costs[key], cost_variances[key] = estimate_cost(key)
        raw = exact_shapley_values(dimension, costs.__getitem__)
        effects = [value / variance for value in raw]
        errors = []
        for i in range(dimension):
            accumulated = 0.0
            for key, spread in cost_variances.items():
                if not key:
                    continue
                if i in key:
                    weight = _subset_weight(len(key) - 1, dimension)
                else:
                    weight = _subset_weight(len(key), dimension)
                accumulated += weight * weight * spread
            errors.append(math.sqrt(accumulated) / variance)
    else:
        permutation_generator = root.child("shapley-permutations").generator()
        contributions: list[list[float]] = [[] for _ in range(dimension)]
        for _ in range(count_permutations):
            order = list(range(dimension))
            permutation_generator.shuffle(order)
            previous = 0.0
            for position, player in enumerate(order):
                if position == dimension - 1:
                    current = variance
                else:
                    current, _ = estimate_cost(frozenset(order[: position + 1]))
                contributions[player].append(current - previous)
                previous = current
        effects = [math.fsum(c) / count_permutations / variance for c in contributions]
        errors = []
        for column in contributions:
            center = math.fsum(column) / len(column)
            spread = math.fsum((c - center) ** 2 for c in column) / (len(column) - 1)
            errors.append(math.sqrt(spread / len(column)) / variance)
    return ShapleyResult(
        quantity=quantity,
        input_names=joint.names,
        effects=tuple(effects),
        standard_errors=tuple(errors),
        variance=variance,
        method=chosen,
        permutations=count_permutations,
        outer_samples=outer,
        inner_samples=inner,
        variance_samples=n_variance,
        evaluations=model.evaluations - before,
    )
