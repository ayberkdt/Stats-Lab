"""AK-MCS: active-learning Kriging (GP) surrogate on a Monte Carlo population.

Echard, Gayton, and Lemaire's AK-MCS replaces most limit-state calls of a
crude Monte Carlo estimate by a Gaussian-process surrogate that is refined
only where the sign of ``G`` is uncertain:

1. draw a Monte Carlo population ``u_1..u_N ~ N(0, I)`` in standard space;
2. fit a GP to ``G`` on a small space-filling design (LHS, Sobol, ...);
3. evaluate the learning function ``U(u) = |mu(u)| / sigma(u)`` on the
   population and add the point with the smallest ``U`` to the training set
   (one true-model call);
4. stop learning when ``min U >= 2`` (every population point has its sign
   predicted with probability at least ``Phi(2) ~ 0.977``);
5. estimate ``P_f = N_f / N`` from the predicted signs; if its c.o.v.
   ``sqrt((1 - P_f) / (N P_f))`` is above the target, enlarge the population
   and resume learning.

The GP lives in standard-normal space. This module depends on
:mod:`probstat.surrogates` (same architecture layer); ``surrogates`` does
not import ``reliability``, so the dependency stays acyclic.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.modeling import standard_normal_quantile
from probstat.propagation import make_design
from probstat.surrogates import KERNELS, GaussianProcessRegression
from probstat.surrogates import fit_gaussian_process as _fit_gp

from .model_based import ModelLimitState

_SQRT5 = math.sqrt(5.0)
_DESIGNS = ("mc", "lhs", "halton", "sobol")


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_positive(value: object, name: str) -> float:
    """Validate a strictly positive finite real option.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


class _IncrementalPosterior:
    """GP posterior over a fixed population with rank-one training updates.

    With fixed hyperparameters (prior mean ``m``, kernel ``k``, noise
    ``s_n^2``) and ``K + s_n^2 I = L L'``, each population point keeps
    ``v(x) = L^-1 k_T(x)``, so ``mu(x) = m + v(x) . z`` with
    ``z = L^-1 (y - m)`` and ``sigma^2(x) = k(x, x) - ||v(x)||^2``. Adding a
    training point ``t`` appends one row to ``L`` and one entry to every
    ``v(x)`` and ``z``, at ``O(n)`` cost per population point instead of the
    ``O(n^2)`` of a fresh prediction. The result equals the batch posterior
    at the same hyperparameters up to rounding.

    References:
        [EQ-SUR-GP-003], [EQ-REL-AKMCS-001],
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """

    def __init__(
        self,
        surrogate: GaussianProcessRegression,
        population: Sequence[Vector],
    ) -> None:
        """Build the posterior of a fitted GP over the population.

        References:
            [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        parameters = surrogate.hyperparameters
        self.kernel = surrogate.kernel
        self.signal = parameters.signal_variance
        self.noise = parameters.noise_variance
        self.inverse_scales = tuple(1.0 / ell for ell in parameters.length_scales)
        self.mean = surrogate.output_mean
        self.training: list[Vector] = []
        self.lower: list[list[float]] = []
        self.z: list[float] = []
        self.v: list[list[float]] = []
        self.mu: list[float] = []
        self.var: list[float] = []
        self.population: list[Vector] = []
        for point, value in zip(surrogate.points, surrogate.values, strict=True):
            self._extend_training(point, value)
        self.add_population(population)

    def scaled(self, point: Vector) -> Vector:
        """Return ``(x_i / ell_i)``, the coordinates in which ``r`` is Euclidean.

        References:
            [EQ-SUR-GP-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return tuple(
            a * inverse for a, inverse in zip(point, self.inverse_scales, strict=True)
        )

    def covariance(self, left: Vector, right: Vector) -> float:
        """Return ``k`` between two points given in scaled coordinates.

        References:
            [EQ-SUR-GP-001], [EQ-SUR-GP-002],
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        r = math.dist(left, right)
        if self.kernel == "squared_exponential":
            return self.signal * math.exp(-0.5 * r * r)
        return (
            self.signal * (1.0 + _SQRT5 * r + 5.0 * r * r / 3.0) * math.exp(-_SQRT5 * r)
        )

    def _solve(self, column: Sequence[float]) -> list[float]:
        """Return ``L^-1 column`` by forward substitution.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        solution: list[float] = []
        for row, entry in zip(self.lower, column, strict=True):
            size = len(solution)
            solution.append((entry - sum(map(operator.mul, row, solution))) / row[size])
        return solution

    def _extend_training(
        self, point: Vector, value: float
    ) -> tuple[list[float], float]:
        """Append one training point to ``L`` and ``z``; return ``(w, pivot)``.

        References:
            [EQ-SUR-GP-003], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        point = self.scaled(point)
        cross = [self.covariance(point, known) for known in self.training]
        w = self._solve(cross)
        pivot_square = self.signal + self.noise - sum(map(operator.mul, w, w))
        pivot = math.sqrt(max(pivot_square, self.noise))
        self.lower.append([*w, pivot])
        residual = value - self.mean - sum(map(operator.mul, w, self.z))
        self.z.append(residual / pivot)
        self.training.append(point)
        return w, pivot

    def add_training(self, point: Vector, value: float) -> None:
        """Condition on one more observation and update the population.

        References:
            [EQ-SUR-GP-003], [EQ-REL-AKMCS-001],
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        row, pivot = self._extend_training(point, value)
        point = self.training[-1]
        z_new = self.z[-1]
        for index, (x, v) in enumerate(zip(self.population, self.v, strict=True)):
            entry = (self.covariance(x, point) - sum(map(operator.mul, v, row))) / pivot
            v.append(entry)
            self.mu[index] += entry * z_new
            self.var[index] -= entry * entry

    def add_population(self, points: Sequence[Vector]) -> None:
        """Append population points with their full posterior state.

        References:
            [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        for raw in points:
            x = self.scaled(raw)
            v = self._solve([self.covariance(x, known) for known in self.training])
            self.population.append(x)
            self.v.append(v)
            self.mu.append(self.mean + sum(map(operator.mul, v, self.z)))
            self.var.append(self.signal - sum(map(operator.mul, v, v)))

    def learning(self, index: int) -> float:
        """Return ``U = |mu| / sigma`` at one population point.

        References:
            [EQ-REL-AKMCS-001], [REF-ECHARD-GAYTON-LEMAIRE-2011].

        """
        variance = self.var[index]
        if variance <= 0.0:
            return 0.0 if self.mu[index] == 0.0 else math.inf
        return abs(self.mu[index]) / math.sqrt(variance)


@dataclass(frozen=True, slots=True)
class AKMCSIteration:
    """State of AK-MCS after one learning step or population change.

    Args:
        training_size: Number of training points (true-model calls on the
            surrogate's training set).
        population: Monte Carlo population size.
        probability: Surrogate estimate ``N_f / N``.
        min_learning: ``min U`` over the population (excluding training
            points).
        coefficient_of_variation: ``sqrt((1 - P_f) / (N P_f))``.

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """

    training_size: int
    population: int
    probability: float
    min_learning: float
    coefficient_of_variation: float


@dataclass(frozen=True, slots=True)
class AKMCSResult:
    """AK-MCS failure-probability estimate and its learning record.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: ``N_f / N`` from the signs of the final GP mean.
        standard_error: ``sqrt(P_f (1 - P_f) / N)``, the Monte Carlo error of
            the population (surrogate classification error is not included).
        coefficient_of_variation: ``sqrt((1 - P_f) / (N P_f))``.
        population: Final population size ``N``.
        predicted_failures: ``N_f``.
        evaluations: True limit-state (model) calls of the method, read from
            the model's evaluation counter; excludes the reference run.
        initial_size: Size of the initial space-filling design.
        added: Points added by active learning.
        min_learning: Final ``min U``.
        converged: Whether learning stopped with ``min U >= threshold`` and
            the c.o.v. target met (or the population cap reached).
        message: Why the run stopped.
        history: One record per learning step or population enlargement.
        surrogate: Final GP fitted on all training data.
        refits: Number of hyperparameter fits.
        reference_probability: Crude Monte Carlo ``P_f`` of the true limit
            state on the same population (``nan`` unless requested).
        reference_evaluations: Model calls of the reference run.
        misclassified: Population points whose predicted sign differs from
            the true sign (``-1`` unless the reference was requested).

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-RASMUSSEN-WILLIAMS-2006].

    """

    limit_state: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    population: int
    predicted_failures: int
    evaluations: int
    initial_size: int
    added: int
    min_learning: float
    converged: bool
    message: str
    history: tuple[AKMCSIteration, ...]
    surrogate: GaussianProcessRegression
    refits: int
    reference_probability: float
    reference_evaluations: int
    misclassified: int


def _population_estimate(means: Sequence[float]) -> tuple[int, float, float]:
    """Return ``(N_f, P_f, c.o.v.)`` from the predicted means.

    References:
        [EQ-REL-AKMCS-002], [EQ-MC-MEAN-001].

    """
    count = len(means)
    failures = sum(1 for value in means if value <= 0.0)
    probability = failures / count
    if failures == 0:
        return failures, probability, math.inf
    return failures, probability, math.sqrt((1.0 - probability) / failures)


def ak_mcs(
    limit_state: ModelLimitState,
    population: int = 10_000,
    *,
    initial_size: int | None = None,
    design: str = "lhs",
    design_bound: float | None = 5.0,
    kernel: str = "matern52",
    learning_threshold: float = 2.0,
    target_cov: float = 0.05,
    population_growth: int | None = None,
    max_population: int = 200_000,
    max_added: int = 200,
    refit_growth: float = 1.5,
    restarts: int = 3,
    nugget: float = 1e-10,
    compare_with_monte_carlo: bool = False,
    seed: int | RandomStream | None = None,
) -> AKMCSResult:
    """Estimate ``P_f`` by AK-MCS (GP active learning on a MC population).

    Learning function and stopping rule [EQ-REL-AKMCS-001]: with GP mean
    ``mu`` and standard deviation ``sigma`` of ``G`` in standard-normal space,
    ``U(u) = |mu(u)| / sigma(u)``; the next training point is
    ``argmin_i U(u_i)`` over the population, and learning stops when
    ``min_i U(u_i) >= learning_threshold`` (``2`` in Echard et al.). The
    estimate [EQ-REL-AKMCS-002] is ``P_f = N_f / N`` with
    ``N_f = #{i: mu(u_i) <= 0}`` and c.o.v. ``sqrt((1 - P_f) / (N P_f))``.
    When the c.o.v. exceeds ``target_cov`` the population grows by
    ``population_growth`` fresh points (default: the initial size) up to
    ``max_population`` and learning resumes.

    Computational choices (not part of the published method): the GP
    hyperparameters are re-estimated by maximum likelihood (warm-started)
    whenever the training set has grown by the factor ``refit_growth`` since
    the last fit and again before learning is allowed to stop; between fits
    the posterior is updated exactly by rank-one Cholesky updates at fixed
    hyperparameters, which costs ``O(n)`` per population point and step. The
    initial design is a space-filling design in the unit cube mapped
    affinely to the box ``[-b, b]^d`` in standard space (``b =
    design_bound``), or to ``N(0, I)`` by ``Phi^-1`` when ``design_bound`` is
    ``None``; Echard et al. draw the initial points from the population. The
    box matters: a small design mapped through ``Phi^-1`` stays within about
    ``|u| < 2`` and can give a GP that is confidently wrong about a distant
    failure region, so that ``min U >= 2`` holds before any failure point is
    seen (observed on the four-branch function with 12 points and seed 3).
    A run that ends without predicting any failure is reported as not
    converged.

    Args:
        limit_state: Model limit state (``G <= 0`` is failure).
        population: Initial Monte Carlo population size ``N`` (at least 10).
        initial_size: Initial design size (default ``max(12, 2 d + 2)``).
        design: ``"lhs"``, ``"sobol"``, ``"halton"``, or ``"mc"``.
        design_bound: Half-width ``b`` of the initial-design box in standard
            space (positive), or ``None`` to sample the input law.
        kernel: ``"matern52"`` or ``"squared_exponential"``.
        learning_threshold: Stopping value for ``min U`` (positive).
        target_cov: Target c.o.v. of the population estimate (positive).
        population_growth: Points added per enlargement.
        max_population: Population cap (at least ``population``).
        max_added: Maximum number of active-learning additions.
        refit_growth: Training-set growth factor (``> 1``) between
            hyperparameter fits.
        restarts: Optimizer starts for the first fit (later fits use one
            warm start, falling back to ``restarts`` starts on failure).
        nugget: Diagonal variance of the GP in standardized units.
        compare_with_monte_carlo: Also evaluate the true limit state on the
            final population (crude Monte Carlo on the same points) and count
            misclassified signs; these calls are reported separately.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, c.o.v., true-model call count, learning history, final
        surrogate, and the optional same-population reference.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If options are invalid or the GP cannot be fitted.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002], [EQ-SUR-GP-003],
        [EQ-SUR-GP-004].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-OWEN-2013, Chapters 2 and 10].

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    state = limit_state
    dimension = state.dimension
    size = _require_count(population, "population", 10)
    first_size = _require_count(
        max(12, 2 * dimension + 2) if initial_size is None else initial_size,
        "initial_size",
        2,
    )
    if design not in _DESIGNS:
        raise ValueError(f"design must be one of {_DESIGNS}")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    threshold = _require_positive(learning_threshold, "learning_threshold")
    cov_target = _require_positive(target_cov, "target_cov")
    growth = _require_count(
        size if population_growth is None else population_growth,
        "population_growth",
        1,
    )
    cap = _require_count(max_population, "max_population", size)
    additions_limit = _require_count(max_added, "max_added", 0)
    refit_factor = _require_positive(refit_growth, "refit_growth")
    if refit_factor <= 1.0:
        raise ValueError("refit_growth must exceed 1")
    starts = _require_count(restarts, "restarts", 1)
    root = resolve_stream(seed)
    population_generator = root.child("ak-mcs", "population").generator()

    def draw(count: int) -> list[Vector]:
        """Return ``count`` standard-normal population points.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998].

        """
        return [
            tuple(population_generator.gauss(0.0, 1.0) for _ in range(dimension))
            for _ in range(count)
        ]

    before = state.model.evaluations
    unit = make_design(
        design, first_size, dimension, seed=root.child("ak-mcs", "design")
    )
    if design_bound is None:
        training_u: list[Vector] = [
            tuple(standard_normal_quantile(p) for p in point) for point in unit.points
        ]
    else:
        bound = _require_positive(design_bound, "design_bound")
        training_u = [
            tuple(bound * (2.0 * p - 1.0) for p in point) for point in unit.points
        ]
    training_g: list[float] = [state.standard(u) for u in training_u]
    refits = 0
    previous_theta: Vector | None = None

    def fit() -> GaussianProcessRegression:
        """Fit the GP to the current training set (warm-started).

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        nonlocal refits, previous_theta
        refits += 1
        try:
            surrogate = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts if previous_theta is None else 1,
                seed=root.child("ak-mcs", "fit", refits),
                initial_log_parameters=previous_theta,
            )
        except ValueError:
            if previous_theta is None:
                raise
            surrogate = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts,
                seed=root.child("ak-mcs", "fit", refits),
            )
        previous_theta = surrogate.log_parameters
        return surrogate

    surrogate = fit()
    fitted_size = len(training_u)
    points = draw(size)
    posterior = _IncrementalPosterior(surrogate, points)
    selected: set[int] = set()
    history: list[AKMCSIteration] = []
    added = 0
    converged = False
    message = ""
    min_learning = math.inf
    while True:
        best_index = -1
        min_learning = math.inf
        for index in range(len(points)):
            if index in selected:
                continue
            value = posterior.learning(index)
            if value < min_learning:
                min_learning, best_index = value, index
        _, probability, cov = _population_estimate(posterior.mu)
        history.append(
            AKMCSIteration(len(training_u), len(points), probability, min_learning, cov)
        )
        if min_learning >= threshold:
            if fitted_size != len(training_u):
                surrogate = fit()
                fitted_size = len(training_u)
                posterior = _IncrementalPosterior(surrogate, points)
                continue
            if cov <= cov_target:
                converged = True
                message = "min U >= threshold and c.o.v. target met"
                break
            if len(points) >= cap:
                converged = probability > 0.0
                message = (
                    "min U >= threshold; population cap reached"
                    if converged
                    else "no failure predicted up to the population cap"
                )
                break
            fresh = draw(min(growth, cap - len(points)))
            points.extend(fresh)
            posterior.add_population(fresh)
            continue
        if added >= additions_limit:
            message = "max_added reached before min U >= threshold"
            break
        point = points[best_index]
        value_g = state.standard(point)
        training_u.append(point)
        training_g.append(value_g)
        selected.add(best_index)
        added += 1
        if len(training_u) >= refit_factor * fitted_size:
            surrogate = fit()
            fitted_size = len(training_u)
            posterior = _IncrementalPosterior(surrogate, points)
        else:
            posterior.add_training(point, value_g)
    evaluations = state.model.evaluations - before
    if fitted_size != len(training_u):
        surrogate = fit()
    failures, probability, cov = _population_estimate(posterior.mu)
    reference = math.nan
    reference_calls = 0
    misclassified = -1
    if compare_with_monte_carlo:
        reference_before = state.model.evaluations
        truth = [state.standard(u) for u in points]
        reference_calls = state.model.evaluations - reference_before
        reference = sum(1 for g in truth if g <= 0.0) / len(points)
        misclassified = sum(
            1
            for g, mu in zip(truth, posterior.mu, strict=True)
            if (g <= 0.0) != (mu <= 0.0)
        )
    return AKMCSResult(
        limit_state=state.label,
        probability=probability,
        standard_error=math.sqrt(probability * (1.0 - probability) / len(points)),
        coefficient_of_variation=cov,
        population=len(points),
        predicted_failures=failures,
        evaluations=evaluations,
        initial_size=first_size,
        added=added,
        min_learning=min_learning,
        converged=converged,
        message=message,
        history=tuple(history),
        surrogate=surrogate,
        refits=refits,
        reference_probability=reference,
        reference_evaluations=reference_calls,
        misclassified=misclassified,
    )
