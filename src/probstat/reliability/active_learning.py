"""AK-MCS: active-learning Kriging (GP) surrogate on a Monte Carlo population.

Echard, Gayton, and Lemaire's AK-MCS replaces most limit-state calls of a
crude Monte Carlo estimate by a Gaussian-process surrogate that is refined
only where the sign of ``G`` is uncertain:

1. draw a Monte Carlo population ``u_1..u_N ~ N(0, I)`` in standard space;
2. fit a GP to ``G`` on a small space-filling design (LHS, maximin LHS,
   Sobol, ...), optionally with extra seed points;
3. evaluate a learning function on the population (default
   ``U(u) = |mu(u)| / sigma(u)``; also Bichon's ``EFF`` and the entropy
   ``H``, see :mod:`probstat.reliability.learning`) and add the best-rated
   point to the training set (one true-model call);
4. stop learning when the learning function's threshold is met (``min U >=
   2``) or, with ``stopping="bounds"``, when the ``P_f`` bounds from
   ``mu -/+ 1.96 sigma`` are narrow enough;
5. estimate ``P_f = N_f / N`` from the predicted signs; if its c.o.v.
   ``sqrt((1 - P_f) / (N P_f))`` is above the target, enlarge the population
   and resume learning.

The surrogate lives in standard-normal space. By default it is a
constant-mean GP updated incrementally over the population
(:class:`~probstat.reliability.population_posterior.GaussianProcessPopulationPosterior`);
``surrogate=`` accepts any :class:`~probstat.surrogates.SurrogateFactory`
instead (ADR-0107). This module depends on :mod:`probstat.surrogates` (same
architecture layer); ``surrogates`` does not import ``reliability``, so the
dependency stays acyclic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.modeling import standard_normal_quantile
from probstat.propagation import make_design
from probstat.sampling import SamplingDesign, maximin_latin_hypercube
from probstat.surrogates import (
    KERNELS,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    ProbabilisticSurrogate,
    SurrogateFactory,
)
from probstat.surrogates import fit_gaussian_process as _fit_gp

from .learning import (
    BOUNDS_QUANTILE,
    LEARNING_FUNCTIONS,
    LEARNING_THRESHOLDS,
    STOPPING_RULES,
    bounds_satisfied,
    probability_bounds,
    select_learning_point,
    stopping_satisfied,
)
from .model_based import ModelLimitState
from .population_posterior import (
    GaussianProcessPopulationPosterior,
    PopulationPosterior,
    SurrogatePopulationPosterior,
)

AKMCS_DESIGNS: tuple[str, ...] = ("mc", "lhs", "lhs-maximin", "halton", "sobol")
"""Initial-design names accepted by :func:`ak_mcs`."""

AKMCS_GP_REGULARIZATION = GaussianProcessRegularization()
"""Default GP hyperparameter regularization of :func:`ak_mcs` (ADR-0054)."""


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_positive(value: object, name: str) -> float:
    """Validate a strictly positive finite real option.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


@dataclass(frozen=True, slots=True)
class AKMCSIteration:
    """State of AK-MCS after one learning step or population change.

    Args:
        training_size: Number of training points (true-model calls on the
            surrogate's training set).
        population: Monte Carlo population size.
        probability: Surrogate estimate ``N_f / N``.
        min_learning: Best learning value over the population, excluding
            training points: ``min U``, ``max EFF``, or ``max H``.
        coefficient_of_variation: ``sqrt((1 - P_f) / (N P_f))``.
        lower_probability: ``P_f^-`` from ``mu + 1.96 sigma <= 0``.
        upper_probability: ``P_f^+`` from ``mu - 1.96 sigma <= 0``.

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-SCHOBI-SUDRET-MARELLI-2017].

    """

    training_size: int
    population: int
    probability: float
    min_learning: float
    coefficient_of_variation: float
    lower_probability: float = math.nan
    upper_probability: float = math.nan


@dataclass(frozen=True, slots=True)
class AKMCSResult:
    """AK-MCS failure-probability estimate and its learning record.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: ``N_f / N`` from the signs of the final surrogate mean.
        standard_error: ``sqrt(P_f (1 - P_f) / N)``, the Monte Carlo error of
            the population (surrogate classification error is not included).
        coefficient_of_variation: ``sqrt((1 - P_f) / (N P_f))``.
        population: Final population size ``N``.
        predicted_failures: ``N_f``.
        evaluations: True limit-state (model) calls of the method, read from
            the model's evaluation counter; excludes the reference run.
        initial_size: Size of the initial training set: the space-filling
            design plus any ``extra_points``.
        added: Points added by active learning.
        min_learning: Final best learning value (``min U``, ``max EFF``, or
            ``max H``).
        converged: Whether learning stopped by the stopping rule and the
            c.o.v. target was met (or the population cap reached).
        message: Why the run stopped.
        history: One record per learning step or population enlargement.
        surrogate: Final surrogate fitted on all training data (a
            :class:`~probstat.surrogates.GaussianProcessRegression` unless a
            ``surrogate`` factory was given).
        refits: Number of surrogate fits (GP hyperparameter fits, or
            factory calls).
        reference_probability: Crude Monte Carlo ``P_f`` of the true limit
            state on the same population (``nan`` unless requested).
        reference_evaluations: Model calls of the reference run.
        misclassified: Population points whose predicted sign differs from
            the true sign (``-1`` unless the reference was requested).
        learning: Learning function used (``"U"``, ``"EFF"``, or ``"H"``).
        learning_threshold: Stopping value of the learning function.
        stopping: ``"learning"`` or ``"bounds"``.
        lower_probability: Final ``P_f^-`` [EQ-REL-AL-BOUNDS-001].
        upper_probability: Final ``P_f^+``.
        training_points: Training inputs in standard-normal space.
        training_values: Limit-state values at ``training_points``.

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
    surrogate: ProbabilisticSurrogate
    refits: int
    reference_probability: float
    reference_evaluations: int
    misclassified: int
    learning: str = "U"
    learning_threshold: float = 2.0
    stopping: str = "learning"
    lower_probability: float = math.nan
    upper_probability: float = math.nan
    training_points: tuple[Vector, ...] = field(default=())
    training_values: tuple[float, ...] = field(default=())


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


def _unit_design(
    design: str, size: int, dimension: int, seed: RandomStream
) -> SamplingDesign:
    """Return the unit-cube initial design of :func:`ak_mcs`.

    References:
        [EQ-SAMP-LHS-001], [EQ-SAMP-MAXIMIN-001], [REF-MORRIS-MITCHELL-1995],
        [REF-OWEN-2013, Chapters 10 and 15].

    """
    if design == "lhs-maximin":
        return maximin_latin_hypercube(size, dimension, seed=seed)
    return make_design(design, size, dimension, seed=seed)


def _design_points(
    design: str,
    size: int,
    dimension: int,
    design_bound: float | None,
    seed: RandomStream,
) -> list[Vector]:
    """Return the initial design in standard-normal space.

    The unit-cube design is mapped affinely to ``[-b, b]^d`` with
    ``b = design_bound``, or through ``Phi^-1`` when ``design_bound`` is
    ``None``.

    Raises:
        ValueError: If ``design_bound`` is not a positive finite real.

    References:
        [EQ-SAMP-LHS-001], [EQ-SAMP-MAXIMIN-001], [REF-MORRIS-MITCHELL-1995].

    """
    unit = _unit_design(design, size, dimension, seed)
    if design_bound is None:
        return [
            tuple(standard_normal_quantile(p) for p in point) for point in unit.points
        ]
    bound = _require_positive(design_bound, "design_bound")
    return [tuple(bound * (2.0 * p - 1.0) for p in point) for point in unit.points]


def _extra_points(
    points: Sequence[Sequence[float]] | None, dimension: int
) -> list[Vector]:
    """Validate the optional extra initial points (standard-normal space).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if points is None:
        return []
    checked: list[Vector] = []
    for point in points:
        vector = tuple(require_finite_real(c, name="extra point") for c in point)
        if len(vector) != dimension:
            raise ValueError("every extra point must have the limit-state dimension")
        checked.append(vector)
    return checked


def ak_mcs(
    limit_state: ModelLimitState,
    population: int = 10_000,
    *,
    initial_size: int | None = None,
    design: str = "lhs",
    design_bound: float | None = 5.0,
    kernel: str = "matern52",
    learning: str = "U",
    learning_threshold: float | None = None,
    stopping: str = "learning",
    bounds_tolerance: float = 0.1,
    target_cov: float = 0.05,
    population_growth: int | None = None,
    max_population: int = 200_000,
    max_added: int = 200,
    refit_growth: float = 1.5,
    restarts: int = 3,
    nugget: float = 1e-10,
    extra_points: Sequence[Sequence[float]] | None = None,
    surrogate: SurrogateFactory | None = None,
    compare_with_monte_carlo: bool = False,
    seed: int | RandomStream | None = None,
    regularization: GaussianProcessRegularization | None = AKMCS_GP_REGULARIZATION,
) -> AKMCSResult:
    """Estimate ``P_f`` by AK-MCS (surrogate active learning on a MC population).

    Learning function and stopping rule: with surrogate mean ``mu`` and
    standard deviation ``sigma`` of ``G`` in standard-normal space, the
    default ``learning="U"`` uses ``U(u) = |mu(u)| / sigma(u)``
    [EQ-REL-AKMCS-001]; the next training point is ``argmin_i U(u_i)`` over
    the population, and learning stops when ``min_i U(u_i) >=
    learning_threshold`` (``2`` in Echard et al.). ``learning="EFF"`` uses
    Bichon's expected feasibility function with ``epsilon = 2 sigma``
    [EQ-REL-AL-EFF-001] (``argmax``, stop at ``max EFF <= 0.001``), and
    ``learning="H"`` the information entropy of Lv, Lu, and Wang
    [EQ-REL-AL-H-001] (``argmax``, stop at ``max H <= 0.5``).
    ``learning_threshold=None`` takes the function's default from
    :data:`~probstat.reliability.LEARNING_THRESHOLDS`. ``EFF`` is in the units
    of ``G`` and ``H`` contains ``ln sigma``, so their thresholds assume ``G``
    of order one. With ``stopping="bounds"`` learning stops instead when
    ``(P_f^+ - P_f^-) / P_f^0 <= bounds_tolerance``, where ``P_f^-`` and
    ``P_f^+`` count the points with ``mu + 1.96 sigma <= 0`` and
    ``mu - 1.96 sigma <= 0`` [EQ-REL-AL-BOUNDS-001]; the learning function
    then only chooses the points. The estimate [EQ-REL-AKMCS-002] is
    ``P_f = N_f / N`` with ``N_f = #{i: mu(u_i) <= 0}`` and c.o.v.
    ``sqrt((1 - P_f) / (N P_f))``. When the c.o.v. exceeds ``target_cov``
    the population grows by ``population_growth`` fresh points (default: the
    initial size) up to ``max_population`` and learning resumes.

    Surrogate: by default a constant-mean GP whose hyperparameters are
    re-estimated by maximum likelihood (warm-started) whenever the training
    set has grown by the factor ``refit_growth`` since the last fit and again
    before learning is allowed to stop; between fits the posterior over the
    population is updated exactly by rank-one Cholesky updates at fixed
    hyperparameters, which costs ``O(n)`` per population point and step
    (:class:`~probstat.reliability.GaussianProcessPopulationPosterior`).
    ``surrogate`` replaces it by any
    :class:`~probstat.surrogates.SurrogateFactory`: the factory is called on
    the initial design and after every added point, and the population is
    re-predicted with the returned surrogate
    (:class:`~probstat.reliability.SurrogatePopulationPosterior`);
    ``kernel``, ``refit_growth``, ``restarts``, ``nugget``, and
    ``regularization`` then do not apply.

    Initial design: a space-filling design in the unit cube (``"lhs"``,
    ``"lhs-maximin"`` (Morris-Mitchell maximin LHS), ``"sobol"``,
    ``"halton"``, or ``"mc"``) mapped affinely to the box ``[-b, b]^d`` in
    standard space (``b = design_bound``), or to ``N(0, I)`` by ``Phi^-1``
    when ``design_bound`` is ``None``; Echard et al. draw the initial points
    from the population. ``extra_points`` (standard-normal coordinates, for
    example the ``design_point_u`` of each FORM result of
    :func:`~probstat.reliability.find_design_points`) are evaluated and added
    after the design. The learning function only sees where the surrogate is
    uncertain, so a failure region that no training point is near can be
    predicted safe with small ``sigma`` and never be visited: a small design
    mapped through ``Phi^-1`` stays within about ``|u| < 2`` and gave a GP
    that was confidently wrong about the distant failure regions of the
    four-branch function, with ``min U >= 2`` before any failure point was
    seen (12 points, seed 3). The box covers the tails up to ``|u| = b``, a
    maximin design spreads the few points evenly over it, and design-point
    seeds place training data in each failure region found by FORM. A run
    that ends without predicting any failure is reported as not converged.

    The GP hyperparameters are fitted with ``regularization`` (default
    :data:`AKMCS_GP_REGULARIZATION`: length scales between a tenth of the
    smallest coordinate gap and 20 times the extent of the training set,
    signal variance in ``[1e-3, 1e3]`` standardized units, diagonal floor
    ``1e-8 sigma_f^2``; ADR-0054). Unregularized maximum likelihood
    (``regularization=None``, the behaviour before ADR-0054) can end at
    near-singular hyperparameters: on ``G = 3 + u1^2/4 - u2`` length scales
    ``1e3``-``1e5`` and signal-to-nugget ``~1e18`` gave ``min U`` of
    ``1e2``-``2e3`` right after the 12-point initial design; on the
    four-branch function with seed 5 a white-noise optimum (length scales
    ``0.002``, far below the smallest design gap ``0.33``) was kept by every
    warm-started refit and learning ran to 198 additions with 7 misclassified
    population points.

    Args:
        limit_state: Model limit state (``G <= 0`` is failure).
        population: Initial Monte Carlo population size ``N`` (at least 10).
        initial_size: Initial design size (default ``max(12, 2 d + 2)``).
        design: One of :data:`AKMCS_DESIGNS`.
        design_bound: Half-width ``b`` of the initial-design box in standard
            space (positive), or ``None`` to sample the input law.
        kernel: ``"matern52"`` or ``"squared_exponential"``.
        learning: ``"U"``, ``"EFF"``, or ``"H"``.
        learning_threshold: Positive stopping value of the learning function,
            or ``None`` for its default (2 for ``U``, 0.001 for ``EFF``, 0.5
            for ``H``).
        stopping: ``"learning"`` (learning-function threshold) or
            ``"bounds"`` (relative width of the ``P_f`` bounds).
        bounds_tolerance: Relative bound width of ``stopping="bounds"``.
        target_cov: Target c.o.v. of the population estimate (positive).
        population_growth: Points added per enlargement.
        max_population: Population cap (at least ``population``).
        max_added: Maximum number of active-learning additions.
        refit_growth: Training-set growth factor (``> 1``) between
            hyperparameter fits.
        restarts: Optimizer starts for the first fit (later fits use one
            warm start, falling back to ``restarts`` starts on failure).
        nugget: Diagonal variance of the GP in standardized units.
        extra_points: Optional standard-normal points added to the initial
            design (each costs one model call).
        surrogate: Optional refit hook ``(points, values, previous) ->
            surrogate`` replacing the incremental GP.
        compare_with_monte_carlo: Also evaluate the true limit state on the
            final population (crude Monte Carlo on the same points) and count
            misclassified signs; these calls are reported separately.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        regularization: GP hyperparameter regularization, or ``None`` for
            unregularized maximum likelihood.

    Returns:
        Estimate, c.o.v., ``P_f`` bounds, true-model call count, learning
        history, final surrogate and training data, and the optional
        same-population reference.

    Raises:
        TypeError: If ``limit_state`` has the wrong type or ``surrogate`` is
            not callable.
        ValueError: If options are invalid or the surrogate cannot be fitted.

    Equation:
        [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002], [EQ-REL-AL-EFF-001],
        [EQ-REL-AL-H-001], [EQ-REL-AL-BOUNDS-001], [EQ-SUR-GP-003],
        [EQ-SUR-GP-004], [EQ-SAMP-MAXIMIN-001].

    References:
        [REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-BICHON-ETAL-2008],
        [REF-LV-LU-WANG-2015], [REF-SCHOBI-SUDRET-MARELLI-2017],
        [REF-MORRIS-MITCHELL-1995], [REF-RASMUSSEN-WILLIAMS-2006],
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
    if design not in AKMCS_DESIGNS:
        raise ValueError(f"design must be one of {AKMCS_DESIGNS}")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if learning not in LEARNING_FUNCTIONS:
        raise ValueError(f"learning must be one of {LEARNING_FUNCTIONS}")
    if stopping not in STOPPING_RULES:
        raise ValueError(f"stopping must be one of {STOPPING_RULES}")
    threshold = (
        LEARNING_THRESHOLDS[learning]
        if learning_threshold is None
        else _require_positive(learning_threshold, "learning_threshold")
    )
    bound_width = _require_positive(bounds_tolerance, "bounds_tolerance")
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
    if surrogate is not None and not callable(surrogate):
        raise TypeError("surrogate must be a callable surrogate factory")
    seeds = _extra_points(extra_points, dimension)
    if stopping == "bounds":
        criterion = "P_f bounds within tolerance"
    elif learning == "U":
        criterion = "min U >= threshold"
    else:
        criterion = f"max {learning} <= threshold"
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
    training_u = _design_points(
        design, first_size, dimension, design_bound, root.child("ak-mcs", "design")
    )
    training_u.extend(seeds)
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
            fitted = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts if previous_theta is None else 1,
                seed=root.child("ak-mcs", "fit", refits),
                initial_log_parameters=previous_theta,
                regularization=regularization,
            )
        except ValueError:
            if previous_theta is None:
                raise
            fitted = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts,
                seed=root.child("ak-mcs", "fit", refits),
                regularization=regularization,
            )
        previous_theta = fitted.log_parameters
        return fitted

    points = draw(size)
    posterior: PopulationPosterior
    model: ProbabilisticSurrogate
    generic: SurrogatePopulationPosterior | None = None
    if surrogate is None:
        gp = fit()
        model = gp
        posterior = GaussianProcessPopulationPosterior(gp, points)
    else:
        generic = SurrogatePopulationPosterior(
            surrogate, training_u, training_g, points
        )
        posterior = generic
        model = generic.surrogate
    incremental = generic is None
    fitted_size = len(training_u)

    def rebuild() -> None:
        """Refit the GP on all training data and rebuild the population posterior.

        References:
            [EQ-SUR-GP-003], [EQ-SUR-GP-004].

        """
        nonlocal model, posterior, fitted_size
        refitted = fit()
        model = refitted
        fitted_size = len(training_u)
        posterior = GaussianProcessPopulationPosterior(refitted, points)

    selected: set[int] = set()
    history: list[AKMCSIteration] = []
    added = 0
    converged = False
    message = ""
    min_learning = math.inf
    while True:
        best_index, min_learning = select_learning_point(
            learning, posterior.means, posterior.variances, selected
        )
        _, probability, cov = _population_estimate(posterior.means)
        lower, _, upper = probability_bounds(
            posterior.means, posterior.variances, BOUNDS_QUANTILE
        )
        history.append(
            AKMCSIteration(
                len(training_u),
                len(points),
                probability,
                min_learning,
                cov,
                lower,
                upper,
            )
        )
        if stopping == "learning":
            done = stopping_satisfied(learning, min_learning, threshold)
        else:
            done = bounds_satisfied(lower, probability, upper, bound_width)
        if done or best_index < 0:
            if incremental and fitted_size != len(training_u):
                rebuild()
                continue
            if cov <= cov_target:
                converged = True
                message = f"{criterion} and c.o.v. target met"
                break
            if len(points) >= cap:
                converged = probability > 0.0
                message = (
                    f"{criterion}; population cap reached"
                    if converged
                    else "no failure predicted up to the population cap"
                )
                break
            fresh = draw(min(growth, cap - len(points)))
            points.extend(fresh)
            posterior.add_population(fresh)
            continue
        if added >= additions_limit:
            message = f"max_added reached before {criterion}"
            break
        point = points[best_index]
        value_g = state.standard(point)
        training_u.append(point)
        training_g.append(value_g)
        selected.add(best_index)
        added += 1
        if incremental and len(training_u) >= refit_factor * fitted_size:
            rebuild()
        else:
            posterior.add_training(point, value_g)
    evaluations = state.model.evaluations - before
    if incremental and fitted_size != len(training_u):
        # Refit on every training point and rebuild the population posterior
        # so that the reported probability and the returned surrogate come
        # from the same model.
        rebuild()
    if generic is not None:
        model = generic.surrogate
        refits = generic.fits
    failures, probability, cov = _population_estimate(posterior.means)
    lower, _, upper = probability_bounds(
        posterior.means, posterior.variances, BOUNDS_QUANTILE
    )
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
            for g, mu in zip(truth, posterior.means, strict=True)
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
        initial_size=first_size + len(seeds),
        added=added,
        min_learning=min_learning,
        converged=converged,
        message=message,
        history=tuple(history),
        surrogate=model,
        refits=refits,
        reference_probability=reference,
        reference_evaluations=reference_calls,
        misclassified=misclassified,
        learning=learning,
        learning_threshold=threshold,
        stopping=stopping,
        lower_probability=lower,
        upper_probability=upper,
        training_points=tuple(training_u),
        training_values=tuple(training_g),
    )
