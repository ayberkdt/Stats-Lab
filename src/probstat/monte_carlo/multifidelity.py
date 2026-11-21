"""Multifidelity Monte Carlo and approximate control variates.

A high-fidelity model ``f_0`` with mean ``mu_0 = E[f_0(Z)]`` is expensive;
cheaper models ``f_1, ..., f_(k-1)`` of the same random input ``Z`` are
correlated with it but have different, unknown means. The estimators here
spend a cost budget on *all* models and stay unbiased for ``mu_0`` because the
low-fidelity models enter only through differences of two sample means of the
same model:

* multifidelity Monte Carlo (MFMC) on nested samples with the closed-form
  optimal coefficients and allocation of Peherstorfer, Willcox and Gunzburger;
* approximate control variates (ACV-IS and ACV-MF) of Gorodetsky, Geraci,
  Eldred and Jakeman with the optimal weights for a given sample design
  (:mod:`probstat.monte_carlo.approximate_control_variates`).

Every variance in this module is a function of the model covariance matrix
``C`` (``C_ij = Cov(f_i, f_j)``) and the sample counts. ``C`` is an input: it
is either known or estimated by :func:`pilot_statistics` from a pilot sample
that must be independent of the estimation samples. With fixed (pilot)
weights every estimator is exactly unbiased; its reported variance is the
variance *predicted from the supplied covariance* and is exact only when that
covariance is the true one.

Models are indexed from zero: model 0 is the high-fidelity model (the sources
call it model 1). The multilevel best linear unbiased estimator and the
method comparison live in :mod:`probstat.monte_carlo.multilevel_blue`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, cholesky, cholesky_solve, symmetrize

ModelFunction = Callable[[Any], float]
InputSampler = Callable[[int, RandomStream], Sequence[Any]]

_MAX_EXHAUSTIVE_MODELS = 12


# --------------------------------------------------------------------------
# Shared validation and sampling helpers
# --------------------------------------------------------------------------


def _as_covariance(covariance: Iterable[Iterable[object]]) -> Matrix:
    """Return a validated symmetric positive-definite model covariance.

    Raises:
        ValueError: If the matrix is not square, symmetric, and positive
            definite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    matrix = symmetrize(covariance, name="covariance")
    try:
        cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            "covariance must be symmetric positive definite (models that are "
            "exactly collinear must be removed)"
        ) from error
    return matrix


def _as_costs(costs: Sequence[float], size: int) -> Vector:
    """Return validated positive per-evaluation model costs.

    Raises:
        ValueError: If the length is not ``size`` or a cost is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    values = tuple(require_finite_real(cost, name="cost") for cost in costs)
    if len(values) != size:
        raise ValueError(f"costs must have one entry per model: expected {size}")
    if any(value <= 0.0 for value in values):
        raise ValueError("costs must be positive")
    return values


def _as_budget(budget: float) -> float:
    """Return a validated positive cost budget.

    Raises:
        ValueError: If the budget is not a positive finite number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    value = require_finite_real(budget, name="budget")
    if value <= 0.0:
        raise ValueError("budget must be positive")
    return value


def _as_count(value: object, *, name: str, minimum: int) -> int:
    """Return a validated integer count of at least ``minimum``.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _as_model_subset(
    models: Sequence[int] | None, size: int, *, name: str = "models"
) -> tuple[int, ...]:
    """Return distinct model indices that start with the high-fidelity model.

    Raises:
        ValueError: If an index is out of range or repeated, or model 0 is
            not the first entry.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if models is None:
        return tuple(range(size))
    subset = tuple(models)
    for index in subset:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{name} must hold integer model indices")
        if not 0 <= index < size:
            raise ValueError(f"{name} holds index {index} outside 0..{size - 1}")
    if len(set(subset)) != len(subset):
        raise ValueError(f"{name} must not repeat a model")
    if not subset or subset[0] != 0:
        raise ValueError(f"{name} must start with the high-fidelity model 0")
    return subset


def _draw_inputs(sampler: InputSampler, count: int, stream: RandomStream) -> list[Any]:
    """Draw ``count`` shared model inputs from ``stream``.

    Raises:
        ValueError: If the sampler does not return exactly ``count`` inputs.

    References:
        [EQ-MC-MF-001], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    inputs = list(sampler(count, stream))
    if len(inputs) != count:
        raise ValueError(
            f"input_sampler returned {len(inputs)} inputs where {count} were requested"
        )
    return inputs


def _evaluate(model: ModelFunction, inputs: Sequence[Any], index: int) -> list[float]:
    """Evaluate one model on a sequence of inputs.

    Raises:
        ValueError: If an output is not a finite real number.

    References:
        [EQ-MC-MF-001], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    return [
        require_finite_real(model(item), name=f"output of model {index}")
        for item in inputs
    ]


def _mean(values: Sequence[float]) -> float:
    """Return the sample mean of a non-empty sequence.

    References:
        [EQ-MC-MEAN-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    return math.fsum(values) / len(values)


def _solve_spd(matrix: Sequence[Sequence[float]], right: Sequence[float]) -> Vector:
    """Solve a symmetric positive-definite system by Cholesky factorization.

    Raises:
        ValueError: If the matrix is not numerically positive definite.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    lower = cholesky(tuple(tuple(row) for row in matrix))
    return cholesky_solve(lower, tuple(right))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PilotStatistics:
    """Joint sample statistics of ``k`` models on one shared pilot sample.

    Args:
        means: Sample means of the models.
        covariance: Unbiased sample covariance matrix (divisor ``n - 1``).
        samples: Pilot sample size ``n``.
        evaluations: Model evaluations spent, ``n`` for every model.

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """

    means: Vector
    covariance: Matrix
    samples: int
    evaluations: tuple[int, ...]

    @property
    def standard_deviations(self) -> Vector:
        """Return the sample standard deviations ``s_i = sqrt(C_ii)``.

        Equation:
            [EQ-MC-MF-001].

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

        """
        return tuple(
            math.sqrt(self.covariance[i][i]) for i in range(len(self.covariance))
        )

    @property
    def correlation(self) -> Matrix:
        """Return the sample correlation matrix ``C_ij / (s_i s_j)``.

        Raises:
            ValueError: If a model has zero sample variance.

        Equation:
            [EQ-MC-MF-001].

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

        """
        deviations = self.standard_deviations
        if any(value <= 0.0 for value in deviations):
            raise ValueError("a model with zero sample variance has no correlation")
        size = len(deviations)
        return tuple(
            tuple(
                1.0
                if i == j
                else self.covariance[i][j] / (deviations[i] * deviations[j])
                for j in range(size)
            )
            for i in range(size)
        )

    def cost(self, costs: Sequence[float]) -> float:
        """Return the pilot cost ``n sum_i w_i``.

        Args:
            costs: Positive cost ``w_i`` of one evaluation of every model.

        Returns:
            Total pilot cost in the units of ``costs``.

        Raises:
            ValueError: If ``costs`` has the wrong length or a non-positive
                entry.

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3]. The pilot
            cost is not part of the estimator budget.

        """
        weights = _as_costs(costs, len(self.means))
        return self.samples * math.fsum(weights)


@dataclass(frozen=True, slots=True)
class MultifidelityEstimate:
    """A multifidelity estimate of the high-fidelity mean with its accounting.

    Args:
        method: ``"MFMC"``, ``"ACV-MF"``, ``"ACV-IS"``, or ``"MLBLUE"``.
        estimate: Estimate of ``E[f_0(Z)]``.
        variance: Estimator variance predicted from the covariance that
            defined the weights; exact only if that covariance is the true
            one and independent of the estimation samples.
        high_fidelity_mean: Plain sample mean of the high-fidelity outputs
            used by the estimator (the Monte Carlo estimate from the same
            high-fidelity evaluations).
        high_fidelity_samples: Number of high-fidelity evaluations.
        evaluations: Evaluations of every supplied model, in model order.
        cost: ``sum_i w_i * evaluations_i``, or ``None`` when no costs were
            given.

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020], [REF-SCHADEN-ULLMANN-2020].

    """

    method: str
    estimate: float
    variance: float
    high_fidelity_mean: float
    high_fidelity_samples: int
    evaluations: tuple[int, ...]
    cost: float | None

    @property
    def standard_error(self) -> float:
        """Return the square root of the predicted estimator variance.

        Equation:
            [EQ-MC-MF-002], [EQ-MC-MF-005], [EQ-MC-MF-009].

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

        """
        return math.sqrt(self.variance)

    @property
    def total_evaluations(self) -> int:
        """Return the number of model evaluations of all fidelities.

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016]. No equation is
            evaluated.

        """
        return sum(self.evaluations)


@dataclass(frozen=True, slots=True)
class MFMCAllocation:
    """Coefficients and nested sample sizes of a multifidelity Monte Carlo run.

    All sequences are aligned with ``models``, which lists the selected model
    indices with the high-fidelity model first and then by decreasing
    ``|rho_0i|``.

    Args:
        models: Selected model indices.
        correlations: Correlations ``rho_0i`` of the selected models with
            the high-fidelity model (the first entry is 1).
        coefficients: Control coefficients ``a_i = rho_0i s_0 / s_i`` (the
            first entry is 1 and multiplies the high-fidelity mean).
        ratios: Optimal real ratios ``r_i = m_i / m_0``.
        continuous_samples: Optimal real sample sizes ``m_i = r_i m_0``.
        samples: Integer sample sizes ``floor(m_i)``, nondecreasing.
        continuous_variance: Estimator variance at ``continuous_samples``.
        variance: Estimator variance at the integer ``samples``.
        mc_variance: Variance ``C_00 w_0 / p`` of plain Monte Carlo that
            spends the whole budget on the high-fidelity model (real-valued
            sample size).
        cost: Cost ``sum_i w_i m_i`` of the integer allocation.
        budget: Cost budget ``p``.
        evaluations: Evaluations of every model of the covariance matrix
            (zero for models that were not selected).

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """

    models: tuple[int, ...]
    correlations: Vector
    coefficients: Vector
    ratios: Vector
    continuous_samples: Vector
    samples: tuple[int, ...]
    continuous_variance: float
    variance: float
    mc_variance: float
    cost: float
    budget: float
    evaluations: tuple[int, ...]

    @property
    def variance_ratio(self) -> float:
        """Return ``variance / mc_variance`` (below one favors MFMC).

        Equation:
            [EQ-MC-MF-003].

        References:
            [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

        """
        return self.variance / self.mc_variance


# --------------------------------------------------------------------------
# Pilot statistics
# --------------------------------------------------------------------------


def pilot_statistics(
    models: Sequence[ModelFunction],
    input_sampler: InputSampler,
    samples: int,
    *,
    seed: int | RandomStream | None = None,
) -> PilotStatistics:
    """Estimate joint means and the covariance matrix of ``k`` models.

    All models are evaluated on the same ``n`` inputs
    ``input_sampler(n, stream)`` with ``stream = root.child("pilot")``; the
    result holds the sample means and the unbiased sample covariance
    ``C_ij = sum_n (y_i^n - ybar_i)(y_j^n - ybar_j) / (n - 1)``.

    The pilot sample must be independent of the samples of the estimator
    that uses these statistics: the unbiasedness of MFMC, ACV, and MLBLUE
    holds for weights that are fixed with respect to the estimation samples.
    The estimators of this package draw from other child streams of the same
    root, so passing one seed to the pilot and to the estimator keeps them
    independent. Predicted variances computed from a pilot covariance are
    themselves estimates.

    Args:
        models: Model callables ``f_i(input) -> float``, high-fidelity first.
        input_sampler: ``input_sampler(n, stream)`` returning ``n``
            independent inputs drawn from ``stream``.
        samples: Pilot sample size ``n`` (at least 2; a positive-definite
            covariance needs ``n > k``).
        seed: Root seed or stream.

    Returns:
        Means, covariance, and the evaluation count.

    Raises:
        ValueError: If ``models`` is empty, ``samples < 2``, the sampler
            returns the wrong number of inputs, or an output is not finite.

    Equation:
        [EQ-MC-MF-001].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    functions = tuple(models)
    if not functions:
        raise ValueError("models must name at least one model")
    count = _as_count(samples, name="samples", minimum=2)
    inputs = _draw_inputs(input_sampler, count, resolve_stream(seed).child("pilot"))
    outputs = [
        _evaluate(function, inputs, index) for index, function in enumerate(functions)
    ]
    means = tuple(_mean(values) for values in outputs)
    centered = [
        [value - mean for value in values]
        for values, mean in zip(outputs, means, strict=True)
    ]
    size = len(functions)
    covariance = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            entry = math.fsum(
                a * b for a, b in zip(centered[i], centered[j], strict=True)
            ) / (count - 1)
            covariance[i][j] = entry
            covariance[j][i] = entry
    return PilotStatistics(
        means=means,
        covariance=tuple(tuple(row) for row in covariance),
        samples=count,
        evaluations=(count,) * size,
    )


# --------------------------------------------------------------------------
# Multifidelity Monte Carlo
# --------------------------------------------------------------------------


def _correlations_with_first(matrix: Matrix) -> Vector:
    """Return ``rho_0i = C_0i / sqrt(C_00 C_ii)`` for every model.

    References:
        [EQ-MC-MF-001], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    return tuple(
        1.0 if i == 0 else matrix[0][i] / math.sqrt(matrix[0][0] * matrix[i][i])
        for i in range(len(matrix))
    )


def _mfmc_order(matrix: Matrix, subset: Sequence[int]) -> tuple[int, ...]:
    """Return ``subset`` with model 0 first, then by decreasing ``|rho_0i|``.

    Ties keep the smaller model index first.

    References:
        [EQ-MC-MF-004], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    rho = _correlations_with_first(matrix)
    rest = sorted((i for i in subset if i != 0), key=lambda i: (-abs(rho[i]), i))
    return (0, *rest)


def _mfmc_violation(rho: Sequence[float], weights: Sequence[float]) -> str | None:
    """Return the first violated MFMC condition, or ``None`` if admissible.

    ``rho`` and ``weights`` are ordered with the high-fidelity model first.
    The conditions are ``1 = |rho_0| > |rho_1| > ... > |rho_(q-1)| > 0`` and
    ``w_(i-1) / w_i > (rho_(i-1)^2 - rho_i^2) / (rho_i^2 - rho_(i+1)^2)``
    with ``rho_q = 0``.

    References:
        [EQ-MC-MF-004], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    squares = [value * value for value in rho] + [0.0]
    for i in range(1, len(rho)):
        if not squares[i - 1] > squares[i] > 0.0:
            return (
                "the correlations with the high-fidelity model must be "
                "non-zero and strictly decreasing in magnitude"
            )
    for i in range(1, len(rho)):
        left = weights[i - 1] / weights[i]
        right = (squares[i - 1] - squares[i]) / (squares[i] - squares[i + 1])
        if not left > right:
            return (
                f"cost ratio w/w' = {left:.6g} of consecutive models at "
                f"position {i} does not exceed the correlation ratio "
                f"{right:.6g}"
            )
    return None


def mfmc_admissible(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    *,
    models: Sequence[int] | None = None,
) -> bool:
    """Check the ordering and cost conditions of the MFMC optimal allocation.

    The selected models are ordered by decreasing ``|rho_0i|``. The
    closed-form allocation [EQ-MC-MF-003] is the budget-constrained optimum
    when, with ``rho_q = 0`` after the last model,

    ``1 > |rho_1| > ... > |rho_(q-1)| > 0`` and
    ``w_(i-1) / w_i > (rho_(i-1)^2 - rho_i^2) / (rho_i^2 - rho_(i+1)^2)``

    for ``i = 1..q-1`` (positions in the ordered subset). The second
    condition is equivalent to strictly increasing ratios ``r_i``, so the
    nested samples are well defined.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        costs: Positive cost of one evaluation of every model.
        models: Model indices to use, starting with 0; default all.

    Returns:
        ``True`` when both conditions hold (always for model 0 alone).

    Raises:
        ValueError: If the covariance, costs, or indices are invalid.

    Equation:
        [EQ-MC-MF-004].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    matrix = _as_covariance(covariance)
    weights = _as_costs(costs, len(matrix))
    order = _mfmc_order(matrix, _as_model_subset(models, len(matrix)))
    rho = _correlations_with_first(matrix)
    return _mfmc_violation([rho[i] for i in order], [weights[i] for i in order]) is None


def _mfmc_variance_ordered(
    matrix: Matrix,
    order: Sequence[int],
    counts: Sequence[float],
    coefficients: Sequence[float],
) -> float:
    """Return the MFMC variance for ordered models and nondecreasing counts.

    ``Var = C_00 / m_0 + sum_(i>=1) (1/m_(i-1) - 1/m_i)
    (a_i^2 C_ii - 2 a_i C_0i)``.

    References:
        [EQ-MC-MF-002], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    terms = [matrix[0][0] / counts[0]]
    for position in range(1, len(order)):
        index = order[position]
        a = coefficients[position]
        gap = 1.0 / counts[position - 1] - 1.0 / counts[position]
        terms.append(gap * (a * a * matrix[index][index] - 2.0 * a * matrix[0][index]))
    return math.fsum(terms)


def mfmc_variance(
    covariance: Iterable[Iterable[object]],
    samples: Sequence[float],
    *,
    models: Sequence[int] | None = None,
    coefficients: Sequence[float] | None = None,
) -> float:
    """Return the variance of the MFMC estimator for given nested samples.

    For the models ``i_0 = 0, i_1, ..., i_(q-1)`` in the given order, sample
    sizes ``m_0 <= m_1 <= ... <= m_(q-1)`` (model ``i_j`` is evaluated on the
    first ``m_j`` of the shared inputs), and coefficients ``a_j``,

    ``Var = C_00 / m_0 + sum_(j=1..q-1) (1/m_(j-1) - 1/m_j)
    (a_j^2 C_jj - 2 a_j C_0j)``.

    The formula holds for any fixed coefficients; the default
    ``a_j = C_0j / C_jj = rho_0j s_0 / s_j`` minimizes it for every sample
    allocation.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        samples: Positive nondecreasing sample sizes, aligned with
            ``models`` (real values are allowed for planning).
        models: Model indices in estimator order, starting with 0; default
            all models in index order.
        coefficients: Coefficients aligned with ``models`` (the first entry
            belongs to the high-fidelity model and is ignored); default the
            optimal ones.

    Returns:
        Estimator variance.

    Raises:
        ValueError: If a length disagrees, the sample sizes are not positive
            and nondecreasing, or the covariance is invalid.

    Equation:
        [EQ-MC-MF-002].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    matrix = _as_covariance(covariance)
    order = _as_model_subset(models, len(matrix))
    counts = [require_finite_real(value, name="samples") for value in samples]
    if len(counts) != len(order):
        raise ValueError("samples must have one entry per selected model")
    if counts[0] <= 0.0 or any(b < a for a, b in itertools.pairwise(counts)):
        raise ValueError("samples must be positive and nondecreasing")
    if coefficients is None:
        weights = [matrix[0][i] / matrix[i][i] for i in order]
    else:
        weights = [require_finite_real(a, name="coefficient") for a in coefficients]
        if len(weights) != len(order):
            raise ValueError("coefficients must have one entry per selected model")
    return _mfmc_variance_ordered(matrix, order, counts, weights)


def _mfmc_plan(
    matrix: Matrix, weights: Vector, budget: float, order: tuple[int, ...]
) -> MFMCAllocation:
    """Build the closed-form MFMC allocation for an admissible ordered subset.

    Raises:
        ValueError: If the budget does not pay for one sample of every
            selected model.

    References:
        [EQ-MC-MF-003], [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    rho_all = _correlations_with_first(matrix)
    rho = [rho_all[i] for i in order]
    squares = [value * value for value in rho] + [0.0]
    denominator = 1.0 - squares[1]
    ratios = [1.0] + [
        math.sqrt(
            weights[0]
            * (squares[j] - squares[j + 1])
            / (weights[order[j]] * denominator)
        )
        for j in range(1, len(order))
    ]
    spend = math.fsum(weights[i] * r for i, r in zip(order, ratios, strict=True))
    first = budget / spend
    if first < 1.0:
        raise ValueError(
            "budget is too small for one evaluation of every selected model "
            f"at the optimal ratios (needs at least {spend:.6g})"
        )
    continuous = tuple(first * r for r in ratios)
    counts = tuple(math.floor(value * (1.0 + 1e-12)) for value in continuous)
    coefficients = tuple(1.0 if i == 0 else matrix[0][i] / matrix[i][i] for i in order)
    closed_form = (
        matrix[0][0]
        / budget
        * math.fsum(
            math.sqrt(weights[i] * (squares[j] - squares[j + 1]))
            for j, i in enumerate(order)
        )
        ** 2
    )
    evaluations = [0] * len(matrix)
    for index, count in zip(order, counts, strict=True):
        evaluations[index] = count
    return MFMCAllocation(
        models=order,
        correlations=tuple(rho),
        coefficients=coefficients,
        ratios=tuple(ratios),
        continuous_samples=continuous,
        samples=counts,
        continuous_variance=closed_form,
        variance=_mfmc_variance_ordered(matrix, order, counts, coefficients),
        mc_variance=matrix[0][0] * weights[0] / budget,
        cost=math.fsum(weights[i] * m for i, m in zip(order, counts, strict=True)),
        budget=budget,
        evaluations=tuple(evaluations),
    )


def mfmc_allocation(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    budget: float,
    *,
    models: Sequence[int] | None = None,
) -> MFMCAllocation:
    """Return the closed-form optimal MFMC coefficients and sample sizes.

    With the selected models ordered by decreasing ``|rho_0i|`` and
    ``rho_q = 0`` after the last one,

    ``a_i = rho_0i s_0 / s_i``,
    ``r_i = sqrt(w_0 (rho_i^2 - rho_(i+1)^2) / (w_i (1 - rho_1^2)))``,
    ``m_0 = p / sum_i w_i r_i``, ``m_i = r_i m_0``,

    which minimizes the estimator variance subject to
    ``sum_i w_i m_i = p`` and ``m_0 <= m_1 <= ...``. The minimum is
    ``C_00 (sum_i sqrt(w_i (rho_i^2 - rho_(i+1)^2)))^2 / p``. Integer sample
    sizes are the floors of ``m_i``, so the integer cost never exceeds the
    budget and ``variance`` (at the integers) is slightly above
    ``continuous_variance``.

    The covariance may come from :func:`pilot_statistics`; it must be
    independent of the estimation samples drawn by :func:`mfmc_estimate`.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        costs: Positive cost of one evaluation of every model.
        budget: Positive cost budget ``p``.
        models: Model indices to use, starting with 0; default all. They are
            reordered by decreasing ``|rho_0i|``.

    Returns:
        Allocation with coefficients, real and integer sample sizes, and
        predicted variances.

    Raises:
        ValueError: If the selected models violate the admissibility
            conditions [EQ-MC-MF-004] (use :func:`select_mfmc_models`), the
            budget does not pay for one sample of each selected model, or an
            input is invalid.

    Equation:
        [EQ-MC-MF-003], [EQ-MC-MF-004].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    matrix = _as_covariance(covariance)
    weights = _as_costs(costs, len(matrix))
    spend = _as_budget(budget)
    order = _mfmc_order(matrix, _as_model_subset(models, len(matrix)))
    rho = _correlations_with_first(matrix)
    violation = _mfmc_violation([rho[i] for i in order], [weights[i] for i in order])
    if violation is not None:
        raise ValueError(
            f"models {order} are not admissible for the MFMC allocation: "
            f"{violation}; select a subset with select_mfmc_models"
        )
    return _mfmc_plan(matrix, weights, spend, order)


def select_mfmc_models(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    budget: float,
    *,
    max_models: int | None = None,
) -> MFMCAllocation:
    """Select the admissible model subset with the smallest MFMC variance.

    Every subset that contains the high-fidelity model is ordered by
    decreasing ``|rho_0i|`` and tested against the admissibility conditions
    [EQ-MC-MF-004]; among the admissible subsets whose optimal allocation
    affords one high-fidelity sample, the one with the smallest closed-form
    variance ``C_00 (sum_i sqrt(w_i (rho_i^2 - rho_(i+1)^2)))^2 / p`` is
    returned (ties prefer fewer models). The search is exhaustive,
    ``2^(k-1)`` subsets, and limited to 12 models.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        costs: Positive cost of one evaluation of every model.
        budget: Positive cost budget ``p``.
        max_models: Optional largest subset size (including model 0).

    Returns:
        Allocation of the best admissible subset. Model 0 alone (plain
        Monte Carlo) is always admissible.

    Raises:
        ValueError: If there are more than 12 models, the budget is below
            the cost of one high-fidelity evaluation, or an input is invalid.

    Equation:
        [EQ-MC-MF-003], [EQ-MC-MF-004].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    matrix = _as_covariance(covariance)
    size = len(matrix)
    weights = _as_costs(costs, size)
    spend = _as_budget(budget)
    if size > _MAX_EXHAUSTIVE_MODELS:
        raise ValueError(
            f"exhaustive selection supports at most {_MAX_EXHAUSTIVE_MODELS} models"
        )
    limit = (
        size
        if max_models is None
        else _as_count(max_models, name="max_models", minimum=1)
    )
    if spend < weights[0]:
        raise ValueError("budget is below the cost of one high-fidelity evaluation")
    rho = _correlations_with_first(matrix)
    best: MFMCAllocation | None = None
    for extra in range(min(limit, size)):
        for chosen in itertools.combinations(range(1, size), extra):
            order = _mfmc_order(matrix, (0, *chosen))
            if _mfmc_violation([rho[i] for i in order], [weights[i] for i in order]):
                continue
            try:
                plan = _mfmc_plan(matrix, weights, spend, order)
            except ValueError:
                continue
            if best is None or plan.continuous_variance < best.continuous_variance * (
                1.0 - 1e-12
            ):
                best = plan
    if best is None:  # pragma: no cover - model 0 alone always qualifies
        raise ValueError("no admissible model subset fits the budget")
    return best


def mfmc_estimate(
    models: Sequence[ModelFunction],
    input_sampler: InputSampler,
    allocation: MFMCAllocation,
    *,
    costs: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
) -> MultifidelityEstimate:
    """Estimate the high-fidelity mean by multifidelity Monte Carlo.

    One input sample of size ``m_(q-1)`` is drawn from
    ``root.child("mfmc")``; the model at position ``j`` of the allocation is
    evaluated on its first ``m_j`` inputs, and

    ``s = ybar^0_(m_0) + sum_(j>=1) a_j (ybar^j_(m_j) - ybar^j_(m_(j-1)))``.

    The estimator is unbiased for ``E[f_0]`` for any fixed coefficients. If
    the coefficients come from a pilot sample, that pilot must be independent
    of the inputs drawn here (a pilot from :func:`pilot_statistics` with the
    same root seed is, because it uses another child stream). The reported
    variance is the allocation's prediction.

    Args:
        models: All model callables, indexed as in the covariance matrix
            that produced ``allocation``.
        input_sampler: ``input_sampler(n, stream)`` returning ``n``
            independent inputs.
        allocation: Result of :func:`mfmc_allocation` or
            :func:`select_mfmc_models`.
        costs: Optional costs of every model; when given, the result reports
            the cost of the evaluations made.
        seed: Root seed or stream.

    Returns:
        Estimate with predicted variance and evaluation accounting.

    Raises:
        ValueError: If the allocation refers to a model that was not
            supplied, the sampler returns the wrong number of inputs, or an
            output is not finite.

    Equation:
        [EQ-MC-MF-002].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016, Section 3].

    """
    functions = tuple(models)
    if not isinstance(allocation, MFMCAllocation):
        raise ValueError("allocation must be an MFMCAllocation")
    if max(allocation.models) >= len(functions):
        raise ValueError("allocation refers to a model that was not supplied")
    weights = None if costs is None else _as_costs(costs, len(functions))
    inputs = _draw_inputs(
        input_sampler, allocation.samples[-1], resolve_stream(seed).child("mfmc")
    )
    evaluations = [0] * len(functions)
    estimate = 0.0
    high_fidelity_mean = 0.0
    for position, index in enumerate(allocation.models):
        count = allocation.samples[position]
        values = _evaluate(functions[index], inputs[:count], index)
        evaluations[index] = count
        if position == 0:
            high_fidelity_mean = _mean(values)
            estimate = high_fidelity_mean
        else:
            previous = allocation.samples[position - 1]
            estimate += allocation.coefficients[position] * (
                _mean(values) - _mean(values[:previous])
            )
    return MultifidelityEstimate(
        method="MFMC",
        estimate=estimate,
        variance=allocation.variance,
        high_fidelity_mean=high_fidelity_mean,
        high_fidelity_samples=allocation.samples[0],
        evaluations=tuple(evaluations),
        cost=None
        if weights is None
        else math.fsum(w * n for w, n in zip(weights, evaluations, strict=True)),
    )
