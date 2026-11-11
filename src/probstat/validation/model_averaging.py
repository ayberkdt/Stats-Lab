"""Model-form uncertainty across competing models: stacking and pseudo-BMA.

When several simulators or statistical models of the same quantity are
credible, choosing one discards the model-form uncertainty between them.
This module combines their *predictive distributions* with weights learned
from out-of-sample predictive performance:

* **stacking** maximizes the mean log score of the weighted mixture,
  ``max_w (1/n) sum_i log sum_k w_k p_k(y_i)`` over the simplex, where
  ``p_k(y_i)`` is model ``k``'s leave-one-out (or held-out validation)
  predictive density at observation ``i``;
* **pseudo-BMA** weights models by ``exp(elpd_k)``, the exponentiated
  summed log predictive density, and **pseudo-BMA+** averages those weights
  over Bayesian-bootstrap resamples of the pointwise values, which
  regularizes them away from 0 and 1.

Stacking optimizes the combined predictive distribution and keeps several
models when each is right somewhere (the M-open case); pseudo-BMA, like BMA,
concentrates on the single best model as ``n`` grows.
:func:`model_averaged_prediction` then splits the variance of the mixture
into the expected within-model variance and the between-model variance, the
latter being the model-form component.

Inputs are pointwise log predictive densities ``lpd[i][k]`` (rows are
observations, columns models), for example ``law_k.log_pdf(y_i)`` for a
simulator's predictive law at validation experiment ``i``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class ModelWeights:
    """Simplex weights over competing models and how they were obtained.

    Args:
        weights: Non-negative weights summing to one, one per model.
        method: ``"stacking"``, ``"pseudo-bma"``, or ``"pseudo-bma+"``.
        log_score: Mean log predictive density of the weighted mixture on
            the supplied points, ``(1/n) sum_i log sum_k w_k p_k(y_i)``.
        elpd: Summed pointwise log predictive density of each model.
        iterations: Fixed-point iterations (stacking) or bootstrap
            replicates (pseudo-BMA+); ``0`` for plain pseudo-BMA.
        optimality_gap: Upper bound on the log-score shortfall from the
            stacking optimum, ``max_k g_k - 1`` (``0`` when not applicable).

    References:
        [EQ-VAL-STACK-001], [EQ-VAL-STACK-002], [REF-YAO-ETAL-2018].

    """

    weights: tuple[float, ...]
    method: str
    log_score: float
    elpd: tuple[float, ...]
    iterations: int
    optimality_gap: float


@dataclass(frozen=True, slots=True)
class ModelAveragedPrediction:
    """Mean and variance decomposition of a weighted mixture of predictions.

    Args:
        mean: Mixture mean ``sum_k w_k mu_k``.
        within_model_variance: ``sum_k w_k sigma_k^2``.
        between_model_variance: ``sum_k w_k (mu_k - mean)^2``, the
            model-form component.
        total_variance: Sum of the two components.

    References:
        [EQ-VAL-STACK-003], [REF-YAO-ETAL-2018, Equation 2.3].

    """

    mean: float
    within_model_variance: float
    between_model_variance: float
    total_variance: float

    @property
    def model_form_share(self) -> float:
        """Return the between-model share of the total variance.

        Returns:
            ``between / total``, or ``0`` when the total variance is zero.

        References:
            [EQ-VAL-STACK-003].

        """
        if self.total_variance == 0.0:
            return 0.0
        return self.between_model_variance / self.total_variance


def _log_density_matrix(
    log_densities: Sequence[Sequence[object]],
) -> tuple[tuple[float, ...], ...]:
    """Validate a rectangular matrix with at least one row and two models.

    Entries may be ``-inf`` (zero density) but not ``nan`` or ``+inf``; every
    row needs at least one finite entry.

    References:
        [REF-YAO-ETAL-2018, Section 2.1]. No equation is evaluated.

    """
    rows: list[tuple[float, ...]] = []
    for i, row in enumerate(log_densities):
        values: list[float] = []
        for k, value in enumerate(row):
            if isinstance(value, (int, float)) and value == -math.inf:
                values.append(-math.inf)
            else:
                values.append(
                    require_finite_real(value, name=f"log_densities[{i}][{k}]")
                )
        if not any(math.isfinite(value) for value in values):
            raise ValueError(f"row {i} gives zero density under every model")
        rows.append(tuple(values))
    if not rows:
        raise ValueError("log_densities must contain at least one observation")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise ValueError("log_densities must be rectangular with at least two models")
    return tuple(rows)


def _row_scaled(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    """Return ``exp(lpd[i][k] - max_k lpd[i][k])`` row by row.

    Rescaling a row by a positive constant leaves the stacking gradient
    ratios and optimum unchanged.

    References:
        [EQ-VAL-STACK-001]. No new equation is introduced.

    """
    scaled: list[list[float]] = []
    for row in matrix:
        peak = max(row)
        scaled.append([math.exp(value - peak) for value in row])
    return scaled


def _mixture_log_score(
    matrix: tuple[tuple[float, ...], ...], weights: Sequence[float]
) -> float:
    """Return ``(1/n) sum_i log sum_k w_k exp(lpd[i][k])`` stably.

    References:
        [EQ-VAL-STACK-001], [REF-YAO-ETAL-2018, Section 2.1].

    """
    total: list[float] = []
    for row in matrix:
        peak = max(row)
        mixture = math.fsum(
            w * math.exp(value - peak) for w, value in zip(weights, row, strict=True)
        )
        total.append(peak + math.log(mixture) if mixture > 0.0 else -math.inf)
    return math.fsum(total) / len(total)


def _elpd(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Return the column sums of the pointwise log densities.

    References:
        [EQ-VAL-STACK-002], [REF-YAO-ETAL-2018, Section 2.4].

    """
    return tuple(
        math.fsum(row[k] for row in matrix) for k in range(len(matrix[0]))
    )


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    """Return ``exp(v_k) / sum_j exp(v_j)`` with ``-inf`` entries mapped to 0.

    References:
        [EQ-VAL-STACK-002]. No new equation is introduced.

    """
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = math.fsum(exps)
    return tuple(value / total for value in exps)


def stacking_weights(
    log_densities: Sequence[Sequence[object]],
    *,
    max_iterations: int = 10000,
    tolerance: float = 1e-9,
) -> ModelWeights:
    """Return log-score stacking weights of predictive distributions.

    The objective ``f(w) = (1/n) sum_i log sum_k w_k p_ik`` is concave on
    the simplex. It is maximized by the expectation-maximization update for
    mixture proportions, ``w_k <- w_k g_k`` with
    ``g_k = (1/n) sum_i p_ik / sum_j w_j p_ij``, which keeps ``w`` on the
    simplex and increases ``f`` monotonically. Concavity gives
    ``f(w*) - f(w) <= max_k g_k - 1``, so iteration stops when that bound
    falls below ``tolerance``.

    Args:
        log_densities: ``lpd[i][k]``, the out-of-sample log predictive
            density of model ``k`` at observation ``i`` (``-inf`` allowed
            where a model gives zero density).
        max_iterations: Positive iteration cap.
        tolerance: Positive bound on the log-score shortfall.

    Returns:
        Weights, achieved mean log score, per-model elpd, iteration count,
        and the final optimality-gap bound.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is malformed, a row has zero density under
            every model, or a setting is invalid.

    Equation:
        [EQ-VAL-STACK-001].

    References:
        [REF-YAO-ETAL-2018, Section 2.1], [REF-MCLACHLAN-PEEL-2000],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 12].

    """
    matrix = _log_density_matrix(log_densities)
    cap = require_non_negative_integer(max_iterations, name="max_iterations")
    if cap == 0:
        raise ValueError("max_iterations must be positive")
    tol = require_non_negative_finite_real(tolerance, name="tolerance")
    if tol == 0.0:
        raise ValueError("tolerance must be positive")
    scaled = _row_scaled(matrix)
    n, models = len(scaled), len(scaled[0])
    weights = [1.0 / models] * models
    gap = math.inf
    iterations = 0
    while iterations < cap:
        mixtures = [
            math.fsum(w * p for w, p in zip(weights, row, strict=True))
            for row in scaled
        ]
        gradient = [
            math.fsum(
                row[k] / mixture
                for row, mixture in zip(scaled, mixtures, strict=True)
            )
            / n
            for k in range(models)
        ]
        gap = max(gradient) - 1.0
        if gap <= tol:
            break
        weights = [w * g for w, g in zip(weights, gradient, strict=True)]
        total = math.fsum(weights)
        weights = [w / total for w in weights]
        iterations += 1
    return ModelWeights(
        weights=tuple(weights),
        method="stacking",
        log_score=_mixture_log_score(matrix, weights),
        elpd=_elpd(matrix),
        iterations=iterations,
        optimality_gap=max(gap, 0.0),
    )


def pseudo_bma_weights(
    log_densities: Sequence[Sequence[object]],
    *,
    n_bootstrap: int = 0,
    seed: int | RandomStream | None = None,
) -> ModelWeights:
    """Return pseudo-BMA weights, or pseudo-BMA+ with a Bayesian bootstrap.

    Pseudo-BMA: ``w_k = exp(elpd_k) / sum_j exp(elpd_j)``. Pseudo-BMA+:
    draw ``a ~ Dirichlet(1, ..., 1)`` over the ``n`` observations, set
    ``z_k = n sum_i a_i lpd[i][k]``, take the softmax of ``z``, and average
    over ``n_bootstrap`` draws.

    Args:
        log_densities: Pointwise out-of-sample log predictive densities.
        n_bootstrap: ``0`` for pseudo-BMA, otherwise the number of Bayesian
            bootstrap draws for pseudo-BMA+.
        seed: Integer seed or stream for the bootstrap; ``None`` is not
            reproducible.

    Returns:
        Weights with the mixture log score and per-model elpd.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is malformed or ``n_bootstrap`` is negative.

    Equation:
        [EQ-VAL-STACK-002].

    References:
        [REF-YAO-ETAL-2018, Section 2.4, Equations 2.5 and 2.6].

    """
    matrix = _log_density_matrix(log_densities)
    draws = require_non_negative_integer(n_bootstrap, name="n_bootstrap")
    elpd = _elpd(matrix)
    if draws == 0:
        weights = _softmax(elpd)
        method = "pseudo-bma"
    else:
        rng = resolve_stream(seed).child("pseudo-bma+").generator()
        n, models = len(matrix), len(matrix[0])
        sums = [0.0] * models
        for _ in range(draws):
            gaps = [rng.expovariate(1.0) for _ in range(n)]
            total = math.fsum(gaps)
            z = [
                n
                * math.fsum(
                    g / total * row[k] for g, row in zip(gaps, matrix, strict=True)
                )
                for k in range(models)
            ]
            for k, weight in enumerate(_softmax(z)):
                sums[k] += weight
        weights = tuple(value / draws for value in sums)
        method = "pseudo-bma+"
    return ModelWeights(
        weights=tuple(weights),
        method=method,
        log_score=_mixture_log_score(matrix, weights),
        elpd=elpd,
        iterations=draws,
        optimality_gap=0.0,
    )


def model_averaged_prediction(
    weights: Sequence[object] | ModelWeights,
    means: Sequence[object],
    variances: Sequence[object],
) -> ModelAveragedPrediction:
    """Combine per-model predictive means and variances by mixture weights.

    Args:
        weights: Simplex weights (or a :class:`ModelWeights`).
        means: Predictive mean ``mu_k`` of each model.
        variances: Non-negative predictive variance ``sigma_k^2`` of each
            model.

    Returns:
        Mixture mean and the within/between variance decomposition.

    Raises:
        TypeError: If a value is not real.
        ValueError: If lengths differ, a weight or variance is negative, or
            the weights do not sum to one within ``1e-9``.

    Equation:
        [EQ-VAL-STACK-003].

    References:
        [REF-YAO-ETAL-2018, Equation 2.3].

    """
    raw = weights.weights if isinstance(weights, ModelWeights) else weights
    w = [
        require_non_negative_finite_real(v, name=f"weights[{k}]")
        for k, v in enumerate(raw)
    ]
    mu = [require_finite_real(v, name=f"means[{k}]") for k, v in enumerate(means)]
    var = [
        require_non_negative_finite_real(v, name=f"variances[{k}]")
        for k, v in enumerate(variances)
    ]
    if not w or len(w) != len(mu) or len(w) != len(var):
        raise ValueError(
            "weights, means, and variances must have equal non-zero length"
        )
    if abs(math.fsum(w) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one")
    mean = math.fsum(wk * mk for wk, mk in zip(w, mu, strict=True))
    within = math.fsum(wk * vk for wk, vk in zip(w, var, strict=True))
    between = math.fsum(wk * (mk - mean) ** 2 for wk, mk in zip(w, mu, strict=True))
    return ModelAveragedPrediction(
        mean=mean,
        within_model_variance=within,
        between_model_variance=between,
        total_variance=within + between,
    )
