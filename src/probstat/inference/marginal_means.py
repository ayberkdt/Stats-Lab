"""Estimated marginal means and Tukey-adjusted pairwise contrasts.

After an ANOVA says a factor matters, the next questions are *which* levels
differ and by how much. Raw marginal means answer them only for balanced
data: with unequal cell sizes they weight the levels of the other factors
by their counts. Estimated marginal means (least-squares means) average the
model's predicted cell means with equal weights over the other factors
[EQ-LM-EMM-001]. Each is a linear function ``L beta`` of the coefficients,
so its standard error is ``sqrt(L Cov(beta) L')``; pairwise differences are
compared with the studentized range (Tukey-Kramer), which controls the
family-wise error of all pairs [EQ-LM-EMM-002].
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from probstat.linalg import as_vector

from .bayesian import student_t_quantile
from .linear_models import LinearModelResult, fit_linear_model
from .model_comparison import Coding, factorial_design
from .studentized_range import studentized_range_quantile, studentized_range_sf

Vector = tuple[float, ...]


@dataclass(frozen=True)
class MarginalMean:
    """Estimated marginal mean of one level.

    Attributes:
        level: The level (as text).
        estimate: Equal-weight average of the predicted cell means.
        standard_error: ``sqrt(L Cov(beta) L')``.
        lower: Lower bound of the pointwise t interval.
        upper: Upper bound of the pointwise t interval.

    References:
        [REF-SEARLE-SPEED-MILLIKEN-1980].

    """

    level: str
    estimate: float
    standard_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class MarginalContrast:
    """Difference of two estimated marginal means with a Tukey adjustment.

    Attributes:
        first: First level.
        second: Second level.
        difference: ``mean(first) - mean(second)``.
        standard_error: Standard error of the difference.
        statistic: ``difference / standard_error`` (a t ratio).
        p_value: Tukey-Kramer adjusted p-value,
            ``P(Q_(k, df) >= sqrt(2) |t|)``.
        lower: Lower simultaneous bound, ``difference - q SE / sqrt(2)``.
        upper: Upper simultaneous bound.

    References:
        [REF-HOCHBERG-TAMHANE-1987], [REF-KRAMER-1956].

    """

    first: str
    second: str
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class MarginalMeans:
    """Estimated marginal means of one factor and their pairwise contrasts.

    Attributes:
        factor: The factor whose levels are compared.
        at: Fixed levels of other factors (simple effects), or empty.
        means: One entry per level in sorted order.
        contrasts: One entry per pair of levels.
        df: Residual degrees of freedom of the model.
        confidence: Level of the intervals.
        fit: The fitted linear model.

    References:
        [REF-SEARLE-SPEED-MILLIKEN-1980], [REF-HOCHBERG-TAMHANE-1987].

    """

    factor: str
    at: tuple[tuple[str, str], ...]
    means: tuple[MarginalMean, ...]
    contrasts: tuple[MarginalContrast, ...]
    df: int
    confidence: float
    fit: LinearModelResult

    def mean(self, level: object) -> MarginalMean:
        """Return the marginal mean of one level.

        Raises:
            KeyError: If the level does not occur.

        References:
            [REF-SEARLE-SPEED-MILLIKEN-1980].

        """
        for entry in self.means:
            if entry.level == str(level):
                return entry
        raise KeyError(level)

    def contrast(self, first: object, second: object) -> MarginalContrast:
        """Return the contrast ``first - second``.

        Raises:
            KeyError: If the pair does not occur.

        References:
            [REF-HOCHBERG-TAMHANE-1987].

        """
        a, b = str(first), str(second)
        for entry in self.contrasts:
            if (entry.first, entry.second) == (a, b):
                return entry
            if (entry.first, entry.second) == (b, a):
                return MarginalContrast(
                    first=a,
                    second=b,
                    difference=-entry.difference,
                    standard_error=entry.standard_error,
                    statistic=-entry.statistic,
                    p_value=entry.p_value,
                    lower=-entry.upper,
                    upper=-entry.lower,
                )
        raise KeyError((first, second))


def _quadratic(
    weights: Sequence[float], covariance: Sequence[Sequence[float]]
) -> float:
    """Return ``L C L'`` for a weight vector ``L``.

    References:
        [REF-SEARLE-SPEED-MILLIKEN-1980].

    """
    return math.fsum(
        weights[i] * covariance[i][j] * weights[j]
        for i in range(len(weights))
        for j in range(len(weights))
    )


def marginal_means(
    factors: Mapping[str, Sequence[object]],
    response: Iterable[object],
    factor: str,
    *,
    at: Mapping[str, object] | None = None,
    interactions: bool | Iterable[str] = True,
    coding: Coding = "sum",
    confidence: float = 0.95,
) -> MarginalMeans:
    """Return estimated marginal means of ``factor`` and Tukey contrasts.

    The model is the factorial linear model of :func:`factorial_design`.
    The reference grid holds every combination of factor levels; the
    marginal mean of level ``a`` averages the predicted means of the grid
    cells with ``factor = a`` (and the ``at`` levels), giving each
    combination of the remaining factors equal weight. In a balanced design
    these are the raw marginal means; with a full interaction model they
    are the unweighted averages of the cell means.

    Args:
        factors: Ordered mapping from factor name to level sequences.
        response: Responses aligned with the factor sequences.
        factor: Factor whose levels are compared.
        at: Levels at which other factors are held (simple effects); for
            example ``{"temperature": 70}``.
        interactions: As in :func:`factorial_design`.
        coding: As in :func:`factorial_design`; the means do not depend on
            it.
        confidence: Level of the pointwise intervals of the means and the
            simultaneous intervals of the contrasts.

    Returns:
        Means, contrasts, residual degrees of freedom, and the fit.

    Raises:
        KeyError: If ``factor`` or an ``at`` factor is unknown, or an ``at``
            level does not occur.
        ValueError: If ``factor`` is also in ``at``, the confidence is not in
            ``(0, 1)``, or the model leaves no residual degrees of freedom.

    Equation:
        [EQ-LM-EMM-001], [EQ-LM-EMM-002].

    References:
        [REF-SEARLE-SPEED-MILLIKEN-1980], [REF-HOCHBERG-TAMHANE-1987],
        [REF-KRAMER-1956], [REF-MONTGOMERY-2017, Chapter 5].

    """
    if factor not in factors:
        raise KeyError(factor)
    fixed = {str(k): str(v) for k, v in (at or {}).items()}
    if factor in fixed:
        raise ValueError("the compared factor cannot also be held fixed")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    names = list(factors)
    levels = {name: sorted({str(v) for v in factors[name]}) for name in names}
    for name, value in fixed.items():
        if name not in levels:
            raise KeyError(name)
        if value not in levels[name]:
            raise KeyError(f"level {value!r} of {name!r} does not occur")
    y = as_vector(response, name="response")
    n = len(y)
    grid = list(itertools.product(*(levels[name] for name in names)))
    combined = {
        name: [str(v) for v in factors[name]] + [cell[i] for cell in grid]
        for i, name in enumerate(names)
    }
    design, _ = factorial_design(combined, interactions=interactions, coding=coding)
    observed, grid_rows = design.rows[:n], design.rows[n:]
    fit = fit_linear_model(observed, y)
    if fit.df_residual <= 0:
        raise ValueError("the model leaves no residual degrees of freedom")
    position = names.index(factor)
    t = student_t_quantile(0.5 + 0.5 * confidence, fit.df_residual)
    weights: dict[str, Vector] = {}
    means: list[MarginalMean] = []
    for level in levels[factor]:
        cells = [
            row
            for cell, row in zip(grid, grid_rows, strict=True)
            if cell[position] == level
            and all(cell[names.index(k)] == v for k, v in fixed.items())
        ]
        vector = tuple(
            math.fsum(row[j] for row in cells) / len(cells)
            for j in range(len(grid_rows[0]))
        )
        weights[level] = vector
        estimate = math.fsum(
            w * b for w, b in zip(vector, fit.coefficients, strict=True)
        )
        error = math.sqrt(max(_quadratic(vector, fit.covariance), 0.0))
        means.append(
            MarginalMean(
                level, estimate, error, estimate - t * error, estimate + t * error
            )
        )
    k = len(levels[factor])
    q = studentized_range_quantile(confidence, k, fit.df_residual)
    contrasts: list[MarginalContrast] = []
    for first, second in itertools.combinations(levels[factor], 2):
        vector = tuple(
            a - b for a, b in zip(weights[first], weights[second], strict=True)
        )
        difference = math.fsum(
            w * b for w, b in zip(vector, fit.coefficients, strict=True)
        )
        error = math.sqrt(max(_quadratic(vector, fit.covariance), 0.0))
        statistic = (
            difference / error if error > 0.0 else math.copysign(math.inf, difference)
        )
        half = q * error / math.sqrt(2.0)
        contrasts.append(
            MarginalContrast(
                first=first,
                second=second,
                difference=difference,
                standard_error=error,
                statistic=statistic,
                p_value=studentized_range_sf(
                    math.sqrt(2.0) * abs(statistic), k, fit.df_residual
                ),
                lower=difference - half,
                upper=difference + half,
            )
        )
    return MarginalMeans(
        factor=factor,
        at=tuple(sorted(fixed.items())),
        means=tuple(means),
        contrasts=tuple(contrasts),
        df=fit.df_residual,
        confidence=confidence,
        fit=fit,
    )
