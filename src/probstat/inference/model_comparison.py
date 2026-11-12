"""Comparing linear models: nested F tests, ANOVA tables, residual checks.

A linear model is judged term by term through the drop in residual sum of
squares when the term's columns join the model [EQ-LM-ANOVA-001]. The three
classical tables differ only in which model the term joins:

- **Type I (sequential):** the model of all earlier terms. The sums of
  squares add up to the model sum of squares, but depend on term order.
- **Type II (marginal):** the model of every term that does not contain it
  (``A`` is tested after ``B`` but not after ``A:B``). This respects
  marginality and is the usual choice without important interactions.
- **Type III (partial):** the full model minus the term. Meaningful for
  main effects in the presence of interactions only with sum-to-zero
  (effect) coding, which :func:`factorial_design` uses by default.

In a balanced design with sum-to-zero coding the three coincide. The
residual checks test two assumptions behind the F tests: constant variance
(Breusch-Pagan, [EQ-LM-BP-001]) and uncorrelated errors in a natural order
(Durbin-Watson, [EQ-LM-DW-001]).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.linalg import as_vector

from .linear_models import DesignMatrix, LinearModelResult, fit_linear_model
from .sampling_distributions import chi_square_sf, f_sf

AnovaKind = Literal["I", "II", "III"]
Coding = Literal["sum", "treatment"]
_NESTING_TOLERANCE = 1e-8


@dataclass(frozen=True)
class NestedFTest:
    """F test of a reduced linear model against a larger one that nests it.

    Attributes:
        statistic: ``((RSS_r - RSS_f) / q) / (RSS_f / df_f)``.
        p_value: Upper-tail ``F(q, df_f)`` probability.
        numerator_df: ``q = df_r - df_f``, the number of dropped constraints.
        denominator_df: Residual degrees of freedom of the full model.
        sum_squares: ``RSS_r - RSS_f``, the sum of squares the extra
            columns explain.

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

    """

    statistic: float
    p_value: float
    numerator_df: int
    denominator_df: int
    sum_squares: float


@dataclass(frozen=True)
class AnovaRow:
    """One term of an ANOVA table.

    Attributes:
        term: Term name (``"A"``, ``"B"``, ``"A:B"``, or ``"Residuals"``).
        df: Degrees of freedom (number of columns of the term).
        sum_squares: Sum of squares of the requested type.
        mean_square: ``sum_squares / df``.
        f_statistic: ``mean_square / MSE``; ``nan`` for the residual row.
        p_value: Upper-tail F probability; ``nan`` for the residual row.

    References:
        [REF-MONTGOMERY-2017, Chapter 5].

    """

    term: str
    df: int
    sum_squares: float
    mean_square: float
    f_statistic: float
    p_value: float


@dataclass(frozen=True)
class AnovaTable:
    """ANOVA table of a linear model.

    Attributes:
        kind: ``"I"``, ``"II"``, or ``"III"``.
        rows: One row per term in the order given, then ``"Residuals"``.
        fit: The full model.

    References:
        [REF-MONTGOMERY-2017, Chapter 5], [REF-FOX-WEISBERG-2019].

    """

    kind: AnovaKind
    rows: tuple[AnovaRow, ...]
    fit: LinearModelResult

    def row(self, term: str) -> AnovaRow:
        """Return the row of one term.

        Raises:
            KeyError: If the term is not in the table.

        References:
            [REF-MONTGOMERY-2017, Chapter 5].

        """
        for entry in self.rows:
            if entry.term == term:
                return entry
        raise KeyError(term)


@dataclass(frozen=True)
class BreuschPaganTest:
    """Breusch-Pagan test of constant error variance.

    Attributes:
        statistic: Lagrange-multiplier statistic.
        p_value: Upper-tail chi-square probability.
        df: Number of non-constant columns of the variance model.
        studentized: Whether Koenker's studentized form was used.

    References:
        [REF-BREUSCH-PAGAN-1979], [REF-KOENKER-1981].

    """

    statistic: float
    p_value: float
    df: int
    studentized: bool


def _same_data(first: LinearModelResult, second: LinearModelResult) -> bool:
    """Return whether two fits used the same response and weights.

    References:
        [REF-PROBSTAT-VALIDATION].

    """
    if first.n_observations != second.n_observations:
        return False
    responses = (
        tuple(f + e for f, e in zip(fit.fitted_values, fit.residuals, strict=True))
        for fit in (first, second)
    )
    y_first, y_second = responses
    scale = max(1.0, max(abs(v) for v in y_first))
    return (
        all(abs(a - b) <= 1e-9 * scale for a, b in zip(y_first, y_second, strict=True))
        and first.weights == second.weights
    )


def _spans(full: LinearModelResult, reduced: LinearModelResult) -> bool:
    """Return whether the full design spans every column of the reduced one.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    columns = len(reduced.design[0])
    for j in range(columns):
        target = [row[j] for row in reduced.design]
        size = max(1.0, max(abs(v) for v in target))
        if all(v == 0.0 for v in target):
            continue
        projection = fit_linear_model(full.design, target, weights=full.weights)
        if max(abs(e) for e in projection.residuals) > _NESTING_TOLERANCE * size:
            return False
    return True


def nested_f_test(reduced: LinearModelResult, full: LinearModelResult) -> NestedFTest:
    """Test whether the extra columns of ``full`` explain significant variance.

    Under normal errors and the reduced model,
    ``F = ((RSS_r - RSS_f) / (df_r - df_f)) / (RSS_f / df_f)`` follows
    ``F(df_r - df_f, df_f)``. Dropping one column gives ``F = t^2`` of that
    coefficient; dropping all non-constant columns gives the overall F test.

    Args:
        reduced: Fit of the smaller model.
        full: Fit of the larger model on the same response and weights.

    Returns:
        The F statistic, its p-value, and degrees of freedom.

    Raises:
        ValueError: If the fits use different data, the full model does not
            have more columns, or its column space does not contain the
            reduced one.

    Equation:
        [EQ-LM-ANOVA-001].

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

    """
    if not _same_data(reduced, full):
        raise ValueError("the two fits must use the same response and weights")
    q = reduced.df_residual - full.df_residual
    if q <= 0:
        raise ValueError("the full model must have more columns than the reduced")
    if not _spans(full, reduced):
        raise ValueError("the reduced model is not nested in the full model")
    extra = max(reduced.residual_sum_squares - full.residual_sum_squares, 0.0)
    denominator = full.residual_sum_squares / full.df_residual
    if denominator <= 0.0:
        raise ValueError("the full model fits exactly; the F test is undefined")
    statistic = (extra / q) / denominator
    return NestedFTest(
        statistic=statistic,
        p_value=f_sf(statistic, q, full.df_residual),
        numerator_df=q,
        denominator_df=full.df_residual,
        sum_squares=extra,
    )


def _factors_of(term: str) -> frozenset[str]:
    """Return the factor names of ``"A:B"``-style term names.

    References:
        [REF-FOX-WEISBERG-2019].

    """
    return frozenset(part.strip() for part in term.split(":"))


def linear_model_anova(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    terms: Mapping[str, Sequence[int]],
    *,
    kind: AnovaKind = "II",
    weights: Iterable[object] | None = None,
) -> AnovaTable:
    """Return the type I, II, or III ANOVA table of a linear model.

    ``terms`` maps each term name to its design columns; columns in no term
    (usually the intercept) are always kept. Interaction terms are named by
    joining factor names with ``":"``; term ``T'`` contains ``T`` when the
    factors of ``T`` are a proper subset of those of ``T'``. With
    ``RSS(S)`` the residual sum of squares of the model with the columns of
    the term set ``S`` (plus the unassigned columns):

    - type I: ``SS(T_k) = RSS(T_1..T_(k-1)) - RSS(T_1..T_k)``;
    - type II: ``SS(T) = RSS(R) - RSS(R + T)`` with ``R`` every term that
      does not contain ``T``;
    - type III: ``SS(T) = RSS(all - T) - RSS(all)``.

    Each ``F = (SS(T) / df_T) / MSE`` with ``MSE`` from the full model.

    Args:
        design: Full design (rows or a :class:`DesignMatrix`).
        response: Responses.
        terms: Ordered mapping from term name to column indices.
        kind: ``"I"``, ``"II"``, or ``"III"``.
        weights: Optional case weights.

    Returns:
        The table with one row per term and the residual row.

    Raises:
        ValueError: If ``kind`` is unknown, a column index is out of range or
            used by two terms, a term has no columns, or the full model
            leaves no residual degrees of freedom.

    Equation:
        [EQ-LM-ANOVA-001], [EQ-LM-ANOVA-002].

    References:
        [REF-MONTGOMERY-2017, Chapter 5], [REF-FOX-WEISBERG-2019],
        [REF-SEARLE-CASELLA-MCCULLOCH-1992].

    """
    if kind not in ("I", "II", "III"):
        raise ValueError("kind must be 'I', 'II', or 'III'")
    source = design.rows if isinstance(design, DesignMatrix) else design
    rows = tuple(tuple(row) for row in source)
    y = as_vector(response, name="response")
    weight_values = None if weights is None else as_vector(weights, name="weights")
    width = len(rows[0]) if rows else 0
    assigned: dict[int, str] = {}
    for name, columns in terms.items():
        if not columns:
            raise ValueError(f"term {name!r} has no columns")
        for column in columns:
            if not 0 <= column < width:
                raise ValueError(f"term {name!r} uses column {column} out of range")
            if column in assigned:
                raise ValueError(
                    f"column {column} is used by {assigned[column]!r} and {name!r}"
                )
            assigned[column] = name
    fixed = [j for j in range(width) if j not in assigned]
    names = list(terms)
    cache: dict[frozenset[str], float] = {}

    def rss(selected: frozenset[str]) -> float:
        """Return the residual sum of squares of the model with ``selected``.

        Equation:
            [EQ-LM-ANOVA-002].

        References:
            [REF-MONTGOMERY-2017, Chapter 5].

        """
        if selected not in cache:
            columns = sorted(fixed + [j for t in selected for j in terms[t]])
            if not columns:
                scale = weight_values or (1.0,) * len(y)
                cache[selected] = math.fsum(
                    w * value * value for w, value in zip(scale, y, strict=True)
                )
            else:
                sub = [tuple(row[j] for j in columns) for row in rows]
                cache[selected] = fit_linear_model(
                    sub, y, weights=weight_values
                ).residual_sum_squares
        return cache[selected]

    full = fit_linear_model(rows, y, weights=weight_values)
    if full.df_residual <= 0:
        raise ValueError("the full model leaves no residual degrees of freedom")
    everything = frozenset(names)
    mse = full.residual_sum_squares / full.df_residual
    table: list[AnovaRow] = []
    for index, name in enumerate(names):
        if kind == "I":
            before = frozenset(names[:index])
            ss = rss(before) - rss(before | {name})
        elif kind == "II":
            mine = _factors_of(name)
            kept = frozenset(
                other
                for other in names
                if other != name and not mine < _factors_of(other)
            )
            ss = rss(kept) - rss(kept | {name})
        else:
            ss = rss(everything - {name}) - rss(everything)
        ss = max(ss, 0.0)
        df = len(terms[name])
        mean_square = ss / df
        statistic = mean_square / mse if mse > 0.0 else math.inf
        table.append(
            AnovaRow(
                term=name,
                df=df,
                sum_squares=ss,
                mean_square=mean_square,
                f_statistic=statistic,
                p_value=f_sf(statistic, df, full.df_residual),
            )
        )
    table.append(
        AnovaRow(
            term="Residuals",
            df=full.df_residual,
            sum_squares=full.residual_sum_squares,
            mean_square=mse,
            f_statistic=math.nan,
            p_value=math.nan,
        )
    )
    return AnovaTable(kind=kind, rows=tuple(table), fit=full)


def factorial_design(
    factors: Mapping[str, Sequence[object]],
    *,
    interactions: bool | Iterable[str] = True,
    coding: Coding = "sum",
) -> tuple[DesignMatrix, dict[str, tuple[int, ...]]]:
    """Build an intercept-plus-factors design with interaction columns.

    A factor with sorted levels ``L_1..L_m`` gets ``m - 1`` columns. Sum
    (effect) coding sets column ``k`` to ``1`` at ``L_k``, ``-1`` at the last
    level ``L_m``, and ``0`` otherwise, so main effects are deviations from
    the grand mean and type III tests of main effects are meaningful with
    interactions present. Treatment coding uses ``1{x = L_k}`` for
    ``k = 2..m`` (``L_1`` is the reference). An interaction column is the
    product of one column of each factor in it.

    Args:
        factors: Ordered mapping from factor name to equally long level
            sequences (compared through ``str``).
        interactions: ``True`` for every interaction of two or more factors
            (full factorial), ``False`` for main effects only, or an
            iterable of names like ``"A:B"``.
        coding: ``"sum"`` or ``"treatment"``.

    Returns:
        The design (intercept first) and the term-to-columns mapping for
        :func:`linear_model_anova`.

    Raises:
        ValueError: If no factor is given, lengths differ, a factor has a
            single level, a factor name contains ``":"``, an interaction
            names an unknown factor, or the coding is unknown.

    Equation:
        [EQ-LM-ANOVA-002].

    References:
        [REF-MONTGOMERY-2017, Chapter 5], [REF-FOX-WEISBERG-2019].

    """
    if coding not in ("sum", "treatment"):
        raise ValueError("coding must be 'sum' or 'treatment'")
    if not factors:
        raise ValueError("at least one factor is required")
    lengths = {len(values) for values in factors.values()}
    if len(lengths) != 1:
        raise ValueError("every factor must have the same length")
    (n,) = lengths
    blocks: dict[str, list[list[float]]] = {}
    labels: dict[str, list[str]] = {}
    for name, values in factors.items():
        if ":" in name:
            raise ValueError("factor names must not contain ':'")
        text = [str(v) for v in values]
        levels = sorted(set(text))
        if len(levels) < 2:
            raise ValueError(f"factor {name!r} needs at least two levels")
        if coding == "sum":
            kept = levels[:-1]
            last = levels[-1]
            columns = [
                [1.0 if v == level else (-1.0 if v == last else 0.0) for v in text]
                for level in kept
            ]
        else:
            kept = levels[1:]
            columns = [[1.0 if v == level else 0.0 for v in text] for level in kept]
        blocks[name] = columns
        labels[name] = [f"{name}[{level}]" for level in kept]
    names = list(factors)
    if interactions is True:
        masks = sorted(
            (m for m in range(1, 1 << len(names)) if bin(m).count("1") >= 2),
            key=lambda m: (bin(m).count("1"), m),
        )
        wanted = [
            ":".join(names[i] for i in range(len(names)) if mask >> i & 1)
            for mask in masks
        ]
    elif interactions is False:
        wanted = []
    else:
        wanted = list(interactions)
    for term in wanted:
        parts = term.split(":")
        if len(parts) < 2 or any(part not in blocks for part in parts):
            raise ValueError(f"interaction {term!r} must join known factors")
        columns = [[1.0] * n]
        names_out = [""]
        for part in parts:
            columns = [
                [a * b for a, b in zip(left, right, strict=True)]
                for left in columns
                for right in blocks[part]
            ]
            names_out = [
                f"{prefix}:{label}" if prefix else label
                for prefix in names_out
                for label in labels[part]
            ]
        blocks[term] = columns
        labels[term] = names_out
    design_columns: list[list[float]] = [[1.0] * n]
    column_names = ["(Intercept)"]
    term_columns: dict[str, tuple[int, ...]] = {}
    for term in [*names, *wanted]:
        start = len(design_columns)
        design_columns.extend(blocks[term])
        column_names.extend(labels[term])
        term_columns[term] = tuple(range(start, len(design_columns)))
    rows = tuple(tuple(column[i] for column in design_columns) for i in range(n))
    return (
        DesignMatrix(
            rows=rows,
            column_names=tuple(column_names),
            intercept=True,
            encodings=(),
        ),
        term_columns,
    )


def breusch_pagan_test(
    fit: LinearModelResult,
    variance_design: Iterable[Iterable[object]] | None = None,
    *,
    studentized: bool = True,
) -> BreuschPaganTest:
    """Test whether the error variance depends on the regressors.

    Regress the squared residuals ``e_i^2`` on the variance design ``Z``
    (default: the model's own design) with an intercept. Koenker's
    studentized statistic is ``n R^2`` of that regression; the original
    Breusch-Pagan statistic is ``ESS / (2 sigma_hat^4)`` with
    ``sigma_hat^2 = RSS / n``, which assumes normal errors and is
    oversized under heavier tails. Both are ``chi^2`` with as many degrees
    of freedom as ``Z`` has non-constant columns under constant variance.

    Args:
        fit: An unweighted linear-model fit with an intercept.
        variance_design: Optional rows of the variables the variance may
            depend on; an intercept column is added when missing.
        studentized: Use Koenker's ``n R^2`` form.

    Returns:
        Statistic, p-value, and degrees of freedom.

    Raises:
        ValueError: If the fit is weighted, the variance design has the wrong
            number of rows, or it has no non-constant column.

    Equation:
        [EQ-LM-BP-001].

    References:
        [REF-BREUSCH-PAGAN-1979], [REF-KOENKER-1981].

    """
    if any(w != 1.0 for w in fit.weights):
        raise ValueError("the Breusch-Pagan test needs an unweighted fit")
    n = fit.n_observations
    source = (
        fit.design
        if variance_design is None
        else tuple(as_vector(row, name="variance_design") for row in variance_design)
    )
    if len(source) != n:
        raise ValueError("the variance design needs one row per observation")
    varying = [
        j
        for j in range(len(source[0]))
        if max(row[j] for row in source) - min(row[j] for row in source) > 0.0
    ]
    if not varying:
        raise ValueError("the variance design has no non-constant column")
    z = [(1.0, *(row[j] for j in varying)) for row in source]
    squared = [e * e for e in fit.residuals]
    auxiliary = fit_linear_model(z, squared)
    df = len(varying)
    if studentized:
        statistic = n * auxiliary.r_squared
    else:
        sigma2 = fit.residual_sum_squares / n
        explained = auxiliary.total_sum_squares - auxiliary.residual_sum_squares
        statistic = explained / (2.0 * sigma2 * sigma2)
    return BreuschPaganTest(
        statistic=statistic,
        p_value=chi_square_sf(statistic, df),
        df=df,
        studentized=studentized,
    )


def durbin_watson(residuals: LinearModelResult | Iterable[object]) -> float:
    """Return the Durbin-Watson statistic of residuals in their natural order.

    ``d = sum_{t>=2} (e_t - e_(t-1))^2 / sum_t e_t^2``, about ``2 (1 - r_1)``
    with ``r_1`` the lag-one autocorrelation: near 2 for uncorrelated
    errors, toward 0 for positive and toward 4 for negative correlation.
    Its null distribution depends on the design; Durbin and Watson give
    bounds ``d_L, d_U`` for the test.

    Args:
        residuals: A linear-model fit or residuals in time order.

    Returns:
        The statistic ``d`` in ``[0, 4]``.

    Raises:
        ValueError: If fewer than two residuals are given or all are zero.

    Equation:
        [EQ-LM-DW-001].

    References:
        [REF-DURBIN-WATSON-1950], [REF-DURBIN-WATSON-1951].

    """
    values = (
        residuals.residuals
        if isinstance(residuals, LinearModelResult)
        else as_vector(residuals, name="residuals")
    )
    if len(values) < 2:
        raise ValueError("at least two residuals are required")
    total = math.fsum(e * e for e in values)
    if total == 0.0:
        raise ValueError("the residuals are all zero")
    return (
        math.fsum((b - a) ** 2 for a, b in zip(values, values[1:], strict=False))
        / total
    )
