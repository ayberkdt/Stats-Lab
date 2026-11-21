"""Two-stage least squares for linear instrumental-variable models.

Model: ``y = X beta + u`` where some columns of ``X`` (the *endogenous*
regressors, for example a confounded treatment) are correlated with ``u``.
Instruments ``Z`` are variables that (i) are associated with the
endogenous regressors (relevance), (ii) affect the outcome only through
them (exclusion), and (iii) share no unmeasured causes with the outcome
(exchangeability). Then ``E[Z'u] = 0`` and two-stage least squares
(EQ-CAUSAL-IV-001) is consistent for ``beta`` when effects are
homogeneous; with one binary instrument and one binary treatment it is
the Wald ratio, which under monotonicity is the average effect among
compliers.

The estimator is consistent but not unbiased, and it is unreliable when
the instruments are weak: the first-stage F statistic of the excluded
instruments (EQ-CAUSAL-IV-003) is reported for that reason. Standard
errors are heteroskedasticity-robust sandwich estimates
(EQ-CAUSAL-IV-002); the classical homoskedastic ones are reported too.
None of the three instrument conditions except relevance can be checked
from the data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.inference import fit_linear_model
from probstat.linalg import Matrix, Vector, as_matrix, as_vector, matmul

from .double_ml import _wald_interval

RobustKind = Literal["HC0", "HC1"]
WEAK_INSTRUMENT_F = 10.0


@dataclass(frozen=True, slots=True)
class InstrumentalVariablesResult:
    """Two-stage least-squares fit.

    Attributes:
        coefficients: Estimates ordered as intercept (if any), endogenous
            regressors, exogenous regressors.
        column_names: Matching names.
        standard_errors: Heteroskedasticity-robust standard errors.
        covariance: Robust sandwich covariance of the coefficients.
        classical_standard_errors: Standard errors under homoskedastic
            errors, ``sqrt(diag(s^2 (Xhat'Xhat)^-1))`` with
            ``s^2 = RSS / (n - K)``.
        residuals: ``y - X beta`` computed with the observed (not the
            fitted) regressors.
        first_stage_f: For each endogenous regressor the F statistic of
            the excluded instruments in its first-stage regression.
        first_stage_df: Numerator and denominator degrees of freedom of
            those F statistics.
        weak_instruments: Whether any first-stage F is below 10, a common
            rule of thumb (a heuristic, not a test).
        n_observations: ``n``.
        n_instruments: Number of excluded instruments.
        n_endogenous: Number of endogenous regressors.
        robust_kind: ``"HC0"`` or ``"HC1"``.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 16], [REF-WHITE-1980].

    """

    coefficients: Vector
    column_names: tuple[str, ...]
    standard_errors: Vector
    covariance: Matrix
    classical_standard_errors: Vector
    residuals: Vector
    first_stage_f: Vector
    first_stage_df: tuple[int, int]
    weak_instruments: bool
    n_observations: int
    n_instruments: int
    n_endogenous: int
    robust_kind: str

    def coefficient(self, name: str) -> float:
        """Return one coefficient by column name.

        Args:
            name: Column name.

        Returns:
            The coefficient.

        Raises:
            KeyError: If the name is unknown.

        References:
            [EQ-CAUSAL-IV-001].

        """
        if name not in self.column_names:
            raise KeyError(f"unknown column {name!r}")
        return self.coefficients[self.column_names.index(name)]

    def confidence_interval(
        self, name: str, level: float = 0.95
    ) -> tuple[float, float]:
        """Return the robust normal-theory interval of one coefficient.

        Args:
            name: Column name.
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds (asymptotic; too narrow with weak
            instruments).

        Raises:
            KeyError: If the name is unknown.
            ValueError: If the level is invalid.

        References:
            [EQ-CAUSAL-IV-002].

        """
        estimate = self.coefficient(name)
        error = self.standard_errors[self.column_names.index(name)]
        return _wald_interval(estimate, error, level)


def _columns(
    values: Iterable[object] | Iterable[Iterable[object]], *, name: str, size: int
) -> Matrix:
    """Coerce a vector or a matrix of regressors into ``n`` rows.

    Args:
        values: ``n`` numbers (one regressor) or ``n`` rows.
        name: Argument name for error messages.
        size: Expected number of rows.

    Returns:
        ``n x p`` rows.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the number of rows is wrong.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    items: list[object] = list(values)
    nested = [item for item in items if isinstance(item, Iterable)]
    if not nested:
        rows: Matrix = tuple((v,) for v in as_vector(items, name=name))
    elif len(nested) == len(items):
        rows = as_matrix(nested, name=name)
    else:
        raise ValueError(f"{name} mixes scalars and rows")
    if len(rows) != size:
        raise ValueError(f"{name} must have one row per observation")
    return rows


def _names(given: Sequence[str] | None, prefix: str, count: int) -> tuple[str, ...]:
    """Return user names or defaults ``prefix0, prefix1, ...``.

    Args:
        given: Optional names.
        prefix: Default prefix.
        count: Number of columns.

    Returns:
        Column names.

    Raises:
        ValueError: If the number of names is wrong.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if given is None:
        return tuple(f"{prefix}{j}" for j in range(count))
    names = tuple(given)
    if len(names) != count:
        raise ValueError(f"expected {count} name(s) with prefix {prefix!r}")
    return names


def two_stage_least_squares(
    outcome: Iterable[object],
    endogenous: Iterable[object] | Iterable[Iterable[object]],
    instruments: Iterable[object] | Iterable[Iterable[object]],
    *,
    exogenous: Iterable[Iterable[object]] | None = None,
    intercept: bool = True,
    robust: RobustKind = "HC0",
    endogenous_names: Sequence[str] | None = None,
    exogenous_names: Sequence[str] | None = None,
) -> InstrumentalVariablesResult:
    """Fit ``y = X beta + u`` by two-stage least squares.

    With ``W`` the included exogenous regressors (intercept and
    ``exogenous``), ``X = [D, W]`` and the full instrument matrix
    ``Zf = [Z, W]``:

    1. first stage: ``Xhat = Zf (Zf'Zf)^-1 Zf'X`` (least-squares fitted
       values; columns of ``W`` are reproduced exactly);
    2. second stage: ``beta = (Xhat'X)^-1 Xhat'y``, which equals the
       least-squares regression of ``y`` on ``Xhat``. In the exactly
       identified case this is ``(Zf'X)^-1 Zf'y``.

    The robust covariance is ``A M A`` with ``A = (Xhat'Xhat)^-1`` and
    ``M = sum_i e_i^2 xhat_i xhat_i'``, ``e = y - X beta`` (``HC0``;
    ``HC1`` multiplies by ``n / (n - K)``).

    The first-stage F statistic of an endogenous regressor compares its
    regression on ``Zf`` with its regression on ``W`` alone:
    ``F = ((RSS_W - RSS_Zf) / q) / (RSS_Zf / (n - q - k_W))``.

    Args:
        outcome: ``n`` outcomes.
        endogenous: ``n`` values or ``n x p`` rows of endogenous
            regressors.
        instruments: ``n`` values or ``n x q`` rows of excluded
            instruments, ``q >= p``.
        exogenous: Optional ``n x k`` included exogenous covariates (they
            instrument themselves).
        intercept: Include an intercept in both stages.
        robust: ``"HC0"`` or ``"HC1"``.
        endogenous_names: Names of the endogenous columns; default
            ``d0, d1, ...``.
        exogenous_names: Names of the exogenous columns; default
            ``x0, x1, ...``.

    Returns:
        :class:`InstrumentalVariablesResult`. Estimates are consistent
        under instrument relevance, exclusion and exchangeability with a
        linear homogeneous effect; standard errors are asymptotic.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, the model is under-identified
            (``q < p``), there are too few observations, or ``robust`` is
            unknown.
        RankDeficientDesignError: If the instruments or the fitted
            regressors are collinear (for example an irrelevant or
            duplicated instrument).

    Equation:
        [EQ-CAUSAL-IV-001], [EQ-CAUSAL-IV-002], [EQ-CAUSAL-IV-003].

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 16], [REF-WHITE-1980].

    """
    if robust not in ("HC0", "HC1"):
        raise ValueError("robust must be 'HC0' or 'HC1'")
    y = as_vector(outcome, name="outcome")
    n = len(y)
    d = _columns(endogenous, name="endogenous", size=n)
    z = _columns(instruments, name="instruments", size=n)
    p, q = len(d[0]), len(z[0])
    if q < p:
        raise ValueError(
            f"under-identified: {q} instrument(s) for {p} endogenous regressor(s)"
        )
    if exogenous is None:
        x_rows: Matrix = tuple(() for _ in range(n))
    else:
        x_rows = _columns(exogenous, name="exogenous", size=n)
    k = len(x_rows[0])
    included: Matrix = tuple(((1.0,) if intercept else ()) + row for row in x_rows)
    k_w = len(included[0])
    names = (
        (("(Intercept)",) if intercept else ())
        + _names(endogenous_names, "d", p)
        + _names(exogenous_names, "x", k)
    )
    full_instruments = tuple(zi + wi for zi, wi in zip(z, included, strict=True))
    if n <= q + k_w:
        raise ValueError("too few observations for the first stage")
    fitted_columns: list[Vector] = []
    f_statistics: list[float] = []
    df2 = n - q - k_w
    for j in range(p):
        column = tuple(row[j] for row in d)
        first = fit_linear_model(full_instruments, column)
        fitted_columns.append(first.fitted_values)
        if k_w:
            restricted = fit_linear_model(included, column).residual_sum_squares
        else:
            restricted = math.fsum(v * v for v in column)
        unrestricted = first.residual_sum_squares
        if unrestricted > 0.0:
            f_statistics.append(
                ((restricted - unrestricted) / q) / (unrestricted / df2)
            )
        else:
            f_statistics.append(math.inf)
    intercept_part: tuple[float, ...] = (1.0,) if intercept else ()
    fitted_design = tuple(
        intercept_part + tuple(col[i] for col in fitted_columns) + x_rows[i]
        for i in range(n)
    )
    actual_design = tuple(intercept_part + d[i] + x_rows[i] for i in range(n))
    second = fit_linear_model(fitted_design, y)
    beta = second.coefficients
    size = len(beta)
    residuals = tuple(
        yi - math.fsum(b * v for b, v in zip(beta, row, strict=True))
        for yi, row in zip(y, actual_design, strict=True)
    )
    bread = second.unscaled_covariance
    meat = tuple(
        tuple(
            math.fsum(
                e * e * row[a] * row[b]
                for e, row in zip(residuals, fitted_design, strict=True)
            )
            for b in range(size)
        )
        for a in range(size)
    )
    covariance = matmul(matmul(bread, meat), bread)
    if robust == "HC1":
        factor = n / (n - size)
        covariance = tuple(tuple(v * factor for v in row) for row in covariance)
    sigma2 = math.fsum(e * e for e in residuals) / (n - size)
    return InstrumentalVariablesResult(
        coefficients=beta,
        column_names=names,
        standard_errors=tuple(
            math.sqrt(max(covariance[a][a], 0.0)) for a in range(size)
        ),
        covariance=covariance,
        classical_standard_errors=tuple(
            math.sqrt(sigma2 * bread[a][a]) for a in range(size)
        ),
        residuals=residuals,
        first_stage_f=tuple(f_statistics),
        first_stage_df=(q, df2),
        weak_instruments=min(f_statistics) < WEAK_INSTRUMENT_F,
        n_observations=n,
        n_instruments=q,
        n_endogenous=p,
        robust_kind=robust,
    )
