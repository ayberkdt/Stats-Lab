"""Sampling-based sensitivity measures from a linear regression of the output.

Any i.i.d. sample ``(x_k, y_k)`` of inputs and output (the runs of a forward
propagation) supports a least-squares fit
``y = b_0 + sum_i b_i x_i + e``. Scaling each slope by the ratio of
standard deviations gives the standardized regression coefficient
``SRC_i = b_i s_i / s_y`` ([EQ-SENS-SRC-001]); the partial correlation
coefficient ``PCC_i`` is the correlation of ``y`` and ``x_i`` once the linear
effect of the other inputs has been removed from both, and follows from the
OLS ``t`` statistic of ``b_i`` ([EQ-SENS-PCC-001]). The rank versions SRRC and
PRCC repeat the fit after replacing every column and the output by their
ranks ([EQ-SENS-RANK-001]); they measure monotone rather than linear
association.

When the measures mean something. ``SRC_i^2`` is a share of the output
variance only when the model is close to linear and the inputs are
uncorrelated: then ``sum_i SRC_i^2 ~ R^2 ~ 1``. Two diagnostics are read
before the indices:

- ``R^2`` is the share of the output variance that the linear fit explains.
  A small ``R^2`` (the result flags ``R^2 < 0.7``) means the indices describe
  only the explained part; nonlinear effects and interactions are absent from
  them, and a variance-based or dependence-based measure is needed.
- ``sum_i SRC_i^2`` differs from ``R^2`` by the cross terms
  ``2 sum_{i<j} SRC_i SRC_j r_ij``, ``r_ij`` the sample correlation of
  inputs ``i`` and ``j``. With independent inputs these terms are sampling
  noise of order ``1/sqrt(n)``; with correlated inputs the variance is shared
  and the squared coefficients are no longer additive shares.

PCC (and PRCC) do not apportion variance: they rank inputs by the strength
of the linear (monotone) relation once the other inputs are accounted for,
and they stay large for an input with a strong effect but a small spread.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from probstat.distributions import StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import fit_linear_model
from probstat.modeling import ProbabilisticModel
from probstat.propagation import DESIGNS, make_design
from probstat.statistics import ranks

LOW_R_SQUARED = 0.7
"""``R^2`` below which the result notes that the linear indices are partial."""

ADDITIVITY_TOLERANCE = 0.05
"""``|sum SRC^2 - R^2|`` above which the result notes shared variance."""


@dataclass(frozen=True, slots=True)
class RegressionSensitivityResult:
    """Regression-based sensitivity measures of one output.

    Args:
        input_names: Input names.
        src: Standardized regression coefficients ``b_i s_i / s_y``.
        src_squared: ``SRC_i^2``.
        src_lower: Lower bounds of the ``confidence`` Student-t interval of
            ``SRC_i`` (sample standard deviations held fixed).
        src_upper: Upper bounds of that interval.
        pcc: Partial correlation coefficients ``t_i / sqrt(t_i^2 + df)``.
        t_statistics: OLS ``t`` statistics of the slopes.
        p_values: Two-sided Student-t p-values of ``H0: b_i = 0``.
        r_squared: Coefficient of determination of the linear fit.
        adjusted_r_squared: ``1 - (1 - R^2)(n - 1)/(n - d - 1)``.
        f_statistic: Overall Fisher statistic of ``H0: b_1 = ... = b_d = 0``.
        f_df: Its degrees of freedom ``(d, n - d - 1)``.
        f_p_value: Upper-tail F probability.
        sum_src_squared: ``sum_i SRC_i^2``.
        sample_size: Number of rows ``n``.
        confidence: Interval level.
        srrc: Standardized rank regression coefficients (``None`` unless
            ``rank=True``).
        srrc_squared: ``SRRC_i^2`` (``None`` unless ``rank=True``).
        prcc: Partial rank correlation coefficients (``None`` unless
            ``rank=True``).
        rank_t_statistics: ``t`` statistics of the rank fit.
        rank_p_values: p-values of the rank fit (nominal: ranks are not
            normal).
        rank_r_squared: ``R^2`` of the rank fit (``nan`` unless
            ``rank=True``).
        notes: Plain-language warnings on the validity of the indices.
        evaluations: Model evaluations spent (zero for given data).

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006],
        [REF-IMAN-CONOVER-1979].

    """

    input_names: tuple[str, ...]
    src: tuple[float, ...]
    src_squared: tuple[float, ...]
    src_lower: tuple[float, ...]
    src_upper: tuple[float, ...]
    pcc: tuple[float, ...]
    t_statistics: tuple[float, ...]
    p_values: tuple[float, ...]
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_df: tuple[int, int]
    f_p_value: float
    sum_src_squared: float
    sample_size: int
    confidence: float
    srrc: tuple[float, ...] | None = None
    srrc_squared: tuple[float, ...] | None = None
    prcc: tuple[float, ...] | None = None
    rank_t_statistics: tuple[float, ...] | None = None
    rank_p_values: tuple[float, ...] | None = None
    rank_r_squared: float = math.nan
    notes: tuple[str, ...] = ()
    evaluations: int = 0

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> SRC_i^2``.

        References:
            [EQ-SENS-SRC-001], [REF-SALTELLI-2008, Chapter 1].

        """
        return dict(zip(self.input_names, self.src_squared, strict=True))

    def ranking(self, *, rank: bool = False) -> tuple[str, ...]:
        """Return input names by decreasing ``|SRC_i|`` (or ``|SRRC_i|``).

        Args:
            rank: Order by the rank coefficients.

        Returns:
            Input names, most influential first.

        Raises:
            ValueError: If ``rank=True`` and the rank fit was not computed.

        References:
            [EQ-SENS-SRC-001], [EQ-SENS-RANK-001], [REF-HELTON-ETAL-2006].

        """
        values = self.src
        if rank:
            if self.srrc is None:
                raise ValueError("rank coefficients were not computed (rank=False)")
            values = self.srrc
        order = sorted(range(len(values)), key=lambda i: -abs(values[i]))
        return tuple(self.input_names[i] for i in order)

    def significant(self, level: float | None = None) -> tuple[str, ...]:
        """Return inputs whose slope differs from zero at ``level``.

        This is the Student test of ``H0: b_i^2 Var(X_i) = 0`` for each input
        separately; with ``d`` inputs about ``d level`` false rejections are
        expected under the global null, so read it with :attr:`f_p_value`.

        Args:
            level: Test level (default ``1 - confidence``).

        Returns:
            Names with ``p_i < level``, in input order.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        Equation:
            [EQ-LM-INF-001].

        References:
            [REF-HELTON-ETAL-2006], [REF-MONTGOMERY-PECK-VINING-2021,
            Chapter 3].

        """
        alpha = 1.0 - self.confidence if level is None else float(level)
        if not 0.0 < alpha < 1.0:
            raise ValueError("level must lie in (0, 1)")
        return tuple(
            name
            for name, p in zip(self.input_names, self.p_values, strict=True)
            if p < alpha
        )


def _validated_sample(
    inputs: Sequence[Sequence[float]],
    output: Sequence[float],
    names: Sequence[str] | None,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[str, ...]]:
    """Validate a rectangular input sample, its output, and the names.

    Returns:
        Columns of the inputs, the output, and the input names.

    Raises:
        ValueError: For ragged, mismatched, or non-finite data.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(
        tuple(require_finite_real(v, name=f"inputs[{k}][{j}]") for j, v in enumerate(r))
        for k, r in enumerate(inputs)
    )
    y = tuple(require_finite_real(v, name=f"output[{k}]") for k, v in enumerate(output))
    n = len(rows)
    if n == 0 or len(y) != n:
        raise ValueError("inputs and output need the same positive length")
    dimension = len(rows[0])
    if dimension == 0 or any(len(r) != dimension for r in rows):
        raise ValueError("inputs must be a rectangular matrix with columns")
    labels = tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    if len(labels) != dimension:
        raise ValueError("names must match the number of input columns")
    columns = tuple(tuple(row[i] for row in rows) for i in range(dimension))
    return columns, y, labels


def _standard_deviation(values: Sequence[float]) -> float:
    """Return the sample standard deviation (divisor ``n - 1``).

    References:
        [REF-SALTELLI-2008, Chapter 1]. No new equation is introduced.

    """
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass(frozen=True, slots=True)
class _LinearFit:
    """Standardized coefficients and inference of one least-squares fit.

    References:
        [EQ-SENS-SRC-001], [EQ-SENS-PCC-001].

    """

    src: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    pcc: tuple[float, ...]
    t_values: tuple[float, ...]
    p_values: tuple[float, ...]
    r_squared: float
    adjusted: float
    f_statistic: float
    f_df: tuple[int, int]
    f_p_value: float


def _standardized_fit(
    columns: Sequence[Sequence[float]],
    y: Sequence[float],
    confidence: float,
) -> _LinearFit:
    """Fit ``y`` on the columns with an intercept and standardize the slopes.

    Raises:
        ValueError: If a column or the output is constant, or the columns are
            linearly dependent.

    Equation:
        [EQ-SENS-SRC-001], [EQ-SENS-PCC-001], [EQ-LM-OLS-001],
        [EQ-LM-INF-001].

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006].

    """
    n = len(y)
    scale_y = _standard_deviation(y)
    if not scale_y > 0.0:
        raise ValueError("the output is constant; regression indices are undefined")
    scales = tuple(_standard_deviation(column) for column in columns)
    for index, scale in enumerate(scales):
        if not scale > 0.0:
            raise ValueError(f"input column {index} is constant")
    design = [(1.0, *(column[k] for column in columns)) for k in range(n)]
    fit = fit_linear_model(design, y)
    df = fit.df_residual
    factor = StudentT(float(df)).quantile(0.5 + 0.5 * confidence)
    src: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    pcc: list[float] = []
    for i, scale in enumerate(scales):
        beta = fit.coefficients[i + 1]
        error = fit.standard_errors[i + 1]
        ratio = scale / scale_y
        src.append(beta * ratio)
        lower.append((beta - factor * error) * ratio)
        upper.append((beta + factor * error) * ratio)
        t_value = fit.t_values[i + 1]
        if math.isnan(t_value):
            pcc.append(0.0)
        elif math.isinf(t_value):
            pcc.append(math.copysign(1.0, t_value))
        else:
            pcc.append(t_value / math.sqrt(t_value * t_value + df))
    return _LinearFit(
        src=tuple(src),
        lower=tuple(lower),
        upper=tuple(upper),
        pcc=tuple(pcc),
        t_values=tuple(fit.t_values[1:]),
        p_values=tuple(fit.p_values[1:]),
        r_squared=fit.r_squared,
        adjusted=fit.adjusted_r_squared,
        f_statistic=fit.f_statistic,
        f_df=fit.f_df,
        f_p_value=fit.f_p_value,
    )


def _notes(
    r_squared: float,
    sum_squared: float,
    rank_r_squared: float,
) -> tuple[str, ...]:
    """Return validity notes for the linear and rank indices.

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006].

    """
    notes: list[str] = []
    if r_squared < LOW_R_SQUARED:
        notes.append(
            f"R^2 = {r_squared:.3f} < {LOW_R_SQUARED}: the linear fit leaves "
            f"{1.0 - r_squared:.0%} of the output variance unexplained, so SRC^2 "
            "describes only the linear part; nonlinear effects and interactions "
            "need variance-based (Sobol) or dependence-based (HSIC) measures"
        )
    if abs(sum_squared - r_squared) > ADDITIVITY_TOLERANCE:
        notes.append(
            f"sum SRC^2 = {sum_squared:.3f} differs from R^2 = {r_squared:.3f}: "
            "the sample inputs are correlated, so the squared coefficients share "
            "variance and are not additive shares"
        )
    if math.isfinite(rank_r_squared) and rank_r_squared - r_squared > 0.1:
        notes.append(
            f"rank R^2 = {rank_r_squared:.3f} exceeds R^2 = {r_squared:.3f}: the "
            "relation is monotone but nonlinear; prefer SRRC and PRCC"
        )
    return tuple(notes)


def regression_sensitivity(
    inputs: Sequence[Sequence[float]],
    output: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    rank: bool = False,
    confidence: float = 0.95,
) -> RegressionSensitivityResult:
    """Compute SRC, SRC^2, PCC (and SRRC, PRCC) from an input-output sample.

    The output is regressed on all inputs with an intercept by ordinary least
    squares. ``SRC_i = b_i s_i / s_y`` with sample standard deviations;
    ``PCC_i = t_i / sqrt(t_i^2 + n - d - 1)``, the identity between the OLS
    ``t`` statistic and the partial correlation of ``y`` and ``x_i`` given
    the other inputs. With ``rank=True`` the fit is repeated on the
    mid-ranks of every column and of the output (SRRC, PRCC, rank ``R^2``).
    The Student and Fisher tests assume a linear model with independent
    normal errors of constant variance; with a deterministic simulator the
    "error" is the nonlinear remainder, so read the p-values as a screening
    device whose calibration improves as ``R^2`` approaches one.

    Args:
        inputs: ``n`` rows of ``d`` input values.
        output: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        rank: Also compute the rank-transformed measures.
        confidence: Level of the SRC intervals and the default level of
            :meth:`RegressionSensitivityResult.significant`.

    Returns:
        The coefficients, tests, fit measures, and validity notes.

    Raises:
        ValueError: For ragged or non-finite data, ``n < d + 3``, a constant
            input or output, linearly dependent inputs, or a confidence
            outside ``(0, 1)``.

    Equation:
        [EQ-SENS-SRC-001], [EQ-SENS-PCC-001], [EQ-SENS-RANK-001],
        [EQ-LM-INF-001], [EQ-LM-FIT-001].

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006],
        [REF-IMAN-CONOVER-1979], [REF-MONTGOMERY-PECK-VINING-2021,
        Chapter 3].

    """
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    columns, y, labels = _validated_sample(inputs, output, names)
    n, dimension = len(y), len(columns)
    if n < dimension + 3:
        raise ValueError("regression indices need at least d + 3 rows")
    linear = _standardized_fit(columns, y, level)
    sum_squared = math.fsum(value * value for value in linear.src)
    rank_fit: _LinearFit | None = None
    if rank:
        rank_fit = _standardized_fit(
            tuple(ranks(column) for column in columns), ranks(y), level
        )
    rank_r_squared = rank_fit.r_squared if rank_fit is not None else math.nan
    return RegressionSensitivityResult(
        input_names=labels,
        src=linear.src,
        src_squared=tuple(value * value for value in linear.src),
        src_lower=linear.lower,
        src_upper=linear.upper,
        pcc=linear.pcc,
        t_statistics=linear.t_values,
        p_values=linear.p_values,
        r_squared=linear.r_squared,
        adjusted_r_squared=linear.adjusted,
        f_statistic=linear.f_statistic,
        f_df=linear.f_df,
        f_p_value=linear.f_p_value,
        sum_src_squared=sum_squared,
        sample_size=n,
        confidence=level,
        srrc=None if rank_fit is None else rank_fit.src,
        srrc_squared=None
        if rank_fit is None
        else tuple(value * value for value in rank_fit.src),
        prcc=None if rank_fit is None else rank_fit.pcc,
        rank_t_statistics=None if rank_fit is None else rank_fit.t_values,
        rank_p_values=None if rank_fit is None else rank_fit.p_values,
        rank_r_squared=rank_r_squared,
        notes=_notes(linear.r_squared, sum_squared, rank_r_squared),
    )


def model_sample(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    design: str = "mc",
    label: str = "given-data",
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], int]:
    """Draw a design from a model's joint law and evaluate one QoI.

    Args:
        model: Probabilistic model (dependent inputs are allowed).
        quantity: QoI name.
        size: Number of rows.
        seed: Root seed or stream.
        design: ``"mc"`` (default), ``"lhs"``, ``"halton"``, or ``"sobol"``.
        label: Child-stream label that keeps different analyses apart.

    Returns:
        Input rows, output values, and the number of model evaluations.

    Raises:
        ValueError: For an invalid size or design.
        KeyError: For an unknown QoI.

    Equation:
        [EQ-SAMP-INV-001], [EQ-UQ-001].

    References:
        [REF-HELTON-ETAL-2006], [REF-OWEN-2013, Chapters 10 and 15].

    """
    model.quantity(quantity)
    if isinstance(size, bool) or not isinstance(size, int) or size < 4:
        raise ValueError("size must be an integer of at least four")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    stream = resolve_stream(seed).child(label)
    unit = make_design(design, size, model.dimension, seed=stream)
    before = model.evaluations
    rows = tuple(tuple(model.inputs.from_unit(point)) for point in unit.points)
    values = tuple(float(model.evaluate(row)[quantity]) for row in rows)
    return rows, values, model.evaluations - before


def model_regression_sensitivity(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    design: str = "mc",
    rank: bool = False,
    confidence: float = 0.95,
) -> RegressionSensitivityResult:
    """Sample a model, run it, and compute its regression-based indices.

    The rows are drawn from the model's joint law through a unit design
    (plain Monte Carlo by default, which keeps the rows i.i.d. as the
    Student and Fisher tests assume; LHS and quasi-random rows give valid
    point estimates but optimistic p-values).

    Args:
        model: Probabilistic model.
        quantity: QoI name.
        size: Number of model runs (at least ``d + 3``).
        seed: Root seed or stream.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        rank: Also compute SRRC and PRCC.
        confidence: Level of the SRC intervals.

    Returns:
        :class:`RegressionSensitivityResult` with ``evaluations`` set.

    Raises:
        ValueError: For invalid settings or a degenerate sample.
        KeyError: For an unknown QoI.

    Equation:
        [EQ-SENS-SRC-001], [EQ-SENS-PCC-001], [EQ-SENS-RANK-001].

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006].

    """
    rows, values, evaluations = model_sample(
        model, quantity, size, seed=seed, design=design, label="regression"
    )
    result = regression_sensitivity(
        rows, values, names=model.inputs.names, rank=rank, confidence=confidence
    )
    return replace(result, evaluations=evaluations)
