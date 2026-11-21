"""Omitted-variable-bias sensitivity of a linear-regression coefficient.

Setting: the *long* regression ``Y = tau D + X'beta + gamma Z + e`` contains
a confounder ``Z`` that was not measured, so only the *short* regression
``Y = tau_res D + X'beta_res + e_res`` can be fitted. Least-squares algebra
gives ``tau_res - tau = gamma delta`` with ``delta`` the coefficient of ``D``
in the regression of ``Z`` on ``(D, X)``. Rewritten in scale-free terms
(EQ-CAUSAL-OVB-002) this is:

``|tau_res - tau| = se(tau_res) sqrt(df) sqrt(R2_Y R2_D / (1 - R2_D))``

where ``R2_Y = R^2_{Y~Z|D,X}`` and ``R2_D = R^2_{D~Z|X}`` are the partial
R-squared of the confounder with the outcome and with the treatment, and
``se`` and ``df`` are the classical standard error and the residual degrees
of freedom of the short regression. The identity is exact in the sample
(it is algebra, not an approximation); the tests confirm it by actually
omitting a simulated confounder.

The module reports the partial R-squared of the treatment with the outcome
(EQ-CAUSAL-OVB-001), the robustness value (EQ-CAUSAL-OVB-003: the common
strength ``R2_Y = R2_D`` that moves the estimate by a fraction ``q``), the
robustness value for statistical significance (EQ-CAUSAL-OVB-005), and the
estimate, standard error and t statistic adjusted for a hypothesized
confounder (EQ-CAUSAL-OVB-004).

The analysis answers "how strong would a confounder have to be"; it does
not say whether such a confounder exists, and it covers one linear
confounder direction (a set of confounders acts through its combined
partial R-squared values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.distributions import StudentT
from probstat.foundations.validation import require_finite_real


def _degrees_of_freedom(value: object) -> int:
    """Validate residual degrees of freedom.

    Args:
        value: Candidate integer.

    Returns:
        The integer.

    Raises:
        ValueError: If the value is not a positive integer.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("degrees_of_freedom must be a positive integer")
    return value


def _unit_interval(value: object, *, name: str, closed_right: bool) -> float:
    """Validate a partial R-squared.

    Args:
        value: Candidate number.
        name: Argument name for error messages.
        closed_right: Whether the value one is allowed.

    Returns:
        The number as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is outside ``[0, 1]`` (or ``[0, 1)``).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked < 0.0 or checked > 1.0 or (checked == 1.0 and not closed_right):
        bound = "[0, 1]" if closed_right else "[0, 1)"
        raise ValueError(f"{name} must lie in {bound}")
    return checked


def partial_r2(t_value: float, degrees_of_freedom: int) -> float:
    """Return the partial R-squared of a regressor from its t statistic.

    ``R^2_{Y~D|X} = t^2 / (t^2 + df)``: the share of the outcome variance
    left by the other regressors that the regressor explains. It follows
    from ``t^2 = df (RSS_without - RSS_with) / RSS_with``.

    Args:
        t_value: Classical t statistic of the coefficient.
        degrees_of_freedom: Residual degrees of freedom of that regression.

    Returns:
        Partial R-squared in ``[0, 1)``.

    Raises:
        TypeError: If ``t_value`` is not real.
        ValueError: If an argument is invalid.

    Equation:
        [EQ-CAUSAL-OVB-001].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    t = require_finite_real(t_value, name="t_value")
    df = _degrees_of_freedom(degrees_of_freedom)
    return t * t / (t * t + df)


def omitted_variable_bias(
    standard_error: float,
    degrees_of_freedom: int,
    r2_outcome: float,
    r2_treatment: float,
) -> float:
    """Return the absolute bias caused by omitting a confounder.

    ``|bias| = se sqrt(df) sqrt(R2_Y R2_D / (1 - R2_D))`` with ``se`` and
    ``df`` from the short regression (the one without the confounder).

    Args:
        standard_error: Classical standard error of the treatment
            coefficient in the short regression.
        degrees_of_freedom: Its residual degrees of freedom.
        r2_outcome: ``R^2_{Y~Z|D,X}`` in ``[0, 1]``.
        r2_treatment: ``R^2_{D~Z|X}`` in ``[0, 1)``.

    Returns:
        Non-negative absolute difference between the short and the long
        coefficient.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-CAUSAL-OVB-002].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    se = require_finite_real(standard_error, name="standard_error")
    if se < 0.0:
        raise ValueError("standard_error must be non-negative")
    df = _degrees_of_freedom(degrees_of_freedom)
    r2y = _unit_interval(r2_outcome, name="r2_outcome", closed_right=True)
    r2d = _unit_interval(r2_treatment, name="r2_treatment", closed_right=False)
    return se * math.sqrt(df) * math.sqrt(r2y * r2d / (1.0 - r2d))


def _robustness_from_f(f: float) -> float:
    """Solve ``R / sqrt(1 - R) = f`` for ``R`` in ``[0, 1)``.

    Args:
        f: Target value; non-positive values give zero.

    Returns:
        ``(sqrt(f^4 + 4 f^2) - f^2) / 2``, the positive root of
        ``R^2 + f^2 R - f^2 = 0``.

    Equation:
        [EQ-CAUSAL-OVB-003].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    if f <= 0.0:
        return 0.0
    f2 = f * f
    # Equivalent to (sqrt(f^4 + 4 f^2) - f^2) / 2 without cancellation.
    return 2.0 * f2 / (math.sqrt(f2 * f2 + 4.0 * f2) + f2)


def robustness_value(
    t_value: float,
    degrees_of_freedom: int,
    *,
    q: float = 1.0,
    alpha: float | None = None,
) -> float:
    """Return the robustness value of a regression coefficient.

    With ``alpha=None``: the common value ``RV_q`` of both partial R-squared
    of a confounder (with treatment and with outcome) at which the bias
    equals ``q |estimate|``; ``f_q = q |t| / sqrt(df)`` and
    ``RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2``. ``q = 1`` moves the
    estimate exactly to zero.

    With ``alpha``: the common value at which the *adjusted* t statistic for
    the null hypothesis ``tau = (1 - q) estimate`` equals the two-sided
    critical value ``t*`` of Student-t on ``df - 1`` degrees of freedom.
    For equal strengths the adjusted standard error is
    ``se sqrt(df / (df - 1))`` whatever the strength, so the condition
    reduces to the same equation with
    ``f_{q,alpha} = q |t| / sqrt(df) - t* / sqrt(df - 1)`` (zero when that
    is not positive: the result is already not significant). This is the
    equal-strength definition; a confounder with unequal partial R-squared
    values both below this value can, in some cases, still remove
    significance, because a larger ``R2_Y`` also changes the standard error.

    Args:
        t_value: Classical t statistic of the treatment coefficient.
        degrees_of_freedom: Residual degrees of freedom.
        q: Fraction of the estimate to be explained away (positive).
        alpha: Optional two-sided significance level in ``(0, 1)``.

    Returns:
        Robustness value in ``[0, 1)``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range, or ``alpha`` is given
            with ``degrees_of_freedom < 2``.

    Equation:
        [EQ-CAUSAL-OVB-003], [EQ-CAUSAL-OVB-005].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    t = require_finite_real(t_value, name="t_value")
    df = _degrees_of_freedom(degrees_of_freedom)
    fraction = require_finite_real(q, name="q")
    if fraction <= 0.0:
        raise ValueError("q must be strictly positive")
    f = fraction * abs(t) / math.sqrt(df)
    if alpha is not None:
        level = require_finite_real(alpha, name="alpha")
        if not 0.0 < level < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if df < 2:
            raise ValueError("alpha needs degrees_of_freedom of at least 2")
        critical = StudentT(float(df - 1)).quantile(1.0 - 0.5 * level)
        f -= critical / math.sqrt(df - 1)
    return _robustness_from_f(f)


@dataclass(frozen=True, slots=True)
class SensitivityStatistics:
    """Sensitivity summary of one regression coefficient.

    Attributes:
        estimate: Coefficient of the treatment in the fitted regression.
        standard_error: Its classical standard error.
        t_value: ``estimate / standard_error``.
        degrees_of_freedom: Residual degrees of freedom.
        partial_r2: ``R^2_{Y~D|X}``; also the strength an extreme confounder
            (one explaining all residual outcome variance) needs with the
            treatment to remove the estimate.
        q: Fraction of the estimate considered.
        alpha: Significance level of ``robustness_value_alpha``.
        robustness_value: ``RV_q``.
        robustness_value_alpha: ``RV_{q,alpha}`` (equal-strength definition).

    References:
        [REF-CINELLI-HAZLETT-2020].

    """

    estimate: float
    standard_error: float
    t_value: float
    degrees_of_freedom: int
    partial_r2: float
    q: float
    alpha: float
    robustness_value: float
    robustness_value_alpha: float


def sensitivity_statistics(
    estimate: float,
    standard_error: float,
    degrees_of_freedom: int,
    *,
    q: float = 1.0,
    alpha: float = 0.05,
) -> SensitivityStatistics:
    """Summarize how fragile a regression coefficient is to a confounder.

    Args:
        estimate: Treatment coefficient of the fitted (short) regression.
        standard_error: Its classical (homoskedastic) standard error.
        degrees_of_freedom: Residual degrees of freedom ``n - p`` (at least
            two).
        q: Fraction of the estimate to be explained away.
        alpha: Two-sided significance level.

    Returns:
        :class:`SensitivityStatistics`.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-CAUSAL-OVB-001], [EQ-CAUSAL-OVB-003], [EQ-CAUSAL-OVB-005].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    value = require_finite_real(estimate, name="estimate")
    se = require_finite_real(standard_error, name="standard_error")
    if se <= 0.0:
        raise ValueError("standard_error must be strictly positive")
    df = _degrees_of_freedom(degrees_of_freedom)
    t = value / se
    return SensitivityStatistics(
        estimate=value,
        standard_error=se,
        t_value=t,
        degrees_of_freedom=df,
        partial_r2=partial_r2(t, df),
        q=float(q),
        alpha=float(alpha),
        robustness_value=robustness_value(t, df, q=q),
        robustness_value_alpha=robustness_value(t, df, q=q, alpha=alpha),
    )


@dataclass(frozen=True, slots=True)
class ConfounderAdjustment:
    """Regression results adjusted for a hypothesized omitted confounder.

    Attributes:
        r2_outcome: Assumed ``R^2_{Y~Z|D,X}``.
        r2_treatment: Assumed ``R^2_{D~Z|X}``.
        bias: Absolute bias ``|tau_res - tau|``.
        adjusted_estimate: Coefficient the long regression would report.
        adjusted_standard_error: Its classical standard error.
        adjusted_t: ``adjusted_estimate / adjusted_standard_error``.
        adjusted_degrees_of_freedom: ``df - 1``.
        reduce: Whether the bias was assumed to have inflated the magnitude
            of the estimate (so the adjustment moves it towards and
            possibly through zero).

    References:
        [REF-CINELLI-HAZLETT-2020].

    """

    r2_outcome: float
    r2_treatment: float
    bias: float
    adjusted_estimate: float
    adjusted_standard_error: float
    adjusted_t: float
    adjusted_degrees_of_freedom: int
    reduce: bool


def adjust_for_confounder(
    estimate: float,
    standard_error: float,
    degrees_of_freedom: int,
    *,
    r2_outcome: float,
    r2_treatment: float,
    reduce: bool = True,
) -> ConfounderAdjustment:
    """Adjust a coefficient, its standard error and t for a confounder.

    The long-regression quantities are

    - ``tau = tau_res -+ |bias|`` (the partial R-squared values do not
      determine the sign; ``reduce`` chooses it),
    - ``se = se_res sqrt((1 - R2_Y) / (1 - R2_D)) sqrt(df / (df - 1))``,
    - ``t = tau / se`` on ``df - 1`` degrees of freedom.

    All three are exact sample identities when the two partial R-squared
    values are those of an actual omitted regressor.

    Args:
        estimate: Treatment coefficient of the short regression.
        standard_error: Its classical standard error.
        degrees_of_freedom: Residual degrees of freedom (at least two).
        r2_outcome: ``R^2_{Y~Z|D,X}`` in ``[0, 1]``.
        r2_treatment: ``R^2_{D~Z|X}`` in ``[0, 1)``.
        reduce: ``True`` if the confounder is assumed to have pushed the
            estimate away from zero, ``False`` if towards zero.

    Returns:
        :class:`ConfounderAdjustment`; ``adjusted_t`` is ``inf`` with the
        sign of the adjusted estimate (``nan`` if that is zero) when
        ``r2_outcome == 1``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-CAUSAL-OVB-002], [EQ-CAUSAL-OVB-004].

    References:
        [REF-CINELLI-HAZLETT-2020].

    """
    value = require_finite_real(estimate, name="estimate")
    se = require_finite_real(standard_error, name="standard_error")
    if se <= 0.0:
        raise ValueError("standard_error must be strictly positive")
    df = _degrees_of_freedom(degrees_of_freedom)
    if df < 2:
        raise ValueError("degrees_of_freedom must be at least 2")
    bias = omitted_variable_bias(se, df, r2_outcome, r2_treatment)
    r2y, r2d = float(r2_outcome), float(r2_treatment)
    sign = 1.0 if value >= 0.0 else -1.0
    adjusted = value - sign * bias if reduce else value + sign * bias
    adjusted_se = se * math.sqrt((1.0 - r2y) / (1.0 - r2d)) * math.sqrt(df / (df - 1))
    if adjusted_se > 0.0:
        adjusted_t = adjusted / adjusted_se
    elif adjusted == 0.0:
        adjusted_t = math.nan
    else:
        adjusted_t = math.copysign(math.inf, adjusted)
    return ConfounderAdjustment(
        r2_outcome=r2y,
        r2_treatment=r2d,
        bias=bias,
        adjusted_estimate=adjusted,
        adjusted_standard_error=adjusted_se,
        adjusted_t=adjusted_t,
        adjusted_degrees_of_freedom=df - 1,
        reduce=bool(reduce),
    )
