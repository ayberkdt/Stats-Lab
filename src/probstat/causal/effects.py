"""Average treatment effect estimation from observational data.

The estimand is the average treatment effect ``tau = E[Y(1)] - E[Y(0)]`` of a
binary treatment ``T`` on an outcome ``Y`` under conditional exchangeability
given covariates ``X``, positivity ``0 < e(X) < 1`` with
``e(x) = P(T = 1 | X = x)``, and consistency. Three estimators are offered:

- **Regression adjustment** (standardization, EQ-CAUSAL-REG-001): separate
  least-squares outcome models per arm (:func:`probstat.inference.
  fit_linear_model`), averaged over the whole sample.
- **Inverse-probability weighting** (EQ-CAUSAL-IPW-001, EQ-CAUSAL-SW-001):
  logistic propensity scores (:func:`probstat.inference.fit_glm`), with
  Horvitz-Thompson or stabilized (Hajek) weights and optional trimming.
- **Augmented IPW** (EQ-CAUSAL-AIPW-001): doubly robust; consistent when
  either the propensity model or the outcome model is correctly specified.

Nonparametric bootstrap intervals reuse
:func:`probstat.inference.bootstrap_confidence_interval` with a seeded
:class:`~probstat.foundations.RandomStream`. Covariate balance is summarized by
standardized mean differences (EQ-CAUSAL-SMD-001).
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from probstat.foundations.random_streams import RandomStream
from probstat.inference import (
    BootstrapInterval,
    GLMResult,
    LinearModelResult,
    bootstrap_confidence_interval,
    fit_glm,
    fit_linear_model,
)
from probstat.linalg import Matrix, Vector, as_matrix, as_vector

EffectMethod = Literal["regression", "ipw", "aipw"]
BootstrapMethod = Literal["percentile", "basic", "normal"]
POSITIVITY_WARNING_BOUND = 0.01


class PositivityWarning(UserWarning):
    """Emitted when propensity scores come close to zero or one.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 3]. No equation is evaluated.

    """


def _binary_treatment(treatment: Iterable[object], size: int) -> tuple[int, ...]:
    """Validate a 0/1 treatment vector with both arms present.

    Args:
        treatment: Treatment indicators (booleans or numbers equal to 0 or 1).
        size: Expected length.

    Returns:
        Integer indicators.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If an entry is not 0 or 1, the length is wrong, or an
            arm is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    values = as_vector(treatment, name="treatment")
    if len(values) != size:
        raise ValueError("treatment must have one entry per observation")
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("treatment entries must be 0 or 1")
    indicators = tuple(int(value) for value in values)
    if 0 not in indicators or 1 not in indicators:
        raise ValueError("both treatment arms must be present")
    return indicators


def _prepare(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object] | None,
) -> tuple[Matrix, tuple[int, ...], Vector]:
    """Validate covariate rows, treatment, and (optionally) outcome.

    Args:
        covariates: ``n x k`` covariate rows (``k >= 1``).
        treatment: ``n`` binary indicators.
        outcome: ``n`` finite outcomes, or ``None``.

    Returns:
        Rows, treatment indicators, and outcomes (empty when ``None``).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or the treatment is invalid.

    References:
        Project validation policy. No equation is evaluated.

    """
    rows = as_matrix(covariates, name="covariates")
    t = _binary_treatment(treatment, len(rows))
    if outcome is None:
        return rows, t, ()
    y = as_vector(outcome, name="outcome")
    if len(y) != len(rows):
        raise ValueError("outcome must have one entry per observation")
    return rows, t, y


def _with_intercept(rows: Matrix) -> Matrix:
    """Prepend a column of ones.

    Args:
        rows: Covariate rows.

    Returns:
        Design rows ``(1, x_1, ..., x_k)``.

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3]. No equation is evaluated.

    """
    return tuple((1.0, *row) for row in rows)


def _check_trim(trim: float | None) -> float | None:
    """Validate a trimming threshold.

    Args:
        trim: ``None`` or a threshold in ``(0, 0.5)``.

    Returns:
        The threshold.

    Raises:
        ValueError: If the threshold is outside ``(0, 0.5)``.

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 16]. No equation is evaluated.

    """
    if trim is None:
        return None
    if isinstance(trim, bool) or not 0.0 < float(trim) < 0.5:
        raise ValueError("trim must lie strictly between 0 and 0.5")
    return float(trim)


@dataclass(frozen=True, slots=True)
class PropensityModel:
    """Logistic-regression propensity scores ``e(x) = P(T = 1 | x)``.

    Attributes:
        scores: Fitted propensity score of each observation.
        fit: Underlying binomial GLM (logit link) with an intercept column.

    References:
        [REF-ROSENBAUM-RUBIN-1983], [REF-IMBENS-RUBIN-2015, Chapter 13].

    """

    scores: Vector
    fit: GLMResult


def fit_propensity_scores(
    covariates: Iterable[Iterable[object]], treatment: Iterable[object]
) -> PropensityModel:
    """Fit ``logit e(x) = b0 + x'b`` by maximum likelihood (IRLS).

    Args:
        covariates: ``n x k`` covariate rows; an intercept is added.
        treatment: ``n`` binary indicators.

    Returns:
        :class:`PropensityModel`.

    Raises:
        SeparationError: If the treatment is perfectly predicted (positivity
            fails in the sample).
        RankDeficientDesignError: If covariates are collinear.
        ValueError: If inputs are invalid.

    Equation:
        [EQ-CAUSAL-PS-001], [EQ-GLM-IRLS-001].

    References:
        [REF-ROSENBAUM-RUBIN-1983], [REF-MCCULLAGH-NELDER-1989, Chapter 4].

    """
    rows, t, _ = _prepare(covariates, treatment, None)
    names = ("(Intercept)", *(f"x{j}" for j in range(len(rows[0]))))
    fit = fit_glm(_with_intercept(rows), t, family="binomial", column_names=names)
    return PropensityModel(scores=fit.fitted_values, fit=fit)


def _propensity(
    rows: Matrix, t: tuple[int, ...], supplied: Iterable[object] | None
) -> Vector:
    """Return fitted or supplied propensity scores and warn near the bounds.

    Args:
        rows: Propensity covariate rows.
        t: Treatment indicators.
        supplied: Optional known scores in ``(0, 1)``.

    Returns:
        Scores.

    Raises:
        ValueError: If supplied scores are outside ``(0, 1)`` or misshaped.

    Warns:
        PositivityWarning: If a score lies outside ``[0.01, 0.99]``.

    Equation:
        [EQ-CAUSAL-PS-001].

    References:
        [REF-ROSENBAUM-RUBIN-1983], [REF-HERNAN-ROBINS-2020, Chapter 12].

    """
    if supplied is None:
        scores = fit_propensity_scores(rows, t).scores
    else:
        scores = as_vector(supplied, name="propensity_scores")
        if len(scores) != len(t):
            raise ValueError("propensity_scores must have one entry per observation")
        if any(not 0.0 < e < 1.0 for e in scores):
            raise ValueError("propensity_scores must lie strictly in (0, 1)")
    extreme = sum(
        1
        for e in scores
        if e < POSITIVITY_WARNING_BOUND or e > 1.0 - POSITIVITY_WARNING_BOUND
    )
    if extreme:
        warnings.warn(
            f"{extreme} propensity score(s) outside [0.01, 0.99]; weights are "
            "unstable and positivity is doubtful (consider trim=)",
            PositivityWarning,
            stacklevel=3,
        )
    return scores


def _arm_models(
    rows: Matrix, t: tuple[int, ...], y: Vector
) -> tuple[Vector, Vector, LinearModelResult, LinearModelResult]:
    """Fit one least-squares outcome model per arm and predict for everyone.

    Args:
        rows: Outcome covariate rows.
        t: Treatment indicators.
        y: Outcomes.

    Returns:
        Predictions ``m_1(x_i)`` and ``m_0(x_i)`` for all units and both fits.

    Raises:
        ValueError: If an arm has no more units than coefficients.
        RankDeficientDesignError: If an arm's design is collinear.

    Equation:
        [EQ-LM-OLS-001], [EQ-LM-PRED-001].

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

    """
    design = _with_intercept(rows)
    fits: list[LinearModelResult] = []
    for arm in (1, 0):
        arm_rows = [row for row, ti in zip(design, t, strict=True) if ti == arm]
        arm_y = [yi for yi, ti in zip(y, t, strict=True) if ti == arm]
        if len(arm_rows) <= len(design[0]):
            raise ValueError(
                f"arm T={arm} has {len(arm_rows)} units; the outcome model needs "
                f"more than {len(design[0])}"
            )
        fits.append(fit_linear_model(arm_rows, arm_y))
    treated_fit, control_fit = fits
    return (
        treated_fit.predict(design),
        control_fit.predict(design),
        treated_fit,
        control_fit,
    )


def _mean_and_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and the standard error ``sd / sqrt(n)``.

    Args:
        values: At least two influence-function values.

    Returns:
        Mean and standard error.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 13]. No equation is evaluated.

    """
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    """A point estimate of the average treatment effect and its context.

    Attributes:
        method: ``"regression"``, ``"ipw"``, ``"ipw-stabilized"``, or
            ``"aipw"``.
        estimand: ``"ATE"`` or ``"ATE (trimmed population)"`` after trimming.
        estimate: Estimated ``E[Y(1)] - E[Y(0)]``.
        standard_error: Influence-function standard error (``nan`` for
            regression adjustment; use :func:`bootstrap_effect`).
        treated_mean: Estimated ``E[Y(1)]``.
        control_mean: Estimated ``E[Y(0)]``.
        n_used: Observations entering the estimate.
        n_trimmed: Observations removed by trimming.
        propensity_scores: Scores of the used observations (IPW and AIPW).
        weights: Inverse-probability weights of the used observations (IPW).

    References:
        [REF-HERNAN-ROBINS-2020, Chapters 12 and 13],
        [REF-IMBENS-RUBIN-2015, Chapter 12].

    """

    method: str
    estimand: str
    estimate: float
    standard_error: float
    treated_mean: float
    control_mean: float
    n_used: int
    n_trimmed: int
    propensity_scores: Vector | None = None
    weights: Vector | None = None

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the Wald interval ``estimate +- z se``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds.

        Raises:
            ValueError: If the level is invalid or no standard error exists.

        References:
            [REF-HERNAN-ROBINS-2020, Chapter 13]. No equation is evaluated.

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between zero and one")
        if not math.isfinite(self.standard_error):
            raise ValueError(f"{self.method} has no analytic standard error")
        z = NormalDist().inv_cdf(0.5 * (1.0 + level))
        return (
            self.estimate - z * self.standard_error,
            self.estimate + z * self.standard_error,
        )


def estimate_regression_adjustment(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
) -> EffectEstimate:
    """Estimate the ATE by standardization over per-arm linear outcome models.

    Fits ``m_t(x) = E[Y | T = t, X = x]`` by least squares within each arm and
    averages ``m_1(x_i) - m_0(x_i)`` over all ``n`` units. Consistent when the
    linear outcome models are correct; biased otherwise.

    Args:
        covariates: ``n x k`` outcome-model covariates.
        treatment: ``n`` binary indicators.
        outcome: ``n`` finite outcomes.

    Returns:
        :class:`EffectEstimate` with ``standard_error = nan``.

    Raises:
        ValueError: If inputs are invalid or an arm is too small.
        RankDeficientDesignError: If an arm's design is collinear.

    Equation:
        [EQ-CAUSAL-REG-001].

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 13], [REF-IMBENS-RUBIN-2015, Chapter 12].

    """
    rows, t, y = _prepare(covariates, treatment, outcome)
    m1, m0, _, _ = _arm_models(rows, t, y)
    treated_mean = math.fsum(m1) / len(m1)
    control_mean = math.fsum(m0) / len(m0)
    return EffectEstimate(
        method="regression",
        estimand="ATE",
        estimate=treated_mean - control_mean,
        standard_error=math.nan,
        treated_mean=treated_mean,
        control_mean=control_mean,
        n_used=len(y),
        n_trimmed=0,
    )


def _trim_mask(scores: Vector, trim: float | None) -> tuple[bool, ...]:
    """Return which units keep ``trim <= e <= 1 - trim``.

    Args:
        scores: Propensity scores.
        trim: Threshold or ``None`` (keep all).

    Returns:
        Keep flags.

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 16]. No equation is evaluated.

    """
    if trim is None:
        return tuple(True for _ in scores)
    return tuple(trim <= e <= 1.0 - trim for e in scores)


def _kept(
    keep: tuple[bool, ...], *columns: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Filter parallel columns by a keep mask.

    Args:
        keep: Keep flags.
        *columns: Parallel sequences.

    Returns:
        Filtered columns.

    References:
        Project validation policy. No equation is evaluated.

    """
    return tuple(
        tuple(value for value, flag in zip(column, keep, strict=True) if flag)
        for column in columns
    )


def estimate_ipw(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
    *,
    stabilized: bool = True,
    trim: float | None = None,
    propensity_scores: Iterable[object] | None = None,
) -> EffectEstimate:
    """Estimate the ATE by inverse-probability weighting.

    Horvitz-Thompson (``stabilized=False``):
    ``mu_1 = (1/n) sum T Y / e``, ``mu_0 = (1/n) sum (1-T) Y / (1-e)`` with
    weights ``W = T/e + (1-T)/(1-e)``. Stabilized: weights
    ``SW = T p / e + (1-T)(1-p) / (1-e)`` with ``p = mean(T)``, and each arm
    mean is the ``SW``-weighted average (Hajek form), which is invariant to
    rescaling the weights within an arm.

    Args:
        covariates: ``n x k`` propensity-model covariates.
        treatment: ``n`` binary indicators.
        outcome: ``n`` finite outcomes.
        stabilized: Use stabilized (Hajek) weights; default ``True``.
        trim: Optional threshold ``a`` in ``(0, 0.5)``; units with
            ``e < a`` or ``e > 1 - a`` are dropped, which changes the estimand
            to the ATE of the remaining overlap population.
        propensity_scores: Optional known scores replacing the logistic fit.

    Returns:
        :class:`EffectEstimate`; its standard error treats the propensity
        scores as known (use :func:`bootstrap_effect` to include their
        estimation).

    Raises:
        ValueError: If inputs are invalid or trimming empties an arm.
        SeparationError: If the propensity model separates the arms.

    Warns:
        PositivityWarning: If scores fall outside ``[0.01, 0.99]``.

    Equation:
        [EQ-CAUSAL-IPW-001], [EQ-CAUSAL-SW-001], [EQ-CAUSAL-PS-001].

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 12], [REF-ROSENBAUM-RUBIN-1983],
        [REF-IMBENS-RUBIN-2015, Chapter 16].

    """
    rows, t_all, y_all = _prepare(covariates, treatment, outcome)
    threshold = _check_trim(trim)
    scores_all = _propensity(rows, t_all, propensity_scores)
    keep = _trim_mask(scores_all, threshold)
    tf, y, e = _kept(keep, [float(v) for v in t_all], y_all, scores_all)
    t = tuple(int(v) for v in tf)
    n = len(t)
    if sum(t) == 0 or sum(t) == n:
        raise ValueError("trimming left a treatment arm empty")
    if stabilized:
        p = sum(t) / n
        weights = tuple(
            p / ei if ti else (1.0 - p) / (1.0 - ei)
            for ti, ei in zip(t, e, strict=True)
        )
        treated_weight = math.fsum(w for w, ti in zip(weights, t, strict=True) if ti)
        control_weight = math.fsum(
            w for w, ti in zip(weights, t, strict=True) if not ti
        )
        mu1 = (
            math.fsum(w * yi for w, yi, ti in zip(weights, y, t, strict=True) if ti)
            / treated_weight
        )
        mu0 = (
            math.fsum(w * yi for w, yi, ti in zip(weights, y, t, strict=True) if not ti)
            / control_weight
        )
        scale1 = treated_weight / n / p
        scale0 = control_weight / n / (1.0 - p)
        influence = [
            (ti * (yi - mu1) / ei) / scale1
            - ((1 - ti) * (yi - mu0) / (1.0 - ei)) / scale0
            for ti, yi, ei in zip(t, y, e, strict=True)
        ]
        method = "ipw-stabilized"
    else:
        weights = tuple(
            1.0 / ei if ti else 1.0 / (1.0 - ei) for ti, ei in zip(t, e, strict=True)
        )
        mu1 = math.fsum(ti * yi / ei for ti, yi, ei in zip(t, y, e, strict=True)) / n
        mu0 = (
            math.fsum(
                (1 - ti) * yi / (1.0 - ei) for ti, yi, ei in zip(t, y, e, strict=True)
            )
            / n
        )
        influence = [
            ti * yi / ei - (1 - ti) * yi / (1.0 - ei)
            for ti, yi, ei in zip(t, y, e, strict=True)
        ]
        method = "ipw"
    _, error = _mean_and_error(influence)
    return EffectEstimate(
        method=method,
        estimand="ATE" if threshold is None else "ATE (trimmed population)",
        estimate=mu1 - mu0,
        standard_error=error,
        treated_mean=mu1,
        control_mean=mu0,
        n_used=n,
        n_trimmed=len(t_all) - n,
        propensity_scores=e,
        weights=weights,
    )


def estimate_aipw(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
    *,
    propensity_covariates: Iterable[Iterable[object]] | None = None,
    outcome_covariates: Iterable[Iterable[object]] | None = None,
    trim: float | None = None,
    propensity_scores: Iterable[object] | None = None,
) -> EffectEstimate:
    """Estimate the ATE by augmented inverse-probability weighting.

    With outcome predictions ``m_t(x)`` and propensity ``e(x)``, each unit
    contributes ``psi_i = m_1 - m_0 + T (Y - m_1) / e - (1-T)(Y - m_0)/(1-e)``;
    the estimate is ``mean(psi)`` and the standard error ``sd(psi)/sqrt(n)``.
    The estimator is consistent if either model is correctly specified.

    Args:
        covariates: Default ``n x k`` covariates for both models.
        treatment: ``n`` binary indicators.
        outcome: ``n`` finite outcomes.
        propensity_covariates: Optional covariates for the propensity model
            only.
        outcome_covariates: Optional covariates for the outcome models only.
        trim: Optional threshold in ``(0, 0.5)``; models are fitted on all
            units and ``psi`` is averaged over the units with
            ``a <= e <= 1 - a``.
        propensity_scores: Optional known scores replacing the logistic fit.

    Returns:
        :class:`EffectEstimate` with the influence-function standard error.

    Raises:
        ValueError: If inputs are invalid, an arm is too small, or trimming
            empties an arm.
        SeparationError: If the propensity model separates the arms.
        RankDeficientDesignError: If a design is collinear.

    Warns:
        PositivityWarning: If scores fall outside ``[0.01, 0.99]``.

    Equation:
        [EQ-CAUSAL-AIPW-001], [EQ-CAUSAL-PS-001], [EQ-CAUSAL-REG-001].

    References:
        [REF-ROBINS-ROTNITZKY-ZHAO-1994], [REF-HERNAN-ROBINS-2020, Chapter 13],
        [REF-IMBENS-RUBIN-2015, Chapter 12].

    """
    rows, t_all, y_all = _prepare(covariates, treatment, outcome)
    threshold = _check_trim(trim)
    ps_rows = (
        rows
        if propensity_covariates is None
        else _prepare(propensity_covariates, t_all, None)[0]
    )
    out_rows = (
        rows
        if outcome_covariates is None
        else _prepare(outcome_covariates, t_all, None)[0]
    )
    scores_all = _propensity(ps_rows, t_all, propensity_scores)
    m1_all, m0_all, _, _ = _arm_models(out_rows, t_all, y_all)
    keep = _trim_mask(scores_all, threshold)
    tf, y, e, m1, m0 = _kept(
        keep, [float(v) for v in t_all], y_all, scores_all, m1_all, m0_all
    )
    t = tuple(int(v) for v in tf)
    n = len(t)
    if sum(t) == 0 or sum(t) == n:
        raise ValueError("trimming left a treatment arm empty")
    treated_terms = [
        a + ti * (yi - a) / ei for a, ti, yi, ei in zip(m1, t, y, e, strict=True)
    ]
    control_terms = [
        b + (1 - ti) * (yi - b) / (1.0 - ei)
        for b, ti, yi, ei in zip(m0, t, y, e, strict=True)
    ]
    influence = [a - b for a, b in zip(treated_terms, control_terms, strict=True)]
    estimate, error = _mean_and_error(influence)
    return EffectEstimate(
        method="aipw",
        estimand="ATE" if threshold is None else "ATE (trimmed population)",
        estimate=estimate,
        standard_error=error,
        treated_mean=statistics.fmean(treated_terms),
        control_mean=statistics.fmean(control_terms),
        n_used=n,
        n_trimmed=len(t_all) - n,
        propensity_scores=e,
    )


def bootstrap_effect(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
    *,
    method: EffectMethod = "aipw",
    replicates: int = 200,
    confidence: float = 0.95,
    interval: BootstrapMethod = "percentile",
    seed: int | RandomStream | None = None,
    stabilized: bool = True,
    trim: float | None = None,
    propensity_covariates: Iterable[Iterable[object]] | None = None,
    outcome_covariates: Iterable[Iterable[object]] | None = None,
) -> BootstrapInterval:
    """Bootstrap an ATE estimator by resampling whole units with replacement.

    Every replicate refits the propensity and outcome models, so the interval
    reflects their estimation. Positivity warnings inside replicates are
    suppressed; the original-sample fit still emits them.

    Args:
        covariates: ``n x k`` covariates.
        treatment: ``n`` binary indicators.
        outcome: ``n`` finite outcomes.
        method: ``"regression"``, ``"ipw"``, or ``"aipw"``.
        replicates: Bootstrap resamples (at least two).
        confidence: Two-sided level in ``(0, 1)``.
        interval: ``"percentile"``, ``"basic"``, or ``"normal"``.
        seed: Integer seed or stream; pass one for reproducibility.
        stabilized: IPW weight type (ignored by other methods).
        trim: Optional trimming threshold (IPW and AIPW).
        propensity_covariates: Optional AIPW propensity covariates.
        outcome_covariates: Optional AIPW outcome covariates.

    Returns:
        :class:`~probstat.inference.BootstrapInterval` for the ATE.

    Raises:
        ValueError: If inputs are invalid, ``method`` or ``interval`` is
            unknown, or a resample cannot be fitted (for example an arm
            becomes too small).

    Equation:
        [EQ-INF-BOOT-001], [EQ-INF-BOOT-002].

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapters 6 and 13],
        [REF-HERNAN-ROBINS-2020, Chapter 13].

    """
    if method not in ("regression", "ipw", "aipw"):
        raise ValueError("method must be 'regression', 'ipw', or 'aipw'")
    if interval not in ("percentile", "basic", "normal"):
        raise ValueError("interval must be 'percentile', 'basic', or 'normal'")
    rows, t, y = _prepare(covariates, treatment, outcome)
    ps_rows = (
        rows
        if propensity_covariates is None
        else _prepare(propensity_covariates, t, None)[0]
    )
    out_rows = (
        rows if outcome_covariates is None else _prepare(outcome_covariates, t, None)[0]
    )

    def statistic(indices: Sequence[int]) -> float:
        """Evaluate the chosen estimator on one resample of unit indices.

        References:
            [REF-EFRON-TIBSHIRANI-1993, Chapter 6]. No equation is evaluated.

        """
        sub_t = [t[i] for i in indices]
        sub_y = [y[i] for i in indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            if method == "regression":
                result = estimate_regression_adjustment(
                    [out_rows[i] for i in indices], sub_t, sub_y
                )
            elif method == "ipw":
                result = estimate_ipw(
                    [ps_rows[i] for i in indices],
                    sub_t,
                    sub_y,
                    stabilized=stabilized,
                    trim=trim,
                )
            else:
                result = estimate_aipw(
                    [rows[i] for i in indices],
                    sub_t,
                    sub_y,
                    propensity_covariates=[ps_rows[i] for i in indices],
                    outcome_covariates=[out_rows[i] for i in indices],
                    trim=trim,
                )
        return result.estimate

    return bootstrap_confidence_interval(
        statistic,
        range(len(t)),
        method=interval,
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class CovariateBalance:
    """Balance summary of one covariate.

    Attributes:
        name: Covariate name.
        treated_mean: (Weighted) mean among the treated.
        control_mean: (Weighted) mean among the controls.
        standardized_difference: ``(treated_mean - control_mean) / s_pooled``.

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 14].

    """

    name: str
    treated_mean: float
    control_mean: float
    standardized_difference: float


@dataclass(frozen=True, slots=True)
class BalanceTable:
    """Standardized mean differences for every covariate.

    Attributes:
        covariates: One :class:`CovariateBalance` per covariate.
        weighted: Whether weights were applied to the means.

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 14].

    """

    covariates: tuple[CovariateBalance, ...]
    weighted: bool

    @property
    def max_absolute_difference(self) -> float:
        """Return the largest absolute standardized difference.

        Returns:
            ``max_j |SMD_j|``.

        References:
            [REF-IMBENS-RUBIN-2015, Chapter 14]. No equation is evaluated.

        """
        return max(abs(row.standardized_difference) for row in self.covariates)


def standardized_mean_differences(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    *,
    weights: Iterable[object] | None = None,
    names: Sequence[str] | None = None,
) -> BalanceTable:
    """Compute standardized mean differences between treatment arms.

    ``SMD_j = (xbar_1j - xbar_0j) / sqrt((s_1j^2 + s_0j^2) / 2)``. With
    ``weights`` (for example IPW weights) the arm means are weighted while the
    denominator keeps the unweighted arm variances, so weighted and
    unweighted SMDs share one scale.

    Args:
        covariates: ``n x k`` covariate rows.
        treatment: ``n`` binary indicators.
        weights: Optional non-negative weights with a positive sum per arm.
        names: Optional covariate names; defaults to ``x0, x1, ...``.

    Returns:
        :class:`BalanceTable`.

    Raises:
        ValueError: If shapes disagree, an arm has fewer than two units, a
            weight is negative, an arm's weights sum to zero, or a covariate
            has zero variance in both arms.

    Equation:
        [EQ-CAUSAL-SMD-001].

    References:
        [REF-IMBENS-RUBIN-2015, Chapter 14].

    """
    rows, t, _ = _prepare(covariates, treatment, None)
    k = len(rows[0])
    labels = tuple(names) if names is not None else tuple(f"x{j}" for j in range(k))
    if len(labels) != k:
        raise ValueError("names must have one entry per covariate")
    if weights is None:
        w = tuple(1.0 for _ in rows)
    else:
        w = as_vector(weights, name="weights")
        if len(w) != len(rows):
            raise ValueError("weights must have one entry per observation")
        if any(value < 0.0 for value in w):
            raise ValueError("weights must be non-negative")
    if min(sum(t), len(t) - sum(t)) < 2:
        raise ValueError("each arm needs at least two units")
    results: list[CovariateBalance] = []
    for j, label in enumerate(labels):
        means: list[float] = []
        variances: list[float] = []
        for arm in (1, 0):
            values = [row[j] for row, ti in zip(rows, t, strict=True) if ti == arm]
            arm_w = [wi for wi, ti in zip(w, t, strict=True) if ti == arm]
            total = math.fsum(arm_w)
            if total <= 0.0:
                raise ValueError(f"weights of arm T={arm} sum to zero")
            means.append(
                math.fsum(wi * v for wi, v in zip(arm_w, values, strict=True)) / total
            )
            variances.append(statistics.variance(values))
        pooled = math.sqrt(0.5 * (variances[0] + variances[1]))
        if pooled == 0.0:
            raise ValueError(f"covariate {label!r} is constant within both arms")
        results.append(
            CovariateBalance(label, means[0], means[1], (means[0] - means[1]) / pooled)
        )
    return BalanceTable(tuple(results), weights is not None)
