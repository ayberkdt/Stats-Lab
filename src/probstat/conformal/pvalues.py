"""Conformal p-values and outlier detection with false-discovery-rate control.

Given calibration scores ``s_1, ..., s_n`` from inliers (larger scores mean
"more unusual") and a test score ``s``, the conformal p-value is

``p = (1 + #{i : s_i >= s}) / (n + 1)``.

If the test point is an inlier exchangeable with the calibration points then
``P(p <= u) <= u`` for every ``u``: the p-value is marginally super-uniform,
and exactly uniform on ``{1/(n+1), ..., 1}`` when scores are almost surely
distinct. "Marginally" means on average over the calibration set; given one
calibration set the p-values are not valid in general.

Several test points share one calibration set, so their p-values are
dependent. Bates, Candes, Lei, Romano and Sesia (2023) show the dependence
is positive (PRDS), hence the Benjamini-Hochberg procedure applied to
conformal p-values controls the false discovery rate at level ``alpha``,
again marginally over the calibration set.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.inference import adjust_p_values


def conformal_p_values(
    calibration_scores: Iterable[float], test_scores: Iterable[float]
) -> tuple[float, ...]:
    """Return ``(1 + #{i : s_i >= s_j}) / (n + 1)`` for each test score.

    Larger scores must indicate stronger departure from the calibration
    (inlier) population. The score function must not have been fitted on
    the calibration points.

    Args:
        calibration_scores: Finite scores of ``n >= 1`` inliers.
        test_scores: Finite scores of the test points.

    Returns:
        One p-value in ``{1/(n+1), ..., 1}`` per test score.

    Raises:
        TypeError: If a score is not real.
        ValueError: If a score is not finite or there is no calibration
            score.

    Equation:
        [EQ-CONF-009].

    References:
        [REF-BATES-ETAL-2023], [REF-VOVK-GAMMERMAN-SHAFER-2005].

    """
    ordered = sorted(
        require_finite_real(score, name="calibration score")
        for score in calibration_scores
    )
    n = len(ordered)
    if n == 0:
        raise ValueError("at least one calibration score is required")
    return tuple(
        (1 + n - bisect.bisect_left(ordered, require_finite_real(s, name="score")))
        / (n + 1)
        for s in test_scores
    )


@dataclass(frozen=True, slots=True)
class ConformalOutlierResult:
    """Conformal p-values and Benjamini-Hochberg outlier decisions.

    Args:
        p_values: Conformal p-values in test order.
        adjusted_p_values: Benjamini-Hochberg adjusted p-values.
        rejected: Whether each test point is declared an outlier.
        alpha: Target false discovery rate.
        calibration_size: Number ``n`` of calibration scores.

    References:
        [EQ-CONF-009], [EQ-CONF-010], [REF-BATES-ETAL-2023].

    """

    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    calibration_size: int

    @property
    def rejection_count(self) -> int:
        """Return the number of declared outliers.

        References:
            [EQ-CONF-010]. No equation is evaluated.

        """
        return sum(self.rejected)

    @property
    def smallest_attainable_p_value(self) -> float:
        """Return ``1/(n + 1)``, the resolution of the conformal p-values.

        With ``m`` test points nothing can be rejected unless
        ``1/(n + 1) <= alpha``, and a lone outlier among ``m`` is found only
        if ``1/(n + 1) <= alpha/m``.

        References:
            [EQ-CONF-009], [REF-BATES-ETAL-2023].

        """
        return 1.0 / (self.calibration_size + 1)


def conformal_outlier_detection(
    calibration_scores: Iterable[float],
    test_scores: Iterable[float],
    *,
    alpha: float = 0.1,
) -> ConformalOutlierResult:
    """Flag outliers by Benjamini-Hochberg on conformal p-values.

    When the calibration scores and the scores of the true inliers among
    the test points are exchangeable (and the test points are otherwise
    independent), the expected proportion of inliers among the flagged
    points is at most ``alpha``. The expectation is marginal over the
    calibration set: for one fixed calibration set the false discovery rate
    can exceed ``alpha``.

    Args:
        calibration_scores: Finite scores of ``n >= 1`` known inliers.
        test_scores: Finite scores of at least one test point.
        alpha: Target false discovery rate in ``(0, 1)``.

    Returns:
        p-values, adjusted p-values, and decisions in test order.

    Raises:
        TypeError: If a score is not real.
        ValueError: If a score is not finite, a score set is empty, or
            ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-009], [EQ-CONF-010].

    References:
        [REF-BATES-ETAL-2023], [REF-BENJAMINI-HOCHBERG-1995].

    """
    calibration = tuple(calibration_scores)
    p_values = conformal_p_values(calibration, test_scores)
    if not p_values:
        raise ValueError("at least one test score is required")
    adjusted = adjust_p_values(p_values, method="bh", alpha=alpha)
    return ConformalOutlierResult(
        p_values=p_values,
        adjusted_p_values=adjusted.adjusted_p_values,
        rejected=adjusted.rejected,
        alpha=adjusted.alpha,
        calibration_size=len(calibration),
    )
