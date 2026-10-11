"""Empirical coverage checks for prediction intervals on held-out test data.

A coverage check counts how many test responses fall inside their intervals.
The count is binomial when test points are independent given the intervals,
so a Clopper-Pearson interval quantifies how precisely the coverage rate has
been measured. Agreement with ``1 - alpha`` on one test set does not prove
conditional coverage; inspect coverage within subgroups for that.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.inference import ConfidenceInterval, clopper_pearson_interval

from .regression import PredictionInterval


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Empirical coverage of prediction intervals on a test set.

    Args:
        covered: Number of responses inside their intervals.
        size: Number of test points.
        coverage: ``covered / size``.
        coverage_interval: Clopper-Pearson interval for the coverage rate.
        mean_width: Mean finite interval width (``inf`` if any is infinite).
        median_width: Median interval width.
        nominal: Target coverage ``1 - alpha``, or ``nan`` if not given.

    References:
        [EQ-CONF-005], [REF-CLOPPER-PEARSON-1934],
        [REF-ANGELOPOULOS-BATES-2023].

    """

    covered: int
    size: int
    coverage: float
    coverage_interval: ConfidenceInterval
    mean_width: float
    median_width: float
    nominal: float

    @property
    def consistent_with_nominal(self) -> bool:
        """Return whether the nominal level lies inside the coverage interval.

        References:
            [REF-CLOPPER-PEARSON-1934]. No equation is evaluated.

        """
        if math.isnan(self.nominal):
            return False
        interval = self.coverage_interval
        return interval.lower <= self.nominal <= interval.upper


def empirical_coverage(
    intervals: Sequence[PredictionInterval],
    responses: Sequence[float],
    *,
    nominal: float | None = None,
    confidence: float = 0.95,
) -> CoverageReport:
    """Measure the empirical coverage of intervals on test responses.

    Args:
        intervals: One prediction interval per test point.
        responses: Observed test responses.
        nominal: Optional target coverage in ``(0, 1)``.
        confidence: Level of the Clopper-Pearson interval for the rate.

    Returns:
        Coverage count, rate, its exact binomial interval, and widths.

    Raises:
        TypeError: If an interval has the wrong type.
        ValueError: If lengths differ, the test set is empty, or a level is
            outside ``(0, 1)``.

    Equation:
        [EQ-CONF-005].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-ANGELOPOULOS-BATES-2023].

    """
    boxes = tuple(intervals)
    values = tuple(require_finite_real(y, name="responses") for y in responses)
    if not boxes:
        raise ValueError("at least one test point is required")
    if len(boxes) != len(values):
        raise ValueError("intervals and responses must have equal length")
    for index, box in enumerate(boxes):
        if not isinstance(box, PredictionInterval):
            raise TypeError(f"intervals[{index}] must be a PredictionInterval")
    target = math.nan
    if nominal is not None:
        target = require_finite_real(nominal, name="nominal")
        if not 0.0 < target < 1.0:
            raise ValueError("nominal must lie strictly inside (0, 1)")
    covered = sum(1 for box, y in zip(boxes, values, strict=True) if box.contains(y))
    widths = [box.width for box in boxes]
    mean_width = (
        math.inf
        if any(math.isinf(width) for width in widths)
        else math.fsum(widths) / len(widths)
    )
    return CoverageReport(
        covered,
        len(boxes),
        covered / len(boxes),
        clopper_pearson_interval(covered, len(boxes), confidence=confidence),
        mean_width,
        statistics.median(widths),
        target,
    )
