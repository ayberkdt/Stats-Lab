"""Local and variance-based global sensitivity analysis."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class LocalSensitivityResult:
    """Central finite-difference gradient and model-evaluation cost.

    References:
        [REF-SALTELLI-2008, Chapter 1].

    """

    point: tuple[float, ...]
    gradient: tuple[float, ...]
    steps: tuple[float, ...]
    evaluations: int

    def __post_init__(self) -> None:
        """Validate aligned finite vectors and exact evaluation accounting.

        Raises:
            TypeError: If fields contain invalid types.
            ValueError: If vector dimensions or evaluation counts disagree.

        References:
            [EQ-SENS-LOCAL-001], [REF-FORNBERG-1988].

        """
        if not self.point or not (
            len(self.point) == len(self.gradient) == len(self.steps)
        ):
            raise ValueError("point, gradient, and steps must align and be non-empty")
        point = _finite_tuple(self.point, "point")
        gradient = _finite_tuple(self.gradient, "gradient")
        steps = _finite_tuple(self.steps, "steps")
        if any(step <= 0.0 for step in steps):
            raise ValueError("steps must be positive")
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        if evaluations != 2 * len(point):
            raise ValueError("evaluations must equal two per coordinate")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "evaluations", evaluations)


@dataclass(frozen=True, slots=True)
class SensitivityDecision:
    """Transparent ranking and reduction advice derived from total effects.

    References:
        [REF-SALTELLI-2008, Chapters 1 and 6].

    """

    priority_inputs: tuple[int, ...]
    fix_candidates: tuple[int, ...]
    threshold: float

    def __post_init__(self) -> None:
        """Validate disjoint decision groups and a non-negative threshold.

        Raises:
            TypeError: If indices or threshold have invalid types.
            ValueError: If groups overlap, repeat indices, or threshold is negative.

        References:
            [EQ-SENS-DECISION-001], [REF-SALTELLI-2008, Chapter 6].

        """
        threshold = require_finite_real(self.threshold, name="threshold")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        priorities = _index_tuple(self.priority_inputs, "priority_inputs")
        candidates = _index_tuple(self.fix_candidates, "fix_candidates")
        if set(priorities) & set(candidates):
            raise ValueError("priority_inputs and fix_candidates must be disjoint")
        object.__setattr__(self, "priority_inputs", priorities)
        object.__setattr__(self, "fix_candidates", candidates)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True, slots=True)
class SobolResult:
    """Pick-freeze Sobol indices, ranking, and evaluation cost.

    Monte Carlo estimates may lie slightly outside ``[0,1]`` and are retained
    rather than silently clipped.

    References:
        [REF-SOBOL-1993], [REF-SALTELLI-2010].

    """

    first_order: tuple[float, ...]
    total_effect: tuple[float, ...]
    second_order: tuple[tuple[float, ...], ...]
    output_variance: float
    sample_size: int
    evaluations: int

    def __post_init__(self) -> None:
        """Validate dimensions, finite indices, variance, and work metadata.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If dimensions, variance, sample size, or work are invalid.

        References:
            [EQ-SENS-ANOVA-001], [REF-SOBOL-1993], [REF-SALTELLI-2010].

        """
        dimension = len(self.first_order)
        if dimension == 0 or len(self.total_effect) != dimension:
            raise ValueError("first and total indices must align and be non-empty")
        first = _finite_tuple(self.first_order, "first_order")
        total = _finite_tuple(self.total_effect, "total_effect")
        if len(self.second_order) != dimension or any(
            len(row) != dimension for row in self.second_order
        ):
            raise ValueError("second_order must be a square matrix")
        second = tuple(
            _finite_tuple(row, f"second_order[{index}]")
            for index, row in enumerate(self.second_order)
        )
        variance = require_finite_real(self.output_variance, name="output_variance")
        if variance <= 0.0:
            raise ValueError("output_variance must be positive")
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        if sample_size < 2 or evaluations < sample_size:
            raise ValueError("sample_size and evaluations are inconsistent")
        object.__setattr__(self, "first_order", first)
        object.__setattr__(self, "total_effect", total)
        object.__setattr__(self, "second_order", second)
        object.__setattr__(self, "output_variance", variance)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "evaluations", evaluations)

    def ranking(self) -> tuple[int, ...]:
        """Return input indices in descending total-effect order.

        References:
            [EQ-SENS-TOTAL-001], [REF-SALTELLI-2010].

        """
        return tuple(
            sorted(
                range(len(self.total_effect)),
                key=self.total_effect.__getitem__,
                reverse=True,
            )
        )

    def negligible_inputs(self, *, threshold: float = 0.01) -> tuple[int, ...]:
        """Return inputs whose estimated total effect does not exceed a threshold.

        References:
            [EQ-SENS-TOTAL-001], [REF-SALTELLI-2008, Chapter 1].

        """
        checked = require_finite_real(threshold, name="threshold")
        if checked < 0.0:
            raise ValueError("threshold must be non-negative")
        return tuple(i for i, value in enumerate(self.total_effect) if value <= checked)

    def decision(self, *, threshold: float = 0.01) -> SensitivityDecision:
        """Separate measurement priorities from dimension-reduction candidates.

        Inputs above the total-effect threshold are ordered as candidates for
        uncertainty reduction; inputs at or below it are candidates for fixing.
        This is a screening rule, not a claim of causal or economic optimality.

        References:
            [EQ-SENS-DECISION-001], [REF-SALTELLI-2008, Chapter 6].

        """
        checked = require_finite_real(threshold, name="threshold")
        if checked < 0.0:
            raise ValueError("threshold must be non-negative")
        candidates = self.negligible_inputs(threshold=checked)
        candidate_set = set(candidates)
        priorities = tuple(i for i in self.ranking() if i not in candidate_set)
        return SensitivityDecision(priorities, candidates, checked)


def central_difference_gradient(
    model: Callable[[tuple[float, ...]], float],
    point: Sequence[object],
    steps: Sequence[object],
) -> LocalSensitivityResult:
    """Estimate a local gradient with coordinatewise central differences.

    Args:
        model: Deterministic scalar model.
        point: Finite evaluation vector.
        steps: Matching finite positive perturbations.

    Returns:
        Gradient and exact count of two model evaluations per coordinate.

    Equation:
        [EQ-SENS-LOCAL-001].

    References:
        [REF-FORNBERG-1988].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    x = tuple(require_finite_real(v, name=f"point[{i}]") for i, v in enumerate(point))
    h = tuple(require_finite_real(v, name=f"steps[{i}]") for i, v in enumerate(steps))
    if not x or len(x) != len(h):
        raise ValueError("point and steps must be non-empty and equally sized")
    if any(value <= 0.0 for value in h):
        raise ValueError("steps must be positive")
    gradient: list[float] = []
    for i, step in enumerate(h):
        plus = list(x)
        minus = list(x)
        plus[i] += step
        minus[i] -= step
        upper = require_finite_real(model(tuple(plus)), name=f"model(plus {i})")
        lower = require_finite_real(model(tuple(minus)), name=f"model(minus {i})")
        gradient.append((upper - lower) / (2.0 * step))
    return LocalSensitivityResult(x, tuple(gradient), h, 2 * len(x))


def estimate_sobol_indices(
    model: Callable[[tuple[float, ...]], float],
    sample_a: Iterable[Sequence[object]],
    sample_b: Iterable[Sequence[object]],
) -> SobolResult:
    """Estimate first-, second-, and total-effect Sobol indices by pick-freeze.

    Args:
        model: Deterministic scalar model of mutually independent inputs.
        sample_a: Independent base sample matrix.
        sample_b: Equally shaped independent base sample matrix.

    Returns:
        Unclipped indices, output variance, sample size, and full model-call cost.

    Equation:
        [EQ-SENS-ANOVA-001], [EQ-SENS-FIRST-001], [EQ-SENS-SECOND-001],
        [EQ-SENS-TOTAL-001].

    References:
        [REF-SOBOL-1993], [REF-SALTELLI-2010].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    a = _matrix(sample_a, "sample_a")
    b = _matrix(sample_b, "sample_b")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("sample matrices must have equal shape")
    if len(a) < 2:
        raise ValueError("at least two rows are required")
    n, dimension = len(a), len(a[0])
    ya = tuple(_evaluate(model, row) for row in a)
    yb = tuple(_evaluate(model, row) for row in b)
    variance = statistics.pvariance(ya + yb)
    if variance == 0.0:
        raise ValueError("Sobol indices require positive output variance")
    first: list[float] = []
    total: list[float] = []
    for i in range(dimension):
        values = tuple(
            _evaluate(model, a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a, b, strict=True)
        )
        first.append(
            math.fsum(yb[k] * (values[k] - ya[k]) for k in range(n)) / n / variance
        )
        total.append(
            0.5 * math.fsum((ya[k] - values[k]) ** 2 for k in range(n)) / n / variance
        )
    second = [[0.0] * dimension for _ in range(dimension)]
    pair_evaluations = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            values = tuple(
                _evaluate(
                    model,
                    tuple(
                        b_row[k] if k in (i, j) else a_row[k] for k in range(dimension)
                    ),
                )
                for a_row, b_row in zip(a, b, strict=True)
            )
            closed = (
                math.fsum(yb[k] * (values[k] - ya[k]) for k in range(n)) / n / variance
            )
            second[i][j] = second[j][i] = closed - first[i] - first[j]
            pair_evaluations += n
    evaluations = n * (2 + dimension) + pair_evaluations
    return SobolResult(
        tuple(first),
        tuple(total),
        tuple(tuple(row) for row in second),
        variance,
        n,
        evaluations,
    )


def _matrix(
    rows: Iterable[Sequence[object]], name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate a non-empty rectangular finite matrix.

    References:
        [REF-SALTELLI-2010]. No sensitivity equation is evaluated.

    """
    result = tuple(
        tuple(
            require_finite_real(v, name=f"{name}[{i}][{j}]") for j, v in enumerate(row)
        )
        for i, row in enumerate(rows)
    )
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return result


def _evaluate(
    model: Callable[[tuple[float, ...]], float], row: tuple[float, ...]
) -> float:
    """Evaluate one finite scalar model response.

    References:
        [REF-SALTELLI-2010]. No new equation is introduced.

    """
    return require_finite_real(model(row), name="model output")


def _finite_tuple(values: Iterable[object], name: str) -> tuple[float, ...]:
    """Return a validated tuple of finite real values.

    References:
        [REF-PYTHON-MATH]. No sensitivity equation is evaluated.

    """
    return tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _index_tuple(values: Iterable[object], name: str) -> tuple[int, ...]:
    """Return distinct non-negative integer indices in declared order.

    References:
        [REF-PYTHON-DATAMODEL]. No sensitivity equation is evaluated.

    """
    result = tuple(
        require_non_negative_integer(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not repeat indices")
    return result
