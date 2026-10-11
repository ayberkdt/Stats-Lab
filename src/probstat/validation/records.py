"""Validation-experiment records and the applicability domain they span.

A :class:`ValidationExperiment` stores what was measured, under which named
conditions, with what measurement uncertainty, and where the data came from.
A :class:`ValidationDomain` is the set of conditions covered by those
experiments. Validation evidence only speaks directly about predictions inside
that domain; :meth:`ValidationDomain.check` reports bounding-box membership,
convex-hull membership, and a scaled distance to the nearest validation point
so that extrapolation is flagged rather than silently accepted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_finite_real,
)

_HULL_TOLERANCE = 1.0e-9
_PIVOT_TOLERANCE = 1.0e-12


def _require_text(value: object, *, name: str) -> str:
    """Validate and strip a required human-readable text field.

    Args:
        value: Candidate text.
        name: Field name used in error messages.

    Returns:
        Stripped non-empty text.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If the stripped text is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _freeze_conditions(conditions: Mapping[str, float]) -> Mapping[str, float]:
    """Validate a condition mapping and return a read-only copy.

    Args:
        conditions: Mapping from condition name to finite value.

    Returns:
        Read-only mapping with float values.

    Raises:
        TypeError: If the argument is not a mapping or a value is not real.
        ValueError: If a name is empty or a value is non-finite.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapter 12]. No equation is evaluated.

    """
    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping")
    frozen: dict[str, float] = {}
    for name, value in conditions.items():
        key = _require_text(name, name="condition name")
        frozen[key] = require_finite_real(value, name=f"conditions[{key!r}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ValidationExperiment:
    """Measured values of one quantity of interest under named conditions.

    Args:
        name: Non-empty identifier of the experiment.
        observations: Measured QoI values (one value or replicate samples).
        conditions: Named deterministic settings (load, temperature, ...)
            under which the measurements were taken; they are passed to the
            model as fixed inputs.
        measurement_uncertainty: Standard uncertainty of a single measurement
            (same unit as the QoI), ``>= 0``.
        provenance: Non-empty description of where the data came from
            (facility, campaign, report, date).
        data_uncertainty: Optional standard uncertainty ``u_D`` of the
            experimental value ``D`` (the observation mean). When omitted it
            defaults to ``measurement_uncertainty / sqrt(n)``, which assumes
            independent random measurement errors and no shared systematic
            error.

    Raises:
        TypeError: If a field has an invalid type.
        ValueError: If text is empty, observations are empty or non-finite,
            or an uncertainty is negative.

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV], [REF-ASME-VV20-2009].

    """

    name: str
    observations: tuple[float, ...]
    conditions: Mapping[str, float] = field(default_factory=dict)
    measurement_uncertainty: float = 0.0
    provenance: str = "unspecified"
    data_uncertainty: float | None = None

    def __post_init__(self) -> None:
        """Validate and freeze every field.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapter 12]. No equation is evaluated.

        """
        object.__setattr__(self, "name", _require_text(self.name, name="name"))
        if isinstance(self.observations, str | bytes):
            raise TypeError("observations must be a sequence of real numbers")
        object.__setattr__(
            self,
            "observations",
            coerce_finite_sample(self.observations, name="observations"),
        )
        object.__setattr__(self, "conditions", _freeze_conditions(self.conditions))
        object.__setattr__(
            self,
            "measurement_uncertainty",
            require_non_negative_finite_real(
                self.measurement_uncertainty, name="measurement_uncertainty"
            ),
        )
        object.__setattr__(
            self, "provenance", _require_text(self.provenance, name="provenance")
        )
        if self.data_uncertainty is not None:
            object.__setattr__(
                self,
                "data_uncertainty",
                require_non_negative_finite_real(
                    self.data_uncertainty, name="data_uncertainty"
                ),
            )

    @property
    def size(self) -> int:
        """Return the number of measured values.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapter 12].

        """
        return len(self.observations)

    @property
    def mean(self) -> float:
        """Return the experimental value ``D`` (mean of the observations).

        Equation:
            [EQ-VAL-ASME-001].

        References:
            [REF-ASME-VV20-2009].

        """
        return math.fsum(self.observations) / len(self.observations)

    @property
    def data_standard_uncertainty(self) -> float:
        """Return ``u_D``, the standard uncertainty of the experimental value.

        Returns:
            ``data_uncertainty`` when declared, otherwise
            ``measurement_uncertainty / sqrt(n)``.

        Equation:
            [EQ-VAL-ASME-001].

        References:
            [REF-ASME-VV20-2009], [REF-JCGM-100-2008, Section 4.2].

        """
        if self.data_uncertainty is not None:
            return self.data_uncertainty
        return self.measurement_uncertainty / math.sqrt(len(self.observations))


@dataclass(frozen=True, slots=True)
class DomainCheck:
    """Where one prediction condition lies relative to the validation domain.

    Args:
        condition: The checked prediction condition.
        inside_box: Whether every coordinate lies within the validation range.
        inside_hull: Whether the point is a convex combination of validation
            conditions (within a small numerical tolerance).
        nearest_distance: Euclidean distance to the closest validation
            condition after scaling every coordinate by its validation range
            (``inf`` if the condition leaves a dimension that had zero range).
        nearest_experiment: Name of the closest validation experiment.
        distance_threshold: Scaled distance above which the point is flagged.
        extrapolation: ``True`` when the point is outside the hull or farther
            than ``distance_threshold`` from every validation condition.
        note: Human-readable explanation.

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV], [REF-ROY-OBERKAMPF-2011].

    """

    condition: Mapping[str, float]
    inside_box: bool
    inside_hull: bool
    nearest_distance: float
    nearest_experiment: str
    distance_threshold: float
    extrapolation: bool
    note: str


@dataclass(frozen=True, slots=True)
class ValidationDomain:
    """Conditions covered by a set of validation experiments.

    Args:
        names: Ordered condition names (may be empty when the experiments
            declare no condition variables).
        points: One condition vector per experiment, ordered as ``names``.
        labels: Experiment name for each point.

    Raises:
        ValueError: If no point is given, lengths disagree, or names repeat.

    References:
        [REF-OBERKAMPF-ROY-2025, Part IV], [REF-ROY-OBERKAMPF-2011].

    """

    names: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the shape of the domain description.

        References:
            [EQ-VAL-DOM-001]. No equation is evaluated.

        """
        names = tuple(_require_text(name, name="names") for name in self.names)
        if len(set(names)) != len(names):
            raise ValueError("condition names must be distinct")
        points = tuple(
            tuple(require_finite_real(v, name="points") for v in point)
            for point in self.points
        )
        if not points:
            raise ValueError("a validation domain needs at least one point")
        for point in points:
            if len(point) != len(names):
                raise ValueError("every point must have one value per name")
        labels = tuple(_require_text(label, name="labels") for label in self.labels)
        if len(labels) != len(points):
            raise ValueError("labels must have one entry per point")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_experiments(
        cls, experiments: Iterable[ValidationExperiment]
    ) -> ValidationDomain:
        """Build the domain spanned by the conditions of validation experiments.

        Args:
            experiments: Experiments that all declare the same condition names.

        Returns:
            Domain with names sorted alphabetically.

        Raises:
            TypeError: If an item is not a :class:`ValidationExperiment`.
            ValueError: If no experiment is given or condition names differ.

        Equation:
            [EQ-VAL-DOM-001].

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        items = tuple(experiments)
        if not items:
            raise ValueError("at least one experiment is required")
        for index, item in enumerate(items):
            if not isinstance(item, ValidationExperiment):
                raise TypeError(f"experiments[{index}] must be a ValidationExperiment")
        names = tuple(sorted(items[0].conditions))
        for item in items[1:]:
            if tuple(sorted(item.conditions)) != names:
                raise ValueError(
                    "all experiments must declare the same condition names; "
                    f"{item.name!r} differs from {items[0].name!r}"
                )
        points = tuple(tuple(item.conditions[name] for name in names) for item in items)
        return cls(names, points, tuple(item.name for item in items))

    @property
    def dimension(self) -> int:
        """Return the number of condition variables.

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        return len(self.names)

    @property
    def lower(self) -> tuple[float, ...]:
        """Return the componentwise minimum of the validation conditions.

        Equation:
            [EQ-VAL-DOM-001].

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        return tuple(min(p[i] for p in self.points) for i in range(self.dimension))

    @property
    def upper(self) -> tuple[float, ...]:
        """Return the componentwise maximum of the validation conditions.

        Equation:
            [EQ-VAL-DOM-001].

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        return tuple(max(p[i] for p in self.points) for i in range(self.dimension))

    def describe(self) -> str:
        """Return a one-line description of the validation domain.

        Returns:
            Text such as ``"load in [1, 3]; temp in [290, 310] (4 points)"``.

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV]. No equation is evaluated.

        """
        count = len(self.points)
        if not self.names:
            return f"no condition variables declared ({count} experiment(s))"
        parts = [
            f"{name} in [{low:.6g}, {high:.6g}]"
            for name, low, high in zip(self.names, self.lower, self.upper, strict=True)
        ]
        return "; ".join(parts) + f" ({count} validation point(s))"

    def check(
        self, condition: Mapping[str, float], *, distance_threshold: float = 0.25
    ) -> DomainCheck:
        """Locate a prediction condition relative to the validation domain.

        Coordinates are scaled to ``[0, 1]`` by the validation range of each
        condition. Convex-hull membership is decided exactly (up to
        ``1e-9`` in scaled units) by a phase-one simplex feasibility problem.

        Args:
            condition: Mapping with a finite value for every domain name
                (extra names are ignored).
            distance_threshold: Non-negative scaled distance above which an
                in-hull point is still flagged as sparse extrapolation.

        Returns:
            Domain check with box, hull, distance, and a textual note.

        Raises:
            KeyError: If a domain name is missing from ``condition``.
            ValueError: If a value or the threshold is invalid.

        Equation:
            [EQ-VAL-DOM-001].

        References:
            [REF-OBERKAMPF-ROY-2025, Part IV],
            [REF-NOCEDAL-WRIGHT-2006, Chapter 13].

        """
        threshold = require_non_negative_finite_real(
            distance_threshold, name="distance_threshold"
        )
        frozen = _freeze_conditions(condition)
        target = tuple(frozen[name] for name in self.names)
        if not self.names:
            return DomainCheck(
                frozen,
                True,
                True,
                0.0,
                self.labels[0],
                threshold,
                False,
                "no condition variables declared; domain membership is trivial",
            )
        lower, upper = self.lower, self.upper
        scaled_points: list[list[float]] = [[] for _ in self.points]
        scaled_target: list[float] = []
        degenerate_violation = False
        inside_box = True
        for i, (low, high) in enumerate(zip(lower, upper, strict=True)):
            width = high - low
            slack = _HULL_TOLERANCE * max(1.0, abs(low), abs(high))
            if target[i] < low - slack or target[i] > high + slack:
                inside_box = False
            if width <= 0.0:
                if abs(target[i] - low) > slack:
                    degenerate_violation = True
                continue
            for row, point in zip(scaled_points, self.points, strict=True):
                row.append((point[i] - low) / width)
            scaled_target.append((target[i] - low) / width)
        if degenerate_violation:
            distances = [math.inf] * len(self.points)
            inside_hull = False
        else:
            distances = [
                math.sqrt(
                    math.fsum(
                        (a - b) ** 2 for a, b in zip(row, scaled_target, strict=True)
                    )
                )
                for row in scaled_points
            ]
            inside_hull = inside_box and (
                not scaled_target
                or _convex_hull_residual(
                    tuple(tuple(row) for row in scaled_points), tuple(scaled_target)
                )
                <= _HULL_TOLERANCE
            )
        nearest = min(range(len(distances)), key=distances.__getitem__)
        distance = distances[nearest]
        extrapolation = (not inside_hull) or distance > threshold
        if not inside_box:
            note = "outside the validation bounding box: extrapolation"
        elif not inside_hull:
            note = "inside the bounding box but outside the convex hull: extrapolation"
        elif distance > threshold:
            note = (
                f"inside the convex hull but {distance:.3g} (scaled) from the nearest "
                f"validation point (> {threshold:.3g}): sparse-coverage interpolation"
            )
        else:
            note = "inside the convex hull and near validation points"
        return DomainCheck(
            frozen,
            inside_box,
            inside_hull,
            distance,
            self.labels[nearest],
            threshold,
            extrapolation,
            note,
        )


def _convex_hull_residual(
    points: Sequence[Sequence[float]], target: Sequence[float]
) -> float:
    """Return the phase-one simplex residual of ``target = sum lambda_i p_i``.

    Solves ``min sum(a)`` subject to ``P lambda + a = x``, ``1' lambda + a_0 =
    1``, ``lambda, a >= 0`` with Bland's anti-cycling rule. A zero optimum
    means ``x`` lies in the convex hull of the points.

    Args:
        points: Scaled validation points, each of length ``d``.
        target: Scaled query point of length ``d``.

    Returns:
        Non-negative optimal sum of artificial variables.

    Raises:
        RuntimeError: If the pivot limit is reached (numerical cycling).

    Equation:
        [EQ-VAL-DOM-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 13].

    """
    n = len(points)
    m = len(target) + 1
    rows: list[list[float]] = []
    for i in range(m):
        if i < m - 1:
            coefficients = [float(point[i]) for point in points]
            rhs = float(target[i])
        else:
            coefficients = [1.0] * n
            rhs = 1.0
        if rhs < 0.0:
            coefficients = [-value for value in coefficients]
            rhs = -rhs
        artificial = [1.0 if k == i else 0.0 for k in range(m)]
        rows.append(coefficients + artificial + [rhs])
    width = n + m
    objective = [-math.fsum(rows[i][j] for i in range(m)) for j in range(n)]
    objective += [0.0] * m
    objective.append(-math.fsum(row[-1] for row in rows))
    basis = [n + i for i in range(m)]
    for _ in range(50 * (width + 1)):
        entering = next(
            (j for j in range(width) if objective[j] < -_PIVOT_TOLERANCE), None
        )
        if entering is None:
            return max(0.0, -objective[-1])
        best_row = -1
        best_ratio = math.inf
        for i, row in enumerate(rows):
            if row[entering] > _PIVOT_TOLERANCE:
                ratio = row[-1] / row[entering]
                tied = abs(ratio - best_ratio) <= _PIVOT_TOLERANCE
                if ratio < best_ratio - _PIVOT_TOLERANCE or (
                    tied and basis[i] < basis[best_row]
                ):
                    best_ratio = ratio
                    best_row = i
        if best_row < 0:
            return max(0.0, -objective[-1])
        pivot_row = rows[best_row]
        pivot = pivot_row[entering]
        rows[best_row] = [value / pivot for value in pivot_row]
        pivot_row = rows[best_row]
        for i, row in enumerate(rows):
            if i != best_row and row[entering] != 0.0:
                factor = row[entering]
                rows[i] = [a - factor * b for a, b in zip(row, pivot_row, strict=True)]
        factor = objective[entering]
        objective = [a - factor * b for a, b in zip(objective, pivot_row, strict=True)]
        basis[best_row] = entering
    raise RuntimeError("convex-hull simplex reached its pivot limit")
