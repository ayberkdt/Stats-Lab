"""Tabular uncertainty budgets, combination rules, and GUM Supplement 1 propagation.

System budgets (a link budget in dB, a pointing budget, a perturbation
budget, a mass budget) list contributions to one output, each with a standard
uncertainty ``u_i``, a sensitivity coefficient ``c_i``, an evaluation type
(GUM Type A or B), degrees of freedom, and often a worst-case half-width.
This module holds that table and combines it by explicit, named rules:

- :class:`BudgetEntry` and :class:`UncertaintyBudget`: the table, the law of
  propagation of uncertainty with correlation terms, Welch-Satterthwaite
  effective degrees of freedom, the expanded uncertainty ``U = k u_c`` with a
  Student-t coverage factor, a contribution table, subsets, and Markdown or
  plain-text rendering.
- Combination rules as separate methods: ``law_of_propagation`` (GUM),
  ``root_sum_square`` (uncorrelated), ``worst_case_sum`` (linear sum of
  absolute contributions) and ``mixed_rule`` (biases linearly, random terms
  in root-sum-square, then summed). Each returns a
  :class:`CombinedUncertainty` that records the rule and whether the number is
  a *standard uncertainty* or a *worst-case bound*
  (:class:`QuantityKind`). The record defines no arithmetic, so a standard
  uncertainty and a bound cannot be added or compared by accident.
- :func:`budget_from_model`: sensitivity coefficients by central differences
  of a model function, with an optional second-order check that flags when
  the linearisation is poor.
- :func:`monte_carlo_budget` and :func:`compare_with_law_of_propagation`:
  propagation of distributions by the Monte Carlo method of JCGM 101:2008
  and that Supplement's validation of the GUM uncertainty framework.
- :func:`adaptive_monte_carlo_budget` (and the batch-contract form
  :func:`adaptive_monte_carlo_budget_procedure`): the adaptive procedure of
  clause 7.9.4, which adds batches of trials until the estimate, ``u(y)`` and
  the interval endpoints have stabilised to a numerical tolerance, and
  reports honestly when it has not.

Relation to neighbouring tools: :class:`~probstat.uq.NumericalErrorBudget`
holds absolute bounds combined by the triangle inequality and no sensitivity
coefficients; ``budget_standard_uncertainty`` in ``probstat.validation``
turns one such bound into a rectangular standard uncertainty (this module's
:meth:`BudgetEntry.rectangular` is the same conversion for any entry).
:func:`probstat.multivariate.delta_method` propagates a mean and a full
covariance matrix through a vector map; a budget is the scalar case with the
covariance written as standard uncertainties and a correlation matrix, plus
the metrological bookkeeping (types, degrees of freedom, expansion).

Assumptions: the law of propagation is a first-order Taylor expansion, so it
is exact only for linear models. Welch-Satterthwaite assumes uncorrelated
inputs; see :meth:`UncertaintyBudget.effective_degrees_of_freedom` for what
is done when a correlation matrix is present. ``correlation`` always means
the correlation of the *input estimates* ``r(x_i, x_j)`` of GUM 5.2.2. In
:func:`monte_carlo_budget` it is instead the correlation matrix of the
Gaussian copula (normal scores) joining the stated marginals; it equals the
Pearson correlation only for normal marginals.

Sources opened for this module: JCGM 100:2008 (GUM) and JCGM 101:2008
(Supplement 1), both read from the BIPM PDFs; clause numbers quoted below
come from those texts. JCGM 102:2011 (multivariate outputs) was not opened.

References:
    [REF-JCGM-100-2008], [REF-JCGM-101-2008], [REF-WELCH-1947],
    [REF-SATTERTHWAITE-1946], [REF-NELSEN-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from probstat.distributions import Normal, StudentT
from probstat.foundations.batch import (
    EvaluationRequest,
    Procedure,
    batch_from_pointwise,
    run_procedure,
    scalar_outputs,
)
from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import cholesky_semidefinite
from probstat.optimization.finite_differences import central_gradient, central_hessian

EVALUATION_TYPES = ("A", "B")
NATURES = ("aleatory", "epistemic")
KINDS = ("random", "bias")
HALF_WIDTH_BASIS = "half_width"
STANDARD_BASIS = "standard_uncertainty"
BASES = (HALF_WIDTH_BASIS, STANDARD_BASIS)
COVERAGE_INTERVALS = ("symmetric", "shortest")
ADEQUACY_STABILIZATION = "stabilisation statistics (JCGM 101, 7.9.4 and 8.2)"
ADEQUACY_PROXY = "standard-error proxy (fixed number of trials)"
ADEQUACY_OTHER_INTERVAL = "not assessed: the adaptive run tracked the other interval"
_CORRELATION_TOLERANCE = 1e-12
_PROBABILITY_FLOOR = 1e-16

RULE_LAW_OF_PROPAGATION = "law of propagation of uncertainty (GUM 5.2.2)"
RULE_ROOT_SUM_SQUARE = "root-sum-square (uncorrelated)"
RULE_WORST_CASE_SUM = "linear sum of absolute contributions"
RULE_MIXED = "biases linearly, random terms root-sum-square, then summed"


class QuantityKind(StrEnum):
    """What a combined number is, so that different kinds are never mixed.

    ``STANDARD_UNCERTAINTY`` is a standard deviation-like quantity (it may be
    multiplied by a coverage factor). ``WORST_CASE_BOUND`` is a limit on the
    magnitude of the error (it must not be multiplied by a coverage factor or
    added to a standard uncertainty).

    References:
        [REF-JCGM-100-2008, Sections 2.3.1 and 6.2].

    """

    STANDARD_UNCERTAINTY = "standard uncertainty"
    WORST_CASE_BOUND = "worst-case bound"


def _check_choice(value: object, choices: Sequence[str], name: str) -> str:
    """Return ``value`` as text after checking it is one of ``choices``.

    Args:
        value: Candidate value (plain text or a string enumeration member).
        choices: Allowed values.
        name: Argument name for error messages.

    Returns:
        The value as a plain ``str``.

    Raises:
        TypeError: If ``value`` is not text.
        ValueError: If it is not one of ``choices``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    text = str(value)
    if text not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}")
    return text


def _positive_dof(value: object) -> float:
    """Validate degrees of freedom: positive, infinity allowed.

    Args:
        value: Candidate degrees of freedom.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not positive or is ``nan``.

    References:
        [REF-JCGM-100-2008, Annex G.3-G.4]. No mathematical equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dof must be a real number")
    number = float(value)
    if math.isnan(number) or number <= 0.0:
        raise ValueError("dof must be positive (infinity allowed)")
    return number


class _InputLaw(Protocol):
    """Structural type of an input law: sampling and the quantile function.

    References:
        [REF-JCGM-101-2008, Section 7.3]. No mathematical equation is
        evaluated.

    """

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw ``size`` observations.

        References:
            [REF-JCGM-101-2008, Section 7.3]. No mathematical equation is
            evaluated.

        """

    def quantile(self, probability: float) -> float:
        """Return the quantile function at ``probability``.

        References:
            [REF-JCGM-101-2008, Section 7.3]. No mathematical equation is
            evaluated.

        """


@dataclass(frozen=True, slots=True)
class BudgetEntry:
    """One line of an uncertainty budget.

    Args:
        name: Unique label of the contribution.
        u: Standard uncertainty ``u(x_i)`` of the input, in the input's unit.
        c: Sensitivity coefficient ``c_i = df/dx_i`` (may be negative; ``+1``
            or ``-1`` for decibel gains and losses).
        evaluation: ``"A"`` (statistical) or ``"B"`` (other information).
        dof: Degrees of freedom of ``u``; ``math.inf`` when ``u`` is taken as
            exactly known (the usual convention for Type B).
        nature: ``"aleatory"`` or ``"epistemic"``.
        kind: ``"random"`` (varies from use to use) or ``"bias"`` (a fixed
            but unknown offset that does not average out).
        half_width: Optional worst-case half-width ``a`` of the input (a
            bound on ``|error|``), in the input's unit. Needed by the
            worst-case and bound-level mixed rules.
        source: Free text saying where the number comes from.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``u`` or ``half_width`` is negative or non-finite,
            ``c`` is non-finite, ``dof`` is not positive, or a choice field is
            not one of its allowed values.

    Equation:
        [EQ-UQ-BUDGET-001].

    References:
        [REF-JCGM-100-2008, Sections 4.2, 4.3 and 5.1.3].

    """

    name: str
    u: float
    c: float = 1.0
    evaluation: str = "B"
    dof: float = math.inf
    nature: str = "aleatory"
    kind: str = "random"
    half_width: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        """Validate and normalise every field.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a numeric field is out of range or a choice is
                unknown.

        References:
            [REF-JCGM-100-2008, Sections 4.2 and 4.3]. No mathematical
            equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("name must be a non-empty string")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        object.__setattr__(
            self, "u", require_non_negative_finite_real(self.u, name="u")
        )
        object.__setattr__(self, "c", require_finite_real(self.c, name="c"))
        object.__setattr__(self, "dof", _positive_dof(self.dof))
        object.__setattr__(
            self,
            "evaluation",
            _check_choice(self.evaluation, EVALUATION_TYPES, "evaluation"),
        )
        object.__setattr__(
            self, "nature", _check_choice(self.nature, NATURES, "nature")
        )
        object.__setattr__(self, "kind", _check_choice(self.kind, KINDS, "kind"))
        if self.half_width is not None:
            object.__setattr__(
                self,
                "half_width",
                require_non_negative_finite_real(self.half_width, name="half_width"),
            )

    @property
    def contribution(self) -> float:
        """Return the signed contribution ``c u`` to the output.

        Equation:
            [EQ-UQ-BUDGET-001].

        References:
            [REF-JCGM-100-2008, Section 5.1.3].

        """
        return self.c * self.u

    @property
    def variance_contribution(self) -> float:
        """Return ``(c u)^2``, the uncorrelated variance contribution.

        Equation:
            [EQ-UQ-BUDGET-001].

        References:
            [REF-JCGM-100-2008, Section 5.1.3].

        """
        return (self.c * self.u) ** 2

    @property
    def bound_contribution(self) -> float | None:
        """Return ``|c| a`` for the worst-case half-width ``a``, or ``None``.

        Equation:
            [EQ-UQ-BUDGET-004].

        References:
            [REF-JCGM-100-2008, Section 5.1.3]. Worst-case bounds are the
            caller's convention; the Guide does not define them.

        """
        return None if self.half_width is None else abs(self.c) * self.half_width

    @classmethod
    def _from_bound(
        cls,
        name: str,
        half_width: float,
        divisor: float,
        *,
        c: float,
        dof: float,
        nature: str,
        kind: str,
        source: str,
    ) -> BudgetEntry:
        """Build a Type B entry with ``u = a / divisor`` and ``half_width = a``.

        Args:
            name: Entry label.
            half_width: Half-width ``a`` of the stated bounds.
            divisor: ``sqrt(3)``, ``sqrt(6)``, ``sqrt(2)`` or a coverage factor.
            c: Sensitivity coefficient.
            dof: Degrees of freedom.
            nature: Aleatory or epistemic.
            kind: Random or bias.
            source: Provenance text.

        Returns:
            The entry.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Sections 4.3.4, 4.3.7 and 4.3.9].

        """
        a = require_non_negative_finite_real(half_width, name="half_width")
        return cls(
            name,
            a / divisor,
            c=c,
            evaluation="B",
            dof=dof,
            nature=nature,
            kind=kind,
            half_width=a,
            source=source,
        )

    @classmethod
    def rectangular(
        cls,
        name: str,
        half_width: float,
        *,
        c: float = 1.0,
        dof: float = math.inf,
        nature: str = "epistemic",
        kind: str = "bias",
        source: str = "",
    ) -> BudgetEntry:
        """Type B entry for limits ``+/- a`` with no further knowledge.

        ``u = a / sqrt(3)`` (a rectangular law on ``[-a, a]``).

        Args:
            name: Entry label.
            half_width: Half-width ``a >= 0``.
            c: Sensitivity coefficient.
            dof: Degrees of freedom (default infinite).
            nature: Defaults to ``"epistemic"``: knowing only bounds is
                incomplete knowledge.
            kind: Defaults to ``"bias"``.
            source: Provenance text.

        Returns:
            The entry, with ``half_width = a``.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Section 4.3.7],
            [REF-JCGM-101-2008, Section 6.4.2].

        """
        return cls._from_bound(
            name,
            half_width,
            math.sqrt(3.0),
            c=c,
            dof=dof,
            nature=nature,
            kind=kind,
            source=source,
        )

    @classmethod
    def triangular(
        cls,
        name: str,
        half_width: float,
        *,
        c: float = 1.0,
        dof: float = math.inf,
        nature: str = "epistemic",
        kind: str = "bias",
        source: str = "",
    ) -> BudgetEntry:
        """Type B entry for limits ``+/- a`` with values near the centre likelier.

        ``u = a / sqrt(6)`` (a symmetric triangular law on ``[-a, a]``).

        Args:
            name: Entry label.
            half_width: Half-width ``a >= 0``.
            c: Sensitivity coefficient.
            dof: Degrees of freedom (default infinite).
            nature: Defaults to ``"epistemic"``.
            kind: Defaults to ``"bias"``.
            source: Provenance text.

        Returns:
            The entry, with ``half_width = a``.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Section 4.3.9],
            [REF-JCGM-101-2008, Section 6.4.5].

        """
        return cls._from_bound(
            name,
            half_width,
            math.sqrt(6.0),
            c=c,
            dof=dof,
            nature=nature,
            kind=kind,
            source=source,
        )

    @classmethod
    def arcsine(
        cls,
        name: str,
        half_width: float,
        *,
        c: float = 1.0,
        dof: float = math.inf,
        nature: str = "aleatory",
        kind: str = "random",
        source: str = "",
    ) -> BudgetEntry:
        """Type B entry for a quantity cycling sinusoidally between ``+/- a``.

        ``u = a / sqrt(2)`` (the arcsine, or U-shaped, law on ``[-a, a]``).

        Args:
            name: Entry label.
            half_width: Half-width ``a >= 0``.
            c: Sensitivity coefficient.
            dof: Degrees of freedom (default infinite).
            nature: Defaults to ``"aleatory"``.
            kind: Defaults to ``"random"``.
            source: Provenance text.

        Returns:
            The entry, with ``half_width = a``.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-101-2008, Section 6.4.6].

        """
        return cls._from_bound(
            name,
            half_width,
            math.sqrt(2.0),
            c=c,
            dof=dof,
            nature=nature,
            kind=kind,
            source=source,
        )

    @classmethod
    def normal(
        cls,
        name: str,
        half_width: float,
        coverage_factor: float,
        *,
        c: float = 1.0,
        dof: float = math.inf,
        nature: str = "aleatory",
        kind: str = "random",
        source: str = "",
    ) -> BudgetEntry:
        """Type B entry from a stated expanded uncertainty and coverage factor.

        ``u = U / k`` where ``U`` is the quoted expanded uncertainty and ``k``
        its coverage factor (``1.96`` for 95 %, ``2.58`` for 99 % under a
        normal law). The quoted ``U`` is stored as ``half_width``.

        Args:
            name: Entry label.
            half_width: Quoted expanded uncertainty ``U >= 0``.
            coverage_factor: Coverage factor ``k > 0`` of the quotation.
            c: Sensitivity coefficient.
            dof: Degrees of freedom (default infinite).
            nature: Defaults to ``"aleatory"``.
            kind: Defaults to ``"random"``.
            source: Provenance text.

        Returns:
            The entry.

        Raises:
            ValueError: If ``coverage_factor`` is not positive.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Section 4.3.4].

        """
        k = require_finite_real(coverage_factor, name="coverage_factor")
        if k <= 0.0:
            raise ValueError("coverage_factor must be positive")
        return cls._from_bound(
            name, half_width, k, c=c, dof=dof, nature=nature, kind=kind, source=source
        )

    @classmethod
    def from_distribution(
        cls,
        name: str,
        distribution: object,
        *,
        c: float = 1.0,
        dof: float = math.inf,
        nature: str = "aleatory",
        kind: str = "random",
        half_width: float | None = None,
        source: str = "",
    ) -> BudgetEntry:
        """Type B entry whose ``u`` is the standard deviation of a law.

        Args:
            name: Entry label.
            distribution: Any :mod:`probstat.distributions` law with a finite
                standard deviation ``std``.
            c: Sensitivity coefficient.
            dof: Degrees of freedom (default infinite).
            nature: Defaults to ``"aleatory"``.
            kind: Defaults to ``"random"``.
            half_width: Optional worst-case half-width.
            source: Provenance text.

        Returns:
            The entry with ``u = distribution.std``.

        Raises:
            TypeError: If ``distribution`` has no ``std``.
            ValueError: If the standard deviation is not finite.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Section 4.3.1],
            [REF-JCGM-101-2008, Section 6.4].

        """
        spread = getattr(distribution, "std", None)
        if isinstance(spread, bool) or not isinstance(spread, (int, float)):
            raise TypeError("distribution must provide a numeric std")
        if not math.isfinite(spread):
            raise ValueError("the distribution has no finite standard deviation")
        return cls(
            name,
            float(spread),
            c=c,
            evaluation="B",
            dof=dof,
            nature=nature,
            kind=kind,
            half_width=half_width,
            source=source,
        )

    @classmethod
    def from_observations(
        cls,
        name: str,
        observations: Sequence[float],
        *,
        c: float = 1.0,
        nature: str = "aleatory",
        kind: str = "random",
        source: str = "",
    ) -> BudgetEntry:
        """Type A entry: standard uncertainty of the mean of repeated readings.

        ``u = s / sqrt(n)`` with ``s`` the experimental standard deviation
        (divisor ``n - 1``) and ``dof = n - 1``.

        Args:
            name: Entry label.
            observations: At least two finite readings.
            c: Sensitivity coefficient.
            nature: Defaults to ``"aleatory"``.
            kind: Defaults to ``"random"``.
            source: Provenance text.

        Returns:
            The entry (``evaluation = "A"``).

        Raises:
            TypeError: If ``observations`` is not a sequence of reals.
            ValueError: If fewer than two readings are given.

        Equation:
            [EQ-UQ-BUDGET-003].

        References:
            [REF-JCGM-100-2008, Sections 4.2.1-4.2.3 and 4.2.6].

        """
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence of numbers")
        values = [require_finite_real(v, name="observation") for v in observations]
        n = len(values)
        if n < 2:
            raise ValueError("at least two observations are required")
        mean = math.fsum(values) / n
        variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
        return cls(
            name,
            math.sqrt(variance / n),
            c=c,
            evaluation="A",
            dof=float(n - 1),
            nature=nature,
            kind=kind,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class CombinedUncertainty:
    """A combined number together with the rule and kind that produced it.

    There is deliberately no arithmetic on this record: a standard
    uncertainty and a worst-case bound are different quantities, and the
    ``kind`` field says which one ``value`` is.

    Args:
        value: The combined value, in the output's unit.
        rule: Name of the combination rule.
        kind: :class:`QuantityKind` of ``value``.
        entries: Number of budget entries combined.
        coverage_factor: Factor ``k`` applied to random terms when ``kind`` is
            a bound built from standard uncertainties; otherwise ``None``.
        notes: Cautions the rule recorded (ignored correlations, and so on).

    References:
        [REF-JCGM-100-2008, Sections 5.1-5.2].

    """

    value: float
    rule: str
    kind: QuantityKind
    entries: int
    coverage_factor: float | None = None
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return one line naming the value, its kind and its rule.

        References:
            [REF-JCGM-100-2008, Section 5.1]. No mathematical equation is
            evaluated.

        """
        factor = (
            "" if self.coverage_factor is None else f", k = {self.coverage_factor:g}"
        )
        return f"{self.value:.6g} ({self.kind.value}; {self.rule}{factor})"


@dataclass(frozen=True, slots=True)
class Contribution:
    """One row of the contribution table.

    Args:
        name: Entry label.
        c: Sensitivity coefficient.
        u: Standard uncertainty of the input.
        contribution: Signed ``c u``.
        variance_contribution: ``(c u)^2``.
        share: Fractional contribution to the combined variance. With
            correlation it is the covariance apportionment
            ``c_i u_i sum_j c_j u_j r_ij / u_c^2``; the shares always sum to
            one and may be negative or exceed one.
        evaluation: ``"A"`` or ``"B"``.
        nature: ``"aleatory"`` or ``"epistemic"``.
        kind: ``"random"`` or ``"bias"``.
        dof: Degrees of freedom.

    Equation:
        [EQ-UQ-BUDGET-005].

    References:
        [REF-JCGM-100-2008, Sections 5.1.3 and 5.2.2].

    """

    name: str
    c: float
    u: float
    contribution: float
    variance_contribution: float
    share: float
    evaluation: str
    nature: str
    kind: str
    dof: float


@dataclass(frozen=True, slots=True)
class ExpandedUncertainty:
    """Expanded uncertainty ``U = k u_c`` with its coverage factor.

    Args:
        value: ``U``.
        standard_uncertainty: Combined standard uncertainty ``u_c``.
        coverage_factor: ``k``, a Student-t quantile at ``effective_dof``
            (the normal quantile when that is infinite).
        coverage_probability: Requested coverage ``p``.
        effective_dof: Welch-Satterthwaite ``nu_eff`` (as used for ``k``).
        warnings: Cautions, for example that correlations were present.

    Equation:
        [EQ-UQ-BUDGET-002].

    References:
        [REF-JCGM-100-2008, Sections 6.2 and G.4].

    """

    value: float
    standard_uncertainty: float
    coverage_factor: float
    coverage_probability: float
    effective_dof: float
    warnings: tuple[str, ...] = ()

    def interval(self, estimate: float) -> tuple[float, float]:
        """Return ``(y - U, y + U)`` for an output estimate ``y``.

        Args:
            estimate: Output estimate.

        Returns:
            The symmetric interval.

        References:
            [REF-JCGM-100-2008, Section 6.2.1]. No mathematical equation is
            evaluated beyond the sum and difference.

        """
        y = require_finite_real(estimate, name="estimate")
        return (y - self.value, y + self.value)


def _checked_correlation(
    correlation: Sequence[Sequence[float]] | None, size: int
) -> tuple[tuple[float, ...], ...] | None:
    """Validate a correlation matrix of the stated size and return it as tuples.

    Args:
        correlation: ``None`` (independent) or a square matrix.
        size: Required dimension.

    Returns:
        ``None`` or a tuple of tuples; the identity matrix is returned as
        ``None``.

    Raises:
        TypeError: If the entries are not real numbers.
        ValueError: If the matrix has the wrong shape, is not symmetric with
            unit diagonal and entries in ``[-1, 1]``, or is not positive
            semidefinite.

    References:
        [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
        evaluated.

    """
    if correlation is None:
        return None
    rows = tuple(
        tuple(require_finite_real(v, name="correlation") for v in row)
        for row in correlation
    )
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"correlation must be a {size} by {size} matrix")
    for i in range(size):
        if abs(rows[i][i] - 1.0) > _CORRELATION_TOLERANCE:
            raise ValueError("correlation must have a unit diagonal")
        for j in range(size):
            if abs(rows[i][j]) > 1.0 + _CORRELATION_TOLERANCE:
                raise ValueError("correlation entries must lie in [-1, 1]")
            if abs(rows[i][j] - rows[j][i]) > _CORRELATION_TOLERANCE:
                raise ValueError("correlation must be symmetric")
    try:
        cholesky_semidefinite(rows)
    except ValueError as error:
        raise ValueError("correlation must be positive semidefinite") from error
    if all(rows[i][j] == 0.0 for i in range(size) for j in range(size) if i != j):
        return None
    return rows


@dataclass(frozen=True, slots=True)
class UncertaintyBudget:
    """A table of uncertainty contributions to one output quantity.

    The combined standard uncertainty is the law of propagation of
    uncertainty of GUM 5.2.2, ``u_c^2 = sum_i sum_j c_i c_j u_i u_j r_ij``,
    with ``r_ij`` the correlation of the input estimates (``r_ii = 1``).

    Args:
        entries: Non-empty sequence of :class:`BudgetEntry` with unique names.
        correlation: Optional symmetric positive-semidefinite correlation
            matrix of the input estimates in entry order, unit diagonal.
            ``None``, or a matrix with zero off-diagonals, means independent
            inputs.

    Raises:
        TypeError: If an entry is not a :class:`BudgetEntry`.
        ValueError: If entries are empty or repeat a name, or the
            correlation matrix is invalid.

    Equation:
        [EQ-UQ-BUDGET-001].

    References:
        [REF-JCGM-100-2008, Sections 5.1.2 and 5.2.2].

    """

    entries: Sequence[BudgetEntry]
    correlation: Sequence[Sequence[float]] | None = None

    def __post_init__(self) -> None:
        """Validate entries and correlation, normalising both to tuples.

        Raises:
            TypeError: If an entry is not a :class:`BudgetEntry`.
            ValueError: If entries are empty or repeat a name, or the
                correlation is invalid.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
            evaluated.

        """
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("a budget needs at least one entry")
        if not all(isinstance(entry, BudgetEntry) for entry in entries):
            raise TypeError("entries must be BudgetEntry instances")
        names = [entry.name for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError("entry names must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self, "correlation", _checked_correlation(self.correlation, len(entries))
        )

    def __len__(self) -> int:
        """Return the number of entries.

        References:
            [REF-JCGM-100-2008, Section 5.1]. No mathematical equation is
            evaluated.

        """
        return len(self.entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the entry names in table order.

        References:
            [REF-JCGM-100-2008, Section 5.1]. No mathematical equation is
            evaluated.

        """
        return tuple(entry.name for entry in self.entries)

    @property
    def has_correlation(self) -> bool:
        """Return whether any off-diagonal correlation is non-zero.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
            evaluated.

        """
        return self.correlation is not None

    def _correlation_row(self, i: int) -> tuple[float, ...]:
        """Return row ``i`` of the correlation matrix (identity if absent).

        Args:
            i: Row index.

        Returns:
            The row as a tuple.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
            evaluated.

        """
        if self.correlation is not None:
            return tuple(self.correlation[i])
        return tuple(1.0 if j == i else 0.0 for j in range(len(self.entries)))

    def _correlated_terms(self) -> tuple[float, ...]:
        """Return ``t_i = c_i u_i sum_j c_j u_j r_ij`` for every entry.

        The terms sum to the combined variance.

        Returns:
            One covariance-apportioned variance term per entry.

        Equation:
            [EQ-UQ-BUDGET-001], [EQ-UQ-BUDGET-005].

        References:
            [REF-JCGM-100-2008, Section 5.2.2].

        """
        contributions = [entry.contribution for entry in self.entries]
        terms: list[float] = []
        for i, own in enumerate(contributions):
            row = self._correlation_row(i)
            terms.append(
                own
                * math.fsum(
                    r * other for r, other in zip(row, contributions, strict=True)
                )
            )
        return tuple(terms)

    def combined_variance(self) -> float:
        """Return ``u_c^2`` of the law of propagation with correlation terms.

        Returns:
            The combined variance, never negative.

        Equation:
            [EQ-UQ-BUDGET-001].

        References:
            [REF-JCGM-100-2008, Sections 5.1.2 and 5.2.2].

        """
        return max(0.0, math.fsum(self._correlated_terms()))

    def combined_standard_uncertainty(self) -> float:
        """Return ``u_c`` of the law of propagation, correlations included.

        Returns:
            The combined standard uncertainty.

        Equation:
            [EQ-UQ-BUDGET-001].

        References:
            [REF-JCGM-100-2008, Sections 5.1.2 and 5.2.2].

        """
        return math.sqrt(self.combined_variance())

    def law_of_propagation(self) -> CombinedUncertainty:
        """Combine by GUM 5.2.2 and return the rule-tagged record.

        Returns:
            :class:`CombinedUncertainty` of kind standard uncertainty.

        Equation:
            [EQ-UQ-BUDGET-001].

        References:
            [REF-JCGM-100-2008, Sections 5.1.2 and 5.2.2].

        """
        return CombinedUncertainty(
            self.combined_standard_uncertainty(),
            RULE_LAW_OF_PROPAGATION,
            QuantityKind.STANDARD_UNCERTAINTY,
            len(self),
        )

    def root_sum_square(self) -> CombinedUncertainty:
        """Combine as ``sqrt(sum (c_i u_i)^2)``, ignoring any correlation.

        Equal to :meth:`law_of_propagation` when the inputs are independent.
        When a correlation matrix is present it is *not* used and a note says
        so; call :meth:`law_of_propagation` to include it.

        Returns:
            :class:`CombinedUncertainty` of kind standard uncertainty.

        Equation:
            [EQ-UQ-BUDGET-004].

        References:
            [REF-JCGM-100-2008, Section 5.1.2].

        """
        value = math.sqrt(math.fsum(e.variance_contribution for e in self.entries))
        notes = (
            ("correlation matrix present but ignored by this rule",)
            if self.has_correlation
            else ()
        )
        return CombinedUncertainty(
            value,
            RULE_ROOT_SUM_SQUARE,
            QuantityKind.STANDARD_UNCERTAINTY,
            len(self),
            notes=notes,
        )

    def worst_case_sum(self, basis: str = HALF_WIDTH_BASIS) -> CombinedUncertainty:
        """Combine as the linear sum of absolute contributions.

        With ``basis="half_width"`` (default) the terms are ``|c_i| a_i`` and
        the result is a *worst-case bound* on the output error; every entry
        needs a ``half_width``. With ``basis="standard_uncertainty"`` the
        terms are ``|c_i| u_i`` and the result is a *standard uncertainty*: the
        largest value ``u_c`` can take under any correlation (the triangle
        inequality), so it is a conservative standard uncertainty, not a bound.

        Args:
            basis: ``"half_width"`` or ``"standard_uncertainty"``.

        Returns:
            :class:`CombinedUncertainty`; its ``kind`` follows ``basis``.

        Raises:
            ValueError: If ``basis`` is unknown or an entry lacks a
                ``half_width`` under the half-width basis.

        Equation:
            [EQ-UQ-BUDGET-004].

        References:
            [REF-JCGM-100-2008, Section 5.2.2] (the linear sum is the case
            ``r_ij = +1`` for same-signed terms).

        """
        basis = _check_choice(basis, BASES, "basis")
        if basis == HALF_WIDTH_BASIS:
            missing = [e.name for e in self.entries if e.half_width is None]
            if missing:
                raise ValueError(f"half_width is required for: {', '.join(missing)}")
            value = math.fsum(abs(e.c) * (e.half_width or 0.0) for e in self.entries)
            kind = QuantityKind.WORST_CASE_BOUND
        else:
            value = math.fsum(abs(e.contribution) for e in self.entries)
            kind = QuantityKind.STANDARD_UNCERTAINTY
        return CombinedUncertainty(value, RULE_WORST_CASE_SUM, kind, len(self))

    def mixed_rule(
        self, *, level: str = "standard", random_coverage_factor: float | None = None
    ) -> CombinedUncertainty:
        """Biases add linearly, random terms add in root-sum-square, then sum.

        ``level="standard"``: each bias contributes ``|c_i| u_i``; random terms
        contribute ``u_random = sqrt(sum (c_j u_j)^2)`` over the random
        entries (their mutual correlations, if any, are honoured); the result
        is a standard uncertainty. ``level="bound"``: each bias contributes
        ``|c_i| a_i`` (its half-width) and the random part is expanded to the
        same footing, ``k u_random``, with ``random_coverage_factor = k``
        required and stated; the result is a bound at that factor. A
        correlation between a bias entry and a random entry is not used and
        a note says so.

        Args:
            level: ``"standard"`` or ``"bound"``.
            random_coverage_factor: ``k > 0``; required for ``"bound"``,
                forbidden for ``"standard"``.

        Returns:
            :class:`CombinedUncertainty` whose ``kind`` follows ``level``.

        Raises:
            ValueError: If ``level`` is unknown, ``random_coverage_factor``
                is missing or given in the wrong level, or a bias entry has no
                ``half_width`` at the bound level.

        Equation:
            [EQ-UQ-BUDGET-004].

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. The rule is an engineering
            convention for systematic and random errors, not defined by the
            Guide.

        """
        level = _check_choice(level, ("standard", "bound"), "level")
        biases = [e for e in self.entries if e.kind == "bias"]
        random_part = self._subset(lambda e: e.kind == "random")
        random_u = random_part.combined_standard_uncertainty() if random_part else 0.0
        notes: list[str] = []
        if self.has_correlation:
            notes.append("correlation between bias and random entries is not used")
        if level == "standard":
            if random_coverage_factor is not None:
                raise ValueError("random_coverage_factor applies to level='bound' only")
            bias_sum = math.fsum(abs(e.contribution) for e in biases)
            return CombinedUncertainty(
                bias_sum + random_u,
                RULE_MIXED,
                QuantityKind.STANDARD_UNCERTAINTY,
                len(self),
                notes=tuple(notes),
            )
        if random_coverage_factor is None:
            raise ValueError("level='bound' requires random_coverage_factor")
        k = require_finite_real(random_coverage_factor, name="random_coverage_factor")
        if k <= 0.0:
            raise ValueError("random_coverage_factor must be positive")
        missing = [e.name for e in biases if e.half_width is None]
        if missing:
            raise ValueError(
                f"half_width is required for bias entries: {', '.join(missing)}"
            )
        bias_sum = math.fsum(abs(e.c) * (e.half_width or 0.0) for e in biases)
        return CombinedUncertainty(
            bias_sum + k * random_u,
            RULE_MIXED,
            QuantityKind.WORST_CASE_BOUND,
            len(self),
            coverage_factor=k,
            notes=tuple(notes),
        )

    def effective_degrees_of_freedom(self, *, correlated: str = "approximate") -> float:
        """Return the Welch-Satterthwaite effective degrees of freedom.

        ``nu_eff = u_c^4 / sum_i (c_i u_i)^4 / nu_i``. Entries with infinite
        ``nu_i`` add nothing to the denominator; if every entry is infinite, or
        ``u_c = 0``, the result is infinite.

        The formula of GUM G.4.1 assumes uncorrelated inputs. With a
        correlation matrix present this function, under the default
        ``correlated="approximate"``, still evaluates the formula with the
        *correlated* ``u_c`` in the numerator and the individual
        ``(c_i u_i)^4 / nu_i`` in the denominator. That is a heuristic with no
        derivation behind it, and :meth:`expanded_uncertainty` says so in its
        ``warnings``. ``correlated="error"`` raises ``ValueError`` instead.

        Args:
            correlated: ``"approximate"`` or ``"error"``.

        Returns:
            ``nu_eff`` (possibly ``math.inf``).

        Raises:
            ValueError: If ``correlated`` is unknown, or ``"error"`` and a
                correlation matrix is present.

        Equation:
            [EQ-UQ-BUDGET-002].

        References:
            [REF-JCGM-100-2008, Section G.4.1, Equation G.2b],
            [REF-WELCH-1947], [REF-SATTERTHWAITE-1946].

        """
        correlated = _check_choice(correlated, ("approximate", "error"), "correlated")
        if self.has_correlation and correlated == "error":
            raise ValueError(
                "Welch-Satterthwaite assumes uncorrelated inputs; "
                "the budget has a correlation matrix"
            )
        u_c = self.combined_standard_uncertainty()
        denominator = math.fsum(
            e.variance_contribution**2 / e.dof
            for e in self.entries
            if math.isfinite(e.dof)
        )
        if u_c == 0.0 or denominator == 0.0:
            return math.inf
        return u_c**4 / denominator

    def expanded_uncertainty(
        self,
        coverage: float = 0.95,
        *,
        correlated: str = "approximate",
        truncate_dof: bool = False,
    ) -> ExpandedUncertainty:
        """Return ``U = k u_c`` with ``k`` the Student-t quantile at ``nu_eff``.

        ``k = t_{(1+p)/2}(nu_eff)``; for infinite ``nu_eff`` the normal
        quantile is used. The interval ``y +/- U`` has approximate coverage
        ``p`` (GUM G.4.1). See :meth:`effective_degrees_of_freedom` for the
        treatment of correlated inputs; the result's ``warnings`` repeat it.

        Args:
            coverage: Coverage probability ``p`` in ``(0, 1)``.
            correlated: ``"approximate"`` or ``"error"``.
            truncate_dof: If true, ``nu_eff`` is truncated to the next lower
                integer before the quantile is taken (GUM G.4.1 Note 1); the
                default uses the non-integer value.

        Returns:
            :class:`ExpandedUncertainty`.

        Raises:
            ValueError: If ``coverage`` is not a probability, or
                ``truncate_dof`` leaves less than one degree of freedom.

        Equation:
            [EQ-UQ-BUDGET-002].

        References:
            [REF-JCGM-100-2008, Sections 6.2 and G.4.1].

        """
        p = require_probability(coverage, name="coverage")
        if p <= 0.0 or p >= 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        nu = self.effective_degrees_of_freedom(correlated=correlated)
        if truncate_dof and math.isfinite(nu):
            nu = float(math.floor(nu))
            if nu < 1.0:
                raise ValueError(
                    "effective degrees of freedom below one after truncation"
                )
        level = 0.5 * (1.0 + p)
        k = Normal().quantile(level) if math.isinf(nu) else StudentT(nu).quantile(level)
        u_c = self.combined_standard_uncertainty()
        warnings: tuple[str, ...] = ()
        if self.has_correlation:
            warnings = (
                "Welch-Satterthwaite assumes uncorrelated inputs; the effective "
                "degrees of freedom with a correlation matrix are a heuristic",
            )
        return ExpandedUncertainty(k * u_c, u_c, k, p, nu, warnings)

    def contributions(self) -> tuple[Contribution, ...]:
        """Return the contribution table, one :class:`Contribution` per entry.

        The variance share of entry ``i`` is ``t_i / u_c^2`` with
        ``t_i = c_i u_i sum_j c_j u_j r_ij``: it equals ``(c_i u_i)^2 / u_c^2``
        for independent inputs, the shares add to exactly one, and under
        correlation a share can be negative (the entry reduces the total) or
        above one. When ``u_c = 0`` every share is ``nan``.

        Returns:
            Rows in table order.

        Equation:
            [EQ-UQ-BUDGET-005].

        References:
            [REF-JCGM-100-2008, Sections 5.1.3 and 5.2.2].

        """
        terms = self._correlated_terms()
        total = math.fsum(terms)
        rows: list[Contribution] = []
        for entry, term in zip(self.entries, terms, strict=True):
            share = term / total if total > 0.0 else math.nan
            rows.append(
                Contribution(
                    entry.name,
                    entry.c,
                    entry.u,
                    entry.contribution,
                    entry.variance_contribution,
                    share,
                    entry.evaluation,
                    entry.nature,
                    entry.kind,
                    entry.dof,
                )
            )
        return tuple(rows)

    def _subset(self, keep: Callable[[BudgetEntry], bool]) -> UncertaintyBudget | None:
        """Return the sub-budget of entries accepted by ``keep``, or ``None``.

        Args:
            keep: Predicate on entries.

        Returns:
            A budget with the matching correlation sub-matrix, or ``None`` if
            nothing matches.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
            evaluated.

        """
        index = [i for i, entry in enumerate(self.entries) if keep(entry)]
        if not index:
            return None
        sub: tuple[tuple[float, ...], ...] | None = None
        if self.correlation is not None:
            sub = tuple(tuple(self.correlation[i][j] for j in index) for i in index)
        return UncertaintyBudget(tuple(self.entries[i] for i in index), sub)

    def select(
        self,
        *,
        evaluation: str | None = None,
        nature: str | None = None,
        kind: str | None = None,
        names: Sequence[str] | None = None,
    ) -> UncertaintyBudget:
        """Return the sub-budget of entries matching every given criterion.

        The correlation matrix is restricted to the selected entries;
        correlation with entries left out is dropped, so the sub-budget's
        ``u_c`` is the uncertainty *of that subset on its own*.

        Args:
            evaluation: ``"A"`` or ``"B"`` to keep.
            nature: ``"aleatory"`` or ``"epistemic"`` to keep.
            kind: ``"random"`` or ``"bias"`` to keep.
            names: Entry names to keep.

        Returns:
            The sub-budget.

        Raises:
            ValueError: If a criterion is unknown, a name is not in the
                budget, or nothing matches.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical
            equation is evaluated.

        """
        if evaluation is not None:
            evaluation = _check_choice(evaluation, EVALUATION_TYPES, "evaluation")
        if nature is not None:
            nature = _check_choice(nature, NATURES, "nature")
        if kind is not None:
            kind = _check_choice(kind, KINDS, "kind")
        wanted = None if names is None else set(names)
        if wanted is not None and not wanted <= set(self.names):
            raise ValueError("names contains an entry that is not in the budget")

        def keep(entry: BudgetEntry) -> bool:
            """Return whether ``entry`` meets every criterion.

            References:
                [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation
                is evaluated.

            """
            return (
                (evaluation is None or entry.evaluation == evaluation)
                and (nature is None or entry.nature == nature)
                and (kind is None or entry.kind == kind)
                and (wanted is None or entry.name in wanted)
            )

        result = self._subset(keep)
        if result is None:
            raise ValueError("no entry matches the selection")
        return result

    def group_by(self, attribute: str) -> dict[str, UncertaintyBudget]:
        """Split into one sub-budget per value of an entry attribute.

        Args:
            attribute: ``"evaluation"``, ``"nature"`` or ``"kind"``.

        Returns:
            Mapping from value to sub-budget, in order of first appearance.
            Correlation between groups is dropped (see :meth:`select`), so
            the group variances add to the total only for independent inputs.

        Raises:
            ValueError: If ``attribute`` is unknown.

        References:
            [REF-JCGM-100-2008, Section 5.2.2]. No mathematical equation is
            evaluated.

        """
        attribute = _check_choice(
            attribute, ("evaluation", "nature", "kind"), "attribute"
        )
        values: list[str] = []
        for entry in self.entries:
            value = str(getattr(entry, attribute))
            if value not in values:
                values.append(value)
        groups: dict[str, UncertaintyBudget] = {}
        for value in values:
            sub = self._subset(lambda e, v=value: str(getattr(e, attribute)) == v)  # type: ignore[misc]
            assert sub is not None
            groups[value] = sub
        return groups

    def to_rows(self, precision: int = 4) -> tuple[tuple[str, ...], ...]:
        """Return the contribution table as text cells, header row first.

        Columns: name, type, nature, kind, ``c``, ``u``, ``c u``, dof,
        variance share. A final row gives the combined standard uncertainty.

        Args:
            precision: Significant digits, at least 1.

        Returns:
            Tuple of rows of strings.

        Raises:
            ValueError: If ``precision`` is less than 1.

        Equation:
            [EQ-UQ-BUDGET-005].

        References:
            [REF-JCGM-100-2008, Section 5.1.3].

        """
        digits = require_non_negative_integer(precision, name="precision")
        if digits < 1:
            raise ValueError("precision must be at least 1")

        def fmt(value: float) -> str:
            """Format a number with the requested significant digits.

            References:
                [REF-PROBSTAT-VALIDATION]. No mathematical equation is
                evaluated.

            """
            return f"{value:.{digits}g}"

        header = ("name", "type", "nature", "kind", "c", "u", "c*u", "dof", "share")
        rows = [header]
        for row in self.contributions():
            rows.append(
                (
                    row.name,
                    row.evaluation,
                    row.nature,
                    row.kind,
                    fmt(row.c),
                    fmt(row.u),
                    fmt(row.contribution),
                    fmt(row.dof),
                    f"{100.0 * row.share:.1f}%",
                )
            )
        rows.append(
            (
                "combined u_c",
                "",
                "",
                "",
                "",
                "",
                fmt(self.combined_standard_uncertainty()),
                "",
                "",
            )
        )
        return tuple(rows)

    def to_markdown(self, precision: int = 4, *, coverage: float | None = None) -> str:
        """Render the budget as a GitHub-flavoured Markdown table.

        Args:
            precision: Significant digits.
            coverage: If given, a line below the table states the effective
                degrees of freedom, ``k`` and ``U`` at that coverage.

        Returns:
            The table text (no trailing newline).

        Equation:
            [EQ-UQ-BUDGET-005].

        References:
            [REF-JCGM-100-2008, Clause 7].

        """
        rows = self.to_rows(precision)
        lines = ["| " + " | ".join(rows[0]) + " |"]
        lines.append("|" + "|".join(" --- " for _ in rows[0]) + "|")
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        if coverage is not None:
            result = self.expanded_uncertainty(coverage)
            lines.append("")
            lines.append(
                f"nu_eff = {result.effective_dof:.{precision}g}, k = "
                f"{result.coverage_factor:.{precision}g}, "
                f"U = {result.value:.{precision}g} (coverage {coverage:g})"
            )
        return "\n".join(lines)

    def to_text(self, precision: int = 4) -> str:
        """Render the budget as an aligned plain-text table.

        Args:
            precision: Significant digits.

        Returns:
            The table text (no trailing newline).

        References:
            [REF-JCGM-100-2008, Clause 7]. No mathematical equation is
            evaluated.

        """
        rows = self.to_rows(precision)
        widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
        return "\n".join(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
            for row in rows
        )


@dataclass(frozen=True, slots=True)
class LinearizationCheck:
    """Second-order check of the law of propagation for a model.

    For Gaussian inputs with covariance ``Sigma`` and model Hessian ``H`` the
    second-order expansion gives a mean shift ``tr(H Sigma) / 2`` and an extra
    variance ``tr(H Sigma H Sigma) / 2`` beyond the linear ``c' Sigma c``.

    Args:
        mean_shift: ``tr(H Sigma) / 2``, the bias of the output mean.
        second_order_variance: ``tr(H Sigma H Sigma) / 2``.
        linear_variance: ``c' Sigma c`` (the law of propagation).
        mean_ratio: ``|mean_shift| / u_c``.
        spread_ratio: ``sqrt(1 + second_order_variance / linear_variance) - 1``.
        tolerance: Threshold used for ``poor``.
        poor: True if either ratio exceeds ``tolerance`` (or the linear
            variance is zero while the second-order terms are not).

    Equation:
        [EQ-UQ-BUDGET-006].

    References:
        [REF-JCGM-100-2008, Section 5.1.2, Note], [REF-JCGM-101-2008,
        Section 5.8].

    """

    mean_shift: float
    second_order_variance: float
    linear_variance: float
    mean_ratio: float
    spread_ratio: float
    tolerance: float
    poor: bool


@dataclass(frozen=True, slots=True)
class ModelBudget:
    """A budget built from a model function, with its output estimate.

    Args:
        budget: The :class:`UncertaintyBudget` with ``c_i = df/dx_i``.
        estimate: ``f`` at the input estimates.
        gradient: The sensitivity coefficients.
        evaluations: Model evaluations spent.
        linearization: The second-order check, or ``None`` if not requested.

    Equation:
        [EQ-UQ-BUDGET-001], [EQ-UQ-BUDGET-006].

    References:
        [REF-JCGM-100-2008, Sections 5.1.2 and 5.1.3].

    """

    budget: UncertaintyBudget
    estimate: float
    gradient: tuple[float, ...]
    evaluations: int
    linearization: LinearizationCheck | None


def _scalar_model(
    function: Callable[[Sequence[float]], float],
) -> Callable[[Sequence[float]], float]:
    """Wrap a model so every value it returns is validated as a finite real.

    Args:
        function: Model of a vector.

    Returns:
        The wrapped model.

    Raises:
        TypeError: If ``function`` is not callable.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if not callable(function):
        raise TypeError("function must be callable")

    def checked(x: Sequence[float]) -> float:
        """Evaluate the model and require a finite real result.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return require_finite_real(function(tuple(x)), name="model output")

    return checked


def budget_from_model(
    function: Callable[[Sequence[float]], float],
    estimates: Sequence[float],
    uncertainties: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    dof: Sequence[float] | None = None,
    evaluation: str = "B",
    nature: str = "aleatory",
    kind: str = "random",
    correlation: Sequence[Sequence[float]] | None = None,
    step: float | Sequence[float] | None = None,
    second_order: bool = True,
    tolerance: float = 0.1,
) -> ModelBudget:
    """Build an uncertainty budget from a model by numerical differentiation.

    Sensitivity coefficients ``c_i = df/dx_i`` come from
    :func:`probstat.optimization.central_gradient` at the input estimates. The
    second-order check uses :func:`probstat.optimization.central_hessian` and
    the Gaussian-input expansion described in :class:`LinearizationCheck`; it
    flags a poor linearisation when the mean shift or the extra spread exceeds
    ``tolerance`` of ``u_c``. The check needs the Gaussian approximation of the
    inputs and uses their stated standard uncertainties and correlation; it is
    a diagnostic, not a replacement for the Monte Carlo validation of
    :func:`compare_with_law_of_propagation`.

    Args:
        function: Scalar model ``f(x)`` of a sequence of floats.
        estimates: Finite input estimates ``x_i``.
        uncertainties: Non-negative standard uncertainties ``u(x_i)``.
        names: Entry names (default ``"x1"``, ``"x2"``, ...).
        dof: Degrees of freedom per input (default infinite).
        evaluation: ``"A"`` or ``"B"`` for all entries.
        nature: ``"aleatory"`` or ``"epistemic"`` for all entries.
        kind: ``"random"`` or ``"bias"`` for all entries.
        correlation: Correlation matrix of the input estimates, or ``None``.
        step: Optional absolute finite-difference step (scalar or per input).
        second_order: Whether to compute the :class:`LinearizationCheck`.
        tolerance: Threshold ratio for the check, positive.

    Returns:
        :class:`ModelBudget`.

    Raises:
        TypeError: If ``function`` is not callable.
        ValueError: If lengths disagree, the model returns a non-finite value,
            or ``tolerance`` is not positive.

    Equation:
        [EQ-UQ-BUDGET-001], [EQ-UQ-BUDGET-006].

    References:
        [REF-JCGM-100-2008, Sections 5.1.2-5.1.4],
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    model = _scalar_model(function)
    x = tuple(require_finite_real(v, name="estimate") for v in estimates)
    u = tuple(
        require_non_negative_finite_real(v, name="uncertainty") for v in uncertainties
    )
    n = len(x)
    if n == 0 or len(u) != n:
        raise ValueError(
            "estimates and uncertainties must have the same non-zero length"
        )
    labels = tuple(names) if names is not None else tuple(f"x{i + 1}" for i in range(n))
    if len(labels) != n:
        raise ValueError("names must have one entry per input")
    freedom = tuple(dof) if dof is not None else (math.inf,) * n
    if len(freedom) != n:
        raise ValueError("dof must have one entry per input")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    typical = tuple(value if value > 0.0 else 1.0 for value in u)
    value = model(x)
    gradient, used = central_gradient(model, x, step=step, typical_scale=typical)
    evaluations = used + 1
    entries = tuple(
        BudgetEntry(
            labels[i],
            u[i],
            c=gradient[i],
            evaluation=evaluation,
            dof=freedom[i],
            nature=nature,
            kind=kind,
        )
        for i in range(n)
    )
    budget = UncertaintyBudget(entries, correlation)
    check: LinearizationCheck | None = None
    if second_order:
        hessian, more = central_hessian(model, x, typical_scale=typical)
        evaluations += more
        corr = budget.correlation
        sigma = [
            [
                u[i] * u[j] * (1.0 if i == j else (0.0 if corr is None else corr[i][j]))
                for j in range(n)
            ]
            for i in range(n)
        ]
        hs = [
            [math.fsum(hessian[i][k] * sigma[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
        shift = 0.5 * math.fsum(hs[i][i] for i in range(n))
        extra = 0.5 * math.fsum(hs[i][j] * hs[j][i] for i in range(n) for j in range(n))
        linear = budget.combined_variance()
        if linear > 0.0:
            mean_ratio = abs(shift) / math.sqrt(linear)
            spread_ratio = math.sqrt(1.0 + extra / linear) - 1.0
            poor = mean_ratio > tol or spread_ratio > tol
        else:
            mean_ratio = math.inf if shift != 0.0 else 0.0
            spread_ratio = math.inf if extra != 0.0 else 0.0
            poor = shift != 0.0 or extra != 0.0
        check = LinearizationCheck(
            shift, extra, linear, mean_ratio, spread_ratio, tol, poor
        )
    return ModelBudget(budget, value, gradient, evaluations, check)


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Output summary of the Monte Carlo propagation of distributions.

    Args:
        estimate: Average ``y`` of the model values (JCGM 101 Eq. 16).
        standard_uncertainty: ``u(y)``, standard deviation with divisor
            ``M - 1`` (Eq. 17).
        coverage: Coverage probability ``p``.
        symmetric_interval: Probabilistically symmetric ``100 p %`` interval.
        shortest_interval: Shortest ``100 p %`` interval.
        size: Number of trials ``M``.
        skewness: Sample skewness of the model values.
        kurtosis: Sample (non-excess) kurtosis of the model values.
        estimate_standard_error: ``u(y) / sqrt(M)``, the Monte Carlo error of
            ``estimate``.
        uncertainty_standard_error: ``u(y) sqrt((kurtosis - 1) / (4 M))``, the
            large-sample Monte Carlo error of ``u(y)``.
        endpoint_standard_errors: Estimated Monte Carlo errors of the two
            endpoints of ``symmetric_interval``, from the local density of the
            sorted values (a spacing estimator, ``0`` where values tie).
        correlated: Whether a Gaussian copula joined the inputs.

    Equation:
        [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Sections 7.6 and 7.7].

    """

    estimate: float
    standard_uncertainty: float
    coverage: float
    symmetric_interval: tuple[float, float]
    shortest_interval: tuple[float, float]
    size: int
    skewness: float
    kurtosis: float
    estimate_standard_error: float
    uncertainty_standard_error: float
    endpoint_standard_errors: tuple[float, float]
    correlated: bool


def sample_coverage_intervals(
    values: Sequence[float], coverage: float = 0.95
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the symmetric and shortest coverage intervals of a sample.

    Follows JCGM 101:2008 Section 7.7.2: with ``q = pM`` (the integer part of
    ``pM + 1/2`` if ``pM`` is not an integer) the sorted values
    ``y_(1) <= ... <= y_(M)`` give intervals ``[y_(r), y_(r+q)]``. The
    probabilistically symmetric interval takes ``r = (M - q)/2`` (the integer
    part of ``(M - q + 1)/2`` if that is not an integer); the shortest takes
    the ``r`` minimising ``y_(r+q) - y_(r)`` (the first, on ties).

    Args:
        values: At least two finite values (any order).
        coverage: Coverage probability ``p`` in ``(0, 1)``.

    Returns:
        ``(symmetric, shortest)``, each a ``(low, high)`` pair.

    Raises:
        ValueError: If the sample is too small for the requested coverage
            (``M - q < 1``) or contains non-finite values.

    Equation:
        [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Sections 5.3.2-5.3.5 and 7.7.2].

    """
    p = require_probability(coverage, name="coverage")
    if p <= 0.0 or p >= 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")
    ordered = sorted(require_finite_real(v, name="value") for v in values)
    return _intervals_from_sorted(ordered, p)


def _intervals_from_sorted(
    ordered: Sequence[float], p: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return symmetric and shortest intervals of sorted values.

    Args:
        ordered: Sorted finite values.
        p: Coverage probability in ``(0, 1)``.

    Returns:
        ``(symmetric, shortest)``.

    Raises:
        ValueError: If ``M - q < 1``.

    Equation:
        [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Section 7.7.2].

    """
    m = len(ordered)
    pm = p * m
    q = int(pm) if pm == int(pm) else int(pm + 0.5)
    if m - q < 1 or q < 1:
        raise ValueError("sample too small for the requested coverage")
    half = (m - q) / 2.0
    r = int(half) if half == int(half) else int((m - q + 1) / 2.0)
    r = max(r, 1)
    symmetric = (ordered[r - 1], ordered[r - 1 + q])
    best = 1
    best_length = ordered[q] - ordered[0]
    for start in range(2, m - q + 1):
        length = ordered[start - 1 + q] - ordered[start - 1]
        if length < best_length:
            best, best_length = start, length
    return symmetric, (ordered[best - 1], ordered[best - 1 + q])


def _endpoint_errors(ordered: Sequence[float], p: float) -> tuple[float, float]:
    """Estimate the Monte Carlo errors of the symmetric-interval endpoints.

    The endpoint is an order statistic at level ``alpha``; its standard error
    is ``sqrt(alpha (1 - alpha) / M) / g`` with ``g`` the output density at the
    quantile, estimated by the spacing of order statistics ``m = sqrt(M)``
    either side.

    Args:
        ordered: Sorted values.
        p: Coverage probability.

    Returns:
        Errors of the low and high endpoints (``0`` where the local spacing
        is zero).

    Equation:
        [EQ-UQ-BUDGET-007].

    References:
        [REF-DAVID-NAGARAJA-2003].

    """
    m = len(ordered)
    pm = p * m
    q = int(pm) if pm == int(pm) else int(pm + 0.5)
    half = (m - q) / 2.0
    r = max(int(half) if half == int(half) else int((m - q + 1) / 2.0), 1)
    width = max(1, int(math.sqrt(m)))
    errors: list[float] = []
    for rank in (r, r + q):
        low = max(1, rank - width)
        high = min(m, rank + width)
        spacing = ordered[high - 1] - ordered[low - 1]
        alpha = rank / m
        if spacing <= 0.0:
            errors.append(0.0)
            continue
        density = (high - low) / m / spacing
        errors.append(math.sqrt(alpha * (1.0 - alpha) / m) / density)
    return errors[0], errors[1]


def _standard_normal_cdf(z: float) -> float:
    """Return the standard normal distribution function ``Phi(z)``.

    Args:
        z: Real argument.

    Returns:
        ``Phi(z)`` through the complementary error function (accurate in the
        lower tail).

    Equation:
        [EQ-UQ-BUDGET-008].

    References:
        [REF-NELSEN-2006] (Gaussian copula).

    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _checked_laws(distributions: Sequence[_InputLaw]) -> tuple[_InputLaw, ...]:
    """Return the input laws after checking that each can be sampled.

    Args:
        distributions: One law per input.

    Returns:
        The laws as a tuple.

    Raises:
        TypeError: If a law lacks ``sample`` or ``quantile``.
        ValueError: If no law is given.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    laws = tuple(distributions)
    if len(laws) == 0:
        raise ValueError("at least one input distribution is required")
    for law in laws:
        if not callable(getattr(law, "sample", None)) or not callable(
            getattr(law, "quantile", None)
        ):
            raise TypeError("each distribution needs sample() and quantile()")
    return laws


def _checked_coverage(coverage: float) -> float:
    """Return a coverage probability strictly between 0 and 1.

    Args:
        coverage: Coverage probability ``p``.

    Returns:
        ``p`` as a float.

    Raises:
        ValueError: If ``p`` is not strictly between 0 and 1.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    p = require_probability(coverage, name="coverage")
    if p <= 0.0 or p >= 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")
    return p


def _checked_stream(seed: int | RandomStream) -> RandomStream:
    """Return the root stream of a seed or stream argument.

    Args:
        seed: Integer seed or :class:`~probstat.foundations.RandomStream`.

    Returns:
        The stream.

    Raises:
        TypeError: If ``seed`` is neither.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if isinstance(seed, bool) or not isinstance(seed, (int, RandomStream)):
        raise TypeError("seed must be an integer or a RandomStream")
    return seed if isinstance(seed, RandomStream) else RandomStream(seed)


def _draw_points(
    laws: Sequence[_InputLaw],
    corr: tuple[tuple[float, ...], ...] | None,
    m: int,
    stream: RandomStream,
) -> list[tuple[float, ...]]:
    """Draw ``m`` input vectors, independent or joined by a Gaussian copula.

    The draw depends on ``stream`` only (each input uses the child stream
    ``("input", i)``, the copula the child ``("copula",)``), so the same
    stream gives the same points however the caller later evaluates them.

    Args:
        laws: One law per input.
        corr: Copula correlation matrix, or ``None`` for independent inputs.
        m: Number of input vectors.
        stream: Stream this draw consumes.

    Returns:
        The ``m`` input vectors.

    Equation:
        [EQ-UQ-BUDGET-007], [EQ-UQ-BUDGET-008].

    References:
        [REF-JCGM-101-2008, Sections 7.3-7.4], [REF-NELSEN-2006].

    """
    n = len(laws)
    if corr is None:
        columns: Sequence[Sequence[float]] = [
            law.sample(m, seed=stream.child("input", i)) for i, law in enumerate(laws)
        ]
    else:
        factor = cholesky_semidefinite(corr)
        generator = stream.child("copula").generator()
        draws: list[list[float]] = [[0.0] * m for _ in range(n)]
        for t in range(m):
            e = [generator.gauss(0.0, 1.0) for _ in range(n)]
            for i in range(n):
                z = math.fsum(factor[i][j] * e[j] for j in range(i + 1))
                u = min(
                    max(_standard_normal_cdf(z), _PROBABILITY_FLOOR),
                    1.0 - _PROBABILITY_FLOOR,
                )
                draws[i][t] = laws[i].quantile(u)
        columns = draws
    return [tuple(column[t] for column in columns) for t in range(m)]


def _summarize(
    outputs: Sequence[float], p: float, correlated: bool
) -> MonteCarloResult:
    """Summarise model values as a :class:`MonteCarloResult`.

    Args:
        outputs: At least 20 finite model values.
        p: Coverage probability in ``(0, 1)``.
        correlated: Whether a copula joined the inputs.

    Returns:
        The estimate, ``u(y)``, both coverage intervals, shape statistics and
        Monte Carlo errors.

    Raises:
        ValueError: If the sample is too small for the coverage.

    Equation:
        [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Sections 7.5-7.7].

    """
    m = len(outputs)
    ordered = sorted(outputs)
    symmetric, shortest = _intervals_from_sorted(ordered, p)
    mean = math.fsum(outputs) / m
    deviations = [v - mean for v in outputs]
    variance = math.fsum(d * d for d in deviations) / (m - 1)
    u_y = math.sqrt(variance)
    population = math.fsum(d * d for d in deviations) / m
    if population > 0.0:
        skew = math.fsum(d**3 for d in deviations) / m / population**1.5
        kurt = math.fsum(d**4 for d in deviations) / m / population**2
    else:
        skew, kurt = 0.0, 0.0
    return MonteCarloResult(
        estimate=mean,
        standard_uncertainty=u_y,
        coverage=p,
        symmetric_interval=symmetric,
        shortest_interval=shortest,
        size=m,
        skewness=skew,
        kurtosis=kurt,
        estimate_standard_error=u_y / math.sqrt(m),
        uncertainty_standard_error=u_y * math.sqrt(max(kurt - 1.0, 0.0) / (4.0 * m)),
        endpoint_standard_errors=_endpoint_errors(ordered, p),
        correlated=correlated,
    )


def monte_carlo_budget(
    function: Callable[[Sequence[float]], float],
    distributions: Sequence[_InputLaw],
    *,
    size: int,
    seed: int | RandomStream,
    correlation: Sequence[Sequence[float]] | None = None,
    coverage: float = 0.95,
) -> MonteCarloResult:
    """Propagate input distributions through a model by Monte Carlo.

    Implements the propagation and summarising stages of JCGM 101:2008
    (Sections 7.3-7.7) with a fixed number of trials ``size``: draw ``M``
    input vectors, evaluate the model, and report the average, the standard
    deviation with divisor ``M - 1``, and the probabilistically symmetric and
    shortest coverage intervals. ``size`` is chosen by the caller;
    :func:`adaptive_monte_carlo_budget` chooses it by the adaptive procedure
    of Section 7.9.4. ``endpoint_standard_errors`` and the other standard
    errors in the result let the caller judge whether ``size`` was enough.
    Choose ``size`` large against ``1/(1 - p)`` (Section 7.2.2).

    Independent inputs are sampled with each law's own ``sample`` from child
    streams of ``seed``. With ``correlation`` the inputs are joined by a
    Gaussian copula: draw standard normals ``e``, form ``z = L e`` with ``L``
    the Cholesky factor of ``correlation``, set ``u_i = Phi(z_i)`` and
    ``x_i = F_i^{-1}(u_i)``. Hence ``correlation`` is the correlation of the
    *normal scores* ``z``. It equals the Pearson correlation of the inputs
    only for normal marginals; for others the Pearson correlation is smaller
    in magnitude, and the rank (Spearman) correlation is
    ``(6/pi) asin(r/2)``. The copula is a modelling choice; Supplement 1
    treats the joint law only for the multivariate Gaussian case (Section
    6.4.8) and discusses the general case as an assignment of a joint PDF.

    Args:
        function: Scalar model of a tuple of floats.
        distributions: One law per input, each with ``sample(size, seed=...)``
            and ``quantile(p)`` (every :mod:`probstat.distributions` law).
        size: Number of trials ``M >= 20``.
        seed: Root seed or :class:`~probstat.foundations.RandomStream`; the
            result is reproducible for a fixed seed.
        correlation: Optional positive-semidefinite correlation matrix of the
            Gaussian copula (unit diagonal), in input order.
        coverage: Coverage probability ``p`` in ``(0, 1)``.

    Returns:
        :class:`MonteCarloResult`.

    Raises:
        TypeError: If ``function`` is not callable, a law lacks ``sample`` or
            ``quantile``, or ``seed`` has the wrong type.
        ValueError: If shapes disagree, the model returns a non-finite value,
            or ``size`` is too small for the coverage.

    Equation:
        [EQ-UQ-BUDGET-007], [EQ-UQ-BUDGET-008].

    References:
        [REF-JCGM-101-2008, Sections 5.9, 7.2-7.7 and 6.4.8],
        [REF-NELSEN-2006].

    """
    model = _scalar_model(function)
    laws = _checked_laws(distributions)
    m = require_non_negative_integer(size, name="size")
    if m < 20:
        raise ValueError("size must be at least 20")
    p = _checked_coverage(coverage)
    stream = _checked_stream(seed)
    corr = _checked_correlation(correlation, len(laws))
    outputs = [model(point) for point in _draw_points(laws, corr, m, stream)]
    return _summarize(outputs, p, corr is not None)


@dataclass(frozen=True, slots=True)
class AdaptiveMonteCarloResult(MonteCarloResult):
    """Result of the adaptive Monte Carlo procedure of JCGM 101 Section 7.9.4.

    It is a :class:`MonteCarloResult` computed from all ``batches * batch_size``
    model values (so ``size`` is that product and every inherited field means
    what it means there), plus the record of how the procedure stopped.

    Args:
        batches: Number ``h`` of Monte Carlo applications that were run.
        batch_size: Number ``M`` of trials in each.
        estimate_deviation: ``s_y``, the standard deviation of the average of
            the per-batch estimates (step g).
        uncertainty_deviation: The counterpart for the per-batch ``u(y)``.
        low_deviation: The counterpart for the per-batch left endpoint.
        high_deviation: The counterpart for the per-batch right endpoint.
        delta: Numerical tolerance used in the last test: ``delta`` of
            Section 7.9.2 from the ``u(y)`` of all values, divided by
            ``tolerance_divisor``.
        significant_digits: ``ndig`` used for ``delta``.
        tolerance_divisor: 1 for Section 7.9.4, 5 for the Section 8.2
            accuracy ``delta / 5``.
        interval: ``"symmetric"`` or ``"shortest"``: the interval whose
            endpoints were tracked by ``low_deviation`` and ``high_deviation``.
        stabilized: True if the stopping rule was met; False if ``max_batches``
            ran out first, in which case the four deviations still describe
            the final state and the numbers may not meet the tolerance
            (Section 7.9.3, Note 1).

    Equation:
        [EQ-UQ-BUDGET-010].

    References:
        [REF-JCGM-101-2008, Sections 7.9.2-7.9.4 and 8.2].

    """

    batches: int
    batch_size: int
    estimate_deviation: float
    uncertainty_deviation: float
    low_deviation: float
    high_deviation: float
    delta: float
    significant_digits: int
    tolerance_divisor: float
    interval: str
    stabilized: bool


def average_standard_deviation(values: Sequence[float]) -> float:
    """Return the standard deviation associated with an average (7.9.4, step g).

    For per-batch values ``v_1, ..., v_h`` with average ``v``, the statistic
    is ``s^2 = sum (v_r - v)^2 / (h (h - 1))``.

    Args:
        values: At least two finite per-batch values.

    Returns:
        ``s``.

    Raises:
        ValueError: If fewer than two values are given or one is not finite.

    Equation:
        [EQ-UQ-BUDGET-010].

    References:
        [REF-JCGM-101-2008, Section 7.9.4, steps g and h].

    """
    data = [require_finite_real(v, name="value") for v in values]
    h = len(data)
    if h < 2:
        raise ValueError("at least two per-batch values are required")
    mean = math.fsum(data) / h
    return math.sqrt(math.fsum((v - mean) ** 2 for v in data) / (h * (h - 1)))


def adaptive_stopping_rule(deviations: Sequence[float], delta: float) -> bool:
    """Return whether the computation has stabilised (7.9.4, steps k and l).

    The procedure continues if any of ``2 s_y``, ``2 s_u(y)``, ``2 s_ylow`` and
    ``2 s_yhigh`` *exceeds* ``delta``; it has stabilised otherwise, so a value
    of exactly ``delta`` stabilises. Any number of deviations is accepted
    (Note 5 of the clause allows leaving the endpoints out).

    Args:
        deviations: Non-negative standard deviations of the averages.
        delta: Positive numerical tolerance.

    Returns:
        ``True`` if every ``2 s`` is at most ``delta``.

    Raises:
        ValueError: If a deviation is negative or not finite, none is given,
            or ``delta`` is not positive.

    Equation:
        [EQ-UQ-BUDGET-010].

    References:
        [REF-JCGM-101-2008, Section 7.9.4, step k and Note 5].

    """
    values = [require_non_negative_finite_real(s, name="deviation") for s in deviations]
    if not values:
        raise ValueError("at least one deviation is required")
    tolerance = require_finite_real(delta, name="delta")
    if tolerance <= 0.0:
        raise ValueError("delta must be positive")
    return all(2.0 * s <= tolerance for s in values)


def _clause_batch_size(coverage: float) -> int:
    """Return ``M = max(J, 10^4)`` with ``J = ceil(100 / (1 - p))`` (step b).

    Args:
        coverage: Coverage probability ``p`` in ``(0, 1)``.

    Returns:
        The number of trials per batch.

    Equation:
        [EQ-UQ-BUDGET-010].

    References:
        [REF-JCGM-101-2008, Section 7.9.4, step b].

    """
    return max(math.ceil(100.0 / (1.0 - coverage)), 10_000)


def _finite_batch_outputs(values: Sequence[object], count: int) -> list[float]:
    """Coerce the outputs of one batch to finite floats.

    Args:
        values: One output per requested point.
        count: Number of requested points.

    Returns:
        The outputs.

    Raises:
        ValueError: If the count differs or an output is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    outputs = scalar_outputs(values, count)
    for value in outputs:
        require_finite_real(value, name="model output")
    return list(outputs)


def adaptive_monte_carlo_budget_procedure(
    distributions: Sequence[_InputLaw],
    *,
    significant_digits: int = 2,
    coverage: float = 0.95,
    batch_size: int | None = None,
    max_batches: int = 100,
    seed: int | RandomStream,
    correlation: Sequence[Sequence[float]] | None = None,
    interval: str = "symmetric",
    tolerance_divisor: float = 1.0,
    label: str = "model",
) -> Procedure[AdaptiveMonteCarloResult]:
    """Return the adaptive Monte Carlo procedure as a batch procedure.

    The generator yields one :class:`~probstat.foundations.EvaluationRequest`
    per batch (``M`` input vectors) and receives the model values in the same
    order, so the model can run anywhere: drive it with
    :func:`~probstat.foundations.run_procedure` or
    :class:`~probstat.foundations.AskTellSession`. The points of batch ``b``
    depend on ``(seed, b)`` only, so the result does not depend on how the
    caller chunks or schedules the evaluation of a request. Arguments are
    validated when this function is called, not at the first request.

    Args:
        distributions: One law per input (``sample`` and ``quantile``).
        significant_digits: ``ndig >= 1``, the digits regarded as meaningful
            in ``u(y)``; the clause suggests 1 or 2.
        coverage: Coverage probability ``p`` in ``(0, 1)``.
        batch_size: Trials per batch ``M >= 20``; ``None`` takes
            ``max(J, 10^4)`` with ``J = ceil(100/(1 - p))`` as in step b.
        max_batches: Largest number of batches, at least 2 (the first
            stability test needs two). The procedure then stops with
            ``stabilized=False`` instead of raising.
        seed: Root seed or :class:`~probstat.foundations.RandomStream`.
        correlation: Optional Gaussian-copula correlation, as in
            :func:`monte_carlo_budget`.
        interval: ``"symmetric"`` or ``"shortest"``: which interval's
            endpoints are tracked by the stopping rule.
        tolerance_divisor: Positive factor dividing ``delta``; use 5 for the
            accuracy ``delta / 5`` recommended in Section 8.2 for validating
            the GUM uncertainty framework.
        label: Model label of the requests.

    Returns:
        A procedure whose return value is an :class:`AdaptiveMonteCarloResult`.

    Raises:
        TypeError: If a law lacks ``sample`` or ``quantile`` or ``seed`` has
            the wrong type.
        ValueError: If an argument is out of range. While the procedure runs,
            a non-finite model value, or a model with zero spread (so that
            ``delta`` is undefined), raises ``ValueError`` as well.

    Equation:
        [EQ-UQ-BUDGET-010], [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Section 7.9.4], [REF-PEP-342].

    """
    laws = _checked_laws(distributions)
    ndig = require_non_negative_integer(significant_digits, name="significant_digits")
    if ndig < 1:
        raise ValueError("significant_digits must be at least 1")
    p = _checked_coverage(coverage)
    if batch_size is None:
        m = _clause_batch_size(p)
    else:
        m = require_non_negative_integer(batch_size, name="batch_size")
    if m < 20:
        raise ValueError("batch_size must be at least 20")
    q = int(p * m) if p * m == int(p * m) else int(p * m + 0.5)
    if m - q < 1:
        raise ValueError("batch_size is too small for the requested coverage")
    limit = require_non_negative_integer(max_batches, name="max_batches")
    if limit < 2:
        raise ValueError("max_batches must be at least 2")
    kind = _check_choice(interval, COVERAGE_INTERVALS, "interval")
    divisor = require_finite_real(tolerance_divisor, name="tolerance_divisor")
    if divisor <= 0.0:
        raise ValueError("tolerance_divisor must be positive")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    stream = _checked_stream(seed)
    corr = _checked_correlation(correlation, len(laws))
    return _adaptive_batches(
        laws, ndig, p, m, limit, stream, corr, kind, divisor, label
    )


def _adaptive_batches(
    laws: Sequence[_InputLaw],
    ndig: int,
    p: float,
    m: int,
    limit: int,
    stream: RandomStream,
    corr: tuple[tuple[float, ...], ...] | None,
    kind: str,
    divisor: float,
    label: str,
) -> Procedure[AdaptiveMonteCarloResult]:
    """Run steps c to l of JCGM 101 Section 7.9.4 as a generator.

    Args:
        laws: Checked input laws.
        ndig: Significant digits for ``delta``.
        p: Coverage probability.
        m: Trials per batch.
        limit: Largest number of batches.
        stream: Root stream; batch ``b`` draws from ``stream.child("batch", b)``.
        corr: Checked copula correlation, or ``None``.
        kind: Tracked interval, ``"symmetric"`` or ``"shortest"``.
        divisor: Divisor of ``delta``.
        label: Model label of the requests.

    Returns:
        The procedure; its return value is the result record.

    Raises:
        ValueError: If a model value is not finite or the pooled ``u(y)`` is
            zero.

    Equation:
        [EQ-UQ-BUDGET-010].

    References:
        [REF-JCGM-101-2008, Section 7.9.4].

    """
    estimates: list[float] = []
    uncertainties: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    values: list[float] = []
    count = 0
    mean = 0.0
    spread = 0.0  # sum of squared deviations of all values so far
    deviations = (math.inf,) * 4
    delta = math.inf
    stabilized = False
    for b in range(limit):
        points = _draw_points(laws, corr, m, stream.child("batch", b))
        raw = yield EvaluationRequest(
            points=tuple(points), label=label, stage=f"batch {b + 1}"
        )
        outputs = _finite_batch_outputs(raw, m)
        batch_mean = math.fsum(outputs) / m
        batch_spread = math.fsum((v - batch_mean) ** 2 for v in outputs)
        symmetric, shortest = _intervals_from_sorted(sorted(outputs), p)
        low, high = symmetric if kind == "symmetric" else shortest
        estimates.append(batch_mean)
        uncertainties.append(math.sqrt(batch_spread / (m - 1)))
        lows.append(low)
        highs.append(high)
        values.extend(outputs)
        # Pairwise combination of sums of squares (all values so far).
        combined = count + m
        shift = batch_mean - mean
        spread += batch_spread + shift * shift * count * m / combined
        mean += shift * m / combined
        count = combined
        if b == 0:
            continue  # step f: a second application is always made
        pooled = math.sqrt(spread / (count - 1))
        if pooled <= 0.0:
            raise ValueError(
                "the model output has zero spread, so the numerical tolerance "
                "is undefined"
            )
        delta = numerical_tolerance(pooled, ndig) / divisor
        deviations = (
            average_standard_deviation(estimates),
            average_standard_deviation(uncertainties),
            average_standard_deviation(lows),
            average_standard_deviation(highs),
        )
        if adaptive_stopping_rule(deviations, delta):
            stabilized = True
            break
    final = _summarize(values, p, corr is not None)
    return AdaptiveMonteCarloResult(
        estimate=final.estimate,
        standard_uncertainty=final.standard_uncertainty,
        coverage=final.coverage,
        symmetric_interval=final.symmetric_interval,
        shortest_interval=final.shortest_interval,
        size=final.size,
        skewness=final.skewness,
        kurtosis=final.kurtosis,
        estimate_standard_error=final.estimate_standard_error,
        uncertainty_standard_error=final.uncertainty_standard_error,
        endpoint_standard_errors=final.endpoint_standard_errors,
        correlated=final.correlated,
        batches=len(estimates),
        batch_size=m,
        estimate_deviation=deviations[0],
        uncertainty_deviation=deviations[1],
        low_deviation=deviations[2],
        high_deviation=deviations[3],
        delta=delta,
        significant_digits=ndig,
        tolerance_divisor=divisor,
        interval=kind,
        stabilized=stabilized,
    )


def adaptive_monte_carlo_budget(
    function: Callable[[Sequence[float]], float],
    distributions: Sequence[_InputLaw],
    *,
    significant_digits: int = 2,
    coverage: float = 0.95,
    batch_size: int | None = None,
    max_batches: int = 100,
    seed: int | RandomStream,
    correlation: Sequence[Sequence[float]] | None = None,
    interval: str = "symmetric",
    tolerance_divisor: float = 1.0,
) -> AdaptiveMonteCarloResult:
    """Propagate input distributions by the adaptive Monte Carlo procedure.

    Implements JCGM 101:2008 Section 7.9.4. Batches of ``M`` trials are run
    one after another (step d); each gives an estimate, a standard
    uncertainty and the endpoints of a ``100 p %`` coverage interval
    (step e). From the second batch on, the standard deviations of the
    averages of those four per-batch quantities over the ``h`` batches are
    formed (steps g and h), the numerical tolerance ``delta`` comes from the
    ``u(y)`` of all ``h M`` values with ``ndig`` significant digits
    (steps i and j, Section 7.9.2), and the computation stops when twice each
    of the four deviations is at most ``delta`` (steps k and l). The result
    is computed from all ``h M`` values. A thin wrapper of
    :func:`adaptive_monte_carlo_budget_procedure` run with the pointwise
    model; use that form for a simulator that evaluates batches elsewhere.

    The procedure is stochastic and, as the clause notes (7.9.3, Note 1), has
    no guarantee: when ``max_batches`` is used up first the record has
    ``stabilized=False`` and the caller must not read the numbers as meeting
    ``delta``. Each batch draws from the child stream ``("batch", b)`` of
    ``seed``, so the result is reproducible.

    Args:
        function: Scalar model of a tuple of floats.
        distributions: One law per input (``sample`` and ``quantile``).
        significant_digits: ``ndig >= 1`` (the clause suggests 1 or 2).
        coverage: Coverage probability ``p`` in ``(0, 1)``.
        batch_size: Trials per batch ``M >= 20``; ``None`` takes
            ``max(J, 10^4)``, ``J = ceil(100/(1 - p))`` (step b), which is
            10 000 for ``p <= 0.99``.
        max_batches: Largest number of batches (at least 2).
        seed: Root seed or :class:`~probstat.foundations.RandomStream`.
        correlation: Optional Gaussian-copula correlation, as in
            :func:`monte_carlo_budget`.
        interval: ``"symmetric"`` or ``"shortest"`` interval tracked by the
            stopping rule.
        tolerance_divisor: Divisor of ``delta``; 5 gives the Section 8.2
            accuracy used to validate the GUM uncertainty framework.

    Returns:
        :class:`AdaptiveMonteCarloResult`.

    Raises:
        TypeError: If ``function`` is not callable, a law lacks ``sample`` or
            ``quantile``, or ``seed`` has the wrong type.
        ValueError: If an argument is out of range, the model returns a
            non-finite value, or the model output has zero spread.

    Equation:
        [EQ-UQ-BUDGET-010], [EQ-UQ-BUDGET-007].

    References:
        [REF-JCGM-101-2008, Sections 7.9.2-7.9.4 and 8.2].

    """
    model = _scalar_model(function)
    procedure = adaptive_monte_carlo_budget_procedure(
        distributions,
        significant_digits=significant_digits,
        coverage=coverage,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed,
        correlation=correlation,
        interval=interval,
        tolerance_divisor=tolerance_divisor,
    )
    return run_procedure(procedure, batch_from_pointwise(model))


@dataclass(frozen=True, slots=True)
class GumValidation:
    """Outcome of validating the GUM uncertainty framework by Monte Carlo.

    Args:
        validated: True if both endpoint differences are at most ``delta``.
        delta: Numerical tolerance ``delta`` of JCGM 101 Section 7.9.2.
        d_low: ``|y - U_p - y_low|`` (Eq. 19).
        d_high: ``|y + U_p - y_high|`` (Eq. 20).
        gum_interval: ``(y - U_p, y + U_p)``.
        monte_carlo_interval: The Monte Carlo interval compared with it.
        interval: ``"symmetric"`` or ``"shortest"``.
        coverage: Coverage probability.
        coverage_factor: ``k_p`` used for ``U_p``.
        significant_digits: ``ndig`` used for ``delta``.
        monte_carlo_adequate: Whether the Monte Carlo results meet the
            Section 8.2 requirement of accuracy ``delta / 5``, with ``delta``
            from the GUM ``u(y)`` of this comparison. For an
            :class:`AdaptiveMonteCarloResult` this is the clause's own test:
            twice each of the four stabilisation deviations (estimate,
            ``u(y)`` and the two endpoints of the compared interval) is at
            most ``delta / 5``. For a fixed-size :class:`MonteCarloResult` it
            is a proxy: twice the Monte Carlo standard error of ``y``,
            ``u(y)`` and of each symmetric-interval endpoint is at most
            ``delta / 5``.
        adequacy_basis: Which of the two tests produced
            ``monte_carlo_adequate``: ``ADEQUACY_STABILIZATION``,
            ``ADEQUACY_PROXY``, or ``ADEQUACY_OTHER_INTERVAL`` (an adaptive
            result that tracked the other interval; its endpoints were not
            tested, so the flag is false).

    Equation:
        [EQ-UQ-BUDGET-009].

    References:
        [REF-JCGM-101-2008, Sections 7.9.2, 8.1.3 and 8.2].

    """

    validated: bool
    delta: float
    d_low: float
    d_high: float
    gum_interval: tuple[float, float]
    monte_carlo_interval: tuple[float, float]
    interval: str
    coverage: float
    coverage_factor: float
    significant_digits: int
    monte_carlo_adequate: bool
    adequacy_basis: str


def numerical_tolerance(value: float, significant_digits: int) -> float:
    """Return the numerical tolerance ``delta`` of a value (JCGM 101, 7.9.2).

    Express ``value = c x 10^l`` with ``c`` an integer of ``ndig`` decimal
    digits; then ``delta = 10^l / 2``. For ``u = 2.0`` and ``ndig = 2`` this is
    ``c = 20``, ``l = -1``, ``delta = 0.05``.

    Args:
        value: Positive finite value (a standard uncertainty).
        significant_digits: ``ndig >= 1``.

    Returns:
        ``delta``.

    Raises:
        ValueError: If ``value`` is not positive or ``ndig < 1``.

    Equation:
        [EQ-UQ-BUDGET-009].

    References:
        [REF-JCGM-101-2008, Section 7.9.2, Equation 18].

    """
    x = require_finite_real(value, name="value")
    ndig = require_non_negative_integer(significant_digits, name="significant_digits")
    if x <= 0.0:
        raise ValueError("value must be positive")
    if ndig < 1:
        raise ValueError("significant_digits must be at least 1")
    exponent = math.floor(math.log10(x)) - ndig + 1
    if round(x / 10.0**exponent) >= 10**ndig:
        exponent += 1
    return 0.5 * 10.0**exponent


def compare_with_law_of_propagation(
    estimate: float,
    standard_uncertainty: float,
    monte_carlo: MonteCarloResult,
    *,
    coverage_factor: float | None = None,
    significant_digits: int = 2,
    interval: str = "symmetric",
) -> GumValidation:
    """Validate the GUM uncertainty framework against a Monte Carlo result.

    Procedure of JCGM 101:2008 Section 8.1.3: form ``delta`` from the GUM
    standard uncertainty with ``ndig`` significant digits (7.9.2), take the
    GUM interval ``y +/- U_p`` with ``U_p = k_p u(y)``, and compare its
    endpoints with those of the Monte Carlo coverage interval; the framework
    is validated if both endpoint differences are at most ``delta``. The
    validation holds for the stated coverage probability only. The Supplement
    does not say whether ``delta`` is formed from the GUM or the Monte Carlo
    ``u(y)``; the GUM value is used here (the two agree to the digits shown in
    its examples).

    Args:
        estimate: GUM output estimate ``y`` (the model at the input estimates).
        standard_uncertainty: GUM combined standard uncertainty ``u(y)``.
        monte_carlo: Result of :func:`monte_carlo_budget` or of
            :func:`adaptive_monte_carlo_budget`. With the adaptive result the
            ``monte_carlo_adequate`` flag applies the clause's own test
            (stabilisation deviations against ``delta / 5``, Section 8.2);
            run it with ``tolerance_divisor=5`` and the same ``ndig`` to
            obtain results to that accuracy.
        coverage_factor: ``k_p``; default the normal quantile at
            ``(1 + p)/2`` for ``p = monte_carlo.coverage`` (pass the Student-t
            factor of an :class:`ExpandedUncertainty` to use that).
        significant_digits: ``ndig`` for ``delta`` (1 or 2 usually).
        interval: ``"symmetric"`` or ``"shortest"`` Monte Carlo interval.

    Returns:
        :class:`GumValidation`.

    Raises:
        TypeError: If ``monte_carlo`` is not a :class:`MonteCarloResult`.
        ValueError: If ``standard_uncertainty`` or ``coverage_factor`` is not
            positive or ``interval`` is unknown.

    Equation:
        [EQ-UQ-BUDGET-009].

    References:
        [REF-JCGM-101-2008, Sections 7.9.2, 8.1.2-8.1.3 and 8.2].

    """
    if not isinstance(monte_carlo, MonteCarloResult):
        raise TypeError("monte_carlo must be a MonteCarloResult")
    kind = _check_choice(interval, COVERAGE_INTERVALS, "interval")
    y = require_finite_real(estimate, name="estimate")
    u = require_finite_real(standard_uncertainty, name="standard_uncertainty")
    if u <= 0.0:
        raise ValueError("standard_uncertainty must be positive")
    p = monte_carlo.coverage
    if coverage_factor is None:
        k = Normal().quantile(0.5 * (1.0 + p))
    else:
        k = require_finite_real(coverage_factor, name="coverage_factor")
        if k <= 0.0:
            raise ValueError("coverage_factor must be positive")
    delta = numerical_tolerance(u, significant_digits)
    gum = (y - k * u, y + k * u)
    reference = (
        monte_carlo.symmetric_interval
        if kind == "symmetric"
        else monte_carlo.shortest_interval
    )
    d_low = abs(gum[0] - reference[0])
    d_high = abs(gum[1] - reference[1])
    if isinstance(monte_carlo, AdaptiveMonteCarloResult):
        if monte_carlo.interval == kind:
            adequate = adaptive_stopping_rule(
                (
                    monte_carlo.estimate_deviation,
                    monte_carlo.uncertainty_deviation,
                    monte_carlo.low_deviation,
                    monte_carlo.high_deviation,
                ),
                delta / 5.0,
            )
            basis = ADEQUACY_STABILIZATION
        else:
            adequate = False
            basis = ADEQUACY_OTHER_INTERVAL
    else:
        errors = (
            monte_carlo.estimate_standard_error,
            monte_carlo.uncertainty_standard_error,
            *monte_carlo.endpoint_standard_errors,
        )
        adequate = all(2.0 * e <= delta / 5.0 for e in errors)
        basis = ADEQUACY_PROXY
    return GumValidation(
        validated=d_low <= delta and d_high <= delta,
        delta=delta,
        d_low=d_low,
        d_high=d_high,
        gum_interval=gum,
        monte_carlo_interval=reference,
        interval=kind,
        coverage=p,
        coverage_factor=k,
        significant_digits=significant_digits,
        monte_carlo_adequate=adequate,
        adequacy_basis=basis,
    )


__all__ = [
    "ADEQUACY_OTHER_INTERVAL",
    "ADEQUACY_PROXY",
    "ADEQUACY_STABILIZATION",
    "BASES",
    "COVERAGE_INTERVALS",
    "AdaptiveMonteCarloResult",
    "BudgetEntry",
    "CombinedUncertainty",
    "Contribution",
    "ExpandedUncertainty",
    "GumValidation",
    "LinearizationCheck",
    "ModelBudget",
    "MonteCarloResult",
    "QuantityKind",
    "UncertaintyBudget",
    "adaptive_monte_carlo_budget",
    "adaptive_monte_carlo_budget_procedure",
    "adaptive_stopping_rule",
    "average_standard_deviation",
    "budget_from_model",
    "compare_with_law_of_propagation",
    "monte_carlo_budget",
    "numerical_tolerance",
    "sample_coverage_intervals",
]
