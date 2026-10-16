"""Quantitative risk register: occurrence probability times uncertain impact.

Each :class:`Risk` either occurs (``B ~ Bernoulli(p)``) or not; if it occurs
its impact is ``I`` (a fixed amount, a discrete law, or a continuous law), so
its exposure is ``X = B I``. With ``B`` independent of ``I``,

    E[X] = p E[I],    Var[X] = p E[I^2] - p^2 E[I]^2.

Assuming the risks are *mutually independent*, the total exposure
``T = sum X_i`` has mean ``sum p_i E[I_i]`` and variance
``sum Var[X_i]`` exactly. Its full distribution is computed

- exactly by convolution when every impact is discrete (fixed amounts are
  one-point laws); the support can grow multiplicatively, so a cap applies;
- by Monte Carlo otherwise.

Independence is a strong assumption: common causes (a shared supplier, the
same test facility, schedule slip) make risks co-occur and fatten the upper
tail. Model such dependence explicitly (for example with a
:class:`~probstat.modeling.ProbabilisticModel` and a Gaussian copula) when it
matters.

Mitigation is described by a cost and the probability and/or impact after
mitigation. The *risk reduction* is the drop in expected exposure, and the
mitigation ROI is ``reduction / cost``. Because expectation is additive,
the reductions of independent risks add exactly, which the waterfall uses.

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from probstat.foundations import (
    RandomStream,
    require_finite_real,
    require_probability,
    resolve_stream,
)

_SUM_TOLERANCE = 1e-9
RANKING_KEYS = ("exposure", "probability", "impact", "roi", "reduction")


@runtime_checkable
class ImpactLaw(Protocol):
    """Structural interface of a continuous impact law.

    References:
        [REF-ROSS-2019, Chapter 5].

    """

    def quantile(self, probability: float) -> float:
        """Return the inverse CDF.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    @property
    def mean(self) -> float:
        """Return ``E[I]``.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    @property
    def variance(self) -> float:
        """Return ``Var[I]``.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...


@dataclass(frozen=True, slots=True)
class DiscreteImpact:
    """A finite discrete impact law.

    Args:
        values: Finite impact values (repeats are merged).
        probabilities: Non-negative probabilities summing to one.

    References:
        [REF-ROSS-2019, Chapter 4].

    """

    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate, sort, and merge the support.

        Raises:
            ValueError: If lengths differ, the law is empty, or the
                probabilities do not sum to one.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        values = tuple(require_finite_real(v, name="impact value") for v in self.values)
        probabilities = tuple(
            require_probability(p, name="impact probability")
            for p in self.probabilities
        )
        if not values or len(values) != len(probabilities):
            raise ValueError("values and probabilities must be non-empty and aligned")
        if abs(math.fsum(probabilities) - 1.0) > _SUM_TOLERANCE:
            raise ValueError("impact probabilities must sum to one")
        merged: dict[float, list[float]] = {}
        for v, p in zip(values, probabilities, strict=True):
            merged.setdefault(v, []).append(p)
        keys = sorted(merged)
        object.__setattr__(self, "values", tuple(keys))
        object.__setattr__(
            self, "probabilities", tuple(math.fsum(merged[k]) for k in keys)
        )

    @property
    def mean(self) -> float:
        """Return ``E[I] = sum v_k q_k``.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        return math.fsum(
            v * p for v, p in zip(self.values, self.probabilities, strict=True)
        )

    @property
    def variance(self) -> float:
        """Return ``Var[I]`` by the two-pass formula.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        mean = self.mean
        return math.fsum(
            p * (v - mean) ** 2
            for v, p in zip(self.values, self.probabilities, strict=True)
        )

    def quantile(self, probability: float) -> float:
        """Return the lower quantile ``inf{v : F(v) >= u}``.

        Args:
            probability: Level in ``(0, 1]``.

        Returns:
            Support value.

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        u = require_probability(probability, name="probability")
        cumulative = 0.0
        for v, p in zip(self.values, self.probabilities, strict=True):
            cumulative += p
            if cumulative >= u - 1e-15:
                return v
        return self.values[-1]


def _impact(value: object, *, name: str) -> float | DiscreteImpact | ImpactLaw:
    """Validate an impact: a finite amount, a discrete law, or a continuous law.

    Args:
        value: Candidate impact.
        name: Field name for error messages.

    Returns:
        The impact.

    Raises:
        TypeError: If it is none of the three kinds.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, DiscreteImpact):
        return value
    if all(hasattr(value, a) for a in ("quantile", "mean", "variance")):
        return value  # type: ignore[return-value]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number or a law")
    try:
        return require_finite_real(value, name=name)
    except TypeError as error:
        raise TypeError(f"{name} must be a number, DiscreteImpact, or a law") from error


def _moments(impact: float | DiscreteImpact | ImpactLaw) -> tuple[float, float]:
    """Return ``(E[I], E[I^2])`` of an impact.

    Args:
        impact: Validated impact.

    Returns:
        First and second raw moments.

    References:
        [REF-ROSS-2019, Chapter 4].

    """
    if isinstance(impact, float):
        return impact, impact * impact
    mean = float(impact.mean)
    return mean, float(impact.variance) + mean * mean


@dataclass(frozen=True)
class Risk:
    """One register entry with optional mitigation.

    Args:
        name: Distinct risk label.
        probability: Probability of occurrence in the period of interest.
        impact: Consequence if it occurs: a number, a
            :class:`DiscreteImpact`, or a continuous law with ``quantile``,
            ``mean`` and ``variance``.
        category: Free-text category (cost, schedule, technical, ...).
        owner: Free-text risk owner.
        mitigation_cost: Non-negative cost of the mitigation.
        mitigated_probability: Probability after mitigation (``None``: same).
        mitigated_impact: Impact after mitigation (``None``: same).
        description: Optional text.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-NASA-RMH-2011].

    """

    name: str
    probability: float
    impact: float | DiscreteImpact | ImpactLaw
    category: str = ""
    owner: str = ""
    mitigation_cost: float = 0.0
    mitigated_probability: float | None = None
    mitigated_impact: float | DiscreteImpact | ImpactLaw | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a value is outside its domain.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        for field_name in ("category", "owner", "description"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        object.__setattr__(
            self,
            "probability",
            require_probability(self.probability, name="probability"),
        )
        object.__setattr__(self, "impact", _impact(self.impact, name="impact"))
        cost = require_finite_real(self.mitigation_cost, name="mitigation_cost")
        if cost < 0.0:
            raise ValueError("mitigation_cost must be non-negative")
        object.__setattr__(self, "mitigation_cost", cost)
        if self.mitigated_probability is not None:
            object.__setattr__(
                self,
                "mitigated_probability",
                require_probability(
                    self.mitigated_probability, name="mitigated_probability"
                ),
            )
        if self.mitigated_impact is not None:
            object.__setattr__(
                self,
                "mitigated_impact",
                _impact(self.mitigated_impact, name="mitigated_impact"),
            )

    @property
    def has_mitigation(self) -> bool:
        """Return whether a mitigated state is specified.

        References:
            Project validation policy. No equation is evaluated.

        """
        return (
            self.mitigated_probability is not None or self.mitigated_impact is not None
        )

    def state(
        self, *, mitigated: bool = False
    ) -> tuple[float, float | DiscreteImpact | ImpactLaw]:
        """Return ``(probability, impact)`` before or after mitigation.

        Args:
            mitigated: Whether to apply the mitigation.

        Returns:
            Probability and impact in effect.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not mitigated:
            return self.probability, self.impact
        p = (
            self.probability
            if self.mitigated_probability is None
            else self.mitigated_probability
        )
        impact = self.impact if self.mitigated_impact is None else self.mitigated_impact
        return p, impact

    def expected_exposure(self, *, mitigated: bool = False) -> float:
        """Return ``E[X] = p E[I]``.

        Args:
            mitigated: Whether to apply the mitigation.

        Equation:
            [EQ-RISK-REG-001].

        References:
            [REF-KAPLAN-GARRICK-1981], [REF-ROSS-2019, Chapter 7].

        """
        p, impact = self.state(mitigated=mitigated)
        return p * _moments(impact)[0]

    def exposure_variance(self, *, mitigated: bool = False) -> float:
        """Return ``Var[X] = p E[I^2] - p^2 E[I]^2``.

        Args:
            mitigated: Whether to apply the mitigation.

        Equation:
            [EQ-RISK-REG-001].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        p, impact = self.state(mitigated=mitigated)
        first, second = _moments(impact)
        return max(p * second - p * p * first * first, 0.0)

    @property
    def risk_reduction(self) -> float:
        """Return the drop in expected exposure achieved by the mitigation.

        Equation:
            [EQ-RISK-REG-003].

        References:
            [REF-NASA-RMH-2011].

        """
        return self.expected_exposure() - self.expected_exposure(mitigated=True)

    @property
    def mitigation_roi(self) -> float:
        """Return ``risk_reduction / mitigation_cost``.

        ``inf`` for a free mitigation with positive reduction, ``nan`` without
        mitigation or for a free mitigation that changes nothing.

        Equation:
            [EQ-RISK-REG-003].

        References:
            [REF-NASA-RMH-2011].

        """
        if not self.has_mitigation:
            return math.nan
        reduction = self.risk_reduction
        if self.mitigation_cost == 0.0:
            return math.inf if reduction > 0.0 else math.nan
        return reduction / self.mitigation_cost


@dataclass(frozen=True)
class ExposureDistribution:
    """Exact finite distribution of the total exposure.

    Args:
        values: Ascending support.
        probabilities: Point probabilities aligned with ``values``.

    References:
        [REF-ROSS-2019, Chapter 7].

    """

    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    @property
    def mean(self) -> float:
        """Return the expected total exposure.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        return math.fsum(
            v * p for v, p in zip(self.values, self.probabilities, strict=True)
        )

    @property
    def variance(self) -> float:
        """Return the variance of the total exposure.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        mean = self.mean
        return math.fsum(
            p * (v - mean) ** 2
            for v, p in zip(self.values, self.probabilities, strict=True)
        )

    def cdf(self, x: float) -> float:
        """Return ``P(T <= x)``.

        Args:
            x: Finite level.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        index = bisect.bisect_right(self.values, require_finite_real(x, name="x"))
        return min(1.0, math.fsum(self.probabilities[:index]))

    def exceedance(self, x: float) -> float:
        """Return ``P(T > x)``, summed over the upper tail for accuracy.

        Args:
            x: Finite level.

        Equation:
            [EQ-RISK-EXC-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        index = bisect.bisect_right(self.values, require_finite_real(x, name="x"))
        return math.fsum(self.probabilities[index:])

    def quantile(self, level: float) -> float:
        """Return the lower quantile (VaR) ``inf{x : P(T <= x) >= level}``.

        Args:
            level: Level in ``(0, 1)``.

        Equation:
            [EQ-RISK-VAR-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        q = require_probability(level, name="level")
        cumulative = 0.0
        for v, p in zip(self.values, self.probabilities, strict=True):
            cumulative += p
            if cumulative >= q - 1e-12:
                return v
        return self.values[-1]


@dataclass(frozen=True, slots=True)
class WaterfallStep:
    """One mitigation step in a risk-reduction waterfall.

    Args:
        risk: Risk mitigated in this step.
        reduction: Drop in expected total exposure.
        cost: Mitigation cost.
        exposure_after: Expected total exposure after this and earlier steps.
        cumulative_cost: Mitigation cost of this and earlier steps.
        net_benefit: ``reduction - cost``.

    References:
        [REF-NASA-RMH-2011].

    """

    risk: str
    reduction: float
    cost: float
    exposure_after: float
    cumulative_cost: float
    net_benefit: float


@dataclass(frozen=True)
class RiskRegister:
    """A set of independent risks with exact totals and mitigation analysis.

    Args:
        risks: Distinctly named risks.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-NASA-RMH-2011].

    """

    risks: tuple[Risk, ...]

    def __post_init__(self) -> None:
        """Validate the register.

        Raises:
            TypeError: If an entry is not a :class:`Risk`.
            ValueError: If it is empty or names repeat.

        References:
            Project validation policy. No equation is evaluated.

        """
        risks = tuple(self.risks)
        if not risks:
            raise ValueError("at least one risk is required")
        for risk in risks:
            if not isinstance(risk, Risk):
                raise TypeError("risks must be Risk instances")
        names = [r.name for r in risks]
        if len(set(names)) != len(names):
            raise ValueError("risk names must be distinct")
        object.__setattr__(self, "risks", risks)

    def expected_exposure(self, *, mitigated: bool = False) -> float:
        """Return ``E[T] = sum p_i E[I_i]``.

        Args:
            mitigated: Whether to apply every mitigation.

        Equation:
            [EQ-RISK-REG-001], [EQ-RISK-REG-002].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        return math.fsum(r.expected_exposure(mitigated=mitigated) for r in self.risks)

    def exposure_variance(self, *, mitigated: bool = False) -> float:
        """Return ``Var[T] = sum Var[X_i]`` (independent risks).

        Args:
            mitigated: Whether to apply every mitigation.

        Equation:
            [EQ-RISK-REG-002].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        return math.fsum(r.exposure_variance(mitigated=mitigated) for r in self.risks)

    def totals_by(
        self, attribute: str, *, mitigated: bool = False
    ) -> Mapping[str, float]:
        """Return expected exposure grouped by ``"category"`` or ``"owner"``.

        Args:
            attribute: ``"category"`` or ``"owner"``.
            mitigated: Whether to apply the mitigations.

        Returns:
            Read-only group-to-expected-exposure mapping.

        Equation:
            [EQ-RISK-REG-002].

        References:
            [REF-NASA-RMH-2011].

        """
        if attribute not in ("category", "owner"):
            raise ValueError("attribute must be 'category' or 'owner'")
        groups: dict[str, list[float]] = {}
        for r in self.risks:
            groups.setdefault(getattr(r, attribute), []).append(
                r.expected_exposure(mitigated=mitigated)
            )
        return MappingProxyType({k: math.fsum(v) for k, v in groups.items()})

    def ranking(self, by: str = "exposure") -> tuple[Risk, ...]:
        """Return risks in descending order of a key (stable on ties).

        Args:
            by: ``"exposure"`` (``p E[I]``), ``"probability"``, ``"impact"``
                (``E[I]``), ``"roi"`` (mitigation ROI; NaN last) or
                ``"reduction"``.

        Returns:
            Sorted risks.

        Raises:
            ValueError: If the key is unknown.

        Equation:
            [EQ-RISK-REG-001], [EQ-RISK-REG-003].

        References:
            [REF-NASA-RMH-2011].

        """
        if by not in RANKING_KEYS:
            raise ValueError(f"by must be one of {RANKING_KEYS}")

        def key(risk: Risk) -> float:
            """Return the sort key of one risk.

            References:
                [EQ-RISK-REG-001]. No new equation.

            """
            if by == "exposure":
                value = risk.expected_exposure()
            elif by == "probability":
                value = risk.probability
            elif by == "impact":
                value = _moments(risk.impact)[0]
            elif by == "roi":
                value = risk.mitigation_roi
            else:
                value = risk.risk_reduction
            return -math.inf if math.isnan(value) else value

        return tuple(sorted(self.risks, key=key, reverse=True))

    def exposure_distribution(
        self, *, mitigated: bool = False, max_support: int = 200_000
    ) -> ExposureDistribution:
        """Return the exact distribution of the total by convolution.

        Values are added in exact rational arithmetic (``Fraction`` of each
        float), so equal totals reached along different paths merge exactly.

        Args:
            mitigated: Whether to apply every mitigation.
            max_support: Largest allowed number of distinct totals.

        Returns:
            Exact finite distribution.

        Raises:
            TypeError: If any impact is a continuous law (use
                :meth:`simulate`).
            ValueError: If the support would exceed ``max_support``.

        Equation:
            [EQ-RISK-REG-004].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        distribution: dict[Fraction, float] = {Fraction(0): 1.0}
        for risk in self.risks:
            p, impact = risk.state(mitigated=mitigated)
            if isinstance(impact, float):
                atoms: tuple[tuple[float, float], ...] = ((impact, 1.0),)
            elif isinstance(impact, DiscreteImpact):
                atoms = tuple(zip(impact.values, impact.probabilities, strict=True))
            else:
                raise TypeError(
                    f"risk {risk.name!r} has a continuous impact; use simulate()"
                )
            law: dict[Fraction, float] = {Fraction(0): 1.0 - p}
            for value, q in atoms:
                key = Fraction(value)
                law[key] = law.get(key, 0.0) + p * q
            combined: dict[Fraction, float] = {}
            for a, pa in distribution.items():
                if pa == 0.0:
                    continue
                for b, pb in law.items():
                    if pb == 0.0:
                        continue
                    combined[a + b] = combined.get(a + b, 0.0) + pa * pb
            if len(combined) > max_support:
                raise ValueError(
                    "exact support exceeds max_support; use simulate() instead"
                )
            distribution = combined
        keys = sorted(distribution)
        return ExposureDistribution(
            tuple(float(k) for k in keys), tuple(distribution[k] for k in keys)
        )

    def simulate(
        self,
        samples: int,
        *,
        mitigated: bool = False,
        seed: int | RandomStream | None = None,
    ) -> tuple[float, ...]:
        """Return ascending Monte Carlo totals for any impact laws.

        Occurrences are independent Bernoulli draws; impacts are drawn by
        inverse transform ``I = F^-1(U)`` independently of everything else.

        Args:
            samples: Positive sample size.
            mitigated: Whether to apply every mitigation.
            seed: Integer seed or stream.

        Returns:
            Sorted total exposures.

        Equation:
            [EQ-RISK-REG-001], [EQ-SAMP-INV-001].

        References:
            [REF-DEVROYE-1986, Section II.2].

        """
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        generator = resolve_stream(seed).generator()
        states = [r.state(mitigated=mitigated) for r in self.risks]
        totals: list[float] = []
        tiny = 2.0**-53
        for _ in range(samples):
            parts: list[float] = []
            for p, impact in states:
                occurs = generator.random() < p
                u = min(max(generator.random(), tiny), 1.0 - tiny)
                if not occurs:
                    continue
                parts.append(
                    impact if isinstance(impact, float) else float(impact.quantile(u))
                )
            totals.append(math.fsum(parts))
        return tuple(sorted(totals))

    def mitigation_waterfall(self, order: str = "roi") -> tuple[WaterfallStep, ...]:
        """Return the expected-exposure waterfall of the specified mitigations.

        Args:
            order: ``"roi"`` (highest ROI first) or ``"given"`` (register
                order). Risks without mitigation are skipped.

        Returns:
            Steps from the unmitigated expected total down to the fully
            mitigated one; reductions add exactly for independent risks.

        Raises:
            ValueError: If ``order`` is unknown.

        Equation:
            [EQ-RISK-REG-003].

        References:
            [REF-NASA-RMH-2011].

        """
        if order == "roi":
            candidates: Sequence[Risk] = [
                r for r in self.ranking("roi") if r.has_mitigation
            ]
        elif order == "given":
            candidates = [r for r in self.risks if r.has_mitigation]
        else:
            raise ValueError("order must be 'roi' or 'given'")
        exposure = self.expected_exposure()
        spent = 0.0
        steps: list[WaterfallStep] = []
        for risk in candidates:
            reduction = risk.risk_reduction
            exposure -= reduction
            spent += risk.mitigation_cost
            steps.append(
                WaterfallStep(
                    risk.name,
                    reduction,
                    risk.mitigation_cost,
                    exposure,
                    spent,
                    reduction - risk.mitigation_cost,
                )
            )
        return tuple(steps)


def register_from_rows(rows: Iterable[Mapping[str, object]]) -> RiskRegister:
    """Build a register from dictionaries (for example parsed CSV rows).

    Args:
        rows: Mappings with the :class:`Risk` field names as keys.

    Returns:
        Validated register.

    References:
        Project validation policy. No equation is evaluated.

    """
    return RiskRegister(tuple(Risk(**dict(row)) for row in rows))  # type: ignore[arg-type]
