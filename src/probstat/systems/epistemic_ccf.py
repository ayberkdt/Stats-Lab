"""Common-cause failure parameters as epistemic inputs.

The parameters of the beta-factor, multiple Greek letter (MGL) and
alpha-factor models are estimated from sparse event data and are uncertain
(Mosleh et al. 1998). This module treats them, and optionally the members'
total failure probability ``Q_t``, as epistemic quantities and propagates
them to the system reliability ``R`` of a :class:`~probstat.systems.SystemModel`
at a fixed time.

Parameters are addressed by ``(group index, name)``: ``"beta"`` for a
beta-factor group; ``"beta"``, ``"gamma"``, ``"delta"``, ... for the MGL
factors; ``"alpha_k"`` (``k >= 2``) for an alpha-factor group, whose
``alpha_1`` absorbs the remainder ``1 - sum_(k >= 2) alpha_k``; and
``"total"`` for the shared ``Q_t`` of a group's members at the analysis time.

- :func:`ccf_parameter_uncertainty` returns the mean and variance of ``R``
  and the share of ``Var[R]`` explained by each parameter and by all CCF
  parameters jointly (first-order variance-based indices). Under the
  probability split ``Q_k = c_k Q_t`` the beta-factor and MGL coefficients
  are polynomials in the parameters, so the Shannon recursion of the BDD
  with polynomial event probabilities in the mixed-moment basis of
  EQ-SYS-MOM-002 gives ``R(theta)`` exactly and the moments *exactly*
  (EQ-SYS-ECCF-001); the cross moment of two copies that share only the
  parameters of interest gives the indices exactly (EQ-SYS-ECCF-002).
  Otherwise (hazard split, uncertain alpha factors) the moments and
  indices are estimated by seeded Monte Carlo with the Saltelli (2010)
  first-order estimator (EQ-SYS-ECCF-003).
- :func:`ccf_interval_bounds` treats parameters known only to lie in
  intervals: it checks numerically whether the unreliability is monotone
  in each parameter, bounds it over the box (exact at the corners for a
  monotone response) and returns the probability box spanned by the
  corner distributions when other parameters have distributions
  (EQ-SYS-ECCF-004).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Literal

from probstat.distributions import Beta
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_probability
from probstat.modeling import IntervalInput, RandomInput
from probstat.numerics import integrate
from probstat.propagation import ProbabilityBox

from .bdd import BDD, MixedMoment
from .ccf import (
    AlphaFactorGroup,
    BetaFactorGroup,
    CCFGroup,
    MultipleGreekLetterGroup,
    expand_common_cause,
)
from .structure import SystemStructure
from .system import SystemModel

ParameterKey = tuple[int, str]
Measure = Literal["reliability", "availability"]
_GREEK = (
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
)
_SUPPORT_TOLERANCE = 1e-12
_MAX_CORNERS = 1024


@dataclass(frozen=True, slots=True)
class EpistemicCCFResult:
    """Epistemic mean, variance and variance shares of the system reliability.

    Args:
        time: Analysis time ``t``.
        measure: ``"reliability"`` (mission) or ``"availability"``.
        basis: CCF split used (``"probability"`` or ``"hazard"``).
        method: ``"exact"`` (BDD moments) or ``"monte-carlo"``.
        parameters: Labels ``"<group>:<name>"`` of the uncertain parameters.
        nominal: ``R`` at the nominal (input-model) parameter values.
        mean: ``E[R]``.
        unreliability_mean: ``E[1 - R]``, computed directly.
        variance: ``Var[R]``.
        standard_deviation: ``sqrt(Var[R])``.
        mean_standard_error: Monte Carlo standard error of ``mean`` (zero
            for the exact method).
        first_order: Label -> ``Var(E[R | theta]) / Var[R]``.
        ccf_contribution: ``Var(E[R | all CCF parameters]) / Var[R]``: the
            share of ``Var[R]`` due to CCF-parameter uncertainty (one when
            only CCF parameters are uncertain).
        ccf_variance: ``ccf_contribution * variance``.
        samples: Monte Carlo base sample size (zero for the exact method).

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-ECCF-002], [REF-APOSTOLAKIS-KAPLAN-1981],
        [REF-SOBOL-1993].

    """

    time: float
    measure: str
    basis: str
    method: str
    parameters: tuple[str, ...]
    nominal: float
    mean: float
    unreliability_mean: float
    variance: float
    standard_deviation: float
    mean_standard_error: float
    first_order: Mapping[str, float]
    ccf_contribution: float
    ccf_variance: float
    samples: int


@dataclass(frozen=True, slots=True)
class CCFIntervalBounds:
    """Unreliability bounds and probability box over interval CCF parameters.

    Args:
        time: Analysis time.
        measure: ``"reliability"`` or ``"availability"``.
        parameters: Labels of the interval parameters.
        monotonicity: Label -> ``"increasing"``, ``"decreasing"``,
            ``"constant"`` or ``"non-monotone"`` (the unreliability along the
            parameter's grid, other parameters at interval midpoints and
            distribution means).
        monotone: Whether every parameter was found monotone (then the
            bounds are attained at corners of the box).
        corners: Interval-parameter values at every corner.
        corner_unreliability: Mean unreliability at every corner (over the
            samples of the distributed parameters, if any).
        unreliability_lower: Smallest mean unreliability over corners and
            grid points.
        unreliability_upper: Largest mean unreliability over corners and
            grid points.
        reliability_lower: ``1 - unreliability_upper``.
        reliability_upper: ``1 - unreliability_lower``.
        pbox: Envelope of the corner distributions of the unreliability.
        samples: Samples of the distributed parameters per corner (one when
            none are distributed).

    References:
        [EQ-SYS-ECCF-004], [REF-FERSON-ET-AL-2003], [REF-MOSLEH-ETAL-1998].

    """

    time: float
    measure: str
    parameters: tuple[str, ...]
    monotonicity: Mapping[str, str]
    monotone: bool
    corners: tuple[Mapping[str, float], ...]
    corner_unreliability: tuple[float, ...]
    unreliability_lower: float
    unreliability_upper: float
    reliability_lower: float
    reliability_upper: float
    pbox: ProbabilityBox
    samples: int

    def contains(self, unreliability: float) -> bool:
        """Return whether a value lies within the unreliability bounds.

        Args:
            unreliability: Candidate unreliability.

        Returns:
            ``unreliability_lower <= value <= unreliability_upper``.

        References:
            [EQ-SYS-ECCF-004]. No further equation is evaluated.

        """
        return self.unreliability_lower <= unreliability <= self.unreliability_upper


def _label(key: ParameterKey) -> str:
    """Return the display label ``"<group>:<name>"`` of a parameter key.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    return f"{key[0]}:{key[1]}"


def _parameter_names(group: CCFGroup) -> tuple[str, ...]:
    """Return the parameter names a group accepts.

    References:
        [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

    """
    m = len(group.components)
    if isinstance(group, BetaFactorGroup):
        return ("beta", "total")
    if isinstance(group, MultipleGreekLetterGroup):
        return (*_GREEK[: m - 1], "total")
    if isinstance(group, AlphaFactorGroup):
        return (*(f"alpha_{k}" for k in range(2, m + 1)), "total")
    return ("total",)


def _nominal_value(group: CCFGroup, name: str) -> float | None:
    """Return a parameter's nominal value (``None`` for ``"total"``).

    References:
        [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

    """
    if name == "total":
        return None
    if isinstance(group, BetaFactorGroup):
        return group.beta
    if isinstance(group, MultipleGreekLetterGroup):
        return group.factors[_GREEK.index(name)]
    assert isinstance(group, AlphaFactorGroup)
    return group.alphas[int(name.split("_")[1]) - 1]


def _with_values(group: CCFGroup, values: Mapping[str, float]) -> CCFGroup:
    """Return a copy of ``group`` with some parameters replaced.

    For an alpha-factor group ``alpha_1 = 1 - sum_(k >= 2) alpha_k``.

    Raises:
        ValueError: If a replaced value is invalid for the model.

    Equation:
        [EQ-SYS-CCF-001], [EQ-SYS-CCF-002], [EQ-SYS-CCF-003].

    References:
        [REF-MOSLEH-ETAL-1998].

    """
    changed = {name: value for name, value in values.items() if name != "total"}
    if not changed:
        return group
    if isinstance(group, BetaFactorGroup):
        return BetaFactorGroup(group.components, changed["beta"])
    if isinstance(group, MultipleGreekLetterGroup):
        factors = list(group.factors)
        for name, value in changed.items():
            factors[_GREEK.index(name)] = value
        return MultipleGreekLetterGroup(group.components, tuple(factors))
    assert isinstance(group, AlphaFactorGroup)
    alphas = list(group.alphas)
    for name, value in changed.items():
        alphas[int(name.split("_")[1]) - 1] = value
    remainder = 1.0 - math.fsum(alphas[1:])
    if remainder < -1e-12:
        raise ValueError("alpha factors k >= 2 must sum to at most one")
    alphas[0] = max(0.0, remainder)
    return AlphaFactorGroup(group.components, tuple(alphas), group.testing)


def _law_of(value: object, *, name: str) -> object:
    """Validate an epistemic law supported on ``[0, 1]``.

    Accepts a :class:`~probstat.modeling.RandomInput` (its distribution is
    used) or any law with ``cdf``, ``quantile``, ``mean`` and ``variance``.

    Raises:
        TypeError: If the law lacks a required method.
        ValueError: If the law puts mass outside ``[0, 1]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        [REF-APOSTOLAKIS-KAPLAN-1981].

    """
    law = value.distribution if isinstance(value, RandomInput) else value
    for attribute in ("cdf", "quantile", "mean", "variance"):
        if not hasattr(law, attribute):
            raise TypeError(f"law of {name} lacks {attribute!r}")
    low = float(law.cdf(0.0))  # type: ignore[attr-defined]
    high = float(law.cdf(1.0))  # type: ignore[attr-defined]
    if low > _SUPPORT_TOLERANCE or high < 1.0 - _SUPPORT_TOLERANCE:
        raise ValueError(
            f"law of {name} must be supported on [0, 1]; truncate it "
            "(probstat.distributions.Truncated) first"
        )
    return law


def _point_moment(value: float) -> MixedMoment:
    """Return ``(j, k) -> v^j (1 - v)^k`` for a point mass at ``v``.

    References:
        [EQ-SYS-MOM-002]. No further equation is evaluated.

    """

    def moment(j: int, k: int) -> float:
        """Return the point-mass mixed moment.

        References:
            [EQ-SYS-MOM-002]. No further equation is evaluated.

        """
        return value**j * (1.0 - value) ** k

    return moment


def _law_moment(law: object) -> MixedMoment:
    """Return ``(j, k) -> E[q^j (1 - q)^k]`` for a law on ``[0, 1]``.

    A :class:`~probstat.distributions.Beta` law uses the closed form of
    EQ-SYS-MOM-002; any other law is integrated against its ``pdf`` by
    adaptive quadrature.

    Raises:
        TypeError: If a non-beta law has no ``pdf``.

    Equation:
        [EQ-SYS-MOM-002].

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981], [REF-PIESSENS-QUADPACK-1983].

    """
    cache: dict[tuple[int, int], float] = {}
    if isinstance(law, Beta):
        a, b = law.alpha, law.beta

        def closed(j: int, k: int) -> float:
            """Return the beta mixed moment.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            value = 1.0
            for i in range(j):
                value *= (a + i) / (a + b + i)
            for i in range(k):
                value *= (b + i) / (a + b + j + i)
            return value

        return closed
    density = getattr(law, "pdf", None)
    if not callable(density):
        raise TypeError("exact moments need a law with pdf(x)")

    def numeric(j: int, k: int) -> float:
        """Integrate ``q^j (1 - q)^k f(q)`` over ``[0, 1]``.

        References:
            [EQ-SYS-MOM-002], [REF-PIESSENS-QUADPACK-1983].

        """
        key = (j, k)
        if key not in cache:
            result = integrate(
                lambda q: q**j * (1.0 - q) ** k * float(density(q)),
                0.0,
                1.0,
                rel_tol=1e-12,
            )
            cache[key] = min(1.0, max(0.0, result.value))
        return cache[key]

    return numeric


def _resolve_measure(value: object) -> Measure:
    """Validate a measure name.

    Raises:
        ValueError: If the measure is unknown.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if value == "reliability":
        return "reliability"
    if value == "availability":
        return "availability"
    raise ValueError("measure must be 'reliability' or 'availability'")


def _require_time(value: object) -> float:
    """Validate a finite non-negative time.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is negative or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("t must be a real number")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError("t must be finite and non-negative")
    return checked


class _Evaluator:
    """System reliability as a function of CCF parameter values at one time.

    The structure is expanded with every CCF event of every order, so that
    parameter values which make a coefficient zero (or positive) need no
    recompilation.

    References:
        [EQ-SYS-CCF-005], [EQ-SYS-CCF-006], [REF-MOSLEH-ETAL-1998].

    """

    def __init__(self, system: SystemModel, t: float, measure: Measure) -> None:
        """Tabulate component probabilities and compile the full expansion.

        Raises:
            TypeError: If ``system`` is not a :class:`SystemModel`.
            ValueError: If the system has no CCF group or members disagree.

        References:
            [EQ-SYS-CCF-005], [REF-MOSLEH-ETAL-1998].

        """
        if not isinstance(system, SystemModel):
            raise TypeError("system must be a SystemModel")
        groups = system.ccf_groups
        if not groups:
            raise ValueError("the system has no CCF group")
        self.system = system
        self.groups = groups
        self.measure = measure
        basis = system.ccf_basis
        if basis == "auto":
            basis = "hazard" if measure == "reliability" else "probability"
        self.basis = basis
        expansion = system.ccf_expansion
        assert expansion is not None
        group_of = expansion.group_of
        self.up: dict[str, float] = {}
        self.down: dict[str, float] = {}
        totals: dict[int, float] = {}
        for name, model in system.components.items():
            if measure == "reliability":
                up, down = model.reliability(t), model.unreliability(t)
            else:
                up, down = model.availability(t), model.unavailability(t)
            up = require_probability(up, name=f"P({name} up)")
            down = require_probability(down, name=f"P({name} down)")
            if name not in group_of:
                self.up[name], self.down[name] = up, down
                continue
            known = totals.setdefault(group_of[name], down)
            if not math.isclose(known, down, rel_tol=1e-9, abs_tol=1e-300):
                raise ValueError(
                    f"members of CCF group {group_of[name]} need equal failure "
                    "probabilities"
                )
        self.totals = totals
        stand_ins = [
            AlphaFactorGroup(
                group.components,
                tuple(1.0 / len(group.components) for _ in group.components),
                "staggered",
            )
            for group in groups
        ]
        self.full = expand_common_cause(system.base_structure.expression, stand_ins)
        self.structure = SystemStructure(self.full.expression)

    def parameter_keys(self) -> dict[ParameterKey, CCFGroup]:
        """Return every admissible parameter key with its group.

        References:
            [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

        """
        return {
            (index, name): group
            for index, group in enumerate(self.groups)
            for name in _parameter_names(group)
        }

    def nominal(self, key: ParameterKey) -> float:
        """Return the nominal value of a parameter.

        References:
            [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

        """
        value = _nominal_value(self.groups[key[0]], key[1])
        return self.totals[key[0]] if value is None else value

    def unreliability(
        self, values: Mapping[ParameterKey, float]
    ) -> tuple[float, float]:
        """Return ``(R, 1 - R)`` for given parameter values.

        Raises:
            ValueError: If a value is invalid for its model.

        Equation:
            [EQ-SYS-CCF-005], [EQ-SYS-CCF-006], [EQ-SYS-BDD-001].

        References:
            [REF-MOSLEH-ETAL-1998], [REF-RAUZY-1993].

        """
        fractions: dict[int, tuple[float, ...]] = {}
        totals = dict(self.totals)
        per_group: dict[int, dict[str, float]] = {}
        for (index, name), value in values.items():
            per_group.setdefault(index, {})[name] = value
        for index, group in enumerate(self.groups):
            chosen = per_group.get(index, {})
            fractions[index] = tuple(_with_values(group, chosen).order_fractions())
            if "total" in chosen:
                totals[index] = require_probability(chosen["total"], name="total")
        p = dict(self.up)
        q = dict(self.down)
        group_of = self.full.group_of
        for event in self.full.events.values():
            index = group_of[event.members[0]]
            fraction = fractions[index][event.order - 1]
            total = totals[index]
            if self.basis == "probability":
                down = min(1.0, fraction * total)
                p[event.name], q[event.name] = 1.0 - down, down
            elif total >= 1.0:
                down = 0.0 if fraction == 0.0 else 1.0
                p[event.name], q[event.name] = 1.0 - down, down
            else:
                exponent = fraction * math.log1p(-total)
                p[event.name], q[event.name] = math.exp(exponent), -math.expm1(exponent)
        bdd, root = self.structure.bdd, self.structure.root
        return (
            bdd.probability(root, p, q, value=True),
            bdd.probability(root, p, q, value=False),
        )


def _validate_keys(
    evaluator: _Evaluator, keys: Sequence[ParameterKey]
) -> dict[ParameterKey, CCFGroup]:
    """Check parameter keys against the system's groups.

    Raises:
        TypeError: If a key is not an ``(int, str)`` pair.
        KeyError: If a key names an unknown group or parameter.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    admissible = evaluator.parameter_keys()
    for key in keys:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or not isinstance(key[1], str)
        ):
            raise TypeError("parameter keys must be (group index, name) pairs")
        if key not in admissible:
            raise KeyError(f"unknown CCF parameter {key!r}")
    return admissible


# ---------------------------------------------------------------- exact path
def _exact_supported(evaluator: _Evaluator, keys: Sequence[ParameterKey]) -> bool:
    """Return whether the moments are polynomial in the uncertain parameters.

    References:
        [EQ-SYS-ECCF-001]. No further equation is evaluated.

    """
    if evaluator.basis != "probability":
        return False
    for index, name in keys:
        if name == "total":
            continue
        if not isinstance(
            evaluator.groups[index], BetaFactorGroup | MultipleGreekLetterGroup
        ):
            return False
    return True


_Clause = tuple[ParameterKey, bool]


def _event_factors(
    evaluator: _Evaluator,
    index: int,
    order: int,
    uncertain: Sequence[ParameterKey],
) -> tuple[float, list[_Clause]]:
    """Return the constant factor and uncertain literals of one CCF event.

    The event probability is ``constant * prod_(literals) (theta or 1 -
    theta)`` [EQ-SYS-ECCF-001].

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-CCF-001], [EQ-SYS-CCF-002].

    """
    group = evaluator.groups[index]
    m = len(group.components)
    literals: list[_Clause] = []
    constant = 1.0
    if (index, "total") in uncertain:
        literals.append(((index, "total"), True))
    else:
        constant *= evaluator.totals[index]
    if isinstance(group, BetaFactorGroup):
        key = (index, "beta")
        if order not in (1, m):
            return 0.0, []
        if key in uncertain:
            literals.append((key, order == m))
        else:
            constant *= group.beta if order == m else 1.0 - group.beta
        return constant, literals
    if isinstance(group, MultipleGreekLetterGroup):
        for position in range(2, order + 1):
            key = (index, _GREEK[position - 2])
            if key in uncertain:
                literals.append((key, True))
            else:
                constant *= group.factors[position - 2]
        if order < m:
            key = (index, _GREEK[order - 1])
            if key in uncertain:
                literals.append((key, False))
            else:
                constant *= 1.0 - group.factors[order - 1]
        constant /= math.comb(m - 1, order - 1)
        return constant, literals
    return constant * group.order_fractions()[order - 1], literals


Monomial = tuple[int, ...]
Polynomial = dict[Monomial, float]


def _add_into(target: Polynomial, source: Polynomial, scale: float = 1.0) -> None:
    """Add ``scale * source`` to ``target`` in place.

    References:
        [EQ-SYS-ECCF-001]. No further equation is evaluated.

    """
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + scale * value


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    """Return the product of two polynomials in the ``theta, 1 - theta`` basis.

    References:
        [EQ-SYS-ECCF-001]. No further equation is evaluated.

    """
    result: Polynomial = {}
    for a, x in left.items():
        for b, y in right.items():
            key = tuple(i + j for i, j in zip(a, b, strict=True))
            result[key] = result.get(key, 0.0) + x * y
    return result


def _event_polynomials(
    constant: float, literals: Sequence[tuple[int, bool]], width: int
) -> tuple[Polynomial, Polynomial]:
    """Return ``(P(event does not occur), P(event occurs))`` as polynomials.

    The occurrence probability is ``a prod_l lit_l`` with ``lit = theta`` or
    ``1 - theta``. Its complement is written with non-negative coefficients
    by expanding ``1 = prod_l (lit_l + (1 - lit_l))``: every product except
    ``prod_l lit_l`` enters with weight one and that product with ``1 - a``.

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-MOM-002].

    """
    zero = (0,) * (2 * width)
    if constant <= 0.0:
        return {zero: 1.0}, {}

    def monomial(flags: Sequence[bool]) -> Monomial:
        """Return the exponent tuple of a product of literals or complements.

        References:
            [EQ-SYS-ECCF-001]. No further equation is evaluated.

        """
        exponent = [0] * (2 * width)
        for (parameter, positive), keep in zip(literals, flags, strict=True):
            exponent[2 * parameter + (0 if positive == keep else 1)] += 1
        return tuple(exponent)

    occurs = {monomial([True] * len(literals)): min(1.0, constant)}
    survives: Polynomial = {}
    for flags in product((True, False), repeat=len(literals)):
        key = monomial(flags)
        weight = max(0.0, 1.0 - constant) if all(flags) else 1.0
        if weight > 0.0:
            survives[key] = survives.get(key, 0.0) + weight
    return survives, occurs


def _reliability_polynomials(
    evaluator: _Evaluator, uncertain: Sequence[ParameterKey]
) -> tuple[Polynomial, Polynomial]:
    """Return ``R(theta)`` and ``1 - R(theta)`` as exact polynomials.

    The Shannon recursion ``P(f) = p_x P(f_1) + q_x P(f_0)`` is evaluated on
    the expanded structure's BDD with polynomial event probabilities in the
    basis ``prod_i theta_i^(j_i) (1 - theta_i)^(k_i)``; every coefficient is
    non-negative, so no cancellation occurs.

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-BDD-001], [REF-RAUZY-1993].

    """
    width = len(uncertain)
    position = {key: i for i, key in enumerate(uncertain)}
    zero = (0,) * (2 * width)
    tables: dict[str, tuple[Polynomial, Polynomial]] = {}
    for name, chance in evaluator.up.items():
        tables[name] = ({zero: chance}, {zero: evaluator.down[name]})
    group_of = evaluator.full.group_of
    for event in evaluator.full.events.values():
        index = group_of[event.members[0]]
        constant, literals = _event_factors(evaluator, index, event.order, uncertain)
        indexed = [(position[key], positive) for key, positive in literals]
        tables[event.name] = _event_polynomials(constant, indexed, width)
    bdd, root = evaluator.structure.bdd, evaluator.structure.root
    works: dict[int, Polynomial] = {BDD.FALSE: {}, BDD.TRUE: {zero: 1.0}}
    fails: dict[int, Polynomial] = {BDD.FALSE: {zero: 1.0}, BDD.TRUE: {}}
    nodes = sorted(
        (n for n in bdd.nodes(root) if n >= 2),
        key=lambda n: bdd.level(bdd.node(n)[0]),
        reverse=True,
    )
    for node in nodes:
        name, low, high = bdd.node(node)
        survives, occurs = tables[name]
        for memo in (works, fails):
            value = _multiply(survives, memo[high])
            _add_into(value, _multiply(occurs, memo[low]))
            memo[node] = value
    return works[root], fails[root]


def _expectation(polynomial: Polynomial, moments: Sequence[MixedMoment]) -> float:
    """Return ``E[poly(theta)]`` for independent parameters.

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-MOM-002].

    """
    return math.fsum(
        coefficient
        * math.prod(moments[i](key[2 * i], key[2 * i + 1]) for i in range(len(moments)))
        for key, coefficient in polynomial.items()
    )


def _conditional_second_moment(
    polynomial: Polynomial, moments: Sequence[MixedMoment], shared: set[int]
) -> float:
    """Return ``E[g(theta_S, theta) g(theta_S, theta')]`` (EQ-SYS-ECCF-002).

    The parameters in ``shared`` are common to both factors, the others are
    independent copies; ``shared`` containing every parameter gives
    ``E[g^2]``.

    References:
        [EQ-SYS-ECCF-002], [REF-SOBOL-1993].

    """
    width = len(moments)
    weights: dict[Monomial, float] = {}
    for key, coefficient in polynomial.items():
        part = tuple(key[2 * i + s] for i in sorted(shared) for s in (0, 1))
        rest = math.prod(
            moments[i](key[2 * i], key[2 * i + 1])
            for i in range(width)
            if i not in shared
        )
        weights[part] = weights.get(part, 0.0) + coefficient * rest
    order = sorted(shared)
    terms = []
    for a, x in weights.items():
        for b, y in weights.items():
            terms.append(
                x
                * y
                * math.prod(
                    moments[i](a[2 * n] + b[2 * n], a[2 * n + 1] + b[2 * n + 1])
                    for n, i in enumerate(order)
                )
            )
    return math.fsum(terms)


def _exact_analysis(
    evaluator: _Evaluator, uncertain: Mapping[ParameterKey, object]
) -> tuple[float, float, float, dict[str, float], float]:
    """Return exact ``E[R]``, ``E[1 - R]``, ``Var[R]`` and variance shares.

    ``Var(E[R | theta_S]) = E[g g'] - E[g]^2`` with ``g`` the reliability or
    its complement polynomial, whichever has the smaller mean, and ``g'`` a
    copy sharing only ``theta_S`` (EQ-SYS-ECCF-002).

    References:
        [EQ-SYS-ECCF-001], [EQ-SYS-ECCF-002], [REF-SOBOL-1993].

    """
    keys = list(uncertain)
    moments = [_law_moment(uncertain[key]) for key in keys]
    works, fails = _reliability_polynomials(evaluator, keys)
    mean = min(1.0, max(0.0, _expectation(works, moments)))
    complement = min(1.0, max(0.0, _expectation(fails, moments)))
    small, base = (works, mean) if mean <= complement else (fails, complement)
    everything = set(range(len(keys)))

    def explained_variance(shared: set[int]) -> float:
        """Return ``Var(E[R | theta_shared])``.

        References:
            [EQ-SYS-ECCF-002]. No further equation is evaluated.

        """
        if not shared:
            return 0.0
        value = _conditional_second_moment(small, moments, shared)
        return max(0.0, value - base * base)

    variance = explained_variance(everything)

    def share(shared: set[int]) -> float:
        """Return ``Var(E[R | theta_shared]) / Var[R]``.

        References:
            [EQ-SYS-ECCF-002]. No further equation is evaluated.

        """
        if variance <= 0.0 or not shared:
            return 0.0
        if shared == everything:
            return 1.0
        return min(1.0, explained_variance(shared) / variance)

    first_order = {_label(key): share({i}) for i, key in enumerate(keys)}
    ccf = {i for i, key in enumerate(keys) if key[1] != "total"}
    return mean, complement, variance, first_order, share(ccf)


# ---------------------------------------------------------- Monte Carlo path
def _monte_carlo_analysis(
    evaluator: _Evaluator,
    uncertain: Mapping[ParameterKey, object],
    samples: int,
    stream: RandomStream,
) -> tuple[float, float, float, float, dict[str, float], float]:
    """Return Monte Carlo moments and Saltelli first-order shares.

    With independent base samples ``A`` and ``B`` and ``A_B^S`` (columns in
    ``S`` from ``B``), ``V_S ~ mean((f(B) - f_bar) (f(A_B^S) - f(A)))``
    (EQ-SYS-ECCF-003); centring ``f(B)`` by the sample mean leaves the
    estimator consistent and removes the noise term proportional to
    ``E[R]``, which dominates when ``R`` is close to one.

    References:
        [EQ-SYS-ECCF-003], [REF-SALTELLI-2010], [REF-SOBOL-1993].

    """
    keys = list(uncertain)
    generator = stream.child("epistemic-ccf").generator()

    def draw() -> list[float]:
        """Return one parameter vector by inversion.

        References:
            [EQ-SYS-ECCF-003], [REF-DEVROYE-1986, Chapter II].

        """
        row = []
        for key in keys:
            u = generator.random()
            u = u if u > 0.0 else 2.0**-53
            row.append(float(uncertain[key].quantile(u)))  # type: ignore[attr-defined]
        return row

    a_rows = [draw() for _ in range(samples)]
    b_rows = [draw() for _ in range(samples)]

    def model(row: Sequence[float]) -> tuple[float, float]:
        """Return ``(R, 1 - R)`` at one parameter vector.

        References:
            [EQ-SYS-CCF-005]. No further equation is evaluated.

        """
        return evaluator.unreliability(dict(zip(keys, row, strict=True)))

    fa = [model(row) for row in a_rows]
    fb = [model(row) for row in b_rows]
    values = [r for r, _ in fa + fb]
    count = len(values)
    mean = math.fsum(values) / count
    complement = math.fsum(u for _, u in fa + fb) / count
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    error = math.sqrt(variance / count)
    labels = [_label(key) for key in keys]

    def explained(columns: set[int]) -> float:
        """Return the Saltelli estimate of ``V_S / Var[R]``.

        References:
            [EQ-SYS-ECCF-003], [REF-SALTELLI-2010].

        """
        if variance <= 0.0 or not columns:
            return 0.0
        if columns == set(range(len(keys))):
            return 1.0
        total = []
        for a_row, b_row, (r_a, _), (r_b, _) in zip(
            a_rows, b_rows, fa, fb, strict=True
        ):
            mixed = [b_row[c] if c in columns else a_row[c] for c in range(len(keys))]
            total.append((r_b - mean) * (model(mixed)[0] - r_a))
        return min(1.0, max(0.0, math.fsum(total) / samples / variance))

    first_order = {label: explained({i}) for i, label in enumerate(labels)}
    ccf = {i for i, key in enumerate(keys) if key[1] != "total"}
    return mean, complement, variance, error, first_order, explained(ccf)


def ccf_parameter_uncertainty(
    system: SystemModel,
    t: float,
    parameters: Mapping[ParameterKey, object],
    *,
    measure: str = "reliability",
    method: str = "auto",
    samples: int = 4000,
    seed: int | RandomStream | None = 0,
) -> EpistemicCCFResult:
    """Propagate uncertain CCF parameters to the system reliability.

    Args:
        system: System with at least one CCF group.
        t: Analysis time.
        parameters: ``(group index, name)`` -> law on ``[0, 1]`` (or a
            :class:`~probstat.modeling.RandomInput`); the parameters are
            independent. Names: see the module docstring.
        measure: ``"reliability"`` (mission) or ``"availability"``.
        method: ``"auto"`` (exact when the coefficients are polynomial in the
            parameters, that is the probability split with beta-factor or MGL
            parameters and any ``"total"``; Monte Carlo otherwise),
            ``"exact"`` or ``"monte-carlo"``.
        samples: Monte Carlo base sample size (at least 10); the Monte Carlo
            path costs ``samples (2 + d + 1)`` system evaluations for ``d``
            parameters.
        seed: Integer seed or stream of the Monte Carlo path.

    Returns:
        Mean, variance, first-order shares and the CCF share of ``Var[R]``.

    Raises:
        TypeError: If an argument has the wrong type.
        KeyError: If a parameter key is unknown.
        ValueError: If ``parameters`` is empty, a law leaves ``[0, 1]``,
            ``"exact"`` is requested where it does not apply, or an argument
            is out of range.

    Equation:
        [EQ-SYS-ECCF-001], [EQ-SYS-ECCF-002], [EQ-SYS-ECCF-003].

    References:
        [REF-MOSLEH-ETAL-1998], [REF-APOSTOLAKIS-KAPLAN-1981],
        [REF-SALTELLI-2010].

    """
    chosen_measure = _resolve_measure(measure)
    time = _require_time(t)
    evaluator = _Evaluator(system, time, chosen_measure)
    if not parameters:
        raise ValueError("at least one uncertain parameter is required")
    _validate_keys(evaluator, list(parameters))
    uncertain = {key: _law_of(law, name=_label(key)) for key, law in parameters.items()}
    if method not in ("auto", "exact", "monte-carlo"):
        raise ValueError("method must be 'auto', 'exact' or 'monte-carlo'")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 10:
        raise ValueError("samples must be an integer of at least 10")
    exact = _exact_supported(evaluator, list(uncertain))
    if method == "exact" and not exact:
        raise ValueError(
            "exact moments need the probability split and beta-factor or MGL "
            "parameters (or 'total')"
        )
    nominal = evaluator.unreliability({})[0]
    labels = tuple(_label(key) for key in uncertain)
    if exact and method != "monte-carlo":
        mean, complement, variance, first_order, share = _exact_analysis(
            evaluator, uncertain
        )
        error, used, kind = 0.0, 0, "exact"
    else:
        mean, complement, variance, error, first_order, share = _monte_carlo_analysis(
            evaluator, uncertain, samples, resolve_stream(seed)
        )
        used, kind = samples, "monte-carlo"
    return EpistemicCCFResult(
        time=time,
        measure=chosen_measure,
        basis=evaluator.basis,
        method=kind,
        parameters=labels,
        nominal=nominal,
        mean=mean,
        unreliability_mean=complement,
        variance=variance,
        standard_deviation=math.sqrt(max(0.0, variance)),
        mean_standard_error=error,
        first_order=MappingProxyType(first_order),
        ccf_contribution=share,
        ccf_variance=share * variance,
        samples=used,
    )


# ------------------------------------------------------------- interval path
def _interval_of(value: object, *, name: str) -> tuple[float, float]:
    """Validate an interval given as :class:`IntervalInput` or a pair.

    Raises:
        TypeError: If the value is neither.
        ValueError: If a bound is not a probability or ``upper < lower``.

    References:
        [REF-FERSON-ET-AL-2003]. No equation is evaluated.

    """
    if isinstance(value, IntervalInput):
        low, high = value.lower, value.upper
    elif isinstance(value, tuple | list) and len(value) == 2:
        low, high = value
    else:
        raise TypeError(f"interval of {name} must be an IntervalInput or a pair")
    low = require_probability(low, name=f"{name} lower")
    high = require_probability(high, name=f"{name} upper")
    if high < low:
        raise ValueError(f"interval of {name} has upper < lower")
    return low, high


def _direction(values: Sequence[float]) -> str:
    """Classify a sequence as increasing, decreasing, constant or neither.

    References:
        [EQ-SYS-ECCF-004]. No further equation is evaluated.

    """
    scale = max((abs(v) for v in values), default=0.0)
    tolerance = 1e-12 * scale + 1e-300
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    if all(abs(d) <= tolerance for d in steps):
        return "constant"
    if all(d >= -tolerance for d in steps):
        return "increasing"
    if all(d <= tolerance for d in steps):
        return "decreasing"
    return "non-monotone"


def ccf_interval_bounds(
    system: SystemModel,
    t: float,
    intervals: Mapping[ParameterKey, object],
    *,
    distributions: Mapping[ParameterKey, object] | None = None,
    measure: str = "reliability",
    grid: int = 11,
    samples: int = 1000,
    seed: int | RandomStream | None = 0,
) -> CCFIntervalBounds:
    """Bound the system unreliability over interval-valued CCF parameters.

    Each interval parameter is scanned on ``grid`` points with the other
    interval parameters at their midpoints (and distributed parameters at
    their means) to classify the response as monotone or not. The
    unreliability is evaluated at every corner of the box; with
    ``distributions``, the same seeded samples of the distributed
    parameters are used at every corner (common random numbers), each
    corner yields an empirical distribution of the unreliability, and the
    probability box is their pointwise envelope (EQ-SYS-ECCF-004). When the
    response is monotone in every parameter the corner values are the exact
    bounds; otherwise the grid extremes are included and the bounds are
    inner approximations.

    Args:
        system: System with at least one CCF group.
        t: Analysis time.
        intervals: ``(group, name)`` -> :class:`~probstat.modeling.IntervalInput`
            or ``(lower, upper)``.
        distributions: Optional ``(group, name)`` -> law on ``[0, 1]`` for
            parameters that are distributed rather than interval-valued.
        measure: ``"reliability"`` or ``"availability"``.
        grid: Points per interval for the monotonicity check (at least 3).
        samples: Samples of the distributed parameters (ignored without
            them).
        seed: Integer seed or stream.

    Returns:
        Bounds, monotonicity flags and the probability box.

    Raises:
        TypeError: If an argument has the wrong type.
        KeyError: If a key is unknown.
        ValueError: If ``intervals`` is empty, a key is both interval and
            distributed, there are more than 10 interval parameters, or an
            argument is out of range.

    Equation:
        [EQ-SYS-ECCF-004].

    References:
        [REF-FERSON-ET-AL-2003], [REF-MOSLEH-ETAL-1998].

    """
    chosen_measure = _resolve_measure(measure)
    time = _require_time(t)
    evaluator = _Evaluator(system, time, chosen_measure)
    if not intervals:
        raise ValueError("at least one interval parameter is required")
    laws = dict(distributions or {})
    _validate_keys(evaluator, [*intervals, *laws])
    overlap = set(intervals) & set(laws)
    if overlap:
        raise ValueError(f"parameters {sorted(overlap)} are both interval and law")
    if len(intervals) > 10:
        raise ValueError("at most 10 interval parameters are supported")
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 3:
        raise ValueError("grid must be an integer of at least 3")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    boxes = {
        key: _interval_of(value, name=_label(key)) for key, value in intervals.items()
    }
    checked = {key: _law_of(law, name=_label(key)) for key, law in laws.items()}
    keys = list(boxes)
    labels = tuple(_label(key) for key in keys)
    means = {key: float(law.mean) for key, law in checked.items()}  # type: ignore[attr-defined]
    middle = {key: 0.5 * (low + high) for key, (low, high) in boxes.items()}
    monotonicity: dict[str, str] = {}
    grid_values: list[float] = []
    for key in keys:
        low, high = boxes[key]
        line = []
        for i in range(grid):
            point = dict(middle)
            point[key] = low + (high - low) * i / (grid - 1)
            point.update(means)
            line.append(evaluator.unreliability(point)[1])
        monotonicity[_label(key)] = _direction(line)
        grid_values.extend(line)
    if checked:
        generator = resolve_stream(seed).child("ccf-interval").generator()
        draws = []
        for _ in range(samples):
            row = {}
            for key, law in checked.items():
                u = generator.random()
                row[key] = float(law.quantile(u if u > 0.0 else 2.0**-53))  # type: ignore[attr-defined]
            draws.append(row)
        used = samples
    else:
        draws = [{}]
        used = 1
    corners: list[Mapping[str, float]] = []
    outputs: list[tuple[float, ...]] = []
    corner_means: list[float] = []
    for combination in product(*(boxes[key] for key in keys)):
        if len(corners) >= _MAX_CORNERS:
            break
        base = dict(zip(keys, combination, strict=True))
        values = sorted(evaluator.unreliability({**base, **row})[1] for row in draws)
        corners.append(MappingProxyType({_label(k): v for k, v in base.items()}))
        outputs.append(tuple(values))
        corner_means.append(math.fsum(values) / len(values))
    levels = sorted({value for row in outputs for value in row})
    cdfs = [[_ecdf(row, level) for level in levels] for row in outputs]
    pbox = ProbabilityBox(
        levels=tuple(levels),
        lower_cdf=tuple(min(c[i] for c in cdfs) for i in range(len(levels))),
        upper_cdf=tuple(max(c[i] for c in cdfs) for i in range(len(levels))),
        sorted_outputs=tuple(outputs),
    )
    candidates = corner_means + ([] if checked else grid_values)
    monotone = all(v != "non-monotone" for v in monotonicity.values())
    if checked and not monotone:
        candidates = candidates + grid_values
    lower, upper = min(candidates), max(candidates)
    return CCFIntervalBounds(
        time=time,
        measure=chosen_measure,
        parameters=labels,
        monotonicity=MappingProxyType(monotonicity),
        monotone=monotone,
        corners=tuple(corners),
        corner_unreliability=tuple(corner_means),
        unreliability_lower=lower,
        unreliability_upper=upper,
        reliability_lower=1.0 - upper,
        reliability_upper=1.0 - lower,
        pbox=pbox,
        samples=used,
    )


def _ecdf(sorted_values: Sequence[float], level: float) -> float:
    """Return the empirical CDF of sorted values at ``level``.

    References:
        [EQ-STAT-005]. No further equation is evaluated.

    """
    return bisect.bisect_right(sorted_values, level) / len(sorted_values)
