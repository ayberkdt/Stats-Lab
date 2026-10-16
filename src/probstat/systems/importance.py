"""Exact component importance measures from the structure BDD.

Notation: ``p_i`` is the reliability and ``q_i = 1 - p_i`` the unreliability
of component ``i``; ``Q`` is the system unreliability; ``Q(1_i)`` is ``Q``
when component ``i`` is failed for certain and ``Q(0_i)`` when it is perfect
(the fault-tree convention, in which the basic event "``i`` failed" is set to
1 or 0). All conditional probabilities come from one top-down and one
bottom-up pass over the BDD (:meth:`~probstat.systems.bdd.BDD.variable_effects`)
and are exact for independent components.

Measures (Rausand and Hoyland 2004; Birnbaum 1969; Lambert 1975; Fussell
1975):

* Birnbaum ``I_B = Q(1_i) - Q(0_i) = dR/dp_i``.
* Criticality (Lambert's failure criticality) ``I_CR = I_B q_i / Q``: the
  probability that ``i`` is failed and critical given the system failed.
* Fussell-Vesely, cut-set definition: ``I_FV = P(at least one minimal cut set
  containing i is failed) / Q``, evaluated exactly (the union probability is
  computed on a BDD of the cut sets that contain ``i``, not with the
  rare-event sum).
* Some PRA codes report as "Fussell-Vesely" the fractional reduction
  ``(Q - Q(0_i)) / Q = 1 - 1/RRW``. Because ``Q`` is linear in ``q_i``,
  ``Q - Q(0_i) = q_i I_B``, so this fraction is *identical* to the criticality
  importance and is not reported separately. It never exceeds the cut-set
  definition (``P(A) >= P(A and not B) = Q - Q(0_i)`` with ``A``, ``B`` the
  unions of the cut sets with and without ``i``).
* Risk achievement worth ``RAW = Q(1_i) / Q`` and risk reduction worth
  ``RRW = Q / Q(0_i)``.
* Improvement potential ``I_IP = R(p_i = 1) - R = I_B q_i``.
* Structural (Birnbaum structural) importance: ``I_B`` with every
  ``p_j = 1/2``, i.e. the fraction of the ``2^(n-1)`` states of the other
  components in which ``i`` is critical.

Undefined ratios (zero denominator) are ``None``; a positive numerator over
a zero denominator is ``math.inf``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .structure import SystemStructure


@dataclass(frozen=True, slots=True)
class ImportanceMeasures:
    """Exact importance measures of one component (or basic event).

    Args:
        component: Component name.
        reliability: ``p_i`` used in the evaluation.
        birnbaum: ``I_B = Q(1_i) - Q(0_i)``.
        criticality: ``I_B q_i / Q = (Q - Q(0_i)) / Q``.
        fussell_vesely: Cut-set definition, exact union probability over
            ``Q``.
        risk_achievement_worth: ``Q(1_i) / Q``.
        risk_reduction_worth: ``Q / Q(0_i)``.
        improvement_potential: ``I_B q_i``.
        structural: Birnbaum importance with every reliability ``1/2``.
        unreliability_if_failed: ``Q(1_i)``.
        unreliability_if_perfect: ``Q(0_i)``.

    References:
        [REF-BIRNBAUM-1969], [REF-LAMBERT-1975], [REF-FUSSELL-1975],
        [REF-RAUSAND-HOYLAND-2004].

    """

    component: str
    reliability: float
    birnbaum: float
    criticality: float | None
    fussell_vesely: float | None
    risk_achievement_worth: float | None
    risk_reduction_worth: float | None
    improvement_potential: float
    structural: float
    unreliability_if_failed: float
    unreliability_if_perfect: float


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a non-negative ratio with explicit undefined cases.

    Args:
        numerator: Non-negative numerator.
        denominator: Non-negative denominator.

    Returns:
        The ratio, ``math.inf`` for a positive numerator over zero, or
        ``None`` for ``0 / 0``.

    References:
        [EQ-SYS-IMP-001], [REF-RAUSAND-HOYLAND-2004].

    """
    if denominator > 0.0:
        return numerator / denominator
    return math.inf if numerator > 0.0 else None


def structural_importance(structure: SystemStructure) -> Mapping[str, float]:
    """Return Birnbaum's structural importance of every component.

    Args:
        structure: Compiled structure.

    Returns:
        Read-only mapping component -> number of critical state vectors of
        the other components divided by ``2^(n-1)``.

    Equation:
        [EQ-SYS-IMP-001].

    References:
        [REF-BIRNBAUM-1969].

    """
    half = dict.fromkeys(structure.components, 0.5)
    effects = structure.bdd.variable_effects(
        structure.root, half, half, structure.components
    )
    return MappingProxyType(
        {name: effects[name].birnbaum for name in structure.components}
    )


def importance_measures(
    structure: SystemStructure,
    reliabilities: Mapping[str, float],
    *,
    unreliabilities: Mapping[str, float] | None = None,
) -> Mapping[str, ImportanceMeasures]:
    """Return every exact importance measure for every component.

    Args:
        structure: Compiled structure.
        reliabilities: Component reliabilities ``p_i``.
        unreliabilities: Optional accurate ``q_i`` (recommended when some
            ``q_i`` are tiny).

    Returns:
        Read-only mapping component -> :class:`ImportanceMeasures`, in BDD
        order.

    Raises:
        KeyError: If a component is missing or unknown.
        ValueError: If a probability is invalid.

    Equation:
        [EQ-SYS-IMP-001], [EQ-SYS-IMP-002], [EQ-SYS-BDD-002].

    References:
        [REF-BIRNBAUM-1969], [REF-LAMBERT-1975], [REF-FUSSELL-1975],
        [REF-RAUSAND-HOYLAND-2004], [REF-RAUZY-1993].

    """
    p, q = structure.probabilities(reliabilities, unreliabilities)
    bdd = structure.bdd
    names = structure.components
    effects = bdd.variable_effects(structure.root, p, q, names)
    system_q = bdd.probability(structure.root, p, q, value=False)
    structural = structural_importance(structure)
    cut_family = structure.cut_family()
    failure_p = q  # P(y_i = 1) in the failure-space (dual) function
    failure_q = p
    result: dict[str, ImportanceMeasures] = {}
    for name in names:
        effect = effects[name]
        q_failed = effect.false_given_zero
        q_perfect = effect.false_given_one
        birnbaum = max(0.0, effect.birnbaum)
        containing = bdd.family_function(bdd.family_onset(cut_family, name))
        union = bdd.probability(containing, failure_p, failure_q)
        improvement = birnbaum * q[name]
        result[name] = ImportanceMeasures(
            component=name,
            reliability=p[name],
            birnbaum=birnbaum,
            criticality=_ratio(improvement, system_q),
            fussell_vesely=_ratio(union, system_q),
            risk_achievement_worth=_ratio(q_failed, system_q),
            risk_reduction_worth=_ratio(system_q, q_perfect),
            improvement_potential=improvement,
            structural=structural[name],
            unreliability_if_failed=q_failed,
            unreliability_if_perfect=q_perfect,
        )
    return MappingProxyType(result)
