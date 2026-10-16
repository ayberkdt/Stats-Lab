"""MIL-STD-882E risk assessment matrix built on :class:`RiskMatrix`.

MIL-STD-882E (System Safety, 2012) assesses a hazard by a *severity
category* (Table I: 1 Catastrophic, 2 Critical, 3 Marginal, 4 Negligible)
and a *probability level* (Table II: A Frequent, B Probable, C Occasional,
D Remote, E Improbable, F Eliminated). Table III maps each risk assessment
code (RAC, e.g. ``"1C"``) to a risk level:

==============  ============  ========  ========  ==========
Probability     1 Catastr.    2 Crit.   3 Marg.   4 Neglig.
==============  ============  ========  ========  ==========
A Frequent      High          High      Serious   Medium
B Probable      High          High      Serious   Medium
C Occasional    High          Serious   Medium    Low
D Remote        Serious       Medium    Medium    Low
E Improbable    Medium        Medium    Medium    Low
F Eliminated    Eliminated    (all severities)
==============  ============  ========  ========  ==========

This default is reproduced from Table III of the standard; any other table
(a programme-specific tailoring) can be supplied instead. The quantitative
definitions of the probability levels and the severity thresholds in
Tables I and II are *not* encoded: they are tailorable, and a programme must
state its own. The matrix inherits every limitation of risk matrices
(Cox 2008); it is a communication and acceptance-authority tool, not a
substitute for quantitative risk measures.

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .decision import RiskMatrix

SEVERITY_CATEGORIES = MappingProxyType(
    {1: "Catastrophic", 2: "Critical", 3: "Marginal", 4: "Negligible"}
)
PROBABILITY_LEVELS = MappingProxyType(
    {
        "A": "Frequent",
        "B": "Probable",
        "C": "Occasional",
        "D": "Remote",
        "E": "Improbable",
        "F": "Eliminated",
    }
)
RISK_LEVELS = ("High", "Serious", "Medium", "Low", "Eliminated")

MIL_STD_882E_TABLE: Mapping[str, tuple[str, str, str, str]] = MappingProxyType(
    {
        "A": ("High", "High", "Serious", "Medium"),
        "B": ("High", "High", "Serious", "Medium"),
        "C": ("High", "Serious", "Medium", "Low"),
        "D": ("Serious", "Medium", "Medium", "Low"),
        "E": ("Medium", "Medium", "Medium", "Low"),
        "F": ("Eliminated", "Eliminated", "Eliminated", "Eliminated"),
    }
)

# Likelihood score: F = 0 ... A = 5.
# Severity score: 4 Negligible = 0 ... 1 Catastrophic = 3.
_LEVEL_ORDER = ("F", "E", "D", "C", "B", "A")


def _likelihood_score(level: str) -> float:
    """Return the numeric likelihood score of a probability level letter.

    Args:
        level: One of ``"A"``-``"F"``.

    Returns:
        ``0`` for F up to ``5`` for A.

    Raises:
        ValueError: If the letter is unknown.

    References:
        [REF-MIL-STD-882E-2012]. No equation is evaluated.

    """
    if level not in PROBABILITY_LEVELS:
        raise ValueError("probability level must be one of A, B, C, D, E, F")
    return float(_LEVEL_ORDER.index(level))


def _severity_score(category: int) -> float:
    """Return the numeric severity score of a severity category.

    Args:
        category: 1 (Catastrophic) to 4 (Negligible).

    Returns:
        ``3`` for category 1 down to ``0`` for category 4.

    Raises:
        ValueError: If the category is unknown.

    References:
        [REF-MIL-STD-882E-2012]. No equation is evaluated.

    """
    if isinstance(category, bool) or category not in SEVERITY_CATEGORIES:
        raise ValueError("severity category must be 1, 2, 3, or 4")
    return float(4 - category)


def mil_std_882e_risk_matrix(
    table: Mapping[str, Sequence[str]] | None = None,
) -> RiskMatrix:
    """Return a :class:`RiskMatrix` encoding a MIL-STD-882E style RAC table.

    Rows are probability levels (likelihood scores F = 0 to A = 5) and
    columns severity categories (scores 4 Negligible = 0 to 1 Catastrophic =
    3). Use :func:`hazard_risk_level` to classify by letter and category.

    Args:
        table: Probability level letter to the four risk levels for severity
            categories 1, 2, 3, 4 (in that order). ``None`` uses
            :data:`MIL_STD_882E_TABLE` (Table III of the standard).

    Returns:
        Risk matrix with integer-score bins.

    Raises:
        ValueError: If a level is missing or a row does not have four labels.

    Equation:
        [EQ-RISK-MATRIX-001].

    References:
        [REF-MIL-STD-882E-2012], [REF-COX-2008].

    """
    source = MIL_STD_882E_TABLE if table is None else table
    if set(source) != set(PROBABILITY_LEVELS):
        raise ValueError("table must define every probability level A-F")
    categories = []
    for level in _LEVEL_ORDER:
        row = tuple(source[level])
        if len(row) != 4:
            raise ValueError("each table row needs four labels (severity 1-4)")
        categories.append(tuple(reversed(row)))
    return RiskMatrix((0.5, 1.5, 2.5, 3.5, 4.5), (0.5, 1.5, 2.5), tuple(categories))


def risk_assessment_code(severity_category: int, probability_level: str) -> str:
    """Return the RAC string, for example ``"1C"``.

    Args:
        severity_category: 1 to 4.
        probability_level: ``"A"`` to ``"F"``.

    Returns:
        Severity digit followed by the level letter.

    References:
        [REF-MIL-STD-882E-2012]. No equation is evaluated.

    """
    _severity_score(severity_category)
    _likelihood_score(probability_level)
    return f"{severity_category}{probability_level}"


def hazard_risk_level(
    matrix: RiskMatrix, severity_category: int, probability_level: str
) -> str:
    """Return the risk level of a hazard from a matrix built by the factory.

    Args:
        matrix: Matrix from :func:`mil_std_882e_risk_matrix`.
        severity_category: 1 (Catastrophic) to 4 (Negligible).
        probability_level: ``"A"`` (Frequent) to ``"F"`` (Eliminated).

    Returns:
        Risk level label.

    Equation:
        [EQ-RISK-MATRIX-001].

    References:
        [REF-MIL-STD-882E-2012], [REF-COX-2008].

    """
    return matrix.classify(
        _likelihood_score(probability_level), _severity_score(severity_category)
    )
