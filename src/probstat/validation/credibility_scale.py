"""Structured credibility assessment of a model against published scales.

Space agencies ask that results of models and simulations used for decisions
be reported with a structured credibility assessment. A
:class:`~probstat.validation.CredibilityRecord` keeps verification, validation
and prediction *evidence* apart but scores nothing. This module adds the
scoring layer: a data-driven :class:`CredibilityScale` (named factors grouped
in categories, ordered levels per factor), an assessor's
:class:`CredibilityAssessment` made of :class:`FactorScore` items, a
:class:`TargetProfile` of required levels, the gap list and ordered
improvement list between the two, and the criticality matrix of NASA-STD-7009B
that decides when the standard applies.

Three published scales ship as constructors, each carrying its source key and
the sections it was transcribed from:

* :func:`nasa_std_7009b_scale` - the M&S capability assessment (5 factors) and
  M&S results assessment (6 factors) of NASA-STD-7009B (2024), levels 0-4;
* :func:`pcmm_scale` - the six elements of the Predictive Capability Maturity
  Model of Oberkampf, Pilch and Trucano (SAND2007-5948), levels 0-3;
* :func:`nasa_cas_2008_scale` - the eight-factor credibility assessment scale of
  the 2008 NASA M&S standard as summarised by Babula et al. (2009), levels
  0-4. It is the historical predecessor of the 7009B scale.

Level descriptors are short paraphrases with a section or table citation, not
the standards' wording; the documents are the authority. Any other scale can
be built from :class:`CredibilityFactor` and :class:`CredibilityLevel`
objects, so a scale that was not available to read here must be supplied by
its user.

Roll-ups follow the source of each scale and nothing else:

* NASA-STD-7009B states that the factors are largely independent and that
  comparing or combining factor assessments should be avoided; the scale has
  ``rollup="none"`` and :meth:`CredibilityAssessment.overall` and
  :meth:`CredibilityAssessment.category_minimum` refuse to run.
* PCMM recommends against aggregation but, when it cannot be avoided, asks for
  the triple (minimum, mean, maximum) over all scores, never the mean alone
  (``rollup="triple"``).
* The 2008 NASA scale defines the overall score as the minimum of the eight
  factor scores (``rollup="minimum"``).

No other average is ever computed. Levels are an ordinal scale: a mean of
levels is meaningful only where a source defines it, and the triple above is
the only place it appears.

Sufficiency is separate from assessment. Neither source prescribes which
level is enough; a :class:`TargetProfile` is supplied by the caller (for a
decision criticality chosen with :func:`criticality_assessment` or by the
programme) and :func:`level_gaps` compares it with the achieved levels. A
:class:`CredibilityAssessment` does not alter evidence and never turns a
record field into a level: :func:`suggest_record_evidence` only lists which
fields of a :class:`~probstat.validation.CredibilityRecord` bear on which
factor, for the assessor to weigh.

Assumptions: a level is an assessor's judgement against a written descriptor,
supported by listed evidence; a scale score is an assessment of the process
and evidence, not a measure of predictive accuracy.

References:
    [REF-NASA-STD-7009B-2024], [REF-OBERKAMPF-PILCH-TRUCANO-2007],
    [REF-BABULA-ETAL-2009], [REF-NASA-HDBK-7009B-2026].

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from probstat.foundations.validation import require_finite_real

from .credibility import CredibilityRecord

ROLLUP_RULES: Final = ("none", "minimum", "triple")
RECORD_EVIDENCE_KINDS: Final = ("verification", "validation", "prediction", "domain")
CRITICALITY_RATINGS: Final = ("R", "Y", "G")
_TENTH_TOLERANCE: Final = 1e-9


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    """Return ``value`` as text, rejecting non-strings and (optionally) blanks.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _texts(values: Iterable[object], name: str) -> tuple[str, ...]:
    """Return a tuple of non-empty strings.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not a string")
    return tuple(_text(item, f"{name}[{i}]") for i, item in enumerate(values))


# ---------------------------------------------------------------------------
# Scale definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CredibilityLevel:
    """One ordered level of a factor with its short descriptor.

    Args:
        value: Ordinal level index (0 is the lowest level).
        descriptor: One-line description of what achieving the level means.

    Raises:
        TypeError: If ``value`` is not an integer or ``descriptor`` is not
            text.
        ValueError: If ``value`` is negative or ``descriptor`` is blank.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E], [REF-OBERKAMPF-PILCH-TRUCANO-2007,
        Section 3.5].

    """

    value: int
    descriptor: str

    def __post_init__(self) -> None:
        """Validate the level.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("value must be an integer")
        if self.value < 0:
            raise ValueError("value must be non-negative")
        _text(self.descriptor, "descriptor")


@dataclass(frozen=True, slots=True)
class CredibilityFactor:
    """A named credibility factor with ordered levels.

    Args:
        name: CredibilityFactor name as the source states it.
        category: Category the factor is grouped in.
        levels: Levels with values ``0, 1, ..., n - 1`` in increasing order.
        question: The question the factor asks, in the assessor's words.
        subfactors: Optional sub-factor names (scored on the same levels in
            :attr:`FactorScore.subscores`).
        record_evidence: Kinds of :class:`~probstat.validation.CredibilityRecord`
            evidence that bear on this factor, from
            :data:`RECORD_EVIDENCE_KINDS`; used only by
            :func:`suggest_record_evidence`.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If levels are not consecutive from zero, or names repeat
            or are blank, or an evidence kind is unknown.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E], [REF-OBERKAMPF-PILCH-TRUCANO-2007].

    """

    name: str
    category: str
    levels: tuple[CredibilityLevel, ...]
    question: str = ""
    subfactors: tuple[str, ...] = ()
    record_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the factor definition.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        _text(self.name, "name")
        _text(self.category, "category")
        _text(self.question, "question", allow_empty=True)
        levels = tuple(self.levels)
        if len(levels) < 2:
            raise ValueError("a factor needs at least two levels")
        for index, level in enumerate(levels):
            if not isinstance(level, CredibilityLevel):
                raise TypeError(f"levels[{index}] must be a CredibilityLevel")
            if level.value != index:
                raise ValueError(
                    "level values must be 0, 1, 2, ... in increasing order"
                )
        subfactors = _texts(self.subfactors, "subfactors")
        if len(set(subfactors)) != len(subfactors):
            raise ValueError("subfactor names must be distinct")
        kinds = _texts(self.record_evidence, "record_evidence")
        for kind in kinds:
            if kind not in RECORD_EVIDENCE_KINDS:
                raise ValueError(f"unknown record evidence kind {kind!r}")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "subfactors", subfactors)
        object.__setattr__(self, "record_evidence", kinds)

    @property
    def max_level(self) -> int:
        """Return the highest level value of the factor.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return len(self.levels) - 1

    def descriptor(self, level: float) -> str:
        """Return the descriptor of the integer part of ``level``.

        Args:
            level: A level of the factor; a non-integer value is read as the
                level below it (the one fully achieved).

        Returns:
            The one-line descriptor.

        Raises:
            ValueError: If ``level`` lies outside ``[0, max_level]``.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        value = require_finite_real(level, name="level")
        if value < 0 or value > self.max_level:
            raise ValueError(f"level must lie in [0, {self.max_level}]")
        return self.levels[int(math.floor(value + _TENTH_TOLERANCE))].descriptor


@dataclass(frozen=True, slots=True)
class LevelCap:
    """A source-defined limit: ``limited`` factors may not exceed ``by`` + margin.

    The PCMM requires the levels of model validation and of uncertainty
    quantification to be no higher than two above the lower of the two
    verification elements (``margin = 2``).

    Args:
        limited: Factors whose level is capped.
        by: Factors whose minimum level sets the cap.
        margin: Allowed excess over that minimum.

    Raises:
        ValueError: If a name list is empty or ``margin`` is negative.

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5], [EQ-VAL-CRED-003].

    """

    limited: tuple[str, ...]
    by: tuple[str, ...]
    margin: int

    def __post_init__(self) -> None:
        """Validate the cap.

        References:
            [EQ-VAL-CRED-003]. No equation is evaluated.

        """
        limited = _texts(self.limited, "limited")
        by = _texts(self.by, "by")
        if not limited or not by:
            raise ValueError("limited and by must not be empty")
        if isinstance(self.margin, bool) or not isinstance(self.margin, int):
            raise TypeError("margin must be an integer")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        object.__setattr__(self, "limited", limited)
        object.__setattr__(self, "by", by)


@dataclass(frozen=True, slots=True)
class CredibilityScale:
    """A credibility scale: factors in categories with ordered levels.

    Args:
        key: Short identifier carrying the source, e.g. ``"NASA-STD-7009B"``.
        title: Human-readable title.
        source_key: Bibliography key of the source document.
        citation: Citation of the document the scale was read from.
        sections: Sections or tables the factors and levels were read from.
        factors: Factors in the order the source lists them.
        rollup: ``"none"`` (the source says not to combine factors),
            ``"minimum"`` (the source defines the overall score as the
            minimum) or ``"triple"`` (minimum, mean, maximum together).
        resolution: Spacing of permitted scores: ``1.0`` for whole levels
            only (no partial credit), ``0.1`` for scores in tenths (a partly
            achieved level, as PCMM permits), ``0.0`` for any real score in
            the level range (a weighted factor score).
        caps: Source-defined limits between factors.
        notes: Cautions from the source that apply to every use.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If factor names repeat, a cap names an unknown factor, or
            ``rollup`` is not recognised.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E], [REF-OBERKAMPF-PILCH-TRUCANO-2007],
        [REF-BABULA-ETAL-2009].

    """

    key: str
    title: str
    source_key: str
    citation: str
    sections: tuple[str, ...]
    factors: tuple[CredibilityFactor, ...]
    rollup: str = "none"
    resolution: float = 1.0
    caps: tuple[LevelCap, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze the scale.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        for name in ("key", "title", "source_key", "citation"):
            _text(getattr(self, name), name)
        _text(self.notes, "notes", allow_empty=True)
        sections = _texts(self.sections, "sections")
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("a scale needs at least one factor")
        for index, factor in enumerate(factors):
            if not isinstance(factor, CredibilityFactor):
                raise TypeError(f"factors[{index}] must be a CredibilityFactor")
        names = [factor.name for factor in factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be distinct")
        if self.rollup not in ROLLUP_RULES:
            raise ValueError(f"rollup must be one of {ROLLUP_RULES}")
        resolution = require_finite_real(self.resolution, name="resolution")
        if resolution < 0:
            raise ValueError("resolution must be non-negative")
        object.__setattr__(self, "resolution", resolution)
        caps = tuple(self.caps)
        for cap in caps:
            if not isinstance(cap, LevelCap):
                raise TypeError("caps must be LevelCap objects")
            for name in (*cap.limited, *cap.by):
                if name not in names:
                    raise ValueError(f"cap refers to unknown factor {name!r}")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "caps", caps)

    @property
    def factor_names(self) -> tuple[str, ...]:
        """Return the factor names in source order.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return tuple(factor.name for factor in self.factors)

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the categories in order of first appearance.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        seen: list[str] = []
        for factor in self.factors:
            if factor.category not in seen:
                seen.append(factor.category)
        return tuple(seen)

    def factor(self, name: str) -> CredibilityFactor:
        """Return the factor called ``name``.

        Args:
            name: Exact factor name.

        Returns:
            The factor.

        Raises:
            ValueError: If the scale has no such factor.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        for factor in self.factors:
            if factor.name == name:
                return factor
        raise ValueError(f"unknown factor {name!r} for scale {self.key!r}")

    def factors_in(self, category: str) -> tuple[CredibilityFactor, ...]:
        """Return the factors of one category.

        Args:
            category: Category name.

        Returns:
            The factors of the category in source order.

        Raises:
            ValueError: If the scale has no such category.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        if category not in self.categories:
            raise ValueError(f"unknown category {category!r} for scale {self.key!r}")
        return tuple(f for f in self.factors if f.category == category)

    def check_level(self, factor: str, level: float, *, name: str = "level") -> float:
        """Validate ``level`` for ``factor`` and return it as a float.

        Args:
            factor: CredibilityFactor name.
            level: Candidate level.
            name: Label used in error messages.

        Returns:
            The level.

        Raises:
            TypeError: If ``level`` is not a real number.
            ValueError: If the level is outside the factor's range or is not
                a multiple of the scale's ``resolution``.

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.1],
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.1].

        """
        value = require_finite_real(level, name=name)
        top = self.factor(factor).max_level
        if value < 0 or value > top:
            raise ValueError(f"{name} for {factor!r} must lie in [0, {top}]")
        step = self.resolution
        if step > 0 and abs(value / step - round(value / step)) > _TENTH_TOLERANCE:
            kind = (
                "a whole level: the scale has no partial credit"
                if step == 1.0
                else f"a multiple of {step:g}"
            )
            raise ValueError(f"{name} for {factor!r} must be {kind}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible description of the scale.

        Returns:
            Nested dictionary accepted by :meth:`from_dict`.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return {
            "key": self.key,
            "title": self.title,
            "source_key": self.source_key,
            "citation": self.citation,
            "sections": list(self.sections),
            "rollup": self.rollup,
            "resolution": self.resolution,
            "notes": self.notes,
            "caps": [
                {"limited": list(c.limited), "by": list(c.by), "margin": c.margin}
                for c in self.caps
            ],
            "factors": [
                {
                    "name": f.name,
                    "category": f.category,
                    "question": f.question,
                    "subfactors": list(f.subfactors),
                    "record_evidence": list(f.record_evidence),
                    "levels": [lv.descriptor for lv in f.levels],
                }
                for f in self.factors
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CredibilityScale:
        """Rebuild a scale from :meth:`to_dict` output.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The scale.

        Raises:
            ValueError: If a required key is missing or a value is invalid.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        try:
            factors = tuple(
                CredibilityFactor(
                    name=f["name"],
                    category=f["category"],
                    levels=tuple(
                        CredibilityLevel(i, d) for i, d in enumerate(f["levels"])
                    ),
                    question=f.get("question", ""),
                    subfactors=tuple(f.get("subfactors", ())),
                    record_evidence=tuple(f.get("record_evidence", ())),
                )
                for f in data["factors"]
            )
            caps = tuple(
                LevelCap(tuple(c["limited"]), tuple(c["by"]), c["margin"])
                for c in data.get("caps", ())
            )
            return cls(
                key=data["key"],
                title=data["title"],
                source_key=data["source_key"],
                citation=data["citation"],
                sections=tuple(data["sections"]),
                factors=factors,
                rollup=data.get("rollup", "none"),
                resolution=data.get("resolution", 1.0),
                caps=caps,
                notes=data.get("notes", ""),
            )
        except KeyError as error:
            raise ValueError(f"scale dictionary lacks key {error.args[0]!r}") from error


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorScore:
    """One assessor's score of one factor, with its evidence and rationale.

    Args:
        factor: CredibilityFactor name.
        level: Achieved level, or ``None`` when the factor is explicitly not
            assessed (a rationale is then required). Level 0, "insufficient
            evidence", is an assessment; ``None`` is the absence of one.
        evidence: Evidence items supporting the level (document names,
            report identifiers, record references).
        rationale: Why the evidence meets the level's descriptor.
        assessor: Free text, e.g. the person or team.
        date: Free text date of the assessment.
        subscores: Optional levels of the factor's sub-factors.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``level`` is negative or non-finite, or a factor is
            marked not assessed without a rationale.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.6], [REF-NASA-STD-7009B-2024,
        Appendix E.1].

    """

    factor: str
    level: float | None
    evidence: tuple[str, ...] = ()
    rationale: str = ""
    assessor: str = ""
    date: str = ""
    subscores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the score and freeze its collections.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        _text(self.factor, "factor")
        _text(self.rationale, "rationale", allow_empty=True)
        _text(self.assessor, "assessor", allow_empty=True)
        _text(self.date, "date", allow_empty=True)
        if self.level is not None:
            level = require_finite_real(self.level, name="level")
            if level < 0:
                raise ValueError("level must be non-negative")
            object.__setattr__(self, "level", level)
        elif not self.rationale.strip():
            raise ValueError(
                f"factor {self.factor!r} is marked not assessed: give the reason "
                "in rationale"
            )
        object.__setattr__(self, "evidence", _texts(self.evidence, "evidence"))
        subscores = {
            _text(k, "subscore name"): require_finite_real(v, name=f"subscores[{k!r}]")
            for k, v in dict(self.subscores).items()
        }
        object.__setattr__(self, "subscores", MappingProxyType(subscores))


@dataclass(frozen=True, slots=True)
class OverallStatement:
    """The roll-up a scale's source defines, with the factors that set it.

    Args:
        rule: ``"minimum"`` or ``"triple"``.
        minimum: Lowest assessed level.
        mean: Arithmetic mean of the assessed levels (``None`` for
            ``"minimum"``; the mean is reported only inside the triple).
        maximum: Highest assessed level (``None`` for ``"minimum"``).
        weakest: Factors at the minimum.
        note: How the source frames the statement.

    References:
        [REF-BABULA-ETAL-2009], [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2],
        [EQ-VAL-CRED-001], [EQ-VAL-CRED-002].

    """

    rule: str
    minimum: float
    mean: float | None
    maximum: float | None
    weakest: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class CredibilityAssessment:
    """Scores of every factor of a scale, each given or marked not assessed.

    Args:
        scale: The scale the scores refer to.
        scores: Exactly one :class:`FactorScore` per factor of the scale.
        subject: What is assessed (model, version, quantity of interest).
        intended_use: The decision or use the assessment is for.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a factor is unknown, repeated, or missing, a level is
            not a level of the scale, a sub-factor is unknown, or a
            source-defined cap between factors is violated.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.2.1.8], [REF-NASA-STD-7009B-2024,
        Section 4.3.6.1], [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5].

    """

    scale: CredibilityScale
    scores: tuple[FactorScore, ...]
    subject: str = ""
    intended_use: str = ""

    def __post_init__(self) -> None:
        """Validate coverage, levels, sub-factors, and caps.

        References:
            [EQ-VAL-CRED-003]. No equation is evaluated.

        """
        if not isinstance(self.scale, CredibilityScale):
            raise TypeError("scale must be a CredibilityScale")
        _text(self.subject, "subject", allow_empty=True)
        _text(self.intended_use, "intended_use", allow_empty=True)
        scores = tuple(self.scores)
        seen: set[str] = set()
        for index, score in enumerate(scores):
            if not isinstance(score, FactorScore):
                raise TypeError(f"scores[{index}] must be a FactorScore")
            if score.factor not in self.scale.factor_names:
                raise ValueError(
                    f"unknown factor {score.factor!r} for scale {self.scale.key!r}"
                )
            if score.factor in seen:
                raise ValueError(f"factor {score.factor!r} is scored twice")
            seen.add(score.factor)
            factor = self.scale.factor(score.factor)
            if score.level is not None:
                self.scale.check_level(score.factor, score.level)
            for sub, value in score.subscores.items():
                if sub not in factor.subfactors:
                    raise ValueError(
                        f"unknown subfactor {sub!r} of factor {score.factor!r}"
                    )
                self.scale.check_level(score.factor, value, name=f"subscore {sub!r}")
        missing = [n for n in self.scale.factor_names if n not in seen]
        if missing:
            raise ValueError(
                "every factor must be scored or marked not assessed; missing: "
                + ", ".join(missing)
            )
        ordered = {score.factor: score for score in scores}
        object.__setattr__(
            self, "scores", tuple(ordered[name] for name in self.scale.factor_names)
        )
        self._check_caps()

    def _check_caps(self) -> None:
        """Raise if an assessed level exceeds a source-defined cap.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5], [EQ-VAL-CRED-003].

        """
        for cap in self.scale.caps:
            driving = [self.level(name) for name in cap.by]
            if any(value is None for value in driving):
                continue
            limit = min(v for v in driving if v is not None) + cap.margin
            for name in cap.limited:
                value = self.level(name)
                if value is not None and value > limit + _TENTH_TOLERANCE:
                    raise ValueError(
                        f"level {value:g} of {name!r} exceeds the limit {limit:g} "
                        f"set by {', '.join(cap.by)} (source rule: at most the "
                        f"lowest of them plus {cap.margin})"
                    )

    def score(self, factor: str) -> FactorScore:
        """Return the score object of ``factor``.

        Args:
            factor: CredibilityFactor name.

        Returns:
            The :class:`FactorScore`.

        Raises:
            ValueError: If the scale has no such factor.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        self.scale.factor(factor)
        for item in self.scores:
            if item.factor == factor:
                return item
        raise AssertionError("unreachable: coverage was validated")  # pragma: no cover

    def level(self, factor: str) -> float | None:
        """Return the achieved level of ``factor`` (``None`` if not assessed).

        Args:
            factor: CredibilityFactor name.

        Returns:
            The level or ``None``.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return self.score(factor).level

    @property
    def unassessed(self) -> tuple[str, ...]:
        """Return the factors marked not assessed.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return tuple(s.factor for s in self.scores if s.level is None)

    @property
    def is_complete(self) -> bool:
        """Return whether every factor has a level.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return not self.unassessed

    def category_minimum(self, *, allow_partial: bool = False) -> dict[str, float]:
        """Return the lowest assessed level in each category.

        The minimum is the weakest-link statement and the only order
        statistic that needs no cardinal meaning of levels. It is refused for
        scales whose source says factors must not be combined.

        Args:
            allow_partial: Skip not-assessed factors instead of raising; a
                category with no assessed factor is then omitted.

        Returns:
            Mapping from category to lowest level.

        Raises:
            ValueError: If the scale has ``rollup="none"``, or a factor is
                not assessed and ``allow_partial`` is false.

        Equation:
            [EQ-VAL-CRED-001].

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.2], [REF-BABULA-ETAL-2009].

        """
        self._require_rollup()
        if not allow_partial and not self.is_complete:
            raise ValueError("not assessed: " + ", ".join(self.unassessed))
        result: dict[str, float] = {}
        for category in self.scale.categories:
            values = [
                v
                for f in self.scale.factors_in(category)
                if (v := self.level(f.name)) is not None
            ]
            if values:
                result[category] = min(values)
        return result

    def overall(self, *, allow_partial: bool = False) -> OverallStatement:
        """Return the overall statement that the scale's source defines.

        Args:
            allow_partial: Use only the assessed factors instead of raising
                when some are not assessed.

        Returns:
            For ``rollup="minimum"``, the minimum and the factors at it. For
            ``rollup="triple"``, the minimum, mean and maximum together.

        Raises:
            ValueError: If the scale has ``rollup="none"`` (the standard says
                factor assessments should not be combined), a factor is not
                assessed and ``allow_partial`` is false, or nothing is
                assessed.

        Equation:
            [EQ-VAL-CRED-001], [EQ-VAL-CRED-002].

        References:
            [REF-BABULA-ETAL-2009], [REF-OBERKAMPF-PILCH-TRUCANO-2007,
            Section 5.2].

        """
        self._require_rollup()
        if not allow_partial and not self.is_complete:
            raise ValueError("not assessed: " + ", ".join(self.unassessed))
        pairs = [(s.factor, s.level) for s in self.scores if s.level is not None]
        if not pairs:
            raise ValueError("no factor is assessed")
        values = [level for _, level in pairs if level is not None]
        low = min(values)
        weakest = tuple(name for name, level in pairs if level == low)
        if self.scale.rollup == "minimum":
            return OverallStatement(
                "minimum",
                low,
                None,
                None,
                weakest,
                "overall score is the minimum of the factor scores (weakest link)",
            )
        triple = pcmm_aggregate_triple(values)
        return OverallStatement(
            "triple",
            triple[0],
            triple[1],
            triple[2],
            weakest,
            "aggregation is not recommended; when needed, report the minimum, "
            "mean and maximum together",
        )

    def _require_rollup(self) -> None:
        """Raise if the scale's source forbids combining factors.

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.2]. No equation is evaluated.

        """
        if self.scale.rollup == "none":
            raise ValueError(
                f"scale {self.scale.key!r} does not define a roll-up: its source "
                "states that factor assessments are largely independent and "
                "should not be combined; report each factor level"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible archive of the assessment.

        Returns:
            Dictionary (with the scale embedded) accepted by
            :meth:`from_dict`.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return {
            "scale": self.scale.to_dict(),
            "subject": self.subject,
            "intended_use": self.intended_use,
            "scores": [
                {
                    "factor": s.factor,
                    "level": s.level,
                    "evidence": list(s.evidence),
                    "rationale": s.rationale,
                    "assessor": s.assessor,
                    "date": s.date,
                    "subscores": dict(s.subscores),
                }
                for s in self.scores
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CredibilityAssessment:
        """Rebuild an assessment from :meth:`to_dict` output.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The assessment (validated again).

        Raises:
            ValueError: If a required key is missing or a value is invalid.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        try:
            scale = CredibilityScale.from_dict(data["scale"])
            scores = tuple(
                FactorScore(
                    factor=s["factor"],
                    level=s["level"],
                    evidence=tuple(s.get("evidence", ())),
                    rationale=s.get("rationale", ""),
                    assessor=s.get("assessor", ""),
                    date=s.get("date", ""),
                    subscores=dict(s.get("subscores", {})),
                )
                for s in data["scores"]
            )
            return cls(
                scale,
                scores,
                data.get("subject", ""),
                data.get("intended_use", ""),
            )
        except KeyError as error:
            raise ValueError(
                f"assessment dictionary lacks key {error.args[0]!r}"
            ) from error


def pcmm_aggregate_triple(scores: Sequence[float]) -> tuple[float, float, float]:
    """Return the PCMM aggregation triple (minimum, mean, maximum).

    PCMM scores are ordinal and the six elements are "apples and oranges", so
    the source recommends against aggregating at all. When pressure to
    condense is unavoidable it asks that the three numbers always be reported
    together so that the worst score stays visible.

    Args:
        scores: Scores to aggregate (all elements of all subsystems being
            combined), at least one.

    Returns:
        ``(min, mean, max)``.

    Raises:
        ValueError: If ``scores`` is empty or contains a non-finite value.

    Equation:
        [EQ-VAL-CRED-002].

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2].

    """
    values = [require_finite_real(s, name="score") for s in scores]
    if not values:
        raise ValueError("scores must not be empty")
    return (min(values), math.fsum(values) / len(values), max(values))


def weighted_evidence_review_level(
    evidence: float, review: float, review_weight: float
) -> float:
    """Return the 2008 NASA factor level from its evidence and review subfactors.

    In the development and operations categories of the 2008 scale a factor
    score is ``(1 - w) * evidence + w * review`` where the technical-review
    weight ``w`` lies in ``[0, 0.3]`` so that review can move a score by at
    most 30 % of the evidence-to-review gap.

    Args:
        evidence: CredibilityLevel of the evidence subfactor.
        review: CredibilityLevel of the technical-review subfactor.
        review_weight: Weight ``w`` of the review subfactor in ``[0, 0.3]``.

    Returns:
        The factor level.

    Raises:
        ValueError: If a level is negative or ``review_weight`` is outside
            ``[0, 0.3]``.

    Equation:
        [EQ-VAL-CRED-004].

    References:
        [REF-BABULA-ETAL-2009, Roll-up Processes].

    """
    e = require_finite_real(evidence, name="evidence")
    r = require_finite_real(review, name="review")
    w = require_finite_real(review_weight, name="review_weight")
    if e < 0 or r < 0:
        raise ValueError("levels must be non-negative")
    if not 0.0 <= w <= 0.3:
        raise ValueError("review_weight must lie in [0, 0.3]")
    return (1.0 - w) * e + w * r


# ---------------------------------------------------------------------------
# Sufficiency: target profile, gaps, improvement order
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Required level per factor for a stated decision.

    Neither shipped source says which level is enough; the profile comes from
    the programme, the technical authority, or the analyst, and its basis is
    recorded with it.

    Args:
        name: Profile name, e.g. ``"flight-critical decision"``.
        required: Required level for each factor it constrains; factors not
            listed have no requirement.
        criticality: Decision criticality the profile is for (free text, for
            example the rating from :func:`criticality_assessment`).
        basis: Where the thresholds come from.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a required level is negative or not finite, or
            ``required`` is empty.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.6], [REF-OBERKAMPF-PILCH-TRUCANO-2007,
        Section 5.1].

    """

    name: str
    required: Mapping[str, float]
    criticality: str = ""
    basis: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze the profile.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        _text(self.name, "name")
        _text(self.criticality, "criticality", allow_empty=True)
        _text(self.basis, "basis", allow_empty=True)
        required = {}
        for factor, level in dict(self.required).items():
            value = require_finite_real(level, name=f"required[{factor!r}]")
            if value < 0:
                raise ValueError("required levels must be non-negative")
            required[_text(factor, "factor")] = value
        if not required:
            raise ValueError("required must name at least one factor")
        object.__setattr__(self, "required", MappingProxyType(required))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible archive of the profile.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        return {
            "name": self.name,
            "required": dict(self.required),
            "criticality": self.criticality,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetProfile:
        """Rebuild a profile from :meth:`to_dict` output.

        Raises:
            ValueError: If a required key is missing or a value is invalid.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        try:
            return cls(
                data["name"],
                dict(data["required"]),
                data.get("criticality", ""),
                data.get("basis", ""),
            )
        except KeyError as error:
            raise ValueError(
                f"profile dictionary lacks key {error.args[0]!r}"
            ) from error


def uniform_profile(
    scale: CredibilityScale, level: int, *, name: str, basis: str = ""
) -> TargetProfile:
    """Return a profile that requires the same level of every factor.

    The level is clipped to each factor's own maximum, so one number works on
    scales whose factors have different numbers of levels.

    Args:
        scale: Scale whose factors are constrained.
        level: Required level.
        name: Profile name.
        basis: Where the threshold comes from.

    Returns:
        The profile.

    Raises:
        ValueError: If ``level`` is negative.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    value = require_finite_real(level, name="level")
    if value < 0:
        raise ValueError("level must be non-negative")
    return TargetProfile(
        name,
        {f.name: min(value, float(f.max_level)) for f in scale.factors},
        basis=basis,
    )


@dataclass(frozen=True, slots=True)
class LevelGap:
    """A factor whose achieved level is below the required level.

    Args:
        factor: CredibilityFactor name.
        category: Its category.
        achieved: Achieved level, ``None`` if the factor is not assessed.
        required: Required level.
        shortfall: ``required - achieved``, ``None`` if not assessed.
        next_level: Descriptor of the level above the achieved one (the
            level to reach next), ``None`` if not assessed.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.8.4], [EQ-VAL-CRED-005].

    """

    factor: str
    category: str
    achieved: float | None
    required: float
    shortfall: float | None
    next_level: str | None


@dataclass(frozen=True, slots=True)
class ImprovementStep:
    """One entry of the ordered improvement list.

    Args:
        rank: Position in the list, starting at 1.
        factor: CredibilityFactor to improve.
        from_level: Achieved level, ``None`` if it must first be assessed.
        to_level: Required level.
        shortfall: Gap, ``None`` if not assessed.
        action: Descriptor of the next level to reach, or a request to
            assess the factor.
        prerequisites: Other factors that a source rule requires to be raised
            first (a cap on this factor).

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.1], [EQ-VAL-CRED-005].

    """

    rank: int
    factor: str
    from_level: float | None
    to_level: float
    shortfall: float | None
    action: str
    prerequisites: tuple[str, ...]


def _validated_requirements(
    assessment: CredibilityAssessment, profile: TargetProfile
) -> dict[str, float]:
    """Check a profile against the assessment's scale and return its levels.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if not isinstance(profile, TargetProfile):
        raise TypeError("profile must be a TargetProfile")
    return {
        factor: assessment.scale.check_level(
            factor, level, name=f"required[{factor!r}]"
        )
        for factor, level in profile.required.items()
    }


def level_gaps(
    assessment: CredibilityAssessment, profile: TargetProfile
) -> tuple[LevelGap, ...]:
    """Return the factors whose achieved level is below the required level.

    A required level of zero never produces a gap. A factor that the profile
    constrains but the assessment marks not assessed is a gap with unknown
    shortfall: absence of an assessment is not sufficiency.

    Args:
        assessment: Achieved levels.
        profile: Required levels.

    Returns:
        Gaps in scale order (empty if the profile is met).

    Raises:
        TypeError: If ``profile`` has the wrong type.
        ValueError: If the profile names an unknown factor or a level the
            factor does not have.

    Equation:
        [EQ-VAL-CRED-005].

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.8.4], [REF-NASA-STD-7009B-2024,
        Appendix E.5].

    """
    required = _validated_requirements(assessment, profile)
    gaps: list[LevelGap] = []
    for factor in assessment.scale.factors:
        if factor.name not in required or required[factor.name] <= 0:
            continue
        need = required[factor.name]
        got = assessment.level(factor.name)
        if got is None:
            gaps.append(LevelGap(factor.name, factor.category, None, need, None, None))
        elif got < need - _TENTH_TOLERANCE:
            nxt = min(factor.max_level, int(math.floor(got + _TENTH_TOLERANCE)) + 1)
            gaps.append(
                LevelGap(
                    factor.name,
                    factor.category,
                    got,
                    need,
                    need - got,
                    factor.levels[nxt].descriptor,
                )
            )
    return tuple(gaps)


def is_sufficient(assessment: CredibilityAssessment, profile: TargetProfile) -> bool:
    """Return whether every constrained factor meets its required level.

    Args:
        assessment: Achieved levels.
        profile: Required levels.

    Returns:
        ``True`` if :func:`level_gaps` is empty.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E.5]. Equation: [EQ-VAL-CRED-005].

    """
    return not level_gaps(assessment, profile)


def improvement_plan(
    assessment: CredibilityAssessment, profile: TargetProfile
) -> tuple[ImprovementStep, ...]:
    """Return the gaps as an ordered improvement list.

    Order (a convenience of this library, not defined by a source): factors
    not yet assessed first, then by decreasing shortfall, ties in scale
    order. Where a source rule caps a factor by other factors (PCMM), the
    capping factors still below what is needed are listed as prerequisites.

    Args:
        assessment: Achieved levels.
        profile: Required levels.

    Returns:
        Steps ranked from 1.

    Raises:
        TypeError: If ``profile`` has the wrong type.
        ValueError: If the profile names an unknown factor or level.

    Equation:
        [EQ-VAL-CRED-005].

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.1].

    """
    gaps = level_gaps(assessment, profile)
    order = {name: i for i, name in enumerate(assessment.scale.factor_names)}
    ranked = sorted(
        gaps,
        key=lambda g: (
            0 if g.shortfall is None else 1,
            0.0 if g.shortfall is None else -g.shortfall,
            order[g.factor],
        ),
    )
    steps: list[ImprovementStep] = []
    for rank, gap in enumerate(ranked, start=1):
        prerequisites: list[str] = []
        for cap in assessment.scale.caps:
            if gap.factor not in cap.limited:
                continue
            needed = gap.required - cap.margin
            for name in cap.by:
                level = assessment.level(name)
                if (level is None or level < needed) and name not in prerequisites:
                    prerequisites.append(name)
        action = (
            "assess this factor first (no level recorded)"
            if gap.next_level is None
            else gap.next_level
        )
        steps.append(
            ImprovementStep(
                rank,
                gap.factor,
                gap.achieved,
                gap.required,
                gap.shortfall,
                action,
                tuple(prerequisites),
            )
        )
    return tuple(steps)


# ---------------------------------------------------------------------------
# Criticality matrix of NASA-STD-7009B, Appendix D
# ---------------------------------------------------------------------------

CONSEQUENCE_LABELS: Final = {
    1: "I: Negligible",
    2: "II: Minor",
    3: "III: Moderate",
    4: "IV: Significant",
    5: "V: Catastrophic",
}
INFLUENCE_LABELS: Final = {
    1: "1: Negligible",
    2: "2: Minor",
    3: "3: Moderate",
    4: "4: Significant",
    5: "5: Controlling",
}
# Figure 1 of NASA-STD-7009B, Appendix D.1: rows are M&S influence 5..1, columns
# are decision consequence I..V.
_CRITICALITY_ROWS: Final = {
    5: "GYRRR",
    4: "GYYRR",
    3: "GYYYR",
    2: "GGGYY",
    1: "GGGGY",
}
_CRITICALITY_MEANING: Final = {
    "R": (
        "clear candidate for completely adhering to the requirements of the standard"
    ),
    "Y": (
        "adherence at the discretion of programme/project management with the "
        "delegated technical authority"
    ),
    "G": "no critical driving force for adherence",
}


@dataclass(frozen=True, slots=True)
class CriticalityAssessment:
    """Applicability of NASA-STD-7009B from decision consequence and influence.

    Args:
        consequence: Decision consequence level, 1 (I: negligible) to 5
            (V: catastrophic).
        influence: M&S influence level, 1 (negligible) to 5 (controlling).
        rating: ``"R"``, ``"Y"`` or ``"G"``.
        meaning: What the rating implies for applying the standard.

    References:
        [REF-NASA-STD-7009B-2024, Appendix D], [EQ-VAL-CRED-006].

    """

    consequence: int
    influence: int
    rating: str
    meaning: str

    @property
    def consequence_label(self) -> str:
        """Return the consequence level name.

        References:
            [REF-NASA-STD-7009B-2024, Appendix D.2]. No equation is evaluated.

        """
        return CONSEQUENCE_LABELS[self.consequence]

    @property
    def influence_label(self) -> str:
        """Return the influence level name.

        References:
            [REF-NASA-STD-7009B-2024, Appendix D.3]. No equation is evaluated.

        """
        return INFLUENCE_LABELS[self.influence]


def criticality_assessment(consequence: int, influence: int) -> CriticalityAssessment:
    """Look up the M&S criticality rating in the matrix of NASA-STD-7009B.

    The matrix combines the consequences of a wrong decision with the degree
    to which the model results influence it. The standard's quantitative
    breakpoints between consequence levels are set per programme and are not
    encoded here; the caller supplies the level numbers. The standard also
    allows programmes to adjust the matrix, so a lookup is a default, not a
    determination.

    Args:
        consequence: Decision consequence level in ``1..5`` (I to V).
        influence: M&S influence level in ``1..5`` (negligible to
            controlling).

    Returns:
        The rating and its meaning.

    Raises:
        TypeError: If a level is not an integer.
        ValueError: If a level is outside ``1..5``.

    Equation:
        [EQ-VAL-CRED-006].

    References:
        [REF-NASA-STD-7009B-2024, Appendix D, Figure 1].

    """
    for name, value in (("consequence", consequence), ("influence", influence)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= value <= 5:
            raise ValueError(f"{name} must lie in 1..5")
    rating = _CRITICALITY_ROWS[influence][consequence - 1]
    return CriticalityAssessment(
        consequence, influence, rating, _CRITICALITY_MEANING[rating]
    )


def criticality_matrix() -> dict[tuple[int, int], str]:
    """Return the whole matrix as ``{(consequence, influence): rating}``.

    Returns:
        Twenty-five ratings.

    Equation:
        [EQ-VAL-CRED-006].

    References:
        [REF-NASA-STD-7009B-2024, Appendix D, Figure 1].

    """
    return {
        (c, i): _CRITICALITY_ROWS[i][c - 1] for i in range(1, 6) for c in range(1, 6)
    }


# ---------------------------------------------------------------------------
# Bridge from CredibilityRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceSuggestion:
    """Evidence of a record that bears on one factor, for the assessor.

    Args:
        factor: CredibilityFactor name.
        kind: Record evidence kind (``"verification"``, ``"validation"``,
            ``"prediction"`` or ``"domain"``).
        items: One-line descriptions of the matching record entries (empty
            if the record holds none of that kind).
        note: Reminder that this is a pointer, not a score.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.6], [REF-ROY-OBERKAMPF-2011].

    """

    factor: str
    kind: str
    items: tuple[str, ...]
    note: str


def suggest_record_evidence(
    record: CredibilityRecord, scale: CredibilityScale
) -> tuple[EvidenceSuggestion, ...]:
    """List which fields of a credibility record bear on which factor.

    The mapping is declared by each factor's ``record_evidence`` and covers
    only what the record holds: numerical-error budgets (verification),
    validation assessments, prediction statements (uncertainty), and the
    domain checks of those predictions. Pedigree, reviews, process
    management and sensitivity analysis are not in a record and get no
    suggestion. A suggestion never sets a level.

    Args:
        record: Existing credibility record.
        scale: Scale whose factors are matched.

    Returns:
        One suggestion per (factor, evidence kind) pair, in scale order.

    Raises:
        TypeError: If an argument has the wrong type.

    References:
        [REF-PROBSTAT-VALIDATION], [REF-ROY-OBERKAMPF-2011]. No mathematical
        equation is evaluated.

    """
    if not isinstance(record, CredibilityRecord):
        raise TypeError("record must be a CredibilityRecord")
    if not isinstance(scale, CredibilityScale):
        raise TypeError("scale must be a CredibilityScale")
    pools: dict[str, tuple[str, ...]] = {
        "verification": tuple(
            f"{v.quantity}: numerical error bound {v.budget.total_bound:.4g}"
            + (
                ""
                if v.observed_order is None
                else f", observed order {v.observed_order:.3g}"
            )
            + f"; {v.description}"
            for v in record.verification
        ),
        "validation": tuple(f"{a.quantity}: {a.verdict}" for a in record.validation),
        "prediction": tuple(
            f"{p.quantity}: {100 * p.probability:g}% interval "
            f"[{p.lower:.4g}, {p.upper:.4g}]; sources: {p.uncertainty_sources}"
            for p in record.predictions
        ),
        "domain": tuple(
            f"{p.quantity}: "
            + ("domain not checked" if p.domain_check is None else p.domain_check.note)
            for p in record.predictions
        ),
    }
    note = "suggestion for the assessor; it does not set a level"
    return tuple(
        EvidenceSuggestion(factor.name, kind, pools[kind], note)
        for factor in scale.factors
        for kind in factor.record_evidence
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(text: str) -> str:
    """Escape a table cell for Markdown.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    return text.replace("|", "\\|").replace("\n", " ")


def _level_text(level: float | None) -> str:
    """Format a level or the not-assessed marker.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    return "not assessed" if level is None else f"{level:g}"


def _table(rows: list[list[str]], *, markdown: bool) -> str:
    """Render rows (first row is the header) as Markdown or aligned text.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if markdown:
        head, *body = rows
        lines = [
            "| " + " | ".join(_cell(c) for c in head) + " |",
            "| " + " | ".join("---" for _ in head) + " |",
        ]
        lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in body)
        return "\n".join(lines)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render_assessment(
    assessment: CredibilityAssessment,
    profile: TargetProfile | None = None,
    *,
    markdown: bool = False,
) -> str:
    """Render the factor table of an assessment as text or Markdown.

    One row per factor with its category, achieved level (and, if a profile
    is given, the required level), and the descriptor of the achieved level.
    The roll-up the scale's source defines (if any) follows the table; for a
    scale without one, a line says that factor levels are reported
    separately.

    Args:
        assessment: Assessment to render.
        profile: Optional required levels shown beside the achieved ones.
        markdown: Produce a Markdown pipe table instead of aligned text.

    Returns:
        The rendered text.

    Raises:
        TypeError: If ``profile`` has the wrong type.
        ValueError: If the profile does not fit the scale.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E.5].

    """
    required = {} if profile is None else _validated_requirements(assessment, profile)
    header = ["Category", "Factor", "Level"]
    if profile is not None:
        header.append("Required")
    header.append("Evidence at the achieved level")
    rows = [header]
    for factor in assessment.scale.factors:
        level = assessment.level(factor.name)
        row = [factor.category, factor.name, _level_text(level)]
        if profile is not None:
            row.append(f"{required[factor.name]:g}" if factor.name in required else "-")
        if level is None:
            row.append("-")
        else:
            text = factor.descriptor(level)
            if abs(level - round(level)) > _TENTH_TOLERANCE:
                text += f" (partly toward level {int(math.floor(level)) + 1})"
            row.append(text)
        rows.append(row)
    title = f"{assessment.scale.title} [{assessment.scale.source_key}]"
    out = [("## " if markdown else "") + title]
    if assessment.subject:
        out.append(f"Subject: {assessment.subject}")
    if assessment.intended_use:
        out.append(f"Intended use: {assessment.intended_use}")
    out.append("")
    out.append(_table(rows, markdown=markdown))
    out.append("")
    if assessment.scale.rollup == "none":
        out.append(
            "Factor levels are reported separately; the source does not define a "
            "combined score."
        )
    else:
        overall = assessment.overall(allow_partial=True)
        suffix = "" if assessment.is_complete else " (assessed factors only)"
        if overall.rule == "minimum":
            out.append(
                f"Overall (minimum): {overall.minimum:g}, set by "
                f"{', '.join(overall.weakest)}{suffix}."
            )
        else:
            out.append(
                f"Aggregate (minimum, mean, maximum): ({overall.minimum:g}, "
                f"{overall.mean:.3g}, {overall.maximum:g}){suffix}."
            )
    return "\n".join(out)


def render_gaps(gaps: Sequence[LevelGap], *, markdown: bool = False) -> str:
    """Render a gap list as text or Markdown.

    Args:
        gaps: Result of :func:`level_gaps`.
        markdown: Produce a Markdown pipe table instead of aligned text.

    Returns:
        The rendered table, or a one-line statement when there is no gap.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.8.4].

    """
    if not gaps:
        return "No gap: every constrained factor meets its required level."
    rows = [["Factor", "Achieved", "Required", "Shortfall", "Next level to reach"]]
    for gap in gaps:
        rows.append(
            [
                gap.factor,
                _level_text(gap.achieved),
                f"{gap.required:g}",
                "unknown" if gap.shortfall is None else f"{gap.shortfall:g}",
                gap.next_level or "assess the factor",
            ]
        )
    return _table(rows, markdown=markdown)


def render_improvement_plan(
    steps: Sequence[ImprovementStep], *, markdown: bool = False
) -> str:
    """Render an improvement list as text or Markdown.

    Args:
        steps: Result of :func:`improvement_plan`.
        markdown: Produce a Markdown pipe table instead of aligned text.

    Returns:
        The rendered table, or a one-line statement when nothing is needed.

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.1].

    """
    if not steps:
        return "Nothing to improve: every constrained factor meets its required level."
    rows = [["Rank", "Factor", "From", "To", "Do", "After raising first"]]
    for step in steps:
        rows.append(
            [
                str(step.rank),
                step.factor,
                _level_text(step.from_level),
                f"{step.to_level:g}",
                step.action,
                ", ".join(step.prerequisites) or "-",
            ]
        )
    return _table(rows, markdown=markdown)


# ---------------------------------------------------------------------------
# Shipped scales
# ---------------------------------------------------------------------------


def _levels(*descriptors: str) -> tuple[CredibilityLevel, ...]:
    """Build levels ``0, 1, ...`` from descriptors in increasing order.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    return tuple(CredibilityLevel(i, d) for i, d in enumerate(descriptors))


_NO_EVIDENCE: Final = "Insufficient evidence."
_STD_SECTIONS: Final = (
    "4.2.1.8 [M&S 48] (capability factors)",
    "4.3.6.1 [M&S 31] (results factors)",
    "4.3.8.4-4.3.8.5 [M&S 50], [M&S 35] (reporting)",
    "Appendix E.1-E.2 (assessment principles)",
    "E.3.1-E.3.6 with Tables 7-11 (capability levels)",
    "E.4.1-E.4.7 with Tables 15-20 (results levels)",
)


def nasa_std_7009b_scale() -> CredibilityScale:
    """Return the credibility scale of NASA-STD-7009B (approved 2024-03-05).

    Eleven factors in two categories: the M&S capability assessment (five
    factors of the development phase, E.3) and the M&S results assessment
    (six factors of the use phase, E.4), each on levels 0-4 with
    "insufficient evidence" at level 0. Level descriptors are paraphrases of
    Tables 7-11 and 15-20.

    The standard's own cautions are carried by the scale: each level is a
    discrete step with no partial credit; where several conditions are
    stated all must hold; the factors are largely independent and comparing
    or combining them should be avoided (``rollup="none"``); the standard
    requires the levels and the gaps to the thresholds to be reported but
    sets no threshold itself. The factor set is described as a minimum; the
    standard warns that altering the level definitions likely adds M&S risk.
    Table 19 gives the scope of a level-4 use/analysis review as "all other
    capability-related factors" where the level-3 row speaks of
    results-related factors; the level-4 descriptor below quotes
    Table 19's "capability-related factors" wording and says so in the
    descriptor itself.

    Returns:
        The scale with ``source_key="REF-NASA-STD-7009B-2024"``.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E], [REF-NASA-HDBK-7009B-2026,
        Appendix D.2].

    """
    cap = "M&S capability assessment (development phase)"
    res = "M&S results assessment (use phase)"
    pedigree = (
        _NO_EVIDENCE,
        "Some data traceable to informal documentation; sources of the significant "
        "data known; uncertainties may be unestimated.",
        "Most data traceable to formal documentation; processes for the significant "
        "data known; uncertainties of all data at least estimated.",
        "All data known and traceable to a sufficiently representative referent; "
        "significant data acceptable in accuracy, precision, uncertainty.",
        "All data known and fully traceable to the real-world system; all data "
        "acceptable in accuracy, precision, uncertainty.",
    )
    management = (
        _NO_EVIDENCE,
        "Roles, process and product requirements informally documented; "
        "configuration management informal.",
        "Process and product requirements formally documented and approved; formal "
        "configuration management.",
        "Formal process rigorously controlled; compliance and product measurements "
        "documented; configuration management rigorous.",
        "Measurements, including user feedback, used to improve both process and "
        "products.",
    )
    factors = (
        CredibilityFactor(
            "M&S Data Pedigree",
            cap,
            _levels(*pedigree),
            "Is the pedigree and quality of the data used to develop the M&S adequate?",
        ),
        CredibilityFactor(
            "M&S Verification",
            cap,
            _levels(
                _NO_EVIDENCE,
                "Informal verification with some evidence of completeness or success.",
                "Documented practices and error-assessment methods; most important "
                "errors meet requirements.",
                "Rigorous practices on the end-to-end model; all important errors "
                "meet requirements.",
                "Reliable practices and error estimation on the end-to-end model; "
                "all errors meet requirements.",
            ),
            "Were the M&S implemented correctly, per their specifications?",
            record_evidence=("verification",),
        ),
        CredibilityFactor(
            "M&S Validation",
            cap,
            _levels(
                _NO_EVIDENCE,
                "Conceptually validated; intended use clearly stated (required "
                "before any higher level).",
                "Favourable comparison with a representative system or higher-"
                "fidelity M&S; validation points inside the operating domain; some "
                "important responses.",
                "Favourable comparison in a representative environment; validation "
                "points significantly span the operating domain; all important "
                "responses.",
                "Favourable comparison with the real system in its operating "
                "environment; points completely span the domain; all responses.",
            ),
            "Did the M&S results compare favourably with the referent, and how close "
            "is the referent to the real-world system?",
            record_evidence=("validation",),
        ),
        CredibilityFactor(
            "M&S Development Technical Review",
            cap,
            _levels(
                _NO_EVIDENCE,
                "Favourable informal internal peer review covering at least data "
                "pedigree, verification and validation.",
                "Favourable formal internal peer review covering at least data "
                "pedigree, verification and validation.",
                "Favourable independent peer review covering all other capability "
                "factors.",
                "Favourable independent peer review plus independent factor "
                "evaluation (outcomes independently reproduced).",
            ),
            "Were technical reviews done, of what scope, how formal and independent?",
        ),
        CredibilityFactor(
            "M&S Development Process/Product Management",
            cap,
            _levels(*management),
            "How well managed were the M&S development processes and products?",
        ),
        CredibilityFactor(
            "M&S Use Assessment",
            res,
            _levels(
                _NO_EVIDENCE,
                "Type, domain and purpose match the proposed use and the M&S can "
                "give the outputs needed; first or substantially different use.",
                "Proposed use inside the M&S limits; moderate departure from prior "
                "uses.",
                "Proposed use inside the verification domain; minor departure from "
                "prior uses.",
                "Proposed use inside the validation domain; nearly identical to "
                "prior uses.",
            ),
            "How well does the proposed use match the permissible uses and past "
            "successful uses?",
            record_evidence=("domain", "validation"),
        ),
        CredibilityFactor(
            "M&S Input Pedigree",
            res,
            _levels(*pedigree),
            "Is the pedigree and quality of the run-time input data adequate?",
        ),
        CredibilityFactor(
            "M&S Uncertainty Characterization",
            res,
            _levels(
                _NO_EVIDENCE,
                "Input uncertainty sources identified with qualitative estimates; "
                "impact on outputs not addressed.",
                "Most sources identified, quantified and classified; propagation "
                "only by reduced-order or reduced-dimension analysis.",
                "Quantitative uncertainty estimates for most outputs after "
                "propagating all known sources.",
                "Statistical analysis of output uncertainty for all outputs after "
                "rigorous, validated propagation of all known sources.",
            ),
            "Is the uncertainty in the current results characterised and propagated?",
            record_evidence=("prediction", "verification"),
        ),
        CredibilityFactor(
            "M&S Results Robustness",
            res,
            _levels(
                _NO_EVIDENCE,
                "Sensitivity estimated by analogy with similar problems.",
                "Sensitivity quantified for a few parameters (under 20 per cent, as "
                "a guideline); combinations unknown.",
                "Quantified for many parameters (20 to 50 per cent) including many "
                "key ones; some combinations.",
                "Quantified for most parameters (over 50 per cent) including most "
                "key ones; many combinations.",
            ),
            "How thoroughly are the sensitivities of the current results known?",
        ),
        CredibilityFactor(
            "M&S Use/Analysis Technical Review",
            res,
            _levels(
                _NO_EVIDENCE,
                "Favourable informal internal peer review covering at least input "
                "pedigree, uncertainty characterisation and robustness.",
                "Favourable formal internal peer review covering at least input "
                "pedigree, uncertainty characterisation and robustness.",
                "Favourable independent peer review covering all other results "
                "factors.",
                "Independent peer review plus independent factor evaluation "
                "(Table 19 words the scope as capability-related factors).",
            ),
            "Were technical reviews of the use and results done, and how independent?",
        ),
        CredibilityFactor(
            "M&S Use Process/Product Management",
            res,
            _levels(*management),
            "How well managed were the M&S use processes and products?",
        ),
    )
    return CredibilityScale(
        key="NASA-STD-7009B",
        title="NASA-STD-7009B credibility assessment (capability and results)",
        source_key="REF-NASA-STD-7009B-2024",
        citation=(
            "NASA-STD-7009B, Standard for Models and Simulations, approved 2024-03-05"
        ),
        sections=_STD_SECTIONS,
        factors=factors,
        rollup="none",
        resolution=1.0,
        notes=(
            "Discrete step function, no partial credit; factors are largely "
            "independent and should not be compared or combined; the standard sets "
            "no threshold levels."
        ),
    )


def pcmm_scale() -> CredibilityScale:
    """Return the Predictive Capability Maturity Model scale (SAND2007-5948).

    Six elements scored 0-3 (Table 1 of the report; element descriptors from
    Table 3 and Sections 4.2.1-4.2.6). Requirements accumulate: a level needs
    the descriptors of the levels below it. Scores in tenths are allowed for
    a partly achieved level (Section 4.1, for example 1.5). The report
    requires model validation and uncertainty quantification to be no more
    than two levels above the lower of code verification and solution
    verification (Section 4.2.5); the scale carries that cap and an
    assessment that violates it is rejected.

    The score is the maturity of the process elements, not the accuracy of
    the predictions. The report recommends against aggregating the six
    scores; when it is done anyway it asks for the minimum, mean and maximum
    together (``rollup="triple"``). It deliberately excludes adequacy for the
    intended use from the scale, so the required levels come from the user.

    Returns:
        The scale with ``source_key="REF-OBERKAMPF-PILCH-TRUCANO-2007"``.

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Table 1], [REF-OBERKAMPF-PILCH-TRUCANO-2007,
        Table 3], [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2].

    """
    cat = "PCMM elements"
    factors = (
        CredibilityFactor(
            "Representation and Geometric Fidelity",
            cat,
            _levels(
                "Judgment only; little or no representational or geometric fidelity.",
                "Significant simplification of system and boundary conditions; "
                "major components defined.",
                "Limited simplification; major and some minor components well defined.",
                "Essentially no simplification; as-built detail of all components.",
            ),
            "What features are neglected because of simplifications or stylizations?",
        ),
        CredibilityFactor(
            "Physics and Material Model Fidelity",
            cat,
            _levels(
                "Judgment only; models unknown or fully empirical; no coupling.",
                "Some physics-based models calibrated on related-system data; "
                "minimal or ad hoc coupling.",
                "All important processes physics-based; calibration with separate- "
                "and integral-effects tests; one-way coupling; some peer review.",
                "All models physics-based with minimal calibration; full two-way "
                "coupling; independent peer review.",
            ),
            "How fundamental are the physics and material models, and how calibrated?",
        ),
        CredibilityFactor(
            "Code Verification",
            cat,
            _levels(
                "Judgment only; minimal software testing; little or no software "
                "quality procedures.",
                "Code under software quality procedures; unit and regression tests; "
                "some benchmark comparisons.",
                "Some algorithms tested for observed order of convergence; some "
                "features tested against benchmarks; some peer review.",
                "All important algorithms tested for observed order; all important "
                "features against rigorous benchmarks; independent peer review.",
            ),
            "Are algorithm deficiencies, software errors and poor practices "
            "corrupting the results?",
            record_evidence=("verification",),
        ),
        CredibilityFactor(
            "Solution Verification",
            cat,
            _levels(
                "Judgment only; numerical errors of unknown or large effect.",
                "Numerical effects on relevant outputs estimated qualitatively; "
                "input/output verified only by the analysts.",
                "Numerical effects estimated quantitatively small on some outputs; "
                "input/output independently verified; some peer review.",
                "Numerical effects small on all important outputs; important "
                "simulations independently reproduced; independent peer review.",
            ),
            "Are numerical solution errors and human procedural errors corrupting "
            "the results?",
            record_evidence=("verification",),
        ),
        CredibilityFactor(
            "Model Validation",
            cat,
            _levels(
                "Judgment only; few comparisons with measurements of similar systems.",
                "Quantitative accuracy assessment, but of outputs or conditions not "
                "directly relevant to the application; experimental uncertainty "
                "large or unknown.",
                "Quantitative predictive accuracy for some key outputs from "
                "separate- and integral-effects tests; experimental uncertainty "
                "characterised for most separate-effects tests.",
                "Predictive accuracy for all important outputs at directly "
                "relevant conditions; experimental uncertainty characterised for "
                "all tests; independent peer review.",
            ),
            "How carefully is the accuracy of simulation and experiment assessed "
            "across the validation hierarchy?",
            record_evidence=("validation",),
        ),
        CredibilityFactor(
            "Uncertainty Quantification and Sensitivity Analysis",
            cat,
            _levels(
                "Only deterministic analyses; uncertainty and sensitivity not "
                "addressed.",
                "Aleatory and epistemic uncertainty propagated without distinction; "
                "informal sensitivity studies; many strong assumptions.",
                "Aleatory and epistemic uncertainty segregated and propagated; "
                "quantitative sensitivity for most parameters; propagation errors "
                "estimated.",
                "Comprehensive treatment and sensitivity analysis of parameters and "
                "models; propagation errors shown small; independent peer review.",
            ),
            "How thoroughly are uncertainties and sensitivities characterised and "
            "propagated?",
            record_evidence=("prediction",),
        ),
    )
    return CredibilityScale(
        key="PCMM",
        title="Predictive Capability Maturity Model",
        source_key="REF-OBERKAMPF-PILCH-TRUCANO-2007",
        citation=(
            "W. L. Oberkampf, M. Pilch, T. G. Trucano, Predictive Capability "
            "Maturity Model for Computational Modeling and Simulation, "
            "SAND2007-5948, Sandia National Laboratories, October 2007"
        ),
        sections=(
            "Table 1 (elements and levels)",
            "Table 3 (descriptors)",
            "4.1 (fractional scores)",
            "4.2.1-4.2.6 (element discussion)",
            "4.2.5 (cap on validation and UQ by verification)",
            "5.1 (project maturity requirements)",
            "5.2 (aggregation)",
        ),
        factors=factors,
        rollup="triple",
        resolution=0.1,
        caps=(
            LevelCap(
                (
                    "Model Validation",
                    "Uncertainty Quantification and Sensitivity Analysis",
                ),
                ("Code Verification", "Solution Verification"),
                2,
            ),
        ),
        notes=(
            "Scores measure the maturity of process elements, not predictive "
            "accuracy; adequacy for the intended use is assessed separately."
        ),
    )


def nasa_cas_2008_scale() -> CredibilityScale:
    """Return the eight-factor credibility scale of the 2008 NASA M&S standard.

    Transcribed from the secondary account of Babula et al. (2009), Section
    IV and Table 1, not from the 2008 standard itself, which was not read.
    Eight factors in three categories (M&S Development, M&S Operations,
    Supporting Evidence), levels 0-4. The five development and operations
    factors have Evidence and Technical Review sub-factors, combined with
    :func:`weighted_evidence_review_level`. The overall score is the minimum
    of the eight factor scores (``rollup="minimum"``); the paper reports that
    an earlier draft used a weighted average and that the minimum replaced
    it. The 2008 text is superseded by NASA-STD-7009B; use
    :func:`nasa_std_7009b_scale` for current work.

    Returns:
        The scale with ``source_key="REF-BABULA-ETAL-2009"``.

    References:
        [REF-BABULA-ETAL-2009, Section IV and Table 1].

    """
    dev = "M&S Development"
    ops = "M&S Operations"
    sup = "Supporting Evidence"
    sub = ("Evidence", "Technical Review")
    factors = (
        CredibilityFactor(
            "Verification",
            dev,
            _levels(
                _NO_EVIDENCE,
                "Conceptual and mathematical models verified.",
                "Unit and regression testing of key features.",
                "Formal numerical error estimation.",
                "Numerical errors small for all important features.",
            ),
            "Were the models implemented correctly, and what is the numerical error?",
            sub,
            ("verification",),
        ),
        CredibilityFactor(
            "Validation",
            dev,
            _levels(
                _NO_EVIDENCE,
                "Conceptual and mathematical models agree with simple referents.",
                "Results agree with experimental data or other M&S on unit problems.",
                "Results agree with experimental data for problems of interest.",
                "Results agree with real-world data.",
            ),
            "How well did the M&S results and the referent data compare?",
            sub,
            ("validation",),
        ),
        CredibilityFactor(
            "Input Pedigree",
            ops,
            _levels(
                _NO_EVIDENCE,
                "Input data traceable to informal documentation.",
                "Input data traceable to formal documentation.",
                "Input data agree with experimental data for problems of interest.",
                "Input data agree with real-world data.",
            ),
            "How confident are we of the current input data?",
            sub,
        ),
        CredibilityFactor(
            "Results Uncertainty",
            ops,
            _levels(
                _NO_EVIDENCE,
                "Qualitative estimates.",
                "Deterministic analysis or expert opinion.",
                "Non-deterministic analysis.",
                "Non-deterministic and numerical analysis.",
            ),
            "What is the uncertainty in the current M&S results?",
            sub,
            ("prediction", "verification"),
        ),
        CredibilityFactor(
            "Results Robustness",
            ops,
            _levels(
                _NO_EVIDENCE,
                "Qualitative estimates.",
                "Sensitivity known for a few parameters.",
                "Sensitivity known for many parameters.",
                "Sensitivity known for most parameters; key sensitivities identified.",
            ),
            "How thoroughly are the sensitivities of the current results known?",
            sub,
        ),
        CredibilityFactor(
            "Use History",
            sup,
            _levels(
                _NO_EVIDENCE,
                "Passes simple tests.",
                "Used before for critical decisions.",
                "Previous predictions were later validated by mission data.",
                "De facto standard.",
            ),
            "Have the current M&S been used successfully before?",
            record_evidence=("domain",),
        ),
        CredibilityFactor(
            "M&S Management",
            sup,
            _levels(
                _NO_EVIDENCE,
                "Managed process.",
                "Established process.",
                "Predictable process.",
                "Continual process improvement.",
            ),
            "How well managed were the M&S processes?",
        ),
        CredibilityFactor(
            "People Qualifications",
            sup,
            _levels(
                _NO_EVIDENCE,
                "Engineering or science degree.",
                "Formal M&S training and experience, and recommended-practice "
                "training.",
                "Advanced degree or extensive M&S experience, and recommended-"
                "practice knowledge.",
                "Extensive experience in and use of recommended practices for this "
                "particular M&S.",
            ),
            "How qualified were the personnel?",
        ),
    )
    return CredibilityScale(
        key="NASA-CAS-2008",
        title="NASA credibility assessment scale of the 2008 M&S standard",
        source_key="REF-BABULA-ETAL-2009",
        citation=(
            "M. Babula, W. J. Bertch, L. L. Green, J. P. Hale, G. E. Mosier, "
            "M. J. Steele, J. Woods, NASA Standard for Models and Simulations: "
            "Credibility Assessment Scale, NTRS 20090005963"
        ),
        sections=(
            "IV (overview: eight factors, three categories)",
            "Table 1 (key aspects of the levels)",
            "Roll-up Processes (minimum; weights)",
        ),
        factors=factors,
        rollup="minimum",
        resolution=0.0,
        notes="Historical scale; secondary source; superseded by NASA-STD-7009B.",
    )
