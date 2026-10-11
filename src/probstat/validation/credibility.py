"""Keep verification, validation, and prediction evidence separate.

A :class:`CredibilityRecord` holds three kinds of evidence that answer
different questions and must not be merged into one number:

* **verification** — how large is the numerical error of the solver
  (:class:`VerificationEvidence`, built on
  :class:`~probstat.uq.NumericalErrorBudget`);
* **validation** — how well do model predictions agree with experimental data
  inside the validation domain (:class:`~probstat.validation.assessment.
  ValidationAssessment`);
* **prediction** — what uncertainty is assigned to a QoI at intended-use
  conditions, and whether those conditions lie inside the validation domain
  (:class:`PredictionStatement`).

:meth:`CredibilityRecord.summary` reports each section and its gaps; it never
computes a single "confidence" score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
)
from probstat.statistics import quantile
from probstat.uq import NumericalErrorBudget

from .assessment import ValidationAssessment
from .metrics import budget_standard_uncertainty
from .records import DomainCheck, ValidationDomain


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Numerical-error evidence for one solver configuration and QoI.

    Args:
        quantity: QoI the budget applies to.
        budget: Absolute numerical error bounds.
        description: How the budget was obtained (grid study, tolerance
            study, Richardson extrapolation, ...).
        observed_order: Optional observed order of accuracy from a refinement
            study.

    Raises:
        TypeError: If ``budget`` has the wrong type.
        ValueError: If text is empty or the order is non-finite.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 6-8], [REF-ROACHE-1998, Chapters 4-5].

    """

    quantity: str
    budget: NumericalErrorBudget
    description: str
    observed_order: float | None = None

    def __post_init__(self) -> None:
        """Validate the verification record.

        References:
            [EQ-VV-003]. No equation is evaluated.

        """
        if not isinstance(self.budget, NumericalErrorBudget):
            raise TypeError("budget must be a NumericalErrorBudget")
        for name in ("quantity", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.observed_order is not None:
            object.__setattr__(
                self,
                "observed_order",
                require_finite_real(self.observed_order, name="observed_order"),
            )

    @property
    def standard_uncertainty(self) -> float:
        """Return ``u_num = bound / sqrt(3)`` (rectangular Type B assumption).

        Equation:
            [EQ-VAL-ASME-002].

        References:
            [REF-JCGM-100-2008, Section 4.3.7].

        """
        return budget_standard_uncertainty(self.budget)


@dataclass(frozen=True, slots=True)
class PredictionStatement:
    """A prediction-uncertainty statement at one intended-use condition.

    Args:
        quantity: QoI name.
        condition: Intended-use conditions.
        lower: Lower interval endpoint.
        upper: Upper interval endpoint.
        probability: Central probability of the interval in ``(0, 1)``.
        uncertainty_sources: What the interval includes (e.g. "input and
            parameter uncertainty; excludes model-form error").
        domain_check: Location relative to the validation domain, if known.

    Raises:
        ValueError: If the interval is reversed, the probability is invalid,
            or the source text is empty.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

    """

    quantity: str
    condition: Mapping[str, float]
    lower: float
    upper: float
    probability: float
    uncertainty_sources: str
    domain_check: DomainCheck | None = None

    def __post_init__(self) -> None:
        """Validate the prediction statement.

        References:
            [EQ-STAT-QUANT-001]. No equation is evaluated.

        """
        lower = require_finite_real(self.lower, name="lower")
        upper = require_finite_real(self.upper, name="upper")
        if upper < lower:
            raise ValueError("upper must not be below lower")
        level = require_finite_real(self.probability, name="probability")
        if not 0.0 < level < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        for name in ("quantity", "uncertainty_sources"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        condition = {
            str(k): require_finite_real(v, name=f"condition[{k!r}]")
            for k, v in dict(self.condition).items()
        }
        object.__setattr__(self, "condition", MappingProxyType(condition))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "probability", level)

    @classmethod
    def from_samples(
        cls,
        quantity: str,
        condition: Mapping[str, float],
        samples: Sequence[float],
        *,
        probability: float = 0.95,
        uncertainty_sources: str,
        domain: ValidationDomain | None = None,
        distance_threshold: float = 0.25,
    ) -> PredictionStatement:
        """Build an equal-tailed sample interval and optional domain check.

        Args:
            quantity: QoI name.
            condition: Intended-use conditions.
            samples: Predictive samples at the condition.
            probability: Central probability in ``(0, 1)``.
            uncertainty_sources: What the samples include.
            domain: Validation domain to check the condition against.
            distance_threshold: Scaled distance flag for the domain check.

        Returns:
            Prediction statement.

        Raises:
            ValueError: If the samples or probability are invalid.

        Equation:
            [EQ-STAT-QUANT-001], [EQ-VAL-DOM-001].

        References:
            [REF-HYNDMAN-FAN-1996], [REF-ROY-OBERKAMPF-2011].

        """
        values = sorted(coerce_finite_sample(samples, name="samples"))
        level = require_finite_real(probability, name="probability")
        if not 0.0 < level < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        check = (
            None
            if domain is None
            else domain.check(condition, distance_threshold=distance_threshold)
        )
        return cls(
            quantity,
            condition,
            quantile(values, 0.5 * (1.0 - level)),
            quantile(values, 0.5 * (1.0 + level)),
            level,
            uncertainty_sources,
            check,
        )


@dataclass(frozen=True, slots=True)
class CredibilityRecord:
    """Separate verification, validation, and prediction evidence.

    Args:
        verification: Numerical-error evidence (possibly empty).
        validation: Validation assessments (possibly empty).
        predictions: Prediction statements (possibly empty).

    Raises:
        TypeError: If an item has the wrong type.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Chapters 1-2 and Part IV].

    """

    verification: tuple[VerificationEvidence, ...] = ()
    validation: tuple[ValidationAssessment, ...] = ()
    predictions: tuple[PredictionStatement, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the three evidence collections.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        for name, kind in (
            ("verification", VerificationEvidence),
            ("validation", ValidationAssessment),
            ("predictions", PredictionStatement),
        ):
            items = tuple(getattr(self, name))
            for index, item in enumerate(items):
                if not isinstance(item, kind):
                    raise TypeError(f"{name}[{index}] must be a {kind.__name__}")
            object.__setattr__(self, name, items)

    @property
    def gaps(self) -> tuple[str, ...]:
        """Return missing or weak pieces of evidence as explicit statements.

        References:
            [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        gaps: list[str] = []
        if not self.verification:
            gaps.append("no numerical-error (verification) evidence recorded")
        if not self.validation:
            gaps.append("no validation evidence recorded")
        if not self.predictions:
            gaps.append("no intended-use prediction statement recorded")
        verified = {item.quantity for item in self.verification}
        validated = {item.quantity for item in self.validation}
        for prediction in self.predictions:
            if prediction.quantity not in verified:
                gaps.append(
                    f"prediction of {prediction.quantity!r} has no verification record"
                )
            if prediction.quantity not in validated:
                gaps.append(
                    f"prediction of {prediction.quantity!r} has no validation record"
                )
            check = prediction.domain_check
            if check is None:
                gaps.append(
                    f"prediction of {prediction.quantity!r} was not checked "
                    "against a validation domain"
                )
            elif check.extrapolation:
                gaps.append(
                    f"prediction of {prediction.quantity!r} is an extrapolation: "
                    f"{check.note}"
                )
        for assessment in self.validation:
            if not assessment.consistent:
                gaps.append(
                    f"validation of {assessment.quantity!r} found inconsistency "
                    "with data at the stated tolerance"
                )
        return tuple(gaps)

    def summary(self) -> str:
        """Return a three-section report plus explicit gaps.

        Returns:
            Multi-line text; no aggregate credibility score is computed.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        lines = ["Verification (are the equations solved correctly?):"]
        if not self.verification:
            lines.append("  - none recorded")
        for item in self.verification:
            order = (
                ""
                if item.observed_order is None
                else f", observed order {item.observed_order:.3g}"
            )
            lines.append(
                f"  - {item.quantity}: numerical error bound "
                f"{item.budget.total_bound:.4g} (u_num={item.standard_uncertainty:.4g}"
                f"{order}); {item.description}"
            )
        lines.append("Validation (do predictions agree with data, where tested?):")
        if not self.validation:
            lines.append("  - none recorded")
        for assessment in self.validation:
            lines.append(f"  - {assessment.verdict}")
        lines.append("Prediction (what uncertainty is assigned at intended use?):")
        if not self.predictions:
            lines.append("  - none recorded")
        for prediction in self.predictions:
            values = ", ".join(f"{k}={v:.6g}" for k, v in prediction.condition.items())
            where = (
                "domain not checked"
                if prediction.domain_check is None
                else prediction.domain_check.note
            )
            lines.append(
                f"  - {prediction.quantity} at ({values}): "
                f"{100 * prediction.probability:g}% interval "
                f"[{prediction.lower:.4g}, {prediction.upper:.4g}] "
                f"({prediction.uncertainty_sources}); {where}"
            )
        lines.append("Gaps:")
        gaps = self.gaps
        if gaps:
            lines.extend(f"  - {gap}" for gap in gaps)
        else:
            lines.append("  - none identified")
        lines.append(
            "These sections are reported separately on purpose; no single "
            "credibility score is computed."
        )
        return "\n".join(lines)
