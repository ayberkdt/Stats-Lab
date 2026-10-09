"""Explicit model-discrepancy records for calibration and prediction."""

from __future__ import annotations

from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real


@dataclass(frozen=True, slots=True)
class AdditiveGaussianDiscrepancy:
    """A zero-mean additive discrepancy term with declared applicability scope.

    This object records an assumption; it does not claim that discrepancy is
    identifiable separately from calibration parameters without additional
    experiments or prior structure.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """

    standard_deviation: float
    description: str
    applicability_domain: str

    def __post_init__(self) -> None:
        """Validate scale and require human-readable scientific scope.

        References:
            [EQ-CAL-DISC-001], [REF-KENNEDY-OHAGAN-2001].

        """
        standard_deviation = require_finite_real(
            self.standard_deviation, name="standard_deviation"
        )
        if standard_deviation < 0.0:
            raise ValueError("standard_deviation must be non-negative")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if (
            not isinstance(self.applicability_domain, str)
            or not self.applicability_domain.strip()
        ):
            raise ValueError("applicability_domain must be a non-empty string")
        object.__setattr__(self, "standard_deviation", standard_deviation)
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(
            self, "applicability_domain", self.applicability_domain.strip()
        )

    @property
    def variance(self) -> float:
        """Return the discrepancy variance used by additive Gaussian predictors.

        Equation:
            [EQ-CAL-DISC-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        return self.standard_deviation * self.standard_deviation
