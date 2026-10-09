"""Dependency-free validation and numerical foundation utilities."""

from .protocols import (
    ContinuousProbabilityLaw,
    CumulativeDistribution,
    DiscreteProbabilityLaw,
    HasMeanAndVariance,
)
from .validation import (
    coerce_finite_sample,
    require_finite_complex,
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)

__all__ = [
    "ContinuousProbabilityLaw",
    "CumulativeDistribution",
    "DiscreteProbabilityLaw",
    "HasMeanAndVariance",
    "coerce_finite_sample",
    "require_finite_complex",
    "require_finite_real",
    "require_non_negative_finite_real",
    "require_probability",
]

