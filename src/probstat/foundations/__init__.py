"""Dependency-free validation and numerical foundation utilities."""

from .validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)

__all__ = [
    "coerce_finite_sample",
    "require_finite_real",
    "require_non_negative_finite_real",
    "require_probability",
]

