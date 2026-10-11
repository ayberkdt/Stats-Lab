"""Multivariate probability distributions and vector-valued operations."""

from probstat.linalg import Matrix, Vector

from .normal import MultivariateNormal
from .transforms import (
    DeltaMethodResult,
    TransformedDensity,
    delta_method,
    transformed_density,
)

__all__ = [
    "DeltaMethodResult",
    "Matrix",
    "MultivariateNormal",
    "TransformedDensity",
    "Vector",
    "delta_method",
    "transformed_density",
]
