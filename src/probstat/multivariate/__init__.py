"""Multivariate probability distributions and vector-valued operations."""

from probstat.linalg import Matrix, Vector

from .normal import MultivariateNormal

__all__ = ["Matrix", "MultivariateNormal", "Vector"]
