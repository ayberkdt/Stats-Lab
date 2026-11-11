"""Multivariate probability distributions and vector-valued operations."""

from probstat.linalg import Matrix, Vector

from .conjugate import (
    DirichletPosterior,
    NormalInverseWishart,
    NormalInverseWishartPosterior,
    dirichlet_posterior,
    normal_inverse_wishart_posterior,
)
from .dirichlet import Dirichlet
from .lkj import LKJCholesky, LKJCorrelation, lkj_log_normalizer
from .multinomial import DirichletMultinomial, Multinomial
from .normal import MultivariateNormal
from .student_t import MultivariateStudentT
from .transforms import (
    DeltaMethodResult,
    TransformedDensity,
    delta_method,
    transformed_density,
)
from .wishart import InverseWishart, Wishart, multivariate_log_gamma

__all__ = [
    "DeltaMethodResult",
    "Dirichlet",
    "DirichletMultinomial",
    "DirichletPosterior",
    "InverseWishart",
    "LKJCholesky",
    "LKJCorrelation",
    "Matrix",
    "Multinomial",
    "MultivariateNormal",
    "MultivariateStudentT",
    "NormalInverseWishart",
    "NormalInverseWishartPosterior",
    "TransformedDensity",
    "Vector",
    "Wishart",
    "delta_method",
    "dirichlet_posterior",
    "lkj_log_normalizer",
    "multivariate_log_gamma",
    "normal_inverse_wishart_posterior",
    "transformed_density",
]
