"""Multivariate probability distributions and vector-valued operations.

Vector- and matrix-valued laws (normal, Student t, Dirichlet, multinomial,
Wishart, LKJ), their conjugate updates, and the delta method.

Main entry points:
    ``MultivariateNormal``: non-degenerate multivariate normal law.
    ``MultivariateStudentT``: heavy-tailed law with a scale matrix and
        degrees of freedom.
    ``Dirichlet``: law on the probability simplex.
    ``Wishart``: law of positive-definite matrices (scale ``S``, ``nu``).
    ``InverseWishart``: law of ``Sigma`` with ``Sigma^-1`` Wishart.
    ``dirichlet_posterior``: Dirichlet-categorical conjugate update.
    ``normal_inverse_wishart_posterior``: conjugate update for an unknown
        mean and covariance.
    ``delta_method``: mean and covariance of a smooth transformation by
        linearization.

Assumptions:
    Scale and covariance matrices are symmetric positive definite.
    Conjugate updates assume independent observations from the stated
    likelihood. ``delta_method`` is exact for affine maps and otherwise
    ignores curvature, so it needs ``g`` to be close to linear over the
    spread of the input.

Common pitfalls:
    - The ``scale_matrix`` of ``MultivariateStudentT`` and the ``S`` of
      ``Wishart`` are scales, not covariances: the t covariance is
      ``nu/(nu - 2) Sigma`` and the Wishart mean is ``nu S``.
    - Undefined moments are returned with ``inf``/``nan`` entries rather
      than raised: the t covariance for ``nu <= 2``, the inverse-Wishart
      mean for ``nu <= p + 1``, and the Dirichlet mode when some
      ``alpha_i <= 1``.
    - ``Dirichlet.log_pdf`` raises ``ValueError`` for a vector whose sum is
      not one but returns ``-inf`` for a negative component.
    - The ``log_marginal_likelihood`` of ``dirichlet_posterior`` is that of
      one ordered sequence; add ``log(n!/prod x_i!)`` for the counts.

Guide: docs/foundations/multivariate-laws.md

References:
    [REF-ANDERSON-2003], [REF-KOTZ-BALAKRISHNAN-JOHNSON-2000],
    [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997], [REF-MURPHY-2012],
    [REF-GELMAN-ETAL-2013].

"""

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
