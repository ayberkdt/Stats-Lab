"""Central probabilistic-model contract shared by every UQ workflow.

Define the joint input law, the forward model, and the quantities of interest
once with :class:`ProbabilisticModel`; propagation, sensitivity, reliability,
calibration, surrogates, and risk analysis all consume the same object.
A :class:`JointInput` combines continuous or discrete marginals with a copula
(Gaussian by default; Student-t, Clayton, Gumbel, Frank and independence in
:mod:`probstat.modeling.copulas`). :class:`RandomInput` carries an aleatory or
epistemic nature and :class:`IntervalInput` describes a quantity known only
by bounds. :class:`RandomFieldInput` exposes a truncated Karhunen-Loeve
expansion of a random field as standard-normal inputs of that object.
"""

from .copulas import (
    ArchimedeanCopula,
    ClaytonCopula,
    Copula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependenceCopula,
    NearestCorrelation,
    StudentTCopula,
    frank_tau,
    nearest_correlation_matrix,
)
from .fitting import (
    CopulaFit,
    GaussianCopulaFit,
    JointInputFit,
    archimedean_theta_from_tau,
    fit_copula,
    fit_gaussian_copula,
    fit_joint_input,
    kendall_tau,
    kendall_to_latent_correlation,
    normal_scores,
    pseudo_observations,
    spearman_to_latent_correlation,
)
from .inputs import (
    ContinuousMarginal,
    DiscreteMarginal,
    IntegerCodedCategorical,
    IntervalInput,
    JointInput,
    RandomInput,
    gauss_hermite,
    is_discrete_law,
    nataf_correlation,
    standard_normal_cdf,
    standard_normal_quantile,
)
from .model import EvaluationCounter, ProbabilisticModel
from .random_field import RandomFieldInput, karhunen_loeve_inputs

__all__ = [
    "ArchimedeanCopula",
    "ClaytonCopula",
    "ContinuousMarginal",
    "Copula",
    "CopulaFit",
    "DiscreteMarginal",
    "EvaluationCounter",
    "FrankCopula",
    "GaussianCopula",
    "GaussianCopulaFit",
    "GumbelCopula",
    "IndependenceCopula",
    "IntegerCodedCategorical",
    "IntervalInput",
    "JointInput",
    "JointInputFit",
    "NearestCorrelation",
    "ProbabilisticModel",
    "RandomFieldInput",
    "RandomInput",
    "StudentTCopula",
    "archimedean_theta_from_tau",
    "fit_copula",
    "fit_gaussian_copula",
    "fit_joint_input",
    "frank_tau",
    "gauss_hermite",
    "is_discrete_law",
    "karhunen_loeve_inputs",
    "kendall_tau",
    "kendall_to_latent_correlation",
    "nataf_correlation",
    "nearest_correlation_matrix",
    "normal_scores",
    "pseudo_observations",
    "spearman_to_latent_correlation",
    "standard_normal_cdf",
    "standard_normal_quantile",
]
