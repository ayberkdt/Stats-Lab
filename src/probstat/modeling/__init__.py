"""Central probabilistic-model contract shared by every UQ workflow.

Define the joint input law, the forward model, and the quantities of interest
once with :class:`ProbabilisticModel`; propagation, sensitivity, reliability,
calibration, surrogates, and risk analysis all consume the same object.
A :class:`JointInput` combines continuous or discrete marginals with a copula
(Gaussian by default; Student-t, Clayton, Gumbel, Frank and independence in
:mod:`probstat.modeling.copulas`). :class:`RandomInput` carries an aleatory or
epistemic nature and :class:`IntervalInput` describes a quantity known only
by bounds. :class:`CVineCopula` and :class:`DVineCopula`
(:mod:`probstat.modeling.vines`) build a copula from bivariate
:class:`PairCopula` blocks, and :func:`fit_vine_copula` fits one by
sequential maximum likelihood. :class:`RandomFieldInput` exposes a truncated
Karhunen-Loeve expansion of a random field as standard-normal inputs of that
object.
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
from .model import EvaluationBatch, EvaluationCounter, ProbabilisticModel
from .random_field import RandomFieldInput, karhunen_loeve_inputs
from .vines import (
    PAIR_COPULA_FAMILIES,
    CVineCopula,
    DVineCopula,
    PairCopula,
    VineCopula,
    VineCopulaFit,
    VineEdge,
    VineEdgeFit,
    fit_vine_copula,
    kendall_independence_p_value,
)

__all__ = [
    "ArchimedeanCopula",
    "CVineCopula",
    "ClaytonCopula",
    "ContinuousMarginal",
    "Copula",
    "CopulaFit",
    "DVineCopula",
    "DiscreteMarginal",
    "EvaluationBatch",
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
    "PAIR_COPULA_FAMILIES",
    "PairCopula",
    "ProbabilisticModel",
    "RandomFieldInput",
    "RandomInput",
    "StudentTCopula",
    "VineCopula",
    "VineCopulaFit",
    "VineEdge",
    "VineEdgeFit",
    "archimedean_theta_from_tau",
    "fit_copula",
    "fit_gaussian_copula",
    "fit_joint_input",
    "fit_vine_copula",
    "frank_tau",
    "gauss_hermite",
    "is_discrete_law",
    "karhunen_loeve_inputs",
    "kendall_independence_p_value",
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
