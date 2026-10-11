"""Information theory: entropy, divergence, and mutual information.

* :mod:`.discrete` — Shannon, joint and conditional entropy, mutual
  information, Kullback-Leibler, cross-entropy, and Jensen-Shannon
  divergences for finite PMFs.
* :mod:`.differential` — closed-form differential entropies and Gaussian
  Kullback-Leibler divergences, plus :func:`differential_entropy` dispatching
  on distribution objects.
* :mod:`.estimators` — Kozachenko-Leonenko entropy and KSG mutual-information
  estimators from samples.
* :mod:`.divergences` — total variation, Hellinger, Bhattacharyya, Renyi and
  alpha-divergences for finite PMFs and, in closed form, Gaussian laws.
* :mod:`.geometry` — Fisher information matrices (closed form and numerical)
  and closed-form Fisher-Rao distances.
* :mod:`.transport` — exact discrete optimal transport with dual
  certificates, log-domain Sinkhorn, and Sinkhorn divergences.
* :mod:`.wasserstein` — Gaussian (Bures) and quantile-based 1-D Wasserstein
  distances and ``W_2`` barycenters.

Entropies, KL and Renyi divergences take ``unit="nats"`` (natural
logarithm) or ``unit="bits"``; transport costs are in the units of the cost
matrix. The package sits at layer 2 and imports only layers 0-1.
"""

from ._units import UNITS, InformationUnit
from .differential import (
    beta_entropy,
    differential_entropy,
    exponential_entropy,
    gamma_entropy,
    gev_entropy,
    kl_multivariate_normal,
    kl_normal,
    lognormal_entropy,
    multivariate_normal_entropy,
    normal_entropy,
    uniform_entropy,
)
from .discrete import (
    conditional_entropy,
    cross_entropy,
    entropy,
    jensen_shannon_divergence,
    joint_entropy,
    kl_divergence,
    marginals,
    mutual_information,
)
from .divergences import (
    alpha_divergence,
    alpha_divergence_gaussian,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    bhattacharyya_distance_gaussian,
    hellinger_distance,
    hellinger_distance_gaussian,
    renyi_divergence,
    renyi_divergence_gaussian,
    total_variation_distance,
    total_variation_gaussian_shared_covariance,
    total_variation_normal,
)
from .estimators import kozachenko_leonenko_entropy, ksg_mutual_information
from .geometry import (
    bernoulli_fisher_information,
    beta_fisher_information,
    binomial_fisher_information,
    categorical_fisher_information,
    exponential_fisher_information,
    fisher_information,
    fisher_rao_bernoulli,
    fisher_rao_binomial,
    fisher_rao_categorical,
    fisher_rao_exponential,
    fisher_rao_normal,
    fisher_rao_normal_mean,
    fisher_rao_poisson,
    gamma_fisher_information,
    multivariate_normal_mean_fisher_information,
    normal_fisher_information,
    numerical_fisher_information,
    poisson_fisher_information,
)
from .transport import (
    OptimalityCertificate,
    SinkhornResult,
    TransportPlan,
    discrete_wasserstein,
    exact_transport,
    power_cost_matrix,
    sinkhorn,
    sinkhorn_divergence,
    squared_euclidean_cost_matrix,
    transport_certificate,
)
from .wasserstein import (
    GaussianBarycenter,
    QuantileBarycenter,
    bures_distance,
    gaussian_barycenter,
    gaussian_wasserstein,
    normal_wasserstein,
    quantile_barycenter,
    quantile_wasserstein,
)

__all__ = [
    "GaussianBarycenter",
    "InformationUnit",
    "OptimalityCertificate",
    "QuantileBarycenter",
    "SinkhornResult",
    "TransportPlan",
    "UNITS",
    "alpha_divergence",
    "alpha_divergence_gaussian",
    "bernoulli_fisher_information",
    "beta_entropy",
    "beta_fisher_information",
    "bhattacharyya_coefficient",
    "bhattacharyya_distance",
    "bhattacharyya_distance_gaussian",
    "binomial_fisher_information",
    "bures_distance",
    "categorical_fisher_information",
    "conditional_entropy",
    "cross_entropy",
    "differential_entropy",
    "discrete_wasserstein",
    "entropy",
    "exact_transport",
    "exponential_entropy",
    "exponential_fisher_information",
    "fisher_information",
    "fisher_rao_bernoulli",
    "fisher_rao_binomial",
    "fisher_rao_categorical",
    "fisher_rao_exponential",
    "fisher_rao_normal",
    "fisher_rao_normal_mean",
    "fisher_rao_poisson",
    "gamma_entropy",
    "gamma_fisher_information",
    "gaussian_barycenter",
    "gaussian_wasserstein",
    "gev_entropy",
    "hellinger_distance",
    "hellinger_distance_gaussian",
    "jensen_shannon_divergence",
    "joint_entropy",
    "kl_divergence",
    "kl_multivariate_normal",
    "kl_normal",
    "kozachenko_leonenko_entropy",
    "ksg_mutual_information",
    "lognormal_entropy",
    "marginals",
    "multivariate_normal_entropy",
    "multivariate_normal_mean_fisher_information",
    "mutual_information",
    "normal_entropy",
    "normal_fisher_information",
    "normal_wasserstein",
    "numerical_fisher_information",
    "poisson_fisher_information",
    "power_cost_matrix",
    "quantile_barycenter",
    "quantile_wasserstein",
    "renyi_divergence",
    "renyi_divergence_gaussian",
    "sinkhorn",
    "sinkhorn_divergence",
    "squared_euclidean_cost_matrix",
    "total_variation_distance",
    "total_variation_gaussian_shared_covariance",
    "total_variation_normal",
    "transport_certificate",
    "uniform_entropy",
]
