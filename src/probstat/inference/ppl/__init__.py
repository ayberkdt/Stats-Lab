"""A small probabilistic-programming interface over the inference engines.

A model is a plain Python function of a handle ``m``::

    def eight_schools(m):
        mu = m.sample("mu", Normal(0.0, 5.0))
        tau = m.sample("tau", TruncatedNormal(0.0, 5.0, lower=0.0))
        theta = m.sample("theta", Normal(mu, tau), shape=8, reparam="noncentered")
        m.observe("y", elementwise(Normal, theta, s), y)

:func:`trace` runs the model once and returns a :class:`ModelIR`: the ordered
latent sites, their supports and bijections to ``R`` (identity, shifted log,
negated shifted log, scaled logit), and the joint log-density on the
unconstrained vector (prior + likelihood + log-Jacobian) with a reverse-mode
gradient when the program is traceable and a central finite-difference
gradient otherwise (ADR-0069). :func:`sample` (NUTS, HMC, or adaptive
Metropolis), :func:`fit_advi`, and :func:`laplace` hand that density to the
existing engines and return draws keyed by site name on the natural scale;
:func:`prior_predictive`, :func:`posterior_predictive`, and
:func:`predictive_pvalues` simulate from the model.

Structured latent sites (ADR-0082) take values on a simplex (Dirichlet),
on ordered vectors (:class:`OrderedVector`), on Cholesky factors of
correlation matrices and on correlation matrices (LKJ), or on covariance
matrices (Wishart, inverse Wishart), through the transforms of
:class:`StructuredSupport`. Scalar discrete sites with a finite support are
summed out by enumeration with ``m.sample(..., marginalize=True)``, locally
inside the iterations of ``m.plate`` and jointly elsewhere, and are
recovered from their exact conditional law (:class:`DiscreteDraws`).

ADR-0084 adds dynamic site structure, nested plates, and vector-valued
discrete sites. The sites that follow a discrete site outside plates may
depend on its value: :func:`trace` records the sites of every configuration
(:class:`TracedStructure`), the unconstrained vector holds the union of the
continuous sites, and a configuration that lacks a site gives it its prior
as a pseudo-prior, which leaves the marginal of every configuration
unchanged. ``m.plate`` nests, with the discrete sites of each inner
iteration enumerated per iteration, and a discrete site may be a vector of
independent elements or a multinomial count vector.
"""

from ._api import (
    PosteriorSamples,
    PredictiveSamples,
    fit_advi,
    laplace,
    posterior_predictive,
    predictive_pvalues,
    prior_predictive,
    sample,
)
from ._discrete import DiscreteDraws, TruncatedDiscrete
from ._errors import ModelSpecificationError
from ._program import (
    Bijection,
    LogDensityTerms,
    ModelHandle,
    ModelIR,
    SiteInfo,
    TracedStructure,
    bijection_for,
    condition,
    elementwise,
    trace,
)
from ._structured import OrderedVector, StructuredSupport

__all__ = [
    "Bijection",
    "DiscreteDraws",
    "LogDensityTerms",
    "ModelHandle",
    "ModelIR",
    "ModelSpecificationError",
    "OrderedVector",
    "PosteriorSamples",
    "PredictiveSamples",
    "SiteInfo",
    "StructuredSupport",
    "TracedStructure",
    "TruncatedDiscrete",
    "bijection_for",
    "condition",
    "elementwise",
    "fit_advi",
    "laplace",
    "posterior_predictive",
    "predictive_pvalues",
    "prior_predictive",
    "sample",
    "trace",
]
