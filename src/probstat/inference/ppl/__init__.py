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
unconstrained vector (prior + likelihood + log-Jacobian) with a central
finite-difference gradient. :func:`sample` (NUTS, HMC, or adaptive
Metropolis), :func:`fit_advi`, and :func:`laplace` hand that density to the
existing engines and return draws keyed by site name on the natural scale;
:func:`prior_predictive`, :func:`posterior_predictive`, and
:func:`predictive_pvalues` simulate from the model.
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
from ._program import (
    Bijection,
    LogDensityTerms,
    ModelHandle,
    ModelIR,
    ModelSpecificationError,
    SiteInfo,
    bijection_for,
    condition,
    elementwise,
    trace,
)

__all__ = [
    "Bijection",
    "LogDensityTerms",
    "ModelHandle",
    "ModelIR",
    "ModelSpecificationError",
    "PosteriorSamples",
    "PredictiveSamples",
    "SiteInfo",
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
