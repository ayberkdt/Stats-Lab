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

Main entry points:
    ``ProbabilisticModel``: inputs, forward model, and QoIs in one object that
        every workflow accepts.
    ``JointInput``: joint input law, marginals plus a copula (Gaussian by
        default).
    ``RandomInput``: one named uncertain input with its marginal law and unit.
    ``nataf_correlation``: latent normal correlation that reproduces a target
        Pearson correlation of two continuous marginals.
    ``fit_joint_input``: marginals and a rank-based Gaussian copula fitted to
        data columns.
    ``CVineCopula``: canonical vine built from bivariate ``PairCopula`` blocks.
    ``fit_vine_copula``: sequential structure, family, and parameter fit of a
        C- or D-vine.
    ``RandomFieldInput``: truncated Karhunen-Loeve field as standard-normal
        inputs.

Assumptions:
    Each marginal is a law with ``pdf``, ``cdf``, ``quantile``, ``mean``, and
    ``variance``; all dependence lives in the copula. The forward model is a
    function of the named random inputs and the fixed parameters only, and
    every QoI it returns is a finite real number; a missing or non-finite QoI
    is a failed run. A Gaussian copula has no tail dependence.

Common pitfalls:
    - ``JointInput(inputs, correlation)`` takes the latent normal correlation
      of the Nataf model, not the Pearson correlation of the physical inputs.
      For two lognormals with log-scale deviations 0.5 and 0.8, a physical
      correlation of 0.6 needs a latent value of 0.6612
      (``nataf_correlation``). ``JointInput.from_pearson`` solves this for a
      whole matrix; it refuses discrete marginals (use ``from_spearman`` or
      ``from_kendall``), and with ``repair=True`` it replaces a latent matrix
      that is not positive definite with a ``UserWarning``, after which the
      targets are matched only approximately.
    - With a non-Gaussian copula or a vine, the standard space is the
      Rosenblatt transform in input order, so FORM results can change when
      the inputs are reordered although the joint law does not.
    - ``ProbabilisticModel.evaluations`` counts completed calls made through
      the object by every workflow so far; it is not reset per study. Failed
      attempts and cache hits are counted separately in
      ``failed_evaluations`` and ``cache_hits``.
    - ``on_failure`` defaults to ``"raise"``. Under ``"skip"`` or ``"record"``
      failed runs are dropped, so every estimate is conditional on success
      and biased unless failing is unrelated to the output.
    - ``evaluate(mapping)`` raises ``ValueError`` for missing, unknown, or
      fixed-parameter names instead of ignoring them.

Guide: docs/foundations/probabilistic-modeling.md

References:
    [REF-LIU-DERKIUREGHIAN-1986], [REF-NELSEN-2006, Chapter 2],
    [REF-LEBRUN-DUTFOY-2009], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009],
    [REF-SMITH-UQ-2024, Chapters 1-5].

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
