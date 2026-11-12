"""Integrated studies: UQ on one probabilistic model, and mission assessment.

:class:`UQStudy` runs every UQ stage on one :class:`ProbabilisticModel`;
:class:`MissionAssessment` combines exact system reliability, epistemic
uncertainty, cost, and risk for a mission system.

Main entry points:
    ``UQStudy``: every UQ stage (propagation, sensitivity, reliability,
        calibration, validation, design, report) on one model and one root
        seed.
    ``StageRecord``: settings, results, warnings, cost, and stream seed of
        one stage run.
    ``TargetedPropagation``: result of ``UQStudy.propagate_until``, sized by
        a relative-error target.
    ``posterior_joint_input``: marginals-plus-Gaussian-copula approximation
        of posterior draws for the next forward stage.
    ``MissionAssessment``: exact system reliability, epistemic uncertainty,
        cost, and risk of a mission system.

Assumptions:
    One ``UQStudy`` owns one probabilistic model and one root stream; every
    stage draws from its own child stream, and the model's evaluation count
    is the cost of the study. Each stage inherits the assumptions of the
    domain function it delegates to.

Common pitfalls:
    - The failure side defaults differ: ``UQStudy.limit_state``,
      ``reliability``, ``design_points``, ``active_reliability``, and ``form``
      use ``failure="below"`` (low values fail), while
      ``UQStudy.propagate_until`` and ``reliability.ModelLimitState`` default
      to ``failure="above"``. Pass ``failure=`` explicitly when one threshold
      feeds several stages.
    - ``UQStudy(model)`` defaults to ``seed=0``, so a study is reproducible
      without a seed, but the k-th run of a stage uses the child stream
      ``(stage, k)``: calling ``study.propagate(...)`` twice gives two
      different samples. Rebuild the study with the same seed to repeat a
      result.
    - Defaults differ from the domain functions: ``UQStudy.propagate`` uses
      ``design="sobol"`` and ``size=4096`` (``propagation.propagate``:
      ``"mc"``), and ``UQStudy.sensitivity`` uses ``bootstrap=200``
      (``sensitivity.sobol_analysis``: ``0``, no interval).
    - ``propagate_until`` stops on a data-dependent rule; with
      ``final="reuse"`` its estimates come from the stopped sample and are
      biased. The default ``final="fresh"`` re-estimates on an independent
      sample of the chosen size.
    - ``adopt_posterior`` and ``posterior_joint_input`` approximate the
      posterior by a Gaussian copula, which loses curved and tail dependence,
      and make adopted inputs independent of the inputs that keep their
      prior.

Guide: docs/foundations/probabilistic-modeling.md

References:
    [REF-SMITH-UQ-2024, Chapters 1-5], [REF-SUDRET-2007, Chapter 1],
    [REF-STEELE-LEA-FLOOD-2014].

"""

from .chaining import (
    POSTERIOR_METHODS,
    EmpiricalMarginal,
    input_law_hash,
    posterior_joint_input,
)
from .mission import (
    MissionAssessment,
    MissionAssessmentResult,
    MissionCost,
    MissionReliability,
    MissionUncertainty,
    render_mission_report,
)
from .study import (
    STAGE_STATUSES,
    DesignPointAnalysis,
    DesignRecommendation,
    DiscrepancyAnalysis,
    DiscrepancyStudyPrediction,
    InformationSensitivity,
    StageRecord,
    TailAnalysis,
    TargetedPropagation,
    UQStudy,
)

__all__ = [
    "POSTERIOR_METHODS",
    "STAGE_STATUSES",
    "DesignPointAnalysis",
    "DesignRecommendation",
    "DiscrepancyAnalysis",
    "DiscrepancyStudyPrediction",
    "EmpiricalMarginal",
    "InformationSensitivity",
    "MissionAssessment",
    "MissionAssessmentResult",
    "MissionCost",
    "MissionReliability",
    "MissionUncertainty",
    "StageRecord",
    "TailAnalysis",
    "TargetedPropagation",
    "UQStudy",
    "input_law_hash",
    "posterior_joint_input",
    "render_mission_report",
]
