"""A single study object that runs every UQ analysis on one model.

``UQStudy`` is the integration layer: it owns one
:class:`~probstat.modeling.ProbabilisticModel`, one root random stream, and a
log of every analysis performed. Each stage draws its randomness from a named
child stream, so the whole study is reproducible from one seed, and every
stage records the forward-model evaluations it charged. The study does not
re-implement any method; it calls the domain packages and keeps their results.

Accounting and provenance:

- Every stage record states its status (``"complete"``, ``"aborted"``, or
  ``"budget-exceeded"``), completed and failed evaluations, cache hits, the
  stage settings, any convergence warnings, and the input law it used
  (a label such as ``"prior"`` or ``"posterior #1"`` with a stable hash).
  A stage that raises after spending evaluations is still recorded, so
  :attr:`UQStudy.total_evaluations` never under-states the cost.
- An optional ``evaluation_budget`` refuses a stage whose cost is known and
  too large before it runs, and stops any other stage at the limit.
- :meth:`UQStudy.adopt_posterior` (or :meth:`UQStudy.with_inputs`) replaces
  the input law, so stages after a calibration use the posterior.
- :meth:`UQStudy.to_dict`, :meth:`UQStudy.to_json`, and
  :meth:`UQStudy.save_json` write a JSON record with the library and Python
  versions, platform, UTC creation time, root seed, and every stage.

The implementation is split by domain: ``_records`` holds the result
records, ``_context`` and ``_base`` the shared bookkeeping, one mixin module
per group of stages holds the stage methods, and ``_core`` composes
:class:`UQStudy` (ADR-0073). Only the names imported here are public.

References:
    [REF-SMITH-UQ-2024, Chapter 1].

"""

from ._core import UQStudy as UQStudy
from ._records import STAGE_STATUSES as STAGE_STATUSES
from ._records import DesignPointAnalysis as DesignPointAnalysis
from ._records import DesignRecommendation as DesignRecommendation
from ._records import DiscrepancyAnalysis as DiscrepancyAnalysis
from ._records import DiscrepancyStudyPrediction as DiscrepancyStudyPrediction
from ._records import InformationSensitivity as InformationSensitivity
from ._records import StageRecord as StageRecord
from ._records import TailAnalysis as TailAnalysis
from ._records import TargetedPropagation as TargetedPropagation
