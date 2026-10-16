"""Forward uncertainty propagation and output summaries.

:func:`nested_propagation` separates aleatory and epistemic uncertainty in a
double loop and returns a probability box and epistemic intervals of output
statistics.
"""

from .forward import (
    OutputSummary,
    PropagationResult,
    expectation_by_quadrature,
    propagate_samples,
)
from .mixed import (
    EpistemicStatistic,
    MixedPropagationResult,
    ProbabilityBox,
    nested_propagation,
)
from .study import (
    DESIGNS,
    OutputDistribution,
    PropagationStudy,
    compare_designs,
    make_design,
    propagate,
    summarize,
)

__all__ = [
    "DESIGNS",
    "EpistemicStatistic",
    "MixedPropagationResult",
    "OutputDistribution",
    "OutputSummary",
    "ProbabilityBox",
    "PropagationResult",
    "PropagationStudy",
    "compare_designs",
    "expectation_by_quadrature",
    "make_design",
    "nested_propagation",
    "propagate",
    "propagate_samples",
    "summarize",
]
