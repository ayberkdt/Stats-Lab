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
    QuantileInterval,
    compare_designs,
    make_design,
    order_statistic_interval,
    propagate,
    replicate_interval,
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
    "QuantileInterval",
    "compare_designs",
    "expectation_by_quadrature",
    "make_design",
    "nested_propagation",
    "order_statistic_interval",
    "propagate",
    "propagate_samples",
    "replicate_interval",
    "summarize",
]
