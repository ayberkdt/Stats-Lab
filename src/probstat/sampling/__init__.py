"""Space-filling and low-discrepancy sampling designs on the unit hypercube."""

from .designs import (
    HALTON_MAX_DIMENSION,
    SOBOL_MAX_DIMENSION,
    SamplingDesign,
    SobolBalanceWarning,
    halton,
    iman_conover,
    latin_hypercube,
    maximin_criterion,
    maximin_latin_hypercube,
    monte_carlo_design,
    radical_inverse,
    sobol,
    star_discrepancy_bound,
)

__all__ = [
    "HALTON_MAX_DIMENSION",
    "SOBOL_MAX_DIMENSION",
    "SamplingDesign",
    "SobolBalanceWarning",
    "halton",
    "iman_conover",
    "latin_hypercube",
    "maximin_criterion",
    "maximin_latin_hypercube",
    "monte_carlo_design",
    "radical_inverse",
    "sobol",
    "star_discrepancy_bound",
]
