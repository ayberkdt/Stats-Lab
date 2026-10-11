"""Space-filling and low-discrepancy sampling designs on the unit hypercube."""

from .designs import (
    SOBOL_MAX_DIMENSION,
    SamplingDesign,
    halton,
    latin_hypercube,
    monte_carlo_design,
    radical_inverse,
    sobol,
    star_discrepancy_bound,
)

__all__ = [
    "SOBOL_MAX_DIMENSION",
    "SamplingDesign",
    "halton",
    "latin_hypercube",
    "monte_carlo_design",
    "radical_inverse",
    "sobol",
    "star_discrepancy_bound",
]
