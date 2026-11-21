"""Space-filling and low-discrepancy sampling designs on the unit hypercube.

Every design is a ``SamplingDesign``: points in ``(0, 1)^d`` with the method
name, whether it was randomized, and its seed. Mapping the points to
physical inputs (inverse CDFs, a copula) happens in ``modeling``, not here.

Main entry points:
    ``monte_carlo_design``: independent uniform points.
    ``latin_hypercube``: one point in every marginal stratum.
    ``maximin_latin_hypercube``: a Latin hypercube optimized for spread.
    ``sobol``: Sobol digital net, optionally shift- or Owen-scrambled.
    ``halton``: Halton sequence, optionally scrambled or shifted.
    ``iman_conover``: reorder a design's columns to a target rank
        correlation without changing its marginal strata.
    ``star_discrepancy_bound``: grid lower bound on the star discrepancy.
    ``rank1_lattice``: rank-1 lattice points, optionally randomly shifted.
    ``cbc_generating_vector``, ``korobov_generating_vector``: generating
        vectors chosen by the shift-averaged worst-case error
        (``lattice_squared_error``).
    ``rqmc_mean``: integral estimate from independently randomized rules
        with a replicate standard error and an approximate t interval.
    ``rqmc_median``, ``median_random_lattice``: median of randomized rules
        (no interval).

Assumptions:
    Designs are uniform on the unit cube with independent coordinates
    (until ``iman_conover`` induces rank correlation). Quasi-Monte Carlo
    error rates assume an integrand of bounded variation; randomized
    designs are needed for unbiased estimates and replicate-based error
    bars. Lattice construction assumes a prime number of points and product
    weights; its criterion refers to one stated Sobolev space.

Common pitfalls:
    - ``sobol`` and ``halton`` are deterministic unless ``scramble`` (or,
      for Halton, ``shift``) is set; the ``seed`` is then ignored, every
      call returns the same points, and replicates give no error estimate.
    - A ``sobol`` size that is not a power of two emits
      ``SobolBalanceWarning``: the points are valid but lose the net
      balance.
    - Starting points differ: ``halton`` skips the origin by default
      (``skip=1``), while unscrambled ``sobol`` centres each point in its
      cell (``centered``), so its first point is ``(0.125, 0.125)`` for
      four points in two dimensions, not the origin.
    - ``seed=None`` draws fresh entropy for the randomized designs; pass an
      integer or ``RandomStream`` to reproduce them.
    - ``star_discrepancy_bound`` is a lower bound evaluated on a grid and
      only for dimension at most four.
    - An unshifted ``rank1_lattice`` contains the origin; shift it before
      mapping through an inverse CDF.
    - ``rqmc_mean`` intervals are Student-t over a few non-normal replicates
      and can under-cover; ``rqmc_median`` and ``median_random_lattice``
      give no interval at all, and the median is not unbiased.
    - ``cbc_generating_vector`` is greedy: optimal in one and two
      dimensions, a heuristic beyond.

Guide: docs/foundations/uncertainty-propagation.md

References:
    [REF-OWEN-2013] (Monte Carlo and quasi-Monte Carlo designs),
    [REF-MCKAY-BECKMAN-CONOVER-1979] (Latin hypercube),
    [REF-SOBOL-1967] and [REF-JOE-KUO-2008] (Sobol sequence),
    [REF-OWEN-1995] (scrambled nets), [REF-HALTON-1960] (Halton sequence),
    [REF-IMAN-CONOVER-1982] (rank correlation),
    [REF-CRANLEY-PATTERSON-1976] (randomly shifted lattices),
    [REF-GODA-LECUYER-2022] and [REF-PAN-OWEN-2023] (median rules),
    [REF-LECUYER-NAKAYAMA-OWEN-TUFFIN-2023] (interval reliability).

"""

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
from .lattice import (
    LatticeRule,
    cbc_generating_vector,
    korobov_generating_vector,
    korobov_vector,
    lattice_squared_error,
    rank1_lattice,
)
from .randomized_integration import (
    RQMC_METHODS,
    RQMCResult,
    median_random_lattice,
    rqmc_mean,
    rqmc_median,
)

__all__ = [
    "HALTON_MAX_DIMENSION",
    "LatticeRule",
    "RQMCResult",
    "RQMC_METHODS",
    "SOBOL_MAX_DIMENSION",
    "SamplingDesign",
    "SobolBalanceWarning",
    "cbc_generating_vector",
    "halton",
    "iman_conover",
    "korobov_generating_vector",
    "korobov_vector",
    "latin_hypercube",
    "lattice_squared_error",
    "maximin_criterion",
    "maximin_latin_hypercube",
    "median_random_lattice",
    "monte_carlo_design",
    "radical_inverse",
    "rank1_lattice",
    "rqmc_mean",
    "rqmc_median",
    "sobol",
    "star_discrepancy_bound",
]
