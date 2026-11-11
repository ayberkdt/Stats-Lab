"""The specification error shared by the probabilistic-programming modules."""

from __future__ import annotations


class ModelSpecificationError(ValueError):
    """A model violates the rules of the probabilistic-programming interface.

    Raised for duplicate or unknown site names, unsupported supports or
    shapes, missing observed data, and models whose site structure changes
    between executions. It is never converted into a ``-inf`` log-density.

    References:
        [REF-CARPENTER-ETAL-2017], [REF-BINGHAM-ETAL-2019].

    """
