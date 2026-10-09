"""Finite sample spaces, probability measures, and exact counting tools."""

from .counting import combinations, factorial, multiplication_rule, permutations
from .finite import FiniteEvent, FiniteProbabilityMeasure, FiniteSampleSpace

__all__ = [
    "FiniteEvent",
    "FiniteProbabilityMeasure",
    "FiniteSampleSpace",
    "combinations",
    "factorial",
    "multiplication_rule",
    "permutations",
]

