"""Finite sample spaces, probability measures, and exact counting tools."""

from .counting import combinations, factorial, multiplication_rule, permutations
from .finite import FiniteEvent, FiniteProbabilityMeasure, FiniteSampleSpace
from .random_variables import FiniteJointDistribution, FiniteRandomVariable

__all__ = [
    "FiniteEvent",
    "FiniteJointDistribution",
    "FiniteProbabilityMeasure",
    "FiniteRandomVariable",
    "FiniteSampleSpace",
    "combinations",
    "factorial",
    "multiplication_rule",
    "permutations",
]

