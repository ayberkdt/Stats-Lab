"""Dependency-neutral structural protocols for univariate probability laws."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

ValueContra = TypeVar("ValueContra", contravariant=True)


@runtime_checkable
class CumulativeDistribution(Protocol):
    """A univariate law that can evaluate a cumulative distribution function.

    The protocol expresses interface compatibility only. Each implementation
    owns its domain validation, numerical method, and equation record.

    References:
        [REF-ROSS-2019, Chapters 4-5].

    """

    def cdf(self, x: float) -> float:
        """Return accumulated probability through ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            A probability in ``[0, 1]``.

        References:
            [REF-ROSS-2019, Chapters 4-5]. The concrete implementation supplies
            the applicable CDF equation.

        """
        ...


@runtime_checkable
class HasMeanAndVariance(Protocol):
    """A probability law exposing finite mean and variance summaries.

    References:
        [REF-ROSS-2019, Chapters 4-6].

    """

    @property
    def mean(self) -> float:
        """Return the law's finite expected value.

        References:
            [REF-ROSS-2019, Chapters 4-6]. The concrete implementation supplies
            the applicable expectation equation.

        """
        ...

    @property
    def variance(self) -> float:
        """Return the law's finite non-negative variance.

        References:
            [REF-ROSS-2019, Chapters 4-6]. The concrete implementation supplies
            the applicable variance equation.

        """
        ...


@runtime_checkable
class DiscreteProbabilityLaw(
    CumulativeDistribution, HasMeanAndVariance, Protocol[ValueContra]
):
    """A discrete univariate law with PMF, CDF, mean, and variance.

    References:
        [REF-ROSS-2019, Chapters 4-6].

    """

    def pmf(self, value: ValueContra) -> float:
        """Return probability mass at ``value``.

        Args:
            value: Candidate discrete support value.

        Returns:
            Non-negative point probability.

        References:
            [REF-ROSS-2019, Chapters 4-6]. The concrete implementation supplies
            the applicable PMF equation.

        """
        ...


@runtime_checkable
class ContinuousProbabilityLaw(
    CumulativeDistribution, HasMeanAndVariance, Protocol
):
    """A continuous univariate law with PDF, CDF, mean, and variance.

    References:
        [REF-ROSS-2019, Chapter 5].

    """

    def pdf(self, x: float) -> float:
        """Return probability density at ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            Non-negative density; a density value is not a point probability.

        References:
            [REF-ROSS-2019, Chapter 5]. The concrete implementation supplies the
            applicable PDF equation.

        """
        ...
