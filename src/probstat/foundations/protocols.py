"""Dependency-neutral structural protocols for univariate probability laws."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

ValueContra = TypeVar("ValueContra", contravariant=True)


@runtime_checkable
class ProbabilityMassFunction(Protocol[ValueContra]):
    """A discrete law that assigns probability mass to candidate values.

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
class LogProbabilityMassFunction(Protocol[ValueContra]):
    """A discrete law that evaluates natural log probability mass.

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.2].

    """

    def log_pmf(self, value: ValueContra) -> float:
        """Return natural log mass at ``value``.

        Args:
            value: Candidate discrete support value.

        Returns:
            Log mass, conventionally negative infinity for zero mass.

        References:
            [REF-NIST-DLMF-GAMMA, Section 5.2]. The implementation supplies the
            applicable mass equation.

        """
        ...


@runtime_checkable
class ProbabilityDensity(Protocol):
    """A continuous law that evaluates a probability density function.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3].

    """

    def pdf(self, x: float) -> float:
        """Return probability density at ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            Non-negative density; a density value is not a point probability.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. The implementation supplies
            the applicable density equation.

        """
        ...


@runtime_checkable
class LogProbabilityDensity(Protocol):
    """A continuous law that evaluates natural log density.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3].

    """

    def log_pdf(self, x: float) -> float:
        """Return natural log density at ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            Log density, conventionally negative infinity for zero density.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. The implementation supplies
            the applicable density equation.

        """
        ...


@runtime_checkable
class SurvivalFunction(Protocol):
    """A law exposing the direct upper-tail probability ``P(X>x)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3].

    """

    def sf(self, x: float) -> float:
        """Return survival probability at ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            Upper-tail probability, preferably evaluated without cancellation.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. The implementation supplies
            the applicable survival equation.

        """
        ...


@runtime_checkable
class QuantileFunction(Protocol):
    """A law exposing a generalized inverse cumulative distribution.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 2].

    """

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF at a cumulative probability.

        Args:
            probability: Target cumulative probability in ``[0, 1]``.

        Returns:
            Smallest ordered value whose CDF reaches the target, with documented
            endpoint conventions.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2]. The implementation supplies
            the applicable quantile equation.

        """
        ...


@runtime_checkable
class HazardFunction(Protocol):
    """A continuous law exposing instantaneous event rate where defined.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 2].

    """

    def hazard(self, x: float) -> float:
        """Return density divided by survival probability at ``x``.

        Args:
            x: Real evaluation point accepted by the implementation.

        Returns:
            Instantaneous rate under the implementation's domain convention.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 2]. The implementation
            supplies the applicable hazard equation.

        """
        ...


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
    ProbabilityMassFunction[ValueContra],
    CumulativeDistribution,
    HasMeanAndVariance,
    Protocol[ValueContra],
):
    """A discrete univariate law with PMF, CDF, mean, and variance.

    References:
        [REF-ROSS-2019, Chapters 4-6].

    """

@runtime_checkable
class ContinuousProbabilityLaw(
    ProbabilityDensity, CumulativeDistribution, HasMeanAndVariance, Protocol
):
    """A continuous univariate law with PDF, CDF, mean, and variance.

    References:
        [REF-ROSS-2019, Chapter 5].

    """

    ...
