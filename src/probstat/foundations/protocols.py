"""Dependency-neutral structural protocols for probability laws.

Univariate protocols describe scalar laws (PMF, PDF, CDF, survival,
quantile, hazard, mean, and variance). The multivariate protocols describe
vector-valued laws on ``R^d`` or the simplex: a dimension with a log density
or log mass function, a mean vector with a covariance matrix, and seeded
sampling of vectors. Analytic families conform by structure; no package
imports another to share them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from .random_streams import RandomStream

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


@runtime_checkable
class HasDimension(Protocol):
    """A vector-valued law on a space of fixed dimension ``d``.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].

    """

    @property
    def dimension(self) -> int:
        """Return the dimension ``d`` of the values.

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3]. No equation is evaluated.

        """
        ...


@runtime_checkable
class MultivariateLogDensity(HasDimension, Protocol):
    """A continuous vector-valued law with a log density on ``R^d``.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].

    """

    def log_pdf(self, value: Iterable[float]) -> float:
        """Return ``log f(x)`` at a vector of length ``d`` (``-inf`` off support).

        Args:
            value: Finite vector accepted by the implementation.

        Returns:
            The natural logarithm of the density.

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3]. The concrete
            implementation supplies the applicable density equation.

        """
        ...


@runtime_checkable
class MultivariateLogMass(HasDimension, Protocol):
    """A discrete vector-valued law with a log probability mass function.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].

    """

    def log_pmf(self, counts: Iterable[int]) -> float:
        """Return ``log P(X = x)`` at an integer vector of length ``d``.

        Args:
            counts: Integer vector accepted by the implementation.

        Returns:
            The natural logarithm of the probability (``-inf`` off support).

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3]. The concrete
            implementation supplies the applicable mass equation.

        """
        ...


@runtime_checkable
class MultivariateMoments(HasDimension, Protocol):
    """A vector-valued law exposing its mean vector and covariance matrix.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].

    """

    @property
    def mean(self) -> tuple[float, ...]:
        """Return the mean vector ``E[X]``.

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3]. The concrete
            implementation supplies the applicable expectation equation.

        """
        ...

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        """Return the covariance matrix ``E[(X - mu)(X - mu)^T]``.

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3]. The concrete
            implementation supplies the applicable covariance equation.

        """
        ...


@runtime_checkable
class MultivariateSampler(HasDimension, Protocol):
    """A vector-valued law that draws reproducible samples.

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[float, ...], ...]:
        """Return ``size`` independent draws, each a vector of length ``d``.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed, stream, or ``None`` (not reproducible).

        Returns:
            A tuple of vectors.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. The concrete implementation supplies
            the applicable sampling construction.

        """
        ...
