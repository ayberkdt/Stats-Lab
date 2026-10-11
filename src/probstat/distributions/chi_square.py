"""Chi-square distribution as the gamma law ``Gamma(nu/2, 1/2)``."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import POSITIVE_FINITE
from .gamma import Gamma
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class ChiSquare:
    """A chi-square law with ``nu`` degrees of freedom.

    ``X ~ chi2(nu)`` is ``Gamma(shape=nu/2, rate=1/2)``; for integer ``nu`` it
    is the law of a sum of ``nu`` squared independent standard normals.

    Args:
        nu: Finite positive degrees of freedom (need not be an integer).

    References:
        [REF-JKB-1994, Chapter 18], [REF-NIST-DISTRIBUTIONS, Chi-Square
        Distribution].

    """

    nu: float
    _gamma: Gamma = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate ``nu`` and cache the equivalent gamma law.

        Raises:
            TypeError: If ``nu`` is not real.
            ValueError: If ``nu`` is non-finite or non-positive.

        References:
            [EQ-DIST-CHI2-001], [REF-JKB-1994, Chapter 18].

        """
        checked = POSITIVE_FINITE.validate(self.nu, name="nu")
        object.__setattr__(self, "nu", checked)
        object.__setattr__(self, "_gamma", Gamma(0.5 * checked, 0.5))

    @property
    def support(self) -> RealSupport:
        """Return non-negative real support.

        References:
            [EQ-DIST-CHI2-001], [REF-JKB-1994, Chapter 18].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def log_pdf(self, x: float) -> float:
        """Evaluate the chi-square log-density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density of ``Gamma(nu/2, 1/2)``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-CHI2-001], [EQ-DIST-GAM-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return self._gamma.log_pdf(x)

    def pdf(self, x: float) -> float:
        """Evaluate the chi-square density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density of ``Gamma(nu/2, 1/2)``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-CHI2-001], [EQ-DIST-GAM-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return self._gamma.pdf(x)

    def cdf(self, x: float) -> float:
        """Evaluate the chi-square CDF ``P(nu/2, x/2)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-CHI2-001], [EQ-DIST-GAM-002].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.2].

        """
        return self._gamma.cdf(x)

    def sf(self, x: float) -> float:
        """Evaluate the direct chi-square survival function ``Q(nu/2, x/2)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-CHI2-001], [EQ-DIST-GAM-002].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.2].

        """
        return self._gamma.sf(x)

    def quantile(self, probability: float) -> float:
        """Return the chi-square quantile.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Non-negative quantile, ``inf`` at one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-GAM-003], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.10], [REF-BRENT-1973, Chapter 4].

        """
        return self._gamma.quantile(probability)

    def isf(self, probability: float) -> float:
        """Return the chi-square inverse survival function.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Critical value ``x`` with ``P(X > x) = q``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.10], [REF-BRENT-1973, Chapter 4].

        """
        return self._gamma.isf(probability)

    @property
    def mean(self) -> float:
        """Return ``E[X] = nu``.

        Equation:
            [EQ-DIST-CHI2-002].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return self.nu

    @property
    def variance(self) -> float:
        """Return ``Var(X) = 2 nu``.

        Equation:
            [EQ-DIST-CHI2-002].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return 2.0 * self.nu

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible chi-square observations as ``2 Gamma(nu/2, 1)``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of non-negative pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-CHI2-001], [EQ-DIST-GAM-005].

        References:
            [REF-MARSAGLIA-TSANG-2000].

        """
        return self._gamma.sample(size, seed=seed)
