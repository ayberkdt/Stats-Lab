"""Named uncertain inputs and their joint law: marginals plus a copula.

A :class:`JointInput` is the single description of input uncertainty shared by
propagation, sensitivity, reliability, calibration priors, and risk analysis.
Marginals are continuous laws with ``cdf``, ``quantile``, ``pdf``, ``mean``,
and ``variance``, or discrete integer-valued laws with ``pmf`` in place of
``pdf`` (Poisson, Binomial, Geometric, NegativeBinomial, Bernoulli, and
Categorical through an integer coding). Dependence is a copula
(:mod:`probstat.modeling.copulas`); the default is a Gaussian copula with
latent normal correlation ``R`` (independence is ``R = I`` or ``None``).

For the Gaussian copula the isoprobabilistic map between physical inputs ``x``
and independent standard-normal variables ``u`` is the Nataf transformation::

    z_i = Phi^-1(F_i(x_i)),   u = L^-1 z,   R = L L'

For every other copula it is the Rosenblatt transformation
``u_k = Phi^-1(C(F_k(x_k) | F_1(x_1), ..., F_{k-1}(x_{k-1})))``, which depends
on the input order. Reliability methods work in ``u``; samplers map
unit-hypercube designs through the inverse of the same map, so every workflow
sees one consistent law.

Discrete inputs are sampled exactly by the generalized inverse
``x = F^-1(p)``. The reverse map is not one-to-one (a whole interval of
latent values maps to one count); :meth:`JointInput.to_standard_normal` uses
the distributional transform ``F(x-) + V (F(x) - F(x-))`` and gradient-based
methods (FORM/SORM) refuse discrete inputs.

Each :class:`RandomInput` carries an :class:`~probstat.uq.UncertaintyNature`
(aleatory by default); :class:`IntervalInput` describes an epistemic quantity
known only by bounds. :mod:`probstat.propagation.mixed` separates the two in a
nested (double-loop) propagation.
"""

from __future__ import annotations

import inspect
import math
import random
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol, runtime_checkable

from probstat.distributions import Categorical
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    dot,
    identity,
    inverse_spd,
    log_det_spd,
    matvec,
    solve_lower,
    symmetric_eigen,
    symmetrize,
)
from probstat.optimization import brent_root
from probstat.sampling import SamplingDesign, monte_carlo_design
from probstat.uq.taxonomy import UncertaintyNature

from .copulas import (
    Copula,
    GaussianCopula,
    IndependenceCopula,
    nearest_correlation_matrix,
)

_STANDARD_NORMAL = NormalDist()
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


@runtime_checkable
class ContinuousMarginal(Protocol):
    """Structural interface required of a marginal input law.

    References:
        [REF-ROSS-2019, Chapter 5].

    """

    def pdf(self, x: float) -> float:
        """Return the density at ``x``.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    def cdf(self, x: float) -> float:
        """Return ``P(X <= x)``.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    @property
    def mean(self) -> float:
        """Return the expectation.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    @property
    def variance(self) -> float:
        """Return the variance.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...


@runtime_checkable
class DiscreteMarginal(Protocol):
    """Structural interface of an integer-valued marginal input law.

    ``quantile`` must be the generalized inverse ``inf{k : F(k) >= p}``; it
    may return an ``int`` or an integral ``float``.

    References:
        [REF-ROSS-2019, Chapter 4].

    """

    def pmf(self, k: int) -> float:
        """Return ``P(X = k)``.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        ...

    def cdf(self, x: float) -> float:
        """Return ``P(X <= x)``.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        ...

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        ...

    @property
    def mean(self) -> float:
        """Return the expectation.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        ...

    @property
    def variance(self) -> float:
        """Return the variance.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        ...


Marginal = ContinuousMarginal | DiscreteMarginal


def _has_static(law: object, attribute: str) -> bool:
    """Return whether ``law`` defines ``attribute`` without evaluating it.

    ``inspect.getattr_static`` finds properties and methods on the type, so
    a property that raises (``StudentT(1).mean`` for the Cauchy law) does not
    make the attribute look missing.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    try:
        inspect.getattr_static(law, attribute)
    except AttributeError:
        return False
    return True


def is_discrete_law(law: object) -> bool:
    """Return whether a marginal law is discrete (has ``pmf`` and no ``pdf``).

    Args:
        law: Candidate marginal law.

    Returns:
        ``True`` for integer-valued laws such as Poisson or Binomial.

    References:
        [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

    """
    return _has_static(law, "pmf") and not _has_static(law, "pdf")


@dataclass(frozen=True, slots=True, init=False)
class IntegerCodedCategorical:
    """A categorical law coded as the integers ``0, ..., K-1``.

    Labels are coded in the support order of the wrapped
    :class:`~probstat.distributions.Categorical`. The code gives the law a
    CDF, a generalized inverse, and moments so that it can be a model input;
    the forward model receives the code as a float and decodes it with
    :meth:`label`. For unordered labels the mean and variance of the code,
    and any copula dependence placed on it, depend on the arbitrary coding
    order.

    Args:
        categorical: Categorical law to code.

    References:
        [REF-ROSS-2019, Section 4.6], [REF-GENEST-NESLEHOVA-2007].

    """

    categorical: Categorical[Any]
    labels: tuple[Any, ...]
    masses: tuple[float, ...]
    _cumulative: tuple[float, ...]

    def __init__(self, categorical: Categorical[Any]) -> None:
        """Freeze the coding and cumulative masses.

        Args:
            categorical: Categorical law to code.

        Raises:
            TypeError: If ``categorical`` is not a Categorical law.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        if not isinstance(categorical, Categorical):
            raise TypeError("categorical must be a Categorical law")
        labels = tuple(categorical.support.values)
        masses = tuple(float(categorical.pmf(label)) for label in labels)
        cumulative: list[float] = []
        running: list[float] = []
        for mass in masses:
            running.append(mass)
            cumulative.append(min(1.0, math.fsum(running)))
        cumulative[-1] = 1.0
        object.__setattr__(self, "categorical", categorical)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "_cumulative", tuple(cumulative))

    def label(self, code: float) -> Any:
        """Return the label of an integer code.

        Args:
            code: Integer code (an integral float is accepted).

        Returns:
            Label.

        Raises:
            ValueError: If the code is not an integer in range.

        References:
            [REF-ROSS-2019, Section 4.6]. No equation is evaluated.

        """
        value = require_finite_real(code, name="code")
        if not value.is_integer() or not 0 <= value < len(self.labels):
            raise ValueError("code must be an integer between 0 and K - 1")
        return self.labels[int(value)]

    def code(self, label: Any) -> int:
        """Return the integer code of a label.

        Args:
            label: Supported label.

        Returns:
            Code in ``0, ..., K-1``.

        Raises:
            KeyError: If the label is not supported.

        References:
            [REF-ROSS-2019, Section 4.6]. No equation is evaluated.

        """
        for index, item in enumerate(self.labels):
            if item == label:
                return index
        raise KeyError(f"unsupported label {label!r}")

    def pmf(self, k: int) -> float:
        """Return the mass of code ``k`` (zero off the codes).

        Args:
            k: Code.

        Returns:
            Probability mass.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        value = require_finite_real(k, name="k")
        if not value.is_integer() or not 0 <= value < len(self.labels):
            return 0.0
        return self.masses[int(value)]

    def log_pmf(self, k: int) -> float:
        """Return the log-mass of code ``k`` (``-inf`` off the codes).

        Args:
            k: Code.

        Returns:
            Log probability mass.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        mass = self.pmf(k)
        return math.log(mass) if mass > 0.0 else -math.inf

    def cdf(self, x: float) -> float:
        """Return ``P(code <= x)``.

        Args:
            x: Finite real point.

        Returns:
            Cumulative probability.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return 0.0
        index = min(int(math.floor(point)), len(self.labels) - 1)
        return self._cumulative[index]

    def sf(self, x: float) -> float:
        """Return ``P(code > x)`` by summing the upper masses.

        Args:
            x: Finite real point.

        Returns:
            Survival probability.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        point = require_finite_real(x, name="x")
        start = 0 if point < 0.0 else int(math.floor(point)) + 1
        return math.fsum(self.masses[start:])

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse ``min{k : F(k) >= p}``.

        Args:
            probability: Probability in ``[0, 1]``.

        Returns:
            Integral code as a float.

        Raises:
            ValueError: If ``probability`` is outside ``[0, 1]``.

        Equation:
            [EQ-SAMP-INV-001].

        References:
            [REF-DEVROYE-1986, Section III.2].

        """
        target = require_finite_real(probability, name="probability")
        if not 0.0 <= target <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        for index, value in enumerate(self._cumulative):
            if value >= target:
                return float(index)
        return float(len(self.labels) - 1)

    @property
    def mean(self) -> float:
        """Return the mean code.

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        return math.fsum(k * mass for k, mass in enumerate(self.masses))

    @property
    def variance(self) -> float:
        """Return the variance of the code.

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        centre = self.mean
        return math.fsum((k - centre) ** 2 * mass for k, mass in enumerate(self.masses))


def standard_normal_cdf(z: float) -> float:
    """Return ``Phi(z)`` with full relative accuracy in both tails.

    Args:
        z: Finite real argument.

    Returns:
        Standard normal CDF ``0.5 erfc(-z / sqrt 2)``.

    Equation:
        [EQ-DIST-NORM-002].

    References:
        [REF-NIST-DLMF, Equation 7.2.2].

    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def standard_normal_quantile(probability: float) -> float:
    """Return ``Phi^-1(p)`` for ``0 < p < 1``.

    Args:
        probability: Open-interval probability.

    Returns:
        Standard normal quantile.

    Raises:
        ValueError: If ``probability`` is not strictly inside ``(0, 1)``.

    Equation:
        [EQ-DIST-NORM-006].

    References:
        [REF-PYTHON-STATISTICS].

    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return _STANDARD_NORMAL.inv_cdf(probability)


def gauss_hermite(order: int) -> tuple[Vector, Vector]:
    """Return probabilists' Gauss-Hermite nodes and weights by Golub-Welsch.

    The rule integrates ``E[g(Z)]`` for ``Z ~ N(0, 1)`` exactly when ``g`` is a
    polynomial of degree at most ``2 order - 1``.

    Args:
        order: Number of nodes, between 1 and 80.

    Returns:
        Ascending nodes and positive weights summing to one.

    Raises:
        ValueError: If ``order`` is outside the supported range.

    Equation:
        [EQ-MOD-GH-001], [EQ-LINALG-005].

    References:
        [REF-GOLUB-WELSCH-1969], [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 80:
        raise ValueError("order must be an integer between 1 and 80")
    jacobi = tuple(
        tuple(math.sqrt(max(i, j)) if abs(i - j) == 1 else 0.0 for j in range(order))
        for i in range(order)
    )
    values, vectors = symmetric_eigen(jacobi)
    weights = tuple(vectors[0][k] ** 2 for k in range(order))
    total = math.fsum(weights)
    return values, tuple(w / total for w in weights)


def _marginal_from_latent(law: Marginal, z: float) -> float:
    """Return ``F^-1(Phi(z))`` with full accuracy in both tails.

    For ``z > 0`` the upper tail is evaluated as ``S^-1(Phi(-z))`` when a
    continuous law provides an inverse survival function ``isf``; otherwise
    ``Phi(z)`` is clamped below one, which saturates near ``z ~ 8``. Discrete
    laws always use the generalized inverse ``quantile`` (an ``isf`` with a
    different tie convention would give a different count).

    Args:
        law: Marginal law.
        z: Latent standard-normal value.

    Returns:
        Physical value with the same tail probability as ``z``.

    Equation:
        [EQ-MOD-NATAF-001], [EQ-DIST-ISF-001].

    References:
        [REF-LIU-DERKIUREGHIAN-1986].

    """
    inverse_survival = None if is_discrete_law(law) else getattr(law, "isf", None)
    if z > 0.0 and inverse_survival is not None:
        return float(inverse_survival(max(standard_normal_cdf(-z), 1e-300)))
    probability = min(max(standard_normal_cdf(z), 1e-300), 1.0 - 2.0**-53)
    return float(law.quantile(probability))


@dataclass(frozen=True, slots=True)
class RandomInput:
    """One named uncertain model input.

    Args:
        name: Non-empty identifier used as the forward-model parameter key.
        distribution: Continuous marginal law, discrete integer-valued law,
            or :class:`~probstat.distributions.Categorical` (coded
            automatically by :class:`IntegerCodedCategorical`).
        description: Optional physical meaning.
        unit: Optional unit text.
        nature: Aleatory (variability, the default) or epistemic
            (incomplete knowledge) uncertainty; a string value is accepted.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-4], [REF-OBERKAMPF-ROY-2010].

    """

    name: str
    distribution: Marginal
    description: str = ""
    unit: str | None = None
    nature: UncertaintyNature = UncertaintyNature.ALEATORY

    def __post_init__(self) -> None:
        """Validate the name, the structural marginal interface, and the nature.

        Required attributes are looked up on the type (``inspect.getattr_static``)
        so that a property which raises for an undefined moment, such as the
        Cauchy mean, does not reject the law.

        Raises:
            TypeError: If the name is not text, the law lacks a method, or
                the nature is not an :class:`~probstat.uq.UncertaintyNature`.
            ValueError: If the name is empty or the nature text is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if isinstance(self.distribution, Categorical):
            object.__setattr__(
                self, "distribution", IntegerCodedCategorical(self.distribution)
            )
        density = "pmf" if is_discrete_law(self.distribution) else "pdf"
        for attribute in (density, "cdf", "quantile", "mean", "variance"):
            if not _has_static(self.distribution, attribute):
                raise TypeError(
                    f"distribution for {self.name!r} lacks required {attribute!r}"
                )
        nature = self.nature
        if isinstance(nature, str) and not isinstance(nature, UncertaintyNature):
            nature = UncertaintyNature(nature)
        if not isinstance(nature, UncertaintyNature):
            raise TypeError("nature must be an UncertaintyNature")
        object.__setattr__(self, "nature", nature)

    @property
    def is_discrete(self) -> bool:
        """Return whether the marginal law is discrete.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        return is_discrete_law(self.distribution)

    @property
    def is_epistemic(self) -> bool:
        """Return whether the input represents epistemic uncertainty.

        References:
            [REF-OBERKAMPF-ROY-2010]. No equation is evaluated.

        """
        return self.nature is UncertaintyNature.EPISTEMIC

    @property
    def mean(self) -> float:
        """Return the marginal expectation.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return float(self.distribution.mean)

    @property
    def standard_deviation(self) -> float:
        """Return the marginal standard deviation.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return math.sqrt(float(self.distribution.variance))

    def log_pdf(self, x: float) -> float:
        """Return the marginal log-density (log-mass for a discrete law).

        For a discrete law the value is ``log P(X = x)`` (density with
        respect to counting measure), ``-inf`` at non-integers.

        Args:
            x: Finite evaluation point.

        Returns:
            Natural log-density, ``-inf`` outside the support.

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        law: Any = self.distribution
        if self.is_discrete:
            value = float(x)
            if not value.is_integer():
                return -math.inf
            try:
                mass = float(law.pmf(int(value)))
            except ValueError:
                return -math.inf
            return math.log(mass) if mass > 0.0 else -math.inf
        method = getattr(law, "log_pdf", None)
        if method is not None:
            try:
                return float(method(x))
            except ValueError:
                return -math.inf
        density = float(law.pdf(x))
        return math.log(density) if density > 0.0 else -math.inf


@dataclass(frozen=True, slots=True)
class IntervalInput:
    """An epistemic quantity known only to lie in ``[lower, upper]``.

    No probability distribution is implied inside the interval; nested
    propagation explores it with a design (corners plus space-filling
    points) and reports bounds.

    Args:
        name: Non-empty identifier used as the forward-model parameter key.
        lower: Finite lower bound.
        upper: Finite upper bound, not below ``lower``.
        description: Optional meaning.
        unit: Optional unit text.

    References:
        [REF-OBERKAMPF-ROY-2010], [REF-FERSON-ET-AL-2003].

    """

    name: str
    lower: float
    upper: float
    description: str = ""
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the name and bounds.

        Raises:
            TypeError: If the name is not text or a bound is not real.
            ValueError: If the name is empty or ``upper < lower``.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        lower = require_finite_real(self.lower, name="lower")
        upper = require_finite_real(self.upper, name="upper")
        if upper < lower:
            raise ValueError("upper must not be below lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def nature(self) -> UncertaintyNature:
        """Return :attr:`~probstat.uq.UncertaintyNature.EPISTEMIC`.

        References:
            [REF-OBERKAMPF-ROY-2010]. No equation is evaluated.

        """
        return UncertaintyNature.EPISTEMIC

    @property
    def width(self) -> float:
        """Return ``upper - lower``.

        References:
            Project validation policy. No equation is evaluated.

        """
        return self.upper - self.lower

    def from_unit(self, value: float) -> float:
        """Map ``t`` in ``[0, 1]`` linearly onto the interval.

        Args:
            value: Coordinate in ``[0, 1]``.

        Returns:
            ``lower + t (upper - lower)``.

        Raises:
            ValueError: If ``value`` is outside ``[0, 1]``.

        References:
            Project validation policy. No equation is evaluated.

        """
        t = require_finite_real(value, name="value")
        if not 0.0 <= t <= 1.0:
            raise ValueError("value must lie in [0, 1]")
        return self.lower + t * (self.upper - self.lower)


def nataf_correlation(
    first: ContinuousMarginal,
    second: ContinuousMarginal,
    pearson: float,
    *,
    order: int = 24,
) -> float:
    """Return the latent normal correlation reproducing a Pearson correlation.

    Solves ``rho_X = E[h_1(Z_1) h_2(Z_2)]`` for the Gaussian-copula parameter
    ``rho_Z`` by Brent's method, where ``h_k(z) = (F_k^-1(Phi(z)) - mu_k)/sigma_k``
    and the double integral uses a tensor Gauss-Hermite rule.

    Args:
        first: First marginal law.
        second: Second marginal law.
        pearson: Target Pearson correlation of the physical pair.
        order: Gauss-Hermite order per axis.

    Returns:
        Latent correlation ``rho_Z`` in ``(-1, 1)``.

    Raises:
        ValueError: If the target is not attainable for these marginals.

    Equation:
        [EQ-MOD-NATAF-002], [EQ-MOD-GH-001].

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    target = require_finite_real(pearson, name="pearson")
    if not -1.0 < target < 1.0:
        raise ValueError("pearson correlation must lie strictly inside (-1, 1)")
    if target == 0.0:
        return 0.0
    nodes, weights = gauss_hermite(order)
    mu1, sd1 = float(first.mean), math.sqrt(float(first.variance))
    mu2, sd2 = float(second.mean), math.sqrt(float(second.variance))

    def standardized(law: ContinuousMarginal, mu: float, sd: float, z: float) -> float:
        """Map a latent normal value to a standardized physical value.

        References:
            [EQ-MOD-NATAF-001], [REF-LIU-DERKIUREGHIAN-1986].

        """
        probability = min(max(standard_normal_cdf(z), 1e-300), 1.0 - 2.0**-53)
        return (float(law.quantile(probability)) - mu) / sd

    first_values = [standardized(first, mu1, sd1, z) for z in nodes]

    def implied(rho: float) -> float:
        """Return the physical correlation implied by latent ``rho`` minus target.

        References:
            [EQ-MOD-NATAF-002], [REF-LIU-DERKIUREGHIAN-1986].

        """
        root = math.sqrt(1.0 - rho * rho)
        total = math.fsum(
            weights[i]
            * weights[j]
            * first_values[i]
            * standardized(second, mu2, sd2, rho * nodes[i] + root * nodes[j])
            for i in range(order)
            for j in range(order)
        )
        return total - target

    limit = 0.9999
    low, high = implied(-limit), implied(limit)
    if low * high > 0.0:
        raise ValueError(
            "target Pearson correlation is not attainable for these marginals"
        )
    return brent_root(implied, -limit, limit, tolerance=1e-10).root


def _uniform_from_discrete(
    law: Any, value: float, jitter: float, *, name: str
) -> tuple[float, float]:
    """Return the distributional transform ``F(x-) + V P(X = x)`` and its complement.

    The complement ``S(x) + (1 - V) P(X = x)`` is returned as well so that
    upper-tail normal scores keep their accuracy.

    Raises:
        ValueError: If ``value`` is not an integer with positive mass.

    Equation:
        [EQ-MOD-DISC-001].

    References:
        [REF-RUSCHENDORF-2009].

    """
    point = float(value)
    mass = float(law.pmf(int(point))) if point.is_integer() else 0.0
    if not mass > 0.0:
        raise ValueError(f"{name}={value} has zero probability under its law")
    below = float(law.cdf(point - 1.0))
    survival_method = getattr(law, "sf", None)
    above = (
        float(survival_method(point))
        if survival_method is not None
        else max(0.0, 1.0 - float(law.cdf(point)))
    )
    lower_value = below + jitter * mass
    upper_value = above + (1.0 - jitter) * mass
    return min(max(lower_value, 1e-300), 1.0), min(max(upper_value, 1e-300), 1.0)


def _validated_names(inputs: tuple[RandomInput, ...]) -> None:
    """Reject non-:class:`RandomInput` items and repeated names.

    Raises:
        TypeError: If an input is not a :class:`RandomInput`.
        ValueError: If there are no inputs or names repeat.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not inputs:
        raise ValueError("at least one input is required")
    for index, item in enumerate(inputs):
        if not isinstance(item, RandomInput):
            raise TypeError(f"inputs[{index}] must be a RandomInput")
    names = [item.name for item in inputs]
    if len(set(names)) != len(names):
        raise ValueError("input names must be distinct")


def _latent_matrix_or_repair(latent: Matrix, *, repair: bool, source: str) -> Matrix:
    """Return a positive-definite latent matrix, repairing it when allowed.

    Pairwise solves can produce a matrix that is not positive definite.
    With ``repair=True`` it is replaced by the nearest correlation matrix
    (Higham's alternating projections, eigenvalues at least ``1e-8``) and a
    :class:`UserWarning` reports the Frobenius distance; otherwise the
    failure is raised.

    Raises:
        ValueError: If the matrix is not positive definite and ``repair`` is
            false.

    Equation:
        [EQ-MOD-COP-NCM-001].

    References:
        [REF-HIGHAM-2002].

    """
    try:
        cholesky(latent)
        return latent
    except ValueError as error:
        if not repair:
            raise ValueError(
                f"latent correlation implied by the {source} targets is not "
                "positive definite"
            ) from error
    result = nearest_correlation_matrix(latent)
    largest = max(
        abs(result.matrix[i][j] - latent[i][j])
        for i in range(len(latent))
        for j in range(len(latent))
    )
    warnings.warn(
        f"latent correlation implied by the {source} targets was not positive "
        f"definite; replaced by the nearest correlation matrix (Frobenius "
        f"distance {result.distance:.3g}, largest entry change {largest:.3g}, "
        f"{result.iterations} iterations)",
        UserWarning,
        stacklevel=3,
    )
    return result.matrix


@dataclass(frozen=True)
class JointInput:
    """Joint law of named inputs: marginals plus a copula.

    ``JointInput(inputs, correlation)`` is the Gaussian copula (Nataf model)
    with latent correlation ``R``; ``JointInput(inputs)`` is independence;
    ``JointInput(inputs, copula=ClaytonCopula(2.0, 3))`` uses any copula from
    :mod:`probstat.modeling.copulas`. For non-Gaussian copulas the standard
    space is defined by the Rosenblatt transform in input order, so FORM
    results can depend on that order (any order gives the same exact law).

    Args:
        inputs: Distinctly named random inputs in model order.
        correlation: Optional latent-normal correlation matrix ``R`` (unit
            diagonal, symmetric positive definite). ``None`` means independence
            unless ``copula`` is given.
        copula: Optional copula of matching dimension; give either
            ``correlation`` or ``copula``. A :class:`GaussianCopula` is stored
            as ``correlation`` and an :class:`IndependenceCopula` as
            ``correlation=None``, so this field is ``None`` unless the copula
            is non-Gaussian and :func:`dataclasses.replace` keeps working on
            ``correlation``. :attr:`dependence` always returns the copula in
            use.

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-NELSEN-2006, Chapter 2],
        [REF-LEBRUN-DUTFOY-2009].

    """

    inputs: tuple[RandomInput, ...]
    correlation: Matrix | None = None
    copula: Copula | None = None
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _precision_minus_identity: Matrix | None = field(
        init=False, repr=False, compare=False
    )
    _log_det: float = field(init=False, repr=False, compare=False)
    _rosenblatt: bool = field(init=False, repr=False, compare=False)
    _discrete: tuple[bool, ...] = field(init=False, repr=False, compare=False)
    _copula: Copula = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate names and the copula; cache the Nataf factors.

        Raises:
            TypeError: If an input is not a :class:`RandomInput` or the copula
                is not a :class:`~probstat.modeling.copulas.Copula`.
            ValueError: If names repeat, both ``correlation`` and ``copula``
                are given, or the dependence is invalid.

        Equation:
            [EQ-LINALG-001], [EQ-MOD-COPULA-001].

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        inputs = tuple(self.inputs)
        _validated_names(inputs)
        object.__setattr__(self, "inputs", inputs)
        discrete = tuple(item.is_discrete for item in inputs)
        object.__setattr__(self, "_discrete", discrete)
        object.__setattr__(self, "_rosenblatt", False)
        size = len(inputs)
        given = self.copula
        if given is not None:
            if not isinstance(given, Copula):
                raise TypeError("copula must be a Copula")
            if self.correlation is not None:
                raise ValueError("give either correlation or copula, not both")
            if given.dimension != size:
                raise ValueError("copula dimension must equal the number of inputs")
            if isinstance(given, GaussianCopula | IndependenceCopula):
                object.__setattr__(self, "copula", None)
                if isinstance(given, GaussianCopula):
                    object.__setattr__(self, "correlation", given.correlation)
            else:
                object.__setattr__(self, "_rosenblatt", True)
                object.__setattr__(self, "_copula", given)
                object.__setattr__(self, "_lower", identity(size))
                object.__setattr__(self, "_precision_minus_identity", None)
                object.__setattr__(self, "_log_det", 0.0)
                return
        if self.correlation is None:
            object.__setattr__(self, "_lower", identity(size))
            object.__setattr__(self, "_precision_minus_identity", None)
            object.__setattr__(self, "_log_det", 0.0)
            object.__setattr__(self, "_copula", IndependenceCopula(size))
            return
        matrix = symmetrize(self.correlation, name="correlation")
        if len(matrix) != size:
            raise ValueError("correlation dimension must equal the number of inputs")
        for i in range(size):
            if abs(matrix[i][i] - 1.0) > 1e-12:
                raise ValueError("correlation matrix must have a unit diagonal")
            for j in range(size):
                if abs(matrix[i][j]) > 1.0:
                    raise ValueError("correlation entries must lie in [-1, 1]")
        try:
            lower = cholesky(matrix)
        except ValueError as error:
            raise ValueError("correlation must be positive definite") from error
        precision = inverse_spd(matrix)
        object.__setattr__(self, "correlation", matrix)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(
            self,
            "_precision_minus_identity",
            tuple(
                tuple(precision[i][j] - (1.0 if i == j else 0.0) for j in range(size))
                for i in range(size)
            ),
        )
        object.__setattr__(self, "_log_det", log_det_spd(matrix))
        object.__setattr__(self, "_copula", GaussianCopula(matrix))

    @classmethod
    def independent(cls, *inputs: RandomInput) -> JointInput:
        """Build a joint law of mutually independent inputs.

        Args:
            *inputs: Random inputs in model order.

        Returns:
            Joint input with identity copula correlation.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        return cls(tuple(inputs))

    @classmethod
    def from_pearson(
        cls,
        inputs: Sequence[RandomInput],
        pearson: Iterable[Iterable[object]],
        *,
        order: int = 24,
        repair: bool = True,
    ) -> JointInput:
        """Build a Nataf joint law that matches target Pearson correlations.

        Each latent correlation is solved pairwise; the pairwise solutions
        need not form a positive-definite matrix. With ``repair=True`` such a
        matrix is replaced by the nearest correlation matrix and a
        :class:`UserWarning` reports the change (the repaired law then
        matches the targets only approximately).

        Args:
            inputs: Continuous random inputs in model order.
            pearson: Target physical Pearson correlation matrix.
            order: Gauss-Hermite order for each pairwise solve.
            repair: Whether to repair a non-positive-definite latent matrix.

        Returns:
            Joint input whose latent correlation reproduces the targets.

        Raises:
            ValueError: If a target is unattainable, an input is discrete
                (the Gauss-Hermite rule cannot integrate a step function
                accurately), or the latent matrix is not positive definite
                and ``repair`` is false.

        Equation:
            [EQ-MOD-NATAF-002], [EQ-MOD-COP-NCM-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-HIGHAM-2002].

        """
        materialized = tuple(inputs)
        _validated_names(materialized)
        if any(item.is_discrete for item in materialized):
            raise ValueError(
                "from_pearson needs continuous marginals; use from_spearman or "
                "from_kendall to specify copula rank correlations"
            )
        target = symmetrize(pearson, name="pearson")
        size = len(materialized)
        if len(target) != size:
            raise ValueError("pearson dimension must equal the number of inputs")
        latent = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                value = nataf_correlation(
                    _continuous(materialized[i].distribution),
                    _continuous(materialized[j].distribution),
                    target[i][j],
                    order=order,
                )
                latent[i][j] = latent[j][i] = value
        matrix = _latent_matrix_or_repair(
            tuple(tuple(row) for row in latent), repair=repair, source="Pearson"
        )
        return cls(materialized, matrix)

    @classmethod
    def from_spearman(
        cls,
        inputs: Sequence[RandomInput],
        spearman: Iterable[Iterable[object]],
        *,
        repair: bool = True,
    ) -> JointInput:
        """Build a Gaussian-copula joint law from target Spearman correlations.

        For the Gaussian copula ``rho_S = (6/pi) arcsin(rho/2)``, inverted
        entrywise as ``rho = 2 sin(pi rho_S / 6)``. Rank correlations are
        properties of the copula, so the targets hold for any continuous
        marginals; for discrete marginals they are the copula's (latent)
        rank correlations, not those of the observed counts.

        Args:
            inputs: Random inputs in model order.
            spearman: Target Spearman correlation matrix.
            repair: Whether to repair a non-positive-definite latent matrix
                (reported with a :class:`UserWarning`).

        Returns:
            Gaussian-copula joint input.

        Raises:
            ValueError: If an entry is outside ``[-1, 1]`` or the latent matrix
                is not positive definite and ``repair`` is false.

        Equation:
            [EQ-MOD-COPULA-FIT-001], [EQ-MOD-COP-NCM-001].

        References:
            [REF-KRUSKAL-1958], [REF-HIGHAM-2002].

        """
        return cls._from_rank_targets(
            inputs, spearman, _spearman_to_latent, repair=repair, source="Spearman"
        )

    @classmethod
    def from_kendall(
        cls,
        inputs: Sequence[RandomInput],
        kendall: Iterable[Iterable[object]],
        *,
        repair: bool = True,
    ) -> JointInput:
        """Build a Gaussian-copula joint law from target Kendall correlations.

        For every elliptical copula ``tau = (2/pi) arcsin(rho)``, inverted
        entrywise as ``rho = sin(pi tau / 2)``.

        Args:
            inputs: Random inputs in model order.
            kendall: Target Kendall's tau matrix.
            repair: Whether to repair a non-positive-definite latent matrix
                (reported with a :class:`UserWarning`).

        Returns:
            Gaussian-copula joint input.

        Raises:
            ValueError: If an entry is outside ``[-1, 1]`` or the latent matrix
                is not positive definite and ``repair`` is false.

        Equation:
            [EQ-MOD-COPULA-FIT-005], [EQ-MOD-COP-NCM-001].

        References:
            [REF-KRUSKAL-1958], [REF-MCNEIL-FREY-EMBRECHTS-2015],
            [REF-HIGHAM-2002].

        """
        return cls._from_rank_targets(
            inputs, kendall, _kendall_to_latent, repair=repair, source="Kendall"
        )

    @classmethod
    def _from_rank_targets(
        cls,
        inputs: Sequence[RandomInput],
        targets: Iterable[Iterable[object]],
        mapping: Callable[[float], float],
        *,
        repair: bool,
        source: str,
    ) -> JointInput:
        """Map a rank-correlation matrix entrywise to a latent correlation.

        Raises:
            ValueError: If dimensions disagree, the diagonal is not one, an
                entry is outside ``[-1, 1]``, or repair is refused.

        Equation:
            [EQ-MOD-COPULA-FIT-001], [EQ-MOD-COPULA-FIT-005].

        References:
            [REF-KRUSKAL-1958], [REF-HIGHAM-2002].

        """
        materialized = tuple(inputs)
        _validated_names(materialized)
        target = symmetrize(targets, name=source.lower())
        size = len(materialized)
        if len(target) != size:
            raise ValueError(f"{source} matrix dimension must equal the inputs")
        latent: list[list[float]] = []
        for i in range(size):
            if abs(target[i][i] - 1.0) > 1e-12:
                raise ValueError(f"{source} matrix must have a unit diagonal")
            row: list[float] = []
            for j in range(size):
                if abs(target[i][j]) > 1.0:
                    raise ValueError(f"{source} entries must lie in [-1, 1]")
                row.append(1.0 if i == j else mapping(target[i][j]))
            latent.append(row)
        matrix = _latent_matrix_or_repair(
            tuple(tuple(row) for row in latent), repair=repair, source=source
        )
        return cls(materialized, matrix)

    @property
    def names(self) -> tuple[str, ...]:
        """Return input names in model order.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return tuple(item.name for item in self.inputs)

    @property
    def dimension(self) -> int:
        """Return the number of inputs.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return len(self.inputs)

    @property
    def is_independent(self) -> bool:
        """Return whether the copula is the independence copula.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        return self.correlation is None and not self._rosenblatt

    @property
    def copula_family(self) -> str:
        """Return the copula family name (``"independence"``, ``"gaussian"``, ...).

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return self._copula.family

    @property
    def dependence(self) -> Copula:
        """Return the copula in use (independence, Gaussian, or the given one).

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return self._copula

    @property
    def uses_rosenblatt(self) -> bool:
        """Return whether the standard space is defined by the Rosenblatt map.

        ``False`` for independence and the Gaussian copula (Nataf map), whose
        FORM results do not depend on the input order.

        References:
            [REF-LEBRUN-DUTFOY-2009]. No equation is evaluated.

        """
        return self._rosenblatt

    @property
    def has_discrete(self) -> bool:
        """Return whether any marginal law is discrete.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        return any(self._discrete)

    @property
    def discrete_names(self) -> tuple[str, ...]:
        """Return the names of discrete inputs in model order.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        return tuple(
            item.name
            for item, flag in zip(self.inputs, self._discrete, strict=True)
            if flag
        )

    def require_continuous(self, method: str = "FORM") -> None:
        """Raise when a gradient-based method meets a discrete input.

        FORM, SORM, and inverse FORM differentiate ``G(u) = g(T^-1(u))``. For a
        discrete input ``T^-1`` is a step function, the gradient is zero
        almost everywhere and the design point is not defined, so these
        methods would return a wrong answer without warning.

        Args:
            method: Method name used in the error message.

        Raises:
            ValueError: If any input is discrete.

        References:
            [REF-DERKIUREGHIAN-2005]. No equation is evaluated.

        """
        if self.has_discrete:
            raise ValueError(
                f"{method} needs continuous inputs; discrete inputs "
                f"{list(self.discrete_names)} make the standard-space limit state "
                "piecewise constant. Use a sampling method (Monte Carlo, subset "
                "simulation, importance sampling) instead."
            )

    def subset(self, names: Sequence[str]) -> JointInput:
        """Return the exact joint law of a subset of the inputs.

        Margins of every supported copula stay in the same family (sub-matrix
        of ``R`` for the Gaussian and t copulas, the same parameter for the
        exchangeable Archimedean copulas).

        Args:
            names: Distinct input names, in the order wanted.

        Returns:
            Joint input of the selected names.

        Raises:
            KeyError: If a name is unknown.
            ValueError: If names repeat or are empty.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        chosen = tuple(names)
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError("names must be a non-empty set of distinct names")
        lookup = {name: index for index, name in enumerate(self.names)}
        for name in chosen:
            if name not in lookup:
                raise KeyError(f"unknown input {name!r}")
        indices = tuple(lookup[name] for name in chosen)
        inputs = tuple(self.inputs[index] for index in indices)
        if self.is_independent:
            return JointInput(inputs)
        if self.correlation is not None:
            return JointInput(
                inputs,
                tuple(tuple(self.correlation[a][b] for b in indices) for a in indices),
            )
        return JointInput(inputs, copula=self._copula.marginal(indices))

    def means(self) -> Vector:
        """Return marginal means in model order.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return tuple(item.mean for item in self.inputs)

    def standard_deviations(self) -> Vector:
        """Return marginal standard deviations in model order.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return tuple(item.standard_deviation for item in self.inputs)

    def as_mapping(self, x: Sequence[float]) -> Mapping[str, float]:
        """Label a physical input vector with input names.

        Args:
            x: Physical vector in model order.

        Returns:
            Name-to-value mapping.

        Raises:
            ValueError: If the length differs from the dimension.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        return dict(zip(self.names, vector, strict=True))

    def as_vector(self, values: Mapping[str, float]) -> Vector:
        """Order a name-to-value mapping into a physical vector.

        Args:
            values: Mapping containing every input name.

        Returns:
            Physical vector in model order.

        Raises:
            KeyError: If an input name is missing.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return tuple(
            require_finite_real(values[name], name=name) for name in self.names
        )

    def _jitter_source(self, seed: int | RandomStream | None) -> random.Random | None:
        """Return the generator for distributional-transform jitter, if any.

        References:
            [REF-RUSCHENDORF-2009]. No equation is evaluated.

        """
        if seed is None or not self.has_discrete:
            return None
        return resolve_stream(seed).child("distributional-transform").generator()

    def _unit_values(
        self, vector: Vector, seed: int | RandomStream | None
    ) -> tuple[tuple[float, float], ...]:
        """Return ``(F_i(x_i), 1 - F_i(x_i))`` pairs, discrete ones transformed.

        Raises:
            ValueError: If a value lies outside the open support.

        Equation:
            [EQ-MOD-COP-SKLAR-001], [EQ-MOD-DISC-001].

        References:
            [REF-RUSCHENDORF-2009], [REF-NELSEN-2006, Section 2.3].

        """
        jitter = self._jitter_source(seed)
        result: list[tuple[float, float]] = []
        for item, value, discrete in zip(
            self.inputs, vector, self._discrete, strict=True
        ):
            law: Any = item.distribution
            if discrete:
                v = 0.5 if jitter is None else jitter.random()
                result.append(_uniform_from_discrete(law, value, v, name=item.name))
                continue
            probability = float(law.cdf(value))
            survival_method = getattr(law, "sf", None)
            upper = (
                float(survival_method(value))
                if survival_method is not None
                else 1.0 - probability
            )
            if not 0.0 < probability < 1.0 or not 0.0 < upper < 1.0:
                raise ValueError(
                    f"{item.name}={value} lies outside the open support of its law"
                )
            result.append((probability, upper))
        return tuple(result)

    def to_latent(
        self, x: Sequence[float], *, seed: int | RandomStream | None = None
    ) -> Vector:
        """Map physical inputs to marginal normal scores ``z``.

        For a discrete input the distributional transform
        ``F(x-) + V P(X = x)`` replaces ``F(x)``; ``V = 1/2`` (mid-point)
        when ``seed`` is ``None``, otherwise ``V ~ Uniform(0, 1)`` from the
        seeded stream. The latent value is then one representative of an
        interval of latent values that all map back to ``x``.

        Args:
            x: Physical vector strictly inside every marginal support.
            seed: Optional seed for the discrete-input randomization.

        Returns:
            ``z_i = Phi^-1(F_i(x_i))``.

        Raises:
            ValueError: If a CDF value is 0 or 1 (outside the open support).

        Equation:
            [EQ-MOD-NATAF-001], [EQ-MOD-DISC-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-RUSCHENDORF-2009].

        """
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        jitter = self._jitter_source(seed)
        latent: list[float] = []
        for item, value, discrete in zip(
            self.inputs, vector, self._discrete, strict=True
        ):
            law: Any = item.distribution
            if discrete:
                v = 0.5 if jitter is None else jitter.random()
                lower_tail, upper_tail = _uniform_from_discrete(
                    law, value, v, name=item.name
                )
                if lower_tail > 0.5:
                    latent.append(-standard_normal_quantile(min(upper_tail, 0.5)))
                else:
                    latent.append(standard_normal_quantile(lower_tail))
                continue
            probability = float(law.cdf(value))
            survival_method = getattr(law, "sf", None)
            if probability > 0.5 and survival_method is not None:
                # Upper tail: use S(x) directly; 1 - F(x) would round to 0.
                upper = float(survival_method(value))
                if not 0.0 < upper < 1.0:
                    raise ValueError(
                        f"{item.name}={value} lies outside the open support of its law"
                    )
                latent.append(-standard_normal_quantile(upper))
                continue
            if not 0.0 < probability < 1.0:
                raise ValueError(
                    f"{item.name}={value} lies outside the open support of its law"
                )
            latent.append(standard_normal_quantile(probability))
        return tuple(latent)

    def to_standard_normal(
        self, x: Sequence[float], *, seed: int | RandomStream | None = None
    ) -> Vector:
        """Map physical inputs to independent standard normals ``u``.

        Gaussian copula: ``u = L^-1 z`` (Nataf). Other copulas:
        ``u_k = Phi^-1(w_k)`` with the Rosenblatt transform ``w`` of
        ``F_i(x_i)``; ``w`` is clamped into ``[1e-300, 1 - 2^-53]``, so
        ``u_k`` saturates near 8.2 on the upper side. Discrete inputs use the
        distributional transform (see :meth:`to_latent`), which is not
        one-to-one: the result is one of many ``u`` that map back to ``x``.

        Args:
            x: Physical vector.
            seed: Optional seed for the discrete-input randomization.

        Returns:
            Independent standard-normal coordinates.

        Equation:
            [EQ-MOD-NATAF-001], [EQ-LINALG-002], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-MELCHERS-BECK-2018, Chapter 5],
            [REF-ROSENBLATT-1952], [REF-HOHENBICHLER-RACKWITZ-1981].

        """
        if not self._rosenblatt:
            latent = self.to_latent(x, seed=seed)
            if self.is_independent:
                return latent
            return solve_lower(self._lower, latent)
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        unit = tuple(
            min(max(lower, 1e-300), 1.0 - 2.0**-53)
            for lower, _ in self._unit_values(vector, seed)
        )
        return tuple(
            standard_normal_quantile(min(max(w, 1e-300), 1.0 - 2.0**-53))
            for w in self._copula.rosenblatt(unit)
        )

    def _from_copula_point(self, point: Sequence[float]) -> Vector:
        """Map a copula point to physical inputs by generalized inverses.

        Equation:
            [EQ-SAMP-INV-001].

        References:
            [REF-DEVROYE-1986, Section II.2].

        """
        return tuple(
            float(item.distribution.quantile(min(max(value, 1e-300), 1.0 - 2.0**-53)))
            for item, value in zip(self.inputs, point, strict=True)
        )

    def from_standard_normal(self, u: Sequence[float]) -> Vector:
        """Map independent standard normals ``u`` to physical inputs.

        Args:
            u: Finite standard-normal vector.

        Returns:
            ``x_i = F_i^-1(Phi(z_i))`` with ``z = L u`` (Gaussian copula) or
            ``x = F^-1(C^-1_Rosenblatt(Phi(u)))`` (other copulas). Discrete
            inputs receive their generalized inverse, which is exact.

        Raises:
            ValueError: If ``u`` has the wrong length.

        Equation:
            [EQ-MOD-NATAF-001], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-ROSENBLATT-1952].

        """
        vector = as_vector(u, name="u")
        if len(vector) != self.dimension:
            raise ValueError("u length must equal the dimension")
        if self._rosenblatt:
            return self._from_copula_point(
                self._copula.inverse_rosenblatt(
                    tuple(
                        min(max(standard_normal_cdf(value), 1e-300), 1.0 - 2.0**-53)
                        for value in vector
                    )
                )
            )
        latent = vector if self.is_independent else matvec(self._lower, vector)
        return tuple(
            _marginal_from_latent(item.distribution, z)
            for item, z in zip(self.inputs, latent, strict=True)
        )

    def from_unit(self, point: Sequence[float]) -> Vector:
        """Map a unit-hypercube design point to a physical input vector.

        Under independence each coordinate goes through its own inverse CDF,
        which preserves Latin-hypercube and low-discrepancy stratification.
        Under a Gaussian copula the point is first mapped to ``u`` by the
        normal quantile, then through :meth:`from_standard_normal`. Under any
        other copula the point is mapped by the inverse Rosenblatt transform
        and the marginal generalized inverses, so
        ``from_unit(p) = from_standard_normal(Phi^-1(p))`` for every copula.

        Args:
            point: Point in ``(0, 1)^d``.

        Returns:
            Physical input vector.

        Raises:
            ValueError: If a coordinate is outside ``(0, 1)``.

        Equation:
            [EQ-MOD-NATAF-001], [EQ-SAMP-INV-001], [EQ-MOD-COP-ROSEN-001].

        References:
            [REF-DEVROYE-1986, Section II.2], [REF-LIU-DERKIUREGHIAN-1986],
            [REF-ROSENBLATT-1952].

        """
        coordinates = as_vector(point, name="point")
        if len(coordinates) != self.dimension:
            raise ValueError("design point length must equal the dimension")
        if any(not 0.0 < value < 1.0 for value in coordinates):
            raise ValueError("design coordinates must lie strictly inside (0, 1)")
        if self.is_independent:
            return tuple(
                float(item.distribution.quantile(value))
                for item, value in zip(self.inputs, coordinates, strict=True)
            )
        if self._rosenblatt:
            return self._from_copula_point(self._copula.inverse_rosenblatt(coordinates))
        return self.from_standard_normal(
            tuple(standard_normal_quantile(value) for value in coordinates)
        )

    def sample_design(self, design: SamplingDesign) -> tuple[Vector, ...]:
        """Map every point of a unit-hypercube design to physical inputs.

        Args:
            design: Design with this joint law's dimension.

        Returns:
            Physical input vectors in design order.

        Raises:
            ValueError: If dimensions disagree.

        Equation:
            [EQ-SAMP-INV-001].

        References:
            [REF-OWEN-2013, Chapter 10].

        """
        if design.dimension != self.dimension:
            raise ValueError("design dimension must equal the input dimension")
        return tuple(self.from_unit(point) for point in design.points)

    def sample(
        self, size: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw independent joint samples by plain Monte Carlo.

        Args:
            size: Positive number of draws.
            seed: Integer seed or stream.

        Returns:
            Physical input vectors.

        Equation:
            [EQ-SAMP-INV-001], [EQ-MOD-NATAF-001].

        References:
            [REF-DEVROYE-1986, Section II.2].

        """
        stream = resolve_stream(seed)
        return self.sample_design(monte_carlo_design(size, self.dimension, seed=stream))

    def _discrete_log_pmf(self, vector: Vector) -> float:
        """Return the joint log-mass or mixed log-density with discrete inputs.

        Independence: sum of marginal log-densities and log-masses. All
        inputs discrete and a copula CDF available: the rectangle
        inclusion-exclusion
        ``P(X = x) = sum_{e in {0,1}^d} (-1)^{|e|} C(F(x - e))``.
        Mixed discrete and continuous inputs under a dependent copula need
        partial derivatives of ``C`` and are not implemented.

        Raises:
            NotImplementedError: For mixed inputs under a dependent copula,
                more than 16 discrete inputs, or a copula without a CDF.

        Equation:
            [EQ-MOD-DISC-002].

        References:
            [REF-GENEST-NESLEHOVA-2007], [REF-NELSEN-2006, Section 2.10].

        """
        if self.is_independent:
            return math.fsum(
                item.log_pdf(value)
                for item, value in zip(self.inputs, vector, strict=True)
            )
        if not all(self._discrete):
            raise NotImplementedError(
                "the joint density of mixed discrete and continuous inputs under a "
                "dependent copula is not implemented"
            )
        if self.dimension > 16:
            raise NotImplementedError(
                "inclusion-exclusion is limited to 16 discrete inputs"
            )
        upper: list[float] = []
        lower: list[float] = []
        for item, value in zip(self.inputs, vector, strict=True):
            if not math.isfinite(item.log_pdf(value)):
                return -math.inf
            law: Any = item.distribution
            upper.append(float(law.cdf(value)))
            lower.append(float(law.cdf(value - 1.0)))
        terms: list[float] = []
        for mask in range(1 << self.dimension):
            corner = tuple(
                lower[i] if mask >> i & 1 else upper[i] for i in range(self.dimension)
            )
            sign = -1.0 if bin(mask).count("1") % 2 else 1.0
            terms.append(sign * self._copula.cdf(corner))
        mass = math.fsum(terms)
        return math.log(mass) if mass > 0.0 else -math.inf

    def log_pdf(self, x: Sequence[float]) -> float:
        """Return the joint log-density of physical inputs.

        Gaussian copula:
        ``log f(x) = sum_i log f_i(x_i) - 0.5 z'(R^-1 - I) z - 0.5 log det R``.
        Other copulas: ``sum_i log f_i(x_i) + log c(F_1(x_1), ..., F_d(x_d))``.
        With discrete inputs: independence gives the mixed density/mass;
        all-discrete inputs give the exact log-mass by inclusion-exclusion
        over the copula CDF (Gaussian copula: two inputs at most); mixed
        inputs under a dependent copula raise :class:`NotImplementedError`.

        Args:
            x: Physical vector.

        Returns:
            Joint log-density; ``-inf`` outside the support.

        Raises:
            NotImplementedError: For mixed discrete/continuous inputs under a
                dependent copula, or a copula without a usable CDF.

        Equation:
            [EQ-MOD-COPULA-001], [EQ-MOD-COP-SKLAR-001], [EQ-MOD-DISC-002].

        References:
            [REF-NELSEN-2006, Section 2.3], [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        if any(self._discrete):
            return self._discrete_log_pmf(vector)
        marginal = math.fsum(
            item.log_pdf(value) for item, value in zip(self.inputs, vector, strict=True)
        )
        if not math.isfinite(marginal) or self.is_independent:
            return marginal
        if self._rosenblatt:
            try:
                unit = tuple(lower for lower, _ in self._unit_values(vector, None))
            except ValueError:
                return -math.inf
            return marginal + self._copula.log_density(unit)
        try:
            latent = self.to_latent(vector)
        except ValueError:
            return -math.inf
        assert self._precision_minus_identity is not None
        quadratic = dot(latent, matvec(self._precision_minus_identity, latent))
        return marginal - 0.5 * quadratic - 0.5 * self._log_det

    def standard_normal_log_pdf(self, u: Sequence[float]) -> float:
        """Return the independent standard-normal log-density of ``u``.

        Args:
            u: Finite vector.

        Returns:
            ``-0.5 ||u||^2 - d log sqrt(2 pi)``.

        Equation:
            [EQ-DIST-MVN-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        vector = as_vector(u, name="u")
        return -0.5 * dot(vector, vector) - len(vector) * _LOG_SQRT_TWO_PI


def _spearman_to_latent(value: float) -> float:
    """Return ``2 sin(pi rho_S / 6)``.

    Equation:
        [EQ-MOD-COPULA-FIT-001].

    References:
        [REF-KRUSKAL-1958].

    """
    return 2.0 * math.sin(math.pi * value / 6.0)


def _kendall_to_latent(value: float) -> float:
    """Return ``sin(pi tau / 2)``.

    Equation:
        [EQ-MOD-COPULA-FIT-005].

    References:
        [REF-KRUSKAL-1958], [REF-MCNEIL-FREY-EMBRECHTS-2015].

    """
    return math.sin(math.pi * value / 2.0)


def _continuous(law: Marginal) -> ContinuousMarginal:
    """Narrow a marginal law to the continuous protocol.

    Raises:
        TypeError: If the law is discrete.

    References:
        Project validation policy. No equation is evaluated.

    """
    if is_discrete_law(law):
        raise TypeError("a continuous marginal law is required")
    return law  # type: ignore[return-value]
