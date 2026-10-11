"""Named uncertain inputs and their joint law through a Gaussian copula.

A :class:`JointInput` is the single description of input uncertainty shared by
propagation, sensitivity, reliability, calibration priors, and risk analysis.
Marginals are any continuous laws with ``cdf``, ``quantile``, ``pdf``,
``mean``, and ``variance``. Dependence is a Gaussian copula with latent normal
correlation ``R``; independence is ``R = I``.

The isoprobabilistic map between physical inputs ``x`` and independent
standard-normal variables ``u`` is the Nataf transformation::

    z_i = Phi^-1(F_i(x_i)),   u = L^-1 z,   R = L L'

Reliability methods work in ``u``; samplers map unit-hypercube designs
through the inverse of the same map, so every workflow sees one consistent law.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Protocol, runtime_checkable

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


@dataclass(frozen=True, slots=True)
class RandomInput:
    """One named uncertain model input.

    Args:
        name: Non-empty identifier used as the forward-model parameter key.
        distribution: Continuous marginal law.
        description: Optional physical meaning.
        unit: Optional unit text.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-4].

    """

    name: str
    distribution: ContinuousMarginal
    description: str = ""
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the name and the structural marginal interface.

        Raises:
            TypeError: If the name is not text or the law lacks a method.
            ValueError: If the name is empty.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for attribute in ("pdf", "cdf", "quantile", "mean", "variance"):
            if not hasattr(self.distribution, attribute):
                raise TypeError(
                    f"distribution for {self.name!r} lacks required {attribute!r}"
                )

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
        """Return the marginal log-density, ``-inf`` outside the support.

        Args:
            x: Finite evaluation point.

        Returns:
            Natural log-density.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        method = getattr(self.distribution, "log_pdf", None)
        if method is not None:
            try:
                return float(method(x))
            except ValueError:
                return -math.inf
        density = float(self.distribution.pdf(x))
        return math.log(density) if density > 0.0 else -math.inf


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


@dataclass(frozen=True)
class JointInput:
    """Joint law of named inputs: marginals plus a Gaussian copula.

    Args:
        inputs: Distinctly named random inputs in model order.
        correlation: Optional latent-normal correlation matrix ``R`` (unit
            diagonal, symmetric positive definite). ``None`` means independence.

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-NELSEN-2006, Chapter 2].

    """

    inputs: tuple[RandomInput, ...]
    correlation: Matrix | None = None
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _precision_minus_identity: Matrix | None = field(
        init=False, repr=False, compare=False
    )
    _log_det: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate names and the copula correlation; cache its factors.

        Raises:
            TypeError: If an input is not a :class:`RandomInput`.
            ValueError: If names repeat or the correlation is invalid.

        Equation:
            [EQ-LINALG-001], [EQ-MOD-COPULA-001].

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("at least one input is required")
        for index, item in enumerate(inputs):
            if not isinstance(item, RandomInput):
                raise TypeError(f"inputs[{index}] must be a RandomInput")
        names = [item.name for item in inputs]
        if len(set(names)) != len(names):
            raise ValueError("input names must be distinct")
        object.__setattr__(self, "inputs", inputs)
        size = len(inputs)
        if self.correlation is None:
            object.__setattr__(self, "_lower", identity(size))
            object.__setattr__(self, "_precision_minus_identity", None)
            object.__setattr__(self, "_log_det", 0.0)
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
    ) -> JointInput:
        """Build a Nataf joint law that matches target Pearson correlations.

        Args:
            inputs: Random inputs in model order.
            pearson: Target physical Pearson correlation matrix.
            order: Gauss-Hermite order for each pairwise solve.

        Returns:
            Joint input whose latent correlation reproduces the targets.

        Raises:
            ValueError: If a target is unattainable or the resulting latent
                matrix is not positive definite.

        Equation:
            [EQ-MOD-NATAF-002].

        References:
            [REF-LIU-DERKIUREGHIAN-1986].

        """
        materialized = tuple(inputs)
        target = symmetrize(pearson, name="pearson")
        size = len(materialized)
        if len(target) != size:
            raise ValueError("pearson dimension must equal the number of inputs")
        latent = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                value = nataf_correlation(
                    materialized[i].distribution,
                    materialized[j].distribution,
                    target[i][j],
                    order=order,
                )
                latent[i][j] = latent[j][i] = value
        return cls(materialized, tuple(tuple(row) for row in latent))

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
        return self.correlation is None

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

    def to_latent(self, x: Sequence[float]) -> Vector:
        """Map physical inputs to correlated latent normals ``z``.

        Args:
            x: Physical vector strictly inside every marginal support.

        Returns:
            ``z_i = Phi^-1(F_i(x_i))``.

        Raises:
            ValueError: If a CDF value is 0 or 1 (outside the open support).

        Equation:
            [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        latent: list[float] = []
        for item, value in zip(self.inputs, vector, strict=True):
            probability = float(item.distribution.cdf(value))
            if not 0.0 < probability < 1.0:
                raise ValueError(
                    f"{item.name}={value} lies outside the open support of its law"
                )
            latent.append(standard_normal_quantile(probability))
        return tuple(latent)

    def to_standard_normal(self, x: Sequence[float]) -> Vector:
        """Map physical inputs to independent standard normals ``u``.

        Args:
            x: Physical vector.

        Returns:
            ``u = L^-1 z`` with ``R = L L'``.

        Equation:
            [EQ-MOD-NATAF-001], [EQ-LINALG-002].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        latent = self.to_latent(x)
        if self.is_independent:
            return latent
        return solve_lower(self._lower, latent)

    def from_standard_normal(self, u: Sequence[float]) -> Vector:
        """Map independent standard normals ``u`` to physical inputs.

        Args:
            u: Finite standard-normal vector.

        Returns:
            ``x_i = F_i^-1(Phi(z_i))`` with ``z = L u``.

        Raises:
            ValueError: If ``u`` has the wrong length.

        Equation:
            [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(u, name="u")
        if len(vector) != self.dimension:
            raise ValueError("u length must equal the dimension")
        latent = vector if self.is_independent else matvec(self._lower, vector)
        physical: list[float] = []
        for item, z in zip(self.inputs, latent, strict=True):
            probability = standard_normal_cdf(z)
            probability = min(max(probability, 1e-300), 1.0 - 2.0**-53)
            physical.append(float(item.distribution.quantile(probability)))
        return tuple(physical)

    def from_unit(self, point: Sequence[float]) -> Vector:
        """Map a unit-hypercube design point to a physical input vector.

        Under independence each coordinate goes through its own inverse CDF,
        which preserves Latin-hypercube and low-discrepancy stratification.
        Under a Gaussian copula the point is first mapped to ``u`` by the
        normal quantile, then through :meth:`from_standard_normal`.

        Args:
            point: Point in ``(0, 1)^d``.

        Returns:
            Physical input vector.

        Raises:
            ValueError: If a coordinate is outside ``(0, 1)``.

        Equation:
            [EQ-MOD-NATAF-001], [EQ-SAMP-INV-001].

        References:
            [REF-DEVROYE-1986, Section II.2], [REF-LIU-DERKIUREGHIAN-1986].

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

    def log_pdf(self, x: Sequence[float]) -> float:
        """Return the joint log-density of physical inputs.

        ``log f(x) = sum_i log f_i(x_i) - 0.5 z'(R^-1 - I) z - 0.5 log det R``.

        Args:
            x: Physical vector.

        Returns:
            Joint log-density; ``-inf`` outside the support.

        Equation:
            [EQ-MOD-COPULA-001].

        References:
            [REF-NELSEN-2006, Section 2.3], [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(x, name="x")
        if len(vector) != self.dimension:
            raise ValueError("input vector length must equal the dimension")
        marginal = math.fsum(
            item.log_pdf(value) for item, value in zip(self.inputs, vector, strict=True)
        )
        if not math.isfinite(marginal) or self.is_independent:
            return marginal
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
