"""Regression-based polynomial chaos expansions on a probabilistic model.

A polynomial chaos expansion (PCE) represents a square-integrable model output
as ``Y = sum_alpha c_alpha Psi_alpha(xi)``, where ``Psi_alpha`` are
multivariate polynomials orthonormal under the law of independent standardized
variables ``xi``. Two bases are provided:

* ``"hermite"`` -- orthonormal probabilists' Hermite polynomials in the
  independent standard-normal space ``u`` of the model's Nataf transformation.
  It accepts any marginals and any Gaussian copula, but non-Gaussian
  marginals become non-polynomial functions of ``u`` and converge slower.
* ``"legendre"`` -- orthonormal Legendre polynomials of affinely mapped
  independent uniform inputs; it is the Wiener-Askey optimal basis for those
  inputs.

Coefficients are fitted by least squares on an experimental design; the
closed-form leave-one-out (LOO) error from the hat matrix estimates the
generalization error at no extra model cost. Mean, variance, and Sobol indices
follow from the coefficients by orthonormality.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.distributions import Uniform
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    householder_qr,
    solve_upper,
)
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.propagation import make_design

BASES = ("hermite", "legendre")
MultiIndex = tuple[int, ...]


def _compensated_dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return ``sum_i a_i b_i`` with compensated summation.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return math.fsum(map(operator.mul, left, right))


# Python 3.12+ ``math.sumprod`` accumulates in extended precision in C; the
# regression inner products dominate PCE cost, so it is used when available.
_dot: Callable[[Sequence[float], Sequence[float]], float] = getattr(
    math, "sumprod", _compensated_dot
)


def multi_indices(
    dimension: int, degree: int, *, q_norm: float = 1.0
) -> tuple[MultiIndex, ...]:
    """Return a total-degree or hyperbolic multi-index truncation set.

    ``A = {alpha in N^d : (sum_i alpha_i^q)^(1/q) <= p}``; ``q = 1`` is the
    total-degree set of cardinality ``C(d + p, p)``; ``q < 1`` removes
    high-order interaction terms first. Indices are ordered by total degree,
    then reverse-lexicographically, so index 0 is the constant term.

    Args:
        dimension: Number of inputs ``d >= 1``.
        degree: Maximal total degree ``p >= 0``.
        q_norm: Hyperbolic norm ``q`` in ``(0, 1]``.

    Returns:
        Ordered multi-indices.

    Raises:
        ValueError: If an argument is out of range.

    Equation:
        [EQ-SUR-PCE-004].

    References:
        [REF-BLATMAN-SUDRET-2011], [REF-SUDRET-2008].

    """
    for value, name, low in ((dimension, "dimension", 1), (degree, "degree", 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}")
    q = require_finite_real(q_norm, name="q_norm")
    if not 0.0 < q <= 1.0:
        raise ValueError("q_norm must lie in (0, 1]")

    def compositions(total: int, slots: int) -> list[MultiIndex]:
        """Return all non-negative integer vectors of a given sum.

        References:
            [EQ-SUR-PCE-004], [REF-SUDRET-2008].

        """
        if slots == 1:
            return [(total,)]
        result: list[MultiIndex] = []
        for first in range(total, -1, -1):
            result.extend(
                (first, *rest) for rest in compositions(total - first, slots - 1)
            )
        return result

    tolerance = 1e-12 * max(1.0, degree)
    selected: list[MultiIndex] = []
    for total in range(degree + 1):
        for alpha in compositions(total, dimension):
            norm = (
                math.fsum(a**q for a in alpha if a > 0) ** (1.0 / q) if total else 0.0
            )
            if norm <= degree + tolerance:
                selected.append(alpha)
    return tuple(selected)


def hermite_orthonormal(x: float, degree: int) -> Vector:
    """Return orthonormal probabilists' Hermite values ``psi_0..psi_p`` at ``x``.

    ``psi_0 = 1``, ``psi_1 = x``, and
    ``psi_(n+1) = (x psi_n - sqrt(n) psi_(n-1)) / sqrt(n+1)``, so that
    ``E[psi_m(Z) psi_n(Z)] = delta_mn`` for ``Z ~ N(0, 1)``.

    Args:
        x: Finite evaluation point.
        degree: Maximal degree ``p >= 0``.

    Returns:
        Tuple of ``p + 1`` values.

    Equation:
        [EQ-SUR-PCE-002].

    References:
        [REF-XIU-KARNIADAKIS-2002], [REF-NIST-DLMF-ORTHOGONAL].

    """
    values = [1.0]
    if degree >= 1:
        values.append(x)
    for n in range(1, degree):
        values.append((x * values[n] - math.sqrt(n) * values[n - 1]) / math.sqrt(n + 1))
    return tuple(values)


def legendre_orthonormal(x: float, degree: int) -> Vector:
    """Return orthonormal Legendre values ``psi_0..psi_p`` at ``x in [-1, 1]``.

    ``(n+1) P_(n+1) = (2n+1) x P_n - n P_(n-1)`` and
    ``psi_n = sqrt(2n+1) P_n``, so that ``E[psi_m(U) psi_n(U)] = delta_mn`` for
    ``U ~ Uniform(-1, 1)``.

    Args:
        x: Finite evaluation point.
        degree: Maximal degree ``p >= 0``.

    Returns:
        Tuple of ``p + 1`` values.

    Equation:
        [EQ-SUR-PCE-003].

    References:
        [REF-XIU-KARNIADAKIS-2002], [REF-NIST-DLMF-ORTHOGONAL].

    """
    legendre = [1.0]
    if degree >= 1:
        legendre.append(x)
    for n in range(1, degree):
        legendre.append(((2 * n + 1) * x * legendre[n] - n * legendre[n - 1]) / (n + 1))
    return tuple(math.sqrt(2 * n + 1) * value for n, value in enumerate(legendre))


@dataclass(frozen=True, slots=True)
class PolynomialChaosSobol:
    """Sobol indices computed analytically from PCE coefficients.

    Args:
        names: Input names in model order.
        first_order: First-order indices ``S_i``.
        total_order: Total-effect indices ``S_Ti``.
        second_order: Pairwise interaction indices ``S_ij`` keyed by name pair.
        variance: PCE output variance ``D``.

    References:
        [REF-SUDRET-2008].

    """

    names: tuple[str, ...]
    first_order: Mapping[str, float]
    total_order: Mapping[str, float]
    second_order: Mapping[tuple[str, str], float]
    variance: float


@dataclass(frozen=True)
class PolynomialChaosExpansion:
    """A fitted polynomial chaos surrogate of one quantity of interest.

    Args:
        inputs: Joint input law the expansion is orthonormal under.
        basis: ``"hermite"`` or ``"legendre"``.
        indices: Multi-index truncation set (index 0 is the constant term).
        coefficients: Least-squares coefficients ``c_alpha``.
        quantity: Name of the approximated quantity of interest.
        design: Physical experimental-design points.
        responses: Model responses at the design points.
        loo_error: Relative leave-one-out error
            ``mean_i((y_i - y_hat_(-i))^2) / Var(y)``.
        empirical_error: Relative training error ``mean_i((y_i - y_hat_i)^2)
            / Var(y)``.
        degree: Maximal total degree ``p``.
        q_norm: Hyperbolic truncation norm ``q``.

    References:
        [REF-XIU-KARNIADAKIS-2002], [REF-SUDRET-2008],
        [REF-BLATMAN-SUDRET-2011].

    """

    inputs: JointInput
    basis: str
    indices: tuple[MultiIndex, ...]
    coefficients: Vector
    quantity: str
    design: tuple[Vector, ...]
    responses: Vector
    loo_error: float
    empirical_error: float
    degree: int
    q_norm: float

    @property
    def size(self) -> int:
        """Return the number of basis terms ``P``.

        References:
            [REF-SUDRET-2008].

        """
        return len(self.indices)

    def standardize(self, x: Sequence[float]) -> Vector:
        """Map a physical input to the expansion's standardized variables.

        Args:
            x: Physical input vector.

        Returns:
            ``u`` (Hermite, Nataf map) or ``xi in [-1, 1]^d`` (Legendre).

        Equation:
            [EQ-MOD-NATAF-001], [EQ-SUR-PCE-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-XIU-KARNIADAKIS-2002].

        """
        return _standardize(self.inputs, self.basis, x)

    def basis_values(self, x: Sequence[float]) -> Vector:
        """Return ``Psi_alpha(xi(x))`` for every multi-index.

        Args:
            x: Physical input vector.

        Returns:
            One value per basis term.

        Equation:
            [EQ-SUR-PCE-001].

        References:
            [REF-XIU-KARNIADAKIS-2002].

        """
        return _basis_row(self.standardize(x), self.basis, self.indices, self.degree)

    def predict(self, x: Sequence[float]) -> float:
        """Evaluate the expansion at a physical input vector.

        Args:
            x: Physical input vector.

        Returns:
            Surrogate prediction ``sum_alpha c_alpha Psi_alpha``.

        Equation:
            [EQ-SUR-PCE-001].

        References:
            [REF-SUDRET-2008].

        """
        return math.fsum(map(operator.mul, self.coefficients, self.basis_values(x)))

    @property
    def mean(self) -> float:
        """Return the expansion mean ``c_0``.

        Equation:
            [EQ-SUR-PCE-007].

        References:
            [REF-SUDRET-2008].

        """
        return self.coefficients[0]

    @property
    def variance(self) -> float:
        """Return the expansion variance ``sum_(alpha != 0) c_alpha^2``.

        Equation:
            [EQ-SUR-PCE-007].

        References:
            [REF-SUDRET-2008].

        """
        return math.fsum(c * c for c in self.coefficients[1:])

    def sobol_indices(self) -> PolynomialChaosSobol:
        """Return first, second, and total Sobol indices from coefficients.

        ``S_u = sum_(alpha in A_u) c_alpha^2 / D`` where ``A_u`` holds the
        multi-indices whose non-zero entries are exactly the set ``u``;
        ``S_Ti`` sums every ``alpha`` with ``alpha_i > 0``.

        Returns:
            Analytic indices of the surrogate.

        Raises:
            ValueError: If the inputs are dependent (the ANOVA decomposition
                then attributes variance to the independent Nataf variables,
                not to the physical inputs) or the variance is zero.

        Equation:
            [EQ-SUR-PCE-008], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

        References:
            [REF-SUDRET-2008], [REF-SOBOL-1993].

        """
        if not self.inputs.is_independent:
            raise ValueError(
                "PCE Sobol indices require independent inputs; dependent inputs "
                "need dependence-aware sensitivity measures"
            )
        variance = self.variance
        if variance <= 0.0:
            raise ValueError("the expansion has zero variance")
        names = self.inputs.names
        dimension = len(names)
        first = [0.0] * dimension
        total = [0.0] * dimension
        second: dict[tuple[str, str], float] = {}
        for alpha, coefficient in zip(
            self.indices[1:], self.coefficients[1:], strict=True
        ):
            share = coefficient * coefficient / variance
            active = [i for i, a in enumerate(alpha) if a > 0]
            for i in active:
                total[i] += share
            if len(active) == 1:
                first[active[0]] += share
            elif len(active) == 2:
                key = (names[active[0]], names[active[1]])
                second[key] = second.get(key, 0.0) + share
        for i in range(dimension):
            for j in range(i + 1, dimension):
                second.setdefault((names[i], names[j]), 0.0)
        return PolynomialChaosSobol(
            names,
            MappingProxyType(dict(zip(names, first, strict=True))),
            MappingProxyType(dict(zip(names, total, strict=True))),
            MappingProxyType(second),
            variance,
        )


def _standardize(inputs: JointInput, basis: str, x: Sequence[float]) -> Vector:
    """Map a physical vector to the standardized PCE variables.

    Args:
        inputs: Joint input law.
        basis: ``"hermite"`` or ``"legendre"``.
        x: Physical input vector.

    Returns:
        Standardized variables.

    Raises:
        ValueError: If a Legendre coordinate lies outside its support.

    Equation:
        [EQ-MOD-NATAF-001], [EQ-SUR-PCE-001].

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-XIU-KARNIADAKIS-2002].

    """
    if basis == "hermite":
        return inputs.to_standard_normal(x)
    vector = as_vector(x, name="x")
    if len(vector) != inputs.dimension:
        raise ValueError("input vector length must equal the dimension")
    result: list[float] = []
    for item, value in zip(inputs.inputs, vector, strict=True):
        law = item.distribution
        assert isinstance(law, Uniform)
        xi = (2.0 * value - (law.lower + law.upper)) / (law.upper - law.lower)
        if not -1.0 - 1e-12 <= xi <= 1.0 + 1e-12:
            raise ValueError(f"{item.name}={value} lies outside its uniform support")
        result.append(xi)
    return tuple(result)


def _basis_row(
    standardized: Vector, basis: str, indices: Sequence[MultiIndex], degree: int
) -> Vector:
    """Evaluate every multivariate basis polynomial at one standardized point.

    Args:
        standardized: Standardized variables.
        basis: ``"hermite"`` or ``"legendre"``.
        indices: Multi-indices.
        degree: Maximal univariate degree needed.

    Returns:
        ``prod_i psi_(alpha_i)(xi_i)`` for every ``alpha``.

    Equation:
        [EQ-SUR-PCE-001].

    References:
        [REF-XIU-KARNIADAKIS-2002].

    """
    family = hermite_orthonormal if basis == "hermite" else legendre_orthonormal
    tables = [family(value, degree) for value in standardized]
    row: list[float] = []
    for alpha in indices:
        product = 1.0
        for table, a in zip(tables, alpha, strict=True):
            if a:
                product *= table[a]
        row.append(product)
    return tuple(row)


def _check_basis(inputs: JointInput, basis: str) -> None:
    """Validate that a basis is applicable to a joint input law.

    Raises:
        ValueError: If the basis is unknown, or Legendre is requested for
            dependent or non-uniform inputs.

    References:
        [REF-XIU-KARNIADAKIS-2002]. No equation is evaluated.

    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    if basis == "legendre":
        if not inputs.is_independent:
            raise ValueError("the Legendre basis requires independent inputs")
        for item in inputs.inputs:
            if not isinstance(item.distribution, Uniform):
                raise ValueError(
                    f"the Legendre basis requires Uniform marginals; "
                    f"{item.name!r} is not uniform"
                )


def polynomial_chaos_from_samples(
    inputs: JointInput,
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    degree: int,
    basis: str = "hermite",
    q_norm: float = 1.0,
    quantity: str = "y",
) -> PolynomialChaosExpansion:
    """Fit PCE coefficients by least squares on given input/output samples.

    ``c = R^-1 Q' y`` from the Householder factorization ``Psi = Q R`` of
    the design matrix, so the solve sees ``kappa(Psi)`` rather than the
    ``kappa(Psi)^2`` of the normal equations ``Psi' Psi c = Psi' y`` used
    before release 0.16 (which failed for high-degree Hermite fits). The
    LOO residual is ``(y_i - y_hat_i) / (1 - h_i)`` with hat-matrix diagonal
    ``h_i = ||q_i||^2`` (row ``i`` of ``Q``); no model is re-evaluated.

    Args:
        inputs: Joint input law of the samples.
        points: Physical input vectors.
        values: Model responses, one per point.
        degree: Maximal total degree ``p``.
        basis: ``"hermite"`` or ``"legendre"``.
        q_norm: Hyperbolic truncation norm ``q`` in ``(0, 1]``.
        quantity: Label of the approximated quantity.

    Returns:
        Fitted expansion with LOO and empirical errors.

    Raises:
        ValueError: If there are fewer samples than basis terms, the design
            matrix is numerically rank deficient (the message names the
            dependent multi-indices), or responses are constant.

    Equation:
        [EQ-SUR-PCE-001], [EQ-SUR-PCE-004], [EQ-SUR-PCE-005],
        [EQ-SUR-PCE-006].

    References:
        [REF-BLATMAN-SUDRET-2011], [REF-SUDRET-2008],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    _check_basis(inputs, basis)
    indices = multi_indices(inputs.dimension, degree, q_norm=q_norm)
    design = tuple(as_vector(point, name="point") for point in points)
    responses = as_vector(values, name="values")
    if len(design) != len(responses):
        raise ValueError("points and values must have equal length")
    count = len(design)
    terms = len(indices)
    if count <= terms:
        raise ValueError(
            f"need more samples ({count}) than basis terms ({terms}) for "
            "least squares with a leave-one-out estimate"
        )
    rows = [
        _basis_row(_standardize(inputs, basis, point), basis, indices, degree)
        for point in design
    ]
    factor = householder_qr(tuple(tuple(row) for row in rows))
    if not factor.full_rank:
        deficient = ", ".join(str(indices[j]) for j in factor.rank_deficient)
        raise ValueError(
            "the experimental design does not identify the coefficients: the "
            f"basis columns for multi-indices {deficient} are numerically "
            "dependent on earlier columns (|R_jj| <= 1e-10 ||psi_j|| in the "
            "Householder QR of Psi); increase the design size or lower the degree"
        )
    q_columns = list(zip(*factor.q, strict=True))
    projected = tuple(_dot(column, responses) for column in q_columns)
    coefficients = solve_upper(factor.r, projected)
    center = math.fsum(responses) / count
    output_variance = math.fsum((y - center) ** 2 for y in responses) / (count - 1)
    if output_variance <= 0.0:
        raise ValueError("responses are constant; nothing to approximate")
    squared_residuals: list[float] = []
    squared_loo: list[float] = []
    for index, (row, y) in enumerate(zip(rows, responses, strict=True)):
        prediction = _dot(coefficients, row)
        residual = y - prediction
        leverage = math.fsum(v * v for v in factor.q[index])
        squared_residuals.append(residual * residual)
        if leverage >= 1.0 - 1e-12:
            squared_loo.append(math.inf)
        else:
            squared_loo.append((residual / (1.0 - leverage)) ** 2)
    return PolynomialChaosExpansion(
        inputs,
        basis,
        indices,
        coefficients,
        quantity,
        design,
        responses,
        math.fsum(squared_loo) / count / output_variance,
        math.fsum(squared_residuals) / count / output_variance,
        degree,
        q_norm,
    )


def fit_polynomial_chaos(
    model: ProbabilisticModel,
    quantity: str | None = None,
    *,
    degree: int,
    basis: str = "hermite",
    q_norm: float = 1.0,
    size: int | None = None,
    design: str = "lhs",
    seed: int | RandomStream | None = None,
) -> PolynomialChaosExpansion:
    """Build a PCE surrogate of one QoI of a probabilistic model.

    The experimental design is drawn on the unit hypercube (``make_design``)
    and mapped through the model's joint input law, so the same object that
    drives propagation, reliability, and sensitivity drives the fit.

    Args:
        model: Probabilistic model.
        quantity: QoI name; defaults to the first declared quantity.
        degree: Maximal total degree ``p``.
        basis: ``"hermite"`` or ``"legendre"``.
        q_norm: Hyperbolic truncation norm ``q``.
        size: Number of model evaluations; defaults to twice the number of
            basis terms.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Integer seed or stream (child ``"pce-design"``).

    Returns:
        Fitted expansion; ``model.evaluations`` grows by ``size``.

    Raises:
        ValueError: If the basis is inapplicable or the design is too small.

    Equation:
        [EQ-SUR-PCE-005], [EQ-SAMP-INV-001].

    References:
        [REF-SUDRET-2008], [REF-BLATMAN-SUDRET-2011].

    """
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    _check_basis(model.inputs, basis)
    terms = len(multi_indices(model.dimension, degree, q_norm=q_norm))
    count = 2 * terms if size is None else size
    stream = resolve_stream(seed).child("pce-design")
    unit = make_design(design, count, model.dimension, seed=stream)
    points = model.inputs.sample_design(unit)
    values = model.evaluate_many(points)[name]
    return polynomial_chaos_from_samples(
        model.inputs,
        points,
        values,
        degree=degree,
        basis=basis,
        q_norm=q_norm,
        quantity=name,
    )


def normalized_information_matrix(expansion: PolynomialChaosExpansion) -> Matrix:
    """Return the normalized information matrix ``Psi' Psi / N`` of a fit.

    For a good design it approaches the identity (orthonormality); large
    off-diagonal entries flag an ill-conditioned regression.

    Args:
        expansion: Fitted expansion.

    Returns:
        ``P`` by ``P`` normalized information matrix.

    Equation:
        [EQ-SUR-PCE-005].

    References:
        [REF-BLATMAN-SUDRET-2011].

    """
    rows = [expansion.basis_values(point) for point in expansion.design]
    columns = list(zip(*rows, strict=True))
    count = len(rows)
    return tuple(
        tuple(math.fsum(map(operator.mul, a, b)) / count for b in columns)
        for a in columns
    )
