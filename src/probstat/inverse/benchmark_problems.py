"""Inverse-problem benchmark catalog with declared truth provenance (roadmap L01-L06).

Every case returns an :class:`InverseBenchmark` that separates what a solver
may see (operator, data, noise covariance, optional prior) from what only a
benchmark may see (the truth and the noise-free data). Seeds for the truth,
the noise and any random design are drawn from DISTINCT children of one
:class:`~probstat.foundations.RandomStream`, so changing the noise seed never
changes the truth.

``same_discretization`` marks whether the data were generated with the SAME
discrete operator that the solver inverts (algebraic verification only: an
"inverse crime") or from an independent finer model, which leaves a genuine
discretization error in the data [EQ-INV-BENCH-001]. Results on
``same_discretization=True`` cases verify algebra, not realistic accuracy.

Cases:

- L01 :func:`diagonal_spectrum_benchmark`: ``sigma_i = i^(-p)`` (mild) or
  ``exp(-c (i - 1))`` (severe), optional exact null directions.
- L04 :func:`deconvolution_benchmark`: matrix-free 1D blur with an explicit
  boundary; piecewise-constant or smooth truth; optional independent
  fine-grid data generation.
- L05 :func:`sparse_sensing_benchmark`: underdetermined Gaussian sensing of
  a sparse truth.
- L06 :func:`correlated_gaussian_benchmark`: Gaussian prior with exponential
  correlation, truth drawn from the prior, for exact-posterior comparisons.

The existing ``backward_heat_problem`` (L02) and ``second_derivative_problem``
(L03) remain the closed-form references; :data:`INVERSE_BENCHMARK_CATALOG`
lists all six with their pathologies and compatible methods.

Experimental (added in 0.28).

References:
    [REF-HANSEN-2008, Chapter 4], [REF-GAZZOLA-HANSEN-NAGY-2019],
    [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    Convolution1DOperator,
    DenseCovariance,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    ScaledIdentityCovariance,
    Vector,
)


@dataclass(frozen=True, slots=True)
class InverseBenchmark:
    """One benchmark instance: solver inputs plus the withheld truth.

    Attributes:
        identifier: Catalog ID (``"L01"``...).
        name: Short description.
        operator: Forward operator ``A`` given to the solver.
        data: Noisy data ``y`` given to the solver.
        noise: Noise covariance factor given to the solver.
        noise_level: Noise standard deviation (scalar, for iid noise).
        realized_noise_norm: ``||y - b||`` of this draw (reported, not
            rescaled).
        truth: Withheld true parameters on the solver's grid.
        exact_data: Withheld noise-free data ``b``.
        same_discretization: ``True`` when ``b = A x_true`` exactly (inverse
            crime, algebraic verification only).
        prior_covariance: Prior covariance factor for Bayesian cases, else
            ``None``.
        singular_values: Exact singular values when known, else ``None``.
        pathologies: Expected difficulties (e.g. ``"nullspace"``).
        seeds: The stream paths used for ``truth`` and ``noise``.

    References:
        [EQ-INV-BENCH-001], [REF-KAIPIO-SOMERSALO-2007].

    """

    identifier: str
    name: str
    operator: LinearOperator
    data: Vector
    noise: ScaledIdentityCovariance
    noise_level: float
    realized_noise_norm: float
    truth: Vector
    exact_data: Vector
    same_discretization: bool
    prior_covariance: DenseCovariance | None
    singular_values: Vector | None
    pathologies: tuple[str, ...]
    seeds: Mapping[str, str] = field(default_factory=dict)


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    """Validate an integer of at least ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _noise(
    exact: Vector, relative_level: float, stream: RandomStream
) -> tuple[Vector, float, float]:
    """Add iid Gaussian noise with ``sd = relative_level ||b|| / sqrt(m)``.

    The draw is NOT rescaled to a fixed norm; the realized norm is returned.

    References:
        [EQ-INV-BENCH-001].

    """
    level = float(relative_level)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError("relative_noise must be finite and positive")
    m = len(exact)
    sigma = level * math.hypot(*exact) / math.sqrt(m)
    if sigma == 0.0:
        raise ValueError("the exact data vanish; a relative noise level is undefined")
    generator = stream.child("noise").generator()
    draws = [generator.gauss(0.0, 1.0) for _ in range(m)]
    data = tuple(b + sigma * e for b, e in zip(exact, draws, strict=True))
    return data, sigma, sigma * math.hypot(*draws)


def diagonal_spectrum_benchmark(
    size: int = 32,
    *,
    decay: str = "polynomial",
    rate: float = 1.0,
    nullity: int = 0,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L01: ``A = diag(sigma)`` with a smooth truth and known spectrum.

    ``sigma_i = i^(-rate)`` (``"polynomial"``, mildly ill-posed) or
    ``exp(-rate (i - 1))`` (``"exponential"``, severely ill-posed); the last
    ``nullity`` values are set to zero. The truth ``x_i = i^(-1)`` satisfies
    the discrete Picard condition for ``rate < 1`` only in the polynomial
    case; for faster decay the Picard plot shows the expected breakdown.

    Args:
        size: ``n >= 2``.
        decay: ``"polynomial"`` or ``"exponential"``.
        rate: Positive decay rate.
        nullity: Number of exact zero singular values (``< n``).
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed.

    Returns:
        :class:`InverseBenchmark` with ``same_discretization=True``.

    References:
        [EQ-INV-BENCH-001], [REF-HANSEN-2008, Section 2.4].

    """
    n = _positive_int(size, "size", 2)
    if decay not in ("polynomial", "exponential"):
        raise ValueError("decay must be 'polynomial' or 'exponential'")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be positive")
    if (
        isinstance(nullity, bool)
        or not isinstance(nullity, int)
        or not 0 <= nullity < n
    ):
        raise ValueError("nullity must be an integer in [0, n)")
    sigma = [
        (i ** (-rate) if decay == "polynomial" else math.exp(-rate * (i - 1)))
        for i in range(1, n + 1)
    ]
    for i in range(n - nullity, n):
        sigma[i] = 0.0
    truth = tuple(1.0 / i for i in range(1, n + 1))
    exact = tuple(s * x for s, x in zip(sigma, truth, strict=True))
    stream = resolve_stream(seed).child("L01")
    data, level, realized = _noise(exact, relative_noise, stream)
    pathologies = ("ill_conditioned",) + (("nullspace",) if nullity else ())
    return InverseBenchmark(
        identifier="L01",
        name=f"diagonal spectrum ({decay}, rate {rate})",
        operator=DiagonalOperator(sigma),
        data=data,
        noise=ScaledIdentityCovariance(n, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=True,
        prior_covariance=None,
        singular_values=tuple(sorted(sigma, reverse=True)),
        pathologies=pathologies,
        seeds=MappingProxyType({"noise": "L01/noise"}),
    )


def _gaussian_kernel(width: float, spacing: float) -> list[float]:
    """Return normalized Gaussian blur weights of standard deviation ``width``.

    References:
        [EQ-INV-BENCH-001].

    """
    half = max(1, int(math.ceil(4.0 * width / spacing)))
    weights = [
        math.exp(-0.5 * ((k * spacing) / width) ** 2) for k in range(-half, half + 1)
    ]
    total = math.fsum(weights)
    return [w / total for w in weights]


def _piecewise_truth(s: float) -> float:
    """Piecewise-constant truth on ``[0, 1]`` with three jumps.

    References:
        [EQ-INV-BENCH-001].

    """
    if s < 0.25:
        return 0.0
    if s < 0.5:
        return 1.0
    if s < 0.7:
        return 0.3
    return 0.8


def _smooth_truth(s: float) -> float:
    """Smooth truth ``exp(-60 (s - 0.4)^2) + 0.5 exp(-120 (s - 0.75)^2)``.

    References:
        [EQ-INV-BENCH-001].

    """
    return math.exp(-60.0 * (s - 0.4) ** 2) + 0.5 * math.exp(-120.0 * (s - 0.75) ** 2)


def deconvolution_benchmark(
    size: int = 64,
    *,
    width: float = 0.03,
    boundary: str = "zero",
    truth: str = "piecewise",
    refinement: int = 1,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L04: matrix-free 1D Gaussian deblurring on ``[0, 1]``.

    The solver's operator is :class:`~probstat.linalg.Convolution1DOperator`
    on ``n`` cell centres. With ``refinement = 1`` the data are ``A x_true``
    (inverse crime). With ``refinement = r > 1`` the truth is sampled on an
    ``r n`` fine grid, blurred there with the same physical kernel and
    boundary, and cell-averaged onto the coarse grid; the solver's truth is
    the cell average of the fine truth, so the data contain a genuine
    discretization error [EQ-INV-BENCH-001].

    Args:
        size: Coarse grid size ``n >= 8``.
        width: Kernel standard deviation in physical units.
        boundary: ``"zero"``, ``"periodic"`` or ``"reflect"``.
        truth: ``"piecewise"`` or ``"smooth"``.
        refinement: Integer factor ``r >= 1`` of the data-generating grid.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed.

    Returns:
        :class:`InverseBenchmark`.

    References:
        [EQ-INV-BENCH-001], [REF-GAZZOLA-HANSEN-NAGY-2019],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    n = _positive_int(size, "size", 8)
    r = _positive_int(refinement, "refinement", 1)
    if truth not in ("piecewise", "smooth"):
        raise ValueError("truth must be 'piecewise' or 'smooth'")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be positive")
    profile = _piecewise_truth if truth == "piecewise" else _smooth_truth
    h = 1.0 / n
    operator = Convolution1DOperator(_gaussian_kernel(width, h), n, boundary=boundary)
    fine_n = r * n
    fine_h = 1.0 / fine_n
    fine_truth = [profile((j + 0.5) * fine_h) for j in range(fine_n)]
    coarse_truth = tuple(
        math.fsum(fine_truth[i * r : (i + 1) * r]) / r for i in range(n)
    )
    if r == 1:
        exact = operator.matvec(coarse_truth)
    else:
        fine_operator = Convolution1DOperator(
            _gaussian_kernel(width, fine_h), fine_n, boundary=boundary
        )
        fine_data = fine_operator.matvec(fine_truth)
        exact = tuple(math.fsum(fine_data[i * r : (i + 1) * r]) / r for i in range(n))
    stream = resolve_stream(seed).child("L04")
    data, level, realized = _noise(exact, relative_noise, stream)
    pathologies = ("ill_conditioned", "edges" if truth == "piecewise" else "smooth")
    return InverseBenchmark(
        identifier="L04",
        name=f"1D deblurring ({boundary}, {truth}, refinement {r})",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(n, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=coarse_truth,
        exact_data=exact,
        same_discretization=r == 1,
        prior_covariance=None,
        singular_values=None,
        pathologies=pathologies,
        seeds=MappingProxyType({"noise": "L04/noise"}),
    )


def sparse_sensing_benchmark(
    observations: int = 20,
    unknowns: int = 50,
    *,
    sparsity: int = 4,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L05: ``y = A x + e``, ``A_ij ~ N(0, 1/m)``, with a sparse truth.

    The support (``sparsity`` indices) and the amplitudes (``+-1`` times
    ``1 + U(0, 1)``) come from the ``truth`` stream, the matrix from the
    ``design`` stream and the noise from the ``noise`` stream.

    Args:
        observations: ``m >= 2``.
        unknowns: ``n > m`` (underdetermined).
        sparsity: Number of non-zeros, ``1 <= s < m``.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed.

    Returns:
        :class:`InverseBenchmark` (same discretization; the case tests
        sparsity priors, not discretization).

    References:
        [EQ-INV-BENCH-001], [REF-DAUBECHIES-DEFRISE-DEMOL-2004].

    """
    m = _positive_int(observations, "observations", 2)
    n = _positive_int(unknowns, "unknowns", m + 1)
    s = _positive_int(sparsity, "sparsity", 1)
    if s >= m:
        raise ValueError("sparsity must be smaller than the number of observations")
    stream = resolve_stream(seed).child("L05")
    design = stream.child("design").generator()
    scale = 1.0 / math.sqrt(m)
    matrix = [[scale * design.gauss(0.0, 1.0) for _ in range(n)] for _ in range(m)]
    generator = stream.child("truth").generator()
    support = sorted(generator.sample(range(n), s))
    truth = [0.0] * n
    for index in support:
        sign = 1.0 if generator.random() < 0.5 else -1.0
        truth[index] = sign * (1.0 + generator.random())
    operator = DenseOperator(matrix)
    exact = operator.matvec(truth)
    data, level, realized = _noise(exact, relative_noise, stream)
    return InverseBenchmark(
        identifier="L05",
        name=f"sparse sensing ({m} x {n}, {s}-sparse)",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(m, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=tuple(truth),
        exact_data=exact,
        same_discretization=True,
        prior_covariance=None,
        singular_values=None,
        pathologies=("underdetermined", "nullspace", "sparse_truth"),
        seeds=MappingProxyType(
            {"design": "L05/design", "truth": "L05/truth", "noise": "L05/noise"}
        ),
    )


def correlated_gaussian_benchmark(
    size: int = 20,
    observations: int = 12,
    *,
    length_scale: float = 0.2,
    prior_variance: float = 1.0,
    noise_level: float = 0.05,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L06: a linear Gaussian problem with a correlated prior.

    Prior ``x ~ N(0, C0)``, ``(C0)_ij = v exp(-|s_i - s_j| / l)`` on ``n``
    points of ``[0, 1]`` (the exponential kernel is SPD for distinct points);
    the truth is a prior draw (``truth`` stream). Observations are local
    averages ``y_k = mean(x_j : j in window k) + e_k`` over ``m`` evenly
    placed windows of three points, ``e ~ N(0, s^2 I)``. The exact posterior
    is available from :func:`probstat.inverse.linear_gaussian_posterior`.

    Args:
        size: ``n >= 4``.
        observations: ``2 <= m <= n - 2``.
        length_scale: Positive correlation length ``l``.
        prior_variance: Positive ``v``.
        noise_level: Positive ``s`` (absolute).
        seed: Root seed.

    Returns:
        :class:`InverseBenchmark` with ``prior_covariance`` set.

    References:
        [EQ-INV-BENCH-001], [REF-STUART-2010].

    """
    n = _positive_int(size, "size", 4)
    m = _positive_int(observations, "observations", 2)
    if m > n - 2:
        raise ValueError("observations must not exceed size - 2")
    for value, label in (
        (length_scale, "length_scale"),
        (prior_variance, "prior_variance"),
        (noise_level, "noise_level"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive")
    grid = [i / (n - 1) for i in range(n)]
    covariance = [
        [prior_variance * math.exp(-abs(a - b) / length_scale) for b in grid]
        for a in grid
    ]
    prior = DenseCovariance(covariance)
    stream = resolve_stream(seed).child("L06")
    generator = stream.child("truth").generator()
    truth = prior.unwhiten([generator.gauss(0.0, 1.0) for _ in range(n)])
    rows = []
    for k in range(m):
        centre = 1 + round(k * (n - 3) / max(m - 1, 1))
        rows.append([1.0 / 3.0 if abs(j - centre) <= 1 else 0.0 for j in range(n)])
    operator = DenseOperator(rows)
    exact = operator.matvec(truth)
    noise_generator = stream.child("noise").generator()
    draws = [noise_generator.gauss(0.0, 1.0) for _ in range(m)]
    data = tuple(b + noise_level * e for b, e in zip(exact, draws, strict=True))
    return InverseBenchmark(
        identifier="L06",
        name=f"correlated Gaussian prior (n={n}, m={m}, l={length_scale})",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(m, noise_level * noise_level),
        noise_level=noise_level,
        realized_noise_norm=noise_level * math.hypot(*draws),
        truth=truth,
        exact_data=exact,
        same_discretization=True,
        prior_covariance=prior,
        singular_values=None,
        pathologies=("underdetermined", "prior_dominated_modes"),
        seeds=MappingProxyType({"truth": "L06/truth", "noise": "L06/noise"}),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Catalog entry of an inverse benchmark family.

    Attributes:
        identifier: ``"L01"``...
        name: Short name.
        builder: Dotted path of the function that builds an instance.
        reference: What the case is verified against.
        pathologies: Expected difficulties.
        methods: Methods the case is meant to exercise.

    References:
        [EQ-INV-BENCH-001].

    """

    identifier: str
    name: str
    builder: str
    reference: str
    pathologies: tuple[str, ...]
    methods: tuple[str, ...]


INVERSE_BENCHMARK_CATALOG: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "L01",
        "diagonal spectrum",
        "inverse.diagonal_spectrum_benchmark",
        "exact SVD and filter factors",
        ("ill_conditioned", "nullspace"),
        (
            "spectral_diagnostics",
            "solve_regularized",
            "landweber",
            "regularization_path",
        ),
    ),
    BenchmarkCase(
        "L02",
        "backward heat",
        "inverse.backward_heat_problem",
        "closed-form sine SVD",
        ("severely_ill_posed",),
        ("spectral_diagnostics", "krylov_regularization", "regularization_path"),
    ),
    BenchmarkCase(
        "L03",
        "second derivative (deriv2)",
        "inverse.second_derivative_problem",
        "midpoint discretization with exact solution",
        ("mildly_ill_posed",),
        ("krylov_regularization", "hybrid_lsqr", "regularization_path"),
    ),
    BenchmarkCase(
        "L04",
        "1D deblurring",
        "inverse.deconvolution_benchmark",
        "fine-grid independent data (refinement > 1)",
        ("ill_conditioned", "edges"),
        ("krylov_regularization", "total_variation_regularized_least_squares"),
    ),
    BenchmarkCase(
        "L05",
        "sparse sensing",
        "inverse.sparse_sensing_benchmark",
        "known sparse support",
        ("underdetermined", "nullspace", "sparse_truth"),
        ("l1_regularized_least_squares", "nonnegative_least_squares"),
    ),
    BenchmarkCase(
        "L06",
        "correlated Gaussian prior",
        "inverse.correlated_gaussian_benchmark",
        "exact Gaussian posterior",
        ("underdetermined", "prior_dominated_modes"),
        ("linear_gaussian_posterior", "low_rank_posterior", "regularization_path"),
    ),
)
"""The implemented benchmark families (roadmap 19, L01-L06)."""
