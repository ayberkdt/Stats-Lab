"""Simplified 2D gravity inversion: a synthetic benchmark (G01).

Experimental (added in 0.28).

This module is a SYNTHETIC test bed for inverse-problem methods. It supports
no claim about the real Earth or Moon, about a real gravity field, about a real
spacecraft or survey, or about any interior structure: the densities, the
grid and the data are generated locally from a stated model and seed.

Forward model. A buried density contrast ``rho(x, z)`` (``z`` is depth,
positive downward) is invariant along the third axis, so each cell is an
infinite horizontal prism and the vertical attraction at the surface point
``(x_o, -h)`` (``h >= 0`` the observation height) is, with the exact line-mass
kernel ``2 G b / (a^2 + b^2)`` integrated over the cell
``[x_1, x_2] x [z_1, z_2]`` [EQ-INV-APP-GRAV-001]::

    g_z = 2 G rho (F(a_2, b_2) - F(a_1, b_2) - F(a_2, b_1) + F(a_1, b_1)),
    F(a, b) = b atan(a / b) + (a / 2) ln(a^2 + b^2),
    a_k = x_k - x_o,   b_k = z_k + h.

``F`` is a primitive of the kernel (``d^2 F / da db = b / (a^2 + b^2)``), so
the cell integral is exact (no quadrature); the tests use an independent
Gauss-Legendre rule as the oracle. Units are canonical: with the density unit
``rho_0`` and the length unit ``L_0``, ``G = 1`` makes ``G rho_0 L_0`` the
gravity unit.

Non-uniqueness. The data are ``m`` surface values and the unknowns ``n`` cells
(``n > m``), so the operator ``A`` has at least ``n - m`` exact null
directions, and the classical potential-theory ambiguity (a surface field is
reproduced by sources at many depths; Blakely 1995) shows as a rapidly decaying
singular spectrum: :func:`gravity_spectrum`, :func:`null_space_element`,
:func:`distinguishability` and :func:`single_layer_fit` measure it
[EQ-INV-APP-GRAV-004, EQ-INV-APP-GRAV-005]. Plain Tikhonov regularization
concentrates the recovered density near the surface; the depth-weighting
penalty ``w(z) = (z + z_0)^(-beta / 2)`` [EQ-INV-APP-GRAV-003] (a heuristic in
the style of Li and Oldenburg, whose exponent should be matched to the measured
sensitivity decay of :func:`sensitivity_depth_exponent`) counteracts that.

Catalog case ``G01`` (:func:`gravity_inversion_benchmark`): the data come from
a FINER set of cells than the inversion mesh (``refinement > 1``) so there is
no inverse crime; ``refinement = 1`` sets ``same_discretization = True`` and is
algebraic verification only [EQ-INV-APP-GRAV-002]. The truth is withheld and
projected on the inversion mesh by block averaging.

References:
    [REF-BLAKELY-1995], [REF-LI-OLDENBURG-1998], [REF-HANSEN-2008],
    [REF-TIKHONOV-1963], [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.benchmark_problems import (
    BenchmarkCase,
    _noise,
    _positive_int,
)
from probstat.inverse.regularization import tikhonov
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    Matrix,
    ScaledIdentityCovariance,
    Vector,
    matvec,
    singular_value_decomposition,
    transpose,
)

_TRUTHS = ("two_bodies", "deep_block", "smooth_anomaly")


def _finite_real(value: object, name: str) -> float:
    """Validate a finite real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    """Validate a finite strictly positive real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    number = _finite_real(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _real_sequence(values: object, name: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite reals from a (non-string) sequence.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of reals")
    out = tuple(_finite_real(v, name) for v in values)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _edges(values: object, name: str) -> tuple[float, ...]:
    """Validate strictly increasing cell edges with a non-negative start.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    edges = _real_sequence(values, name)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing with >= 2 entries")
    return edges


def _primitive(a: float, b: float) -> float:
    """Return ``F(a, b) = b atan(a / b) + (a / 2) ln(a^2 + b^2)``.

    The limit ``b -> 0`` is ``a ln|a|`` (and ``0`` at the origin), so a
    station on the top face of a cell is finite.

    References:
        [EQ-INV-APP-GRAV-001].

    """
    if b == 0.0:
        return a * math.log(abs(a)) if a != 0.0 else 0.0
    return b * math.atan(a / b) + 0.5 * a * math.log(a * a + b * b)


def prism_attraction(
    x_obs: float,
    height: float,
    x_edges: tuple[float, float],
    z_edges: tuple[float, float],
    *,
    gravitational_constant: float = 1.0,
) -> float:
    """Return the vertical attraction of one unit-density 2D prism.

    Args:
        x_obs: Horizontal position of the observation point.
        height: Observation height ``h >= 0`` above the surface ``z = 0``.
        x_edges: ``(x_1, x_2)`` with ``x_1 < x_2``.
        z_edges: Depths ``(z_1, z_2)`` with ``0 <= z_1 < z_2``.
        gravitational_constant: ``G > 0`` (canonical units: ``1``).

    Returns:
        ``g_z`` per unit density, positive downward, ``2 G`` times the exact
        integral of the line-mass kernel over the cell.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the edges are not ordered, a depth is negative, the
            height is negative or ``G`` is not positive.

    References:
        [EQ-INV-APP-GRAV-001], [REF-BLAKELY-1995].

    """
    xo = _finite_real(x_obs, "x_obs")
    h = _finite_real(height, "height")
    g = _positive(gravitational_constant, "gravitational_constant")
    x1, x2 = _finite_real(x_edges[0], "x_edges"), _finite_real(x_edges[1], "x_edges")
    z1, z2 = _finite_real(z_edges[0], "z_edges"), _finite_real(z_edges[1], "z_edges")
    if h < 0.0 or z1 < 0.0 or x2 <= x1 or z2 <= z1:
        raise ValueError(
            "need height >= 0, 0 <= z_1 < z_2 and x_1 < x_2 for a prism below"
            " the observation plane"
        )
    a1, a2, b1, b2 = x1 - xo, x2 - xo, z1 + h, z2 + h
    return (
        2.0
        * g
        * (
            _primitive(a2, b2)
            - _primitive(a1, b2)
            - _primitive(a2, b1)
            + _primitive(a1, b1)
        )
    )


def gravity_kernel_matrix(
    stations: Sequence[float],
    x_edges: Sequence[float],
    z_edges: Sequence[float],
    *,
    height: float = 0.0,
    gravitational_constant: float = 1.0,
) -> Matrix:
    """Return the dense ``m x n`` matrix of cell attractions.

    Cells are ordered depth-major (``j = i_z n_x + i_x``); row ``i`` is the
    observation point ``stations[i]`` [EQ-INV-APP-GRAV-001].

    Args:
        stations: Horizontal observation positions (non-empty).
        x_edges: Strictly increasing horizontal cell edges.
        z_edges: Strictly increasing depth edges, ``z_edges[0] >= 0``.
        height: Observation height ``>= 0``.
        gravitational_constant: ``G > 0``.

    Returns:
        ``A`` with ``A_ij = g_z`` of cell ``j`` at station ``i`` for unit
        density.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an edge list is not increasing, a depth is negative or
            the height is negative.

    References:
        [EQ-INV-APP-GRAV-001].

    """
    xs = _real_sequence(stations, "stations")
    ex = _edges(x_edges, "x_edges")
    ez = _edges(z_edges, "z_edges")
    if ez[0] < 0.0:
        raise ValueError("z_edges must start at a non-negative depth")
    h = _finite_real(height, "height")
    rows = []
    for xo in xs:
        row = []
        for k in range(len(ez) - 1):
            for j in range(len(ex) - 1):
                row.append(
                    prism_attraction(
                        xo,
                        h,
                        (ex[j], ex[j + 1]),
                        (ez[k], ez[k + 1]),
                        gravitational_constant=gravitational_constant,
                    )
                )
        rows.append(tuple(row))
    return tuple(rows)


def depth_weights(
    depth_centers: Sequence[float], *, beta: float = 3.0, z0: float = 0.1
) -> Vector:
    """Return the depth-weighting penalty weights ``(z + z_0)^(-beta / 2)``.

    The weights multiply the density in the Tikhonov penalty
    ``||W x||^2``, so shallow cells (large ``W``) are penalized more than
    deep ones and the recovered density is pushed down. ``beta`` and ``z0``
    are tuning choices (a heuristic after Li and Oldenburg); match ``beta`` to
    the measured sensitivity decay of :func:`sensitivity_depth_exponent`
    [EQ-INV-APP-GRAV-003].

    Args:
        depth_centers: Cell-centre depths ``> -z0`` (one per cell).
        beta: Exponent ``>= 0`` (``0`` gives ordinary Tikhonov).
        z0: Offset ``> 0`` avoiding the singularity at the surface.

    Returns:
        One weight per cell.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``z0`` is not positive, ``beta`` is negative or a
            depth does not exceed ``-z0``.

    References:
        [EQ-INV-APP-GRAV-003], [REF-LI-OLDENBURG-1998].

    """
    zs = _real_sequence(depth_centers, "depth_centers")
    b = _finite_real(beta, "beta")
    offset = _positive(z0, "z0")
    if b < 0.0:
        raise ValueError("beta must be non-negative")
    if min(zs) <= -offset:
        raise ValueError("depth_centers must exceed -z0")
    return tuple((z + offset) ** (-0.5 * b) for z in zs)


def sensitivity_depth_exponent(matrix: Matrix, depth_centers: Sequence[float]) -> float:
    """Return the log-log slope of the column norms of ``A`` against depth.

    A least-squares line through ``(log z_j, log ||A e_j||)`` over the cells
    with ``z_j > 0``; the sensitivity of a cell scales like ``z^slope`` and
    ``beta = -2 slope`` makes the depth weights of :func:`depth_weights` equal
    to the sensitivity decay [EQ-INV-APP-GRAV-003]. Surface effects (large
    cells, edges) make this a rough empirical guide.

    Args:
        matrix: ``A`` of shape ``m x n``.
        depth_centers: ``n`` cell-centre depths.

    Returns:
        The fitted slope (negative for a decaying sensitivity).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes disagree, fewer than two cells have positive
            depth or all depths are equal.

    References:
        [EQ-INV-APP-GRAV-003].

    """
    zs = _real_sequence(depth_centers, "depth_centers")
    if not matrix or len(matrix[0]) != len(zs):
        raise ValueError("depth_centers must have one entry per column of A")
    points = [
        (math.log(z), math.log(math.sqrt(sum(row[j] ** 2 for row in matrix))))
        for j, z in enumerate(zs)
        if z > 0.0
    ]
    if len(points) < 2:
        raise ValueError("at least two cells with positive depth are required")
    mx = math.fsum(p[0] for p in points) / len(points)
    my = math.fsum(p[1] for p in points) / len(points)
    sxx = math.fsum((p[0] - mx) ** 2 for p in points)
    if sxx == 0.0:
        raise ValueError("the depths must not all be equal")
    return math.fsum((p[0] - mx) * (p[1] - my) for p in points) / sxx


def depth_weighted_tikhonov(
    matrix: Matrix,
    data: Sequence[float],
    regularization: float,
    weights: Sequence[float],
) -> Vector:
    """Return the minimizer of ``||A x - y||^2 + lambda^2 ||W x||^2``.

    A thin wrapper of :func:`probstat.inverse.regularization.tikhonov` with
    the diagonal penalty ``W = diag(weights)`` (stacked QR, no normal
    equations) [EQ-INV-APP-GRAV-003]. Unit weights give ordinary Tikhonov.

    Args:
        matrix: ``A`` of shape ``m x n``.
        data: Data ``y`` of length ``m``.
        regularization: ``lambda > 0``.
        weights: ``n`` positive weights.

    Returns:
        The regularized density vector.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes disagree or a weight is not positive.

    References:
        [EQ-INV-APP-GRAV-003], [REF-TIKHONOV-1963].

    """
    w = _real_sequence(weights, "weights")
    if not matrix or len(w) != len(matrix[0]) or any(v <= 0.0 for v in w):
        raise ValueError("weights must be positive, one per column of A")
    penalty = tuple(
        tuple(w[i] if i == j else 0.0 for j in range(len(w))) for i in range(len(w))
    )
    return tikhonov(
        matrix, data, _positive(regularization, "regularization"), penalty=penalty
    ).solution


def centroid_depth(density: Sequence[float], depth_centers: Sequence[float]) -> float:
    """Return the ``|density|``-weighted mean depth of a density vector.

    Args:
        density: Cell densities (not all zero).
        depth_centers: Cell-centre depths, one per cell.

    Returns:
        ``sum |rho_j| z_j / sum |rho_j|``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes disagree or the density vanishes.

    References:
        [EQ-INV-APP-GRAV-003].

    """
    rho = _real_sequence(density, "density")
    zs = _real_sequence(depth_centers, "depth_centers")
    if len(rho) != len(zs):
        raise ValueError("density and depth_centers must have equal length")
    total = math.fsum(abs(r) for r in rho)
    if total == 0.0:
        raise ValueError("the density vanishes")
    return math.fsum(abs(r) * z for r, z in zip(rho, zs, strict=True)) / total


# ---------------------------------------------------------------------------
# Spectrum, null space and depth ambiguity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GravitySpectrum:
    """Singular spectrum of a gravity operator and what it resolves.

    Attributes:
        singular_values: The ``min(m, n)`` singular values of ``A``,
            descending.
        numerical_rank: Count above ``rank_tolerance * s_1``.
        null_dimension: ``n - numerical_rank``: the dimension of the null
            space (density patterns that produce exactly no data).
        resolved_modes: Number of singular values of ``A P^(1/2) / sigma``
            exceeding one (``P = prior_sd^2 I``): modes whose unit-prior
            signal exceeds the noise.
        condition_number: ``s_1 / s_r`` over the numerical rank ``r``.

    References:
        [EQ-INV-APP-GRAV-004].

    """

    singular_values: Vector
    numerical_rank: int
    null_dimension: int
    resolved_modes: int
    condition_number: float


def _svd_of_transpose(matrix: Matrix) -> tuple[Vector, Matrix]:
    """Return ``(s, V)`` of ``A`` from the tall SVD of ``A'``.

    ``V`` is ``n x r`` with the right singular vectors of ``A`` as columns.

    References:
        [EQ-INV-APP-GRAV-004].

    """
    svd = singular_value_decomposition(transpose(matrix))
    return tuple(svd.singular_values), svd.u


def gravity_spectrum(
    matrix: Matrix,
    noise_sd: float,
    *,
    prior_sd: float = 1.0,
    rank_tolerance: float = 1e-12,
) -> GravitySpectrum:
    """Measure the singular spectrum, rank and null dimension of ``A``.

    Args:
        matrix: ``A`` of shape ``m x n``.
        noise_sd: Noise standard deviation ``sigma > 0``.
        prior_sd: Unit-prior standard deviation (``> 0``) used to count the
            resolved modes.
        rank_tolerance: Relative threshold for the numerical rank.

    Returns:
        :class:`GravitySpectrum`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a scale is not positive or the matrix is empty.

    References:
        [EQ-INV-APP-GRAV-004].

    """
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    sigma = _positive(noise_sd, "noise_sd")
    sd = _positive(prior_sd, "prior_sd")
    tol = _positive(rank_tolerance, "rank_tolerance")
    sv, _ = _svd_of_transpose(matrix)
    rank = sum(1 for v in sv if v > tol * sv[0])
    return GravitySpectrum(
        singular_values=sv,
        numerical_rank=rank,
        null_dimension=len(matrix[0]) - rank,
        resolved_modes=sum(1 for v in sv if v * sd / sigma > 1.0),
        condition_number=sv[0] / sv[rank - 1],
    )


def null_space_element(
    matrix: Matrix, vector: Sequence[float], *, rank_tolerance: float = 1e-12
) -> Vector:
    """Return the projection of ``vector`` onto the null space of ``A``.

    ``N v = v - V_r V_r' v`` with ``V_r`` the right singular vectors above the
    rank tolerance. Adding any multiple of the result to a density leaves the
    data unchanged up to roundoff: the exact non-uniqueness
    [EQ-INV-APP-GRAV-004].

    Args:
        matrix: ``A`` of shape ``m x n``.
        vector: A vector of length ``n``.
        rank_tolerance: Relative threshold for the numerical rank.

    Returns:
        The null-space component.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes disagree.

    References:
        [EQ-INV-APP-GRAV-004].

    """
    v = _real_sequence(vector, "vector")
    if not matrix or len(matrix[0]) != len(v):
        raise ValueError("vector must have one entry per column of A")
    tol = _positive(rank_tolerance, "rank_tolerance")
    sv, basis = _svd_of_transpose(matrix)
    out = list(v)
    for k, s in enumerate(sv):
        if s <= tol * sv[0]:
            continue
        coeff = math.fsum(basis[i][k] * v[i] for i in range(len(v)))
        for i in range(len(v)):
            out[i] -= coeff * basis[i][k]
    return tuple(out)


def distinguishability(
    matrix: Matrix,
    noise_sd: float,
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return ``||A (x_1 - x_2)|| / sigma`` for iid noise.

    The whitened data difference of two densities, in noise units: a value
    well below one means the observations cannot tell them apart, whatever
    the method [EQ-INV-APP-GRAV-004].

    Args:
        matrix: ``A`` of shape ``m x n``.
        noise_sd: Noise standard deviation ``> 0``.
        first: First density (length ``n``).
        second: Second density (length ``n``).

    Returns:
        The whitened distance of the two predictions.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes disagree.

    References:
        [EQ-INV-APP-GRAV-004].

    """
    a = _real_sequence(first, "first")
    b = _real_sequence(second, "second")
    if not matrix or len(matrix[0]) != len(a) or len(a) != len(b):
        raise ValueError("densities must have one entry per column of A")
    diff = matvec(matrix, tuple(p - q for p, q in zip(a, b, strict=True)))
    return math.hypot(*diff) / _positive(noise_sd, "noise_sd")


def single_layer_fit(
    matrix: Matrix,
    data: Sequence[float],
    x_cells: int,
    layer: int,
    damping: float,
) -> tuple[Vector, float]:
    """Fit the data with the density of ONE depth layer only.

    The minimizer of ``||A_L x - y||^2 + damping^2 ||x||^2`` over the cells
    of layer ``layer`` (the equivalent-layer idea: a single sheet at any depth
    reproduces a surface field up to resolution), returning the full-length
    density (zero outside the layer) and the data residual norm. A residual at
    the noise level means the data cannot rule that depth out
    [EQ-INV-APP-GRAV-005].

    Args:
        matrix: ``A`` of shape ``m x n`` with depth-major cell order.
        data: Data ``y`` (length ``m``).
        x_cells: Number of cells per layer ``n_x``.
        layer: Layer index ``0 <= layer < n / n_x``.
        damping: ``damping > 0`` (an explicit ridge, never added silently).

    Returns:
        ``(density, residual_norm)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the sizes or the layer index are invalid.

    References:
        [EQ-INV-APP-GRAV-005], [REF-BLAKELY-1995].

    """
    nx = _positive_int(x_cells, "x_cells")
    if not matrix or len(matrix[0]) % nx != 0:
        raise ValueError("the number of columns must be a multiple of x_cells")
    layers = len(matrix[0]) // nx
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < layers:
        raise ValueError("layer must be an integer in [0, layers)")
    cols = range(layer * nx, (layer + 1) * nx)
    sub = tuple(tuple(row[j] for j in cols) for row in matrix)
    sol = tikhonov(sub, data, _positive(damping, "damping")).solution
    full = [0.0] * len(matrix[0])
    for k, j in enumerate(cols):
        full[j] = sol[k]
    resid = [a - b for a, b in zip(matvec(matrix, tuple(full)), data, strict=True)]
    return tuple(full), math.hypot(*resid)


# ---------------------------------------------------------------------------
# Benchmark G01
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GravityBenchmark:
    """G01: one 2D gravity inversion instance (solver inputs plus truth).

    The first fields mirror
    :class:`~probstat.inverse.benchmark_problems.InverseBenchmark`; the last
    ones describe the inversion mesh.

    Attributes:
        identifier: ``"G01"``.
        name: Short description.
        operator: ``A`` (``m x n``) of the INVERSION mesh given to the solver.
        data: Noisy data ``y``.
        noise: Noise covariance factor.
        noise_level: Noise standard deviation.
        realized_noise_norm: ``||y - b||`` of this draw (not rescaled).
        truth: Withheld density on the inversion mesh (block means of the fine
            truth).
        exact_data: Withheld noise-free data from the FINE cells.
        same_discretization: ``True`` only for ``refinement = 1`` (inverse
            crime).
        prior_covariance: Exponential prior covariance of the cell densities,
            or ``None``.
        singular_values: Singular values of ``A``.
        pathologies: Expected difficulties.
        seeds: Stream paths.
        matrix: The dense ``A``.
        stations: Observation positions.
        height: Observation height.
        x_edges: Horizontal edges of the inversion mesh.
        z_edges: Depth edges of the inversion mesh.
        depth_centers: Cell-centre depths (depth-major order).
        refinement: Fine cells per coarse cell in each direction.
        truth_name: The named truth.

    References:
        [EQ-INV-APP-GRAV-002].

    """

    identifier: str
    name: str
    operator: DenseOperator
    data: Vector
    noise: ScaledIdentityCovariance
    noise_level: float
    realized_noise_norm: float
    truth: Vector
    exact_data: Vector
    same_discretization: bool
    prior_covariance: DenseCovariance | None
    singular_values: Vector
    pathologies: tuple[str, ...]
    matrix: Matrix
    stations: tuple[float, ...]
    height: float
    x_edges: tuple[float, ...]
    z_edges: tuple[float, ...]
    depth_centers: Vector
    refinement: int
    truth_name: str
    seeds: Mapping[str, str] = field(default_factory=dict)


def _truth_density(name: str, x: float, z: float, width: float, depth: float) -> float:
    """Return the named synthetic density at a point.

    References:
        [EQ-INV-APP-GRAV-002].

    """
    u, v = x / width, z / depth
    if name == "deep_block":
        return 1.0 if 0.55 <= u < 0.77 and 0.5 <= v < 0.82 else 0.0
    if name == "two_bodies":
        shallow = 1.0 if 0.12 <= u < 0.29 and 0.04 <= v < 0.27 else 0.0
        deep = 0.6 if 0.56 <= u < 0.81 and 0.46 <= v < 0.86 else 0.0
        return shallow + deep
    return math.exp(-(((u - 0.5) / 0.15) ** 2 + ((v - 0.45) / 0.2) ** 2))


def _centers(edges: Sequence[float]) -> tuple[float, ...]:
    """Return the midpoints of consecutive edges.

    References:
        [EQ-INV-APP-GRAV-002].

    """
    return tuple(0.5 * (a + b) for a, b in zip(edges, edges[1:], strict=False))


def gravity_inversion_benchmark(
    *,
    x_cells: int = 12,
    z_cells: int = 5,
    stations: int = 16,
    width: float = 4.0,
    depth: float = 1.0,
    height: float = 0.05,
    truth: str = "two_bodies",
    refinement: int = 3,
    relative_noise: float = 0.02,
    prior_sd: float = 0.5,
    prior_length_scale: float | None = 0.8,
    seed: int | RandomStream | None = 11,
) -> GravityBenchmark:
    """Return G01: a SYNTHETIC 2D gravity-anomaly inversion with finer truth.

    The inversion mesh has ``x_cells x z_cells`` uniform cells on
    ``[0, width] x [0, depth]``; the data are the vertical attractions at
    ``stations`` equispaced surface points (height ``height``) of a truth
    defined on a FINER mesh (``refinement`` times as many cells per direction,
    cell-centre sampling of a piecewise or smooth analytic density that does
    not align with the coarse cells), so the inverse crime is avoided unless
    ``refinement = 1`` [EQ-INV-APP-GRAV-002]. Noise is iid Gaussian at a level
    relative to ``||b|| / sqrt(m)`` and is not rescaled. The solver is given
    ``A``, ``y``, the noise and (optionally) an exponential prior covariance
    ``v exp(-d / l)`` between cell centres; the truth stays in the
    result for scoring. The case is built to be non-unique: ``n > m`` (exact
    null space) and a rapidly decaying spectrum (depth/density ambiguity), see
    :func:`gravity_spectrum`. SYNTHETIC: no real field or interior is
    represented.

    Args:
        x_cells: Horizontal inversion cells (``>= 2``).
        z_cells: Depth layers (``>= 2``).
        stations: Observation points (``>= 2``).
        width: Horizontal extent ``> 0``.
        depth: Model depth ``> 0``.
        height: Observation height ``>= 0``.
        truth: ``"two_bodies"``, ``"deep_block"`` or ``"smooth_anomaly"``.
        refinement: Fine cells per coarse cell per direction (``>= 1``).
        relative_noise: Noise level ``> 0`` relative to ``||b|| / sqrt(m)``.
        prior_sd: Prior standard deviation ``> 0`` (used when a prior is
            built).
        prior_length_scale: Correlation length ``> 0`` of the exponential
            prior, or ``None`` for no prior.
        seed: Root seed (only the noise is random).

    Returns:
        :class:`GravityBenchmark`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or the truth is unknown.

    References:
        [EQ-INV-APP-GRAV-002], [REF-BLAKELY-1995],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    if truth not in _TRUTHS:
        raise ValueError(f"truth must be one of {_TRUTHS}")
    nx = _positive_int(x_cells, "x_cells", 2)
    nz = _positive_int(z_cells, "z_cells", 2)
    m = _positive_int(stations, "stations", 2)
    r = _positive_int(refinement, "refinement")
    w = _positive(width, "width")
    d = _positive(depth, "depth")
    h = _finite_real(height, "height")
    if h < 0.0:
        raise ValueError("height must be non-negative")
    level = _positive(relative_noise, "relative_noise")
    sd = _positive(prior_sd, "prior_sd")
    ex = tuple(w * j / nx for j in range(nx + 1))
    ez = tuple(d * k / nz for k in range(nz + 1))
    fx = tuple(w * j / (nx * r) for j in range(nx * r + 1))
    fz = tuple(d * k / (nz * r) for k in range(nz * r + 1))
    xs = tuple(w * (i + 0.5) / m for i in range(m))
    a_coarse = gravity_kernel_matrix(xs, ex, ez, height=h)
    a_fine = gravity_kernel_matrix(xs, fx, fz, height=h)
    cx, cz = _centers(fx), _centers(fz)
    fine = tuple(_truth_density(truth, x, z, w, d) for z in cz for x in cx)
    nxf = nx * r
    coarse = tuple(
        math.fsum(
            fine[(k * r + dz) * nxf + (j * r + dx)]
            for dz in range(r)
            for dx in range(r)
        )
        / (r * r)
        for k in range(nz)
        for j in range(nx)
    )
    exact = matvec(a_fine, fine)
    stream = resolve_stream(seed).child("G01")
    data, sigma, realized = _noise(exact, level, stream)
    prior: DenseCovariance | None = None
    centers_x, centers_z = _centers(ex), _centers(ez)
    cells = [(x, z) for z in centers_z for x in centers_x]
    if prior_length_scale is not None:
        ell = _positive(prior_length_scale, "prior_length_scale")
        prior = DenseCovariance(
            [
                [
                    sd * sd * math.exp(-math.hypot(p[0] - q[0], p[1] - q[1]) / ell)
                    for q in cells
                ]
                for p in cells
            ]
        )
    sv, _ = _svd_of_transpose(a_coarse)
    return GravityBenchmark(
        identifier="G01",
        name=(
            f"2D gravity anomaly ({truth}, {nx}x{nz} cells, {m} stations, "
            f"refinement {r})"
        ),
        operator=DenseOperator(a_coarse),
        data=data,
        noise=ScaledIdentityCovariance(m, sigma * sigma),
        noise_level=sigma,
        realized_noise_norm=realized,
        truth=coarse,
        exact_data=exact,
        same_discretization=r == 1,
        prior_covariance=prior,
        singular_values=sv,
        pathologies=(
            "underdetermined",
            "nullspace",
            "depth_nonuniqueness",
            "severely_ill_conditioned",
        ),
        matrix=a_coarse,
        stations=xs,
        height=h,
        x_edges=ex,
        z_edges=ez,
        depth_centers=tuple(z for z in centers_z for _ in centers_x),
        refinement=r,
        truth_name=truth,
        seeds=MappingProxyType({"noise": "G01/noise"}),
    )


GRAVITY_BENCHMARK_CATALOG: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "G01",
        "2D gravity anomaly",
        "inverse.benchmark_gravity.gravity_inversion_benchmark",
        "exact prism primitive checked against independent quadrature; "
        "finer-cell truth",
        (
            "underdetermined",
            "nullspace",
            "depth_nonuniqueness",
            "severely_ill_conditioned",
        ),
        (
            "depth_weighted_tikhonov",
            "linear_gaussian_posterior",
            "spectral_diagnostics",
            "regularization_path",
        ),
    ),
)
"""Catalog entries of this module (roadmap 20C gravity, synthetic)."""
