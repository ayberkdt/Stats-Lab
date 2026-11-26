"""Extended inverse benchmark cases: images, tomography, nonlinear and PDE models.

Experimental (added in 0.28). Continues the catalog of
:mod:`probstat.inverse.benchmark_problems` (L01-L06) with

- L07a :func:`image_deblur_benchmark`: 2D separable Gaussian blur of a
  synthetic phantom with an exact matrix-free transpose;
- L07b :func:`masked_tomography_benchmark`: parallel-beam line integrals from
  exact ray-pixel intersection lengths with a matched scatter-add transpose;
- N01 :func:`exponential_decay_benchmark`: ``y_i = a exp(-r t_i)`` with
  analytic JVP/VJP and an identifiable or a weak design;
- N02 :func:`squared_map_benchmark`: elementwise square and pairwise product
  maps with structural multimodality;
- P01 :func:`elliptic_benchmark`: log-conductivity inversion on
  :class:`~probstat.inverse.pde_models.EllipticDiffusion1D` with data from a
  finer mesh;
- P02 :func:`heat_initial_condition_benchmark`: initial-temperature inversion
  on :class:`~probstat.inverse.heat_models.HeatEquation1D` with data from
  more time steps.

Linear cases return the existing
:class:`~probstat.inverse.benchmark_problems.InverseBenchmark`; nonlinear
cases return :class:`NonlinearBenchmark`, the same contract with a
:class:`~probstat.inverse.gauss_newton.DifferentiableForward` model in place
of the operator. Seeds for truth and noise come from distinct children of one
:class:`~probstat.foundations.RandomStream`, and ``same_discretization`` marks
data generated with the very discretization the solver inverts (algebraic
verification only) versus an independent finer model
[EQ-INV-BENCH2-008].

The tomography operator is a SMALL DISCRETIZED PROTOTYPE: each ray is cut by
the pixel boundaries and the exact intersection lengths are stored, so the
discrete operator is exact for piecewise-constant images on the pixel grid and
its transpose is the exact adjoint. It is not a CT-grade projector (no
detector model, no fan or cone geometry, no attenuation physics, no
sub-pixel interpolation) [EQ-INV-BENCH2-002]. The Siddon ray tracing idea is
only named here; no bibliography entry for it exists in the project, so no
claim is attributed to it.

Experimental (added in 0.28).

References:
    [REF-HANSEN-2008], [REF-GAZZOLA-HANSEN-NAGY-2019],
    [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.benchmark_problems import (
    BenchmarkCase,
    InverseBenchmark,
    _gaussian_kernel,
    _noise,
    _positive_int,
)
from probstat.inverse.gauss_newton import CallbackForward, DifferentiableForward
from probstat.inverse.heat_models import HeatEquation1D, heat_semigroup_singular_values
from probstat.inverse.pde_models import EllipticDiffusion1D
from probstat.linalg import (
    DenseCovariance,
    OperatorBase,
    ScaledIdentityCovariance,
    Vector,
)

_LINE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class NonlinearBenchmark:
    """One nonlinear benchmark instance: solver inputs plus the withheld truth.

    Attributes:
        identifier: Catalog ID (``"N01"``...).
        name: Short description.
        model: Differentiable forward map ``G`` given to the solver.
        data: Noisy data ``y`` given to the solver.
        noise: Noise covariance factor given to the solver.
        noise_level: Noise standard deviation (scalar, for iid noise).
        realized_noise_norm: ``||y - b||`` of this draw (reported, not
            rescaled).
        truth: Withheld true parameters on the solver's parametrization.
        exact_data: Withheld noise-free data ``b``.
        same_discretization: ``True`` when ``b = G(x_true)`` exactly with the
            solver's model (inverse crime, algebraic verification only).
        prior_mean: Prior mean for Bayesian cases, else ``None``.
        prior_covariance: Prior covariance factor for Bayesian cases, else
            ``None``.
        pathologies: Expected difficulties (e.g. ``"multimodal"``).
        seeds: The stream paths used for ``truth`` and ``noise``.

    References:
        [EQ-INV-BENCH2-008], [REF-KAIPIO-SOMERSALO-2007].

    """

    identifier: str
    name: str
    model: DifferentiableForward
    data: Vector
    noise: ScaledIdentityCovariance
    noise_level: float
    realized_noise_norm: float
    truth: Vector
    exact_data: Vector
    same_discretization: bool
    prior_mean: Vector | None
    prior_covariance: DenseCovariance | None
    pathologies: tuple[str, ...]
    seeds: Mapping[str, str] = field(default_factory=dict)


def _real_positive(value: object, name: str) -> float:
    """Validate a finite positive real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


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


def _exponential_prior(
    points: Sequence[float], variance: float, length_scale: float
) -> DenseCovariance:
    """Return the exponential-kernel covariance ``v exp(-|s - s'| / l)``.

    References:
        [EQ-INV-BENCH2-006], [REF-KAIPIO-SOMERSALO-2007].

    """
    return DenseCovariance(
        [
            [variance * math.exp(-abs(a - b) / length_scale) for b in points]
            for a in points
        ]
    )


# ---------------------------------------------------------------------------
# L07a: 2D image deblurring
# ---------------------------------------------------------------------------


def _correlate_line(
    line: Sequence[float], kernel: Sequence[float], *, transpose: bool
) -> list[float]:
    """Apply a zero-boundary 1D correlation (or its transpose) to one line.

    With ``half = (len(kernel) - 1) / 2`` the matrix is
    ``A[i, i + d] = kernel[d + half]``; its transpose is the correlation with
    the reversed kernel.

    References:
        [EQ-INV-BENCH2-001].

    """
    n = len(line)
    half = (len(kernel) - 1) // 2
    weights = list(reversed(kernel)) if transpose else list(kernel)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        total = 0.0
        for j in range(lo, hi + 1):
            total += weights[j - i + half] * line[j]
        out[i] = total
    return out


class GaussianBlur2DOperator(OperatorBase):
    """Separable 2D Gaussian blur on a ``rows x cols`` image with zero boundary.

    ``A = B_r (x) B_c`` acts on row-major vectors (pixel ``(i, j)`` is entry
    ``i * cols + j``): the 1D blur ``B_c`` along every row, then ``B_r`` along
    every column, each with zero values outside the image
    [EQ-INV-BENCH2-001]. The image is the unit square; the kernel standard
    deviation ``width`` is physical, so the pixel spacings are ``1 / rows`` and
    ``1 / cols``. ``rmatvec`` applies the reversed 1D kernels in the same two
    passes, which is the exact Euclidean transpose; no dense matrix is formed.

    Attributes:
        rows: Number of pixel rows.
        cols: Number of pixel columns.
        width: Kernel standard deviation in physical units.

    References:
        [EQ-INV-BENCH2-001], [REF-HANSEN-2008], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """

    __slots__ = ("_col_kernel", "_row_kernel", "cols", "rows", "width")

    def __init__(self, rows: int, cols: int, *, width: float = 0.05) -> None:
        """Build the operator.

        Args:
            rows: Number of rows ``>= 2``.
            cols: Number of columns ``>= 2``.
            width: Positive physical kernel standard deviation.

        Raises:
            ValueError: If a dimension is not an integer ``>= 2`` or ``width``
                is not positive and finite.
            TypeError: If ``width`` is not a real number.

        References:
            [EQ-INV-BENCH2-001].

        """
        m = _positive_int(rows, "rows", 2)
        n = _positive_int(cols, "cols", 2)
        sigma = _real_positive(width, "width")
        self.rows = m
        self.cols = n
        self.width = sigma
        self._row_kernel = _gaussian_kernel(sigma, 1.0 / m)
        self._col_kernel = _gaussian_kernel(sigma, 1.0 / n)
        super().__init__(m * n, m * n, name="2D Gaussian blur")

    @property
    def kernel_half_widths(self) -> tuple[int, int]:
        """Return the kernel half-widths ``(rows direction, columns direction)``.

        Pixels at least this far from every image edge see the full kernel.

        Returns:
            The two half-widths in pixels.

        References:
            [EQ-INV-BENCH2-001].

        """
        return (len(self._row_kernel) - 1) // 2, (len(self._col_kernel) - 1) // 2

    def _apply(self, vector: Vector, *, transpose: bool) -> list[float]:
        """Apply the two separable passes.

        References:
            [EQ-INV-BENCH2-001].

        """
        m, n = self.rows, self.cols
        grid = [
            _correlate_line(
                vector[i * n : (i + 1) * n], self._col_kernel, transpose=transpose
            )
            for i in range(m)
        ]
        out = [0.0] * (m * n)
        for j in range(n):
            column = _correlate_line(
                [grid[i][j] for i in range(m)], self._row_kernel, transpose=transpose
            )
            for i in range(m):
                out[i * n + j] = column[i]
        return out

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Apply ``A`` (blur along rows, then columns).

        References:
            [EQ-INV-BENCH2-001].

        """
        return self._apply(vector, transpose=False)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Apply ``A'`` (reversed kernels).

        References:
            [EQ-INV-BENCH2-001].

        """
        return self._apply(vector, transpose=True)


def _phantom_value(y: float, x: float) -> float:
    """Return the analytic phantom at ``(y, x)`` in the unit square.

    A bright rectangle ``[0.2, 0.5] x [0.15, 0.45]`` (value 1), a dim rectangle
    ``[0.1, 0.3] x [0.55, 0.85]`` (value 0.6) and a disc of radius 0.2 centred
    at ``(0.65, 0.65)`` (value 0.8, drawn on top).

    References:
        [EQ-INV-BENCH2-001].

    """
    if (y - 0.65) ** 2 + (x - 0.65) ** 2 <= 0.2**2:
        return 0.8
    if 0.2 <= y <= 0.5 and 0.15 <= x <= 0.45:
        return 1.0
    if 0.1 <= y <= 0.3 and 0.55 <= x <= 0.85:
        return 0.6
    return 0.0


def _phantom_image(rows: int, cols: int) -> list[float]:
    """Sample the phantom at the pixel centres (row-major).

    References:
        [EQ-INV-BENCH2-001].

    """
    return [
        _phantom_value((i + 0.5) / rows, (j + 0.5) / cols)
        for i in range(rows)
        for j in range(cols)
    ]


def _block_average(
    fine: Sequence[float], rows: int, cols: int, factor: int
) -> tuple[float, ...]:
    """Average ``factor x factor`` blocks of a row-major fine image.

    References:
        [EQ-INV-BENCH2-001].

    """
    fine_cols = cols * factor
    out = []
    for i in range(rows):
        for j in range(cols):
            block = [
                fine[(i * factor + a) * fine_cols + j * factor + b]
                for a in range(factor)
                for b in range(factor)
            ]
            out.append(math.fsum(block) / (factor * factor))
    return tuple(out)


def image_deblur_benchmark(
    rows: int = 16,
    cols: int = 16,
    *,
    width: float = 0.05,
    relative_noise: float = 1e-2,
    refinement: int = 1,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L07a: 2D Gaussian deblurring of a synthetic phantom.

    The solver's operator is :class:`GaussianBlur2DOperator` on the
    ``rows x cols`` grid. With ``refinement = 1`` the data are ``A x_true``
    (inverse crime). With ``refinement = r > 1`` the analytic phantom is
    sampled on an ``r rows x r cols`` grid, blurred there with the same
    physical kernel and zero boundary, and block-averaged to the coarse grid;
    the solver's truth is the block average of the fine phantom, so the data
    carry a genuine discretization error [EQ-INV-BENCH2-001]. Vectors are
    row-major.

    Args:
        rows: Coarse number of rows ``>= 2``.
        cols: Coarse number of columns ``>= 2``.
        width: Physical kernel standard deviation on the unit square.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        refinement: Integer factor ``r >= 1`` of the data-generating grid.
        seed: Root seed (only the noise is random; the phantom is analytic).

    Returns:
        :class:`~probstat.inverse.benchmark_problems.InverseBenchmark`.

    Raises:
        ValueError: On out-of-range dimensions, ``width``, ``refinement`` or
            ``relative_noise``.

    References:
        [EQ-INV-BENCH2-001], [REF-GAZZOLA-HANSEN-NAGY-2019],
        [REF-HANSEN-2008].

    """
    m = _positive_int(rows, "rows", 2)
    n = _positive_int(cols, "cols", 2)
    r = _positive_int(refinement, "refinement", 1)
    operator = GaussianBlur2DOperator(m, n, width=width)
    if r == 1:
        truth = tuple(_phantom_image(m, n))
        exact = operator.matvec(truth)
    else:
        fine_rows, fine_cols = r * m, r * n
        fine_truth = _phantom_image(fine_rows, fine_cols)
        truth = _block_average(fine_truth, m, n, r)
        fine_operator = GaussianBlur2DOperator(fine_rows, fine_cols, width=width)
        fine_data = fine_operator.matvec(fine_truth)
        exact = _block_average(fine_data, m, n, r)
    stream = resolve_stream(seed).child("L07a")
    data, level, realized = _noise(exact, relative_noise, stream)
    return InverseBenchmark(
        identifier="L07a",
        name=f"2D deblurring ({m}x{n}, refinement {r})",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(m * n, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=r == 1,
        prior_covariance=None,
        singular_values=None,
        pathologies=("ill_conditioned", "edges"),
        seeds=MappingProxyType({"noise": "L07a/noise"}),
    )


# ---------------------------------------------------------------------------
# L07b: masked tomography prototype
# ---------------------------------------------------------------------------


def ray_pixel_intersections(
    size: int, angle: float, offset: float
) -> tuple[tuple[int, float], ...]:
    """Return the exact intersection lengths of one ray with a pixel grid.

    The grid is ``[0, size]^2`` with unit pixels; pixel ``(iy, ix)`` has index
    ``iy * size + ix``. The ray has direction ``(cos angle, sin angle)`` and
    passes through ``centre + offset (-sin angle, cos angle)`` with
    ``centre = (size / 2, size / 2)``. The parameters at which the ray enters
    and leaves the square and crosses every vertical and horizontal grid line
    are sorted; consecutive parameters bound segments inside a single pixel,
    whose length is the parameter difference (the direction is a unit vector)
    [EQ-INV-BENCH2-002]. A ray lying exactly on a grid line is assigned wholly
    to the pixel row or column of larger index (clamped at the grid edge); a
    ray that misses the square returns an empty tuple.

    Args:
        size: Pixels per side ``>= 1``.
        angle: Direction angle in radians.
        offset: Signed distance of the ray from the grid centre.

    Returns:
        ``(pixel index, length)`` pairs with positive lengths.

    Raises:
        ValueError: If ``size`` is not a positive integer or an angle or
            offset is not finite.

    References:
        [EQ-INV-BENCH2-002], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """
    n = _positive_int(size, "size", 1)
    theta = _finite_real(angle, "angle")
    shift = _finite_real(offset, "offset")
    c, s = math.cos(theta), math.sin(theta)
    centre = 0.5 * n
    origin = (centre - shift * s, centre + shift * c)
    direction = (c, s)
    lower = -math.inf
    upper = math.inf
    for axis in (0, 1):
        d = direction[axis]
        p = origin[axis]
        if abs(d) <= _LINE_TOLERANCE:
            if not 0.0 <= p <= n:
                return ()
            continue
        a, b = (0.0 - p) / d, (n - p) / d
        lower = max(lower, min(a, b))
        upper = min(upper, max(a, b))
    if not upper - lower > _LINE_TOLERANCE:
        return ()
    cuts = [lower, upper]
    for axis in (0, 1):
        d = direction[axis]
        if abs(d) <= _LINE_TOLERANCE:
            continue
        for k in range(1, n):
            t = (k - origin[axis]) / d
            if lower < t < upper:
                cuts.append(t)
    cuts.sort()
    segments: list[tuple[int, float]] = []
    for t0, t1 in zip(cuts[:-1], cuts[1:], strict=True):
        length = t1 - t0
        if length <= _LINE_TOLERANCE:
            continue
        mid = 0.5 * (t0 + t1)
        ix = min(max(math.floor(origin[0] + mid * c), 0), n - 1)
        iy = min(max(math.floor(origin[1] + mid * s), 0), n - 1)
        segments.append((iy * n + ix, length))
    return tuple(segments)


def parallel_beam_rays(
    size: int,
    angles: int | Sequence[float],
    rays_per_angle: int,
    *,
    angle_span: float = math.pi,
) -> tuple[tuple[float, float], ...]:
    """Return ``(angle, offset)`` pairs of a parallel-beam acquisition.

    An integer ``angles`` means that many equispaced angles
    ``a * angle_span / angles`` (``a = 0..angles-1``); a sequence is used as
    given. At angle ``theta`` the detector spans the projected width
    ``W = size (|cos| + |sin|)`` of the square and the ``R`` ray offsets are
    ``((k + 1/2) / R - 1/2) W``, so every ray cuts the square with a positive
    chord. Choose ``angle_span < pi`` for a limited-angle (masked) problem
    [EQ-INV-BENCH2-002].

    Args:
        size: Pixels per side ``>= 2``.
        angles: Number of angles ``>= 1`` or the angles in radians.
        rays_per_angle: Number of rays per angle ``>= 1``.
        angle_span: Angular range ``(0, pi]`` of an integer ``angles``.

    Returns:
        The rays, angle-major.

    Raises:
        ValueError: On out-of-range arguments.

    References:
        [EQ-INV-BENCH2-002], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """
    n = _positive_int(size, "size", 2)
    count = _positive_int(rays_per_angle, "rays_per_angle", 1)
    if isinstance(angles, int) and not isinstance(angles, bool):
        number = _positive_int(angles, "angles", 1)
        span = _real_positive(angle_span, "angle_span")
        if span > math.pi:
            raise ValueError("angle_span must not exceed pi")
        values = [a * span / number for a in range(number)]
    else:
        if isinstance(angles, (str, bytes)) or not isinstance(angles, Sequence):
            raise TypeError("angles must be an integer or a sequence of reals")
        values = [_finite_real(a, "angle") for a in angles]
        if not values:
            raise ValueError("angles must not be empty")
    rays = []
    for theta in values:
        width = n * (abs(math.cos(theta)) + abs(math.sin(theta)))
        for k in range(count):
            rays.append((theta, ((k + 0.5) / count - 0.5) * width))
    return tuple(rays)


class RayTransformOperator(OperatorBase):
    """Discrete line-integral operator from exact ray-pixel intersection lengths.

    Row ``k`` of ``A`` holds, for the ``k``-th ray, the intersection length with
    every pixel it crosses (stored sparsely); ``matvec`` is the weighted sum of
    pixel values per ray and ``rmatvec`` scatter-adds the same lengths, so the
    transpose is exact [EQ-INV-BENCH2-002]. A small discretized prototype, not
    a CT-grade projector.

    Attributes:
        size: Pixels per side.
        rays: The ``(angle, offset)`` pairs, one per row.

    References:
        [EQ-INV-BENCH2-002], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """

    __slots__ = ("_segments", "rays", "size")

    def __init__(self, size: int, rays: Sequence[tuple[float, float]]) -> None:
        """Trace every ray once and store the sparse rows.

        Args:
            size: Pixels per side ``>= 2``.
            rays: Non-empty sequence of ``(angle, offset)`` pairs.

        Raises:
            ValueError: If ``rays`` is empty or malformed, or ``size`` is
                invalid.

        References:
            [EQ-INV-BENCH2-002].

        """
        n = _positive_int(size, "size", 2)
        if isinstance(rays, (str, bytes)) or not isinstance(rays, Sequence):
            raise TypeError("rays must be a sequence of (angle, offset) pairs")
        if not rays:
            raise ValueError("rays must not be empty")
        checked: list[tuple[float, float]] = []
        for ray in rays:
            if len(ray) != 2:
                raise ValueError("each ray must be an (angle, offset) pair")
            checked.append(
                (_finite_real(ray[0], "angle"), _finite_real(ray[1], "offset"))
            )
        self.size = n
        self.rays = tuple(checked)
        self._segments = tuple(ray_pixel_intersections(n, a, t) for a, t in checked)
        super().__init__(len(checked), n * n, name="ray transform")

    @property
    def segments(self) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Return the sparse rows: ``(pixel, length)`` pairs per ray.

        Returns:
            One tuple of pairs per ray.

        References:
            [EQ-INV-BENCH2-002].

        """
        return self._segments

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return the line integrals of the pixel values.

        References:
            [EQ-INV-BENCH2-002].

        """
        return [
            math.fsum(length * vector[pixel] for pixel, length in row)
            for row in self._segments
        ]

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Scatter-add the same lengths (back-projection).

        References:
            [EQ-INV-BENCH2-002].

        """
        out = [0.0] * (self.size * self.size)
        for row, value in zip(self._segments, vector, strict=True):
            for pixel, length in row:
                out[pixel] += length * value
        return out


# Additive discs in fractions of the grid side: (cx, cy, radius, value).
_DISCS = ((0.35, 0.40, 0.20, 1.0), (0.65, 0.65, 0.15, 0.5), (0.70, 0.30, 0.08, 0.8))


def _disc_phantom(size: int, subsamples: int = 4) -> tuple[float, ...]:
    """Return the pixel averages of the additive disc phantom (row-major).

    Each pixel is the mean over ``subsamples^2`` sub-cell centres.

    References:
        [EQ-INV-BENCH2-003].

    """
    values = []
    for iy in range(size):
        for ix in range(size):
            total = 0.0
            for a in range(subsamples):
                for b in range(subsamples):
                    y = (iy + (a + 0.5) / subsamples) / size
                    x = (ix + (b + 0.5) / subsamples) / size
                    for cx, cy, radius, value in _DISCS:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                            total += value
            values.append(total / (subsamples * subsamples))
    return tuple(values)


def _analytic_line_integral(size: int, angle: float, offset: float) -> float:
    """Return the exact line integral of the disc phantom along one ray.

    A disc of radius ``R`` at distance ``q`` from the line contributes
    ``value * 2 sqrt(R^2 - q^2)`` (the discs lie inside the square)
    [EQ-INV-BENCH2-003].

    References:
        [EQ-INV-BENCH2-003].

    """
    c, s = math.cos(angle), math.sin(angle)
    centre = 0.5 * size
    px, py = centre - offset * s, centre + offset * c
    total = 0.0
    for cx, cy, radius, value in _DISCS:
        dist = abs((cx * size - px) * s - (cy * size - py) * c)
        reach = radius * size
        if dist < reach:
            total += value * 2.0 * math.sqrt(reach * reach - dist * dist)
    return total


def masked_tomography_benchmark(
    size: int = 12,
    angles: int | Sequence[float] = 12,
    *,
    rays_per_angle: int = 12,
    angle_span: float = math.pi,
    analytic: bool = False,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return L07b: a small parallel-beam tomography prototype.

    The operator is :class:`RayTransformOperator` built from exact ray-pixel
    intersection lengths of a parallel-beam geometry
    (:func:`parallel_beam_rays`). The phantom is a sum of three discs
    (pixel averages of the analytic phantom). This is a SMALL DISCRETIZED
    PROTOTYPE, not a CT-grade projector: no detector model, no fan or cone
    beam, no sub-pixel interpolation of the rays.

    With ``analytic=False`` the data are ``A x_true`` from the SAME
    discretization (``same_discretization=True``, algebra verification only).
    With ``analytic=True`` the data are the exact analytic line integrals of
    the discs, independent of the pixel grid, so the discretization error of
    the piecewise-constant pixel model is present
    (``same_discretization=False``) [EQ-INV-BENCH2-003]. Choose
    ``angle_span < pi`` for a limited-angle (masked) problem; rays are
    stored in angle-major order.

    Args:
        size: Pixels per side ``>= 4``.
        angles: Number of angles or the angles in radians.
        rays_per_angle: Rays per angle.
        angle_span: Angular range ``(0, pi]`` for an integer ``angles``.
        analytic: Generate data from analytic disc line integrals.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed (only the noise is random).

    Returns:
        :class:`~probstat.inverse.benchmark_problems.InverseBenchmark`.

    Raises:
        ValueError: On out-of-range arguments.

    References:
        [EQ-INV-BENCH2-002], [EQ-INV-BENCH2-003],
        [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-HANSEN-2008].

    """
    n = _positive_int(size, "size", 4)
    rays = parallel_beam_rays(n, angles, rays_per_angle, angle_span=angle_span)
    operator = RayTransformOperator(n, rays)
    truth = _disc_phantom(n)
    if analytic:
        exact = tuple(_analytic_line_integral(n, a, t) for a, t in rays)
    else:
        exact = operator.matvec(truth)
    stream = resolve_stream(seed).child("L07b")
    data, level, realized = _noise(exact, relative_noise, stream)
    m = len(rays)
    limited = isinstance(angles, int) and angle_span < math.pi
    pathologies = ("ill_conditioned", "nullspace") + (
        ("limited_angle",) if limited else ()
    )
    return InverseBenchmark(
        identifier="L07b",
        name=f"parallel-beam tomography prototype ({n}x{n}, {m} rays)",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(m, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=not analytic,
        prior_covariance=None,
        singular_values=None,
        pathologies=pathologies,
        seeds=MappingProxyType({"noise": "L07b/noise"}),
    )


# ---------------------------------------------------------------------------
# N01: exponential decay
# ---------------------------------------------------------------------------


def exponential_decay_benchmark(
    times: Sequence[float] | None = None,
    amplitude: float = 2.0,
    rate: float = 1.0,
    *,
    count: int = 12,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
    design: str | None = None,
) -> NonlinearBenchmark:
    """Return N01: ``y_i = a exp(-r t_i) + e_i`` with analytic derivatives.

    The parameters are ``x = (a, r)``. The Jacobian rows are
    ``(exp(-r t_i), -a t_i exp(-r t_i))`` [EQ-INV-BENCH2-004]. Without
    explicit ``times`` the design chooses them: ``"identifiable"`` spans
    ``[0, 3 / r]`` (the signal decays to 5%) and ``"weak"`` uses only late
    times in ``[5 / r, 6 / r]``, where a change of ``a`` is almost
    compensated by a change of ``r`` and the Jacobian is nearly rank one.
    Explicit ``times`` give the ``"custom"`` design.

    Args:
        times: Optional non-negative observation times (``>= 3`` values with
            at least two distinct).
        amplitude: True nonzero amplitude ``a``.
        rate: True positive decay rate ``r``.
        count: Number of generated times (``>= 3``) when ``times`` is omitted.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed (only the noise is random).
        design: ``"identifiable"`` (default), ``"weak"`` or ``"custom"``.

    Returns:
        :class:`NonlinearBenchmark` with ``same_discretization=True``.

    Raises:
        ValueError: On inconsistent or out-of-range arguments.

    References:
        [EQ-INV-BENCH2-004], [REF-SEBER-WILD-2003],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    a = _finite_real(amplitude, "amplitude")
    if a == 0.0:
        raise ValueError("amplitude must be nonzero")
    r = _real_positive(rate, "rate")
    if design is None:
        design = "custom" if times is not None else "identifiable"
    if design not in ("identifiable", "weak", "custom"):
        raise ValueError("design must be 'identifiable', 'weak' or 'custom'")
    if design == "custom" and times is None:
        raise ValueError("design='custom' needs explicit times")
    if design != "custom" and times is not None:
        raise ValueError("explicit times require design='custom'")
    if times is None:
        k = _positive_int(count, "count", 3)
        lo, hi = (0.0, 3.0 / r) if design == "identifiable" else (5.0 / r, 6.0 / r)
        grid = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
    else:
        if isinstance(times, (str, bytes)) or not isinstance(times, Sequence):
            raise TypeError("times must be a sequence of reals")
        grid = [_finite_real(t, "time") for t in times]
        if len(grid) < 3 or len(set(grid)) < 2 or min(grid) < 0.0:
            raise ValueError("times need >= 3 non-negative values, two distinct")
    m = len(grid)

    def value(x: Vector) -> Vector:
        """Return ``a exp(-r t)``.

        References:
            [EQ-INV-BENCH2-004].

        """
        return tuple(x[0] * math.exp(-x[1] * t) for t in grid)

    def jvp(x: Vector, v: Vector) -> Vector:
        """Return ``J(x) v``.

        References:
            [EQ-INV-BENCH2-004].

        """
        out = []
        for t in grid:
            e = math.exp(-x[1] * t)
            out.append(e * v[0] - x[0] * t * e * v[1])
        return tuple(out)

    def vjp(x: Vector, w: Vector) -> Vector:
        """Return ``J(x)' w``.

        References:
            [EQ-INV-BENCH2-004].

        """
        es = [math.exp(-x[1] * t) for t in grid]
        return (
            math.fsum(wi * e for wi, e in zip(w, es, strict=True)),
            -x[0] * math.fsum(wi * t * e for wi, t, e in zip(w, grid, es, strict=True)),
        )

    model = CallbackForward(2, m, value, jvp, vjp)
    truth = (a, r)
    exact = value(truth)
    stream = resolve_stream(seed).child("N01")
    data, level, realized = _noise(exact, relative_noise, stream)
    pathologies = ("nonlinear",) + (
        ("ill_conditioned", "parameter_tradeoff") if design == "weak" else ()
    )
    return NonlinearBenchmark(
        identifier="N01",
        name=f"exponential decay ({design} design, {m} times)",
        model=model,
        data=data,
        noise=ScaledIdentityCovariance(m, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=True,
        prior_mean=None,
        prior_covariance=None,
        pathologies=pathologies,
        seeds=MappingProxyType({"noise": "N01/noise"}),
    )


# ---------------------------------------------------------------------------
# N02: squared and product maps
# ---------------------------------------------------------------------------


def squared_map_benchmark(
    dimension: int = 2,
    *,
    kind: str = "square",
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> NonlinearBenchmark:
    """Return N02: a structurally ambiguous nonlinear map.

    ``kind="square"`` is ``G(x) = x o x`` (elementwise): ``G(x) = G(-x)``, in
    fact ``G`` is invariant under every sign pattern, so the exact noise-free
    minimizers are the ``2^n`` points ``(+-x_1, ..., +-x_n)``
    [EQ-INV-BENCH2-005]. ``kind="product"`` is ``y_j = x_(2j) x_(2j+1)``
    (``dimension`` even): each observation fixes only a product, leaving a
    one-dimensional hyperbola per pair, and ``G(x) = G(-x)`` again. The truth
    entries are ``+-(0.5 + U)`` with random sign (``truth`` stream); a solver
    started near the origin sits at a saddle or at a point of vanishing
    Jacobian.

    Args:
        dimension: Number of parameters ``>= 1`` (even ``>= 2`` for
            ``"product"``).
        kind: ``"square"`` or ``"product"``.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed.

    Returns:
        :class:`NonlinearBenchmark` with ``same_discretization=True`` and the
        pathology ``"multimodal"``.

    Raises:
        ValueError: On inconsistent or out-of-range arguments.

    References:
        [EQ-INV-BENCH2-005], [REF-KAIPIO-SOMERSALO-2007].

    """
    n = _positive_int(dimension, "dimension", 1)
    if kind not in ("square", "product"):
        raise ValueError("kind must be 'square' or 'product'")
    if kind == "product" and (n < 2 or n % 2 != 0):
        raise ValueError("kind='product' needs an even dimension >= 2")
    m = n if kind == "square" else n // 2
    stream = resolve_stream(seed).child("N02")
    generator = stream.child("truth").generator()
    truth = tuple(
        (1.0 if generator.random() < 0.5 else -1.0) * (0.5 + generator.random())
        for _ in range(n)
    )

    def value(x: Vector) -> Vector:
        """Return ``x o x`` or the pairwise products.

        References:
            [EQ-INV-BENCH2-005].

        """
        if kind == "square":
            return tuple(v * v for v in x)
        return tuple(x[2 * j] * x[2 * j + 1] for j in range(m))

    def jvp(x: Vector, v: Vector) -> Vector:
        """Return ``J(x) v``.

        References:
            [EQ-INV-BENCH2-005].

        """
        if kind == "square":
            return tuple(2.0 * a * b for a, b in zip(x, v, strict=True))
        return tuple(
            x[2 * j + 1] * v[2 * j] + x[2 * j] * v[2 * j + 1] for j in range(m)
        )

    def vjp(x: Vector, w: Vector) -> Vector:
        """Return ``J(x)' w``.

        References:
            [EQ-INV-BENCH2-005].

        """
        if kind == "square":
            return tuple(2.0 * a * b for a, b in zip(x, w, strict=True))
        out = [0.0] * n
        for j in range(m):
            out[2 * j] = x[2 * j + 1] * w[j]
            out[2 * j + 1] = x[2 * j] * w[j]
        return tuple(out)

    model = CallbackForward(n, m, value, jvp, vjp)
    exact = value(truth)
    data, level, realized = _noise(exact, relative_noise, stream)
    pathologies = ("multimodal",) + (
        ("nonidentifiable", "scale_ambiguity") if kind == "product" else ()
    )
    return NonlinearBenchmark(
        identifier="N02",
        name=f"{kind} map (n={n})",
        model=model,
        data=data,
        noise=ScaledIdentityCovariance(m, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=True,
        prior_mean=None,
        prior_covariance=None,
        pathologies=pathologies,
        seeds=MappingProxyType({"truth": "N02/truth", "noise": "N02/noise"}),
    )


# ---------------------------------------------------------------------------
# P01: elliptic log-conductivity
# ---------------------------------------------------------------------------


def _log_conductivity(s: float) -> float:
    """Return the smooth true log-conductivity ``0.5 sin(2 pi s) + 0.3 s``.

    References:
        [EQ-INV-BENCH2-006].

    """
    return 0.5 * math.sin(2.0 * math.pi * s) + 0.3 * s


def _elliptic_source(s: float) -> float:
    """Return the smooth positive source ``1 + s``.

    References:
        [EQ-INV-BENCH2-006].

    """
    return 1.0 + s


def elliptic_benchmark(
    cells: int = 16,
    sensors: int | Sequence[float] = 8,
    *,
    refinement: int = 1,
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> NonlinearBenchmark:
    """Return P01: log-conductivity inversion for ``-(k u')' = 1 + s``.

    The solver's model is
    :class:`~probstat.inverse.pde_models.EllipticDiffusion1D` with ``cells``
    cells, unit length and homogeneous Dirichlet values; the parameter is
    ``x = log k`` per cell. The truth is the smooth function
    ``0.5 sin(2 pi s) + 0.3 s``. With ``refinement = r > 1`` the data are
    generated by the SAME physical problem on an ``r``-times finer mesh (the
    sensor interpolation of that mesh), and the solver's truth is the cell
    average of the fine truth, so a genuine discretization error remains in
    the data (``same_discretization=False``) [EQ-INV-BENCH2-006]. A smooth
    exponential-kernel field prior with zero mean is supplied for Bayesian
    use; it is not centred on the truth.

    Args:
        cells: Coarse number of cells ``>= 4``.
        sensors: Number of equispaced interior sensors ``>= 1`` or their
            positions in ``(0, 1)``.
        refinement: Integer factor ``r >= 1`` of the data mesh.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed (only the noise is random).

    Returns:
        :class:`NonlinearBenchmark`.

    Raises:
        ValueError: On out-of-range arguments.

    References:
        [EQ-INV-BENCH2-006], [REF-KAIPIO-SOMERSALO-2007],
        [REF-STUART-2010].

    """
    n = _positive_int(cells, "cells", 4)
    r = _positive_int(refinement, "refinement", 1)
    if isinstance(sensors, int) and not isinstance(sensors, bool):
        count = _positive_int(sensors, "sensors", 1)
        positions = [(j + 1) / (count + 1) for j in range(count)]
    else:
        if isinstance(sensors, (str, bytes)) or not isinstance(sensors, Sequence):
            raise TypeError("sensors must be an integer or a sequence of reals")
        positions = [_finite_real(s, "sensor") for s in sensors]
    model = EllipticDiffusion1D(n, sensors=positions, source=_elliptic_source)
    fine_n = r * n
    fine_truth = [_log_conductivity((j + 0.5) / fine_n) for j in range(fine_n)]
    truth = tuple(math.fsum(fine_truth[i * r : (i + 1) * r]) / r for i in range(n))
    if r == 1:
        exact = model.value(truth)
    else:
        fine_model = EllipticDiffusion1D(
            fine_n, sensors=positions, source=_elliptic_source
        )
        exact = fine_model.value(fine_truth)
    stream = resolve_stream(seed).child("P01")
    data, level, realized = _noise(exact, relative_noise, stream)
    centres = [(i + 0.5) / n for i in range(n)]
    return NonlinearBenchmark(
        identifier="P01",
        name=f"elliptic log-conductivity ({n} cells, refinement {r})",
        model=model,
        data=data,
        noise=ScaledIdentityCovariance(len(positions), level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=r == 1,
        prior_mean=(0.0,) * n,
        prior_covariance=_exponential_prior(centres, 1.0, 0.3),
        pathologies=("nonlinear", "ill_conditioned", "smoothing"),
        seeds=MappingProxyType({"noise": "P01/noise"}),
    )


# ---------------------------------------------------------------------------
# P02: heat initial condition
# ---------------------------------------------------------------------------


def _initial_profile(s: float) -> float:
    """Return the smooth initial temperature ``exp(-60 (s-0.4)^2) + 0.5 s (1-s)``.

    References:
        [EQ-INV-BENCH2-007].

    """
    return math.exp(-60.0 * (s - 0.4) ** 2) + 0.5 * s * (1.0 - s)


def heat_initial_condition_benchmark(
    cells: int = 24,
    *,
    final_time: float = 0.05,
    steps: int = 10,
    step_refinement: int = 1,
    scheme: str = "implicit_euler",
    relative_noise: float = 1e-2,
    seed: int | RandomStream | None = 11,
) -> InverseBenchmark:
    """Return P02: recover the initial temperature from the final profile.

    The solver's operator is
    :class:`~probstat.inverse.heat_models.HeatEquation1D` with ``steps`` time
    steps, observing the state at ``final_time`` at all ``cells`` interior
    nodes. With ``step_refinement = q > 1`` the data are generated by the
    same operator with ``q * steps`` steps (same spatial grid, same scheme),
    so the time-discretization error of the coarse model is present in the
    data (``same_discretization=False``) [EQ-INV-BENCH2-007]. The returned
    ``singular_values`` are the closed-form values of the coarse operator,
    which decay like ``exp(-k^2 pi^2 T)``: severely ill-posed.

    Args:
        cells: Number of interior nodes ``>= 4``.
        final_time: Horizon ``T > 0``.
        steps: Coarse number of time steps ``>= 1``.
        step_refinement: Integer factor ``q >= 1`` of the data time steps.
        scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        seed: Root seed (only the noise is random).

    Returns:
        :class:`~probstat.inverse.benchmark_problems.InverseBenchmark`.

    Raises:
        ValueError: On out-of-range arguments.

    References:
        [EQ-INV-BENCH2-007], [REF-HANSEN-2008], [REF-KAIPIO-SOMERSALO-2007].

    """
    n = _positive_int(cells, "cells", 4)
    horizon = _real_positive(final_time, "final_time")
    count = _positive_int(steps, "steps", 1)
    q = _positive_int(step_refinement, "step_refinement", 1)
    operator = HeatEquation1D(n, final_time=horizon, steps=count, scheme=scheme)
    h = 1.0 / (n + 1)
    truth = tuple(_initial_profile((j + 1) * h) for j in range(n))
    if q == 1:
        exact = operator.matvec(truth)
    else:
        fine = HeatEquation1D(n, final_time=horizon, steps=q * count, scheme=scheme)
        exact = fine.matvec(truth)
    stream = resolve_stream(seed).child("P02")
    data, level, realized = _noise(exact, relative_noise, stream)
    return InverseBenchmark(
        identifier="P02",
        name=f"heat initial condition ({n} nodes, step refinement {q})",
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(n, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=truth,
        exact_data=exact,
        same_discretization=q == 1,
        prior_covariance=None,
        singular_values=heat_semigroup_singular_values(
            n, final_time=horizon, steps=count, scheme=scheme
        ),
        pathologies=("severely_ill_posed", "smoothing"),
        seeds=MappingProxyType({"noise": "P02/noise"}),
    )


EXTENDED_BENCHMARK_CATALOG: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "L07a",
        "2D deblurring",
        "inverse.image_deblur_benchmark",
        "fine-grid independent data (refinement > 1)",
        ("ill_conditioned", "edges"),
        ("krylov_regularization", "total_variation_regularized_least_squares"),
    ),
    BenchmarkCase(
        "L07b",
        "parallel-beam tomography prototype",
        "inverse.masked_tomography_benchmark",
        "analytic disc line integrals (analytic=True)",
        ("ill_conditioned", "nullspace", "limited_angle"),
        ("krylov_regularization", "hybrid_lsqr"),
    ),
    BenchmarkCase(
        "N01",
        "exponential decay",
        "inverse.exponential_decay_benchmark",
        "analytic Jacobian; identifiable versus weak design",
        ("nonlinear", "ill_conditioned", "parameter_tradeoff"),
        ("gauss_newton_krylov", "iteratively_regularized_gauss_newton"),
    ),
    BenchmarkCase(
        "N02",
        "squared and product maps",
        "inverse.squared_map_benchmark",
        "exact sign-symmetric optima G(x) = G(-x)",
        ("multimodal", "nonidentifiable"),
        ("gauss_newton_krylov", "multistart"),
    ),
    BenchmarkCase(
        "P01",
        "elliptic log-conductivity",
        "inverse.elliptic_benchmark",
        "fine-mesh independent data (refinement > 1)",
        ("nonlinear", "ill_conditioned", "smoothing"),
        ("gauss_newton_krylov", "iteratively_regularized_gauss_newton"),
    ),
    BenchmarkCase(
        "P02",
        "heat initial condition",
        "inverse.heat_initial_condition_benchmark",
        "closed-form sine singular values; finer time steps",
        ("severely_ill_posed", "smoothing"),
        ("spectral_diagnostics", "krylov_regularization", "hybrid_lsqr"),
    ),
)
"""The extended benchmark families (roadmap 19: L07a, L07b, N01, N02, P01, P02)."""
