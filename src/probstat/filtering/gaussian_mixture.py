"""Gaussian splitting and mixture propagation through nonlinear maps.

A Gaussian propagated through a nonlinear map is no longer Gaussian, and a
single sigma-point transform summarises the image by its first two moments
only. Splitting repairs this at the input: replace ``N(m, P)`` by a mixture
of narrower Gaussians that has the same mean and covariance, push each narrow
component through the map (over a small region the map is closer to linear
and a sigma-point transform is more accurate), and recombine. The result is a
mixture whose moment-matched mean and covariance are closer to the exact
moments and whose density shows the non-Gaussian shape.

The module has three parts:

- :func:`univariate_splitting`: an approximation of the standard normal
  density by an odd number ``N`` of equally spaced components with equal
  variance ``sigma^2``, weights ``w_i`` symmetric about zero. For a chosen
  component standard deviation ``sigma`` the weights are the minimiser of the
  integrated squared density difference ``int (f_mix - phi)^2`` (closed-form
  Gaussian product integrals), the spacing ``h`` is fixed by the variance
  condition ``h^2 sum_i w_i i^2 = 1 - sigma^2``, and the mean is zero by
  symmetry; so the mixture has the exact mean and variance of the standard
  normal for every ``sigma`` in ``(0, 1)``. Nothing is read from a published
  table; the solutions for 3, 5, and 7 components are cached as constants that
  the tests re-derive.
- :func:`split_gaussian`: the multivariate split along one direction. In
  coordinates whitened by ``P`` the split acts along the unit vector ``e`` and
  leaves the orthogonal directions untouched, so the mean and covariance of
  the mixture equal ``m`` and ``P`` exactly (to rounding).
- :func:`propagate_gaussian_mixture` and its generator form
  :func:`propagate_gaussian_mixture_procedure`: split, push every component
  through ``g`` with the unscented transform of
  :mod:`probstat.filtering.extended`, and return the output mixture. Optional
  recursive splitting refines the components whose
  :func:`nonlinearity_index` exceeds a threshold.

The generator form follows the batch evaluation contract of
:mod:`probstat.foundations.batch`: every refinement level issues one request
holding the ``(2n + 1)`` sigma points of every component, so a vectorised or
queued simulator evaluates a whole level at once. The pointwise wrapper
drives the same generator, so the two give identical results.

Assumptions: the splitting has the exact mean and variance of the normal but
not its fourth moment or density; its density error is the stated squared L2
distance. The unscented transform inside each component is exact for the mean
of a quadratic map and for the covariance of a linear one, nothing more; the
output mixture is a better, not an exact, summary. Sigma-point evaluation
needs a deterministic map: two calls with equal inputs must give equal
outputs. Process noise is not added; add it to the covariances of the output
mixture if the map is a prediction step.

References:
    [REF-DEMARS-BISHOP-JAH-2013], [REF-HUBER-ETAL-2008],
    [REF-VITTALDEV-RUSSELL-2016], [REF-JULIER-UHLMANN-2004],
    [REF-SORENSON-ALSPACH-1971], [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from probstat.foundations.batch import (
    BatchModel,
    Point,
    Procedure,
    batch_from_pointwise,
    request,
    run_procedure,
    vector_outputs,
)
from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import require_finite_real
from probstat.inference import GaussianMixture
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    outer,
    solve_spd,
    symmetric_eigen,
    symmetrize,
)
from probstat.optimization import minimize_bfgs

from .extended import UnscentedWeights, unscented_transform

DEFAULT_COMPONENT_SIGMA = {3: 0.7, 5: 0.5, 7: 0.4}
"""Component standard deviation used by default for 3, 5, and 7 components."""

# Weights solved by ``_solve_splitting`` for the default component standard
# deviations (centre first, then outward); the tests re-derive them.
_CACHED_WEIGHTS: dict[int, tuple[float, ...]] = {
    3: (0.5847645246393008, 0.2076177376803496),
    5: (0.4114726887704962, 0.23750305764905616, 0.05676059796569572),
    7: (
        0.3127897538182607,
        0.23069124327578794,
        0.08890667573803429,
        0.024007204077047418,
    ),
}


@dataclass(frozen=True, slots=True)
class UnivariateSplitting:
    """Equal-variance, equally spaced mixture approximating ``N(0, 1)``.

    The component ``i`` (``i = -K..K``, ``N = 2K + 1``) has weight
    ``weights[i + K]``, mean ``offsets[i + K] = i h`` and standard deviation
    ``sigma``. The weights are symmetric, so the mean is zero, and
    ``h`` satisfies ``sum_i w_i (i h)^2 + sigma^2 = 1``.

    Args:
        components: Odd number of components ``N``.
        sigma: Common component standard deviation in ``(0, 1]``.
        weights: Component weights (positive, sum one).
        offsets: Component means ``i h``.
        squared_l2_error: ``int (f_mix - phi)^2 dx``.

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-HUBER-ETAL-2008].

    """

    components: int
    sigma: float
    weights: tuple[float, ...]
    offsets: tuple[float, ...]
    squared_l2_error: float

    @property
    def spacing(self) -> float:
        """Return the distance ``h`` between neighbouring component means.

        Equation:
            [EQ-FILT-GMSPLIT-001].

        References:
            [REF-DEMARS-BISHOP-JAH-2013].

        """
        if self.components == 1:
            return 0.0
        return self.offsets[self.components // 2 + 1]

    @property
    def relative_l2_error(self) -> float:
        """Return the L2 error relative to the norm of the normal density.

        ``sqrt(int (f_mix - phi)^2 / int phi^2)`` with
        ``int phi^2 = 1 / (2 sqrt(pi))``.

        Equation:
            [EQ-FILT-GMSPLIT-001].

        References:
            [REF-DEMARS-BISHOP-JAH-2013].

        """
        return math.sqrt(max(self.squared_l2_error, 0.0) * 2.0 * math.sqrt(math.pi))


def _gaussian_value(x: float, variance: float) -> float:
    """Return the ``N(0, variance)`` density at ``x``.

    References:
        [REF-HUBER-ETAL-2008]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    return math.exp(-0.5 * x * x / variance) / math.sqrt(2.0 * math.pi * variance)


def _squared_l2(weights: Sequence[float], spacing: float, sigma: float) -> float:
    """Return ``int (sum_i w_i N(i h, sigma^2) - phi)^2`` by product integrals.

    ``int N(x; a, s^2) N(x; b, t^2) dx = N(a - b; 0, s^2 + t^2)``, so the
    integral is a finite double sum.

    References:
        [REF-HUBER-ETAL-2008]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    count = len(weights)
    half = count // 2
    total = 0.0
    cross = 0.0
    for a in range(count):
        cross += weights[a] * _gaussian_value((a - half) * spacing, 1.0 + sigma**2)
        for b in range(count):
            total += (
                weights[a]
                * weights[b]
                * _gaussian_value((a - b) * spacing, 2.0 * sigma**2)
            )
    return total - 2.0 * cross + _gaussian_value(0.0, 2.0)


def _expand(logits: Sequence[float], count: int) -> tuple[float, ...]:
    """Return symmetric positive weights from ``K`` free logits.

    The centre logit is fixed at zero and the weights are a softmax over the
    ``2K + 1`` components, which keeps them positive and summing to one.

    References:
        [REF-HUBER-ETAL-2008]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    half = count // 2
    full = [0.0, *logits]
    peak = max(full)
    raw = [math.exp(v - peak) for v in full]
    unique = [raw[abs(i)] for i in range(-half, half + 1)]
    scale = math.fsum(unique)
    return tuple(v / scale for v in unique)


def _spacing(weights: Sequence[float], sigma: float) -> float:
    """Return ``h`` from the variance condition ``h^2 sum w_i i^2 = 1 - sigma^2``.

    References:
        [REF-DEMARS-BISHOP-JAH-2013]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    half = len(weights) // 2
    second = math.fsum(w * (i - half) ** 2 for i, w in enumerate(weights))
    return math.sqrt((1.0 - sigma**2) / second)


def _solve_splitting(count: int, sigma: float) -> UnivariateSplitting:
    """Minimise the squared L2 distance over the symmetric weights.

    The unknowns are ``K`` softmax logits; the spacing follows from the
    variance condition, so every iterate satisfies the moment conditions.
    The squared distance is minimised by BFGS from a Gaussian-shaped start.

    References:
        [REF-NOCEDAL-WRIGHT-2006], [REF-HUBER-ETAL-2008]. Evaluates
        [EQ-FILT-GMSPLIT-001].

    """
    half = count // 2

    def objective(logits: Vector) -> float:
        """Return the squared L2 distance in units of ``int phi^2``.

        References:
            [REF-HUBER-ETAL-2008]. Evaluates [EQ-FILT-GMSPLIT-001].

        """
        weights = _expand(logits, count)
        return _squared_l2(weights, _spacing(weights, sigma), sigma) * (
            2.0 * math.sqrt(math.pi)
        )

    start = tuple(-0.25 * k * k for k in range(1, half + 1))
    fit = minimize_bfgs(objective, start, gradient_tolerance=1.0e-13)
    weights = _expand(fit.x, count)
    return _assemble(weights, sigma)


def _assemble(weights: Sequence[float], sigma: float) -> UnivariateSplitting:
    """Build the record from weights, deriving spacing and the L2 error.

    References:
        [REF-DEMARS-BISHOP-JAH-2013]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    count = len(weights)
    half = count // 2
    total = math.fsum(weights)
    normalised = tuple(w / total for w in weights)
    spacing = _spacing(normalised, sigma)
    return UnivariateSplitting(
        count,
        sigma,
        normalised,
        tuple((i - half) * spacing for i in range(count)),
        _squared_l2(normalised, spacing, sigma),
    )


@lru_cache(maxsize=64)
def _cached_splitting(count: int, sigma: float) -> UnivariateSplitting:
    """Return the solved splitting, using the stored weights when they apply.

    References:
        [REF-HUBER-ETAL-2008]. Evaluates [EQ-FILT-GMSPLIT-001].

    """
    stored = _CACHED_WEIGHTS.get(count)
    if stored is not None and DEFAULT_COMPONENT_SIGMA.get(count) == sigma:
        return _assemble(tuple(reversed(stored[1:])) + stored, sigma)
    return _solve_splitting(count, sigma)


def univariate_splitting(
    components: int = 3, *, sigma: float | None = None
) -> UnivariateSplitting:
    """Return the equal-variance mixture approximation of ``N(0, 1)``.

    For ``components = 1`` the answer is the normal itself. Otherwise the
    weights minimise ``int (sum_i w_i N(i h, sigma^2) - phi)^2`` over
    symmetric positive weights with ``h`` set by ``h^2 sum_i w_i i^2 =
    1 - sigma^2``. A smaller ``sigma`` gives narrower components (a better
    linearisation inside each) and a larger density error; the sizes 3, 5,
    and 7 default to ``sigma`` = 0.7, 0.5, 0.4.

    Args:
        components: Odd positive number of components.
        sigma: Component standard deviation in ``(0, 1)``; required for sizes
            other than 1, 3, 5, and 7.

    Returns:
        Weights, offsets, and the squared L2 error.

    Raises:
        TypeError: If ``components`` is not an integer or ``sigma`` is not
            real.
        ValueError: If ``components`` is even or non-positive, ``sigma`` is
            outside ``(0, 1)``, or no default exists and ``sigma`` is
            missing.

    Equation:
        [EQ-FILT-GMSPLIT-001].

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-HUBER-ETAL-2008],
        [REF-VITTALDEV-RUSSELL-2016].

    """
    if isinstance(components, bool) or not isinstance(components, int):
        raise TypeError("components must be an integer")
    if components < 1 or components % 2 == 0:
        raise ValueError("components must be a positive odd integer")
    if components == 1:
        return UnivariateSplitting(1, 1.0, (1.0,), (0.0,), 0.0)
    if sigma is None:
        if components not in DEFAULT_COMPONENT_SIGMA:
            raise ValueError("sigma is required for this number of components")
        width = DEFAULT_COMPONENT_SIGMA[components]
    else:
        width = require_finite_real(sigma, name="sigma")
        if not 0.0 < width < 1.0:
            raise ValueError("sigma must lie in (0, 1)")
    return _cached_splitting(components, width)


@dataclass(frozen=True, slots=True)
class GaussianMixtureState:
    """Finite Gaussian mixture ``sum_i w_i N(m_i, C_i)`` with moment matching.

    Args:
        weights: Non-negative weights summing to one.
        means: One mean vector per component.
        covariances: One symmetric positive-semidefinite covariance per
            component. Density evaluation and sampling need positive
            definite components.

    References:
        [REF-SORENSON-ALSPACH-1971], [REF-BISHOP-2006, Section 9.2].

    """

    weights: Vector
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]

    def __post_init__(self) -> None:
        """Validate and normalise the containers.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If shapes disagree, a weight is negative, or the
                weights do not sum to one.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No
            mathematical equation is evaluated.

        """
        weights = as_vector(self.weights, name="weights")
        means = tuple(as_vector(m, name="means") for m in self.means)
        covariances = tuple(
            tuple(as_vector(row, name="covariances") for row in matrix)
            for matrix in self.covariances
        )
        if (
            not weights
            or len(means) != len(weights)
            or len(covariances) != len(weights)
        ):
            raise ValueError("weights, means and covariances must align")
        if any(w < 0.0 for w in weights):
            raise ValueError("weights must be non-negative")
        if abs(math.fsum(weights) - 1.0) > 1.0e-8:
            raise ValueError("weights must sum to one")
        size = len(means[0])
        if size == 0 or any(len(m) != size for m in means):
            raise ValueError("means must share a positive dimension")
        for matrix in covariances:
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError("covariances must be square with the mean dimension")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)

    @property
    def components(self) -> int:
        """Return the number of components.

        References:
            [REF-BISHOP-2006, Section 9.2]. No equation is evaluated.

        """
        return len(self.weights)

    @property
    def dimension(self) -> int:
        """Return the dimension of the random vector.

        References:
            [REF-BISHOP-2006, Section 9.2]. No equation is evaluated.

        """
        return len(self.means[0])

    @property
    def mean(self) -> Vector:
        """Return the mixture mean ``sum_i w_i m_i``.

        Equation:
            [EQ-FILT-GMPROP-001].

        References:
            [REF-SORENSON-ALSPACH-1971].

        """
        return tuple(
            math.fsum(w * m[j] for w, m in zip(self.weights, self.means, strict=True))
            for j in range(self.dimension)
        )

    @property
    def covariance(self) -> Matrix:
        """Return ``sum_i w_i [C_i + (m_i - m)(m_i - m)']``.

        Equation:
            [EQ-FILT-GMPROP-001].

        References:
            [REF-SORENSON-ALSPACH-1971].

        """
        centre = self.mean
        size = self.dimension
        rows = []
        for r in range(size):
            row = []
            for c in range(size):
                row.append(
                    math.fsum(
                        w * (cov[r][c] + (m[r] - centre[r]) * (m[c] - centre[c]))
                        for w, m, cov in zip(
                            self.weights, self.means, self.covariances, strict=True
                        )
                    )
                )
            rows.append(tuple(row))
        return tuple(rows)

    def as_mixture(self) -> GaussianMixture:
        """Return the same mixture as an inference-layer ``GaussianMixture``.

        Returns:
            A validated mixture with density, log density and sampling.

        Raises:
            ValueError: If a component covariance is not positive definite.

        References:
            [REF-BISHOP-2006, Section 9.2]. No equation is evaluated.

        """
        keep = [i for i, w in enumerate(self.weights) if w > 0.0]
        total = math.fsum(self.weights[i] for i in keep)
        return GaussianMixture(
            tuple(self.weights[i] / total for i in keep),
            tuple(self.means[i] for i in keep),
            tuple(self.covariances[i] for i in keep),
        )

    def log_pdf(self, x: Iterable[float]) -> float:
        """Return the log mixture density at ``x``.

        Args:
            x: Point of the state dimension.

        Returns:
            ``log sum_i w_i N(x; m_i, C_i)``.

        Raises:
            ValueError: If a component covariance is not positive definite or
                the point has the wrong dimension.

        References:
            [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

        """
        return self.as_mixture().log_pdf(tuple(x))

    def pdf(self, x: Iterable[float]) -> float:
        """Return the mixture density at ``x``.

        Args:
            x: Point of the state dimension.

        Returns:
            ``sum_i w_i N(x; m_i, C_i)``.

        Raises:
            ValueError: If a component covariance is not positive definite or
                the point has the wrong dimension.

        References:
            [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

        """
        return math.exp(self.log_pdf(x))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw ``size`` points by ancestral sampling.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream; ``None`` draws fresh entropy.

        Returns:
            The draws as vectors.

        Raises:
            ValueError: If a component covariance is not positive definite.

        References:
            [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

        """
        return self.as_mixture().sample(size, seed=seed)


def _covariance_matrix(covariance: Iterable[Iterable[float]], size: int) -> Matrix:
    """Return a validated symmetric positive-definite covariance.

    Raises:
        ValueError: If the shape is wrong or the matrix is not symmetric
            positive definite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    matrix = symmetrize(covariance, name="covariance")
    if len(matrix) != size:
        raise ValueError("covariance must be square with the mean dimension")
    cholesky(matrix)
    return matrix


def split_gaussian(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    *,
    direction: Iterable[float] | None = None,
    components: int = 3,
    sigma: float | None = None,
) -> GaussianMixtureState:
    """Split ``N(m, P)`` into a mixture with the same mean and covariance.

    Let ``d`` be the split direction and ``q = d'P^-1 d``. With the splitting
    ``(w_i, h_i)`` of :func:`univariate_splitting` the components are

    ``m_i = m + h_i d / sqrt(q)`` and
    ``C = P - (1 - sigma^2) d d' / q`` (the same for every component).

    The mixture mean is ``m`` because the weights are symmetric, and the
    covariance is ``C + (sum_i w_i h_i^2) d d' / q = P`` because
    ``sum_i w_i h_i^2 = 1 - sigma^2``. Only the variance of ``P`` along the
    whitened direction ``L'^-1 ... d`` shrinks to ``sigma^2`` of its value;
    orthogonal directions are untouched. Splitting along the principal
    eigenvector (the default) shrinks the direction of largest uncertainty.

    Args:
        mean: State mean ``m``.
        covariance: Symmetric positive-definite covariance ``P``.
        direction: Non-zero direction ``d`` in state space (any scale and
            sign); defaults to the eigenvector of the largest eigenvalue.
        components: Odd number of components (1 returns ``N(m, P)``).
        sigma: Component standard deviation in whitened units; see
            :func:`univariate_splitting`.

    Returns:
        The mixture; its :attr:`~GaussianMixtureState.mean` and
        :attr:`~GaussianMixtureState.covariance` equal ``m`` and ``P`` to
        rounding.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``P`` is not symmetric positive
            definite, or ``direction`` is zero.

    Equation:
        [EQ-FILT-GMSPLIT-002].

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-VITTALDEV-RUSSELL-2016],
        [REF-HUBER-ETAL-2008].

    """
    centre = as_vector(mean, name="mean")
    if not centre:
        raise ValueError("mean must contain at least one component")
    matrix = _covariance_matrix(covariance, len(centre))
    splitting = univariate_splitting(components, sigma=sigma)
    if direction is None:
        _, vectors = symmetric_eigen(matrix)
        axis = tuple(row[-1] for row in vectors)
    else:
        axis = as_vector(direction, name="direction")
        if len(axis) != len(centre):
            raise ValueError("direction must have the mean dimension")
        if all(v == 0.0 for v in axis):
            raise ValueError("direction must be non-zero")
    solved = solve_spd(matrix, axis)
    quadratic = math.fsum(a * s for a, s in zip(axis, solved, strict=True))
    unit = tuple(a / math.sqrt(quadratic) for a in axis)
    shrink = (1.0 - splitting.sigma**2) if components > 1 else 0.0
    spread = outer(unit, unit)
    component_covariance = tuple(
        tuple(
            0.5 * (matrix[i][j] + matrix[j][i]) - shrink * spread[i][j]
            for j in range(len(centre))
        )
        for i in range(len(centre))
    )
    means = tuple(
        tuple(c + offset * u for c, u in zip(centre, unit, strict=True))
        for offset in splitting.offsets
    )
    return GaussianMixtureState(
        splitting.weights,
        means,
        tuple(component_covariance for _ in splitting.weights),
    )


def _sigma_points(
    mean: Vector, covariance: Matrix, weights: UnscentedWeights
) -> tuple[Vector, ...]:
    """Return the sigma points the unscented transform would evaluate.

    References:
        [REF-JULIER-UHLMANN-2004]. Evaluates [EQ-FILT-UKF-001].

    """
    return unscented_transform(
        lambda point: point, mean, covariance, weights=weights
    ).sigma_points


def _replay(images: Sequence[Vector]) -> Callable[[Vector], Vector]:
    """Return a function that hands back the stored images in call order.

    The unscented transform evaluates its map once per sigma point, in
    order; replaying the stored outputs lets it compute its moments without
    a second evaluation of the (possibly remote) model.

    References:
        [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

    """
    table = iter(images)

    def next_image(_point: Vector) -> Vector:
        """Return the next stored image.

        References:
            [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

        """
        return next(table)

    return next_image


@dataclass(frozen=True, slots=True)
class _Component:
    """One input component awaiting evaluation.

    References:
        [REF-SORENSON-ALSPACH-1971]. No equation is evaluated.

    """

    weight: float
    mean: Vector
    covariance: Matrix
    depth: int


def _curvature_index(
    points: Sequence[Vector], images: Sequence[Vector], output_covariance: Matrix
) -> tuple[float, int]:
    """Return the relative curvature index and the axis that attains it.

    ``max_i || (g(m + s_i) + g(m - s_i))/2 - g(m) || / sqrt(tr C_y)`` over the
    sigma-point axes ``s_i``. It is zero for an affine map and measures the
    part of the image that a linearisation misses relative to its spread.

    References:
        [REF-JULIER-UHLMANN-2004]. Evaluates [EQ-FILT-GMPROP-002].

    """
    size = (len(points) - 1) // 2
    scale = math.sqrt(
        max(0.0, sum(output_covariance[j][j] for j in range(len(images[0]))))
    )
    if scale == 0.0:
        return 0.0, 0
    best = 0.0
    axis = 0
    for i in range(size):
        plus, minus, centre = images[1 + i], images[1 + size + i], images[0]
        gap = math.sqrt(
            math.fsum(
                (0.5 * (a + b) - c) ** 2
                for a, b, c in zip(plus, minus, centre, strict=True)
            )
        )
        if gap > best:
            best, axis = gap, i
    return best / scale, axis


def nonlinearity_index(
    function: Callable[[Vector], Sequence[float]],
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    *,
    unscented: UnscentedWeights | None = None,
) -> float:
    """Return the relative curvature index of ``g`` over ``N(m, P)``.

    With sigma-point offsets ``s_i`` (the unscented transform of
    :func:`probstat.filtering.unscented_transform`), the index is
    ``max_i || (g(m + s_i) + g(m - s_i))/2 - g(m) || / sqrt(tr C_y)`` where
    ``C_y`` is the unscented output covariance. It vanishes exactly for an
    affine ``g`` and grows with the second-order term that a linear
    propagation ignores, in units of the output spread. It reuses the sigma
    points of the transform, so it costs no extra evaluations inside
    :func:`propagate_gaussian_mixture`.

    Args:
        function: Map ``g`` (deterministic).
        mean: Input mean.
        covariance: Positive-definite input covariance.
        unscented: Sigma-point parameters; default ``alpha=1, beta=2,
            kappa=0``.

    Returns:
        A non-negative number; ``0`` for a constant map.

    Raises:
        ValueError: If shapes disagree or the covariance is not positive
            definite.

    Equation:
        [EQ-FILT-GMPROP-002].

    References:
        [REF-JULIER-UHLMANN-2004], [REF-DEMARS-BISHOP-JAH-2013].

    """
    centre = as_vector(mean, name="mean")
    matrix = _covariance_matrix(covariance, len(centre))
    weights = unscented if unscented is not None else UnscentedWeights()
    points = _sigma_points(centre, matrix, weights)
    images = tuple(as_vector(function(p), name="g(sigma point)") for p in points)
    moments = unscented_transform(_replay(images), centre, matrix, weights=weights)
    return _curvature_index(points, images, moments.covariance)[0]


@dataclass(frozen=True, slots=True)
class GaussianMixturePropagation:
    """Output of :func:`propagate_gaussian_mixture`.

    Args:
        mixture: The output mixture (weights sum to one).
        nonlinearity: Curvature index of each output component at the level
            where it was accepted.
        depths: Splitting depth of each output component (0 for the initial
            components).
        levels: Evaluation requests issued (one per refinement level).
        evaluations: Total number of ``g`` evaluations.

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-JULIER-UHLMANN-2004].

    """

    mixture: GaussianMixtureState
    nonlinearity: tuple[float, ...]
    depths: tuple[int, ...]
    levels: int
    evaluations: int

    @property
    def mean(self) -> Vector:
        """Return the moment-matched output mean.

        Equation:
            [EQ-FILT-GMPROP-001].

        References:
            [REF-SORENSON-ALSPACH-1971].

        """
        return self.mixture.mean

    @property
    def covariance(self) -> Matrix:
        """Return the moment-matched output covariance.

        Equation:
            [EQ-FILT-GMPROP-001].

        References:
            [REF-SORENSON-ALSPACH-1971].

        """
        return self.mixture.covariance


def propagate_gaussian_mixture_procedure(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    *,
    components: int = 3,
    direction: Iterable[float] | None = None,
    sigma: float | None = None,
    unscented: UnscentedWeights | None = None,
    max_depth: int = 0,
    nonlinearity_threshold: float = 0.1,
    split_components: int = 3,
    max_components: int = 81,
    label: str = "map",
) -> Procedure[GaussianMixturePropagation]:
    """Return the generator form of :func:`propagate_gaussian_mixture`.

    Yields one :class:`~probstat.foundations.batch.EvaluationRequest` per
    refinement level, holding the ``2n + 1`` sigma points of every pending
    component, and returns the :class:`GaussianMixturePropagation`. Drive it
    with :func:`~probstat.foundations.batch.run_procedure` and a batch model
    or with an :class:`~probstat.foundations.batch.AskTellSession` when the
    evaluation happens elsewhere. Arguments as in
    :func:`propagate_gaussian_mixture`.

    Raises:
        TypeError: If an entry is not real or an integer argument is not an
            integer.
        ValueError: If shapes disagree, the covariance is not positive
            definite, ``max_depth < 0``, the threshold is negative, or
            ``max_components`` is below ``components``.

    Equation:
        [EQ-FILT-GMPROP-001], [EQ-FILT-GMPROP-002].

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-JULIER-UHLMANN-2004].

    """
    centre = as_vector(mean, name="mean")
    if not centre:
        raise ValueError("mean must contain at least one component")
    matrix = _covariance_matrix(covariance, len(centre))
    for value, name in ((max_depth, "max_depth"), (max_components, "max_components")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    threshold = require_finite_real(
        nonlinearity_threshold, name="nonlinearity_threshold"
    )
    if threshold < 0.0:
        raise ValueError("nonlinearity_threshold must be non-negative")
    if max_components < components:
        raise ValueError("max_components must be at least components")
    parameters = unscented if unscented is not None else UnscentedWeights()
    start = split_gaussian(
        centre, matrix, direction=direction, components=components, sigma=sigma
    )
    pending = [
        _Component(w, m, c, 0)
        for w, m, c in zip(start.weights, start.means, start.covariances, strict=True)
    ]
    accepted: list[tuple[_Component, Vector, Matrix, float]] = []
    levels = 0
    evaluations = 0
    while pending:
        plans = [_sigma_points(c.mean, c.covariance, parameters) for c in pending]
        flat: tuple[Point, ...] = tuple(p for plan in plans for p in plan)
        outputs = yield request(flat, label=label, stage=f"level {levels}")
        rows = vector_outputs(outputs, len(flat))
        levels += 1
        evaluations += len(flat)
        cursor = 0
        refine: list[_Component] = []
        budget = max_components - len(accepted) - len(pending)
        for component, plan in zip(pending, plans, strict=True):
            images = rows[cursor : cursor + len(plan)]
            cursor += len(plan)
            moments = unscented_transform(
                _replay(images),
                component.mean,
                component.covariance,
                weights=parameters,
            )
            index, axis = _curvature_index(plan, images, moments.covariance)
            extra = split_components - 1
            if component.depth < max_depth and index > threshold and budget >= extra:
                budget -= extra
                offset = tuple(
                    a - b for a, b in zip(plan[1 + axis], component.mean, strict=True)
                )
                piece = split_gaussian(
                    component.mean,
                    component.covariance,
                    direction=offset,
                    components=split_components,
                    sigma=sigma
                    if sigma is not None and split_components == components
                    else None,
                )
                refine.extend(
                    _Component(component.weight * w, m, c, component.depth + 1)
                    for w, m, c in zip(
                        piece.weights, piece.means, piece.covariances, strict=True
                    )
                )
            else:
                accepted.append((component, moments.mean, moments.covariance, index))
        pending = refine
    mixture = GaussianMixtureState(
        tuple(item[0].weight for item in accepted),
        tuple(item[1] for item in accepted),
        tuple(item[2] for item in accepted),
    )
    return GaussianMixturePropagation(
        mixture,
        tuple(item[3] for item in accepted),
        tuple(item[0].depth for item in accepted),
        levels,
        evaluations,
    )


def propagate_gaussian_mixture(
    function: Callable[[Vector], Sequence[float]],
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    *,
    components: int = 3,
    direction: Iterable[float] | None = None,
    sigma: float | None = None,
    unscented: UnscentedWeights | None = None,
    max_depth: int = 0,
    nonlinearity_threshold: float = 0.1,
    split_components: int = 3,
    max_components: int = 81,
    batch_model: BatchModel | None = None,
    max_batch_size: int | None = None,
) -> GaussianMixturePropagation:
    """Propagate ``N(m, P)`` through ``g`` as a Gaussian mixture.

    The input is split along ``direction`` (default: principal eigenvector)
    into ``components`` Gaussians with the exact mean and covariance of
    ``N(m, P)`` (:func:`split_gaussian`); each component goes through the
    unscented transform of ``g``; the output mixture has the components'
    unscented means and covariances and the same weights. Its moment-matched
    mean and covariance (:attr:`GaussianMixturePropagation.mean`,
    ``.covariance``) are the quantities to compare with a single unscented
    transform. For an affine ``g`` they equal ``A m + b`` and ``A P A'``
    exactly. For a quadratic ``g`` the mean is exact for any split because
    the unscented mean of a quadratic is exact per component.

    With ``max_depth > 0`` every component whose
    :func:`nonlinearity_index` exceeds ``nonlinearity_threshold`` is replaced
    by ``split_components`` children split along its sigma axis of largest
    curvature, up to ``max_depth`` times and ``max_components`` output
    components. Each level is one batch of sigma points.

    Args:
        function: Deterministic map ``g`` from a state vector to a vector.
        mean: Input mean ``m``.
        covariance: Positive-definite input covariance ``P``.
        components: Odd number of initial components.
        direction: Split direction; default principal eigenvector.
        sigma: Component standard deviation in whitened units.
        unscented: Sigma-point parameters.
        max_depth: Maximum number of refinement levels (0: no refinement).
        nonlinearity_threshold: Refine a component when its index exceeds
            this value.
        split_components: Odd number of children per refinement.
        max_components: Upper limit on the number of output components.
        batch_model: Optional model that evaluates a sequence of points at
            once; it replaces ``function`` when given.
        max_batch_size: Largest number of points per ``batch_model`` call.

    Returns:
        The output mixture with its nonlinearity indices, depths, levels, and
        evaluation count.

    Raises:
        TypeError: If an entry is not real or an integer argument is not an
            integer.
        ValueError: If shapes disagree, ``P`` is not positive definite, or an
            option is outside its range.

    Equation:
        [EQ-FILT-GMPROP-001], [EQ-FILT-GMPROP-002].

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-VITTALDEV-RUSSELL-2016],
        [REF-JULIER-UHLMANN-2004].

    """
    procedure = propagate_gaussian_mixture_procedure(
        mean,
        covariance,
        components=components,
        direction=direction,
        sigma=sigma,
        unscented=unscented,
        max_depth=max_depth,
        nonlinearity_threshold=nonlinearity_threshold,
        split_components=split_components,
        max_components=max_components,
    )
    model = batch_model if batch_model is not None else batch_from_pointwise(function)
    return run_procedure(procedure, model, max_batch_size=max_batch_size)


__all__ = [
    "DEFAULT_COMPONENT_SIGMA",
    "GaussianMixturePropagation",
    "GaussianMixtureState",
    "UnivariateSplitting",
    "nonlinearity_index",
    "propagate_gaussian_mixture",
    "propagate_gaussian_mixture_procedure",
    "split_gaussian",
    "univariate_splitting",
]
