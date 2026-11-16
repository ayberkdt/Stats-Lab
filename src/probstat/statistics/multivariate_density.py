"""Multivariate Gaussian kernel density estimation with a covariance-shaped kernel.

The estimate of a sample ``x_1..x_N`` in ``R^d`` is

    ``f_h(x) = (1/N) sum_i phi_d(x - x_i; h^2 R)``

where ``R`` is the sample covariance matrix (divisor ``N - 1``) and ``h`` a
scalar bandwidth factor, so every kernel has the shape of the data cloud
(Silverman 1986, Chapter 4). Rules of thumb: Silverman's
``h = (4 / (N (d + 2)))^(1/(d+4))`` (the normal-reference AMISE factor) and
Scott's ``h = N^(-1/(d+4))``. The factor can also be chosen by maximizing the
leave-one-out log-likelihood
``sum_i log((1/(N-1)) sum_(j != i) phi_d(x_i - x_j; h^2 R))``.

Evaluation whitens once with the Cholesky factor ``R = L L'``, so a density
value costs ``O(N d^2)`` and the likelihood search ``O(N^2 d)`` per factor.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import cholesky, solve_lower

from .bandwidth_selection import BandwidthSelection, _loo_terms, _maximize_log_bandwidth

RULES = ("silverman", "scott", "likelihood")
_LOG_2PI = math.log(2.0 * math.pi)


def multivariate_kde_factor(
    size: int, dimension: int, *, rule: str = "silverman"
) -> float:
    """Return a rule-of-thumb bandwidth factor for a covariance-shaped kernel.

    ``silverman``: ``(4 / (N (d + 2)))^(1/(d+4))``; ``scott``:
    ``N^(-1/(d+4))``. For ``d = 1`` Silverman's factor is ``(4/3)^(1/5)
    N^(-1/5) = 1.0592 N^(-1/5)``, the normal reference rule.

    Args:
        size: Sample size ``N >= 1``.
        dimension: Dimension ``d >= 1``.
        rule: ``"silverman"`` or ``"scott"``.

    Returns:
        The factor ``h``; the kernel covariance is ``h^2 R``.

    Raises:
        ValueError: For a non-positive size or dimension or an unknown rule.

    Equation:
        [EQ-STAT-KDE-006].

    References:
        [REF-SILVERMAN-1986, Chapter 4], [REF-SCOTT-2015].

    """
    n = require_non_negative_integer(size, name="size")
    d = require_non_negative_integer(dimension, name="dimension")
    if n < 1 or d < 1:
        raise ValueError("size and dimension must be positive")
    if rule == "silverman":
        return math.pow(4.0 / (n * (d + 2.0)), 1.0 / (d + 4.0))
    if rule == "scott":
        return math.pow(n, -1.0 / (d + 4.0))
    raise ValueError("rule must be 'silverman' or 'scott'")


def _points(points: Iterable[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Validate a sample of finite ``d``-vectors with more points than dimensions.

    Raises:
        ValueError: For ragged, empty, or too few points.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(
        tuple(require_finite_real(value, name="points") for value in row)
        for row in points
    )
    if not rows or not rows[0]:
        raise ValueError("points must be a non-empty sequence of vectors")
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise ValueError("every point must have the same dimension")
    if len(rows) <= dimension:
        raise ValueError("a covariance-shaped kernel needs more points than dimensions")
    return rows


def _covariance(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return the sample mean and covariance (divisor ``N - 1``).

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapter 1]. No new equation.

    """
    n, d = len(rows), len(rows[0])
    mean = tuple(math.fsum(row[k] for row in rows) / n for k in range(d))
    covariance = tuple(
        tuple(
            math.fsum((row[i] - mean[i]) * (row[j] - mean[j]) for row in rows) / (n - 1)
            for j in range(d)
        )
        for i in range(d)
    )
    return mean, covariance


@dataclass(frozen=True, slots=True)
class MultivariateKernelDensity:
    """Gaussian KDE ``f_h(x) = (1/N) sum_i phi_d(x - x_i; h^2 R)``.

    Built by :func:`gaussian_kde_multivariate`.

    Args:
        points: The sample, one tuple per observation.
        factor: Bandwidth factor ``h > 0``.
        covariance: Sample covariance ``R`` (divisor ``N - 1``).
        selection: Cross-validation record when ``h`` was selected by the
            likelihood, else ``None``.

    References:
        [REF-SILVERMAN-1986, Chapter 4].

    """

    points: tuple[tuple[float, ...], ...]
    factor: float
    covariance: tuple[tuple[float, ...], ...]
    selection: BandwidthSelection | None
    _lower: tuple[tuple[float, ...], ...]
    _whitened: tuple[tuple[float, ...], ...]
    _log_det: float

    @property
    def dimension(self) -> int:
        """Return the dimension ``d``.

        References:
            [REF-SILVERMAN-1986, Chapter 4]. No equation is evaluated.

        """
        return len(self.points[0])

    @property
    def kernel_covariance(self) -> tuple[tuple[float, ...], ...]:
        """Return the kernel covariance ``h^2 R``.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Chapter 4].

        """
        h2 = self.factor * self.factor
        return tuple(tuple(h2 * value for value in row) for row in self.covariance)

    def log_pdf(self, point: Sequence[float]) -> float:
        """Return ``log f_h(x)`` by a shifted log-sum-exp over the kernels.

        Args:
            point: A ``d``-vector.

        Returns:
            Log density.

        Raises:
            ValueError: If the point has the wrong dimension or a non-finite
                coordinate.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Chapter 4].

        """
        x = tuple(require_finite_real(value, name="point") for value in point)
        d = self.dimension
        if len(x) != d:
            raise ValueError(f"point must have dimension {d}")
        z = solve_lower(self._lower, x)
        inverse = 1.0 / (self.factor * self.factor)
        squares = [
            math.fsum((a - b) ** 2 for a, b in zip(z, y, strict=True))
            for y in self._whitened
        ]
        nearest = min(squares)
        mass = math.fsum(math.exp(-0.5 * (q - nearest) * inverse) for q in squares)
        normalizer = (
            0.5 * d * _LOG_2PI + d * math.log(self.factor) + 0.5 * self._log_det
        )
        return math.log(mass / len(squares)) - 0.5 * nearest * inverse - normalizer

    def pdf(self, point: Sequence[float]) -> float:
        """Return the density ``f_h(x)``.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Chapter 4].

        """
        return math.exp(self.log_pdf(point))

    @property
    def mean(self) -> tuple[float, ...]:
        """Return the mean of the estimate, the sample mean.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Chapter 4].

        """
        n = len(self.points)
        return tuple(
            math.fsum(row[k] for row in self.points) / n for k in range(self.dimension)
        )

    @property
    def mixture_covariance(self) -> tuple[tuple[float, ...], ...]:
        """Return ``((N - 1)/N + h^2) R``, the covariance of the estimate.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Chapter 4].

        """
        n = len(self.points)
        weight = (n - 1) / n + self.factor * self.factor
        return tuple(tuple(weight * value for value in row) for row in self.covariance)

    def draw(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[float, ...], ...]:
        """Draw from the estimate: a resampled point plus ``h L Z``.

        Args:
            size: Number of draws, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of ``d``-vectors.

        Equation:
            [EQ-STAT-KDE-006].

        References:
            [REF-SILVERMAN-1986, Section 6.4].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        d = self.dimension
        lower = self._lower
        draws: list[tuple[float, ...]] = []
        for _ in range(count):
            centre = generator.choice(self.points)
            normal = [generator.gauss(0.0, 1.0) for _ in range(d)]
            draws.append(
                tuple(
                    centre[i]
                    + self.factor
                    * math.fsum(lower[i][k] * normal[k] for k in range(i + 1))
                    for i in range(d)
                )
            )
        return tuple(draws)


def multivariate_kde_loo_log_likelihood(
    points: Iterable[Sequence[float]], factor: float, *, leave_one_out: bool = True
) -> float:
    """Return the leave-one-out log-likelihood of a bandwidth factor.

    ``sum_i log((1/(N-1)) sum_(j != i) phi_d(x_i - x_j; h^2 R))``; with
    ``leave_one_out=False`` the in-sample version, which grows without bound
    as ``h -> 0``.

    Args:
        points: More than ``d`` finite ``d``-vectors with a non-singular
            sample covariance.
        factor: Bandwidth factor ``h > 0``.
        leave_one_out: Leave ``x_i`` out of its own density estimate.

    Returns:
        The log-likelihood in nats.

    Raises:
        ValueError: For an invalid sample, a singular covariance, or a
            non-positive factor.

    Equation:
        [EQ-STAT-KDE-006].

    References:
        [REF-DUIN-1976], [REF-SILVERMAN-1986, Section 3.4].

    """
    rows = _points(points)
    h = require_finite_real(factor, name="factor")
    if not h > 0.0:
        raise ValueError("factor must be positive")
    _, covariance = _covariance(rows)
    lower = _cholesky(covariance)
    whitened = tuple(solve_lower(lower, row) for row in rows)
    squares = _whitened_rows(whitened)
    value, _ = _loo_terms(squares, h, len(rows[0]), leave_one_out=leave_one_out)
    return value - 0.5 * len(rows) * _log_det_lower(lower)


def _cholesky(
    covariance: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Return the Cholesky factor of the sample covariance.

    Raises:
        ValueError: If the sample covariance is singular.

    References:
        [EQ-LINALG-001]. No new equation.

    """
    try:
        return cholesky(covariance)
    except ValueError:
        raise ValueError(
            "the sample covariance is singular; the points lie in a lower-"
            "dimensional subspace"
        ) from None


def _log_det_lower(lower: Sequence[Sequence[float]]) -> float:
    """Return ``log det(L L') = 2 sum log L_ii``.

    References:
        [EQ-LINALG-001]. No new equation.

    """
    return 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))


def _whitened_rows(
    whitened: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Return, for each ``i``, the squared whitened distances to every ``j != i``.

    References:
        [EQ-STAT-KDE-006]. No new equation.

    """
    return tuple(
        tuple(
            math.fsum((a - b) ** 2 for a, b in zip(x, y, strict=True))
            for j, y in enumerate(whitened)
            if j != i
        )
        for i, x in enumerate(whitened)
    )


def gaussian_kde_multivariate(
    points: Iterable[Sequence[float]],
    *,
    bandwidth: float | str = "silverman",
    lower: float | None = None,
    upper: float | None = None,
) -> MultivariateKernelDensity:
    """Return a Gaussian KDE whose kernel covariance is ``h^2 R``.

    Args:
        points: More than ``d`` finite ``d``-vectors with a non-singular
            sample covariance ``R``.
        bandwidth: Positive factor ``h``, ``"silverman"``, ``"scott"``, or
            ``"likelihood"`` (maximize the leave-one-out log-likelihood on
            ``[lower, upper]``, by default ``[0.1, 2]`` times Silverman's
            factor).
        lower: Lower end of the likelihood search interval.
        upper: Upper end of the likelihood search interval.

    Returns:
        :class:`MultivariateKernelDensity` with ``pdf``, ``log_pdf``,
        ``draw``, ``mean``, and ``mixture_covariance``.

    Raises:
        ValueError: For an invalid sample, a singular covariance, an unknown
            rule, or a non-positive factor or interval.

    Warns:
        RuntimeWarning: When the likelihood optimum lies on an end of the
            search interval.

    Equation:
        [EQ-STAT-KDE-006].

    References:
        [REF-SILVERMAN-1986, Chapter 4],
        [REF-SCOTT-2015], [REF-DUIN-1976].

    """
    rows = _points(points)
    n, d = len(rows), len(rows[0])
    _, covariance = _covariance(rows)
    factor_lower = _cholesky(covariance)
    whitened = tuple(solve_lower(factor_lower, row) for row in rows)
    log_det = _log_det_lower(factor_lower)
    selection: BandwidthSelection | None = None
    if isinstance(bandwidth, str):
        if bandwidth not in RULES:
            raise ValueError(f"bandwidth must be a positive number or one of {RULES}")
        if bandwidth == "likelihood":
            reference = multivariate_kde_factor(n, d, rule="silverman")
            low = 0.1 * reference if lower is None else float(lower)
            high = 2.0 * reference if upper is None else float(upper)
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError("the search interval must satisfy 0 < lower < upper")
            squares = _whitened_rows(whitened)

            def criterion(h: float) -> tuple[float, float]:
                """Return the LOO log-likelihood and its ``log h`` derivative.

                References:
                    [EQ-STAT-KDE-006]. No new equation.

                """
                value, slope = _loo_terms(squares, h, d, leave_one_out=True)
                return value - 0.5 * n * log_det, slope

            h, value, at_boundary, evaluations = _maximize_log_bandwidth(
                criterion, low, high
            )
            if at_boundary:
                warnings.warn(
                    "the likelihood bandwidth optimum lies on an end of the "
                    f"search interval [{low:.6g}, {high:.6g}]",
                    RuntimeWarning,
                    stacklevel=2,
                )
            selection = BandwidthSelection(
                bandwidth=h,
                method="likelihood",
                criterion=value,
                lower=low,
                upper=high,
                at_boundary=at_boundary,
                evaluations=evaluations,
            )
        else:
            h = multivariate_kde_factor(n, d, rule=bandwidth)
    else:
        h = require_finite_real(bandwidth, name="bandwidth")
        if not h > 0.0:
            raise ValueError("bandwidth must be positive")
    return MultivariateKernelDensity(
        rows, h, covariance, selection, factor_lower, whitened, log_det
    )
