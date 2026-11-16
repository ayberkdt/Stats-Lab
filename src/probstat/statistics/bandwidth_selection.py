"""Data-driven bandwidths for Gaussian kernel density estimates.

Three cross-validation criteria select the bandwidth ``h`` of
``f_h(x) = (1/(n h)) sum_i phi((x - x_i)/h)`` from the data:

* ``"likelihood"``: maximize the leave-one-out log-likelihood
  ``L_LOO(h) = sum_i log((1/(n-1)) sum_(j != i) phi_h(x_i - x_j))``
  (Duin 1976). Without leaving ``x_i`` out the log-likelihood grows without
  bound as ``h -> 0``, because every term contains ``phi_h(0) = 1/(h
  sqrt(2 pi))``.
* ``"ucv"``: minimize the unbiased (least-squares) cross-validation score
  ``UCV(h) = 1/(2 n h sqrt(pi)) + (1/(n^2 h sqrt(pi))) sum_(i<j)
  (exp(-D^2/4) - sqrt(8) exp(-D^2/2))``, ``D = (x_i - x_j)/h``, an
  estimate of ``ISE(h) - int f^2`` (Bowman 1984, in the form of
  Scott and Terrell 1987 that R's ``bw.ucv`` evaluates).
* ``"bcv"``: minimize the biased cross-validation score
  ``BCV(h) = 1/(2 n h sqrt(pi)) + (1/(64 n^2 h sqrt(pi))) sum_(i<j)
  exp(-D^2/4)(D^4 - 12 D^2 + 12)`` (Scott and Terrell 1987, R's ``bw.bcv``).

The criterion is evaluated on a logarithmic grid of the search interval, and
the best grid point is refined by Brent's method on the analytic derivative
with respect to ``log h``. The best point is the global optimum for the
likelihood and UCV; for BCV, which decreases towards zero as ``h -> inf``,
it is the interior local minimum of largest ``h`` (the boundary only when
there is none), the minimum that R's ``optimize`` finds in practice. The
default interval is R's ``[0.1 h_os, h_os]`` with the oversmoothed
bandwidth ``h_os = 1.144 s n^(-1/5)`` (up to ``2 h_os`` for the
likelihood). A minimum on an end of the interval is
reported with ``at_boundary=True`` and a ``RuntimeWarning``, as R warns.

Every criterion costs ``O(n^2)`` kernel evaluations; R bins the pairwise
distances (``nb = 1000`` bins) instead, so its values differ from the exact
ones by the binning error.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.optimization import brent_root

METHODS = ("likelihood", "ucv", "bcv")
_SQRT_PI = math.sqrt(math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT8 = math.sqrt(8.0)
_DELMAX = 1000.0
_GRID_SIZE = 41


@dataclass(frozen=True, slots=True)
class BandwidthSelection:
    """Outcome of a cross-validated bandwidth search.

    Args:
        bandwidth: Selected kernel standard deviation (or bandwidth factor
            for the multivariate estimate).
        method: ``"likelihood"``, ``"ucv"``, or ``"bcv"``.
        criterion: Criterion value at ``bandwidth`` (the log-likelihood for
            ``"likelihood"``, the score to minimize otherwise).
        lower: Lower end of the search interval.
        upper: Upper end of the search interval.
        at_boundary: Whether the optimum lies on an end of the interval.
        evaluations: Number of criterion evaluations.

    References:
        [REF-SCOTT-TERRELL-1987], [REF-DUIN-1976].

    """

    bandwidth: float
    method: str
    criterion: float
    lower: float
    upper: float
    at_boundary: bool
    evaluations: int


def _sample(values: Iterable[float]) -> tuple[float, ...]:
    """Validate a sample of at least two finite observations, not all equal.

    Raises:
        ValueError: For fewer than two observations or a constant sample.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    sample = tuple(coerce_finite_sample(values, name="values"))
    if len(sample) < 2:
        raise ValueError("bandwidth selection needs at least two observations")
    if min(sample) == max(sample):
        raise ValueError("bandwidth selection needs a non-constant sample")
    return sample


def _positive_bandwidth(bandwidth: float) -> float:
    """Validate a positive finite bandwidth.

    Raises:
        ValueError: If the bandwidth is not positive and finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    h = float(bandwidth)
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError("bandwidth must be positive and finite")
    return h


def _pair_squares(sample: Sequence[float]) -> tuple[float, ...]:
    """Return the squared differences ``(x_i - x_j)^2`` of all pairs ``i < j``.

    References:
        [EQ-STAT-KDE-005]. No new equation.

    """
    return tuple(
        (sample[i] - sample[j]) ** 2
        for i in range(len(sample))
        for j in range(i + 1, len(sample))
    )


def _row_squares(sample: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Return, for each ``i``, the squared differences to every ``j != i``.

    References:
        [EQ-STAT-KDE-004]. No new equation.

    """
    return tuple(
        tuple((x - y) ** 2 for j, y in enumerate(sample) if j != i)
        for i, x in enumerate(sample)
    )


def _loo_terms(
    rows: Sequence[Sequence[float]], h: float, dimension: int, *, leave_one_out: bool
) -> tuple[float, float]:
    """Return the Gaussian-kernel log-likelihood and its ``log h`` derivative.

    Each row holds the squared (whitened) distances ``q_ij`` from point ``i``
    to the other points; with ``leave_one_out=False`` the zero self-distance
    is added. The log-likelihood is ``sum_i log((1/m) sum_j phi_d(q_ij))``
    with ``phi_d(q) = exp(-q/(2h^2)) / ((2 pi)^(d/2) h^d)`` and ``m`` the
    number of terms; the inner sums are shifted by their smallest distance,
    so they never underflow.

    References:
        [EQ-STAT-KDE-004], [EQ-STAT-KDE-006], [REF-DUIN-1976].

    """
    inverse = 1.0 / (h * h)
    total: list[float] = []
    slope: list[float] = []
    for row in rows:
        distances = list(row) + ([0.0] if not leave_one_out else [])
        nearest = min(distances)
        weights = [math.exp(-0.5 * (q - nearest) * inverse) for q in distances]
        mass = math.fsum(weights)
        total.append(math.log(mass / len(distances)) - 0.5 * nearest * inverse)
        slope.append(
            math.fsum(q * inverse * w for q, w in zip(distances, weights, strict=True))
            / mass
        )
    count = len(rows)
    value = math.fsum(total) - count * dimension * (math.log(h) + _LOG_SQRT_2PI)
    return value, math.fsum(slope) - count * dimension


def kde_loo_log_likelihood(
    values: Iterable[float], bandwidth: float, *, leave_one_out: bool = True
) -> float:
    """Return the cross-validated log-likelihood of a Gaussian KDE bandwidth.

    ``L_LOO(h) = sum_i log((1/(n-1)) sum_(j != i) phi_h(x_i - x_j))``. With
    ``leave_one_out=False`` the in-sample log-likelihood ``sum_i log((1/n)
    sum_j phi_h(x_i - x_j))`` is returned instead; it is at least
    ``n log(1/(n h sqrt(2 pi)))`` and so tends to ``+inf`` as ``h -> 0``,
    which is why the observation must be left out before maximizing.

    Args:
        values: At least two finite observations, not all equal.
        bandwidth: Kernel standard deviation ``h > 0``.
        leave_one_out: Leave ``x_i`` out of its own density estimate.

    Returns:
        The log-likelihood in nats (``-inf`` is not returned; distant points
        give large negative values computed without underflow).

    Raises:
        ValueError: For fewer than two observations, a constant sample, or a
            non-positive bandwidth.

    Equation:
        [EQ-STAT-KDE-004].

    References:
        [REF-DUIN-1976], [REF-SILVERMAN-1986, Section 3.4].

    """
    sample = _sample(values)
    h = _positive_bandwidth(bandwidth)
    return _loo_terms(_row_squares(sample), h, 1, leave_one_out=leave_one_out)[0]


def _ucv_terms(squares: Sequence[float], n: int, h: float) -> tuple[float, float]:
    """Return ``UCV(h)`` and ``d UCV / d log h``.

    With ``s = D^2``: ``T(s) = e^(-s/4) - sqrt(8) e^(-s/2)`` and
    ``-T - 2 s T' = e^(-s/4)(s/2 - 1) + sqrt(8) e^(-s/2)(1 - s)``.

    References:
        [EQ-STAT-KDE-005], [REF-SCOTT-TERRELL-1987].

    """
    inverse = 1.0 / (h * h)
    value_terms: list[float] = []
    slope_terms: list[float] = []
    for square in squares:
        s = square * inverse
        if s >= _DELMAX:
            continue
        quarter = math.exp(-0.25 * s)
        half = math.exp(-0.5 * s)
        value_terms.append(quarter - _SQRT8 * half)
        slope_terms.append(quarter * (0.5 * s - 1.0) + _SQRT8 * half * (1.0 - s))
    scale = 1.0 / (n * h * _SQRT_PI)
    value = scale * (0.5 + math.fsum(value_terms) / n)
    slope = scale * (-0.5 + math.fsum(slope_terms) / n)
    return value, slope


def _bcv_terms(squares: Sequence[float], n: int, h: float) -> tuple[float, float]:
    """Return ``BCV(h)`` and ``d BCV / d log h``.

    With ``s = D^2``: ``B(s) = e^(-s/4)(s^2 - 12 s + 12)`` and
    ``B + 2 s B' = e^(-s/4)(-s^3/2 + 11 s^2 - 42 s + 12)``.

    References:
        [EQ-STAT-KDE-005], [REF-SCOTT-TERRELL-1987].

    """
    inverse = 1.0 / (h * h)
    value_terms: list[float] = []
    slope_terms: list[float] = []
    for square in squares:
        s = square * inverse
        if s >= _DELMAX:
            continue
        quarter = math.exp(-0.25 * s)
        value_terms.append(quarter * (s * s - 12.0 * s + 12.0))
        slope_terms.append(quarter * (-0.5 * s**3 + 11.0 * s * s - 42.0 * s + 12.0))
    scale = 1.0 / (2.0 * n * h * _SQRT_PI)
    value = scale * (1.0 + math.fsum(value_terms) / (32.0 * n))
    slope = scale * (-1.0 - math.fsum(slope_terms) / (32.0 * n))
    return value, slope


def kde_ucv(values: Iterable[float], bandwidth: float) -> float:
    """Return the unbiased (least-squares) cross-validation score ``UCV(h)``.

    ``UCV(h) = 1/(2 n h sqrt(pi)) + (1/(n^2 h sqrt(pi))) sum_(i<j)
    (exp(-D^2/4) - sqrt(8) exp(-D^2/2))``, ``D = (x_i - x_j)/h``: an
    estimate of ``int (f_h - f)^2 - int f^2`` whose minimizer is the
    least-squares cross-validation bandwidth. Pairs with ``D^2 >= 1000``
    contribute nothing (``exp(-250)``), as in R.

    Args:
        values: At least two finite observations, not all equal.
        bandwidth: Kernel standard deviation ``h > 0``.

    Returns:
        The score (lower is better).

    Raises:
        ValueError: For fewer than two observations, a constant sample, or a
            non-positive bandwidth.

    Equation:
        [EQ-STAT-KDE-005].

    References:
        [REF-SCOTT-TERRELL-1987], [REF-BOWMAN-1984].

    """
    sample = _sample(values)
    h = _positive_bandwidth(bandwidth)
    return _ucv_terms(_pair_squares(sample), len(sample), h)[0]


def kde_bcv(values: Iterable[float], bandwidth: float) -> float:
    """Return the biased cross-validation score ``BCV(h)``.

    ``BCV(h) = 1/(2 n h sqrt(pi)) + (1/(64 n^2 h sqrt(pi))) sum_(i<j)
    exp(-D^2/4)(D^4 - 12 D^2 + 12)``, ``D = (x_i - x_j)/h``: the asymptotic
    MISE with ``int f''^2`` replaced by a kernel estimate. It tends to zero
    as ``h -> inf``, so its useful minimum is the local one inside the
    search interval.

    Args:
        values: At least two finite observations, not all equal.
        bandwidth: Kernel standard deviation ``h > 0``.

    Returns:
        The score (lower is better).

    Raises:
        ValueError: For fewer than two observations, a constant sample, or a
            non-positive bandwidth.

    Equation:
        [EQ-STAT-KDE-005].

    References:
        [REF-SCOTT-TERRELL-1987].

    """
    sample = _sample(values)
    h = _positive_bandwidth(bandwidth)
    return _bcv_terms(_pair_squares(sample), len(sample), h)[0]


def oversmoothed_bandwidth(values: Iterable[float]) -> float:
    """Return ``h_os = 1.144 s n^(-1/5)``, the upper end of R's search interval.

    Args:
        values: At least two finite observations, not all equal.

    Returns:
        The oversmoothed bandwidth with the ``n - 1`` standard deviation.

    Raises:
        ValueError: For fewer than two observations or a constant sample.

    Equation:
        [EQ-STAT-KDE-005].

    References:
        [REF-SCOTT-TERRELL-1987], [REF-R-CORE-TEAM].

    """
    sample = _sample(values)
    n = len(sample)
    centre = math.fsum(sample) / n
    sd = math.sqrt(math.fsum((v - centre) ** 2 for v in sample) / (n - 1))
    return 1.144 * sd * math.pow(n, -0.2)


def _maximize_log_bandwidth(
    criterion: Callable[[float], tuple[float, float]],
    lower: float,
    upper: float,
    *,
    grid_size: int = _GRID_SIZE,
    largest_local: bool = False,
) -> tuple[float, float, bool, int]:
    """Maximize ``c(h)`` given ``(value, d value / d log h)`` by grid and Brent.

    The criterion is evaluated at ``grid_size`` log-spaced points of
    ``[lower, upper]``. The best grid point is the global maximum, or with
    ``largest_local=True`` the interior local maximum of largest ``h`` (the
    global one when there is no interior local maximum). When it is
    interior, the root of the derivative in the two adjacent cells is found
    by Brent's method on ``log h``.

    Returns:
        ``(h, value, at_boundary, evaluations)``.

    References:
        [EQ-STAT-KDE-004], [REF-BRENT-1973, Chapter 4].

    """
    log_low, log_high = math.log(lower), math.log(upper)
    step = (log_high - log_low) / (grid_size - 1)
    grid = [log_low + index * step for index in range(grid_size)]
    grid[-1] = log_high
    values = [criterion(math.exp(u))[0] for u in grid]
    evaluations = grid_size
    best = max(range(grid_size), key=values.__getitem__)
    if largest_local:
        local = [
            index
            for index in range(1, grid_size - 1)
            if values[index] >= values[index - 1] and values[index] >= values[index + 1]
        ]
        if local:
            best = local[-1]
    if best in (0, grid_size - 1):
        return math.exp(grid[best]), values[best], True, evaluations

    def slope(u: float) -> float:
        """Return ``d c / d log h`` at ``h = e^u``.

        References:
            [EQ-STAT-KDE-004]. No new equation.

        """
        nonlocal evaluations
        evaluations += 1
        return float(criterion(math.exp(u))[1])

    left, right = grid[best - 1], grid[best + 1]
    slope_left, slope_right = slope(left), slope(right)
    if slope_left > 0.0 > slope_right:
        root = brent_root(slope, left, right, tolerance=1e-12).root
    else:
        root = grid[best]
    h = math.exp(root)
    value = criterion(h)[0]
    evaluations += 1
    if value < values[best]:
        h, value = math.exp(grid[best]), values[best]
    return h, value, False, evaluations


def select_kde_bandwidth(
    values: Iterable[float],
    *,
    method: str = "likelihood",
    lower: float | None = None,
    upper: float | None = None,
    grid_size: int = _GRID_SIZE,
) -> BandwidthSelection:
    """Select a Gaussian-kernel bandwidth by cross-validation.

    ``"likelihood"`` maximizes the leave-one-out log-likelihood
    (:func:`kde_loo_log_likelihood`); ``"ucv"`` and ``"bcv"`` minimize
    :func:`kde_ucv` and :func:`kde_bcv` (R's ``bw.ucv`` and ``bw.bcv``
    criteria, evaluated on exact rather than binned distances). The default
    search interval is ``[0.1 h_os, h_os]`` (``[0.1 h_os, 2 h_os]`` for the
    likelihood) with ``h_os`` from :func:`oversmoothed_bandwidth`.

    Args:
        values: At least two finite observations, not all equal.
        method: ``"likelihood"``, ``"ucv"``, or ``"bcv"``.
        lower: Lower end of the search interval.
        upper: Upper end of the search interval, above ``lower``.
        grid_size: Number of log-spaced grid points (at least 3).

    Returns:
        :class:`BandwidthSelection`.

    Raises:
        ValueError: For an invalid sample, method, interval, or grid size.

    Warns:
        RuntimeWarning: When the optimum lies on an end of the interval, for
            example when every observation is repeated (heavily rounded
            data), which makes the leave-one-out likelihood grow without
            bound as ``h -> 0``.

    Equation:
        [EQ-STAT-KDE-004], [EQ-STAT-KDE-005].

    References:
        [REF-DUIN-1976], [REF-BOWMAN-1984], [REF-SCOTT-TERRELL-1987],
        [REF-R-CORE-TEAM].

    """
    sample = _sample(values)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 3:
        raise ValueError("grid_size must be an integer of at least 3")
    reference = oversmoothed_bandwidth(sample)
    low = 0.1 * reference if lower is None else _positive_bandwidth(lower)
    if upper is None:
        high = (2.0 if method == "likelihood" else 1.0) * reference
    else:
        high = _positive_bandwidth(upper)
    if not low < high:
        raise ValueError("the search interval must satisfy lower < upper")
    n = len(sample)
    if method == "likelihood":
        rows = _row_squares(sample)

        def criterion(h: float) -> tuple[float, float]:
            """Return the LOO log-likelihood and its ``log h`` derivative.

            References:
                [EQ-STAT-KDE-004]. No new equation.

            """
            return _loo_terms(rows, h, 1, leave_one_out=True)

    else:
        squares = _pair_squares(sample)
        terms = _ucv_terms if method == "ucv" else _bcv_terms

        def criterion(h: float) -> tuple[float, float]:
            """Return the negated score and its ``log h`` derivative.

            References:
                [EQ-STAT-KDE-005]. No new equation.

            """
            value, slope = terms(squares, n, h)
            return -value, -slope

    h, value, at_boundary, evaluations = _maximize_log_bandwidth(
        criterion, low, high, grid_size=grid_size, largest_local=method == "bcv"
    )
    if at_boundary:
        warnings.warn(
            f"the {method} bandwidth optimum lies on an end of the search "
            f"interval [{low:.6g}, {high:.6g}]",
            RuntimeWarning,
            stacklevel=2,
        )
    return BandwidthSelection(
        bandwidth=h,
        method=method,
        criterion=value if method == "likelihood" else -value,
        lower=low,
        upper=high,
        at_boundary=at_boundary,
        evaluations=evaluations,
    )
