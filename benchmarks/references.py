"""Reference values computed without probstat.

Every function here uses only the standard library (``math``, ``random``), so
a defect in the library cannot contaminate the answer it is judged against.
Closed forms are used where they exist; otherwise a one-dimensional composite
Simpson rule with enough panels that its error is far below any sampling error
in the benchmark (checked against a second rule in ``tests/test_benchmarks.py``).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence

SQRT2 = math.sqrt(2.0)


def normal_tail(x: float) -> float:
    """Return ``Phi(-x) = P(Z > x)`` accurately in the upper tail."""
    return 0.5 * math.erfc(x / SQRT2)


def normal_pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def simpson(
    function: Callable[[float], float], lower: float, upper: float, panels: int
) -> float:
    """Return the composite Simpson approximation of a 1-D integral.

    Args:
        function: Smooth integrand.
        lower: Lower limit.
        upper: Upper limit.
        panels: Even number of sub-intervals.

    Returns:
        The integral estimate.

    """
    if panels < 2 or panels % 2:
        raise ValueError("panels must be an even integer of at least two")
    step = (upper - lower) / panels
    total = function(lower) + function(upper)
    total += 4.0 * math.fsum(
        function(lower + (2 * i - 1) * step) for i in range(1, panels // 2 + 1)
    )
    total += 2.0 * math.fsum(
        function(lower + 2 * i * step) for i in range(1, panels // 2)
    )
    return total * step / 3.0


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def linear_failure_probability(beta: float) -> float:
    """Return ``P_f = Phi(-beta)`` of ``G = beta - alpha . U`` with unit ``alpha``."""
    return normal_tail(beta)


def parabolic_failure_probability(
    beta: float, curvature: float, panels: int = 4000
) -> float:
    """Return ``P_f`` of ``G = beta + kappa u1^2 / 2 - u2`` in standard space.

    Conditioning on ``u1 = z`` gives ``P_f = E[Phi(-(beta + kappa z^2 / 2))]``,
    integrated over ``[-12, 12]`` (the omitted normal mass is below 1e-32).
    """
    return simpson(
        lambda z: normal_pdf(z) * normal_tail(beta + 0.5 * curvature * z * z),
        -12.0,
        12.0,
        panels,
    )


def four_branch(u: Sequence[float], k: float = 6.0) -> float:
    """Return the four-branch series-system limit state (Echard et al. 2011)."""
    x1, x2 = u[0], u[1]
    s = 0.1 * (x1 - x2) ** 2
    return min(
        3.0 + s - (x1 + x2) / SQRT2,
        3.0 + s + (x1 + x2) / SQRT2,
        (x1 - x2) + k / SQRT2,
        (x2 - x1) + k / SQRT2,
    )


def four_branch_failure_probability(k: float = 6.0, panels: int = 4000) -> float:
    """Return the four-branch ``P_f`` by a 1-D quadrature in rotated coordinates.

    With ``a = (x1 + x2)/sqrt 2`` and ``b = (x1 - x2)/sqrt 2`` (independent
    standard normals) the system fails when ``|b| >= k/2`` or
    ``|a| >= 3 + 0.2 b^2``, so
    ``P_f = 2 Phi(-k/2) + int_{|b| < k/2} phi(b) 2 Phi(-(3 + 0.2 b^2)) db``.
    """
    half = 0.5 * k
    inner = simpson(
        lambda b: normal_pdf(b) * 2.0 * normal_tail(3.0 + 0.2 * b * b),
        -half,
        half,
        panels,
    )
    return 2.0 * normal_tail(half) + inner


def four_branch_monte_carlo(samples: int, seed: int) -> tuple[float, float]:
    """Return a crude Monte Carlo ``(P_f, standard error)`` of the four-branch system.

    Uses :class:`random.Random` directly (no probstat code) so it is an
    independent check of :func:`four_branch_failure_probability`.
    """
    generator = random.Random(seed)
    gauss = generator.gauss
    failures = 0
    for _ in range(samples):
        if four_branch((gauss(0.0, 1.0), gauss(0.0, 1.0))) <= 0.0:
            failures += 1
    probability = failures / samples
    return probability, math.sqrt(probability * (1.0 - probability) / samples)


#: Means and standard deviations of the oscillator inputs except the load
#: ``F1``: mass, the two spring constants, the yield displacement, and the
#: pulse duration (Gayton, Bourinet, and Lemaire 2003).
OSCILLATOR_FIXED_LAWS = {
    "m": (1.0, 0.05),
    "c1": (1.0, 0.1),
    "c2": (0.1, 0.01),
    "r": (0.5, 0.05),
    "t1": (1.0, 0.2),
}


def oscillator(m: float, c1: float, c2: float, r: float, f1: float, t1: float) -> float:
    """Return the non-linear oscillator limit state ``3 r - |peak displacement|``.

    ``g = 3 r - |2 F1 / (m w0^2) sin(w0 t1 / 2)|`` with
    ``w0 = sqrt((c1 + c2) / m)``; failure is ``g <= 0``.
    """
    omega = math.sqrt((c1 + c2) / m)
    return 3.0 * r - abs(2.0 * f1 / (m * omega * omega) * math.sin(omega * t1 / 2.0))


def oscillator_importance_sampling(
    samples: int,
    seed: int,
    force_mean: float,
    force_std: float,
    center: Sequence[float],
) -> tuple[float, float]:
    """Return an importance-sampling ``(P_f, standard error)`` of the oscillator.

    The six inputs ``(m, c1, c2, r, F1, t1)`` are independent normals, written
    as ``x_i = mu_i + sigma_i u_i`` with standard normal ``u``. Samples are
    drawn from ``N(center, I)`` in ``u`` and weighted by
    ``phi(u) / phi(u - c) = exp(|c|^2 / 2 - u . c)``. The estimator is
    unbiased for any ``center``; a centre at the design point keeps its
    variance small. Uses :class:`random.Random` only (no probstat code).
    """
    laws = (
        OSCILLATOR_FIXED_LAWS["m"],
        OSCILLATOR_FIXED_LAWS["c1"],
        OSCILLATOR_FIXED_LAWS["c2"],
        OSCILLATOR_FIXED_LAWS["r"],
        (force_mean, force_std),
        OSCILLATOR_FIXED_LAWS["t1"],
    )
    shift = tuple(center)
    half_norm = 0.5 * math.fsum(c * c for c in shift)
    generator = random.Random(seed)
    gauss = generator.gauss
    total = 0.0
    squares = 0.0
    for _ in range(samples):
        u = [c + gauss(0.0, 1.0) for c in shift]
        x = [mu + sd * value for (mu, sd), value in zip(laws, u, strict=True)]
        if oscillator(*x) <= 0.0:
            weight = math.exp(
                half_norm - math.fsum(a * c for a, c in zip(u, shift, strict=True))
            )
            total += weight
            squares += weight * weight
    probability = total / samples
    variance = max(squares / samples - probability * probability, 0.0)
    return probability, math.sqrt(variance / (samples - 1))


def lognormal_failure_probability(
    mu_r: float, sigma_r: float, mu_s: float, sigma_s: float
) -> float:
    """Return ``P(R <= S)`` for independent lognormal capacity and demand.

    ``ln R - ln S ~ N(mu_r - mu_s, sigma_r^2 + sigma_s^2)``, so
    ``P_f = Phi(-(mu_r - mu_s) / sqrt(sigma_r^2 + sigma_s^2))``.
    """
    return normal_tail((mu_r - mu_s) / math.hypot(sigma_r, sigma_s))


# ---------------------------------------------------------------------------
# Sensitivity and propagation
# ---------------------------------------------------------------------------


def ishigami(x: Sequence[float], a: float = 7.0, b: float = 0.1) -> float:
    """Return the Ishigami function ``sin x1 + a sin^2 x2 + b x3^4 sin x1``."""
    return math.sin(x[0]) + a * math.sin(x[1]) ** 2 + b * x[2] ** 4 * math.sin(x[0])


def ishigami_moments(a: float = 7.0, b: float = 0.1) -> tuple[float, float]:
    """Return the exact mean ``a/2`` and variance of Ishigami on ``U(-pi, pi)^3``."""
    pi4 = math.pi**4
    variance = a * a / 8.0 + b * pi4 / 5.0 + b * b * pi4 * pi4 / 18.0 + 0.5
    return 0.5 * a, variance


def ishigami_indices(
    a: float = 7.0, b: float = 0.1
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return exact first-order and total Sobol indices of Ishigami.

    ``V1 = (1 + b pi^4/5)^2 / 2``, ``V2 = a^2/8``, ``V3 = 0`` and the only
    interaction ``V13 = 8 b^2 pi^8 / 225`` (Saltelli et al. 2008, Sec. 4.9).
    """
    _, variance = ishigami_moments(a, b)
    pi4 = math.pi**4
    v1 = 0.5 * (1.0 + b * pi4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * pi4 * pi4 / 225.0
    first = (v1 / variance, v2 / variance, 0.0)
    total = ((v1 + v13) / variance, v2 / variance, v13 / variance)
    return first, total


def g_function(x: Sequence[float], a: Sequence[float]) -> float:
    """Return the Sobol G-function ``prod (|4 x_i - 2| + a_i) / (1 + a_i)``."""
    return math.prod(
        (abs(4.0 * xi - 2.0) + ai) / (1.0 + ai) for xi, ai in zip(x, a, strict=True)
    )


def g_function_indices(
    a: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return exact first-order and total indices of the G-function.

    ``V_i = 1 / (3 (1 + a_i)^2)``, ``V = prod (1 + V_i) - 1``,
    ``S_i = V_i / V`` and ``S_Ti = V_i prod_{j != i} (1 + V_j) / V``.
    """
    partial = [1.0 / (3.0 * (1.0 + ai) ** 2) for ai in a]
    variance = math.prod(1.0 + v for v in partial) - 1.0
    first = tuple(v / variance for v in partial)
    total = tuple(
        v * math.prod(1.0 + w for j, w in enumerate(partial) if j != i) / variance
        for i, v in enumerate(partial)
    )
    return first, total


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve a small dense linear system by Gaussian elimination with pivoting."""
    size = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, size):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, size + 1):
                rows[r][c] -= factor * rows[col][c]
    solution = [0.0] * size
    for r in range(size - 1, -1, -1):
        tail = math.fsum(rows[r][c] * solution[c] for c in range(r + 1, size))
        solution[r] = (rows[r][size] - tail) / rows[r][r]
    return solution


def linear_gaussian_shapley_reference(
    coefficients: Sequence[float], covariance: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """Return Shapley effects of ``Y = b'X``, ``X ~ N(mu, Sigma)``, independently.

    ``v(J) = Var(E[Y | X_J]) = s_J' Sigma_JJ^-1 s_J`` with ``s = Sigma b``;
    the Shapley values of ``v`` divided by ``Var Y = b' Sigma b``.
    """
    d = len(coefficients)
    s = [
        math.fsum(covariance[i][j] * coefficients[j] for j in range(d))
        for i in range(d)
    ]
    variance = math.fsum(b * si for b, si in zip(coefficients, s, strict=True))

    def value(subset: tuple[int, ...]) -> float:
        """Return ``Var(E[Y | X_J])`` for the coalition ``J``."""
        if not subset:
            return 0.0
        block = [[covariance[i][j] for j in subset] for i in subset]
        part = [s[i] for i in subset]
        return math.fsum(p * q for p, q in zip(part, _solve(block, part), strict=True))

    effects = []
    for i in range(d):
        others = [j for j in range(d) if j != i]
        total = 0.0
        for size in range(d):
            weight = (
                math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            )
            for subset in itertools.combinations(others, size):
                with_i = tuple(sorted(subset + (i,)))
                total += weight * (value(with_i) - value(subset))
        effects.append(total / variance)
    return tuple(effects)


def exponential_integral(rates: Sequence[float]) -> tuple[float, float]:
    """Return mean and variance of ``f(x) = exp(sum c_i x_i)`` on ``U(0, 1)^d``.

    ``E f = prod (e^c - 1)/c`` and ``E f^2 = prod (e^{2c} - 1)/(2c)``.
    """
    mean = math.prod(math.expm1(c) / c for c in rates)
    second = math.prod(math.expm1(2.0 * c) / (2.0 * c) for c in rates)
    return mean, second - mean * mean


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def linear_gaussian_posterior(
    conditions: Sequence[float],
    observations: Sequence[float],
    noise_sd: float,
    prior_sd: float,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """Return the exact posterior of ``(a, b)`` in ``y = a + b t + eps``.

    Prior ``N(0, prior_sd^2 I)``, known noise: precision
    ``P = I/prior_sd^2 + X'X/sigma^2``, mean ``P^-1 X'y / sigma^2``.

    Returns:
        Posterior mean and covariance.

    """
    tau = 1.0 / (noise_sd * noise_sd)
    n = len(conditions)
    st = math.fsum(conditions)
    stt = math.fsum(t * t for t in conditions)
    p00 = 1.0 / prior_sd**2 + n * tau
    p01 = st * tau
    p11 = 1.0 / prior_sd**2 + stt * tau
    r0 = math.fsum(observations) * tau
    r1 = math.fsum(t * y for t, y in zip(conditions, observations, strict=True)) * tau
    det = p00 * p11 - p01 * p01
    c00, c01, c11 = p11 / det, -p01 / det, p00 / det
    mean = (c00 * r0 + c01 * r1, c01 * r0 + c11 * r1)
    return mean, ((c00, c01), (c01, c11))


def normal_quantile(probability: float) -> float:
    """Return ``Phi^-1(p)`` by bisection on ``erfc`` (no library code)."""
    lower, upper = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if 1.0 - normal_tail(middle) < probability:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


# ---------------------------------------------------------------------------
# Inverse problems and data assimilation
# ---------------------------------------------------------------------------


def backward_heat_solution(size: int) -> tuple[float, ...]:
    """Return the exact initial state ``(27/4) s (1 - s)^2`` at ``s_j = j/(n+1)``."""
    h = 1.0 / (size + 1)
    return tuple(6.75 * s * (1.0 - s) ** 2 for s in ((j + 1) * h for j in range(size)))


def deriv2_solution(size: int) -> tuple[float, ...]:
    """Return the exact ``deriv2`` solution ``f(t) = t`` at the midpoints."""
    return tuple((i + 0.5) / size for i in range(size))


def deblur_truth(size: int, refinement: int) -> tuple[float, ...]:
    """Return the coarse-grid truth of the 1-D deblurring problem ``L04``.

    The truth is the piecewise-constant profile (0 on ``[0, 0.25)``, 1 on
    ``[0.25, 0.5)``, 0.3 on ``[0.5, 0.7)``, 0.8 on ``[0.7, 1]``) sampled at the
    centres of a grid ``refinement`` times finer than the ``size`` coarse
    cells and averaged over each coarse cell. This is the definition stated in
    the docstring of ``probstat.inverse.deconvolution_benchmark``, written
    out again here without probstat.
    """

    def profile(s: float) -> float:
        """Return the piecewise-constant truth at ``s``."""
        if s < 0.25:
            return 0.0
        if s < 0.5:
            return 1.0
        if s < 0.7:
            return 0.3
        return 0.8

    fine = size * refinement
    samples = [profile((j + 0.5) / fine) for j in range(fine)]
    return tuple(
        math.fsum(samples[i * refinement : (i + 1) * refinement]) / refinement
        for i in range(size)
    )


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the lower Cholesky factor ``L`` (``L L' = matrix``) of an SPD matrix."""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - math.fsum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if total <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][i] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _cholesky_solve(
    lower: Sequence[Sequence[float]], rhs: Sequence[float]
) -> list[float]:
    """Solve ``L L' x = rhs`` by forward and backward substitution."""
    size = len(rhs)
    forward = [0.0] * size
    for i in range(size):
        forward[i] = (
            rhs[i] - math.fsum(lower[i][k] * forward[k] for k in range(i))
        ) / lower[i][i]
    solution = [0.0] * size
    for i in range(size - 1, -1, -1):
        tail = math.fsum(lower[k][i] * solution[k] for k in range(i + 1, size))
        solution[i] = (forward[i] - tail) / lower[i][i]
    return solution


def correlated_gaussian_posterior_mean(
    data: Sequence[float],
    size: int,
    observations: int,
    length_scale: float,
    prior_variance: float,
    noise_level: float,
) -> tuple[float, ...]:
    """Return the exact posterior mean of the correlated-Gaussian problem ``L06``.

    Model (the definition in the docstring of
    ``probstat.inverse.correlated_gaussian_benchmark``, rewritten here without
    probstat): ``x ~ N(0, C0)`` with ``C0_ij = v exp(-|s_i - s_j| / l)`` on
    ``s_i = i / (n - 1)``; observations ``y_k`` are the mean of three
    neighbouring points centred at ``1 + round(k (n - 3) / max(m - 1, 1))``
    plus ``N(0, s^2)`` noise. The posterior mean solves the normal equations
    ``(A'A / s^2 + C0^-1) x = A'y / s^2``; ``C0^-1`` and the solve both use
    a Cholesky factorization. The *data* are an input (they come from
    probstat's seeded stream); everything computed from them is independent.
    """
    grid = [i / (size - 1) for i in range(size)]
    prior = [
        [prior_variance * math.exp(-abs(a - b) / length_scale) for b in grid]
        for a in grid
    ]
    rows = []
    for k in range(observations):
        centre = 1 + round(k * (size - 3) / max(observations - 1, 1))
        rows.append([1.0 / 3.0 if abs(j - centre) <= 1 else 0.0 for j in range(size)])
    precision_factor = _cholesky(prior)
    inverse_columns = [
        _cholesky_solve(precision_factor, [1.0 if i == j else 0.0 for i in range(size)])
        for j in range(size)
    ]
    tau = 1.0 / (noise_level * noise_level)
    system = [
        [
            inverse_columns[i][j]
            + tau * math.fsum(rows[k][i] * rows[k][j] for k in range(observations))
            for j in range(size)
        ]
        for i in range(size)
    ]
    rhs = [
        tau * math.fsum(rows[k][i] * data[k] for k in range(observations))
        for i in range(size)
    ]
    return tuple(_cholesky_solve(_cholesky(system), rhs))


Matrix2 = tuple[tuple[float, float], tuple[float, float]]


def kalman_filter_2d(
    transition: Matrix2,
    process: Matrix2,
    observation_variance: float,
    initial_mean: tuple[float, float],
    initial_covariance: Matrix2,
    observations: Sequence[float],
) -> tuple[tuple[float, float], Matrix2]:
    """Return the final filtered mean and covariance of a 2-state model.

    The state is observed through its first component, ``y_k = x_k[0] + v_k``;
    each step predicts ``m = F m``, ``P = F P F' + Q`` and updates with the
    scalar gain ``K = P[:, 0] / (P[0][0] + r)`` (plain 2 x 2 algebra).
    """
    (f00, f01), (f10, f11) = transition
    m0, m1 = initial_mean
    (p00, p01), (_, p11) = initial_covariance
    for y in observations:
        m0, m1 = f00 * m0 + f01 * m1, f10 * m0 + f11 * m1
        # F P F' + Q, P symmetric.
        a00 = f00 * p00 + f01 * p01
        a01 = f00 * p01 + f01 * p11
        a10 = f10 * p00 + f11 * p01
        a11 = f10 * p01 + f11 * p11
        p00 = a00 * f00 + a01 * f01 + process[0][0]
        p01 = a00 * f10 + a01 * f11 + process[0][1]
        p11 = a10 * f10 + a11 * f11 + process[1][1]
        s = p00 + observation_variance
        k0, k1 = p00 / s, p01 / s
        innovation = y - m0
        m0, m1 = m0 + k0 * innovation, m1 + k1 * innovation
        p00, p01, p11 = (1.0 - k0) * p00, (1.0 - k0) * p01, p11 - k1 * p01
    return (m0, m1), ((p00, p01), (p01, p11))


#: Additive discs of the tomography phantom (catalog case L07b) in fractions of
#: the grid side, ``(cx, cy, radius, value)``: copied from the docstring-level
#: definition of ``probstat.inverse.masked_tomography_benchmark`` so that the
#: reference below shares the phantom but not the code.
TOMOGRAPHY_DISCS = (
    (0.35, 0.40, 0.20, 1.0),
    (0.65, 0.65, 0.15, 0.5),
    (0.70, 0.30, 0.08, 0.8),
)


def _disc_rectangle_area(
    cx: float, cy: float, radius: float, x0: float, x1: float, y0: float, y1: float
) -> float:
    """Return the area of a disc intersected with ``[x0, x1] x [y0, y1]``.

    The vertical overlap length is integrated in ``x`` by the composite
    midpoint rule between its breakpoints (disc edges ``cx +- R`` and the
    abscissae where the circle crosses ``y0`` or ``y1``); with 400 cells per
    piece the error is below 1e-6 of the pixel area for the phantom used
    here (square-root behaviour at the disc edges).
    """
    lo, hi = max(x0, cx - radius), min(x1, cx + radius)
    if lo >= hi:
        return 0.0
    breaks = {lo, hi}
    for y in (y0, y1):
        if abs(y - cy) < radius:
            half = math.sqrt(radius * radius - (y - cy) ** 2)
            for x in (cx - half, cx + half):
                if lo < x < hi:
                    breaks.add(x)
    points = sorted(breaks)
    area = 0.0
    cells = 400
    for a, b in zip(points[:-1], points[1:], strict=True):
        width = (b - a) / cells
        for i in range(cells):
            x = a + (i + 0.5) * width
            half = math.sqrt(max(radius * radius - (x - cx) ** 2, 0.0))
            top, bottom = min(y1, cy + half), max(y0, cy - half)
            if top > bottom:
                area += (top - bottom) * width
    return area


def disc_phantom_cell_averages(size: int) -> tuple[float, ...]:
    """Return the exact pixel averages of the disc phantom (row-major).

    Pixel ``(iy, ix)`` covers ``[ix/n, (ix+1)/n] x [iy/n, (iy+1)/n]`` of the
    unit square; its value is the sum over discs of ``value * area / h^2``.
    The library benchmark's withheld truth samples 4 x 4 points per pixel
    instead, so the two differ by the sampling error of that rule.
    """
    h = 1.0 / size
    values = []
    for iy in range(size):
        for ix in range(size):
            total = 0.0
            for cx, cy, radius, value in TOMOGRAPHY_DISCS:
                total += value * _disc_rectangle_area(
                    cx, cy, radius, ix * h, (ix + 1) * h, iy * h, (iy + 1) * h
                )
            values.append(total / (h * h))
    return tuple(values)


def exponential_decay_least_squares(
    times: Sequence[float], data: Sequence[float]
) -> tuple[float, float]:
    """Return the least-squares ``(a, r)`` of ``y = a exp(-r t)`` by profiling.

    For fixed ``r`` the best amplitude is ``a(r) = sum y e / sum e^2`` with
    ``e = exp(-r t)``; the profiled residual is minimized over ``r`` by a
    coarse scan on ``(0, 20]`` followed by golden-section refinement to a
    bracket width of 1e-13. Plain standard-library arithmetic.
    """

    def profile(rate: float) -> tuple[float, float]:
        """Return ``(residual sum of squares, a(r))``."""
        e = [math.exp(-rate * t) for t in times]
        amplitude = math.fsum(y * v for y, v in zip(data, e, strict=True)) / math.fsum(
            v * v for v in e
        )
        rss = math.fsum((y - amplitude * v) ** 2 for y, v in zip(data, e, strict=True))
        return rss, amplitude

    scan = [20.0 * (i + 1) / 2000 for i in range(2000)]
    best = min(range(len(scan)), key=lambda i: profile(scan[i])[0])
    low = scan[max(best - 1, 0)] if best > 0 else 1e-9
    high = scan[min(best + 1, len(scan) - 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c = high - golden * (high - low)
    d = low + golden * (high - low)
    while high - low > 1e-13:
        if profile(c)[0] < profile(d)[0]:
            high, d = d, c
            c = high - golden * (high - low)
        else:
            low, c = c, d
            d = low + golden * (high - low)
    rate = 0.5 * (low + high)
    return profile(rate)[1], rate


def squared_map_least_squares(data: Sequence[float]) -> tuple[float, ...]:
    """Return ``|x|`` of the least-squares fit of ``y_i = x_i^2`` (catalog N02).

    The problem separates per entry: ``(x_i^2 - y_i)^2`` is minimized by
    ``x_i = +-sqrt(y_i)`` for ``y_i > 0`` and by ``x_i = 0`` otherwise. Every
    sign pattern fits equally well, so only the magnitudes are a reference.
    """
    return tuple(math.sqrt(y) if y > 0.0 else 0.0 for y in data)


def log_conductivity_cell_averages(cells: int) -> tuple[float, ...]:
    """Return exact cell averages of ``0.5 sin(2 pi s) + 0.3 s`` (catalog P01).

    Cell ``i`` covers ``[i/n, (i+1)/n]``; the average is the integral
    ``-cos(2 pi s) / (4 pi) + 0.15 s^2`` between the cell ends divided by the
    width. The library benchmark averages fine-cell midpoints instead, so the
    two differ by that midpoint rule's error.
    """
    h = 1.0 / cells

    def primitive(s: float) -> float:
        """Return an antiderivative of the log-conductivity."""
        return -math.cos(2.0 * math.pi * s) / (4.0 * math.pi) + 0.15 * s * s

    return tuple((primitive((i + 1) * h) - primitive(i * h)) / h for i in range(cells))


def heat_initial_profile(cells: int) -> tuple[float, ...]:
    """Return ``exp(-60 (s - 0.4)^2) + 0.5 s (1 - s)`` at the interior nodes.

    Nodes ``s_j = (j + 1) / (n + 1)``, ``j = 0..n-1`` (catalog P02).
    """
    h = 1.0 / (cells + 1)
    return tuple(
        math.exp(-60.0 * ((j + 1) * h - 0.4) ** 2)
        + 0.5 * (j + 1) * h * (1.0 - (j + 1) * h)
        for j in range(cells)
    )
