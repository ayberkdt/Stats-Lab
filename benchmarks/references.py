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
