"""Localization, adaptive inflation and transport updates for inverse ensembles.

Experimental (added in 0.28).

Extensions of :mod:`probstat.inverse.regularized_ensemble` for ensembles on a
generic forward map ``G: R^n -> R^m`` with a covariance factor of
:mod:`probstat.linalg` for the observation noise ``Gamma``:

* :func:`localized_ensemble_kalman_update`, the stochastic Kalman update with a
  Gaspari-Cohn taper applied as a Schur product to the sample
  parameter-observation and observation-observation covariances. Distances are
  supplied by the caller. Without a taper it is the plain stochastic update, and
  its gain converges to the exact Kalman gain as the ensemble grows.
* :func:`estimate_inflation` and :func:`adaptive_inflation`, a scalar
  multiplicative inflation of the prior spread estimated from the innovation
  statistics of the update itself, with bounds, reported per update.
* :func:`ensemble_transform_particle_filter`, the ensemble transform particle
  filter (ETPF) update: likelihood weights, an exact optimal coupling between
  the weighted and the uniform ensemble (the discrete solver of
  :mod:`probstat.information.transport`, squared Euclidean cost), and the
  deterministic barycentric transform.
* :func:`square_map_reference`, a quadrature reference posterior for the
  bimodal problem ``y = x^2 + noise`` used to compare the methods.

What is and is not claimed. A Schur-product taper of a sample covariance is a
regularization with a documented bias, not a derivation of the exact Bayesian
update; the gain with a taper differs from the exact one by design. The
inflation estimator is a moment estimator and is unbiased only when the
innovation is Gaussian with the stated covariance. The ETPF mean equals the
weighted mean exactly, but the barycentric transform contracts the spread
(the transformed particles are conditional means of the coupling), so the
transformed covariance is not the weighted covariance and is not the
posterior covariance. The stochastic ensemble Kalman update assumes a
Gaussian prediction-parameter relationship and cannot represent a bimodal
posterior ([EQ-INV-ENX-007]).

The Gaspari-Cohn function is re-implemented here rather than imported from
:mod:`probstat.data_assimilation.localization` because the data assimilation
package is a higher layer than this one (``tests/test_architecture.py``).

References:
    [REF-GASPARI-COHN-1999], [REF-EVENSEN-2003], [REF-IGLESIAS-LAW-STUART-2013],
    [REF-PEYRE-CUTURI-2019], [REF-REICH-2013], [REF-ANDERSON-2007],
    [REF-WANG-BISHOP-2003], [EQ-INV-ENX-001].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.information.transport import (
    exact_transport,
    squared_euclidean_cost_matrix,
)
from probstat.inverse.regularized_ensemble import (
    _mean,
    _validate_update_inputs,
    _vector,
)
from probstat.linalg import (
    BlockDiagonalCovariance,
    DenseCovariance,
    DiagonalCovariance,
    Matrix,
    Vector,
    cholesky,
    cholesky_solve,
)

_CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalizedUpdate:
    """Updated ensemble and diagnostics of a localized Kalman update.

    Args:
        ensemble: Updated members; the input members when ``converged`` is
            false.
        analysis_mean: ``x_bar + K (y - g_bar)``, the update of the ensemble
            mean with the (tapered) gain. It does not depend on the
            perturbations; the mean of ``ensemble`` equals it only up to the
            perturbation noise.
        gain: The ``n x m`` gain ``K = (C_xg o rho_xg)(C_gg o rho_gg + alpha
            Gamma)^-1``.
        localized: ``True`` when a taper was applied.
        inflation: The factor ``alpha`` on ``Gamma``.
        converged: ``False`` if the tapered observation-space matrix was not
            positive definite (the taper matrix ``rho_gg`` is then not
            positive semi-definite for the supplied distances).
        stop_reason: ``"ok"`` or ``"tapered_system_not_spd"``.

    References:
        [EQ-INV-ENX-001], [EQ-INV-ENX-002].

    """

    ensemble: tuple[Vector, ...]
    analysis_mean: Vector
    gain: Matrix
    localized: bool
    inflation: float
    converged: bool
    stop_reason: str


@dataclass(frozen=True, slots=True)
class InflationReport:
    """Innovation-based multiplicative inflation of one update.

    Args:
        raw: Unbounded estimate ``(|L^-1 d|^2 - m) / tr(Zw Zw')``.
        factor: The factor applied after the bounds.
        lower: Lower bound.
        upper: Upper bound.
        clipped: ``True`` if a bound changed the raw estimate.
        innovation_norm_squared: ``|L^-1 (y - g_bar)|^2``.
        spread_trace: ``tr(Zw Zw')``, the whitened prediction variance.
        observation_count: ``m``.

    References:
        [EQ-INV-ENX-003].

    """

    raw: float
    factor: float
    lower: float
    upper: float
    clipped: bool
    innovation_norm_squared: float
    spread_trace: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class InflatedEnsemble:
    """Ensemble and predictions after the adaptive multiplicative inflation.

    Args:
        ensemble: Members ``x_bar + sqrt(lambda)(x_j - x_bar)``.
        predictions: ``g_bar + sqrt(lambda)(g_j - g_bar)``.
        report: The :class:`InflationReport` of this update.

    References:
        [EQ-INV-ENX-004].

    """

    ensemble: tuple[Vector, ...]
    predictions: tuple[Vector, ...]
    report: InflationReport


@dataclass(frozen=True, slots=True)
class ETPFResult:
    """Result of one ensemble transform particle filter update.

    Args:
        ensemble: Transformed members ``x_j^+ = J sum_i t_ij x_i``.
        weights: Normalized likelihood weights ``w_i``.
        coupling: Optimal coupling ``t`` (rows ``w_i``, columns ``1/J``).
        weighted_mean: ``sum_i w_i x_i``; equals the mean of ``ensemble`` up
            to rounding.
        mean: Mean of the transformed ensemble.
        weighted_covariance: ``sum_i w_i (x_i - mu_w)(x_i - mu_w)'``.
        covariance: ``(1/J) sum_j (x_j^+ - mean)(x_j^+ - mean)'``; it is
            dominated by ``weighted_covariance`` in the Loewner order.
        effective_sample_size: ``1 / sum_i w_i^2``.
        transport_cost: Optimal cost ``sum t_ij |x_i - x_j|^2``.
        optimal: Verdict of the complementary-slackness certificate.
        converged: ``optimal`` and the weights did not collapse.
        stop_reason: ``"ok"``, ``"weight_collapse"`` (effective sample size
            below 1.5) or ``"coupling_not_certified"``.

    References:
        [EQ-INV-ENX-005], [EQ-INV-ENX-006], [EQ-INV-ENX-007].

    """

    ensemble: tuple[Vector, ...]
    weights: Vector
    coupling: Matrix
    weighted_mean: Vector
    mean: Vector
    weighted_covariance: Matrix
    covariance: Matrix
    effective_sample_size: float
    transport_cost: float
    optimal: bool
    converged: bool
    stop_reason: str


@dataclass(frozen=True, slots=True)
class SquareMapReference:
    """Quadrature posterior of the scalar model ``y = x^2 + noise``.

    Args:
        mass_positive: ``P(x > 0 | y)``, the occupation of the positive mode.
        mean: Posterior mean.
        variance: Posterior variance.
        valley_mass: ``P(|x| < sqrt(y) / 2 | y)`` for ``y > 0`` (zero for
            ``y <= 0``), the posterior mass between the two modes.
        nodes: Number of quadrature nodes.

    References:
        [EQ-INV-ENX-008].

    """

    mass_positive: float
    mean: float
    variance: float
    valley_mass: float
    nodes: int


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------


def gaspari_cohn_taper(distance: float, half_width: float) -> float:
    """Return the Gaspari-Cohn fifth-order compactly supported taper.

    With ``z = |d| / c``: ``-z^5/4 + z^4/2 + 5z^3/8 - 5z^2/3 + 1`` for
    ``z <= 1``, ``z^5/12 - z^4/2 + 5z^3/8 + 5z^2/3 - 5z + 4 - 2/(3z)`` for
    ``1 < z <= 2``, and ``0`` beyond ([EQ-INV-ENX-001]). It equals one at zero,
    ``5/24`` at ``d = c`` and vanishes for ``d >= 2c``.

    Args:
        distance: Distance ``d`` (absolute value used).
        half_width: Half-width ``c > 0``.

    Returns:
        The taper value in ``[0, 1]``.

    Raises:
        ValueError: If the distance is not finite or ``c`` is not positive
            and finite.

    Equation:
        [EQ-INV-ENX-001].

    References:
        [REF-GASPARI-COHN-1999].

    """
    d = abs(require_finite_real(distance, name="distance"))
    c = require_finite_real(half_width, name="half_width")
    if c <= 0.0:
        raise ValueError("half_width must be positive")
    z = d / c
    if z <= 1.0:
        return (((-0.25 * z + 0.5) * z + 0.625) * z - 5.0 / 3.0) * z * z + 1.0
    if z <= 2.0:
        value = (
            ((((z / 12.0 - 0.5) * z + 0.625) * z + 5.0 / 3.0) * z - 5.0) * z
            + 4.0
            - 2.0 / (3.0 * z)
        )
        return max(value, 0.0)
    return 0.0


def _distance_matrix(values: object, rows: int, cols: int, name: str) -> Matrix:
    """Return a validated non-negative ``rows x cols`` distance matrix.

    Args:
        values: Candidate nested sequence.
        rows: Required number of rows.
        cols: Required number of columns.
        name: Name used in error messages.

    Returns:
        The matrix as tuples of floats.

    Raises:
        TypeError: If ``values`` is not a nested sequence of real numbers.
        ValueError: If the shape is wrong or an entry is negative or not
            finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a nested sequence of real numbers")
    matrix = tuple(_vector(row, f"{name}[{i}]") for i, row in enumerate(values))
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must have shape {rows} x {cols}")
    if any(v < 0.0 for row in matrix for v in row):
        raise ValueError(f"{name} must be non-negative")
    return matrix


def localized_ensemble_kalman_update(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
    *,
    parameter_observation_distances: Sequence[Sequence[float]] | None = None,
    observation_distances: Sequence[Sequence[float]] | None = None,
    half_width: float | None = None,
    inflation: float = 1.0,
    seed: int | RandomStream | None = None,
) -> LocalizedUpdate:
    """Apply a stochastic Kalman update with Schur-product localization.

    With anomalies ``X`` (``n x J``) and ``Z`` (``m x J``) divided by
    ``sqrt(J - 1)``, the sample covariances are ``C_xg = X Z'`` and
    ``C_gg = Z Z'``. With the Gaspari-Cohn taper ``rho`` evaluated at the
    caller's distances (``D_xg`` between parameter ``i`` and observation
    ``k``, ``D_gg`` between observations) the tapered gain is

    ``K = (C_xg o rho(D_xg)) (C_gg o rho(D_gg) + alpha Gamma)^-1``

    and ``x_j^+ = x_j + K (y + sqrt(alpha) eta_j - g_j)`` with
    ``eta_j ~ N(0, Gamma)`` ([EQ-INV-ENX-002]). ``rho(D_gg)`` has to be
    positive semi-definite for the Schur product to be; this holds for
    Gaspari-Cohn on points of a line and is checked only through the Cholesky
    factorization of the tapered system (``converged`` and ``stop_reason``).

    The taper suppresses the sample correlation between distant quantities,
    which with a small ensemble is mostly sampling error, at the price of a
    bias for true long-range correlation. It is a regularization, not the
    exact Bayesian update ([EQ-INV-ENX-001]). With ``half_width=None`` no
    taper is applied and the gain is the plain sample gain, which converges
    to the exact Kalman gain for linear ``G`` and Gaussian prior as ``J``
    grows.

    Args:
        ensemble: ``J >= 2`` members of length ``n``.
        predictions: ``J`` predictions of length ``m``.
        data: Observation ``y`` of length ``m``.
        noise: Covariance factor of ``Gamma`` with dimension ``m``.
        parameter_observation_distances: ``n x m`` non-negative distances;
            required with ``half_width``.
        observation_distances: ``m x m`` symmetric non-negative distances;
            required with ``half_width``.
        half_width: Gaspari-Cohn half-width ``c > 0`` or ``None`` for no
            localization.
        inflation: ``alpha > 0`` on ``Gamma``.
        seed: Integer seed or stream for the perturbations (drawn member by
            member from ``resolve_stream(seed).generator()``).

    Returns:
        A :class:`LocalizedUpdate`.

    Raises:
        TypeError: If ``noise`` is not a covariance factor or entries are not
            real numbers.
        ValueError: If shapes disagree, values are not finite, ``inflation``
            or ``half_width`` is not positive, distances are missing,
            negative or ``observation_distances`` is not symmetric.

    Equation:
        [EQ-INV-ENX-001], [EQ-INV-ENX-002].

    References:
        [REF-GASPARI-COHN-1999], [REF-EVENSEN-2003],
        [REF-IGLESIAS-LAW-STUART-2013].

    """
    members, preds, observed = _validate_update_inputs(
        ensemble, predictions, data, noise
    )
    alpha = require_finite_real(inflation, name="inflation")
    if alpha <= 0.0:
        raise ValueError("inflation must be positive")
    count, n, m = len(members), len(members[0]), len(observed)
    taper_xg: Matrix | None = None
    taper_gg: Matrix | None = None
    if half_width is not None:
        if parameter_observation_distances is None or observation_distances is None:
            raise ValueError("distances are required when half_width is given")
        d_xg = _distance_matrix(
            parameter_observation_distances, n, m, "parameter_observation_distances"
        )
        d_gg = _distance_matrix(observation_distances, m, m, "observation_distances")
        scale_d = max(1.0, max(max(row) for row in d_gg))
        if any(
            abs(d_gg[i][k] - d_gg[k][i]) > 1e-12 * scale_d
            for i in range(m)
            for k in range(i)
        ):
            raise ValueError("observation_distances must be symmetric")
        taper_xg = tuple(
            tuple(gaspari_cohn_taper(v, half_width) for v in row) for row in d_xg
        )
        taper_gg = tuple(
            tuple(gaspari_cohn_taper(v, half_width) for v in row) for row in d_gg
        )
    mean_x = _mean(members)
    mean_g = _mean(preds)
    root = 1.0 / math.sqrt(count - 1)
    ax = [
        tuple((a - c) * root for a, c in zip(r, mean_x, strict=True)) for r in members
    ]
    az = [tuple((a - c) * root for a, c in zip(r, mean_g, strict=True)) for r in preds]
    c_xg = [
        [math.fsum(ax[j][i] * az[j][k] for j in range(count)) for k in range(m)]
        for i in range(n)
    ]
    c_gg = [
        [math.fsum(az[j][p] * az[j][q] for j in range(count)) for q in range(m)]
        for p in range(m)
    ]
    if taper_xg is not None and taper_gg is not None:
        c_xg = [[c_xg[i][k] * taper_xg[i][k] for k in range(m)] for i in range(n)]
        c_gg = [[c_gg[p][q] * taper_gg[p][q] for q in range(m)] for p in range(m)]
    unit = [tuple(1.0 if i == k else 0.0 for i in range(m)) for k in range(m)]
    gamma_columns = [noise.apply(e) for e in unit]
    system: Matrix = tuple(
        tuple(
            0.5 * (c_gg[p][q] + c_gg[q][p])
            + alpha * 0.5 * (gamma_columns[q][p] + gamma_columns[p][q])
            for q in range(m)
        )
        for p in range(m)
    )
    try:
        lower = cholesky(system)
    except ValueError:
        return LocalizedUpdate(
            ensemble=tuple(members),
            analysis_mean=mean_x,
            gain=tuple(tuple(0.0 for _ in range(m)) for _ in range(n)),
            localized=half_width is not None,
            inflation=alpha,
            converged=False,
            stop_reason="tapered_system_not_spd",
        )
    gain: Matrix = tuple(tuple(cholesky_solve(lower, tuple(row))) for row in c_xg)
    innovation = tuple(y - g for y, g in zip(observed, mean_g, strict=True))
    shift = tuple(
        math.fsum(k * v for k, v in zip(row, innovation, strict=True)) for row in gain
    )
    analysis_mean = tuple(a + b for a, b in zip(mean_x, shift, strict=True))
    generator = resolve_stream(seed).generator()
    sqrt_alpha = math.sqrt(alpha)
    updated: list[Vector] = []
    for member, pred in zip(members, preds, strict=True):
        draw = noise.unwhiten(tuple(generator.gauss(0.0, 1.0) for _ in range(m)))
        residual = tuple(
            y + sqrt_alpha * e - g for y, e, g in zip(observed, draw, pred, strict=True)
        )
        step = tuple(
            math.fsum(k * v for k, v in zip(row, residual, strict=True)) for row in gain
        )
        updated.append(tuple(a + b for a, b in zip(member, step, strict=True)))
    return LocalizedUpdate(
        ensemble=tuple(updated),
        analysis_mean=analysis_mean,
        gain=gain,
        localized=half_width is not None,
        inflation=alpha,
        converged=True,
        stop_reason="ok",
    )


# ---------------------------------------------------------------------------
# Adaptive inflation
# ---------------------------------------------------------------------------


def estimate_inflation(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
    *,
    lower: float = 1.0,
    upper: float = 10.0,
) -> InflationReport:
    """Estimate a multiplicative prior inflation from the innovation.

    If the prior prediction covariance is ``P / lambda`` with the true value
    ``P`` (a prior that is too narrow by the factor ``lambda``), the
    innovation ``d = y - mean_j g_j`` has the whitened second moment
    ``E |L^-1 d|^2 = tr(W P W') + m`` with ``W = L^-1``, while the whitened
    sample prediction variance is ``tr(Zw Zw') ~ tr(W P W') / lambda``. The
    moment estimator is

    ``lambda_hat = (|L^-1 d|^2 - m) / tr(Zw Zw')``

    ([EQ-INV-ENX-003]), clipped to ``[lower, upper]``. It needs a Gaussian
    innovation with covariance ``H P H' + Gamma`` (linear ``G``), is a ratio
    and so slightly biased for small ``J`` (the ensemble mean error adds a
    relative ``1/J``), and is noisy for small ``m``: its sampling standard
    deviation is of order ``sqrt(2 m) / tr(Zw Zw')``. Raising ``lower`` above
    ``1`` forbids deflation.

    Args:
        ensemble: ``J >= 2`` members (validated only; the estimate uses the
            predictions).
        predictions: ``J`` predictions of length ``m``.
        data: Observation of length ``m``.
        noise: Covariance factor of ``Gamma``.
        lower: Positive lower bound.
        upper: Upper bound ``>= lower``.

    Returns:
        An :class:`InflationReport`.

    Raises:
        TypeError: If ``noise`` is not a covariance factor or entries are not
            real numbers.
        ValueError: If shapes disagree, the bounds are invalid, or the
            predictions have zero spread (no estimate exists).

    Equation:
        [EQ-INV-ENX-003].

    References:
        [REF-WANG-BISHOP-2003], [REF-ANDERSON-2007].

    """
    members, preds, observed = _validate_update_inputs(
        ensemble, predictions, data, noise
    )
    low = require_finite_real(lower, name="lower")
    high = require_finite_real(upper, name="upper")
    if low <= 0.0 or high < low:
        raise ValueError("bounds must satisfy 0 < lower <= upper")
    count, m = len(members), len(observed)
    mean_g = _mean(preds)
    spread = math.fsum(
        v * v
        for row in preds
        for v in noise.whiten(tuple(a - c for a, c in zip(row, mean_g, strict=True)))
    ) / (count - 1)
    if spread <= 0.0:
        raise ValueError("the predictions have zero spread; inflation is undefined")
    norm2 = math.fsum(
        v * v
        for v in noise.whiten(
            tuple(y - g for y, g in zip(observed, mean_g, strict=True))
        )
    )
    raw = (norm2 - m) / spread
    factor = min(max(raw, low), high)
    return InflationReport(
        raw=raw,
        factor=factor,
        lower=low,
        upper=high,
        clipped=factor != raw,
        innovation_norm_squared=norm2,
        spread_trace=spread,
        observation_count=m,
    )


def adaptive_inflation(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
    *,
    lower: float = 1.0,
    upper: float = 10.0,
) -> InflatedEnsemble:
    """Inflate an ensemble by the innovation-based factor.

    Computes ``lambda`` with :func:`estimate_inflation` and scales the
    anomalies, ``x_j -> x_bar + sqrt(lambda)(x_j - x_bar)`` and likewise for
    the predictions ([EQ-INV-ENX-004]). The sample means are unchanged and the
    sample covariances are multiplied by ``lambda`` exactly. For a nonlinear
    ``G`` scaling the predictions this way is an approximation: pass
    predictions recomputed from the inflated members instead if that matters.

    Args:
        ensemble: ``J >= 2`` members.
        predictions: ``J`` predictions.
        data: Observation.
        noise: Covariance factor of ``Gamma``.
        lower: Positive lower bound of ``lambda``.
        upper: Upper bound of ``lambda``.

    Returns:
        An :class:`InflatedEnsemble` with its :class:`InflationReport`.

    Raises:
        TypeError: See :func:`estimate_inflation`.
        ValueError: See :func:`estimate_inflation`.

    Equation:
        [EQ-INV-ENX-003], [EQ-INV-ENX-004].

    References:
        [REF-ANDERSON-2007], [REF-WANG-BISHOP-2003].

    """
    report = estimate_inflation(
        ensemble, predictions, data, noise, lower=lower, upper=upper
    )
    members, preds, _ = _validate_update_inputs(ensemble, predictions, data, noise)
    root = math.sqrt(report.factor)
    mean_x = _mean(members)
    mean_g = _mean(preds)
    return InflatedEnsemble(
        ensemble=tuple(
            tuple(c + root * (a - c) for a, c in zip(row, mean_x, strict=True))
            for row in members
        ),
        predictions=tuple(
            tuple(c + root * (a - c) for a, c in zip(row, mean_g, strict=True))
            for row in preds
        ),
        report=report,
    )


# ---------------------------------------------------------------------------
# Ensemble transform particle filter
# ---------------------------------------------------------------------------


def _weighted_covariance(
    rows: Sequence[Vector], weights: Sequence[float], center: Vector
) -> Matrix:
    """Return ``sum_i w_i (x_i - c)(x_i - c)'``.

    References:
        [EQ-INV-ENX-007].

    """
    n = len(center)
    return tuple(
        tuple(
            math.fsum(
                w * (r[i] - center[i]) * (r[k] - center[k])
                for r, w in zip(rows, weights, strict=True)
            )
            for k in range(n)
        )
        for i in range(n)
    )


def ensemble_transform_particle_filter(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
    *,
    tolerance: float = 1e-12,
) -> ETPFResult:
    """Apply one ensemble transform particle filter update.

    The Gaussian likelihood gives the weights
    ``w_i proportional to exp(-|L^-1 (y - g_i)|^2 / 2)`` ([EQ-INV-ENX-005]).
    The coupling ``t`` minimizes ``sum_ij t_ij |x_i - x_j|^2`` over the
    matrices with row sums ``w_i`` and column sums ``1/J``, solved exactly by
    :func:`probstat.information.transport.exact_transport`. The deterministic
    transform is ``x_j^+ = J sum_i t_ij x_i`` ([EQ-INV-ENX-006]).

    Consequences ([EQ-INV-ENX-007]): the mean of the transformed ensemble is
    ``sum_i w_i x_i`` exactly, because the column sums are ``1/J`` and the row
    sums are ``w_i``. Each ``x_j^+`` is a conditional mean of the coupling, so
    by the law of total variance the transformed covariance is dominated by
    the weighted covariance: the update contracts the spread by an amount
    that does not vanish at finite ``J``, and the transformed covariance is
    not the posterior covariance. The update has no randomness; the coupling
    can be non-unique when the optimum is degenerate (the solver is
    deterministic, so the result is reproducible).

    Args:
        ensemble: ``J >= 2`` members of length ``n``.
        predictions: ``J`` predictions of length ``m``.
        data: Observation of length ``m``.
        noise: Covariance factor of ``Gamma``.
        tolerance: Relative reduced-cost tolerance of the transport solver.

    Returns:
        An :class:`ETPFResult`.

    Raises:
        TypeError: If ``noise`` is not a covariance factor or entries are not
            real numbers.
        ValueError: If shapes disagree or values are not finite.

    Equation:
        [EQ-INV-ENX-005], [EQ-INV-ENX-006], [EQ-INV-ENX-007].

    References:
        [REF-REICH-2013], [REF-PEYRE-CUTURI-2019].

    """
    members, preds, observed = _validate_update_inputs(
        ensemble, predictions, data, noise
    )
    count = len(members)
    log_weights = [
        -0.5
        * math.fsum(
            v * v
            for v in noise.whiten(
                tuple(y - g for y, g in zip(observed, row, strict=True))
            )
        )
        for row in preds
    ]
    top = max(log_weights)
    raw = [math.exp(lw - top) for lw in log_weights]
    total = math.fsum(raw)
    weights = tuple(v / total for v in raw)
    ess = 1.0 / math.fsum(w * w for w in weights)
    cost = squared_euclidean_cost_matrix(members, members)
    plan = exact_transport(weights, [1.0 / count] * count, cost, tolerance=tolerance)
    coupling: Matrix = tuple(tuple(row) for row in plan.plan)
    n = len(members[0])
    new_members = tuple(
        tuple(
            count * math.fsum(coupling[i][j] * members[i][k] for i in range(count))
            for k in range(n)
        )
        for j in range(count)
    )
    weighted_mean = tuple(
        math.fsum(w * r[k] for r, w in zip(members, weights, strict=True))
        for k in range(n)
    )
    new_mean = _mean(list(new_members))
    optimal = plan.certificate().optimal
    if not optimal:
        reason = "coupling_not_certified"
    elif ess < 1.5:
        reason = "weight_collapse"
    else:
        reason = "ok"
    return ETPFResult(
        ensemble=new_members,
        weights=weights,
        coupling=coupling,
        weighted_mean=weighted_mean,
        mean=new_mean,
        weighted_covariance=_weighted_covariance(members, weights, weighted_mean),
        covariance=_weighted_covariance(
            list(new_members), [1.0 / count] * count, new_mean
        ),
        effective_sample_size=ess,
        transport_cost=plan.cost,
        optimal=optimal,
        converged=reason == "ok",
        stop_reason=reason,
    )


# ---------------------------------------------------------------------------
# Quadrature reference
# ---------------------------------------------------------------------------


def square_map_reference(
    prior_mean: float,
    prior_sd: float,
    data: float,
    noise_sd: float,
    *,
    nodes: int = 40001,
    width: float = 10.0,
) -> SquareMapReference:
    """Return the quadrature posterior of ``y = x^2 + e``, ``e ~ N(0, s^2)``.

    The density ``p(x | y) proportional to N(x; mu, sigma^2) N(y; x^2, s^2)``
    is integrated by the trapezoid rule on ``nodes`` equispaced points of
    ``[mu - width sigma, mu + width sigma]`` ([EQ-INV-ENX-008]). For ``y > 0``
    the posterior has two modes near ``+-sqrt(y)``; the mass of ``x > 0`` is
    the occupation of the positive mode. The mass at the node ``x = 0`` is
    split between the sides.

    Args:
        prior_mean: Prior mean ``mu``.
        prior_sd: Prior standard deviation ``sigma > 0``.
        data: Observation ``y``.
        noise_sd: Noise standard deviation ``s > 0``.
        nodes: Number of grid points (``>= 3``).
        width: Half-range in prior standard deviations (``> 0``).

    Returns:
        A :class:`SquareMapReference`.

    Raises:
        ValueError: If a scale is not positive, ``nodes < 3``, values are not
            finite or the integral underflows to zero.

    Equation:
        [EQ-INV-ENX-008].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.4], [REF-KAIPIO-SOMERSALO-2007].

    """
    mu = require_finite_real(prior_mean, name="prior_mean")
    sigma = require_finite_real(prior_sd, name="prior_sd")
    y = require_finite_real(data, name="data")
    s = require_finite_real(noise_sd, name="noise_sd")
    half = require_finite_real(width, name="width")
    if sigma <= 0.0 or s <= 0.0 or half <= 0.0:
        raise ValueError("prior_sd, noise_sd and width must be positive")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 3:
        raise ValueError("nodes must be an integer of at least 3")
    step = 2.0 * half * sigma / (nodes - 1)
    grid = [mu - half * sigma + i * step for i in range(nodes)]
    logs = [-0.5 * ((x - mu) / sigma) ** 2 - 0.5 * ((y - x * x) / s) ** 2 for x in grid]
    top = max(logs)
    dens = [math.exp(v - top) for v in logs]
    trap = [0.5 if i in (0, nodes - 1) else 1.0 for i in range(nodes)]
    total = math.fsum(w * d for w, d in zip(trap, dens, strict=True)) * step
    if not total > 0.0:
        raise ValueError("the posterior integral is not positive")
    post = [w * d * step / total for w, d in zip(trap, dens, strict=True)]
    mean = math.fsum(p * x for p, x in zip(post, grid, strict=True))
    var = math.fsum(p * (x - mean) ** 2 for p, x in zip(post, grid, strict=True))
    positive = math.fsum(
        p * (1.0 if x > 0.0 else 0.5 if x == 0.0 else 0.0)
        for p, x in zip(post, grid, strict=True)
    )
    valley = (
        math.fsum(
            p for p, x in zip(post, grid, strict=True) if abs(x) < 0.5 * math.sqrt(y)
        )
        if y > 0.0
        else 0.0
    )
    return SquareMapReference(
        mass_positive=positive,
        mean=mean,
        variance=var,
        valley_mass=valley,
        nodes=nodes,
    )
