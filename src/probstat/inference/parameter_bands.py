"""Pointwise CDF and quantile bands from the parameter uncertainty of a fit.

:func:`parameter_uncertainty_band` propagates the estimation uncertainty of a
fitted family to its CDF and quantiles. It draws parameter vectors from the
asymptotic normal approximation ``N(theta_hat, Cov(theta_hat))`` of a
:class:`~probstat.inference.FitResult` (inverse observed information) or a
:class:`~probstat.inference.MomentFitResult` (delta-method covariance),
rejects and counts draws outside the parameter space, and evaluates
``F(x; theta*)`` on a grid. It reports pointwise quantile bands of the
draws, the min/max envelope, and the delta-method band
``F(x; theta_hat) +- z sqrt(g' Cov g)`` with ``g = dF/dtheta``.

The pointwise quantile band converges, as the number of draws grows, to the
band of the normal approximation. The min/max envelope does not converge:
it spreads with the number of draws (for a scalar normal parameter like
``sqrt(2 log R)`` standard errors) and is not a confidence band.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from probstat.distributions import Normal
from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import Matrix, Vector, cholesky, cholesky_semidefinite
from probstat.optimization import central_gradient
from probstat.statistics import quantile as sample_quantile

from .fitting import DistributionFamily, FitResult, family_by_name
from .moment_fitting import MomentFitResult

_STANDARD_NORMAL = Normal(0.0, 1.0)


@dataclass(frozen=True)
class ParameterUncertaintyBand:
    """Pointwise CDF and quantile bands induced by parameter uncertainty.

    Args:
        family: Family label of the fit.
        grid: Points ``x`` at which the CDF is evaluated.
        level: Pointwise coverage level of the bands.
        cdf_estimate: ``F(x; theta_hat)`` on the grid.
        lower: Pointwise ``(1 - level)/2`` sample quantile (type 7) of
            ``F(x; theta*)`` over the accepted draws.
        upper: Pointwise ``(1 + level)/2`` sample quantile.
        envelope_lower: Pointwise minimum over the accepted draws.
        envelope_upper: Pointwise maximum over the accepted draws.
        delta_standard_errors: Delta-method standard error of
            ``F(x; theta_hat)`` on the grid.
        delta_lower: ``max(0, F - z se)``.
        delta_upper: ``min(1, F + z se)``.
        quantile_probabilities: Probabilities ``p`` of the quantile bands.
        quantile_estimate: ``F^-1(p; theta_hat)``.
        quantile_lower: Pointwise lower sample quantile of ``F^-1(p; theta*)``.
        quantile_upper: Pointwise upper sample quantile of ``F^-1(p; theta*)``.
        quantile_delta_lower: ``F^-1(p; theta_hat) - z se``.
        quantile_delta_upper: ``F^-1(p; theta_hat) + z se``.
        draws_requested: Number of normal parameter draws generated.
        accepted_draws: Draws inside the parameter space, used for the bands.
        rejected_draws: Draws outside the parameter space (discarded).
        parameter_draws: The accepted parameter vectors, in draw order.
        method: Description of the construction.

    References:
        [EQ-INF-BAND-001], [EQ-INF-BAND-002],
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """

    family: str
    grid: Vector
    level: float
    cdf_estimate: Vector
    lower: Vector
    upper: Vector
    envelope_lower: Vector
    envelope_upper: Vector
    delta_standard_errors: Vector
    delta_lower: Vector
    delta_upper: Vector
    quantile_probabilities: Vector
    quantile_estimate: Vector
    quantile_lower: Vector
    quantile_upper: Vector
    quantile_delta_lower: Vector
    quantile_delta_upper: Vector
    draws_requested: int
    accepted_draws: int
    rejected_draws: int
    parameter_draws: tuple[Vector, ...] = field(repr=False)
    method: str = (
        "draws from the asymptotic normal law of the estimates; pointwise "
        "quantile band, min/max envelope, and delta-method band"
    )

    @property
    def rejected_fraction(self) -> float:
        """Return the fraction of draws that fell outside the parameter space.

        References:
            [EQ-INF-BAND-001]. No equation is evaluated.

        """
        return self.rejected_draws / self.draws_requested


def _resolve_family(
    fit: FitResult | MomentFitResult, family: DistributionFamily | None
) -> DistributionFamily:
    """Return the family that rebuilds laws from parameter vectors.

    Raises:
        TypeError: If ``family`` is not a :class:`DistributionFamily`.
        ValueError: If the family does not match the fit.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    if family is None:
        try:
            return family_by_name(fit.family)
        except KeyError as error:
            raise ValueError(
                f"{fit.family!r} is not a ready-made family; pass family="
            ) from error
    if not isinstance(family, DistributionFamily):
        raise TypeError("family must be a DistributionFamily")
    if family.name != fit.family or family.parameter_names != fit.parameter_names:
        raise ValueError("family does not match the fit")
    return family


def _validated_covariance(fit: FitResult | MomentFitResult) -> Matrix:
    """Return the fit covariance when it is finite, square, and symmetric.

    Raises:
        ValueError: If the covariance is missing or has non-finite entries.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    matrix = fit.covariance
    k = len(fit.estimates)
    if matrix is None:
        raise ValueError(
            f"the {fit.family} fit has no covariance ({fit.message}); "
            "parameter-uncertainty bands need one"
        )
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(rows) != k or any(len(row) != k for row in rows):
        raise ValueError("covariance must be square with one row per parameter")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(
            f"the {fit.family} covariance has undefined entries ({fit.message}); "
            "parameter-uncertainty bands need a finite covariance"
        )
    return rows


def _square_root(matrix: Matrix) -> Matrix:
    """Return a lower factor ``L`` with ``L L' = matrix``.

    References:
        [EQ-LINALG-001], [EQ-DIST-MVN-005].

    """
    try:
        return cholesky(matrix)
    except ValueError:
        return cholesky_semidefinite(matrix)


def _admissible(family: DistributionFamily, theta: Vector) -> bool:
    """Return whether ``theta`` lies in the parameter space of ``family``.

    Positive parameters must be positive and below their bound, and the
    family constructor must accept the vector.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    for index, (kind, value) in enumerate(
        zip(family.parameter_kinds, theta, strict=True)
    ):
        if not math.isfinite(value):
            return False
        if kind == "positive":
            bound = family.bound(index)
            if not value > 0.0 or (bound is not None and value > bound):
                return False
    try:
        family.build(theta)
    except (ValueError, ArithmeticError):
        return False
    return True


def _evaluator(
    family: DistributionFamily, point: float, *, quantile: bool
) -> Callable[[Vector], float]:
    """Return ``theta -> F(point; theta)`` or ``theta -> F^-1(point; theta)``.

    References:
        [EQ-INF-BAND-002]. No new equation is evaluated.

    """

    def evaluate(theta: Vector) -> float:
        """Return the CDF or quantile of the law built from ``theta``.

        References:
            [EQ-INF-BAND-002]. No new equation is evaluated.

        """
        law = family.build(theta)
        return law.quantile(point) if quantile else law.cdf(point)

    return evaluate


def _delta_error(
    function: Callable[[Vector], float],
    estimates: Vector,
    covariance: Matrix,
    scales: Vector,
) -> float:
    """Return ``sqrt(g' Cov g)`` with ``g`` the central-difference gradient.

    References:
        [EQ-INF-BAND-002], [EQ-OPT-FD-001].

    """
    gradient, _ = central_gradient(function, estimates, typical_scale=scales)
    variance = math.fsum(
        gi * covariance[i][j] * gj
        for i, gi in enumerate(gradient)
        for j, gj in enumerate(gradient)
    )
    return math.sqrt(max(variance, 0.0))


def parameter_uncertainty_band(
    fit: FitResult | MomentFitResult,
    grid: Iterable[float],
    *,
    family: DistributionFamily | None = None,
    draws: int = 1000,
    level: float = 0.95,
    quantile_probabilities: Iterable[float] = (),
    seed: int | RandomStream | None = None,
) -> ParameterUncertaintyBand:
    """Propagate parameter uncertainty of a fit to pointwise CDF bands.

    Parameter vectors ``theta*_r = theta_hat + L z_r`` (``L L' = Cov``,
    ``z_r`` standard normal, ``r = 1..R``) are drawn from the asymptotic
    normal approximation of the estimator; draws outside the parameter space
    (a non-positive scale, a bound exceeded, a vector the constructor
    rejects) are discarded and counted, which truncates the normal
    approximation. For every grid point the band reports

    - the pointwise ``(1 - level)/2`` and ``(1 + level)/2`` sample quantiles
      of ``F(x; theta*_r)``, which converge as ``R`` grows;
    - the pointwise min/max envelope, which widens without limit as ``R``
      grows and is not a confidence band;
    - the delta-method band ``F(x; theta_hat) +- z_(1-alpha/2) se(x)``,
      ``se(x)^2 = g' Cov g``, ``g = dF(x; theta)/dtheta`` (clipped to
      ``[0, 1]``).

    All three are pointwise: each holds at one ``x`` at a time, not jointly
    over the grid. Quantiles ``F^-1(p; theta)`` get the same sample-quantile
    and delta-method bands when ``quantile_probabilities`` is given.

    Args:
        fit: Maximum-likelihood or moment fit with a finite covariance.
        grid: Finite points at which the CDF is evaluated.
        family: Family that rebuilds laws from parameter vectors; looked up
            by name for the ready-made families when omitted.
        draws: Number ``R >= 2`` of normal parameter draws.
        level: Pointwise level in ``(0, 1)``.
        quantile_probabilities: Optional probabilities in ``(0, 1)`` for
            quantile bands.
        seed: Seed or :class:`RandomStream`; the draws are generated in
            order from one stream, so a run with more draws extends a run
            with fewer.

    Returns:
        Bands on the grid, quantile bands, draw counts, and the accepted
        parameter draws.

    Raises:
        TypeError: If ``family`` or ``draws`` has the wrong type.
        ValueError: If the fit has no finite covariance, the family does not
            match the fit, the grid is empty, ``draws < 2``, ``level`` or a
            probability is outside ``(0, 1)``, or fewer than two draws are
            accepted.

    Equation:
        [EQ-INF-BAND-001], [EQ-INF-BAND-002], [EQ-STAT-QUANT-001].

    References:
        [REF-CASELLA-BERGER-2002, Sections 5.5.4 and 10.1],
        [REF-VAN-DER-VAART-1998, Chapter 3], [REF-HYNDMAN-FAN-1996].

    """
    if not isinstance(fit, FitResult | MomentFitResult):
        raise TypeError("fit must be a FitResult or a MomentFitResult")
    chosen = _resolve_family(fit, family)
    covariance = _validated_covariance(fit)
    points = coerce_finite_sample(grid)
    count = require_non_negative_integer(draws, name="draws")
    if count < 2:
        raise ValueError("draws must be at least 2")
    coverage = require_probability(level, name="level")
    if not 0.0 < coverage < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    probabilities = tuple(
        require_probability(p, name="quantile probability")
        for p in quantile_probabilities
    )
    if any(not 0.0 < p < 1.0 for p in probabilities):
        raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
    estimates = tuple(float(value) for value in fit.estimates)
    k = len(estimates)
    factor = _square_root(covariance)
    normals = _STANDARD_NORMAL.sample(count * k, seed=resolve_stream(seed))
    accepted: list[Vector] = []
    for r in range(count):
        z = normals[r * k : (r + 1) * k]
        theta = tuple(
            estimates[i] + math.fsum(factor[i][j] * z[j] for j in range(i + 1))
            for i in range(k)
        )
        if _admissible(chosen, theta):
            accepted.append(theta)
    if len(accepted) < 2:
        raise ValueError("fewer than two draws fell inside the parameter space")
    laws = [chosen.build(theta) for theta in accepted]
    alpha_low, alpha_high = (1.0 - coverage) / 2.0, (1.0 + coverage) / 2.0
    critical = _STANDARD_NORMAL.quantile(alpha_high)
    fitted = chosen.build(estimates)
    scales = tuple(
        max(abs(value), error)
        if math.isfinite(error) and error > 0.0
        else abs(value) or 1.0
        for value, error in zip(
            estimates,
            (math.sqrt(max(covariance[i][i], 0.0)) for i in range(k)),
            strict=True,
        )
    )

    lower: list[float] = []
    upper: list[float] = []
    envelope_lower: list[float] = []
    envelope_upper: list[float] = []
    delta_errors: list[float] = []
    for x in points:
        values = [law.cdf(x) for law in laws]
        lower.append(sample_quantile(values, alpha_low))
        upper.append(sample_quantile(values, alpha_high))
        envelope_lower.append(min(values))
        envelope_upper.append(max(values))
        delta_errors.append(
            _delta_error(
                _evaluator(chosen, x, quantile=False),
                estimates,
                covariance,
                scales,
            )
        )
    cdf_estimate = tuple(fitted.cdf(x) for x in points)

    quantile_estimate = tuple(fitted.quantile(p) for p in probabilities)
    quantile_lower: list[float] = []
    quantile_upper: list[float] = []
    quantile_errors: list[float] = []
    for p in probabilities:
        values = [law.quantile(p) for law in laws]
        quantile_lower.append(sample_quantile(values, alpha_low))
        quantile_upper.append(sample_quantile(values, alpha_high))
        quantile_errors.append(
            _delta_error(
                _evaluator(chosen, p, quantile=True),
                estimates,
                covariance,
                scales,
            )
        )

    return ParameterUncertaintyBand(
        family=fit.family,
        grid=points,
        level=coverage,
        cdf_estimate=cdf_estimate,
        lower=tuple(lower),
        upper=tuple(upper),
        envelope_lower=tuple(envelope_lower),
        envelope_upper=tuple(envelope_upper),
        delta_standard_errors=tuple(delta_errors),
        delta_lower=tuple(
            max(0.0, f - critical * e)
            for f, e in zip(cdf_estimate, delta_errors, strict=True)
        ),
        delta_upper=tuple(
            min(1.0, f + critical * e)
            for f, e in zip(cdf_estimate, delta_errors, strict=True)
        ),
        quantile_probabilities=probabilities,
        quantile_estimate=quantile_estimate,
        quantile_lower=tuple(quantile_lower),
        quantile_upper=tuple(quantile_upper),
        quantile_delta_lower=tuple(
            q - critical * e
            for q, e in zip(quantile_estimate, quantile_errors, strict=True)
        ),
        quantile_delta_upper=tuple(
            q + critical * e
            for q, e in zip(quantile_estimate, quantile_errors, strict=True)
        ),
        draws_requested=count,
        accepted_draws=len(accepted),
        rejected_draws=count - len(accepted),
        parameter_draws=tuple(accepted),
    )
