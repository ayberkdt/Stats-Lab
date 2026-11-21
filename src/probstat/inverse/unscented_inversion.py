"""Unscented Kalman inversion: a deterministic Gaussian posterior iteration.

The method carries a Gaussian ``N(m_n, C_n)`` in the standard-normal
coordinates ``u`` of the prior of a
:class:`~probstat.inverse.CalibrationProblem` and repeats a prediction step
that inflates the covariance and a Kalman analysis step whose moments come
from ``2 d + 1`` sigma points pushed through the forward model
([EQ-INV-UKI-001], [EQ-INV-UKI-002]). No random numbers and no derivatives
are used, and one iteration costs ``2 d + 1`` forward evaluations
whatever the number of observations.

The hyperparameters are fixed by a derivation, not left to the user
([EQ-INV-UKI-003]): the observation is augmented with the prior
(``y_a = (y, 0)``, ``G_a(u) = (G(u), u)``, ``Gamma_a = diag(Gamma, I)``), the
prediction is ``m^ = m_n``, ``C^ = 2 C_n`` (``alpha = 1``,
``Sigma_w = C_n``), and the analysis uses ``Sigma_nu = 2 Gamma_a``. For an
affine forward map the precision then obeys
``C_{n+1}^-1 = (C_n^-1 + P) / 2`` with ``P`` the posterior precision, so the
iteration converges to the exact posterior mean and covariance and the
error halves at every iteration.

Contrast with the ensemble methods of this package:
:func:`~probstat.inverse.ensemble_kalman_inversion` collapses to a point
(its spread is not a posterior), and
:func:`~probstat.inverse.ensemble_smoother_mda` is a random ensemble with
``O(J^-1/2)`` sampling error. Unscented Kalman inversion has no sampling
error, but its cost grows with the dimension, it stores a dense ``d x d``
covariance, and for a nonlinear model its fixed point is a Gaussian
approximation of the posterior that is exact only in the affine case.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, cholesky, cholesky_solve

from .ensemble_inversion import _observation_variances
from .nonlinear import CalibrationProblem


@dataclass(frozen=True, slots=True)
class UnscentedInversionResult:
    """Gaussian posterior approximation of an unscented Kalman inversion.

    Histories have one entry per Gaussian: the prior ``N(0, I)`` and the
    Gaussian after each iteration (``iterations + 1`` entries), except
    ``misfit_history``, which has one entry per iteration (the misfit at the
    mean before that iteration's update, from the central sigma point).

    Args:
        parameter_names: Calibration-parameter names.
        standard_mean: Final mean ``m`` in the coordinates ``u``.
        standard_covariance: Final covariance ``C`` in the coordinates ``u``.
        posterior_mean: Mean of ``theta = T^-1(u)`` under ``N(m, C)`` by the
            unscented transform.
        posterior_covariance: Covariance of ``theta`` under ``N(m, C)`` by
            the unscented transform. For Gaussian priors ``T^-1`` is affine
            and both are exact images of ``(m, C)``.
        standard_mean_history: ``m_n`` per Gaussian.
        standard_spread_history: Square roots of the diagonal of ``C_n``.
        mean_history: Unscented mean of ``theta`` per Gaussian.
        spread_history: Unscented standard deviations of ``theta``.
        misfit_history: Whitened data misfit ``|Gamma^-1/2 (y - G(m_n))|``
            before each update.
        change_history: Convergence measure of each iteration, the largest
            of ``|m_{n+1} - m_n|_i / sqrt(C_{n+1,ii})`` and
            ``|C_{n+1,ij} - C_{n,ij}| / sqrt(C_{n+1,ii} C_{n+1,jj})``.
        iterations: Number of iterations performed.
        converged: Whether the last change fell below ``tolerance``. When
            ``False`` the Gaussian is an unconverged iterate. ``True`` says
            only that the iteration reached a fixed point, not that the
            fixed point is close to the posterior: for a forward map that is
            even about the prior mean (``G(u) = G(-u)``) the prior itself is
            a fixed point and is returned after one iteration, whatever the
            data.
        tolerance: Convergence tolerance on the change.
        kappa: Sigma-point spread parameter.
        forward_evaluations: Evaluations of the full observation operator,
            ``(2 d + 1) iterations``.
        model_calls: Forward-model calls, ``forward_evaluations`` times the
            number of experiments.

    References:
        [REF-HUANG-SCHNEIDER-STUART-2022], [EQ-INV-UKI-001],
        [EQ-INV-UKI-003].

    """

    parameter_names: tuple[str, ...]
    standard_mean: Vector
    standard_covariance: Matrix
    posterior_mean: Vector
    posterior_covariance: Matrix
    standard_mean_history: tuple[Vector, ...]
    standard_spread_history: tuple[Vector, ...]
    mean_history: tuple[Vector, ...]
    spread_history: tuple[Vector, ...]
    misfit_history: Vector
    change_history: Vector
    iterations: int
    converged: bool
    tolerance: float
    kappa: float
    forward_evaluations: int
    model_calls: int

    def mean(self) -> Mapping[str, float]:
        """Return the posterior mean of ``theta`` keyed by name.

        Returns:
            Read-only mapping from parameter name to the unscented mean.

        Equation:
            [EQ-INV-UKI-002].

        References:
            [REF-HUANG-SCHNEIDER-STUART-2022].

        """
        return MappingProxyType(
            dict(zip(self.parameter_names, self.posterior_mean, strict=True))
        )

    def covariance(self) -> Matrix:
        """Return the posterior covariance of ``theta``.

        Returns:
            Symmetric covariance matrix in parameter order.

        Equation:
            [EQ-INV-UKI-002].

        References:
            [REF-HUANG-SCHNEIDER-STUART-2022].

        """
        return self.posterior_covariance

    def summary(self) -> str:
        """Return a one-paragraph description of the run.

        Returns:
            Text with iterations, convergence, posterior means and standard
            deviations, and the forward cost.

        References:
            [REF-HUANG-SCHNEIDER-STUART-2022]. No equation is evaluated.

        """
        estimates = ", ".join(
            f"{name} = {center:.6g} (sd {spread:.3g})"
            for name, center, spread in zip(
                self.parameter_names,
                self.posterior_mean,
                self.spread_history[-1],
                strict=True,
            )
        )
        state = (
            "converged"
            if self.converged
            else f"NOT converged (last change {self.change_history[-1]:.3g})"
        )
        return (
            f"Unscented Kalman inversion: {self.iterations} iterations, {state}; "
            f"{estimates}; {self.forward_evaluations} forward evaluations "
            f"({self.model_calls} model calls). The Gaussian is exact for an "
            "affine forward map and an approximation otherwise."
        )


def unscented_kalman_inversion(
    problem: CalibrationProblem,
    *,
    iterations: int = 50,
    tolerance: float = 1e-8,
    kappa: float = 1.0,
) -> UnscentedInversionResult:
    """Iterate a Gaussian approximation of the posterior with sigma points.

    Starting from the prior ``m_0 = 0``, ``C_0 = I`` in the coordinates
    ``u``, each iteration ([EQ-INV-UKI-001])

    1. predicts ``m^ = m_n``, ``C^ = 2 C_n``;
    2. forms the sigma points ``m^`` and ``m^ +/- sqrt(d + kappa) L e_i``
       with ``L L' = C^`` and weights ``kappa / (d + kappa)`` and
       ``1 / (2 (d + kappa))`` ([EQ-INV-UKI-002]), and evaluates the
       augmented map ``G_a(u) = (G(u), u)`` at each;
    3. computes the weighted mean ``y^`` and covariances ``C^uy``, ``C^yy``
       and updates ``m_{n+1} = m^ + C^uy (C^yy + 2 Gamma_a)^-1 (y_a - y^)``,
       ``C_{n+1} = C^ - C^uy (C^yy + 2 Gamma_a)^-1 C^uy'`` with
       ``y_a = (y, 0)`` and ``Gamma_a = diag(Gamma, I)``.

    For ``G(u) = A u + b`` the unscented moments are exact and the update is
    ``C_{n+1}^-1 = (C_n^-1 + P) / 2``,
    ``C_{n+1}^-1 m_{n+1} = (C_n^-1 m_n + A' Gamma^-1 (y - b)) / 2`` with
    ``P = A' Gamma^-1 A + I`` ([EQ-INV-UKI-003]). The unique fixed point is
    the posterior mean and covariance, and the error of the precision and of
    the precision-weighted mean halves at every iteration. For a nonlinear
    model the fixed point is a Gaussian approximation built from sigma-point
    (statistical) linearization; it need not be close to a skewed or
    multimodal posterior and the iteration is not guaranteed to converge.
    Example (``G(theta) = theta^2``, noise 0.2, ``y = 1``, posterior modes
    near ``-1`` and ``+1``): with the prior ``N(0, 1)`` the sigma points are
    symmetric, the cross-covariance with the data vanishes, and the run
    stops after one iteration with ``converged=True`` at the prior
    ``N(0, 1)`` (also for ``y = 4``, where the posterior standard deviation
    is 2.0); with the prior ``N(0.3, 1)`` (posterior mean 0.28, 64 % of the
    mass on the positive mode) it returns ``N(0.15, 0.98^2)`` for
    ``kappa = 1`` and ``N(-0.98, 0.10^2)``, the minor mode, for
    ``kappa = 0``.

    Args:
        problem: Calibration problem whose noise models are all known-scale
            Gaussian.
        iterations: Maximum number of iterations, at least one.
        tolerance: The iteration stops when the scaled change of mean and
            covariance (see :class:`UnscentedInversionResult`) is at most
            this non-negative value; zero runs all iterations.
        kappa: Non-negative sigma-point parameter; the points lie at
            ``sqrt(d + kappa)`` standard deviations. The default 1 is a
            library choice; it does not affect the affine case.

    Returns:
        Final Gaussian with its iteration history and a convergence flag.

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem`.
        ValueError: For unknown-noise or Student-t problems,
            ``iterations < 1``, a negative ``tolerance`` or ``kappa``, or a
            covariance that lost positive definiteness.

    Equation:
        [EQ-INV-UKI-001], [EQ-INV-UKI-002], [EQ-INV-UKI-003].

    References:
        [REF-HUANG-SCHNEIDER-STUART-2022], [REF-JULIER-UHLMANN-2004].

    """
    variances = _observation_variances(problem)
    limit = require_non_negative_integer(iterations, name="iterations")
    if limit == 0:
        raise ValueError("iterations must be at least one")
    threshold = require_finite_real(tolerance, name="tolerance")
    if threshold < 0.0:
        raise ValueError("tolerance must be non-negative")
    spread = require_finite_real(kappa, name="kappa")
    if spread < 0.0:
        raise ValueError(
            "kappa must be non-negative so that every sigma-point weight is "
            "non-negative"
        )
    dimension = problem.model.dimension
    inputs = problem.model.inputs
    scales = tuple(math.sqrt(value) for value in variances)
    # Whitened augmented data (Gamma^-1/2 y, 0).
    data = (
        tuple(
            value / scale for value, scale in zip(problem.observed, scales, strict=True)
        )
        + (0.0,) * dimension
    )
    size = len(data)
    observations = len(scales)

    mean: Vector = (0.0,) * dimension
    covariance: Matrix = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )
    standard_mean_history: list[Vector] = [mean]
    standard_spread_history: list[Vector] = [_diagonal_roots(covariance)]
    physical = _physical_moments(problem, mean, covariance, spread)
    mean_history: list[Vector] = [physical[0]]
    spread_history: list[Vector] = [_diagonal_roots(physical[1])]
    misfit_history: list[float] = []
    change_history: list[float] = []
    converged = False
    performed = 0

    while performed < limit:
        predicted = tuple(tuple(2.0 * value for value in row) for row in covariance)
        points, weights = _sigma_points(mean, predicted, spread)
        images = [
            tuple(
                value / scale
                for value, scale in zip(
                    problem.predictions(inputs.from_standard_normal(point)),
                    scales,
                    strict=True,
                )
            )
            + point
            for point in points
        ]
        misfit_history.append(
            math.sqrt(
                math.fsum(
                    (y - g) ** 2
                    for y, g in zip(
                        data[:observations], images[0][:observations], strict=True
                    )
                )
            )
        )
        center = tuple(
            math.fsum(w * image[p] for w, image in zip(weights, images, strict=True))
            for p in range(size)
        )
        deviations_y = [
            tuple(value - c for value, c in zip(image, center, strict=True))
            for image in images
        ]
        deviations_u = [
            tuple(value - c for value, c in zip(point, mean, strict=True))
            for point in points
        ]
        # Whitened innovation covariance C^yy + 2 I (Sigma_nu = 2 Gamma_a).
        system = tuple(
            tuple(
                math.fsum(
                    w * row[p] * row[q]
                    for w, row in zip(weights, deviations_y, strict=True)
                )
                + (2.0 if p == q else 0.0)
                for q in range(size)
            )
            for p in range(size)
        )
        cross = tuple(
            tuple(
                math.fsum(
                    w * du[i] * dy[p]
                    for w, du, dy in zip(
                        weights, deviations_u, deviations_y, strict=True
                    )
                )
                for p in range(size)
            )
            for i in range(dimension)
        )
        lower = cholesky(system)
        gain = tuple(cholesky_solve(lower, row) for row in cross)
        innovation = tuple(y - c for y, c in zip(data, center, strict=True))
        updated_mean = tuple(
            mean[i] + math.fsum(k * r for k, r in zip(gain[i], innovation, strict=True))
            for i in range(dimension)
        )
        raw = [
            [
                predicted[i][j]
                - math.fsum(k * c for k, c in zip(gain[i], cross[j], strict=True))
                for j in range(dimension)
            ]
            for i in range(dimension)
        ]
        updated_covariance = tuple(
            tuple(0.5 * (raw[i][j] + raw[j][i]) for j in range(dimension))
            for i in range(dimension)
        )
        if any(updated_covariance[i][i] <= 0.0 for i in range(dimension)) or not all(
            math.isfinite(value) for value in updated_mean
        ):
            raise ValueError(
                "the unscented Kalman inversion lost a positive-definite "
                f"covariance at iteration {performed + 1}"
            )
        roots = _diagonal_roots(updated_covariance)
        change = max(
            max(
                abs(new - old) / root
                for new, old, root in zip(updated_mean, mean, roots, strict=True)
            ),
            max(
                abs(updated_covariance[i][j] - covariance[i][j]) / (roots[i] * roots[j])
                for i in range(dimension)
                for j in range(dimension)
            ),
        )
        mean, covariance = updated_mean, updated_covariance
        performed += 1
        change_history.append(change)
        standard_mean_history.append(mean)
        standard_spread_history.append(roots)
        physical = _physical_moments(problem, mean, covariance, spread)
        mean_history.append(physical[0])
        spread_history.append(_diagonal_roots(physical[1]))
        if change <= threshold:
            converged = True
            break

    evaluations = (2 * dimension + 1) * performed
    return UnscentedInversionResult(
        parameter_names=problem.parameter_names,
        standard_mean=mean,
        standard_covariance=covariance,
        posterior_mean=physical[0],
        posterior_covariance=physical[1],
        standard_mean_history=tuple(standard_mean_history),
        standard_spread_history=tuple(standard_spread_history),
        mean_history=tuple(mean_history),
        spread_history=tuple(spread_history),
        misfit_history=tuple(misfit_history),
        change_history=tuple(change_history),
        iterations=performed,
        converged=converged,
        tolerance=threshold,
        kappa=spread,
        forward_evaluations=evaluations,
        model_calls=evaluations * len(problem.experiments),
    )


def _diagonal_roots(matrix: Matrix) -> Vector:
    """Return the square roots of the diagonal of a covariance matrix.

    References:
        [EQ-INV-UKI-001] (marginal standard deviations of the Gaussian).

    """
    return tuple(math.sqrt(max(matrix[k][k], 0.0)) for k in range(len(matrix)))


def _sigma_points(
    mean: Vector, covariance: Matrix, kappa: float
) -> tuple[list[Vector], Vector]:
    """Return the ``2 d + 1`` sigma points and weights of ``N(mean, covariance)``.

    Points: ``mean`` and ``mean +/- sqrt(d + kappa) L e_i`` with ``L`` the
    lower Cholesky factor. Weights: ``kappa / (d + kappa)`` for the centre
    and ``1 / (2 (d + kappa))`` for the others. The weighted mean and
    covariance of the points equal ``mean`` and ``covariance`` exactly, so
    the transform is exact for affine maps.

    Raises:
        ValueError: If ``covariance`` is not positive definite.

    References:
        [EQ-INV-UKI-002], [REF-JULIER-UHLMANN-2004].

    """
    dimension = len(mean)
    lower = cholesky(covariance)
    radius = math.sqrt(dimension + kappa)
    points: list[Vector] = [mean]
    for sign in (1.0, -1.0):
        for column in range(dimension):
            points.append(
                tuple(
                    mean[i] + sign * radius * lower[i][column] for i in range(dimension)
                )
            )
    outer = 0.5 / (dimension + kappa)
    weights = (kappa / (dimension + kappa),) + (outer,) * (2 * dimension)
    return points, weights


def _physical_moments(
    problem: CalibrationProblem, mean: Vector, covariance: Matrix, kappa: float
) -> tuple[Vector, Matrix]:
    """Return the unscented mean and covariance of ``theta = T^-1(u)``.

    Pushes the sigma points of ``N(mean, covariance)`` through the prior's
    inverse Nataf map; no forward-model evaluation is involved.

    References:
        [EQ-INV-UKI-002], [REF-JULIER-UHLMANN-2004].

    """
    points, weights = _sigma_points(mean, covariance, kappa)
    images = [problem.model.inputs.from_standard_normal(point) for point in points]
    dimension = len(mean)
    center = tuple(
        math.fsum(w * image[k] for w, image in zip(weights, images, strict=True))
        for k in range(dimension)
    )
    matrix = tuple(
        tuple(
            math.fsum(
                w * (image[i] - center[i]) * (image[j] - center[j])
                for w, image in zip(weights, images, strict=True)
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    return center, matrix
