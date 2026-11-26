"""Ensemble Kalman updates on a generic forward callback and Tikhonov EKI.

Experimental (added in 0.28).

:mod:`probstat.inverse.ensemble_inversion` runs EKI and ES-MDA on a
:class:`~probstat.inverse.CalibrationProblem` with a diagonal noise covariance.
This module works on a generic vector forward map ``G: R^n -> R^m`` and accepts
any covariance factor of :mod:`probstat.linalg` (diagonal, dense or block
diagonal) for the observation noise ``Gamma``. It provides

* :func:`ensemble_kalman_update`, one Kalman-type update of an ensemble, either
  stochastic (perturbed observations) or deterministic (ensemble transform
  square root), solved in observation space (``m x m``) or in ensemble space
  (``J x J``), which is cheaper when ``m >> J``;
* :func:`tikhonov_ensemble_kalman_inversion` (TEKI), the iteration obtained by
  applying the update to the augmented system with forward map ``[G(x); x]``,
  data ``[y; mu]`` and noise ``blockdiag(Gamma, C0)``. The TEKI formulation
  follows the augmented-system construction of the regularized ensemble Kalman
  methods built on Iglesias, Law and Stuart (2013); the references below are
  the sources verified for this library.

Notation. ``J`` is the ensemble size, ``x_j`` the members (length ``n``),
``g_j = G(x_j)`` the predictions (length ``m``), ``Gamma = L L'`` the noise
covariance with the factor ``L`` of the covariance object (``whiten`` is
``L^-1``), and all sample covariances use the divisor ``J - 1``.

Ensemble Kalman inversion, with or without the Tikhonov augmentation, is a
derivative-free *optimizer*, not a posterior sampler: the ensemble stays in the
affine span of the initial ensemble and its spread collapses as the iteration
proceeds. The final spread of :func:`tikhonov_ensemble_kalman_inversion` is
not a posterior uncertainty.

References:
    [REF-IGLESIAS-LAW-STUART-2013], [REF-SCHILLINGS-STUART-2017],
    [REF-EMERICK-REYNOLDS-2013], [REF-BISHOP-ETHERTON-MAJUMDAR-2001],
    [EQ-INV-TEKI-001].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    BlockDiagonalCovariance,
    DenseCovariance,
    DiagonalCovariance,
    Matrix,
    Vector,
    cholesky,
    cholesky_solve,
    is_covariance_factor,
    symmetric_eigen,
)

_CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance

EnsembleSolve = Literal["observation", "ensemble"]
"""Space in which the Kalman system is solved."""

_SOLVES = ("observation", "ensemble")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleKalmanUpdate:
    """Updated ensemble and diagnostics of one Kalman-type update.

    Args:
        ensemble: Updated members ``x_j^+``, each of length ``n``.
        mean: Updated ensemble mean.
        innovation_misfit: Whitened norm ``|L^-1 (y - mean_j g_j)|`` of the
            innovation of the prior ensemble-mean prediction.
        inflation: The inflation ``alpha`` used (noise ``alpha Gamma``).
        perturbed: ``True`` for the stochastic update, ``False`` for the
            deterministic square-root update.
        solve: ``"observation"`` or ``"ensemble"``.

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [EQ-INV-TEKI-001].

    """

    ensemble: tuple[Vector, ...]
    mean: Vector
    innovation_misfit: float
    inflation: float
    perturbed: bool
    solve: str


@dataclass(frozen=True, slots=True)
class TikhonovEnsembleResult:
    """Final ensemble and history of a Tikhonov ensemble Kalman inversion.

    Histories have one entry per evaluated ensemble: the prior ensemble, then
    the ensemble after each update (``iterations + 1`` entries). The final
    ensemble is evaluated once more so that its misfit is recorded, which is
    counted in :attr:`forward_evaluations`.

    The misfit is the whitened norm of the augmented residual of the
    *ensemble-mean prediction*, ``|W ([y; mu] - mean_j [G(x_j); x_j])|`` with
    ``W = blockdiag(Gamma, C0)^-1/2``. Because ``G`` is nonlinear in general
    this is the mean of the predictions, not the prediction ``G(mean x)`` of
    the mean ([EQ-INV-TEKI-005]).

    This is the outcome of an optimizer. The final spread is the spread of a
    collapsing ensemble and is NOT a posterior uncertainty.

    Args:
        ensemble: Final members.
        initial_ensemble: Prior ensemble ``x_j ~ N(mu, C0)``.
        mean_history: Ensemble mean per evaluated ensemble.
        spread_history: Trace of the sample covariance (divisor ``J - 1``) per
            evaluated ensemble.
        misfit_history: Augmented whitened misfit of the mean prediction.
        data_misfit_history: Whitened misfit of the data block alone,
            ``|Gamma^-1/2 (y - mean_j G(x_j))|``.
        evaluation_history: Cumulative forward evaluations at each record.
        iterations: Number of updates performed.
        forward_evaluations: Total evaluations of ``G``,
            ``J (iterations + 1)``.
        perturbed: Whether the stochastic update was used.
        stop_reason: ``"iterations"`` (the iteration budget was spent).
        seed: Root seed of the random stream.

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-SCHILLINGS-STUART-2017],
        [EQ-INV-TEKI-004], [EQ-INV-TEKI-005].

    """

    ensemble: tuple[Vector, ...]
    initial_ensemble: tuple[Vector, ...]
    mean_history: tuple[Vector, ...]
    spread_history: Vector
    misfit_history: Vector
    data_misfit_history: Vector
    evaluation_history: tuple[int, ...]
    iterations: int
    forward_evaluations: int
    perturbed: bool
    stop_reason: str
    seed: int

    @property
    def ensemble_size(self) -> int:
        """Return the number of members ``J``.

        References:
            [REF-IGLESIAS-LAW-STUART-2013]. No equation is evaluated.

        """
        return len(self.ensemble)

    @property
    def mean(self) -> Vector:
        """Return the final ensemble mean, the TEKI point estimate.

        References:
            [EQ-INV-TEKI-005].

        """
        return self.mean_history[-1]


# ---------------------------------------------------------------------------
# Public algorithms
# ---------------------------------------------------------------------------


def ensemble_kalman_update(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
    *,
    perturbed: bool = True,
    inflation: float = 1.0,
    seed: int | RandomStream | None = None,
    solve: EnsembleSolve = "observation",
) -> EnsembleKalmanUpdate:
    """Apply one Kalman-type update to an ensemble.

    Let ``X`` (``n x J``) and ``Z`` (``m x J``) be the anomalies (member minus
    ensemble mean) of the members and predictions divided by ``sqrt(J - 1)``,
    so ``C_xg = X Z'`` and ``C_gg = Z Z'``, and let ``Gamma = L L'``. The
    stochastic update ([EQ-INV-TEKI-001]) is

    ``x_j^+ = x_j + C_xg (C_gg + alpha Gamma)^-1 (y + eta_j - g_j)``,
    ``eta_j ~ N(0, alpha Gamma)``.

    Whitening with ``L`` gives ``Zw = L^-1 Z`` and
    ``C_xg (C_gg + alpha Gamma)^-1 = X Zw' (Zw Zw' + alpha I_m)^-1 L^-1``.
    ``solve="observation"`` factors the ``m x m`` matrix
    ``Zw Zw' + alpha I_m`` by Cholesky. ``solve="ensemble"`` uses the
    push-through (Woodbury) identity ``Zw' (Zw Zw' + alpha I_m)^-1 =
    (Zw' Zw + alpha I_J)^-1 Zw'`` and factors the ``J x J`` matrix
    ``Zw' Zw + alpha I_J`` ([EQ-INV-TEKI-002]); no inverse is formed. The two
    agree to rounding and the ensemble form is cheaper for ``m >> J``.

    With ``perturbed=False`` the update is the deterministic ensemble
    transform square root ([EQ-INV-TEKI-003]): the mean moves by the Kalman
    gain applied to the unperturbed innovation ``y - mean_j g_j``, and the
    anomalies are multiplied by the symmetric transform
    ``T = (I_J + Zw' Zw / alpha)^-1/2`` computed from a symmetric
    eigen-decomposition. ``T`` maps the all-ones vector to itself, so the
    mean is unchanged by it. The updated sample mean and covariance equal
    ``x_bar + K (y - g_bar)`` and ``C_xx - K C_gx`` with
    ``K = C_xg (C_gg + alpha Gamma)^-1``, exactly for the ensemble's own
    sample statistics when ``G`` is linear. ``T`` is only one of the
    transforms ``T Q`` (``Q`` orthogonal with ``Q 1 = 1``) that give the same
    mean and covariance; the symmetric choice is used because it keeps the
    anomalies close to the prior ones.

    The members stay in the affine span of the prior ensemble. A small
    ensemble (``J - 1 < n``) therefore only corrects the data-informed
    subspace it spans; an ensemble with identical members gives zero anomalies
    and a zero update.

    Args:
        ensemble: ``J >= 2`` members, each of length ``n``.
        predictions: ``J`` predictions ``g_j = G(x_j)``, each of length ``m``.
        data: Observation ``y`` of length ``m``.
        noise: Covariance factor of ``Gamma`` with dimension ``m``
            (diagonal, dense or block diagonal from :mod:`probstat.linalg`).
        perturbed: ``True`` for perturbed observations
            ``eta_j ~ N(0, alpha Gamma)``, ``False`` for the deterministic
            square root.
        inflation: ``alpha > 0``, the factor on ``Gamma``.
        seed: Integer seed or stream for the perturbations; the draws are
            made member by member from ``resolve_stream(seed).generator()``.
            Ignored when ``perturbed`` is false.
        solve: ``"observation"`` or ``"ensemble"``.

    Returns:
        The updated ensemble with diagnostics.

    Raises:
        TypeError: If ``noise`` is not a covariance factor, ``perturbed`` is
            not a bool, or entries are not real numbers.
        ValueError: If ``J < 2``, dimensions disagree, values are not finite,
            ``inflation`` is not positive, or ``solve`` is unknown.

    Equation:
        [EQ-INV-TEKI-001], [EQ-INV-TEKI-002], [EQ-INV-TEKI-003].

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013],
        [REF-BISHOP-ETHERTON-MAJUMDAR-2001].

    """
    members, preds, observed = _validate_update_inputs(
        ensemble, predictions, data, noise
    )
    alpha = require_finite_real(inflation, name="inflation")
    if alpha <= 0.0:
        raise ValueError("inflation must be positive")
    if not isinstance(perturbed, bool):
        raise TypeError("perturbed must be a bool")
    if solve not in _SOLVES:
        raise ValueError("solve must be 'observation' or 'ensemble'")
    count = len(members)
    mean_x = _mean(members)
    mean_g = _mean(preds)
    scale = 1.0 / math.sqrt(count - 1)
    anomalies_x = [
        tuple((a - c) * scale for a, c in zip(row, mean_x, strict=True))
        for row in members
    ]
    anomalies_z = [
        noise.whiten(tuple((a - c) * scale for a, c in zip(row, mean_g, strict=True)))
        for row in preds
    ]
    solver = _make_solver(anomalies_z, alpha, solve)
    misfit = math.sqrt(
        math.fsum(
            v * v
            for v in noise.whiten(
                tuple(y - g for y, g in zip(observed, mean_g, strict=True))
            )
        )
    )

    if perturbed:
        generator = resolve_stream(seed).generator()
        root = math.sqrt(alpha)
        updated: list[Vector] = []
        for member, pred in zip(members, preds, strict=True):
            draw = noise.unwhiten(
                tuple(generator.gauss(0.0, 1.0) for _ in range(len(observed)))
            )
            innovation = tuple(
                y + root * e - g for y, e, g in zip(observed, draw, pred, strict=True)
            )
            step = _apply_gain(anomalies_x, solver(noise.whiten(innovation)))
            updated.append(tuple(a + b for a, b in zip(member, step, strict=True)))
        new_mean = _mean(updated)
        return EnsembleKalmanUpdate(
            ensemble=tuple(updated),
            mean=new_mean,
            innovation_misfit=misfit,
            inflation=alpha,
            perturbed=True,
            solve=solve,
        )

    innovation = tuple(y - g for y, g in zip(observed, mean_g, strict=True))
    shift = _apply_gain(anomalies_x, solver(noise.whiten(innovation)))
    posterior_mean = tuple(a + b for a, b in zip(mean_x, shift, strict=True))
    transform = _square_root_transform(anomalies_z, alpha)
    root_count = math.sqrt(count - 1)
    new_members: list[Vector] = []
    for j in range(count):
        new_members.append(
            tuple(
                posterior_mean[i]
                + root_count
                * math.fsum(anomalies_x[k][i] * transform[k][j] for k in range(count))
                for i in range(len(posterior_mean))
            )
        )
    return EnsembleKalmanUpdate(
        ensemble=tuple(new_members),
        mean=posterior_mean,
        innovation_misfit=misfit,
        inflation=alpha,
        perturbed=False,
        solve=solve,
    )


def tikhonov_ensemble_kalman_inversion(
    forward: Callable[[Vector], Sequence[float]],
    data: Sequence[float],
    *,
    noise: _CovarianceFactor,
    prior_mean: Sequence[float],
    prior_covariance: _CovarianceFactor,
    ensemble_size: int,
    iterations: int,
    seed: int | RandomStream | None = None,
    perturbed: bool = True,
    solve: EnsembleSolve = "observation",
) -> TikhonovEnsembleResult:
    """Run Tikhonov ensemble Kalman inversion (TEKI) on a forward callback.

    The augmented system ([EQ-INV-TEKI-004]) has forward map
    ``Ga(x) = [G(x); x]``, data ``ya = [y; mu]`` and noise
    ``Gamma_a = blockdiag(Gamma, C0)``. The initial ensemble is drawn from the
    prior ``x_j ~ N(mu, C0)`` and each iteration applies
    :func:`ensemble_kalman_update` with ``alpha = 1`` to the augmented system,
    with fresh perturbations ``eta_j ~ N(0, Gamma_a)`` when ``perturbed``.
    The TEKI formulation follows the augmented-system construction of the
    regularized ensemble Kalman methods based on Iglesias, Law and Stuart
    (2013); no further source is cited because none is registered in this
    library.

    TEKI is a derivative-free OPTIMIZER. For a linear ``G`` the
    augmented least-squares functional
    ``|Gamma^-1/2 (y - G x)|^2 + |C0^-1/2 (x - mu)|^2``
    ([EQ-INV-TEKI-005]) is the negative log posterior up to a factor two, and
    its minimizer is the Tikhonov (MAP) solution. The ensemble mean moves
    towards that minimizer as the iterations proceed (in the subspace spanned
    by the ensemble; exactly in the large-ensemble limit and when the
    ensemble spans the parameter space, ``J - 1 >= n``). The ensemble
    covariance, however, collapses, so the final spread is NOT a posterior
    uncertainty and the members are not posterior samples. Use
    :func:`~probstat.inverse.linear_gaussian_posterior` for a linear
    posterior, or an approximate sampler for nonlinear problems.

    Every member remains in the affine span of the initial ensemble. Each
    iteration evaluates ``G`` once per member, and the final ensemble is
    evaluated once more to record its misfit, so the total cost is
    ``J (iterations + 1)`` evaluations.

    Args:
        forward: Callable mapping a length-``n`` vector to a length-``m``
            prediction. Exceptions it raises propagate unchanged.
        data: Observation ``y`` of length ``m``.
        noise: Covariance factor of ``Gamma`` (dimension ``m``).
        prior_mean: Prior mean ``mu`` of length ``n``.
        prior_covariance: Covariance factor of ``C0`` (dimension ``n``).
        ensemble_size: Number of members ``J >= 2``.
        iterations: Number of updates, at least one.
        seed: Integer seed or stream. The prior ensemble uses the child
            ``"ensemble"`` and update ``i`` the child
            ``("perturbations", i)``.
        perturbed: Stochastic (``True``) or deterministic square-root
            (``False``) update.
        solve: ``"observation"`` or ``"ensemble"``; the augmented system has
            dimension ``m + n``, so ``"ensemble"`` pays off when ``m + n``
            exceeds ``J``.

    Returns:
        Final ensemble with the iteration history.

    Raises:
        TypeError: If ``forward`` is not callable, a covariance is not a
            covariance factor, or arguments have the wrong type.
        ValueError: If ``J < 2``, ``iterations < 1``, dimensions disagree,
            values are not finite, or ``forward`` returns a vector of the
            wrong length or with non-finite entries.

    Equation:
        [EQ-INV-TEKI-004], [EQ-INV-TEKI-005].

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-SCHILLINGS-STUART-2017],
        [REF-EMERICK-REYNOLDS-2013].

    """
    if not callable(forward):
        raise TypeError("forward must be callable")
    observed = _vector(data, "data")
    mu = _vector(prior_mean, "prior_mean")
    for name, factor, size in (
        ("noise", noise, len(observed)),
        ("prior_covariance", prior_covariance, len(mu)),
    ):
        if not is_covariance_factor(factor):
            raise TypeError(f"{name} must be a covariance factor of probstat.linalg")
        if factor.dimension != size:
            raise ValueError(f"{name} dimension must be {size}, got {factor.dimension}")
    count = require_non_negative_integer(ensemble_size, name="ensemble_size")
    if count < 2:
        raise ValueError("ensemble_size must be at least two")
    steps = require_non_negative_integer(iterations, name="iterations")
    if steps < 1:
        raise ValueError("iterations must be at least one")
    if not isinstance(perturbed, bool):
        raise TypeError("perturbed must be a bool")
    if solve not in _SOLVES:
        raise ValueError("solve must be 'observation' or 'ensemble'")

    stream = resolve_stream(seed)
    n = len(mu)
    m = len(observed)
    augmented_noise = BlockDiagonalCovariance([noise, prior_covariance])
    augmented_data = observed + mu
    generator = stream.child("ensemble").generator()
    members: list[Vector] = []
    for _ in range(count):
        z = tuple(generator.gauss(0.0, 1.0) for _ in range(n))
        members.append(
            tuple(c + d for c, d in zip(mu, prior_covariance.unwhiten(z), strict=True))
        )
    initial = tuple(members)

    means: list[Vector] = []
    spreads: list[float] = []
    misfits: list[float] = []
    data_misfits: list[float] = []
    evaluations: list[int] = []
    total = 0
    for step in range(steps + 1):
        augmented = []
        for member in members:
            prediction = _vector(forward(member), "forward output")
            if len(prediction) != m:
                raise ValueError(
                    f"forward returned length {len(prediction)}, expected {m}"
                )
            augmented.append(prediction + member)
        total += count
        center = _mean(members)
        center_g = _mean(augmented)
        means.append(center)
        spreads.append(
            math.fsum((row[i] - center[i]) ** 2 for row in members for i in range(n))
            / (count - 1)
        )
        residual = tuple(a - b for a, b in zip(augmented_data, center_g, strict=True))
        misfits.append(
            math.sqrt(math.fsum(v * v for v in augmented_noise.whiten(residual)))
        )
        data_misfits.append(
            math.sqrt(math.fsum(v * v for v in noise.whiten(residual[:m])))
        )
        evaluations.append(total)
        if step == steps:
            break
        members = list(
            ensemble_kalman_update(
                members,
                augmented,
                augmented_data,
                augmented_noise,
                perturbed=perturbed,
                inflation=1.0,
                seed=stream.child("perturbations", step),
                solve=solve,
            ).ensemble
        )
    return TikhonovEnsembleResult(
        ensemble=tuple(members),
        initial_ensemble=initial,
        mean_history=tuple(means),
        spread_history=tuple(spreads),
        misfit_history=tuple(misfits),
        data_misfit_history=tuple(data_misfits),
        evaluation_history=tuple(evaluations),
        iterations=steps,
        forward_evaluations=total,
        perturbed=perturbed,
        stop_reason="iterations",
        seed=stream.seed,
    )


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _vector(values: object, name: str) -> Vector:
    """Return a finite real vector as a tuple of floats.

    Args:
        values: Candidate sequence of real numbers.
        name: Name used in error messages.

    Returns:
        Tuple of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If it is empty or has a non-finite entry.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(
        require_finite_real(v, name=f"{name}[{i}]") for i, v in enumerate(values)
    )


def _validate_update_inputs(
    ensemble: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    data: Sequence[float],
    noise: _CovarianceFactor,
) -> tuple[list[Vector], list[Vector], Vector]:
    """Validate and normalize the inputs of :func:`ensemble_kalman_update`.

    Args:
        ensemble: Candidate members.
        predictions: Candidate predictions.
        data: Candidate observation.
        noise: Candidate covariance factor.

    Returns:
        Members, predictions and data as tuples of floats.

    Raises:
        TypeError: If ``noise`` is not a covariance factor or entries are not
            real numbers.
        ValueError: If ``J < 2``, lengths disagree or values are not finite.

    References:
        [REF-PROBSTAT-VALIDATION], [EQ-INV-TEKI-001].

    """
    if isinstance(ensemble, (str, bytes)) or not isinstance(ensemble, Sequence):
        raise TypeError("ensemble must be a sequence of vectors")
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence of vectors")
    members = [_vector(row, f"ensemble[{j}]") for j, row in enumerate(ensemble)]
    preds = [_vector(row, f"predictions[{j}]") for j, row in enumerate(predictions)]
    if len(members) < 2:
        raise ValueError("the ensemble needs at least two members")
    if len(preds) != len(members):
        raise ValueError("predictions must have one row per ensemble member")
    if any(len(row) != len(members[0]) for row in members):
        raise ValueError("all members must have the same length")
    if any(len(row) != len(preds[0]) for row in preds):
        raise ValueError("all predictions must have the same length")
    observed = _vector(data, "data")
    if len(observed) != len(preds[0]):
        raise ValueError("data length must equal the prediction length")
    if not is_covariance_factor(noise):
        raise TypeError("noise must be a covariance factor of probstat.linalg")
    if noise.dimension != len(observed):
        raise ValueError("noise dimension must equal the data length")
    return members, preds, observed


def _mean(rows: Sequence[Vector]) -> Vector:
    """Return the componentwise mean of equal-length vectors.

    References:
        [EQ-INV-TEKI-001].

    """
    count = len(rows)
    return tuple(math.fsum(row[i] for row in rows) / count for i in range(len(rows[0])))


def _make_solver(
    anomalies_z: Sequence[Vector], alpha: float, solve: str
) -> Callable[[Vector], Vector]:
    """Return the map ``r -> coefficients c`` with ``Delta x = X c``.

    With ``Zw`` the whitened scaled prediction anomalies, the coefficients of
    a whitened innovation ``r`` are ``Zw' (Zw Zw' + alpha I_m)^-1 r`` for
    ``solve="observation"`` and ``(Zw' Zw + alpha I_J)^-1 Zw' r`` for
    ``"ensemble"`` ([EQ-INV-TEKI-002]).

    Args:
        anomalies_z: ``J`` whitened, ``1/sqrt(J-1)``-scaled anomalies (length
            ``m`` each).
        alpha: Inflation.
        solve: ``"observation"`` or ``"ensemble"``.

    Returns:
        Callable returning the ``J`` coefficients.

    References:
        [EQ-INV-TEKI-002].

    """
    count = len(anomalies_z)
    size = len(anomalies_z[0])
    if solve == "observation":
        system: Matrix = tuple(
            tuple(
                math.fsum(z[p] * z[q] for z in anomalies_z) + (alpha if p == q else 0.0)
                for q in range(size)
            )
            for p in range(size)
        )
        lower = cholesky(system)

        def observation_space(residual: Vector) -> Vector:
            """Solve the ``m x m`` system and map back with ``Zw'``.

            References:
                [EQ-INV-TEKI-002].

            """
            solved = cholesky_solve(lower, residual)
            return tuple(
                math.fsum(a * b for a, b in zip(z, solved, strict=True))
                for z in anomalies_z
            )

        return observation_space
    gram: Matrix = tuple(
        tuple(
            math.fsum(
                a * b for a, b in zip(anomalies_z[i], anomalies_z[j], strict=True)
            )
            + (alpha if i == j else 0.0)
            for j in range(count)
        )
        for i in range(count)
    )
    lower_j = cholesky(gram)

    def ensemble_space(residual: Vector) -> Vector:
        """Solve the ``J x J`` system with the right-hand side ``Zw' r``.

        References:
            [EQ-INV-TEKI-002].

        """
        rhs = tuple(
            math.fsum(a * b for a, b in zip(z, residual, strict=True))
            for z in anomalies_z
        )
        return cholesky_solve(lower_j, rhs)

    return ensemble_space


def _apply_gain(anomalies_x: Sequence[Vector], coefficients: Vector) -> Vector:
    """Return ``X c`` for scaled parameter anomalies ``X`` (``J`` columns).

    References:
        [EQ-INV-TEKI-001].

    """
    return tuple(
        math.fsum(
            c * column[i] for c, column in zip(coefficients, anomalies_x, strict=True)
        )
        for i in range(len(anomalies_x[0]))
    )


def _square_root_transform(anomalies_z: Sequence[Vector], alpha: float) -> Matrix:
    """Return ``T = (I_J + Zw' Zw / alpha)^-1/2`` by symmetric eigen-decomposition.

    The matrix is symmetric positive definite with eigenvalues at least one,
    so ``T`` exists and is unique among symmetric square roots.

    Args:
        anomalies_z: ``J`` whitened, scaled prediction anomalies.
        alpha: Inflation.

    Returns:
        The ``J x J`` symmetric transform.

    References:
        [EQ-INV-TEKI-003], [REF-BISHOP-ETHERTON-MAJUMDAR-2001].

    """
    count = len(anomalies_z)
    matrix: Matrix = tuple(
        tuple(
            math.fsum(
                a * b for a, b in zip(anomalies_z[i], anomalies_z[j], strict=True)
            )
            / alpha
            + (1.0 if i == j else 0.0)
            for j in range(count)
        )
        for i in range(count)
    )
    values, vectors = symmetric_eigen(matrix)
    weights = tuple(1.0 / math.sqrt(v) for v in values)
    return tuple(
        tuple(
            math.fsum(vectors[i][k] * weights[k] * vectors[j][k] for k in range(count))
            for j in range(count)
        )
        for i in range(count)
    )
