"""Derivative-free ensemble Kalman inversion and the ES-MDA smoother.

Both methods move an ensemble of parameter vectors towards the data with a
Kalman-type update whose gain is built from ensemble covariances between the
parameters and the predicted observations. No gradient of the forward model
is needed: each iteration costs one forward evaluation per member, and the
members can be evaluated independently.

The ensemble lives in the Nataf standard-normal space ``u`` of the prior of a
:class:`~probstat.inverse.CalibrationProblem`, where the prior is exactly
``N(0, I)``; every member is mapped to the physical parameters
``theta = T^-1(u)`` before the forward model runs. With Gaussian priors and a
forward model linear in ``theta`` the composite map is affine in ``u``.

Ensemble Kalman inversion (:func:`ensemble_kalman_inversion`)
    Iterates the update of Iglesias, Law and Stuart (2013) with the
    observation-error covariance ``Gamma`` ([EQ-INV-EKI-001]). It is a
    derivative-free *optimizer and iterative regularizer* of the data misfit:
    every member stays in the span of the initial ensemble and the ensemble
    collapses towards a point as the iterations proceed. Its spread is not a
    posterior uncertainty. For ill-posed problems the iteration should be
    stopped early, for example by the discrepancy principle
    ([EQ-INV-EKI-002]).

Ensemble smoother with multiple data assimilation (:func:`ensemble_smoother_mda`)
    Assimilates the same data ``N_a`` times with inflated error covariances
    ``alpha_i Gamma`` and ``sum 1/alpha_i = 1`` (Emerick and Reynolds 2013;
    [EQ-INV-ESMDA-001]). For a forward model affine in ``u`` and Gaussian
    errors the composition of the updates is the exact Bayesian update, so
    the ensemble samples the posterior as the ensemble size grows. For a
    nonlinear model it is an approximation whose error the user must assess,
    for instance against :meth:`CalibrationProblem.sample_posterior`.

Only problems whose noise models are all known-scale Gaussian are accepted:
the update needs ``Gamma`` as a fixed matrix. ``Gamma`` is diagonal with
entries ``sigma_q^2 + sigma_delta_q^2`` (noise plus declared discrepancy
variance) in the order of :attr:`CalibrationProblem.observed`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, cholesky, cholesky_solve

from .nonlinear import CalibrationProblem

_STOPPING_RULES = ("iterations", "discrepancy")
_INFLATION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleInversionResult:
    """Final ensemble and iteration history of an ensemble Kalman inversion.

    Histories have one entry per evaluated ensemble: the prior ensemble, then
    the ensemble after each update (so ``updates + 1`` entries). The data
    misfit is ``|Gamma^-1/2 (y - mean_j G(theta_j))|``, the whitened norm of
    the residual of the ensemble-mean prediction ([EQ-INV-EKI-002]).

    Args:
        method: ``"ensemble Kalman inversion"`` or ``"ES-MDA"``.
        parameter_names: Calibration-parameter names.
        ensemble: Final members in physical coordinates ``theta``.
        standard_ensemble: Final members in the prior's standard-normal
            coordinates ``u``.
        mean_history: Ensemble mean of ``theta`` per evaluated ensemble.
        spread_history: Ensemble standard deviation of each ``theta``
            coordinate per evaluated ensemble (divisor ``N - 1``).
        standard_mean_history: Ensemble mean of ``u`` per evaluated ensemble.
        standard_spread_history: Ensemble standard deviation of each ``u``
            coordinate per evaluated ensemble.
        misfit_history: Whitened misfit of the ensemble-mean prediction per
            evaluated ensemble.
        inflation_coefficients: ``alpha_i`` used by each update (all one for
            ensemble Kalman inversion).
        updates: Number of ensemble updates performed.
        stopped_by: ``"iterations"`` (the iteration budget or the inflation
            schedule was exhausted) or ``"discrepancy"``.
        discrepancy_threshold: ``tau * eta`` of the discrepancy principle, or
            ``nan`` when it was not used.
        perturbed_observations: Whether each member saw its own perturbed
            copy of the data.
        forward_evaluations: Evaluations of the full observation operator,
            ``N (updates + 1)``.
        model_calls: Forward-model calls, ``forward_evaluations`` times the
            number of experiments.
        seed: Root seed of the random stream.

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013],
        [EQ-INV-EKI-001], [EQ-INV-ESMDA-001].

    """

    method: str
    parameter_names: tuple[str, ...]
    ensemble: tuple[Vector, ...]
    standard_ensemble: tuple[Vector, ...]
    mean_history: tuple[Vector, ...]
    spread_history: tuple[Vector, ...]
    standard_mean_history: tuple[Vector, ...]
    standard_spread_history: tuple[Vector, ...]
    misfit_history: Vector
    inflation_coefficients: Vector
    updates: int
    stopped_by: str
    discrepancy_threshold: float
    perturbed_observations: bool
    forward_evaluations: int
    model_calls: int
    seed: int

    @property
    def ensemble_size(self) -> int:
        """Return the number of ensemble members ``N``.

        References:
            [REF-IGLESIAS-LAW-STUART-2013]. No equation is evaluated.

        """
        return len(self.ensemble)

    def mean(self) -> Mapping[str, float]:
        """Return the final ensemble mean of ``theta`` keyed by name.

        Returns:
            Read-only mapping from parameter name to the ensemble mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-EMERICK-REYNOLDS-2013].

        """
        return MappingProxyType(
            dict(zip(self.parameter_names, self.mean_history[-1], strict=True))
        )

    def covariance(self) -> Matrix:
        """Return the final ensemble covariance of ``theta`` (divisor ``N - 1``).

        Returns:
            Symmetric sample covariance matrix in parameter order.

        Equation:
            [EQ-INV-EKI-001].

        References:
            [REF-EMERICK-REYNOLDS-2013], [REF-EVENSEN-2003].

        """
        return _covariance(self.ensemble)

    def standard_covariance(self) -> Matrix:
        """Return the final ensemble covariance of ``u`` (divisor ``N - 1``).

        Returns:
            Symmetric sample covariance matrix in parameter order.

        Equation:
            [EQ-INV-EKI-001].

        References:
            [REF-SCHILLINGS-STUART-2017] (collapse of this covariance under
            ensemble Kalman inversion).

        """
        return _covariance(self.standard_ensemble)

    def draws(self) -> tuple[Mapping[str, float], ...]:
        """Return the final members keyed by parameter name.

        Returns:
            One read-only mapping per ensemble member.

        References:
            [REF-EMERICK-REYNOLDS-2013]. No equation is evaluated.

        """
        return tuple(
            MappingProxyType(dict(zip(self.parameter_names, member, strict=True)))
            for member in self.ensemble
        )

    def summary(self) -> str:
        """Return a one-paragraph description of the run.

        Returns:
            Text with method, ensemble size, updates, stopping reason, misfit
            reduction, final means and spreads, and the forward cost.

        References:
            [REF-IGLESIAS-LAW-STUART-2013], [REF-IGLESIAS-2016]. No equation
            is evaluated.

        """
        estimates = ", ".join(
            f"{name} = {center:.6g} (spread {spread:.3g})"
            for name, center, spread in zip(
                self.parameter_names,
                self.mean_history[-1],
                self.spread_history[-1],
                strict=True,
            )
        )
        text = (
            f"{self.method}: {self.ensemble_size} members, {self.updates} "
            f"updates (stopped by {self.stopped_by}); whitened misfit of the "
            f"mean prediction {self.misfit_history[0]:.4g} -> "
            f"{self.misfit_history[-1]:.4g}"
        )
        if not math.isnan(self.discrepancy_threshold):
            text += f" (threshold {self.discrepancy_threshold:.4g})"
        text += (
            f"; {estimates}; {self.forward_evaluations} forward evaluations "
            f"({self.model_calls} model calls)."
        )
        if self.method == "ensemble Kalman inversion":
            text += (
                " The spread is that of a collapsing optimizer ensemble, not a "
                "posterior uncertainty."
            )
        return text


# ---------------------------------------------------------------------------
# Public algorithms
# ---------------------------------------------------------------------------


def ensemble_kalman_inversion(
    problem: CalibrationProblem,
    *,
    ensemble_size: int,
    iterations: int,
    seed: int | RandomStream | None = None,
    perturb_observations: bool = True,
    stopping: str = "iterations",
    discrepancy_factor: float = 1.5,
    noise_level: float | None = None,
) -> EnsembleInversionResult:
    """Minimize the data misfit by iterated ensemble Kalman updates.

    Draws ``u_j ~ N(0, I)``, ``j = 1..N``, and repeats
    ``u_j <- u_j + C^up (C^pp + Gamma)^-1 (y + eta_j - G(u_j))`` with
    ensemble covariances ``C^up``, ``C^pp`` and, when
    ``perturb_observations`` is true, fresh ``eta_j ~ N(0, Gamma)`` at every
    iteration ([EQ-INV-EKI-001]). ``G(u)`` is the forward model at
    ``theta = T^-1(u)`` (the problem's Nataf map).

    This is a derivative-free optimizer and iterative regularizer, **not a
    posterior sampler**. The members remain in the affine span of the initial
    ensemble, and the ensemble covariance shrinks towards zero as the
    iterations proceed (in the linear case, ``n`` iterations with perturbed
    data approximate the posterior of the likelihood raised to the power
    ``n``). Use the ensemble mean as a point estimate; use
    :func:`ensemble_smoother_mda` or :meth:`CalibrationProblem.sample_posterior`
    for uncertainty.

    With ``stopping="discrepancy"`` the iteration stops before the first
    update at which the whitened misfit of the ensemble-mean prediction
    satisfies ``|Gamma^-1/2 (y - mean_j G(u_j))| <= tau eta``
    ([EQ-INV-EKI-002], Iglesias 2016); ``iterations`` is then the maximum
    number of updates. Stopping at the noise level keeps the iteration from
    fitting the noise of an ill-posed problem.

    Args:
        problem: Calibration problem whose noise models are all known-scale
            Gaussian.
        ensemble_size: Number of members ``N >= 2``.
        iterations: Number of updates (``stopping="iterations"``) or the
            maximum number of updates (``"discrepancy"``); at least one.
        seed: Integer seed or stream. The prior ensemble uses the child
            stream ``"ensemble"`` and the perturbations of update ``n`` the
            child ``("perturbations", n)``.
        perturb_observations: Whether each member sees its own perturbed
            data ``y + eta_j``. Iglesias, Law and Stuart (2013) report more
            accurate estimates with perturbations than without.
        stopping: ``"iterations"`` (default) or ``"discrepancy"``.
        discrepancy_factor: ``tau >= 1`` of the discrepancy principle. The
            default 1.5 is a library choice satisfying ``tau > 1``.
        noise_level: ``eta``, the whitened norm of the noise realization.
            Defaults to ``sqrt(M)`` for ``M`` observations, the root mean
            square of ``|Gamma^-1/2 eps|`` for Gaussian errors.

    Returns:
        Final ensemble with its iteration history.

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem` or
            ``perturb_observations`` is not a bool.
        ValueError: For unknown-noise or Student-t problems, ``N < 2``,
            ``iterations < 1``, an unknown ``stopping`` rule,
            ``discrepancy_factor < 1``, or a non-positive ``noise_level``.

    Equation:
        [EQ-INV-EKI-001], [EQ-INV-EKI-002].

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-IGLESIAS-2016],
        [REF-SCHILLINGS-STUART-2017].

    """
    variances = _observation_variances(problem)
    if stopping not in _STOPPING_RULES:
        raise ValueError("stopping must be 'iterations' or 'discrepancy'")
    updates = require_non_negative_integer(iterations, name="iterations")
    if updates == 0:
        raise ValueError("iterations must be at least one")
    if not isinstance(perturb_observations, bool):
        raise TypeError("perturb_observations must be a bool")
    threshold = math.nan
    if stopping == "discrepancy":
        tau = require_finite_real(discrepancy_factor, name="discrepancy_factor")
        if tau < 1.0:
            raise ValueError(
                "discrepancy_factor must be at least one: a smaller tau asks "
                "the ensemble to fit the data below the noise level"
            )
        if noise_level is None:
            eta = math.sqrt(len(variances))
        else:
            eta = require_finite_real(noise_level, name="noise_level")
            if eta <= 0.0:
                raise ValueError("noise_level must be positive")
        threshold = tau * eta
    return _run(
        problem,
        variances,
        ensemble_size=ensemble_size,
        inflation=tuple(1.0 for _ in range(updates)),
        seed=seed,
        perturb=perturb_observations,
        threshold=threshold,
        method="ensemble Kalman inversion",
    )


def ensemble_smoother_mda(
    problem: CalibrationProblem,
    *,
    ensemble_size: int,
    inflation_coefficients: Sequence[float],
    seed: int | RandomStream | None = None,
) -> EnsembleInversionResult:
    """Sample the posterior approximately by multiple data assimilation.

    Draws ``u_j ~ N(0, I)`` and for ``i = 1..N_a`` applies
    ``u_j <- u_j + C^up (C^pp + alpha_i Gamma)^-1
    (y + sqrt(alpha_i) Gamma^1/2 z_ij - G(u_j))`` with ``z_ij ~ N(0, I)``,
    where ``sum_i 1/alpha_i = 1`` ([EQ-INV-ESMDA-001]). A single coefficient
    ``alpha = 1`` is the one-step ensemble smoother.

    The inflation condition makes the product of the tempered likelihoods
    ``L^(1/alpha_i)`` equal to the full likelihood. For a forward model
    affine in ``u`` (Gaussian priors and a model linear in ``theta``) and
    Gaussian errors every update is an exact Gaussian conditioning, so the
    ensemble samples the exact posterior as ``N -> infinity``. For a
    nonlinear model the result is an approximation; more, smaller steps
    (larger ``alpha_i``) usually reduce its error.

    Args:
        problem: Calibration problem whose noise models are all known-scale
            Gaussian.
        ensemble_size: Number of members ``N >= 2``.
        inflation_coefficients: ``alpha_1, ..., alpha_{N_a}``, each positive
            and finite, with ``sum 1/alpha_i = 1`` to a relative tolerance of
            ``1e-9``.
        seed: Integer seed or stream, with the same child streams as
            :func:`ensemble_kalman_inversion`.

    Returns:
        Final ensemble with its assimilation history; the histories have
        ``N_a + 1`` entries (the final ensemble is evaluated once more for
        its misfit).

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem` or
            ``inflation_coefficients`` is a string.
        ValueError: For unknown-noise or Student-t problems, ``N < 2``, an
            empty, non-positive, or non-finite inflation sequence, or
            ``sum 1/alpha_i != 1``.

    Equation:
        [EQ-INV-ESMDA-001].

    References:
        [REF-EMERICK-REYNOLDS-2013], [REF-EVENSEN-2003].

    """
    variances = _observation_variances(problem)
    if isinstance(inflation_coefficients, (str, bytes)):
        raise TypeError("inflation_coefficients must be a sequence of numbers")
    alphas = tuple(
        require_finite_real(value, name=f"inflation_coefficients[{index}]")
        for index, value in enumerate(inflation_coefficients)
    )
    if not alphas:
        raise ValueError("inflation_coefficients must not be empty")
    if any(value <= 0.0 for value in alphas):
        raise ValueError("inflation coefficients must be positive")
    total = math.fsum(1.0 / value for value in alphas)
    if abs(total - 1.0) > _INFLATION_TOLERANCE:
        raise ValueError(
            "inflation coefficients must satisfy sum(1 / alpha) = 1 so that "
            f"the data are assimilated exactly once in total; got {total:.12g}"
        )
    return _run(
        problem,
        variances,
        ensemble_size=ensemble_size,
        inflation=alphas,
        seed=seed,
        perturb=True,
        threshold=math.nan,
        method="ES-MDA",
    )


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _observation_variances(problem: CalibrationProblem) -> Vector:
    """Return the diagonal of ``Gamma`` aligned with ``problem.observed``.

    Each entry is ``sigma_q^2 + sigma_delta_q^2`` of the observed quantity's
    known Gaussian noise model.

    Args:
        problem: Calibration problem.

    Returns:
        Positive observation-error variances in experiment-then-quantity
        order.

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem`.
        ValueError: If a noise scale is unknown or a noise model is
            Student-t, with the reason ensemble Kalman updates cannot use it.

    Equation:
        [EQ-CAL-LIK-001], [EQ-INV-EKI-001].

    References:
        [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013].

    """
    if not isinstance(problem, CalibrationProblem):
        raise TypeError("problem must be a CalibrationProblem")
    by_name = {item.quantity: item for item in problem.noise}
    for item in problem.noise:
        if not item.is_known:
            raise ValueError(
                f"noise for {item.quantity!r} has an unknown scale; ensemble "
                "Kalman updates need the observation-error covariance Gamma "
                "as a fixed matrix. Declare a known standard_deviation or "
                "use CalibrationProblem.sample_posterior, which samples the "
                "noise scale"
            )
        if item.degrees_of_freedom is not None:
            raise ValueError(
                f"noise for {item.quantity!r} is Student-t; the ensemble "
                "Kalman update is a Gaussian conditioning on second moments "
                "and has no form for heavy-tailed errors (whose variance is "
                "infinite for nu <= 2). Use "
                "CalibrationProblem.sample_posterior"
            )
    variances: list[float] = []
    for experiment in problem.experiments:
        for quantity in experiment.observations:
            item = by_name[quantity]
            sigma = item.standard_deviation
            assert sigma is not None  # guaranteed by the checks above
            variances.append(sigma * sigma + item.discrepancy_variance)
    if len(variances) != problem.observation_count:
        raise ValueError("observation layout does not match the problem")
    return tuple(variances)


def _mean(members: Sequence[Vector]) -> Vector:
    """Return the componentwise ensemble mean.

    Equation:
        [EQ-MC-MEAN-001].

    References:
        [REF-EVENSEN-2003].

    """
    count = len(members)
    return tuple(
        math.fsum(member[k] for member in members) / count
        for k in range(len(members[0]))
    )


def _spread(members: Sequence[Vector], center: Vector) -> Vector:
    """Return the componentwise ensemble standard deviation (divisor ``N - 1``).

    References:
        [REF-EVENSEN-2003]. [EQ-INV-EKI-001] uses the same divisor.

    """
    denominator = len(members) - 1
    return tuple(
        math.sqrt(
            math.fsum((member[k] - center[k]) ** 2 for member in members) / denominator
        )
        for k in range(len(center))
    )


def _covariance(members: Sequence[Vector]) -> Matrix:
    """Return the sample covariance of ensemble members (divisor ``N - 1``).

    References:
        [REF-EVENSEN-2003], [EQ-INV-EKI-001].

    """
    center = _mean(members)
    size = len(center)
    denominator = len(members) - 1
    deviations = [
        tuple(value - c for value, c in zip(member, center, strict=True))
        for member in members
    ]
    rows: list[list[float]] = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = math.fsum(d[i] * d[j] for d in deviations) / denominator
            rows[i][j] = value
            rows[j][i] = value
    return tuple(tuple(row) for row in rows)


def _whitened_gain(
    deviations_u: Sequence[Vector],
    deviations_g: Sequence[Vector],
    shift: float,
) -> Matrix:
    """Return the whitened ensemble Kalman gain ``A'B (B'B + c I_M)^-1``.

    ``A`` (``N x d``) and ``B`` (``N x M``) hold the deviations of ``u`` and of
    the whitened predictions ``Gamma^-1/2 G(u)`` from their ensemble means and
    ``c = alpha (N - 1)``. When ``M > N`` the equal form
    ``A' (B B' + c I_N)^-1 B`` (push-through identity) is used, so the
    factored system always has the smaller of the two dimensions. Both
    systems are positive definite for ``c > 0``.

    Returns:
        ``d x M`` gain applied to whitened innovations.

    References:
        [EQ-INV-EKI-001], [REF-EVENSEN-2003] (ensemble gain from
        ensemble anomalies).

    """
    count = len(deviations_g)
    size = len(deviations_g[0])
    dimension = len(deviations_u[0])
    if size <= count:
        system = tuple(
            tuple(
                math.fsum(b[p] * b[q] for b in deviations_g)
                + (shift if p == q else 0.0)
                for q in range(size)
            )
            for p in range(size)
        )
        lower = cholesky(system)
        cross = (
            tuple(
                math.fsum(
                    a[k] * b[p] for a, b in zip(deviations_u, deviations_g, strict=True)
                )
                for p in range(size)
            )
            for k in range(dimension)
        )
        # The system is symmetric, so row k of P S^-1 is S^-1 applied to row k of P.
        return tuple(cholesky_solve(lower, row) for row in cross)
    system = tuple(
        tuple(
            math.fsum(
                x * y for x, y in zip(deviations_g[i], deviations_g[j], strict=True)
            )
            + (shift if i == j else 0.0)
            for j in range(count)
        )
        for i in range(count)
    )
    lower = cholesky(system)
    rows: list[Vector] = []
    for k in range(dimension):
        weights = cholesky_solve(lower, tuple(a[k] for a in deviations_u))
        rows.append(
            tuple(
                math.fsum(w * b[p] for w, b in zip(weights, deviations_g, strict=True))
                for p in range(size)
            )
        )
    return tuple(rows)


def _run(
    problem: CalibrationProblem,
    variances: Vector,
    *,
    ensemble_size: int,
    inflation: Vector,
    seed: int | RandomStream | None,
    perturb: bool,
    threshold: float,
    method: str,
) -> EnsembleInversionResult:
    """Run the shared ensemble loop of EKI and ES-MDA.

    For every evaluated ensemble: predict, record the history, test the
    discrepancy principle, and, if updates remain, apply [EQ-INV-EKI-001]
    with ``Gamma`` replaced by ``alpha_n Gamma``.

    References:
        [EQ-INV-EKI-001], [EQ-INV-EKI-002], [EQ-INV-ESMDA-001],
        [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013].

    """
    count = require_non_negative_integer(ensemble_size, name="ensemble_size")
    if count < 2:
        raise ValueError("ensemble_size must be at least two")
    stream = resolve_stream(seed)
    dimension = problem.model.dimension
    inputs = problem.model.inputs
    observed = problem.observed
    scales = tuple(math.sqrt(value) for value in variances)
    experiments = len(problem.experiments)

    generator = stream.child("ensemble").generator()
    members: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    ]

    mean_history: list[Vector] = []
    spread_history: list[Vector] = []
    standard_mean_history: list[Vector] = []
    standard_spread_history: list[Vector] = []
    misfit_history: list[float] = []
    evaluations = 0
    used: list[float] = []
    stopped_by = "iterations"
    physical: list[Vector] = []

    step = 0
    while True:
        physical = [inputs.from_standard_normal(member) for member in members]
        # Whitened predictions Gamma^-1/2 G(u_j).
        predictions = [
            tuple(
                value / scale
                for value, scale in zip(problem.predictions(theta), scales, strict=True)
            )
            for theta in physical
        ]
        evaluations += count
        center_u = _mean(members)
        center_theta = _mean(physical)
        center_g = _mean(predictions)
        standard_mean_history.append(center_u)
        standard_spread_history.append(_spread(members, center_u))
        mean_history.append(center_theta)
        spread_history.append(_spread(physical, center_theta))
        whitened_data = tuple(
            value / scale for value, scale in zip(observed, scales, strict=True)
        )
        misfit = math.sqrt(
            math.fsum(
                (y - g) ** 2 for y, g in zip(whitened_data, center_g, strict=True)
            )
        )
        misfit_history.append(misfit)
        if not math.isnan(threshold) and misfit <= threshold:
            stopped_by = "discrepancy"
            break
        if step == len(inflation):
            break
        alpha = inflation[step]
        members = _update(
            members,
            center_u,
            predictions,
            center_g,
            whitened_data,
            alpha,
            stream.child("perturbations", step) if perturb else None,
        )
        used.append(alpha)
        step += 1

    return EnsembleInversionResult(
        method=method,
        parameter_names=problem.parameter_names,
        ensemble=tuple(physical),
        standard_ensemble=tuple(members),
        mean_history=tuple(mean_history),
        spread_history=tuple(spread_history),
        standard_mean_history=tuple(standard_mean_history),
        standard_spread_history=tuple(standard_spread_history),
        misfit_history=tuple(misfit_history),
        inflation_coefficients=tuple(used),
        updates=step,
        stopped_by=stopped_by,
        discrepancy_threshold=threshold,
        perturbed_observations=perturb,
        forward_evaluations=evaluations,
        model_calls=evaluations * experiments,
        seed=stream.seed,
    )


def _update(
    members: Sequence[Vector],
    center_u: Vector,
    predictions: Sequence[Vector],
    center_g: Vector,
    whitened_data: Vector,
    alpha: float,
    perturbations: RandomStream | None,
) -> list[Vector]:
    """Apply one ensemble Kalman update in whitened observation space.

    With ``A`` and ``B`` the deviations of ``u`` and of
    ``Gamma^-1/2 G(u)`` from their ensemble means, ``C^up = A'B
    Gamma^1/2 / (N - 1)`` and ``C^pp = Gamma^1/2 B'B Gamma^1/2 / (N - 1)``,
    so ``C^up (C^pp + alpha Gamma)^-1 = A'B (B'B + alpha (N - 1) I)^-1
    Gamma^-1/2``. The whitened innovation of member ``j`` is
    ``Gamma^-1/2 (y - G(u_j)) + sqrt(alpha) z_j`` with ``z_j ~ N(0, I)``
    drawn member by member when ``perturbations`` is given.

    References:
        [EQ-INV-EKI-001], [EQ-INV-ESMDA-001], [REF-EMERICK-REYNOLDS-2013].

    """
    count = len(members)
    deviations_u = [
        tuple(value - c for value, c in zip(member, center_u, strict=True))
        for member in members
    ]
    deviations_g = [
        tuple(value - c for value, c in zip(prediction, center_g, strict=True))
        for prediction in predictions
    ]
    gain = _whitened_gain(deviations_u, deviations_g, alpha * (count - 1))
    root = math.sqrt(alpha)
    generator = None if perturbations is None else perturbations.generator()
    updated: list[Vector] = []
    for member, prediction in zip(members, predictions, strict=True):
        innovation = tuple(
            y - g for y, g in zip(whitened_data, prediction, strict=True)
        )
        if generator is not None:
            innovation = tuple(
                value + root * generator.gauss(0.0, 1.0) for value in innovation
            )
        updated.append(
            tuple(
                value + math.fsum(k * r for k, r in zip(row, innovation, strict=True))
                for value, row in zip(member, gain, strict=True)
            )
        )
    return updated
