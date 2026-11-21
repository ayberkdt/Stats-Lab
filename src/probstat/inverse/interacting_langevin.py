"""Affine-invariant interacting Langevin sampler in gradient-free form.

An ensemble of ``J`` parameter vectors follows a Langevin diffusion that is
preconditioned by the ensemble's own covariance ``C`` and whose gradient of
the data misfit is replaced by an ensemble difference, so that no derivative
of the forward model is needed ([EQ-INV-ALDI-001], [EQ-INV-ALDI-002]). The
ensemble lives in the standard-normal coordinates ``u`` of the prior of a
:class:`~probstat.inverse.CalibrationProblem` (prior ``N(0, I)``), exactly as
in :mod:`probstat.inverse.ensemble_inversion`, and the target is the
posterior ``pi(u) ~ exp(-Phi(u))`` with
``Phi(u) = |Gamma^-1/2 (y - G(u))|^2 / 2 + |u|^2 / 2``.

Contrast with the Kalman-update methods of the same package:

* :func:`~probstat.inverse.ensemble_kalman_inversion` is an optimizer. Its
  ensemble collapses and its spread is not a posterior uncertainty.
* :func:`~probstat.inverse.ensemble_smoother_mda` is a finite sequence of
  Kalman updates. It is exact for an affine forward map as ``J -> infinity``
  and has an ``O(1/J)`` bias for a finite ensemble.
* :func:`interacting_langevin_sampler` is a stochastic dynamics that is run
  for a long time and averaged. With the finite-ensemble correction
  ``(d + 1)/J (u_j - m)`` and an affine forward map, the product of ``J``
  copies of the posterior is invariant for the continuous-time dynamics at
  any ``J > d + 1``, so a small ensemble suffices; the price is many more
  forward evaluations (``J`` per time step) and a time-discretization bias
  of the order of the step.

For a nonlinear forward map the ensemble difference is not the gradient and
the stationary law is an approximation of the posterior that the user must
check, for instance against :meth:`CalibrationProblem.sample_posterior`.
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
from probstat.linalg import Matrix, Vector

from .ensemble_inversion import _mean, _observation_variances, _spread
from .nonlinear import CalibrationProblem

_BATCHES = 10
_DIVERGENCE_BOUND = 1e3


@dataclass(frozen=True, slots=True)
class InteractingLangevinResult:
    """Time-averaged posterior estimates and history of a Langevin ensemble.

    Histories have one entry per evaluated ensemble: the prior ensemble and
    the ensemble after each time step (``steps + 1`` entries). Posterior
    estimates pool all members of the ensembles ``burn_in, ..., steps`` with
    weights proportional to the step size chosen at each ensemble
    ([EQ-INV-ALDI-003]).

    Args:
        method: ``"interacting Langevin (corrected)"`` or
            ``"ensemble Kalman sampler (uncorrected)"``.
        parameter_names: Calibration-parameter names.
        ensemble: Final members in physical coordinates ``theta``.
        standard_ensemble: Final members in the prior's standard-normal
            coordinates ``u``.
        posterior_mean: Time-averaged mean of ``theta``.
        posterior_covariance: Pooled time-averaged covariance of ``theta``.
        standard_posterior_mean: Time-averaged mean of ``u``.
        standard_posterior_covariance: Pooled time-averaged covariance of
            ``u``.
        standard_mean_standard_error: Batch-means standard error of each
            coordinate of ``standard_posterior_mean`` (ten batches of equal
            step count); ``nan`` when fewer than twenty ensembles are
            averaged. It measures the time-averaging error only, not the
            discretization bias or the error of the gradient-free
            approximation, and it is somewhat optimistic when the batches
            are short: over 40 replicates of 2000 steps on a two-parameter
            linear problem (``J`` from 4 to 8) the root-mean-square error of
            the mean was 1.1 to 1.6 reported standard errors, where ten
            independent batches would give 1.13.
        mean_history: Ensemble mean of ``theta`` per evaluated ensemble.
        spread_history: Ensemble standard deviation of each ``theta``
            coordinate per evaluated ensemble (divisor ``J - 1``).
        standard_mean_history: Ensemble mean of ``u`` per evaluated ensemble.
        standard_spread_history: Ensemble standard deviation of each ``u``
            coordinate per evaluated ensemble (divisor ``J - 1``).
        misfit_history: Whitened misfit ``|Gamma^-1/2 (y - mean_j G(u_j))|``
            of the ensemble-mean prediction per evaluated ensemble.
        step_history: Time step chosen at each evaluated ensemble; the last
            entry is computed but not taken.
        steps: Number of time steps taken.
        burn_in: Number of leading ensembles left out of the averages.
        corrected: Whether the finite-ensemble correction was applied.
        adaptive: Whether the step was adapted ([EQ-INV-ALDI-003]).
        forward_evaluations: Evaluations of the full observation operator,
            ``J (steps + 1)``.
        model_calls: Forward-model calls, ``forward_evaluations`` times the
            number of experiments.
        seed: Root seed of the random stream.

    References:
        [REF-GARBUNO-INIGO-NUSKEN-REICH-2020],
        [REF-GARBUNO-INIGO-ETAL-2020], [EQ-INV-ALDI-001],
        [EQ-INV-ALDI-003].

    """

    method: str
    parameter_names: tuple[str, ...]
    ensemble: tuple[Vector, ...]
    standard_ensemble: tuple[Vector, ...]
    posterior_mean: Vector
    posterior_covariance: Matrix
    standard_posterior_mean: Vector
    standard_posterior_covariance: Matrix
    standard_mean_standard_error: Vector
    mean_history: tuple[Vector, ...]
    spread_history: tuple[Vector, ...]
    standard_mean_history: tuple[Vector, ...]
    standard_spread_history: tuple[Vector, ...]
    misfit_history: Vector
    step_history: Vector
    steps: int
    burn_in: int
    corrected: bool
    adaptive: bool
    forward_evaluations: int
    model_calls: int
    seed: int

    @property
    def ensemble_size(self) -> int:
        """Return the number of ensemble members ``J``.

        References:
            [REF-GARBUNO-INIGO-NUSKEN-REICH-2020]. No equation is evaluated.

        """
        return len(self.ensemble)

    @property
    def sampling_time(self) -> float:
        """Return the integrated time ``sum_n h_n`` of the averaged ensembles.

        For an affine forward map the stationary dynamics relaxes at unit
        rate, so this is roughly the number of independent ensembles that
        enter the averages.

        References:
            [EQ-INV-ALDI-003], [REF-GARBUNO-INIGO-NUSKEN-REICH-2020].

        """
        return math.fsum(self.step_history[self.burn_in :])

    def mean(self) -> Mapping[str, float]:
        """Return the time-averaged posterior mean of ``theta`` keyed by name.

        Returns:
            Read-only mapping from parameter name to the posterior mean.

        Equation:
            [EQ-INV-ALDI-003].

        References:
            [REF-GARBUNO-INIGO-NUSKEN-REICH-2020].

        """
        return MappingProxyType(
            dict(zip(self.parameter_names, self.posterior_mean, strict=True))
        )

    def covariance(self) -> Matrix:
        """Return the pooled time-averaged covariance of ``theta``.

        Returns:
            Symmetric covariance matrix in parameter order.

        Equation:
            [EQ-INV-ALDI-003].

        References:
            [REF-GARBUNO-INIGO-NUSKEN-REICH-2020].

        """
        return self.posterior_covariance

    def draws(self) -> tuple[Mapping[str, float], ...]:
        """Return the final members keyed by parameter name.

        The members of one ensemble are approximately posterior draws but
        only ``J`` of them; the time averages use every ensemble after the
        burn-in.

        Returns:
            One read-only mapping per ensemble member.

        References:
            [REF-GARBUNO-INIGO-NUSKEN-REICH-2020]. No equation is evaluated.

        """
        return tuple(
            MappingProxyType(dict(zip(self.parameter_names, member, strict=True)))
            for member in self.ensemble
        )

    def summary(self) -> str:
        """Return a one-paragraph description of the run.

        Returns:
            Text with method, ensemble size, steps, integrated time,
            posterior means and standard deviations, and the forward cost.

        References:
            [REF-GARBUNO-INIGO-NUSKEN-REICH-2020]. No equation is evaluated.

        """
        estimates = ", ".join(
            f"{name} = {center:.6g} (sd {math.sqrt(max(variance, 0.0)):.3g})"
            for name, center, variance in zip(
                self.parameter_names,
                self.posterior_mean,
                (
                    self.posterior_covariance[k][k]
                    for k in range(len(self.parameter_names))
                ),
                strict=True,
            )
        )
        text = (
            f"{self.method}: {self.ensemble_size} members, {self.steps} "
            f"{'adaptive' if self.adaptive else 'fixed'} Euler-Maruyama steps, "
            f"averages over time {self.sampling_time:.4g} after {self.burn_in} "
            f"burn-in ensembles; {estimates}; {self.forward_evaluations} "
            f"forward evaluations ({self.model_calls} model calls)."
        )
        if not self.corrected:
            text += (
                " Without the finite-ensemble correction the covariance is "
                "biased low for a small ensemble."
            )
        return text


def interacting_langevin_sampler(
    problem: CalibrationProblem,
    *,
    ensemble_size: int,
    steps: int,
    burn_in: int | None = None,
    seed: int | RandomStream | None = None,
    correction: bool = True,
    step_size: float | None = None,
    step_factor: float = 0.1,
) -> InteractingLangevinResult:
    """Sample the posterior with gradient-free interacting Langevin dynamics.

    With ensemble mean ``m``, covariance
    ``C = sum_k (u_k - m)(u_k - m)' / J`` and predictions ``g_k = G(u_k)``
    with mean ``g_bar``, every member follows ([EQ-INV-ALDI-001],
    [EQ-INV-ALDI-002])

    ``du_j = -[C^ug Gamma^-1 (g_j - y) + C u_j] dt + (d + 1)/J (u_j - m) dt
    + sqrt(2) C^1/2 dW_j``,

    where ``C^ug = sum_k (u_k - m)(g_k - g_bar)' / J`` and
    ``C^1/2 = (u_1 - m, ..., u_J - m) / sqrt(J)`` is a ``d x J`` matrix, so
    the noise of each member is a combination of the ensemble deviations
    driven by ``J`` independent Brownian motions. The term ``C u_j`` is the
    exact preconditioned gradient of the prior ``N(0, I)``.

    Exactness. For an affine map ``G(u) = A u + b`` the ensemble difference
    equals ``C A' Gamma^-1 (g_j - y)``, the preconditioned gradient of the
    data misfit, without error. The correction is the divergence
    ``div_{u_j} C = (d + 1)/J (u_j - m)``; with it the probability flux of the
    Fokker-Planck equation of the whole ensemble vanishes at the product
    density ``prod_j pi(u_j)``, which is therefore invariant for every
    ensemble size at which ``C`` is non-singular. This library requires
    ``J >= d + 2`` (``J > d + 1``, the condition under which the cited
    authors state the result). Without the correction (the ensemble Kalman
    sampler of Garbuno-Inigo, Hoffmann, Li and Stuart) the stationary
    ensemble covariance is too small: for an affine map its expectation with
    divisor ``J - 1`` is ``(J - d - 2)/(J - 1)`` times the posterior
    covariance ([EQ-INV-ALDI-004]; derived in closed form for ``d = 1`` and
    confirmed by simulation for ``d = 2`` in the tests).

    Discretization. Explicit Euler-Maruyama with step ``h_n``. With
    ``step_size=None`` the step is ``step_factor / (tr C + tr C^gg)``, where
    ``C^gg`` is the ensemble covariance of the whitened predictions
    ([EQ-INV-ALDI-003]). For an affine map the trace bounds the largest
    eigenvalue of the drift matrix ``C (I + A' Gamma^-1 A)``, so the
    deterministic part of the step is a contraction for ``step_factor < 2``;
    in stationarity the step is about ``step_factor / d``. This rule is a
    heuristic: it is not an error-controlled integrator, the invariant law
    of the discrete chain differs from the posterior by ``O(h)``, and the
    averages are weighted by ``h_n`` to compensate the state dependence of
    the step to leading order. The sign and size of the remaining bias
    depend on ``J``: on a two-parameter linear problem (40 replicates of
    2000 steps) the ratio of the estimated to the exact posterior variance
    was ``0.97`` to ``1.02`` (replicate standard error ``0.02``) for
    ``step_factor = 0.1`` and ``J`` from 4 to 8, and for ``step_factor =
    0.4`` it was ``0.96`` at ``J = 4``, ``1.01`` at ``J = 5``, ``1.04`` at
    ``J = 6`` and ``1.06`` at ``J = 8`` (standard error ``0.01``). The
    unweighted average of the same runs was 12 % to 34 % too large.

    Args:
        problem: Calibration problem whose noise models are all known-scale
            Gaussian.
        ensemble_size: Number of members ``J >= d + 2``.
        steps: Number of time steps, at least one.
        burn_in: Number of leading ensembles left out of the time averages;
            defaults to ``steps // 2``. Must not exceed ``steps``.
        seed: Integer seed or stream. The prior ensemble uses the child
            stream ``"ensemble"`` (the same draw as
            :func:`ensemble_kalman_inversion`) and the Brownian increments of
            step ``n`` the child ``("noise", n)``.
        correction: Whether to add ``(d + 1)/J (u_j - m)``. ``False`` gives
            the uncorrected ensemble Kalman sampler, kept for comparison.
        step_size: Fixed step ``h > 0``; ``None`` (default) adapts the step.
            A fixed step must resolve the initial transient, whose rate for
            the prior ensemble is of the order of ``tr(A' Gamma^-1 A)``.
        step_factor: ``eta`` of the adaptive rule, in ``(0, 2)``. The default
            0.1 is a library choice.

    Returns:
        Time-averaged posterior estimates with the ensemble history.

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem` or
            ``correction`` is not a bool.
        ValueError: For unknown-noise or Student-t problems,
            ``J < d + 2``, ``steps < 1``, ``burn_in > steps``, a
            non-positive ``step_size``, ``step_factor`` outside ``(0, 2)``,
            or an ensemble that degenerated or diverged (a coordinate
            beyond 1000 prior standard deviations).

    Equation:
        [EQ-INV-ALDI-001], [EQ-INV-ALDI-002], [EQ-INV-ALDI-003],
        [EQ-INV-ALDI-004].

    References:
        [REF-GARBUNO-INIGO-NUSKEN-REICH-2020],
        [REF-GARBUNO-INIGO-ETAL-2020].

    """
    variances = _observation_variances(problem)
    if not isinstance(correction, bool):
        raise TypeError("correction must be a bool")
    count = require_non_negative_integer(ensemble_size, name="ensemble_size")
    dimension = problem.model.dimension
    if count < dimension + 2:
        raise ValueError(
            f"ensemble_size must be at least d + 2 = {dimension + 2}: the "
            "ensemble covariance that preconditions the dynamics is singular "
            "for fewer members"
        )
    total = require_non_negative_integer(steps, name="steps")
    if total == 0:
        raise ValueError("steps must be at least one")
    if burn_in is None:
        discard = total // 2
    else:
        discard = require_non_negative_integer(burn_in, name="burn_in")
        if discard > total:
            raise ValueError("burn_in must not exceed steps")
    fixed = math.nan
    if step_size is not None:
        fixed = require_finite_real(step_size, name="step_size")
        if fixed <= 0.0:
            raise ValueError("step_size must be positive")
    factor = require_finite_real(step_factor, name="step_factor")
    if not 0.0 < factor < 2.0:
        raise ValueError(
            "step_factor must lie in (0, 2): larger values make the explicit "
            "Euler step of the drift unstable"
        )

    stream = resolve_stream(seed)
    inputs = problem.model.inputs
    scales = tuple(math.sqrt(value) for value in variances)
    whitened_data = tuple(
        value / scale for value, scale in zip(problem.observed, scales, strict=True)
    )
    generator = stream.child("ensemble").generator()
    members: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    ]
    kappa = (dimension + 1) / count if correction else 0.0

    mean_history: list[Vector] = []
    spread_history: list[Vector] = []
    standard_mean_history: list[Vector] = []
    standard_spread_history: list[Vector] = []
    misfit_history: list[float] = []
    step_history: list[float] = []
    standard_moments = _WeightedMoments(dimension)
    physical_moments = _WeightedMoments(dimension)
    physical: list[Vector] = []

    for index in range(total + 1):
        if not all(
            math.isfinite(value) and abs(value) <= _DIVERGENCE_BOUND
            for member in members
            for value in member
        ):
            raise ValueError(
                f"the ensemble diverged at step {index} (a coordinate left "
                f"|u| <= {_DIVERGENCE_BOUND:g} prior standard deviations); "
                "reduce step_factor or step_size"
            )
        physical = [inputs.from_standard_normal(member) for member in members]
        predictions = [
            tuple(
                value / scale
                for value, scale in zip(problem.predictions(theta), scales, strict=True)
            )
            for theta in physical
        ]
        center_u = _mean(members)
        center_theta = _mean(physical)
        center_g = _mean(predictions)
        deviations_u = [
            tuple(value - c for value, c in zip(member, center_u, strict=True))
            for member in members
        ]
        deviations_g = [
            tuple(value - c for value, c in zip(prediction, center_g, strict=True))
            for prediction in predictions
        ]
        covariance = _second_moment(deviations_u, deviations_u)
        trace_u = math.fsum(covariance[k][k] for k in range(dimension))
        trace_g = math.fsum(value * value for row in deviations_g for value in row)
        trace_g /= count
        if trace_u <= 0.0:
            raise ValueError("the ensemble degenerated to a single point")
        step = factor / (trace_u + trace_g) if step_size is None else fixed

        standard_mean_history.append(center_u)
        standard_spread_history.append(_spread(members, center_u))
        mean_history.append(center_theta)
        spread_history.append(_spread(physical, center_theta))
        misfit_history.append(
            math.sqrt(
                math.fsum(
                    (y - g) ** 2 for y, g in zip(whitened_data, center_g, strict=True)
                )
            )
        )
        step_history.append(step)
        if index >= discard:
            standard_moments.add(members, center_u, step)
            physical_moments.add(physical, center_theta, step)
        if index == total:
            break
        members = _langevin_step(
            members,
            deviations_u,
            predictions,
            deviations_g,
            covariance,
            whitened_data,
            step,
            kappa,
            stream.child("noise", index),
        )

    experiments = len(problem.experiments)
    evaluations = count * (total + 1)
    return InteractingLangevinResult(
        method=(
            "interacting Langevin (corrected)"
            if correction
            else "ensemble Kalman sampler (uncorrected)"
        ),
        parameter_names=problem.parameter_names,
        ensemble=tuple(physical),
        standard_ensemble=tuple(members),
        posterior_mean=physical_moments.mean(),
        posterior_covariance=physical_moments.covariance(),
        standard_posterior_mean=standard_moments.mean(),
        standard_posterior_covariance=standard_moments.covariance(),
        standard_mean_standard_error=standard_moments.standard_error(),
        mean_history=tuple(mean_history),
        spread_history=tuple(spread_history),
        standard_mean_history=tuple(standard_mean_history),
        standard_spread_history=tuple(standard_spread_history),
        misfit_history=tuple(misfit_history),
        step_history=tuple(step_history),
        steps=total,
        burn_in=discard,
        corrected=correction,
        adaptive=step_size is None,
        forward_evaluations=evaluations,
        model_calls=evaluations * experiments,
        seed=stream.seed,
    )


class _WeightedMoments:
    """Accumulate step-weighted first and second moments of ensembles."""

    def __init__(self, dimension: int) -> None:
        """Start empty accumulators for vectors of length ``dimension``.

        References:
            [EQ-INV-ALDI-003].

        """
        self._dimension = dimension
        self._weights: list[float] = []
        self._centers: list[Vector] = []
        self._seconds: list[list[float]] = []

    def add(self, members: Sequence[Vector], center: Vector, weight: float) -> None:
        """Record one ensemble with time weight ``weight``.

        References:
            [EQ-INV-ALDI-003].

        """
        size = self._dimension
        count = len(members)
        self._weights.append(weight)
        self._centers.append(center)
        self._seconds.append(
            [
                math.fsum(member[i] * member[j] for member in members) / count
                for i in range(size)
                for j in range(i, size)
            ]
        )

    def mean(self) -> Vector:
        """Return ``sum_n h_n m_n / sum_n h_n``.

        References:
            [EQ-INV-ALDI-003].

        """
        total = math.fsum(self._weights)
        return tuple(
            math.fsum(
                w * center[k]
                for w, center in zip(self._weights, self._centers, strict=True)
            )
            / total
            for k in range(self._dimension)
        )

    def covariance(self) -> Matrix:
        """Return the pooled second moment about the weighted mean.

        References:
            [EQ-INV-ALDI-003].

        """
        size = self._dimension
        total = math.fsum(self._weights)
        center = self.mean()
        rows = [[0.0] * size for _ in range(size)]
        position = 0
        for i in range(size):
            for j in range(i, size):
                value = (
                    math.fsum(
                        w * second[position]
                        for w, second in zip(self._weights, self._seconds, strict=True)
                    )
                    / total
                    - center[i] * center[j]
                )
                rows[i][j] = value
                rows[j][i] = value
                position += 1
        return tuple(tuple(row) for row in rows)

    def standard_error(self) -> Vector:
        """Return the batch-means standard error of the weighted mean.

        The recorded ensembles are split into ten consecutive batches of
        equal count; the standard error is the weighted standard deviation
        of the batch means divided by the square root of the number of
        batches.

        References:
            [EQ-INV-ALDI-003], [EQ-MC-SE-001].

        """
        count = len(self._weights)
        if count < 2 * _BATCHES:
            return tuple(math.nan for _ in range(self._dimension))
        length = count // _BATCHES
        center = self.mean()
        total = math.fsum(self._weights[: length * _BATCHES])
        errors: list[float] = []
        for k in range(self._dimension):
            accumulated = 0.0
            for batch in range(_BATCHES):
                rows = range(batch * length, (batch + 1) * length)
                weight = math.fsum(self._weights[n] for n in rows)
                value = (
                    math.fsum(self._weights[n] * self._centers[n][k] for n in rows)
                    / weight
                )
                accumulated += (weight / total) * (value - center[k]) ** 2
            errors.append(math.sqrt(accumulated / (_BATCHES - 1)))
        return tuple(errors)


def _second_moment(left: Sequence[Vector], right: Sequence[Vector]) -> Matrix:
    """Return ``sum_k left_k right_k' / J`` for ensemble deviations.

    References:
        [EQ-INV-ALDI-001] (covariances with divisor ``J``).

    """
    count = len(left)
    return tuple(
        tuple(
            math.fsum(a[i] * b[j] for a, b in zip(left, right, strict=True)) / count
            for j in range(len(right[0]))
        )
        for i in range(len(left[0]))
    )


def _langevin_step(
    members: Sequence[Vector],
    deviations_u: Sequence[Vector],
    predictions: Sequence[Vector],
    deviations_g: Sequence[Vector],
    covariance: Matrix,
    whitened_data: Vector,
    step: float,
    kappa: float,
    noise: RandomStream,
) -> list[Vector]:
    """Apply one Euler-Maruyama step of the interacting Langevin dynamics.

    ``u_j <- u_j - h [C^ug r_j + C u_j - kappa (u_j - m)]
    + sqrt(2 h / J) sum_k (u_k - m) xi_jk`` with whitened residuals
    ``r_j = Gamma^-1/2 (g_j - y)``, the whitened cross-covariance ``C^ug``
    (divisor ``J``), and ``xi_jk ~ N(0, 1)`` drawn member by member.

    References:
        [EQ-INV-ALDI-001], [EQ-INV-ALDI-002], [EQ-INV-ALDI-003],
        [REF-GARBUNO-INIGO-NUSKEN-REICH-2020].

    """
    count = len(members)
    dimension = len(members[0])
    cross = _second_moment(deviations_u, deviations_g)
    amplitude = math.sqrt(2.0 * step / count)
    generator = noise.generator()
    updated: list[Vector] = []
    for member, deviation, prediction in zip(
        members, deviations_u, predictions, strict=True
    ):
        residual = tuple(g - y for g, y in zip(prediction, whitened_data, strict=True))
        draws = [generator.gauss(0.0, 1.0) for _ in range(count)]
        updated.append(
            tuple(
                member[i]
                - step
                * (
                    math.fsum(c * r for c, r in zip(cross[i], residual, strict=True))
                    + math.fsum(
                        c * u for c, u in zip(covariance[i], member, strict=True)
                    )
                    - kappa * deviation[i]
                )
                + amplitude
                * math.fsum(
                    other[i] * xi for other, xi in zip(deviations_u, draws, strict=True)
                )
                for i in range(dimension)
            )
        )
    return updated
