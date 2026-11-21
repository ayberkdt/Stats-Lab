"""Consensus-based sampling and optimization for calibration problems.

An ensemble in the standard-normal coordinates ``u`` of the prior of a
:class:`~probstat.inverse.CalibrationProblem` is reweighted by
``exp(-beta Phi(u))`` with the posterior potential
``Phi(u) = |Gamma^-1/2 (y - G(u))|^2 / 2 + |u|^2 / 2``, and every member is
relaxed towards the weighted mean and refreshed with Gaussian noise whose
covariance is the weighted covariance divided by ``lambda``
([EQ-INV-CBS-001], [EQ-INV-CBS-002]). Only values of ``Phi`` are needed: no
derivative and, unlike the Kalman updates of
:mod:`probstat.inverse.ensemble_inversion`, no covariance between parameters
and predictions.

Two modes share the iteration:

* ``lambda = 1 / (1 + beta)`` (sampling). A Gaussian target ``N(mu, Sigma)``
  is a fixed point of the mean-field iteration ([EQ-INV-CBS-003]): the
  ensemble neither collapses nor spreads. For a non-Gaussian posterior the
  ensemble converges to a Gaussian-like approximation, not to the posterior.
* ``lambda = 1`` (optimization). The covariance contracts and the ensemble
  concentrates near the minimizer of ``Phi``, the MAP point in ``u``.

Contrast: :func:`~probstat.inverse.ensemble_kalman_inversion` minimizes the
data misfit only and collapses within the span of its initial ensemble;
:func:`~probstat.inverse.ensemble_smoother_mda` reaches an approximate
posterior ensemble in a fixed, small number of updates. Consensus-based
sampling needs more iterations and a larger ensemble, because the
self-normalized weights use only a fraction of the members and the weighted
covariance carries an ``O(1/J_eff)`` bias that the iteration feeds back.
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
from probstat.linalg import Matrix, Vector, cholesky

from .ensemble_inversion import _covariance, _mean, _observation_variances, _spread
from .nonlinear import CalibrationProblem

_MODES = ("sampling", "optimization")
_BISECTIONS = 60


@dataclass(frozen=True, slots=True)
class ConsensusSamplingResult:
    """Final ensemble and history of a consensus-based iteration.

    Histories of the ensemble have one entry per evaluated ensemble
    (``iterations + 1``); ``inverse_temperature_history`` and
    ``effective_size_history`` have one entry per update.

    Args:
        mode: ``"sampling"`` or ``"optimization"``.
        parameter_names: Calibration-parameter names.
        ensemble: Final members in physical coordinates ``theta``.
        standard_ensemble: Final members in the coordinates ``u``.
        mean_history: Ensemble mean of ``theta`` per evaluated ensemble.
        spread_history: Ensemble standard deviation of each ``theta``
            coordinate per evaluated ensemble (divisor ``J - 1``).
        standard_mean_history: Ensemble mean of ``u`` per evaluated ensemble.
        standard_spread_history: Ensemble standard deviation of each ``u``
            coordinate per evaluated ensemble.
        misfit_history: Whitened misfit ``|Gamma^-1/2 (y - mean_j G(u_j))|``
            of the ensemble-mean prediction per evaluated ensemble.
        potential_history: Smallest ``Phi(u_j)`` per evaluated ensemble.
        inverse_temperature_history: ``beta_n`` used by each update.
        effective_size_history: ``J_eff = 1 / sum_j w_j^2`` of each update.
        best_standard_point: Evaluated member with the smallest ``Phi`` over
            the whole run, in the coordinates ``u``.
        best_point: The same member in physical coordinates.
        best_potential: Its value of ``Phi``.
        relaxation: ``alpha`` of the update.
        iterations: Number of updates performed.
        forward_evaluations: Evaluations of the full observation operator,
            ``J (iterations + 1)``.
        model_calls: Forward-model calls, ``forward_evaluations`` times the
            number of experiments.
        seed: Root seed of the random stream.

    References:
        [REF-CARRILLO-ETAL-2022], [EQ-INV-CBS-001], [EQ-INV-CBS-002].

    """

    mode: str
    parameter_names: tuple[str, ...]
    ensemble: tuple[Vector, ...]
    standard_ensemble: tuple[Vector, ...]
    mean_history: tuple[Vector, ...]
    spread_history: tuple[Vector, ...]
    standard_mean_history: tuple[Vector, ...]
    standard_spread_history: tuple[Vector, ...]
    misfit_history: Vector
    potential_history: Vector
    inverse_temperature_history: Vector
    effective_size_history: Vector
    best_standard_point: Vector
    best_point: Vector
    best_potential: float
    relaxation: float
    iterations: int
    forward_evaluations: int
    model_calls: int
    seed: int

    @property
    def ensemble_size(self) -> int:
        """Return the number of ensemble members ``J``.

        References:
            [REF-CARRILLO-ETAL-2022]. No equation is evaluated.

        """
        return len(self.ensemble)

    def mean(self) -> Mapping[str, float]:
        """Return the final ensemble mean of ``theta`` keyed by name.

        Returns:
            Read-only mapping from parameter name to the ensemble mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-CARRILLO-ETAL-2022].

        """
        return MappingProxyType(
            dict(zip(self.parameter_names, self.mean_history[-1], strict=True))
        )

    def covariance(self) -> Matrix:
        """Return the final ensemble covariance of ``theta`` (divisor ``J - 1``).

        In sampling mode this estimates the posterior covariance (exactly
        targeted only for a Gaussian posterior); in optimization mode it is
        the spread of a contracting ensemble, not an uncertainty.

        Returns:
            Symmetric sample covariance matrix in parameter order.

        Equation:
            [EQ-INV-CBS-003].

        References:
            [REF-CARRILLO-ETAL-2022].

        """
        return _covariance(self.ensemble)

    def standard_covariance(self) -> Matrix:
        """Return the final ensemble covariance of ``u`` (divisor ``J - 1``).

        Returns:
            Symmetric sample covariance matrix in parameter order.

        Equation:
            [EQ-INV-CBS-003].

        References:
            [REF-CARRILLO-ETAL-2022].

        """
        return _covariance(self.standard_ensemble)

    def draws(self) -> tuple[Mapping[str, float], ...]:
        """Return the final members keyed by parameter name.

        Returns:
            One read-only mapping per ensemble member.

        References:
            [REF-CARRILLO-ETAL-2022]. No equation is evaluated.

        """
        return tuple(
            MappingProxyType(dict(zip(self.parameter_names, member, strict=True)))
            for member in self.ensemble
        )

    def summary(self) -> str:
        """Return a one-paragraph description of the run.

        Returns:
            Text with mode, ensemble size, iterations, effective sample
            size, final means and spreads, and the forward cost.

        References:
            [REF-CARRILLO-ETAL-2022]. No equation is evaluated.

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
            f"Consensus-based {self.mode}: {self.ensemble_size} members, "
            f"{self.iterations} updates, last beta "
            f"{self.inverse_temperature_history[-1]:.4g} with effective size "
            f"{self.effective_size_history[-1]:.4g}; {estimates}; "
            f"{self.forward_evaluations} forward evaluations "
            f"({self.model_calls} model calls)."
        )
        if self.mode == "optimization":
            text += (
                " The spread is that of a contracting optimizer ensemble, not "
                "a posterior uncertainty."
            )
        else:
            text += (
                " The ensemble targets the posterior exactly only if it is Gaussian."
            )
        return text


def consensus_based_sampling(
    problem: CalibrationProblem,
    *,
    ensemble_size: int,
    iterations: int,
    mode: str = "sampling",
    inverse_temperature: float = 1.0,
    relaxation: float = 0.5,
    minimum_effective_fraction: float = 0.5,
    seed: int | RandomStream | None = None,
) -> ConsensusSamplingResult:
    """Sample or minimize the posterior potential by consensus-based updates.

    Draws ``u_j ~ N(0, I)`` and repeats, with weights
    ``w_j ~ exp(-beta_n Phi(u_j))`` normalized to one
    ([EQ-INV-CBS-001], [EQ-INV-CBS-002]):

    ``m_b = sum_j w_j u_j``, ``C_b = sum_j w_j (u_j - m_b)(u_j - m_b)'``,
    ``u_j <- m_b + alpha (u_j - m_b) + sqrt((1 - alpha^2) / lambda)
    C_b^1/2 xi_j``, ``xi_j ~ N(0, I)``,

    with ``lambda = 1 / (1 + beta_n)`` in sampling mode and ``lambda = 1`` in
    optimization mode. ``Phi(u) = |Gamma^-1/2 (y - G(u))|^2 / 2 + |u|^2 / 2``.

    Gaussian fixed point ([EQ-INV-CBS-003]). If the ensemble law is
    ``N(m, C)`` and ``Phi`` is quadratic with minimizer ``mu`` and Hessian
    ``Sigma^-1`` (affine forward map), the weighted law is Gaussian with
    ``C_b^-1 = C^-1 + beta Sigma^-1``, and one update gives mean
    ``m_b + alpha (m - m_b)`` and covariance
    ``alpha^2 C + (1 - alpha^2) C_b / lambda``. With
    ``lambda = 1 / (1 + beta)`` the pair ``(mu, Sigma)`` is a fixed point for
    every ``beta > 0`` and ``alpha`` in ``[0, 1)``; with ``lambda = 1`` the
    only fixed covariance is zero and the mean tends to ``mu``. This is a
    statement about the mean-field (infinite-ensemble) iteration. With ``J``
    members the weighted covariance is biased low by about ``1 / J_eff``,
    which the iteration amplifies to a relative covariance deficit of about
    ``(1 + beta) / (beta J_eff)``.

    Effective sample size. Early weights can be carried by a few members
    when the posterior is much narrower than the prior. Each update
    therefore uses the largest ``beta_n <= inverse_temperature`` with
    ``J_eff(beta_n) >= minimum_effective_fraction * J`` (bisection). The
    fixed point above holds for any ``beta``, so this choice changes the
    speed, not the target of the mean-field iteration; it is a library
    heuristic.

    Args:
        problem: Calibration problem whose noise models are all known-scale
            Gaussian.
        ensemble_size: Number of members ``J >= d + 2``.
        iterations: Number of updates, at least one.
        mode: ``"sampling"`` (default) or ``"optimization"``.
        inverse_temperature: Upper bound ``beta > 0`` of the weights'
            inverse temperature.
        relaxation: ``alpha`` in ``[0, 1)``; zero redraws the ensemble from
            the fitted Gaussian at every update.
        minimum_effective_fraction: Target lower bound of ``J_eff / J`` in
            ``(0, 1)``.
        seed: Integer seed or stream. The prior ensemble uses the child
            stream ``"ensemble"`` and the noise of update ``n`` the child
            ``("noise", n)``.

    Returns:
        Final ensemble with its iteration history.

    Raises:
        TypeError: If ``problem`` is not a :class:`CalibrationProblem`.
        ValueError: For unknown-noise or Student-t problems, ``J < d + 2``,
            ``iterations < 1``, an unknown ``mode``, a non-positive
            ``inverse_temperature``, ``relaxation`` outside ``[0, 1)``,
            ``minimum_effective_fraction`` outside ``(0, 1)``, or a weighted
            covariance that is not positive definite.

    Equation:
        [EQ-INV-CBS-001], [EQ-INV-CBS-002], [EQ-INV-CBS-003].

    References:
        [REF-CARRILLO-ETAL-2022].

    """
    variances = _observation_variances(problem)
    if mode not in _MODES:
        raise ValueError("mode must be 'sampling' or 'optimization'")
    count = require_non_negative_integer(ensemble_size, name="ensemble_size")
    dimension = problem.model.dimension
    if count < dimension + 2:
        raise ValueError(
            f"ensemble_size must be at least d + 2 = {dimension + 2} for a "
            "non-singular weighted covariance"
        )
    updates = require_non_negative_integer(iterations, name="iterations")
    if updates == 0:
        raise ValueError("iterations must be at least one")
    beta_max = require_finite_real(inverse_temperature, name="inverse_temperature")
    if beta_max <= 0.0:
        raise ValueError("inverse_temperature must be positive")
    alpha = require_finite_real(relaxation, name="relaxation")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("relaxation must lie in [0, 1)")
    fraction = require_finite_real(
        minimum_effective_fraction, name="minimum_effective_fraction"
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("minimum_effective_fraction must lie in (0, 1)")

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

    mean_history: list[Vector] = []
    spread_history: list[Vector] = []
    standard_mean_history: list[Vector] = []
    standard_spread_history: list[Vector] = []
    misfit_history: list[float] = []
    potential_history: list[float] = []
    beta_history: list[float] = []
    size_history: list[float] = []
    best: tuple[float, Vector, Vector] | None = None
    physical: list[Vector] = []

    for index in range(updates + 1):
        physical = [inputs.from_standard_normal(member) for member in members]
        predictions = [
            tuple(
                value / scale
                for value, scale in zip(problem.predictions(theta), scales, strict=True)
            )
            for theta in physical
        ]
        potentials = [
            0.5
            * (
                math.fsum(
                    (y - g) ** 2 for y, g in zip(whitened_data, prediction, strict=True)
                )
                + math.fsum(value * value for value in member)
            )
            for member, prediction in zip(members, predictions, strict=True)
        ]
        lowest = min(range(count), key=potentials.__getitem__)
        if best is None or potentials[lowest] < best[0]:
            best = (potentials[lowest], members[lowest], physical[lowest])
        center_u = _mean(members)
        center_theta = _mean(physical)
        center_g = _mean(predictions)
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
        potential_history.append(potentials[lowest])
        if index == updates:
            break
        beta, weights, effective = _tempered_weights(
            potentials, beta_max, fraction * count
        )
        beta_history.append(beta)
        size_history.append(effective)
        scale = (1.0 + beta) if mode == "sampling" else 1.0
        members = _consensus_update(
            members, weights, alpha, scale, stream.child("noise", index)
        )

    assert best is not None  # at least one ensemble was evaluated
    evaluations = count * (updates + 1)
    return ConsensusSamplingResult(
        mode=mode,
        parameter_names=problem.parameter_names,
        ensemble=tuple(physical),
        standard_ensemble=tuple(members),
        mean_history=tuple(mean_history),
        spread_history=tuple(spread_history),
        standard_mean_history=tuple(standard_mean_history),
        standard_spread_history=tuple(standard_spread_history),
        misfit_history=tuple(misfit_history),
        potential_history=tuple(potential_history),
        inverse_temperature_history=tuple(beta_history),
        effective_size_history=tuple(size_history),
        best_standard_point=best[1],
        best_point=best[2],
        best_potential=best[0],
        relaxation=alpha,
        iterations=updates,
        forward_evaluations=evaluations,
        model_calls=evaluations * len(problem.experiments),
        seed=stream.seed,
    )


def _weights(potentials: Sequence[float], beta: float) -> tuple[Vector, float]:
    """Return normalized weights ``exp(-beta Phi_j)`` and ``1 / sum w_j^2``.

    The smallest potential is subtracted before exponentiation, which
    leaves the normalized weights unchanged and avoids underflow.

    References:
        [EQ-INV-CBS-001].

    """
    lowest = min(potentials)
    raw = [math.exp(-beta * (value - lowest)) for value in potentials]
    total = math.fsum(raw)
    weights = tuple(value / total for value in raw)
    return weights, 1.0 / math.fsum(w * w for w in weights)


def _tempered_weights(
    potentials: Sequence[float], beta_max: float, target: float
) -> tuple[float, Vector, float]:
    """Return the largest ``beta <= beta_max`` with ``J_eff >= target``.

    ``J_eff(beta)`` decreases from ``J`` at ``beta = 0``; if it is at least
    ``target`` at ``beta_max`` that value is used, otherwise the crossing is
    located by bisection.

    Returns:
        ``(beta, weights, J_eff)``.

    References:
        [EQ-INV-CBS-001] (weights), [EQ-INV-CBS-003] (the Gaussian fixed
        point does not depend on ``beta``).

    """
    weights, effective = _weights(potentials, beta_max)
    if effective >= target:
        return beta_max, weights, effective
    low, high = 0.0, beta_max
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        if _weights(potentials, middle)[1] >= target:
            low = middle
        else:
            high = middle
    if low <= 0.0:
        raise ValueError(
            "no positive inverse temperature reaches the requested effective "
            "sample size; the potentials of the ensemble are too disparate"
        )
    weights, effective = _weights(potentials, low)
    return low, weights, effective


def _consensus_update(
    members: Sequence[Vector],
    weights: Vector,
    alpha: float,
    scale: float,
    noise: RandomStream,
) -> list[Vector]:
    """Apply one consensus-based update to every member.

    ``u_j <- m_b + alpha (u_j - m_b) + sqrt((1 - alpha^2) scale) L xi_j``
    with ``L L' = C_b`` the Cholesky factor of the weighted covariance and
    ``scale = 1 / lambda``.

    Raises:
        ValueError: If ``C_b`` is not positive definite (the weights are
            carried by too few distinct members).

    References:
        [EQ-INV-CBS-001], [EQ-INV-CBS-002], [REF-CARRILLO-ETAL-2022].

    """
    dimension = len(members[0])
    center = tuple(
        math.fsum(w * member[k] for w, member in zip(weights, members, strict=True))
        for k in range(dimension)
    )
    covariance = tuple(
        tuple(
            math.fsum(
                w * (member[i] - center[i]) * (member[j] - center[j])
                for w, member in zip(weights, members, strict=True)
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    try:
        lower = cholesky(covariance)
    except ValueError as error:
        raise ValueError(
            "the weighted ensemble covariance is not positive definite; the "
            "ensemble has collapsed numerically. Use a larger ensemble, fewer "
            "iterations in optimization mode, or a larger "
            "minimum_effective_fraction"
        ) from error
    amplitude = math.sqrt((1.0 - alpha * alpha) * scale)
    generator = noise.generator()
    updated: list[Vector] = []
    for member in members:
        draws = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
        updated.append(
            tuple(
                center[i]
                + alpha * (member[i] - center[i])
                + amplitude * math.fsum(lower[i][k] * draws[k] for k in range(i + 1))
                for i in range(dimension)
            )
        )
    return updated
