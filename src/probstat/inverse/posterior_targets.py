"""Prior-whitened posterior targets and sampler adapters for inverse problems.

This module does not contain a sampler. It states one explicit *target
contract* for a Bayesian inverse problem with a Gaussian prior and a
differentiable forward map, and adapts that target to the samplers the library
already has: :func:`probstat.inference.preconditioned_crank_nicolson`,
:func:`probstat.inference.no_u_turn_sampler` and
:func:`probstat.inference.hamiltonian_monte_carlo`. The simulation-based
calibration helper of :mod:`probstat.inference.sbc` is reused for the
linear-Gaussian oracle check.

Coordinates and densities
    With the prior ``x ~ N(mu, P P')`` (``P`` an ``n x r`` factor) and the
    noise ``eps ~ N(0, Gamma)``, ``Gamma = C C'``, ``W = C^-1``, the target
    works in the whitened coordinate ``z in R^r`` defined by
    ``x = mu + P z`` ([EQ-INV-TGT-001]). The prior of ``z`` is exactly
    ``N(0, I)``. The map ``z -> x`` is affine with the constant Jacobian
    ``P``, so no coordinate Jacobian depends on ``z``: the density *of ``z``*
    is ``p(y | mu + P z) N(z; 0, I)``, and a change of variables to ``x``
    would only add the constant ``log |det P|`` (when ``P`` is square), which
    never changes a sampler. All quantities here are densities in ``z``.

    - ``log_likelihood(z) = -1/2 ||W (G(mu + P z) - y)||^2`` omits the
      normalizing constant ``-m/2 log 2 pi - 1/2 log det Gamma``, which is
      stored in :attr:`WhitenedGaussianPriorTarget.log_likelihood_constant`.
    - ``log_prior(z) = -1/2 ||z||^2 - r/2 log 2 pi`` includes its constant.
    - ``log_density(z) = log_likelihood(z) + log_prior(z)`` is the log
      posterior density of ``z`` up to the likelihood constant.
    - ``gradient(z) = -P' J' W' W (G(x) - y) - z`` uses one cached forward
      value and one adjoint product (``vjp``) ([EQ-INV-TGT-002]).

Do not add the prior twice
    The preconditioned Crank-Nicolson (pCN) proposal is reversible with
    respect to ``N(0, I)``, so the prior cancels in the acceptance ratio and
    pCN must be given :meth:`~WhitenedGaussianPriorTarget.log_likelihood`
    only ([EQ-INV-TGT-003]). Gradient-based samplers (HMC, NUTS) need the full
    :meth:`~WhitenedGaussianPriorTarget.log_density`. Passing ``log_density``
    to pCN would count the prior twice and silently sample the wrong law.

Failed forward evaluations
    A non-finite forward value makes ``log_likelihood`` return ``-inf`` (zero
    likelihood, rejected by every sampler) and ``gradient`` raise
    ``ValueError`` (which HMC and NUTS treat as leaving the support). Nothing
    is clipped or repaired.

Simulation-based calibration
    :func:`simulation_based_calibration_linear_gaussian` checks, for the
    linear-Gaussian case where the posterior is known exactly, that the rank
    statistic of Talts et al. is uniform ([EQ-INV-TGT-004]). It validates a
    *sampler* under the *simulated* model, not the real-world model: a
    calibrated algorithm combined with a wrong forward model or prior still
    passes. p-values are screening heuristics, not proofs.

Experimental (added in 0.28).

References:
    [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-STUART-2010],
    [REF-TALTS-ETAL-2018], [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.inference import (
    HamiltonianResult,
    MCMCResult,
    hamiltonian_monte_carlo,
    no_u_turn_sampler,
    preconditioned_crank_nicolson,
    simulation_based_calibration,
)
from probstat.linalg import (
    BlockDiagonalCovariance,
    DenseCovariance,
    DiagonalCovariance,
    LinearOperator,
    Vector,
    as_vector,
    require_operator,
)

from .gauss_newton import DifferentiableForward
from .linear_gaussian import linear_gaussian_posterior

_LOG_TWO_PI = math.log(2.0 * math.pi)
_COVARIANCE_TYPES = (DiagonalCovariance, DenseCovariance, BlockDiagonalCovariance)
CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance

SBC_CAVEAT = (
    "Simulation-based calibration is performed under the simulated model "
    "(the declared prior and likelihood). It checks the sampler, not the "
    "real-world model: a calibrated algorithm with a wrong forward model, "
    "prior or noise law still passes. p-values are screening heuristics."
)


def _require_covariance(noise: object, size: int) -> CovarianceFactor:
    """Return ``noise`` after checking type and dimension.

    Raises:
        TypeError: If ``noise`` is not a covariance factor.
        ValueError: If its dimension differs from ``size``.

    References:
        [EQ-INV-TGT-001].

    """
    if not isinstance(noise, _COVARIANCE_TYPES):
        raise TypeError("noise must be a probstat.linalg covariance factor")
    if noise.dimension != size:
        raise ValueError(f"noise has dimension {noise.dimension}, expected {size}")
    return noise


@dataclass(frozen=True, slots=True)
class TargetCounts:
    """Exact work done by a :class:`WhitenedGaussianPriorTarget`.

    Attributes:
        log_likelihood: Calls of ``log_likelihood`` (``log_density`` counts
            one).
        gradient: Calls of ``gradient``.
        forward: Forward evaluations ``G(x)`` actually performed (a gradient
            call at the point of the preceding density call reuses its value).
        vjp: Adjoint products ``J(x)' w``.

    References:
        [EQ-INV-TGT-002].

    """

    log_likelihood: int
    gradient: int
    forward: int
    vjp: int


class WhitenedGaussianPriorTarget:
    """Posterior of ``y = G(x) + eps`` in prior-whitened coordinates ``z``.

    With ``x = mu + P z`` the prior of ``z`` is ``N(0, I)`` and the
    log-posterior of ``z`` is ``-1/2 ||W (G(mu + P z) - y)||^2 - 1/2 ||z||^2``
    up to constants [EQ-INV-TGT-001]. The densities are in ``z`` coordinates;
    since the map is affine with constant Jacobian ``P`` no ``z``-dependent
    Jacobian is missing, only a constant. See the module docstring for what
    each method includes and for the rule that pCN takes ``log_likelihood``
    only.

    One forward value is cached (the last ``z``), so ``log_density(z)``
    followed by ``gradient(z)`` costs one forward evaluation and one ``vjp``.
    The object is stateful (cache, counters) and not thread safe.

    References:
        [EQ-INV-TGT-001], [EQ-INV-TGT-002],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-STUART-2010].

    """

    __slots__ = (
        "_cache",
        "_counts",
        "_data",
        "_mean",
        "_model",
        "_noise",
        "_prior_factor",
        "log_likelihood_constant",
    )

    def __init__(
        self,
        model: DifferentiableForward,
        data: Sequence[float],
        *,
        noise: object,
        prior_mean: Sequence[float] | None = None,
        prior_factor: object,
    ) -> None:
        """Validate and store the problem.

        Args:
            model: Differentiable forward map ``G`` (``value`` and ``vjp``
                are used).
            data: Observation ``y`` of length ``m``.
            noise: Covariance factor of ``Gamma`` (dimension ``m``).
            prior_mean: ``mu`` of length ``n`` (default zeros).
            prior_factor: ``P`` (``n x r``) with prior covariance ``P P'``: a
                ``LinearOperator``, a covariance factor's operator or a dense
                matrix.

        Raises:
            TypeError: If ``model`` is not a differentiable forward map or an
                argument has the wrong type.
            ValueError: If dimensions disagree or an input is not finite.

        References:
            [EQ-INV-TGT-001].

        """
        if not isinstance(model, DifferentiableForward):
            raise TypeError("model must implement DifferentiableForward")
        n = model.input_dimension
        m = model.output_dimension
        y = as_vector(data, name="data")
        if len(y) != m:
            raise ValueError(f"data has length {len(y)}, expected {m}")
        covariance = _require_covariance(noise, m)
        if prior_mean is None:
            mu: Vector = (0.0,) * n
        else:
            mu = as_vector(prior_mean, name="prior_mean")
            if len(mu) != n:
                raise ValueError(f"prior_mean has length {len(mu)}, expected {n}")
        factor: LinearOperator
        if isinstance(prior_factor, _COVARIANCE_TYPES):
            factor = prior_factor.factor_operator()
        else:
            factor = require_operator(prior_factor, name="prior_factor")
        rows, columns = factor.shape
        if rows != n:
            raise ValueError(f"prior_factor has {rows} rows, expected {n}")
        if columns < 1:
            raise ValueError("prior_factor must have at least one column")
        self._model = model
        self._data = y
        self._noise = covariance
        self._mean = mu
        self._prior_factor: LinearOperator = factor
        self._cache: tuple[Vector, Vector, Vector] | None = None
        self._counts = [0, 0, 0, 0]
        self.log_likelihood_constant = -0.5 * (m * _LOG_TWO_PI + covariance.log_det())

    @property
    def dimension(self) -> int:
        """Return ``r``, the number of whitened coordinates.

        References:
            [EQ-INV-TGT-001].

        """
        return self._prior_factor.shape[1]

    @property
    def physical_dimension(self) -> int:
        """Return ``n``, the length of ``x``.

        References:
            [EQ-INV-TGT-001].

        """
        return self._prior_factor.shape[0]

    @property
    def counts(self) -> TargetCounts:
        """Return a snapshot of the exact work counters.

        References:
            [EQ-INV-TGT-002].

        """
        log_likelihood, gradient, forward, vjp = self._counts
        return TargetCounts(log_likelihood, gradient, forward, vjp)

    def _check_z(self, z: Sequence[float]) -> Vector:
        """Return ``z`` as a finite vector of length ``r``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-TGT-001].

        """
        point = as_vector(z, name="z")
        if len(point) != self.dimension:
            raise ValueError(f"z has length {len(point)}, expected {self.dimension}")
        return point

    def to_physical(self, z: Sequence[float]) -> Vector:
        """Return ``x = mu + P z``.

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            Physical parameters of length ``n``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-TGT-001].

        """
        shift = self._prior_factor.matvec(self._check_z(z))
        return tuple(a + b for a, b in zip(self._mean, shift, strict=True))

    def _evaluate(self, z: Vector) -> tuple[Vector, Vector]:
        """Return ``(x, G(x))`` using the one-entry cache.

        References:
            [EQ-INV-TGT-001].

        """
        cache = self._cache
        if cache is not None and cache[0] == z:
            return cache[1], cache[2]
        x = self.to_physical(z)
        value = tuple(float(v) for v in self._model.value(x))
        self._counts[2] += 1
        self._cache = (z, x, value)
        return x, value

    def log_likelihood(self, z: Sequence[float]) -> float:
        """Return ``-1/2 ||W (G(mu + P z) - y)||^2`` (no normalizing constant).

        This is the function to pass to pCN. A non-finite forward value gives
        ``-inf`` (zero likelihood); nothing is clipped.

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            The whitened half misfit, ``<= 0``, or ``-inf``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-TGT-001], [EQ-INV-TGT-003].

        """
        point = self._check_z(z)
        self._counts[0] += 1
        _, value = self._evaluate(point)
        if not all(math.isfinite(v) for v in value):
            return -math.inf
        residual = tuple(g - d for g, d in zip(value, self._data, strict=True))
        whitened = self._noise.whiten(residual)
        try:
            return -0.5 * math.fsum(e * e for e in whitened)
        except OverflowError:
            return -math.inf

    def log_prior(self, z: Sequence[float]) -> float:
        """Return ``-1/2 ||z||^2 - r/2 log 2 pi`` (the standard normal log density).

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            ``log N(z; 0, I)``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-TGT-001].

        """
        point = self._check_z(z)
        return -0.5 * math.fsum(v * v for v in point) - 0.5 * len(point) * _LOG_TWO_PI

    def log_density(self, z: Sequence[float]) -> float:
        """Return ``log_likelihood(z) + log_prior(z)``.

        This is the target for gradient-based samplers. It omits only
        :attr:`log_likelihood_constant`; to compare with a normalized
        posterior add that attribute.

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            The posterior log density of ``z`` up to the likelihood constant,
            or ``-inf`` for a failed forward evaluation.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-TGT-001].

        """
        return self.log_likelihood(z) + self.log_prior(z)

    def gradient(self, z: Sequence[float]) -> Vector:
        """Return ``-P' J' W' W (G(x) - y) - z`` for ``x = mu + P z``.

        Uses the cached forward value of the same ``z`` when ``log_density``
        was just evaluated there, plus one ``vjp``.

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            The gradient of :meth:`log_density`.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong, an entry is not finite, or the
                forward value or its adjoint is not finite at ``z`` (HMC and
                NUTS treat this as leaving the support).

        References:
            [EQ-INV-TGT-002].

        """
        point = self._check_z(z)
        self._counts[1] += 1
        x, value = self._evaluate(point)
        if not all(math.isfinite(v) for v in value):
            raise ValueError("forward value is not finite at z")
        residual = tuple(g - d for g, d in zip(value, self._data, strict=True))
        weights = self._noise.whiten_transpose(self._noise.whiten(residual))
        adjoint = self._model.vjp(x, weights)
        self._counts[3] += 1
        if not all(math.isfinite(v) for v in adjoint):
            raise ValueError("adjoint product is not finite at z")
        pulled = self._prior_factor.rmatvec(adjoint)
        return tuple(-p - zi for p, zi in zip(pulled, point, strict=True))


def _initial_states(
    target: WhitenedGaussianPriorTarget,
    initial: Sequence[float] | Sequence[Sequence[float]] | None,
    chains: int,
    seed: int | RandomStream | None,
) -> Sequence[float] | Sequence[Sequence[float]]:
    """Return the starts: user-given, or independent standard normal draws.

    References:
        [EQ-INV-TGT-001].

    """
    if initial is not None:
        return initial
    generator = resolve_stream(seed).child("targets", "starts").generator()
    return tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(target.dimension))
        for _ in range(chains)
    )


def _check_target(target: object) -> WhitenedGaussianPriorTarget:
    """Return ``target`` or raise ``TypeError``.

    References:
        [EQ-INV-TGT-001].

    """
    if not isinstance(target, WhitenedGaussianPriorTarget):
        raise TypeError("target must be a WhitenedGaussianPriorTarget")
    return target


def sample_pcn(
    target: WhitenedGaussianPriorTarget,
    initial: Sequence[float] | Sequence[Sequence[float]] | None = None,
    *,
    draws: int,
    warmup: int,
    chains: int = 4,
    thin: int = 1,
    step: float = 0.25,
    target_acceptance: float = 0.25,
    adapt: bool = True,
    seed: int | RandomStream | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Run the library's pCN sampler on a whitened target (``z`` coordinates).

    Calls :func:`probstat.inference.preconditioned_crank_nicolson` with
    ``target.log_likelihood`` only: the standard normal prior of ``z`` is the
    pCN reference law and is never evaluated, so it is not added twice
    [EQ-INV-TGT-003]. No new sampler is implemented. Dimension-robust
    acceptance is not claimed for arbitrary nonlinear targets; test selected
    functionals under refinement.

    Args:
        target: The whitened target.
        initial: One start or one per chain (``z`` coordinates); by default
            independent standard normal (prior) draws from ``seed``.
        draws: Kept draws per chain.
        warmup: Warm-up and step-adaptation iterations per chain.
        chains: Number of chains.
        thin: Thinning interval.
        step: Initial ``beta`` in ``(0, 1]``.
        target_acceptance: Warm-up acceptance target.
        adapt: Adapt ``beta`` during warm-up only.
        seed: Integer seed or stream.
        check_convergence: Emit convergence warnings.

    Returns:
        :class:`~probstat.inference.MCMCResult` in ``z`` coordinates; map to
        ``x`` with :func:`physical_posterior`. Its ``evaluations`` field is
        the exact number of likelihood evaluations.

    Raises:
        TypeError: If ``target`` is not a :class:`WhitenedGaussianPriorTarget`.
        ValueError: As raised by the underlying sampler (invalid run lengths,
            a start with non-finite likelihood, ...).

    References:
        [EQ-INV-TGT-003], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    checked = _check_target(target)
    starts = _initial_states(checked, initial, chains, seed)
    return preconditioned_crank_nicolson(
        checked.log_likelihood,
        starts,
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        step=step,
        target_acceptance=target_acceptance,
        adapt=adapt,
        seed=seed,
        names=[f"z{i}" for i in range(checked.dimension)],
        check_convergence=check_convergence,
    )


def sample_hmc(
    target: WhitenedGaussianPriorTarget,
    initial: Sequence[float] | Sequence[Sequence[float]] | None = None,
    *,
    draws: int,
    warmup: int = 500,
    chains: int = 4,
    thin: int = 1,
    method: Literal["nuts", "hmc"] = "nuts",
    max_tree_depth: int = 10,
    leapfrog_steps: int = 16,
    metric: str = "diag",
    target_acceptance: float = 0.8,
    seed: int | RandomStream | None = None,
    check_convergence: bool = True,
) -> HamiltonianResult:
    """Run the library's NUTS or static HMC on a whitened target.

    Calls :func:`probstat.inference.no_u_turn_sampler` (or
    :func:`probstat.inference.hamiltonian_monte_carlo`) with the *full*
    ``target.log_density`` and the analytic ``target.gradient`` (adjoint
    based; no finite differences and no autodiff are used). The prior is part
    of ``log_density`` here, as gradient samplers do not cancel it
    [EQ-INV-TGT-001], [EQ-INV-TGT-002]. No new sampler is implemented.

    Args:
        target: The whitened target.
        initial: One start or one per chain (``z`` coordinates); by default
            independent standard normal draws from ``seed``.
        draws: Kept draws per chain.
        warmup: Adaptation iterations per chain.
        chains: Number of chains.
        thin: Thinning interval.
        method: ``"nuts"`` (default) or ``"hmc"``.
        max_tree_depth: NUTS tree depth limit.
        leapfrog_steps: Leapfrog steps (static HMC only).
        metric: ``"diag"``, ``"dense"`` or ``"unit"`` inverse metric.
        target_acceptance: Dual-averaging target.
        seed: Integer seed or stream.
        check_convergence: Emit convergence warnings.

    Returns:
        :class:`~probstat.inference.HamiltonianResult` in ``z`` coordinates
        (``.mcmc`` holds the draws); map with :func:`physical_posterior`.

    Raises:
        TypeError: If ``target`` is not a :class:`WhitenedGaussianPriorTarget`.
        ValueError: If ``method`` is unknown or the sampler rejects a setting.

    References:
        [EQ-INV-TGT-001], [EQ-INV-TGT-002], [REF-STUART-2010].

    """
    checked = _check_target(target)
    if method not in ("nuts", "hmc"):
        raise ValueError("method must be 'nuts' or 'hmc'")
    starts = _initial_states(checked, initial, chains, seed)
    names = [f"z{i}" for i in range(checked.dimension)]
    if method == "nuts":
        return no_u_turn_sampler(
            checked.log_density,
            starts,
            gradient=checked.gradient,
            draws=draws,
            warmup=warmup,
            chains=chains,
            thin=thin,
            max_tree_depth=max_tree_depth,
            metric=metric,
            target_acceptance=target_acceptance,
            seed=seed,
            names=names,
            check_convergence=check_convergence,
        )
    return hamiltonian_monte_carlo(
        checked.log_density,
        starts,
        gradient=checked.gradient,
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        leapfrog_steps=leapfrog_steps,
        metric=metric,
        target_acceptance=target_acceptance,
        seed=seed,
        names=names,
        check_convergence=check_convergence,
    )


def physical_posterior(
    target: WhitenedGaussianPriorTarget,
    result: MCMCResult | HamiltonianResult,
    *,
    names: Sequence[str] | None = None,
) -> MCMCResult:
    """Map sampler output from ``z`` back to ``x = mu + P z``.

    Every draw is pushed through :meth:`WhitenedGaussianPriorTarget.to_physical`
    so that R-hat, ESS and MCSE can be recomputed for physical quantities.
    The stored log-density trace stays on the ``z`` scale.

    Args:
        target: The target that produced ``result``.
        result: Draws in ``z`` coordinates.
        names: Names of the ``n`` physical coordinates (default
            ``x0, x1, ...``).

    Returns:
        :class:`~probstat.inference.MCMCResult` in ``x`` coordinates.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the draw dimension is not ``r`` or ``names`` has the
            wrong length.

    References:
        [EQ-INV-TGT-001].

    """
    checked = _check_target(target)
    chains = result.mcmc if isinstance(result, HamiltonianResult) else result
    if not isinstance(chains, MCMCResult):
        raise TypeError("result must be an MCMCResult or HamiltonianResult")
    if chains.dimension != checked.dimension:
        raise ValueError(
            f"draws have dimension {chains.dimension}, expected {checked.dimension}"
        )
    labels = (
        [f"x{i}" for i in range(checked.physical_dimension)]
        if names is None
        else list(names)
    )
    if len(labels) != checked.physical_dimension:
        raise ValueError("names must match the physical dimension")
    return chains.transformed(checked.to_physical, labels)


@dataclass(frozen=True, slots=True)
class LinearGaussianSBCResult:
    """Outcome of :func:`simulation_based_calibration_linear_gaussian`.

    Attributes:
        ranks: Per coordinate, one rank in ``{0, ..., draws}`` per replicate.
        chi_square: Pearson statistic per coordinate.
        p_values: Upper-tail chi-square p-value per coordinate (``bins - 1``
            degrees of freedom).
        histograms: Per coordinate, the counts in each rank bin.
        draws: Posterior draws ``L`` per replicate.
        bins: Histogram bins.
        replicates: Number of simulated data sets.
        sampler: ``"exact"`` or ``"custom"``.
        caveat: The standing caveat: SBC under the simulated model does not
            validate the real-world model.

    References:
        [EQ-INV-TGT-004], [REF-TALTS-ETAL-2018].

    """

    ranks: tuple[tuple[int, ...], ...]
    chi_square: tuple[float, ...]
    p_values: tuple[float, ...]
    histograms: tuple[tuple[int, ...], ...]
    draws: int
    bins: int
    replicates: int
    sampler: str
    caveat: str

    def min_p_value(self) -> float:
        """Return the smallest per-coordinate p-value.

        References:
            [EQ-INV-TGT-004].

        """
        return min(self.p_values)

    def uniform_at(self, level: float = 1e-3) -> bool:
        """Return whether every coordinate has ``p > level`` (a screening rule).

        This is a heuristic screen, not a calibrated family-wise test; with
        ``k`` coordinates a correct sampler fails it with probability about
        ``k level``.

        Args:
            level: Per-coordinate p-value threshold in ``(0, 1)``.

        Returns:
            ``True`` when no coordinate rejects uniformity at ``level``.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        References:
            [EQ-INV-TGT-004].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        return all(p > level for p in self.p_values)


def simulation_based_calibration_linear_gaussian(
    operator: object,
    noise: object,
    prior_mean: Sequence[float] | None,
    prior_covariance: object,
    *,
    replicates: int,
    posterior_draws: int,
    seed: int | RandomStream | None = 0,
    sampler: Literal["exact"]
    | Callable[[Vector, int, RandomStream], Sequence[Sequence[float]]] = "exact",
    bins: int | None = None,
) -> LinearGaussianSBCResult:
    """Run simulation-based calibration of a linear-Gaussian posterior sampler.

    For each replicate: ``theta ~ N(mu, C0)``, ``y = A theta + eps`` with
    ``eps ~ N(0, Gamma)``, then ``posterior_draws`` draws are obtained from
    the exact conditional simulation of
    :func:`probstat.inverse.linear_gaussian_posterior` (``sampler="exact"``)
    or from a user callable ``sampler(data, count, stream) -> draws`` (draws
    should be nearly independent; thin Markov chains first). The ranks of each
    coordinate of ``theta`` among its draws are computed by the existing
    :func:`probstat.inference.simulation_based_calibration` (randomized tie
    breaking from the replicate's ``"ties"`` stream) and tested for
    uniformity with a chi-square statistic [EQ-INV-TGT-004].

    The exact sampler must pass. Passing says the *sampler* is calibrated for
    the *simulated* model; it does not validate the real-world model
    (:data:`SBC_CAVEAT`).

    Args:
        operator: ``A`` (``m x n``), a ``LinearOperator`` or dense matrix.
        noise: Covariance factor of ``Gamma`` (dimension ``m``).
        prior_mean: ``mu`` (``None`` for zeros).
        prior_covariance: Covariance factor of ``C0`` (dimension ``n``).
        replicates: Number of simulated data sets (at least two).
        posterior_draws: Draws ``L >= 1`` per replicate.
        seed: Integer seed or stream.
        sampler: ``"exact"`` or ``(data, count, stream) -> draws``.
        bins: Rank-histogram bins (default of the existing helper).

    Returns:
        :class:`LinearGaussianSBCResult` with rank arrays, chi-square
        statistics, p-values and the caveat.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree, counts are invalid, or a custom
            sampler returns the wrong number or dimension of draws.

    References:
        [EQ-INV-TGT-004], [REF-TALTS-ETAL-2018],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    forward = require_operator(operator, name="operator")
    m, n = forward.shape
    checked_noise = _require_covariance(noise, m)
    checked_prior = _require_covariance(prior_covariance, n)
    mu: Vector = (
        (0.0,) * n if prior_mean is None else as_vector(prior_mean, name="prior_mean")
    )
    if len(mu) != n:
        raise ValueError(f"prior_mean has length {len(mu)}, expected {n}")
    count = require_non_negative_integer(posterior_draws, name="posterior_draws")
    if count < 1:
        raise ValueError("posterior_draws must be at least 1")
    if sampler != "exact" and not callable(sampler):
        raise ValueError("sampler must be 'exact' or a callable")

    def draw_prior(generator: random.Random) -> Vector:
        """Return ``theta ~ N(mu, C0)``.

        References:
            [EQ-INV-TGT-004].

        """
        xi = [generator.gauss(0.0, 1.0) for _ in range(n)]
        shift = checked_prior.unwhiten(xi)
        return tuple(a + b for a, b in zip(mu, shift, strict=True))

    def simulate(theta: object, generator: random.Random) -> Vector:
        """Return ``y = A theta + eps``, ``eps ~ N(0, Gamma)``.

        References:
            [EQ-INV-TGT-004].

        """
        xi = [generator.gauss(0.0, 1.0) for _ in range(m)]
        noise_draw = checked_noise.unwhiten(xi)
        image = forward.matvec(as_vector(theta, name="theta"))  # type: ignore[arg-type]
        return tuple(a + b for a, b in zip(image, noise_draw, strict=True))

    def posterior(data: object, stream: RandomStream) -> Sequence[Sequence[float]]:
        """Return ``posterior_draws`` draws of ``theta | data``.

        References:
            [EQ-INV-TGT-004].

        """
        observed = as_vector(data, name="data")  # type: ignore[arg-type]
        if callable(sampler):
            return sampler(observed, count, stream)
        exact = linear_gaussian_posterior(
            forward,
            observed,
            noise=checked_noise,
            prior_mean=mu,
            prior_covariance=checked_prior,
        )
        return exact.sample(count, seed=stream)

    result = simulation_based_calibration(
        draw_prior,
        simulate,
        posterior,
        replicates=replicates,
        bins=bins,
        seed=seed,
    )
    return LinearGaussianSBCResult(
        ranks=result.ranks,
        chi_square=result.chi_square,
        p_values=result.p_values,
        histograms=result.histograms,
        draws=result.draws,
        bins=result.bins,
        replicates=result.replicates,
        sampler="exact" if sampler == "exact" else "custom",
        caveat=SBC_CAVEAT,
    )
