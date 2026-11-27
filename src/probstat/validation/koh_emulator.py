"""Kennedy-O'Hagan calibration of an expensive simulator through an emulator.

**Experimental (added in 0.28).** Synthetic-data workflows only; nothing here
describes a real simulator.

The simulator ``eta(x, t)`` (control input ``x`` in ``R^p``, calibration
parameter ``t`` in ``R^q``) is run ``n`` times on a fixed design
``(x_j^s, t_j^s)`` giving ``z_j = eta(x_j^s, t_j^s)``. Field data
``y_i = zeta(x_i) + eps_i`` at ``N`` control inputs are modelled with

``zeta(x) = eta(x, theta) + delta(x)``, ``eps_i ~ N(0, s_i^2)``,

where ``theta`` is the unknown true value of the calibration parameter and
``delta ~ GP(0, k_delta)`` is the model discrepancy (Kennedy and O'Hagan
2001; Higdon et al. 2004). The simulator is emulated by a Gaussian process
``eta ~ GP(m, k_eta)`` on the joint ``(x, t)`` space with an ARD kernel
fitted to the simulator runs alone (library routine
:func:`probstat.surrogates.fit_gaussian_process`).

**Modular approach.** The emulator hyperparameters ``(m, sigma_f^2, ell,
nugget)`` are fixed at that simulator-only fit; only ``theta`` and the
discrepancy/noise hyperparameters ``psi`` are inferred. The fully joint
alternative (re-estimating the emulator from simulator and field data
together) is not implemented. The results object records
``emulator_mode = "modular"``.

**Joint likelihood** [EQ-VAL-KOHEM-002]. With ``u = (z, y)`` the
simulator-and-field vector,

``u | theta, psi ~ N(m 1, Sigma)``, ``Sigma = [[A, B], [B', C + D + E]]``,

``A_jj' = k_eta((x_j^s, t_j^s), (x_j'^s, t_j'^s)) + nu 1[j=j']``,
``B_ji = k_eta((x_j^s, t_j^s), (x_i, theta))``,
``C_ii' = k_eta((x_i, theta), (x_i', theta))``,
``D_ii' = k_delta(x_i, x_i')``, ``E = diag(s_i^2)``,

with ``nu`` the emulator nugget. Because ``A`` does not depend on ``(theta,
psi)``, the joint density factorizes as ``p(z) p(y | z, theta, psi)`` with
``y | z ~ N(mu, S)``, ``mu = m 1 + B' A^-1 (z - m 1)`` and
``S = C - B' A^-1 B + D + E`` [EQ-VAL-KOHEM-003]; the code evaluates this
form with one Cholesky factorization of ``A`` (at construction) and one of
``S`` per call, and :meth:`KOHEmulatorCalibration.log_joint_likelihood`
adds the constant ``log p(z)`` so that it equals the full joint log density.

**Posterior** [EQ-VAL-KOHEM-004]: ``p(theta, psi | z, y)`` is proportional to
joint likelihood times prior, with ``theta`` uniform on a box and the log
discrepancy amplitude, the log length scales (relative to the control range)
and, optionally, the log noise sd normal. Sampling is by coordinate-wise
slice sampling (default) or transitional MCMC.

**Predictive decomposition** [EQ-VAL-KOHEM-005, EQ-VAL-KOHEM-006]. For a new
control input ``x*`` and one posterior draw ``(theta, psi)`` the reality
``zeta* = eta(x*, theta) + delta*`` is Gaussian given ``(z, y)``. Write
``V_eta``, ``V_delta`` for the conditional variances of ``eta*`` and
``delta*`` and ``C_ed`` for their conditional covariance. The law of total
variance over the posterior draws gives, exactly,

``Var(y*) = E[V_eta] + E[V_delta] + 2 E[C_ed] + Var(E[zeta* | draw])
+ E[s*^2]``,

called the *emulator*, *discrepancy*, *emulator-discrepancy covariance*,
*parameter* and *noise* parts. The covariance term is not attributed to
either field because the data constrain only their sum.

**Identifiability** [EQ-VAL-KOHEM-007]. ``theta`` and ``delta`` are confounded
(Brynjarsdottir and O'Hagan 2014): a shift of ``theta`` can be absorbed by
``delta``. :meth:`KOHEmulatorCalibration.identifiability` reports the
posterior-to-prior variance contraction of ``theta``, its posterior
correlation with the log discrepancy amplitude and length scales, and a
comparison with the "no discrepancy" calibration, which is overconfident and
biased when the simulator is structurally wrong.

Limitations: a constant emulator mean and no scale factor ``rho`` between
simulator and reality; scalar output; cost ``O((n + N)^3)`` per prediction
draw, so intended for tens to a few hundred runs.

Parameter vectors are ``(theta, log sigma_delta, log (ell_delta / range),
log s)`` where the last two groups exist only when the discrepancy is on and
the noise is estimated respectively; see
:attr:`KOHEmulatorCalibration.names`.

References:
    [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
    [REF-BAYARRI-ETAL-2007], [REF-BRYNJARSDOTTIR-OHAGAN-2014],
    [REF-RASMUSSEN-WILLIAMS-2006].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import (
    MCMCResult,
    bulk_effective_sample_size,
    slice_sampler,
    split_rhat,
    transitional_mcmc,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, solve_lower
from probstat.modeling import standard_normal_cdf
from probstat.surrogates.gaussian_process import (
    KERNELS,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
    kernel_value,
)

SAMPLERS = ("slice", "tmcmc")
_LOG_TWO_PI = math.log(2.0 * math.pi)
_VARIANCE_ROUNDING = 1e-9


@dataclass(frozen=True, slots=True)
class LogNormalPrior:
    """Log-normal prior, given as a normal law for the logarithm.

    Args:
        mean: Mean of the logarithm.
        sd: Standard deviation of the logarithm (positive).

    Raises:
        ValueError: If ``mean`` is not finite or ``sd`` is not positive.

    References:
        [EQ-VAL-KOHEM-004], [REF-HIGDON-ETAL-2004].

    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``mean`` is not finite or ``sd`` is not positive.

        References:
            [EQ-VAL-KOHEM-004].

        """
        require_finite_real(self.mean, name="mean")
        if require_finite_real(self.sd, name="sd") <= 0.0:
            raise ValueError("sd must be positive")

    def log_density(self, log_value: float) -> float:
        """Return the normal log density of the logarithm.

        Args:
            log_value: The logarithm of the parameter.

        Returns:
            ``log N(log_value; mean, sd^2)``.

        References:
            [EQ-VAL-KOHEM-004].

        """
        z = (log_value - self.mean) / self.sd
        return -0.5 * z * z - math.log(self.sd) - 0.5 * _LOG_TWO_PI


@dataclass(frozen=True, slots=True)
class KOHResult:
    """Posterior sample of a Kennedy-O'Hagan emulator calibration.

    Args:
        names: Coordinate names of the parameter vector.
        samples: Pooled posterior draws (one vector per draw).
        theta_dimension: Number ``q`` of leading coordinates that are ``theta``.
        sampler: ``"slice"`` or ``"tmcmc"``.
        discrepancy: Whether a discrepancy term was modelled.
        emulator_mode: Always ``"modular"`` (emulator hyperparameters fixed).
        converged: Convergence verdict (see ``stop_reason``).
        stop_reason: ``"converged"``, ``"rhat"`` (split R-hat above the
            limit), ``"ess"`` (bulk ESS below the limit) or a TMCMC stop
            reason.
        rhat_max: Largest split R-hat over coordinates (``nan`` for TMCMC).
        ess_min: Smallest bulk effective sample size (``nan`` for TMCMC).
        evaluations: Log-posterior evaluations.
        failed_evaluations: Evaluations rejected because the covariance was
            not numerically positive definite (treated as zero density).
        log_evidence: TMCMC log marginal likelihood of ``(y | z)``
            (``nan`` for slice sampling or when TMCMC did not converge).
        mcmc: The underlying :class:`MCMCResult` for slice sampling.

    References:
        [EQ-VAL-KOHEM-004], [REF-NEAL-2003], [REF-CHING-CHEN-2007].

    """

    names: tuple[str, ...]
    samples: tuple[Vector, ...]
    theta_dimension: int
    sampler: str
    discrepancy: bool
    emulator_mode: str
    converged: bool
    stop_reason: str
    rhat_max: float
    ess_min: float
    evaluations: int
    failed_evaluations: int
    log_evidence: float
    mcmc: MCMCResult | None

    def column(self, name_or_index: str | int) -> Vector:
        """Return the draws of one coordinate.

        Args:
            name_or_index: Coordinate name or index.

        Returns:
            One value per draw.

        Raises:
            ValueError: If the name is unknown or the index out of range.

        References:
            [EQ-VAL-KOHEM-004].

        """
        if isinstance(name_or_index, str):
            if name_or_index not in self.names:
                raise ValueError(f"unknown coordinate {name_or_index!r}")
            index = self.names.index(name_or_index)
        else:
            index = name_or_index
            if not 0 <= index < len(self.names):
                raise ValueError("coordinate index out of range")
        return tuple(sample[index] for sample in self.samples)

    def mean(self) -> Vector:
        """Return the posterior mean of every coordinate.

        Returns:
            Vector of sample means.

        References:
            [EQ-VAL-KOHEM-004].

        """
        count = len(self.samples)
        return tuple(
            math.fsum(sample[k] for sample in self.samples) / count
            for k in range(len(self.names))
        )

    def sd(self) -> Vector:
        """Return the posterior standard deviation of every coordinate.

        Returns:
            Vector of sample standard deviations (denominator ``M - 1``).

        References:
            [EQ-VAL-KOHEM-004].

        """
        count = len(self.samples)
        means = self.mean()
        return tuple(
            math.sqrt(
                math.fsum((sample[k] - means[k]) ** 2 for sample in self.samples)
                / (count - 1)
            )
            for k in range(len(self.names))
        )

    def theta_interval(self, index: int, level: float = 0.95) -> tuple[float, float]:
        """Return the central credible interval of one ``theta`` coordinate.

        Args:
            index: Coordinate of ``theta`` (``0 <= index < q``).
            level: Credible level in ``(0, 1)``.

        Returns:
            Lower and upper empirical quantiles.

        Raises:
            ValueError: If ``index`` or ``level`` is invalid.

        References:
            [EQ-VAL-KOHEM-004].

        """
        if not 0 <= index < self.theta_dimension:
            raise ValueError("index must address a theta coordinate")
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        values = sorted(self.column(index))
        tail = 0.5 * (1.0 - level)
        return _empirical_quantile(values, tail), _empirical_quantile(
            values, 1.0 - tail
        )


@dataclass(frozen=True, slots=True)
class KOHPrediction:
    """Posterior predictive of reality with an exact variance decomposition.

    All sequences have one entry per new control input. ``total_variance`` is
    the variance of the mixture over posterior draws; the five parts add up
    to it (``component_sum``) [EQ-VAL-KOHEM-006].

    Args:
        controls: New control inputs.
        mean: Predictive mean.
        emulator: ``E[V_eta]``, emulator (simulator interpolation) variance.
        discrepancy: ``E[V_delta]``, discrepancy variance.
        covariance: ``2 E[C_ed]``, emulator-discrepancy covariance part.
        parameter: Variance across draws of the conditional mean (posterior
            uncertainty of ``theta`` and the hyperparameters).
        noise: ``E[s*^2]`` when the prediction is of a new observation, else 0.
        total_variance: Mixture variance, computed from the per-draw means
            and variances.
        lower: Lower end of the central interval of the Gaussian mixture.
        upper: Upper end of the central interval.
        level: Interval level.
        draws: Number of posterior draws used.
        includes_noise: Whether the noise part is included.

    References:
        [EQ-VAL-KOHEM-005], [EQ-VAL-KOHEM-006].

    """

    controls: tuple[Vector, ...]
    mean: Vector
    emulator: Vector
    discrepancy: Vector
    covariance: Vector
    parameter: Vector
    noise: Vector
    total_variance: Vector
    lower: Vector
    upper: Vector
    level: float
    draws: int
    includes_noise: bool

    def component_sum(self) -> Vector:
        """Return the sum of the five variance parts per control input.

        Returns:
            ``emulator + discrepancy + covariance + parameter + noise``.

        References:
            [EQ-VAL-KOHEM-006].

        """
        return tuple(
            math.fsum(parts)
            for parts in zip(
                self.emulator,
                self.discrepancy,
                self.covariance,
                self.parameter,
                self.noise,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    """Identifiability diagnostics of ``theta`` [EQ-VAL-KOHEM-007].

    Args:
        theta_prior_sd: Prior standard deviation (uniform box).
        theta_posterior_sd: Posterior standard deviation.
        contraction: ``1 - posterior variance / prior variance`` per
            coordinate; near zero means the data did not inform ``theta``.
        weakly_identified: ``contraction < contraction_threshold``.
        contraction_threshold: Screening threshold used (a convention, not a
            test with a size).
        amplitude_correlation: Posterior correlation of each ``theta``
            coordinate with the log discrepancy amplitude (``nan`` entries
            without a discrepancy).
        length_scale_correlation: Per ``theta`` coordinate, correlations with
            each log discrepancy length scale (empty without a discrepancy).
        sd_ratio: Posterior sd with discrepancy over posterior sd without
            (``()`` when no comparison run is given).
        mean_shift_in_reference_sd: ``(mean_with - mean_without) /
            sd_without`` per coordinate (``()`` without comparison).
        reference_overconfident: Per coordinate, ``True`` when the means differ
            by more than two no-discrepancy posterior sds, i.e. the
            no-discrepancy interval excludes the with-discrepancy mean
            (``()`` without comparison).
        truth_z_with: ``(mean - truth) / sd`` with the discrepancy, when a
            synthetic truth is supplied.
        truth_z_without: Same for the no-discrepancy run.

    References:
        [EQ-VAL-KOHEM-007], [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """

    theta_prior_sd: Vector
    theta_posterior_sd: Vector
    contraction: Vector
    weakly_identified: tuple[bool, ...]
    contraction_threshold: float
    amplitude_correlation: Vector
    length_scale_correlation: tuple[Vector, ...]
    sd_ratio: Vector
    mean_shift_in_reference_sd: Vector
    reference_overconfident: tuple[bool, ...]
    truth_z_with: Vector
    truth_z_without: Vector


def _empirical_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated quantile of sorted values.

    Args:
        sorted_values: Ascending values.
        probability: Probability in ``[0, 1]``.

    Returns:
        Interpolated quantile (type 7).

    References:
        [EQ-VAL-KOHEM-004].

    """
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] + fraction * (sorted_values[high] - sorted_values[low])


def _correlation_coefficient(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Pearson correlation, ``nan`` if either input is constant.

    Args:
        left: First sample.
        right: Second sample of equal length.

    Returns:
        Sample correlation coefficient.

    References:
        [EQ-VAL-KOHEM-007].

    """
    count = len(left)
    mean_left = math.fsum(left) / count
    mean_right = math.fsum(right) / count
    sxx = math.fsum((a - mean_left) ** 2 for a in left)
    syy = math.fsum((b - mean_right) ** 2 for b in right)
    if sxx <= 0.0 or syy <= 0.0:
        return math.nan
    sxy = math.fsum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    )
    return sxy / math.sqrt(sxx * syy)


def _matrix_rows(rows: Sequence[Sequence[float]], name: str) -> tuple[Vector, ...]:
    """Validate a rectangular, finite, non-empty matrix of floats.

    Args:
        rows: Candidate rows.
        name: Name for messages.

    Returns:
        Rows as float tuples.

    Raises:
        ValueError: If empty, ragged or non-finite.

    References:
        [EQ-VAL-KOHEM-002].

    """
    checked = tuple(as_vector(row, name=name) for row in rows)
    if not checked or not checked[0]:
        raise ValueError(f"{name} must be non-empty")
    width = len(checked[0])
    if any(len(row) != width for row in checked):
        raise ValueError(f"{name} rows must have equal length")
    return checked


def _mixture_cdf(
    value: float, means: Sequence[float], sds: Sequence[float], weight: float
) -> float:
    """Return the CDF of an equally weighted Gaussian mixture.

    Args:
        value: Evaluation point.
        means: Component means.
        sds: Component standard deviations (zero allowed: a point mass).
        weight: ``1 / number of components``.

    Returns:
        Mixture CDF at ``value``.

    References:
        [EQ-VAL-KOHEM-006].

    """
    total = 0.0
    for mean, sd in zip(means, sds, strict=True):
        if sd > 0.0:
            total += standard_normal_cdf((value - mean) / sd)
        elif value >= mean:
            total += 1.0
    return total * weight


def _mixture_quantile(
    probability: float, means: Sequence[float], sds: Sequence[float]
) -> float:
    """Return a quantile of an equally weighted Gaussian mixture by bisection.

    Args:
        probability: Probability in ``(0, 1)``.
        means: Component means.
        sds: Component standard deviations.

    Returns:
        The quantile, accurate to the bisection tolerance.

    References:
        [EQ-VAL-KOHEM-006].

    """
    weight = 1.0 / len(means)
    low = min(m - 9.0 * s for m, s in zip(means, sds, strict=True)) - 1e-12
    high = max(m + 9.0 * s for m, s in zip(means, sds, strict=True)) + 1e-12
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _mixture_cdf(middle, means, sds, weight) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class KOHEmulatorCalibration:
    """Kennedy-O'Hagan calibration of an expensive simulator by an emulator.

    Construction fits (or validates) the emulator, factorizes the
    simulator-run covariance once and stores the data; the methods evaluate
    the log likelihood and posterior, sample the posterior and predict reality.

    Args:
        simulator_controls: ``n x p`` control inputs of the simulator runs.
        simulator_parameters: ``n x q`` calibration parameters of the runs.
        simulator_outputs: ``n`` simulator outputs.
        field_controls: ``N x p`` control inputs of the field data.
        field_outputs: ``N`` field observations.
        theta_bounds: ``q`` pairs ``(low, high)``: the uniform prior box of
            ``theta`` (it should cover the design range in ``t``; the
            emulator is not trusted outside it).
        noise_sd: Known observation noise sd, one value or one per
            observation. Exactly one of ``noise_sd`` and ``noise_prior``.
        noise_prior: Log-normal prior of one unknown noise sd.
        discrepancy: Include the GP discrepancy ``delta`` (default ``True``);
            ``False`` is the "no discrepancy" calibration.
        discrepancy_kernel: ``"squared_exponential"``, ``"matern52"`` or
            ``"matern32"``.
        amplitude_prior: Prior of the discrepancy sd ``sigma_delta`` in output
            units. Default: ``LogNormalPrior(log(0.1 s), 1)`` with ``s`` the
            simulator output standard deviation (a stated default: the
            identifiability depends on it, see the module docstring).
        length_scale_prior: Prior of ``ell_delta / range`` per control
            dimension, ``range`` the width of the control inputs over runs
            and field data. Default ``LogNormalPrior(log(0.5), 0.5)``.
        emulator: Optional fitted :class:`GaussianProcessRegression` on the
            joint ``(x, t)`` points ``simulator_controls ++ simulator_parameters``
            and the given outputs; fitted here when ``None``.
        emulator_kernel: Kernel of the fitted emulator.
        emulator_nugget: Emulator nugget, standardized units.
        emulator_restarts: Optimizer restarts of the emulator fit.
        emulator_regularization: Optional
            :class:`~probstat.surrogates.GaussianProcessRegularization` for the
            emulator fit (bounds the conditioning of nearly polynomial
            simulators).
        seed: Seed or stream for the emulator fit and sampling defaults.

    Raises:
        TypeError: If ``emulator`` has the wrong type.
        ValueError: For inconsistent shapes, non-finite data, empty or
            reversed bounds, a noise specification that is not exactly one of
            the two options, an unknown kernel, an emulator that does not
            match the data, or a simulator-run covariance that is not
            positive definite.

    Equation:
        [EQ-VAL-KOHEM-001], [EQ-VAL-KOHEM-002], [EQ-VAL-KOHEM-003].

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
        [REF-RASMUSSEN-WILLIAMS-2006].

    """

    def __init__(
        self,
        simulator_controls: Sequence[Sequence[float]],
        simulator_parameters: Sequence[Sequence[float]],
        simulator_outputs: Sequence[float],
        field_controls: Sequence[Sequence[float]],
        field_outputs: Sequence[float],
        *,
        theta_bounds: Sequence[tuple[float, float]],
        noise_sd: float | Sequence[float] | None = None,
        noise_prior: LogNormalPrior | None = None,
        discrepancy: bool = True,
        discrepancy_kernel: str = "matern52",
        amplitude_prior: LogNormalPrior | None = None,
        length_scale_prior: LogNormalPrior | None = None,
        emulator: GaussianProcessRegression | None = None,
        emulator_kernel: str = "matern52",
        emulator_nugget: float = 1e-8,
        emulator_restarts: int = 3,
        emulator_regularization: GaussianProcessRegularization | None = None,
        seed: int | RandomStream | None = None,
    ) -> None:
        """Validate the data, fit the emulator and factorize the run covariance.

        Raises:
            TypeError: If ``emulator`` has the wrong type.
            ValueError: See the class docstring.

        Equation:
            [EQ-VAL-KOHEM-001], [EQ-VAL-KOHEM-002].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        controls = _matrix_rows(simulator_controls, "simulator_controls")
        parameters = _matrix_rows(simulator_parameters, "simulator_parameters")
        outputs = as_vector(simulator_outputs, name="simulator_outputs")
        field_x = _matrix_rows(field_controls, "field_controls")
        field_y = as_vector(field_outputs, name="field_outputs")
        runs = len(outputs)
        if len(controls) != runs or len(parameters) != runs:
            raise ValueError("simulator inputs and outputs must have equal length")
        if len(field_x) != len(field_y):
            raise ValueError("field controls and outputs must have equal length")
        p = len(controls[0])
        q = len(parameters[0])
        if len(field_x[0]) != p:
            raise ValueError("field controls must have the simulator control width")
        bounds = tuple((float(a), float(b)) for a, b in theta_bounds)
        if len(bounds) != q:
            raise ValueError("theta_bounds must have one pair per parameter column")
        for low, high in bounds:
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("theta_bounds must be finite with low < high")
        if (noise_sd is None) == (noise_prior is None):
            raise ValueError("give exactly one of noise_sd and noise_prior")
        if discrepancy_kernel not in KERNELS:
            raise ValueError(f"discrepancy_kernel must be one of {KERNELS}")
        if noise_prior is not None and not isinstance(noise_prior, LogNormalPrior):
            raise TypeError("noise_prior must be a LogNormalPrior")
        self._noise_known: Vector | None = None
        if noise_sd is not None:
            if isinstance(noise_sd, (int, float)):
                known: Vector = (require_finite_real(noise_sd, name="noise_sd"),) * len(
                    field_y
                )
            else:
                known = as_vector(noise_sd, name="noise_sd")
                if len(known) != len(field_y):
                    raise ValueError("noise_sd must have one entry per observation")
            if any(value <= 0.0 for value in known):
                raise ValueError("noise_sd must be positive")
            self._noise_known = known
            self._noise_scalar = len(set(known)) == 1
        else:
            self._noise_scalar = True
        joint = tuple(c + t for c, t in zip(controls, parameters, strict=True))
        if emulator is None:
            emulator = fit_gaussian_process(
                joint,
                outputs,
                kernel=emulator_kernel,
                nugget=emulator_nugget,
                restarts=emulator_restarts,
                regularization=emulator_regularization,
                seed=seed,
            )
        elif not isinstance(emulator, GaussianProcessRegression):
            raise TypeError("emulator must be a GaussianProcessRegression")
        elif emulator.points != joint or emulator.values != outputs:
            raise ValueError("emulator must be fitted to the given simulator runs")
        self.emulator = emulator
        self.emulator_mode = "modular"
        hyper = emulator.hyperparameters
        self._m = emulator.output_mean
        self._signal = hyper.signal_variance
        self._ells = hyper.length_scales
        self._nugget = hyper.noise_variance
        self._kernel = emulator.kernel
        self._joint_points = joint
        self._outputs = outputs
        self._field_x = field_x
        self._field_y = field_y
        self._runs = runs
        self._count = len(field_y)
        self._p = p
        self._q = q
        self.theta_bounds = bounds
        self.discrepancy = bool(discrepancy)
        self.discrepancy_kernel = discrepancy_kernel
        self.noise_prior = noise_prior
        spread = []
        for axis in range(p):
            column = [row[axis] for row in controls] + [row[axis] for row in field_x]
            width = max(column) - min(column)
            if width <= 0.0:
                raise ValueError(f"control input {axis} is constant")
            spread.append(width)
        self._ranges: Vector = tuple(spread)
        self.amplitude_prior = amplitude_prior or LogNormalPrior(
            math.log(0.1 * emulator.output_scale), 1.0
        )
        self.length_scale_prior = length_scale_prior or LogNormalPrior(
            math.log(0.5), 0.5
        )
        names = [f"theta{k}" for k in range(q)]
        if self.discrepancy:
            names.append("log_sigma_delta")
            names.extend(f"log_ell_delta{k}" for k in range(p))
        if noise_prior is not None:
            names.append("log_noise_sd")
        self.names: tuple[str, ...] = tuple(names)
        self._cache: tuple[Vector, Vector, Matrix] | None = None
        self._failures = 0
        self._evaluations = 0
        matrix = [
            [
                kernel_value(self._kernel, joint[a], joint[b], self._ells, self._signal)
                + (self._nugget if a == b else 0.0)
                for b in range(runs)
            ]
            for a in range(runs)
        ]
        try:
            self._lower_a = cholesky(tuple(tuple(row) for row in matrix))
        except ValueError as error:
            raise ValueError(
                "simulator-run covariance is not positive definite; use a larger "
                "emulator_nugget or remove duplicate runs"
            ) from error
        centred = tuple(value - self._m for value in outputs)
        self._whitened_z = solve_lower(self._lower_a, centred)
        log_det = 2.0 * math.fsum(math.log(self._lower_a[k][k]) for k in range(runs))
        self.simulator_log_likelihood = -0.5 * (
            runs * _LOG_TWO_PI + log_det + math.fsum(v * v for v in self._whitened_z)
        )

    @property
    def dimension(self) -> int:
        """Return the length of the parameter vector.

        Returns:
            ``q`` plus the discrepancy and noise coordinates.

        References:
            [EQ-VAL-KOHEM-004].

        """
        return len(self.names)

    def unpack(self, phi: Sequence[float]) -> tuple[Vector, float, Vector, Vector]:
        """Split a parameter vector into its natural-scale parts.

        Args:
            phi: Parameter vector (see :attr:`names`).

        Returns:
            ``(theta, sigma_delta, ell_delta, noise_variances)``: the
            discrepancy sd (0 without discrepancy), its length scales in
            control units (empty without discrepancy) and the ``N`` noise
            variances.

        Raises:
            ValueError: If ``phi`` has the wrong length or is not finite.

        References:
            [EQ-VAL-KOHEM-004].

        """
        vector = as_vector(phi, name="phi")
        if len(vector) != self.dimension:
            raise ValueError(f"phi must have {self.dimension} entries")
        theta = vector[: self._q]
        cursor = self._q
        sigma_delta = 0.0
        ells: Vector = ()
        if self.discrepancy:
            sigma_delta = math.exp(vector[cursor])
            ells = tuple(
                math.exp(vector[cursor + 1 + k]) * self._ranges[k]
                for k in range(self._p)
            )
            cursor += 1 + self._p
        if self.noise_prior is not None:
            variance = math.exp(2.0 * vector[cursor])
            variances: Vector = (variance,) * self._count
        else:
            assert self._noise_known is not None
            variances = tuple(s * s for s in self._noise_known)
        return theta, sigma_delta, ells, variances

    def log_prior(self, phi: Sequence[float]) -> float:
        """Return the normalized log prior density of a parameter vector.

        ``theta`` is uniform on the box (``-inf`` outside); the log
        hyperparameters are independent normals [EQ-VAL-KOHEM-004].

        Args:
            phi: Parameter vector.

        Returns:
            Log prior density.

        References:
            [EQ-VAL-KOHEM-004].

        """
        vector = as_vector(phi, name="phi")
        if len(vector) != self.dimension:
            raise ValueError(f"phi must have {self.dimension} entries")
        total = 0.0
        for value, (low, high) in zip(vector, self.theta_bounds, strict=False):
            if not low <= value <= high:
                return -math.inf
            total -= math.log(high - low)
        cursor = self._q
        if self.discrepancy:
            total += self.amplitude_prior.log_density(vector[cursor])
            for k in range(self._p):
                total += self.length_scale_prior.log_density(vector[cursor + 1 + k])
            cursor += 1 + self._p
        if self.noise_prior is not None:
            total += self.noise_prior.log_density(vector[cursor])
        return total

    def _theta_point(self, i: int, theta: Vector) -> Vector:
        """Return the joint emulator input ``(x_i, theta)``.

        Args:
            i: Field index.
            theta: Calibration parameter.

        Returns:
            Concatenated vector.

        References:
            [EQ-VAL-KOHEM-002].

        """
        return self._field_x[i] + theta

    def _eta(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Return the emulator prior covariance ``k_eta``.

        Args:
            left: Joint point.
            right: Joint point.

        Returns:
            Kernel value with the fitted ARD hyperparameters.

        References:
            [EQ-VAL-KOHEM-001].

        """
        return kernel_value(self._kernel, left, right, self._ells, self._signal)

    def _delta(
        self, left: Sequence[float], right: Sequence[float], sigma: float, ells: Vector
    ) -> float:
        """Return the discrepancy covariance ``k_delta``.

        Args:
            left: Control input.
            right: Control input.
            sigma: Discrepancy sd (0 gives zero).
            ells: Length scales.

        Returns:
            ``k_delta(left, right)``.

        References:
            [EQ-VAL-KOHEM-002].

        """
        if sigma == 0.0:
            return 0.0
        return kernel_value(self.discrepancy_kernel, left, right, ells, sigma * sigma)

    def _emulator_part(self, theta: Vector) -> tuple[Vector, Matrix]:
        """Return ``mu`` and ``C - B' A^-1 B`` for one ``theta`` (cached).

        The last ``theta`` is cached: coordinate-wise samplers that move only
        the hyperparameters then skip the ``O(n^2 N)`` solves.

        Args:
            theta: Calibration parameter.

        Returns:
            Conditional mean of the field data given the runs and the
            emulator part of its conditional covariance [EQ-VAL-KOHEM-003].

        References:
            [EQ-VAL-KOHEM-003].

        """
        cached = self._cache
        if cached is not None and cached[0] == theta:
            return cached[1], cached[2]
        columns: list[Vector] = []
        means: list[float] = []
        for i in range(self._count):
            point = self._theta_point(i, theta)
            cross = tuple(self._eta(point, run) for run in self._joint_points)
            white = solve_lower(self._lower_a, cross)
            columns.append(white)
            means.append(
                self._m
                + math.fsum(a * b for a, b in zip(white, self._whitened_z, strict=True))
            )
        rows: list[list[float]] = [[0.0] * self._count for _ in range(self._count)]
        for i in range(self._count):
            point_i = self._theta_point(i, theta)
            for j in range(i + 1):
                value = self._eta(point_i, self._theta_point(j, theta)) - math.fsum(
                    a * b for a, b in zip(columns[i], columns[j], strict=True)
                )
                rows[i][j] = value
                rows[j][i] = value
        mean = tuple(means)
        base = tuple(tuple(row) for row in rows)
        self._cache = (theta, mean, base)
        return mean, base

    def _field_covariance(
        self, phi: Sequence[float]
    ) -> tuple[Vector, Matrix, Vector, Vector]:
        """Return the conditional mean and covariance of the field data.

        Args:
            phi: Parameter vector.

        Returns:
            ``(mu, S, theta, noise variances)`` of ``y | z`` [EQ-VAL-KOHEM-003].

        References:
            [EQ-VAL-KOHEM-003].

        """
        theta, sigma, ells, noise = self.unpack(phi)
        mean, base = self._emulator_part(theta)
        rows: list[list[float]] = []
        for i in range(self._count):
            row = list(base[i])
            for j in range(self._count):
                row[j] += self._delta(self._field_x[i], self._field_x[j], sigma, ells)
            row[i] += noise[i]
            rows.append(row)
        return mean, tuple(tuple(row) for row in rows), theta, noise

    def log_conditional_likelihood(self, phi: Sequence[float]) -> float:
        """Return ``log p(y | z, theta, psi)`` [EQ-VAL-KOHEM-003].

        Args:
            phi: Parameter vector (``theta`` inside its box).

        Returns:
            The Gaussian log density of the field data given the runs.

        Raises:
            ValueError: If ``phi`` is invalid or ``S`` is not numerically
                positive definite.

        Equation:
            [EQ-VAL-KOHEM-003].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        mean, covariance, _, _ = self._field_covariance(phi)
        try:
            lower = cholesky(covariance)
        except ValueError as error:
            raise ValueError(
                "conditional field covariance is not positive definite"
            ) from error
        residual = tuple(y - m for y, m in zip(self._field_y, mean, strict=True))
        white = solve_lower(lower, residual)
        log_det = 2.0 * math.fsum(math.log(lower[k][k]) for k in range(self._count))
        return -0.5 * (
            self._count * _LOG_TWO_PI + log_det + math.fsum(v * v for v in white)
        )

    def log_joint_likelihood(self, phi: Sequence[float]) -> float:
        """Return ``log p(z, y | theta, psi)``, the full joint log density.

        Equals :attr:`simulator_log_likelihood` plus
        :meth:`log_conditional_likelihood` [EQ-VAL-KOHEM-002].

        Args:
            phi: Parameter vector.

        Returns:
            Joint Gaussian log density of simulator runs and field data.

        Raises:
            ValueError: As :meth:`log_conditional_likelihood`.

        Equation:
            [EQ-VAL-KOHEM-002].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        return self.simulator_log_likelihood + self.log_conditional_likelihood(phi)

    def log_posterior(self, phi: Sequence[float]) -> float:
        """Return the unnormalized log posterior [EQ-VAL-KOHEM-004].

        Args:
            phi: Parameter vector.

        Returns:
            ``log prior + log p(y | z, theta, psi)`` (``-inf`` outside the
            prior support).

        Raises:
            ValueError: If the covariance is not numerically positive
                definite inside the support.

        Equation:
            [EQ-VAL-KOHEM-004].

        References:
            [REF-HIGDON-ETAL-2004].

        """
        prior = self.log_prior(phi)
        if prior == -math.inf:
            return -math.inf
        return prior + self.log_conditional_likelihood(phi)

    def _guarded(self, function: str, phi: Sequence[float]) -> float:
        """Evaluate a density, counting numerical failures as zero density.

        Args:
            function: ``"posterior"`` or ``"likelihood"``.
            phi: Parameter vector.

        Returns:
            The log density, or ``-inf`` after a counted failure.

        References:
            [EQ-VAL-KOHEM-004].

        """
        self._evaluations += 1
        try:
            if function == "posterior":
                return self.log_posterior(phi)
            return self.log_conditional_likelihood(phi)
        except ValueError:
            self._failures += 1
            return -math.inf

    def _posterior_target(self, phi: Vector) -> float:
        """Sampler target: the log posterior with counted failures.

        Args:
            phi: Parameter vector.

        Returns:
            Log posterior or ``-inf``.

        References:
            [EQ-VAL-KOHEM-004].

        """
        return self._guarded("posterior", phi)

    def _likelihood_target(self, phi: Vector) -> float:
        """TMCMC target: the conditional log likelihood with counted failures.

        Args:
            phi: Parameter vector.

        Returns:
            Log likelihood or ``-inf``.

        References:
            [EQ-VAL-KOHEM-004].

        """
        return self._guarded("likelihood", phi)

    def _prior_draw(self, rng: random.Random) -> Vector:
        """Draw one parameter vector from the prior.

        Args:
            rng: A ``random.Random``.

        Returns:
            Prior draw.

        References:
            [EQ-VAL-KOHEM-004].

        """
        draw = [rng.uniform(low, high) for low, high in self.theta_bounds]
        if self.discrepancy:
            draw.append(rng.gauss(self.amplitude_prior.mean, self.amplitude_prior.sd))
            draw.extend(
                rng.gauss(self.length_scale_prior.mean, self.length_scale_prior.sd)
                for _ in range(self._p)
            )
        if self.noise_prior is not None:
            draw.append(rng.gauss(self.noise_prior.mean, self.noise_prior.sd))
        return tuple(draw)

    def sample(
        self,
        *,
        sampler: str = "slice",
        draws: int = 400,
        warmup: int = 200,
        chains: int = 2,
        thin: int = 1,
        particles: int = 500,
        rhat_limit: float = 1.1,
        min_ess: float = 50.0,
        seed: int | RandomStream | None = None,
    ) -> KOHResult:
        """Sample the posterior of ``theta`` and the hyperparameters.

        Slice sampling starts each chain from a prior draw (over-dispersed
        starts) and reports split R-hat and bulk ESS; transitional MCMC also
        returns the log marginal likelihood of the field data given the runs.

        Args:
            sampler: ``"slice"`` or ``"tmcmc"``.
            draws: Kept draws per chain (slice).
            warmup: Discarded iterations per chain (slice).
            chains: Chains (slice; at least 2 for R-hat).
            thin: Thinning (slice).
            particles: TMCMC sample size.
            rhat_limit: Largest accepted split R-hat (slice).
            min_ess: Smallest accepted bulk ESS (slice).
            seed: Integer seed or stream.

        Returns:
            :class:`KOHResult`; ``converged`` is ``False`` with the reason in
            ``stop_reason`` when a diagnostic fails (draws are still returned).

        Raises:
            ValueError: For invalid options or when no prior draw has a
                finite posterior density.

        Equation:
            [EQ-VAL-KOHEM-004].

        References:
            [REF-NEAL-2003], [REF-CHING-CHEN-2007].

        """
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}")
        stream = resolve_stream(seed)
        start_failures = self._failures
        start_evaluations = self._evaluations
        if sampler == "tmcmc":
            tm = transitional_mcmc(
                self._prior_draw,
                self.log_prior,
                self._likelihood_target,
                particles=particles,
                seed=stream.child("tmcmc"),
                names=self.names,
            )
            return KOHResult(
                names=self.names,
                samples=tm.particles,
                theta_dimension=self._q,
                sampler="tmcmc",
                discrepancy=self.discrepancy,
                emulator_mode=self.emulator_mode,
                converged=tm.converged,
                stop_reason=tm.stop_reason,
                rhat_max=math.nan,
                ess_min=math.nan,
                evaluations=self._evaluations - start_evaluations,
                failed_evaluations=self._failures - start_failures,
                log_evidence=tm.log_evidence,
                mcmc=None,
            )
        if chains < 2:
            raise ValueError("slice sampling needs at least two chains for R-hat")
        generator = stream.child("starts").generator()
        starts: list[Vector] = []
        for _ in range(chains):
            for _attempt in range(100):
                candidate = self._prior_draw(generator)
                if self._guarded("posterior", candidate) > -math.inf:
                    starts.append(candidate)
                    break
            else:
                raise ValueError("no prior draw with finite posterior density")
        widths = [0.25 * (high - low) for low, high in self.theta_bounds]
        widths.extend([0.5] * (self.dimension - self._q))
        mcmc = slice_sampler(
            self._posterior_target,
            starts,
            draws=draws,
            warmup=warmup,
            chains=chains,
            thin=thin,
            widths=widths,
            seed=stream.child("slice"),
            names=self.names,
            check_convergence=False,
        )
        rhat = 0.0
        ess = math.inf
        for index in range(self.dimension):
            columns = mcmc.chain_columns(index)
            rhat = max(rhat, split_rhat(columns))
            ess = min(ess, bulk_effective_sample_size(columns))
        if not rhat <= rhat_limit:
            reason = "rhat"
        elif not ess >= min_ess:
            reason = "ess"
        else:
            reason = "converged"
        return KOHResult(
            names=self.names,
            samples=mcmc.pooled(),
            theta_dimension=self._q,
            sampler="slice",
            discrepancy=self.discrepancy,
            emulator_mode=self.emulator_mode,
            converged=reason == "converged",
            stop_reason=reason,
            rhat_max=rhat,
            ess_min=ess,
            evaluations=self._evaluations - start_evaluations,
            failed_evaluations=self._failures - start_failures,
            log_evidence=math.nan,
            mcmc=mcmc,
        )

    def without_discrepancy(self) -> KOHEmulatorCalibration:
        """Return the same problem with ``delta = 0`` (no refit of the emulator).

        Returns:
            A new calibration sharing data and the fitted emulator.

        References:
            [REF-BRYNJARSDOTTIR-OHAGAN-2014], [EQ-VAL-KOHEM-007].

        """
        return KOHEmulatorCalibration(
            tuple(point[: self._p] for point in self._joint_points),
            tuple(point[self._p :] for point in self._joint_points),
            self._outputs,
            self._field_x,
            self._field_y,
            theta_bounds=self.theta_bounds,
            noise_sd=None if self._noise_known is None else self._noise_known,
            noise_prior=self.noise_prior,
            discrepancy=False,
            discrepancy_kernel=self.discrepancy_kernel,
            emulator=self.emulator,
        )

    def joint_covariance(self, phi: Sequence[float]) -> Matrix:
        """Return the full ``(n + N) x (n + N)`` covariance of ``(z, y)``.

        Args:
            phi: Parameter vector.

        Returns:
            ``Sigma`` of [EQ-VAL-KOHEM-002] (runs first, then field data).

        Raises:
            ValueError: If ``phi`` is invalid.

        Equation:
            [EQ-VAL-KOHEM-002].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        theta, sigma, ells, noise = self.unpack(phi)
        size = self._runs + self._count
        points = self._joint_points + tuple(
            self._theta_point(i, theta) for i in range(self._count)
        )
        rows = [[0.0] * size for _ in range(size)]
        for a in range(size):
            for b in range(a + 1):
                value = self._eta(points[a], points[b])
                if a >= self._runs and b >= self._runs:
                    i = a - self._runs
                    j = b - self._runs
                    value += self._delta(
                        self._field_x[i], self._field_x[j], sigma, ells
                    )
                    if i == j:
                        value += noise[i]
                elif a == b:
                    value += self._nugget
                rows[a][b] = value
                rows[b][a] = value
        return tuple(tuple(row) for row in rows)

    def predict(
        self,
        result: KOHResult,
        controls: Sequence[Sequence[float]],
        *,
        include_noise: bool = True,
        noise_sd: float | None = None,
        level: float = 0.95,
        max_draws: int = 100,
    ) -> KOHPrediction:
        """Predict reality at new control inputs with a variance decomposition.

        For each (evenly spaced) posterior draw the Gaussian conditional of
        ``zeta* = eta(x*, theta) + delta*`` given ``(z, y)`` is computed from
        the full joint covariance [EQ-VAL-KOHEM-005]; the draws are then
        combined by the law of total variance [EQ-VAL-KOHEM-006].

        Args:
            result: A :class:`KOHResult` of this calibration.
            controls: ``m x p`` new control inputs.
            include_noise: Predict a new observation (add the noise part).
            noise_sd: Noise sd of the new observations; required for
                ``include_noise`` when the noise is known per observation with
                unequal values, ignored when the noise is estimated.
            level: Central interval level of the mixture.
            max_draws: Largest number of posterior draws used.

        Returns:
            :class:`KOHPrediction`.

        Raises:
            ValueError: For a result of another problem, invalid controls,
                a missing noise sd, a variance more negative than rounding
                allows, or a non-positive-definite joint covariance.

        Equation:
            [EQ-VAL-KOHEM-005], [EQ-VAL-KOHEM-006].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004].

        """
        if result.names != self.names:
            raise ValueError("result belongs to a different calibration")
        new_x = _matrix_rows(controls, "controls")
        if len(new_x[0]) != self._p:
            raise ValueError("controls must have the simulator control width")
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        if isinstance(max_draws, bool) or not isinstance(max_draws, int):
            raise ValueError("max_draws must be a positive integer")
        if max_draws < 1:
            raise ValueError("max_draws must be a positive integer")
        fixed_noise: float | None = None
        if include_noise and self.noise_prior is None:
            if noise_sd is not None:
                fixed_noise = require_finite_real(noise_sd, name="noise_sd")
                if fixed_noise < 0.0:
                    raise ValueError("noise_sd must be non-negative")
            elif self._noise_scalar and self._noise_known is not None:
                fixed_noise = self._noise_known[0]
            else:
                raise ValueError("noise_sd is required for unequal known noise")
        total_draws = len(result.samples)
        used = min(max_draws, total_draws)
        indices = sorted({(k * total_draws) // used for k in range(used)})
        stacked = tuple(self._outputs) + tuple(self._field_y)
        centred = tuple(v - self._m for v in stacked)
        m_new = len(new_x)
        means: list[list[float]] = [[] for _ in range(m_new)]
        variances: list[list[float]] = [[] for _ in range(m_new)]
        v_eta = [0.0] * m_new
        v_delta = [0.0] * m_new
        cov_ed = [0.0] * m_new
        noise_part = [0.0] * m_new
        for index in indices:
            phi = result.samples[index]
            lower = cholesky(self.joint_covariance(phi))
            residual = solve_lower(lower, centred)
            theta, sigma, ells, _ = self.unpack(phi)
            if include_noise:
                if self.noise_prior is not None:
                    noise_variance = math.exp(2.0 * phi[-1])
                else:
                    assert fixed_noise is not None
                    noise_variance = fixed_noise**2
            else:
                noise_variance = 0.0
            for n, x in enumerate(new_x):
                point = x + theta
                c_eta = tuple(
                    self._eta(point, run) for run in self._joint_points
                ) + tuple(
                    self._eta(point, self._theta_point(i, theta))
                    for i in range(self._count)
                )
                c_delta = (0.0,) * self._runs + tuple(
                    self._delta(x, self._field_x[i], sigma, ells)
                    for i in range(self._count)
                )
                w_eta = solve_lower(lower, c_eta)
                w_delta = solve_lower(lower, c_delta)
                mean = self._m + math.fsum(
                    (a + b) * r
                    for a, b, r in zip(w_eta, w_delta, residual, strict=True)
                )
                ve = self._signal - math.fsum(a * a for a in w_eta)
                vd = sigma * sigma - math.fsum(b * b for b in w_delta)
                ce = -math.fsum(a * b for a, b in zip(w_eta, w_delta, strict=True))
                scale = self._signal + sigma * sigma
                for label, value in (("emulator", ve), ("discrepancy", vd)):
                    if value < -_VARIANCE_ROUNDING * scale:
                        raise ValueError(
                            f"conditional {label} variance is negative beyond "
                            "rounding; the joint covariance is ill-conditioned"
                        )
                ve = max(ve, 0.0)
                vd = max(vd, 0.0)
                means[n].append(mean)
                variances[n].append(ve + vd + 2.0 * ce + noise_variance)
                v_eta[n] += ve
                v_delta[n] += vd
                cov_ed[n] += 2.0 * ce
                noise_part[n] += noise_variance
        count = len(indices)
        out_mean: list[float] = []
        parameter: list[float] = []
        total: list[float] = []
        lower_ends: list[float] = []
        upper_ends: list[float] = []
        tail = 0.5 * (1.0 - level)
        for n in range(m_new):
            centre = math.fsum(means[n]) / count
            spread = math.fsum((m - centre) ** 2 for m in means[n]) / count
            mixture = spread + math.fsum(variances[n]) / count
            out_mean.append(centre)
            parameter.append(spread)
            total.append(mixture)
            sds = [math.sqrt(max(v, 0.0)) for v in variances[n]]
            lower_ends.append(_mixture_quantile(tail, means[n], sds))
            upper_ends.append(_mixture_quantile(1.0 - tail, means[n], sds))
        return KOHPrediction(
            controls=new_x,
            mean=tuple(out_mean),
            emulator=tuple(v / count for v in v_eta),
            discrepancy=tuple(v / count for v in v_delta),
            covariance=tuple(v / count for v in cov_ed),
            parameter=tuple(parameter),
            noise=tuple(v / count for v in noise_part),
            total_variance=tuple(total),
            lower=tuple(lower_ends),
            upper=tuple(upper_ends),
            level=level,
            draws=count,
            includes_noise=include_noise,
        )

    def identifiability(
        self,
        result: KOHResult,
        *,
        reference: KOHResult | None = None,
        truth: Sequence[float] | None = None,
        contraction_threshold: float = 0.5,
    ) -> IdentifiabilityReport:
        """Report prior-to-posterior contraction and confounding of ``theta``.

        Args:
            result: Posterior of this calibration.
            reference: Posterior of the no-discrepancy calibration of the
                same data (see :meth:`without_discrepancy`), for the
                comparison fields.
            truth: Known synthetic ``theta`` for the ``truth_z_*`` fields.
            contraction_threshold: ``weakly_identified`` below this
                contraction, in ``[0, 1]``.

        Returns:
            :class:`IdentifiabilityReport`.

        Raises:
            ValueError: For results of another problem, a wrong-length truth
                or an invalid threshold.

        Equation:
            [EQ-VAL-KOHEM-007].

        References:
            [REF-BRYNJARSDOTTIR-OHAGAN-2014].

        """
        if result.names != self.names:
            raise ValueError("result belongs to a different calibration")
        if not 0.0 <= contraction_threshold <= 1.0:
            raise ValueError("contraction_threshold must lie in [0, 1]")
        q = self._q
        if reference is not None and reference.theta_dimension != q:
            raise ValueError("reference has a different theta dimension")
        if truth is not None and len(as_vector(truth, name="truth")) != q:
            raise ValueError("truth must have one entry per theta coordinate")
        prior_sd = tuple(
            (high - low) / math.sqrt(12.0) for low, high in self.theta_bounds
        )
        post_sd = result.sd()[:q]
        post_mean = result.mean()[:q]
        contraction = tuple(
            1.0 - (s / s0) ** 2 for s, s0 in zip(post_sd, prior_sd, strict=True)
        )
        amplitude: list[float] = []
        lengths: list[Vector] = []
        for k in range(q):
            if self.discrepancy:
                column = result.column(k)
                amplitude.append(_correlation_coefficient(column, result.column(q)))
                lengths.append(
                    tuple(
                        _correlation_coefficient(column, result.column(q + 1 + a))
                        for a in range(self._p)
                    )
                )
            else:
                amplitude.append(math.nan)
                lengths.append(())
        ratio: Vector = ()
        shift: Vector = ()
        over: tuple[bool, ...] = ()
        z_without: Vector = ()
        if reference is not None:
            ref_sd = reference.sd()[:q]
            ref_mean = reference.mean()[:q]
            ratio = tuple(a / b for a, b in zip(post_sd, ref_sd, strict=True))
            shift = tuple(
                (a - b) / s for a, b, s in zip(post_mean, ref_mean, ref_sd, strict=True)
            )
            over = tuple(abs(value) > 2.0 for value in shift)
            if truth is not None:
                z_without = tuple(
                    (m - t) / s for m, t, s in zip(ref_mean, truth, ref_sd, strict=True)
                )
        z_with: Vector = ()
        if truth is not None:
            z_with = tuple(
                (m - t) / s for m, t, s in zip(post_mean, truth, post_sd, strict=True)
            )
        return IdentifiabilityReport(
            theta_prior_sd=prior_sd,
            theta_posterior_sd=post_sd,
            contraction=contraction,
            weakly_identified=tuple(c < contraction_threshold for c in contraction),
            contraction_threshold=contraction_threshold,
            amplitude_correlation=tuple(amplitude),
            length_scale_correlation=tuple(lengths),
            sd_ratio=ratio,
            mean_shift_in_reference_sd=shift,
            reference_overconfident=over,
            truth_z_with=z_with,
            truth_z_without=z_without,
        )
