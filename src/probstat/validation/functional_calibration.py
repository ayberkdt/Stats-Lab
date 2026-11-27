"""Calibration of a simulator with field-valued output (functional output).

**Experimental (added in 0.28).** Synthetic-data workflows only; nothing here
describes a real simulator.

The simulator ``eta(x, t)`` maps a control input ``x`` in ``R^p`` and a
calibration parameter ``t`` in ``R^q`` to a *field* ``eta(x, t)(s_l)``,
``l = 1..m``, on a one-dimensional grid of locations ``s_l`` (a temperature
or pressure profile, say). The method follows the high-dimensional-output
calibration of Higdon, Gattiker, Williams and Rightley (2008), restricted to
the modular, reduced-data form described below.

**Basis** [EQ-VAL-FUNCEM-001]. The ``n`` simulator fields ``z_i`` are centred
per location, ``mu_l = mean_i z_i(s_l)``, and divided by one global scale
``c = sqrt(sum_il (z_i(s_l) - mu_l)^2 / (n m))`` (a single scale keeps
isotropic noise isotropic). With the singular value decomposition
``Z = U S Phi'`` of the scaled matrix ``Z_il = (z_i(s_l) - mu_l) / c`` the
first ``p_eta`` right singular vectors ``Phi`` (columns, orthonormal on the
grid) form the basis and ``w_ij = (Phi' Z_i)_j`` are the weights. The number
of components is the smallest ``p_eta`` with ``sum_{j<=p_eta} S_j^2 >=
threshold sum_j S_j^2`` (or is given). The truncation error is reported as the
omitted energy fraction and as the per-location residual variance
``tau_l^2 = c^2 mean_i (Z_il - (Phi w_i)_l)^2``.

**Emulator** [EQ-VAL-FUNCEM-002]. Each weight ``w_j(x, t)`` gets an
independent Gaussian process with constant mean and an ARD kernel fitted to the
runs alone (library routine :func:`probstat.surrogates.fit_gaussian_process`).
The weights are uncorrelated across the runs by construction of the SVD but
are *modelled* as independent processes; this is an approximation. The field
emulator is Gaussian: ``eta(x, t)(s_l) = mu_l + c (Phi w(x, t))_l`` with
covariance ``c^2 Phi diag(v_j) Phi'`` where ``v_j`` is the GP predictive
variance, plus the truncation variance ``tau_l^2`` as a *separate* diagonal
term, independent of ``(x, t)`` (exact on the training grid, where the
residual is orthogonal to the basis; an approximation elsewhere). The
emulator hyperparameters are fixed at that fit (``emulator_mode =
"modular"``); the fully joint refit with field data is not implemented.

**Reduced data** [EQ-VAL-FUNCEM-003]. The field observation
``y_e(s^f_l)`` at control ``x_e`` is on the same grid, or on another grid
inside the simulator range, in which case the basis and the mean are
linearly interpolated to the field grid (``Phi_f``, ``mu_f``). With
``G = Phi_f' Phi_f`` (the identity on the simulator grid) the *projected
data* is the generalized least-squares coefficient vector

``a_e = G^-1 Phi_f' (y_e - mu_f) / c``.

The model ``y_e = eta(x_e, theta) + delta_e + eps_e``, ``eps_e ~ N(0, s^2 I)``,
projected the same way, reads ``a_e = w(x_e, theta) + P v_e + e_e`` with
``P = G^-1 Phi_f' D / c``, ``D`` the discrepancy basis on the field grid and
``e_e ~ N(0, (s / c)^2 G^-1)`` (exactly, for iid noise).

**What is approximated.** The likelihood is that of the projected data ``a``
only. The component of the data orthogonal to the span of ``Phi_f`` is
discarded: it is uninformative about ``theta`` when the truncation error is
small, but it also carries information about the noise level, so an estimated
noise sd is determined from ``p_eta E`` numbers (``E`` experiments) only, and
a discrepancy that is orthogonal to the basis is invisible. The emulator
truncation term does not enter the likelihood (it is orthogonal to the span on
the simulator grid).

**Discrepancy** [EQ-VAL-FUNCEM-004]. ``delta_e = D v_e`` in a small user basis
``D`` (columns, for example Gaussian bumps from :func:`gaussian_bump_basis`)
with ``v ~ N(0, sigma_delta^2 I)``, either shared by all experiments
(``shared_discrepancy = True``, a systematic model error) or independent per
experiment. Without a discrepancy basis ``delta = 0``.

**Posterior** [EQ-VAL-FUNCEM-005]. Conditional on the runs the projected data
are Gaussian with mean ``E[w(x_e, theta) | runs]`` and covariance
``S(theta) + P Sigma_v P' + R`` (emulator, discrepancy, noise), evaluated with
one Cholesky factorization per call (the emulator conditioning is cached for
the last ``theta``). The prior is uniform for ``theta`` on a box and
log-normal for the discrepancy sd and (optionally) the noise sd. Sampling is by
slice sampling or transitional MCMC; the result type is
:class:`probstat.validation.KOHResult`.

**Predictive** [EQ-VAL-FUNCEM-006]. For a draw ``(theta, psi)`` the latent
weights, discrepancy weights and the projected data are jointly Gaussian; the
field at a new control is ``mu_f + c Phi_f w* + D v*`` plus the truncation term
and (optionally) noise. The law of total variance over the draws splits the
pointwise variance into *emulator*, *truncation*, *discrepancy*, *covariance*
(emulator-discrepancy), *parameter* and *noise* parts that add up exactly.

**Diagnostics** [EQ-VAL-FUNCEM-007, EQ-VAL-FUNCEM-008]. Emulator validation on
held-out runs in the reduced space and in the full field; posterior contraction
``1 - Var_post / Var_prior`` of ``theta``; and a comparison with the scalar
Kennedy-O'Hagan calibration (:class:`probstat.validation.KOHEmulatorCalibration`)
applied to one output location, which measures what the field data gain.

Limitations: one-dimensional output grid; constant GP mean; no simulator scale
factor; the discrepancy depends on the output location only (not on ``x``);
cost per likelihood call is ``O(n E p_eta + (E p_eta)^3)``.

References:
    [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008], [REF-HIGDON-ETAL-2004],
    [REF-KENNEDY-OHAGAN-2001], [REF-BAYARRI-ETAL-2007],
    [REF-RASMUSSEN-WILLIAMS-2006], [REF-GOLUB-VANLOAN-2013].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import (
    bulk_effective_sample_size,
    slice_sampler,
    split_rhat,
    transitional_mcmc,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    matmul,
    singular_value_decomposition,
    solve_lower,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import standard_normal_cdf
from probstat.surrogates.gaussian_process import (
    KERNELS,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
    kernel_value,
)
from probstat.validation.koh_emulator import (
    SAMPLERS,
    KOHEmulatorCalibration,
    KOHResult,
    LogNormalPrior,
)

_LOG_TWO_PI = math.log(2.0 * math.pi)
_RANK_TOLERANCE = 1e-12
_VARIANCE_ROUNDING = 1e-8
_QUANTILE_ITERATIONS = 48


def _rows(
    rows: Sequence[Sequence[float]], name: str, *, allow_empty_width: bool = False
) -> tuple[Vector, ...]:
    """Validate a rectangular, finite matrix given as rows.

    Args:
        rows: Candidate rows.
        name: Name for messages.
        allow_empty_width: Accept rows of length zero (no control inputs).

    Returns:
        Rows as float tuples.

    Raises:
        ValueError: If there are no rows, the rows are ragged or empty (unless
            allowed) or an entry is not finite.

    References:
        [EQ-VAL-FUNCEM-003].

    """
    checked = tuple(as_vector(row, name=name) if len(row) else () for row in rows)
    if not checked:
        raise ValueError(f"{name} must be non-empty")
    width = len(checked[0])
    if width == 0 and not allow_empty_width:
        raise ValueError(f"{name} must be non-empty")
    if any(len(row) != width for row in checked):
        raise ValueError(f"{name} rows must have equal length")
    return checked


def _locations(values: Sequence[float], name: str) -> Vector:
    """Validate strictly increasing finite grid locations.

    Args:
        values: Candidate locations (at least two).
        name: Name for messages.

    Returns:
        The locations as floats.

    Raises:
        ValueError: If fewer than two, non-finite or not strictly increasing.

    References:
        [EQ-VAL-FUNCEM-001].

    """
    grid = as_vector(values, name=name)
    if len(grid) < 2:
        raise ValueError(f"{name} needs at least two locations")
    if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    return grid


def _interpolation(source: Vector, target: Vector) -> tuple[tuple[int, float], ...]:
    """Return linear interpolation stencils from ``source`` to ``target``.

    Args:
        source: Increasing source grid.
        target: Target locations inside the source range.

    Returns:
        For each target the pair ``(i, t)`` meaning ``(1 - t) f_i + t f_{i+1}``.

    Raises:
        ValueError: If a target lies outside the source range (no
            extrapolation).

    References:
        [EQ-VAL-FUNCEM-003].

    """
    stencils: list[tuple[int, float]] = []
    for point in target:
        if point < source[0] or point > source[-1]:
            raise ValueError("field locations must lie inside the simulator grid")
        low, high = 0, len(source) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if source[middle] <= point:
                low = middle
            else:
                high = middle
        t = (point - source[low]) / (source[low + 1] - source[low])
        stencils.append((low, t))
    return tuple(stencils)


def _interpolate(
    values: Sequence[float], stencils: tuple[tuple[int, float], ...]
) -> Vector:
    """Apply interpolation stencils to grid values.

    Args:
        values: Values on the source grid.
        stencils: Output of :func:`_interpolation`.

    Returns:
        Interpolated values.

    References:
        [EQ-VAL-FUNCEM-003].

    """
    return tuple((1.0 - t) * values[i] + t * values[i + 1] for i, t in stencils)


def gaussian_bump_basis(
    locations: Sequence[float], centers: Sequence[float], width: float
) -> Matrix:
    """Return a discrepancy basis of Gaussian bumps on a grid.

    Column ``k`` is ``exp(-0.5 ((s - center_k) / width)^2)`` evaluated at the
    grid locations (a smooth, local, unit-height bump) [EQ-VAL-FUNCEM-004].

    Args:
        locations: Output grid.
        centers: Bump centres.
        width: Bump standard deviation (positive, grid units).

    Returns:
        ``m x k`` matrix as rows over locations.

    Raises:
        ValueError: If the grid or centres are empty or invalid or ``width`` is
            not positive.

    References:
        [EQ-VAL-FUNCEM-004], [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

    """
    grid = as_vector(locations, name="locations")
    cs = as_vector(centers, name="centers")
    if not grid or not cs:
        raise ValueError("locations and centers must be non-empty")
    if require_finite_real(width, name="width") <= 0.0:
        raise ValueError("width must be positive")
    return tuple(
        tuple(math.exp(-0.5 * ((s - c) / width) ** 2) for c in cs) for s in grid
    )


@dataclass(frozen=True, slots=True)
class FieldBasis:
    """Principal-component basis of simulator output fields.

    Args:
        locations: Output grid ``s_l``.
        mean: Per-location mean of the runs (output units).
        scale: Global scale ``c`` (output units).
        modes: ``p_eta`` orthonormal modes ``Phi_j`` (each of length ``m``).
        singular_values: All singular values of the scaled, centred matrix.
        energy_fractions: Cumulative energy fractions, one per singular value.
        components: Number ``p_eta`` of retained modes.
        truncation_fraction: Omitted energy fraction ``sum_{j>p} S_j^2 /
            sum_j S_j^2`` (the relative squared truncation error of the runs).
        truncation_variance: Per-location variance ``tau_l^2`` of the
            training residual, in output units squared.
        weights: ``n x p_eta`` scaled weights of the runs.

    References:
        [EQ-VAL-FUNCEM-001], [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

    """

    locations: Vector
    mean: Vector
    scale: float
    modes: tuple[Vector, ...]
    singular_values: Vector
    energy_fractions: Vector
    components: int
    truncation_fraction: float
    truncation_variance: Vector
    weights: tuple[Vector, ...]

    def project(self, field: Sequence[float]) -> Vector:
        """Return the scaled weights ``Phi' (z - mu) / c`` of one field.

        Args:
            field: Field on the simulator grid (output units).

        Returns:
            ``p_eta`` weights.

        Raises:
            ValueError: If the length does not match the grid.

        References:
            [EQ-VAL-FUNCEM-001].

        """
        values = as_vector(field, name="field")
        if len(values) != len(self.locations):
            raise ValueError("field must have one value per grid location")
        centred = [(v - m) / self.scale for v, m in zip(values, self.mean, strict=True)]
        return tuple(
            math.fsum(a * b for a, b in zip(mode, centred, strict=True))
            for mode in self.modes
        )

    def reconstruct(self, weights: Sequence[float]) -> Vector:
        """Return the field ``mu + c Phi w`` of scaled weights.

        Args:
            weights: ``p_eta`` scaled weights.

        Returns:
            Field on the simulator grid (output units).

        Raises:
            ValueError: If the number of weights is not ``p_eta``.

        References:
            [EQ-VAL-FUNCEM-001].

        """
        w = as_vector(weights, name="weights")
        if len(w) != self.components:
            raise ValueError("weights must have one entry per component")
        return tuple(
            self.mean[loc]
            + self.scale
            * math.fsum(w[j] * self.modes[j][loc] for j in range(self.components))
            for loc in range(len(self.locations))
        )


def fit_field_basis(
    fields: Sequence[Sequence[float]],
    locations: Sequence[float] | None = None,
    *,
    variance_threshold: float = 0.999,
    components: int | None = None,
) -> FieldBasis:
    """Fit the principal-component basis of simulator fields [EQ-VAL-FUNCEM-001].

    Args:
        fields: ``n`` simulator fields of ``m`` values (one row per run).
        locations: Grid of ``m`` strictly increasing locations (default
            ``0, 1, ..., m - 1``).
        variance_threshold: Smallest cumulative energy fraction in ``(0, 1]``
            when ``components`` is ``None``.
        components: Fixed number of components (overrides the threshold).

    Returns:
        :class:`FieldBasis`.

    Raises:
        ValueError: If the data are invalid, fewer than two runs, all runs are
            identical, ``components`` exceeds the numerical rank, or the
            threshold is outside ``(0, 1]``.

    Equation:
        [EQ-VAL-FUNCEM-001].

    References:
        [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008], [REF-GOLUB-VANLOAN-2013].

    """
    data = _rows(fields, "fields")
    runs, count = len(data), len(data[0])
    if runs < 2:
        raise ValueError("at least two simulator runs are required")
    grid = (
        tuple(float(loc) for loc in range(count))
        if locations is None
        else _locations(locations, "locations")
    )
    if len(grid) != count:
        raise ValueError("locations must have one entry per field value")
    if not 0.0 < require_finite_real(variance_threshold, name="threshold") <= 1.0:
        raise ValueError("variance_threshold must lie in (0, 1]")
    mean = tuple(math.fsum(row[loc] for row in data) / runs for loc in range(count))
    energy = math.fsum(
        (row[loc] - mean[loc]) ** 2 for row in data for loc in range(count)
    )
    if energy <= 0.0:
        raise ValueError("simulator fields are identical; nothing to calibrate")
    scale = math.sqrt(energy / (runs * count))
    scaled = tuple(
        tuple((row[loc] - mean[loc]) / scale for loc in range(count)) for row in data
    )
    svd = singular_value_decomposition(scaled)
    sigma = svd.singular_values
    squares = [s * s for s in sigma]
    total = math.fsum(squares)
    cumulative: list[float] = []
    running = 0.0
    for value in squares:
        running += value
        cumulative.append(running / total)
    numerical = sum(1 for s in sigma if s > _RANK_TOLERANCE * sigma[0])
    if components is None:
        chosen = next(
            (k + 1 for k, c in enumerate(cumulative) if c >= variance_threshold),
            len(sigma),
        )
        chosen = min(chosen, numerical)
    else:
        if isinstance(components, bool) or not isinstance(components, int):
            raise ValueError("components must be a positive integer")
        if components < 1:
            raise ValueError("components must be a positive integer")
        if components > numerical:
            raise ValueError(
                f"components={components} exceeds the numerical rank {numerical}"
            )
        chosen = components
    modes = tuple(tuple(svd.v[loc][j] for loc in range(count)) for j in range(chosen))
    weights = tuple(
        tuple(
            math.fsum(a * b for a, b in zip(mode, row, strict=True)) for mode in modes
        )
        for row in scaled
    )
    residual = []
    for loc in range(count):
        residual.append(
            scale
            * scale
            * math.fsum(
                (
                    scaled[i][loc]
                    - math.fsum(weights[i][j] * modes[j][loc] for j in range(chosen))
                )
                ** 2
                for i in range(runs)
            )
            / runs
        )
    omitted = math.fsum(squares[chosen:]) / total
    return FieldBasis(
        locations=grid,
        mean=mean,
        scale=scale,
        modes=modes,
        singular_values=sigma,
        energy_fractions=tuple(cumulative),
        components=chosen,
        truncation_fraction=omitted,
        truncation_variance=tuple(residual),
        weights=weights,
    )


class _WeightProcess:
    """Conditional Gaussian process of one weight given the runs.

    Holds the fixed hyperparameters of a fitted regression and the Cholesky
    factor of the run covariance.

    Args:
        gp: Fitted regression of the weight on the joint ``(x, t)`` inputs.

    Raises:
        ValueError: If the run covariance is not positive definite.

    References:
        [EQ-VAL-FUNCEM-002], [REF-RASMUSSEN-WILLIAMS-2006].

    """

    def __init__(self, gp: GaussianProcessRegression) -> None:
        """Factorize the run covariance.

        Raises:
            ValueError: If the run covariance is not positive definite.

        References:
            [EQ-VAL-FUNCEM-002].

        """
        hyper = gp.hyperparameters
        self.mean = gp.output_mean
        self.signal = hyper.signal_variance
        self.scales = hyper.length_scales
        self.nugget = hyper.noise_variance
        self.kernel = gp.kernel
        self.points = gp.points
        size = len(gp.points)
        matrix = tuple(
            tuple(
                kernel_value(
                    self.kernel, gp.points[a], gp.points[b], self.scales, self.signal
                )
                + (self.nugget if a == b else 0.0)
                for b in range(size)
            )
            for a in range(size)
        )
        try:
            self.lower = cholesky(matrix)
        except ValueError as error:
            raise ValueError(
                "run covariance of a weight is not positive definite; use a "
                "larger emulator nugget or remove duplicate runs"
            ) from error
        centred = tuple(v - self.mean for v in gp.values)
        self.whitened = solve_lower(self.lower, centred)

    def conditional(self, points: Sequence[Vector]) -> tuple[Vector, Matrix]:
        """Return the joint conditional mean and covariance at ``points``.

        Args:
            points: Joint ``(x, t)`` inputs.

        Returns:
            Mean vector and covariance matrix of the latent weight function
            given the runs (without the nugget).

        References:
            [EQ-VAL-FUNCEM-002].

        """
        columns = [
            solve_lower(
                self.lower,
                tuple(
                    kernel_value(self.kernel, point, run, self.scales, self.signal)
                    for run in self.points
                ),
            )
            for point in points
        ]
        means = tuple(
            self.mean
            + math.fsum(a * b for a, b in zip(col, self.whitened, strict=True))
            for col in columns
        )
        count = len(points)
        rows = [[0.0] * count for _ in range(count)]
        for a in range(count):
            for b in range(a + 1):
                value = kernel_value(
                    self.kernel, points[a], points[b], self.scales, self.signal
                ) - math.fsum(
                    x * y for x, y in zip(columns[a], columns[b], strict=True)
                )
                rows[a][b] = value
                rows[b][a] = value
        return means, tuple(tuple(row) for row in rows)


@dataclass(frozen=True, slots=True)
class FieldEmulatorPrediction:
    """Gaussian emulator prediction of whole fields [EQ-VAL-FUNCEM-002].

    Args:
        weight_mean: ``N x p_eta`` predictive means of the weights.
        weight_variance: ``N x p_eta`` predictive variances of the weights.
        field_mean: ``N x m`` field means (output units).
        emulator_variance: ``N x m`` variance from the weight uncertainty,
            ``c^2 Phi diag(v) Phi'`` diagonal.
        truncation_variance: ``m`` basis-truncation variance (shared by all
            inputs), a separate term.
        total_variance: ``emulator_variance + truncation_variance``.

    References:
        [EQ-VAL-FUNCEM-002].

    """

    weight_mean: tuple[Vector, ...]
    weight_variance: tuple[Vector, ...]
    field_mean: tuple[Vector, ...]
    emulator_variance: tuple[Vector, ...]
    truncation_variance: Vector
    total_variance: tuple[Vector, ...]


@dataclass(frozen=True, slots=True)
class EmulatorValidation:
    """Held-out validation of a field emulator [EQ-VAL-FUNCEM-007].

    Args:
        holdout: Number of held-out runs.
        components: Number of weights.
        weight_rmse: Root-mean-square error of each weight prediction.
        weight_standardized_rmse: ``sqrt(mean z^2)`` per weight with ``z`` the
            error over the predictive sd (about 1 for a calibrated emulator).
        weight_coverage: Fraction of held-out weights inside the central
            predictive interval of ``level``.
        truncation_rmse: Root-mean-square reconstruction error of the held-out
            fields by the basis alone (output units).
        field_rmse: Root-mean-square error of the full-field prediction.
        field_relative_rmse: ``field_rmse`` over the root-mean-square
            deviation of the held-out fields from the training mean.
        field_coverage: Fraction of held-out field values inside the central
            interval of the total (emulator plus truncation) variance.
        level: Interval level.

    References:
        [EQ-VAL-FUNCEM-007].

    """

    holdout: int
    components: int
    weight_rmse: Vector
    weight_standardized_rmse: Vector
    weight_coverage: Vector
    truncation_rmse: float
    field_rmse: float
    field_relative_rmse: float
    field_coverage: float
    level: float


def _normal_quantile(probability: float) -> float:
    """Return the standard normal quantile by bisection.

    Args:
        probability: Probability in ``(0, 1)``.

    Returns:
        ``z`` with ``Phi(z) = probability`` to about ``1e-12``.

    References:
        [EQ-VAL-FUNCEM-007].

    """
    low, high = -12.0, 12.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if standard_normal_cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class FieldEmulator:
    """Gaussian emulator of a field-valued simulator [EQ-VAL-FUNCEM-002].

    Fits the principal-component basis of the run fields and an independent
    Gaussian process to each retained weight on the joint ``(x, t)`` input.

    Args:
        controls: ``n x p`` control inputs of the runs (rows may be empty when
            the simulator has no control input).
        parameters: ``n x q`` calibration parameters of the runs.
        fields: ``n x m`` output fields.
        locations: Output grid (default ``0..m-1``).
        variance_threshold: Energy fraction for the number of components.
        components: Fixed number of components.
        kernel: GP kernel (``"matern52"``, ``"squared_exponential"``,
            ``"matern32"``).
        nugget: GP nugget in standardized units.
        restarts: Optimizer restarts of each GP fit.
        regularization: Optional GP hyperparameter regularization.
        seed: Seed or stream for the GP fits.

    Raises:
        ValueError: For inconsistent shapes, a bad kernel, too few runs for the
            retained components, or a run covariance that is not positive
            definite.

    Equation:
        [EQ-VAL-FUNCEM-001], [EQ-VAL-FUNCEM-002].

    References:
        [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008],
        [REF-RASMUSSEN-WILLIAMS-2006].

    """

    def __init__(
        self,
        controls: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[float]],
        fields: Sequence[Sequence[float]],
        *,
        locations: Sequence[float] | None = None,
        variance_threshold: float = 0.999,
        components: int | None = None,
        kernel: str = "matern52",
        nugget: float = 1e-8,
        restarts: int = 2,
        regularization: GaussianProcessRegularization | None = None,
        seed: int | RandomStream | None = None,
    ) -> None:
        """Fit the basis and the weight processes.

        Raises:
            ValueError: See the class docstring.

        Equation:
            [EQ-VAL-FUNCEM-001], [EQ-VAL-FUNCEM-002].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        cx = _rows(controls, "controls", allow_empty_width=True)
        cp = _rows(parameters, "parameters")
        data = _rows(fields, "fields")
        if not len(cx) == len(cp) == len(data):
            raise ValueError("controls, parameters and fields need equal length")
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.basis = fit_field_basis(
            data,
            locations,
            variance_threshold=variance_threshold,
            components=components,
        )
        if len(data) <= self.basis.components:
            raise ValueError("need more runs than retained components")
        self.controls = cx
        self.parameters = cp
        self.fields = data
        self.kernel = kernel
        self.control_width = len(cx[0])
        self.parameter_width = len(cp[0])
        joint = tuple(c + t for c, t in zip(cx, cp, strict=True))
        stream = resolve_stream(seed)
        gps: list[GaussianProcessRegression] = []
        for j in range(self.basis.components):
            gps.append(
                fit_gaussian_process(
                    joint,
                    tuple(row[j] for row in self.basis.weights),
                    kernel=kernel,
                    nugget=nugget,
                    restarts=restarts,
                    regularization=regularization,
                    seed=stream.child(f"weight{j}"),
                )
            )
        self.gps: tuple[GaussianProcessRegression, ...] = tuple(gps)
        self._processes = tuple(_WeightProcess(gp) for gp in gps)

    def weight_moments(
        self, points: Sequence[Vector]
    ) -> tuple[tuple[Vector, ...], tuple[Matrix, ...]]:
        """Return the conditional weight means and covariances at joint inputs.

        Args:
            points: Joint ``(x, t)`` inputs.

        Returns:
            Per weight, the mean vector and the covariance matrix over the
            inputs.

        References:
            [EQ-VAL-FUNCEM-002].

        """
        moments = [process.conditional(points) for process in self._processes]
        return tuple(m for m, _ in moments), tuple(c for _, c in moments)

    def predict(
        self,
        controls: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[float]],
    ) -> FieldEmulatorPrediction:
        """Predict whole fields at new ``(x, t)`` inputs.

        Args:
            controls: ``N x p`` controls (rows may be empty if ``p = 0``).
            parameters: ``N x q`` calibration parameters.

        Returns:
            :class:`FieldEmulatorPrediction`.

        Raises:
            ValueError: If shapes do not match the training inputs.

        Equation:
            [EQ-VAL-FUNCEM-002].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        cx = _rows(controls, "controls", allow_empty_width=True)
        cp = _rows(parameters, "parameters")
        if len(cx) != len(cp):
            raise ValueError("controls and parameters need equal length")
        if len(cx[0]) != self.control_width or len(cp[0]) != self.parameter_width:
            raise ValueError("inputs must have the training widths")
        points = tuple(c + t for c, t in zip(cx, cp, strict=True))
        means, covs = self.weight_moments(points)
        count = len(points)
        basis = self.basis
        w_mean = tuple(
            tuple(means[j][i] for j in range(basis.components)) for i in range(count)
        )
        w_var = tuple(
            tuple(max(covs[j][i][i], 0.0) for j in range(basis.components))
            for i in range(count)
        )
        grid = len(basis.locations)
        c2 = basis.scale**2
        field_mean = tuple(basis.reconstruct(w) for w in w_mean)
        emulator = tuple(
            tuple(
                c2
                * math.fsum(
                    w_var[i][j] * basis.modes[j][loc] ** 2
                    for j in range(basis.components)
                )
                for loc in range(grid)
            )
            for i in range(count)
        )
        total = tuple(
            tuple(e + t for e, t in zip(row, basis.truncation_variance, strict=True))
            for row in emulator
        )
        return FieldEmulatorPrediction(
            weight_mean=w_mean,
            weight_variance=w_var,
            field_mean=field_mean,
            emulator_variance=emulator,
            truncation_variance=basis.truncation_variance,
            total_variance=total,
        )

    def validate(
        self,
        controls: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[float]],
        fields: Sequence[Sequence[float]],
        *,
        level: float = 0.95,
    ) -> EmulatorValidation:
        """Validate the emulator on held-out runs [EQ-VAL-FUNCEM-007].

        Reduced space: the held-out field is projected on the basis and
        compared with the predicted weights. Full field: the predicted field
        is compared with the held-out field, with intervals from the emulator
        plus truncation variance.

        Args:
            controls: ``N x p`` held-out controls.
            parameters: ``N x q`` held-out parameters.
            fields: ``N x m`` held-out simulator fields.
            level: Central interval level in ``(0, 1)``.

        Returns:
            :class:`EmulatorValidation`.

        Raises:
            ValueError: For invalid shapes or level.

        Equation:
            [EQ-VAL-FUNCEM-007].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        data = _rows(fields, "fields")
        if len(data[0]) != len(self.basis.locations):
            raise ValueError("fields must have one value per grid location")
        prediction = self.predict(controls, parameters)
        if len(data) != len(prediction.field_mean):
            raise ValueError("controls, parameters and fields need equal length")
        count = len(data)
        p = self.basis.components
        grid = len(self.basis.locations)
        z_crit = _normal_quantile(0.5 + 0.5 * level)
        weight_err = [[0.0] * count for _ in range(p)]
        weight_z = [[0.0] * count for _ in range(p)]
        weight_hit = [0] * p
        for i, row in enumerate(data):
            truth = self.basis.project(row)
            for j in range(p):
                err = prediction.weight_mean[i][j] - truth[j]
                sd = math.sqrt(prediction.weight_variance[i][j])
                weight_err[j][i] = err
                z = err / sd if sd > 0.0 else (0.0 if err == 0.0 else math.inf)
                weight_z[j][i] = z
                weight_hit[j] += abs(z) <= z_crit
        trunc_sq = 0.0
        field_sq = 0.0
        spread_sq = 0.0
        field_hit = 0
        for i, row in enumerate(data):
            recon = self.basis.reconstruct(self.basis.project(row))
            for loc in range(grid):
                trunc_sq += (recon[loc] - row[loc]) ** 2
                err = prediction.field_mean[i][loc] - row[loc]
                field_sq += err * err
                spread_sq += (row[loc] - self.basis.mean[loc]) ** 2
                sd = math.sqrt(prediction.total_variance[i][loc])
                field_hit += abs(err) <= z_crit * sd
        cells = count * grid
        return EmulatorValidation(
            holdout=count,
            components=p,
            weight_rmse=tuple(
                math.sqrt(math.fsum(e * e for e in weight_err[j]) / count)
                for j in range(p)
            ),
            weight_standardized_rmse=tuple(
                math.sqrt(math.fsum(z * z for z in weight_z[j]) / count)
                for j in range(p)
            ),
            weight_coverage=tuple(weight_hit[j] / count for j in range(p)),
            truncation_rmse=math.sqrt(trunc_sq / cells),
            field_rmse=math.sqrt(field_sq / cells),
            field_relative_rmse=(
                math.sqrt(field_sq / spread_sq) if spread_sq > 0.0 else math.nan
            ),
            field_coverage=field_hit / cells,
            level=level,
        )


@dataclass(frozen=True, slots=True)
class FieldPrediction:
    """Posterior predictive of the field with an exact variance decomposition.

    Every entry is ``N x m`` (one row per new control, one column per field
    location). The six parts add up to ``total_variance``
    [EQ-VAL-FUNCEM-006].

    Args:
        controls: New control inputs.
        locations: Field grid.
        mean: Predictive mean.
        lower: Lower end of the central interval of the Gaussian mixture.
        upper: Upper end.
        total_variance: Mixture variance.
        emulator: ``E[V_eta]``, emulator variance from the weights.
        truncation: Basis-truncation variance (separate term).
        discrepancy: ``E[V_delta]``.
        covariance: ``2 E[C_ed]``, emulator-discrepancy covariance part.
        parameter: Variance across draws of the conditional mean.
        noise: Noise variance when a new observation is predicted, else 0.
        level: Interval level.
        draws: Number of posterior draws used.
        includes_noise: Whether the noise part is included.

    References:
        [EQ-VAL-FUNCEM-006].

    """

    controls: tuple[Vector, ...]
    locations: Vector
    mean: tuple[Vector, ...]
    lower: tuple[Vector, ...]
    upper: tuple[Vector, ...]
    total_variance: tuple[Vector, ...]
    emulator: tuple[Vector, ...]
    truncation: tuple[Vector, ...]
    discrepancy: tuple[Vector, ...]
    covariance: tuple[Vector, ...]
    parameter: tuple[Vector, ...]
    noise: tuple[Vector, ...]
    level: float
    draws: int
    includes_noise: bool

    def component_sum(self) -> tuple[Vector, ...]:
        """Return the sum of the six variance parts per point and location.

        Returns:
            ``emulator + truncation + discrepancy + covariance + parameter +
            noise``.

        References:
            [EQ-VAL-FUNCEM-006].

        """
        return tuple(
            tuple(
                math.fsum(parts)
                for parts in zip(
                    self.emulator[n],
                    self.truncation[n],
                    self.discrepancy[n],
                    self.covariance[n],
                    self.parameter[n],
                    self.noise[n],
                    strict=True,
                )
            )
            for n in range(len(self.mean))
        )


@dataclass(frozen=True, slots=True)
class ScalarComparison:
    """Field calibration against the scalar one at a single output location.

    Both calibrations use the same experiments, prior box, noise model and
    discrepancy switch; the scalar one sees only the output at ``location``
    [EQ-VAL-FUNCEM-008].

    Args:
        location: Index of the grid location used by the scalar calibration.
        scalar_mean: Posterior mean of ``theta`` (scalar calibration).
        scalar_sd: Posterior sd of ``theta`` (scalar calibration).
        field_mean: Posterior mean of ``theta`` (field calibration).
        field_sd: Posterior sd of ``theta`` (field calibration).
        sd_ratio: ``scalar_sd / field_sd`` per coordinate; above one means the
            field data tightened the posterior.
        scalar_contraction: ``1 - Var_post / Var_prior`` (scalar).
        field_contraction: Same (field).
        scalar_result: The scalar :class:`KOHResult`.

    References:
        [EQ-VAL-FUNCEM-008].

    """

    location: int
    scalar_mean: Vector
    scalar_sd: Vector
    field_mean: Vector
    field_sd: Vector
    sd_ratio: Vector
    scalar_contraction: Vector
    field_contraction: Vector
    scalar_result: KOHResult


def _mixture_quantile_fast(
    probability: float, means: Sequence[float], sds: Sequence[float]
) -> float:
    """Return a quantile of an equally weighted Gaussian mixture by bisection.

    Args:
        probability: Probability in ``(0, 1)``.
        means: Component means.
        sds: Component standard deviations.

    Returns:
        The quantile, accurate to about ``1e-14`` of the bracket width.

    References:
        [EQ-VAL-FUNCEM-006].

    """
    weight = 1.0 / len(means)
    low = min(m - 9.0 * s for m, s in zip(means, sds, strict=True)) - 1e-12
    high = max(m + 9.0 * s for m, s in zip(means, sds, strict=True)) + 1e-12
    for _ in range(_QUANTILE_ITERATIONS):
        middle = 0.5 * (low + high)
        total = 0.0
        for mean, sd in zip(means, sds, strict=True):
            if sd > 0.0:
                total += standard_normal_cdf((middle - mean) / sd)
            elif middle >= mean:
                total += 1.0
        if total * weight < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class FunctionalCalibration:
    """Calibration of a field-valued simulator from projected field data.

    Parameter vectors are ``(theta, log sigma_delta, log s)`` where the last
    two groups exist only with a discrepancy basis and an unknown noise sd
    respectively; see :attr:`names`.

    Args:
        emulator: Fitted :class:`FieldEmulator`.
        field_controls: ``E x p`` controls of the field experiments (``None``
            allowed when the simulator has no control input).
        field_data: ``E x m_f`` field observations.
        theta_bounds: ``q`` pairs ``(low, high)``: the uniform prior box.
        noise_sd: Known iid noise sd in output units. Exactly one of
            ``noise_sd`` and ``noise_prior``.
        noise_prior: Log-normal prior of one unknown noise sd.
        field_locations: Grid of the observations inside the simulator grid
            (default: the simulator grid).
        discrepancy_basis: ``m_f x k`` basis ``D`` (rows over field locations)
            of the discrepancy, or ``None`` for no discrepancy.
        shared_discrepancy: One discrepancy ``v`` for all experiments
            (default) or independent ones.
        amplitude_prior: Prior of ``sigma_delta`` (output units). Default
            ``LogNormalPrior(log(0.1 c), 1)`` with ``c`` the field scale (a
            stated default; the identifiability depends on it).

    Raises:
        TypeError: If ``emulator`` or a prior has the wrong type.
        ValueError: For inconsistent shapes, invalid bounds or noise, a field
            grid outside the simulator grid, or a basis on the field grid that
            is rank deficient.

    Equation:
        [EQ-VAL-FUNCEM-003], [EQ-VAL-FUNCEM-004], [EQ-VAL-FUNCEM-005].

    References:
        [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008], [REF-HIGDON-ETAL-2004],
        [REF-KENNEDY-OHAGAN-2001].

    """

    def __init__(
        self,
        emulator: FieldEmulator,
        field_controls: Sequence[Sequence[float]] | None,
        field_data: Sequence[Sequence[float]],
        *,
        theta_bounds: Sequence[tuple[float, float]],
        noise_sd: float | None = None,
        noise_prior: LogNormalPrior | None = None,
        field_locations: Sequence[float] | None = None,
        discrepancy_basis: Sequence[Sequence[float]] | None = None,
        shared_discrepancy: bool = True,
        amplitude_prior: LogNormalPrior | None = None,
    ) -> None:
        """Validate the data and project it on the emulator basis.

        Raises:
            TypeError: If ``emulator`` or a prior has the wrong type.
            ValueError: See the class docstring.

        Equation:
            [EQ-VAL-FUNCEM-003], [EQ-VAL-FUNCEM-004].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        if not isinstance(emulator, FieldEmulator):
            raise TypeError("emulator must be a FieldEmulator")
        data = _rows(field_data, "field_data")
        count = len(data)
        width = emulator.control_width
        if field_controls is None:
            if width != 0:
                raise ValueError("field_controls are required for this emulator")
            controls: tuple[Vector, ...] = ((),) * count
        else:
            controls = _rows(field_controls, "field_controls", allow_empty_width=True)
            if len(controls) != count:
                raise ValueError("field_controls and field_data need equal length")
            if len(controls[0]) != width:
                raise ValueError("field_controls must have the simulator control width")
        bounds = tuple((float(a), float(b)) for a, b in theta_bounds)
        if len(bounds) != emulator.parameter_width:
            raise ValueError("theta_bounds must have one pair per parameter column")
        for low, high in bounds:
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("theta_bounds must be finite with low < high")
        if (noise_sd is None) == (noise_prior is None):
            raise ValueError("give exactly one of noise_sd and noise_prior")
        if noise_sd is not None and require_finite_real(noise_sd, name="noise_sd") <= 0:
            raise ValueError("noise_sd must be positive")
        if noise_prior is not None and not isinstance(noise_prior, LogNormalPrior):
            raise TypeError("noise_prior must be a LogNormalPrior")
        if amplitude_prior is not None and not isinstance(
            amplitude_prior, LogNormalPrior
        ):
            raise TypeError("amplitude_prior must be a LogNormalPrior")
        basis = emulator.basis
        sim_grid = basis.locations
        if field_locations is None:
            field_grid = sim_grid
        else:
            field_grid = _locations(field_locations, "field_locations")
        if len(data[0]) != len(field_grid):
            raise ValueError("field_data must have one value per field location")
        same = field_grid == sim_grid
        p = basis.components
        if same:
            modes_f = basis.modes
            mean_f = basis.mean
            trunc_f = basis.truncation_variance
        else:
            stencils = _interpolation(sim_grid, field_grid)
            modes_f = tuple(_interpolate(mode, stencils) for mode in basis.modes)
            mean_f = _interpolate(basis.mean, stencils)
            trunc_f = _interpolate(basis.truncation_variance, stencils)
        gram = tuple(
            tuple(
                math.fsum(a * b for a, b in zip(modes_f[i], modes_f[j], strict=True))
                for j in range(p)
            )
            for i in range(p)
        )
        eigenvalues, _ = symmetric_eigen(gram)
        if eigenvalues[0] <= _RANK_TOLERANCE * eigenvalues[-1]:
            raise ValueError(
                "basis restricted to the field grid is rank deficient; "
                "use more field locations or fewer components"
            )
        self.projection_condition = eigenvalues[-1] / eigenvalues[0]
        gram_inverse = inverse_spd(gram)
        self.emulator = emulator
        self.field_locations: Vector = field_grid
        self.field_controls = controls
        self.field_data = data
        self.theta_bounds = bounds
        self.noise_sd = None if noise_sd is None else float(noise_sd)
        self.noise_prior = noise_prior
        self.shared_discrepancy = bool(shared_discrepancy)
        self.emulator_mode = "modular"
        self._count = count
        self._p = p
        self._q = emulator.parameter_width
        self._scale = basis.scale
        self._modes_f = modes_f
        self._mean_f = mean_f
        self._trunc_f = trunc_f
        self._gram_inverse = gram_inverse
        scale = basis.scale
        self.reduced_data: tuple[Vector, ...] = tuple(
            tuple(
                math.fsum(
                    gram_inverse[j][i]
                    * math.fsum(
                        modes_f[i][loc] * (row[loc] - mean_f[loc]) / scale
                        for loc in range(len(field_grid))
                    )
                    for i in range(p)
                )
                for j in range(p)
            )
            for row in data
        )
        self._basis_d: tuple[Vector, ...] | None = None
        self._projection: tuple[Vector, ...] = ()
        self._k = 0
        if discrepancy_basis is not None:
            basis_d = _rows(discrepancy_basis, "discrepancy_basis")
            if len(basis_d) != len(field_grid):
                raise ValueError("discrepancy_basis needs one row per field location")
            self._basis_d = basis_d
            self._k = len(basis_d[0])
            cross = tuple(
                tuple(
                    math.fsum(
                        modes_f[i][loc] * basis_d[loc][k] for loc in range(len(basis_d))
                    )
                    for k in range(self._k)
                )
                for i in range(p)
            )
            self._projection = tuple(
                tuple(
                    math.fsum(gram_inverse[j][i] * cross[i][k] for i in range(p))
                    / scale
                    for k in range(self._k)
                )
                for j in range(p)
            )
        self.discrepancy = self._basis_d is not None
        self.amplitude_prior = amplitude_prior or LogNormalPrior(
            math.log(0.1 * scale), 1.0
        )
        names = [f"theta{k}" for k in range(self._q)]
        if self.discrepancy:
            names.append("log_sigma_delta")
        if noise_prior is not None:
            names.append("log_noise_sd")
        self.names: tuple[str, ...] = tuple(names)
        self._cache: tuple[Vector, tuple[Vector, ...], tuple[Matrix, ...]] | None = None
        self._failures = 0
        self._evaluations = 0

    @property
    def dimension(self) -> int:
        """Return the length of the parameter vector.

        Returns:
            ``q`` plus the discrepancy and noise coordinates.

        References:
            [EQ-VAL-FUNCEM-005].

        """
        return len(self.names)

    def unpack(self, phi: Sequence[float]) -> tuple[Vector, float, float]:
        """Split a parameter vector into natural-scale parts.

        Args:
            phi: Parameter vector (see :attr:`names`).

        Returns:
            ``(theta, sigma_delta, noise_sd)``; ``sigma_delta`` is 0 without a
            discrepancy.

        Raises:
            ValueError: If ``phi`` has the wrong length or is not finite.

        References:
            [EQ-VAL-FUNCEM-005].

        """
        vector = as_vector(phi, name="phi")
        if len(vector) != self.dimension:
            raise ValueError(f"phi must have {self.dimension} entries")
        theta = vector[: self._q]
        cursor = self._q
        sigma = 0.0
        if self.discrepancy:
            sigma = math.exp(vector[cursor])
            cursor += 1
        if self.noise_prior is not None:
            noise = math.exp(vector[cursor])
        else:
            assert self.noise_sd is not None
            noise = self.noise_sd
        return theta, sigma, noise

    def log_prior(self, phi: Sequence[float]) -> float:
        """Return the normalized log prior density [EQ-VAL-FUNCEM-005].

        Args:
            phi: Parameter vector.

        Returns:
            Log prior (``-inf`` outside the ``theta`` box).

        Raises:
            ValueError: If ``phi`` is invalid.

        References:
            [EQ-VAL-FUNCEM-005].

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
            cursor += 1
        if self.noise_prior is not None:
            total += self.noise_prior.log_density(vector[cursor])
        return total

    def _joint_points(self, theta: Vector) -> tuple[Vector, ...]:
        """Return the emulator inputs ``(x_e, theta)`` of the experiments.

        Args:
            theta: Calibration parameter.

        Returns:
            One joint input per experiment.

        References:
            [EQ-VAL-FUNCEM-003].

        """
        return tuple(x + theta for x in self.field_controls)

    def _emulator_moments(
        self, theta: Vector
    ) -> tuple[tuple[Vector, ...], tuple[Matrix, ...]]:
        """Return the cached conditional weight moments at the experiments.

        Args:
            theta: Calibration parameter.

        Returns:
            Per weight, the mean over experiments and the covariance.

        References:
            [EQ-VAL-FUNCEM-002].

        """
        cached = self._cache
        if cached is not None and cached[0] == theta:
            return cached[1], cached[2]
        means, covs = self.emulator.weight_moments(self._joint_points(theta))
        self._cache = (theta, means, covs)
        return means, covs

    def _reduced_covariance(
        self, sigma: float, noise: float, covs: tuple[Matrix, ...]
    ) -> Matrix:
        """Assemble the covariance of the projected data [EQ-VAL-FUNCEM-005].

        Args:
            sigma: Discrepancy sd (output units).
            noise: Noise sd (output units).
            covs: Emulator weight covariances over the experiments.

        Returns:
            ``(E p) x (E p)`` covariance, ordered ``(experiment, weight)``.

        References:
            [EQ-VAL-FUNCEM-003], [EQ-VAL-FUNCEM-004].

        """
        p, count = self._p, self._count
        size = count * p
        rows = [[0.0] * size for _ in range(size)]
        for j in range(p):
            for e in range(count):
                for f in range(count):
                    rows[e * p + j][f * p + j] += covs[j][e][f]
        s2 = (noise / self._scale) ** 2
        for e in range(count):
            for i in range(p):
                for j in range(p):
                    rows[e * p + i][e * p + j] += s2 * self._gram_inverse[i][j]
        if self.discrepancy and sigma > 0.0:
            ppt = [
                [
                    math.fsum(
                        a * b
                        for a, b in zip(
                            self._projection[i], self._projection[j], strict=True
                        )
                    )
                    for j in range(p)
                ]
                for i in range(p)
            ]
            v2 = sigma * sigma
            for e in range(count):
                for f in range(count):
                    if e != f and not self.shared_discrepancy:
                        continue
                    for i in range(p):
                        for j in range(p):
                            rows[e * p + i][f * p + j] += v2 * ppt[i][j]
        return tuple(tuple(row) for row in rows)

    def log_likelihood(self, phi: Sequence[float]) -> float:
        """Return ``log p(a | runs, theta, psi)`` of the projected data.

        Args:
            phi: Parameter vector (``theta`` inside its box).

        Returns:
            Gaussian log density of the projected data given the runs
            [EQ-VAL-FUNCEM-005].

        Raises:
            ValueError: If ``phi`` is invalid or the covariance is not
                numerically positive definite.

        Equation:
            [EQ-VAL-FUNCEM-005].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        theta, sigma, noise = self.unpack(phi)
        means, covs = self._emulator_moments(theta)
        covariance = self._reduced_covariance(sigma, noise, covs)
        try:
            lower = cholesky(covariance)
        except ValueError as error:
            raise ValueError(
                "reduced-data covariance is not positive definite"
            ) from error
        residual = tuple(
            self.reduced_data[e][j] - means[j][e]
            for e in range(self._count)
            for j in range(self._p)
        )
        white = solve_lower(lower, residual)
        size = len(residual)
        log_det = 2.0 * math.fsum(math.log(lower[k][k]) for k in range(size))
        return -0.5 * (size * _LOG_TWO_PI + log_det + math.fsum(v * v for v in white))

    def log_posterior(self, phi: Sequence[float]) -> float:
        """Return the unnormalized log posterior [EQ-VAL-FUNCEM-005].

        Args:
            phi: Parameter vector.

        Returns:
            ``log prior + log likelihood`` (``-inf`` outside the support).

        Raises:
            ValueError: If the covariance is not numerically positive definite
                inside the support.

        Equation:
            [EQ-VAL-FUNCEM-005].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        prior = self.log_prior(phi)
        if prior == -math.inf:
            return -math.inf
        return prior + self.log_likelihood(phi)

    def _guarded(self, function: str, phi: Sequence[float]) -> float:
        """Evaluate a density, counting numerical failures as zero density.

        Args:
            function: ``"posterior"`` or ``"likelihood"``.
            phi: Parameter vector.

        Returns:
            The log density, or ``-inf`` after a counted failure.

        References:
            [EQ-VAL-FUNCEM-005].

        """
        self._evaluations += 1
        try:
            if function == "posterior":
                return self.log_posterior(phi)
            return self.log_likelihood(phi)
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
            [EQ-VAL-FUNCEM-005].

        """
        return self._guarded("posterior", phi)

    def _likelihood_target(self, phi: Vector) -> float:
        """TMCMC target: the log likelihood with counted failures.

        Args:
            phi: Parameter vector.

        Returns:
            Log likelihood or ``-inf``.

        References:
            [EQ-VAL-FUNCEM-005].

        """
        return self._guarded("likelihood", phi)

    def _prior_draw(self, rng: random.Random) -> Vector:
        """Draw one parameter vector from the prior.

        Args:
            rng: A ``random.Random``.

        Returns:
            Prior draw.

        References:
            [EQ-VAL-FUNCEM-005].

        """
        draw = [rng.uniform(low, high) for low, high in self.theta_bounds]
        if self.discrepancy:
            draw.append(rng.gauss(self.amplitude_prior.mean, self.amplitude_prior.sd))
        if self.noise_prior is not None:
            draw.append(rng.gauss(self.noise_prior.mean, self.noise_prior.sd))
        return tuple(draw)

    def sample(
        self,
        *,
        sampler: str = "slice",
        draws: int = 300,
        warmup: int = 150,
        chains: int = 2,
        thin: int = 1,
        particles: int = 400,
        rhat_limit: float = 1.1,
        min_ess: float = 50.0,
        seed: int | RandomStream | None = None,
    ) -> KOHResult:
        """Sample the posterior of ``theta`` and the hyperparameters.

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
            ValueError: For invalid options or when no prior draw has a finite
                posterior density.

        Equation:
            [EQ-VAL-FUNCEM-005].

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

    def contraction(self, result: KOHResult) -> Vector:
        """Return the posterior-to-prior variance contraction of ``theta``.

        Args:
            result: A result of this calibration.

        Returns:
            ``1 - Var_post / Var_prior`` per ``theta`` coordinate, with the
            uniform-box prior variance ``(high - low)^2 / 12`` [EQ-VAL-FUNCEM-008].

        Raises:
            ValueError: For a result of another calibration.

        References:
            [EQ-VAL-FUNCEM-008].

        """
        if result.names != self.names:
            raise ValueError("result belongs to a different calibration")
        sds = result.sd()
        return tuple(
            1.0 - sds[k] ** 2 / ((high - low) ** 2 / 12.0)
            for k, (low, high) in enumerate(self.theta_bounds)
        )

    def compare_with_scalar(
        self,
        result: KOHResult,
        location: int,
        *,
        sampler: str = "slice",
        draws: int = 300,
        warmup: int = 150,
        chains: int = 2,
        seed: int | RandomStream | None = None,
        emulator_restarts: int = 2,
    ) -> ScalarComparison:
        """Compare with the scalar KOH calibration at one output location.

        The scalar calibration receives the simulator output and the field
        observations at grid index ``location`` only, the same experiments,
        box, noise and discrepancy switch (its GP discrepancy is a function of
        the control input, unlike the basis discrepancy here), and fits its own
        emulator [EQ-VAL-FUNCEM-008].

        Args:
            result: Result of this field calibration.
            location: Grid index of the single output used.
            sampler: Sampler of the scalar calibration.
            draws: Kept draws per chain (slice).
            warmup: Warm-up per chain (slice).
            chains: Chains (slice).
            seed: Integer seed or stream for the scalar fit and sampler.
            emulator_restarts: Optimizer restarts of the scalar emulator.

        Returns:
            :class:`ScalarComparison`.

        Raises:
            ValueError: If the simulator has no control input, the field grid
                differs from the simulator grid, the index is invalid or the
                result belongs to another calibration.

        Equation:
            [EQ-VAL-FUNCEM-008].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004].

        """
        if result.names != self.names:
            raise ValueError("result belongs to a different calibration")
        emulator = self.emulator
        if emulator.control_width < 1:
            raise ValueError("scalar comparison needs at least one control input")
        if self.field_locations != emulator.basis.locations:
            raise ValueError("scalar comparison needs the field on the simulator grid")
        if not 0 <= location < len(self.field_locations):
            raise ValueError("location index out of range")
        scalar = KOHEmulatorCalibration(
            emulator.controls,
            emulator.parameters,
            tuple(row[location] for row in emulator.fields),
            self.field_controls,
            tuple(row[location] for row in self.field_data),
            theta_bounds=self.theta_bounds,
            noise_sd=self.noise_sd,
            noise_prior=self.noise_prior,
            discrepancy=self.discrepancy,
            emulator_restarts=emulator_restarts,
            seed=seed,
        )
        outcome = scalar.sample(
            sampler=sampler, draws=draws, warmup=warmup, chains=chains, seed=seed
        )
        q = self._q
        scalar_sd = outcome.sd()[:q]
        field_sd = result.sd()[:q]
        scalar_prior = tuple(
            (high - low) ** 2 / 12.0 for low, high in self.theta_bounds
        )
        return ScalarComparison(
            location=location,
            scalar_mean=outcome.mean()[:q],
            scalar_sd=scalar_sd,
            field_mean=result.mean()[:q],
            field_sd=field_sd,
            sd_ratio=tuple(a / b for a, b in zip(scalar_sd, field_sd, strict=True)),
            scalar_contraction=tuple(
                1.0 - s * s / v for s, v in zip(scalar_sd, scalar_prior, strict=True)
            ),
            field_contraction=tuple(
                1.0 - s * s / v for s, v in zip(field_sd, scalar_prior, strict=True)
            ),
            scalar_result=outcome,
        )

    def predict(
        self,
        result: KOHResult,
        controls: Sequence[Sequence[float]] | None = None,
        *,
        include_noise: bool = True,
        level: float = 0.95,
        max_draws: int = 60,
    ) -> FieldPrediction:
        """Predict the field at new controls with a variance decomposition.

        For each (evenly spaced) posterior draw the latent weights and
        discrepancy weights at the experiments and the new controls are
        conditioned on the projected data (Gaussian update), the field
        moments follow by linear maps, and the draws are combined by the law of
        total variance [EQ-VAL-FUNCEM-006].

        Args:
            result: A result of this calibration.
            controls: ``N x p`` new controls (``None`` for a simulator without
                controls: the field at ``theta``).
            include_noise: Predict a new observation (add the noise variance).
            level: Central interval level of the mixture.
            max_draws: Largest number of posterior draws used.

        Returns:
            :class:`FieldPrediction` on the field grid.

        Raises:
            ValueError: For a result of another calibration, invalid controls,
                level or draw count, a conditional variance more negative than
                rounding allows, or a non-positive-definite covariance.

        Equation:
            [EQ-VAL-FUNCEM-006].

        References:
            [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

        """
        if result.names != self.names:
            raise ValueError("result belongs to a different calibration")
        width = self.emulator.control_width
        if controls is None:
            if width != 0:
                raise ValueError("controls are required for this emulator")
            new_x: tuple[Vector, ...] = ((),)
        else:
            new_x = _rows(controls, "controls", allow_empty_width=True)
            if len(new_x[0]) != width:
                raise ValueError("controls must have the simulator control width")
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        if isinstance(max_draws, bool) or not isinstance(max_draws, int):
            raise ValueError("max_draws must be a positive integer")
        if max_draws < 1:
            raise ValueError("max_draws must be a positive integer")
        total_draws = len(result.samples)
        used = min(max_draws, total_draws)
        indices = sorted({(k * total_draws) // used for k in range(used)})
        grid = len(self.field_locations)
        count_new = len(new_x)
        means: list[list[list[float]]] = [
            [[] for _ in range(grid)] for _ in range(count_new)
        ]
        variances: list[list[list[float]]] = [
            [[] for _ in range(grid)] for _ in range(count_new)
        ]
        v_eta = [[0.0] * grid for _ in range(count_new)]
        v_delta = [[0.0] * grid for _ in range(count_new)]
        cov_ed = [[0.0] * grid for _ in range(count_new)]
        noise_part = [[0.0] * grid for _ in range(count_new)]
        for index in indices:
            phi = result.samples[index]
            theta, sigma, noise = self.unpack(phi)
            draw = self._draw_moments(theta, sigma, noise, new_x)
            for n in range(count_new):
                for loc in range(grid):
                    ve, vd, ce, mean = draw[n][loc]
                    nv = noise * noise if include_noise else 0.0
                    means[n][loc].append(mean)
                    variances[n][loc].append(ve + vd + ce + self._trunc_f[loc] + nv)
                    v_eta[n][loc] += ve
                    v_delta[n][loc] += vd
                    cov_ed[n][loc] += ce
                    noise_part[n][loc] += nv
        count = len(indices)
        tail = 0.5 * (1.0 - level)
        out_mean, lower, upper, total, parameter = [], [], [], [], []
        for n in range(count_new):
            row_mean, row_lo, row_hi, row_tot, row_par = [], [], [], [], []
            for loc in range(grid):
                centre = math.fsum(means[n][loc]) / count
                spread = math.fsum((m - centre) ** 2 for m in means[n][loc]) / count
                row_mean.append(centre)
                row_par.append(spread)
                row_tot.append(spread + math.fsum(variances[n][loc]) / count)
                sds = [math.sqrt(max(v, 0.0)) for v in variances[n][loc]]
                row_lo.append(_mixture_quantile_fast(tail, means[n][loc], sds))
                row_hi.append(_mixture_quantile_fast(1.0 - tail, means[n][loc], sds))
            out_mean.append(tuple(row_mean))
            lower.append(tuple(row_lo))
            upper.append(tuple(row_hi))
            total.append(tuple(row_tot))
            parameter.append(tuple(row_par))
        return FieldPrediction(
            controls=new_x,
            locations=self.field_locations,
            mean=tuple(out_mean),
            lower=tuple(lower),
            upper=tuple(upper),
            total_variance=tuple(total),
            emulator=tuple(tuple(v / count for v in row) for row in v_eta),
            truncation=tuple(self._trunc_f for _ in range(count_new)),
            discrepancy=tuple(tuple(v / count for v in row) for row in v_delta),
            covariance=tuple(tuple(v / count for v in row) for row in cov_ed),
            parameter=tuple(parameter),
            noise=tuple(tuple(v / count for v in row) for row in noise_part),
            level=level,
            draws=count,
            includes_noise=include_noise,
        )

    def _draw_moments(
        self,
        theta: Vector,
        sigma: float,
        noise: float,
        new_x: tuple[Vector, ...],
    ) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
        """Return per-location conditional moments of the field for one draw.

        Conditions the latent weights (emulator) and discrepancy weights at the
        experiments and at the new controls on the projected data.

        Args:
            theta: Calibration parameter.
            sigma: Discrepancy sd (output units).
            noise: Noise sd (output units).
            new_x: New controls.

        Returns:
            For each new control and location the tuple ``(emulator variance,
            discrepancy variance, 2 x covariance, mean)`` in output units.

        Raises:
            ValueError: If a variance is negative beyond rounding or the data
                covariance is not positive definite.

        References:
            [EQ-VAL-FUNCEM-006].

        """
        p, count, k = self._p, self._count, self._k if self.discrepancy else 0
        new_count = len(new_x)
        points = self._joint_points(theta) + tuple(x + theta for x in new_x)
        total_points = len(points)
        means_w, covs_w = self.emulator.weight_moments(points)
        shared = self.shared_discrepancy
        v_blocks = 1 if shared else total_points
        lw = total_points * p
        lv = k * v_blocks if self.discrepancy else 0
        size = lw + lv
        prior = [[0.0] * size for _ in range(size)]
        mean_l = [0.0] * size
        for j in range(p):
            for a in range(total_points):
                mean_l[a * p + j] = means_w[j][a]
                for b in range(total_points):
                    prior[a * p + j][b * p + j] = covs_w[j][a][b]
        if self.discrepancy:
            for t in range(lv):
                prior[lw + t][lw + t] = sigma * sigma
        obs = count * p
        h = [[0.0] * size for _ in range(obs)]
        for e in range(count):
            for j in range(p):
                h[e * p + j][e * p + j] = 1.0
                if self.discrepancy:
                    block = 0 if shared else e
                    for kk in range(k):
                        h[e * p + j][lw + block * k + kk] = self._projection[j][kk]
        noise_cov = [[0.0] * obs for _ in range(obs)]
        s2 = (noise / self._scale) ** 2
        for e in range(count):
            for i in range(p):
                for j in range(p):
                    noise_cov[e * p + i][e * p + j] = s2 * self._gram_inverse[i][j]
        h_t = transpose(tuple(tuple(r) for r in h))
        prior_t = tuple(tuple(r) for r in prior)
        cross = matmul(prior_t, h_t)
        data_cov = matmul(tuple(tuple(r) for r in h), cross)
        data_cov = tuple(
            tuple(data_cov[a][b] + noise_cov[a][b] for b in range(obs))
            for a in range(obs)
        )
        try:
            lower = cholesky(data_cov)
        except ValueError as error:
            raise ValueError(
                "reduced-data covariance is not positive definite"
            ) from error
        residual = tuple(
            self.reduced_data[e][j]
            - math.fsum(h[e * p + j][c] * mean_l[c] for c in range(size))
            for e in range(count)
            for j in range(p)
        )
        alpha = cholesky_solve(lower, residual)
        solved = [cholesky_solve(lower, tuple(cross[r])) for r in range(size)]
        out: list[tuple[tuple[float, float, float, float], ...]] = []
        grid = len(self.field_locations)
        scale = self._scale
        for n in range(new_count):
            point = count + n
            w_idx = [point * p + j for j in range(p)]
            v_idx = (
                [lw + (0 if shared else point) * k + kk for kk in range(k)]
                if self.discrepancy
                else []
            )
            q_idx = w_idx + v_idx
            post_mean = [
                mean_l[r]
                + math.fsum(a * b for a, b in zip(cross[r], alpha, strict=True))
                for r in q_idx
            ]
            post_cov = [
                [
                    prior[r][s]
                    - math.fsum(a * b for a, b in zip(cross[r], solved[s], strict=True))
                    for s in q_idx
                ]
                for r in q_idx
            ]
            row: list[tuple[float, float, float, float]] = []
            for loc in range(grid):
                phi_l = [self._modes_f[j][loc] for j in range(p)]
                d_l = (
                    [self._basis_d[loc][kk] / scale for kk in range(k)]  # type: ignore[index]
                    if self.discrepancy
                    else []
                )
                coeffs = phi_l + d_l
                mean_std = math.fsum(
                    c * m for c, m in zip(coeffs, post_mean, strict=True)
                )
                ve = math.fsum(
                    phi_l[a] * post_cov[a][b] * phi_l[b]
                    for a in range(p)
                    for b in range(p)
                )
                vd = math.fsum(
                    d_l[a] * post_cov[p + a][p + b] * d_l[b]
                    for a in range(len(d_l))
                    for b in range(len(d_l))
                )
                ce = 2.0 * math.fsum(
                    phi_l[a] * post_cov[a][p + b] * d_l[b]
                    for a in range(p)
                    for b in range(len(d_l))
                )
                prior_var = math.fsum(
                    phi_l[a] ** 2 * prior[w_idx[a]][w_idx[a]] for a in range(p)
                ) + math.fsum(d_l[a] ** 2 * sigma * sigma for a in range(len(d_l)))
                floor = -_VARIANCE_ROUNDING * prior_var
                for label, value in (("emulator", ve), ("discrepancy", vd)):
                    if value < floor:
                        raise ValueError(
                            f"conditional {label} variance is negative beyond "
                            "rounding; the covariance is ill-conditioned"
                        )
                c2 = scale * scale
                row.append(
                    (
                        c2 * max(ve, 0.0),
                        c2 * max(vd, 0.0),
                        c2 * ce,
                        self._mean_f[loc] + scale * mean_std,
                    )
                )
            out.append(tuple(row))
        return tuple(out)
