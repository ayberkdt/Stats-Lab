"""Inference-method comparison on synthetic expansion-history posteriors.

Experimental (added in 0.28).

This module is a SYNTHETIC TEST BED. Every redshift, noise draw and datum is
generated locally from a declared model and a predeclared seed; nothing here is
a constraint on the real universe, and no Bayes factor computed here is a
statement about dark energy. Its purpose is to compare *inference methods* on a
small nonlinear posterior whose reference can be computed by deterministic
quadrature.

Models. A spatially flat FLRW background with matter and one dark-energy
component, radiation neglected (``z <~ 2``), observed through supernova-like
distance moduli ``mu_i + M + eps_i`` (``eps_i ~ N(0, sigma^2)``) with the
magnitude offset ``M ~ N(0, s_M^2)`` marginalized analytically. Four nested or
competing parameterizations of ``w(z)`` share ``ln X(z) = 3 int_0^z (1 + w) /
(1 + t) dt`` in closed form [EQ-INV-APP-COSMO-010]:

- ``"lcdm"``: ``w = -1`` (``X = 1``);
- ``"wcdm"``: constant ``w`` (``X = (1 + z)^(3 (1 + w))``);
- ``"cpl"``: ``w = w0 + wa z / (1 + z)`` (Chevallier-Polarski, Linder);
- ``"pw2"``: two redshift bins, ``w1`` for ``z < z_b`` and ``w2`` beyond.

``ln X`` is linear in the ``w`` parameters, which gives the analytic
Jacobian. The magnitude offset is marginalized exactly, because it enters
linearly with a Gaussian prior: the data covariance becomes
``sigma^2 I + s_M^2 1 1'`` [EQ-INV-APP-COSMO-011]. ``H0`` is fixed (it is
degenerate with ``M``) and ``Omega_m`` is either fixed
(``free_omega=False``, the default) or has a Gaussian prior.

Methods compared on the same whitened target
(:class:`~probstat.inverse.posterior_targets.WhitenedGaussianPriorTarget`):

- (a) Laplace approximation at the MAP with the Gauss-Newton Hessian
  ``J'J + I`` in prior-whitened coordinates [EQ-INV-APP-COSMO-012];
- (b) pCN through :func:`probstat.inverse.sample_pcn`;
- (c) tempered SMC with pCN mutation, repeated over predeclared child seeds
  (:func:`~probstat.inverse.posterior_sampling.repeat_tempered_smc`) for the
  log-evidence and its run-to-run standard error;
- reference: the tensor Gauss-Legendre quadrature of
  :func:`~probstat.inverse.posterior_sampling.reference_for_target`, available
  for one or two free parameters (a Laplace-centred box whose truncation and
  refinement error are reported). With three or more free parameters there is
  NO reference and the comparison says so.

Errors are the mean error in reference standard deviations, the standard
deviation ratio and the largest error of the 2.5/50/97.5 % marginal quantiles
in reference standard deviations [EQ-INV-APP-COSMO-013]. Cost is the exact
number of likelihood evaluations plus measured wall-clock seconds (machine
dependent). Model comparison reports ``ln B_i0 = ln Z_i - ln Z_0`` with its
propagated Monte Carlo standard error and, where available, the quadrature
value with its refinement error [EQ-INV-APP-COSMO-014]. The Bayes factor depends
on the prior widths declared in :func:`model_spec`; a sensitivity sweep is
provided by :func:`prior_width_sensitivity`. Evidence uses normalized proper
priors and the same data for every model.

Mock data come from a different code path than the inverted model: the truth
``w(z)`` is a function (including one outside every candidate family), and
``mu`` is built by cumulative trapezoid integration on a fine grid, whereas the
inversion uses a 6-point Gauss-Legendre rule (``same_discretization=False``).
Truths inside a candidate family (``"lcdm"``, ``"cpl"``) are NOT crime-free in
the model sense, only in the numerical sense.

References:
    [REF-HOGG-1999], [REF-CHEVALLIER-POLARSKI-2001], [REF-LINDER-2003],
    [REF-DELMORAL-DOUCET-JASRA-2006], [REF-LEE-WHITELEY-2018],
    [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-KASS-RAFTERY-1995],
    [REF-STUART-2010].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    Matrix,
    Vector,
    inverse_spd,
    log_det_spd,
)
from probstat.optimization import least_squares

from .gauss_newton import CallbackForward
from .posterior_sampling import (
    QuadratureReference,
    TemperedSMCPosterior,
    reference_for_target,
    repeat_tempered_smc,
    tempered_smc_from_target,
)
from .posterior_targets import (
    WhitenedGaussianPriorTarget,
    physical_posterior,
    sample_pcn,
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC: mock distance moduli generated from a declared truth and a "
    "predeclared seed. Nothing here constrains the real universe, and no Bayes "
    "factor computed here is evidence about dark energy."
)

SPEED_OF_LIGHT_KM_S = 299792.458

PREDECLARED_SEEDS = MappingProxyType(
    {
        "data": 20260601,
        "pcn": 20260602,
        "smc": 20260603,
        "smc_repeat": 20260604,
    }
)

QUANTILE_LEVELS: tuple[float, ...] = (0.025, 0.5, 0.975)

_KINDS = ("lcdm", "wcdm", "cpl", "pw2")
_TRUTHS = ("lcdm", "cpl", "smooth_step")
_LN10 = math.log(10.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)


def _finite(value: object, name: str) -> float:
    """Return ``value`` as a finite float.

    References:
        [EQ-INV-APP-COSMO-010]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    """Return ``value`` as a finite positive float.

    References:
        [EQ-INV-APP-COSMO-010]. No equation is evaluated.

    """
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _count(value: object, name: str, minimum: int = 1) -> int:
    """Return ``value`` as an integer at least ``minimum``.

    References:
        [EQ-INV-APP-COSMO-010]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _gauss_legendre(order: int) -> tuple[Vector, Vector]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` (Newton).

    References:
        [REF-HOGG-1999]. No cosmological equation is evaluated.

    """
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(order):
        x = math.cos(math.pi * (i + 0.75) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, order + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            derivative = order * (x * p1 - p0) / (x * x - 1.0)
            step = p1 / derivative
            x -= step
            if abs(step) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    pairs = sorted(zip(nodes, weights, strict=True))
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declaration of one candidate model and its proper priors.

    Attributes:
        name: Label (``kind`` plus ``+om`` when ``Omega_m`` is free).
        kind: ``"lcdm"``, ``"wcdm"``, ``"cpl"`` or ``"pw2"``.
        free_omega: Whether ``Omega_m`` is a parameter.
        omega_m: Fixed value, or the prior mean when free.
        omega_sd: Prior standard deviation of ``Omega_m`` (when free).
        h0: Fixed ``H0`` in km/s/Mpc (degenerate with the offset).
        z_break: Break redshift of ``"pw2"`` (``0`` otherwise).
        w_mean: Prior means of the ``w`` parameters.
        w_sd: Prior standard deviations of the ``w`` parameters.
        offset_sd: Prior standard deviation ``s_M`` of the offset (mag).

    References:
        [EQ-INV-APP-COSMO-010], [EQ-INV-APP-COSMO-011].

    """

    name: str
    kind: str
    free_omega: bool
    omega_m: float
    omega_sd: float
    h0: float
    z_break: float
    w_mean: tuple[float, ...]
    w_sd: tuple[float, ...]
    offset_sd: float

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the free parameter names (``omega_m`` first when free).

        Returns:
            Names in the order of the parameter vector.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        names = {
            "lcdm": (),
            "wcdm": ("w",),
            "cpl": ("w0", "wa"),
            "pw2": ("w1", "w2"),
        }[self.kind]
        return (("omega_m",) if self.free_omega else ()) + names

    @property
    def dimension(self) -> int:
        """Return the number of free parameters.

        Returns:
            ``len(parameter_names)``.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        return len(self.parameter_names)

    @property
    def prior_mean(self) -> Vector:
        """Return the prior mean vector of the free parameters.

        Returns:
            Means in the order of :attr:`parameter_names`.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        return ((self.omega_m,) if self.free_omega else ()) + self.w_mean

    @property
    def prior_sd(self) -> Vector:
        """Return the prior standard deviation vector.

        Returns:
            Standard deviations in the order of :attr:`parameter_names`.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        return ((self.omega_sd,) if self.free_omega else ()) + self.w_sd


def model_spec(
    kind: str,
    *,
    free_omega: bool = False,
    omega_m: float = 0.3,
    omega_sd: float = 0.05,
    h0: float = 70.0,
    z_break: float = 0.5,
    prior_scale: float = 1.0,
    offset_sd: float = 0.5,
) -> ModelSpec:
    """Return a :class:`ModelSpec` with the declared default priors.

    Priors: ``w, w0, w1, w2 ~ N(-1, (0.5 s)^2)``, ``wa ~ N(0, s^2)`` with the
    width multiplier ``s = prior_scale``; ``Omega_m ~ N(omega_m, omega_sd^2)``
    when free. These are declared assumptions; the evidence depends on them
    (Occam factor), see :func:`prior_width_sensitivity`.

    Args:
        kind: ``"lcdm"``, ``"wcdm"``, ``"cpl"`` or ``"pw2"``.
        free_omega: Make ``Omega_m`` a parameter.
        omega_m: Fixed value or prior mean of ``Omega_m`` (``> 0``).
        omega_sd: Prior standard deviation of ``Omega_m``.
        h0: Fixed ``H0`` (km/s/Mpc, ``> 0``).
        z_break: Break of ``"pw2"`` (``> 0``).
        prior_scale: Multiplier of the ``w`` prior widths (``> 0``).
        offset_sd: Prior standard deviation of the offset (mag, ``> 0``).

    Returns:
        The specification.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``kind`` is unknown or a value is not positive
            (``"lcdm"`` with fixed ``Omega_m`` is allowed and has zero free
            parameters).

    References:
        [EQ-INV-APP-COSMO-010].

    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    if not isinstance(free_omega, bool):
        raise TypeError("free_omega must be a bool")
    om = _positive(omega_m, "omega_m")
    scale = _positive(prior_scale, "prior_scale")
    w_mean: tuple[float, ...]
    w_sd: tuple[float, ...]
    if kind == "lcdm":
        w_mean, w_sd = (), ()
    elif kind == "wcdm":
        w_mean, w_sd = (-1.0,), (0.5 * scale,)
    elif kind == "cpl":
        w_mean, w_sd = (-1.0, 0.0), (0.5 * scale, 1.0 * scale)
    else:
        w_mean, w_sd = (-1.0, -1.0), (0.5 * scale, 0.5 * scale)
    return ModelSpec(
        kind + ("+om" if free_omega else ""),
        kind,
        free_omega,
        om,
        _positive(omega_sd, "omega_sd"),
        _positive(h0, "h0"),
        _positive(z_break, "z_break") if kind == "pw2" else 0.0,
        w_mean,
        w_sd,
        _positive(offset_sd, "offset_sd"),
    )


class _Integrator:
    """Segmented Gauss-Legendre evaluation of the distance integral.

    ``ln X = c + g . p`` is linear in the ``w`` parameters ``p`` for every
    model (``c`` and ``g`` are tabulated per node, padded to two entries), so
    values and gradients of ``1 / E`` need one exponential and one power per
    node.

    References:
        [EQ-INV-APP-COSMO-010].

    """

    __slots__ = (
        "_prefactor",
        "_segments",
        "_seg_end",
        "free_omega",
        "n_w",
        "omega_fixed",
    )

    def __init__(
        self, spec: ModelSpec, redshifts: Sequence[float], order: int = 6
    ) -> None:
        """Tabulate nodes, weights and the per-node ``ln X`` coefficients.

        Args:
            spec: The model.
            redshifts: Strictly increasing positive redshifts.
            order: Gauss-Legendre points per segment.

        Raises:
            ValueError: If the redshifts are not positive and increasing, or
                ``z_break`` is not inside the redshift range for ``"pw2"``.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        zs = tuple(float(z) for z in redshifts)
        if (
            not zs
            or zs[0] <= 0.0
            or any(b <= a for a, b in zip(zs, zs[1:], strict=False))
        ):
            raise ValueError("redshifts must be positive and strictly increasing")
        breaks = set(zs)
        if spec.kind == "pw2":
            if not 0.0 < spec.z_break < zs[-1]:
                raise ValueError("z_break must lie inside the redshift range")
            breaks.add(spec.z_break)
        edges = [0.0, *sorted(breaks)]
        x_nodes, x_weights = _gauss_legendre(order)
        segments: list[tuple[tuple[float, float, float, float, float], ...]] = []
        for a, b in zip(edges, edges[1:], strict=False):
            half, mid = 0.5 * (b - a), 0.5 * (a + b)
            records = []
            for xi, wi in zip(x_nodes, x_weights, strict=True):
                z = mid + half * xi
                length = math.log1p(z)
                ratio = z / (1.0 + z)
                c0, g0, g1 = 0.0, 0.0, 0.0
                if spec.kind == "wcdm":
                    c0, g0 = 3.0 * length, 3.0 * length
                elif spec.kind == "cpl":
                    c0, g0, g1 = 3.0 * length, 3.0 * length, 3.0 * (length - ratio)
                elif spec.kind == "pw2":
                    low = math.log1p(min(z, spec.z_break))
                    high = max(0.0, length - math.log1p(spec.z_break))
                    c0, g0, g1 = 3.0 * (low + high), 3.0 * low, 3.0 * high
                records.append((half * wi, (1.0 + z) ** 3, c0, g0, g1))
            segments.append(tuple(records))
        self._segments = tuple(segments)
        ordered = edges[1:]
        self._seg_end = tuple(ordered.index(z) for z in zs)
        self._prefactor = tuple((1.0 + z) * SPEED_OF_LIGHT_KM_S / spec.h0 for z in zs)
        self.free_omega = spec.free_omega
        self.omega_fixed = spec.omega_m
        self.n_w = len(spec.w_mean)

    def moduli(
        self, theta: Sequence[float], *, jacobian: bool
    ) -> tuple[Vector, list[Vector]]:
        """Return distance moduli ``mu`` and optionally their Jacobian rows.

        Args:
            theta: Free parameters (``Omega_m`` first when free).
            jacobian: Also return the analytic Jacobian.

        Returns:
            ``(mu, rows)``; ``mu`` and ``rows`` contain NaN where
            ``E^2 <= 0`` or the exponential overflows.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        free = self.free_omega
        om = theta[0] if free else self.omega_fixed
        omc = 1.0 - om
        w_params = tuple(theta[1:] if free else theta)
        p0 = w_params[0] if self.n_w > 0 else 0.0
        p1 = w_params[1] if self.n_w > 1 else 0.0
        n_theta = len(theta)
        values: list[float] = []
        grads: list[tuple[float, float, float]] = []
        try:
            for records in self._segments:
                acc = d_om = d_0 = d_1 = 0.0
                for weight, c3, c0, g0, g1 in records:
                    x = math.exp(c0 + g0 * p0 + g1 * p1)
                    e2 = om * c3 + omc * x
                    if e2 <= 0.0:
                        raise ValueError("E^2 is not positive")
                    inv = e2**-0.5
                    acc += weight * inv
                    if jacobian:
                        factor = -0.5 * weight * inv * inv * inv
                        d_om += (c3 - x) * factor
                        scaled = omc * x * factor
                        d_0 += scaled * g0
                        d_1 += scaled * g1
                values.append(acc)
                grads.append((d_om, d_0, d_1))
        except (OverflowError, ValueError):
            nan = float("nan")
            size = len(self._seg_end)
            return (nan,) * size, [(nan,) * n_theta for _ in range(size)]
        totals: list[float] = []
        total_grads: list[tuple[float, float, float]] = []
        run = 0.0
        run_grad = [0.0, 0.0, 0.0]
        for value, grad in zip(values, grads, strict=True):
            run += value
            run_grad[0] += grad[0]
            run_grad[1] += grad[1]
            run_grad[2] += grad[2]
            totals.append(run)
            total_grads.append((run_grad[0], run_grad[1], run_grad[2]))
        mu = []
        rows: list[Vector] = []
        scale = 5.0 / _LN10
        n_w = self.n_w
        for i, end in enumerate(self._seg_end):
            integral = totals[end]
            mu.append(5.0 * math.log10(self._prefactor[i] * integral) + 25.0)
            if jacobian:
                grad = total_grads[end]
                chosen: tuple[float, ...] = (grad[0],) if free else ()
                chosen += grad[1 : 1 + n_w]
                rows.append(tuple(scale * d / integral for d in chosen))
        return tuple(mu), rows


def offset_marginal_covariance(
    size: int, sigma: float, offset_sd: float
) -> list[list[float]]:
    """Return ``sigma^2 I + s_M^2 1 1'`` as a dense matrix.

    The offset ``M ~ N(0, s_M^2)`` enters every datum with unit coefficient;
    integrating it out of the Gaussian likelihood leaves this covariance
    [EQ-INV-APP-COSMO-011].

    Args:
        size: Number of data (``>= 1``).
        sigma: Independent noise standard deviation (``> 0``).
        offset_sd: Offset prior standard deviation (``> 0``).

    Returns:
        The ``size x size`` covariance.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a value is out of range.

    References:
        [EQ-INV-APP-COSMO-011].

    """
    m = _count(size, "size")
    s2 = _positive(sigma, "sigma") ** 2
    o2 = _positive(offset_sd, "offset_sd") ** 2
    return [[o2 + (s2 if i == j else 0.0) for j in range(m)] for i in range(m)]


def offset_marginal_log_likelihood(
    residual: Sequence[float], sigma: float, offset_sd: float
) -> float:
    """Return ``log N(r; 0, sigma^2 I + s_M^2 1 1')`` in closed form.

    By the Sherman-Morrison identity,
    ``C^-1 = (I - s_M^2 / (sigma^2 + m s_M^2) 1 1') / sigma^2`` and
    ``log det C = (m - 1) log sigma^2 + log (sigma^2 + m s_M^2)``
    [EQ-INV-APP-COSMO-011]; the tests compare this with the dense Cholesky
    evaluation.

    Args:
        residual: ``G(theta) - y`` of length ``m >= 1``.
        sigma: Noise standard deviation (``> 0``).
        offset_sd: Offset prior standard deviation (``> 0``).

    Returns:
        The normalized log density.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the residual is empty or not finite.

    References:
        [EQ-INV-APP-COSMO-011].

    """
    r = tuple(_finite(v, "residual entry") for v in residual)
    if not r:
        raise ValueError("residual must be non-empty")
    m = len(r)
    s2 = _positive(sigma, "sigma") ** 2
    o2 = _positive(offset_sd, "offset_sd") ** 2
    total = math.fsum(r)
    quad = (math.fsum(v * v for v in r) - o2 * total * total / (s2 + m * o2)) / s2
    log_det = (m - 1) * math.log(s2) + math.log(s2 + m * o2)
    return -0.5 * (quad + log_det + m * _LOG_TWO_PI)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionData:
    """SYNTHETIC distance-modulus data with the declared truth withheld.

    Attributes:
        truth: Truth label (``"lcdm"``, ``"cpl"`` or ``"smooth_step"``).
        redshifts: Strictly increasing redshifts.
        data: ``mu_true + M_true + eps``.
        exact_data: Noise-free data (diagnostic; not an input of any solver).
        sigma: Noise standard deviation (mag).
        true_offset: Offset ``M`` used to generate the data.
        omega_m_true: ``Omega_m`` of the truth.
        h0: ``H0`` of the truth and of the inversion.
        seed: Root seed of the noise.
        same_discretization: ``False`` always (the data come from a fine
            trapezoid grid, the inversion from Gauss-Legendre).
        notice: :data:`SYNTHETIC_NOTICE`.

    References:
        [EQ-INV-APP-COSMO-010].

    """

    truth: str
    redshifts: Vector
    data: Vector
    exact_data: Vector
    sigma: float
    true_offset: float
    omega_m_true: float
    h0: float
    seed: int
    same_discretization: bool
    notice: str


def _truth_w(name: str) -> Callable[[float], float]:
    """Return the equation of state ``w(z)`` of a named truth.

    References:
        [EQ-INV-APP-COSMO-010].

    """
    if name == "lcdm":
        return lambda z: -1.0
    if name == "cpl":
        return lambda z: -0.9 - 0.4 * z / (1.0 + z)
    return lambda z: -1.0 + 0.3 * 0.5 * (1.0 - math.tanh((z - 0.6) / 0.25))


def _fine_moduli(
    w: Callable[[float], float],
    redshifts: Sequence[float],
    omega_m: float,
    h0: float,
    subdivisions: int,
) -> Vector:
    """Return ``mu`` by cumulative trapezoid integration on a fine grid.

    Independent of the Gauss-Legendre inversion model and of the closed forms
    of ``ln X``: ``ln X`` and ``int dt / E`` are both accumulated on the same
    grid (``subdivisions`` trapezoids per redshift interval).

    References:
        [EQ-INV-APP-COSMO-010].

    """
    grid = [0.0]
    for a, b in zip((0.0, *redshifts), redshifts, strict=False):
        for k in range(1, subdivisions + 1):
            grid.append(a + (b - a) * k / subdivisions)
    ln_x = 0.0
    integral = 0.0
    previous_f = 3.0 * (1.0 + w(0.0))
    previous_inv = 1.0
    out: list[float] = []
    results: dict[int, float] = {}
    for index, (z_prev, z) in enumerate(zip(grid, grid[1:], strict=False), start=1):
        f = 3.0 * (1.0 + w(z)) / (1.0 + z)
        ln_x += 0.5 * (previous_f + f) * (z - z_prev)
        e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m) * math.exp(ln_x)
        inv = e2**-0.5
        integral += 0.5 * (previous_inv + inv) * (z - z_prev)
        previous_f, previous_inv = f, inv
        if index % subdivisions == 0:
            results[index // subdivisions - 1] = integral
    for i, z in enumerate(redshifts):
        d_l = (1.0 + z) * SPEED_OF_LIGHT_KM_S / h0 * results[i]
        out.append(5.0 * math.log10(d_l) + 25.0)
    return tuple(out)


def synthetic_expansion_data(
    truth: str = "cpl",
    *,
    redshifts: Sequence[float] | None = None,
    count: int = 16,
    z_max: float = 1.4,
    sigma: float = 0.05,
    offset: float = 0.1,
    omega_m: float = 0.3,
    h0: float = 70.0,
    seed: int | None = None,
) -> ExpansionData:
    """Return SYNTHETIC supernova-like data from a declared truth.

    The truths are ``"lcdm"`` (``w = -1``), ``"cpl"`` (``w = -0.9 - 0.4 z /
    (1 + z)``, inside the CPL family) and ``"smooth_step"`` (``w`` moves
    from about ``-0.85`` to ``-1`` around ``z = 0.6``, outside every
    candidate family, so even the flexible models are misspecified). The
    noise is the only random ingredient and uses the predeclared seed
    ``PREDECLARED_SEEDS["data"]`` unless one is given
    [EQ-INV-APP-COSMO-010].

    Args:
        truth: ``"lcdm"``, ``"cpl"`` or ``"smooth_step"``.
        redshifts: Increasing positive redshifts (default ``count`` points
            evenly in ``[z_max / count, z_max]``).
        count: Number of default redshifts (``>= 3``).
        z_max: Largest default redshift (``> 0``).
        sigma: Noise standard deviation in magnitudes (``> 0``).
        offset: True magnitude offset ``M``.
        omega_m: True matter density (``> 0``).
        h0: ``H0`` in km/s/Mpc (``> 0``).
        seed: Root seed of the noise.

    Returns:
        The data with the truth label (the truth parameters are not stored
        in a form a solver could read).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: On an unknown truth or invalid redshifts.

    References:
        [EQ-INV-APP-COSMO-010], [REF-HOGG-1999].

    """
    if truth not in _TRUTHS:
        raise ValueError(f"truth must be one of {_TRUTHS}")
    if redshifts is None:
        n_obs = _count(count, "count", 3)
        top = _positive(z_max, "z_max")
        zs = tuple(top * (i + 1) / n_obs for i in range(n_obs))
    else:
        zs = tuple(_positive(z, "redshift") for z in redshifts)
        if any(b <= a for a, b in zip(zs, zs[1:], strict=False)):
            raise ValueError("redshifts must be strictly increasing")
    level = _positive(sigma, "sigma")
    shift = _finite(offset, "offset")
    om = _positive(omega_m, "omega_m")
    hubble = _positive(h0, "h0")
    root = PREDECLARED_SEEDS["data"] if seed is None else seed
    exact = tuple(
        mu + shift for mu in _fine_moduli(_truth_w(truth), zs, om, hubble, 400)
    )
    generator = resolve_stream(root).child("expansion-data", truth).generator()
    data = tuple(v + level * generator.gauss(0.0, 1.0) for v in exact)
    return ExpansionData(
        truth, zs, data, exact, level, shift, om, hubble, root, False, SYNTHETIC_NOTICE
    )


# ---------------------------------------------------------------------------
# Inference problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionInferenceProblem:
    """One model confronted with one synthetic data set.

    Attributes:
        spec: The model.
        data: The data.
        forward: Differentiable forward map ``theta -> mu`` (``None`` for a
            model without free parameter).
        noise: Offset-marginalized data covariance.
        target: The prior-whitened target (``None`` without free parameter).
        fixed_log_evidence: Exact ``log p(y)`` for a model with no free
            parameter (NaN otherwise).

    References:
        [EQ-INV-APP-COSMO-011], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """

    spec: ModelSpec
    data: ExpansionData
    forward: CallbackForward | None
    noise: DenseCovariance
    target: WhitenedGaussianPriorTarget | None
    fixed_log_evidence: float
    _integrator: _Integrator

    def moduli(self, theta: Sequence[float]) -> Vector:
        """Return the model distance moduli at ``theta``.

        Args:
            theta: Free parameters of the model.

        Returns:
            ``mu_i`` (without offset); NaN entries mark an invalid point.

        Raises:
            ValueError: If ``theta`` has the wrong length.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        if len(theta) != self.spec.dimension:
            raise ValueError(f"theta needs {self.spec.dimension} entries")
        return self._integrator.moduli(tuple(theta), jacobian=False)[0]

    def jacobian(self, theta: Sequence[float]) -> list[Vector]:
        """Return the analytic Jacobian rows ``d mu_i / d theta``.

        Args:
            theta: Free parameters of the model.

        Returns:
            ``m`` rows of length ``n``.

        Raises:
            ValueError: If ``theta`` has the wrong length.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        if len(theta) != self.spec.dimension:
            raise ValueError(f"theta needs {self.spec.dimension} entries")
        return self._integrator.moduli(tuple(theta), jacobian=True)[1]

    def to_physical(self, z: Sequence[float]) -> Vector:
        """Map whitened ``z`` to parameters ``mu + sd * z``.

        Args:
            z: Whitened coordinates.

        Returns:
            Physical parameters.

        References:
            [EQ-INV-APP-COSMO-012].

        """
        return tuple(
            m + s * v
            for m, s, v in zip(self.spec.prior_mean, self.spec.prior_sd, z, strict=True)
        )


def build_problem(
    spec: ModelSpec, data: ExpansionData, *, quadrature_order: int = 6
) -> ExpansionInferenceProblem:
    """Assemble the whitened target of a model and a data set.

    The likelihood is ``N(y; mu(theta), sigma^2 I + s_M^2 1 1')`` with the
    offset integrated out, the prior is the diagonal Gaussian of ``spec``
    (so the whitened prior is exactly ``N(0, I)``). A model with no free
    parameter carries its exact evidence instead of a target
    [EQ-INV-APP-COSMO-011].

    Args:
        spec: The model.
        data: The synthetic data.
        quadrature_order: Gauss-Legendre points per segment (``>= 2``).

    Returns:
        The problem.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the redshift range is incompatible with ``spec``.

    References:
        [EQ-INV-APP-COSMO-010], [EQ-INV-APP-COSMO-011].

    """
    if not isinstance(spec, ModelSpec) or not isinstance(data, ExpansionData):
        raise TypeError("spec must be a ModelSpec and data an ExpansionData")
    order = _count(quadrature_order, "quadrature_order", 2)
    integrator = _Integrator(spec, data.redshifts, order)
    m = len(data.redshifts)
    covariance = DenseCovariance(
        offset_marginal_covariance(m, data.sigma, spec.offset_sd)
    )
    if spec.dimension == 0:
        mu = integrator.moduli((), jacobian=False)[0]
        residual = [a - b for a, b in zip(mu, data.data, strict=True)]
        exact = offset_marginal_log_likelihood(residual, data.sigma, spec.offset_sd)
        return ExpansionInferenceProblem(
            spec, data, None, covariance, None, exact, integrator
        )

    def value(theta: Vector) -> Vector:
        """Return the moduli at ``theta``.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        return integrator.moduli(theta, jacobian=False)[0]

    def jacobian(theta: Vector) -> list[Vector]:
        """Return the analytic Jacobian rows at ``theta``.

        References:
            [EQ-INV-APP-COSMO-010].

        """
        return integrator.moduli(theta, jacobian=True)[1]

    forward = CallbackForward(spec.dimension, m, value, jacobian=jacobian)
    target = WhitenedGaussianPriorTarget(
        forward,
        data.data,
        noise=covariance,
        prior_mean=spec.prior_mean,
        prior_factor=DiagonalCovariance([s * s for s in spec.prior_sd]),
    )
    return ExpansionInferenceProblem(
        spec, data, forward, covariance, target, math.nan, integrator
    )


# ---------------------------------------------------------------------------
# Laplace approximation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LaplaceFit:
    """Laplace approximation at the MAP with the Gauss-Newton Hessian.

    Attributes:
        converged: Whether the optimizer met a tolerance.
        message: Optimizer termination reason (or the failure).
        z_map: MAP in whitened coordinates.
        theta_map: MAP in physical coordinates.
        covariance: Gauss-Newton covariance of the physical parameters.
        sd: Marginal standard deviations (physical).
        z_sd: Marginal standard deviations (whitened).
        log_evidence: ``log L(z*) + log N(z*) + d/2 log 2 pi - 1/2 log det H``.
        log_det_hessian: ``log det (J'J + I)`` (whitened).
        evaluations: Residual evaluations of the optimizer.

    References:
        [EQ-INV-APP-COSMO-012].

    """

    converged: bool
    message: str
    z_map: Vector
    theta_map: Vector
    covariance: Matrix
    sd: Vector
    z_sd: Vector
    log_evidence: float
    log_det_hessian: float
    evaluations: int


def laplace_fit(problem: ExpansionInferenceProblem) -> LaplaceFit:
    """Return the Laplace approximation of a problem at its MAP.

    The MAP minimizes ``1/2 ||W (G(mu + S z) - y)||^2 + 1/2 ||z||^2`` by the
    library's Levenberg-Marquardt from the prior mean (``z = 0``) with the
    analytic Jacobian. The Hessian is the Gauss-Newton one,
    ``H = J_z' J_z + I`` with ``J_z = W J S``; it omits the residual-curvature
    term, which is the difference from the exact Hessian of the nonlinear
    problem [EQ-INV-APP-COSMO-012]. A failure is reported in
    ``converged`` and ``message``, not hidden.

    Args:
        problem: A problem with at least one free parameter.

    Returns:
        :class:`LaplaceFit`.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If the model has no free parameter.

    References:
        [EQ-INV-APP-COSMO-012], [REF-NOCEDAL-WRIGHT-2006].

    """
    if not isinstance(problem, ExpansionInferenceProblem):
        raise TypeError("problem must be an ExpansionInferenceProblem")
    target = problem.target
    forward = problem.forward
    if target is None or forward is None:
        raise ValueError("the model has no free parameter")
    noise = problem.noise
    spec = problem.spec
    n = spec.dimension
    sd = spec.prior_sd
    data = problem.data.data

    def residual(z: Vector) -> Vector:
        """Return the stacked whitened data and prior residuals.

        References:
            [EQ-INV-APP-COSMO-012].

        """
        mu = forward.value(problem.to_physical(z))
        r = noise.whiten(tuple(a - b for a, b in zip(mu, data, strict=True)))
        return (*r, *z)

    def jac(z: Vector) -> list[Vector]:
        """Return the Jacobian of the stacked residual.

        References:
            [EQ-INV-APP-COSMO-012].

        """
        rows = problem.jacobian(problem.to_physical(z))
        columns = [
            noise.whiten(tuple(rows[i][j] * sd[j] for i in range(len(rows))))
            for j in range(n)
        ]
        stacked: list[Vector] = [
            tuple(columns[j][i] for j in range(n)) for i in range(len(rows))
        ]
        stacked += [tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n)]
        return stacked

    nan = math.nan
    try:
        fit = least_squares(
            residual, (0.0,) * n, jacobian=jac, tolerance=1e-12, max_iterations=200
        )
    except ValueError as error:
        return LaplaceFit(
            False, f"optimizer failed: {error}", (), (), (), (), (), nan, nan, 0
        )
    z_star = fit.x
    hessian = [
        [math.fsum(row[i] * row[j] for row in fit.jacobian) for j in range(n)]
        for i in range(n)
    ]
    cov_z = inverse_spd(tuple(tuple(r) for r in hessian))
    log_det = log_det_spd(tuple(tuple(r) for r in hessian))
    theta = problem.to_physical(z_star)
    covariance = tuple(
        tuple(sd[i] * sd[j] * cov_z[i][j] for j in range(n)) for i in range(n)
    )
    log_likelihood = target.log_likelihood(z_star) + target.log_likelihood_constant
    evidence = (
        log_likelihood
        + target.log_prior(z_star)
        + 0.5 * n * _LOG_TWO_PI
        - 0.5 * log_det
    )
    return LaplaceFit(
        fit.converged,
        fit.message,
        z_star,
        theta,
        covariance,
        tuple(math.sqrt(covariance[i][i]) for i in range(n)),
        tuple(math.sqrt(cov_z[i][i]) for i in range(n)),
        evidence,
        log_det,
        fit.evaluations,
    )


# ---------------------------------------------------------------------------
# Method summaries and comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MethodSummary:
    """Posterior summary of one method, with errors against the reference.

    Attributes:
        method: ``"laplace"``, ``"pcn"``, ``"smc"`` or ``"quadrature"``.
        converged: Method-level success flag.
        note: Explanation, warnings or failure reason.
        mean: Posterior mean (physical, per parameter).
        sd: Posterior standard deviation.
        quantiles: Per parameter, the values at :data:`QUANTILE_LEVELS`.
        mean_error_sd: ``|mean - mean_ref| / sd_ref`` per parameter (NaN when
            no reference).
        sd_ratio: ``sd / sd_ref`` per parameter.
        quantile_error_sd: Largest quantile error over the levels, in
            reference standard deviations, per parameter.
        log_evidence: ``log p(y)`` estimate (NaN when the method has none).
        log_evidence_se: Standard error of ``log_evidence`` (NaN when
            unavailable; for SMC the across-run standard error of the mean).
        evaluations: Likelihood (or residual) evaluations spent.
        seconds: Measured wall-clock time (machine dependent).

    References:
        [EQ-INV-APP-COSMO-013].

    """

    method: str
    converged: bool
    note: str
    mean: Vector
    sd: Vector
    quantiles: tuple[Vector, ...]
    mean_error_sd: Vector
    sd_ratio: Vector
    quantile_error_sd: Vector
    log_evidence: float
    log_evidence_se: float
    evaluations: int
    seconds: float


@dataclass(frozen=True, slots=True)
class InferenceComparison:
    """Laplace, pCN and tempered SMC compared with a quadrature reference.

    Attributes:
        model: Model label.
        parameter_names: Free parameter names.
        reference: Quadrature summary (``None`` without reference).
        reference_note: Why there is or is no reference, with its
            truncation and refinement diagnostics.
        methods: Summaries of the methods (reference excluded).
        smc_log_evidences: ``log Z_hat`` of every SMC run.
        smc_log_evidence_run_std: Run-to-run standard deviation of
            ``log Z_hat`` (one-run error).
        failures: Human-readable failures (empty when none).
        seeds: The predeclared seeds used.
        notice: :data:`SYNTHETIC_NOTICE`.

    References:
        [EQ-INV-APP-COSMO-013].

    """

    model: str
    parameter_names: tuple[str, ...]
    reference: MethodSummary | None
    reference_note: str
    methods: tuple[MethodSummary, ...]
    smc_log_evidences: tuple[float, ...]
    smc_log_evidence_run_std: float
    failures: tuple[str, ...]
    seeds: tuple[tuple[str, int], ...]
    notice: str

    def method(self, name: str) -> MethodSummary:
        """Return the summary of the method called ``name``.

        Args:
            name: ``"laplace"``, ``"pcn"`` or ``"smc"``.

        Returns:
            The summary.

        Raises:
            KeyError: If there is no such method.

        References:
            [EQ-INV-APP-COSMO-013].

        """
        for item in self.methods:
            if item.method == name:
                return item
        raise KeyError(name)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the type-7 sample quantile of sorted values.

    References:
        [EQ-INV-APP-COSMO-013].

    """
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (position - low) * (
        sorted_values[high] - sorted_values[low]
    )


def _errors(
    mean: Vector,
    sd: Vector,
    quantiles: tuple[Vector, ...],
    reference: MethodSummary | None,
) -> tuple[Vector, Vector, Vector]:
    """Return mean error, sd ratio and quantile error in reference sd units.

    References:
        [EQ-INV-APP-COSMO-013].

    """
    n = len(mean)
    if reference is None:
        nan = (math.nan,) * n
        return nan, nan, nan
    mean_error = tuple(
        abs(mean[i] - reference.mean[i]) / reference.sd[i] for i in range(n)
    )
    ratio = tuple(sd[i] / reference.sd[i] for i in range(n))
    q_error = tuple(
        max(
            abs(quantiles[i][k] - reference.quantiles[i][k]) / reference.sd[i]
            for k in range(len(QUANTILE_LEVELS))
        )
        for i in range(n)
    )
    return mean_error, ratio, q_error


def _sample_summary(
    method: str,
    samples: Sequence[Vector],
    reference: MethodSummary | None,
    *,
    note: str,
    log_evidence: float,
    log_evidence_se: float,
    evaluations: int,
    seconds: float,
) -> MethodSummary:
    """Summarize physical samples and compare with the reference.

    References:
        [EQ-INV-APP-COSMO-013].

    """
    count = len(samples)
    n = len(samples[0])
    mean = tuple(math.fsum(s[i] for s in samples) / count for i in range(n))
    sd = tuple(
        math.sqrt(math.fsum((s[i] - mean[i]) ** 2 for s in samples) / (count - 1))
        for i in range(n)
    )
    quantiles = tuple(
        tuple(_quantile(sorted(s[i] for s in samples), q) for q in QUANTILE_LEVELS)
        for i in range(n)
    )
    errors = _errors(mean, sd, quantiles, reference)
    return MethodSummary(
        method,
        True,
        note,
        mean,
        sd,
        quantiles,
        *errors,
        log_evidence,
        log_evidence_se,
        evaluations,
        seconds,
    )


def _laplace_summary(
    fit: LaplaceFit, reference: MethodSummary | None, seconds: float
) -> MethodSummary:
    """Summarize a Laplace fit (normal quantiles) against the reference.

    References:
        [EQ-INV-APP-COSMO-012], [EQ-INV-APP-COSMO-013].

    """
    if not fit.converged:
        n = len(fit.theta_map)
        nan = (math.nan,) * n
        return MethodSummary(
            "laplace", False, fit.message, nan, nan, (), nan, nan, nan,
            math.nan, math.nan, fit.evaluations, seconds,
        )  # fmt: skip
    normal = NormalDist()
    levels = [normal.inv_cdf(q) for q in QUANTILE_LEVELS]
    quantiles = tuple(
        tuple(fit.theta_map[i] + fit.sd[i] * v for v in levels)
        for i in range(len(fit.sd))
    )
    errors = _errors(fit.theta_map, fit.sd, quantiles, reference)
    return MethodSummary(
        "laplace",
        True,
        "Gauss-Newton Hessian at the MAP; mean = MAP",
        fit.theta_map,
        fit.sd,
        quantiles,
        *errors,
        fit.log_evidence,
        math.nan,
        fit.evaluations,
        seconds,
    )


def _reference_summary(
    problem: ExpansionInferenceProblem,
    fit: LaplaceFit,
    *,
    panels: int,
    order: int,
    box_sd: float,
    refine_check: bool,
    refinement_tolerance: float,
) -> tuple[MethodSummary | None, QuadratureReference | None, str]:
    """Return the quadrature summary (physical), the raw reference and a note.

    References:
        [EQ-INV-APP-COSMO-013], [REF-DAVIS-RABINOWITZ-1984].

    """
    spec = problem.spec
    target = problem.target
    n = spec.dimension
    if target is None or n > 2:
        return None, None, f"no quadrature reference for {n} free parameters"
    if fit.converged:
        bounds = [
            (
                max(fit.z_map[i] - box_sd * fit.z_sd[i], -8.0),
                min(fit.z_map[i] + box_sd * fit.z_sd[i], 8.0),
            )
            for i in range(n)
        ]
        where = f"Laplace-centred box +/- {box_sd} sd"
    else:
        bounds = [(-6.0, 6.0)] * n
        where = "prior box [-6, 6] (Laplace failed)"
    ref = reference_for_target(
        target,
        bounds,
        panels=panels,
        order=order,
        refine_check=refine_check,
        refinement_tolerance=refinement_tolerance,
    )
    mean = problem.to_physical(ref.mean)
    sd = tuple(s * v for s, v in zip(spec.prior_sd, ref.std(), strict=True))
    quantiles = tuple(
        tuple(
            spec.prior_mean[i] + spec.prior_sd[i] * value
            for _, value in ref.quantiles[i]
        )
        for i in range(n)
    )
    nan = (math.nan,) * n
    summary = MethodSummary(
        "quadrature",
        not ref.warnings,
        "; ".join(ref.warnings) if ref.warnings else "no warnings",
        mean,
        sd,
        quantiles,
        nan,
        nan,
        nan,
        ref.log_normalizer,
        ref.refinement_log_error,
        ref.evaluations,
        0.0,
    )
    note = (
        f"{where}; panels={panels}, order={order}; truncation estimate "
        f"{ref.truncation_mass_estimate:.2e}; boundary density ratio "
        f"{ref.boundary_density_ratio:.2e}; |dlogZ| under doubled panels "
        f"{ref.refinement_log_error:.2e}"
    )
    return summary, ref, note


def compare_inference_methods(
    problem: ExpansionInferenceProblem,
    *,
    particles: int = 300,
    runs: int = 4,
    mutation_steps: int = 3,
    pcn_draws: int = 400,
    pcn_warmup: int = 300,
    pcn_chains: int = 3,
    panels: int = 12,
    quadrature_order: int = 8,
    box_sd: float = 14.0,
    refine_check: bool = True,
    refinement_tolerance: float = 1e-4,
    seed: int | None = None,
) -> InferenceComparison:
    """Compare Laplace, pCN and tempered SMC with a quadrature reference.

    All three methods act on the same
    :class:`~probstat.inverse.posterior_targets.WhitenedGaussianPriorTarget`
    (prior ``N(0, I)`` in ``z``). (a) :func:`laplace_fit`; (b) pCN through
    :func:`~probstat.inverse.posterior_targets.sample_pcn` with the library's
    step adaptation; (c) tempered SMC with pCN mutation repeated ``runs``
    times over child seeds of ``PREDECLARED_SEEDS["smc_repeat"]``, giving the
    log-evidence mean, its run-to-run standard error and one run's particles
    for the moments. The reference exists for one or two free parameters. A
    method that fails or misses the reference by more than a posterior
    standard deviation is listed in ``failures``; nothing is hidden or
    repaired [EQ-INV-APP-COSMO-013].

    Args:
        problem: A problem with at least one free parameter.
        particles: SMC particles (``>= 2``).
        runs: Independent SMC runs (``>= 2``).
        mutation_steps: pCN moves per particle and stage.
        pcn_draws: Kept pCN draws per chain.
        pcn_warmup: pCN warm-up per chain.
        pcn_chains: pCN chains.
        panels: Quadrature panels per dimension.
        quadrature_order: Gauss-Legendre points per quadrature panel.
        box_sd: Half-width of the quadrature box in Laplace standard
            deviations.
        refine_check: Run the doubled-panel check of the reference.
        refinement_tolerance: Largest tolerated change of the reference
            ``log Z`` under doubled panels before a failure is listed (the
            default ``1e-4`` is far below the Monte Carlo error of the
            evidence).
        seed: Root seed for pCN and SMC (default the predeclared seeds).

    Returns:
        :class:`InferenceComparison`.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If the model has no free parameter or an option is
            invalid.

    References:
        [EQ-INV-APP-COSMO-013], [EQ-INV-SMP-001], [EQ-INV-SMP-002],
        [REF-DELMORAL-DOUCET-JASRA-2006], [REF-LEE-WHITELEY-2018].

    """
    if not isinstance(problem, ExpansionInferenceProblem):
        raise TypeError("problem must be an ExpansionInferenceProblem")
    target = problem.target
    if target is None:
        raise ValueError("the model has no free parameter")
    spec = problem.spec
    failures: list[str] = []

    clock = time.perf_counter()
    fit = laplace_fit(problem)
    laplace_seconds = time.perf_counter() - clock
    reference, _, note = _reference_summary(
        problem,
        fit,
        panels=panels,
        order=quadrature_order,
        box_sd=box_sd,
        refine_check=refine_check,
        refinement_tolerance=refinement_tolerance,
    )
    if reference is not None and not reference.converged:
        failures.append(f"quadrature reference: {reference.note}")
    methods = [_laplace_summary(fit, reference, laplace_seconds)]
    if not fit.converged:
        failures.append(f"laplace: {fit.message}")

    pcn_seed = PREDECLARED_SEEDS["pcn"] if seed is None else seed
    clock = time.perf_counter()
    chain = sample_pcn(
        target,
        draws=pcn_draws,
        warmup=pcn_warmup,
        chains=pcn_chains,
        seed=pcn_seed,
        check_convergence=False,
    )
    physical = physical_posterior(target, chain, names=spec.parameter_names)
    pcn = _sample_summary(
        "pcn",
        physical.pooled(),
        reference,
        note=f"acceptance {chain.acceptance_rate:.3f}",
        log_evidence=math.nan,
        log_evidence_se=math.nan,
        evaluations=chain.evaluations,
        seconds=time.perf_counter() - clock,
    )
    methods.append(pcn)

    smc_seed = PREDECLARED_SEEDS["smc_repeat"] if seed is None else seed
    collected: list[TemperedSMCPosterior] = []

    def run(stream: RandomStream) -> TemperedSMCPosterior:
        """Run one tempered SMC and keep the result.

        References:
            [EQ-INV-SMP-001].

        """
        outcome = tempered_smc_from_target(
            target,
            particles=particles,
            mutation="pcn",
            mutation_steps=mutation_steps,
            seed=stream,
        )
        collected.append(outcome)
        return outcome

    clock = time.perf_counter()
    summary = repeat_tempered_smc(run, runs=runs, seed=smc_seed)
    smc_seconds = time.perf_counter() - clock
    first = next(r for r in collected if r.converged)
    smc = _sample_summary(
        "smc",
        first.samples,
        reference,
        note=(
            f"{summary.failed} failed of {runs}; stages {len(first.temperatures) - 1}"
            f" (run 0); moments from one run"
        ),
        log_evidence=summary.log_evidence_mean,
        log_evidence_se=summary.log_evidence_mean_se,
        evaluations=sum(r.evaluations for r in collected),
        seconds=smc_seconds,
    )
    methods.append(smc)
    if summary.failed:
        failures.append(f"smc: {summary.failed} of {runs} runs failed")

    if reference is not None:
        for item in methods:
            if not item.converged:
                continue
            worst = max(item.mean_error_sd)
            if worst > 1.0:
                failures.append(
                    f"{item.method}: mean error {worst:.2f} reference sd (> 1)"
                )
        smc_gap = abs(smc.log_evidence - reference.log_evidence)
        refinement = (
            reference.log_evidence_se
            if math.isfinite(reference.log_evidence_se)
            else 0.0
        )
        allowance = 4.0 * max(smc.log_evidence_se, 1e-12) + refinement
        if smc_gap > allowance:
            failures.append(
                f"smc: log-evidence differs from quadrature by {smc_gap:.3f} "
                f"(> {allowance:.3f}, four standard errors plus refinement error)"
            )
    return InferenceComparison(
        spec.name,
        spec.parameter_names,
        reference,
        note,
        tuple(methods),
        summary.log_evidences,
        summary.log_evidence_std,
        tuple(failures),
        (
            ("pcn", pcn_seed),
            ("smc_repeat", smc_seed),
            ("data", problem.data.seed),
        ),
        SYNTHETIC_NOTICE,
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    """Evidence of one model with its Bayes factor against the baseline.

    Attributes:
        model: Model label.
        dimension: Number of free parameters.
        smc_log_evidence: SMC mean of ``log Z_hat`` (exact for zero
            parameters).
        smc_log_evidence_se: Across-run standard error of that mean (0 when
            exact).
        smc_run_std: Run-to-run standard deviation of one run.
        quadrature_log_evidence: Quadrature ``log Z`` (NaN when none).
        quadrature_error: Doubled-panel change of the quadrature value (NaN
            when the doubled-panel check was switched off).
        laplace_log_evidence: Laplace estimate (NaN when none).
        log_bayes_factor: ``ln B`` against the baseline from SMC.
        log_bayes_factor_se: Propagated standard error.
        log_bayes_factor_quadrature: ``ln B`` from quadrature (NaN when a
            side has none).
        log_bayes_factor_quadrature_error: Sum of the two refinement errors.
        category: Kass-Raftery verbal category of ``2 ln B`` and a warning
            when it is not larger than three standard errors.

    References:
        [EQ-INV-APP-COSMO-014].

    """

    model: str
    dimension: int
    smc_log_evidence: float
    smc_log_evidence_se: float
    smc_run_std: float
    quadrature_log_evidence: float
    quadrature_error: float
    laplace_log_evidence: float
    log_bayes_factor: float
    log_bayes_factor_se: float
    log_bayes_factor_quadrature: float
    log_bayes_factor_quadrature_error: float
    category: str


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Bayes factors of competing models on the same SYNTHETIC data.

    Attributes:
        truth: Truth label of the data.
        baseline: Baseline model label.
        rows: One :class:`EvidenceRow` per model (baseline first).
        comparisons: Per-model :class:`InferenceComparison` (without a
            comparison for a model with no free parameter).
        notice: :data:`SYNTHETIC_NOTICE`.

    References:
        [EQ-INV-APP-COSMO-014].

    """

    truth: str
    baseline: str
    rows: tuple[EvidenceRow, ...]
    comparisons: tuple[InferenceComparison, ...]
    notice: str


def evidence_category(log_bayes_factor: float) -> str:
    """Return the Kass-Raftery category of ``2 ln B``.

    The scale (``0-2`` not worth more than a bare mention, ``2-6`` positive,
    ``6-10`` strong, ``> 10`` very strong, for the favored side) is a verbal
    convention, not a calibrated probability, and depends on the prior widths
    [EQ-INV-APP-COSMO-014].

    Args:
        log_bayes_factor: ``ln B`` (``B > 1`` favors the numerator).

    Returns:
        A label such as ``"positive for the model"``.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is not finite.

    References:
        [EQ-INV-APP-COSMO-014], [REF-KASS-RAFTERY-1995].

    """
    value = 2.0 * _finite(log_bayes_factor, "log_bayes_factor")
    side = "the model" if value >= 0.0 else "the baseline"
    size = abs(value)
    if size < 2.0:
        label = "not worth more than a bare mention"
    elif size < 6.0:
        label = "positive"
    elif size < 10.0:
        label = "strong"
    else:
        label = "very strong"
    return f"{label} for {side}" if size >= 2.0 else label


def compare_models(
    data: ExpansionData,
    specs: Sequence[ModelSpec],
    *,
    particles: int = 300,
    runs: int = 4,
    mutation_steps: int = 3,
    panels: int = 12,
    quadrature_order: int = 8,
    refine_check: bool = True,
    refinement_tolerance: float = 1e-4,
    seed: int | None = None,
    pcn_draws: int = 200,
    pcn_warmup: int = 150,
) -> ModelComparison:
    """Compare competing models by SMC and quadrature evidence.

    The first spec is the baseline. For every other model the log Bayes factor
    ``ln B_i0 = ln Z_i - ln Z_0`` and its standard error
    ``sqrt(se_i^2 + se_0^2)`` (independent runs; ``se = 0`` for a model with no
    free parameter, whose evidence is exact) are reported together with the
    quadrature value and its refinement error when both sides have one
    [EQ-INV-APP-COSMO-014]. Models are compared on the same data with
    normalized proper priors. These Bayes factors are properties of the
    synthetic data and the declared priors only.

    Args:
        data: The synthetic data.
        specs: At least two model specifications.
        particles: SMC particles.
        runs: SMC runs per model (``>= 2``).
        mutation_steps: pCN moves per stage.
        panels: Quadrature panels per dimension.
        quadrature_order: Gauss-Legendre points per quadrature panel.
        refine_check: Doubled-panel check of the quadrature.
        refinement_tolerance: Doubled-panel tolerance of the reference.
        seed: Root seed override.
        pcn_draws: Kept pCN draws (the pCN summary is part of the
            per-model comparison).
        pcn_warmup: pCN warm-up.

    Returns:
        :class:`ModelComparison`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If fewer than two specs are given or two share a label.

    References:
        [EQ-INV-APP-COSMO-014], [REF-KASS-RAFTERY-1995].

    """
    models = tuple(specs)
    if len(models) < 2:
        raise ValueError("at least two models are needed")
    if len({s.name for s in models}) != len(models):
        raise ValueError("model labels must be distinct")
    rows: list[EvidenceRow] = []
    comparisons: list[InferenceComparison] = []
    base: tuple[float, float, float, float] | None = None
    for spec in models:
        problem = build_problem(spec, data)
        if problem.target is None:
            exact = problem.fixed_log_evidence
            ev = (exact, 0.0, exact, 0.0)
            row_values = (exact, 0.0, 0.0, exact, 0.0, math.nan)
        else:
            comparison = compare_inference_methods(
                problem,
                particles=particles,
                runs=runs,
                mutation_steps=mutation_steps,
                panels=panels,
                quadrature_order=quadrature_order,
                refine_check=refine_check,
                refinement_tolerance=refinement_tolerance,
                pcn_draws=pcn_draws,
                pcn_warmup=pcn_warmup,
                seed=seed,
            )
            comparisons.append(comparison)
            smc = comparison.method("smc")
            lap = comparison.method("laplace")
            ref = comparison.reference
            q_value = ref.log_evidence if ref is not None else math.nan
            q_error = ref.log_evidence_se if ref is not None else math.nan
            ev = (
                smc.log_evidence,
                smc.log_evidence_se,
                q_value,
                q_error if ref is not None else 0.0,
            )
            row_values = (
                smc.log_evidence,
                smc.log_evidence_se,
                comparison.smc_log_evidence_run_std,
                q_value,
                q_error,
                lap.log_evidence,
            )
        if base is None:
            base = ev
        lnb = ev[0] - base[0]
        se = math.hypot(ev[1], base[1])
        q_lnb = ev[2] - base[2] if not math.isnan(ev[2] + base[2]) else math.nan
        q_err = ev[3] + base[3]
        if spec is models[0]:
            category = "baseline"
        else:
            category = evidence_category(lnb)
            if abs(lnb) <= 3.0 * se:
                category += " (within 3 Monte Carlo standard errors of zero)"
        rows.append(
            EvidenceRow(
                spec.name,
                spec.dimension,
                row_values[0],
                row_values[1],
                row_values[2],
                row_values[3],
                row_values[4],
                row_values[5],
                lnb,
                se,
                q_lnb,
                q_err,
                category,
            )
        )
    return ModelComparison(
        data.truth, models[0].name, tuple(rows), tuple(comparisons), SYNTHETIC_NOTICE
    )


@dataclass(frozen=True, slots=True)
class PriorSensitivityRow:
    """Quadrature ``ln B`` of a model against the baseline at one prior width.

    Attributes:
        prior_scale: Multiplier of the ``w`` prior widths.
        log_bayes_factor: ``ln Z_model - ln Z_baseline`` by quadrature.
        error: Sum of the refinement errors.

    References:
        [EQ-INV-APP-COSMO-014].

    """

    prior_scale: float
    log_bayes_factor: float
    error: float


def prior_width_sensitivity(
    data: ExpansionData,
    kind: str,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
    *,
    panels: int = 12,
    box_sd: float = 14.0,
) -> tuple[PriorSensitivityRow, ...]:
    """Return how ``ln B`` (model vs LCDM) moves with the ``w`` prior width.

    Every row uses the deterministic quadrature evidence (no Monte Carlo),
    LCDM being exact. The Occam factor of a model with free ``w`` parameters
    grows with the prior volume, so ``ln B`` decreases as the prior widens
    unless the data support the extra freedom; this sweep shows the
    dependence the single table of :func:`compare_models` hides
    [EQ-INV-APP-COSMO-014].

    Args:
        data: The synthetic data.
        kind: ``"wcdm"``, ``"cpl"`` or ``"pw2"`` (fixed ``Omega_m``, so at
            most two parameters).
        scales: Positive multipliers of the prior widths.
        panels: Quadrature panels per dimension.
        box_sd: Half-width of the box in Laplace standard deviations.

    Returns:
        One row per scale.

    Raises:
        ValueError: If ``kind`` is ``"lcdm"`` or a scale is not positive, or
            the Laplace fit fails.

    References:
        [EQ-INV-APP-COSMO-014].

    """
    if kind == "lcdm":
        raise ValueError("kind must have free parameters")
    baseline = build_problem(model_spec("lcdm"), data).fixed_log_evidence
    rows = []
    for scale in scales:
        problem = build_problem(model_spec(kind, prior_scale=scale), data)
        fit = laplace_fit(problem)
        if not fit.converged:
            raise ValueError(f"Laplace fit failed at prior_scale={scale}")
        _, ref, _ = _reference_summary(
            problem,
            fit,
            panels=panels,
            order=8,
            box_sd=box_sd,
            refine_check=True,
            refinement_tolerance=1e-4,
        )
        assert ref is not None
        rows.append(
            PriorSensitivityRow(
                scale, ref.log_normalizer - baseline, ref.refinement_log_error
            )
        )
    return tuple(rows)
