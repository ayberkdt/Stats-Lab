"""Generalized linear models fitted by iteratively reweighted least squares.

A GLM relates the mean ``mu_i = E[y_i]`` of an exponential-dispersion response
to a linear predictor through a link, ``g(mu_i) = eta_i = x_i' beta + o_i``,
with ``Var(y_i) = phi V(mu_i) / w_i``. Fisher scoring for ``beta`` is a
sequence of weighted least-squares problems (IRLS) on the working response
``z = eta - o + (y - mu) / mu'(eta)`` with working weights
``W = w mu'(eta)^2 / V(mu)``; every WLS solve uses the Householder QR of
:mod:`probstat.linalg`.

Supported families and links: Gaussian (identity), binomial (logit, probit),
Poisson (log), and Gamma (log, inverse). For binomial data the response is a
proportion ``y_i in [0, 1]`` and the prior weight is the number of trials.

Logistic and probit maximum-likelihood estimates do not exist under complete
or quasi-complete separation: the likelihood keeps increasing along a
direction ``d`` with ``(2 y_i - 1) x_i'd >= 0`` for every observation. The
fit reports this with a checked certificate instead of returning huge,
meaningless coefficients silently.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from probstat.linalg import (
    DEFAULT_RANK_TOLERANCE,
    HouseholderQR,
    Matrix,
    Vector,
    as_matrix,
    as_vector,
)

from .bayesian import student_t_quantile
from .fitting import information_criteria
from .linear_models import (
    DesignMatrix,
    _coerce_design,
    _factorize,
    _positive_weights,
    _validate_level,
)
from .sampling_distributions import standard_normal_sf, student_t_two_sided_p

FamilyName = Literal["gaussian", "binomial", "poisson", "gamma"]
LinkName = Literal["identity", "logit", "probit", "log", "inverse"]
SeparationPolicy = Literal["error", "warn"]

_STANDARD_NORMAL = NormalDist()
_ALLOWED_LINKS: dict[str, tuple[str, ...]] = {
    "gaussian": ("identity",),
    "binomial": ("logit", "probit"),
    "poisson": ("log",),
    "gamma": ("inverse", "log"),
}
_LOGIT_CLAMP = 30.0
_PROBIT_CLAMP = 8.0
_LOG_CLAMP = 700.0
_SEPARATION_MOVEMENT = 1e-4
_SEPARATION_RELATIVE = 1e-6


class GLMConvergenceWarning(UserWarning):
    """Warns that IRLS stopped at the iteration limit without converging.

    References:
        [REF-MCCULLAGH-NELDER-1989, Section 2.5].

    """


class SeparationWarning(UserWarning):
    """Warns that a binomial fit shows complete or quasi-complete separation.

    References:
        [REF-ALBERT-ANDERSON-1984].

    """


class SeparationError(ValueError):
    """Raised when binomial maximum-likelihood estimates do not exist.

    References:
        [REF-ALBERT-ANDERSON-1984].

    """


@dataclass(frozen=True, slots=True)
class GLMFamily:
    """An exponential-dispersion family paired with a link function.

    Attributes:
        name: ``"gaussian"``, ``"binomial"``, ``"poisson"``, or ``"gamma"``.
        link_name: ``"identity"``, ``"logit"``, ``"probit"``, ``"log"``, or
            ``"inverse"``.

    References:
        [REF-MCCULLAGH-NELDER-1989, Chapter 2], [REF-NELDER-WEDDERBURN-1972].

    """

    name: str
    link_name: str

    def __post_init__(self) -> None:
        """Reject unsupported family/link combinations.

        Raises:
            ValueError: If the family or link is not supported.

        References:
            [REF-MCCULLAGH-NELDER-1989, Chapter 2]. No equation is evaluated.

        """
        if self.name not in _ALLOWED_LINKS:
            raise ValueError(
                f"unknown family {self.name!r}; choose from {sorted(_ALLOWED_LINKS)}"
            )
        if self.link_name not in _ALLOWED_LINKS[self.name]:
            raise ValueError(
                f"link {self.link_name!r} is not supported for family {self.name!r}; "
                f"choose from {_ALLOWED_LINKS[self.name]}"
            )

    @property
    def fixed_dispersion(self) -> bool:
        """Return whether the dispersion is known to be one.

        Returns:
            ``True`` for binomial and Poisson families.

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2]. No equation is evaluated.

        """
        return self.name in ("binomial", "poisson")

    def link(self, mu: float) -> float:
        """Return ``eta = g(mu)``.

        Args:
            mu: Mean inside the family's valid range.

        Returns:
            Linear predictor value.

        Equation:
            [EQ-GLM-FAM-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2].

        """
        if self.link_name == "identity":
            return mu
        if self.link_name == "logit":
            return math.log(mu / (1.0 - mu))
        if self.link_name == "probit":
            return _STANDARD_NORMAL.inv_cdf(mu)
        if self.link_name == "log":
            return math.log(mu)
        return 1.0 / mu

    def inverse_link(self, eta: float) -> float:
        """Return ``mu = g^-1(eta)``; logit and probit predictors are clamped.

        Logit predictors are clipped to ``[-30, 30]`` and probit predictors to
        ``[-8, 8]`` so fitted probabilities stay strictly inside ``(0, 1)``.

        Args:
            eta: Linear predictor value.

        Returns:
            Mean value (``inf`` for a log link beyond ``eta = 700``).

        Equation:
            [EQ-GLM-FAM-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2].

        """
        if self.link_name == "identity":
            return eta
        if self.link_name == "logit":
            clipped = max(-_LOGIT_CLAMP, min(_LOGIT_CLAMP, eta))
            if clipped >= 0.0:
                return 1.0 / (1.0 + math.exp(-clipped))
            exponential = math.exp(clipped)
            return exponential / (1.0 + exponential)
        if self.link_name == "probit":
            clipped = max(-_PROBIT_CLAMP, min(_PROBIT_CLAMP, eta))
            return 0.5 * math.erfc(-clipped / math.sqrt(2.0))
        if self.link_name == "log":
            return math.exp(eta) if eta <= _LOG_CLAMP else math.inf
        return 1.0 / eta if eta != 0.0 else math.inf

    def mu_eta(self, eta: float) -> float:
        """Return the derivative ``d mu / d eta`` at ``eta``.

        Args:
            eta: Linear predictor value.

        Returns:
            Derivative of the inverse link (evaluated at the clamped value for
            logit and probit links).

        Equation:
            [EQ-GLM-FAM-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.5].

        """
        if self.link_name == "identity":
            return 1.0
        if self.link_name == "logit":
            clipped = max(-_LOGIT_CLAMP, min(_LOGIT_CLAMP, eta))
            exponential = math.exp(-abs(clipped))
            return exponential / (1.0 + exponential) ** 2
        if self.link_name == "probit":
            clipped = max(-_PROBIT_CLAMP, min(_PROBIT_CLAMP, eta))
            return math.exp(-0.5 * clipped * clipped) / math.sqrt(2.0 * math.pi)
        if self.link_name == "log":
            return math.exp(min(eta, _LOG_CLAMP))
        return -1.0 / (eta * eta)

    def variance(self, mu: float) -> float:
        """Return the variance function ``V(mu)``.

        Args:
            mu: Mean value.

        Returns:
            ``1``, ``mu(1-mu)``, ``mu``, or ``mu^2``.

        Equation:
            [EQ-GLM-FAM-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2].

        """
        if self.name == "gaussian":
            return 1.0
        if self.name == "binomial":
            return mu * (1.0 - mu)
        if self.name == "poisson":
            return mu
        return mu * mu

    def valid_mean(self, mu: float) -> bool:
        """Return whether ``mu`` lies in the family's mean space.

        Args:
            mu: Candidate mean.

        Returns:
            ``True`` inside ``(0, 1)`` (binomial), ``(0, inf)`` (Poisson and
            Gamma), or the real line (Gaussian).

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2]. No equation is evaluated.

        """
        if not math.isfinite(mu):
            return False
        if self.name == "binomial":
            return 0.0 < mu < 1.0
        if self.name in ("poisson", "gamma"):
            return mu > 0.0
        return True

    def unit_deviance(self, y: float, mu: float) -> float:
        """Return the unit deviance ``d(y, mu)``.

        Args:
            y: Observed response.
            mu: Fitted mean.

        Returns:
            Non-negative contribution to the deviance (before prior weights).

        Equation:
            [EQ-GLM-DEV-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.3].

        """
        if self.name == "gaussian":
            return (y - mu) ** 2
        if self.name == "binomial":
            return 2.0 * (_xlogy_ratio(y, mu) + _xlogy_ratio(1.0 - y, 1.0 - mu))
        if self.name == "poisson":
            return 2.0 * (_xlogy_ratio(y, mu) - (y - mu))
        return 2.0 * (-math.log(y / mu) + (y - mu) / mu)

    def initial_mean(self, y: float, weight: float) -> float:
        """Return the IRLS starting mean for one observation.

        Args:
            y: Observed response.
            weight: Prior weight (trials for binomial data).

        Returns:
            ``(w y + 1/2)/(w + 1)`` (binomial), ``y + 0.1`` (Poisson), or ``y``.

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.5]. No equation is evaluated.

        """
        if self.name == "binomial":
            return (weight * y + 0.5) / (weight + 1.0)
        if self.name == "poisson":
            return y + 0.1
        return y


def _xlogy_ratio(x: float, y: float) -> float:
    """Return ``x log(x / y)`` with the convention ``0 log 0 = 0``.

    References:
        [REF-MCCULLAGH-NELDER-1989, Section 2.3]. No named equation is needed.

    """
    return 0.0 if x == 0.0 else x * math.log(x / y)


def glm_family(
    family: FamilyName | str, link: LinkName | str | None = None
) -> GLMFamily:
    """Return a validated family/link pair, using the canonical default link.

    Args:
        family: ``"gaussian"``, ``"binomial"``, ``"poisson"``, or ``"gamma"``.
        link: Optional link; defaults to identity, logit, log, and inverse
            respectively.

    Returns:
        :class:`GLMFamily`.

    Raises:
        ValueError: If the combination is unsupported.

    Equation:
        [EQ-GLM-FAM-001].

    References:
        [REF-MCCULLAGH-NELDER-1989, Section 2.2].

    """
    if family not in _ALLOWED_LINKS:
        raise ValueError(
            f"unknown family {family!r}; choose from {sorted(_ALLOWED_LINKS)}"
        )
    return GLMFamily(name=family, link_name=link or _ALLOWED_LINKS[family][0])


@dataclass(frozen=True, slots=True)
class GLMResult:
    """Immutable IRLS fit of a generalized linear model.

    Attributes:
        family: Family name.
        link: Link name.
        coefficients: Maximum-likelihood ``beta``.
        standard_errors: Wald standard errors ``sqrt(phi (X'WX)^-1_jj)``.
        statistics: Wald ratios ``beta_j / se_j``.
        p_values: Two-sided p-values from the normal law (fixed dispersion)
            or Student t on ``n - p`` df (estimated dispersion).
        statistic_distribution: ``"normal"`` or ``"t"``.
        column_names: Design-column names.
        covariance: ``phi (X'WX)^-1`` at the final working weights.
        unscaled_covariance: ``(X'WX)^-1``.
        fitted_values: Fitted means ``mu``.
        linear_predictor: ``eta = X beta + offset``.
        working_weights: Final IRLS weights ``w mu'(eta)^2 / V(mu)``.
        prior_weights: Prior weights (binomial trials).
        offset: Offsets ``o``.
        deviance: Residual deviance ``sum w d(y, mu)``.
        null_deviance: Deviance of the intercept-only (or offset-only) model.
        df_residual: ``n - p``.
        df_null: ``n - 1`` with an intercept, otherwise ``n``.
        pearson_chi2: ``sum w (y - mu)^2 / V(mu)``.
        dispersion: ``1`` (binomial, Poisson) or ``X^2 / (n - p)``.
        dispersion_estimated: Whether ``dispersion`` was estimated.
        deviance_residuals: ``sign(y - mu) sqrt(w d(y, mu))``.
        pearson_residuals: ``sqrt(w) (y - mu) / sqrt(V(mu))``.
        log_likelihood: Full log-likelihood at the fit.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``k = p`` (+1 for an estimated dispersion).
        iterations: IRLS iterations used.
        converged: Whether the relative coefficient step fell below tolerance.
        separation_detected: Whether a separation certificate was found.
        separated_observations: Indices perfectly predicted along the
            separating direction.
        n_observations: ``n``.
        n_parameters: ``p``.

    References:
        [REF-MCCULLAGH-NELDER-1989, Chapter 2].

    """

    family: str
    link: str
    coefficients: Vector
    standard_errors: Vector
    statistics: Vector
    p_values: Vector
    statistic_distribution: str
    column_names: tuple[str, ...]
    covariance: Matrix
    unscaled_covariance: Matrix
    fitted_values: Vector
    linear_predictor: Vector
    working_weights: Vector
    prior_weights: Vector
    offset: Vector
    deviance: float
    null_deviance: float
    df_residual: int
    df_null: int
    pearson_chi2: float
    dispersion: float
    dispersion_estimated: bool
    deviance_residuals: Vector
    pearson_residuals: Vector
    log_likelihood: float
    aic: float
    bic: float
    parameter_count: int
    iterations: int
    converged: bool
    separation_detected: bool
    separated_observations: tuple[int, ...]
    n_observations: int
    n_parameters: int

    def coefficient_intervals(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float], ...]:
        """Return Wald intervals ``beta_j +- q se_j``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per coefficient, with a normal quantile
            for fixed dispersion and a Student-t quantile otherwise.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-GLM-WALD-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.5].

        """
        checked = _validate_level(level)
        upper_level = 0.5 * (1.0 + checked)
        quantile = (
            _STANDARD_NORMAL.inv_cdf(upper_level)
            if self.statistic_distribution == "normal"
            else student_t_quantile(upper_level, self.df_residual)
        )
        return tuple(
            (b - quantile * s, b + quantile * s)
            for b, s in zip(self.coefficients, self.standard_errors, strict=True)
        )

    def predict(
        self,
        rows: Iterable[Iterable[object]],
        *,
        offset: Iterable[object] | None = None,
        scale: Literal["response", "link"] = "response",
    ) -> Vector:
        """Return predictions at new design rows.

        Args:
            rows: Design rows encoded like the fitting design.
            offset: Optional offsets for the new rows; default zero.
            scale: ``"response"`` for ``mu`` or ``"link"`` for ``eta``.

        Returns:
            Predictions in row order.

        Raises:
            ValueError: If shapes or ``scale`` are invalid.

        Equation:
            [EQ-GLM-FAM-001].

        References:
            [REF-MCCULLAGH-NELDER-1989, Section 2.2].

        """
        if scale not in ("response", "link"):
            raise ValueError("scale must be 'response' or 'link'")
        checked = as_matrix(rows, name="rows")
        if len(checked[0]) != self.n_parameters:
            raise ValueError("new rows must have one entry per design column")
        offsets = (
            as_vector(offset, name="offset")
            if offset is not None
            else tuple(0.0 for _ in checked)
        )
        if len(offsets) != len(checked):
            raise ValueError("offset must have one entry per row")
        family = GLMFamily(self.family, self.link)
        etas = tuple(
            math.fsum(b * x for b, x in zip(self.coefficients, row, strict=True)) + o
            for row, o in zip(checked, offsets, strict=True)
        )
        if scale == "link":
            return etas
        return tuple(family.inverse_link(eta) for eta in etas)


@dataclass(frozen=True, slots=True)
class _IRLSState:
    """Internal outcome of one IRLS run.

    References:
        [REF-MCCULLAGH-NELDER-1989, Section 2.5].

    """

    beta: Vector
    mu: Vector
    eta: Vector
    deviance: float
    iterations: int
    converged: bool
    history: tuple[Vector, ...]


def _deviance(family: GLMFamily, y: Vector, mu: Vector, w: Vector) -> float:
    """Return ``sum_i w_i d(y_i, mu_i)``.

    References:
        [EQ-GLM-DEV-001], [REF-MCCULLAGH-NELDER-1989, Section 2.3].

    """
    return math.fsum(
        wi * family.unit_deviance(yi, mi) for yi, mi, wi in zip(y, mu, w, strict=True)
    )


def _evaluate(
    family: GLMFamily, rows: Matrix, beta: Vector, offset: Vector
) -> tuple[Vector, Vector, bool]:
    """Return ``(eta, mu, valid)`` for coefficients ``beta``.

    References:
        [EQ-GLM-FAM-001], [REF-MCCULLAGH-NELDER-1989, Section 2.2].

    """
    eta = tuple(
        math.fsum(b * x for b, x in zip(beta, row, strict=True)) + o
        for row, o in zip(rows, offset, strict=True)
    )
    mu = tuple(family.inverse_link(value) for value in eta)
    valid = all(family.valid_mean(value) for value in mu)
    if family.link_name == "inverse":
        valid = valid and all(value > 0.0 for value in eta)
    return eta, mu, valid


def _working_system(
    family: GLMFamily, y: Vector, w: Vector, eta: Vector, mu: Vector, offset: Vector
) -> tuple[Vector, Vector]:
    """Return working weights and working responses for one IRLS step.

    References:
        [EQ-GLM-IRLS-001], [REF-MCCULLAGH-NELDER-1989, Section 2.5].

    """
    weights: list[float] = []
    responses: list[float] = []
    for yi, wi, ei, mi, oi in zip(y, w, eta, mu, offset, strict=True):
        derivative = family.mu_eta(ei)
        weights.append(wi * derivative * derivative / family.variance(mi))
        responses.append(ei - oi + (yi - mi) / derivative)
    return tuple(weights), tuple(responses)


def _weighted_factor(
    rows: Matrix, weights: Vector, names: tuple[str, ...], rank_tolerance: float
) -> HouseholderQR:
    """Factor ``W^(1/2) X``, rejecting rank deficiency.

    References:
        [EQ-LINALG-009], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    roots = tuple(math.sqrt(value) for value in weights)
    scaled = tuple(
        tuple(r * value for value in row) for r, row in zip(roots, rows, strict=True)
    )
    return _factorize(scaled, names, rank_tolerance)


def _irls(
    family: GLMFamily,
    rows: Matrix,
    y: Vector,
    w: Vector,
    offset: Vector,
    names: tuple[str, ...],
    *,
    max_iterations: int,
    tolerance: float,
    rank_tolerance: float,
) -> _IRLSState:
    """Run Fisher scoring as iteratively reweighted least squares.

    Step halving (up to 30 halvings) is applied when a step leaves the mean
    space or increases the deviance. Convergence is declared when the
    relative step ``max_j |delta beta_j| / (|beta_j| + 0.1)`` is below
    ``tolerance``. A relative-deviance criterion alone would stop Fisher
    scoring (linearly convergent for non-canonical links) with coefficient
    errors near ``sqrt(tolerance)``, because the deviance is flat at the
    optimum, and would declare diverging separated fits converged.

    References:
        [EQ-GLM-IRLS-001], [REF-MCCULLAGH-NELDER-1989, Section 2.5].

    """
    mu = tuple(family.initial_mean(yi, wi) for yi, wi in zip(y, w, strict=True))
    eta = tuple(family.link(value) for value in mu)
    deviance = _deviance(family, y, mu, w)
    beta: Vector | None = None
    history: list[Vector] = []
    converged = False
    iteration = 0
    for step in range(1, max_iterations + 1):
        iteration = step
        weights, responses = _working_system(family, y, w, eta, mu, offset)
        factor = _weighted_factor(rows, weights, names, rank_tolerance)
        roots = tuple(math.sqrt(value) for value in weights)
        candidate = factor.solve(
            tuple(r * z for r, z in zip(roots, responses, strict=True))
        )
        new_eta, new_mu, valid = _evaluate(family, rows, candidate, offset)
        new_deviance = _deviance(family, y, new_mu, w) if valid else math.inf
        if beta is None and not (valid and math.isfinite(new_deviance)):
            raise ValueError(
                "IRLS produced an invalid mean on the first step; "
                "try a different link (for example 'log')"
            )
        halvings = 0
        while beta is not None and (
            not valid
            or not math.isfinite(new_deviance)
            or new_deviance > deviance + 1e-12 * (abs(deviance) + 1.0)
        ):
            if halvings == 30:
                raise ValueError("IRLS step halving failed to reduce the deviance")
            halvings += 1
            candidate = tuple(
                0.5 * (a + b) for a, b in zip(candidate, beta, strict=True)
            )
            new_eta, new_mu, valid = _evaluate(family, rows, candidate, offset)
            new_deviance = _deviance(family, y, new_mu, w) if valid else math.inf
        relative_step = (
            max(
                abs(a - b) / (abs(a) + 0.1)
                for a, b in zip(candidate, beta, strict=True)
            )
            if beta is not None
            else math.inf
        )
        beta, eta, mu, deviance = candidate, new_eta, new_mu, new_deviance
        history.append(beta)
        if relative_step < tolerance:
            converged = True
            break
    assert beta is not None
    return _IRLSState(
        beta=beta,
        mu=mu,
        eta=eta,
        deviance=deviance,
        iterations=iteration,
        converged=converged,
        history=tuple(history[-4:]),
    )


def _separation_certificate(
    rows: Matrix, y: Vector, history: tuple[Vector, ...]
) -> tuple[int, ...] | None:
    """Return perfectly predicted indices if a separating direction is found.

    A direction ``d`` certifies non-existence of the binomial MLE when
    ``x_i'd >= 0`` for ``y_i = 1``, ``x_i'd <= 0`` for ``y_i = 0``, and
    ``x_i'd = 0`` for ``0 < y_i < 1``, with at least one strict inequality.
    The candidate is the last IRLS step. It is used only if it still moves
    some linear predictor by more than ``1e-4`` (the steps of a converged
    regular fit are far smaller), and the inequalities are checked to the
    relative tolerance ``1e-6 max_i |x_i'd|``.

    References:
        [EQ-GLM-SEP-001], [REF-ALBERT-ANDERSON-1984].

    """
    if len(history) < 2:
        return None
    candidates = [
        tuple(a - b for a, b in zip(history[-1], history[-2], strict=True)),
    ]
    for direction in candidates:
        moves = tuple(
            math.fsum(d * x for d, x in zip(direction, row, strict=True))
            for row in rows
        )
        scale = max(abs(value) for value in moves)
        if scale <= _SEPARATION_MOVEMENT:
            continue
        slack = _SEPARATION_RELATIVE * scale
        consistent = True
        for move, response in zip(moves, y, strict=True):
            if response == 1.0:
                consistent = move >= -slack
            elif response == 0.0:
                consistent = move <= slack
            else:
                consistent = abs(move) <= slack
            if not consistent:
                break
        if consistent:
            return tuple(i for i, move in enumerate(moves) if abs(move) > slack)
    return None


def _log_likelihood(
    family: GLMFamily, y: Vector, mu: Vector, w: Vector, deviance: float
) -> float:
    """Return the full log-likelihood with a plug-in dispersion when needed.

    References:
        [EQ-GLM-AIC-001], [REF-MCCULLAGH-NELDER-1989, Chapter 2].

    """
    if family.name == "binomial":
        terms = []
        for yi, mi, wi in zip(y, mu, w, strict=True):
            successes = wi * yi
            failures = wi - successes
            term = math.lgamma(wi + 1.0) - math.lgamma(successes + 1.0)
            term -= math.lgamma(failures + 1.0)
            if successes > 0.0:
                term += successes * math.log(mi)
            if failures > 0.0:
                term += failures * math.log(1.0 - mi)
            terms.append(term)
        return math.fsum(terms)
    if family.name == "poisson":
        return math.fsum(
            wi * ((yi * math.log(mi) if yi > 0.0 else 0.0) - mi - math.lgamma(yi + 1.0))
            for yi, mi, wi in zip(y, mu, w, strict=True)
        )
    total_weight = math.fsum(w)
    if family.name == "gaussian":
        n = len(y)
        if deviance <= 0.0:
            return math.inf
        value = -0.5 * n * (math.log(2.0 * math.pi * deviance / n) + 1.0)
        return value + 0.5 * math.fsum(math.log(wi) for wi in w)
    dispersion = deviance / total_weight
    if dispersion <= 0.0:
        return math.inf
    shape = 1.0 / dispersion
    terms = []
    for yi, mi, wi in zip(y, mu, w, strict=True):
        scale = mi * dispersion
        density = (
            -math.lgamma(shape)
            - shape * math.log(scale)
            + (shape - 1.0) * math.log(yi)
            - yi / scale
        )
        terms.append(wi * density)
    return math.fsum(terms)


def _validate_response(family: GLMFamily, y: Vector) -> None:
    """Check that responses lie in the family's support.

    References:
        [REF-MCCULLAGH-NELDER-1989, Chapter 2]. No equation is evaluated.

    """
    if family.name == "binomial" and any(not 0.0 <= v <= 1.0 for v in y):
        raise ValueError("binomial responses must be proportions in [0, 1]")
    if family.name == "poisson" and any(v < 0.0 for v in y):
        raise ValueError("Poisson responses must be non-negative")
    if family.name == "gamma" and any(v <= 0.0 for v in y):
        raise ValueError("Gamma responses must be strictly positive")


def fit_glm(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    family: FamilyName | str = "gaussian",
    link: LinkName | str | None = None,
    weights: Iterable[object] | None = None,
    offset: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
    on_separation: SeparationPolicy = "error",
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> GLMResult:
    """Fit a generalized linear model by iteratively reweighted least squares.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` rows; an intercept is any
            column identically one.
        response: Responses; proportions in ``[0, 1]`` for binomial data,
            non-negative counts for Poisson, positive values for Gamma.
        family: ``"gaussian"``, ``"binomial"``, ``"poisson"``, or ``"gamma"``.
        link: Optional link (see :func:`glm_family`).
        weights: Positive prior weights; binomial trials per observation.
        offset: Known additive terms in the linear predictor (for example
            ``log(exposure)`` for Poisson rates).
        column_names: Names for a raw design.
        max_iterations: Positive IRLS iteration limit.
        tolerance: Relative coefficient-step convergence tolerance in
            ``(0, 1)``.
        on_separation: ``"error"`` raises :class:`SeparationError`;
            ``"warn"`` emits :class:`SeparationWarning` and returns the
            diverged fit flagged with ``separation_detected``.
        rank_tolerance: Relative threshold for declaring a column dependent.

    Returns:
        :class:`GLMResult`.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        SeparationError: If binomial separation is certified and
            ``on_separation="error"``.
        ValueError: If inputs are invalid, ``n <= p``, or IRLS cannot find a
            valid step.

    Warns:
        GLMConvergenceWarning: If the iteration limit is reached.
        SeparationWarning: If separation is certified and
            ``on_separation="warn"``.

    Equation:
        [EQ-GLM-IRLS-001], [EQ-GLM-FAM-001], [EQ-GLM-DEV-001],
        [EQ-GLM-PEARSON-001], [EQ-GLM-WALD-001], [EQ-GLM-AIC-001],
        [EQ-GLM-SEP-001].

    References:
        [REF-MCCULLAGH-NELDER-1989, Chapters 2, 4, 6, and 8],
        [REF-NELDER-WEDDERBURN-1972], [REF-ALBERT-ANDERSON-1984].

    """
    glm = glm_family(family, link)
    if on_separation not in ("error", "warn"):
        raise ValueError("on_separation must be 'error' or 'warn'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    rows, names, has_intercept = _coerce_design(design, column_names)
    y = as_vector(response, name="response")
    n, p = len(rows), len(rows[0])
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    if n <= p:
        raise ValueError("GLM needs more observations than coefficients")
    _validate_response(glm, y)
    w = _positive_weights(weights, n)
    offsets = as_vector(offset, name="offset") if offset is not None else (0.0,) * n
    if len(offsets) != n:
        raise ValueError("offset must have one entry per observation")
    state = _irls(
        glm,
        rows,
        y,
        w,
        offsets,
        names,
        max_iterations=max_iterations,
        tolerance=tolerance,
        rank_tolerance=rank_tolerance,
    )
    separated: tuple[int, ...] | None = None
    if glm.name == "binomial":
        separated = _separation_certificate(rows, y, state.history)
    if separated is not None:
        message = (
            "binomial maximum-likelihood estimates do not exist: complete or "
            "quasi-complete separation (coefficients diverge; "
            f"{len(separated)} observation(s) perfectly predicted: "
            f"{list(separated)[:10]})"
        )
        if on_separation == "error":
            raise SeparationError(message)
        warnings.warn(message, SeparationWarning, stacklevel=2)
    elif not state.converged:
        warnings.warn(
            f"IRLS did not converge in {max_iterations} iterations",
            GLMConvergenceWarning,
            stacklevel=2,
        )
    working, _ = _working_system(glm, y, w, state.eta, state.mu, offsets)
    factor = _weighted_factor(rows, working, names, rank_tolerance)
    gram_inverse = factor.gram_inverse()
    df = n - p
    pearson = math.fsum(
        wi * (yi - mi) ** 2 / glm.variance(mi)
        for yi, mi, wi in zip(y, state.mu, w, strict=True)
    )
    estimated = not glm.fixed_dispersion
    dispersion = pearson / df if estimated else 1.0
    covariance = tuple(tuple(dispersion * v for v in row) for row in gram_inverse)
    errors = tuple(math.sqrt(max(covariance[j][j], 0.0)) for j in range(p))
    statistics: list[float] = []
    p_values: list[float] = []
    for b, s in zip(state.beta, errors, strict=True):
        ratio = b / s if s > 0.0 else math.copysign(math.inf, b)
        statistics.append(ratio)
        if not math.isfinite(ratio):
            p_values.append(0.0)
        elif estimated:
            p_values.append(student_t_two_sided_p(ratio, df))
        else:
            p_values.append(2.0 * standard_normal_sf(abs(ratio)))
    if has_intercept:
        ones = tuple((1.0,) for _ in range(n))
        null_state = _irls(
            glm,
            ones,
            y,
            w,
            offsets,
            ("(Intercept)",),
            max_iterations=max_iterations,
            tolerance=tolerance,
            rank_tolerance=rank_tolerance,
        )
        null_deviance = null_state.deviance
        df_null = n - 1
    else:
        null_mu = tuple(glm.inverse_link(o) for o in offsets)
        null_deviance = (
            _deviance(glm, y, null_mu, w)
            if all(glm.valid_mean(m) for m in null_mu)
            else math.nan
        )
        df_null = n
    deviance_residuals = tuple(
        math.copysign(math.sqrt(max(wi * glm.unit_deviance(yi, mi), 0.0)), yi - mi)
        for yi, mi, wi in zip(y, state.mu, w, strict=True)
    )
    pearson_residuals = tuple(
        math.sqrt(wi) * (yi - mi) / math.sqrt(glm.variance(mi))
        for yi, mi, wi in zip(y, state.mu, w, strict=True)
    )
    log_likelihood = _log_likelihood(glm, y, state.mu, w, state.deviance)
    k = p + (1 if estimated else 0)
    aic, _, bic = information_criteria(log_likelihood, k, n)
    return GLMResult(
        family=glm.name,
        link=glm.link_name,
        coefficients=state.beta,
        standard_errors=errors,
        statistics=tuple(statistics),
        p_values=tuple(p_values),
        statistic_distribution="t" if estimated else "normal",
        column_names=names,
        covariance=covariance,
        unscaled_covariance=gram_inverse,
        fitted_values=state.mu,
        linear_predictor=state.eta,
        working_weights=working,
        prior_weights=w,
        offset=offsets,
        deviance=state.deviance,
        null_deviance=null_deviance,
        df_residual=df,
        df_null=df_null,
        pearson_chi2=pearson,
        dispersion=dispersion,
        dispersion_estimated=estimated,
        deviance_residuals=deviance_residuals,
        pearson_residuals=pearson_residuals,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        parameter_count=k,
        iterations=state.iterations,
        converged=state.converged,
        separation_detected=separated is not None,
        separated_observations=separated or (),
        n_observations=n,
        n_parameters=p,
    )
