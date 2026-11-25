"""Expert elicitation: laws from stated quantiles, three-point judgements, pooling.

In early design there are no data for inputs such as a specific impulse, an
efficiency, a drag coefficient or a surface absorptance; the input law comes
from engineers' judgement ("most likely 0.28, I would be surprised below 0.26
or above 0.30"). This module turns such statements into laws and combines
several experts.

Elicited quantiles to a law:

- :func:`fit_from_quantiles` fits one family to ``(probability, value)`` pairs
  by weighted least squares on the quantile function,
  ``min_theta sum_i w_i (Q(p_i; theta) - v_i)^2``. The solution is closed form
  when the model is linear in the parameters (normal, Student-t with fixed
  degrees of freedom, uniform; the two-pair lognormal) and a
  Levenberg-Marquardt refinement otherwise (gamma, Weibull, beta on a stated
  interval, triangular, PERT). The result carries the per-quantile residuals
  and the quantiles the fitted law implies at other probabilities, so the
  expert can be shown what the fit says and object.
- :func:`compare_elicited_fits` fits several families and ranks them by their
  misfit; the choice of the law stays with the analyst.

Three-point judgements: :func:`pert_from_three_points` and
:func:`triangular_from_three_points` take absolute bounds;
:func:`pert_from_quantiles` and :func:`triangular_from_quantiles` take the
"low" and "high" as quantiles (for example 5 % and 95 %), solve for the
absolute bounds, and report when no such law exists.

Pooling several experts' laws (each result is a law of the library):

- :func:`linear_opinion_pool` returns the weighted
  :class:`~probstat.distributions.Mixture`;
- :func:`logarithmic_opinion_pool` returns a
  :class:`~probstat.distributions.Normal` when every law is normal and a
  numerically normalised :class:`LogarithmicPool` otherwise;
- :func:`quantile_average` (Vincentization) returns a
  :class:`~probstat.distributions.Normal` for normal experts and a
  :class:`VincentizedLaw` otherwise.

The pools differ in the properties they keep. The linear pool preserves
marginals (the pool of the marginals is the marginal of the pool) but is not
externally Bayesian: pooling and then updating with a likelihood differs from
updating each opinion and then pooling. The logarithmic pool with weights
summing to one is externally Bayesian, because the likelihood factors out of
``prod f_i^{w_i}``, but does not preserve marginals. Quantile averaging
preserves the location-scale family of its inputs and is a pool of quantile
functions, not of probabilities, so it has neither property in general. The
two properties are derived and checked numerically in the tests;
Genest and Zidek (1986) is the survey in which this taxonomy is set out
(confirmed bibliographically only), and Clemen and Winkler (1999) discuss the
practical choice for risk analysis.

Performance-based weights follow the classical model of Cooke (1991) as
described for EXCALIBUR by Hanea and Nane (2021), whose chapter was compared
with the code (calibration statistic and its ``k`` degrees of freedom,
information, intrinsic range with the 10 % overshoot default, weights and
cut-off optimisation agree; its worked values are pinned in the tests). Cooke's
book itself was not opened. Each expert states quantiles for *seed variables*
whose realizations are known:

- :func:`cooke_calibration` is the p-value of the likelihood-ratio statistic
  ``2 N I(s, p)`` against the chi-square law with ``k`` degrees of freedom,
  ``s`` the empirical and ``p`` the theoretical inter-quantile bin
  frequencies;
- :func:`cooke_information` is the mean relative information of the expert's
  piecewise-uniform density with respect to a uniform or log-uniform
  background on the intrinsic range (extended by an overshoot);
- :func:`cooke_weights` combines them, with the calibration cut-off;
- :func:`cooke_classical_model` runs the whole procedure with the global
  weights decision maker and optimises the cut-off by the decision maker's
  own calibration times information;
- :func:`cooke_decision_maker_law` returns the pooled law of one item.

Relation to neighbouring tools: ``distributions.PERT`` and ``Triangular`` are
the laws returned by the three-point functions; ``fit_distribution`` fits data,
this module fits judgements; the SHELF package fits by least squares on the
distribution function, this module on the quantile function.

Assumptions: the elicited quantiles are the expert's honest, coherent
statements (strictly increasing values). A fit is a summary of those
statements, not an estimate with sampling error. Cooke's calibration p-value
is the chi-square approximation of the likelihood-ratio test, adequate only
when the number of seed variables is not small, and treats the seed variables
as independent. The decision maker uses global (not item-specific) weights and
a common background and intrinsic range for all experts on an item.

References:
    [REF-OHAGAN-ETAL-2006], [REF-COOKE-1991], [REF-HANEA-NANE-2021],
    [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999],
    [REF-SHELF-OAKLEY-OHAGAN], [REF-NASA-PRA-2011], [REF-MALCOLM-1959],
    [REF-VOSE-2008].

"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from probstat.distributions import (
    PERT,
    Affine,
    Beta,
    ChiSquare,
    Gamma,
    LogNormal,
    Mixture,
    Normal,
    StudentT,
    Triangular,
    Uniform,
    Weibull,
)
from probstat.foundations.validation import require_finite_real
from probstat.numerics.gauss_kronrod import kronrod_panel
from probstat.numerics.integration import integrate
from probstat.optimization import brent_root, least_squares

__all__ = [
    "COOKE_PROBABILITIES",
    "ELICITATION_FAMILIES",
    "CalibrationScore",
    "CookeResult",
    "CutoffCandidate",
    "DecisionMakerScore",
    "ElicitedFitComparison",
    "ExpertAssessment",
    "InformationScore",
    "LogarithmicPool",
    "QuantileFit",
    "ThreePointFit",
    "VincentizedLaw",
    "compare_elicited_fits",
    "cooke_bin_probabilities",
    "cooke_calibration",
    "cooke_classical_model",
    "cooke_decision_maker_law",
    "cooke_information",
    "cooke_intrinsic_ranges",
    "cooke_weights",
    "fit_from_quantiles",
    "linear_opinion_pool",
    "logarithmic_opinion_pool",
    "pert_from_quantiles",
    "pert_from_three_points",
    "quantile_average",
    "triangular_from_quantiles",
    "triangular_from_three_points",
]

ELICITATION_FAMILIES = (
    "normal",
    "student_t",
    "lognormal",
    "gamma",
    "weibull",
    "beta",
    "triangular",
    "pert",
    "uniform",
)
DEFAULT_FEEDBACK_PROBABILITIES = (0.01, 0.05, 0.5, 0.95, 0.99)
COOKE_PROBABILITIES = (0.05, 0.5, 0.95)
BACKGROUNDS = ("uniform", "log-uniform")
Background = Literal["uniform", "log-uniform"]
_BOUNDED_FAMILIES = ("beta", "triangular", "pert")
_ALIASES = {"studentt": "student_t", "t": "student_t", "log_normal": "lognormal"}
_NORMAL = Normal(0.0, 1.0)
_FAILURES = (ValueError, ArithmeticError)
_PENALTY = 1.0e12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _reals(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    """Return ``values`` as a tuple of finite floats.

    Raises:
        TypeError: If ``values`` is not a sequence or an entry is not real.
        ValueError: If an entry is non-finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a sequence of real numbers")
    return tuple(require_finite_real(v, name=name) for v in values)


def _increasing(values: tuple[float, ...], *, name: str) -> None:
    """Require strictly increasing entries.

    Raises:
        ValueError: If two consecutive entries are not strictly increasing.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")


def _probability_grid(
    probabilities: Sequence[float], *, name: str = "probabilities", minimum: int = 1
) -> tuple[float, ...]:
    """Validate strictly increasing probabilities inside ``(0, 1)``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than ``minimum`` entries are given, an entry is
            outside ``(0, 1)``, or the entries do not increase strictly.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    grid = _reals(probabilities, name=name)
    if len(grid) < minimum:
        raise ValueError(f"{name} needs at least {minimum} entries")
    if any(not 0.0 < p < 1.0 for p in grid):
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    _increasing(grid, name=name)
    return grid


def _positive_weights(
    weights: Sequence[float] | None, count: int, *, name: str = "weights"
) -> tuple[float, ...]:
    """Return positive finite weights (equal when ``None``), not normalised.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the length is wrong or an entry is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if weights is None:
        return (1.0,) * count
    result = _reals(weights, name=name)
    if len(result) != count:
        raise ValueError(f"{name} must have {count} entries")
    if any(w <= 0.0 for w in result):
        raise ValueError(f"{name} must be positive")
    return result


def _normalised(weights: tuple[float, ...]) -> tuple[float, ...]:
    """Return ``weights`` scaled to sum to one.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


def _check_laws(laws: Sequence[Any]) -> tuple[Any, ...]:
    """Require at least one law with the structural marginal interface.

    Raises:
        TypeError: If an entry lacks ``pdf``, ``cdf``, ``quantile``, ``mean``
            or ``variance``.
        ValueError: If no law is given.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    result = tuple(laws)
    if not result:
        raise ValueError("at least one law is required")
    for index, law in enumerate(result):
        for attribute in ("pdf", "cdf", "quantile", "mean", "variance"):
            if not hasattr(law, attribute):
                raise TypeError(f"law {index} lacks required {attribute!r}")
    return result


def _bounds_pair(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    """Validate an interval ``(low, high)`` with ``low < high``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If ``bounds`` is not a pair or ``low >= high``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if bounds is None:
        return None
    pair = _reals(bounds, name="bounds")
    if len(pair) != 2 or not pair[0] < pair[1]:
        raise ValueError("bounds must be a pair (low, high) with low < high")
    return pair[0], pair[1]


# ---------------------------------------------------------------------------
# Laws from elicited quantiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuantileFit:
    """A law fitted to elicited quantiles, with its residuals and feedback.

    Args:
        family: Family label (one of :data:`ELICITATION_FAMILIES`).
        law: The fitted law.
        parameters: ``(name, value)`` pairs, including fixed ones (``nu``,
            ``lam``, the interval of a bounded family).
        probabilities: Elicited cumulative probabilities.
        values: Elicited quantiles.
        fitted_values: ``Q(p_i)`` of the fitted law.
        residuals: ``fitted_values - values``.
        weights: Least-squares weights used.
        misfit: Weighted root-mean-square residual
            ``sqrt(sum w r^2 / sum w)``, in the units of the values.
        max_abs_residual: Largest ``|r_i|``.
        feedback: ``(probability, quantile)`` of the fitted law at the
            feedback probabilities, to show back to the expert.
        n_parameters: Number of fitted (free) parameters.
        method: ``"closed-form"`` or ``"least-squares"``.
        converged: Whether the optimiser reported convergence (always
            ``True`` for a closed form).

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """

    family: str
    law: Any
    parameters: tuple[tuple[str, float], ...]
    probabilities: tuple[float, ...]
    values: tuple[float, ...]
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    weights: tuple[float, ...]
    misfit: float
    max_abs_residual: float
    feedback: tuple[tuple[float, float], ...]
    n_parameters: int
    method: str
    converged: bool

    def parameter(self, name: str) -> float:
        """Return the parameter called ``name``.

        Raises:
            KeyError: If the fit has no such parameter.

        References:
            Project API policy [REF-PROBSTAT-API]. No mathematical equation is
            evaluated.

        """
        for key, value in self.parameters:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class ElicitedFitComparison:
    """Several families fitted to the same elicited quantiles.

    Args:
        fits: Fits ordered by increasing misfit (ties by fewer parameters).
        skipped: ``(family, reason)`` for families that could not be fitted
            to these statements.
        probabilities: Elicited probabilities.
        values: Elicited quantiles.

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """

    fits: tuple[QuantileFit, ...]
    skipped: tuple[tuple[str, str], ...]
    probabilities: tuple[float, ...]
    values: tuple[float, ...]

    @property
    def ranking(self) -> tuple[str, ...]:
        """Return the family labels from smallest to largest misfit.

        References:
            Project API policy [REF-PROBSTAT-API]. No mathematical equation is
            evaluated.

        """
        return tuple(fit.family for fit in self.fits)

    @property
    def lowest_misfit(self) -> QuantileFit:
        """Return the fit with the smallest misfit (not a recommendation).

        The misfit does not penalise parameters: a family with as many
        parameters as quantiles always fits exactly. The analyst chooses the
        law after reading every fit's residuals and feedback.

        References:
            Project API policy [REF-PROBSTAT-API]. No mathematical equation is
            evaluated.

        """
        return self.fits[0]


def _linear_fit(
    z: tuple[float, ...], values: tuple[float, ...], weights: tuple[float, ...]
) -> tuple[float, float]:
    """Return ``(a, b)`` minimising ``sum w (a + b z - v)^2``.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """
    total = math.fsum(weights)
    z_bar = math.fsum(w * x for w, x in zip(weights, z, strict=True)) / total
    v_bar = math.fsum(w * x for w, x in zip(weights, values, strict=True)) / total
    sxx = math.fsum(w * (x - z_bar) ** 2 for w, x in zip(weights, z, strict=True))
    sxy = math.fsum(
        w * (x - z_bar) * (v - v_bar)
        for w, x, v in zip(weights, z, values, strict=True)
    )
    slope = sxy / sxx
    return v_bar - slope * z_bar, slope


def _refine(
    build: Callable[[tuple[float, ...]], Any],
    starts: Sequence[tuple[float, ...]],
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    weights: tuple[float, ...],
) -> tuple[tuple[float, ...], bool]:
    """Minimise the weighted quantile misfit from several starts.

    Returns:
        The unconstrained parameters of the lowest cost and its convergence
        flag.

    Raises:
        ValueError: If no start gives a valid law.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-MORE-1978], [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

    """
    roots = tuple(math.sqrt(w) for w in weights)
    scale = max(1.0, max(abs(v) for v in values))

    def residual(point: tuple[float, ...]) -> tuple[float, ...]:
        """Return ``sqrt(w)(Q(p; theta) - v)`` with a penalty for invalid laws.

        References:
            [EQ-INF-ELICIT-001]. No new equation.

        """
        try:
            law = build(point)
            out = tuple(
                r * (law.quantile(p) - v)
                for r, p, v in zip(roots, probabilities, values, strict=True)
            )
        except _FAILURES:
            return (_PENALTY * scale,) * len(values)
        if any(not math.isfinite(x) for x in out):
            return (_PENALTY * scale,) * len(values)
        return out

    best: tuple[float, tuple[float, ...], bool] | None = None
    for start in starts:
        try:
            result = least_squares(residual, start, tolerance=1e-14, max_iterations=400)
        except _FAILURES:
            continue
        if best is None or result.cost < best[0]:
            best = (result.cost, tuple(result.x), result.converged)
    if best is None or best[0] >= 0.5 * (_PENALTY * scale) ** 2:
        raise ValueError("no admissible law reproduces these quantile statements")
    return best[1], best[2]


def _sigmoid(x: float) -> float:
    """Return ``1/(1 + exp(-x))`` without overflow.

    References:
        Project numerical policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    """Return ``log(p/(1-p))``.

    References:
        Project numerical policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    return math.log(p / (1.0 - p))


def _fit_linear_family(
    name: str,
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    weights: tuple[float, ...],
    df: float,
) -> tuple[Any, tuple[tuple[str, float], ...], str, bool, int]:
    """Closed-form weighted least squares for normal, Student-t and uniform.

    ``v = a + b z`` with ``z`` the standard quantile at ``p``: normal
    ``(mu, sigma) = (a, b)``, Student-t ``(mu, sigma) = (a, b)`` at fixed
    ``nu``, uniform ``(lower, upper) = (a, a + b)`` with ``z = p``.

    Raises:
        ValueError: If the least-squares scale is not positive or, for a
            uniform law, the interval does not contain the elicited values.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """
    if name == "normal":
        z = tuple(_NORMAL.quantile(p) for p in probabilities)
    elif name == "student_t":
        base = StudentT(df)
        z = tuple(base.quantile(p) for p in probabilities)
    else:
        z = probabilities
    a, b = _linear_fit(z, values, weights)
    if not b > 0.0:
        raise ValueError("the quantile statements imply a non-positive scale")
    if name == "normal":
        return Normal(a, b), (("mu", a), ("sigma", b)), "closed-form", True, 2
    if name == "student_t":
        law = StudentT(df, a, b)
        return law, (("nu", df), ("mu", a), ("sigma", b)), "closed-form", True, 2
    return Uniform(a, a + b), (("lower", a), ("upper", a + b)), "closed-form", True, 2


def _positive_starts(
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    weights: tuple[float, ...],
) -> tuple[float, float]:
    """Return a lognormal ``(mu, sigma)`` least-squares start on the log scale.

    Raises:
        ValueError: If a value is not positive or the log-scale scale is not
            positive.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006].

    """
    if any(v <= 0.0 for v in values):
        raise ValueError("this family needs positive values")
    z = tuple(_NORMAL.quantile(p) for p in probabilities)
    mu, sigma = _linear_fit(z, tuple(math.log(v) for v in values), weights)
    if not sigma > 0.0:
        raise ValueError("the quantile statements imply a non-positive scale")
    return mu, sigma


def _fit_positive_family(
    name: str,
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    weights: tuple[float, ...],
) -> tuple[Any, tuple[tuple[str, float], ...], str, bool, int]:
    """Fit a lognormal, gamma or Weibull law to positive quantiles.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """
    mu, sigma = _positive_starts(probabilities, values, weights)
    if name == "lognormal":
        if len(values) == 2:
            return (
                LogNormal(mu, sigma),
                (("mu", mu), ("sigma", sigma)),
                "closed-form",
                True,
                2,
            )

        def build_lognormal(u: tuple[float, ...]) -> Any:
            """Return ``LogNormal(u0, exp(u1))``.

            References:
                [EQ-INF-ELICIT-001]. No new equation.

            """
            return LogNormal(u[0], math.exp(u[1]))

        point, ok = _refine(
            build_lognormal,
            [(mu, math.log(sigma))],
            probabilities,
            values,
            weights,
        )
        return (
            build_lognormal(point),
            (("mu", point[0]), ("sigma", math.exp(point[1]))),
            "least-squares",
            ok,
            2,
        )
    if name == "weibull":
        # ln Q = ln scale + (1/shape) ln(-ln(1-p)): linear on the log scale.
        h = tuple(math.log(-math.log1p(-p)) for p in probabilities)
        intercept, slope = _linear_fit(h, tuple(math.log(v) for v in values), weights)
        if not slope > 0.0:
            raise ValueError("the quantile statements imply a non-positive scale")

        def build_weibull(u: tuple[float, ...]) -> Any:
            """Return ``Weibull(exp(u0), exp(u1))``.

            References:
                [EQ-INF-ELICIT-001]. No new equation.

            """
            return Weibull(math.exp(u[0]), math.exp(u[1]))

        starts = [(-math.log(slope), intercept), (0.0, math.log(values[-1]))]
        point, ok = _refine(build_weibull, starts, probabilities, values, weights)
        shape, scale = math.exp(point[0]), math.exp(point[1])
        return (
            Weibull(shape, scale),
            (("shape", shape), ("scale", scale)),
            "least-squares",
            ok,
            2,
        )
    # Gamma: starts from the lognormal-moment match and two shape guesses.
    mean = math.exp(mu + 0.5 * sigma**2)
    variance = (math.exp(sigma**2) - 1.0) * mean**2

    def build_gamma(u: tuple[float, ...]) -> Any:
        """Return ``Gamma(exp(u0), exp(u1))``.

        References:
            [EQ-INF-ELICIT-001]. No new equation.

        """
        return Gamma(math.exp(u[0]), math.exp(u[1]))

    median = values[len(values) // 2]
    starts = [(math.log(mean * mean / variance), math.log(mean / variance))]
    for shape0 in (0.5, 2.0, 10.0, 50.0):
        starts.append((math.log(shape0), math.log(shape0 / median)))
    point, ok = _refine(build_gamma, starts, probabilities, values, weights)
    shape, rate = math.exp(point[0]), math.exp(point[1])
    return (
        Gamma(shape, rate),
        (("shape", shape), ("rate", rate)),
        "least-squares",
        ok,
        2,
    )


def _fit_bounded_family(
    name: str,
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    weights: tuple[float, ...],
    bounds: tuple[float, float] | None,
    shape: float,
) -> tuple[Any, tuple[tuple[str, float], ...], str, bool, int]:
    """Fit a beta (stated interval), triangular or PERT law.

    With ``bounds`` the interval is fixed; beta then fits ``(alpha, beta)``
    and triangular and PERT fit only the mode. Without ``bounds`` triangular
    and PERT fit the lower end, the mode and the upper end.

    Raises:
        ValueError: If a beta law has no ``bounds``, a value lies outside the
            bounds, or fewer pairs than free parameters are given.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008], [REF-KOTZ-VANDORP-2004, Chapter 1].

    """
    if name == "beta":
        if bounds is None:
            raise ValueError("a beta fit needs the interval bounds=(low, high)")
    if bounds is not None and not all(bounds[0] < v < bounds[1] for v in values):
        raise ValueError("elicited values must lie strictly inside bounds")
    if name == "beta":
        assert bounds is not None
        low, high = bounds
        y = tuple((v - low) / (high - low) for v in values)
        mean_y, sd_y = _linear_fit(
            tuple(_NORMAL.quantile(p) for p in probabilities), y, weights
        )
        mean_y = min(max(mean_y, 0.02), 0.98)
        sd_y = min(max(sd_y, 1e-3), 0.9 * math.sqrt(mean_y * (1.0 - mean_y)))
        total = mean_y * (1.0 - mean_y) / sd_y**2 - 1.0

        def build_beta(u: tuple[float, ...]) -> Any:
            """Return ``Affine(Beta(exp(u0), exp(u1)), low, high - low)``.

            References:
                [EQ-INF-ELICIT-001]. No new equation.

            """
            return Affine(Beta(math.exp(u[0]), math.exp(u[1])), low, high - low)

        starts = [
            (math.log(mean_y * total), math.log((1.0 - mean_y) * total)),
            (math.log(2.0), math.log(2.0)),
            (0.0, 0.0),
            (math.log(2.0), math.log(6.0)),
            (math.log(6.0), math.log(2.0)),
        ]
        point, ok = _refine(build_beta, starts, probabilities, values, weights)
        a, b = math.exp(point[0]), math.exp(point[1])
        return (
            build_beta(point),
            (("alpha", a), ("beta", b), ("lower", low), ("upper", high)),
            "least-squares",
            ok,
            2,
        )

    def make(low: float, mode: float, high: float) -> Any:
        """Return the triangular or PERT law with these three points.

        References:
            [EQ-DIST-TRI-001], [EQ-DIST-PERT-001].

        """
        return (
            Triangular(low, mode, high)
            if name == "triangular"
            else PERT(low, mode, high, shape)
        )

    if bounds is not None:
        low, high = bounds

        def build_mode(u: tuple[float, ...]) -> Any:
            """Return the law with the mode ``low + (high - low) sigmoid(u0)``.

            References:
                [EQ-INF-ELICIT-001]. No new equation.

            """
            return make(low, low + (high - low) * _sigmoid(u[0]), high)

        point, ok = _refine(
            build_mode,
            [(-2.0,), (0.0,), (2.0,)],
            probabilities,
            values,
            weights,
        )
        mode = low + (high - low) * _sigmoid(point[0])
        parameters = (("lower", low), ("mode", mode), ("upper", high))
        return (
            build_mode(point),
            _tag_pert(name, parameters, shape),
            ("least-squares"),
            ok,
            1,
        )
    if len(values) < 3:
        raise ValueError("a three-point family needs at least 3 pairs without bounds")
    spread = values[-1] - values[0]

    def build_free(u: tuple[float, ...]) -> Any:
        """Return the law with ends ``v_min - R e^{u0}``, ``v_max + R e^{u2}``.

        References:
            [EQ-INF-ELICIT-001]. No new equation.

        """
        low_ = values[0] - spread * math.exp(u[0])
        high_ = values[-1] + spread * math.exp(u[2])
        return make(low_, low_ + (high_ - low_) * _sigmoid(u[1]), high_)

    half = math.log(0.5)
    free: list[tuple[float, ...]] = [(half, b, half) for b in (0.0, -1.5, 1.5)]
    free += [(math.log(0.2), 0.0, math.log(0.2)), (math.log(2.0), 0.0, math.log(2.0))]
    point, ok = _refine(build_free, free, probabilities, values, weights)
    law = build_free(point)
    if name == "triangular":
        parameters = (("lower", law.lower), ("mode", law.mode), ("upper", law.upper))
    else:
        parameters = (("lower", law.low), ("mode", law.mode), ("upper", law.high))
    return law, _tag_pert(name, parameters, shape), "least-squares", ok, 3


def _tag_pert(
    name: str, parameters: tuple[tuple[str, float], ...], shape: float
) -> tuple[tuple[str, float], ...]:
    """Append the fixed PERT shape ``lam`` to the parameter list.

    References:
        [REF-VOSE-2008]. No mathematical equation is evaluated.

    """
    return parameters + (("lam", shape),) if name == "pert" else parameters


def _canonical_family(family: str) -> str:
    """Return the canonical family label.

    Raises:
        TypeError: If ``family`` is not a string.
        ValueError: If the label is unknown.

    References:
        Project API policy [REF-PROBSTAT-API]. No mathematical equation is
        evaluated.

    """
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    key = family.strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in ELICITATION_FAMILIES:
        known = ", ".join(ELICITATION_FAMILIES)
        raise ValueError(f"unknown family {family!r}; known families: {known}")
    return key


def fit_from_quantiles(
    family: str,
    probabilities: Sequence[float],
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    bounds: tuple[float, float] | None = None,
    degrees_of_freedom: float = 5.0,
    shape: float = 4.0,
    feedback_probabilities: Sequence[float] = DEFAULT_FEEDBACK_PROBABILITIES,
) -> QuantileFit:
    """Fit a law to elicited ``(probability, value)`` pairs.

    The fit minimises ``sum_i w_i (Q(p_i; theta) - v_i)^2`` over the family's
    free parameters. It is closed form for ``normal``, ``student_t`` (fixed
    ``degrees_of_freedom``) and ``uniform`` (``Q`` is linear in the
    parameters) and for a two-pair ``lognormal``; otherwise a
    Levenberg-Marquardt refinement from several starts. Families: ``normal``,
    ``student_t`` (location-scale), ``lognormal``, ``gamma`` (shape, rate),
    ``weibull``, ``beta`` on the interval ``bounds`` (an affine map of a
    standard beta), ``triangular`` and ``pert`` (``bounds`` fix the ends and
    the mode is fitted; without ``bounds`` the ends are fitted too and at
    least three pairs are needed), ``uniform``. When the number of pairs
    equals the number of parameters and a law reproduces them, the residuals
    are zero up to the optimiser tolerance.

    Args:
        family: Family label (case-insensitive, ``-`` or space for ``_``).
        probabilities: Strictly increasing cumulative probabilities in
            ``(0, 1)``, at least two.
        values: Strictly increasing elicited quantiles.
        weights: Optional positive least-squares weights (equal by default).
        bounds: Interval ``(low, high)`` for ``beta``, ``triangular`` and
            ``pert`` (required for ``beta``); rejected for other families.
        degrees_of_freedom: Fixed Student-t degrees of freedom.
        shape: PERT shape weight ``lam`` (default 4).
        feedback_probabilities: Probabilities at which the fitted law's
            quantiles are reported to the expert.

    Returns:
        A :class:`QuantileFit` with residuals ``fitted - elicited`` and the
        feedback quantiles.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the family is unknown, the statements are not
            increasing, fewer pairs than free parameters are given, a
            positive family gets non-positive values, ``bounds`` is misused,
            or no admissible law reproduces the statements.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN],
        [REF-NASA-PRA-2011], [REF-MORE-1978].

    """
    name = _canonical_family(family)
    probs = _probability_grid(probabilities, minimum=2)
    vals = _reals(values, name="values")
    if len(vals) != len(probs):
        raise ValueError("probabilities and values must have the same length")
    _increasing(vals, name="values")
    w = _positive_weights(weights, len(probs))
    interval = _bounds_pair(bounds)
    if interval is not None and name not in _BOUNDED_FAMILIES:
        raise ValueError(f"bounds are not used by the {name} family")
    feedback_grid = _probability_grid(feedback_probabilities, name="feedback")
    df = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    lam = require_finite_real(shape, name="shape")
    if df <= 0.0 or lam <= 0.0:
        raise ValueError("degrees_of_freedom and shape must be positive")
    if name in ("normal", "student_t", "uniform"):
        law, params, method, ok, n_free = _fit_linear_family(name, probs, vals, w, df)
    elif name in ("lognormal", "gamma", "weibull"):
        law, params, method, ok, n_free = _fit_positive_family(name, probs, vals, w)
    else:
        law, params, method, ok, n_free = _fit_bounded_family(
            name, probs, vals, w, interval, lam
        )
    if len(vals) < n_free:
        raise ValueError(f"the {name} family needs at least {n_free} pairs")
    fitted = tuple(float(law.quantile(p)) for p in probs)
    residuals = tuple(f - v for f, v in zip(fitted, vals, strict=True))
    misfit = math.sqrt(
        math.fsum(x * r * r for x, r in zip(w, residuals, strict=True)) / math.fsum(w)
    )
    feedback = tuple((p, float(law.quantile(p))) for p in feedback_grid)
    return QuantileFit(
        name,
        law,
        params,
        probs,
        vals,
        fitted,
        residuals,
        w,
        misfit,
        max(abs(r) for r in residuals),
        feedback,
        n_free,
        method,
        ok,
    )


def compare_elicited_fits(
    probabilities: Sequence[float],
    values: Sequence[float],
    *,
    families: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    bounds: tuple[float, float] | None = None,
    degrees_of_freedom: float = 5.0,
    shape: float = 4.0,
    feedback_probabilities: Sequence[float] = DEFAULT_FEEDBACK_PROBABILITIES,
) -> ElicitedFitComparison:
    """Fit several families to the same statements and rank them by misfit.

    Every fit is returned with its residuals and feedback quantiles; the
    ranking is by the weighted RMS misfit (ties by fewer parameters) and is
    not a model-selection criterion, because a family with as many
    parameters as quantiles fits exactly. A family that cannot be fitted
    (``beta`` without ``bounds``, a positive family with a negative value, a
    three-point family with fewer than three pairs and no bounds) is listed
    in ``skipped`` with the reason instead of failing the comparison.

    Args:
        probabilities: Elicited probabilities, strictly increasing.
        values: Elicited quantiles, strictly increasing.
        families: Family labels; all of :data:`ELICITATION_FAMILIES` by
            default.
        weights: Optional positive least-squares weights.
        bounds: Interval for the bounded families; ignored by the others.
        degrees_of_freedom: Fixed Student-t degrees of freedom.
        shape: PERT shape weight.
        feedback_probabilities: Probabilities for the feedback quantiles.

    Returns:
        An :class:`ElicitedFitComparison`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the statements are invalid or no family can be
            fitted.

    Equation:
        [EQ-INF-ELICIT-001].

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN].

    """
    names = ELICITATION_FAMILIES if families is None else tuple(families)
    if not names:
        raise ValueError("at least one family is required")
    probs = _probability_grid(probabilities, minimum=2)
    vals = _reals(values, name="values")
    fits: list[QuantileFit] = []
    skipped: list[tuple[str, str]] = []
    for family in names:
        canonical = _canonical_family(family)
        try:
            fits.append(
                fit_from_quantiles(
                    canonical,
                    probs,
                    vals,
                    weights=weights,
                    bounds=bounds if canonical in _BOUNDED_FAMILIES else None,
                    degrees_of_freedom=degrees_of_freedom,
                    shape=shape,
                    feedback_probabilities=feedback_probabilities,
                )
            )
        except _FAILURES as error:
            if len(vals) != len(probs) or "strictly increasing" in str(error):
                raise
            skipped.append((canonical, str(error)))
    if not fits:
        raise ValueError(
            "no family could be fitted: "
            + "; ".join(f"{name}: {reason}" for name, reason in skipped)
        )
    fits.sort(key=lambda fit: (fit.misfit, fit.n_parameters))
    return ElicitedFitComparison(tuple(fits), tuple(skipped), probs, vals)


# ---------------------------------------------------------------------------
# Three-point judgements
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThreePointFit:
    """A three-point law whose bounds were solved from stated quantiles.

    Args:
        family: ``"pert"`` or ``"triangular"``.
        law: The law (:class:`~probstat.distributions.PERT` or
            :class:`~probstat.distributions.Triangular`).
        low: Solved absolute lower bound.
        mode: The stated most likely value.
        high: Solved absolute upper bound.
        probability_low: Probability of the stated lower quantile.
        probability_high: Probability of the stated upper quantile.
        quantile_low: Stated lower quantile.
        quantile_high: Stated upper quantile.

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008], [REF-OHAGAN-ETAL-2006].

    """

    family: str
    law: Any
    low: float
    mode: float
    high: float
    probability_low: float
    probability_high: float
    quantile_low: float
    quantile_high: float


def _three_point_values(low: float, mode: float, high: float) -> tuple[float, ...]:
    """Validate ``low < high`` and ``low <= mode <= high``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the ordering fails.

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008]. No mathematical equation is
        evaluated.

    """
    a = require_finite_real(low, name="low")
    m = require_finite_real(mode, name="mode")
    b = require_finite_real(high, name="high")
    if not a < b:
        raise ValueError("low must be below high")
    if not a <= m <= b:
        raise ValueError("mode must lie in [low, high]")
    return a, m, b


def pert_from_three_points(
    low: float, mode: float, high: float, *, shape: float = 4.0
) -> PERT:
    """Return the beta-PERT law of a (low, most likely, high) judgement.

    Args:
        low: Absolute minimum.
        mode: Most likely value in ``[low, high]``.
        high: Absolute maximum, above ``low``.
        shape: PERT shape weight ``lam`` (default 4, the mean
            ``(low + 4 mode + high)/6``).

    Returns:
        :class:`~probstat.distributions.PERT`.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``low < high`` or ``low <= mode <= high`` fails, or
            ``shape`` is not positive.

    Equation:
        [EQ-DIST-PERT-001].

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008].

    """
    a, m, b = _three_point_values(low, mode, high)
    return PERT(a, m, b, require_finite_real(shape, name="shape"))


def triangular_from_three_points(low: float, mode: float, high: float) -> Triangular:
    """Return the triangular law of a (low, most likely, high) judgement.

    Args:
        low: Absolute minimum.
        mode: Most likely value in ``[low, high]``.
        high: Absolute maximum, above ``low``.

    Returns:
        :class:`~probstat.distributions.Triangular`.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``low < high`` or ``low <= mode <= high`` fails.

    Equation:
        [EQ-DIST-TRI-001].

    References:
        [REF-KOTZ-VANDORP-2004, Chapter 1].

    """
    a, m, b = _three_point_values(low, mode, high)
    return Triangular(a, m, b)


def _solve_three_point(
    family: str,
    unit_quantile: Callable[[float, float], float],
    build: Callable[[float, float, float], Any],
    q_low: float,
    mode: float,
    q_high: float,
    p_low: float,
    p_high: float,
) -> ThreePointFit:
    """Solve the absolute bounds of a three-point law from two quantiles.

    With the mode at fraction ``m`` of ``[low, high]`` the law is
    ``X = low + W B_m`` and the two quantile equations reduce to one in ``m``,
    ``(mode - q_low)/(q_high - q_low) = (m - B_m(p_low)) / (B_m(p_high) -
    B_m(p_low))``, solved by Brent's method on ``[0, 1]``; then
    ``W = (q_high - q_low)/(B_m(p_high) - B_m(p_low))``.

    Raises:
        ValueError: If the inputs are mis-ordered or no ``m`` in ``[0, 1]``
            solves the equation, with the admissible interval of the mode.

    Equation:
        [EQ-INF-ELICIT-002].

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008], [REF-OHAGAN-ETAL-2006].

    """
    ql = require_finite_real(q_low, name="q_low")
    qh = require_finite_real(q_high, name="q_high")
    md = require_finite_real(mode, name="mode")
    pl = require_finite_real(p_low, name="p_low")
    ph = require_finite_real(p_high, name="p_high")
    if not 0.0 < pl < ph < 1.0:
        raise ValueError("need 0 < p_low < p_high < 1")
    if not ql < qh:
        raise ValueError("the lower quantile must be below the upper quantile")
    target = (md - ql) / (qh - ql)

    def ratio(m: float) -> float:
        """Return ``(m - B_m(p_low)) / (B_m(p_high) - B_m(p_low))``.

        References:
            [EQ-INF-ELICIT-002]. No new equation.

        """
        bl, bh = unit_quantile(pl, m), unit_quantile(ph, m)
        return (m - bl) / (bh - bl)

    at_zero, at_one = ratio(0.0), ratio(1.0)
    if not at_zero - 1e-12 <= target <= at_one + 1e-12:
        lo_mode, hi_mode = ql + at_zero * (qh - ql), ql + at_one * (qh - ql)
        raise ValueError(
            f"no {family} law has quantiles ({ql}, {qh}) at probabilities "
            f"({pl}, {ph}) and mode {md}; the mode must lie in "
            f"[{lo_mode:.6g}, {hi_mode:.6g}]"
        )
    if target <= at_zero:
        fraction = 0.0
    elif target >= at_one:
        fraction = 1.0
    else:
        fraction = brent_root(
            lambda m: ratio(m) - target, 0.0, 1.0, tolerance=1e-15
        ).root
    bl, bh = unit_quantile(pl, fraction), unit_quantile(ph, fraction)
    width = (qh - ql) / (bh - bl)
    low = md - fraction * width
    high = low + width
    return ThreePointFit(family, build(low, md, high), low, md, high, pl, ph, ql, qh)


def pert_from_quantiles(
    q_low: float,
    mode: float,
    q_high: float,
    *,
    p_low: float = 0.05,
    p_high: float = 0.95,
    shape: float = 4.0,
) -> ThreePointFit:
    """Return the PERT law with stated quantiles and most likely value.

    The expert states a most likely value and a low and a high *quantile*
    (default 5 % and 95 %) instead of absolute bounds; the absolute bounds
    that reproduce both quantiles are solved for. The mode may lie below
    ``q_low`` or above ``q_high`` for a strongly skewed law; no law exists
    when it lies outside the interval reported in the error message.

    Args:
        q_low: Value with ``P(X <= q_low) = p_low``.
        mode: Most likely value.
        q_high: Value with ``P(X <= q_high) = p_high``, above ``q_low``.
        p_low: Probability of the lower quantile.
        p_high: Probability of the upper quantile.
        shape: PERT shape weight.

    Returns:
        A :class:`ThreePointFit` with the solved ``low`` and ``high``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the ordering is invalid or no PERT law reproduces the
            statements.

    Equation:
        [EQ-INF-ELICIT-002], [EQ-DIST-PERT-001].

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008], [REF-OHAGAN-ETAL-2006].

    """
    lam = require_finite_real(shape, name="shape")
    if lam <= 0.0:
        raise ValueError("shape must be positive")
    return _solve_three_point(
        "pert",
        lambda p, m: PERT(0.0, m, 1.0, lam).quantile(p),
        lambda low, md, high: PERT(low, md, high, lam),
        q_low,
        mode,
        q_high,
        p_low,
        p_high,
    )


def triangular_from_quantiles(
    q_low: float,
    mode: float,
    q_high: float,
    *,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> ThreePointFit:
    """Return the triangular law with stated quantiles and most likely value.

    Like :func:`pert_from_quantiles` for the triangular law, whose quantile
    function is explicit.

    Args:
        q_low: Value with ``P(X <= q_low) = p_low``.
        mode: Most likely value.
        q_high: Value with ``P(X <= q_high) = p_high``, above ``q_low``.
        p_low: Probability of the lower quantile.
        p_high: Probability of the upper quantile.

    Returns:
        A :class:`ThreePointFit` with the solved ``low`` and ``high``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the ordering is invalid or no triangular law
            reproduces the statements.

    Equation:
        [EQ-INF-ELICIT-002], [EQ-DIST-TRI-001].

    References:
        [REF-KOTZ-VANDORP-2004, Chapter 1], [REF-OHAGAN-ETAL-2006].

    """
    return _solve_three_point(
        "triangular",
        lambda p, m: Triangular(0.0, m, 1.0).quantile(p),
        lambda low, md, high: Triangular(low, md, high),
        q_low,
        mode,
        q_high,
        p_low,
        p_high,
    )


# ---------------------------------------------------------------------------
# Opinion pools
# ---------------------------------------------------------------------------


def linear_opinion_pool(
    laws: Sequence[Any], weights: Sequence[float] | None = None
) -> Mixture:
    """Return the linear opinion pool ``f = sum_i w_i f_i``.

    The pool is the finite mixture of the experts' laws, so ``F = sum w_i
    F_i``, ``E[X] = sum w_i mu_i`` and ``Var X = sum w_i (sigma_i^2 + (mu_i -
    mu)^2)``. It preserves marginals (the pool of the marginal laws is the
    marginal of the pooled law) but is not externally Bayesian, and it is
    multimodal when the experts disagree.

    Args:
        laws: Expert laws with the marginal interface.
        weights: Positive weights (equal by default), normalised to sum one.

    Returns:
        :class:`~probstat.distributions.Mixture`.

    Raises:
        TypeError: If a law lacks the marginal interface.
        ValueError: If no law is given or a weight is not positive.

    Equation:
        [EQ-INF-ELICIT-003].

    References:
        [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999],
        [REF-NASA-PRA-2011].

    """
    components = _check_laws(laws)
    w = _normalised(_positive_weights(weights, len(components)))
    return Mixture(components, w)


def _log_density(law: Any, x: float) -> float:
    """Return ``log f(x)`` of a law (``-inf`` where the density is zero).

    References:
        [EQ-INF-ELICIT-003]. No new equation.

    """
    try:
        if hasattr(law, "log_pdf"):
            return float(law.log_pdf(x))
        density = float(law.pdf(x))
    except _FAILURES:
        return -math.inf
    return math.log(density) if density > 0.0 else -math.inf


@dataclass(frozen=True, slots=True)
class LogarithmicPool:
    """A numerically normalised logarithmic pool ``g ∝ prod_i f_i^{w_i}``.

    The density is ``g(x) = exp(sum_i w_i log f_i(x) - log_normaliser)``. The
    normalising constant, the distribution function, the quantiles and the
    moments are computed by Gauss-Kronrod quadrature on panels whose ends
    are quantiles of the experts' laws, over the hull ``[lower,
    upper]`` of the experts' ``[tail, 1 - tail]`` quantile ranges. Mass of
    the pool outside that interval is neglected; because every factor is
    raised to a power ``w_i < 1`` the pool's tails decay more slowly than the
    experts', so the neglected mass is of order ``tail`` to the power of the
    smallest weight at most. The object has the marginal interface ``pdf``, ``cdf``,
    ``quantile``, ``mean`` and ``variance`` and can be a mixture component.

    Args:
        laws: The experts' laws.
        weights: Normalised weights summing to one.
        lower: Lower end of the integration interval.
        upper: Upper end of the integration interval.
        log_normaliser: ``log`` of the integral of ``prod f_i^{w_i}``.
        edges: Panel boundaries in ``[lower, upper]``.
        cumulative: Pool mass up to each panel boundary (normalised, with
            ``cumulative[0] = 0`` and ``cumulative[-1] = 1``).

    References:
        [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999].

    """

    laws: tuple[Any, ...]
    weights: tuple[float, ...]
    lower: float
    upper: float
    log_normaliser: float
    edges: tuple[float, ...]
    cumulative: tuple[float, ...]

    def log_pdf(self, x: float) -> float:
        """Return ``sum_i w_i log f_i(x) - log_normaliser``.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        point = require_finite_real(x, name="x")
        total = 0.0
        for w, law in zip(self.weights, self.laws, strict=True):
            value = _log_density(law, point)
            if value == -math.inf:
                return -math.inf
            total += w * value
        return total - self.log_normaliser

    def pdf(self, x: float) -> float:
        """Return the pooled density ``exp(log_pdf(x))``.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -math.inf else 0.0

    def _panel_mass(self, a: float, b: float) -> float:
        """Return the integral of the pooled density over ``[a, b]``.

        References:
            [EQ-INF-ELICIT-003]. No new equation.

        """
        if b <= a:
            return 0.0
        return kronrod_panel(self.pdf, a, b)[0]

    def cdf(self, x: float) -> float:
        """Return ``P(X <= x)`` from the panel table and one partial panel.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        point = require_finite_real(x, name="x")
        if point <= self.lower:
            return 0.0
        if point >= self.upper:
            return 1.0
        index = bisect.bisect_right(self.edges, point) - 1
        mass = self.cumulative[index] + self._panel_mass(self.edges[index], point)
        return min(1.0, max(0.0, mass))

    def sf(self, x: float) -> float:
        """Return ``1 - cdf(x)``.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        return 1.0 - self.cdf(x)

    def quantile(self, probability: float) -> float:
        """Return the ``p``-quantile by bisection on the panel table and Brent.

        Raises:
            ValueError: If ``probability`` is not in ``[0, 1]``.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 <= p <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if p == 0.0:
            return self.lower
        if p == 1.0:
            return self.upper
        index = min(bisect.bisect_right(self.cumulative, p) - 1, len(self.edges) - 2)
        base = self.cumulative[index]
        left, right = self.edges[index], self.edges[index + 1]
        if self.cumulative[index + 1] <= p:
            return right
        return brent_root(
            lambda x: base + self._panel_mass(left, x) - p,
            left,
            right,
            tolerance=1e-14,
        ).root

    def _moment(self, centre: float, order: int) -> float:
        """Return ``E[(X - centre)^order]`` by panel quadrature.

        References:
            [EQ-INF-ELICIT-003]. No new equation.

        """
        return math.fsum(
            kronrod_panel(lambda x: (x - centre) ** order * self.pdf(x), a, b)[0]
            for a, b in zip(self.edges, self.edges[1:], strict=False)
        )

    @property
    def mean(self) -> float:
        """Return ``E[X]`` by panel quadrature.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        return self._moment(0.0, 1)

    @property
    def variance(self) -> float:
        """Return ``Var X`` by panel quadrature about the mean.

        Equation:
            [EQ-INF-ELICIT-003].

        References:
            [REF-GENEST-ZIDEK-1986].

        """
        return self._moment(self.mean, 2)


def logarithmic_opinion_pool(
    laws: Sequence[Any],
    weights: Sequence[float] | None = None,
    *,
    tail: float = 1e-12,
    panels_per_law: int = 100,
) -> Normal | LogarithmicPool:
    """Return the logarithmic opinion pool ``g ∝ prod_i f_i^{w_i}``.

    The weights are normalised to sum to one, which makes the pool
    externally Bayesian. If every law is a :class:`~probstat.distributions.Normal`
    the pool is the normal law with precision ``sum_i w_i / sigma_i^2`` and
    mean ``sum_i w_i mu_i / sigma_i^2`` divided by that precision, returned
    as a ``Normal``. Otherwise the pool is normalised numerically and
    returned as a :class:`LogarithmicPool`, a law object with the marginal
    interface (``pdf``, ``cdf``, ``quantile``, ``mean``, ``variance``) rather
    than a library family, because the pool of non-normal laws belongs to no
    named family. Unlike the linear pool it is unimodal for log-concave
    experts and concentrates between them; it does not preserve marginals.

    Args:
        laws: Expert laws with the marginal interface.
        weights: Positive weights (equal by default), normalised to sum one.
        tail: Tail probability defining each expert's effective range
            ``[Q(tail), Q(1 - tail)]``; the pool is integrated over the
            hull of those ranges.
        panels_per_law: Quantile panel boundaries per law for the quadrature.

    Returns:
        ``Normal`` for all-normal experts, else a :class:`LogarithmicPool`.

    Raises:
        TypeError: If a law lacks the marginal interface.
        ValueError: If no law is given, a weight is not positive, ``tail`` is
            outside ``(0, 0.5)``, or the pooled density is zero at every
            panel boundary (experts with disjoint supports: the pool is then
            not defined).

    Equation:
        [EQ-INF-ELICIT-003].

    References:
        [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999],
        [REF-NASA-PRA-2011].

    """
    components = _check_laws(laws)
    w = _normalised(_positive_weights(weights, len(components)))
    if all(isinstance(law, Normal) for law in components):
        precision = math.fsum(
            wi / law.sigma**2 for wi, law in zip(w, components, strict=True)
        )
        mean = (
            math.fsum(
                wi * law.mu / law.sigma**2
                for wi, law in zip(w, components, strict=True)
            )
            / precision
        )
        return Normal(mean, math.sqrt(1.0 / precision))
    eps = require_finite_real(tail, name="tail")
    if not 0.0 < eps < 0.5:
        raise ValueError("tail must lie in (0, 0.5)")
    count = int(require_finite_real(panels_per_law, name="panels_per_law"))
    if count < 4:
        raise ValueError("panels_per_law must be at least 4")
    lower = min(law.quantile(eps) for law in components)
    upper = max(law.quantile(1.0 - eps) for law in components)
    grid = [(k + 0.5) / count for k in range(count)]
    points = {lower, upper}
    for law in components:
        for p in grid:
            x = float(law.quantile(p))
            if lower < x < upper:
                points.add(x)
    step = (upper - lower) / 64.0
    points.update(lower + k * step for k in range(1, 64))
    edges = tuple(sorted(points))

    def log_g(x: float) -> float:
        """Return ``sum_i w_i log f_i(x)`` (``-inf`` where a density is zero).

        References:
            [EQ-INF-ELICIT-003]. No new equation.

        """
        total = 0.0
        for wi, law in zip(w, components, strict=True):
            value = _log_density(law, x)
            if value == -math.inf:
                return -math.inf
            total += wi * value
        return total

    shift = max(log_g(x) for x in edges)
    if shift == -math.inf:
        raise ValueError("the pooled density vanishes on the overlap of the experts")
    masses = [
        kronrod_panel(lambda x: math.exp(log_g(x) - shift), a, b)[0]
        for a, b in zip(edges, edges[1:], strict=False)
    ]
    total = math.fsum(masses)
    running = 0.0
    cumulative = [0.0]
    for mass in masses:
        running += mass
        cumulative.append(min(1.0, running / total))
    cumulative[-1] = 1.0
    return LogarithmicPool(
        components,
        w,
        edges[0],
        edges[-1],
        shift + math.log(total),
        edges,
        tuple(cumulative),
    )


@dataclass(frozen=True, slots=True)
class VincentizedLaw:
    """The law whose quantile function is ``Q(p) = sum_i w_i Q_i(p)``.

    The mean is ``sum_i w_i mu_i`` exactly; the density at ``x`` is
    ``1 / sum_i w_i / f_i(Q_i(p))`` with ``p = F(x)``; the distribution
    function inverts ``Q`` by Brent's method on the logit scale. The object
    has the marginal interface and can be a mixture component.

    Args:
        laws: The experts' laws.
        weights: Normalised weights summing to one.

    References:
        [REF-CLEMEN-WINKLER-1999], [REF-GENEST-ZIDEK-1986].

    """

    laws: tuple[Any, ...]
    weights: tuple[float, ...]

    def quantile(self, probability: float) -> float:
        """Return ``sum_i w_i Q_i(p)``.

        Raises:
            ValueError: If ``probability`` is not in ``(0, 1)``.

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 < p < 1.0:
            raise ValueError("probability must lie in (0, 1)")
        return math.fsum(
            w * float(law.quantile(p))
            for w, law in zip(self.weights, self.laws, strict=True)
        )

    def cdf(self, x: float) -> float:
        """Return ``F(x)``, the inverse of the averaged quantile function.

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        point = require_finite_real(x, name="x")

        def gap(t: float) -> float:
            """Return ``Q(sigmoid(t)) - x``.

            References:
                [EQ-INF-ELICIT-004]. No new equation.

            """
            return self.quantile(_sigmoid(t)) - point

        try:
            low, high = gap(-36.0), gap(36.0)
        except _FAILURES:
            raise ValueError("x is outside the numerically resolvable range") from None
        if low >= 0.0:
            return 0.0
        if high <= 0.0:
            return 1.0
        return _sigmoid(brent_root(gap, -36.0, 36.0, tolerance=1e-14).root)

    def sf(self, x: float) -> float:
        """Return ``1 - cdf(x)``.

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        return 1.0 - self.cdf(x)

    def pdf(self, x: float) -> float:
        """Return ``1 / sum_i (w_i / f_i(Q_i(F(x))))``.

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        p = self.cdf(x)
        if not 0.0 < p < 1.0:
            return 0.0
        total = 0.0
        for w, law in zip(self.weights, self.laws, strict=True):
            density = float(law.pdf(float(law.quantile(p))))
            if density <= 0.0:
                return 0.0
            total += w / density
        return 1.0 / total

    @property
    def mean(self) -> float:
        """Return ``sum_i w_i E[X_i]``.

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        return math.fsum(
            w * float(law.mean) for w, law in zip(self.weights, self.laws, strict=True)
        )

    @property
    def variance(self) -> float:
        """Return ``int_0^1 (Q(p) - mu)^2 dp`` by double-exponential quadrature.

        Raises:
            ArithmeticError: If the integral does not converge (for example
                a heavy-tailed expert with infinite variance).

        Equation:
            [EQ-INF-ELICIT-004].

        References:
            [REF-CLEMEN-WINKLER-1999].

        """
        centre = self.mean
        result = integrate(
            lambda p: (self.quantile(p) - centre) ** 2, 0.0, 1.0, rel_tol=1e-10
        )
        if not result.converged:
            raise ArithmeticError("the variance integral did not converge")
        return float(result.value)


def quantile_average(
    laws: Sequence[Any], weights: Sequence[float] | None = None
) -> Normal | VincentizedLaw:
    """Return the quantile-averaged (Vincentized) pool ``Q = sum_i w_i Q_i``.

    Averaging quantile functions keeps a location-scale family closed: for
    normal experts the result is the :class:`~probstat.distributions.Normal`
    with the weighted mean of the locations and of the scales (not of the
    variances), returned as such. For other laws the result is a
    :class:`VincentizedLaw`. The pool never widens beyond the experts' range
    of quantiles and, unlike the linear pool, stays unimodal for experts of
    one location-scale family; it preserves neither marginals nor external
    Bayesianity in general.

    Args:
        laws: Expert laws with the marginal interface.
        weights: Positive weights (equal by default), normalised to sum one.

    Returns:
        ``Normal`` for all-normal experts, else a :class:`VincentizedLaw`.

    Raises:
        TypeError: If a law lacks the marginal interface.
        ValueError: If no law is given or a weight is not positive.

    Equation:
        [EQ-INF-ELICIT-004].

    References:
        [REF-CLEMEN-WINKLER-1999], [REF-GENEST-ZIDEK-1986].

    """
    components = _check_laws(laws)
    w = _normalised(_positive_weights(weights, len(components)))
    if all(isinstance(law, Normal) for law in components):
        return Normal(
            math.fsum(wi * law.mu for wi, law in zip(w, components, strict=True)),
            math.fsum(wi * law.sigma for wi, law in zip(w, components, strict=True)),
        )
    return VincentizedLaw(components, w)


# ---------------------------------------------------------------------------
# Cooke's classical model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationScore:
    """Statistical accuracy of one expert on the seed variables.

    Args:
        score: Calibration ``1 - F_{chi2(k)}(2 N I(s, p))``, the p-value of
            the likelihood-ratio statistic.
        statistic: ``2 N I(s, p)``.
        bin_counts: Realizations in each inter-quantile bin.
        empirical_probabilities: ``s_j``, the bin counts over ``N``.
        theoretical_probabilities: ``p_j``, from the stated probabilities.
        degrees_of_freedom: ``k``, the number of stated quantiles.
        n_realizations: ``N``.

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    score: float
    statistic: float
    bin_counts: tuple[int, ...]
    empirical_probabilities: tuple[float, ...]
    theoretical_probabilities: tuple[float, ...]
    degrees_of_freedom: int
    n_realizations: int


@dataclass(frozen=True, slots=True)
class InformationScore:
    """Mean relative information of one expert about a background measure.

    Args:
        score: Mean over items of the relative information.
        by_item: Relative information of each item.

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    score: float
    by_item: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExpertAssessment:
    """One expert's scores and weight in the classical model.

    Args:
        calibration: The :class:`CalibrationScore`.
        information: Mean information score.
        information_by_item: Information on each seed item.
        score: Unnormalised weight ``C I`` if the calibration reaches the
            cut-off, else ``0``.
        weight: Normalised weight of the decision maker.

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    calibration: CalibrationScore
    information: float
    information_by_item: tuple[float, ...]
    score: float
    weight: float


@dataclass(frozen=True, slots=True)
class DecisionMakerScore:
    """Calibration and information of a decision maker on the seed items.

    Args:
        calibration: Calibration score of the decision maker.
        information: Mean information score of the decision maker.
        score: ``calibration.score * information``.
        quantiles: The decision maker's quantiles at the stated
            probabilities, per seed item.

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    calibration: CalibrationScore
    information: float
    score: float
    quantiles: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class CutoffCandidate:
    """The global-weights decision maker for one calibration cut-off.

    Args:
        cutoff: Calibration cut-off ``alpha``.
        weights: Normalised expert weights.
        score: The decision maker's :class:`DecisionMakerScore`.

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    cutoff: float
    weights: tuple[float, ...]
    score: DecisionMakerScore


@dataclass(frozen=True, slots=True)
class CookeResult:
    """Outcome of Cooke's classical model on a set of seed variables.

    Args:
        probabilities: Stated cumulative probabilities.
        background: ``"uniform"`` or ``"log-uniform"``.
        overshoot: Fractional range extension of the intrinsic range.
        ranges: Intrinsic range ``(lower, upper)`` of each seed item.
        experts: Per-expert scores and weights.
        cutoff: The calibration cut-off used.
        optimized: Whether ``cutoff`` was chosen by maximising the decision
            maker's score.
        weights: Normalised expert weights at ``cutoff``.
        decision_maker: Score of the performance-weighted decision maker.
        equal_weight_decision_maker: Score of the equal-weight decision
            maker, for comparison.
        cutoff_scan: Every candidate cut-off examined (empty if ``cutoff``
            was given).

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """

    probabilities: tuple[float, ...]
    background: str
    overshoot: float
    ranges: tuple[tuple[float, float], ...]
    experts: tuple[ExpertAssessment, ...]
    cutoff: float
    optimized: bool
    weights: tuple[float, ...]
    decision_maker: DecisionMakerScore
    equal_weight_decision_maker: DecisionMakerScore
    cutoff_scan: tuple[CutoffCandidate, ...]


def cooke_bin_probabilities(
    probabilities: Sequence[float] = COOKE_PROBABILITIES,
) -> tuple[float, ...]:
    """Return the theoretical inter-quantile bin probabilities ``p_j``.

    For stated probabilities ``(0.05, 0.5, 0.95)`` the four bins have
    probabilities ``(0.05, 0.45, 0.45, 0.05)``.

    Args:
        probabilities: Strictly increasing probabilities in ``(0, 1)``.

    Returns:
        ``(p_1, p_2 - p_1, ..., 1 - p_k)``.

    Raises:
        ValueError: If the probabilities are invalid.

    Equation:
        [EQ-INF-ELICIT-005].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    grid = _probability_grid(probabilities)
    edges = (0.0, *grid, 1.0)
    return tuple(b - a for a, b in zip(edges, edges[1:], strict=False))


def _check_rows(
    rows: Sequence[Sequence[float]], width: int, *, name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate items x quantiles with strictly increasing rows of ``width``.

    Raises:
        ValueError: If a row has the wrong length or does not increase
            strictly.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    result = tuple(_reals(row, name=name) for row in rows)
    if not result:
        raise ValueError(f"{name} needs at least one item")
    for row in result:
        if len(row) != width:
            raise ValueError(f"every row of {name} needs {width} quantiles")
        _increasing(row, name=name)
    return result


def cooke_calibration(
    quantiles: Sequence[Sequence[float]],
    realizations: Sequence[float],
    *,
    probabilities: Sequence[float] = COOKE_PROBABILITIES,
) -> CalibrationScore:
    """Return Cooke's calibration score of one expert on the seed variables.

    A realization ``r`` of item ``i`` falls in bin ``j`` when ``q_{i,j} < r
    <= q_{i,j+1}`` (bin 0 is ``r <= q_{i,1}``, the last bin ``r >
    q_{i,k}``). With empirical bin frequencies ``s`` and theoretical ``p``,
    ``2 N I(s, p) = 2 N sum_j s_j ln(s_j/p_j)`` is asymptotically chi-square
    with ``k`` degrees of freedom when the expert is calibrated (``0 ln 0 =
    0``); the calibration is its upper-tail probability.

    Args:
        quantiles: For each seed item the expert's ``k`` strictly increasing
            quantiles at ``probabilities``.
        realizations: The ``N`` known values.
        probabilities: Stated probabilities (default ``0.05, 0.5, 0.95``).

    Returns:
        A :class:`CalibrationScore`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the shapes disagree or a row does not increase.

    Equation:
        [EQ-INF-ELICIT-005].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021, Section 3.2.1],
        [REF-CASELLA-BERGER-2002, Section 10.3].

    """
    grid = _probability_grid(probabilities)
    rows = _check_rows(quantiles, len(grid), name="quantiles")
    truth = _reals(realizations, name="realizations")
    if len(truth) != len(rows):
        raise ValueError("need one realization per item")
    theory = cooke_bin_probabilities(grid)
    counts = [0] * (len(grid) + 1)
    for row, value in zip(rows, truth, strict=True):
        counts[bisect.bisect_left(row, value)] += 1
    n = len(truth)
    frequencies = tuple(c / n for c in counts)
    information = math.fsum(
        s * math.log(s / p) for s, p in zip(frequencies, theory, strict=True) if s > 0.0
    )
    statistic = max(0.0, 2.0 * n * information)
    score = float(ChiSquare(float(len(grid))).sf(statistic))
    return CalibrationScore(
        score, statistic, tuple(counts), frequencies, theory, len(grid), n
    )


def _background(background: str) -> Background:
    """Validate the background label.

    Raises:
        ValueError: If the label is unknown.

    References:
        [REF-COOKE-1991]. No mathematical equation is evaluated.

    """
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}")
    return background  # type: ignore[return-value]


def _fraction(
    background: Background, a: float, b: float, lower: float, upper: float
) -> float:
    """Return the background measure of ``[a, b]`` inside ``[lower, upper]``.

    Equation:
        [EQ-INF-ELICIT-006].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    if background == "uniform":
        return (b - a) / (upper - lower)
    return math.log(b / a) / math.log(upper / lower)


def cooke_intrinsic_ranges(
    expert_quantiles: Sequence[Sequence[Sequence[float]]],
    realizations: Sequence[float] | None = None,
    *,
    overshoot: float = 0.1,
    background: str = "uniform",
) -> tuple[tuple[float, float], ...]:
    """Return the intrinsic range ``(lower, upper)`` of each item.

    The range of an item is the smallest interval holding every expert's
    quantiles and, when given, the realization; each end is then moved
    outward by ``overshoot`` times the width (``10 %`` by default) so that
    no bin of the background measure has zero mass. For the log-uniform
    background the width and the extension are measured on the logarithmic
    scale and all values must be positive.

    Args:
        expert_quantiles: ``expert_quantiles[e][i]`` is expert ``e``'s
            quantile row for item ``i``.
        realizations: Optional known value of each item (seed items).
        overshoot: Non-negative fractional extension.
        background: ``"uniform"`` or ``"log-uniform"``.

    Returns:
        One ``(lower, upper)`` per item.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``overshoot`` is negative, or the
            log-uniform background meets a non-positive value.

    Equation:
        [EQ-INF-ELICIT-006].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    kind = _background(background)
    k = require_finite_real(overshoot, name="overshoot")
    if k < 0.0:
        raise ValueError("overshoot must be non-negative")
    experts = [
        list(map(lambda r: _reals(r, name="quantiles"), e)) for e in expert_quantiles
    ]
    if not experts or not experts[0]:
        raise ValueError("need at least one expert and one item")
    items = len(experts[0])
    if any(len(e) != items for e in experts):
        raise ValueError("every expert must answer the same items")
    truth = None if realizations is None else _reals(realizations, name="realizations")
    if truth is not None and len(truth) != items:
        raise ValueError("need one realization per item")
    ranges: list[tuple[float, float]] = []
    for i in range(items):
        pool = [x for e in experts for x in e[i]]
        if truth is not None:
            pool.append(truth[i])
        if kind == "log-uniform":
            if any(x <= 0.0 for x in pool):
                raise ValueError("the log-uniform background needs positive values")
            pool = [math.log(x) for x in pool]
        low, high = min(pool), max(pool)
        pad = k * (high - low)
        low, high = low - pad, high + pad
        if kind == "log-uniform":
            low, high = math.exp(low), math.exp(high)
        if not low < high:
            raise ValueError("an item has a degenerate intrinsic range")
        ranges.append((low, high))
    return tuple(ranges)


def _item_information(
    row: tuple[float, ...],
    bounds: tuple[float, float],
    theory: tuple[float, ...],
    background: Background,
) -> float:
    """Return ``sum_j p_j ln(p_j / mu_j)`` of one item's piecewise-uniform law.

    Raises:
        ValueError: If a quantile lies outside the intrinsic range.

    Equation:
        [EQ-INF-ELICIT-006].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    lower, upper = bounds
    edges = (lower, *row, upper)
    if not lower <= row[0] or not row[-1] <= upper:
        raise ValueError("quantiles must lie inside the intrinsic range")
    total = 0.0
    for p, a, b in zip(theory, edges, edges[1:], strict=False):
        if not a < b:
            raise ValueError("a bin has zero width; increase the overshoot")
        total += p * math.log(p / _fraction(background, a, b, lower, upper))
    return total


def cooke_information(
    quantiles: Sequence[Sequence[float]],
    ranges: Sequence[tuple[float, float]],
    *,
    probabilities: Sequence[float] = COOKE_PROBABILITIES,
    background: str = "uniform",
) -> InformationScore:
    """Return Cooke's information score of one expert.

    The expert's density on item ``i`` is uniform within each inter-quantile
    bin of the intrinsic range ``[L_i, U_i]`` with the bin probabilities
    ``p_j``. Its relative information with respect to the uniform (or
    log-uniform) background is ``sum_j p_j ln(p_j / mu_j)``, ``mu_j`` the
    background measure of bin ``j``; the score is the mean over items. It does
    not depend on the realizations except through the intrinsic range.

    Args:
        quantiles: For each item the expert's strictly increasing quantiles.
        ranges: Intrinsic range of each item (see
            :func:`cooke_intrinsic_ranges`).
        probabilities: Stated probabilities.
        background: ``"uniform"`` or ``"log-uniform"``.

    Returns:
        An :class:`InformationScore`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, a quantile lies outside its range, or
            a bin has zero background measure.

    Equation:
        [EQ-INF-ELICIT-006].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    kind = _background(background)
    grid = _probability_grid(probabilities)
    rows = _check_rows(quantiles, len(grid), name="quantiles")
    if len(ranges) != len(rows):
        raise ValueError("need one range per item")
    theory = cooke_bin_probabilities(grid)
    by_item = tuple(
        _item_information(row, (float(r[0]), float(r[1])), theory, kind)
        for row, r in zip(rows, ranges, strict=True)
    )
    return InformationScore(math.fsum(by_item) / len(by_item), by_item)


def cooke_weights(
    calibration: Sequence[float],
    information: Sequence[float],
    cutoff: float,
) -> tuple[float, ...]:
    """Return the normalised performance weights ``C_e I_e 1{C_e >= alpha}``.

    Args:
        calibration: Calibration score of each expert.
        information: Information score of each expert.
        cutoff: Calibration cut-off ``alpha``; experts with a lower
            calibration get weight zero.

    Returns:
        Weights summing to one.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the lengths differ, a score is negative, or every
            expert has weight zero.

    Equation:
        [EQ-INF-ELICIT-007].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    c = _reals(calibration, name="calibration")
    info = _reals(information, name="information")
    if len(c) != len(info) or not c:
        raise ValueError("calibration and information need one entry per expert")
    if any(x < 0.0 for x in c + info):
        raise ValueError("scores must be non-negative")
    alpha = require_finite_real(cutoff, name="cutoff")
    raw = tuple(ci * ii if ci >= alpha else 0.0 for ci, ii in zip(c, info, strict=True))
    total = math.fsum(raw)
    if not total > 0.0:
        raise ValueError("every expert has weight zero at this cut-off")
    return tuple(x / total for x in raw)


def _decision_maker_item(
    weights: Sequence[float],
    rows: Sequence[tuple[float, ...]],
    bounds: tuple[float, float],
    theory: tuple[float, ...],
    grid: tuple[float, ...],
    background: Background,
) -> tuple[tuple[float, ...], float]:
    """Return the decision maker's quantiles and information on one item.

    The decision maker's density is ``sum_e w_e f_e`` with each ``f_e``
    piecewise uniform on its bins; it is constant on the cells cut by all the
    experts' bin edges, so its cell masses, quantiles (linear interpolation of
    the distribution function, on the log scale for the log-uniform
    background) and relative information are exact.

    Equation:
        [EQ-INF-ELICIT-006], [EQ-INF-ELICIT-007].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    lower, upper = bounds
    edge_sets = [(lower, *row, upper) for row in rows]
    cuts = sorted({x for edges in edge_sets for x in edges})
    masses: list[float] = []
    measures: list[float] = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        mass = 0.0
        for w, edges in zip(weights, edge_sets, strict=True):
            if w <= 0.0:
                continue
            j = bisect.bisect_right(edges, a) - 1
            j = min(j, len(theory) - 1)
            mass += (
                w
                * theory[j]
                * _fraction(background, a, b, lower, upper)
                / _fraction(background, edges[j], edges[j + 1], lower, upper)
            )
        masses.append(mass)
        measures.append(_fraction(background, a, b, lower, upper))
    information = math.fsum(
        m * math.log(m / mu) for m, mu in zip(masses, measures, strict=True) if m > 0.0
    )
    quantiles: list[float] = []
    for p in grid:
        running = 0.0
        for index, mass in enumerate(masses):
            if running + mass >= p or index == len(masses) - 1:
                a, b = cuts[index], cuts[index + 1]
                t = 0.0 if mass <= 0.0 else min(1.0, max(0.0, (p - running) / mass))
                if background == "uniform":
                    quantiles.append(a + t * (b - a))
                else:
                    quantiles.append(a * (b / a) ** t)
                break
            running += mass
    return tuple(quantiles), information


def _decision_maker_score(
    weights: Sequence[float],
    expert_rows: Sequence[Sequence[tuple[float, ...]]],
    realizations: tuple[float, ...],
    ranges: Sequence[tuple[float, float]],
    grid: tuple[float, ...],
    background: Background,
) -> DecisionMakerScore:
    """Score the decision maker with the given weights on the seed items.

    Equation:
        [EQ-INF-ELICIT-007].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    theory = cooke_bin_probabilities(grid)
    quantile_rows: list[tuple[float, ...]] = []
    information: list[float] = []
    for i, bounds in enumerate(ranges):
        q, info = _decision_maker_item(
            weights, [e[i] for e in expert_rows], bounds, theory, grid, background
        )
        quantile_rows.append(q)
        information.append(info)
    calibration = cooke_calibration(quantile_rows, realizations, probabilities=grid)
    mean_information = math.fsum(information) / len(information)
    return DecisionMakerScore(
        calibration,
        mean_information,
        calibration.score * mean_information,
        tuple(quantile_rows),
    )


def cooke_classical_model(
    expert_quantiles: Sequence[Sequence[Sequence[float]]],
    realizations: Sequence[float],
    *,
    probabilities: Sequence[float] = COOKE_PROBABILITIES,
    background: str = "uniform",
    overshoot: float = 0.1,
    cutoff: float | None = None,
) -> CookeResult:
    """Run Cooke's classical model with the global weights decision maker.

    Each expert's calibration and information are computed on the seed
    variables (:func:`cooke_calibration`, :func:`cooke_information`); the
    weight of expert ``e`` is ``C_e I_e`` if ``C_e >= alpha`` and ``0``
    otherwise, normalised. The decision maker pools the experts' piecewise
    uniform densities linearly with those weights. When ``cutoff`` is
    ``None`` the cut-off ``alpha`` is optimised: every positive expert
    calibration is a candidate, the decision maker is scored on the seed items
    (its own calibration times its own information) for each, and the best
    candidate (the smaller cut-off on a tie) is kept. Weights are *global*
    (one information score per expert), not item-specific.

    Args:
        expert_quantiles: ``expert_quantiles[e][i]`` is expert ``e``'s ``k``
            strictly increasing quantiles for seed item ``i``.
        realizations: The known value of each seed item.
        probabilities: Stated probabilities (default ``0.05, 0.5, 0.95``).
        background: ``"uniform"`` or ``"log-uniform"``.
        overshoot: Fractional extension of the intrinsic range.
        cutoff: Fixed calibration cut-off, or ``None`` to optimise it.

    Returns:
        A :class:`CookeResult` exposing each expert's calibration and
        information, the weights, the decision maker's scores and the
        optimisation scan.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, no expert has positive calibration
            (when optimising), or a fixed cut-off leaves no weight.

    Equation:
        [EQ-INF-ELICIT-005], [EQ-INF-ELICIT-006], [EQ-INF-ELICIT-007].

    References:
        [REF-COOKE-1991], [REF-NASA-PRA-2011], [REF-CLEMEN-WINKLER-1999].

    """
    kind = _background(background)
    grid = _probability_grid(probabilities)
    experts = tuple(
        tuple(_check_rows(e, len(grid), name="quantiles")) for e in expert_quantiles
    )
    if not experts:
        raise ValueError("need at least one expert")
    truth = _reals(realizations, name="realizations")
    if any(len(e) != len(truth) for e in experts):
        raise ValueError("every expert needs one quantile row per realization")
    ranges = cooke_intrinsic_ranges(
        experts, truth, overshoot=overshoot, background=kind
    )
    calibration = tuple(
        cooke_calibration(e, truth, probabilities=grid) for e in experts
    )
    information = tuple(
        cooke_information(e, ranges, probabilities=grid, background=kind)
        for e in experts
    )
    c_scores = tuple(c.score for c in calibration)
    i_scores = tuple(i.score for i in information)

    def candidate(alpha: float) -> CutoffCandidate:
        """Return the decision maker for the cut-off ``alpha``.

        Equation:
            [EQ-INF-ELICIT-007].

        References:
            [REF-COOKE-1991], [REF-HANEA-NANE-2021].

        """
        w = cooke_weights(c_scores, i_scores, alpha)
        return CutoffCandidate(
            alpha, w, _decision_maker_score(w, experts, truth, ranges, grid, kind)
        )

    scan: tuple[CutoffCandidate, ...] = ()
    if cutoff is None:
        levels = sorted({c for c in c_scores if c > 0.0})
        if not levels:
            raise ValueError("no expert has a positive calibration score")
        scan = tuple(candidate(alpha) for alpha in levels)
        best = scan[0]
        for item in scan[1:]:
            if item.score.score > best.score.score:
                best = item
    else:
        best = candidate(require_finite_real(cutoff, name="cutoff"))
    raw = tuple(
        c * i if c >= best.cutoff else 0.0
        for c, i in zip(c_scores, i_scores, strict=True)
    )
    assessments = tuple(
        ExpertAssessment(cal, info.score, info.by_item, r, w)
        for cal, info, r, w in zip(
            calibration, information, raw, best.weights, strict=True
        )
    )
    equal = (1.0 / len(experts),) * len(experts)
    return CookeResult(
        grid,
        kind,
        float(overshoot),
        ranges,
        assessments,
        best.cutoff,
        cutoff is None,
        best.weights,
        best.score,
        _decision_maker_score(equal, experts, truth, ranges, grid, kind),
        scan,
    )


@dataclass(frozen=True, slots=True)
class _LogUniformBin:
    """Law with density proportional to ``1/x`` on ``[lower, upper]``.

    References:
        [EQ-INF-ELICIT-006]. No new equation.

    """

    lower: float
    upper: float

    def _span(self) -> float:
        """Return ``ln(upper/lower)``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        return math.log(self.upper / self.lower)

    def pdf(self, x: float) -> float:
        """Return ``1/(x ln(upper/lower))`` on the support, else ``0``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        if x < self.lower or x > self.upper:
            return 0.0
        return 1.0 / (x * self._span())

    def cdf(self, x: float) -> float:
        """Return ``ln(x/lower)/ln(upper/lower)`` clipped to ``[0, 1]``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        if x <= self.lower:
            return 0.0
        if x >= self.upper:
            return 1.0
        return math.log(x / self.lower) / self._span()

    def quantile(self, probability: float) -> float:
        """Return ``lower (upper/lower)^p``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        return float(self.lower * (self.upper / self.lower) ** probability)

    @property
    def mean(self) -> float:
        """Return ``(upper - lower)/ln(upper/lower)``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        return (self.upper - self.lower) / self._span()

    @property
    def variance(self) -> float:
        """Return the variance ``(upper^2 - lower^2)/(2 ln(upper/lower)) - mean^2``.

        References:
            [EQ-INF-ELICIT-006]. No new equation.

        """
        second = (self.upper**2 - self.lower**2) / (2.0 * self._span())
        return second - self.mean**2


def cooke_decision_maker_law(
    weights: Sequence[float],
    expert_rows: Sequence[Sequence[float]],
    bounds: tuple[float, float],
    *,
    probabilities: Sequence[float] = COOKE_PROBABILITIES,
    background: str = "uniform",
) -> Mixture:
    """Return the decision maker's law on one item as a mixture of bins.

    Expert ``e`` contributes ``w_e p_j`` to the uniform (or log-uniform)
    law on bin ``j`` of the intrinsic range, so the result is the linear
    pool of the experts' piecewise-uniform densities. Use it for a target
    item with the weights of :func:`cooke_classical_model` and the range of
    :func:`cooke_intrinsic_ranges` computed without realization.

    Args:
        weights: Non-negative expert weights (normalised internally); experts
            with zero weight are dropped.
        expert_rows: Each expert's strictly increasing quantiles for the item.
        bounds: Intrinsic range ``(lower, upper)`` of the item.
        probabilities: Stated probabilities.
        background: ``"uniform"`` or ``"log-uniform"``.

    Returns:
        :class:`~probstat.distributions.Mixture`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, a quantile lies outside the range, a
            bin has zero width, or every weight is zero.

    Equation:
        [EQ-INF-ELICIT-007].

    References:
        [REF-COOKE-1991], [REF-HANEA-NANE-2021].

    """
    kind = _background(background)
    grid = _probability_grid(probabilities)
    w = _reals(weights, name="weights")
    rows = _check_rows(expert_rows, len(grid), name="expert_rows")
    if len(w) != len(rows):
        raise ValueError("need one weight per expert")
    if any(x < 0.0 for x in w) or not math.fsum(w) > 0.0:
        raise ValueError("weights must be non-negative with a positive sum")
    lower, upper = float(bounds[0]), float(bounds[1])
    theory = cooke_bin_probabilities(grid)
    total = math.fsum(w)
    components: list[Any] = []
    masses: list[float] = []
    for weight, row in zip(w, rows, strict=True):
        if weight <= 0.0:
            continue
        if not lower <= row[0] or not row[-1] <= upper:
            raise ValueError("quantiles must lie inside the intrinsic range")
        edges = (lower, *row, upper)
        for p, a, b in zip(theory, edges, edges[1:], strict=False):
            if not a < b:
                raise ValueError("a bin has zero width; increase the overshoot")
            components.append(
                Uniform(a, b) if kind == "uniform" else _LogUniformBin(a, b)
            )
            masses.append(weight / total * p)
    return Mixture(tuple(components), tuple(masses))
