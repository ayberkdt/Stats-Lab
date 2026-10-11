"""Predictions that report parametric, numerical, and model-form uncertainty.

A prediction at an intended-use condition carries three different kinds of
uncertainty that answer different questions (Roy and Oberkampf 2011):

* **parametric** — spread of the QoI induced by uncertain inputs or
  calibrated parameters (propagated or posterior-predictive samples);
* **numerical** — discretization error of the solver at the settings used,
  from a :class:`~probstat.uq.SolutionVerification` statement
  ``phi_1 +/- U_num``;
* **model form** — the error of the model equations themselves, here an
  additive zero-mean Gaussian discrepancy whose variance is estimated at
  validation points and extrapolated by an explicit rule.

:func:`predict_with_uncertainty_components` keeps the three components as
separate fields and reports a combined interval together with the exact rule
used to form it. The module lives in ``validation`` (layer 5) because the
model-form component needs the validation domain and validation comparisons;
``propagation`` (layer 4) may not import them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_finite_real,
)
from probstat.statistics import quantile
from probstat.uq import SolutionVerification

from .assessment import ValidationAssessment
from .credibility import PredictionStatement
from .discrepancy import AdditiveGaussianDiscrepancy
from .metrics import AsmeComparison
from .records import DomainCheck, ValidationDomain

NumericalTreatment = Literal["interval", "uniform", "normal"]
"""How the numerical uncertainty ``U_num`` enters the combined prediction."""

_NORMAL_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class DiscrepancyEstimate:
    """Method-of-moments estimate of an additive discrepancy variance.

    Args:
        discrepancy: Estimated zero-mean Gaussian discrepancy.
        mean_error: Mean comparison error ``mean(E_i)`` (a non-zero value
            signals systematic bias that the zero-mean model absorbs into its
            variance).
        mean_squared_error: ``mean(E_i^2)``.
        mean_validation_variance: ``mean(u_val,i^2)``.
        count: Number of validation comparisons.
        truncated: ``True`` when ``mean(E^2) < mean(u_val^2)`` so the
            variance estimate was truncated at zero.

    References:
        [REF-ASME-VV20-2009], [REF-KENNEDY-OHAGAN-2001].

    """

    discrepancy: AdditiveGaussianDiscrepancy
    mean_error: float
    mean_squared_error: float
    mean_validation_variance: float
    count: int
    truncated: bool


def estimate_discrepancy(
    evidence: ValidationAssessment | Sequence[AsmeComparison],
    *,
    description: str = "additive model-form discrepancy estimated at "
    "validation points",
    applicability_domain: str | None = None,
) -> DiscrepancyEstimate:
    """Estimate ``sigma_delta^2`` from validation comparison errors.

    Under the ASME V&V 20 error model each comparison error is
    ``E_i = delta_model,i + (errors with variance u_val,i^2)``. Treating
    ``delta_model`` as zero-mean with common variance gives the moment
    estimate ``sigma_delta^2 = max(0, mean(E_i^2) - mean(u_val,i^2))``
    ([EQ-VAL-DISC-002]). A systematic bias is absorbed into the variance;
    :attr:`DiscrepancyEstimate.mean_error` reports it so it is not hidden.

    Args:
        evidence: A validation assessment (its per-experiment ASME
            comparisons are used) or comparisons directly.
        description: Description stored on the discrepancy.
        applicability_domain: Scope text; defaults to the assessment's
            validation-domain description and is required for raw
            comparisons.

    Returns:
        Estimate with the discrepancy record and its ingredients.

    Raises:
        TypeError: If an item is not an :class:`AsmeComparison`.
        ValueError: If no comparison is given or the scope is missing.

    Equation:
        [EQ-VAL-DISC-002], [EQ-VAL-ASME-001].

    References:
        [REF-ASME-VV20-2009], [REF-KENNEDY-OHAGAN-2001],
        [REF-ROY-OBERKAMPF-2011].

    """
    if isinstance(evidence, ValidationAssessment):
        comparisons = tuple(item.asme for item in evidence.experiments)
        scope = (
            f"validation domain {evidence.domain.describe()}"
            if applicability_domain is None
            else applicability_domain
        )
    else:
        comparisons = tuple(evidence)
        if applicability_domain is None:
            raise ValueError(
                "applicability_domain is required when comparisons are given directly"
            )
        scope = applicability_domain
    if not comparisons:
        raise ValueError("at least one validation comparison is required")
    for index, item in enumerate(comparisons):
        if not isinstance(item, AsmeComparison):
            raise TypeError(f"comparisons[{index}] must be an AsmeComparison")
    count = len(comparisons)
    mean_error = math.fsum(item.comparison_error for item in comparisons) / count
    mean_square = math.fsum(item.comparison_error**2 for item in comparisons) / count
    mean_variance = (
        math.fsum(item.validation_uncertainty**2 for item in comparisons) / count
    )
    variance = mean_square - mean_variance
    truncated = variance < 0.0
    discrepancy = AdditiveGaussianDiscrepancy(
        math.sqrt(max(variance, 0.0)), description, scope
    )
    return DiscrepancyEstimate(
        discrepancy, mean_error, mean_square, mean_variance, count, truncated
    )


@dataclass(frozen=True, slots=True)
class DiscrepancyAtCondition:
    """Model-form discrepancy evaluated at one prediction condition.

    Args:
        condition: Prediction condition.
        base_standard_deviation: ``sigma_delta,0`` at validation points.
        growth_rate: ``kappa`` of the extrapolation rule.
        distance: Scaled distance ``d`` to the nearest validation point.
        standard_deviation: ``sigma_delta(x)``.
        domain_check: Location relative to the validation domain.
        rule: Text of the extrapolation rule applied.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-KENNEDY-OHAGAN-2001].

    """

    condition: Mapping[str, float]
    base_standard_deviation: float
    growth_rate: float
    distance: float
    standard_deviation: float
    domain_check: DomainCheck | None
    rule: str

    @property
    def variance(self) -> float:
        """Return ``sigma_delta(x)^2``.

        Equation:
            [EQ-VAL-DISC-001].

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        return self.standard_deviation * self.standard_deviation


@dataclass(frozen=True, slots=True)
class DiscrepancyExtrapolation:
    """Extrapolate a discrepancy variance away from the validation points.

    Rule ([EQ-VAL-DISC-001]): with ``d(x)`` the Euclidean distance from the
    condition to the nearest validation condition after scaling every
    coordinate by its validation range ([EQ-VAL-DOM-001]),

    ``sigma_delta^2(x) = sigma_delta,0^2 (1 + kappa d(x))``.

    The variance equals the validation estimate at validation points and
    grows linearly with distance; ``kappa = 0`` keeps it constant. Inside the
    hull between sparse points ``d > 0`` too, so the rule is mildly
    conservative there. The growth rate is a declared modelling assumption,
    not a quantity estimated from data.

    Args:
        discrepancy: Discrepancy estimated at the validation points.
        domain: Validation domain that defines ``d(x)``.
        growth_rate: Non-negative ``kappa`` per unit scaled distance.
        distance_threshold: Threshold of the domain check (reporting only).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a number is invalid.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

    """

    discrepancy: AdditiveGaussianDiscrepancy
    domain: ValidationDomain
    growth_rate: float
    distance_threshold: float = 0.25

    def __post_init__(self) -> None:
        """Validate the rule's inputs.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        if not isinstance(self.discrepancy, AdditiveGaussianDiscrepancy):
            raise TypeError("discrepancy must be an AdditiveGaussianDiscrepancy")
        if not isinstance(self.domain, ValidationDomain):
            raise TypeError("domain must be a ValidationDomain")
        object.__setattr__(
            self,
            "growth_rate",
            require_non_negative_finite_real(self.growth_rate, name="growth_rate"),
        )
        object.__setattr__(
            self,
            "distance_threshold",
            require_non_negative_finite_real(
                self.distance_threshold, name="distance_threshold"
            ),
        )

    def at(self, condition: Mapping[str, float]) -> DiscrepancyAtCondition:
        """Evaluate the extrapolated discrepancy at one condition.

        Args:
            condition: Mapping with a value for every domain variable.

        Returns:
            Discrepancy at the condition with its domain check.

        Raises:
            KeyError: If a domain variable is missing.
            ValueError: If the condition leaves a validation dimension that
                had zero range (the scaled distance is then infinite).

        Equation:
            [EQ-VAL-DISC-001], [EQ-VAL-DOM-001].

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        check = self.domain.check(condition, distance_threshold=self.distance_threshold)
        distance = check.nearest_distance
        if not math.isfinite(distance):
            raise ValueError(
                "the condition leaves a validation dimension with zero range; "
                "the scaled distance is infinite and no extrapolation is defined"
            )
        base = self.discrepancy.standard_deviation
        sigma = base * math.sqrt(1.0 + self.growth_rate * distance)
        rule = (
            f"sigma_delta^2(x) = sigma_0^2 (1 + kappa d) with sigma_0={base:.4g}, "
            f"kappa={self.growth_rate:g}, d={distance:.4g} (scaled distance to the "
            f"nearest validation point {check.nearest_experiment!r})"
        )
        return DiscrepancyAtCondition(
            check.condition, base, self.growth_rate, distance, sigma, check, rule
        )


@dataclass(frozen=True, slots=True)
class ComponentPrediction:
    """A prediction with parametric, numerical, and model-form parts kept apart.

    Args:
        quantity: QoI name.
        condition: Intended-use condition.
        probability: Central probability of every interval.
        parametric_mean: Mean of the parametric samples.
        parametric_standard_deviation: Sample standard deviation.
        parametric_interval: Equal-tailed interval of the parametric samples.
        numerical: Solution-verification statement, or ``None`` if numerical
            uncertainty was not supplied.
        numerical_treatment: ``"interval"``, ``"uniform"``, or ``"normal"``.
        numerical_half_width: ``U_num`` (zero when not supplied).
        numerical_standard_uncertainty: ``u_num`` for the probabilistic
            treatments, ``None`` for the interval treatment.
        discrepancy: Model-form discrepancy at the condition, or ``None``.
        combined_interval: Combined interval (see ``combination_rule``).
        combination_rule: Exact rule used to form ``combined_interval``.
        variance_budget: Variances of the probabilistically treated
            components (``parametric``, ``numerical``, ``model_form``).
        excluded: Components that are not part of the combined interval.
        parametric_samples: Parametric predictive samples.
        combined_samples: Samples of the probabilistic sum.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

    """

    quantity: str
    condition: Mapping[str, float]
    probability: float
    parametric_mean: float
    parametric_standard_deviation: float
    parametric_interval: tuple[float, float]
    numerical: SolutionVerification | None
    numerical_treatment: NumericalTreatment
    numerical_half_width: float
    numerical_standard_uncertainty: float | None
    discrepancy: DiscrepancyAtCondition | None
    combined_interval: tuple[float, float]
    combination_rule: str
    variance_budget: Mapping[str, float]
    excluded: tuple[str, ...]
    parametric_samples: tuple[float, ...] = field(repr=False)
    combined_samples: tuple[float, ...] = field(repr=False)

    def variance_shares(self) -> Mapping[str, float]:
        """Return each budget variance as a fraction of their sum.

        Returns:
            Read-only mapping summing to one (all zeros if the sum is zero).

        Equation:
            [EQ-VAL-PRED-001].

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        total = math.fsum(self.variance_budget.values())
        if total == 0.0:
            return MappingProxyType(dict.fromkeys(self.variance_budget, 0.0))
        return MappingProxyType(
            {name: value / total for name, value in self.variance_budget.items()}
        )

    def uncertainty_sources(self) -> str:
        """Return a sentence naming what the combined interval includes.

        Returns:
            Text suitable for :class:`PredictionStatement`.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        included = ["parametric (propagated samples)"]
        if self.numerical is not None:
            included.append(
                f"numerical U_num={self.numerical_half_width:.4g} "
                f"({self.numerical_treatment})"
            )
        if self.discrepancy is not None:
            included.append(
                f"model form sigma_delta={self.discrepancy.standard_deviation:.4g}"
            )
        text = "includes " + "; ".join(included)
        if self.excluded:
            text += "; excludes " + "; ".join(self.excluded)
        return text

    def to_prediction_statement(self) -> PredictionStatement:
        """Return the combined interval as a credibility prediction statement.

        Returns:
            Statement whose sources list the included and excluded components
            and whose domain check comes from the discrepancy (if any).

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        return PredictionStatement(
            self.quantity,
            self.condition,
            self.combined_interval[0],
            self.combined_interval[1],
            self.probability,
            self.uncertainty_sources(),
            None if self.discrepancy is None else self.discrepancy.domain_check,
        )

    def summary(self) -> str:
        """Return a multi-line report with one line per component.

        Returns:
            Text report; the combined interval is printed with its rule.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        values = ", ".join(f"{k}={v:.6g}" for k, v in self.condition.items())
        level = f"{100 * self.probability:g}%"
        lines = [f"Prediction of {self.quantity} at ({values or 'nominal'}):"]
        low, high = self.parametric_interval
        lines.append(
            f"  - parametric: mean {self.parametric_mean:.6g}, sd "
            f"{self.parametric_standard_deviation:.4g}, {level} [{low:.6g}, {high:.6g}]"
        )
        if self.numerical is None:
            lines.append("  - numerical: not supplied (excluded)")
        else:
            u_text = (
                "no distribution assigned"
                if self.numerical_standard_uncertainty is None
                else f"u_num={self.numerical_standard_uncertainty:.4g}"
            )
            lines.append(
                f"  - numerical: +/- {self.numerical_half_width:.4g} "
                f"({self.numerical_treatment}; {u_text}; {self.numerical.method})"
            )
        if self.discrepancy is None:
            lines.append("  - model form: not supplied (excluded)")
        else:
            where = (
                ""
                if self.discrepancy.domain_check is None
                else f"; {self.discrepancy.domain_check.note}"
            )
            lines.append(
                f"  - model form: sigma_delta={self.discrepancy.standard_deviation:.4g}"
                f" ({self.discrepancy.rule}{where})"
            )
        shares = ", ".join(
            f"{name} {100 * share:.1f}%"
            for name, share in self.variance_shares().items()
        )
        lines.append(f"  - variance shares of probabilistic components: {shares}")
        c_low, c_high = self.combined_interval
        lines.append(
            f"  - combined {level}: [{c_low:.6g}, {c_high:.6g}]; rule: "
            f"{self.combination_rule}"
        )
        if self.excluded:
            lines.append("  - not included: " + "; ".join(self.excluded))
        return "\n".join(lines)


def predict_with_uncertainty_components(
    quantity: str,
    condition: Mapping[str, float],
    parametric_samples: Sequence[float],
    *,
    numerical: SolutionVerification | None = None,
    numerical_treatment: NumericalTreatment = "interval",
    discrepancy: DiscrepancyExtrapolation | AdditiveGaussianDiscrepancy | None = None,
    probability: float = 0.95,
    seed: int | RandomStream | None = None,
) -> ComponentPrediction:
    """Combine parametric, numerical, and model-form uncertainty explicitly.

    Let ``y_s`` be the parametric samples. With seeded independent draws
    ``d_s ~ N(0, sigma_delta(x)^2)`` and, for the probabilistic numerical
    treatments, ``e_s ~ Uniform(-U, U)`` (``"uniform"``) or
    ``e_s ~ N(0, (U / 1.96)^2)`` (``"normal"``), the probabilistic sum is
    ``z_s = y_s + e_s + d_s`` ([EQ-VAL-PRED-001]) and the combined interval
    is its equal-tailed sample interval. With ``"interval"`` the numerical
    error is an epistemic interval without a distribution: ``e_s = 0`` and
    the interval of ``y + d`` is widened by ``U`` on each side
    ([EQ-VAL-PRED-002]). Independence of the three components is assumed and
    stated in the rule; the separate components are always returned.

    Args:
        quantity: QoI name.
        condition: Intended-use condition.
        parametric_samples: Predictive samples carrying parametric
            uncertainty (from ``propagation.propagate``,
            ``validation.predict_at_conditions``, or a posterior push-forward).
        numerical: Solution-verification statement supplying ``U_num``.
        numerical_treatment: ``"interval"`` (default, no distribution),
            ``"uniform"``, or ``"normal"``.
        discrepancy: Extrapolation rule (preferred) or a constant
            discrepancy; ``None`` excludes model-form uncertainty.
        probability: Central probability in ``(0, 1)``.
        seed: Seed or stream of the numerical and discrepancy draws.

    Returns:
        Component prediction.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If fewer than two samples are given, the probability or
            treatment is invalid, or the discrepancy cannot be evaluated.

    Equation:
        [EQ-VAL-PRED-001], [EQ-VAL-PRED-002], [EQ-VAL-DISC-001],
        [EQ-VV-UNUM-001], [EQ-STAT-QUANT-001].

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV],
        [REF-JCGM-100-2008, Sections 4.3.7 and 6.2], [REF-HYNDMAN-FAN-1996].

    """
    if not isinstance(quantity, str) or not quantity.strip():
        raise ValueError("quantity must be a non-empty string")
    if not isinstance(condition, Mapping):
        raise TypeError("condition must be a mapping")
    frozen_condition = MappingProxyType(
        {
            str(key): require_finite_real(value, name=f"condition[{key!r}]")
            for key, value in condition.items()
        }
    )
    samples = coerce_finite_sample(parametric_samples, name="parametric_samples")
    if len(samples) < 2:
        raise ValueError("at least two parametric samples are required")
    level = require_finite_real(probability, name="probability")
    if not 0.0 < level < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if numerical_treatment not in ("interval", "uniform", "normal"):
        raise ValueError("numerical_treatment must be interval, uniform, or normal")
    if numerical is not None and not isinstance(numerical, SolutionVerification):
        raise TypeError("numerical must be a SolutionVerification")
    at_condition: DiscrepancyAtCondition | None
    if discrepancy is None:
        at_condition = None
    elif isinstance(discrepancy, DiscrepancyExtrapolation):
        at_condition = discrepancy.at(frozen_condition)
    elif isinstance(discrepancy, AdditiveGaussianDiscrepancy):
        at_condition = DiscrepancyAtCondition(
            frozen_condition,
            discrepancy.standard_deviation,
            0.0,
            0.0,
            discrepancy.standard_deviation,
            None,
            "constant discrepancy (no extrapolation rule and no domain check; "
            f"declared scope: {discrepancy.applicability_domain})",
        )
    else:
        raise TypeError(
            "discrepancy must be a DiscrepancyExtrapolation or "
            "AdditiveGaussianDiscrepancy"
        )

    count = len(samples)
    mean = math.fsum(samples) / count
    variance = math.fsum((value - mean) ** 2 for value in samples) / (count - 1)
    tail = 0.5 * (1.0 - level)
    ordered = sorted(samples)
    parametric_interval = (quantile(ordered, tail), quantile(ordered, 1.0 - tail))

    root = resolve_stream(seed)
    half_width = 0.0 if numerical is None else numerical.expanded_uncertainty
    budget: dict[str, float] = {"parametric": variance}
    excluded: list[str] = []
    u_num: float | None = None
    numerical_draws = [0.0] * count
    if numerical is None:
        excluded.append("numerical uncertainty (no solution verification supplied)")
    elif numerical_treatment == "interval":
        excluded.append(
            "numerical uncertainty from the variance budget (treated as an "
            "interval, not a distribution)"
        )
    else:
        generator = root.child("prediction", "numerical").generator()
        if numerical_treatment == "uniform":
            u_num = half_width / math.sqrt(3.0)
            numerical_draws = [
                generator.uniform(-half_width, half_width) for _ in range(count)
            ]
        else:
            u_num = half_width / _NORMAL_95
            numerical_draws = [generator.gauss(0.0, u_num) for _ in range(count)]
        budget["numerical"] = u_num * u_num
    discrepancy_draws = [0.0] * count
    if at_condition is None:
        excluded.append("model-form uncertainty (no discrepancy supplied)")
    else:
        budget["model_form"] = at_condition.variance
        if at_condition.standard_deviation > 0.0:
            generator = root.child("prediction", "discrepancy").generator()
            discrepancy_draws = [
                generator.gauss(0.0, at_condition.standard_deviation)
                for _ in range(count)
            ]
    combined = tuple(
        y + e + d
        for y, e, d in zip(samples, numerical_draws, discrepancy_draws, strict=True)
    )
    combined_sorted = sorted(combined)
    low = quantile(combined_sorted, tail)
    high = quantile(combined_sorted, 1.0 - tail)
    parts = ["parametric samples"]
    if at_condition is not None:
        parts.append("independent N(0, sigma_delta(x)^2) discrepancy draws")
    if numerical is not None and numerical_treatment != "interval":
        parts.append(f"independent {numerical_treatment} numerical-error draws")
    rule = "equal-tailed sample interval of " + " + ".join(parts)
    if numerical is not None and numerical_treatment == "interval":
        low -= half_width
        high += half_width
        rule += f", widened by U_num={half_width:.4g} on each side (interval treatment)"
    rule += f"; root seed {root.seed}"
    return ComponentPrediction(
        quantity,
        frozen_condition,
        level,
        mean,
        math.sqrt(variance),
        parametric_interval,
        numerical,
        numerical_treatment,
        half_width,
        u_num,
        at_condition,
        (low, high),
        rule,
        MappingProxyType(budget),
        tuple(excluded),
        samples,
        combined,
    )
