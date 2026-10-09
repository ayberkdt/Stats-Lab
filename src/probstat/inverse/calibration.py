"""Linear inverse-problem calibration and identifiability diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import require_finite_real
from probstat.inference import (
    GaussianLinearPosterior,
    LinearRegressionResult,
    condition_linear_gaussian,
    fit_simple_linear_regression,
)
from probstat.multivariate import MultivariateNormal


class PredictionTarget(Enum):
    """Distinguish latent physical means from future noisy measurements.

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3], [REF-SMITH-UQ-2024, Chapter 8].

    """

    LATENT_RESPONSE = "latent_response"
    FUTURE_OBSERVATION = "future_observation"


@dataclass(frozen=True, slots=True)
class IdentifiabilityDiagnostic:
    """Rank and conditioning evidence for a two-parameter linear inverse map.

    References:
        [REF-TARANTOLA-2005, Chapter 3].

    """

    rank: int
    dimension: int
    condition_number: float
    identifiable: bool
    warning: str | None


@dataclass(frozen=True, slots=True)
class LinearCalibrationResult:
    """Frequentist fit and Bayesian posterior for one linear calibration task.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 3],
        [REF-MONTGOMERY-PECK-VINING-2021, Chapters 1-2].

    """

    regression: LinearRegressionResult
    posterior: GaussianLinearPosterior
    identifiability: IdentifiabilityDiagnostic
    calibration_inputs: tuple[float, ...]

    def design_rows(
        self, prediction_inputs: Iterable[object]
    ) -> tuple[tuple[float, float], ...]:
        """Build intercept-and-slope rows for a declared prediction domain.

        Args:
            prediction_inputs: Finite locations where predictions are required.

        Returns:
            Rows ``(1, x)`` kept separate from the calibration inputs.

        Equation:
            [EQ-REG-LINE-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        checked = coerce_finite_sample(prediction_inputs, name="prediction_inputs")
        return tuple((1.0, value) for value in checked)


def diagnose_linear_identifiability(
    predictors: Iterable[object], *, condition_warning: object = 1.0e8
) -> IdentifiabilityDiagnostic:
    """Diagnose rank and spectral conditioning of an intercept-slope design.

    Args:
        predictors: Finite calibration inputs.
        condition_warning: Positive threshold above which practical weakness is
            reported even though exact rank is two.

    Returns:
        Exact two-column rank, Gram-matrix condition number, and warning.

    Equation:
        [EQ-INV-IDENT-001].

    References:
        [REF-TARANTOLA-2005, Chapter 3],
        [REF-GOLUB-VANLOAN-2013, Section 8.1].

    """
    values = coerce_finite_sample(predictors, name="predictors")
    threshold = require_finite_real(condition_warning, name="condition_warning")
    if threshold <= 1.0:
        raise ValueError("condition_warning must be greater than one")
    count = float(len(values))
    sum_x = math.fsum(values)
    sum_x2 = math.fsum(value * value for value in values)
    determinant = count * sum_x2 - sum_x * sum_x
    tolerance = math.ulp(max(count * sum_x2, sum_x * sum_x, 1.0)) * 16.0
    if determinant <= tolerance:
        return IdentifiabilityDiagnostic(
            rank=1,
            dimension=2,
            condition_number=math.inf,
            identifiable=False,
            warning="The intercept and slope are not jointly identifiable.",
        )
    trace = count + sum_x2
    discriminant = math.sqrt(max(0.0, trace * trace - 4.0 * determinant))
    largest = 0.5 * (trace + discriminant)
    smallest = determinant / largest
    condition_number = largest / smallest
    warning = (
        "The design is full rank but weakly conditioned."
        if condition_number >= threshold
        else None
    )
    return IdentifiabilityDiagnostic(
        rank=2,
        dimension=2,
        condition_number=condition_number,
        identifiable=True,
        warning=warning,
    )


def calibrate_linear_gaussian(
    predictors: Iterable[object],
    observations: Iterable[object],
    *,
    observation_variance: object,
    prior: MultivariateNormal,
) -> LinearCalibrationResult:
    """Calibrate intercept and slope with OLS and a conjugate Gaussian posterior.

    Args:
        predictors: Finite calibration inputs with non-zero variation.
        observations: Paired finite measured responses.
        observation_variance: Known positive measurement-noise variance.
        prior: Two-dimensional Gaussian prior ordered as intercept then slope.

    Returns:
        A frequentist fit, posterior law, identifiability evidence, and an
        immutable record of the calibration domain.

    Equation:
        [EQ-REG-OLS-001], [EQ-BAYES-LIN-POST-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.3],
        [REF-MONTGOMERY-PECK-VINING-2021, Chapters 1-2].

    """
    x_values = coerce_finite_sample(predictors, name="predictors")
    y_values = coerce_finite_sample(observations, name="observations")
    if not isinstance(prior, MultivariateNormal):
        raise TypeError("prior must be a MultivariateNormal")
    if prior.dimension != 2:
        raise ValueError("linear calibration prior must have dimension two")
    identifiability = diagnose_linear_identifiability(x_values)
    if not identifiability.identifiable:
        raise ValueError(identifiability.warning)
    regression = fit_simple_linear_regression(x_values, y_values)
    posterior = condition_linear_gaussian(
        prior,
        tuple((1.0, value) for value in x_values),
        y_values,
        observation_variance=observation_variance,
    )
    return LinearCalibrationResult(
        regression,
        posterior,
        identifiability,
        x_values,
    )
