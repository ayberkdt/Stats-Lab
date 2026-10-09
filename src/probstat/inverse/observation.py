"""Observation and noise contracts for inverse problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import require_finite_real
from probstat.inference import gaussian_log_likelihood

ParameterVector = tuple[float, ...]
ForwardResponse = Callable[[float, ParameterVector], float]


@dataclass(frozen=True, slots=True)
class GaussianObservationModel:
    """Measured responses linked to a forward model by additive Gaussian error.

    ``observation_variance`` represents measurement noise. The separate
    ``discrepancy_variance`` represents unresolved model inadequacy; combining
    the values numerically does not make the two uncertainty sources
    scientifically interchangeable.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-SMITH-UQ-2024, Chapter 8].

    """

    inputs: tuple[float, ...]
    observations: tuple[float, ...]
    observation_variance: float
    discrepancy_variance: float = 0.0

    def __post_init__(self) -> None:
        """Validate paired data and separated non-negative variance components.

        References:
            [EQ-INV-OBS-001], [REF-KENNEDY-OHAGAN-2001].

        """
        inputs = coerce_finite_sample(self.inputs, name="inputs")
        observations = coerce_finite_sample(
            self.observations, name="observations"
        )
        if len(inputs) != len(observations):
            raise ValueError("inputs and observations must have equal length")
        observation_variance = require_finite_real(
            self.observation_variance, name="observation_variance"
        )
        discrepancy_variance = require_finite_real(
            self.discrepancy_variance, name="discrepancy_variance"
        )
        if observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if discrepancy_variance < 0.0:
            raise ValueError("discrepancy_variance must be non-negative")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_variance", observation_variance)
        object.__setattr__(self, "discrepancy_variance", discrepancy_variance)

    @property
    def total_error_variance(self) -> float:
        """Return the modeled sum of independent noise and discrepancy variance.

        Equation:
            [EQ-INV-OBS-001], [EQ-CAL-DISC-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        return self.observation_variance + self.discrepancy_variance

    def predictions(
        self, forward_model: ForwardResponse, parameters: Iterable[object]
    ) -> tuple[float, ...]:
        """Evaluate a forward response at every observation input.

        Args:
            forward_model: Callable ``f(input, parameters)``.
            parameters: Non-empty finite parameter vector.

        Returns:
            Finite predictions paired with observations.

        References:
            [REF-SMITH-UQ-2024, Chapter 8]. No new equation is introduced.

        """
        if not callable(forward_model):
            raise TypeError("forward_model must be callable")
        checked_parameters = coerce_finite_sample(parameters, name="parameters")
        predicted = tuple(
            require_finite_real(
                forward_model(value, checked_parameters),
                name=f"prediction[{index}]",
            )
            for index, value in enumerate(self.inputs)
        )
        return predicted

    def residuals(
        self, forward_model: ForwardResponse, parameters: Iterable[object]
    ) -> tuple[float, ...]:
        """Return observed-minus-predicted calibration residuals.

        Equation:
            [EQ-INV-RESID-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        predicted = self.predictions(forward_model, parameters)
        return tuple(
            observed - center
            for observed, center in zip(self.observations, predicted, strict=True)
        )

    def log_likelihood(
        self, forward_model: ForwardResponse, parameters: Iterable[object]
    ) -> float:
        """Evaluate the independent Gaussian likelihood of the observed data.

        Equation:
            [EQ-BAYES-LIKE-001], [EQ-INV-OBS-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 3.2].

        """
        predicted = self.predictions(forward_model, parameters)
        return gaussian_log_likelihood(
            self.observations,
            predicted,
            variance=self.total_error_variance,
        )

    def root_mean_squared_residual(
        self, forward_model: ForwardResponse, parameters: Iterable[object]
    ) -> float:
        """Return the root mean squared calibration residual.

        Equation:
            [EQ-INV-RMSE-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        residuals = self.residuals(forward_model, parameters)
        mean_square = math.fsum(value * value for value in residuals) / len(
            residuals
        )
        return math.sqrt(mean_square)
