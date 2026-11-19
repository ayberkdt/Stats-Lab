"""Posterior push-forward: from calibrated parameters back to predictions.

This closes the forward-backward loop. Calibration (backward) turns data into
posterior parameter draws ``theta^(s)``; prediction (forward) evaluates the
model at those draws under new conditions ``c*``:

* latent prediction ``g_q(theta^(s); c*)`` carries parameter uncertainty only;
* an observation prediction adds one independent noise draw
  ``e^(s)`` (Gaussian or Student-t, scale ``sqrt(sigma^2 + sigma_delta^2)``,
  with ``sigma`` taken from the draw when it was calibrated), or, for a
  quantity with a kernel error correlation, one joint Gaussian error vector
  across the prediction conditions.

The two are kept separate so a user can see how much of a predictive interval
is parameter uncertainty and how much is measurement noise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import cholesky, matvec
from probstat.modeling import ProbabilisticModel
from probstat.statistics import quantile
from probstat.uq import evaluate_model

from .nonlinear import CalibrationPosterior, CalibrationProblem, NoiseModel

DEFAULT_PREDICTION_PROBABILITIES = (0.025, 0.5, 0.975)


@dataclass(frozen=True, slots=True)
class PredictedQuantity:
    """Predictive samples of one QoI under one condition.

    Args:
        quantity: QoI name.
        condition: Prediction conditions.
        latent: ``g_q(theta^(s); c*)`` (parameter uncertainty only).
        observed: Latent plus noise draws, or ``None`` if not requested.
        probabilities: Quantile levels.
        latent_quantiles: Quantiles of ``latent``.
        observed_quantiles: Quantiles of ``observed`` or ``None``.

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3], [REF-SMITH-UQ-2024, Chapter 8].

    """

    quantity: str
    condition: Mapping[str, float]
    latent: tuple[float, ...]
    observed: tuple[float, ...] | None
    probabilities: tuple[float, ...]
    latent_quantiles: tuple[float, ...]
    observed_quantiles: tuple[float, ...] | None

    @property
    def mean(self) -> float:
        """Return the posterior mean of the latent prediction.

        Equation:
            [EQ-CAL-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        return math.fsum(self.latent) / len(self.latent)

    @property
    def latent_variance(self) -> float:
        """Return the sample variance due to parameter uncertainty.

        Equation:
            [EQ-CAL-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        return _variance(self.latent)

    @property
    def observed_variance(self) -> float:
        """Return the sample variance of noisy predictions.

        Raises:
            ValueError: If noise draws were not requested.

        Equation:
            [EQ-CAL-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        if self.observed is None:
            raise ValueError("observation noise was not requested")
        return _variance(self.observed)

    def interval(
        self, probability: float = 0.95, *, observed: bool = False
    ) -> tuple[float, float]:
        """Return an equal-tailed sample interval.

        Args:
            probability: Central probability in ``(0, 1)``.
            observed: Use noisy predictions instead of latent ones.

        Returns:
            Lower and upper type-7 sample quantiles.

        Raises:
            ValueError: If the level is invalid or noise was not requested.

        Equation:
            [EQ-CAL-PRED-001], [EQ-STAT-QUANT-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        level = require_finite_real(probability, name="probability")
        if not 0.0 < level < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        values = self.latent
        if observed:
            if self.observed is None:
                raise ValueError("observation noise was not requested")
            values = self.observed
        ordered = sorted(values)
        return (
            quantile(ordered, 0.5 * (1.0 - level)),
            quantile(ordered, 0.5 * (1.0 + level)),
        )


@dataclass(frozen=True, slots=True)
class PosteriorPrediction:
    """Push-forward predictions for several QoIs and conditions.

    Args:
        predictions: One entry per (condition, quantity) pair.
        draws: Number of parameter draws propagated.
        evaluations: Forward-model evaluations used.

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3].

    """

    predictions: tuple[PredictedQuantity, ...]
    draws: int
    evaluations: int

    def get(self, quantity: str, condition_index: int = 0) -> PredictedQuantity:
        """Return the prediction of a QoI under the indexed condition.

        Raises:
            KeyError: If the pair was not predicted.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        matches = [p for p in self.predictions if p.quantity == quantity]
        if not 0 <= condition_index < len(matches):
            raise KeyError(f"no prediction for {quantity!r} at {condition_index}")
        return matches[condition_index]


def _variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance (zero for one value).

    References:
        [EQ-MC-VAR-001]. No new equation is introduced.

    """
    if len(values) < 2:
        return 0.0
    center = math.fsum(values) / len(values)
    return math.fsum((v - center) ** 2 for v in values) / (len(values) - 1)


def _noise_draw(
    generator: random.Random, scale: float, degrees_of_freedom: float | None
) -> float:
    """Draw one Gaussian or Student-t error of a given scale.

    References:
        [EQ-CAL-LIK-001], [REF-DEVROYE-1986, Chapter IX].

    """
    z = generator.gauss(0.0, 1.0)
    if degrees_of_freedom is None:
        return scale * z
    chi_square = 2.0 * generator.gammavariate(0.5 * degrees_of_freedom, 1.0)
    return scale * z / math.sqrt(chi_square / degrees_of_freedom)


def _correlated_conditions(
    requested: Sequence[str],
    overrides: Mapping[str, float],
    noise_models: Mapping[str, NoiseModel],
    conditions: Sequence[Mapping[str, float]],
    fixed: Mapping[str, float],
    samples: Sequence[Mapping[str, float]],
) -> dict[str, tuple[float, ...]]:
    """Return the kernel condition values of each correlated requested QoI.

    Raises:
        ValueError: For a declared-matrix correlation (undefined at new
            conditions), a missing condition value, repeated condition values
            without a discrepancy variance, or draws without ``length[q]``.

    References:
        [EQ-CAL-CORR-001], [EQ-CAL-PRED-001].

    """
    selected: dict[str, tuple[float, ...]] = {}
    for quantity in requested:
        if quantity in overrides or quantity not in noise_models:
            continue
        item = noise_models[quantity]
        structure = item.correlation
        if structure is None:
            continue
        if structure.condition is None:
            raise ValueError(
                f"the error covariance of {quantity!r} is a declared matrix over "
                "the calibration experiments and does not define errors at new "
                "conditions; use include_noise=False or give "
                "noise_standard_deviations for it"
            )
        name = structure.condition
        points: list[float] = []
        for index, condition in enumerate(conditions):
            if name in condition:
                points.append(float(condition[name]))
            elif name in fixed:
                points.append(float(fixed[name]))
            else:
                raise ValueError(
                    f"prediction condition {index} lacks {name!r}, needed for the "
                    f"correlated noise of {quantity!r}"
                )
        if item.discrepancy_variance == 0.0 and len(set(points)) < len(points):
            raise ValueError(
                f"repeated values of {name!r} make the correlated noise of "
                f"{quantity!r} singular across the prediction conditions"
            )
        if not structure.is_known and f"length[{quantity}]" not in samples[0]:
            raise ValueError(f"draws lack length[{quantity}] for noise draws")
        selected[quantity] = tuple(points)
    return selected


def _correlated_draw(
    generator: random.Random,
    item: NoiseModel,
    points: Sequence[float],
    sample: Mapping[str, float],
) -> tuple[float, ...]:
    """Draw one joint Gaussian error vector across the prediction conditions.

    ``e = L z`` with ``L L' = sigma^2 M + sigma_delta^2 I`` and ``z`` standard
    normal, ``M`` the kernel matrix of the prediction conditions.

    References:
        [EQ-CAL-CORR-002], [REF-RASMUSSEN-WILLIAMS-2006, Section A.2].

    """
    structure = item.correlation
    assert structure is not None
    sigma = (
        item.standard_deviation
        if item.standard_deviation is not None
        else sample[f"sigma[{item.quantity}]"]
    )
    length = (
        structure.length
        if structure.length is not None
        else sample[f"length[{item.quantity}]"]
    )
    shape = structure.correlation_matrix(points, length)
    covariance = tuple(
        tuple(
            sigma * sigma * shape[i][j] + (item.discrepancy_variance if i == j else 0.0)
            for j in range(len(points))
        )
        for i in range(len(points))
    )
    lower = cholesky(covariance)
    normals = tuple(generator.gauss(0.0, 1.0) for _ in points)
    return matvec(lower, normals)


def predict(
    source: CalibrationProblem | ProbabilisticModel,
    posterior: CalibrationPosterior | Sequence[Mapping[str, float]],
    quantities: Sequence[str] | None = None,
    conditions: Sequence[Mapping[str, float]] = ({},),
    *,
    include_noise: bool = False,
    noise_standard_deviations: Mapping[str, float] | None = None,
    probabilities: Sequence[float] = DEFAULT_PREDICTION_PROBABILITIES,
    max_draws: int | None = None,
    seed: int | RandomStream | None = None,
) -> PosteriorPrediction:
    """Propagate posterior parameter draws through the forward model.

    Each draw ``theta^(s)`` (and, when calibrated, ``sigma^(s)``) is evaluated
    under every condition, merged as ``fixed``, then condition, then
    parameters. Latent predictions contain parameter uncertainty only. With
    ``include_noise`` one independent error per draw is added: for a
    :class:`CalibrationProblem` the QoI's :class:`NoiseModel` is used (known
    ``sigma``, or the draw's ``sigma[q]``; Student-t when declared; plus the
    discrepancy variance); otherwise ``noise_standard_deviations`` gives a
    Gaussian scale, which also overrides/extends the problem's noise models.

    For a quantity with a kernel :class:`ErrorCorrelation` the errors of one
    draw are drawn jointly across the prediction conditions from the same
    kernel (and the draw's ``sigma[q]`` and ``length[q]`` when calibrated),
    as a new measurement series that is independent of the calibration
    series: they are not conditioned on the calibration residuals. A
    declared-matrix correlation is defined only for the calibration
    experiments, so noise draws for such a quantity raise unless
    ``noise_standard_deviations`` gives an independent scale.

    Args:
        source: Calibration problem or bare probabilistic model.
        posterior: Calibration posterior or named parameter draws.
        quantities: QoIs to predict; defaults to all model QoIs.
        conditions: Prediction conditions (one prediction set per mapping).
        include_noise: Whether to add observation-noise draws.
        noise_standard_deviations: Gaussian noise scales per QoI.
        probabilities: Reported quantile levels.
        max_draws: Optional cap; draws are thinned evenly to at most this many.
        seed: Seed for noise draws.

    Returns:
        Latent and (optionally) noisy predictive samples with quantiles.

    Raises:
        TypeError: If ``source`` or ``posterior`` has the wrong type.
        ValueError: If draws are missing parameters, a QoI is unknown, a
            condition shadows a parameter, a noise scale is unavailable, or
            correlated noise is requested where it is not defined.

    Equation:
        [EQ-CAL-PRED-001], [EQ-UQ-001], [EQ-CAL-CORR-002].

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3], [REF-SMITH-UQ-2024, Chapter 8],
        [REF-KENNEDY-OHAGAN-2001].

    """
    if isinstance(source, CalibrationProblem):
        model = source.model
        noise_models: dict[str, NoiseModel] = {
            item.quantity: item for item in source.noise
        }
    elif isinstance(source, ProbabilisticModel):
        model = source
        noise_models = {}
    else:
        raise TypeError("source must be a CalibrationProblem or ProbabilisticModel")
    if isinstance(posterior, CalibrationPosterior):
        samples = posterior.draws()
    else:
        samples = tuple(MappingProxyType(dict(sample)) for sample in posterior)
    if not samples:
        raise ValueError("at least one posterior draw is required")
    if max_draws is not None:
        cap = require_non_negative_integer(max_draws, name="max_draws")
        if cap == 0:
            raise ValueError("max_draws must be positive")
        if len(samples) > cap:
            stride = len(samples) / cap
            samples = tuple(samples[int(k * stride)] for k in range(cap))
    names = model.inputs.names
    for index, sample in enumerate(samples):
        missing = [name for name in names if name not in sample]
        if missing:
            raise ValueError(f"draw {index} lacks parameters {missing}")
    requested = model.quantity_names if quantities is None else tuple(quantities)
    for quantity in requested:
        model.quantity(quantity)
    condition_list = tuple(MappingProxyType(dict(c)) for c in conditions)
    if not condition_list:
        raise ValueError("at least one condition mapping is required")
    for condition in condition_list:
        shadow = set(condition) & set(names)
        if shadow:
            raise ValueError(f"conditions shadow parameters {sorted(shadow)}")
    levels = tuple(require_finite_real(p, name="probability") for p in probabilities)
    overrides = dict(noise_standard_deviations or {})
    if include_noise:
        for quantity in requested:
            if quantity in overrides:
                value = require_finite_real(overrides[quantity], name=quantity)
                if value < 0.0:
                    raise ValueError("noise standard deviations must be non-negative")
            elif quantity not in noise_models:
                raise ValueError(f"no noise scale available for {quantity!r}")
            elif not noise_models[quantity].is_known and (
                f"sigma[{quantity}]" not in samples[0]
            ):
                raise ValueError(f"draws lack sigma[{quantity}] for noise draws")
    correlated = (
        _correlated_conditions(
            requested, overrides, noise_models, condition_list, model.fixed, samples
        )
        if include_noise
        else {}
    )
    generator = resolve_stream(seed).generator()
    start = model.evaluations
    latent: dict[tuple[int, str], list[float]] = {
        (c, q): [] for c in range(len(condition_list)) for q in requested
    }
    observed: dict[tuple[int, str], list[float]] = {key: [] for key in latent}
    for sample in samples:
        joint = {
            quantity: _correlated_draw(
                generator, noise_models[quantity], points, sample
            )
            for quantity, points in correlated.items()
        }
        for c_index, condition in enumerate(condition_list):
            values: dict[str, float] = dict(model.fixed)
            values.update(condition)
            values.update({name: sample[name] for name in names})
            model.counter.add()
            outputs = evaluate_model(model.model, values, model.quantities)
            for quantity in requested:
                center = outputs[quantity]
                latent[(c_index, quantity)].append(center)
                if not include_noise:
                    continue
                if quantity in overrides:
                    error = _noise_draw(generator, float(overrides[quantity]), None)
                elif quantity in joint:
                    error = joint[quantity][c_index]
                else:
                    item = noise_models[quantity]
                    sigma = (
                        item.standard_deviation
                        if item.standard_deviation is not None
                        else sample[f"sigma[{quantity}]"]
                    )
                    scale = math.sqrt(sigma * sigma + item.discrepancy_variance)
                    error = _noise_draw(generator, scale, item.degrees_of_freedom)
                observed[(c_index, quantity)].append(center + error)
    predictions: list[PredictedQuantity] = []
    for c_index, condition in enumerate(condition_list):
        for quantity in requested:
            key = (c_index, quantity)
            latent_values = tuple(latent[key])
            ordered = sorted(latent_values)
            noisy = tuple(observed[key]) if include_noise else None
            noisy_quantiles = (
                tuple(quantile(sorted(noisy), p) for p in levels)
                if noisy is not None
                else None
            )
            predictions.append(
                PredictedQuantity(
                    quantity,
                    condition,
                    latent_values,
                    noisy,
                    levels,
                    tuple(quantile(ordered, p) for p in levels),
                    noisy_quantiles,
                )
            )
    return PosteriorPrediction(
        tuple(predictions), len(samples), model.evaluations - start
    )
