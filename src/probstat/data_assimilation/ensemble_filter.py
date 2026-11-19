"""The ensemble Kalman filter loop and its analysis variants.

One function, :func:`ensemble_kalman_filter`, runs the forecast-analysis
cycle for every analysis of this package, so the random streams, the time
indexing, the missing-observation rule, the inflation, and the recorded
moments are the same for all of them (ADR-0018, ADR-0112):

* ``"stochastic"`` (default): perturbed observations [EQ-DA-ENKF-001],
  optionally with Schur-product localization [EQ-DA-LOC-002];
* ``"etkf"``: symmetric-square-root ensemble transform [EQ-DA-ETKF-001];
* ``"eakf"``: serial ensemble adjustment [EQ-DA-EAKF-001];
* ``"letkf"``: local ensemble transform with R-localization
  [EQ-DA-LETKF-001].

Each cycle forecasts the ensemble, applies the multiplicative inflation to
the forecast anomalies, analyses (skipped for a ``None`` observation), and
then applies the optional relaxation [EQ-DA-RELAX-001], [EQ-DA-RELAX-002].

References:
    [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
    [REF-EVENSEN-2003], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
    [REF-ANDERSON-2001].

"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_matrix

from .enkf import (
    AdditiveGaussianModel,
    Ensemble,
    _stochastic_update,
    ensemble_forecast,
    inflate,
    initial_ensemble,
)
from .localization import Localization, _letkf, _local_sets
from .relaxation import Relaxation
from .square_root import eakf_analysis, etkf_analysis

ANALYSIS_METHODS = ("stochastic", "etkf", "eakf", "letkf")
"""Analysis variants of :func:`ensemble_kalman_filter`."""


@dataclass(frozen=True, slots=True)
class EnsembleKalmanResult:
    """Forward-pass record of an ensemble Kalman filter.

    Args:
        analysis_means: Analysis means for ``k = 0..T`` (index 0 is the
            initial ensemble).
        analysis_covariances: Analysis ensemble covariances for ``k = 0..T``.
        forecast_means: Forecast means for ``k = 1..T`` (after inflation).
        forecast_covariances: Forecast covariances for ``k = 1..T``.
        ensemble: Final analysis ensemble.
        inflation: Multiplicative covariance inflation factor used.
        analysis: Analysis variant, one of :data:`ANALYSIS_METHODS`.
        localization: Localization used, or ``None``.
        relaxation: Relaxation used, or ``None``.

    References:
        [REF-EVENSEN-1994], [REF-EVENSEN-2003],
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """

    analysis_means: tuple[Vector, ...]
    analysis_covariances: tuple[Matrix, ...]
    forecast_means: tuple[Vector, ...]
    forecast_covariances: tuple[Matrix, ...]
    ensemble: Ensemble
    inflation: float
    analysis: str = "stochastic"
    localization: Localization | None = None
    relaxation: Relaxation | None = None

    @property
    def ensemble_size(self) -> int:
        """Return the number of members.

        References:
            [REF-EVENSEN-2003].

        """
        return self.ensemble.size


@dataclass(frozen=True, slots=True)
class _Cycle:
    """One forecast-analysis cycle as seen by the filter and the smoother.

    Args:
        forecast: Forecast ensemble after inflation.
        analysis: Analysis ensemble after relaxation (the forecast when the
            observation is missing).
        transforms: Ensemble transforms ``E^a_j = E^f_j T_j`` of the cycle,
            or ``None`` when no analysis took place or none was requested.

    References:
        [EQ-DA-ENKS-001].

    """

    forecast: Ensemble
    analysis: Ensemble
    transforms: tuple[Matrix, ...] | None


@dataclass(frozen=True, slots=True)
class _Settings:
    """Validated options of one filter run.

    References:
        [REF-EVENSEN-2003]. No equation is evaluated.

    """

    data: tuple[Sequence[float] | None, ...]
    plan: tuple[Sequence[float] | None, ...]
    inflation: float
    analysis: str
    localization: Localization | None
    relaxation: Relaxation | None


def _settings(
    model: AdditiveGaussianModel,
    observations: Sequence[Sequence[float] | None],
    *,
    inflation: float,
    analysis: str,
    localization: Localization | None,
    relaxation: Relaxation | None,
    controls: Sequence[Sequence[float] | None] | None,
) -> _Settings:
    """Validate the options shared by the filter and the smoother.

    Raises:
        ValueError: If an option is invalid or options do not fit together.

    References:
        [REF-EVENSEN-2003], [REF-HUNT-KOSTELICH-SZUNYOGH-2007]. No equation
        is evaluated.

    """
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    if controls is None:
        plan: tuple[Sequence[float] | None, ...] = tuple(None for _ in data)
    else:
        plan = tuple(controls)
        if len(plan) != len(data):
            raise ValueError("controls must contain one entry per time step")
    factor = require_finite_real(inflation, name="inflation")
    if factor < 1.0:
        raise ValueError("inflation must be at least one")
    if analysis not in ANALYSIS_METHODS:
        raise ValueError(
            f"analysis must be one of {ANALYSIS_METHODS}, got {analysis!r}"
        )
    if localization is not None:
        if not isinstance(localization, Localization):
            raise TypeError("localization must be a Localization")
        if analysis in ("etkf", "eakf"):
            raise ValueError(
                f"analysis={analysis!r} is global; use 'letkf' or 'stochastic' "
                "for a localized analysis"
            )
        localization.check(model.state_dimension, model.observation_dimension)
    elif analysis == "letkf":
        raise ValueError("analysis='letkf' needs a localization")
    if relaxation is not None and not isinstance(relaxation, Relaxation):
        raise TypeError("relaxation must be a Relaxation")
    return _Settings(data, plan, factor, analysis, localization, relaxation)


def _cycles(
    model: AdditiveGaussianModel,
    settings: _Settings,
    ensemble: Ensemble,
    root: RandomStream,
    *,
    transforms: bool,
) -> Iterator[_Cycle]:
    """Yield the forecast-analysis cycles of one run.

    Stream children ``("forecast", k)`` and ``("analysis", k)`` (the latter
    used only by the stochastic analysis) keep the random sources separate.

    References:
        [EQ-DA-ENKF-001], [EQ-DA-ETKF-001], [EQ-DA-EAKF-001],
        [EQ-DA-LETKF-001], [EQ-DA-RELAX-001], [EQ-DA-RELAX-002],
        [REF-EVENSEN-2003].

    """
    noise = as_matrix(model.observation_covariance, name="observation_covariance")
    localization = settings.localization
    state_weights: Matrix | None = None
    observation_weights: Matrix | None = None
    local_sets = None
    if localization is not None:
        state_weights = localization.state_observation_weights()
        if settings.analysis == "letkf":
            local_sets = _local_sets(state_weights)
        else:
            observation_weights = localization.observation_weights()
    for step, (observation, control) in enumerate(
        zip(settings.data, settings.plan, strict=True)
    ):
        forecast = inflate(
            ensemble_forecast(
                model, ensemble, seed=root.child("forecast", step), control=control
            ),
            settings.inflation,
        )
        if observation is None:
            ensemble = forecast
            yield _Cycle(forecast, forecast, None)
            continue
        weights: tuple[Matrix, ...] | None
        if settings.analysis == "stochastic":
            outcome, weights = _stochastic_update(
                forecast,
                observation,
                model.observe,
                noise,
                root.child("analysis", step),
                state_weights=state_weights,
                observation_weights=observation_weights,
                transforms=transforms,
            )
            updated = outcome.ensemble
        else:
            if settings.analysis == "etkf":
                result = etkf_analysis(forecast, observation, model.observe, noise)
            elif settings.analysis == "eakf":
                result = eakf_analysis(forecast, observation, model.observe, noise)
            else:
                assert local_sets is not None
                result = _letkf(forecast, observation, model.observe, noise, local_sets)
            updated = result.ensemble
            weights = result.transforms if transforms else None
        if settings.relaxation is not None:
            relaxed = settings.relaxation.apply(forecast, updated)
            if weights is not None:
                weights = settings.relaxation.compose(forecast, updated, weights)
            updated = relaxed
        ensemble = updated
        yield _Cycle(forecast, updated, weights)


def ensemble_kalman_filter(
    model: AdditiveGaussianModel,
    observations: Sequence[Sequence[float] | None],
    *,
    ensemble_size: int,
    inflation: float = 1.0,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
    analysis: str = "stochastic",
    localization: Localization | None = None,
    relaxation: Relaxation | None = None,
) -> EnsembleKalmanResult:
    """Run an ensemble Kalman filter over an observation record.

    Each step forecasts the ensemble, applies the declared multiplicative
    inflation to the forecast anomalies, analyses the observation (skipped
    for a ``None`` observation), and applies the optional relaxation.
    ``analysis`` selects the update: ``"stochastic"`` perturbed observations
    (the default; with ``localization`` the gain is Schur-localized),
    ``"etkf"`` the symmetric-square-root ensemble transform, ``"eakf"`` the
    serial ensemble adjustment (diagonal ``R``), and ``"letkf"`` the local
    ensemble transform (``localization`` required). Stream children
    ``"initial"``, ``("forecast", k)``, and ``("analysis", k)`` keep every
    random source reproducible and separate; the default options reproduce
    the stochastic filter of earlier releases exactly.

    Args:
        model: Additive-Gaussian state-space model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        ensemble_size: Number of members ``N >= 2``.
        inflation: Covariance inflation factor ``>= 1``.
        seed: Integer seed or stream.
        controls: Optional controls (linear models only).
        analysis: One of :data:`ANALYSIS_METHODS`.
        localization: Coordinates and Gaspari-Cohn half-width, for the
            ``"stochastic"`` and ``"letkf"`` analyses.
        relaxation: Optional RTPP or RTPS relaxation after each analysis.

    Returns:
        Forecast and analysis moments for every step, the final ensemble, and
        the settings used.

    Raises:
        ValueError: If arguments are invalid or do not fit together.
        TypeError: If ``localization`` or ``relaxation`` has the wrong type.

    Equation:
        [EQ-DA-ENKF-001], [EQ-DA-ENKF-002], [EQ-FILT-BAYES-001],
        [EQ-DA-ETKF-001], [EQ-DA-EAKF-001], [EQ-DA-LOC-002],
        [EQ-DA-LETKF-001], [EQ-DA-RELAX-001], [EQ-DA-RELAX-002].

    References:
        [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
        [REF-EVENSEN-2003], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
        [REF-ANDERSON-2001], [REF-HOUTEKAMER-MITCHELL-2001],
        [REF-WHITAKER-HAMILL-2012].

    """
    settings = _settings(
        model,
        observations,
        inflation=inflation,
        analysis=analysis,
        localization=localization,
        relaxation=relaxation,
        controls=controls,
    )
    root = resolve_stream(seed)
    ensemble = initial_ensemble(model, ensemble_size, seed=root.child("initial"))
    analysis_means = [ensemble.mean]
    analysis_covariances = [ensemble.covariance]
    forecast_means: list[Vector] = []
    forecast_covariances: list[Matrix] = []
    for cycle in _cycles(model, settings, ensemble, root, transforms=False):
        forecast_means.append(cycle.forecast.mean)
        forecast_covariances.append(cycle.forecast.covariance)
        ensemble = cycle.analysis
        analysis_means.append(ensemble.mean)
        analysis_covariances.append(ensemble.covariance)
    return EnsembleKalmanResult(
        tuple(analysis_means),
        tuple(analysis_covariances),
        tuple(forecast_means),
        tuple(forecast_covariances),
        ensemble,
        settings.inflation,
        settings.analysis,
        settings.localization,
        settings.relaxation,
    )
