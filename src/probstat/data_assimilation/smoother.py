"""Fixed-lag ensemble Kalman smoother (EnKS).

Every analysis of this package is a linear combination of the forecast
members: component ``j`` of the analysis members is ``E^a_j = E^f_j T_j``
with an ``N`` by ``N`` transform ``T_j`` (one shared ``T`` for a global
analysis). The ensemble Kalman smoother of Evensen & van Leeuwen (2000)
applies the same combination to the ensembles of earlier times, which are
still paired member by member with the forecast:
``E_(j|k) = E_(j|k-1) T_k`` [EQ-DA-ENKS-001]. For a linear-Gaussian model
and a large ensemble the result approaches the Rauch-Tung-Striebel smoother;
no backward integration of the model is needed.

The smoother reuses the filter cycle of
:func:`~probstat.data_assimilation.ensemble_kalman_filter` with the same
streams, so its filter part is identical to a filter run with the same
seed. With a localized analysis (Schur-localized stochastic or LETKF) and
with RTPS relaxation the transform differs between state components; each
lagged component is then updated with the transform of its own component,
an implementation choice that reduces to [EQ-DA-ENKS-001] without
localization.

References:
    [REF-EVENSEN-VANLEEUWEN-2000], [REF-EVENSEN-2003, Appendix D].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Matrix, Vector

from .enkf import AdditiveGaussianModel, Ensemble, initial_ensemble
from .ensemble_filter import EnsembleKalmanResult, _cycles, _settings
from .localization import Localization
from .relaxation import Relaxation


@dataclass(frozen=True, slots=True)
class EnsembleSmootherResult:
    """Fixed-lag ensemble smoother record.

    Args:
        smoothed_means: Means of ``x_k`` given ``y_1..y_min(k+lag, T)`` for
            ``k = 0..T``.
        smoothed_covariances: The corresponding ensemble covariances.
        smoothed_ensembles: The smoothed ensembles themselves.
        lag: Number of later observations used for each state.
        filter: The filter run that produced the transforms.

    References:
        [EQ-DA-ENKS-001], [REF-EVENSEN-VANLEEUWEN-2000].

    """

    smoothed_means: tuple[Vector, ...]
    smoothed_covariances: tuple[Matrix, ...]
    smoothed_ensembles: tuple[Ensemble, ...]
    lag: int
    filter: EnsembleKalmanResult


def _transform(ensemble: Ensemble, transforms: tuple[Matrix, ...]) -> Ensemble:
    """Return ``E T`` componentwise for one shared or per-component transforms.

    References:
        [EQ-DA-ENKS-001], [REF-EVENSEN-2003, Appendix D].

    """
    members = ensemble.members
    count = len(members)
    dimension = len(members[0])
    shared = len(transforms) == 1
    columns = []
    for j in range(dimension):
        weights = transforms[0] if shared else transforms[j]
        values = [member[j] for member in members]
        columns.append(
            [
                math.fsum(values[a] * weights[a][i] for a in range(count))
                for i in range(count)
            ]
        )
    return Ensemble._trusted(
        tuple(tuple(column[i] for column in columns) for i in range(count))
    )


def ensemble_kalman_smoother(
    model: AdditiveGaussianModel,
    observations: Sequence[Sequence[float] | None],
    *,
    ensemble_size: int,
    lag: int,
    inflation: float = 1.0,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
    analysis: str = "stochastic",
    localization: Localization | None = None,
    relaxation: Relaxation | None = None,
) -> EnsembleSmootherResult:
    """Run the fixed-lag ensemble Kalman smoother over an observation record.

    The filter cycle of
    :func:`~probstat.data_assimilation.ensemble_kalman_filter` runs with the
    same options and streams; after the analysis at time ``k`` the stored
    ensembles of times ``k - lag .. k - 1`` are multiplied by that
    analysis' transforms [EQ-DA-ENKS-001]. A state leaves the window, and is
    final, once ``lag`` later observation times have been processed. A
    missing observation leaves the window unchanged. ``lag = 0`` returns the
    filter analyses; ``lag >= T`` gives the fixed-interval smoother.

    Args:
        model: Additive-Gaussian state-space model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        ensemble_size: Number of members ``N >= 2``.
        lag: Non-negative smoothing lag.
        inflation: Covariance inflation factor ``>= 1``.
        seed: Integer seed or stream.
        controls: Optional controls (linear models only).
        analysis: Analysis variant of the filter.
        localization: Localization of the filter analysis.
        relaxation: Relaxation of the filter analysis.

    Returns:
        Smoothed moments and ensembles for ``k = 0..T`` and the filter record.

    Raises:
        ValueError: If arguments are invalid.
        TypeError: If ``localization`` or ``relaxation`` has the wrong type.

    Equation:
        [EQ-DA-ENKS-001].

    References:
        [REF-EVENSEN-VANLEEUWEN-2000], [REF-EVENSEN-2003, Appendix D].

    """
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError("lag must be a non-negative integer")
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
    window: list[tuple[int, Ensemble]] = [(0, ensemble)]
    final: list[Ensemble] = []
    for time, cycle in enumerate(
        _cycles(model, settings, ensemble, root, transforms=True), start=1
    ):
        forecast_means.append(cycle.forecast.mean)
        forecast_covariances.append(cycle.forecast.covariance)
        ensemble = cycle.analysis
        analysis_means.append(ensemble.mean)
        analysis_covariances.append(ensemble.covariance)
        while window and window[0][0] < time - lag:
            final.append(window.pop(0)[1])
        if cycle.transforms is not None:
            weights = cycle.transforms
            window = [(when, _transform(past, weights)) for when, past in window]
        window.append((time, ensemble))
    final.extend(past for _, past in window)
    record = EnsembleKalmanResult(
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
    return EnsembleSmootherResult(
        tuple(past.mean for past in final),
        tuple(past.covariance for past in final),
        tuple(final),
        lag,
        record,
    )
