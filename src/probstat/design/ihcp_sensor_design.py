"""Bayesian optimal design of embedded thermocouples for heat-flux inversion.

Experimental (added in 0.28).

Problem. A one-dimensional slab (thermal-protection-system style, SI units)
is heated on ``x = 0``; the surface flux ``q(t)`` (W/m^2), piecewise constant on
``n_q`` intervals of length ``Delta`` (s), is reconstructed from embedded
thermocouple readings (K). A sensor at depth ``x`` read every ``s`` intervals
contributes the rows of the lower-triangular Toeplitz sensitivity matrix of
:class:`~probstat.inverse.heat_conduction_inverse.SlabFluxOperator`
(EQ-INV-IHCP-006) at the reading times ``s Delta, 2 s Delta, ...``
[EQ-DES-IHCPD-001]. With a Matern Gaussian-process prior
``q ~ N(m0 1, C0)`` (EQ-INV-IHCP-010) and independent noise ``N(0, sigma^2)``
the posterior covariance of ``q`` is exact,
``C = (C0^-1 + A' A / sigma^2)^-1``, and does not depend on the data, so a
design is judged before any experiment.

Criteria. The A-, D- and goal-oriented criteria are those of
:func:`~probstat.design.linear_optimal.linear_design_criteria`
(EQ-DES-LIN-001 to EQ-DES-LIN-004), evaluated on the stacked candidate rows
of the selected sensors; this module only groups rows into physical sensors.
Two goal functionals are built in [EQ-DES-IHCPD-003]: the total absorbed heat
``E = integral q dt ~ Delta sum_j q_j`` (J/m^2) and the flux in the interval
containing a chosen peak time.

Selection. :func:`greedy_sensor_selection` adds the physical sensor with the
best criterion value at each step and :func:`exhaustive_sensor_selection`
enumerates all subsets of a given size; :func:`greedy_gap` measures the
difference, so no optimality of the greedy rule is assumed
[EQ-DES-IHCPD-004]. The helper studies measure, rather than assume, how the
optimal depths move with the noise level and the prior length scale
(:func:`optimal_depth_study`), the value of a second sensor against halving
the noise of one (:func:`second_sensor_value`, EQ-DES-IHCPD-007) and which
flux frequencies a sensor depth resolves (:func:`sensor_frequency_profile`,
EQ-DES-IHCPD-006). :func:`posterior_error_monte_carlo` checks the design
formulas against the empirical error variance of
:func:`~probstat.inverse.heat_conduction_inverse.bayesian_flux_posterior`
over data simulated from the prior (EQ-DES-IHCPD-005).

Assumptions:
    Linear, constant-property conduction in the finite-volume model of the
    inverse module; exact sensor positions and calibration; independent
    Gaussian reading noise of one standard deviation; a stationary Matern
    prior of known amplitude and length scale. The synthetic slab is NOT a real
    heat shield. Results hold for the stated prior and noise model, not for
    model error or temperature-dependent properties.

Common pitfalls:
    - The A criterion is in (W/m^2)^2 and depends on the discretization of
      ``q``; the goal variances are physically interpretable (J/m^2 for the
      absorbed heat, W/m^2 for the peak flux).
    - Two sensors at the same depth are replicates, not new information about
      depth; by default a depth may be used once (``distinct_depths``).
    - A very deep sensor responds late and weakly: its information may be
      negligible within the horizon, which :class:`IHCPDesignEvaluation` shows
      as a variance reduction near zero.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from probstat.design.linear_optimal import (
    LinearDesignCriteria,
    linear_design_criteria,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inverse.heat_conduction_inverse import (
    SlabFluxOperator,
    SlabProperties,
    bayesian_flux_posterior,
    matern_covariance,
)
from probstat.linalg import Matrix, Vector, cholesky, matvec, symmetric_eigen

_CRITERIA = ("a", "d", "total_heat", "peak_flux")
_NEGLIGIBLE_GAIN = 1e-12


def _positive(name: str, value: object) -> float:
    """Return ``value`` as a finite positive float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not positive and finite.

    References:
        [EQ-DES-IHCPD-001]. Input validation only.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _count(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an integer ``>= minimum``.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        [EQ-DES-IHCPD-001]. Input validation only.

    """
    number = require_non_negative_integer(value, name=name)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _check_criterion(criterion: str) -> str:
    """Validate a criterion name.

    Raises:
        ValueError: If it is not one of ``a``, ``d``, ``total_heat``,
            ``peak_flux``.

    References:
        [EQ-DES-IHCPD-002]. Input validation only.

    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    return criterion


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IHCPSensorCandidate:
    """One candidate thermocouple: a depth and a sampling interval.

    Attributes:
        depth: Sensor depth ``x`` in m from the heated face.
        sample_every: Readings every this many flux intervals.
        rows: Indices of the candidate's readings in
            :attr:`IHCPDesignProblem.rows` (time ordered).

    References:
        [EQ-DES-IHCPD-001].

    """

    depth: float
    sample_every: int
    rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IHCPDesignProblem:
    """A finite design problem for thermocouple placement.

    Attributes:
        slab: The :class:`SlabProperties` (synthetic in the tests).
        candidates: The candidate sensors.
        rows: Stacked reading rows of all candidates (K per W/m^2), one row per
            reading, a matrix with ``n_q`` columns.
        noise_std: Reading noise floor ``sigma`` in K.
        relative_noise: Fraction ``f`` of the reference reading added in
            quadrature to the noise (``0`` for constant noise).
        reference_readings: ``|a_r' q_ref|`` for every stacked row, the clean
            reading under the reference flux (K; zeros without a reference).
        flux_times: Interval midpoint times (s).
        interval_length: ``Delta`` in s.
        prior_covariance: Matern covariance ``C0`` of ``q`` ((W/m^2)^2).
        amplitude: Prior standard deviation ``s`` (W/m^2).
        length_scale: Prior length scale ``l`` (s).
        smoothness: Matern smoothness (``0.5``, ``1.5`` or ``2.5``).
        peak_time: Time (s) whose interval defines the peak-flux goal.
        total_heat_goal: ``Delta (1, ..., 1)``, so ``c' q ~ integral q dt``.
        peak_flux_goal: Unit vector of the interval containing ``peak_time``.
        excluded_depths: Depths removed because their reference reading
            exceeds the survivable limit (m).

    References:
        [EQ-DES-IHCPD-001], [EQ-DES-IHCPD-003].

    """

    slab: SlabProperties
    candidates: tuple[IHCPSensorCandidate, ...]
    rows: Matrix
    noise_std: float
    relative_noise: float
    reference_readings: Vector
    flux_times: Vector
    interval_length: float
    prior_covariance: Matrix
    amplitude: float
    length_scale: float
    smoothness: float
    peak_time: float
    total_heat_goal: Vector
    peak_flux_goal: Vector
    excluded_depths: tuple[float, ...]

    @property
    def noise_variances(self) -> Vector:
        """Return the per-row noise variance ``sigma^2 + (f r)^2``.

        With ``f = 0`` every row has the variance ``sigma^2``; otherwise the
        variance grows with the reference reading ``r`` of the row, as for a
        thermocouple whose tolerance is partly proportional to the reading
        [EQ-DES-IHCPD-001].

        Returns:
            One variance per stacked row.

        References:
            [EQ-DES-IHCPD-001].

        """
        floor = self.noise_std * self.noise_std
        return tuple(
            floor + (self.relative_noise * r) ** 2 for r in self.reference_readings
        )

    @property
    def depths(self) -> tuple[float, ...]:
        """Return the candidate depths (m), in candidate order.

        Returns:
            One depth per candidate.

        References:
            [EQ-DES-IHCPD-001].

        """
        return tuple(c.depth for c in self.candidates)


def build_ihcp_design_problem(
    slab: SlabProperties,
    depths: Sequence[float],
    *,
    flux_intervals: int,
    interval_length: float,
    noise_std: float,
    amplitude: float,
    length_scale: float,
    smoothness: float = 0.5,
    sample_every_options: Sequence[int] = (1,),
    relative_noise: float = 0.0,
    reference_flux: Sequence[float] | None = None,
    max_reading: float | None = None,
    peak_time: float | None = None,
    cells: int = 40,
    substeps: int = 4,
) -> IHCPDesignProblem:
    """Build the candidate rows, the Matern prior and the two goals.

    For every depth and every sampling option ``s`` the readings at
    ``s Delta, 2 s Delta, ...`` are the rows ``s - 1, 2 s - 1, ...`` of the
    sensitivity matrix of a single-sensor
    :class:`~probstat.inverse.heat_conduction_inverse.SlabFluxOperator`
    (EQ-INV-IHCP-006, EQ-DES-IHCPD-001).

    Args:
        slab: Slab properties.
        depths: Candidate depths in m, each valid for the model (inside
            ``[dx / 2, L - dx / 2]``) and pairwise distinct.
        flux_intervals: Number ``n_q`` of flux intervals.
        interval_length: ``Delta`` in s.
        noise_std: Reading noise ``sigma`` in K.
        amplitude: Prior standard deviation of ``q`` in W/m^2.
        length_scale: Prior length scale in s.
        smoothness: Matern smoothness ``0.5``, ``1.5`` or ``2.5``.
        sample_every_options: Allowed sampling periods in flux intervals; each
            must divide ``flux_intervals``.
        relative_noise: Fraction ``f >= 0`` of the clean reading under
            ``reference_flux`` added in quadrature to the noise floor, so a hot
            (shallow) sensor is noisier: ``var = sigma^2 + (f r)^2``. A
            modelling choice for thermocouple tolerances, not a standard.
        reference_flux: Representative flux history (W/m^2, length ``n_q``)
            used for ``relative_noise`` and ``max_reading``.
        max_reading: Survivable temperature rise of a thermocouple in K. A
            depth whose largest clean reading under ``reference_flux`` exceeds
            it is not a candidate (listed in ``excluded_depths``); it needs a
            ``reference_flux``.
        peak_time: Time in s of the peak-flux goal; the middle of the horizon
            by default.
        cells: Cells of the forward model.
        substeps: Time steps per flux interval.

    Returns:
        The :class:`IHCPDesignProblem`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a depth repeats, a sampling option does not divide
            ``flux_intervals``, ``peak_time`` is outside the horizon or an
            argument is out of range.

    References:
        [EQ-DES-IHCPD-001], [EQ-DES-IHCPD-003], [EQ-INV-IHCP-006],
        [EQ-INV-IHCP-010], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    n_q = _count("flux_intervals", flux_intervals, 1)
    delta = _positive("interval_length", interval_length)
    sigma = _positive("noise_std", noise_std)
    fraction = require_finite_real(relative_noise, name="relative_noise")
    if fraction < 0.0:
        raise ValueError("relative_noise must be non-negative")
    reference: Vector | None = None
    if reference_flux is not None:
        reference = tuple(
            require_finite_real(v, name="reference_flux") for v in reference_flux
        )
        if len(reference) != n_q:
            raise ValueError("reference_flux must have flux_intervals entries")
    if fraction > 0.0 and reference is None:
        raise ValueError("relative_noise > 0 needs a reference_flux")
    limit = None if max_reading is None else _positive("max_reading", max_reading)
    if limit is not None and reference is None:
        raise ValueError("max_reading needs a reference_flux")
    if isinstance(depths, (str, bytes)):
        raise TypeError("depths must be a sequence of depths")
    depth_list = tuple(require_finite_real(d, name="depth") for d in depths)
    if not depth_list:
        raise ValueError("at least one candidate depth is required")
    if len(set(depth_list)) != len(depth_list):
        raise ValueError("candidate depths must be distinct")
    options = tuple(_count("sample_every option", s, 1) for s in sample_every_options)
    if not options or len(set(options)) != len(options):
        raise ValueError("sample_every_options must be non-empty and distinct")
    if any(n_q % s != 0 for s in options):
        raise ValueError("every sample_every option must divide flux_intervals")
    horizon = n_q * delta
    peak = 0.5 * horizon if peak_time is None else _positive("peak_time", peak_time)
    if peak > horizon:
        raise ValueError("peak_time must lie inside the horizon")
    rows: list[tuple[float, ...]] = []
    candidates: list[IHCPSensorCandidate] = []
    times: Vector | None = None
    excluded: list[float] = []
    for depth in depth_list:
        operator = SlabFluxOperator(
            slab,
            [depth],
            flux_intervals=n_q,
            interval_length=delta,
            cells=cells,
            substeps=substeps,
        )
        times = operator.flux_times
        dense = operator.sensitivity_matrix()
        if limit is not None:
            assert reference is not None
            hottest = max(
                abs(math.fsum(a * q for a, q in zip(row, reference, strict=True)))
                for row in dense
            )
            if hottest > limit:
                excluded.append(depth)
                continue
        for every in options:
            start = len(rows)
            rows.extend(dense[every - 1 :: every])
            candidates.append(
                IHCPSensorCandidate(
                    depth, every, tuple(range(start, start + n_q // every))
                )
            )
    if not candidates:
        raise ValueError("every candidate depth exceeds max_reading")
    assert times is not None
    prior = matern_covariance(
        times, amplitude=amplitude, length_scale=length_scale, smoothness=smoothness
    )
    peak_index = min(int(peak / delta), n_q - 1)
    readings = tuple(
        0.0
        if reference is None
        else abs(math.fsum(a * q for a, q in zip(row, reference, strict=True)))
        for row in rows
    )
    return IHCPDesignProblem(
        slab,
        tuple(candidates),
        tuple(rows),
        sigma,
        fraction,
        readings,
        times,
        delta,
        prior,
        float(amplitude),
        float(length_scale),
        float(smoothness),
        peak,
        (delta,) * n_q,
        tuple(1.0 if j == peak_index else 0.0 for j in range(n_q)),
        tuple(excluded),
    )


def ihcp_problem_with(
    problem: IHCPDesignProblem,
    *,
    noise_std: float | None = None,
    relative_noise: float | None = None,
    amplitude: float | None = None,
    length_scale: float | None = None,
) -> IHCPDesignProblem:
    """Return the problem with another noise level or prior; rows are reused.

    Args:
        problem: The base problem.
        noise_std: New reading noise floor in K, or ``None`` to keep it.
        relative_noise: New relative noise fraction, or ``None`` to keep it
            (a positive value needs the problem to have a reference flux).
        amplitude: New prior amplitude in W/m^2, or ``None``.
        length_scale: New prior length scale in s, or ``None``.

    Returns:
        A new :class:`IHCPDesignProblem`.

    Raises:
        TypeError: If ``problem`` is not an :class:`IHCPDesignProblem` or a
            value is not a real number.
        ValueError: If a value is not positive.

    References:
        [EQ-DES-IHCPD-001], [EQ-INV-IHCP-010].

    """
    if not isinstance(problem, IHCPDesignProblem):
        raise TypeError("problem must be an IHCPDesignProblem")
    sigma = (
        problem.noise_std if noise_std is None else _positive("noise_std", noise_std)
    )
    fraction = (
        problem.relative_noise
        if relative_noise is None
        else require_finite_real(relative_noise, name="relative_noise")
    )
    if fraction < 0.0:
        raise ValueError("relative_noise must be non-negative")
    if fraction > 0.0 and not any(problem.reference_readings):
        raise ValueError(
            "relative_noise > 0 needs a problem built with a reference_flux"
        )
    amp = problem.amplitude if amplitude is None else amplitude
    ell = problem.length_scale if length_scale is None else length_scale
    prior = matern_covariance(
        problem.flux_times,
        amplitude=amp,
        length_scale=ell,
        smoothness=problem.smoothness,
    )
    return replace(
        problem,
        noise_std=sigma,
        relative_noise=fraction,
        prior_covariance=prior,
        amplitude=float(amp),
        length_scale=float(ell),
    )


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IHCPDesignEvaluation:
    """Posterior criteria of a set of sensors.

    Attributes:
        selected: Candidate indices (in the order given).
        depths: Their depths (m).
        a_optimal: ``trace(C)`` ((W/m^2)^2).
        d_optimal: ``log det(I + H)`` (larger is better).
        total_heat_variance: ``c' C c`` for the absorbed heat ((J/m^2)^2).
        peak_flux_variance: ``C_jj`` at the peak interval ((W/m^2)^2).
        prior_a: ``trace(C0)``.
        prior_total_heat_variance: ``c' C0 c``.
        prior_peak_flux_variance: ``C0_jj``.
        criteria: The full
            :class:`~probstat.design.linear_optimal.LinearDesignCriteria`
            (``goal_variance`` is the total-heat variance).

    References:
        [EQ-DES-IHCPD-002], [EQ-DES-IHCPD-003].

    """

    selected: tuple[int, ...]
    depths: tuple[float, ...]
    a_optimal: float
    d_optimal: float
    total_heat_variance: float
    peak_flux_variance: float
    prior_a: float
    prior_total_heat_variance: float
    prior_peak_flux_variance: float
    criteria: LinearDesignCriteria

    def value(self, criterion: str) -> float:
        """Return the value of a criterion (``d`` is maximized, others minimized).

        Args:
            criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.

        Returns:
            The criterion value.

        Raises:
            ValueError: For an unknown criterion.

        References:
            [EQ-DES-IHCPD-002].

        """
        _check_criterion(criterion)
        if criterion == "a":
            return self.a_optimal
        if criterion == "d":
            return self.d_optimal
        if criterion == "total_heat":
            return self.total_heat_variance
        return self.peak_flux_variance

    def reduction(self, criterion: str) -> float:
        """Return the relative variance reduction ``1 - post / prior``.

        For ``"d"`` the information gain ``d / 2`` in nats is returned.

        Args:
            criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.

        Returns:
            A fraction in ``[0, 1]`` (nats for ``"d"``).

        Raises:
            ValueError: For an unknown criterion.

        References:
            [EQ-DES-IHCPD-002].

        """
        _check_criterion(criterion)
        if criterion == "d":
            return 0.5 * self.d_optimal
        prior = {
            "a": self.prior_a,
            "total_heat": self.prior_total_heat_variance,
            "peak_flux": self.prior_peak_flux_variance,
        }[criterion]
        return 1.0 - self.value(criterion) / prior


def _selected_rows(problem: IHCPDesignProblem, selected: Sequence[int]) -> list[int]:
    """Return the stacked-row indices of the chosen candidates.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If an index is out of range or repeated.

    References:
        [EQ-DES-IHCPD-001]. Input validation only.

    """
    chosen = tuple(selected)
    for index in chosen:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("selected must contain integers")
        if not 0 <= index < len(problem.candidates):
            raise ValueError("selected index out of range")
    if len(set(chosen)) != len(chosen):
        raise ValueError("selected indices must be distinct")
    return [r for index in chosen for r in problem.candidates[index].rows]


def ihcp_design_criteria(
    problem: IHCPDesignProblem, selected: Sequence[int]
) -> IHCPDesignEvaluation:
    """Evaluate the A, D, total-heat and peak-flux criteria of a sensor set.

    The reading rows of the selected candidates are stacked and passed to
    :func:`~probstat.design.linear_optimal.linear_design_criteria` with the
    Matern prior and the total-heat goal; the peak-flux variance is the
    diagonal entry of the returned posterior covariance at the peak interval
    [EQ-DES-IHCPD-002], [EQ-DES-IHCPD-003]. An empty selection returns the
    prior values.

    Args:
        problem: The design problem.
        selected: Distinct candidate indices.

    Returns:
        :class:`IHCPDesignEvaluation`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an index is out of range or repeated.

    References:
        [EQ-DES-IHCPD-002], [EQ-DES-IHCPD-003], [EQ-DES-LIN-001],
        [EQ-DES-LIN-004], [REF-ALEXANDERIAN-2021].

    """
    if not isinstance(problem, IHCPDesignProblem):
        raise TypeError("problem must be an IHCPDesignProblem")
    indices = tuple(selected)
    stacked = _selected_rows(problem, indices)
    result = linear_design_criteria(
        problem.rows,
        problem.noise_variances,
        stacked,
        prior_covariance=problem.prior_covariance,
        goal=problem.total_heat_goal,
    )
    assert result.posterior_covariance is not None
    assert result.goal_variance is not None
    peak = problem.peak_flux_goal.index(1.0)
    prior = problem.prior_covariance
    n = len(prior)
    prior_heat = math.fsum(
        a * b
        for a, b in zip(
            problem.total_heat_goal, matvec(prior, problem.total_heat_goal), strict=True
        )
    )
    return IHCPDesignEvaluation(
        indices,
        tuple(problem.candidates[i].depth for i in indices),
        result.a_optimal,
        result.d_optimal,
        result.goal_variance,
        result.posterior_covariance[peak][peak],
        math.fsum(prior[i][i] for i in range(n)),
        prior_heat,
        prior[peak][peak],
        result,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IHCPSensorSelection:
    """A chosen sensor set with its criterion history.

    Attributes:
        indices: Selected candidate indices (greedy: in selection order;
            exhaustive: ascending).
        depths: Their depths (m).
        sample_every: Their sampling periods.
        criterion: Criterion name.
        value: Final criterion value.
        step_values: Criterion value after each greedy step (a single entry
            for the exhaustive optimum).
        gains: Improvement of each greedy step (positive is better for both
            directions of optimization).
        prior_value: Criterion value of the empty design.
        evaluation: The :class:`IHCPDesignEvaluation` of the final set.
        method: ``"greedy"`` or ``"exhaustive"``.
        ranking: ``(indices, value)`` of every subset, best first
            (exhaustive only; empty for greedy).
        stop_reason: ``"size_reached"`` or ``"no_positive_gain"``.

    References:
        [EQ-DES-IHCPD-004].

    """

    indices: tuple[int, ...]
    depths: tuple[float, ...]
    sample_every: tuple[int, ...]
    criterion: str
    value: float
    step_values: Vector
    gains: Vector
    prior_value: float
    evaluation: IHCPDesignEvaluation
    method: str
    ranking: tuple[tuple[tuple[int, ...], float], ...]
    stop_reason: str


def _better(criterion: str, candidate: float, incumbent: float) -> bool:
    """Return whether ``candidate`` beats ``incumbent`` for the criterion.

    References:
        [EQ-DES-IHCPD-004]. A comparison only.

    """
    return candidate > incumbent if criterion == "d" else candidate < incumbent


def _depth_conflict(
    problem: IHCPDesignProblem, chosen: Sequence[int], candidate: int
) -> bool:
    """Return whether ``candidate`` repeats the depth of a chosen sensor.

    References:
        [EQ-DES-IHCPD-004]. A comparison only.

    """
    depth = problem.candidates[candidate].depth
    return any(problem.candidates[i].depth == depth for i in chosen)


def greedy_sensor_selection(
    problem: IHCPDesignProblem,
    size: int,
    *,
    criterion: str = "a",
    distinct_depths: bool = True,
) -> IHCPSensorSelection:
    """Add physical sensors one at a time, each time the best by the criterion.

    Every remaining candidate is scored by
    :func:`ihcp_design_criteria` on the current set plus that candidate (the
    library criteria are reused as they are, with no rank-one shortcut); ties
    go to the lowest index. No optimality is claimed: :func:`greedy_gap`
    measures the difference to :func:`exhaustive_sensor_selection`
    [EQ-DES-IHCPD-004]. A step with a gain below ``1e-12`` of the current value
    stops the selection with ``stop_reason = "no_positive_gain"``.

    Args:
        problem: The design problem.
        size: Number ``k >= 1`` of sensors to select.
        criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.
        distinct_depths: Forbid a second candidate at an already used depth.

    Returns:
        :class:`IHCPSensorSelection`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``size`` exceeds what the candidates allow or the
            criterion is unknown.

    References:
        [EQ-DES-IHCPD-004], [EQ-DES-LIN-006], [REF-ALEXANDERIAN-2021].

    """
    if not isinstance(problem, IHCPDesignProblem):
        raise TypeError("problem must be an IHCPDesignProblem")
    _check_criterion(criterion)
    k = _count("size", size, 1)
    empty = ihcp_design_criteria(problem, ())
    prior_value = 0.0 if criterion == "d" else empty.value(criterion)
    chosen: list[int] = []
    values: list[float] = []
    gains: list[float] = []
    current = prior_value
    stop = "size_reached"
    evaluation = empty
    while len(chosen) < k:
        best: tuple[int, float, IHCPDesignEvaluation] | None = None
        for index in range(len(problem.candidates)):
            if index in chosen:
                continue
            if distinct_depths and _depth_conflict(problem, chosen, index):
                continue
            trial = ihcp_design_criteria(problem, (*chosen, index))
            value = trial.value(criterion)
            if best is None or _better(criterion, value, best[1]):
                best = (index, value, trial)
        if best is None:
            raise ValueError("size exceeds the number of admissible sensors")
        gain = best[1] - current if criterion == "d" else current - best[1]
        if gain <= _NEGLIGIBLE_GAIN * max(abs(current), 1.0) and chosen:
            stop = "no_positive_gain"
            break
        chosen.append(best[0])
        values.append(best[1])
        gains.append(gain)
        current = best[1]
        evaluation = best[2]
    return IHCPSensorSelection(
        tuple(chosen),
        tuple(problem.candidates[i].depth for i in chosen),
        tuple(problem.candidates[i].sample_every for i in chosen),
        criterion,
        current,
        tuple(values),
        tuple(gains),
        prior_value,
        evaluation,
        "greedy",
        (),
        stop,
    )


def exhaustive_sensor_selection(
    problem: IHCPDesignProblem,
    size: int,
    *,
    criterion: str = "a",
    distinct_depths: bool = True,
    max_combinations: int = 5000,
) -> IHCPSensorSelection:
    """Enumerate every admissible subset of ``size`` sensors (the oracle).

    Args:
        problem: The design problem.
        size: Number ``k >= 1`` of sensors.
        criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.
        distinct_depths: Skip subsets that use a depth twice.
        max_combinations: Refuse enumerations of more subsets than this
            (``C(N, k)`` before the depth filter).

    Returns:
        :class:`IHCPSensorSelection` with the full ranking, best first; ties
        are broken by the lexicographic order of the indices.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If no admissible subset exists or ``C(N, k)`` exceeds the
            cap.

    References:
        [EQ-DES-IHCPD-004], [EQ-DES-LIN-008], [REF-ALEXANDERIAN-2021].

    """
    if not isinstance(problem, IHCPDesignProblem):
        raise TypeError("problem must be an IHCPDesignProblem")
    _check_criterion(criterion)
    k = _count("size", size, 1)
    cap = _count("max_combinations", max_combinations, 1)
    total = math.comb(len(problem.candidates), k)
    if total > cap:
        raise ValueError(f"{total} combinations exceed max_combinations={cap}")
    scored: list[tuple[tuple[int, ...], float, IHCPDesignEvaluation]] = []
    for subset in itertools.combinations(range(len(problem.candidates)), k):
        depths = [problem.candidates[i].depth for i in subset]
        if distinct_depths and len(set(depths)) != len(depths):
            continue
        trial = ihcp_design_criteria(problem, subset)
        scored.append((subset, trial.value(criterion), trial))
    if not scored:
        raise ValueError("no admissible subset of this size exists")
    sign = -1.0 if criterion == "d" else 1.0
    scored.sort(key=lambda item: (sign * item[1], item[0]))
    best = scored[0]
    empty = ihcp_design_criteria(problem, ())
    return IHCPSensorSelection(
        best[0],
        tuple(problem.candidates[i].depth for i in best[0]),
        tuple(problem.candidates[i].sample_every for i in best[0]),
        criterion,
        best[1],
        (best[1],),
        (),
        0.0 if criterion == "d" else empty.value(criterion),
        best[2],
        "exhaustive",
        tuple((s, v) for s, v, _ in scored),
        "size_reached",
    )


@dataclass(frozen=True, slots=True)
class IHCPGreedyGap:
    """Greedy against exhaustive selection of the same size.

    Attributes:
        greedy: The greedy selection.
        exhaustive: The exhaustive optimum.
        same_set: Whether both pick the same sensors.
        relative_gap: ``|greedy - optimum| / |optimum - prior|``, the share of
            the optimal improvement that greedy misses (``0`` when equal).
        size: Number of sensors.

    References:
        [EQ-DES-IHCPD-004].

    """

    greedy: IHCPSensorSelection
    exhaustive: IHCPSensorSelection
    same_set: bool
    relative_gap: float
    size: int


def greedy_gap(
    problem: IHCPDesignProblem,
    size: int,
    *,
    criterion: str = "a",
    distinct_depths: bool = True,
    max_combinations: int = 5000,
) -> IHCPGreedyGap:
    """Measure how far the greedy selection is from the exhaustive optimum.

    The gap is expressed as the fraction of the optimal improvement over the
    empty design that the greedy set forgoes, so it is comparable across
    criteria and noise levels [EQ-DES-IHCPD-004].

    Args:
        problem: The design problem.
        size: Number of sensors.
        criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.
        distinct_depths: Forbid a repeated depth.
        max_combinations: Cap on the enumeration.

    Returns:
        :class:`IHCPGreedyGap`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: As the two selection routines, or if the greedy rule
            stopped before reaching ``size``.

    References:
        [EQ-DES-IHCPD-004], [REF-ALEXANDERIAN-2021].

    """
    greedy = greedy_sensor_selection(
        problem, size, criterion=criterion, distinct_depths=distinct_depths
    )
    if len(greedy.indices) != size:
        raise ValueError("greedy selection stopped before reaching size")
    best = exhaustive_sensor_selection(
        problem,
        size,
        criterion=criterion,
        distinct_depths=distinct_depths,
        max_combinations=max_combinations,
    )
    improvement = abs(best.value - best.prior_value)
    gap = abs(greedy.value - best.value) / improvement if improvement > 0.0 else 0.0
    return IHCPGreedyGap(
        greedy,
        best,
        set(greedy.indices) == set(best.indices),
        gap,
        size,
    )


# ---------------------------------------------------------------------------
# Studies of the physical findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IHCPDepthStudyRow:
    """Optimal depths of one (noise, length scale, size) cell.

    Attributes:
        noise_std: Reading noise (K).
        length_scale: Prior length scale (s).
        size: Number of sensors ``k``.
        depths: Exhaustive-optimal depths (m, ascending candidate order).
        greedy_depths: Greedy depths (m, in selection order).
        value: Optimal criterion value.
        relative_gap: Share of the optimal improvement missed by greedy.
        reduction: Relative reduction (or information gain for ``"d"``) of the
            optimum.

    References:
        [EQ-DES-IHCPD-004].

    """

    noise_std: float
    length_scale: float
    size: int
    depths: tuple[float, ...]
    greedy_depths: tuple[float, ...]
    value: float
    relative_gap: float
    reduction: float


def optimal_depth_study(
    problem: IHCPDesignProblem,
    sizes: Sequence[int],
    noise_stds: Sequence[float],
    length_scales: Sequence[float],
    *,
    criterion: str = "a",
) -> tuple[IHCPDepthStudyRow, ...]:
    """Measure the optimal depths over a grid of noise levels and length scales.

    For every pair ``(sigma, l)`` the problem is re-priored with
    :func:`ihcp_problem_with` and the optimal sets of each size are found
    exhaustively, with the greedy set and its gap alongside. Nothing is assumed
    about the trend: the rows are measurements [EQ-DES-IHCPD-004].

    Args:
        problem: The base design problem.
        sizes: Sensor counts ``k``.
        noise_stds: Reading noise levels in K.
        length_scales: Prior length scales in s.
        criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.

    Returns:
        One :class:`IHCPDepthStudyRow` per ``(noise, length scale, size)``,
        noise-major, then length scale, then size.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a grid is empty or a value is invalid.

    References:
        [EQ-DES-IHCPD-004], [REF-ALEXANDERIAN-2021].

    """
    if not sizes or not noise_stds or not length_scales:
        raise ValueError("sizes, noise_stds and length_scales must be non-empty")
    rows: list[IHCPDepthStudyRow] = []
    for sigma in noise_stds:
        for ell in length_scales:
            base = ihcp_problem_with(problem, noise_std=sigma, length_scale=ell)
            for size in sizes:
                gap = greedy_gap(base, size, criterion=criterion)
                best = gap.exhaustive
                rows.append(
                    IHCPDepthStudyRow(
                        base.noise_std,
                        base.length_scale,
                        size,
                        best.depths,
                        gap.greedy.depths,
                        best.value,
                        gap.relative_gap,
                        best.evaluation.reduction(criterion),
                    )
                )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class IHCPSecondSensorValue:
    """A second sensor against halving the noise of one.

    Attributes:
        criterion: Criterion name.
        prior: Value of the empty design.
        one: Best single sensor at noise ``sigma``.
        one_half_noise: Best single sensor at noise ``sigma / 2``.
        two: Best pair at noise ``sigma``.
        one_depth: Depth of ``one``.
        one_half_noise_depth: Depth of ``one_half_noise``.
        two_depths: Depths of the best pair.
        winner: ``"second_sensor"`` or ``"half_noise"``.

    References:
        [EQ-DES-IHCPD-007].

    """

    criterion: str
    prior: float
    one: float
    one_half_noise: float
    two: float
    one_depth: float
    one_half_noise_depth: float
    two_depths: tuple[float, ...]
    winner: str


def second_sensor_value(
    problem: IHCPDesignProblem, *, criterion: str = "a"
) -> IHCPSecondSensorValue:
    """Compare the best pair of sensors with the best single sensor at half noise.

    Halving the noise standard deviation of every reading multiplies the Fisher
    information of the sensor by four, a second distinct sensor adds a different
    row space [EQ-DES-IHCPD-007]. Both options are scored with the exhaustive
    optimum of their own problem; the result states which is better for the
    criterion (smaller variance, larger ``d``), it does not assume it.

    Args:
        problem: The design problem.
        criterion: ``"a"``, ``"d"``, ``"total_heat"`` or ``"peak_flux"``.

    Returns:
        :class:`IHCPSecondSensorValue`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If fewer than two distinct depths exist.

    References:
        [EQ-DES-IHCPD-007], [REF-ALEXANDERIAN-2021].

    """
    _check_criterion(criterion)
    one = exhaustive_sensor_selection(problem, 1, criterion=criterion)
    two = exhaustive_sensor_selection(problem, 2, criterion=criterion)
    halved = ihcp_problem_with(
        problem,
        noise_std=0.5 * problem.noise_std,
        relative_noise=0.5 * problem.relative_noise,
    )
    half = exhaustive_sensor_selection(halved, 1, criterion=criterion)
    pair_better = _better(criterion, two.value, half.value)
    return IHCPSecondSensorValue(
        criterion,
        one.prior_value,
        one.value,
        half.value,
        two.value,
        one.depths[0],
        half.depths[0],
        two.depths,
        "second_sensor" if pair_better else "half_noise",
    )


@dataclass(frozen=True, slots=True)
class IHCPFrequencyProfile:
    """Which flux modes a sensor set resolves.

    Modes are the eigenvectors of the prior covariance ordered by decreasing
    prior variance (smooth, low-frequency first for a stationary kernel).

    Attributes:
        selected: Candidate indices.
        depths: Their depths (m).
        prior_variances: Prior variance of each mode (eigenvalues of ``C0``).
        resolved: ``1 - v' C v / lambda`` per mode, in ``[0, 1]``: the share of
            the prior variance of the mode removed by the data.
        cutoff_modes: Number of modes with ``resolved >= 0.5``.
        low_resolved: Mean ``resolved`` over the leading third of the modes.
        high_resolved: Mean ``resolved`` over the trailing third.
        total_heat_ratio: Posterior over prior variance of the absorbed heat.
        peak_flux_ratio: Posterior over prior variance of the peak flux.

    References:
        [EQ-DES-IHCPD-006].

    """

    selected: tuple[int, ...]
    depths: tuple[float, ...]
    prior_variances: Vector
    resolved: Vector
    cutoff_modes: int
    low_resolved: float
    high_resolved: float
    total_heat_ratio: float
    peak_flux_ratio: float


def sensor_frequency_profile(
    problem: IHCPDesignProblem, selected: Sequence[int]
) -> IHCPFrequencyProfile:
    """Measure the prior-mode resolution of a sensor set.

    With ``C0 = V diag(lambda) V'`` and the posterior covariance ``C``, the
    resolved share of mode ``i`` is ``1 - v_i' C v_i / lambda_i``; the leading
    modes are the slowly varying flux components that carry the absorbed
    energy, the trailing ones the rapid components that locate a peak
    [EQ-DES-IHCPD-006].

    Args:
        problem: The design problem.
        selected: Candidate indices (non-empty).

    Returns:
        :class:`IHCPFrequencyProfile`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``selected`` is empty or invalid.

    References:
        [EQ-DES-IHCPD-006], [REF-HANSEN-2008], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    if not tuple(selected):
        raise ValueError("selected must not be empty")
    evaluation = ihcp_design_criteria(problem, selected)
    cov = evaluation.criteria.posterior_covariance
    assert cov is not None
    values, vectors = symmetric_eigen(problem.prior_covariance)
    n = len(values)
    order = range(n - 1, -1, -1)
    prior_variances: list[float] = []
    resolved: list[float] = []
    for j in order:
        mode = tuple(vectors[i][j] for i in range(n))
        posterior = math.fsum(
            a * b for a, b in zip(mode, matvec(cov, mode), strict=True)
        )
        prior_variances.append(values[j])
        resolved.append(1.0 - posterior / values[j])
    third = max(n // 3, 1)
    return IHCPFrequencyProfile(
        evaluation.selected,
        evaluation.depths,
        tuple(prior_variances),
        tuple(resolved),
        sum(1 for r in resolved if r >= 0.5),
        math.fsum(resolved[:third]) / third,
        math.fsum(resolved[-third:]) / third,
        evaluation.total_heat_variance / evaluation.prior_total_heat_variance,
        evaluation.peak_flux_variance / evaluation.prior_peak_flux_variance,
    )


def depth_frequency_study(
    problem: IHCPDesignProblem,
) -> tuple[IHCPFrequencyProfile, ...]:
    """Return the frequency profile of every single-sensor candidate.

    Args:
        problem: The design problem.

    Returns:
        One :class:`IHCPFrequencyProfile` per candidate, in candidate order.

    Raises:
        TypeError: If ``problem`` is not an :class:`IHCPDesignProblem`.

    References:
        [EQ-DES-IHCPD-006].

    """
    if not isinstance(problem, IHCPDesignProblem):
        raise TypeError("problem must be an IHCPDesignProblem")
    return tuple(
        sensor_frequency_profile(problem, (i,)) for i in range(len(problem.candidates))
    )


# ---------------------------------------------------------------------------
# Validation against the estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IHCPMonteCarloCheck:
    """Design variances against the empirical error variances of the estimator.

    Attributes:
        replicates: Number of simulated data sets.
        design_variances: Diagonal of ``C`` from the design formulas.
        empirical_variances: Mean square of ``posterior mean - q`` per interval.
        reported_variances: Squares of ``FluxPosterior.std`` (the estimator's
            own marginal variances).
        design_total_heat: ``c' C c`` from the design formulas.
        empirical_total_heat: Mean square of ``c'(mean - q)``.
        design_peak_flux: ``C_jj`` at the peak interval.
        empirical_peak_flux: Mean square of the peak-interval error.
        relative_standard_error: ``sqrt(2 / replicates)``, the relative
            standard error of a mean-square estimate of a zero-mean Gaussian
            error.
        max_relative_deviation: Largest ``|empirical / design - 1|`` over the
            intervals, the total heat and the peak flux.
        max_reported_deviation: Largest relative difference between
            ``reported_variances`` and ``design_variances`` (deterministic;
            rounding only).

    References:
        [EQ-DES-IHCPD-005].

    """

    replicates: int
    design_variances: Vector
    empirical_variances: Vector
    reported_variances: Vector
    design_total_heat: float
    empirical_total_heat: float
    design_peak_flux: float
    empirical_peak_flux: float
    relative_standard_error: float
    max_relative_deviation: float
    max_reported_deviation: float


def posterior_error_monte_carlo(
    problem: IHCPDesignProblem,
    selected: Sequence[int],
    *,
    replicates: int = 300,
    seed: int | RandomStream = 0,
) -> IHCPMonteCarloCheck:
    """Check the design variances against the estimator over prior simulations.

    Each replicate draws ``q ~ N(0, C0)`` and ``y = A_sel q + S e`` (``S`` the
    per-row noise standard deviations), whitens rows and data by ``S^-1`` and
    runs
    :func:`~probstat.inverse.heat_conduction_inverse.bayesian_flux_posterior`
    with unit noise and the same prior, which is the same posterior; it
    records ``mean - q``. Under the model
    the error is ``N(0, C)`` with ``C`` the data-independent design covariance
    [EQ-DES-IHCPD-005], so its mean square estimates ``C_ii``, ``c' C c`` and
    ``C_jj`` with relative standard error ``sqrt(2 / replicates)``. The
    deviations are reported, not judged; the caller states the tolerance.

    Args:
        problem: The design problem.
        selected: Candidate indices (non-empty).
        replicates: Number of simulated data sets (``>= 2``).
        seed: Seed or stream for the draws.

    Returns:
        :class:`IHCPMonteCarloCheck`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``selected`` is empty or ``replicates < 2``.

    References:
        [EQ-DES-IHCPD-005], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    if not tuple(selected):
        raise ValueError("selected must not be empty")
    count = _count("replicates", replicates, 2)
    evaluation = ihcp_design_criteria(problem, selected)
    cov = evaluation.criteria.posterior_covariance
    assert cov is not None
    stacked = _selected_rows(problem, selected)
    scale = tuple(math.sqrt(problem.noise_variances[r]) for r in stacked)
    matrix = tuple(
        tuple(a / sd for a in problem.rows[r])
        for r, sd in zip(stacked, scale, strict=True)
    )
    n = len(problem.flux_times)
    factor = cholesky(problem.prior_covariance)
    generator = resolve_stream(seed).generator()
    peak = problem.peak_flux_goal.index(1.0)
    sums = [0.0] * n
    reported: Vector = ()
    heat_sum = 0.0
    peak_sum = 0.0
    for _ in range(count):
        z = tuple(generator.gauss(0.0, 1.0) for _ in range(n))
        truth = matvec(factor, z)
        clean = matvec(matrix, truth)
        data = tuple(v + generator.gauss(0.0, 1.0) for v in clean)
        posterior = bayesian_flux_posterior(
            matrix,
            data,
            1.0,
            problem.flux_times,
            amplitude=problem.amplitude,
            length_scale=problem.length_scale,
            smoothness=problem.smoothness,
            prior_mean=0.0,
        )
        error = tuple(m - t for m, t in zip(posterior.mean, truth, strict=True))
        for i, e in enumerate(error):
            sums[i] += e * e
        heat = math.fsum(
            g * e for g, e in zip(problem.total_heat_goal, error, strict=True)
        )
        heat_sum += heat * heat
        peak_sum += error[peak] ** 2
        reported = tuple(s * s for s in posterior.std)
    design = tuple(cov[i][i] for i in range(n))
    empirical = tuple(s / count for s in sums)
    deviation = max(
        [abs(e / d - 1.0) for e, d in zip(empirical, design, strict=True)]
        + [
            abs(heat_sum / count / evaluation.total_heat_variance - 1.0),
            abs(peak_sum / count / evaluation.peak_flux_variance - 1.0),
        ]
    )
    reported_deviation = max(
        abs(r / d - 1.0) for r, d in zip(reported, design, strict=True)
    )
    return IHCPMonteCarloCheck(
        count,
        design,
        empirical,
        reported,
        evaluation.total_heat_variance,
        heat_sum / count,
        evaluation.peak_flux_variance,
        peak_sum / count,
        math.sqrt(2.0 / count),
        deviation,
        reported_deviation,
    )
