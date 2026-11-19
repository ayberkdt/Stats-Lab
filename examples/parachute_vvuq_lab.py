"""Parachuted-package laboratory: verification, propagation, and validation.

A package leaves a plane at height ``z_a`` with the plane's horizontal speed
``s_p`` and falls under gravity and quadratic aerodynamic drag in a
horizontal wind ``s_w``. The modelled landing abscissa ``y_mod`` solves

    m x'' = m g - A (x' - s_w) |x' - s_w|,  x'(0) = (s_p, 0),  x(0) = (x0, z_a),

and the simulated abscissa ``y_sim`` replaces the exact solution by explicit
Euler steps of size ``dt`` (the example of the ATHENS course "Introduction to
V&V-UQ", REF-PERRIN-ATHENS-VVUQ). The laboratory answers the course's three
questions with the library and with checks whose answers are known:

1. **Verification.** Does ``y_sim`` converge to ``y_mod`` as ``dt -> 0``, at
   which order, and how should ``dt`` be chosen? *Code verification* first
   checks the solver against exact solutions of simplified models: linear
   drag (closed form through the Lambert W function) must show the
   theoretical orders 1, 2, and 4 of Euler, Heun, and fourth-order
   Runge-Kutta (RK4); the drag-free parabola must be reproduced; the
   quadratic-drag package must land at its terminal speed. *Solution
   verification* then estimates the error of Euler, with the ground crossing
   interpolated linearly in the last step, on the real model, which has no
   closed form: the observed order, Richardson extrapolation, and the Grid
   Convergence Index (GCI) come from ``probstat.uq`` and are checked against
   an RK4 reference whose crossing is located on the cubic Hermite
   interpolant. ``dt`` is the largest step whose verified error bound stays
   below 5% of the parametric standard deviation and inside the Euler
   stability limit.
2. **Propagation.** Stated laws (wind aleatory; dropping position, plane
   speed, mass, and drag epistemic) are pushed through the simulator with a
   randomized Sobol design: the 90% landing interval ``I_alpha`` and the
   Sobol ranking of the six inputs.
3. **Validation.** Synthetic "measured" landings come from a richer truth
   whose wind grows with height (a power-law profile) while the simulator
   uses the 10 m wind. The area metric and u-pooling expose the
   disagreement, ``I_alpha`` undercovers real landings, and three
   model-error structures fitted on a separate campaign are compared: a
   zero-mean Gaussian inflation, a constant bias, and a bias proportional to
   the wind.

Definitions:
    state ``(x, z, u, w)``: horizontal and vertical position (m) and velocity
        (m/s); the package lands when ``z = 0``; the landing abscissa ``y``
        is ``x`` at that instant.
    inputs ``x = (x0, z_a, s_p, s_w, m, A)``: dropping abscissa (m) and
        height (m), plane speed (m/s), 10 m wind speed (m/s), mass (kg), and
        drag factor ``A = rho C_d S / 2`` (kg/m; a scalar here, the course
        allows a matrix).
    ``y_mod``: exact landing abscissa of the model; ``y_sim(dt)``: Euler
        approximation; the RK4 Hermite solution at ``dt = 0.001 s`` stands in
        for ``y_mod``.
    code verification: comparison of the solver with exact solutions of
        simplified models (linear drag ``m x'' = m g - k (x' - s_w)`` with
        ``k = 10 kg/s``; no drag); the observed order against an exact
        value is ``log2(|e(dt)| / |e(dt/2)|)``.
    solution verification: estimation of the numerical error of the real
        model from the solutions on several steps, without an exact value.
    terminal speed ``v_t = sqrt(m g / A)``: the fall speed relative to the
        air at which quadratic drag balances weight.
    event location: "first step below ground" reports the first grid point
        with ``z <= 0``; "linear" interpolates the last step; "Hermite"
        uses the cubic Hermite interpolant of the step.
    observed order ``p``, Richardson extrapolation, and GCI: three-grid
        estimates of the order, of the zero-step value, and of the
        numerical uncertainty of the finest grid.
    step bound ``B(dt) = |y(dt) - y_ext| + U``: verified error bound of a
        production step ``dt`` from the grids ``dt, dt/2, dt/4``.
    ``I_alpha``: equal-tailed interval with ``P(y_sim in I_alpha) = 1 -
        alpha``, ``alpha = 0.1``.
    truth: power-law wind ``s_w (z / 10)^0.1`` above 1 m, the true mass and
        drag ``(100 kg, 10 kg/m)``, and drop-to-drop scatter of position,
        height, and plane speed; landings are measured with a 2 m standard
        error.
    model error ``delta``: ``y_real - y_mod``; structures: zero-mean
        Gaussian, constant bias ``b``, and wind-proportional ``beta s_w``.

Assumptions:
    - Gravity ``g = 9.81 m/s^2`` is exact and the Earth is flat; drag is
      quadratic in the speed relative to the air; the wind is horizontal and,
      in the simulator, constant in time and height.
    - The stated laws are teaching choices, not measured ones: ``x0 ~
      U(-25, 25)`` m, ``z_a ~ U(290, 310)`` m, ``s_p ~ U(55, 65)`` m/s,
      ``m ~ U(95, 105)`` kg, ``A ~ U(8, 12)`` kg/m (epistemic) and ``s_w ~
      N(5, 2^2)`` m/s (aleatory, the 10 m wind at drop time).
    - The synthetic truth is an assumption, not physics validated here: the
      exponent 0.1 of the power-law profile, a profile frozen during the
      fall, no gusts, the true mass and drag inside the stated ranges, and
      drop-to-drop scatter ``x0 ~ N(0, 8^2)``, ``z_a ~ N(300, 4^2)``,
      ``s_p ~ N(60, 2^2)``. The conclusions about which error structure
      works hold for this truth only.
    - The model-error terms assume residuals independent across drops with
      the predictive variance of each drop plus the measurement variance;
      their priors are ``N(0, 30^2)`` m for the bias and ``N(0, 15^2)`` s for
      the wind coefficient.
    - The order-statistic intervals of the limits of ``I_alpha`` treat the
      randomized Sobol points as independent draws; they are read as
      approximate.
    - Code verification on linear drag tests the solver, not the drag law:
      the same ``integrate_to_event`` code runs both models, and only the
      right-hand side differs.
    - The calibration campaign that fits the model error and the validation
      campaign that judges it are separate draws.

Equations:
    EQ-NUM-ODE-001: explicit Euler, Heun, and RK4 steps.
    EQ-NUM-ODE-002: event location (first step, linear, cubic Hermite).
    EQ-LAB-PARA-001: drag-free landing abscissa.
    EQ-LAB-PARA-002: linear-drag landing abscissa via the Lambert W function.
    EQ-LAB-PARA-003: power-law wind profile of the synthetic truth.
    EQ-LAB-PARA-004: Euler stability limit of the drag transient.
    EQ-LAB-PARA-005: verified error bound of a production step.
    EQ-LAB-PARA-006: terminal speed under quadratic drag.
    EQ-VV-001: observed order under uniform refinement.
    EQ-VV-002: Richardson extrapolation.
    EQ-VV-GCI-002: three-grid observed order.
    EQ-VV-GCI-004: fine-grid Grid Convergence Index.
    EQ-VV-LSQ-001: least-squares observed order from four grids.
    EQ-VV-UNUM-001: numerical-uncertainty statement.
    EQ-SAMP-RQMC-001: randomized Sobol propagation.
    EQ-PROP-QCI-001: order-statistic intervals of the interval limits.
    EQ-SENS-FIRST-001: first-order Sobol indices.
    EQ-SENS-TOTAL-001: total-effect Sobol indices and the ranking.
    EQ-VAL-AREA-001: area validation metric.
    EQ-VAL-UPOOL-001: u-pooling transform.
    EQ-VAL-UPOOL-002: pooled area metric.
    EQ-VAL-ASME-001: comparison error and validation uncertainty.
    EQ-VAL-DISC-002: moment estimate of the discrepancy variance.
    EQ-BAYES-LIN-POST-001: conjugate posterior of the bias terms.
    EQ-STAT-QUANT-001: sample quantiles of the predictive laws.

Diagnostics:
    - Code verification: the observed orders under linear drag must lie
      within about 0.3 of 1, 2, and 4; Heun and RK4 with Hermite location
      reproduce the drag-free landing to rounding; the landing speed equals
      ``sqrt(m g / A)`` and the drift relative to the air vanishes.
    - Solution verification: the error ratio under halving (2 for Euler, 16 for
      RK4), the observed order of successive triplets, the asymptotic-range
      warnings of ``verify_solution``, and whether the GCI band contains the
      RK4 reference.
    - Step choice: the verified bound must hold (bound at least the error
      against the reference) at every corner of the input box, and ``dt``
      must lie below the stability limit.
    - Propagation: order-statistic intervals of the two limits of
      ``I_alpha``, the shift of ``I_alpha`` when the same inputs are run
      with the reference solver, and Student-t intervals of the Sobol
      indices over eight randomizations.
    - Validation: the u-pooling Kolmogorov-Smirnov p-value and the largest
      area metric on the validation campaign, and the coverage of 2000
      truth landings with its binomial standard error.

Cost and error:
    - About 100 s in pure Python on a desktop computer; the run prints its
      wall-clock time. Euler at ``dt = 0.05 s`` takes about 650 steps per
      landing.
    - Code verification: 4 steps x 3 schemes under linear drag, 6 drag-free
      runs, and one reference run. Solution verification: 6 steps x 4
      schemes at the nominal inputs and 5 steps x 17 points for the step
      choice.
    - Propagation: 1024 Sobol points (plus 1024 reference runs); Sobol
      indices: 512 rows x 8 = 4096 runs in 8 randomizations.
    - Validation: 2 campaigns x 12 drops x 200 predictive runs, 3 winds x
      400 conditional runs, and 2000 + 3 x 300 truth landings (RK4 at
      ``dt = 0.1 s``, error below 0.03 m).
    - A coverage ``c`` from ``n`` truth landings has standard error
      ``sqrt(c (1 - c) / n)``: 0.0067 at ``c = 0.9`` and ``n = 2000``;
      differences smaller than two standard errors are not read.

References:
    REF-PERRIN-ATHENS-VVUQ: the parachuted-package example and its questions.
    REF-HAIRER-NORSETT-WANNER-1993: explicit Runge-Kutta methods, dense
        output, and event location.
    REF-HAIRER-WANNER-1996: stability limit of explicit methods.
    REF-DOWELL-JARRATT-1971: Illinois root finding inside the last step.
    REF-ROACHE-1998: grid convergence index and Richardson extrapolation.
    REF-CELIK-ETAL-2008: the three-grid procedure.
    REF-ECA-HOEKSTRA-2014: least-squares observed order.
    REF-CORLESS-ETAL-1996: the Lambert W function.
    REF-PETERSON-HENNESSEY-1978: power-law wind profiles.
    REF-HALLIDAY-RESNICK-WALKER-2014: drag-free projectile motion.
    REF-SALTELLI-2010: Sobol index estimators.
    REF-FERSON-OBERKAMPF-GINZBURG-2008: area metric and u-pooling.
    REF-ROY-OBERKAMPF-2011: numerical, parametric, and model-form
        uncertainty in prediction.
    REF-KENNEDY-OHAGAN-2001: model discrepancy.
    REF-GELMAN-ETAL-2013: conjugate Gaussian updating.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from probstat.distributions import Normal, Uniform
from probstat.foundations.random_streams import RandomStream
from probstat.inference import condition_linear_gaussian
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.numerics import EventSolution, integrate_to_event
from probstat.propagation import OutputDistribution, QuantileInterval
from probstat.sensitivity import IndexInterval, ModelSobolResult
from probstat.statistics import quantile
from probstat.uq import (
    GridConvergenceResult,
    LeastSquaresConvergence,
    QuantityOfInterest,
    SolutionVerification,
    UncertaintyNature,
    grid_convergence_index,
    least_squares_convergence,
    observed_order_of_accuracy,
    refinement_study,
    richardson_extrapolation,
    verify_solution,
)
from probstat.validation import (
    ValidationAssessment,
    ValidationExperiment,
    assess_predictions,
    estimate_discrepancy,
    predict_at_conditions,
)
from probstat.visualization import save_figure, theme
from probstat.workflow import UQStudy

# Root seed of every random draw: designs, campaigns, truth, and noise.
SEED = 2026
GRAVITY = 9.81
NOMINAL = {"x0": 0.0, "z_a": 300.0, "s_p": 60.0, "s_w": 5.0, "m": 100.0, "A": 10.0}
STEP_LADDER = (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625)
CANDIDATE_STEPS = (0.1, 0.05, 0.025)
REFERENCE_STEP = 0.001
CODE_LADDER = (0.2, 0.1, 0.05, 0.025)
LINEAR_DAMPING = 10.0
TOLERANCE_FRACTION = 0.05
ALPHA = 0.10
WIND_MEAN = 5.0
WIND_SD = 2.0
WIND_EXPONENT = 0.10
REFERENCE_HEIGHT = 10.0
TRUE_MASS = 100.0
TRUE_DRAG = 10.0
DROP_SCATTER = {"x0": (0.0, 8.0), "z_a": (300.0, 4.0), "s_p": (60.0, 2.0)}
MEASUREMENT_SD = 2.0
TRUTH_STEP = 0.1
PILOT_SIZE = 256
PROPAGATION_SIZE = 1024
SOBOL_SIZE = 512
CAMPAIGN_SIZE = 12
PREDICTIVE_SIZE = 200
CONDITIONAL_SIZE = 400
TRUTH_DROPS = 2000
CONDITIONAL_TRUTH_DROPS = 300
VALIDATION_TOLERANCE = 25.0
BIAS_PRIOR_SD = 30.0
SLOPE_PRIOR_SD = 15.0
QOI = QuantityOfInterest("y", "landing abscissa", "m")
SCHEMES = (
    ("Euler, linear", "euler", "linear"),
    ("Euler, first step below", "euler", "step"),
    ("RK4, first step below", "rk4", "step"),
    ("RK4, Hermite", "rk4", "hermite"),
)
CODE_SCHEMES = (
    ("Euler, linear", "euler", "linear"),
    ("Heun, Hermite", "heun", "hermite"),
    ("RK4, Hermite", "rk4", "hermite"),
)
STRUCTURES = ("parametric only", "Gaussian inflation", "constant bias", "bias ~ wind")
OUTPUT = Path(__file__).parent / "generated"


# ---------------------------------------------------------------------------
# The simulator and its closed-form checks
# ---------------------------------------------------------------------------


def power_law_wind(reference_wind: float, height: float) -> float:
    """Return the wind of the synthetic truth at a height.

    ``s_w(z) = s_w,10 (max(z, 1) / 10)^0.1``: the 10 m wind measured at the
    drop zone, growing with height; below 1 m the profile is frozen.

    Args:
        reference_wind: Wind at the reference height of 10 m (m/s).
        height: Height above ground (m).

    Returns:
        Horizontal wind speed (m/s).

    Equation:
        [EQ-LAB-PARA-003].

    References:
        [EQ-LAB-PARA-003], [REF-PETERSON-HENNESSEY-1978].

    """
    return reference_wind * (max(height, 1.0) / REFERENCE_HEIGHT) ** WIND_EXPONENT


def drop_rhs(
    parameters: Mapping[str, float],
    *,
    drag: str = "quadratic",
    wind_profile: Callable[[float, float], float] | None = None,
) -> Callable[[float, Sequence[float]], tuple[float, float, float, float]]:
    """Return the right-hand side of the package's equations of motion.

    With state ``(x, z, u, w)`` and relative air velocity
    ``v_r = (u - s_w(z), w)``: quadratic drag gives
    ``(u, w, -(A/m)|v_r| v_r,x, -g - (A/m)|v_r| v_r,z)``; linear drag
    (``A`` read as ``k``, kg/s) gives ``(u, w, -(k/m) v_r,x, -g - (k/m) w)``.

    Args:
        parameters: ``s_w``, ``m``, and ``A``.
        drag: ``"quadratic"`` (the course model) or ``"linear"``.
        wind_profile: ``(s_w, z) -> wind``; ``None`` keeps ``s_w`` constant.

    Returns:
        The function ``f(t, state)``.

    Raises:
        ValueError: If ``drag`` is unknown or the mass is not positive.

    References:
        [REF-PERRIN-ATHENS-VVUQ], [EQ-LAB-PARA-002].

    """
    wind = float(parameters["s_w"])
    mass = float(parameters["m"])
    if not mass > 0.0:
        raise ValueError(f"mass must be positive, got {mass!r}")
    rate = float(parameters["A"]) / mass
    if drag not in {"quadratic", "linear"}:
        raise ValueError(f"drag must be 'quadratic' or 'linear', got {drag!r}")
    quadratic = drag == "quadratic"
    gravity = GRAVITY

    def rhs(t: float, state: Sequence[float]) -> tuple[float, float, float, float]:
        """Return ``(x', z', u', w')`` at one state.

        References:
            [REF-PERRIN-ATHENS-VVUQ].

        """
        _, height, u, w = state
        air = wind if wind_profile is None else wind_profile(wind, height)
        relative = u - air
        factor = rate * math.hypot(relative, w) if quadratic else rate
        return (u, w, -factor * relative, -gravity - factor * w)

    return rhs


def simulate_drop(
    parameters: Mapping[str, float],
    *,
    dt: float,
    method: str = "euler",
    location: str = "linear",
    drag: str = "quadratic",
    wind_profile: Callable[[float, float], float] | None = None,
    record: bool = False,
) -> EventSolution:
    """Integrate one drop until the package reaches the ground.

    Args:
        parameters: ``x0``, ``z_a``, ``s_p``, ``s_w``, ``m``, ``A``.
        dt: Time step (s).
        method: ``"euler"``, ``"heun"``, or ``"rk4"``.
        location: ``"step"``, ``"linear"``, or ``"hermite"``.
        drag: ``"quadratic"`` or ``"linear"``.
        wind_profile: Height-dependent wind of the truth, or ``None``.
        record: Keep the trajectory.

    Returns:
        The landing event; ``state[0]`` is the landing abscissa.

    Equation:
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002].

    References:
        [EQ-NUM-ODE-001], [REF-PERRIN-ATHENS-VVUQ].

    """
    start = (
        float(parameters["x0"]),
        float(parameters["z_a"]),
        float(parameters["s_p"]),
        0.0,
    )
    return integrate_to_event(
        drop_rhs(parameters, drag=drag, wind_profile=wind_profile),
        start,
        lambda t, state: state[1],
        step=dt,
        method=method,
        location=location,
        record=record,
    )


def landing_abscissa(parameters: Mapping[str, float], **options: Any) -> float:
    """Return the simulated landing abscissa ``y_sim`` (m).

    Args:
        parameters: Inputs of :func:`simulate_drop`.
        **options: Keyword options of :func:`simulate_drop`.

    Returns:
        The horizontal position at the located landing.

    References:
        [EQ-NUM-ODE-002], [REF-PERRIN-ATHENS-VVUQ].

    """
    return simulate_drop(parameters, **options).state[0]


def drop_model(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Forward model ``y = y_sim(x; dt)``: Euler with linear ground crossing.

    Args:
        parameters: The six inputs and the numerical setting ``dt``.

    Returns:
        Mapping with the QoI ``y``.

    Equation:
        [EQ-NUM-ODE-001].

    References:
        [REF-PERRIN-ATHENS-VVUQ], [EQ-NUM-ODE-002].

    """
    return {"y": landing_abscissa(parameters, dt=parameters["dt"])}


def drag_free_landing(x0: float, height: float, plane_speed: float) -> float:
    """Return the exact landing abscissa without drag or wind.

    ``y = x0 + s_p sqrt(2 z_a / g)``.

    Args:
        x0: Dropping abscissa (m).
        height: Dropping height (m).
        plane_speed: Plane speed (m/s).

    Returns:
        Landing abscissa (m).

    Equation:
        [EQ-LAB-PARA-001].

    References:
        [EQ-LAB-PARA-001], [REF-HALLIDAY-RESNICK-WALKER-2014].

    """
    return x0 + plane_speed * math.sqrt(2.0 * height / GRAVITY)


def lambert_w0(x: float) -> float:
    """Return the principal branch ``W_0(x)`` on ``[-1/e, 0)``.

    Halley's iteration on ``w e^w - x`` from the branch-point series
    ``w = -1 + p - p^2/3 + 11 p^3/72``, ``p = sqrt(2 (e x + 1))``.

    Args:
        x: Argument in ``[-1/e, 0)``.

    Returns:
        ``w >= -1`` with ``w e^w = x``.

    Raises:
        ValueError: If ``x`` is outside ``[-1/e, 0)``.

    Equation:
        [EQ-LAB-PARA-002].

    References:
        [REF-CORLESS-ETAL-1996].

    """
    if not -1.0 / math.e <= x < 0.0:
        raise ValueError(f"x must lie in [-1/e, 0), got {x!r}")
    p = math.sqrt(max(0.0, 2.0 * (math.e * x + 1.0)))
    w = -1.0 + p - p * p / 3.0 + 11.0 * p**3 / 72.0
    for _ in range(60):
        exp_w = math.exp(w)
        residual = w * exp_w - x
        plus = w + 1.0
        if plus == 0.0 or residual == 0.0:
            break
        correction = residual / (exp_w * plus - (w + 2.0) * residual / (2.0 * plus))
        w -= correction
        if abs(correction) <= 4.0 * math.ulp(abs(w) + 1.0):
            break
    return w


def linear_drag_landing(
    x0: float,
    height: float,
    plane_speed: float,
    wind: float,
    mass: float,
    damping: float,
) -> tuple[float, float]:
    """Return the exact landing abscissa and time under linear drag.

    With ``m x'' = m g - k (x' - s_w)`` and ``tau = m / k``:
    ``X(t) = x0 + s_w t + (s_p - s_w) tau (1 - e^(-t/tau))``,
    ``Z(t) = z_a - g tau t + g tau^2 (1 - e^(-t/tau))``; ``Z(T) = 0`` gives
    ``T = tau (1 + c + W_0(-e^(-(1 + c))))`` with ``c = z_a / (g tau^2)``.

    Args:
        x0: Dropping abscissa (m).
        height: Dropping height ``z_a > 0`` (m).
        plane_speed: Plane speed (m/s).
        wind: Constant wind (m/s).
        mass: Mass (kg).
        damping: Linear drag coefficient ``k > 0`` (kg/s).

    Returns:
        ``(X(T), T)``.

    Raises:
        ValueError: If the height, mass, or damping is not positive.

    Equation:
        [EQ-LAB-PARA-002].

    References:
        [EQ-LAB-PARA-002], [REF-CORLESS-ETAL-1996].

    """
    if not (height > 0.0 and mass > 0.0 and damping > 0.0):
        raise ValueError("height, mass, and damping must be positive")
    tau = mass / damping
    c = height / (GRAVITY * tau * tau)
    landing = tau * (1.0 + c + lambert_w0(-math.exp(-(1.0 + c))))
    decay = -math.expm1(-landing / tau)
    return x0 + wind * landing + (plane_speed - wind) * tau * decay, landing


def terminal_speed(mass: float, drag: float) -> float:
    """Return the terminal speed of quadratic drag in still air relative to it.

    ``v_t = sqrt(m g / A)``: the speed at which drag balances weight. Once
    the horizontal relative velocity has decayed, the package falls at
    ``v_t`` relative to the air.

    Args:
        mass: Mass (kg).
        drag: Drag factor ``A`` (kg/m).

    Returns:
        Terminal speed (m/s).

    Equation:
        [EQ-LAB-PARA-006].

    References:
        [EQ-LAB-PARA-006], [REF-HALLIDAY-RESNICK-WALKER-2014].

    """
    return math.sqrt(mass * GRAVITY / drag)


def stability_limit(parameters: Mapping[str, float]) -> float:
    """Return the largest stable Euler step at the start of the fall.

    The drag Jacobian ``-(A/m)(|v| I + v v^T / |v|)`` has eigenvalues
    ``-(A/m)|v|`` and ``-2 (A/m)|v|``; explicit Euler is stable for
    ``dt <= m / (A |v_r|)``, and the relative speed ``|s_p - s_w|`` is
    largest at release.

    Args:
        parameters: ``s_p``, ``s_w``, ``m``, ``A``.

    Returns:
        The step limit (s).

    Equation:
        [EQ-LAB-PARA-004].

    References:
        [EQ-LAB-PARA-004], [REF-HAIRER-WANNER-1996, Section IV.2].

    """
    relative = abs(parameters["s_p"] - parameters["s_w"])
    return parameters["m"] / (parameters["A"] * relative)


def input_law() -> JointInput:
    """Return the stated law of the six inputs.

    References:
        [REF-PERRIN-ATHENS-VVUQ], [REF-ROY-OBERKAMPF-2011].

    """
    epistemic = UncertaintyNature.EPISTEMIC
    return JointInput.independent(
        RandomInput("x0", Uniform(-25.0, 25.0), "dropping abscissa", "m", epistemic),
        RandomInput("z_a", Uniform(290.0, 310.0), "dropping height", "m", epistemic),
        RandomInput("s_p", Uniform(55.0, 65.0), "plane speed", "m/s", epistemic),
        RandomInput("s_w", Normal(WIND_MEAN, WIND_SD), "10 m wind speed", "m/s"),
        RandomInput("m", Uniform(95.0, 105.0), "package mass", "kg", epistemic),
        RandomInput("A", Uniform(8.0, 12.0), "drag factor", "kg/m", epistemic),
    )


def probabilistic_model(dt: float, *, conditional: bool = False) -> ProbabilisticModel:
    """Return the simulator as a probabilistic model at a fixed step.

    Args:
        dt: Euler step (s).
        conditional: Drop the wind from the random inputs, so that it is
            set as an experimental condition.

    Returns:
        The model with QoI ``y``.

    References:
        [EQ-NUM-ODE-001], [REF-PERRIN-ATHENS-VVUQ].

    """
    law = input_law()
    if conditional:
        law = law.subset(("x0", "z_a", "s_p", "m", "A"))
    return ProbabilisticModel(
        law, drop_model, (QOI,), name="euler-drop", fixed={"dt": dt}
    )


# ---------------------------------------------------------------------------
# 1. Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeVerification:
    """Code verification: the solver against closed-form solutions.

    Args:
        linear_exact: Exact landing abscissa under linear drag (m).
        linear_errors: Error of each scheme of ``CODE_SCHEMES`` on
            ``CODE_LADDER`` under linear drag (m).
        linear_orders: Observed order ``log2(|e(dt)| / |e(dt/2)|)`` of each
            scheme for each halving.
        drag_free: ``(scheme, dt, error)`` without drag or wind, against the
            closed form (m).
        terminal: Vertical speed at landing of the quadratic-drag reference
            and the terminal speed ``sqrt(m g / A)`` (m/s).
        drift: Horizontal speed relative to the air at landing (m/s).

    References:
        [EQ-LAB-PARA-001], [EQ-LAB-PARA-002], [EQ-LAB-PARA-006],
        [REF-ROACHE-1998].

    """

    linear_exact: float
    linear_errors: dict[str, tuple[float, ...]]
    linear_orders: dict[str, tuple[float, ...]]
    drag_free: tuple[tuple[str, float, float], ...]
    terminal: tuple[float, float]
    drift: float


def run_code_verification() -> CodeVerification:
    """Compare the solver with exact solutions of simplified models.

    Linear drag ``m x'' = m g - k (x' - s_w)`` has the closed-form landing
    of :func:`linear_drag_landing`, so the error of each scheme is known
    exactly and its observed order must match the theoretical one (Euler 1,
    Heun 2, RK4 with Hermite location 4). Without drag the trajectory is a
    parabola, which the cubic Hermite interpolant reproduces to rounding.
    The quadratic-drag reference must land at the terminal speed.

    Returns:
        The code-verification evidence.

    References:
        [EQ-LAB-PARA-001], [EQ-LAB-PARA-002], [EQ-LAB-PARA-006],
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002], [REF-ROACHE-1998].

    """
    linear = {**NOMINAL, "A": LINEAR_DAMPING}
    exact, _ = linear_drag_landing(
        linear["x0"],
        linear["z_a"],
        linear["s_p"],
        linear["s_w"],
        linear["m"],
        LINEAR_DAMPING,
    )
    errors = {
        label: tuple(
            landing_abscissa(
                linear, dt=dt, method=method, location=location, drag="linear"
            )
            - exact
            for dt in CODE_LADDER
        )
        for label, method, location in CODE_SCHEMES
    }
    orders = {
        label: tuple(
            math.log2(abs(values[i]) / abs(values[i + 1]))
            for i in range(len(values) - 1)
        )
        for label, values in errors.items()
    }
    free = drag_free_landing(NOMINAL["x0"], NOMINAL["z_a"], NOMINAL["s_p"])
    vacuum = {**NOMINAL, "A": 0.0, "s_w": 0.0}
    drag_free = tuple(
        (
            label,
            dt,
            landing_abscissa(vacuum, dt=dt, method=method, location=location) - free,
        )
        for label, method, location in CODE_SCHEMES
        for dt in (0.1, 0.05)
    )
    landing = simulate_drop(
        NOMINAL, dt=REFERENCE_STEP, method="rk4", location="hermite"
    )
    _, _, u, w = landing.state
    return CodeVerification(
        exact,
        errors,
        orders,
        drag_free,
        (-w, terminal_speed(NOMINAL["m"], NOMINAL["A"])),
        u - NOMINAL["s_w"],
    )


@dataclass(frozen=True)
class Verification:
    """Solution verification at the nominal inputs.

    Args:
        reference: RK4 Hermite landing abscissa at ``REFERENCE_STEP``.
        reference_change: Change of the reference when its step is doubled.
        errors: Error against the reference of each scheme on
            ``STEP_LADDER``.
        euler_orders: Observed Euler order of each successive triplet.
        statement: Verification statement of the finest Euler triplet.
        gci: Three-grid GCI of the same triplet.
        least_squares: Least-squares order of the four finest Euler grids.
        richardson: Richardson extrapolation of the two finest Euler grids.

    References:
        [EQ-VV-001], [EQ-VV-GCI-004], [REF-ROACHE-1998].

    """

    reference: float
    reference_change: float
    errors: dict[str, tuple[float, ...]]
    euler_orders: tuple[float, ...]
    statement: SolutionVerification
    gci: GridConvergenceResult
    least_squares: LeastSquaresConvergence
    richardson: float


def run_verification() -> Verification:
    """Run the convergence study of the four schemes at the nominal inputs.

    References:
        [EQ-VV-001], [EQ-VV-002], [EQ-VV-GCI-002], [EQ-VV-LSQ-001],
        [REF-CELIK-ETAL-2008].

    """
    reference = landing_abscissa(
        NOMINAL, dt=REFERENCE_STEP, method="rk4", location="hermite"
    )
    coarser = landing_abscissa(
        NOMINAL, dt=2.0 * REFERENCE_STEP, method="rk4", location="hermite"
    )
    errors = {
        label: tuple(
            landing_abscissa(NOMINAL, dt=dt, method=method, location=location)
            - reference
            for dt in STEP_LADDER
        )
        for label, method, location in SCHEMES
    }
    euler = [reference + error for error in errors["Euler, linear"]]
    orders = tuple(
        observed_order_of_accuracy(euler[i], euler[i + 1], euler[i + 2])
        for i in range(len(euler) - 2)
    )
    study = refinement_study(
        drop_model,
        QOI,
        NOMINAL,
        STEP_LADDER[-3:],
        spacing_name="dt",
        model_name="euler-drop",
        theoretical_order=1.0,
    )
    gci = grid_convergence_index(STEP_LADDER[-3:], euler[-3:])
    least_squares = least_squares_convergence(
        STEP_LADDER[2:], euler[2:], theoretical_order=1.0
    )
    richardson = richardson_extrapolation(
        euler[-2], euler[-1], refinement_ratio=2.0, order=1.0
    )
    return Verification(
        reference,
        abs(coarser - reference),
        errors,
        orders,
        study.verification,
        gci,
        least_squares,
        richardson,
    )


@dataclass(frozen=True)
class StepCandidate:
    """Verified error of one candidate production step over the input box.

    Args:
        step: Candidate ``dt`` (s).
        bound: Largest verified bound ``B(dt)`` over the points (m).
        error: Largest error against the RK4 reference over the points (m).
        holds: Number of points where the bound covers the error.
        points: Number of points.
        stable: Whether ``dt`` is below the stability limit everywhere.

    References:
        [EQ-LAB-PARA-005].

    """

    step: float
    bound: float
    error: float
    holds: int
    points: int
    stable: bool


@dataclass(frozen=True)
class StepChoice:
    """Choice of the production step.

    Args:
        candidates: Every candidate with its verified bound.
        stability: Smallest Euler stability limit over the points (s).
        pilot_sd: Standard deviation of ``y`` from the pilot propagation (m).
        tolerance: ``TOLERANCE_FRACTION * pilot_sd`` (m).
        chosen: Largest admissible candidate (s).

    References:
        [EQ-LAB-PARA-004], [EQ-LAB-PARA-005].

    """

    candidates: tuple[StepCandidate, ...]
    stability: float
    pilot_sd: float
    tolerance: float
    chosen: float


def box_points() -> list[dict[str, float]]:
    """Return the nominal point and the 16 corners that stress the transient.

    The corners combine the ends of ``s_p``, ``A``, ``m`` and the 5% and
    95% quantiles of the wind; ``x0`` and ``z_a`` stay nominal because they
    only shift and stretch the trajectory.

    References:
        [EQ-LAB-PARA-004].

    """
    wind = Normal(WIND_MEAN, WIND_SD)
    points = [dict(NOMINAL)]
    for speed, drag, mass, level in product(
        (55.0, 65.0), (8.0, 12.0), (95.0, 105.0), (0.05, 0.95)
    ):
        points.append(
            {**NOMINAL, "s_p": speed, "A": drag, "m": mass, "s_w": wind.quantile(level)}
        )
    return points


def step_bound(
    parameters: Mapping[str, float], dt: float
) -> tuple[float, SolutionVerification]:
    """Return the verified error bound ``B(dt)`` of a production step.

    ``B(dt) = |y(dt) - y_ext| + U`` with ``y_ext`` and ``U`` from
    :func:`~probstat.uq.verify_solution` on the grids ``dt, dt/2, dt/4``.

    Args:
        parameters: Inputs of one drop.
        dt: Candidate step (s).

    Returns:
        The bound (m) and the verification statement.

    Equation:
        [EQ-LAB-PARA-005].

    References:
        [EQ-LAB-PARA-005], [EQ-VV-UNUM-001], [REF-ROACHE-1998].

    """
    spacings = (dt, dt / 2.0, dt / 4.0)
    values = [landing_abscissa(parameters, dt=h) for h in spacings]
    statement = verify_solution("y", spacings, values, theoretical_order=1.0)
    extrapolated = (
        statement.extrapolated if statement.extrapolated is not None else values[-1]
    )
    return abs(values[0] - extrapolated) + statement.expanded_uncertainty, statement


def choose_step(seed: RandomStream) -> StepChoice:
    """Choose the largest stable step whose verified bound meets the tolerance.

    Args:
        seed: Stream of the pilot propagation.

    Returns:
        The candidates, the stability limit, the pilot spread, and ``dt``.

    References:
        [EQ-LAB-PARA-004], [EQ-LAB-PARA-005], [REF-ROY-OBERKAMPF-2011].

    """
    pilot = ProbabilisticModel(
        input_law(),
        lambda p: {
            "y": landing_abscissa(p, dt=TRUTH_STEP, method="rk4", location="hermite")
        },
        (QOI,),
        name="rk4-pilot",
    )
    pilot_sd = (
        UQStudy(pilot, seed=seed)
        .propagate(PILOT_SIZE, design="lhs")["y"]
        .standard_deviation
    )
    tolerance = TOLERANCE_FRACTION * pilot_sd
    points = box_points()
    stability = min(stability_limit(point) for point in points)
    references = [
        landing_abscissa(point, dt=0.005, method="rk4", location="hermite")
        for point in points
    ]
    candidates = []
    for dt in CANDIDATE_STEPS:
        bounds, errors = [], []
        for point, reference in zip(points, references, strict=True):
            bound, _ = step_bound(point, dt)
            bounds.append(bound)
            errors.append(abs(landing_abscissa(point, dt=dt) - reference))
        holds = sum(b >= e for b, e in zip(bounds, errors, strict=True))
        candidates.append(
            StepCandidate(
                dt, max(bounds), max(errors), holds, len(points), dt < stability
            )
        )
    admissible = [c.step for c in candidates if c.stable and c.bound <= tolerance]
    chosen = max(admissible) if admissible else min(CANDIDATE_STEPS)
    return StepChoice(tuple(candidates), stability, pilot_sd, tolerance, chosen)


# ---------------------------------------------------------------------------
# 2. Propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Propagation:
    """Forward propagation of the stated laws at the chosen step.

    Args:
        output: Landing abscissa sample (Euler at the chosen step).
        winds: Wind of each sample, in sample order.
        interval: ``I_alpha`` (m).
        lower: Order-statistic interval of the lower limit.
        upper: Order-statistic interval of the upper limit.
        reference_interval: ``I_alpha`` of the same inputs run with RK4.
        sobol: Sobol indices with replicate intervals.
        evaluations: Simulator runs of the propagation and of the indices.

    References:
        [EQ-SAMP-RQMC-001], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    """

    output: OutputDistribution
    winds: tuple[float, ...]
    interval: tuple[float, float]
    lower: QuantileInterval
    upper: QuantileInterval
    reference_interval: tuple[float, float]
    sobol: ModelSobolResult
    evaluations: int


def sobol_index(
    sobol: ModelSobolResult, name: str
) -> tuple[IndexInterval, IndexInterval]:
    """Return the first-order and total-effect intervals of one input.

    References:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    """
    position = sobol.input_names.index(name)
    return sobol.first_order[position], sobol.total_effect[position]


def equal_tailed(values: Sequence[float]) -> tuple[float, float]:
    """Return the ``alpha/2`` and ``1 - alpha/2`` sample quantiles.

    References:
        [EQ-STAT-QUANT-001].

    """
    return quantile(values, ALPHA / 2.0), quantile(values, 1.0 - ALPHA / 2.0)


def run_propagation(dt: float, seed: RandomStream) -> Propagation:
    """Propagate the input law, compute ``I_alpha`` and the Sobol indices.

    Args:
        dt: Production step (s).
        seed: Stream of the study.

    Returns:
        The propagation results.

    References:
        [EQ-SAMP-RQMC-001], [EQ-PROP-QCI-001], [REF-SALTELLI-2010].

    """
    study = UQStudy(probabilistic_model(dt), seed=seed)
    propagated = study.propagate(PROPAGATION_SIZE, design="sobol")
    output = propagated["y"]
    columns = {name: propagated.input_column(name) for name in propagated.input_names}
    rows = [
        {name: columns[name][i] for name in columns} for i in range(PROPAGATION_SIZE)
    ]
    reference = [
        landing_abscissa(row, dt=dt, method="rk4", location="hermite") for row in rows
    ]
    sobol = study.sensitivity("y", SOBOL_SIZE, design="sobol", bootstrap=1)
    return Propagation(
        output,
        tuple(columns["s_w"]),
        equal_tailed(output.values),
        output.quantile_interval(ALPHA / 2.0),
        output.quantile_interval(1.0 - ALPHA / 2.0),
        equal_tailed(reference),
        sobol,
        propagated.model_evaluations + sobol.evaluations,
    )


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------


def draw_drop(rng: random.Random, wind: float | None = None) -> dict[str, float]:
    """Draw the conditions of one real drop.

    Args:
        rng: Random generator.
        wind: Fixed 10 m wind, or ``None`` to draw it from its law.

    Returns:
        The true inputs of the drop.

    References:
        [REF-PERRIN-ATHENS-VVUQ].

    """
    drop = {name: rng.gauss(mean, sd) for name, (mean, sd) in DROP_SCATTER.items()}
    drop["s_w"] = rng.gauss(WIND_MEAN, WIND_SD) if wind is None else wind
    drop["m"] = TRUE_MASS
    drop["A"] = TRUE_DRAG
    return drop


def true_landing(drop: Mapping[str, float]) -> float:
    """Return the landing abscissa of the synthetic truth (no measurement error).

    Args:
        drop: True inputs of the drop.

    Returns:
        Landing abscissa (m) under the power-law wind, RK4 Hermite at
        ``TRUTH_STEP``.

    References:
        [EQ-LAB-PARA-003], [EQ-NUM-ODE-002].

    """
    return landing_abscissa(
        drop,
        dt=TRUTH_STEP,
        method="rk4",
        location="hermite",
        wind_profile=power_law_wind,
    )


@dataclass(frozen=True)
class Campaign:
    """A measured campaign of single drops with model predictions.

    Args:
        experiments: One validation experiment per drop (condition ``s_w``).
        winds: Measured 10 m wind of each drop.
        predictions: ``y_sim`` samples at each drop's wind.

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """

    experiments: tuple[ValidationExperiment, ...]
    winds: tuple[float, ...]
    predictions: tuple[tuple[float, ...], ...]

    def residuals(self) -> tuple[list[float], list[float]]:
        """Return the residuals ``y_obs - mean(y_sim)`` and their variances.

        References:
            [EQ-VAL-ASME-001].

        """
        residuals, variances = [], []
        for experiment, sample in zip(self.experiments, self.predictions, strict=True):
            centre = sum(sample) / len(sample)
            spread = sum((y - centre) ** 2 for y in sample) / (len(sample) - 1)
            residuals.append(experiment.observations[0] - centre)
            variances.append(spread + MEASUREMENT_SD**2)
        return residuals, variances


def run_campaign(name: str, dt: float, root: RandomStream) -> Campaign:
    """Simulate one campaign of measured drops and predict each one.

    Args:
        name: ``"calibration"`` or ``"validation"``.
        dt: Production step (s).
        root: Root stream.

    Returns:
        The campaign.

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [EQ-UQ-001].

    """
    rng = root.child("campaign", name).generator()
    model = probabilistic_model(dt, conditional=True)
    experiments, winds, predictions = [], [], []
    for index in range(CAMPAIGN_SIZE):
        drop = draw_drop(rng)
        measured = true_landing(drop) + rng.gauss(0.0, MEASUREMENT_SD)
        winds.append(drop["s_w"])
        experiments.append(
            ValidationExperiment(
                f"{name}-{index + 1}",
                (measured,),
                {"s_w": drop["s_w"]},
                measurement_uncertainty=MEASUREMENT_SD,
                provenance="synthetic drop under a power-law wind profile",
            )
        )
        predictions.append(
            predict_at_conditions(
                model,
                "y",
                {"s_w": drop["s_w"]},
                PREDICTIVE_SIZE,
                design="lhs",
                seed=root.child("prediction", name, index),
            )
        )
    return Campaign(tuple(experiments), tuple(winds), tuple(predictions))


@dataclass(frozen=True)
class ErrorStructure:
    """One model-error structure fitted on the calibration campaign.

    ``delta(s_w) = mean_0 + mean_1 s_w + e`` with ``(mean_0, mean_1)`` drawn
    from a Gaussian posterior and ``e ~ N(0, inflation^2)``.

    Args:
        name: One of ``STRUCTURES``.
        offset: Posterior mean and standard deviation of the constant (m).
        slope: Posterior mean and standard deviation of the wind
            coefficient (s).
        inflation: Standard deviation of the zero-mean term (m).

    References:
        [EQ-BAYES-LIN-POST-001], [EQ-VAL-DISC-002], [REF-KENNEDY-OHAGAN-2001].

    """

    name: str
    offset: tuple[float, float] = (0.0, 0.0)
    slope: tuple[float, float] = (0.0, 0.0)
    inflation: float = 0.0

    def draw(self, wind: float, rng: random.Random) -> float:
        """Draw one model error at a wind speed.

        References:
            [EQ-BAYES-LIN-POST-001].

        """
        value = rng.gauss(*self.offset) + rng.gauss(*self.slope) * wind
        return value + (rng.gauss(0.0, self.inflation) if self.inflation else 0.0)

    def describe(self) -> str:
        """Return the fitted term as text.

        References:
            [EQ-BAYES-LIN-POST-001].

        """
        if self.inflation:
            return f"delta ~ N(0, {self.inflation:.1f}^2) m"
        if self.slope[1]:
            mean, sd = self.slope
            return f"delta = beta s_w, beta = {mean:.2f} +/- {sd:.2f} s"
        if self.offset[1]:
            return f"delta = b, b = {self.offset[0]:.1f} +/- {self.offset[1]:.1f} m"
        return "delta = 0"


def fit_structures(
    campaign: Campaign, seed: RandomStream
) -> tuple[ErrorStructure, ...]:
    """Fit the three model-error structures on the calibration campaign.

    The bias terms use the conjugate posterior of a Gaussian linear model
    whose rows are scaled by the standard deviation of each residual
    (predictive spread plus measurement error); the inflation is the
    library's moment estimate of the discrepancy variance.

    Args:
        campaign: Calibration campaign.
        seed: Stream of the assessment noise.

    Returns:
        The four structures of ``STRUCTURES``, the first one empty.

    References:
        [EQ-BAYES-LIN-POST-001], [EQ-VAL-DISC-002], [REF-GELMAN-ETAL-2013].

    """
    residuals, variances = campaign.residuals()
    scale = [math.sqrt(v) for v in variances]
    observations = [r / s for r, s in zip(residuals, scale, strict=True)]

    def posterior(rows: list[list[float]], prior_sd: float) -> tuple[float, float]:
        """Return the posterior mean and sd of one bias coefficient.

        References:
            [EQ-BAYES-LIN-POST-001].

        """
        law = condition_linear_gaussian(
            MultivariateNormal([0.0], [[prior_sd**2]]),
            rows,
            observations,
            observation_variance=1.0,
        )
        return float(law.mean[0]), math.sqrt(float(law.covariance[0][0]))

    assessment = assess_predictions(
        "y",
        campaign.experiments,
        campaign.predictions,
        tolerance=VALIDATION_TOLERANCE,
        seed=seed,
        model_name="euler-drop",
        unit="m",
    )
    inflation = estimate_discrepancy(assessment).discrepancy.standard_deviation
    offset = posterior([[1.0 / s] for s in scale], BIAS_PRIOR_SD)
    slope = posterior(
        [[w / s] for w, s in zip(campaign.winds, scale, strict=True)], SLOPE_PRIOR_SD
    )
    return (
        ErrorStructure(STRUCTURES[0]),
        ErrorStructure(STRUCTURES[1], inflation=inflation),
        ErrorStructure(STRUCTURES[2], offset=offset),
        ErrorStructure(STRUCTURES[3], slope=slope),
    )


@dataclass(frozen=True)
class StructureCheck:
    """How one model-error structure fares against independent evidence.

    Args:
        structure: The fitted structure.
        assessment: Validation assessment on the validation campaign.
        interval: ``I_alpha`` of ``y_sim + delta``.
        coverage: Fraction of truth landings inside ``I_alpha``.
        conditional: Coverage of the conditional interval at each test wind.

    References:
        [EQ-VAL-UPOOL-002], [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """

    structure: ErrorStructure
    assessment: ValidationAssessment
    interval: tuple[float, float]
    coverage: float
    conditional: tuple[float, ...]


@dataclass(frozen=True)
class Validation:
    """Validation of the simulator and of the model-error structures.

    Args:
        calibration: Campaign used to fit the model error.
        validation: Campaign used to judge it.
        checks: One check per structure.
        truth_count: Number of truth landings behind each coverage.
        truth_mean: Mean truth landing (m).
        test_winds: Winds of the conditional coverage check.
        reference_coverage: Coverage of the reference-solver ``I_alpha``.
        truth: Truth landings of the unconditional coverage check.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """

    calibration: Campaign
    validation: Campaign
    checks: tuple[StructureCheck, ...]
    truth_count: int
    truth_mean: float
    test_winds: tuple[float, ...]
    reference_coverage: float
    truth: tuple[float, ...]


def coverage(values: Sequence[float], interval: tuple[float, float]) -> float:
    """Return the fraction of values inside a closed interval.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """
    low, high = interval
    return sum(low <= value <= high for value in values) / len(values)


def run_validation(
    dt: float, propagation: Propagation, root: RandomStream
) -> Validation:
    """Compare the simulator with the synthetic truth and fit the model error.

    Args:
        dt: Production step (s).
        propagation: Unconditional propagation at ``dt``.
        root: Root stream.

    Returns:
        The validation results.

    References:
        [EQ-VAL-AREA-001], [EQ-VAL-UPOOL-001], [REF-ROY-OBERKAMPF-2011].

    """
    calibration = run_campaign("calibration", dt, root)
    validation = run_campaign("validation", dt, root)
    structures = fit_structures(calibration, root.child("assessment", "calibration"))
    truth_rng = root.child("truth").generator()
    truth = tuple(true_landing(draw_drop(truth_rng)) for _ in range(TRUTH_DROPS))
    unit = Normal(0.0, 1.0)
    test_winds = tuple(
        WIND_MEAN + WIND_SD * unit.quantile(level)
        for level in (1.0 / 6.0, 0.5, 5.0 / 6.0)
    )
    conditional_model = probabilistic_model(dt, conditional=True)
    conditional_predictions = [
        predict_at_conditions(
            conditional_model,
            "y",
            {"s_w": wind},
            CONDITIONAL_SIZE,
            design="lhs",
            seed=root.child("conditional", index),
        )
        for index, wind in enumerate(test_winds)
    ]
    conditional_truth = [
        [
            true_landing(draw_drop(truth_rng, wind))
            for _ in range(CONDITIONAL_TRUTH_DROPS)
        ]
        for wind in test_winds
    ]
    checks = []
    for index, structure in enumerate(structures):
        rng = root.child("model-error", index).generator()
        corrected = [
            [y + structure.draw(wind, rng) for y in sample]
            for sample, wind in zip(
                validation.predictions, validation.winds, strict=True
            )
        ]
        assessment = assess_predictions(
            "y",
            validation.experiments,
            corrected,
            tolerance=VALIDATION_TOLERANCE,
            seed=root.child("assessment", "validation", index),
            model_name=f"euler-drop, {structure.name}",
            unit="m",
        )
        unconditional = [
            y + structure.draw(wind, rng)
            for y, wind in zip(
                propagation.output.values, propagation.winds, strict=True
            )
        ]
        interval = equal_tailed(unconditional)
        conditional = tuple(
            coverage(
                observed, equal_tailed([y + structure.draw(wind, rng) for y in sample])
            )
            for sample, observed, wind in zip(
                conditional_predictions, conditional_truth, test_winds, strict=True
            )
        )
        checks.append(
            StructureCheck(
                structure, assessment, interval, coverage(truth, interval), conditional
            )
        )
    return Validation(
        calibration,
        validation,
        tuple(checks),
        len(truth),
        sum(truth) / len(truth),
        test_winds,
        coverage(truth, propagation.reference_interval),
        truth,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabResults:
    """Every result of the laboratory.

    Args:
        code: Code verification against closed forms.
        verification: Solution verification at the nominal inputs.
        step: Step choice.
        propagation: Forward propagation at the chosen step.
        validation: Validation and model-error structures.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """

    code: CodeVerification
    verification: Verification
    step: StepChoice
    propagation: Propagation
    validation: Validation


def run_analysis(
    seed: int = SEED, progress: Callable[[str], None] | None = None
) -> LabResults:
    """Run verification, propagation, and validation with one root seed.

    Args:
        seed: Root seed.
        progress: Called with the name of each stage before it starts.

    Returns:
        Every result of the laboratory.

    References:
        [REF-PERRIN-ATHENS-VVUQ], [REF-ROY-OBERKAMPF-2011].

    """
    report = progress if progress is not None else (lambda stage: None)
    root = RandomStream(seed)
    report("code verification")
    code = run_code_verification()
    report("solution verification")
    verification = run_verification()
    report("step choice")
    step = choose_step(root.child("pilot"))
    report("propagation and Sobol indices")
    propagation = run_propagation(step.chosen, root.child("study"))
    report("validation campaigns and truth landings")
    validation = run_validation(step.chosen, propagation, root)
    return LabResults(code, verification, step, propagation, validation)


def binomial_se(fraction: float, count: int) -> float:
    """Return the binomial standard error of an observed fraction.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """
    return math.sqrt(fraction * (1.0 - fraction) / count)


def print_report(results: LabResults) -> None:
    """Print the numbers of the laboratory.

    References:
        [REF-PERRIN-ATHENS-VVUQ].

    """
    c = results.code
    print("1a. Code verification against closed-form solutions")
    print(
        f"   linear drag (k = {LINEAR_DAMPING:g} kg/s), exact landing "
        f"{c.linear_exact:.6f} m (Lambert W)"
    )
    print("   dt [s]   " + "".join(f"{label:>16}" for label, _, _ in CODE_SCHEMES))
    for index, dt in enumerate(CODE_LADDER):
        row = "".join(
            f"{c.linear_errors[label][index]:>16.3e}" for label, _, _ in CODE_SCHEMES
        )
        print(f"   {dt:<9g}{row}")
    for label, _, _ in CODE_SCHEMES:
        orders = ", ".join(f"{p:.2f}" for p in c.linear_orders[label])
        print(f"   observed order, {label}: {orders}")
    for label, dt, error in c.drag_free:
        print(f"   no drag, {label}, dt = {dt:g} s: error {error:+.2e} m")
    speed, limit = c.terminal
    print(
        f"   quadratic drag: landing speed {speed:.6f} m/s against terminal speed "
        f"sqrt(m g / A) = {limit:.6f} m/s; drift relative to the air {c.drift:+.1e} m/s"
    )

    v = results.verification
    print("\n1b. Solution verification at the nominal inputs")
    print(
        f"   reference y_mod = {v.reference:.4f} m (RK4 Hermite, dt = "
        f"{REFERENCE_STEP:g} s; change on doubling dt {v.reference_change:.1e} m)"
    )
    header = "   dt [s]   " + "".join(f"{label:>25}" for label, _, _ in SCHEMES)
    print(header)
    for index, dt in enumerate(STEP_LADDER):
        row = "".join(f"{v.errors[label][index]:>25.3e}" for label, _, _ in SCHEMES)
        print(f"   {dt:<9g}{row}")
    for label, _, _ in SCHEMES:
        errors = v.errors[label]
        ratios = ", ".join(
            f"{errors[i] / errors[i + 1]:.2f}" for i in range(1, len(errors) - 1)
        )
        print(f"   error ratio under halving, {label}: {ratios}")
    print(
        "   observed Euler order of successive triplets: "
        + ", ".join(f"{p:.2f}" for p in v.euler_orders)
    )
    print(f"   {v.statement.summary()}")
    band = (
        v.statement.fine_value - v.statement.expanded_uncertainty,
        v.statement.fine_value + v.statement.expanded_uncertainty,
    )
    inside = band[0] <= v.reference <= band[1]
    print(
        f"   band [{band[0]:.4f}, {band[1]:.4f}] m "
        f"{'contains' if inside else 'misses'} the RK4 reference"
    )
    print(
        f"   GCI (dt = {STEP_LADDER[-1]:g} s): {v.gci.gci_fine:.3f} m; true error "
        f"{abs(v.errors['Euler, linear'][-1]):.3f} m; extrapolated "
        f"{v.gci.extrapolated:.4f} m (error {v.gci.extrapolated - v.reference:+.4f} m)"
    )
    print(
        f"   least squares on {v.least_squares.grid_count} grids: p = "
        f"{v.least_squares.order:.3f}, U = {v.least_squares.uncertainty:.3f} m"
    )
    print(
        f"   Richardson (p = 1) of the two finest grids: {v.richardson:.4f} m "
        f"(error {v.richardson - v.reference:+.4f} m, against "
        f"{v.errors['Euler, linear'][-1]:+.4f} m for the finest grid)"
    )

    s = results.step
    print("\n   Step choice over 17 points of the input box")
    print(
        f"   pilot sd of y = {s.pilot_sd:.1f} m; tolerance "
        f"{TOLERANCE_FRACTION:.0%} of it = {s.tolerance:.2f} m; Euler stability "
        f"limit {s.stability:.3f} s"
    )
    for c in s.candidates:
        print(
            f"   dt = {c.step:<6g} bound {c.bound:6.2f} m, error {c.error:6.2f} m, "
            f"bound holds at {c.holds}/{c.points}, stable {c.stable}"
        )
    print(f"   chosen dt = {s.chosen:g} s")

    p = results.propagation
    print("\n2. Propagation (randomized Sobol, Euler at the chosen step)")
    print(
        f"   mean {p.output.mean:.1f} m, sd {p.output.standard_deviation:.1f} m; "
        f"I_alpha = [{p.interval[0]:.1f}, {p.interval[1]:.1f}] m"
    )
    print(
        f"   95% order-statistic intervals: lower limit [{p.lower.lower:.1f}, "
        f"{p.lower.upper:.1f}], upper limit [{p.upper.lower:.1f}, {p.upper.upper:.1f}]"
    )
    print(
        f"   same inputs with RK4: [{p.reference_interval[0]:.1f}, "
        f"{p.reference_interval[1]:.1f}] m"
    )
    for name in p.sobol.ranking():
        first, total = sobol_index(p.sobol, name)
        print(
            f"   {name:>4}: S1 = {first.estimate:.3f} "
            f"[{first.lower:.3f}, {first.upper:.3f}], ST = {total.estimate:.3f} "
            f"[{total.lower:.3f}, {total.upper:.3f}]"
        )
    print(f"   simulator runs: {p.evaluations}")

    val = results.validation
    print("\n3. Validation against the synthetic truth")
    count = val.truth_count
    for check in val.checks:
        a = check.assessment
        largest = max(e.area for e in a.experiments)
        cond = ", ".join(f"{c:.3f}" for c in check.conditional)
        print(
            f"   {check.structure.name:<19} {check.structure.describe()}\n"
            f"      validation campaign: max area {largest:.1f} m, u-pooling area "
            f"{a.u_pooling.area:.3f}, KS p = {a.u_pooling.ks.p_value:.3g}, "
            f"{'consistent' if a.consistent else 'inconsistent'}\n"
            f"      I_alpha = [{check.interval[0]:.1f}, {check.interval[1]:.1f}] m, "
            f"coverage {check.coverage:.3f} +/- "
            f"{binomial_se(check.coverage, count):.3f}; "
            f"at winds {', '.join(f'{w:.2f}' for w in val.test_winds)} m/s: {cond}"
        )
    print(
        f"   reference-solver I_alpha covers {val.reference_coverage:.3f} of the "
        f"{count} truth landings"
    )
    law = input_law()
    stated = law.standard_deviations()[law.names.index("x0")]
    print(
        "   at a fixed wind the corrected intervals over-cover: the stated law of "
        f"x0 has sd {stated:.1f} m against the truth's drop scatter of "
        f"{DROP_SCATTER['x0'][1]:.1f} m"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _legible(colour: str, minimum: float = 4.5) -> str:
    """Mix a series colour toward the ink until text in it is legible.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = theme.active_palette()

    def luminance(rgb: Sequence[float]) -> float:
        """Return the relative luminance of an sRGB colour.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    canvas = luminance(to_rgb(palette.canvas))
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        weight = step / 20
        mixed = [(1 - weight) * b + weight * i for b, i in zip(base, ink, strict=True)]
        value = luminance(mixed)
        light, dark = max(value, canvas), min(value, canvas)
        if (light + 0.05) / (dark + 0.05) >= minimum:
            return str(to_hex(mixed))
    return palette.ink


def _label(
    ax: Any,
    x: float,
    y: float,
    text: str,
    colour: str,
    *,
    offset: tuple[float, float] = (6.0, 0.0),
    ha: str = "left",
    va: str = "center",
    contrast: float = 4.5,
) -> None:
    """Write a direct label in a legible tone of its series colour.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=_legible(colour, contrast),
        annotation_clip=False,
    )


def _panels(count: int = 1, *, height: float = 4.3) -> tuple[Any, list[Any]]:
    """Return a themed figure with ``count`` axes side by side.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    plt = theme.pyplot()
    figure, grid = plt.subplots(1, count, figsize=(7.2, height), squeeze=False)
    axes = list(grid[0])
    for ax in axes:
        theme.style_axes(ax)
    if count > 1:
        figure.subplots_adjust(wspace=0.34)
    return figure, axes


def _plain_log(axis: Any) -> None:
    """Write plain decimal numbers on a logarithmic axis.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def plain(value: float, _: float) -> str:
        """Return a positional decimal such as ``0.0001``.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return f"{value:.12f}".rstrip("0").rstrip(".")

    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


STRUCTURE_ROLES = {
    "parametric only": "model",
    "Gaussian inflation": "series-3",
    "constant bias": "series-4",
    "bias ~ wind": "series-5",
}
SCHEME_ROLES = {
    "Euler, linear": "model",
    "Euler, first step below": "context",
    "RK4, first step below": "series-4",
    "RK4, Hermite": "observed",
}


def draw_convergence(results: LabResults) -> Any:
    """Draw the error of each scheme against the step on log-log axes.

    References:
        [EQ-NUM-ODE-002], [EQ-VV-001], [REF-HAIRER-NORSETT-WANNER-1993].

    """
    v = results.verification
    figure, (ax,) = _panels(height=4.6)
    steps = list(STEP_LADDER[1:])
    for label, _, location in SCHEMES:
        errors = [abs(e) for e in v.errors[label][1:]]
        colour = theme.role_colour(SCHEME_ROLES[label])
        if location == "step":
            theme.line(ax, steps, errors, color=colour, dashes=theme.FITTED)
        else:
            theme.line(ax, steps, errors, color=colour)
            theme.points(ax, steps, errors, color=colour, size=22.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0045, 0.4)
    ax.set_ylim(5e-8, 30.0)
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1])
    _plain_log(ax.xaxis)
    _label(
        ax,
        steps[0],
        abs(v.errors["Euler, linear"][1]),
        "Euler, linear (slope 1)",
        theme.role_colour("model"),
        offset=(6.0, 8.0),
    )
    _label(
        ax,
        steps[0],
        abs(v.errors["Euler, first step below"][1]),
        "Euler, first step below",
        theme.role_colour("context"),
        offset=(6.0, -9.0),
    )
    _label(
        ax,
        steps[0],
        abs(v.errors["RK4, first step below"][1]),
        "RK4, first step below (erratic)",
        theme.role_colour("series-4"),
        offset=(6.0, -2.0),
    )
    _label(
        ax,
        steps[0],
        abs(v.errors["RK4, Hermite"][1]),
        "RK4, Hermite (slope 4)",
        theme.role_colour("observed"),
        offset=(6.0, 2.0),
    )
    theme.reference_line(ax, x=results.step.chosen)
    _label(
        ax,
        results.step.chosen,
        1e-7,
        f"chosen dt = {results.step.chosen:g} s",
        theme.role_colour("reference"),
        offset=(-4.0, 6.0),
        ha="right",
        va="bottom",
    )
    theme.axis_labels(
        ax, "Time step dt", "Error of the landing abscissa", x_unit="s", y_unit="m"
    )
    theme.title_block(
        ax,
        "Euler converges at order 1; without event interpolation RK4 loses its order 4",
        "Absolute error against RK4 with Hermite event location at dt = 0.001 s; "
        "nominal inputs",
    )
    return figure


def draw_code_verification(results: LabResults) -> Any:
    """Draw the error against the exact linear-drag landing on log-log axes.

    References:
        [EQ-LAB-PARA-002], [EQ-NUM-ODE-001], [REF-ROACHE-1998].

    """
    c = results.code
    figure, (ax,) = _panels(height=4.6)
    steps = list(CODE_LADDER)
    roles = {
        "Euler, linear": "model",
        "Heun, Hermite": "series-4",
        "RK4, Hermite": "observed",
    }
    for label, _, _ in CODE_SCHEMES:
        errors = [abs(e) for e in c.linear_errors[label]]
        colour = theme.role_colour(roles[label])
        theme.line(ax, steps, errors, color=colour)
        theme.points(ax, steps, errors, color=colour, size=22.0)
        order = c.linear_orders[label][-1]
        _label(ax, steps[0], errors[0], f"{label}: order {order:.1f}", colour)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.018, 0.25)
    ax.set_ylim(3e-11, 30.0)
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    _plain_log(ax.xaxis)
    ax.set_xticks(steps)
    theme.axis_labels(
        ax, "Time step dt", "Error against the exact landing", x_unit="s", y_unit="m"
    )
    theme.title_block(
        ax,
        "Code verification: each scheme converges at its theoretical order",
        f"Linear drag k = {LINEAR_DAMPING:g} kg/s, exact landing from the Lambert W "
        "function; labels give the order of the finest halving",
    )
    return figure


def draw_step_choice(results: LabResults) -> Any:
    """Draw the verified bound and the true error of each candidate step.

    References:
        [EQ-LAB-PARA-004], [EQ-LAB-PARA-005].

    """
    s = results.step
    figure, (ax,) = _panels()
    steps = [c.step for c in s.candidates]
    bounds = [c.bound for c in s.candidates]
    errors = [c.error for c in s.candidates]
    model, observed = theme.role_colour("model"), theme.role_colour("observed")
    theme.line(ax, steps, bounds, color=model)
    theme.points(ax, steps, bounds, color=model, size=34.0, alpha=0.75)
    theme.line(ax, steps, errors, color=observed, dashes=theme.FITTED)
    theme.points(ax, steps, errors, color=observed, size=26.0, alpha=0.75)
    theme.reference_line(ax, y=s.tolerance)
    ax.axvline(
        s.stability,
        color=theme.role_colour("context"),
        linewidth=theme.REFERENCE_WIDTH,
        linestyle=(0, (2.0, 2.0)),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.011, 0.17)
    ax.set_ylim(0.5, 10.0)
    _plain_log(ax.xaxis)
    _plain_log(ax.yaxis)
    ax.set_xticks(steps)
    ax.set_yticks([0.5, 1.0, 2.0, 5.0, 10.0])
    _label(
        ax,
        steps[0],
        bounds[0],
        "verified bound B(dt)",
        model,
        offset=(-8.0, 5.0),
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        steps[0],
        errors[0],
        "error against RK4",
        observed,
        offset=(-8.0, -5.0),
        ha="right",
        va="top",
    )
    _label(
        ax,
        0.012,
        s.tolerance,
        f"{TOLERANCE_FRACTION:.0%} of the parametric sd",
        theme.role_colour("reference"),
        offset=(0.0, 6.0),
        va="bottom",
    )
    _label(
        ax,
        s.stability,
        bounds[-1],
        "Euler stability limit",
        theme.role_colour("context"),
        offset=(-4.0, 0.0),
        ha="right",
    )
    theme.axis_labels(
        ax, "Time step dt", "Largest error over the input box", x_unit="s", y_unit="m"
    )
    theme.title_block(
        ax,
        f"The largest step whose verified bound stays below {s.tolerance:.1f} m "
        f"is dt = {s.chosen:g} s",
        "Worst case over the nominal point and 16 corners of the input box; "
        "bound from the grids dt, dt/2, dt/4",
    )
    return figure


def draw_propagation(results: LabResults) -> Any:
    """Draw the landing distribution with ``I_alpha`` and the truth landings.

    References:
        [EQ-STAT-QUANT-001], [EQ-LAB-PARA-003].

    """
    p, val = results.propagation, results.validation
    figure, (ax,) = _panels()
    edges = [-100.0 + 15.0 * k for k in range(37)]
    ax.hist(
        p.output.values,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=theme.LINE_WIDTH,
        color=theme.role_colour("model"),
    )
    ax.hist(
        val.truth,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=theme.CONTEXT_WIDTH,
        color=theme.role_colour("observed"),
        linestyle=(0, (4.0, 2.0)),
    )
    low, high = p.interval
    ax.axvspan(low, high, color=theme.role_colour("model"), alpha=0.08, zorder=0)
    ax.set_xlim(-100.0, 440.0)
    top = ax.get_ylim()[1]
    ax.set_ylim(0.0, top * 1.18)
    _label(
        ax,
        0.5 * (low + high),
        top * 1.08,
        f"I_alpha = [{low:.0f}, {high:.0f}] m (90%)",
        theme.role_colour("model"),
        offset=(0.0, 0.0),
        ha="center",
        contrast=5.2,
    )
    _label(
        ax,
        60.0,
        0.0025,
        "simulator y_sim",
        theme.role_colour("model"),
        offset=(-6.0, 0.0),
        ha="right",
    )
    _label(
        ax,
        330.0,
        0.0035,
        "truth (wind grows with height)",
        theme.role_colour("observed"),
        offset=(4.0, 0.0),
    )
    theme.axis_labels(ax, "Landing abscissa", theme.axis_title("density"), x_unit="m")
    coverage_raw = val.checks[0].coverage
    theme.title_block(
        ax,
        f"I_alpha holds {coverage_raw:.0%} of the true landings, not 90%",
        f"Densities of {PROPAGATION_SIZE} simulated landings (randomized Sobol, "
        f"Euler dt = {results.step.chosen:g} s) and {val.truth_count} landings "
        "of the synthetic truth",
    )
    return figure


def draw_sobol(results: LabResults) -> Any:
    """Draw first-order and total Sobol indices with their intervals.

    References:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [REF-SALTELLI-2010].

    """
    sobol = results.propagation.sobol
    names = list(reversed(sobol.ranking()))
    figure, (ax,) = _panels()
    first_style = theme.entity_style("first-order index")
    total_style = theme.entity_style("total-effect index")
    for position, name in enumerate(names):
        for offset, index, style in (
            (0.14, sobol_index(sobol, name)[0], first_style),
            (-0.14, sobol_index(sobol, name)[1], total_style),
        ):
            colour = theme.role_colour(style.role)
            if math.isfinite(index.lower):
                ax.hlines(
                    position + offset,
                    max(index.lower, -0.02),
                    index.upper,
                    color=colour,
                    linewidth=theme.CONTEXT_WIDTH,
                )
            theme.points(
                ax,
                [index.estimate],
                [position + offset],
                color=colour,
                marker=style.marker,
                size=34.0,
            )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(-0.02, 1.0)
    ax.grid(axis="y", visible=False)
    top = sobol.ranking()[0]
    _label(
        ax,
        sobol_index(sobol, top)[0].lower,
        len(names) - 1 + 0.14,
        "first order",
        theme.role_colour(first_style.role),
        offset=(-8.0, 0.0),
        ha="right",
    )
    _label(
        ax,
        sobol_index(sobol, top)[1].lower,
        len(names) - 1 - 0.14,
        "total effect",
        theme.role_colour(total_style.role),
        offset=(-8.0, 0.0),
        ha="right",
    )
    theme.axis_labels(ax, "Sobol index", "Input")
    theme.title_block(
        ax,
        f"The wind explains {sobol_index(sobol, top)[0].estimate:.0%} of the landing "
        "variance; plane speed, mass, and height barely matter",
        f"{SOBOL_SIZE} rows x 8 randomized Sobol replicates; bars: Student-t 95% "
        "intervals over the replicates",
    )
    return figure


def draw_residuals(results: LabResults) -> Any:
    """Draw measured minus predicted landings against the wind.

    References:
        [EQ-BAYES-LIN-POST-001], [REF-KENNEDY-OHAGAN-2001].

    """
    val = results.validation
    figure, (ax,) = _panels()
    for campaign, colour, marker, hollow in (
        (val.calibration, theme.role_colour("observed"), "o", None),
        (val.validation, theme.role_colour("context"), "s", True),
    ):
        residuals, variances = campaign.residuals()
        ax.errorbar(
            campaign.winds,
            residuals,
            yerr=[1.96 * math.sqrt(v) for v in variances],
            fmt="none",
            ecolor=colour,
            elinewidth=1.0,
            alpha=0.6,
        )
        ax.scatter(
            campaign.winds,
            residuals,
            s=30.0,
            marker=marker,
            facecolors=colour if hollow is None else "none",
            edgecolors=colour,
            linewidths=1.4,
            alpha=0.75,
            zorder=4.0,
        )
    grid = [0.0 + 0.1 * k for k in range(101)]
    offset = val.checks[2].structure.offset[0]
    slope = val.checks[3].structure.slope[0]
    theme.line(
        ax,
        grid,
        [offset] * len(grid),
        color=theme.role_colour(STRUCTURE_ROLES["constant bias"]),
        dashes=theme.FITTED,
    )
    theme.line(
        ax,
        grid,
        [slope * w for w in grid],
        color=theme.role_colour(STRUCTURE_ROLES["bias ~ wind"]),
    )
    theme.reference_line(ax, y=0.0)
    ax.set_xlim(0.0, 10.0)
    _label(
        ax,
        9.6,
        slope * 9.6,
        f"beta s_w, beta = {slope:.1f} s",
        theme.role_colour(STRUCTURE_ROLES["bias ~ wind"]),
        offset=(0.0, 10.0),
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        0.3,
        offset,
        f"constant b = {offset:.0f} m",
        theme.role_colour(STRUCTURE_ROLES["constant bias"]),
        offset=(0.0, 8.0),
        va="bottom",
    )
    theme.axis_labels(
        ax,
        "Measured 10 m wind s_w",
        "Measured minus predicted landing",
        x_unit="m/s",
        y_unit="m",
    )
    theme.title_block(
        ax,
        "The model error grows with the wind: the simulator misses the stronger "
        "wind aloft",
        f"Dots: {CAMPAIGN_SIZE} calibration drops; open squares: {CAMPAIGN_SIZE} "
        "validation drops; bars: +/- 1.96 predictive sd; lines fitted on the "
        "calibration drops",
    )
    return figure


def draw_coverage(results: LabResults) -> Any:
    """Draw the coverage of each model-error structure with standard errors.

    References:
        [EQ-VAL-UPOOL-002], [REF-ROY-OBERKAMPF-2011].

    """
    val = results.validation
    figure, (ax,) = _panels(height=4.6)
    names = list(STRUCTURES)
    columns = ["all winds"] + [f"s_w = {w:.1f} m/s" for w in val.test_winds]
    roles = tuple(STRUCTURE_ROLES[name] for name in names)
    markers = ("o", "s", "^", "D")
    for row, check in enumerate(val.checks):
        values = [check.coverage, *check.conditional]
        counts = [val.truth_count] + [CONDITIONAL_TRUTH_DROPS] * len(val.test_winds)
        colour = theme.role_colour(roles[row])
        xs = [k + (row - 1.5) * 0.16 for k in range(len(values))]
        for x, c, n in zip(xs, values, counts, strict=True):
            ax.vlines(
                x,
                c - 1.96 * binomial_se(c, n),
                c + 1.96 * binomial_se(c, n),
                color=colour,
                linewidth=theme.CONTEXT_WIDTH,
            )
        theme.points(ax, xs, values, color=colour, marker=markers[row], size=32.0)
    theme.reference_line(ax, y=1.0 - ALPHA)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="x", visible=False)
    ys = [0.06, 0.14, 0.22, 0.30]
    for row, name in enumerate(names):
        _label(
            ax,
            len(columns) - 0.55,
            ys[row],
            name,
            theme.role_colour(roles[row]),
            offset=(0.0, 0.0),
            ha="right",
        )
    theme.axis_labels(ax, "Truth landings", theme.axis_title("empirical_coverage"))
    best = val.checks[3]
    theme.title_block(
        ax,
        "Only the wind-proportional bias reaches 90% coverage overall and at "
        f"every wind ({best.coverage:.3f} overall, at least "
        f"{min(best.conditional):.3f} per wind)",
        f"Coverage of 90% intervals: {val.truth_count} truth landings for all "
        f"winds, {CONDITIONAL_TRUTH_DROPS} per fixed wind; bars: +/- 1.96 "
        "binomial standard errors",
    )
    return figure


FIGURES: tuple[tuple[str, Callable[[LabResults], Any]], ...] = (
    ("parachute-code-verification.png", draw_code_verification),
    ("parachute-convergence.png", draw_convergence),
    ("parachute-step-choice.png", draw_step_choice),
    ("parachute-propagation.png", draw_propagation),
    ("parachute-sobol.png", draw_sobol),
    ("parachute-model-error.png", draw_residuals),
    ("parachute-coverage.png", draw_coverage),
)


def write_figures(results: LabResults, output: Path) -> list[Path]:
    """Draw and save every figure of the laboratory.

    Args:
        results: Output of :func:`run_analysis`.
        output: Destination folder.

    Returns:
        The written paths (empty without Matplotlib).

    References:
        [REF-MATPLOTLIB].

    """
    if not theme.matplotlib_available():
        return []
    plt = theme.pyplot()
    written = []
    with theme.chart_theme(theme.active_palette()):
        for name, draw in FIGURES:
            figure = draw(results)
            written.append(save_figure(figure, output / name))
            plt.close(figure)
    return written


def main() -> None:
    """Run the laboratory, print its report, and save its figures.

    References:
        [REF-PERRIN-ATHENS-VVUQ], [REF-MATPLOTLIB].

    """
    started = time.perf_counter()
    print(
        "Verifying, propagating, and validating the parachuted package "
        "(one to two minutes)."
    )

    def stage(name: str) -> None:
        """Print the stage that starts and the time elapsed so far.

        References:
            [REF-PERRIN-ATHENS-VVUQ].

        """
        print(f"  [{time.perf_counter() - started:5.0f} s] {name}", flush=True)

    results = run_analysis(progress=stage)
    print()
    print_report(results)
    written = write_figures(results, OUTPUT)
    raw, best = results.validation.checks[0], results.validation.checks[3]
    print(
        f"\nResult: dt = {results.step.chosen:g} s makes numerical error negligible; "
        f"I_alpha covers {raw.coverage:.1%} of true landings without a model-error "
        f"term and {best.coverage:.1%} with a wind-proportional bias."
    )
    print(f"Figures: {OUTPUT}")
    print(f"Wrote {len(written)} figures in {time.perf_counter() - started:.0f} s.")


if __name__ == "__main__":
    main()
