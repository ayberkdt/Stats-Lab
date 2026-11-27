"""Hot-fire test campaign laboratory: calibration, discrepancy and the next test.

Run with ``PYTHONPATH=src python examples/hotfire_test_campaign_lab.py``. The
lab follows an engine development campaign on a synthetic engine from the
simulator runs to the choice of the next test: (a) a synthetic "real" engine
whose combustion efficiency depends on mixture ratio and chamber pressure
through a term the simulator lacks (model-form error), (b) a Gaussian-process
emulator of the simulator over (mixture ratio, chamber pressure, combustion
efficiency) validated on held-out runs with Bastos-O'Hagan diagnostics,
(c) Kennedy-O'Hagan calibration of the efficiency from eight synthetic
hot-fire measurements of vacuum specific impulse, with and without a model
discrepancy, with the predictive of an untested operating point split into its
variance parts and the coverage of both calibrations measured on held-out
operating points, (d) calibration of the Bartz heat-transfer multiplier from
the wall-temperature profile of the regeneratively cooled nozzle (a field
output) compared with a single-thermocouple calibration, and (e) the choice of
the next test by expected information gain on the efficiency, greedy against
exhaustive on a small candidate set. Every number is printed; with Matplotlib
installed two figures are written to ``examples/generated/``.

EVERY CASE IS SYNTHETIC. The propellant table, the engine, the "real" engine
and its efficiency law, the nozzle, the noise and all measurements are
invented demonstration values; none is data from a real engine, test stand or
vehicle, and the physical models are textbook idealizations.

Definitions:
    ``Isp_vac``: vacuum specific impulse ``CF_vac c* / g0`` of an ideal
        frozen-flow conical nozzle, in s, with ``c* = eta c*_ideal``; the
        simulator ``Isp_vac(OF, p_c, eta)`` is therefore exactly linear in the
        combustion efficiency ``eta`` and does not depend on ``p_c``.
    ``OF`` and ``p_c``: mixture ratio and chamber pressure (MPa), the control
        inputs of the simulator and of the tests.
    ``eta``: the calibration parameter, the combustion (``c*``) efficiency in
        ``[0.88, 1]``.
    "real" engine: ``Isp_vac^real = Isp_vac(OF, p_c, eta_real(OF, p_c))`` with
        ``eta_real = 0.975 - 0.045 (OF - 2.4)^2 + 0.005 (p_c - 7)``, a
        quadratic mixture-ratio loss and a pressure term that no constant
        ``eta`` can reproduce: the model-form error.
    measurement: ``Isp_vac^real`` plus independent Gaussian noise of standard
        deviation ``ISP_NOISE`` seconds (thrust and flow rate reduced to one
        number).
    Kennedy-O'Hagan calibration: ``zeta(x) = eta_sim(x, theta) + delta(x)``
        with a GP discrepancy ``delta``; "without discrepancy" sets
        ``delta = 0``.
    coverage: fraction of held-out operating points whose real ``Isp_vac``
        lies inside the central 95 % predictive interval of the latent
        reality (no measurement noise).
    wall temperature profile: gas-side wall temperature at 24 stations of a
        regeneratively cooled conical nozzle, in K, a function of chamber
        pressure and the Bartz multiplier ``b``.
    expected information gain (EIG): ``E[log p(y | eta, d) - log p(y | d)]`` of
        a design ``d`` of additional tests, in nats.

Assumptions:
    - The simulator is the textbook ideal-rocket model with a synthetic
      mixture-ratio table; it is not validated against any hardware.
    - The real engine differs from the simulator by the efficiency law above,
      which the calibration does not know: the discrepancy GP is a generic
      smooth prior (Matern 5/2, log-normal amplitude and length-scale priors),
      not the true law.
    - The measurement noise is Gaussian with the known sd ``ISP_NOISE``; the
      discrepancy and ``eta`` are partly confounded (Brynjarsdottir and
      O'Hagan), so the posterior of ``eta`` with a discrepancy is wide and is
      not an estimate of any single "true" efficiency.
    - The emulator is modular: its hyperparameters are fitted to the simulator
      runs alone and then held fixed; the extrapolation to a mixture ratio
      outside the tested range relies on the discrepancy prior.
    - Part (d) generates the field data from the same nozzle model with a
      different Bartz multiplier and Gaussian noise, so it has no model-form
      error and its posterior is calibrated by construction; its point is the
      information content of the profile, not the realism of the model.
    - The EIG of part (e) models a new measurement as ``eta Isp_vac(x, 1)``
      plus independent noise whose variance adds the discrepancy variance at
      ``x``; repeated tests at one point are treated as independent although
      a real discrepancy is systematic, which overstates repeated tests. The
      posterior of ``eta`` is approximated by a normal prior.

Equations:
    EQ-AERO-LRE-005: specific impulse ``CF c* / g0`` with ``c* = eta c*_ideal``.
    EQ-AERO-LRE-008: mixture-ratio table of the chamber gas.
    EQ-AERO-COOL-004: wall network of the cooled nozzle.
    EQ-AERO-COOL-006: coolant marching of the cooled nozzle.
    EQ-SUR-GPB-003: standardized emulator errors.
    EQ-SUR-GPB-004: Mahalanobis distance of the emulator.
    EQ-VAL-KOHEM-004: posterior of the calibration parameter and discrepancy.
    EQ-VAL-KOHEM-005: predictive of reality at a new control input.
    EQ-VAL-KOHEM-006: exact variance decomposition of the predictive.
    EQ-VAL-KOHEM-007: identifiability and the no-discrepancy comparison.
    EQ-VAL-FUNCEM-002: Gaussian emulator of a field-valued simulator.
    EQ-VAL-FUNCEM-005: posterior of the field calibration.
    EQ-VAL-FUNCEM-008: contraction and comparison with the scalar calibration.
    EQ-OED-EIG-LIN-001: closed-form EIG of a linear-Gaussian experiment.
    EQ-DES-NOED-001: nested Monte Carlo EIG.
    EQ-DES-NOED-003: Laplace EIG.
    EQ-DES-NOED-004: exhaustive and greedy design search.

Diagnostics:
    - The Isp emulator is validated on held-out simulator runs (Mahalanobis
      distance against its expectation, 95 % coverage, flags); the field
      emulator is validated in the reduced and in the full field.
    - Every sampler reports split R-hat, bulk effective sample size and its
      stop reason; a calibration that does not converge is printed as such.
    - The five predictive variance parts are checked to add up to the total.
    - The coverage of both calibrations is measured on held-out operating
      points of the "real" engine, and the lab states from that measurement
      which one is overconfident.
    - The single-thermocouple calibrations use the same experiments, noise
      and prior as the field calibration; the ratio of posterior standard
      deviations is printed.
    - The Laplace EIG is compared with its closed form and with nested Monte
      Carlo; greedy and exhaustive designs are compared on the same random
      numbers.

Cost and error:
    - Simulator: 40 training and 15 held-out runs of a closed-form model.
    - Calibration: slice sampling with 2 chains of 250 kept draws after 150
      warm-up iterations per model; predictions use up to 60 posterior draws.
      Coverage is a binomial fraction of 30 operating points, with the
      binomial standard error printed (the points are dependent, so it is
      optimistic); the numbers are one seed and are not averaged over seeds.
    - Cooling: 24 field-emulator runs and 8 held-out runs; 2 chains of 200
      kept draws; the scalar comparisons add one emulator and sampler each.
    - EIG: 300 prior draws for Laplace (its standard error is zero because
      the Jacobian of a linear model does not depend on the draw), 400 outer
      and 40 inner draws for nested Monte Carlo per candidate (biased upward
      by the Jensen gap of the inner log-mean, which is O(1/40); standard
      errors printed), and 200 outer draws per design in the nested
      exhaustive and greedy searches over 15 designs.
    - The whole lab runs in well under two minutes; its wall time is printed.

References:
    REF-SUTTON-BIBLARZ-2017: ideal-rocket performance relations.
    REF-BARTZ-1957: convective heat-transfer coefficient of nozzles.
    REF-KENNEDY-OHAGAN-2001: Bayesian calibration of computer models.
    REF-BRYNJARSDOTTIR-OHAGAN-2014: confounding of parameter and discrepancy.
    REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008: calibration with functional
        output.
    REF-BASTOS-OHAGAN-2009: emulator diagnostics.
    REF-CHALONER-VERDINELLI-1995: Bayesian experimental design.
    REF-HUAN-MARZOUK-2013: nonlinear Bayesian design by expected information
        gain.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.aerospace.liquid_engine import (
    MixtureRatioTable,
    ideal_rocket_performance,
)
from probstat.aerospace.nozzle_cooling import (
    ChannelGeometry,
    CoolantProperties,
    GasProperties,
    conical_contour,
    solve_regenerative_cooling,
    station_profile,
)
from probstat.design.nonlinear_optimal import (
    NonlinearExperimentModel,
    exhaustive_nonlinear_design,
    greedy_nonlinear_design,
    laplace_eig,
    nested_monte_carlo_eig,
)
from probstat.foundations.random_streams import resolve_stream
from probstat.sampling import latin_hypercube
from probstat.surrogates.gaussian_process import (
    GaussianProcessRegularization,
    fit_gaussian_process,
)
from probstat.surrogates.gp_bayes import validate_gp_emulator
from probstat.validation.functional_calibration import (
    FieldEmulator,
    FunctionalCalibration,
)
from probstat.validation.koh_emulator import KOHEmulatorCalibration

# Root seed; every stage below derives its own seeds from it.
SEED = 2027
OUTPUT = Path(__file__).parent / "generated"

ISP_NOISE = 0.5  # s, sd of the measured vacuum specific impulse
THROAT_AREA = 0.0314  # m^2
EXPANSION_RATIO = 40.0
HALF_ANGLE = 0.2618  # rad, 15 degrees
LOWER = (2.0, 5.0, 0.88)  # (OF, p_c in MPa, eta) of the simulator design
UPPER = (3.2, 9.0, 1.0)
MIXTURE_TABLE = MixtureRatioTable(
    (1.8, 2.2, 2.6, 3.0, 3.4),
    (3450.0, 3560.0, 3600.0, 3580.0, 3500.0),
    (1.24, 1.23, 1.22, 1.21, 1.20),
    (21.0, 22.5, 24.0, 25.5, 27.0),
    label="synthetic kerosene-like table (invented values)",
)
# (OF, p_c in MPa) of the eight hot-fire tests: four at 7 MPa and four elsewhere.
TESTS = (
    (2.1, 7.0),
    (2.4, 7.0),
    (2.7, 7.0),
    (3.0, 7.0),
    (2.25, 5.5),
    (2.55, 8.5),
    (2.85, 5.5),
    (2.4, 8.5),
)
UNTESTED = (3.15, 7.0)  # untested operating point, beyond the tested OF range
CANDIDATES = (
    (2.0, 5.0),
    (2.0, 9.0),
    (2.6, 7.0),
    (3.15, 5.0),
    (3.15, 7.0),
    (3.15, 9.0),
)  # candidate next tests

COOLING_OF = 2.6  # mixture ratio of the cooled-nozzle test point
COOLING_BARTZ_TRUTH = 1.18  # synthetic generating Bartz multiplier
COOLING_NOISE = 8.0  # K, thermocouple noise sd
COOLING_PRESSURES = (5.0, 6.5, 8.0)  # MPa of the three field experiments
COOLING_BOUNDS = (0.8, 1.5)  # uniform prior box of the Bartz multiplier


@dataclass(frozen=True)
class Budget:
    """Sample sizes of one run of the lab.

    Attributes:
        sim_runs: Training runs of the Isp emulator.
        sim_holdout: Held-out runs of the Isp emulator.
        draws: Kept slice-sampler draws per chain of the Isp calibrations.
        warmup: Warm-up iterations per chain of the Isp calibrations.
        predict_draws: Posterior draws used by predictions.
        coverage_points: Held-out operating points for the coverage.
        cooling_runs: Training runs of the field emulator.
        cooling_holdout: Held-out runs of the field emulator.
        cooling_draws: Kept draws per chain of the field calibration.
        cooling_warmup: Warm-up iterations per chain of the field calibration.
        laplace_samples: Prior draws of the Laplace EIG.
        nested_outer: Outer draws of the nested Monte Carlo EIG.
        nested_inner: Inner draws of the nested Monte Carlo EIG.

    References:
        [EQ-DES-NOED-003]. No equation is evaluated.

    """

    sim_runs: int = 40
    sim_holdout: int = 15
    draws: int = 250
    warmup: int = 150
    predict_draws: int = 60
    coverage_points: int = 30
    cooling_runs: int = 24
    cooling_holdout: int = 8
    cooling_draws: int = 200
    cooling_warmup: int = 100
    laplace_samples: int = 300
    nested_outer: int = 400
    nested_inner: int = 40


FULL = Budget()
QUICK = Budget(
    sim_runs=30,
    sim_holdout=10,
    draws=60,
    warmup=40,
    predict_draws=20,
    coverage_points=12,
    cooling_runs=16,
    cooling_holdout=5,
    cooling_draws=80,
    cooling_warmup=40,
    laplace_samples=100,
    nested_outer=150,
    nested_inner=20,
)


def headline(text: str) -> None:
    """Print a section heading.

    Args:
        text: Heading.

    References:
        [EQ-AERO-LRE-005]. No equation is evaluated.

    """
    print(f"\n== {text} ==")


# ---------------------------------------------------------------------------
# (a) the simulator and the synthetic real engine
# ---------------------------------------------------------------------------


def simulator_isp(mixture_ratio: float, pressure_mpa: float, eta: float) -> float:
    """Return the simulated vacuum specific impulse.

    Args:
        mixture_ratio: Oxidizer-to-fuel ratio ``OF`` inside the table range.
        pressure_mpa: Chamber pressure in MPa.
        eta: Combustion (``c*``) efficiency in ``(0, 1]``.

    Returns:
        ``Isp_vac`` in s; linear in ``eta``.

    References:
        [EQ-AERO-LRE-005], [EQ-AERO-LRE-008], [REF-SUTTON-BIBLARZ-2017].

    """
    state = MIXTURE_TABLE.at(mixture_ratio)
    return ideal_rocket_performance(
        gamma=state.gamma,
        molecular_weight=state.molecular_weight,
        chamber_temperature=state.chamber_temperature,
        chamber_pressure=pressure_mpa * 1.0e6,
        throat_area=THROAT_AREA,
        expansion_ratio=EXPANSION_RATIO,
        combustion_efficiency=eta,
        divergence_half_angle=HALF_ANGLE,
    ).vacuum_specific_impulse


def real_efficiency(mixture_ratio: float, pressure_mpa: float) -> float:
    """Return the efficiency of the synthetic real engine.

    Args:
        mixture_ratio: ``OF``.
        pressure_mpa: Chamber pressure in MPa.

    Returns:
        ``0.975 - 0.045 (OF - 2.4)^2 + 0.005 (p_c - 7)``, the term the
        simulator (a constant ``eta``) lacks.

    References:
        [EQ-AERO-LRE-005]. The law is invented for this lab.

    """
    return 0.975 - 0.045 * (mixture_ratio - 2.4) ** 2 + 0.005 * (pressure_mpa - 7.0)


def real_isp(mixture_ratio: float, pressure_mpa: float) -> float:
    """Return the vacuum specific impulse of the synthetic real engine.

    Args:
        mixture_ratio: ``OF``.
        pressure_mpa: Chamber pressure in MPa.

    Returns:
        ``Isp_vac`` in s of the simulator evaluated at the real efficiency.

    References:
        [EQ-AERO-LRE-005].

    """
    return simulator_isp(
        mixture_ratio, pressure_mpa, real_efficiency(mixture_ratio, pressure_mpa)
    )


def unit_to_box(point: tuple[float, ...]) -> tuple[float, ...]:
    """Map a point of the unit cube to the simulator design box.

    Args:
        point: Coordinates in ``(0, 1)^3``.

    Returns:
        ``(OF, p_c, eta)``.

    References:
        [EQ-AERO-LRE-005]. No equation is evaluated.

    """
    return tuple(a + (b - a) * u for u, a, b in zip(point, LOWER, UPPER, strict=True))


def truth_part() -> dict[str, Any]:
    """Describe the synthetic engine and its model-form error.

    Returns:
        The model-form error of the real engine against the best constant
        efficiency, on a grid of the operating region.

    References:
        [EQ-AERO-LRE-005], [REF-SUTTON-BIBLARZ-2017].

    """
    headline("(a) the synthetic engine: simulator and the term it lacks")
    print(
        "real efficiency eta_real = 0.975 - 0.045 (OF - 2.4)^2 + 0.005 (p_c - 7); "
        "the simulator has one constant eta (model-form error by construction)"
    )
    grid = [(of, pc) for of in (2.0, 2.4, 2.8, 3.2) for pc in (5.0, 7.0, 9.0)]
    best = statistics.fmean(real_efficiency(of, pc) for of, pc in grid)
    errors = [real_isp(of, pc) - simulator_isp(of, pc, best) for of, pc in grid]
    print(
        f"mean real efficiency on the 12-point grid {best:.4f} (taken as the "
        f"constant eta of the simulator); real minus simulated Isp_vac: "
        f"min {min(errors):+.2f} s, "
        f"max {max(errors):+.2f} s (measurement noise sd {ISP_NOISE} s)"
    )
    print(
        f"Isp_vac(OF=2.6, p_c=7 MPa, eta=1) = {simulator_isp(2.6, 7.0, 1.0):.3f} s; "
        f"real engine at the untested point {UNTESTED}: "
        f"{real_isp(*UNTESTED):.3f} s"
    )
    return {"best_eta": best, "errors": errors}


# ---------------------------------------------------------------------------
# (b) the simulator design and the emulator
# ---------------------------------------------------------------------------


def emulator_part(budget: Budget) -> dict[str, Any]:
    """Run the simulator design, fit the GP emulator and validate it.

    Args:
        budget: Sample sizes.

    Returns:
        The runs, the emulator and its validation.

    References:
        [EQ-SUR-GPB-003], [EQ-SUR-GPB-004], [REF-BASTOS-OHAGAN-2009].

    """
    headline("(b) simulator design and Gaussian-process emulator")
    design = [
        unit_to_box(u)
        for u in latin_hypercube(budget.sim_runs, 3, seed=SEED + 1).points
    ]
    held = [
        unit_to_box(u)
        for u in latin_hypercube(budget.sim_holdout, 3, seed=SEED + 2).points
    ]
    outputs = [simulator_isp(*p) for p in design]
    held_outputs = [simulator_isp(*p) for p in held]
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp = fit_gaussian_process(
            design,
            outputs,
            seed=SEED + 3,
            restarts=2,
            nugget=1.0e-6,
            regularization=GaussianProcessRegularization(),
        )
    check = validate_gp_emulator(gp, held, held_outputs)
    rmse = math.sqrt(
        statistics.fmean(
            (gp.predict(p) - y) ** 2 for p, y in zip(held, held_outputs, strict=True)
        )
    )
    print(
        f"{budget.sim_runs} LHS runs over OF in [{LOWER[0]}, {UPPER[0]}], p_c in "
        f"[{LOWER[1]}, {UPPER[1]}] MPa, eta in [{LOWER[2]}, {UPPER[2]}]; "
        f"output range {min(outputs):.1f}-{max(outputs):.1f} s"
    )
    print(
        f"held-out {budget.sim_holdout} runs: rmse={rmse:.4f} s "
        f"Mahalanobis={check.mahalanobis:.1f} (expected "
        f"{check.mahalanobis_expected:.0f}) coverage={check.coverage:.3f} "
        f"passed={check.passed} conditioning warnings {len(caught)} "
        f"fit+validate {time.perf_counter() - started:.1f} s"
    )
    for flag in check.flags:
        print(f"    flag: {flag}")
    return {
        "design": design,
        "outputs": outputs,
        "gp": gp,
        "check": check,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# (c) calibration with and without discrepancy
# ---------------------------------------------------------------------------


def hotfire_data(seed: int) -> list[float]:
    """Generate the measured vacuum specific impulses of the eight tests.

    Args:
        seed: Seed of the Gaussian measurement noise.

    Returns:
        One noisy ``Isp_vac`` in s per test.

    References:
        [EQ-AERO-LRE-005].

    """
    generator = resolve_stream(seed).generator()
    return [real_isp(of, pc) + generator.gauss(0.0, ISP_NOISE) for of, pc in TESTS]


def posterior_line(name: str, result: Any) -> str:
    """Format the posterior of the efficiency and the sampler diagnostics.

    Args:
        name: Label of the calibration.
        result: A ``KOHResult``.

    Returns:
        One printable line.

    References:
        [EQ-VAL-KOHEM-004].

    """
    low, high = result.theta_interval(0)
    return (
        f"{name:15s} eta mean={result.mean()[0]:.4f} sd={result.sd()[0]:.4f} "
        f"95% [{low:.4f}, {high:.4f}] converged={result.converged} "
        f"({result.stop_reason}) rhat={result.rhat_max:.3f} "
        f"ess={result.ess_min:.0f} evaluations={result.evaluations}"
    )


def coverage_points(count: int) -> list[tuple[float, float]]:
    """Return held-out operating points over the whole operating region.

    Args:
        count: Number of points.

    Returns:
        ``(OF, p_c)`` pairs from a Latin hypercube.

    References:
        [EQ-VAL-KOHEM-005]. No equation is evaluated.

    """
    return [
        (
            LOWER[0] + (UPPER[0] - LOWER[0]) * u[0],
            LOWER[1] + (UPPER[1] - LOWER[1]) * u[1],
        )
        for u in latin_hypercube(count, 2, seed=SEED + 7).points
    ]


def calibration_part(budget: Budget, emulator: dict[str, Any]) -> dict[str, Any]:
    """Calibrate the efficiency with and without discrepancy and compare them.

    Args:
        budget: Sample sizes.
        emulator: Result of :func:`emulator_part`.

    Returns:
        Posteriors, predictions at the untested point, the variance split and
        the measured coverage of both calibrations.

    References:
        [EQ-VAL-KOHEM-004], [EQ-VAL-KOHEM-005], [EQ-VAL-KOHEM-006],
        [EQ-VAL-KOHEM-007], [REF-KENNEDY-OHAGAN-2001].

    """
    headline("(c) Kennedy-O'Hagan calibration, with and without discrepancy")
    data = hotfire_data(SEED + 4)
    print(
        f"hot-fire tests (OF, p_c MPa) -> measured Isp_vac (s), noise sd {ISP_NOISE}:"
    )
    for (of, pc), value in zip(TESTS, data, strict=True):
        print(
            f"  OF={of:.2f} p_c={pc:.1f}: measured {value:.2f} "
            f"(real {real_isp(of, pc):.2f})"
        )
    design = emulator["design"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_delta = KOHEmulatorCalibration(
            [p[:2] for p in design],
            [p[2:] for p in design],
            emulator["outputs"],
            TESTS,
            data,
            theta_bounds=[(LOWER[2], UPPER[2])],
            noise_sd=ISP_NOISE,
            emulator=emulator["gp"],
            seed=SEED + 5,
        )
        without = with_delta.without_discrepancy()
    runs = {}
    for name, calibration in (("with delta", with_delta), ("without delta", without)):
        started = time.perf_counter()
        runs[name] = calibration.sample(
            draws=budget.draws, warmup=budget.warmup, seed=SEED + 6
        )
        print(
            posterior_line(name, runs[name])
            + f" ({time.perf_counter() - started:.1f} s)"
        )
    report = with_delta.identifiability(
        runs["with delta"], reference=runs["without delta"]
    )
    print(
        f"identifiability: contraction of eta with delta = "
        f"{report.contraction[0]:.3f} (weakly identified: "
        f"{report.weakly_identified[0]}), correlation with log discrepancy "
        f"amplitude {report.amplitude_correlation[0]:+.2f}; posterior sd with / "
        f"without delta = {report.sd_ratio[0]:.1f}; mean shift "
        f"{report.mean_shift_in_reference_sd[0]:+.1f} no-delta sd"
    )
    truth = real_isp(*UNTESTED)
    predictions = {}
    for name, calibration in (("with delta", with_delta), ("without delta", without)):
        predictions[name] = calibration.predict(
            runs[name],
            [UNTESTED],
            include_noise=True,
            noise_sd=ISP_NOISE,
            max_draws=budget.predict_draws,
        )
    print(
        f"\nuntested point OF={UNTESTED[0]}, p_c={UNTESTED[1]} MPa: "
        f"real Isp_vac = {truth:.3f} s"
    )
    for name, prediction in predictions.items():
        parts = {
            "emulator": prediction.emulator[0],
            "discrepancy": prediction.discrepancy[0],
            "emu-delta covariance": prediction.covariance[0],
            "parameter": prediction.parameter[0],
            "noise": prediction.noise[0],
        }
        total = prediction.total_variance[0]
        latent_sd = math.sqrt(total - prediction.noise[0])
        print(
            f"  {name:13s} mean={prediction.mean[0]:.3f} s "
            f"error={prediction.mean[0] - truth:+.3f} s "
            f"({(prediction.mean[0] - truth) / latent_sd:+.2f} latent sd) "
            f"95% [{prediction.lower[0]:.3f}, {prediction.upper[0]:.3f}] "
            f"draws={prediction.draws}"
        )
        print(
            "    variance parts (s^2): "
            + ", ".join(f"{key}={value:.4g}" for key, value in parts.items())
            + f"; total={total:.4g}; sum of parts={prediction.component_sum()[0]:.4g}"
        )
    points = coverage_points(budget.coverage_points)
    covers = {}
    for name, calibration in (("with delta", with_delta), ("without delta", without)):
        latent = calibration.predict(
            runs[name],
            points,
            include_noise=False,
            max_draws=budget.predict_draws,
        )
        inside = [
            lo <= real_isp(*p) <= hi
            for p, lo, hi in zip(points, latent.lower, latent.upper, strict=True)
        ]
        fraction = sum(inside) / len(inside)
        width = statistics.fmean(
            hi - lo for lo, hi in zip(latent.lower, latent.upper, strict=True)
        )
        error = math.sqrt(
            statistics.fmean(
                (m - real_isp(*p)) ** 2
                for p, m in zip(points, latent.mean, strict=True)
            )
        )
        covers[name] = {"coverage": fraction, "width": width, "rmse": error}
        print(
            f"  coverage of the real Isp_vac by the 95% interval, "
            f"{len(points)} held-out points, {name:13s}: {fraction:.3f} "
            f"(binomial se {math.sqrt(fraction * (1 - fraction) / len(points)):.3f}) "
            f"mean width {width:.2f} s, rmse of the mean {error:.2f} s"
        )
    low = min(covers, key=lambda key: covers[key]["coverage"])
    difference = covers["with delta"]["coverage"] - covers["without delta"]["coverage"]
    print(
        f"  measured coverage difference (with minus without) = {difference:+.3f}; "
        f"the calibration {low!r} is overconfident here (lower coverage "
        f"{covers[low]['coverage']:.3f} against a nominal 0.95)"
    )
    return {
        "runs": runs,
        "predictions": predictions,
        "truth": truth,
        "covers": covers,
        "calibrations": {"with delta": with_delta, "without delta": without},
        "data": data,
        "overconfident": low,
        "report": report,
    }


# ---------------------------------------------------------------------------
# (d) field output: the nozzle wall temperature profile
# ---------------------------------------------------------------------------


def cooled_nozzle() -> dict[str, Any]:
    """Return the synthetic cooled nozzle at the cooling test point.

    The gas state is the mixture-table state at ``COOLING_OF`` and the
    characteristic velocity is the simulated one at efficiency 0.975.

    Returns:
        The station profile, gas, coolant, channel and ``c*``.

    References:
        [EQ-AERO-COOL-004], [EQ-AERO-LRE-008], [REF-BARTZ-1957].

    """
    state = MIXTURE_TABLE.at(COOLING_OF)
    performance = ideal_rocket_performance(
        gamma=state.gamma,
        molecular_weight=state.molecular_weight,
        chamber_temperature=state.chamber_temperature,
        chamber_pressure=7.0e6,
        throat_area=THROAT_AREA,
        expansion_ratio=EXPANSION_RATIO,
        combustion_efficiency=0.975,
    )
    contour = conical_contour(
        throat_radius=0.05,
        chamber_radius=0.09,
        cylinder_length=0.20,
        expansion_ratio=15.0,
    )
    return {
        "profile": station_profile(contour, state.gamma, 24),
        "gas": GasProperties(
            state.gamma, state.molecular_weight, state.chamber_temperature, 8.0e-5
        ),
        "coolant": CoolantProperties(2100.0, 0.12, 3.5e-4, 700.0),
        "channel": ChannelGeometry(100, 1.2e-3, 2.0e-3, 3.0),
        "cstar": performance.characteristic_velocity,
    }


def wall_temperatures(
    nozzle: dict[str, Any], pressure_mpa: float, bartz: float
) -> tuple[float, ...]:
    """Solve the cooled nozzle and return the gas-side wall temperatures.

    Args:
        nozzle: Result of :func:`cooled_nozzle`.
        pressure_mpa: Chamber pressure in MPa.
        bartz: Bartz multiplier.

    Returns:
        Wall temperature in K at every station.

    Raises:
        RuntimeError: If the cooling solve did not converge.

    References:
        [EQ-AERO-COOL-004], [EQ-AERO-COOL-006].

    """
    solution = solve_regenerative_cooling(
        nozzle["profile"],
        gas=nozzle["gas"],
        coolant=nozzle["coolant"],
        channel=nozzle["channel"],
        chamber_pressure=pressure_mpa * 1.0e6,
        characteristic_velocity=nozzle["cstar"],
        coolant_mass_flow=11.0,
        coolant_inlet_temperature=290.0,
        wall_conductivity=320.0,
        wall_thickness=1.0e-3,
        bartz_multiplier=bartz,
    )
    if not solution.converged:
        raise RuntimeError("cooling solve did not converge")
    return tuple(solution.gas_wall_temperature)


def cooling_part(budget: Budget) -> dict[str, Any]:
    """Calibrate the Bartz multiplier from the wall temperature profile.

    The field calibration uses all 24 stations of three experiments; the
    scalar calibrations use one station each (a single thermocouple).

    Args:
        budget: Sample sizes.

    Returns:
        The field posterior and the two scalar comparisons.

    References:
        [EQ-VAL-FUNCEM-002], [EQ-VAL-FUNCEM-005], [EQ-VAL-FUNCEM-008],
        [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008].

    """
    headline("(d) field output: nozzle wall temperature, Bartz multiplier")
    nozzle = cooled_nozzle()
    profile = nozzle["profile"]
    low, high = COOLING_BOUNDS
    design = latin_hypercube(budget.cooling_runs, 2, seed=SEED + 8).points
    controls = [(4.5 + 3.5 * u[0],) for u in design]
    parameters = [(low + (high - low) * u[1],) for u in design]
    fields = [
        wall_temperatures(nozzle, c[0], p[0])
        for c, p in zip(controls, parameters, strict=True)
    ]
    held = latin_hypercube(budget.cooling_holdout, 2, seed=SEED + 9).points
    held_controls = [(4.5 + 3.5 * u[0],) for u in held]
    held_parameters = [(low + (high - low) * u[1],) for u in held]
    held_fields = [
        wall_temperatures(nozzle, c[0], p[0])
        for c, p in zip(held_controls, held_parameters, strict=True)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field_emulator = FieldEmulator(
            controls, parameters, fields, locations=profile.x, seed=SEED + 10
        )
    validation = field_emulator.validate(held_controls, held_parameters, held_fields)
    print(
        f"{budget.cooling_runs} runs over p_c in [4.5, 8.0] MPa and Bartz multiplier "
        f"in [{low}, {high}]; {field_emulator.basis.components} basis component(s); "
        f"nozzle c*={nozzle['cstar']:.0f} m/s, gas Tc="
        f"{nozzle['gas'].chamber_temperature:.0f} K"
    )
    print(
        f"emulator on {validation.holdout} held-out runs: weight standardized rmse "
        f"{tuple(round(v, 2) for v in validation.weight_standardized_rmse)}, "
        f"weight coverage {validation.weight_coverage}, field rmse "
        f"{validation.field_rmse:.2f} K ({validation.field_relative_rmse:.3%} of the "
        f"spread), field coverage {validation.field_coverage:.3f}, conditioning "
        f"warnings {len(caught)}"
    )
    generator = resolve_stream(SEED + 11).generator()
    field_controls = [(p,) for p in COOLING_PRESSURES]
    field_data = [
        [
            temperature + generator.gauss(0.0, COOLING_NOISE)
            for temperature in wall_temperatures(nozzle, p, COOLING_BARTZ_TRUTH)
        ]
        for p in COOLING_PRESSURES
    ]
    calibration = FunctionalCalibration(
        field_emulator,
        field_controls,
        field_data,
        theta_bounds=[COOLING_BOUNDS],
        noise_sd=COOLING_NOISE,
    )
    started = time.perf_counter()
    result = calibration.sample(
        draws=budget.cooling_draws, warmup=budget.cooling_warmup, seed=SEED + 12
    )
    mean, sd = result.mean()[0], result.sd()[0]
    interval = result.theta_interval(0)
    print(
        f"field calibration (3 experiments x 24 stations, noise sd {COOLING_NOISE} "
        f"K): b mean={mean:.4f} sd={sd:.4f} 95% [{interval[0]:.4f}, "
        f"{interval[1]:.4f}] generating b={COOLING_BARTZ_TRUTH} "
        f"(error {(mean - COOLING_BARTZ_TRUTH) / sd:+.2f} sd) "
        f"converged={result.converged} ({result.stop_reason}) "
        f"rhat={result.rhat_max:.3f} ess={result.ess_min:.0f} "
        f"contraction={calibration.contraction(result)[0]:.4f} "
        f"({time.perf_counter() - started:.1f} s)"
    )
    comparisons = {}
    for label, index in (
        ("throat thermocouple", profile.throat_index),
        ("exit thermocouple", len(profile.x) - 1),
    ):
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison = calibration.compare_with_scalar(
                result,
                index,
                draws=budget.cooling_draws,
                warmup=budget.cooling_warmup,
                seed=SEED + 13,
            )
        comparisons[label] = comparison
        print(
            f"  {label:19s} (station {index}, x={profile.x[index]:.3f} m): "
            f"b mean={comparison.scalar_mean[0]:.4f} sd={comparison.scalar_sd[0]:.4f} "
            f"converged={comparison.scalar_result.converged}; posterior sd ratio "
            f"single/field = {comparison.sd_ratio[0]:.2f} "
            f"({time.perf_counter() - started:.1f} s)"
        )
    return {
        "result": result,
        "mean": mean,
        "sd": sd,
        "comparisons": comparisons,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# (e) the next test
# ---------------------------------------------------------------------------


def design_part(budget: Budget, calibrated: dict[str, Any]) -> dict[str, Any]:
    """Choose the next tests by expected information gain on the efficiency.

    The model of a new measurement at candidate ``k`` is ``eta I_k`` with
    ``I_k = Isp_vac(x_k, 1)`` (exact, since the simulator is linear in ``eta``)
    plus Gaussian noise of variance ``ISP_NOISE^2 + sd_delta(x_k)^2``, with the
    discrepancy sd of the calibration with discrepancy. The prior of ``eta`` is
    the normal law matching that posterior. Because the model is linear in
    ``eta``, the Laplace estimator is exact up to the Monte Carlo average over
    identical Jacobians, and it needs no inner sample; nested Monte Carlo is
    the independent, biased-upward check.

    Args:
        budget: Sample sizes.
        calibrated: Result of :func:`calibration_part`.

    Returns:
        EIG per candidate and the designs found by each search.

    References:
        [EQ-OED-EIG-LIN-001], [EQ-DES-NOED-001], [EQ-DES-NOED-003],
        [EQ-DES-NOED-004], [REF-CHALONER-VERDINELLI-1995],
        [REF-HUAN-MARZOUK-2013].

    """
    headline("(e) the next test: expected information gain on the efficiency")
    result = calibrated["runs"]["with delta"]
    prior_mean, prior_sd = result.mean()[0], result.sd()[0]
    delta_prediction = calibrated["calibrations"]["with delta"].predict(
        result,
        list(CANDIDATES),
        include_noise=False,
        max_draws=budget.predict_draws,
    )
    delta_sd = [math.sqrt(v) for v in delta_prediction.discrepancy]
    unit_isp = [simulator_isp(of, pc, 1.0) for of, pc in CANDIDATES]
    noise = [math.hypot(ISP_NOISE, d) for d in delta_sd]

    def forward(theta: tuple[float, ...], index: float) -> float:
        """Return the mean measurement ``eta I_k`` at candidate ``k``.

        References:
            [EQ-AERO-LRE-005].

        """
        return theta[0] * unit_isp[int(index)]

    def noise_sd(index: float) -> float:
        """Return the noise sd at candidate ``k``.

        References:
            [EQ-VAL-KOHEM-006].

        """
        return noise[int(index)]

    model = NonlinearExperimentModel(
        forward, (prior_mean,), ((prior_sd**2,),), noise_sd
    )
    print(
        f"prior of eta (normal approximation of the posterior with discrepancy): "
        f"mean={prior_mean:.4f} sd={prior_sd:.4f}"
    )
    print(
        "method: Laplace (exact for a model linear in eta, no inner sample) for the "
        "search; nested Monte Carlo (biased upward) and the closed form as checks"
    )
    singles = []
    for index, (of, pc) in enumerate(CANDIDATES):
        laplace = laplace_eig(
            model, [float(index)], samples=budget.laplace_samples, seed=SEED + 14
        )
        nested = nested_monte_carlo_eig(
            model,
            [float(index)],
            outer=budget.nested_outer,
            inner=budget.nested_inner,
            seed=SEED + 15,
        )
        closed = 0.5 * math.log(1.0 + (prior_sd * unit_isp[index] / noise[index]) ** 2)
        singles.append(
            {
                "laplace": laplace,
                "nested": nested,
                "closed": closed,
                "noise": noise[index],
            }
        )
        print(
            f"  candidate {index} (OF={of}, p_c={pc}): "
            f"sd_delta={delta_sd[index]:.2f} s "
            f"noise sd={noise[index]:.2f} s | EIG Laplace={laplace.value:.4f} "
            f"se={laplace.standard_error:.4f} | nested={nested.value:.4f} "
            f"se={nested.standard_error:.4f} | closed form={closed:.4f} nats"
        )
    indices = [float(k) for k in range(len(CANDIDATES))]
    searches = {}
    for method, samples, inner in (
        ("laplace", budget.laplace_samples, 1),
        ("nested", budget.nested_outer // 2, budget.nested_inner),
    ):
        exhaustive = exhaustive_nonlinear_design(
            model,
            indices,
            2,
            method=method,
            samples=samples,
            inner=inner,
            seed=SEED + 16,
        )
        greedy = greedy_nonlinear_design(
            model,
            indices,
            2,
            method=method,
            samples=samples,
            inner=inner,
            seed=SEED + 16,
        )
        searches[method] = {"exhaustive": exhaustive, "greedy": greedy}
        for name, found in (("exhaustive", exhaustive), ("greedy", greedy)):
            chosen = sorted(int(d) for d in found.design)
            print(
                f"  {method:7s} {name:10s} two tests: candidates {chosen} "
                f"EIG={found.value:.4f} se={found.standard_error:.4f} "
                f"model evaluations={found.model_evaluations}"
            )
        same = sorted(exhaustive.design) == sorted(greedy.design)
        print(f"  {method}: greedy equals exhaustive: {same}")
    pair = sorted(int(d) for d in searches["laplace"]["exhaustive"].design)
    closed_pair = 0.5 * math.log(
        1.0 + sum((prior_sd * unit_isp[k] / noise[k]) ** 2 for k in pair)
    )
    print(
        f"  closed form of the best pair {pair}: {closed_pair:.4f} nats against "
        f"Laplace {searches['laplace']['exhaustive'].value:.4f}; for this model the "
        "objective is a monotone function of a sum over the tests, so greedy "
        "selection is optimal and the agreement is a check, not a finding"
    )
    best = max(range(len(CANDIDATES)), key=lambda k: singles[k]["laplace"].value)
    print(
        f"best single next test: candidate {best} "
        f"(OF={CANDIDATES[best][0]}, p_c={CANDIDATES[best][1]} MPa)"
    )
    return {
        "singles": singles,
        "best": best,
        "searches": searches,
        "closed_pair": closed_pair,
        "prior": (prior_mean, prior_sd),
        "unit_isp": unit_isp,
        "noise": noise,
    }


def summary(
    emulator: dict[str, Any],
    calibrated: dict[str, Any],
    cooling: dict[str, Any],
    design: dict[str, Any],
) -> None:
    """Print the campaign summary.

    Args:
        emulator: Result of :func:`emulator_part`.
        calibrated: Result of :func:`calibration_part`.
        cooling: Result of :func:`cooling_part`.
        design: Result of :func:`design_part`.

    References:
        [EQ-VAL-KOHEM-007], [EQ-VAL-FUNCEM-008], [EQ-DES-NOED-004].

    """
    headline("campaign summary (synthetic)")
    check = emulator["check"]
    with_delta = calibrated["predictions"]["with delta"]
    without = calibrated["predictions"]["without delta"]
    covers = calibrated["covers"]
    ratios = [c.sd_ratio[0] for c in cooling["comparisons"].values()]
    print(
        f"emulator: held-out rmse {emulator['rmse']:.4f} s, Mahalanobis "
        f"{check.mahalanobis:.1f} (expected {check.mahalanobis_expected:.0f}), "
        f"passed={check.passed}"
    )
    print(
        f"untested point, real {calibrated['truth']:.2f} s: with delta "
        f"{with_delta.mean[0]:.2f} [{with_delta.lower[0]:.2f}, "
        f"{with_delta.upper[0]:.2f}]; without {without.mean[0]:.2f} "
        f"[{without.lower[0]:.2f}, {without.upper[0]:.2f}]"
    )
    print(
        f"coverage of the real Isp_vac: with delta "
        f"{covers['with delta']['coverage']:.3f}, without "
        f"{covers['without delta']['coverage']:.3f}; overconfident: "
        f"{calibrated['overconfident']}"
    )
    print(
        f"Bartz multiplier: {cooling['mean']:.4f} +- {cooling['sd']:.4f} (generating "
        f"{COOLING_BARTZ_TRUTH}); single-thermocouple sd is "
        f"{min(ratios):.1f}-{max(ratios):.1f} times larger"
    )
    best = design["best"]
    pair = sorted(int(d) for d in design["searches"]["laplace"]["exhaustive"].design)
    print(
        f"next test: candidate {best} (OF={CANDIDATES[best][0]}, "
        f"p_c={CANDIDATES[best][1]} MPa), "
        f"EIG {design['singles'][best]['laplace'].value:.3f} nats; best two tests "
        f"{pair}"
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def has_matplotlib() -> bool:
    """Return whether Matplotlib is importable.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def save(figure: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    Args:
        figure: Matplotlib figure.
        name: File name.

    Returns:
        The file name written.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def prediction_figure(calibrated: dict[str, Any], budget: Budget) -> str:
    """Draw the calibrated Isp bands with and without discrepancy.

    Args:
        calibrated: Result of :func:`calibration_part`.
        budget: Sample sizes (draws used by the predictions).

    Returns:
        The file name written.

    References:
        [EQ-VAL-KOHEM-005], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    slice_pressure = UNTESTED[1]
    grid = [2.0 + 1.2 * k / 24 for k in range(25)]
    controls = [(of, slice_pressure) for of in grid]
    axes = theme.new_axes(figsize=(7.6, 4.8))
    palette = {"with delta": "model", "without delta": "series-3"}
    labels = []
    for name, role in palette.items():
        prediction = calibrated["calibrations"][name].predict(
            calibrated["runs"][name],
            controls,
            include_noise=False,
            max_draws=budget.predict_draws,
        )
        colour = theme.role_colour(role)
        theme.band(axes, grid, prediction.lower, prediction.upper, color=colour)
        theme.line(axes, grid, prediction.mean, color=colour)
        shown = name.replace("delta", "discrepancy")
        labels.append((grid[-1], prediction.mean[-1], shown, colour))
    theme.line(
        axes,
        grid,
        [real_isp(of, slice_pressure) for of in grid],
        color=theme.role_colour("reference"),
        dashes=(0.0, (5.0, 3.0)),
    )
    labels.append(
        (
            grid[-1],
            real_isp(grid[-1], slice_pressure),
            "real engine",
            theme.role_colour("reference"),
        )
    )
    on_slice = [
        (of, value)
        for (of, pc), value in zip(TESTS, calibrated["data"], strict=True)
        if pc == slice_pressure
    ]
    other = [
        (of, value)
        for (of, pc), value in zip(TESTS, calibrated["data"], strict=True)
        if pc != slice_pressure
    ]
    theme.points(
        axes,
        [of for of, _ in on_slice],
        [v for _, v in on_slice],
        color=theme.role_colour("observed"),
        size=44.0,
    )
    theme.points(
        axes,
        [of for of, _ in other],
        [v for _, v in other],
        color=theme.role_colour("observed"),
        size=44.0,
        marker="D",
        alpha=0.45,
    )
    theme.reference_line(axes, x=UNTESTED[0], entity="threshold", label="untested")
    axes.set_xlim(grid[0], grid[-1] + 0.05)
    theme.direct_labels(axes, labels)
    covers = calibrated["covers"]
    without = calibrated["predictions"]["without delta"]
    theme.axis_labels(
        axes,
        "Mixture ratio O/F",
        f"Vacuum specific impulse at {slice_pressure:g} MPa (s)",
    )
    theme.title_block(
        axes,
        f"Discrepancy term lifts 95% coverage from "
        f"{covers['without delta']['coverage']:.0%} to "
        f"{covers['with delta']['coverage']:.0%}",
        f"{len(TESTS)} synthetic tests (circles: {slice_pressure:g} MPa); without "
        f"the term the untested point is off by "
        f"{without.mean[0] - calibrated['truth']:+.1f} s",
    )
    return save(axes.figure, "hotfire-campaign-isp-bands.png")


def information_figure(design: dict[str, Any]) -> str:
    """Draw the expected information gain of the candidate next tests.

    Args:
        design: Result of :func:`design_part`.

    Returns:
        The file name written.

    References:
        [EQ-DES-NOED-003], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    singles = design["singles"]
    order = sorted(range(len(singles)), key=lambda k: singles[k]["laplace"].value)
    rows = list(range(len(order)))
    axes = theme.new_axes(figsize=(7.6, 4.4))
    theme.points(
        axes,
        [singles[k]["laplace"].value for k in order],
        rows,
        color=theme.role_colour("model"),
        size=52.0,
    )
    theme.points(
        axes,
        [singles[k]["nested"].value for k in order],
        rows,
        color=theme.role_colour("series-3"),
        size=44.0,
        marker="D",
        alpha=0.8,
    )
    axes.set_yticks(rows)
    axes.set_yticklabels(
        [f"OF {CANDIDATES[k][0]:g}, {CANDIDATES[k][1]:g} MPa" for k in order]
    )
    theme.axis_labels(axes, "Expected information gain on the efficiency (nats)", "")
    best = design["best"]
    theme.title_block(
        axes,
        f"Testing OF {CANDIDATES[best][0]:g} at {CANDIDATES[best][1]:g} MPa gains "
        f"{singles[best]['laplace'].value:.2f} nats",
        "Circles: Laplace (exact for this linear model); diamonds: nested Monte Carlo "
        "(upward bias); one added test",
    )
    return save(axes.figure, "hotfire-campaign-information-gain.png")


def main() -> None:
    """Run the five parts, print every number, and write the figures.

    References:
        [EQ-VAL-KOHEM-005], [EQ-VAL-FUNCEM-008], [EQ-DES-NOED-004].

    """
    started = time.perf_counter()
    print("Hot-fire test campaign laboratory (all cases synthetic)")
    truth_part()
    emulator = emulator_part(FULL)
    calibrated = calibration_part(FULL, emulator)
    cooling = cooling_part(FULL)
    design = design_part(FULL, calibrated)
    summary(emulator, calibrated, cooling, design)
    print(f"\ntotal wall time {time.perf_counter() - started:.1f} s")
    if not has_matplotlib():
        print("Matplotlib is not installed; figures skipped")
        return
    written = (
        prediction_figure(calibrated, FULL),
        information_figure(design),
    )
    print("figures: " + ", ".join(written))


if __name__ == "__main__":
    main()
