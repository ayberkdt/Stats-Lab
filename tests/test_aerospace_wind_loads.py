"""Tests for the synthetic wind-load dispersion of a planar ascent.

Every number is synthetic: the vehicle is the illustrative two-stage vehicle
of the ascent tests, the wind field is a demonstration random process and the
``q alpha`` limits are invented. Closed forms first (relative-flow geometry,
the exact cosine-basis expansion, the no-wind identity with
``simulate_ascent``), then independent oracles (the first-order small-wind
slope computed without ``relative_flow``), then the reliability, sensitivity
and extreme-value workflows, then the failure cases.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import unittest

from probstat.aerospace.ascent import (
    DragModel,
    GuidanceSpec,
    StageSpec,
    VehicleSpec,
    atmosphere,
    simulate_ascent,
)
from probstat.aerospace.wind_loads import (
    SYNTHETIC_DEMONSTRATION_WIND,
    SyntheticWindField,
    relative_flow,
    small_wind_slope,
    wind_extreme_study,
    wind_load_history,
    wind_load_limit_state,
    wind_loads_model,
    wind_reliability_study,
)
from probstat.distributions import Normal
from probstat.reliability import form_hlrf
from probstat.sensitivity import sobol_analysis

_DRAG = DragModel.tabulated(
    [(0.0, 0.3), (0.8, 0.3), (1.2, 0.5), (2.0, 0.4), (5.0, 0.25), (50.0, 0.2)]
)
_GUIDANCE = GuidanceSpec(8.0, 0.5)
_FIELD = SYNTHETIC_DEMONSTRATION_WIND
_LIMIT = 3600.0  # Pa rad, synthetic design limit (exceeded by about 2 % of profiles)


def _vehicle() -> VehicleSpec:
    """Illustrative two-stage vehicle (synthetic numbers, not a real launcher)."""
    first = StageSpec(4000.0, 42000.0, 1.4e6, 300.0, 1.0, 3.0, coast_after=2.0)
    second = StageSpec(800.0, 4500.0, 60e3, 340.0, 0.1, 3.0)
    return VehicleSpec((first, second), 500.0)


def _model(limit: float = _LIMIT, **overrides: object):  # type: ignore[no-untyped-def]
    """Wind-load model with vehicle dispersions (synthetic)."""
    config: dict[str, object] = {
        "q_alpha_limit": limit,
        "thrust_scale": Normal(1.0, 0.01),
        "drag_scale": Normal(1.0, 0.05),
        "max_step": 2.0,
    }
    config.update(overrides)
    return wind_loads_model(
        _vehicle(),
        _GUIDANCE,
        _DRAG,
        _FIELD,
        **config,  # type: ignore[arg-type]
    )


def _trajectory(max_step: float = 2.0) -> tuple[tuple[float, ...], ...]:
    """Record the nominal trajectory."""
    result = simulate_ascent(
        _vehicle(), _GUIDANCE, drag=_DRAG, max_step=max_step, record=True
    )
    assert result.status == "burnout"
    return result.trajectory


_XI = (0.4, -0.7, 0.9, 0.2, -0.3, 0.5, 0.1, -0.2)


class WindFieldTests(unittest.TestCase):
    """Cosine random-field expansion of the synthetic wind."""

    def test_weights_and_names(self) -> None:
        """Weights sum to one, decay, and the coordinates are named."""
        weights = _FIELD.weights
        self.assertAlmostEqual(math.fsum(weights), 1.0, places=14)
        self.assertTrue(all(a > b for a, b in zip(weights, weights[1:], strict=False)))
        self.assertEqual(_FIELD.coordinate_names[0], "wind_1")
        self.assertEqual(len(_FIELD.coordinate_names), 8)
        self.assertGreater(_FIELD.retained_spectral_fraction, 0.99)

    def test_mean_profile_and_taper(self) -> None:
        """Mean equals the jet speed at the jet altitude; taper 1 -> 0."""
        self.assertAlmostEqual(_FIELD.mean(12000.0), 35.0, places=12)
        self.assertAlmostEqual(
            _FIELD.mean(12000.0 + 3500.0), 35.0 * math.exp(-0.5), places=12
        )
        self.assertEqual(_FIELD.taper(0.0), 1.0)
        self.assertEqual(_FIELD.taper(30000.0), 1.0)
        self.assertEqual(_FIELD.taper(40000.0), 0.0)
        self.assertEqual(_FIELD.taper(55000.0), 0.0)
        self.assertAlmostEqual(_FIELD.taper(35000.0), 0.5, places=12)
        self.assertEqual(_FIELD.speed(45000.0, _XI), 0.0)

    def test_basis_function_closed_form(self) -> None:
        """A unit coordinate gives sigma T sqrt(2 lam_k) cos(k pi h / H)."""
        h = 7300.0
        lam = _FIELD.weights
        for k in range(8):
            xi = [0.0] * 8
            xi[k] = 1.0
            expected = _FIELD.fluctuation_std * math.sqrt(lam[k])
            if k > 0:
                expected *= math.sqrt(2.0) * math.cos(k * math.pi * h / 40000.0)
            self.assertAlmostEqual(_FIELD.perturbation(h, xi), expected, places=12)

    def test_variance_matches_the_expansion(self) -> None:
        """Sample variance of the perturbation matches sigma^2 sum lam phi^2."""
        h = 9000.0
        lam = _FIELD.weights
        exact = _FIELD.fluctuation_std**2 * (
            lam[0]
            + sum(
                2.0 * lam[k] * math.cos(k * math.pi * h / 40000.0) ** 2
                for k in range(1, 8)
            )
        )
        rng = random.Random(11)
        draws = [
            _FIELD.perturbation(h, [rng.gauss(0.0, 1.0) for _ in range(8)])
            for _ in range(4000)
        ]
        variance = sum(d * d for d in draws) / len(draws)
        # Var of a sample variance of a normal: 2 sigma^4 / n.
        self.assertAlmostEqual(variance, exact, delta=4.0 * exact * math.sqrt(2 / 4000))

    def test_validation(self) -> None:
        """Invalid parameters and coordinates are rejected."""
        base = {
            "jet_speed": 30.0,
            "jet_altitude": 10000.0,
            "jet_width": 3000.0,
            "fluctuation_std": 5.0,
            "correlation_length": 5000.0,
            "top_altitude": 40000.0,
        }
        for key, bad in (
            ("jet_width", 0.0),
            ("fluctuation_std", -1.0),
            ("correlation_length", -5.0),
            ("top_altitude", 0.0),
            ("jet_altitude", -1.0),
            ("terms", 0),
        ):
            with self.assertRaises(ValueError, msg=key):
                SyntheticWindField(**{**base, key: bad})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SyntheticWindField(**{**base, "terms": 2.5})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            # 2 terms cannot represent a 1 km correlation length over 40 km.
            SyntheticWindField(**{**base, "correlation_length": 1000.0, "terms": 2})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _FIELD.perturbation(1000.0, (0.0, 1.0))
        with self.assertRaises(ValueError):
            _FIELD.speed(1000.0, [math.nan] * 8)


class RelativeFlowTests(unittest.TestCase):
    """Air-relative velocity and angle of attack."""

    def test_vertical_flight(self) -> None:
        """Vertical rise in a crosswind: tan(alpha) = w / v."""
        v_rel, alpha = relative_flow(300.0, 0.5 * math.pi, 40.0)
        self.assertAlmostEqual(v_rel, math.hypot(300.0, 40.0), places=10)
        self.assertAlmostEqual(alpha, math.atan2(40.0, 300.0), places=12)

    def test_horizontal_flight(self) -> None:
        """Horizontal flight: wind is along the axis, so alpha is zero."""
        v_rel, alpha = relative_flow(300.0, 0.0, 40.0)
        self.assertAlmostEqual(v_rel, 260.0, places=10)
        self.assertEqual(alpha, 0.0)
        _, alpha_head = relative_flow(300.0, 0.0, -40.0)
        self.assertEqual(alpha_head, 0.0)

    def test_no_wind(self) -> None:
        """Without wind alpha = 0 and V_rel = v at every flight-path angle."""
        for gamma in (-0.3, 0.0, 0.7, 1.2, 0.5 * math.pi):
            v_rel, alpha = relative_flow(512.0, gamma, 0.0)
            self.assertAlmostEqual(v_rel, 512.0, places=10)
            self.assertEqual(alpha, 0.0)

    def test_matches_vector_construction(self) -> None:
        """Independent vector construction of V_rel and the angle."""
        v, gamma, w = 420.0, 0.8, 25.0
        rel = (v * math.cos(gamma) - w, v * math.sin(gamma))
        axis = (math.cos(gamma), math.sin(gamma))
        dot = rel[0] * axis[0] + rel[1] * axis[1]
        cross = axis[0] * rel[1] - axis[1] * rel[0]
        v_rel, alpha = relative_flow(v, gamma, w)
        self.assertAlmostEqual(v_rel, math.hypot(*rel), places=10)
        self.assertAlmostEqual(alpha, math.atan2(cross, dot), places=12)
        self.assertAlmostEqual(alpha, math.acos(dot / math.hypot(*rel)), places=9)

    def test_rejects_bad_input(self) -> None:
        """Negative speed, non-finite and non-numeric inputs are rejected."""
        with self.assertRaises(ValueError):
            relative_flow(-1.0, 0.5, 1.0)
        with self.assertRaises(ValueError):
            relative_flow(10.0, math.inf, 1.0)
        with self.assertRaises(TypeError):
            relative_flow("10", 0.5, 1.0)  # type: ignore[arg-type]


class LoadHistoryTests(unittest.TestCase):
    """Loads along the recorded trajectory."""

    def test_no_wind_reproduces_simulate_ascent_q(self) -> None:
        """Zero amplitude: q history equals rho v^2 / 2 of the ascent, exactly."""
        result = simulate_ascent(
            _vehicle(), _GUIDANCE, drag=_DRAG, max_step=2.0, record=True
        )
        history = wind_load_history(result.trajectory, _FIELD, _XI, amplitude=0.0)
        for (_, h, v, _, _), q in zip(
            result.trajectory, history.dynamic_pressures, strict=True
        ):
            self.assertEqual(q, 0.5 * atmosphere(max(h, 0.0)).density * v * v)
        self.assertEqual(history.max_dynamic_pressure, result.max_dynamic_pressure)
        self.assertEqual(history.max_q_alpha, 0.0)
        self.assertTrue(all(a == 0.0 for a in history.angles_of_attack))

    def test_no_wind_when_the_field_is_zero(self) -> None:
        """A field with no jet and no fluctuation is the no-wind case."""
        calm = SyntheticWindField(0.0, 10000.0, 3000.0, 0.0, 5000.0, 40000.0)
        trajectory = _trajectory()
        history = wind_load_history(trajectory, calm, [0.0] * 8)
        reference = wind_load_history(trajectory, _FIELD, _XI, amplitude=0.0)
        self.assertEqual(history.dynamic_pressures, reference.dynamic_pressures)
        self.assertEqual(history.max_q_alpha, 0.0)

    def test_max_q_location_is_physical(self) -> None:
        """Max q is of order 1e5 Pa near 11 km for this synthetic vehicle."""
        history = wind_load_history(_trajectory(), _FIELD, _XI, amplitude=0.0)
        self.assertGreater(history.max_dynamic_pressure, 1.0e5)
        self.assertGreater(history.altitude_of_max_q, 8000.0)
        self.assertLess(history.altitude_of_max_q, 14000.0)

    def test_small_wind_is_linear_in_amplitude(self) -> None:
        """Max q|alpha| / a approaches the closed-form first-order slope."""
        trajectory = _trajectory()
        slope = small_wind_slope(trajectory, _FIELD, _XI)
        self.assertGreater(slope, 0.0)
        measured = []
        for amplitude in (1e-4, 2e-4, 4e-4):
            history = wind_load_history(trajectory, _FIELD, _XI, amplitude=amplitude)
            measured.append(history.max_q_alpha / amplitude)
            # O(w/v): wind <= 50 m/s * 4e-4 over v >= 50 m/s gives < 4e-4.
            self.assertAlmostEqual(measured[-1] / slope, 1.0, delta=2e-3)
        self.assertAlmostEqual(measured[1] / measured[0], 1.0, delta=1e-3)
        self.assertAlmostEqual(measured[2] / measured[0], 1.0, delta=1e-3)
        # The sign of the wind does not matter at first order.
        negative = wind_load_history(trajectory, _FIELD, _XI, amplitude=-1e-4)
        self.assertAlmostEqual(
            negative.max_q_alpha / 1e-4 / slope, measured[0] / slope, delta=2e-3
        )

    def test_full_wind_departs_from_linearity(self) -> None:
        """At amplitude 1 the response is no longer the linear one (measured)."""
        trajectory = _trajectory()
        slope = small_wind_slope(trajectory, _FIELD, _XI)
        full = wind_load_history(trajectory, _FIELD, _XI).max_q_alpha
        ratio = full / slope
        self.assertNotAlmostEqual(ratio, 1.0, delta=1e-3)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 1.5)

    def test_load_scale_and_dependence_on_density(self) -> None:
        """Scale q alpha with the density multiplier (first order in rho)."""
        trajectory = _trajectory()
        base = wind_load_history(trajectory, _FIELD, _XI)
        dense = wind_load_history(trajectory, _FIELD, _XI, density_scale=1.1)
        self.assertAlmostEqual(dense.max_q_alpha / base.max_q_alpha, 1.1, delta=1e-6)

    def test_speed_floor_excludes_the_pad(self) -> None:
        """Points slower than min_speed are not in the q alpha maximum."""
        trajectory = _trajectory()
        history = wind_load_history(trajectory, _FIELD, _XI, min_speed=200.0)
        self.assertLess(history.evaluated_points, len(trajectory))
        for v, qa in zip(history.speeds, history.q_alpha, strict=True):
            if v < 200.0:
                self.assertEqual(qa, 0.0)
        self.assertEqual(history.max_q_alpha, max(history.q_alpha))

    def test_rejects_bad_input(self) -> None:
        """Wrong field type, empty trajectory and an unreachable speed floor."""
        trajectory = _trajectory()
        with self.assertRaises(TypeError):
            wind_load_history(trajectory, "field", _XI)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            wind_load_history((), _FIELD, _XI)
        with self.assertRaises(ValueError):
            wind_load_history(trajectory, _FIELD, _XI, min_speed=1.0e9)
        with self.assertRaises(ValueError):
            wind_load_history(trajectory, _FIELD, _XI, density_scale=0.0)
        with self.assertRaises(ValueError):
            wind_load_history(trajectory, _FIELD, _XI[:3])
        with self.assertRaises(TypeError):
            small_wind_slope(trajectory, 3.0, _XI)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            small_wind_slope(trajectory, _FIELD, _XI, min_speed=1.0e9)


class ModelTests(unittest.TestCase):
    """The probabilistic model."""

    def test_structure_and_consistency(self) -> None:
        """Inputs, outputs and agreement with a direct history at the mean."""
        model = _model()
        self.assertEqual(model.dimension, 10)
        self.assertEqual(
            model.inputs.names,
            (*_FIELD.coordinate_names, "thrust_scale", "drag_scale"),
        )
        self.assertEqual(
            model.quantity_names,
            (
                "max_dynamic_pressure",
                "altitude_of_max_q",
                "max_q_alpha",
                "altitude_of_max_q_alpha",
                "q_alpha_margin",
            ),
        )
        out = model.evaluate(model.nominal())
        history = wind_load_history(_trajectory(), _FIELD, [0.0] * 8)
        self.assertAlmostEqual(out["max_q_alpha"], history.max_q_alpha, places=9)
        self.assertAlmostEqual(
            out["altitude_of_max_q_alpha"], history.altitude_of_max_q_alpha, places=9
        )
        self.assertAlmostEqual(out["q_alpha_margin"], _LIMIT - out["max_q_alpha"])
        self.assertGreater(out["max_q_alpha"], 0.0)

    def test_vehicle_dispersion_acts_on_loads(self) -> None:
        """Thrust and drag scale change q alpha through the trajectory only."""
        model = _model()
        names = model.inputs.names
        point = {n: 0.0 for n in names}
        point["thrust_scale"] = 1.0
        point["drag_scale"] = 1.0
        base = model.evaluate(point)["max_q_alpha"]
        point["thrust_scale"] = 1.03
        faster = model.evaluate(point)["max_q_alpha"]
        self.assertNotEqual(faster, base)
        point["thrust_scale"] = 1.0
        point["drag_scale"] = 1.15
        self.assertNotEqual(model.evaluate(point)["max_q_alpha"], base)

    def test_fixed_amplitude_and_picklable_forward(self) -> None:
        """A fixed wind amplitude halves the load at first order; pickling."""
        full = _model(thrust_scale=1.0, drag_scale=1.0)
        small = _model(thrust_scale=1.0, drag_scale=1.0, wind_amplitude=1e-4)
        point = [0.3, -0.2, 0.5, 0.1, 0.0, 0.4, -0.1, 0.2]
        a = full.evaluate(point)["max_q_alpha"]
        b = small.evaluate(point)["max_q_alpha"]
        self.assertLess(b, 1e-3 * a)
        clone = pickle.loads(pickle.dumps(full.model))
        self.assertEqual(
            clone(
                dict(zip(full.inputs.names, point, strict=True))
                | {"thrust_scale": 1.0, "drag_scale": 1.0, "wind_amplitude": 1.0}
            )["max_q_alpha"],
            a,
        )

    def test_amplitude_as_random_input(self) -> None:
        """A distribution on the amplitude adds an input."""
        model = _model(wind_amplitude=Normal(1.0, 0.1))
        self.assertEqual(model.dimension, 11)
        self.assertIn("wind_amplitude", model.inputs.names)

    def test_failure_cases(self) -> None:
        """Bad arguments and failed ascents."""
        kw = {"q_alpha_limit": 1000.0}
        with self.assertRaises(TypeError):
            wind_loads_model("v", _GUIDANCE, _DRAG, _FIELD, **kw)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            wind_loads_model(_vehicle(), "g", _DRAG, _FIELD, **kw)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            wind_loads_model(_vehicle(), _GUIDANCE, 0.3, _FIELD, **kw)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            wind_loads_model(_vehicle(), _GUIDANCE, _DRAG, None, **kw)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _model(limit=0.0)
        with self.assertRaises(ValueError):
            _model(max_step=0.0)
        with self.assertRaises(TypeError):
            _model(thrust_scale=True)
        # A thrust-to-weight ratio below 1 is a failed run, not a number.
        weak = _model(thrust_scale=0.4, drag_scale=1.0)
        with self.assertRaises(RuntimeError):
            weak.evaluate([0.0] * 8)
        with self.assertRaises(ValueError):
            _model(thrust_scale=-1.0, drag_scale=1.0).evaluate([0.0] * 8)


class ReliabilityTests(unittest.TestCase):
    """FORM, subset simulation and Monte Carlo on max q alpha."""

    def test_methods_agree_and_wind_dominates(self) -> None:
        """The three estimators agree within their stated bands (measured)."""
        model = _model()
        study = wind_reliability_study(
            model, mc_samples=600, samples_per_level=150, seed=3
        )
        self.assertTrue(study.form_converged)
        self.assertTrue(study.subset_converged)
        self.assertGreater(study.mc_probability, 0.003)
        self.assertLess(study.mc_probability, 0.06)
        # FORM against Monte Carlo: 3 standard errors of the MC estimate.
        self.assertLess(
            abs(study.form_probability - study.mc_probability),
            3.0 * study.mc_standard_error,
        )
        # Subset against Monte Carlo: combined band, c.o.v. upper bound.
        combined = math.hypot(
            study.mc_standard_error, study.subset_cov_upper * study.subset_probability
        )
        self.assertLess(
            abs(study.subset_probability - study.mc_probability), 3 * combined
        )
        # The same limit state in FORM terms.
        form = form_hlrf(wind_load_limit_state(model))
        self.assertAlmostEqual(form.probability, study.form_probability, places=12)
        wind_share = sum(v for k, v in form.importance_factors.items() if "wind" in k)
        self.assertGreater(wind_share, 0.95)
        self.assertGreater(form.beta, 1.5)

    def test_failure_side_of_the_limit_state(self) -> None:
        """A higher limit lowers the failure probability (FORM)."""
        low = form_hlrf(wind_load_limit_state(_model(3300.0)))
        high = form_hlrf(wind_load_limit_state(_model(4000.0)))
        self.assertGreater(low.probability, high.probability)
        self.assertGreater(high.beta, low.beta)


class SensitivityTests(unittest.TestCase):
    """Sobol ranking of wind coordinates against vehicle dispersions."""

    def test_wind_coordinates_dominate(self) -> None:
        """Wind coordinates carry the variance; the vehicle inputs do not."""
        model = _model()
        result = sobol_analysis(model, "max_q_alpha", 32, seed=9)
        first = dict(
            zip(
                result.input_names,
                (i.estimate for i in result.first_order),
                strict=True,
            )
        )
        total = dict(
            zip(
                result.input_names,
                (i.estimate for i in result.total_effect),
                strict=True,
            )
        )
        wind = sum(total[n] for n in _FIELD.coordinate_names)
        vehicle = total["thrust_scale"] + total["drag_scale"]
        self.assertGreater(wind, 0.8)
        self.assertLess(vehicle, 0.3)
        # The largest first-order index belongs to the first two (long-wave)
        # cosine coordinates, which move the jet-altitude wind most.
        best = max(first, key=lambda k: first[k])
        self.assertIn(best, ("wind_1", "wind_2"))


class ExtremeValueTests(unittest.TestCase):
    """GPD and GEV tail fits against the simulation estimate."""

    def test_tail_fit_against_form_and_monte_carlo(self) -> None:
        """GPD and GEV exceedance estimates are within a factor of 3 of FORM."""
        model = _model()
        study = wind_extreme_study(model, _LIMIT, samples=400, block_size=20, seed=5)
        self.assertEqual(study.samples, 400)
        self.assertTrue(study.gpd_converged)
        self.assertAlmostEqual(study.exceedances, 40, delta=1)
        self.assertGreater(study.threshold, study.quantiles[1] - 1e-9)
        self.assertLess(study.quantiles[0], study.quantiles[1])
        self.assertLess(study.quantiles[1], study.quantiles[2])
        form = form_hlrf(wind_load_limit_state(model)).probability
        for estimate in (study.gpd_probability, study.gev_probability):
            self.assertGreater(estimate, form / 3.0)
            self.assertLess(estimate, form * 3.0)
        # The raw empirical frequency has 8 +/- 3 hits at this size.
        self.assertLess(abs(study.empirical_probability - form), 0.03)

    def test_failure_cases(self) -> None:
        """Argument checking."""
        model = _model()
        with self.assertRaises(TypeError):
            wind_extreme_study("m", 1000.0, samples=200)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            wind_extreme_study(model, 1000.0, samples=50)
        with self.assertRaises(ValueError):
            wind_extreme_study(model, 1000.0, samples=200, threshold_quantile=0.3)
        with self.assertRaises(ValueError):
            wind_extreme_study(model, 1000.0, samples=200, block_size=50)
        with self.assertRaises(ValueError):
            # The limit lies below the POT threshold (about 3.3e3 Pa rad).
            wind_extreme_study(model, 500.0, samples=200, seed=1)
        with self.assertRaises(ValueError):
            wind_extreme_study(model, -1.0, samples=200)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Larger studies at a small exceedance probability (synthetic limit)."""

    def test_small_probability_study(self) -> None:
        """FORM, subset, Monte Carlo and EVT at limit 4000 Pa rad; Sobol."""
        model = _model(4000.0)
        study = wind_reliability_study(
            model, mc_samples=6000, samples_per_level=400, seed=21
        )
        print(f"\nslow reliability: {study}")
        combined = math.hypot(
            study.mc_standard_error, study.subset_cov_upper * study.subset_probability
        )
        self.assertLess(
            abs(study.subset_probability - study.mc_probability), 3 * combined
        )
        extreme = wind_extreme_study(
            model, 4000.0, samples=6000, block_size=30, seed=23
        )
        print(f"slow extreme: {extreme}")
        self.assertLess(extreme.gpd_probability, 5.0 * max(study.mc_probability, 1e-4))
        result = sobol_analysis(model, "max_q_alpha", 256, seed=9)
        for name, i1, it in zip(
            result.input_names, result.first_order, result.total_effect, strict=True
        ):
            print(f"slow sobol {name}: S1={i1.estimate:.3f} ST={it.estimate:.3f}")


if __name__ == "__main__":
    unittest.main()
