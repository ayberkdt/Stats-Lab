"""Hierarchical composition: Markov subsystems and subsystems as blocks."""

from __future__ import annotations

import math
import unittest

from probstat.systems import (
    ExponentialComponent,
    MarkovComponent,
    RepairableComponent,
    SubsystemComponent,
    SystemModel,
    parallel,
    repairable_k_out_of_n,
    series,
    single_repairable_unit,
    standby_system,
)


class MarkovComponentTests(unittest.TestCase):
    """A CTMC subsystem behaves as an exact block."""

    def test_single_repairable_unit_matches_closed_form_component(self) -> None:
        """Markov block equals the closed-form repairable component."""
        lam, mu = 2e-3, 0.1
        markov = MarkovComponent(single_repairable_unit(lam, mu))
        closed = RepairableComponent(lam, mu)
        for t in (0.0, 1.0, 50.0, 500.0):
            self.assertAlmostEqual(markov.availability(t), closed.availability(t), 13)
            self.assertAlmostEqual(markov.reliability(t), math.exp(-lam * t), 13)
            self.assertAlmostEqual(
                markov.unavailability(t), closed.unavailability(t), 13
            )
        self.assertLess(markov.last_error_bound, 1e-11)

    def test_cold_standby_block_in_series(self) -> None:
        """Series of a cold-standby pair and an exponential unit factorises."""
        lam, rate_b = 1e-3, 5e-4
        standby = MarkovComponent(
            standby_system(spares=1, failure_rate=lam, standby_failure_rate=0.0)
        )
        system = SystemModel(
            series("pumps", "valve"),
            {"pumps": standby, "valve": ExponentialComponent(rate_b)},
        )
        t = 1000.0
        # Cold standby, perfect switching: R = e^{-lam t}(1 + lam t).
        expected = math.exp(-lam * t) * (1.0 + lam * t) * math.exp(-rate_b * t)
        self.assertAlmostEqual(system.reliability(t), expected, 12)
        self.assertAlmostEqual(system.unreliability(t), 1.0 - expected, 12)

    def test_repairable_two_of_three_availability(self) -> None:
        """Steady regime of a 2-out-of-3 repairable block inside a system."""
        block = MarkovComponent(repairable_k_out_of_n(3, 2, 1e-2, 1.0))
        system = SystemModel(parallel("train"), {"train": block})
        a = system.availability(1e4)
        self.assertAlmostEqual(
            a, block.model.steady_state_availability(), 10
        )

    def test_rejects_wrong_types(self) -> None:
        """Only Markov reliability models and valid tolerances are accepted."""
        with self.assertRaises(TypeError):
            MarkovComponent(ExponentialComponent(1.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MarkovComponent(single_repairable_unit(1.0, 1.0), tolerance=0.5)
        with self.assertRaises(ValueError):
            MarkovComponent(single_repairable_unit(1.0, 1.0)).reliability(-1.0)


class SubsystemComponentTests(unittest.TestCase):
    """A subsystem block reproduces the flat structure exactly."""

    def test_hierarchy_equals_flat_structure(self) -> None:
        """Modular decomposition gives the flat-system reliability."""
        rates = {"a": 1e-3, "b": 2e-3, "c": 5e-4, "d": 1e-4}
        models = {name: ExponentialComponent(rate) for name, rate in rates.items()}
        flat = SystemModel(
            series(parallel("a", "b"), parallel("c", "d")), models
        )
        left = SubsystemComponent(
            SystemModel(parallel("a", "b"), {k: models[k] for k in "ab"})
        )
        right = SubsystemComponent(
            SystemModel(parallel("c", "d"), {k: models[k] for k in "cd"})
        )
        nested = SystemModel(series("L", "R"), {"L": left, "R": right})
        for t in (10.0, 300.0, 3000.0):
            self.assertAlmostEqual(nested.reliability(t), flat.reliability(t), 14)
            self.assertAlmostEqual(nested.availability(t), flat.availability(t), 14)
        mttf_flat = flat.mttf().value
        self.assertAlmostEqual(nested.mttf().value / mttf_flat, 1.0, 8)

    def test_requires_system_model(self) -> None:
        """Only system models are wrapped."""
        with self.assertRaises(TypeError):
            SubsystemComponent(ExponentialComponent(1.0))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
