"""Reproducible laboratory: scoring, calibrating, and shift-testing forecasts.

A synthetic process has a known heteroscedastic predictive law
``Y_i ~ Normal(m_i, s_i)``. Four forecasters are compared on the same 2000
cases with proper scores, PIT diagnostics, interval coverage, and sharpness:

* ``ideal``: the true law ``Normal(m_i, s_i)``;
* ``overconfident``: ``Normal(m_i, s_i / 2)`` (right centre, too narrow);
* ``climatology``: one unconditional ``Normal`` fitted to all outcomes
  (calibrated on average but not sharp);
* ``ensemble``: 20 members drawn from the ideal law.

The binary event ``Y > 0`` is then scored with the Brier decomposition and a
reliability table, and a calibration period is compared with a deployment
period without a shift and with two sizes of shift. Only numbers are
printed; no figures are produced.
"""

from __future__ import annotations

import math
import random
import time

from probstat.distributions import Normal
from probstat.inference import kolmogorov_smirnov_two_sample_test
from probstat.validation import (
    brier_decomposition,
    crps_ensemble,
    crps_normal,
    dawid_sebastiani_score,
    energy_distance_test,
    ensemble_pit,
    log_score,
    mmd_test,
    pit_coverage,
    pit_histogram,
    pit_values,
    reliability_diagram,
    sharpness_summary,
    wasserstein_distance,
    weighted_interval_score,
)

CASES = 2000
MEMBERS = 20
SEED = 20261012


def simulate(
    generator: random.Random, size: int, shift: float = 0.0
) -> tuple[list[float], list[float], list[float]]:
    """Draw case means, scales, and outcomes of the synthetic process.

    Args:
        generator: Seeded generator.
        size: Number of cases.
        shift: Additive shift of the outcomes (deployment drift).

    Returns:
        Means ``m_i``, scales ``s_i``, and outcomes ``y_i``.

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

    """
    means = [generator.uniform(-2.0, 2.0) for _ in range(size)]
    scales = [0.5 + generator.random() for _ in range(size)]
    outcomes = [
        generator.gauss(m, s) + shift for m, s in zip(means, scales, strict=True)
    ]
    return means, scales, outcomes


def mean(values: list[float]) -> float:
    """Return the arithmetic mean.

    References:
        [REF-NIST-EDA].

    """
    return math.fsum(values) / len(values)


def main() -> None:
    """Score four forecasters, the event forecasts, and two shift scenarios.

    References:
        [REF-GNEITING-RAFTERY-2007], [REF-GNEITING-BALABDAOUI-RAFTERY-2007],
        [REF-SZEKELY-RIZZO-2013], [REF-GRETTON-ETAL-2012].

    """
    start = time.perf_counter()
    generator = random.Random(SEED)
    means, scales, outcomes = simulate(generator, CASES)
    clim_mu = mean(outcomes)
    clim_sigma = math.sqrt(mean([(y - clim_mu) ** 2 for y in outcomes]))
    laws = {
        "ideal": [Normal(m, s) for m, s in zip(means, scales, strict=True)],
        "overconfident": [
            Normal(m, 0.5 * s) for m, s in zip(means, scales, strict=True)
        ],
        "climatology": [Normal(clim_mu, clim_sigma) for _ in means],
    }
    ensembles = [
        [generator.gauss(m, s) for _ in range(MEMBERS)]
        for m, s in zip(means, scales, strict=True)
    ]

    print(f"[1/4] Scoring four forecasters on {CASES} cases (seed {SEED}) ...")
    rows: list[tuple[str, float, float, float, list[float], list[float]]] = []
    for name, forecasts in laws.items():
        crps = mean(
            [
                crps_normal(y, f.mu, f.sigma)
                for f, y in zip(forecasts, outcomes, strict=True)
            ]
        )
        logs = mean([log_score(f, y) for f, y in zip(forecasts, outcomes, strict=True)])
        dss = mean(
            [
                dawid_sebastiani_score(y, f.mu, f.sigma**2)
                for f, y in zip(forecasts, outcomes, strict=True)
            ]
        )
        u = list(pit_values(forecasts, outcomes))
        widths = list(sharpness_summary(forecasts, levels=(0.9,)).mean_widths)
        rows.append((name, crps, logs, dss, u, widths))
    ens_crps = mean(
        [crps_ensemble(e, y) for e, y in zip(ensembles, outcomes, strict=True)]
    )
    ens_fair = mean(
        [
            crps_ensemble(e, y, fair=True)
            for e, y in zip(ensembles, outcomes, strict=True)
        ]
    )
    ens_dss = mean(
        [
            dawid_sebastiani_score(
                y,
                mean(e),
                mean([(x - mean(e)) ** 2 for x in e]) * MEMBERS / (MEMBERS - 1),
            )
            for e, y in zip(ensembles, outcomes, strict=True)
        ]
    )
    ens_u = list(ensemble_pit(ensembles, outcomes, seed=SEED))
    ens_widths = list(sharpness_summary(ensembles, levels=(0.9,)).mean_widths)
    table = [*rows, ("ensemble", ens_crps, math.nan, ens_dss, ens_u, ens_widths)]
    print("  Proper scores (mean over cases, smaller is better)")
    print(f"  {'forecaster':<14}{'CRPS':>8}{'LogS':>8}{'DSS':>8}")
    for name, crps, logs, dss, _, _ in table:
        log_text = "n/a" if math.isnan(logs) else f"{logs:.4f}"
        print(f"  {name:<14}{crps:>8.4f}{log_text:>8}{dss:>8.4f}")
    print(f"  ensemble fair CRPS {ens_fair:.4f} vs ideal closed form {rows[0][1]:.4f}")
    print("  Calibration and sharpness")
    print(
        f"  {'forecaster':<14}{'PIT p':>10}{'edge':>7}"
        f"{'cov50':>7}{'cov90':>7}{'w90':>7}"
    )
    for name, _, _, _, u, widths in table:
        histogram = pit_histogram(u, bins=10)
        cov50, cov90 = pit_coverage(u, levels=(0.5, 0.9))
        print(
            f"  {name:<14}{histogram.chi_square.p_value:>10.3g}"
            f"{histogram.edge_ratio:>7.2f}{cov50.rate:>7.3f}{cov90.rate:>7.3f}"
            f"{widths[0]:>7.3f}"
        )

    print("\n[2/4] Quantile forecast of the ideal law (50/80/95 % intervals) ...")
    alphas = (0.5, 0.2, 0.05)
    wis = []
    for law, y in zip(laws["ideal"], outcomes, strict=True):
        intervals = [(a, law.quantile(a / 2), law.quantile(1 - a / 2)) for a in alphas]
        wis.append(weighted_interval_score(y, law.mu, intervals).score)
    print(f"  mean WIS {mean(wis):.4f} (mean CRPS {rows[0][1]:.4f})")

    print("\n[3/4] Event Y > 0: Brier decomposition by 10 bins ...")
    events = [y > 0.0 for y in outcomes]
    print(
        f"  {'forecaster':<14}{'BS':>8}{'REL':>8}{'RES':>8}"
        f"{'UNC':>8}{'WBV':>8}{'WBC':>8}"
    )
    residual = 0.0
    for name, forecasts in laws.items():
        probabilities = [f.sf(0.0) for f in forecasts]
        parts = brier_decomposition(probabilities, events, bins=10)
        residual = max(residual, abs(parts.identity_residual))
        print(
            f"  {name:<14}{parts.score:>8.4f}{parts.reliability:>8.4f}"
            f"{parts.resolution:>8.4f}{parts.uncertainty:>8.4f}"
            f"{parts.within_bin_variance:>8.4f}{parts.within_bin_covariance:>8.4f}"
        )
    print(f"  BS = REL - RES + UNC + WBV - WBC; largest residual {residual:.1e}")
    for name in ("ideal", "overconfident"):
        probabilities = [f.sf(0.0) for f in laws[name]]
        diagram = reliability_diagram(
            probabilities,
            events,
            bins=5,
            consistency_level=0.95,
            resamples=500,
            seed=SEED,
        )
        print(f"  Reliability table, {name} (95 % consistency band)")
        print(f"    {'bin':<12}{'n':>6}{'mean f':>9}{'freq':>8}{'band':>18}")
        for item in diagram.bins:
            span = f"[{item.lower:.1f}, {item.upper:.1f})"
            band = f"[{item.consistency_lower:.3f}, {item.consistency_upper:.3f}]"
            print(
                f"    {span:<12}{item.count:>6}{item.mean_forecast:>9.3f}"
                f"{item.observed_frequency:>8.3f}{band:>18}"
            )
        print(f"    bins outside band: {list(diagram.outside_bands)}")

    print("\n[4/4] Shift: calibration vs deployment period (n=300 each) ...")
    _, _, calibration = simulate(generator, 300)
    centre = mean(calibration)
    marginal_sd = math.sqrt(mean([(y - centre) ** 2 for y in calibration]))
    print(f"  marginal outcome sd in the calibration period: {marginal_sd:.3f}")
    print(
        f"  {'scenario':<12}{'KS p':>10}{'energy p':>10}{'MMD p':>8}{'W1':>7}{'W2':>7}"
    )
    detected: list[str] = []
    for label, shift in (("no shift", 0.0), ("+0.4", 0.4), ("+0.8", 0.8)):
        _, _, deployment = simulate(generator, 300, shift)
        ks = kolmogorov_smirnov_two_sample_test(calibration, deployment)
        energy = energy_distance_test(
            calibration, deployment, permutations=499, seed=SEED
        )
        mmd = mmd_test(calibration, deployment, permutations=199, seed=SEED)
        w1 = wasserstein_distance(calibration, deployment)
        w2 = wasserstein_distance(calibration, deployment, order=2)
        print(
            f"  {label:<12}{ks.p_value:>10.3g}{energy.p_value:>10.3g}"
            f"{mmd.p_value:>8.3g}{w1:>7.3f}{w2:>7.3f}"
        )
        if max(ks.p_value, energy.p_value, mmd.p_value) < 0.05:
            detected.append(label)
    best = min(table, key=lambda row: row[1])[0]
    print(
        f"\nDone in {time.perf_counter() - start:.1f} s: lowest CRPS '{best}'; "
        f"shift detected by all three tests at 5 %: {detected or 'none'}. "
        "Next: docs/foundations/forecast-calibration.md."
    )


if __name__ == "__main__":
    main()
