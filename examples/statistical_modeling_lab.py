"""Statistical modeling laboratory: distributions, regression, time series, survival.

Four classical tasks on seeded synthetic data, each with its diagnostic
figure: three suppliers' strength distributions compared as violins with
box plots, a multiple regression with residual diagnostics, an AR(2) series
identified from its correlogram and forecast with exact intervals, and two
Kaplan-Meier survival curves compared by the log-rank test.
"""

from __future__ import annotations

import math
from pathlib import Path

from probstat.foundations import RandomStream
from probstat.inference import fit_linear_model
from probstat.inference.survival import kaplan_meier, log_rank_test, survival_data
from probstat.timeseries import (
    acf,
    fit_arima,
    forecast_arima,
    pacf,
    simulate_arma,
    white_noise_band,
)
from probstat.visualization import save_figure
from probstat.visualization.statistics_plots import (
    plot_correlogram,
    plot_distribution_comparison,
    plot_forecast,
    plot_regression_diagnostics,
    plot_survival_curves,
)


def main() -> None:
    """Fit the three models and write their figures.

    References:
        [REF-HINTZE-NELSON-1998], [REF-MONTGOMERY-PECK-VINING-2021],
        [REF-BROCKWELL-DAVIS-2016],
        [REF-KAPLAN-MEIER-1958], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    output = Path(__file__).parent / "generated"
    root = RandomStream(2026)

    rng = root.child("suppliers").generator()
    strengths = {
        "Supplier A": [rng.gauss(52.0, 2.0) for _ in range(150)],
        "Supplier B": [48.0 + rng.lognormvariate(1.0, 0.6) for _ in range(150)],
        "Supplier C": [
            rng.gauss(50.0, 1.2) if rng.random() < 0.7 else rng.gauss(55.0, 1.2)
            for _ in range(150)
        ],
    }
    for name, values in strengths.items():
        below = sum(v < 49.0 for v in values)
        print(f"{name}: {below} of {len(values)} below the 49 MPa limit")
    axes = plot_distribution_comparison(
        strengths,
        value_label="Tensile strength (MPa)",
        reference=49.0,
        reference_label="lower spec limit",
    )
    save_figure(axes.figure, output / "modeling-distribution-comparison.png")
    plt.close(axes.figure)

    rng = root.child("regression").generator()
    rows, response = [], []
    for _ in range(120):
        x1, x2 = rng.uniform(0.0, 10.0), rng.uniform(0.0, 5.0)
        rows.append((1.0, x1, x2))
        response.append(2.0 + 0.8 * x1 - 1.5 * x2 + rng.gauss(0.0, 1.0))
    model = fit_linear_model(rows, response, column_names=("intercept", "x1", "x2"))
    for name, estimate, error in zip(
        model.column_names, model.coefficients, model.standard_errors, strict=True
    ):
        print(f"{name:9s} {estimate:8.4f} (SE {error:.4f})")
    print(f"R^2={model.r_squared:.4f}, residual SE={model.residual_standard_error:.4f}")
    figure = plot_regression_diagnostics(
        model.fitted_values, model.studentized_residuals
    )
    save_figure(figure, output / "modeling-regression-diagnostics.png")
    plt.close(figure)

    series = simulate_arma((0.6, -0.3), (), 300, seed=root.child("arma"))
    band = white_noise_band(len(series))
    axes = plot_correlogram(pacf(series, 20), band, kind="PACF")
    save_figure(axes.figure, output / "modeling-pacf.png")
    plt.close(axes.figure)
    axes = plot_correlogram(acf(series, 20), band, kind="ACF")
    save_figure(axes.figure, output / "modeling-acf.png")
    plt.close(axes.figure)
    fit = fit_arima(series, (2, 0, 0))
    print(
        f"AR(2) fit: ar={tuple(round(a, 4) for a in fit.ar)} "
        f"SE={tuple(round(e, 4) for e in fit.standard_errors[:2])} AIC={fit.aic:.2f}"
    )
    forecast = forecast_arima(fit, 24)
    axes = plot_forecast(
        series[-80:],
        forecast.mean,
        forecast.lower,
        forecast.upper,
        level=forecast.level,
        name="Series",
    )
    save_figure(axes.figure, output / "modeling-forecast.png")
    plt.close(axes.figure)

    rng = root.child("survival").generator()

    def cohort(scale: float) -> tuple[list[float], list[bool]]:
        """Return Weibull lifetimes censored by a uniform follow-up.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 3].

        """
        times, events = [], []
        for _ in range(80):
            life = scale * (-math.log(1.0 - rng.random())) ** (1.0 / 1.5)
            follow_up = rng.uniform(2.0, 12.0)
            times.append(min(life, follow_up))
            events.append(life <= follow_up)
        return times, events

    control = survival_data(*cohort(5.0))
    treated = survival_data(*cohort(8.0))
    test = log_rank_test(control, treated)
    print(f"log-rank chi2={test.statistic:.3f}, p={test.p_value:.4f}")
    curves = {}
    for label, data in (("control", control), ("treated", treated)):
        estimate = kaplan_meier(data)
        curves[label] = (
            estimate.times,
            estimate.survival,
            estimate.lower,
            estimate.upper,
        )
    axes = plot_survival_curves(
        curves,
        test_label=f"log-rank p = {test.p_value:.3g}",
        time_label="Time (years)",
    )
    save_figure(axes.figure, output / "modeling-survival.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
