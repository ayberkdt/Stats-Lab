"""Statistical modeling laboratory: distributions, regression, time series, survival.

Four classical tasks on seeded synthetic data, each with its diagnostic
figure: three suppliers' strength distributions compared as violins with
box plots, a multiple regression with residual diagnostics, an AR(2) series
identified from its correlogram and forecast with exact intervals, and two
Kaplan-Meier survival curves compared by the log-rank test. Run with
``PYTHONPATH=src python examples/statistical_modeling_lab.py``; the figures
are written to ``examples/generated/``.

Definitions:
    supplier strengths: 150 tensile strengths per supplier (normal, shifted
        lognormal, and a 70/30 normal mixture), checked against a 49 MPa
        lower specification limit.
    ``y = b0 + b1 x1 + b2 x2 + e``: the regression model, true coefficients
        ``(2, 0.8, -1.5)`` and unit noise standard deviation.
    AR(2): ``x_t = 0.6 x_(t-1) - 0.3 x_(t-2) + e_t``, 300 observations.
    ``S(t)``: survival function of Weibull lifetimes (shape 1.5, scales 5
        and 8) under uniform censoring on ``[2, 12]``.

Assumptions:
    - Regression errors are independent, homoscedastic, and normal, as the
      t-based standard errors require.
    - The simulated series is stationary and Gaussian, so the exact Kalman
      likelihood and the Gaussian forecast intervals apply.
    - Censoring is independent of the lifetimes (non-informative), as
      Kaplan-Meier and the log-rank test require.
    - Each task uses its own child stream of the root seed, so the tasks do
      not share random numbers.

Equations:
    EQ-RNG-SPLITMIX-001: derivation of the four child streams.
    EQ-STAT-KDE-001: kernel density of the violins.
    EQ-STAT-QUANT-001: quartiles of the box plots.
    EQ-LM-OLS-001: least-squares coefficients.
    EQ-LM-INF-001: coefficient standard errors.
    EQ-LM-DIAG-002: studentized residuals in the diagnostics figure.
    EQ-TS-SIM-001: ARMA simulation recursion.
    EQ-TS-ACF-001: sample autocorrelation.
    EQ-TS-PACF-001: partial autocorrelation by Durbin-Levinson.
    EQ-TS-BAND-001: white-noise correlogram band.
    EQ-TS-LIK-001: exact ARMA likelihood maximized by ``fit_arima``.
    EQ-INF-IC-001: AIC of the AR(2) fit.
    EQ-TS-FC-001: exact multi-step forecast and interval.
    EQ-SURV-KM-001: Kaplan-Meier product-limit estimator.
    EQ-SURV-KM-002: Greenwood variance.
    EQ-SURV-KM-003: log-log pointwise interval of the survival curves.
    EQ-SURV-LR-001: log-rank statistic.

Diagnostics:
    - Counts below the specification limit per supplier, read with the
      violin shapes (the mixture of Supplier C has the heaviest lower tail).
    - Studentized residuals against fitted values for the regression; the
      estimates can be compared with the true coefficients.
    - The PACF cutting off after lag 2 identifies the AR order; the fitted
      coefficients can be compared with ``(0.6, -0.3)``.
    - The log-rank statistic and the separated survival curves with their
      pointwise intervals.

Cost and error:
    - No model evaluations beyond the fits; all error is sampling error of
      the synthetic data. Regression and AR coefficients are printed with
      their standard errors; the survival intervals are pointwise 95%.
    - The white-noise band is pointwise, so about one lag in twenty crosses
      it by chance.

References:
    REF-HINTZE-NELSON-1998: violin plots.
    REF-SILVERMAN-1986: kernel density estimation.
    REF-MONTGOMERY-PECK-VINING-2021: linear regression analysis.
    REF-BROCKWELL-DAVIS-2016: time series and forecasting.
    REF-BOX-JENKINS-2015: ARIMA identification and estimation.
    REF-KAPLAN-MEIER-1958: product-limit estimator.
    REF-GREENWOOD-1926: Greenwood variance.
    REF-MANTEL-1966: log-rank test.
    REF-KLEIN-MOESCHBERGER-2003: survival analysis.

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

# Root seed; each task draws from its own child stream.
SEED = 2026


def main() -> None:
    """Fit the three models and write their figures.

    References:
        [REF-HINTZE-NELSON-1998], [REF-MONTGOMERY-PECK-VINING-2021],
        [REF-BROCKWELL-DAVIS-2016],
        [REF-KAPLAN-MEIER-1958], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    output = Path(__file__).parent / "generated"
    root = RandomStream(SEED)

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
        value_label="Tensile strength [MPa]",
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
        time_label="Time to event [years]",
    )
    save_figure(axes.figure, output / "modeling-survival.png")
    plt.close(axes.figure)


if __name__ == "__main__":
    main()
