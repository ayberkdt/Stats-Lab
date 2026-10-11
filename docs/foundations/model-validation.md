# Model Validation and Prediction Credibility

## What validation asks

Verification asks whether the equations are solved correctly. **Validation**
asks whether the model, as solved, agrees with reality closely enough *for a
stated use*: one quantity of interest (QoI), one set of operating conditions,
one tolerance. Prediction then asks what uncertainty to assign at the
conditions where the model will actually be used, which may lie outside
everything that was tested.

`probstat.validation` keeps these questions apart. Its assessment never says a
model is valid in general. The strongest statement it makes is: *the model's
predictions of this QoI are consistent (or inconsistent) with these data at
tolerance X, within validation domain D.*

## Records

- `ValidationExperiment` stores a name, the measured QoI values (one value or
  replicates), the named conditions (load, temperature, ...), the standard
  uncertainty of one measurement `sigma_m`, and a provenance text. The
  experimental value is `D = mean(y)`. Its standard uncertainty `u_D` defaults
  to `sigma_m / sqrt(n)`, which assumes independent random errors and no shared
  systematic error. Pass `data_uncertainty` when that assumption does not
  hold.
- `ValidationDomain.from_experiments` collects the condition vectors.
  `check(condition)` reports bounding-box membership, exact convex-hull
  membership (a phase-one simplex feasibility problem in range-scaled
  coordinates), and the scaled Euclidean distance to the nearest validation
  point [EQ-VAL-DOM-001]. A condition is flagged as extrapolation when it lies
  outside the hull, or farther than `distance_threshold` (default `0.25` in
  scaled units) from every tested condition.

For a triangle of conditions `(0,0), (1,0), (0,1)`, the point `(0.8, 0.8)` is
inside the bounding box but outside the hull, so it is extrapolation. The
point `(0.3, 0.3)` is inside the hull but `0.283` from the nearest vertex, so
it is flagged as sparse coverage at the default threshold.

## Predictions at experimental conditions

`predict_at_conditions(model, quantity, conditions, size, seed=...)` merges
the conditions over `model.fixed` and calls `propagate`. A condition may not
name a random input, because that would silently replace part of the input
law. User-supplied samples, for example a posterior push-forward from
`probstat.inverse.predict`, go to `assess_predictions` directly.

By default the assessment adds seeded `N(0, sigma_m^2)` noise to the latent
predictive samples before comparing distributions with data. The comparison
then concerns what a *measurement* would look like. The ASME comparison and
the Bayes factor use the latent samples and account for noise analytically.

## Metrics

No metric is a universal score. Each answers a narrower question.

| Metric | Question | Unit | Equation |
|---|---|---|---|
| Area metric | How far apart are the predictive CDF and the data ECDF? | QoI | [EQ-VAL-AREA-001] |
| u-pooling | Are pooled observations distributed like their predictions? | probability | [EQ-VAL-UPOOL-001], [EQ-VAL-UPOOL-002] |
| Mahalanobis | Are residuals plausible under a Gaussian summary? | dimensionless | [EQ-VAL-MAHAL-001] |
| Reliability | How likely is agreement within `epsilon`? | probability | [EQ-VAL-REL-001] |
| ASME V&V 20 | What interval contains the model error? | QoI | [EQ-VAL-ASME-001] |
| Bayes factor | Does the model's predictive density beat a declared alternative? | log odds | [EQ-VAL-BF-001] |

### Area metric (Ferson, Oberkampf, and Ginzburg)

`d = integral |F(y) - S_n(y)| dy`. With both CDFs as step functions, the
integral is a finite sum of rectangles, so `area_metric` is exact. These hand
values are checked in `tests/test_model_validation.py`:

- identical samples: `d = 0`;
- point mass at `0` against data `{1, 3}`: `d = (1 + 3)/2 = 2`;
- prediction `{0, 1, 2, 3}` against data `{1.5, 2.5}`:
  `0.25 + 0.25 + 0 + 0.125 + 0.125 = 0.75`;
- `N(1, 2^2)` against the single observation `y = 2`: the analytic value is
  `E|X - y| = s[z(2 Phi(z) - 1) + 2 phi(z)] = 1.791186` with `z = 0.5`. A
  20 000-point quantile grid gives `1.791171`, and the test tolerance is
  `2e-3`. For `z = 0` the value is `2 s phi(0) = 0.797885 s`. The grid gave
  `0.797877 s`.

The area metric keeps the units of the QoI, so a tolerance can be stated in
engineering terms. It also has a floor: with one observation per experiment,
even a perfect model has an area near `0.8` predictive standard deviations.
Choose the tolerance with that floor in mind.

### u-pooling

Each observation becomes `u = F_i(y)` through the predictive ECDF of its own
experiment. If the predictions are right, the pooled `u` values are close to
`Uniform(0, 1)`, however much the conditions differ between experiments.
`u_pooling` reports the exact area to the uniform CDF (from `0` to `1/2`) and a
one-sample Kolmogorov-Smirnov test [EQ-INF-GOF-KS-001]. A single `u = 1/2`
gives area `0.25`. A single `u = 0` gives the maximum, `0.5`.

The seeded test uses 12 experiments with 4 observations each. Data drawn from
the predictive laws give area `0.0374`, KS `D = 0.117`, and `p = 0.51`. A
one-sigma bias in every experiment gives area `0.298`, `D = 0.503`, and
`p = 2.3e-11`.

### Gaussian residuals and the reliability metric

`mahalanobis_metric` computes `D^2 = r' C^-1 r` and the chi-square tail. The
covariance `C` must include the predictive, measurement, and any discrepancy
variance. The p-value is exact only for a Gaussian predictive law. For
residuals `(1, 2)` with identity covariance, `D^2 = 5` and
`p = e^-2.5 = 0.0821`.

`reliability_metric` estimates `P(|Y_pred - Y_obs| < epsilon)`. For predictions
`{0, 1, 2, 3}`, the observation `1`, and `epsilon = 1.5`, it gives `3/4`.

### ASME V&V 20 comparison

`E = S - D` and `u_val = sqrt(u_num^2 + u_input^2 + u_D^2)`. The model error
lies in `[E - k u_val, E + k u_val]`. Under V&V 20 this is an *estimate of
model error*, not a pass/fail test. Worked numbers: `S = 10.5`, `D = 10.0`,
`u_num = 0.3`, `u_input = 0.4`, `u_D = 1.2` give `E = 0.5`, `u_val = 1.3`, and,
with `k = 2`, `delta_model in [-2.1, 3.1]`. The error is not resolved. With
`S = 7`, `D = 10`, and `u_val = 0.5`, `|E| = 3 > 1`, so the model error is
resolved and approximately `-3`.

A `NumericalErrorBudget` is a conservative absolute bound, not a standard
uncertainty. `budget_standard_uncertainty` converts it with the explicit GUM
rectangular assumption `u_num = B / sqrt(3)` [EQ-VAL-ASME-002].

### Bayes factor

`validation_bayes_factor` compares the model predictive density
`p_M(y) = (1/S) sum_s N(y; y_s, h^2)` with an alternative `p_A` that the user
declares:

- `ValidationAlternative.uniform(a, b)`: each observation is only known to lie
  in `[a, b]`;
- `ValidationAlternative.normal(m, t)`: a model-independent normal;
- `ValidationAlternative.inflated(c)`: a normal centered at the predictive mean
  with `c` times the predictive standard deviation. The assessment uses this by
  default, with `c = 5`.

With known measurement noise `h = sigma_m`, `p_M` is a Monte Carlo estimate of
the exact predictive density of a noisy observation. Otherwise `h` is
Silverman's bandwidth. The result is read on the Kass-Raftery scale of
`2 log B`. Bayes factors depend strongly on how broad the alternative is: a
very wide alternative favors almost any model (the Jeffreys-Lindley effect).
That is why the alternative is always printed next to `log B`.

## The assessment report

`assess_validation(model, quantity, experiments, tolerance=X, seed=s)` draws
experiment `i` from stream `root.child("validation", "prediction", i)` and its
noise from `root.child("validation", "noise", i)`. One seed therefore
reproduces every number. The verdict is **consistent** only if:

1. every per-experiment area metric is `<= X`, and
2. the u-pooling KS p-value is `>= significance` (default `0.05`).

The ASME, reliability, Mahalanobis, and Bayes-factor results are reported as
evidence. They do not change the verdict.

Worked example (seed `3`, 400 samples per experiment): the spring model
`y = k * load`, `k ~ N(2, 0.1^2)`, against four synthetic experiments at loads
`1, 2, 3, 4`, six replicates each, with `sigma_m = 0.05`.

| Quantity | Unbiased data | Data biased by +1 mm |
|---|---|---|
| maximum area metric (tolerance 0.4 mm) | 0.3005 | 1.295 |
| u-pooling area / KS p-value | 0.0415 / 0.784 | 0.492 / 4.9e-20 |
| Mahalanobis `D^2` on 24 dof (p) | 38.5 (0.031) | 756.5 (3e-144) |
| summed `log B` vs inflated(5) | +20.7 | -1154 |
| ASME error resolved? | no (all four) | yes (all four) |
| verdict | consistent | inconsistent |

The unbiased case shows why the Gaussian residual check does not drive the
verdict on its own. Its p-value of `0.031` comes from one finite data set. Over
40 independent synthetic data sets, the Mahalanobis and KS checks rejected at
the 5% level 2 and 4 times, with mean p-values `0.46` and `0.53`. That is what
a calibrated test should do when the model is correct.

The summary also checks the intended-use conditions. `load = 2.5` is inside
the hull and near the tested points. `load = 8` is outside the bounding box
and is reported as extrapolation.

## Verification, validation, and prediction stay separate

`CredibilityRecord` holds three tuples:

- `VerificationEvidence`: a numerical error budget, how it was obtained, and
  an optional observed order of accuracy;
- `ValidationAssessment`: the scoped verdict and its evidence;
- `PredictionStatement`: an interval at an intended-use condition, what it
  includes, and its domain check.

`summary()` prints each section separately, followed by a list of gaps. Gaps
include a missing verification record, a prediction without a validation
record, an unchecked domain, extrapolation, or an inconsistent assessment.
The record computes no single "confidence" scalar on purpose, because
numerical error, model-data agreement, and predictive spread have different
units and different failure modes.

## What validation cannot claim

- **No general validity.** Agreement with these data, for this QoI, at this
  tolerance does not carry over to other QoIs, other regimes, or tighter
  tolerances.
- **No automatic extrapolation.** Outside the convex hull of the validation
  conditions, the evidence does not apply directly. Predictions there need
  explicit model-form uncertainty (for example an
  `AdditiveGaussianDiscrepancy` with a declared applicability scope) and
  expert justification.
- **Few data, weak tests.** With few observations, every metric has low power.
  "Consistent" can simply mean that the data cannot tell.
- **Calibration data are not validation data.** Reusing calibration
  observations for validation makes agreement look better than it is. Keep the
  data sets separate.
- **Assumptions are part of the result.** The independence of the V&V 20 error
  sources, the rectangular conversion of numerical bounds, conditional
  independence across replicates, and the choice of Bayes-factor alternative
  all affect the numbers. They are reported, not hidden.

## Sources

- [REF-FERSON-OBERKAMPF-GINZBURG-2008] — area metric and u-pooling.
- [REF-ASME-VV20-2009] — comparison error and validation uncertainty.
- [REF-ROY-OBERKAMPF-2011] — separating numerical, model-form, and input
  uncertainty in prediction.
- [REF-OBERKAMPF-ROY-2025, Chapter 12 and Part IV] — model accuracy assessment
  and predictive capability.
- [REF-LIU-CHEN-ARENDT-HUANG-2011] — comparison of validation-metric families.
- [REF-REBBA-MAHADEVAN-2008] — model reliability metric and Bayesian
  hypothesis testing for validation.
- [REF-KASS-RAFTERY-1995] — Bayes factors and their interpretation scale.
- [REF-JCGM-100-2008, Section 4.3.7] — rectangular Type B evaluation.
- [REF-SILVERMAN-1986, Section 3.4.2] — kernel bandwidth rule.
