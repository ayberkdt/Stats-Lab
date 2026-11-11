# ADR-0080: Real-data fitting case studies

- Status: Accepted
- Date: 2026-11-11

## Context

The UQ-1 roadmap item records that "real-data fitting case studies remain":
every fitting laboratory draws seeded synthetic data, so no example shows the
fitting tools on data whose generating law is unknown, rounded, contaminated,
or too short to rank families. A case study is only useful if its data can
be traced to a page of a publication, checked against a second source, and
redistributed without restriction. Two methods needed for classical count
data were also missing: the negative binomial with a real size parameter
(the library's `NegativeBinomial` takes an integer size) and an exact
interval for a Poisson mean.

## Decision

1. **Location of the data.** The records live in
   `probstat.statistics.datasets`, not in a new top-level package. A data set
   is a finite sample with units and provenance; `statistics` (layer 1)
   already owns finite samples, the module imports only
   `probstat.foundations`, and any layer from 1 upward can therefore use it.
   A separate `datasets` package would add a layer entry and a stability label
   for five frozen records. `inference` (layer 2) was rejected because the
   records are independent of any estimator. No package was created, so
   `tests/test_architecture.py` is unchanged.
2. **Record format.** `DataColumn(name, unit, values)` and `DataSet(key,
   title, description, columns, primary_source, secondary_source, licence,
   references)`, both frozen and validated (equal column lengths, distinct
   names, finite values, `REF-*` keys). `DataSet.expand_frequencies` turns a
   frequency table into one count per unit. `DATASETS` and `dataset_by_key`
   list and retrieve the records.
3. **Data sets included, each read from the primary source and checked
   against a secondary source:**

   | Record | Primary source and locator | Checked against |
   |---|---|---|
   | `BORTKIEWICZ_HORSE_KICKS` (109, 65, 22, 3, 1 corps-years with 0-4 deaths) | [REF-BORTKIEWICZ-1898], Section 12, page 25 (Internet Archive scan) | [REF-YORK-BORTKIEWICZ-TRANSCRIPTION] |
   | `RUTHERFORD_GEIGER_SCINTILLATIONS` (2608 intervals, 10097 particles) | [REF-RUTHERFORD-GEIGER-1910], table on page 701 (scan) | [REF-HANLEY-RUTHERFORD-GEIGER] |
   | `NEWCOMB_PASSAGE_TIMES` (66 values, Stigler's order) | [REF-NEWCOMB-VELOCITY-OF-LIGHT] via [REF-STIGLER-1977, Table 5] | [REF-R-MASS-NEWCOMB], value by value |
   | `LIEBLEIN_ZELEN_BALL_BEARINGS` (23 endurances) | [REF-LIEBLEIN-ZELEN-1956], worksheet on page 286 (NIST scan) | [REF-SMRD-LZBEARING] (size and unit) |
   | `STUDENT_SLEEP` (10 patients, two drugs) | [REF-STUDENT-1908], page 20, from [REF-CUSHNY-PEEBLES-1905] | [REF-R-DATASETS-SLEEP] |

   For Newcomb and Student the values were checked against the secondary
   listing; the primary pages were not available for direct reading, which
   each record states. No candidate data set was left out.
4. **Licences.** Bortkiewicz (1898), Rutherford and Geiger (1910), Student
   (1908), and Cushny and Peebles (1905) were published before 1930, so no
   United States copyright subsists; Newcomb's measurements and the
   Lieblein-Zelen paper are works of the United States Government
   (17 U.S.C. 105). In every case the values are counts or measurements,
   which are facts and are not protected by copyright.
5. **Count models** (`inference/count_models.py`): `fit_poisson_counts`
   ([EQ-INF-COUNT-001]); `fit_negative_binomial_counts` with real size,
   profile score root by Brent's method, and an explicit Poisson boundary
   (`size = inf`, `at_boundary=True`) when the divisor-`n` variance does not
   exceed the mean ([EQ-INF-COUNT-002], [REF-LEVIN-REEDS-1977]);
   `poisson_mean_interval`, Garwood's exact interval ([EQ-INF-COUNT-003],
   [REF-GARWOOD-1936]); `count_goodness_of_fit`, Pearson's statistic on cells
   pooled to at least five expected units with a parametric bootstrap that
   refits every replicate ([EQ-INF-CHI2-GOF-001], [EQ-INF-GOF-PB-001]).
6. **Robust location and diagnostics:** `statistics.trimmed_mean`
   ([EQ-STAT-TRIM-001]); `visualization.fit_plots.plot_rootogram` and
   `hanging_rootogram_bars` ([EQ-VIZ-ROOT-001], [REF-KLEIBER-ZEILEIS-2016])
   and `visualization.fit_plots.plot_qq` ([EQ-INF-GOF-QQ-001]).
7. **Laboratory.** `examples/real_data_fitting_lab.py` (seed `20261111`)
   fits Poisson against negative binomial for both count tables, normal
   against Student-t for Newcomb with a classical-versus-robust location
   table, Weibull, lognormal, and gamma for the bearings, and a paired t test
   for the sleep data; it reports AIC, BIC, Akaike weights, calibrated
   goodness of fit, and confidence intervals, and saves two rootograms, a
   Q-Q plot, and a fitted CDF. `results_markdown` renders the tables quoted
   in `docs/foundations/uncertainty-modeling-and-fitting.md`.

## Consequences

- New public names: `statistics.DataColumn`, `DataSet`, `DATASETS`,
  `dataset_by_key`, `BORTKIEWICZ_HORSE_KICKS`,
  `RUTHERFORD_GEIGER_SCINTILLATIONS`, `NEWCOMB_PASSAGE_TIMES`,
  `LIEBLEIN_ZELEN_BALL_BEARINGS`, `STUDENT_SLEEP`, `trimmed_mean`;
  `inference.CountFit`, `CountGoodnessOfFit`, `fit_poisson_counts`,
  `fit_negative_binomial_counts`, `poisson_mean_interval`,
  `count_goodness_of_fit`. The module `visualization.fit_plots` follows the
  other plot modules and is not re-exported from `probstat.visualization`.
  `docs/api-reference.md` must be regenerated for the new names.
- The case studies reproduce what the sources state: Bortkiewicz's expected
  column to the printed decimal, the agreement Rutherford and Geiger reported,
  and Stigler's observation that light trimming improves on the mean; they
  also show two negative results that synthetic labs never show: a
  negative binomial fit that collapses to its Poisson boundary, and three
  lifetime families that 23 observations cannot rank.
- The negative binomial is fitted with its mean fixed at the sample mean,
  which is the profile maximizer; a regression version (NB2 GLM) is not
  included. Two parameters are counted in AIC and BIC even on the boundary.
- The laboratory runs in about ten seconds; its bootstrap p-values carry
  Monte Carlo error of at most 0.016 (999 refits) and 0.035 (199 refits).

## Verification

`tests/test_real_data_case_studies.py`:

- data integrity against the sources: 200 corps-years, 122 deaths, mean
  0.61; Bortkiewicz's expected column 108.7, 66.3, 20.2, 4.1, 0.6, 0.1
  reproduced by `200 p(k; 0.61)` rounded to one decimal; 2608 intervals,
  10097 particles, mean 3.87, 525 intervals with three particles, and the
  printed theoretical row within 0.51 of `2608 p(k; x_bar)` for `k = 0..12`
  (the printed integers are rounded; the largest gap is 140.50 against 140);
  Newcomb's 66 values with median 27 and the same multiset as an independent
  transcription of the MASS listing; 23 ascending bearing lives with the
  worksheet's rounded median 68; mean sleep difference 1.58 h;
- closed forms: Poisson MLE equals the sample mean with
  `SE = sqrt(x_bar / n)` and the library `Poisson` log-mass; the negative
  binomial mass equals `NegativeBinomial(r, r/(r+mu))` for integer `r`; the
  score root dominates a 81 x 21 grid of `(r, mu)`; the large-`r` expansion
  of the profile likelihood; the boundary fit has the Poisson
  log-likelihood and AIC larger by two; Garwood limits for two degrees of
  freedom equal `-log(0.025)` and `-log(0.975)`; Pearson's statistic on
  Bortkiewicz's pooled cells equals the hand computation; the Weibull,
  gamma, and lognormal fits equal the direct estimators;
- published conclusions: Poisson not rejected for both count tables
  (Bortkiewicz page 25; Rutherford and Geiger page 702, including the excess
  of nearly five per cent at four particles); the Student-t model beats the
  normal for Newcomb, the normal is rejected and the 10% trimmed mean is
  closer to 33.02 than the mean ([REF-STIGLER-1977], abstract); the three
  bearing families lie within two AIC units and none is rejected
  ([REF-BURNHAM-ANDERSON-2002, Chapter 2]);
- the case-study block of the guide equals `results_markdown` output
  character for character; the laboratory docstring has the sections
  Definitions, Assumptions, Equations, Diagnostics, Cost and error, and
  References in that order with registered keys and an integer `SEED`; the
  laboratory finishes in under 60 s.

The four figures were checked with a figure audit (font sizes, clipping,
text-line collisions, contrast) and by eye; the remaining warnings come from
the shared theme (13 pt titles, label hues near 4:1 contrast) and from tied
integer values in the Q-Q plot.
