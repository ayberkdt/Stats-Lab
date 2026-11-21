# ADR-0131 — A method laboratory: deriving estimators and certifying claims

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0049 (benchmark suite), ADR-0087 (property and metamorphic
  testing), ADR-0077 (performance profiles)

## Context

Every package so far *contains* methods: a user calls MLMC, a control
variate, or the jackknife as finished procedures. Nothing helped a user who
wants to build a method the library does not have, and nothing turned "is my
new estimator actually unbiased, and is it better?" into a reproducible
procedure with a stated error probability. The literature offers two things
that can be mechanized, and one that cannot:

- **Construction principles** that map estimators to estimators under
  explicit conditions: randomized truncation (Rhee and Glynn 2015; McLeish
  2011), extrapolation, best linear unbiased combination (Schaden and
  Ullmann 2020), control variates, jackknife and bootstrap bias correction
  (Quenouille 1956), and influence-function corrections (Hampel 1974;
  Kennedy 2022; Carone, Luedtke and van der Laan 2019).
- **Certification practice**: simulation studies in the ADEMP structure
  with Monte Carlo standard errors (Morris, White and Crowther 2019),
  pre-specification against flexible designs (Pawel, Kook and Reeve 2024),
  equivalence logic for "supported" versus "not refuted" (Schuirmann 1987),
  metamorphic relations (Chen et al. 2018), and fixed-budget selection
  (Karnin, Koren and Somekh 2013).
- **Not mechanizable**: proving the regularity conditions of a
  construction, and inventing an estimand or a structural idea. Published
  program-search systems optimize against a trusted automatic scorer; we
  found no demonstrated automated discovery of a statistical estimator that
  comes with guarantees.

## Decision

1. **New package `probstat.methodlab`** (layer 7, stability `experimental`)
   with five modules: `estimators`, `combinators`, `certify`,
   `simulation_study`, `racing`. It imports `foundations`, `linalg`,
   `distributions`, and `inference` only.
2. **One contract.** An `Estimator` is a name, a sampler
   `RandomStream -> Draw(value, cost)`, and a lineage (derivation steps,
   oldest first). Unbiasedness is a claim, never a type property.
   Replicate `i` uses the child stream `("replicate", i)`, so estimators
   replicated with one seed share random numbers.
3. **Combinators return estimators** and append one sentence to the
   lineage: `single_term_estimator`, `coupled_sum_estimator` (with
   `geometric_levels`, `finite_levels`, `optimal_level_probabilities`,
   `truncation_design`), `thinned_estimator`, `sum_of_estimators`,
   `extrapolated_estimator`, `combine_unbiased`,
   `control_variate_estimator`, `antithetic_estimator`. Weights and
   coefficients are estimated from a pilot with its own seed and frozen, so
   linear combinations stay exactly unbiased. Data-based recipes act on
   statistics: `jackknife_bias_corrected`, `bootstrap_bias_corrected`,
   `empirical_influence` (central-difference Gateaux derivative in the
   observation weights), `one_step_estimator`.
4. **Four verdicts** ([EQ-MLAB-VERDICT-001]): `refuted`, `supported` (only
   with a tolerance declared before the run and the whole interval inside
   the band), `not refuted`, `inconclusive`. Checks default to 99%
   intervals. `check_coverage` uses a Clopper-Pearson interval;
   `check_variance` and `check_efficiency_gain` use the distribution-free
   standard error of a sample variance ([EQ-MLAB-MCSE-002]).
   `check_variance_stability` is labelled a heuristic. A `MethodCard`
   reports lineage, scope, checks, and a scoped verdict; it never says
   "validated".
5. **Efficiency is variance times mean cost** ([EQ-MLAB-WNV-001]), not
   variance.
6. **Simulation studies are pre-specified.** `StudyProtocol` holds the
   ADEMP items, repetitions, seed, and primary comparison, and has a
   SHA-256 fingerprint; `run_simulation_study` refuses to run when an
   expected fingerprint differs. All methods analyse the same data sets;
   failures are counted per method; every measure of Morris, White and
   Crowther's Table 6 that we implement carries its Monte Carlo standard
   error.
7. **Selection is separated from certification.** `rank_by_efficiency` and
   `sequential_halving` pick a candidate; the documentation requires fresh
   seeds for certifying it.

## Consequences

- On the guide's example (`E[integral_0^1 exp(Z x) dx]`, level-2 trapezoid
  rule, exact bias 0.0086) 40 000 replicates leave the biased rule "not
  refuted" (bias estimate 0.0065, standard error 0.0038); with a control
  variate the same rule is refuted (0.0086, standard error 0.0008). The
  derived estimator (control variate, randomized correction evaluated with
  probability 0.25) is "supported" as unbiased within 0.005 and is 14.4
  times (99% interval 10.3 to 20.2) more efficient than the same
  construction without the control variate.
- Over 400 seeds a true unbiasedness claim was refuted at the 95% level
  between 6 and 36 times (the test asserts this band around the nominal
  20).
- The jackknife correction of the plug-in variance equals the unbiased
  variance to rounding; numerical influence values of the mean, variance,
  and a ratio of means equal the analytical ones to 6-8 decimals.
- The package cannot check the conditions of a construction. A
  randomized-truncation estimator with a level law that is too light has
  infinite variance and still passes `check_unbiasedness` by eye; the guide
  and the package docstring say so.
- The guide figure was rendered through `tools/build_guide_figures.py` and
  `theme.fit_text_to_canvas`, but the datawiz audit step could not be run
  (the toolkit is not available in the build environment used here); it
  must be re-run where datawiz is installed.
- Citations were confirmed bibliographically by web search except
  [REF-NIST-FIPS-180-4], which was entered from memory of the standard.

## Rejected alternatives

- **A symbolic or program-search "method discoverer".** No demonstrated
  basis for estimators with guarantees, and results would be claims the
  library could not back.
- **A boolean `unbiased` attribute on estimators.** It would be asserted by
  the author and trusted by readers; a claim tested by a check is honest.
- **A single "pass/fail" certification.** It hides the difference between
  "supported within a tolerance" and "not refuted because the run was too
  small", which the guide's first example shows matters.
- **Estimating combination weights from the combined draws.** Simpler to
  call, but biased at order `1/n`.
- **Layer 8.** A new layer for one package; layer 7 next to `workflow`
  suffices because nothing imports `methodlab`.
- **Hyperband, irace, F-race.** Useful, but their primary sources were not
  all confirmed in this work; only sequential halving is implemented.

## Verification

`tests/test_methodlab.py` (42 tests, about 50 s): exact enumeration of the
expectation of both randomized estimators over finite level laws; the
closed-form variance of the single-term and thinned estimators; the
Cauchy-Schwarz bound attained by the optimal level law and not beaten by 50
random laws; the classical Richardson pair and exact cancellation of two
orders; best linear unbiased weights against a brute-force grid; the
control-variate variance `(1 - rho^2) Var` for `exp(U)` with control `U`;
antithetic exactness for linear integrands; jackknife and bootstrap
corrections; analytical influence functions and the delta-method standard
error of a ratio; every verdict in both directions; the false-refutation
rate; hand-computed performance measures; the normal mean-versus-median
study; fingerprint mismatch; failure counting; sequential halving with and
without noise. `tests/test_documentation_examples.py` runs the guide.

## References

[REF-RHEE-GLYNN-2015], [REF-MCLEISH-2011], [REF-SCHADEN-ULLMANN-2020],
[REF-QUENOUILLE-1956], [REF-HAMPEL-1974], [REF-KENNEDY-2022],
[REF-HINES-ETAL-2022], [REF-CARONE-LUEDTKE-VANDERLAAN-2019],
[REF-MORRIS-WHITE-CROWTHER-2019], [REF-PAWEL-KOOK-REEVE-2024],
[REF-SCHUIRMANN-1987], [REF-CHEN-ETAL-2018],
[REF-KARNIN-KOREN-SOMEKH-2013], [REF-GLASSERMAN-2003].
