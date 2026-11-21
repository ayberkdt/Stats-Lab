# Literature Survey and Gap Ledger (November 2026)

This document records the survey behind release 0.27: which recent methods
the library lacked, which of them were implemented, what the literature of
2023-2026 has changed, and which questions appear to be open. It is a
working ledger for deciding what to build or investigate next, not a review
article.

## How far to trust this document

- **Bibliographic data** (authors, year, title, venue, DOI or arXiv
  identifier) were confirmed by web search on 21 November 2026 unless an
  entry is marked *unconfirmed*. Full texts were not opened: the publisher
  and arXiv hosts were not reachable from the build environment.
- **Formulas were not copied from the sources.** Every method that was
  implemented was re-derived and tested against an independent oracle (a
  closed form, exact enumeration, a guarantee checked by simulation, or
  agreement with an existing routine in a special case). The decision
  record of each work package lists the oracle and the measured agreement.
  Where a constant or a bound could not be derived, it is not printed in the
  library; the docstring refers to the source instead.
- **"Open problem" statements** rest on abstracts and search snippets, not
  on the future-work sections of the papers. Items marked *speculation* are
  conjectures of this survey: a literature check is required before any of
  them is described as new.
- **Section and equation locators** were removed from citations that could
  not be checked.

## 1. What was missing and what was added

| Area | Added in 0.27 | Primary sources | Decision record |
|---|---|---|---|
| Multifidelity sampling | MFMC with closed-form allocation and model selection; ACV-IS and ACV-MF; multilevel BLUE with a certified continuous allocation; a comparison table | Peherstorfer, Willcox, Gunzburger 2016 (doi:10.1137/15M1046472); Gorodetsky, Geraci, Eldred, Jakeman 2020 (doi:10.1016/j.jcp.2020.109257); Schaden, Ullmann 2020 (doi:10.1137/19M1263534); Croci, Willcox, Wright 2023 (doi:10.1016/j.cma.2023.116130) | ADR-0117 |
| Anytime-valid inference | e-value calibration and merging; e-BH; betting tests and confidence sequences for bounded means; universal inference | Vovk, Wang 2021 (doi:10.1214/20-AOS2020); Wang, Ramdas 2022 (arXiv:2009.02824); Waudby-Smith, Ramdas 2024 (doi:10.1093/jrsssb/qkad009); Wasserman, Ramdas, Balakrishnan 2020 (doi:10.1073/pnas.1922664117) | ADR-0119 |
| Prediction-powered inference | PPI and power-tuned PPI for a mean | Angelopoulos et al. 2023 (doi:10.1126/science.adi6000); Angelopoulos, Duchi, Zrnic 2023 (arXiv:2311.01453) | ADR-0119 |
| Conformal prediction | weighted (covariate shift); fixed-weight nonexchangeable; conformal risk control; conformal p-values with BH; Mondrian; CV+; online quantile tracking | Tibshirani et al. 2019 (arXiv:1904.06019); Barber et al. 2023 (doi:10.1214/23-AOS2276); Angelopoulos et al. 2024 (arXiv:2208.02814); Bates et al. 2023 (doi:10.1214/22-AOS2244) | ADR-0120 |
| Causal inference | double/debiased ML; group-time DiD; omitted-variable-bias sensitivity; synthetic control | Chernozhukov et al. 2018 (doi:10.1111/ectj.12097); Callaway, Sant'Anna 2021 (doi:10.1016/j.jeconom.2020.12.001); Cinelli, Hazlett 2020 (doi:10.1111/rssb.12348); Abadie, Diamond, Hainmueller 2010 (doi:10.1198/jasa.2009.ap08746) | ADR-0121 |
| Bayesian computation | PSIS with k-hat and PSIS-LOO; nested R-hat; SVGD; slice and elliptical slice sampling; MALA; stretch move; parallel tempering with stepping-stone evidence | Vehtari et al. 2024 (JMLR 25(72), arXiv:1507.02646); Liu, Wang 2016 (arXiv:1608.04471); Murray, Adams, MacKay 2010 (arXiv:1001.0175); Goodman, Weare 2010 (doi:10.2140/camcos.2010.5.65); Syed et al. 2022 (doi:10.1111/rssb.12464) | ADR-0122 |
| Sensitivity analysis | rank-based first-order Sobol indices; active subspaces; moment-independent indices; kernel Sobol indices | Gamboa et al. 2022 (doi:10.3150/21-BEJ1421); Constantine, Dow, Wang 2014 (doi:10.1137/130916138); Borgonovo 2007 (doi:10.1016/j.ress.2006.04.015); da Veiga 2021 (arXiv:2101.05487) | ADR-0123 |
| Rare events | improved cross-entropy; sequential importance sampling; BUS | Papaioannou, Geyer, Straub 2019 (doi:10.1016/j.ress.2019.106564); Papaioannou, Papadimitriou, Straub 2016 (doi:10.1016/j.strusafe.2016.06.002); Straub, Papaioannou 2015 (doi:10.1061/(ASCE)EM.1943-7889.0000839) | ADR-0124 |
| Surrogates and optimization | recursive co-kriging; stochastic kriging; Bayesian quadrature; Bayesian optimization with a stable log-EI, UCB, constraints, knowledge gradient | Kennedy, O'Hagan 2000 (doi:10.1093/biomet/87.1.1); Ankenman, Nelson, Staum 2010 (doi:10.1287/opre.1090.0754); Jones, Schonlau, Welch 1998 (doi:10.1023/A:1008306431147); Ament et al. 2023 (arXiv:2310.20708) | ADR-0125 |
| Sequential decisions | LinUCB, linear Thompson sampling; off-policy evaluation; finite MDPs; best-arm identification | Li et al. 2010 (doi:10.1145/1772690.1772758); Agrawal, Goyal 2013 (PMLR 28(3)); Dudík, Langford, Li 2011 (arXiv:1103.4601); Karnin, Koren, Somekh 2013 (PMLR 28(3)) | ADR-0126 |
| Robust statistics, dependence, transport | median-of-means, Catoni; Ledoit-Wolf shrinkage; Chatterjee's xi, distance correlation; sliced and unbalanced transport | Catoni 2012 (doi:10.1214/11-AIHP454); Ledoit, Wolf 2004 (doi:10.1016/S0047-259X(03)00096-4); Chatterjee 2021 (doi:10.1080/01621459.2020.1758115); Székely, Rizzo, Bakirov 2007 (doi:10.1214/009053607000000505); Bonneel et al. 2015 (doi:10.1007/s10851-014-0506-3) | ADR-0127 |
| Quasi-Monte Carlo | rank-1 lattices with random shifts, component-by-component and Korobov construction; mean and median of randomized rules; construction-free median lattice rule | Cranley, Patterson 1976 (doi:10.1137/0713071); Goda, L'Ecuyer 2022 (arXiv:2201.09413); Pan, Owen 2023 (arXiv:2111.12676) | ADR-0118 |
| Survival and time series | Aalen-Johansen cumulative incidence; restricted-mean contrasts; Bayesian online change-point detection; forecast reconciliation (OLS, WLS, MinT) | Aalen, Johansen 1978; Adams, MacKay 2007 (arXiv:0710.3742); Wickramasuriya, Athanasopoulos, Hyndman 2019 (doi:10.1080/01621459.2018.1448825) | ADR-0128 |
| Simulation-based inference | conditional flow matching; flow-matching posterior estimation; neural ratio estimation (small networks, one dimension) | Lipman et al. 2023 (arXiv:2210.02747); Wildberger et al. 2023 (arXiv:2305.17161); Hermans, Begy, Louppe 2020 (arXiv:1903.04057) | ADR-0129 |
| Ensemble inversion | interacting Langevin sampler with finite-ensemble correction; unscented Kalman inversion; consensus-based sampling | Garbuno-Inigo et al. 2020 (arXiv:1903.08866, arXiv:1912.02859); Huang, Schneider, Stuart 2022 (doi:10.1016/j.jcp.2022.111262); Carrillo et al. 2022 (doi:10.1111/sapm.12470) | ADR-0130 |
| Method development | estimator combinators, certification checks, pre-specified simulation studies, candidate selection | Rhee, Glynn 2015 (doi:10.1287/opre.2015.1404); Morris, White, Crowther 2019 (doi:10.1002/sim.8086); Pawel, Kook, Reeve 2024 (doi:10.1002/bimj.202200091) | ADR-0131 |

A row lists what its decision record reports as implemented and tested; the
record also states what was left out of each package.

## 2. What changed in 2023-2026

**Multifidelity estimation converged on linear estimators.** Grouped
approximate control variates (Gorodetsky, Jakeman, Eldred 2024,
arXiv:2402.14736) contain every multilevel BLUE, and sample allocation
moved from the closed form of MFMC to convex programming (Croci, Willcox,
Wright 2023). The pilot-sample problem is now treated as an
exploration-exploitation trade-off (Dixon et al., arXiv:2505.09828). The
library solves the allocation as a smooth program rather than a
semidefinite program and certifies the result with a lower bound; it does
not implement grouped ACV or pilot-budget optimization.

**Quasi-Monte Carlo: the median replaces the mean.** The median of
independently randomized rules adapts to unknown smoothness (Pan, Owen
2023, arXiv:2111.12676; Goda, L'Ecuyer 2022, arXiv:2201.09413), and the
validity of confidence intervals from a few randomizations is under
scrutiny (L'Ecuyer, Nakayama, Owen, Tuffin 2023, Winter Simulation
Conference; Jain, Hickernell, Owen, Sorokin 2025, arXiv:2504.18677). The
library implements the median of randomized lattice and digital rules and,
following that caution, attaches no interval to the median; the measured
coverage of the Student-t interval of the mean is reported in ADR-0118.

**Sensitivity analysis moved to given-data estimators.** Rank-based
estimation (Gamboa et al. 2022) gives all first-order indices from one
sample; kernel methods extend this to higher orders (da Veiga et al.,
arXiv:2303.17832). For dependent inputs, proportional marginal effects
(Herin et al. 2024, doi:10.1137/22M153032X) complement Shapley effects. The
first is implemented; the others are not.

**Conformal prediction generalized its guarantee.** Conformal risk control
extends coverage to monotone losses; Gibbs, Cherian and Candès
(arXiv:2305.12616) recast conditional coverage as coverage over a class of
covariate shifts; online methods moved from fixed to decaying step sizes
(Angelopoulos, Barber, Bates 2024, arXiv:2402.01139). The reference text is
Angelopoulos, Barber and Bates, "Theoretical Foundations of Conformal
Prediction" (arXiv:2411.11824).

**Anytime-valid inference became a toolkit.** The review of Ramdas,
Grünwald, Vovk and Shafer (2023, doi:10.1214/23-STS894), betting confidence
sequences, asymptotic confidence sequences (Waudby-Smith et al. 2024,
doi:10.1214/24-AOS2408), and e-detectors for change detection (Shin, Ramdas,
Rinaldo 2024, doi:10.51387/23-NEJSDS51). The library implements the bounded
case only.

**Bayesian computation.** The final version of the PSIS paper (2024)
replaced the fixed 0.7 threshold for k-hat by a sample-size-dependent one,
which the library uses. Microcanonical samplers gained a Metropolis
adjustment (Robnik, Cohn-Gordon, Seljak 2025, arXiv:2503.01707);
deterministic-objective variational inference (Giordano, Ingram, Broderick
2024, arXiv:2304.05527) and batch-and-match (Cai et al. 2024,
arXiv:2402.14758) improved Gaussian variational inference. In
simulation-based inference, flow matching (Wildberger et al. 2023,
arXiv:2305.17161) and consistency models displaced discrete-flow posterior
estimation. Of the items in this paragraph, PSIS, flow matching,
flow-matching posterior estimation, and neural ratio estimation are
implemented (the last three for small problems); the samplers and the
variational methods named here are not.

**Bayesian optimization.** Expected improvement fails numerically where it
underflows; its logarithm fixes this (Ament et al. 2023). Several papers
argue that plain Gaussian processes with sensible length-scale priors or
initialization are competitive in high dimension (Hvarfner, Hellsten, Nardi
2024, arXiv:2402.02229; Papenmeier, Poloczek, Nardi 2025, arXiv:2502.09198);
the matter did not look settled at the time of the survey.

**Causal inference.** Heterogeneity-robust difference-in-differences
estimators replaced two-way fixed effects under staggered adoption (Roth et
al. 2023, doi:10.1016/j.jeconom.2023.03.008).

**Automated method discovery.** Program-search systems (FunSearch,
Romera-Paredes et al. 2024, doi:10.1038/s41586-023-06924-6; AlphaEvolve,
Novikov et al. 2025, arXiv:2506.13131) found new combinatorial
constructions where a cheap, trusted, automatic scorer exists. The survey
found no demonstrated automated discovery of a statistical estimator that
comes with guarantees. `probstat.methodlab` is therefore built as
mechanized recombination plus disciplined falsification, and says so.

## 3. Open questions, and what the library can do about each

Each entry names the evidence for the gap and the experiment the library
now supports. *Speculation* marks conjectures of this survey.

1. **Pilot-estimated covariances beyond the mean.** Explore-exploit theory
   for multifidelity estimation covers the high-fidelity mean (Dixon et
   al., arXiv:2505.09828); multifidelity Sobol estimation assumes pilot
   statistics (Qian et al. 2018, doi:10.1137/17M1151006). *Experiment:*
   `monte_carlo.pilot_statistics` with varying pilot sizes, certified by
   `methodlab.check_variance` against the predicted variance.
2. **Confidence intervals for median quasi-Monte Carlo.** The median of
   randomizations has no accompanying interval with guaranteed coverage
   (*speculation*, from reading Pan-Owen and Jain et al.). *Experiment:*
   replicates of `sampling.rqmc_median` fed to
   `inference.betting_confidence_sequence` on bounded integrands, checked
   with `methodlab.check_coverage`.
3. **Inference for total-order indices from one sample.** A central limit
   theorem exists for the rank-based first-order estimator only
   (*speculation* for total order). *Experiment:* replicate studies with
   `methodlab.run_simulation_study` on the Ishigami benchmark.
4. **Conformal guarantees propagated to derived quantities.** Coverage of a
   surrogate's predictions is marginal; bounds on a failure probability or
   a Sobol index computed from the surrogate do not follow (*speculation*).
   *Experiment:* `conformal.conformalize_surrogate` inside
   `reliability.ak_mcs`, certified against the exact benchmark values.
5. **Stopping rules in active-learning reliability.** The stopping
   criterion is treated as a heuristic module in the benchmark of
   Moustapha, Marelli and Sudret (2022, arXiv:2106.01713). *Experiment:*
   anytime-valid stopping with `inference.betting_mean_test` on the
   misclassification indicator of fresh samples.
6. **Multifidelity estimators for rare events.** Combining MLBLUE-type
   allocation with adaptive importance densities, whose covariances change
   with the proposal, appears unexplored (*speculation*). *Experiment:*
   `monte_carlo.mlblue_allocation` with the improved cross-entropy
   densities of `reliability`.
7. **Overconfident amortized posteriors.** Hermans et al. (2022,
   arXiv:2110.06581) show that neural simulation-based inference can be
   overconfident, and the library's own flow-matching posterior showed
   three of six calibration p-values below 0.1 (ADR-0129). *Experiment:* simulation-based calibration
   (`inference` SBC) and `inference.pareto_smooth` k-hat as a per-data-set
   diagnostic where the likelihood is tractable (*speculation* that this
   diagnostic has not been studied for amortized estimators).
8. **Bootstrap failure for Chatterjee's coefficient away from
   independence.** Lin and Han (2024, doi:10.1093/biomet/asae004).
   *Experiment:* `methodlab.check_coverage` on bootstrap and
   subsampling intervals for `statistics.chatterjee_correlation`.
9. **Prediction-powered inference outside its assumptions.** Validity needs
   labelled and unlabelled data from one law and a predictor fitted
   independently; finite-sample or anytime-valid versions look
   under-explored (*speculation*). *Experiment:* betting confidence
   sequences on the bounded rectifier term.
10. **Parallel-trends violations with staggered adoption.** Roth et al.
    (2023) describe this as an active area. A sensitivity analysis for
    group-time effects in the style of the omitted-variable-bias bounds
    would be a natural combination (*speculation*).
11. **Derived estimators with a certified error bar.** Randomized
    truncation gives unbiasedness at the price of heavy tails; a confidence
    interval that remains valid when the variance is barely finite needs
    robust mean estimation (median-of-means, Catoni) (*speculation* about
    the combination). *Experiment:* `methodlab.single_term_estimator`
    summarized by `statistics.median_of_means`, with
    `methodlab.check_coverage`.

## 4. Verified sources not yet implemented

Ordered by the value the survey assigned; each needs the same derivation
and oracle discipline before it enters the library.

- Fast component-by-component construction of lattice rules and
  guaranteed-coverage intervals for randomized quasi-Monte Carlo (Jain et
  al. 2025).
- Multi-index Monte Carlo (Haji-Ali, Nobile, Tempone 2016,
  doi:10.1007/s00211-015-0734-5); multifidelity Sobol estimation (Qian et
  al. 2018).
- Proportional marginal effects (Herin et al. 2024); nearest-neighbour
  Shapley effects (Broto, Bachoc, Depecker 2020, doi:10.1137/18M1234631).
- Neural likelihood estimation (Papamakarios, Sterratt, Murray 2019,
  arXiv:1805.07226), minibatch optimal-transport pairing for flow matching
  (Tong et al. 2024, arXiv:2302.00482), and consistency-model posterior
  estimation (Schmitt et al. 2024, arXiv:2312.05440).
- Pathfinder (Zhang et al. 2022, arXiv:2108.03782); unbiased MCMC with
  couplings (Jacob, O'Leary, Atchadé 2020, doi:10.1111/rssb.12336);
  microcanonical Langevin Monte Carlo (Robnik et al. 2023,
  arXiv:2212.08549).
- Iterative ensemble Kalman smoother (Bocquet, Sakov 2014) and hybrid
  ensemble-variational assimilation.
- Generalized and power posteriors (Bissiri, Holmes, Walker 2016,
  doi:10.1111/rssb.12158); martingale posteriors (Fong, Holmes, Walker
  2023, doi:10.1093/jrsssb/qkad005).
- Subdistribution-hazard regression for competing risks (Fine, Gray 1999;
  page and DOI details unconfirmed).
- Regression discontinuity with robust intervals (Calonico, Cattaneo,
  Titiunik 2014, doi:10.3982/ECTA11757); fixed-X knockoffs (Barber, Candès
  2015, doi:10.1214/15-AOS1337).
- Probabilistic ODE solvers (Schober, Särkkä, Hennig 2019,
  doi:10.1007/s11222-017-9798-7); Wasserstein distributionally robust
  optimization (Mohajerin Esfahani, Kuhn 2018,
  doi:10.1007/s10107-017-1172-1).
- Time-variant reliability by PHI2 (Andrieu-Renaud, Sudret, Lemaire 2004,
  doi:10.1016/j.ress.2003.10.005).

## 5. Sources that could not be confirmed

Named in the survey from memory and not cited anywhere in the library:
Bellman (1957) and Howard (1960) as primary sources for dynamic
programming; Vovk (2013) on Mondrian conformal predictors; Roberts and
Rosenthal (1998) on optimal scaling of MALA; Zhang and Stephens (2009) on
the generalized Pareto fit used by PSIS (the estimator is implemented from
the reference implementation's constants and named in prose only); Sloan
and Reztsov (2002) and Nuyens and Cools (2006) on component-by-component
lattice construction; Fine and Gray (1999) page and DOI details.
