# ADR-0136 — Expert elicitation and opinion pooling

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0059 (finite mixtures), ADR-0048 (API stability)

## Context

In early design, inputs such as a specific impulse, an efficiency, a drag
coefficient or a surface absorptance have no data; the input law comes from
engineers' judgement ("most likely 0.28, surprised below 0.26 or above
0.30"). The library had the laws (`PERT`, `Triangular`, `Mixture`,
`maximum_entropy`) and data fitting (`fit_distribution`), but nothing that
turns stated quantiles into a law, shows the consequences back to the expert,
or combines several experts, with or without performance weights.

## Decision

1. **One module, `probstat.inference.elicitation`** (layer 2). It needs only
   distributions, optimization and numerics (layers 0 and 1).
2. **Fit on the quantile function, in value units.** `fit_from_quantiles`
   minimises `sum w (Q(p; theta) - v)^2` ([EQ-INF-ELICIT-001]). This differs
   from the SHELF package, which (per its `fitdist` documentation, the only
   part read) minimises squared errors on the distribution function. The
   quantile misfit is in the units of the input, so a residual of 0.002 on a
   specific impulse fraction reads directly; it is also linear for
   location-scale families, which gives closed forms. Closed form for normal,
   Student-t at fixed degrees of freedom, uniform and the two-pair lognormal;
   Levenberg-Marquardt (`probstat.optimization.least_squares`) from several
   starts for gamma, Weibull, beta on a stated interval, triangular and PERT.
   Every fit returns residuals and feedback quantiles.
3. **Student-t uses the library's `StudentT(nu, mu, sigma)`**, which already
   is the location-scale law, instead of wrapping a standard t in `Affine`;
   beta on a stated interval does use `Affine(Beta, low, high - low)`.
4. **Ranking is by misfit and is not a selection rule.**
   `compare_elicited_fits` lists every fit, sorted by misfit with ties by
   fewer parameters; families that cannot be fitted are reported in
   `skipped` with the reason. A family with as many parameters as statements
   fits exactly, so no information criterion is offered: judgement has no
   likelihood.
5. **Three-point laws from quantiles.** With the mode fraction `m` as the
   only unknown, the two quantile equations reduce to one monotone equation
   ([EQ-INF-ELICIT-002]) solved by Brent's method on `[0, 1]`; the width and
   the bounds follow. The mode may lie outside the stated quantile interval
   for skewed laws; when no `m` solves the equation the error states the
   interval the mode must lie in.
6. **Pools return library laws where one exists.** The linear pool is a
   `Mixture`. The logarithmic pool is a `Normal` for all-normal experts
   (precision-weighted closed form) and otherwise a `LogarithmicPool`, a
   frozen object with the marginal interface (`pdf`, `cdf`, `quantile`,
   `mean`, `variance`), because the pool of non-normal laws belongs to no
   family. It is normalised by Gauss-Kronrod quadrature on panels whose ends
   are quantiles of the experts' laws, over the hull of their
   `[tail, 1 - tail]` ranges, with the cumulative mass tabulated so that the
   distribution function and quantile cost one partial panel. Neglected mass
   outside the hull is of order `tail` to the smallest weight at most
   (the pool's tails decay more slowly than the experts' because the factors
   are raised to powers below one). Quantile averaging is a `Normal` for
   normal experts and otherwise a `VincentizedLaw` (quantile function
   `sum w Q`, density `1 / sum (w / f(Q))`, mean exact). Weights are
   normalised to sum to one for all three pools.
7. **Distinguishing properties** (derived, and checked in the tests, not
   copied): the linear pool preserves marginals and is not externally
   Bayesian (its posterior reweights experts by evidence); the logarithmic
   pool with weights summing to one is externally Bayesian because a
   likelihood factors out of `prod f_i^{w_i}`, and does not preserve
   marginals; quantile averaging keeps a location-scale family closed and has
   neither property in general ([EQ-INF-ELICIT-003], [EQ-INF-ELICIT-004]).
8. **Cooke's classical model** ([EQ-INF-ELICIT-005] to [EQ-INF-ELICIT-007]):
   calibration is the chi-square tail probability of `2 N I(s, p)` with `k`
   degrees of freedom for `k` stated quantiles; information is the mean
   relative information of the piecewise-uniform density against a uniform or
   log-uniform background on the intrinsic range (hull of all experts'
   quantiles and the realization, extended by an overshoot, 10 % default); the
   weight is `C I` above the cut-off; the decision maker pools the densities
   linearly with global weights and is scored on the seed items with its own
   exact quantiles and information (it is constant on cells, so both are
   exact). The cut-off is optimised over the experts' calibration scores,
   smaller cut-off on a tie. Every expert's calibration and information are
   exposed. `cooke_decision_maker_law` gives the pooled law of an item as a
   `Mixture` of uniform (or log-uniform) bins.

## Sources

Opened and read after the first version of this record (source verification
pass):

- Hanea and Nane (2021), "An In-Depth Perspective on the Classical Model"
  [REF-HANEA-NANE-2021] (TU Delft repository, accepted manuscript), Sections
  3.1, 3.2.1, 3.2.2, 3.3, 4 and 4.1. Compared with the code point by point:
  - *Calibration* (Section 3.2.1, Eq. 1): `2 M I(s, p)` with the empirical
    bin frequencies `s` and `p = (0.05, 0.45, 0.45, 0.05)`, asymptotically
    chi-square with 3 degrees of freedom for the three percentiles; bins are
    `x <= q5`, `q5 < x <= q50`, `q50 < x <= q95`, `q95 < x`. Agreement: the
    code uses `k` degrees of freedom for `k` stated quantiles (3 for the
    default), the same bin convention, and `0 ln 0 = 0`. The chapter's worked
    values (best attainable score 0.83 for ten seed items at `(0.1, 0.4, 0.4,
    0.1)`, 0.39 at `(0.1, 0.6, 0.2, 0.1)`, twenty items needed to reach 1) are
    reproduced (0.8283, 0.3946, 1.0) and pinned in
    `test_published_calibration_examples`. One sentence of the chapter, that an
    expert with `(0.3, 0.2, 0.2, 0.3)` scores "of order 10^-2", does not match
    the formula for ten items (0.0018, of order 10^-3); the code follows the
    formula and the statement is not pinned.
  - *Intrinsic range* (Section 3.1): `[L - k (U - L), U + k (U - L)]` with
    `L` the smallest of the experts' lowest percentiles and, for a seed item,
    the realization; `U` correspondingly; `k = 10 %` by default in EXCALIBUR.
    Agreement, including that items of interest use the experts' percentiles
    only (`realizations=None`). The chapter says the range may be fixed by the
    analyst for some items (a percentage on `[0, 100]`); the code takes the
    range as an argument of `cooke_information` for that case.
  - *Information* (Section 3.2.2, Eq. 2, Table 2): relative information of the
    piecewise-uniform density with respect to the uniform (or log-uniform)
    background on the intrinsic range, averaged over items. Agreement; Table
    2 (assessments `(5, 15, 25)`, `(40, 50, 60)`, `(15, 17, 75)`, `(30, 50,
    70)` on `[0, 100]`, scores 1.21, 1.14, 1.15, 0.55) is reproduced to within
    one unit of the second decimal (the formula gives 1.218267, 1.145726,
    1.151478, 0.550661; the first two entries appear truncated rather than
    rounded, since 1.2183 would round to 1.22) and pinned
    in `test_published_information_examples`. The chapter's rule of thumb for
    a log-uniform background (assessments spanning more than four orders of
    magnitude) is not enforced; the background is an argument.
  - *Weights and cut-off* (Sections 3.3, 4, 4.1): combined score `Cal x I x
    1{Cal >= alpha}`, normalised (global weights, Eq. 3); the distinct
    decision makers are obtained at the experts' calibration scores as
    cut-offs; the optimised global-weights decision maker is the cut-off that
    maximises its own combined score. Agreement with `cooke_weights` and
    `cooke_classical_model`. The tie rule (smaller cut-off) is this library's.
  - *Differences kept and stated:* (1) the code averages an expert's
    information over the seed items only, the chapter defines `I(e)` as the
    average over the seed questions in Section 3.2.2 and notes that it can
    also be computed for the questions of interest; EXCALIBUR-style slides
    (Aspinall, Cabot Institute summer school 2013, read) say the information
    entering the weight is estimated from all variables, so a user who needs
    that must compute it with `cooke_information` over the target items too;
    (2) the overshoot for the log-uniform background is applied on the
    logarithmic scale, which the chapter does not specify; (3) item weights
    are not implemented (listed under "Deliberately left out").
- The SHELF CRAN reference manual, version 1.13.0, entry `fitdist` (read in
  full) [REF-SHELF-OAKLEY-OHAGAN]: least squares on the distribution function,
  optional weights, a user-chosen Student-t degrees of freedom (default 3),
  skew-normal, gamma, log-normal, log Student-t, mirrored and beta families,
  best fit by the smallest sum of squares. The quantile-function objective,
  the default of 5 degrees of freedom and the absence of a skew-normal family
  are this library's choices.

Not opened: Cooke (1991) and O'Hagan et al. (2006) (books), Genest and Zidek
(1986) (the Project Euclid page returned no text), Clemen and Winkler (1999)
(publisher paywall), the Cooke-Goossens procedures guide EUR 18820 (no open
copy found), and the SHELF vignettes. The pooling properties in decision 7
therefore remain derived and numerically checked, not compared with the survey.
[REF-NASA-PRA-2011] is cited for expert judgement as an input to risk models.

## Consequences

- Elicited judgements enter the uncertainty workflow as ordinary laws: the
  results are inputs for `modeling` marginals and `propagation`.
- The scoring conventions agree with the open description of EXCALIBUR in
  Hanea and Nane (2021) (see Sources). A user who needs EXCALIBUR's item
  weights, or the information of an expert averaged over the items of interest
  as well, must compute them separately (`cooke_information` accepts any
  items); the log-scale overshoot for the log-uniform background is this
  library's convention.
- Deliberately left out: item-specific (information per item) weights,
  dependence among seed variables (the calibration p-value assumes
  independence), elicitation of dependence between inputs, probability
  (rather than quantile) elicitation protocols, bootstrap or Bayesian
  uncertainty about the expert's statements, and a Bayesian pooling model
  (supra-Bayesian). A normal quantile fit also cannot represent skewness;
  compare families before choosing.
