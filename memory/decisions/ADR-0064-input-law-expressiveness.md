# ADR-0064: Input-law expressiveness — copulas, discrete inputs, epistemic separation

- Status: Accepted (proposed during the 2026-10-16 session, workstream W7)
- Date: 2026-10-16

## Context

The 2026-10-16 core-mechanics audit found three gaps in the input law that
every stage consumes. `JointInput` supported only a Gaussian copula (P1.9),
so no tail dependence could be modelled. Discrete and categorical inputs
were rejected; a duck-typed shim got a wrong copula mapping (P1.4). The Cauchy
law could not be an input, because `RandomInput.__post_init__` used `hasattr`
and `StudentT(1).mean` raises (P0.14). Finally, `UncertaintyNature` existed
but was descriptive only. There was no interval input, no p-box and no nested
propagation (P1.6).

## Decision

1. **Copula interface** (`probstat.modeling.copulas`). An abstract `Copula`
   provides: `log_density`, `density`, `cdf` (generic boundary rules, then
   `_interior_cdf`), `kendall_tau(i, j)` and `tail_dependence(i, j)` in closed
   form, `rosenblatt`/`inverse_rosenblatt`, `marginal(indices)`, and
   `sample(size, seed)`. The families are `IndependenceCopula`,
   `GaussianCopula`, `StudentTCopula(R, nu)`, and the exchangeable
   Archimedean `ClaytonCopula` (theta > 0), `GumbelCopula` (theta >= 1) and
   `FrankCopula` (theta != 0 for d = 2, theta > 0 for d >= 3), up to d = 100.
   - Archimedean conditionals are `psi^(k-1)(t_k)/psi^(k-1)(t_{k-1})`. They
     are computed in log form. Clayton uses closed-form derivatives and
     inverse. Gumbel derivatives use a coefficient recursion whose terms are
     all non-negative, so there is no cancellation; I derived it for this
     work and the tests check it against mpmath derivatives. Frank
     derivatives are polylogarithms of negative order, built from Eulerian
     polynomials. Gumbel and Frank conditionals are inverted by Brent's
     method on `log s`.
   - Sampling: `Copula.sample` uses the stochastic representation. This is
     the Marshall–Olkin frailty for Archimedean families: Gamma for Clayton,
     Kanter's positive stable for Gumbel, Kemp's logarithmic algorithm for
     Frank. The t copula uses chi-square mixing. Negative-theta Frank uses
     conditional inversion. `JointInput.from_unit`/`sample` always use the
     conditional inverse. That keeps `from_unit(p) =
     from_standard_normal(Phi^-1(p))` and keeps QMC/LHS designs meaningful.
     The frailty samplers serve as an independent check.
   - Frank's Kendall tau uses the first Debye function. Below |x| = 2 it is
     summed as a Bernoulli series, with exact rational Bernoulli numbers.
     Above that the exponential series is used, plus `D_1(-x) = D_1(x) + x/2`.
     There is no cancellation near theta = 0.
2. **JointInput integration, backward compatible.** There is a new third
   field, `copula`. `JointInput(inputs, correlation)` is unchanged, and so is
   its Gaussian code path. A scratch script compared it with the committed
   version on 2,000 random points for `from_unit`, `from_standard_normal`,
   `to_standard_normal`, `to_latent`, `log_pdf` and `sample`, with zero
   mismatches. A `GaussianCopula` passed as `copula` is stored as
   `correlation`, and an `IndependenceCopula` as `None`. So `copula` is
   non-`None` only for non-Gaussian copulas, and `dataclasses.replace(joint,
   correlation=...)` keeps working. The `dependence` property returns the
   copula in use. `is_independent` is now `correlation is None and not
   uses_rosenblatt`. New members: `copula_family`, `uses_rosenblatt`,
   `has_discrete`, `discrete_names`, `require_continuous(method)`, and
   `subset(names)`. `subset` returns exact margins in every family.
   For non-Gaussian copulas `to_standard_normal`/`from_standard_normal`
   use the Rosenblatt map in input order.
   **Order dependence:** any order gives the same exact law, but FORM/SORM
   results can change with the order (Hohenbichler & Rackwitz 1981; Lebrun &
   Dutfoy 2009). The test `RosenblattOrderTests` shows this for
   `P(load >= 3)`, with a Gumbel `load`, a lognormal `other` and Clayton(2.5).
   With `load` first, FORM is exact (beta = 1.65890). With `load` second,
   FORM gives 1.53490, because the limit state is curved in `u`.
   For the Gaussian copula the Nataf and Rosenblatt maps agree up to a
   rotation, so FORM does not depend on the order.
3. **Rank-correlation constructors and nearest-correlation repair.** There
   are two new constructors: `JointInput.from_spearman` (`rho = 2
   sin(pi r_s/6)`) and `from_kendall` (`rho = sin(pi tau/2)`). There is also
   `nearest_correlation_matrix`, Higham's (2002) Dykstra-corrected
   alternating projections. Its default eigenvalue floor is 1e-8, so the
   result factorizes. `from_pearson`, `from_spearman` and `from_kendall` take
   `repair=True` by default. When the pairwise latent matrix is not positive
   definite, they replace it and emit a `UserWarning` with the Frobenius
   distance, the largest entry change and the iteration count. With
   `repair=False` they raise. Behaviour change: `from_pearson` with a non-PD
   target used to raise; it now warns and repairs. `fit_gaussian_copula`
   keeps its eigenvalue-clipping repair, so its results are unchanged.
4. **Copula fitting** (`probstat.modeling.fitting`). New functions:
   `kendall_tau` (tau-b, O(n²) pair count), `kendall_to_latent_correlation`,
   `pseudo_observations`, `archimedean_theta_from_tau`, and `fit_copula(data,
   family, method="kendall" | "pseudo-likelihood", nu=None)`, which returns
   `CopulaFit`.
   - Elliptical families use tau inversion with repair. The t copula needs
     `nu`, or with pseudo-likelihood gets a profile fit of `nu` with `R`
     fixed from tau.
   - Archimedean families invert the mean pairwise tau. Maximum
     pseudo-likelihood works on a transformed bounded range: a grid scan
     followed by golden-section search. A maximum on the edge of the range
     is flagged with `at_boundary`.
   - No standard error is reported, because the naive pseudo-information
     understates the variance (Genest–Ghoudi–Rivest 1995).
5. **Discrete and categorical inputs.**
   - `RandomInput` checks its required attributes with
     `inspect.getattr_static`. This fixes the Cauchy case: properties are
     never evaluated.
   - A law with `pmf` and no `pdf` is discrete. `Categorical` is wrapped
     automatically in `IntegerCodedCategorical`, which codes the labels
     0..K-1 in support order and provides `label(code)` and `code(label)`.
   - Sampling uses the generalized inverse `quantile`, which is exact.
     Discrete laws never use `isf`, because its tie convention may differ.
     `quantile` may return an int or an integral float; `float()` is applied.
   - `to_latent`/`to_standard_normal(x, seed=None)` use the distributional
     transform `F(x-) + V P(X=x)`, with V = 1/2 if no seed is given and
     seeded uniform otherwise. The latent map is not one-to-one, which the
     docstrings state.
   - `require_continuous` is called by `form_hlrf`, `sorm` and
     `inverse_form`. `design_for_target_beta` and `find_design_points` reach
     it through `form_hlrf`. These methods now raise a clear `ValueError`.
     `shapley_effects` refuses non-Gaussian copulas and discrete inputs
     under a dependent copula, because its conditional sampler is Gaussian.
     `from_pearson` refuses discrete margins, because Gauss–Hermite cannot
     integrate a step function.
   - `JointInput.log_pdf` with discrete inputs depends on the copula. Under
     independence it is the mixed density/mass. With all inputs discrete it
     is the exact mass by inclusion–exclusion over the copula CDF
     (Archimedean in any d up to 16, Gaussian for d <= 2). Mixed discrete
     and continuous inputs under a dependent copula raise
     `NotImplementedError`.
6. **Epistemic/aleatory separation.**
   - `RandomInput` gets a `nature: UncertaintyNature` field, aleatory by
     default; string values are accepted.
   - `IntervalInput(name, lower, upper)` represents a quantity known only by
     its bounds.
   - `probstat.propagation.mixed.nested_propagation(model, quantity, *,
     intervals, outer_size, inner_size, inner_design, corners, thresholds,
     probabilities, refine, common_random_numbers, levels, seed)` runs a
     double loop.
     - Outer loop: an LHS over the epistemic dimensions (stream
       `("outer",)`), plus all 2^k interval corners, plus an optional bounded
       compass search (Kolda–Lewis–Torczon) that spends `refine` inner loops
       on each statistic bound.
     - Inner loop: the exact aleatory sub-law, on stream `("inner", i)`, or
       `("inner", 0)` for every point with common random numbers (the
       default). Epistemic values are passed as `fixed` parameters of a
       `dataclasses.replace`d model, so evaluations are charged to the
       caller's counter.
   - The result, `MixedPropagationResult`, holds the CDF family, a
     `ProbabilityBox` (`bounds(y)` and `exceedance_bounds(t)`), an
     `EpistemicStatistic` for the mean, each `P(Y>t)` and each quantile
     (values, min/max, argmin/argmax, and a quantile over outer points), and
     `model_evaluations`.
   - Aleatory and epistemic inputs must be independent. A Gaussian copula is
     accepted only if its cross block is zero; any other dependent copula is
     refused.

## Evidence

Each item was checked against an independent oracle. The reference values
were computed in a scratch environment (mpmath 1.4, scipy 1.18) and pasted
into the tests as literals.

- Archimedean density, CDF and Rosenblatt conditionals (d = 2, 3) agree to
  ≤ 1e-12 with mixed partial derivatives of the closed-form CDF (mpmath, 50
  digits). This covers Clayton 2.3, Gumbel 1.7, Frank 4.5 and Frank −3.
  Kendall tau agrees with the Genest–MacKay integral `1 + 4∫φ/φ'` to 1e-11.
  The Frank tau agrees with mpmath quadrature of the Debye integral to
  1e-15, including theta = −1e-3.
- Bivariate Gaussian-copula CDF (Plackett identity): agrees to 1e-13 with
  mpmath one-dimensional conditioning. t-copula log-density and conditionals
  agree with scipy `multivariate_t` and the textbook conditional t law to
  1e-11. The t tail coefficient agrees with scipy to 1e-12.
- Samplers, n = 20,000: the Clayton, Gumbel and Frank frailty samplers
  (d = 3), the negative-Frank and Gaussian samplers, and the t sampler all
  match the exact CDF within 5 SE at three points, and match Kendall tau.
- Nearest correlation for Higham's example `[[1,1,0],[1,1,1],[0,1,1]]`: the
  off-diagonals are 0.7606899 and 0.1572981. They match the paper's four
  digits and an independent scipy BFGS fit over angle-parametrized
  correlation matrices to 2e-8.
- Kendall tau-b agrees exactly with `scipy.stats.kendalltau`, with and
  without ties. Pseudo-likelihood estimates agree with scipy's bounded
  optimizer to 2e-6 relative, and their maxima to 1e-9. The estimates are
  Clayton 3.23857, Gumbel 4.71969 and Frank 15.9579.
- Discrete joint masses:
  - Gaussian copula with Poisson(2) and Poisson(3.5) margins agrees with
    mpmath to 1e-13.
  - Clayton(2) with Binomial(4, 0.3) and Binomial(3, 0.6) margins agrees
    with mpmath to 1e-14.
  - Brute-force enumeration sums to 1 within 1e-13, and its margins equal
    the marginal pmf within 1e-14.
- Nested propagation for `Y = X + E`, `X ~ N(0,1)`, `E ∈ [−0.5, 1]`: the
  p-box equals `[Φ(y−1), Φ(y+0.5)]` within 1/n at every level, which is the
  exact error bound for a 1-D LHS. The mean interval width equals b − a to
  1e-12. Exceedance and 0.9-quantile intervals match the analytic values.
  Evaluations = (5 design + 2 corners) × 2000 = 14,000, and the model's own
  counter shows the same. An interior minimum of `E[X + (E − 0.3)²]` that
  the corners miss is found by refinement at E = 0.3 ± 0.01.

## Consequences

- Positive: the input law now covers tail dependence, count and category
  inputs, heavy tails (Cauchy) and intervals, and every stage receives them
  through the same `JointInput`. Methods that would give wrong answers
  (FORM/SORM with discrete inputs, Shapley or Sobol with non-Gaussian
  dependence) refuse them with a message that names the reason.
- Limitations:
  - (a) A Rosenblatt conditional within ε of one keeps only absolute
    accuracy 2^-53 in its complement. Round-trip error grows like
    eps/min(w, 1 − w), about 2e-6 for Clayton(8) at (0.05, 0.9).
    `to_standard_normal` saturates near |u| = 8.2 on the upper side.
  - (b) The Gaussian-copula CDF exists only for d ≤ 2, and the t-copula CDF
    not at all, so discrete masses need d ≤ 2 under these copulas.
  - (c) The mixed discrete/continuous density under a dependent copula is
    not implemented.
  - (d) Only exchangeable Archimedean copulas are provided, with no nested
    or vine constructions. The bivariate Frank copula with theta < 0 has no
    frailty sampler.
  - (e) Copula fitting reports no standard errors, and `kendall_tau` is
    O(n²).
  - (f) Nested-propagation bounds over a finite design are inner
    (non-conservative) estimates. Without common random numbers, inner
    sampling noise widens the envelopes. Refinement moves only the interval
    coordinates.
  - (g) `UQStudy` has no mixed-uncertainty stage yet, and its
    report label `"Gaussian copula"` (`workflow/study.py`, owned by W1)
    should use `JointInput.copula_family`; `visualization/study.py` already
    does.
- Citation note: the brief cited "Oberkampf & Roy (2010) A Comprehensive
  Framework for VV&UQ…, Cambridge UP". That conflates two works. The 2010
  Cambridge book is *Verification and Validation in Scientific Computing*
  (now REF-OBERKAMPF-ROY-2010). The "Comprehensive Framework" paper is Roy &
  Oberkampf 2011, CMAME (REF-ROY-OBERKAMPF-2011, already present). Both are
  cited. For p-boxes the code cites Ferson et al. 2003, SAND2002-4015,
  without page-level detail.
