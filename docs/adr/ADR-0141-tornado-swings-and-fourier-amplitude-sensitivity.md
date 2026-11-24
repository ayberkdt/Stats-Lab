# ADR-0141 — Tornado swings and Fourier amplitude sensitivity testing

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0065 (sensitivity intervals), ADR-0098 (figure-set contract)

## Context

The sensitivity package has Sobol, Morris, DGSM, Shapley, HSIC, regression and
Borgonovo indices but two common tools were missing. Design reviews show
one-at-a-time swings as a tornado chart, and the Fourier amplitude sensitivity
test (FAST) is the classical cheap estimator of first-order and total
variance-based indices: its cost is `d R N` runs for the extended method and
`R N` for the random balance variant, with no pick-freeze matrices.

## Decision

1. **Tornado data, `sensitivity.tornado`.** `tornado_analysis(model, quantity,
   low=0.1, high=0.9, baseline="median", values=None)` moves each input alone
   to its low and high marginal quantile around a baseline (median, mean, or
   a point) and records, per input, both outputs, the swing, its sign and the
   share `(Delta/2)^2 / sum` ([EQ-SENS-TORNADO-001]). Bars are sorted by
   absolute swing with ties in input order. `values={"name": (low, high)}`
   and the plain-function entry `tornado_function(function, low, base, high)`
   cover inputs known as ranges. The docstrings say first what a tornado
   chart cannot show: interactions, dependence, non-monotone responses.
2. **Dependent inputs are refused.** A joint law with a non-identity copula
   raises `ValueError`. The alternative was to document the scenario meaning
   ("input `i` at its marginal quantile, the others at their marginal
   medians"). It was rejected because that point is not a plausible draw of
   the law and the swing would be read as an effect under the law; the
   conditional alternative (move `i`, set the others at their conditional
   median) changes what the baseline is for every input and has no
   established chart convention. A user who wants a scenario tornado of a
   dependent model passes the scenario values to `tornado_function`, where
   the intent is explicit.
3. **Non-monotonicity flag, [EQ-SENS-TORNADO-002].** The baseline output is
   the third point of every one-at-a-time segment; if it lies outside the
   interval of the two end outputs the response has an extremum between the
   ends (intermediate-value argument). It costs nothing, flags `x^2`, and
   misses S-shaped responses, which the docstring says. The flag is withheld
   (`bracketed=False`) when the baseline is outside the end values, as a mean
   beyond the high quantile of a skewed law can be. Shares are `nan`, not
   zero, when every swing is zero.
4. **Extended FAST, `sensitivity.fast`.** Search curve
   `u_k = 1/2 + arcsin(sin(omega_k s + phi_k)) / pi` mapped to each marginal
   through its quantile function ([EQ-SENS-FAST-001]); per input one curve
   with `omega_max = (N - 1) // (2 M)` for that input and a low-frequency set
   for the others; `S_i` from the harmonics `p omega_max`, `p <= M`, and
   `S_Ti = 1 - V_(-i) / V` from the spectrum up to `omega_max / 2`
   ([EQ-SENS-FAST-002]); `V` is the population variance of the curve outputs
   (Parseval on an odd grid, so no high-frequency sum is needed). Random
   phases `phi_k` are redrawn for each of `R >= 2` resamples; the reported
   spread is the standard error of the mean over resamples. The interference
   factor `M` is a parameter (default 4).
5. **Sample-size rule, [EQ-SENS-FAST-003].** `N` odd, `N >= 2 M omega_max + 1`
   (Nyquist), complementary frequencies distinct integers up to
   `omega_max / (2 M)`. The library adds a rule that is its own and was found
   by experiment: for `d >= 3` neighbouring low frequencies are at least two
   apart. With the pair (1, 2) the Ishigami first-order errors were up to
   0.10 (phase-resampling spread up to 0.11); with (1, 3) they were 0.02 and
   stayed there up to `omega_max / (2 M) = 8`. Together this gives
   `N >= 4 M^2 max(2 d - 3, 1) + 1`, exposed as `fast_sample_size(d, M)`;
   shorter curves are refused with the required size in the message. A
   second experiment showed that at fixed `M` the estimate does not change
   with a longer curve (identical to three digits for `N = 513, 1025, 2049`
   at `M = 4`): the truncation error at `M` harmonics dominates, so the
   documentation tells users to raise `M`, not `N`. Ishigami at `M = 8`
   gives indices within about 0.01 of the analytic values.
6. **Random balance design, [EQ-SENS-RBD-001], [EQ-SENS-RBD-002].** One
   sample of `N` odd points: every input follows `tau(s_j)`, `tau(s) = 1/2 +
   arcsin(sin s) / pi`, in its own random order; for input `i` the output is
   re-sorted to the grid order and the power of harmonics `1 .. M` over `V`
   is `S_i^raw`. The bias correction of Tissot and Prieur
   `S = (S^raw - lambda) / (1 - lambda)`, `lambda = 2 M / N`, is on by
   default; the result keeps the raw means (`raw`) so the bias is visible. It
   is derived from the white-noise argument `E|c_m|^2 = |c_m^true|^2 +
   V_(-i) / N` and the tests check the prediction `E S^raw = lambda` for an
   inactive input and the exact algebra between the two means. Only first
   order is offered; RBD-FAST of Tarantola et al. for higher orders (several
   frequencies per permutation group) was left out.
7. **Independence is required.** Model entries raise `ValueError` for a
   non-identity copula with a message naming `shapley_effects`; the plain
   entries take independent marginals by construction.
8. **Plot, `visualization.plot_tornado(result, top=None, ...)`.** Horizontal
   bars sorted by swing around a dashed baseline reference; each row has a
   solid bar (input at its low value) and a hatched bar (input at its high
   value), so colour is not the only carrier of direction; the input value is
   written at each bar end; `*` marks a non-monotone input; title states the
   largest swing. Colours are the theme's observed and model hues and text is
   at least 9 pt; the result record is the only input, as for the other plots.

## Consequences

- Both estimators are deterministic given the seed. Their accuracy is
  checked against closed forms (additive linear model, product of two
  uniforms with `S = 3/7`, `S_T = 4/7`, the Ishigami function), against the
  library's Sobol estimator within combined standard errors, with the
  allowance for the truncation of the harmonic series stated in the tests.
- FAST does not reach the Sobol estimator's generality: it is exact in the
  limit of many harmonics only for smooth responses, and the sample-size rule
  makes it costly for many inputs (`N` grows linearly with `d`, so the cost
  grows with `d^2 R`). It is cheaper than pick-freeze for `d` up to a few
  tens and a smooth model, and the guide says to compare with
  `sobol_analysis` when the answer matters.
- The tornado chart's swing is not a variance share; the share column is the
  quadratic-sum share of a linear response and the documentation says so.

## Sources and what was left out

- Read in full this session: Tissot and Prieur (2012) (estimator,
  expectation with the noise term, corrected estimator, sample construction);
  the Plischke-Tarantola-Mara slides for the RBD construction.
- Confirmed bibliographically only (formulas re-derived, not copied): Cukier
  et al. (1973), Saltelli, Tarantola and Chan (1999), Tarantola, Gatelli and
  Mara (2006), Howard (1988). Whether Howard (1988) names tornado diagrams was
  not verified. Saltelli et al. (2008) is cited as already registered.
- Left out: second- and higher-order RBD-FAST, the Plischke EASI estimator
  (which works on any sample), conditional or scenario tornadoes for
  dependent inputs, tornado charts with probability-weighted ends, and the
  interference-free frequency sets of Schaibly and Shuler (the library uses a
  plain integer plan with the spacing rule above).
