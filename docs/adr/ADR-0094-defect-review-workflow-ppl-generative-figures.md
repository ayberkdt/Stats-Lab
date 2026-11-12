# ADR-0094: Defect review of the study workflow, probabilistic programs, generative models, and figures

- Status: Accepted
- Date: 2026-11-12

## Context

A review of `probstat.workflow` (`UQStudy`, chaining, mission),
`probstat.inference.ppl`, `probstat.generative`, and `probstat.visualization`
(all modules except `visualization/foundations.py`) looked for results that
are wrong, misleading, or poorly explained. A defect counts only when a test
with an independent oracle fails on the code before the change. The review
also closes the follow-up recorded in ADR-0065: `UQStudy.sensitivity` still
called the single-quantity Sobol path with `bootstrap=200`.

## Decision

1. **`UQStudy.sensitivity` uses one shared design for every QoI.** The stage
   calls `sensitivity.sobol_analysis_all` for all QoIs of the model on the
   stage stream. Each model call returns every QoI, so the first call costs
   `size (d + 2)` runs, as before, and a later call for another QoI with the
   same `size`, `design`, `bootstrap`, and input-law hash reuses the stored
   design at no cost. The known cost given to the budget check is then zero.
   The record of a reusing stage carries the stream seed of the stage that
   ran the design. The returned type is still `ModelSobolResult`, and the
   indices of the requested QoI are bit-for-bit those of `sobol_analysis`
   with the same stream, so seeded results of one-QoI studies do not change.
   A QoI with zero variance makes the joint call fail. Each QoI is then
   estimated alone on a replay view of the model, which returns the stored
   outputs of points already run, so no simulator run is repeated. A
   failing QoI is dropped, or raised when it is the requested one. The
   summary now states the interval method. With the default
   `design="sobol"`, `bootstrap=200` gives Student-t intervals over 8
   independent randomizations of `size // 8` rows (ADR-0065), not a
   bootstrap, and the summary also lists the interval limits and the other
   QoIs that the design estimated. Before the change the record showed
   `bootstrap=200` with no interval and no method.
   - Cost: `k` QoIs cost `size (d + 2)` runs in total instead of
     `k size (d + 2)`. The requested QoI gets the same numbers at the same
     cost. The indices of different QoIs now come from one design and are
     correlated, where separate calls made them independent.
   - Accuracy: the default stays `bootstrap=200`, so seeded results stay
     the same. ADR-0065 measured the price of the 8-way split: the
     seed-to-seed standard deviation of `S_1` (Ishigami, 512 rows) is 0.025,
     against 0.016 for one design without intervals.
     `sensitivity(..., bootstrap=0)` gives the more accurate point
     estimates without intervals.
2. **Lower-tail threshold rule in `UQStudy.extremes`.** The summary recorded
   `threshold_rule = q0.95` for a lower-tail analysis whose threshold, in QoI
   units, is the 0.05 quantile, so the report printed `q0.95` next to a
   threshold close to the 5 % quantile of the output. The rule now names
   the quantile in QoI units with the QoI (`q0.05 of g`). By the type-7
   identity `quantile(-y, p) = -quantile(y, 1 - p)` this is exact.
3. **Planar and radial flow log-determinants near the invertibility
   boundary** ([EQ-GEN-FLOW-008]). `PlanarFlow` formed `w'u_hat` as
   `-1 + softplus(w'u)` and then `1 + (1 - tanh^2) w'u_hat`. For `w'u` below
   about -37 the margin `softplus(w'u)` vanished in the subtraction, so the
   log-determinant was wrong (-36.044 instead of -36.000 at `w'u = -36`) or
   raised `ValueError: math domain error` (`w'u = -40`, `z = 0`). Also,
   `1 - tanh^2` rounded to zero for `|w'z + b| > 19`, which dropped a term
   of 1.7e-7 at `w'u_hat = 1e10`. `RadialFlow` had the same cancellation in
   `beta = -alpha + softplus(beta_raw)` (-70.77 instead of -71.27 at
   `beta_raw = -36`, `z = z0`; a domain error at -40). Both layers now
   carry the margin as computed and use sums of non-negative terms:
   `tanh^2 + m sech^2` with `sech^2 = 4 e / (1 + e)^2` for the planar
   determinant and the slope of the inverse root, and `(r + m) / (alpha + r)`
   and `(r (2 alpha + r) + m alpha) / (alpha + r)^2` for the radial factors
   and the inverse. The maps are unchanged.
4. **ActNorm data initialization with a constant coordinate.** The check
   `spread <= 0` missed a constant column whose mean rounds one unit in the
   last place away from its value. For `0.1` repeated three times the mean
   is `0.10000000000000002`, the spread `1.4e-17`, and the layer got a
   log-scale of -38.8, which gave a log-density of +35.9 per point. A
   column now counts as constant when its spread is at most
   `4 eps max |x|` or its extremes are equal, and the error says
   "constant to working precision".
5. **Return-level figure title** (`visualization.plot_return_levels`). Every
   level whose limits were not both finite was titled "its upper confidence
   limit is unbounded". That was wrong for a probability-weighted-moments
   fit (`method="point"`, `nan` limits: there is no interval) and for an
   open lower limit. The subtitle also named a band that was not drawn. The
   title now states each case ("no confidence interval is available (point
   estimate only)", "... ends at U and its lower limit is unbounded",
   "... starts at L and its upper limit is unbounded", "... spans L to U").
   The band is named only when it is drawn. The title block wraps the
   subtitle between its parts.

### Examined and rejected

- `UQStudy`: two runs of eleven stages with one seed give identical JSON,
  apart from the provenance block (creation time and platform, by design).
  Pickle round trips are equal. Budget refusal, cache hits, calibration
  evaluation counts (likelihood calls times experiments), `adopt_posterior`
  summaries, propagation labels, the Morris and Shapley titles, and the
  reliability-stage inequality signs agree with the data.
- PPL: duplicate names across sample, observe, deterministic, discrete,
  plate, and Markov sites raise. Plates and chains of size 0 raise a clear
  error. Size 1, discrete laws with one support point (global, in a plate,
  in a chain), and a Markov chain after a structure-changing discrete site
  match brute-force enumeration exactly. Observing a structured or joint
  law off its support gives `-inf` at every point, and `initial_point`
  then names the data and `ModelIR.terms`.
- PPL boundary transforms: `sigmoid(u)` rounds to 1 for `u > 36.7` and
  `exp(u)` underflows for `u < -745`. The log-density there is `-inf`
  although the exact value is finite. This matters only for very small
  shape parameters, for example about 8 % of the mass of `Beta(0.05, 0.05)`
  above `u = 36.7` and about 47 % of `Gamma(0.001, 0.001)` below
  `u = -745`. A fix needs log-densities written in the unconstrained
  coordinate, which `distributions` does not provide. The limitation is
  shared with every transform-based sampler that evaluates the density at
  `x = T(u)`, so it is recorded here and not changed.
- Generative: the implicit score-matching term is `|grad E|^2 / 2 - Lap E`
  and the denoising residual is `-grad E + xi / s` (Hyvarinen 2005; Vincent
  2011), both with the correct sign. The MALA ratio uses
  `|xi|^2 / 2 - |x - y - h g(y) / 2|^2 / (2 h)`. At extreme step sizes the
  proposal is absorbed by rounding, with acceptance 1 and no movement, or
  ULA leaves the support and raises. The affine-coupling round trip at the
  tanh bound is exact to about 1e-15 relative.
- Figures: 28 f-string titles and about 70 composed titles were checked
  against their sources (Sobol, Morris, Shapley, mutual information, design
  points, violin and box plots, reliability diagram, transport plans,
  multiple imputation, method comparison, output distribution). The
  design-point title compares the system probability with the first FORM
  point, which is the plain FORM answer as stated.

### Reported to other owners

- `sensitivity.sobol_analysis_all` (and `sobol_analysis`) raise
  `ValueError: sample_size and evaluations are inconsistent` when an
  `EvaluationCache` serves some design points, because the evaluation count
  is the counter delta and cache hits are not counted. A repeat with the
  same seed under one cache reproduces it. `UQStudy.sensitivity` avoids it
  with its replay view. The library code is not changed here.

## Consequences

- Reports of sensitivity stages gain the design source, the interval
  method, the interval limits, and the list of other QoIs. A second
  sensitivity call on another QoI costs nothing. A study unpickled from an
  older version simply has no stored designs.
- The lower-tail extremes summary changes one string (`q0.05 of g`
  instead of `q0.95`). Planar and radial log-determinants change only in
  the last bits away from the boundary.
- Return-level titles for open or missing intervals read differently, and
  their subtitle can wrap onto a second line.

## Verification

- `tests/test_review_workflow_ppl_generative.py`: 15 tests. 11 fail on the
  code before this change and all 15 pass after it. Oracles: recomputation
  with `sobol_analysis` on the documented stream, with counted model runs;
  the type-7 quantile identity on the stage's own sample; 50-digit decimal
  log-determinants from `numerics.reference_exp` and `reference_log`
  (agreement within `1e-12` relative, or within `1e-15` absolute for a
  log-determinant near zero); exact constant data; and the limits stored
  in `ReturnLevel`.
- Existing tests of the touched packages pass. These are the workflow,
  study, mission, cost, surrogate, generative, and visualization suites,
  the documentation contract, and the architecture test.
- Figure audit: `datawiz.figur_denetle(fig, ortam="web")` after the
  library's own `fit_text_to_canvas` gives 0 HATA on all 8 return-level
  figures (point, open lower, open upper, and finite intervals, each in the
  light and dark palettes). The one remaining warning, the 4.06:1 contrast
  of the model-hue direct label on the light canvas, comes from the
  palette and was already there.
