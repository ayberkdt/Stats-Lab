# ADR-0058: Central model contract, evaluation engine, and study accounting

- Status: Accepted (proposed during the 2026-10-16 session, workstream W1)
- Date: 2026-10-16

## Context

The core-mechanics audit (`memory/sessions/2026-10-16-core-mechanics-gap-audit.md`)
found that one bad simulator run aborted a whole stage and lost its good runs
(0.1), that `EvaluationCounter.add()` ran before the simulator call (0.2),
that a stage that raised vanished from the study accounts (0.3), that
`calibrate()` did not change the input law of later stages (0.4), that
`evaluate(Mapping)` did not check names (0.16), that NaN "not applicable"
sentinels broke dataclass equality on Python 3.13 (0.17), that
`propagate_until` had optional-stopping bias, a fixed 95% level, and the
event `Y >= t` while `propagate` uses `Y > t` (0.9), that the information and
extremes stages reused quasi-random samples and depended on stage order
(0.11), that unconverged FORM was reported without a flag (0.12), that the
lint/type gates drifted (0.18), and that the HTML report crashed without
Matplotlib (0.19). It also listed missing mechanics: a failure policy (P1.1),
stage chaining (P1.2), persistence and provenance (P1.3), an evaluation
engine (P1.5), and honest accuracy statements (P1.10).

## Decision

1. **Failure policy** (`probstat.uq.evaluation`, `ProbabilisticModel`).
   `on_failure="raise"` (default; the original exception propagates
   unchanged), `"skip"` (drop the run, log its error text), or `"record"`
   (drop the run, log its inputs and error text). A run failure is an
   exception of the simulator, a non-mapping output, a missing QoI, or a
   non-real or non-finite QoI value; invalid *arguments* always raise. No
   policy imputes a value. `EvaluationCounter` gains `failed` and
   `cache_hits`; `count` is incremented only after a successful run.
   `FailedEvaluation` records are NaN-free. Batch evaluation drops failed
   runs; a single `evaluate()` call cannot drop its own run, so under
   `"skip"`/`"record"` it raises `ModelEvaluationError` after logging (FORM,
   MCMC, and pick-freeze designs need every value). `evaluate_model` takes
   the same policy (overloaded return type: `None` marks a dropped run) and a
   `passthrough=` set of names whose values are not validated.
2. **Statistical honesty.** Dropping failures conditions every estimate on
   success; it is unbiased only when failing is unrelated to the output
   (missing completely at random). `propagate` warns with the failure
   fraction; `PropagationStudy` lists the failures and keeps `inputs`
   aligned with the kept outputs; every study record adds `failed runs`,
   `failure fraction`, and a warning; the report prints failed evaluations
   in bold.
3. **Strict names, non-float fixed values.** A mapping passed to
   `evaluate` must name exactly the random inputs; missing, unknown, and
   fixed-parameter names raise `ValueError` with each problem named. `fixed`
   may hold strings, integers, Booleans, and tuples; floats must be finite.
   All fixed values reach the model unchanged (integers are no longer turned
   into floats). Limitation: `inverse` and `design` call `evaluate_model`
   on `dict(model.fixed)` directly, so calibration and Bayesian design still
   require numeric fixed values until those packages route through the model
   (or pass `passthrough=`).
4. **Picklability and derived counters.** `FrozenMapping` replaces
   `MappingProxyType` in the model, its outputs, propagation results, and
   stage records. `probstat.uq.evaluation` also registers a `copyreg`
   reducer for `MappingProxyType`, so the read-only result fields of every
   other package pickle too (unpickling builds a proxy over a private copy).
   `dataclasses.replace(model, ...)` gives the new model its own counter
   starting at zero, linked to the source counter as `parent`: increments
   propagate upwards, so validation's conditioned models and RBDO trial
   models (which use `replace`) are still charged to the study model.
   Counter `limit`s are checked along the parent chain.
5. **Evaluation engine.** `evaluate_many(points, *, executor=None,
   batch=None, cache=None, on_failure=None)` returns an `EvaluationBatch`
   (a read-only mapping with `indices`, `failures`, `model_evaluations`,
   `failed_evaluations`, `cache_hits`). An executor runs module-level jobs
   through `executor.map`, so results are in input order and identical to
   sequential evaluation; under `"raise"` all submitted runs are counted
   before the first failure is re-raised. A model may define
   `evaluate_batch(list_of_mappings) -> list_of_mappings` (members may be
   exception instances); a wrong output count is a contract error that
   always raises. `EvaluationCache` memoises validated outputs keyed by the
   complete parameter mapping (exact match, successful runs only); repeated
   points within one batch are run once. Cache hits are never evaluations.
   `model.evaluation_context(cache=, executor=)` installs both for code that
   only receives the model; `UQStudy(..., cache=True, executor=...)` does so
   for every stage. Framework overhead fell from 5.9 to 3.7 microseconds per
   `evaluate` call (fast paths for tuple inputs and dict outputs).
   Sensitivity (`sobol_analysis`, `shapley_effects`, `morris_screening`)
   evaluates point by point through `scalar_function`, so it uses the cache
   but not the executor; parallel pick-freeze designs need a change in
   `probstat.sensitivity` (not owned by W1).
6. **NaN sentinels.** `TargetedPropagation.probability`,
   `probability_standard_error`, `probability_interval` and
   `TailAnalysis.pot_probability` are `None` when not applicable; the
   extremes summary writes `"n/a (...)"` for return periods beyond the
   sample. Genuine numerical NaN (e.g. an undefined rank correlation) is
   kept. This fixes `test_reproducible_by_seed_and_streams_advance` on 3.13.
7. **`propagate_until`.** Events are `Y > t` (`failure="above"`, as in
   `propagate`) and its complement `Y <= t` (`"below"`). `confidence=` sets
   the Student-t mean interval and the Wilson interval. The sequential run is
   a pilot that only chooses `n`; with `final="fresh"` (default) every
   reported number comes from an independent sample of size `n` on the
   stream `stage/final/k` [EQ-WF-SEQ-FRESH-001]. Given `n` that sample is an
   ordinary fixed-size Monte Carlo sample, so estimates are unbiased and
   intervals keep their nominal coverage; the price is at most
   `2 * max_samples` runs. `final="reuse"` keeps the legacy behaviour and
   warns. The pilot streams are unchanged, so `reuse` reproduces earlier
   pilots.
8. **Accuracy statements** (`probstat.propagation`). `propagate(...,
   replicates=r)` draws the design `r` times on `("propagation", design,
   "replicate", k)`; `OutputDistribution.mean_interval` and
   `exceedance_interval` give Student-t intervals over replicate estimates
   [EQ-PROP-RQMC-CI-001] (exceedance falls back to the pooled Wilson interval,
   labelled, when replicates show no spread). Without `replicates` the
   stream and outputs are unchanged. `quantile_interval` is the
   distribution-free order-statistic interval [EQ-PROP-QCI-001];
   `variance_interval` is the chi-square interval (normal outputs only, not
   robust to kurtosis) or a seeded percentile bootstrap [EQ-PROP-VARCI-001].
9. **Stage correctness.** `extremes` and `information_sensitivity` draw a
   fresh plain Monte Carlo sample from their own stream by default, so their
   results do not depend on which stages ran before; `reuse=True` reuses the
   latest propagation only when it is plain Monte Carlo (otherwise a fresh
   sample is drawn and the source says why). `design` other than `"mc"` is
   rejected. `form`, `reliability`, and `design_points` record FORM
   convergence (`"yes"` or `"no (reason)"`) and warn when unconverged. The
   reliability stage reads the flag by repeating the deterministic HL-RF
   search under a stage-local cache, so the repeat costs no evaluation.
10. **Stage chaining.** `UQStudy.adopt_posterior(inputs=None, *,
    method="gaussian_copula" | "empirical")` approximates the latest plain
    calibration posterior by marginals plus a Gaussian copula
    (`workflow.posterior_joint_input`): parametric families by AIC with a
    Spearman copula, or `EmpiricalMarginal` (type-7 interpolated empirical
    law, support cut at the draws) with a normal-score copula
    [EQ-WF-EMPIRICAL-001]. Inputs not adopted keep their prior laws and are
    made independent of the adopted ones. `UQStudy.with_inputs(law)` sets
    any law. Both record an `input_law` stage (mapped to input
    characterization) that lists posterior and adopted moments side by side.
    Every `StageRecord` states `input_law` and `input_law_hash`
    (`workflow.input_law_hash`, SHA-256 of names, marginal parameters, and
    copula). The parameter is named `inputs` (not `quantities`) because the
    calibrated objects are inputs, not QoIs.
11. **Surrogate studies.** `emulate()` keeps the child study in
    `UQStudy.children`; its records have `evaluation_kind="surrogate"`, the
    parent report and HTML render them after the emulation stage under
    "Surrogate evaluations", and the parent totals never include them.
12. **Persistence and provenance.** `StageRecord` gains `status`,
    `failed_evaluations`, `cache_hits`, `error`, `settings` (bound call
    arguments, compacted), `warnings`, `input_law`, `input_law_hash`, and
    `evaluation_kind`, plus `to_dict`/`to_json`. `UQStudy.to_dict`,
    `to_json` (`allow_nan=False`; non-finite floats become strings), and
    `save_json` add provenance (library, Python, platform, UTC creation
    time, root seed), the model, the input law, the budget, totals, and
    surrogate studies. The report and the HTML show the same provenance; the
    HTML footer names the root seed.
13. **Accounting and budget.** Every stage method runs under a `_stage`
    decorator: it binds the call arguments as settings, installs the study's
    cache, executor, and a counter limit, and records an aborted stage
    (status `"aborted"` or `"budget-exceeded"`, cost, error text) when the
    method raises after spending evaluations; argument errors before any run
    leave no record. `latest()` returns complete records only; aborted
    records still advance the stage stream and are skipped as workflow
    evidence. `UQStudy(..., evaluation_budget=N)` (completed plus failed
    runs) refuses stages with a known minimum cost above the remainder
    before they run (propagation, sensitivity, screening, risk, validation,
    fresh samples, Monte Carlo rows of reliability and AK-MCS) and stops any
    other stage at the limit.
14. **Gates.** The six `(str, Enum)` classes are `StrEnum`; `str()`,
    `format()`, and f-strings of every member equal its value (tested). The
    mypy error in `transport_plots._ramp_word` is fixed. Dev tools are pinned
    to `ruff>=0.6,<0.17` and `mypy>=1.11,<3`.
15. **Matplotlib.** `chart_theme` uses the guarded import with an
    actionable message. `study_html`/`write_study_html` degrade to a
    figure-less report that says how to install the extra; figure functions
    still raise the actionable `ImportError`.

## Evidence

- Order-statistic ranks and coverages equal SciPy `binom` values for
  `(n, p) = (100, 0.5), (1000, 0.9), (20, 0.5), (4096, 0.95)`:
  `(40, 61, 0.96480)`, `(881, 919, 0.95491)`, `(6, 15, 0.95861)`,
  `(3863, 3919, 0.95527)`; `n = 59, p = 0.05` has no lower rank. Chi-square
  quantiles match `chi2.ppf(0.975, 49) = 70.22241356643451` to 1e-12; the
  Student-t quantile matches `t.ppf(0.975, 9) = 2.262157162798205`.
- 200 seeded samples of 101 normals: median-interval coverage agrees with
  the exact rank coverage within three binomial standard errors.
- Scrambled Sobol, `x1 + x2^2` (mean 5/6), 8 x 64 points, 60 seeds: the
  replicate interval covers in at least 52 runs and is at most a quarter of
  the width of the independent-draw interval of the pooled sample.
- `propagate_until`, `Y ~ N(1, 0.5)`, 300 seeds: mean target 5% gives bias
  +1.04% (3.6 standard errors) when reusing the pilot and +0.11% with a fresh
  sample; `P(Y > 1.64) = 0.10027` at a 20% target gives +4.9% (4.2 SE)
  against -1.4% (1.2 SE). Fresh coverage: 0.927 (mean) and 0.927
  (probability), within three binomial SE of 0.95.
- Posterior adoption: prior `theta ~ N(1, 0.5)`, four observations with
  known sd 0.1. The conjugate posterior is `N(1.99010, 0.04975)`; after
  `adopt_posterior(["theta"])` a 4096-point Sobol propagation gives mean
  1.9929 and sd 0.0461, and the non-calibrated load keeps sd 1.00.
- README quick start reproduces 0.124, ('load', 'strength'), 0.0362 / 0.0395,
  49.8 exactly; `examples/uq_study_complete_lab.py` gives the same 139,821
  evaluations as before the change.

## Consequences

- Positive: a study survives bad runs when asked, never under-reports its
  cost, states which input law each number used, can be saved as JSON and
  pickled, reports honest error bars for QMC, and its sequential stage no
  longer over-states precision.
- Behaviour changes: `extremes` and `information_sensitivity` no longer
  reuse the latest propagation unless `reuse=True` (and then only plain MC);
  `propagate_until` costs about twice as many runs by default and reports
  `P[y > t]`; fixed integers reach the model as integers; `evaluate(mapping)`
  rejects extra names.
- Limitations: (a) inverse/design paths still need numeric fixed values and
  call `counter.add()` before the simulator (decision 3); (b) no
  imputation policy and no per-stage budget; (c) the posterior approximation
  is a Gaussian copula (no banana or tail dependence); (d) sensitivity is
  not parallelised (decision 5); (e) surrogate approximation error is not
  propagated into child-study results; (f) `SurrogateModel` still stores a
  `MappingProxyType`, which now pickles through the registered reducer.
