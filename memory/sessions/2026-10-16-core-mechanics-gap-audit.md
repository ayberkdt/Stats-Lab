# 2026-10-16 Core-mechanics gap audit

Scope: the mechanics every analysis depends on — the model contract
(`modeling`, `uq`), input laws (`distributions`, copula), sampling and
propagation, the `UQStudy` orchestration, and the inference/optimization
machinery that calibration and reliability call. No library code was changed
in this session; every item below was reproduced with a script or a grep
(scripts were kept outside the repository).

State at audit time: version 0.15.0, 1,178 unittest tests (1 failure, 70
skipped because Matplotlib was absent), `ruff check .` 6 errors with ruff
0.16.8, `mypy src/probstat` 1 error with mypy 2.3.1.

## Priority 0 — verified defects in fundamental mechanics

| # | Defect | Evidence | Consequence |
|---|---|---|---|
| 0.1 | One NaN, inf, missing QoI, or exception from the simulator aborts the whole stage; partial outputs are discarded. | `uq/taxonomy.py:203-212`, nothing catches it in `propagation`, `sensitivity`, `workflow/study.py` | A 1,000-run study loses 499 good runs on one bad run. |
| 0.2 | `EvaluationCounter.add()` runs before the simulator call. | `modeling/model.py:183` | Failed calls are counted as "complete" evaluations. |
| 0.3 | Evaluations of a failed stage vanish from the study accounts. | `study.total_evaluations` = 1000 while `model.evaluations` = 3000 after a mid-stage exception | The report under-states real cost. |
| 0.4 | `calibrate()` does not change the input law used by later stages. | `self._model` assigned only at `workflow/study.py:495`; prior N(1,0.5) → posterior 1.997±0.021, later `propagate` still gives mean 0.999, sd 0.500 | Breaks the "one model, every question" promise: forward/reliability after calibration silently use the prior. |
| 0.5 | No path from a posterior back to a `JointInput`. | `JointInput` supports only marginals + Gaussian copula (`modeling/inputs.py`) | Root cause of 0.4. |
| 0.6 | `UQStudy.propagate()` fails for more than 16 inputs with its default design. | Sobol direction numbers stop at d=16 (`sampling/designs.py:74-91`); `sobol_analysis` falls back silently to Halton for d≥9 | The default entry point crashes on a moderate-dimension model. |
| 0.7 | BFGS stopping rule scales with \|f\|: `‖g‖ ≤ tol·max(1,\|f\|)`. | `optimization/solvers.py:330`; with an additive constant 1e9 it returns the start point (5,5), `converged=True`, ‖g‖=8 | Negative log-likelihoods with large constants stop early, silently; MLE, MAP, Laplace and calibration inherit it. |
| 0.8 | Finite-difference step floored at 1e-6 absolute, no one-sided fallback. | `optimization/solvers.py:126` | 17.5 % gradient error for parameters on a 1e-6 scale; crash near domain boundaries. |
| 0.9 | `propagate_until` has optional-stopping bias and a fixed 95 % interval. | `workflow/study.py:808-842`; p=0.02: +9.6 % bias, Wilson coverage 0.938; lognormal mean: coverage 0.853 vs 0.925 at fixed n | The "stop at target accuracy" claim is not statistically valid. Its exceedance event is Y≥t while `propagate` uses Y>t. |
| 0.10 | Error bars under quasi-random designs are not honest. | single scrambled Sobol run, N=1024: reported SE 2.2e-2, true replicate spread 7.3e-4; Sobol-index bootstrap under QMC (`sensitivity/model_based.py:263-268`) gives coverage 1.00 | Users cannot tell how accurate a QMC result is. |
| 0.11 | `information_sensitivity` and `extremes` reuse the latest propagation sample, which is Sobol by default, and the result depends on stage order. | `workflow/study.py:2101-2138`, slice near 2339; MI[load] 0.378 (Sobol reuse) vs 0.65 (MC), true ≈0.63; `extremes` ξ −0.259 vs −0.103 depending on order | Wrong numbers by default; breaks reproducibility-by-seed. |
| 0.12 | Non-converged FORM is reported without its flag. | `workflow/study.py:1299-1315` | SORM and design-point IS build on an unconverged point silently. |
| 0.13 | Incomplete beta loses accuracy for large first shape near x→1. | `distributions/_special.py:604,661` | `StudentT(1e14).sf(3)` = 1.34928e-3 vs 1.34990e-3; Beta/F with large parameters off by 1e-9 to 1e-7. |
| 0.14 | Cauchy cannot be a model input. | `RandomInput("x", StudentT(1))` raises inside `hasattr` because `mean` raises (`modeling/inputs.py:248`) | Heavy-tailed inputs are excluded by an interface accident. |
| 0.15 | PCE solves normal equations. | `surrogates/polynomial_chaos.py:573-584`; degree-20 Hermite fit fails in 33/40 seeds, QR (`linalg/qr.py`, already present) succeeds | Surrogate fitting fails where it should not. |
| 0.16 | `evaluate(Mapping)` does not check names. | `modeling/model.py:179-184` | Missing inputs reach the simulator; misspelled names pass silently; a user value for a fixed parameter is overwritten silently. |
| 0.17 | Python 3.13 test failure from NaN sentinels in result dataclasses. | `tests/test_workflow_propagate_until.py:49`; 3.13 dataclass `__eq__` compares fields, `nan != nan`; 76 `math.nan` uses in `src/` | CI matrix includes 3.13; the "not applicable" value should be `None`, not NaN. |
| 0.18 | Lint/type gate drift. | 6 × ruff UP042 (`uq/taxonomy.py`, `uq/verification.py`, `uq/workflow_record.py`), 1 × mypy `visualization/transport_plots.py:101`; `pyproject.toml` pins only lower bounds | CI fails whenever a newer ruff/mypy is installed. |
| 0.19 | HTML report crashes without Matplotlib. | `visualization/theme.py:301` imports Matplotlib outside the guarded import at 174-178 | The README quick start does not mention the extra. |

## Priority 1 — missing fundamental mechanics

These are not extensions; each one is a contract the central model lacks.

1. **Simulator-run robustness.** A failure policy (raise / skip-and-record /
   impute), failed runs logged with their inputs, counted separately, and
   partial stage results kept. Global and per-stage evaluation budgets.
2. **Stage chaining.** Calibration produces an input law (empirical,
   KDE, or mixture joint) that later stages use, or an explicit
   `study.with_inputs(...)`; each report section states which input law it
   used. `emulate()` keeps its child stages in the parent report and carries
   the surrogate error forward instead of only printing it.
3. **Persistence and provenance.** Models, results and studies cannot be
   pickled (`MappingProxyType` in `modeling/model.py:110`); there is no
   JSON export, save/load or resume anywhere in `src/probstat`. Reports
   record no library version, Python version, timestamp, model hash, or
   stage settings, and the HTML never prints the root seed while claiming
   reproducibility.
4. **Model-contract breadth.** Vector/field/time-series QoIs (today every
   QoI is one float); discrete and categorical inputs (Poisson, Binomial,
   Categorical are rejected; a duck-typed shim produces a wrong copula
   mapping); non-float fixed parameters (strings, options).
5. **Evaluation engine.** A batch/vectorised `ForwardModel`, optional
   parallel execution (`concurrent.futures`), an input-keyed cache so
   stages reuse runs, pick-freeze runs shared across all QoIs (today a second
   QoI costs a second full design), and a subprocess / file-template wrapper
   for external simulators. The quick start uses 42,049 evaluations with no
   reuse between stages; framework overhead is ≈21 µs per call.
6. **Epistemic versus aleatory separation.** `UncertaintyNature` exists
   (`uq/taxonomy.py:14`) but is descriptive only: `RandomInput` has no
   nature, and there is no interval, p-box, or nested (double-loop)
   propagation. For a VV-UQ library that cites Oberkampf and Roy this is
   the largest conceptual gap. Calibration also treats every random input as
   an unknown constant (`inverse/nonlinear.py:531`).
7. **Automatic differentiation.** Every gradient is a central difference:
   NUTS/HMC (`inference/hamiltonian.py:548`), ADVI
   (`inference/variational.py:479`), the PPL (`inference/ppl/_program.py:1372`),
   BFGS, FORM (`reliability/model_based.py:533`), Laplace. That costs 2d+1
   evaluations per gradient (BFGS at d=50: 6,692 vs 356 evaluations with an
   analytic gradient). A forward-mode dual-number type is the minimum.
8. **Distribution contract.** No base class enforces the protocols in
   `foundations/protocols.py`; `log_cdf` is absent everywhere, `log_sf`
   and `entropy` almost everywhere, discrete laws have no `isf`, and
   `Categorical` has no `cdf`, `quantile`, `mean`. Undefined moments
   either raise or return inf depending on the family. `cdf/sf/pdf` raise
   on ±inf while `quantile(0/1)` returns ±inf. Discrete `quantile` returns
   a float that `pmf` rejects. All 23 samplers reject `RandomStream`
   (they call `random.Random(seed)` directly) and accept negative, str or
   float seeds.
9. **Dependence beyond the Gaussian copula.** No t, Clayton, Gumbel, Frank or
   vine copula: no tail dependence can be modelled.
10. **Accuracy reporting.** Randomised-QMC replicate intervals in
    `propagate`; confidence intervals for quantiles and the variance;
    replicate- or block-bootstrap for Sobol indices under QMC; NaN (not 0.0)
    for second-order indices that were not computed
    (`sensitivity/model_based.py:294`).
11. **Optimizer basics.** Box bounds or constraints, a Wolfe line search,
    an initial Hessian scaling, multi-start, and complex-step or adaptive
    finite differences.

## Priority 2 — extensions that build on the fundamentals

- MCMC: bulk/tail ESS, ESS on split chains (`inference/mcmc.py:1138`
  uses whole chains), quantile MCSE, automatic warnings (R-hat > 1.01,
  divergences, E-BFMI < 0.3; a 10-D funnel returns a biased result with no
  warning), multinomial NUTS, a built-in SBC harness, PSIS-LOO/WAIC/Pareto-k.
- GP: one factorization per likelihood-and-gradient evaluation (today it is
  done twice, `surrogates/gaussian_process.py:900-920`, and K⁻¹ is
  formed explicitly at 426); adaptive jitter; more kernels; a mean function;
  low-rank or inducing-point path (n=500 fit ≈ 2 h today).
- PCE: LARS/OMP sparse fit with adaptive degree by LOO; more bases.
- Sampling: the full Joe-Kuo table, Owen scrambling, a power-of-two warning,
  a proper Halton randomization (digit permutation keeps 0 fixed: mean 0.445
  over 200 seeds), maximin and Iman-Conover LHS (correlated LHS keeps
  stratification on the first input only).
- Sensitivity: given-data methods that reuse existing samples, DGSM,
  FAST/RBD.
- Distributions: Laplace, Rayleigh, PERT, Pareto, von Mises, discrete
  uniform, hypergeometric; generic Affine, one-sided Truncated, Mixture and
  Empirical/KDE wrappers that satisfy `ContinuousMarginal`; Dirichlet,
  multinomial, multivariate t, singular MVN; MLE for the remaining families.
- Verification: connect `uq/experiment.py` (`ComputationalExperiment`,
  `refinement_study`) to `UQStudy.verify`, which today takes user-computed
  grid values.

## What was checked and holds

- README quick-start numbers reproduce exactly (0.124, 0.0362, 0.0395, 49.8).
- Stages that draw their own samples are order-independent under the seed
  tree; evaluation counts match real calls in every successful stage.
- Scrambled Sobol beats MC as expected on smooth functions and works with
  correlated inputs through `from_unit`.
- NUTS passed a 1,000-replicate SBC check on a normal-normal model; split
  and rank-normalized R-hat and Geyer ESS are implemented; 4 chains are the
  default.
- Gamma, chi-square, normal and moderate-parameter t/F are accurate to
  ≤1e-13 against a 70-digit reference.

## Documentation drift

- `memory/MEMORY.md` still says the current release is 0.13.0 and points
  to UQ-7 as the next step.
- `memory/ROADMAP.md` lists as open several items that exist: Gamma, Beta,
  chi-square, Student-t, F and logistic (Phase 2), multilevel Monte Carlo
  (UQ-3), and Shapley effects (UQ-5). The roadmap cannot be used to plan
  the work above until it is reconciled.

## Suggested order

1. P0 correctness fixes 0.1-0.3, 0.7, 0.9-0.12, 0.17-0.18 (small, local,
   each with a regression test).
2. Model contract: failure policy, strict names, picklable model and results,
   batch/parallel/cache hook (P1.1, P1.3, P1.5) — every stage benefits.
3. Stage chaining and posterior-to-input law (P0.4-0.5, P1.2).
4. Distribution base class and seed handling (P1.8), then discrete inputs and
   non-Gaussian copulas (P1.4, P1.9).
5. Epistemic/aleatory separation (P1.6) and forward-mode AD (P1.7), each
   behind an ADR.
