# ADR-0133 — Uncertainty budgets, combination rules and Monte Carlo propagation

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0132 (batch evaluation contract)

## Context

Spacecraft system budgets (a link budget in dB, a pointing budget, a
perturbation budget, a mass budget) were combined by hand. The library had
`uq.NumericalErrorBudget`, additive absolute bounds with no sensitivity
coefficients; `validation.budget_standard_uncertainty`, one rectangular
conversion; and `multivariate.delta_method`, propagation of a mean and a full
covariance through a vector map. None holds a table of contributions with
evaluation types, degrees of freedom and worst-case half-widths, and none
names the rule by which lines are added. The failure the design has to
prevent is silent: a worst-case bound added to a standard uncertainty, or a
Welch-Satterthwaite factor applied to correlated inputs.

## Decision

1. **A table object and named rules.** `BudgetEntry` (name, `u`, `c`,
   Type A/B, degrees of freedom, aleatory/epistemic, random/bias, optional
   worst-case half-width, source) and `UncertaintyBudget(entries,
   correlation)`. The combined standard uncertainty is the law of
   propagation with correlation terms, `(c*u)' R (c*u)` ([EQ-UQ-BUDGET-001]).
2. **Kind is part of the result.** `root_sum_square`, `worst_case_sum`,
   `mixed_rule` and `law_of_propagation` return `CombinedUncertainty`
   carrying the rule and a `QuantityKind` (standard uncertainty or worst-case
   bound). The record defines no arithmetic or ordering. `worst_case_sum`
   takes a `basis`: half-widths give a bound; standard uncertainties give the
   largest `u_c` under any correlation, a conservative standard uncertainty,
   not a bound. `mixed_rule` has a `standard` level (biases `|c| u`) and a
   `bound` level (biases `|c| a`, random part `k` times the root-sum-square)
   that requires the factor `k` explicitly. The mixed rule is a system
   engineering convention; the Guide does not define it ([EQ-UQ-BUDGET-004]).
3. **Welch-Satterthwaite with correlation.** The formula assumes uncorrelated
   inputs. With a correlation matrix the default evaluates it with the
   correlated `u_c` in the numerator, a heuristic with no derivation, and the
   result carries a warning; `correlated="error"` refuses ([EQ-UQ-BUDGET-002]).
   The Student-t factor comes from `distributions.StudentT` (the normal
   quantile for infinite degrees of freedom); non-integer degrees of freedom
   are used as they are unless `truncate_dof=True` (GUM G.4.1 Note 1).
4. **Variance share under correlation.** `share_i = c_i u_i (sum_j r_ij c_j
   u_j) / u_c^2`: each covariance term is split equally between its inputs.
   The shares add to one, reduce to the usual ratio when independent, and may
   be negative ([EQ-UQ-BUDGET-005]). Other apportionments exist (Shapley);
   this one is cheap and exact in sum.
5. **Subsets.** `select` and `group_by` restrict the correlation matrix to the
   chosen entries; correlation with entries left out is dropped, so a
   sub-budget is the uncertainty of that subset alone.
6. **`budget_from_model`.** Coefficients by `optimization.central_gradient`
   (its default step rule), with `central_hessian` for an optional check:
   for Gaussian inputs the mean shift `tr(H S)/2` and extra variance
   `tr(H S H S)/2` relative to `u_c` flag a poor linearisation. The
   third-derivative terms of the full second-order propagation are not
   computed, so the check is a diagnostic, not a correction
   ([EQ-UQ-BUDGET-006]).
7. **Monte Carlo and validation.** `monte_carlo_budget` implements Sections
   7.3-7.7 of JCGM 101:2008 with a fixed number of trials, reporting the
   estimate, `u(y)`, the probabilistically symmetric and the shortest coverage
   intervals (Section 7.7.2) and estimated Monte Carlo errors of each
   ([EQ-UQ-BUDGET-007]). `compare_with_law_of_propagation` implements Section
   8.1.3 (endpoint differences against the tolerance of Section 7.9.2) and
   reports a proxy for the accuracy requirement of Section 8.2
   ([EQ-UQ-BUDGET-009]). Seeds go through `foundations.random_streams`: each
   input draws from a child stream. (The proxy applies to a result of fixed
   size; see item 10 for an adaptive result.)
8. **`correlation` means two things, stated each time.** In a budget it is the
   correlation of the input *estimates* (GUM 5.2.2). In `monte_carlo_budget`
   it is the correlation of a Gaussian copula on normal scores
   ([EQ-UQ-BUDGET-008]); it equals the Pearson correlation only for normal
   marginals.
9. **Placement.** `probstat.uq`, layer 2, importing only `distributions`,
   `optimization`, `linalg` and `foundations`.
10. **Adaptive Monte Carlo (JCGM 101, 7.9.4).**
    `adaptive_monte_carlo_budget` runs batches of `M = max(J, 10^4)` trials,
    `J = ceil(100/(1 - p))` (step b; `batch_size=None` means this, an explicit
    value is allowed because the Supplement calls the choice arbitrary). For
    each batch it forms the estimate, `u(y)` and the endpoints of the chosen
    interval (symmetric or shortest), from the second batch on the standard
    deviations of the four averages (steps g and h), `delta` from the `u(y)`
    of all values (steps i and j, `numerical_tolerance`), and stops when no
    `2 s` exceeds `delta` (step k; equality stabilises). The final record is
    a `MonteCarloResult` subclass computed from all values, plus the batch
    count, the four deviations, `delta`, `ndig`, the tracked interval and
    `stabilized` ([EQ-UQ-BUDGET-010]). The stopping rule and the deviation
    are separate pure functions (`adaptive_stopping_rule`,
    `average_standard_deviation`) so that they are tested on hand-built
    sequences without sampling. When `max_batches` (at least 2, because step
    f always makes a second application) is used up, the function returns
    with `stabilized=False`; it never raises for that and never reports
    stabilisation it did not test. A model with zero spread leaves `delta`
    undefined and raises. `tolerance_divisor=5` is the `delta / 5` of
    Section 8.2.
11. **Batch contract.** Batch `b` draws from the child stream `("batch", b)`
    of the seed, so its points do not depend on how earlier batches were
    evaluated or on how many batches follow. The procedure form
    `adaptive_monte_carlo_budget_procedure` yields one `EvaluationRequest`
    per batch (ADR-0132) and the plain function is a thin wrapper that runs
    it with the pointwise model; `monte_carlo_budget` shares the sampling and
    summarising helpers, so its results are unchanged. The three ways of
    driving it (`run_procedure` with any `max_batch_size`, an ask-tell
    session, the plain function) give equal records.
12. **Validation uses the clause's own statistic.** For an adaptive result
    `compare_with_law_of_propagation` sets `monte_carlo_adequate` by testing
    twice each of the four deviations against `delta / 5`, with `delta` from
    the GUM `u(y)` of the comparison (Section 8.2 replaces `delta` by
    `delta / 5` in step k). The deviations cover the interval the run
    tracked; asking for the other interval gives `False` with the basis
    `ADEQUACY_OTHER_INTERVAL` rather than a guess. A fixed-size result keeps
    the standard-error proxy, labelled `ADEQUACY_PROXY` in the new
    `GumValidation.adequacy_basis` field.

## Consequences

- A budget can be rendered (`to_rows`, `to_markdown`, `to_text`) and its
  numbers are traceable to a rule and a kind.
- The Monte Carlo output reports its own errors so that a caller judges it on
  them; tests do the same and pin no seeded digits.
- Pure-Python Monte Carlo costs about a second per 10^5 trials of a cheap
  model (more with a copula, one quantile call per input and trial).
- The adaptive procedure keeps every model value in memory (`h M` floats) and
  its stopping test uses a standard deviation estimated from `h` batch values
  that can be as few as two, so it can stop early by luck. Over 200 seeds the
  additive and the lognormal test cases met `delta` at both endpoints in
  about 84 % and 86 % of runs, not 95 %; the tests assert a deliberately
  lower rate on the binomial scale. Section 8.2 is the clause's own remedy
  (`delta / 5`).
- The Gaussian copula is one dependence model. It is a choice of the caller,
  not implied by a coefficient.

## Sources and what was left out

JCGM 100:2008 and JCGM 101:2008 were opened from the BIPM PDFs and the cited
clauses read: GUM 4.2, 4.3.4, 4.3.7, 4.3.9, 5.1.2, 5.2.2, 6.2, G.4.1;
Supplement 1: 5.3, 6.4.2-6.4.8, 7.2-7.7, 7.9.2-7.9.4, 8.1-8.2, 9.2.3. The
Supplement's additive rectangular example (interval `+/- 2 sqrt(3)(2 -
(3/5)^(1/4))`) is reproduced by the tests against an independent Irwin-Hall
oracle. The copula reference is bibliographic only; the construction is
re-derived. Left out deliberately: the histogram variant of Section 8.2
Note (cell frequencies updated as the computation proceeds, to avoid holding
all values), the stabilisation test on `y` and `u(y)` only (7.9.4 Note 5),
JCGM 102 (multivariate outputs, not opened), higher-order
Taylor terms in the law of propagation, Type B degrees of freedom from a
relative uncertainty of `u` (GUM G.4.2; the caller supplies `dof`), and any
mission-standard table of budget terms.

## Verification

`tests/test_uncertainty_budget.py`: closed forms for linear models and for the
two-term correlation formula; `multivariate.delta_method` as an independent
oracle; a hand-computed Welch-Satterthwaite example (`900/73`) with the t
quantile pinned from R; the exact lognormal law of a product/quotient; the
Irwin-Hall law of a sum of rectangles (also for the adaptive procedure,
with the success fraction judged on the binomial scale); the stopping rule
on hand-built sequences of batch statistics; a replay of the loop with
`statistics`; scheduling independence of the batch contract; the shortest
interval equal to the
symmetric one for a symmetric law and strictly shorter for an exponential
law; copula marginals and variances; the decision rule of the validation.

## References

[REF-JCGM-100-2008], [REF-JCGM-101-2008], [REF-WELCH-1947],
[REF-SATTERTHWAITE-1946], [REF-NELSEN-2006].
