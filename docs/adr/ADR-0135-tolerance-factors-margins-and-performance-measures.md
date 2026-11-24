# ADR-0135 — Normal tolerance factors, margins, and technical performance measures

- Status: Accepted
- Date: 2026-11-24
- Extends: the Wilks order-statistic bounds of `probstat.propagation.wilks`.

## Context

A design team states requirements as "the link closes with 99 % probability
at 90 % confidence" or "dry mass stays below the allocation". The library had
distribution-free Wilks bounds but no normal-theory tolerance factor, no
quantification of margin against uncertainty, and no way to follow a
performance measure over project milestones. Margins in practice are fixed
percentages whose meaning (relative to the estimate or to the allocation) is
often ambiguous and whose probabilistic content is unstated.

## Decision

1. **Tolerance factors in `propagation/tolerance.py`.**
   - One-sided exact factor `k = t'(gamma; n-1, z_p sqrt(n)) / sqrt(n)`
     ([EQ-PROP-TOL-001]). The library has no noncentral `t` and none is added
     to `probstat.distributions`: the distribution function is a private
     integral of `Phi(t s - delta)` against the chi density of
     `S = sqrt(V / nu)` ([EQ-PROP-TOL-002]), evaluated by the adaptive
     Gauss-Kronrod rule of `probstat.numerics` over the range holding all but
     `2e-17` of the mass, and inverted by Brent's method from
     `probstat.optimization`.
   - Two-sided exact factor: the integral equation over the standardized
     sample-mean deviation `z` with the chi-square survival function
     ([EQ-PROP-TOL-003]). The half-width `r(z)` does not depend on `k`, so it
     is computed once on a fixed composite 15-point Kronrod rule over
     `[0, 8.5]` and the root search in `k` reuses it.
   - Howe's approximation as a named alternative ([EQ-PROP-TOL-004]); the
     docstring and tests record its error at `p = gamma = 0.95` (+2.1 % at
     `n = 3`, -0.3 % at `n = 10`, below 0.002 % at `n = 1000`). Exact is the
     default. There is no approximate one-sided method.
   - Public surface: `normal_tolerance_factor`, `normal_tolerance_bound`,
     `normal_tolerance_interval`, `normal_tolerance_sample_size` (smallest `n`
     whose factor is at most a target), and `normality_price`, which reports
     the normal and the Wilks bound of one sample together with the Wilks
     sample-size requirement. Normality is stated as an assumption in the
     module and every docstring that depends on it.
2. **Margins in `propagation/margins.py`.**
   - `Requirement` is an upper limit, a lower limit, or a band.
     `quantify_margin` returns `M`, `U` and `M/U` per side
     ([EQ-PROP-MARGIN-001]) with `U` from Wilks, normal tolerance, or given
     moments; the binding side of a band is the one with the smaller ratio.
     `U <= 0` (bound on the safe side) is defined to give an infinite ratio
     for a positive margin instead of a division by zero or a negative ratio.
   - `quantify_margin_mixed` takes a `ProbabilityBox` or
     `MixedPropagationResult` and splits `U` exactly into
     `U_aleatory + U_epistemic`: the aleatory part is the distance to the
     quantile at the most favourable outer point. This is a convention of
     this library, stated in the docstring, not a definition taken from the
     QMU literature; the inner-loop sampling error and the coverage of the
     epistemic region by the outer points are not accounted for.
   - `compliance_probability` reuses
     `probstat.inference.clopper_pearson_interval` and adds the one-sided
     exact lower bound (the two-sided interval at level `2 conf - 1`).
   - Contingency arithmetic is explicit ([EQ-PROP-MARGIN-002]):
     `predicted = basic (1 + g)`, margin relative to the estimate or to the
     allocation, `convert_margin_fraction` between them. All percentages are
     arguments; no agency value is encoded. The probability bridge
     ([EQ-PROP-MARGIN-003]) is `contingency_for_confidence` (type-7 quantile,
     or the Wilks bound with a statistical confidence) and
     `confidence_of_contingency`.
   - `TechnicalPerformanceMeasure` holds ordered `Milestone` records
     (estimate, standard uncertainty or samples, planned value, planned
     margin) and reports the margin history, compliance probabilities, the
     gap to the planned margin, and flags (`requirement-violated`,
     `below-plan`, `low-ratio`) with a plain-text table. A milestone may carry
     samples or a standard uncertainty, not both; a compliance probability
     from a standard uncertainty assumes a normal law centred at the estimate.
3. **Layering.** Both modules live in `propagation` (layer 4) and import only
   `foundations`, `numerics`, `optimization`, `statistics`, `distributions`
   and `inference` (layers 0-2) plus the sibling `wilks` and `mixed` modules.

## Sources and verification

- NIST/SEMATECH e-Handbook, Section 7.2.6.3 (read): Howe's formula and the
  noncentral-t one-sided factor.
- NASA/SP-2016-6105 Rev 2 web edition, Sections 6.2 and 6.7 (read through a
  page summary): tracking of technical performance measures and performance
  margins. No percentage or threshold was taken from it.
- Howe (1969), Krishnamoorthy and Mathew (2009), Odeh and Owen (1980),
  Helton (2011), and Pilch, Trucano and Helton (SAND2006-5001) were confirmed
  bibliographically only; the mathematics was re-derived. No tabulated value
  was copied. The QMU definitions here are stated in this ADR and the module
  docstring and are consistent with, but not copied from, that literature.
- Oracles: base R 4.6.2 `qt(p, nu, ncp)` for the one-sided factor (agreement
  to `1e-9` at `n <= 100`; at `n = 1000` R warns about lost precision and the
  library factor instead satisfies the defining integral, checked by an
  independent Simpson quadrature and by R's `integrate`); R `qchisq` for
  Howe; an independent reversed-order integration in R (`integrate`,
  `uniroot`) and in the tests for the two-sided factor; seeded Monte Carlo
  coverage on the binomial standard error; R `binom.test` for the
  Clopper-Pearson interval; closed forms for the contingency arithmetic and
  the mixed split.

## Consequences

- A user can state a normal-theory 95 %/95 % bound from ten samples and see
  next to it what the distribution-free bound needs (59 samples).
- The two-sided exact factor costs about half a second per evaluation
  (several hundred root solves), so `normal_tolerance_sample_size` with the
  exact two-sided method takes seconds; Howe's method is instant.
- The two-sided equation cancels two normal probabilities, so `p` within
  `1e-10` of one is rejected.
- Left out deliberately: noncentral-t as a public distribution, tolerance
  factors for non-normal families (log-normal users transform the data),
  regression and random-effects tolerance intervals, simultaneous
  tolerance limits, and sample-based confidence statements for the epistemic
  extremes of a probability box. Margin policies of any agency are left to
  the user's inputs.
