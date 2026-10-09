# Reliability and Rare-Event Analysis

Reliability analysis turns an engineering margin into a probability statement.
This release uses one convention everywhere: a limit-state value `g(x)>0` is
safe and `g(x)<=0` is failure [EQ-REL-LIMIT-001]. Equality is deliberately part
of failure, avoiding method-dependent boundary behavior.

## Direct Monte Carlo

Failure probability is `P_f=P(g(X)<=0)` and reliability is `1-P_f`
[EQ-REL-PF-001]. `estimate_failure_probability` evaluates explicit input
samples, returns the event count, charged model calls, plug-in standard error,
and a Wilson interval. Zero observed failures therefore does not become a false
claim that risk is zero: the interval retains a positive upper bound.

Direct Monte Carlo is model-agnostic but inefficient when `P_f` is very small.
The expected failure count is `N P_f`; credible resolution of a rare event may
require an unaffordable number of model calls.

## Rare-event importance sampling

Proposal draws may be moved toward failure and reweighted by the target-to-
proposal density ratio [EQ-REL-IS-001]. The implementation accepts explicit
likelihood ratios so multivariate transformations remain the caller's visible
responsibility. It reports ordinary weighted-contribution uncertainty, observed
proposal failures, weight ESS, and maximum normalized weight. Incorrect support
or ratios bias the answer; a large proposal failure count alone proves nothing.

## FORM, reliability index, and design point

In independent standard-normal space, FORM finds the boundary point nearest the
origin. Its signed distance is the reliability index `beta`, and the first-order
probability is `Phi(-beta)` [EQ-REL-FORM-001].

`linear_normal_form` solves this constrained problem exactly for an affine
boundary. It returns `beta`, the most-probable failure/design point, unit normal,
boundary residual, and probability. This is a firm testable foundation, not a
claim that nonlinear FORM is already general: non-normal transformations,
gradients, iterative optimization, convergence diagnostics, and multiple design
points remain future work.

## Directional sampling

Independent standard-normal vectors separate into a uniform direction and a
chi-distributed radius. For a failure region that begins once per ray,
directional sampling integrates the radial tail analytically and samples only
directions [EQ-REL-DIR-001]. The API requires the radial entry boundary
explicitly. Non-star-shaped regions, multiple crossings, or missed failure
lobes violate this contract and need richer search/partition methods.

## Reproducible method comparison

Run `python examples/reliability_lab.py`. The lab compares direct Monte Carlo,
affine FORM, and directional sampling on the same bivariate Gaussian boundary,
then estimates a five-sigma normal tail with mean-shift importance sampling. It
writes `examples/generated/reliability-method-comparison.png` with uncertainty
bars and the analytic affine reference.

## Sources

- [REF-MELCHERS-BECK-2018, Chapters 2-6].
- [REF-OWEN-2013, Chapter 9].
- [REF-WILSON-1927].
