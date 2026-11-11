# ADR-0014 — One central probabilistic model for every UQ workflow

- Status: Accepted
- Date: 2026-10-11

## Context

Through release 0.13 each workflow defined its own model contract:
propagation used a named `ForwardModel`, reliability a `LimitState` over
tuples, sensitivity a bare callable on tuples, and calibration only a
two-parameter line. Input uncertainty was sampled by hand in every lab with
hand-picked seeds. Correlated inputs could not be expressed at all. A study
had to restate its model for each method, and the methods could not be
chained (calibrate, then propagate the posterior, then assess risk).

Linear algebra (Cholesky, SPD solves) was duplicated in `multivariate` and
`inference`, and there was no shared optimizer or low-discrepancy sampler.

## Decision

1. `probstat.modeling.ProbabilisticModel` = `JointInput` + `ForwardModel` +
   quantities of interest (+ optional fixed parameters) is the single model
   contract. Every new workflow accepts it; older APIs stay for teaching and
   backward compatibility.
2. `JointInput` is marginals plus a Gaussian copula. The Nataf
   transformation is the one bridge between physical inputs, latent normals,
   independent standard normals, and unit-hypercube designs.
3. New foundation packages own shared numerics: `linalg` (layer 0),
   `optimization` and `sampling` (layer 1), `foundations.random_streams`.
   `multivariate` and `inference` now import `linalg` instead of private
   copies.
4. Randomness flows through named `RandomStream` children derived by
   SplitMix64, so a single root seed reproduces a full study.
5. Forward-model evaluations are counted by the model object; workflows
   report the calls they charge.
6. Layers: `modeling` sits at layer 3 beside `monte_carlo`, below
   propagation (4) and every analysis package (5+), so all of them may use it
   without cycles.

## Consequences

- Correlated, non-normal inputs are available to every method at once.
- Quasi-random designs (Sobol, Halton, LHS) are a parameter of propagation and
  sensitivity instead of lab-specific code.
- The Gaussian copula cannot express tail dependence; vine or other copulas
  would need a new `JointInput` implementation behind the same interface.
- Variance-based sensitivity refuses dependent inputs rather than returning
  numbers without an ANOVA interpretation.
- Pure-Python dense algebra limits problem size; ADR-0008's numerical-backend
  question remains open and now has one place (`linalg`) to plug into.

## Rejected alternatives

- Converting each workflow's callable shape ad hoc: keeps duplication and
  hides the joint law.
- Independent marginals only: makes correlated engineering inputs impossible
  and silently wrong when users assume independence.
- Python's `hash()` for stream derivation: salted per process, so results
  would not reproduce.
