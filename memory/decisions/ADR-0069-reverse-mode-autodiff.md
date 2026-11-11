# ADR-0069: Reverse-mode automatic differentiation and gradient-based inference

- Status: Accepted
- Date: 2026-11-11

## Context

ADR-0065 added forward-mode automatic differentiation (`numerics.Dual`,
`numerics.admath`) and wired it into HMC, NUTS, ADVI, Laplace and the
probabilistic-programming layer, with a recorded finite-difference fallback.
Its "Not done" list names reverse mode. A vector-tangent forward evaluation
carries a length-`d` tangent through every operation, so one gradient costs
about `d` float evaluations; for `d` in the tens or hundreds this dominates
the sampler's arithmetic even though only one call of the user function is
made. Reverse (adjoint) mode records the evaluation once and obtains the
whole gradient from one backward sweep whose work is a small multiple of the
evaluation's, independent of `d` (Griewank and Walther 2008, Chapters 3 and
4; Baydin et al. 2018, Section 3.2). Second derivatives follow by
differentiating the sweep in forward mode (Griewank and Walther 2008,
Chapter 5; Pearlmutter 1994).

## Decision

1. **`probstat.numerics.reverse_autodiff`** (exported from
   `probstat.numerics`): `ReverseVariable` records each operation on an
   append-only tape as (parent indices, local partials) and supports the
   same programs as forward mode: `+ - * / **`, comparisons, `bool`,
   `hash`, `abs`, and every `admath` function (exp, expm1, log [base],
   log1p, log2, log10, sqrt, pow, sin, cos, tan, asin, acos, atan, atan2,
   sinh, cosh, tanh, erf, erfc, lgamma, digamma, hypot, fabs, fsum,
   isfinite/isnan/isinf). Drivers: `reverse_value_and_gradient`,
   `reverse_gradient`, `vector_jacobian_product` (seed `ybar = w`),
   `reverse_jacobian` (one recording, one sweep per output),
   `reverse_value_gradient_and_hessian` / `reverse_hessian` (forward over
   reverse: the tape is recorded on first-order duals `x_i + e_i eps`, so
   each input adjoint is a dual whose tangent is a Hessian row; symmetrized
   like the forward-mode Hessian) and `hessian_vector_product` (single
   tangent `v`, never forms `H`). Equations: [EQ-AD-REVERSE-001],
   [EQ-AD-FOR-001].
2. **Iterative sweep, flat storage.** Nodes store parent indices, never
   references to other variables, and the backward sweep is a plain loop
   from the output index down; float-zero adjoints are skipped. Tapes of
   any length are therefore free of recursion-limit and deep-deallocation
   problems.
3. **Errors.** `float(v)`, `int`, `round`, `floor`, `ceil`, `trunc`,
   `complex`, `//`, `%`, indexing, every `math.*` function and float-only
   validators raise `AutodiffUnsupportedError` (a `TypeError`), as for
   duals. Combining a `ReverseVariable` with a `Dual`, or variables of two
   tapes (nested reverse mode), raises `AutodiffUnsupportedError`;
   constructing `ReverseVariable` directly raises `TypeError`. Points with
   an unbounded or undefined derivative raise the new
   `NonDifferentiableError` (a `ValueError`, so the samplers and the PPL
   map it to `-inf`): `sqrt` at 0, `asin`/`acos` at +-1, `0 ** c` with
   `c < 1` (`c != 0`), `atan2(0, 0)`, a variable exponent of a base
   `<= 0`, a constant negative base with a variable exponent. The kinks
   with bounded one-sided derivatives keep the forward-mode convention
   (derivative 0 at `abs(0)` and `hypot(0, ..., 0)`, an element of the
   subdifferential, Griewank and Walther 2008, Chapter 14), so forward and
   reverse mode agree everywhere both are defined. `trigamma` of a
   variable and the Hessian of `digamma` raise `AutodiffUnsupportedError`.
4. **`admath` dispatch without an import cycle.** `numerics.autodiff`
   defines a private base class `_TapeOperand`; every `admath` function
   hands such an operand to its `_admath` method before falling back to
   `math`, and the `is*` predicates and comparisons unwrap it.
   `ReverseVariable` subclasses it, so `autodiff` never imports the
   reverse-mode module.
5. **`GradientEvaluator` moved** from `numerics.autodiff` to the new
   `numerics.gradient_oracle` (it needs both modes; the public name
   `probstat.numerics.GradientEvaluator` is unchanged). Options
   (`numerics.GRADIENT_OPTIONS`): `"auto"` (forward mode with
   finite-difference fallback, unchanged), `"auto-reverse"` (reverse mode
   with the same fallback), `"autodiff"` and its alias `"forward"`,
   `"reverse"` (one mode, raise if unsupported), `"finite-difference"` and
   its alias `"finite"`, or a callable. Recorded `method`: `"autodiff"`
   (forward mode, the ADR-0065 label), `"reverse"`, `"finite-difference"`,
   `"analytic"`. Reverse mode skips the sweep when the recorded value is
   not finite.
6. **Samplers.** `hamiltonian_monte_carlo`, `no_u_turn_sampler`, `advi`
   and `laplace_approximation` accept every option above. Laplace with
   `"reverse"`/`"auto-reverse"` passes the reverse-mode gradient to BFGS
   and takes the forward-over-reverse Hessian (`hessian_method="reverse"`).
   Counting is unchanged and honest: one recorded call of the log-density
   is one log-density evaluation and one gradient evaluation; the backward
   sweep calls nothing; an `"auto-reverse"` fallback costs one failed
   recorded call per chain, exactly like `"auto"`.
7. **Default of the samplers stays `"auto"` (forward mode).** Reverse mode
   is faster from about `d = 10` (Verification), but not uniformly at
   `d = 2` (one of two repeated runs was slower), and switching the default
   would change the recorded `gradient_method` and, at rounding level, the
   draws of every seeded run. It is therefore opt-in for direct sampler
   calls.
8. **PPL.** In `probstat.inference.ppl` the option `"auto"` (and `None`)
   now means `"auto-reverse"` (`ppl_gradient_option`) for `sample`,
   `fit_advi`, `laplace`, `ModelIR.gradient` and
   `ModelIR.gradient_evaluator`: a traceable program (model code and
   distribution objects that accept the variables) is differentiated in
   reverse mode, anything else falls back to central differences and the
   method and reason are recorded in `PosteriorSamples.gradient_method`
   and `gradient_fallback_reason`. Density mode passes `ReverseVariable`
   values through `log_pdf` results and deterministic sites like duals.
   `"autodiff"`/`"forward"` still select forward mode. The distribution
   classes of `probstat.distributions` still validate parameters as reals,
   so a latent parameter of a library distribution remains non-traceable
   (ADR-0065, item 5).

## Consequences

- Seeded runs of `ppl.sample`, `ppl.fit_advi` and `ppl.laplace` on
  traceable programs now use reverse-mode gradients; draws equal the
  forward-mode draws to rounding (tested to 1e-9 over 240 iterations), and
  `gradient_method` reads `"reverse"` instead of `"autodiff"`. One existing
  assertion (`tests/test_mcmc_diagnostics_autodiff.py`, dual-capable PPL
  model) was updated accordingly. Programs that fall back are unaffected.
- `probstat.numerics.autodiff.GradientEvaluator` and its private type
  aliases no longer exist at that module path; the package-level name is
  unchanged.
- The forward-mode `admath` functions take one extra `isinstance` check for
  non-dual arguments.
- Reverse mode keeps every intermediate on the tape, so memory grows
  linearly with the operation count (no checkpointing).
- Stability: as in ADR-0065, the module is experimental inside the stable
  `numerics` package until the next minor release.

## Verification

Tests: `tests/test_reverse_autodiff.py` (24 tests, 61 subtests).

- Closed forms and forward mode: value, first and second derivative of 27
  unary rules (every unary `admath` function except `digamma` and
  `trigamma`, powers, `rpow`, reciprocal, negation, `abs`) agree with closed forms and with forward mode to
  `1e-14` (first) and `1e-13` (second) relative; `digamma' = trigamma` and
  `lgamma'' = trigamma` equal mpmath literals (`mpmath.polygamma(1, 2.5)`
  at 40 digits) to `1e-12`; eleven binary and variadic cases (`atan2`,
  `hypot`, `fsum`, variable powers, quotients, log bases, `max`/`min`)
  agree with forward-mode gradients and Hessians; the 13-term mpmath oracle
  of `tests/test_autodiff.py` is reproduced to `1e-14` (gradient) and
  `1e-13` (Hessian).
- Models: extended Rosenbrock (`d = 10`) gradient and tridiagonal Hessian,
  and the logistic-regression log-likelihood (40 rows, 5 coefficients)
  gradient `X'(y - p)`, Hessian `-X' W X` and a Hessian-vector product,
  all to `1e-13` or better (relative to the larger of 1 and the entry, or
  to the largest entry for Rosenbrock). `reverse_jacobian`
  equals the forward Jacobian; `vector_jacobian_product` equals `w' J`.
- Deep tapes: a 50 000-step chain `x <- 0.5 sin(x) + 0.3` (150 000 nodes)
  and a 200 000-addition sum are differentiated without recursion error
  and match the closed-form chain-rule products.
- Samplers: NUTS and static HMC on a 10-D Gaussian (two chains, 100
  warm-up and 100 draws, seed 7) with `gradient="reverse"` reproduce the
  forward-mode draws to `1e-9`; `"forward"` reproduces them bit for bit;
  reverse mode makes one log-density call per gradient, finite differences
  21 (`2 d + 1`). Laplace with reverse mode returns the exact covariance of
  a correlated Gaussian. A five-parameter traceable PPL program (NUTS, two
  chains of 120 + 120, seed 11) gives the same draws in reverse and forward
  mode to `1e-9` with more than five times fewer model executions than
  finite differences; library-distribution programs fall back with the
  reason "real number".

Cost (benchmark script outside the test suite, Windows 11, CPython 3.12.1,
best of five timings, two repeated runs; the ranges are the two runs). For each method: user-
function calls per gradient, and the wall time of one gradient divided by
the wall time of one float evaluation. "Gaussian" is
`-sum (x_i / s_i)^2 / 2`; "logistic" is the logit log-likelihood with
`n = 100` rows. Central differences are `optimization.central_gradient`.

- Gaussian, `d = 10`: finite differences 20 calls, ratio 50-55; forward
  1 call, 80-87; reverse 1 call, 56-57.
- Logistic, `d = 10`: finite differences 20 calls, 18-31; forward 1 call,
  31-33; reverse 1 call, 17-27.
- Gaussian, `d = 100`: finite differences 200 calls, 152-262; forward
  1 call, 246-422; reverse 1 call, 34-61.
- Logistic, `d = 100`: finite differences 200 calls, 219-245; forward
  1 call, 215-226; reverse 1 call, 39-42.
- `d = 2` (for the default decision): Gaussian forward 29-32, reverse
  21-32; logistic forward 9-12, reverse 7-18.
- NUTS, 10-D Gaussian, two chains of 300 + 300, seed 7: 9 321 gradients
  and 9 321 log-density calls for both automatic modes (197 190 calls with
  finite differences); wall time 0.93-1.29 s reverse, 1.26-1.70 s forward,
  1.14-1.34 s finite differences.

Reverse mode is about 5-6 times faster than forward mode at `d = 100` and
about 1.5 times at `d = 10`. In pure Python the per-operation bookkeeping
dominates, so the measured ratio to a float evaluation (34-61 at
`d = 100`) is well above the small constant of the operation-count bound;
the decisive saving relative to finite differences is the number of user-
function calls (1 instead of `2 d + 1`), which matters whenever the
log-density is expensive relative to its arithmetic.

Contract checks: `tests/test_documentation_contract.py`,
`tests/test_architecture.py`, ruff, strict mypy and the full test suite
pass; `tests/test_api_reference.py` reports only the ten new public names.
