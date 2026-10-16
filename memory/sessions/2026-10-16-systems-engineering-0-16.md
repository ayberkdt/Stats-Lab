# 2026-10-16 Systems engineering and core fixes (0.16.0)

## Goal

Fix the gaps found by the core-mechanics audit (same date), and turn the
library into a framework a systems engineer can apply to any mission:
analytic (non-Monte-Carlo) success/failure probabilities by Markov chains,
exact system reliability from subsystem reliabilities, and risk and cost
analysis that work together.

## How the work was organised

Seven parallel workstreams, each in its own worktree with its own ADR,
merged and integrated afterwards:

| Workstream | ADR | Scope |
|---|---|---|
| W1 | 0058 | model contract, evaluation engine, UQStudy accounting and chaining |
| W2 | 0059 | optimizers, incomplete beta, distribution contract and families, sampling, PCE/GP numerics |
| W3 | 0060 | exact system reliability (BDD), importance, CCF, phased missions |
| W4 | 0061 | CTMC, Markov reliability and availability, Markov phased missions, stage models |
| W5 | 0062, 0063 | cost analysis; risk register, FMECA, 882E, F-N, reliability data, allocation |
| W7 | 0064 | copulas, discrete and interval inputs, nested propagation |
| W8 | 0065 | forward-mode autodiff, MCMC diagnostics, sensitivity fixes |
| Integration | 0066 | adapters, MissionAssessment, lab, guide, release |

## Verification

- Every workstream compared its numbers with an independent oracle: closed
  forms, brute-force enumeration in exact rationals, and scipy/mpmath
  values computed in a scratch environment and pasted into tests as
  literals.
- The integration added its own independent check: random coherent
  structures against brute force (error 2e-16), and random CTMCs against
  `scipy.linalg.expm` (error 5e-15).
- The README quick-start numbers are unchanged (0.124, 0.0362, 0.0395,
  49.8).
- Final gates:
  - ruff and mypy (strict) are clean;
  - 1,582 unittest tests pass, with 0 failures both with and without
    Matplotlib;
  - all 32 example labs run;
  - the quick benchmarks show 0 unexpected failures.

## Limits of this session

- The sandbox could not reach primary sources: NASA, NRC and authors' sites
  were blocked. Formulas come from the cited standard literature and were
  checked numerically rather than against the books' text.
- The MIL-STD-882E Table III cells and the Joe–Kuo licence notice were
  written from knowledge. Confirm both against the source documents before
  relying on them.
- GitHub push was refused (403). The work was handed over as a git bundle.
