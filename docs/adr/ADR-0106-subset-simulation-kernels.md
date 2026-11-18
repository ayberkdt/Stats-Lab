# ADR-0106: Interchangeable Markov-chain kernels for subset simulation

- Status: Accepted
- Date: 2026-11-18

## Context

`subset_simulation` populated every conditional level with the
component-wise modified Metropolis algorithm of Au and Beck, written inline
in the level loop. That kernel is robust in low dimension, but in high
dimension its chains stay strongly correlated at deep levels: on
`G = beta - sum u_i / sqrt(d)` with `d = 100` and `beta = 5`, the observed
c.o.v. is about 0.6 against about 0.4 for the conditional-sampling kernels
of Papaioannou et al. (2015) and the rejection-sampling Hamiltonian kernel of
Wang, Broccardo and Song (2019). The guide and several tests quote seeded
numbers of the existing kernel, so a new kernel must not change them.

## Decision

1. **One keyword selects the kernel.** `subset_simulation(...,
   sampler="mma" | "cs" | "acs" | "hmc")`, listed in `SUBSET_SAMPLERS`, with
   `"mma"` the default. Kernel options are plain keywords with the values of
   the source papers: `correlation=0.8` (CS), `initial_scale=0.6`,
   `target_acceptance=0.44`, `adaptation_chains=None` (that is
   `ceil(100 p0)`), `scale_reference="unit" | "seeds"` (aCS), and
   `trajectory_time=pi/4` (HMC).
2. **Private chain and level functions.** Each kernel is a private chain
   function that appends one chain (seed first) to shared point and value
   lists and returns its acceptance count, plus a level function that runs
   the chains of one level and returns a `_LevelSample` (states, chain
   lengths, accepted and proposed counts, the reported scale, and the
   adapted scale carried to the next level). The level loop, thresholds,
   stopping rule, and c.o.v. computation are shared by all kernels.
3. **Bit-identical default.** The modified Metropolis chain consumes the
   generator in exactly the order of the former inline code, starts its
   chains from the seeds in ascending order of `G`, and evaluates the limit
   state only when a coordinate moved. Validation of the new keywords draws
   no random numbers. `tests/test_subset_samplers.py` pins seeded results
   recorded before the refactor; counts and products of `p0` are compared
   exactly and thresholds and c.o.v. values with a relative tolerance of
   `1e-12`, so last-digit libm differences between platforms cannot fail
   it.
4. **Adaptive kernels follow the sources.** aCS permutes the seeds, uses
   `floor(N / N_s)` states per chain with the remainder on the first chains,
   and adapts `lambda` every `N_a` chains with gain `1 / sqrt(i + 1)`,
   carrying `lambda` across levels [EQ-REL-SUBSET-005]. RS-HMC uses the exact
   Gaussian flow with a fresh momentum per step [EQ-REL-HMCSS-001] and
   adapts `t_f` toward the acceptance band `[0.3, 0.5]` [EQ-REL-HMCSS-002];
   carrying `t_f` across levels is this library's choice, mirroring aCS.
   Unequal chain lengths use the pair-count generalization of the level
   c.o.v. [EQ-REL-SUBSET-006]; equal lengths keep the original arithmetic.
5. **No bouncing variant.** Basic-bouncing HMC needs the limit-state
   gradient and a hitting-time search on every trajectory, and its cost and
   robustness depend on the response; it is not provided. The `"hmc"` kernel
   is RS-HMC, which needs no gradient.
6. **Opt-in comparison rows.** `compare_reliability_methods` accepts
   `"acs"` (row `Subset-aCS`) and `"hmc-ss"` (row `Subset-HMC`) in
   `OPTIONAL_METHODS`; `METHODS` and the default rows are unchanged.

## Consequences

- Existing seeded results, the guide's worked example, and default
  comparisons are unchanged.
- `SubsetSimulationResult` gains `sampler` and `level_scales` with defaults,
  so code that constructs or unpacks it by keyword keeps working.
- RS-HMC with a full momentum refresh equals conditional sampling with
  `rho = cos t_f`; the two kernels differ only in how the scale is adapted.
- Plain conditional sampling with a fixed `rho` is not uniformly better than
  modified Metropolis at deep levels; the guide recommends the adaptive
  kernels for high-dimensional problems.

## References

- [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
  [REF-WANG-BROCCARDO-SONG-2019], [REF-COTTER-ROBERTS-STUART-WHITE-2013].
