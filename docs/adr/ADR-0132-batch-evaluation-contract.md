# ADR-0132 — Batch evaluation contract: procedures, requests, ask-tell sessions

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0048 (API stability), ADR-0131 (method laboratory)

## Context

Every adaptive algorithm of 0.27 evaluates its model as a Python function of
one point: subset simulation advances one chain state at a time, the
improved cross-entropy method loops over its population, the multifidelity
estimators call one model per input. That is the right interface for a
closed-form test function and the wrong one for the simulators the methods
were written for. A trajectory code that advances thousands of states per
call on a GPU, a job scheduler, or a simulator in another process all have
the same property: the cost is per *call*, not per point, and the caller,
not the algorithm, knows how to run a batch.

Two designs were available. Passing a vectorised callable keeps control in
the algorithm and does not help when the evaluation cannot happen inside a
Python call (a queued job that returns tomorrow). Inverting control
("ask-tell") solves that but, written by hand, forces every algorithm into
an explicit state machine, and the library would then hold two
implementations of each method that must agree.

## Decision

1. **One implementation per algorithm, written as a generator.** A
   *procedure* yields `EvaluationRequest(points, label, stage)` and receives
   the outputs in request order ([EQ-BATCH-001]); its `return` value is the
   result record. `label` names the model or fidelity (`"high"`, `"low"`),
   so a multifidelity procedure needs no knowledge of the models.
2. **Three drivers, one code path** in `probstat.foundations.batch`:
   `run_procedure(procedure, models)` with a batch callable or a mapping
   from label to batch callable (optional `max_batch_size` splits a request
   into chunks and concatenates the outputs); `AskTellSession(procedure)`
   with `ask()`, `tell(values)`, `done`, `result`, `evaluations`,
   `requests`; and `batch_from_pointwise(f)` to reuse a function of one
   point. A procedure cannot observe which driver runs it, so the three give
   identical results by construction; the tests check it.
3. **Duck-typed outputs.** `scalar_outputs` and `vector_outputs` convert any
   iterable with one entry (or one row) per point to tuples of floats and
   check the count. A NumPy array returned by a vectorised simulator is
   accepted without the library importing NumPy, so the standard-library
   runtime rule is unchanged. Non-finite outputs are passed through: what a
   failed run means is the algorithm's decision.
4. **Placement.** The module lives in `foundations` (layer 0) because
   packages on several layers are meant to use it, and a lower layer cannot
   import a higher one. Its users today are `probstat.uq` and
   `probstat.filtering`. `foundations` is labelled stable; the batch names
   are new after 0.27.0 (unreleased) and follow the deprecation policy of
   ADR-0048 from the release that ships them.
5. **Rule for new code.** A new algorithm that evaluates a model adaptively
   is written as `<name>_procedure` plus a thin `<name>` wrapper around
   `run_procedure`, issues the largest requests its mathematics permits
   (a whole population, all chains of one step, a whole level), and reports
   requests and evaluations per label. Estimators that only post-process
   simulator output take arrays, never a model.

## Consequences

- Algorithms written against the contract use a batch simulator at its
  native throughput, and the same procedure runs unchanged against a
  scheduler through `AskTellSession`.
- The transport adds a Python-level cost per request, not per point; it is
  negligible next to any simulator worth batching.
- The 0.27 pointwise algorithms are not rewritten; the pointwise functions
  remain the reference in their tests. Two procedures are written against
  the contract so far: `probstat.uq.adaptive_monte_carlo_budget_procedure`
  (ADR-0133) and `probstat.filtering.propagate_gaussian_mixture_procedure`
  (ADR-0140). There is no batch form of the rare-event estimators
  (`probstat.reliability`) or of the pick-freeze designs
  (`probstat.sensitivity`); both are not implemented.
- A generator can be driven only once. A procedure function must be called
  again for a second run; the wrappers do this.
- The contract says nothing about failures of individual evaluations,
  retries, or checkpointing. A caller who needs to persist a session must
  log the answered requests and replay them, which works because procedures
  are deterministic given the seed and the responses.

## Rejected alternatives

- **Vectorised callables only.** Simpler, but excludes out-of-process
  evaluation and gives no natural place for labels.
- **Hand-written ask-tell classes per algorithm.** Two implementations per
  method; the audit of 0.27 found most defects exactly where two code paths
  were supposed to agree.
- **`asyncio` coroutines.** Would impose an event loop on every caller for
  no gain: requests of one procedure are sequential by definition.
- **A NumPy-based array interface.** Faster post-processing, but it breaks
  the standard-library runtime contract that the rest of the library and
  its installation story depend on.

## Verification

`tests/test_batch_contract.py`: `run_procedure`, chunked evaluation with
`max_batch_size=1`, and a hand-driven `AskTellSession` return the same value
for a two-stage procedure whose second request depends on the first
response; request, output-count and label validation; a non-request yield
is rejected.

## References

[REF-PEP-342] for the generator protocol. The ask-and-tell pattern is
common in optimisation software; no specific source is cited because none
was verified for this record.
