# ADR-0048: API stability labels and deprecation policy

- Status: Accepted (proposed during the 2026-10-12 overnight session; the
  owner may relabel packages at any time)
- Date: 2026-10-12

## Context

The roadmap listed "API stability labels and deprecation policy" as the last
open Phase 0 item. The library is pre-1.0 and grew by a dozen packages in one
release; users need to know which names they can rely on.

## Decision

1. Every top-level package has one label in
   `probstat.foundations.stability.PACKAGE_STABILITY`:
   - **stable**: change only after a deprecation period of at least one minor
     release (foundations, numerics, linalg, probability, statistics,
     distributions, sampling, optimization);
   - **provisional**: design settled, signatures may change, deprecated first
     where a shim is practical (the UQ core: modeling, propagation, inverse,
     reliability, sensitivity, surrogates, validation, risk, filtering,
     workflow, visualization, inference, and the remaining 0.14 packages);
   - **experimental**: added in 0.15, may change in any release (causal,
     design, information, conformal, timeseries).
2. Deprecations use the `deprecated(since=, removal=, alternative=)`
   decorator, which emits `DeprecationWarning` at the caller and appends a
   `Deprecated` note to the docstring; the changelog lists every deprecation
   and removal.
3. `tools/generate_api_reference.py` refuses to render a package without a
   label and prints the label in each package heading; a test keeps the
   label table equal to the set of installed packages.

## Consequences

- New packages must be labelled (normally *experimental*) before the API
  reference can be regenerated.
- Below 1.0 the labels state intent, not a semantic-versioning guarantee.
