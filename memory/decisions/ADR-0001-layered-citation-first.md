# ADR-0001: Layered, Citation-First Architecture

- Status: Accepted
- Date: 2026-10-09

## Context

The library must grow from elementary probability to large modern systems
without circular dependencies or unverifiable formulas. It must also teach the
material without reducing mathematical scope.

## Decision

Use strict downward dependency layers rooted in a small `foundations` package.
Keep visualization optional and above mathematical domains. Give every
implemented equation a permanent registry ID and require source keys in
callable docstrings. Maintain long-horizon state in versioned memory files.

The initial runtime uses the Python standard library. Matplotlib is optional.
Large numerical frameworks require a future ADR that states ownership,
precision, array, device, and reproducibility consequences.

## Consequences

- Mathematical provenance is reviewable from code to equation to bibliography.
- Architecture tests can reject cycles and upward imports early.
- Some docstrings are longer, but the public API becomes self-explanatory.
- Domain convenience imports are restricted; callers import from explicit
  subpackages.
- Advanced vectorization is postponed until its abstraction boundary is clear.

