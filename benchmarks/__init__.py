"""Reproducible benchmark suite for probstat (roadmap Phase 6).

The suite compares methods on problems with known answers and records
correctness (error against the reference), calibration (error in units of the
method's own reported standard error, coverage of its reported interval),
numerical stability (failures and warnings, caught and reported), and
computational cost (model evaluations and wall time).

Modules:

* :mod:`benchmarks.references` -- closed forms and quadratures computed
  without probstat, so a library bug cannot leak into a reference value;
* :mod:`benchmarks.problems` -- the problem registry and model builders;
* :mod:`benchmarks.methods` -- one adapter per library method;
* :mod:`benchmarks.harness` -- seeded execution, timing, failure capture,
  and aggregation;
* :mod:`benchmarks.report` -- the Markdown report;
* ``benchmarks/run.py`` -- the command-line entry point.
"""
