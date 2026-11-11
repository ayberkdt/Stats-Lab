# ADR-0073: Module layout of the study workflow and the distribution families

- Status: Accepted
- Date: 2026-11-11

## Context

`src/probstat/workflow/study.py` had grown to 3,853 lines, about 1,500 more
than the next-largest module (`modeling/copulas.py`, 2,306 lines). It held
eight result records, the stage bookkeeping (`_StageContext`, the `_stage`
decorator, `_Accumulators`), six private helpers, and the `UQStudy` class
with 54 members: 20 for construction, properties, and stage bookkeeping,
and 34 stage, reporting, and helper methods covering propagation,
sensitivity, reliability, risk, calibration, emulation, verification,
validation, experimental design, and reporting.
Unrelated stages shared one file, so reviews and concurrent changes to
different stages touched the same module.

`probstat.distributions` keeps one family per module (`normal.py`,
`gumbel.py`, ...), with two exceptions: `simple_families.py` (Cauchy,
Laplace, Rayleigh, Pareto) and `more_families.py` (PERT, DiscreteUniform,
Hypergeometric). Their names describe neither the families they contain nor
a shared mathematical idea.

## Decision

### 1. `probstat.workflow.study` becomes a subpackage

| Module | Contents |
|---|---|
| `_records.py` | `STAGE_STATUSES`, `StageRecord`, `TargetedPropagation`, `DesignPointAnalysis`, `TailAnalysis`, `InformationSensitivity`, `DesignRecommendation`, `DiscrepancyAnalysis`, `DiscrepancyStudyPrediction` |
| `_context.py` | `_StageContext`, `_Accumulators`, `_orient_return_level`, `_format_settings`, `_converged_text`, `_format_condition`, `_sized`, `_setting_value` |
| `_base.py` | `_StudyBase`, the `_stage` decorator and its type `_StageDecorator` |
| `_propagation.py` | `_PropagationStages`: `propagate`, `propagate_until` |
| `_sensitivity.py` | `_SensitivityStages`: `sensitivity`, `shapley`, `screening`, `information_sensitivity` |
| `_reliability.py` | `_ReliabilityStages`: `limit_state`, `reliability`, `design_points`, `active_reliability`, `form`, `risk`, `extremes` |
| `_calibration.py` | `_CalibrationStages`: `calibrate`, `calibrate_with_discrepancy`, `predict`, `emulate`, `adopt_posterior` |
| `_validation.py` | `_ValidationStages`: `verify`, `numerical_uncertainty`, `validate` |
| `_design.py` | `_DesignStages`: `design` |
| `_reporting.py` | `_ReportingStages`: `provenance`, `to_dict`, `to_json`, `save_json`, `report` |
| `_core.py` | `UQStudy`, composed from the seven mixins: constructor, `__getstate__`, `from_data`, read-only properties, `latest`, `with_inputs` |
| `__init__.py` | the module docstring and the public re-exports |

Typing. `_StudyBase` declares every instance attribute that `UQStudy.__init__`
assigns, including `_children: list[tuple[int, Self]]`, and implements the
members that several stage domains use: `model`, `total_evaluations`,
`total_failed_evaluations`, `remaining_budget`, `stream`,
`_require_budget`, `_running`, `_warn`, `_record`, `_set_inputs`,
`_fresh_cost`, and `_output_sample`. Every mixin derives from `_StudyBase`,
so each mixin type-checks on its own under `mypy --strict`. `_stage` returns
a `_StageDecorator` protocol whose generic `__call__` keeps the exact
signature of the decorated method, including the mixin type of `self`; no
`Any` or `cast` is introduced.

Acyclic imports. `emulate` builds its child study through the hook
`_surrogate_study`, which `_StudyBase` declares and `UQStudy` overrides with
`UQStudy(model, seed=..., name=..., evaluation_kind="surrogate")`, the same
call as before. No mixin imports `_core`; the module graph is
`_records <- _context <- _base <- mixins <- _core <- __init__`.

Public surface. `__init__.py` re-exports exactly the ten names that other
code imported from the former module (`STAGE_STATUSES`, the eight records,
and `UQStudy`), with explicit `name as name` aliases. It defines no
`__all__`: the former module had none, and
`tools/generate_api_reference.py` lists every public subpackage that defines
one, which would duplicate the `probstat.workflow` table. The regenerated
`docs/api-reference.md` is unchanged.

Moved code. Every definition and docstring moved verbatim except for four
typing-driven edits: `_stage` and its inner functions are typed with the
mixin type variable instead of `UQStudy`; `UQStudy.__init__` assigns the
attributes declared on `_StudyBase` without re-annotating them; `emulate`
returns `Self` and constructs its child through `_surrogate_study`; and
`_report_lines` annotates its child list with `Self`.

### 2. One module per distribution family

`simple_families.py` and `more_families.py` are replaced by `cauchy.py`,
`laplace.py`, `rayleigh.py`, `pareto.py`, `pert.py`, `discrete_uniform.py`,
and `hypergeometric.py`. Class bodies and docstrings moved verbatim. The
module-level helpers each have a single user and move with it:
`EULER_GAMMA` to `rayleigh.py`, `_EXACT_POPULATION` and `_log_comb` to
`hypergeometric.py`. Moving them to the private `_special.py` would add an
import without sharing anything. `probstat.distributions.__all__` is
unchanged.

The two old modules are removed without compatibility shims. Neither module
path was part of the documented API: both were absent from every `__all__`
and from the API reference, and no code in the repository imported them
other than `probstat.distributions.__init__`. `distributions` carries the
*stable* label (ADR-0048), which governs public names; none changes. Code
that imported the module paths directly breaks, and the release notes of the
next version should list the removal.

### 3. Tests that encode file locations

Two tests referred to the former file rather than to behaviour and now
point at the new layout; no assertion changed.

- `tests/test_uq0_prediction_workflow.py` collects the recorded stage kinds
  from the source of every module in `workflow/study/` instead of
  `workflow/study.py`.
- `tests/test_uq_study_accounting.py` patches `form_hlrf` where the
  reliability stages look it up,
  `probstat.workflow.study._reliability.form_hlrf`.

`docs/curriculum/athens-2026-coverage.md` lists `src/probstat/workflow/study`
as evidence for ATHENS-1.4, and `docs/architecture.md` gains a
module-layout section. The equation registry needed no change: its `- Code:`
entries name public dotted paths (`workflow.UQStudy.propagate_until`,
`distributions.Cauchy`, ...), which still resolve.

## Consequences

- Results, stage records, JSON records, reports, and public names are
  unchanged.
- Introspection reflects the defining modules: `UQStudy.__module__` is
  `probstat.workflow.study._core`, the records report
  `probstat.workflow.study._records`, stage methods carry qualified names
  such as `_PropagationStages.propagate` (visible in bound-method `repr` and
  in Python's argument-error messages for undecorated methods), and the
  annotation of `emulate` reads `Self` instead of `UQStudy`. Each family
  class reports its own module, for example `probstat.distributions.cauchy`.
- Pickles written from now on reference the defining modules. Study and
  record pickles written by earlier versions still load, because every
  pickled name is re-exported at `probstat.workflow.study`. Pickles of the
  seven moved families written by earlier versions reference the removed
  modules and no longer load.
- A new stage goes into the mixin of its domain; state shared between
  domains is declared on `_StudyBase`. Tests patch a dependency in the stage
  module that looks it up.
- `probstat.workflow.study` no longer exceeds 2,500 lines anywhere; its
  largest module is `_calibration.py` (612 lines).

### Follow-ups found during the review (not changed here)

No other module exceeds 2,500 lines, and no other module is named
`*_families.py`, `misc.py`, `utils.py`, or similar. The following are
candidates for later, separately reviewed changes:

1. `distributions/wrappers.py` (914 lines) groups three unrelated
   constructions, `Affine`, `Mixture`, and `Truncated`, under a generic name;
   `affine.py`, `mixture.py`, and `truncated.py` would match the
   one-concept-per-module convention.
2. `optimization/solvers.py` (1,944 lines, 27 top-level definitions) holds
   BFGS, Nelder-Mead, Levenberg-Marquardt, Brent, and finite differences;
   splitting by solver family would mirror the package description.
3. The Euler-Mascheroni constant is defined four times
   (`distributions/gumbel.py`, `distributions/rayleigh.py`,
   `inference/fitting.py`, `information/differential.py`). All four literals
   round to the same double, so one shared definition in
   `numerics.special` would be numerically neutral.
4. Three modules approach the 2,500-line threshold:
   `modeling/copulas.py` (2,306), `processes/ctmc.py` (2,273), and
   `inference/hamiltonian.py` (2,043).
5. Several module names describe a style rather than their content:
   `visualization/foundations.py` (1,145 lines, which also shares its name
   with the `foundations` package), `reliability/analysis.py`,
   `sensitivity/analysis.py`, `reliability/model_based.py`, and
   `sensitivity/model_based.py`.

None of these was changed here: items 1, 2, and 5 rename modules of stable or
provisional packages, item 3 touches packages outside this change, and
item 4 needs a design decision for each module.

## Verification

- Moved code: a script compared every top-level definition and every
  `UQStudy` member of the former modules (read with `git show HEAD:<path>`)
  against the new files. All 10 distribution blocks and 70 of 74 study
  blocks appear verbatim in exactly one new module; the four that differ
  are the edits listed under Decision 1.
- Output equivalence: a script running propagation (fixed-size and
  targeted), Sobol, Shapley, Morris, reliability, design points, FORM,
  extremes, information sensitivity, emulation with a child study,
  calibration, prediction, and posterior adoption printed each study's JSON
  record, Markdown report, and pickle round-trip, plus samples, means, and
  pickle round-trips of the seven moved families. The outputs of the former
  and the new source trees are identical except for the creation time of
  the child study.
- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`: 1581 passed,
  1 skipped before and after the change.
- `python -m ruff check .` and `python -m mypy src/probstat` (strict) pass.
- `python benchmarks/run.py --quick` and every script in `examples/` run
  without error.
- `PYTHONPATH=src python tools/generate_api_reference.py` reproduces the
  committed `docs/api-reference.md` unchanged.
