# ADR-0096: Release-quality packaging and the remaining module-structure follow-ups

- Status: Accepted
- Date: 2026-11-12

## Context

### Packaging

The distribution `probstat-library` had never been built and installed
outside the source tree in CI. Building version 0.19.0 with `python -m build`
in a fresh environment showed:

- the core metadata named the authors "Probability and Statistics Library
  contributors", declared no project URLs, classified the library as
  `Development Status :: 2 - Pre-Alpha`, and listed Python 3.11 and 3.12 but
  not 3.13, although CI tests all three [REF-PYPA-CORE-METADATA];
- `Operating System` and `Typing :: Typed` classifiers were missing, although
  the package is pure Python and ships `py.typed`;
- the source distribution contained the 137 test modules (setuptools adds
  `tests/test*.py` by default) but not `docs/`, `examples/`, or
  `benchmarks/`, which the tests read, so the packaged tests could not run;
  it did not contain `CHANGELOG.md` or `CITATION.cff`;
- `probstat.__version__`, `pyproject.toml`, and `CITATION.cff` each carry the
  version, and nothing checked that they agree.

The wheel itself was sound: the package, `py.typed`, `LICENSE`, and
`LICENSES/JOE-KUO-SOBOL.txt`, and nothing else.

### Module structure

ADR-0073 listed follow-ups that rename modules. Two are taken up here
(items 2 and 5 of that list); `distributions/wrappers.py` (item 1) and the
other visualization modules are handled separately.

- `optimization/solvers.py` (1,947 lines, 38 top-level definitions) held
  finite differences, BFGS with its line search, Nelder-Mead, multi-start,
  Levenberg-Marquardt, Brent, the box transformation, and four result
  records.
- `visualization/foundations.py` (1,153 lines) shared its name with the
  `probstat.foundations` package and did not say what it draws.

Neither module path is public API. Neither module defines `__all__`; both
packages re-export every public name from their `__init__`;
`docs/api-reference.md` lists only `probstat.optimization` and
`probstat.visualization`. Outside the two `__init__` modules the repository
referred to the paths in one test (`from probstat.optimization.solvers
import _strong_wolfe`), two equation-registry `- Code:` lines
(`visualization.foundations.plot_empirical_cdf` and
`...plot_categorical_pmf`), six evidence paths of the ATHENS coverage
matrix, ADR-0073 itself, and the generated mathematical reference. None of
these documents the path as an import location for users.

## Decision

### 1. Distribution metadata

`pyproject.toml` now declares:

- `authors` and `maintainers`: "Ayberk", the author name of every commit and
  of the git configuration of the repository; `CITATION.cff` names no
  person. No e-mail address is published, so none is given.
- `readme = { file = "README.md", content-type = "text/markdown" }`.
- `[project.urls]`: Homepage and Repository
  `https://github.com/ayberkdt/Stats-Lab`, Issues `.../issues`, Changelog
  `.../blob/main/CHANGELOG.md`, and Documentation `.../tree/main/docs` (the
  guides, API reference, and mathematical reference live in the repository).
- Classifiers: `Development Status :: 3 - Alpha`; Intended Audience
  Education and Science/Research; `Operating System :: OS Independent`;
  `Programming Language :: Python :: 3`, `3 :: Only`, `3.11`, `3.12`,
  `3.13`; `Topic :: Education`, `Topic :: Scientific/Engineering`,
  `Topic :: Scientific/Engineering :: Mathematics`; `Typing :: Typed`. All
  thirteen are in the `trove-classifiers` list (version 2026.9.21.13) and
  none is deprecated. No `License ::` classifier is added: the licence is the SPDX
  expression `BSD-3-Clause` (core metadata 2.4 `License-Expression`, under
  which licence classifiers are deprecated); the Joe-Kuo direction-number
  licence is also BSD 3-clause, so the expression covers both licence files.
- Keywords: probability, statistics, uncertainty-quantification,
  reliability, monte-carlo, bayesian-inference, sensitivity-analysis,
  risk-analysis, education, scientific-computing.

**Development status.** PyPI defines the stages only by name. "Pre-Alpha"
understated the state: 2,151 tests, strict typing, and CI on three Python
versions back a library whose results are checked against independent
oracles. "Beta" would overstate API stability: of the 33 top-level packages
in `probstat.foundations.stability.PACKAGE_STABILITY`, 8 are *stable*,
17 *provisional*, and 8 *experimental*, the version is 0.x, and module
paths still move (ADR-0073 and this record). "Alpha" (usable, API still
changing) matches. The classifier should move to Beta when the provisional
packages are promoted.

The version stays static in `pyproject.toml`, as the release procedure
expects; `tests/test_packaging_metadata.py` now fails when `pyproject.toml`,
`probstat.__version__`, and `CITATION.cff` disagree, when the Python
classifiers and `requires-python` differ from the CI matrix, when a project
URL leaves the repository, or when a licence file is missing.

### 2. Source distribution contents

A new `MANIFEST.in` adds `CHANGELOG.md` and `CITATION.cff` and prunes
`tests/`. The suite depends on `docs/`, `examples/`, and `benchmarks/`;
shipping those would turn the sdist into a copy of the repository, and
shipping the tests without them ships tests that fail. The complete tree is
the tagged repository. The sdist now holds `src/probstat` (identical to the
source tree), `pyproject.toml`, `README.md`, `LICENSE`, `LICENSES/`,
`CHANGELOG.md`, `CITATION.cff`, `MANIFEST.in`, and the files setuptools
generates (`PKG-INFO`, `setup.cfg`, `src/probstat_library.egg-info/`).

### 3. Wheel smoke test and CI job

`tools/smoke_test_wheel.py` runs in an environment into which the wheel was
installed, in isolated mode (`python -I`) so that neither the working
directory nor `PYTHONPATH` can supply the source tree. It fails when
`probstat` is imported from `src/`; when `probstat.__version__` differs from
`importlib.metadata.version("probstat-library")`
[REF-PYTHON-IMPORTLIB-METADATA]; when `py.typed` or a licence file is not
installed; when any module of the package fails to import or a name in a
package's `__all__` does not resolve (modules that need a missing optional
dependency are reported and skipped); and when a propagation and a FORM
analysis of `g = R - S` with `R ~ N(5, 0.5)`, `S ~ N(3, 0.4)` disagree with
the closed form `beta = 2 / sqrt(0.41)` [EQ-REL-FORM-001] (the Monte Carlo
mean is judged against four standard errors).

`.github/workflows/ci.yml` gains a `package` job on Python 3.13: build the
sdist and wheel in a dedicated environment (`build` and `twine` only), run
`twine check --strict`, install the wheel into a second clean environment
with no extras, and run the smoke test from outside the checkout. The `test`
job is unchanged.

### 4. `probstat.optimization`: one module per solver family

| Module | Contents |
|---|---|
| `finite_differences.py` | `central_gradient`, `forward_gradient`, `central_jacobian`, `central_hessian`, the step rules and one-sided fallbacks |
| `quasi_newton.py` | `minimize_bfgs`, `_bfgs_core`, `_strong_wolfe`, `_interpolate`, `_gradient_verdict`, line-search constants |
| `nelder_mead.py` | `minimize_nelder_mead` |
| `multistart.py` | `multi_start` |
| `nonlinear_least_squares.py` | `least_squares` (Levenberg-Marquardt), `gauss_newton_covariance` |
| `root_finding.py` | `brent_root` |
| `bounds.py` | `Bounds`, `BoxTransform` |
| `results.py` | `OptimizationResult`, `LeastSquaresResult`, `RootResult`, `MultiStartResult` |
| `_common.py` | `ScalarFunction`, `VectorFunction`, `_EPS`, `_finite_value` |

`stochastic.py` (Adam) is unchanged. A module is not named after the
function it holds (`least_squares`, `multi_start`): a submodule of that name
would shadow the function as an attribute of the package for
`import probstat.optimization.least_squares` and for the registry resolver,
which imports the longest importable prefix of a `- Code:` target.
Every definition and its docstring moved verbatim, with its preceding
comment; only the import blocks and the module docstrings are new.
`probstat.optimization.__all__` is unchanged, and the package docstring
now carries the former module docstring and the module map.

### 5. `visualization/foundations.py` becomes `visualization/foundation_plots.py`

The file was renamed with its history (`git mv`); the code is unchanged and
the module docstring gains one paragraph listing what it draws.
`probstat.visualization.__init__`, the two registry `- Code:` lines (edited
in place), the ATHENS evidence paths (ATHENS-10.5 now points at the
`src/probstat/optimization` package), `docs/architecture.md` (module
layout), and `docs/foundations/visual-language.md` (new section "Foundation
figures") follow. No laboratory imported the module path; every laboratory
imports from `probstat.visualization`.

### 6. Removal without aliases

`probstat.optimization.solvers` and `probstat.visualization.foundations` are
removed without deprecated alias modules, as ADR-0073 did for
`simple_families` and `more_families`: neither path appears in an `__all__`,
in the API reference, or in any document as a place to import from. The
`stable` label of `optimization` governs public names, and none changes.
The release notes of the next version should list the two removed paths.

## Consequences

- Public names, results, and figures are unchanged.
- Introspection reports the defining modules, for example
  `minimize_bfgs.__module__ == "probstat.optimization.quasi_newton"` and
  `plot_normal_pdf.__module__ == "probstat.visualization.foundation_plots"`.
  The generated mathematical reference groups records by that module, so
  its `index.md`, `optimization.md`, `statistics.md`, and `distributions.md`
  pages need regenerating; `docs/api-reference.md` is unchanged.
- Pickles of `OptimizationResult`, `LeastSquaresResult`, `RootResult`,
  `MultiStartResult`, and `BoxTransform` written by earlier versions name
  `probstat.optimization.solvers` and no longer load (checked: a 0.19.0
  pickle raises `ModuleNotFoundError`). The canonical descriptions of
  `workflow.chaining` include the module of a dataclass, so a hash over a
  value that contains one of these records would change; input-law hashes
  cover distributions only and are unaffected.
- Code that imported either old path, or the private `_strong_wolfe` from
  `solvers`, must import from `probstat.optimization` (or the new private
  module).
- The largest optimization module is now `quasi_newton.py` (580 lines).
- The sdist no longer contains the tests. The README links to repository
  paths (`docs/...`, `examples/...`) and images under `docs/images/` with
  relative URLs; on a package index those links and images do not resolve.
  Rewriting them as absolute repository URLs is a change to `README.md` and
  is left to the documentation owner.
- Every pull request now builds and installs the distributions, so a
  missing file, a broken import outside the source tree, or a version
  mismatch fails CI.

## Verification

- Distributions: `python -m build` in a fresh virtual environment
  (`build` 1.6.1, `twine` 7.0.0) built `probstat_library-0.19.0.tar.gz` and
  `probstat_library-0.19.0-py3-none-any.whl`. A script compared their
  contents with `git ls-files src/probstat`: the wheel holds exactly the 337
  package files (336 modules and `py.typed`) plus `METADATA`, `WHEEL`,
  `RECORD`, `top_level.txt`, and both licence files; the sdist's
  `src/probstat` equals the source tree, with no `__pycache__` or `.pyc`
  file and none of the removed modules. `twine check --strict` passed for
  both files.
- Installed wheel: in a second fresh environment containing only `pip` and
  the wheel (no Matplotlib, no NumPy), `python -I tools/smoke_test_wheel.py`,
  run from outside the checkout, imported `probstat` 0.19.0 from
  `site-packages`, imported all 335 modules (none skipped), and reported
  `mean 1.9780 (exact 2), FORM beta 3.123475 (exact 3.123475)`; the
  metadata version matched. Run against the source tree (`PYTHONPATH=src`),
  the same script fails its location check, as intended.
- Moved code: a script compared every top-level definition of the former
  `solvers.py` (from the 0.19.0 tree) with the new modules: all 38 appear
  verbatim in exactly one module. The rename of `foundations.py` is a
  98 % similarity rename whose only change is the docstring paragraph.
- Output equivalence: a script printing the `repr` of central and forward
  gradients, Jacobian, Hessian, BFGS (free and bounded), Nelder-Mead,
  multi-start, Levenberg-Marquardt, the Gauss-Newton covariance, Brent, and
  a `BoxTransform` produced byte-identical output on the former and the new
  tree.
- Figures: the seven laboratories that call a function of
  `foundation_plots` (`generate_foundation_figures`, `basic_monte_carlo_lab`,
  `advanced_monte_carlo_lab`, `statistical_fitting_lab`,
  `model_selection_lab`, `real_data_fitting_lab`,
  `interactive_explainers_lab`) were run in the porcelain and the graphite
  palette with every saved figure drawn by one of its 15 functions passed
  to `datawiz.figur_denetle(fig, ortam="web")`: 38 figures covering all 15
  functions, 0 HATA, 12 UYARI, 128 BİLGİ. The warnings are 9-10 pt text
  below 4.5:1 contrast (direct labels in a series hue on the canvas, and
  the cell values of the mass matrix in the dark palette), a mass matrix
  without a colour bar (`renk-cubugu-yok`; every cell is labelled with its
  value), and a 4.9 pt offset between the two y-axis labels of
  `plot_continuous_functions`.
  The same audit on the 0.19.0 tree gives the same findings for every
  figure, so the move changes none of them. The PNGs were inspected: no
  overlapping text.
- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`: 2151 passed,
  1 skipped on the 0.19.0 tree; 2156 passed (the five new packaging tests
  included), 1 skipped on the new tree, where the only failures are the
  four subtests of `tests/test_math_reference.py` for the stale generated
  pages named above. Regenerating the mathematical reference changes
  exactly those four pages, and only in module grouping and the renamed
  `- Code:` line. `tests/test_api_reference.py` passes unchanged.
- `python -m ruff check .` and `python -m mypy src/probstat` (strict) pass;
  all 34 scripts in `examples/` run with `MPLBACKEND=Agg`.
