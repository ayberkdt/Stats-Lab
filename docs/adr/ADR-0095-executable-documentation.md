# ADR-0095: Executable documentation — every example runs and every quoted number is checked

- Status: Accepted
- Date: 2026-11-12

## Context

Features are often misunderstood because the documentation has drifted
from the code: a renamed argument, a number from an older run, a snippet
that relies on a variable defined in another file. Before this decision
only the six prerequisite guides and the two measure-theory guides were
executed (`tests/test_prerequisite_guides.py`, `tests/test_measure_theory.py`):
54 of the 91 Python blocks. The other 37, in `README.md` and the remaining
foundation guides, were never run, and inline values were never compared.
Running them all once showed:

- 8 blocks in 6 guides failed outright, all with `NameError`: snippets in
  `generative-models.md`, `model-discrepancy.md`,
  `nonlinear-calibration-and-mcmc.md`, `stochastic-processes-and-random-fields.md`,
  `visual-language.md`, and `probabilistic-programming.md` used data,
  models, or imports that existed only in a lab or a test;
- the figure description quoted in `visual-language.md` belonged to a figure
  the snippet did not draw;
- the opening example of `probabilistic-programming.md` (four chains of
  1000 + 1000 NUTS draws) took 227 s;
- inline results such as `forward["margin"].quantile(0.05)  # 0.124` in the
  README were never compared with the code.

The package docstrings were one to a few lines for most of the 33
packages, and no guide had a section on common mistakes (failure side,
one- and two-sided tests, nugget and jitter, seeds, units, parameter
conventions).

## Decision

1. **One harness for all documentation.** `tests/_documentation.py` parses
   the fenced blocks of `README.md`, every guide in `docs/foundations` and
   `docs/prerequisites`, and `docs/learning-path.md`. It executes every
   `python` block in document order, one namespace per file, inside a
   temporary working directory with the `Agg` Matplotlib backend
   ([REF-PYTHON-AST], [REF-PYTHON-UNITTEST]). `tests/test_documentation_examples.py`
   generates one test per file.
2. **Quoted values are checked to the shown precision.** A top-level
   expression (or one-name assignment, or a `print` of one argument) whose
   trailing comment starts with a Python literal is compared with that
   literal: `# 0.124` accepts `[0.1235, 0.1245]`; strings, names, and
   tuples of strings must agree exactly; prose after the literal is
   ignored. An expression followed by comment-only lines is compared with
   them as its value or `repr`, and a `print` followed by comment lines that
   end the block is compared with its output. A `text` block directly after
   a Python block is the block's standard output, compared number by number
   at the shown precision and character by character elsewhere.
3. **Exemptions are explicit and rare.** A block preceded by
   `<!-- not-executed: reason -->` is skipped and counted; the test allows at
   most three, each with a reason.
4. **Snippets are fixed in the text, not in the harness.** Missing data and
   models were added to the snippets themselves, using the seeds and sizes
   the surrounding text cites. A block too slow to run in the test suite was
   reduced in the text to the configuration the guide already reports.
   Stochastic results of seeded MCMC and optimizer trajectories are not
   quoted inline, because their digits differ across platforms.
5. **Prose checks.** The same test module requires that every dotted
   library name in inline code (`reliability.form_hlrf`,
   `probstat.modeling.JointInput`, `UQStudy.reliability`) and every
   `from probstat... import` resolves; that every relative link, image, and
   heading anchor exists; and that the README's counts (packages, public
   names, equation records, labs, ATHENS topics, distribution families),
   version badge, optional extras, stage table, and package table match the
   code and the generated references.
6. **Clarity contract.** Every package docstring keeps its first line (the
   API reference is generated from it) and adds `Main entry points:`,
   `Assumptions:`, `Common pitfalls:`, a `Guide:` line naming its foundation
   or prerequisite guide, and `References:` with registered keys. Every
   foundation and prerequisite guide has a `## Common pitfalls` section.
   Each pitfall states a fact about the code that was read in the source or
   demonstrated by an executed snippet in that section, so most of them are
   now themselves checked by the harness.
7. **No duplicate execution.** `tests/test_prerequisite_guides.py` keeps only
   the prerequisite contract (each snippet paired with its output block);
   `tests/test_measure_theory.py` reuses the cached harness run for its
   additional checks of the interpretation guide.

## Consequences

- A stale number, a renamed API, or a snippet that depends on hidden state
  fails a named test that points at the file and line.
- Changing a default, a seed policy, or a parameter convention now fails
  the guide that documents it, so the documentation has to change with the
  code.
- The README counts must be updated together with the generated API
  reference and the equation registry; `ReadmeTests` fails until they are.
- The documentation tests add about 70 s to the test suite on the
  development machine; the two slowest files are the probabilistic
  programming guide (eight-schools NUTS, about 22 s) and the generative
  models guide (a flow and a score-matching fit, about 20 s).
- Writers must keep inline quotes to deterministic quantities: exact or
  closed-form values, counts, defaults, seeded designs, and converged
  estimates; MCMC means are reported in prose with their Monte Carlo error.

## Verification

- Final run of the harness over 56 Markdown files (55 contain Python):
  **145 blocks executed, 1 exempted, 309 inline values and 63 output blocks
  or printed comments compared, 0 failures.** Before this decision: 91
  Python blocks, of which the 39 prerequisite and 15 measure-theory blocks
  were executed and no inline value was compared.
- The single exemption is the tuned pseudo-prior outline of
  `probabilistic-programming.md` (it contains `...` for the sampler
  settings; the full run is in `tests/test_ppl_dynamic_structure.py`).
- README quick start: a fresh virtual environment (Python 3.12.1,
  `pip install -e ".[visualization]"`, Matplotlib 3.11.2, NumPy 2.5.4)
  ran both README blocks with 5 quoted values matching: 0.124, `('load',
  'strength')`, 0.0362 (FORM), 0.0395 (crude Monte Carlo, newly checked),
  and 49.8 (CVaR at 0.99).
- Corrections to the documentation found by the harness or by the clarity
  pass (the code was right in every case):
  - executable fixes: the six `NameError` guides listed above; the
    describe-figure output of `visual-language.md`; the 227 s
    eight-schools run reduced to the guide's own test configuration
    (seed 1, two chains of 150 + 150); the information matrix of
    `experimental-design.md` quoted in decimals so it can be compared;
  - new checked values in existing sections: least-squares estimates and
    standard errors of `nonlinear-calibration-and-mcmc.md` (5.0031, 0.3008;
    0.0680, 0.0064) and the random-field example of
    `stochastic-processes-and-random-fields.md` (5 modes, mean 2.08386,
    variance 0.22814);
  - stale statements: `StudentT.mean` and `.variance` return `nan` for
    `nu <= 1` (the guide said they raise); `RandomInput` accepts such a law;
    the binomial CDF already uses the incomplete beta for `n > 256`;
    geometric, negative binomial, uniform, exponential, gamma, and beta
    laws, direct survival functions, `FiniteRandomVariable`, multiple design
    points, CCF parameter uncertainty, discrepancy processes, f-divergences
    beyond KL and JS, adaptive conformal intervals for time series, RQMC
    designs, and nested p-box propagation all exist although guides called
    them future work; distribution `sample` accepts a `RandomStream`;
    `inconsistent_items` flags only mode-ratio sums above one; the cost-risk
    analysis records a missing correlation in `notes` rather than warning;
    a poorly conditioned linear calibration design sets a `warning` string
    rather than raising a Python warning; the validation-domain distance of
    the point (0.3, 0.3) is 0.424, not 0.283; the Gaussian-process nugget is
    enlarged automatically (`adaptive_nugget=True`, up to `1e-4` with a
    `GaussianProcessJitterWarning`); polynomial-chaos coefficients are solved
    by Householder QR, also corrected in the record of `EQ-SUR-PCE-005`.
- 33 package docstrings and 54 guides received the clarity contract;
  `test_package_docstrings_state_purpose_entry_points_and_pitfalls` and
  `test_every_guide_has_a_common_pitfalls_section` enforce it.
- No library code changed: every disagreement between text and behaviour was
  a documentation error.
- Commands: `PYTHONPATH=src python -m pytest -q tests/test_documentation_examples.py
  tests/test_prerequisite_guides.py tests/test_measure_theory.py
  tests/test_learning_paths.py tests/test_documentation_contract.py
  tests/test_architecture.py tests/test_api_reference.py`, `ruff check .`, and
  `mypy src/probstat` pass. `tests/test_math_reference.py` reports
  `docs/math-reference/surrogates.md` as stale until the generated
  mathematical reference is rebuilt from the corrected `EQ-SUR-PCE-005`.
