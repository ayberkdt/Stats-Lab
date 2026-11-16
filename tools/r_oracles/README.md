# R oracles

`parity.R` produces the reference values of the probstat-versus-R parity
suite. It evaluates R on literal copies of classic R datasets and writes
`tests/data/r_parity.json`; `tests/test_r_parity.py` reads that file and
repeats every case in probstat, so the tests never call R.

## Regenerating

From the repository root:

```text
Rscript tools/r_oracles/parity.R
```

An optional argument names a different output file. The script needs base R
with its recommended packages (stats, MASS, nlme, survival) and the
contributed packages fitdistrplus, ismev, and e1071; it installs and
downloads nothing. It stops if a literal dataset differs from the copy
shipped with R, or if a call does not return a named list of numbers. The
output is deterministic: rerunning it with the same R and package versions
reproduces the file byte for byte.

## The JSON file

- `r_version`, `packages`: `R.version.string` and `packageVersion()` of
  each package used.
- `helpers`: the source of the two small R helpers that the calls use
  (`htest_values` renames the parts of an `htest` object;
  `abs_deviation_anova` is the base-R Levene and Brown-Forsythe statistic).
- `datasets`: every input, column by column.
- `cases`: one record per comparison with `id`, `area`, `target` (the dotted
  probstat name), `r_call` (the exact string that was evaluated), `inputs`,
  `r_value` (named arrays), `tolerance` (`relative`, `absolute`), and
  `tolerance_reason`; optional `note` (a definitional choice) and `probstat`
  (arguments the Python side needs).
- `skipped`: R functions that have no comparable probstat counterpart, with
  the reason.

A value passes when `|probstat - R| <= absolute + relative * |R|`.

## Adding a case

1. Add an `add_case` (or `add_law` for a distribution function) call to
   `parity.R`. The call string must return a named list of numeric vectors.
   Use relative 1e-10 or tighter when both sides evaluate the same closed form
   or special function; a looser tolerance needs a reason that names the
   algorithm that limits the agreement.
2. If R and probstat use different, equally valid definitions, change the R
   call to probstat's definition and explain it in `note`, or record the
   function with `add_skip`. Never widen a tolerance to absorb a definitional
   difference.
3. Regenerate the JSON file and register an adapter for the new id in
   `tests/test_r_parity.py`; a test fails while a case has no adapter.
4. Update the case count quoted in the "Agreement with R" section of
   `docs/foundations/hypothesis-testing-and-bootstrap.md`; a test compares it
   with the JSON file.
