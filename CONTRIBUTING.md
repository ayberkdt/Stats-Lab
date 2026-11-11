# Contribution Contract

Every change must preserve four contracts.

## 1. Dependency contract

Dependencies point downward only:

```text
visualization
    |
distributions    statistics    probability
          \          |          /
                 foundations
```

Sibling domain packages must not import one another merely for convenience.
Shared primitives belong in `foundations`. Higher-level systems will be added
as new layers rather than pushed into lower layers.

## 2. Documentation contract

Every module, class, function, method, and property has a docstring. A public
callable docstring describes parameters, returns, errors, the mathematical
meaning, and source keys. If code evaluates or defines a mathematical equation,
the docstring includes its equation ID.

`docs/api-reference.md` is generated from every package's `__all__`. After
adding or renaming a public name, regenerate it with
`PYTHONPATH=src python tools/generate_api_reference.py`;
`tests/test_api_reference.py` fails while it is stale.

Every top-level package carries a stability label (`stable`, `provisional`,
`experimental`) in `probstat.foundations.stability` (ADR-0048). A new package
starts as `experimental`. Removing or changing a name in a `stable` package
requires a deprecation period of at least one minor release: wrap the old
name with `probstat.foundations.deprecated(since=..., removal=...,
alternative=...)` and record it in the changelog.

## 3. Equation contract

Every implemented or documented equation receives one stable ID:

```text
EQ-<DOMAIN>-<NUMBER>
```

The equation must appear in `docs/equation-registry.md` with:

- symbols and domain assumptions;
- a plain-language interpretation;
- code locations; and
- at least one bibliography key from `docs/references.md`.

IDs are never recycled. A superseded equation remains in the registry and is
marked as such.

## 4. Decision records

Update [`docs/roadmap.md`](docs/roadmap.md) when scope changes. Record
architectural choices as numbered ADRs under [`docs/adr/`](docs/adr), with
context, decision, consequences, and the verification evidence behind them.

## Definition of done

- Tests pass with `python -m unittest discover -s tests -v`.
- The architecture test reports no dependency cycle or upward import.
- The documentation test reports no undocumented callable, unknown equation
  ID, or unknown reference key.
- The API reference is current (`tests/test_api_reference.py`).
- Numerical edge cases and error behavior are tested.
- User-facing examples remain executable, and every lab in `examples/`
  satisfies the lab contract of `tests/test_lab_contract.py` (ADR-0078):
  definitions, assumptions, registered equations, diagnostics, cost and
  error, references, a `SEED` constant, themed figures, and a linked
  regression test.

