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

## 4. Memory contract

Update `memory/ROADMAP.md` when scope changes. Record architectural choices as
ADRs under `memory/decisions/`. Add a short session note for meaningful work,
including decisions, verification results, and the next safe step.

## Definition of done

- Tests pass with `python -m unittest discover -s tests -v`.
- The architecture test reports no dependency cycle or upward import.
- The documentation test reports no undocumented callable, unknown equation
  ID, or unknown reference key.
- Numerical edge cases and error behavior are tested.
- User-facing examples remain executable.

