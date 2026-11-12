"""Smoke-test an installed ``probstat-library`` distribution.

Run it with the interpreter of an environment into which the wheel was
installed, from any directory, in isolated mode so that neither the working
directory nor ``PYTHONPATH`` can put the source tree on the path::

    python -I tools/smoke_test_wheel.py

The script checks that ``probstat`` is imported from the installed
distribution and not from ``src/``; that ``probstat.__version__`` equals the
installed metadata version; that ``py.typed`` and both licence files are
installed; that every module of the package imports and every name in a
package's ``__all__`` resolves; and that a small propagation and a FORM
analysis of a linear limit state in two normal variables reproduce the closed
form. Modules that need an optional dependency that is not installed (for
example Matplotlib) are reported and skipped. The exit status is non-zero
when any check fails.

References:
    [REF-PYPA-CORE-METADATA], [REF-PYTHON-IMPORTLIB-METADATA], and
    [EQ-REL-FORM-001] for the closed-form reliability index.

"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import pkgutil
import sys
from pathlib import Path

DISTRIBUTION = "probstat-library"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_DEPENDENCIES = frozenset({"matplotlib", "numpy"})
REQUIRED_FILES = ("probstat/py.typed", "licenses/LICENSE", "JOE-KUO-SOBOL.txt")


def check_installation() -> list[str]:
    """Check where ``probstat`` comes from, its version, and installed files.

    Returns:
        Failure messages; empty when every check passes.

    References:
        [REF-PYPA-CORE-METADATA], [REF-PYTHON-IMPORTLIB-METADATA].
        No equation is evaluated.

    """
    import probstat

    failures: list[str] = []
    location = Path(probstat.__file__).resolve()
    if location.is_relative_to(REPOSITORY_ROOT / "src"):
        failures.append(f"probstat was imported from the source tree: {location}")
    metadata_version = importlib.metadata.version(DISTRIBUTION)
    if probstat.__version__ != metadata_version:
        failures.append(
            f"probstat.__version__ {probstat.__version__!r} differs from the "
            f"metadata version {metadata_version!r}"
        )
    files = importlib.metadata.distribution(DISTRIBUTION).files or []
    installed = [file.as_posix() for file in files]
    for required in REQUIRED_FILES:
        if not any(name.endswith(required) for name in installed):
            failures.append(f"installed distribution lacks {required}")
    print(f"probstat {probstat.__version__} from {location.parent}")
    return failures


def check_imports() -> list[str]:
    """Import every module of ``probstat`` and resolve every ``__all__`` name.

    Returns:
        Failure messages; empty when every module imports.

    References:
        [REF-PYTHON-IMPORTLIB-METADATA]. No equation is evaluated.

    """
    import probstat

    failures: list[str] = []
    imported = skipped = 0
    for info in pkgutil.walk_packages(probstat.__path__, prefix="probstat."):
        try:
            module = importlib.import_module(info.name)
        except ModuleNotFoundError as error:
            if error.name in OPTIONAL_DEPENDENCIES:
                skipped += 1
                print(f"skipped {info.name}: optional dependency {error.name}")
                continue
            failures.append(f"{info.name}: {error!r}")
            continue
        except Exception as error:  # noqa: BLE001 - report every import failure
            failures.append(f"{info.name}: {error!r}")
            continue
        imported += 1
        for name in getattr(module, "__all__", ()):
            if not hasattr(module, name):
                failures.append(f"{info.name}.__all__ names missing {name!r}")
    print(f"imported {imported} modules, skipped {skipped}")
    return failures


def check_example() -> list[str]:
    """Propagate and run FORM on ``g = R - S`` with independent normals.

    With ``R ~ N(5, 0.5)`` and ``S ~ N(3, 0.4)`` the margin is normal with
    mean 2 and standard deviation ``sqrt(0.41)``, so the reliability index is
    ``beta = 2 / sqrt(0.41)`` exactly and FORM is exact. The Monte Carlo mean
    is judged against four standard errors, not against pinned digits.

    Returns:
        Failure messages; empty when both analyses agree with the closed form.

    References:
        [EQ-REL-FORM-001], [REF-MELCHERS-BECK-2018, Chapters 2 and 3].

    """
    from probstat.distributions import Normal
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.propagation import propagate
    from probstat.reliability import ModelLimitState, form_hlrf
    from probstat.uq import QuantityOfInterest

    inputs = JointInput.independent(
        RandomInput("strength", Normal(5.0, 0.5)),
        RandomInput("load", Normal(3.0, 0.4)),
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"margin": p["strength"] - p["load"]},
        (QuantityOfInterest("margin", "strength minus load", "kN"),),
    )
    failures: list[str] = []
    size = 2000
    sd = math.sqrt(0.41)
    mean = propagate(model, size, seed=2026)["margin"].mean
    if abs(mean - 2.0) > 4.0 * sd / math.sqrt(size):
        failures.append(f"Monte Carlo mean {mean} is not within 4 s.e. of 2")
    form = form_hlrf(ModelLimitState(model, "margin", 0.0, failure="below"))
    beta = 2.0 / sd
    probability = 0.5 * math.erfc(beta / math.sqrt(2.0))
    if not form.converged or abs(form.beta - beta) > 1e-6 * beta:
        failures.append(f"FORM beta {form.beta} differs from {beta}")
    if abs(form.probability - probability) > 1e-5 * probability:
        failures.append(
            f"FORM probability {form.probability} differs from {probability}"
        )
    print(f"mean {mean:.4f} (exact 2), FORM beta {form.beta:.6f} (exact {beta:.6f})")
    return failures


def main() -> int:
    """Run every check and print the failures.

    Returns:
        Process exit status: 0 when every check passes, 1 otherwise.

    References:
        [REF-PYPA-CORE-METADATA]. No equation is evaluated.

    """
    failures = check_installation() + check_imports() + check_example()
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("smoke test passed" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
