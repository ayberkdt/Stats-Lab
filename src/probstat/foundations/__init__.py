"""Dependency-free validation and numerical foundation utilities.

Every other package builds on these primitives: argument validators, the
structural protocols that distribution and law objects satisfy, the
SplitMix64 seed tree behind every ``seed=`` argument, and the stability
labels of the public API.

Main entry points:
    ``RandomStream``: a node of a reproducible seed tree; ``child`` and
        ``spawn`` derive independent named substreams.
    ``derive_seed``: the 64-bit child seed of a root seed and a key path.
    ``resolve_stream``: turns ``None``, an integer, or a stream into a stream.
    ``require_finite_real``: the finite-real validator used by every package.
    ``require_probability``: validates a number in the closed interval [0, 1].
    ``package_stability``: the ``stable``/``provisional``/``experimental``
        label of a top-level package.
    ``deprecated``: decorator that warns before a public name is removed.

Assumptions:
    Seeds and integer stream keys are non-negative Python integers and text
    keys are strings; booleans are rejected. Reproducibility is defined
    within this library: the same seed and key path give the same seed
    value and, through Python's ``random.Random``, the same draws. It does
    not carry over to other languages or libraries. Distinct key paths give
    unrelated 64-bit seeds by SplitMix64 mixing of the path.

Common pitfalls:
    - ``seed=None`` (and ``resolve_stream(None)``) draws fresh
      operating-system entropy and is never reproducible; pass an integer
      or a ``RandomStream``.
    - ``RandomStream.generator()`` returns a new ``random.Random`` seeded
      with the node's seed on every call, so two calls replay the same
      numbers. Keep one generator per stream, or derive a child per use.
    - Key paths are ordered and nesting matters: ``child("a", "b")``,
      ``child("b", "a")`` and ``child("a").child("b")`` are three different
      streams, and the integer key ``0`` differs from the text key ``"0"``.
    - Reusing one integer seed for two inputs makes their draws dependent
      (for example ``Uniform().sample(n, seed=5)`` and
      ``Exponential().sample(n, seed=5)`` are monotone transforms of the
      same uniforms). Give each input its own ``root.child(name)``.
    - The validators reject ``bool`` where a number is expected
      (``require_finite_real(True, name="x")`` raises ``TypeError``) and
      ``require_probability`` accepts the endpoints 0 and 1.

Guide: docs/foundations/probabilistic-modeling.md

References:
    [REF-STEELE-LEA-FLOOD-2014] (SplitMix64 seed derivation),
    [REF-MATSUMOTO-NISHIMURA-1998] (the Mersenne Twister behind
    ``generator``), [REF-PROBSTAT-VALIDATION], [REF-PROBSTAT-API].

"""

from .protocols import (
    ContinuousProbabilityLaw,
    CumulativeDistribution,
    DiscreteProbabilityLaw,
    HasDimension,
    HasMeanAndVariance,
    HazardFunction,
    LogProbabilityDensity,
    LogProbabilityMassFunction,
    MultivariateLogDensity,
    MultivariateLogMass,
    MultivariateMoments,
    MultivariateSampler,
    ProbabilityDensity,
    ProbabilityMassFunction,
    QuantileFunction,
    SurvivalFunction,
)
from .random_streams import RandomStream, derive_seed, resolve_stream, splitmix64
from .stability import PACKAGE_STABILITY, Stability, deprecated, package_stability
from .validation import (
    coerce_finite_sample,
    require_finite_complex,
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)

__all__ = [
    "HasDimension",
    "MultivariateLogDensity",
    "MultivariateLogMass",
    "MultivariateMoments",
    "MultivariateSampler",
    "ContinuousProbabilityLaw",
    "CumulativeDistribution",
    "DiscreteProbabilityLaw",
    "HazardFunction",
    "HasMeanAndVariance",
    "LogProbabilityDensity",
    "LogProbabilityMassFunction",
    "PACKAGE_STABILITY",
    "ProbabilityDensity",
    "ProbabilityMassFunction",
    "QuantileFunction",
    "RandomStream",
    "Stability",
    "SurvivalFunction",
    "coerce_finite_sample",
    "deprecated",
    "derive_seed",
    "package_stability",
    "require_finite_complex",
    "require_finite_real",
    "require_non_negative_finite_real",
    "require_probability",
    "resolve_stream",
    "splitmix64",
]

