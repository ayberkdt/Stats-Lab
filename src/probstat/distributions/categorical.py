"""Categorical distribution over arbitrary hashable labels."""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

from probstat.foundations.validation import require_non_negative_integer

from .constraints import SIMPLEX
from .support import FiniteSupport

LabelT = TypeVar("LabelT", bound=Hashable)


@dataclass(frozen=True, slots=True, init=False)
class Categorical(Generic[LabelT]):
    """An unordered finite distribution over distinct hashable labels.

    Categorical labels have no canonical ordering or numeric encoding. The class
    therefore exposes PMF, log-PMF, entropy, mode, and sampling—but deliberately
    does not invent a CDF, mean, or variance.

    Args:
        probabilities: Non-empty label-to-probability mapping summing to one.

    References:
        [REF-ROSS-2019, Section 4.6].

    """

    support: FiniteSupport[LabelT]
    _probabilities: Mapping[LabelT, float] = field(repr=False)

    def __init__(self, probabilities: Mapping[LabelT, float]) -> None:
        """Validate labels and freeze a normalized probability mapping.

        Args:
            probabilities: Label-to-probability mapping in desired display order.

        Raises:
            TypeError: If a label is unhashable or a probability is not real.
            ValueError: If input is empty, a probability is invalid, or the
                vector does not sum to one.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        support = FiniteSupport(probabilities.keys())
        masses = SIMPLEX.validate(probabilities.values(), name="probabilities")
        normalized = dict(zip(support.values, masses, strict=True))
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "_probabilities", MappingProxyType(normalized))

    @property
    def probabilities(self) -> Mapping[LabelT, float]:
        """Return the read-only normalized label-to-mass mapping.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        return self._probabilities

    def pmf(self, label: LabelT) -> float:
        """Return probability mass at ``label``.

        Args:
            label: Candidate category label.

        Returns:
            Label mass, or zero for an absent or unhashable label.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        try:
            return self._probabilities.get(label, 0.0)
        except TypeError:
            return 0.0

    def log_pmf(self, label: LabelT) -> float:
        """Return natural log probability mass at ``label``.

        Args:
            label: Candidate category label.

        Returns:
            Log mass, or negative infinity for zero/absent mass.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        mass = self.pmf(label)
        return -math.inf if mass == 0.0 else math.log(mass)

    def probability_of(self, labels: Iterable[LabelT]) -> float:
        """Return total mass of the union of supplied category labels.

        Repeated labels are counted once, matching event semantics.

        Args:
            labels: Category labels forming an event.

        Returns:
            Probability of the label set; absent labels contribute zero.

        Raises:
            TypeError: If a supplied label is unhashable.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        try:
            unique = set(labels)
        except TypeError as error:
            raise TypeError("event labels must be hashable") from error
        return math.fsum(self.pmf(label) for label in unique)

    @property
    def mode(self) -> tuple[LabelT, ...]:
        """Return all labels tied for maximal probability mass.

        Returns:
            One or more labels in declared order.

        References:
            [EQ-DIST-CAT-001], [REF-ROSS-2019, Section 4.6].

        """
        maximum = max(self._probabilities.values())
        return tuple(
            label for label in self.support if self._probabilities[label] == maximum
        )

    @property
    def entropy(self) -> float:
        """Return Shannon entropy in nats with ``0 log 0 = 0``.

        Equation:
            [EQ-DIST-CAT-002].

        References:
            [REF-COVER-THOMAS-2006, Section 2.1].

        """
        return -math.fsum(
            mass * math.log(mass)
            for mass in self._probabilities.values()
            if mass > 0.0
        )

    def sample(
        self, size: int = 1, *, seed: int | None = None
    ) -> tuple[LabelT, ...]:
        """Draw reproducible categorical observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local generator.

        Returns:
            Tuple of labels drawn according to their masses.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter III]. Sampling introduces no new equation.

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        return tuple(
            generator.choices(
                self.support.values,
                weights=tuple(self._probabilities.values()),
                k=checked_size,
            )
        )

