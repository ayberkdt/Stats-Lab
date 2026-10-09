"""Bernoulli distribution for a single binary trial."""

from __future__ import annotations

import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class Bernoulli:
    """A Bernoulli distribution with success probability ``p``.

    Args:
        p: Probability of outcome one, constrained to ``[0, 1]``.

    References:
        [REF-ROSS-2019, Section 4.2].

    """

    p: float

    def __post_init__(self) -> None:
        """Validate and normalize the success probability.

        Raises:
            TypeError: If ``p`` is not real.
            ValueError: If ``p`` is non-finite or outside ``[0, 1]``.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        object.__setattr__(self, "p", require_probability(self.p))

    def pmf(self, k: int) -> float:
        """Evaluate probability mass at integer outcome ``k``.

        Args:
            k: Candidate outcome. Outcomes outside ``{0, 1}`` have mass zero.

        Returns:
            ``P(X = k)``.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-BER-001].

        References:
            [REF-ROSS-2019, Section 4.2].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k == 0:
            return 1.0 - self.p
        if k == 1:
            return self.p
        return 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the Bernoulli cumulative distribution function at ``x``.

        Args:
            x: Real-valued evaluation point.

        Returns:
            ``P(X <= x)`` as a right-continuous step function.

        Raises:
            TypeError: If ``x`` is not a real number.
            ValueError: If ``x`` is not finite.

        Equation:
            [EQ-DIST-BER-002].

        References:
            [REF-ROSS-2019, Section 4.2].

        """
        from probstat.foundations.validation import coerce_finite_sample

        point = coerce_finite_sample((x,), name="x")[0]
        if point < 0.0:
            return 0.0
        if point < 1.0:
            return 1.0 - self.p
        return 1.0

    @property
    def mean(self) -> float:
        """Return expected value ``E[X]``.

        Equation:
            [EQ-DIST-BER-003].

        References:
            [REF-ROSS-2019, Section 4.2].

        """
        return self.p

    @property
    def variance(self) -> float:
        """Return variance ``Var(X)``.

        Equation:
            [EQ-DIST-BER-004].

        References:
            [REF-ROSS-2019, Section 4.2].

        """
        return self.p * (1.0 - self.p)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[int, ...]:
        """Draw reproducible pseudo-random Bernoulli observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            A tuple containing ``size`` zeros and ones.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter II]. Sampling uses inverse comparison
            with independent uniforms and does not introduce a new equation.

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        return tuple(int(generator.random() < self.p) for _ in range(checked_size))

