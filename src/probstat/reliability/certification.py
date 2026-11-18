"""Certifying a failure probability by simulation.

A system is acceptable when its failure probability ``p_f = P(y(X) > S)``
does not exceed an acceptable risk ``alpha``. Only an estimate ``p_hat_n``
from ``n`` model runs is available, so the decision rule adds a safety margin
and certifies when ``p_hat_n + c(n, alpha, beta) <= alpha``. The margin is
chosen so that a system with ``p_f >= alpha`` is certified with probability at
most ``beta``, the false-certification probability. Two probabilities are
therefore involved: the risk ``alpha`` on the system and the confidence
``1 - beta`` on the estimator.

Rules:

- ``"normal"``: ``c = z_(1-beta) sqrt(alpha (1 - alpha) / n)``, from the
  central limit theorem; ``max_(p_f >= alpha) P(certify)`` tends to ``beta``
  as ``n`` grows but may exceed it for small ``n alpha``.
- ``"clopper-pearson"``: certify when the exact one-sided Clopper-Pearson
  upper bound of level ``1 - beta`` is at most ``alpha``. This keeps the
  false-certification probability at most ``beta`` for every ``n``.

Under either rule the decision depends on the failure count ``X ~
Binomial(n, p_f)`` only through ``X <= k_n``, so the probability of
certifying is a binomial distribution function, evaluated here as a
regularized incomplete beta function (no sum over ``n`` terms; ``n`` up to
``1e10`` and beyond).

For conditioned sampling with moving particles (Guyader et al. 2011; Walter
2015) the number of moves ``M`` is ``Poisson(-N ln p_f)`` for ``N``
particles, the estimator is ``(1 - 1/N)^M``, and the same certification
question has an exact Poisson answer
(:func:`moving_particles_certification_probability`). The particle
algorithm itself is not implemented here.

Assumptions: model runs are independent draws of the input law and the model
is deterministic. The guarantee concerns sampling error only, not the
modeling of the input law or of the model itself.

References:
    [REF-CLOPPER-PEARSON-1934], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011],
    [REF-WALTER-2015].

"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

from probstat.distributions import Binomial, Poisson
from probstat.foundations.validation import require_finite_real

CERTIFICATION_RULES = ("normal", "clopper-pearson")
# Number of lattice plateaus checked above the bisection result when the
# stable certification sample size is located.
CERTIFICATION_SCAN_PLATEAUS = 32
# Largest sample size explored by the sample-size search.
CERTIFICATION_MAX_SIZE = 10**15


@dataclass(frozen=True, slots=True)
class CertificationProbability:
    """Probability that a Monte Carlo study certifies the system.

    Args:
        size: Number of model runs ``n``.
        failure_probability: True failure probability ``p_f``.
        risk: Acceptable risk ``alpha``.
        false_certification: Allowed false-certification probability
            ``beta``.
        rule: ``"normal"`` or ``"clopper-pearson"``.
        max_failures: Largest failure count ``k_n`` that still certifies;
            ``-1`` when no outcome certifies.
        probability: ``P(X <= k_n)`` for ``X ~ Binomial(n, p_f)``.

    References:
        [REF-CLOPPER-PEARSON-1934].

    """

    size: int
    failure_probability: float
    risk: float
    false_certification: float
    rule: str
    max_failures: int
    probability: float


@dataclass(frozen=True, slots=True)
class CertificationSampleSize:
    """Sample size after which certification is likely.

    Args:
        size: Smallest ``n`` from which the certification probability stays
            at or above ``power`` (the start of the first lattice plateau
            after which no plateau dips below ``power``).
        first_size: Smallest ``n`` whose certification probability reaches
            ``power``; ``first_size <= size`` because the probability is not
            monotone in ``n``.
        probability: Certification probability at ``size``.
        max_failures: Largest certifying failure count at ``size``.
        power: Required certification probability.
        failure_probability: True failure probability ``p_f < alpha``.
        risk: Acceptable risk ``alpha``.
        false_certification: Allowed false-certification probability
            ``beta``.
        rule: ``"normal"`` or ``"clopper-pearson"``.

    References:
        [REF-CLOPPER-PEARSON-1934].

    """

    size: int
    first_size: int
    probability: float
    max_failures: int
    power: float
    failure_probability: float
    risk: float
    false_certification: float
    rule: str


@dataclass(frozen=True, slots=True)
class MovingParticlesCertification:
    """Certification probability of a moving-particles estimate.

    Args:
        particles: Number of particles ``N``.
        failure_probability: True failure probability ``p_f``.
        risk: Acceptable risk ``alpha``.
        false_certification: Allowed false-certification probability
            ``beta``.
        steps: Model calls per particle move (Markov-chain steps).
        min_moves: Smallest move count ``m*`` that certifies.
        probability: ``P(M >= m*)`` for ``M ~ Poisson(-N ln p_f)``.
        expected_moves: ``E[M] = -N ln p_f``.
        expected_calls: ``N + steps * E[M]``, the expected number of model
            calls (``N`` initial draws, ``steps`` calls per move).

    References:
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015].

    """

    particles: int
    failure_probability: float
    risk: float
    false_certification: float
    steps: int
    min_moves: int
    probability: float
    expected_moves: float
    expected_calls: float


def _open_level(value: float, *, name: str) -> float:
    """Return a finite level strictly inside ``(0, 1)``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is outside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    level = require_finite_real(value, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return level


def _closed_probability(value: float, *, name: str) -> float:
    """Return a finite probability in ``[0, 1]``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is outside ``[0, 1]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    level = require_finite_real(value, name=name)
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return level


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer, rejecting booleans.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is smaller than one.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _check_rule(rule: str) -> str:
    """Return a known certification rule name.

    Raises:
        ValueError: If ``rule`` is unknown.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if rule not in CERTIFICATION_RULES:
        raise ValueError(f"rule must be one of {CERTIFICATION_RULES}")
    return rule


def _largest_true(predicate: Callable[[int], bool], low: int, high: int) -> int:
    """Return the largest ``k`` in ``[low, high)`` with ``predicate(k)`` true.

    ``predicate`` must be true at ``low`` and false at ``high`` and switch
    once in between.

    References:
        Bisection on an integer lattice [REF-PROBSTAT-VALIDATION]. No
        equation is evaluated.

    """
    while high - low > 1:
        middle = (low + high) // 2
        if predicate(middle):
            low = middle
        else:
            high = middle
    return low


def _first_true(predicate: Callable[[int], bool], guess: int, *, limit: int) -> int:
    """Return the smallest ``n >= 1`` with ``predicate(n)`` true.

    ``predicate`` must be monotone (false, then true). The bracket is found
    on a geometric grid around ``guess`` and refined by bisection.

    Raises:
        ValueError: If ``predicate`` stays false up to ``limit``.

    References:
        Bisection on an integer lattice [REF-PROBSTAT-VALIDATION]. No
        equation is evaluated.

    """
    guess = min(max(1, guess), limit)
    if predicate(guess):
        high, low = guess, guess // 2
        while low >= 1 and predicate(low):
            high, low = low, low // 2
    else:
        low, high = guess, min(2 * guess, limit)
        while not predicate(high):
            if high >= limit:
                raise ValueError(
                    f"the requested certification needs more than {limit} runs"
                )
            low, high = high, min(2 * high, limit)
    while high - low > 1:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def certification_margin(
    size: int, risk: float, false_certification: float = 0.05
) -> float:
    """Return the normal-theory certification margin ``c(n, alpha, beta)``.

    ``c = z_(1-beta) sqrt(alpha (1 - alpha) / n)`` with ``z_(1-beta)`` the
    standard-normal quantile. Since ``sqrt(n)(p_hat_n - p_f)`` tends to
    ``N(0, p_f (1 - p_f))``, the rule ``p_hat_n + c <= alpha`` certifies a
    system with ``p_f = alpha`` with probability tending to ``beta``.

    Args:
        size: Number of model runs ``n`` (positive integer).
        risk: Acceptable risk ``alpha`` strictly inside ``(0, 1)``.
        false_certification: ``beta`` strictly inside ``(0, 1)``; a value at
            or above one half gives a non-positive margin.

    Returns:
        The margin ``c``.

    Raises:
        TypeError: If ``size`` is not an integer or a level is not real.
        ValueError: If a level is outside ``(0, 1)``.

    Equation:
        [EQ-REL-CERT-001].

    References:
        [REF-ROSS-2019, Section 8.3].

    """
    n = _positive_integer(size, name="size")
    alpha = _open_level(risk, name="risk")
    beta = _open_level(false_certification, name="false_certification")
    z = NormalDist().inv_cdf(1.0 - beta)
    return z * math.sqrt(alpha * (1.0 - alpha) / n)


def _certifies(size: int, count: int, alpha: float, beta: float, rule: str) -> bool:
    """Return whether ``count`` failures in ``size`` runs certify.

    ``"normal"``: ``count / n + c <= alpha``. ``"clopper-pearson"``:
    ``P(Binomial(n, alpha) <= count) <= beta``. Both are monotone in
    ``count``, so ``k_n >= count`` exactly when this is true.

    References:
        [REF-CLOPPER-PEARSON-1934]. Evaluates [EQ-REL-CERT-001] and
        [EQ-REL-CERT-002].

    """
    if count > size:
        return False
    if rule == "normal":
        margin = NormalDist().inv_cdf(1.0 - beta) * math.sqrt(
            alpha * (1.0 - alpha) / size
        )
        return count / size + margin <= alpha
    return float(Binomial(size, alpha).cdf(count)) <= beta


def _max_failures(size: int, alpha: float, beta: float, rule: str) -> int:
    """Return the largest failure count ``k_n`` that certifies (``-1`` if none).

    ``"normal"``: the largest ``k <= n`` with ``k / n + c <= alpha``.
    ``"clopper-pearson"``: the upper bound of level ``1 - beta`` is at most
    ``alpha`` exactly when ``P(Binomial(n, alpha) <= k) <= beta``.

    References:
        [REF-CLOPPER-PEARSON-1934]. Evaluates [EQ-REL-CERT-001] and
        [EQ-REL-CERT-002].

    """
    if rule == "normal":
        margin = NormalDist().inv_cdf(1.0 - beta) * math.sqrt(
            alpha * (1.0 - alpha) / size
        )
        threshold = size * (alpha - margin)
        k = -1 if threshold < 0.0 else min(size, math.floor(threshold))
        while k + 1 <= size and (k + 1) / size + margin <= alpha:
            k += 1
        while k >= 0 and k / size + margin > alpha:
            k -= 1
        return k
    law = Binomial(size, alpha)
    if float(law.cdf(0)) > beta:
        return -1
    return _largest_true(lambda k: float(law.cdf(k)) <= beta, 0, size)


def certification_probability(
    size: int,
    failure_probability: float,
    risk: float,
    false_certification: float = 0.05,
    *,
    rule: str = "normal",
) -> CertificationProbability:
    """Return the exact probability that ``n`` runs certify the system.

    With ``X ~ Binomial(n, p_f)`` failures, the rule certifies exactly when
    ``X <= k_n``, so ``P(certify) = P(X <= k_n) = I_(1-p_f)(n - k_n, k_n + 1)``
    (regularized incomplete beta function). For the normal rule ``k_n`` is
    the largest ``k`` with ``k/n + z_(1-beta) sqrt(alpha (1-alpha)/n) <=
    alpha``; for the Clopper-Pearson rule it is the largest ``k`` with
    ``P(Binomial(n, alpha) <= k) <= beta``, which gives ``P(certify) <=
    beta`` whenever ``p_f >= alpha``, for every ``n``.

    Args:
        size: Number of model runs ``n`` (positive integer).
        failure_probability: True failure probability ``p_f`` in ``[0, 1]``.
        risk: Acceptable risk ``alpha`` strictly inside ``(0, 1)``.
        false_certification: ``beta`` strictly inside ``(0, 1)``.
        rule: ``"normal"`` or ``"clopper-pearson"``.

    Returns:
        The certifying failure count and the certification probability.

    Raises:
        TypeError: If ``size`` is not an integer or a level is not real.
        ValueError: If a level is out of range or ``rule`` is unknown.

    Equation:
        [EQ-REL-CERT-001], [EQ-REL-CERT-002].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    n = _positive_integer(size, name="size")
    p_f = _closed_probability(failure_probability, name="failure_probability")
    alpha = _open_level(risk, name="risk")
    beta = _open_level(false_certification, name="false_certification")
    _check_rule(rule)
    k = _max_failures(n, alpha, beta, rule)
    probability = 0.0 if k < 0 else float(Binomial(n, p_f).cdf(k))
    return CertificationProbability(n, p_f, alpha, beta, rule, k, probability)


def certification_sample_size(
    failure_probability: float,
    risk: float,
    false_certification: float = 0.05,
    *,
    power: float = 0.95,
    rule: str = "normal",
) -> CertificationSampleSize:
    """Return the number of runs after which certification is likely.

    The certification probability ``P_n`` is not monotone in ``n``: the
    certifying count ``k_n`` is an integer, so ``P_n`` jumps up when
    ``k_n`` increases and decreases while ``k_n`` stays constant. The runs
    therefore split into plateaus ``{n : k_n = k}``, and on each plateau the
    smallest value of ``P_n`` is at its last ``n``. The returned ``size`` is
    the smallest ``n`` from which ``P_n >= power`` for all larger ``n``: the
    start of the first plateau after which no plateau end falls below
    ``power``. ``first_size`` is the smallest ``n`` with ``P_n >= power``.

    Search: plateau starts are found on a geometric grid of ``n`` refined by
    bisection; the first good plateau is located by doubling and bisection
    on ``k``, then the next ``CERTIFICATION_SCAN_PLATEAUS`` plateaus are
    checked and the start is moved past any plateau that dips. Plateau ends
    increase towards one for ``p_f < alpha``; beyond the scanned window this
    monotonicity is assumed.

    Args:
        failure_probability: True failure probability ``p_f`` in
            ``[0, alpha)``.
        risk: Acceptable risk ``alpha`` strictly inside ``(0, 1)``.
        false_certification: ``beta`` strictly inside ``(0, 1)``.
        power: Required certification probability strictly inside ``(0, 1)``.
        rule: ``"normal"`` or ``"clopper-pearson"``.

    Returns:
        The stable and the first sample size with the probability at the
        stable size.

    Raises:
        TypeError: If a level is not real.
        ValueError: If a level is out of range, ``p_f >= alpha``, ``rule`` is
            unknown, or more than ``CERTIFICATION_MAX_SIZE`` runs are needed.

    Equation:
        [EQ-REL-CERT-002].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    p_f = _closed_probability(failure_probability, name="failure_probability")
    alpha = _open_level(risk, name="risk")
    beta = _open_level(false_certification, name="false_certification")
    target = _open_level(power, name="power")
    _check_rule(rule)
    if p_f >= alpha:
        raise ValueError(
            "failure_probability must be below risk; otherwise the "
            "certification probability stays near or below false_certification"
        )
    z = NormalDist().inv_cdf(1.0 - beta)
    spread = z * math.sqrt(alpha * (1.0 - alpha))
    starts: dict[int, int] = {}

    def start(k: int) -> int:
        """Return the first ``n`` with ``k_n >= k`` (cached).

        References:
            [EQ-REL-CERT-002]. No new equation.

        """
        if k not in starts:
            # n alpha - spread sqrt(n) = k solved for sqrt(n).
            root = (spread + math.sqrt(spread * spread + 4.0 * alpha * k)) / (
                2.0 * alpha
            )
            starts[k] = _first_true(
                lambda n: _certifies(n, k, alpha, beta, rule),
                math.ceil(root * root),
                limit=CERTIFICATION_MAX_SIZE,
            )
        return starts[k]

    def value(n: int, k: int) -> float:
        """Return ``P(Binomial(n, p_f) <= k)``.

        References:
            [EQ-REL-CERT-002]. No new equation.

        """
        return float(Binomial(n, p_f).cdf(k))

    def end_passes(k: int) -> bool:
        """Return whether plateau ``k`` keeps ``P_n >= power`` to its end.

        References:
            [EQ-REL-CERT-002]. No new equation.

        """
        first, following = start(k), start(k + 1)
        if following == first:  # empty plateau
            return True
        return value(following - 1, k) >= target

    def start_passes(k: int) -> bool:
        """Return whether ``P_n >= power`` at the start of plateau ``k``.

        References:
            [EQ-REL-CERT-002]. No new equation.

        """
        return value(start(k), k) >= target

    def smallest(predicate: Callable[[int], bool]) -> int:
        """Return the smallest ``k >= 0`` with ``predicate`` true (doubling, bisection).

        References:
            Bisection on an integer lattice [REF-PROBSTAT-VALIDATION]. No
            equation is evaluated.

        """
        if predicate(0):
            return 0
        low, high = 0, 1
        while not predicate(high):
            low, high = high, 2 * high
        while high - low > 1:
            middle = (low + high) // 2
            if predicate(middle):
                high = middle
            else:
                low = middle
        return high

    plateau = smallest(end_passes)
    offset = 1
    while offset <= CERTIFICATION_SCAN_PLATEAUS:
        if not end_passes(plateau + offset):
            plateau += offset + 1
            offset = 1
        else:
            offset += 1
    stable = start(plateau)
    first_plateau = smallest(start_passes)
    first = min(start(first_plateau), stable)
    return CertificationSampleSize(
        size=stable,
        first_size=first,
        probability=value(stable, plateau),
        max_failures=_max_failures(stable, alpha, beta, rule),
        power=target,
        failure_probability=p_f,
        risk=alpha,
        false_certification=beta,
        rule=rule,
    )


def moving_particles_certification_probability(
    particles: int,
    failure_probability: float,
    risk: float,
    false_certification: float = 0.05,
    *,
    steps: int = 10,
) -> MovingParticlesCertification:
    """Return the probability that a moving-particles estimate certifies.

    For ``N`` particles of an idealized moving-particles run the number of
    moves needed to push every particle above the threshold is
    ``M ~ Poisson(-N ln p_f)`` and ``p_hat = (1 - 1/N)^M``. Observing
    ``M = m``, the exact one-sided Poisson lower bound ``lambda_L(m)`` of
    level ``1 - beta`` (``P(Poisson(lambda_L) >= m) = beta``) gives the upper
    bound ``exp(-lambda_L / N)`` of ``p_f``. It is at most ``alpha`` exactly
    when ``P(Poisson(-N ln alpha) >= m) <= beta``, i.e. when ``m >= m*``.
    Hence ``P(certify) = P(Poisson(-N ln p_f) >= m*)``, at most ``beta`` for
    every ``p_f >= alpha``.

    Cost: ``N`` initial draws plus ``steps`` model calls per move, so the
    expected number of calls is ``N (1 - steps ln p_f)``. The course slide
    plots against ``N Q (1 - ln p_f)`` with ``Q`` Markov-chain steps, which
    also charges ``Q`` calls to each initial draw; the two differ by
    ``N (Q - 1)``, small next to ``-N Q ln p_f`` for rare events.

    Args:
        particles: Number of particles ``N`` (positive integer).
        failure_probability: True failure probability ``p_f`` in ``[0, 1]``.
        risk: Acceptable risk ``alpha`` strictly inside ``(0, 1)``.
        false_certification: ``beta`` strictly inside ``(0, 1)``.
        steps: Model calls per move (positive integer).

    Returns:
        The certifying move count, the certification probability, and the
        expected number of moves and model calls.

    Raises:
        TypeError: If an integer argument is not an integer or a level is
            not real.
        ValueError: If a level is out of range.

    Equation:
        [EQ-REL-CERT-003].

    References:
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015].

    """
    n = _positive_integer(particles, name="particles")
    p_f = _closed_probability(failure_probability, name="failure_probability")
    alpha = _open_level(risk, name="risk")
    beta = _open_level(false_certification, name="false_certification")
    q = _positive_integer(steps, name="steps")
    rate = -n * math.log(alpha)
    law = Poisson(rate)
    z = NormalDist().inv_cdf(1.0 - beta)
    guess = max(1, math.ceil(rate + z * math.sqrt(rate)))
    # Smallest m >= 1 with P(Poisson(rate) >= m) <= beta.
    min_moves = _first_true(
        lambda m: float(law.sf(m - 1)) <= beta, guess, limit=CERTIFICATION_MAX_SIZE
    )
    if p_f == 0.0:
        probability, moves = 1.0, math.inf
    elif p_f == 1.0:
        probability, moves = 0.0, 0.0
    else:
        moves = -n * math.log(p_f)
        probability = float(Poisson(moves).sf(min_moves - 1))
    return MovingParticlesCertification(
        particles=n,
        failure_probability=p_f,
        risk=alpha,
        false_certification=beta,
        steps=q,
        min_moves=min_moves,
        probability=probability,
        expected_moves=moves,
        expected_calls=n + q * moves,
    )
