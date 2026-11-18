"""Surrogate posteriors over a candidate population for active learning.

An active-learning reliability method keeps the predictive mean and
variance of a surrogate at every point of a candidate population (the Monte
Carlo population of AK-MCS, the samples of one subset level, ...), adds one
true-model observation at a time, and reads the updated predictions back.
:class:`PopulationPosterior` is that contract (ADR-0107):

- :class:`GaussianProcessPopulationPosterior`: exact rank-one updates of a
  constant-mean Gaussian process at fixed hyperparameters, ``O(n)`` per
  population point and step, bit-identical to the batch factorization of
  :func:`probstat.surrogates.fit_gaussian_process` (ADR-0049);
- :class:`SurrogatePopulationPosterior`: any
  :class:`~probstat.surrogates.ProbabilisticSurrogate` refitted through a
  :class:`~probstat.surrogates.SurrogateFactory` after every observation and
  re-evaluated on the whole population.

Both keep their training set when the population is extended
(:meth:`add_population`) or replaced (:meth:`replace_population`), which is
what a method with changing candidate pools needs.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from probstat.linalg import Vector
from probstat.surrogates import (
    GaussianProcessRegression,
    ProbabilisticSurrogate,
    SurrogateFactory,
    kernel_value,
)

__all__ = [
    "GaussianProcessPopulationPosterior",
    "PopulationPosterior",
    "SurrogatePopulationPosterior",
]


@runtime_checkable
class PopulationPosterior(Protocol):
    """Predictive mean and variance of a surrogate over a candidate population.

    ``means[i]`` and ``variances[i]`` belong to ``population[i]``; the lists
    are updated in place by :meth:`add_training`, :meth:`add_population`,
    and :meth:`replace_population`.

    References:
        [EQ-SUR-GP-003], [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is
        evaluated by the protocol.

    """

    @property
    def population(self) -> Sequence[Vector]:
        """Return the candidate points.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        ...

    @property
    def means(self) -> Sequence[float]:
        """Return the predictive means over the population.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated by the protocol.

        """
        ...

    @property
    def variances(self) -> Sequence[float]:
        """Return the predictive variances over the population.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated by the protocol.

        """
        ...

    @property
    def training_points(self) -> Sequence[Vector]:
        """Return the inputs the posterior is conditioned on.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        ...

    def add_training(self, point: Sequence[float], value: float) -> None:
        """Condition on one more observation and update the population.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated by the protocol.

        """
        ...

    def add_population(self, points: Sequence[Sequence[float]]) -> None:
        """Append candidate points with their predictions.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated by the protocol.

        """
        ...

    def replace_population(self, points: Sequence[Sequence[float]]) -> None:
        """Drop the current candidates and predict at ``points`` instead.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated by the protocol.

        """
        ...


class GaussianProcessPopulationPosterior:
    """GP posterior over a population with rank-one training updates.

    With fixed hyperparameters (prior mean ``m``, kernel ``k``, noise
    ``s_n^2``) and ``K + s_n^2 I = L L'``, each population point keeps
    ``v(x) = L^-1 k_T(x)``, so ``mu(x) = m + v(x) . z`` with
    ``z = L^-1 (y - m)`` and ``sigma^2(x) = k(x, x) - ||v(x)||^2``. Adding a
    training point ``t`` appends one row to ``L`` and one entry to every
    ``v(x)`` and ``z``, at ``O(n)`` cost per population point instead of the
    ``O(n^2)`` of a fresh prediction. The result equals the batch posterior
    at the same hyperparameters up to rounding.

    The factorization reproduces :func:`probstat.surrogates.fit_gaussian_process`
    exactly: standardized output units, the signal variance ``exp(theta_0)``
    and nugget the fit used, :func:`~probstat.surrogates.kernel_value`, and
    ``math.fsum`` in every Cholesky and triangular-solve sum, so the initial
    factor is bit-identical to the batch factor; ``mu`` and ``sigma^2`` are
    mapped back to original units. This matters because maximum likelihood can
    end at hyperparameters whose kernel matrix is numerically singular
    (signal-to-nugget ratio near ``1 / eps``), accepted only because the fit's
    own factorization succeeded. Re-deriving the factor with different
    rounding (original units, plain ``sum``) made the pivots collapse to the
    noise floor and the population means meaningless (benchmark suite,
    ADR-0049: ``|mu - mu_batch|`` up to ``1e9`` and ``P_f = 0.64`` for a true
    ``8.2e-4`` on a parabolic limit state).

    The model is the constant-mean (simple-kriging) GP of the fit; a GP with
    a regression trend (universal kriging, PC-Kriging) needs the generic
    :class:`SurrogatePopulationPosterior` or its own update rule. Variances
    are not clipped, so cancellation can leave tiny negative values; the
    learning functions treat them as zero.

    Args:
        surrogate: Fitted GP whose hyperparameters and training data are used.
        population: Initial candidate points.

    References:
        [EQ-SUR-GP-003], [EQ-REL-AKMCS-001],
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """

    def __init__(
        self,
        surrogate: GaussianProcessRegression,
        population: Sequence[Sequence[float]],
    ) -> None:
        """Build the posterior of a fitted GP over the population.

        Raises:
            TypeError: If ``surrogate`` is not a
                :class:`~probstat.surrogates.GaussianProcessRegression`.

        References:
            [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        if not isinstance(surrogate, GaussianProcessRegression):
            raise TypeError("surrogate must be a GaussianProcessRegression")
        theta = surrogate.log_parameters
        self.surrogate = surrogate
        self.kernel = surrogate.kernel
        # Standardized units, exactly as fit_gaussian_process factorized them.
        self.signal = math.exp(theta[0])
        # nugget + tau sigma_f^2 [+ sigma_n^2], the fit's own diagonal noise.
        self.noise = surrogate.standardized_noise_variance
        self.scales = surrogate.hyperparameters.length_scales
        self.mean = surrogate.output_mean
        self.scale = surrogate.output_scale
        self.training: list[Vector] = []
        self.lower: list[list[float]] = []
        self.z: list[float] = []
        self.v: list[list[float]] = []
        self.mu: list[float] = []
        self.var: list[float] = []
        self._population: list[Vector] = []
        for point, value in zip(surrogate.points, surrogate.values, strict=True):
            self._extend_training(point, value)
        self.add_population(population)

    @property
    def population(self) -> list[Vector]:
        """Return the candidate points.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        return self._population

    @property
    def means(self) -> list[float]:
        """Return the predictive means ``mu`` over the population.

        References:
            [EQ-SUR-GP-003].

        """
        return self.mu

    @property
    def variances(self) -> list[float]:
        """Return the predictive variances ``sigma^2`` over the population.

        References:
            [EQ-SUR-GP-003].

        """
        return self.var

    @property
    def training_points(self) -> list[Vector]:
        """Return the inputs the posterior is conditioned on.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated.

        """
        return self.training

    def covariance(self, left: Vector, right: Vector) -> float:
        """Return the standardized-unit kernel ``k`` between two inputs.

        References:
            [EQ-SUR-GP-001], [EQ-SUR-GP-002],
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return kernel_value(self.kernel, left, right, self.scales, self.signal)

    def _solve(self, column: Sequence[float]) -> list[float]:
        """Return ``L^-1 column`` by forward substitution (``fsum`` sums).

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        solution: list[float] = []
        for row, entry in zip(self.lower, column, strict=True):
            size = len(solution)
            solution.append(
                (entry - math.fsum(map(operator.mul, row, solution))) / row[size]
            )
        return solution

    def _extend_training(
        self, point: Sequence[float], value: float
    ) -> tuple[list[float], float]:
        """Append one training point to ``L`` and ``z``; return ``(w, pivot)``.

        References:
            [EQ-SUR-GP-003], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        point = tuple(point)
        cross = [self.covariance(point, known) for known in self.training]
        w = self._solve(cross)
        pivot_square = (self.signal + self.noise) - math.fsum(map(operator.mul, w, w))
        pivot = math.sqrt(max(pivot_square, self.noise))
        self.lower.append([*w, pivot])
        target = (value - self.mean) / self.scale
        self.z.append((target - math.fsum(map(operator.mul, w, self.z))) / pivot)
        self.training.append(point)
        return w, pivot

    def add_training(self, point: Sequence[float], value: float) -> None:
        """Condition on one more observation and update the population.

        References:
            [EQ-SUR-GP-003], [EQ-REL-AKMCS-001],
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        row, pivot = self._extend_training(point, value)
        new_point = self.training[-1]
        z_new = self.z[-1]
        scale = self.scale
        for index, (x, v) in enumerate(zip(self._population, self.v, strict=True)):
            entry = (
                self.covariance(x, new_point) - math.fsum(map(operator.mul, v, row))
            ) / pivot
            v.append(entry)
            self.mu[index] += scale * entry * z_new
            self.var[index] -= scale * scale * entry * entry

    def add_population(self, points: Sequence[Sequence[float]]) -> None:
        """Append population points with their full posterior state.

        References:
            [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        scale = self.scale
        for raw in points:
            x = tuple(raw)
            v = self._solve([self.covariance(x, known) for known in self.training])
            self._population.append(x)
            self.v.append(v)
            self.mu.append(self.mean + scale * math.fsum(map(operator.mul, v, self.z)))
            self.var.append(
                scale * scale * (self.signal - math.fsum(map(operator.mul, v, v)))
            )

    def replace_population(self, points: Sequence[Sequence[float]]) -> None:
        """Drop the candidates and predict at ``points``, keeping the training set.

        References:
            [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        self._population.clear()
        self.v.clear()
        self.mu.clear()
        self.var.clear()
        self.add_population(points)


class SurrogatePopulationPosterior:
    """Population predictions of a refitted probabilistic surrogate.

    Every :meth:`add_training` call passes all training data and the current
    surrogate to ``factory`` and re-evaluates the whole population with the
    returned surrogate, so the cost per step is one refit plus ``N``
    predictions. Use it for surrogates without an incremental update (a GP
    with a trend, an ensemble, ...); for a constant-mean GP the
    :class:`GaussianProcessPopulationPosterior` is much cheaper.

    Args:
        factory: Refit hook ``(points, values, previous) -> surrogate``.
        points: Initial training inputs.
        values: Initial training responses.
        population: Initial candidate points.
        surrogate: Optional surrogate already fitted to ``points`` and
            ``values`` (otherwise ``factory(points, values, None)`` is
            called).

    References:
        [EQ-SUR-GP-003], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """

    def __init__(
        self,
        factory: SurrogateFactory,
        points: Sequence[Sequence[float]],
        values: Sequence[float],
        population: Sequence[Sequence[float]],
        *,
        surrogate: ProbabilisticSurrogate | None = None,
    ) -> None:
        """Fit (or adopt) the surrogate and predict over the population.

        Raises:
            TypeError: If ``factory`` is not callable or returns an object
                without ``predict_with_variance``.
            ValueError: If the training lengths differ or a prediction is not
                finite.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        if not callable(factory):
            raise TypeError("factory must be callable")
        if len(points) != len(values):
            raise ValueError("points and values must have the same length")
        self.factory = factory
        self.training: list[Vector] = [tuple(float(c) for c in p) for p in points]
        self.values: list[float] = [float(value) for value in values]
        self.fits = 0
        self.surrogate = self._checked(
            self._refit(None) if surrogate is None else surrogate
        )
        self._population: list[Vector] = []
        self.mu: list[float] = []
        self.var: list[float] = []
        self.add_population(population)

    @staticmethod
    def _checked(surrogate: object) -> ProbabilisticSurrogate:
        """Return ``surrogate`` after checking the probabilistic interface.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        if not isinstance(surrogate, ProbabilisticSurrogate):
            raise TypeError(
                "the surrogate must provide predict and predict_with_variance"
            )
        return surrogate

    def _refit(self, previous: ProbabilisticSurrogate | None) -> ProbabilisticSurrogate:
        """Call the factory on the current training set.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        self.fits += 1
        return self._checked(self.factory(self.training, self.values, previous))

    def _predict(self, point: Vector) -> tuple[float, float]:
        """Return a validated ``(mean, variance)`` with the variance floored at 0.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated beyond the surrogate's.

        """
        mean, variance = self.surrogate.predict_with_variance(point)
        mean, variance = float(mean), float(variance)
        if not (math.isfinite(mean) and math.isfinite(variance)):
            raise ValueError("the surrogate returned a non-finite prediction")
        return mean, max(variance, 0.0)

    @property
    def population(self) -> list[Vector]:
        """Return the candidate points.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        return self._population

    @property
    def means(self) -> list[float]:
        """Return the predictive means over the population.

        References:
            [EQ-SUR-GP-003].

        """
        return self.mu

    @property
    def variances(self) -> list[float]:
        """Return the predictive variances over the population.

        References:
            [EQ-SUR-GP-003].

        """
        return self.var

    @property
    def training_points(self) -> list[Vector]:
        """Return the training inputs.

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011]. No equation is evaluated.

        """
        return self.training

    def add_training(self, point: Sequence[float], value: float) -> None:
        """Append one observation, refit, and re-predict the population.

        References:
            [EQ-SUR-GP-003], [REF-ECHARD-GAYTON-LEMAIRE-2011].

        """
        self.training.append(tuple(float(c) for c in point))
        self.values.append(float(value))
        self.surrogate = self._refit(self.surrogate)
        for index, x in enumerate(self._population):
            self.mu[index], self.var[index] = self._predict(x)

    def add_population(self, points: Sequence[Sequence[float]]) -> None:
        """Append candidate points with their predictions.

        References:
            [EQ-SUR-GP-003].

        """
        for raw in points:
            x = tuple(float(c) for c in raw)
            mean, variance = self._predict(x)
            self._population.append(x)
            self.mu.append(mean)
            self.var.append(variance)

    def replace_population(self, points: Sequence[Sequence[float]]) -> None:
        """Drop the candidates and predict at ``points``, keeping the training set.

        References:
            [EQ-SUR-GP-003].

        """
        self._population.clear()
        self.mu.clear()
        self.var.clear()
        self.add_population(points)
