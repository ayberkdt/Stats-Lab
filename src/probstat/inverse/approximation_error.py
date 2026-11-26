"""Bayesian approximation-error model for paired forward-model fidelities.

Experimental (added in 0.28).

A cheap ("coarse") forward map ``G_c`` replaces an expensive ("fine") map
``G_f``. Kaipio and Somersalo model the discrepancy
``e(x) = G_f(x) - G_c(x)`` as a random vector with a Gaussian law estimated
from paired evaluations at parameter samples drawn from the prior, and
absorb it into the likelihood: ``y = G_c(x) + e + eps`` with
``e ~ N(mu_e, Sigma_e)`` and ``eps ~ N(0, Gamma)``, i.e. the coarse model
with data shift ``y - mu_e`` and noise covariance ``Gamma + Sigma_e``.

Assumptions that the user must be able to defend:

* ``e`` is independent of the noise ``eps`` (so the covariances add);
* in the basic model ``e`` is independent of ``x`` (its mean and covariance
  do not depend on the parameter). The cross-covariance ``Cov(e, x)`` is
  estimated on request as a diagnostic of this assumption;
* the model is trained on the *prior*; the posterior may concentrate where
  the error behaves differently, so a prior-trained model can be wrong in
  the region that matters. Compare prior-predictive and posterior errors;
* the held-out evaluation must never be used to tune and to score the same
  quantity: the validation part of the samples is excluded from every
  fitted quantity here.

A full sample covariance of an ``m``-vector needs ``K_train > m`` training
pairs to be positive definite; with fewer pairs a shrinkage policy is
mandatory and no jitter is added silently. Coverage improvements are not
guaranteed: they hold when the Gaussian additive assumptions are adequate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Matrix, Vector, as_matrix, as_vector, symmetric_eigen
from probstat.linalg.covariance_operators import DenseCovariance
from probstat.statistics import ledoit_wolf_covariance

__all__ = ["ApproximationErrorModel", "estimate_approximation_error"]

_QUANTILE_LEVEL = 0.95
_RANK_TOLERANCE_FACTOR = 1e-12


@dataclass(frozen=True, slots=True)
class ApproximationErrorModel:
    """Fitted Gaussian model of the paired error ``e = G_fine - G_coarse``.

    Args:
        mean: Sample mean ``mu_e`` of the training errors.
        covariance: ``Sigma_e`` (possibly shrunk), a symmetric positive
            definite ``m x m`` matrix accepted by ``DenseCovariance``.
        parameter_mean: Sample mean of the training parameters.
        cross_covariance: ``Cov(e, x)`` (``m x n``, divisor ``K - 1``) or
            ``None`` if not requested.
        shrinkage_policy: ``"none"``, ``"ledoit_wolf"`` or ``"fixed"``.
        shrinkage_intensity: Intensity ``lambda`` actually applied (0 for
            no shrinkage).
        training_size: Number ``K_train`` of pairs used for the fit.
        validation_size: Number of held-out pairs (0 if none).
        training_indices: Sample indices used to fit.
        validation_indices: Held-out sample indices, disjoint from the
            training indices.
        sample_rank: Numerical rank of the unshrunk sample covariance.
        validation_mahalanobis_mean: Mean of ``(e - mu_e)' Sigma_e^-1
            (e - mu_e)`` over held-out pairs (``m`` if the Gaussian model is
            right) or ``None`` without held-out data.
        validation_exceedance_fraction: Fraction of held-out Mahalanobis
            statistics above the chi-square 95% quantile with ``m`` degrees
            of freedom (about 0.05 if the model is right) or ``None``.
        chi_square_quantile: That 95% quantile.
        fine_evaluations: Number of fine-model calls made.
        coarse_evaluations: Number of coarse-model calls made.

    References:
        [EQ-INV-AE-001], [EQ-INV-AE-002], [EQ-INV-AE-003], [EQ-INV-AE-004],
        [REF-KAIPIO-SOMERSALO-2007].

    """

    mean: Vector
    covariance: Matrix
    parameter_mean: Vector
    cross_covariance: Matrix | None
    shrinkage_policy: str
    shrinkage_intensity: float
    training_size: int
    validation_size: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    sample_rank: int
    validation_mahalanobis_mean: float | None
    validation_exceedance_fraction: float | None
    chi_square_quantile: float
    fine_evaluations: int
    coarse_evaluations: int

    def error_covariance(self) -> DenseCovariance:
        """Return ``Sigma_e`` as a dense covariance operator.

        Returns:
            ``DenseCovariance(Sigma_e)``.

        References:
            [EQ-INV-AE-001].

        """
        return DenseCovariance(self.covariance)

    def combined_noise(
        self, noise_covariance: Sequence[Sequence[float]]
    ) -> DenseCovariance:
        """Return ``Gamma + Sigma_e`` assuming ``e`` independent of the noise.

        Args:
            noise_covariance: Dense symmetric ``m x m`` noise covariance
                ``Gamma``.

        Returns:
            ``DenseCovariance(Gamma + Sigma_e)``.

        Raises:
            ValueError: If the size differs from ``m`` or the sum is not
                symmetric positive definite.

        References:
            [EQ-INV-AE-003], [REF-KAIPIO-SOMERSALO-2007].

        """
        gamma = as_matrix(noise_covariance, name="noise_covariance")
        m = len(self.mean)
        if len(gamma) != m or any(len(row) != m for row in gamma):
            raise ValueError(f"noise_covariance must be {m} x {m}")
        total = [
            [gamma[i][j] + self.covariance[i][j] for j in range(m)] for i in range(m)
        ]
        return DenseCovariance(total)

    def corrected_data(self, data: Sequence[float]) -> Vector:
        """Return ``y - mu_e``, the data seen by the coarse model.

        Args:
            data: Observation ``y`` of length ``m``.

        Returns:
            ``y - mu_e``.

        Raises:
            ValueError: If the length differs from ``m`` or an entry is not
                finite.

        References:
            [EQ-INV-AE-003].

        """
        y = as_vector(data, name="data")
        if len(y) != len(self.mean):
            raise ValueError(f"data must have length {len(self.mean)}")
        return tuple(a - b for a, b in zip(y, self.mean, strict=True))


def _evaluate(
    function: Callable[[Sequence[float]], Sequence[float]],
    point: Vector,
    name: str,
) -> Vector:
    """Call a model and validate its output.

    Args:
        function: Forward map.
        point: Parameter vector.
        name: Label for error messages.

    Returns:
        The finite output vector.

    Raises:
        ValueError: If the output is empty, non-finite or not a vector.

    References:
        [EQ-INV-AE-001].

    """
    try:
        return as_vector(function(point), name=f"{name} output")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} model returned an invalid output: {error}") from error


def _sample_covariance(rows: list[Vector], mean: Vector) -> Matrix:
    """Return the sample covariance with divisor ``K - 1``.

    Args:
        rows: Observations (``K`` vectors).
        mean: Their mean.

    Returns:
        ``(1/(K-1)) sum (r - mean)(r - mean)'``.

    References:
        [EQ-INV-AE-001].

    """
    m = len(mean)
    k = len(rows)
    centred = [[r[i] - mean[i] for i in range(m)] for r in rows]
    scale = 1.0 / (k - 1)
    return tuple(
        tuple(scale * math.fsum(row[i] * row[j] for row in centred) for j in range(m))
        for i in range(m)
    )


def _numerical_rank(matrix: Matrix) -> int:
    """Return the number of eigenvalues above ``1e-12`` of the largest.

    Args:
        matrix: Symmetric positive semidefinite matrix.

    Returns:
        Numerical rank.

    References:
        [EQ-INV-AE-002].

    """
    values, _ = symmetric_eigen(matrix)
    top = max(abs(v) for v in values)
    if top == 0.0:
        return 0
    return sum(1 for v in values if v > _RANK_TOLERANCE_FACTOR * top)


def _resolve_shrinkage(
    shrinkage: str | float | None,
) -> tuple[str, float | None]:
    """Validate the shrinkage argument.

    Args:
        shrinkage: ``None``, ``"ledoit_wolf"`` or a number in ``[0, 1]``.

    Returns:
        ``(policy, fixed_intensity)``.

    Raises:
        TypeError: If the type is unsupported.
        ValueError: If the string is unknown or the number out of range.

    References:
        [EQ-INV-AE-002].

    """
    if shrinkage is None:
        return "none", None
    if isinstance(shrinkage, str):
        if shrinkage != "ledoit_wolf":
            raise ValueError("shrinkage must be None, 'ledoit_wolf' or a float")
        return "ledoit_wolf", None
    if isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float)):
        raise TypeError("shrinkage must be None, 'ledoit_wolf' or a float")
    value = float(shrinkage)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("a numeric shrinkage must lie in [0, 1]")
    return ("none", None) if value == 0.0 else ("fixed", value)


def estimate_approximation_error(
    fine: Callable[[Sequence[float]], Sequence[float]],
    coarse: Callable[[Sequence[float]], Sequence[float]],
    prior_samples: Sequence[Sequence[float]],
    *,
    shrinkage: str | float | None = None,
    validation_fraction: float = 0.0,
    cross_covariance: bool = False,
    seed: int | RandomStream | None = None,
) -> ApproximationErrorModel:
    """Estimate the Gaussian approximation-error model from paired runs.

    Both models are evaluated once at every sample ``x_k`` (drawn by the
    caller from the prior, independent of any data), ``e_k = G_f(x_k) -
    G_c(x_k)``. A random disjoint subset of ``round(validation_fraction K)``
    samples is held out; the mean ``mu_e`` and the covariance
    ``S = (1/(K-1)) sum (e_k - mu_e)(e_k - mu_e)'`` use the training part
    only. Shrinkage policies, with ``D = diag(S)``:

    * ``None`` (or ``0.0``): ``Sigma_e = S``; requires ``K_train > m``,
      otherwise ``S`` is singular and a ``ValueError`` is raised;
    * a number ``lambda`` in ``(0, 1]``: ``Sigma_e = (1 - lambda) S +
      lambda D`` (shrinkage towards the diagonal);
    * ``"ledoit_wolf"``: the intensity ``lambda`` of
      ``probstat.statistics.ledoit_wolf_covariance`` on the centred
      training errors, applied as ``Sigma_e = (1 - lambda) S + lambda
      (tr(S)/m) I`` (the target of the Ledoit-Wolf estimator; ``S`` here
      has divisor ``K - 1``, which rescales the library result by
      ``K/(K - 1)``).

    The held-out statistic is ``d_k = (e_k - mu_e)' Sigma_e^-1 (e_k -
    mu_e)``; under a correct Gaussian model it is approximately
    ``chi2(m)`` (mean ``m``, 5% above the 95% quantile; the training
    estimation error inflates it slightly for small ``K_train``).

    Args:
        fine: ``x -> G_f(x)`` (vector of length ``m``).
        coarse: ``x -> G_c(x)`` (same length).
        prior_samples: Parameter vectors drawn by the caller from the prior.
        shrinkage: See above.
        validation_fraction: Fraction in ``[0, 1)`` held out for validation
            (at least one held-out pair if positive).
        cross_covariance: Also estimate ``Cov(e, x)`` on the training part.
        seed: Seed or stream for the split.

    Returns:
        :class:`ApproximationErrorModel`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If fewer than two training pairs remain, a model output
            is non-finite or of inconsistent length, the sample covariance
            is singular without adequate shrinkage (``K_train <= m`` or rank
            deficiency), or the shrunk covariance is not positive definite.

    References:
        [EQ-INV-AE-001], [EQ-INV-AE-002], [EQ-INV-AE-004], [EQ-INV-AE-005],
        [REF-KAIPIO-SOMERSALO-2007], [REF-LEDOIT-WOLF-2004].

    """
    if not callable(fine) or not callable(coarse):
        raise TypeError("fine and coarse must be callable")
    policy, fixed = _resolve_shrinkage(shrinkage)
    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction, (int, float)
    ):
        raise TypeError("validation_fraction must be a real number")
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    points = [as_vector(p, name="prior sample") for p in prior_samples]
    count = len(points)
    if count < 3:
        raise ValueError("at least three prior samples are required")
    n = len(points[0])
    if any(len(p) != n for p in points):
        raise ValueError("prior samples must have a common length")

    errors: list[Vector] = []
    for point in points:
        fine_value = _evaluate(fine, point, "fine")
        coarse_value = _evaluate(coarse, point, "coarse")
        if len(fine_value) != len(coarse_value):
            raise ValueError("fine and coarse outputs must have the same length")
        errors.append(
            tuple(a - b for a, b in zip(fine_value, coarse_value, strict=True))
        )
    m = len(errors[0])
    if any(len(e) != m for e in errors):
        raise ValueError("model outputs must have a common length")

    order = list(range(count))
    n_val = 0
    if fraction > 0.0:
        resolve_stream(seed).child("split").generator().shuffle(order)
        n_val = max(1, round(fraction * count))
    validation_idx = tuple(sorted(order[:n_val]))
    training_idx = tuple(sorted(order[n_val:]))
    k = len(training_idx)
    if k < 2:
        raise ValueError("fewer than two training pairs remain")
    if policy == "none" and k <= m:
        raise ValueError(
            f"a full covariance of dimension {m} needs more than {m} training "
            f"pairs (got {k}); use shrinkage or more samples"
        )

    train_e = [errors[i] for i in training_idx]
    mean = tuple(math.fsum(e[i] for e in train_e) / k for i in range(m))
    sample = _sample_covariance(train_e, mean)
    rank = _numerical_rank(sample)
    if policy == "none" and rank < m:
        raise ValueError(
            f"sample covariance is rank deficient (rank {rank} < {m}); use shrinkage"
        )

    intensity = 0.0
    if policy == "fixed":
        assert fixed is not None
        intensity = fixed
        covariance: Matrix = tuple(
            tuple(
                (1.0 - intensity) * sample[i][j]
                + (intensity * sample[i][j] if i == j else 0.0)
                for j in range(m)
            )
            for i in range(m)
        )
    elif policy == "ledoit_wolf":
        intensity = ledoit_wolf_covariance(train_e).shrinkage
        target = math.fsum(sample[i][i] for i in range(m)) / m
        covariance = tuple(
            tuple(
                (1.0 - intensity) * sample[i][j]
                + (intensity * target if i == j else 0.0)
                for j in range(m)
            )
            for i in range(m)
        )
    else:
        covariance = sample
    try:
        model_cov = DenseCovariance(covariance)
    except ValueError as error:
        raise ValueError(
            "the estimated error covariance is not positive definite; "
            "increase shrinkage or the number of samples"
        ) from error

    mean_x: Vector = tuple(
        math.fsum(points[i][j] for i in training_idx) / k for j in range(n)
    )
    cross: Matrix | None = None
    if cross_covariance:
        cross = tuple(
            tuple(
                math.fsum(
                    (errors[idx][i] - mean[i]) * (points[idx][j] - mean_x[j])
                    for idx in training_idx
                )
                / (k - 1)
                for j in range(n)
            )
            for i in range(m)
        )

    quantile = ChiSquare(float(m)).quantile(_QUANTILE_LEVEL)
    mah_mean: float | None = None
    exceed: float | None = None
    if n_val > 0:
        stats = [
            model_cov.quadratic_form([errors[i][j] - mean[j] for j in range(m)])
            for i in validation_idx
        ]
        mah_mean = math.fsum(stats) / n_val
        exceed = sum(1 for s in stats if s > quantile) / n_val

    return ApproximationErrorModel(
        mean=mean,
        covariance=covariance,
        parameter_mean=mean_x,
        cross_covariance=cross,
        shrinkage_policy=policy,
        shrinkage_intensity=intensity,
        training_size=k,
        validation_size=n_val,
        training_indices=training_idx,
        validation_indices=validation_idx,
        sample_rank=rank,
        validation_mahalanobis_mean=mah_mean,
        validation_exceedance_fraction=exceed,
        chi_square_quantile=quantile,
        fine_evaluations=count,
        coarse_evaluations=count,
    )
