"""Closed-form conjugate updates for categorical and multivariate normal data.

Two families are covered:

* the Dirichlet prior for categorical or multinomial probabilities, whose
  posterior is Dirichlet and whose posterior predictive law of future counts
  is Dirichlet-multinomial;
* the normal-inverse-Wishart prior for a multivariate normal with unknown
  mean and covariance, whose posterior is normal-inverse-Wishart and whose
  posterior predictive law of one new observation is multivariate Student-t.

The update functions are pure: they return frozen result records and never
modify the prior.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions._special import log_gamma_ratio
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector, as_vector, log_det_spd

from ._support import count_vector, positive_real, scaled, spd_matrix
from .dirichlet import Dirichlet
from .multinomial import DirichletMultinomial
from .normal import MultivariateNormal
from .student_t import MultivariateStudentT
from .wishart import InverseWishart, multivariate_log_gamma


@dataclass(frozen=True, slots=True)
class DirichletPosterior:
    """Result of a Dirichlet-categorical (multinomial) conjugate update.

    Args:
        prior: Dirichlet prior on the category probabilities.
        counts: Observed category counts.
        posterior: Dirichlet posterior ``Dir(alpha + counts)``.
        log_marginal_likelihood: Log prior-predictive probability of one
            particular observed sequence with these counts. For the
            unordered count vector add the log multinomial coefficient, which
            gives ``DirichletMultinomial(n, alpha).log_pmf(counts)``.

    References:
        [REF-GELMAN-ETAL-2013, Section 3.4], [REF-MURPHY-2012, Section 3.4].

    """

    prior: Dirichlet
    counts: tuple[int, ...]
    posterior: Dirichlet
    log_marginal_likelihood: float

    @property
    def total(self) -> int:
        """Return the number of observations ``n = sum_i counts_i``.

        References:
            [EQ-CONJ-DIR-001], [REF-GELMAN-ETAL-2013, Section 3.4].

        """
        return sum(self.counts)

    @property
    def predictive_probabilities(self) -> Vector:
        """Return ``P(next category = i | data) = (alpha_i + x_i) / (A + n)``.

        References:
            [EQ-CONJ-DIR-002], [REF-MURPHY-2012, Section 3.4].

        """
        return self.posterior.mean

    def predictive(self, trials: int) -> DirichletMultinomial:
        """Return the posterior predictive law of counts in ``trials`` new draws.

        Args:
            trials: Non-negative number of future categorical trials.

        Returns:
            ``DirichletMultinomial(trials, alpha + counts)``.

        Raises:
            TypeError: If ``trials`` is not an integer.
            ValueError: If ``trials`` is negative.

        Equation:
            [EQ-CONJ-DIR-002].

        References:
            [REF-MURPHY-2012, Section 3.4], [REF-MOSIMANN-1962].

        """
        return DirichletMultinomial(trials, self.posterior.alpha)


def dirichlet_posterior(prior: Dirichlet, counts: Iterable[int]) -> DirichletPosterior:
    """Update a Dirichlet prior with observed category counts.

    ``p | x ~ Dirichlet(alpha + x)``. The log marginal likelihood of one
    observed sequence is ``log Gamma(A) - log Gamma(A + n)
    + sum_i [log Gamma(alpha_i + x_i) - log Gamma(alpha_i)]``.

    Args:
        prior: Dirichlet prior with ``K`` categories.
        counts: ``K`` non-negative integer counts (integral floats accepted).

    Returns:
        The frozen update record.

    Raises:
        TypeError: If ``prior`` is not a :class:`Dirichlet` or a count is not
            integral.
        ValueError: If the number of counts differs from ``K`` or a count is
            negative.

    Equation:
        [EQ-CONJ-DIR-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.4], [REF-MURPHY-2012, Section 3.4],
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if not isinstance(prior, Dirichlet):
        raise TypeError("prior must be a Dirichlet")
    checked = count_vector(counts, name="counts", length=prior.dimension)
    if any(count < 0 for count in checked):
        raise ValueError("counts must be non-negative")
    terms: list[float] = []
    if sum(checked):
        terms.append(-log_gamma_ratio(prior.concentration, sum(checked)))
    for count, shape in zip(checked, prior.alpha, strict=True):
        if count:
            terms.append(log_gamma_ratio(shape, count))
    posterior = Dirichlet(
        tuple(shape + count for shape, count in zip(prior.alpha, checked, strict=True))
    )
    return DirichletPosterior(prior, checked, posterior, math.fsum(terms))


@dataclass(frozen=True, slots=True)
class NormalInverseWishart:
    """Normal-inverse-Wishart law of a mean vector and covariance matrix.

    ``Sigma ~ InverseWishart_p(Psi, nu)`` and
    ``mu | Sigma ~ N(m, Sigma / kappa)``.

    Args:
        location: Prior mean location ``m`` (non-empty, finite).
        kappa: Prior pseudo-sample size ``kappa > 0`` for the mean.
        scale: Symmetric positive-definite ``p x p`` scale ``Psi``.
        dof: Real degrees of freedom ``nu > p - 1``.

    References:
        [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"],
        [REF-GELMAN-ETAL-2013, Section 3.6], [REF-MURPHY-2012, Section 4.6.3].

    """

    location: Vector
    kappa: float
    scale: Matrix
    dof: float

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If dimensions disagree, ``kappa <= 0``, the scale is
                not symmetric positive definite, or ``dof <= p - 1``.

        References:
            [EQ-CONJ-NIW-001], [REF-GELMAN-ETAL-2013, Section 3.6].

        """
        location = as_vector(self.location, name="location")
        if not location:
            raise ValueError("location must contain at least one component")
        matrix = spd_matrix(self.scale, name="scale")[0]
        if len(matrix) != len(location):
            raise ValueError("scale order must equal the location dimension")
        kappa = positive_real(self.kappa, name="kappa")
        dof = InverseWishart(matrix, self.dof).dof
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "scale", matrix)
        object.__setattr__(self, "dof", dof)

    @property
    def dimension(self) -> int:
        """Return the vector dimension ``p``.

        References:
            [EQ-CONJ-NIW-001], [REF-GELMAN-ETAL-2013, Section 3.6].

        """
        return len(self.location)

    @property
    def covariance_marginal(self) -> InverseWishart:
        """Return the marginal law ``Sigma ~ InverseWishart_p(Psi, nu)``.

        References:
            [EQ-CONJ-NIW-001], [REF-GELMAN-ETAL-2013, Section 3.6].

        """
        return InverseWishart(self.scale, self.dof)

    @property
    def mean_marginal(self) -> MultivariateStudentT:
        """Return the marginal law of ``mu``, ``t_(nu-p+1)(m, Psi / (kappa (nu-p+1)))``.

        References:
            [EQ-CONJ-NIW-003], [REF-GELMAN-ETAL-2013, Section 3.6],
            [REF-MURPHY-2012, Section 4.6.3].

        """
        dof = self.dof - self.dimension + 1.0
        return MultivariateStudentT(
            self.location, scaled(self.scale, 1.0 / (self.kappa * dof)), dof
        )

    @property
    def predictive(self) -> MultivariateStudentT:
        """Return the predictive law of one new observation ``x ~ N(mu, Sigma)``.

        ``x ~ t_(nu-p+1)(m, Psi (kappa + 1) / (kappa (nu - p + 1)))``.

        References:
            [EQ-CONJ-NIW-003], [REF-GELMAN-ETAL-2013, Section 3.6],
            [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"].

        """
        dof = self.dof - self.dimension + 1.0
        factor = (self.kappa + 1.0) / (self.kappa * dof)
        return MultivariateStudentT(self.location, scaled(self.scale, factor), dof)

    def log_pdf(
        self, mean: Iterable[float], covariance: Iterable[Iterable[object]]
    ) -> float:
        """Evaluate the joint log-density of ``(mu, Sigma)``.

        ``log p(mu, Sigma) = log N(mu | m, Sigma / kappa)
        + log InverseWishart(Sigma | Psi, nu)``.

        Args:
            mean: Mean vector ``mu`` of dimension ``p``.
            covariance: Symmetric ``p x p`` covariance ``Sigma``.

        Returns:
            Joint log-density; ``-inf`` when ``Sigma`` is not positive
            definite.

        Raises:
            TypeError: If entries are not real.
            ValueError: If shapes disagree or ``covariance`` is not symmetric.

        Equation:
            [EQ-CONJ-NIW-001], [EQ-DIST-MVN-001], [EQ-DIST-IWISH-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 3.6],
            [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"].

        """
        matrix = tuple(tuple(row) for row in covariance)
        log_covariance = self.covariance_marginal.log_pdf(matrix)
        if log_covariance == -math.inf:
            return -math.inf
        symmetric = spd_matrix(matrix, name="covariance")[0]
        conditional = MultivariateNormal(
            self.location, scaled(symmetric, 1.0 / self.kappa)
        )
        return conditional.log_pdf(mean) + log_covariance

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[Vector, Matrix], ...]:
        """Draw reproducible ``(mu, Sigma)`` pairs.

        ``Sigma`` comes from the inverse-Wishart sampler on the child stream
        ``"covariance"``; ``mu | Sigma`` from the multivariate normal sampler
        on the child stream ``("mean", k)`` for draw ``k``.

        Args:
            size: Number of pairs, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of ``(mean, covariance)`` pairs.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-CONJ-NIW-001], [EQ-DIST-IWISH-003], [EQ-DIST-MVN-005].

        References:
            [REF-GELMAN-ETAL-2013, Section 3.6], [REF-SMITH-HOCKING-1972].

        """
        checked_size = require_non_negative_integer(size, name="size")
        stream = resolve_stream(seed)
        covariances = self.covariance_marginal.sample(
            checked_size, seed=stream.child("covariance")
        )
        pairs: list[tuple[Vector, Matrix]] = []
        for index, covariance in enumerate(covariances):
            conditional = MultivariateNormal(
                self.location, scaled(covariance, 1.0 / self.kappa)
            )
            mean = conditional.sample(1, seed=stream.child("mean", index))[0]
            pairs.append((mean, covariance))
        return tuple(pairs)


@dataclass(frozen=True, slots=True)
class NormalInverseWishartPosterior:
    """Result of a normal-inverse-Wishart conjugate update.

    Args:
        prior: Normal-inverse-Wishart prior.
        posterior: Normal-inverse-Wishart posterior.
        sample_size: Number of observations ``n``.
        sample_mean: Sample mean ``xbar``.
        scatter: Centred cross-product matrix ``S = sum (x_i - xbar)(x_i - xbar)'``.
        log_marginal_likelihood: Log prior-predictive density of the data.

    References:
        [REF-GELMAN-ETAL-2013, Section 3.6],
        [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"].

    """

    prior: NormalInverseWishart
    posterior: NormalInverseWishart
    sample_size: int
    sample_mean: Vector
    scatter: Matrix
    log_marginal_likelihood: float

    @property
    def predictive(self) -> MultivariateStudentT:
        """Return the posterior predictive law of one new observation.

        References:
            [EQ-CONJ-NIW-003], [REF-GELMAN-ETAL-2013, Section 3.6].

        """
        return self.posterior.predictive


def normal_inverse_wishart_posterior(
    prior: NormalInverseWishart, observations: Iterable[Iterable[float]]
) -> NormalInverseWishartPosterior:
    """Update a normal-inverse-Wishart prior with multivariate normal data.

    With ``n`` observations, sample mean ``xbar`` and scatter ``S``:
    ``kappa_n = kappa_0 + n``, ``nu_n = nu_0 + n``,
    ``m_n = (kappa_0 m_0 + n xbar) / kappa_n`` and
    ``Psi_n = Psi_0 + S + (kappa_0 n / kappa_n)(xbar - m_0)(xbar - m_0)'``.
    The log marginal likelihood is
    ``-(n p / 2) log pi + log Gamma_p(nu_n/2) - log Gamma_p(nu_0/2)
    + (nu_0/2) log|Psi_0| - (nu_n/2) log|Psi_n|
    + (p/2)(log kappa_0 - log kappa_n)``.

    Args:
        prior: Normal-inverse-Wishart prior of dimension ``p``.
        observations: At least one finite vector of dimension ``p``.

    Returns:
        The frozen update record.

    Raises:
        TypeError: If ``prior`` is not a :class:`NormalInverseWishart` or an
            observation is not an iterable of reals.
        ValueError: If there are no observations or an observation has the
            wrong dimension.

    Equation:
        [EQ-CONJ-NIW-002], [EQ-CONJ-NIW-004], [EQ-NUM-MVLGAMMA-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.6],
        [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"],
        [REF-MURPHY-2012, Section 4.6.3].

    """
    if not isinstance(prior, NormalInverseWishart):
        raise TypeError("prior must be a NormalInverseWishart")
    try:
        raw = tuple(observations)
    except TypeError as error:
        raise TypeError("observations must be an iterable of vectors") from error
    rows = tuple(
        as_vector(row, name=f"observations[{index}]") for index, row in enumerate(raw)
    )
    if not rows:
        raise ValueError("at least one observation is required")
    p = prior.dimension
    if any(len(row) != p for row in rows):
        raise ValueError("every observation must have the prior dimension")
    n = len(rows)
    sample_mean = tuple(math.fsum(row[j] for row in rows) / n for j in range(p))
    centred = tuple(
        tuple(row[j] - sample_mean[j] for j in range(p)) for row in rows
    )
    scatter = tuple(
        tuple(math.fsum(row[i] * row[j] for row in centred) for j in range(p))
        for i in range(p)
    )
    kappa_n = prior.kappa + n
    dof_n = prior.dof + n
    shift = tuple(sample_mean[j] - prior.location[j] for j in range(p))
    shrink = prior.kappa * n / kappa_n
    location_n = tuple(
        (prior.kappa * prior.location[j] + n * sample_mean[j]) / kappa_n
        for j in range(p)
    )
    scale_n = tuple(
        tuple(
            prior.scale[i][j] + scatter[i][j] + shrink * shift[i] * shift[j]
            for j in range(p)
        )
        for i in range(p)
    )
    posterior = NormalInverseWishart(location_n, kappa_n, scale_n, dof_n)
    log_evidence = math.fsum(
        (
            -0.5 * n * p * math.log(math.pi),
            multivariate_log_gamma(0.5 * dof_n, p),
            -multivariate_log_gamma(0.5 * prior.dof, p),
            0.5 * prior.dof * log_det_spd(prior.scale),
            -0.5 * dof_n * log_det_spd(posterior.scale),
            0.5 * p * (math.log(prior.kappa) - math.log(kappa_n)),
        )
    )
    return NormalInverseWishartPosterior(
        prior, posterior, n, sample_mean, scatter, log_evidence
    )
