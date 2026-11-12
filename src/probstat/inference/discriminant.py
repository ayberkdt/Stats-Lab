"""Linear and quadratic discriminant analysis.

Gaussian class-conditional laws ``x | class k ~ N(mu_k, Sigma_k)`` with
prior probabilities ``pi_k`` give the Bayes classifier. A common covariance
gives linear discriminant functions (LDA, EQ-MVA-005) and Fisher's
canonical discriminant directions (EQ-MVA-007); class covariances give
quadratic ones (QDA, EQ-MVA-006).

References:
    [REF-FISHER-1936], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009],
    [REF-MARDIA-KENT-BIBBY-1979].

"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    cholesky,
    get_backend,
    log_det_spd,
    solve_lower,
    solve_spd,
    solve_upper,
    transpose,
)

__all__ = [
    "LinearDiscriminant",
    "QuadraticDiscriminant",
    "fit_linear_discriminant",
    "fit_quadratic_discriminant",
]


def _labelled_rows(
    data: Iterable[Iterable[object]], labels: Iterable[Hashable]
) -> tuple[tuple[Vector, ...], tuple[Hashable, ...]]:
    """Validate rows and labels of equal length.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    rows = tuple(
        tuple(float(require_finite_real(v, name="data")) for v in row) for row in data
    )
    tags = tuple(labels)
    if not rows or len(rows) != len(tags):
        raise ValueError("data and labels must be non-empty and of equal length")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("rows must be non-empty and of equal length")
    return rows, tags


def _classes_and_priors(
    tags: Sequence[Hashable], priors: Mapping[Hashable, float] | None
) -> tuple[tuple[Hashable, ...], Vector]:
    """Return the sorted classes and their prior probabilities.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3]. No equation is evaluated.

    """
    classes = tuple(sorted(set(tags), key=lambda c: (type(c).__name__, str(c))))
    if len(classes) < 2:
        raise ValueError("discriminant analysis needs at least two classes")
    if priors is None:
        values = tuple(tags.count(c) / len(tags) for c in classes)
    else:
        if set(priors) != set(classes):
            raise ValueError("priors must name every class exactly")
        values = tuple(
            float(require_finite_real(priors[c], name="priors")) for c in classes
        )
        if any(not v > 0.0 for v in values):
            raise ValueError("priors must be positive")
        total = math.fsum(values)
        values = tuple(v / total for v in values)
    return classes, values


def _class_moments(
    rows: Sequence[Vector], tags: Sequence[Hashable], cls: Hashable
) -> tuple[int, Vector, Matrix]:
    """Return the count, mean, and scatter matrix of one class.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. No named equation is evaluated.

    """
    members = [row for row, tag in zip(rows, tags, strict=True) if tag == cls]
    n, p = len(members), len(members[0])
    mean = tuple(math.fsum(r[j] for r in members) / n for j in range(p))
    centred = [[r[j] - mean[j] for j in range(p)] for r in members]
    scatter = tuple(
        tuple(math.fsum(c[i] * c[j] for c in centred) for j in range(p))
        for i in range(p)
    )
    return n, mean, scatter


def _softmax(scores: Sequence[float]) -> Vector:
    """Return ``exp(s_k) / sum_j exp(s_j)`` without overflow.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3]. No named equation
        is evaluated.

    """
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


def _new_rows(data: Iterable[Iterable[object]]) -> tuple[Vector, ...]:
    """Validate rows to be classified or projected.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    return tuple(
        tuple(require_finite_real(v, name="data") for v in row) for row in data
    )


def _check_width(rows: Sequence[Vector], width: int) -> None:
    """Reject rows of the wrong dimension.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if any(len(row) != width for row in rows):
        raise ValueError("rows must have one value per fitted variable")


@dataclass(frozen=True, slots=True)
class LinearDiscriminant:
    """Fitted linear discriminant analysis.

    Args:
        classes: Class labels in sorted order.
        priors: Prior probabilities ``pi_k``.
        means: Class means ``mu_k``.
        covariance: Pooled within-class covariance (divisor ``n - K``).
        coefficients: ``Sigma^-1 mu_k`` per class.
        constants: ``-mu_k' Sigma^-1 mu_k / 2 + log pi_k`` per class.
        directions: Fisher's canonical discriminant directions ``a_r``
            (rows), normalized to ``a' Sigma a = 1``.
        discriminant_ratios: Between- to within-class variance ratios
            ``a_r' B a_r`` in decreasing order.

    References:
        [EQ-MVA-005], [EQ-MVA-007], [REF-FISHER-1936],
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

    """

    classes: tuple[Hashable, ...]
    priors: Vector
    means: tuple[Vector, ...]
    covariance: Matrix
    coefficients: tuple[Vector, ...]
    constants: Vector
    directions: tuple[Vector, ...]
    discriminant_ratios: Vector

    def decision_function(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Return the linear discriminant scores of each row.

        ``delta_k(x) = x' Sigma^-1 mu_k - mu_k' Sigma^-1 mu_k / 2 + log pi_k``.

        References:
            [EQ-MVA-005], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        rows = _new_rows(data)
        _check_width(rows, len(self.covariance))
        return tuple(
            tuple(
                math.fsum(b * x for b, x in zip(beta, row, strict=True)) + c
                for beta, c in zip(self.coefficients, self.constants, strict=True)
            )
            for row in rows
        )

    def predict_proba(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Return posterior class probabilities, one row per point.

        References:
            [EQ-MVA-005], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        return tuple(_softmax(scores) for scores in self.decision_function(data))

    def predict(self, data: Iterable[Iterable[object]]) -> tuple[Hashable, ...]:
        """Return the class of largest posterior probability.

        References:
            [EQ-MVA-005], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        return tuple(
            self.classes[max(range(len(scores)), key=lambda k: scores[k])]
            for scores in self.decision_function(data)
        )

    def transform(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Project rows on the canonical discriminant directions.

        References:
            [EQ-MVA-007], [REF-FISHER-1936].

        """
        rows = _new_rows(data)
        _check_width(rows, len(self.covariance))
        return tuple(
            tuple(
                math.fsum(a * x for a, x in zip(direction, row, strict=True))
                for direction in self.directions
            )
            for row in rows
        )


def fit_linear_discriminant(
    data: Iterable[Iterable[object]],
    labels: Iterable[Hashable],
    *,
    priors: Mapping[Hashable, float] | None = None,
) -> LinearDiscriminant:
    """Fit linear discriminant analysis with a pooled covariance.

    With ``S_W = sum_k sum_(i in k) (x_i - mu_k)(x_i - mu_k)' / (n - K)``
    the Bayes rule for Gaussian classes with a common covariance assigns
    ``x`` to the class of largest ``delta_k(x)`` (EQ-MVA-005). Fisher's
    directions maximize ``a' B a / a' S_W a`` for the between-class
    covariance ``B = sum_k pi_k (mu_k - m)(mu_k - m)'``, ``m = sum_k pi_k
    mu_k``; with ``S_W = L L'`` they are ``L^-T`` times the eigenvectors of
    ``L^-1 B L^-T`` (EQ-MVA-007), at most ``min(K - 1, p)`` of them.

    Args:
        data: Rows of observations.
        labels: Class label of each row.
        priors: Prior probability per class (default: class proportions).

    Returns:
        The fitted discriminant.

    Raises:
        ValueError: For fewer than two classes, invalid priors, or a
            singular pooled covariance (``n - K`` must exceed ``p``).

    Equation:
        [EQ-MVA-005], [EQ-MVA-007].

    References:
        [REF-FISHER-1936], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

    """
    rows, tags = _labelled_rows(data, labels)
    classes, prior = _classes_and_priors(tags, priors)
    p, n, k = len(rows[0]), len(rows), len(classes)
    if n - k < p:
        raise ValueError("the pooled covariance needs n - K >= p observations")
    moments = [_class_moments(rows, tags, c) for c in classes]
    covariance = tuple(
        tuple(math.fsum(m[2][i][j] for m in moments) / (n - k) for j in range(p))
        for i in range(p)
    )
    try:
        lower = cholesky(covariance)
    except ValueError as error:
        raise ValueError("the pooled within-class covariance is singular") from error
    means = tuple(m[1] for m in moments)
    coefficients = tuple(solve_spd(covariance, mu) for mu in means)
    constants = tuple(
        -0.5 * math.fsum(b * x for b, x in zip(beta, mu, strict=True)) + math.log(pi)
        for beta, mu, pi in zip(coefficients, means, prior, strict=True)
    )
    centre = tuple(
        math.fsum(pi * mu[j] for pi, mu in zip(prior, means, strict=True))
        for j in range(p)
    )
    between = tuple(
        tuple(
            math.fsum(
                pi * (mu[i] - centre[i]) * (mu[j] - centre[j])
                for pi, mu in zip(prior, means, strict=True)
            )
            for j in range(p)
        )
        for i in range(p)
    )
    # S_W = L L' and C = L^-1 B L^-T: the columns of M = L^-1 B are L^-1 b_j
    # (B is symmetric), and the columns of C = L^-1 M' are L^-1 (row i of M).
    m_columns = [solve_lower(lower, between[j]) for j in range(p)]
    c_columns = [
        solve_lower(lower, tuple(m_columns[j][i] for j in range(p))) for i in range(p)
    ]
    symmetric = tuple(
        tuple(0.5 * (c_columns[j][i] + c_columns[i][j]) for j in range(p))
        for i in range(p)
    )
    values, vectors = get_backend().symmetric_eigen(symmetric)
    order = list(range(p - 1, -1, -1))[: min(k - 1, p)]
    upper = transpose(lower)
    directions: list[Vector] = []
    for index in order:
        eigenvector = tuple(vectors[r][index] for r in range(p))
        direction = solve_upper(upper, eigenvector)
        pivot = max(range(p), key=lambda j: abs(direction[j]))
        sign = -1.0 if direction[pivot] < 0.0 else 1.0
        directions.append(tuple(sign * v for v in direction))
    return LinearDiscriminant(
        classes=classes,
        priors=prior,
        means=means,
        covariance=covariance,
        coefficients=coefficients,
        constants=constants,
        directions=tuple(directions),
        discriminant_ratios=tuple(max(values[i], 0.0) for i in order),
    )


@dataclass(frozen=True, slots=True)
class QuadraticDiscriminant:
    """Fitted quadratic discriminant analysis.

    Args:
        classes: Class labels in sorted order.
        priors: Prior probabilities ``pi_k``.
        means: Class means ``mu_k``.
        covariances: Class covariances ``Sigma_k`` (divisor ``n_k - 1``).
        log_determinants: ``log det Sigma_k``.

    References:
        [EQ-MVA-006], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

    """

    classes: tuple[Hashable, ...]
    priors: Vector
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    log_determinants: Vector

    def decision_function(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Return the quadratic discriminant scores of each row.

        ``delta_k(x) = -log|Sigma_k|/2 - (x - mu_k)' Sigma_k^-1 (x - mu_k)/2
        + log pi_k``.

        References:
            [EQ-MVA-006], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        rows = _new_rows(data)
        _check_width(rows, len(self.means[0]))
        out: list[Vector] = []
        for row in rows:
            scores: list[float] = []
            for mu, sigma, logdet, pi in zip(
                self.means,
                self.covariances,
                self.log_determinants,
                self.priors,
                strict=True,
            ):
                diff = tuple(x - m for x, m in zip(row, mu, strict=True))
                solved = solve_spd(sigma, diff)
                quad = math.fsum(a * b for a, b in zip(diff, solved, strict=True))
                scores.append(-0.5 * logdet - 0.5 * quad + math.log(pi))
            out.append(tuple(scores))
        return tuple(out)

    def predict_proba(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Return posterior class probabilities, one row per point.

        References:
            [EQ-MVA-006], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        return tuple(_softmax(scores) for scores in self.decision_function(data))

    def predict(self, data: Iterable[Iterable[object]]) -> tuple[Hashable, ...]:
        """Return the class of largest posterior probability.

        References:
            [EQ-MVA-006], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

        """
        return tuple(
            self.classes[max(range(len(scores)), key=lambda k: scores[k])]
            for scores in self.decision_function(data)
        )


def fit_quadratic_discriminant(
    data: Iterable[Iterable[object]],
    labels: Iterable[Hashable],
    *,
    priors: Mapping[Hashable, float] | None = None,
) -> QuadraticDiscriminant:
    """Fit quadratic discriminant analysis with one covariance per class.

    The Bayes rule for Gaussian classes with unequal covariances assigns
    ``x`` to the class of largest quadratic discriminant ``delta_k(x)``
    (EQ-MVA-006); each ``Sigma_k`` is the unbiased class covariance.

    Args:
        data: Rows of observations.
        labels: Class label of each row.
        priors: Prior probability per class (default: class proportions).

    Returns:
        The fitted discriminant.

    Raises:
        ValueError: For fewer than two classes, invalid priors, or a
            singular class covariance (each class needs more than ``p``
            observations).

    Equation:
        [EQ-MVA-006].

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 4.3].

    """
    rows, tags = _labelled_rows(data, labels)
    classes, prior = _classes_and_priors(tags, priors)
    p = len(rows[0])
    means: list[Vector] = []
    covariances: list[Matrix] = []
    log_dets: list[float] = []
    for cls in classes:
        n_k, mu, scatter = _class_moments(rows, tags, cls)
        if n_k <= p:
            raise ValueError(f"class {cls!r} needs more than {p} observations")
        sigma = tuple(tuple(v / (n_k - 1) for v in row) for row in scatter)
        try:
            log_dets.append(log_det_spd(sigma))
        except ValueError as error:
            raise ValueError(f"the covariance of class {cls!r} is singular") from error
        means.append(mu)
        covariances.append(sigma)
    return QuadraticDiscriminant(
        classes=classes,
        priors=prior,
        means=tuple(means),
        covariances=tuple(covariances),
        log_determinants=tuple(log_dets),
    )
