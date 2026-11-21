"""Double/debiased machine learning with cross-fitting.

**Partially linear model.** ``Y = theta D + g(X) + e`` with
``E[e | D, X] = 0`` and ``D = m(X) + v``. With ``l(X) = E[Y | X]`` the
residuals ``Y~ = Y - l(X)`` and ``D~ = D - m(X)`` satisfy
``Y~ = theta D~ + e``, and the "partialling-out" score
``psi = (Y~ - theta D~) D~`` is Neyman orthogonal: its derivative with
respect to the nuisance functions ``(l, m)`` vanishes at the truth, so
first-order errors of the fitted nuisances do not enter ``theta_hat``
(EQ-CAUSAL-DML-001). The nuisances are fitted by user-supplied learners
with K-fold cross-fitting: each unit's residual uses learners trained on
the other folds, which removes the own-observation overfitting bias.

**Interactive model.** For a binary treatment the cross-fitted augmented
IPW score (EQ-CAUSAL-DML-003) estimates the average treatment effect
without the constant-effect restriction of the partially linear model.

Assumptions: unconfoundedness given ``X`` (no unmeasured confounding),
overlap, i.i.d. units, and nuisance estimators whose errors converge fast
enough (the product of the two root-mean-square errors is
``o(n^(-1/2))``; ``n^(-1/4)`` each suffices). The standard errors are
asymptotic; nothing here guarantees finite-sample coverage, and a learner
that converges too slowly (for example nearest neighbours in many
dimensions) leaves a bias the interval does not show.

A *learner* is any callable ``fit(rows, targets) -> predict`` where
``predict(rows)`` returns one prediction per row. :func:`linear_learner`,
:func:`ridge_learner`, :func:`knn_learner` and :func:`logistic_learner`
build learners from library routines.
"""

from __future__ import annotations

import heapq
import math
import statistics
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import fit_glm, fit_linear_model, fit_ridge
from probstat.linalg import Matrix, Vector, as_matrix, as_vector

from .effects import EffectEstimate, PositivityWarning, _binary_treatment

Predictor = Callable[[Matrix], Sequence[float]]
Learner = Callable[[Matrix, Vector], Predictor]


def _wald_interval(
    estimate: float, standard_error: float, level: float
) -> tuple[float, float]:
    """Return the normal-theory interval ``estimate +- z se``.

    Args:
        estimate: Point estimate.
        standard_error: Finite standard error.
        level: Two-sided level in ``(0, 1)``.

    Returns:
        Lower and upper bounds.

    Raises:
        ValueError: If the level is invalid or the standard error is not
            finite.

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018] (asymptotic normality of the
        estimator). No equation is evaluated.

    """
    if isinstance(level, bool) or not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    if not math.isfinite(standard_error):
        raise ValueError("no finite standard error is available")
    z = NormalDist().inv_cdf(0.5 * (1.0 + level))
    return estimate - z * standard_error, estimate + z * standard_error


def linear_learner() -> Learner:
    """Return a learner that fits least squares with an intercept.

    Returns:
        Learner ``fit(rows, targets) -> predict`` based on
        :func:`probstat.inference.fit_linear_model`; fitting raises if a
        training fold has no more rows than coefficients or is collinear.

    Equation:
        [EQ-LM-OLS-001].

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

    """

    def fit(rows: Matrix, targets: Vector) -> Predictor:
        """Fit ``targets ~ 1 + rows`` by least squares.

        References:
            [EQ-LM-OLS-001].

        """
        model = fit_linear_model(tuple((1.0, *row) for row in rows), targets)

        def predict(new_rows: Matrix) -> Sequence[float]:
            """Predict with the fitted coefficients.

            References:
                [EQ-LM-PRED-001].

            """
            return model.predict(tuple((1.0, *row) for row in new_rows))

        return predict

    return fit


def ridge_learner(penalty: float) -> Learner:
    """Return a learner that fits ridge regression with a fixed penalty.

    Args:
        penalty: Ridge penalty ``lambda >= 0`` on standardized predictors
            (see :func:`probstat.inference.fit_ridge`).

    Returns:
        Learner based on :func:`probstat.inference.fit_ridge` with an
        unpenalized intercept.

    Raises:
        TypeError: If the penalty is not real.
        ValueError: If the penalty is negative.

    Equation:
        [EQ-PEN-RIDGE-001].

    References:
        [REF-HOERL-KENNARD-1970].

    """
    lam = require_finite_real(penalty, name="penalty")
    if lam < 0.0:
        raise ValueError("penalty must be non-negative")

    def fit(rows: Matrix, targets: Vector) -> Predictor:
        """Fit ridge regression on one training fold.

        References:
            [EQ-PEN-RIDGE-001].

        """
        model = fit_ridge(rows, targets, penalty=lam)
        return model.predict

    return fit


def knn_learner(neighbours: int, *, standardize: bool = True) -> Learner:
    """Return a k-nearest-neighbour regression learner.

    The prediction at ``x`` is the mean target of the ``k`` training rows
    closest to ``x`` in Euclidean distance (ties broken by training
    order). With ``standardize`` each column is divided by its training
    standard deviation first (constant columns are left alone). The search
    is exhaustive: ``O(n_train)`` distances per prediction.

    Args:
        neighbours: ``k >= 1``; a training fold must have at least ``k``
            rows.
        standardize: Scale columns by their training standard deviation.

    Returns:
        Learner ``fit(rows, targets) -> predict``.

    Raises:
        ValueError: If ``neighbours`` is not a positive integer.

    Equation:
        [EQ-CAUSAL-KNN-001].

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 2.3].

    """
    if isinstance(neighbours, bool) or not isinstance(neighbours, int):
        raise ValueError("neighbours must be a positive integer")
    if neighbours < 1:
        raise ValueError("neighbours must be a positive integer")
    k = neighbours

    def fit(rows: Matrix, targets: Vector) -> Predictor:
        """Store the (scaled) training rows and targets.

        References:
            [EQ-CAUSAL-KNN-001].

        """
        if len(rows) < k:
            raise ValueError(
                f"knn_learner needs at least {k} training rows, got {len(rows)}"
            )
        width = len(rows[0])
        scales = [1.0] * width
        if standardize and len(rows) > 1:
            for j in range(width):
                spread = statistics.pstdev(row[j] for row in rows)
                if spread > 0.0:
                    scales[j] = spread
        train = [[v / s for v, s in zip(row, scales, strict=True)] for row in rows]
        single = [row[0] for row in train] if width == 1 else None

        def predict(new_rows: Matrix) -> Sequence[float]:
            """Average the targets of the ``k`` nearest training rows.

            References:
                [EQ-CAUSAL-KNN-001].

            """
            out: list[float] = []
            for row in new_rows:
                if single is not None:
                    x0 = row[0] / scales[0]
                    distances = [abs(v - x0) for v in single]
                else:
                    point = [v / s for v, s in zip(row, scales, strict=True)]
                    distances = [math.dist(point, other) for other in train]
                nearest = heapq.nsmallest(
                    k, zip(distances, range(len(train)), strict=True)
                )
                out.append(math.fsum(targets[i] for _, i in nearest) / k)
            return out

        return predict

    return fit


def logistic_learner() -> Learner:
    """Return a learner that fits a logistic regression with an intercept.

    Intended for the propensity score of
    :func:`dml_average_treatment_effect`; targets must be 0 or 1.

    Returns:
        Learner based on :func:`probstat.inference.fit_glm` (binomial
        family, logit link) predicting probabilities; fitting raises
        ``SeparationError`` when a training fold is separated.

    Equation:
        [EQ-CAUSAL-PS-001], [EQ-GLM-IRLS-001].

    References:
        [REF-ROSENBAUM-RUBIN-1983], [REF-MCCULLAGH-NELDER-1989, Chapter 4].

    """

    def fit(rows: Matrix, targets: Vector) -> Predictor:
        """Fit ``logit P(target = 1) = b0 + rows'b`` by IRLS.

        References:
            [EQ-CAUSAL-PS-001].

        """
        model = fit_glm(tuple((1.0, *row) for row in rows), targets, family="binomial")

        def predict(new_rows: Matrix) -> Sequence[float]:
            """Predict probabilities for new rows.

            References:
                [EQ-CAUSAL-PS-001].

            """
            return model.predict(tuple((1.0, *row) for row in new_rows))

        return predict

    return fit


def _fold_assignment(
    size: int, folds: int, seed: int | RandomStream | None
) -> tuple[int, ...]:
    """Assign units to folds of nearly equal size at random.

    A random permutation from the child stream ``"folds"`` is dealt round
    robin, so fold sizes differ by at most one. ``folds == 1`` puts every
    unit in fold 0 (no cross-fitting) and uses no randomness.

    Args:
        size: Number of units.
        folds: Number of folds, between 1 and ``size``.
        seed: Seed or stream.

    Returns:
        Fold index of each unit.

    Raises:
        ValueError: If ``folds`` is not an integer in ``[1, size]``.

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018] (random K-fold partition). No
        equation is evaluated.

    """
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 1:
        raise ValueError("folds must be a positive integer")
    if folds > size:
        raise ValueError("folds must not exceed the number of observations")
    if folds == 1:
        return (0,) * size
    order = list(range(size))
    resolve_stream(seed).child("folds").generator().shuffle(order)
    assignment = [0] * size
    for position, unit in enumerate(order):
        assignment[unit] = position % folds
    return tuple(assignment)


def _predictions(predictor: Predictor, rows: Matrix, *, name: str) -> tuple[float, ...]:
    """Call a fitted predictor and validate its output.

    Args:
        predictor: Fitted predictor.
        rows: Rows to predict.
        name: Learner name for error messages.

    Returns:
        One finite prediction per row.

    Raises:
        ValueError: If the predictor returns the wrong number of values or
            a non-finite value.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    values = tuple(float(v) for v in predictor(rows))
    if len(values) != len(rows):
        raise ValueError(f"{name} returned {len(values)} predictions for {len(rows)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} returned a non-finite prediction")
    return values


def _cross_fit(
    learner: Learner,
    rows: Matrix,
    targets: Vector,
    assignment: tuple[int, ...],
    *,
    name: str,
    train_mask: Sequence[bool] | None = None,
) -> tuple[float, ...]:
    """Predict every unit from a learner trained without its own fold.

    With a single fold the learner is trained on, and predicts, all units
    (in-sample, no cross-fitting).

    Args:
        learner: Nuisance learner.
        rows: Covariate rows.
        targets: Regression targets.
        assignment: Fold index of each unit.
        name: Learner name for error messages.
        train_mask: Optional flags restricting the *training* units (used
            for per-arm outcome models); predictions cover all units.

    Returns:
        Out-of-fold predictions in the original order.

    Raises:
        ValueError: If a training set is empty or the predictor misbehaves.

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018] (cross-fitting). No equation is
        evaluated.

    """
    folds = max(assignment) + 1
    n = len(rows)
    out = [0.0] * n
    for fold in range(folds):
        test = [i for i in range(n) if assignment[i] == fold]
        train = [
            i
            for i in range(n)
            if (folds == 1 or assignment[i] != fold)
            and (train_mask is None or train_mask[i])
        ]
        if not train:
            raise ValueError(f"{name}: a training fold is empty")
        predictor = learner(
            tuple(rows[i] for i in train), tuple(targets[i] for i in train)
        )
        predicted = _predictions(predictor, tuple(rows[i] for i in test), name=name)
        for i, value in zip(test, predicted, strict=True):
            out[i] = value
    return tuple(out)


def _r_squared(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return ``1 - sum (y - yhat)^2 / sum (y - ybar)^2``.

    Args:
        targets: Observed values.
        predictions: Predictions (out of fold under cross-fitting, so the
            value can be negative).

    Returns:
        Predictive R-squared, or ``nan`` when the targets are constant.

    References:
        [EQ-LM-FIT-001].

    """
    mean = statistics.fmean(targets)
    total = math.fsum((y - mean) ** 2 for y in targets)
    if total == 0.0:
        return math.nan
    error = math.fsum((y - p) ** 2 for y, p in zip(targets, predictions, strict=True))
    return 1.0 - error / total


@dataclass(frozen=True, slots=True)
class DoubleMLResult:
    """Cross-fitted estimate of ``theta`` in the partially linear model.

    Attributes:
        estimate: ``theta_hat``.
        standard_error: Asymptotic (heteroskedasticity-robust sandwich)
            standard error.
        n_observations: Number of units.
        n_folds: Number of folds ``K``.
        cross_fitted: ``False`` when ``K = 1`` (in-sample nuisance fits;
            the orthogonal score is used but overfitting bias is not
            removed).
        outcome_residuals: ``Y - l_hat(X)``.
        treatment_residuals: ``D - m_hat(X)``.
        outcome_r_squared: Predictive R-squared of ``l_hat`` (out of fold).
        treatment_r_squared: Predictive R-squared of ``m_hat``.
        fold_assignment: Fold index of each unit.
        nuisance_fits: Number of learner fits (``2 K``), the cost.

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018].

    """

    estimate: float
    standard_error: float
    n_observations: int
    n_folds: int
    cross_fitted: bool
    outcome_residuals: Vector
    treatment_residuals: Vector
    outcome_r_squared: float
    treatment_r_squared: float
    fold_assignment: tuple[int, ...]
    nuisance_fits: int

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the asymptotic interval ``estimate +- z se``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds.

        Raises:
            ValueError: If the level is invalid.

        References:
            [EQ-CAUSAL-DML-002].

        """
        return _wald_interval(self.estimate, self.standard_error, level)


def dml_partially_linear(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
    *,
    outcome_learner: Learner | None = None,
    treatment_learner: Learner | None = None,
    folds: int = 5,
    seed: int | RandomStream | None = None,
) -> DoubleMLResult:
    """Estimate ``theta`` in ``Y = theta D + g(X) + e`` by double ML.

    Steps: (1) split the units into ``folds`` random folds; (2) for each
    fold fit ``l(X) = E[Y | X]`` and ``m(X) = E[D | X]`` on the other folds
    and form the residuals ``Y~ = Y - l_hat(X)``, ``D~ = D - m_hat(X)`` of
    the held-out units; (3) pool all residuals and solve the orthogonal
    score: ``theta_hat = sum(D~ Y~) / sum(D~^2)``; (4)
    ``se^2 = mean(D~^2 (Y~ - theta_hat D~)^2) / (n mean(D~^2)^2)``.

    The treatment may be continuous or binary. With ``folds=1`` the
    nuisances are fitted and evaluated on the full sample; with linear
    learners this reproduces the least-squares coefficient of ``D`` in the
    regression of ``Y`` on ``(1, D, X)`` exactly (Frisch-Waugh-Lovell).

    Args:
        covariates: ``n x k`` covariate rows ``X``.
        treatment: ``n`` treatment values ``D``.
        outcome: ``n`` outcomes ``Y``.
        outcome_learner: Learner for ``E[Y | X]``; default
            :func:`linear_learner`.
        treatment_learner: Learner for ``E[D | X]``; default
            :func:`linear_learner`.
        folds: Number of folds ``K`` (``1`` disables cross-fitting).
        seed: Seed or stream of the fold assignment; pass one for
            reproducibility.

    Returns:
        :class:`DoubleMLResult`. The estimate is consistent and the
        interval asymptotically valid under unconfoundedness, a constant
        (or, as a weighted average, heterogeneous) linear effect, and
        nuisance errors whose product is ``o(n^(-1/2))``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``folds`` is invalid, a learner
            misbehaves, or the treatment residuals are all zero (the
            treatment is perfectly predicted by the covariates).

    Equation:
        [EQ-CAUSAL-DML-001], [EQ-CAUSAL-DML-002].

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018].

    """
    rows = as_matrix(covariates, name="covariates")
    d = as_vector(treatment, name="treatment")
    y = as_vector(outcome, name="outcome")
    n = len(rows)
    if len(d) != n or len(y) != n:
        raise ValueError("treatment and outcome need one entry per observation")
    if n < 2:
        raise ValueError("at least two observations are required")
    assignment = _fold_assignment(n, folds, seed)
    learn_y = linear_learner() if outcome_learner is None else outcome_learner
    learn_d = linear_learner() if treatment_learner is None else treatment_learner
    l_hat = _cross_fit(learn_y, rows, y, assignment, name="outcome_learner")
    m_hat = _cross_fit(learn_d, rows, d, assignment, name="treatment_learner")
    y_res = tuple(a - b for a, b in zip(y, l_hat, strict=True))
    d_res = tuple(a - b for a, b in zip(d, m_hat, strict=True))
    denominator = math.fsum(v * v for v in d_res)
    if denominator <= 0.0:
        raise ValueError(
            "treatment residuals are all zero: the covariates predict the "
            "treatment perfectly, so theta is not identified"
        )
    theta = math.fsum(a * b for a, b in zip(d_res, y_res, strict=True)) / denominator
    meat = math.fsum(
        (dv * (yv - theta * dv)) ** 2 for dv, yv in zip(d_res, y_res, strict=True)
    )
    return DoubleMLResult(
        estimate=theta,
        standard_error=math.sqrt(meat) / denominator,
        n_observations=n,
        n_folds=folds,
        cross_fitted=folds > 1,
        outcome_residuals=y_res,
        treatment_residuals=d_res,
        outcome_r_squared=_r_squared(y, l_hat),
        treatment_r_squared=_r_squared(d, m_hat),
        fold_assignment=assignment,
        nuisance_fits=2 * folds,
    )


def dml_average_treatment_effect(
    covariates: Iterable[Iterable[object]],
    treatment: Iterable[object],
    outcome: Iterable[object],
    *,
    outcome_learner: Learner | None = None,
    propensity_learner: Learner | None = None,
    folds: int = 5,
    clip: float = 0.01,
    seed: int | RandomStream | None = None,
) -> EffectEstimate:
    """Estimate the ATE of a binary treatment by cross-fitted AIPW.

    For each fold the outcome learner is trained separately on the treated
    and on the control units of the other folds (``m_1``, ``m_0``) and the
    propensity learner on all units of the other folds (``e``). The
    held-out units receive the doubly robust score
    ``psi = m_1 - m_0 + T (Y - m_1) / e - (1 - T)(Y - m_0) / (1 - e)``;
    the estimate is ``mean(psi)`` and the standard error
    ``sd(psi) / sqrt(n)``. With ``folds=1``, linear outcome models and a
    logistic propensity model this is :func:`probstat.causal.estimate_aipw`.

    Args:
        covariates: ``n x k`` covariate rows.
        treatment: ``n`` binary indicators.
        outcome: ``n`` outcomes.
        outcome_learner: Learner for ``E[Y | T = t, X]``; default
            :func:`linear_learner`.
        propensity_learner: Learner for ``P(T = 1 | X)`` (any regression
            learner on the 0/1 indicator); default :func:`logistic_learner`.
        folds: Number of folds ``K`` (``1`` disables cross-fitting).
        clip: Predicted propensities are clipped to ``[clip, 1 - clip]``;
            in ``(0, 0.5)``.
        seed: Seed or stream of the fold assignment.

    Returns:
        :class:`~probstat.causal.EffectEstimate` with
        ``method="aipw-crossfit"`` (``"aipw-learners"`` when ``folds=1``)
        and an asymptotic standard error.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If inputs are invalid, a training fold lacks an arm,
            or a learner misbehaves.

    Warns:
        PositivityWarning: If a predicted propensity was clipped; clipping
            biases the estimate when overlap is truly poor.

    Equation:
        [EQ-CAUSAL-DML-003], [EQ-CAUSAL-AIPW-001].

    References:
        [REF-CHERNOZHUKOV-ET-AL-2018], [REF-ROBINS-ROTNITZKY-ZHAO-1994].

    """
    rows = as_matrix(covariates, name="covariates")
    t = _binary_treatment(treatment, len(rows))
    y = as_vector(outcome, name="outcome")
    n = len(rows)
    if len(y) != n:
        raise ValueError("outcome must have one entry per observation")
    bound = require_finite_real(clip, name="clip")
    if not 0.0 < bound < 0.5:
        raise ValueError("clip must lie strictly between 0 and 0.5")
    assignment = _fold_assignment(n, folds, seed)
    learn_y = linear_learner() if outcome_learner is None else outcome_learner
    learn_e = logistic_learner() if propensity_learner is None else propensity_learner
    treated = [ti == 1 for ti in t]
    control = [ti == 0 for ti in t]
    m1 = _cross_fit(
        learn_y, rows, y, assignment, name="outcome_learner", train_mask=treated
    )
    m0 = _cross_fit(
        learn_y, rows, y, assignment, name="outcome_learner", train_mask=control
    )
    raw = _cross_fit(
        learn_e,
        rows,
        tuple(float(ti) for ti in t),
        assignment,
        name="propensity_learner",
    )
    scores = tuple(min(max(e, bound), 1.0 - bound) for e in raw)
    clipped = sum(1 for a, b in zip(raw, scores, strict=True) if a != b)
    if clipped:
        warnings.warn(
            f"{clipped} predicted propensity score(s) clipped to "
            f"[{bound}, {1.0 - bound}]; overlap is doubtful",
            PositivityWarning,
            stacklevel=2,
        )
    treated_terms = [
        a + ti * (yi - a) / e for a, ti, yi, e in zip(m1, t, y, scores, strict=True)
    ]
    control_terms = [
        b + (1 - ti) * (yi - b) / (1.0 - e)
        for b, ti, yi, e in zip(m0, t, y, scores, strict=True)
    ]
    influence = [a - b for a, b in zip(treated_terms, control_terms, strict=True)]
    return EffectEstimate(
        method="aipw-crossfit" if folds > 1 else "aipw-learners",
        estimand="ATE",
        estimate=statistics.fmean(influence),
        standard_error=statistics.stdev(influence) / math.sqrt(n),
        treated_mean=statistics.fmean(treated_terms),
        control_mean=statistics.fmean(control_terms),
        n_used=n,
        n_trimmed=0,
        propensity_scores=scores,
    )
