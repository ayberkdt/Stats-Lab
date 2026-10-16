"""Distribution contract, new families, wrappers, and large-shape incomplete beta.

Reference values were computed in a separate verification environment and
pasted as literals (no numerical library is imported here):

* ``SCIPY``: scipy 1.18 ``stats`` distributions (pdf, logpdf, cdf, sf, ppf,
  isf, ``stats("mvsk")``, entropy), with mpmath 1.4 (40 digits) quadrature or
  bisection for truncated-gamma moments, mixture and KDE quantiles;
* ``BETA_GRID``: ``I_x(a, b)`` (``lower``) and ``I_(1-x)(b, a)``
  (``upper``) at ``x = 1 - y`` exactly, from the all-positive hypergeometric
  series ``y^b x^a / (b B(a, b)) sum (a+b)_n/(b+1)_n y^n`` at 80 digits, or a
  60-digit Lentz continued fraction for the deep lower tail, both in mpmath;
* ``STUDENT_T`` and ``FISHER_F``: the same mpmath evaluations of
  ``I_(nu/(nu+t^2))(nu/2, 1/2)/2`` and ``I_(d2/(d2+d1 f))(d2/2, d1/2)``;
* ``CONTINUOUS_SHAPE`` and ``DISCRETE_SHAPE``: scipy skewness, excess
  kurtosis, entropy and (discrete) ``isf``.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import (
    PERT,
    Affine,
    Beta,
    Binomial,
    Cauchy,
    ChiSquare,
    ContinuousDistribution,
    DiscreteDistribution,
    DiscreteUniform,
    Empirical,
    Exponential,
    FisherF,
    Gamma,
    Geometric,
    Gumbel,
    Hypergeometric,
    Laplace,
    Logistic,
    LogNormal,
    Mixture,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    Rayleigh,
    StudentT,
    Triangular,
    Truncated,
    TruncatedNormal,
    Uniform,
    Weibull,
)
from probstat.distributions._special import regularized_beta_pair
from probstat.foundations import RandomStream
from probstat.modeling import JointInput, RandomInput
from probstat.sampling import latin_hypercube
from probstat.statistics import gaussian_kde

SCIPY = {
    "affine_gamma": {
        "cdf": [
            0.0015505093491957362,
            0.30621891841327875,
            0.9130698145443954,
            0.9995605931889086,
        ],
        "entropy": 2.0176299819568353,
        "mvsk": [-2.333333333333333, 4.444444444444444, -1.2649110640673518, 2.4],
        "pdf": [
            0.0010012920976118264,
            0.1459565199889244,
            0.17309961315613717,
            0.010750901245913203,
        ],
        "ppf": [-36.041599012159, -3.042953322769936, -0.07353865797488135],
    },
    "cauchy": {
        "cdf": [
            6.366191357475968e-07,
            0.14758361765043326,
            0.5,
            0.7048327646991335,
            0.9836907030754379,
        ],
        "entropy": 3.224171427529236,
        "isf": [
            636619772368.5814,
            7.155367074350508,
            1.0,
            -5.155367074350509,
            -636619789.3724186,
        ],
        "logpdf": [
            -28.082605821221,
            -3.447314978843446,
            -1.8378770664093453,
            -2.284164169037765,
            -7.781332394330966,
        ],
        "mvsk": [math.nan, math.nan, math.nan, math.nan],
        "pdf": [
            6.366184991274e-13,
            0.03183098861837907,
            0.15915494309189535,
            0.10185916357881303,
            0.00041745558843775827,
        ],
        "ppf": [
            -636619772366.5814,
            -5.155367074350508,
            1.0,
            7.155367074350509,
            636619791.3724186,
        ],
        "sf": [
            0.9999993633808644,
            0.8524163823495667,
            0.5,
            0.2951672353008665,
            0.016309296924562074,
        ],
    },
    "dunif": {
        "cdf": [0.0, 0.125, 0.375, 0.75, 1.0],
        "entropy": 2.0794415416798357,
        "isf": [5.0, 5.0, 1.0, -2.0, -2.0],
        "mvsk": [1.5, 5.25, 0.0, -1.2380952380952381],
        "pmf": [0.0, 0.125, 0.125, 0.125, 0.125],
        "ppf": [-2.0, -2.0, 1.0, 4.0, 5.0],
        "sf": [1.0, 0.875, 0.625, 0.25, 0.0],
    },
    "empirical": {
        "q1": [-2.0, -2.0, 1.3, 2.2, 5.0, 7.5],
        "q7": [-2.0, -1.2799999999999998, 1.288, 2.2, 5.249999999999999, 7.5],
    },
    "hypergeom": {
        "cdf": [
            0.04602034214577739,
            0.23644934412830448,
            0.5506571973994743,
            0.992675161073836,
            1.0,
        ],
        "entropy": 1.6074959441662018,
        "isf": [6.0, 4.0, 2.0, 1.0, 0.0],
        "mvsk": [2.4, 1.4889795918367348, 0.26634159016854425, -0.12090419699514744],
        "pmf": [
            0.04602034214577739,
            0.1904290019825271,
            0.3142078532711697,
            0.038700087499674865,
            6.4250596515923595e-09,
        ],
        "ppf": [0.0, 1.0, 2.0, 4.0, 6.0],
        "sf": [
            0.9539796578542226,
            0.7635506558716956,
            0.44934280260052567,
            0.0073248389261639305,
            0.0,
        ],
    },
    "kde": {
        "ppf": [
            -6.198464917304053,
            -1.202545580790251,
            2.217718773182608,
            7.50031924598129,
            12.194218469766156,
        ]
    },
    "laplace": {
        "cdf": [
            3.23511746282273e-26,
            0.06766764161830635,
            0.5,
            0.9628632108928331,
            0.9999999999999937,
        ],
        "entropy": 1.0,
        "isf": [
            15.922814607175368,
            -0.5418546340629224,
            -1.0,
            -1.3465735902799727,
            -4.107304049211095,
        ],
        "logpdf": [-58.0, -2.0, 0.0, -2.6, -32.0],
        "mvsk": [-1.0, 0.5, 0.0, 3.0],
        "pdf": [
            6.47023492564546e-26,
            0.1353352832366127,
            1.0,
            0.07427357821433388,
            1.2664165549094176e-14,
        ],
        "ppf": [
            -17.92281460717537,
            -1.4581453659370776,
            -1.0,
            -0.6534264097200273,
            2.1073040492110953,
        ],
        "sf": [
            1.0,
            0.9323323583816936,
            0.5,
            0.03713678910716694,
            6.332082774547088e-15,
        ],
    },
    "mixture_normal": {
        "cdf": [
            9.501372549935977e-06,
            0.15000000069061134,
            0.28090276824148364,
            0.6495950305905109,
            0.9999777441352452,
        ],
        "isf_1e-14": 7.4942744847214815,
        "mean": 2.0999999999999996,
        "ppf": [
            -6.864623880145045,
            -0.9674215661017108,
            1.8835416020588736,
            2.7190866547520494,
            3.5339796748048213,
            5.998710626783774,
        ],
        "var": 2.3649999999999998,
    },
    "pareto": {
        "cdf": [0.0, 0.6336426225664344, 0.9558058261758408, 0.9992844582472, 1.0],
        "entropy": 0.41129200600589344,
        "isf": [
            199.99999999999994,
            2.721580000348754,
            2.333058079152233,
            2.022927440819229,
            2.0000000000444444,
        ],
        "logpdf": [
            0.8109302162163289,
            -0.41635931601182496,
            -3.0013792768633705,
            -8.040978302171222,
            -46.03363233657298,
        ],
        "mvsk": [
            2.5714285714285716,
            0.5877551020408164,
            5.465943944999486,
            146.44444444444446,
        ],
        "pdf": [
            2.25,
            0.6594432793804179,
            0.04971844555217913,
            0.0003219937887599697,
            1.0182337649086285e-20,
        ],
        "ppf": [
            2.0000000004444445,
            2.1320341627809487,
            2.333058079152233,
            3.8917754351527765,
            333.62010130582274,
        ],
        "sf": [
            1.0,
            0.36635737743356556,
            0.04419417382415922,
            0.0007155417527999327,
            2.262741699796952e-17,
        ],
    },
    "pert": {
        "cdf": [
            4.207887585479187e-09,
            0.11992733461001337,
            0.3481374328691094,
            0.8939657155746442,
            0.9999999999968467,
        ],
        "entropy": 1.8100508395182628,
        "isf": [
            9.928851542592637,
            6.064204993802896,
            3.638532776608698,
            1.8965595762306562,
            1.0061315978144414,
        ],
        "logpdf": [
            -9.439987903594435,
            -1.6194202268976445,
            -1.418720399009541,
            -2.345266644253829,
            -20.46388973654597,
        ],
        "mvsk": [
            3.8333333333333335,
            2.4960317460317456,
            0.5274650660152336,
            -0.2957074721780604,
        ],
        "pdf": [
            7.948136998362793e-05,
            0.19801346869328598,
            0.24202351236558972,
            0.09582164843182978,
            1.2961211848670389e-09,
        ],
        "ppf": [
            1.000158134960332,
            1.8965595762306564,
            3.638532776608698,
            6.064204993802898,
            9.615782133318236,
        ],
        "sf": [
            0.9999999957921124,
            0.8800726653899867,
            0.6518625671308905,
            0.1060342844253558,
            3.153337094416022e-12,
        ],
    },
    "rayleigh": {
        "cdf": [
            2.2221975310470962e-05,
            0.054040531093234534,
            0.3934693402873666,
            0.8646647167633873,
            0.9999999999999873,
        ],
        "entropy": 1.347499350278958,
        "isf": [
            10.179210636622667,
            2.327633480483281,
            1.7661150337732119,
            0.688565407539631,
            2.1212968798029437e-06,
        ],
        "logpdf": [
            -5.4161226244266425,
            -1.5596329523318297,
            -0.9054651081081644,
            -1.712317927548219,
            -30.32602356642833,
        ],
        "mvsk": [
            1.8799712059732503,
            0.9657082647114827,
            0.6311106578189364,
            0.24508930068763846,
        ],
        "pdf": [
            0.004444345680109731,
            0.21021321531261453,
            0.40435377314175563,
            0.1804470443154836,
            6.75422162618355e-14,
        ],
        "ppf": [
            2.1213203436126758e-05,
            1.2669006463508874,
            1.7661150337732119,
            3.2189490394340208,
            11.150771030315353,
        ],
        "sf": [
            0.9999777780246896,
            0.9459594689067654,
            0.6065306597126334,
            0.1353352832366127,
            1.2664165549094176e-14,
        ],
    },
    "trunc_gamma_upper": {
        "cdf": [
            0.003397200935604551,
            0.1252991581390339,
            0.4324334751752253,
            0.9563348203897293,
        ],
        "mean": 1.1056577303218378,
        "ppf": [
            0.0037129505114367337,
            0.7200873138110375,
            1.1015943946779785,
            1.9764418147480471,
        ],
        "var": 0.2353711481075258,
    },
    "trunc_lognormal_tail": {
        "cdf": [0.4985843246581474, 0.9307203386135501, 0.9990593072056556],
        "ppf": [5.000000710710761, 5.502084771731518, 8.89404698882161],
        "sf": [0.5014156753418526, 0.0692796613864499, 0.0009406927943443143],
    },
    "trunc_normal_lower": {
        "cdf": [
            0.0,
            0.17973974872821907,
            0.6594658731979527,
            0.9797941238969496,
            0.9999999852323159,
        ],
        "entropy": math.nan,
        "isf": [
            6.42300010243864,
            2.2155420634827037,
            1.8329688658931016,
            1.612323684358436,
            1.5000000005158156,
        ],
        "logpdf": [
            0.6620058676192171,
            0.5070058676192168,
            -0.2129941323807829,
            -2.7129941323807825,
            -16.212994132380786,
        ],
        "mvsk": [
            1.9386771666225433,
            0.1495465935502024,
            1.4378299224716806,
            2.5345540384904934,
        ],
        "pdf": [
            1.9386771666225433,
            1.660312549736474,
            0.8081608791828815,
            0.0663378846556157,
            9.094652574011539e-08,
        ],
        "ppf": [
            1.5000000005158156,
            1.612323684358436,
            1.8329688658931016,
            2.2155420634827037,
            6.42300010243864,
        ],
        "sf": [
            1.0,
            0.8202602512717809,
            0.34053412680204737,
            0.020205876103050344,
            1.4767684116376638e-08,
        ],
    },
}

BETA_GRID = [
    (10000.0, 0.5, 4.999750012499375e-08, 0.9747738247230048, 0.025226175276995277),
    (10000.0, 1000.0, 9.090909090909092e-05, 1.0, 0.0),
    (100000.0, 100.0, 9.99000999000999e-07, 1.0, 9.2283310843789e-259),
    (1000000.0, 2.5, 2.499993750015625e-09, 0.9999999061365632, 9.386343676305399e-08),
    (100000000.0, 0.5, 4.9999999750000005e-12, 0.9747728794644991, 0.02522712053550096),
    (100000000.0, 1000.0, 9.999900000999992e-09, 1.0, 0.0),
    (10000000000.0, 100.0, 9.999999900000001e-12, 1.0, 9.705029985784407e-259),
    (
        50000000000000.0,
        2.5,
        4.99999999999975e-17,
        0.9999999061361532,
        9.386384682947242e-08,
    ),
]

STUDENT_T = [
    (30.0, 0.5, 0.31036150244256366),
    (30.0, 3.0, 0.002694982032825973),
    (30.0, 8.0, 3.1329112378503795e-09),
    (30.0, -2.0, 0.9726874775185085),
    (1000.0, 0.5, 0.3085925404169374),
    (1000.0, 3.0, 0.0013833545221190963),
    (1000.0, 8.0, 1.7133307411957373e-15),
    (1000.0, -2.0, 0.9771148267533741),
    (1000000.0, 0.5, 0.30853759373618567),
    (1000000.0, 3.0, 0.0013499312707108985),
    (1000000.0, 8.0, 6.22753171660126e-16),
    (1000000.0, -2.0, 0.9772497330743404),
    (100000000.0, 0.5, 0.30853753927608896),
    (100000000.0, 3.0, 0.0013498983640187472),
    (100000000.0, 8.0, 6.22102625411477e-16),
    (100000000.0, -2.0, 0.9772498667020466),
    (10000000000.0, 0.5, 0.3085375387314879),
    (10000000000.0, 3.0, 0.0013498980349539809),
    (10000000000.0, 8.0, 6.220961231067057e-16),
    (10000000000.0, -2.0, 0.977249868038323),
    (1000000000000.0, 0.5, 0.3085375387260419),
    (1000000000000.0, 3.0, 0.0013498980316633334),
    (1000000000000.0, 8.0, 6.220960580839736e-16),
    (1000000000000.0, -2.0, 0.9772498680516858),
    (100000000000000.0, 0.5, 0.30853753872598744),
    (100000000000000.0, 3.0, 0.001349898031630427),
    (100000000000000.0, 8.0, 6.220960574337463e-16),
    (100000000000000.0, -2.0, 0.9772498680518195),
]

FISHER_F = [
    (5.0, 10000.0, 0.5, 0.7764856621550201, 0.22351433784497987),
    (5.0, 10000.0, 2.0, 0.0753344106469027, 0.9246655893530973),
    (5.0, 10000.0, 4.0, 0.0012589312164184255, 0.9987410687835816),
    (5.0, 1000000.0, 0.5, 0.7764949769974423, 0.22350502300255776),
    (5.0, 1000000.0, 2.0, 0.0752362378553057, 0.9247637621446942),
    (5.0, 1000000.0, 4.0, 0.0012498223640969206, 0.9987501776359031),
    (5.0, 10000000000.0, 0.5, 0.77649507111391, 0.22350492888608992),
    (5.0, 10000000000.0, 2.0, 0.07523524624568312, 0.9247647537543169),
    (5.0, 10000000000.0, 4.0, 0.0012497305722112746, 0.9987502694277888),
]

CONTINUOUS_SHAPE = {
    "beta": (0.28571428571428575, -0.6428571428571429, -0.2349066497880008),
    "chi2": (1.2649110640673518, 2.4, 2.423095090065),
    "exponential": (2.0, 6.0, 0.3068528194400547),
    "f": (3.079201435678005, 24.33333333333335, 1.0815894027424697),
    "gamma": (1.2649110640673518, 2.4, 1.3244828013968901),
    "gumbel": (1.1395470994046486, 2.4, 1.5772156649015328),
    "logistic": (0.0, 1.2, 2.0),
    "lognormal": (1.7501896550697178, 5.898445673784778, 0.7257913526447274),
    "normal": (0.0, 0.0, 2.112085713764618),
    "t5": (0.0, 6.0, 1.627502672414396),
    "triangular": (0.4224039833745502, -0.6, 1.1931471805599454),
    "truncnorm": (0.322545912833092, -0.7198275420771054, 1.0050201254964886),
    "uniform": (0.0, -1.2, 1.0986122886681098),
    "weibull": (1.0719865728909583, 1.3904035615957744, 1.4800872940856251),
}

DISCRETE_SHAPE = {
    "binomial": (
        0.19518001458970663,
        -0.06190476190476191,
        2.1325386416614887,
        [6.0, 11.0, 19.0, 20.0],
    ),
    "geometric": (
        2.0124611797498106,
        6.05,
        2.5020121176909393,
        [3.0, 20.0, 92.0, 154.0],
    ),
    "negbinom": (
        1.0159443179342345,
        1.5321428571428573,
        3.043109411412921,
        [8.0, 26.0, 80.0, 122.0],
    ),
    "poisson": (
        0.5345224838248488,
        0.2857142857142857,
        2.015172522512972,
        [3.0, 8.0, 20.0, 27.0],
    ),
}


def relative_error(got: float, expected: float) -> float:
    """Return ``|got - expected| / |expected|`` (absolute error at zero)."""
    if expected == 0.0:
        return abs(got)
    return abs(got - expected) / abs(expected)


class IncompleteBetaLargeShapeTests(unittest.TestCase):
    """Defect 0.13: ``I_x(a, b)`` for a large first shape near ``x = 1``."""

    def test_grid_matches_mpmath_to_1e_12(self) -> None:
        """Both tails match 40-digit references for ``a`` up to ``5e13``."""
        for a, b, y, lower, upper in BETA_GRID:
            with self.subTest(a=a, b=b, y=y):
                got_lower = regularized_beta_pair(1.0 - y, y, a, b)
                got_upper = regularized_beta_pair(y, 1.0 - y, b, a)
                if lower >= 1e-300:
                    self.assertLess(relative_error(got_lower, lower), 1e-12)
                if upper >= 1e-300:
                    self.assertLess(relative_error(got_upper, upper), 1e-12)

    def test_student_t_with_huge_degrees_of_freedom(self) -> None:
        """``StudentT(nu).sf`` keeps 1e-12 accuracy for ``nu`` up to ``1e14``."""
        for nu, t, expected in STUDENT_T:
            with self.subTest(nu=nu, t=t):
                self.assertLess(relative_error(StudentT(nu).sf(t), expected), 1e-12)
        # The audit's example: 1.34928e-3 before the fix, 1.34990e-3 exact.
        self.assertAlmostEqual(StudentT(1e14).sf(3.0) / 1.3498980316304269e-3, 1.0, 12)

    def test_fisher_f_with_huge_denominator(self) -> None:
        """``FisherF(d1, d2)`` for ``d2`` up to ``1e10`` in both tails."""
        for d1, d2, f, sf, cdf in FISHER_F:
            with self.subTest(d1=d1, d2=d2, f=f):
                law = FisherF(d1, d2)
                self.assertLess(relative_error(law.sf(f), sf), 1e-12)
                self.assertLess(relative_error(law.cdf(f), cdf), 1e-12)


class ContractTests(unittest.TestCase):
    """Defect P1.8: one contract for every univariate family."""

    LAWS = (
        Normal(1.0, 2.0),
        Gamma(2.5, 1.5),
        Beta(2.0, 3.0),
        StudentT(3.0),
        Cauchy(),
        Laplace(),
        Rayleigh(),
        Pareto(2.0, 3.0),
        PERT(0.0, 1.0, 4.0),
        Truncated(Normal(), lower=0.5),
        Mixture((Normal(), Normal(3.0, 0.5)), (0.5, 0.5)),
        Affine(Exponential(1.0), 2.0, -1.5),
        Empirical([0.3, 1.2, -0.4, 2.2, 0.9]),
    )

    def test_infinite_arguments_return_limits(self) -> None:
        """``cdf(-inf) = 0``, ``cdf(inf) = 1``, ``pdf(+-inf) = 0`` everywhere."""
        for law in self.LAWS:
            with self.subTest(law=type(law).__name__):
                self.assertEqual(law.cdf(-math.inf), 0.0)
                self.assertEqual(law.cdf(math.inf), 1.0)
                self.assertEqual(law.sf(-math.inf), 1.0)
                self.assertEqual(law.sf(math.inf), 0.0)
                self.assertEqual(law.pdf(math.inf), 0.0)
                self.assertEqual(law.pdf(-math.inf), 0.0)
                self.assertEqual(law.log_cdf(-math.inf), -math.inf)
                self.assertEqual(law.log_sf(math.inf), -math.inf)
                with self.assertRaises(ValueError):
                    law.cdf(math.nan)

    def test_every_law_is_a_model_input(self) -> None:
        """Every law, including heavy-tailed ones, can be a model input."""
        for index, law in enumerate(self.LAWS):
            with self.subTest(law=type(law).__name__):
                self.assertIsInstance(law, ContinuousDistribution)
                item = RandomInput(f"x{index}", law)
                joint = JointInput((item,))
                points = joint.sample_design(latin_hypercube(16, 1, seed=index))
                self.assertEqual(len(points), 16)
                self.assertTrue(all(math.isfinite(p[0]) for p in points))
        self.assertIsNotNone(RandomInput("x", StudentT(1.0)))

    def test_defaults_are_consistent(self) -> None:
        """``median``, ``std``, ``log_cdf``, ``isf`` agree with primary methods."""
        for law in self.LAWS:
            with self.subTest(law=type(law).__name__):
                self.assertAlmostEqual(law.median, law.quantile(0.5), 12)
                x = law.quantile(0.3)
                self.assertAlmostEqual(law.log_cdf(x), math.log(law.cdf(x)), 10)
                self.assertAlmostEqual(law.log_sf(x), math.log(law.sf(x)), 10)
                self.assertAlmostEqual(law.isf(0.7), law.quantile(0.3), 8)
                variance = law.variance
                if math.isfinite(variance):
                    self.assertAlmostEqual(law.std, math.sqrt(variance), 12)

    def test_undefined_and_infinite_moments_never_raise(self) -> None:
        """Undefined moments are ``nan``; divergent ones are ``inf``."""
        for law in (StudentT(1.0), StudentT(0.5), Cauchy(2.0, 3.0)):
            self.assertTrue(math.isnan(law.mean))
            self.assertTrue(math.isnan(law.variance))
            self.assertTrue(math.isnan(law.skewness))
            self.assertTrue(math.isnan(law.kurtosis))
        self.assertEqual(StudentT(1.5).variance, math.inf)
        self.assertEqual(StudentT(3.5).kurtosis, math.inf)
        self.assertAlmostEqual(StudentT(6.0).kurtosis, 3.0)
        self.assertEqual(Pareto(1.0, 0.8).mean, math.inf)
        self.assertEqual(Pareto(1.0, 1.5).variance, math.inf)
        self.assertEqual(Pareto(1.0, 2.5).skewness, math.inf)
        self.assertTrue(math.isnan(Pareto(1.0, 1.5).kurtosis))
        self.assertAlmostEqual(Cauchy(2.0, 3.0).median, 2.0)
        self.assertAlmostEqual(Cauchy().entropy, StudentT(1.0).entropy, 12)

    def test_numerical_shape_and_entropy_defaults_match_scipy(self) -> None:
        """Quadrature defaults reproduce closed forms to about 1e-9."""
        laws = {
            "normal": Normal(1.0, 2.0),
            "gamma": Gamma(2.5, 1.5),
            "beta": Beta(2.0, 3.0),
            "weibull": Weibull(1.5, 2.0),
            "lognormal": LogNormal(0.0, 0.5),
            "logistic": Logistic(0.0, 1.0),
            "gumbel": Gumbel(0.0, 1.0),
            "exponential": Exponential(2.0),
            "uniform": Uniform(-1.0, 2.0),
            "triangular": Triangular(0.0, 1.0, 4.0),
            "truncnorm": TruncatedNormal(0.0, 1.0, -1.0, 2.0),
            "chi2": ChiSquare(5.0),
            "f": FisherF(5.0, 12.0),
            "t5": StudentT(5.0),
        }
        for name, law in laws.items():
            skew, kurt, entropy = CONTINUOUS_SHAPE[name]
            with self.subTest(name):
                self.assertAlmostEqual(
                    law.skewness, skew, delta=1e-8 * max(1, abs(skew))
                )
                self.assertAlmostEqual(
                    law.kurtosis, kurt, delta=1e-7 * max(1, abs(kurt))
                )
                self.assertAlmostEqual(law.entropy, entropy, delta=1e-9)


class DiscreteContractTests(unittest.TestCase):
    """Discrete laws: integral floats, ``isf``, and mass-based moments."""

    LAWS = {
        "poisson": Poisson(3.5),
        "binomial": Binomial(20, 0.3),
        "geometric": Geometric(0.2),
        "negbinom": NegativeBinomial(4, 0.3),
    }

    def test_quantile_output_is_accepted_by_pmf(self) -> None:
        """``pmf(quantile(p))`` works and equals ``pmf(int(...))``."""
        for law in (
            *self.LAWS.values(),
            DiscreteUniform(-2, 5),
            Hypergeometric(50, 12, 10),
        ):
            with self.subTest(law=type(law).__name__):
                self.assertIsInstance(law, DiscreteDistribution)
                k = law.quantile(0.6)
                self.assertEqual(law.pmf(k), law.pmf(int(k)))
                self.assertEqual(law.log_pmf(k), law.log_pmf(int(k)))
                self.assertEqual(law.pmf(math.inf), 0.0)
                self.assertEqual(law.cdf(math.inf), 1.0)
                self.assertEqual(law.cdf(-math.inf), 0.0)
                with self.assertRaises(TypeError):
                    law.pmf(1.5)

    def test_isf_shape_and_entropy_match_scipy(self) -> None:
        """``isf`` (deep tails), skewness, excess kurtosis, entropy versus scipy."""
        for name, law in self.LAWS.items():
            skew, kurt, entropy, isf = DISCRETE_SHAPE[name]
            with self.subTest(name):
                self.assertEqual([law.isf(q) for q in (0.5, 0.01, 1e-9, 1e-15)], isf)
                self.assertAlmostEqual(law.skewness, skew, delta=1e-10)
                self.assertAlmostEqual(law.kurtosis, kurt, delta=1e-10)
                self.assertAlmostEqual(law.entropy, entropy, delta=1e-12)
                self.assertEqual(law.isf(1.0), float(law.support.lower))


class NewFamilyTests(unittest.TestCase):
    """New families against scipy reference values."""

    def check(self, law: object, ref: dict[str, list[float]], points, probs) -> None:
        """Compare every reference entry present in ``ref``."""
        for key, method in (
            ("pdf", "pdf"),
            ("logpdf", "log_pdf"),
            ("pmf", "pmf"),
            ("cdf", "cdf"),
            ("sf", "sf"),
        ):
            if key not in ref:
                continue
            for x, expected in zip(points, ref[key], strict=True):
                got = getattr(law, method)(x)
                with self.subTest(key=key, x=x):
                    if key == "logpdf":
                        self.assertAlmostEqual(
                            got, expected, delta=1e-12 * max(1, abs(expected))
                        )
                    else:
                        self.assertLess(relative_error(got, expected), 1e-12)
        for key, method in (("ppf", "quantile"), ("isf", "isf")):
            for p, expected in zip(probs, ref[key], strict=True):
                with self.subTest(key=key, p=p):
                    self.assertLess(
                        relative_error(getattr(law, method)(p), expected), 1e-9
                    )
        mean, variance, skew, kurt = ref["mvsk"]
        for got, expected in (
            (law.mean, mean),
            (law.variance, variance),
            (law.skewness, skew),
            (law.kurtosis, kurt),
        ):
            if math.isnan(expected):
                self.assertTrue(math.isnan(got))
            else:
                self.assertAlmostEqual(
                    got, expected, delta=1e-10 * max(1, abs(expected))
                )
        self.assertAlmostEqual(law.entropy, ref["entropy"], delta=1e-10)

    def test_cauchy(self) -> None:
        """Cauchy(1, 2), including tails far beyond ``atan`` cancellation."""
        self.check(
            Cauchy(1.0, 2.0),
            SCIPY["cauchy"],
            [-1e6, -3.0, 1.0, 2.5, 40.0],
            [1e-12, 0.1, 0.5, 0.9, 1 - 1e-9],
        )

    def test_laplace(self) -> None:
        """Laplace(-1, 0.5)."""
        self.check(
            Laplace(-1.0, 0.5),
            SCIPY["laplace"],
            [-30.0, -2.0, -1.0, 0.3, 15.0],
            [1e-15, 0.2, 0.5, 0.75, 0.999],
        )

    def test_rayleigh(self) -> None:
        """Rayleigh(1.5)."""
        self.check(
            Rayleigh(1.5),
            SCIPY["rayleigh"],
            [0.01, 0.5, 1.5, 3.0, 12.0],
            [1e-10, 0.3, 0.5, 0.9, 1 - 1e-12],
        )

    def test_pareto(self) -> None:
        """Pareto(2, 4.5) type I."""
        self.check(
            Pareto(2.0, 4.5),
            SCIPY["pareto"],
            [2.0, 2.5, 4.0, 10.0, 1e4],
            [1e-9, 0.25, 0.5, 0.95, 1 - 1e-10],
        )

    def test_pert(self) -> None:
        """PERT(1, 3, 10) with ``lam = 4`` equals Beta(1 + 8/9, 1 + 28/9) on [1, 10]."""
        law = PERT(1.0, 3.0, 10.0)
        self.assertAlmostEqual(law.mean, (1.0 + 4.0 * 3.0 + 10.0) / 6.0)
        self.check(
            law,
            SCIPY["pert"],
            [1.0001, 2.0, 3.0, 6.0, 9.99],
            [1e-8, 0.1, 0.5, 0.9, 0.99999],
        )
        with self.assertRaises(ValueError):
            PERT(0.0, 5.0, 4.0)

    def test_discrete_uniform(self) -> None:
        """DiscreteUniform(-2, 5).

        scipy's ``randint.ppf(0)`` is ``low - 1``; the convention here is the
        support minimum, so that entry of the reference was set to ``-2``.
        """
        self.check(
            DiscreteUniform(-2, 5),
            SCIPY["dunif"],
            [-3, -2, 0, 3, 5],
            [0.0, 0.1, 0.5, 0.875, 0.99],
        )

    def test_hypergeometric(self) -> None:
        """Hypergeometric(N=50, K=12, n=10) and exact rational masses."""
        law = Hypergeometric(50, 12, 10)
        self.check(
            law,
            SCIPY["hypergeom"],
            [0, 1, 2, 5, 10],
            [0.001, 0.1, 0.5, 0.9, 0.999],
        )
        total = math.comb(50, 10)
        for k in range(11):
            exact = math.comb(12, k) * math.comb(38, 10 - k) / total
            self.assertLess(relative_error(law.pmf(k), exact), 1e-12)
        self.assertAlmostEqual(math.fsum(law.pmf(k) for k in range(11)), 1.0, 15)


class WrapperTests(unittest.TestCase):
    """Affine, Truncated, Mixture, Empirical, and KDE quantiles."""

    def test_affine_reflected_gamma(self) -> None:
        """``1 - 2 Y`` with ``Y ~ Gamma(2.5, rate 1.5)``."""
        law = Affine(Gamma(2.5, 1.5), 1.0, -2.0)
        ref = SCIPY["affine_gamma"]
        for x, pdf, cdf in zip(
            (-12.0, -3.0, 0.0, 0.9), ref["pdf"], ref["cdf"], strict=True
        ):
            self.assertLess(relative_error(law.pdf(x), pdf), 1e-12)
            self.assertLess(relative_error(law.cdf(x), cdf), 1e-12)
        for p, expected in zip((1e-10, 0.3, 0.9), ref["ppf"], strict=True):
            self.assertLess(relative_error(law.quantile(p), expected), 1e-10)
        mean, variance, skew, kurt = ref["mvsk"]
        self.assertAlmostEqual(law.mean, mean, 12)
        self.assertAlmostEqual(law.variance, variance, 12)
        self.assertAlmostEqual(law.skewness, skew, 8)
        self.assertAlmostEqual(law.kurtosis, kurt, 7)
        self.assertAlmostEqual(law.entropy, ref["entropy"], 8)
        with self.assertRaises(ValueError):
            Affine(Normal(), 0.0, 0.0)

    def test_one_sided_truncated_normal(self) -> None:
        """``Normal`` truncated below at 1.5 equals scipy ``truncnorm(1.5, inf)``."""
        law = Truncated(Normal(), lower=1.5)
        ref = SCIPY["trunc_normal_lower"]
        for x, pdf, cdf, sf in zip(
            (1.5, 1.6, 2.0, 3.0, 6.0), ref["pdf"], ref["cdf"], ref["sf"], strict=True
        ):
            self.assertLess(relative_error(law.pdf(x), pdf), 1e-12)
            self.assertLess(relative_error(law.cdf(x), cdf), 1e-11)
            self.assertLess(relative_error(law.sf(x), sf), 1e-11)
        for p, q, ppf in zip(
            (1e-9, 0.2, 0.5, 0.8, 1 - 1e-9),
            (1e-9, 0.2, 0.5, 0.8, 1 - 1e-9),
            ref["ppf"],
            strict=True,
        ):
            # p = 1 - 1e-9 is rounded to the nearest double, which moves the
            # target tail mass by about 1e-7 relative: allow 1e-8 there.
            self.assertLess(relative_error(law.quantile(p), ppf), 1e-8)
            del q
        for q, isf in zip((1e-9, 0.2, 0.5, 0.8, 1 - 1e-9), ref["isf"], strict=True):
            self.assertLess(relative_error(law.isf(q), isf), 1e-9)
        mean, variance, _, _ = ref["mvsk"]
        self.assertAlmostEqual(law.mean, mean, 9)
        self.assertAlmostEqual(law.variance, variance, 9)
        compare = TruncatedNormal(0.0, 1.0, 1.5, None)
        self.assertAlmostEqual(law.cdf(2.2), compare.cdf(2.2), 13)

    def test_truncated_gamma_and_upper_tail_lognormal(self) -> None:
        """Upper truncation of a gamma; an interval deep in a lognormal tail."""
        law = Truncated(Gamma(2.5, 1.5), upper=2.0)
        ref = SCIPY["trunc_gamma_upper"]
        for x, cdf in zip((0.1, 0.5, 1.0, 1.9), ref["cdf"], strict=True):
            self.assertLess(relative_error(law.cdf(x), cdf), 1e-12)
        for p, ppf in zip((1e-6, 0.25, 0.5, 0.99), ref["ppf"], strict=True):
            self.assertLess(relative_error(law.quantile(p), ppf), 1e-10)
        self.assertAlmostEqual(law.mean, ref["mean"], 10)
        self.assertAlmostEqual(law.variance, ref["var"], 10)
        tail = Truncated(LogNormal(0.0, 0.5), 5.0, 9.0)
        ref = SCIPY["trunc_lognormal_tail"]
        for x, cdf, sf in zip((5.5, 7.0, 8.9), ref["cdf"], ref["sf"], strict=True):
            self.assertLess(relative_error(tail.cdf(x), cdf), 1e-11)
            self.assertLess(relative_error(tail.sf(x), sf), 1e-10)
        for p, ppf in zip((1e-6, 0.5, 0.999), ref["ppf"], strict=True):
            self.assertLess(relative_error(tail.quantile(p), ppf), 1e-10)
        with self.assertRaises(ValueError):
            Truncated(Uniform(0.0, 1.0), 2.0, 3.0)

    def test_mixture_quantile_by_bracketing(self) -> None:
        """``0.3 N(0,1) + 0.7 N(3, 0.5)``: CDF, quantiles, far upper tail."""
        law = Mixture((Normal(), Normal(3.0, 0.5)), (0.3, 0.7))
        ref = SCIPY["mixture_normal"]
        for x, cdf in zip((-4.0, 0.0, 1.5, 3.0, 5.0), ref["cdf"], strict=True):
            self.assertLess(relative_error(law.cdf(x), cdf), 1e-13)
        for p, ppf in zip(
            (1e-12, 0.05, 0.3, 0.5, 0.9, 1 - 1e-9), ref["ppf"], strict=True
        ):
            self.assertLess(relative_error(law.quantile(p), ppf), 1e-11)
        self.assertLess(relative_error(law.isf(1e-14), ref["isf_1e-14"]), 1e-11)
        self.assertAlmostEqual(law.mean, ref["mean"], 13)
        self.assertAlmostEqual(law.variance, ref["var"], 13)
        draws = law.sample(4000, seed=5)
        self.assertEqual(draws, law.sample(4000, seed=RandomStream(5)))
        self.assertAlmostEqual(sum(draws) / len(draws), 2.1, delta=0.1)
        with self.assertRaises(ValueError):
            Mixture((Normal(),), (0.5,))

    def test_empirical_quantiles_match_numpy_methods(self) -> None:
        """Linear = numpy ``linear`` (type 7); step = ``inverted_cdf`` (type 1)."""
        data = [3.1, -0.4, 2.2, 7.5, 2.2, 0.9, 5.0, 1.3, -2.0, 4.4]
        ref = SCIPY["empirical"]
        linear = Empirical(data)
        step = Empirical(data, method="step")
        for p, expected in zip(
            (0.0, 0.05, 0.33, 0.5, 0.9, 1.0), ref["q7"], strict=True
        ):
            self.assertAlmostEqual(linear.quantile(p), expected, 12)
        for p, expected in zip(
            (0.05, 0.1, 0.33, 0.5, 0.9, 1.0), ref["q1"], strict=True
        ):
            self.assertAlmostEqual(step.quantile(p), expected, 12)
        for p in (0.05, 0.33, 0.71, 0.95):
            self.assertAlmostEqual(linear.cdf(linear.quantile(p)), p, 12)
        self.assertAlmostEqual(step.cdf(2.2), 0.6)
        self.assertAlmostEqual(step.mean, sum(data) / len(data), 13)
        # Moments of the interpolated law by brute-force midpoint integration.
        grid = 200_000
        width = 1.0 / grid
        values = [linear.quantile((i + 0.5) * width) for i in range(grid)]
        mean = math.fsum(values) * width
        self.assertAlmostEqual(linear.mean, mean, 9)
        variance = math.fsum((v - mean) ** 2 for v in values) * width
        self.assertAlmostEqual(linear.variance, variance, 6)
        with self.assertRaises(ValueError):
            step.pdf(1.0)

    def test_kde_quantile(self) -> None:
        """KDE quantiles against a 40-digit bisection of the KDE CDF."""
        data = [3.1, -0.4, 2.2, 7.5, 2.2, 0.9, 5.0, 1.3, -2.0, 4.4]
        kde = gaussian_kde(data, bandwidth=0.7)
        for p, expected in zip(
            (1e-10, 0.1, 0.5, 0.95, 1 - 1e-12), SCIPY["kde"]["ppf"], strict=True
        ):
            self.assertLess(relative_error(kde.quantile(p), expected), 1e-11)
        self.assertAlmostEqual(
            kde.variance, sum((v - kde.mean) ** 2 for v in data) / 10 + 0.49
        )
        item = RandomInput("x", kde)
        self.assertAlmostEqual(item.mean, kde.mean)
        self.assertEqual(kde.draw(5, seed=1), kde.draw(5, seed=RandomStream(1)))


class SeedTests(unittest.TestCase):
    """Defect P1.8 (seeds): ``int | RandomStream | None`` and strict validation."""

    SAMPLERS = (
        Normal(),
        Gamma(2.0),
        Beta(2.0, 2.0),
        StudentT(4.0),
        Poisson(3.0),
        Binomial(5, 0.4),
        Cauchy(),
        Laplace(),
        Pareto(1.0, 2.0),
        PERT(0.0, 1.0, 2.0),
        DiscreteUniform(0, 3),
        Hypergeometric(10, 4, 3),
        Truncated(Normal(), lower=0.0),
        Affine(Normal(), 1.0, 2.0),
    )

    def test_stream_and_integer_seeds_agree(self) -> None:
        """An integer seed and ``RandomStream`` of that seed give equal draws."""
        for law in self.SAMPLERS:
            with self.subTest(law=type(law).__name__):
                self.assertEqual(
                    law.sample(5, seed=11), law.sample(5, seed=RandomStream(11))
                )
                self.assertEqual(len(law.sample(3)), 3)

    def test_invalid_seeds_are_rejected(self) -> None:
        """Negative, boolean, float, and text seeds raise clear errors."""
        for law in self.SAMPLERS:
            with self.subTest(law=type(law).__name__):
                with self.assertRaises(ValueError):
                    law.sample(2, seed=-1)
                for bad in (True, 1.5, "7"):
                    with self.assertRaises(TypeError):
                        law.sample(2, seed=bad)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
