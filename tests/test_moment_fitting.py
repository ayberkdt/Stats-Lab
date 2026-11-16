"""Tests for method-of-moments fitting (``inference.fit_distribution_moments``).

Oracles:

* R 4.6.2 (2026-10-29 ucrt) with fitdistrplus 1.2.6 and evd 2.3.7.1,
  computed offline; R is not called by the tests. ``R_CLOSED_FORM`` holds
  ``fitdist(x, distr, method = "mme")$estimate`` for ``distr`` in
  ``norm, lnorm, exp, gamma, logis, unif, beta`` (closed formulas, population
  variance ``(n - 1)/n var(x)``). ``R_ROOTS`` holds the Weibull and GEV
  moment equations solved in R by ``uniroot(..., tol = 1e-15)`` with R's
  ``gamma`` (Weibull: ``gamma(1 + 2/k)/gamma(1 + 1/k)^2 - 1 = v/m^2``; GEV:
  the textbook skewness ``sign(xi)(g3 - 3 g1 g2 + 2 g1^3)/(g2 - g1^2)^1.5``)
  and the Gumbel closed formula ``beta = sqrt(6 v)/pi``,
  ``loc = m + digamma(1) beta``. ``R_OPTIMIZED`` holds
  ``fitdist(x, "weibull" | "gumbel" | "gev", method = "mme", order = 1:k,
  memp = function(x, order) mean(x^order), control = list(reltol = 1e-16))``
  with user-defined ``mweibull``, ``mgumbel``, ``mgev`` (Nelder-Mead; its
  accuracy is that of the optimizer; the GEV fits of ``wind50`` and
  ``gev60`` did not converge in R and are omitted). ``R_VCOV_FITDIST`` holds
  ``fitdist(x, distr, method = "mme")$vcov`` (``calcvcov = TRUE``) for
  ``norm, lnorm, exp, gamma``. For ``beta``, ``unif``, and ``logis`` the
  ``calcvcov`` result of fitdistrplus 1.2.6 is wrong: its internal moment
  functions ``mbeta``, ``munif``, ``mlogis`` return the second moment for
  orders 3 and 4, which ``mme.Ahat`` needs (the ``beta40`` uniform "vcov"
  has a negative variance). ``R_VCOV_CORRECTED`` therefore holds
  ``fitdistrplus:::mme.vcov(par, NULL, 1:k, x, m, memp, NULL)`` at the moment
  estimate with correct moment functions written in R (raw moments from
  cumulants for Gumbel and logistic, ``(b^(r+1) - a^(r+1))/((r+1)(b-a))``
  for the uniform, ``prod((a + j)/(a + b + j))`` for the beta,
  ``scale^r gamma(1 + r/shape)`` for the Weibull, the binomial expansion of
  ``gamma(1 - i xi)`` for the GEV). fitdistrplus differentiates the moment
  map numerically, so its covariances carry relative errors up to about
  ``5e-5``.
* Closed forms: delta-method covariances of the normal, exponential,
  uniform, logistic, and gamma estimators; the Rayleigh skewness
  ``2 sqrt(pi)(pi - 3)/(4 - pi)^1.5`` (GEV ``xi = -1/2``) and the
  exponential skewness ``2`` (``xi = -1``); Gumbel cumulants
  ``k_r = (r - 1)! zeta(r)``.
* Seeded simulations judged by their Monte Carlo standard errors.

``wind50`` is the first 50 ``wind_max`` values of the ATHENS ``data_wind.dat``
file; the other data sets were drawn with probstat samplers (seeds 101-104)
and rounded to six decimals. On all 400 ``wind_max`` values, fitdistrplus
gives ``meanlog = 2.978689811447``, ``sdlog = 0.161122240569`` (lognormal)
and ``mean = 19.9189193194``, ``sd = 3.2303231873`` (normal), which
``fit_distribution_moments`` reproduces to ``5e-15``.

References:
    [EQ-INF-MOM-001], [EQ-INF-MOM-002], [EQ-INF-MOM-003], [EQ-INF-MOM-004],
    [EQ-EST-006], [REF-DELIGNETTE-MULLER-DUTANG-2015].

"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Gamma, Normal, Uniform
from probstat.foundations import RandomStream
from probstat.inference import (
    BETA_FAMILY,
    EXPONENTIAL_FAMILY,
    GAMMA_FAMILY,
    GEV_FAMILY,
    GUMBEL_FAMILY,
    LOGISTIC_FAMILY,
    LOGNORMAL_FAMILY,
    MOMENT_FAMILIES,
    NORMAL_FAMILY,
    STUDENT_T_FAMILY,
    UNIFORM_FAMILY,
    WEIBULL_FAMILY,
    DistributionFamily,
    MomentFitResult,
    evaluate_estimator,
    fit_distribution_moments,
    moment_fitting,
    uniform_maximum_risks,
)

FAMILIES: dict[str, DistributionFamily] = {
    "norm": NORMAL_FAMILY,
    "lnorm": LOGNORMAL_FAMILY,
    "exp": EXPONENTIAL_FAMILY,
    "gamma": GAMMA_FAMILY,
    "logis": LOGISTIC_FAMILY,
    "unif": UNIFORM_FAMILY,
    "beta": BETA_FAMILY,
    "weibull": WEIBULL_FAMILY,
    "gumbel": GUMBEL_FAMILY,
    "gev": GEV_FAMILY,
}

DATASETS: dict[str, tuple[float, ...]] = {
    "wind50": (
        18.6515175658683,
        16.932604633958,
        18.5121267020092,
        18.4809434722624,
        17.252441735754,
        21.0107694999361,
        16.6848466806125,
        19.8602417056273,
        19.6442411639239,
        23.9249075189026,
        19.9671465243467,
        17.0277539907003,
        20.1984871508146,
        21.0994389097136,
        16.4807124151805,
        20.7344041042839,
        21.4760810858122,
        20.8950533036682,
        22.3085491101921,
        15.6883700907697,
        17.3283527711254,
        21.2673655459591,
        19.4514811910373,
        25.7764854119099,
        20.6857656723623,
        24.925546759725,
        17.8842590291136,
        24.0061143489463,
        21.0193229758994,
        16.6304472543603,
        22.4270120008507,
        15.212955068943,
        16.2417526139,
        16.343590370026,
        19.7291671317885,
        19.0830187532973,
        20.4586343844211,
        22.5938290322235,
        19.6070374151884,
        21.8136126170399,
        19.0810620553307,
        17.034146006981,
        25.8646473951053,
        21.2207933349258,
        24.0322952394691,
        20.119121779628,
        19.0528915958451,
        20.6813903185994,
        18.6027683303799,
        16.4388663326088,
    ),
    "gamma40": (
        2.059196,
        2.570761,
        0.821759,
        4.17717,
        2.470452,
        8.854498,
        5.108845,
        4.110967,
        1.739113,
        6.007436,
        2.081574,
        4.439037,
        8.551018,
        7.465094,
        5.635385,
        1.877363,
        1.062391,
        6.003567,
        1.51002,
        2.840054,
        7.882727,
        3.534661,
        3.750475,
        4.273775,
        3.010793,
        1.446881,
        0.351754,
        3.384909,
        2.107689,
        0.853103,
        6.249287,
        3.831147,
        4.067684,
        2.347556,
        5.29109,
        0.629198,
        4.579752,
        0.692625,
        4.543676,
        2.50794,
    ),
    "beta40": (
        0.284168,
        0.199133,
        0.40655,
        0.584276,
        0.323907,
        0.097762,
        0.288054,
        0.228156,
        0.065665,
        0.376593,
        0.220529,
        0.277379,
        0.016253,
        0.560459,
        0.098021,
        0.469714,
        0.374365,
        0.452991,
        0.38981,
        0.210071,
        0.236394,
        0.289929,
        0.348844,
        0.110945,
        0.596947,
        0.244456,
        0.410657,
        0.47845,
        0.195634,
        0.445385,
        0.403852,
        0.325641,
        0.286864,
        0.739506,
        0.474412,
        0.454044,
        0.482025,
        0.482984,
        0.099628,
        0.105769,
    ),
    "beta30": (
        0.683726,
        0.341058,
        0.265135,
        0.128708,
        0.351297,
        0.553832,
        0.121592,
        0.351877,
        0.111638,
        0.308141,
        0.228072,
        0.5278,
        0.240624,
        0.205999,
        0.202943,
        0.640532,
        0.488151,
        0.232319,
        0.273096,
        0.054417,
        0.257314,
        0.398067,
        0.240928,
        0.238199,
        0.305831,
        0.00198,
        0.631992,
        0.350627,
        0.475845,
        0.208128,
    ),
    "gev60": (
        14.112161,
        12.993496,
        11.965576,
        9.167843,
        9.794086,
        11.855495,
        12.464106,
        6.4935,
        8.725563,
        13.224017,
        8.295341,
        11.518019,
        9.788799,
        11.463477,
        12.907018,
        11.118688,
        10.125359,
        9.746996,
        11.331856,
        11.369238,
        10.223351,
        10.520057,
        10.882291,
        9.366336,
        8.827055,
        11.050579,
        8.107805,
        11.204873,
        12.4286,
        12.624511,
        9.32808,
        10.250883,
        13.54535,
        9.901454,
        15.072985,
        8.976675,
        12.725033,
        7.220604,
        11.586765,
        12.455662,
        10.940858,
        11.393197,
        13.427588,
        15.425402,
        8.824099,
        10.811512,
        7.196532,
        10.057674,
        8.545808,
        8.673565,
        11.35034,
        7.921258,
        8.621891,
        6.111472,
        12.514283,
        9.273281,
        7.759248,
        11.329666,
        11.878114,
        11.381373,
    ),
}

# fitdist(x, distr, method = "mme")$estimate (closed formulas)
R_CLOSED_FORM: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("wind50", "norm", (19.8288874020265, 2.62585772740887)),
    ("wind50", "lnorm", (2.97844752421451, 0.131850748474346)),
    ("wind50", "exp", (0.0504314730183903,)),
    ("wind50", "gamma", (57.0235577528797, 2.87578201422697)),
    ("wind50", "logis", (19.8288874020265, 1.44771124038711)),
    ("wind50", "unif", (15.280768404707, 24.377006399346)),
    ("gamma40", "norm", (3.61806055, 2.20852862898386)),
    ("gamma40", "lnorm", (1.1275811356788, 0.562773465827723)),
    ("gamma40", "exp", (0.276391173165966,)),
    ("gamma40", "gamma", (2.68377185887319, 0.741770852583768)),
    ("gamma40", "logis", (3.61806055, 1.21762564952508)),
    ("gamma40", "unif", (-0.207223245370487, 7.44334434537049)),
    ("beta40", "beta", (2.40276535123986, 4.91369245904945)),
    ("beta40", "norm", (0.32840555, 0.162850688467066)),
    ("beta40", "lnorm", (-1.22343501184663, 0.468890201266269)),
    ("beta40", "exp", (3.04501553034046,)),
    ("beta40", "gamma", (4.06669665188546, 12.3831544621748)),
    ("beta40", "logis", (0.32840555, 0.0897842901912261)),
    ("beta40", "unif", (0.04633988352747, 0.61047121647253)),
    ("beta30", "beta", (2.02819566290526, 4.43111670613832)),
    ("beta30", "norm", (0.3139956, 0.16993226054649)),
    ("beta30", "lnorm", (-1.28681631742641, 0.506833328650542)),
    ("beta30", "exp", (3.18475800297839,)),
    ("beta30", "gamma", (3.41425107900949, 10.8735634480531)),
    ("beta30", "logis", (0.3139956, 0.0936885655036242)),
    ("beta30", "unif", (0.019664290888448, 0.608326909111552)),
    ("gev60", "norm", (10.6366124, 2.0257018067902)),
    ("gev60", "lnorm", (2.34648831457016, 0.188752404964362)),
    ("gev60", "exp", (0.094014895193511,)),
    ("gev60", "gamma", (27.5711979685109, 2.5921032873691)),
    ("gev60", "logis", (10.6366124, 1.11682793959157)),
    ("gev60", "unif", (7.1279939496553, 14.1452308503447)),
)
# uniroot (tol = 1e-15) on the moment equations; Gumbel closed formula
R_ROOTS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("wind50", "weibull", (9.03162478235153, 20.9360758634347)),
    ("wind50", "gumbel", (18.6471117092475, 2.04737287055543)),
    ("wind50", "gev", (18.7714536746261, 2.43124607828629, -0.165008285550062)),
    ("gamma40", "weibull", (1.68489277453393, 4.05248339161587)),
    ("gamma40", "gumbel", (2.62410515656932, 1.72198270745172)),
    ("gamma40", "gev", (2.68216793910482, 1.93728325739041, -0.104088592753295)),
    ("beta40", "weibull", (2.12078485777299, 0.370809469672442)),
    ("beta40", "gumbel", (0.255114075304372, 0.126974160876474)),
    ("beta40", "gev", (0.267150569852955, 0.158229594866664, -0.231297514352093)),
    ("beta30", "weibull", (1.92437668850434, 0.353988637341226)),
    ("beta30", "gumbel", (0.237517041075562, 0.132495639974506)),
    ("beta30", "gev", (0.24347830220459, 0.152821928477724, -0.13079415352477)),
    ("gev60", "weibull", (6.11378994419375, 11.4534554676003)),
    ("gev60", "gumbel", (9.72493880432296, 1.5794332190076)),
    ("gev60", "gev", (9.91491724217087, 2.02255124974354, -0.276540973381526)),
)
# fitdist(..., method = "mme", order, memp, control = list(reltol = 1e-16))
R_OPTIMIZED: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("wind50", "weibull", (9.03162302473018, 20.9360759781226)),
    ("wind50", "gumbel", (18.6471117293725, 2.04737284073788)),
    ("gamma40", "weibull", (1.68489275389547, 4.0524833750244)),
    ("gamma40", "gumbel", (2.62410517962419, 1.72198268902508)),
    ("gamma40", "gev", (2.68216791197547, 1.93728327342586, -0.10408859293483)),
    ("beta40", "weibull", (2.12078485329164, 0.370809469673828)),
    ("beta40", "gumbel", (0.255114075634503, 0.126974160557242)),
    ("beta40", "gev", (0.267150573025992, 0.158229599376073, -0.231297557719301)),
    ("beta30", "weibull", (1.92437668122825, 0.35398863727891)),
    ("beta30", "gumbel", (0.237517041097984, 0.132495640047812)),
    ("beta30", "gev", (0.243478300547465, 0.152821939699927, -0.130794199142228)),
    ("gev60", "weibull", (6.11378659015113, 11.4534556119733)),
    ("gev60", "gumbel", (9.72493882578946, 1.57943319460598)),
)
# fitdist(x, distr, method = "mme")$vcov (calcvcov)
R_VCOV_FITDIST: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    (
        "wind50",
        "norm",
        (
            0.137902576091858,
            -2.82241272476634e-08,
            -2.82241271065686e-08,
            0.0689513076978328,
        ),
    ),
    (
        "wind50",
        "lnorm",
        (
            0.000347765240471067,
            -8.21638548365164e-07,
            -8.21638548363923e-07,
            0.000183130960668047,
        ),
    ),
    ("wind50", "exp", (5.08666694190292e-05,)),
    (
        "wind50",
        "gamma",
        (132.355236956155, 6.6748689196425, 6.67486891964243, 0.339524037883784),
    ),
    (
        "gamma40",
        "norm",
        (
            0.121939967626034,
            -2.14429738409905e-09,
            -2.14429738409905e-09,
            0.0609699833214985,
        ),
    ),
    (
        "gamma40",
        "lnorm",
        (
            0.00899355798126237,
            -0.00251217770966559,
            -0.0025121777096656,
            0.00994355808374998,
        ),
    ),
    ("gamma40", "exp", (0.00190980201421722,)),
    (
        "gamma40",
        "gamma",
        (0.494320178482111, 0.136625733130416, 0.136625733130415, 0.0428876193189662),
    ),
    (
        "beta40",
        "norm",
        (
            0.000663008668354938,
            -8.76924401149726e-13,
            -8.76924401149726e-13,
            0.000331504333321631,
        ),
    ),
    (
        "beta40",
        "lnorm",
        (
            0.00578623191591094,
            -0.000841498261821839,
            -0.000841498261821838,
            0.00523248747886173,
        ),
    ),
    ("beta40", "exp", (0.231802990005776,)),
    (
        "beta40",
        "gamma",
        (1.03023503501014, 3.1370814583391, 3.1370814583391, 10.4951334876081),
    ),
    (
        "beta30",
        "norm",
        (
            0.000962565772481333,
            1.43477220844919e-12,
            1.43477220844919e-12,
            0.000481282885690078,
        ),
    ),
    (
        "beta30",
        "lnorm",
        (
            0.00923209705366919,
            -0.00177344208104553,
            -0.00177344208104553,
            0.00906486236116394,
        ),
    ),
    ("beta30", "exp", (0.338089450845466,)),
    (
        "beta30",
        "gamma",
        (1.00475784851577, 3.19991062675718, 3.19991062675718, 11.3452633278239),
    ),
    (
        "gev60",
        "norm",
        (
            0.068391130167218,
            -5.17016591778302e-10,
            -5.17016592501326e-10,
            0.0341955666212634,
        ),
    ),
    (
        "gev60",
        "lnorm",
        (
            0.000594335526938789,
            -4.2511683988653e-06,
            -4.25116839886525e-06,
            0.000330218944037976,
        ),
    ),
    ("gev60", "exp", (0.000147313341957068,)),
    (
        "gev60",
        "gamma",
        (26.2579702702784, 2.46864053559631, 2.46864053559632, 0.236150606507077),
    ),
)
# fitdistrplus:::mme.vcov with corrected moment functions
R_VCOV_CORRECTED: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    (
        "wind50",
        "weibull",
        (1.31257119554946, 0.0907388497683581, 0.0907388497683598, 0.120277276739153),
    ),
    (
        "wind50",
        "gumbel",
        (
            0.097903380651582,
            0.00803351234880598,
            0.00803351234880637,
            0.0922181635926232,
        ),
    ),
    (
        "wind50",
        "logis",
        (
            0.137902576091858,
            -1.55607740755081e-08,
            -1.55607740737196e-08,
            0.0335338831255585,
        ),
    ),
    (
        "wind50",
        "unif",
        (0.220644026094844, 0.0551610503631406, 0.0551610503631403, 0.220644177546307),
    ),
    (
        "wind50",
        "gev",
        (
            0.14418477004177,
            0.0168647501916143,
            -0.0118544027297241,
            0.0168647501916382,
            0.0725420636111775,
            -0.0105817900372735,
            -0.0118544027297273,
            -0.0105817900372418,
            0.00764026692767286,
        ),
    ),
    (
        "gamma40",
        "weibull",
        (0.0461141499506305, 0.0274972468721423, 0.0274972468721423, 0.161073970118875),
    ),
    (
        "gamma40",
        "gumbel",
        (
            0.0865707733379065,
            0.00710362075531027,
            0.00710362075531026,
            0.0815436716319632,
        ),
    ),
    (
        "gamma40",
        "logis",
        (
            0.121939967626034,
            -1.18221312739308e-09,
            -1.18221312739308e-09,
            0.029652245019168,
        ),
    ),
    (
        "gamma40",
        "unif",
        (0.195103974514301, 0.0487759879261613, 0.0487759879261613, 0.195103934673889),
    ),
    (
        "gamma40",
        "gev",
        (
            0.116208044137787,
            0.022274075167561,
            -0.0128369686067348,
            0.0222740751675612,
            0.0640177129392414,
            -0.0117158660068124,
            -0.0128369686067347,
            -0.0117158660068121,
            0.0111196158176296,
        ),
    ),
    (
        "beta40",
        "weibull",
        (
            0.069557483012003,
            0.00242759455137659,
            0.0024275945513766,
            0.000849049739806071,
        ),
    ),
    (
        "beta40",
        "gumbel",
        (
            0.000470700253945753,
            3.86236107493854e-05,
            3.86236107493851e-05,
            0.00044336703279532,
        ),
    ),
    (
        "beta40",
        "logis",
        (
            0.000663008668354938,
            -4.83473591023076e-13,
            -4.83473591023076e-13,
            0.000161224375037238,
        ),
    ),
    (
        "beta40",
        "unif",
        (
            0.00106081387163331,
            0.000265203468114319,
            0.000265203468114319,
            0.0010608138655578,
        ),
    ),
    (
        "beta40",
        "beta",
        (0.288538574113763, 0.531291941523093, 0.531291941523092, 1.2953855238734),
    ),
    (
        "beta40",
        "gev",
        (
            0.000758847769873682,
            3.71011657038049e-05,
            -0.000985855964021114,
            3.7101165703804e-05,
            0.000368108342200908,
            -0.000920483025199286,
            -0.000985855964021104,
            -0.000920483025199277,
            0.00950346425611328,
        ),
    ),
    (
        "beta30",
        "weibull",
        (
            0.0773981693991626,
            0.00312251042312321,
            0.00312251042312322,
            0.00125391880145761,
        ),
    ),
    (
        "beta30",
        "gumbel",
        (
            0.000683369575893772,
            5.60743304977586e-05,
            5.60743304977585e-05,
            0.000643686800476977,
        ),
    ),
    (
        "beta30",
        "logis",
        (
            0.000962565772481333,
            7.91030551246704e-13,
            7.91030551246704e-13,
            0.000234067927864392,
        ),
    ),
    (
        "beta30",
        "unif",
        (
            0.00154010523357937,
            0.000385026315550345,
            0.000385026315550346,
            0.00154010522524527,
        ),
    ),
    (
        "beta30",
        "beta",
        (0.277396020587752, 0.5393008499915, 0.539300849991502, 1.43976489961858),
    ),
    (
        "beta30",
        "gev",
        (
            0.000956376157648086,
            0.000150064765400587,
            -0.00128284799701302,
            0.000150064765400588,
            0.000501548876441379,
            -0.00114281463589332,
            -0.00128284799701304,
            -0.00114281463589335,
            0.0135994012585807,
        ),
    ),
    (
        "gev60",
        "weibull",
        (0.456062644362501, 0.0452601442847486, 0.0452601442847488, 0.0651320184904868),
    ),
    (
        "gev60",
        "gumbel",
        (
            0.0485539997276454,
            0.00398413058770518,
            0.0039841305877052,
            0.0457344973208509,
        ),
    ),
    (
        "gev60",
        "logis",
        (
            0.068391130167218,
            -2.8504623782698e-10,
            -2.85046238464777e-10,
            0.0166307298931947,
        ),
    ),
    (
        "gev60",
        "unif",
        (0.109425868602017, 0.0273563637295817, 0.0273563637295819, 0.109425891996259),
    ),
    (
        "gev60",
        "gev",
        (
            0.0827407500732126,
            0.000583247380668913,
            -0.00895187910184902,
            0.000583247380671992,
            0.0399731741689646,
            -0.00858091030701293,
            -0.00895187910185555,
            -0.00858091030702012,
            0.00679600207751217,
        ),
    ),
)


def relative_error(actual: float, expected: float) -> float:
    """Return ``|actual - expected| / |expected|``.

    References:
        Test helper. No equation is evaluated.

    """
    return abs(actual - expected) / abs(expected)


def covariance_error(fit: MomentFitResult, flat: tuple[float, ...]) -> float:
    """Return ``max |C_ij - R_ij| / sqrt(R_ii R_jj)`` for a flattened matrix.

    References:
        Test helper. No equation is evaluated.

    """
    assert fit.covariance is not None
    k = len(fit.estimates)
    worst = 0.0
    for i in range(k):
        for j in range(k):
            scale = math.sqrt(flat[i * k + i] * flat[j * k + j])
            worst = max(worst, abs(fit.covariance[i][j] - flat[j * k + i]) / scale)
    return worst


def population_moments(values: tuple[float, ...]) -> tuple[float, float, float]:
    """Return the mean and the divisor-``n`` central moments of orders 2 and 4.

    References:
        Test helper. No equation is evaluated.

    """
    n = len(values)
    mean = math.fsum(values) / n
    m2 = math.fsum((x - mean) ** 2 for x in values) / n
    m4 = math.fsum((x - mean) ** 4 for x in values) / n
    return mean, m2, m4


class RReferenceEstimates(unittest.TestCase):
    """Estimates agree with fitdistrplus on five data sets."""

    def test_closed_form_families_match_fitdist_mme(self) -> None:
        """Closed formulas agree with ``fitdist(method = "mme")`` to 1e-12."""
        for name, label, expected in R_CLOSED_FORM:
            with self.subTest(data=name, family=label):
                fit = fit_distribution_moments(DATASETS[name], FAMILIES[label])
                for value, reference in zip(fit.estimates, expected, strict=True):
                    self.assertLess(relative_error(value, reference), 1e-12)

    def test_weibull_gumbel_gev_match_r_root_solutions(self) -> None:
        """Weibull and GEV roots and the Gumbel formula agree with R to 1e-10."""
        for name, label, expected in R_ROOTS:
            with self.subTest(data=name, family=label):
                fit = fit_distribution_moments(DATASETS[name], FAMILIES[label])
                for value, reference in zip(fit.estimates, expected, strict=True):
                    self.assertLess(relative_error(value, reference), 1e-10)

    def test_numerical_fitdist_mme_agrees_to_optimizer_accuracy(self) -> None:
        """Fitdistrplus's Nelder-Mead moment matching agrees to 1e-6."""
        for name, label, expected in R_OPTIMIZED:
            with self.subTest(data=name, family=label):
                fit = fit_distribution_moments(DATASETS[name], FAMILIES[label])
                for value, reference in zip(fit.estimates, expected, strict=True):
                    self.assertLess(relative_error(value, reference), 1e-6)

    def test_fitted_moments_equal_sample_moments(self) -> None:
        """Every fitted law has the sample mean and divisor-n variance."""
        for family in MOMENT_FAMILIES:
            for name, values in DATASETS.items():
                if not family.admits(values):
                    continue
                with self.subTest(family=family.name, data=name):
                    try:
                        fit = fit_distribution_moments(values, family)
                    except ValueError:
                        self.assertEqual(family.name, "Beta")
                        continue
                    mean, variance, _ = population_moments(values)
                    law = fit.distribution
                    self.assertLess(relative_error(law.mean, mean), 1e-12)  # type: ignore[attr-defined]
                    if len(fit.estimates) > 1:
                        self.assertLess(
                            relative_error(law.variance, variance),  # type: ignore[attr-defined]
                            1e-11,
                        )
                    self.assertEqual(fit.sample_moments[0], mean)
                    self.assertEqual(fit.method, "moments")


class RReferenceCovariances(unittest.TestCase):
    """Delta-method covariances agree with fitdistrplus and closed forms."""

    def test_model_covariance_matches_fitdist_calcvcov(self) -> None:
        """Normal, lognormal, exponential, gamma: within fitdistrplus's 1e-4."""
        for name, label, flat in R_VCOV_FITDIST:
            with self.subTest(data=name, family=label):
                fit = fit_distribution_moments(DATASETS[name], FAMILIES[label])
                self.assertLess(covariance_error(fit, flat), 1e-4)

    def test_model_covariance_matches_corrected_mme_vcov(self) -> None:
        """All other families against ``mme.vcov`` with correct moments: 1e-5."""
        for name, label, flat in R_VCOV_CORRECTED:
            with self.subTest(data=name, family=label):
                fit = fit_distribution_moments(DATASETS[name], FAMILIES[label])
                self.assertLess(covariance_error(fit, flat), 1e-5)

    def test_closed_form_model_covariances(self) -> None:
        """Delta-method formulas of five families hold to 1e-8."""
        values = DATASETS["wind50"]
        n = len(values)
        mean, v, _ = population_moments(values)
        normal = fit_distribution_moments(values, NORMAL_FAMILY).covariance
        assert normal is not None
        self.assertLess(relative_error(normal[0][0], v / n), 1e-9)
        self.assertLess(relative_error(normal[1][1], v / (2 * n)), 1e-8)
        self.assertLess(abs(normal[0][1]), 1e-9 * v / n)
        rate = 1.0 / mean
        exponential = fit_distribution_moments(values, EXPONENTIAL_FAMILY)
        assert exponential.covariance is not None
        self.assertLess(relative_error(exponential.covariance[0][0], rate**2 / n), 1e-9)
        width = 2.0 * math.sqrt(3.0 * v)
        uniform = fit_distribution_moments(values, UNIFORM_FAMILY).covariance
        assert uniform is not None
        self.assertLess(relative_error(uniform[0][0], 2 * width**2 / (15 * n)), 1e-8)
        self.assertLess(relative_error(uniform[0][1], width**2 / (30 * n)), 1e-8)
        logistic = fit_distribution_moments(values, LOGISTIC_FAMILY)
        assert logistic.covariance is not None
        s = logistic.estimates[1]
        self.assertLess(
            relative_error(logistic.covariance[1][1], 0.8 * s * s / n), 1e-8
        )
        # Gamma: shape = m^2/v, rate = m/v, with gamma central moments.
        gamma = fit_distribution_moments(values, GAMMA_FAMILY)
        a, b = gamma.estimates
        mu2, mu3, mu4 = a / b**2, 2 * a / b**3, (3 * a * a + 6 * a) / b**4
        cov_m = ((mu2, mu3), (mu3, mu4 - mu2 * mu2))
        grad = ((2 * mean / v, -(mean**2) / v**2), (1 / v, -mean / v**2))
        expected = [
            [
                math.fsum(
                    grad[i][p] * cov_m[p][q] * grad[j][q]
                    for p in range(2)
                    for q in range(2)
                )
                / n
                for j in range(2)
            ]
            for i in range(2)
        ]
        assert gamma.covariance is not None
        for i in range(2):
            for j in range(2):
                self.assertLess(
                    relative_error(gamma.covariance[i][j], expected[i][j]), 1e-8
                )

    def test_empirical_covariance_closed_form_for_the_normal(self) -> None:
        """Sandwich: Var(sd) = (m4 - v^2)/(4 v n), Cov = m3/(2 sd n)."""
        values = DATASETS["gamma40"]
        n = len(values)
        mean, v, m4 = population_moments(values)
        m3 = math.fsum((x - mean) ** 3 for x in values) / n
        fit = fit_distribution_moments(values, NORMAL_FAMILY, covariance="empirical")
        assert fit.covariance is not None
        self.assertEqual(fit.covariance_method, "empirical")
        self.assertLess(relative_error(fit.covariance[0][0], v / n), 1e-9)
        self.assertLess(
            relative_error(fit.covariance[1][1], (m4 - v * v) / (4 * v * n)), 1e-8
        )
        self.assertLess(
            relative_error(fit.covariance[0][1], m3 / (2 * math.sqrt(v) * n)), 1e-8
        )

    def test_standard_errors_match_simulated_spread(self) -> None:
        """Gamma shape SE equals the Monte Carlo SD within 4 standard errors."""
        root = RandomStream(20261116)
        truth = Gamma(3.0, 2.0)
        shapes: list[float] = []
        errors: list[float] = []
        for r in range(400):
            values = truth.sample(200, seed=root.child("data", r))
            fit = fit_distribution_moments(values, GAMMA_FAMILY)
            shapes.append(fit.estimates[0])
            errors.append(fit.standard_errors[0])
        count = len(shapes)
        center = math.fsum(shapes) / count
        spread = math.sqrt(math.fsum((x - center) ** 2 for x in shapes) / (count - 1))
        typical = math.fsum(errors) / count
        # The SD of a normal-sample SD has relative error 1/sqrt(2(R-1)).
        self.assertLess(abs(typical / spread - 1.0), 4.0 / math.sqrt(2 * (count - 1)))


class GevSkewness(unittest.TestCase):
    """The cancellation-free GEV skewness and central moments."""

    def test_exact_values(self) -> None:
        """Exponential, Rayleigh, and Gumbel skewness values."""
        rayleigh = 2.0 * math.sqrt(math.pi) * (math.pi - 3.0) / (4.0 - math.pi) ** 1.5
        self.assertAlmostEqual(moment_fitting._gev_skewness(-1.0), -2.0, delta=1e-13)
        self.assertLess(
            relative_error(moment_fitting._gev_skewness(-0.5), -rayleigh), 1e-13
        )
        gumbel = 12.0 * math.sqrt(6.0) * 1.2020569031595942 / math.pi**3
        self.assertLess(
            relative_error(moment_fitting._gev_skewness(0.0), gumbel), 1e-15
        )
        self.assertEqual(moment_fitting._gev_skewness(0.34), math.inf)

    def test_small_shapes_agree_with_quadrature(self) -> None:
        """Series and log-gamma branches agree with trapezoid moments to 1e-11."""
        for xi in (
            -0.09,
            -0.0501,
            -0.0499,
            -0.01,
            -1e-6,
            1e-6,
            0.003,
            0.0499,
            0.0501,
            0.09,
        ):
            with self.subTest(xi=xi):
                mu2, mu3 = moment_fitting._gev_central_moments_quadrature(xi, 3)
                self.assertLess(
                    relative_error(moment_fitting._gev_skewness(xi), mu3 / mu2**1.5),
                    1e-11,
                )

    def test_quadrature_reproduces_gumbel_cumulants(self) -> None:
        """At xi = 0 the central moments follow from k_r = (r-1)! zeta(r)."""
        z3, z5 = 1.2020569031595942, 1.0369277551433699
        k2, k3, k4 = math.pi**2 / 6, 2 * z3, math.pi**4 / 15
        k5, k6 = 24 * z5, 8 * math.pi**6 / 63
        exact = (
            k2,
            k3,
            k4 + 3 * k2**2,
            k5 + 10 * k3 * k2,
            k6 + 15 * k4 * k2 + 10 * k3**2 + 15 * k2**3,
        )
        moments = moment_fitting._gev_central_moments(0.0, 6)
        for value, reference in zip(moments, exact, strict=True):
            self.assertLess(relative_error(value, reference), 1e-13)

    def test_quadrature_and_gamma_forms_meet_at_the_switch(self) -> None:
        """Both central-moment forms agree to 1e-9 at xi = +-0.1 and +-0.15."""
        for xi in (-0.15, -0.1, 0.1, 0.15):
            gammas = [1.0] + [math.gamma(1.0 - i * xi) for i in range(1, 7)]
            closed = [
                math.fsum(
                    math.comb(r, i) * gammas[i] * (-gammas[1]) ** (r - i)
                    for i in range(r + 1)
                )
                / xi**r
                for r in range(2, 7)
            ]
            quadrature = moment_fitting._gev_central_moments_quadrature(xi, 6)
            for value, reference in zip(quadrature, closed, strict=True):
                self.assertLess(relative_error(value, reference), 1e-7)

    def test_heavy_tail_has_no_model_covariance(self) -> None:
        """For xi >= 1/6 the sixth moment is infinite: model SEs are nan."""
        values = tuple(1.0 + 0.01 * i for i in range(19)) + (50.0,)
        model = fit_distribution_moments(values, GEV_FAMILY)
        self.assertGreater(model.estimates[2], 1.0 / 6.0)
        self.assertTrue(all(math.isnan(se) for se in model.standard_errors))
        self.assertIsNone(model.covariance)
        self.assertIn("order 6", model.message)
        empirical = fit_distribution_moments(values, GEV_FAMILY, covariance="empirical")
        self.assertTrue(all(math.isfinite(se) for se in empirical.standard_errors))


class MomentsVersusLikelihood(unittest.TestCase):
    """The uniform-endpoint example: 2 xbar against (n+1)/n max."""

    def test_risks_by_simulation(self) -> None:
        """Simulated MSEs match theta^2/(3n) and theta^2/(n(n+2)) within 4 SE."""
        theta, n = 3.0, 10
        risks = {risk.label: risk.mse for risk in uniform_maximum_risks(theta, n)}
        self.assertAlmostEqual(risks["2 mean"], theta**2 / (3 * n), places=14)
        self.assertAlmostEqual(
            risks["(n+1)/n max"], theta**2 / (n * (n + 2)), places=14
        )
        law = Uniform(0.0, theta)

        def sampler(size: int, rng: object) -> tuple[float, ...]:
            """Draw ``size`` uniform values from the replication generator.

            References:
                Test helper. No equation is evaluated.

            """
            return tuple(law.quantile(rng.random()) for _ in range(size))  # type: ignore[attr-defined]

        moments = evaluate_estimator(
            lambda xs: 2.0 * math.fsum(xs) / len(xs), sampler, theta, n, 4000, seed=7
        )
        maximum = evaluate_estimator(
            lambda xs: (n + 1) / n * max(xs), sampler, theta, n, 4000, seed=7
        )
        for evaluation, exact in (
            (moments, risks["2 mean"]),
            (maximum, risks["(n+1)/n max"]),
        ):
            self.assertLess(abs(evaluation.mse - exact), 4.0 * evaluation.mse_se)


class MomentFitValidation(unittest.TestCase):
    """Errors, support warnings, and result fields."""

    def test_unsupported_families_raise(self) -> None:
        """Student t and custom families have no moment rule."""
        with self.assertRaisesRegex(ValueError, "StudentT"):
            fit_distribution_moments(DATASETS["wind50"], STUDENT_T_FAMILY)
        custom = DistributionFamily(
            "Normal",
            ("mu", "sigma"),
            ("location", "positive"),
            lambda p: Normal(p[0], p[1]),
            lambda s: (0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "no moment rule"):
            fit_distribution_moments(DATASETS["wind50"], custom)
        with self.assertRaises(TypeError):
            fit_distribution_moments(DATASETS["wind50"], "Normal")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "covariance"):
            fit_distribution_moments(
                DATASETS["wind50"],
                NORMAL_FAMILY,
                covariance="robust",  # type: ignore[arg-type]
            )

    def test_domain_is_enforced(self) -> None:
        """Lognormal and beta moments need positive and unit-interval data."""
        with self.assertRaisesRegex(ValueError, "domain"):
            fit_distribution_moments((-1.0, 2.0, 3.0), LOGNORMAL_FAMILY)
        with self.assertRaisesRegex(ValueError, "domain"):
            fit_distribution_moments(DATASETS["wind50"], BETA_FAMILY)

    def test_uniform_fit_may_exclude_observations(self) -> None:
        """M -+ sqrt(3 v) can fall inside the range: log-likelihood is -inf."""
        fit = fit_distribution_moments(DATASETS["wind50"], UNIFORM_FAMILY)
        self.assertGreater(fit.estimates[0], min(DATASETS["wind50"]))
        self.assertEqual(fit.log_likelihood, -math.inf)
        self.assertIn("outside the fitted support", fit.message)

    def test_result_fields(self) -> None:
        """Names, maps, and the matched moments of a GEV fit."""
        fit = fit_distribution_moments(DATASETS["gamma40"], GEV_FAMILY)
        self.assertEqual(fit.parameter_count, 3)
        self.assertEqual(set(fit.params), {"mu", "sigma", "xi"})
        self.assertEqual(set(fit.standard_error_map), {"mu", "sigma", "xi"})
        self.assertEqual(fit.moment_names, ("mean", "variance", "third central moment"))
        self.assertEqual(fit.sample_size, 40)
        self.assertTrue(math.isfinite(fit.log_likelihood))


if __name__ == "__main__":
    unittest.main()
