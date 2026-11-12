"""Internal special functions needed by probability-distribution algorithms."""

from __future__ import annotations

import math
from fractions import Fraction

from probstat.numerics._double_double import (
    LOG_SQRT_TWO_PI,
    STIRLING_SERIES_START,
    dd_add,
    dd_div,
    dd_exp,
    dd_log1pmx,
    dd_mul,
    dd_scaled_log_ratio,
    erfc_double_double,
    half_log_minus_stirling,
    stirling_error_float,
    two_product,
    two_sum,
)
from probstat.numerics.special import digamma, log_minus_digamma

from .constraints import NON_NEGATIVE_FINITE, POSITIVE_FINITE, PROBABILITY

_EPSILON = 8.0 * math.ulp(1.0)
# A series whose terms decay with ratio r leaves a tail of about
# term r/(1-r); stopping at 8 ulp(1) cost up to 30 ulp (r ~ 0.7), so the
# gamma series stops at ulp(1)/16. A continued fraction whose factors have
# reached 8 ulp(1) of one keeps iterating, for at most 32 more steps, until a
# factor lies within half an ulp of one (its rounding noise can keep it
# above any smaller bound).
_SERIES_TOLERANCE = math.ulp(1.0) / 16.0
_FRACTION_FINE = 0.5 * math.ulp(1.0)
_FRACTION_EXTRA = 32
_FPMIN = 1e-300
_MAX_ITERATIONS = 10_000
# Terms needed by the P(a, x) series near x ~ a grow like sqrt(a): the term
# ratio x/(a+k) makes the k-th term about exp(-k^2/(2a)), which falls below
# the convergence threshold after roughly 8.6 sqrt(a) terms.
_SERIES_SQRT_FACTOR = 12.0
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
# Above this shape the log prefactors are formed from Stirling differences
# instead of differences of lgamma values of size ~a log a.
LARGE_SHAPE = 100.0
_TEMME_MIN_SHAPE = 20.0
_TEMME_MAX_OFFSET = 0.3
# Drop the terms c_k(eta) a^-k once a^-k falls below this bound (every
# |c_k(eta)| is below 0.007 for |eta| <= 0.34 and k <= 12).
_TEMME_TERM_FLOOR = 1e-22
# Taylor coefficients (in eta, 26 per row) of Temme's c_k(eta), k = 0..12,
# the coefficients of R_a(eta) = exp(-a eta^2/2)/sqrt(2 pi a) sum c_k(eta)
# a^-k. Derived in exact rational arithmetic from mu - log(1 + mu) =
# eta^2/2 (mu = lambda - 1), c_0 = 1/mu - 1/eta and the recursion
# c_k = c_(k-1)'(eta)/eta + (-1)^k g_k/mu with the Stirling coefficients
# g_k of Gamma(z) (DLMF 8.12.9-8.12.10 and 5.11.3); the derivation is
# repeated in tests/test_numerical_accuracy_follow_ups.py. For |eta| <= 0.34
# (the region |lambda - 1| <= 0.3) the omitted Taylor terms are below 1e-22.
_TEMME_COEFFICIENTS = (
    (
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
        -2.5514193994946248e-11,
        -5.830772132550426e-11,
        2.4361948020667415e-11,
        -5.0276692801141755e-12,
        1.1004392031956135e-13,
        3.371763262400985e-13,
        -1.392388722418162e-13,
        2.8534893807047445e-14,
        -5.139111834242572e-16,
        -1.9752288294349442e-15,
        8.099521156704561e-16,
    ),
    (
        -0.001851851851851852,
        -0.003472222222222222,
        0.0026455026455026454,
        -0.0009902263374485596,
        0.00020576131687242798,
        -4.018775720164609e-07,
        -1.8098550334489977e-05,
        7.64916091608111e-06,
        -1.6120900894563446e-06,
        4.647127802807434e-09,
        1.378633446915721e-07,
        -5.752545603517705e-08,
        1.1951628599778148e-08,
        -1.7543241719747647e-11,
        -1.0091543710600413e-09,
        4.162792991842583e-10,
        -8.56390702649298e-11,
        6.067215101604758e-14,
        7.1624989648114856e-12,
        -2.933186643771437e-12,
        5.996696365683689e-13,
        -2.1671786527323313e-16,
        -4.978339972369262e-14,
        2.0291628823713425e-14,
        -4.13125571381061e-15,
        8.286516239883097e-19,
    ),
    (
        0.004133597883597883,
        -0.0026813271604938273,
        0.0007716049382716049,
        2.0093878600823047e-06,
        -0.0001073665322636516,
        5.2923448829120125e-05,
        -1.2760635188618728e-05,
        3.423578734096138e-08,
        1.3721957309062934e-06,
        -6.298992138380055e-07,
        1.4280614206064242e-07,
        -2.0477098421990866e-10,
        -1.409252991086752e-08,
        6.228974084922022e-09,
        -1.3670488396617114e-09,
        9.428356159014678e-13,
        1.2872252400089318e-10,
        -5.5645956134363323e-11,
        1.197593554636698e-11,
        -4.1689782251838634e-15,
        -1.0940640427884595e-12,
        4.662239946390136e-13,
        -9.905105763906907e-14,
        1.8931876768373515e-17,
        8.859221872591127e-15,
        -3.737820398046405e-15,
    ),
    (
        0.0006494341563786008,
        0.00022947209362139917,
        -0.0004691894943952557,
        0.00026772063206283885,
        -7.561801671883977e-05,
        -2.396505113867297e-07,
        1.1082654115347302e-05,
        -5.6749528269915965e-06,
        1.4230900732435883e-06,
        -2.7861080291528143e-11,
        -1.6958404091930278e-07,
        8.099464905388083e-08,
        -1.9111168485973655e-08,
        2.3928620439808118e-12,
        2.0620131815488797e-09,
        -9.460496661855133e-10,
        2.1541049775774907e-10,
        -1.388823336813903e-14,
        -2.1894761681963938e-11,
        9.790998951171684e-12,
        -2.178219188018096e-12,
        6.208819573407901e-17,
        2.126978363279737e-13,
        -9.344688791517433e-14,
        2.045367122678285e-14,
        -2.58260790403495e-19,
    ),
    (
        -0.0008618882909167117,
        0.0007840392217200666,
        -0.0002990724803031902,
        -1.4638452578843418e-06,
        6.641498215465122e-05,
        -3.968365047179435e-05,
        1.1375726970678419e-05,
        2.507497226237533e-10,
        -1.6954149536558305e-06,
        8.907507532205309e-07,
        -2.292934834000805e-07,
        2.956794137544049e-11,
        2.8865829742708783e-08,
        -1.4189739437803219e-08,
        3.4463580499464896e-09,
        -2.3024517174528067e-13,
        -3.9409233028046403e-10,
        1.86023389685045e-10,
        -4.356323005056618e-11,
        1.278600101629623e-15,
        4.67927502665792e-12,
        -2.149246470613483e-12,
        4.908815614809652e-13,
        -6.33859148489156e-18,
        -5.045332069080094e-14,
        2.2722958222901286e-14,
    ),
    (
        -0.00033679855336635813,
        -6.972813758365857e-05,
        0.0002772753244959392,
        -0.00019932570516188847,
        6.797780477937208e-05,
        1.419062920643967e-07,
        -1.3594048189768693e-05,
        8.018470256334202e-06,
        -2.291481176508095e-06,
        -3.252473551298454e-10,
        3.4652846491085265e-07,
        -1.8447187191171344e-07,
        4.8240967037894184e-08,
        -1.7989466721743514e-14,
        -6.306194500013523e-09,
        3.162417628774568e-09,
        -7.840924253697429e-10,
        5.192679165254041e-15,
        9.358944242306784e-11,
        -4.513426216163278e-11,
        1.0799129993116828e-11,
        -3.661886712685252e-17,
        -1.210902069055155e-12,
        5.680743584990564e-13,
        -1.3249659916340829e-13,
        1.8987240764284076e-19,
    ),
    (
        0.0005313079364639922,
        -0.0005921664373536939,
        0.0002708782096718045,
        7.902353232660328e-07,
        -8.153969367561969e-05,
        5.61168275310625e-05,
        -1.8329116582843375e-05,
        -3.0796134506033047e-09,
        3.465155368803609e-06,
        -2.0291327396058603e-06,
        5.788792863149004e-07,
        2.338630673826657e-13,
        -8.828600746330484e-08,
        4.7435958880408125e-08,
        -1.2545415020710383e-08,
        8.649648858010293e-14,
        1.6846058979264062e-09,
        -8.575492823577594e-10,
        2.1598224929232125e-10,
        -7.613230520476153e-16,
        -2.6639822008536144e-11,
        1.3065700536611057e-11,
        -3.1799163902367977e-12,
        4.710976121367431e-18,
        3.6902800842763465e-13,
        -1.7612674046201426e-13,
    ),
    (
        0.00034436760689237765,
        5.171790908260592e-05,
        -0.00033493161081142234,
        0.0002812695154763237,
        -0.00010976582244684731,
        -1.2741009095484485e-07,
        2.7744451511563645e-05,
        -1.8263488805711332e-05,
        5.7876949497350525e-06,
        4.93875893393627e-10,
        -1.0595367014026043e-06,
        6.166714376110408e-07,
        -1.7562973359060463e-07,
        -1.297447328701544e-12,
        2.695423606288966e-08,
        -1.4578352908731272e-08,
        3.887645959386175e-09,
        -3.881002251019412e-17,
        -5.327994173877286e-10,
        2.7437977643314844e-10,
        -6.995796092070568e-11,
        2.589986387486848e-17,
        8.856689099669639e-12,
        -4.403168815871311e-12,
        1.0865561947091654e-12,
        -2.0467988447416678e-19,
    ),
    (
        -0.0006526239185953094,
        0.0008394987206720873,
        -0.000438297098541721,
        -6.969091458420552e-07,
        0.00016644846642067547,
        -0.00012783517679769218,
        4.629953263691304e-05,
        4.557909867922708e-09,
        -1.0595271125805195e-05,
        6.783342904865167e-06,
        -2.1075476666258803e-06,
        -1.7213731432817144e-11,
        3.773587741611098e-07,
        -2.1867506700122867e-07,
        6.220228804018927e-08,
        6.597703826733e-16,
        -9.590386497425686e-09,
        5.213214492280807e-09,
        -1.3991589583935709e-09,
        5.382058999060575e-16,
        1.9484714275467745e-10,
        -1.0127287556389682e-10,
        2.6077347197254926e-11,
        -5.090418699993299e-18,
        -3.3721464474854593e-12,
        1.6953089140808568e-12,
    ),
    (
        -0.0005967612901927463,
        -7.204895416020011e-05,
        0.0006782308837667328,
        -0.0006401475260262758,
        0.00027750107634328704,
        1.819700838046515e-07,
        -8.479507117068503e-05,
        6.105192082501531e-05,
        -2.1073920183404862e-05,
        -8.858589014125599e-10,
        4.5284535953805374e-06,
        -2.8427815022504407e-06,
        8.708234177864641e-07,
        3.6886101871706966e-12,
        -1.534469519070206e-07,
        8.862466778790695e-08,
        -2.5184812301826817e-08,
        -1.0225912098215092e-14,
        3.896947075815478e-09,
        -2.1267304792235634e-09,
        5.737013552805138e-10,
        -1.8877498501697116e-19,
        -8.093153869465787e-11,
        4.23827232834492e-11,
        -1.1002224534207725e-11,
        2.3327607706802836e-19,
    ),
    (
        0.0013324454494800656,
        -0.0019144384985654776,
        0.0011089369134596636,
        9.9324041226423e-07,
        -0.0005087450129309319,
        0.00042735056665392886,
        -0.00016858853767910798,
        -8.1301893922785e-09,
        4.5284402370562144e-05,
        -3.127053674781734e-05,
        1.044986828530338e-05,
        4.8435226265680926e-11,
        -2.148256587345626e-06,
        1.329369701097492e-06,
        -4.029569309210103e-07,
        -1.756787766632329e-13,
        7.014504316366825e-08,
        -4.040787734999483e-08,
        1.1474026743371964e-08,
        3.964274685356394e-18,
        -1.7804938269892715e-09,
        9.748026254873165e-10,
        -2.6405338676507616e-10,
        5.79487516340376e-18,
        3.764774955354384e-11,
        -1.983951296757828e-11,
    ),
    (
        0.001579727660730835,
        0.00016251626278391583,
        -0.0020633421035543276,
        0.00213896861856891,
        -0.0010108559391263003,
        -3.99127055299192e-07,
        0.0003623502508476469,
        -0.00028143901463712157,
        0.00010449513336495887,
        2.12114184918303e-09,
        -2.5779417251947842e-05,
        1.7281818956040464e-05,
        -5.641377387290428e-06,
        -1.1024320105776174e-11,
        1.1223224418895174e-06,
        -6.869339637952674e-07,
        2.0653236975414888e-07,
        4.6714772409838506e-14,
        -3.5609886164949055e-08,
        2.0470855345905963e-08,
        -5.809173863328336e-09,
        -1.3328212875828647e-16,
        9.035460439133513e-10,
        -4.959878251733084e-10,
        1.3481607129399748e-10,
        -1.670378498659395e-21,
    ),
    (
        -0.004072512119514016,
        0.00640336283380807,
        -0.004041016108167662,
        -2.1837328028662328e-06,
        0.002174044180125464,
        -0.001970044051841889,
        0.0008359546974796246,
        1.9445447567109655e-08,
        -0.000257793871204217,
        0.00019009987368139304,
        -6.769649993743896e-05,
        -1.4440629666426571e-10,
        1.5712512518742267e-05,
        -1.0304008744776894e-05,
        3.304517767401387e-06,
        7.982976024232571e-13,
        -6.4097794149313e-07,
        3.8894624761300054e-07,
        -1.161834764494887e-07,
        -2.8168086305964423e-15,
        1.9878012911297094e-08,
        -1.1407719956357511e-08,
        3.2355857064185554e-09,
        4.1759462466484876e-20,
        -5.042311271810582e-10,
        2.7740247286170716e-10,
    ),
)

def stirling_error(a: float) -> float:
    """Return ``lgamma(a) - [(a - 1/2) log a - a + log sqrt(2 pi)]``.

    For ``a >= 15`` the asymptotic series
    ``sum_(k=1..7) B_(2k) / (2k (2k-1) a^(2k-1))`` is summed directly (its
    truncation error is below ``4e-18`` there); smaller arguments use the
    defining difference, whose terms are then of modest size.

    Args:
        a: Finite positive argument.

    Returns:
        The Stirling-series remainder, positive and of order ``1/(12 a)``
        for large ``a``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if a < STIRLING_SERIES_START:
        return math.lgamma(a) - (a - 0.5) * math.log(a) + a - _LOG_SQRT_TWO_PI
    return stirling_error_float(a)


def log1pmx(t: float) -> float:
    """Return ``log(1 + t) - t`` without cancellation for small ``|t|``.

    For ``|t| <= 1/2`` the identity ``log(1+t) - t = -t y + 2 sum_(k>=1)
    y^(2k+1)/(2k+1)`` with ``y = t/(2+t)`` is summed (``|y| <= 1/3``, so the
    series gains about one decimal digit per term); otherwise the direct
    difference is used.

    Args:
        t: Real argument greater than ``-1``.

    Returns:
        Non-positive value ``log(1 + t) - t``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992] (function ``rlog1``).

    """
    if abs(t) > 0.5:
        return math.log1p(t) - t
    y = t / (2.0 + t)
    square = y * y
    power = y * square
    series = 0.0
    denominator = 3.0
    while True:
        term = power / denominator
        series += term
        if abs(term) <= 1e-17 * abs(series):
            break
        power *= square
        denominator += 2.0
    return 2.0 * series - t * y


def log_gamma_kernel(a: float, x: float, rate: float = 1.0) -> float:
    """Return ``a log y - y - lgamma(a)`` at ``y = rate x``, accurately for large ``a``.

    Below :data:`LARGE_SHAPE` the three terms are added directly. Above it the
    equivalent form ``a [log(y/a) - (y/a - 1)] + log(a)/2 - log sqrt(2 pi) -
    stirling_error(a)`` is used: every term is then of order one near the
    mode instead of order ``a log a``, so ``exp`` of the result keeps full
    relative accuracy for shapes up to ``1e15`` and beyond. When
    ``rate != 1`` the offset ``rate x - a`` is formed in exact rational
    arithmetic, so the product ``rate x`` is never rounded first.

    Args:
        a: Finite positive shape.
        x: Finite positive point.
        rate: Finite positive multiplier; ``rate x`` must be finite.

    Returns:
        Logarithm of ``y^a exp(-y) / Gamma(a)`` with ``y = rate x``.

    Equation:
        [EQ-NUM-STIRLING-001], [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11], [REF-DIDONATO-MORRIS-1992].

    """
    if a < LARGE_SHAPE:
        if rate == 1.0:
            return -x + a * math.log(x) - math.lgamma(a)
        scaled = rate * x
        return -scaled + a * (math.log(rate) + math.log(x)) - math.lgamma(a)
    if rate == 1.0:
        offset = x - a
    else:
        offset = float(Fraction(rate) * Fraction(x) - Fraction(a))
    t = offset / a
    if abs(t) <= 0.5:
        core = a * log1pmx(t)
    else:
        core = a * (math.log(rate) + math.log(x) - math.log(a)) - offset
    return core + 0.5 * math.log(a) - _LOG_SQRT_TWO_PI - stirling_error(a)


def log_gamma_ratio(a: float, b: float) -> float:
    """Return ``lgamma(a + b) - lgamma(a)`` accurately when ``a`` is large.

    Above :data:`LARGE_SHAPE` the identity ``b log a - b + (a + b - 1/2)
    log(1 + b/a) + stirling_error(a + b) - stirling_error(a)`` replaces the
    difference of two numbers of size ``a log a``.

    Args:
        a: Finite positive argument.
        b: Finite positive increment.

    Returns:
        ``log(Gamma(a + b) / Gamma(a))``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if a < LARGE_SHAPE:
        return math.lgamma(a + b) - math.lgamma(a)
    return (
        b * math.log(a)
        - b
        + (a + b - 0.5) * math.log1p(b / a)
        + stirling_error(a + b)
        - stirling_error(a)
    )


def log_beta_kernel(
    a: float,
    b: float,
    x: float,
    complement: float,
    offset: Fraction | None = None,
) -> float:
    """Return ``a log x + b log(1 - x) - log B(a, b)`` without cancellation.

    Below :data:`LARGE_SHAPE` (for the larger shape) the terms are added
    directly. Otherwise, with ``1 + t = x (a+b)/a`` and ``1 + s = (1-x)
    (a+b)/b`` (so ``a t + b s = 0``), the kernel equals
    ``a [log(1+t) - t] + b [log(1+s) - s] + log(ab/(a+b))/2 - log sqrt(2 pi)
    - stirling_error(a) - stirling_error(b) + stirling_error(a+b)``. Both
    bracketed terms are non-positive, so nothing cancels. The shared offset
    ``d = x (a+b) - a`` is formed in exact rational arithmetic from whichever
    of ``x`` and ``1 - x`` is the smaller (and hence more accurate) input.

    Args:
        a: Finite positive first shape.
        b: Finite positive second shape.
        x: Point strictly inside ``(0, 1)``.
        complement: ``1 - x`` formed without cancellation.
        offset: Optional exact value of ``x (a + b) - a`` supplied by callers
            that know ``x`` only as a rounded ratio of exact inputs (for
            example the F law, where it equals ``ab (f - 1)/(af + b)``).

    Returns:
        Logarithm of ``x^a (1 - x)^b / B(a, b)``.

    Equation:
        [EQ-NUM-STIRLING-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-DIDONATO-MORRIS-1992].

    """
    log_point = math.log(x) if x <= 0.5 else math.log1p(-complement)
    log_other = math.log(complement) if complement <= 0.5 else math.log1p(-x)
    if max(a, b) < LARGE_SHAPE:
        return (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * log_point
            + b * log_other
        )
    first, second = Fraction(a), Fraction(b)
    if offset is None:
        total = first + second
        if x <= 0.5:
            offset = Fraction(x) * total - first
        else:
            offset = second - Fraction(complement) * total
    t = float(offset / first)
    s = float(-offset / second)
    if abs(t) <= 0.5:
        first_part = a * log1pmx(t)
    else:
        # log(1 + t) = log(x (a+b)/a); the logarithm of x is already accurate.
        first_part = a * (log_point + math.log1p(b / a) - t)
    if abs(s) <= 0.5:
        second_part = b * log1pmx(s)
    else:
        second_part = b * (log_other + math.log1p(a / b) - s)
    return (
        first_part
        + second_part
        + 0.5 * (math.log(a) + math.log(b) - math.log(a + b))
        - _LOG_SQRT_TWO_PI
        - stirling_error(a)
        - stirling_error(b)
        + stirling_error(a + b)
    )


def log_beta_function(a: float, b: float) -> float:
    """Return ``log B(a, b)`` with Stirling differences for large arguments.

    Args:
        a: Finite positive first argument.
        b: Finite positive second argument.

    Returns:
        ``lgamma(a) + lgamma(b) - lgamma(a + b)``; when the larger argument
        exceeds :data:`LARGE_SHAPE` the large-argument pair is combined
        through :func:`log_gamma_ratio` or the Stirling form, so the result
        keeps absolute accuracy near ``1e-15`` even when it is small compared
        with ``a log a``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.11 and 5.12].

    """
    small, large = min(a, b), max(a, b)
    if large < LARGE_SHAPE:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if small < LARGE_SHAPE:
        return math.lgamma(small) - log_gamma_ratio(large, small)
    return (
        _LOG_SQRT_TWO_PI
        + (small - 0.5) * math.log(small)
        + (large - 0.5) * math.log(large)
        - (a + b - 0.5) * math.log(a + b)
        + stirling_error(small)
        + stirling_error(large)
        - stirling_error(a + b)
    )


# Below this exponent the prefactors underflow (``exp(-745.2)`` is the
# smallest subnormal), so the float exponent is used as it is.
_PREFACTOR_UNDERFLOW = -760.0
# Double-double products and quotients stay exact for magnitudes in this
# range; arguments outside it (where no cancellation is at stake for the
# prefactors anyway) keep the float exponent.
_SAFE_RANGE = (1e-280, 1e280)


def _in_safe_range(*values: float) -> bool:
    """Return whether every value lies in the double-double safe range.

    Args:
        *values: Positive floats to check.

    Returns:
        ``True`` when all values are within ``[1e-280, 1e280]``.

    References:
        [EQ-NUM-DD-001], [REF-DEKKER-1971].

    """
    low, high = _SAFE_RANGE
    return all(low <= value <= high for value in values)


def gamma_prefactor(a: float, x: float) -> float:
    """Return ``x^a exp(-x) / Gamma(a)`` with a double-double exponent.

    The exponent is regrouped as ``a log1pmx((x - a)/a) + log(a)/2 -
    log sqrt(2 pi) - stirling_error(a)`` (valid for every ``a > 0``) and each
    term is formed in double-double arithmetic, with ``stirling_error`` of
    shapes below 15 lifted by the recurrence of ``Gamma``. The float
    exponent of :func:`log_gamma_kernel` has an absolute error of about
    ``2^-53`` times the size of its largest cancelling term (``a log x``,
    ``x``, ``lgamma(a)``), which ``exp`` turns into the same relative error:
    1400 ulp at ``(a, x) = (200, 605)``. Here the exponent's error is below
    ``1e-18``, so the prefactor carries one ``exp`` rounding.

    Args:
        a: Finite positive shape.
        x: Finite positive point.

    Returns:
        The prefactor of the incomplete gamma series and continued fraction.

    Equation:
        [EQ-NUM-GAMMA-003], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1986], [REF-DEKKER-1971],
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    estimate = log_gamma_kernel(a, x)
    if estimate < _PREFACTOR_UNDERFLOW or not _in_safe_range(a, x, x / a):
        return math.exp(estimate)
    exponent = dd_add(
        dd_scaled_log_ratio(a, (x, 0.0)), half_log_minus_stirling((a, 0.0))
    )
    return dd_exp(dd_add(exponent, (-LOG_SQRT_TWO_PI[0], -LOG_SQRT_TWO_PI[1])))


def beta_prefactor(a: float, b: float, x: float, complement: float) -> float:
    """Return ``x^a (1 - x)^b / B(a, b)`` with a double-double exponent.

    With ``1 + t = x (a+b)/a`` and ``1 + s = (1-x)(a+b)/b`` the exponent is
    ``a log1pmx(t) + b log1pmx(s) + log(ab/(a+b))/2 - log sqrt(2 pi) -
    stirling_error(a) - stirling_error(b) + stirling_error(a+b)`` for all
    positive shapes (``a t + b s = 0``), every term in double-double
    arithmetic. The smaller of ``x`` and ``1 - x`` is taken as exact and the
    other is formed from it without rounding.

    Args:
        a: Finite positive first shape.
        b: Finite positive second shape.
        x: Point strictly inside ``(0, 1)``.
        complement: ``1 - x`` formed without cancellation.

    Returns:
        The prefactor of the incomplete beta continued fraction.

    Equation:
        [EQ-NUM-BETA-003], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-DEKKER-1971],
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    estimate = log_beta_kernel(a, b, x, complement)
    total = two_sum(a, b)
    if estimate < _PREFACTOR_UNDERFLOW or not _in_safe_range(
        a, b, x, complement, total[0], x * total[0] / a, complement * total[0] / b
    ):
        return math.exp(estimate)
    if x <= 0.5:
        point = (x, 0.0)
        other = two_sum(1.0, -x)
    else:
        other = (complement, 0.0)
        point = two_sum(1.0, -complement)
    exponent = dd_add(
        dd_scaled_log_ratio(a, dd_mul(point, total)),
        dd_scaled_log_ratio(b, dd_mul(other, total)),
    )
    shapes = dd_add(
        half_log_minus_stirling((a, 0.0)), half_log_minus_stirling((b, 0.0))
    )
    combined = half_log_minus_stirling(total)
    shapes = dd_add(shapes, (-combined[0], -combined[1]))
    exponent = dd_add(exponent, shapes)
    return dd_exp(dd_add(exponent, (-LOG_SQRT_TWO_PI[0], -LOG_SQRT_TWO_PI[1])))


def regularized_gamma_p(a: float, x: float) -> float:
    """Return the lower regularized incomplete gamma function ``P(a, x)``.

    A convergent series is used for ``x < a + 1``; otherwise the complement of
    a modified-Lentz continued fraction for ``Q(a, x)`` is used. For
    ``a >= 20`` and ``|x - a| <= 0.3 a``, where both expansions would need
    ``O(sqrt(a))`` terms, Temme's uniform asymptotic expansion is used. The
    prefactor ``x^a e^-x / Gamma(a)`` has a double-double exponent
    (:func:`gamma_prefactor`), so the error does not grow with the shape.

    Args:
        a: Finite positive shape parameter.
        x: Finite non-negative evaluation point.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the selected expansion does not converge.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-GAMMA-003], [EQ-NUM-GAMMA-004].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2, 8.9, and 8.12], [REF-TEMME-1979].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 0.0
    if _use_temme(shape, point):
        return _gamma_temme(shape, point)[0]
    if point < shape + 1.0:
        return _gamma_series_p(shape, point)
    return _clamp_probability(1.0 - _gamma_continued_fraction_q(shape, point))


def regularized_gamma_q(a: float, x: float) -> float:
    """Return the upper regularized incomplete gamma function ``Q(a, x)``.

    Args:
        a: Finite positive shape parameter.
        x: Finite non-negative evaluation point.

    Returns:
        The complement ``1 - P(a, x)`` evaluated through the numerically
        appropriate expansion (series, continued fraction, or Temme's uniform
        expansion for ``a >= 20`` near ``x = a``) and clamped to ``[0, 1]``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the selected expansion does not converge.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-GAMMA-003], [EQ-NUM-GAMMA-004].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2, 8.9, and 8.12], [REF-TEMME-1979].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 1.0
    if _use_temme(shape, point):
        return _gamma_temme(shape, point)[1]
    if point < shape + 1.0:
        return _clamp_probability(1.0 - _gamma_series_p(shape, point))
    return _gamma_continued_fraction_q(shape, point)


def _use_temme(a: float, x: float) -> bool:
    """Return whether Temme's uniform expansion should evaluate ``P, Q``.

    Args:
        a: Validated positive shape.
        x: Validated positive point.

    Returns:
        ``True`` for ``a >= 20`` and ``|x - a| <= 0.3 a``, where the series
        and the continued fraction need ``O(sqrt(a))`` terms whose rounding
        errors accumulate (measured: 26 ulp at ``a = 200``, 570 ulp at
        ``a = 1e5``).

    References:
        [EQ-NUM-GAMMA-004], [REF-TEMME-1979], [REF-DIDONATO-MORRIS-1986].

    """
    return a >= _TEMME_MIN_SHAPE and abs(x - a) <= _TEMME_MAX_OFFSET * a


def _gamma_temme(a: float, x: float) -> tuple[float, float]:
    """Evaluate ``(P(a, x), Q(a, x))`` by Temme's uniform asymptotic expansion.

    With ``lambda = x/a`` and ``eta = sign(lambda - 1) sqrt(2 (lambda - 1 -
    log lambda))``, ``Q = erfc(w)/2 + R`` and ``P = erfc(-w)/2 - R`` with
    ``w = eta sqrt(a/2)`` and ``R = exp(-w^2) / sqrt(2 pi a) sum_(k<=12)
    c_k(eta) a^-k``. ``w^2 = -a log1pmx((x - a)/a)`` is formed in
    double-double arithmetic, so neither ``exp(-w^2)`` nor ``erfc(w)``
    inherits the ``2 w^2``-fold amplification of a rounded argument; the
    coefficients are summed from their Taylor series in ``eta``. For
    ``a >= 20`` the last included term ``c_12/a^12`` is below ``2e-18``
    relative to ``R``, and ``|R| <= Q/9`` in this region.

    Args:
        a: Shape with ``a >= 20``.
        x: Point with ``|x - a| <= 0.3 a``.

    Returns:
        The pair ``(P, Q)``, each clamped to ``[0, 1]``.

    Equation:
        [EQ-NUM-GAMMA-004], [EQ-NUM-DD-003].

    References:
        [REF-TEMME-1979], [REF-NIST-DLMF-GAMMA, Section 8.12],
        [REF-DEKKER-1971].

    """
    shape = (a, 0.0)
    t = dd_div(two_sum(x, -a), shape)
    exponent = dd_mul(shape, dd_log1pmx(t))
    square = (-exponent[0], -exponent[1])
    root = math.sqrt(square[0])
    if root > 0.0:
        product, error = two_product(root, root)
        low = (((square[0] - product) - error) + square[1]) / (2.0 * root)
    else:
        low = 0.0
    sign = 1.0 if t[0] >= 0.0 else -1.0
    argument = (sign * root, sign * low)
    eta = argument[0] / math.sqrt(0.5 * a)
    inverse = 1.0 / a
    power = 1.0
    total = 0.0
    for row in _TEMME_COEFFICIENTS:
        value = 0.0
        for coefficient in reversed(row):
            value = value * eta + coefficient
        total += value * power
        power *= inverse
        if power < _TEMME_TERM_FLOOR:
            break
    remainder = dd_exp(exponent) / math.sqrt(2.0 * math.pi * a) * total
    upper = 0.5 * erfc_double_double(argument) + remainder
    lower = 0.5 * erfc_double_double((-argument[0], -argument[1])) - remainder
    return _clamp_probability(lower), _clamp_probability(upper)


def _gamma_series_p(a: float, x: float) -> float:
    """Evaluate ``P(a, x)`` through its convergent power series.

    Args:
        a: Validated positive shape.
        x: Validated positive point satisfying the series branch criterion.

    Returns:
        Lower regularized incomplete gamma value.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration
            cap ``10_000 + 12 sqrt(a)``. Near ``x ~ a`` the series needs about
            ``8.6 sqrt(a)`` terms, so the work grows like ``sqrt(a)`` for very
            large shapes; the prefactor ``x^a e^-x / Gamma(a)`` comes from
            :func:`log_gamma_kernel`.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.9].

    """
    term = 1.0 / a
    total = term
    denominator = a
    limit = _MAX_ITERATIONS + int(_SERIES_SQRT_FACTOR * math.sqrt(a))
    for _ in range(limit):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) <= abs(total) * _SERIES_TOLERANCE:
            return _clamp_probability(total * gamma_prefactor(a, x))
    raise ArithmeticError("regularized-gamma series did not converge")


def _gamma_continued_fraction_q(a: float, x: float) -> float:
    """Evaluate ``Q(a, x)`` with a stabilized continued fraction.

    Args:
        a: Validated positive shape.
        x: Validated positive point satisfying the continued-fraction criterion.

    Returns:
        Upper regularized incomplete gamma value.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration cap.

    Equation:
        [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.9].

    """
    b = x + 1.0 - a
    if abs(b) < _FPMIN:
        b = _FPMIN
    c = 1.0 / _FPMIN
    d = 1.0 / b
    fraction = d
    limit = _MAX_ITERATIONS + int(_SERIES_SQRT_FACTOR * math.sqrt(a))
    settled: int | None = None
    for index in range(1, limit + 1):
        coefficient = -float(index) * (float(index) - a)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= _EPSILON:
            settled = index if settled is None else settled
            if abs(delta - 1.0) <= _FRACTION_FINE or index - settled >= _FRACTION_EXTRA:
                return _clamp_probability(gamma_prefactor(a, x) * fraction)
    raise ArithmeticError("regularized-gamma continued fraction did not converge")


def _clamp_probability(value: float) -> float:
    """Clamp a nearly valid computed probability to the unit interval.

    Args:
        value: Floating result from a converged special-function expansion.

    Returns:
        ``value`` restricted to ``[0, 1]``.

    References:
        [REF-NIST-DLMF-GAMMA]. This is a roundoff guard and introduces no new
        mathematical equation.

    """
    return max(0.0, min(1.0, value))


def regularized_beta(x: float, a: float, b: float) -> float:
    """Return the regularized incomplete beta function ``I_x(a, b)``.

    The implementation chooses the direct or symmetry-transformed continued
    fraction so the smaller tail is evaluated before any complement; the
    prefactor ``x^a (1-x)^b / B(a, b)`` has a double-double exponent
    (:func:`beta_prefactor`).

    Args:
        x: Finite evaluation point in ``[0, 1]``.
        a: Finite positive first shape parameter.
        b: Finite positive second shape parameter.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the continued fraction does not converge.

    Equation:
        [EQ-NUM-BETA-001], [EQ-NUM-BETA-003].

    References:
        [REF-NIST-DLMF-BETA, Sections 8.17 and 8.18].

    """
    point = PROBABILITY.validate(x, name="x")
    return regularized_beta_pair(point, 1.0 - point, a, b)


def regularized_beta_pair(x: float, complement: float, a: float, b: float) -> float:
    """Return ``I_x(a, b)`` when ``x`` and ``1 - x`` are both supplied.

    Callers that can form ``1 - x`` directly (for example ``z^2/(nu + z^2)``
    beside ``nu/(nu + z^2)``) pass it as ``complement``, so neither the
    logarithmic prefactor nor the symmetry-branch continued fraction has to
    recover a small quantity from a difference with one.

    Args:
        x: Finite evaluation point in ``[0, 1]``.
        complement: ``1 - x`` evaluated without cancellation.
        a: Finite positive first shape parameter.
        b: Finite positive second shape parameter.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain or
            ``|x + complement - 1| > 1e-12``.
        ArithmeticError: If the continued fraction does not converge.

    Equation:
        [EQ-NUM-BETA-001], [EQ-NUM-BETA-003].

    References:
        [REF-NIST-DLMF-BETA, Sections 8.17 and 8.18].

    """
    point = PROBABILITY.validate(x, name="x")
    other = PROBABILITY.validate(complement, name="complement")
    first_shape = POSITIVE_FINITE.validate(a, name="a")
    second_shape = POSITIVE_FINITE.validate(b, name="b")
    if abs(point + other - 1.0) > 1e-12:
        raise ValueError("complement must equal 1 - x")
    if point == 0.0:
        return 0.0
    if other == 0.0:
        return 1.0
    series = _beta_large_shape(first_shape, second_shape, point, other)
    if series is not None:
        return series
    front = beta_prefactor(first_shape, second_shape, point, other)
    split = (first_shape + 1.0) / (first_shape + second_shape + 2.0)
    if point < split:
        result = (
            front
            * _beta_continued_fraction(first_shape, second_shape, point)
            / first_shape
        )
        return _clamp_probability(result)
    reflected = (
        front
        * _beta_continued_fraction(second_shape, first_shape, other)
        / second_shape
    )
    return _clamp_probability(1.0 - reflected)


_LOG_SINHC_TERMS = 80
_BETA_SERIES_MIN_SHAPE = 1e4
# -log(1 - t) <= 1, the region where the sinh series converges quickly.
_BETA_SERIES_MAX_POINT = -math.expm1(-1.0)
_log_sinhc_cache: list[float] = []


def _beta_large_shape(a: float, b: float, x: float, y: float) -> float | None:
    """Return ``I_x(a, b)`` from the large-shape gamma series when it applies.

    The series of :func:`_beta_gamma_series` is used when the larger shape is
    at least ``1e4``, the point on the side of the smaller shape satisfies
    ``-log(1 - t) <= 1``, and the first correction ratio
    ``|s - 1| s (s + 1) / (24 N^2)`` is at most one (so for ``s`` up to about
    ``(24 L^2)^(1/3)``; for example ``s <= 1.3e3`` at ``L = 1e4``). There
    the continued fraction converges slowly near ``x ~ a/(a+b)`` and its
    local stopping test (``|delta - 1| <= 8 eps``) is met while many
    factors ``1 + O(1/a)`` remain, which cost up to three significant
    digits for ``a ~ 1e13``. Of the pair ``I_t(s, L)`` and its complement
    the smaller one is computed and the other obtained by subtraction from
    one.

    Args:
        a: Validated first shape.
        b: Validated second shape.
        x: Point strictly inside ``(0, 1)``.
        y: ``1 - x`` formed without cancellation.

    Returns:
        ``I_x(a, b)``, or ``None`` when the series does not apply.

    Equation:
        [EQ-NUM-BETA-002].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    if max(a, b) < _BETA_SERIES_MIN_SHAPE:
        return None
    if a >= b:
        large, small, point, complement = a, b, y, x
    else:
        large, small, point, complement = b, a, x, y
    if point > _BETA_SERIES_MAX_POINT:
        return None
    big_n = large + 0.5 * (small - 1.0)
    if abs(small - 1.0) * small * (small + 1.0) > 24.0 * big_n * big_n:
        # The first correction e_1 Gamma(s+2)/(Gamma(s) N^2) exceeds one:
        # both shapes are large and the series would need too many terms.
        return None
    lower_side = _beta_gamma_series(large, small, point, complement, upper=True)
    if lower_side is None:
        return None
    if lower_side > 0.5:
        other = _beta_gamma_series(large, small, point, complement, upper=False)
        if other is None:
            return None
        lower_side = 1.0 - other
        upper_side = other
    else:
        upper_side = 1.0 - lower_side
    # lower_side = I_t(small, large); upper_side = I_(1-t)(large, small).
    return _clamp_probability(upper_side if a >= b else lower_side)


def _log_sinhc_coefficients() -> list[float]:
    """Return ``l_k = B_(2k) / (2k (2k)!)``, the series of ``log(sinh(v/2)/(v/2))``.

    ``log(sinh(v/2)/(v/2)) = sum_(k>=1) l_k v^(2k)`` for ``|v| < 2 pi``. The
    Bernoulli numbers are generated once in exact rational arithmetic
    (Akiyama-Tanigawa recurrence) and cached as floats.

    Returns:
        Coefficients ``l_1, ..., l_80`` (``l_1 = 1/24``, ``l_2 = -1/2880``).

    Equation:
        [EQ-NUM-BETA-002].

    References:
        [REF-NIST-DLMF, Sections 4.19 and 24.2] (the hyperbolic form follows
        from ``log(sin z / z)`` with ``z -> i z``).

    """
    if _log_sinhc_cache:
        return _log_sinhc_cache
    order = 2 * _LOG_SINHC_TERMS
    row = [Fraction(0)] * (order + 1)
    bernoulli = [Fraction(0)] * (order + 1)
    for m in range(order + 1):
        row[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            row[j - 1] = j * (row[j - 1] - row[j])
        bernoulli[m] = row[0]
    factorial = Fraction(1)
    for k in range(1, _LOG_SINHC_TERMS + 1):
        factorial *= (2 * k - 1) * (2 * k)
        _log_sinhc_cache.append(float(bernoulli[2 * k] / (2 * k * factorial)))
    return _log_sinhc_cache


def _beta_gamma_series(
    large: float, small: float, point: float, complement: float, *, upper: bool
) -> float | None:
    """Return ``I_t(s, L)`` or ``I_(1-t)(L, s)`` by a gamma-function series.

    With ``v = -log(1 - u)`` the beta integral becomes
    ``I_t(s, L) = Gamma(s+L)/(Gamma(s) Gamma(L)) int_0^w v^(s-1)
    phi(v) exp(-N v) dv`` with ``w = -log(1 - t)``, ``N = L + (s-1)/2`` and
    ``phi(v) = (sinh(v/2)/(v/2))^(s-1) = sum_n e_n v^(2n)`` (convergent for
    ``|v| < 2 pi``). Integrating term by term gives the convergent series

    ``I_t(s, L) = Gamma(s+L)/(Gamma(L) N^s) sum_n e_n [Gamma(s+2n)/Gamma(s)]
    N^(-2n) P(s + 2n, N w)``,

    and the same terms with ``Q`` in place of ``P`` give the complementary
    integral over ``[w, inf)``, an asymptotic expansion in ``1/N`` that is
    truncated before its terms grow (the large-``L`` expansion behind
    DiDonato and Morris's ``BGRAT``). The ``P`` series has non-negative terms
    for ``s >= 1``. The prefactor's logarithm ``L log1pmx(s/L) + (s - 1/2)
    log(1 + s/L) - s log(1 + (s-1)/(2L)) + stirling(L+s) - stirling(L)``
    is formed from small terms only, so no ``s log L``-sized cancellation
    occurs. ``e_n`` follow from ``n e_n = (s - 1) sum_(k=1..n) k l_k
    e_(n-k)`` with the coefficients of :func:`_log_sinhc_coefficients`.

    Args:
        large: Large shape ``L``.
        small: Small shape ``s``.
        point: Small-side point ``t`` with ``-log(1 - t) <= 1``.
        complement: ``1 - t`` formed without cancellation.
        upper: Return ``I_t(s, L)`` (``P`` series) when true, else
            ``I_(1-t)(L, s)`` (``Q`` series).

    Returns:
        The requested value (accurate in relative terms when it is the
        smaller of the pair), or ``None`` if the series does not settle within
        80 terms.

    Equation:
        [EQ-NUM-BETA-002], [EQ-NUM-GAMMA-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    big_n = large + 0.5 * (small - 1.0)
    w = -math.log1p(-point) if point <= 0.5 else -math.log(complement)
    z = big_n * w
    ratio = small / large
    log_prefactor = (
        large * log1pmx(ratio)
        + (small - 0.5) * math.log1p(ratio)
        - small * math.log1p(0.5 * (small - 1.0) / large)
        + stirling_error(large + small)
        - stirling_error(large)
    )
    coefficients = _log_sinhc_coefficients()
    exponent = small - 1.0
    e_values = [1.0]
    weight = 1.0
    total = 0.0
    previous = math.inf
    for n in range(_LOG_SINHC_TERMS):
        shape = small + 2.0 * n
        if n > 0:
            following = math.fsum(
                k * coefficients[k - 1] * e_values[n - k] for k in range(1, n + 1)
            )
            e_values.append(exponent * following / n)
            weight *= (shape - 2.0) * (shape - 1.0) / (big_n * big_n)
        factor = e_values[n] * weight
        if upper:
            term = factor * regularized_gamma_p(shape, z)
        else:
            term = factor * regularized_gamma_q(shape, z)
            if n > 1 and abs(term) > abs(previous):
                break
            previous = term
        total += term
        if n > 0 and abs(term) <= 1e-17 * abs(total):
            break
    else:
        return None
    return _clamp_probability(math.exp(log_prefactor) * total)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued-fraction factor.

    Args:
        a: Validated positive first shape.
        b: Validated positive second shape.
        x: Validated point strictly between zero and one.

    Returns:
        Continued-fraction multiplier used by ``I_x(a,b)``.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration cap.

    Equation:
        [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17].

    """
    total_shape = a + b
    a_plus_one = a + 1.0
    a_minus_one = a - 1.0
    c = 1.0
    d = 1.0 - total_shape * x / a_plus_one
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    fraction = d
    settled: int | None = None
    for index in range(1, _MAX_ITERATIONS + 1):
        doubled = 2.0 * index
        coefficient = (
            index * (b - index) * x / ((a_minus_one + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        fraction *= d * c

        coefficient = -(
            (a + index)
            * (total_shape + index)
            * x
            / ((a + doubled) * (a_plus_one + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= _EPSILON:
            settled = index if settled is None else settled
            if abs(delta - 1.0) <= _FRACTION_FINE or index - settled >= _FRACTION_EXTRA:
                return fraction
    raise ArithmeticError("regularized-beta continued fraction did not converge")


_ENTROPY_ASYMPTOTIC_SHAPE = 15.0


def beta_entropy_nats(a: float, b: float) -> float:
    """Return the ``Beta(a, b)`` differential entropy in nats.

    ``h = log B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)``.
    When ``max(a, b) >= 15`` each large argument ``x`` is expanded as
    ``log Gamma(x) = (x - 1/2) log x - x + log sqrt(2 pi) + e(x)`` and
    ``psi(x) = log x - L(x)`` (Stirling remainder ``e``, ``L = log - psi``),
    so the terms of size ``x log x`` and ``x`` cancel analytically; the direct
    sum loses all digits for shapes near ``1e15``.

    Args:
        a: Positive first shape.
        b: Positive second shape.

    Returns:
        Differential entropy (``<= 0``, zero for the uniform ``a = b = 1``).

    Raises:
        TypeError: If a shape is not real.
        ValueError: If a shape is not positive and finite.

    Equation:
        [EQ-INFO-DH-003], [EQ-NUM-STIRLING-001], [EQ-NUM-DIGAMMA-001].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA, Sections
        5.2 and 5.11].

    """
    first = POSITIVE_FINITE.validate(a, name="alpha")
    second = POSITIVE_FINITE.validate(b, name="beta")
    small, large = min(first, second), max(first, second)
    if large < _ENTROPY_ASYMPTOTIC_SHAPE:
        log_beta = (
            math.lgamma(first) + math.lgamma(second) - math.lgamma(first + second)
        )
        return (
            log_beta
            - (first - 1.0) * digamma(first)
            - (second - 1.0) * digamma(second)
            + (first + second - 2.0) * digamma(first + second)
        )
    total = small + large
    # Large-argument part: (log x)/2 + e(x) + (x - 1) L(x) for the larger
    # shape, minus 3 (log s)/2 + e(s) + (s - 2) L(s) for the sum.
    value = (
        0.5 * math.log(large)
        - 1.5 * math.log(total)
        + stirling_error(large)
        - stirling_error(total)
        + (large - 1.0) * log_minus_digamma(large)
        - (total - 2.0) * log_minus_digamma(total)
    )
    if small >= _ENTROPY_ASYMPTOTIC_SHAPE:
        # The linear terms -a - b + s cancel exactly; one log sqrt(2 pi) stays.
        value += (
            _LOG_SQRT_TWO_PI
            + 0.5 * math.log(small)
            + stirling_error(small)
            + (small - 1.0) * log_minus_digamma(small)
        )
    else:
        # The linear terms -large + s equal the small shape exactly.
        value += math.lgamma(small) - (small - 1.0) * digamma(small) + small
    return value
