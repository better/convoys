from autograd.numpy import exp, log
from numpy import vectorize
import mpmath
from autograd.extend import primitive, defvjp
from autograd.scipy.special import gamma

# See https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives

def upper_inc_gamma(k, x):
    return float(mpmath.gammainc(k, a=float(x)))


def T3(s, x):
    # m=3, n=0, p=2, q=3
    return float(mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=float(x)))


upper_inc_gamma_vectorized = vectorize(upper_inc_gamma)
T3_vectorized = vectorize(T3)


@primitive
def upper_inc_gamma_primitive(k, x):
    return upper_inc_gamma_vectorized(k, x)


defvjp(
    upper_inc_gamma_primitive,
    lambda ans, k, x: lambda g: g * (log(x) * upper_inc_gamma_vectorized(k, x) + x * T3_vectorized(k, x)),
    lambda ans, k, x: lambda g: g * -1 * exp(-x) * x ** (k-1),
)


def gammainc(k, x):
    return 1.0 - upper_inc_gamma_primitive(k, x) / gamma(k)
