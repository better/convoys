import math
from numpy import exp, log
import mpmath
from autograd import grad
from autograd.extend import primitive, defvjp
from autograd.scipy.special import gamma
import scipy.special

# See https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives

@primitive
def upper_inc_gamma(k, x):
    return float(mpmath.gammainc(k, a=x))


def T3(s, x):
    # m=3, n=0, p=2, q=3
    return float(mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=x))


defvjp(
    upper_inc_gamma,
    lambda ans, k, x: lambda g: g * (log(x) * upper_inc_gamma(k, x) + x * T3(k, x)),
    lambda ans, k, x: lambda g: g * -1 * exp(-x) * x ** (k-1),
)


def gammainc(k, x):
    return 1.0 - upper_inc_gamma(k, x) / gamma(k)
