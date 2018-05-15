from autograd.extend import primitive, defvjp
from scipy.special import gammainc as gammainc_orig


# See https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
#
#def upper_inc_gamma(k, x):
#    return float(mpmath.gammainc(k, a=float(x)))
#
#
#def T3(s, x):
#    # m=3, n=0, p=2, q=3
#    return float(mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=float(x)))
#
#
#upper_inc_gamma_vectorized = vectorize(upper_inc_gamma)
#T3_vectorized = vectorize(T3)
#
#
#@primitive
#def upper_inc_gamma_primitive(k, x):
#    return upper_inc_gamma_vectorized(k, x)
#
#
#defvjp(
#    upper_inc_gamma_primitive,
#    lambda ans, k, x: lambda g: g * (log(x) * upper_inc_gamma_vectorized(k, x) + x * T3_vectorized(k, x)),
#    lambda ans, k, x: lambda g: g * -1 * exp(-x) * x ** (k-1),
#)
#


@primitive
def gammainc(k, x):
    return gammainc_orig(k, x)


g_eps = 1e-6
defvjp(
    gammainc,
    lambda ans, k, x: lambda g: g * (gammainc_orig(k + g_eps, x) - ans) / g_eps,
    lambda ans, k, x: lambda g: g * (gammainc_orig(k, x + g_eps) - ans) / g_eps,
)
