from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f  # This is not documented
from scipy.special import gammainc as gammainc_orig

@primitive
def gammainc(k, x):
    ''' Super janky hack to make it possible for autograd to compute gradients of gammainc.

    There are two problems with autograd.scipy.special.gammainc:
    1. It doesn't let you take the gradient with respect to k
    2. The gradient with respect to x is really slow

    As a really stupid workaround, because we don't need the numbers to be 100% exact,
    we just approximate the gradient.

    As a side note, if you truly want to compute the correct derivative, there's some
    math here that's enough: https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
    T(3, s, x) can be implemented as mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=x)
    I wasted a few hours on this but sadly it turns out to be extremely slow.
    '''
    return gammainc_orig(k, x)


G_EPS = 1e-6
defvjp(
    gammainc,
    lambda ans, k, x: unbroadcast_f(k, lambda g: g * (gammainc_orig(k + G_EPS, x) - ans) / G_EPS),
    lambda ans, k, x: unbroadcast_f(x, lambda g: g * (gammainc_orig(k, x + G_EPS) - ans) / G_EPS),
)
