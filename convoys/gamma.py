from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f  # This is not documented
from scipy.special import gammainc as gammainc_orig

__all__ = ['gammainc']


@primitive
def gammainc(k, x):
    ''' Lower regularized incomplete gamma function.

    We rely on `scipy.special.gammainc 
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html>`_
    for this. However, there is a number of issues using this function
    together with `autograd <https://github.com/HIPS/autograd>`_:

    1. It doesn't let you take the gradient with respect to k
    2. The gradient with respect to x is really slow

    As a really stupid workaround, because we don't need the numbers to
    be 100% exact, we just approximate the gradient.

    Side note 1: if you truly want to compute the correct derivative, see the
    `Wikipedia articule about the Incomplete gamma function
    <https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives>`_
    where the T(3, s, x) function can be implemented as

    .. code-block:: python

       def T3(s, x):
           return mpmath.meijerg(a_s=([], [0, 0]), b_s=([s-1, -1, -1], []), z=x)

    I wasted a few hours on this but sadly it turns out to be extremely slow.

    Side note 2: TensorFlow actually has a `similar bug
    <https://github.com/tensorflow/tensorflow/issues/17995>`_
    '''
    return gammainc_orig(k, x)


G_EPS = 1e-6
defvjp(
    gammainc,
    lambda ans, k, x: unbroadcast_f(
        k, lambda g: g * (gammainc_orig(k + G_EPS, x) - ans) / G_EPS),
    lambda ans, k, x: unbroadcast_f(
        x, lambda g: g * (gammainc_orig(k, x + G_EPS) - ans) / G_EPS),
)
