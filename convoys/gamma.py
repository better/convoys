from autograd.extend import primitive, defvjp
from autograd import numpy as np
from autograd.scipy.special import gamma
from autograd.numpy.numpy_vjps import unbroadcast_f  # This is not documented
from scipy.special import gammainc as _scipy_gammainc, gammaincc as _scipy_gammaincc

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
    return _scipy_gammainc(k, x)


delta = 1e-6

defvjp(
    gammainc,
    lambda ans, a, x: unbroadcast_f(
        a,
        lambda g: g
        * (
            -gammainc(a + 2 * delta, x)
            + 8 * gammainc(a + delta, x)
            - 8 * gammainc(a - delta, x)
            + gammainc(a - 2 * delta, x)
        )
        / (12 * delta),
    ),
    lambda ans, a, x: unbroadcast_f(x, lambda g: g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)


gammaincc = primitive(_scipy_gammaincc)


defvjp(
    gammaincc,
    lambda ans, a, x: unbroadcast_f(
        a,
        lambda g: g
        * (
            -gammaincc(a + 2 * delta, x)
            + 8 * gammaincc(a + delta, x)
            - 8 * gammaincc(a - delta, x)
            + gammaincc(a - 2 * delta, x)
        )
        / (12 * delta),
    ),
    lambda ans, a, x: unbroadcast_f(x, lambda g: -g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
)
