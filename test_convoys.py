import numpy
import random
import scipy.stats
from convoys import Exponential, Gamma


def test_exponential_model(c=0.05, lambd=0.1, n=1000000):
    # With a really long observation window, the rate should converge to the measured
    C = numpy.array([random.random() < c and scipy.stats.expon.rvs(scale=1.0/lambd) or 0.0 for x in range(n)])
    N = numpy.array([100 for c in C])
    B = numpy.array([bool(c > 0) for c in C])
    model = Exponential()
    model.fit(C, N, B)
    assert 0.95*c < model.params['c'] < 1.05*c
    assert 0.95*lambd < model.params['lambd'] < 1.05*lambd


def test_gamma_model(c=0.05, lambd=0.1, k=10.0, n=100000):
    C = numpy.array([random.random() < c and scipy.stats.gamma.rvs(a=k, scale=1.0/lambd) or 0.0 for x in range(n)])
    N = numpy.array([1000 for c in C])
    B = numpy.array([bool(c > 0) for c in C])
    model = Gamma()
    model.fit(C, N, B)
    assert 0.95*c < model.params['c'] < 1.05*c
    assert 0.95*lambd < model.params['lambd'] < 1.05*lambd
    assert 0.95*k < model.params['k'] < 1.05*k
