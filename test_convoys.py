import numpy
import random
import scipy.stats
from convoys import Exponential

def test_exponential_model(c=0.05, lambd=0.1, n=100000):
    # With a really long observation window, the rate should converge to the measured
    C = numpy.array([random.random() < c and scipy.stats.expon.rvs(scale=1.0/lambd) or 0.0 for x in range(n)])
    N = numpy.array([100 for c in C])
    B = numpy.array([bool(c > 0) for c in C])
    model = Exponential()
    model.fit(C, N, B)
    c_est = model.params['c']
    lambd_est = model.params['lambd']
    assert 0.95*c < c_est < 1.05*c
    assert 0.95*lambd < lambd_est < 1.05*lambd
