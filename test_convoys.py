import datetime
import matplotlib
import numpy
import pytest
import random
import scipy.stats
matplotlib.use('Agg')  # Needed for matplotlib to run in Travis
import convoys
import convoys.regression


def sample_weibull(k, lambd):
    # scipy.stats is garbage for this
    # exp(-(x * lambda)^k) = y
    return (-numpy.log(random.random())) ** (1.0/k) / lambd

def generate_censored_data(N, E, C):
    B = numpy.array([random.random() < c and e < n for n, e, c in zip(N, E, C)])
    T = numpy.array([e if b else n for e, b, n in zip(E, B, N)])
    return B, T


def test_exponential_regression_model(c=0.3, lambd=0.1, n=100000):
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))  # did it convert
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))  # time now
    E = scipy.stats.expon.rvs(scale=1./lambd, size=(n,))  # time of event
    B, T = generate_censored_data(N, E, C)
    model = convoys.regression.ExponentialRegression()
    model.fit(X, B, T)
    assert 0.95*c < model.predict_final([1]) < 1.05*c
    t = 10
    d = 1 - numpy.exp(-lambd*t)
    assert 0.95*c*d < model.predict([1], t) < 1.05*c*d

    # Check the confidence intervals
    y, y_lo, y_hi = model.predict_final([1], ci=0.95)
    c_lo = scipy.stats.beta.ppf(0.025, n*c, n*(1-c))
    c_hi = scipy.stats.beta.ppf(0.975, n*c, n*(1-c))
    assert 0.95*c < y < 1.05*c
    assert 0.70*(c_hi-c_lo) < (y_hi-y_lo) < 1.30*(c_hi-c_lo)


def test_weibull_regression_model(cs=[0.3, 0.5, 0.7], lambd=0.1, k=0.5, n=100000):
    X = numpy.array([[1] + [r % len(cs) == j for j in range(len(cs))] for r in range(n)])
    C = numpy.array([bool(random.random() < cs[r % len(cs)]) for r in range(n)])
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))
    E = numpy.array([sample_weibull(k, lambd) for r in range(n)])
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.WeibullRegression()
    model.fit(X, B, T)
    for r, c in enumerate(cs):
        x = [1] + [int(r == j) for j in range(len(cs))]
        assert 0.95 * c < model.predict_final(x) < 1.05 * c


def test_weibull_regression_model_ci(c=0.3, lambd=0.1, k=0.5, n=100000):
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))
    E = numpy.array([sample_weibull(k, lambd) for r in range(n)])
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.WeibullRegression()
    model.fit(X, B, T)
    y, y_lo, y_hi = model.predict_final([1], ci=0.95)
    c_lo = scipy.stats.beta.ppf(0.025, n*c, n*(1-c))
    c_hi = scipy.stats.beta.ppf(0.975, n*c, n*(1-c))
    assert 0.95*c < y < 1.05 * c
    assert 0.70*(c_hi-c_lo) < (y_hi-y_lo) < 1.30*(c_hi-c_lo)


def test_gamma_regression_model(c=0.3, lambd=0.1, k=3.0, n=100000):
    # Something is a bit wacky with this one.
    # If I replace N with a smaller observation window, it breaks.
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = numpy.ones((n,)) * 1000.
    E = scipy.stats.gamma.rvs(a=k, scale=1.0/lambd, size=(n,))
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.GammaRegression()
    model.fit(X, B, T)
    assert 0.95*c < model.predict_final([1]) < 1.05*c
    assert 0.95*k < model.params['k'] < 1.05*k


def _get_data(c=0.3, k=10, lambd=0.1, n=1000):
    data = []
    now = datetime.datetime(2000, 7, 1)
    for x in range(n):
        date_a = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=random.random()*100)
        if random.random() < c:
            delay = scipy.stats.gamma.rvs(a=k, scale=1.0/lambd)
            date_b = date_a + datetime.timedelta(days=delay)
            if date_b < now:
                data.append(('foo', date_a, date_b, now))
            else:
                data.append(('foo', date_a, None, now))
        else:
            data.append(('foo', date_a, None, now))
    return data


def test_plot_cohorts():
    convoys.plot_cohorts(_get_data(), projection='gamma')


@pytest.mark.skip
def test_plot_conversion():
    convoys.plot_timeseries(_get_data(), window=datetime.timedelta(days=7), model='gamma')
