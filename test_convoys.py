import autograd
import datetime
import flaky
import matplotlib
import numpy
import pandas
import pytest
import random
import scipy.stats
matplotlib.use('Agg')  # Needed for matplotlib to run in Travis
import convoys
import convoys.plotting
import convoys.regression
import convoys.single
import convoys.utils


def sample_weibull(k, lambd):
    # scipy.stats is garbage for this
    # exp(-(x * lambda)^k) = y
    return (-numpy.log(random.random())) ** (1.0/k) / lambd


def generate_censored_data(N, E, C):
    B = numpy.array([c and e < n for n, e, c in zip(N, E, C)])
    T = numpy.array([e if b else n for e, b, n in zip(E, B, N)])
    return B, T


def test_kaplan_meier_model():
    data = [
            (2, 0),
            (3, 0),
            (6, 1),
            (6, 1),
            (7, 1),
            (10, 0)
        ]
    now = pandas.Timestamp('2019-01-22')  # fix now end date for easier testing
    created_array = [now - pandas.DateOffset(t) for t, e in data]
    converted_array = [ts + pandas.DateOffset(t) if e == 1 else numpy.nan for ts, (t, e) in zip(created_array, data)]
    df = pandas.DataFrame({
        'created_at': created_array,
        'converted_at': converted_array,
        'group': 1
    })
    df['now'] = now
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df,
        converted='converted_at',
        created='created_at',
        unit='days'
    )
    m = convoys.multi.KaplanMeier()
    m.fit(G, B, T)
    assert m.predict(0, 9) == 0.75


def test_output_shapes(c=0.3, lambd=0.1, n=1000, k=5):
    X = numpy.random.randn(n, k)
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))
    E = scipy.stats.expon.rvs(scale=1./lambd, size=(n,))
    B, T = generate_censored_data(N, E, C)

    # Fit model with ci
    model = convoys.regression.Exponential(mcmc=True)
    model.fit(X, B, T)

    # Generate output without ci
    assert model.predict(X[0], 0).shape == ()
    assert model.predict([X[0], X[1]], 0).shape == (2,)
    assert model.predict([X[0]], [0, 1, 2, 3]).shape == (4,)
    assert model.predict([X[0], X[1], X[2]], [0, 1, 2]).shape == (3,)
    assert model.predict([[X[0], X[1]]], [[0], [1], [2]]).shape == (3, 2)
    assert model.predict([[X[0]], [X[1]]], [[0, 1, 2]]).shape == (2, 3)

    # Generate output with ci (same as above plus (3,))
    assert model.predict_ci(X[0], 0, ci=0.8).shape == (3,)
    assert model.predict_ci([X[0], X[1]], 0, ci=0.8).shape == (2, 3)
    assert model.predict_ci([X[0]], [0, 1, 2, 3], ci=0.8).shape == (4, 3)
    assert model.predict_ci([X[0], X[1], X[2]], [0, 1, 2], ci=0.8) \
                .shape == (3, 3)
    assert model.predict_ci([[X[0], X[1]]], [[0], [1], [2]], ci=0.8) \
                .shape == (3, 2, 3)
    assert model.predict_ci([[X[0]], [X[1]]], [[0, 1, 2]], ci=0.8) \
                .shape == (2, 3, 3)

    # Assert old interface still works
    assert model.cdf(X[0], 0).shape == ()
    assert model.cdf(X[0], 0, ci=0.8).shape == (3,)

    # Fit model without ci (should be the same)
    model = convoys.regression.Exponential(mcmc=False)
    model.fit(X, B, T)
    assert model.predict(X[0], 0).shape == ()
    assert model.predict([X[0], X[1]], [0, 1]).shape == (2,)


@flaky.flaky
def test_exponential_regression_model(c=0.3, lambd=0.1, n=10000):
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))  # did it convert
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))  # time now
    E = scipy.stats.expon.rvs(scale=1./lambd, size=(n,))  # time of event
    B, T = generate_censored_data(N, E, C)
    model = convoys.regression.Exponential(mcmc=True)
    model.fit(X, B, T)
    assert 0.80*c < model.predict([1], float('inf')) < 1.30*c
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd*t)
        assert 0.80*c*d < model.predict([1], t) < 1.30*c*d

    # Check the confidence intervals
    assert model.predict_ci([1], float('inf'), ci=0.95).shape == (3,)
    assert model.predict_ci([1], [0, 1, 2, 3], ci=0.95).shape == (4, 3)
    y, y_lo, y_hi = model.predict_ci([1], float('inf'), ci=0.95)
    assert 0.80*c < y < 1.30*c

    # Check the random variates
    will_convert, convert_at = model.rvs([1], n_curves=10000, n_samples=1)
    assert 0.80*c < numpy.mean(will_convert) < 1.30*c
    convert_times = convert_at[will_convert]
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd*t)
        assert 0.70*d < (convert_times < t).mean() < 1.30*d

    # Fit a linear model
    model = convoys.regression.Exponential(mcmc=False, flavor='linear')
    model.fit(X, B, T)
    model_c = model.params['map']['b'] + model.params['map']['beta'][0]
    assert 0.9*c < model_c < 1.1*c
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd*t)
        assert 0.80*c*d < model.predict([1], t) < 1.30*c*d


@flaky.flaky
def test_weibull_regression_model(cs=[0.3, 0.5, 0.7],
                                  lambd=0.1, k=0.5, n=10000):
    X = numpy.array([[r % len(cs) == j for j in range(len(cs))]
                     for r in range(n)])
    C = numpy.array([bool(random.random() < cs[r % len(cs)])
                     for r in range(n)])
    N = scipy.stats.uniform.rvs(scale=5./lambd, size=(n,))
    E = numpy.array([sample_weibull(k, lambd)
                     for r in range(n)])
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.Weibull()
    model.fit(X, B, T)

    # Validate shape of results
    x = numpy.ones((len(cs),))
    assert model.predict(x, float('inf')).shape == ()
    assert model.predict(x, 1).shape == ()
    assert model.predict(x, [1, 2, 3, 4]).shape == (4,)

    # Check results
    for r, c in enumerate(cs):
        x = [int(r == j) for j in range(len(cs))]
        assert 0.80 * c < model.predict(x, float('inf')) < 1.30 * c

    # Fit a linear model
    model = convoys.regression.Weibull(mcmc=False, flavor='linear')
    model.fit(X, B, T)
    model_cs = model.params['map']['b'] + model.params['map']['beta']
    for model_c, c in zip(model_cs, cs):
        assert 0.8 * c < model_c < 1.2 * c


@flaky.flaky
def test_gamma_regression_model(c=0.3, lambd=0.1, k=3.0, n=10000):
    # TODO: this one seems very sensitive to large values for N (i.e. less censoring)
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = scipy.stats.uniform.rvs(scale=20./lambd, size=(n,))
    E = scipy.stats.gamma.rvs(a=k, scale=1.0/lambd, size=(n,))
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.Gamma()
    model.fit(X, B, T)
    assert 0.80*c < model.predict([1], float('inf')) < 1.30*c
    assert 0.80*k < numpy.mean(model.params['map']['k']) < 1.30*k

    # Fit a linear model
    model = convoys.regression.Gamma(mcmc=False, flavor='linear')
    model.fit(X, B, T)
    model_c = model.params['map']['b'] + model.params['map']['beta'][0]
    assert 0.9*c < model_c < 1.1*c


@flaky.flaky
def test_linear_model(n=10000, m=5, k=3.0, lambd=0.1):
    # Generate data with little censoring
    # The coefficients should be quite close to their actual value
    cs = numpy.random.dirichlet(numpy.ones(m))
    X = numpy.random.binomial(n=1, p=0.5, size=(n, m))
    C = numpy.random.rand(n) < numpy.dot(X, cs.T)
    N = scipy.stats.uniform.rvs(scale=20./lambd, size=(n,))
    E = numpy.array([sample_weibull(k, lambd) for r in range(n)])
    B, T = generate_censored_data(N, E, C)

    model = convoys.regression.Weibull(mcmc=False, flavor='linear')
    model.fit(X, B, T)

    # Check the fitted parameters
    model_cs = model.params['map']['b'] + model.params['map']['beta']
    for model_c, c in zip(model_cs, cs):
        assert c - 0.03 < model_c < c + 0.03
    model_lambds = numpy.exp(model.params['map']['a'] +
                             model.params['map']['alpha'])
    for model_lambd in model_lambds:
        assert 0.95*lambd < model_lambd < 1.05*lambd

    # Check predictions
    for i, c in enumerate(cs):
        x = numpy.array([float(j == i) for j in range(m)])
        p = model.predict(x, float('inf'))
        assert c - 0.03 < p < c + 0.03
        t = 10.0
        p = model.predict(x, t)
        f = 1 - numpy.exp(-(t*lambd)**k)
        assert c*f - 0.03 < p < c*f + 0.03


@flaky.flaky
def test_exponential_pooling(c=0.5, lambd=0.01, n=10000, ks=[1, 2, 3]):
    # Generate one series of n observations with c conversion rate
    # Then k1...kn series with zero conversion
    # The predicted conversion rates should go towards c for the small cohorts
    G = numpy.zeros(n + sum(ks))
    C = numpy.zeros(n + sum(ks))
    N = numpy.zeros(n + sum(ks))
    E = numpy.zeros(n + sum(ks))
    offset = 0
    for i, k in enumerate([n] + ks):
        G[offset:offset+k] = i
        offset += k
    C[:n] = scipy.stats.bernoulli.rvs(c, size=(n,))
    N[:] = 1000.
    E[:n] = scipy.stats.expon.rvs(scale=1./lambd, size=(n,))
    B, T = generate_censored_data(N, E, C)

    # Fit model
    model = convoys.multi.Exponential()
    model.fit(G, B, T)

    # Generate predictions for each cohort
    c = numpy.array([model.predict(i, float('inf')) for i in range(1+len(ks))])
    assert numpy.all(c[1:] > 0.25)  # rough check
    assert numpy.all(c[1:] < 0.50)  # same
    assert numpy.all(numpy.diff(c) < 0)  # c should be monotonically decreasing


def _generate_dataframe(cs=[0.3, 0.5, 0.7], k=0.5, lambd=0.1, n=1000):
    groups = [r % len(cs) for r in range(n)]
    C = numpy.array([bool(random.random() < cs[g]) for g in groups])
    N = scipy.stats.expon.rvs(scale=10./lambd, size=(n,))
    E = numpy.array([sample_weibull(k, lambd) for r in range(n)])
    B, T = generate_censored_data(N, E, C)

    x2t = lambda x: datetime.datetime(2000, 1, 1) + datetime.timedelta(days=x)
    return pandas.DataFrame(data=dict(
        group=['Group %d' % g for g in groups],
        created=[x2t(0) for g in groups],
        converted=[x2t(t) if b else None for t, b in zip(T, B)],
        now=[x2t(n) for n in N]
    ))


def test_convert_dataframe(n=1000):
    df = _generate_dataframe(n=n)
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    assert G.shape == B.shape == T.shape == (n,)


def test_convert_dataframe_features(n=1000):
    df = _generate_dataframe(n=n)
    df['features'] = [tuple(numpy.random.randn() for z in range(3))
                      for g in df['group']]
    df = df.drop('group', axis=1)
    unit, groups, (X, B, T) = convoys.utils.get_arrays(df)
    assert X.shape == (n, 3)

    # Generate from multiple columns
    df = _generate_dataframe(n=n)
    df['feature_1'] = [numpy.random.randn() for g in df['group']]
    df['feature_2'] = [numpy.random.randn() for g in df['group']]
    df = df.drop('group', axis=1)
    unit, groups, (X, B, T) = convoys.utils.get_arrays(
        df, features=('feature_1', 'feature_2'))
    assert X.shape == (n, 2)


def test_convert_dataframe_infer_now():
    df = _generate_dataframe()
    df = df.drop('now', axis=1)

    unit, groups, (G1, B1, T1) = convoys.utils.get_arrays(df, unit='days')

    # Now, let's make the timezone-naive objects timezone aware
    utc = datetime.timezone.utc
    local = datetime.datetime.now(utc).astimezone().tzinfo
    df[['created', 'converted']] = df[['created', 'converted']].applymap(
        lambda z: z.replace(tzinfo=local))
    unit, groups, (G2, B2, T2) = convoys.utils.get_arrays(df, unit='days')

    # Convert everything to UTC and make sure it's still the same
    df[['created', 'converted']] = df[['created', 'converted']].applymap(
        lambda z: z.tz_convert(utc))
    unit, groups, (G3, B3, T3) = convoys.utils.get_arrays(df, unit='days')

    # Let's check that all deltas are the same
    # There will be some slight clock drift, so let's accept up to 3s
    for t1, t2, t3 in zip(T1, T2, T3):
        assert 0 <= t2 - t1 < 3.0 / (24*60*60)
        assert 0 <= t3 - t1 < 3.0 / (24*60*60)


def test_convert_dataframe_timedeltas():
    df = _generate_dataframe()

    unit, groups, (G1, B1, T1) = convoys.utils.get_arrays(df, unit='days')
    df2 = pandas.DataFrame({'group': df['group'],
                            'converted': df['converted'] - df['created'],
                            'now': df['now'] - df['created']})
    unit, groups, (G2, B2, T2) = convoys.utils.get_arrays(df2, unit='days')

    for t1, t2 in zip(T1, T2):
        assert 0 <= t2 - t1 < 3.0 / (24*60*60)


def test_convert_dataframe_more_args():
    df = _generate_dataframe()
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df, max_groups=2)
    assert len(groups) <= 2
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df, group_min_size=9999)
    assert G.shape == (0,)


def test_convert_dataframe_created_at_nan(n=1000):
    df = _generate_dataframe(n=n)
    df.loc[df.index[0], 'created'] = None
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    assert numpy.issubdtype(G.dtype, numpy.integer)
    assert numpy.issubdtype(B.dtype, numpy.bool_)
    assert numpy.issubdtype(T.dtype, numpy.number)


def _test_plot_cohorts(model='weibull', extra_model=None):
    df = _generate_dataframe()
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    matplotlib.pyplot.clf()
    convoys.plotting.plot_cohorts(G, B, T, model=model, ci=0.95, groups=groups)
    matplotlib.pyplot.legend()
    if extra_model:
        convoys.plotting.plot_cohorts(G, B, T, model=extra_model,
                                      plot_kwargs=dict(linestyle='--',
                                                       alpha=0.1))
    matplotlib.pyplot.savefig('%s-%s.png' % (model, extra_model)
                              if extra_model is not None else '%s.png' % model)


def test_plot_cohorts_model():
    df = _generate_dataframe()
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    model = convoys.multi.Exponential(mcmc=None)
    model.fit(G, B, T)
    matplotlib.pyplot.clf()
    convoys.plotting.plot_cohorts(G, B, T, model=model, groups=groups)
    matplotlib.pyplot.legend()

    with pytest.raises(Exception):
        convoys.plotting.plot_cohorts(G, B, T, model='bad', groups=groups)

    with pytest.raises(Exception):
        convoys.plotting.plot_cohorts(G, B, T, model=model, groups=groups,
                                      specific_groups=['Nonsense'])


@flaky.flaky
def test_plot_cohorts_kaplan_meier():
    _test_plot_cohorts(model='kaplan-meier')


@flaky.flaky
def test_plot_cohorts_weibull():
    _test_plot_cohorts(model='weibull')


@flaky.flaky
def test_plot_cohorts_two_models():
    _test_plot_cohorts(model='kaplan-meier', extra_model='weibull')


def test_plot_cohorts_subplots():
    df = _generate_dataframe()
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    matplotlib.pyplot.clf()
    fix, axes = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    for ax in axes.flatten():
        convoys.plotting.plot_cohorts(G, B, T, groups=groups, ax=ax)
        ax.legend()
    matplotlib.pyplot.savefig('subplots.png')


def test_marriage_example():
    from examples.marriage import run
    run()


def test_dob_violations_example():
    from examples.dob_violations import run
    run()
