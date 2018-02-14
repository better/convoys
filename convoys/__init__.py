import abc
import bisect
import datetime
import lifelines
import math
import numpy
import random
import seaborn
import scipy.optimize
import six
from autograd import jacobian, hessian, grad
from autograd.scipy.special import expit, gamma, gammainc, gammaincc, gammaln
from autograd.numpy import exp, log, sum
from matplotlib import pyplot

LOG_EPS = 1e-12  # Used for log likelihood


def log1pexp(x):
    # returns log(1 + exp(x))
    p, q = min(x, 0), max(x, 0)
    return q + log(exp(p-q) + 1)



def get_timescale(t):
    def get_timedelta_converter(t_factor):
        return lambda td: td.total_seconds() * t_factor

    if type(t) in (int, float):
        return '', lambda x: x
    elif t >= datetime.timedelta(days=1):
        return 'Days', get_timedelta_converter(1./(24*60*60))
    elif t >= datetime.timedelta(hours=1):
        return 'Hours', get_timedelta_converter(1./(60*60))
    elif t >= datetime.timedelta(minutes=1):
        return 'Minutes', get_timedelta_converter(1./60)
    else:
        return 'Minutes', get_timedelta_converter(1)


def get_arrays(data, t_converter):
    C = [t_converter(converted_at - created_at) if converted_at is not None else 1.0
         for created_at, converted_at, now in data]
    N = [t_converter(now - created_at)
         for created_at, converted_at, now in data]
    B = [bool(converted_at is not None)
         for created_at, converted_at, now in data]
    return numpy.array(C), numpy.array(N), numpy.array(B)


@six.add_metaclass(abc.ABCMeta)
class Model():
    def __init__(self, params={}):
        self.params = params

    @abc.abstractmethod
    def fit(self, C, N, B):
        pass

    @abc.abstractmethod
    def predict(self, ts, ci=None):
        pass

    @abc.abstractmethod
    def predict_final(self, ci=None):
        pass

    @abc.abstractmethod
    def predict_time(self, ci=None):
        pass


class SimpleBinomial(Model):
    def fit(self, C, N, B):
        self.params['a'] = len(B)
        self.params['b'] = sum(B)

    def predict(self, ts, ci=None):
        a, b = self.params['a'], self.params['b']
        def rep(x):
            return numpy.ones(len(ts)) * x
        if ci is not None:
            return ts, rep(b/a), rep(scipy.stats.beta.ppf((1 - ci)/2, b, a-b)), rep(scipy.stats.beta.ppf((1 + ci)/2, b, a-b))
        else:
            return ts, rep(b/a)

    def predict_final(self, ci=None):
        a, b = self.params['a'], self.params['b']
        if ci is not None:
            return b/a, scipy.stats.beta.ppf((1 - ci)/2, b, a-b), scipy.stats.beta.ppf((1 + ci)/2, b, a-b)
        else:
            return b/a

    def predict_time(self):
        return 0.0


class Basic(Model):
    def fit(self, C, N, B, n_limit=30):
        n, k = len(C), 0
        self.ts = [0]
        self.ns = [n]
        self.ks = [k]
        events = [(c, 1, 0) for c, n, b in zip(C, N, B) if b] + \
                 [(n, -int(b), -1) for c, n, b in zip(C, N, B)]
        for t, k_delta, n_delta in sorted(events):
            k += k_delta
            n += n_delta
            self.ts.append(t)
            self.ks.append(k)
            self.ns.append(n)
            if n < n_limit:
                break

    def predict(self, ts, ci=None):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        ks = numpy.array([self.ks[j] for j in js if j < len(self.ks)])
        ns = numpy.array([self.ns[j] for j in js if j < len(self.ns)])
        ts = numpy.array([ts[j] for j in js if j < len(self.ns)])
        if ci is not None:
            return ts, ks / ns, scipy.stats.beta.ppf((1 - ci)/2, ks, ns-ks), scipy.stats.beta.ppf((1 + ci)/2, ks, ns-ks)
        else:
            return ts, ks / ns


class KaplanMeier(Model):
    def fit(self, C, N, B):
        T = [c if b else n for c, n, b in zip(C, N, B)]
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(T, event_observed=B)
        self.ts = kmf.survival_function_.index.values
        self.ps = 1.0 - kmf.survival_function_['KM_estimate'].values
        self.ps_hi = 1.0 - kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        self.ps_lo = 1.0 - kmf.confidence_interval_['KM_estimate_upper_0.95'].values

    def predict(self, ts, ci=None):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[j] for j in js if j < len(self.ts)])
        if ci is not None:
            return (array_lookup(self.ts), array_lookup(self.ps), array_lookup(self.ps_lo), array_lookup(self.ps_hi))
        else:
            return (array_lookup(self.ts), array_lookup(self.ps))

    def predict_final(self, ci=None):
        if ci is not None:
            return (self.ps[-1], self.ps_lo[-1], self.ps_hi[-1])
        else:
            return self.ps[-1]

    def predict_time(self, ci=None):
        # TODO: should not use median here, but mean is no good
        def median(ps):
            i = bisect.bisect_left(ps, 0.5)
            return self.ts[min(i, len(ps)-1)]
        if ci is not None:
            return median(self.ps), median(self.ps_lo), median(self.ps_hi)
        else:
            return median(self.ps)


def fit_beta(c, fc):
    # Approximate a Beta distribution for c by fitting two things
    # 1. second derivative wrt c should match the second derivative of a beta
    #   LL = (a-1)log(c) + (b-1)log(1-c)
    #   dLL = (a-1)/c - (b-1)/(1-c)
    #   ddLL = -(a-1)/c^2 - (b-1)/(1-c)^2 = -h
    #   a(-1/c^2) + b(-1/(1-c)^2) = -h - 1/c^2 - 1/(1-c)^2
    # 2. mode should match c, i.e. (a-1)/(a+b-2) = c <=> a-1 = c(a+b-2)
    #   a(1-c) - bc = 1-2c
    h = grad(grad(fc))(c)
    M = numpy.array([[-1/c**2, -1/(1-c)**2],
                     [(1-c), -c]])
    q = numpy.array([-h - 1/c**2 - 1/(1-c)**2, 1 - 2*c])
    try:
        a, b = numpy.linalg.solve(M, q)
    except numpy.linalg.linalg.LinAlgError:
        a, b = 1, 1
    return a, b


class Exponential(Model):
    def fit(self, C, N, B):
        def transform(x):
            p, q = x
            return (expit(p), log1pexp(q))
        def f(x):
            c, lambd = x
            LL_observed = log(c) + log(lambd) - lambd*C
            LL_censored = log((1 - c) + c * exp(-lambd*N))
            neg_LL = -sum(B * LL_observed + (1 - B) * LL_censored)
            return neg_LL

        g = lambda x: f(transform(x))
        res = scipy.optimize.minimize(
            fun=g,
            jac=jacobian(g),
            hess=hessian(g),
            x0=(0, 0),
            method='trust-ncg')
        c, lambd = transform(res.x)
        fc = lambda c: f((c, lambd))
        a, b = fit_beta(c, fc)
        self.params = dict(a=a, b=b, lambd=lambd)

    def predict(self, ts, ci=None):
        a, b, lambd = self.params['a'], self.params['b'], self.params['lambd']
        y = 1 - exp(-ts * lambd)
        if ci is not None:
            return ts, a / (a + b) * y, scipy.stats.beta.ppf((1 - ci)/2, a, b) * y, scipy.stats.beta.ppf((1 + ci)/2, a, b) * y
        else:
            return ts, y

    def predict_final(self, ci=None):
        a, b = self.params['a'], self.params['b']
        if not ci:
            return a / (a + b)
        else:
            return (a / (a + b),
                    scipy.stats.beta.ppf((1 - ci)/2, a, b),
                    scipy.stats.beta.ppf((1 + ci)/2, a, b))

    def predict_time(self):
        return 1.0 / self.params['lambd']


class Weibull(Model):
    def fit(self, C, N, B):
        def transform(x):
            p, q, r = x
            return (expit(p), log1pexp(q), log1pexp(r))
        def f(x):
            c, lambd, k = x
            # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
            LL_observed = log(c) + log(k) + log(lambd) + (k-1)*(log(C) + log(lambd)) - (C*lambd)**k
            # CDF of Weibull: 1 - exp(-(t * lambda)^k)
            LL_censored = log((1-c) + c * exp(-(N*lambd)**k))
            neg_LL = -sum(B * LL_observed + (1 - B) * LL_censored)
            return neg_LL

        g = lambda x: f(transform(x))
        res = scipy.optimize.minimize(
            fun=g,
            jac=jacobian(g),
            hess=hessian(g),
            x0=(0, 0, 0),
            method='trust-ncg')
        c, lambd, k = transform(res.x)
        fc = lambda c: f((c, lambd, k))
        a, b = fit_beta(c, fc)
        self.params = dict(a=a, b=b, lambd=lambd, k=k)

    def predict(self, ts, ci=None):
        a, b, lambd, k = self.params['a'], self.params['b'], self.params['lambd'], self.params['k']
        y = 1 - exp(-(ts*lambd)**k)
        if ci is not None:
            return ts, a / (a + b) * y, scipy.stats.beta.ppf((1 - ci)/2, a, b) * y, scipy.stats.beta.ppf((1 + ci)/2, a, b) * y
        else:
            return ts, y

    def predict_final(self, ci=None):
        a, b = self.params['a'], self.params['b']
        if ci is not None:
            return a / (a + b), scipy.stats.beta.ppf((1 - ci)/2, a, b), scipy.stats.beta.ppf((1 + ci)/2, a, b)
        else:
            return a / (a + b)

    def predict_time(self):
        return gamma(1 + 1./self.params['k']) / self.params['lambd']


class Gamma(Model):
    def fit(self, C, N, B):
        # TODO(erikbern): should compute Jacobian of this one
        def transform(x):
            p, q, r = x
            return (expit(p), log1pexp(q), log1pexp(r))
        def f(x):
            c, lambd, k = x
            neg_LL = 0
            # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
            LL_observed = log(c) - gammaln(k) + k*log(lambd) + (k-1)*log(C + LOG_EPS) - lambd*C
            # CDF of gamma: gammainc(k, lambda * t)
            LL_censored = log((1-c) + c * gammaincc(k, lambd*N) + LOG_EPS)
            neg_LL = -sum(B * LL_observed + (1 - B) * LL_censored)
            return neg_LL

        g = lambda x: f(transform(x))
        res = scipy.optimize.minimize(
            fun=g,
            x0=(0, 0, 0),
            method='Nelder-Mead')
        c, lambd, k = transform(res.x)
        fc = lambda c: f((c, lambd, k))
        a, b = fit_beta(c, fc)
        self.params = dict(a=a, b=b, lambd=lambd, k=k)

    def predict(self, ts, ci=None):
        a, b, lambd, k = self.params['a'], self.params['b'], self.params['lambd'], self.params['k']
        y = gammainc(k, lambd*ts)
        if ci is not None:
            return ts, a / (a + b) * y, scipy.stats.beta.ppf((1 - ci)/2, a, b) * y, scipy.stats.beta.ppf((1 + ci)/2, a, b) * y
        else:
            return ts, y

    def predict_final(self, ci=None):
        a, b = self.params['a'], self.params['b']
        if ci is not None:
            return a / (a + b), scipy.stats.beta.ppf((1 - ci)/2, a, b), scipy.stats.beta.ppf((1 + ci)/2, a, b)
        else:
            return a / (a + b)

    def predict_time(self):
        return self.params['k'] / self.params['lambd']


def sample_event(model, t, hi=1e3):
    # We are now at time t. Generate a random event whether the user is going to convert or not
    # TODO: this is a hacky thing until we have a "invert CDF" method on each model
    def pred(t):
        ts = numpy.array([t])
        return model.predict(ts)[1][-1]
    y = pred(t)
    r = y + random.random() * (1 - y)
    if pred(hi) < r:
        return None
    lo = t
    for j in range(20):
        mid = (lo + hi) / 2
        if pred(mid) < r:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


def split_by_group(data, group_min_size, max_groups):
    js = {}
    for group, created_at, converted_at, now in data:
        if converted_at is not None and converted_at < created_at:
            print('created at', created_at, 'but converted at', converted_at)
            continue
        if now < created_at:
            print('created at', created_at, 'but now is', now)
            continue
        js.setdefault(group, []).append((created_at, converted_at, now))
    groups = list(js.keys())

    # Remove groups with too few data points
    groups = [group for group in groups if len(js[group]) >= group_min_size]

    # Require at least one conversion per group
    groups = [group for group in groups if any(converted_at for _, converted_at, _ in js[group]) > 0]

    # Pick the top groups
    groups = sorted(groups, key=lambda group: len(js[group]), reverse=True)[:max_groups]

    # Sort groups lexicographically
    groups = sorted(groups)

    return groups, js


_models = {
    'basic': Basic,
    'kaplan-meier': KaplanMeier,
    'simple-binomial': SimpleBinomial,
    'exponential': Exponential,
    'weibull': Weibull,
    'gamma': Gamma,
}

def plot_cohorts(data, t_max=None, title=None, group_min_size=0, max_groups=100, model='kaplan-meier', projection=None):
    # Set x scale
    if t_max is None:
        t_max = max(now - created_at for group, created_at, converted_at, now in data)
    t_unit, t_converter = get_timescale(t_max)
    t_max = t_converter(t_max)

    # Split data by group
    groups, js = split_by_group(data, group_min_size, max_groups)

    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    y_max = 0
    for group, color in zip(sorted(groups), colors):
        C, N, B = get_arrays(js[group], t_converter)
        t = numpy.linspace(0, t_max, 1000)

        m = _models[model]()
        m.fit(C, N, B)

        label = '%s (n=%.0f, k=%.0f)' % (group, len(B), sum(B))

        if projection:
            p = _models[projection]()
            p.fit(C, N, B)
            p_t, p_y, p_y_lo, p_y_hi = p.predict(t, ci=0.95)
            p_y_final, p_y_lo_final, p_y_hi_final = p.predict_final(ci=0.95)
            label += ' projected: %.2f%% (%.2f%% - %.2f%%)' % (100.*p_y_final, 100.*p_y_lo_final, 100.*p_y_hi_final)
            pyplot.plot(p_t, 100. * p_y, color=color, linestyle=':', alpha=0.7)
            pyplot.fill_between(p_t, 100. * p_y_lo, 100. * p_y_hi, color=color, alpha=0.2)

        m_t, m_y = m.predict(t)
        pyplot.plot(m_t, 100. * m_y, color=color, label=label)
        y_max = max(y_max, 110. * max(m_y))

    if title:
        pyplot.title(title)
    pyplot.xlim([0, t_max])
    pyplot.ylim([0, y_max])
    pyplot.xlabel(t_unit)
    pyplot.ylabel('Conversion rate %')
    pyplot.legend()
    pyplot.gca().grid(True)


def plot_timeseries(data, window, model='kaplan-meier', group_min_size=0, max_groups=100, window_min_size=1, stride=None, title=None, time=False):
    if stride is None:
        stride = window

    # Find limits
    t_lo = min(created_at for _, created_at, _, _ in data)
    t_hi = min(now for _, _, _, now in data)
    t_unit, t_converter = get_timescale(t_hi - t_lo)

    # Split data by group
    groups, js = split_by_group(data, group_min_size, max_groups)

    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    y_max = 0
    for group, color in zip(sorted(groups), colors):
        t1 = t_lo
        ts, ys, y_los, y_his = [], [], [], []
        js[group].sort(key=lambda x: x[0])
        created_ats = [created_at for created_at, _, _ in js[group]]
        while True:
            t2 = t1 + window
            i1 = bisect.bisect_left(created_ats, t1)
            i2 = bisect.bisect_left(created_ats, t2)
            if i2 >= len(js[group]):
                break
            data = js[group][i1:i2]
            t1 += stride

            C, N, B = get_arrays(data, t_converter)
            if sum(B) < window_min_size:
                continue

            p = _models[model]()
            p.fit(C, N, B)

            if time:
                y, y_lo, y_hi = p.predict_time(ci=0.95)
            else:
                y, y_lo, y_hi = p.predict_final(ci=0.95)
            print('%30s %40s %.4f %.4f %.4f' % (group, t1, y, y_lo, y_hi))
            ts.append(t2)
            ys.append(y)
            y_los.append(y_lo)
            y_his.append(y_hi)

        if not time:
            ys, y_los, y_his = (100.*numpy.array(x) for x in (ys, y_los, y_his))
        pyplot.plot(ts, ys, color=color, label='%s (%d)' % (group, len(js[group])))
        pyplot.fill_between(ts, y_los, y_his, color=color, alpha=0.2)
        y_max = max(y_max, 1.1 * max(ys))

    if title:
        pyplot.title(title)
    if time:
        pyplot.ylabel('Average time to conversion (%s)' % t_unit)
    else:
        pyplot.ylabel('Conversion rate %')
    pyplot.ylim([0, y_max])
    pyplot.legend()
    pyplot.gca().grid(True)
