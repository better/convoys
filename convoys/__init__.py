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
from scipy.special import gamma, gammainc
from matplotlib import pyplot


def get_timescale(t):
    if t >= datetime.timedelta(days=1):
        t_factor, t_unit = 1./(24*60*60), 'Days'
    elif t >= datetime.timedelta(hours=1):
        t_factor, t_unit = 1./(60*60), 'Hours'
    elif t >= datetime.timedelta(minutes=1):
        t_factor, t_unit = 1./60, 'Minutes'
    else:
        t_factor, t_unit = 1, 'Seconds'
    return t_factor, t_unit


def get_arrays(data, t_factor):
    C = [(converted_at - created_at).total_seconds() * t_factor if converted_at is not None else 0.0
         for created_at, converted_at, now in data]
    N = [(now - created_at).total_seconds() * t_factor
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
    def predict(self, ts, confidence_interval=False):
        pass

    @abc.abstractmethod
    def predict_final(self, confidence_interval=False):
        pass


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

    def predict(self, ts, confidence_interval=False):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        ks = numpy.array([self.ks[j] for j in js if j < len(self.ks)])
        ns = numpy.array([self.ns[j] for j in js if j < len(self.ns)])
        ts = numpy.array([ts[j] for j in js if j < len(self.ns)])
        if confidence_interval:
            return ts, ks / ns, scipy.stats.beta.ppf(0.05, ks, ns-ks), scipy.stats.beta.ppf(0.95, ks, ns-ks)
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

    def predict(self, ts, confidence_interval=False):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[j] for j in js if j < len(self.ts)])
        if confidence_interval:
            return (array_lookup(self.ts), array_lookup(self.ps), array_lookup(self.ps_lo), array_lookup(self.ps_hi))
        else:
            return (array_lookup(self.ts), array_lookup(self.ps))

    def predict_final(self, confidence_interval=False):
        if self.confidence_interval:
            return (self.ps[-1], self.ps_lo[-1], self.ps_hi[-1])
        else:
            return self.ps[-1]


class Exponential(Model):
    def fit(self, C, N, B):
        def f(x):
            c, lambd = x
            neg_LL, neg_LL_deriv_c, neg_LL_deriv_lambd = 0, 0, 0
            likelihood_observed = c * lambd * numpy.exp(-lambd*C)
            likelihood_censored = (1 - c) + c * numpy.exp(-lambd*N)
            neg_LL = -numpy.sum(numpy.log(B * likelihood_observed + (1 - B) * likelihood_censored))
            neg_LL_deriv_c = -numpy.sum(B * 1/c + (1 - B) * (-1 + numpy.exp(-lambd*N)) / likelihood_censored)
            neg_LL_deriv_lambd = -numpy.sum(B * (1/lambd - C) + (1 - B) * (c * -N * numpy.exp(-lambd*N)) / likelihood_censored)
            return neg_LL, numpy.array([neg_LL_deriv_c, neg_LL_deriv_lambd])

        c_initial = numpy.mean(B)
        lambd_initial = 1.0 / max(N)
        lambd_max = 30.0 / max(N)
        lambd = self.params.get('lambd')
        res = scipy.optimize.minimize(
            fun=f,
            x0=(c_initial, lambd_initial),
            bounds=((1e-4, 1-1e-4),
                    (lambd, lambd) if lambd else (1e-4, lambd_max)),
            method='L-BFGS-B',
            jac=True)
        c, lambd = res.x
        self.params = dict(c=c, lambd=lambd)

    def predict(self, ts):
        c, lambd = self.params['c'], self.params['lambd']
        return ts, c * (1 - numpy.exp(-ts * lambd))

    def predict_final(self):
        return self.params['c']


class Gamma(Model):
    def fit(self, C, N, B):
        # TODO(erikbern): should compute Jacobian of this one
        def f(x):
            c, lambd, k = x
            neg_LL = 0
            # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
            likelihood_observed = c * 1/gamma(k) * lambd**k * C**(k-1) * numpy.exp(-lambd*C)
            # CDF of gamma: 1.0 / gamma(k) * gammainc(k, lambda * t)
            likelihood_censored = (1 - c) + c * (1 - gammainc(k, lambd*N))
            neg_LL = -numpy.sum(numpy.log(B * likelihood_observed + (1 - B) * likelihood_censored))
            return neg_LL

        c_initial = numpy.mean(B)
        lambd_initial = 1.0 / max(C)
        lambd_max = 30.0 / max(C)
        k_initial = 10.0
        lambd = self.params.get('lambd')
        k = self.params.get('k')
        res = scipy.optimize.minimize(
            fun=f,
            x0=(c_initial, lambd_initial, k_initial),
            bounds=((1e-4, 1-1e-4),
                    (lambd, lambd) if lambd else (1e-4, lambd_max),
                    (k, k) if k else (1.0, 30.0)),
            method='L-BFGS-B')
        c, lambd, k = res.x
        self.params = dict(c=c, lambd=lambd, k=k)

    def predict(self, ts):
        c, lambd, k = self.params['c'], self.params['lambd'], self.params['k']
        return ts, c * gammainc(k, lambd*ts)

    def predict_final(self):
        return self.params['c']


class Bootstrapper(Model):
    def __init__(self, base_fitter, n_bootstraps=100):
        self.models = [base_fitter() for i in range(n_bootstraps)]

    def fit(self, C, N, B):
        CNB = list(zip(C, N, B))
        for model in self.models:
            CNB_bootstrapped = [random.choice(CNB) for _ in CNB]
            C_bootstrapped = numpy.array([c for c, n, b in CNB_bootstrapped])
            N_bootstrapped = numpy.array([n for c, n, b in CNB_bootstrapped])
            B_bootstrapped = numpy.array([b for c, n, b in CNB_bootstrapped])
            model.fit(C_bootstrapped, N_bootstrapped, B_bootstrapped)

    def predict(self, ts, confidence_interval=False):
        all_ts = numpy.array([model.predict(ts)[1] for model in self.models])
        if confidence_interval:
            return (ts, numpy.mean(all_ts, axis=0), numpy.percentile(all_ts, 5, axis=0), numpy.percentile(all_ts, 95, axis=0))
        else:
            return (ts, numpy.mean(all_ts, axis=0))

    def predict_final(self, confidence_interval=False):
        all_ps = numpy.array([model.predict_final() for model in self.models])
        if confidence_interval:
            return (numpy.mean(all_ps), numpy.percentile(all_ps, 5), numpy.percentile(all_ps, 95))
        else:
            return numpy.mean(all_ps)


def split_by_group(data, group_min_size, max_groups):
    js = {}
    for group, created_at, converted_at, now in data:
        if converted_at is not None and converted_at < created_at:
            print('created at', created_at, 'but converted at', converted_at)
            continue
        js.setdefault(group, []).append((created_at, converted_at, now))

    # Remove groups with too few data points
    groups = [group for group, data_points in js.items() if len(data_points) >= group_min_size]

    # Require at least one conversion per group
    groups = [group for group, data_points in js.items() if any(converted_at for _, converted_at, _ in data_points) > 0]

    # Pick the top groups
    groups = sorted(groups, key=lambda group: len(js[group]), reverse=True)[:max_groups]

    # Sort groups lexicographically
    groups = sorted(groups)

    return groups, js


def get_params(js, projection, share_params, t_factor):
    if share_params:
        if projection == 'exponential':
            m = Exponential()
        elif projection == 'gamma':
            m = Gamma()
        else:
            raise Exception('sharing params only works if projection is exponential/gamma (was: %s)' % projection)
        pooled_data = sum(js.values(), [])
        C, N, B = get_arrays(pooled_data, t_factor)
        m.fit(C, N, B)
        if share_params is True:
            return {k: m.params[k] for k in ['k', 'lambd'] if k in m.params}
        else:
            return {k: m.params[k] for k in params}
    else:
        return {}


def plot_cohorts(data, t_max=None, title=None, group_min_size=0, max_groups=100, model='kaplan-meier', projection=None, share_params=False):
    # Set x scale
    if t_max is None:
        t_max = max(now - created_at for group, created_at, converted_at, now in data)
    t_factor, t_unit = get_timescale(t_max)
    t_max = t_max.total_seconds() * t_factor

    # Split data by group
    groups, js = split_by_group(data, group_min_size, max_groups)

    # Get shared params
    params = get_params(js, projection, share_params, t_factor)

    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    y_max = 0
    for group, color in zip(sorted(groups), colors):
        C, N, B = get_arrays(js[group], t_factor)
        t = numpy.linspace(0, t_max, 1000)

        if model == 'basic':
            m = Basic()
        elif model == 'kaplan-meier':
            m = KaplanMeier()
        m.fit(C, N, B)

        label = '%s (n=%.0f, k=%.0f)' % (group, len(B), sum(B))

        if projection is True:
            p = m
        else:
            if projection == 'exponential':
                p = Bootstrapper(lambda: Exponential(params=params))
            elif projection == 'gamma':
                p = Bootstrapper(lambda: Gamma(params=params))
            else:
                raise Exception('projection must be exponential/gamma (was: %s)' % projection)
            p.fit(C, N, B)

        if projection:
            p_t, p_y, p_y_lo, p_y_hi = p.predict(t, confidence_interval=True)
            p_y_final, p_y_lo_final, p_y_hi_final = p.predict_final(confidence_interval=True)
            label += ' projected: %.2f%% (%.2f%% - %.2f%%)' % (100.*p_y_final, 100.*p_y_lo_final, 100.*p_y_hi_final)
            y_max = max(y_max, 90. * max(p_y_hi))
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
    pyplot.tight_layout()


def plot_conversion(data, window, projection, group_min_size=0, max_groups=100, stride=None, share_params=False, title=None):
    if stride is None:
        stride = window

    # Find limits
    t_lo = min(created_at for _, created_at, _, _ in data)
    t_hi = min(now for _, _, _, now in data)
    t_factor, t_unit = get_timescale(t_hi - t_lo)

    # Split data by group
    groups, js = split_by_group(data, group_min_size, max_groups)

    # Get shared params
    params = get_params(js, projection, share_params, t_factor)

    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    for group, color in zip(sorted(groups), colors):
        t1 = t_lo
        ts, ys, y_los, y_his = [], [], [], []
        js[group].sort()
        created_ats = [created_at for created_at, _, _ in js[group]]
        while True:
            t2 = t1 + window
            i1 = bisect.bisect_left(created_ats, t1)
            i2 = bisect.bisect_left(created_ats, t2)
            if i1 >= len(js[group]):
                break
            data = js[group][i1:i2]
            t1 += stride

            C, N, B = get_arrays(data, t_factor)
            if sum(B) == 0:
                continue

            if projection == 'exponential':
                p = Bootstrapper(lambda: Exponential(params=params))
            elif projection == 'gamma':
                p = Bootstrapper(lambda: Gamma(params=params))
            else:
                raise Exception('projection must be exponential/gamma (was: %s)' % projection)
            p.fit(C, N, B)

            y, y_lo, y_hi = p.predict_final(confidence_interval=True)
            print('%30s %40s %.4f %.4f %.4f' % (group, t1, y, y_lo, y_hi))
            ts.append(t2)
            ys.append(y)
            y_los.append(y_lo)
            y_his.append(y_hi)

        ys, y_los, y_his = (100.*numpy.array(x) for x in (ys, y_los, y_his))
        pyplot.plot(ts, ys, color=color, label=group)
        pyplot.fill_between(ts, y_los, y_his, color=color, alpha=0.2)

    if title:
        pyplot.title(title)
    pyplot.ylabel('Conversion rate %')
    pyplot.legend()
    pyplot.gca().grid(True)
    pyplot.tight_layout()
