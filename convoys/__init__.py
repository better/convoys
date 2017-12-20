import abc
import bisect
import datetime
import lifelines
import math
import numpy
import random
import seaborn
import scipy.optimize
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


def datetime_to_float(data):
    deltas = [now - created_at for created_at, converted_at, now in data]
    t = numpy.percentile(deltas, 90)
    t_factor, t_unit = get_timescale(t)
    T = [(converted_at - created_at).total_seconds() * t_factor if converted_at is not None else (now - created_at).total_seconds() * t_factor
         for created_at, converted_at, now in data]
    E = [bool(converted_at is not None)
         for created_at, converted_at, now in data]
    return numpy.array(T), numpy.array(E), t_factor, t_unit


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, T, E):
        pass

    @abc.abstractmethod
    def predict(self, ts, confidence_interval=False):
        pass


class KaplanMeier(Model):
    def fit(self, T, E):
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(T, event_observed=E)
        self.ts = kmf.survival_function_.index.values
        self.ps = 1.0 - kmf.survival_function_['KM_estimate'].values
        self.ps_hi = 1.0 - kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        self.ps_lo = 1.0 - kmf.confidence_interval_['KM_estimate_upper_0.95'].values

    def predict(self, ts, confidence_interval=False):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[j] if j < len(a) else float('nan') for j in js])
        if confidence_interval:
            return (array_lookup(self.ps), array_lookup(self.ps_lo), array_lookup(self.ps_hi))
        else:
            return array_lookup(self.ps)


class Exponential(Model):
    def fit(self, T, E, params={}):
        def f(x):
            c, lambd = x
            neg_LL, neg_LL_deriv_c, neg_LL_deriv_lambd = 0, 0, 0
            likelihood_observed = c * lambd * numpy.exp(-lambd*T)
            likelihood_censored = (1 - c) + c * numpy.exp(-lambd*T)
            neg_LL = -numpy.sum(numpy.log(E * likelihood_observed + (1 - E) * likelihood_censored))
            neg_LL_deriv_c = -numpy.sum(E * 1/c + (1 - E) * (-1 + numpy.exp(-lambd*T)) / likelihood_censored)
            neg_LL_deriv_lambd = -numpy.sum(E * (1/lambd - T) + (1 - E) * (c * -T * numpy.exp(-lambd*T)) / likelihood_censored)
            return neg_LL, numpy.array([neg_LL_deriv_c, neg_LL_deriv_lambd])

        c_initial = numpy.mean(E)
        lambd_initial = 1.0 / max(T)
        lambd_max = 30.0 / max(T)
        lambd = params.get('lambd')
        res = scipy.optimize.minimize(
            fun=f,
            x0=(c_initial, lambd_initial),
            bounds=((1e-4, 1-1e-4),
                    (lambd, lambd) if lambd else (1e-4, lambd_max)),
            method='L-BFGS-B',
            jac=True)
        c, lambd = res.x
        self.params = dict(c=c, lambd=lambd)

    def predict(self, t):
        c, lambd = self.params['c'], self.params['lambd']
        return c * (1 - numpy.exp(-t * lambd))


class Gamma(Model):
    def fit(self, T, E, params={}):
        # TODO(erikbern): should compute Jacobian of this one
        def f(x):
            c, lambd, k = x
            neg_LL = 0
            # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
            likelihood_observed = c * 1/gamma(k) * lambd**k * T**(k-1) * numpy.exp(-lambd*T)
            # CDF of gamma: 1.0 / gamma(k) * gammainc(k, lambda * t)
            likelihood_censored = (1 - c) + c * (1 - gammainc(k, lambd*T))
            neg_LL = -numpy.sum(numpy.log(E * likelihood_observed + (1 - E) * likelihood_censored))
            return neg_LL

        c_initial = numpy.mean(E)
        lambd_initial = 1.0 / max(T)
        lambd_max = 30.0 / max(T)
        k_initial = 10.0
        lambd = params.get('lambd')
        k = params.get('k')
        res = scipy.optimize.minimize(
            fun=f,
            x0=(c_initial, lambd_initial, k_initial),
            bounds=((1e-4, 1-1e-4),
                    (lambd, lambd) if lambd else (1e-4, lambd_max),
                    (k, k) if k else (1.0, 30.0)),
            method='L-BFGS-B')
        c, lambd, k = res.x
        self.params = dict(c=c, lambd=lambd, k=k)

    def predict(self, t):
        c, lambd, k = self.params['c'], self.params['lambd'], self.params['k']
        return c * gammainc(k, lambd*t)


class Bootstrapper(Model):
    def __init__(self, base_fitter, n_bootstraps=100):
        self.models = [base_fitter() for i in range(n_bootstraps)]

    def fit(self, T, E):
        TE = list(zip(T, E))
        for model in self.models:
            TE_bootstrapped = [random.choice(TE) for _ in TE]
            T_bootstrapped = numpy.array([t for t, e in TE_bootstrapped])
            E_bootstrapped = numpy.array([e for t, e in TE_bootstrapped])
            model.fit(T_bootstrapped, E_bootstrapped)

    def predict(self, ts, confidence_interval=False):
        all_ts = numpy.array([model.predict(ts) for model in self.models])
        if confidence_interval:
            return (numpy.mean(all_ts, axis=0),
                    numpy.percentile(all_ts, 5, axis=0),
                    numpy.percentile(all_ts, 95, axis=0))
        else:
            return numpy.mean(all_ts, axis=0)


def plot_conversion(data, title, group_min_size=0, max_groups=100, model='kaplan-meier', share_params=False):
    js = {}
    for group, created_at, converted_at, now in data:
        if converted_at is not None and converted_at < created_at:
            # TODO: move this into the preprocessing functions
            print('created at', created_at, 'but converted at', converted_at)
            converted = None
        js.setdefault(group, []).append((created_at, converted_at, now))

    if share_params:
        # TODO: Fit shared parameters for all models if requested
        raise

    # Remove groups with too few data points
    groups = [group for group, data_points in js.items() if len(data_points) >= group_min_size]

    # Require at least one conversion per group
    groups = [group for group, data_points in js.items() if any(converted_at for _, converted_at, _ in data_points) > 0]

    # Pick the top groups
    groups = sorted(groups, key=lambda group: len(js[group]), reverse=True)[:max_groups]

    # Sort groups lexicographically
    groups = sorted(groups)

    xlim = 180  # TODO FIX

    if share_params:
        # TODO: pool data, fit base parameters
        pass
    shared_params = {}
    
    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    for group, color in zip(sorted(groups), colors):
        print(group)
        if model == 'kaplan-meier':
            m = KaplanMeier()
        elif model == 'exponential':
            m = Bootstrapper(lambda: Exponential(params=shared_params))
        elif model == 'gamma':
            m = Bootstrapper(lambda: Gamma(params=shared_params))
        T, E, t_factor, t_unit = datetime_to_float(js[group]) # TODO: do outside this loop
        m.fit(T, E)
        #ylim = max(ylim, 90. * p_hi[i], 110. * p[i])
        #xlim = max(xlim, min(t_max, t[i]))

        # label = '%s (n=%.0f, k=%.0f): %.1f%% (%.1f-%.1f%%)' % (group, len(js[group]), int(len(js[group]) * p[i]), 100. * p[i], 100.*p_lo[i], 100.*p_hi[i])
        t = numpy.linspace(0, xlim, 100)
        p, p_lo, p_hi = m.predict(t, confidence_interval=True)
        m2 = KaplanMeier()
        m2.fit(T, E)
        p = m2.predict(t)
        
        pyplot.plot(t, 100. * p, color=color, label=group)
        pyplot.fill_between(t, 100. * p_lo, 100. * p_hi, color=color, alpha=0.2)

    pyplot.title(title)
    #pyplot.xlim([0, xlim])
    #pyplot.ylim([0, ylim])
    #pyplot.xlabel(t_unit)
    pyplot.ylabel('Conversion rate %')
    pyplot.legend()
    pyplot.gca().grid(True)
    pyplot.tight_layout()
