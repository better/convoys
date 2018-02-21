import abc
import bisect
import datetime
import lifelines
import math
import numpy
import random
import seaborn
import six
from matplotlib import pyplot
from convoys.model import Model
from convoys.regression import ExponentialRegression, WeibullRegression, GammaRegression


def get_timescale(t):
    def get_timedelta_converter(t_factor):
        return lambda td: td.total_seconds() * t_factor

    if type(t) != datetime.timedelta:
        # Assume numeric type
        return '', lambda x: x
    elif t >= datetime.timedelta(days=1):
        return 'Days', get_timedelta_converter(1./(24*60*60))
    elif t >= datetime.timedelta(hours=1):
        return 'Hours', get_timedelta_converter(1./(60*60))
    elif t >= datetime.timedelta(minutes=1):
        return 'Minutes', get_timedelta_converter(1./60)
    else:
        return 'Minutes', get_timedelta_converter(1)


class KaplanMeier(Model):
    # TODO: This model really isn't built for regression so right now we fake it
    def fit(self, X, B, T):
        self.ts = []
        self.ps = []
        self.ps_hi = []
        self.ps_lo = []
        for i in range(X.shape[1]):
            kmf = lifelines.KaplanMeierFitter()
            kmf.fit([t for x, t in zip(X, T) if x[i]], event_observed=[b for x, b in zip(X, B) if x[i]])
            self.ts.append(kmf.survival_function_.index.values)
            self.ps.append(1.0 - kmf.survival_function_['KM_estimate'].values)
            self.ps_hi.append(1.0 - kmf.confidence_interval_['KM_estimate_lower_0.95'].values)
            self.ps_lo.append(1.0 - kmf.confidence_interval_['KM_estimate_upper_0.95'].values)

    def predict(self, x, ts, ci=None):
        i = next(i for i, z in enumerate(x) if i > 0 and z > 0)  # TODO: this is terrible
        js = [bisect.bisect_left(self.ts[i], t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[j] for j in js if j < len(self.ts[i])])
        if ci is not None:
            return (array_lookup(self.ts[i]), array_lookup(self.ps[i]), array_lookup(self.ps_lo[i]), array_lookup(self.ps_hi[i]))
        else:
            return (array_lookup(self.ts[i]), array_lookup(self.ps[i]))

    def predict_final(self, x, ci=None):
        i = next(i for i, z in enumerate(x) if i > 0 and z > 0)  # TODO: this is terrible
        if ci is not None:
            return (self.ps[i][-1], self.ps_lo[i][-1], self.ps_hi[i][-1])
        else:
            return self.ps[i][-1]

    def predict_time(self, x, ci=None):
        raise


def sample_event(model, x, t, hi=1e3):
    # We are now at time t. Generate a random event whether the user is going to convert or not
    # TODO: this is a hacky thing until we have a "invert CDF" method on each model
    def pred(t):
        ts = numpy.array([t])
        return model.predict(x, ts)[1][-1]
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


def split_by_group(data, group_min_size, max_groups, t_converter):
    data_by_group = {}
    for group, created_at, converted_at, now in data:
        if converted_at is not None and converted_at < created_at:
            print('created at', created_at, 'but converted at', converted_at)
            continue
        if now < created_at:
            print('created at', created_at, 'but now is', now)
            continue
        data_by_group.setdefault(group, []).append((
            bool(converted_at is not None),
            t_converter(converted_at - created_at) if converted_at is not None else t_converter(now - created_at)
        ))
    groups = list(data_by_group.keys())

    # Remove groups with too few data points
    groups = [group for group in groups if len(data_by_group[group]) >= group_min_size]

    # Require at least one conversion per group
    groups = [group for group in groups if any(b for t, b in data_by_group[group])]

    # Pick the top groups
    groups = sorted(groups, key=lambda group: len(data_by_group[group]), reverse=True)[:max_groups]

    # Sort groups lexicographically
    groups = sorted(groups)

    # Build dummy one hot vectors
    one_hots = []
    for i, group in enumerate(groups):
        one_hots.append(numpy.array([1] + [int(j == i) for j in range(len(groups))]))

    # Build matrices
    XBT = []
    ns, ks = [], []
    for i, group in enumerate(groups):
        for b, t in data_by_group[group]:
            XBT.append((one_hots[i], b, t))
        ns.append(len(data_by_group[group]))
        ks.append(sum(b for b, t in data_by_group[group]))
    random.shuffle(XBT)  # Just in case
    X = numpy.array([x for x, b, t in XBT])
    B = numpy.array([b for x, b, t in XBT])
    T = numpy.array([t for x, b, t in XBT])

    return groups, one_hots, X, B, T, ns, ks


_models = {
    'kaplan-meier': KaplanMeier,
    'exponential': ExponentialRegression,
    'weibull': WeibullRegression,
    'gamma': GammaRegression,
}

def plot_cohorts(data, t_max=None, title=None, group_min_size=0, max_groups=100, model='kaplan-meier', projection=None):
    # Set x scale
    if t_max is None:
        t_max = max(now - created_at for group, created_at, converted_at, now in data)
    t_unit, t_converter = get_timescale(t_max)
    t_max = t_converter(t_max)

    # Split data by group and get matrices
    groups, one_hots, X, B, T, ns, ks = split_by_group(data, group_min_size, max_groups, t_converter)

    # Fit models
    m = _models[model]()
    m.fit(X, B, T)
    if projection:
        p = _models[projection]()
        p.fit(X, B, T)

    # PLOT
    colors = seaborn.color_palette('hls', len(groups))
    y_max = 0
    t = numpy.linspace(0, t_max, 1000)
    for group, color, x, n, k in zip(groups, colors, one_hots, ns, ks):
        label = '%s (n=%.0f, k=%.0f)' % (group, n, k)

        if projection:
            p_t, p_y, p_y_lo, p_y_hi = p.predict(x, t, ci=0.95)
            p_y_final, p_y_lo_final, p_y_hi_final = p.predict_final(x, ci=0.95)
            label += ' projected: %.2f%% (%.2f%% - %.2f%%)' % (100.*p_y_final, 100.*p_y_lo_final, 100.*p_y_hi_final)
            pyplot.plot(p_t, 100. * p_y, color=color, linestyle=':', alpha=0.7)
            pyplot.fill_between(p_t, 100. * p_y_lo, 100. * p_y_hi, color=color, alpha=0.2)

        m_t, m_y = m.predict(x, t)
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

            X, B, T = get_arrays(data, t_converter)
            if sum(B) < window_min_size:
                continue

            p = _models[model]()
            p.fit(X, B, T)

            if time:
                y, y_lo, y_hi = p.predict_time([1], ci=0.95)
            else:
                y, y_lo, y_hi = p.predict_final([1], ci=0.95)
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
