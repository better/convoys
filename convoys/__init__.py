import abc
import datetime
import lifelines
import math
import numpy
import random
import seaborn
from matplotlib import pyplot
from convoys.multi import Exponential, Weibull, Gamma, KaplanMeier, Nonparametric


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


def get_arrays(groups, data, t_converter):
    G, B, T = [], [], []
    group2j = dict((group, j) for j, group in enumerate(groups))
    for group, created_at, converted_at, now in data:
        if converted_at is not None and converted_at < created_at:
            print('created at', created_at, 'but converted at', converted_at)
            continue
        if now < created_at:
            print('created at', created_at, 'but now is', now)
            continue
        if group in group2j:
            G.append(group2j[group])
            B.append(converted_at is not None)
            T.append(t_converter(converted_at - created_at) if converted_at is not None else t_converter(now - created_at))
    return numpy.array(G), numpy.array(B), numpy.array(T)


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


def get_groups(data, group_min_size, max_groups):
    group2count = {}
    for group, created_at, converted_at, now in data:
        group2count[group] = group2count.get(group, 0) + 1

    # Remove groups with too few data points
    # Pick the top groups
    # Sort groups lexicographically
    groups = [group for group, count in group2count.items() if count >= group_min_size]
    groups = sorted(groups, key=group2count.get, reverse=True)[:max_groups]
    return sorted(groups)


_models = {
    'kaplan-meier': KaplanMeier,
    'nonparametric': Nonparametric,
    'exponential': Exponential,
    'weibull': Weibull,
    'gamma': Gamma,
}


def plot_cohorts(data, t_max=None, title=None, group_min_size=0, max_groups=100, model='kaplan-meier', extra_model=None):
    # Set x scale
    if t_max is None:
        t_max = max(now - created_at for group, created_at, converted_at, now in data)
    t_unit, t_converter = get_timescale(t_max)
    t_max = t_converter(t_max)

    # Split data by group and get data
    groups = get_groups(data, group_min_size, max_groups)
    G, B, T = get_arrays(groups, data, t_converter)

    # Fit model
    m = _models[model]()
    m.fit(G, B, T)
    if extra_model is not None:
        extra_m = _models[extra_model]()
        extra_m.fit(G, B, T)

    # Plot
    colors = seaborn.color_palette('hls', len(groups))
    t = numpy.linspace(0, t_max, 1000)
    y_max = 0
    result = []
    for j, (group, color) in enumerate(zip(groups, colors)):
        n = sum(1 for g in G if g == j)  # TODO: slow
        k = sum(1 for g, b in zip(G, B) if g == j and b)  # TODO: slow
        label = '%s (n=%.0f, k=%.0f)' % (group, n, k)
        p_y, p_y_lo, p_y_hi = m.predict(j, t, ci=0.95).T
        p_y_final, p_y_lo_final, p_y_hi_final = m.predict_final(j, ci=0.95)
        label += ' projected: %.2f%% (%.2f%% - %.2f%%)' % (100.*p_y_final, 100.*p_y_lo_final, 100.*p_y_hi_final)
        pyplot.plot(t, 100. * p_y, color=color, alpha=0.7, label=label)
        pyplot.fill_between(t, 100. * p_y_lo, 100. * p_y_hi, color=color, alpha=0.2)
        if extra_model is not None:
            extra_p_y = extra_m.predict(j, t)
            pyplot.plot(t, 100. * extra_p_y, color=color, linestyle=':', alpha=0.7)
        result.append((group, p_y_final, p_y_lo_final, p_y_hi_final))
        y_max = max(y_max, 110. * max(p_y))

    if title:
        pyplot.title(title)
    pyplot.xlim([0, t_max])
    pyplot.ylim([0, y_max])
    pyplot.xlabel(t_unit)
    pyplot.ylabel('Conversion rate %')
    pyplot.legend()
    pyplot.gca().grid(True)
    return result
