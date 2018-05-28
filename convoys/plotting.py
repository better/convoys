import datetime
import numpy
from matplotlib import pyplot
import convoys.multi


_models = {
    'kaplan-meier': convoys.multi.KaplanMeier,
    'exponential': lambda: convoys.multi.Exponential(ci=True),
    'weibull': lambda: convoys.multi.Weibull(ci=True),
    'gamma': lambda: convoys.multi.Gamma(ci=True),
    'generalized-gamma': lambda: convoys.multi.GeneralizedGamma(ci=True),
}


def plot_cohorts(G, B, T, t_max=None, title=None, model='kaplan-meier', ci=0.95, extra_model=None):
    # Set x scale
    if t_max is None:
        t_max = max(T)

    groups = set(G)  # TODO: fix

    # Fit model
    m = _models[model]()
    m.fit(G, B, T)
    if extra_model is not None:
        extra_m = _models[extra_model]()
        extra_m.fit(G, B, T)

    # Plot
    colors = pyplot.get_cmap('tab10').colors
    colors = [colors[i % len(colors)] for i in range(len(groups))]
    t = numpy.linspace(0, t_max, 1000)
    y_max = 0
    result = []
    for j, (group, color) in enumerate(zip(groups, colors)):
        n = sum(1 for g in G if g == j)  # TODO: slow
        k = sum(1 for g, b in zip(G, B) if g == j and b)  # TODO: slow
        label = '%s (n=%.0f, k=%.0f)' % (group, n, k)

        if ci is not None:
            p_y, p_y_lo, p_y_hi = m.cdf(j, t, ci=ci).T
            pyplot.plot(t, 100. * p_y, color=color, linewidth=1.5, alpha=0.7, label=label)
            pyplot.fill_between(t, 100. * p_y_lo, 100. * p_y_hi, color=color, alpha=0.2)
        else:
            p_y = m.cdf(j, t).T
            pyplot.plot(t, 100. * p_y, color=color, linewidth=1.5, alpha=0.7, label=label)

        if extra_model is not None:
            extra_p_y = extra_m.cdf(j, t)
            pyplot.plot(t, 100. * extra_p_y, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        y_max = max(y_max, 110. * max(p_y))

    if title:
        pyplot.title(title)
    pyplot.xlim([0, t_max])
    pyplot.ylim([0, y_max])
    # pyplot.xlabel(t_unit)
    pyplot.ylabel('Conversion rate %')
    pyplot.legend()
    pyplot.gca().grid(True)
    return m
