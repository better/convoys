from matplotlib import pyplot
import pickle
import convoys.plotting
import convoys.utils


def run():
    print('loading data')
    f = open('examples/marriage.pickle', 'rb')
    df = pickle.load(f)
    df = df.sample(n=10000)  # Subsample to make it faster

    print('converting to arrays')
    _, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='race', created='born', converted='married', group_min_size=100)

    for model in ['kaplan-meier', 'generalized-gamma']:
        print('plotting', model)
        pyplot.clf()
        convoys.plotting.plot_cohorts(G, B, T, model=model, ci=0.95, groups=groups)
        pyplot.legend()
        pyplot.savefig('marriage-race-%s.png' % model)

    print('converting to arrays (for decades)')
    df = df[(1940 <= df['born']) & (df['born'] < 1990)]
    df['decade'] = df['born'].apply(lambda year: '%ds' % (10*(year//10)))
    _, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='decade', created='born', converted='married')
    print('plotting generalized-gamma')
    pyplot.clf()
    convoys.plotting.plot_cohorts(G, B, T, model='generalized-gamma', groups=groups)
    pyplot.legend()
    print('overlaying kaplan-meier nonparametric')
    convoys.plotting.plot_cohorts(G, B, T, model='kaplan-meier', groups=groups, plot_args={'linestyle': '--'})
    pyplot.savefig('marriage-decade.png')


if __name__ == '__main__':
    run()
