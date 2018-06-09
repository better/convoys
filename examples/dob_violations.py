from matplotlib import pyplot
import pickle
import convoys.plotting
import convoys.utils


def run():
    print('loading data')
    f = open('examples/dob_violations.pickle', 'rb')
    df = pickle.load(f)
    df = df.sample(n=10000)  # Subsample to make it faster
    print(df)

    print('converting to arrays')
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='type', created='issue_date', converted='disposition_date',
        unit='Years', group_min_size=100)

    for model in ['kaplan-meier', 'weibull']:
        print('plotting', model)
        pyplot.clf()
        convoys.plotting.plot_cohorts(G, B, T, model=model, ci=0.95, groups=groups, t_max=30)
        pyplot.legend()
        pyplot.savefig('dob-violations-%s.png' % model)


if __name__ == '__main__':
    run()
