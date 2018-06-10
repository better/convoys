from matplotlib import pyplot
import pickle
import convoys.plotting
import convoys.utils


def run():
    print('loading data')
    f = open('examples/marriage.pickle', 'rb')
    df = pickle.load(f)
    df = df.sample(n=10000)  # Subsample to make it faster
    print(df)

    _, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='sex', created='born', converted='married')

    pyplot.figure(figsize=(12, 9))
    convoys.plotting.plot_cohorts(G, B, T, model='generalized-gamma', groups=groups)
    pyplot.legend()
    convoys.plotting.plot_cohorts(G, B, T, model='kaplan-meier', groups=groups, plot_args={'linestyle': '--'})
    pyplot.savefig('marriage-combined.png')


if __name__ == '__main__':
    run()
