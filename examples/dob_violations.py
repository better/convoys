from matplotlib import pyplot
import datetime
import pandas
import convoys.plotting
import convoys.utils


def run():
    print('loading data')
    df = pandas.read_pickle('examples/dob_violations.pickle')
    print(df['issue_date'])
    print(df['issue_date'].dtype)
    print(df['issue_date'] < datetime.date(2018, 1, 1))
    df = df[df['issue_date'] < datetime.date(2018, 1, 1)]

    print('converting to arrays')
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='type', created='issue_date',
        converted='disposition_date',
        unit='years', group_min_size=100)

    for model in ['kaplan-meier', 'weibull']:
        print('plotting', model)
        pyplot.figure(figsize=(9, 6))
        convoys.plotting.plot_cohorts(G, B, T, model=model, ci=0.95,
                                      groups=groups, t_max=30)
        pyplot.legend()
        pyplot.xlabel(unit)
        pyplot.savefig('dob-violations-%s.png' % model)

    pyplot.figure(figsize=(9, 6))
    df['bucket'] = df['issue_date'].apply(
        lambda d: '%d-%d' % (5*(d.year//5), 5*(d.year//5)+4)
    )
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups='bucket', created='issue_date',
        converted='disposition_date',
        unit='years', group_min_size=500)
    convoys.plotting.plot_cohorts(G, B, T, model='kaplan-meier',
                                  groups=groups, t_max=30, ci=0.95)
    convoys.plotting.plot_cohorts(G, B, T, model='weibull',
                                  groups=groups, t_max=30,
                                  plot_kwargs={'linestyle': '--'})
    pyplot.legend()
    pyplot.xlabel(unit)
    pyplot.savefig('dob-violations-combined.png')


if __name__ == '__main__':
    run()
