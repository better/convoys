import pandas 
from convoys.utils import get_arrays
from convoys.plotting import plot_cohorts
from convoys.multi import Weibull
df = pandas.read_pickle('examples/dob_violations.pickle')
unit, groups, (G, B, T) = get_arrays(
    df, groups='type', created='issue_date', converted='disposition_date',
    unit='years', group_min_size=100)

plot_cohorts(G, B, T, model='kaplan-meier', ci=0.95, groups=groups)
model=Weibull()
model.fit(G,B,T)