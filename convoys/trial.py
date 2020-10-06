import pandas as pd
from convoys.utils import get_arrays
df = pd.read_pickle('examples/dob_violations.pickle')
unit, groups, (G, B, T) = get_arrays(
    df, groups='type', created='issue_date', converted='disposition_date',
    unit='years', group_min_size=100)

convoys.plotting.plot_cohorts(G, B, T, model='kaplan-meier', ci=0.95, groups=groups)
