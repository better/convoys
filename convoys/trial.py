import pandas 
from convoys.utils import get_arrays,get_groups,get_arrays_for_double_hierachy
from convoys.regression import double_hierarchy_weibull_loss, generalized_gamma_loss,get_probabilities
from convoys.plotting import plot_cohorts
from convoys.multi import Weibull
from autograd import numpy 
df = pandas.read_pickle('examples/dob_violations.pickle')
unit, groups, (G, B, T) = get_arrays(
    df, groups='type', created='issue_date', converted='disposition_date',
    unit='years', group_min_size=1000)

# plot_cohorts(G, B, T, model='kaplan-meier', ci=0.95, groups=groups)
model = Weibull()
model.fit(G,B,T)

unit, groups_list1,groups_list2, (X,B,T)=get_arrays_for_double_hierachy(df, group1='category',group2='type', group_min_size1=3000,group_min_size2=3000,max_groups_1=-1, 
               max_groups_2=-1, created='issue_date',
               converted='disposition_date', now=None, unit='years')

_,n_group1,n_group2 = X.shape
n_group = n_group1*n_group2
x = numpy.zeros(8+2*n_group1+2*n_group)
fix_k = 1
fix_p = None