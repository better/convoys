import pandas 
from convoys.utils import get_arrays,get_groups
from convoys.plotting import plot_cohorts
from convoys.multi import Weibull
from autograd import numpy 
df = pandas.read_pickle('examples/dob_violations.pickle')
unit, groups, (G, B, T) = get_arrays(
    df, groups='type', created='issue_date', converted='disposition_date',
    unit='years', group_min_size=1000)

plot_cohorts(G, B, T, model='kaplan-meier', ci=0.95, groups=groups)
model=Weibull()
model.fit(G,B,T)



# for double hierarchy, parameters you need are 
# x, X, B, T, W, fix_k, fix_p,
# x is the single row vector storing all pararmeters and hyperparameters, length is 8+2*n_group1+2*n_group2
# X is (N, G1,G2) boolean mask
# B is the boolean indicator (N,)
# T is the time indicator (N,)
# W is sample weight,ignore for now
# fix_k is 1 to use Weibull for testing in the first iteration 
# fix_p is None
def get_arrays_for_double_hierachy(data, group1='category',group2='type', group_min_size1=1000,group_min_size2=1000,max_groups_1=-1, 
               max_groups_2=-1, created=None,
               converted=None, now=None, unit=None):
    res = []
    groups_list1 = get_groups(data[group1], group_min_size1, max_groups_1)
    groups_list2 = get_groups(data[group2], group_min_size2, max_groups_2)
    # first you want an indicator of group2 for each of the N datapoints. The best way is using pandas 
    if len(groups_list1)!=data[group1].nunique():
        groups_list1+=["unknown"]
        G_1 = pandas.get_dummies(pandas.Categorical(data[group1], categories=groups_list1).fillna('unknown'), drop_first=False)
    else:
         G_1 = pandas.get_dummies(pandas.Categorical(data[group1], categories=groups_list1), drop_first=False)
    group_list1 = G_1.columns.to_list()
  

    if len(groups_list2)!=data[group2].nunique():
        groups_list2+=["unknown"]
        G_2 = pandas.get_dummies(pandas.Categorical(data[group2], categories=groups_list2).fillna('unknown'), drop_first=False)
    else:
        G_2 = pandas.get_dummies(pandas.Categorical(data[group2], categories=groups_list2), drop_first=False)
    # TODO: use scipy sparse matrix because this is super memory-inefficient
    group_list2 = G_2.columns.to_list()
    G_2 = G_2.to_numpy()
    G_1=numpy.expand_dims(G_1,axis=-1)
    G_2=numpy.expand_dims(G_2,axis=-2)
    G_1=numpy.repeat(G_1,len(group_list2),axis=-1)
    G_2=numpy.repeat(G_2, len(group_list1),axis=-2)
    X=G_1*G_2 # you don't need to have G if you already get X matrix
    
    if now is None and 'now' in data.columns:
        now = 'now'
   
    B = ~pandas.isnull(data[converted]).values

    def _calculate_T(row):
        if not pandas.isnull(row[converted]):
            if created is not None:
                return _sub(row[converted], row[created])
            else:
                return row[converted]
        else:
            if created is not None:
                if now is not None:
                    return _sub(row[now], row[created])
                else:
                    return (datetime.datetime.now(tz=row[created].tzinfo)
                            - row[created])
            else:
                return row[now]

    T_deltas = data.apply(_calculate_T, axis=1)
    max_T_delta = T_deltas.max()
    unit, converter = get_timescale(max_T_delta, unit)
    T = T_deltas.apply(converter).to_numpy()
    res.append(T)
    return unit, groups_list, (X,B,T)

