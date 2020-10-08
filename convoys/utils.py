import datetime
import numpy
import pandas
__all__ = ['get_arrays']


def get_timescale(t, unit):
    ''' Take a datetime or a numerical type, return two things:

    1. A unit
    2. A function that converts it to numerical form
    '''
    def get_timedelta_converter(t_factor):
        return lambda td: td.total_seconds() * t_factor

    if not isinstance(t, datetime.timedelta) or \
            not isinstance(t, pandas.Timedelta):
        # Assume numeric type
        return None, lambda x: float(x)
    for u, f in [('years', 365.25*24*60*60), ('days', 24*60*60),
                 ('hours', 60*60), ('minutes', 60), ('seconds', 1)]:
        if u == unit or (unit is None and t >= datetime.timedelta(seconds=f)):
            return u, get_timedelta_converter(1./f)
    raise Exception('Could not find unit for %f and %s' % (t, unit))


def get_groups(data, group_min_size, max_groups):
    ''' Picks the top groups out of a dataset

    1. Remove groups with too few data points
    2. Pick the top groups
    3. Sort groups lexicographically
    '''
    group2count = {}
    for group in data:
        group2count[group] = group2count.get(group, 0) + 1

    groups = [group for group, count in group2count.items() if count >= group_min_size]
    if max_groups >= 0:
        groups = sorted(groups, key=group2count.get, reverse=True)[:max_groups]
    return sorted(groups, key=lambda g: (g is None, g))  # Put Nones last


def _sub(a, b):
    # Computes a - b for a bunch of different cases
    if isinstance(a, datetime.datetime) and a.tzinfo is not None:
        return a.astimezone(b.tzinfo) - b
    else:
        # Either naive timestamps or numerical type
        return a - b


def get_arrays(data, features=None, groups=None, created=None,
               converted=None, now=None, unit=None,
               group_min_size=0, max_groups=-1):
    ''' Converts a dataframe to a list of numpy arrays.

    Generates either feature data, or group data.

    :param data: Pandas dataframe
    :param features: string (optional), refers to a column in the dataframe
        containing features, each being a 1d-vector or list of features.
        If not provided, then it it will look for a column in the dataframe
        named "features". This argument can also be a list of columns.
    :param groups: string (optional), refers to a column in the dataframe
        containing the groups for each row. If not provided, then it will
        look for a column in the dataframe named "groups".
    :param created: string (optional), refers to a column in the dataframe
        containing timestamps of when each item was "created". If not
        provided, then it will look for a column in the dataframe named
        "created".
    :param converted: string, refers to a column in the dataframe
        containing timestamps of when each item converted. If there is no
        column containing creation values, then the converted values should
        be timedeltas denoting time until conversion. If this argument is
        not provided, then it will look for a column in the dataframe named
        "created".
    :param now: string (optional), refers to a column in the dataframe
        containing the point in time up until which we have observed
        non-conversion. If there is no column containing creation value,
        then these values should be timedeltas. If this argument is not
        provided, the current timestamp will be used.
    :param unit: string (optional), time unit to use when converting to
        numerical values. Has to be one of "years", "days", "hours",
        "minutes", or "seconds". If not provided, then a choice will be
        made based on the largest time interval in the inputs.
    :param group_min_size: integer (optional), only include groups that
        has at least this many observations
    :param max_groups: integer (optional), only include the `n` largest
        groups
    :returns: tuple (unit, groups, arrays)

        `unit` is the unit chosen. Will be one of "years", "days", "hours",
        "minutes", or "seconds". If the `unit` parameter is passed, this
        will be the same.

        `groups` is a list of strings containing the groups. Will be `None`
         if `groups` is not set.

        `arrays` is a tuple of numpy arrays `(G, B, T)` or `(X, B, T)`
        containing the transformed input in numerical format. `G`, `B`, `T`
        will all be 1D numpy arrays. `X` will be a 2D numpy array.
    '''
    res = []

    # First, construct either the `X` or the `G` array
    if features is None and groups is None:
        if 'group' in data.columns:
            groups = 'group'
        elif 'features' in data.columns:
            features = 'features'
        else:
            raise Exception('Neither of the `features` or `group` parameters'
                            ' was provided, and there was no `features` or'
                            ' `groups` dataframe column')
    if groups is not None:
        groups_list = get_groups(data[groups], group_min_size, max_groups)
        group2j = dict((group, j) for j, group in enumerate(groups_list))
        # Remove rows for rare groups
        data = data[data[groups].isin(group2j.keys())]
        G = data[groups].apply(lambda g: group2j.get(g, -1)).values
        res.append(G)
    else:
        groups_list = None
        if type(features) == tuple:
            features = list(features)  # Otherwise sad Panda
        X = numpy.array([numpy.array(z) for z in data[features].values])
        res.append(X)

    # Next, construct the `B` and `T` arrays
    if converted is None:
        if 'converted' in data.columns:
            converted = 'converted'
        else:
            raise Exception('The `converted` parameter was not provided'
                            ' and there was no `converted` dataframe column')
    if now is None and 'now' in data.columns:
        now = 'now'
    if created is None and 'created' in data.columns:
        created = 'created'
    B = ~pandas.isnull(data[converted]).values
    res.append(B)

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
    return unit, groups_list, tuple(res)



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
               max_groups_2=-1, created='issue_date',
               converted='disposition_date', now=None, unit='years'):
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
    return unit, groups_list1,groups_list2, (X,B,T)


