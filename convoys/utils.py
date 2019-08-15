import datetime
import pandas

__all__ = ['get_arrays']


def get_timescale(t, unit):
    ''' Take a datetime or a numerical type, return two things:

    1. A unit
    2. A function that converts it to numerical form
    '''
    def get_timedelta_converter(t_factor):
        return lambda td: td.total_seconds() * t_factor

    if not isinstance(t, datetime.timedelta):
        # Assume numeric type
        return '', lambda x: x
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

    Each input refers to a column in the dataframe.

    TODO: more doc
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
        groups_list = []
        X = data[features].values
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
                    return _sub(datetime.datetime.now(), row[created])
            else:
                return row[now]

    T_raw = data.apply(lambda x: _calculate_T(x), axis=1)
    unit, converter = get_timescale(max(T_raw), unit)
    T = [converter(t) for t in T_raw]
    res.append(T)
    return unit, groups_list, tuple(res)

