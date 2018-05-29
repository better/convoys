import datetime
import pandas


def get_timescale(t):
    ''' Take a datetime or a numerical type, return two things:

    1. A unit
    2. A function that converts it to numerical form
    '''
    def get_timedelta_converter(t_factor):
        return lambda td: td.total_seconds() * t_factor

    if not isinstance(t, datetime.timedelta):
        # Assume numeric type
        return '', lambda x: x
    elif t >= datetime.timedelta(days=1):
        return 'Days', get_timedelta_converter(1./(24*60*60))
    elif t >= datetime.timedelta(hours=1):
        return 'Hours', get_timedelta_converter(1./(60*60))
    elif t >= datetime.timedelta(minutes=1):
        return 'Minutes', get_timedelta_converter(1./60)
    else:
        return 'Minutes', get_timedelta_converter(1)


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
    return sorted(groups)


def get_arrays(data, features=None, groups=None, created=None,
               converted=None, now=None, group_min_size=0, max_groups=-1):
    ''' Converts a dataframe to a list of numpy arrays.

    Each input refers to a column in the dataframe.

    TODO: more doc
    '''
    res = []

    # First, construct either the `X` or the `G` array
    if features is None and groups is None:
        if 'groups' in data.columns:
            groups = 'groups'
        elif 'features' in data.columns:
            features = 'features'
        else:
            raise Exception('Neither of the `features` or `groups` parameters was'
                            ' provided, and there was no `features` or `groups`'
                            ' dataframe column')
    if groups is not None:
        group2j = dict((group, j) for j, group in enumerate(get_groups(data[groups], group_min_size, max_groups)))
        data = data[data[groups].isin(group2j.keys())]  # Remove rows for rare groups
        G = data[groups].apply(lambda g: group2j[g]).values
        res.append(G)
    else:
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
    T_raw = []
    for i, row in data.iterrows():
        # TODO: this stuff should be vectorized, kind of ugly
        if not pandas.isnull(row[converted]):
            if created is not None:
                T_raw.append(row[converted] - row[created])
            else:
                T_raw.append(row[converted])
        else:
            if created is not None:
                if now is not None:
                    T_raw.append(row[now] - row[created])
                else:
                    T_raw.append(datetime.datetime.now(tzinfo=row[created].tzinfo) - row[created_at])
            else:
                T_raw.append(row[now])
    unit, converter = get_timescale(max(T_raw))
    T = [converter(t) for t in T_raw]
    res.append(T)

    return unit, tuple(res)
