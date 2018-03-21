import numpy
from convoys import regression
from convoys import single


class MultiModel:
    pass  # TODO


class RegressionToMulti(MultiModel):
    def __init__(self, *args, **kwargs):
        self._base_model = self._base_model_cls(*args, **kwargs)

    def fit(self, G, B, T):
        self._n_groups = max(G) + 1
        X = numpy.zeros((len(G), self._n_groups+1))
        X[:,0] = 1.0
        for i, group in enumerate(G):
            X[i,group+1] = 1
        self._base_model.fit(X, B, T)

    def _get_x(self, group):
        x = numpy.zeros(self._n_groups+1)
        x[0] = 1
        x[group+1] = 1
        return x

    def predict(self, group, t, *args, **kwargs):
        return self._base_model.predict(self._get_x(group), t, *args, **kwargs)

    def predict_final(self, group, *args, **kwargs):
        return self._base_model.predict_final(self._get_x(group), *args, **kwargs)

    def predict_time(self, group, *args, **kwargs):
        return self._base_model.predict_time(self._get_x(group), *args, **kwargs)


class SingleToMulti(MultiModel):
    def __init__(self, *args, **kwargs):
        self._base_model_init = lambda: self._base_model_cls(*args, **kwargs)

    def fit(self, G, B, T):
        group2bt = {}
        for g, b, t in zip(G, B, T):
            group2bt.setdefault(g, []).append((b, t))
        self._group2model = {}
        for g, BT in group2bt.items():
            self._group2model[g] = self._base_model_init()
            self._group2model[g].fit([b for b, t in BT], [t for b, t in BT])

    def predict(self, group, t, *args, **kwargs):
        return self._group2model[group].predict(t, *args, **kwargs)

    def predict_final(self, group, *args, **kwargs):
        return self._group2model[group].predict_final(*args, **kwargs)

    def predict_time(self, group, *args, **kwargs):
        return self._group2model[group].predict_time(*args, **kwargs)


class Exponential(RegressionToMulti):
    _base_model_cls = regression.Exponential


class Weibull(RegressionToMulti):
    _base_model_cls = regression.Weibull


class Gamma(RegressionToMulti):
    _base_model_cls = regression.Gamma


class Nonparametric(SingleToMulti):
    _base_model_cls = single.Nonparametric
