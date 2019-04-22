import numpy
from convoys import regression
from convoys import single

__all__ = ['KaplanMeier', 'Exponential', 'Weibull', 'Gamma',
           'GeneralizedGamma']


class MultiModel:
    pass  # TODO


class RegressionToMulti(MultiModel):
    def __init__(self, *args, **kwargs):
        self.base_model = self._base_model_cls(*args, **kwargs)

    def fit(self, G, B, T, verbose=False):
        ''' Fits the model

        :param G: numpy vector of shape :math:`n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        '''
        G = numpy.array(G, dtype=numpy.int)
        n, = G.shape
        self._n_groups = max(G) + 1
        X = numpy.zeros((n, self._n_groups), dtype=numpy.bool)
        for i, group in enumerate(G):
            X[i,group] = 1
        self.base_model.fit(X, B, T, verbose=verbose)

    def _get_x(self, group):
        x = numpy.zeros(self._n_groups)
        x[group] = 1
        return x

    def cdf(self, group, *args, **kwargs):
        return self.base_model.cdf(self._get_x(group), *args, **kwargs)

    def rvs(self, group, *args, **kwargs):
        return self.base_model.rvs(self._get_x(group), *args, **kwargs)


class SingleToMulti(MultiModel):
    def __init__(self, *args, **kwargs):
        self.base_model_init = lambda: self._base_model_cls(*args, **kwargs)

    def fit(self, G, B, T, verbose=False):
        ''' Fits the model

        :param G: numpy vector of shape :math:`n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        '''
        group2bt = {}
        for g, b, t in zip(G, B, T):
            group2bt.setdefault(g, []).append((b, t))
        self._group2model = {}
        for g, BT in group2bt.items():
            self._group2model[g] = self.base_model_init()
            self._group2model[g].fit([b for b, t in BT], [t for b, t in BT], verbose=verbose)

    def cdf(self, group, t, *args, **kwargs):
        return self._group2model[group].cdf(t, *args, **kwargs)


class Exponential(RegressionToMulti):
    ''' Multi-group version of :class:`convoys.regression.Exponential`.'''
    _base_model_cls = regression.Exponential


class Weibull(RegressionToMulti):
    ''' Multi-group version of :class:`convoys.regression.Weibull`.'''
    _base_model_cls = regression.Weibull


class Gamma(RegressionToMulti):
    ''' Multi-group version of :class:`convoys.regression.Gamma`.'''
    _base_model_cls = regression.Gamma


class GeneralizedGamma(RegressionToMulti):
    ''' Multi-group version of :class:`convoys.regression.GeneralizedGamma`.'''
    _base_model_cls = regression.GeneralizedGamma


class KaplanMeier(SingleToMulti):
    ''' Multi-group version of :class:`convoys.single.KaplanMeier`.'''
    _base_model_cls = single.KaplanMeier
