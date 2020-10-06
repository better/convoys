from deprecated.sphinx import deprecated
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

    def fit(self, G, B, T):
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
            X[i,group] = 1 # one hot encoded boolean mask indicating group 
        self.base_model.fit(X, B, T)

    def _get_x(self, group): # for each individual rows 
        x = numpy.zeros(self._n_groups)
        x[group] = 1
        return x

    def predict(self, group, t):
        return self.base_model.predict(self._get_x(group), t)

    def predict_ci(self, group, t, ci):
        return self.base_model.predict_ci(self._get_x(group), t, ci)

    def rvs(self, group, *args, **kwargs):
        return self.base_model.rvs(self._get_x(group), *args, **kwargs)

    @deprecated(version='0.2.0',
                reason='Use :meth:`predict` or :meth:`predict_ci` instead.')
    def cdf(self, group, t, ci=None):
        '''Returns the predicted values.'''
        if ci is not None:
            return self.predict_ci(group, t, ci)
        else:
            return self.predict(group, t)


class SingleToMulti(MultiModel):
    def __init__(self, *args, **kwargs):
        self.base_model_init = lambda: self._base_model_cls(*args, **kwargs)

    def fit(self, G, B, T):
        ''' Fits the model

        :param G: numpy vector of shape :math:`n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        '''
        group2bt = {}
        for g, b, t in zip(G, B, T):
            group2bt.setdefault(g, []).append((b, t)) # convert the values into individual item in a dictionary {G_value, [B_value, T_value]}
        self._group2model = {}
        for g, BT in group2bt.items():
            self._group2model[g] = self.base_model_init()
            self._group2model[g].fit([b for b, t in BT], [t for b, t in BT])

    def predict(self, group, t):
        return self._group2model[group].predict(t)

    def predict_ci(self, group, t, ci):
        return self._group2model[group].predict_ci(t, ci)

    @deprecated(version='0.2.0',
                reason='Use :meth:`predict` or :meth:`predict_ci` instead')
    def cdf(self, group, t, ci=None):
        '''Returns the predicted values.'''
        if ci is not None:
            return self.predict_ci(group, t, ci)
        else:
            return self.predict(group, t)


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
