from deprecated.sphinx import deprecated
import numpy
from scipy.special import expit, logit
import scipy.stats
import warnings

__all__ = ['KaplanMeier']


class SingleModel:
    pass  # TODO


class KaplanMeier(SingleModel):
    ''' Implementation of the Kaplan-Meier nonparametric method. '''
    def fit(self, B, T):
        ''' Fits the model

        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        '''
        # See https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
        BT = [(b, t) for b, t in zip(B, T)
              if t >= 0 and 0 <= float(b) <= 1]
        if len(BT) < len(B):
            n_removed = len(B) - len(BT)
            warnings.warn('Warning! Removed %d/%d entries from inputs where '
                          'T < 0 or B not 0/1' % (n_removed, len(B)))
        B, T = ([z[i] for z in BT] for i in range(2))
        n = len(T)
        self._ts = [0.0]
        self._ss = [1.0]
        self._vs = [0.0]
        sum_var_terms = 0.0
        prod_s_terms = 1.0
        for t, b in sorted(zip(T, B)):
            d = float(b)
            self._ts.append(t)
            prod_s_terms *= 1 - d/n
            self._ss.append(prod_s_terms)
            if d == n == 1:
                sum_var_terms = float('inf')
            else:
                sum_var_terms += d / (n*(n-d))
            if sum_var_terms > 0:
                self._vs.append(1 / numpy.log(prod_s_terms)**2 * sum_var_terms)
            else:
                self._vs.append(0)
            n -= 1

        # Just prevent overflow warning when computing the confidence interval
        eps = 1e-9
        self._ss_clipped = numpy.clip(self._ss, eps, 1.0-eps)

    def predict(self, t):
        '''Returns the predicted values.'''
        t = numpy.array(t)
        res = numpy.zeros(t.shape)
        for indexes, value in numpy.ndenumerate(t):
            j = numpy.searchsorted(self._ts, value, side='right') - 1
            if j >= len(self._ts) - 1:
                # Make the plotting stop at the last value of t
                res[indexes] = float('nan')
            else:
                res[indexes] = 1 - self._ss[j]
        return res

    def predict_ci(self, t, ci=0.8):
        '''Returns the predicted values with a confidence interval.'''
        t = numpy.array(t)
        res = numpy.zeros(t.shape + (3,))
        for indexes, value in numpy.ndenumerate(t):
            j = numpy.searchsorted(self._ts, value, side='right') - 1
            if j >= len(self._ts) - 1:
                # Make the plotting stop at the last value of t
                res[indexes] = [float('nan')]*3
            else:
                z_lo, z_hi = scipy.stats.norm.ppf([(1-ci)/2, (1+ci)/2])
                res[indexes] = (
                    1 - self._ss[j],
                    1 - numpy.exp(-numpy.exp(
                            numpy.log(-numpy.log(self._ss_clipped[j]))
                            + z_hi * self._vs[j]**0.5)),
                    1 - numpy.exp(-numpy.exp(
                            numpy.log(-numpy.log(self._ss_clipped[j]))
                            + z_lo * self._vs[j]**0.5))
                    )
        return res

    @deprecated(version='0.2.0',
                reason='Use :meth:`predict` or :meth:`predict_ci` instead.')
    def cdf(self, t, ci=None):
        '''Returns the predicted values.'''
        if ci is not None:
            return self.predict_ci(t)
        else:
            return self.predict(t)
