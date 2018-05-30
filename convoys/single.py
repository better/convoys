import numpy
from scipy.special import expit, logit
import scipy.stats

__all__ = ['KaplanMeier']


class SingleModel:
    pass  # TODO


class KaplanMeier(SingleModel):
    def fit(self, B, T):
        # See https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
        n = len(T)
        self._ts = []
        self._ss = []
        self._vs = []
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
        self.get_j = lambda t: numpy.searchsorted(self._ts, t)

    def _get_value_at(self, j, ci):
        if ci:
            z_lo, z_hi = scipy.stats.norm.ppf([(1-ci)/2, (1+ci)/2])
            return (
                1 - self._ss[j],
                1 - numpy.exp(-numpy.exp(numpy.log(-numpy.log(self._ss[j])) + z_hi * self._vs[j]**0.5)),
                1 - numpy.exp(-numpy.exp(numpy.log(-numpy.log(self._ss[j])) + z_lo * self._vs[j]**0.5))
            )
        else:
            return 1 - self._ss[j]

    def cdf(self, t, ci=None):
        t = numpy.array(t)
        res = numpy.zeros(t.shape + (3,) if ci else t.shape)
        for indexes, value in numpy.ndenumerate(t):
            j = self.get_j(value)
            if j == len(self._ts):
                # Make the plotting stop at the last value of t
                res[indexes] = [float('nan')]*3 if ci else float('nan')
            else:
                res[indexes] = self._get_value_at(j, ci)
        return res
