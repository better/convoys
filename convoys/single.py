import bisect
import lifelines
import numpy
from scipy.special import expit
import tensorflow as tf
from convoys import tf_utils


class SingleModel:
    pass  # TODO


class KaplanMeier(SingleModel):
    def fit(self, B, T):
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(T, event_observed=B)
        self.ts = kmf.survival_function_.index.values
        self.ps = 1.0 - kmf.survival_function_['KM_estimate'].values
        self.ps_hi = 1.0 - kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        self.ps_lo = 1.0 - kmf.confidence_interval_['KM_estimate_upper_0.95'].values

    def predict(self, ts, ci=None):
        # TODO: should also handle scalars
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[min(j, len(self.ts)-1)] for j in js])
        if ci is not None:
            return (array_lookup(self.ps), array_lookup(self.ps_lo), array_lookup(self.ps_hi))
        else:
            return array_lookup(self.ps)

    def predict_final(self, ci=None):
        if ci is not None:
            return (self.ps[-1], self.ps_lo[-1], self.ps_hi[-1])
        else:
            return self.ps[-1]

    def predict_time(self, ci=None):
        # TODO: should not use median here, but mean is no good
        def median(ps):
            i = bisect.bisect_left(ps, 0.5)
            return self.ts[min(i, len(ps)-1)]
        if ci is not None:
            return median(self.ps), median(self.ps_lo), median(self.ps_hi)
        else:
            return median(self.ps)


def _fit_beta_distribution_from_sigmoid(c, h):
    # TODO: should write down the derivation of this somewhere
    M = numpy.array([[-1/c**2, -1/(1-c)**2],
                     [(1-c), -c]])
    q = numpy.array([-h/(c*(1-c))**2 - 1/c**2 - 1/(1-c)**2, 1 - 2*c])
    return numpy.linalg.solve(M, q)


class Nonparametric(SingleModel):
    def fit(self, B, T, n=100):
        # We're going to fit c and p_0, p_1, ...
        # so that the probability of conversion at time i is c * (1 - p_0) * ... p_i
        # What's the total likelihood
        # For items that did convert:
        # L = c * (1 - p_0) * ... * (1 - p_{i-1}) * p_i
        # For items that did not convert:
        # L = 1 - c + c * (1 - p_0) * ... * (1 - p_{i})
        # Need to sum up the log of that
        # We also replace the p_i's with sigmoids just to make the problem unconstrained (and note that 1-s(z) = s(-z))
        all_ts = list(sorted(t for b, t in zip(B, T) if b))
        n = min(n, len(all_ts))
        js = [int(round(1.0 * len(all_ts) * (z + 1) / n - 1)) for z in range(n)]
        self.ts = [all_ts[j] for j in js]
        self.get_j = lambda t: min(bisect.bisect_left(self.ts, t), n-1)  # TODO: numpy.searchsorted?
        count_observed = numpy.zeros((n,), dtype=numpy.float32)
        count_unobserved = numpy.zeros((n,), dtype=numpy.float32)
        for i, (b, t) in enumerate(zip(B, T)):
            j = self.get_j(t)
            if b:
                count_observed[j] += 1
            else:
                count_unobserved[j] += 1

        z = tf.Variable(tf.zeros((n,)))
        log_survived_until = tf.cumsum(tf.log(tf.sigmoid(-z)), exclusive=True)
        log_survived_after = tf.cumsum(tf.log(tf.sigmoid(-z)))
        log_observed = tf.log(tf.sigmoid(z))

        beta = tf.Variable(tf.zeros([]))
        c = tf.sigmoid(beta)

        B = numpy.array(B, dtype=numpy.float32)
        T = numpy.array(T, dtype=numpy.float32)

        LL_observed = tf.log(c) + log_survived_until + log_observed
        LL_unobserved = tf.log(1 - c + c * tf.exp(log_survived_after))
        LL = tf.reduce_sum(count_observed * LL_observed + count_unobserved * LL_unobserved, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL, (z, beta))
            # Note: we only store the diagonal of the Hessian, since empirically, off-diagonal
            # elements are almost zero, and working with the full covariance matrix causes
            # numpy.random.multivariate_normal to break.
            self.params = {
                'beta': sess.run(beta),
                'beta_std': tf_utils.get_hessian(sess, LL, beta) ** -0.5,
                'a': numpy.zeros((n,)),
                'b': numpy.zeros((n,)),
            }
            c = sess.run(tf.sigmoid(z))
            # TODO: seems inefficient to compute the full hessian and then discard everything but the diagonal
            h = numpy.diag(tf_utils.get_hessian(sess, LL, z))
            for i in range(len(c)):
                self.params['a'][i], self.params['b'][i] = _fit_beta_distribution_from_sigmoid(c[i], h[i])


    def predict(self, t, ci=None, n=1000):
        t = numpy.array(t)
        if ci:
            betas = numpy.random.normal(self.params['beta'], self.params['beta_std'], n)
            cs = numpy.random.beta(self.params['a'], self.params['b'], size=(n,) + self.params['a'].shape).T
        else:
            betas = self.params['beta']
            cs = (self.params['a'] - 1) / (self.params['a'] + self.params['b'] - 2)

        c = expit(betas)
        survived_until = numpy.cumprod(1 - cs, axis=0)
        f = c * (1 - survived_until)
        m = tf_utils.predict(f, ci)
        res = numpy.zeros(t.shape + (3,) if ci else t.shape)
        for indexes, value in numpy.ndenumerate(t):
            j = self.get_j(value)
            res[indexes] = m[j]
        return res

    def predict_final(self, ci=None, n=1000):
        if ci:
            betas = numpy.random.normal(self.params['beta'], self.params['beta_std'], n)
            return tf_utils.predict(expit(betas), ci)
        else:
            return expit(self.params['beta'])
