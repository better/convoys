import bisect
import lifelines
import numpy
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


class Nonparametric(SingleModel):
    def fit(self, B, T, n=100):
        # We're going to fit c and p_0, p_1, ...
        # so that the probability of conversion at time i is c * (1 - p_0) * ... p_i
        # What's the total likelihood
        # For items that did convert:
        # L = c * (1 - p_0) * ... * (1 - p_{i-1}) * p_i
        # For items that did not convert:
        # L = 1 - c + c * (1 - p_0) * ... * (1 - p_{i-1})
        # Need to sum up the log of that
        # Let's replace the p_i's with sigmoids
        n = min(n, len(B))
        js = [int(round(z)) for z in numpy.linspace(0, len(B), n-1, endpoint=False)]
        ts = list(sorted(T))
        ts = [ts[j] for j in js]
        M = numpy.zeros((len(B), n), dtype=numpy.float32)
        N = numpy.zeros((len(B), n), dtype=numpy.float32)
        for i, (b, t) in enumerate(zip(B, T)):
            j = bisect.bisect_left(ts, t)
            M[i,0:j] = -1
            N[i,0:j] = 1
            if b:
                M[i,j] = 1
                N[i,j] = 1
        M = tf.constant(M)
        z = tf.Variable(tf.zeros((n,)))
        Z = tf.expand_dims(z, 0)
        l = tf.reduce_sum(tf.log(tf.sigmoid(M * Z)) * N, 1)
        beta = tf.Variable(tf.zeros([]))
        c = tf.sigmoid(beta)
        B = tf.constant(B.astype(numpy.float32))
        LL_observed = (tf.log(c) + l)
        LL_unobserved = (tf.log(1 - c + c * tf.exp(l)))
        LL = tf.reduce_sum(B * LL_observed + (1 - B) * LL_unobserved, 0)
        with tf.Session() as sess:
            tf_utils.optimize(sess, LL, (z, beta))

            print('c:', sess.run(c))
            w = sess.run(c * (1 - tf.exp(tf.cumsum(tf.log(tf.sigmoid(-z))))))
            from matplotlib import pyplot
            pyplot.plot(w)
            pyplot.savefig('nonparametric.png')

