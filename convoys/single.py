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
    def fit(self, B, T, n=3):
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
        print('len(all_ts):', len(all_ts))
        n = min(n, len(all_ts))
        js = [int(round(1.0 * len(all_ts) * (z + 1) / n - 1)) for z in range(n)]
        print('js:', js)
        ts = [all_ts[j] for j in js]
        print('len(ts):', len(ts))
        count_observed = numpy.zeros((n,), dtype=numpy.float32)
        count_unobserved = numpy.zeros((n,), dtype=numpy.float32)
        for i, (b, t) in enumerate(zip(B, T)):
            j = bisect.bisect_left(ts, t)
            j = min(j, n-1)
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
        B = B.astype(numpy.float32)

        LL_observed = tf.log(c) + log_survived_until + log_observed
        LL_unobserved = tf.log(1 - c + c * tf.exp(log_survived_after))
        LL = tf.reduce_sum(count_observed * LL_observed + count_unobserved * LL_unobserved, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL, (z, beta))

            print('sum of probabilities of conversion:', sess.run(tf.exp(log_survived_until + log_observed)))

            print('survival rates', sess.run(tf.log(tf.sigmoid(-z))))
            print('log survived', sess.run(log_survived_until))
            print('log survived + endpoint', sess.run(log_survived_after))
            print('LL_observed', sess.run(LL_observed))
            print('LL_unobserved', sess.run(LL_unobserved))
            print('c:', sess.run(c))
            w = sess.run(c * (1 - tf.exp(log_survived_until)))
            from matplotlib import pyplot
            pyplot.plot([0] + ts, w)
            pyplot.savefig('nonparametric.png')
