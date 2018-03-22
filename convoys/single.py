import numpy
from scipy.special import expit, logit
import scipy.stats
import tensorflow as tf
from convoys import tf_utils


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
            sum_var_terms += d / (n*(n-d))
            self._vs.append(1 / numpy.log(prod_s_terms)**2 * sum_var_terms)
            n -= 1
        self.get_j = lambda t: min(numpy.searchsorted(self._ts, t), len(self._ts)-1)

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

    def predict(self, t, ci=None):
        res = numpy.zeros(t.shape + (3,) if ci else t.shape)
        for indexes, value in numpy.ndenumerate(t):
            j = self.get_j(value)
            res[indexes] = self._get_value_at(j, ci)
        return res

    def predict_final(self, ci=None):
        return self._get_value_at(len(self._ts)-1, ci)


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
        self.get_j = lambda t: min(numpy.searchsorted(self.ts, t), n-1)
        count_observed = numpy.zeros((n,), dtype=numpy.float32)
        count_unobserved = numpy.zeros((n,), dtype=numpy.float32)
        for i, (b, t) in enumerate(zip(B, T)):
            j = self.get_j(t)
            if b:
                count_observed[j] += 1
            else:
                count_unobserved[j] += 1

        B = numpy.array(B, dtype=numpy.float32)
        T = numpy.array(T, dtype=numpy.float32)

        z = tf.Variable(tf.zeros((n,)))
        log_survived_until = tf.cumsum(tf.log(tf.sigmoid(-z)), exclusive=True)
        log_survived_after = tf.cumsum(tf.log(tf.sigmoid(-z)))
        log_observed = tf.log(tf.sigmoid(z))

        beta = tf.Variable(tf.zeros([]))
        c = tf.sigmoid(beta)

        def get_LL(log_survived_until, log_survived_after, log_observed):
            LL_observed = tf.log(c) + log_survived_until + log_observed
            LL_unobserved = tf.log(1 - c + c * tf.exp(log_survived_after))
            return tf.reduce_sum(count_observed * LL_observed + count_unobserved * LL_unobserved, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, get_LL(log_survived_until, log_survived_after, log_observed), (z, beta))

            # At this point, we're going to reparametrize the problem to be a function of log_survived_after
            # We can't do that with the original problem, since that would induce negative values of log_observed
            log_cs = sess.run(log_survived_after)
            z = tf.Variable(logit(numpy.exp(log_cs)))
            log_survived_after = tf.log(tf.sigmoid(z))
            log_survived_until = tf.concat([[0], tf.slice(log_survived_after, [0], [len(log_cs)-1])], axis=0)
            log_observed = tf.log(1 - tf.exp(log_survived_after - log_survived_until))
            LL = get_LL(log_survived_until, log_survived_after, log_observed)
            sess.run(z.initializer)

            # Note: we only store the diagonal of the Hessian, since empirically, off-diagonal
            # elements are almost zero.
            self.params = {
                'beta': sess.run(beta),
                'beta_std': tf_utils.get_hessian(sess, LL, beta) ** -0.5,
                'z': sess.run(z),
                'z_std': numpy.diag(tf_utils.get_hessian(sess, LL, z)) ** -0.5,  # TODO: seems inefficient
            }
            self.params['z_std'] = 0  # not sure what's up, will revisit this

    def predict(self, t, ci=None, n=1000):
        t = numpy.array(t)
        if ci:
            betas = numpy.random.normal(self.params['beta'], self.params['beta_std'], n)
            zs = numpy.random.normal(self.params['z'], self.params['z_std'], size=(n,) + self.params['z'].shape).T
            zs = numpy.clip(zs, -10, 10)  # Fix crazy outliers
        else:
            betas = self.params['beta']
            zs = self.params['z']

        c = expit(betas)
        survived_until = expit(zs)
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
