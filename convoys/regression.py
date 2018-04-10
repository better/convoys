import numpy # TODO: remove
from scipy.special import expit, gamma, gammainc  # TODO: remove
import scipy.stats
import tensorflow as tf
from convoys import tf_utils


class LinearCombination:
    def __init__(self, X, k):
        self.beta = tf.Variable(tf.zeros([k]))
        self.b = tf.Variable(tf.zeros([]))
        self.y = tf.squeeze(tf.matmul(X, tf.expand_dims(self.beta, -1)), 1) + self.b

    def params(self, sess, LL, feed_dict):
        return sess.run([
            self.beta,
            self.b,
            tf.hessians(-LL, [self.beta])[0],
            tf.hessians(-LL, [self.b])[0]
        ], feed_dict=feed_dict)

    @staticmethod
    def sample(params, x, ci, n):
        beta, b, beta_hessian, b_hessian = params
        mean = numpy.dot(x, beta) + b
        if ci is None:
            return mean
        else:
            x = numpy.array(x)
            # TODO: if x is a zero vector, this triggers some weird warning
            # TODO: we shouldn't assume that beta and b are independent
            inv_var_beta = numpy.dot(numpy.dot(x.T, beta_hessian), x)
            inv_var_b = b_hessian**2
            return mean + scipy.stats.norm.rvs(scale=(1/inv_var_beta + 1/inv_var_b)**0.5, size=(n,))


class RegressionModel(object):
    pass


class GeneralizedGamma(RegressionModel):
    # https://en.wikipedia.org/wiki/Generalized_gamma_distribution
    # Note however that lambda is a^-1 in WP's notation
    # Note also that k = d/p so d = k*p
    def fit(self, X, B, T, k=None, p=None):
        n_features = X.shape[1]
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        a = LinearCombination(X_batch, n_features)
        b = LinearCombination(X_batch, n_features)
        lambd = tf.exp(a.y)
        c = tf.sigmoid(b.y)

        if k is None:
            k = tf.Variable(1.0, name='k', trainable=False)
            should_update_k = True
        else:
            k = tf.constant(k, tf.float32)
            should_update_k = False

        if p is None:
            log_p_var = tf.Variable(tf.zeros([]), name='log_p')
            p = tf.exp(log_p_var)
        else:
            p = tf.constant(p, tf.float32)

        # PDF: p*lambda^(k*p) / gamma(k) * t^(k*p-1) * exp(-(x*lambda)^p)
        log_pdf = \
            tf.log(p) + (k*p) * tf.log(lambd) \
            - tf.lgamma(k) + (k*p-1) * tf.log(T_batch) \
            - (T_batch*lambd)**p
        cdf = tf.igamma(k, (T_batch*lambd)**p)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_batch * LL_observed + (1 - B_batch) * LL_censored, 0)

        with tf.Session() as sess:
            feed_dict = {X_batch: X, B_batch: B, T_batch: T}
            tf_utils.optimize(
                sess, LL, feed_dict,
                update_callback=(tf_utils.get_tweaker(sess, LL, k, feed_dict)
                                 if should_update_k else None))
            self.params = {
                'a': a.params(sess, LL, feed_dict),
                'b': b.params(sess, LL, feed_dict),
                'k': sess.run(k),
                'p': sess.run(p),
            }

    def cdf(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        a = LinearCombination.sample(self.params['a'], x, ci, n)
        b = LinearCombination.sample(self.params['b'], x, ci, n)
        return tf_utils.predict(
            expit(b) * gammainc(
                self.params['k'],
                numpy.multiply.outer(t, numpy.exp(a))**self.params['p']),
            ci)


class Exponential(GeneralizedGamma):
    def fit(self, X, B, T):
        super(Exponential, self).fit(X, B, T, k=1, p=1)


class Weibull(GeneralizedGamma):
    def fit(self, X, B, T):
        super(Weibull, self).fit(X, B, T, k=1)


class Gamma(GeneralizedGamma):
    def fit(self, X, B, T):
        super(Gamma, self).fit(X, B, T, p=1)
