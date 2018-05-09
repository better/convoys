import numpy
from scipy.special import expit, gammainc, gammaincinv
import scipy.stats
import tensorflow as tf
from convoys import tf_utils


class LinearCombination:
    def __init__(self, X, k):
        self.beta = tf.Variable(tf.zeros([k]))
        self.b = tf.Variable(tf.zeros([]))
        self.y = tf.squeeze(tf.matmul(X, tf.expand_dims(self.beta, -1)), 1) + self.b
        self.log_sigma = tf.Variable(tf.zeros([]))
        self.sigma = tf.exp(self.log_sigma)
        # log PDF of normal distribution
        self.LL_term = \
            -tf.reduce_sum(self.beta**2) / (2*self.sigma**2) + \
            -k**self.log_sigma

    def params(self, sess, LL, feed_dict):
        return sess.run([
            self.beta,
            self.b,
            tf.hessians(-LL, [self.beta])[0],
            tf.hessians(-LL, [self.b])[0],
            self.sigma,
        ], feed_dict=feed_dict)

    @staticmethod
    def sample(params, x, ci, n):
        beta, b, beta_hessian, b_hessian, sigma = params
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
    def fit(self, X, B, T, W=None, k=None, p=None):
        n_features = X.shape[1]
        if W is None:
            W = numpy.ones(B.shape)
        X_batch, B_batch, T_batch, W_batch = tf_utils.get_batch_placeholders((X, B, T, W))

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

        LL_batch = tf.reduce_sum(
            W_batch * B_batch * LL_observed +
            W_batch * (1 - B_batch) * LL_censored, 0)
        LL_global = a.LL_term + b.LL_term

        with tf.Session() as sess:
            feed_dict = {X_batch: X, B_batch: B, T_batch: T, W_batch: W}
            tf_utils.optimize(
                sess, LL_batch, LL_global, feed_dict,
                update_callback=(
                    tf_utils.get_tweaker(sess, LL_batch + LL_global, k, feed_dict)
                    if should_update_k else None))
            self.params = {
                'a': a.params(sess, LL_batch + LL_global, feed_dict),
                'b': b.params(sess, LL_batch + LL_global, feed_dict),
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

    def rvs(self, x, n_curves=1, n_samples=1, T=None):
        # Samples values from this distribution
        # T is optional and means we already observed non-conversion until T
        if T is None:
            T = numpy.zeros((n_curves, n_samples))
        else:
            assert T.shape == (n_curves, n_samples)
        a = LinearCombination.sample(self.params['a'], x, 1, n_curves)
        b = LinearCombination.sample(self.params['b'], x, 1, n_curves)
        B = numpy.zeros((n_curves, n_samples), dtype=numpy.bool)
        C = numpy.zeros((n_curves, n_samples))
        for i, (a, b) in enumerate(zip(a, b)):
            z = numpy.random.uniform(size=(n_samples,))
            cdf_now = expit(b) * gammainc(
                self.params['k'],
                numpy.multiply.outer(T[i], numpy.exp(a))**self.params['p'])
            cdf_final = expit(b)
            adjusted_z = cdf_now + (1 - cdf_now) * z
            B[i] = (adjusted_z < cdf_final)
            y = adjusted_z / cdf_final
            x = gammaincinv(self.params['k'], y)
            # x = (t * exp(a))**p
            C[i] = x**(1./self.params['p']) / numpy.exp(a)
            C[i][~B[i]] = 0

        return B, C


class Exponential(GeneralizedGamma):
    def fit(self, X, B, T, W):
        super(Exponential, self).fit(X, B, T, W, k=1, p=1)


class Weibull(GeneralizedGamma):
    def fit(self, X, B, T, W):
        super(Weibull, self).fit(X, B, T, W, k=1)


class Gamma(GeneralizedGamma):
    def fit(self, X, B, T, W):
        super(Gamma, self).fit(X, B, T, W, p=1)
