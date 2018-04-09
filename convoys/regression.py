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


class RegressionModel:
    pass


class Exponential(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        a = LinearCombination(X_batch, k)
        b = LinearCombination(X_batch, k)
        lambd = tf.exp(a.y)
        c = tf.sigmoid(b.y)

        log_pdf = tf.log(lambd) - T_batch*lambd
        cdf = 1 - tf.exp(-(T_batch * lambd))

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_batch * LL_observed + (1 - B_batch) * LL_censored, 0)

        with tf.Session() as sess:
            feed_dict = {X_batch: X, B_batch: B, T_batch: T}
            tf_utils.optimize(sess, LL, feed_dict)
            self.params = {
                'a': a.params(sess, LL, feed_dict),
                'b': b.params(sess, LL, feed_dict),
            }

    def cdf(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        a = LinearCombination.sample(self.params['a'], x, ci, n)
        b = LinearCombination.sample(self.params['b'], x, ci, n)
        return tf_utils.predict(expit(b) * (1 - numpy.exp(numpy.multiply.outer(-t, numpy.exp(a)))), ci)


class Weibull(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        log_k_var = tf.Variable(tf.zeros([]), name='log_k')
        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        a = LinearCombination(X_batch, k)
        b = LinearCombination(X_batch, k)
        k = tf.exp(log_k_var)
        lambd = tf.exp(a.y)
        c = tf.sigmoid(b.y)

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        log_pdf = tf.log(k) + tf.log(lambd) + (k-1)*(tf.log(T_batch) + tf.log(lambd)) - (T_batch*lambd)**k
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        cdf = 1 - tf.exp(-(T_batch * lambd)**k)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_batch * LL_observed + (1 - B_batch) * LL_censored, 0)

        with tf.Session() as sess:
            feed_dict = {X_batch: X, B_batch: B, T_batch: T}
            tf_utils.optimize(sess, LL, feed_dict)
            self.params = {
                'a': a.params(sess, LL, feed_dict),
                'b': b.params(sess, LL, feed_dict),
                'k': sess.run(k),
            }

    def cdf(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        a = LinearCombination.sample(self.params['a'], x, ci, n)
        b = LinearCombination.sample(self.params['b'], x, ci, n)
        return tf_utils.predict(expit(b) * (1 - numpy.exp(-numpy.multiply.outer(t, numpy.exp(a))**self.params['k'])), ci)


class Gamma(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        a = LinearCombination(X_batch, k)
        b = LinearCombination(X_batch, k)
        k = tf.Variable(2.0, name='k', trainable=False)
        lambd = tf.exp(a.y)
        c = tf.sigmoid(b.y)

        # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
        log_pdf = -tf.lgamma(k) + k*tf.log(lambd) + (k-1)*tf.log(T_batch) - lambd*T_batch
        # CDF of gamma: gammainc(k, lambda * t)
        cdf = tf.igamma(k, lambd * T_batch)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_batch * LL_observed + (1 - B_batch) * LL_censored, 0)
        feed_dict = {X_batch: X, B_batch: B, T_batch: T}

        new_k = tf.placeholder(tf.float32, shape=[])
        assign_k = tf.assign(k, new_k)

        def update_k(sess):
            # tf.igamma doesn't compute the gradient wrt a properly
            # So let's just try small perturbations
            k_value = sess.run(k)
            res = {}
            for k_mult in [0.97, 1.0, 1.03]:
                sess.run(assign_k, feed_dict={new_k: k_value * k_mult})
                res[k_value * k_mult] = sess.run(LL, feed_dict=feed_dict)
            sess.run(assign_k, feed_dict={new_k: max(res.keys(), key=res.get)})

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL, feed_dict, update_callback=update_k)
            self.params = {
                'a': a.params(sess, LL, feed_dict),
                'b': b.params(sess, LL, feed_dict),
                'k': sess.run(k),
            }

    def cdf(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        a = LinearCombination.sample(self.params['a'], x, ci, n)
        b = LinearCombination.sample(self.params['b'], x, ci, n)
        return tf_utils.predict(expit(b) * gammainc(self.params['k'], numpy.multiply.outer(t, numpy.exp(a))), ci)
