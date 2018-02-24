import numpy # TODO: remove
from scipy.special import expit  # TODO: remove
import scipy.stats
import tensorflow as tf

from convoys.model import Model

def _get_placeholders(n, k):
    return (
        tf.placeholder(tf.float32, [n, k]),
        tf.placeholder(tf.float32, [n]),
        tf.placeholder(tf.float32, [n])
    )


def _optimize(sess, target, feed_dict):
    learning_rate_input = tf.placeholder(tf.float32, [])
    optimizer = tf.train.AdamOptimizer(learning_rate_input).minimize(-target)

    # TODO(erikbern): this is going to add more and more variables every time we run this
    sess.run(tf.global_variables_initializer())

    best_cost, best_step, step = float('-inf'), 0, 0
    learning_rate = 0.1
    while True:
        feed_dict[learning_rate_input] = learning_rate
        sess.run(optimizer, feed_dict=feed_dict)
        cost = sess.run(target, feed_dict=feed_dict)
        if cost > best_cost:
            best_cost, best_step = cost, step
        if step - best_step > 40:
            learning_rate /= 10
            best_cost = float('-inf')
        if learning_rate < 1e-6:
            break
        step += 1
        if step % 100 == 0:
            print('step %6d (lr %6.6f): %9.2f' % (step, learning_rate, cost))


def _get_params(sess, params):
    return {key: sess.run(param) for key, param in params.items()}


class Regression(Model):
    def __init__(self, L2_reg=1.0):
        self._L2_reg = L2_reg
        self._sess = tf.Session()

    def __del__(self):
        self._sess.close()

    def predict(self, x, t, ci=None):
        t = numpy.array(t)
        z = self._cdf(x, t)
        if ci:
            c, c_lo, c_hi = self.predict_final(x, ci)
            return (t, c*z, c_lo*z, c_hi*z)
        else:
            c = self.predict_final(x)
            return (t, c*z)

    def predict_final(self, x, ci=None):
        # TODO: should take advantage of tensorflow here!!!
        x = numpy.array(x)
        def f(x, d=0):
            return expit(numpy.dot(x, self.params['beta']) + d)
        if ci:
            # inv_var = numpy.dot(numpy.dot(x.T, self.params['beta_hessian']), x)
            # lo, hi = (scipy.stats.norm.ppf(p, scale=inv_var**-0.5) for p in ((1 - ci)/2, (1 + ci)/2))
            lo, hi = 0, 0
            return f(x), f(x, lo), f(x, hi)
        else:
            return f(x)

    def predict_time(self):
        pass  # TODO: implement


class ExponentialRegression(Regression):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_placeholders(n, k)

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        log_pdf = lambda T: tf.log(lambd) - T*lambd
        cdf = lambda T: 1 - tf.exp(-(T * lambd))

        LL_observed = tf.log(c) + log_pdf(T_input)
        LL_censored = tf.log((1-c) + c * (1 - cdf(T_input)))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        _optimize(self._sess, LL_penalized, {X_input: X, B_input: B, T_input: T})
        self.params = _get_params(self._sess, {'beta': beta, 'alpha': alpha})
        self._cdf = lambda x, T: 1 - numpy.exp(-T * numpy.exp(numpy.dot(x, self.params['alpha'])))


class WeibullRegression(Regression):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_placeholders(n, k)

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        log_pdf = lambda T: tf.log(k) + tf.log(lambd) + (k-1)*(tf.log(T) + tf.log(lambd)) - (T*lambd)**k
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        cdf = lambda T: 1 - tf.exp(-(T * lambd)**k)

        LL_observed = tf.log(c) + log_pdf(T_input)
        LL_censored = tf.log((1-c) + c * (1 - cdf(T_input)))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        _optimize(self._sess, LL_penalized, {X_input: X, B_input: B, T_input: T})
        self.params = _get_params(self._sess, {'beta': beta, 'alpha': alpha, 'k': k})
        self._cdf = lambda x, T: 1 - numpy.exp(-(T * numpy.exp(numpy.dot(x, self.params['alpha'])))**self.params['k'])
        print(self.params)


class GammaRegression(Regression):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_placeholders(n, k)

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
        log_pdf = lambda T: -tf.lgamma(k) + k*tf.log(lambd) + (k-1)*tf.log(T) - lambd*T
        # CDF of gamma: gammainc(k, lambda * t)
        cdf = lambda T: tf.igamma(k, lambd * T)

        LL_observed = tf.log(c) + log_pdf(T_input)
        LL_censored = tf.log((1-c) + c * (1 - cdf(T_input)))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        _optimize(self._sess, LL_penalized, {X_input: X, B_input: B, T_input: T})
        self.params = _get_params(self._sess, {'beta': beta, 'alpha': alpha, 'k': k})
        self._cdf = lambda x, T: 1 - numpy.igamma(self.params['k'], T * numpy.exp(numpy.dot(x, self.params['alpha'])))
