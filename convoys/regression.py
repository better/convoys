import numpy # TODO: remove
from scipy.special import expit, gamma, gammainc  # TODO: remove
import scipy.stats
import tensorflow as tf
from convoys import tf_utils


tf.logging.set_verbosity(3)

def _get_constants(args):
    return (tf.constant(arg.astype(numpy.float32)) for arg in args)


def _get_params(sess, params):
    return {key: sess.run(param) for key, param in params.items()}


def _get_hessian(sess, f, param):
    return sess.run(tf.hessians(-f, [param]))[0]


def _fix_t(t):
    # TODO: this is stupid, should at least have tests for it
    t = numpy.array(t)
    if len(t.shape) == 0:
        return t
    elif len(t.shape) == 1:
        return numpy.array([[z] for z in t])
    else:
        return t


def _sample_hessian(x, value, hessian, n, ci):
    if ci is None:
        return numpy.dot(x, value)
    else:
        x = numpy.array(x)
        inv_var = numpy.dot(numpy.dot(x.T, hessian), x)
        return numpy.dot(x, value) + scipy.stats.norm.rvs(scale=inv_var**-0.5, size=(1, n))


def _predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        axis = len(func_values.shape)-1
        return numpy.mean(func_values, axis=axis), numpy.percentile(func_values, (1-ci)*50, axis=axis), numpy.percentile(func_values, (1+ci)*50, axis=axis)


class RegressionModel:
    def __init__(self, L2_reg=1.0):
        self._L2_reg = L2_reg


class Exponential(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_constants((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        log_pdf = tf.log(lambd) - T_input*lambd
        cdf = 1 - tf.exp(-(T_input * lambd))

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta))
            self.params = _get_params(sess, {'beta': beta, 'alpha': alpha})
            self.params['alpha_hessian'] = _get_hessian(sess, LL_penalized, alpha)
            self.params['beta_hessian'] = _get_hessian(sess, LL_penalized, beta)

    def predict(self, x, t, ci=None, n=1000):
        t = _fix_t(t)
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta) * (1 - numpy.exp(-t * numpy.exp(x_prod_alpha))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return _predict(1./numpy.exp(x_prod_alpha), ci)


class Weibull(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_constants((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        log_pdf = tf.log(k) + tf.log(lambd) + (k-1)*(tf.log(T_input) + tf.log(lambd)) - (T_input*lambd)**k
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        cdf = 1 - tf.exp(-(T_input * lambd)**k)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta, log_k_var))
            self.params = _get_params(sess, {'beta': beta, 'alpha': alpha, 'k': k})
            self.params['alpha_hessian'] = _get_hessian(sess, LL_penalized, alpha)
            self.params['beta_hessian'] = _get_hessian(sess, LL_penalized, beta)

    def predict(self, x, t, ci=None, n=1000):
        t = _fix_t(t)
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta) * (1 - numpy.exp(-(t * numpy.exp(x_prod_alpha))**self.params['k'])), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return _predict(1./numpy.exp(x_prod_alpha) * gamma(1 + 1./self.params['k']), ci)


class Gamma(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_input, B_input, T_input = _get_constants((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X_input, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
        log_pdf = -tf.lgamma(k) + k*tf.log(lambd) + (k-1)*tf.log(T_input) - lambd*T_input
        # CDF of gamma: gammainc(k, lambda * t)
        cdf = tf.igamma(k, lambd * T_input)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta, log_k_var))
            self.params = _get_params(sess, {'beta': beta, 'alpha': alpha, 'k': k})
            self.params['alpha_hessian'] = _get_hessian(sess, LL_penalized, alpha)
            self.params['beta_hessian'] = _get_hessian(sess, LL_penalized, beta)

    def predict(self, x, t, ci=None, n=1000):
        t = _fix_t(t)
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta) * (1 - gammainc(self.params['k'], t * numpy.exp(x_prod_alpha))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = _sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return _predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = _sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return _predict(self.params['k']/numpy.exp(x_prod_alpha), ci)
