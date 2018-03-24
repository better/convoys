import numpy # TODO: remove
from scipy.special import expit, gamma, gammainc  # TODO: remove
import tensorflow as tf
from convoys import tf_utils


class RegressionModel:
    def __init__(self, L2_reg=1.0):
        self._L2_reg = L2_reg


class Exponential(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X, B, T = (z.astype(numpy.float32) for z in (X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        X_prod_alpha = tf.squeeze(tf.matmul(X, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X, tf.expand_dims(beta, -1)), 1)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        log_pdf = tf.log(lambd) - T*lambd
        cdf = 1 - tf.exp(-(T * lambd))

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B * LL_observed + (1 - B) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta))
            self.params = {
                'beta': sess.run(beta),
                'alpha': sess.run(alpha),
                'alpha_hessian': tf_utils.get_hessian(sess, LL_penalized, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL_penalized, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta) * (1 - numpy.exp(numpy.multiply.outer(-t, numpy.exp(x_prod_alpha)))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return tf_utils.predict(1./numpy.exp(x_prod_alpha), ci)


class Weibull(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X, B, T = (z.astype(numpy.float32) for z in (X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        log_pdf = tf.log(k) + tf.log(lambd) + (k-1)*(tf.log(T) + tf.log(lambd)) - (T*lambd)**k
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        cdf = 1 - tf.exp(-(T * lambd)**k)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B * LL_observed + (1 - B) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta, log_k_var))
            self.params = {
                'beta': sess.run(beta),
                'alpha': sess.run(alpha),
                'k': sess.run(k),
                'alpha_hessian': tf_utils.get_hessian(sess, LL_penalized, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL_penalized, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta) * (1 - numpy.exp(-numpy.multiply.outer(t, numpy.exp(x_prod_alpha))**self.params['k'])), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return tf_utils.predict(1./numpy.exp(x_prod_alpha) * gamma(1 + 1./self.params['k']), ci)


class Gamma(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X, B, T = (z.astype(numpy.float32) for z in (X, B, T))

        alpha = tf.Variable(tf.zeros([k]), 'alpha')
        beta = tf.Variable(tf.zeros([k]), 'beta')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')
        X_prod_alpha = tf.squeeze(tf.matmul(X, tf.expand_dims(alpha, -1)), 1)
        X_prod_beta = tf.squeeze(tf.matmul(X, tf.expand_dims(beta, -1)), 1)
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
        log_pdf = -tf.lgamma(k) + k*tf.log(lambd) + (k-1)*tf.log(T) - lambd*T
        # CDF of gamma: gammainc(k, lambda * t)
        cdf = tf.igamma(k, lambd * T)

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B * LL_observed + (1 - B) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL_penalized, (alpha, beta, log_k_var), method='Powell')
            self.params = {
                'beta': sess.run(beta),
                'alpha': sess.run(alpha),
                'k': sess.run(k),
                'alpha_hessian': tf_utils.get_hessian(sess, LL_penalized, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL_penalized, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta) * gammainc(self.params['k'], numpy.multiply.outer(t, numpy.exp(x_prod_alpha))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci)
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci)
        return tf_utils.predict(self.params['k']/numpy.exp(x_prod_alpha), ci)
