import numpy # TODO: remove
from scipy.special import expit  # TODO: remove
import scipy.stats
import tensorflow as tf

from convoys import Model

class Regression(Model):
    # This will replace the model in __init__.py soon.
    def __init__(self, log_pdf, cdf, extra_params, L2_reg=1.0):
        self._L2_reg = L2_reg
        self._log_pdf = log_pdf
        self._cdf = cdf
        self._extra_params = extra_params
        self._sess = tf.Session()

    def __del__(self):
        self._sess.close()

    def fit(self, X, B, T):
        # TODO: should do this in constructor, but the shape of X isn't known at that point
        n, k = X.shape

        X_input = tf.placeholder(tf.float32, [None, k])
        B_input = tf.placeholder(tf.float32, [None])
        T_input = tf.placeholder(tf.float32, [None])
        beta = tf.Variable(tf.zeros([k]), 'beta')

        X_prod_beta = tf.squeeze(tf.matmul(X_input, tf.expand_dims(beta, -1)), 1)
        c = tf.sigmoid(X_prod_beta)  # Conversion rates for each example

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        LL_observed = tf.log(c) + self._log_pdf(T_input)
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        LL_censored = tf.log((1-c) + c * (1 - self._cdf(T_input)))

        LL = tf.reduce_sum(B_input * LL_observed + (1 - B_input) * LL_censored, 0)
        LL_penalized = LL - self._L2_reg * tf.reduce_sum(beta * beta, 0)

        step_var = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.03, step_var, 1, 0.999)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(-LL_penalized, global_step=step_var)

        # TODO(erikbern): this is going to add more and more variables every time we run this
        self._sess.run(tf.global_variables_initializer())

        best_cost, best_step, step = float('-inf'), 0, 0
        while True:
            feed_dict = {X_input: X, B_input: B, T_input: T}
            self._sess.run(optimizer, feed_dict=feed_dict)
            cost = self._sess.run(LL_penalized, feed_dict=feed_dict)
            if cost > best_cost:
                best_cost, best_step = cost, step
            if step - best_step > 100:
                break
            step += 1
            if step % 100 == 0:
                print(step, cost, self._sess.run(learning_rate))

        self.params = dict(
            beta=self._sess.run(beta),
            beta_hessian=self._sess.run(
                -tf.hessians(LL_penalized, [beta])[0],
                feed_dict=feed_dict,
            ),
            **self._extra_params(self._sess)
        )

    def predict(self):
        pass  # TODO: implement

    def predict_final(self, x, ci=None):
        # TODO: should take advantage of tensorflow here!!!
        x = numpy.array(x)
        def f(x, d=0):
            return expit(numpy.dot(x, self.params['beta']) + d)
        if ci:
            inv_var = numpy.dot(numpy.dot(x.T, self.params['beta_hessian']), x)
            lo, hi = (scipy.stats.norm.ppf(p, scale=inv_var**-0.5) for p in ((1 - ci)/2, (1 + ci)/2))
            return f(x), f(x, lo), f(x, hi)
        else:
            return f(x)

    def predict_time(self):
        pass  # TODO: implement


class ExponentialRegression(Regression):
    def __init__(self, L2_reg=1.0):
        log_lambd_var = tf.Variable(tf.zeros([]), 'log_lambd')
        lambd = tf.exp(log_lambd_var)

        log_pdf = lambda T: -T*lambd
        cdf = lambda T: 1 - tf.exp(-(T * lambd))

        return super(ExponentialRegression, self).__init__(
            log_pdf=log_pdf,
            cdf=cdf,
            extra_params=lambda sess: dict(lambd=sess.run(lambd)),
            L2_reg=L2_reg)


class WeibullRegression(Regression):
    def __init__(self, L2_reg=1.0):
        log_lambd_var = tf.Variable(tf.zeros([]), 'log_lambd')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')

        lambd = tf.exp(log_lambd_var)
        k = tf.exp(log_k_var)

        # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
        log_pdf = lambda T: tf.log(k) + tf.log(lambd) + (k-1)*(tf.log(T) + tf.log(lambd)) - (T*lambd)**k
        # CDF of Weibull: 1 - exp(-(t * lambda)^k)
        cdf = lambda T: 1 - tf.exp(-(T * lambd)**k)

        return super(WeibullRegression, self).__init__(
            log_pdf=log_pdf,
            cdf=cdf,
            extra_params=lambda sess: dict(k=sess.run(k),
                                           lambd=sess.run(lambd)),
            L2_reg=L2_reg)


class GammaRegression(Regression):
    def __init__(self, L2_reg=1.0):
        log_lambd_var = tf.Variable(tf.zeros([]), 'log_lambd')
        log_k_var = tf.Variable(tf.zeros([]), 'log_k')

        lambd = tf.exp(log_lambd_var)
        k = tf.exp(log_k_var)

        # PDF of gamma: 1.0 / gamma(k) * lambda ^ k * t^(k-1) * exp(-t * lambda)
        log_pdf = lambda T: -tf.lgamma(k) + k*tf.log(lambd) + (k-1)*tf.log(T) - lambd*T
        # CDF of gamma: gammainc(k, lambda * t)
        cdf = lambda T: tf.igamma(k, lambd * T)

        return super(GammaRegression, self).__init__(
            log_pdf=log_pdf,
            cdf=cdf,
            extra_params=lambda sess: dict(k=sess.run(k),
                                           lambd=sess.run(lambd)),
            L2_reg=L2_reg)
