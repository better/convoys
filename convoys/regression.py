import numpy # TODO: remove
from scipy.special import expit, gamma, gammainc  # TODO: remove
import tensorflow as tf
from convoys import tf_utils


class RegressionModel:
    pass


class Exponential(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), name='alpha')
        beta = tf.Variable(tf.zeros([k]), name='beta')
        a = tf.Variable(tf.zeros([]), name='a')
        b = tf.Variable(tf.zeros([]), name='b')
        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        X_prod_alpha = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(alpha, -1)), 1) + a
        X_prod_beta = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(beta, -1)), 1) + b
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

        log_pdf = tf.log(lambd) - T_batch*lambd
        cdf = 1 - tf.exp(-(T_batch * lambd))

        LL_observed = tf.log(c) + log_pdf
        LL_censored = tf.log((1-c) + c * (1 - cdf))

        LL = tf.reduce_sum(B_batch * LL_observed + (1 - B_batch) * LL_censored, 0)

        with tf.Session() as sess:
            feed_dict = {X_batch: X, B_batch: B, T_batch: T}
            tf_utils.optimize(sess, LL, feed_dict)
            self.params = {
                'alpha': sess.run(alpha),
                'beta': sess.run(beta),
                'a': sess.run(a),  # TODO: store hessian
                'b': sess.run(b),  # TODO: store hessian
                'alpha_hessian': tf_utils.get_hessian(sess, LL, feed_dict, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL, feed_dict, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['b']
        return tf_utils.predict(expit(x_prod_beta) * (1 - numpy.exp(numpy.multiply.outer(-t, numpy.exp(x_prod_alpha)))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['b']
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        return tf_utils.predict(1./numpy.exp(x_prod_alpha), ci)


class Weibull(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), name='alpha')
        beta = tf.Variable(tf.zeros([k]), name='beta')
        a = tf.Variable(tf.zeros([]), name='a')
        b = tf.Variable(tf.zeros([]), name='b')
        log_k_var = tf.Variable(tf.zeros([]), name='log_k')
        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        X_prod_alpha = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(alpha, -1)), 1) + a
        X_prod_beta = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(beta, -1)), 1) + b
        k = tf.exp(log_k_var)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

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
                'alpha': sess.run(alpha),
                'beta': sess.run(beta),
                'a': sess.run(a),  # TODO: store hessian
                'b': sess.run(b),  # TODO: store hessian
                'k': sess.run(k),
                'alpha_hessian': tf_utils.get_hessian(sess, LL, feed_dict, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL, feed_dict, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['b']
        return tf_utils.predict(expit(x_prod_beta) * (1 - numpy.exp(-numpy.multiply.outer(t, numpy.exp(x_prod_alpha))**self.params['k'])), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['b']
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        return tf_utils.predict(1./numpy.exp(x_prod_alpha) * gamma(1 + 1./self.params['k']), ci)


class Gamma(RegressionModel):
    def fit(self, X, B, T):
        n, k = X.shape
        X_batch, B_batch, T_batch = tf_utils.get_batch_placeholders((X, B, T))

        alpha = tf.Variable(tf.zeros([k]), name='alpha')
        beta = tf.Variable(tf.zeros([k]), name='beta')
        a = tf.Variable(tf.zeros([]), name='a')
        b = tf.Variable(tf.zeros([]), name='b')
        X_batch = tf.nn.dropout(X_batch, keep_prob=0.5)
        X_prod_alpha = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(alpha, -1)), 1) + a
        X_prod_beta = tf.squeeze(tf.matmul(X_batch, tf.expand_dims(beta, -1)), 1) + b
        k = tf.Variable(2.0, name='k', trainable=False)
        lambd = tf.exp(X_prod_alpha)
        c = tf.sigmoid(X_prod_beta)

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
            import time
            t0 = time.time()
            # tf.igamma doesn't compute the gradient wrt a properly
            # So let's just try small perturbations
            k_value = sess.run(k)
            res = {}
            for k_mult in [0.99, 1.0, 1.01]:
                sess.run(assign_k, feed_dict={new_k: k_value * k_mult})
                res[k_value * k_mult] = sess.run(LL, feed_dict=feed_dict)
            sess.run(assign_k, feed_dict={new_k: max(res.keys(), key=res.get)})

        with tf.Session() as sess:
            tf_utils.optimize(sess, LL, feed_dict, update_callback=update_k)
            self.params = {
                'alpha': sess.run(alpha),
                'beta': sess.run(beta),
                'a': sess.run(a),  # TODO: store hessian
                'b': sess.run(b),  # TODO: store hessian
                'k': sess.run(k),
                'alpha_hessian': tf_utils.get_hessian(sess, LL, feed_dict, alpha),
                'beta_hessian': tf_utils.get_hessian(sess, LL, feed_dict, beta),
            }

    def predict(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['b']
        return tf_utils.predict(expit(x_prod_beta) * gammainc(self.params['k'], numpy.multiply.outer(t, numpy.exp(x_prod_alpha))), ci)

    def predict_final(self, x, ci=None, n=1000):
        x_prod_beta = tf_utils.sample_hessian(x, self.params['beta'], self.params['beta_hessian'], n, ci) + self.params['a']
        return tf_utils.predict(expit(x_prod_beta), ci)

    def predict_time(self, x, ci=None, n=1000):
        x_prod_alpha = tf_utils.sample_hessian(x, self.params['alpha'], self.params['alpha_hessian'], n, ci) + self.params['a']
        return tf_utils.predict(self.params['k']/numpy.exp(x_prod_alpha), ci)
