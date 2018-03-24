import numpy
import scipy.stats
import sys
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

tf.logging.set_verbosity(3)


def get_hessian(sess, f, param):
    return sess.run(tf.hessians(-f, [param]))[0]


def optimize(sess, target, variables, method='L-BFGS-B'):
    optimizer = ScipyOptimizerInterface(-target, method=method, options={'disp': True})
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess)


def sample_hessian(x, value, hessian, n, ci):
    if ci is None:
        return numpy.dot(x, value)
    else:
        x = numpy.array(x)
        # TODO: if x is a zero vector, this triggers some weird warning
        inv_var = numpy.dot(numpy.dot(x.T, hessian), x)
        return numpy.dot(x, value) + scipy.stats.norm.rvs(scale=inv_var**-0.5, size=(n,))


def predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        # Replace the last axis with a 3-element vector
        y = numpy.mean(func_values, axis=-1)
        y_lo = numpy.percentile(func_values, (1-ci)*50, axis=-1)
        y_hi = numpy.percentile(func_values, (1+ci)*50, axis=-1)
        return numpy.stack((y, y_lo, y_hi), axis=-1)
