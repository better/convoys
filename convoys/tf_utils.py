import numpy
import random
import sys
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

tf.logging.set_verbosity(3)


def optimize(sess, target, method='SLSQP'):
    sess.run(tf.global_variables_initializer())

    def loss_callback(loss):
        sys.stderr.write('loss: %30.6f%30s\r' % (loss, ''))

    optimizer = ScipyOptimizerInterface(
        loss=-target,
        method=method,
        options={'maxiter': 10000, 'xtol': 1e-9, 'ftol': 1e-9})
    optimizer.minimize(
        sess,
        fetches=[target],
        loss_callback=loss_callback)
    sys.stderr.write('\n')


def predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        # Replace the last axis with a 3-element vector
        y = numpy.mean(func_values, axis=-1)
        y_lo = numpy.percentile(func_values, (1-ci)*50, axis=-1)
        y_hi = numpy.percentile(func_values, (1+ci)*50, axis=-1)
        return numpy.stack((y, y_lo, y_hi), axis=-1)
