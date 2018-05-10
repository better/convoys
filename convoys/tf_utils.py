import numpy
import random
import sys
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

tf.logging.set_verbosity(3)


def optimize(sess, target, update_callback=None):
    sess.run(tf.global_variables_initializer())

    def step_callback(*args):
        if update_callback:
            update_callback(sess)

    def loss_callback(loss):
        sys.stderr.write('Current loss: %30.3f\r' % loss)

    optimizer = ScipyOptimizerInterface(loss=-target,
                                        method='SLSQP',
                                        options={'maxiter': 10000})
    optimizer.minimize(sess,
                       fetches=[target],
                       step_callback=step_callback,
                       loss_callback=loss_callback)

    sys.stderr.write('\n')


def get_tweaker(sess, target, z):
    new_z = tf.placeholder(tf.float32, shape=z.shape)
    assign_z = tf.assign(z, new_z)

    def tweak_z(sess):
        # tf.igamma doesn't compute the gradient wrt a properly
        # So let's just try small perturbations
        # https://github.com/tensorflow/tensorflow/issues/17995
        z_value = sess.run(z)
        res = {}
        for z_mult in [0.97, 1.0, 1.03]:
            sess.run(assign_z, feed_dict={new_z: z_value * z_mult})
            res[z_value * z_mult] = sess.run(target)
        sess.run(assign_z, feed_dict={new_z: max(res.keys(), key=res.get)})

    return tweak_z


def predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        # Replace the last axis with a 3-element vector
        y = numpy.mean(func_values, axis=-1)
        y_lo = numpy.percentile(func_values, (1-ci)*50, axis=-1)
        y_hi = numpy.percentile(func_values, (1+ci)*50, axis=-1)
        return numpy.stack((y, y_lo, y_hi), axis=-1)
