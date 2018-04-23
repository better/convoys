import numpy
import random
import sys
import tensorflow as tf

tf.logging.set_verbosity(3)


def get_batch_placeholders(vs):
    return [tf.placeholder(tf.float32, shape=((None,) + v.shape[1:])) for v in vs]


def optimize(sess, target_batch, target_global=None, placeholders={},
             batch_size=128):
    if placeholders:
        n = int(list(placeholders.values())[0].shape[0])
        indexes = list(range(n))
    else:
        n = 1
        indexes = []

    learning_rate_input = tf.placeholder(tf.float32, [])
    optimizer_batch = tf.train.AdamOptimizer(learning_rate_input) \
        .minimize(-target_batch)
    if target_global is not None:
        optimizer_global = tf.train.AdamOptimizer(learning_rate_input) \
            .minimize(-target_global)
    sess.run(tf.global_variables_initializer())

    best_cost, best_step, step = float('-inf'), 0, 0
    learning_rate = 3e-3
    while True:
        cost = 0
        random.shuffle(indexes)
        for i in range(0, n, batch_size):
            feed_dict_batch = dict(
                [(learning_rate_input, learning_rate)] +
                [(placeholder, v[indexes[i:min(i+batch_size, n)]])
                 for placeholder, v in placeholders.items()])
            sess.run(optimizer_batch, feed_dict=feed_dict_batch)
            cost += sess.run(target_batch, feed_dict=feed_dict_batch)

        if target_global is not None:
            feed_dict_global = dict(
                [(learning_rate_input, learning_rate)] +
                [(placeholder, v) for placeholder, v in placeholders.items()])
            sess.run(optimizer_global, feed_dict=feed_dict_global)
            cost += sess.run(target_global, feed_dict=feed_dict_global)

        if cost > best_cost:
            best_cost, best_step = cost, step
        if step - best_step > 20:
            break
        sys.stdout.write('step %6d (%6d since best): %14.3f%30s' % (step, step-best_step, cost, ''))
        sys.stdout.write('\n' if step % 100 == 0 else '\r')
        sys.stdout.flush()
        step += 1
        yield  # Let the caller do any custom stuff here


def predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        # Replace the last axis with a 3-element vector
        y = numpy.mean(func_values, axis=-1)
        y_lo = numpy.percentile(func_values, (1-ci)*50, axis=-1)
        y_hi = numpy.percentile(func_values, (1+ci)*50, axis=-1)
        return numpy.stack((y, y_lo, y_hi), axis=-1)
