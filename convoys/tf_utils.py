import numpy
import sklearn.utils
import sys
import tensorflow as tf

tf.logging.set_verbosity(3)


def get_batch_placeholders(vs):
    return [tf.placeholder(tf.float32, shape=((None,) + v.shape[1:])) for v in vs]


def optimize(sess, target_batch, target_global=None, placeholders={},
             batch_size=1024, update_callback=None):
    if placeholders:
        n = int(list(placeholders.values())[0].shape[0])
    else:
        n = 1

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
        shuffled = sklearn.utils.shuffle(*placeholders.values())
        for i in range(0, n, batch_size):
            feed_dict_batch = dict(
                [(learning_rate_input, learning_rate)] +
                [(placeholder, v[i:min(i+batch_size, n)])
                 for placeholder, v in zip(placeholders.keys(), shuffled)])
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

        if update_callback:
            update_callback(sess)


def get_tweaker(sess, target, z, feed_dict):
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
            res[z_value * z_mult] = sess.run(target, feed_dict=feed_dict)
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
