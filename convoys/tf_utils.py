import numpy
import scipy.stats
import sys
import tensorflow as tf

tf.logging.set_verbosity(3)


def get_hessian(sess, f, param):
    return sess.run(tf.hessians(-f, [param]))[0]


def optimize(sess, target, variables):
    learning_rate_input = tf.placeholder(tf.float32, [])
    optimizer = tf.train.AdamOptimizer(learning_rate_input).minimize(-target)

    best_state_variables = [tf.Variable(tf.zeros(v.shape)) for v in variables]
    store_best_state = [tf.assign(v, u) for (u, v) in zip(variables, best_state_variables)]
    restore_best_state = [tf.assign(u, v) for (u, v) in zip(variables, best_state_variables)]
    sess.run(tf.global_variables_initializer())

    best_step, step = 0, 0
    dec_learning_rate = 1.0
    best_cost = sess.run(target)
    any_var_is_nan = tf.is_nan(tf.add_n([tf.reduce_sum(v) for v in variables]))

    while True:
        inc_learning_rate = 10**(min(step, 240)//40-6)
        learning_rate = min(inc_learning_rate, dec_learning_rate)
        sess.run(optimizer, feed_dict={learning_rate_input: learning_rate})
        if sess.run(any_var_is_nan):
            cost = float('-inf')
        else:
            cost = sess.run(target)
        if cost > best_cost:
            best_cost, best_step = cost, step
            sess.run(store_best_state)
        elif str(cost) in ('-inf', 'nan') or step - best_step > 40:
            sess.run(restore_best_state)
            dec_learning_rate = learning_rate / 10
            best_step = step
        if learning_rate < 1e-6:
            sys.stdout.write('\n')
            break
        step += 1
        sys.stdout.write('step %6d (lr %6.6f): %14.3f%30s' % (step, learning_rate, cost, ''))
        sys.stdout.write('\n' if step % 100 == 0 else '\r')
        sys.stdout.flush()


def fix_t(t):
    # TODO: this is stupid, should at least have tests for it
    t = numpy.array(t)
    if len(t.shape) == 0:
        return t
    elif len(t.shape) == 1:
        return numpy.array([[z] for z in t])
    else:
        return t


def sample_hessian(x, value, hessian, n, ci):
    if ci is None:
        return numpy.dot(x, value)
    else:
        x = numpy.array(x)
        inv_var = numpy.dot(numpy.dot(x.T, hessian), x)
        return numpy.dot(x, value) + scipy.stats.norm.rvs(scale=inv_var**-0.5, size=(1, n))


def predict(func_values, ci):
    if ci is None:
        return func_values
    else:
        axis = len(func_values.shape)-1
        return numpy.mean(func_values, axis=axis), numpy.percentile(func_values, (1-ci)*50, axis=axis), numpy.percentile(func_values, (1+ci)*50, axis=axis)
