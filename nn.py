import tensorflow as tf
import numpy as np
import compute_time
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.core.protobuf import rewriter_config_pb2

def neural_net(x, neurons, is_training, name, mv_decay=0.9, dtype=tf.float32):

    def _batch_normalization(_x):
        beta = tf.get_variable('beta', [_x.get_shape()[-1]],
                           dtype, init_ops.zeros_initializer())
        gamma = tf.get_variable('gamma', [_x.get_shape()[-1]],
                            dtype, init_ops.ones_initializer())
        mv_mean = tf.get_variable('mv_mean', [_x.get_shape()[-1]],
                              dtype, init_ops.zeros_initializer(),
                              trainable=False)
        mv_variance = tf.get_variable('mv_variance', [_x.get_shape()[-1]],
                                  dtype, init_ops.ones_initializer(),
                                  trainable=False)
        mean, variance = tf.nn.moments(_x, [0], name='moments')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                         assign_moving_average(mv_mean, mean,
                                               mv_decay, True))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                         assign_moving_average(mv_variance, variance,
                                               mv_decay, False))
        mean, variance = tf.cond(is_training,
        lambda: (mean, variance),
        lambda: (mv_mean, mv_variance))
        return tf.nn.batch_normalization(_x, mean, variance,
                                     beta, gamma, 1e-6)

    def _layer(_x, out_size, activation_fn):
        w = tf.get_variable('weights',
                    [_x.get_shape().as_list()[-1], out_size],
        dtype, initializers.xavier_initializer())
        return activation_fn(_batch_normalization(tf.matmul(_x, w)))

    with tf.variable_scope(name):
        x = _batch_normalization(x)
        for i in range(len(neurons)):
            with tf.variable_scope('layer_%i_' % (i + 1)):
                x = _layer(x, neurons[i],
    tf.nn.tanh if i < len(neurons)-1 else tf.identity)
    return x

def kolmogorov_train_and_test(xi, x_sde, phi, u_reference, neurons,
    lr_boundaries, lr_values, train_steps,mc_rounds, mc_freq, file_name,
                      dtype=tf.float32):

    def _approximate_errors():
        lr, gs = sess.run([learning_rate, global_step])
        l1_err, l2_err, li_err = 0., 0., 0.
        rel_l1_err, rel_l2_err, rel_li_err = 0., 0., 0.
        for _ in range(mc_rounds):
            l1, l2, li, rl1, rl2, rli \
            = sess.run([err_l_1, err_l_2, err_l_inf,
                    rel_err_l_1, rel_err_l_2, rel_err_l_inf],
                   feed_dict={is_training: False})
            l1_err, l2_err, li_err = (l1_err + l1, l2_err + l2,
                              np.maximum(li_err, li))
            rel_l1_err, rel_l2_err, rel_li_err \
                = (rel_l1_err + rl1, rel_l2_err + rl2,
                np.maximum(rel_li_err, rli))
        l1_err, l2_err = l1_err / mc_rounds, np.sqrt(l2_err / mc_rounds)
        rel_l1_err, rel_l2_err \
            = rel_l1_err / mc_rounds, np.sqrt(rel_l2_err / mc_rounds)
        t_mc = compute_time.time()
        file_out.write('%i, %f, %f, %f, %f, %f, %f, %f, '
                       '%f, %f\n' % (gs, l1_err,  l2_err, li_err,
                                     rel_l1_err, rel_l2_err, rel_li_err, lr,
                                     t1_train - t0_train, t_mc - t1_train))
        file_out.flush()

    t0_train = compute_time.time()
    is_training = tf.placeholder(tf.bool, [])
    u_approx = neural_net(xi, neurons, is_training, 'u_approx', dtype=dtype)
    loss = tf.reduce_mean(tf.squared_difference(u_approx, phi(x_sde)))
    err = tf.abs(u_approx - u_reference)
    err_l_1 = tf.reduce_mean(err)
    err_l_2 = tf.reduce_mean(err ** 2)
    err_l_inf = tf.reduce_max(err)
    rel_err = err / tf.maximum(u_reference, 1e-8)
    rel_err_l_1 = tf.reduce_mean(rel_err)
    rel_err_l_2 = tf.reduce_mean(rel_err ** 2)
    rel_err_l_inf = tf.reduce_max(rel_err)
    global_step = tf.get_variable('global_step', [], tf.int32,
                                  tf.constant_initializer(0),
                                  trainable=False)
    learning_rate = tf.train.piecewise_constant(global_step,
                                                lr_boundaries,
                                                lr_values)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'u_approx')
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    file_out = open(file_name, 'w')
    file_out.write('step, l1_err, l2_err, li_err, l1_rel, '
                   'l2_rel, li_rel, learning_rate, time_train, time_mc\n')
    config_proto = tf.ConfigProto()
    off =  rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    with tf.Session(config=config_proto) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_steps):
            if step % mc_freq == 0:
                t1_train = compute_time.time()
                _approximate_errors()
                t0_train = compute_time.time()
            sess.run(train_op, feed_dict={is_training: True})
        t1_train = compute_time.time()
        _approximate_errors()
    file_out.close()
