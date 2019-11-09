import numpy as np
import tensorflow as tf
from operations import matmul_tf, mse_tf


def check_matmul():
    n = 8
    m = 32
    k = 16

    with tf.GradientTape(persistent=True) as t:
        a = tf.get_variable(dtype=tf.float32, shape=(n, k), name='a')
        b = tf.get_variable(dtype=tf.float32, shape=(k, m), name='b')

        tf_c = tf.matmul(a, b)
        my_c = matmul_tf(a, b)

        tf_d = tf.exp(tf_c, 3)
        my_d = tf.exp(my_c, 3)

    tf_dc_da, tf_dc_db = t.gradient(tf_c, [a, b])
    my_dc_da, my_dc_db = t.gradient(my_c, [a, b])

    tf_dd_da, tf_dd_db = t.gradient(tf_d, [a, b])
    my_dd_da, my_dd_db = t.gradient(my_d, [a, b])

    c_diff = np.max(np.abs(tf_c.numpy() - my_c.numpy()))
    d_diff = np.max(np.abs(tf_d.numpy() - my_d.numpy()))
    dc_da_diff = np.max(np.abs(tf_dc_da.numpy() - my_dc_da.numpy()))
    dc_db_diff = np.max(np.abs(tf_dc_db.numpy() - my_dc_db.numpy()))
    dd_da_diff = np.max(np.abs(tf_dd_da.numpy() - my_dd_da.numpy()))
    dd_db_diff = np.max(np.abs(tf_dd_db.numpy() - my_dd_db.numpy()))

    if any([v > 0.0001 for v in [c_diff, d_diff, dc_da_diff, dc_db_diff, dd_da_diff, dd_db_diff]]):
        print("FAILED (matmul) :(")
    else:
        print("PASSED (matmul) :)")


def check_mse():
    n = 32

    with tf.GradientTape(persistent=True) as t:
        y_true = tf.get_variable(dtype=tf.float32, shape=(n,), name='y_true')
        y_pred = tf.get_variable(dtype=tf.float32, shape=(n,), name='y_pred')

        tf_c = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
        my_c = mse_tf(y_true, y_pred)

        tf_d = tf.exp(tf_c, 3)
        my_d = tf.exp(my_c, 3)

    tf_dc_da, tf_dc_db = t.gradient(tf_c, [y_true, y_pred])
    my_dc_da, my_dc_db = t.gradient(my_c, [y_true, y_pred])

    tf_dd_da, tf_dd_db = t.gradient(tf_d, [y_true, y_pred])
    my_dd_da, my_dd_db = t.gradient(my_d, [y_true, y_pred])

    c_diff = np.max(np.abs(tf_c.numpy() - my_c.numpy()))
    d_diff = np.max(np.abs(tf_d.numpy() - my_d.numpy()))
    dc_da_diff = np.max(np.abs(tf_dc_da.numpy() - my_dc_da.numpy()))
    dc_db_diff = np.max(np.abs(tf_dc_db.numpy() - my_dc_db.numpy()))
    dd_da_diff = np.max(np.abs(tf_dd_da.numpy() - my_dd_da.numpy()))
    dd_db_diff = np.max(np.abs(tf_dd_db.numpy() - my_dd_db.numpy()))

    if any([v > 0.0001 for v in [c_diff, d_diff, dc_da_diff, dc_db_diff, dd_da_diff, dd_db_diff]]):
        print("FAILED (mse) :(")
    else:
        print("PASSED (mse) :)")


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.set_random_seed(311602536)

    check_matmul()
    check_mse()
