import tensorflow as tf
import numpy as np


# ----------------- relu example --------------------

# the relu itself
def relu_numpy(x: np.ndarray):
    result = np.zeros_like(x)
    result[x > 0] = x[x > 0]
    return result


# the relu gradient
def relu_grad_numpy(x: np.ndarray, dy: np.ndarray):
    # x and y should have the same shapes.
    result = np.zeros_like(x)
    result[x > 0] = dy[x > 0]
    return result


# the relu tensorflow operation
@tf.custom_gradient
def relu_tf(x):
    result = tf.numpy_function(relu_numpy, [x], tf.float32, name='my_relu_op')

    def grad(dy):
        return tf.numpy_function(relu_grad_numpy, [x, dy], tf.float32, name='my_relu_grad_op')

    return result, grad


# ----------------- batch matrix multiplication --------------
# a.shape = (n, k)
# b.shape = (k, m)
def matmul_numpy(a: np.ndarray, b: np.ndarray):
    result = np.matmul(a, b)
    return result


# dy_dab.shape = (n, m), which is the same shape as a * b
# dy_da.shape = a.shape, which is (n, k)
# dy_db.shape = b.shape, which is (k, m)
def matmul_grad_numpy(a: np.ndarray, b: np.ndarray, dy_dab: np.ndarray):
    # Note that the shape of dy_da is the same shape of a - (n, k).
    # Indeed, dy_dab is of shape (n, m) and b^T is of shape (m, k)
    dy_da = np.matmul(dy_dab, np.transpose(b))

    # Note that the shape of dy_db is the same shape of b - (k, m).
    # Indeed, dy_dab is of shape (n, m) and a^T is of shape (k, n)
    dy_db = np.matmul(np.transpose(a), dy_dab)

    return [dy_da, dy_db]


@tf.custom_gradient
def matmul_tf(a, b):
    result = tf.numpy_function(matmul_numpy, [a, b], tf.float32, name='my_matmul_op')

    def grad(dy_dab):
        return tf.numpy_function(matmul_grad_numpy, [a, b, dy_dab],
                                 [tf.float32, tf.float32], name='my_matmul_grad_op')

    return result, grad


# ----------------- mse loss --------------
# y.shape = (batch)
# ypredict.shape = (batch)
# the result is a scalar
# dloss_dyPredict.shape = YOUR ANSWER HERE
def mse_numpy(y, ypredict):
    loss = np.mean(np.square(y - ypredict))
    return loss


def mse_grad_numpy(y, yPredict, dy):  # dy is gradient from next node in the graph, not the gradient of our y!
    dloss_dyPredict = dy * (1 / y.shape[0]) * 2 * (yPredict - y)
    dloss_dy = dy * (1 / y.shape[0]) * 2 * (y - yPredict)
    return [dloss_dy, dloss_dyPredict]


@tf.custom_gradient
def mse_tf(y, y_predict):
    # use tf.numpy_function

    loss = tf.numpy_function(mse_numpy, [y, y_predict], tf.float32, name='my_mse_op')

    def grad(dy):
        return tf.numpy_function(mse_grad_numpy,
                                 [y, y_predict, dy],
                                 [tf.float32, tf.float32], name='my_mse_grad_op')

    return loss, grad
