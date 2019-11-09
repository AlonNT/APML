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
    result = None   # YOUR CODE HERE

    return result


# dy_dab.shape = YOUR ANSWER HERE
# dy_da.shape = a.shape
# dy_db.shape = b.shape
def matmul_grad_numpy(a: np.ndarray, b: np.ndarray, dy_dab: np.ndarray):
    dy_da = None    # YOUR CODE HERE
    dy_db = None    # YOUR CODE HERE
    return [dy_da, dy_db]


@tf.custom_gradient
def matmul_tf(a, b):
    # use tf.numpy_function

    result = None   # YOUR CODE HERE

    def grad(dy_dab):
        return None  # YOUR CODE HERE

    return result, grad


# ----------------- mse loss --------------
# y.shape = (batch)
# ypredict.shape = (batch)
# the result is a scalar
# dloss_dyPredict.shape = YOUR ANSWER HERE
def mse_numpy(y, ypredict):
    loss = None   # YOUR CODE HERE
    return loss


def mse_grad_numpy(y, yPredict, dy):  # dy is gradient from next node in the graph, not the gradient of our y!
    dloss_dyPredict = None    # YOUR CODE HERE
    dloss_dy = None           # YOUR CODE HERE
    return [dloss_dy, dloss_dyPredict]


@tf.custom_gradient
def mse_tf(y, y_predict):
    # use tf.numpy_function

    loss = None   # YOUR CODE HERE

    def grad(dy):
        return None  # YOUR CODE HERE

    return loss, grad





