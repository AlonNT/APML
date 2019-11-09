import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operations import matmul_tf, mse_tf


def load_data():
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    X = np.array(boston_dataset.data)
    y = np.array(boston_dataset.target)
    return X, y


def model(x: tf.Tensor):
    """
    linear regression model: y_predict = W*x + b
    please use your matrix multiplication implementation.
    :param x: symbolic tensor with shape (batch, dim)
    :return:  a tuple that contains: 1.symbolic tensor y_predict, 2. list of the variables used in the model: [W, b]
                the result shape is (batch)
    """
    batch, dim = x.shape
    
    w = tf.get_variable(name='w', shape=(dim, 1), dtype=tf.float32)
    b = tf.get_variable(name='b', shape=(1,), dtype=tf.float32)

    y_predict = matmul_tf(x, w) + b

    return y_predict, [w, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :return: list contains the mean loss from each epoch.
    """
    # YOUR CODE HERE
    pass


def main():
    # losses = train(50, 0.01, 32)
    # plt.plot(losses)
    # plt.show()
    pass


if __name__== "__main__":
  main()

