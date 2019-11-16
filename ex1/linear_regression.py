import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ex1.operations import matmul_tf, mse_tf


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

    y_predict = tf.add(matmul_tf(x, w), b)
    # y_predict = tf.add(tf.matmul(x, w), b)  # Works the same :)

    return y_predict, [w, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :param batch_size: number of samples in each mini-batch
    :return: list contains the mean loss from each epoch.
    """
    features, labels = load_data()
    features_normalized = ((features - features.mean(axis=0)) / features.std(axis=0))
    n_samples, dim = features.shape

    x = tf.placeholder(dtype=tf.float32, shape=(None, dim), name='x')
    y_true = tf.placeholder(dtype=tf.float32, shape=(None,), name='y')

    y_predict, [w, b] = model(x)
    loss = mse_tf(y_true, y_predict)

    # This returns the losses for all samples  and not the reduced loss,
    # but other than that it is the same as my mse function.
    # loss = tf.keras.losses.MSE(y_true, y_predict)

    grad = tf.gradients(ys=[loss], xs=[w, b], name='gradients')

    update_w = tf.assign_sub(w, learning_rate * grad[0])
    update_b = tf.assign_sub(b, learning_rate * grad[1])

    training_step = tf.group(update_w, update_b)

    epochs_losses = list()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            permutation = np.random.permutation(n_samples)
            features_shuf = features_normalized[permutation]
            labels_shuf = labels[permutation]

            batches_losses = list()
            start_i = 0
            while start_i < n_samples:
                end_i = start_i + batch_size

                batch_data = features_shuf[start_i:end_i]
                batch_labels = labels_shuf[start_i:end_i]

                batch_loss, _ = sess.run(fetches=[loss, training_step],
                                         feed_dict={x: batch_data,
                                                    y_true: batch_labels})

                batches_losses.append(batch_loss)
                start_i = end_i

            epochs_losses.append(np.array(batches_losses).mean())

    return epochs_losses


def main():
    losses = train(50, 0.0001, 32)
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
