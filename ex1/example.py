import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(0)


def generate_data():
    bias = 5
    x = np.random.randn(1000)
    y = x + bias
    return x, y


def model(x):
    b = tf.get_variable('b', shape=[1, 1], dtype=tf.float32)
    return x + b, b


def train(batch_size, learning_rate, epochs):
    x_ph = tf.placeholder(tf.float32, shape=[batch_size], name='x-input')
    y_ph = tf.placeholder(tf.float32, shape=[batch_size], name='y-input')

    # the model
    with tf.name_scope("model"):
        y_predict, bias = model(x_ph)
        loss = tf.reduce_sum(tf.pow(y_ph - y_predict, 2, name='non-reduced-loss'), name='reduced_loss')

    gradients = tf.gradients(loss, [bias])
    grad_b = gradients[0]

    training_step = tf.assign(bias, bias - grad_b * learning_rate)

    # finished to declare the computational graph! lets take a closer look using tensorboard:
    writer = tf.summary.FileWriter('/home/omri/PycharmProjects/tf_recitation_2/graphs/graph', tf.get_default_graph())

    # lets load the data and start training!
    x, y = generate_data()
    losses = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            epoch_losses = []
            for i in range(y.size // batch_size):
                x_batch = x[i * batch_size:(i + 1) * batch_size]
                y_batch = y[i * batch_size:(i + 1) * batch_size]

                _, loss_value = sess.run([training_step, loss], feed_dict={x_ph: x_batch, y_ph: y_batch})
                print(loss_value)
                epoch_losses.append(loss_value)

            losses.append(np.mean(epoch_losses))
        b_val = sess.run(bias)

    print('the learned bias is {}'.format(b_val))
    return losses


def main():
    losses = train(50, 0.001, 32)
    plt.plot(losses)
    plt.grid()
    plt.show()
    pass


if __name__== "__main__":
  main()

