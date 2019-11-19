import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import mlp, conv_net


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_steps = 100000
print_steps = 1000
save_checkpoint_steps = 1000


def get_data():
    """
    Get the Fashion MNIST dataset, in the proper data-types and shapes.
    The images are transformed from uint8 in 0,...,255 to float in [0,1].
    The labels are transformed from uint8 to int32.
    """
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    n_train, h, w = x_train.shape
    n_test = x_test.shape[0]
    n_labels = len(np.unique(y_train))

    # Reshape the images to include a channels dimension (which is 1),
    # convert them to float32 and divide by 255 to get a value between 0 and 1
    x_train = x_train.reshape(-1, h, w, 1).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, h, w, 1).astype(np.float32) / 255.0

    # Convert the labels to int32 and not uint8, because this is what
    # TensorFlow wants (in the loss function sparse_softmax_cross_entropy_with_logits).
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test, n_train, n_test, n_labels, h, w


def train(model_fn, batch_size, learning_rate=None, **model_kwargs):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :return:
    """
    x_train, y_train, x_test, y_test, n_train, n_test, n_labels, h, w = get_data()

    x = tf.placeholder(dtype=tf.float32, shape=(None, h, w, 1), name='x')
    y = tf.placeholder(dtype=tf.int32, shape=(None,), name='y')
    test_mode = tf.placeholder_with_default(
        input=tf.constant(value=False, dtype=tf.bool, shape=(), name='test_mode_default'),
        shape=(),
        name='test_mode'
    )

    # Define the model.
    model_kwargs['test_mode'] = test_mode
    y_predict = model_fn(x, n_labels, **model_kwargs)

    # Define the loss function.
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=y_predict, name='non_reduced_loss'),
        name='reduced_loss'
    )

    # Define the optimizer.
    optimizer_kwargs = dict() if learning_rate is None else {'learning_rate': learning_rate}
    optimizer = tf.train.AdamOptimizer(**optimizer_kwargs).minimize(loss)

    # Define accuracy operator.
    correct_pred = tf.equal(tf.cast(tf.argmax(y_predict, axis=1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add loss and accuracy to the summary, in order to view it in TensorBoard.
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    summarize = tf.summary.merge_all()

    # Collect losses and accuracies, both for train-data and for test-data.
    train_losses = list()
    train_accuracies = list()
    test_losses = list()
    test_accuracies = list()

    init = tf.global_variables_initializer()

    # Define the directories that will be created with the TensorBoard data and checkpoints.
    now_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logs_dir_name = os.path.join('logs', model_fn.__name__, now_str)
    checkpoint_directory = os.path.join(logs_dir_name, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    saver = tf.train.Saver(max_to_keep=num_steps)
    train_writer = tf.summary.FileWriter(os.path.join(logs_dir_name, 'train'),
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(logs_dir_name, 'test'),
                                        tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_steps):
            # Sample a random mini-batch of samples from the training data.
            train_batch_indices = np.random.choice(n_train, size=batch_size)
            x_train_batch = x_train[train_batch_indices]
            y_train_batch = y_train[train_batch_indices]

            # Run the graph in that mini-batch, including the optimizer to update the weights.
            train_loss, train_accuracy, train_summary, _ = sess.run(
                fetches=[loss, accuracy, summarize, optimizer],
                feed_dict={x: x_train_batch, y: y_train_batch}
            )

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_writer.add_summary(train_summary, i)

            # Sample a random mini-batch of samples from the testing data.
            test_batch_indices = np.random.choice(n_test, size=batch_size)
            x_test_batch = x_test[test_batch_indices]
            y_test_batch = y_test[test_batch_indices]

            # Run the graph in that mini-batch, excluding the optimizer (to avoid
            # update the weights according to the test data, strictly forbidden :))
            test_loss, test_accuracy, test_summary = sess.run(
                fetches=[loss, accuracy, summarize],
                feed_dict={x: x_test_batch, y: y_test_batch, test_mode: True}
            )

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_writer.add_summary(test_summary, i)

            # Every print_steps iterations print train-loss.
            if i % print_steps == 0:
                print("Iter {:05d} train-loss {:.2f} train-accuracy {:.2f}".format(i, train_loss, train_accuracy))
                print("Iter {:05d}  test-loss {:.2f}  test-accuracy {:.2f}".format(i, test_loss, test_accuracy))

            # Every save_checkpoint_steps iterations save a checkpoint.
            if i % save_checkpoint_steps == 0:
                saver.save(sess, save_path=checkpoint_prefix, global_step=i)

    # After the training was finished, load the latest checkpoint and evaluate the model on
    # all samples in the test-data.
    all_test_losses = list()
    all_test_accuracies = list()
    with tf.Session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
        new_saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        new_saver.restore(sess, latest_checkpoint)
        for i in range(n_test // batch_size):
            x_test_batch = x_test[i:i+batch_size]
            y_test_batch = y_test[i:i+batch_size]

            # Run the graph in that mini-batch, including the optimizer to update the weights.
            test_loss, test_accuracy = sess.run(
                fetches=[loss, accuracy],
                feed_dict={x: x_test_batch, y: y_test_batch}
            )
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)

        all_test_loss = np.array(all_test_losses).mean()
        all_test_accuracy = np.array(all_test_accuracies).mean()
        print("Total test-loss {:.2f} test-accuracy {:.2f}".format(all_test_loss, all_test_accuracy))

    train_writer.close()
    test_writer.close()


def find_adversarial_image(checkpoint):
    """
    Finds and plots the original image with the true-label and prediction,
    and the adversarial image with the (wrong) prediction.
    :param checkpoint: A checkpoint of a trained model.
    """
    x_train, y_train, x_test, y_test, n_train, n_test, n_labels, h, w = get_data()

    # Load the saved graph.
    new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
    graph = tf.get_default_graph()

    # Extract the placeholders for the loaded graph,
    # and create additional tensors which calculate the classes' probabilities
    # and final class prediction (argmax of the probabilities).
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    test_mode = graph.get_tensor_by_name('test_mode:0')
    predict_logits = graph.get_tensor_by_name('predict:0')
    predict_prob = tf.nn.softmax(logits=predict_logits, axis=1)
    # predict_class = tf.argmax(predict_prob, axis=1)
    loss = graph.get_tensor_by_name('reduced_loss:0')

    # Sample a random image from the training data, and sample a wrong label for it.
    i = np.random.randint(n_train)
    image = x_train[i]
    true_label = y_train[i]
    target_label = np.random.choice(list(set(np.arange(n_labels)) - {true_label}))

    # Create the image-loss, which implicates that the
    # resulting image will be close to the original one.
    image_tensor = tf.constant(value=image, dtype=tf.float32, name='source_image')
    image_loss = tf.reduce_mean(tf.abs(tf.subtract(x, image_tensor, name='sub'), 'abs'), name='image_loss')

    # Define the new loss as the weighted sum of the original loss and the image-loss.
    image_loss_weight = 0.05
    new_loss = tf.add(loss, image_loss_weight * image_loss)

    # Create a symbolic tensor calculating the gradient
    # of the new loss with respect to the input image.
    grad = tf.gradients(ys=new_loss, xs=[x])

    curr_image = image.copy().reshape(1, 28, 28, 1)
    orig_classes_prob = None
    target_label_reshaped = np.array([target_label], dtype=np.int32)

    with tf.Session() as sess:
        new_saver.restore(sess, checkpoint)

        for i in range(10000):
            # Calculate the gradient with respect to the input image,
            # as well as the predicted classes' probabilities.
            grad_image, classes_prob = sess.run(
                [grad, predict_prob],
                feed_dict={x: curr_image, y: target_label_reshaped, test_mode: True}
            )

            # Take the relevant values, as the sess.run return a list of nested values...
            grad_image = grad_image[0][0]
            classes_prob = classes_prob[0]

            # In case this is the first iteration, save the classes' probabilities
            # as they are the original prediction.
            if i == 0:
                orig_classes_prob = np.copy(classes_prob)

            # print('True/Target-label probabilities = {:.2f} ; {:.2f}'.format(classes_prob[target_label],
            #                                                                  classes_prob[true_label]))

            if classes_prob[target_label] > 0.95:
                break

            # Update the current-image with respect to the gradient of the new loss function.
            # This makes the loss function decrease, so the prediction gets close to the target
            # label, and the image remains not fat from the original one.
            learning_rate = 0.001
            curr_image -= learning_rate * grad_image

    # Plot the original image, the added noise, and the final adversarial image.
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('{}, w.p. {:.4f}'.format(class_names[true_label], orig_classes_prob[true_label]))

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(curr_image[0, :, :, 0] - image[:, :, 0], cmap='gray')
    plt.title('Add noise...')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(curr_image[0, :, :, 0], cmap='gray')
    plt.title('{}, w.p. {:.4f}'.format(class_names[target_label], classes_prob[target_label]))
    plt.show()


def main():
    # train(mlp, 64)
    # train(mlp, 64, dropout_rate=0.25)
    # train(conv_net, 64)
    train(conv_net, 64, dropout_rate=0.25)
    # find_adversarial_image(checkpoint='logs/conv_net/2019_11_16_16_38_00_drop_025/checkpoints/ckpt-50000')
    pass


if __name__ == "__main__":
    main()
