import tensorflow as tf


def mlp(x: tf.Tensor, nlabels):
    """
    multi layer perceptrone: x -> linear > relu > linear.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # YOUR CODE HERE
    pass


def conv_net(x: tf.Tensor, nlabels):
    """
    convnet.
    in the convolution use 3x3 filteres with 1x1 strides, 20 filters each time.
    in the  maxpool use 2x2 pooling.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # YOUR CODE HERE
    pass