import tensorflow as tf


def get_modified_dropout_rate(test_mode: tf.Tensor = None, dropout_rate: int = 0):
    """
    Define the dropout-rate to be 0 in case the input is in test-mode.
    :param test_mode: A Tensor holding a scaler boolean value,
                      indicating whether this run is in test-mode
    :param dropout_rate: The drop-out rate, a number in [0,1)
    :return: A float tensor holding the dropout rate (in case this run is in train-mode),
             or 0 (in case this run is in test-mode).
    """
    if test_mode is None:
        test_mode = tf.constant(value=0, dtype=tf.float32, shape=(), name='test_mode')

    train_mode = tf.math.logical_not(test_mode, name='train_mode')
    train_mode_float = tf.cast(train_mode, tf.float32, name='train_mode_float')
    dropout_rate_orig = tf.constant(value=dropout_rate,
                                    dtype=tf.float32,
                                    shape=(),
                                    name='dropout_rate_orig')
    dropout_rate_tensor = tf.multiply(train_mode_float, dropout_rate_orig, name='dropout_rate')

    return dropout_rate_tensor


def get_mlp_variables(input_channels: int, n_labels: int, hidden_layer_dim: int):
    """
    :param input_channels: Number of input channels.
    :param n_labels: Number of labels.
    :param hidden_layer_dim: Number of channels in the hidden layer.
    :return: A dictionary containing the variables needed for the
             Multi-Layer-Perceptron's computational graph.
    """
    variables = dict()

    variables['w1'] = tf.get_variable(name='w1', shape=(input_channels, hidden_layer_dim), trainable=True)
    variables['b1'] = tf.get_variable(name='b1', shape=(hidden_layer_dim,), trainable=True)
    variables['w2'] = tf.get_variable(name='w2', shape=(hidden_layer_dim, n_labels), trainable=True)
    variables['b2'] = tf.get_variable(name='b2', shape=(n_labels,), trainable=True)

    return variables


def mlp(x: tf.Tensor, n_labels: int, test_mode: tf.Tensor = None, dropout_rate: int = 0):
    """
    multi layer perceptron: x -> linear > relu > linear.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param n_labels: the dimension of the output.
    :param test_mode: a tensor holding a boolean value which indicates if this is test-mode.
                      This is needed in order to avoid dropout while testing the model.
    :param dropout_rate: if given a number greater than 0,
                         adds a dropout layer on the hidden layer
                         (just before the last affine layer).
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # The number of channels in the Multi-Layer-Perceptron hidden layer
    hidden_layer_dim = 100

    height = x.shape[1]
    width = x.shape[2]

    dropout_rate_tensor = get_modified_dropout_rate(test_mode, dropout_rate)
    variables = get_mlp_variables(height * width, n_labels, hidden_layer_dim)

    # Build the computational graph
    x_flat = tf.reshape(x, shape=(-1, height * width), name='x_flat')
    affine = tf.add(tf.matmul(x_flat, variables['w1'], name='affine_matmul'), variables['b1'], name='affine')
    relu = tf.nn.relu(affine, name='relu')
    drop = tf.nn.dropout(x=relu, rate=dropout_rate_tensor, name='drop')
    predict = tf.add(tf.matmul(drop, variables['w2'], name='predict_matmul'), variables['b2'], name='predict')

    return predict


def conv2d_block(input_tensor: tf.Tensor, kernel: tf.Tensor, bias: tf.Tensor,
                 strides: tuple, padding: str, name: str):
    """
    Constructs a convolution block,
    containing a convolution that follows by an addition of a bias vector.
    :param input_tensor: A 4D Tensor, in the data-format NHWC.
    :param kernel: A Tensor of shape [filter_height, filter_width, in_channels, out_channels]
    :param bias: A Tensor containing the bias vector
    :param strides: The stride of the sliding window for each dimension of `input_tensor`
    :param padding: Either the `string` `"SAME"` or `"VALID"` indicating
                    the type of padding algorithm to use.
    :param name: The name of this convolution block.
    :return: A Tensor which is the composition of the convolution and bias addition layers.
    """
    conv = tf.nn.conv2d(input_tensor, kernel, strides, padding, name=f'{name}_conv')
    b = tf.add(conv, bias, name=f'{name}_bias')

    return b


def get_conv_net_variables(filter_size: int, num_channels: int, affine_dimension: int, n_labels: int):
    """
    Define variables to store the weights of the ConvNet.
    These include two filters for the convolution layers, in addition to bias vectors,
    and a matrix & bias-vector for the last affine layer.
    :param filter_size: The size of the filter.
    :param num_channels: Number of output channels of each convolution.
    :param affine_dimension: Number of channels in the affine dimension layer.
    :param n_labels: Number of labels.
    :return: A dictionary containing the variables needed for the ConvNet's computational graph.
    """
    variables = dict()

    variables['k1'] = tf.get_variable(name='k1',
                                      shape=(filter_size, filter_size, 1, num_channels),
                                      trainable=True)
    variables['b1'] = tf.get_variable(name='b1',
                                      shape=(num_channels,),
                                      trainable=True)
    variables['k2'] = tf.get_variable(name='k2',
                                      shape=(filter_size, filter_size, num_channels, num_channels),
                                      trainable=True)
    variables['b2'] = tf.get_variable(name='b2',
                                      shape=(num_channels,),
                                      trainable=True)
    variables['w'] = tf.get_variable(name='w',
                                     shape=(affine_dimension, n_labels),
                                     trainable=True)
    variables['b'] = tf.get_variable(name='b',
                                     shape=(n_labels,),
                                     trainable=True)

    return variables


def conv_net(x: tf.Tensor, n_labels: int, test_mode: tf.Tensor = None, dropout_rate: int = 0):
    """
    ConvNet.
    in the convolution use 3x3 filters with 1x1 strides, 20 filters each time.
    in the  maxpool use 2x2 pooling.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param n_labels: the dimension of the output.
    :param test_mode: a tensor holding a boolean value which indicates if this is test-mode.
                      This is needed in order to avoid dropout while testing the model.
    :param dropout_rate: if given a number greater than 0,
                         adds a dropout after each MaxPool layer (twice total).
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    num_channels = 20
    filter_size = 3
    pool_size = 2
    stride_size = 1

    # Define H & W. Note that the number of input channels is 1.
    height = x.shape[1]
    width = x.shape[2]

    # The dimension of the flattened vector that will enter the affine layer
    # is the following calculation, because there are two pooling layers each of pool_size.
    affine_dimension = (height // (pool_size ** 2)) * (width // (pool_size ** 2)) * num_channels

    dropout_rate_tensor = get_modified_dropout_rate(test_mode, dropout_rate)
    variables = get_conv_net_variables(filter_size, num_channels, affine_dimension, n_labels)

    # Build the model.
    x_reshaped = tf.reshape(tensor=x,
                            shape=(-1, height, width, 1),
                            name='x_reshaped')
    conv1 = conv2d_block(input_tensor=x_reshaped,
                         kernel=variables['k1'],
                         strides=(1, stride_size, stride_size, 1),
                         padding='SAME',
                         name='conv1',
                         bias=variables['b1'])
    pool1 = tf.nn.max_pool2d(input=conv1,
                             ksize=(1, pool_size, pool_size, 1),
                             strides=(1, pool_size, pool_size, 1),
                             padding='SAME',
                             name='pool1')
    drop1 = tf.nn.dropout(x=pool1,
                          rate=dropout_rate_tensor,
                          name='drop1')
    conv2 = conv2d_block(input_tensor=drop1,
                         kernel=variables['k2'],
                         strides=(1, stride_size, stride_size, 1),
                         padding='SAME',
                         name='conv2',
                         bias=variables['b2'])
    pool2 = tf.nn.max_pool2d(input=conv2,
                             ksize=(1, pool_size, pool_size, 1),
                             strides=(1, pool_size, pool_size, 1),
                             padding='SAME',
                             name='pool2')
    pool2_flat = tf.reshape(tensor=pool2,
                            shape=(-1, affine_dimension),
                            name='pool2_flat')
    drop2 = tf.nn.dropout(x=pool2_flat,
                          rate=dropout_rate_tensor,
                          name='drop2')
    predict = tf.add(tf.matmul(drop2,
                               variables['w'],
                               name='matmul'),
                     variables['b'],
                     name='predict')

    return predict
