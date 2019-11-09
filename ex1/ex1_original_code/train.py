import tensorflow as tf
from model import mlp, conv_net




def train(model_fn, batch_size, learning_rate=None):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :return:
    """
    # YOUR CODE HERE
    pass


def main():

    # train(mlp, 64)
    # train (conv_net, 64)
    pass


if __name__== "__main__":
    main()

