
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

nnet = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('conv', layers.Conv2DLayer),
        ('pool', layers.MaxPool2DLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    hidden_num_units = 100,
    conv_num_filters=8, conv_filter_size=(2, 2), pool_pool_size=(2, 2),
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )