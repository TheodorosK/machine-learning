#!/usr/bin/env python
'''This module defines a multi-level perceptron and convolutional MLP.
'''
import abc
import json

import lasagne
import numpy as np
import theano
import theano.tensor as T


class EndOfEpochUpdate(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, epoch, total_epochs):
        pass


class AdaptiveLearningRateLog(EndOfEpochUpdate):
    def __init__(self, config, shared_learn_rate, shared_momentum):
        super(AdaptiveLearningRateLog, self).__init__()
        shared_learn_rate.set_value(10.0 ** config['log_learning_rate_start'])
        shared_momentum.set_value(config['momentum'])
        self.__config = config
        self.__shared_learn_rate = shared_learn_rate
        self.__shared_momentum = shared_momentum

        self.__one_time_init_done = False
        self.__learning_values = None  # Set by __try_one_time_init

    def __try_one_time_init(self, total_epochs):
        if self.__one_time_init_done:
            return
        self.__one_time_init_done = True

        self.__learning_values = np.logspace(
            self.__config['log_learning_rate_start'],
            self.__config['log_learning_rate_end'],
            total_epochs)

    def update(self, epoch, total_epochs):
        self.__try_one_time_init(total_epochs)

        new_learning_rate = self.__learning_values[epoch]
        print "Updated learning-rate: %e" % new_learning_rate
        self.__shared_learn_rate.set_value(np.float32(new_learning_rate))


class AdaptiveLearningRateLin(EndOfEpochUpdate):
    def __init__(self, config, shared_learn_rate, shared_momentum):
        super(AdaptiveLearningRateLin, self).__init__()
        shared_learn_rate.set_value(config['learning_rate_start'])
        shared_momentum.set_value(config['momentum'])
        self.__config = config
        self.__shared_learn_rate = shared_learn_rate
        self.__shared_momentum = shared_momentum

        self.__one_time_init_done = False
        self.__learning_values = None  # Set by __try_one_time_init

    def __try_one_time_init(self, total_epochs):
        if self.__one_time_init_done:
            return
        self.__one_time_init_done = True

        self.__learning_values = np.linspace(
            self.__config['learning_rate_start'],
            self.__config['learning_rate_end'],
            total_epochs)

    def update(self, epoch, total_epochs):
        self.__try_one_time_init(total_epochs)

        new_learning_rate = self.__learning_values[epoch]
        print "Updated learning-rate: %e" % new_learning_rate
        self.__shared_learn_rate.set_value(np.float32(new_learning_rate))


class StaticLearningRate(EndOfEpochUpdate):
    def __init__(self, config, shared_learn_rate, shared_momentum):
        super(StaticLearningRate, self).__init__()
        shared_learn_rate.set_value(config['learning_rate'])
        shared_momentum.set_value(config['momentum'])

    def update(self, epoch, total_epochs):
        pass


def LearningRateFactory(config, shared_learn_rate, shared_momentum):
    if all(k in config.keys() for k in
           ['log_learning_rate_start', 'log_learning_rate_end']):
        print "Enabling Adaptive Learning Rate (Log)"
        return AdaptiveLearningRateLog(
            config, shared_learn_rate, shared_momentum)
    elif all(k in config.keys() for k in
             ['learning_rate_start', 'learning_rate_end']):
        print "Enabling Adaptive Learning Rate (Linear)"
        return AdaptiveLearningRateLin(
            config, shared_learn_rate, shared_learn_rate)
    else:
        print "Enabling Static Learning Rate"
        return StaticLearningRate(config, shared_learn_rate, shared_momentum)


class MultiLevelPerceptron:
    '''Base Class for the Multi-level Perceptron (defines the interface)

    All MLP-like classes should inherit from this one to be compatible with
    the batch-processing classes.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def build_network(self):
        '''Compiles the network.
        '''
        pass

    @abc.abstractmethod
    def train(self, x_values, y_values):
        '''Updates the model and calculates RMSE loss for the given X, Y data.
        '''
        pass

    @abc.abstractmethod
    def validate(self, x_values, y_values):
        '''Returns the RMSE loss for the given model and a set of X, Y data.
        '''
        pass

    @abc.abstractmethod
    def predict(self, x_values):
        '''Predicts Y from the model and the given X data.
        '''
        pass

    @abc.abstractmethod
    def get_state(self):
        '''Get the state of the model and return it.
        '''
        pass

    @abc.abstractmethod
    def set_state(self, state):
        '''Set the state of the model from the argument.
        '''
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class ConvolutionalMLP(MultiLevelPerceptron):
    '''Convolutional MLP Definition.
    '''
    __LINEARITY_TYPES = {
        'rectify': lasagne.nonlinearities.rectify,
        'tanh': lasagne.nonlinearities.tanh,
        'leaky_rectify': lasagne.nonlinearities.leaky_rectify
    }

    # This isn't great, but it's a one-off
    def __init__(self, config, input_shape, output_width):
        '''Creates a Convolutional Multi-level MultiLevel

        '''
        super(ConvolutionalMLP, self).__init__()
        self.__config = config
        self.__input_shape = (config['batchsize'],) + input_shape
        self.__output_width = output_width

        self.__input_var = T.tensor4('input')
        self.__target_var = T.matrix('target')

        self.__network = None
        self._create_network()
        self.__train_fn = None
        self.__validate_fn = None
        self.__adaptive_update = None

    def _create_network(self):
        if self.__network is not None:
            raise AssertionError('Cannot call BuildNetwork more than once')

        # pylint: disable=redefined-variable-type
        nonlinearity = self.__LINEARITY_TYPES[self.__config['nonlinearity']]

        # Input Layer
        lyr = lasagne.layers.InputLayer(self.__input_shape, self.__input_var,
                                        name='input')
        if 'input_drop_rate' in self.__config:
            lyr = lasagne.layers.DropoutLayer(
                lyr,
                p=self.__config['input_drop_rate'],
                name='input_dropout')

        # 2d Convolutional Layers
        if 'conv' in self.__config:
            i = 0
            for conv in self.__config['conv']:
                lyr = lasagne.layers.Conv2DLayer(
                    lyr,
                    num_filters=conv['filter_count'],
                    filter_size=tuple(conv['filter_size']),
                    nonlinearity=nonlinearity,
                    name=('conv_2d_%d' % i))
                lyr = lasagne.layers.MaxPool2DLayer(
                    lyr,
                    pool_size=tuple(conv['pooling_size']),
                    name=('pool_2d_%d' % i))
                i += 1

        # Hidden Layers
        if 'hidden' in self.__config:
            i = 0
            for hidden in self.__config['hidden']:
                lyr = lasagne.layers.DenseLayer(
                    lyr,
                    num_units=hidden['width'],
                    nonlinearity=nonlinearity,
                    W=lasagne.init.GlorotUniform(),
                    name=('dense_%d' % i))

                if 'dropout' in hidden and hidden['dropout'] != 0:
                    lyr = lasagne.layers.DropoutLayer(
                        lyr,
                        p=hidden['dropout'],
                        name=('dropout_%d' % i))
                i += 1

        if 'output_nonlinearity' in self.__config:
            output_nonlinearity = self.__LINEARITY_TYPES[
                self.__config['output_nonlinearity']]
        else:
            output_nonlinearity = None

        # Output Layer
        self.__network = lasagne.layers.DenseLayer(
            lyr, num_units=self.__output_width,
            nonlinearity=output_nonlinearity,
            name='output')

    def __update_nesterov_params(self, epoch, num_epochs):
        self.__learning_rate.set_value(1e-9)
        print self.__learning_rate.get_value()

    def epoch_done_tasks(self, epoch, num_epochs):
        self.__adaptive_update.update(epoch, num_epochs)

    def build_network(self):
        # The output of the entire network is the prediction, define loss to be
        # the RMSE of the predicted values.
        prediction = lasagne.layers.get_output(self.__network)
        loss = lasagne.objectives.squared_error(prediction, self.__target_var)
        loss = lasagne.objectives.aggregate(loss, mode='mean')

        # Grab the parameters and define the update scheme.
        params = lasagne.layers.get_all_params(self.__network, trainable=True)
        learning_rate = theano.shared(np.float32(0.))
        momentum = theano.shared(np.float32(0.))
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

        self.__adaptive_update = LearningRateFactory(
            self.__config, learning_rate, momentum)

        # For testing the output, use the deterministic parts of the output
        # (this turns off noise-sources, if we had any and possibly does things
        # related to dropout layers, etc.).  Again, loss is defined using rmse.
        test_prediction = lasagne.layers.get_output(
            self.__network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(
            test_prediction, self.__target_var)

        # test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')

        # Create the training and validation functions that we'll use to train
        # the model and validate the results.
        self.__train_fn = theano.function(
            [self.__input_var, self.__target_var], loss, updates=updates)
        self.__validate_fn = theano.function(
            [self.__input_var, self.__target_var], test_loss)

    def predict(self, x_values):
        return(lasagne.layers.get_output(
            self.__network, x_values, deterministic=True).eval())

    def train(self, x_values, y_values):
        return self.__train_fn(x_values, y_values)

    def validate(self, x_values, y_values):
        return self.__validate_fn(x_values, y_values)

    def get_state(self):
        return lasagne.layers.get_all_param_values(self.__network)

    def set_state(self, state):
        lasagne.layers.set_all_param_values(self.__network, state)

    def __str__(self):
        ret_string = "Convoluational MLP:\n%s\n" % (
            json.dumps(self.__config, sort_keys=True))

        lyrs = lasagne.layers.get_all_layers(self.__network)
        ret_string += "  Layer Shapes:\n"
        for lyr in lyrs:
            ret_string += "\t%20s = %s\n" % (
                lyr.name, lasagne.layers.get_output_shape(lyr))
        return ret_string
