#!/usr/bin/env python
import abc

import numpy as np
import lasagne
import theano
import theano.tensor as T

class MultiLevelPerceptron:
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def BuildNetwork(self):
		'''Compiles the network.
		'''
		pass

	@abc.abstractmethod
	def Train(self, X, Y):
		'''Updates the model and calculates RMSE loss for the given X, Y data.
		'''
		pass

	@abc.abstractmethod
	def Validate(self, X, Y):
		'''Returns the RMSE loss for the given model and a set of X, Y data.
		'''
		pass

	@abc.abstractmethod
	def Predict(self, X):
		'''Predicts Y from the model and the given X data.
		'''
		pass

	@abc.abstractmethod
	def __str__(self): pass

class ConvolutionalMLP(MultiLevelPerceptron):
	def __init__(self, input_shape, input_drop_rate, 
		conv_filter_sizes, conv_filter_count, conv_pooling_size,
		hidden_layer_widths, hidden_drop_rate, hidden_layer_nonlinearity,
		output_width, learning_rate, momentum):
		'''Creates a Convolutional Multi-level MultiLevel

		'''
		self.__input_shape = input_shape
		self.__input_drop_rate = input_drop_rate

		if ((len(conv_filter_sizes) != len(conv_filter_count)) or
			(len(conv_filter_sizes) != len(conv_pooling_size))):
			raise ValueError('conv_filter_sizes, conv_filter_count, and '
				'conv_pooling_size, must be all be the same length')
		self.__conv_filter_sizes = conv_filter_sizes
		self.__conv_filter_count = conv_filter_count
		self.__conv_pooling_size = conv_pooling_size

		if (len(hidden_drop_rate) != len(hidden_layer_widths)):
			raise ValueError('hidden_drop_rate must be same length as '
				'hidden_layer_width')

		self.__hidden_layer_widths = hidden_layer_widths
		self.__hidden_drop_rate = hidden_drop_rate
		self.__hidden_layer_nonlinearity = hidden_layer_nonlinearity
		self.__output_width = output_width

		self.__input_var = T.dtensor4('input')
		self.__target_var = T.dmatrix('target')

		self.__network = None
		self._CreateNetwork()
		self.__train_fn = None
		self.__validate_fn = None
		self.__learning_rate = learning_rate
		self.__momentum = momentum

	def _CreateNetwork(self):
		if (self.__network is not None):
			raise AssertionError('Cannot call BuildNetwork more than once')

		# Input Layer
		lyr = lasagne.layers.InputLayer(self.__input_shape, self.__input_var,
			name='input')
		if (self.__input_drop_rate != 0):
			lyr = lasagne.layers.DropoutLayer(lyr, 
				p=self.__input_drop_rate,
				name='input_dropout')

		# 2d Convolutional Layers
		for i in range(len(self.__conv_filter_sizes)):
			lyr = lasagne.layers.Conv2DLayer(
				lyr, 
				num_filters=self.__conv_filter_count[i], 
				filter_size=self.__conv_filter_sizes[i],
				nonlinearity=self.__hidden_layer_nonlinearity,
				name=('conv_2d_%d' % i))
			lyr = lasagne.layers.MaxPool2DLayer(
				lyr,
				pool_size=self.__conv_pooling_size[i],
				name=('pool_2d_%d' % i))

		# Hidden Layers
		for i in range(len(self.__hidden_layer_widths)):
			lyr = lasagne.layers.DenseLayer(
				lyr, 
				num_units=self.__hidden_layer_widths[i],
				nonlinearity=self.__hidden_layer_nonlinearity,
				W=lasagne.init.GlorotUniform(),
				name=('dense_%d' % i))

			if (self.__hidden_drop_rate[i] != 0):
				lyr = lasagne.layers.DropoutLayer(
					lyr, 
					p=self.__hidden_drop_rate[i],
					name=('dropout_%d' % i))

		# Output Layer
		self.__network = lasagne.layers.DenseLayer(lyr,
			num_units=self.__output_width,
			nonlinearity=self.__hidden_layer_nonlinearity,
			name='output')

	def BuildNetwork(self):
		# The output of the entire network is the prediction, define loss to be
		# the RMSE of the predicted values.
		prediction = lasagne.layers.get_output(self.__network)
		loss = lasagne.objectives.squared_error(prediction, self.__target_var)
		loss = lasagne.objectives.aggregate(loss, mode='mean')

		# Grab the parameters and define the update scheme.
		params = lasagne.layers.get_all_params(self.__network, trainable=True)
		updates = lasagne.updates.nesterov_momentum(loss, params, 
			learning_rate=self.__learning_rate, momentum=self.__momentum)

		# For testing the output, use the deterministic parts of the output
		# (this turns off noise-sources, if we had any and possibly does things
		# related to dropout layers, etc.).  Again, loss is defined as the rmse.
		test_prediction = lasagne.layers.get_output(self.__network,
			deterministic=True)
		test_loss = lasagne.objectives.squared_error(test_prediction,
			self.__target_var)
		test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')

		# Create the training and validation functions that we'll use to train
		# the model and validate the results.
		self.__train_fn = theano.function(
			[self.__input_var, self.__target_var], loss, updates=updates)
		self.__validate_fn = theano.function(
			[self.__input_var, self.__target_var], test_loss)

	def Predict(self, X):
		return(lasagne.layers.get_output(self.__network, X, 
			deterministic=True).eval())

	def Train(self, X, Y):
		return(self.__train_fn(X, Y))

	def Validate(self, X, Y):
		return(self.__validate_fn(X, Y))

	def __str__(self):
				# if (self.network is None):
		str=("Convoluational MLP:\n"
			 "  Parameters:\n"
			 "\tinput_dropout_rate=%s\n"
			 "\tconv_filter_size=%s\n"
			 "\tconv_pooling_size=%s\n"
			 "\thidden_dropout=%s\n" % (
				self.__input_drop_rate, self.__conv_filter_sizes,
				self.__conv_pooling_size, self.__hidden_drop_rate))

		lyrs = lasagne.layers.get_all_layers(self.__network)
		str += "  Layer Shapes:\n"
		for lyr in lyrs:
			str += "\t%20s = %s\n" % (
				lyr.name, lasagne.layers.get_output_shape(lyr))
		return(str)
