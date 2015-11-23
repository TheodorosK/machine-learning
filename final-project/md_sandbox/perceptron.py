#!/usr/bin/env python
import abc

import numpy as np
import lasagne
import theano
import theano.tensor as T

class MultiLevelPerceptron:
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def BuildNetwork(self): pass

	@abc.abstractmethod
	def __str__(self): pass

class SimpleMLP(MultiLevelPerceptron):
	def __init__(self, batchsize, input_width, input_drop_rate,
		hidden_layer_widths, hidden_drop_rate, hidden_layer_nonlinearity,
		output_width):
		self.__batchsize = batchsize
		self.__input_width = input_width
		self.__input_drop_rate = input_drop_rate

		if (len(hidden_drop_rate) != len(hidden_layer_widths)):
			raise ValueError('hidden_drop_rate must be same length as '
				'hidden_layer_width')

		self.__hidden_layer_widths = hidden_layer_widths
		self.__hidden_drop_rate = hidden_drop_rate
		self.__hidden_layer_nonlinearity = hidden_layer_nonlinearity
		self.__output_width = output_width

		self.__input_var = T.matrix('input')
		self.__target_var = T.ivector('target')

		self.network = None
		self.train_fn = None
		self.valid_fn = None

	def BuildNetwork(self):
		if (self.network is not None):
			raise AssertionError('Cannot call BuildNetwork more than once')

		lyr = lasagne.layers.InputLayer(
			(self.__batchsize, self.__input_width), self.__input_var)
		if (self.__input_drop_rate != 0):
			lyr = lasagne.layers.DropoutLayer(lyr, p=self.__input_drop_rate)

		for i in range(len(self.__hidden_layer_widths)):
			lyr = lasagne.layers.DenseLayer(
				lyr, 
				num_units=self.__hidden_layer_widths[i],
				nonlinearity=self.__hidden_layer_nonlinearity,
				W=lasagne.init.GlorotUniform())

			if (self.__hidden_drop_rate[i] != 0):
				lyr = lasagne.layers.DropoutLayer(
					lyr,p=self.__hidden_drop_rate[i])

		self.network = lasagne.layers.DenseLayer(lyr,
			num_units=self.__output_width,
			nonlinearity=lasagne.nonlinearities.softmax)

		self.__BuildTheanoFns()

	def Predict(self, X):
		prediction = lasagne.layers.get_output(self.network, X, deterministic=True).eval()
		return(np.argmax(prediction, 1))

	def __BuildTheanoFns(self):
		prediction = lasagne.layers.get_output(self.network)
		loss = lasagne.objectives.categorical_crossentropy(
			prediction, self.__target_var).mean()

		params = lasagne.layers.get_all_params(self.network)
		updates = lasagne.updates.nesterov_momentum(loss, params, 
			learning_rate=0.01, momentum=0.9)


		# Create a loss expression for validation/testing. The crucial difference
		# here is that we do a deterministic forward pass through the network,
		# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(
			self.network, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(
			test_prediction, self.__target_var).mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.__target_var),
		                  dtype=theano.config.floatX)

		self.prediction = prediction
		self.train_fn = theano.function([self.__input_var, self.__target_var], loss,
			updates=updates)
		self.val_fn = theano.function([self.__input_var, self.__target_var],
			[test_loss, test_acc])

	def __str__(self):
		return("SimpleMLP: input_width=%d (dropout=%.2f), output_width=1 "
			"(%d categories), %d hidden '%s' layers of size=%s (dropout=%s)" % (
				self.__input_width, self.__input_drop_rate, 
				self.__output_width, len(self.__hidden_layer_widths), 
				self.__hidden_layer_nonlinearity.__name__,
				self.__hidden_layer_widths, self.__hidden_drop_rate))
