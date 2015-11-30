#!/usr/bin/env python
import time

import numpy as np

class BatchedTrainer(object):
	def __init__(self, mlp, batchsize, dataset, num_epochs):
		self.__mlp = mlp
		self.__batchsize = batchsize
		self.__dataset = dataset
		self.__num_epochs = num_epochs

	@staticmethod
	def __Iterate(data, batchsize, shuffle=False):
		indices = np.arange(len(data['X']))
		if shuffle:
			np.random.shuffle(indices)
		for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
			yield(indices[start_idx : start_idx + batchsize])
	
	@staticmethod
	def __RunBatches(data, batchsize, fn, shuffle=False):
		accum_err = 0
		batch_cnt = 0
		for indices in BatchedTrainer.__Iterate(data, batchsize, shuffle):
			accum_err += fn(data['X'][indices], data['Y'][indices])
			batch_cnt += 1
		return(accum_err / (batch_cnt * batchsize))

	def __RunOneEpoch(self, epoch):
		start_time = time.time()
		
		train_rmse = BatchedTrainer.__RunBatches(
			self.__dataset['train'], self.__batchsize, 
			self.__mlp.Train, shuffle=True)
		valid_rmse = BatchedTrainer.__RunBatches(
			self.__dataset['validate'], self.__batchsize, 
			self.__mlp.Validate, shuffle=False)

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, self.__num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_rmse))
		print("  validation loss:\t\t{:.6f}".format(valid_rmse))

	def Predict(self, X):
		'''Predict Y values using the current state of the model and X.
		'''
		assert(len(X) > 0)
		# Calculate the number of extra data-points we need to add to fill up 
		# an entire batch
		fill_amt = (np.ceil(float(len(X)) / float(self.__batchsize)) *
			self.__batchsize - len(X))
		# Now pad X using the first row of X until we have a multiple of the 
		# batch-size.
		new_X = np.append(X, np.repeat([X[0]], fill_amt, axis=0), axis=0)
		new_Y = None
		for start_idx in range(0, len(new_X), self.__batchsize):
			Y = self.__mlp.Predict(new_X[start_idx : start_idx + self.__batchsize])
			# Since we don't technically know the shape of Y ahead of time,
			# this works around this.
			if new_Y is None:
				new_Y = Y
			else:
				new_Y = np.append(new_Y, Y, axis=0)
		# Only return the non-padded values of Y.
		return(new_Y[0:len(X)])

	def Train(self):
		for epoch in range(self.__num_epochs):
			self.__RunOneEpoch(epoch)

		test_rmse = BatchedTrainer.__RunBatches(
			self.__dataset['test'], self.__batchsize, 
			self.__mlp.Validate, shuffle=False)
		print("Final results:")
		print("  test loss:\t\t\t{:.6f}".format(test_rmse))
