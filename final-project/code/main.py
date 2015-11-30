#!/usr/bin/env python
import operator
import time

import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import confusion_matrix

import fileio
import perceptron

batchsize = 20

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def PartitionDataset(dataset, train, test, validate, shuffle=True):
	"""Partitions a Dataset into a 'train', 'test', and 'validate'

	Args:
		datatset: A dictionary containing an X/Y that can be partitioned.
		train: percentage (0-100) of data-points to put in training partition.
		test: percentage of data-points to put in test partition.
		validate: percentage of data-points to put in validate partition.
		shuffle: an optional variable that shuffles the data-sets before
			partitioning them.
	Returns:
		A dict containing named partitioned data returned in a dict.  For
			example:

			{'train': {'X': <dataset.Xs>, 'Y': <dataset.Ys>},
			 'test': {'X': <dataset.Xs>, 'Y': <dataset.Ys>},
			 'validate': {'X': <dataset.Xs>, 'Y': <dataset.Ys>}}
	"""
	# The sizes should add up to 100%.
	assert((train + test + validate) == 100)

	num_samples = len(dataset['Y'])
	# Create a list of the indices to use when slicing the dataset and
	# optionally shuffle it.
	indices = range(num_samples)
	if shuffle:
		np.random.shuffle(indices)

	labels = ["train", "test", "validate"]
	boundaries = [train/100., (train + test)/100., 1]
	start = 0
	partitioned = {}
	for i in range(len(labels)):
		# Find the stopping index.
		stop = np.floor(boundaries[i] * num_samples)

		# Find the index range
		idx = indices[int(start):int(stop)]

		# Create the partition dictionary.
		part = {'X': dataset['X'][idx], 'Y': dataset['Y'][idx]}
		partitioned[labels[i]] = part

		# The new start is the the old stop.
		start = stop
	return(partitioned)

#
# Load the Dataset
#
start_time = time.time()
fr = fileio.FaceReader("../data/training.csv", "../data/training.pkl.gz", fast_nrows=100)
fr.Read()
print("Read Took {:.3f}s".format(time.time() - start_time))

#
# Partition the Dataset
#
np.random.seed(0x0FEDFACE)

start_time = time.time()
partitions = PartitionDataset({'X': fr.X, 'Y': fr.Y}, 60, 20, 20)
print("Partition Took {:.3f}s".format(time.time() - start_time))

for k in partitions.keys():
	print("%s X.shape=%s, Y.shape=%s" % (
		k, partitions[k]['X'].shape, partitions[k]['Y'].shape))

# TODO(mdelio) should we pickle here for reproducibility/interruptibiity?

#
# Instantiate and Build the Convolutional Multi-Level Perceptron
#
mlp = perceptron.ConvolutionalMLP(
	(batchsize, 1, 96, 96), # input shape
	0.2, # input drop-rate
	[96*96, 96*96], # hidden_layer_widths,
	[0.5, 0.5], # hidden_drop_rate,
	lasagne.nonlinearities.rectify, # hidden_layer_nonlinearity,
	30) # output_width
print(mlp)
start_time = time.time()
mlp.BuildNetwork()
print("Building Network Took {:.3f}s".format(time.time() - start_time))

#
# Finally, launch the training loop.
#
print("Starting training...")
# We iterate over epochs:
num_epochs=20
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(
		partitions['train']['X'],
		partitions['train']['Y'], batchsize, shuffle=False):
    	print(train_batches)
        inputs, targets = batch
        train_err += mlp.train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_batches = 0
    for batch in iterate_minibatches(
		partitions['validate']['X'],
		partitions['validate']['Y'], batchsize, shuffle=False):
    	print(val_batches)
        inputs, targets = batch
        err = mlp.val_fn(inputs, targets)
        val_err += err
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

# After training, we compute and print the test error:
test_err = 0
test_batches = 0
for batch in iterate_minibatches(
	partitions['test']['X'],
	partitions['test']['Y'], batchsize, shuffle=False):
    inputs, targets = batch
    err = mlp.val_fn(inputs, targets)
    test_err += err
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

# 	Predict the first 500 (batchsize)
y_pred = mlp.Predict(test_X[0:batchsize])

import code
code.interact(local=locals())