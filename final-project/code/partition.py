#!/usr/bin/env python 

import numpy as np

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
