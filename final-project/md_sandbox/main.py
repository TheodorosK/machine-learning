#!/usr/bin/env python
import abc
import operator
import time

import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import confusion_matrix

import fileio
import perceptron

# class Batch:
# 	def __init__(self, batchsize, shuffle):
# 		self.batchsize = batchsize
# 		self.shuffle = shuffle

# 	def __accumulate(iterable, func=operator.add):
# 	    'Return running totals'
# 	    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
# 	    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
# 	    it = iter(iterable)
# 	    try:
# 	        total = next(it)
# 	    except StopIteration:
# 	        return
# 	    yield total
# 	    for element in it:
# 	        total = func(total, element)
# 	        yield total
# 	def Iterate(inputs, targets, fn):
# 		__accumulate = 


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

data = fileio.CSVReader("train.csv", "test.csv")
data.Read()

batchsize = 500

mlp = perceptron.SimpleMLP(batchsize, 477, 0.2, [477+6, 477+6], [0.5, 0.5], 
	lasagne.nonlinearities.rectify, 6)
print(mlp)
mlp.BuildNetwork()

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
num_epochs=3
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(data.train_X, data.train_Y, batchsize, shuffle=False):
        inputs, targets = batch
        train_err += mlp.train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(data.test_X, data.test_Y, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = mlp.val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(data.test_X, data.test_Y, batchsize, shuffle=False):
    inputs, targets = batch
    err, acc = mlp.val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

# Predict the first 500 (batchsize)
y_pred = mlp.Predict(data.test_X[0:batchsize])
print("OOS Confusion Matrix (first %d):" % batchsize)
print(confusion_matrix(data.test_Y[0:batchsize], y_pred))

# Drop to a console for further study.
import code
code.interact(local=locals())