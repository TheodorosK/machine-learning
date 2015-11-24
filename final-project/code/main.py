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

batchsize = 100

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

fr = fileio.FaceReader("../data/training.csv", "../data/training.pkl.gz", fast_nrows=200)
fr.Read()

X, Y = fr.GetData()

train_X = X[0:100]
train_Y = Y[0:100]

test_X = X[100:200]
test_Y = Y[100:200]

mlp = perceptron.ConvolutionalMLP(
	(batchsize, 1, 96, 96), # input shape
	0.2, # input drop-rate
	[96*96, 96*96], # hidden_layer_widths,
	[0.5, 0.5], # hidden_drop_rate,
	lasagne.nonlinearities.rectify, # hidden_layer_nonlinearity,
	30) # output_width
print(mlp)
mlp.BuildNetwork()

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
num_epochs=10
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(train_X, train_Y, batchsize, shuffle=False):
    	print(train_batches)
        inputs, targets = batch
        train_err += mlp.train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_batches = 0
    for batch in iterate_minibatches(test_X, test_Y, batchsize, shuffle=False):
    	print(val_batches)
        inputs, targets = batch
        err = mlp.val_fn(inputs, targets)
        val_err += err
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  training loss:\t\t{:.6f}".format(train_err))
    print("  validation loss:\t\t{:.6f}".format(val_err))

# After training, we compute and print the test error:
test_err = 0
test_batches = 0
for batch in iterate_minibatches(test_X, test_Y, batchsize, shuffle=False):
    inputs, targets = batch
    err = mlp.val_fn(inputs, targets)
    test_err += err
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

# Predict the first 500 (batchsize)
y_pred = mlp.Predict(test_X[0:batchsize])

import code
code.interact(local=locals())