#!/usr/bin/env python

import lasagne

import fileio
import perceptron

fr = fileio.FaceReader("../data/training.csv", "../data/training.pkl.gz", fast_nrows=10)
fr.Read()

mlp = perceptron.ConvolutionalMLP(
	(10, 1, 96, 96), # input shape
	0.2, # input drop-rate
	[96*96, 96*96], # hidden_layer_widths,
	[0.5, 0.5], # hidden_drop_rate,
	lasagne.nonlinearities.rectify, # hidden_layer_nonlinearity,
	30) # output_width
print(mlp)
mlp.BuildNetwork()

import code
code.interact(local=locals())