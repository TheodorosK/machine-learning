#!/usr/bin/env python
'''Batch Processing Classes.
'''
import time

import numpy as np


class BatchedTrainer(object):
    '''Performs Batch processing/training using a multi-layer perceptron.
    '''
    def __init__(self, mlp, batchsize, dataset):
        self.__mlp = mlp
        self.__batchsize = batchsize
        self.__dataset = dataset

    @staticmethod
    def __iterate(data, batchsize, shuffle=False):
        indices = np.arange(len(data['X']))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
            yield indices[start_idx: start_idx + batchsize]

    @staticmethod
    def __run_batches(data, batchsize, func, shuffle=False):
        accum_err = 0
        batch_cnt = 0
        for indices in BatchedTrainer.__iterate(data, batchsize, shuffle):
            accum_err += func(data['X'][indices], data['Y'][indices])
            batch_cnt += 1
        return accum_err / (batch_cnt * batchsize)

    def __train_one_epoch(self):
        train_rmse = BatchedTrainer.__run_batches(
            self.__dataset['train'], self.__batchsize,
            self.__mlp.train, shuffle=True)
        valid_rmse = BatchedTrainer.__run_batches(
            self.__dataset['validate'], self.__batchsize,
            self.__mlp.validate, shuffle=False)
        return (train_rmse, valid_rmse)

    def predict_y(self, x_values):
        '''Predict Y values using the current state of the model and X.
        '''
        assert len(x_values) > 0
        # Calculate the number of extra data-points we need to add to fill up
        # an entire batch
        fill_amt = (np.ceil(float(len(x_values)) / float(self.__batchsize)) *
                    self.__batchsize - len(x_values))
        # Now pad X using the first row of X until we have a multiple of the
        # batch-size.
        x_resized = np.append(
            x_values, np.repeat([x_values[0]], fill_amt, axis=0), axis=0)
        y_resized = None
        for start_idx in range(0, len(x_resized),
                               self.__batchsize):
            y_pred = self.__mlp.predict(
                x_resized[start_idx: start_idx + self.__batchsize])
            # Since we don't technically know the shape of Y ahead of time,
            # this works around this.
            if y_resized is None:
                y_resized = y_pred
            else:
                y_resized = np.append(y_resized, y_pred, axis=0)
        # Only return the non-padded values of Y.
        return y_resized[0:len(x_values)]

    def train(self, num_epochs):
        '''Train the model over the specified epochs.
        '''

        for epoch in range(num_epochs):
            start_time = time.time()
            train_rmse, valid_rmse = self.__train_one_epoch()
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print "  training loss:\t\t{:.6f}".format(train_rmse)
            print "  validation loss:\t\t{:.6f}".format(valid_rmse)

        test_rmse = BatchedTrainer.__run_batches(
            self.__dataset['test'], self.__batchsize,
            self.__mlp.validate, shuffle=False)
        print "Final results:"
        print "  test loss:\t\t\t{:.6f}".format(test_rmse)
