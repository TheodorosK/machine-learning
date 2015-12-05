#!/usr/bin/env python
'''Batch Processing Classes.
'''
import cPickle as pickle
import gzip
import os
import sys
import time

import numpy as np


class TrainingResumer(object):
    '''Helper class to resume training
    '''

    def __init__(self, mlp, status_file, state_file_fmt, interval):
        self.__mlp = mlp
        self.__status_file = status_file
        self.__state_file_fmt = state_file_fmt
        self.__interval = interval

    def resume_training(self):
        '''Resumes the training from the last valid epoch if available or
        begins a new training setup.

        Resturns:
            the starting epoch number (1-indexed).
        '''
        if not os.path.exists(self.__status_file):
            print "No training resume file found, starting from scratch"
            return 1

        #
        # Determine which epoch we can resume from
        #
        epoch_num = None
        with open(self.__status_file, "r") as status_fd:
            epoch_num = [int(x) for x in status_fd.readline().split()][0]
        print "Resuming from epoch %d" % epoch_num

        start_time = time.time()
        state_file = (self.__state_file_fmt % epoch_num)
        assert os.path.exists(state_file)

        #
        # Read the state
        #
        state = None
        with gzip.open(state_file, "rb") as state_fd:
            unpickler = pickle.Unpickler(state_fd)
            state = unpickler.load()
        self.__mlp.set_state(state)

        print "  took {:.3f}s".format(time.time() - start_time)
        return epoch_num + 1

    def end_epoch(self, epoch_num):
        '''Call this when the epoch training has finished
        '''
        if (epoch_num % self.__interval) != 0:
            return

        print "Saving resume file for epoch %d" % epoch_num
        start_time = time.time()
        state_file = (self.__state_file_fmt % epoch_num)

        #
        # Write the state out first
        #
        state = self.__mlp.get_state()
        with gzip.open(state_file, "wb") as state_fd:
            pickler = pickle.Pickler(state_fd, protocol=2)
            pickler.dump(state)

        #
        # Update the status file to mark which epoch was written
        #
        with open(self.__status_file, "w") as status_fd:
            status_fd.write('%d' % epoch_num)

        print "  took {:.3f}s".format(time.time() - start_time)


class BatchedTrainer(object):
    '''Performs Batch processing/training using a multi-layer perceptron.
    '''
    def __init__(self, mlp, batchsize, dataset, logger, resumer):
        self.__mlp = mlp
        self.__batchsize = batchsize
        self.__dataset = dataset
        self.__logger = logger
        self.__resumer = resumer

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
        accum_accuracy = 0
        batch_cnt = 0
        sys.stdout.write('[{}]'.format(len(data['X'])/batchsize))
        for indices in BatchedTrainer.__iterate(data, batchsize, shuffle):
            err, accuracy = func(data['X'][indices], data['Y'][indices])
            accum_err += err
            accum_accuracy += accuracy
            batch_cnt += 1
            sys.stdout.write('.')
            sys.stdout.flush()
        print "done"
        return (accum_err / (batch_cnt * batchsize),
                accum_accuracy / (batch_cnt * batchsize))

    def __train_one_epoch(self):
        sys.stdout.write("  train")
        train_loss, train_rmse = BatchedTrainer.__run_batches(
            self.__dataset['train'], self.__batchsize,
            self.__mlp.train, shuffle=True)

        sys.stdout.write("  valid")
        valid_loss, _ = BatchedTrainer.__run_batches(
            self.__dataset['validate'], self.__batchsize,
            self.__mlp.validate, shuffle=False)

        return (train_loss, train_rmse, np.mean(valid_loss, axis=0))

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
        for epoch in range(self.__resumer.resume_training(), num_epochs+1):
            start_time = time.time()
            print "Epoch {} of {}".format(epoch, num_epochs)
            train_loss, train_rmse, valid_rmse = self.__train_one_epoch()
            self.__logger.log(
                np.concatenate(([train_loss, train_rmse], valid_rmse)), epoch)
            print "    took {:.3f}s".format(time.time() - start_time)
            print "  training loss:\t\t{:.6f}".format(train_loss)
            print "  training rmse:\t\t{:.6f}".format(train_rmse)
            print "  validation rmse:\t\t{:.6f}".format(np.mean(valid_rmse))

            self.__mlp.epoch_done_tasks(epoch-1, num_epochs)
            self.__resumer.end_epoch(epoch)
