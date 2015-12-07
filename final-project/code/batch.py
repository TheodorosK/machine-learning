#!/usr/bin/env python
'''Batch Processing Classes.
'''
import cPickle as pickle
import gzip
import os
import sys
import time

import numpy as np

import preprocess


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

        for name in self.__dataset.keys():
            self.__dataset[name]['Y'] = np.concatenate(
                (self.__dataset[name]['Y'], self.__dataset[name]['Missing']),
                axis=1)
            del self.__dataset[name]['Missing']

    @staticmethod
    def __iterate(data, batchsize, shuffle=False):
        indices = np.arange(len(data['X']))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
            yield indices[start_idx: start_idx + batchsize]

    def __run_batches(self, name, func, shuffle=False):
        # Initialize some accumulator variables
        accum_loss = 0
        accum_rmse = 0
        accum_bin_loss = 0
        batch_cnt = 0

        # Write out the number of batches that will be run for the user.
        sys.stdout.write('[{}]'.format(len(
            self.__dataset[name]['X']) / self.__batchsize))
        for indices in BatchedTrainer.__iterate(
                self.__dataset[name], self.__batchsize, shuffle):

            # Run the supplied function and accumulate the error
            loss, rmse, bin_loss = func(
                self.__dataset[name]['X'][indices],
                self.__dataset[name]['Y'][indices])
            accum_loss += loss
            accum_rmse += rmse
            accum_bin_loss += bin_loss
            batch_cnt += 1

            # Write out the breadcrumb for the user.
            sys.stdout.write('.')
            sys.stdout.flush()

        # Part of the breadcrumb
        print "done"

        # The accumulated erorr needs to be scaled down by the number of
        # batches and the batch-size to be comparable between the training
        # and validation set (since they're different sized)
        return (accum_loss / (batch_cnt * self.__batchsize),
                accum_rmse / (batch_cnt * self.__batchsize),
                accum_bin_loss / (batch_cnt * self.__batchsize))

    def __train_one_epoch(self):
        sys.stdout.write("  train")
        train_loss, train_rmse, train_bin_loss = self.__run_batches(
            'train', self.__mlp.train, shuffle=True)

        sys.stdout.write("  valid")
        _, valid_rmse, valid_bin_loss = self.__run_batches(
            'validate', self.__mlp.validate, shuffle=False)

        return (train_loss,
                np.mean(train_rmse, axis=0),
                np.mean(train_bin_loss, axis=0),
                np.mean(valid_rmse, axis=0),
                np.mean(valid_bin_loss, axis=0))

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
            (train_loss, train_rmse, train_bin_loss,
                valid_rmse, valid_bin_loss) = self.__train_one_epoch()

            self.__logger.log(
                np.concatenate(([train_loss, np.mean(train_rmse),
                                 np.mean(train_bin_loss)],
                                valid_rmse, valid_bin_loss)),
                epoch)
            print "    took {:.3f}s".format(time.time() - start_time)
            print "  training loss:\t\t{:.6f}".format(train_loss)
            print "  training rmse:\t\t{:.6f}".format(
                np.mean(train_rmse))
            print "  training bin-loss:\t\t{:.6f}".format(
                np.mean(train_bin_loss))
            print "  validation rmse:\t\t{:.6f}".format(
                np.mean(valid_rmse))
            print "  validation bin-loss:\t\t{:.6f}".format(
                np.mean(valid_bin_loss))

            self.__mlp.epoch_done_tasks(epoch-1, num_epochs)
            self.__resumer.end_epoch(epoch)
