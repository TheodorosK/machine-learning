#!/usr/bin/env python
'''File IO Classes
'''
import abc
import gzip
import os
import cPickle as pickle

import numpy as np
import pandas as pd


class DataReader:
    '''Defines the interface used to read data from disk.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._x_values = None
        self._y_values = None

    def _set_xy(self, x_values, y_values):
        self._x_values = x_values
        self._y_values = y_values

    @abc.abstractmethod
    def load_file(self):
        '''Loads the file and returns the contents as a dict.

        Returns:
            The following format:
                {'X': <data>, 'Y' = <data>}
        '''
        pass

    def get_data(self):
        '''Returns the data previously loaded into memory.
        '''
        return {'X': self._x_values, 'Y': self._y_values}


class FaceReader(DataReader):
    '''Reads the facial keypoint training data and caches using pickler.
    '''
    def __init__(self, filename, picklefile, fast_nrows=None):
        super(FaceReader, self).__init__()
        self.__filename = filename
        self.__picklefile = picklefile
        self.__fast_nrows = fast_nrows

    @staticmethod
    def __read_csv_file(filename, nrows):
        data = pd.read_csv(
            filename, sep=r'\s|,', engine='python',
            header=1, index_col=False, nrows=nrows).values
        x_values = data[:, 30:]
        y_values = data[:, 0:30]
        return (x_values, y_values)

    @staticmethod
    def __reshape_data(x_values):
        return((np.asarray(x_values, dtype='float64') / 255.).reshape(
            len(x_values), 1, 96, 96))

    def load_file(self):
        if self.__fast_nrows is not None:
            print "Using Fast-Path, CSV Load"
            x_values, y_values = FaceReader.__read_csv_file(
                self.__filename, self.__fast_nrows)
            self._set_xy(FaceReader.__reshape_data(x_values), y_values)
            return

        if not os.path.exists(self.__picklefile):
            print "Pickle Doesn't Exist, Loading CSV"
            x_values, y_values = FaceReader.__read_csv_file(
                self.__filename, self.__fast_nrows)
            print "Creating Pickle File"
            pickle_fd = gzip.open(self.__picklefile, 'wb')
            pickler = pickle.Pickler(pickle_fd)
            pickler.dump(x_values)
            pickler.dump(y_values)
            pickle_fd.close()
            assert os.path.exists(self.__picklefile)

        print "Loading Pickle File"
        pickle_fd = gzip.open(self.__picklefile, 'rb')
        unpickler = pickle.Unpickler(pickle_fd)
        x_values = unpickler.load()
        y_values = unpickler.load()
        pickle_fd.close()

        self._set_xy(FaceReader.__reshape_data(x_values), y_values)
        return self.get_data()
