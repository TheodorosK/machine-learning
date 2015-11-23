#!/usr/bin/env python
import abc

import numpy as np
import pandas
from sklearn import preprocessing

class DataReader:
	__metaclass__ = abc.ABCMeta
	
	@abc.abstractmethod
	def Read(self): pass

class CSVReader(DataReader):
	def __init__(self, trainfile, testfile):
		self.__trainfile = trainfile
		self.__testfile = testfile

	def _Read(self, filename):
		df = pandas.io.parsers.read_csv(filename)
		Y = df['y']
		X = df[filter(lambda c: c != 'y', df.columns.values)]
		return((Y, X))

	def Read(self):
		train_Y, train_X = self._Read(self.__trainfile)
		test_Y,  test_X  = self._Read(self.__testfile)
		# Encode Categorical Values
		enc = preprocessing.LabelEncoder()
		enc.fit(train_Y)
		self.train_Y = enc.transform(train_Y).astype(np.int32)
		self.test_Y = enc.transform(test_Y).astype(np.int32)

		# Encode Floats
		self.train_X = train_X.astype(np.float32)
		self.test_X = test_X.astype(np.float32)
