#!/usr/bin/env python
import abc
import gzip
import os
import cPickle as pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

class DataReader:
	__metaclass__ = abc.ABCMeta
	
	@abc.abstractmethod
	def Read(self): pass

	@abc.abstractmethod
	def GetData(self): pass

# class CSVReader(DataReader):
# 	def __init__(self, trainfile, testfile):
# 		self.__trainfile = trainfile
# 		self.__testfile = testfile

# 	def _Read(self, filename):
# 		df = pd.io.parsers.read_csv(filename)
# 		Y = df['y']
# 		X = df[filter(lambda c: c != 'y', df.columns.values)]
# 		return((Y, X))

# 	def Read(self):
# 		train_Y, train_X = self._Read(self.__trainfile)
# 		test_Y,  test_X  = self._Read(self.__testfile)
# 		# Encode Categorical Values
# 		enc = preprocessing.LabelEncoder()
# 		enc.fit(train_Y)
# 		self.train_Y = enc.transform(train_Y).astype(np.int32)
# 		self.test_Y = enc.transform(test_Y).astype(np.int32)

# 		# Encode Floats
# 		self.train_X = train_X.astype(np.float32)
# 		self.test_X = test_X.astype(np.float32)

class FaceReader(DataReader):
	def __init__(self, filename, picklefile, fast_nrows=None):
		self.__filename = filename
		self.__picklefile = picklefile
		self.__fast_nrows = fast_nrows

	def _ReadCSV(self):
		data = pd.read_csv(self.__filename, sep='\s|,', engine='python', 
			header=1, index_col=False, nrows=self.__fast_nrows).values
		X = data[:, 30:]
		Y = data[:, 0:30]
		return(X, Y)

	def _Shape(self, X):
		return((np.asarray(X, dtype='float64') / 255.).reshape(
			len(X), 1, 96, 96))

	def Read(self):
		if (self.__fast_nrows is not None):
			print("Using Fast-Path, CSV Load")
			X, Y = self._ReadCSV()
			self.X = self._Shape(X)
			self.Y = Y
			return

		if (not os.path.exists(self.__picklefile)):
			print("Pickle Doesn't Exist, Loading CSV")
			X, Y = self._ReadCSV()
			print("Creating Pickle File")
			f = gzip.open(self.__picklefile, 'wb')
			p = pickle.Pickler(f)
			p.dump(X)
			p.dump(Y)
			f.close()
			assert(os.path.exists(self.__picklefile))

		print("Loading Pickle File")
		f = gzip.open(self.__picklefile, 'rb')
		p = pickle.Unpickler(f)
		X = p.load()
		Y = p.load()
		f.close()

		self.X = self._Shape(X)
		self.Y = Y

	def GetData(self):
		return(self.X, self.Y)
