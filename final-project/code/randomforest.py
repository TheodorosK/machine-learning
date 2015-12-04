import code
import os
import time

import lasagne
import numpy as np
import sklearn.ensemble

import fileio
import partition

faces_csv = os.path.abspath("../data/training.csv")
faces_pickled = os.path.abspath("../data/training.pkl.gz")
num_rows = 1000

start_time = time.time()
faces = fileio.FaceReader(faces_csv, faces_pickled,
                          fast_nrows=500)
faces.load_file()
print "Read Took {:.3f}s".format(time.time() - start_time)

# Convert raw data from float64 to floatX (32/64-bit depending on GPU/CPU)
raw_data = faces.get_data()
raw_data['X'] = lasagne.utils.floatX(raw_data['X']) # (100, 1, 96, 96)
raw_data['Y'] = lasagne.utils.floatX(raw_data['Y']) # (100, 30)

# Flatten X
X = np.ndarray((raw_data['X'].shape[0], 
	raw_data['X'].shape[2]*raw_data['X'].shape[3]))
for i in range(raw_data['X'].shape[0]):
	X[i, :] = np.ndarray.flatten(raw_data['X'][i][0])
raw_data['X'] = X

# Partition data
partitioner = partition.Partitioner(
        raw_data, {'train': 60, 'validate': 40},
        "partition_indices.pkl")
partitions = partitioner.run()
print "Partition Took {:.3f}s".format(time.time() - start_time)

for k in partitions.keys()	:
    print("%20s X.shape=%s, Y.shape=%s" % (
        k, partitions[k]['X'].shape, partitions[k]['Y'].shape))

num_features = raw_data['Y'].shape[1]
err = np.empty((1,num_features))

partitions['train']['X'].shape # (300, 1, 9216)
partitions['train']['Y'].shape # (300, 30)

for i in range(5): #range(partitions['train']['Y'].shape[1]):
	print "*"*80 + "\n"
	print "Feature #" + str(i+1)

	start_time = time.time()
	
	# Drop NaNs
	to_keep = ~(np.isnan(partitions['train']['Y']).any(1))
	X = partitions['train']['X'][to_keep]
	Y = partitions['train']['Y'][to_keep]
	print "Dropping samples with NaNs: {:3.1f}% dropped".format(float(sum(~to_keep))/float(len(to_keep))*100.)

	rf = sklearn.ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
	rf = rf.fit(X, Y[:,i])
	yhat = rf.predict(partitions['validate']['X'])
	err[0][i] = np.sqrt(np.nanmean(np.power(yhat - partitions['validate']['Y'][:,i], 2)))
	print "RMSE = " + str(err[0][i])
	print "calculation time: {:.3f}s".format(time.time() - start_time)

print "Done"

# Drop into a console so that we do anything additional we need.
code.interact(local=locals())

# rf = sklearn.ensemble.RandomForestRegressor(n_jobs=-1)
	# rf = rf.fit(partitions['train']['X'], 
	# 	partitions['train']['Y'][:,i])
	# yhat = rf.predict(partitions['validate']['X'])
	# err[0][i] = np.sqrt(np.mean(np.power(yhat - partitions['validate']['Y'][:,i], 2)))
	

# X = np.ndarray((raw_data['X'].shape[0], 
# 	raw_data['X'].shape[2]*raw_data['X'].shape[3]))
# for i in range(raw_data['X'].shape[0]):
# 	X[i, :] = np.ndarray.flatten(raw_data['X'][i][0])
# Y = raw_data['Y'][:,0]

# testX = X[0:80,:]
# valX = X[80:100,:]
# testY = Y[0:80]
# valY = Y[80:100]

# rf = sklearn.ensemble.RandomForestRegressor(n_estimators=10, 
# 	criterion="mse", n_jobs=1, verbose=1)

# rf = rf.fit(X=testX, y=testY)

# out = rf.predict(valX)

# err = np.sqrt(np.mean(np.power(out-valY,2)))
# print err

