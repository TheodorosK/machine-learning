#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import code
import time

import lasagne
import numpy as np

import batch
import fileio
import partition
import perceptron


def main():
    '''Trains the Model.
    '''
    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(
        "../data/training.csv",
        "../data/training.pkl.gz",
        fast_nrows=110)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    # class FakeData(object):
    #   def __init__(self):
    #       self.X = np.arange(110)+1
    #       self.Y = np.arange(110)+101
    # fr = FakeData()

    #
    # Partition the Dataset
    #
    np.random.seed(0x0FEDFACE)

    start_time = time.time()
    partitioner = partition.Partitioner(
        faces.get_data(), {'train': 54.343, 'validate': 25.657, 'test': 20})
    partitions = partitioner.run()

    print "Partition Took {:.3f}s".format(time.time() - start_time)

    for k in partitions.keys():
        print("%s X.shape=%s, Y.shape=%s" % (
            k, partitions[k]['X'].shape, partitions[k]['Y'].shape))

    # fixme(mdelio) should we pickle here for reproducibility/interruptibiity?

    #
    # Instantiate and Build the Convolutional Multi-Level Perceptron
    #
    batchsize = 20
    start_time = time.time()
    mlp = perceptron.ConvolutionalMLP(
        (batchsize, 1, 96, 96),  # input shape
        0.2,                     # input drop-rate
        [(6, 6), (2, 2)],        # 2D filter sizes for each layer
        [7, 10],                 # 2D filter count at each layer
        [(3, 3), (2, 2)],        # Pooling size at each layer
        [69*69, 69*69],          # hidden_layer_widths
        [0.5, 0.5],              # hidden_drop_rate
        lasagne.nonlinearities.rectify,  # hidden_layer_nonlinearity
        30,                      # output width
        1e-4,                    # learning rate
        0.8                      # momentum
        )
    print mlp
    mlp.build_network()
    print "Building Network Took {:.3f}s".format(time.time() - start_time)

    #
    # Finally, launch the training loop.
    #
    print "Starting training..."
    trainer = batch.BatchedTrainer(mlp, batchsize, partitions)
    trainer.train(30)

    y_pred = trainer.predict_y(partitions['test']['X'])
    print y_pred[0]
    print partitions['test']['Y'][0]

    code.interact(local=locals())

if __name__ == "__main__":
    main()
