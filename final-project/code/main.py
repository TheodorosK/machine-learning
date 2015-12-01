#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import code
import datetime
import optparse
import os
import sys
import time

import lasagne
import numpy as np

import batch
import data_logger
import fileio
import partition
import perceptron


class Tee(object):
    '''Tees file descriptors so that writes are made everywhere simultaneously.
    '''
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        '''Writes to all file descriptors.
        '''
        # pylint: disable=invalid-name
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        '''Flushes all file descriptors.
        '''
        # pylint: disable=invalid-name
        for f in self.files:
            f.flush()


def main(options):
    '''Trains the Model.
    '''
    #
    # Change to the run data directory
    #
    os.chdir(options.run_data_path)

    # Tee the output to a logfile.
    console_fd = open('console_log.txt', 'a')
    sys.stdout = Tee(sys.stdout, console_fd)

    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(
        "../../data/training.csv",
        "../../data/training.pkl.gz",
        fast_nrows=110)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    # fixme(mdelio) if we drop the nas, we drop a lot of data (2/3 of it).
    raw_data = faces.get_data()
    to_keep = ~(np.isnan(raw_data['Y']).any(1))
    raw_data['X'] = raw_data['X'][to_keep]
    raw_data['Y'] = raw_data['Y'][to_keep]
    print "Dropping samples with NaNs: {:3.1f}% dropped".format(
        float(sum(~to_keep))/float(len(to_keep))*100.)

    #
    # Partition the Dataset
    #
    np.random.seed(0x0FEDFACE)

    start_time = time.time()
    partitioner = partition.Partitioner(
        raw_data, {'train': 60, 'validate': 20, 'test': 20},
        "partition_indices.pkl")
    partitions = partitioner.run()
    print "Partition Took {:.3f}s".format(time.time() - start_time)

    for k in partitions.keys():
        print("%20s X.shape=%s, Y.shape=%s" % (
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
    loss_log = data_logger.CSVEpochLogger(
        "loss_%05d.csv", "loss.csv",
        np.concatenate((['train_loss'], faces.get_labels()['Y'])))
    resumer = batch.TrainingResumer(mlp, "epochs_done.txt",
                                    "state_%05d.pkl.gz", 2)
    trainer = batch.BatchedTrainer(mlp, batchsize, partitions,
                                   loss_log, resumer)
    trainer.train(3)

    y_pred = trainer.predict_y(partitions['test']['X'])
    print y_pred[0]
    print partitions['test']['Y'][0]

    code.interact(local=locals())

#
# Parse Arguments and call main.
#
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option(
        "-d", "--data-dir", dest='run_data_path',
        default=datetime.datetime.now().strftime('run_%Y-%m-%d__%H_%M_%S'))

    options, args = parser.parse_args()

    # Create a directory for our work.
    if os.path.exists(options.run_data_path):
        print "Using existing data directory at %s" % options.run_data_path
    else:
        print "Creating data directory at %s" % options.run_data_path
        os.mkdir(options.run_data_path)

    main(options)
