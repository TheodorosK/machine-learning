#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import code
import datetime
import json
import optparse
import os
import sys
import time

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


def load_nnet_config(filename, options):
    '''Load the neural network config file and address overrides
    '''
    print "Loading NNET configuration from %s" % filename
    assert os.path.exists(filename)
    with open(filename) as json_fp:
        nnet_config = json.load(json_fp)

    # Handle command-line overrides that affect the config
    if options.batchsize is not None:
        nnet_config['batchsize'] = options.batchsize

    return nnet_config


def real_main(options):
    '''This loads the data, manipulates it and trains the model, it's the glue
    that does all of the work.
    '''
    # pylint: disable=too-many-locals
    # Yes we have a lot of variables, but this is the main function.

    # Make the config file path absolute (so that a chdir won't affect it)
    config_file_path = os.path.abspath(options.config_file)

    #
    # Change to the run data directory
    #
    os.chdir(options.run_data_path)

    # Tee the output to a logfile.
    console_fd = open('console_log.txt', 'a')
    sys.stdout = Tee(sys.stdout, console_fd)

    # Load the nnet_config
    nnet_config = load_nnet_config(config_file_path, options)

    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(
        "../../data/training.csv",
        "../../data/training.pkl.gz",
        fast_nrows=options.num_rows)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    # fixme(mdelio) if we drop the NaNs, we drop a lot of data (2/3 of it).
    # if we don't though, our loss function is invalid, which means our model
    # cannot train.
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

    #
    # Run any transformations on the training dataset here.
    #

    #
    # Instantiate and Build the Convolutional Multi-Level Perceptron
    #
    # batchsize = options.batchsize
    start_time = time.time()
    mlp = perceptron.ConvolutionalMLP(nnet_config, (1, 96, 96), 30)
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
    resumer = batch.TrainingResumer(
        mlp, "epochs_done.txt", "state_%05d.pkl.gz",
        options.save_state_interval)
    trainer = batch.BatchedTrainer(mlp, nnet_config['batchsize'], partitions,
                                   loss_log, resumer)
    trainer.train(options.num_epochs)

    #
    # Run the final predict_y to see the output against the actual one.
    # This is more of a sanity check for us.
    #
    y_pred = trainer.predict_y(partitions['test']['X'])
    print y_pred[0]
    print partitions['test']['Y'][0]

    # Drop into a console so that we do anything additional we need.
    code.interact(local=locals())


def main():
    '''Parse Arguments and call real_main
    '''
    #
    # Create list of options and parse them.
    #
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--data_dir', dest='run_data_path', type="string",
        metavar="PATH",
        default=datetime.datetime.now().strftime('run_%Y-%m-%d__%H_%M_%S'),
        help="directory to place run information and state")
    parser.add_option(
        '-e', '--epochs', dest='num_epochs', type="int", metavar="EPOCHS",
        default=100,
        help="number of epochs to train against")
    parser.add_option(
        '-i', '--interval', dest='save_state_interval', type="int",
        metavar="EPOCHS",
        default=10,
        help="how often (in epochs) to save the internal state of the model")
    parser.add_option(
        '-n', '--num_rows', dest='num_rows', type="int", metavar="ROWS",
        default=None,
        help="how many rows from the dataset to load (leave blank for all)")
    parser.add_option(
        '-b', '--batchsize', dest='batchsize', type="int", metavar="ROWS",
        default=None,
        help="override the batchsize specified in config_file")
    parser.add_option(
        '-c', '--config_file', dest='config_file', type='string',
        metavar="FILE", default="configs/default.cfg",
        help="neural network configuration file")

    options, _ = parser.parse_args()

    # Create a directory for our work.
    if os.path.exists(options.run_data_path):
        print "Using existing data directory at %s" % options.run_data_path
    else:
        print "Creating data directory at %s" % options.run_data_path
        os.mkdir(options.run_data_path)

    real_main(options)


if __name__ == "__main__":
    main()
