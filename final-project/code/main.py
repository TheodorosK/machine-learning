#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import code
import datetime
import json
import optparse
import os
import subprocess
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


def get_git_revision_hash():
    '''Returns the current git revision as a string
    '''
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


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
    print "Tee done; subsequent prints appear on terminal and console_log.txt"

    #
    # Log the current git revision/cmdline for reproducibility
    #
    print "git rev = %s" % get_git_revision_hash()
    print "cmdline = '%s'" % " ".join(sys.argv)

    # Load the nnet_config
    nnet_config = load_nnet_config(config_file_path, options)

    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(options.faces_csv, options.faces_pickled,
                              fast_nrows=options.num_rows)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    #
    # Map/Drop NaNs
    #
    raw_data = faces.get_data()
    if options.drop_nans:
        to_keep = ~(np.isnan(raw_data['Y']).any(1))
        raw_data['X'] = raw_data['X'][to_keep]
        raw_data['Y'] = raw_data['Y'][to_keep]
        print "Dropping samples with NaNs: {:3.1f}% dropped".format(
            float(sum(~to_keep))/float(len(to_keep))*100.)
    else:
        to_replace = np.isnan(raw_data['Y'])
        raw_data['Y'][to_replace] = options.nan_cardinal
        print "Replaced NaNs with cardinal value of %d [%3.1f%% of data]" % (
            options.nan_cardinal,
            float(np.sum(to_replace))/float(to_replace.size)*100.)

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
        '-o', '--output_dir', dest='run_data_path', type="string",
        metavar="PATH",
        default=datetime.datetime.now().strftime('run_%Y-%m-%d__%H_%M_%S'),
        help=("directory to place run information and state "
              "[default: run_<TIMESTAMP>]"))
    parser.add_option(
        '-e', '--epochs', dest='num_epochs', type="int", metavar="EPOCHS",
        default=100,
        help="number of epochs to train against [default: %default]")
    parser.add_option(
        '-i', '--interval', dest='save_state_interval', type="int",
        metavar="EPOCHS",
        default=10,
        help=("how often (in epochs) to save internal model state"
              "[default: %default]"))
    parser.add_option(
        '-b', '--batchsize', dest='batchsize', type="int", metavar="ROWS",
        default=None,
        help="override the batchsize specified in config_file")
    parser.add_option(
        '-c', '--config_file', dest='config_file', type='string',
        metavar="FILE", default="configs/default.cfg",
        help="neural network configuration file [default: %default]")

    data_group = optparse.OptionGroup(
        parser, "Data Options", "Options for controlling the input data.")
    data_group.add_option(
        '--faces_csv', dest='faces_csv', type='string',
        metavar="PATH", default=os.path.abspath("../data/training.csv"),
        help="path to the faces CSV file [default: %default]")
    data_group.add_option(
        '--faces_pickle', dest='faces_pickled', type='string',
        metavar="PATH", default=os.path.abspath("../data/training.pkl.gz"),
        help="path to the faces pickle file [default: %default]")
    data_group.add_option(
        '-n', '--num_rows', dest='num_rows', type="int", metavar="ROWS",
        default=None,
        help="rows from the dataset to use [default: all]")
    data_group.add_option(
        '--drop_nans', dest='drop_nans', action="store_true",
        help=("option to drop target NaNs instead of mapping to cardinal "
              "[default: map]"))
    data_group.add_option(
        '--nan_cardinal', dest='nan_cardinal', type='int', metavar="VALUE",
        default=-1,
        help=("cardinal value to use for target NaNs [default: %default]"))
    parser.add_option_group(data_group)

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