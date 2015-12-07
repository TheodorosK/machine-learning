#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import argparse
import code
import datetime
import json
import os
import re
import subprocess
import sys
import time

import argcomplete
import lasagne
import numpy as np
import pandas as pd

import batch
import data_logger
import fancy_perceptron
import fileio
import partition
import perceptron
import preprocess


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


def get_version():
    '''Returns the current git revision as a string
    '''
    return "git rev = %s" % (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())


def print_partition_shapes(partitions):
    for k in partitions.keys():
        print("%20s X.shape=%s, Y.shape=%s" % (
            k, partitions[k]['X'].shape, partitions[k]['Y'].shape))


def load_nnet_config(options):
    '''Load the neural network config file and address overrides
    '''
    print "Loading NNET configuration from %s" % options.config_file
    assert os.path.exists(options.config_file)
    with open(options.config_file) as json_fp:
        nnet_config = json.load(json_fp)

    # Handle command-line overrides that affect the config
    if options.batchsize is not None:
        nnet_config['batchsize'] = options.batchsize

    return nnet_config


def select_data(options, feature_cols, partitioned):
    #
    # Select the columns.
    #
    selected = {}
    for name in partitioned.keys():
        selected[name] = {}
        selected[name]['X'] = partitioned[name]['X']
        selected[name]['Y'] = partitioned[name]['Y'][:, feature_cols]
        missing_shape = list(selected[name]['Y'].shape)
        missing_shape[1] /= 2
        selected[name]['Missing'] = np.empty(
            tuple(missing_shape), dtype='float32')

    #
    # Map/Drop NaNs
    #
    replaced = 0
    total = 0
    for name in partitioned.keys():
        to_replace = np.isnan(selected[name]['Y'])
        odd_cols = range(1, len(feature_cols), 2)
        even_cols = range(0, len(feature_cols), 2)
        missing = to_replace[:, odd_cols]
        assert np.all(to_replace[:, odd_cols] == to_replace[:, even_cols])

        col_means = np.nanmean(selected[name]['Y'], axis=0)

        selected[name]['Missing'] = np.float32(missing)
        for i in range(len(feature_cols)):
            selected[name]['Y'][:, i][to_replace[:, i]] = col_means[i]

        assert np.allclose(np.mean(selected[name]['Y'], axis=0), col_means)

        replaced += np.sum(to_replace)
        total += to_replace.size

    print "Replaced NaNs with column mean [%3.1f%% of data]" % (
        float(replaced)/float(total)*100.)
    return selected


def combine_loss_main(options):
    with open(options.feature_group_file) as feat_fd:
        feature_groups = json.load(feat_fd)

    start_time = time.time()
    combine_loss(options, feature_groups)
    print "Combining CSVs took {:.3f}s".format(time.time() - start_time)


def combine_loss(options, feature_groups):
    assert os.getcwd() == options.run_data_path

    loss_dfs = {}

    for feature_idx, (feature_name, feature_cols) in enumerate(sorted(
            feature_groups.items())):
        if not os.path.exists(feature_name):
            print "Could not find directory for %s" % feature_name
            continue
        os.chdir(feature_name)

        with open('loss.csv') as loss_fd:
            df = pd.read_csv(loss_fd, index_col="epoch")

        df.rename(columns={
            "train_loss": "train_loss_" + feature_name,
            "train_rmse": "train_rmse_" + feature_name
        }, inplace=True)

        loss_dfs[feature_name] = df

        os.chdir(options.run_data_path)

    assert os.getcwd() == options.run_data_path

    aggregated_loss = pd.concat(loss_dfs.values(), axis=1)
    with open("loss.csv", 'w') as write_fd:
        aggregated_loss.to_csv(write_fd)


def train_feature(options, selected, nnet_config, feature_info):
    #
    # Instantiate and Build the Convolutional Multi-Level Perceptron
    #
    start_time = time.time()
    if options.amputate:
        print "Chose an Amputated MLP"
        perceptron_type = perceptron.AmputatedMLP
    else:
        print "Chose an Convolutional MLP"
        perceptron_type = fancy_perceptron.FancyPerceptron
    mlp = perceptron_type(
        nnet_config, (1, 96, 96), len(feature_info['cols']))
    print mlp
    mlp.build_network()
    print "Building Network Took {:.3f}s".format(time.time() - start_time)

    #
    # Finally, launch the training loop.
    #

    feat_miss_header = [re.sub(r"_[xy]$", "", f) for
                        f in feature_info['col_labels']]
    feat_miss_header = ["bin_" + f for (i, f) in enumerate(feat_miss_header)
                        if i in range(0, len(feat_miss_header), 2)]
    print feat_miss_header

    print "Starting training..."
    loss_log = data_logger.CSVEpochLogger(
        "loss_%05d.csv", "loss.csv",
        np.concatenate((
            ['train_loss', 'train_coord_rmse', 'train_exist_loss'],
            feature_info['col_labels'], feat_miss_header)))
    resumer = batch.TrainingResumer(
        mlp, "epochs_done.txt", "state_%05d.pkl.gz",
        options.save_state_interval)
    trainer = batch.BatchedTrainer(mlp, nnet_config['batchsize'],
                                   selected, loss_log, resumer)
    trainer.train(options.num_epochs)

    if options.amputate or options.predict:
        def write_pred(data, filename, header):
            data_frame = pd.DataFrame(data)
            with open(filename, 'w') as file_desc:
                data_frame.to_csv(file_desc, header=header, index=False)

        missing = ["pr_missing_"+re.sub(r"_[xy]$", "", f) for (i, f) in
                   enumerate(feature_info['col_labels']) if i in
                   range(0, len(feature_info['col_labels']), 2)]
        header = np.concatenate((feature_info['col_labels'], missing))

        last_layer_train = trainer.predict_y(selected['train']['X'])
        last_layer_val = trainer.predict_y(selected['validate']['X'])
        write_pred(last_layer_train, "last_layer_train.csv", header)
        write_pred(last_layer_val, "last_layer_val.csv", header)
        write_pred(selected['train']['Y'], "y_train.csv", header)
        write_pred(selected['validate']['Y'], "y_validate.csv", header)


def select_features(options):
    with open(options.feature_group_file) as feature_fd:
        feature_dict = json.load(feature_fd)

    if options.feature_groups is None:
        features_to_train = feature_dict
    else:
        if not all(k in feature_dict for k in options.feature_groups):
            raise KeyError(
                ("one or more of the following features cannot be found %s" %
                    options.feature_groups))
        features_to_train = dict((k, feature_dict[k]) for k in
                                 options.feature_groups if k in feature_dict)

    return features_to_train


def train_main(options):
    '''This loads the data, manipulates it and trains the model, it's the glue
    that does all of the work.
    '''
    # Load the nnet_config
    nnet_config = load_nnet_config(options)

    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(options.faces_csv, options.faces_pickled,
                              fast_nrows=options.num_rows)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    #
    # Partition the Dataset
    #
    # Convert raw data from float64 to floatX
    # (32/64-bit depending on GPU/CPU)
    typed_data = faces.get_data()
    typed_data['X'] = lasagne.utils.floatX(typed_data['X'])
    typed_data['Y'] = lasagne.utils.floatX(typed_data['Y'])

    start_time = time.time()
    partitioner = partition.Partitioner(
        typed_data, {'train': 70, 'validate': 30},
        os.path.join(options.run_data_path, "partition_indices.pkl"))
    partitions = partitioner.run()
    print_partition_shapes(partitions)
    print "Partition Took {:.3f}s".format(time.time() - start_time)

    #
    # Run any transformations on the training dataset here.
    #

    #
    # Which features to predict
    #
    features_to_train = select_features(options)
    for feature_index, (feature_name, feature_cols) in enumerate(sorted(
            features_to_train.items())):
        #
        # Setup environment for training a feature
        #
        print "\n\n%s\nFeature Set %s (%d of %d)\n%s" % (
            "#" * 80, feature_name, feature_index+1,
            len(features_to_train), "#" * 80)

        feature_path = os.path.abspath(feature_name)
        print "Changing to %s" % feature_path
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        os.chdir(feature_path)

        #
        # Select the Training data
        #
        # Select feature columns
        start_time = time.time()
        feature_col_labels = (
            [faces.get_labels()['Y'][i] for i in feature_cols])
        print "Selecting features %s for %s" % (
            feature_col_labels, feature_name)
        selected = select_data(options, feature_cols, partitions)
        print_partition_shapes(selected)
        print "Selecting Data Took {:.3f}s".format(time.time() - start_time)

        feature_info = {
            'cols': feature_cols,
            'col_labels': feature_col_labels
        }
        train_feature(options, selected, nnet_config, feature_info)

        #
        # Change back to the run directory for the next run.
        #
        print "Changing to %s" % options.run_data_path
        os.chdir(options.run_data_path)

        # Drop into a console so that we do anything additional we need.
        if options.drop_to_console:
            code.interact(local=locals())

    # Finally combine all of the loss-functions to produce a loss output.
    start_time = time.time()
    combine_loss(options, features_to_train)
    print "Combining CSVs took {:.3f}s".format(time.time() - start_time)


def main():
    '''Parse Arguments and call real_main
    '''
    #
    # Create list of options and parse them.
    #
    parser = argparse.ArgumentParser(
        version=get_version(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'action', nargs='?', choices=('train', 'loss'),
        default='train', help="action to perform")
    parser.add_argument(
        '-o', '--output_dir', dest='run_data_path',
        metavar="PATH",
        default=datetime.datetime.now().strftime('run_%Y-%m-%d__%H_%M_%S'),
        help="directory to place run information and state")
    parser.add_argument(
        '-c', '--config_file', dest='config_file',
        metavar="FILE", default="configs/default.cfg",
        help="neural network configuration file")
    parser.add_argument(
        '--console', dest='drop_to_console', action="store_true",
        help="drop to console after finishing processing")
    parser.add_argument(
        '--feature_groups', dest='feature_groups', metavar="FEAT",
        default=None, nargs="+",
        help="feature groups to train")
    parser.add_argument(
        '--amputate', dest='amputate', action="store_true",
        help="train a neural network and save output of penultimate layer")
    parser.add_argument(
        '--predict', dest='predict', action="store_true",
        help="predict output after training")

    train_group = parser.add_argument_group(
        "Training Control", "Options for Controlling Training")
    train_group.add_argument(
        '-e', '--epochs', dest='num_epochs', type=int, metavar="EPOCHS",
        default=100,
        help="number of epochs to train against")
    train_group.add_argument(
        '-i', '--interval', dest='save_state_interval', type=int,
        metavar="EPOCHS",
        default=10,
        help="how often (in epochs) to save internal model state")
    train_group.add_argument(
        '-b', '--batchsize', dest='batchsize', type=int, metavar="ROWS",
        default=None,
        help="override the batchsize specified in config_file")

    data_group = parser.add_argument_group(
        "Data Options", "Options for controlling the input data.")
    data_group.add_argument(
        '--faces_csv', dest='faces_csv',
        metavar="PATH", default=os.path.abspath("../data/training.csv"),
        help="path to the faces CSV file")
    data_group.add_argument(
        '--faces_pickle', dest='faces_pickled',
        metavar="PATH", default=os.path.abspath("../data/training.pkl.gz"),
        help="path to the faces pickle file")
    data_group.add_argument(
        '--feature_group_file', dest='feature_group_file',
        metavar="PATH", default=os.path.abspath("feature_groups.json"),
        help="path to the featuer groups")
    data_group.add_argument(
        '--num_rows', dest='num_rows', type=int, metavar="ROWS",
        default=None,
        help="override to specify number of first rows to use")

    argcomplete.autocomplete(parser)
    options = parser.parse_args()

    # Create a directory for our work.
    if os.path.exists(options.run_data_path):
        print "Using existing data directory at %s" % options.run_data_path
    else:
        print "Creating data directory at %s" % options.run_data_path
        os.mkdir(options.run_data_path)

    options.run_data_path = os.path.abspath(options.run_data_path)
    options.feature_group_file = os.path.abspath(options.feature_group_file)

    # pylint: disable=too-many-locals
    # Yes we have a lot of variables, but this is the main function.

    # Make the config file path absolute (so that a chdir won't affect it)
    options.config_file = os.path.abspath(options.config_file)

    #
    # Change to the run data directory
    #
    print "Changing to Run Directory=%s" % options.run_data_path
    os.chdir(options.run_data_path)

    # Tee the output to a logfile.
    console_fd = open('console_log.txt', 'a')
    sys.stdout = Tee(sys.stdout, console_fd)
    sys.stderr = Tee(sys.stderr, console_fd)
    print "Tee done; subsequent prints appear on terminal and console_log.txt"

    #
    # Log the current git revision/cmdline for reproducibility
    #
    print get_version()
    print "cmdline = '%s'" % " ".join(sys.argv)

    # Run the function for the specified action.
    {
        'train': train_main,
        'loss': combine_loss_main,
    }[options.action](options)


if __name__ == "__main__":
    main()
