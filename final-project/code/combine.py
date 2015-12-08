#!/usr/bin/env python
import argparse
import os
import re
import time

import pandas as pd
import numpy as np


COORD_COLUMNS = [
    "left_eye_center_x",            "left_eye_center_y",
    "right_eye_center_x",           "right_eye_center_y",
    "left_eye_inner_corner_x",      "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",      "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",     "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",     "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",     "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",     "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",    "right_eyebrow_outer_end_y",
    "nose_tip_x",                   "nose_tip_y",
    "mouth_left_corner_x",          "mouth_left_corner_y",
    "mouth_right_corner_x",         "mouth_right_corner_y",
    "mouth_center_top_lip_x",       "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",    "mouth_center_bottom_lip_y"]


def missing_cols_names():
    ordered_cols = [re.sub(r'_[xy]$', '', f) for f in COORD_COLUMNS]
    selected_cols = ([c for (i, c) in enumerate(ordered_cols) if i
                     in range(0, len(ordered_cols), 2)])
    assert set(selected_cols) == set(ordered_cols)
    return ['missing_' + c for c in selected_cols]


def process(in_dir, in_filename, out_filepath):
    candidate_sources = (
        [d for d in os.listdir(in_dir)
            if os.path.isdir(os.path.join(in_dir, d))])

    sources = (
        [d for d in candidate_sources if
            os.path.exists(os.path.join(in_dir, d, in_filename))])

    def process_file(source):
        y_hat_path = os.path.join(in_dir, source, in_filename)
        return pd.read_csv(y_hat_path, engine='c', index_col=0)

    start_time = time.time()
    print "Reading files"
    frames = [process_file(s) for s in sources]
    print [df.shape for df in frames]
    print "  took {:.3f}s".format(time.time() - start_time)

    start_time = time.time()
    print "Concatenating Dataframes"
    result = pd.concat(frames, axis=1)
    all_column_names = np.concatenate((COORD_COLUMNS, missing_cols_names()))
    result.sort_index(inplace=True)
    result = result[all_column_names]
    print "  took {:.3f}s".format(time.time() - start_time)

    start_time = time.time()
    print "Writing output to %s" % out_filepath
    result.to_csv(out_filepath)
    print "  took {:.3f}s".format(time.time() - start_time)


def real_main(options):
    datasources = {
        "valid": {
            "pred": "last_layer_val.csv",
            "actual": "y_validate.csv"
        },
        "train": {
            "pred": "last_layer_train.csv",
            "actual": "y_train.csv"
        }
    }

    for source_name, source_dict in datasources.items():
        for type_name, filename in source_dict.items():
            out_file = (
                "combined_" + "_".join([source_name, type_name]) + '.csv')
            process(options.in_dir,
                    filename, os.path.join(options.in_dir, out_file))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--dir', dest='in_dir', help="Input Directory", required=True)

    options = parser.parse_args()

    real_main(options)

if __name__ == "__main__":
    # missing_cols_names()
    main()
