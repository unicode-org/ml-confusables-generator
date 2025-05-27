# Copyright (C) 2020 and later: Unicode, Inc. and others.
# License & terms of use: http://www.unicode.org/copyright.html

import ast
import cv2
import os
import random
import shutil

def calculate_from_path(metric, path1, path2):
    """Calculate distance between the two images specified by file path.

    Args:
        metric: Function, distance metric to be used.
        path1: Str, path to the first image.
        path2: Str, path to the second image.

    Returns:
        distance: Float, distance between the two images.
    """

    try:
        img1 = cv2.imread(path1)
    except FileNotFoundError:
        print('Image at path1 not found.')
        raise

    try:
        img2 = cv2.imread(path2)
    except FileNotFoundError:
        print('Image at path2 not found.')
        raise

    return metric(img1, img2)

def train_test_split(train_dir, test_dir, num_test=100):
    """Split dataset (already created) into training and testing datasets.
    Expect all data record to be in train_dir. Expect test_dir to be either
    empty or does not exists.


    Args:
        train_dir: Str, relative path to the directory containing training
            images.
        test_dir: Str, relative path to test directory (expect to be non-exist
            or emtpy).
        num_test: Int, number of test records.

    Returns:
        num_train: Int, total number of training records.
        num_test: Int, total number of test records.

    Raises:
        OSError: If no images found in out_dir.
        ValueError: If num_test is larger than total number of records.
        OSError: If test data already exists.
    """
    # Get absolute path to train and test data directory
    train_dir_abs = os.path.abspath(train_dir)
    test_dir_abs = os.path.join(os.getcwd(), test_dir)

    # Create test dir
    os.makedirs(test_dir_abs, exist_ok=True)

    # Get total number of training records
    num_total = len(next(os.walk(train_dir_abs))[2])
    num_exist = len(next(os.walk(test_dir_abs))[2])
    if num_total == 0:
        raise OSError('No data found in specified out_dir.')
    if num_test > num_total:
        raise ValueError('Expect num_test to be smaller than total number '
                         'of records.')
    if num_exist != 0:
        raise OSError('Test data already exists.')
    num_train = num_total - num_test

    # Do train/test split
    print('Creating train test split with {} total records...'
          .format(num_total))
    print('Train size: {}'.format(num_train))
    print('Test size: {}'.format(num_test))
    filenames = random.sample(os.listdir(train_dir_abs), num_test)
    for filename in filenames:
        srcpath = os.path.join(train_dir_abs, filename)
        shutil.move(srcpath, test_dir_abs)
    print('Train test split successfully created.')

    # Check number of classes in each split
    class_train = set([name.split('_')[0] for name in
                       os.listdir(train_dir_abs)])
    class_test = set([name.split('_')[0] for name in
                      os.listdir(test_dir_abs)])
    no_missing_class = class_test.issubset(class_train)
    print('Training dataset has {} categories.'.format(len(class_train)))
    print('Test dataset has {} categories.'.format(len(class_test)))
    print('All test categories in training data: {}'
          .format(no_missing_class))

    return num_train, num_test

def confusables_dis_to_confusables():
    """Read in the content of confusables_dis.txt and make confusabels.txt."""
    with open('confusables_dis.txt') as fin,\
        open('confusables.txt', 'w+') as fout:
        for line in fin:
            # char is the anchor character
            char = line[0]
            # confs_ids is a list of confusable-distance tuple
            confs_dis = ast.literal_eval(line[3:])
            # confs is a list of confusables
            confs = [pair[0] for pair in confs_dis]

            # Write to confusables.txt
            fout.write(char)
            fout.write(": ")
            fout.write(str(confs))
            fout.write('\n')
