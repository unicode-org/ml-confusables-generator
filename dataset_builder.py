# Copyright (C) 2020 and later: Google, Inc.

import configparser
import data_preprocessing
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import pathlib
import tensorflow as tf


# AUTOTUNE allows TensorFlow to find a good allocation of CPU budget for
# performance optimization
AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetBuilder:
    def __init__(self, config_path='configs/sample_config.ini', one_hot=True):
        """Read and set configuration from config file (.ini file) and create
        tf.Dataset object or input function according to configuration.

        Args:
            config_path: Str, path to config (.ini) file.
            one_hot: Bool, whether or not to return label as one-hot encoding.

        Raises:
            ValueError: if values in config file does not have the correct type.
        """
        # Set one-hot encoding setting
        self.ONE_HOT = one_hot

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Get (and check) configuration
        # Get dataset info
        self._TRAIN_DATA_DIR = config.get('DATASET', 'TRAIN_DATA_DIR')
        self._TEST_DATA_DIR = config.get('DATASET', 'TEST_DATA_DIR')
        self._LABEL_FILE = config.get('DATASET', 'LABEL_FILE')

        # Get image info
        self._HEIGHT = config.getint('IMAGE', 'HEIGHT')
        self._WIDTH = config.getint('IMAGE', 'WIDTH')
        self._GRAYSCALE_IN = config.getboolean('IMAGE', 'GRAYSCALE_IN')
        self._GRAYSCALE_OUT = config.getboolean('IMAGE', 'GRAYSCALE_OUT')

        # Get traning and testing spec
        self._TRAIN_BATCH_SIZE = config.getint('TRAIN_TEST_SPEC',
                                              'TRAIN_BATCH_SIZE')
        self._TEST_BATCH_SIZE = config.getint('TRAIN_TEST_SPEC',
                                             'TEST_BATCH_SIZE')
        self._SHUFFLE_BUFFER_SIZE = config.getint('TRAIN_TEST_SPEC',
                                                 'SHUFFLE_BUFFER_SIZE')
        self._PREFETCH_BUFFER_SIZE = config.getint('TRAIN_TEST_SPEC',
                                                  'PREFETCH_BUFFER_SIZE')

        # Get data augmentation spec
        self._RANDOM_ROTATE = config.getboolean('DATA_AUG', 'RANDOM_ROTATE')
        self._ROTATE_STDDEV = config.getfloat('DATA_AUG', 'ROTATE_STDDEV')

        self._RANDOM_ZOOM = config.getboolean('DATA_AUG', 'RANDOM_ZOOM')
        self._ZOOM_PERCENT = config.getfloat('DATA_AUG', 'ZOOM_PERCENT')
        self._ZOOM_STDDEV = config.getfloat('DATA_AUG', 'ZOOM_STDDEV')

        self._RESIZE = config.getboolean('DATA_AUG', 'RESIZE')
        self._RESIZE_HEIGHT = config.getint('DATA_AUG', 'RESIZE_HEIGHT')
        self._RESIZE_WIDTH = config.getint('DATA_AUG', 'RESIZE_WIDTH')

        # Label conversion
        self._CLASS_NAMES = [line.strip() for line in
                             open(self._LABEL_FILE).readlines()]
        self._NUM_CLASSES = len(self._CLASS_NAMES) # Number of classes

    def _get_data_preprocessing_fns(self):
        """Get multiple data preprocessing functions with partial positional
        arguments assigned with corresponding configuration.

        Returns: EasyDict, allowing accessing dict values as attributes.
        """
        # Create new functions with partial positional arguments assigned
        process_path_fn = \
            partial(data_preprocessing.process_path,
                    one_hot=self.ONE_HOT,
                    num_classes=self._NUM_CLASSES,
                    class_names=self._CLASS_NAMES)
        process_img_path_fn = data_preprocessing.process_img_path
        convert_format_fn = \
            partial(data_preprocessing.convert_format,
                    grayscale_in=self._GRAYSCALE_IN,
                    grayscale_out=self._GRAYSCALE_OUT)
        random_rotate_fn = \
            partial(data_preprocessing.random_rotate,
                    stddev=self._ROTATE_STDDEV)
        random_zoom_fn = \
            partial(data_preprocessing.random_zoom,
                    max_percent=self._ZOOM_PERCENT,
                    stddev=self._ZOOM_STDDEV,
                    img_height=self._HEIGHT,
                    img_width=self._WIDTH)
        resize_fn = \
            partial(data_preprocessing.resize,
                    height=self._HEIGHT,
                    width=self._WIDTH)

        funcs = edict({'process_path': process_path_fn,
                       'process_img_path': process_img_path_fn,
                       'convert_format': convert_format_fn,
                       'random_rotate': random_rotate_fn,
                       'random_zoom': random_zoom_fn,
                       'resize': resize_fn})

        return funcs

    def get_train_dataset(self, filter_size=None):
        """Get training dataset. For the purpose of triplet selection, restrict
        dataset to have certain number of labels if specified. See
        https://www.tensorflow.org/addons/tutorials/losses_triplet.

        Args:
            filter_size: Int or None, if filter_size is None, do nothing.
                         If filter_size is Int, restrict datset to only have
                         filter_size number of classes. The classes to include
                         is randomly selected form all classes.

        Returns:
            ds: tf.Dataset, TensorFlow dataset object for training. Each entry
                is (image, label) pair.
        """
        # Get filename dataset (each entry is a filename)
        data_dir = pathlib.Path(self._TRAIN_DATA_DIR)
        list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

        # Create data pre-processing functions
        funcs = self._get_data_preprocessing_fns()

        # Get labeled dataset (each entry is (image, label) tuple)
        ds = list_ds.map(funcs.process_path, num_parallel_calls=AUTOTUNE)

        # Execute when filter_size is not None or 0
        if filter_size:
            # Filter using filter_size
            labels = tf.constant(np.random.choice(self._NUM_CLASSES,
                                                  filter_size,
                                                  replace=False))
            ds = ds.filter(lambda img, label:
                           tf.reduce_any(tf.equal(label,labels)))

        # Format conversion
        ds = ds.map(funcs.convert_format, num_parallel_calls=AUTOTUNE)
        # Map rotate function
        if self._RANDOM_ROTATE:
            ds = ds.map(funcs.random_rotate, num_parallel_calls=AUTOTUNE)
        # Map zoom-in function
        if self._RANDOM_ZOOM:
            ds = ds.map(funcs.random_zoom, num_parallel_calls=AUTOTUNE)
        # Image resizing
        ds = ds.map(funcs.resize, num_parallel_calls=AUTOTUNE)

        # Shuffle, batch, repeat, prefetch
        ds = ds.shuffle(buffer_size=self._SHUFFLE_BUFFER_SIZE)
        ds = ds.batch(self._TRAIN_BATCH_SIZE)
        ds = ds.prefetch(buffer_size=self._PREFETCH_BUFFER_SIZE)

        return ds

    def get_test_dataset(self):
        """Get test dataset.

        Returns:
            ds: tf.Dataset, TensorFlow dataset object for testing. Each entry
                is (image, label) pair.
        """
        # Get filenames
        data_dir = pathlib.Path(self._TEST_DATA_DIR)
        list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

        # Create data pre-processing functions
        funcs = self._get_data_preprocessing_fns()

        # Get labeled dataset
        ds = list_ds.map(funcs.process_path, num_parallel_calls=AUTOTUNE)
        # Format conversion
        ds = ds.map(funcs.convert_format, num_parallel_calls=AUTOTUNE)
        # Resizing
        ds = ds.map(funcs.resize, num_parallel_calls=AUTOTUNE)

        # Batch, prefetch
        ds = ds.batch(self._TEST_BATCH_SIZE)
        ds = ds.prefetch(buffer_size=self._PREFETCH_BUFFER_SIZE)

        return ds

    def get_filename_dataset(self, data_dir):
        """For prediciton only! No label file needed! Given a directory of
        images, return datatset with images and filenames.

        Args:
            data_dir: Str, path to image directory.

        Returns:
            ds: tf.Dataset, TensorFlow dataset object for training. Each entry
                is (image, filename) pair.
        """
        # Get filenames
        data_dir = pathlib.Path(data_dir)
        list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

        # Create data pre-processing functions
        funcs = self._get_data_preprocessing_fns()

        # Get FAKE labeled dataset
        ds = list_ds.map(funcs.process_img_path, num_parallel_calls=AUTOTUNE)
        # Format conversion
        ds = ds.map(funcs.convert_format, num_parallel_calls=AUTOTUNE)
        # Resizing
        ds = ds.map(funcs.resize, num_parallel_calls=AUTOTUNE)

        # Batch, prefetch
        ds = ds.batch(1)

        return ds


    def get_train_input_fn(self, input_name):
        """For tf.estimator training. Create train_input_fn that returns a
        tf.Dataset when called. Each entry in tf.Dataset is a
        {input_name: image}, label pair.

        Args:
            input_name: Str, name of the input tensor. Required by tf.estimator.

        Returns:
            train_input_fn: Function, returns tf.Dataset when called.
        """
        def train_input_fn():
            # Get filenames
            data_dir = pathlib.Path(self._TRAIN_DATA_DIR)
            list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

            # Create data pre-processing functions
            funcs = self._get_data_preprocessing_fns()

            # Get labeled dataset
            ds = list_ds.map(funcs.process_path, num_parallel_calls=AUTOTUNE)
            # Format conversion
            ds = ds.map(funcs.convert_format, num_parallel_calls=AUTOTUNE)
            # Map rotate function
            if self._RANDOM_ROTATE:
                ds = ds.map(funcs.random_rotate, num_parallel_calls=AUTOTUNE)
            # Map zoom-in function
            if self._RANDOM_ZOOM:
                ds = ds.map(funcs.random_zoom, num_parallel_calls=AUTOTUNE)
            # Resizing
            ds = ds.map(funcs.resize, num_parallel_calls=AUTOTUNE)

            # Prepare for tf.estimator
            ds = ds.map(lambda img, label: ({input_name: img}, label))

            # Shuffle, batch, repeat, prefetch
            ds = ds.shuffle(buffer_size=self._SHUFFLE_BUFFER_SIZE)
            ds = ds.batch(self._TRAIN_BATCH_SIZE)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=self._PREFETCH_BUFFER_SIZE)

            return ds
        return train_input_fn

    def get_eval_input_fn(self, input_name):
        """For tf.estimator evaluation. Create eval_input_fn that returns a
        tf.Dataset when called. Each entry in tf.Dataset is a
        {input_name: image}, label pair.

        Args:
            input_name: Str, name of the input tensor. Required by tf.estimator.

        Returns:
            train_input_fn: Function, returns tf.Dataset when called.
        """
        def eval_input_fn():
            # Get filenames
            data_dir = pathlib.Path(self._TEST_DATA_DIR)
            list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

            # Create data pre-processing functions
            funcs = self._get_data_preprocessing_fns()

            # Get labeled dataset
            ds = list_ds.map(funcs.process_path, num_parallel_calls=AUTOTUNE)
            # Format conversion
            ds = ds.map(funcs.convert_format, num_parallel_calls=AUTOTUNE)
            # Resizing
            ds = ds.map(funcs.resize, num_parallel_calls=AUTOTUNE)

            # Prepare for tf.estimator
            ds = ds.map(lambda img, label: ({input_name: img}, label))

            # Batch, prefetch
            ds = ds.batch(self._TEST_BATCH_SIZE)
            ds = ds.prefetch(buffer_size=self._PREFETCH_BUFFER_SIZE)

            return ds
        return eval_input_fn
    

if __name__ == "__main__":
    db = DatasetBuilder()
    db.ONE_HOT = False
    train_input_fn = db.get_eval_input_fn(input_name = 'resnet50_input')
    train_ds = train_input_fn()
    # train_ds = db.get_test_dataset()
    for features_batch, labels_batch in train_ds.take(1):
        print(features_batch)
        print(labels_batch)
        # import pdb;pdb.set_trace()
