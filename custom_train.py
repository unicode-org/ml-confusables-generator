# Copyright (C) 2020 and later: Google, Inc.

import argparse
from argparse import RawDescriptionHelpFormatter
import configparser
from dataset_builder import DatasetBuilder
from model_builder import ModelBuilder
import os
import re
import tensorflow as tf
import tensorflow_addons as tfa


class ModelTrainer:
    def __init__(self, config_path='configs/sample_config.ini'):
        """Read and set configuration from config file (.ini file) and create
        keras.Model object or input function according to configuration. To add
        new model, simply add new base model to self._MODEL_MAP.

        Args:
            config_path: Str, path to config (.ini) file.

        Raises:
            ValueError: if values in config file does not have the correct type.
            ValueError: if optimizer does not exists in predefined map.
        """
        # Pre-defined learning rate schedules
        self._LR_SCHEDULE_MAP = {
            'ExponentialDecay':
                tf.keras.optimizers.schedules.ExponentialDecay,
            'PiecewiseConstantDecay':
                tf.keras.optimizers.schedules.PiecewiseConstantDecay,
            'PolynomialDecay':
                tf.keras.optimizers.schedules.PolynomialDecay,
        }

        # Pre-defined optimizers
        self._OPTIMIZER_MAP = {
            'Adam':
                tf.keras.optimizers.Adam,
            'RMSprop':
                tf.keras.optimizers.RMSprop,
        }

        # Pre-defined losses
        # IMPORTANT: DON'T USE TRIPLET HARD LOSS! EXTREMELY HARD TO TRAIN!
        self._LOSS_MAP = {
            'CrossEntropy':
                tf.keras.losses.CategoricalCrossentropy,
            'TripletHard':
                tfa.losses.TripletHardLoss,
            'TripletSemiHard':
                tfa.losses.TripletSemiHardLoss,

        }

        # Pre-defined metrics
        self._METRIC_MAP = {
            'Accuracy':
            tf.keras.metrics.CategoricalAccuracy,
        }

        # Get custom dataset
        self.datset_builder = DatasetBuilder(config_path=config_path)
        self.model_builder = ModelBuilder(config_path=config_path)

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Get classifier training config
        self._CLS_CKPT_DIR = config.get('CLASSIFIER_TRAINING', 'CKPT_DIR')
        self._CLS_MAX_STEP = config.getint('CLASSIFIER_TRAINING', 'MAX_STEP')
        self._CLS_OPTIMIZER = config.get('CLASSIFIER_TRAINING', 'OPTIMIZER')
        self._CLS_LR_BOUNDARIES = [
            int(boundary.strip()) for boundary in
            config.get('CLASSIFIER_TRAINING', 'LR_BOUNDARIES').split(',')
        ]
        self._CLS_LR_VALUES = [
            float(value.strip()) for value in
            config.get('CLASSIFIER_TRAINING', 'LR_VALUES').split(',')
        ]

        # Get triplet training config
        self._TPL_INIT_DIR = config.get('TRIPLET_TRAINING', 'INIT_DIR')
        self._TPL_CKPT_DIR = config.get('TRIPLET_TRAINING', 'CKPT_DIR')
        self._TPL_CYCLES = config.getint('TRIPLET_TRAINING', 'CYCLES')
        self._TPL_EPOCHS = config.getint('TRIPLET_TRAINING', 'EPOCHS')
        self._TPL_FILTER_SIZE = config.getint('TRIPLET_TRAINING', 'FILTER_SIZE')
        self._TPL_MARGIN = config.getfloat('TRIPLET_TRAINING', 'MARGIN')
        self._TPL_OPTIMIZER = config.get('TRIPLET_TRAINING', 'OPTIMIZER')
        self._TPL_LR_VALUE = config.getfloat('TRIPLET_TRAINING',
                                             'LEARNING_RATE')
        self._TPL_FREEZE_VARS = [
            var.strip() for var in
            config.get('TRIPLET_TRAINING', 'FREEZE_VARS').split(',')
        ]

        # Throw exception if optimizer is not defined
        if self._CLS_OPTIMIZER not in self._OPTIMIZER_MAP.keys():
            raise ValueError("CLASSIFIER_TRAINING OPTIMIZER not defined.")
        if self._TPL_OPTIMIZER not in self._OPTIMIZER_MAP.keys():
            raise ValueError("TRIPLET_TRAINING OPTIMIZER not defined.")

    def train_classifier(self):
        '''Train classifer according to specs in config file.'''
        # When training classifier, we uses one-hot encoding as label
        self.datset_builder.ONE_HOT = True

        # Create full model using model_builder
        model, input_name = self.model_builder.create_full_model()
        # Sanity check
        model.summary()

        # Set learning rate schedule
        boundaries = self._CLS_LR_BOUNDARIES
        values = self._CLS_LR_VALUES
        lr_schedule = self._LR_SCHEDULE_MAP['PiecewiseConstantDecay'](
            boundaries=boundaries, values=values)
        # Use learning reate schedule to create optimizer
        optimizer = self._OPTIMIZER_MAP[self._CLS_OPTIMIZER](
            learning_rate=lr_schedule)
        # Create loss function
        loss = self._LOSS_MAP['CrossEntropy'](from_logits=True)
        # Add accuracy metrics
        accuracy = self._METRIC_MAP['Accuracy']()
        model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

        # Build tf.estimator
        estimator = tf.keras.estimator \
            .model_to_estimator(keras_model=model, model_dir=self._CLS_CKPT_DIR)
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.datset_builder.get_train_input_fn(input_name),
            max_steps=self._CLS_MAX_STEP)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.datset_builder.get_eval_input_fn(input_name))

        # Start training
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def _freeze_vars(self, model):
        """Freeze variables in the model based on regular expressions in
        self._TPL_FREEZE_VARS.

        Args:
            model: tf.keras.Model, the model within which variables are frozen.
        """
        # Get regular expressions in config file.
        freeze_var_res = self._TPL_FREEZE_VARS
        # Get layers that matches regular expression.
        freeze_layers = [layer for layer in model.layers if
                         any(re.match(str(pattern), layer.name) for pattern in
                             freeze_var_res)]
        # Freeze layers.
        print('Freezing {} layers.'.format(str(len(freeze_layers))))
        for layer in freeze_layers:
            print('Freezing layer {}.'.format(layer.name))
            layer.trainable = False

    def train_triplet_transfer(self):
        """Train encoder with triplet loss according to specs in config file."""
        # When training using triplet loss, we avoid using one-hot encoding
        self.datset_builder.ONE_HOT = False

        # Create full model using model_builder
        model, input_name = self.model_builder.create_full_model()
        # Sanity check
        model.summary()

        # Build optimizer
        optimizer = self._OPTIMIZER_MAP[self._TPL_OPTIMIZER](self._TPL_LR_VALUE)

        # Load initial weights from self._TPL_INIT_DIR
        init_dir = self._TPL_INIT_DIR
        latest = tf.train.latest_checkpoint(init_dir)
        model.load_weights(latest)

        # Get ResNet50 model
        resnet_model = model.layers[0]
        # Freeze specified variables
        self._freeze_vars(resnet_model)

        # Create loss function
        loss = self._LOSS_MAP['TripletSemiHard'](self._TPL_MARGIN)
        model.compile(optimizer=optimizer, loss=loss)

        # Train triplet model
        # In each cycle, a new training dataset with N labels are generated and
        # training is carried out for M epochs.
        # Total number of cycles = self._TPL_CYCLES
        # N = self._TPL_FILTER_SIZE
        # M = self._TPL_EPOCHS
        for i in range(self._TPL_CYCLES):
            print('Cycle #{}'.format(i+1))
            train_dataset = self.datset_builder.get_train_dataset(
                filter_size=self._TPL_FILTER_SIZE)
            history = model.fit(
                train_dataset,
                epochs=self._TPL_EPOCHS
            )
            # Store weights every 50 cycles
            if (i+1) % 50 == 0:
                model.save_weights(self._TPL_CKPT_DIR + '_#{}'.format(i+1))
        model.save_weights(self._TPL_CKPT_DIR)

if __name__ == "__main__":
    formatter = RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description='Usage: \n',
                                     formatter_class=formatter)
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to config file.')
    parser.add_argument('--mode', type=str, required=True,
                        help='The mode of training, one of "classifier" or '
                             '"triplet".')
    args = parser.parse_args()

    # Get config file and training mode from cli
    config_file = args.config_file
    mode = args.mode

    # Check that config file exists, if not, raise ValueError
    if not os.path.isfile(config_file):
        raise ValueError('Config file does not exist.')

    mt = ModelTrainer(config_path=config_file)
    if mode == "classifier":
        mt.train_classifier()
    elif mode == "triplet":
        mt.train_triplet_transfer()
    else:
        raise ValueError('Training mode must be one of "classifier" or '
                         '"triplet"')
