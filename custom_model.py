# Copyright (C) 2020 and later: Google, Inc.

import configparser
import tensorflow as tf

class ModelBuilder:
    def __init__(self, config_path='configs/sample_config.ini'):
        """Read and set configuration from config file (.ini file) and create
        keras.Model object or input function according to configuration. To add
        new model, simply add new base model to self._MODEL_MAP.

        Args:
            config_path: Str, path to config (.ini) file.

        Raises:
            ValueError: if values in config file does not have the correct type.
            ValueError: if model name does not exists in pre-defined map.
        """
        # Rre-defined models
        self._MODEL_MAP = {
            'ResNet50': tf.keras.applications.ResNet50,
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'VGG16': tf.keras.applications.VGG16,
        }

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Model info
        self._MODEL_NAME = config.get('MODEL', 'NAME')
        self._INPUT_SHAPE = (config.getint('MODEL', 'INPUT_DIM1'),
                            config.getint('MODEL', 'INPUT_DIM2'),
                            config.getint('MODEL', 'INPUT_DIM3'))
        self._OUTPUT_SHAPE = config.getint('MODEL', 'OUTPUT_SHAPE')

        # Throw exception if model name is not defined
        if self._MODEL_NAME not in self._MODEL_MAP.keys():
            raise ValueError('MODEL NAME in config file undefined.')

    def create_full_model(self):
        """Create end-to-end model for training.

        Returns:
            full_model: tf.keras.Model, model not yet compiled.
            input_name: Str, name for the model input.
        """
        base_model = self._MODEL_MAP[self._MODEL_NAME](
            input_shape=self._INPUT_SHAPE, include_top=False)
        base_model.trainable = True

        full_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self._OUTPUT_SHAPE),
        ])
        input_name = full_model.input.name.split(":")[0]

        return full_model, input_name

if __name__ == "__main__":
    mb = ModelBuilder()
    model, input_name = mb.create_full_model()
    model.summary()

