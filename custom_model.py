r"""Pre-defined TensorFlow (Keras) models."""
import tensorflow as tf
import numpy as np

# Model info
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 1000

# Enable pre-defined models
MODEL_MAP = {
    'ResNet50': tf.keras.applications.ResNet50,
    'MobileNetV2': tf.keras.applications.MobileNetV2,
    'VGG16': tf.keras.applications.VGG16,
}

def create_full_model(model_name):
    """Create end-to-end model for training.

    Args:
        model_name: Str, one of the available model names in MODEL_MAP

    Returns:
        full_model: tf.keras.Model, model not yet compiled.
        input_name: Str, name for the model input.
    """
    base_model = MODEL_MAP[model_name](input_shape=INPUT_SHAPE,
                                       include_top=False)
    base_model.trainable = True

    full_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    input_name = full_model.input.name.split(":")[0]

    return full_model, input_name

if __name__ == "__main__":
    model = create_full_model('ResNet50')
    model.summary()

