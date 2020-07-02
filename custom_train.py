import tensorflow as tf
from custom_dataset import train_input_fn, test_input_fn
from custom_model import create_full_model, compile_model

if __name__ == "__main__":
    model = create_full_model('ResNet50')
    model = compile_model(model)
    model.summary()

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
    estimator.train(input_fn=train_input_fn(), steps=500)