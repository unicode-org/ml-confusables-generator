import tensorflow as tf
from custom_dataset import get_train_input_fn, get_test_input_fn
from custom_model import create_full_model, compile_model

if __name__ == "__main__":
    model = create_full_model('ResNet50')
    model = compile_model(model)
    model.summary()

    input_name = model.input.name.split(':')[0]

    estimator = tf.keras.estimator\
        .model_to_estimator(keras_model=model, model_dir='ckpts/resnet_eval')
    estimator.train(input_fn=get_train_input_fn(input_name), steps=500)
    estimator.evaluate(input_fn=get_test_input_fn(input_name), steps=100)