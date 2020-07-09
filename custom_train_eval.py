import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from custom_dataset import get_train_input_fn, get_eval_input_fn, get_train_dataset, get_test_dataset
from custom_model import create_full_model
import os

# Pre-defined learning rate schedules
LR_SCHEDULE_MAP = {
    'ExponentialDecay':
        tf.keras.optimizers.schedules.ExponentialDecay,
    'PiecewiseConstantDecay':
        tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    'PolynomialDecay':
        tf.keras.optimizers.schedules.PolynomialDecay,
}

# Pre-defined optimizers
OPTIMIZER_MAP = {
    'Adam':
        tf.keras.optimizers.Adam,
    'RMSprop':
        tf.keras.optimizers.RMSprop,
}

# Pre-defined losses
# IMPORTANT: DON'T USE TRIPLET HARD LOSS! EXTREMELY HARD TO TRAIN!
LOSS_MAP = {
    'CrossEntropy':
        tf.keras.losses.CategoricalCrossentropy,
    'TripletHard':
        tfa.losses.TripletHardLoss,
    'TripletSemiHard':
        tfa.losses.TripletSemiHardLoss,

}

# Pre-defined metrics
METRIC_MAP = {
    'Accuracy':
    tf.keras.metrics.CategoricalAccuracy,
}


def train_categorical(model_name='ResNet50'):
    '''Use this function as tutorial.'''
    model, input_name = create_full_model(model_name)
    model.summary()

    boundaries = [20000, 25000]
    values = [0.001, 0.0001, 0.00001]
    lr_schedule = LR_SCHEDULE_MAP['PiecewiseConstantDecay'](
        boundaries=boundaries, values=values)
    optimizer = OPTIMIZER_MAP['Adam'](learning_rate=lr_schedule)

    loss = LOSS_MAP['CrossEntropy'](from_logits=True)

    accuracy = METRIC_MAP['Accuracy']()
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    estimator = tf.keras.estimator \
        .model_to_estimator(keras_model=model, model_dir='ckpts/' + model_name)
    train_spec = tf.estimator.TrainSpec(input_fn=get_train_input_fn(input_name),
                                        max_steps=30000)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_eval_input_fn(input_name))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def deprecated_train_triplet():
    '''THIS DOES NOT WORK!!'''
    model, input_name = create_full_model('ResNet50')
    model.summary()

    optimizer = OPTIMIZER_MAP['RMSprop'](0.001)
    loss = LOSS_MAP['TripletSemiHard']()
    model.compile(optimizer=optimizer, loss=loss)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='ckpts/ResNet50Base')
    train_spec = tf.estimator.TrainSpec(input_fn=get_train_input_fn(input_name), max_steps=30000)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_eval_input_fn(input_name))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def deprecated2_train_triplet():
    '''THIS DOES NOT WORK!!'''
    model, input_name = create_full_model('ResNet50')
    model.summary()

    optimizer = OPTIMIZER_MAP['RMSprop'](0.001)

    checkpoint_directory = "ckpts/ResNet50Base"

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

    loss = LOSS_MAP['TripletSemiHard']()
    model.compile(optimizer=optimizer, loss=loss)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='ckpts/ResNet50Triplet')
    train_spec = tf.estimator.TrainSpec(input_fn=get_train_input_fn(input_name), max_steps=500)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_eval_input_fn(input_name))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    status.assert_consumed()

def train_triplet():
    model, input_name = create_full_model('ResNet50')
    model.summary()

    optimizer = OPTIMIZER_MAP['Adam'](0.001)

    checkpoint_directory = "ckpts/ResNet50Base"

    latest = tf.train.latest_checkpoint(checkpoint_directory)
    model.load_weights(latest)

    loss = LOSS_MAP['TripletSemiHard']()
    model.compile(optimizer=optimizer, loss=loss)

    # import pdb; pdb.set_trace()
    test_dataset = get_test_dataset()
    for i in range(200):
        train_dataset = get_train_dataset(filter_size=20)
        history = model.fit(
            train_dataset,
            epochs=8
        )
    import pdb; pdb.set_trace()

    # Evaluate the network
    results = model.predict(test_dataset)




if __name__ == "__main__":
    # model, input_name = create_full_model('ResNet50')
    # model.summary()
    #
    # boundaries = [20000, 25000, 30000]
    # values = [0.001, 0.0001, 0.00001, 0.000001]
    # lr_schedule = LR_SCHEDULE_MAP['PiecewiseConstantDecay'](
    #     boundaries=boundaries, values=values)
    # optimizer = OPTIMIZER_MAP['RMSprop'](learning_rate=lr_schedule)
    #
    # loss = LOSS_MAP['CrossEntropy'](from_logits=True)
    #
    # accuracy = METRIC_MAP['Accuracy']()
    # model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    #
    #
    # estimator = tf.keras.estimator\
    #     .model_to_estimator(keras_model=model, model_dir='ckpts/ResNet50RMSprop')
    # train_spec = tf.estimator.TrainSpec(input_fn=get_train_input_fn(input_name),
    #                                     max_steps=35000)
    # eval_spec = tf.estimator.EvalSpec(input_fn=get_eval_input_fn(input_name))
    #
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    # estimator.evaluate(input_fn=get_eval_input_fn(input_name), steps=100)

    # estimator.predict(input_fn=get_eval_input_fn(input_name))

    train_triplet()
