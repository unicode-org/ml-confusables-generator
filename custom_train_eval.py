import tensorflow as tf
from custom_dataset import get_train_input_fn, get_eval_input_fn
from custom_model import create_full_model

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
LOSS_MAP = {
    'CrossEntropy':
        tf.keras.losses.CategoricalCrossentropy,
}


if __name__ == "__main__":
    model, input_name = create_full_model('MobileNetV2')
    model.summary()

    boundaries = [20000, 25000]
    values = [0.001, 0.0001, 0.00001]
    lr_schedule = LR_SCHEDULE_MAP['PiecewiseConstantDecay'](
        boundaries=boundaries, values=values)
    optimizer = OPTIMIZER_MAP['Adam'](learning_rate=lr_schedule)

    loss = LOSS_MAP['CrossEntropy'](from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    estimator = tf.keras.estimator\
        .model_to_estimator(keras_model=model, model_dir='ckpts/MobileNetV2')
    train_spec = tf.estimator.TrainSpec(input_fn=get_train_input_fn(input_name),
                                        max_steps=30000)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_eval_input_fn(input_name))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)