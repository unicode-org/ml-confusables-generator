"""Script for building TensorFlow dataset."""
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import functools
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Dataset Metadata
TRAIN_DATA_DIR = 'charset_1k'
TEST_DATA_DIR = 'charset_1k_test'
LABEL_FILE = 'source/charset_1k.txt'
NUM_TRAIN = 47900
NUM_TEST = 100

# Image
HEIGHT = 20
WIDTH = 20
GRAYSCALE_IN = False
GRAYSCALE_OUT = True

# Train/test Specifications
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 1000 # higher: slower, better shuffling
PREFETCH_BUFFER_SIZE = 2

# Data Augmentation
RANDOM_ROTATE = True # See implementation
ROTATE_STDDEV = 0.35 # Stddev of normal distribution before rounding

RANDOM_ZOOM = True # See implementation
MAX_ZOOM_PERCENT = 4
ZOOM_STDDEV = 0.4

# Labels list
CLASS_NAMES = []
with open(LABEL_FILE) as f:
    for line in f:
        code_point = line.split('\n')[0]
        CLASS_NAMES.append(code_point)

def get_label(file_path):
    # Convert path to file name
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    # Derive label from file name
    class_name = tf.strings.split(file_name, '_')[0]
    label = tf.reduce_min(tf.where(tf.equal(CLASS_NAMES, class_name)))
    return label

def decode_img(img):
    # Convert compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img)
    # Convert data type to float between 0 and 1
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_path(file_path):
    # Get label and image Tensor
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def convert_format(img, label, grayscale_in, grayscale_out):
    # Convert between graycale and rgb
    if grayscale_in and not grayscale_out:
        # If input is grayscale and output is rgb
        img = tf.image.grayscale_to_rgb(img)  # use tensorflow function
    elif not grayscale_in and grayscale_out:
        # If input is rgb and output is grayscale
        # img = tf.reduce_mean(img, axis=2)
        img = tf.image.rgb_to_grayscale(img)
    return img,label

def augment(img, label):
    """Data augmentation. For the purpose of this project, most of the data
    should not be augmented!!

    Args:
        img: tf.Tensor, representing the image
        label: tf.Tensor, representing the corresponding label
    """
    # Randomly rotate by -3 and 3 degrees
    if RANDOM_ROTATE:
        # Follows normal distribution whose magnitude is more than 2 standard
        # deviations from the mean are dropped and re-picked
        degree = tf.random.truncated_normal(shape=[], stddev=ROTATE_STDDEV)
        # Rounds half to even. Also known as bankers rounding.
        degree = tf.math.round(degree)
        img = tfa.image.rotate(img, degree)

    # Randomly zoom in on image
    if RANDOM_ZOOM:
        # Generate 5 crop settings, ranging from a 0% to n% crop.
        scales = list(np.arange((100-MAX_ZOOM_PERCENT)/100, 1.0, 0.01))
        scales.reverse()
        boxes = np.zeros((len(scales), 4))
        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]
        crops = tf.image.crop_and_resize([img], boxes=boxes,
                                         box_indices=np.zeros(len(scales)),
                                         crop_size=(HEIGHT, WIDTH))
        # I am personally shamed of this implementation here
        # TODO: Change distribution here
        # TODO: Add fault proof
        idx = tf.random.truncated_normal(shape=[], stddev=ZOOM_STDDEV)
        idx = tf.math.abs(idx) # idx >= 0
        idx = tf.math.round(idx) # Bankers rounding again
        idx = tf.cast(idx, tf.dtypes.int32)
        img = crops[idx]

    return img, label

def train_input_fn():
    # Get filenames
    data_dir = pathlib.Path(TRAIN_DATA_DIR)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

    # Get labeled dataset
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # Format conversion
    ds = ds.map(functools.partial(convert_format, grayscale_in=GRAYSCALE_IN,
                                  grayscale_out=GRAYSCALE_OUT))
    # Data augmentation
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    # Prepare for tf.estimator
    ds = ds.map(lambda img, label: ({'dense_input': img}, label))

    # Shuffle, batch, repeat, prefetch
    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    ds = ds.batch(TRAIN_BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

    return ds

def test_input_fn():
    # Get filenames
    data_dir = pathlib.Path(TEST_DATA_DIR)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))

    # Get labeled dataset
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # Format conversion
    ds = ds.map(functools.partial(convert_format, grayscale_in=GRAYSCALE_IN,
                                  grayscale_out=GRAYSCALE_OUT))
    # Prepare for tf.estimator
    ds = ds.map(lambda img, label: ({'dense_input': img}, label))

    # Shuffle, batch, repeat, prefetch
    ds = ds.batch(TEST_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

    return ds

def val_input_fn():
    return test_input_fn()



if __name__ == "__main__":
    train_ds = train_input_fn()
    for features_batch, labels_batch in train_ds.take(1):
        print(features_batch)
        print(labels_batch)
    # data_dir = pathlib.Path('charset_1k')
    # list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

    # for file_path in list_ds.take(5):
    #     print(file_path)
    #     print(type(file_path))
        # file_name = tf.strings.split(file_path, os.path.sep)[-1]
        # label = tf.strings.split(file_name, '_')[0]
        # img = tf.io.read_file(file_path)
        # print(decode_img(img))

    # for i in range(10):
        # rand = tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)

    # print(tf.math.round(tf.random.truncated_normal(shape=[40], stddev=0.5)))

    # labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # labeled_ds = labeled_ds.map(functools.partial(convert_format,
    #                                               grayscale_in=GRAYSCALE_IN,
    #                                               grayscale_out=GRAYSCALE_OUT))
    # labeled_ds = labeled_ds.map(augment, num_parallel_calls=AUTOTUNE)
    # for entry in labeled_ds.take(5):
    #     print(entry)